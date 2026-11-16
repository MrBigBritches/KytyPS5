#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/firstRunReport.h"
#include "common/gpuStall.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

namespace {

using Common::GpuStall::Clock;
using Common::GpuStall::StallTracker;
using std::chrono::milliseconds;

// The existing cases do not look at the cause, which Scope now requires, so they all name this one.
constexpr Common::GpuStall::HoldCause kAnyCause = Common::GpuStall::HoldCause::Other;
constexpr milliseconds kThreshold {250};
// How long a waiter thread may take to start and park before the test gives up.
constexpr std::chrono::seconds kParkLimit {10};
// Guards against a missed signal leaving the waiter parked until some later event. Real wakeups
// take a few milliseconds, and the bound is loose so a busy test machine does not fail it.
constexpr milliseconds kReleaseLimit {500};
// A threshold no test reaches, for checking that a short Scope never holds.
constexpr std::chrono::hours kNeverReached {1};

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "GpuStallTests: failed: %s\n", message);
    std::abort();
  }
}

void TestTrackerNothingActive() {
  StallTracker tracker;
  Check(!tracker.IsStalled(Clock::time_point {} + kNeverReached, kThreshold),
        "an empty tracker reported a stall");
}

void TestTrackerThreshold() {
  StallTracker tracker;
  const auto start = Clock::time_point {} + milliseconds(1000);
  tracker.Begin(start);
  Check(!tracker.IsStalled(start + kThreshold - milliseconds(1), kThreshold),
        "a stall held before reaching the threshold");
  Check(tracker.IsStalled(start + kThreshold, kThreshold), "a stall at the threshold did not hold");
  Check(tracker.IsStalled(start + kThreshold * 4, kThreshold), "a long stall did not hold");
  // A stall that ends exactly at the threshold releases at once.
  tracker.End(start);
  Check(!tracker.IsStalled(start + kThreshold, kThreshold), "an ended stall still held");
}

// The oldest stall decides, so a newer short one does not hold once the old one ends.
void TestTrackerOverlapping() {
  StallTracker tracker;
  const auto old_start = Clock::time_point {} + milliseconds(1000);
  const auto new_start = old_start + milliseconds(200);
  tracker.Begin(old_start);
  tracker.Begin(new_start);
  Check(tracker.IsStalled(old_start + kThreshold, kThreshold),
        "the oldest overlapping stall did not hold");
  tracker.End(old_start);
  Check(!tracker.IsStalled(old_start + kThreshold, kThreshold),
        "a newer stall under the threshold held");
  Check(tracker.IsStalled(new_start + kThreshold, kThreshold),
        "the remaining stall did not hold at its own threshold");
  tracker.End(new_start);
}

// Short compiles one after another restart the clock and never add up to a hold.
void TestTrackerBackToBack() {
  StallTracker tracker;
  auto start = Clock::time_point {} + milliseconds(1000);
  for (int compile = 0; compile < 10; ++compile) {
    tracker.Begin(start);
    const auto end = start + kThreshold - milliseconds(1);
    Check(!tracker.IsStalled(end, kThreshold), "back-to-back short stalls held");
    tracker.End(start);
    start = end;
  }
}

void TestDisabledNeverWaits() {
  Common::GpuStall::Configure(false, milliseconds(0));
  Common::GpuStall::Scope scope(kAnyCause);
  Check(!Common::GpuStall::IsStalled(), "a disabled module reported a stall");
  Check(!Common::GpuStall::WaitWhileStalled(), "a disabled module held");
}

void TestShortScopeNeverWaits() {
  Common::GpuStall::Configure(true, kNeverReached);
  Common::GpuStall::Scope scope(kAnyCause);
  bool held = true;
  std::thread waiter([&] { held = Common::GpuStall::WaitWhileStalled(); });
  waiter.join();
  Check(!held, "a Scope under the threshold held");
  Common::GpuStall::Configure(false);
}

// The GPU thread would deadlock if it waited for its own Scope to end.
void TestScopeOwnerNeverWaits() {
  Common::GpuStall::Configure(true, milliseconds(0));
  Common::GpuStall::Scope scope(kAnyCause);
  Check(Common::GpuStall::IsStalled(), "a Scope at the threshold was not reported");
  Check(!Common::GpuStall::WaitWhileStalled(), "the thread owning a Scope waited on itself");
  Common::GpuStall::Configure(false);
}

// Turning holding on while a compile is already running holds the next push once it is past
// the threshold, which is zero here.
void TestEnableWithActiveScope() {
  Common::GpuStall::Configure(false, milliseconds(0));
  Common::GpuStall::Scope scope(kAnyCause);
  Check(!Common::GpuStall::IsStalled(), "a disabled module reported a stall");
  Common::GpuStall::Configure(true, milliseconds(0));
  Check(Common::GpuStall::IsStalled(), "enabling with an old Scope did not stall");
  Common::GpuStall::Configure(false);
}

// Starts a waiter during a stall, lets release() end it, and checks it waited and woke promptly.
template <typename Release>
void CheckWaiterReleased(Release release, const char *what) {
  Common::GpuStall::Configure(true, milliseconds(0));
  auto scope = std::make_unique<Common::GpuStall::Scope>(kAnyCause);
  std::atomic<bool> done {false};
  bool held = false;
  Clock::time_point woke;
  std::thread waiter([&] {
    held = Common::GpuStall::WaitWhileStalled();
    woke = Clock::now();
    done = true;
  });
  // Waits until the waiter is parked, so the release below cannot race ahead of it.
  const auto park_deadline = Clock::now() + kParkLimit;
  while (Common::GpuStall::WaiterCount() == 0 && !done && Clock::now() < park_deadline)
    std::this_thread::yield();
  Check(!done && Common::GpuStall::WaiterCount() == 1, what);
  const auto released = Clock::now();
  release(scope);
  waiter.join();
  Check(held, "the waiter did not report that it held");
  Check(woke - released < kReleaseLimit, "the waiter was released too slowly");
  Common::GpuStall::Configure(false);
}

void TestScopeEndReleasesWaiter() {
  CheckWaiterReleased([](auto &scope) { scope.reset(); }, "a waiter did not hold during a stall");
}

void TestDisableReleasesWaiter() {
  CheckWaiterReleased([](auto &) { Common::GpuStall::Configure(false); },
                      "a waiter did not hold before the module was disabled");
}

// Only the call that waited reports resuming from a hold. The next call on the same thread, with
// no stall, clears it again.
void TestResumedFromHold() {
  Check(!Common::GpuStall::ResumedFromHold(), "a thread with no call that waited reported a hold");
  Common::GpuStall::Configure(true, milliseconds(0));
  auto scope = std::make_unique<Common::GpuStall::Scope>(kAnyCause);
  std::atomic<bool> held_once {false};
  bool resumed = false;
  bool cleared = false;
  std::thread waiter([&] {
    Common::GpuStall::WaitWhileStalled();
    resumed = Common::GpuStall::ResumedFromHold();
    held_once = true;
    Common::GpuStall::WaitWhileStalled(); // The Scope has ended, so this does not wait.
    cleared = !Common::GpuStall::ResumedFromHold();
  });
  const auto park_deadline = Clock::now() + kParkLimit;
  while (Common::GpuStall::WaiterCount() == 0 && !held_once && Clock::now() < park_deadline)
    std::this_thread::yield();
  Check(!held_once && Common::GpuStall::WaiterCount() == 1, "the waiter did not hold");
  scope.reset();
  waiter.join();
  Common::GpuStall::Configure(false);
  Check(resumed, "a call that waited did not report resuming from a hold");
  Check(cleared, "a call with no stall reported resuming from a hold");
}

// Starts the given number of waiter threads during a stall and waits until all are parked.
std::vector<std::thread> StartParkedWaiters(uint32_t count) {
  std::vector<std::thread> waiters;
  for (uint32_t waiter = 0; waiter < count; ++waiter)
    waiters.emplace_back([] { Common::GpuStall::WaitWhileStalled(); });
  const auto park_deadline = Clock::now() + kParkLimit;
  while (Common::GpuStall::WaiterCount() < count && Clock::now() < park_deadline)
    std::this_thread::yield();
  Check(Common::GpuStall::WaiterCount() == count, "waiters did not park during the stall");
  return waiters;
}

// Holds are counted once per stall, however many audio threads waited, and only when one did.
// Overlapping Scopes form a single stall.
void TestHoldCounters() {
  using Common::DebugCounters::Counter;
  const auto holds = [] { return Common::DebugCounters::Get(Counter::AudioHolds); };
  Common::DebugCounters::SetEnabled(true);
  Common::GpuStall::Configure(true, milliseconds(0));
  const auto before = holds();
  const auto held_ns_before = Common::DebugCounters::Get(Counter::AudioHoldNs);
  {
    auto scope = std::make_unique<Common::GpuStall::Scope>(kAnyCause);
    auto waiters = StartParkedWaiters(2);
    scope.reset();
    for (auto &waiter : waiters) waiter.join();
  }
  Check(holds() == before + 1, "one stall holding two threads was not counted once");
  Check(Common::DebugCounters::Get(Counter::AudioHoldNs) > held_ns_before,
        "a counted hold added no held time");
  { Common::GpuStall::Scope scope(kAnyCause); }
  Check(holds() == before + 1, "a stall that held nothing was counted");
  {
    auto outer = std::make_unique<Common::GpuStall::Scope>(kAnyCause);
    auto inner = std::make_unique<Common::GpuStall::Scope>(kAnyCause);
    auto waiters = StartParkedWaiters(1);
    inner.reset();
    Check(holds() == before + 1, "a stall was counted before its last Scope ended");
    outer.reset();
    for (auto &waiter : waiters) waiter.join();
  }
  Check(holds() == before + 2, "overlapping Scopes were not counted as one stall");
  Common::GpuStall::Configure(false);
  Common::DebugCounters::SetEnabled(false);
}

using Common::GpuStall::Breakdown;
using Common::GpuStall::HoldBreakdown;
using Common::GpuStall::HoldCause;
using Common::GpuStall::HoldHistogram;

// How long a hold lasts in the cases below, so the held time is well above timer noise.
constexpr milliseconds kHoldTime {20};

// The position of a cause in the breakdown's tables.
size_t Index(HoldCause cause) { return static_cast<size_t>(cause); }

// Starts Scopes of the given causes in order, parks one audio waiter, and ends the Scopes in the order
// given by end_order, which holds the indexes into causes. The one hold that results is counted when
// the last Scope ends. With reclassify set, the first Scope's cause is changed while it is held.
void RunHold(const std::vector<HoldCause> &causes, const std::vector<size_t> &end_order,
             HoldCause reclassify_first_to = HoldCause::Other, bool reclassify = false) {
  Common::GpuStall::Configure(true, milliseconds(0));
  std::vector<std::unique_ptr<Common::GpuStall::Scope>> scopes;
  for (const auto cause : causes) scopes.push_back(std::make_unique<Common::GpuStall::Scope>(cause));
  auto waiters = StartParkedWaiters(1);
  std::this_thread::sleep_for(kHoldTime);
  if (reclassify) scopes[0]->Reclassify(reclassify_first_to);
  for (const auto index : end_order) scopes[index].reset();
  for (auto &waiter : waiters) waiter.join();
  Common::GpuStall::Configure(false);
}

// The held nanoseconds of every cause added together.
uint64_t SumNs(const HoldBreakdown &breakdown) {
  uint64_t total = 0;
  for (const auto nanoseconds : breakdown.cause_ns) total += nanoseconds;
  return total;
}

// When Scopes overlap, the whole hold goes to the cause of the oldest, whichever ends last. The two
// causes differ so a charge to the wrong Scope shows. The baseline is the same overlap with the old
// Scope ending last, where the oldest and the last to end are the same Scope, so only the first run
// tells "oldest" from "last to end".
void TestOverlapChargesTheOldestScope() {
  const auto older = HoldCause::RequiredGraphics;
  const auto newer = HoldCause::WarmWaited;
  const auto held_before = Common::GpuStall::HeldNs();
  const auto before = Breakdown();
  RunHold({older, newer}, {0, 1}); // The old Scope ends first, and the new one ends the stall.
  const auto after = Breakdown();
  const auto held = Common::GpuStall::HeldNs() - held_before;
  Check(held > 0, "the overlapping stall held nothing");
  Check(after.cause_ns[Index(older)] - before.cause_ns[Index(older)] == held,
        "the hold was not charged in full to the oldest Scope");
  Check(after.cause_ns[Index(newer)] == before.cause_ns[Index(newer)],
        "the Scope that ended the stall was charged");
  Check(after.cause_count[Index(older)] - before.cause_count[Index(older)] == 1,
        "the oldest Scope's cause was not counted once");

  const auto mid = Breakdown();
  RunHold({older, newer}, {1, 0}); // The old Scope outlasts the new one.
  const auto last = Breakdown();
  Check(last.cause_count[Index(older)] - mid.cause_count[Index(older)] == 1 &&
            last.cause_count[Index(newer)] == mid.cause_count[Index(newer)],
        "an old Scope that ended last was not charged");
}

// A site that learns its cause late changes it before the Scope ends, and the hold follows.
void TestReclassifyBeforeTheScopeEnds() {
  const auto before = Breakdown();
  RunHold({HoldCause::WarmInline}, {0}, HoldCause::GeneratedWaited, true);
  const auto after = Breakdown();
  Check(after.cause_count[Index(HoldCause::GeneratedWaited)] -
                before.cause_count[Index(HoldCause::GeneratedWaited)] == 1,
        "a reclassified Scope was not charged to its new cause");
  Check(after.cause_count[Index(HoldCause::WarmInline)] ==
            before.cause_count[Index(HoldCause::WarmInline)],
        "a reclassified Scope was still charged to its first cause");
}

// The causes add up to the total that the tracker counts on its own. Three holds of three causes, so
// a cause missing from the table is a hold that is in the total and in no cause. The total is read
// from HeldNs, which is a counter of its own and not computed from the causes. The same three holds
// go through the global histogram and the stored records, which must show them too. A histogram that
// puts every hold in bucket 0 or never keeps the maximum, and a record that is not stored, all differ
// from the three known holds, which are each at least kHoldTime long.
void TestHeldCausesSumToTotal() {
  const auto held_before  = Common::GpuStall::HeldNs();
  const auto holds_before = Common::GpuStall::HoldCount();
  const auto before       = Breakdown();
  RunHold({HoldCause::RequiredCompute}, {0});
  std::this_thread::sleep_for(milliseconds(5));
  RunHold({HoldCause::PrescanWaited}, {0});
  RunHold({HoldCause::IneligibleDraw}, {0});
  const auto after = Breakdown();
  for (const auto cause :
       {HoldCause::RequiredCompute, HoldCause::PrescanWaited, HoldCause::IneligibleDraw}) {
    Check(after.cause_ns[Index(cause)] > before.cause_ns[Index(cause)],
          "a hold was not charged to its cause");
    Check(after.cause_count[Index(cause)] - before.cause_count[Index(cause)] == 1,
          "a cause was not counted once");
  }
  Check(Common::GpuStall::HoldCount() - holds_before == 3, "the tracker did not count three holds");
  Check(SumNs(after) - SumNs(before) == Common::GpuStall::HeldNs() - held_before,
        "the causes do not add up to the tracker's held time");
  const auto total = Common::GpuStall::HeldNs();
  const auto difference = SumNs(after) > total ? SumNs(after) - total : total - SumNs(after);
  Check(difference < 1'000'000, "the causes since the start differ from the total by 1 ms or more");
  Check(after.histogram.count - before.histogram.count == 3, "the histogram did not count three holds");

  // The three records are the last three stored, in the order the holds ended.
  const auto records = Common::GpuStall::HoldRecords();
  Check(records.size() >= 3 && after.exact.overflow == 0, "the holds were not stored");
  const std::vector<HoldCause> expected_causes {HoldCause::RequiredCompute, HoldCause::PrescanWaited,
                                                HoldCause::IneligibleDraw};
  HoldHistogram expected;
  uint64_t      longest = 0;
  for (size_t position = 0; position < 3; ++position) {
    const auto &record = records[records.size() - 3 + position];
    Check(record.cause == expected_causes[position], "a stored hold has the wrong cause");
    Check(record.held_ns >= static_cast<uint64_t>(kHoldTime.count()) * 1'000'000,
          "a stored hold is shorter than it was held");
    expected.Add(record.held_ns);
    longest = std::max(longest, record.held_ns);
  }
  Check(after.exact.count - before.exact.count == 3, "the exact figures did not count three holds");
  Check(after.histogram.max_ns >= longest, "the histogram did not keep the longest hold");
  // max_ms is nanoseconds divided by a million, so compare in milliseconds with a margin far below
  // a nanosecond. Scaling it back up can round to just under the whole nanoseconds it came from.
  constexpr double kMsMargin = 1e-9;
  Check(after.exact.max_ms + kMsMargin >= static_cast<double>(longest) / 1e6,
        "the exact max is below a hold");
  Check(after.histogram.buckets[0] == before.histogram.buckets[0],
        "a hold of at least kHoldTime landed in the shortest bucket");
  for (size_t bucket = 0; bucket < Common::GpuStall::HOLD_BUCKET_COUNT; ++bucket) {
    const auto grown = after.histogram.buckets[bucket] - before.histogram.buckets[bucket];
    Check(grown == expected.buckets[bucket], "a hold landed in the wrong histogram bucket");
  }
}

// The steady clock in nanoseconds since its epoch, the clock LastThrottleSignalNs is kept on.
int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

// A hold of guest audio that starts is a throttle signal, read while the waiter is still parked, so
// the signal comes from the hold's start and not its end. The baseline parks a waiter under the same
// Scope with a threshold no test reaches, so nothing holds and the signal stays where it was. A hold
// is not an audio low-water signal, so it leaves that count alone.
void TestHoldStartSignalsThrottle() {
  const auto before_baseline = Common::GpuStall::LastThrottleSignalNs();
  Common::GpuStall::Configure(true, kNeverReached);
  {
    Common::GpuStall::Scope scope(kAnyCause);
    std::thread waiter([] { Common::GpuStall::WaitWhileStalled(); });
    waiter.join();
  }
  Common::GpuStall::Configure(false);
  Check(Common::GpuStall::LastThrottleSignalNs() == before_baseline, "a Scope that never held signalled the throttle");

  // The steady clock ticks far finer than this, so a hold that starts after it is later than any
  // signal before.
  std::this_thread::sleep_for(milliseconds(1));
  const auto low_water_before = Common::GpuStall::AudioLowWaterCount();
  Common::GpuStall::Configure(true, milliseconds(0));
  auto scope = std::make_unique<Common::GpuStall::Scope>(kAnyCause);
  const auto before_hold = SteadyNowNs();
  auto waiters = StartParkedWaiters(1);
  const auto during_hold = Common::GpuStall::LastThrottleSignalNs();
  const auto parked = SteadyNowNs();
  scope.reset();
  for (auto &waiter : waiters) waiter.join();
  Common::GpuStall::Configure(false);
  Check(during_hold >= before_hold && during_hold <= parked, "a hold that started did not signal the throttle");
  Check(Common::GpuStall::AudioLowWaterCount() == low_water_before, "a hold counted as an audio low-water signal");
}

// An audio low-water signal counts one and moves the throttle time to now. The hold above is the
// baseline that moves the throttle time without counting, so the count comes from this call alone.
void TestAudioLowWaterSignals() {
  std::this_thread::sleep_for(milliseconds(1));
  const auto count_before = Common::GpuStall::AudioLowWaterCount();
  const auto before       = SteadyNowNs();
  Common::GpuStall::NoteAudioLowWater();
  const auto after  = SteadyNowNs();
  const auto signal = Common::GpuStall::LastThrottleSignalNs();
  Check(Common::GpuStall::AudioLowWaterCount() == count_before + 1, "a low-water signal was not counted once");
  Check(signal >= before && signal <= after, "a low-water signal did not move the throttle time to now");
}

// An underrun counts toward UnderrunsAfterLowWater only when a low-water signal came before it on its
// port. The baseline is the same call for an underrun with no signal before it, which leaves the count
// alone, so a count that ignores the flag fails the first check and one that never counts the second.
void TestUnderrunsAfterLowWater() {
  const auto before = Common::GpuStall::UnderrunsAfterLowWater();
  Common::GpuStall::NoteUnderrun(false);
  Check(Common::GpuStall::UnderrunsAfterLowWater() == before, "an underrun with no low-water signal was counted");
  Common::GpuStall::NoteUnderrun(true);
  Check(Common::GpuStall::UnderrunsAfterLowWater() == before + 1, "an underrun after a low-water signal was not counted");
}

// The hook starts off, so a run that never turns it on keeps the audio submit as it was. Run first,
// before any case sets the gate. The baseline is the same call with the gate turned on, which signals,
// so a hook that starts on fails the first check and the call is shown able to signal.
void TestAudioLowWaterHookStartsOff() {
  constexpr uint32_t kPeriod = 4096;
  const bool off_at_start = !Common::GpuStall::NoteAudioQueueFill(true, static_cast<int>(kPeriod), kPeriod, 4 * kPeriod);
  Common::GpuStall::SetAudioLowWaterHookEnabled(true);
  const bool signals_on = Common::GpuStall::NoteAudioQueueFill(true, static_cast<int>(kPeriod), kPeriod, 4 * kPeriod);
  Common::GpuStall::SetAudioLowWaterHookEnabled(false);
  Check(off_at_start, "the audio low-water hook was on before anything turned it on");
  Check(signals_on, "the same call did not signal with the hook turned on");
}

// With the gate on, a fill under the mark signals only from a primed queue and not on the first submit
// after a hold, which drained the queue on purpose. Every call passes the same fill of one period
// against a floor of four. A waiter thread is held once and then makes the resumed call, and the same
// thread makes the baseline call after a second wait that did not hold, so the calls differ only in
// whether the queue is primed or the thread is resuming. A negative fill, SDL's error value, is passed
// from a primed queue on the test thread, which is not resuming, so it differs from the baseline only
// in the fill. A hook that drops any of the three tests fails.
void TestAudioLowWaterConditions() {
  constexpr uint32_t kPeriod   = 4096;
  constexpr uint32_t kFloor    = 4 * kPeriod;
  constexpr int      kQueued   = static_cast<int>(kPeriod);
  constexpr int      kSdlError = -1;
  Common::GpuStall::SetAudioLowWaterHookEnabled(true);
  const bool unprimed   = Common::GpuStall::NoteAudioQueueFill(false, kQueued, kPeriod, kFloor);
  const bool error_fill = Common::GpuStall::NoteAudioQueueFill(true, kSdlError, kPeriod, kFloor);
  Common::GpuStall::Configure(true, milliseconds(0));
  auto scope = std::make_unique<Common::GpuStall::Scope>(kAnyCause);
  std::atomic<bool> resumed_call {true};
  std::atomic<bool> baseline_call {false};
  std::thread waiter([&] {
    Common::GpuStall::WaitWhileStalled();
    resumed_call = Common::GpuStall::NoteAudioQueueFill(true, kQueued, kPeriod, kFloor);
    Common::GpuStall::WaitWhileStalled(); // The Scope has ended, so this does not wait.
    baseline_call = Common::GpuStall::NoteAudioQueueFill(true, kQueued, kPeriod, kFloor);
  });
  const auto park_deadline = Clock::now() + kParkLimit;
  while (Common::GpuStall::WaiterCount() == 0 && Clock::now() < park_deadline)
    std::this_thread::yield();
  const bool parked = Common::GpuStall::WaiterCount() == 1;
  scope.reset();
  waiter.join();
  Common::GpuStall::Configure(false);
  Common::GpuStall::SetAudioLowWaterHookEnabled(false);
  Check(parked, "the waiter did not hold");
  Check(baseline_call, "a primed fill under the mark did not signal on a thread that was not resuming");
  Check(!unprimed, "a queue that was not primed signalled");
  Check(!error_fill, "an SDL error fill signalled");
  Check(!resumed_call, "the first submit after a hold signalled");
}

// Runs one hold with `jobs` graphics prescan jobs counted when it begins. The jobs end while the
// waiter is still parked, so the count at the hold's end is always 0.
void RunHoldBegunWithGraphicsJobs(size_t jobs) {
  std::vector<std::unique_ptr<Common::FirstRunReport::PrescanGraphicsRunningScope>> running;
  for (size_t job = 0; job < jobs; ++job)
    running.push_back(std::make_unique<Common::FirstRunReport::PrescanGraphicsRunningScope>());
  Common::GpuStall::Configure(true, milliseconds(0));
  auto scope = std::make_unique<Common::GpuStall::Scope>(kAnyCause);
  auto waiters = StartParkedWaiters(1);
  running.clear();
  scope.reset();
  for (auto &waiter : waiters) waiter.join();
  Common::GpuStall::Configure(false);
}

// A stored hold carries the graphics prescan jobs running when it began, saturated at a byte. The
// first hold begins with 300 jobs and the second with none, the baseline, and both sets of jobs are
// gone by the hold's end. A count never stored, read at the hold's end, not read back, or not
// saturated differs from {255, 0}.
void TestHoldRecordsGraphicsRunning() {
  constexpr size_t kManyJobs = 300;
  Common::FirstRunReport::Configure(true, false);
  RunHoldBegunWithGraphicsJobs(kManyJobs);
  RunHoldBegunWithGraphicsJobs(0);
  Common::FirstRunReport::Configure(false, false);
  const auto records = Common::GpuStall::HoldRecords();
  Check(records.size() >= 2, "the two holds were not stored");
  Check(records[records.size() - 2].prescan_graphics_running == 255 &&
            records[records.size() - 1].prescan_graphics_running == 0,
        "a stored hold did not carry the graphics prescan jobs running when it began");
}

// The role marks the waited-prebuild boost reads. An unmarked thread is the baseline. A deferred build
// scope gives it the role only while the scope lives, so a scope that never sets the role or never
// gives it back fails there. A thread marked as the GPU thread keeps that role inside and after the
// same scope, as when it runs a deferred build inline, so a scope that always sets the role fails
// there. A thread started inside the scope still reads None, so a mark shared between threads fails.
// Both marked threads end before the next test, so the main thread stays unmarked.
void TestThreadRoleMarks() {
  using Common::GpuStall::CurrentThreadRole;
  using Common::GpuStall::ThreadRole;
  using Common::GpuStall::ThreadRoleScope;
  ThreadRole before = ThreadRole::Gpu;
  ThreadRole inside = ThreadRole::None;
  ThreadRole after  = ThreadRole::Gpu;
  std::thread unmarked([&] {
    before = CurrentThreadRole();
    {
      const ThreadRoleScope role(ThreadRole::DeferredBuild);
      inside = CurrentThreadRole();
    }
    after = CurrentThreadRole();
  });
  unmarked.join();
  Check(before == ThreadRole::None && inside == ThreadRole::DeferredBuild && after == ThreadRole::None,
        "a deferred build scope did not mark an unmarked thread only while it lived");
  ThreadRole gpu_inside = ThreadRole::None;
  ThreadRole gpu_after  = ThreadRole::None;
  ThreadRole other      = ThreadRole::Gpu;
  std::thread gpu([&] {
    Common::GpuStall::SetThreadRole(ThreadRole::Gpu);
    {
      const ThreadRoleScope role(ThreadRole::DeferredBuild);
      gpu_inside = CurrentThreadRole();
      std::thread([&] { other = CurrentThreadRole(); }).join();
    }
    gpu_after = CurrentThreadRole();
  });
  gpu.join();
  Check(gpu_inside == ThreadRole::Gpu && gpu_after == ThreadRole::Gpu,
        "a deferred build scope replaced or cleared the GPU thread's role");
  Check(other == ThreadRole::None, "another thread read the GPU thread's role");
  Check(!Common::GpuStall::BlocksDraws(ThreadRole::None) && Common::GpuStall::BlocksDraws(ThreadRole::Gpu) &&
            Common::GpuStall::BlocksDraws(ThreadRole::DeferredBuild),
        "only the GPU thread and a deferred build worker should block draws");
}

// The exact percentiles are nearest rank over the sorted holds. Ten holds of 10 to 100 ms give p50 of
// 50 and p95 of 100, where a rank that rounds down gives 90 for p95. Nineteen holds of 300 ms and one
// of 600 ms give an exact p95 of 300, which the buckets, with a bound of 500, could not show.
void TestExactPercentiles() {
  constexpr uint64_t kNsPerMs = 1'000'000;
  Check(Common::GpuStall::ExactPercentileMs({}, 0.5) == 0.0, "an empty list reported a hold");
  std::vector<uint64_t> ladder;
  for (uint64_t step = 1; step <= 10; ++step) ladder.push_back(step * 10 * kNsPerMs);
  Check(Common::GpuStall::ExactPercentileMs(ladder, 0.5) == 50.0,
        "the exact p50 is not the fifth shortest");
  Check(Common::GpuStall::ExactPercentileMs(ladder, 0.95) == 100.0, "the exact p95 is not the tenth");
  Check(Common::GpuStall::ExactPercentileMs(ladder, 0.0) == 10.0,
        "a rank below the first was not clamped");
  std::vector<uint64_t> mostly(19, 300 * kNsPerMs);
  mostly.push_back(600 * kNsPerMs);
  Check(Common::GpuStall::ExactPercentileMs(mostly, 0.95) == 300.0, "the exact p95 followed the bucket");
}

// Nine holds of 30 ms and one of 700 ms, so p50 lands in the 30 ms bucket and p95 in the 700 ms one.
// A bucket that is one off gives p50 100 instead of 50, and the boundary checks show which side an
// exact bound falls on. The percentile is the bucket's upper bound, never above the longest hold.
void TestHoldHistogramPercentiles() {
  constexpr uint64_t kNsPerMs = 1'000'000;
  HoldHistogram histogram;
  Check(histogram.PercentileMs(0.5) == 0.0 && histogram.MaxMs() == 0.0, "an empty histogram reported a hold");
  for (int hold = 0; hold < 9; ++hold) histogram.Add(30 * kNsPerMs);
  histogram.Add(700 * kNsPerMs);
  Check(histogram.count == 10, "the histogram lost a hold");
  Check(histogram.PercentileMs(0.5) == 50.0, "p50 is not the upper bound of the 30 ms bucket");
  Check(histogram.PercentileMs(0.95) == 700.0, "p95 is not the longest hold, clamped from its bucket bound");
  Check(histogram.MaxMs() == 700.0, "the max is wrong");

  // An exact bound belongs to the bucket above it, and the last bucket has no bound.
  Check(HoldHistogram::BucketOf(0) == 0, "0 ns is not in the first bucket");
  Check(HoldHistogram::BucketOf(50 * kNsPerMs - 1) == 2, "just under a bound left its bucket");
  Check(HoldHistogram::BucketOf(50 * kNsPerMs) == 3, "an exact bound stayed in the lower bucket");
  Check(HoldHistogram::BucketOf(30000 * kNsPerMs) == Common::GpuStall::HOLD_BUCKET_COUNT - 1,
        "a hold past the last bound is not in the last bucket");
  HoldHistogram slow;
  slow.Add(40000 * kNsPerMs);
  Check(slow.PercentileMs(0.5) == 40000.0, "a hold in the last bucket did not report its own length");
}

// The audio low-water hook signals the throttle only with its gate on. Both runs pass the same primed
// fill of one period against a floor of four periods, which is under the two-period mark, from a
// thread that is not resuming from a hold, so they differ only in the gate. A hook that ignored the
// gate would signal in the first run, and one that never signals fails the second. The signal count
// and the throttle time are read around each call, and the gate is left off. The throttle time must
// move forward across the gate-on call, after a short sleep, so an earlier signal such as a hold
// start cannot stand in for it.
void TestAudioLowWaterHookGate() {
  constexpr uint32_t kPeriod = 4096;
  constexpr uint32_t kFloor  = 4 * kPeriod;
  constexpr int      kQueued = static_cast<int>(kPeriod);
  Check(!Common::GpuStall::ResumedFromHold(), "the test thread is resuming from a hold, which skips the hook");

  Common::GpuStall::SetAudioLowWaterHookEnabled(false);
  const auto count_off  = Common::GpuStall::AudioLowWaterCount();
  const auto signal_off = Common::GpuStall::LastThrottleSignalNs();
  Check(!Common::GpuStall::NoteAudioQueueFill(true, kQueued, kPeriod, kFloor),
        "the hook reported a low-water signal with the gate off");
  Check(Common::GpuStall::AudioLowWaterCount() == count_off &&
            Common::GpuStall::LastThrottleSignalNs() == signal_off,
        "a low-water signal was recorded with the gate off");

  // The steady clock ticks far finer than this, so a signal made now is later than any before it.
  std::this_thread::sleep_for(milliseconds(1));
  Common::GpuStall::SetAudioLowWaterHookEnabled(true);
  const auto count_on  = Common::GpuStall::AudioLowWaterCount();
  const auto signal_on = Common::GpuStall::LastThrottleSignalNs();
  const bool signalled = Common::GpuStall::NoteAudioQueueFill(true, kQueued, kPeriod, kFloor);
  Common::GpuStall::SetAudioLowWaterHookEnabled(false);
  Check(signalled, "the hook did not report a low-water signal with the gate on");
  Check(Common::GpuStall::AudioLowWaterCount() == count_on + 1 &&
            Common::GpuStall::LastThrottleSignalNs() > signal_on,
        "the low-water signal was not recorded with the gate on");
}

// With the gate on, the mark is the smaller of two periods and the floor less half a period. A floor
// of two periods gives a mark of 1.5 periods, and a floor of eight periods a mark of two. A fill at
// the mark signals nothing and one byte under it signals, so a mark off by a byte either way fails,
// and a mark of two periods for every floor fails the two-period floor, whose 1.5-period fill would
// then signal.
void TestAudioLowWaterMarkBoundary() {
  constexpr uint32_t kPeriod = 4096;
  struct MarkCase {
    uint32_t floor;
    uint32_t mark;
  };
  for (const auto &[floor, mark] : {MarkCase{2 * kPeriod, 3 * kPeriod / 2}, MarkCase{8 * kPeriod, 2 * kPeriod}}) {
    Common::GpuStall::SetAudioLowWaterHookEnabled(true);
    const bool at_mark    = Common::GpuStall::NoteAudioQueueFill(true, static_cast<int>(mark), kPeriod, floor);
    const bool under_mark = Common::GpuStall::NoteAudioQueueFill(true, static_cast<int>(mark) - 1, kPeriod, floor);
    Common::GpuStall::SetAudioLowWaterHookEnabled(false);
    Check(!at_mark, "a fill at the low-water mark signalled");
    Check(under_mark, "a fill one byte under the low-water mark did not signal");
  }
}

// Config::Load turns the low-water hook on only when both the compute and the graphics prescan are
// asked for. Each case loads default options with one combination and then makes the same primed
// call, which signals whenever the hook is on, so only both may signal. Loading the defaults again
// turns the hook off, and Shutdown lets go of the config. Run last, since a load also sets the stall
// hold and the counters.
void TestConfigLoadGatesTheHook() {
  constexpr uint32_t kPeriod = 4096;
  struct GateCase {
    bool compute;
    bool graphics;
    bool signals;
  };
  Config::Initialize();
  for (const auto &[compute, graphics, signals] :
       {GateCase{true, true, true}, GateCase{true, false, false}, GateCase{false, true, false},
        GateCase{false, false, false}}) {
    Config::ConfigOptions options;
    options.prescan_compute_builds     = compute;
    options.prescan_graphics_libraries = graphics;
    Config::Load(options);
    Check(Common::GpuStall::NoteAudioQueueFill(true, static_cast<int>(kPeriod), kPeriod, 4 * kPeriod) == signals,
          "the hook was on for a prescan combination other than both");
  }
  Config::Load(Config::ConfigOptions {});
  Check(!Common::GpuStall::NoteAudioQueueFill(true, static_cast<int>(kPeriod), kPeriod, 4 * kPeriod),
        "the default options left the hook on");
  Config::Shutdown();
}

} // namespace

int main() {
  TestAudioLowWaterHookStartsOff();
  TestTrackerNothingActive();
  TestAudioLowWaterHookGate();
  TestAudioLowWaterMarkBoundary();
  TestTrackerThreshold();
  TestTrackerOverlapping();
  TestTrackerBackToBack();
  TestDisabledNeverWaits();
  TestShortScopeNeverWaits();
  TestScopeOwnerNeverWaits();
  TestEnableWithActiveScope();
  TestScopeEndReleasesWaiter();
  TestDisableReleasesWaiter();
  TestResumedFromHold();
  TestHoldCounters();
  TestOverlapChargesTheOldestScope();
  TestReclassifyBeforeTheScopeEnds();
  TestHeldCausesSumToTotal();
  TestHoldStartSignalsThrottle();
  TestAudioLowWaterSignals();
  TestUnderrunsAfterLowWater();
  TestAudioLowWaterConditions();
  TestHoldRecordsGraphicsRunning();
  TestThreadRoleMarks();
  TestHoldHistogramPercentiles();
  TestExactPercentiles();
  TestConfigLoadGatesTheHook();
  std::puts("GpuStallTests: all cases passed");
  return 0;
}
