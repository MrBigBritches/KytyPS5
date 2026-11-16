#include "common/gpuStall.h"

#include "common/debugCounters.h"
#include "common/logging/log.h"
#include "common/threads.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <optional>
#include <utility>

namespace Common::GpuStall {

namespace {

struct State {
	Mutex           mutex;
	CondVar         released;
	StallTracker    tracker;
	Clock::duration threshold = GPU_STALL_AUDIO_HOLD_THRESHOLD;
	bool            enabled   = false;
	uint32_t        waiters   = 0;
	// When the first submit of the current stall was held, so a stall that held several audio
	// threads is counted once.
	std::optional<Clock::time_point> first_hold;
	// The Scope that was oldest when the current stall first held audio, and its cause. The cause is
	// copied when that Scope ends, because a site may reclassify it until then. Id 0 means no hold.
	uint64_t  hold_scope_id = 0;
	HoldCause hold_cause    = HoldCause::Other;
};

// Built on first use, so a Scope or audio push during static initialization finds it ready. It is
// never destroyed, so a thread still waiting at process exit cannot touch a dead mutex.
State& GetState() {
	static State& state = *new State;
	return state;
}

// Lock-free copies of the enabled flag and the Scope count, so an audio push with no compile in
// progress returns without touching the mutex.
std::atomic<bool>     g_enabled {false};
std::atomic<uint32_t> g_active_scopes {0};
std::atomic<uint64_t> g_hold_count {0};
// A lock-free copy of whether first_hold is set, so a worker can tell whether audio is held now
// without taking the mutex. It is written under the mutex wherever first_hold changes.
std::atomic<bool>     g_hold_in_progress {false};
std::atomic<uint64_t> g_held_ns {0};
std::atomic<uint64_t> g_scope_count {0};
// Scope ids start at 1, since 0 means a stall begun without one.
std::atomic<uint64_t> g_next_scope_id {1};

// The hold breakdown, kept as atomics so recording a hold takes no lock and allocates nothing.
std::array<std::atomic<uint64_t>, HOLD_CAUSE_COUNT>  g_cause_ns {};
std::array<std::atomic<uint64_t>, HOLD_CAUSE_COUNT>  g_cause_count {};
std::array<std::atomic<uint64_t>, HOLD_BUCKET_COUNT> g_hold_buckets {};
std::atomic<uint64_t>                                g_hold_max_ns {0};

// The stored holds, written without a lock like the report's underrun slots. A slot's held time is
// stored as one more than its value and last, with release, so 0 means the slot is not published.
std::array<std::atomic<uint64_t>, HOLD_RECORD_CAPACITY> g_record_held_plus_one {};
std::array<std::atomic<int64_t>, HOLD_RECORD_CAPACITY>  g_record_end_ns {};
std::array<std::atomic<uint8_t>, HOLD_RECORD_CAPACITY>  g_record_cause {};
std::atomic<size_t>                                     g_record_next {0};
std::atomic<uint64_t>                                   g_record_overflow {0};
// The last throttle signal on the steady clock in nanoseconds, and the audio low-water signals.
std::atomic<int64_t>                                    g_last_throttle_signal_ns {0};
std::atomic<uint64_t>                                   g_audio_low_water {0};

// Nanoseconds since the steady clock's epoch, the report's NowNs clock, for the throttle signals.
int64_t SteadyNowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
}

constexpr double NS_PER_MS = 1e6;

constexpr std::array<const char*, HOLD_CAUSE_COUNT> CAUSE_NAMES {
    "required_graphics",          "required_compute",    "warm_inline",
    "warm_inline_paused",         "generated_inline",    "generated_inline_paused",
    "warm_waited",
    "generated_waited",           "prescan_waited",      "prescan_in_flight",
    "prescan_never_posted_async", "prescan_never_posted_indirect",
    "prescan_never_posted_deep",  "prescan_refused",     "deferred_draw_resolve",
    "ineligible_draw",            "library_once",        "prescan_queued",
    "other"};

// A name missing from the table would be a null in the report, so the table must be full.
constexpr bool AllCausesNamed() {
	for (const char* name: CAUSE_NAMES) {
		if (name == nullptr) {
			return false;
		}
	}
	return true;
}
static_assert(AllCausesNamed(), "every HoldCause needs a name");

// Keeps one finished hold, without allocating. The slot is reserved with one relaxed increment, and
// a hold past the capacity only adds to the overflow count, so the index cannot wrap.
void StoreHoldRecord(HoldCause cause, uint64_t held_ns, int64_t end_ns) {
	if (g_record_next.load(std::memory_order_relaxed) >= HOLD_RECORD_CAPACITY) {
		g_record_overflow.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const size_t slot = g_record_next.fetch_add(1, std::memory_order_relaxed);
	if (slot >= HOLD_RECORD_CAPACITY) {
		// Several calls passed the check together and took the last slots.
		g_record_overflow.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	g_record_end_ns[slot].store(end_ns, std::memory_order_relaxed);
	g_record_cause[slot].store(static_cast<uint8_t>(cause), std::memory_order_relaxed);
	g_record_held_plus_one[slot].store(held_ns + 1, std::memory_order_release);
}

// Adds one finished hold to the cause table, the histogram and the records. The total is added by
// the caller from the same value, so it is counted apart from the causes.
void RecordHold(HoldCause cause, uint64_t held_ns, int64_t end_ns) {
	const auto index = static_cast<size_t>(cause);
	g_cause_ns[index].fetch_add(held_ns, std::memory_order_relaxed);
	g_cause_count[index].fetch_add(1, std::memory_order_relaxed);
	g_hold_buckets[HoldHistogram::BucketOf(held_ns)].fetch_add(1, std::memory_order_relaxed);
	auto seen_max = g_hold_max_ns.load(std::memory_order_relaxed);
	while (held_ns > seen_max &&
	       !g_hold_max_ns.compare_exchange_weak(seen_max, held_ns, std::memory_order_relaxed)) {
	}
	StoreHoldRecord(cause, held_ns, end_ns);
}

// A thread that owns a Scope would wait on its own release, so it never waits.
thread_local uint32_t t_owned_scopes = 0;

// Whether this thread's most recent WaitWhileStalled call waited.
thread_local bool t_resumed_from_hold = false;

int64_t ElapsedMs(Clock::time_point since) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - since).count();
}

} // namespace

const char* HoldCauseName(HoldCause cause) {
	return CAUSE_NAMES[static_cast<size_t>(cause)];
}

size_t HoldHistogram::BucketOf(uint64_t held_ns) {
	constexpr uint64_t NS_PER_MS_INT = 1'000'000;
	for (size_t bucket = 0; bucket < HOLD_BUCKET_BOUNDS_MS.size(); bucket++) {
		if (held_ns < HOLD_BUCKET_BOUNDS_MS[bucket] * NS_PER_MS_INT) {
			return bucket;
		}
	}
	return HOLD_BUCKET_BOUNDS_MS.size();
}

void HoldHistogram::Add(uint64_t held_ns) {
	buckets[BucketOf(held_ns)]++;
	count++;
	max_ns = std::max(max_ns, held_ns);
}

double HoldHistogram::MaxMs() const {
	return static_cast<double>(max_ns) / NS_PER_MS;
}

double ExactPercentileMs(std::span<const uint64_t> sorted_ns, double q) {
	if (sorted_ns.empty()) {
		return 0.0;
	}
	// Nearest rank, counted from 1, so the rank is never below the first or past the last.
	const auto count = static_cast<double>(sorted_ns.size());
	const auto rank =
	    std::clamp<size_t>(static_cast<size_t>(std::ceil(q * count)), 1, sorted_ns.size());
	return static_cast<double>(sorted_ns[rank - 1]) / NS_PER_MS;
}

double HoldHistogram::PercentileMs(double q) const {
	if (count == 0) {
		return 0.0;
	}
	// Nearest rank, so p50 of ten holds is the fifth shortest and p95 is the tenth.
	const auto rank =
	    std::max<uint64_t>(1, static_cast<uint64_t>(std::ceil(q * static_cast<double>(count))));
	uint64_t seen = 0;
	for (size_t bucket = 0; bucket < buckets.size(); bucket++) {
		seen += buckets[bucket];
		if (seen >= rank) {
			// The last bucket has no bound, and no bucket reports more than the longest hold.
			const double bound = bucket < HOLD_BUCKET_BOUNDS_MS.size() ? HOLD_BUCKET_BOUNDS_MS[bucket]
			                                                           : MaxMs();
			return std::min(bound, MaxMs());
		}
	}
	return MaxMs();
}

void StallTracker::Begin(Clock::time_point start) {
	Begin(start, 0);
}

void StallTracker::Begin(Clock::time_point start, uint64_t id) {
	m_starts.insert({start, id});
}

void StallTracker::End(Clock::time_point start) {
	// The lowest id sorts first, so this finds a stall of that start whatever its id.
	const auto found = m_starts.lower_bound({start, 0});
	if (found != m_starts.end() && found->first == start) {
		m_starts.erase(found);
	}
}

void StallTracker::End(Clock::time_point start, uint64_t id) {
	const auto found = m_starts.find({start, id});
	if (found != m_starts.end()) {
		m_starts.erase(found);
	}
}

bool StallTracker::IsStalled(Clock::time_point now, Clock::duration threshold) const {
	return !m_starts.empty() && now - m_starts.begin()->first >= threshold;
}

std::optional<uint64_t> StallTracker::OldestId() const {
	if (m_starts.empty()) {
		return std::nullopt;
	}
	return m_starts.begin()->second;
}

void Configure(bool enabled, Clock::duration threshold) {
	auto& state = GetState();
	{
		LockGuard lock(state.mutex);
		state.enabled   = enabled;
		state.threshold = threshold;
		g_enabled.store(enabled, std::memory_order_relaxed);
	}
	// Waiters recheck the new settings, and a disable lets them all go.
	state.released.SignalAll();
}

Scope::Scope(HoldCause cause)
    : m_start(Clock::now()), m_id(g_next_scope_id.fetch_add(1, std::memory_order_relaxed)),
      m_cause(cause) {
	auto& state = GetState();
	LockGuard lock(state.mutex);
	state.tracker.Begin(m_start, m_id);
	g_active_scopes.fetch_add(1, std::memory_order_relaxed);
	// Release, after the active count, so a lock-free reader that sees this Scope counted with an
	// acquire load also sees it active. The prescan's build samples rely on that pairing.
	g_scope_count.fetch_add(1, std::memory_order_release);
	t_owned_scopes++;
}

Scope::~Scope() {
	auto&                            state = GetState();
	std::optional<Clock::time_point> stall_first_hold;
	HoldCause                        stall_cause = HoldCause::Other;
	{
		LockGuard lock(state.mutex);
		state.tracker.End(m_start, m_id);
		// The Scope that was oldest at the first hold is charged, however many others overlapped it
		// or outlasted it. Its cause is read now so that a late Reclassify still counts.
		if (m_id == state.hold_scope_id) {
			state.hold_cause = m_cause;
		}
		// Overlapping Scopes make one stall, which ends with the last of them.
		if (g_active_scopes.fetch_sub(1, std::memory_order_relaxed) == 1) {
			stall_first_hold = std::exchange(state.first_hold, std::nullopt);
			stall_cause      = std::exchange(state.hold_cause, HoldCause::Other);
			state.hold_scope_id = 0;
			// The count goes up before the flag drops, and the release pairs with HoldInProgress's
			// acquire, so a reader that sees the hold over also sees it counted. A hold can then
			// never slip between the two reads of a worker that samples both.
			if (stall_first_hold.has_value()) {
				g_hold_count.fetch_add(1, std::memory_order_relaxed);
			}
			g_hold_in_progress.store(false, std::memory_order_release);
		}
		t_owned_scopes--;
	}
	state.released.SignalAll();
	if (stall_first_hold.has_value()) {
		const auto held_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		    Clock::now() - *stall_first_hold);
		g_held_ns.fetch_add(static_cast<uint64_t>(held_ns.count()), std::memory_order_relaxed);
		const auto end_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
		                        Clock::now().time_since_epoch())
		                        .count();
		RecordHold(stall_cause, static_cast<uint64_t>(held_ns.count()), end_ns);
		DebugCounters::Add(DebugCounters::Counter::AudioHolds);
		DebugCounters::Add(DebugCounters::Counter::AudioHoldNs,
		                   static_cast<uint64_t>(held_ns.count()));
	}
}

namespace {

// Waits while a Scope is past the threshold and holding is on, and reports whether it waited.
bool HoldWhileStalled() {
	if (!g_enabled.load(std::memory_order_relaxed) ||
	    g_active_scopes.load(std::memory_order_relaxed) == 0 || t_owned_scopes != 0) {
		return false;
	}
	auto&             state      = GetState();
	bool              held       = false;
	Clock::time_point hold_start = {};
	{
		LockGuard lock(state.mutex);
		// Wakeups can be spurious or periodic, so the condition is checked again after each one.
		while (state.enabled && state.tracker.IsStalled(Clock::now(), state.threshold)) {
			if (!held) {
				held       = true;
				hold_start = Clock::now();
				state.waiters++;
				if (!state.first_hold.has_value()) {
					state.first_hold = hold_start;
					g_hold_in_progress.store(true, std::memory_order_relaxed);
					g_last_throttle_signal_ns.store(SteadyNowNs(), std::memory_order_relaxed);
					// The stall is past the threshold, so the oldest Scope is the one that crossed it.
					state.hold_scope_id = state.tracker.OldestId().value_or(0);
				}
			}
			state.released.Wait(&state.mutex);
		}
		if (held) {
			state.waiters--;
		}
	}
	// Logged after unlocking, so slow log output never delays the GPU thread ending its Scope.
	if (held) {
		LOGF("GpuStall: held guest audio for %lld ms while a pipeline compile blocked the GPU\n",
		     static_cast<long long>(ElapsedMs(hold_start)));
	}
	return held;
}

} // namespace

bool WaitWhileStalled() {
	// Set on every call, so only the first submit after a hold is treated as resuming from it.
	t_resumed_from_hold = HoldWhileStalled();
	return t_resumed_from_hold;
}

bool ResumedFromHold() {
	return t_resumed_from_hold;
}

bool IsStalled() {
	if (!g_enabled.load(std::memory_order_relaxed) ||
	    g_active_scopes.load(std::memory_order_relaxed) == 0) {
		return false;
	}
	auto&     state = GetState();
	LockGuard lock(state.mutex);
	return state.enabled && state.tracker.IsStalled(Clock::now(), state.threshold);
}

uint64_t HoldCount() {
	return g_hold_count.load(std::memory_order_relaxed);
}

bool HoldInProgress() {
	return g_hold_in_progress.load(std::memory_order_acquire);
}

uint64_t HeldNs() {
	return g_held_ns.load(std::memory_order_relaxed);
}

HoldBreakdown Breakdown() {
	HoldBreakdown result;
	for (size_t cause = 0; cause < HOLD_CAUSE_COUNT; cause++) {
		result.cause_ns[cause]    = g_cause_ns[cause].load(std::memory_order_relaxed);
		result.cause_count[cause] = g_cause_count[cause].load(std::memory_order_relaxed);
	}
	for (size_t bucket = 0; bucket < HOLD_BUCKET_COUNT; bucket++) {
		result.histogram.buckets[bucket] = g_hold_buckets[bucket].load(std::memory_order_relaxed);
		result.histogram.count += result.histogram.buckets[bucket];
	}
	result.histogram.max_ns = g_hold_max_ns.load(std::memory_order_relaxed);

	// Sorted here, when the report is written, and never when a hold ends.
	std::vector<uint64_t> durations;
	for (const auto& record: HoldRecords()) {
		durations.push_back(record.held_ns);
	}
	std::ranges::sort(durations);
	result.exact.count    = durations.size();
	result.exact.overflow = g_record_overflow.load(std::memory_order_relaxed);
	result.exact.p50_ms   = ExactPercentileMs(durations, 0.50);
	result.exact.p95_ms   = ExactPercentileMs(durations, 0.95);
	result.exact.max_ms   = result.histogram.MaxMs();
	return result;
}

std::vector<HoldRecord> HoldRecords() {
	std::vector<HoldRecord> records;
	const size_t taken = std::min(g_record_next.load(std::memory_order_relaxed), HOLD_RECORD_CAPACITY);
	for (size_t slot = 0; slot < taken; slot++) {
		// Acquire pairs with the release store of the held time, which follows the other fields.
		const auto held_plus_one = g_record_held_plus_one[slot].load(std::memory_order_acquire);
		if (held_plus_one != 0) {
			const auto cause = g_record_cause[slot].load(std::memory_order_relaxed);
			records.push_back({.held_ns = held_plus_one - 1,
			                   .end_ns  = g_record_end_ns[slot].load(std::memory_order_relaxed),
			                   .cause   = static_cast<HoldCause>(cause)});
		}
	}
	return records;
}

int64_t LastThrottleSignalNs() noexcept {
	return g_last_throttle_signal_ns.load(std::memory_order_relaxed);
}

void NoteAudioLowWater() noexcept {
	g_audio_low_water.fetch_add(1, std::memory_order_relaxed);
	g_last_throttle_signal_ns.store(SteadyNowNs(), std::memory_order_relaxed);
}

uint64_t AudioLowWaterCount() noexcept {
	return g_audio_low_water.load(std::memory_order_relaxed);
}

// Acquire pairs with the release in the Scope's constructor, so a Scope seen counted here is also
// seen active by a later AnyScopeActive on this thread.
uint64_t ScopeCount() {
	return g_scope_count.load(std::memory_order_acquire);
}

bool AnyScopeActive() {
	return g_active_scopes.load(std::memory_order_relaxed) != 0;
}

uint32_t WaiterCount() {
	auto&     state = GetState();
	LockGuard lock(state.mutex);
	return state.waiters;
}

} // namespace Common::GpuStall
