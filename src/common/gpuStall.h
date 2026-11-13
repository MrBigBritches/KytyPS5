#ifndef KYTY_COMMON_GPU_STALL_H_
#define KYTY_COMMON_GPU_STALL_H_

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <span>
#include <utility>
#include <vector>

// Holds guest audio output while the GPU command thread is stuck in a long pipeline compile.
// Games that pace cutscenes by their audio clock otherwise skip ahead while no frames are drawn.
namespace Common::GpuStall {

using Clock = std::chrono::steady_clock;

// Compiles shorter than this let audio run on. The host queue hides a hold of about 50 ms at
// most, and most cold compiles in measured games finish under 250 ms, so holding for those would
// add audible dropouts to stalls nobody notices.
inline constexpr std::chrono::milliseconds GPU_STALL_AUDIO_HOLD_THRESHOLD {250};

/// Why the GPU command thread was stopped, so a hold of guest audio can be blamed on one thing.
/// PrescanWaited and PrescanQueued are charged by the warm takes of prescan modules, a wait for a
/// worker's build and a build the GPU thread ran itself because it was still queued. The other
/// prescan causes have no call sites yet. They are PrescanInFlight, for a request still in the
/// helper, the three PrescanNeverPosted causes and PrescanRefused, and they are reserved for the
/// GPU thread's compile of a module the prescan did not deliver, so the report's schema does not
/// change when they get sites. Other is a gate value that must stay 0 in a finished run. WarmInlinePaused and
/// GeneratedInlinePaused are causes of their own and not subsets, so the causes add up to the total
/// held time. The cost of the pause is their sum, and the generator's misses are GeneratedInline
/// plus GeneratedInlinePaused. LibraryOnce is reserved and
/// has no site, because a wait for a library sits inside the graphics compile's Scope and is charged
/// to that, so a gate must not read it. Other must stay the last value, since the count derives
/// from it.
enum class HoldCause : uint8_t {
	RequiredGraphics,
	RequiredCompute,
	WarmInline,
	WarmInlinePaused,
	GeneratedInline,
	GeneratedInlinePaused,
	WarmWaited,
	GeneratedWaited,
	PrescanWaited,
	PrescanInFlight,
	PrescanNeverPostedAsync,
	PrescanNeverPostedIndirect,
	PrescanNeverPostedDeep,
	PrescanRefused,
	DeferredDrawResolve,
	IneligibleDraw,
	LibraryOnce,
	PrescanQueued,
	Other,
};
inline constexpr size_t HOLD_CAUSE_COUNT = static_cast<size_t>(HoldCause::Other) + 1;

/// The name a cause has in the report, in lower case with underscores.
[[nodiscard]] const char* HoldCauseName(HoldCause cause);

/// Upper bounds in ms of the hold histogram. A hold counts in the first bucket whose bound is above
/// it, and a hold at or past the last bound counts in the extra bucket after them.
inline constexpr std::array<uint32_t, 11> HOLD_BUCKET_BOUNDS_MS {10,  25,   50,   100,  250, 500,
                                                                   1000, 2500, 5000, 10000, 30000};
inline constexpr size_t HOLD_BUCKET_COUNT = HOLD_BUCKET_BOUNDS_MS.size() + 1;

/// Fixed buckets of hold durations, for the report's display summary. A gate reads the exact figures
/// in HoldExact instead, since a bucket is far wider than the margin a gate allows. It never
/// allocates, so a hold can be recorded from the GPU thread.
struct HoldHistogram {
	std::array<uint64_t, HOLD_BUCKET_COUNT> buckets {};
	uint64_t                                count  = 0;
	uint64_t                                max_ns = 0;

	/// The index of the bucket a hold of this many nanoseconds counts in.
	[[nodiscard]] static size_t BucketOf(uint64_t held_ns);
	/// Counts one hold. The process-wide histogram is filled by atomics instead, so this is for a
	/// histogram that a test or a caller builds by hand.
	void Add(uint64_t held_ns);
	/// The nearest-rank percentile for q between 0 and 1, in ms. It is the upper bound of the bucket
	/// that holds the rank, so it is at most one bucket wide of the truth, and it never exceeds the
	/// longest hold. 0 when nothing was added.
	[[nodiscard]] double PercentileMs(double q) const;
	/// The longest hold in ms.
	[[nodiscard]] double MaxMs() const;
};

/// One finished hold of guest audio. end_ns is on the steady clock.
struct HoldRecord {
	uint64_t  held_ns = 0;
	int64_t   end_ns  = 0;
	HoldCause cause   = HoldCause::Other;
};

/// How many holds are kept one by one. Past it they are only counted in overflow, and the exact
/// percentiles then cover the first holds only.
inline constexpr size_t HOLD_RECORD_CAPACITY = 4096;

/// Exact figures of the held durations, from the stored records. max_ms covers every hold, including
/// those past the capacity.
struct HoldExact {
	uint64_t count    = 0;
	uint64_t overflow = 0;
	double   p50_ms   = 0.0;
	double   p95_ms   = 0.0;
	double   max_ms   = 0.0;
};

/// The nearest-rank percentile of durations that are sorted ascending, in ms, for q between 0 and 1.
/// The rank is the ceiling of q times the count, so p50 of ten holds is the fifth shortest and p95
/// is the tenth. 0 for an empty list.
[[nodiscard]] double ExactPercentileMs(std::span<const uint64_t> sorted_ns, double q);

/// What the holds since the process began add up to, by cause, with the histogram of their lengths.
/// cause_ns adds up to HeldNs(), and cause_count to HoldCount(), up to holds that end while this is
/// read.
struct HoldBreakdown {
	std::array<uint64_t, HOLD_CAUSE_COUNT> cause_ns {};
	std::array<uint64_t, HOLD_CAUSE_COUNT> cause_count {};
	HoldHistogram                          histogram;
	HoldExact                              exact;
};

// Bookkeeping of the stalls in progress, driven by explicit time points so tests can use a fake
// clock. Not thread safe on its own.
class StallTracker {
public:
	/// Records a stall that began at start.
	void Begin(Clock::time_point start);
	/// Records a stall that began at start and is told apart from others by id, which is not 0.
	void Begin(Clock::time_point start, uint64_t id);
	/// Removes one stall that began at start.
	void End(Clock::time_point start);
	/// Removes the stall that began at start with this id.
	void End(Clock::time_point start, uint64_t id);
	/// True when the oldest stall in progress has lasted at least threshold.
	[[nodiscard]] bool IsStalled(Clock::time_point now, Clock::duration threshold) const;
	/// The id of the oldest stall in progress, which is the one that crosses the threshold first. It is
	/// 0 for a stall begun without an id, and nothing when none is in progress.
	[[nodiscard]] std::optional<uint64_t> OldestId() const;

private:
	// Ordered so the oldest start is first. Equal starts are ordered by id.
	std::multiset<std::pair<Clock::time_point, uint64_t>> m_starts;
};

/// Turns holding on or off and sets the threshold. Turning it off releases every held thread.
void Configure(bool enabled, Clock::duration threshold = GPU_STALL_AUDIO_HOLD_THRESHOLD);

// Marks the GPU command thread as blocked for its lifetime. Construct one only where that thread
// itself waits, never around work it hands to another thread, or audio would stop while the game
// keeps drawing. A thread that owns a Scope never waits in WaitWhileStalled, so a Scope must be
// destroyed on the thread that created it. The cause has no default, so every site must name one.
// When Scopes overlap, a hold is charged to the cause of the oldest one active when audio was first
// held, which is the one that crossed the threshold.
class Scope {
public:
	explicit Scope(HoldCause cause);
	~Scope();
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

	/// Changes the cause, for a site that learns what it waited for only after it began waiting. It
	/// takes effect for a hold that is still in progress, since the cause is read when the Scope ends.
	void Reclassify(HoldCause cause) noexcept { m_cause = cause; }

private:
	Clock::time_point m_start;
	uint64_t          m_id;
	HoldCause         m_cause;
};

/// Blocks the caller while holding is on and a Scope has lasted at least the threshold.
/// Returns true when it waited. Guest audio submit entry points call this before anything else,
/// and must not hold a lock the GPU command thread could need, or the compile could never end.
/// There is no time limit. A compile that never ends has already frozen the game, and stalls
/// of over 30 s are real, so giving up early would bring back the skipped scene. Each call also
/// records whether it waited, which ResumedFromHold reports.
bool WaitWhileStalled();

/// True when this thread's most recent WaitWhileStalled call waited. The first audio submit after a
/// hold finds the host queue drained on purpose, and uses this to keep that out of underrun counts.
[[nodiscard]] bool ResumedFromHold();

/// True when holding is on and a Scope has lasted at least the threshold. Unlike
/// WaitWhileStalled it does not exempt the thread that owns the Scope.
[[nodiscard]] bool IsStalled();

/// Stalls that held at least one thread since the process began. Unlike the performance panel's
/// counter it always counts, so a run report can read it.
[[nodiscard]] uint64_t HoldCount();

/// Nanoseconds that guest audio was held in total since the process began, counted when each stall
/// ends, from the first thread it held to the end of the stall. Unlike the performance panel's
/// counter it always counts. A stall shorter than the threshold holds nothing and adds nothing, and
/// a stall still in progress is not in the total yet, so read it before and after to get a window.
/// A stall is counted in the window where it ends, so a window nested inside an outer scope reads 0.
[[nodiscard]] uint64_t HeldNs();

/// The holds since the process began by cause and length, read without a lock. The figures are
/// separate relaxed loads, so a hold that ends meanwhile may show in one and not another.
[[nodiscard]] HoldBreakdown Breakdown();

/// The stored hold records in slot reservation order, which matches end order because stall ends
/// are serialized, for the exact figures and for tests. It
/// allocates, so it is for the report writer and never for a hold.
[[nodiscard]] std::vector<HoldRecord> HoldRecords();

/// Scopes created since the process began, so a test can tell one scope from several. Unlike
/// HoldCount it does not depend on a thread being held while each scope lasts.
[[nodiscard]] uint64_t ScopeCount();

/// True while any Scope exists, held long enough to hold audio or not. Work on other threads reads
/// it with ScopeCount to tell whether it overlapped a stop of the GPU thread.
[[nodiscard]] bool AnyScopeActive();

/// Number of threads waiting in WaitWhileStalled, so tests can tell when a waiter is parked.
[[nodiscard]] uint32_t WaiterCount();

} // namespace Common::GpuStall

#endif /* KYTY_COMMON_GPU_STALL_H_ */
