#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PRESCAN_SCANNER_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PRESCAN_SCANNER_H

#include "common/spscRing.h"
#include "graphics/guest_gpu/command_processor/scanJobQueue.h"
#include "graphics/guest_gpu/command_processor/scanScratch.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

// The scanner thread of the submit-time compute prescan. Submitting threads copy each command list
// into the job queue and go on. This one thread walks the lists in each queue's order, keeps one
// register shadow per queue, and offers every dispatch it finds to the preparer through the hand-off.
// It never takes a lock a submitting thread or the GPU thread waits on for long, never reads guest
// memory except through a reader that cannot fault, and allocates nothing once its storage has grown
// to the largest job.

// Only named as a return type here, so the report's header stays out of every file that includes this.
namespace Common::FirstRunReport {
struct PrescanScannerReport;
struct PrescanGraphicsReport;
} // namespace Common::FirstRunReport

namespace Libs::Graphics {

// The hand-off lives with the pipeline cache. Only a reference is kept here, so this header does not
// pull the host side's headers into the command processor's.
struct PrescanHandoff;

/// How old a command list may be when the scanner reaches it and still be walked. It is a backstop
/// against a starved scanner, which would otherwise spend its time on a backlog of old lists, and not a
/// tuning knob, so it does not depend on the game. Lists the GPU thread has already begun are skipped
/// through NoteBegan, whatever their age.
inline constexpr std::chrono::milliseconds PRESCAN_LAG_BOUND {250};
/// The least time a walk is given, however near the lag bound its list is. A walk's wall time includes
/// any time the scanner thread was preempted, which reached 15 to 18 ms in one game.
inline constexpr std::chrono::milliseconds PRESCAN_MIN_WALK_TIME {8};
/// The most bytes one call to the inner reader asks for. The production reader holds the guest address
/// space's shared lock for the whole copy, so a large indirect buffer read in one call would hold up
/// anything that maps or unmaps memory meanwhile.
inline constexpr uint64_t SCAN_READ_CHUNK_BYTES = uint64_t {64} * 1024;
/// A read of one chunk that takes longer than this is counted as slow.
inline constexpr std::chrono::milliseconds SCAN_SLOW_READ {1};
/// How many buckets each of the scanner's histograms has. Bucket n counts values below 2^n and at least
/// half that, bucket 0 counts zero, and the last takes all larger ones.
inline constexpr size_t PRESCAN_HISTOGRAM_BUCKETS = 24;

/// A fixed histogram by powers of two, which never allocates.
using PrescanHistogram = std::array<uint64_t, PRESCAN_HISTOGRAM_BUCKETS>;

/// What the scanner's reader measured about the calls it made to the inner reader.
struct ScanReadTiming {
	/// Calls to the inner reader, one per chunk.
	uint64_t calls = 0;
	/// The longest single call, in microseconds.
	uint64_t max_us = 0;
	/// Calls that took longer than SCAN_SLOW_READ.
	uint64_t slow = 0;
};

/// Reads guest memory through `inner` in steps of at most SCAN_READ_CHUNK_BYTES, so that no single
/// call holds the inner reader's lock for long. It stops at the first chunk that fails and returns
/// false, as the inner reader would have for the whole range. When `stop` is given and set, it gives up
/// before the next chunk and returns false, so a scanner being stopped is not held by a long read.
/// When `timing` is given, each chunk's call is timed into it.
[[nodiscard]] bool ChunkedScanRead(const ScanMemoryReader& inner, uint64_t address, void* destination,
                                   uint64_t bytes, const std::atomic<bool>* stop = nullptr,
                                   ScanReadTiming* timing = nullptr);

/// A command list copied for the scanner and what the copy cost, ready to be pushed.
struct PreparedScanJob {
	ScanJob job;
	/// How long the copy took, the wait for the queue's lock included, in microseconds.
	double copy_us = 0.0;
};

/// What the scanner has counted. Stats() returns a copy.
struct PrescanScannerStats {
	/// What every walk counted, summed.
	ScanStats scan;
	/// What the job queue counted.
	ScanJobQueueStats queue;
	/// Jobs the thread took from the queue and finished, which counts walks, resets and lag skips.
	uint64_t jobs = 0;
	/// Command lists older than the lag bound, which were not walked and left their queue in doubt.
	uint64_t lag_skipped = 0;
	/// 1 once the thread stopped because the preparer had stopped.
	uint64_t stopped_by_preparer = 0;
	/// What became of the dispatches offered to the preparer, see PrescanOffer.
	uint64_t offers_posted  = 0;
	uint64_t offers_skipped = 0;
	uint64_t offers_dropped = 0;
	/// How many times the preparer was woken, at most once per walked job.
	uint64_t wakes = 0;
	/// How long each walk took, in microseconds.
	uint64_t         scan_us_max = 0;
	PrescanHistogram scan_us_histogram {};
	/// How old each command list was when the thread took it, in milliseconds. Reset markers are left out.
	uint64_t         lag_ms_max = 0;
	PrescanHistogram lag_ms_histogram {};
	/// How long each Push took the submitting thread, in microseconds, the wait for the lock included.
	uint64_t         push_us_max = 0;
	PrescanHistogram push_us_histogram {};
	/// The calls the walks made to the inner reader.
	ScanReadTiming reads;
	/// What the graphics walk counted, when a graphics shadow is attached.
	GraphicsScanStats graphics;
	/// Draws handed to the graphics sink.
	uint64_t draws_offered = 0;
};

/// Walks this long or longer are counted apart in the report. It is a power of two, so the count is
/// exact from the histogram, whose buckets split at powers of two.
inline constexpr uint64_t PRESCAN_LONG_WALK_US = 1024;

/// The first-run report's view of the scanner's figures. Each histogram becomes nearest-rank
/// percentiles. A bucket spans a factor of two, so each percentile is the top of the bucket its rank
/// falls in, capped at the exact maximum, which makes it an upper bound.
[[nodiscard]] Common::FirstRunReport::PrescanScannerReport PrescanReportOf(const PrescanScannerStats& stats);

/// The scanner's part of the prescan_graphics block (experimental), the graphics walk's counts with the
/// draws handed to the sink.
[[nodiscard]] Common::FirstRunReport::PrescanGraphicsReport PrescanGraphicsScanReportOf(const PrescanScannerStats& stats);

/// Runs the scanner thread and owns everything it uses. The hand-off belongs to the caller and must
/// outlive this. Submitting threads call PrepareCommands or PrepareReset outside the order gate and
/// Push inside it. The GPU thread calls NoteBegan. Any thread may call Stats.
class PrescanScanner {
public:
	/// Builds the queue, the shadows and the scratch. Every shadow starts in doubt, since a scanner that
	/// starts while the game runs has not seen the writes that set the queues' programs.
	/// @param handoff        where dispatches go, which must outlive the scanner.
	/// @param inner_reader   the reader that cannot fault, which the scanner wraps in ChunkedScanRead.
	/// @param lag_bound      how old a list may be and still be walked.
	/// @param min_walk_time  the least time a walk is given.
	/// @throws std::invalid_argument when the reader is empty or the times are not positive.
	PrescanScanner(PrescanHandoff& handoff, ScanMemoryReader inner_reader,
	               std::chrono::microseconds lag_bound     = PRESCAN_LAG_BOUND,
	               std::chrono::microseconds min_walk_time = PRESCAN_MIN_WALK_TIME);
	/// Stops and joins the thread, so that a throw between Start and Stop never ends the process.
	~PrescanScanner();
	PrescanScanner(const PrescanScanner&)            = delete;
	PrescanScanner& operator=(const PrescanScanner&) = delete;

	/// Copies a command list for the scanner and times the copy. Any thread, outside the order gate.
	/// The job must be pushed afterwards, even when it was not copied, so that its buffer goes back.
	[[nodiscard]] PreparedScanJob PrepareCommands(uint32_t queue_id, std::span<const uint32_t> commands);

	/// A marker that resets the queue's shadow, for a flip suspend point.
	[[nodiscard]] static ScanJob PrepareReset(uint32_t queue_id);

	/// Queues a prepared job, inside the order gate, and returns its number in its queue, or 0 when the
	/// queue is stopped. The time it takes is recorded without a lock.
	uint64_t Push(ScanJob job, double copy_us);

	/// Says that the GPU thread began the queue's job `seq`. Lock-free, and safe after Stop.
	void NoteBegan(uint32_t queue_id, uint64_t seq);

	/// Makes the thread also walk the graphics queue's state with `graphics`, and hand the draws each
	/// walk finds to `sink`, for the graphics prescan (experimental). The thread owns the shadow from Start
	/// on. It is reset by the graphics queue's reset markers and put in doubt where that queue's compute
	/// shadow is. An empty sink drops the draws.
	/// @throws std::logic_error when the thread was already started, std::invalid_argument when graphics
	/// is null.
	void AttachGraphics(std::unique_ptr<GraphicsScan> graphics, GraphicsDrawSink sink);
	/// Whether a graphics shadow was attached. Set before Start or never, so any thread may ask after.
	[[nodiscard]] bool HasGraphics() const noexcept { return m_graphics != nullptr; }

	/// Starts the thread.
	/// @throws std::logic_error when it was already started, or Stop was already called, since a stopped
	/// queue would end the new thread at once.
	void Start();

	/// Stops the queue, which wakes the thread, and joins it. Jobs still queued are dropped. A walk in the
	/// middle of a buffer read gives up at its next chunk, but a walk that reads nothing runs on until its
	/// time cap, so the wait is bounded by the lag bound. Safe to call more than once and from more than
	/// one thread.
	void Stop();

	/// Whether the thread was started and has since ended, which it does on Stop, when the preparer
	/// stopped, or on an error.
	[[nodiscard]] bool Finished() const;

	/// A copy of everything counted so far. The scanner's own counts, the queue's and the push figures
	/// are each read under their own lock or atomically, so together they are not one consistent
	/// snapshot, and `jobs` may trail `queue.pushed` by a job in flight.
	[[nodiscard]] PrescanScannerStats Stats() const;

private:
	// What one job counted, gathered without the stats lock and merged after.
	struct JobCounts {
		ScanStats scan;
		bool      walked      = false;
		bool      lag_skipped = false;
		bool      has_lag     = false;
		uint64_t  lag_ms      = 0;
		uint64_t  scan_us     = 0;
		uint64_t  posted      = 0;
		uint64_t  skipped     = 0;
		uint64_t  dropped     = 0;
		GraphicsScanStats graphics;
		uint64_t          draws_offered = 0;
	};

	void ThreadMain() noexcept;
	void Loop();
	void RunJob(const ScanJob& job);
	void WalkAndOffer(const ScanJob& job, CsShadow& shadow, JobCounts& counts);
	void OfferDispatches(uint64_t seq, JobCounts& counts);
	void OfferDraws(uint64_t seq, JobCounts& counts);
	[[nodiscard]] GraphicsScan* GraphicsOf(uint32_t queue_id) const;
	void Record(const JobCounts& counts);
	[[nodiscard]] std::chrono::microseconds WalkTimeFor(std::chrono::steady_clock::duration lag) const;
	[[nodiscard]] CsShadow& ShadowOf(uint32_t queue_id);

	PrescanHandoff&           m_handoff;
	ScanMemoryReader          m_inner_reader;
	std::chrono::microseconds m_lag_bound;
	std::chrono::microseconds m_min_walk_time;
	// Set by Stop before the queue is stopped, and read by the reader between chunks.
	std::atomic<bool> m_stopping {false};
	std::atomic<bool> m_finished {false};
	ScanJobQueue      m_queue;

	// Only the scanner thread touches these, so they need no lock.
	SubmitScanner                          m_scanner;
	std::array<CsShadow, SCAN_MAX_QUEUES>  m_shadows;
	DispatchCache                          m_cache;
	std::vector<ScannedDispatch>           m_dispatches;
	ScanScratch                            m_scratch;
	ScanReadTiming                         m_read_timing;
	// The graphics queue's graphics shadow and where its draws go, both set before Start or never.
	std::unique_ptr<GraphicsScan> m_graphics;
	GraphicsDrawSink              m_graphics_sink;

	// Push runs on the submitting threads, so its figures are atomics and never wait on the stats lock.
	// They start a cache line of their own, so a submitter's Push does not fight the scanner thread over
	// the line that holds its read timing.
	alignas(Common::SPSC_FALSE_SHARING_BYTES) std::atomic<uint64_t> m_push_us_max {0};
	std::array<std::atomic<uint64_t>, PRESCAN_HISTOGRAM_BUCKETS>    m_push_us_histogram {};

	// Held only to merge one job's counts or to copy them out, never across a walk or an offer. It starts
	// a line of its own for the same reason.
	alignas(Common::SPSC_FALSE_SHARING_BYTES) mutable std::mutex m_stats_mutex;
	PrescanScannerStats                                          m_stats;

	std::mutex  m_stop_mutex;
	std::thread m_thread;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PRESCAN_SCANNER_H
