#include "graphics/guest_gpu/command_processor/prescanScanner.h"

#include "common/logging/log.h"
#include "common/threadPriority.h"
#include "graphics/host_gpu/renderer/pipeline/prescanHandoff.h"
#include "graphics/host_gpu/renderer/pipeline/prescanVariants.h"

#include <algorithm>
#include <bit>
#include <exception>
#include <fmt/format.h>
#include <stdexcept>
#include <utility>

namespace Libs::Graphics {

namespace {

using Clock = std::chrono::steady_clock;

// Queue 0 is the graphics ring and every other queue is an async compute queue, as the command
// processor numbers them.
constexpr uint32_t GRAPHICS_QUEUE_ID = 0;

// Whole microseconds or milliseconds of a duration, with a negative one counted as zero.
template <typename Unit>
uint64_t WholeUnits(Clock::duration duration) {
	return static_cast<uint64_t>(std::max<int64_t>(0, std::chrono::duration_cast<Unit>(duration).count()));
}

// Adds one value to a histogram by powers of two.
void AddToHistogram(PrescanHistogram& histogram, uint64_t value) {
	histogram[std::min<size_t>(std::bit_width(value), PRESCAN_HISTOGRAM_BUCKETS - 1)]++;
}

// Raises an atomic maximum. Only the submitting threads race on it, and a lost race retries.
void RaiseMax(std::atomic<uint64_t>& maximum, uint64_t value) {
	uint64_t seen = maximum.load(std::memory_order_relaxed);
	while (seen < value && !maximum.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
	}
}

// The constructor's arguments, checked in its initializer list so that a bad one throws before the
// job queue allocates its buffers.
ScanMemoryReader CheckedReader(ScanMemoryReader reader) {
	if (!reader) {
		throw std::invalid_argument("PrescanScanner needs a reader");
	}
	return reader;
}

std::chrono::microseconds CheckedTime(std::chrono::microseconds time) {
	if (time.count() <= 0) {
		throw std::invalid_argument("PrescanScanner's lag bound and least walk time must be positive");
	}
	return time;
}

} // namespace

bool ChunkedScanRead(const ScanMemoryReader& inner, uint64_t address, void* destination, uint64_t bytes,
                     const std::atomic<bool>* stop, ScanReadTiming* timing) {
	auto* out = static_cast<uint8_t*>(destination);
	for (uint64_t done = 0; done < bytes;) {
		if (stop != nullptr && stop->load(std::memory_order_relaxed)) {
			return false;
		}
		const uint64_t step  = std::min(bytes - done, SCAN_READ_CHUNK_BYTES);
		const auto     start = timing != nullptr ? Clock::now() : Clock::time_point {};
		const bool     read  = inner(address + done, out + done, step);
		if (timing != nullptr) {
			const auto took = Clock::now() - start;
			timing->calls++;
			timing->max_us = std::max(timing->max_us, WholeUnits<std::chrono::microseconds>(took));
			timing->slow += took > SCAN_SLOW_READ ? 1 : 0;
		}
		if (!read) {
			return false;
		}
		done += step;
	}
	return true;
}

PrescanScanner::PrescanScanner(PrescanHandoff& handoff, ScanMemoryReader inner_reader,
                               std::chrono::microseconds lag_bound, std::chrono::microseconds min_walk_time)
    : m_handoff(handoff),
      m_inner_reader(CheckedReader(std::move(inner_reader))),
      m_lag_bound(CheckedTime(lag_bound)),
      m_min_walk_time(CheckedTime(min_walk_time)),
      m_scanner([this](uint64_t address, void* destination, uint64_t bytes) {
	      return ChunkedScanRead(m_inner_reader, address, destination, bytes, &m_stopping, &m_read_timing);
      }),
      m_cache(PRESCAN_PROBE_INTERVAL) {
	// The scanner has seen none of the writes that set the queues' programs so far.
	for (auto& shadow: m_shadows) {
		shadow.MarkUncertain();
	}
	m_dispatches.reserve(SCAN_DISPATCH_CAPACITY);
}

PrescanScanner::~PrescanScanner() {
	try {
		Stop();
	} catch (...) {
		// Only a join from the scanner thread itself can fail, which nothing does. If one ever did, the
		// thread would still be joinable and its destructor would end the process.
	}
}

// The queue stamps the job's arrival before it takes its lock or copies, so the time since then is the
// whole copy, the wait for the lock included, and costs one clock read.
PreparedScanJob PrescanScanner::PrepareCommands(uint32_t queue_id, std::span<const uint32_t> commands) {
	PreparedScanJob prepared {.job = m_queue.MakeCommands(queue_id, commands)};
	prepared.copy_us = std::chrono::duration<double, std::micro>(Clock::now() - prepared.job.arrival).count();
	return prepared;
}

ScanJob PrescanScanner::PrepareReset(uint32_t queue_id) {
	return ScanJobQueue::MakeReset(queue_id);
}

uint64_t PrescanScanner::Push(ScanJob job, double copy_us) {
	const auto     start    = Clock::now();
	const uint64_t seq      = m_queue.Push(std::move(job), copy_us);
	const uint64_t micros   = WholeUnits<std::chrono::microseconds>(Clock::now() - start);
	const size_t   bucket   = std::min<size_t>(std::bit_width(micros), PRESCAN_HISTOGRAM_BUCKETS - 1);
	m_push_us_histogram[bucket].fetch_add(1, std::memory_order_relaxed);
	RaiseMax(m_push_us_max, micros);
	return seq;
}

void PrescanScanner::NoteBegan(uint32_t queue_id, uint64_t seq) {
	m_queue.NoteBegan(queue_id, seq);
}

void PrescanScanner::Start() {
	std::lock_guard lock(m_stop_mutex);
	if (m_thread.joinable() || m_finished.load(std::memory_order_acquire)) {
		throw std::logic_error("PrescanScanner was already started");
	}
	if (m_stopping.load(std::memory_order_relaxed)) {
		throw std::logic_error("PrescanScanner was already stopped");
	}
	m_thread = std::thread([this] { ThreadMain(); });
}

// The flag goes first so that a walk in a long read gives up at its next chunk, and the queue's stop
// then wakes a thread that waits for a job.
void PrescanScanner::Stop() {
	std::lock_guard lock(m_stop_mutex);
	m_stopping.store(true, std::memory_order_relaxed);
	m_queue.Stop();
	if (m_thread.joinable()) {
		m_thread.join();
	}
}

bool PrescanScanner::Finished() const {
	return m_finished.load(std::memory_order_acquire);
}

PrescanScannerStats PrescanScanner::Stats() const {
	PrescanScannerStats stats;
	{
		std::lock_guard lock(m_stats_mutex);
		stats = m_stats;
	}
	// The queue's lock is taken apart from the stats lock, so the two are never nested.
	stats.queue       = m_queue.Stats();
	stats.push_us_max = m_push_us_max.load(std::memory_order_relaxed);
	for (size_t bucket = 0; bucket < PRESCAN_HISTOGRAM_BUCKETS; ++bucket) {
		stats.push_us_histogram[bucket] = m_push_us_histogram[bucket].load(std::memory_order_relaxed);
	}
	return stats;
}

// The thread's body. Any error stops the queue, so the submitting threads stop copying lists that
// nobody would walk, and is logged, since an uncaught throw would end the process while the game runs
// on fine without the prescan. No signal stack is set up, because the scanner's reads never fault.
void PrescanScanner::ThreadMain() noexcept {
	// Below normal, like the preparer, so the guest's threads win any contention.
	(void)Common::LowerCurrentThreadPriority();
	try {
		Loop();
	} catch (const std::exception& error) {
		m_queue.Stop();
		try {
			Log::WriteToConsoleAndLog(fmt::format("Prescan: the scanner stopped on an error: {}\n", error.what()));
		} catch (...) {
		}
	} catch (...) {
		m_queue.Stop();
		try {
			Log::WriteToConsoleAndLog("Prescan: the scanner stopped on an unknown error\n");
		} catch (...) {
		}
	}
	m_finished.store(true, std::memory_order_release);
}

// Each Pop gives back the buffer of the job before it, so the last job's buffer is returned on the
// way out. The preparer's flag is read once per job. Once it is set nothing drains the ring, so the
// queue is stopped as well, which makes the submitting threads' copies return at once.
void PrescanScanner::Loop() {
	ScanJob job;
	while (m_queue.Pop(job)) {
		if (m_handoff.preparer_stopped.load(std::memory_order_acquire)) {
			{
				std::lock_guard lock(m_stats_mutex);
				m_stats.stopped_by_preparer = 1;
			}
			m_queue.Stop();
			break;
		}
		RunJob(job);
	}
	m_queue.Recycle(std::move(job));
}

// A reset marker always resets the shadow, however late, since skipping it would leave the shadow
// wrong for every later list of the queue. A command list older than the lag bound is not walked,
// and its queue's shadow is put in doubt, since the list may have set a program the scanner never saw.
void PrescanScanner::RunJob(const ScanJob& job) {
	JobCounts  counts;
	CsShadow&  shadow  = ShadowOf(job.queue_id);
	const auto lag     = Clock::now() - job.arrival;
	const bool commands = job.kind == ScanJob::Kind::Commands;
	if (commands) {
		counts.has_lag = true;
		counts.lag_ms  = WholeUnits<std::chrono::milliseconds>(lag);
	}
	if (commands && lag > m_lag_bound) {
		shadow.MarkUncertain();
		counts.lag_skipped = true;
	} else {
		if (commands) {
			m_scanner.SetTimeCap(WalkTimeFor(lag));
		}
		WalkAndOffer(job, shadow, counts);
	}
	Record(counts);
}

// Walks the job into a local count, so the stats lock is not held during the walk, then offers what
// it found. The dispatch storage is emptied first, since a walk appends to it.
void PrescanScanner::WalkAndOffer(const ScanJob& job, CsShadow& shadow, JobCounts& counts) {
	m_dispatches.clear();
	const auto start = Clock::now();
	RunScanJob(m_scanner, job, job.queue_id != GRAPHICS_QUEUE_ID, shadow, m_dispatches, counts.scan, &m_cache,
	           &m_scratch);
	counts.scan_us = WholeUnits<std::chrono::microseconds>(Clock::now() - start);
	counts.walked  = job.kind == ScanJob::Kind::Commands;
	OfferDispatches(job.seq, counts);
}

// Offers each dispatch and wakes the preparer once if any went into the ring, since a wake may enter
// the kernel and one per dispatch would cost the scanner more than the walk.
void PrescanScanner::OfferDispatches(uint64_t seq, JobCounts& counts) {
	for (const auto& dispatch: m_dispatches) {
		switch (OfferToPreparer(m_handoff, m_cache, MakePrescanItem(dispatch, seq))) {
			case PrescanOffer::Posted: counts.posted++; break;
			case PrescanOffer::Skipped: counts.skipped++; break;
			case PrescanOffer::Dropped: counts.dropped++; break;
		}
	}
	if (counts.posted != 0) {
		WakePreparer(m_handoff);
	}
}

// Merges one job's counts, and the reader's timing since the last job, under the stats lock.
void PrescanScanner::Record(const JobCounts& counts) {
	std::lock_guard lock(m_stats_mutex);
	m_stats.jobs++;
	m_stats.scan.Add(counts.scan);
	m_stats.lag_skipped += counts.lag_skipped ? 1 : 0;
	m_stats.offers_posted += counts.posted;
	m_stats.offers_skipped += counts.skipped;
	m_stats.offers_dropped += counts.dropped;
	m_stats.wakes += counts.posted != 0 ? 1 : 0;
	if (counts.walked) {
		m_stats.scan_us_max = std::max(m_stats.scan_us_max, counts.scan_us);
		AddToHistogram(m_stats.scan_us_histogram, counts.scan_us);
	}
	if (counts.has_lag) {
		m_stats.lag_ms_max = std::max(m_stats.lag_ms_max, counts.lag_ms);
		AddToHistogram(m_stats.lag_ms_histogram, counts.lag_ms);
	}
	m_stats.reads.calls += m_read_timing.calls;
	m_stats.reads.max_us = std::max(m_stats.reads.max_us, m_read_timing.max_us);
	m_stats.reads.slow += m_read_timing.slow;
	m_read_timing = {};
}

// What is left of the lag bound, but never less than the least walk time, so a list near the bound
// still gets a useful walk even when the thread was preempted on the way.
std::chrono::microseconds PrescanScanner::WalkTimeFor(Clock::duration lag) const {
	const auto left = m_lag_bound - std::chrono::duration_cast<std::chrono::microseconds>(lag);
	return std::clamp(left, m_min_walk_time, std::max(m_min_walk_time, m_lag_bound));
}

// A queue id past the last slot shares it, as the job queue does.
CsShadow& PrescanScanner::ShadowOf(uint32_t queue_id) {
	return m_shadows[std::min<size_t>(queue_id, SCAN_MAX_QUEUES - 1)];
}

} // namespace Libs::Graphics
