#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_JOB_QUEUE_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_JOB_QUEUE_H

#include "graphics/guest_gpu/command_processor/scanOrderGate.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

// The hand-off from the submitting threads to the one scanner thread of the off-thread prescan. A
// submitting thread only copies the top-level command list into a preallocated buffer and queues it,
// and never reads an indirect buffer, takes a prescan lock or allocates once the queue is built. The
// queue is first in, first out, so a queue's submissions reach the scanner in the order they were
// submitted, which keeps its register shadow right. It is bounded and drops a command list when full.
// Reset markers are never dropped, since a missed reset leaves the shadow wrong for everything after it.
// Anything that keeps the scanner from walking a list that was submitted leaves that queue's shadow in
// doubt, and the next list of the queue carries the flag.
namespace Libs::Graphics {

/// The most dwords of a command list the copy keeps, which is the scan's whole read budget. A longer list
/// is cut to it and counted as truncated. The copy must keep what the walk may read, or a list the budget
/// would have scanned is cut before the scanner sees it.
inline constexpr size_t SCAN_COPY_CAP_DWORDS = size_t {1} << 20;
static_assert(SCAN_COPY_CAP_DWORDS >= SCAN_DWORD_BUDGET, "the copy must hold what the walk may read");
/// The most command lists waiting for the scanner thread.
inline constexpr size_t SCAN_JOB_QUEUE_CAP = 256;
/// How many reset markers may wait beyond the capacity. Markers are never dropped, so they get room of
/// their own, and a scanner a few hundred flips behind is the only thing that fills it.
inline constexpr size_t SCAN_MARKER_RESERVE = 64;
/// The sizes of the preallocated copy buffers in dwords, which are 16 KB, 128 KB, 512 KB and 4 MB. A copy
/// takes the smallest that fits it.
inline constexpr std::array<size_t, 4> SCAN_BUFFER_CLASS_DWORDS = {size_t {1} << 12, size_t {1} << 15,
                                                                    size_t {1} << 17, size_t {1} << 20};
/// How many small buffers there are beyond one per queued job, for the lists being copied and the one
/// the scanner holds. A medium buffer is held for one in 8 jobs and a large one for one in 32, and the
/// few 4 MB lists a game submits are served by two.
inline constexpr size_t SCAN_SMALL_BUFFER_SLACK    = 32;
inline constexpr size_t SCAN_MEDIUM_BUFFER_DIVISOR = 8;
inline constexpr size_t SCAN_LARGE_BUFFER_DIVISOR  = 32;
inline constexpr size_t SCAN_HUGE_BUFFER_COUNT     = 2;
/// The longest copy time the histogram tells apart, in microseconds. A longer one is clamped to it.
inline constexpr double SCAN_COPY_US_CLAMP = 1.0e12;
/// How many buckets the copy time histogram has. Bucket n counts copies of fewer than 2^n microseconds
/// and at least half that, bucket 0 counts those under a microsecond, and the last takes all longer ones.
inline constexpr size_t SCAN_COPY_HISTOGRAM_BUCKETS = 24;

/// One preallocated copy buffer, which belongs to a size class of the queue that made it.
struct ScanBuffer {
	std::unique_ptr<uint32_t[]> data;
	uint32_t                    capacity   = 0;
	uint8_t                     size_class = 0;
	[[nodiscard]] explicit      operator bool() const { return data != nullptr; }
};

/// One thing for the scanner thread to do.
struct ScanJob {
	enum class Kind : uint8_t {
		/// Walk a copied command list.
		Commands,
		/// Reset the queue's shadow, which is what a flip suspend point does to the command processor.
		Reset,
	};

	Kind     kind     = Kind::Commands;
	uint32_t queue_id = 0;
	/// The number the queue gave the job in its queue's order, from 1. 0 until it is pushed.
	uint64_t seq = 0;
	/// How many dwords of the list the buffer holds.
	uint32_t dwords = 0;
	/// The list was longer than the copy, and its tail was not kept.
	bool truncated = false;
	/// Whether the list was copied, which is false when the queue was stopped or no buffer was free.
	bool copied = false;
	/// Set by Pop when the shadow of the queue can no longer be trusted, because a list was dropped or
	/// skipped as late and the scanner has not yet passed it. The consumer marks the shadow uncertain first.
	bool                                  shadow_uncertain = false;
	ScanBuffer                            buffer;
	std::chrono::steady_clock::time_point arrival;

	/// The copied list.
	[[nodiscard]] std::span<const uint32_t> CommandSpan() const { return {buffer.data.get(), dwords}; }
};

/// What the queue counted.
struct ScanJobQueueStats {
	uint64_t pushed = 0;
	/// Command lists that were dropped, for any reason.
	uint64_t dropped = 0;
	/// The part of `dropped` that was lost because no buffer was free to copy it into.
	uint64_t dropped_no_buffer = 0;
	/// Command lists the scanner skipped because the GPU thread had already begun them.
	uint64_t late_skipped = 0;
	/// Reset markers that found the queue full of markers. It is 0 unless the scanner is badly behind.
	uint64_t markers_lost   = 0;
	uint64_t truncated      = 0;
	uint64_t copied_dwords  = 0;
	size_t   peak           = 0;
	uint64_t copy_us_max    = 0;
	/// The copy times, as the histogram SCAN_COPY_HISTOGRAM_BUCKETS describes.
	std::array<uint64_t, SCAN_COPY_HISTOGRAM_BUCKETS> copy_us_histogram {};
};

class ScanJobQueue {
public:
	/// Builds the queue and every buffer it will ever use, which is where all of its allocation happens.
	explicit ScanJobQueue(size_t capacity = SCAN_JOB_QUEUE_CAP, size_t copy_cap_dwords = SCAN_COPY_CAP_DWORDS)
	    : m_capacity(capacity), m_copy_cap(copy_cap_dwords), m_ring(capacity + SCAN_MARKER_RESERVE) {
		const std::array<size_t, SCAN_BUFFER_CLASS_DWORDS.size()> counts = {
		    capacity + SCAN_SMALL_BUFFER_SLACK, std::max<size_t>(1, capacity / SCAN_MEDIUM_BUFFER_DIVISOR),
		    std::max<size_t>(1, capacity / SCAN_LARGE_BUFFER_DIVISOR), SCAN_HUGE_BUFFER_COUNT};
		for (size_t size_class = 0; size_class < counts.size(); ++size_class) {
			m_free[size_class].reserve(counts[size_class]);
			for (size_t index = 0; index < counts[size_class]; ++index) {
				ScanBuffer buffer;
				buffer.data       = std::make_unique_for_overwrite<uint32_t[]>(SCAN_BUFFER_CLASS_DWORDS[size_class]);
				buffer.capacity   = static_cast<uint32_t>(SCAN_BUFFER_CLASS_DWORDS[size_class]);
				buffer.size_class = static_cast<uint8_t>(size_class);
				m_free[size_class].push_back(std::move(buffer));
			}
		}
	}

	/// Copies a command list into a preallocated buffer, keeping at most the copy cap. The copy reads only
	/// the list it is given, which the caller guarantees is readable, and nothing else. A stopped queue is
	/// seen before anything is copied. When no buffer is free the job comes back with `copied` false, and
	/// pushing it counts a drop. When the best fitting size is gone a larger one is used, and failing that
	/// the largest smaller one, which cuts the list and counts as truncated.
	[[nodiscard]] ScanJob MakeCommands(uint32_t queue_id, std::span<const uint32_t> commands) {
		ScanJob job;
		job.kind     = ScanJob::Kind::Commands;
		job.queue_id = queue_id;
		job.arrival  = std::chrono::steady_clock::now();
		if (m_stopped.load(std::memory_order_acquire)) {
			return job;
		}
		const size_t wanted = std::min(commands.size(), m_copy_cap);
		if (wanted != 0) {
			{
				std::scoped_lock lock(m_mutex);
				job.buffer = TakeBuffer(wanted);
			}
			if (!job.buffer) {
				return job;
			}
		}
		const size_t kept = std::min<size_t>(wanted, job.buffer.capacity);
		job.truncated     = kept < commands.size();
		job.dwords        = static_cast<uint32_t>(kept);
		job.copied        = true;
		if (kept != 0) {
			std::memcpy(job.buffer.data.get(), commands.data(), kept * sizeof(uint32_t));
		}
		return job;
	}

	/// A marker that resets the queue's shadow when the scanner reaches it.
	[[nodiscard]] static ScanJob MakeReset(uint32_t queue_id) {
		ScanJob job;
		job.kind     = ScanJob::Kind::Reset;
		job.queue_id = queue_id;
		job.copied   = true;
		job.arrival  = std::chrono::steady_clock::now();
		return job;
	}

	/// Queues a job and returns the sequence number it got in its queue, or 0 when the queue is stopped.
	/// When the queue is full the oldest command list is dropped and counted, preferring one the GPU
	/// thread has already begun, and a marker is never dropped. `copy_us` is what the copy cost the caller.
	/// Every drop leaves the queue's shadow in doubt.
	uint64_t Push(ScanJob job, double copy_us = 0.0) {
		uint64_t seq = 0;
		{
			std::scoped_lock lock(m_mutex);
			if (m_stopped.load(std::memory_order_relaxed)) {
				return 0;
			}
			auto& queue = m_queues[QueueSlot(job.queue_id)];
			seq         = ++queue.next_seq;
			job.seq     = seq;
			m_stats.pushed++;
			if (job.kind == ScanJob::Kind::Commands && job.copied) {
				NoteCopy(job, copy_us);
			}
			if (!Admit(std::move(job))) {
				return seq;
			}
			m_stats.peak = std::max(m_stats.peak, m_count);
		}
		m_ready.notify_one();
		return seq;
	}

	/// Waits for the next job the scanner should walk. A list the GPU thread began before the scanner
	/// reached it is skipped and counted, with the doubt it leaves. The buffer of the job `out` still
	/// holds from the last call goes back to the pool. False once the queue is stopped.
	[[nodiscard]] bool Pop(ScanJob& out) {
		std::unique_lock lock(m_mutex);
		ReleaseBuffer(std::move(out.buffer));
		for (;;) {
			m_ready.wait(lock, [this] { return m_stopped.load(std::memory_order_relaxed) || m_count != 0; });
			if (m_stopped.load(std::memory_order_relaxed)) {
				return false;
			}
			if (TakeNext(out)) {
				return true;
			}
		}
	}

	/// Like Pop, but returns false at once when there is nothing to hand out, which is when the queue is
	/// empty or every job left was skipped as late. For tests and for a caller that must not wait.
	[[nodiscard]] bool TryPop(ScanJob& out) {
		std::scoped_lock lock(m_mutex);
		ReleaseBuffer(std::move(out.buffer));
		return !m_stopped.load(std::memory_order_relaxed) && TakeNext(out);
	}

	/// Gives a finished job's buffer back to the pool.
	void Recycle(ScanJob&& job) {
		std::scoped_lock lock(m_mutex);
		ReleaseBuffer(std::move(job.buffer));
	}

	/// Says that the GPU thread began the queue's job `seq`, so the scanner may skip it and every earlier
	/// one. It only moves forward, takes no lock and is safe from any thread.
	void NoteBegan(uint32_t queue_id, uint64_t seq) {
		auto& watermark = m_queues[QueueSlot(queue_id)].watermark;
		auto  seen      = watermark.load(std::memory_order_relaxed);
		while (seen < seq && !watermark.compare_exchange_weak(seen, seq, std::memory_order_release,
		                                                      std::memory_order_relaxed)) {
		}
	}

	/// The highest job number of the queue the GPU thread has begun.
	[[nodiscard]] uint64_t Watermark(uint32_t queue_id) const {
		return m_queues[QueueSlot(queue_id)].watermark.load(std::memory_order_acquire);
	}

	/// Wakes the scanner for good. Jobs still queued are dropped.
	void Stop() {
		{
			std::scoped_lock lock(m_mutex);
			m_stopped.store(true, std::memory_order_release);
			for (size_t index = 0; index < m_count; ++index) {
				ReleaseBuffer(std::move(At(index).buffer));
			}
			m_head  = 0;
			m_count = 0;
		}
		m_ready.notify_all();
	}

	/// A copy of what the queue has counted so far.
	[[nodiscard]] ScanJobQueueStats Stats() const {
		std::scoped_lock lock(m_mutex);
		return m_stats;
	}

	/// How many jobs are waiting, late ones included.
	[[nodiscard]] size_t Size() const {
		std::scoped_lock lock(m_mutex);
		return m_count;
	}

private:
	// What the queue keeps for each queue id. It is padded to a cache line, since the GPU thread writes
	// the watermark of one queue while submitting threads write the others.
	struct alignas(64) QueueState {
		uint64_t              next_seq = 0;
		std::atomic<uint64_t> watermark {0};
		bool                  shadow_in_doubt = false;
		// The highest number of a list or marker that was dropped or skipped, so that a reset marker
		// knows whether the doubt it leaves behind is older than itself.
		uint64_t doubt_seq = 0;
	};

	// The slot that keeps a queue id's state, which is the id itself except past the last slot.
	static size_t QueueSlot(uint32_t queue_id) { return std::min<size_t>(queue_id, SCAN_MAX_QUEUES - 1); }

	// The job at a position counted from the oldest. The caller holds the lock.
	ScanJob& At(size_t position) { return m_ring[(m_head + position) % m_ring.size()]; }

	// Records one copy in the totals and the histogram. The caller holds the lock.
	void NoteCopy(const ScanJob& job, double copy_us) {
		m_stats.copied_dwords += job.dwords;
		m_stats.truncated += job.truncated ? 1 : 0;
		const auto micros = static_cast<uint64_t>(std::clamp(copy_us, 0.0, SCAN_COPY_US_CLAMP));
		const auto bucket = std::min<size_t>(std::bit_width(micros), SCAN_COPY_HISTOGRAM_BUCKETS - 1);
		m_stats.copy_us_histogram[bucket]++;
		m_stats.copy_us_max = std::max(m_stats.copy_us_max, micros);
	}

	// Puts the queue's shadow in doubt because of the job with this number. The caller holds the lock.
	void NoteDoubt(const ScanJob& job) {
		auto& queue           = m_queues[QueueSlot(job.queue_id)];
		queue.shadow_in_doubt = true;
		queue.doubt_seq       = std::max(queue.doubt_seq, job.seq);
	}

	// Takes the next job that should be handed out from the front, skipping late lists, and false when
	// none is left. A reset marker clears the doubt that comes from jobs older than itself, since the
	// reset puts the shadow right, and leaves the doubt of newer ones. A list does the same, so the doubt
	// lasts until the scanner is past the job that caused it. The caller holds the lock.
	bool TakeNext(ScanJob& out) {
		while (m_count != 0) {
			ScanJob job = std::move(m_ring[m_head]);
			m_head      = (m_head + 1) % m_ring.size();
			m_count--;
			auto& queue = m_queues[QueueSlot(job.queue_id)];
			if (job.kind == ScanJob::Kind::Reset) {
				if (job.seq > queue.doubt_seq) {
					queue.shadow_in_doubt = false;
				}
				out = std::move(job);
				return true;
			}
			if (job.seq <= queue.watermark.load(std::memory_order_acquire)) {
				m_stats.late_skipped++;
				NoteDoubt(job);
				ReleaseBuffer(std::move(job.buffer));
				continue;
			}
			// A list older than the job that caused the doubt carries it too but does not clear it, since a
			// program it writes is still followed by the missing job. Only a list newer than every doubt
			// clears it.
			job.shadow_uncertain = queue.shadow_in_doubt;
			if (job.seq > queue.doubt_seq) {
				queue.shadow_in_doubt = false;
			}
			out = std::move(job);
			return true;
		}
		return false;
	}

	// Takes a free buffer for a list of `dwords`, trying the best fit and the larger sizes, and then the
	// smaller ones from the largest down. Empty when every size is out. The caller holds the lock.
	ScanBuffer TakeBuffer(size_t dwords) {
		size_t best = 0;
		while (best + 1 < SCAN_BUFFER_CLASS_DWORDS.size() && SCAN_BUFFER_CLASS_DWORDS[best] < dwords) {
			best++;
		}
		for (size_t size_class = best; size_class < m_free.size(); ++size_class) {
			if (!m_free[size_class].empty()) {
				return PopFree(size_class);
			}
		}
		for (size_t size_class = best; size_class-- > 0;) {
			if (!m_free[size_class].empty()) {
				return PopFree(size_class);
			}
		}
		return {};
	}

	// Takes the last free buffer of a size class, which must have one. The caller holds the lock.
	ScanBuffer PopFree(size_t size_class) {
		ScanBuffer buffer = std::move(m_free[size_class].back());
		m_free[size_class].pop_back();
		return buffer;
	}

	// Gives a buffer back to the free list of its size. It never allocates, since a list was reserved for
	// every buffer the queue made. The caller holds the lock.
	void ReleaseBuffer(ScanBuffer&& buffer) {
		if (buffer) {
			m_free[buffer.size_class].push_back(std::move(buffer));
		}
	}

	// Where in the queue the job to drop is, counted from the oldest, or the count when there is none. The
	// oldest command list that the GPU thread has begun comes first, since the scanner would skip it
	// anyway, and then the oldest of all. Markers are never chosen. The caller holds the lock.
	size_t FindVictim() {
		size_t oldest = m_count;
		for (size_t position = 0; position < m_count; ++position) {
			const ScanJob& job = At(position);
			if (job.kind != ScanJob::Kind::Commands) {
				continue;
			}
			if (job.seq <= m_queues[QueueSlot(job.queue_id)].watermark.load(std::memory_order_acquire)) {
				return position;
			}
			oldest = std::min(oldest, position);
		}
		return oldest;
	}

	// Counts a dropped command list, gives its buffer back and puts the queue's shadow in doubt.
	void NoteDrop(ScanJob&& job) {
		m_stats.dropped++;
		NoteDoubt(job);
		ReleaseBuffer(std::move(job.buffer));
	}

	// Takes the job at a position out, keeping the order of the rest. The caller holds the lock.
	ScanJob RemoveAt(size_t position) {
		ScanJob removed = std::move(At(position));
		for (size_t index = position; index + 1 < m_count; ++index) {
			At(index) = std::move(At(index + 1));
		}
		m_count--;
		return removed;
	}

	// Puts the job in the queue, dropping what it must to make room. False when it was not queued, which
	// is when it is itself the list to drop or it is a marker with no room at all. A list evicts the
	// oldest list once the queue holds the capacity, while a marker waits in the reserve and evicts a list
	// only when the reserve is full too. The caller holds the lock.
	bool Admit(ScanJob&& job) {
		const bool is_commands = job.kind == ScanJob::Kind::Commands;
		if (is_commands && !job.copied) {
			m_stats.dropped_no_buffer++;
			NoteDrop(std::move(job));
			return false;
		}
		if (m_count >= m_capacity && (is_commands || m_count == m_ring.size())) {
			const size_t victim = FindVictim();
			if (victim < m_count) {
				NoteDrop(RemoveAt(victim));
			} else if (is_commands) {
				// Nothing but markers waits, so the new list is the one that goes.
				NoteDrop(std::move(job));
				return false;
			}
		}
		if (m_count == m_ring.size()) {
			// A marker with no room in the reserve either. Nothing can be dropped to make it, so the
			// shadow is put in doubt in its place and the loss is counted.
			m_stats.markers_lost++;
			NoteDoubt(job);
			return false;
		}
		At(m_count) = std::move(job);
		m_count++;
		return true;
	}

	mutable std::mutex                                                   m_mutex;
	std::condition_variable                                              m_ready;
	size_t                                                               m_capacity;
	size_t                                                               m_copy_cap;
	// A ring with room for the capacity and the marker reserve, which never grows.
	std::vector<ScanJob>                                                 m_ring;
	size_t                                                               m_head  = 0;
	size_t                                                               m_count = 0;
	std::array<std::vector<ScanBuffer>, SCAN_BUFFER_CLASS_DWORDS.size()> m_free;
	std::array<QueueState, SCAN_MAX_QUEUES>                              m_queues;
	ScanJobQueueStats                                                    m_stats;
	std::atomic<bool>                                                    m_stopped {false};
};

/// Runs one job against its queue's shadow the way the scanner thread does. A reset marker resets the
/// shadow. A list is walked after the shadow takes the doubt the queue left it, and a list that was
/// copied short leaves the shadow in doubt after the walk, since the part that was cut may have changed
/// the program. A caller that keeps its own ScanScratch passes it, and the walk then uses it in place of
/// the calling thread's own. A graphics shadow, when given, takes the same resets and doubts as the
/// compute one, and needs the caller's scratch.
inline void RunScanJob(const SubmitScanner& scanner, const ScanJob& job, bool async_queue, CsShadow& shadow,
                       std::vector<ScannedDispatch>& dispatches, ScanStats& stats,
                       DispatchCache* cache = nullptr, ScanScratch* scratch = nullptr,
                       GraphicsScan* graphics = nullptr) {
	if (job.kind == ScanJob::Kind::Reset) {
		shadow.Reset();
		if (graphics != nullptr) {
			graphics->Reset();
		}
		return;
	}
	if (job.shadow_uncertain) {
		shadow.MarkUncertain();
		if (graphics != nullptr) {
			graphics->MarkUncertain();
		}
	}
	if (scratch != nullptr) {
		scanner.Scan(job.CommandSpan(), job.queue_id, async_queue, shadow, dispatches, stats, *scratch, cache,
		             graphics);
	} else {
		scanner.Scan(job.CommandSpan(), job.queue_id, async_queue, shadow, dispatches, stats, cache);
	}
	if (job.truncated) {
		shadow.MarkUncertain();
		if (graphics != nullptr) {
			graphics->MarkUncertain();
		}
	}
}

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_JOB_QUEUE_H
