#include "graphics/host_gpu/renderer/cache/bufferCache.h"

#include "common/alignment.h"
#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/commandScheduler.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace Libs::Graphics {

namespace {

constexpr uint64_t MiB           = 1024 * 1024;
constexpr uint64_t GdsBufferSize = 64 * 1024;
// GPU-modified ranges come as many small scattered islands, so a readback widens to a window.
constexpr uint64_t DownloadWindowSize = 512 * 1024;
// Staging offset alignment for BDA buffer copies, the same as ordinary buffer uploads use.
constexpr uint64_t StagingCopyAlignment = 4;
// Packed download ranges start on their own cache line, as in shadPS4.
constexpr uint64_t DownloadCopyAlignment = 64;

std::pair<Buffer*, uint64_t> UploadMemory(GraphicContext& graphics, CommandScheduler& scheduler,
                                        StreamBuffer& staging, uint64_t size, uint64_t alignment,
                                        vk::BufferUsageFlags usage, const auto& write) {
	const auto [mapped, offset] = staging.Map(size, alignment);
	if (mapped != nullptr) {
		write(mapped, offset);
		staging.Commit();
		return {&staging, offset};
	}

	// Large transfers use the same one-time host allocation as ordinary buffer uploads.
	auto temporary = std::make_unique<Buffer>(graphics, scheduler, MemoryUsage::Upload, 0,
	                                         usage, size);
	write(temporary->Mapped().data(), 0);
	temporary->Flush(0, size);
	auto* buffer = temporary.get();
	scheduler.DeferOperation([owner = std::move(temporary)]() mutable { owner.reset(); });
	return {buffer, 0};
}

/// Calls visit(address, size) for each part of the range that claimed does not cover, in address
/// order.
template <class Visit>
void ForEachUnclaimed(const RangeSet& claimed, uint64_t address, uint64_t size, Visit&& visit) {
	const uint64_t end    = address + size;
	uint64_t       cursor = address;
	claimed.ForEachInRange(address, size, [&](uint64_t begin, uint64_t last) {
		if (begin > cursor) {
			visit(cursor, begin - cursor);
		}
		cursor = last;
	});
	if (cursor < end) {
		visit(cursor, end - cursor);
	}
}

/// Returns the tracker pages a batch already downloads. The main window is rounded out to whole
/// pages, because its caller unmarks it only after the batch and a recent write that shares one of
/// its pages must not collect that page a second time.
template <class Batch>
RangeSet ClaimedDownloadPages(const Batch& batch, uint64_t vaddr, uint64_t size) {
	RangeSet   claimed;
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	claimed.Add(begin, Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE) - begin);
	for (size_t index = 1; index < batch.size(); ++index) {
		claimed.Add(batch[index].unmark_address, batch[index].unmark_size);
	}
	return claimed;
}

} // namespace

void BufferCache::WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source,
                                  uint64_t size) {
	auto* bytes = static_cast<const uint8_t*>(source);
	while (size != 0) {
		const auto chunk  = std::min(size, m_staging_buffer.Size());
		// Copy waits for ring reuse and flushes the fresh host data before submission.
		const auto offset = m_staging_buffer.Copy(bytes, chunk, 4);
		const auto destination_offset = buffer.Offset(address);
		EXIT_IF(destination_offset > buffer.Size() || chunk > buffer.Size() - destination_offset);
		m_scheduler.EndRendering();
		const auto command = m_scheduler.Current().Handle();
		vk::BufferMemoryBarrier2 before {};
		before.srcStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		before.srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		before.dstStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		before.dstAccessMask = vk::AccessFlagBits2::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer        = buffer.Handle();
		before.offset        = destination_offset;
		before.size          = chunk;
		vk::DependencyInfo dependency {};
		dependency.dependencyFlags          = vk::DependencyFlagBits::eByRegion;
		dependency.bufferMemoryBarrierCount = 1;
		dependency.pBufferMemoryBarriers    = &before;
		command.pipelineBarrier2(dependency);
		const vk::BufferCopy copy {offset, destination_offset, chunk};
		command.copyBuffer(m_staging_buffer.Handle(), buffer.Handle(), 1, &copy);
		auto after          = before;
		after.srcStageMask  = vk::PipelineStageFlagBits2::eTransfer;
		after.srcAccessMask = vk::AccessFlagBits2::eTransferWrite;
		after.dstStageMask  = vk::PipelineStageFlagBits2::eAllCommands;
		after.dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite;
		dependency.pBufferMemoryBarriers = &after;
		command.pipelineBarrier2(dependency);
		bytes += chunk;
		address += chunk;
		size -= chunk;
	}
}

void BufferCache::Register(BufferId id) {
	ChangeRegister<true>(id);
}

void BufferCache::Unregister(BufferId id) {
	ChangeRegister<false>(id);
}

template <bool insert>
void BufferCache::ChangeRegister(BufferId id) {
	auto& buffer = m_slot_buffers[id];
	PageTable::PageRange pages {};
	EXIT_IF(!(GuestRange {buffer.CpuAddress(), buffer.Size()}.Valid()) ||
	        !PageTable::TryGetPageRange(buffer.CpuAddress(), buffer.Size(), pages));
	for (size_t page = pages.first; page < pages.last_exclusive; ++page) {
		if constexpr (insert) {
			m_page_table[page] = id;
		} else {
			m_page_table[page] = {};
		}
	}
	const auto size_pages = pages.last_exclusive - pages.first;
	const auto table_offset = PageIndex(buffer.CpuAddress()) * sizeof(vk::DeviceAddress);
	if constexpr (insert) {
		m_buffers.emplace(buffer.CpuAddress(), id);
		m_total_used_memory += buffer.Size();
		buffer.lru_id = m_lru_cache.Insert(id, m_gc_tick);
		std::vector<vk::DeviceAddress> addresses;
		addresses.reserve(size_pages);
		for (uint64_t i = 0; i < size_pages; ++i) {
			addresses.push_back(buffer.BufferDeviceAddress() + (i << CACHING_PAGEBITS));
		}
		WriteDataBuffer(m_bda_pagetable_buffer, table_offset,
		                addresses.data(), addresses.size() * sizeof(vk::DeviceAddress));
		QueueBdaSync(buffer.CpuAddress(), buffer.Size());
	} else {
		m_buffers.erase(m_buffers.find(buffer.CpuAddress()));
		m_total_used_memory -= buffer.Size();
		m_lru_cache.Free(buffer.lru_id);
		m_bda_pagetable_buffer.Fill(table_offset,
		                            size_pages * sizeof(vk::DeviceAddress), 0);
		buffer.is_deleted = true;
	}
}

void BufferCache::TouchBuffer(const Buffer& buffer) {
	m_lru_cache.Touch(buffer.lru_id, m_gc_tick);
}

void BufferCache::DeleteBuffer(BufferId id) {
	Unregister(id);
	if (m_scheduler.Active()) {
		m_scheduler.DeferOperation([this, id] { m_slot_buffers.erase(id); });
	} else {
		m_slot_buffers.erase(id);
	}
}

/// Takes the GPU-dirty ranges of a buffer window out of the tracking sets and lists the copies
/// that download them. The result has no copies when nothing in the window was GPU-dirty.
BufferCache::PendingDownload BufferCache::CollectDownloadCopies(const Buffer& buffer,
                                                                uint64_t vaddr, uint64_t size) {
	PendingDownload pending;
	pending.buffer_address = buffer.CpuAddress();
	m_memory_tracker.ForEachDownloadRange<false>(
	    vaddr, size, [&](uint64_t address, uint64_t bytes) noexcept {
		    m_memory_tracker.ValidateGpuDirtyPages(m_gpu_modified_ranges, address, bytes,
		                                           "buffer download");
		    m_gpu_modified_ranges.ForEachInRange(address, bytes, [&](uint64_t start, uint64_t end) {
			    pending.copies.emplace_back(start - pending.buffer_address, pending.total_size,
			                                end - start);
			    // Start each packed range on its own 64-byte cache line.
			    pending.total_size += Common::AlignUp(end - start, DownloadCopyAlignment);
		    });
		    m_gpu_modified_ranges.Subtract(address, bytes);
		    ForgetColorMetadataChecks(address, bytes);
	    });
	return pending;
}

/// Reserves download memory for all the pending copies at once, from the stream buffer or from a
/// temporary buffer that the pending downloads then share. One reservation for the whole batch
/// lets the ring wait as a single download would, which is safe because nothing of the batch is
/// recorded yet. Reserving per download could wrap onto an earlier download's unpublished bytes.
void BufferCache::ReserveDownloadMemory(std::span<PendingDownload* const> items) {
	uint64_t total_size = 0;
	for (const auto* item: items) {
		// A second reservation would orphan the first one's temporary buffer and offsets.
		EXIT_IF(item->mapped != nullptr);
		total_size += item->total_size;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferDownloadBytes, total_size);
	auto [mapped, offset] = m_download_buffer.Map(total_size, DownloadCopyAlignment);
	std::shared_ptr<Buffer> temporary;
	if (mapped == nullptr) {
		temporary = std::make_shared<Buffer>(m_graphics, m_scheduler, MemoryUsage::Download, 0,
		                                     vk::BufferUsageFlagBits::eTransferDst, total_size);
		mapped    = temporary->Mapped().data();
	} else {
		m_download_buffer.Commit();
	}
	// Each download is a slice of the reservation. Every total_size is already a multiple of
	// DownloadCopyAlignment, so each slice starts on its own cache line.
	uint64_t slice = 0;
	for (auto* item: items) {
		item->temporary = temporary;
		item->mapped    = mapped + slice;
		item->offset    = offset + slice;
		for (auto& copy: item->copies) {
			copy.dstOffset += item->offset;
		}
		slice += item->total_size;
	}
}

/// Records the reserved copies into the current command, wrapped in the barriers that make the
/// GPU writes visible to the transfer and the downloaded bytes visible to the host.
void BufferCache::RecordDownload(const Buffer& buffer, const PendingDownload& pending) {
	EXIT_IF(pending.mapped == nullptr);
	const auto& download = pending.temporary ? *pending.temporary : m_download_buffer;

	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto              native = command.Handle();
	vk::BufferMemoryBarrier before {};
	before.srcAccessMask       = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
	before.dstAccessMask       = vk::AccessFlagBits::eTransferRead;
	before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	before.buffer              = buffer.Handle();
	before.offset              = 0;
	before.size                = buffer.Size();
	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, {}, 0, nullptr, 1, &before, 0,
	                       nullptr);
	native.copyBuffer(buffer.Handle(), download.Handle(),
	                  static_cast<uint32_t>(pending.copies.size()), pending.copies.data());

	auto after          = before;
	after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after.dstAccessMask = vk::AccessFlagBits::eHostRead;
	after.buffer        = download.Handle();
	after.offset        = pending.offset;
	after.size          = pending.total_size;
	native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                       vk::PipelineStageFlagBits::eAllCommands |
	                           vk::PipelineStageFlagBits::eHost,
	                       {}, 0, nullptr, 1, &after, 0, nullptr);
}

/// Copies finished download bytes into guest memory. The GPU must have completed the copies.
void BufferCache::PublishDownload(const PendingDownload& pending) {
	(pending.temporary ? *pending.temporary : m_download_buffer)
	    .Invalidate(pending.offset, pending.total_size);
	for (const auto& copy: pending.copies) {
		Libs::LibKernel::Memory::WriteBacking(pending.buffer_address + copy.srcOffset,
		                                      pending.mapped + (copy.dstOffset - pending.offset),
		                                      copy.size);
	}
}

template <bool async>
bool BufferCache::DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size) {
	auto pending = CollectDownloadCopies(buffer, vaddr, size);
	if (pending.copies.empty()) {
		return false;
	}
	PendingDownload* const items[] = {&pending};
	ReserveDownloadMemory(items);
	RecordDownload(buffer, pending);
	if constexpr (async) {
		m_scheduler.DeferPriorityOperation(
		    [this, pending = std::move(pending)] { PublishDownload(pending); });
	} else {
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
		PublishDownload(pending);
	}
	return true;
}

/// Finds the buffer that owns a hot entry. The page table is read directly because FindBuffer
/// would create a buffer on a miss, and a hot entry may outlive its buffer.
BufferId BufferCache::FindHotBuffer(uint64_t address, uint64_t size) {
	const auto* owner = m_page_table.Find(address >> PageTable::kPageBits);
	if (owner == nullptr || IsBufferInvalid(*owner) ||
	    !m_slot_buffers[*owner].IsInBounds(address, size)) {
		return {};
	}
	return *owner;
}

/// Returns an upper bound on the bytes that downloading a range would add to the ring. It counts
/// every GPU-written range inside it, which is at least what the page tracker lets through.
uint64_t BufferCache::EstimateDownloadBytes(uint64_t address, uint64_t size) const {
	uint64_t bytes = 0;
	m_gpu_modified_ranges.ForEachInRange(address, size, [&](uint64_t start, uint64_t end) {
		bytes += Common::AlignUp(end - start, DownloadCopyAlignment);
	});
	return bytes;
}

/// Adds the GPU-written bytes of recently read-back buffers to a batch, so the wait that the main
/// download needs anyway also clears their next fault. A buffer is skipped when it is the main
/// buffer, was already added, no longer exists, or would cost more than what is left of budget.
/// The main-buffer skip is needed for correctness. The main buffer is nearly always hot, and
/// collecting it again would revisit window pages the caller has not unmarked yet, which breaks the
/// debug invariant in ValidateGpuDirtyPages and downloads bytes outside the main window. The
/// duplicate skip only saves a repeated estimate, since the first collect already removed the
/// buffer's ranges.
void BufferCache::CollectHotDownloads(BufferId main_id, uint64_t& budget,
                                      std::vector<BatchedDownload>& batch) {
	std::vector<BufferId> seen;
	seen.reserve(MaxHotReadbackEntries + 1);
	seen.push_back(main_id);
	m_hot_readbacks.ForEachHot(m_scheduler.CurrentTick(), [&](uint64_t address, uint64_t size) {
		// Merged buffers can leave two entries that resolve to the same buffer.
		const auto id = FindHotBuffer(address, size);
		if (!id || std::ranges::find(seen, id) != seen.end()) {
			return;
		}
		seen.push_back(id);
		auto&      hot      = m_slot_buffers[id];
		const auto estimate = EstimateDownloadBytes(hot.CpuAddress(), hot.Size());
		if (estimate == 0 || estimate > budget) {
			return;
		}
		const auto hot_address = hot.CpuAddress();
		const auto hot_size    = hot.Size();
		auto       pending     = CollectDownloadCopies(hot, hot_address, hot_size);
		if (pending.copies.empty()) {
			return;
		}
		budget -= std::min(budget, pending.total_size);
		batch.push_back({&hot, std::move(pending), hot_address, hot_size});
	});
}

/// Walks the recent write ring newest first and collects each finished span that is not claimed
/// yet. Every visited span is claimed afterwards, so an older entry under it is clipped to what it
/// adds. Pending spans are claimed without being collected, which keeps an older finished write
/// that the GPU is rewriting out of the download. Claims are whole tracker pages, so a pending
/// write also holds back finished bytes that share its pages. That only leaves them for their own
/// fault. The walk ends once the budget is spent.
void BufferCache::CollectRecentWrites(RangeSet& claimed, uint64_t& budget,
                                      std::vector<BatchedDownload>& batch) {
	auto& master = m_scheduler.GetMasterSemaphore();
	master.Refresh();
	const auto known_tick = master.KnownGpuTick();
	if (known_tick == 0) {
		return;
	}
	// An entry carries the tick that was current when its buffer was bound, and a submit during
	// draw setup can put the write itself in the next tick. Only ticks before the known one are
	// treated as finished, so the entry's own write is done. That says nothing about a later
	// write the ring never records, such as a large buffer or an image, that is bound but not yet
	// recorded. Like the hot buffers, this relies on no readback running between such a bind and
	// its recording.
	m_recent_writes.ForEachNewest(known_tick - 1, [&](uint64_t address, uint64_t size,
	                                                  bool finished) {
		if (finished) {
			ForEachUnclaimed(claimed, address, size, [&](uint64_t piece, uint64_t piece_size) {
				CollectRecentPiece(piece, piece_size, budget, batch);
			});
		}
		claimed.Add(address, size);
		return budget > 0;
	});
}

/// Collects the GPU-written bytes of one unclaimed piece of a recent write and counts it for the
/// overlay. A piece is skipped when no single buffer holds it or when it would cost more than
/// budget.
void BufferCache::CollectRecentPiece(uint64_t address, uint64_t size, uint64_t& budget,
                                     std::vector<BatchedDownload>& batch) {
	const auto id = FindHotBuffer(address, size);
	if (!id) {
		return;
	}
	const auto estimate = EstimateDownloadBytes(address, size);
	if (estimate == 0 || estimate > budget) {
		return;
	}
	auto& buffer  = m_slot_buffers[id];
	auto  pending = CollectDownloadCopies(buffer, address, size);
	if (pending.copies.empty()) {
		return;
	}
	budget -= std::min(budget, pending.total_size);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferCoDownloads);
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferCoDownloadBytes,
	                           pending.total_size);
	batch.push_back({&buffer, std::move(pending), address, size});
}

/// Downloads the main window, the GPU-written bytes of hot buffers and the finished recent small
/// writes with one GPU wait. The extras are marked clean here, and the caller does that for the
/// main window. Nothing is marked CPU-modified, since only a faulting write does that. Returns
/// false when the window had nothing.
bool BufferCache::DownloadWithHotBuffers(BufferId main_id, Buffer& buffer, uint64_t vaddr,
                                         uint64_t size) {
	auto main_pending = CollectDownloadCopies(buffer, vaddr, size);
	if (main_pending.copies.empty()) {
		return false;
	}
	std::vector<BatchedDownload> batch;
	batch.reserve(MaxHotReadbackEntries + 1);
	batch.push_back({&buffer, std::move(main_pending)});
	uint64_t budget = MaxHotReadbackBytes;
	CollectHotDownloads(main_id, budget, batch);
	auto claimed = ClaimedDownloadPages(batch, vaddr, size);
	CollectRecentWrites(claimed, budget, batch);
	// The pointers are taken once the batch has stopped growing, so none can be invalidated.
	std::vector<PendingDownload*> pendings;
	pendings.reserve(batch.size());
	for (auto& item: batch) {
		pendings.push_back(&item.pending);
	}
	ReserveDownloadMemory(pendings);
	for (const auto& item: batch) {
		RecordDownload(*item.buffer, item.pending);
	}
	const auto tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(tick);
	m_scheduler.WaitPriorityOperations(tick);
	for (const auto& item: batch) {
		PublishDownload(item.pending);
	}
	for (size_t index = 1; index < batch.size(); ++index) {
		m_memory_tracker.UnmarkRegionAsGpuModified(batch[index].unmark_address,
		                                           batch[index].unmark_size);
	}
	return true;
}

BufferCache::BufferCache(GraphicContext& graphics, CommandScheduler& scheduler,
                         PageManager& page_manager, TextureCache& texture_cache)
    : m_graphics(graphics), m_scheduler(scheduler), m_fault_manager(graphics, scheduler, *this),
      m_gds_buffer(graphics, scheduler, MemoryUsage::Stream, 0, AllFlags, GdsBufferSize),
      m_bda_pagetable_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 0, AllFlags,
                             BDA_PAGETABLE_SIZE),
      m_memory_tracker(page_manager),
      m_staging_buffer(graphics, scheduler, MemoryUsage::Upload, 512 * MiB),
      m_stream_buffer(graphics, scheduler, MemoryUsage::Stream, 64 * MiB),
      m_download_buffer(graphics, scheduler, MemoryUsage::Download, 64 * MiB),
      m_device_buffer(graphics, scheduler, MemoryUsage::DeviceLocal, 128 * MiB),
      m_texture_cache(texture_cache), m_batch_hot_readbacks(Config::BatchHotReadbacksEnabled()) {
	std::memset(m_gds_buffer.Mapped().data(), 0, static_cast<size_t>(m_gds_buffer.Size()));
	m_gds_buffer.Flush(0, m_gds_buffer.Size());
	SetVulkanObjectNameF(m_graphics.device, m_bda_pagetable_buffer.Handle(),
	                     "BDA Page Table Buffer");
	m_scheduler.BeginCommand();
	for (auto* buffer: {&m_bda_pagetable_buffer, m_fault_manager.GetFaultBuffer()}) {
		buffer->Fill(0, buffer->Size(), 0);
	}
	const auto null_id =
	    m_slot_buffers.insert(m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0, AllFlags, 16);
	EXIT_IF(null_id != NULL_BUFFER_ID);
	SetVulkanObjectNameF(m_graphics.device, GetBuffer(null_id).Handle(), "Kyty.NullBuffer");
	if (!m_graphics.CanReportMemoryUsage()) {
		return;
	}
	constexpr int64_t GiB              = 1024ll * 1024 * 1024;
	constexpr int64_t target_threshold = 8 * GiB;
	const auto        budget =
	    static_cast<int64_t>(std::min<uint64_t>(m_graphics.GetTotalMemoryBudget(), INT64_MAX));
	const auto threshold = std::min(budget, target_threshold);
	const auto expected  = std::min(budget - 6 * threshold / 10, budget - GiB);
	const auto critical  = std::min(budget - 2 * threshold / 10, budget - GiB / 2);
	m_trigger_gc_memory  = static_cast<uint64_t>(std::max<int64_t>(expected, GiB));
	m_critical_gc_memory = static_cast<uint64_t>(std::max<int64_t>(critical, 2 * GiB));
}

BufferCache::~BufferCache() {
	if (!m_gpu_modified_ranges.Empty()) {
		EXIT("BufferCache: destroyed with pending GPU-modified ranges\n");
	}
	for (const auto& [vaddr, id]: m_buffers) {
		(void)vaddr;
		const auto& buffer = m_slot_buffers[id];
		if (m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size())) {
			EXIT("BufferCache: destroyed with GPU-modified buffer\n");
		}
	}
}

void BufferCache::InvalidateMemory(uint64_t vaddr, uint64_t size) {
	m_memory_tracker.InvalidateRegion(vaddr, size,
	                                  [this, vaddr, size] { ReadMemory(vaddr, size, true); });
	// Writes from threads other than the GPU thread (game threads, file I/O and so on) land before
	// the work that reads them is submitted, or before a wait the command processor is blocked on
	// is released. So their BDA synchronization events can wait until the command processor starts
	// or resumes a submission. Synchronizing them at every BDA draw instead uploads data meant for
	// later frames again and again.
	QueueBdaSync(vaddr, size, !GuestGpu::IsGpuThread());
}

// Downloads the GPU-dirty bytes of every buffer overlapping the range, then clears their
// ownership. Pages are not marked CPU-dirty, so the cached data stays valid for a later resume.
void BufferCache::FlushGpuDirtyForSuspend(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid suspend range\n");
	}
	// Guest releases are cache-page aligned, so anything else means the caller got the range wrong.
	if ((vaddr | size) % CACHING_PAGESIZE != 0) {
		EXIT("BufferCache: suspend range is not cache-page aligned\n");
	}
	const auto end    = vaddr + size;
	bool       queued = false;
	auto       it     = m_buffers.upper_bound(vaddr);
	if (it != m_buffers.begin()) {
		--it;
	}
	// Walk the registered buffers only. FindBuffer would create or merge buffers over the range.
	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer = m_slot_buffers[it->second];
		const auto start  = std::max(buffer.CpuAddress(), vaddr);
		const auto finish = std::min(buffer.CpuAddress() + buffer.Size(), end);
		if (start < finish) {
			queued |= DownloadBufferMemory<true>(buffer, start, finish - start);
		}
	}
	if (queued) {
		// Publish every queued download before the host view goes away, as the garbage collector does.
		const auto tick = m_scheduler.CurrentTick();
		m_scheduler.Wait(tick);
		m_scheduler.WaitPriorityOperations(tick);
	}
	m_memory_tracker.UnmarkRegionAsGpuModified(vaddr, size);
	if (m_gpu_modified_ranges.Intersects(vaddr, size)) {
		EXIT("BufferCache: suspend retained GPU-dirty bytes\n");
	}
}

void BufferCache::NoteHotReadback(uint64_t address, uint64_t size) {
	if (m_batch_hot_readbacks) {
		m_hot_readbacks.Note(address, size, m_scheduler.CurrentTick());
	}
}

void BufferCache::NoteRecentWrite(uint64_t address, uint64_t size) {
	if (m_batch_hot_readbacks) {
		m_recent_writes.Note(address, size, m_scheduler.CurrentTick());
	}
}

void BufferCache::ReadMemory(uint64_t vaddr, uint64_t size, bool is_write) {
	if (!GuestGpu::IsGpuThread() && CommandScheduler::InDeferredOperation()) {
		EXIT("unsupported buffer readback from an asynchronous GPU completion, "
		     "addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
		     vaddr, size);
	}
	m_scheduler.Context().GetGpu().SendCommandSync([this, vaddr, size, is_write] {
		if (is_write && !IsRegionRegistered(vaddr, size)) {
			return;
		}
		const auto buffer_id = FindBuffer(vaddr, size);
		auto&      buffer    = m_slot_buffers[buffer_id];
		if (IsRegionGpuModified(vaddr, size)) {
			const auto start =
			    std::max(Common::AlignDown(vaddr, DownloadWindowSize), buffer.CpuAddress());
			const auto end = std::min(std::max(start + DownloadWindowSize, vaddr + size),
			                          buffer.CpuAddress() + buffer.Size());
			const bool downloaded =
			    m_batch_hot_readbacks
			        ? DownloadWithHotBuffers(buffer_id, buffer, start, end - start)
			        : DownloadBufferMemory<false>(buffer, start, end - start);
			if (downloaded) {
				m_memory_tracker.UnmarkRegionAsGpuModified(start, end - start);
				NoteHotReadback(buffer.CpuAddress(), buffer.Size());
			}
		}
		if (is_write) {
			m_memory_tracker.MarkRegionAsCpuModified(vaddr, size);
			QueueBdaSync(vaddr, size);
		}
	});
}

BufferId BufferCache::FindBuffer(uint64_t vaddr, uint64_t size) {
	if (vaddr == 0) {
		return NULL_BUFFER_ID;
	}
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid buffer discovery request\n");
	}
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			return *owner;
		}
	}
	return CreateBuffer(vaddr, size);
}

BufferId BufferCache::FindBufferForShaderRead(uint64_t vaddr, uint64_t size) {
	const auto id = FindBuffer(vaddr, size);
	if (!IsBufferInvalid(id)) {
		TouchBuffer(m_slot_buffers[id]);
	}
	return id;
}

BufferCache::OverlapResult BufferCache::ResolveOverlaps(uint64_t vaddr, uint64_t size) {
	static constexpr int      StreamLeapThreshold = 16;
	static constexpr uint64_t StreamLeapSize      = CACHING_PAGESIZE * 128;

	auto       begin      = vaddr;
	auto       end        = vaddr + size;
	const auto find_first = [&](uint64_t address) {
		auto first = m_buffers.lower_bound(address);
		if (first != m_buffers.begin()) {
			const auto  previous = std::prev(first);
			const auto& buffer   = m_slot_buffers[previous->second];
			if (buffer.CpuAddress() + buffer.Size() > address) {
				first = previous;
			}
		}
		return first;
	};
	auto first           = find_first(begin);
	auto last            = first;
	int  stream_score    = 0;
	bool has_stream_leap = false;
	for (; last != m_buffers.end() && last->first < end; ++last) {
		const auto& buffer        = m_slot_buffers[last->second];
		const auto  buffer_begin  = buffer.CpuAddress();
		const auto  buffer_end    = buffer_begin + buffer.Size();
		const bool  expands_left  = buffer_begin < begin;
		const bool  expands_right = buffer_end > end;
		begin                     = std::min(begin, buffer_begin);
		end                       = std::max(end, buffer_end);
		if (!has_stream_leap && (stream_score += buffer.StreamScore()) > StreamLeapThreshold) {
			has_stream_leap = true;
			// Reserve space in the incoming stream's direction of growth.
			// The old buffer extending left of the request predicts growth to the right, and vice versa.
			if (expands_left) {
				end += std::min(StreamLeapSize, (vaddr < LOWER_ADDRESS_SIZE ? LOWER_ADDRESS_SIZE
				                                       : LibKernel::Memory::kExtendedMemoryBase +
				                                             LibKernel::Memory::kExtendedMemorySize) - end);
			}
			if (expands_right) {
				const auto minimum = vaddr < LOWER_ADDRESS_SIZE
				                         ? CACHING_PAGESIZE * 2
				                         : LibKernel::Memory::kExtendedMemoryBase;
				if (begin > minimum) {
					begin -= std::min(StreamLeapSize, begin - minimum);
				}
				first = find_first(begin);
				begin = std::min(begin, first->first);
			}
		}
	}
	return {first, last, begin, end, has_stream_leap};
}

void BufferCache::JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score) {
	auto& new_buffer = m_slot_buffers[new_id];
	auto& overlap    = m_slot_buffers[overlap_id];
	if (accumulate_stream_score) {
		new_buffer.IncreaseStreamScore(overlap.StreamScore() + 1);
	}
	new_buffer.CopyFrom(m_scheduler.Current(), overlap, 0,
	                    overlap.CpuAddress() - new_buffer.CpuAddress(), overlap.Size());
	DeleteBuffer(overlap_id);
}

BufferId BufferCache::CreateBuffer(uint64_t vaddr, uint64_t size) {
	const auto end = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	vaddr = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	size               = end - vaddr;
	const auto overlap = ResolveOverlaps(vaddr, size);

	const auto id = m_slot_buffers.insert(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, overlap.begin,
	    AllFlags | vk::BufferUsageFlagBits::eShaderDeviceAddress, overlap.end - overlap.begin);
	const auto& buffer = m_slot_buffers[id];
	SetVulkanObjectNameF(m_graphics.device, buffer.Handle(),
	                     "Kyty.GameBuffer[guest=0x{:016x} size=0x{:x}]", overlap.begin,
	                     overlap.end - overlap.begin);
	for (auto it = overlap.first; it != overlap.last;) {
		const auto old_id = (it++)->second;
		JoinOverlap(id, old_id, !overlap.has_stream_leap);
	}
	Register(id);
	return id;
}

bool BufferCache::SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size, bool is_written,
                                    bool is_texel_buffer) {
	std::vector<vk::BufferCopy> copies;
	uint64_t                    total_size = 0;
	vk::Buffer                  source;
	m_memory_tracker.ForEachUploadRange(
	    vaddr, size, is_written,
	    [&](uint64_t address, uint64_t bytes) noexcept {
		    copies.emplace_back(total_size, buffer.Offset(address), bytes);
		    total_size += bytes;
	    },
	    [&]() noexcept { source = UploadCopies(buffer, copies, total_size); });
	if (source) {
		auto& command = m_scheduler.Current();
		command.EndRendering();
		const auto native = command.Handle();
		vk::BufferMemoryBarrier before {};
		before.srcAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
		                       vk::AccessFlagBits::eTransferRead |
		                       vk::AccessFlagBits::eTransferWrite;
		before.dstAccessMask       = vk::AccessFlagBits::eTransferWrite;
		before.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		before.buffer              = buffer.Handle();
		before.offset              = 0;
		before.size                = buffer.Size();
		native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
		                       vk::PipelineStageFlagBits::eTransfer,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &before, 0, nullptr);
		native.copyBuffer(source, buffer.Handle(), static_cast<uint32_t>(copies.size()),
		                  copies.data());
		auto after          = before;
		after.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
		after.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;
		native.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
		                       vk::PipelineStageFlagBits::eAllCommands,
		                       vk::DependencyFlagBits::eByRegion, 0, nullptr, 1, &after, 0, nullptr);
	}
	if (is_texel_buffer && !is_written) {
		return SynchronizeBufferFromImage(buffer, vaddr, size);
	}
	return false;
}

vk::Buffer BufferCache::UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
                                     uint64_t total_size) {
	if (copies.empty()) {
		return nullptr;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, total_size);

	const auto source = UploadMemory(m_graphics, m_scheduler, m_staging_buffer, total_size, 4,
	    vk::BufferUsageFlagBits::eTransferSrc, [&](uint8_t* mapped, uint64_t base_offset) {
		for (auto& copy: copies) {
			const auto address = buffer.CpuAddress() + copy.dstOffset;
			if (!LibKernel::Memory::CopySparseMemory(address, mapped + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			copy.srcOffset += base_offset;
		}
	});
	return source.first->Handle();
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBuffer(uint64_t vaddr, uint64_t size,
                                                       bool is_written, bool is_texel_buffer,
                                                       BufferId id) {
	EXIT_IF(!(GuestRange {vaddr, size}.Valid()));

	if (!is_written && size <= CACHING_PAGESIZE &&
	    !m_memory_tracker.IsRegionGpuModified(vaddr, size) &&
	    m_memory_tracker.IsRegionCpuModified(vaddr, size)) {
		const auto alignment = std::max<uint64_t>(
		    m_graphics.physical_device_properties.limits.minUniformBufferOffsetAlignment, 1);
		auto [mapped, offset] = m_stream_buffer.Map(size, alignment, false);
		if (mapped != nullptr) {
			if (!LibKernel::Memory::CopySparseMemory(vaddr, mapped, size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			m_stream_buffer.Commit();
			return {&m_stream_buffer, offset};
		}
	}

	if (IsBufferInvalid(id) || !m_slot_buffers[id].IsInBounds(vaddr, size)) {
		id = FindBuffer(vaddr, size);
	}
	auto& buffer = m_slot_buffers[id];
	TouchBuffer(buffer);
	(void)SynchronizeBuffer(buffer, vaddr, size, is_written, is_texel_buffer);
	if (is_written) {
		m_gpu_modified_ranges.Add(vaddr, size);
		ForgetColorMetadataChecks(vaddr, size);
		NoteRecentWrite(vaddr, size);
	}
	return {&buffer, buffer.Offset(vaddr)};
}

std::pair<Buffer*, uint64_t> BufferCache::ObtainBufferForImage(uint64_t vaddr, uint64_t size) {
	const auto* owner = m_page_table.Find(vaddr >> PageTable::kPageBits);
	if (owner != nullptr && *owner) {
		auto& buffer = m_slot_buffers[*owner];
		if (buffer.IsInBounds(vaddr, size)) {
			TouchBuffer(buffer);
			(void)SynchronizeBuffer(buffer, vaddr, size, false, false);
			return {&buffer, buffer.Offset(vaddr)};
		}
	}
	if (IsRegionGpuModified(vaddr, size)) {
		return ObtainBuffer(vaddr, size, false, false);
	}

	return UploadMemory(m_graphics, m_scheduler, m_staging_buffer, size, 16,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eStorageBuffer,
	    [&](uint8_t* mapped, uint64_t) {
		if (!Libs::LibKernel::Memory::TryReadSparseBacking(vaddr, mapped, size)) {
			EXIT("BufferCache: failed to read mapped guest image backing\n");
		}
	});
}

void BufferCache::FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds) {
	if ((vaddr & 3u) != 0 || size == 0 || (size & 3u) != 0 || size > UINT64_MAX - vaddr) {
		EXIT("BufferCache: fill range must be dword aligned\n");
	}
	if (is_gds) {
		if (vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - vaddr) {
			EXIT("BufferCache: GDS fill range is out of bounds\n");
		}
		m_gds_buffer.Fill(vaddr, size, value);
		return;
	}
	if (vaddr == 0) {
		EXIT("BufferCache: invalid fill memory address\n");
	}
	(void)m_texture_cache.ClearMeta(vaddr);
	if (!IsRegionGpuModified(vaddr, size)) {
		// Invalidate cached buffers, images and image heaps for the whole range in one call.
		// Letting the write fault page by page costs one fault per page on the GPU thread.
		// Images end up as those faults would have left them on a fully watched range.
		// If the range is not fully mapped, fill anyway so the write faults as it did before.
		(void)m_scheduler.Context().InvalidateSequentialWrite(vaddr, size);
		auto* destination = reinterpret_cast<uint32_t*>(vaddr);
		std::fill(destination, destination + size / sizeof(uint32_t), value);
		return;
	}

	m_texture_cache.InvalidateMemoryFromGPU(vaddr, size);
	auto [dst, dst_offset] = ObtainBuffer(vaddr, size, true, true);
	dst->Fill(dst_offset, size, value);
}

void BufferCache::CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
                             bool src_gds) {
	const bool dst_memory = !dst_gds;
	const bool src_memory = !src_gds;
	if ((dst_memory && dst_vaddr == 0) || (src_memory && src_vaddr == 0) || size == 0 ||
	    ((dst_gds || src_gds) && ((dst_vaddr | src_vaddr | size) & 3u) != 0) ||
	    size > UINT64_MAX - dst_vaddr || size > UINT64_MAX - src_vaddr || (dst_gds && src_gds) ||
	    (dst_gds && (dst_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - dst_vaddr)) ||
	    (src_gds && (src_vaddr > m_gds_buffer.Size() || size > m_gds_buffer.Size() - src_vaddr))) {
		EXIT("BufferCache: invalid copy range, src=0x%016" PRIx64 " dst=0x%016" PRIx64
		     " size=0x%016" PRIx64 " src_gds=%d dst_gds=%d\n",
		     src_vaddr, dst_vaddr, size, static_cast<int>(src_gds), static_cast<int>(dst_gds));
	}
	if (src_memory && dst_memory && !IsRegionGpuModified(dst_vaddr, size) &&
	    !IsRegionGpuModified(src_vaddr, size) && !m_texture_cache.FindImageFromRange(src_vaddr, size)) {
		// Invalidate the destination in one call rather than faulting on each page. The source
		// is not GPU modified so reading it cannot fault. An unmapped range falls through to a
		// write that faults as before. The copy's write order is unspecified, so images are
		// invalidated as if the copy ran front to back like a fill, one rule on every platform.
		(void)m_scheduler.Context().InvalidateSequentialWrite(dst_vaddr, size);
		// A guest copy may overlap itself, where memcpy is undefined. memmove copies as if through a
		// temporary buffer, which the GPU path below does as well.
		std::memmove(reinterpret_cast<void*>(dst_vaddr), reinterpret_cast<const void*>(src_vaddr),
		             size);
		return;
	}

	auto& command = m_scheduler.Current();
	if (dst_memory) {
		m_texture_cache.InvalidateMemoryFromGPU(dst_vaddr, size);
	}
	const auto src_id      = src_memory ? FindBuffer(src_vaddr, size) : BufferId {};
	const auto dst_id      = dst_memory ? FindBuffer(dst_vaddr, size) : BufferId {};
	auto [src, src_offset] = src_memory ? ObtainBuffer(src_vaddr, size, false, true, src_id)
	                                    : std::pair {&m_gds_buffer, src_vaddr};
	auto [dst, dst_offset] = dst_memory ? ObtainBuffer(dst_vaddr, size, true, true, dst_id)
	                                    : std::pair {&m_gds_buffer, dst_vaddr};
	// Cached buffers never overlap, so overlapping guest ranges always share one buffer.
	if (src == dst && src_offset < dst_offset + size && dst_offset < src_offset + size) {
		// A copy onto itself changes no bytes, and ObtainBuffer has already marked the range.
		if (src_offset != dst_offset) {
			CopyOverlapping(command, *dst, src_offset, dst_offset, size);
		}
		return;
	}
	dst->CopyFrom(command, *src, src_offset, dst_offset, size);
}

void BufferCache::CopyOverlapping(CommandBuffer& command, Buffer& buffer, uint64_t source_offset,
                                  uint64_t destination_offset, uint64_t size) {
	// Vulkan forbids overlapping regions in one copy, so the bytes go through a scratch buffer.
	// Each CopyFrom places barriers around its copy, which orders the two copies.
	auto scratch = std::make_unique<Buffer>(
	    m_graphics, m_scheduler, MemoryUsage::DeviceLocal, 0,
	    vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, size);
	scratch->CopyFrom(command, buffer, source_offset, 0, size);
	buffer.CopyFrom(command, *scratch, 0, destination_offset, size);
	// The copies run later on the GPU, so the scratch buffer is freed once they have finished.
	m_scheduler.DeferOperation([owner = std::move(scratch)]() mutable { owner.reset(); });
}

bool BufferCache::IsRegionRegistered(uint64_t vaddr, uint64_t size) {
	if (!GuestRange {vaddr, size}.Valid()) {
		EXIT("BufferCache: invalid registered-region query\n");
	}
	// Cached buffers are ordered and non-overlapping. The last buffer beginning before the query
	// end is therefore the only possible intersection.
	const auto candidate = m_buffers.lower_bound(vaddr + size);
	if (candidate == m_buffers.begin()) {
		return false;
	}
	const auto& [address, id] = *std::prev(candidate);
	return address + m_slot_buffers[id].Size() > vaddr;
}

void BufferCache::MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0) {
			m_checked_color_metadata[slot].Add(vaddr, size);
		}
	}
}

bool BufferCache::IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const {
	for (uint32_t slot = 0; slot < m_checked_color_metadata.size(); slot++) {
		if ((keys & (1u << slot)) != 0 && !m_checked_color_metadata[slot].Contains(vaddr, size)) {
			return false;
		}
	}
	return true;
}

void BufferCache::ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size) {
	for (auto& checked: m_checked_color_metadata) {
		checked.Subtract(vaddr, size);
	}
}

bool BufferCache::IsRegionGpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionGpuModified(vaddr, size);
}

bool BufferCache::HasGpuDirtyBytes(uint64_t vaddr, uint64_t size) {
	return m_gpu_modified_ranges.Intersects(vaddr, size);
}

bool BufferCache::IsRegionCpuModified(uint64_t vaddr, uint64_t size) {
	return m_memory_tracker.IsRegionCpuModified(vaddr, size);
}

bool BufferCache::IsStaleReleaseCandidate(const Buffer& buffer) {
	return !m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size()) &&
	       !m_gpu_modified_ranges.Intersects(buffer.CpuAddress(), buffer.Size()) &&
	       !m_release_protected.Intersects(buffer.CpuAddress(), buffer.Size());
}

void BufferCache::RememberReleasedRange(uint64_t begin, uint64_t end) {
	m_recent_releases.push_back({begin, end});
	if (m_recent_releases.size() > RECENT_RELEASE_CAPACITY) {
		m_recent_releases.pop_front();
	}
}

void BufferCache::RecreateFaultedRange(uint64_t vaddr, uint64_t size) {
	// A shader read zeros here, from a range stale release dropped. It is read only through BDA,
	// where no binding marks it as used, so it would only be released again.
	const auto end       = vaddr + size;
	uint64_t   refaulted = 0;
	std::erase_if(m_recent_releases, [&](const ReleasedRange& range) {
		if (range.end <= vaddr || range.begin >= end) {
			return false;
		}
		m_release_protected.Add(range.begin, range.end - range.begin);
		refaulted++;
		return true;
	});
	if (refaulted != 0) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferReleaseRefaults,
		                           refaulted);
		const auto message = fmt::format(
		    "GPU: a shader read a released buffer at 0x{:016x}, keeping that range cached\n",
		    vaddr);
		if (m_refaults_logged++ < REFAULT_CONSOLE_LOG_LIMIT) {
			Log::WriteToConsoleAndLog(message);
		} else {
			LOGF("%s", message.c_str());
		}
	}
	(void)FindBuffer(vaddr, size);
}

void BufferCache::ForgetReleasedRanges(uint64_t vaddr, uint64_t size) {
	const auto begin = Common::AlignDown(vaddr, CACHING_PAGESIZE);
	const auto end   = Common::AlignUp(vaddr + size, CACHING_PAGESIZE);
	m_release_protected.Subtract(begin, end - begin);
	std::erase_if(m_recent_releases, [&](const ReleasedRange& range) {
		return range.begin < end && range.end > begin;
	});
}

// A cached buffer keeps its pages write-protected. Once the game reuses that memory for other
// data, every CPU write there faults and re-uploads into a buffer nothing reads any more.
void BufferCache::ReleaseStaleBuffers(uint64_t tick) {
	if (tick < STALE_RELEASE_AGE_TICKS || tick % STALE_RELEASE_INTERVAL_TICKS != 0) {
		return;
	}
	KYTY_PROFILER_BLOCK("BufferCache::ReleaseStaleBuffers");
	// Dropping a clean buffer without a download is safe only because shaders never write guest
	// memory through BDA (ShaderInfoCollection rejects them), so every GPU write is tracked.
	// Nothing in this loop may wait on the scheduler. A wait can run deferred operations that
	// create or join buffers and change the LRU list while it is being walked.
	uint32_t released = 0;
	m_lru_cache.ForEachItemBelow(tick - STALE_RELEASE_AGE_TICKS, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		EXIT_IF(buffer.is_deleted);
		if (!IsStaleReleaseCandidate(buffer)) {
			return false;
		}
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		RememberReleasedRange(buffer.CpuAddress(), buffer.CpuAddress() + buffer.Size());
		DeleteBuffer(id);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferReleases);
		return ++released == STALE_RELEASE_MAX_PER_PASS;
	});
}

void BufferCache::RunGarbageCollector() {
	const auto tick = m_gc_tick++;
	ReleaseStaleBuffers(tick);
	if (m_graphics.CanReportMemoryUsage()) {
		m_total_used_memory = m_graphics.GetDeviceMemoryUsage();
	}
	if (m_total_used_memory < m_trigger_gc_memory) {
		return;
	}

	const bool     aggressive = m_total_used_memory >= m_critical_gc_memory;
	const uint64_t age        = std::min<uint64_t>(aggressive ? 80 : 160, tick);
	const size_t   limit      = aggressive ? 64 : 32;

	std::vector<BufferId> dirty_buffers;
	size_t                retire_count = 0;
	m_lru_cache.ForEachItemBelow(tick - age, [&](BufferId id) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.ValidateGpuDirtyOwnership(m_gpu_modified_ranges, buffer.CpuAddress(),
		                                           buffer.Size(), "garbage collection");
		const bool dirty = m_memory_tracker.IsRegionGpuModified(buffer.CpuAddress(), buffer.Size());
		if (dirty && !aggressive) {
			return false;
		}
		if (dirty) {
			EXIT_NOT_IMPLEMENTED(!DownloadBufferMemory<true>(buffer, buffer.CpuAddress(), buffer.Size()));
			dirty_buffers.push_back(id);
		} else {
			m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
			DeleteBuffer(id);
			Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferEvictions);
		}
		return ++retire_count == limit;
	});
	if (dirty_buffers.empty()) {
		return;
	}

	// Publish all queued downloads before releasing their tracked pages and owners.
	const auto completion_tick = m_scheduler.CurrentTick();
	m_scheduler.Wait(completion_tick);
	m_scheduler.WaitPriorityOperations(completion_tick);
	for (const auto id: dirty_buffers) {
		auto& buffer = m_slot_buffers[id];
		m_memory_tracker.UnmarkRegionAsGpuModified(buffer.CpuAddress(), buffer.Size());
		m_memory_tracker.MarkRegionAsCpuModified(buffer.CpuAddress(), buffer.Size());
		Unregister(id);
		m_slot_buffers.erase(id);
	}
}

void BufferCache::ProcessFaultBuffer() {
	m_fault_manager.ProcessFaultBuffer();
}

// Queues the pages of [vaddr, vaddr + size) for the next BDA synchronization. A deferred event
// waits in its own set until PromoteBdaSync.
void BufferCache::QueueBdaSync(uint64_t vaddr, uint64_t size, bool deferred) {
	const auto begin = Common::AlignDown(vaddr, TRACKER_PAGE_SIZE);
	const auto end   = Common::AlignUp(vaddr + size, TRACKER_PAGE_SIZE);
	std::lock_guard lock(m_bda_sync_mutex);
	if (deferred) {
		m_bda_sync_deferred.Add(begin, end - begin);
		m_has_deferred_bda_sync.store(true, std::memory_order_release);
	} else {
		m_bda_sync.Add(begin, end - begin);
	}
}

// Moves the deferred BDA events into the set TakeBdaSync() returns.
void BufferCache::PromoteBdaSync() {
	// The command processor calls this at every submission slice and every passing wait. Most of
	// the time nothing was deferred since the last call, so we can skip the lock. A game thread
	// sets the flag before it submits or releases the wait, so its event will not be missed.
	if (!m_has_deferred_bda_sync.load(std::memory_order_acquire)) {
		return;
	}
	std::lock_guard lock(m_bda_sync_mutex);
	m_has_deferred_bda_sync.store(false, std::memory_order_relaxed);
	// The immediate set is typically empty here, since every BDA draw takes it. Then, instead of
	// moving the deferred set range by range, it is moved as a whole.
	if (m_bda_sync.Empty()) {
		std::swap(m_bda_sync, m_bda_sync_deferred);
	} else {
		m_bda_sync_deferred.ForEach(
		    [this](uint64_t begin, uint64_t end) { m_bda_sync.Add(begin, end - begin); });
	}
	m_bda_sync_deferred = {};
}

RangeSet BufferCache::TakeBdaSync() {
	std::lock_guard lock(m_bda_sync_mutex);
	return std::exchange(m_bda_sync, {});
}

void BufferCache::SynchronizeBda(const RangeSet& mapped) {
	// Take the queue before walking so writes queued during synchronization stay pending.
	const auto pending = TakeBdaSync();

	// Only walk ranges inside buffers since tracking state exists for those (a mapped range
	// can be huge). A draw can find a few hundred dirty buffers, so their copies are collected
	// first and then uploaded through a single staging allocation, between one pair of barriers,
	// instead of an allocation and two whole-buffer barriers per buffer.
	m_bda_upload.Clear();
	pending.ForEach([&](uint64_t begin, uint64_t end) {
		// Queued pages that are no longer mapped have nothing to upload.
		mapped.ForEachInRange(begin, end - begin, [&](uint64_t start, uint64_t finish) {
			CollectBdaCopies(start, finish);
		});
	});

	FlushBdaCopies();
}

// Uploads everything in m_bda_upload, if anything, and empties it.
void BufferCache::FlushBdaCopies() {
	// Nothing left to upload. Skip the staging allocation and the barriers.
	if (m_bda_upload.batches.empty()) {
		return;
	}
	UploadBdaCopies();
	m_bda_upload.Clear();
}

// Adds the CPU-modified pages to m_bda_upload for every buffer overlapping [begin, end).
void BufferCache::CollectBdaCopies(uint64_t begin, uint64_t end) {
	// Start from the last buffer that starts at or before `begin` as it may overlap.
	auto it = m_buffers.upper_bound(begin);

	if (it != m_buffers.begin()) {
		--it;
	}

	for (; it != m_buffers.end() && it->first < end; ++it) {
		auto&      buffer        = m_slot_buffers[it->second];
		const auto overlap_begin = std::max(buffer.CpuAddress(), begin);
		const auto overlap_end   = std::min(buffer.CpuAddress() + buffer.Size(), end);
		// That first buffer may end before `begin`.
		if (overlap_begin >= overlap_end) {
			continue;
		}
		CollectBdaBufferCopies(buffer, overlap_begin, overlap_end);
	}
}

// Adds the CPU-modified pages of the provided `buffer` in [begin, end) to m_bda_upload.
void BufferCache::CollectBdaBufferCopies(Buffer& buffer, uint64_t begin, uint64_t end) {
	// Past the staging ring's size, UploadMemory makes a one-off host allocation of the whole
	// upload. So if this range could push the upload past it, upload what was collected so far
	// first. The range size is an upper bound on what this buffer adds.
	if (m_bda_upload.size + (end - begin) > m_staging_buffer.Size()) {
		FlushBdaCopies();
	}
	auto&      upload = m_bda_upload;
	const auto first  = upload.copies.size();

	// Most queued pages were uploaded by an ordinary binding after they were queued. The tracker
	// only reports the runs of pages that are still CPU-modified, and marks them clean and
	// write-protected as it reports them. Each run goes into the staging allocation after the
	// previous one.
	const auto add_copy = [&](uint64_t address, uint64_t size) noexcept {
		upload.copies.emplace_back(upload.size, buffer.Offset(address), size);
		upload.sources.push_back(address);
		upload.size += size;
	};
	m_memory_tracker.ForEachUploadRange(begin, end - begin, false, add_copy, []() noexcept {});

	const auto added = upload.copies.size() - first;
	// An ordinary binding already uploaded every page queued here.
	if (added == 0) {
		return;
	}

	// Copies into the same buffer that directly follow the previous batch are joined to it and
	// share one copy command.
	if (!upload.batches.empty()) {
		auto& last = upload.batches.back();
		if (last.buffer == &buffer && last.first + last.count == first) {
			last.count += added;
			return;
		}
	}
	upload.batches.push_back({&buffer, first, added});
}

// Uploads everything in m_bda_upload through one staging allocation.
void BufferCache::UploadBdaCopies() {
	auto& upload = m_bda_upload;
	Common::DebugCounters::Add(Common::DebugCounters::Counter::BufferUploadBytes, upload.size);

	// Read each run from guest memory into its place in the staging allocation. The pages were
	// write-protected while they were collected, so a write racing with this read faults and
	// queues its page for the next synchronization.
	const auto fill_staging = [&](uint8_t* staging_memory, uint64_t staging_offset) {
		for (size_t i = 0; i < upload.copies.size(); ++i) {
			auto& copy = upload.copies[i];
			if (!LibKernel::Memory::CopySparseMemory(upload.sources[i],
			                                         staging_memory + copy.srcOffset, copy.size)) {
				EXIT("BufferCache: failed to read sparse guest buffer memory\n");
			}
			// Offsets so far count from the start of this upload whereas the copy command needs
			// them from the start of the staging buffer.
			copy.srcOffset += staging_offset;
		}
	};
	const auto staging =
	    UploadMemory(m_graphics, m_scheduler, m_staging_buffer, upload.size, StagingCopyAlignment,
	                 vk::BufferUsageFlagBits::eTransferSrc, fill_staging)
	        .first->Handle();
	RecordBdaCopies(staging);
}

// Records the copies in m_bda_upload from `staging`, between one pair of global barriers.
void BufferCache::RecordBdaCopies(vk::Buffer staging) {
	const auto& upload = m_bda_upload;
	// Copies cannot be recorded inside a render pass.
	auto& command = m_scheduler.Current();
	command.EndRendering();
	const auto native = command.Handle();

	// Earlier GPU work may still read or write these buffers. Let it finish before the copies.
	vk::MemoryBarrier before_copies {};
	vk::AccessFlags   source_access_mask =
	    vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite |
	    vk::AccessFlagBits::eTransferRead | vk::AccessFlagBits::eTransferWrite;

	before_copies.srcAccessMask = source_access_mask;
	before_copies.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

	native.pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands,
	                       vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
	                       1, &before_copies, 0, nullptr, 0, nullptr);

	for (const auto& batch: upload.batches) {
		uint32_t              batch_count = static_cast<uint32_t>(batch.count);
		const vk::BufferCopy* buffer_copy = upload.copies.data() + batch.first;

		native.copyBuffer(staging, batch.buffer->Handle(), batch_count, buffer_copy);
	}

	// Make the new data visible to everything that runs after the copies.
	vk::MemoryBarrier after_copies {};
	after_copies.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	after_copies.dstAccessMask = vk::AccessFlagBits::eMemoryRead | vk::AccessFlagBits::eMemoryWrite;

	native.pipelineBarrier(
	    vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllCommands,
	    vk::DependencyFlagBits::eByRegion, 1, &after_copies, 0, nullptr, 0, nullptr);
}

} // namespace Libs::Graphics
