#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/memoryTracker.h"
#include "graphics/host_gpu/rangeSet.h"
#include "graphics/host_gpu/renderer/cache/faultManager.h"
#include "graphics/host_gpu/renderer/cache/hotReadbackTable.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/recentGpuWrites.h"
#include "graphics/host_gpu/renderer/cache/streamBuffer.h"

#include <array>
#include <atomic>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
class CommandScheduler;
class TextureCache;

using BufferId = Common::SlotId;
inline constexpr BufferId NULL_BUFFER_ID {0};

class BufferCache {
public:
	static constexpr uint32_t CACHING_PAGEBITS  = 14;
	static constexpr uint64_t CACHING_PAGESIZE  = uint64_t {1} << CACHING_PAGEBITS;
	static constexpr uint64_t CACHING_NUMPAGES  = (LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemorySize) >> CACHING_PAGEBITS;
	static constexpr uint64_t BDA_PAGETABLE_SIZE =
	    CACHING_NUMPAGES * sizeof(vk::DeviceAddress);
	// A clean buffer that no binding has used for this many GC ticks (one per suspend point,
	// usually once a frame) is released even without memory pressure. Releases run every
	// STALE_RELEASE_INTERVAL_TICKS, at most STALE_RELEASE_MAX_PER_PASS at a time, so a scene
	// change spreads its releases out.
	static constexpr uint64_t STALE_RELEASE_AGE_TICKS      = 300;
	static constexpr uint64_t STALE_RELEASE_INTERVAL_TICKS = 60;
	static constexpr uint32_t STALE_RELEASE_MAX_PER_PASS   = 128;
	// Released ranges remembered to spot a shader reading one again. Older ones are forgotten,
	// and a read of those is still handled, just not counted or protected.
	static constexpr size_t RECENT_RELEASE_CAPACITY = 16384;
	// Refaults logged to the console as well, so they show up with the overlay closed.
	static constexpr uint64_t REFAULT_CONSOLE_LOG_LIMIT = 16;

	static constexpr uint64_t PageIndex(uint64_t address) {
		return (address < LOWER_ADDRESS_SIZE
		            ? address
		            : address - LibKernel::Memory::kExtendedMemoryBase + LOWER_ADDRESS_SIZE) >>
		       CACHING_PAGEBITS;
	}
	static constexpr uint64_t GuestAddress(uint64_t offset) {
		return offset < LOWER_ADDRESS_SIZE
		           ? offset
		           : offset - LOWER_ADDRESS_SIZE + LibKernel::Memory::kExtendedMemoryBase;
	}

	BufferCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	            TextureCache& texture_cache);
	~BufferCache();
	KYTY_CLASS_NO_COPY(BufferCache);

	void                   InvalidateMemory(uint64_t vaddr, uint64_t size);
	// Publishes GPU-written bytes of a range about to lose its host view, without marking pages
	// CPU-dirty. Registered buffers and their tracker state are otherwise left as they are.
	void                   FlushGpuDirtyForSuspend(uint64_t vaddr, uint64_t size);
	void                   ReadMemory(uint64_t vaddr, uint64_t size, bool is_write = false);
	[[nodiscard]] Buffer&  GetBuffer(BufferId id) { return m_slot_buffers[id]; }
	[[nodiscard]] BufferId FindBuffer(uint64_t vaddr, uint64_t size);
	/// Finds or creates the buffer for a range that shaders read through BDA without a
	/// descriptor binding, and counts that as a use so the garbage collector keeps it.
	/// FindBuffer alone is a lookup and does not.
	[[nodiscard]] BufferId FindBufferForShaderRead(uint64_t vaddr, uint64_t size);
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBuffer(uint64_t vaddr, uint64_t size,
	                                                        bool     is_written,
	                                                        bool     is_texel_buffer = false,
	                                                        BufferId id              = {});
	[[nodiscard]] StreamBuffer&                GetUtilityBuffer(MemoryUsage usage) noexcept {
		switch (usage) {
			case MemoryUsage::Upload: return m_staging_buffer;
			case MemoryUsage::Stream: return m_stream_buffer;
			case MemoryUsage::Download: return m_download_buffer;
			case MemoryUsage::DeviceLocal: return m_device_buffer;
		}
		EXIT("BufferCache: invalid utility-buffer usage\n");
	}
	[[nodiscard]] const Buffer* GetGdsBuffer() const noexcept { return &m_gds_buffer; }
	[[nodiscard]] Buffer* GetBdaPageTableBuffer() noexcept { return &m_bda_pagetable_buffer; }
	[[nodiscard]] Buffer* GetFaultBuffer() noexcept { return m_fault_manager.GetFaultBuffer(); }
	[[nodiscard]] std::pair<Buffer*, uint64_t> ObtainBufferForImage(uint64_t vaddr, uint64_t size);
	/// Fills GDS or guest memory with a dword. Ranges the GPU has not modified are written on the
	/// CPU after one bulk invalidation, others are filled on the GPU. GPU thread only.
	void FillBuffer(uint64_t vaddr, uint64_t size, uint32_t value, bool is_gds);
	/// Copies between GDS and guest memory. Memory to memory copies run on the CPU after one bulk
	/// invalidation of the destination when neither range is GPU modified and the source backs no
	/// cached image. Others are copied on the GPU. GPU thread only.
	void CopyBuffer(uint64_t dst_vaddr, uint64_t src_vaddr, uint64_t size, bool dst_gds,
	                bool src_gds);
	// Cache-index and exact dirty-range queries require GPU-thread serialization.
	[[nodiscard]] bool IsRegionRegistered(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool HasGpuDirtyBytes(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionCpuModified(uint64_t vaddr, uint64_t size);
	[[nodiscard]] bool IsRegionGpuModified(uint64_t vaddr, uint64_t size);
	// Color metadata the GPU found free of the clear keys in `keys` (bits of code >> 5) stays so
	// until a GPU write or download.
	void               MarkColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys);
	[[nodiscard]] bool IsColorMetadataChecked(uint64_t vaddr, uint64_t size, uint32_t keys) const;
	void               ProcessFaultBuffer();
	// Uploads the queued BDA pages inside `mapped` to the buffers that hold them.
	void SynchronizeBda(const RangeSet& mapped);
	// Queue after the range's state change becomes visible, or a drain could consume it first. A
	// deferred event (a write from any thread but the GPU thread) is held back until
	// PromoteBdaSync().
	void                   QueueBdaSync(uint64_t vaddr, uint64_t size, bool deferred = false);
	// Creates the buffer for a range a shader read through BDA while no buffer cached it. If the
	// range was released as stale, it is protected from stale release from then on.
	void RecreateFaultedRange(uint64_t vaddr, uint64_t size);
	// Drops release history and protection inside an unmapped range, which holds new data if it
	// is mapped again.
	void               ForgetReleasedRanges(uint64_t vaddr, uint64_t size);
	[[nodiscard]] RangeSet TakeBdaSync();
	// Makes game-thread writes visible to the next BDA synchronization.
	void               PromoteBdaSync();
	void               RunGarbageCollector();

private:
	/// Copies between overlapping ranges of one buffer on the GPU, with memmove's result.
	void CopyOverlapping(CommandBuffer& command, Buffer& buffer, uint64_t source_offset,
	                     uint64_t destination_offset, uint64_t size);
	friend struct BufferCacheTestAccess;

	bool IsBufferInvalid(BufferId id) const { return !id || m_slot_buffers[id].is_deleted; }

	using BufferMap = std::map<uint64_t, BufferId>;
	struct OverlapResult {
		BufferMap::iterator first;
		BufferMap::iterator last;
		uint64_t            begin;
		uint64_t            end;
		bool                has_stream_leap;
	};

	using PageTable = MultiLevelPageTable<BufferId, CACHING_PAGEBITS, 44, 20>;
	static_assert(CACHING_PAGESIZE == (uint64_t {1} << PageTable::kPageBits));
	void WriteDataBuffer(Buffer& buffer, uint64_t address, const void* source, uint64_t size);
	void TouchBuffer(const Buffer& buffer);
	[[nodiscard]] OverlapResult ResolveOverlaps(uint64_t vaddr, uint64_t size);
	void JoinOverlap(BufferId new_id, BufferId overlap_id, bool accumulate_stream_score);
	[[nodiscard]] BufferId CreateBuffer(uint64_t vaddr, uint64_t size);
	void                   Register(BufferId id);
	void Unregister(BufferId id);
	template <bool insert>
	void ChangeRegister(BufferId id);
	void DeleteBuffer(BufferId id);
	// Adds the CPU-modified pages to m_bda_upload for every buffer overlapping [begin, end).
	void CollectBdaCopies(uint64_t begin, uint64_t end);
	// Adds the CPU-modified pages of the provided `buffer` in [begin, end) to m_bda_upload.
	void CollectBdaBufferCopies(Buffer& buffer, uint64_t begin, uint64_t end);
	// Uploads everything in m_bda_upload, if anything, and empties it.
	void FlushBdaCopies();
	// Uploads everything in m_bda_upload through one staging allocation.
	void UploadBdaCopies();
	// Records the copies in m_bda_upload from `staging`, between one pair of global barriers.
	void RecordBdaCopies(vk::Buffer staging);

	// Whether a buffer may be released without a download, meaning the GPU wrote none of it.
	[[nodiscard]] bool IsStaleReleaseCandidate(const Buffer& buffer);
	// Releases clean buffers that no binding has used for STALE_RELEASE_AGE_TICKS.
	void                     ReleaseStaleBuffers(uint64_t tick);
	// Records a stale release, so a later fault on that range counts as a refault.
	void                     RememberReleasedRange(uint64_t begin, uint64_t end);
	[[nodiscard]] bool SynchronizeBuffer(Buffer& buffer, uint64_t vaddr, uint64_t size,
	                                     bool is_written, bool is_texel_buffer);
	[[nodiscard]] vk::Buffer UploadCopies(Buffer& buffer, std::span<vk::BufferCopy> copies,
	                                      uint64_t total_size);
	[[nodiscard]] bool SynchronizeBufferFromImage(Buffer& buffer, uint64_t vaddr, uint64_t size);
	// A download in four stages, so several buffers can share one GPU wait. Collect takes the
	// dirty ranges, Reserve picks the download memory, Record queues the copies, and Publish
	// writes the finished bytes to guest memory.
	struct PendingDownload {
		std::vector<vk::BufferCopy> copies;
		uint64_t                    total_size     = 0;
		uint64_t                    buffer_address = 0;
		uint64_t                    offset         = 0;
		uint8_t*                    mapped         = nullptr;
		// Owns the memory behind mapped when the stream buffer was too full to serve it. The
		// downloads of one batch share it, because the batch makes a single reservation.
		std::shared_ptr<Buffer> temporary;
	};
	[[nodiscard]] PendingDownload CollectDownloadCopies(const Buffer& buffer, uint64_t vaddr,
	                                                    uint64_t size);
	// A buffer's pending download inside a batch of downloads that share one GPU wait.
	struct BatchedDownload {
		Buffer*         buffer = nullptr;
		PendingDownload pending;
		// The range to mark clean once published. It must equal what was collected, because a
		// partial span must not clear GPU-dirty pages that were never downloaded. The main
		// entry leaves it empty, as its caller unmarks the main window.
		uint64_t unmark_address = 0;
		uint64_t unmark_size    = 0;
	};
	void ReserveDownloadMemory(std::span<PendingDownload* const> items);
	void RecordDownload(const Buffer& buffer, const PendingDownload& pending);
	void PublishDownload(const PendingDownload& pending);
	// Resolves a hot entry to the buffer that now owns it, without creating one on a miss.
	[[nodiscard]] BufferId FindHotBuffer(uint64_t address, uint64_t size);
	[[nodiscard]] uint64_t EstimateDownloadBytes(uint64_t address, uint64_t size) const;
	void CollectHotDownloads(BufferId main_id, uint64_t& budget, std::vector<BatchedDownload>& batch);
	// Adds the finished small GPU writes the recent write ring remembers to a batch, skipping
	// spans already in claimed and charging what it collects to budget.
	void CollectRecentWrites(RangeSet& claimed, uint64_t& budget,
	                         std::vector<BatchedDownload>& batch);
	// Collects one unclaimed piece of a recent write when a single buffer holds it and it fits.
	void CollectRecentPiece(uint64_t address, uint64_t size, uint64_t& budget,
	                        std::vector<BatchedDownload>& batch);
	[[nodiscard]] bool DownloadWithHotBuffers(BufferId main_id, Buffer& buffer, uint64_t vaddr,
	                                          uint64_t size);
	// Synchronous downloads publish before returning; asynchronous callers wait before reuse.
	template <bool async>
	[[nodiscard]] bool DownloadBufferMemory(Buffer& buffer, uint64_t vaddr, uint64_t size);
	void               ForgetColorMetadataChecks(uint64_t vaddr, uint64_t size);
	// Records a buffer that took a sync readback, whatever its cause. Does nothing without
	// --batch-hot-readbacks.
	void NoteHotReadback(uint64_t address, uint64_t size);
	// Records a GPU write so a later sync readback can download it once finished. Does nothing
	// without --batch-hot-readbacks. The tick is the one current when the buffer is bound, and a
	// submit during draw setup can land the write in the next tick, so treat it as a lower bound.
	void NoteRecentWrite(uint64_t address, uint64_t size);

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	FaultManager                                      m_fault_manager;
	Buffer                                            m_gds_buffer;
	Buffer                                            m_bda_pagetable_buffer;
	Common::SlotVector<Buffer>                        m_slot_buffers;
	Common::LeastRecentlyUsedCache<BufferId, uint64_t> m_lru_cache;
	BufferMap                                         m_buffers;
	PageTable                                         m_page_table;
	RangeSet                                          m_gpu_modified_ranges;
	// One per TileManager::ColorClearSlots key slot.
	std::array<RangeSet, 8>                           m_checked_color_metadata;
	MemoryTracker                                     m_memory_tracker;
	std::mutex                                        m_bda_sync_mutex;
	RangeSet                                          m_bda_sync;
	RangeSet                                          m_bda_sync_deferred;
	// Set while m_bda_sync_deferred may be non-empty. Allows PromoteBdaSync to skip the lock.
	std::atomic<bool>                                 m_has_deferred_bda_sync {false};
	StreamBuffer                                      m_staging_buffer;
	StreamBuffer                                      m_stream_buffer;
	StreamBuffer                                      m_download_buffer;
	StreamBuffer                                      m_device_buffer;
	TextureCache&                                     m_texture_cache;
	// The buffer copies that BDA synchronization collects and uploads together. Synchronization
	// runs before every BDA draw, so they are kept between calls to reuse the vectors' storage.
	// Only the GPU thread synchronizes BDA, so they need no lock.
	struct BdaUpload {
		// A set of BufferCopy entries placed back to back in `copies`, all writing into the same
		// buffer. Enables recording them with a single copy command.
		struct Batch {
			Buffer* buffer;
			size_t  first;
			size_t  count;
		};
		std::vector<vk::BufferCopy> copies;  // Staging offset, buffer offset and size of each.
		std::vector<uint64_t>       sources; // Guest address of each copy.
		std::vector<Batch>          batches;
		uint64_t                    size = 0; // Staging bytes the copies use.

		void Clear() {
			copies.clear();
			sources.clear();
			batches.clear();
			size = 0;
		}
	};
	BdaUpload                                         m_bda_upload;
	// Read once because the fault path is hot and the flag must not change while the cache lives.
	const bool                                        m_batch_hot_readbacks;
	HotReadbackTable                                  m_hot_readbacks;
	RecentGpuWriteRing                                m_recent_writes;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t m_trigger_gc_memory  = 1ull * 1024 * 1024 * 1024;
	uint64_t m_critical_gc_memory = 2ull * 1024 * 1024 * 1024;
	uint64_t m_gc_tick            = 0;
	struct ReleasedRange {
		uint64_t begin;
		uint64_t end;
	};
	// Stale releases, oldest first, so a shader reading one again can be spotted.
	std::deque<ReleasedRange> m_recent_releases;
	// Ranges a shader read again after a stale release. Stale release keeps them from then on.
	RangeSet m_release_protected;
	// Refaults logged so far, so only the first REFAULT_CONSOLE_LOG_LIMIT reach the console.
	uint64_t m_refaults_logged = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_BUFFERCACHE_H_
