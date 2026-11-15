#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/lruCache.h"
#include "common/slotVector.h"
#include "graphics/host_gpu/pageManager.h"
#include "graphics/host_gpu/regionManager.h"
#include "graphics/host_gpu/renderer/cache/multiLevelPageTable.h"
#include "graphics/host_gpu/renderer/cache/readbackGuard.h"
#include "graphics/host_gpu/renderer/image/blitHelper.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/renderer/image/tiler.h"

#include <atomic>
#include <bitset>
#include <map>
#include <optional>
#include <span>
#include <unordered_set>
#include <vector>

namespace Common::FirstRunReport {
struct SkipTargetFacts;
} // namespace Common::FirstRunReport

namespace Libs::Graphics {

struct GraphicContext;
class Buffer;
class BufferCache;
class CommandBuffer;
class CommandScheduler;
class RenderExecutor;
class TextureCache;
struct TextureCacheTestAccess;

namespace BcAlias {
[[nodiscard]] bool Matches(const ImageInfo& requested, const ImageInfo& cached);
[[nodiscard]] bool Publish(TextureCache& cache, Image& image);
void Resolve(TextureCache& cache, ImageId cached, ImageId merged);
} // namespace BcAlias

/// Whether --async-pipelines-prefer-skip is on, which also turns on per-view history. With it, a
/// render target view that is not the whole image, such as one layer or one mip, keeps a history of
/// its own exact range while the image's table has room, and the skip rule judges it by that history
/// instead of refusing it as a partial view. Config is fixed after load, so any drawing thread may
/// read it, like the other skip rule flags.
[[nodiscard]] bool ViewHistoryEnabled();

/// How a view relates to its image for the history. Whole is one level with every layer, of an image
/// with one level. Layer selects some layers of that level, Mip any other level selection, and Outside
/// a range past the image or of no levels or layers.
enum class ViewRangeScope : uint8_t { Whole, Layer, Mip, Outside };

/// Places a view range against an image of `levels` levels and `layers` layers.
[[nodiscard]] ViewRangeScope ClassifyViewRange(uint32_t levels, uint32_t layers, const ImageViewInfo& view);

class TextureCache {
public:
	enum class BindingType : uint8_t { Texture, Storage, RenderTarget, DepthTarget, VideoOut };

	struct ImageDesc {
		ImageInfo     info;
		ImageViewInfo view_info;
		BindingType   type = BindingType::Texture;
		bool          heap = false; // PS5: resolved for a resident image heap.
	};

	/// Reads the readback options once. Fault waits need their own flag and a readback option.
	TextureCache(GraphicContext& graphics, CommandScheduler& scheduler, PageManager& page_manager,
	             BufferCache& buffer_cache);
	~TextureCache();
	KYTY_CLASS_NO_COPY(TextureCache);

	[[nodiscard]] ImageId       FindImage(ImageDesc& desc, bool exact_format = false);
	void                        UpdateImage(ImageId id);
	[[nodiscard]] ImageId       FindImageFromRange(uint64_t address, uint64_t size,
	                                               bool ensure_valid = true);
	[[nodiscard]] vk::ImageView FindTexture(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindRenderTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] vk::ImageView FindDepthTarget(ImageId id, const ImageDesc& desc);
	[[nodiscard]] Image&        GetImage(ImageId id) {
		auto& image = m_slot_images[id];
		TouchImage(image);
		return image;
	}
	void MarkGpuWritten(ImageId id);

	/// Notes that a draw in the current frame ran into the view of the image, and that a clear of
	/// the whole target came first when `cleared`. Called on the GPU thread for the draws that ran,
	/// while pipelines build in the background. A frame is one guest flip. A view that is not the
	/// whole image of one level and all its layers is ignored, because the history belongs to the
	/// image and a layer or mip drawn every frame would vouch for one drawn once. With
	/// --async-pipelines-prefer-skip such a view is noted in the history of its own exact range
	/// instead, while the table has room. Its clear repairs only that range, so it leaves the
	/// image's skip mark.
	void NoteRenderTargetUse(ImageId id, const ImageViewInfo& view, bool cleared,
	                         bool htile_cleared = false);
	/// How the view relates to its image, see ClassifyViewRange.
	[[nodiscard]] ViewRangeScope ViewScopeOf(ImageId id, const ImageViewInfo& view);
	/// How a target's recent frames judge it for the gate. Only Rebuilt lets a draw skip.
	enum class RebuildVerdict : uint8_t {
		Rebuilt,
		/// The view is not the whole image, as one level with all its layers.
		PartialView,
		/// Fewer than the minimum draws in this frame or in the one before.
		TooFewDraws,
		/// Fewer draws too, and the frame before shows no clear either, so more draws would not
		/// have been enough.
		TooFewDrawsUncleared,
		/// Enough draws, but no clear of the whole target in both frames.
		NotCleared,
		/// No register clear in both frames, but an HTILE clear waited in both.
		HtileClearedOnly,
	};
	/// Judges the view. It needs at least `min_draws` draws into it in this frame and in the one
	/// before, and, when `needs_clear`, a clear of the whole target in both. The draws asked about
	/// are not yet noted, so they are not their own evidence. With --async-pipelines-prefer-skip a
	/// partial view is judged the same way by the history of its exact range, and is a partial view
	/// only when that range has none.
	[[nodiscard]] RebuildVerdict JudgeRebuiltEveryFrame(ImageId id, const ImageViewInfo& view,
	                                                    bool needs_clear, uint32_t min_draws);
	/// What the first-run report keeps of a target the skip rule refused a draw for. That is the image's
	/// size, layers, levels and format, how the view relates to it, and the draws and clears of this
	/// frame and the one before in the same history JudgeRebuiltEveryFrame reads. It judges nothing.
	[[nodiscard]] Common::FirstRunReport::SkipTargetFacts DescribeSkipTarget(ImageId id, const ImageViewInfo& view);
	/// Notes that a draw into the image was skipped, as a depth target when `depth`. Returns true
	/// when a resident image heap references the image, which then gets no mark. The heap resolves
	/// its own bindings out of reach of TakeSkippedDraws, and it could select the image, so the
	/// caller reports the skip at once as a possible read.
	/// A skip of an image no heap references yet keeps its mark, and NoteHeapSelected reports it
	/// when a heap element first selects the image.
	[[nodiscard]] bool NoteSkippedTarget(ImageId id, bool depth);
	/// Takes the skipped draws noted for the image since its last whole-target clear, and says
	/// whether they were depth draws. Called when a shader binding resolves to the image. Zero
	/// when none are waiting. A color image is never cleared in view of the CPU, so its count only
	/// resets here, and the same skipped draws are never reported twice.
	///
	/// Limits, so a count is read as a signal and nothing more. Zero does not mean safe and
	/// nonzero does not mean damage. A read through another ImageId is not seen, such as an aliased
	/// format or a stencil association. Copies are not seen, such as ResolveColorTargets,
	/// CopyDepthToColor and DepthStencilCopy, nor buffer reads, nor presenting the image. Storage
	/// bindings are not counted, and an image load that only reads resolves to a Texture binding,
	/// because Storage means the shader writes the image. The binding is resolved before the draw
	/// that uses it knows whether it will be skipped, so a draw that is itself skipped can count
	/// marks of earlier skips for a read that never happens.
	[[nodiscard]] uint32_t TakeSkippedDraws(ImageId id, bool* depth);
	/// Called when a resident image heap element selects the image. Counts the heap's reference and
	/// reports the skips marked on the image so far as possible reads, under a heap class.
	void NoteHeapSelected(ImageId id);

	[[nodiscard]] bool ClearImageFromBuffer(CommandBuffer& command, uint64_t address, uint64_t size,
	                                        uint32_t packed_clear);
	void               InvalidateMemory(uint64_t address, uint64_t size);
	/// Invalidates images for a CPU write treated as running front to back over the range, such as
	/// a DMA fill or copy, as its page-by-page faults on a fully watched range would have.
	void               InvalidateSequentialWrite(uint64_t address, uint64_t size);
	void               InvalidateMemoryFromGPU(uint64_t address, uint64_t size);
	void               InvalidateBufferWrite(uint64_t address, uint64_t size);
	[[nodiscard]] bool IsRegionRegistered(uint64_t address, uint64_t size);

	[[nodiscard]] bool IsMeta(uint64_t address);
	[[nodiscard]] bool IsMetaCleared(uint64_t address, uint32_t slice);
	[[nodiscard]] bool ClearMeta(uint64_t address);
	// One buffer binding of a compute dispatch that may be a metadata clear.
	struct MetaClearTarget {
		uint64_t address = 0;
		uint64_t size    = 0;     // Bytes the descriptor covers.
		bool     bounded = true;  // An unbounded descriptor's size says nothing about the write.
		bool     written = false; // A binding that is not written must not be a metadata surface.
	};
	// Checks every target and, only if all pass, marks each written metadata surface cleared.
	// Returns false and changes nothing otherwise.
	[[nodiscard]] bool ClearMetaSurfaces(std::span<const MetaClearTarget> targets);
	[[nodiscard]] bool TouchMeta(uint64_t address, uint32_t slice, bool is_clear);

	void UnmapMemory(uint64_t address, uint64_t size);
	// Releases a range that the guest is likely to map again. With --keep-gpu-textures-on-reuse it
	// keeps the clean images in the range and marks them suspended, otherwise it unmaps the range.
	void SuspendMemory(uint64_t address, uint64_t size);
	// Clears the suspended mark on the images in a range that was suspended.
	void ResumeMemory(uint64_t address, uint64_t size);
	/// Records GPU-to-guest copies for enrolled images. Entries that can never be downloaded are
	/// dropped.
	/// @param keep_unready true keeps entries that fail only for now, to retry at the next call.
	/// false drops them. Such an entry was written by the CPU after the GPU, so guest memory is
	/// already newer, and a retry before a label would land a frame or more later.
	/// @return true if at least one copy was queued. Its guest write lands as a priority operation.
	bool ProcessDownloadImages(bool keep_unready = true);
	/// @return true if image readbacks are enabled at all. It is set at construction (only tests
	/// change it later), so it needs no lock.
	[[nodiscard]] bool ReadbacksEnabled() const noexcept {
		return m_readback_linear_images || m_readback_tiled_images;
	}
	/// @return true while a queued readback has not yet been written to guest memory. This counts
	/// the write-backs of evicted images too, even with readbacks off, since an evicted image's
	/// data exists only in its pending write-back.
	[[nodiscard]] bool HasReadbacksInFlight() const noexcept {
		return m_readbacks_in_flight.load(std::memory_order_acquire) != 0;
	}
	/// @return true if qualifying readbacks hold their guest pages. It is set at construction (only
	/// tests change it later), so it needs no lock.
	[[nodiscard]] bool ReadbackFaultWaitEnabled() const noexcept { return m_readback_fault_wait; }
	/// @return the guard holding the guest pages of pending readbacks.
	[[nodiscard]] ReadbackGuard& GetReadbackGuard() noexcept { return m_readback_guard; }
	void RunGarbageCollector();

private:
	struct MetaDataInfo {
		// PS5 DB_DEPTH_VIEW encodes 13-bit base and last array-slice indices.
		std::bitset<8192> clear_mask;
		// Largest metadata range registered at this address. Two live depth images can share an
		// HTILE base, and keeping the larger size keeps the larger image's clears eligible. The
		// entry is dropped when any image using it is freed, even if another still uses it, and the
		// next binding registers it again with its own size.
		uint64_t size = 0;
	};

	struct OverlapResult {
		ImageId image;
		int32_t mip   = -1;
		int32_t layer = -1;
	};

	// Byte range of a registered image, kept apart from Image so lookups read 16 dense bytes
	// instead of a large object. An unregistered slot holds an empty span.
	struct ImageSpan {
		uint64_t begin = 0;
		uint64_t end   = 0;
	};

	using ImageIds       = InlinePageOwnerList<ImageId, 16>;
	using ImagePageTable = MultiLevelPageTable<ImageIds, 20, 44, 14>;

	// Callers pass a nonempty, valid guest range.
	template <typename Func>
	static void ForEachPage(uint64_t address, size_t size, Func&& func) {
		const uint64_t page_end = (address + size - 1) >> ImagePageTable::kPageBits;
		for (uint64_t page = address >> ImagePageTable::kPageBits; page <= page_end; ++page) {
			func(page);
		}
	}

	[[nodiscard]] ImageId     InsertImage(const ImageInfo& info);
	void                      RegisterImage(ImageId id);
	void                      UnregisterImage(ImageId id);
	void                      DeleteImage(ImageId id);
	void                      FreeImage(ImageId id);
	void                      TouchImage(Image& image);
	void                      TrackImage(ImageId id);
	void                      TrackImageHead(ImageId id);
	void                      TrackImageTail(ImageId id);
	void                      UntrackImage(ImageId id);
	void                      UntrackImageHead(ImageId id);
	void                      UntrackImageTail(ImageId id);
	void                      MarkAsMaybeDirty(ImageId id, Image& image);
	void                      TrackImageDownload(ImageId id, Image& image);
	[[nodiscard]] static bool SameBacking(const ImageInfo& cached, const ImageInfo& requested,
	                                      bool exact_format);
	[[nodiscard]] static BindingType UploadBinding(const Image& image);
	// Colour attachment uploads need whole layers of single sample images. Other bindings pass.
	[[nodiscard]] static bool IsValidColorAttachmentUpload(const Image& image, BindingType binding);

	// Caller holds m_lock. Walks the page owners and tests overlap against m_image_spans, so no
	// Image object is read or written. Results are page-major, then in registration order.
	[[nodiscard]] ImageIds      FindImagesInRegion(uint64_t address, uint64_t size) const;
	// Whether the image was last used too long ago to be replaced. It counts scheduler ticks, or
	// frames while the submit cadence makes the ticks too fast to mean frames.
	[[nodiscard]] bool          AccessedLongAgo(const Image& image) const;
	// Same walk with an optional count of span overlap tests, which the tests use to prove each
	// owner is tested at most once.
	[[nodiscard]] ImageIds      FindImagesInRegionCounted(uint64_t address, uint64_t size,
	                                                      uint64_t* overlap_tests) const;
	[[nodiscard]] OverlapResult ResolveOverlap(const ImageInfo& requested, BindingType binding,
	                                           ImageId cached, ImageId merged);
	void                        RetireReplacedLayout(ImageId cached, ImageId merged);
	[[nodiscard]] ImageId       ResolveDepthOverlap(const ImageInfo& requested, BindingType binding,
	                                                ImageId cached);
	[[nodiscard]] ImageId       ExpandImage(const ImageInfo& info, ImageId source);
	void                        RefreshImage(ImageId id);
	// Whether every viewed metadata slice is already checked for the keys. The slices are one
	// range lookup when they touch, and otherwise are looked up one by one.
	[[nodiscard]] bool          IsColorClearChecked(uint64_t address, uint64_t slice_size,
	                                                uint64_t mip_size, uint32_t count,
	                                                uint32_t keys) const;
	// Marks every viewed metadata slice checked for the keys, as one range when the slices touch.
	void                        MarkColorClearChecked(uint64_t address, uint64_t slice_size,
	                                                  uint64_t mip_size, uint32_t count,
	                                                  uint32_t keys);
	void                        MaterializeColorClear(ImageId id, const ImageDesc& desc,
	                                                uint32_t metadata_base_layer);
	void                        InitializeImage(ImageId id);
	[[nodiscard]] bool          CanDownload(const Image& image) const;
	void UploadImage(Image& image, Buffer& source, uint64_t source_offset);
	void DownloadImage(Image& image, Buffer& destination, uint64_t destination_offset,
	                   uint64_t destination_size, uint32_t levels);
	void DownloadDepth(Image& image, Buffer& destination, uint64_t destination_offset);
	void CommitGpuWrite(Image& image);
	// Caller holds m_lock. Volume layer ranges select depth slices.
	void ClearImage(CommandBuffer& command, ImageId id, vk::Format format,
	                const vk::ImageSubresourceRange& range, const vk::ClearValue& clear);
	// Caller holds m_lock and ends rendering. Begins rendering `layers` layers of one color mip
	// through a `format` view, clearing them or, when `clear` is null, loading them; returns the
	// render area.
	vk::Rect2D BeginColorRendering(Image& image, vk::Format format, uint32_t level,
	                               uint32_t layer, uint32_t layers, const vk::ClearValue* clear,
	                               vk::CommandBuffer command);
	void RefreshCopySource(ImageId id);
	[[nodiscard]] bool CopyD16(Image& destination, Image& source);
	void               CopyImage(ImageId destination, ImageId source);
	[[nodiscard]] ImageId AssociateStencil(ImageId depth, GuestRange stencil);
	void CopyImageMip(ImageId destination, ImageId source, uint32_t mip, uint32_t layer);
	void ValidateImageDesc(const ImageDesc& desc) const;
	// Caller holds m_lock. A sequential write marks only the images its per-page faults reach.
	void InvalidateCpuAliases(uint64_t address, uint64_t size, bool sequential_write);

	// Caller holds m_lock. Whether the collector must skip the image, which it also touches.
	[[nodiscard]] bool HeldFromCollection(Image& image);
	// Caller holds m_lock. Refresh only the buffer-written range of an image, returning false
	// when the full upload has to run instead.
	[[nodiscard]] bool TryPartialRefresh(Image& image);
	void               RefreshLinearRange(Image& image, GuestRange range);
	// What a tiled partial refresh touches. The range is relative to the image data.
	struct TiledRefreshPlan {
		RefreshByteRange           range;
		std::vector<RefreshRegion> regions;
	};
	[[nodiscard]] static std::optional<TiledRefreshPlan> PlanTiledRefresh(const Image& image,
	                                                                      GuestRange   range);
	[[nodiscard]] bool RefreshTiledRange(Image& image, GuestRange range);
	void               MergeTiledRegions(Image& image, const TiledRefreshPlan& plan);
	void DownloadForMerge(Image& image, const TileManager::Result& scratch,
	                      std::span<const RefreshRegion> regions);
	void UploadMerged(Image& image, const TileManager::Result& scratch,
	                  std::span<const RefreshRegion> regions);
	[[nodiscard]] bool DownloadImageMemory(ImageId id);
	// Caller holds m_lock. Erases the surface metadata whose base lies in the range.
	void EraseSurfaceMetas(uint64_t address, uint64_t size);
	// Caller holds m_lock. Whether an image can stay registered while its range is suspended.
	[[nodiscard]] bool CanKeepOnSuspend(const Image& image, uint64_t address, uint64_t size) const;

	GraphicContext&                                   m_graphics;
	CommandScheduler&                                 m_scheduler;
	TrackingSpinLock                                  m_lock;
	PageManager&                                      m_page_manager;
	BlitHelper                                        m_blit_helper;
	TileManager                                       m_tiler;
	BufferCache&                                      m_buffer_cache;
	ReadbackGuard                                     m_readback_guard;
	Common::SlotVector<Image>                         m_slot_images;
	ImagePageTable                                    m_image_page_table;
	// Indexed by ImageId. Written only by RegisterImage and UnregisterImage, under m_lock.
	std::vector<ImageSpan>                            m_image_spans;
	Common::LeastRecentlyUsedCache<ImageId, uint64_t> m_lru_cache;
	std::unordered_set<ImageId>                       m_download_images;
	std::map<uint64_t, MetaDataInfo>                  m_surface_metas;
	uint64_t                                          m_total_used_memory  = 0;
	uint64_t                                          m_trigger_gc_memory  = 0;
	uint64_t                                          m_pressure_gc_memory = 1536ull * 1024 * 1024;
	uint64_t         m_critical_gc_memory     = 3ull * 1024 * 1024 * 1024;
	// Written and read without the lock only on the GPU thread.
	uint64_t         m_gc_tick                = 0;
	bool             m_readback_linear_images = false;
	bool             m_readback_tiled_images  = false;
	bool             m_readback_fault_wait    = false;
	// Readbacks recorded by DownloadImageMemory() whose guest write has not run yet.
	std::atomic<uint32_t> m_readbacks_in_flight {0};
	bool             m_keep_textures_on_suspend = false;
	bool             m_partial_buffer_refresh = false;

	/// Holds a readback's guest pages if it qualifies, and returns what was held, which may be
	/// nothing. Needs a successful backing read first, so private mappings are never held. A thread
	/// copying from one keeps the address-space lock while it reads, so a fault there could wait
	/// with that lock held, and the landing needs it.
	[[nodiscard]] GuestRange HoldReadback(const Image& image);
	/// Records the copy of an image's bytes to the download buffer.
	void RecordReadbackCopy(Image& image, Buffer& download, uint64_t offset, GuestRange range);

	friend struct TextureCacheTestAccess;
	friend class BufferCache;
	friend class RenderExecutor;
	friend class ImageHeap;
	friend bool BcAlias::Matches(const ImageInfo&, const ImageInfo&);
	friend bool BcAlias::Publish(TextureCache&, Image&);
	friend void BcAlias::Resolve(TextureCache&, ImageId, ImageId);
};

// Decodes a color metadata code into its clear color. Exposed so tests can compare it with the
// key mask.
[[nodiscard]] bool DecodeColorClear(const TextureCache::ImageDesc& desc, uint8_t code,
                                    vk::ClearColorValue& clear);
// Bit key is set when the clear code key << ColorClearKeyShift selects a clear for the image.
[[nodiscard]] uint32_t ColorClearKeys(const TextureCache::ImageDesc& desc);
// Whether the viewed metadata slices tile one range. A partial mip leaves gaps between slices
// that are never marked, and fewer than two slices gain nothing from a combined lookup.
[[nodiscard]] constexpr bool ColorClearSlicesContiguous(uint64_t mip_size, uint64_t slice_size,
                                                        uint32_t count) noexcept {
	return count >= 2 && mip_size == slice_size;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_TEXTURECACHE_H_
