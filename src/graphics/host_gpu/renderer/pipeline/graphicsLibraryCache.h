#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GRAPHICSLIBRARYCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GRAPHICSLIBRARYCACHE_H_

// Graphics pipeline libraries that are shared by every pipeline that needs the same one. A library
// is named by the key of its part (pipelineParts.h), so two pipelines that agree on, for example,
// their vertex shader and rasterizer state share one pre-rasterization library and compile it once.
//
// The first thread that asks for a library makes it on its own thread. Any other thread that asks
// for the same library while it is being made waits for that one build, and a thread that asks
// after it is made takes it at once. A thread only ever waits for a build that is already running
// on another thread, never for one that is queued, so the pipeline build queue cannot deadlock on it.
//
// A pipeline linked from libraries is in use as long as its libraries are, and the specification
// says the application keeps a library alive for as long as any pipeline linked from it is in use
// (Vulkan specification, Pipeline Libraries). Nothing is ever evicted, and the cache destroys every
// library when it is destroyed, so the owner must destroy it after every pipeline linked from it.

#include "common/assert.h"
#include "common/common.h"
#include "common/firstRunReport.h"
#include "common/gpuStall.h"
#include "common/threadPriority.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineParts.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>

namespace Libs::Graphics {

struct GraphicContext;

// The four libraries of a pipeline, in pipeline order. Their order matches
// Common::FirstRunReport::LibraryKind, so the report can name them.
enum class LibraryPart : uint8_t { VertexInput, PreRaster, Fragment, Output };
inline constexpr std::size_t LIBRARY_PART_COUNT = 4;
static_assert(LIBRARY_PART_COUNT == Common::FirstRunReport::LIBRARY_KIND_COUNT);

// A step of building a pipeline from libraries, for the test seam below. The link is the join.
enum class LibraryStep : uint8_t { VertexInput, PreRaster, Fragment, Output, Link };

// What the cache keeps of a library. The pipeline layout is the one the library was made with. A
// library that has no layout, which are the vertex input and the fragment output libraries, keeps
// none. It stays alive with the library, since the layout belongs to the library's creation.
struct LibraryHandles {
	vk::Pipeline       pipeline = nullptr;
	vk::PipelineLayout layout   = nullptr;
};

// What the cache counted of one kind of library. built and failed are builders that ran, hits are
// requests a finished library answered, including one that waited for a build under way, and
// refused_hits are requests answered by an entry the driver refused, which leave the caller to
// build the pipeline whole.
struct LibraryCounts {
	uint64_t built        = 0;
	uint64_t hits         = 0;
	uint64_t failed       = 0;
	uint64_t refused_hits = 0;
};

// What became of the fragment libraries the graphics prescan built ahead (experimental). first_hits are
// entries a later ordinary request found finished, counted once per entry, and waits are ordinary
// requests that found the prebuild still running and waited for it in its once flag, which is how a
// normal priority thread can end up waiting on a below normal prebuild. key_differs are ordinary
// requests that built a fragment library themselves for a pixel shader that was prebuilt under
// another key, such as a different sample count.
struct PrebuiltLibraryCounts {
	uint64_t built         = 0;
	uint64_t already_there = 0;
	uint64_t first_hits    = 0;
	uint64_t waits         = 0;
	double   wait_ms       = 0.0;
	double   wait_max_ms   = 0.0;
	uint64_t key_differs   = 0;
	// Prebuilds a waiting request raised to normal priority, waiters that found every boost slot
	// taken and waited unboosted, and the most prebuilds boosted at once.
	uint64_t boosts             = 0;
	uint64_t boosts_refused_cap = 0;
	uint64_t boosts_peak        = 0;
	// Of the boosts, those a waiter that blocks a draw made, to its own priority and past the cap.
	uint64_t boosts_blocking = 0;
};

// The most prebuilds that run boosted at once, so boosted prebuilds cannot crowd out the game's threads.
inline constexpr uint64_t MAX_CONCURRENT_BOOSTS = 2;

// Finds or builds libraries under per-entry once flags. A map lock guards only the lookup, and the
// build runs outside it, so builds of different libraries run side by side.
class GraphicsLibraryCache {
public:
	using DestroyFunction = std::function<void(const LibraryHandles&)>;

	// A cache that destroys its libraries on `device`, which must outlive it. It is defined here so that
	// the unit test, which has no Vulkan loader, can link the rest of the cache.
	explicit GraphicsLibraryCache(vk::Device device)
	    : GraphicsLibraryCache([device](const LibraryHandles& handles) {
		      device.destroyPipeline(handles.pipeline, nullptr);
		      device.destroyPipelineLayout(handles.layout, nullptr);
	      }) {}
	// A cache that destroys through the given function, which tests use to count the libraries.
	explicit GraphicsLibraryCache(DestroyFunction destroy);
	// Destroys every library. Every pipeline linked from them must be gone by then.
	~GraphicsLibraryCache();
	KYTY_CLASS_NO_COPY(GraphicsLibraryCache);

	// Returns the library for the key and makes it with `build` if nobody has yet. `build` runs on
	// the calling thread, at most once per key over the cache's life, and returns null handles when
	// the driver refuses. A refusal is kept, so the key is not retried and the caller builds the
	// pipeline whole. The handle stays valid until the cache is destroyed. `build` must not call
	// back into the cache.
	template <class Build>
	[[nodiscard]] LibraryHandles GetOrBuild(const VertexInputKey& key, Build&& build) {
		return Get(m_vertex_input, LibraryPart::VertexInput, key, build);
	}
	template <class Build>
	[[nodiscard]] LibraryHandles GetOrBuild(const PreRasterKey& key, Build&& build) {
		return Get(m_pre_raster, LibraryPart::PreRaster, key, build);
	}
	template <class Build>
	[[nodiscard]] LibraryHandles GetOrBuild(const FragmentKey& key, Build&& build) {
		return Get(m_fragment, LibraryPart::Fragment, key, build);
	}
	template <class Build>
	[[nodiscard]] LibraryHandles GetOrBuild(const FragmentOutputKey& key, Build&& build) {
		return Get(m_output, LibraryPart::Output, key, build);
	}
	// Builds the fragment library for the key ahead of any draw, for the graphics prescan
	// (experimental), and marks its entry prebuilt so the first ordinary request that uses it is counted.
	// Returns whether this call built it. An entry that is already there is left as it is.
	template <class Build>
	bool GetOrBuildPrebuilt(const FragmentKey& key, Build&& build) {
		return Prebuild(m_fragment, LibraryPart::Fragment, key, key.pixel_id, build);
	}
	// The same for the pre-rasterization library of a vertex shader, named by its id.
	template <class Build>
	bool GetOrBuildPrebuilt(const PreRasterKey& key, Build&& build) {
		return Prebuild(m_pre_raster, LibraryPart::PreRaster, key, key.stage_ids[0], build);
	}
	// What became of the prebuilt libraries of a part so far. Only fragment and pre-rasterization
	// libraries are ever prebuilt.
	[[nodiscard]] PrebuiltLibraryCounts PrebuiltCounts(LibraryPart part = LibraryPart::Fragment) const;

	// How a request raises a prebuild's thread and how the thread drops back. The defaults are the
	// system's, Common::CurrentThreadSystemId and BoostThreadToNormal. The prebuild reads its own level
	// with current_priority before anyone can boost it, and restore puts it back there, which
	// Common::SetCurrentThreadSystemPriority does by default, falling back to the lowest step when the
	// level could not be read or set. Hooks without current_priority restore Common::UNREAD_THREAD_PRIORITY,
	// and hooks without restore leave the thread where the boost put it.
	// A waiter that blocks a draw, the GPU thread or a deferred build worker, raises the prebuild to
	// its own priority, read by current_priority, through boost_to, which Common::BoostThreadToPriority
	// does by default. Hooks without boost_to boost such a waiter as any other.
	struct PriorityHooks {
		std::function<uint64_t()>          current_thread;
		std::function<bool(uint64_t)>      boost;
		std::function<void(int)>           restore;
		std::function<int()>               current_priority;
		std::function<bool(uint64_t, int)> boost_to;
	};
	// Replaces the hooks, so a test can see a boost. Set it before any prebuild runs.
	void SetPriorityHooksForTest(PriorityHooks hooks) { m_priority = std::move(hooks); }

	// How many libraries of a part the cache holds, failed ones included.
	[[nodiscard]] std::size_t Size(LibraryPart part) const;
	// What the cache counted of a part.
	[[nodiscard]] LibraryCounts Counts(LibraryPart part) const;
	// Counts a fast link and tells the report. The pipeline made holds the libraries until it is
	// destroyed, so the link is also live until NoteUnlinked.
	void NoteLink(double ms);
	// Says that a pipeline linked from this cache's libraries has been destroyed.
	void NoteUnlinked();
	// How many links were counted over the cache's life, which is what the report shows.
	[[nodiscard]] uint64_t Links() const;
	// How many linked pipelines have not been destroyed yet. The cache stops the process if it is
	// destroyed while any are, since a linked pipeline needs its libraries.
	[[nodiscard]] uint64_t LiveLinks() const;
	// Counts a pipeline built whole and tells the report.
	void NoteFallback(Common::FirstRunReport::FallbackKind reason);
	// How many pipelines were built whole for a reason.
	[[nodiscard]] uint64_t Fallbacks(Common::FirstRunReport::FallbackKind reason) const;

private:
	struct Entry {
		std::once_flag once;
		LibraryHandles handles;
		// Set when the build is over, so a request that finds it set did not wait for anyone.
		std::atomic<bool> done {false};
		// Set by a prebuild before it builds, and the first ordinary request after it claims the entry.
		std::atomic<bool> prebuilt {false};
		std::atomic<bool> prebuilt_claimed {false};
		// The prebuild's thread, as the system knows it, and whether a request has boosted it. See
		// TryBoostPrebuild for the states.
		std::atomic<uint64_t> builder_thread {0};
		std::atomic<uint8_t>  boost_state {0};
	};

	// A prebuild's boost states. Running until a request starts to boost it, Boosting while that
	// request asks the system, then Boosted if the system accepted or Running again if not, and
	// Finished once the build is over, after which nothing boosts it.
	static constexpr uint8_t BOOST_NONE     = 0;
	static constexpr uint8_t BOOST_RUNNING  = 1;
	static constexpr uint8_t BOOST_BOOSTING = 2;
	static constexpr uint8_t BOOST_BOOSTED  = 3;
	static constexpr uint8_t BOOST_FINISHED = 4;

	// The prebuilt counts, written by any thread.
	struct PrebuiltCounters {
		std::atomic<uint64_t> built {0};
		std::atomic<uint64_t> already_there {0};
		std::atomic<uint64_t> first_hits {0};
		std::atomic<uint64_t> waits {0};
		std::atomic<uint64_t> wait_us {0};
		std::atomic<uint64_t> wait_max_us {0};
		std::atomic<uint64_t> key_differs {0};
		std::atomic<uint64_t> boosts {0};
		std::atomic<uint64_t> boosts_refused_cap {0};
		std::atomic<uint64_t> boosts_peak {0};
		std::atomic<uint64_t> boosts_blocking {0};
	};

	// One table of entries. Entries are never erased, so a reference to one stays valid while other
	// threads add to the map.
	template <class Key>
	struct Table {
		mutable std::mutex                                       mutex;
		std::unordered_map<Key, Entry, PartKeyHash<Key>>         entries;
	};

	// Builds a library ahead under the entry's once flag and marks it prebuilt. `stage_id` is the id of
	// the shader the part is named by, which an ordinary build under another key is checked against.
	template <class Key, class Build>
	bool Prebuild(Table<Key>& table, LibraryPart part, const Key& key, uint64_t stage_id, Build& build) {
		Entry* entry = FindEntry(table, key);
		bool   built = false;
		std::call_once(entry->once, [&] {
			built = true;
			// Read before the prebuilt flag is set, since no request can boost the thread until then, so
			// this is the level the prescan's worker chose and the one a boost drops back to.
			const int builder_priority =
			    m_priority.current_priority ? m_priority.current_priority() : Common::UNREAD_THREAD_PRIORITY;
			// The builder is recorded and marked running before the prebuilt flag is set, so a request
			// that sees the flag can boost it. Both are set before the build, so a request that waits
			// on it can tell it waited for a prebuild.
			entry->builder_thread.store(m_priority.current_thread(), std::memory_order_relaxed);
			entry->boost_state.store(BOOST_RUNNING, std::memory_order_relaxed);
			entry->prebuilt.store(true, std::memory_order_release);
			NotePrebuiltStage(part, stage_id);
			{
				const PrebuildBoostGuard guard {*this, *entry, builder_priority};
				const auto               start = std::chrono::steady_clock::now();
				entry->handles                 = build();
				NoteBuilt(part, entry->handles.pipeline != nullptr, Common::FirstRunReport::ElapsedMs(start));
			}
			entry->done.store(true, std::memory_order_release);
		});
		auto& counters = Prebuilt(part);
		(built ? counters.built : counters.already_there).fetch_add(1, std::memory_order_relaxed);
		return built;
	}

	// Finds or adds the entry for the key under the map lock.
	template <class Key>
	Entry* FindEntry(Table<Key>& table, const Key& key) {
		std::scoped_lock lock(table.mutex);
		return &table.entries.try_emplace(key).first->second;
	}

	// Finds the entry under the map lock and then builds it under the entry's once flag.
	template <class Key, class Build>
	LibraryHandles Get(Table<Key>& table, LibraryPart part, const Key& key, Build& build) {
		Entry* entry = FindEntry(table, key);
		// Clocks are read only when the report is on, or when the request may wait, since a wait on a
		// prebuild is counted either way.
		const bool timed      = Common::FirstRunReport::Enabled();
		bool       built_here = false;
		const bool finished   = entry->done.load(std::memory_order_acquire);
		const bool waits      = !finished;
		const auto wait_start = waits ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
		// A prebuild runs below normal priority, and this request is about to wait for it in the once
		// flag, so the prebuild is raised first rather than left to lose to every other thread.
		if (waits && entry->prebuilt.load(std::memory_order_acquire)) {
			TryBoostPrebuild(*entry, part, Common::GpuStall::BlocksDraws(Common::GpuStall::CurrentThreadRole()));
		}
		std::call_once(entry->once, [&] {
			built_here       = true;
			const auto start = timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point {};
			entry->handles   = build();
			NoteBuilt(part, entry->handles.pipeline != nullptr,
			          timed ? Common::FirstRunReport::ElapsedMs(start) : 0.0);
			entry->done.store(true, std::memory_order_release);
		});
		if (!built_here) {
			NoteHit(part, entry->handles.pipeline != nullptr);
			// The call returned when the other thread's build finished, so the time since the start is wait.
			const double waited_ms = waits ? Common::FirstRunReport::ElapsedMs(wait_start) : 0.0;
			if (waits && timed) {
				Common::FirstRunReport::LibraryPipelineScope::NoteWaited(waited_ms);
			}
			// Read after the once flag, which orders it after the prebuild that set it.
			if (entry->prebuilt.load(std::memory_order_acquire)) {
				NotePrebuiltUse(*entry, part, waits, waited_ms);
			}
		} else if constexpr (std::is_same_v<Key, FragmentKey>) {
			NoteBuiltHere(part, key.pixel_id);
		} else if constexpr (std::is_same_v<Key, PreRasterKey>) {
			NoteBuiltHere(part, key.stage_ids[0]);
		}
		return entry->handles;
	}

	// Counts an ordinary request answered by a prebuilt entry, as its first hit or as a wait on it.
	void NotePrebuiltUse(Entry& entry, LibraryPart part, bool waited, double waited_ms);
	// Raises a running prebuild's thread, once per prebuild, and counts it. A `blocking` waiter, one
	// that holds up a draw, raises it to its own priority past MAX_CONCURRENT_BOOSTS, and any other to
	// normal priority within it.
	void TryBoostPrebuild(Entry& entry, LibraryPart part, bool blocking);
	// Ends a prebuild's boost window on its own thread, dropping it back to `builder_priority`, the level
	// it read before the build, if a request boosted it.
	void FinishPrebuildBoost(Entry& entry, int builder_priority);
	// Ends a prebuild's boost window when its build is over, however it ends. A build that throws
	// would otherwise skip the drop, leaving its thread raised and its boost slot and tag taken for
	// good, so the cap would refuse every later boost.
	class PrebuildBoostGuard {
	public:
		PrebuildBoostGuard(GraphicsLibraryCache& cache, Entry& entry, int builder_priority)
		    : m_cache(cache), m_entry(entry), m_builder_priority(builder_priority) {}
		~PrebuildBoostGuard() { m_cache.FinishPrebuildBoost(m_entry, m_builder_priority); }
		KYTY_CLASS_NO_COPY(PrebuildBoostGuard);

	private:
		GraphicsLibraryCache& m_cache;
		Entry&                m_entry;
		int                   m_builder_priority;
	};
	static void RaiseMaxCount(std::atomic<uint64_t>& maximum, uint64_t value);
	// Remembers a prebuilt pixel shader id, and counts an ordinary build of a fragment library for one
	// that was prebuilt under another key.
	void NotePrebuiltStage(LibraryPart part, uint64_t stage_id);
	void NoteBuiltHere(LibraryPart part, uint64_t stage_id);
	// The prebuilt counters of a part.
	[[nodiscard]] PrebuiltCounters& Prebuilt(LibraryPart part) { return m_prebuilt[static_cast<std::size_t>(part)]; }
	[[nodiscard]] const PrebuiltCounters& Prebuilt(LibraryPart part) const { return m_prebuilt[static_cast<std::size_t>(part)]; }

	// Count a builder that ran, and a request an entry answered, here and in the report.
	void NoteBuilt(LibraryPart part, bool succeeded, double ms);
	void NoteHit(LibraryPart part, bool usable);
	// Destroys every library once. The destructor calls it.
	template <class Key>
	void DestroyTable(Table<Key>& table);

	DestroyFunction                                          m_destroy;
	Table<VertexInputKey>                                    m_vertex_input;
	Table<PreRasterKey>                                      m_pre_raster;
	Table<FragmentKey>                                       m_fragment;
	Table<FragmentOutputKey>                                 m_output;
	std::array<std::atomic<uint64_t>, LIBRARY_PART_COUNT>    m_built {};
	std::array<std::atomic<uint64_t>, LIBRARY_PART_COUNT>    m_hits {};
	std::array<std::atomic<uint64_t>, LIBRARY_PART_COUNT>    m_failed {};
	std::array<std::atomic<uint64_t>, LIBRARY_PART_COUNT>    m_refused_hits {};
	std::atomic<uint64_t>                                    m_links {0};
	std::atomic<uint64_t>                                    m_live_links {0};
	std::array<std::atomic<uint64_t>, Common::FirstRunReport::FALLBACK_KIND_COUNT> m_fallbacks {};
	std::array<PrebuiltCounters, LIBRARY_PART_COUNT>         m_prebuilt;
	PriorityHooks                                            m_priority;
	// Prebuilds boosted now, which MAX_CONCURRENT_BOOSTS caps.
	std::atomic<uint64_t>                                    m_active_boosts {0};
	// The pixel shader ids that have a prebuilt fragment library. Read only when a request builds a
	// fragment library itself, which is a pipeline build anyway.
	mutable std::mutex                                       m_prebuilt_pixels_mutex;
	std::array<std::unordered_set<uint64_t>, LIBRARY_PART_COUNT> m_prebuilt_ids;
};

// Whether a pipeline is to be linked from libraries. That needs a library cache, the library mode on,
// and a pipeline that libraries cover (GplEligible). When the mode is on and the pipeline is not
// covered, the cache counts a pipeline built whole against the reason, and false is returned. With the
// mode off or no cache nothing is counted, since nothing was asked for.
[[nodiscard]] bool ShouldUseLibraries(const GraphicContext& graphics, GraphicsLibraryCache* cache,
                                      const GraphicsPipelineKey&                 key,
                                      const std::array<const ShaderModule*, 4>& stages);

// Builds a graphics pipeline by getting or building its four libraries from `cache` and linking
// them. `pipeline` must already have its split layout (CreateGraphicsLayouts), and the pipeline layout
// it names is the layout of the link. Returns false, with pipeline.pipeline null, when the driver
// refused a library or the link, so the caller builds the whole pipeline with the same layout. The
// pipeline is eligible (GplEligible) and its stages are indexed like GraphicsPipelineKey::stage_ids.
[[nodiscard]] bool CreatePipelineFromLibraries(GraphicContext& graphics, GraphicsLibraryCache& cache,
                                               PipelineCache::Pipeline&   pipeline,
                                               const GraphicsPipelineKey& key,
                                               const std::array<const ShaderModule*, 4>& stages,
                                               vk::PipelineCache driver_cache);

// How many pipelines every cache in the process has linked from libraries so far. A tripwire for
// tests, which read it as a difference to see that pipelines they did not make themselves, such as
// those of the existing draw tests, were linked from libraries or were not.
[[nodiscard]] uint64_t TotalLinkedPipelines();

// A seam for tests, which makes the driver seem to refuse. When set, the code that builds libraries
// calls it just before each step and fails the step if it returns true. Pass null to clear it. It
// is called from any thread that builds pipelines, so it must be thread-safe.
using LibraryFaultInjector = bool (*)(LibraryStep step);
void SetLibraryFaultInjector(LibraryFaultInjector injector);
// Whether the step is to fail. False when no injector is set.
[[nodiscard]] bool LibraryStepShouldFail(LibraryStep step);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GRAPHICSLIBRARYCACHE_H_
