#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/firstRunReport.h"
#include "common/gpuStall.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorSetLayoutCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"
#include "graphics/host_gpu/renderer/pipeline/prescanMiss.h"
#include "graphics/host_gpu/renderer/pipeline/retainedRecords.h"
#include "graphics/host_gpu/renderer/pipeline/warmBuildSet.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/storegen/generatedDrainPlanner.h"
#include "graphics/shader/storegen/generatedOrder.h"
#include "graphics/shader/storegen/prescanTransport.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics {

/// Where the module of a warm build came from, which decides what a stop for it is charged to.
enum class WarmOrigin : uint8_t {
	Store,     ///< The shader store's own records, from WarmUp.
	Generated, ///< The shader store generator.
	Prescan,   ///< The submit-time prescan, which builds on its own warm set.
};

/// The cause a dispatch's stop for a warm-up build is charged to, once Take has said how it got the
/// build. A build a worker had started is a wait, and one no worker had started ran on the GPU thread.
/// Inline builds are told apart by whether the queue was paused and by whether the module is
/// generated, and a generated module that ran inline while paused has its own cause, so neither
/// wins. The cost of the pause is WarmInlinePaused plus GeneratedInlinePaused, and the generator's
/// misses are GeneratedInline plus GeneratedInlinePaused. A build that was already done stops
/// nothing and is charged like an inline one. A prescan module is charged to PrescanWaited when a
/// worker had it, and to PrescanQueued when the GPU thread ran it or found it done, whatever the
/// pause, since the pause never holds the prescan's set.
[[nodiscard]] Common::GpuStall::HoldCause WarmTakeCause(PipelineBuildQueue::Acquired how, bool paused,
                                                        WarmOrigin origin);

struct GraphicContext;
struct ShaderParams;
class CommandBuffer;
class GraphicsLibraryCache;
class PrescanPreparer;
struct PrescanPreparerConfig;

namespace ShaderRecompiler {
struct CompileOptions;
struct ShaderSource;
struct TranslateResult;
} // namespace ShaderRecompiler

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

namespace StoreGen {
struct SupervisorConfig;
class StoreSupervisor;
struct GenRow;
enum class DrainVerdict : uint8_t;
struct GenerationRowInputs;
} // namespace StoreGen

namespace ShaderRecompiler::Spirv {
struct Profile;
} // namespace ShaderRecompiler::Spirv

#pragma pack(push, 1)

struct PipelineStaticParameters {
	bool                       negative_one_to_one      = false;
	bool                       depth_clip_enable        = true;
	vk::PrimitiveTopology      topology                 = vk::PrimitiveTopology::ePointList;
	// 0 without tessellation, 3 for a rectangle list, else the input control points.
	uint32_t                   patch_control_points     = 0;
	bool                       primitive_restart_enable = false;
	uint32_t                   samples                  = 1;
	bool                       sample_shading_enable    = false;
	bool                       alpha_to_coverage_enable = false;
	uint32_t                   color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	bool                       provoking_vtx_last                                 = false;
	vk::PolygonMode            polygon_mode                                       = vk::PolygonMode::eFill;
	uint8_t                    color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                    alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                    alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                       separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                       blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
};

struct PipelineRenderingState {
	std::array<vk::Format, RENDER_COLOR_ATTACHMENTS_MAX> color_formats {};
	vk::Format                                           depth_format   = vk::Format::eUndefined;
	vk::Format                                           stencil_format = vk::Format::eUndefined;
	uint32_t                                             color_count    = 0;
};

struct PipelineVertexInputState {
	struct Binding {
		uint32_t stride   = 0;
		uint8_t  instance = 0;
	};
	struct Attribute {
		uint32_t offset  = 0;
		uint8_t  binding = 0;
	};

	std::array<Binding, ShaderVertexInputInfo::RES_MAX>   bindings {};
	std::array<Attribute, ShaderVertexInputInfo::RES_MAX> attributes {};
	uint8_t                                               binding_count   = 0;
	uint8_t                                               attribute_count = 0;
};

// Everything a graphics pipeline is made of besides the stage modules its ids name. Aligned so
// that rendering.color_formats can be passed to Vulkan.
struct alignas(4) GraphicsPipelineKey {
	// VS, LS or mesh; HS; TES; PS. 0 marks an absent stage.
	std::array<uint64_t, 4>  stage_ids {};
	PipelineRenderingState   rendering;
	PipelineVertexInputState vertex_input;
	PipelineStaticParameters static_params;

	bool operator==(const GraphicsPipelineKey& other) const;
};

#pragma pack(pop)

// Keys are compared and hashed as bytes.
static_assert(std::has_unique_object_representations_v<GraphicsPipelineKey>);

struct DecodedModuleRecord;

// A compiled stage with what pipeline creation needs from it besides the key.
struct ShaderModule {
	uint64_t                                     id     = 0;
	vk::ShaderModule                             module = nullptr;
	// PS only, and only with --graphics-pipeline-library on a device that has it. The same SPIR-V with
	// its descriptor set 0 resources moved to the pixel set of the split layout. Pipelines built with
	// that layout use this handle and every other pipeline keeps `module`. Null otherwise.
	vk::ShaderModule                             gpl_module = nullptr;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<uint32_t>                        function_code;
	// VS or LS: the host format of each vertex input attribute.
	std::vector<vk::Format>                      vertex_formats;
	// Rectangle-list PS: the parameters it reads.
	std::vector<RectListParameter>               rect_parameters;
	// Indirect mesh draw: a task shader reads the records and launches the groups.
	bool                                         mesh_task = false;
	// A prescan module's store record, kept until its first use appends it to the shader store.
	// Empty for every other module, and for a prescan module without a store.
	RetainedRecordSlot                           retained;
};

// A copy of a module silently drops its retained record, so a container that copied modules when
// it grows would lose them. Containers move only what is nothrow move constructible.
static_assert(std::is_nothrow_move_constructible_v<ShaderModule>);

// A draw's topology, shader stages (from PipelineCache::GetGraphicsPrograms) and targets.
struct DrawRenderState {
	RenderDepthInfo                      depth_info;
	RenderColorInfo                      color_info[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	uint32_t                             color_count                              = 0;
	// Color targets the pixel program writes.
	uint32_t                             mrt_mask                                 = 0;
	vk::PrimitiveTopology                topology = vk::PrimitiveTopology::ePointList;
	bool                                 primitive_restart_enable                 = false;
	bool                                 ps_active                                = false;
	std::array<ShaderVertexInputInfo, 3> vertex_info;
	ShaderPixelInputInfo                 ps_input_info;
	// Input: the user-data slots a GPU-read indirect draw patches.
	ShaderIndirectOffsets                indirect_offsets;
	bool                                 indirect_draw = false;

	[[nodiscard]] std::span<const ShaderVertexInputInfo> VertexStages() const {
		return {vertex_info.data(), vertex_info[1].stage ? 3u : 1u};
	}
};

// Writes the shader store on its own thread, running jobs in the order they were posted.
class ShaderStoreWriter {
public:
	// Keeps the first `kept` bytes of the store at `path`, or starts it over with `header`. The
	// thread runs only if that succeeded.
	void Open(const std::filesystem::path& path, std::span<const uint8_t> header, uint64_t kept);
	explicit operator bool() const { return m_thread.joinable(); }
	// Compresses `record` and appends it on the store thread, in call order.
	void Append(std::vector<uint8_t>&& record);

private:
	friend struct PipelineCacheTestAccess;

	void Post(Common::UniqueFunction<void>&& job);

	Common::File                             m_file;
	std::mutex                               m_mutex;
	std::condition_variable_any              m_ready;
	std::queue<Common::UniqueFunction<void>> m_jobs;
	// Declared last, so it drains the queue and joins before the rest is destroyed.
	std::jthread m_thread;
};

// What the intake of generated modules did, for the report. The drain counters belong to the GPU
// thread, so read them from it or after StopStoreGeneration.
struct StoreGenerationStats {
	StoreGen::GeneratedDrainCounters drain;
	// What the prescan's own planner decided about the prescan's modules, which drain leaves out.
	StoreGen::GeneratedDrainCounters prescan_drain;
	// Modules the generator's thread posted, and the shader module handles it made and the cache
	// destroyed, which are equal once nothing is waiting.
	uint64_t posted            = 0;
	uint64_t handles_created   = 0;
	uint64_t handles_destroyed = 0;
	// Records whose first-of-program plan could not be decoded, so they were never posted.
	uint64_t bad_plans = 0;
	// Posted modules that the stop destroyed before any drain took them.
	uint64_t undrained_at_stop = 0;
	// Posted modules that are waiting for a drain.
	uint64_t pending = 0;
	// The store records of prescan modules, which wait for each module's first use.
	RetainedRecordCounters prescan_records;
};

// The longest the warm-up queue is held while deferred draws wait for their pipelines. A resolve
// waits for the parallel builds of one submission, and the measured wait of the worst frame was
// 3.8 to 7.5 s (4 to 6 workers, 1.0 to 1.34 times contention, design section 7). Ten seconds is
// above that, so a normal resolve ends the pause first with Resume, while a resolve that never
// comes cannot starve the warm-up for longer than this.
inline constexpr std::chrono::seconds WARM_UP_PAUSE_LIMIT {10};

// How many workers build the pipelines of the prescan's modules. The prototype measured 35.0, 21.7
// and 20.4 s of LEGO compute stalls with 1, 2 and 3, and the audio review accepted 3 below normal.
inline constexpr uint32_t PRESCAN_BUILD_WORKERS = 3;

// What the prescan's pipeline builds did. A build overlapped a stop of the GPU thread when a
// GpuStall Scope was active at its start or end, or one began while it ran, and it ran on a worker.
// It overlapped a hold of guest audio when a hold was in progress at its start or end, or one ended
// while it ran, and it ran on a worker. in_scope_ms and in_hold_ms are the whole time of those
// builds, so they are upper bounds on the overlap, and nothing may gate on them. total_ms is the
// time of the worker builds only, since it is what those two are measured against. builds counts
// every build, on a worker or on the GPU thread, and worker_builds only the workers' builds, which
// is the count in_scope and in_hold are fractions of.
struct PrescanBuildStats {
	uint64_t builds        = 0;
	uint64_t worker_builds = 0;
	uint64_t in_scope      = 0;
	uint64_t in_hold       = 0;
	double   total_ms      = 0.0;
	double   in_scope_ms   = 0.0;
	double   in_hold_ms    = 0.0;
};

/// The first-run report's view of how the drain treated the prescan's modules. The timing comes
/// from `both`, the generator's planner, which times every drain of either producer's modules,
/// since the prescan's planner never times a drain of its own.
[[nodiscard]] Common::FirstRunReport::PrescanDrainReport
PrescanDrainReportOf(const StoreGen::GeneratedDrainCounters& prescan,
                     const StoreGen::GeneratedDrainCounters& both);
/// The first-run report's view of the prescan's builds.
[[nodiscard]] Common::FirstRunReport::PrescanBuildsReport
PrescanBuildsReportOf(const PrescanBuildStats& builds);
/// The first-run report's view of what became of the prescan's store records.
[[nodiscard]] Common::FirstRunReport::PrescanRecordsReport
PrescanRecordsReportOf(const RetainedRecordCounters& records);
/// The first-run report's view of the GPU thread's own compute compiles by reason, with every
/// reason and every refusing verdict named, those still at 0 too.
[[nodiscard]] Common::FirstRunReport::PrescanMissesReport
PrescanMissesReportOf(const PrescanMissCounts& counts);

// What a draw that has no pipeline yet does while the pipeline builds. A skipped draw leaves out its
// commands, a deferred draw becomes a hole in the command stream that is recorded once the pipeline
// is built, and a blocked draw waits for the pipeline on the GPU thread.
enum class DrawFate : uint8_t { Skip, Defer, Block };

// What ChooseDrawFate weighs for one draw that missed its pipeline.
struct DrawFateInputs {
	// --deferred-pipeline-draws is on, and the draw is allowed to become a hole.
	bool defer_allowed = false;
	// --async-pipelines is on, and the S2 rules allow the draw to go without its pipeline.
	bool skip_allowed = false;
};

// Decides the fate of a draw. Policy B is the default and defers a draw first, so no geometry is
// lost, and falls back to an S2 skip only for a draw that cannot be deferred. Policy A asks for the
// skip first. It is one constant away.
inline constexpr bool SKIP_BEFORE_DEFER = false;
[[nodiscard]] constexpr DrawFate ChooseDrawFate(const DrawFateInputs& inputs) noexcept {
	if (SKIP_BEFORE_DEFER && inputs.skip_allowed) {
		return DrawFate::Skip;
	}
	if (inputs.defer_allowed) {
		return DrawFate::Defer;
	}
	return inputs.skip_allowed ? DrawFate::Skip : DrawFate::Block;
}

// Asked when a draw's pipeline is still being built in the background, to say whether the draw may
// go without it or wait out of line.
class PipelineSkipPolicy {
public:
	virtual ~PipelineSkipPolicy() = default;
	// Called on the GPU thread, only for a draw that has no pipeline yet and only when
	// --async-pipelines is on, since the S2 rules are what it judges.
	[[nodiscard]] virtual bool MaySkip() const = 0;
	// The name of the rule that refused the draw at the last MaySkip, or null if it did not. A
	// draw that missed its pipeline and was refused is counted under that name.
	[[nodiscard]] virtual const char* Refusal() const { return nullptr; }
	// The name of what stops the draw itself from being deferred, or null when nothing does. It
	// covers the draw's own state, and the pipeline cache adds what its shaders need. A policy that
	// does not override it never defers.
	[[nodiscard]] virtual const char* DeferBlocker() const { return "unsupported"; }
};

// The owning renderer serializes access, including saves while the GPU is running.
class PipelineCache {
public:
	// Opens the driver cache. Nothing from the shader store is loaded until WarmUp.
	PipelineCache(GraphicContext& graphics, ShaderRecompiler::IR::ImageHeapSource* image_heap);
	~PipelineCache();
	KYTY_CLASS_NO_COPY(PipelineCache);
	// Replays the shader store at this path and keeps appending to it. An empty path keeps nothing
	// across runs. Pipelines read the image heap's descriptor set layout, so the owner calls this
	// once the heap exists, and only once, because a second call would load every record again.
	void WarmUp(const std::filesystem::path& path);
	void Save();

	struct Pipeline {
		vk::PipelineLayout      pipeline_layout       = nullptr;
		vk::Pipeline            pipeline              = nullptr;
		vk::DescriptorSetLayout descriptor_set_layout = nullptr;
		// The push constant range's stages, which every update names.
		vk::ShaderStageFlags    push_stages;
		// No descriptor set has to be allocated for a draw of this pipeline. With the split layout
		// that is a VS with no bindings and a PS set that is pushed.
		bool                    uses_push_descriptors = false;
		bool                    uses_image_heap       = false;
		// The split layout of --graphics-pipeline-library. Set 0 holds the VS bindings, set 1 is the
		// image heap and set 2 holds the PS bindings. `descriptor_set_layout` is null then, and the
		// two layouts below belong to the cache's DescriptorSetLayoutCache and are shared, so the
		// pipeline does not destroy them.
		bool                    uses_split_sets       = false;
		vk::DescriptorSetLayout vertex_set_layout     = nullptr;
		vk::DescriptorSetLayout pixel_set_layout      = nullptr;
		// Set 2 is a push set. The VS set is never one, since a layout may have only one push set.
		bool                    pixel_set_is_push     = false;
		// The pipeline was linked from graphics pipeline libraries and not built whole.
		bool                    from_libraries        = false;
		// The cache whose libraries the pipeline was linked from, or null. Destroying the pipeline
		// tells that cache, which checks that no pipeline outlives its libraries.
		GraphicsLibraryCache*   linked_from           = nullptr;
	};

	// Reads the shaders and registers bound in `command` into the state's shader stages.
	void GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state);
	// Null skips the dispatch.
	const ShaderModule* GetComputeProgram(const HW::ComputeShaderInfo& regs,
	                                      ShaderComputeInputInfo&      input_info);

	// Waits for the pipeline if it is not built yet.
	Pipeline& GetGraphicsPipeline(const CommandBuffer& command, const DrawRenderState& state);
	// Set by AcquireGraphicsPipeline when the draw is to become a hole. The draw then waits for the
	// pipeline stored under this key, whose build is already queued or done.
	struct DeferredDraw {
		bool                active = false;
		GraphicsPipelineKey key {};
		// For the report, what the S2 rules said of the draw, the refusing rule's name or one of the
		// first-run report's HOLE_VERDICT names. A static string, set with `active`.
		const char*         skip_verdict = nullptr;
	};
	// Like GetGraphicsPipeline, but returns null instead of waiting when a background build has
	// not finished and the draw may go without. Without a policy, or when builds are not in the
	// background, it always returns a pipeline. A miss is judged by ChooseDrawFate. A draw that may be
	// skipped (only with --async-pipelines) starts the build and returns null. With `deferred` given
	// and --deferred-pipeline-draws on, an eligible draw starts the build, sets `deferred` and
	// returns null. Every other miss builds on the calling thread or waits for the running build.
	[[nodiscard]] Pipeline* AcquireGraphicsPipeline(const CommandBuffer&   command,
	                                                const DrawRenderState& state,
	                                                const PipelineSkipPolicy* policy,
	                                                DeferredDraw* deferred = nullptr);
	Pipeline& GetComputePipeline(const ShaderComputeInputInfo& input_info);
	// What one resolve of deferred draws spent in the cache. wait_ms is the wall time of the whole
	// wait, including a build this thread ran itself. save_ms is the driver cache save after it.
	struct DeferredAcquireStats {
		uint32_t distinct_keys = 0;
		double   wait_ms       = 0.0;
		double   save_ms       = 0.0;
		// The critical key, the one with a build whose own acquire took longest, so the report can find
		// the holes that waited on it. Looked for only with the first-run report on.
		GraphicsPipelineKey critical_key {};
		bool                has_critical = false;
	};
	// Returns a copy of the pipeline for each key, in the same order, waiting for the background
	// builds in the order the keys are given. The builds were requested when their draws were met, so
	// the workers run them side by side and the loop only waits. A key may repeat, and a build that an
	// earlier drain or a blocking draw of the same key already took is not taken again. With
	// hold_audio the whole wait is one GpuStall scope, so guest audio holds once for the resolve and
	// never per key. The driver cache is saved once afterwards, outside the scope. Every key must be
	// built or pending. GPU thread only, and it touches nothing but this cache.
	[[nodiscard]] std::vector<Pipeline>
	AcquireDeferredPipelines(std::span<const GraphicsPipelineKey> keys, bool hold_audio,
	                         DeferredAcquireStats& stats);
	// True while graphics pipelines are built in the background. That is --async-pipelines or
	// --deferred-pipeline-draws, and it does not mean that draws are skipped. Skips follow
	// Config::AsyncPipelinesEnabled() alone.
	[[nodiscard]] bool BuildsInBackground() const noexcept { return m_build_queue != nullptr; }
	// Drops queued builds, waits for running ones and destroys finished pipelines nobody has
	// taken. Later requests build on the calling thread. The owner calls it while the objects the
	// builds read, such as the image heap's layout, still exist. Calling it again does nothing.
	void StopBuilds();
	// Posts a module that the shader store generator made, from the generator's thread. The record is
	// decoded off the GPU thread: its shader module is created here, and the plan of a program is
	// decoded for the first record of the program only. The GPU thread takes the module in at the top
	// of a shader lookup. Does nothing unless generation was enabled, and after it was stopped it
	// only destroys what it made. Never touches a cache map.
	void PostGeneratedModule(DecodedModuleRecord&&           record,
	                         const StoreGen::GeneratedOrder& order = {});
	// Ends the intake of generated modules and destroys the ones no drain took. Call it before
	// StopWarmBuilds, since a module that was taken in belongs to the warm queue. Does nothing
	// when there is no generation, and calling it again does nothing. It touches the backlog, which
	// belongs to the GPU thread, so it must not run while that thread can still drain: the destructor
	// and a StopBuilds that follows the GPU's shutdown meet that. The generator's thread has to be
	// stopped and joined first by whoever owns it, since it posts into the state this ends.
	void StopStoreGeneration();
	// What the intake did so far, or nothing without generation. The drain counters and the backlog
	// belong to the GPU thread, so call it from that thread, or after StopStoreGeneration. The pending
	// count is exact only then.
	[[nodiscard]] std::optional<StoreGenerationStats> GenerationStats() const;
	// Holds the warm-up queue from new builds, from the first hole of a submission until its resolve
	// ends, so the compiles the resolve waits for do not compete with it for cores. Only with
	// --background-warm-up, --deferred-pipeline-draws and --pause-warm-up-for-draws, and never the queue of the deferred draws
	// themselves. Asking again while held keeps the first deadline, so repeats cannot extend it, and
	// the warm-up limit ends a pause whose resolve never comes. Running warm builds finish, and a
	// compute dispatch that needs a queued one still runs it on the GPU thread.
	void PauseWarmBuilds();
	// Lets the warm-up queue take builds again. Does nothing when it is not held.
	void ResumeWarmBuilds();
	// How many warm-up compute builds the GPU thread ran itself while the queue was held. It is an
	// upper bound on what the pause cost, since a worker busy with something else might not have
	// reached the build either.
	[[nodiscard]] uint32_t WarmInlineDuringPause() const noexcept { return m_warm_inline_during_pause; }

	// Starts the submit-time prescan's preparer, its hand-off and its own warm set, with the
	// transport the factory makes. With no factory it is the Stream worker in a child process
	// behind a StreamSupervisor, which is what a real run uses, and tests pass the in-process one.
	// The prescan then stays off when no private directory for the helper's files can be had.
	// The host values come from this device and Config, as the runtime translates with them.
	// Modules it builds enter the generated intake, which is made if the generator did not make it.
	// Calling it again does nothing. A GuestGpu whose options turn the prescan on calls it before
	// it starts its thread, and attaches the scanner that fills the hand-off.
	// Call it after WarmUp and on the thread that owns the program cache. The preparer is seeded
	// with the skip keys of every module the cache holds then, which reads the program cache. Before
	// WarmUp the seed is empty and each loaded record is reported through a ring of
	// PRESCAN_KNOWN_CAPACITY, which a big store overflows, so some stored modules are compiled again.
	// On another thread the seed would race the GPU thread's adoptions. WarmUp logs a store loaded
	// after the prescan started.
	void EnablePrescan(StoreGen::PrescanTransportFactory make_transport = {});
	// The hand-off the scanner pushes into, or null without the prescan. It lives as long as the
	// cache, and the scanner must be joined before StopPrescan.
	[[nodiscard]] PrescanHandoff* PrescanInbox() noexcept { return m_prescan_handoff.get(); }
	// The preparer, or null without the prescan.
	[[nodiscard]] PrescanPreparer* Prescan() noexcept { return m_prescan.get(); }
	// Stops the preparer and its transport and joins it, so nothing posts into the intake after.
	// The preparer stays, so its counters can still be read. Calling it again does nothing.
	void StopPrescan();
	// What the prescan's pipeline builds did so far. Any thread.
	[[nodiscard]] PrescanBuildStats PrescanBuilds() const;
	// How many of the GPU thread's own compute compiles were charged to PrescanInFlight only because
	// the unit was in flight at the lookup's miss, the helper having answered before the Scope
	// opened. Any thread.
	[[nodiscard]] uint64_t PrescanInFlightAtMissOnly() const noexcept {
		return m_in_flight_at_miss_only.load(std::memory_order_relaxed);
	}
	// The GPU thread's own compute pipeline compiles with the prescan on, by the reason the prescan
	// did not deliver the module. Each count is exact when read, and they are read one by one, so a
	// compile counted meanwhile may show in one and not another, though never in built_undrained or
	// refused_by without its reason. All 0 without the prescan. Any thread.
	[[nodiscard]] PrescanMissCounts PrescanMisses() const noexcept;

private:
	friend struct PipelineCacheTestAccess;

	struct Program {
		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		// A deque, so a module keeps its address while later ones are added. Queued background
		// builds point at their stage modules.
		std::deque<ShaderModule>                     modules;
		bool                                         skip_dispatch = false;

		[[nodiscard]] std::span<const uint32_t> FunctionCode() const;
		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime);
		template <bool HasCalls>
		ShaderModule* FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                         uint32_t                                push_data_cursor);
	};

	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	// Variants of one source share a bucket; ProgramKey equality compares the state.
	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const { return key.hash; }
	};

	struct GraphicsPipelineKeyHash {
		std::size_t operator()(const GraphicsPipelineKey& key) const;
	};

	using ModuleMap = std::unordered_map<uint64_t, const ShaderModule*>;
	using Stages    = std::array<const ShaderModule*, 4>;

	// One pipeline built in the background. A worker fills `pipeline` and then sets `completed`.
	// Everything else is the GPU thread's.
	struct GraphicsBuild {
		GraphicsBuild(const GraphicsPipelineKey& request_key, const Stages& request_stages);

		GraphicsPipelineKey                   key;
		Stages                                stages;
		Pipeline                              pipeline;
		std::chrono::steady_clock::time_point requested;
		// Written by the worker before `completed`, so read it only after seeing that set.
		double                                ready_ms = 0.0;
		std::atomic<bool>                     completed {false};
		bool                                  adopted        = false;
		uint32_t                              skipped_draws  = 0;
	};

	// A build that is queued or running, with the handle to wait for it.
	struct PendingBuild {
		std::shared_ptr<GraphicsBuild>  build;
		PipelineBuildQueue::JobPtr      job;
	};

	GraphicContext&                      m_graphics;
	// The set layouts of the split pipelines, shared by them and by the libraries that come later.
	// It is declared before everything that builds or holds a pipeline, so it is destroyed after the
	// build queue and after the destructor body has destroyed every pipeline layout. The descriptor
	// heap in RenderContext is destroyed after this cache and keeps handles only as map keys, which
	// it never looks up again once the cache is gone.
	DescriptorSetLayoutCache             m_set_layouts;
	// The libraries of --graphics-pipeline-library, shared by the pipelines linked from them. It is
	// always made and only used while the library mode is on. It is declared after the set layouts,
	// which its libraries' pipeline layouts name, and the destructor body destroys it after every
	// pipeline, since a
	// linked pipeline is in use as long as the libraries are (Vulkan specification, Pipeline
	// Libraries). Worker builds use it, so it is made before the build queue and outlives it.
	std::unique_ptr<GraphicsLibraryCache> m_libraries;
	ShaderRecompiler::IR::ImageHeapSource* m_image_heap;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash> m_program_cache;
	// Modules compiled per stage.
	std::array<uint32_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> m_compiled {};
	ProgramKey                             m_lookup_key;
	vk::PipelineCache             m_driver_cache = nullptr;
	std::filesystem::path         m_driver_cache_path;
	std::chrono::steady_clock::time_point m_next_driver_cache_save;
	// Set by a worker too, when a background build adds to the driver cache, so the save at exit
	// that follows StopBuilds writes the builds that finished after the last one.
	std::atomic<bool>             m_driver_cache_dirty {false};
	std::unordered_map<GraphicsPipelineKey, Pipeline, GraphicsPipelineKeyHash> m_graphics_pipelines;
	std::unordered_map<uint64_t, Pipeline>                                     m_compute_pipelines;
	std::unordered_map<uint64_t, ShaderTessellationInputInfo>                  m_tess_layouts;
	// Records everything first created; closed without a shader store.
	ShaderStoreWriter m_store;
	// Null unless --async-pipelines or --deferred-pipeline-draws is on. Only CreatePipelineInternal
	// runs on its workers.
	std::unique_ptr<PipelineBuildQueue>                                          m_build_queue;
	// A deferred draw that the S2 rules would have skipped, or with --async-pipelines one they refused,
	// kept until its resolve reports it. `refusal` is the refusing rule's name, null for a draw they
	// would have skipped. Only the GPU thread touches it.
	struct WouldSkip {
		std::shared_ptr<GraphicsBuild>        build;
		std::chrono::steady_clock::time_point deferred_at;
		const char*                           refusal = nullptr;
	};
	std::vector<WouldSkip> m_would_skip;
	// Builds not yet taken into m_graphics_pipelines. Only the GPU thread touches it, like
	// m_graphics_pipelines.
	std::unordered_map<GraphicsPipelineKey, PendingBuild, GraphicsPipelineKeyHash> m_pending_graphics;
	// Finished builds waiting for the GPU thread. Workers add to the list under the mutex and then
	// set the flag, and the GPU thread clears the flag before it takes the list.
	std::mutex                                                                   m_published_mutex;
	std::vector<std::shared_ptr<GraphicsBuild>>                                  m_published;
	std::atomic<bool>                                                            m_has_published {false};
	// Null unless --background-warm-up is on and a shader store started builds. Holds the compute
	// pipelines the warm-up builds, by module id, until a lookup adopts them. Only the GPU thread
	// takes from it, and a worker writes only into the build's own storage.
	std::unique_ptr<WarmBuildSet<uint64_t, Pipeline>>                            m_warm_compute;
	// Whether PauseWarmBuilds holds the warm queue, and how many builds ran on the GPU thread meanwhile.
	// Only the GPU thread touches them.
	bool                                                                         m_warm_paused = false;
	uint32_t                                                                     m_warm_inline_during_pause = 0;
	// Ids of the modules whose warm build came from the generator. The GPU thread fills it when it
	// starts such a build and reads it before a stall, so a hold never waits on a lock for it.
	std::unordered_set<uint64_t>                                                 m_generated_warm_ids;
	// Module ids in the order their records appear in the shader store, one entry per record. Filled
	// while WarmUp loads the store and emptied once the warm-up has queued its builds.
	std::vector<uint64_t>                                                        m_module_record_order;
	// Zero sizes the warm queue from --warm-workers, or without it the hardware threads. A test
	// sets one to make the queue run jobs strictly in order, and it wins over the option.
	uint32_t                                                                     m_warm_worker_count_override = 0;
	// Generated modules (--generate-shader-store). Null unless generation is enabled, so a cache
	// without it has one pointer and one flag for it and nothing else. The inbox is filled by the
	// generator's thread and drained by the GPU thread, like the published builds.
	struct GeneratedModule;
	struct GeneratedIntake;
	struct GenerationState;
	std::unique_ptr<GenerationState> m_generation;
	std::atomic<bool>                m_has_generated {false};
	// A test changes the generator's settings here, such as the program it starts and its timeouts,
	// before WarmUp.
	std::function<void(StoreGen::SupervisorConfig&)> m_supervisor_tweak;
	// The submit-time prescan, all null unless EnablePrescan ran. The hand-off is declared before
	// the preparer, which reads it until it is joined. The prescan's modules build on their own warm
	// set, which PauseWarmBuilds never holds. The preparer posts into the generated intake, so
	// StopStoreGeneration stops it first.
	std::unique_ptr<PrescanHandoff>                   m_prescan_handoff;
	std::unique_ptr<WarmBuildSet<uint64_t, Pipeline>> m_prescan_compute;
	std::unique_ptr<PrescanPreparer>                  m_prescan;
	// The preparer's transport when it is the helper in a child process, for the report's child
	// figures, or null. The preparer owns it, so it lives as long as m_prescan. The factory sets it
	// while EnablePrescan makes the preparer, and nothing changes it once that succeeded.
	StoreGen::SupervisorPrescanTransport* m_prescan_supervisor = nullptr;
	// Written by the prescan's build workers, read by anyone.
	struct PrescanBuildCounters {
		std::atomic<uint64_t> builds {0};
		std::atomic<uint64_t> worker_builds {0};
		std::atomic<uint64_t> in_scope {0};
		std::atomic<uint64_t> in_hold {0};
		std::atomic<uint64_t> total_us {0};
		std::atomic<uint64_t> in_scope_us {0};
		std::atomic<uint64_t> in_hold_us {0};
	};
	PrescanBuildCounters m_prescan_builds;
	// A test changes the preparer's settings here before EnablePrescan makes it, such as a memory
	// reader for host memory.
	std::function<void(PrescanPreparerConfig&)> m_prescan_tweak;
	// A test hook every prescan build calls after it sampled the stall state and before it compiles,
	// on whichever thread runs it, so a test can open a Scope in the middle of a build. A worker's
	// prescan_running unit is already counted when it runs, so the hook can read it. Set before
	// EnablePrescan and never changed after. Empty in production.
	std::function<void()> m_prescan_build_hook;
	// A test hook every prescan build calls in its start sample, between reading the Scope count and
	// the Scope flag, so a test can open a Scope at the one moment the read order decides. Set before
	// EnablePrescan and never changed after. Empty in production.
	std::function<void()> m_prescan_sample_hook;
	// The compute module the GPU thread last compiled itself with the prescan on, kept from the
	// lookup's miss until its pipeline's Scope opens so that stop can be charged to PrescanInFlight.
	// The unit is the module's prescan unit and the flag says whether the preparer had it in flight
	// at the miss. Only the GPU thread notes them, and every compute lookup clears them, as does
	// picking the cause. A lookup whose dispatch never reaches a pipeline compile leaves its note
	// until the next compute lookup, and the id check keeps it from naming another module.
	uint64_t m_inline_compute_id        = 0;
	uint64_t m_inline_compute_unit      = 0;
	bool     m_inline_compute_in_flight = false;
	// What else the miss read, kept with the note for the compile's reason. The reason the miss
	// gives if the unit turns out not in flight, the unit's verdict, and whether the verdict was
	// Built while the generated inbox had modules waiting. Only the GPU thread touches them.
	PrescanMiss        m_inline_compute_unposted  = PrescanMiss::NeverPosted;
	PrescanUnitVerdict m_inline_compute_verdict   = PrescanUnitVerdict::None;
	bool               m_inline_compute_undrained = false;
	// How many GPU-thread compiles were charged to PrescanInFlight by the check at the miss alone,
	// with the unit no longer in flight when the Scope opened. Written by the GPU thread, read by
	// anyone.
	std::atomic<uint64_t> m_in_flight_at_miss_only {0};
	// The counts behind PrescanMisses. Written by the GPU thread, read by anyone.
	struct PrescanMissCounters {
		std::array<std::atomic<uint64_t>, PRESCAN_MISS_COUNT>            by_reason {};
		std::array<std::atomic<uint64_t>, PRESCAN_REFUSED_VERDICT_COUNT> refused_by {};
		std::atomic<uint64_t>                                            built_undrained {0};
	};
	PrescanMissCounters m_prescan_misses;
	// A test sets the buffer depth the miss reads in place of the command processor's, which a
	// lookup outside a submission does not have. 0 reads the real one. It is read only at a miss
	// with the prescan on, so production pays nothing for it.
	uint32_t m_dispatch_depth_for_test = 0;
	// Test hooks on the GPU thread's own compute compile with the prescan on. The miss hook runs just
	// after the in-flight check at the lookup's miss and before translation, and the Scope hook runs
	// inside the pipeline compile's Scope. Set before the lookup. Empty in production.
	std::function<void()> m_inline_miss_hook;
	std::function<void()> m_inline_scope_hook;

	template <typename InputInfo>
	const ShaderModule* GetProgram(const ShaderParams& params, InputInfo& input_info,
	                               uint32_t& push_data_cursor);
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions  options,
	                            ShaderRecompiler::TranslateResult translated,
	                            uint32_t                          push_data_cursor);
	// The id of a module compiled from these inputs. The same inputs give the same id in every
	// run, and the shader store keys its module records by it, so it must never change for inputs
	// an existing store already holds. This forwards to ShaderModuleId in graphics/shader/moduleId.h,
	// which everything that makes a module must take its id from.
	[[nodiscard]] static uint64_t ModuleId(const ProgramKey& key, uint32_t push_data_start_dword,
	                                       const ShaderRecompiler::IR::ResourceSpecialization& specialization,
	                                       std::span<const uint32_t>                           function_code);
	void InitializeDriverCache();
	void CheckpointDriverCache();

	// Fills the key a draw's pipeline is stored under. The stages are the draw's compiled modules.
	[[nodiscard]] GraphicsPipelineKey MakeGraphicsPipelineKey(const CommandBuffer&   command,
	                                                          const DrawRenderState& state,
	                                                          const Stages&          stages) const;
	// Creates the pipeline on this thread and stores it, as every miss does without background builds.
	// The cause names what a hold of guest audio during the compile is charged to.
	Pipeline& CreateGraphicsPipelineHere(Pipeline& slot, const GraphicsPipelineKey& key,
	                                     const Stages& stages, Common::GpuStall::HoldCause cause);
	// What AcquireBuiltPipeline tells its caller about a draw that missed its pipeline. The caller
	// sets timed when the report is on, and start is then read just before the draw's fate is chosen,
	// so the drain of other keys' builds is left out of its wait. blocker is the reason the draw could
	// not be deferred, or null.
	struct MissedDraw {
		bool                                  timed   = false;
		const char*                           blocker = nullptr;
		std::chrono::steady_clock::time_point start {};
	};
	// The background-build side of AcquireGraphicsPipeline. `missed` is filled in for a draw that
	// missed its pipeline and left alone otherwise.
	[[nodiscard]] Pipeline* AcquireBuiltPipeline(const GraphicsPipelineKey& key,
	                                             const Stages&              stages,
	                                             const DrawRenderState&     state,
	                                             const PipelineSkipPolicy*  policy,
	                                             DeferredDraw*              deferred,
	                                             MissedDraw&                missed);
	// Reports a draw that could not be deferred and then waited for its pipeline, with the time from
	// the miss until the pipeline was there.
	static void NoteIneligibleWait(const MissedDraw& missed, const Stages& stages);
	// The reason a missing pipeline's draw cannot become a hole, or null when it can. The draw's own
	// state comes from the policy, and the shaders add the image heap and, unless
	// --deferred-non-push-draws is on, the descriptor kind.
	[[nodiscard]] const char* DeferBlocker(const GraphicsPipelineKey& key, const Stages& stages,
	                                       const PipelineSkipPolicy* policy) const;
	// What a draw that missed its pipeline does. `may_skip` is set when the S2 rules allow a skip, and
	// the reason the draw cannot be deferred, if any, is counted whatever the fate turns out to be and
	// returned in `blocker`, which is null otherwise.
	[[nodiscard]] DrawFate ChooseMissedDrawFate(const GraphicsPipelineKey& key, const Stages& stages,
	                                            const PipelineSkipPolicy* policy,
	                                            const DeferredDraw* deferred, bool& may_skip,
	                                            const char*& blocker);
	// Notes a deferred draw the S2 rules would have skipped, or refused under `refusal`, to report how
	// long it waited once its build is done.
	void NoteDeferredWouldSkip(const std::shared_ptr<GraphicsBuild>& build, const char* refusal = nullptr);
	// Notes a deferred draw for the report by what the S2 rules said of it, nothing when they were not
	// asked, which is without --async-pipelines. Returns the verdict for the draw's hole.
	[[nodiscard]] const char* NoteDeferredSkipVerdict(const std::shared_ptr<GraphicsBuild>& build,
	                                                  const PipelineSkipPolicy* policy, bool may_skip);
	// Reports the draws noted by NoteDeferredWouldSkip, which a resolve has just waited for.
	void ReportDeferredWouldSkip();
	// Queues the build of a key that has none and stores its record.
	void RequestBuild(const GraphicsPipelineKey& key, const Stages& stages);
	// Runs on a worker, or on the GPU thread when it claims the job.
	void RunBuild(const std::shared_ptr<GraphicsBuild>& build);
	// Waits for, or runs, a queued build and takes its pipeline. With resolve_waited_ms set the caller
	// owns the stall scope, the timing and the driver cache save, as a resolve of deferred draws does,
	// and a wait for a running build is added to it instead of being recorded here. Otherwise the
	// cause names what a hold of guest audio during the wait is charged to. Every caller names one.
	// With resolve_waited_ms set no Scope is made here, so the cause is not read, and the caller's
	// own Scope carries its cause.
	Pipeline& FinishBuild(const std::shared_ptr<GraphicsBuild>& build,
	                      const PipelineBuildQueue::JobPtr&      job,
	                      Common::GpuStall::HoldCause            wait_cause,
	                      double*                                resolve_waited_ms = nullptr);
	// The pipeline of one key of a resolve. See AcquireDeferredPipelines.
	[[nodiscard]] Pipeline AcquireDeferredPipeline(const GraphicsPipelineKey& key,
	                                               double&                    waited_ms);
	// AcquireDeferredPipeline, timed, keeping a key with a build whose acquire took longer than
	// `critical_ms` as the critical one and raising `critical_ms` to it.
	void AcquireTimedDeferredPipeline(const GraphicsPipelineKey& key, double& waited_ms,
	                                  std::vector<Pipeline>& built, double& critical_ms,
	                                  DeferredAcquireStats& stats);
	// Moves a finished build's pipeline into m_graphics_pipelines. GPU thread only.
	Pipeline& AdoptBuild(std::shared_ptr<GraphicsBuild> build);
	// Takes every finished build the workers have published.
	void DrainPublishedBuilds();
	// Creates the compute pipelines of the loaded modules, or with --background-warm-up queues them
	// in the order their records appear in the shader store.
	void AddComputeWarmUp(const ModuleMap& modules, std::vector<Common::UniqueFunction<void>>& jobs);
	// Starts the warm-up build of a compute module on the warm queue. It never stores a record,
	// since the warm-up only builds what the store already holds. A module that generation made
	// passes its row, which the build stamps with the time it finished and how long it took. The
	// row outlives the build, since the generation state lives as long as the cache. The lane says
	// whether workers build it with the rest or only when nothing else is waiting.
	void StartWarmCompute(const ShaderModule& module, StoreGen::GenRow* row = nullptr,
	                      PipelineBuildQueue::Lane lane = PipelineBuildQueue::Lane::Front);
	// Starts the build of a prescan module on the prescan's warm set. Like StartWarmCompute it never
	// stores a record, and the module's id stays out of m_generated_warm_ids, so a stop for it is
	// charged to the prescan's causes. It counts the builds that overlapped a stop of the GPU
	// thread or a hold of guest audio, and a worker's build counts toward the prescan_running
	// underrun tag while it runs.
	void StartPrescanCompute(const ShaderModule& module);
	// What a prescan build reads of the stall state at its start and end. Defined in the source.
	struct StallSample;
	// Counts one finished prescan build, with the stall state from its start and how long it took.
	// Only a worker's build can overlap a stop or a hold, since a build on the GPU thread is the stop.
	void CountPrescanBuild(bool on_worker, const StallSample& before,
	                       std::chrono::steady_clock::duration elapsed);
	// The warm set that has a build pending for the module, or null. A module is on one set only,
	// since the prescan's modules are adopted only when their program has no module with the id.
	[[nodiscard]] WarmBuildSet<uint64_t, Pipeline>* WarmSetPending(uint64_t id) const;
	// Moves the module's warm-up pipeline into m_compute_pipelines, waiting for it or building it
	// here if it is not done. Null when no warm-up build is pending for the module.
	[[nodiscard]] Pipeline* TakeWarmCompute(const ShaderComputeInputInfo& input_info);
	// The part of TakeWarmCompute for a build that is queued or running on `set`. It runs or waits
	// for the build inside the stall and the timing that the GPU thread owes for that.
	[[nodiscard]] std::optional<WarmBuildSet<uint64_t, Pipeline>::Taken>
	TakeUnfinishedWarmCompute(uint64_t id, WarmBuildSet<uint64_t, Pipeline>& set);
	// Counts and logs a warm-up compute build that stopped the GPU thread for `stopped_ms`, and says
	// whether the queue was held when it did. Builds already done when taken stop nothing and are not
	// counted.
	void NoteWarmComputeStop(PipelineBuildQueue::Acquired how, double stopped_ms, bool queue_held);
	// Stops the warm queue and the prescan's set and destroys the pipelines they built that nobody
	// took.
	void StopWarmBuilds();
	// Stops one warm set, destroys the pipelines in it that nobody took and resets it.
	void StopWarmSet(std::unique_ptr<WarmBuildSet<uint64_t, Pipeline>>& set);
	// The preparer's settings, taken from this device and Config, with the skip keys of the modules
	// the cache already holds.
	[[nodiscard]] PrescanPreparerConfig
	MakePrescanConfig(StoreGen::PrescanTransportFactory make_transport);
	// pipelinePrescan.cpp. The prescan's skip key of a module of the program cache, through the one
	// function the preparer makes it with.
	[[nodiscard]] static uint64_t SkipKeyOf(const ProgramKey& key, const ShaderModule& module);
	// The skip keys of every compute module without GPU-selected calls in the program cache, which
	// seed a new preparer. Only the thread that owns the program cache calls it.
	[[nodiscard]] std::vector<uint64_t> KnownModuleKeys() const;
	// Tells the prescan's preparer the skip key of a compute module the cache took in from the
	// shader store or the generator, so it sends no Compile for that module. It does nothing without
	// the prescan, whose seed covers what was taken in before it started, and nothing once the
	// preparer stopped. Only the thread that owns the program cache calls it.
	void NoteKnownModule(const ProgramKey& key, const ShaderModule& module);
	// Forgets the inline compute compile noted so far. Called at the top of every compute lookup,
	// hits included, and once the compile's cause is picked. Inline, since it runs on every lookup.
	void ResetInlineCompute() noexcept {
		m_inline_compute_id        = 0;
		m_inline_compute_unit      = 0;
		m_inline_compute_in_flight = false;
		m_inline_compute_unposted  = PrescanMiss::NeverPosted;
		m_inline_compute_verdict   = PrescanUnitVerdict::None;
		m_inline_compute_undrained = false;
	}
	// pipelinePrescan.cpp. Notes, at a compute lookup's miss on the GPU thread, the prescan unit of
	// the program in m_lookup_key and whether the preparer has it in flight. Translation may take
	// hundreds of ms outside any Scope, and the helper may answer meanwhile, so the answer at the
	// miss counts as well as the one when the Scope opens. It also notes the reason the miss gives
	// should the unit not be in flight, from the unit's verdict, whether the prescan had stopped,
	// and the dispatch's depth, indirect address and queue. Those are read here only, so a verdict
	// that appears during translation is lateness, which the in-flight check at the Scope catches
	// if the unit was sent meanwhile. It does nothing without the prescan or on another thread.
	void NoteInlineComputeMiss(const ShaderComputeInputInfo& input_info);
	// Keeps the id of the module a noted miss compiled, for InlineComputeCause.
	void NoteInlineComputeModule(uint64_t module_id) noexcept;
	// The hold cause of the GPU thread's pipeline compile of `module_id`, picked before its Scope
	// opens with lock-free reads only. It is the cause of the compile's PrescanMiss. That is
	// ModulePresent when the module is not the one the last noted miss compiled, InFlight when the
	// preparer had its unit in flight at the miss or has it now, and the reason the miss noted
	// otherwise. The table is kept by unit, so a request for another specialization of the same
	// unit counts too. A stop charged by the check at the miss alone is counted in
	// m_in_flight_at_miss_only. With the prescan on, the reason is counted once here and nowhere
	// else, so the counts add up to the inline compute Scopes. It forgets the note.
	[[nodiscard]] Common::GpuStall::HoldCause InlineComputeCause(uint64_t module_id) noexcept;
	// pipelinePrescan.cpp. Counts one inline compute compile under its reason, and a refused one
	// under its verdict too. Relaxed atomics with release fences between, so a reader that sees a
	// narrower count also sees the wider one it is part of.
	void NoteDispatchCompile(PrescanMiss miss, PrescanUnitVerdict verdict, bool undrained) noexcept;
	// pipelineGeneration.cpp. Makes the module of a prescan record on the preparer's thread and posts
	// it, as PostGeneratedModule does for the generator. The record's bytes, as the helper encoded
	// them, travel with the module so its first use can store them without encoding them again. Empty
	// bytes keep nothing.
	void PostPrescanModule(DecodedModuleRecord&& record, std::vector<uint8_t>&& record_bytes);
	// Links an adopted prescan module's record into the list of records waiting for their first use,
	// or drops it when there is no shader store, since prescan modules persist only with one. Only
	// the thread that owns the program cache calls it.
	void RetainPrescanRecord(ShaderModule& module);
	// The check a lookup makes on the module it found. Every module but a prescan one waiting for
	// its first use has no record, so the cost is one test. Inline so the test needs no call.
	void StoreOnFirstUse(const Program& program, ShaderModule& module) {
		if (module.retained) {
			StoreRetainedRecord(program, module);
		}
	}
	// Appends a prescan module's record to the shader store at the module's first use and frees it,
	// so the module is stored once. A record the cap dropped is only freed, and so is the record of
	// a program with GPU-selected calls, which CompileModule never stores either. Constant time
	// apart from the append. Only the thread that owns the program cache calls it.
	void StoreRetainedRecord(const Program& program, ShaderModule& module);

	// Makes the state of the generated modules intake. Called once, before the generator starts.
	void EnableStoreGeneration();
	// Starts the shader store generator for the title whose store is at `store_path`, which WarmUp
	// calls after the player's own records were queued. The generator writes next to the store.
	void StartStoreGeneration(const std::filesystem::path& store_path, std::span<const uint8_t> header);
	// Builds the module of a record on the generator's thread. Nothing when its plan cannot be
	// decoded.
	[[nodiscard]] std::optional<GeneratedModule>
	MakeGeneratedModule(DecodedModuleRecord&& record, const StoreGen::GeneratedOrder& order = {});
	// The same for any producer, with that producer's own intake state.
	[[nodiscard]] std::optional<GeneratedModule>
	MakeGeneratedModule(DecodedModuleRecord&& record, const StoreGen::GeneratedOrder& order,
	                    GeneratedIntake& intake);
	// Puts a module in the inbox, or destroys it when the intake has stopped.
	void PushGenerated(GeneratedModule&& item);
	// Takes posted modules into the program cache and queues their builds. GPU thread only. It takes
	// a bounded number, and stops after a bounded time, so a drain is always short.
	void DrainGenerated();
	// Copies what the drain counted, and with the prescan on the prescan's record counts, under the
	// state's mutex, where the report's writer reads them from its own thread. GPU thread only.
	void PublishDrain();
	// Gives the first-run report the source of the generation's figures. StartStoreGeneration calls
	// it, and a test of the stand-in intake does the same.
	void RegisterGenerationReport();
	// pipelinePrescan.cpp. The prescan's figures for the first-run report. Live, the drain and
	// record figures come from what the drain last published, and the writer's thread may call it
	// at any time. Final, the drain figures come from the planners themselves, which is only exact
	// once nothing drains. Call it only with the prescan on. It holds no lock across another.
	[[nodiscard]] Common::FirstRunReport::PrescanCacheReport
	BuildPrescanReport(bool final_figures) const;
	// pipelinePrescan.cpp. Publishes the prescan's record counts once the stop has released the
	// list, so the unused ones are final, and hands the report the final figures in place of the
	// source. StopStoreGeneration calls it once nothing posts or drains. It does nothing without
	// the prescan.
	void FinishPrescanReport();
	// The figures of the generation for the first-run report. Live, they come from what the drain
	// last published, and the writer's thread may call it at any time. Final, they come from the
	// planner itself, which is only exact once nothing drains. It takes the state's mutex briefly
	// and holds no lock across anything else.
	[[nodiscard]] Common::FirstRunReport::ShaderStoreGenerationReport
	BuildGenerationReport(bool final_figures) const;
	// The rows of generated modules, refused modules and misses, copied under the state's mutex.
	// The report's own rows are not in it.
	[[nodiscard]] StoreGen::GenerationRowInputs SnapshotGenerationRows() const;
	// Takes one module in, or drops it as the planner says. Either way a generator module leaves a
	// row for the report, and a prescan module is counted.
	void AdoptGenerated(GeneratedModule&& item);
	using ProgramIterator = std::unordered_map<ProgramKey, Program, ProgramKeyHash>::iterator;
	// The program a module joins, made with the module's plan when it is new.
	Program* InstallGeneratedProgram(GeneratedModule& item, StoreGen::DrainFacts& facts,
	                                 ProgramIterator& entry);
	// Puts an adopted module in its program and queues its build on the set of its producer.
	void TakeGeneratedIn(GeneratedModule&& item, Program& program);
	// Records that the runtime had to translate a compute program with generation on, for the
	// report's classes. id is 0 for a program that the runtime skips. Any thread.
	void NoteGeneratedMiss(uint64_t code_hash, uint64_t id, bool has_calls, bool skip_dispatch);
	// The row of a module that was taken in, which its warm build stamps. And the row of one
	// refused.
	StoreGen::GenRow* NoteAdopted(const GeneratedModule& item);
	void              NoteRefused(const GeneratedModule& item, StoreGen::DrainVerdict verdict);
	// Destroys the Vulkan handles of a module that was never taken in.
	void DestroyGeneratedHandles(GeneratedModule& item);
	// The runtime translated a program that generation made, so its plan is compared with the plan
	// the program was made with.
	void CheckGeneratedPlan(Program& program, const ShaderRecompiler::TranslateResult& translated);

	// pipelineSerialization.cpp
	bool LoadRecord(std::span<const uint8_t>& records);
	bool LoadModuleRecord(DecodedModuleRecord&& decoded);
	void StoreModule(const ProgramKey& key, const Program& program, const ShaderModule& module,
	                 const std::vector<uint32_t>& spirv);
	void StorePipeline(const GraphicsPipelineKey& key);
	void StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout);
	// Every module by id, with id 0 naming an absent stage.
	[[nodiscard]] ModuleMap MapModules() const;
	// False when a stage id of the key is not in `modules`.
	bool CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
	                            const ModuleMap& modules);
};

// The shader store of the running title; empty unless --shader-cache is on in a Release build
// without debug dumps, which change the SPIR-V.
std::filesystem::path ShaderStorePath();
// Host capabilities that change the emitted SPIR-V.
ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics);
// The subgroup size a compute shader runs with on this host, which the compute static key depends on.
// The pipeline cache and the shader store worker's request both use it.
uint32_t HostComputeSubgroupSize(const GraphicContext& graphics);

// The descriptor set numbers of the split layout. The image heap keeps ImageHeapSet (1), which is
// hard-coded in the SPIR-V, so the VS set is below it and the PS set above it.
inline constexpr uint32_t SPLIT_VERTEX_SET = 0;
inline constexpr uint32_t SPLIT_PIXEL_SET  = 2;
inline constexpr uint32_t SPLIT_SET_COUNT  = 3;

// Whether a graphics pipeline is built with the split layout. That needs the library mode to be on
// and a pipeline that libraries could build. The layout code and the prediction below both ask this.
[[nodiscard]] bool UsesSplitLayout(const GraphicContext& graphics, const GraphicsPipelineKey& key,
                                   const std::array<const ShaderModule*, 4>& stages);

// Whether a pipeline over these stages uses push descriptors, answered from the shaders alone by the
// code that builds the pipeline layout, so a draw can be judged before its pipeline is built. With
// the split layout it also depends on the key, which says whether that layout is used.
// Stages are indexed like GraphicsPipelineKey::stage_ids.
[[nodiscard]] bool PredictPushDescriptors(const GraphicContext&                       graphics,
                                          const GraphicsPipelineKey&                 key,
                                          const std::array<const ShaderModule*, 4>& stages);

// Makes the PS module of the split layout, or returns null when the library mode is off or `stage`
// is not a pixel shader. The SPIR-V is the one the plain module was made from.
[[nodiscard]] vk::ShaderModule CompileSplitPixelModule(const GraphicContext& graphics,
                                                       ShaderType stage,
                                                       std::span<const uint32_t> spirv);

// Judges the bindings of one descriptor set against the device limits. Returns the text of the
// error the layout code stops with, or an empty string when they fit, and sets the descriptor count.
[[nodiscard]] std::string DescriptorLimitError(const vk::PhysicalDeviceLimits&                limits,
                                               std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                               uint64_t& descriptor_count);

// The same for the VS set and the PS set of the split layout, which are judged together since the
// limits on set contents count the whole pipeline layout.
[[nodiscard]] std::string SplitLayoutLimitError(const vk::PhysicalDeviceLimits&                limits,
                                                std::span<const vk::DescriptorSetLayoutBinding> vertex,
                                                std::span<const vk::DescriptorSetLayoutBinding> pixel);

// Destroys what a pipeline owns, whichever layout it has. Handles that were never made are skipped.
// The set layouts of a split pipeline are shared and belong to the DescriptorSetLayoutCache.
void DestroyPipelineObjects(vk::Device device, const PipelineCache::Pipeline& pipeline);
// How many set layouts DestroyPipelineObjects has destroyed so far, for tests. Layouts of the
// DescriptorSetLayoutCache never count, since only the cache destroys them.
[[nodiscard]] uint64_t PipelineSetLayoutDestroyCount();

// Stages are indexed like GraphicsPipelineKey::stage_ids. The set layouts of a split pipeline come
// from `set_layouts`, which must outlive the pipeline. With `libraries` given and the library mode on,
// an eligible pipeline is linked from the libraries of that cache, which must outlive the pipeline
// too, and any other pipeline, or one whose libraries the driver refused, is built whole.
void CreatePipelineInternal(GraphicContext& graphics, DescriptorSetLayoutCache& set_layouts,
                            PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache,
                            GraphicsLibraryCache*                      libraries = nullptr);
void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
