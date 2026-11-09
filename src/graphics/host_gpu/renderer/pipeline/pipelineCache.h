#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINECACHE_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "common/file.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/pipeline/descriptorSetLayoutCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"
#include "graphics/host_gpu/renderer/pipeline/warmBuildSet.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <filesystem>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <span>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;
struct ShaderParams;
class CommandBuffer;
class GraphicsLibraryCache;

namespace ShaderRecompiler {
struct CompileOptions;
struct ShaderSource;
struct TranslateResult;
} // namespace ShaderRecompiler

namespace ShaderRecompiler::IR {
class ImageHeapSource;
} // namespace ShaderRecompiler::IR

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
};

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

// The longest the warm-up queue is held while deferred draws wait for their pipelines. A resolve
// waits for the parallel builds of one submission, and the measured wait of the worst frame was
// 3.8 to 7.5 s (4 to 6 workers, 1.0 to 1.34 times contention, design section 7). Ten seconds is
// above that, so a normal resolve ends the pause first with Resume, while a resolve that never
// comes cannot starve the warm-up for longer than this.
inline constexpr std::chrono::seconds WARM_UP_PAUSE_LIMIT {10};

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
	// A deferred draw that the S2 rules would have skipped, kept until its resolve reports it. Only
	// the GPU thread touches it.
	struct WouldSkip {
		std::shared_ptr<GraphicsBuild>        build;
		std::chrono::steady_clock::time_point deferred_at;
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
	// Module ids in the order their records appear in the shader store, one entry per record. Filled
	// while WarmUp loads the store and emptied once the warm-up has queued its builds.
	std::vector<uint64_t>                                                        m_module_record_order;
	// Zero sizes the warm queue from the hardware threads. A test sets one to make the queue run
	// jobs strictly in order.
	uint32_t                                                                     m_warm_worker_count_override = 0;

	template <typename InputInfo>
	const ShaderModule* GetProgram(const ShaderParams& params, InputInfo& input_info,
	                               uint32_t& push_data_cursor);
	ShaderModule& CompileModule(Program& program, const ProgramKey& key,
	                            ShaderRecompiler::CompileOptions  options,
	                            ShaderRecompiler::TranslateResult translated,
	                            uint32_t                          push_data_cursor);
	void InitializeDriverCache();
	void CheckpointDriverCache();

	// Fills the key a draw's pipeline is stored under. The stages are the draw's compiled modules.
	[[nodiscard]] GraphicsPipelineKey MakeGraphicsPipelineKey(const CommandBuffer&   command,
	                                                          const DrawRenderState& state,
	                                                          const Stages&          stages) const;
	// Creates the pipeline on this thread and stores it, as every miss does without background builds.
	Pipeline& CreateGraphicsPipelineHere(Pipeline& slot, const GraphicsPipelineKey& key,
	                                     const Stages& stages);
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
	// Notes that the S2 rules would have skipped a draw that was deferred, to report how long it
	// waited once its build is done.
	void NoteDeferredWouldSkip(const std::shared_ptr<GraphicsBuild>& build);
	// Reports the draws noted by NoteDeferredWouldSkip, which a resolve has just waited for.
	void ReportDeferredWouldSkip();
	// Queues the build of a key that has none and stores its record.
	void RequestBuild(const GraphicsPipelineKey& key, const Stages& stages);
	// Runs on a worker, or on the GPU thread when it claims the job.
	void RunBuild(const std::shared_ptr<GraphicsBuild>& build);
	// Waits for, or runs, a queued build and takes its pipeline. With resolve_waited_ms set the caller
	// owns the stall scope, the timing and the driver cache save, as a resolve of deferred draws does,
	// and a wait for a running build is added to it instead of being recorded here.
	Pipeline& FinishBuild(const std::shared_ptr<GraphicsBuild>& build,
	                      const PipelineBuildQueue::JobPtr&      job,
	                      double*                                resolve_waited_ms = nullptr);
	// The pipeline of one key of a resolve. See AcquireDeferredPipelines.
	[[nodiscard]] Pipeline AcquireDeferredPipeline(const GraphicsPipelineKey& key,
	                                               double&                    waited_ms);
	// Moves a finished build's pipeline into m_graphics_pipelines. GPU thread only.
	Pipeline& AdoptBuild(std::shared_ptr<GraphicsBuild> build);
	// Takes every finished build the workers have published.
	void DrainPublishedBuilds();
	// Creates the compute pipelines of the loaded modules, or with --background-warm-up queues them
	// in the order their records appear in the shader store.
	void AddComputeWarmUp(const ModuleMap& modules, std::vector<Common::UniqueFunction<void>>& jobs);
	// Starts the warm-up build of a compute module on the warm queue. It never stores a record,
	// since the warm-up only builds what the store already holds.
	void StartWarmCompute(const ShaderModule& module);
	// Moves the module's warm-up pipeline into m_compute_pipelines, waiting for it or building it
	// here if it is not done. Null when no warm-up build is pending for the module.
	[[nodiscard]] Pipeline* TakeWarmCompute(const ShaderComputeInputInfo& input_info);
	// The part of TakeWarmCompute for a build that is queued or running. It runs or waits for the
	// build inside the stall and the timing that the GPU thread owes for that.
	[[nodiscard]] std::optional<WarmBuildSet<uint64_t, Pipeline>::Taken>
	TakeUnfinishedWarmCompute(uint64_t id);
	// Counts and logs a warm-up compute build that stopped the GPU thread for `stopped_ms`, and says
	// whether the queue was held when it did. Builds already done when taken stop nothing and are not
	// counted.
	void NoteWarmComputeStop(PipelineBuildQueue::Acquired how, double stopped_ms, bool queue_held);
	// Stops the warm queue and destroys the pipelines it built that nobody took.
	void StopWarmBuilds();

	// pipelineSerialization.cpp
	bool LoadRecord(std::span<const uint8_t>& records);
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
