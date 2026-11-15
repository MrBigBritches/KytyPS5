#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_

#include "common/abi.h"
#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
class Shader;
} // namespace HW

struct GraphicContext;
struct DrawCallInfo;
struct DrawEmitInfo;
struct DrawIndexBufferSource;
struct DrawView;
struct MeshDrawPlan;
class DrawRecordEnv;
class RenderContext;
class CommandScheduler;
struct RenderExecutorTestAccess;

enum class DrawOffsetSource : uint8_t {
	DrawState,
	IndirectArgs,
};

// Nonzero address: the GPU reads count Vulkan indirect draw records, stride bytes apart.
struct DrawIndirectArgs {
	uint64_t              address = 0;
	uint32_t              count   = 0;
	uint32_t              stride  = 0;
	ShaderIndirectOffsets offsets;
};

struct DrawIndexArgs {
	uint32_t         index_count         = 0;
	const void*      index_addr          = nullptr;
	uint32_t         instance_count      = 0;
	uint32_t         index_type_and_size = 0;
	int32_t          base_vertex         = 0;
	uint32_t         first_instance      = 0;
	DrawOffsetSource offset_source       = DrawOffsetSource::DrawState;
	// With indirect records, index_count is INDEX_BUFFER_SIZE.
	DrawIndirectArgs indirect;
};

struct DrawAutoArgs {
	uint32_t         vertex_count   = 0;
	uint32_t         instance_count = 0;
	uint32_t         first_vertex   = 0;
	uint32_t         first_instance = 0;
	DrawOffsetSource offset_source  = DrawOffsetSource::DrawState;
	DrawIndirectArgs indirect;
};

struct SubmitInfo {
	static constexpr uint32_t MaxSemaphores = 3;

	std::array<vk::Semaphore, MaxSemaphores>          wait_semaphores {};
	std::array<uint64_t, MaxSemaphores>               wait_ticks {};
	std::array<vk::PipelineStageFlags, MaxSemaphores> wait_stages {};
	std::array<vk::Semaphore, MaxSemaphores>          signal_semaphores {};
	std::array<uint64_t, MaxSemaphores>               signal_ticks {};
	uint32_t                                          num_wait_semaphores   = 0;
	uint32_t                                          num_signal_semaphores = 0;

	void AddWait(vk::Semaphore semaphore, uint64_t tick = 1,
	             vk::PipelineStageFlags stage = vk::PipelineStageFlagBits::eAllCommands) {
		EXIT_IF(semaphore == nullptr || num_wait_semaphores >= MaxSemaphores);
		wait_semaphores[num_wait_semaphores] = semaphore;
		wait_ticks[num_wait_semaphores]      = tick;
		wait_stages[num_wait_semaphores++]   = stage;
	}

	void AddSignal(vk::Semaphore semaphore, uint64_t tick = 1) {
		EXIT_IF(semaphore == nullptr || num_signal_semaphores >= MaxSemaphores);
		signal_semaphores[num_signal_semaphores] = semaphore;
		signal_ticks[num_signal_semaphores++]    = tick;
	}
};

class CommandBuffer {
public:
	~CommandBuffer() = default;

	KYTY_CLASS_NO_COPY(CommandBuffer);

	[[nodiscard]] bool IsInvalid() const noexcept { return m_buffer == nullptr; }

	void BeginRendering(const RenderState& state) const;
	void EndRendering() const;

	// Every command is recorded through the handle, so asking for it marks the buffer as used.
	[[nodiscard]] vk::CommandBuffer Handle() const noexcept {
		m_used = true;
		return m_buffer;
	}
	[[nodiscard]] GraphicContext&   GetGraphics() const noexcept { return m_graphics; }
	[[nodiscard]] RenderContext&    GetContext() const noexcept { return m_context; }
	[[nodiscard]] HW::Context&      GetRegisters() const noexcept { return *m_registers; }
	[[nodiscard]] HW::UserConfig&   GetUserConfig() const noexcept { return *m_user_config; }
	[[nodiscard]] HW::Shader&       GetShaders() const noexcept { return *m_shaders; }

private:
	explicit CommandBuffer(CommandScheduler& scheduler);
	void Bind(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders) noexcept {
		m_registers   = &registers;
		m_user_config = &user_config;
		m_shaders     = &shaders;
	}

	void Begin();
	void End() const;

	RenderContext&      m_context;
	GraphicContext&     m_graphics;
	vk::CommandBuffer   m_buffer = nullptr;
	mutable RenderState m_render_state;
	mutable bool        m_rendering   = false;
	// Whether anything may have been recorded since the buffer began. It errs toward true.
	mutable bool        m_used        = false;
	HW::Context*        m_registers   = nullptr;
	HW::UserConfig*     m_user_config = nullptr;
	HW::Shader*         m_shaders     = nullptr;

	friend class CommandScheduler;
	friend struct CommandBufferTestAccess;
};

namespace Pssr {
struct Request;
}

class RenderExecutor {
public:
	explicit RenderExecutor(RenderContext& context): m_context(context) {}
	~RenderExecutor();
	KYTY_CLASS_NO_COPY(RenderExecutor);

	void DispatchDirect(CommandBuffer& buffer, uint32_t thread_group_x, uint32_t thread_group_y,
	                    uint32_t thread_group_z, uint32_t mode, bool async_compute);
	void DispatchIndirect(CommandBuffer& buffer, uint64_t args_addr, uint32_t mode,
	                      bool async_compute);
	void ProcessPssr(CommandBuffer& buffer, const Pssr::Request& request);

	void PrepareBindings(const ShaderStageRuntime& runtime, PreparedBindings& prepared);
	void                           FindBuffers(std::span<PreparedBindings* const> stages);
	void                           RebindBuffers(PreparedBindings& bindings);
	void                           RebindImages(PreparedBindings& bindings);
	void CommitBindings(CommandBuffer& buffer, vk::PipelineBindPoint pipeline_bind_point,
	                    const PipelineCache::Pipeline&     pipeline,
	                    std::span<PreparedBindings* const> bindings);
	/// Moves the images the stages sample or store into their layouts and records the GDS barriers.
	/// This is the eager half of CommitBindings. It changes CPU-tracked layouts, so it runs in draw
	/// order.
	void TransitBindingImages(CommandBuffer& buffer, std::span<PreparedBindings* const> bindings);
	/// Writes push constants and descriptors for stages whose images were already transited.
	/// This is the half that may run later. It reads only the given modules, never a stage runtime.
	void WriteBindingDescriptors(vk::CommandBuffer vk_buffer,
	                             vk::PipelineBindPoint pipeline_bind_point,
	                             const PipelineCache::Pipeline& pipeline,
	                             std::span<PreparedBindings* const> bindings,
	                             std::span<const ShaderModule* const> modules);
	/// The same, with the address buffers and the scratch the caller provides. This form never
	/// reaches the buffer cache and never uses the executor's own scratch.
	void WriteBindingDescriptors(vk::CommandBuffer vk_buffer,
	                             vk::PipelineBindPoint pipeline_bind_point,
	                             const PipelineCache::Pipeline& pipeline,
	                             std::span<PreparedBindings* const> bindings,
	                             std::span<const ShaderModule* const> modules,
	                             const AddressBufferInfos& address_buffers,
	                             DescriptorScratch& scratch);
	/// The BDA page table and fault buffer views of the stages that bind them.
	[[nodiscard]] AddressBufferInfos CaptureAddressBuffers(std::span<const ShaderModule* const> modules);
	/// Records the commands of a prepared draw into `command`. This is the half of a draw that needs
	/// its pipeline. It reads only the view and the pipeline, never the registers, the caches or an
	/// image's tracked layout, and it never transits an image. `env` decides how the draw's
	/// rendering begins and what surrounds its draw call.
	void RecordDraw(vk::CommandBuffer command, const DrawView& view,
	                const PipelineCache::Pipeline& pipeline, DrawRecordEnv& env);
	[[nodiscard]] TextureBinding ResolveTexture(const ShaderRecompiler::IR::ImageResource&   resource,
	                                            const ShaderRecompiler::IR::DescriptorValue& value);

private:
	// False, with nothing drawn, when the draw's indirect records need CPU counts.
	bool DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args);
	bool DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args);

	struct GraphicsBindings {
		std::array<PreparedBindings, 3> vertex;
		PreparedBindings                pixel;
	};

	void PrepareGraphicsBindings(std::span<PreparedBindings* const> stages,
	                             std::span<RenderColorInfo> colors);
	void ResolveRenderColorTarget(CommandBuffer& buffer, RenderColorInfo& target,
	                              uint32_t render_target_slot, bool ignore_target_mask = false,
	                              bool exact_format = false);
	void ResolveRenderDepthTarget(CommandBuffer& buffer, RenderDepthInfo& target);
	[[nodiscard]] bool DepthStencilCopy(CommandBuffer& buffer);
	[[nodiscard]] bool CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                   const DrawEmitInfo& emit, DrawRenderState& state);
	[[nodiscard]] bool PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                          DrawRenderState& state);
	void                      ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
	                                              DrawRenderState& state, const DrawEmitInfo& emit,
	                                              const DrawIndexBufferSource& index_source);
	/// The eager half of a draw, everything that must happen in order when the draw is met. It
	/// returns the draw's pipeline, or null when there is nothing to record. With
	/// `bind_vertex_buffers` the vertex buffers are bound before the image transitions. With
	/// `deferred` given, a draw whose pipeline is building may wait for it instead. It then returns
	/// null with `deferred` set, and leaves the vertex buffers unbound.
	[[nodiscard]] const PipelineCache::Pipeline*
	PrepareDraw(CommandBuffer& buffer, const DrawCallInfo& draw, DrawRenderState& state,
	            const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
	            bool bind_vertex_buffers, DrawView& view,
	            PipelineCache::DeferredDraw* deferred = nullptr);
	/// Turns a prepared draw that waits for its pipeline into a hole in the scheduler's command
	/// stream, after applying any clear it consumed. Nothing fetched from the command buffer
	/// before the call may be used after it. `skip_verdict` goes with the hole for the report.
	void CreateDrawHole(CommandBuffer& buffer, DrawView& view, const GraphicsPipelineKey& key,
	                    const char* skip_verdict);
	/// Resolves the bindings of every stage the draw runs and lists them in the view.
	void PrepareDrawBindings(DrawRenderState& state, DrawView& view);
	/// Records the commands a draw needs at the moment it is met, which are the image transitions
	/// and, when asked, the vertex buffer bind. It also captures the register state into the view.
	void RecordEagerDrawCommands(CommandBuffer& buffer, DrawRenderState& state,
	                             const MeshDrawPlan& mesh, const PipelineCache::Pipeline* pipeline,
	                             vk::ImageAspectFlags feedback_aspects, bool bind_vertex_buffers,
	                             DrawView& view);
	/// Records a prepared draw into the scheduler's current buffer, sharing its rendering with the
	/// draws around it. This is what a draw met in order uses.
	void RecordBlockingDraw(CommandBuffer& buffer, const DrawView& view,
	                        const PipelineCache::Pipeline& pipeline);
	[[nodiscard]] RenderState AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
	                                               uint32_t color_count, RenderDepthInfo& depth,
	                                               vk::ImageAspectFlags& feedback_aspects,
	                                               std::span<PreparedBindings* const> stages = {});
	[[nodiscard]] bool        ResolveColorTargets(CommandBuffer& buffer);
	[[nodiscard]] bool        ConsumeMetadataColorOperation(CommandBuffer& buffer);
	void                      BindImage(ImageId id, bool storage);
	void                      BindRenderTarget(ImageId id);
	void                      ResetBindings();
	[[nodiscard]] bool        TryConsumeComputeMetaClear(const ShaderComputeInputInfo& input,
	                                                     const CommandBuffer&          buffer);
	[[nodiscard]] bool TryConsumeComputeFill(const ShaderComputeInputInfo& input,
	                                        CommandBuffer& command, uint32_t group_x,
	                                        uint32_t group_y, uint32_t group_z, uint32_t mode);

	RenderContext&                        m_context;
	GraphicsBindings                     m_graphics_bindings;
	PreparedBindings                     m_compute_bindings;
	std::vector<ImageId>                  m_bound_images;
	DescriptorScratch                     m_descriptor_scratch;
	vk::DescriptorSetLayout               m_pssr_descriptor_layout = nullptr;
	vk::PipelineLayout                    m_pssr_pipeline_layout   = nullptr;
	vk::Pipeline                          m_pssr_pipeline          = nullptr;

	friend class CommandProcessor;
	friend struct RenderExecutorTestAccess;
};

[[nodiscard]] bool ResolveComputeBufferFill(const ShaderComputeInputInfo& input, uint32_t group_x,
                                            uint32_t group_y, uint32_t group_z, uint32_t mode,
                                            ShaderBufferResource& descriptor,
                                            uint32_t& packed_clear, uint64_t& size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRENDER_H_ */
