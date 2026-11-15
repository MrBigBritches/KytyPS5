#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWRECORD_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWRECORD_H_

#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <cstdint>

namespace Libs::Graphics {

struct ShaderModule;

// How many dwords the guest keeps for its user clip planes, four for each of six planes.
inline constexpr uint32_t MaxUserClipPlaneDwords = 24;

struct DrawCallInfo {
	bool     indexed        = true;
	uint32_t index_count    = 0;
	uint32_t instance_count = 0;

	[[nodiscard]] const char* Name() const { return indexed ? "DrawIndex" : "DrawIndexAuto"; }
};

// vertex_offset is the first vertex of non-indexed draws.
struct DrawEmitInfo {
	int32_t          vertex_offset  = 0;
	uint32_t         first_instance = 0;
	DrawIndirectArgs indirect;
};

struct DrawIndexBufferSource {
	uint64_t    address            = 0;
	const void* host_data          = nullptr;
	uint64_t    size               = 0;
	uint32_t    guest_element_size = 0;
};

struct PreparedIndexBuffer {
	vk::Buffer     buffer = nullptr;
	vk::DeviceSize offset = 0;
	vk::IndexType  type   = vk::IndexType::eUint16;
};

struct PreparedVertexBuffers {
	static constexpr uint32_t MaxBuffers = ShaderVertexInputInfo::RES_MAX;

	std::array<vk::Buffer, MaxBuffers>     buffers {};
	std::array<vk::DeviceSize, MaxBuffers> offsets {};
	std::array<vk::DeviceSize, MaxBuffers> sizes {};
	uint32_t                               count = 0;
};

// Everything that recording a draw needs, as plain data. PrepareDraw fills it while the draw is met,
// when the guest registers, the caches and the CPU-tracked image layouts are still the draw's own.
// RecordDraw then reads only this and the pipeline, so it can run later, into another command
// buffer, after the registers and the caches have moved on. The bindings and the dynamic state are
// borrowed from the executor and the scheduler and only stay valid until the draw is recorded, so a
// record kept for later must copy them. The bindings still hold runtime pointers into the draw's
// stack frame, which RecordDraw never reads.
struct DrawView {
	std::array<PreparedBindings*, MaxBindingStages>   bindings {};
	// The module of each stage, taken before its runtime goes away with the draw's stack frame.
	std::array<const ShaderModule*, MaxBindingStages> modules {};
	uint32_t                                          stage_count = 0;
	AddressBufferInfos                                address_buffers;

	DrawCallInfo draw;
	DrawEmitInfo emit;
	// What the draw read from the registers, kept so recording never reads them.
	Prospero::PrimitiveType primitive_type = Prospero::PrimitiveType::kNone;
	uint32_t                object_id      = 0;
	// How many dwords of mesh parameters or the object id lead the draw's push constants.
	uint32_t                                  draw_dwords      = 0;
	uint32_t                                  user_clip_planes = 0;
	std::array<uint32_t, MaxUserClipPlaneDwords> user_clip_plane_dwords {};

	// Mesh parameters. The index address and size are the guest's, for the mesh shader to read.
	bool              mesh_active        = false;
	bool              mesh_task          = false;
	uint32_t          mesh_groups        = 0;
	vk::DeviceAddress task_params        = 0;
	uint64_t          index_address      = 0;
	uint32_t          index_element_size = 0;

	PreparedVertexBuffers vertex_buffers;
	// Set when the vertex buffers were already bound before the image transitions, as a draw
	// recorded straight away does to keep its commands in their old order.
	bool                vertex_buffers_bound = false;
	PreparedIndexBuffer index_buffer;
	// The indirect arguments, when the GPU reads them.
	vk::Buffer     args_buffer = nullptr;
	vk::DeviceSize args_offset = 0;

	DynamicState*        dynamic_state = nullptr;
	RenderState          rendering;
	vk::PipelineStageFlags shader_write_stages;

	[[nodiscard]] std::span<PreparedBindings* const> Bindings() const {
		return {bindings.data(), stage_count};
	}
	[[nodiscard]] std::span<const ShaderModule* const> Modules() const {
		return {modules.data(), stage_count};
	}
};

// What recording a draw takes from whoever owns its command buffer. A draw recorded as it is met
// shares its pass with the draws around it, and counts its samples. A draw recorded later has a
// buffer to itself.
class DrawRecordEnv {
public:
	virtual ~DrawRecordEnv() = default;

	// Space for the descriptor writes, which no other write may be using.
	[[nodiscard]] virtual DescriptorScratch& Scratch() = 0;
	// Called once the dynamic state is committed, to open the rendering the draw is in.
	virtual void BeginRendering(vk::CommandBuffer command, const RenderState& rendering) = 0;
	// Called after the pipeline is bound. The result goes back to EndDraw.
	[[nodiscard]] virtual int32_t BeginDraw(vk::CommandBuffer command) = 0;
	// Called after the draw commands.
	virtual void EndDraw(vk::CommandBuffer command, int32_t token, const DrawCallInfo& draw) = 0;
};

// A draw whose pipeline is not built when it is met. It owns copies of everything RecordDraw reads,
// so the draw can be recorded later into a command buffer of its own, after the draw's stack frame,
// the registers and the scheduler's dynamic state have moved on. The bindings lose their runtime
// pointers, since those point into the stack frame, and the dynamic state has every bit dirty, since
// the buffer it is recorded into starts with none of it set. The view's pointers refer to the hole's
// own copies, so a hole never moves.
class DrawHole {
public:
	// The view must come from a draw prepared without early vertex buffer binds, since the commands
	// of the hole's buffer have to bind them. Its bindings and dynamic state are copied.
	// `skip_verdict` is what the S2 rules said of the draw, a static name kept for the report, or null.
	// `skip_targets` are the targets that refused it, kept for the report, or null.
	DrawHole(const DrawView& view, const GraphicsPipelineKey& key, const char* skip_verdict = nullptr,
	         std::shared_ptr<const Common::FirstRunReport::SkipTargets> skip_targets = nullptr);
	KYTY_CLASS_NO_COPY(DrawHole);

	/// The key of the pipeline the draw waits for.
	[[nodiscard]] const GraphicsPipelineKey& Key() const noexcept { return m_key; }
	/// What the S2 rules said of the draw when it was deferred, or null when nothing was kept.
	[[nodiscard]] const char* SkipVerdict() const noexcept { return m_skip_verdict; }
	/// The hole as the report counts it, its verdict and refusing targets together, so a resolve row
	/// cannot take one without the other. `critical` says its key is the one the resolve waited on.
	[[nodiscard]] Common::FirstRunReport::HoleVerdict Verdict(bool critical) const noexcept {
		return {.verdict = m_skip_verdict, .critical = critical, .targets = m_skip_targets.get()};
	}
	/// The draw to record, over this hole's own copies.
	[[nodiscard]] const DrawView& View() const noexcept { return m_view; }

private:
	GraphicsPipelineKey                              m_key;
	const char*                                      m_skip_verdict = nullptr;
	std::shared_ptr<const Common::FirstRunReport::SkipTargets> m_skip_targets;
	DrawView                                         m_view;
	std::array<PreparedBindings, MaxBindingStages>   m_bindings;
	DynamicState                                     m_dynamic_state;
};

// Whether beginning this rendering would clear an attachment. Colour attachments always load, so
// only depth and stencil can. A hole's rendering must not, because the clear is applied once, in
// the pass the draw's eager half records where the draw was met.
[[nodiscard]] bool RenderStateClears(const RenderState& state);

// Begins and ends dynamic rendering on a command buffer of the caller's own. Unlike the same calls
// on CommandBuffer they remember nothing, so each begin opens a pass and each end closes one.
void RecordBeginRendering(vk::CommandBuffer command, const RenderState& state);
void RecordEndRendering(vk::CommandBuffer command);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DRAWRECORD_H_
