#include "graphics/host_gpu/renderer/renderDraw.h"

#include "common/assert.h"
#include "common/common.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/firstRunReport.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/stringUtils.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/tile.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/drawRecord.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"
#include "kernel/eventQueue.h"
#include "kernel/memory.h"
#include "kernel/pthread.h"
#include "libs/errno.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

std::pair<int32_t, uint32_t> ResolveDrawOffsets(uint32_t index_offset,
	                                           const ShaderVertexInputInfo& vs_input_info) {
	auto     vertex_offset   = static_cast<int32_t>(index_offset);
	uint32_t instance_offset = 0;
	if (!vs_input_info.has_dynamic_vertex_fetch) {
		return {vertex_offset, instance_offset};
	}

	EXIT_IF(!vs_input_info.stage);
	const auto& program   = vs_input_info.stage.module->info;
	const auto& resources = *vs_input_info.stage.resources;
	if (index_offset == 0 &&
	    program.info.vertex_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.vertex_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			vertex_offset = static_cast<int32_t>(resources.user_data[index]);
		}
	}
	if (program.info.instance_offset_sgpr >= static_cast<int32_t>(program.user_data_base)) {
		const auto index =
		    static_cast<uint32_t>(program.info.instance_offset_sgpr) - program.user_data_base;
		if (index < resources.user_data.size()) {
			instance_offset = resources.user_data[index];
		}
	}

	return {vertex_offset, instance_offset};
}

static std::atomic<uint32_t> g_draw_state_log_count   = 0;
static std::atomic<uint32_t> g_draw_input_log_count   = 0;
static std::atomic<uint32_t> g_mrt_state_log_count    = 0;

static std::atomic<uint32_t> g_framebuffer_skip_log_count = 0;

static float ConvertPolygonOffsetConstantFactor(float guest_factor, const HW::PolyOffset& offset,
                                                vk::Format host_depth_format) {
	if (offset.db_is_float_fmt) {
		return guest_factor;
	}

	int host_depth_bits = 0;
	switch (host_depth_format) {
		case vk::Format::eD16Unorm:
		case vk::Format::eD16UnormS8Uint: host_depth_bits = 16; break;
		case vk::Format::eD24UnormS8Uint: host_depth_bits = 24; break;
		default:
			// A fixed-point guest bias cannot be represented exactly by a floating-point host
			// attachment without VK_EXT_depth_bias_control.
			return guest_factor;
	}
	return std::ldexp(guest_factor, host_depth_bits + offset.neg_num_db_bits);
}

static const char* RenderColorTypeName(const RenderColorInfo& color) {
	return color.image_id ? "RenderTexture" : "NoColorOutput";
}

static void LogFramebufferSkip(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               uint32_t index_count, uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	auto log_id = g_framebuffer_skip_log_count.fetch_add(1, std::memory_order_relaxed);
	if (log_id >= 128) {
		return;
	}

	LOGF(
	    "DrawFramebufferSkip[%u]: %s color=%s color_addr=0x%010" PRIx64 " color_size=0x%016" PRIx64
	    " color_image=%s depth_format=%s depth_image=%s depth_vaddr_num=%d target_mask=0x%08" PRIx32
	    " prim=%u index_count=%u flags=0x%08" PRIx32 "\n",
	    log_id, draw_name, RenderColorTypeName(color), color.desc.info.data.address,
	    color.desc.info.data.size, color.image_id ? "yes" : "no",
	    vk::to_string(depth.desc.view_info.format).c_str(), depth.image_id ? "yes" : "no",
	    static_cast<int>(!depth.desc.info.data.Empty()) +
	        static_cast<int>(depth.desc.info.HasStencil()),
	    ctx.GetRenderTargetMask(), static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags);
}

static void LogMrtState(const char* draw_name, const CommandBuffer& buffer,
                        const ShaderPixelInputInfo& ps_input_info) {
	const auto& ctx            = buffer.GetRegisters();
	const auto& sh_regs        = ctx.GetShaderRegisters();
	const auto  rt_mask        = ctx.GetRenderTargetMask();
	const auto  cb_shader_mask = sh_regs.m_cbShaderMask;
	const auto& bc0            = ctx.GetBlendControl(0);

	auto log_id = g_mrt_state_log_count.fetch_add(1);
	if (log_id >= 32) {
		return;
	}

	LOGF("MrtState[%u]: %s rt_mask=0x%08" PRIx32 " cb_shader_mask=0x%08" PRIx32
	     " blend0=%s src=%u dst=%u alpha_src=%u alpha_dst=%u sep_alpha=%s\n",
	     log_id, draw_name, rt_mask, cb_shader_mask, bc0.enable ? "true" : "false",
	     bc0.color_srcblend, bc0.color_destblend, bc0.alpha_srcblend, bc0.alpha_destblend,
	     bc0.separate_alpha_blend ? "true" : "false");

	for (uint32_t i = 0; i < 8; i++) {
		const auto& rt  = ctx.GetRenderTarget(i);
		const auto& bc  = ctx.GetBlendControl(i);
		const auto  ctm = (rt_mask >> (i * 4u)) & 0x0fu;
		const auto  csm = (cb_shader_mask >> (i * 4u)) & 0x0fu;

		if (rt.base.addr == 0 && ps_input_info.target_output_mode[i] == 0 && ctm == 0 && csm == 0 &&
		    !bc.enable) {
			continue;
		}

		LOGF("MrtState[%u]: slot=%u addr=0x%010" PRIx64
		     " target_mask=0x%x shader_mask=0x%x out_mode=%u"
		     " fmt=0x%08" PRIx32 " nfmt=0x%08" PRIx32 " order=0x%08" PRIx32
		     " width=%u height=%u tile=%u"
		     " blend=%s src=%u dst=%u alpha_src=%u alpha_dst=%u\n",
		     log_id, i, rt.base.addr, ctm, csm, ps_input_info.target_output_mode[i],
		     static_cast<uint32_t>(rt.info.format), static_cast<uint32_t>(rt.info.channel_type),
		     static_cast<uint32_t>(rt.info.channel_order), rt.attrib2.width + 1,
		     rt.attrib2.height + 1, static_cast<uint32_t>(rt.attrib3.tile_mode),
		     bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.alpha_srcblend,
		     bc.alpha_destblend);
	}
}

static void LogDrawTargetState(const char* draw_name, const RenderColorInfo& color,
                               const RenderDepthInfo& depth, const CommandBuffer& buffer,
                               const ShaderPixelInputInfo& ps_input_info, uint32_t index_count,
                               uint32_t flags) {
	const auto& ctx  = buffer.GetRegisters();
	const auto& ucfg = buffer.GetUserConfig();
	if (!color.image_id) {
		return;
	}

	auto log_id = g_draw_state_log_count.fetch_add(1);
	if (log_id >= 192) {
		return;
	}

	const auto& cc             = ctx.GetColorControl();
	const auto& bc             = ctx.GetBlendControl(color.target_slot);
	const auto& dc             = ctx.GetDepthControl();
	const auto& vp             = ctx.GetScreenViewport();
	const auto& vp0            = vp.viewports[0];
	const auto& ps_resources   = ps_input_info.stage.module->info.info;
	const auto  sampled_images = std::count_if(
	    ps_resources.images.begin(), ps_resources.images.end(), [](const auto& image) {
		    return image.resource_class == ShaderRecompiler::IR::ImageResourceClass::Sampled;
	    });

	const auto extent = color.Extent();
	const auto sc     = calc_final_scissor(vp, ctx.GetScanModeControl(), extent, 0);

	LOGF(
	    "DrawTargetState[%u]: frame=%d %s target=%s addr=0x%010" PRIx64
	    " extent=%ux%u prim=%u index_count=%u flags=0x%08" PRIx32 " color_mask=0x%08" PRIx32
	    " cc_mode=%u cc_op=0x%02x"
	    " blend=%s src=%u dst=%u comb=%u ps_tex=%d sampled=%d storage=%d ps_kill=%s target_mode0=%u"
	    " depth_test=%s depth_write=%s depth_func=%u depth_clear=%s viewport=(%.1f,%.1f %.1fx%.1f) "
	    "scissor=(%d,%d)-(%d,%d)\n",
	    log_id, buffer.GetContext().GetGpu().GetFrameNum(), draw_name, RenderColorTypeName(color),
	    color.desc.info.data.address, extent.width, extent.height,
	    static_cast<uint32_t>(ucfg.GetPrimType()), index_count, flags, ctx.GetRenderTargetMask(),
	    cc.mode, cc.op,
	    bc.enable ? "true" : "false", bc.color_srcblend, bc.color_destblend, bc.color_comb_fcn,
	    static_cast<int>(ps_resources.images.size()), static_cast<int>(sampled_images),
	    static_cast<int>(ps_resources.images.size() - sampled_images),
	    ps_input_info.ps_pixel_kill_enable ? "true" : "false", ps_input_info.target_output_mode[0],
	    dc.z_enable ? "true" : "false", dc.z_write_enable ? "true" : "false", dc.zfunc,
	    depth.depth_clear_enable ? "true" : "false", vp0.xoffset - vp0.xscale,
	    vp0.yoffset - vp0.yscale, vp0.xscale * 2.0f, vp0.yscale * 2.0f, sc.left, sc.top, sc.right,
	    sc.bottom);

	LogMrtState(draw_name, buffer, ps_input_info);
}

static void LogDrawInputState(const CommandBuffer& buffer, const RenderColorInfo& color,
                              const ShaderVertexInputInfo& vs_input_info,
                              uint32_t index_type_and_size, uint32_t index_count,
                              const void* index_addr) {
	auto log_id = g_draw_input_log_count.fetch_add(1);
	if (log_id >= 64) {
		return;
	}

	LOGF("DrawInputState[%u]: frame=%d target=%s addr=0x%010" PRIx64
	     " index_type=%u index_count=%u index_addr=0x%016" PRIx64
	     " vs_resources=%d vs_buffers=%d\n",
	     log_id, buffer.GetContext().GetGpu().GetFrameNum(), RenderColorTypeName(color),
	     color.desc.info.data.address, index_type_and_size, index_count,
	     reinterpret_cast<uint64_t>(index_addr), vs_input_info.resources_num,
	     vs_input_info.buffers_num);

	for (int bi = 0; bi < vs_input_info.buffers_num; bi++) {
		const auto& b = vs_input_info.buffers[bi];
		LOGF("DrawInputState[%u]: vb[%d] addr=0x%010" PRIx64
		     " stride=%u records=%u fetch_index=%u\n",
		     log_id, bi, b.addr, b.stride, b.num_records, b.fetch_index);

		const auto* bytes = reinterpret_cast<const uint8_t*>(b.addr);
		if (bytes != nullptr && b.stride != 0) {
			const uint32_t records = std::min<uint32_t>(b.num_records, 4u);
			for (uint32_t rec = 0; rec < records; rec++) {
				const auto* rec_bytes = bytes + static_cast<uint64_t>(rec) * b.stride;
				const auto  dword_num = std::min<uint32_t>(b.stride / 4u, 12u);
				uint32_t    raw[12]   = {};
				float       flt[12]   = {};
				for (uint32_t i = 0; i < dword_num; i++) {
					std::memcpy(&raw[i], rec_bytes + i * 4u, sizeof(raw[i]));
					std::memcpy(&flt[i], rec_bytes + i * 4u, sizeof(flt[i]));
				}
				LOGF("DrawInputState[%u]: vb[%d].rec[%u] stride=%u dwords=%u raw=%08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " %08" PRIx32 " %08" PRIx32 " %08" PRIx32
				     " f=(%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f)\n",
				     log_id, bi, rec, b.stride, dword_num, raw[0], raw[1], raw[2], raw[3], raw[4],
				     raw[5], raw[6], raw[7], raw[8], flt[0], flt[1], flt[2], flt[3], flt[4], flt[5],
				     flt[6], flt[7], flt[8]);

				for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
					const auto& r  = vs_input_info.resources[ai];
					const auto& rd = vs_input_info.resources_dst[ai];
					if (rd.buffer_index != bi) {
						continue;
					}
					const auto offset = static_cast<uint32_t>(r.Base48() - b.addr);
					if (offset + 4u <= b.stride &&
					    r.Format() == Prospero::BufferFormat::k8_8_8_8UNorm) {
						uint32_t packed = 0;
						std::memcpy(&packed, rec_bytes + offset, sizeof(packed));
						const auto r8 = (packed >> 0u) & 0xffu;
						const auto g8 = (packed >> 8u) & 0xffu;
						const auto b8 = (packed >> 16u) & 0xffu;
						const auto a8 = (packed >> 24u) & 0xffu;
						LOGF("DrawInputState[%u]: vb[%d].rec[%u].attr[%d] dst=v%d fmt=56 "
						     "rgba8=%02" PRIx32 "%02" PRIx32 "%02" PRIx32 "%02" PRIx32
						     " rgba=(%.3f,%.3f,%.3f,%.3f)\n",
						     log_id, bi, rec, ai, rd.register_start, r8, g8, b8, a8,
						     static_cast<double>(r8) / 255.0, static_cast<double>(g8) / 255.0,
						     static_cast<double>(b8) / 255.0, static_cast<double>(a8) / 255.0);
					}
				}
			}
		}

		for (int ai = 0; ai < vs_input_info.resources_num; ai++) {
			const auto& r  = vs_input_info.resources[ai];
			const auto& rd = vs_input_info.resources_dst[ai];
			if (rd.buffer_index != bi) {
				continue;
			}
			LOGF("DrawInputState[%u]: attr[%d] offset=%u dst=v%d regs=%d fetch_index=%u "
			     "sharp=%08" PRIx32 " %08" PRIx32 " %08" PRIx32 " %08" PRIx32 "\n",
			     log_id, ai, static_cast<uint32_t>(r.Base48() - b.addr), rd.register_start,
			     rd.registers_num, rd.fetch_index, r.fields[0], r.fields[1], r.fields[2], r.fields[3]);
		}
	}
}

static void UpdateGraphicsDynamicState(const CommandBuffer& buffer, DynamicState& dynamic_state,
                                       const ShaderVertexInputInfo& vs_input_info,
                                       const RenderDepthInfo& depth, const RenderState& rendering) {
	KYTY_PROFILER_FUNCTION();

	const auto& ctx = buffer.GetRegisters();
	const auto&        vp  = ctx.GetScreenViewport();
	const vk::Extent2D framebuffer_extent {rendering.width, rendering.height};
	const auto& outputs = vs_input_info.stage.module->info.info.outputs;
	const bool  indexed_viewports =
	    std::any_of(outputs.begin(), outputs.end(), [](const auto& output) {
		    return output.kind == ShaderRecompiler::IR::StageOutputKind::ViewportIndex;
	    });
	constexpr uint32_t viewport_slots = std::size(HW::ScreenViewport {}.viewports);
	std::array<vk::Viewport, viewport_slots> viewports {};
	std::array<vk::Rect2D, viewport_slots>   scissors {};
	const uint32_t viewport_count = indexed_viewports ? viewport_slots : 1;
	for (uint32_t i = 0; i < viewport_count; i++) {
		const auto& guest    = vp.viewports[i];
		auto&       viewport = viewports[i];
		if (ctx.GetClipControl().clip_disable) {
			const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
			viewport.width  = static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u));
			viewport.height = static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u));
		} else {
			viewport.x      = guest.xoffset - guest.xscale;
			viewport.y      = guest.yoffset - guest.yscale;
			viewport.width  = guest.xscale * 2.0f;
			viewport.height = guest.yscale * 2.0f;
		}
		viewport.minDepth =
		    guest.zoffset - (ctx.GetClipControl().dx_clip_space ? 0.0f : guest.zscale);
		viewport.maxDepth = guest.zscale + guest.zoffset;

		const auto final_scissor =
		    calc_final_scissor(vp, ctx.GetScanModeControl(), framebuffer_extent, i);
		auto& scissor  = scissors[i];
		scissor.offset = {final_scissor.left, final_scissor.top};
		scissor.extent = {static_cast<uint32_t>(final_scissor.right - final_scissor.left),
		                  static_cast<uint32_t>(final_scissor.bottom - final_scissor.top)};
		if (viewport.width == 0.0f) {
			// Keep empty slots at their guest index; Vulkan requires a positive viewport width.
			viewport.width = 1.0f;
			scissor.extent = {0, 0};
		}
	}
	dynamic_state.SetViewports(std::span {viewports}.first(viewport_count));
	dynamic_state.SetScissors(std::span {scissors}.first(viewport_count));

	float line_width = ctx.GetLineWidth();
	if (line_width != 1.0f) {
		static bool logged = false;
		if (!logged) {
			LOGF("Render: temporary: clamping Vulkan line width %f to 1.0 because wideLines is "
			     "not enabled\n",
			     line_width);
			logged = true;
		}
		line_width = 1.0f;
	}
	dynamic_state.SetLineWidth(line_width);
	const auto&      blend = ctx.GetBlendColor();
	const std::array blend_constants {blend.red, blend.green, blend.blue, blend.alpha};
	dynamic_state.SetBlendConstants(blend_constants);
	dynamic_state.SetDepthTestEnabled(depth.depth_test_enable);
	dynamic_state.SetDepthWriteEnabled(depth.depth_write_enable);
	dynamic_state.SetDepthCompareOp(depth.depth_compare_op);
#if !defined(__APPLE__)
	dynamic_state.SetDepthBoundsTestEnabled(depth.depth_bounds_test_enable);
	if (depth.depth_bounds_test_enable) {
		dynamic_state.SetDepthBounds(ctx.GetDepthBoundsMin(), ctx.GetDepthBoundsMax());
	}
#endif

	const auto& mode              = ctx.GetModeControl();
	const auto& poly_offset       = ctx.GetPolyOffset();
	const bool  use_front         = mode.poly_offset_front_enable && !mode.cull_front;
	const bool  use_back          = mode.poly_offset_back_enable && !mode.cull_back;
	const bool  depth_bias_enable = use_front || use_back;
	dynamic_state.SetDepthBiasEnabled(depth_bias_enable);
	if (depth_bias_enable) {
		// Vulkan has one bias for both faces. Prefer a visible front face when both are enabled.
		const float guest_constant_factor =
		    use_front ? poly_offset.front_offset : poly_offset.back_offset;
		const float constant_factor = ConvertPolygonOffsetConstantFactor(
		    guest_constant_factor, poly_offset, depth.desc.view_info.format);
		const float slope_factor =
		    (use_front ? poly_offset.front_scale : poly_offset.back_scale) / 16.0f;
		dynamic_state.SetDepthBias(constant_factor, poly_offset.clamp, slope_factor);
	}

	// Rect lists are drawn without culling.
	vk::CullModeFlags cull_mode {};
	if (!Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		if (mode.cull_front) {
			cull_mode |= vk::CullModeFlagBits::eFront;
		}
		if (mode.cull_back) {
			cull_mode |= vk::CullModeFlagBits::eBack;
		}
	}
	dynamic_state.SetCullMode(cull_mode);
	dynamic_state.SetFrontFace(mode.face ? vk::FrontFace::eClockwise
	                                     : vk::FrontFace::eCounterClockwise);

	dynamic_state.SetStencilTestEnabled(depth.stencil_test_enable);
	if (depth.stencil_test_enable) {
		const auto& front = depth.stencil_front;
		const auto& back  = depth.stencil_back;
		dynamic_state.SetStencilOps({front.failOp, front.passOp, front.depthFailOp, front.compareOp},
		                            {back.failOp, back.passOp, back.depthFailOp, back.compareOp});
		dynamic_state.SetStencilCompareMasks(front.compareMask, back.compareMask);
		dynamic_state.SetStencilWriteMasks(front.writeMask, back.writeMask);
		dynamic_state.SetStencilReferences(front.reference, back.reference);
	}

#if defined(__APPLE__)
	// MoltenVK has no VK_EXT_color_write_enable; the pipeline is created without the
	// eColorWriteEnableEXT dynamic state and relies on the static colorWriteMask instead.
#else
	vk::Bool32 enable[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	for (uint32_t slot = 0; slot < rendering.num_color_attachments; slot++) {
		enable[slot] = rendering.color_attachments[slot].image_view != nullptr;
	}
	dynamic_state.SetColorWriteEnables(std::span {enable}.first(rendering.num_color_attachments));
#endif
}

static bool DrawHasValidVertexShader(const HW::Shader& sh_ctx) {

	const auto& vs = sh_ctx.GetVs();
	return vs.es_regs.data_addr != 0;
}

// Whether an HTILE clear is waiting to be applied to the depth target by the next draw into it.
static bool HtileClearPending(TextureCache& cache, const RenderDepthInfo& depth) {
	const auto& metadata = depth.desc.info.metadata;
	return metadata.kind == ImageMetadataKind::Htile &&
	       cache.IsMetaCleared(metadata.range.address, depth.desc.view_info.base_layer);
}

// Whether the render area of the draw is the whole depth target. The area is the smallest of the
// attachments in width, height and layers, and a load-op clear covers all of it whatever the
// scissor says. A color target smaller than the depth target in any of them shrinks the area, so
// the clear no longer covers the depth.
static bool ClearCoversDepthTarget(const DrawRenderState& state) {
	const auto&     extent = state.depth_info.desc.info.extent;
	const auto      layers = state.depth_info.desc.view_info.layer_count;
	const std::span colors {state.color_info, state.color_count};
	return std::ranges::all_of(colors, [&](const RenderColorInfo& color) {
		const auto size = color.Extent();
		return size.width >= extent.width && size.height >= extent.height &&
		       color.desc.view_info.layer_count >= layers;
	});
}

void NoteDrawTargets(TextureCache& cache, const DrawRenderState& state) {
	for (uint32_t i = 0; i < state.color_count; i++) {
		cache.NoteRenderTargetUse(state.color_info[i].image_id, state.color_info[i].desc.view_info,
		                          false);
	}
	const auto& depth = state.depth_info;
	if (depth.image_id) {
		// Only the clear register counts. An HTILE clear is a compute fill whose extent the CPU
		// cannot see, so it may cover a tile or a region and is not taken as a clear.
		const bool cleared = depth.depth_clear_enable && ClearCoversDepthTarget(state);
		// An HTILE clear still waiting is noted apart, to tell a target cleared only that way from
		// one never cleared. It runs before AcquireRenderTargets consumes the clear.
		cache.NoteRenderTargetUse(depth.image_id, depth.desc.view_info, cleared,
		                          HtileClearPending(cache, depth));
	}
}

// Whether any stage of the draw can change something besides the render targets.
static bool DrawWritesBeyondTargets(const DrawRenderState& state) {
	const auto stages = state.VertexStages();
	return std::ranges::any_of(stages, [](const ShaderVertexInputInfo& stage) {
		       return HasShaderOutputsBeyondTargets(stage.stage);
	       }) ||
	       (state.ps_active && HasShaderOutputsBeyondTargets(state.ps_input_info.stage));
}

const char* SkipRefusalName(SkipRefusal refusal) {
	switch (refusal) {
		case SkipRefusal::None: return "none";
		case SkipRefusal::NoDepth: return "no_depth_target";
		case SkipRefusal::Writer: return "writer";
		case SkipRefusal::DepthOnlyModeColor: return "depth_only_mode_color";
		case SkipRefusal::PartialView: return "partial_view";
		case SkipRefusal::HtileOnlyClear: return "htile_only_clear";
		case SkipRefusal::DepthNotCleared: return "depth_not_cleared";
		case SkipRefusal::TooFewDraws: return "too_few_draws";
		case SkipRefusal::TooFewDrawsAlsoUncleared: return "too_few_draws_also_uncleared";
		case SkipRefusal::ColorHistory: return "color_history";
	}
	return "none";
}

// Turns the verdict on the depth target into the rule it broke.
static SkipRefusal DepthRefusal(TextureCache::RebuildVerdict verdict) {
	switch (verdict) {
		case TextureCache::RebuildVerdict::Rebuilt: return SkipRefusal::None;
		case TextureCache::RebuildVerdict::PartialView: return SkipRefusal::PartialView;
		case TextureCache::RebuildVerdict::TooFewDraws: return SkipRefusal::TooFewDraws;
		case TextureCache::RebuildVerdict::TooFewDrawsUncleared:
			return SkipRefusal::TooFewDrawsAlsoUncleared;
		case TextureCache::RebuildVerdict::NotCleared: return SkipRefusal::DepthNotCleared;
		case TextureCache::RebuildVerdict::HtileClearedOnly: return SkipRefusal::HtileOnlyClear;
	}
	return SkipRefusal::None;
}

// Counts, for the report, each target of a draw the rule allowed that was judged by its view's own
// history, which --async-pipelines-prefer-skip turns on. Without the flag a partial view is never
// allowed, so every partial view of an allowed draw came through that history.
static void NoteViewHistoryVerdict(TextureCache& cache, const DrawRenderState& state) {
	if (!ViewHistoryEnabled() || !Common::FirstRunReport::Enabled()) {
		return;
	}
	using Common::FirstRunReport::ViewHistoryCount;
	const auto note = [&](ImageId id, const ImageViewInfo& view, bool depth) {
		const auto scope = cache.ViewScopeOf(id, view);
		if (scope == ViewRangeScope::Layer) {
			Common::FirstRunReport::AddViewHistoryCount(depth ? ViewHistoryCount::DepthLayer : ViewHistoryCount::ColorLayer);
		} else if (scope == ViewRangeScope::Mip) {
			Common::FirstRunReport::AddViewHistoryCount(depth ? ViewHistoryCount::DepthMip : ViewHistoryCount::ColorMip);
		}
	};
	note(state.depth_info.image_id, state.depth_info.desc.view_info, true);
	for (const auto& color: std::span {state.color_info, state.color_count}) {
		note(color.image_id, color.desc.view_info, false);
	}
}

SkipRefusal ExplainSkipRefusal(TextureCache& cache, const DrawRenderState& state, bool depth_only,
                               uint32_t* refusing_color) {
	// A color-only pass shows no sign of being rebuilt, because a color clear is applied by the GPU
	// and the CPU never sees it. It waits for the pipeline. In depth-only mode so does every draw
	// that has a color target.
	if (!state.depth_info.image_id) {
		return SkipRefusal::NoDepth;
	}
	if (DrawWritesBeyondTargets(state)) {
		return SkipRefusal::Writer;
	}
	if (depth_only && state.color_count != 0) {
		return SkipRefusal::DepthOnlyModeColor;
	}
	const auto& depth = state.depth_info;
	const auto  verdict = cache.JudgeRebuiltEveryFrame(depth.image_id, depth.desc.view_info, true,
	                                                   MIN_DRAWS_FOR_SKIP);
	if (const auto refusal = DepthRefusal(verdict); refusal != SkipRefusal::None) {
		return refusal;
	}
	for (uint32_t index = 0; index < state.color_count; index++) {
		const auto& color  = state.color_info[index];
		const auto  judged = cache.JudgeRebuiltEveryFrame(color.image_id, color.desc.view_info,
		                                                  false, MIN_DRAWS_FOR_SKIP);
		if (judged == TextureCache::RebuildVerdict::PartialView) {
			return SkipRefusal::PartialView;
		}
		if (judged != TextureCache::RebuildVerdict::Rebuilt) {
			if (refusing_color != nullptr) {
				*refusing_color = index;
			}
			return SkipRefusal::ColorHistory;
		}
	}
	NoteViewHistoryVerdict(cache, state);
	return SkipRefusal::None;
}

std::shared_ptr<const Common::FirstRunReport::SkipTargets>
DescribeSkipRefusal(TextureCache& cache, const DrawRenderState& state, SkipRefusal refusal, uint32_t refusing_color) {
	if (refusal != SkipRefusal::ColorHistory || !Common::FirstRunReport::Enabled() ||
	    refusing_color >= state.color_count) {
		return nullptr;
	}
	auto        targets = std::make_shared<Common::FirstRunReport::SkipTargets>();
	const auto& color   = state.color_info[refusing_color];
	targets->color      = cache.DescribeSkipTarget(color.image_id, color.desc.view_info);
	targets->color_slot = color.target_slot;
	targets->depth      = cache.DescribeSkipTarget(state.depth_info.image_id, state.depth_info.desc.view_info);
	return targets;
}

bool DrawMaySkipPipeline(TextureCache& cache, const DrawRenderState& state, bool depth_only) {
	return ExplainSkipRefusal(cache, state, depth_only) == SkipRefusal::None;
}

namespace {

// The parts of a draw that keep it from becoming a hole and that are known before its pipeline.
// Indirect draws read an arguments buffer, so they block unless --deferred-indirect-draws asks for
// them (experimental), since the draw view keeps the buffer and offset the hole records with. Task
// mesh draws, which are indirect, upload their task parameters when they are met, and always block. A
// direct mesh draw keeps everything its recording reads in the view (groups, index address, draw
// dwords), so it may become a hole when --deferred-mesh-draws asks for it (experimental), and blocks
// otherwise.
struct DeferFacts {
	bool mesh      = false;
	bool indirect  = false;
	bool mesh_task = false;
};

// Lets a draw go without its pipeline when the gate allows it, keeps the rule that refused it, and
// says whether the draw could be deferred instead.
class DrawSkipPolicy final: public PipelineSkipPolicy {
public:
	/// Remembers the draw and the mode to judge it in.
	DrawSkipPolicy(RenderContext& context, const DrawRenderState& state, bool depth_only,
	               const DeferFacts& facts)
	    : m_context(context), m_cache(context.GetTextureCache()), m_state(state),
	      m_depth_only(depth_only), m_facts(facts) {}

	/// Whether the remembered draw may go without its pipeline. A refusal is kept for Refusal.
	/// Without --async-pipelines nothing is skipped, and the history checks are not run.
	[[nodiscard]] bool MaySkip() const override {
		if (!Config::AsyncPipelinesEnabled()) {
			return false;
		}
		m_refusing_color = 0;
		m_refusal        = ExplainSkipRefusal(m_cache, m_state, m_depth_only, &m_refusing_color);
		return m_refusal == SkipRefusal::None;
	}

	/// What stops the remembered draw from being recorded later, or null. A guest query that is open
	/// counts the draw's samples, and a hole would see the query state of the time it is recorded.
	[[nodiscard]] const char* DeferBlocker() const override {
		if (m_context.GetOcclusionQueries().HasOpenQuery()) {
			return "open_query";
		}
		// The arguments buffer and the task flag come first, so a task draw is never counted as a
		// plain mesh draw, and the mesh counts say how many direct mesh draws could be deferred.
		// A task draw is indirect too, and has its own reason.
		if (m_facts.mesh_task) {
			return "mesh_task";
		}
		if (m_facts.indirect && !Config::DeferredIndirectDrawsEnabled()) {
			return "indirect";
		}
		if (m_facts.mesh && !m_facts.mesh_task && !Config::DeferredMeshDrawsEnabled()) {
			return "mesh_direct";
		}
		if (!m_context.GetCommandScheduler().CanSplitHere()) {
			return "thread";
		}
		auto& scheduler = m_context.GetCommandScheduler();
		if (!scheduler.HoleCapReached()) {
			return nullptr;
		}
		scheduler.RequestHoleFlush();
		return "cap";
	}

	/// The name of the rule that refused the draw at the last question, or null.
	[[nodiscard]] const char* Refusal() const override {
		return m_refusal == SkipRefusal::None ? nullptr : SkipRefusalName(m_refusal);
	}

	/// The targets that refused the draw at the last question, for the report, or null. Described only
	/// here, when the draw becomes a hole, so a refused draw that blocks or skips does no lock or string
	/// work. The cache asks within the same AcquireGraphicsPipeline call on the same thread, and the
	/// histories change only when a draw's targets are noted after that call or when the garbage
	/// collector ticks at a flip, so they are those the question judged.
	[[nodiscard]] std::shared_ptr<const Common::FirstRunReport::SkipTargets> RefusedTargets() const override {
		return DescribeSkipRefusal(m_cache, m_state, m_refusal, m_refusing_color);
	}

private:
	RenderContext&         m_context;
	TextureCache&          m_cache;
	const DrawRenderState& m_state;
	bool                   m_depth_only;
	DeferFacts             m_facts;
	mutable SkipRefusal    m_refusal = SkipRefusal::None;
	mutable uint32_t       m_refusing_color = 0;
};

} // namespace

// Gets the draw's pipeline, or null when a background build is not done and the draw may go
// without it or wait out of line, which `deferred` says. Without background builds it always waits,
// exactly as before. A null `deferred` means the caller cannot make holes. The draw's targets are
// noted after the answer, and only when the draw will run, so the draw is not evidence for itself
// and a skipped draw proves nothing. A draw refused a skip is reported by the pipeline cache.
static PipelineCache::Pipeline* AcquireDrawPipeline(RenderContext& context, CommandBuffer& buffer,
                                                    const DrawRenderState& state,
                                                    const DeferFacts&      facts,
                                                    PipelineCache::DeferredDraw* deferred) {
	auto& pipelines = context.GetPipelineCache();
	if (!pipelines.BuildsInBackground()) {
		return &pipelines.GetGraphicsPipeline(buffer, state);
	}
	const DrawSkipPolicy policy {context, state, Config::AsyncPipelinesDepthOnly(), facts};
	auto*                pipeline = pipelines.AcquireGraphicsPipeline(buffer, state, &policy, deferred);
	if (pipeline != nullptr || (deferred != nullptr && deferred->active)) {
		NoteDrawTargets(context.GetTextureCache(), state);
	}
	return pipeline;
}

// Leaves out the commands of a draw that has no pipeline yet. A depth or HTILE clear the draw
// consumed would be lost with it, so the pass still begins when there was one. The targets keep a
// mark of the skip, which is reported if something samples them before a clear repairs them. A
// target a resident image heap references gets no mark, and is reported as a possible read at
// once, since the heap resolves its bindings out of reach of the mark.
static void SkipDrawWithoutPipeline(RenderContext& context, CommandBuffer& buffer,
                                    const RenderState& rendering, const DrawRenderState& state) {
	if (state.depth_info.load_clear_aspects) {
		buffer.BeginRendering(rendering);
	}
	context.GetOcclusionQueries().CountSkippedDraw();
	auto& textures = context.GetTextureCache();
	for (uint32_t i = 0; i < state.color_count; i++) {
		if (textures.NoteSkippedTarget(state.color_info[i].image_id, false)) {
			Common::FirstRunReport::AddSampledAfterSkip("heap_color", 1);
		}
	}
	if (state.depth_info.image_id && textures.NoteSkippedTarget(state.depth_info.image_id, true)) {
		Common::FirstRunReport::AddSampledAfterSkip("heap_depth", 1);
	}
	Common::FirstRunReport::AddSkip(Common::FirstRunReport::SkipKind::Draw,
	                                state.color_count != 0 ? "depth_and_color" : "depth_only");
}

RenderState RenderExecutor::AcquireRenderTargets(CommandBuffer& buffer, RenderColorInfo* colors,
                                                 uint32_t color_count, RenderDepthInfo& depth,
                                                 vk::ImageAspectFlags& feedback_aspects,
                                                 std::span<PreparedBindings* const> stages) {
	EXIT_IF(colors == nullptr || color_count > RENDER_COLOR_ATTACHMENTS_MAX);
	feedback_aspects = {};
	auto&       cache = m_context.GetTextureCache();
	RenderState state {};
	state.width                 = std::numeric_limits<uint32_t>::max();
	state.height                = std::numeric_limits<uint32_t>::max();
	state.num_layers            = std::numeric_limits<uint32_t>::max();
	state.num_color_attachments = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		auto& target = colors[i];
		EXIT_IF(!target.image_id);
		const auto image_view = cache.FindRenderTarget(target.image_id, target.desc);
		auto&      image      = cache.GetImage(target.image_id);
		EXIT_IF(image.backing.samples != target.desc.info.samples || image_view == nullptr);
		const auto& view   = target.desc.view_info;
		const auto  layout = image.binding.is_bound ? vk::ImageLayout::eGeneral
		                                            : vk::ImageLayout::eColorAttachmentOptimal;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access =
		    vk::AccessFlagBits2::eColorAttachmentRead | vk::AccessFlagBits2::eColorAttachmentWrite;
		image.Transit(layout, image.binding.attachment_access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		const auto extent       = target.Extent();
		state.width             = std::min(state.width, extent.width);
		state.height            = std::min(state.height, extent.height);
		state.num_layers        = std::min(state.num_layers, view.layer_count);
		state.num_color_attachments = std::max(state.num_color_attachments, target.target_slot + 1);
		auto& attachment            = state.color_attachments[target.target_slot];
		attachment.image_view   = image_view;
		attachment.image_layout = layout;
	}
	if (depth.image_id) {
		const auto  image_view = cache.FindDepthTarget(depth.image_id, depth.desc);
		const auto& metadata   = depth.desc.info.metadata;
		const bool  meta_clear = HtileClearPending(cache, depth);
		depth.load_clear_aspects = {};
		if (depth.depth_clear_enable || meta_clear)
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eDepth;
		// Apply pending clears to compressed stencil as well as depth.
		if (depth.stencil_clear_enable || (meta_clear && metadata.stencil_compressed))
			depth.load_clear_aspects |= vk::ImageAspectFlagBits::eStencil;
		if (meta_clear &&
		    !cache.TouchMeta(metadata.range.address, depth.desc.view_info.base_layer, false)) {
			EXIT("failed to consume HTile clear state\n");
		}
		auto& image = cache.GetImage(depth.image_id);
		EXIT_IF(image_view == nullptr || image.backing.samples != depth.desc.info.samples);
		const auto draw_writes = depth.AttachmentWriteAspects();
		vk::ImageAspectFlags sampled_aspects;
		for (const auto* stage: stages) {
			for (const auto& binding: stage->images) {
				if (binding.image_id != depth.image_id ||
				    binding.desc.type != TextureCache::BindingType::Texture) continue;
				const auto native =
				    std::ranges::find(image.views, binding.image_view, &CachedImageView::view);
				EXIT_IF(native == image.views.end());
				sampled_aspects |= native->info.aspect;
				feedback_aspects |= DepthFeedbackAspects(draw_writes, depth.desc.view_info,
				                                         native->info);
			}
		}
		if (feedback_aspects && !m_context.GetGraphics().attachment_feedback_loop_enabled) {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("depth attachment feedback loop is not supported by the host\n");
			}
		}
		auto layout = depth_attachment_layout(depth);
		if (sampled_aspects & ~DepthReadableAspects(layout)) {
			layout = m_context.GetGraphics().attachment_feedback_loop_enabled
			             ? vk::ImageLayout::eAttachmentFeedbackLoopOptimalEXT
			             : vk::ImageLayout::eGeneral;
		}
		// The attachment store writes even when guest depth/stencil tests do not.
		const auto access = vk::AccessFlagBits2::eDepthStencilAttachmentRead |
		                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
		image.binding.attachment_layout = layout;
		image.binding.attachment_access = access;
		const auto& view                = depth.desc.view_info;
		image.Transit(layout, access,
		              ImageSubresourceRange {view.base_level, view.level_count, view.base_layer,
		                                     view.layer_count},
		              buffer.Handle());
		state.width               = std::min(state.width, depth.desc.info.extent.width);
		state.height              = std::min(state.height, depth.desc.info.extent.height);
		state.num_layers          = std::min(state.num_layers, view.layer_count);
		const auto aspects        = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		auto&      attachment     = state.depth_stencil_attachment;
		attachment.image_view     = image_view;
		attachment.image_layout   = layout;
		attachment.clear_value[0] = std::bit_cast<uint32_t>(depth.depth_clear_value);
		attachment.clear_value[1] = depth.stencil_clear_value;
		attachment.has_depth      = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.depth_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eDepth);
		attachment.has_stencil    = static_cast<bool>(aspects & vk::ImageAspectFlagBits::eStencil);
		attachment.stencil_clear =
		    static_cast<bool>(depth.load_clear_aspects & vk::ImageAspectFlagBits::eStencil);
	}
	if (color_count == 0 && !depth.image_id) {
		const auto& limits = buffer.GetGraphics().GetPhysicalDeviceProperties().limits;
		state.width        = limits.maxFramebufferWidth;
		state.height       = limits.maxFramebufferHeight;
	}
	if (state.num_layers == std::numeric_limits<uint32_t>::max()) {
		state.num_layers = 1;
	}
	EXIT_IF(state.width == 0 || state.height == 0 || state.num_layers == 0 ||
	        state.width == std::numeric_limits<uint32_t>::max() ||
	        state.height == std::numeric_limits<uint32_t>::max());
	return state;
}

enum class CbColorMode : uint8_t {
	Disable            = 0,
	Normal             = 1,
	EliminateFastClear = 2,
	Resolve            = 3,
	FmaskDecompress    = 5,
	DccDecompress      = 6,
};

bool RenderExecutor::ConsumeMetadataColorOperation(CommandBuffer& buffer) {
	const auto& ctx  = buffer.GetRegisters();
	const auto  mode = ctx.GetColorControl().mode;
	if (mode == static_cast<uint8_t>(CbColorMode::EliminateFastClear)) {
		// FCE can precede any ordinary draw. Discover the target to materialize its native
		// clear metadata before the expanded image is sampled.
		RenderColorInfo target {};
		ResolveRenderColorTarget(buffer, target, 0, true);
		return true;
	}
	// Metadata/decompression shaders must not export colors into the expanded image.
	return mode == static_cast<uint8_t>(CbColorMode::FmaskDecompress) ||
	       mode == static_cast<uint8_t>(CbColorMode::DccDecompress);
}

// The task shader's inputs for an indirect mesh draw: Params in gpu_mesh_indirect.task.
struct MeshTaskParams {
	vk::DeviceAddress records        = 0;
	uint64_t          index_address  = 0;
	uint32_t          stride         = 0;
	uint32_t          index_size     = 0;
	uint32_t          index_limit    = 0;
	uint32_t          primitive_size = 0;
	uint32_t          group_step     = 0;
	uint32_t          max_groups     = 0;
	uint32_t          max_instances  = 0;
	uint32_t          max_total      = 0;
};

static uint64_t VertexBufferDescriptorSize(int binding, const ShaderVertexInputInfo& info) {
	const auto& buffer = info.buffers[binding];
	if (buffer.stride != 0 || buffer.num_records == 0) {
		return static_cast<uint64_t>(buffer.stride) * buffer.num_records;
	}

	uint64_t size = 0;
	for (int i = 0; i < info.resources_num; i++) {
		if (info.resources_dst[i].buffer_index != binding) {
			continue;
		}
		const auto& resource = info.resources[i];
		// RDNA2 OOB_SELECT=2 only checks NumRecords != 0. A constant attribute still
		// fetches its entire format; NumRecords is not a byte count in this mode.
		const uint64_t extent = resource.OutOfBounds() == 2
		                            ? resource.Base48() - buffer.addr +
		                                  ShaderRecompiler::Format::GetFormatInfo(resource.Format()).byte_size
		                            : buffer.num_records;
		size = std::max(size, extent);
	}
	return size;
}

struct VertexBufferRange {
	uint64_t                     base_address  = 0;
	uint64_t                     requested_end = 0;
	uint64_t                     acquired_end  = 0;
	std::pair<Buffer*, uint64_t> binding;

	[[nodiscard]] uint64_t RequestedSize() const { return requested_end - base_address; }
};

static PreparedVertexBuffers AcquireVertexBuffers(CommandBuffer&               buffer,
                                                  const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX);

	// Collect the non-empty guest vertex ranges.
	std::array<uint64_t, ShaderVertexInputInfo::RES_MAX>          sizes {};
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> ranges {};
	uint32_t                                                      range_count = 0;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = VertexBufferDescriptorSize(i, vs_input_info);
		sizes[i]           = size;
		if (size == 0) {
			continue;
		}
		if (vertex.addr == 0 || size > UINT64_MAX - vertex.addr) {
			EXIT("invalid vertex buffer range: addr=0x%016" PRIx64 " size=0x%016" PRIx64 "\n",
			     vertex.addr, size);
		}
		ranges[range_count++] = {vertex.addr, vertex.addr + size};
	}

	std::sort(ranges.begin(), ranges.begin() + range_count,
	          [](const VertexBufferRange& left, const VertexBufferRange& right) {
		          return left.base_address < right.base_address;
	          });

	// Merge overlapping or touching ranges before acquiring host buffers.
	std::array<VertexBufferRange, ShaderVertexInputInfo::RES_MAX> merged_ranges {};
	uint32_t                                                      merged_count = 0;
	for (uint32_t i = 0; i < range_count; i++) {
		const auto& range = ranges[i];
		if (merged_count != 0 &&
		    merged_ranges[merged_count - 1].requested_end >= range.base_address) {
			merged_ranges[merged_count - 1].requested_end =
			    std::max(merged_ranges[merged_count - 1].requested_end, range.requested_end);
			continue;
		}
		merged_ranges[merged_count++] = {range.base_address, range.requested_end};
	}

	auto& cache = buffer.GetContext().GetBufferCache();
	for (uint32_t i = 0; i < merged_count; i++) {
		auto& range = merged_ranges[i];
		// PPSA20298
		const auto size =
		    Libs::LibKernel::Memory::ClampRangeSize(range.base_address, range.RequestedSize());
		range.acquired_end = range.base_address + size;
		range.binding      = cache.ObtainBuffer(range.base_address, size, false);
	}

	// Rebuild slot bindings, offsetting non-empty slots into their acquired merged range.
	PreparedVertexBuffers prepared;
	prepared.count         = static_cast<uint32_t>(vs_input_info.buffers_num);
	vk::Buffer null_buffer = nullptr;
	for (int i = 0; i < vs_input_info.buffers_num; i++) {
		const auto& vertex = vs_input_info.buffers[i];
		const auto  size   = sizes[i];
		if (size == 0) {
			if (null_buffer == nullptr) {
				null_buffer = cache.GetBuffer(NULL_BUFFER_ID).Handle();
			}
			prepared.buffers[i] = null_buffer;
			prepared.offsets[i] = 0;
			continue;
		}

		const auto range = std::find_if(merged_ranges.begin(), merged_ranges.begin() + merged_count,
		                                [&](const VertexBufferRange& value) {
			                                return vertex.addr >= value.base_address &&
			                                       vertex.addr < value.acquired_end;
		                                });
		if (range == merged_ranges.begin() + merged_count) {
			EXIT("vertex buffer address is outside the acquired range: addr=0x%016" PRIx64 "\n",
			     vertex.addr);
		}

		prepared.buffers[i] = range->binding.first->Handle();
		prepared.offsets[i] = range->binding.second + vertex.addr - range->base_address;
		prepared.sizes[i]   = std::min(size, range->acquired_end - vertex.addr);
	}

	return prepared;
}

static bool GetDrawTopology(const HW::UserConfig& ucfg, vk::PrimitiveTopology& topology) {

	topology = vk::PrimitiveTopology::ePointList;

	switch (ucfg.GetPrimType()) {
		case Prospero::PrimitiveType::kNone: return false;
		case Prospero::PrimitiveType::kPointList:
			topology = vk::PrimitiveTopology::ePointList;
			break;
		case Prospero::PrimitiveType::kLineList: topology = vk::PrimitiveTopology::eLineList; break;
		case Prospero::PrimitiveType::kLineStrip:
			topology = vk::PrimitiveTopology::eLineStrip;
			break;
		case Prospero::PrimitiveType::kTriList:
			topology = vk::PrimitiveTopology::eTriangleList;
			break;
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kPolygon:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		case Prospero::PrimitiveType::kTriStrip:
			topology = vk::PrimitiveTopology::eTriangleStrip;
			break;
		case Prospero::PrimitiveType::kPatch:
			if (!Config::TessellationEnabled()) return false;
			[[fallthrough]];
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kRectListLegacy:
			topology = vk::PrimitiveTopology::ePatchList;
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			topology = vk::PrimitiveTopology::eTriangleFan;
			break;
		default: {
			static std::atomic_bool logged = false;
			if (!logged.exchange(true, std::memory_order_relaxed)) {
				std::printf("Skipping draw with unknown primitive type: %u\n",
				            static_cast<uint32_t>(ucfg.GetPrimType()));
			}
			return false;
		}
	}

	return true;
}

static bool ResolvePrimitiveRestart(const CommandBuffer& buffer,
                                    const DrawIndexBufferSource& source) {
	const auto control = buffer.GetUserConfig().GetPrimitiveResetControl();
	EXIT_NOT_IMPLEMENTED((control & ~0x3u) != 0);
	if ((control & 0x1u) == 0) {
		return false;
	}
	switch (buffer.GetUserConfig().GetPrimType()) {
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip: break;
		default: return false;
	}

	const auto element_size = source.guest_element_size;
	const auto index_mask   = UINT32_MAX >> ((4 - element_size) * 8);
	const auto reset_index  = buffer.GetRegisters().GetPrimitiveResetIndex();
	if ((control & 0x2u) != 0 && (reset_index & ~index_mask) != 0) {
		return false;
	}
	const auto restart_index = reset_index & index_mask;
	if (restart_index == index_mask) {
		// Use native restart; the 8-bit path widens its marker to 0xffff.
		return true;
	}

	// A game can set a custom reset value without using it in the index buffer.
	// Keep restart off in that case; fail if we actually find the value.
	// Scan before preparing draw resources: readback can restart the command buffer.
	EXIT_NOT_IMPLEMENTED(source.address == 0);
	const auto* indices = reinterpret_cast<const uint8_t*>(source.address);
	for (uint64_t offset = 0; offset < source.size; offset += element_size) {
		uint32_t index = 0;
		std::memcpy(&index, indices + offset, element_size);
		EXIT_NOT_IMPLEMENTED(index == restart_index);
	}
	return false;
}

bool RenderExecutor::PrepareDrawRenderState(CommandBuffer& buffer, const DrawCallInfo& draw,
                                            DrawRenderState& state) {
	m_context.GetImageHeap().BeginDraw();
	m_context.GetPipelineCache().GetGraphicsPrograms(buffer, state);
	// Vertex programs beyond the host mesh limits or with unsupported call targets are not
	// compiled.
	if (state.vertex_info[0].stage.module == nullptr) {
		return false;
	}
	for (uint32_t slot = 0; slot < RENDER_COLOR_ATTACHMENTS_MAX; slot++) {
		if ((state.mrt_mask & (1u << slot)) != 0) {
			ResolveRenderColorTarget(buffer, state.color_info[state.color_count], slot);
			if (state.color_info[state.color_count].image_id) {
				state.color_count++;
			}
		}
	}
	ResolveRenderDepthTarget(buffer, state.depth_info);
	// TODO(PPSA09243): Vulkan passes take one sample count. Where PS5 attachments mix counts,
	// keep those matching the raster count instead of resolving multisampled shadows.
	const auto colors       = std::span {state.color_info, state.color_count};
	const auto pass_samples = state.depth_info.image_id ? state.depth_info.desc.info.samples
	                                                    : state.color_info[0].desc.info.samples;
	if (std::ranges::any_of(colors, [&](const RenderColorInfo& color) {
		    return color.desc.info.samples != pass_samples;
	    })) {
		const auto raster =
		    render_sample_count(buffer.GetRegisters().GetAaConfig().msaa_num_samples);
		const auto dropped = std::ranges::remove_if(colors, [&](const RenderColorInfo& color) {
			return color.desc.info.samples != raster;
		});
		state.color_count -= static_cast<uint32_t>(dropped.size());
		if (state.depth_info.image_id && state.depth_info.desc.info.samples != raster) {
			state.depth_info = {};
		}
		static std::atomic_bool logged = false;
		if (!logged.exchange(true, std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(
			    "Warning: dropping attachments whose sample count differs from the raster.\n");
		}
	}

	if (state.color_count == 0 && !state.depth_info.image_id && !state.ps_active) {
		LogFramebufferSkip(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   draw.index_count, 0);
		return false;
	}

	return true;
}

static PreparedIndexBuffer PrepareIndexBuffer(CommandBuffer&               buffer,
                                              const DrawIndexBufferSource& source) {
	PreparedIndexBuffer prepared;
	if (source.size == 0) {
		return prepared;
	}
	prepared.type =
	    source.guest_element_size == 4 ? vk::IndexType::eUint32 : vk::IndexType::eUint16;
	if (source.host_data != nullptr) {
		auto& stream = buffer.GetContext().GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
		prepared.offset = stream.Copy(source.host_data, source.size, 16);
		prepared.buffer = stream.Handle();
	} else {
		auto [buffer_ptr, offset] =
		    buffer.GetContext().GetBufferCache().ObtainBuffer(source.address, source.size, false);
		prepared.buffer = buffer_ptr->Handle();
		prepared.offset = offset;
	}
	return prepared;
}

static void CommitVertexBuffers(vk::CommandBuffer            vk_buffer,
                                const PreparedVertexBuffers& prepared) {
	if (prepared.count != 0) {
		// Guest descriptor bounds must survive allocation merging in the cache.
		vk_buffer.bindVertexBuffers2(0, prepared.count, prepared.buffers.data(),
		                             prepared.offsets.data(), prepared.sizes.data(), nullptr);
	}
}

static void CommitIndexBuffer(vk::CommandBuffer vk_buffer, const PreparedIndexBuffer& prepared) {
	if (prepared.buffer == nullptr) {
		return;
	}
	vk_buffer.bindIndexBuffer(prepared.buffer, prepared.offset, prepared.type);
}

static void LogDrawStateIfNeeded(const CommandBuffer& buffer, const DrawCallInfo& draw,
	                             const DrawRenderState& state, uint32_t index_type_and_size,
                                 const void* index_addr) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}

	if (!draw.indexed && !Prospero::IsRectList(buffer.GetUserConfig().GetPrimType())) {
		return;
	}

	if (state.ps_active) {
		LogDrawTargetState(draw.Name(), state.color_info[0], state.depth_info, buffer,
		                   state.ps_input_info, draw.index_count, 0);
	}
	LogDrawInputState(buffer, state.color_info[0], state.vertex_info[0], index_type_and_size,
	                  draw.index_count, index_addr);
}

// Records the draw calls of a plain draw. The primitive type is the one the draw was met with.
static void EmitDrawPrimitives(Prospero::PrimitiveType primitive_type, vk::CommandBuffer vk_buffer,
                               const DrawCallInfo& draw, const DrawEmitInfo& emit) {
	switch (primitive_type) {
		case Prospero::PrimitiveType::kPointList:
		case Prospero::PrimitiveType::kLineList:
		case Prospero::PrimitiveType::kLineStrip:
		case Prospero::PrimitiveType::kTriList:
		case Prospero::PrimitiveType::kTriFan:
		case Prospero::PrimitiveType::kTriStrip:
		case Prospero::PrimitiveType::kRectList:
		case Prospero::PrimitiveType::kPolygon:
		case Prospero::PrimitiveType::kRectListLegacy:
		case Prospero::PrimitiveType::kPatch:
			if (draw.indexed) {
				vk_buffer.drawIndexed(draw.index_count, draw.instance_count, 0, emit.vertex_offset,
				                      emit.first_instance);
			} else {
				vk_buffer.draw(draw.index_count, draw.instance_count,
				               static_cast<uint32_t>(emit.vertex_offset), emit.first_instance);
			}
			break;
		case Prospero::PrimitiveType::kQuadListLegacy:
			EXIT_NOT_IMPLEMENTED((draw.index_count & 0x3u) != 0);
			for (uint32_t i = 0; i < draw.index_count; i += 4) {
				if (draw.indexed) {
					vk_buffer.drawIndexed(4, draw.instance_count, i, emit.vertex_offset,
					                      emit.first_instance);
				} else {
					vk_buffer.draw(4, draw.instance_count,
					               i + static_cast<uint32_t>(emit.vertex_offset),
					               emit.first_instance);
				}
			}
			break;
		default: EXIT("unknown primitive type: %u\n", static_cast<uint32_t>(primitive_type));
	}
}

bool RenderExecutor::CopyDepthToColor(CommandBuffer& buffer, const DrawCallInfo& draw,
                                       const DrawEmitInfo& emit, DrawRenderState& state) {
	if (!buffer.GetRegisters().GetRenderControl().copy_depth_to_color) {
		return false;
	}
	auto& ucfg = buffer.GetUserConfig();
	const auto& hw = buffer.GetRegisters();
	auto& depth = state.depth_info;
	auto& color = state.color_info[0];
	EXIT_NOT_IMPLEMENTED(
	    !m_context.GetGraphics().maintenance8_enabled ||
	    !Prospero::IsRectList(ucfg.GetPrimType()) || draw.indexed || draw.index_count != 3 ||
	    draw.instance_count != 1 || emit.vertex_offset != 0 || emit.first_instance != 0 ||
	    state.color_count != 1 || color.target_slot != 0 || !depth.image_id ||
	    color.desc.info.samples != 1 || depth.desc.info.extent.width != color.Extent().width ||
	    depth.desc.info.extent.height != color.Extent().height ||
	    depth.desc.view_info.layer_count != 1 || color.desc.view_info.layer_count != 1 ||
	    hw.GetBlendControl(0).enable || state.ps_input_info.ps_pixel_kill_enable ||
	    state.ps_input_info.ps_depth_export_enable ||
	    state.ps_input_info.ps_sample_mask_export_enable);
	vk::ImageAspectFlags feedback;
	const auto rendering = AcquireRenderTargets(buffer, &color, 1, depth, feedback);
	if (depth.load_clear_aspects) {
		buffer.BeginRendering(rendering);
	}
	auto& cache = m_context.GetTextureCache();
	auto& source = cache.GetImage(depth.image_id);
	auto& destination = cache.GetImage(color.image_id);
	EXIT_NOT_IMPLEMENTED((source.backing.format != vk::Format::eD32Sfloat &&
	                      source.backing.format != vk::Format::eD32SfloatS8Uint) ||
	                     destination.backing.format != vk::Format::eR32Sfloat);
	const auto rect = calc_final_scissor(hw.GetScreenViewport(), hw.GetScanModeControl(),
	                                    {rendering.width, rendering.height}, 0);
	if (rect.right > rect.left && rect.bottom > rect.top) {
		const vk::ImageCopy region {
		    {vk::ImageAspectFlagBits::eDepth, depth.desc.view_info.base_level,
		     depth.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {vk::ImageAspectFlagBits::eColor, color.desc.view_info.base_level,
		     color.desc.view_info.base_layer, 1}, {rect.left, rect.top, 0},
		    {static_cast<uint32_t>(rect.right - rect.left),
		     static_cast<uint32_t>(rect.bottom - rect.top), 1}};
		buffer.EndRendering();
		const auto command = buffer.Handle();
		source.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead,
		               {}, command);
		destination.Transit(vk::ImageLayout::eTransferDstOptimal,
		                    vk::AccessFlagBits2::eTransferWrite, {}, command);
		command.copyImage(source.backing.image, vk::ImageLayout::eTransferSrcOptimal,
		                  destination.backing.image, vk::ImageLayout::eTransferDstOptimal,
		                  1, &region);
	}
	return true;
}

// The mesh facts of a draw that decide how it launches.
struct MeshDrawPlan {
	bool     active = false;
	bool     task   = false;
	uint32_t groups = 0;
};

// Stops on a mesh draw the host cannot launch as the guest asked.
static void CheckMeshDrawSupported(const HW::UserConfig& ucfg, const DrawCallInfo& draw,
                                   const DrawRenderState& state) {
	const bool  primitive_restart_enable = state.primitive_restart_enable;
	const auto& mesh                     = state.vertex_info[0].mesh;
	EXIT_NOT_IMPLEMENTED(mesh.fast_launch && (draw.indexed || primitive_restart_enable));
	static std::atomic_bool restart_warned = false;
	if (primitive_restart_enable && !restart_warned.exchange(true, std::memory_order_relaxed)) {
		std::printf("Warning: primitive restart is not implemented for mesh shaders; "
		            "continuing draw (primitive=%u indexed=%u)\n",
		            static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed);
	}
	if (mesh.primitives_per_group == 0) {
		EXIT("unsupported mesh draw: primitive=%u indexed=%u restart=%u\n",
		     static_cast<uint32_t>(ucfg.GetPrimType()), draw.indexed, primitive_restart_enable);
	}
}

// Works out how a draw launches. Nothing comes back when a mesh draw has no primitives or no
// instances, because there is nothing to record for it.
static std::optional<MeshDrawPlan> PlanMeshDraw(const HW::UserConfig& ucfg,
                                                const GraphicContext& graphics,
                                                const DrawCallInfo& draw,
                                                const DrawRenderState& state,
                                                const DrawEmitInfo& emit) {
	const auto&  stage = state.vertex_info[0].stage;
	MeshDrawPlan plan {.active = stage.module->info.stage == ShaderType::Mesh,
	                   .task   = stage.module->mesh_task};
	const auto& limits = graphics.mesh_shader_properties;
	if (plan.active) {
		CheckMeshDrawSupported(ucfg, draw, state);
	}
	if (plan.task) {
		EXIT_NOT_IMPLEMENTED(emit.indirect.count > std::min(limits.maxTaskWorkGroupCount[0],
		                                                    limits.maxTaskWorkGroupTotalCount));
		return plan;
	}
	if (!plan.active) {
		return plan;
	}
	const auto& mesh       = state.vertex_info[0].mesh;
	const auto  primitives = mesh.InputPrimitiveCount(draw.index_count);
	if (primitives == 0 || draw.instance_count == 0) {
		return std::nullopt;
	}
	plan.groups = (primitives - 1u) / mesh.primitives_per_group + 1u;
	if (plan.groups > limits.maxMeshWorkGroupCount[0] ||
	    draw.instance_count > limits.maxMeshWorkGroupCount[1] ||
	    static_cast<uint64_t>(plan.groups) * draw.instance_count >
	        limits.maxMeshWorkGroupTotalCount) {
		EXIT("mesh draw exceeds host workgroup limits: %ux%u\n", plan.groups, draw.instance_count);
	}
	return plan;
}

// The buffer the GPU reads a draw's indirect records from, or nothing for a plain draw.
static std::pair<Buffer*, uint64_t> AcquireIndirectArgs(RenderContext&     context,
                                                         const DrawCallInfo& draw,
                                                         const DrawEmitInfo& emit) {
	if (emit.indirect.address == 0) {
		return {};
	}
	const auto size = static_cast<uint64_t>(emit.indirect.count - 1) * emit.indirect.stride +
	                  (draw.indexed ? sizeof(vk::DrawIndexedIndirectCommand)
	                                : sizeof(vk::DrawIndirectCommand));
	auto args = context.GetBufferCache().ObtainBuffer(emit.indirect.address, size, false);
	EXIT_IF((args.second & 3u) != 0);
	return args;
}

// Puts the inputs of an indirect mesh draw's task shader in the stream buffer, and returns where.
static vk::DeviceAddress UploadMeshTaskParams(RenderContext& context, const DrawCallInfo& draw,
                                              const DrawEmitInfo&                 emit,
                                              const DrawIndexBufferSource&        index_source,
                                              const std::pair<Buffer*, uint64_t>& args,
                                              const ShaderVertexInputInfo&        vertex) {
	const auto&          limits = context.GetGraphics().mesh_shader_properties;
	const auto&          mesh   = vertex.mesh;
	const MeshTaskParams params {
	    .records        = args.first->BufferDeviceAddress() + args.second,
	    .index_address  = index_source.address,
	    .stride         = emit.indirect.stride,
	    .index_size     = index_source.guest_element_size,
	    .index_limit    = draw.index_count,
	    .primitive_size = mesh.InputPrimitiveSize(),
	    .group_step     = mesh.primitives_per_group * mesh.InputPrimitiveStep(),
	    .max_groups     = limits.maxMeshWorkGroupCount[0],
	    .max_instances  = limits.maxMeshWorkGroupCount[1],
	    .max_total      = limits.maxMeshWorkGroupTotalCount,
	};
	auto& stream = context.GetBufferCache().GetUtilityBuffer(MemoryUsage::Stream);
	return stream.BufferDeviceAddress() + stream.Copy(&params, sizeof(params), 8);
}

// The stages of a draw that write buffers, which the draw must fence off from the memory's users.
static vk::PipelineStageFlags DrawShaderWriteStages(const DrawRenderState& state) {
	vk::PipelineStageFlags shader_write_stages = {};
	for (const auto& stage: state.VertexStages()) {
		if (HasShaderBufferWrites(stage.stage)) {
			shader_write_stages |= ShaderPipelineStages(NativeShaderStage(stage.logical_stage));
		}
	}
	if (state.ps_active && HasShaderBufferWrites(state.ps_input_info.stage)) {
		shader_write_stages |= vk::PipelineStageFlagBits::eFragmentShader;
	}
	return shader_write_stages;
}

// Copies what recording reads from the registers into the view, and updates the dynamic state that
// the recording commits. Both read the guest state, which belongs to this draw only until the next.
static void CaptureDrawState(RenderContext& context, CommandBuffer& buffer,
                             const DrawRenderState& state, const MeshDrawPlan& mesh,
                             vk::ImageAspectFlags feedback_aspects, DrawView& view) {
	using ShaderRecompiler::IR::PushData;
	const auto& ucfg          = buffer.GetUserConfig();
	const auto  vertex_stages = state.VertexStages();
	view.primitive_type       = ucfg.GetPrimType();
	view.object_id            = ucfg.GetObjectId();
	view.draw_dwords =
	    PushData::GraphicsDrawDwords(mesh.active, state.ps_input_info.object_primitive_id);
	view.user_clip_planes = vertex_stages.back().user_clip_planes;
	if (view.user_clip_planes != 0) {
		view.user_clip_plane_dwords = buffer.GetRegisters().GetUserClipPlanes();
	}
	auto& dynamic_state = context.GetCommandScheduler().GetDynamicState();
	UpdateGraphicsDynamicState(buffer, dynamic_state, vertex_stages.back(), state.depth_info,
	                           view.rendering);
	dynamic_state.SetAttachmentFeedbackLoopAspects(feedback_aspects);
	view.dynamic_state       = &dynamic_state;
	view.shader_write_stages = DrawShaderWriteStages(state);
}

// Resolves the bindings of every stage the draw runs, and lists them in the view.
void RenderExecutor::PrepareDrawBindings(DrawRenderState& state, DrawView& view) {
	auto&      bindings      = m_graphics_bindings;
	const auto vertex_stages = state.VertexStages();
	for (uint32_t i = 0; i < vertex_stages.size(); i++) {
		PrepareBindings(state.vertex_info[i].stage, bindings.vertex[i]);
		view.bindings[view.stage_count++] = &bindings.vertex[i];
	}
	if (state.ps_active) {
		PrepareBindings(state.ps_input_info.stage, bindings.pixel);
		view.bindings[view.stage_count++] = &bindings.pixel;
	}
	PrepareGraphicsBindings(view.Bindings(), std::span {state.color_info, state.color_count});
	for (uint32_t i = 0; i < view.stage_count; i++) {
		view.modules[i] = view.bindings[i]->Runtime().module;
	}
}

// Copies the plain facts of a draw into its view.
static void CaptureDrawFacts(const DrawCallInfo& draw, const DrawEmitInfo& emit,
                             const DrawIndexBufferSource& index_source, const MeshDrawPlan& mesh,
                             const std::pair<Buffer*, uint64_t>& args, DrawView& view) {
	view.draw               = draw;
	view.emit               = emit;
	view.mesh_active        = mesh.active;
	view.mesh_task          = mesh.task;
	view.mesh_groups        = mesh.groups;
	view.index_address      = index_source.address;
	view.index_element_size = index_source.guest_element_size;
	if (args.first != nullptr) {
		view.args_buffer = args.first->Handle();
		view.args_offset = args.second;
	}
}

// The commands a draw records at the moment it is met. The image transitions change layouts the CPU
// tracks, so they cannot wait for the pipeline. The vertex buffers go first when the caller wants
// the commands in their old order. A draw with no pipeline yet passes null, and it never uses the
// image heap, since that is a reason not to defer it.
void RenderExecutor::RecordEagerDrawCommands(CommandBuffer& buffer, DrawRenderState& state,
                                             const MeshDrawPlan& mesh,
                                             const PipelineCache::Pipeline* pipeline,
                                             vk::ImageAspectFlags feedback_aspects,
                                             bool bind_vertex_buffers, DrawView& view) {
	if (bind_vertex_buffers) {
		CommitVertexBuffers(buffer.Handle(), view.vertex_buffers);
		view.vertex_buffers_bound = true;
	}
	if (pipeline != nullptr && pipeline->uses_image_heap) {
		m_context.GetImageHeap().Commit(buffer, vk::PipelineBindPoint::eGraphics,
		                                pipeline->pipeline_layout);
	}
	TransitBindingImages(buffer, view.Bindings());
	CaptureDrawState(m_context, buffer, state, mesh, feedback_aspects, view);
	view.address_buffers = CaptureAddressBuffers(view.Modules());
}

// Registers the original guest indices of an indexed mesh draw for shader reads. Shaders read them
// only through BDA, and PrepareGraphicsBindings synchronizes registered BDA ranges before any draw
// commands are committed, so the lookup counts as a use and keeps the buffer resident.
static void RegisterMeshIndices(RenderContext& context, const DrawCallInfo& draw,
                                const DrawIndexBufferSource& index_source) {
	(void)context.GetBufferCache().FindBufferForShaderRead(
	    index_source.address,
	    static_cast<uint64_t>(draw.index_count) * index_source.guest_element_size);
}

// Does everything a draw must do in order when it is met. That covers the resources it reads, its
// targets and the image layouts it needs, and the state it takes from the registers. What is left
// for RecordDraw is plain commands. Returns the pipeline to record with, or null when nothing is
// left to record because the draw was skipped or has no work. With `deferred` given, a draw may
// instead wait for its pipeline. Then it returns null with `deferred` set, and the view holds a
// draw whose vertex buffers are not bound, for the caller to turn into a hole.
const PipelineCache::Pipeline* RenderExecutor::PrepareDraw(
    CommandBuffer& buffer, const DrawCallInfo& draw, DrawRenderState& state,
    const DrawEmitInfo& emit, const DrawIndexBufferSource& index_source,
    bool bind_vertex_buffers, DrawView& view, PipelineCache::DeferredDraw* deferred) {
	const auto mesh =
	    PlanMeshDraw(buffer.GetUserConfig(), m_context.GetGraphics(), draw, state, emit);
	if (!mesh) {
		return nullptr;
	}
	if (mesh->active && draw.indexed) {
		RegisterMeshIndices(m_context, draw, index_source);
	}
	PrepareDrawBindings(state, view);
	if (!mesh->active) {
		view.vertex_buffers = AcquireVertexBuffers(buffer, state.vertex_info[0]);
		view.index_buffer   = PrepareIndexBuffer(buffer, index_source);
	}
	const auto args     = AcquireIndirectArgs(m_context, draw, emit);
	auto*      pipeline = AcquireDrawPipeline(m_context, buffer, state,
	                                          {mesh->active, args.first != nullptr, mesh->task}, deferred);
	const bool is_hole  = deferred != nullptr && deferred->active;
	vk::ImageAspectFlags feedback_aspects;
	// Runs even for a skipped draw, since it applies the clears the draw consumes and moves the
	// targets to their attachment layouts.
	view.rendering = AcquireRenderTargets(buffer, state.color_info, state.color_count,
	                                      state.depth_info, feedback_aspects, view.Bindings());
	if (pipeline == nullptr && !is_hole) {
		// Returns normally, because the caller treats a false result as a request to replay the
		// draw on the CPU.
		SkipDrawWithoutPipeline(m_context, buffer, view.rendering, state);
		return nullptr;
	}
	view.task_params        = mesh->task ? UploadMeshTaskParams(m_context, draw, emit, index_source,
	                                                            args, state.vertex_info[0])
	                                     : 0;
	CaptureDrawFacts(draw, emit, index_source, *mesh, args, view);
	// Resource preparation above may synchronously finish and restart the scheduler. From this
	// point onward, every operation targets the current command buffer and cannot touch guest
	// memory. A hole's own buffer binds the vertex buffers, since the segment's would come first.
	RecordEagerDrawCommands(buffer, state, *mesh, pipeline, feedback_aspects,
	                        bind_vertex_buffers && !is_hole, view);
	return pipeline;
}

// Records the draw's mesh, object id and clip plane push constants.
static void RecordDrawPushConstants(vk::CommandBuffer vk_buffer, const DrawView& view,
                                    const PipelineCache::Pipeline& pipeline) {
	using ShaderRecompiler::IR::PushData;
	if (view.draw_dwords != 0) {
		std::array<uint32_t, PushData::MeshDrawDwordCount> draw_data {};
		if (view.mesh_task) {
			draw_data[0] = static_cast<uint32_t>(view.task_params);
			draw_data[1] = static_cast<uint32_t>(view.task_params >> 32u);
		} else if (view.mesh_active) {
			draw_data = {view.draw.index_count,
			             static_cast<uint32_t>(view.emit.vertex_offset),
			             view.index_element_size,
			             static_cast<uint32_t>(view.index_address),
			             static_cast<uint32_t>(view.index_address >> 32u)};
		} else {
			draw_data[0] = view.object_id;
		}
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages, 0,
		                        view.draw_dwords * sizeof(uint32_t), draw_data.data());
	}
	if (view.user_clip_planes != 0) {
		// Each plane up to the highest enabled one takes four dwords.
		vk_buffer.pushConstants(pipeline.pipeline_layout, pipeline.push_stages,
		                        PushData::UserClipPlaneDword * sizeof(uint32_t),
		                        std::bit_width(view.user_clip_planes) * 4u * sizeof(uint32_t),
		                        view.user_clip_plane_dwords.data());
	}
}

// Records the draw call of a prepared draw.
static void RecordDrawCall(vk::CommandBuffer vk_buffer, const DrawView& view) {
	const auto& draw = view.draw;
	const auto& emit = view.emit;
	if (view.mesh_task) {
		vk_buffer.drawMeshTasksEXT(emit.indirect.count, 1, 1);
	} else if (view.mesh_active) {
		vk_buffer.drawMeshTasksEXT(view.mesh_groups, draw.instance_count, 1);
	} else if (view.args_buffer == nullptr) {
		EmitDrawPrimitives(view.primitive_type, vk_buffer, draw, emit);
	} else if (draw.indexed) {
		vk_buffer.drawIndexedIndirect(view.args_buffer, view.args_offset, emit.indirect.count,
		                              emit.indirect.stride);
	} else {
		vk_buffer.drawIndirect(view.args_buffer, view.args_offset, emit.indirect.count,
		                       emit.indirect.stride);
	}
}

void RenderExecutor::RecordDraw(vk::CommandBuffer vk_buffer, const DrawView& view,
                                const PipelineCache::Pipeline& pipeline, DrawRecordEnv& env) {
	if (!view.vertex_buffers_bound) {
		CommitVertexBuffers(vk_buffer, view.vertex_buffers);
	}
	WriteBindingDescriptors(vk_buffer, vk::PipelineBindPoint::eGraphics, pipeline, view.Bindings(),
	                        view.Modules(), view.address_buffers, env.Scratch());
	CommitIndexBuffer(vk_buffer, view.index_buffer);
	RecordDrawPushConstants(vk_buffer, view, pipeline);
	view.dynamic_state->Commit(m_context.GetGraphics(), vk_buffer);
	env.BeginRendering(vk_buffer, view.rendering);
	vk_buffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline.pipeline);
	const int32_t token = env.BeginDraw(vk_buffer);
	RecordDrawCall(vk_buffer, view);
	env.EndDraw(vk_buffer, token, view.draw);
}

namespace {

// Records a draw as it is met. That means into the scheduler's current command buffer, in a
// rendering it shares with the draws around it, and with its samples counted for any open guest
// occlusion query. The barriers that fence off a stage's buffer writes belong here, since they end
// the rendering that the wrapper tracks.
class BlockingDrawEnv final: public DrawRecordEnv {
public:
	BlockingDrawEnv(RenderContext& context, CommandBuffer& buffer, DescriptorScratch& scratch,
	                vk::PipelineStageFlags shader_write_stages)
	    : m_context(context), m_buffer(buffer), m_scratch(scratch),
	      m_shader_write_stages(shader_write_stages) {}

	DescriptorScratch& Scratch() override { return m_scratch; }

	void BeginRendering(vk::CommandBuffer command, const RenderState& rendering) override {
		if (m_shader_write_stages) {
			// As for dispatches, buffer writes wait for earlier accesses to the memory.
			m_buffer.EndRendering();
			ShaderWriteHazardBarrier(command, m_shader_write_stages);
		}
		m_buffer.BeginRendering(rendering);
	}

	int32_t BeginDraw(vk::CommandBuffer command) override {
		return m_context.GetOcclusionQueries().Begin(command);
	}

	void EndDraw(vk::CommandBuffer command, int32_t token, const DrawCallInfo& draw) override {
		m_context.GetOcclusionQueries().End(command, token);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances,
		                           draw.instance_count);
		m_context.GetCommandScheduler().NoteRecordedWork();
		if (m_shader_write_stages) {
			m_buffer.EndRendering();
			ShaderWriteBarrier(command, m_shader_write_stages);
		}
	}

private:
	RenderContext&         m_context;
	CommandBuffer&         m_buffer;
	DescriptorScratch&     m_scratch;
	vk::PipelineStageFlags m_shader_write_stages;
};

} // namespace

// Records a prepared draw as it is met, through an env that forwards to the command buffer wrapper.
void RenderExecutor::RecordBlockingDraw(CommandBuffer& buffer, const DrawView& view,
                                        const PipelineCache::Pipeline& pipeline) {
	BlockingDrawEnv env(m_context, buffer, m_descriptor_scratch, view.shader_write_stages);
	RecordDraw(buffer.Handle(), view, pipeline, env);
}

// Turns a prepared draw whose pipeline is still building into a hole in the command stream. The
// hole is recorded when the scheduler next submits, between the commands before it and after it.
// The caller must not use a command buffer it fetched before this call, since the scheduler ends
// it here and starts another.
void RenderExecutor::CreateDrawHole(CommandBuffer& buffer, DrawView& view,
                                    const GraphicsPipelineKey& key, const char* skip_verdict,
                                    std::shared_ptr<const Common::FirstRunReport::SkipTargets> skip_targets) {
	if (RenderStateClears(view.rendering)) {
		// The clear was consumed from the registers when the draw was met, so it is applied here, in
		// the pass before the hole. The hole's own pass loads, as it would after any earlier pass.
		buffer.BeginRendering(view.rendering);
		view.rendering.depth_stencil_attachment.depth_clear   = false;
		view.rendering.depth_stencil_attachment.stencil_clear = false;
	}
	auto& scheduler = m_context.GetCommandScheduler();
	// Counted where the draw is met, since the hole is recorded later and the cadence counts now.
	scheduler.NoteRecordedWork();
	scheduler.SplitForHole(std::make_unique<DrawHole>(view, key, skip_verdict, std::move(skip_targets)));
}

// Records one prepared draw. Resources are prepared and the targets acquired for every draw. With
// background pipeline builds on, a draw whose pipeline is not ready may then stop short of the
// commands that bind and draw, and still report a normal draw. With --deferred-pipeline-draws it
// may instead become a hole that is recorded once the pipeline is built.
void RenderExecutor::ExecutePreparedDraw(CommandBuffer& buffer, const DrawCallInfo& draw,
                                         DrawRenderState& state, const DrawEmitInfo& emit,
                                         const DrawIndexBufferSource& index_source) {
	if (CopyDepthToColor(buffer, draw, emit, state)) {
		return;
	}
	DrawView                    view;
	PipelineCache::DeferredDraw deferred;
	// The vertex buffers are bound ahead of the image transitions, where this draw always had them.
	const auto* pipeline = PrepareDraw(buffer, draw, state, emit, index_source, true, view,
	                                   Config::DeferredPipelineDrawsEnabled() ? &deferred : nullptr);
	if (deferred.active) {
		CreateDrawHole(buffer, view, deferred.key, deferred.skip_verdict, std::move(deferred.skip_targets));
		return;
	}
	if (pipeline != nullptr) {
		RecordBlockingDraw(buffer, view, *pipeline);
	}
}

// Vulkan offsets for a prepared draw, or nullopt when its indirect records need the CPU:
// quad-list draws for counts, tessellation draws for the index/SGPR prolog and mesh draws without
// room for the task payload. Indirect records carry their own offsets.
static std::optional<DrawEmitInfo> ResolveDrawEmit(const HW::UserConfig&        ucfg,
                                                   const ShaderVertexInputInfo& vs,
                                                   DrawOffsetSource             source,
                                                   int32_t                      first_vertex,
                                                   uint32_t                     first_instance,
                                                   const DrawIndirectArgs&      indirect) {
	if ((indirect.address != 0 || source == DrawOffsetSource::IndirectArgs) &&
	    ucfg.GetIndexOffset() != 0) {
		static std::once_flag warning_once;
		std::call_once(warning_once, [&] {
			Log::WriteToConsoleAndLog(fmt::format(
			    "GPU: indirect draw ignores GE_INDX_OFFSET=0x{:08x}; geometry may be incorrect.\n",
			    ucfg.GetIndexOffset()));
		});
	}
	if (indirect.address != 0) {
		if (!vs.indirect_draw ||
		    ucfg.GetPrimType() == Prospero::PrimitiveType::kQuadListLegacy ||
		    vs.indirect_offsets != indirect.offsets) {
			return std::nullopt;
		}
		return DrawEmitInfo {.indirect = indirect};
	}
	if (source == DrawOffsetSource::IndirectArgs) {
		// Native fetches consume the patched SGPRs; rewritten fetches need Vulkan offsets.
		return vs.has_dynamic_vertex_fetch ? DrawEmitInfo {first_vertex, first_instance} : DrawEmitInfo {};
	}
	const auto [vertex_offset, instance_offset] = ResolveDrawOffsets(ucfg.GetIndexOffset(), vs);
	return DrawEmitInfo {vertex_offset + first_vertex, instance_offset};
}

bool RenderExecutor::DrawIndex(CommandBuffer& buffer, const DrawIndexArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (args.index_count == 0 || (args.instance_count == 0 && args.indirect.address == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndex():Shader:\n");
		uc_print("GraphicsRenderDrawIndex():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndex():Parameters:\n"
		     "\t index_type_and_size = 0x%08" PRIx32 "\n"
		     "\t index_count         = 0x%08" PRIx32 "\n"
		     "\t index_addr          = 0x%016" PRIx64 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t base_vertex         = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.index_type_and_size, args.index_count,
		     reinterpret_cast<uint64_t>(args.index_addr), args.instance_count,
		     static_cast<uint32_t>(args.base_vertex), args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	DrawRenderState state {.indirect_offsets = args.indirect.offsets,
	                       .indirect_draw    = args.indirect.address != 0};
	if (!GetDrawTopology(ucfg, state.topology)) {
		return true;
	}

	DrawIndexBufferSource index_source {};
	index_source.address = reinterpret_cast<uint64_t>(args.index_addr);
	switch (static_cast<Prospero::IndexType>(args.index_type_and_size)) {
		case Prospero::IndexType::kIndex16: index_source.guest_element_size = 2; break;
		case Prospero::IndexType::kIndex32: index_source.guest_element_size = 4; break;
		case Prospero::IndexType::kIndex8: index_source.guest_element_size = 1; break;
		default: EXIT("unknown index_type_and_size: %u\n", args.index_type_and_size);
	}
	index_source.size = static_cast<uint64_t>(args.index_count) * index_source.guest_element_size;
	state.primitive_restart_enable = ResolvePrimitiveRestart(buffer, index_source);

	std::vector<uint16_t> expanded_indices;
	if (index_source.guest_element_size == 1) {
		EXIT_NOT_IMPLEMENTED(args.index_addr == nullptr);
		const auto* src = static_cast<const uint8_t*>(args.index_addr);
		expanded_indices.resize(args.index_count);
		for (uint32_t i = 0; i < args.index_count; i++) {
			expanded_indices[i] =
			    state.primitive_restart_enable && src[i] == 0xffu ? 0xffffu : src[i];
		}
		index_source.host_data = expanded_indices.data();
		index_source.size      = expanded_indices.size() * sizeof(uint16_t);
	}

	const DrawCallInfo draw {true, args.index_count, args.instance_count};
	if (!PrepareDrawRenderState(buffer, draw, state)) {
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, args.index_type_and_size,
	                     args.index_addr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  args.base_vertex, args.first_instance, args.indirect);
	if (emit) {
		ExecutePreparedDraw(buffer, draw, state, *emit, index_source);
	}
	ResetBindings();
	return emit.has_value();
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
bool RenderExecutor::DrawAuto(CommandBuffer& buffer, const DrawAutoArgs& args) {
	KYTY_PROFILER_FUNCTION();

	EXIT_IF(args.offset_source == DrawOffsetSource::DrawState && args.first_instance != 0);
	m_context.GetCommandScheduler().PopPendingOperations();
	auto& ucfg   = buffer.GetUserConfig();
	auto& sh_ctx = buffer.GetShaders();

	Common::LockGuard lock(m_context.GetMutex());
	if (args.indirect.address == 0 && (args.vertex_count == 0 || args.instance_count == 0)) {
		return true;
	}

	if (ConsumeMetadataColorOperation(buffer) || DepthStencilCopy(buffer) ||
	    ResolveColorTargets(buffer)) {
		ResetBindings();
		return true;
	}

	if (!DrawHasValidVertexShader(sh_ctx)) {
		return true;
	}

	if (graphics_debug_dump_enabled()) {
		LOGF("GraphicsRenderDrawIndexAuto():Shader:\n");
		uc_print("GraphicsRenderDrawIndexAuto():UserConfig:", ucfg);
		hw_print(buffer);

		LOGF("GraphicsRenderDrawIndexAuto():Parameters:\n"
		     "\t vertex_count        = 0x%08" PRIx32 "\n"
		     "\t instance_count      = 0x%08" PRIx32 "\n"
		     "\t first_vertex        = 0x%08" PRIx32 "\n"
		     "\t first_instance      = 0x%08" PRIx32 "\n",
		     args.vertex_count, args.instance_count, args.first_vertex, args.first_instance);
	}

	uc_check(ucfg);

	hw_check(buffer);

	const DrawCallInfo draw {false, args.vertex_count, args.instance_count};

	DrawRenderState state {.indirect_offsets = args.indirect.offsets,
	                       .indirect_draw    = args.indirect.address != 0};
	if (!GetDrawTopology(ucfg, state.topology)) {
		ResetBindings();
		return true;
	}
	if (!PrepareDrawRenderState(buffer, draw, state)) {
		ResetBindings();
		return true;
	}

	const bool rect_list = Prospero::IsRectList(ucfg.GetPrimType());
	if (rect_list && state.vertex_info[0].buffers_num == 0 &&
	    state.vertex_info[0].stage.module->info.param_export_mask == 0 &&
	    state.ps_input_info.input_num != 0) {
		if (graphics_debug_dump_enabled()) {
			LOGF("DrawIndexAuto: skipping rect-list draw with no VS param exports and PS inputs: "
			     "ps_inputs=%u ps=0x%016" PRIx64 " es=0x%016" PRIx64 " gs=0x%016" PRIx64 "\n",
			     state.ps_input_info.input_num, sh_ctx.GetPs().ps_regs.data_addr,
			     sh_ctx.GetVs().es_regs.data_addr, sh_ctx.GetVs().gs_regs.data_addr);
		}
		ResetBindings();
		return true;
	}

	LogDrawStateIfNeeded(buffer, draw, state, 0, nullptr);
	const auto emit = ResolveDrawEmit(ucfg, state.vertex_info[0], args.offset_source,
	                                  static_cast<int32_t>(args.first_vertex), args.first_instance,
	                                  args.indirect);
	if (emit) {
		ExecutePreparedDraw(buffer, draw, state, *emit, {});
	}
	ResetBindings();
	return emit.has_value();
}

bool RenderExecutor::ResolveColorTargets(CommandBuffer& buffer) {
	const auto& hw = buffer.GetRegisters();
	if (hw.GetColorControl().mode != 3) {
		return false;
	}

	const auto& src_rt = hw.GetRenderTarget(0);
	const auto& dst_rt = hw.GetRenderTarget(1);
	if (src_rt.base.addr == 0 || dst_rt.base.addr == 0) {
		return false;
	}

	RenderColorInfo src {};
	RenderColorInfo dst {};
	ResolveRenderColorTarget(buffer, src, 0, true, true);
	ResolveRenderColorTarget(buffer, dst, 1, true, true);
	if (src.desc.info.data.address == dst.desc.info.data.address &&
	    src.guest_mip_level == dst.guest_mip_level &&
	    src.guest_array_layer == dst.guest_array_layer) {
		return true;
	}

	auto& cache = m_context.GetTextureCache();
	cache.MarkGpuWritten(dst.image_id);
	auto& source      = cache.GetImage(src.image_id);
	auto& destination = cache.GetImage(dst.image_id);
	destination.Resolve(source,
	                    {src.desc.view_info.base_level, 1, src.desc.view_info.base_layer, 1},
	                    {dst.desc.view_info.base_level, 1, dst.desc.view_info.base_layer, 1});
	return true;
}

} // namespace Libs::Graphics
