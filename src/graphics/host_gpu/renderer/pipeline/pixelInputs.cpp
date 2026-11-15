#include "graphics/host_gpu/renderer/pipeline/pixelInputs.h"

#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <iterator>
#include <optional>

namespace Libs::Graphics {

namespace {

// The slot an object primitive id output always takes.
constexpr uint32_t OBJECT_PRIMITIVE_ID_SLOT = 7;
// The export mode a packed UNorm target needs, and the channels it must write.
constexpr uint8_t  PACKED_UNORM_EXPORT_MODE = 9;
constexpr uint32_t PACKED_UNORM_CHANNELS    = 7;
// Export modes that rule out a logical alpha blend source, which are no export and the one that
// carries no alpha.
constexpr uint8_t EXPORT_MODE_NONE      = 0;
constexpr uint8_t EXPORT_MODE_NO_ALPHA  = 7;
// MRT1 is where the second blend source goes, so it is cleared from the integer masks.
constexpr uint8_t SECOND_SOURCE_EXPORT_BIT = 2;

// Fills the mapping of each written target that has a base address, through a resolver that either
// EXITs or says it cannot. Returns false at the first target the resolver refuses.
template <typename Resolve>
bool FillTargetExportMapping(const HW::Context& context, uint32_t color_output_mask,
                             PixelTargetExportMapping& mapping, Resolve&& resolve) {
	mapping = {};
	for (uint32_t slot = 0; slot < PIXEL_COLOR_TARGETS; slot++) {
		const auto& rt = context.GetRenderTarget(slot);
		if ((color_output_mask & (1u << slot)) == 0 || rt.base.addr == 0) {
			continue;
		}
		const std::optional<RenderTargetFormatInfo> format = resolve(rt);
		if (!format.has_value()) {
			return false;
		}
		mapping[slot] = format->export_mapping;
	}
	return true;
}

// Adds the integer target masks, and the packed UNorm mask with its round modes, from each enabled
// export's target. Stops at a packed UNorm export the host cannot write.
PixelInputStatus AddTargetFormats(const HW::Context& context, ShaderPixelInputInfo& pixel_info,
                                  PackedUnormRejection* rejection) {
	for (uint32_t export_index = 0; export_index < PIXEL_COLOR_TARGETS; export_index++) {
		const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
		if (slot >= PIXEL_COLOR_TARGETS || pixel_info.target_output_mode[export_index] == 0u) continue;
		const auto& rt          = context.GetRenderTarget(slot);
		const auto  target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
		if (rt.base.addr == 0 || target_mask == 0u) continue;
		if (rt.info.channel_type == Prospero::ChannelType::kUInt) {
			pixel_info.target_uint_mask |= 1u << export_index;
		} else if (rt.info.channel_type == Prospero::ChannelType::kSInt) {
			pixel_info.target_sint_mask |= 1u << export_index;
		}
		const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
		if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
		const auto mapping     = pixel_info.target_export_mapping[export_index];
		const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
		const bool blend       = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
		if (blend || (mapping.ApplyMask(target_mask) & PACKED_UNORM_CHANNELS) != PACKED_UNORM_CHANNELS ||
		    (mapping.ApplyMask(shader_mask) & PACKED_UNORM_CHANNELS) != PACKED_UNORM_CHANNELS ||
		    pixel_info.target_output_mode[export_index] != PACKED_UNORM_EXPORT_MODE) {
			if (rejection != nullptr) {
				*rejection = {slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				              shader_mask, mapping.packed, blend};
			}
			return PixelInputStatus::UnsupportedPackedUnorm;
		}
		pixel_info.target_packed_unorm_mask |= 1u << export_index;
		pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
	}
	return PixelInputStatus::Ok;
}

// Which vertex the flat inputs come from. A rect list outside a mesh or tessellated draw interpolates
// as a rectangle.
ShaderPixelParameterMode ParameterModeOf(const HW::Context& context, const PixelDrawShape& shape) {
	if (!shape.mesh_active && !shape.tess_active && Prospero::IsRectList(shape.primitive_type)) {
		return ShaderPixelParameterMode::Rectangle;
	}
	return context.GetModeControl().provoking_vtx_last ? ShaderPixelParameterMode::LastVertex
	                                                   : ShaderPixelParameterMode::FirstVertex;
}

// Whether target 0's blend reads a second source, which the shader then writes to MRT1.
bool BlendUsesDualSource(const HW::BlendControl& blend) {
	return BlendFactorIsDualSource(blend.color_srcblend) || BlendFactorIsDualSource(blend.color_destblend) ||
	       (blend.separate_alpha_blend &&
	        (BlendFactorIsDualSource(blend.alpha_srcblend) || BlendFactorIsDualSource(blend.alpha_destblend)));
}

// The logical alpha a blend of target 0 needs from the shader when the target's channel mapping moves
// alpha, which only a lone export to target 0 can give through MRT1.
ShaderAlphaBlendSource AlphaBlendSourceOf(const HW::BlendControl& blend, const ShaderPixelInputInfo& pixel_info) {
	const bool lone_export = std::all_of(std::begin(pixel_info.target_output_mode) + 1,
	                                     std::end(pixel_info.target_output_mode),
	                                     [](uint8_t mode) { return mode == 0; });
	if (pixel_info.target_output_mode[0] == EXPORT_MODE_NONE ||
	    pixel_info.target_output_mode[0] == EXPORT_MODE_NO_ALPHA || !lone_export) {
		return ShaderAlphaBlendSource::None;
	}
	switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
		case BlendMappingSupport::SourceAlpha: return ShaderAlphaBlendSource::SourceAlpha;
		case BlendMappingSupport::SourceAlphaOne: return ShaderAlphaBlendSource::SourceAlphaOne;
		case BlendMappingSupport::SourceAlphaZero: return ShaderAlphaBlendSource::SourceAlphaZero;
		default: return ShaderAlphaBlendSource::None;
	}
}

// Sets dual-source blending and the alpha blend source from target 0's blend. Both make MRT1 carry the
// second source.
void AddBlendSource(const HW::Context& context, ShaderPixelInputInfo& pixel_info) {
	const auto& blend   = context.GetBlendControl(0);
	const bool  blended = blend.enable && !context.GetRenderTarget(0).info.blend_bypass;
	pixel_info.dual_source_blending = blended && BlendUsesDualSource(blend);
	if (pixel_info.dual_source_blending) {
		// MRT1 supplies the second blend source for target 0, and blended targets are not integer.
		pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
		pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		pixel_info.target_uint_mask &= ~SECOND_SOURCE_EXPORT_BIT;
		pixel_info.target_sint_mask &= ~SECOND_SOURCE_EXPORT_BIT;
		return;
	}
	if (!blended) {
		return;
	}
	pixel_info.alpha_blend_source = AlphaBlendSourceOf(blend, pixel_info);
	if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
		pixel_info.dual_source_blending     = true;
		pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
		pixel_info.target_export_mapping[1] = {};
	}
}

} // namespace

uint32_t PixelColorOutputMask(const HW::Context& context) {
	const auto& sh         = context.GetShaderRegisters();
	const auto  write_mask = context.GetRenderTargetMask() & sh.m_cbShaderMask;
	uint32_t    mask       = context.HasObjectPrimitiveIdOutput() ? 1u << OBJECT_PRIMITIVE_ID_SLOT : 0;
	for (uint32_t index = 0; index < PIXEL_COLOR_TARGETS; index++) {
		const auto slot = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		if (slot >= PIXEL_COLOR_TARGETS) {
			break;
		}
		if (sh.target_output_mode[index] != 0 && render_target_mask_slot(write_mask, slot) != 0) {
			mask |= 1u << slot;
		}
	}
	return mask;
}

bool PixelStageActive(const HW::Context& context, const HW::PixelShaderInfo& pixel_regs,
                      uint32_t color_output_mask) {
	const auto& sh = context.GetShaderRegisters();
	const auto& db = sh.db_shader_control;
	return pixel_regs.ps_regs.data_addr != 0 &&
	       (color_output_mask != 0 || db.shader_kill_enable || db.shader_z_export_enable ||
	        db.shader_mask_export_enable || db.shader_dual_export_enable || db.shader_execute_on_noop ||
	        sh.AlphaToMask());
}

bool TryPixelTargetExportMapping(const HW::Context& context, uint32_t color_output_mask,
                                 PixelTargetExportMapping& mapping) {
	return FillTargetExportMapping(context, color_output_mask, mapping, [](const HW::RenderTarget& rt) {
		return TryTextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order);
	});
}

PixelTargetExportMapping PixelTargetExportMappingOf(const HW::Context& context, uint32_t color_output_mask) {
	PixelTargetExportMapping mapping {};
	(void)FillTargetExportMapping(context, color_output_mask, mapping, [](const HW::RenderTarget& rt) {
		return std::optional(
		    TextureGetRenderTargetFormat(rt.info.format, rt.info.channel_type, rt.info.channel_order));
	});
	return mapping;
}

PixelInputStatus FinishPixelInputInfo(const HW::Context& context, const PixelDrawShape& shape,
                                      ShaderPixelInputInfo& pixel_info, PackedUnormRejection* rejection) {
	pixel_info.object_primitive_id = shape.object_primitive_id;
	if (AddTargetFormats(context, pixel_info, rejection) != PixelInputStatus::Ok) {
		return PixelInputStatus::UnsupportedPackedUnorm;
	}
	pixel_info.parameter_mode = ParameterModeOf(context, shape);
	AddBlendSource(context, pixel_info);
	return PixelInputStatus::Ok;
}

} // namespace Libs::Graphics
