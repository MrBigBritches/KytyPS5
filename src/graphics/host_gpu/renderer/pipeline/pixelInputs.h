#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIXEL_INPUTS_H
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIXEL_INPUTS_H

#include "graphics/guest_gpu/gpu_defs.h"

#include <array>
#include <cstdint>

// The draw-time part of a pixel shader's input info, which depends only on the context registers and
// the primitive. GetGraphicsPrograms uses it on the GPU thread, and the graphics prescan uses it on the
// scanner's shadow, so the two cannot disagree on a key. None of it EXITs, so the prescan can run it on
// state it cannot vouch for, and the GPU thread EXITs on the answers as it did before.
namespace Libs::Graphics {

namespace HW {
class Context;
struct PixelShaderInfo;
} // namespace HW
struct ShaderPixelInputInfo;

/// How many color targets a draw has, which is also how many export mappings the key holds.
inline constexpr uint32_t PIXEL_COLOR_TARGETS = 8;

/// The export mapping of each color target, by slot.
using PixelTargetExportMapping = std::array<Prospero::ColorComponentMapping, PIXEL_COLOR_TARGETS>;

/// The color target slots the pixel shader writes, as a bit per slot. A slot counts when its export is
/// enabled and both CB_TARGET_MASK and CB_SHADER_MASK leave a channel of it on. An object primitive id
/// output takes slot 7 whatever the masks say.
[[nodiscard]] uint32_t PixelColorOutputMask(const HW::Context& context);

/// Whether the pixel shader runs. It needs a program, and then color outputs or a depth or coverage
/// side effect, since a shader without color outputs still runs for those.
[[nodiscard]] bool PixelStageActive(const HW::Context& context, const HW::PixelShaderInfo& pixel_regs,
                                    uint32_t color_output_mask);

/// Fills the export mapping of each written color target with a base address, as the GPU thread does
/// before it prepares the pixel program, and leaves the rest default.
/// @return false where a target's format is one TextureGetRenderTargetFormat would EXIT on. The mapping
///         is then incomplete and must not be used.
[[nodiscard]] bool TryPixelTargetExportMapping(const HW::Context& context, uint32_t color_output_mask,
                                               PixelTargetExportMapping& mapping);

/// The same mapping, which EXITs on an unsupported target format as the GPU thread always has.
[[nodiscard]] PixelTargetExportMapping PixelTargetExportMappingOf(const HW::Context& context,
                                                                  uint32_t color_output_mask);

/// What a draw's primitive and vertex stage add to the pixel input info.
struct PixelDrawShape {
	/// VGT_PRIMITIVE_TYPE.
	Prospero::PrimitiveType primitive_type = Prospero::PrimitiveType::kTriList;
	/// Whether the draw outputs the object primitive id, see HW::Context::HasObjectPrimitiveIdOutput.
	bool object_primitive_id = false;
	/// Whether the vertex stage runs as a mesh shader.
	bool mesh_active = false;
	/// Whether the draw is tessellated.
	bool tess_active = false;
};

/// Whether FinishPixelInputInfo could fill the info.
enum class PixelInputStatus : uint8_t {
	Ok,
	/// A packed UNorm target export the host cannot write, which the GPU thread EXITs on.
	UnsupportedPackedUnorm,
};

/// Why a packed UNorm export was refused, for the GPU thread's EXIT message.
struct PackedUnormRejection {
	uint32_t slot         = 0;
	uint32_t export_index = 0;
	uint32_t mode         = 0;
	uint32_t target_mask  = 0;
	uint32_t shader_mask  = 0;
	uint32_t mapping      = 0;
	bool     blend        = false;
};

/// Adds what the targets, the blend state and the primitive decide to an info that
/// ShaderGetStaticInputInfoPS has filled. That is the object primitive id, the integer and packed UNorm
/// target masks with their round modes, the parameter mode, dual-source blending and the alpha blend
/// source.
/// @param rejection when given, says why a packed UNorm export was refused.
/// @return UnsupportedPackedUnorm, with the info part filled, where the GPU thread would EXIT.
[[nodiscard]] PixelInputStatus FinishPixelInputInfo(const HW::Context& context, const PixelDrawShape& shape,
                                                    ShaderPixelInputInfo& pixel_info,
                                                    PackedUnormRejection* rejection = nullptr);

} // namespace Libs::Graphics

#endif // GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIXEL_INPUTS_H
