#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_

#include "graphics/shader/recompiler/ir/ShaderIR.h"

namespace Libs::Graphics::ShaderRecompiler::IR {

// Collects the shader interface, resource usage and instruction features after resource tracking.
void CollectShaderInfo(Program& program, ShaderStageInputInfo input_info);

// The bit of a specialized buffer's packed stride that carries the descriptor's ADD_TID.
inline constexpr uint32_t BUFFER_ADD_TID_STRIDE_BIT = 1u << 20u;

// Whether a buffer with this packed stride asks for ADD_TID in a stage that does not support it, which
// only compute does. CollectShaderInfo EXITs on it, and the graphics prescan rejects it first.
[[nodiscard]] constexpr bool BufferAddTidUnsupported(ShaderType stage, uint32_t packed_stride) {
	return (packed_stride & BUFFER_ADD_TID_STRIDE_BIT) != 0u && stage != ShaderType::Compute;
}

// EXITs as CollectShaderInfo does when BufferAddTidUnsupported holds. CollectShaderInfo calls it, and a
// test calls it to show that a guard in front of it was missing.
void RequireBufferAddTidSupported(ShaderType stage, uint32_t packed_stride);

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERINFOCOLLECTION_H_ */
