#ifndef GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GRAPHICS_COMPILE_OPTIONS_H
#define GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GRAPHICS_COMPILE_OPTIONS_H

#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <cstdint>

// The per-stage compile options of a graphics shader. GetProgram builds them on the GPU thread and the
// graphics prescan builds them on its worker, and the module id leaves most of them out, so both take
// them from here. Two copies that drifted would build a different module under the same key.
namespace Libs::Graphics {

/// The user SGPR a vertex, local or evaluation program's user data starts at. These run as RDNA 2
/// merged hardware stages, which reserve s0 to s7 for system values, and their shader params hold only
/// the user registers. Mesh and control programs get the reserved slots at the front of their params
/// instead, so their user data starts at 0.
inline constexpr uint32_t VERTEX_USER_DATA_BASE = 8;

/// The wave size of every tessellation control program. PrepareTessellationPrograms refuses a wave32
/// hull stage, so this never follows the draw's wave size.
inline constexpr uint32_t TESS_CONTROL_WAVE_SIZE = 64;

/// What a graphics stage's options take from the host rather than from the draw. The GPU thread reads
/// these from the config at each lookup, and the prescan reads them once when its sink is made.
struct GraphicsCompileHost {
	bool                                    structured_jump_tables = false;
	/// GetProgram leaves this at the default, since CompileModule puts the device's own profile in.
	const ShaderRecompiler::Spirv::Profile* profile                = &ShaderRecompiler::Spirv::DefaultProfile;
	bool                                    emit_debug_info        = false;
};

/// The options of a vertex-info stage, which are its stage, input info, wave size and user data base,
/// plus the host fields. Mesh and tessellation control programs start their user data at 0 and take
/// their own wave size, and every other stage starts it at VERTEX_USER_DATA_BASE.
/// The caller sets user_data, shader_hash and dump_ir, and back_code for a mesh stage.
/// @param info the stage's input info, which must outlive the options.
/// @param host the host fields.
[[nodiscard]] ShaderRecompiler::CompileOptions GraphicsStageCompileOptions(const ShaderVertexInputInfo& info,
                                                                           const GraphicsCompileHost&   host);

/// The options of a pixel program, with its user data at 0 and the draw's wave size, plus the host
/// fields. The caller sets user_data, shader_hash and dump_ir.
/// @param info the pixel input info, which must outlive the options.
/// @param host the host fields.
[[nodiscard]] ShaderRecompiler::CompileOptions GraphicsStageCompileOptions(const ShaderPixelInputInfo& info,
                                                                           const GraphicsCompileHost&  host);

} // namespace Libs::Graphics

#endif // GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GRAPHICS_COMPILE_OPTIONS_H
