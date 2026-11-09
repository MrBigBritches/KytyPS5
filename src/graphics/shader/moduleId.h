#ifndef GRAPHICS_SHADER_MODULE_ID_H
#define GRAPHICS_SHADER_MODULE_ID_H

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics {

/// The id of a shader module compiled from these inputs. The same inputs give the same id in every
/// run, and the shader store keys its module records by it, so it must never change for inputs an
/// existing store already holds. The pipeline cache and the store generator both take their ids
/// from here, so a generated record is found by the id the runtime computes.
/// @param stage            the shader stage.
/// @param hash             the code hash of the program.
/// @param user_data_count  the program's user data dword count.
/// @param static_state     the static key words from BuildStageStaticKey.
/// @param push_data_start_dword the module's push data start, PushData::NoStart when it has none.
/// @param specialization   the resource specialization the module was compiled with.
/// @param function_code    the GPU-selected function code, empty for a program with no calls.
[[nodiscard]] uint64_t
ShaderModuleId(ShaderType stage, uint64_t hash, uint32_t user_data_count,
               const std::vector<uint32_t>& static_state, uint32_t push_data_start_dword,
               const ShaderRecompiler::IR::ResourceSpecialization& specialization,
               std::span<const uint32_t>                           function_code);

} // namespace Libs::Graphics

#endif // GRAPHICS_SHADER_MODULE_ID_H
