#include "graphics/shader/moduleId.h"

#include "graphics/shader/recompiler/ir/Serialize.h"

#include <xxhash.h>

namespace Libs::Graphics {

uint64_t ShaderModuleId(ShaderType stage, uint64_t hash, uint32_t user_data_count,
                        const std::vector<uint32_t>& static_state, uint32_t push_data_start_dword,
                        const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                        std::span<const uint32_t>                           function_code) {
	ShaderRecompiler::IR::Writer inputs;
	ShaderRecompiler::IR::IoAll(inputs, stage, hash, user_data_count, static_state,
	                            push_data_start_dword, specialization);
	inputs.Raw(function_code.data(), function_code.size_bytes()); // Last, so it needs no size.
	return XXH3_64bits(inputs.bytes.data(), inputs.bytes.size());
}

// Serialized as ShaderModuleId serializes the same fields, so each field is length-prefixed and two
// keys cannot collide by moving bytes from one field to the next.
uint64_t ShaderModuleSkipKey(ShaderType stage, uint64_t hash, uint32_t user_data_count,
                             const std::vector<uint32_t>&                        static_state,
                             const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	ShaderRecompiler::IR::Writer inputs;
	ShaderRecompiler::IR::IoAll(inputs, stage, hash, user_data_count, static_state, specialization);
	return XXH3_64bits(inputs.bytes.data(), inputs.bytes.size());
}

} // namespace Libs::Graphics
