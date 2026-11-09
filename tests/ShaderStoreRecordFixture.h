// Synthetic compiled modules for the shader store record tests. Every field the record stores gets
// a value no other field shares, so a record with two fields swapped, one dropped or one taken from
// another module differs in its bytes. The same builder feeds the golden capture, which runs it
// through the pipeline cache's encoding from before EncodeModuleRecord existed.
#ifndef TESTS_SHADER_STORE_RECORD_FIXTURE_H
#define TESTS_SHADER_STORE_RECORD_FIXTURE_H

#include "graphics/shader/shaderStoreRecord.h"

#include <cstdint>
#include <vector>

namespace StoreRecordFixture {

using namespace Libs::Graphics;
namespace IR = Libs::Graphics::ShaderRecompiler::IR;

/// The inputs of one module record. Variant 0 is a compute module with an empty tail and variant 1
/// is a vertex module that fills every field of the tail.
struct Module {
	ShaderType                     stage           = ShaderType::Unknown;
	uint64_t                       hash            = 0;
	uint32_t                       user_data_count = 0;
	std::vector<uint32_t>          static_state;
	IR::ResourcePlan               plan;
	uint64_t                       id = 0;
	IR::CompiledShaderInfo         info;
	IR::ResourceSpecialization     specialization;
	std::vector<int32_t>           vertex_formats;
	std::vector<RectListParameter> rect_parameters;
	bool                           mesh_task = false;
	std::vector<uint32_t>          spirv;

	/// Views of this module for EncodeModuleRecord.
	[[nodiscard]] ModuleRecordInputs Inputs() const& {
		return {.stage           = stage,
		        .hash            = hash,
		        .user_data_count = user_data_count,
		        .static_state    = static_state,
		        .plan            = plan,
		        .id              = id,
		        .info            = info,
		        .specialization  = specialization,
		        .vertex_formats  = vertex_formats,
		        .rect_parameters = rect_parameters,
		        .mesh_task       = mesh_task,
		        .spirv           = spirv};
	}
	// The inputs are views, so a temporary module would leave them dangling.
	ModuleRecordInputs Inputs() && = delete;
};

constexpr uint32_t USER_DATA_REGISTERS = 4;
constexpr uint32_t SPIRV_MAGIC         = 0x07230203;

// Four user data registers feeding one descriptor and one buffer that reads it. The register
// numbers depend on the variant, so the two plans differ in their value graph and not only in a
// count.
inline void FillPlan(IR::ResourcePlan& plan, uint32_t variant, ShaderType stage, uint64_t hash) {
	plan.stage           = stage;
	plan.shader_hash     = hash;
	plan.user_data_base  = 8 + variant;
	plan.user_data_count = USER_DATA_REGISTERS + variant;
	IR::DescriptorSource source;
	for (uint32_t word = 0; word < USER_DATA_REGISTERS; word++) {
		auto& inst = plan.value_storage.emplace_back(IR::ValueOpcode::GetUserData);
		inst.SetArg(0, IR::Value(static_cast<IR::ScalarReg>(word + 4 * variant)));
		source.dwords[source.dword_count++] = IR::Value(&inst);
	}
	plan.descriptor_sources.push_back(std::move(source));
	IR::BufferResource buffer;
	buffer.source           = 0;
	buffer.descriptor_index = variant;
	buffer.first_use_pc     = 0x40 + variant;
	buffer.max_byte_extent  = 0x1000 * (variant + 1);
	buffer.read             = true;
	buffer.written          = variant == 1;
	buffer.live             = true;
	plan.info.buffers.push_back(buffer);
	plan.srt_plan_complete = true;
}

// Distinct counts and masks, so that a swap of two bindings fields shows.
inline void FillInfo(IR::CompiledShaderInfo& info, uint32_t variant, ShaderType stage,
                     uint64_t hash) {
	info.stage              = stage;
	info.shader_hash        = hash;
	info.wave_size          = variant == 0 ? 64 : 32;
	info.user_data_base     = 8 + variant;
	info.param_export_mask  = 0x0F0F0000u + variant;
	info.has_address_writes = variant == 1;
	info.info.buffer_u8     = true;
	info.info.float64       = variant == 1;
	uint32_t count          = 1 + variant;
	for (auto& descriptors: info.bindings.descriptor_counts) {
		descriptors = count++;
	}
	info.bindings.descriptor_mask       = 0x1122334455667788ull ^ variant;
	info.bindings.push_data_start_dword = 7 + variant;
	info.bindings.dispatch_thread_dword = 21 + variant;
	info.bindings.memory_offset_dword   = 33 + variant;
}

inline void FillSpecialization(IR::ResourceSpecialization& spec, uint32_t variant) {
	IR::ResourceSpecialization::Buffer buffer;
	buffer.packed_stride              = 16 + variant;
	buffer.descriptor_swizzle         = 0x1234 + variant;
	buffer.zero_stride_oob            = true;
	buffer.indirect_mapping_offset    = 5;
	buffer.indirect_search_iterations = 9;
	spec.buffers.push_back(buffer);
	buffer.scalar_empty  = true;
	buffer.packed_stride = 48 + variant;
	spec.buffers.push_back(buffer);
	IR::ResourceSpecialization::Image image;
	image.mip_count      = 6 + variant;
	image.shader_swizzle = 0x2345 + variant;
	image.cube           = variant == 1;
	spec.images.push_back(image);
	IR::ResourceSpecialization::Sampler sampler;
	sampler.indirect_mapping_offset = 3 + variant;
	spec.samplers.push_back(sampler);
}

/// Builds module `variant`, which is 0 or 1.
inline Module Make(uint32_t variant) {
	Module module;
	module.stage           = variant == 0 ? ShaderType::Compute : ShaderType::Vertex;
	module.hash            = 0xA1B2C3D4E5F60718ull + variant;
	module.user_data_count = USER_DATA_REGISTERS + variant;
	module.static_state    = {module.user_data_count, 0x20 + variant, 32, 4, 0, 9};
	module.id              = 0x0102030405060708ull + variant;
	FillPlan(module.plan, variant, module.stage, module.hash);
	FillInfo(module.info, variant, module.stage, module.hash);
	FillSpecialization(module.specialization, variant);
	module.spirv = {SPIRV_MAGIC, 0x00010600, 0xDEADBEEF + variant, 0, 0x11, 0x22, 0x33 + variant};
	if (variant == 1) {
		module.vertex_formats  = {37, 109, 50};
		module.rect_parameters = {{.input_location = 1, .output_location = 2, .flat = true},
		                          {.input_location = 3, .output_location = 4, .flat = false}};
		module.mesh_task       = true;
	}
	return module;
}

} // namespace StoreRecordFixture

#endif // TESTS_SHADER_STORE_RECORD_FIXTURE_H
