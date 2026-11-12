#include "graphics/shader/storegen/computeTranslation.h"

#include "graphics/shader/shaderCompiler.h"

#include <algorithm>

namespace Libs::Graphics::StoreGen {

TranslationSetup::TranslationSetup(const ShaderComputeInputInfo& info, uint32_t user_data_count,
                                   uint64_t code_hash, const HostValues& host)
    : m_info(info), m_user_data(user_data_count, 0u) {
	m_options.wave_size              = m_info.wave_size;
	m_options.stage                  = ShaderType::Compute;
	m_options.input_info.compute     = &m_info;
	m_options.shader_hash            = code_hash;
	m_options.dump_ir                = false;
	m_options.structured_jump_tables = host.structured_jump_tables;
	m_options.user_data              = m_user_data;
	m_options.profile                = &host.profile;
}

std::optional<ShaderComputeInputInfo> MakePrescanInputInfo(const PrescanDispatchInputs& dispatch,
                                                           uint32_t scratch_size_dwords,
                                                           const ComputeKeyDevice& device) {
	// The command processor turns this into a direct dispatch with counts it reads from guest
	// memory, so an info built here would not be the runtime's.
	if (dispatch.indirect && dispatch.thread_dimensions) {
		return std::nullopt;
	}
	ShaderComputeInputInfo info {};
	info.async_compute              = dispatch.async_compute;
	info.dispatch_thread_dimensions = dispatch.thread_dimensions;
	info.indirect_args_address      = dispatch.indirect_args_address;
	std::ranges::copy(dispatch.workgroup_counts, info.workgroup_counts);
	info.host_subgroup_size = device.host_subgroup_size;
	ShaderGetStaticInputInfoCS(dispatch.cs_regs, scratch_size_dwords, info);
	// The module id does not carry this flag, so a wrong value here would give a module with the
	// runtime's id and other SPIR-V. The shared helper keeps it the runtime's decision.
	info.lds_storage =
	    ComputeLdsNeedsStorage(info.lds_size_dwords, device.max_compute_shared_memory_bytes);
	return info;
}

} // namespace Libs::Graphics::StoreGen
