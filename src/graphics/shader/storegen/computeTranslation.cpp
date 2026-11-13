#include "graphics/shader/storegen/computeTranslation.h"

#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>

namespace Libs::Graphics::StoreGen {

namespace {

namespace IR = ShaderRecompiler::IR;

// Whether every flag byte of a store header is 0 or 1. The header is read once more as plain bytes
// in the order the fields are written, which is the build stamp, the jump table flag, the profile
// (three flags, the device clock period and six flags) and the shared memory limit. A flag read
// into a bool from any other byte value would be undefined behaviour. If the profile gains a
// field this read comes up short and the header is refused, which fails every test at once.
bool HeaderFlagsAreBooleans(std::span<const uint8_t> bytes) {
	IR::Reader reader {bytes};
	uint64_t   stamp       = 0;
	uint8_t    jump_tables = 0;
	uint8_t    profile_flags[9] {};
	uint32_t   clock_period = 0;
	uint32_t   shared       = 0;
	IR::IoAll(reader, stamp, jump_tables, profile_flags[0], profile_flags[1], profile_flags[2],
	          clock_period, profile_flags[3], profile_flags[4], profile_flags[5], profile_flags[6],
	          profile_flags[7], profile_flags[8], shared);
	bool booleans = jump_tables <= 1;
	for (const auto flag: profile_flags) {
		booleans = booleans && flag <= 1;
	}
	return reader.ok && reader.bytes.empty() && booleans;
}

} // namespace

std::optional<HostValues> DecodeHostHeader(std::span<const uint8_t> bytes) {
	if (!HeaderFlagsAreBooleans(bytes)) {
		return std::nullopt;
	}
	IR::Reader reader {bytes};
	uint64_t   stamp = 0;
	HostValues host;
	IR::IoAll(reader, stamp, host.structured_jump_tables, host.profile, host.max_shared_bytes);
	if (!reader.ok || !reader.bytes.empty()) {
		return std::nullopt;
	}
	return host;
}

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
