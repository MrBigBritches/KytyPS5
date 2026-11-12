#include "common/alignment.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"

#include <cstring>
#include <vector>

// The parts of a compute shader's static key that need no graphics device, kept apart from
// shader.cpp so that the shader store worker and its tests can link them without the rest of the
// emulator.
namespace Libs::Graphics {

namespace {

constexpr uint32_t FOOTER_SIZE_BYTES   = 0x30;
constexpr uint32_t FOOTER_VERSION      = 1;
constexpr uint32_t FOOTER_VERSION_WORD = 4;
constexpr uint32_t FOOTER_CODE_WORD    = 5;
constexpr uint32_t FOOTER_LITERAL_WORD = 6;
constexpr uint32_t FOOTER_MAGIC_BYTES  = 8;
constexpr uint32_t LITERAL_ALIGNMENT   = 16;
constexpr uint32_t LDS_DWORDS_PER_UNIT = 128;
constexpr uint32_t BYTES_PER_DWORD     = 4;
constexpr uint32_t ASYNC_COMPUTE_SHIFT = 1;
constexpr uint32_t FLOAT_MODE_SHIFT    = 8;

} // namespace

uint32_t ShaderProgramSize(const uint32_t* binary, uint32_t size) {
	if (size < FOOTER_SIZE_BYTES) return size;
	// Magic, then version, code size and literal buffer size at +0x10.
	const auto* footer = binary + (size - FOOTER_SIZE_BYTES) / sizeof(uint32_t);
	const auto  end    = Common::AlignUp(uint64_t {footer[FOOTER_CODE_WORD]}, LITERAL_ALIGNMENT) +
	                 footer[FOOTER_LITERAL_WORD];
	if (std::memcmp(footer, "barefoot", FOOTER_MAGIC_BYTES) != 0 ||
	    footer[FOOTER_VERSION_WORD] != FOOTER_VERSION || footer[FOOTER_CODE_WORD] == 0 ||
	    end % sizeof(uint32_t) != 0 || end > size - FOOTER_SIZE_BYTES) {
		return size;
	}
	return static_cast<uint32_t>(end);
}

void ShaderGetStaticInputInfoCS(const HW::CsStageRegisters& cs_regs, uint32_t scratch_size_dwords,
                                ShaderComputeInputInfo& info) {
	info.threads_num[0]      = cs_regs.num_thread_x;
	info.threads_num[1]      = cs_regs.num_thread_y;
	info.threads_num[2]      = cs_regs.num_thread_z;
	info.lds_size_dwords     = static_cast<uint32_t>(cs_regs.lds_size) * LDS_DWORDS_PER_UNIT;
	info.scratch_size_dwords = scratch_size_dwords;
	info.float_mode          = cs_regs.float_mode;
	info.group_id[0]         = cs_regs.tgid_x_en != 0;
	info.group_id[1]         = cs_regs.tgid_y_en != 0;
	info.group_id[2]         = cs_regs.tgid_z_en != 0;
	info.wave_size           = cs_regs.wave_size;
	info.thread_ids_num      = cs_regs.tidig_comp_cnt + 1;
	info.tg_size_en          = cs_regs.tg_size_en != 0;

	info.workgroup_register = cs_regs.user_sgpr;
}

bool ComputeLdsNeedsStorage(uint32_t lds_dwords, uint32_t max_shared_bytes) {
	return lds_dwords * BYTES_PER_DWORD > max_shared_bytes;
}

void BuildStageStaticKey(const ShaderComputeInputInfo& info, std::vector<uint32_t>& key) {
	key.clear();
	key.push_back(info.workgroup_register);
	key.push_back(info.wave_size | (static_cast<uint32_t>(info.float_mode) << FLOAT_MODE_SHIFT));
	key.push_back(info.host_subgroup_size);
	key.push_back(info.thread_ids_num);
	key.push_back(info.lds_size_dwords);
	key.push_back(info.scratch_size_dwords);
	key.push_back(static_cast<uint32_t>(info.dispatch_thread_dimensions) |
	              (static_cast<uint32_t>(info.async_compute) << ASYNC_COMPUTE_SHIFT));
	for (int i = 0; i < 3; i++) {
		key.push_back(info.threads_num[i]);
		key.push_back(static_cast<uint32_t>(info.group_id[i]));
	}
	key.push_back(static_cast<uint32_t>(info.tg_size_en));
}

} // namespace Libs::Graphics
