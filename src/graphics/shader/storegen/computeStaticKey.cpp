#include "graphics/shader/storegen/computeStaticKey.h"

#include "graphics/guest_gpu/csRegisters.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/storegen/shaderHeaderLayout.h"

#include <cstddef>

namespace Libs::Graphics::StoreGen {

namespace {

namespace Layout = HeaderLayout;

// The layout constants are the offsets of the struct the runtime reads the header through.
static_assert(offsetof(Shader, file_header) == Layout::OFFSET_MAGIC);
static_assert(offsetof(Shader, version) == Layout::OFFSET_VERSION);
static_assert(offsetof(Shader, sh_registers) == Layout::OFFSET_SH_REGISTERS);
static_assert(offsetof(Shader, shader_size) == Layout::OFFSET_SHADER_SIZE);
static_assert(offsetof(Shader, scratch_size_dw_per_thread) == Layout::OFFSET_SCRATCH_DWORDS);
static_assert(offsetof(Shader, type) == Layout::OFFSET_TYPE);
static_assert(offsetof(Shader, num_sh_registers) == Layout::OFFSET_NUM_SH_REGISTERS);

constexpr uint32_t BYTES_PER_DWORD = 4;

// Applies the header's SH register table to a fresh compute stage.
// Returns false when the table is missing or out of range, or lacks either shader resource
// register.
bool DecodeHeaderRegisters(std::span<const uint8_t> header, HW::CsStageRegisters& cs_regs) {
	uint64_t sh_relative = 0;
	uint8_t  count       = 0;
	if (!Layout::Read(header, Layout::OFFSET_SH_REGISTERS, sh_relative) ||
	    !Layout::Read(header, Layout::OFFSET_NUM_SH_REGISTERS, count)) {
		return false;
	}
	// The runtime leaves a null table pointer null and would read through it, so a table that is
	// listed but not placed is not a usable header.
	if (sh_relative == 0 && count != 0) {
		return false;
	}
	// The table pointer is stored relative to its own field. Unsigned wrap-around lets a table that
	// precedes the field work, and a wild value fails the range check below.
	const uint64_t table     = Layout::OFFSET_SH_REGISTERS + sh_relative;
	bool           has_rsrc1 = false;
	bool           has_rsrc2 = false;
	for (uint64_t index = 0; index < count; index++) {
		uint32_t       reg_offset = 0;
		uint32_t       value      = 0;
		const uint64_t at         = table + index * Layout::SH_REGISTER_SIZE;
		if (!Layout::Read(header, at, reg_offset) ||
		    !Layout::Read(header, at + sizeof(uint32_t), value)) {
			return false;
		}
		has_rsrc1 = has_rsrc1 || reg_offset == Pm4::COMPUTE_PGM_RSRC1;
		has_rsrc2 = has_rsrc2 || reg_offset == Pm4::COMPUTE_PGM_RSRC2;
		// A register the compute decoder does not know does not affect the key, so it is skipped
		// where the PM4 handler would stop the emulator.
		(void)DecodeCsRegister(cs_regs, reg_offset, value);
	}
	return has_rsrc1 && has_rsrc2;
}

} // namespace

std::optional<ComputeStaticKey> BuildComputeStaticKey(std::span<const uint8_t>     header,
                                                      const ComputeKeyDevice&      device,
                                                      const ComputeDispatchInputs& dispatch) {
	uint16_t scratch_dwords = 0;
	if (!Layout::Read(header, Layout::OFFSET_SCRATCH_DWORDS, scratch_dwords)) {
		return std::nullopt;
	}
	HW::CsStageRegisters cs_regs;
	cs_regs.wave_size = static_cast<uint8_t>(dispatch.wave_size);
	if (!DecodeHeaderRegisters(header, cs_regs)) {
		return std::nullopt;
	}
	ComputeStaticKey key;
	ShaderGetStaticInputInfoCS(cs_regs, scratch_dwords, key.input_info);
	key.input_info.host_subgroup_size         = device.host_subgroup_size;
	key.input_info.dispatch_thread_dimensions = dispatch.dispatch_thread_dimensions;
	key.input_info.async_compute              = dispatch.async_compute;
	// The same comparison GetComputeProgram makes.
	key.input_info.lds_storage =
	    key.input_info.lds_size_dwords * BYTES_PER_DWORD > device.max_compute_shared_memory_bytes;
	key.user_data_count = cs_regs.user_sgpr;
	BuildStageStaticKey(key.input_info, key.state);
	return key;
}

} // namespace Libs::Graphics::StoreGen
