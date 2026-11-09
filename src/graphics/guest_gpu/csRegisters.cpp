#include "graphics/guest_gpu/csRegisters.h"

#include "graphics/guest_gpu/pm4.h"

namespace Libs::Graphics {

namespace {

// Extracts one bit-field of a register value.
constexpr uint32_t Field(uint32_t value, uint32_t shift, uint32_t mask) {
	return (value >> shift) & mask;
}

// COMPUTE_PGM_RSRC1 carries the VGPR count, float mode and the shader mode bits.
void DecodeRsrc1(HW::CsStageRegisters& cs_regs, uint32_t value) {
	cs_regs.vgprs =
	    Field(value, Pm4::COMPUTE_PGM_RSRC1_VGPRS_SHIFT, Pm4::COMPUTE_PGM_RSRC1_VGPRS_MASK);
	cs_regs.priority =
	    Field(value, Pm4::COMPUTE_PGM_RSRC1_PRIORITY_SHIFT, Pm4::COMPUTE_PGM_RSRC1_PRIORITY_MASK);
	cs_regs.float_mode                = Field(value, Pm4::COMPUTE_PGM_RSRC1_FLOAT_MODE_SHIFT,
	                                          Pm4::COMPUTE_PGM_RSRC1_FLOAT_MODE_MASK);
	cs_regs.dx10_clamp                = Field(value, Pm4::COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT,
	                                          Pm4::COMPUTE_PGM_RSRC1_DX10_CLAMP_MASK) != 0u;
	cs_regs.debug_mode                = Field(value, Pm4::COMPUTE_PGM_RSRC1_DEBUG_MODE_SHIFT,
	                                          Pm4::COMPUTE_PGM_RSRC1_DEBUG_MODE_MASK) != 0u;
	cs_regs.ieee_mode                 = Field(value, Pm4::COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT,
	                                          Pm4::COMPUTE_PGM_RSRC1_IEEE_MODE_MASK) != 0u;
	cs_regs.fp16_overflow             = Field(value, Pm4::COMPUTE_PGM_RSRC1_FP16_OVFL_SHIFT,
	                                          Pm4::COMPUTE_PGM_RSRC1_FP16_OVFL_MASK) != 0u;
	cs_regs.threadgroup_configuration = Field(value, Pm4::COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT,
	                                          Pm4::COMPUTE_PGM_RSRC1_WGP_MODE_MASK) != 0u;
	cs_regs.require_forward_progress  = Field(value, Pm4::COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT,
	                                          Pm4::COMPUTE_PGM_RSRC1_FWD_PROGRESS_MASK) != 0u;
}

// COMPUTE_PGM_RSRC2 carries the user SGPR count, the system value enables and the LDS size.
void DecodeRsrc2(HW::CsStageRegisters& cs_regs, uint32_t value) {
	cs_regs.scratch_en = Field(value, Pm4::COMPUTE_PGM_RSRC2_SCRATCH_EN_SHIFT,
	                           Pm4::COMPUTE_PGM_RSRC2_SCRATCH_EN_MASK) != 0u;
	cs_regs.user_sgpr =
	    Field(value, Pm4::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT, Pm4::COMPUTE_PGM_RSRC2_USER_SGPR_MASK);
	cs_regs.tgid_x_en      = Field(value, Pm4::COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT,
	                               Pm4::COMPUTE_PGM_RSRC2_TGID_X_EN_MASK) != 0u;
	cs_regs.tgid_y_en      = Field(value, Pm4::COMPUTE_PGM_RSRC2_TGID_Y_EN_SHIFT,
	                               Pm4::COMPUTE_PGM_RSRC2_TGID_Y_EN_MASK) != 0u;
	cs_regs.tgid_z_en      = Field(value, Pm4::COMPUTE_PGM_RSRC2_TGID_Z_EN_SHIFT,
	                               Pm4::COMPUTE_PGM_RSRC2_TGID_Z_EN_MASK) != 0u;
	cs_regs.tg_size_en     = Field(value, Pm4::COMPUTE_PGM_RSRC2_TG_SIZE_EN_SHIFT,
	                               Pm4::COMPUTE_PGM_RSRC2_TG_SIZE_EN_MASK) != 0u;
	cs_regs.tidig_comp_cnt = Field(value, Pm4::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT,
	                               Pm4::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK);
	cs_regs.lds_size =
	    Field(value, Pm4::COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT, Pm4::COMPUTE_PGM_RSRC2_LDS_SIZE_MASK);
}

} // namespace

bool DecodeCsRegister(HW::CsStageRegisters& cs_regs, uint32_t reg_offset, uint32_t value) {
	switch (reg_offset) {
		case Pm4::COMPUTE_PGM_LO:
			cs_regs.data_addr &= 0xFFFFFF00000000FFull;
			cs_regs.data_addr |= static_cast<uint64_t>(value) << 8u;
			return true;
		case Pm4::COMPUTE_PGM_HI:
			cs_regs.data_addr &= 0xFFFF00FFFFFFFFFFull;
			cs_regs.data_addr |= (static_cast<uint64_t>(value) & 0xffu) << 40u;
			return true;
		case Pm4::COMPUTE_PGM_RSRC1: DecodeRsrc1(cs_regs, value); return true;
		case Pm4::COMPUTE_PGM_RSRC2: DecodeRsrc2(cs_regs, value); return true;
		case Pm4::COMPUTE_PGM_RSRC3:
			cs_regs.shared_vgprs = Field(value, Pm4::COMPUTE_PGM_RSRC3_SHARED_VGPRS_SHIFT,
			                             Pm4::COMPUTE_PGM_RSRC3_SHARED_VGPRS_MASK);
			return true;
		case Pm4::COMPUTE_NUM_THREAD_X: cs_regs.num_thread_x = value; return true;
		case Pm4::COMPUTE_NUM_THREAD_Y: cs_regs.num_thread_y = value; return true;
		case Pm4::COMPUTE_NUM_THREAD_Z: cs_regs.num_thread_z = value; return true;
		// Accepted and ignored, as no host state depends on them.
		case Pm4::COMPUTE_START_X:
		case Pm4::COMPUTE_START_Y:
		case Pm4::COMPUTE_START_Z:
		case Pm4::COMPUTE_RESOURCE_LIMITS:
		case Pm4::COMPUTE_TMPRING_SIZE:
		case Pm4::COMPUTE_PACE_ID: return true;
		default: return false;
	}
}

} // namespace Libs::Graphics
