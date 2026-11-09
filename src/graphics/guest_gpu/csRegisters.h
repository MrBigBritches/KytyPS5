#ifndef GRAPHICS_GUEST_GPU_CS_REGISTERS_H
#define GRAPHICS_GUEST_GPU_CS_REGISTERS_H

#include "graphics/guest_gpu/hardwareContext.h"

#include <cstdint>

namespace Libs::Graphics {

/// Applies one compute SH register write to a decoded compute stage.
/// The PM4 handler and the shader store generator both call this, so a program's header registers
/// decode exactly as the registers the game sends at run time.
/// @param cs_regs   the stage to update. It is left untouched when the register is unsupported.
/// @param reg_offset the register offset within the SH range, such as Pm4::COMPUTE_PGM_RSRC1.
/// @param value     the value written to the register.
/// @return false when the register is not a compute register the emulator handles.
[[nodiscard]] bool DecodeCsRegister(HW::CsStageRegisters& cs_regs, uint32_t reg_offset,
                                    uint32_t value);

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_CS_REGISTERS_H
