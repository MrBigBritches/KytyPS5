#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_

#include "graphics/shader/shader.h"

#include <array>
#include <span>
#include <vector>

namespace Libs::Graphics {

namespace HW {
class Context;
class UserConfig;
struct CsStageRegisters;
} // namespace HW

struct ShaderParams {
	std::span<const uint32_t> code;
	std::array<uint32_t, 40>  user_data {}; // 32 user SGPRs plus the merged-stage s0:s7 prefix.
	uint32_t                  user_data_count = 0;
	uint64_t                  hash = 0;
	std::span<const uint32_t> back_code;

	[[nodiscard]] uint64_t Base() const {
		return reinterpret_cast<uint64_t>(code.data());
	}
};

void BuildStageStaticKey(const ShaderVertexInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageStaticKey(const ShaderPixelInputInfo& input_info, std::vector<uint32_t>& key);
void BuildStageStaticKey(const ShaderComputeInputInfo& input_info, std::vector<uint32_t>& key);

/// Fills the static part of a compute shader's input info from its stage registers.
/// The runtime and the shader store generator both call this, so the two cannot disagree on how
/// registers map to the key. The device fields host_subgroup_size and lds_storage are left alone.
void ShaderGetStaticInputInfoCS(const HW::CsStageRegisters& cs_regs, uint32_t scratch_size_dwords,
                                ShaderComputeInputInfo& info);

/// Whether a compute program's LDS is larger than the device's shared memory, so the recompiler
/// backs it with a storage buffer. GetComputeProgram, the shader store generator and the pre-scan
/// all decide it here, because the module id does not carry the answer.
/// @param lds_dwords       the program's LDS size in dwords, ShaderComputeInputInfo::lds_size_dwords.
/// @param max_shared_bytes the device's maxComputeSharedMemorySize limit.
[[nodiscard]] bool ComputeLdsNeedsStorage(uint32_t lds_dwords, uint32_t max_shared_bytes);

ShaderParams PrepareProgram(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            const HW::UserConfig& user_config, ShaderVertexInputInfo& input_info);
// Each stage's tess holds the VGT fields; the caller adds the layout reflected from the programs.
std::array<ShaderParams, 3>
PrepareTessellationPrograms(const HW::VertexShaderInfo& regs, const HW::Context& context,
                            std::array<ShaderVertexInputInfo, 3>& input_info);
ShaderParams PrepareProgram(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    ShaderPixelInputInfo&                               input_info);
ShaderParams PrepareProgram(const HW::ComputeShaderInfo& regs, ShaderComputeInputInfo& input_info);

/// Fills the part of a pixel shader's input info that its registers decide. The GPU thread calls it
/// through PrepareProgram and the graphics prescan calls it on its own shadow, so the two agree on how
/// registers map to the key. It EXITs where ShaderPsInputsSupported is false, so a caller that cannot
/// afford that checks first.
/// @param regs                  the pixel stage registers.
/// @param sh                    the context's shader registers.
/// @param target_export_mapping the export mapping of each color target, by slot.
/// @param data                  the mapped shader's scratch size and input semantics. The semantics
///                              are read through a guest pointer, so a caller off the GPU thread passes
///                              none and adds ShaderCustomInterpolationMask itself.
/// @param ps_info               the info to fill.
void ShaderGetStaticInputInfoPS(
    const HW::PixelShaderInfo& regs, const HW::ShaderRegisters& sh,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping,
    const ShaderMappedData& data, ShaderPixelInputInfo& ps_info);

/// Fills the static input info of a plain vertex shader from copies of its attribute and buffer
/// tables, as PrepareProgram does from the guest pointers in its user registers, for the graphics
/// prescan. Where PrepareProgram would EXIT on the tables it returns false instead.
/// @param attrib_reg the user register of the attribute table pointer, or -1 for none.
/// @param buffer_reg the user register of the buffer table pointer.
/// @param semantics  the input semantics, copied from the binary.
/// @param attrib     the attribute table, at least one word past the largest semantic.
/// @param buffer     the buffer table, four words per buffer the attributes name.
[[nodiscard]] bool ShaderTryStaticVertexInputInfo(const HW::ShaderRegisters& sh, uint32_t scratch_size_dwords,
                                                  int attrib_reg, int buffer_reg,
                                                  std::span<const ShaderSemantic> semantics,
                                                  const uint32_t* attrib, const uint32_t* buffer,
                                                  ShaderVertexInputInfo& info);

/// Whether ShaderGetStaticInputInfoPS can run on these registers without EXIT, which it does on input
/// bits it does not support, on SPI_PS_INPUT_ENA differing from SPI_PS_INPUT_ADDR, and on more
/// interpolators than it holds.
[[nodiscard]] bool ShaderPsInputsSupported(const HW::ShaderRegisters& sh);

/// The interpolators a pixel shader interpolates itself, one bit per input. Each one whose semantic is
/// custom and not 16-bit sets its bit, for the first input_num inputs and at most 32.
/// @param semantics the shader's input semantics, as copied from the binary.
/// @param input_num how many interpolators SPI_PS_IN_CONTROL enables.
[[nodiscard]] uint32_t ShaderCustomInterpolationMask(std::span<const ShaderSemantic> semantics,
                                                     uint32_t                        input_num);

/// Whether the vertex stage runs as a mesh shader, which is when ES and GS are merged or a native NGG
/// GS uses a workgroup. PrepareProgram decides it here, and so does the prescan, which needs it for the
/// pixel shader's push data start.
/// @param shader_stages VGT_SHADER_STAGES_EN.
/// @param type          the binary type of the program at the ES address.
/// @param gs_lds_size   SPI_SHADER_PGM_RSRC2_GS LDS_SIZE.
[[nodiscard]] bool VertexStageIsMesh(uint32_t shader_stages, Prospero::ShaderBinaryType type,
                                     uint32_t gs_lds_size);

} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_SHADERCOMPILER_H_ */
