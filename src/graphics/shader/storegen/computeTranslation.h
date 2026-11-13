#ifndef GRAPHICS_SHADER_STOREGEN_COMPUTE_TRANSLATION_H
#define GRAPHICS_SHADER_STOREGEN_COMPUTE_TRANSLATION_H

#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/prescanInputs.h"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// How a compute program is translated away from the GPU thread, so that the module it gives has the
// id the runtime would compute for the same dispatch. The shader store generator and the pre-scan
// both use it.
namespace Libs::Graphics::StoreGen {

/// The values of the store header that translation depends on. PipelineCache writes them after the
/// build stamp, and the generator reads them back with DecodeHostHeader.
struct HostValues {
	bool                             structured_jump_tables = false;
	ShaderRecompiler::Spirv::Profile profile;
	uint32_t                         max_shared_bytes = 0;
};

/// Reads the store header the way PipelineCache writes it, which is the build stamp, then the host
/// values. The build stamp does not affect translation and is skipped.
/// @return nothing when the bytes are not exactly one header or a flag byte is not 0 or 1.
[[nodiscard]] std::optional<HostValues> DecodeHostHeader(std::span<const uint8_t> bytes);

/// Writes a store header, the build stamp and then the host values, which DecodeHostHeader reads
/// back. PipelineCache writes its store's header with it, and the pre-scan writes the header its
/// helper gets with it, so the two can never be laid out differently.
/// @param host        the host values translation depends on.
/// @param build_stamp the build's stamp. The helper skips it, so the pre-scan passes 0.
[[nodiscard]] std::vector<uint8_t> EncodeHostHeader(const HostValues& host, uint64_t build_stamp);

/// The translation options of one compute program, set as the runtime sets them for a compute
/// program. The user data is zeroed, since translation reads only how many registers there are. The
/// options point into this object and into `host`, so this object stays where it is and `host`
/// outlives it.
class TranslationSetup {
public:
	/// Sets the options for one program.
	/// @param info            the program's compute input info, as the runtime would build it.
	/// @param user_data_count how many user data registers the program reads, rsrc2.user_sgpr.
	/// @param code_hash       the hash of the program's code, as the shader map computes it.
	/// @param host            the host values the store header records.
	TranslationSetup(const ShaderComputeInputInfo& info, uint32_t user_data_count,
	                 uint64_t code_hash, const HostValues& host);
	TranslationSetup(const TranslationSetup&)            = delete;
	TranslationSetup& operator=(const TranslationSetup&) = delete;

	/// The options to translate and compile the program with.
	ShaderRecompiler::CompileOptions& Options() { return m_options; }

private:
	ShaderComputeInputInfo           m_info;
	std::vector<uint32_t>            m_user_data;
	ShaderRecompiler::CompileOptions m_options;
};

/// Builds the compute input info the runtime would build for a scanned dispatch, without a device.
/// The fields come from the same places as on the GPU thread. RenderExecutor's dispatch sets the
/// queue, launch mode, group counts and indirect address, and GetComputeProgram sets the host
/// subgroup size, decodes the registers and decides whether LDS needs a storage buffer.
///
/// An indirect dispatch in thread dimensions cannot be prescanned. The command processor reads its
/// arguments from guest memory and dispatches it as a direct one, so the runtime's info has the
/// group counts read from memory and no indirect address, and neither is known at scan time.
/// @param dispatch            what the scan saw of the dispatch.
/// @param scratch_size_dwords the scratch size the shader map holds for the program. The runtime
///                            reads it from there and not from a register, so the caller does too.
/// @param device              the host subgroup size and the device's shared memory limit.
/// @return nothing for an indirect dispatch in thread dimensions, otherwise the info.
[[nodiscard]] std::optional<ShaderComputeInputInfo>
MakePrescanInputInfo(const PrescanDispatchInputs& dispatch, uint32_t scratch_size_dwords,
                     const ComputeKeyDevice& device);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_COMPUTE_TRANSLATION_H
