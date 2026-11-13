#ifndef GRAPHICS_SHADER_STOREGEN_COMPUTE_COMPILE_H
#define GRAPHICS_SHADER_STOREGEN_COMPUTE_COMPILE_H

#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/storegen/computeTranslation.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// The one compile path of a compute program away from the GPU thread. The shader store generator,
// the Stream worker and the tests that hold them to the runtime all go through it, so their
// translation, compilation, module id and record cannot drift apart. It sits beside
// computeTranslation rather than in it, since it needs the whole recompiler and the setup does not,
// which keeps the setup's own test light.
namespace Libs::Graphics::StoreGen {

/// What names a compute program for the store, which is what its module id and record are made of
/// besides the compiled module. The referenced values must outlive every call that takes the unit.
struct ComputeUnit {
	/// The program's code, as the shader map hashes it.
	std::span<const uint32_t> code;
	/// The hash of the code, as the shader map computes it, which the id and record carry.
	uint64_t code_hash = 0;
	/// How many user data registers the program reads, rsrc2.user_sgpr.
	uint32_t user_data_count = 0;
	/// The static key words from BuildStageStaticKey.
	const std::vector<uint32_t>& static_state;
};

/// What came of building a translated program.
enum class ComputeBuildStatus : uint8_t {
	Built,                 ///< The module compiled, and the record holds it.
	ResourceCountMismatch, ///< The program lists more resources than the specialization.
};

/// One compiled module.
struct ComputeBuild {
	ComputeBuildStatus status = ComputeBuildStatus::Built;
	/// The push data start the compiled module really has, which is the one its id carries.
	uint32_t push_start = 0;
	/// The id the runtime computes for the module, from ShaderModuleId.
	uint64_t id = 0;
	/// The EncodeModuleRecord bytes, empty unless the module was built.
	std::vector<uint8_t> record;
};

/// Whether a translated program lists more buffers, images or samplers than a specialization
/// describes, which ApplyResourceSpecialization refuses with EXIT. A specialization may have more
/// entries than the program, since indirect children are appended to it, so only a program with
/// more of one kind counts.
/// @param buffers  the number of buffers the translated program lists.
/// @param images   the number of images it lists.
/// @param samplers the number of samplers it lists.
[[nodiscard]] bool
ProgramExceedsSpecialization(size_t buffers, size_t images, size_t samplers,
                             const ShaderRecompiler::IR::ResourceSpecialization& specialization);

/// Translates a program the way the runtime does.
/// @param unit  the program.
/// @param setup the options it is translated with, from the same input info as its static key.
/// @return nothing for a program with GPU-selected calls, which the store does not hold.
[[nodiscard]] std::optional<ShaderRecompiler::TranslateResult>
TranslateComputeUnit(const ComputeUnit& unit, TranslationSetup& setup);

/// Compiles a translated program with a specialization, works out the id the runtime computes for
/// the module and encodes its record. The id takes the compiled module's own push data start, so a
/// record is filed under the id a lookup of the same module would compute.
/// @param translated     the program from TranslateComputeUnit, which the compile consumes.
/// @param unit           the program, as it was translated.
/// @param setup          the options it was translated with, or ones made from the same values.
/// @param specialization the resources to compile it for.
[[nodiscard]] ComputeBuild
CompileComputeUnit(ShaderRecompiler::TranslateResult translated, const ComputeUnit& unit,
                   TranslationSetup&                                   setup,
                   const ShaderRecompiler::IR::ResourceSpecialization& specialization);

/// Translates and compiles a program in one go, as TranslateComputeUnit and CompileComputeUnit do.
/// The caller has already found that the program makes no calls, which is what the generator's
/// prediction and the runtime's own lookup check first.
[[nodiscard]] ComputeBuild
BuildComputeUnit(const ComputeUnit& unit, TranslationSetup& setup,
                 const ShaderRecompiler::IR::ResourceSpecialization& specialization);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_COMPUTE_COMPILE_H
