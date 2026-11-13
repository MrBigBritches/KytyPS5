#include "graphics/shader/storegen/computeCompile.h"

#include "graphics/shader/moduleId.h"
#include "graphics/shader/shaderStoreRecord.h"

#include <utility>

namespace Libs::Graphics::StoreGen {

namespace IR = ShaderRecompiler::IR;

namespace {

// Whether a prepared source has a GPU-selected call, which the store does not hold.
bool SourceHasCalls(const ShaderRecompiler::ShaderSource& source) {
	return source.unsupported_call_pc != UINT32_MAX || !source.calls.empty();
}

} // namespace

bool ProgramExceedsSpecialization(size_t buffers, size_t images, size_t samplers,
                                  const IR::ResourceSpecialization& specialization) {
	return buffers > specialization.buffers.size() || images > specialization.images.size() ||
	       samplers > specialization.samplers.size();
}

std::optional<ShaderRecompiler::TranslateResult> TranslateComputeUnit(const ComputeUnit& unit,
                                                                      TranslationSetup&  setup) {
	const auto source = ShaderRecompiler::PrepareShaderSource(unit.code, setup.Options());
	if (SourceHasCalls(source)) {
		return std::nullopt;
	}
	return ShaderRecompiler::TranslateProgram(source.decoded, setup.Options());
}

ComputeBuild CompileComputeUnit(ShaderRecompiler::TranslateResult translated, const ComputeUnit& unit,
                                TranslationSetup&                 setup,
                                const IR::ResourceSpecialization& specialization) {
	// The plan is taken before the compile, which consumes the program it is extracted from.
	const auto  plan   = IR::ExtractResourcePlan(translated.program);
	const auto& listed = translated.program.info;
	// The check ApplyResourceSpecialization makes with EXIT, made here so a bad specialization is
	// reported instead of ending the process.
	if (ProgramExceedsSpecialization(listed.buffers.size(), listed.images.size(),
	                                 listed.samplers.size(), specialization)) {
		return {ComputeBuildStatus::ResourceCountMismatch, 0, 0, {}};
	}
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), setup.Options(),
	                                               specialization, 0);
	const auto   info = std::move(result.program).TakeCompiledInfo();
	ComputeBuild build;
	build.push_start = info.bindings.push_data_start_dword;
	build.id = ShaderModuleId(ShaderType::Compute, unit.code_hash, unit.user_data_count,
	                          unit.static_state, build.push_start, specialization, {});
	build.record = EncodeModuleRecord({.stage           = ShaderType::Compute,
	                                   .hash            = unit.code_hash,
	                                   .user_data_count = unit.user_data_count,
	                                   .static_state    = unit.static_state,
	                                   .plan            = plan,
	                                   .id              = build.id,
	                                   .info            = info,
	                                   .specialization  = specialization,
	                                   .vertex_formats  = {},
	                                   .rect_parameters = {},
	                                   .mesh_task       = false,
	                                   .spirv           = result.spirv});
	return build;
}

ComputeBuild BuildComputeUnit(const ComputeUnit& unit, TranslationSetup& setup,
                              const IR::ResourceSpecialization& specialization) {
	// No call check here, since every caller has made it already, and the generator's records must
	// stay what they were before this path was shared.
	const auto source = ShaderRecompiler::PrepareShaderSource(unit.code, setup.Options());
	auto translated   = ShaderRecompiler::TranslateProgram(source.decoded, setup.Options());
	return CompileComputeUnit(std::move(translated), unit, setup, specialization);
}

} // namespace Libs::Graphics::StoreGen
