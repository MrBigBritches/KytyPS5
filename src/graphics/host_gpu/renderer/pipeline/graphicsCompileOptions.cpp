#include "graphics/host_gpu/renderer/pipeline/graphicsCompileOptions.h"

namespace Libs::Graphics {

namespace {

// The fields every graphics stage takes the same way, from its wave size and the host.
ShaderRecompiler::CompileOptions HostOptions(uint32_t wave_size, const GraphicsCompileHost& host) {
	ShaderRecompiler::CompileOptions options {.wave_size = wave_size};
	options.emit_debug_info        = host.emit_debug_info;
	options.structured_jump_tables = host.structured_jump_tables;
	options.profile                = host.profile;
	return options;
}

} // namespace

ShaderRecompiler::CompileOptions GraphicsStageCompileOptions(const ShaderVertexInputInfo& info,
                                                             const GraphicsCompileHost&   host) {
	auto options              = HostOptions(info.wave_size, host);
	options.stage             = info.logical_stage;
	options.input_info.vertex = &info;
	options.user_data_base    = VERTEX_USER_DATA_BASE;
	// Their params already hold the reserved registers in front of the user data.
	if (options.stage == ShaderType::Mesh) {
		options.user_data_base = 0;
		options.wave_size      = info.mesh.wave_size;
	} else if (options.stage == ShaderType::TessellationControl) {
		options.user_data_base = 0;
		options.wave_size      = TESS_CONTROL_WAVE_SIZE;
	}
	return options;
}

ShaderRecompiler::CompileOptions GraphicsStageCompileOptions(const ShaderPixelInputInfo& info,
                                                             const GraphicsCompileHost&  host) {
	auto options             = HostOptions(info.wave_size, host);
	options.stage            = ShaderType::Pixel;
	options.input_info.pixel = &info;
	return options;
}

} // namespace Libs::Graphics
