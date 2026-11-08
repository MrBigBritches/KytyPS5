#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_

namespace Libs::Graphics {

/// What --graphics-pipeline-library needs from the device. Each field is one requirement.
struct PipelineLibrarySupport {
	// The user asked for --graphics-pipeline-library.
	bool flag = false;
	// The device has VK_KHR_pipeline_library and VK_EXT_graphics_pipeline_library.
	bool extensions = false;
	// The graphicsPipelineLibrary feature.
	bool feature = false;
	// Linking libraries is fast. Without it a link costs about a compile and saves nothing.
	bool fast_linking = false;
	// The vertex and pixel libraries may disagree on interpolation decorations. Only the pixel
	// shader's inputs carry them, so without this the libraries could not be built separately.
	bool independent_interpolation = false;
};

/// The first requirement of --graphics-pipeline-library the device misses, for the log, or null
/// when every one is met and pipelines may be built from libraries.
[[nodiscard]] constexpr const char* GraphicsPipelineLibraryBlocker(const PipelineLibrarySupport& support) {
	if (!support.flag) {
		return "--graphics-pipeline-library is off";
	}
	if (!support.extensions) {
		return "VK_KHR_pipeline_library or VK_EXT_graphics_pipeline_library is missing";
	}
	if (!support.feature) {
		return "the graphicsPipelineLibrary feature is missing";
	}
	if (!support.fast_linking) {
		return "graphicsPipelineLibraryFastLinking is missing";
	}
	if (!support.independent_interpolation) {
		return "graphicsPipelineLibraryIndependentInterpolationDecoration is missing";
	}
	return nullptr;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINELIBRARY_H_
