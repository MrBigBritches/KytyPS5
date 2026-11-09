#include "common/assert.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineParts.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <array>
#include <chrono>
#include <fmt/format.h>

namespace Libs::Graphics {

namespace {

// Every library is made with this flag, and none asks to be kept for link time optimization. A
// pipeline linked from them is a fast link, which copies what the libraries compiled.
constexpr vk::PipelineCreateFlags LIBRARY_CREATE_FLAGS = vk::PipelineCreateFlagBits::eLibraryKHR;

// The set layouts and push stages of the pipeline that asked for the libraries. They come from its
// layout code, so a library and the pipeline linked from it agree on every set they both define.
struct LibraryLayoutSource {
	vk::DescriptorSetLayout vertex_set;
	vk::DescriptorSetLayout heap_set;
	vk::DescriptorSetLayout pixel_set;
	vk::ShaderStageFlags    push_stages;
};

// What every library build reads, which is the device, the driver cache and the pipeline asked for.
struct LibraryRequest {
	GraphicContext&                            graphics;
	vk::PipelineCache                          driver_cache;
	const GraphicsPipelineKey&                 key;
	const std::array<const ShaderModule*, 4>& stages;
};

// Logs a driver refusal. The caller builds the whole pipeline instead, so this is not fatal.
void LogRefusal(const char* what, vk::Result result) {
	Log::WriteToConsoleAndLog(fmt::format(
	    "Pipeline library: {} refused ({}), building the whole pipeline instead\n", what,
	    vk::to_string(result)));
}

// Makes the layout of a pre-rasterization or fragment shader library. A set the library does not know
// is null, because the sets of independent layouts that are not null must match when the libraries
// are linked, and the sets of the other stage are not the library's to name.
vk::PipelineLayout CreateLibraryLayout(GraphicContext& graphics, const LibraryLayoutSource& source,
                                       vk::DescriptorSetLayout vertex_set,
                                       vk::DescriptorSetLayout pixel_set) {
	const auto set_layouts = SplitSetLayouts(vertex_set, source.heap_set, pixel_set);
	const vk::PushConstantRange push_constants {source.push_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	vk::PipelineLayoutCreateInfo create {};
	create.flags                  = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
	create.setLayoutCount         = SPLIT_SET_COUNT;
	create.pSetLayouts            = set_layouts.data();
	create.pushConstantRangeCount = 1;
	create.pPushConstantRanges    = &push_constants;
	vk::PipelineLayout layout = nullptr;
	const auto         result = graphics.device.createPipelineLayout(&create, nullptr, &layout);
	if (result != vk::Result::eSuccess) {
		LogRefusal("pipeline layout", result);
		return nullptr;
	}
	return layout;
}

// Creates one library from a filled create info, or returns null handles after logging a refusal.
// A refused library takes its layout with it, since nothing else would destroy it.
LibraryHandles CreateLibrary(const LibraryRequest& request, LibraryStep step, const char* name,
                             const vk::GraphicsPipelineCreateInfo& info, vk::PipelineLayout layout) {
	auto&        device   = request.graphics.device;
	vk::Pipeline pipeline = nullptr;
	auto         result   = vk::Result::eErrorInitializationFailed;
	// The observer sees each library's create info, as it sees the whole pipeline's.
	if (const auto observer = GetGraphicsCreateInfoObserver()) {
		observer(request.key, request.stages, info);
	}
	if (!LibraryStepShouldFail(step)) {
		result = device.createGraphicsPipelines(request.driver_cache, 1, &info, nullptr, &pipeline);
	}
	if (result != vk::Result::eSuccess) {
		LogRefusal(name, result);
		device.destroyPipelineLayout(layout, nullptr);
		return {};
	}
	return {pipeline, layout};
}

// The vertex input library has no layout and no shader.
LibraryHandles BuildVertexInputLibrary(const LibraryRequest& request) {
	VertexInputParts parts;
	FillVertexInputParts(parts, request.key, *request.stages[VERTEX_STAGE_INDEX]);
	vk::GraphicsPipelineLibraryCreateInfoEXT library {
	    .flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eVertexInputInterface};
	vk::GraphicsPipelineCreateInfo info {};
	info.pNext               = &library;
	info.flags               = LIBRARY_CREATE_FLAGS;
	info.pVertexInputState   = &parts.vertex_input;
	info.pInputAssemblyState = &parts.input_assembly;
	return CreateLibrary(request, LibraryStep::VertexInput, "vertex input library", info, nullptr);
}

// The pre-rasterization library has the VS, the rasterizer state and the VS set of the layout. Its
// rendering info only says it has no view mask, since the formats are the output library's.
LibraryHandles BuildPreRasterLibrary(const LibraryRequest& request, const LibraryLayoutSource& source) {
	PreRasterParts    parts;
	DynamicStateParts dynamic;
	FillPreRasterParts(parts, request.graphics, request.key, request.stages, HelperModules {});
	FillDynamicStateParts(dynamic, request.graphics, request.key);
	const auto layout = CreateLibraryLayout(request.graphics, source, source.vertex_set, nullptr);
	if (layout == nullptr) {
		return {};
	}
	vk::GraphicsPipelineLibraryCreateInfoEXT library {
	    .flags = vk::GraphicsPipelineLibraryFlagBitsEXT::ePreRasterizationShaders};
	vk::PipelineRenderingCreateInfo view_mask_only {.pNext = &library};
	vk::GraphicsPipelineCreateInfo  info {};
	info.pNext               = &view_mask_only;
	info.flags               = LIBRARY_CREATE_FLAGS;
	info.stageCount          = parts.stage_count;
	info.pStages             = parts.stages.data();
	info.pTessellationState  = parts.tessellation_state;
	info.pViewportState      = &parts.viewport;
	info.pRasterizationState = &parts.rasterizer;
	info.pDynamicState       = &dynamic.pre_raster_info;
	info.layout              = layout;
	return CreateLibrary(request, LibraryStep::PreRaster, "pre-rasterization library", info, layout);
}

// The fragment shader library has the PS, the multisample state, a depth and stencil state that is
// never null, and the PS set of the layout. The PS is the module with its resources in set 2.
LibraryHandles BuildFragmentLibrary(const LibraryRequest& request, const LibraryLayoutSource& source) {
	FragmentShaderParts parts;
	DynamicStateParts   dynamic;
	const auto*         pixel = request.stages[PIXEL_STAGE_INDEX];
	FillFragmentShaderParts(parts, request.key, pixel);
	if (pixel != nullptr) {
		EXIT_IF(pixel->gpl_module == nullptr);
		parts.stage.module = pixel->gpl_module;
	}
	FillDynamicStateParts(dynamic, request.graphics, request.key);
	const auto layout = CreateLibraryLayout(request.graphics, source, nullptr, source.pixel_set);
	if (layout == nullptr) {
		return {};
	}
	vk::GraphicsPipelineLibraryCreateInfoEXT library {
	    .flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentShader};
	vk::PipelineRenderingCreateInfo view_mask_only {.pNext = &library};
	vk::GraphicsPipelineCreateInfo  info {};
	info.pNext              = &view_mask_only;
	info.flags              = LIBRARY_CREATE_FLAGS;
	info.stageCount         = parts.has_stage ? 1u : 0u;
	info.pStages            = &parts.stage;
	info.pMultisampleState  = &parts.multisampling;
	info.pDepthStencilState = &parts.depth_stencil;
	info.pDynamicState      = &dynamic.fragment_info;
	info.layout             = layout;
	return CreateLibrary(request, LibraryStep::Fragment, "fragment shader library", info, layout);
}

// The fragment output library has the attachment formats, blending and the multisample state. It
// has no layout.
LibraryHandles BuildOutputLibrary(const LibraryRequest& request) {
	FragmentOutputParts parts;
	DynamicStateParts   dynamic;
	FillFragmentOutputParts(parts, request.key);
	FillDynamicStateParts(dynamic, request.graphics, request.key);
	vk::GraphicsPipelineLibraryCreateInfoEXT library {
	    .flags = vk::GraphicsPipelineLibraryFlagBitsEXT::eFragmentOutputInterface};
	parts.rendering.pNext = &library;
	vk::GraphicsPipelineCreateInfo info {};
	info.pNext              = &parts.rendering;
	info.flags              = LIBRARY_CREATE_FLAGS;
	info.pColorBlendState   = &parts.blending;
	info.pMultisampleState  = &parts.multisampling;
	info.pDynamicState      = &dynamic.output_info;
	return CreateLibrary(request, LibraryStep::Output, "fragment output library", info, nullptr);
}

// Links the four libraries into the pipeline. The link takes the full layout of the pipeline, and no
// state of its own, since every part has been given to a library.
bool LinkLibraries(GraphicContext& graphics, vk::PipelineCache driver_cache,
                   GraphicsLibraryCache& cache, PipelineCache::Pipeline& pipeline,
                   const std::array<vk::Pipeline, LIBRARY_PART_COUNT>& libraries) {
	vk::PipelineLibraryCreateInfoKHR link {};
	link.libraryCount = static_cast<uint32_t>(libraries.size());
	link.pLibraries   = libraries.data();
	vk::GraphicsPipelineCreateInfo info {};
	info.pNext  = &link;
	info.layout = pipeline.pipeline_layout;
	const auto start  = std::chrono::steady_clock::now();
	auto       result = vk::Result::eErrorInitializationFailed;
	if (!LibraryStepShouldFail(LibraryStep::Link)) {
		result = graphics.device.createGraphicsPipelines(driver_cache, 1, &info, nullptr, &pipeline.pipeline);
	}
	if (result != vk::Result::eSuccess) {
		LogRefusal("link", result);
		pipeline.pipeline = nullptr;
		return false;
	}
	cache.NoteLink(
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
	return true;
}

} // namespace

bool ShouldUseLibraries(const GraphicContext& graphics, GraphicsLibraryCache* cache,
                        const GraphicsPipelineKey&                 key,
                        const std::array<const ShaderModule*, 4>& stages) {
	if (cache == nullptr || !graphics.graphics_pipeline_library_enabled) {
		return false;
	}
	if (!GplEligible(key, stages)) {
		cache->NoteFallback(GplFallbackReason(key, stages));
		return false;
	}
	return true;
}

bool CreatePipelineFromLibraries(GraphicContext& graphics, GraphicsLibraryCache& cache,
                                 PipelineCache::Pipeline& pipeline, const GraphicsPipelineKey& key,
                                 const std::array<const ShaderModule*, 4>& stages,
                                 vk::PipelineCache                          driver_cache) {
	EXIT_IF(!pipeline.uses_split_sets || pipeline.pipeline_layout == nullptr);
	// Both library layouts name the heap set, and a null one would make them disagree with the link.
	EXIT_IF(graphics.image_heap_layout == nullptr);
	const LibraryLayoutSource source {pipeline.vertex_set_layout, graphics.image_heap_layout,
	                                  pipeline.pixel_set_layout, pipeline.push_stages};
	const LibraryRequest request {graphics, driver_cache, key, stages};
	const auto vertex_input = cache.GetOrBuild(MakeVertexInputKey(key, stages),
	                                           [&] { return BuildVertexInputLibrary(request); });
	const auto pre_raster   = cache.GetOrBuild(MakePreRasterKey(key, stages),
	                                           [&] { return BuildPreRasterLibrary(request, source); });
	const auto fragment     = cache.GetOrBuild(MakeFragmentKey(key, stages),
	                                           [&] { return BuildFragmentLibrary(request, source); });
	const auto output       = cache.GetOrBuild(MakeFragmentOutputKey(key, stages),
	                                           [&] { return BuildOutputLibrary(request); });
	const std::array<vk::Pipeline, LIBRARY_PART_COUNT> libraries {
	    vertex_input.pipeline, pre_raster.pipeline, fragment.pipeline, output.pipeline};
	for (const auto library: libraries) {
		if (library == nullptr) {
			return false;
		}
	}
	return LinkLibraries(graphics, driver_cache, cache, pipeline, libraries);
}

} // namespace Libs::Graphics
