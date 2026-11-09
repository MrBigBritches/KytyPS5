#include "common/assert.h"
#include "common/common.h"
#include "common/firstRunReport.h"
#include "common/profiler.h"
#include "gpu_blit_shaders/gpu_mesh_indirect_spv.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineParts.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/host_gpu/renderer/renderTarget.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <fmt/format.h>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics {

static vk::BlendFactor GetBlendFactor(uint32_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kZero: return vk::BlendFactor::eZero;
		case Prospero::BlendFactor::kOne: return vk::BlendFactor::eOne;
		case Prospero::BlendFactor::kSrcColor: return vk::BlendFactor::eSrcColor;
		case Prospero::BlendFactor::kOneMinusSrcColor: return vk::BlendFactor::eOneMinusSrcColor;
		case Prospero::BlendFactor::kSrcAlpha: return vk::BlendFactor::eSrcAlpha;
		case Prospero::BlendFactor::kOneMinusSrcAlpha: return vk::BlendFactor::eOneMinusSrcAlpha;
		case Prospero::BlendFactor::kDstAlpha: return vk::BlendFactor::eDstAlpha;
		case Prospero::BlendFactor::kOneMinusDstAlpha: return vk::BlendFactor::eOneMinusDstAlpha;
		case Prospero::BlendFactor::kDstColor: return vk::BlendFactor::eDstColor;
		case Prospero::BlendFactor::kOneMinusDstColor: return vk::BlendFactor::eOneMinusDstColor;
		case Prospero::BlendFactor::kSrcAlphaSaturate: return vk::BlendFactor::eSrcAlphaSaturate;
		case Prospero::BlendFactor::kConstantColor: return vk::BlendFactor::eConstantColor;
		case Prospero::BlendFactor::kOneMinusConstantColor:
			return vk::BlendFactor::eOneMinusConstantColor;
		case Prospero::BlendFactor::kSrc1Color: return vk::BlendFactor::eSrc1Color;
		case Prospero::BlendFactor::kOneMinusSrc1Color: return vk::BlendFactor::eOneMinusSrc1Color;
		case Prospero::BlendFactor::kSrc1Alpha: return vk::BlendFactor::eSrc1Alpha;
		case Prospero::BlendFactor::kOneMinusSrc1Alpha: return vk::BlendFactor::eOneMinusSrc1Alpha;
		case Prospero::BlendFactor::kConstantAlpha: return vk::BlendFactor::eConstantAlpha;
		case Prospero::BlendFactor::kOneMinusConstantAlpha:
			return vk::BlendFactor::eOneMinusConstantAlpha;
		default: EXIT("unknown factor: %u\n", factor);
	}
	return vk::BlendFactor::eZero;
}

static vk::BlendOp GetBlendOp(uint32_t op) {
	switch (static_cast<Prospero::BlendOp>(op)) {
		case Prospero::BlendOp::kAdd: return vk::BlendOp::eAdd;
		case Prospero::BlendOp::kSubtract: return vk::BlendOp::eSubtract;
		case Prospero::BlendOp::kMin: return vk::BlendOp::eMin;
		case Prospero::BlendOp::kMax: return vk::BlendOp::eMax;
		case Prospero::BlendOp::kReverseSubtract: return vk::BlendOp::eReverseSubtract;
		default: EXIT("unknown op: %u\n", op);
	}
	return vk::BlendOp::eAdd;
}

static void AddLayoutBindings(std::vector<vk::DescriptorSetLayoutBinding>& descriptor_bindings,
                              const ShaderRecompiler::IR::CompiledShaderInfo& program,
                              vk::ShaderStageFlagBits              stage) {
	for (auto mask = program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
		const auto index = std::countr_zero(mask);
		const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
		descriptor_bindings.push_back(
		    {ShaderRecompiler::IR::NativeBinding(program.stage, kind), NativeDescriptorType(kind),
		     program.bindings.descriptor_counts[index], stage, nullptr});
	}
}

// Whether a layout of this many descriptors uses push descriptors. The layout code and the
// prediction for a pipeline not built yet both ask this.
static bool UsesPushDescriptors(const GraphicContext& graphics, uint64_t descriptor_count) {
	return descriptor_count <= graphics.max_push_descriptors;
}

// The layout bindings that the stages of a graphics pipeline add to its descriptor set.
static std::vector<vk::DescriptorSetLayoutBinding>
GatherLayoutBindings(const std::array<const ShaderModule*, 4>& stages) {
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	for (const auto* stage: stages) {
		if (stage != nullptr) {
			AddLayoutBindings(bindings, stage->info, NativeShaderStage(stage->info.stage));
		}
	}
	return bindings;
}

namespace {

// The bindings of the two sets of the split layout, and how the PS set is bound. Both the layout code
// and the prediction build it, so a draw is judged by the layout it will get.
struct SplitSetPlan {
	std::vector<vk::DescriptorSetLayoutBinding> vertex;
	std::vector<vk::DescriptorSetLayoutBinding> pixel;
	uint64_t                                    pixel_count = 0;
	// The PS set is pushed. It is the only set that can be.
	bool                                        pixel_push = false;
	// No descriptor set has to be allocated, which needs a VS with no bindings and a pushed PS set.
	bool                                        push_only = false;
};

} // namespace

static SplitSetPlan PlanSplitSets(const GraphicContext&                       graphics,
                                  const std::array<const ShaderModule*, 4>& stages) {
	SplitSetPlan plan;
	// A pipeline the split layout is used for has no tessellation stages, so only these two count.
	const auto* vertex = stages[VERTEX_STAGE_INDEX];
	const auto* pixel  = stages[PIXEL_STAGE_INDEX];
	if (vertex != nullptr) {
		AddLayoutBindings(plan.vertex, vertex->info, NativeShaderStage(vertex->info.stage));
	}
	if (pixel != nullptr) {
		AddLayoutBindings(plan.pixel, pixel->info, NativeShaderStage(pixel->info.stage));
	}
	for (const auto& binding: plan.pixel) {
		plan.pixel_count += binding.descriptorCount;
	}
	plan.pixel_push = UsesPushDescriptors(graphics, plan.pixel_count);
	plan.push_only  = plan.vertex.empty() && plan.pixel_push;
	return plan;
}

bool UsesSplitLayout(const GraphicContext& graphics, const GraphicsPipelineKey& key,
                     const std::array<const ShaderModule*, 4>& stages) {
	return graphics.graphics_pipeline_library_enabled && GplEligible(key, stages);
}

bool PredictPushDescriptors(const GraphicContext& graphics, const GraphicsPipelineKey& key,
                            const std::array<const ShaderModule*, 4>& stages) {
	if (UsesSplitLayout(graphics, key, stages)) {
		return PlanSplitSets(graphics, stages).push_only;
	}
	uint64_t descriptor_count = 0;
	for (const auto& binding: GatherLayoutBindings(stages)) {
		descriptor_count += binding.descriptorCount;
	}
	return UsesPushDescriptors(graphics, descriptor_count);
}

vk::ShaderModule CompileSplitPixelModule(const GraphicContext& graphics, ShaderType stage,
                                         std::span<const uint32_t> spirv) {
	if (!graphics.graphics_pipeline_library_enabled || stage != ShaderType::Pixel) {
		return nullptr;
	}
	// Only this copy is patched, so module ids and the shader store keep the plain SPIR-V.
	const auto moved = MoveDescriptorSet(spirv, EMITTED_DESCRIPTOR_SET, SPLIT_PIXEL_SET);
	return CompileSPV(moved, graphics.device);
}

// Counts the set layouts that DestroyPipelineObjects destroyed. A tripwire for tests, since the
// layouts of a split pipeline belong to the cache and must never be counted here. Tests read it as
// a before and after difference, which holds while no other thread destroys pipelines meanwhile.
static std::atomic<uint64_t> g_pipeline_set_layout_destroys {0};

uint64_t PipelineSetLayoutDestroyCount() {
	return g_pipeline_set_layout_destroys.load();
}

// Destroys a set layout that a pipeline owns, and counts it. Every set layout destroy of a pipeline
// goes through here, so the count shows a pipeline destroying a layout.
static void DestroyOwnedSetLayout(vk::Device device, vk::DescriptorSetLayout layout) {
	if (layout != nullptr) {
		device.destroyDescriptorSetLayout(layout, nullptr);
		g_pipeline_set_layout_destroys++;
	}
}

void DestroyPipelineObjects(vk::Device device, const PipelineCache::Pipeline& pipeline) {
	device.destroyPipeline(pipeline.pipeline, nullptr);
	if (pipeline.linked_from != nullptr) {
		pipeline.linked_from->NoteUnlinked();
	}
	device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
	// Only this layout is the pipeline's own. A split pipeline has none, and its vertex and pixel set
	// layouts are shared, so they are left for the DescriptorSetLayoutCache to destroy once.
	DestroyOwnedSetLayout(device, pipeline.descriptor_set_layout);
}

std::string DescriptorLimitError(const vk::PhysicalDeviceLimits&                limits,
                                 std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                 uint64_t&                                       descriptor_count) {
	descriptor_count = 0;
	uint64_t storage_buffers = 0;
	uint64_t samplers = 0;
	std::array<uint64_t, 32> stage_buffers {}, stage_resources {}, stage_samplers {};
	for (const auto& binding: bindings) {
		descriptor_count += binding.descriptorCount;
		const bool storage = binding.descriptorType == vk::DescriptorType::eStorageBuffer;
		const bool sampler = binding.descriptorType == vk::DescriptorType::eSampler;
		if (storage) storage_buffers += binding.descriptorCount;
		if (sampler) samplers += binding.descriptorCount;
		for (auto stages = static_cast<uint32_t>(binding.stageFlags); stages != 0; stages &= stages - 1u) {
			const auto stage = std::countr_zero(stages);
			if (storage) stage_buffers[stage] += binding.descriptorCount;
			if (sampler) stage_samplers[stage] += binding.descriptorCount;
			else
				stage_resources[stage] += binding.descriptorCount;
			if (stage_buffers[stage] > limits.maxPerStageDescriptorStorageBuffers ||
			    stage_resources[stage] > limits.maxPerStageResources ||
			    stage_samplers[stage] > limits.maxPerStageDescriptorSamplers) {
				return fmt::format(
				    "native descriptor stage {} exceeds device limits: buffers={}/{} resources={}/{} "
				    "samplers={}/{}\n", stage, stage_buffers[stage],
				    limits.maxPerStageDescriptorStorageBuffers, stage_resources[stage],
				    limits.maxPerStageResources, stage_samplers[stage],
				    limits.maxPerStageDescriptorSamplers);
			}
		}
	}
	if (storage_buffers > limits.maxDescriptorSetStorageBuffers) {
		return fmt::format("native storage-buffer descriptors exceed device limit: {}/{}\n",
		                   storage_buffers, limits.maxDescriptorSetStorageBuffers);
	}
	if (samplers > limits.maxDescriptorSetSamplers) {
		return fmt::format("native sampler descriptors exceed device limit: {}/{}\n", samplers,
		                   limits.maxDescriptorSetSamplers);
	}
	return {};
}

std::string SplitLayoutLimitError(const vk::PhysicalDeviceLimits&                limits,
                                  std::span<const vk::DescriptorSetLayoutBinding> vertex,
                                  std::span<const vk::DescriptorSetLayoutBinding> pixel) {
	// The set counts of the limits cover a whole pipeline layout, so the two sets are judged together.
	std::vector<vk::DescriptorSetLayoutBinding> both(vertex.begin(), vertex.end());
	both.insert(both.end(), pixel.begin(), pixel.end());
	uint64_t descriptor_count = 0;
	return DescriptorLimitError(limits, both, descriptor_count);
}

// Checks the bindings of one descriptor set against the device limits, which stops the process when
// one is exceeded, and returns how many descriptors the set has.
static uint64_t CheckDescriptorLimits(const GraphicContext&                         graphics,
                                      std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	uint64_t   descriptor_count = 0;
	const auto error =
	    DescriptorLimitError(graphics.GetPhysicalDeviceProperties().limits, bindings, descriptor_count);
	if (!error.empty()) {
		EXIT("%s", error.c_str());
	}
	return descriptor_count;
}

// The create flags of a split set layout, which are push or none.
static vk::DescriptorSetLayoutCreateFlags SetLayoutFlags(bool push) {
	return push ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	            : vk::DescriptorSetLayoutCreateFlags {};
}

// Creates a descriptor set layout, which may have no bindings.
static vk::DescriptorSetLayout CreateSetLayout(const GraphicContext&                         graphics,
                                               std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                               bool push) {
	vk::DescriptorSetLayoutCreateInfo create {};
	create.flags        = push ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
	                           : vk::DescriptorSetLayoutCreateFlags {};
	create.bindingCount = static_cast<uint32_t>(bindings.size());
	create.pBindings    = bindings.data();
	vk::DescriptorSetLayout layout = nullptr;
	EXIT_IF(graphics.device.createDescriptorSetLayout(&create, nullptr, &layout) !=
	        vk::Result::eSuccess);
	return layout;
}

// Creates the descriptor set layout and the pipeline layout that uses it.
static void CreateLayouts(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                          std::span<const vk::DescriptorSetLayoutBinding> bindings) {
	const auto descriptor_count    = CheckDescriptorLimits(graphics, bindings);
	pipeline.uses_push_descriptors = UsesPushDescriptors(graphics, descriptor_count);
	pipeline.descriptor_set_layout =
	    CreateSetLayout(graphics, bindings, pipeline.uses_push_descriptors);
	const vk::PushConstantRange push_constants {pipeline.push_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	const std::array set_layouts {pipeline.descriptor_set_layout, graphics.image_heap_layout};

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.setLayoutCount         = pipeline.uses_image_heap ? 2u : 1u;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;

	EXIT_NOT_IMPLEMENTED(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &pipeline.pipeline_layout) !=
	                     vk::Result::eSuccess);
}

// One push constant range for both stages, whatever the pipeline has, so that the layouts of
// different pipelines agree and a library layout can be built from either stage alone.
static constexpr vk::ShaderStageFlags SPLIT_PUSH_STAGES =
    vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;

// Creates the split layout. Set 0 holds the VS bindings, set 1 is the image heap exactly as in the
// single-set layout, and set 2 holds the PS bindings, pushed when the PS alone fits. The sets are
// independent, so a pipeline library that knows one stage can still be given a layout. The two set
// layouts, empty ones too, come from the shared cache. The pipeline layout stays the pipeline's own.
static void CreateSplitLayouts(GraphicContext& graphics, DescriptorSetLayoutCache& shared_layouts,
                               PipelineCache::Pipeline& pipeline,
                               const std::array<const ShaderModule*, 4>& stages) {
	const auto plan = PlanSplitSets(graphics, stages);
	const auto limit_error = SplitLayoutLimitError(graphics.GetPhysicalDeviceProperties().limits,
	                                               plan.vertex, plan.pixel);
	if (!limit_error.empty()) {
		EXIT("%s", limit_error.c_str());
	}
	pipeline.uses_split_sets       = true;
	pipeline.pixel_set_is_push     = plan.pixel_push;
	pipeline.uses_push_descriptors = plan.push_only;
	pipeline.vertex_set_layout     = shared_layouts.GetOrCreate(plan.vertex, {});
	pipeline.pixel_set_layout      =
	    shared_layouts.GetOrCreate(plan.pixel, SetLayoutFlags(plan.pixel_push));
	pipeline.push_stages           = SPLIT_PUSH_STAGES;
	const vk::PushConstantRange push_constants {pipeline.push_stages, 0,
	                                            ShaderRecompiler::IR::NativePushConstantSize};
	// The image heap is in the layout even when this pipeline does not use it, since the layout of a
	// part library cannot know what the other stage uses.
	const auto set_layouts = SplitSetLayouts(pipeline.vertex_set_layout, graphics.image_heap_layout,
	                                         pipeline.pixel_set_layout);

	vk::PipelineLayoutCreateInfo pipeline_layout_info {};
	pipeline_layout_info.flags                  = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
	pipeline_layout_info.setLayoutCount         = SPLIT_SET_COUNT;
	pipeline_layout_info.pSetLayouts            = set_layouts.data();
	pipeline_layout_info.pushConstantRangeCount = 1;
	pipeline_layout_info.pPushConstantRanges    = &push_constants;
	EXIT_NOT_IMPLEMENTED(graphics.device.createPipelineLayout(&pipeline_layout_info, nullptr,
	                                                          &pipeline.pipeline_layout) !=
	                     vk::Result::eSuccess);
}

// The create info of one stage, whose entry point is always main.
static vk::PipelineShaderStageCreateInfo MakeStageInfo(vk::ShaderStageFlagBits stage,
                                                       vk::ShaderModule        module) {
	return {.stage = stage, .module = module, .pName = "main"};
}

// Both the fragment shader part and the fragment output part carry this state, and they must agree,
// so they fill it from the same place.
static void FillMultisampleState(vk::PipelineMultisampleStateCreateInfo& multisampling,
                                 const PipelineStaticParameters&         static_params) {
	multisampling.sampleShadingEnable  = static_params.sample_shading_enable ? VK_TRUE : VK_FALSE;
	multisampling.rasterizationSamples = vulkan_sample_count(static_params.samples);
	multisampling.minSampleShading     = 1.0f;
	multisampling.alphaToCoverageEnable =
	    static_params.alpha_to_coverage_enable ? VK_TRUE : VK_FALSE;
}

// Fills the vertex bindings, attributes and input assembly from the key and the VS formats.
void FillVertexInputParts(VertexInputParts& out, const GraphicsPipelineKey& key,
                          const ShaderModule& vertex) {
	const auto& vertex_input  = key.vertex_input;
	const auto& static_params = key.static_params;
	for (uint32_t binding = 0; binding < vertex_input.binding_count; binding++) {
		out.bindings[binding].binding   = binding;
		out.bindings[binding].stride    = vertex_input.bindings[binding].stride;
		out.bindings[binding].inputRate = vertex_input.bindings[binding].instance
		                                      ? vk::VertexInputRate::eInstance
		                                      : vk::VertexInputRate::eVertex;
	}
	for (uint32_t index = 0; index < vertex_input.attribute_count; index++) {
		out.attributes[index].binding  = vertex_input.attributes[index].binding;
		out.attributes[index].location = index;
		out.attributes[index].offset   = vertex_input.attributes[index].offset;
		out.attributes[index].format   = vertex.vertex_formats[index];
	}
	out.vertex_input.vertexBindingDescriptionCount   = vertex_input.binding_count;
	out.vertex_input.pVertexBindingDescriptions      = out.bindings.data();
	out.vertex_input.vertexAttributeDescriptionCount = vertex_input.attribute_count;
	out.vertex_input.pVertexAttributeDescriptions    = out.attributes.data();

	out.input_assembly.topology = static_params.topology;
	out.input_assembly.primitiveRestartEnable =
	    static_params.primitive_restart_enable ? VK_TRUE : VK_FALSE;
}

// Appends one stage to the pre-rasterization part, stopping if the stage array would overflow.
static void AppendPreRasterStage(PreRasterParts& out, const vk::PipelineShaderStageCreateInfo& stage) {
	EXIT_IF(out.stage_count >= PIPELINE_STAGES_MAX);
	out.stages[out.stage_count++] = stage;
}

// Adds the shader stages of the pre-rasterization part. The draw's own stages come first, which
// are the VS, HS and TES, and then the helper stages.
static void FillPreRasterStages(PreRasterParts&                           out,
                                const std::array<const ShaderModule*, 4>& stages,
                                const HelperModules&                       helpers) {
	out.stage_count = 0;
	// The PS belongs to the fragment shader part.
	for (size_t index = 0; index < PRE_RASTER_STAGE_COUNT; index++) {
		if (stages[index] != nullptr) {
			AppendPreRasterStage(out, MakeStageInfo(NativeShaderStage(stages[index]->info.stage),
			                                        stages[index]->module));
		}
	}
	out.helper_begin = out.stage_count;
	if (helpers.task) {
		AppendPreRasterStage(out, MakeStageInfo(vk::ShaderStageFlagBits::eTaskEXT, helpers.task));
	}
	if (helpers.tess_control) {
		AppendPreRasterStage(out, MakeStageInfo(vk::ShaderStageFlagBits::eTessellationControl,
		                                        helpers.tess_control));
		AppendPreRasterStage(out, MakeStageInfo(vk::ShaderStageFlagBits::eTessellationEvaluation,
		                                        helpers.tess_eval));
	}
}

// Fills the pre-rasterization stages, viewport, rasterizer and tessellation state.
void FillPreRasterParts(PreRasterParts& out, const GraphicContext& graphics,
                        const GraphicsPipelineKey&                 key,
                        const std::array<const ShaderModule*, 4>& stages,
                        const HelperModules&                       helpers) {
	const auto& static_params = key.static_params;
	FillPreRasterStages(out, stages, helpers);

	out.depth_clip_control.negativeOneToOne = (static_params.negative_one_to_one ? VK_TRUE : VK_FALSE);
	out.viewport.pNext                      = &out.depth_clip_control;
	out.depth_clip.depthClipEnable          = static_params.depth_clip_enable ? VK_TRUE : VK_FALSE;

	// MoltenVK lacks VK_EXT_depth_clip_enable; omit the depth-clip struct on macOS and accept
	// Vulkan's default depth clipping (enabled) instead of the PS5's clamp behavior.
#if !defined(__APPLE__)
	// The DB clamps depth to the viewport range after polygon offset is applied.
	out.rasterizer.depthClampEnable = VK_TRUE;
	out.rasterizer.pNext            = &out.depth_clip;
#endif
	EXIT_NOT_IMPLEMENTED(static_params.provoking_vtx_last &&
	                     !graphics.provoking_vertex_last_enabled);
	if (graphics.provoking_vertex_last_enabled) {
		out.provoking_vertex.provokingVertexMode = static_params.provoking_vtx_last
		    ? vk::ProvokingVertexModeEXT::eLastVertex : vk::ProvokingVertexModeEXT::eFirstVertex;
		out.provoking_vertex.pNext = out.rasterizer.pNext;
		out.rasterizer.pNext       = &out.provoking_vertex;
	}
	out.rasterizer.polygonMode = static_params.polygon_mode;

	out.tessellation.patchControlPoints = static_params.patch_control_points;
	out.tessellation_state = static_params.patch_control_points != 0 ? &out.tessellation : nullptr;
}

// Fills the PS stage and the multisample state, beside a value-initialised depth-stencil state.
void FillFragmentShaderParts(FragmentShaderParts& out, const GraphicsPipelineKey& key,
                             const ShaderModule* pixel) {
	out.has_stage = pixel != nullptr;
	if (pixel != nullptr) {
		out.stage = MakeStageInfo(NativeShaderStage(pixel->info.stage), pixel->module);
	}
	FillMultisampleState(out.multisampling, key.static_params);
}

// Fills the blend state of one color attachment.
static void FillBlendAttachment(vk::PipelineColorBlendAttachmentState& attachment,
                                const PipelineStaticParameters& static_params, uint32_t index) {
	EXIT_NOT_IMPLEMENTED((static_params.color_mask[index] & ~0x0fu) != 0);
	attachment.colorWriteMask = vk::ColorComponentFlags {static_params.color_mask[index]};
	attachment.blendEnable    = static_params.blend_enable[index] ? VK_TRUE : VK_FALSE;
	attachment.srcColorBlendFactor = GetBlendFactor(static_params.color_srcblend[index]);
	attachment.dstColorBlendFactor = GetBlendFactor(static_params.color_destblend[index]);
	attachment.colorBlendOp        = GetBlendOp(static_params.color_comb_fcn[index]);
	// Without separate alpha blending the alpha channel blends like the color channels.
	const bool separate_alpha = static_params.separate_alpha_blend[index];
	attachment.srcAlphaBlendFactor =
	    (separate_alpha ? GetBlendFactor(static_params.alpha_srcblend[index])
	                    : attachment.srcColorBlendFactor);
	attachment.dstAlphaBlendFactor =
	    (separate_alpha ? GetBlendFactor(static_params.alpha_destblend[index])
	                    : attachment.dstColorBlendFactor);
	attachment.alphaBlendOp =
	    (separate_alpha ? GetBlendOp(static_params.alpha_comb_fcn[index]) : attachment.colorBlendOp);
}

// Fills the blend attachments, blend state, rendering info and multisample state.
void FillFragmentOutputParts(FragmentOutputParts& out, const GraphicsPipelineKey& key) {
	const auto& rendering = key.rendering;
	for (uint32_t index = 0; index < rendering.color_count; index++) {
		FillBlendAttachment(out.attachments[index], key.static_params, index);
	}
	out.blending.attachmentCount = rendering.color_count;
	out.blending.pAttachments    = out.attachments.data();

	out.rendering.colorAttachmentCount    = rendering.color_count;
	out.rendering.pColorAttachmentFormats = rendering.color_formats.data();
	out.rendering.depthAttachmentFormat   = rendering.depth_format;
	out.rendering.stencilAttachmentFormat = rendering.stencil_format;
	FillMultisampleState(out.multisampling, key.static_params);
}

namespace {

// Which libraries read a dynamic state, as bits.
enum DynamicOwner : unsigned {
	OwnerPreRaster = 1,
	OwnerFragment  = 2,
	OwnerOutput    = 4,
};

// A dynamic state and the libraries that read it, as OwnerPreRaster, OwnerFragment and OwnerOutput bits.
struct OwnedState {
	vk::DynamicState state;
	unsigned         owners;
};

// Adds the state to the whole list and to the list of every library that reads it.
void AddOwnedState(DynamicStateParts& out, const OwnedState& owned) {
	out.states.push_back(owned.state);
	if ((owned.owners & OwnerPreRaster) != 0) out.pre_raster_states.push_back(owned.state);
	if ((owned.owners & OwnerFragment) != 0) out.fragment_states.push_back(owned.state);
	if ((owned.owners & OwnerOutput) != 0) out.output_states.push_back(owned.state);
}

// Points a create info at a state list.
void PointAtStates(vk::PipelineDynamicStateCreateInfo& info,
                   const std::vector<vk::DynamicState>& states) {
	info.dynamicStateCount = static_cast<uint32_t>(states.size());
	info.pDynamicStates    = states.data();
}

} // namespace

// Fills the whole dynamic state list in the order the monolithic pipeline has always had, and cuts
// the list of each library from it. The viewport, scissor, rasterizer and depth bias states belong to
// the pre-rasterization part. Depth and stencil states belong to the fragment shader part, and blend
// constants and color write enable to the fragment output part. The feedback loop state replaces the
// feedback loop create flags, which belong to the fragment output part. It is also listed with the
// fragment shader part, since depth and stencil attachments are among what it switches, and the
// validation layer accepts it there.
void FillDynamicStateParts(DynamicStateParts& out, const GraphicContext& graphics,
                           const GraphicsPipelineKey& key) {
	using State = vk::DynamicState;
	for (const auto& owned: {
	         OwnedState {State::eViewportWithCount, OwnerPreRaster},
	         OwnedState {State::eScissorWithCount, OwnerPreRaster},
	         OwnedState {State::eLineWidth, OwnerPreRaster},
	         OwnedState {State::eCullMode, OwnerPreRaster},
	         OwnedState {State::eFrontFace, OwnerPreRaster},
	         OwnedState {State::eDepthTestEnable, OwnerFragment},
	         OwnedState {State::eDepthWriteEnable, OwnerFragment},
	         OwnedState {State::eDepthCompareOp, OwnerFragment},
	         OwnedState {State::eDepthBiasEnable, OwnerPreRaster},
	         OwnedState {State::eDepthBias, OwnerPreRaster},
	         OwnedState {State::eStencilTestEnable, OwnerFragment},
	         OwnedState {State::eStencilOp, OwnerFragment},
	         OwnedState {State::eStencilCompareMask, OwnerFragment},
	         OwnedState {State::eStencilReference, OwnerFragment},
	         OwnedState {State::eStencilWriteMask, OwnerFragment},
	         OwnedState {State::eBlendConstants, OwnerOutput},
	     }) {
		AddOwnedState(out, owned);
	}
#if !defined(__APPLE__)
	AddOwnedState(out, {State::eDepthBoundsTestEnable, OwnerFragment});
	AddOwnedState(out, {State::eDepthBounds, OwnerFragment});
	if (key.rendering.color_count != 0) {
		AddOwnedState(out, {State::eColorWriteEnableEXT, OwnerOutput});
	}
#endif
	if (graphics.attachment_feedback_loop_enabled) {
		AddOwnedState(out, {State::eAttachmentFeedbackLoopEnableEXT, OwnerFragment | OwnerOutput});
	}
	PointAtStates(out.info, out.states);
	PointAtStates(out.pre_raster_info, out.pre_raster_states);
	PointAtStates(out.fragment_info, out.fragment_states);
	PointAtStates(out.output_info, out.output_states);
}

// Sets up the push stages, the image heap flag and the layouts of a graphics pipeline. The layout
// is shared by all four parts, so it is set up apart from them. A pipeline that libraries could
// build gets the split layout when the library mode is on, and every other one gets the single set.
static void CreateGraphicsLayouts(GraphicContext& graphics, DescriptorSetLayoutCache& set_layouts,
                                  PipelineCache::Pipeline& pipeline,
                                  const GraphicsPipelineKey&                 key,
                                  const std::array<const ShaderModule*, 4>& stages,
                                  bool                                       has_task_stage) {
	for (const auto* stage: stages) {
		if (stage != nullptr) {
			pipeline.uses_image_heap |= stage->info.info.uses_image_heap;
		}
	}
	if (UsesSplitLayout(graphics, key, stages)) {
		CreateSplitLayouts(graphics, set_layouts, pipeline, stages);
		return;
	}
	const auto descriptor_bindings = GatherLayoutBindings(stages);
	pipeline.push_stages = vk::ShaderStageFlagBits::eFragment;
	if (has_task_stage) {
		pipeline.push_stages |= vk::ShaderStageFlagBits::eTaskEXT;
	}
	for (const auto* stage: stages) {
		if (stage != nullptr) {
			pipeline.push_stages |= NativeShaderStage(stage->info.stage);
		}
	}
	CreateLayouts(graphics, pipeline, descriptor_bindings);
}

namespace {

// The four parts and the dynamic states of one monolithic pipeline, built in place and joined
// by AssembleMonolithicInfo.
struct MonolithicParts {
	VertexInputParts                                                   vertex_input;
	PreRasterParts                                                     pre_raster;
	FragmentShaderParts                                                fragment_shader;
	FragmentOutputParts                                                fragment_output;
	DynamicStateParts                                                  dynamic;
	// The stages of the whole pipeline in the order the driver has always been given.
	std::array<vk::PipelineShaderStageCreateInfo, PIPELINE_STAGES_MAX> stages {};
	uint32_t                                                           stage_count = 0;
	InPlaceOnly                                                        in_place;
};

// Joins the stages of the pre-rasterization and fragment shader parts. The PS goes between the
// draw's own stages and the helper stages, which is the order the monolithic pipeline has always
// used.
void ComposeMonolithicStages(MonolithicParts& parts) {
	const auto& pre   = parts.pre_raster;
	uint32_t&   count = parts.stage_count;
	count             = 0;
	// The pre-rasterization stages and the PS together must still fit the array.
	EXIT_IF(pre.stage_count + (parts.fragment_shader.has_stage ? 1u : 0u) > PIPELINE_STAGES_MAX);
	for (uint32_t index = 0; index < pre.helper_begin; index++) {
		parts.stages[count++] = pre.stages[index];
	}
	if (parts.fragment_shader.has_stage) {
		parts.stages[count++] = parts.fragment_shader.stage;
	}
	for (uint32_t index = pre.helper_begin; index < pre.stage_count; index++) {
		parts.stages[count++] = pre.stages[index];
	}
}

// Points a create info at the composed parts. A mesh pipeline has no vertex input or input
// assembly state, and a pipeline without a depth or stencil target has no depth-stencil state.
void AssembleMonolithicInfo(vk::GraphicsPipelineCreateInfo& info, MonolithicParts& parts,
                            bool mesh, bool with_depth, vk::PipelineLayout layout) {
	ComposeMonolithicStages(parts);
	info.pNext               = &parts.fragment_output.rendering;
	info.stageCount          = parts.stage_count;
	info.pStages             = parts.stages.data();
	info.pVertexInputState   = mesh ? nullptr : &parts.vertex_input.vertex_input;
	info.pInputAssemblyState = mesh ? nullptr : &parts.vertex_input.input_assembly;
	info.pTessellationState  = parts.pre_raster.tessellation_state;
	info.pViewportState      = &parts.pre_raster.viewport;
	info.pRasterizationState = &parts.pre_raster.rasterizer;
	info.pMultisampleState   = &parts.fragment_shader.multisampling;
	info.pDepthStencilState  = (with_depth ? &parts.fragment_shader.depth_stencil : nullptr);
	info.pColorBlendState    = &parts.fragment_output.blending;
	info.pDynamicState       = &parts.dynamic.info;
	info.layout              = layout;
}

} // namespace

// Makes the modules a pipeline needs besides its draw's own, which are the task shader of an
// indirect mesh draw and the tessellation shaders of a rectangle list. The caller destroys them.
static HelperModules CreateHelperModules(GraphicContext& graphics, const ShaderModule& vertex,
                                         const ShaderModule* pixel, bool rect_list) {
	HelperModules helpers;
	if (vertex.mesh_task) {
		helpers.task = CompileSPV(GPU_MESH_INDIRECT_SPV, graphics.device);
	}
	if (rect_list) {
		std::span<const RectListParameter> parameters;
		if (pixel != nullptr) parameters = pixel->rect_parameters;
		const auto shaders   = BuildRectListShaders(parameters, vertex.info.param_export_mask);
		helpers.tess_control = CompileSPV(shaders.control, graphics.device);
		helpers.tess_eval    = CompileSPV(shaders.evaluation, graphics.device);
	}
	return helpers;
}

// Links the pipeline from libraries when the library mode is on and the pipeline is one libraries
// cover. Returns true when the pipeline is made. A pipeline that libraries do not cover, or whose
// libraries the driver refused, is counted as built whole and left to the caller, and `layouts_ready`
// says whether its layouts were made here, so the caller does not make them twice.
static bool TryLinkFromLibraries(GraphicContext& graphics, DescriptorSetLayoutCache& set_layouts,
                                 GraphicsLibraryCache* libraries, PipelineCache::Pipeline& pipeline,
                                 const GraphicsPipelineKey&                 key,
                                 const std::array<const ShaderModule*, 4>& stages,
                                 vk::PipelineCache driver_cache, bool& layouts_ready) {
	layouts_ready = false;
	if (!ShouldUseLibraries(graphics, libraries, key, stages)) {
		return false;
	}
	// The link needs the full layout, which is the same one a whole pipeline would get.
	CreateGraphicsLayouts(graphics, set_layouts, pipeline, key, stages, false);
	layouts_ready = true;
	if (CreatePipelineFromLibraries(graphics, *libraries, pipeline, key, stages, driver_cache)) {
		pipeline.from_libraries = true;
		pipeline.linked_from    = libraries;
		return true;
	}
	libraries->NoteFallback(Common::FirstRunReport::FallbackKind::CreateFailed);
	return false;
}

// Builds the pipeline whole, from the four builders and one create info. `layouts_ready` says that a
// failed attempt at libraries already made the layouts.
static void CreateWholePipeline(GraphicContext& graphics, DescriptorSetLayoutCache& set_layouts,
                                PipelineCache::Pipeline&                   pipeline,
                                const GraphicsPipelineKey&                 key,
                                const std::array<const ShaderModule*, 4>& stages,
                                vk::PipelineCache driver_cache, bool layouts_ready) {
	const auto& rendering  = key.rendering;
	const auto& vertex     = *stages[VERTEX_STAGE_INDEX];
	const auto* pixel      = stages[PIXEL_STAGE_INDEX];
	const bool  with_depth = rendering.depth_format != vk::Format::eUndefined ||
	                         rendering.stencil_format != vk::Format::eUndefined;
	const bool  mesh       = vertex.info.stage == ShaderType::Mesh;
	EXIT_NOT_IMPLEMENTED(mesh && !graphics.mesh_shader_enabled);
	const bool rect_list = key.static_params.patch_control_points != 0 && stages[1] == nullptr;
	const auto helpers   = CreateHelperModules(graphics, vertex, pixel, rect_list);

	MonolithicParts parts;
	FillVertexInputParts(parts.vertex_input, key, vertex);
	FillPreRasterParts(parts.pre_raster, graphics, key, stages, helpers);
	FillFragmentShaderParts(parts.fragment_shader, key, pixel);
	FillFragmentOutputParts(parts.fragment_output, key);
	FillDynamicStateParts(parts.dynamic, graphics, key);
	if (!layouts_ready) {
		CreateGraphicsLayouts(graphics, set_layouts, pipeline, key, stages,
		                      static_cast<bool>(helpers.task));
	}

	if (pipeline.uses_split_sets && pixel != nullptr) {
		// The PS resources are in set 2 of this layout, which only the patched module says.
		EXIT_IF(pixel->gpl_module == nullptr);
		parts.fragment_shader.stage.module = pixel->gpl_module;
	}
	vk::GraphicsPipelineCreateInfo pipeline_info {};
	AssembleMonolithicInfo(pipeline_info, parts, mesh, with_depth, pipeline.pipeline_layout);
	if (const auto observer = GetGraphicsCreateInfoObserver()) {
		observer(key, stages, pipeline_info);
	}

	const auto result = graphics.device.createGraphicsPipelines(driver_cache, 1, &pipeline_info,
	                                                            nullptr, &pipeline.pipeline);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);

	graphics.device.destroyShaderModule(helpers.tess_control, nullptr);
	graphics.device.destroyShaderModule(helpers.tess_eval, nullptr);
	graphics.device.destroyShaderModule(helpers.task, nullptr);
}

void CreatePipelineInternal(GraphicContext& graphics, DescriptorSetLayoutCache& set_layouts,
                            PipelineCache::Pipeline& pipeline,
                            const GraphicsPipelineKey&                 key,
                            const std::array<const ShaderModule*, 4>& stages,
                            vk::PipelineCache                          driver_cache,
                            GraphicsLibraryCache*                      libraries) {
	const auto& vertex = *stages[VERTEX_STAGE_INDEX];
	const auto* pixel  = stages[PIXEL_STAGE_INDEX];
	// Times the whole build, including the rect-list shaders and the helper modules, since all of
	// it is spent before the pipeline can be used. Warm-up and worker threads record it too,
	// tagged so they are not counted as GPU thread waiting. The libraries are got or built inside the
	// timer, so the row also covers the waits for other threads' libraries and the link.
	Common::FirstRunReport::CompileScope compile_timer(
	    Common::FirstRunReport::Kind::Graphics, vertex.id, pixel != nullptr ? pixel->id : 0,
	    GuestGpu::IsGpuThread());
	bool layouts_ready = false;
	if (TryLinkFromLibraries(graphics, set_layouts, libraries, pipeline, key, stages, driver_cache,
	                         layouts_ready)) {
		return;
	}
	CreateWholePipeline(graphics, set_layouts, pipeline, key, stages, driver_cache, layouts_ready);
}

void CreatePipelineInternal(GraphicContext& graphics, PipelineCache::Pipeline& pipeline,
                            const ShaderModule& module, vk::PipelineCache driver_cache) {
	// Times the layouts as well as the driver call, since both are spent before the pipeline works.
	Common::FirstRunReport::CompileScope compile_timer(
	    Common::FirstRunReport::Kind::Compute, module.id, 0, GuestGpu::IsGpuThread());
	vk::PipelineShaderStageCreateInfo                     comp_shader_stage_info {};
	vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
	comp_shader_stage_info.stage  = vk::ShaderStageFlagBits::eCompute;
	comp_shader_stage_info.module = module.module;
	comp_shader_stage_info.pName  = "main";
	const auto wave_size = module.info.wave_size;
	if (graphics.compute_subgroup_size_control_enabled &&
	    wave_size >= graphics.min_subgroup_size && wave_size <= graphics.max_subgroup_size) {
		comp_subgroup_size.requiredSubgroupSize = wave_size;
		comp_shader_stage_info.pNext            = &comp_subgroup_size;
	}

	std::vector<vk::DescriptorSetLayoutBinding> descriptor_bindings;
	AddLayoutBindings(descriptor_bindings, module.info, vk::ShaderStageFlagBits::eCompute);
	pipeline.uses_image_heap = module.info.info.uses_image_heap;
	pipeline.push_stages     = vk::ShaderStageFlagBits::eCompute;
	CreateLayouts(graphics, pipeline, descriptor_bindings);

	vk::ComputePipelineCreateInfo info {};
	if (module.info.info.uses_gds) {
		info.flags = vk::PipelineCreateFlagBits::eDispatchBase;
	}
	info.stage  = comp_shader_stage_info;
	info.layout = pipeline.pipeline_layout;

	const auto result =
	    graphics.device.createComputePipelines(driver_cache, 1, &info, nullptr, &pipeline.pipeline);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess);
}

} // namespace Libs::Graphics
