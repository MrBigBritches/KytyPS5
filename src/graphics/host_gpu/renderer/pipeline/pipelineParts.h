#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEPARTS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEPARTS_H_

// A graphics pipeline is made of four parts that Vulkan lets a pipeline library build apart. This
// header has the two things each part needs. A part key names every input of the part, so two
// pipelines with equal keys can share one library. A part builder fills the Vulkan state of the
// part in place. The monolithic CreatePipelineInternal composes the four builders, and a library
// build uses the same builders one by one, so the two cannot disagree about the state of a part.

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace Libs::Graphics {

struct GraphicContext;

// The most shader stages a pipeline can have. A draw has up to four modules (VS, HS, TES, PS), and
// the helper stages add a task shader or the two rectangle-list tessellation shaders. Seven is the
// worst case if all of them were present, and no draw has more than four today.
inline constexpr uint32_t PIPELINE_STAGES_MAX = 7;

// Where the stages of a pipeline sit in GraphicsPipelineKey::stage_ids and in the `stages` array.
inline constexpr std::size_t VERTEX_STAGE_INDEX = 0;
inline constexpr std::size_t PIXEL_STAGE_INDEX  = 3;
// The stages before the PS (VS or LS or mesh, HS, TES), which the pre-rasterization part owns.
inline constexpr std::size_t PRE_RASTER_STAGE_COUNT = PIXEL_STAGE_INDEX;

#pragma pack(push, 1)

// Everything the vertex input library depends on. Whole arrays are kept, including the entries
// past the counts, so a stale entry splits a key the same way it splits the pipeline key.
struct alignas(4) VertexInputKey {
	PipelineVertexInputState                                 vertex_input;
	// The host format of each attribute, from the VS module and not from the key.
	std::array<vk::Format, ShaderVertexInputInfo::RES_MAX>   attribute_formats {};
	vk::PrimitiveTopology                                    topology                 = vk::PrimitiveTopology::ePointList;
	bool                                                     primitive_restart_enable = false;
	// Explicit, so that the key has no padding bytes to compare.
	uint8_t                                                  padding[1] = {};

	bool operator==(const VertexInputKey& other) const;
};

// Everything the pre-rasterization library depends on. The stages are named by id, since the
// ids stand for the SPIR-V that a module was made from. The descriptor set layout of the stages is
// left out on purpose. That is only correct once each stage set layout depends on that stage alone
// (independent sets, Step 3). Today the push range and the heap presence are unions over all the
// stages of a pipeline, so a library built from this key must not use the pipeline layout.
struct alignas(8) PreRasterKey {
	// VS or LS (or mesh), HS and TES.
	std::array<uint64_t, PRE_RASTER_STAGE_COUNT> stage_ids {};
	bool                    negative_one_to_one  = false;
	bool                    depth_clip_enable    = true;
	bool                    provoking_vtx_last   = false;
	vk::PolygonMode         polygon_mode         = vk::PolygonMode::eFill;
	uint32_t                patch_control_points = 0;
	// Explicit, so that the key has no padding bytes to compare.
	uint8_t                 padding[5] = {};

	bool operator==(const PreRasterKey& other) const;
};

// Everything the fragment shader library depends on. The attachment formats are not here, since
// fragment shader state uses only the view mask of the rendering info and the formats belong to the
// fragment output library. A depth-stencil state is always passed to a fragment shader library,
// which VUID-VkGraphicsPipelineCreateInfo-renderPass-09035 needs, and the builder passes a value-initialised one.
// The descriptor set layout of the PS is left out on purpose, on the same condition as for
// PreRasterKey. It is only correct once each stage set layout depends on that stage alone.
struct alignas(8) FragmentKey {
	uint64_t   pixel_id                 = 0;
	uint32_t   samples                  = 1;
	bool       sample_shading_enable    = false;
	bool       alpha_to_coverage_enable = false;
	// Explicit, so that the key has no padding bytes to compare.
	uint8_t    padding[2] = {};

	bool operator==(const FragmentKey& other) const;
};

// Everything the fragment output library depends on, which includes every attachment format. The
// multisample fields are in the fragment key too, because both libraries carry a multisample state and they must agree.
struct alignas(4) FragmentOutputKey {
	PipelineRenderingState rendering;
	uint32_t               samples                  = 1;
	bool                   sample_shading_enable    = false;
	bool                   alpha_to_coverage_enable = false;
	uint32_t               color_mask[RENDER_COLOR_ATTACHMENTS_MAX]           = {};
	uint8_t                color_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                color_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                color_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	uint8_t                alpha_srcblend[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                alpha_comb_fcn[RENDER_COLOR_ATTACHMENTS_MAX]       = {};
	uint8_t                alpha_destblend[RENDER_COLOR_ATTACHMENTS_MAX]      = {};
	bool                   separate_alpha_blend[RENDER_COLOR_ATTACHMENTS_MAX] = {};
	bool                   blend_enable[RENDER_COLOR_ATTACHMENTS_MAX]         = {};
	// Explicit, so that the key has no padding bytes to compare.
	uint8_t                padding[2] = {};

	bool operator==(const FragmentOutputKey& other) const;
};

#pragma pack(pop)

// Part keys are compared and hashed as bytes, like the pipeline key.
static_assert(std::has_unique_object_representations_v<VertexInputKey>);
static_assert(std::has_unique_object_representations_v<PreRasterKey>);
static_assert(std::has_unique_object_representations_v<FragmentKey>);
static_assert(std::has_unique_object_representations_v<FragmentOutputKey>);
// The alignment is what the id and format fields need, and a size that is a multiple of it means
// the explicit padding is exactly what is missing.
static_assert(alignof(VertexInputKey) == 4 && sizeof(VertexInputKey) % 4 == 0);
static_assert(alignof(PreRasterKey) == 8 && sizeof(PreRasterKey) % 8 == 0);
static_assert(alignof(FragmentKey) == 8 && sizeof(FragmentKey) % 8 == 0);
static_assert(alignof(FragmentOutputKey) == 4 && sizeof(FragmentOutputKey) % 4 == 0);

// Hashes the bytes of a part key.
[[nodiscard]] std::size_t HashPartKeyBytes(const void* bytes, std::size_t size);

// A hash functor for a part key type, for the unordered maps of a library cache.
template <class PartKey>
struct PartKeyHash {
	std::size_t operator()(const PartKey& part_key) const {
		return HashPartKeyBytes(&part_key, sizeof(part_key));
	}
};

// Stages are indexed like GraphicsPipelineKey::stage_ids in the four functions below.
// Makes the vertex input key, reading the VS module (stages[0]) for its attribute formats.
[[nodiscard]] VertexInputKey   MakeVertexInputKey(const GraphicsPipelineKey&                 key,
                                                  const std::array<const ShaderModule*, 4>& stages);
// Makes the pre-rasterization key from the VS, HS and TES ids and the raster fields.
[[nodiscard]] PreRasterKey     MakePreRasterKey(const GraphicsPipelineKey&                 key,
                                                const std::array<const ShaderModule*, 4>& stages);
// Makes the fragment shader key from the PS id and the multisample fields.
[[nodiscard]] FragmentKey      MakeFragmentKey(const GraphicsPipelineKey&                 key,
                                               const std::array<const ShaderModule*, 4>& stages);
// Makes the fragment output key from the attachment formats, blend arrays and multisample fields.
[[nodiscard]] FragmentOutputKey MakeFragmentOutputKey(const GraphicsPipelineKey&                 key,
                                                      const std::array<const ShaderModule*, 4>& stages);

// Whether the pipeline can be built from libraries. A rectangle list (tessellation with no HS, so
// the shaders are made from the PS), a tessellation pipeline, and a mesh or task pipeline are
// built whole.
[[nodiscard]] bool GplEligible(const GraphicsPipelineKey&                 key,
                               const std::array<const ShaderModule*, 4>& stages);

// A member of this type makes the struct that holds it non-copyable and non-movable. The part
// structs below point into themselves with pNext and array pointers, so a copy would point into
// the original. This is composition on purpose, not a base class.
struct InPlaceOnly {
	InPlaceOnly()                              = default;
	InPlaceOnly(const InPlaceOnly&)            = delete;
	InPlaceOnly(InPlaceOnly&&)                 = delete;
	InPlaceOnly& operator=(const InPlaceOnly&) = delete;
	InPlaceOnly& operator=(InPlaceOnly&&)      = delete;
};

// Modules that CreatePipelineInternal makes besides the draw's own. A null handle means the
// pipeline has no such stage. The caller owns them and destroys them after the pipeline exists.
struct HelperModules {
	// The task shader of an indirect mesh draw.
	vk::ShaderModule task          = nullptr;
	// The rectangle-list tessellation shaders.
	vk::ShaderModule tess_control  = nullptr;
	vk::ShaderModule tess_eval     = nullptr;
};

// Vertex input and input assembly. A mesh pipeline leaves both out of its create info.
struct VertexInputParts {
	std::array<vk::VertexInputAttributeDescription, ShaderVertexInputInfo::RES_MAX> attributes {};
	std::array<vk::VertexInputBindingDescription, ShaderVertexInputInfo::RES_MAX>   bindings {};
	vk::PipelineVertexInputStateCreateInfo                                          vertex_input {};
	vk::PipelineInputAssemblyStateCreateInfo                                        input_assembly {};
	InPlaceOnly                                                                     in_place;
};

// Shader stages of the pre-rasterization part, then viewport, rasterizer and tessellation state.
struct PreRasterParts {
	// VS, HS and TES first, then the helper stages from helper_begin on. A monolithic pipeline
	// puts the PS between the two groups.
	std::array<vk::PipelineShaderStageCreateInfo, PIPELINE_STAGES_MAX> stages {};
	uint32_t                                                           stage_count  = 0;
	uint32_t                                                           helper_begin = 0;
	vk::PipelineViewportDepthClipControlCreateInfoEXT                  depth_clip_control {};
	vk::PipelineViewportStateCreateInfo                                viewport {};
	vk::PipelineRasterizationDepthClipStateCreateInfoEXT               depth_clip {};
	vk::PipelineRasterizationProvokingVertexStateCreateInfoEXT         provoking_vertex {};
	vk::PipelineRasterizationStateCreateInfo                           rasterizer {};
	vk::PipelineTessellationStateCreateInfo                            tessellation {};
	// Points at `tessellation`, or null when the pipeline has no tessellation state.
	const vk::PipelineTessellationStateCreateInfo*                     tessellation_state = nullptr;
	InPlaceOnly                                                        in_place;
};

// The pixel shader stage, multisample state and depth-stencil state.
struct FragmentShaderParts {
	bool                                     has_stage = false;
	vk::PipelineShaderStageCreateInfo        stage {};
	vk::PipelineMultisampleStateCreateInfo   multisampling {};
	// Value-initialised and always passed by a fragment shader library. A monolithic pipeline passes
	// it only when the targets have a depth or stencil format.
	vk::PipelineDepthStencilStateCreateInfo  depth_stencil {};
	InPlaceOnly                              in_place;
};

// Blend state, attachment formats and the multisample state this part shares with the fragment
// shader part. `rendering.pColorAttachmentFormats` points into the pipeline key, so these parts
// must not outlive the key they were filled from.
struct FragmentOutputParts {
	std::array<vk::PipelineColorBlendAttachmentState, RENDER_COLOR_ATTACHMENTS_MAX> attachments {};
	vk::PipelineColorBlendStateCreateInfo                                           blending {};
	vk::PipelineRenderingCreateInfo                                                 rendering {};
	vk::PipelineMultisampleStateCreateInfo                                          multisampling {};
	InPlaceOnly                                                                     in_place;
};

// The dynamic states, which every part and the monolithic pipeline take from the same list.
struct DynamicStateParts {
	std::vector<vk::DynamicState>      states;
	vk::PipelineDynamicStateCreateInfo info {};
	InPlaceOnly                        in_place;
};

// Fills the vertex input and input assembly state from the key and the VS module formats.
void FillVertexInputParts(VertexInputParts& out, const GraphicsPipelineKey& key,
                          const ShaderModule& vertex);
// Fills the pre-rasterization stages and viewport, rasterizer and tessellation state.
// `stages` is indexed like GraphicsPipelineKey::stage_ids.
void FillPreRasterParts(PreRasterParts& out, const GraphicContext& graphics,
                        const GraphicsPipelineKey&                 key,
                        const std::array<const ShaderModule*, 4>& stages,
                        const HelperModules&                       helpers);
// Fills the PS stage and the multisample state, beside a value-initialised depth-stencil state.
// `pixel` may be null.
void FillFragmentShaderParts(FragmentShaderParts& out, const GraphicsPipelineKey& key,
                             const ShaderModule* pixel);
// Fills the blend attachments, blend state, rendering info and multisample state.
void FillFragmentOutputParts(FragmentOutputParts& out, const GraphicsPipelineKey& key);
// Fills the one dynamic state list that every part and the monolithic pipeline share.
void FillDynamicStateParts(DynamicStateParts& out, const GraphicContext& graphics,
                           const GraphicsPipelineKey& key);

// A seam for tests. When set, CreatePipelineInternal calls it with the finished create info of
// each monolithic graphics pipeline just before the driver call, so a test can read the state the
// parts were composed into. `stages` is the draw's modules. Pass null to clear it. Pipelines are
// built on worker threads too, so the observer may be called concurrently and must be thread-safe.
using GraphicsCreateInfoObserver = void (*)(const GraphicsPipelineKey&                 key,
                                            const std::array<const ShaderModule*, 4>& stages,
                                            const vk::GraphicsPipelineCreateInfo&      create_info);
void SetGraphicsCreateInfoObserver(GraphicsCreateInfoObserver observer);
// Null when no observer is set.
[[nodiscard]] GraphicsCreateInfoObserver GetGraphicsCreateInfoObserver();

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PIPELINEPARTS_H_
