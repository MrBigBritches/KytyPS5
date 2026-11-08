#include "graphics/host_gpu/renderer/pipeline/pipelineParts.h"

#include <xxhash.h>

#include <algorithm>
#include <atomic>
#include <cstring>

namespace Libs::Graphics {

namespace {

std::atomic<GraphicsCreateInfoObserver> g_create_info_observer {nullptr};

// Whether two part keys hold the same bytes. Part keys have no padding, so this is value equality.
template <class PartKey>
bool SameBytes(const PartKey& left, const PartKey& right) {
	return std::memcmp(&left, &right, sizeof(PartKey)) == 0;
}

} // namespace

bool VertexInputKey::operator==(const VertexInputKey& other) const {
	return SameBytes(*this, other);
}
bool PreRasterKey::operator==(const PreRasterKey& other) const {
	return SameBytes(*this, other);
}
bool FragmentKey::operator==(const FragmentKey& other) const {
	return SameBytes(*this, other);
}
bool FragmentOutputKey::operator==(const FragmentOutputKey& other) const {
	return SameBytes(*this, other);
}

std::size_t HashPartKeyBytes(const void* bytes, std::size_t size) {
	return XXH3_64bits(bytes, size);
}

// Copies the vertex input state and the VS formats of the used attributes into the key.
VertexInputKey MakeVertexInputKey(const GraphicsPipelineKey&                 key,
                                  const std::array<const ShaderModule*, 4>& stages) {
	VertexInputKey result;
	result.vertex_input             = key.vertex_input;
	result.topology                 = key.static_params.topology;
	result.primitive_restart_enable = key.static_params.primitive_restart_enable;
	// The pipeline reads one format per attribute from the VS module, so the formats are part of
	// the key and the VS id is not. Two different VS modules can share a library if they agree.
	const auto* vertex = stages[VERTEX_STAGE_INDEX];
	if (vertex != nullptr) {
		const auto used_attributes = std::min<std::size_t>(
		    key.vertex_input.attribute_count, vertex->vertex_formats.size());
		for (std::size_t index = 0; index < used_attributes; index++) {
			result.attribute_formats[index] = vertex->vertex_formats[index];
		}
	}
	return result;
}

// Takes the stage ids and the raster fields that the pre-rasterization state is made from.
PreRasterKey MakePreRasterKey(const GraphicsPipelineKey& key,
                              const std::array<const ShaderModule*, 4>& /*stages*/) {
	PreRasterKey result;
	// The key's ids are the ids of the modules in `stages`, which is how the stages are found.
	for (std::size_t index = 0; index < PRE_RASTER_STAGE_COUNT; index++) {
		result.stage_ids[index] = key.stage_ids[index];
	}
	result.negative_one_to_one  = key.static_params.negative_one_to_one;
	result.depth_clip_enable    = key.static_params.depth_clip_enable;
	result.provoking_vtx_last   = key.static_params.provoking_vtx_last;
	result.polygon_mode         = key.static_params.polygon_mode;
	result.patch_control_points = key.static_params.patch_control_points;
	return result;
}

// Takes the PS id and the multisample fields that the fragment shader state is made from.
FragmentKey MakeFragmentKey(const GraphicsPipelineKey& key,
                            const std::array<const ShaderModule*, 4>& /*stages*/) {
	FragmentKey result;
	result.pixel_id                 = key.stage_ids[PIXEL_STAGE_INDEX];
	result.samples                  = key.static_params.samples;
	result.sample_shading_enable    = key.static_params.sample_shading_enable;
	result.alpha_to_coverage_enable = key.static_params.alpha_to_coverage_enable;
	return result;
}

// Takes the formats, blend arrays and multisample fields that the fragment output is made from.
FragmentOutputKey MakeFragmentOutputKey(const GraphicsPipelineKey& key,
                                        const std::array<const ShaderModule*, 4>& /*stages*/) {
	FragmentOutputKey result;
	const auto&       params = key.static_params;
	result.rendering                = key.rendering;
	result.samples                  = params.samples;
	result.sample_shading_enable    = params.sample_shading_enable;
	result.alpha_to_coverage_enable = params.alpha_to_coverage_enable;
	std::memcpy(result.color_mask, params.color_mask, sizeof(result.color_mask));
	std::memcpy(result.color_srcblend, params.color_srcblend, sizeof(result.color_srcblend));
	std::memcpy(result.color_comb_fcn, params.color_comb_fcn, sizeof(result.color_comb_fcn));
	std::memcpy(result.color_destblend, params.color_destblend, sizeof(result.color_destblend));
	std::memcpy(result.alpha_srcblend, params.alpha_srcblend, sizeof(result.alpha_srcblend));
	std::memcpy(result.alpha_comb_fcn, params.alpha_comb_fcn, sizeof(result.alpha_comb_fcn));
	std::memcpy(result.alpha_destblend, params.alpha_destblend, sizeof(result.alpha_destblend));
	std::memcpy(result.separate_alpha_blend, params.separate_alpha_blend,
	            sizeof(result.separate_alpha_blend));
	std::memcpy(result.blend_enable, params.blend_enable, sizeof(result.blend_enable));
	return result;
}

bool GplEligible(const GraphicsPipelineKey&                 key,
                 const std::array<const ShaderModule*, 4>& stages) {
	const auto* vertex = stages[VERTEX_STAGE_INDEX];
	if (vertex == nullptr) {
		return false;
	}
	// Patch points mean tessellation. A rectangle list is that with no HS, because its TCS and TES
	// are made from the PS, so this one test covers both.
	const bool patched         = key.static_params.patch_control_points != 0;
	const bool has_hull_domain = stages[1] != nullptr || stages[2] != nullptr;
	const bool mesh            = vertex->info.stage == ShaderType::Mesh || vertex->mesh_task;
	return !patched && !has_hull_domain && !mesh;
}

void SetGraphicsCreateInfoObserver(GraphicsCreateInfoObserver observer) {
	g_create_info_observer.store(observer, std::memory_order_release);
}

GraphicsCreateInfoObserver GetGraphicsCreateInfoObserver() {
	return g_create_info_observer.load(std::memory_order_acquire);
}

} // namespace Libs::Graphics
