// Tests the part keys that split a graphics pipeline into four libraries, and the rule that decides
// which pipelines may be built from libraries.

#include "graphics/host_gpu/renderer/pipeline/pipelineParts.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace Libs::Graphics;

// A new field of GraphicsPipelineKey changes this size. Add the field to AllKeyFields below, so that
// the completeness test flips it, and to the part key that holds it, and only then update the size.
static_assert(sizeof(GraphicsPipelineKey) == 516,
              "GraphicsPipelineKey changed. Add the new field to AllKeyFields and to a part key.");

namespace {

// Stops the test with a message when a check fails.
void Check(bool passed, const std::string& what) {
	if (!passed) {
		std::fprintf(stderr, "PipelinePartsTests: failed: %s\n", what.c_str());
		std::abort();
	}
}

// One bit per part key, to say which parts a field belongs to.
enum PartBit : unsigned {
	VertexInputBit    = 1,
	PreRasterBit      = 2,
	FragmentBit       = 4,
	FragmentOutputBit = 8
};

constexpr uint64_t kVertexShaderId = 0x1111;
constexpr uint64_t kPixelShaderId  = 0x2222;
constexpr uint32_t kBaselineBindings   = 2;
constexpr uint32_t kBaselineAttributes = 3;
constexpr uint32_t kBaselineColorCount = 2;

// A pipeline key with its shader modules. The key points at nothing, but the part keys read the
// VS module for its attribute formats, so the two travel together.
struct Pipeline {
	GraphicsPipelineKey key;
	ShaderModule        vertex;
	ShaderModule        hull;
	ShaderModule        evaluation;
	ShaderModule        pixel;

	std::array<const ShaderModule*, 4> Stages() {
		return {&vertex, key.stage_ids[1] != 0 ? &hull : nullptr,
		        key.stage_ids[2] != 0 ? &evaluation : nullptr, &pixel};
	}
};

// A baseline whose every field has a value that a flip changes. Zero filled fields would hide a
// part key that read the wrong array, since every array would look alike.
void FillBaseline(Pipeline& pipeline) {
	auto& key = pipeline.key;
	key.stage_ids[VERTEX_STAGE_INDEX] = kVertexShaderId;
	key.stage_ids[PIXEL_STAGE_INDEX] = kPixelShaderId;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		key.rendering.color_formats[index] = vk::Format::eR8G8B8A8Unorm;
	}
	key.rendering.color_count    = kBaselineColorCount;
	key.rendering.depth_format   = vk::Format::eD32Sfloat;
	key.rendering.stencil_format = vk::Format::eS8Uint;
	key.vertex_input.binding_count   = kBaselineBindings;
	key.vertex_input.attribute_count = kBaselineAttributes;
	for (uint32_t index = 0; index < ShaderVertexInputInfo::RES_MAX; index++) {
		key.vertex_input.bindings[index]   = {16 + index, static_cast<uint8_t>(index & 1)};
		key.vertex_input.attributes[index] = {4 * index, static_cast<uint8_t>(index & 1)};
	}
	auto& params = key.static_params;
	params.topology = vk::PrimitiveTopology::eTriangleList;
	params.samples  = 4;
	for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
		params.color_mask[index]           = 0xf;
		params.color_srcblend[index]       = 1;
		params.color_comb_fcn[index]       = 1;
		params.color_destblend[index]      = 1;
		params.alpha_srcblend[index]       = 1;
		params.alpha_comb_fcn[index]       = 1;
		params.alpha_destblend[index]      = 1;
		params.separate_alpha_blend[index] = true;
		params.blend_enable[index]         = true;
	}
	pipeline.vertex.id = kVertexShaderId;
	pipeline.vertex.info.stage = ShaderType::Vertex;
	pipeline.vertex.vertex_formats = {vk::Format::eR32G32B32Sfloat, vk::Format::eR32G32Sfloat,
	                                  vk::Format::eR8G8B8A8Unorm};
	pipeline.pixel.id = kPixelShaderId;
	pipeline.pixel.info.stage = ShaderType::Pixel;
}

// The four part keys of a pipeline.
struct PartKeys {
	VertexInputKey    vertex_input;
	PreRasterKey      pre_raster;
	FragmentKey       fragment;
	FragmentOutputKey output;
};

PartKeys MakePartKeys(Pipeline& pipeline) {
	const auto stages = pipeline.Stages();
	return {MakeVertexInputKey(pipeline.key, stages), MakePreRasterKey(pipeline.key, stages),
	        MakeFragmentKey(pipeline.key, stages), MakeFragmentOutputKey(pipeline.key, stages)};
}

// The parts whose key differs between two pipelines, as PartBit flags. A part whose key differs
// must also hash differently, otherwise a library cache would find the wrong entry.
unsigned ChangedParts(const PartKeys& before, const PartKeys& after, const std::string& what) {
	unsigned changed = 0;
	const auto compare = [&](auto before_key, auto after_key, PartBit bit) {
		using Key = decltype(before_key);
		const bool equal = before_key == after_key;
		const bool same_hash = PartKeyHash<Key> {}(before_key) == PartKeyHash<Key> {}(after_key);
		Check(equal == same_hash, what + ": equal keys and equal hashes disagree");
		if (!equal) changed |= bit;
	};
	compare(before.vertex_input, after.vertex_input, VertexInputBit);
	compare(before.pre_raster, after.pre_raster, PreRasterBit);
	compare(before.fragment, after.fragment, FragmentBit);
	compare(before.output, after.output, FragmentOutputBit);
	return changed;
}

// Changes a value to a different one. A bool flips, an enum steps to its next value, and a number
// goes up by one, wrapping for the small types.
template <class Value>
void Bump(Value& value) {
	if constexpr (std::is_same_v<Value, bool>) {
		value = !value;
	} else if constexpr (std::is_enum_v<Value>) {
		value = static_cast<Value>(static_cast<std::underlying_type_t<Value>>(value) + 1);
	} else {
		value = static_cast<Value>(value + 1);
	}
}

// One field of GraphicsPipelineKey, or one vertex format of the VS module, the parts that must see
// it, and how many bytes of the key it covers (zero for a vertex format, which is not in the key).
struct Field {
	std::string                    name;
	std::size_t                    key_bytes;
	unsigned                       expected_parts;
	std::function<void(Pipeline&)> flip;
};

// Lists the fields of one array, one entry per element.
template <class Element, class Access>
void AddArray(std::vector<Field>& fields, const std::string& name, std::size_t count,
              unsigned parts, Access access) {
	for (std::size_t index = 0; index < count; index++) {
		fields.push_back({name + "[" + std::to_string(index) + "]", sizeof(Element), parts,
		                  [=](Pipeline& pipeline) { Bump(access(pipeline, index)); }});
	}
}

// Every field of GraphicsPipelineKey, written out by name. It is not made by flipping raw bytes,
// since a key is aligned and a flip of a padding byte would change nothing a part key could see.
// The sizes add up to sizeof(GraphicsPipelineKey), which a test checks.
std::vector<Field> AllKeyFields() {
	std::vector<Field> fields;
	constexpr auto colors   = RENDER_COLOR_ATTACHMENTS_MAX;
	constexpr auto vertices = ShaderVertexInputInfo::RES_MAX;
	AddArray<uint64_t>(fields, "stage_ids", PRE_RASTER_STAGE_COUNT, PreRasterBit,
	                   [](Pipeline& p, size_t i) -> auto& { return p.key.stage_ids[i]; });
	fields.push_back({"stage_ids[PIXEL_STAGE_INDEX]", sizeof(uint64_t), FragmentBit,
	                  [](Pipeline& p) { Bump(p.key.stage_ids[PIXEL_STAGE_INDEX]); }});
	AddArray<vk::Format>(
	    fields, "rendering.color_formats", colors, FragmentOutputBit,
	    [](Pipeline& p, size_t i) -> auto& { return p.key.rendering.color_formats[i]; });
	fields.push_back({"rendering.depth_format", sizeof(vk::Format), FragmentOutputBit,
	                  [](Pipeline& p) { Bump(p.key.rendering.depth_format); }});
	fields.push_back({"rendering.stencil_format", sizeof(vk::Format), FragmentOutputBit,
	                  [](Pipeline& p) { Bump(p.key.rendering.stencil_format); }});
	fields.push_back({"rendering.color_count", sizeof(uint32_t), FragmentOutputBit,
	                  [](Pipeline& p) { Bump(p.key.rendering.color_count); }});
	AddArray<uint32_t>(
	    fields, "vertex_input.bindings.stride", vertices, VertexInputBit,
	    [](Pipeline& p, size_t i) -> auto& { return p.key.vertex_input.bindings[i].stride; });
	AddArray<uint8_t>(
	    fields, "vertex_input.bindings.instance", vertices, VertexInputBit,
	    [](Pipeline& p, size_t i) -> auto& { return p.key.vertex_input.bindings[i].instance; });
	AddArray<uint32_t>(
	    fields, "vertex_input.attributes.offset", vertices, VertexInputBit,
	    [](Pipeline& p, size_t i) -> auto& { return p.key.vertex_input.attributes[i].offset; });
	AddArray<uint8_t>(
	    fields, "vertex_input.attributes.binding", vertices, VertexInputBit,
	    [](Pipeline& p, size_t i) -> auto& { return p.key.vertex_input.attributes[i].binding; });
	fields.push_back({"vertex_input.binding_count", 1, VertexInputBit,
	                  [](Pipeline& p) { Bump(p.key.vertex_input.binding_count); }});
	fields.push_back({"vertex_input.attribute_count", 1, VertexInputBit,
	                  [](Pipeline& p) { Bump(p.key.vertex_input.attribute_count); }});
	return fields;
}

// The fields of PipelineStaticParameters, written out by name.
std::vector<Field> StaticParameterFields() {
	std::vector<Field> fields;
	constexpr auto colors = RENDER_COLOR_ATTACHMENTS_MAX;
	constexpr unsigned multisample_parts = FragmentBit | FragmentOutputBit;
	const auto scalar = [&](const char* name, std::size_t bytes, unsigned parts, auto access) {
		fields.push_back({name, bytes, parts, [=](Pipeline& p) { Bump(access(p)); }});
	};
#define KYTY_STATIC_FIELD(member, bytes, parts) \
	scalar(#member, bytes, parts, [](Pipeline& p) -> auto& { return p.key.static_params.member; })
	KYTY_STATIC_FIELD(negative_one_to_one, 1, PreRasterBit);
	KYTY_STATIC_FIELD(depth_clip_enable, 1, PreRasterBit);
	KYTY_STATIC_FIELD(topology, 4, VertexInputBit);
	KYTY_STATIC_FIELD(patch_control_points, 4, PreRasterBit);
	KYTY_STATIC_FIELD(primitive_restart_enable, 1, VertexInputBit);
	KYTY_STATIC_FIELD(samples, 4, multisample_parts);
	KYTY_STATIC_FIELD(sample_shading_enable, 1, multisample_parts);
	KYTY_STATIC_FIELD(alpha_to_coverage_enable, 1, multisample_parts);
	KYTY_STATIC_FIELD(provoking_vtx_last, 1, PreRasterBit);
	KYTY_STATIC_FIELD(polygon_mode, 4, PreRasterBit);
#undef KYTY_STATIC_FIELD
#define KYTY_BLEND_ARRAY(Element, member) \
	AddArray<Element>(fields, #member, colors, FragmentOutputBit, \
	                  [](Pipeline& p, size_t i) -> auto& { return p.key.static_params.member[i]; })
	KYTY_BLEND_ARRAY(uint32_t, color_mask);
	KYTY_BLEND_ARRAY(uint8_t, color_srcblend);
	KYTY_BLEND_ARRAY(uint8_t, color_comb_fcn);
	KYTY_BLEND_ARRAY(uint8_t, color_destblend);
	KYTY_BLEND_ARRAY(uint8_t, alpha_srcblend);
	KYTY_BLEND_ARRAY(uint8_t, alpha_comb_fcn);
	KYTY_BLEND_ARRAY(uint8_t, alpha_destblend);
	KYTY_BLEND_ARRAY(bool, separate_alpha_blend);
	KYTY_BLEND_ARRAY(bool, blend_enable);
#undef KYTY_BLEND_ARRAY
	return fields;
}

// The used entries of the VS module's attribute formats. They are in no key field, yet they
// reach the pipeline.
std::vector<Field> VertexFormatFields() {
	std::vector<Field> fields;
	AddArray<vk::Format>(fields, "vertex.vertex_formats", kBaselineAttributes, VertexInputBit,
	                     [](Pipeline& p, size_t i) -> auto& { return p.vertex.vertex_formats[i]; });
	for (auto& field: fields) field.key_bytes = 0;
	return fields;
}

// Every field the completeness tests flip: the key fields, the static parameters, and the used
// vertex formats of the VS module.
std::vector<Field> EveryField() {
	auto fields = AllKeyFields();
	for (auto& field: StaticParameterFields()) fields.push_back(std::move(field));
	for (auto& field: VertexFormatFields()) fields.push_back(std::move(field));
	return fields;
}

// What else could make this pass. The listed sizes add up to sizeof(GraphicsPipelineKey) only if
// every field is listed, so a field left out of the list fails here even when the static_assert
// above still holds, for example after a field changed type. The bytes of the list are the only
// thing checked here, so a flip that did nothing is caught by the next test.
void TestFieldListCoversTheKey() {
	std::size_t listed = 0;
	for (const auto& field: EveryField()) listed += field.key_bytes;
	Check(listed == sizeof(GraphicsPipelineKey), "the field list does not cover the key");
}

// Each field of the key, and each used vertex format, is flipped alone from the baseline. Some part
// key must change, since a field that no part key sees would let two pipelines that differ in it
// share a library, which is a lasting wrong pipeline. What else could pass this is a part key that
// changes for every flip, so the next test also checks which parts changed.
void TestEveryFieldReachesAPartKey() {
	Pipeline baseline;
	FillBaseline(baseline);
	const auto baseline_keys = MakePartKeys(baseline);
	for (const auto& field: EveryField()) {
		Pipeline flipped;
		FillBaseline(flipped);
		field.flip(flipped);
		const auto changed = ChangedParts(baseline_keys, MakePartKeys(flipped), field.name);
		Check(changed != 0, "no part key sees " + field.name);
	}
}

// Each field reaches exactly the parts that use it in the Vulkan state, so libraries are shared
// as much as the state allows. Without this a field copied into every part key would pass the
// test above and share nothing.
void TestEveryFieldReachesTheRightParts() {
	Pipeline baseline;
	FillBaseline(baseline);
	const auto baseline_keys = MakePartKeys(baseline);
	for (const auto& field: EveryField()) {
		Pipeline flipped;
		FillBaseline(flipped);
		field.flip(flipped);
		const auto changed = ChangedParts(baseline_keys, MakePartKeys(flipped), field.name);
		Check(changed == field.expected_parts,
		      field.name + " reached parts " + std::to_string(changed) + " and not " +
		          std::to_string(field.expected_parts));
	}
}

// Two pipelines that differ in one thing share the libraries that do not use it. A part key that
// took in an unrelated field would make the unequal part also unequal here. Each case also names
// the one part that must differ, so a pipeline pair that differed everywhere would fail.
void TestSharing() {
	Pipeline base;
	FillBaseline(base);
	const auto base_keys = MakePartKeys(base);
	const auto changed_parts = [&](auto change) {
		Pipeline other;
		FillBaseline(other);
		change(other);
		return ChangedParts(base_keys, MakePartKeys(other), "sharing");
	};
	Check(changed_parts([](Pipeline& p) {
		p.key.static_params.blend_enable[0]   = false;
		p.key.static_params.color_srcblend[0] = 5;
	}) == FragmentOutputBit, "a blend change must change only the fragment output key");
	Check(changed_parts([](Pipeline& p) {
		p.key.static_params.topology = vk::PrimitiveTopology::eTriangleStrip;
	}) == VertexInputBit, "a topology change must change only the vertex input key");
	Check(changed_parts([](Pipeline& p) { p.key.stage_ids[PIXEL_STAGE_INDEX] = 0x3333; }) == FragmentBit,
	      "a PS change must change only the fragment key");
	// The VS id names the shader and not its attribute formats, so a different VS with the same
	// formats shares the vertex input library and cannot share the pre-rasterization one.
	Check(changed_parts([](Pipeline& p) {
		p.key.stage_ids[VERTEX_STAGE_INDEX] = 0x4444;
		p.vertex.id                         = 0x4444;
	}) ==
	          PreRasterBit,
	      "a VS with the same formats must change only the pre-rasterization key");
	Check(changed_parts([](Pipeline& p) {
		p.vertex.vertex_formats[1] = vk::Format::eR16G16Sfloat;
	}) == VertexInputBit, "a VS with other formats must change only the vertex input key");
}

// A table of pipelines and whether they may be built from libraries. The baseline row passes, and
// every other row differs from it in one thing, so a rule that said no to everything would fail
// the baseline and a rule that ignored a thing would pass its row.
void TestGplEligible() {
	struct Row {
		const char*                  name;
		bool                         eligible;
		std::function<void(Pipeline&)> change;
	};
	const Row rows[] = {
	    {"plain pipeline", true, [](Pipeline&) {}},
	    {"rectangle list", false, [](Pipeline& p) { p.key.static_params.patch_control_points = 3; }},
	    {"tessellation", false, [](Pipeline& p) {
		     p.key.static_params.patch_control_points = 3;
		     p.key.stage_ids[1] = 0x5555;
		     p.key.stage_ids[2] = 0x6666;
	     }},
	    {"hull and domain without patch points", false,
	     [](Pipeline& p) { p.key.stage_ids[1] = 0x5555; }},
	    {"mesh", false, [](Pipeline& p) { p.vertex.info.stage = ShaderType::Mesh; }},
	    {"indirect mesh with a task shader", false, [](Pipeline& p) { p.vertex.mesh_task = true; }},
	};
	for (const auto& row: rows) {
		Pipeline pipeline;
		FillBaseline(pipeline);
		row.change(pipeline);
		Check(GplEligible(pipeline.key, pipeline.Stages()) == row.eligible, row.name);
	}
	Pipeline no_vertex;
	FillBaseline(no_vertex);
	auto stages = no_vertex.Stages();
	stages[0] = nullptr;
	Check(!GplEligible(no_vertex.key, stages), "a pipeline without a VS must not be eligible");
	// A pixel-less pipeline, as a depth-only draw, is eligible.
	Pipeline depth_only;
	FillBaseline(depth_only);
	auto depth_stages = depth_only.Stages();
	depth_stages[3] = nullptr;
	Check(GplEligible(depth_only.key, depth_stages), "a depth-only pipeline must be eligible");
}

} // namespace

int main() {
	TestFieldListCoversTheKey();
	TestEveryFieldReachesAPartKey();
	TestEveryFieldReachesTheRightParts();
	TestSharing();
	TestGplEligible();
	std::printf("PipelinePartsTests: ok\n");
	return 0;
}
