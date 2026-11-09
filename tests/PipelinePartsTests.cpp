// Tests the part keys that split a graphics pipeline into four libraries, and the rule that decides
// which pipelines may be built from libraries.

#include "PipelinePartsFixtures.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

using namespace Libs::Graphics;
using namespace PartFixtures;

namespace {

// Stops the test with a message when a check fails.
void Check(bool passed, const std::string& what) {
	if (!passed) {
		std::fprintf(stderr, "PipelinePartsTests: failed: %s\n", what.c_str());
		std::abort();
	}
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

// The reason a pipeline is built whole is named by what makes it ineligible. Each row differs from
// the others in the one thing it names, and the reasons are all different, so a function that said the
// same thing for every pipeline would fail. A mesh pipeline that also has patch points is a mesh, since
// a mesh pipeline has no tessellation.
void TestGplFallbackReason() {
	using Common::FirstRunReport::FallbackKind;
	struct Row {
		const char*                    name;
		FallbackKind                   reason;
		std::function<void(Pipeline&)> change;
	};
	const Row rows[] = {
	    {"rectangle list", FallbackKind::RectList,
	     [](Pipeline& p) { p.key.static_params.patch_control_points = 3; }},
	    {"tessellation", FallbackKind::Tess, [](Pipeline& p) {
		     p.key.static_params.patch_control_points = 3;
		     p.key.stage_ids[1] = 0x5555;
		     p.key.stage_ids[2] = 0x6666;
	     }},
	    {"hull and domain without patch points", FallbackKind::Tess,
	     [](Pipeline& p) { p.key.stage_ids[1] = 0x5555; }},
	    {"mesh", FallbackKind::Mesh, [](Pipeline& p) { p.vertex.info.stage = ShaderType::Mesh; }},
	    {"indirect mesh with a task shader", FallbackKind::Mesh,
	     [](Pipeline& p) { p.vertex.mesh_task = true; }},
	    {"mesh with patch points", FallbackKind::Mesh, [](Pipeline& p) {
		     p.vertex.info.stage = ShaderType::Mesh;
		     p.key.static_params.patch_control_points = 3;
	     }},
	};
	for (const auto& row: rows) {
		Pipeline pipeline;
		FillBaseline(pipeline);
		row.change(pipeline);
		const auto stages = pipeline.Stages();
		Check(!GplEligible(pipeline.key, stages), std::string(row.name) + " must not be eligible");
		Check(GplFallbackReason(pipeline.key, stages) == row.reason,
		      std::string(row.name) + " got another reason");
	}
}

// A module made of a header and the given instructions, to feed MoveDescriptorSet. Each
// instruction is its opcode and operands, and the word count is added here.
std::vector<uint32_t> MakeModule(const std::vector<std::vector<uint32_t>>& instructions) {
	constexpr uint32_t magic = 0x07230203u, version = 0x00010500u, id_bound = 100u, length_shift = 16u;
	std::vector<uint32_t> words {magic, version, 0u, id_bound, 0u};
	for (const auto& instruction: instructions) {
		words.push_back((static_cast<uint32_t>(instruction.size()) << length_shift) | instruction[0]);
		words.insert(words.end(), instruction.begin() + 1, instruction.end());
	}
	return words;
}

// Moving a set rewrites the DescriptorSet decorations with that value, and only those. The module
// has the decorations of two sets, a binding decoration with a set-like value, a constant whose
// literal is the decoration number and the set number, and a decoration of another kind. A walk that
// edited a word because of its value would change the constant or the binding.
void TestMoveDescriptorSet() {
	constexpr uint32_t op_decorate = 71, op_constant = 43, set_decoration = 34, binding_decoration = 33;
	constexpr uint32_t emitted_set = 0, heap_set = 1, pixel_set = 2;
	const auto module = MakeModule({
	    {op_decorate, 10, set_decoration, emitted_set},
	    {op_decorate, 11, binding_decoration, emitted_set},
	    {op_decorate, 12, set_decoration, heap_set},
	    {op_decorate, 13, set_decoration, emitted_set},
	    {op_constant, 1, 14, set_decoration, emitted_set},
	    {op_decorate, 15, 3u},
	});
	const auto moved = MoveDescriptorSet(module, emitted_set, pixel_set);
	std::size_t changed = 0;
	Check(moved.size() == module.size(), "the patched module changed size");
	for (std::size_t word = 0; word < module.size(); ++word) {
		changed += module[word] != moved[word];
	}
	Check(changed == 2, "exactly the two set 0 decorations must change");
	// The header is 5 words, then the four-word decorations start at 5, 9, 13 and 17, and the
	// constant at 21. The set value of a decoration is its last word.
	Check(moved[8] == pixel_set && moved[20] == pixel_set,
	      "the set 0 decorations must become the pixel set");
	Check(moved[16] == heap_set, "the set 1 decoration must stay");
	Check(moved[12] == emitted_set, "a binding decoration with the value of the set must stay");
	Check(moved[25] == emitted_set, "a constant with the value of the set must stay");
	// Moving a set that no decoration has changes nothing.
	Check(MoveDescriptorSet(module, 7, pixel_set) == module, "a set no decoration has changed a word");
	// A module whose last instruction is a set 0 decoration cut one word short has nothing returned,
	// so no half walked copy is used. The same module whole is patched, which is its baseline.
	auto cut = MakeModule({{op_decorate, 10, set_decoration, emitted_set}});
	Check(TryMoveDescriptorSet(cut, emitted_set, pixel_set).has_value(), "a whole module was refused");
	cut.pop_back();
	Check(!TryMoveDescriptorSet(cut, emitted_set, pixel_set).has_value(),
	      "a cut module was returned, patched or not");
	auto no_magic = module;
	no_magic[0] = 0;
	Check(!TryMoveDescriptorSet(no_magic, emitted_set, pixel_set).has_value(),
	      "a module without the SPIR-V magic was walked");
}

// The layouts of the split layout go to the slots the SPIR-V hard-codes. Fake handles tell them
// apart, so a layout in another slot fails without a device.
void TestSplitSetLayouts() {
	const auto handle = [](uintptr_t value) {
		return vk::DescriptorSetLayout(reinterpret_cast<VkDescriptorSetLayout>(value));
	};
	const auto layouts = SplitSetLayouts(handle(0x10), handle(0x20), handle(0x30));
	Check(layouts.size() == SPLIT_SET_COUNT, "the split layout must have three sets");
	Check(layouts[SPLIT_VERTEX_SET] == handle(0x10), "the VS set must be at its slot");
	Check(layouts[ShaderRecompiler::IR::ImageHeapSet] == handle(0x20),
	      "the image heap layout must be at the heap set the SPIR-V names");
	Check(layouts[SPLIT_PIXEL_SET] == handle(0x30), "the PS set must be at its slot");
}

} // namespace

int main() {
	TestFieldListCoversTheKey();
	TestEveryFieldReachesAPartKey();
	TestEveryFieldReachesTheRightParts();
	TestSharing();
	TestGplEligible();
	TestGplFallbackReason();
	TestMoveDescriptorSet();
	TestSplitSetLayouts();
	std::printf("PipelinePartsTests: ok\n");
	return 0;
}
