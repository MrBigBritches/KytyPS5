// Tests the rule that decides whether --graphics-pipeline-library builds pipelines from libraries.

#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

using Libs::Graphics::GraphicsPipelineLibraryBlocker;
using Libs::Graphics::PipelineLibrarySupport;

namespace {

// Stops the test with a message when a check fails.
void Check(bool passed, const char* what) {
	if (!passed) {
		std::fprintf(stderr, "PipelineLibraryTests: failed: %s\n", what);
		std::abort();
	}
}

// A device and flag that meet every requirement, the baseline each row changes in one field.
constexpr PipelineLibrarySupport AllMet() {
	return {.flag = true, .extensions = true, .feature = true, .fast_linking = true, .independent_interpolation = true};
}

// With every requirement met nothing blocks, and each row that misses exactly one requirement is
// blocked and names that one. A rule that ignored a requirement would let its row through, and one
// that checked the wrong field would name another.
void TestEachRequirementBlocks() {
	Check(GraphicsPipelineLibraryBlocker(AllMet()) == nullptr, "a device meeting every requirement was blocked");
	struct Row {
		bool PipelineLibrarySupport::*field;
		const char*                   needle;
	};
	constexpr Row rows[] = {
	    {&PipelineLibrarySupport::flag, "--graphics-pipeline-library"},
	    {&PipelineLibrarySupport::extensions, "VK_EXT_graphics_pipeline_library"},
	    {&PipelineLibrarySupport::feature, "graphicsPipelineLibrary feature"},
	    {&PipelineLibrarySupport::fast_linking, "FastLinking"},
	    {&PipelineLibrarySupport::independent_interpolation, "IndependentInterpolation"},
	};
	for (const auto& row: rows) {
		auto support        = AllMet();
		support.*row.field  = false;
		const char* blocker = GraphicsPipelineLibraryBlocker(support);
		Check(blocker != nullptr, "a device missing one requirement was not blocked");
		Check(std::strstr(blocker, row.needle) != nullptr, "the blocker named another requirement");
	}
}

} // namespace

int main() {
	TestEachRequirementBlocks();
	std::printf("PipelineLibraryTests: ok\n");
	return 0;
}
