// Tests the pre-scan's input info builder without a device. The GPU checks in
// ShaderRecompilerComputeTests compare it with the runtime for the fields in the static key. These
// cover the fields outside it, which a wrong module would not show until its resources are walked,
// and the dispatch the builder must refuse. Each check that proves a behaviour has a baseline that
// differs only in that behaviour, as the comments say.
#include "graphics/guest_gpu/command_processor/prescanDispatchInputs.h"
#include "graphics/shader/storegen/computeTranslation.h"

#include <array>
#include <cstdint>
#include <cstdio>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

constexpr uint64_t PROGRAM_ADDRESS     = 0x0000004567890100ull;
constexpr uint64_t ARGS_ADDRESS        = 0x0000000123456780ull;
constexpr uint32_t THREADS_X           = 64;
constexpr uint32_t WAVE_SIZE           = 64;
constexpr uint32_t SCRATCH_DWORDS      = 0;
constexpr uint32_t HOST_SUBGROUP       = 64;
constexpr uint32_t SHARED_LIMIT        = 32768;
constexpr StoreGen::ComputeKeyDevice DEVICE {HOST_SUBGROUP, SHARED_LIMIT};
// One value per axis, so a copy that dropped or swapped an axis shows.
constexpr std::array<uint32_t, 3> GROUP_COUNTS {3, 5, 7};

// A dispatch of one program as the scanner records it, with the given launch bits.
ScannedDispatch MakeDispatch(bool indirect, bool thread_dimensions) {
	ScannedDispatch dispatch;
	dispatch.program_address        = PROGRAM_ADDRESS;
	dispatch.regs.cs_regs.data_addr = PROGRAM_ADDRESS;
	dispatch.regs.cs_regs.num_thread_x = THREADS_X;
	dispatch.regs.cs_regs.num_thread_y = 1;
	dispatch.regs.cs_regs.num_thread_z = 1;
	dispatch.regs.cs_regs.wave_size    = WAVE_SIZE;
	dispatch.indirect                  = indirect;
	dispatch.thread_dimensions         = thread_dimensions;
	return dispatch;
}

// A direct dispatch's group counts reach the info, which hands them to the resource walk. The info
// starts with zero counts, so a builder or adapter that left them out fails.
void TestWorkgroupCountsReachInfo() {
	auto dispatch             = MakeDispatch(false, false);
	dispatch.workgroup_counts = GROUP_COUNTS;
	const auto info = StoreGen::MakePrescanInputInfo(PrescanInputsOf(dispatch), SCRATCH_DWORDS, DEVICE);
	CHECK(info.has_value());
	if (!info) {
		return;
	}
	for (size_t axis = 0; axis < GROUP_COUNTS.size(); ++axis) {
		CHECK(info->workgroup_counts[axis] == GROUP_COUNTS[axis]);
	}
	CHECK(info->indirect_args_address == 0);
}

// An indirect dispatch's argument address reaches the info, which the resource walk reads group
// counts through. The info starts with address 0, so a builder or adapter that left it out fails.
void TestIndirectAddressReachesInfo() {
	auto dispatch                  = MakeDispatch(true, false);
	dispatch.indirect_args_address = ARGS_ADDRESS;
	const auto info = StoreGen::MakePrescanInputInfo(PrescanInputsOf(dispatch), SCRATCH_DWORDS, DEVICE);
	CHECK(info.has_value());
	if (info) {
		CHECK(info->indirect_args_address == ARGS_ADDRESS);
	}
}

// An indirect dispatch in thread dimensions gives no info, since the command processor dispatches
// it as a direct one with counts it reads from memory. The baselines are the same dispatch with
// either bit alone, which both give an info, so a builder that refused everything fails.
void TestIndirectThreadDimensionsRefused() {
	const auto info_of = [](bool indirect, bool thread_dimensions) {
		return StoreGen::MakePrescanInputInfo(PrescanInputsOf(MakeDispatch(indirect, thread_dimensions)),
		                                      SCRATCH_DWORDS, DEVICE);
	};
	CHECK(!info_of(true, true).has_value());
	CHECK(info_of(true, false).has_value());
	CHECK(info_of(false, true).has_value());
}

} // namespace

int main() {
	TestWorkgroupCountsReachInfo();
	TestIndirectAddressReachesInfo();
	TestIndirectThreadDimensionsRefused();
	if (g_failures != 0) {
		std::fprintf(stderr, "ComputeTranslationTests: %d checks failed\n", g_failures);
		return 1;
	}
	std::printf("ComputeTranslationTests: all checks passed\n");
	return 0;
}
