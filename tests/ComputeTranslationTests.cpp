// Tests the pre-scan's input info builder without a device. The GPU checks in
// ShaderRecompilerComputeTests compare it with the runtime for the fields in the static key. These
// cover the fields outside it, which a wrong module would not show until its resources are walked,
// and the dispatch the builder must refuse. Each check that proves a behaviour has a baseline that
// differs only in that behaviour, as the comments say.
#include "graphics/guest_gpu/command_processor/prescanDispatchInputs.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
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

// The header the pre-scan's helper gets is written by EncodeHostHeader, and PipelineCache now
// writes its store's header with it too. Its bytes must be the ones PipelineCache wrote before,
// which is the stamp, the jump table flag, the profile and the shared memory limit through IoAll,
// so a store written by an earlier build still matches. And DecodeHostHeader must read back every
// value. Each value differs from its default, so a decoder that gave defaults fails, and fields
// written in another order fail both checks. The mutation is to write the profile before the jump
// table flag.
void TestHostHeaderRoundTrips() {
	constexpr uint64_t STAMP        = 0x1122334455667788ull;
	constexpr uint32_t CLOCK_PERIOD = 41;
	constexpr uint32_t SHARED_BYTES = 65536;
	StoreGen::HostValues host;
	host.structured_jump_tables         = true;
	host.profile.native_float_min_max   = true;
	host.profile.device_clock_period_ps = CLOCK_PERIOD;
	host.profile.elide_exec_selects     = true;
	host.max_shared_bytes               = SHARED_BYTES;
	const auto bytes = StoreGen::EncodeHostHeader(host, STAMP);

	ShaderRecompiler::IR::Writer earlier;
	auto                         stamp       = STAMP;
	auto                         jump_tables = host.structured_jump_tables;
	auto                         profile     = host.profile;
	auto                         shared      = host.max_shared_bytes;
	ShaderRecompiler::IR::IoAll(earlier, stamp, jump_tables, profile, shared);
	CHECK(bytes == earlier.bytes);

	const auto decoded = StoreGen::DecodeHostHeader(bytes);
	CHECK(decoded.has_value());
	if (decoded) {
		CHECK(decoded->structured_jump_tables && decoded->max_shared_bytes == SHARED_BYTES);
		CHECK(decoded->profile.native_float_min_max && decoded->profile.elide_exec_selects);
		CHECK(decoded->profile.device_clock_period_ps == CLOCK_PERIOD);
		CHECK(!decoded->profile.maintenance8 && !decoded->profile.select_guarded_loads);
	}
}

} // namespace

int main() {
	TestWorkgroupCountsReachInfo();
	TestIndirectAddressReachesInfo();
	TestIndirectThreadDimensionsRefused();
	TestHostHeaderRoundTrips();
	if (g_failures != 0) {
		std::fprintf(stderr, "ComputeTranslationTests: %d checks failed\n", g_failures);
		return 1;
	}
	std::printf("ComputeTranslationTests: all checks passed\n");
	return 0;
}
