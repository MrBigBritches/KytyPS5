// Tests the submit-time PM4 walk that finds compute dispatches. The walk reads indirect buffers through
// a reader the test supplies, so no guest memory or device is needed. Each check that proves a
// behaviour has a baseline that differs only in that behaviour, as the comments say.
#include "graphics/guest_gpu/command_processor/scanJobQueue.h"
#include "graphics/guest_gpu/command_processor/scanOrderGate.h"
#include "graphics/guest_gpu/command_processor/scanScratch.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/guest_gpu/pm4.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <numeric>
#include <new>
#include <set>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

// Counts the allocations made while it is switched on, from every thread, by the operator new below.
std::atomic<bool>     g_count_allocations {false};
std::atomic<uint64_t> g_allocation_count {0};

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Stream = std::vector<uint32_t>;

constexpr uint64_t PROGRAM_ADDRESS = 0x0000004567890100ull;
constexpr uint32_t WAVE32_BIT      = 1u << 15u;
constexpr uint32_t USER_SGPRS      = 2;
constexpr uint32_t RSRC2_USER_SHIFT = 1;
constexpr uint32_t RSRC2_TGID_X_EN  = 1u << 7u;
// The shader type of a PM4 header sits in bits 1 and 2, and compute is 1.
constexpr uint32_t HEADER_SHADER_TYPE_COMPUTE = 1u << 1u;
constexpr uint32_t DISPATCH_PACKET_DWORDS     = 5;
// The type 2 filler word, which the walk steps over.
constexpr uint32_t FILLER_WORD = 0x80000000u;

// Guest memory made of a few regions, which fails every read that is not inside one.
struct FakeMemory {
	std::map<uint64_t, Stream>   regions;
	mutable uint32_t             reads = 0;
	mutable std::vector<uint64_t> read_addresses;

	ScanMemoryReader Reader() const {
		return [this](uint64_t address, void* destination, uint64_t bytes) {
			++reads;
			read_addresses.push_back(address);
			for (const auto& [base, words]: regions) {
				const uint64_t size = words.size() * sizeof(uint32_t);
				if (address >= base && address + bytes <= base + size) {
					std::memcpy(destination,
					            reinterpret_cast<const uint8_t*>(words.data()) + (address - base), bytes);
					return true;
				}
			}
			return false;
		};
	}

	// Whether any read began at the address.
	bool WasRead(uint64_t address) const {
		return std::find(read_addresses.begin(), read_addresses.end(), address) != read_addresses.end();
	}
};

void Append(Stream& stream, const Stream& more) {
	stream.insert(stream.end(), more.begin(), more.end());
}

// A SET_SH_REG packet writing consecutive registers from `reg`.
Stream SetSh(uint32_t reg, const Stream& values) {
	Stream packet {KYTY_PM4(2 + values.size(), Pm4::IT_SET_SH_REG, 0), reg};
	Append(packet, values);
	return packet;
}

// The packets that make a compute program current, as a game's shader setup does. The program reads
// `user_sgprs` user data registers.
Stream SetProgram(uint64_t address = PROGRAM_ADDRESS, uint32_t user_sgprs = USER_SGPRS) {
	Stream stream;
	Append(stream, SetSh(Pm4::COMPUTE_PGM_LO, {static_cast<uint32_t>(address >> 8u),
	                                           static_cast<uint32_t>(address >> 40u)}));
	Append(stream, SetSh(Pm4::COMPUTE_PGM_RSRC1,
	                     {0u, (user_sgprs << RSRC2_USER_SHIFT) | RSRC2_TGID_X_EN}));
	Append(stream, SetSh(Pm4::COMPUTE_NUM_THREAD_X, {64u, 1u, 1u}));
	return stream;
}

Stream Dispatch(uint32_t x, uint32_t y, uint32_t z, uint32_t mode) {
	return {KYTY_PM4(5, Pm4::IT_DISPATCH_DIRECT, 0), x, y, z, mode};
}

Stream IndirectBuffer(uint64_t address, uint32_t dwords, bool chain = false) {
	return {KYTY_PM4(4, Pm4::IT_INDIRECT_BUFFER, 0), static_cast<uint32_t>(address),
	        static_cast<uint32_t>(address >> 32u), dwords | (chain ? 1u << 20u : 0u)};
}

// A COND_EXEC that may skip the `skipped` dwords after it.
Stream CondExec(uint32_t skipped) {
	return {KYTY_PM4(5, Pm4::IT_COND_EXEC, 0), 0x1000u, 0u, 0u, skipped};
}

// A BRANCH packet. Payload word 0 is the mode, 1 to 6 are the compare fields the walk ignores, then
// come the then buffer (address low and high, size) and the else buffer.
Stream BranchSized(uint32_t mode, uint64_t then_address, uint32_t then_dwords, uint64_t else_address,
                   uint32_t else_dwords) {
	return {KYTY_PM4(14, Pm4::IT_INDIRECT_BUFFER, 0), mode, 0u, 0u, 0u, 0u, 0u, 0u,
	        static_cast<uint32_t>(then_address), static_cast<uint32_t>(then_address >> 32u), then_dwords,
	        static_cast<uint32_t>(else_address), static_cast<uint32_t>(else_address >> 32u), else_dwords};
}

// A BRANCH whose two buffers have the same size.
Stream Branch(uint32_t mode, uint64_t then_address, uint64_t else_address, uint32_t dwords) {
	return BranchSized(mode, then_address, dwords, else_address, dwords);
}

// An IT_SET_BASE with the given shader type bits, base index and address words.
Stream SetBase(uint32_t shader_type_bits, uint32_t index, uint32_t low, uint32_t high) {
	return {KYTY_PM4(4, Pm4::IT_SET_BASE, 0) | shader_type_bits, index, low, high};
}

// A DISPATCH_INDIRECT in its base form, which carries only an offset from the tracked base.
Stream DispatchIndirectBase(uint32_t offset) {
	return {KYTY_PM4(3, Pm4::IT_DISPATCH_INDIRECT, 0), offset, 0u};
}

struct ScanResult {
	std::vector<ScannedDispatch> dispatches;
	ScanStats                    stats;
};

ScanResult Scan(const Stream& commands, const FakeMemory& memory, CsShadow& shadow,
                ScanLimits limits = {}, bool async = false) {
	ScanResult result;
	SubmitScanner(memory.Reader(), limits).Scan(commands, 3, async, shadow, result.dispatches, result.stats);
	return result;
}

// The dispatch sits behind an indirect buffer. The baseline is the same dispatch with the buffer
// unreadable (TestUnreadableBufferSkipped), which finds none, so the dispatch here is found by
// following the buffer and not by anything in the top-level list.
void TestDispatchBehindIndirectBuffer() {
	FakeMemory memory;
	memory.regions[0x10000] = Dispatch(8, 4, 2, 0);
	Stream commands         = SetProgram();
	Append(commands, SetSh(Pm4::COMPUTE_USER_DATA_0, {0xAAAAu, 0xBBBBu}));
	Append(commands, IndirectBuffer(0x10000, 5));
	CsShadow shadow;
	const auto result = Scan(commands, memory, shadow);
	CHECK(result.dispatches.size() == 1);
	CHECK(result.stats.ib_followed == 1);
	if (result.dispatches.empty()) {
		return;
	}
	const auto& dispatch = result.dispatches[0];
	CHECK(dispatch.program_address == PROGRAM_ADDRESS);
	CHECK(dispatch.regs.cs_regs.user_sgpr == USER_SGPRS);
	CHECK(dispatch.regs.cs_user_sgpr.value[0] == 0xAAAAu && dispatch.regs.cs_user_sgpr.value[1] == 0xBBBBu);
	CHECK(dispatch.workgroup_counts[0] == 8 && dispatch.workgroup_counts[1] == 4 &&
	      dispatch.workgroup_counts[2] == 2);
	CHECK(dispatch.queue == 3 && !dispatch.async_queue);
}

// Wave size comes from CS_W32_EN of the initiator. The two streams differ only in that bit, so
// anything else that set the wave size (the register default of 64, a stale shadow) shows as the
// 32 case failing, and a walk that always said 32 shows as the 64 case failing.
void TestWaveSizeFromInitiator() {
	FakeMemory memory;
	Stream     wave64 = SetProgram();
	Append(wave64, Dispatch(1, 1, 1, 0));
	Stream wave32 = SetProgram();
	Append(wave32, Dispatch(1, 1, 1, WAVE32_BIT));
	CsShadow shadow64;
	CsShadow shadow32;
	const auto sixty_four = Scan(wave64, memory, shadow64);
	const auto thirty_two = Scan(wave32, memory, shadow32);
	CHECK(sixty_four.dispatches.size() == 1 && thirty_two.dispatches.size() == 1);
	if (sixty_four.dispatches.empty() || thirty_two.dispatches.empty()) {
		return;
	}
	CHECK(sixty_four.dispatches[0].regs.cs_regs.wave_size == 64);
	CHECK(thirty_two.dispatches[0].regs.cs_regs.wave_size == 32);
	CHECK(sixty_four.dispatches[0].key_hash != thirty_two.dispatches[0].key_hash);
	CHECK(sixty_four.dispatches[0].program_address == thirty_two.dispatches[0].program_address);
	// The initiator is kept whole, so the runtime's other bit readers see what the game sent.
	CHECK(thirty_two.dispatches[0].initiator == WAVE32_BIT);
}

// How many user data registers the program reads is part of its key, since the runtime's module
// differs by it. The two streams differ only in that count, and the control stream sets different user
// data values under the same count, which must not move the key.
void TestUserDataCountSeparatesKeys() {
	constexpr uint32_t MORE_SGPRS = USER_SGPRS + 1;
	FakeMemory         memory;
	const auto         key_of = [&](uint32_t user_sgprs, uint32_t user_value) {
        Stream stream = SetProgram(PROGRAM_ADDRESS, user_sgprs);
        Append(stream, SetSh(Pm4::COMPUTE_USER_DATA_0, {user_value, user_value}));
        Append(stream, Dispatch(1, 1, 1, 0));
        CsShadow   shadow;
        const auto result = Scan(stream, memory, shadow);
        CHECK(result.dispatches.size() == 1);
        return result.dispatches.empty() ? uint64_t {0} : result.dispatches[0].key_hash;
	};
	CHECK(key_of(USER_SGPRS, 1) != key_of(MORE_SGPRS, 1));
	CHECK(key_of(USER_SGPRS, 1) == key_of(USER_SGPRS, 2));
}

// A dispatch in thread dimensions mode turns threads into groups the way the runtime does, and the
// same dispatch without the bit keeps the raw counts, which is the baseline.
void TestThreadDimensionsRoundUp() {
	FakeMemory memory;
	Stream     threads = SetProgram();
	Append(threads, Dispatch(130, 1, 1, Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS));
	Stream groups = SetProgram();
	Append(groups, Dispatch(130, 1, 1, 0));
	CsShadow shadow_threads;
	CsShadow shadow_groups;
	const auto by_threads = Scan(threads, memory, shadow_threads);
	const auto by_groups  = Scan(groups, memory, shadow_groups);
	CHECK(by_threads.dispatches.size() == 1 && by_groups.dispatches.size() == 1);
	if (by_threads.dispatches.empty() || by_groups.dispatches.empty()) {
		return;
	}
	CHECK(by_threads.dispatches[0].thread_dimensions && !by_groups.dispatches[0].thread_dimensions);
	CHECK(by_threads.dispatches[0].workgroup_counts[0] == 3);
	CHECK(by_groups.dispatches[0].workgroup_counts[0] == 130);
}

// An indirect buffer at an address nobody maps is skipped and counted, the packets after it are
// still walked, and the reader is the only thing that touched it.
void TestUnreadableBufferSkipped() {
	FakeMemory memory;
	Stream     commands = SetProgram();
	Append(commands, IndirectBuffer(0xdead0000ull, 64));
	Append(commands, Dispatch(2, 2, 2, 0));
	CsShadow shadow;
	const auto result = Scan(commands, memory, shadow);
	CHECK(result.stats.ib_unreadable == 1);
	CHECK(result.stats.ib_followed == 0);
	CHECK(result.dispatches.size() == 1);
	CHECK(memory.reads == 1);
	// The buffer may have set the program, so the dispatch after it carries the doubt.
	CHECK(!result.dispatches.empty() && result.dispatches[0].uncertain_program);
	// The same buffer made readable, holding a dispatch, adds one, so only the readable one was walked.
	memory.regions[0xdead0000ull] = Stream(59, 0x80000000u);
	Append(memory.regions[0xdead0000ull], Dispatch(9, 9, 9, 0));
	CsShadow shadow_readable;
	const auto readable = Scan(commands, memory, shadow_readable);
	CHECK(readable.dispatches.size() == 2 && readable.stats.ib_unreadable == 0);
	CHECK(readable.dispatches.size() == 2 && !readable.dispatches[1].uncertain_program);
}

// How many unreadable buffers a walk counted, and whether it left the program in doubt.
using Doubt = std::pair<uint64_t, bool>;

// A null address and a misaligned one both name a buffer that cannot be read, so each is skipped,
// counted and leaves the program uncertain. An empty buffer at the same address is the baseline, which
// the processor does not follow, so it is neither counted nor a doubt.
void TestBadBufferAddressIsUncertain() {
	constexpr uint64_t MISALIGNED = 0x10002;
	FakeMemory         memory;
	const auto         doubt_of = [&](uint64_t address, uint32_t dwords) {
        Stream stream = SetProgram();
        Append(stream, IndirectBuffer(address, dwords));
        CsShadow   shadow;
        const auto result = Scan(stream, memory, shadow);
        return Doubt(result.stats.ib_unreadable, !shadow.certain);
	};
	CHECK(doubt_of(0, DISPATCH_PACKET_DWORDS) == Doubt(1, true));
	CHECK(doubt_of(MISALIGNED, DISPATCH_PACKET_DWORDS) == Doubt(1, true));
	CHECK(doubt_of(0, 0) == Doubt(0, false));
}

// A packet the processor would not run leaves the program uncertain, and a packet cut off at the end
// of a buffer that was read short of its size is the budget's doing, so it is not a bad packet. The two
// streams hold the same words and differ only in whether the buffer was read short.
void TestBadPacketIsUncertain() {
	constexpr uint32_t BAD_HEADER = 0x12345678u;
	constexpr uint32_t FILLERS    = 62;
	constexpr uint32_t CUT_AT     = 2;
	FakeMemory         memory;
	Stream             garbage = SetProgram();
	garbage.push_back(BAD_HEADER);
	CsShadow   bad_shadow;
	const auto bad = Scan(garbage, memory, bad_shadow);
	CHECK(bad.stats.bad_packets == 1 && !bad_shadow.certain);
	CsShadow clean_shadow;
	(void)Scan(SetProgram(), memory, clean_shadow);
	CHECK(clean_shadow.certain);

	// A buffer of fillers and then the start of a dispatch. Read whole at its own size, the dispatch is
	// cut off by the buffer, which is a bad packet. Read short, it is cut off by the budget.
	Stream buffer(FILLERS, FILLER_WORD);
	const Stream whole_dispatch = Dispatch(1, 1, 1, 0);
	Append(buffer, Stream(whole_dispatch.begin(), whole_dispatch.begin() + CUT_AT));
	memory.regions[0x50000] = buffer;
	Stream whole = SetProgram();
	Append(whole, IndirectBuffer(0x50000, static_cast<uint32_t>(buffer.size())));
	CsShadow   whole_shadow;
	const auto read_whole = Scan(whole, memory, whole_shadow);
	CHECK(read_whole.stats.bad_packets == 1 && read_whole.stats.budget_exhausted == 0);

	Stream     claimed = SetProgram();
	const auto claimed_dwords = static_cast<uint32_t>(buffer.size()) + DISPATCH_PACKET_DWORDS - CUT_AT;
	Append(claimed, IndirectBuffer(0x50000, claimed_dwords));
	memory.regions[0x50000] = buffer;
	Append(memory.regions[0x50000], Stream(DISPATCH_PACKET_DWORDS - CUT_AT, FILLER_WORD));
	ScanLimits tight;
	tight.dword_budget = static_cast<uint32_t>(claimed.size() + buffer.size());
	CsShadow   short_shadow;
	const auto read_short = Scan(claimed, memory, short_shadow, tight);
	CHECK(read_short.stats.bad_packets == 0 && read_short.stats.budget_exhausted >= 1);
}

// The shadow lives across submissions, so a dispatch that sets no program of its own inherits one,
// and the reset packet clears it. A fresh shadow is the baseline for the inherited case.
void TestShadowPersistsAndResets() {
	FakeMemory memory;
	CsShadow   shadow;
	const auto first = Scan(SetProgram(), memory, shadow);
	CHECK(first.dispatches.empty());
	const auto second = Scan(Dispatch(1, 1, 1, 0), memory, shadow);
	CHECK(second.dispatches.size() == 1 && second.stats.program_inherited == 1);
	CHECK(second.stats.no_shader_state == 0);

	CsShadow fresh;
	const auto none = Scan(Dispatch(1, 1, 1, 0), memory, fresh);
	CHECK(none.dispatches.empty() && none.stats.no_shader_state == 1);

	Stream reset {0xC0001024u, 0u};
	Append(reset, Dispatch(1, 1, 1, 0));
	const auto after_reset = Scan(reset, memory, shadow);
	CHECK(after_reset.dispatches.empty());
	CHECK(after_reset.stats.resets == 1 && after_reset.stats.no_shader_state == 1);
}

// A buffer that chains to itself ends, and the dispatch in it is found once.
void TestSelfReferenceTerminates() {
	FakeMemory memory;
	Stream     buffer = IndirectBuffer(0x20000, 9);
	Append(buffer, Dispatch(1, 1, 1, 0));
	memory.regions[0x20000] = buffer;
	Stream commands         = SetProgram();
	Append(commands, IndirectBuffer(0x20000, 9));
	CsShadow shadow;
	const auto result = Scan(commands, memory, shadow);
	CHECK(result.dispatches.size() == 1);
	CHECK(result.stats.ib_revisited == 1);
	// A buffer that was skipped as a revisit may have held writes, so the program is in doubt.
	CHECK(!shadow.certain);
}

// The walk is linear, so a dispatch inside the span a COND_EXEC may skip and one after it are both
// found. The packet itself changes nothing.
void TestCondExecCoversBothArms() {
	FakeMemory memory;
	Stream     commands = SetProgram();
	Append(commands, {KYTY_PM4(6, Pm4::IT_COND_EXEC, 0), 0x1000u, 0u, 0u, 5u, 0u});
	Append(commands, Dispatch(1, 1, 1, 0));
	Append(commands, Dispatch(2, 2, 2, 0));
	CsShadow shadow;
	const auto result = Scan(commands, memory, shadow);
	CHECK(result.dispatches.size() == 2);
}

// A mode 2 branch with an else buffer names a then and an else buffer, and both are walked. The
// processor always runs one of them and chains to it, so the packets after the branch in the parent are
// not walked. The dispatch placed after it is the baseline for that, and is not found.
void TestBranchWalksBothArms() {
	FakeMemory memory;
	memory.regions[0x30000] = Dispatch(1, 1, 1, 0);
	memory.regions[0x31000] = Dispatch(1, 1, 1, WAVE32_BIT);
	Stream commands = SetProgram();
	Append(commands, Branch(2, 0x30000, 0x31000, 5));
	Append(commands, Dispatch(7, 7, 7, 0));
	CsShadow shadow;
	const auto result = Scan(commands, memory, shadow);
	CHECK(result.dispatches.size() == 2);
	CHECK(result.stats.branch_mode2 == 1 && result.stats.branch_mode2_empty_else == 0);
	// Which arm the processor takes is unknown, so the state after the branch is in doubt.
	CHECK(!shadow.certain);
	if (result.dispatches.size() == 2) {
		CHECK(result.dispatches[0].regs.cs_regs.wave_size != result.dispatches[1].regs.cs_regs.wave_size);
	}
}

// A mode 1 branch whose then arm is not taken, like a mode 2 one with an empty else, falls through to the
// parent's next packets. The then arm sets program B and dispatches, and the tail dispatch inherits. It
// must be found under program A, since the tail only runs when the then arm did not, and be marked
// uncertain. The same stream as a mode 2 branch with an else buffer chains and has no tail dispatch, so
// the tail is the fall through and not something the arms add. The empty else form is checked as well.
void TestBranchFallsThroughToTail() {
	constexpr uint64_t THEN_ADDRESS = 0x30000;
	constexpr uint64_t ELSE_ADDRESS = 0x31000;
	const uint64_t     program_b    = PROGRAM_ADDRESS + 0x10000u;
	FakeMemory         memory;
	Stream             then_arm = SetProgram(program_b);
	Append(then_arm, Dispatch(1, 1, 1, 0));
	memory.regions[THEN_ADDRESS] = then_arm;
	memory.regions[ELSE_ADDRESS] = Stream(DISPATCH_PACKET_DWORDS, FILLER_WORD);
	const auto then_dwords       = static_cast<uint32_t>(then_arm.size());

	const auto run = [&](const Stream& branch) {
		Stream stream = SetProgram();
		Append(stream, branch);
		Append(stream, Dispatch(2, 1, 1, 0));
		CsShadow shadow;
		return Scan(stream, memory, shadow);
	};
	const auto check_fall_through = [&](const ScanResult& result) {
		CHECK(result.dispatches.size() == 2);
		if (result.dispatches.size() != 2) {
			return;
		}
		CHECK(result.dispatches[0].program_address == program_b && !result.dispatches[0].uncertain_program);
		CHECK(result.dispatches[1].program_address == PROGRAM_ADDRESS);
		CHECK(result.dispatches[1].uncertain_program);
	};
	const auto mode_one = run(BranchSized(1, THEN_ADDRESS, then_dwords, ELSE_ADDRESS, DISPATCH_PACKET_DWORDS));
	check_fall_through(mode_one);
	CHECK(mode_one.stats.branch_mode1 == 1);
	const auto empty_else = run(BranchSized(2, THEN_ADDRESS, then_dwords, ELSE_ADDRESS, 0));
	check_fall_through(empty_else);
	CHECK(empty_else.stats.branch_mode2 == 1 && empty_else.stats.branch_mode2_empty_else == 1);

	const auto chained = run(BranchSized(2, THEN_ADDRESS, then_dwords, ELSE_ADDRESS, DISPATCH_PACKET_DWORDS));
	CHECK(chained.dispatches.size() == 1 && chained.dispatches[0].program_address == program_b);
}

// The else arm of a mode 2 branch starts from the state the branch met and its writes are dropped, so
// the then arm's dispatch is found under the parent's program and not under one the else arm set. The
// state after the branch is the then arm's, and the else arm's program never shows.
void TestElseArmWritesAreDropped() {
	constexpr uint64_t THEN_ADDRESS = 0x30000;
	constexpr uint64_t ELSE_ADDRESS = 0x31000;
	const uint64_t     else_program = PROGRAM_ADDRESS + 0x10000u;
	FakeMemory         memory;
	memory.regions[THEN_ADDRESS] = Dispatch(1, 1, 1, 0);
	const Stream else_arm        = SetProgram(else_program);
	memory.regions[ELSE_ADDRESS] = else_arm;
	Stream commands              = SetProgram();
	Append(commands, BranchSized(2, THEN_ADDRESS, DISPATCH_PACKET_DWORDS, ELSE_ADDRESS,
	                             static_cast<uint32_t>(else_arm.size())));
	CsShadow   shadow;
	const auto result = Scan(commands, memory, shadow);
	CHECK(result.dispatches.size() == 1 && result.dispatches[0].program_address == PROGRAM_ADDRESS);
	CHECK(shadow.regs.cs_regs.data_addr == PROGRAM_ADDRESS);
}

// A mode 1 branch runs its then buffer or nothing, so the walk must not read the else buffer even
// when its words are garbage. The streams differ only in the mode, and the mode 2 one, which does run
// an else arm, reads the same buffer, so the missing read is the mode and not a bad address.
void TestBranchModeOneNeverReadsElse() {
	constexpr uint64_t THEN_ADDRESS = 0x30000;
	constexpr uint64_t ELSE_ADDRESS = 0x31000;
	FakeMemory         memory;
	memory.regions[THEN_ADDRESS] = Dispatch(1, 1, 1, 0);
	memory.regions[ELSE_ADDRESS] = Stream(DISPATCH_PACKET_DWORDS, 0xDEADBEEFu);

	Stream mode_one = SetProgram();
	Append(mode_one, Branch(1, THEN_ADDRESS, ELSE_ADDRESS, DISPATCH_PACKET_DWORDS));
	CsShadow   shadow_one;
	const auto one = Scan(mode_one, memory, shadow_one);
	CHECK(one.dispatches.size() == 1 && one.stats.ib_followed == 1);
	CHECK(memory.WasRead(THEN_ADDRESS) && !memory.WasRead(ELSE_ADDRESS));
	CHECK(one.stats.bad_packets == 0);

	memory.read_addresses.clear();
	Stream mode_two = SetProgram();
	Append(mode_two, Branch(2, THEN_ADDRESS, ELSE_ADDRESS, DISPATCH_PACKET_DWORDS));
	CsShadow   shadow_two;
	const auto two = Scan(mode_two, memory, shadow_two);
	CHECK(memory.WasRead(ELSE_ADDRESS) && two.stats.ib_followed == 2 && two.stats.bad_packets == 1);
}

// An indirect register list sets the program from memory the walk reads. The list at an unreadable
// address is the baseline, which leaves no program and so no dispatch.
void TestIndirectRegisterList() {
	FakeMemory memory;
	memory.regions[0x40000] = {Pm4::COMPUTE_PGM_LO, static_cast<uint32_t>(PROGRAM_ADDRESS >> 8u),
	                           Pm4::COMPUTE_PGM_HI, static_cast<uint32_t>(PROGRAM_ADDRESS >> 40u)};
	Stream commands {KYTY_PM4(5, Pm4::IT_SET_SH_REG_INDIRECT, 0), 0x40000u, 0u, 0u, 2u};
	Append(commands, Dispatch(1, 1, 1, 0));
	CsShadow shadow;
	const auto result = Scan(commands, memory, shadow);
	CHECK(result.dispatches.size() == 1 && result.dispatches[0].program_address == PROGRAM_ADDRESS);

	FakeMemory empty;
	CsShadow   shadow_empty;
	const auto missing = Scan(commands, empty, shadow_empty);
	// The list has its own count, apart from indirect buffers, and the unread writes leave a doubt.
	CHECK(missing.dispatches.empty() && missing.stats.no_shader_state == 1 &&
	      missing.stats.register_list_unreadable == 1 && missing.stats.ib_unreadable == 0);
	CHECK(!shadow_empty.certain);
	CHECK(shadow.certain);
}

// The budget bounds what a submission reads. The same buffer under a large budget is the baseline,
// where the dispatch at its end is found.
void TestBudgetBoundsReads() {
	FakeMemory memory;
	Stream     big(200, 0x80000000u);
	Append(big, Dispatch(1, 1, 1, 0));
	memory.regions[0x50000] = big;
	Stream commands         = SetProgram();
	Append(commands, IndirectBuffer(0x50000, static_cast<uint32_t>(big.size())));
	CsShadow wide;
	const auto found = Scan(commands, memory, wide);
	CHECK(found.dispatches.size() == 1 && found.stats.budget_exhausted == 0);
	CHECK(wide.certain);
	CsShadow narrow;
	ScanLimits tight;
	tight.dword_budget = 64;
	const auto bounded = Scan(commands, memory, narrow, tight);
	CHECK(bounded.dispatches.empty() && bounded.stats.budget_exhausted >= 1);
	CHECK(bounded.stats.dwords_read <= 64);
	// The part of the list that was not read may have changed the program.
	CHECK(!narrow.certain);
}

// The async flag of the queue travels with the dispatch and is part of the key hash.
void TestAsyncQueueFlag() {
	FakeMemory memory;
	Stream     commands = SetProgram();
	Append(commands, Dispatch(1, 1, 1, 0));
	CsShadow graphics_shadow;
	CsShadow async_shadow;
	const auto graphics = Scan(commands, memory, graphics_shadow, {}, false);
	const auto async    = Scan(commands, memory, async_shadow, {}, true);
	CHECK(graphics.dispatches.size() == 1 && async.dispatches.size() == 1);
	if (graphics.dispatches.empty() || async.dispatches.empty()) {
		return;
	}
	CHECK(!graphics.dispatches[0].async_queue && async.dispatches[0].async_queue);
	CHECK(graphics.dispatches[0].key_hash != async.dispatches[0].key_hash);
}

// A dispatch behind six nested indirect buffers. The old limit of four misses it and the limit now
// in force finds it, so the dispatch is found by the depth and by nothing else. The walks are the
// same stream, so the only difference is the limit.
void TestDepthSixFoundOnlyBeyondOldLimit() {
	constexpr uint32_t OLD_MAX_DEPTH = 4;
	constexpr uint32_t LEVELS        = 6;
	FakeMemory         memory;
	// Level n calls level n + 1, and the last level holds the dispatch.
	for (uint32_t level = 1; level <= LEVELS; ++level) {
		const uint64_t address = 0x60000u + level * 0x1000u;
		memory.regions[address] =
		    level == LEVELS ? Dispatch(1, 1, 1, 0) : IndirectBuffer(address + 0x1000u, level + 1 == LEVELS ? 5 : 4);
	}
	Stream commands = SetProgram();
	Append(commands, IndirectBuffer(0x61000u, 4));
	CsShadow   current_shadow;
	const auto current = Scan(commands, memory, current_shadow);
	CHECK(current.dispatches.size() == 1);
	CHECK(current.stats.depth_max == LEVELS);
	ScanLimits old_limits;
	old_limits.max_depth = OLD_MAX_DEPTH;
	CsShadow   old_shadow;
	const auto old = Scan(commands, memory, old_shadow, old_limits);
	CHECK(old.dispatches.empty() && old.stats.ib_depth_limited >= 1);
	// The buffer that was not followed may have held writes, which the full limit never leaves.
	CHECK(!old_shadow.certain && current_shadow.certain);
	CHECK(SCAN_MAX_IB_DEPTH > LEVELS);
}

// A long run of chained buffers adds no nesting, as the processor replaces the buffer it is in. The
// same run as calls runs into the limit, which is the baseline.
void TestChainAddsNoDepth() {
	constexpr uint32_t LINKS = SCAN_MAX_IB_DEPTH + 4;
	FakeMemory         chained;
	FakeMemory         called;
	for (uint32_t link = 1; link <= LINKS; ++link) {
		const uint64_t address = 0x80000u + link * 0x1000u;
		const uint64_t next    = address + 0x1000u;
		chained.regions[address] =
		    link == LINKS ? Dispatch(1, 1, 1, 0) : IndirectBuffer(next, link + 1 == LINKS ? 5 : 4, true);
		called.regions[address] =
		    link == LINKS ? Dispatch(1, 1, 1, 0) : IndirectBuffer(next, link + 1 == LINKS ? 5 : 4, false);
	}
	Stream commands = SetProgram();
	Append(commands, IndirectBuffer(0x81000u, 4, true));
	Stream call_commands = SetProgram();
	Append(call_commands, IndirectBuffer(0x81000u, 4, false));
	CsShadow chain_shadow;
	CsShadow call_shadow;
	const auto by_chain = Scan(commands, chained, chain_shadow);
	const auto by_call  = Scan(call_commands, called, call_shadow);
	CHECK(by_chain.dispatches.size() == 1 && by_chain.stats.ib_depth_limited == 0);
	CHECK(by_call.dispatches.empty() && by_call.stats.ib_depth_limited >= 1);
}

// Three buffers at different addresses holding the same dispatch, so none is a revisit, used by the
// settled cache tests. The stream runs two dispatches of one program, calls each buffer, then sets a
// second program and dispatches once, which is six dispatches of two keys.
struct RepeatedStream {
	FakeMemory memory;
	Stream     stream;
	uint64_t   second_program = PROGRAM_ADDRESS + 0x10000u;
};

RepeatedStream MakeRepeatedStream() {
	RepeatedStream repeated;
	for (uint32_t copy = 0; copy < 3; ++copy) {
		repeated.memory.regions[0x90000u + copy * 0x1000u] = Dispatch(1, 1, 1, 0);
	}
	repeated.stream = SetProgram();
	Append(repeated.stream, Dispatch(1, 1, 1, 0));
	Append(repeated.stream, Dispatch(1, 1, 1, 0));
	for (uint32_t copy = 0; copy < 3; ++copy) {
		Append(repeated.stream, IndirectBuffer(0x90000u + copy * 0x1000u, 5));
	}
	Append(repeated.stream, SetProgram(repeated.second_program));
	Append(repeated.stream, Dispatch(2, 1, 1, 0));
	return repeated;
}

// Walks the stream with a cache and a fresh shadow.
ScanResult ScanWithCache(const RepeatedStream& repeated, DispatchCache& cache) {
	ScanResult result;
	CsShadow   shadow;
	SubmitScanner(repeated.memory.Reader()).Scan(repeated.stream, 3, false, shadow, result.dispatches,
	                                             result.stats, &cache);
	return result;
}

// The same dispatches are found with the cache off and with it on, and the cache removes only the
// repeats. The first walk with it on finds every distinct program and key, later walks skip what the
// caller remembered, and a program the caller has not settled is still found.
void TestSettledCacheKeepsCoverage() {
	constexpr uint32_t PROBE_INTERVAL = 64;
	const auto         repeated       = MakeRepeatedStream();
	CsShadow           off_shadow;
	const auto         off = Scan(repeated.stream, repeated.memory, off_shadow);
	CHECK(off.dispatches.size() == 6);
	std::set<uint64_t> off_keys;
	for (const auto& dispatch: off.dispatches) {
		off_keys.insert(dispatch.key_hash);
	}
	CHECK(off_keys.size() == 2);

	// With the cache on, nothing is remembered yet, so the walk finds exactly what it did with it off.
	DispatchCache cache(PROBE_INTERVAL);
	const auto    first = ScanWithCache(repeated, cache);
	CHECK(first.dispatches.size() == off.dispatches.size());
	std::set<uint64_t> on_keys;
	for (const auto& dispatch: first.dispatches) {
		on_keys.insert(dispatch.key_hash);
		cache.Remember(dispatch.key_hash);
	}
	CHECK(on_keys == off_keys);

	// The program is remembered, so the next walk skips its dispatches and copies none of them.
	const auto skipped = ScanWithCache(repeated, cache);
	CHECK(skipped.dispatches.empty() && skipped.stats.dispatches_cached == 6);

	// Only the first program stays remembered. The second, which the caller never settled, is found.
	DispatchCache partial(PROBE_INTERVAL);
	partial.Remember(first.dispatches.front().key_hash);
	const auto mixed = ScanWithCache(repeated, partial);
	CHECK(mixed.dispatches.size() == 1 && mixed.dispatches[0].program_address == repeated.second_program);
	CHECK(mixed.stats.dispatches_cached == 5);
}

// At the interval a remembered key is let through, flagged as a probe, and forgotten until remembered.
void TestCacheProbesAtInterval() {
	constexpr uint32_t SMALL_INTERVAL = 4;
	DispatchCache      probing(SMALL_INTERVAL);
	probing.Remember(7);
	for (uint32_t hit = 1; hit < SMALL_INTERVAL; ++hit) {
		CHECK(probing.Check(7) == DispatchCache::Verdict::Skip);
	}
	CHECK(probing.Check(7) == DispatchCache::Verdict::Probe);
	CHECK(probing.Check(7) == DispatchCache::Verdict::Pass);
	CHECK(probing.Check(8) == DispatchCache::Verdict::Pass);
}

// A dispatch with an uncertain program never asks the cache, so it is not skipped and it does not use up
// the probe a certain dispatch is owed. The same dispatch with a certain program, which only differs in
// the doubt, is skipped, and the probe still comes at the interval after the uncertain ones.
void TestUncertainDispatchLeavesCacheAlone() {
	constexpr uint32_t INTERVAL = 4;
	FakeMemory         memory;
	Stream             certain = SetProgram();
	Append(certain, Dispatch(1, 1, 1, 0));
	// An unreadable buffer after the program write puts the same dispatch in doubt.
	Stream uncertain = SetProgram();
	Append(uncertain, IndirectBuffer(0xdead0000ull, 4));
	Append(uncertain, Dispatch(1, 1, 1, 0));

	DispatchCache cache(INTERVAL);
	CsShadow      first_shadow;
	const auto    first = Scan(certain, memory, first_shadow);
	CHECK(first.dispatches.size() == 1);
	if (first.dispatches.empty()) {
		return;
	}
	cache.Remember(first.dispatches[0].key_hash);
	for (uint32_t round = 0; round < INTERVAL * 2; ++round) {
		ScanResult result;
		CsShadow   shadow;
		SubmitScanner(memory.Reader()).Scan(uncertain, 3, false, shadow, result.dispatches, result.stats, &cache);
		CHECK(result.dispatches.size() == 1 && result.dispatches[0].uncertain_program);
		if (result.dispatches.size() != 1) {
			return;
		}
		CHECK(result.dispatches[0].key_hash == first.dispatches[0].key_hash && !result.dispatches[0].force_probe);
	}
	// The certain dispatches see a cache nobody has touched, so skips run up to the interval.
	for (uint32_t hit = 1; hit < INTERVAL; ++hit) {
		CHECK(cache.Check(first.dispatches[0].key_hash) == DispatchCache::Verdict::Skip);
	}
	CHECK(cache.Check(first.dispatches[0].key_hash) == DispatchCache::Verdict::Probe);
}

// A walk of far more buffers than the visited table starts with finds every dispatch and revisits none.
void TestManyBuffersGrowVisitedSet() {
	constexpr uint32_t BUFFERS = 700;
	FakeMemory         memory;
	Stream             stream = SetProgram();
	for (uint32_t index = 0; index < BUFFERS; ++index) {
		const uint64_t address = 0x200000u + index * 0x100u;
		memory.regions[address] = Dispatch(1, 1, 1, 0);
		Append(stream, IndirectBuffer(address, 5));
	}
	CsShadow   shadow;
	ScanLimits limits;
	// The fake memory searches its regions linearly, so a slow machine must not trip the time cap.
	limits.time_cap = std::chrono::seconds(30);
	const auto result = Scan(stream, memory, shadow, limits);
	CHECK(result.dispatches.size() == BUFFERS && result.stats.ib_revisited == 0);
	// The same stream again, on the same thread's reused table, behaves the same.
	CsShadow   again_shadow;
	const auto again = Scan(stream, memory, again_shadow, limits);
	CHECK(again.dispatches.size() == BUFFERS && again.stats.ib_revisited == 0);
}

// A buffer called under two different programs is walked under both, since what it dispatches is
// decided by the program, and under the same program twice it is walked once. The streams differ only
// in the second program, so a set keyed on the address and size alone, which merges the calls, shows
// as the first case finding one dispatch where it should find two.
void TestVisitedSetKeepsProgramStatesApart() {
	constexpr uint64_t SHARED_BUFFER = 0x70000;
	const uint64_t     other_program = PROGRAM_ADDRESS + 0x10000u;
	FakeMemory         memory;
	memory.regions[SHARED_BUFFER] = Dispatch(1, 1, 1, 0);

	Stream two_programs = SetProgram(PROGRAM_ADDRESS);
	Append(two_programs, IndirectBuffer(SHARED_BUFFER, DISPATCH_PACKET_DWORDS));
	Append(two_programs, SetProgram(other_program));
	Append(two_programs, IndirectBuffer(SHARED_BUFFER, DISPATCH_PACKET_DWORDS));
	CsShadow   different_shadow;
	const auto different = Scan(two_programs, memory, different_shadow);
	CHECK(different.dispatches.size() == 2 && different.stats.ib_revisited == 0);
	CHECK(different_shadow.certain);
	if (different.dispatches.size() == 2) {
		CHECK(different.dispatches[0].program_address == PROGRAM_ADDRESS);
		CHECK(different.dispatches[1].program_address == other_program);
	}

	Stream one_program = SetProgram(PROGRAM_ADDRESS);
	Append(one_program, IndirectBuffer(SHARED_BUFFER, DISPATCH_PACKET_DWORDS));
	Append(one_program, SetProgram(PROGRAM_ADDRESS));
	Append(one_program, IndirectBuffer(SHARED_BUFFER, DISPATCH_PACKET_DWORDS));
	CsShadow   same_shadow;
	const auto same = Scan(one_program, memory, same_shadow);
	CHECK(same.dispatches.size() == 1 && same.stats.ib_revisited == 1);
	// The shared buffer only dispatches, so skipping its repeat is no doubt.
	CHECK(same_shadow.certain && same.stats.ib_revisit_doubts == 0);
}

// Calls a helper buffer twice under one state and says whether the walk was left in doubt, with the
// counts of revisits and of revisits that doubted.
struct RevisitRun {
	bool     certain         = false;
	uint64_t revisited       = 0;
	uint64_t revisit_doubts  = 0;
};

RevisitRun RunHelperTwice(const Stream& helper) {
	constexpr uint64_t HELPER_ADDRESS = 0x70000;
	FakeMemory         memory;
	memory.regions[HELPER_ADDRESS] = helper;
	Stream stream                  = SetProgram();
	for (int call = 0; call < 2; ++call) {
		Append(stream, IndirectBuffer(HELPER_ADDRESS, static_cast<uint32_t>(helper.size())));
	}
	Append(stream, Dispatch(1, 1, 1, 0));
	CsShadow   shadow;
	const auto result = Scan(stream, memory, shadow);
	return {shadow.certain, result.stats.ib_revisited, result.stats.ib_revisit_doubts};
}

// A helper buffer called again under the same state is skipped, and that is only a doubt if its first
// walk touched compute key state. One that writes user data cannot change the module, so skipping its
// repeat keeps the program certain, and dispatches of engines that call such helpers all the time can be
// cached. A helper that rewrites the program is the baseline, which does leave the program in doubt.
void TestRevisitDoubtsOnlyAfterKeyStateWrites() {
	const auto user_data = RunHelperTwice(SetSh(Pm4::COMPUTE_USER_DATA_0, {1u, 2u}));
	CHECK(user_data.revisited == 1 && user_data.revisit_doubts == 0 && user_data.certain);
	const auto program = RunHelperTwice(SetProgram());
	CHECK(program.revisited == 1 && program.revisit_doubts == 1 && !program.certain);
}

// A buffer that was read short of its size may be rewritten by the part nobody read, so a program write
// in the read part does not make the program certain. The parent's dispatch after the buffer and the
// inherited one in the next submission are both uncertain. The same streams under the default budget
// read the whole buffer and stay certain.
void TestTruncatedBufferEndsUncertain() {
	constexpr uint64_t BUFFER_ADDRESS = 0x70000;
	constexpr uint32_t UNREAD_TAIL    = 20;
	FakeMemory         memory;
	Stream             buffer = SetProgram(PROGRAM_ADDRESS + 0x10000u);
	const auto         program_dwords = static_cast<uint32_t>(buffer.size());
	buffer.insert(buffer.end(), UNREAD_TAIL, FILLER_WORD);
	memory.regions[BUFFER_ADDRESS] = buffer;
	Stream list                    = SetProgram();
	Append(list, IndirectBuffer(BUFFER_ADDRESS, static_cast<uint32_t>(buffer.size())));
	Append(list, Dispatch(1, 1, 1, 0));

	const auto run = [&](ScanLimits limits) {
		CsShadow   shadow;
		const auto after   = Scan(list, memory, shadow, limits);
		const auto next    = Scan(Dispatch(2, 1, 1, 0), memory, shadow);
		CHECK(after.dispatches.size() == 1 && next.dispatches.size() == 1);
		const bool both = after.dispatches.size() == 1 && next.dispatches.size() == 1;
		return std::pair<bool, bool>(both && after.dispatches[0].uncertain_program,
		                             both && next.dispatches[0].uncertain_program);
	};
	// Enough budget for the program write and some fillers, but not the whole buffer.
	ScanLimits tight;
	tight.dword_budget = static_cast<uint32_t>(list.size()) + program_dwords + UNREAD_TAIL / 2;
	const auto cut = run(tight);
	CHECK(cut.first && cut.second);
	const auto whole = run(ScanLimits {});
	CHECK(!whole.first && !whole.second);
}

// IT_SET_BASE in its compute form gives the base a DISPATCH_INDIRECT in its base form adds its offset
// to, as the processor does. The stream without it, and the one whose SET_BASE is the draw form or
// another index, resolve to 0, which the dispatch reports as unknown, so only a tracked compute base
// gives an address. The low three bits of the address and the high bits of its upper word are dropped,
// as in the processor.
void TestSetBaseResolvesBaseForm() {
	constexpr uint32_t BASE_LOW     = 0x23456000u;
	constexpr uint32_t BASE_HIGH    = 1u;
	constexpr uint32_t OFFSET       = 0x40;
	constexpr uint64_t EXPECTED     = (uint64_t {BASE_HIGH} << 32u) + BASE_LOW + OFFSET;
	constexpr uint32_t JUNK_LOW     = 0x7u;
	constexpr uint32_t JUNK_HIGH    = 0xFFFF0000u;
	FakeMemory         memory;
	const auto         args_of = [&](const Stream& set_base) {
        Stream stream = SetProgram();
        Append(stream, set_base);
        Append(stream, DispatchIndirectBase(OFFSET));
        CsShadow   shadow;
        const auto result = Scan(stream, memory, shadow);
        CHECK(result.dispatches.size() == 1);
        return result.dispatches.empty() ? ~uint64_t {0} : result.dispatches[0].indirect_args_address;
	};
	CHECK(args_of(SetBase(HEADER_SHADER_TYPE_COMPUTE, 1, BASE_LOW | JUNK_LOW, BASE_HIGH | JUNK_HIGH)) == EXPECTED);
	CHECK(args_of({}) == 0);
	CHECK(args_of(SetBase(0, 1, BASE_LOW, BASE_HIGH)) == 0);
	CHECK(args_of(SetBase(HEADER_SHADER_TYPE_COMPUTE, 2, BASE_LOW, BASE_HIGH)) == 0);

	// The address form names its arguments itself and needs no base.
	Stream address_form = SetProgram();
	Append(address_form, Stream {KYTY_PM4(4, Pm4::IT_DISPATCH_INDIRECT, 0), 0x5000u, 0u, 0u});
	CsShadow   address_shadow;
	const auto by_address = Scan(address_form, memory, address_shadow);
	CHECK(by_address.dispatches.size() == 1 && by_address.dispatches[0].indirect_args_address == 0x5000u);
}

// The reset packet clears the base the way it clears the processor's own, so a base form dispatch after
// it resolves nothing. Without the reset the same dispatch, in the next submission, resolves the base,
// so only the reset takes it away.
void TestResetClearsDispatchBase() {
	constexpr uint32_t BASE_LOW = 0x23456000u;
	constexpr uint32_t OFFSET   = 0x40;
	FakeMemory         memory;
	CsShadow           shadow;
	Stream             first = SetProgram();
	Append(first, SetBase(HEADER_SHADER_TYPE_COMPUTE, 1, BASE_LOW, 0));
	(void)Scan(first, memory, shadow);
	CHECK(shadow.dispatch_args_base == BASE_LOW);

	const auto kept = Scan(DispatchIndirectBase(OFFSET), memory, shadow);
	CHECK(kept.dispatches.size() == 1 && kept.dispatches[0].indirect_args_address == BASE_LOW + OFFSET);

	Stream reset {0xC0001024u, 0u};
	Append(reset, SetProgram());
	Append(reset, DispatchIndirectBase(OFFSET));
	const auto cleared = Scan(reset, memory, shadow);
	CHECK(cleared.dispatches.size() == 1 && cleared.dispatches[0].indirect_args_address == 0);
}

// What a COND_EXEC region run said about three dispatches. One is inside the region, one follows it in
// the same list, and one inherits the program in the next submission.
struct RegionRun {
	bool     inside_uncertain    = false;
	bool     after_uncertain     = false;
	bool     inherited_uncertain = false;
	uint64_t counted             = 0;
};

// Runs a list of a program, the region body, a dispatch after it, and then a dispatch of a later
// submission. The body is behind a COND_EXEC that may skip it, unless `with_cond_exec` is false.
RegionRun RunRegion(const Stream& region_body, bool with_cond_exec, const FakeMemory& memory = FakeMemory {}) {
	Stream     list = SetProgram();
	if (with_cond_exec) {
		Append(list, CondExec(static_cast<uint32_t>(region_body.size())));
	}
	Append(list, region_body);
	Append(list, Dispatch(3, 1, 1, 0));
	CsShadow   shadow;
	const auto first     = Scan(list, memory, shadow);
	const auto inherited = Scan(Dispatch(2, 1, 1, 0), memory, shadow);
	CHECK(first.dispatches.size() == 2 && inherited.dispatches.size() == 1);
	if (first.dispatches.size() != 2 || inherited.dispatches.size() != 1) {
		return {};
	}
	return {first.dispatches[0].uncertain_program, first.dispatches[1].uncertain_program,
	        inherited.dispatches[0].uncertain_program, first.stats.uncertain_program};
}

// A COND_EXEC region that changes the program leaves it uncertain once the region ends, so the dispatch
// after the region in the same list and the one that inherits in the next submission are marked, while a
// dispatch inside the region after its own program write is still certain. The same stream without the
// COND_EXEC packet is the baseline where every dispatch is certain. A region that writes nothing but
// user data is also harmless, and a later program write makes the program certain again.
void TestCondExecRegionMakesProgramUncertain() {
	const uint64_t other_program = PROGRAM_ADDRESS + 0x10000u;
	Stream         changes_program = SetProgram(other_program);
	Append(changes_program, Dispatch(1, 1, 1, 0));
	Stream user_data_only = SetSh(Pm4::COMPUTE_USER_DATA_0, {5u, 6u});
	Append(user_data_only, Dispatch(1, 1, 1, 0));

	const auto baseline = RunRegion(changes_program, false);
	CHECK(!baseline.inside_uncertain && !baseline.after_uncertain && !baseline.inherited_uncertain);
	CHECK(baseline.counted == 0);

	const auto region = RunRegion(changes_program, true);
	CHECK(!region.inside_uncertain && region.after_uncertain && region.inherited_uncertain);
	CHECK(region.counted == 1);

	const auto harmless = RunRegion(user_data_only, true);
	CHECK(!harmless.inside_uncertain && !harmless.after_uncertain && !harmless.inherited_uncertain);

	// A program write after the doubt makes it certain again.
	FakeMemory memory;
	Stream     list = SetProgram();
	Append(list, CondExec(static_cast<uint32_t>(changes_program.size())));
	Append(list, changes_program);
	Append(list, SetProgram(other_program + 0x10000u));
	Append(list, Dispatch(3, 1, 1, 0));
	CsShadow   shadow;
	const auto rewritten = Scan(list, memory, shadow);
	CHECK(rewritten.dispatches.size() == 2 && !rewritten.dispatches[1].uncertain_program);
}

// A COND_EXEC region that runs to the end of its buffer has no packet after it to close it, yet the
// program is still in doubt for the next submission. The count may also overrun the buffer, which ends
// the region at the buffer's end. The same list without the packet is the baseline and stays certain.
void TestCondExecRegionEndingTheBufferIsUncertain() {
	constexpr uint32_t OVERRUN = 100;
	FakeMemory         memory;
	Stream             body = SetProgram(PROGRAM_ADDRESS + 0x10000u);
	const auto         inherits_uncertain = [&](uint32_t skipped, bool with_cond_exec) {
        Stream list = SetProgram();
        if (with_cond_exec) {
            Append(list, CondExec(skipped));
        }
        Append(list, body);
        CsShadow shadow;
        (void)Scan(list, memory, shadow);
        const auto next = Scan(Dispatch(2, 1, 1, 0), memory, shadow);
        CHECK(next.dispatches.size() == 1);
        return !next.dispatches.empty() && next.dispatches[0].uncertain_program;
	};
	const auto exact = static_cast<uint32_t>(body.size());
	CHECK(inherits_uncertain(exact, true));
	CHECK(inherits_uncertain(exact + OVERRUN, true));
	CHECK(!inherits_uncertain(exact, false));
}

// A COND_EXEC region judges what its own packets and the buffers it calls wrote together. A call to a
// buffer that sets a program leaves the program in doubt after the region, though the region holds no
// register write of its own. A call to a buffer that writes only user data is the baseline and stays
// certain. An inline program write followed by a call to a clean helper is still in doubt, since the
// helper's clean walk must not hide the region's own write.
void TestCondExecRegionJudgesCalledBuffers() {
	constexpr uint64_t PROGRAM_CALLEE = 0x70000;
	constexpr uint64_t USER_CALLEE    = 0x71000;
	const uint64_t     other_program  = PROGRAM_ADDRESS + 0x10000u;
	FakeMemory         memory;
	Stream             sets_program = SetProgram(other_program);
	Append(sets_program, Dispatch(1, 1, 1, 0));
	Stream user_only = SetSh(Pm4::COMPUTE_USER_DATA_0, {1u, 2u});
	Append(user_only, Dispatch(1, 1, 1, 0));
	memory.regions[PROGRAM_CALLEE] = sets_program;
	memory.regions[USER_CALLEE]    = user_only;

	const Stream call_program = IndirectBuffer(PROGRAM_CALLEE, static_cast<uint32_t>(sets_program.size()));
	const Stream call_user    = IndirectBuffer(USER_CALLEE, static_cast<uint32_t>(user_only.size()));
	const auto   program_run  = RunRegion(call_program, true, memory);
	CHECK(!program_run.inside_uncertain && program_run.after_uncertain && program_run.inherited_uncertain);
	const auto user_run = RunRegion(call_user, true, memory);
	CHECK(!user_run.inside_uncertain && !user_run.after_uncertain && !user_run.inherited_uncertain);

	Stream inline_then_call = SetProgram(other_program);
	Append(inline_then_call, call_user);
	const auto mixed_run = RunRegion(inline_then_call, true, memory);
	CHECK(!mixed_run.inside_uncertain && mixed_run.after_uncertain && mixed_run.inherited_uncertain);
}

// A predicated packet may be skipped by the processor, so one that writes compute state leaves the
// program uncertain, applied after the packet's own writes so that a program write in it does not undo
// the doubt. One that writes only user data cannot change the module and leaves the program certain.
// The two streams differ only in the register the predicated packet writes.
void TestPredicatedPacketsOnlyDoubtWrites() {
	constexpr uint32_t PREDICATE_BIT = 1;
	FakeMemory         memory;
	const auto         certain_after = [&](uint32_t reg, const Stream& values) {
        Stream stream = SetProgram();
        Stream packet = SetSh(reg, values);
        packet[0] |= PREDICATE_BIT;
        Append(stream, packet);
        CsShadow shadow;
        (void)Scan(stream, memory, shadow);
        return shadow.certain;
	};
	const uint32_t low  = static_cast<uint32_t>((PROGRAM_ADDRESS + 0x10000u) >> 8u);
	const uint32_t high = static_cast<uint32_t>((PROGRAM_ADDRESS + 0x10000u) >> 40u);
	CHECK(!certain_after(Pm4::COMPUTE_PGM_LO, {low, high}));
	CHECK(certain_after(Pm4::COMPUTE_USER_DATA_0, {1u}));
	// A predicated dispatch changes no state the key reads either.
	Stream predicated_dispatch = SetProgram();
	Stream dispatch            = Dispatch(1, 1, 1, 0);
	dispatch[0] |= PREDICATE_BIT;
	Append(predicated_dispatch, dispatch);
	CsShadow   dispatch_shadow;
	const auto result = Scan(predicated_dispatch, memory, dispatch_shadow);
	CHECK(result.dispatches.size() == 1 && dispatch_shadow.certain);
}

// A branch and a walk cut by the time cap each leave the program uncertain, and a plain packet stream
// does not, which is the baseline. One half of the program address is not enough to make it certain
// again, and both halves are, so a half written program cannot be taken for a whole one.
void TestOtherDoubtsMakeProgramUncertain() {
	FakeMemory memory;
	memory.regions[0x30000] = Dispatch(1, 1, 1, 0);

	CsShadow plain;
	(void)Scan(SetProgram(), memory, plain);
	CHECK(plain.certain);

	Stream branch = SetProgram();
	Append(branch, Branch(1, 0x30000, 0x30000, DISPATCH_PACKET_DWORDS));
	CsShadow branch_shadow;
	(void)Scan(branch, memory, branch_shadow);
	CHECK(!branch_shadow.certain);

	// A cut by the time cap. The stream is the one TestTimeCapChecksClockWhileWalking cuts.
	Stream cut = SetProgram();
	for (uint32_t index = 0; index < SCAN_CLOCK_CHECK_PACKETS * 2u; ++index) {
		cut.push_back(FILLER_WORD);
	}
	ScanLimits no_time;
	no_time.time_cap = std::chrono::microseconds(0);
	CsShadow   cut_shadow;
	const auto cut_result = Scan(cut, memory, cut_shadow, no_time);
	CHECK(cut_result.stats.time_capped == 1 && !cut_shadow.certain);

	// Marking by hand, as a dropped list will. One half is not enough and the second makes it certain.
	CsShadow marked;
	(void)Scan(SetProgram(), memory, marked);
	marked.MarkUncertain();
	CHECK(!marked.certain);
	(void)Scan(SetSh(Pm4::COMPUTE_PGM_LO, {1u}), memory, marked);
	CHECK(!marked.certain);
	(void)Scan(SetSh(Pm4::COMPUTE_PGM_HI, {1u}), memory, marked);
	CHECK(marked.certain);
}

// A dispatch whose program was uncertain is not remembered as settled, so the next one with the same
// key is still posted. The same answer for a certain dispatch is remembered and skipped, which is
// the baseline.
void TestUncertainDispatchIsNeverCached() {
	using Answer = DispatchCache::PostAnswer;
	constexpr uint32_t INTERVAL = 4;
	DispatchCache      cache(INTERVAL);
	CHECK(!cache.Note(21, Answer::Settled, true));
	CHECK(cache.Check(21) == DispatchCache::Verdict::Pass);
	CHECK(!cache.Note(22, Answer::Busy, true));
	CHECK(cache.Check(22) == DispatchCache::Verdict::Pass);
	CHECK(cache.Note(23, Answer::Settled, false));
	CHECK(cache.Check(23) == DispatchCache::Verdict::Skip);
}

// A walk that has used its time stops. A reader that takes longer than the cap on every read ends the
// walk after the first, while the same stream read at full speed under the same cap finds every
// dispatch and is not capped, so the stop is the cap and not the stream.
void TestTimeCapStopsSlowReader() {
	constexpr uint32_t BUFFERS = 50;
	FakeMemory         memory;
	Stream             stream = SetProgram();
	for (uint32_t index = 0; index < BUFFERS; ++index) {
		const uint64_t address = 0xA00000u + index * 0x100u;
		memory.regions[address] = Dispatch(1, 1, 1, 0);
		Append(stream, IndirectBuffer(address, 5));
	}
	ScanLimits limits;
	limits.time_cap = std::chrono::milliseconds(10);

	CsShadow   fast_shadow;
	const auto fast = Scan(stream, memory, fast_shadow, limits);
	CHECK(fast.dispatches.size() == BUFFERS && fast.stats.time_capped == 0);

	// The same memory read slowly, where each read spins for longer than the whole cap.
	const auto  inner = memory.Reader();
	ScanMemoryReader slow = [&](uint64_t address, void* destination, uint64_t bytes) {
		const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
		while (std::chrono::steady_clock::now() < until) {
		}
		return inner(address, destination, bytes);
	};
	ScanResult slow_result;
	CsShadow   slow_shadow;
	SubmitScanner(slow, limits).Scan(stream, 3, false, slow_shadow, slow_result.dispatches, slow_result.stats);
	CHECK(slow_result.stats.time_capped == 1);
	CHECK(!slow_result.dispatches.empty() && slow_result.dispatches.size() < BUFFERS);
}

// A long run of packets with no reads is cut by the clock check every few packets. With the time at
// zero it is capped before the dispatch at its end, and under the normal cap the same stream finds it.
void TestTimeCapChecksClockWhileWalking() {
	FakeMemory memory;
	Stream     stream = SetProgram();
	for (uint32_t index = 0; index < SCAN_CLOCK_CHECK_PACKETS * 8u; ++index) {
		stream.push_back(0x80000000u);
	}
	Append(stream, Dispatch(1, 1, 1, 0));
	CsShadow   normal_shadow;
	const auto normal = Scan(stream, memory, normal_shadow);
	CHECK(normal.dispatches.size() == 1 && normal.stats.time_capped == 0);
	ScanLimits none;
	none.time_cap = std::chrono::microseconds(0);
	CsShadow   capped_shadow;
	const auto capped = Scan(stream, memory, capped_shadow, none);
	CHECK(capped.dispatches.empty() && capped.stats.time_capped == 1);
}

// The storage that holds a scan's dispatches does not grow across scans once it is reserved. Three
// scans of 200 dispatches reuse one vector, its capacity stays put and no scan counts an allocation.
// The same scans into a vector that was never reserved do grow it, which is what the counter sees.
void TestDispatchStorageIsReused() {
	FakeMemory memory;
	Stream     stream = SetProgram();
	for (uint32_t index = 0; index < 200; ++index) {
		Append(stream, SetSh(Pm4::COMPUTE_USER_DATA_0, {index, 0u}));
		Append(stream, Dispatch(1, 1, 1, 0));
	}
	ScanLimits limits;
	limits.time_cap = std::chrono::seconds(30);

	std::vector<ScannedDispatch> reserved;
	reserved.reserve(SCAN_DISPATCH_CAPACITY);
	const size_t capacity = reserved.capacity();
	uint64_t     allocations = 0;
	for (int round = 0; round < 3; ++round) {
		reserved.clear();
		CsShadow  shadow;
		ScanStats stats;
		SubmitScanner(memory.Reader(), limits).Scan(stream, 3, false, shadow, reserved, stats);
		allocations += stats.scan_allocations;
		CHECK(reserved.size() == 200);
		CHECK(reserved.capacity() == capacity);
	}
	CHECK(allocations == 0);

	std::vector<ScannedDispatch> unreserved;
	uint64_t                     grown = 0;
	for (int round = 0; round < 3; ++round) {
		unreserved.clear();
		CsShadow  shadow;
		ScanStats stats;
		SubmitScanner(memory.Reader(), limits).Scan(stream, 3, false, shadow, unreserved, stats);
		grown += stats.scan_allocations;
	}
	CHECK(grown >= 1);
}

// The body of TestNoAllocationAfterWarmup. It runs on a thread of its own, since the visited table and
// the buffer scratch live as long as the thread and the earlier tests have already grown this one's.
void CheckNoAllocationAfterWarmup() {
	constexpr uint32_t BUFFERS = 700;
	FakeMemory         memory;
	Stream             calls = SetProgram();
	for (uint32_t index = 0; index < BUFFERS; ++index) {
		const uint64_t address = 0x300000u + index * 0x100u;
		memory.regions[address] = Dispatch(1, 1, 1, 0);
		Append(calls, IndirectBuffer(address, DISPATCH_PACKET_DWORDS));
	}
	ScanLimits limits;
	limits.time_cap = std::chrono::seconds(30);
	std::vector<ScannedDispatch> found;
	found.reserve(SCAN_DISPATCH_CAPACITY);

	// The first walk of a thread allocates the visited table and the scratch, and says so. The dispatch
	// storage was reserved and holds 700, so the count here comes from the table and the scratch alone.
	CsShadow  warm_shadow;
	ScanStats warm;
	SubmitScanner(memory.Reader(), limits).Scan(calls, 3, false, warm_shadow, found, warm);
	CHECK(found.size() == BUFFERS);
	CHECK(warm.scan_allocations >= 1);
	// The same walk again, with nothing grown, allocates nothing.
	found.clear();
	CsShadow  again_shadow;
	ScanStats again;
	SubmitScanner(memory.Reader(), limits).Scan(calls, 3, false, again_shadow, found, again);
	CHECK(found.size() == BUFFERS && again.scan_allocations == 0);

	// A scan of exactly the reserved count does not grow the dispatch storage and one dispatch more
	// does, once. That is the baseline that shows the counter counts growth and is not always zero.
	Stream exact = SetProgram();
	for (size_t index = 0; index < SCAN_DISPATCH_CAPACITY; ++index) {
		Append(exact, Dispatch(1, 1, 1, 0));
	}
	std::vector<ScannedDispatch> at_capacity;
	at_capacity.reserve(SCAN_DISPATCH_CAPACITY);
	CsShadow  exact_shadow;
	ScanStats exact_stats;
	SubmitScanner(memory.Reader(), limits).Scan(exact, 3, false, exact_shadow, at_capacity, exact_stats);
	CHECK(at_capacity.size() == SCAN_DISPATCH_CAPACITY && exact_stats.scan_allocations == 0);

	Append(exact, Dispatch(1, 1, 1, 0));
	std::vector<ScannedDispatch> over_capacity;
	over_capacity.reserve(SCAN_DISPATCH_CAPACITY);
	CsShadow  over_shadow;
	ScanStats over_stats;
	SubmitScanner(memory.Reader(), limits).Scan(exact, 3, false, over_shadow, over_capacity, over_stats);
	CHECK(over_capacity.size() == SCAN_DISPATCH_CAPACITY + 1 && over_stats.scan_allocations == 1);
}

// Once a thread's reused storage has been through one walk of a size, a second walk of that size
// allocates nothing, and the counter is not simply stuck at zero, since the first walk and a walk past
// the dispatch capacity both report growth.
void TestNoAllocationAfterWarmup() {
	std::thread worker(CheckNoAllocationAfterWarmup);
	worker.join();
}

// A dispatch whose post says a snapshot is in flight is remembered and skipped, like a settled one, and
// the probe still lets one through at the interval. A dispatch that was posted, or whose post was
// contended, is not remembered, since a later one has something to do.
void TestBusyPostIsRememberedAndProbed() {
	constexpr uint32_t INTERVAL = 4;
	using Answer                = DispatchCache::PostAnswer;
	DispatchCache cache(INTERVAL);
	CHECK(cache.Check(11) == DispatchCache::Verdict::Pass);
	CHECK(cache.Note(11, Answer::Busy, false));
	for (uint32_t hit = 1; hit < INTERVAL; ++hit) {
		CHECK(cache.Check(11) == DispatchCache::Verdict::Skip);
	}
	CHECK(cache.Check(11) == DispatchCache::Verdict::Probe);

	CHECK(cache.Note(12, Answer::Settled, false));
	CHECK(cache.Check(12) == DispatchCache::Verdict::Skip);
	CHECK(!cache.Note(13, Answer::Posted, false));
	CHECK(cache.Check(13) == DispatchCache::Verdict::Pass);
	CHECK(!cache.Note(14, Answer::Contended, false));
	CHECK(cache.Check(14) == DispatchCache::Verdict::Pass);
	CHECK(!cache.Note(15, Answer::Other, false));
	CHECK(cache.Check(15) == DispatchCache::Verdict::Pass);
}

// The default before the scan moved off the submitting thread, which a game's top level lists outgrew.
constexpr uint32_t OLD_DWORD_BUDGET = 1u << 17;

// A top level list of `fillers` filler words that ends in a dispatch, after a program is set. Whether
// the dispatch is found says whether the walk got to the end of the list.
Stream LongListEndingInDispatch(size_t fillers) {
	Stream list = SetProgram();
	list.insert(list.end(), fillers, FILLER_WORD);
	Append(list, Dispatch(1, 1, 1, 0));
	return list;
}

// The default budget reads a list longer than the old one whole, and the shadow stays certain. The old
// budget on the same list, which differs only in the limit, cuts it before the dispatch, finds nothing
// and leaves the program in doubt. The same list behind an indirect buffer is read through the arena.
void TestRaisedBudgetScansLongListCertain() {
	constexpr size_t LONG_LIST_FILLERS = size_t {OLD_DWORD_BUDGET} + 70000;
	static_assert(LONG_LIST_FILLERS > OLD_DWORD_BUDGET && LONG_LIST_FILLERS < SCAN_DWORD_BUDGET);
	FakeMemory memory;
	ScanLimits slow_machine;
	slow_machine.time_cap = std::chrono::seconds(30);
	ScanLimits old_limits = slow_machine;
	old_limits.dword_budget = OLD_DWORD_BUDGET;

	const Stream list = LongListEndingInDispatch(LONG_LIST_FILLERS);
	CsShadow     raised_shadow;
	const auto   raised = Scan(list, memory, raised_shadow, slow_machine);
	CHECK(raised.dispatches.size() == 1 && raised.stats.budget_exhausted == 0);
	CHECK(raised_shadow.certain);
	CHECK(!raised.dispatches.empty() && !raised.dispatches[0].uncertain_program);
	CsShadow   old_shadow;
	const auto old = Scan(list, memory, old_shadow, old_limits);
	CHECK(old.dispatches.empty() && old.stats.budget_exhausted >= 1 && !old_shadow.certain);

	// The same list in an indirect buffer, which the walk reads into the arena.
	Stream buffer(LONG_LIST_FILLERS, FILLER_WORD);
	Append(buffer, Dispatch(1, 1, 1, 0));
	memory.regions[0x900000u] = buffer;
	Stream behind             = SetProgram();
	Append(behind, IndirectBuffer(0x900000u, static_cast<uint32_t>(buffer.size())));
	CsShadow   behind_shadow;
	const auto through_buffer = Scan(behind, memory, behind_shadow, slow_machine);
	CHECK(through_buffer.dispatches.size() == 1 && through_buffer.stats.budget_exhausted == 0);
	CHECK(behind_shadow.certain);
	CsShadow   behind_old_shadow;
	const auto through_old = Scan(behind, memory, behind_old_shadow, old_limits);
	CHECK(through_old.dispatches.empty() && through_old.stats.budget_exhausted >= 1);
}

// Builds `levels` buffers that call each other, each holding `fillers` filler words, and a dispatch in
// the last. Returns the top level list that calls the first.
Stream MakeDeepStream(FakeMemory& memory, uint32_t levels, size_t fillers) {
	constexpr uint64_t LEVEL_SPACING = 0x400000u;
	constexpr uint64_t FIRST_LEVEL   = 0x4000000u;
	for (uint32_t level = 1; level <= levels; ++level) {
		Stream buffer(fillers, FILLER_WORD);
		if (level == levels) {
			Append(buffer, Dispatch(1, 1, 1, 0));
		} else {
			const auto next_size = static_cast<uint32_t>(fillers + (level + 1 == levels ? DISPATCH_PACKET_DWORDS : 4u));
			Append(buffer, IndirectBuffer(FIRST_LEVEL + (level + 1ull) * LEVEL_SPACING, next_size));
		}
		memory.regions[FIRST_LEVEL + level * LEVEL_SPACING] = std::move(buffer);
	}
	const auto first_size = static_cast<uint32_t>(fillers + (levels == 1 ? DISPATCH_PACKET_DWORDS : 4u));
	Stream     top        = SetProgram();
	Append(top, IndirectBuffer(FIRST_LEVEL + LEVEL_SPACING, first_size));
	return top;
}

// The body of TestArenaStopsGrowingAfterWarmup, on a thread of its own so that no earlier test has
// grown this thread's arena.
void CheckArenaStopsGrowing() {
	constexpr uint32_t DEEP_LEVELS   = 15;
	constexpr size_t   DEEP_FILLERS  = 20000;
	constexpr uint32_t WIDE_BUFFERS  = 700;
	ScanLimits         limits;
	limits.time_cap = std::chrono::seconds(30);
	std::vector<ScannedDispatch> found;
	found.reserve(SCAN_DISPATCH_CAPACITY);
	const auto walk = [&](const Stream& stream, const FakeMemory& memory) {
		found.clear();
		CsShadow  shadow;
		ScanStats stats;
		SubmitScanner(memory.Reader(), limits).Scan(stream, 3, false, shadow, found, stats);
		return stats.scan_allocations;
	};

	// The first walk of the thread is a shallow one with small buffers. It allocates the arena, once.
	FakeMemory shallow_memory;
	const auto shallow = MakeDeepStream(shallow_memory, 2, 100);
	CHECK(walk(shallow, shallow_memory) >= 1 && found.size() == 1);

	// A deep walk of buffers a thousand times as large is not met by growth, since the arena was sized by
	// the budget. A per level scratch grew here, and an arena that is rebuilt per walk would too.
	FakeMemory deep_memory;
	const auto deep = MakeDeepStream(deep_memory, DEEP_LEVELS, DEEP_FILLERS);
	for (int round = 0; round < 3; ++round) {
		CHECK(walk(deep, deep_memory) == 0 && found.size() == 1);
	}

	// A wide walk grows the visited table once, which is a different piece of storage, and then not again.
	FakeMemory wide_memory;
	Stream     wide = SetProgram();
	for (uint32_t index = 0; index < WIDE_BUFFERS; ++index) {
		const uint64_t address = 0x300000u + index * 0x100u;
		wide_memory.regions[address] = Dispatch(1, 1, 1, 0);
		Append(wide, IndirectBuffer(address, DISPATCH_PACKET_DWORDS));
	}
	(void)walk(wide, wide_memory);
	for (int round = 0; round < 3; ++round) {
		CHECK(walk(wide, wide_memory) == 0 && found.size() == WIDE_BUFFERS);
	}
}

// After its first walk a thread's scratch storage stops growing, whatever deep or wide streams follow.
void TestArenaStopsGrowingAfterWarmup() {
	std::thread worker(CheckArenaStopsGrowing);
	worker.join();
}

// Reset starts a new generation and touches no slot, so the old entries are still stored, and yet none
// is found. The baseline is the same set without the Reset, which finds every key. A mark left on an
// old entry does not leak into the walk that follows. A generation counter that wraps clears the table,
// since an entry from a whole cycle earlier would look current.
void TestVisitedSetResetKeepsOldEntriesHidden() {
	constexpr uint64_t KEYS = 2000;
	VisitedSet         set;
	bool               clean = false;
	for (uint64_t key = 1; key <= KEYS; ++key) {
		CHECK(set.Insert(key * 2, clean));
	}
	set.MarkClean(2 * 100);
	CHECK(!set.Insert(2 * 100, clean) && clean);
	CHECK(set.TakeGrowths() >= 1);

	// Without a Reset every key is found.
	for (uint64_t key = 1; key <= KEYS; ++key) {
		CHECK(!set.Insert(key * 2, clean));
	}

	set.Reset();
	CHECK(set.HoldsStaleEntry(2 * 100) && set.HoldsStaleEntry(KEYS * 2));
	for (uint64_t key = 1; key <= KEYS; ++key) {
		CHECK(set.Insert(key * 2, clean));
	}
	// The mark the key carried before the Reset is gone with it.
	CHECK(!set.Insert(2 * 100, clean) && !clean);
	CHECK(set.TakeGrowths() == 0);

	VisitedSet wrapping;
	CHECK(wrapping.Insert(2 * 7, clean));
	wrapping.SetGenerationForTest(UINT32_MAX);
	CHECK(wrapping.Insert(2 * 9, clean));
	wrapping.Reset();
	CHECK(!wrapping.HoldsStaleEntry(2 * 7) && !wrapping.HoldsStaleEntry(2 * 9));
	CHECK(wrapping.Insert(2 * 7, clean) && wrapping.Insert(2 * 9, clean));
}

// What one dispatch looked like to the scan, for comparing two ways of running it.
struct Seen {
	uint32_t queue;
	uint64_t program;
	uint64_t key_hash;
	bool     operator==(const Seen&) const = default;
};

// One step of a run, which is a command list on a queue or the reset a flip suspend point makes on it.
struct Step {
	uint32_t queue;
	bool     reset;
	Stream   commands;
};

// Works through jobs the way the scanner thread does, which is what C14a will wire up, and says what
// each dispatch looked like.
std::vector<Seen> DrainQueue(ScanJobQueue& queue, const FakeMemory& memory,
                             std::map<uint32_t, CsShadow>& shadows) {
	std::vector<Seen> found;
	ScanJob           job;
	while (queue.TryPop(job)) {
		ScanResult result;
		RunScanJob(SubmitScanner(memory.Reader()), job, false, shadows[job.queue_id], result.dispatches,
		           result.stats);
		for (const auto& dispatch: result.dispatches) {
			found.push_back({job.queue_id, dispatch.program_address, dispatch.key_hash});
		}
	}
	queue.Recycle(std::move(job));
	return found;
}

// Runs the steps the way the off-thread mode does. The submit side copies and queues, and then the
// scanner side takes the jobs in order and walks the copies with the same reader. `with_resets` false
// leaves the reset markers out.
std::vector<Seen> RunThroughQueue(const std::vector<Step>& steps, const FakeMemory& memory, bool with_resets) {
	ScanJobQueue queue;
	for (const auto& step: steps) {
		if (step.reset) {
			if (with_resets) {
				(void)queue.Push(ScanJobQueue::MakeReset(step.queue));
			}
			continue;
		}
		(void)queue.Push(queue.MakeCommands(step.queue, step.commands));
	}
	std::map<uint32_t, CsShadow> shadows;
	return DrainQueue(queue, memory, shadows);
}

// Runs the steps by scanning each list directly against a shadow for each queue, the way the scan on the
// submitting thread does.
std::vector<Seen> RunDirectly(const std::vector<Step>& steps, const FakeMemory& memory) {
	std::map<uint32_t, CsShadow> shadows;
	std::vector<Seen>            found;
	for (const auto& step: steps) {
		if (step.reset) {
			shadows[step.queue].Reset();
			continue;
		}
		ScanResult result;
		SubmitScanner(memory.Reader()).Scan(step.commands, step.queue, false, shadows[step.queue],
		                                    result.dispatches, result.stats);
		for (const auto& dispatch: result.dispatches) {
			found.push_back({step.queue, dispatch.program_address, dispatch.key_hash});
		}
	}
	return found;
}

// The steps of TestQueuedScanMatchesDirectScan, which use two queues, an indirect buffer, a reset marker
// between submissions and a reset packet inside a list.
std::vector<Step> MixedQueueSteps(FakeMemory& memory) {
	memory.regions[0xB0000u]      = Dispatch(4, 1, 1, 0);
	const uint64_t second_program = PROGRAM_ADDRESS + 0x20000u;
	Stream         first          = SetProgram();
	Append(first, IndirectBuffer(0xB0000u, 5));
	Stream other = SetProgram(second_program);
	Append(other, Dispatch(2, 2, 2, WAVE32_BIT));
	Stream inherits = Dispatch(1, 1, 1, 0);
	Stream reset_inside {0xC0001024u, 0u};
	Append(reset_inside, Dispatch(1, 1, 1, 0));
	Stream again = SetProgram();
	Append(again, Dispatch(9, 9, 9, 0));
	return {
	    {3, false, first},        // finds one dispatch through the buffer
	    {5, false, other},        // another queue, with its own program
	    {3, false, inherits},     // inherits queue 3's program
	    {3, true, {}},            // the flip suspend point
	    {3, false, inherits},     // nothing to inherit now
	    {5, false, inherits},     // queue 5 was not reset, so it inherits
	    {5, false, reset_inside}, // a reset packet inside the list
	    {3, false, again},
	};
}

// Scanning through the queue finds what scanning the lists directly finds, in the same order. Without
// the reset marker the run differs, so the marker is what makes the later dispatch of queue 3 disappear.
void TestQueuedScanMatchesDirectScan() {
	FakeMemory memory;
	const auto steps  = MixedQueueSteps(memory);
	const auto direct = RunDirectly(steps, memory);
	CHECK(direct.size() == 5);
	CHECK(RunThroughQueue(steps, memory, true) == direct);
	// Without the marker the shadow of queue 3 is not reset, and the dispatch after it is found.
	CHECK(RunThroughQueue(steps, memory, false).size() == direct.size() + 1);
}

// Takes every job the queue hands out, as whether it is a reset marker and its queue id, in order.
std::vector<std::pair<bool, uint32_t>> PopAll(ScanJobQueue& queue) {
	std::vector<std::pair<bool, uint32_t>> met;
	ScanJob                                job;
	while (queue.TryPop(job)) {
		met.emplace_back(job.kind == ScanJob::Kind::Reset, job.queue_id);
	}
	queue.Recycle(std::move(job));
	return met;
}

// A full queue drops its oldest command list and counts it, and keeps the markers and the order of
// what is left. Dropping the newest instead, or a marker, changes what the scanner meets.
void TestQueueDropsOldestCommands() {
	ScanJobQueue queue(3, 64);
	const auto   list = [&](uint32_t tag) { return queue.MakeCommands(tag, Stream {1, 2, 3}); };
	(void)queue.Push(list(1));
	(void)queue.Push(list(2));
	(void)queue.Push(ScanJobQueue::MakeReset(7));
	(void)queue.Push(list(3)); // drops 1
	(void)queue.Push(list(4)); // drops 2
	(void)queue.Push(list(5)); // drops 3, and the marker stays
	CHECK(queue.Stats().dropped == 3);
	CHECK(queue.Size() == 3);
	const std::vector<std::pair<bool, uint32_t>> expected {{true, 7}, {false, 4}, {false, 5}};
	CHECK(PopAll(queue) == expected);

	// Only markers waiting, so nothing is dropped however many arrive.
	ScanJobQueue markers(2, 64);
	for (uint32_t index = 0; index < 5; ++index) {
		(void)markers.Push(ScanJobQueue::MakeReset(index));
	}
	CHECK(markers.Stats().dropped == 0 && markers.Size() == 5);
}

// Reset markers are never dropped, however full the queue is and whatever is waiting. The baseline is
// the command lists on the same queue, which are dropped and counted, so a queue that never dropped
// anything could not pass. Dropping the oldest job of any kind would take the first marker at the
// second list.
void TestResetMarkersNeverDropped() {
	ScanJobQueue queue(3, 64);
	const auto   list = [&](uint32_t tag) { return queue.MakeCommands(tag, Stream {1, 2, 3}); };
	(void)queue.Push(ScanJobQueue::MakeReset(10));
	(void)queue.Push(ScanJobQueue::MakeReset(11));
	(void)queue.Push(list(1));
	(void)queue.Push(list(2)); // drops list 1
	(void)queue.Push(list(3)); // drops list 2
	// The queue now holds two markers and the third list, and markers arrive at a full queue. They wait in
	// the reserve and evict nothing, so list 3 survives and only lists 1 and 2 were dropped.
	for (uint32_t marker = 20; marker < 25; ++marker) {
		(void)queue.Push(ScanJobQueue::MakeReset(marker));
	}
	CHECK(queue.Stats().dropped == 2 && queue.Stats().markers_lost == 0);
	const std::vector<std::pair<bool, uint32_t>> expected {{true, 10}, {true, 11}, {false, 3}, {true, 20},
	                                                       {true, 21}, {true, 22}, {true, 23}, {true, 24}};
	CHECK(PopAll(queue) == expected);
}

// Pushing past the capacity drops a list the GPU thread has already begun before an older one it has
// not. The two runs differ only in the watermark, so the order the survivors come out in is the
// preference and not the age.
void TestDropPrefersExecutedJobs() {
	const auto run = [](bool began) {
		ScanJobQueue queue(3, 64);
		const auto   list = [&](uint32_t tag) { return queue.MakeCommands(tag, Stream {1, 2, 3}); };
		(void)queue.Push(list(5));
		(void)queue.Push(list(3));
		(void)queue.Push(list(3));
		if (began) {
			queue.NoteBegan(3, 1);
		}
		(void)queue.Push(list(6));
		return PopAll(queue);
	};
	// Queue 3's first list began, so it is the one that goes, and the older list of queue 5 stays.
	const std::vector<std::pair<bool, uint32_t>> preferred {{false, 5}, {false, 3}, {false, 6}};
	CHECK(run(true) == preferred);
	// With nothing begun the oldest, which is queue 5's, goes.
	const std::vector<std::pair<bool, uint32_t>> plain {{false, 3}, {false, 3}, {false, 6}};
	CHECK(run(false) == plain);
}

// The scanner skips the lists the GPU thread began and counts them, and walks the rest. The baseline is
// the same queue with no watermark, which hands out every list. Ignoring the watermark would hand out
// the late ones too.
void TestWatermarkSkipsLateJobs() {
	const auto fill = [](ScanJobQueue& queue) {
		for (uint32_t index = 0; index < 3; ++index) {
			(void)queue.Push(queue.MakeCommands(3, Stream {1, 2, 3}));
		}
		(void)queue.Push(queue.MakeCommands(4, Stream {1, 2, 3}));
	};
	ScanJobQueue plain;
	fill(plain);
	ScanJob job;
	uint32_t handed_out = 0;
	while (plain.TryPop(job)) {
		handed_out++;
	}
	CHECK(handed_out == 4 && plain.Stats().late_skipped == 0);

	ScanJobQueue queue;
	fill(queue);
	queue.NoteBegan(3, 2);
	// The watermark only moves forward.
	queue.NoteBegan(3, 1);
	CHECK(queue.Watermark(3) == 2 && queue.Watermark(4) == 0);
	std::vector<std::pair<uint32_t, uint64_t>> met;
	std::vector<bool>                          doubts;
	while (queue.TryPop(job)) {
		met.emplace_back(job.queue_id, job.seq);
		doubts.push_back(job.shadow_uncertain);
	}
	// Queue 3's third list and queue 4's first, in order. The skipped lists left queue 3 in doubt, and
	// queue 4, which skipped nothing, is not.
	const std::vector<std::pair<uint32_t, uint64_t>> expected {{3, 3}, {4, 1}};
	CHECK(met == expected);
	const std::vector<bool> expected_doubts {true, false};
	CHECK(doubts == expected_doubts);
	CHECK(queue.Stats().late_skipped == 2);
}

// A dropped list leaves the next list of its queue walked under a shadow that is marked uncertain, so
// the dispatch it finds carries the doubt. The baseline is the same lists with room for all of them,
// where that dispatch is certain. With the mark missing, the drop would leave it certain.
void TestDropMarksShadowUncertain() {
	const auto run = [](size_t capacity) {
		FakeMemory   memory;
		ScanJobQueue queue(capacity, 64);
		CsShadow     shadow;
		(void)queue.Push(queue.MakeCommands(3, SetProgram()));
		ScanJob job;
		CHECK(queue.TryPop(job));
		ScanResult first;
		RunScanJob(SubmitScanner(memory.Reader()), job, false, shadow, first.dispatches, first.stats);
		CHECK(shadow.certain);
		// Two more lists. The first of them is the one a full queue drops when the third arrives.
		(void)queue.Push(queue.MakeCommands(3, SetSh(Pm4::COMPUTE_USER_DATA_0, {1u, 2u})));
		(void)queue.Push(queue.MakeCommands(3, Dispatch(1, 1, 1, 0)));
		(void)queue.Push(queue.MakeCommands(3, Dispatch(2, 1, 1, 0)));
		// Hand jobs out until a dispatch turns up, which is the first dispatch list in both runs.
		ScanResult second;
		while (second.dispatches.empty() && queue.TryPop(job)) {
			RunScanJob(SubmitScanner(memory.Reader()), job, false, shadow, second.dispatches, second.stats);
		}
		queue.Recycle(std::move(job));
		CHECK(second.dispatches.size() == 1);
		return second.dispatches.empty() ? false : second.dispatches[0].uncertain_program;
	};
	CHECK(run(8) == false);
	// Capacity 2 holds the user data list and the first dispatch, and the third push drops the user data
	// list, so the first dispatch is the next one handed out.
	CHECK(run(2) == true);
}

// A list longer than the copy cap is cut to it and counted, and a shorter one is copied whole.
void TestCopyIsCapped() {
	ScanJobQueue queue(8, 8);
	Stream       longer(20);
	for (uint32_t index = 0; index < longer.size(); ++index) {
		longer[index] = index + 100;
	}
	auto cut = queue.MakeCommands(1, longer);
	CHECK(cut.CommandSpan().size() == 8 && cut.truncated && cut.CommandSpan()[7] == 107);
	auto whole = queue.MakeCommands(1, Stream {1, 2, 3});
	CHECK(whole.CommandSpan().size() == 3 && !whole.truncated);
	(void)queue.Push(std::move(cut));
	(void)queue.Push(std::move(whole));
	CHECK(queue.Stats().truncated == 1 && queue.Stats().copied_dwords == 11);
}

// A list the copy cut short leaves the shadow uncertain after its walk, since the part that was cut may
// have changed the program. The same list under a cap that holds it whole leaves the shadow certain, so
// the difference is the cut and not the stream. The cut keeps the dispatch, so what was read is found.
void TestTruncatedCopyLeavesShadowUncertain() {
	Stream list = SetProgram();
	Append(list, Dispatch(1, 1, 1, 0));
	const size_t kept = list.size();
	Append(list, SetProgram(PROGRAM_ADDRESS + 0x10000u));
	const auto run = [&](size_t copy_cap) {
		FakeMemory   memory;
		ScanJobQueue queue(4, copy_cap);
		CsShadow     shadow;
		(void)queue.Push(queue.MakeCommands(3, list));
		ScanJob job;
		CHECK(queue.TryPop(job));
		ScanResult result;
		RunScanJob(SubmitScanner(memory.Reader()), job, false, shadow, result.dispatches, result.stats);
		CHECK(result.dispatches.size() == 1);
		return shadow.certain;
	};
	CHECK(run(list.size()));
	CHECK(!run(kept));
}

// A list takes the smallest buffer that holds it, so a small one does not tie up a large one. A queue
// that always handed out the largest would show in every capacity here.
void TestBuffersFitBySizeClass() {
	ScanJobQueue queue;
	const Stream small(100, 1u);
	const Stream medium(5000, 1u);
	const Stream large(100000, 1u);
	auto         small_job  = queue.MakeCommands(1, small);
	auto         medium_job = queue.MakeCommands(1, medium);
	auto         large_job  = queue.MakeCommands(1, large);
	CHECK(small_job.buffer.capacity == SCAN_BUFFER_CLASS_DWORDS[0]);
	CHECK(medium_job.buffer.capacity == SCAN_BUFFER_CLASS_DWORDS[1]);
	CHECK(large_job.buffer.capacity == SCAN_BUFFER_CLASS_DWORDS[2]);
	CHECK(small_job.CommandSpan().size() == 100 && large_job.CommandSpan().size() == 100000);
}

// A stopped queue copies nothing and queues nothing. The queue is stopped before the copy, so the job
// has no buffer, and pushing it is refused without a count. A live queue copies the same list.
void TestStoppedQueueCopiesNothing() {
	ScanJobQueue live;
	const auto   copied = live.MakeCommands(1, Stream {1, 2, 3});
	CHECK(copied.copied && copied.buffer && copied.dwords == 3);

	ScanJobQueue stopped;
	stopped.Stop();
	auto job = stopped.MakeCommands(1, Stream {1, 2, 3});
	CHECK(!job.copied && !job.buffer && job.dwords == 0);
	CHECK(stopped.Push(std::move(job)) == 0);
	CHECK(stopped.Stats().pushed == 0 && stopped.Stats().dropped == 0 && stopped.Size() == 0);
	ScanJob out;
	CHECK(!stopped.Pop(out));
}

// When every buffer that could hold a list is out, the copy is refused and counted rather than
// allocated. The first jobs get buffers and the later ones do not, and those are the ones counted.
void TestNoBufferDropsInsteadOfAllocating() {
	ScanJobQueue queue(2, 64);
	const Stream list(100, 1u);
	std::vector<ScanJob> held;
	uint32_t             refused = 0;
	// A capacity of 2 has 34 small, one medium, one large and the huge buffers.
	constexpr uint32_t BUFFERS = 2 + SCAN_SMALL_BUFFER_SLACK + 1 + 1 + SCAN_HUGE_BUFFER_COUNT;
	for (uint32_t index = 0; index < BUFFERS + 4; ++index) {
		held.push_back(queue.MakeCommands(1, list));
		refused += held.back().copied ? 0 : 1;
	}
	CHECK(refused == 4);
	for (auto& job: held) {
		(void)queue.Push(std::move(job));
	}
	CHECK(queue.Stats().dropped_no_buffer == 4);
	// Only the lists that were copied have a copy time, and the refused ones have none.
	const auto stats = queue.Stats();
	CHECK(std::accumulate(stats.copy_us_histogram.begin(), stats.copy_us_histogram.end(), uint64_t {0}) == BUFFERS);
}

// The copy times fall in a fixed histogram, by powers of two of a microsecond, with the longest kept.
// A list of samples would be unbounded, and this has no storage that grows.
void TestCopyTimeHistogram() {
	ScanJobQueue queue;
	for (const double micros: {0.2, 1.0, 3.0, 100.0, 1.0e12}) {
		(void)queue.Push(queue.MakeCommands(1, Stream {1}), micros);
	}
	const auto stats = queue.Stats();
	CHECK(stats.copy_us_histogram[0] == 1 && stats.copy_us_histogram[1] == 1);
	CHECK(stats.copy_us_histogram[2] == 1 && stats.copy_us_histogram[7] == 1);
	CHECK(stats.copy_us_histogram[SCAN_COPY_HISTOGRAM_BUCKETS - 1] == 1);
	CHECK(stats.copy_us_max == 1000000000000ull);
}

// Two producers submit to one queue and the order of the sequence numbers the scan gave is the order the
// pushes reach the command processor's queue. A yield between the two steps widens the window a thread
// could slip into, so without the gate's lock a number is overtaken within a few rounds. The pushes go
// through ScanOrderGate, which production uses, so the lock that is mutated is the real one.
void TestPerQueueOrderPreserved() {
	constexpr uint32_t PRODUCERS    = 2;
	constexpr uint32_t PER_PRODUCER = 20000;
	constexpr uint32_t QUEUE        = 3;
	ScanJobQueue          queue(8, 64);
	ScanOrderGate         gate;
	std::mutex            cp_mutex;
	std::vector<uint64_t> cp_order;
	cp_order.reserve(size_t {PRODUCERS} * PER_PRODUCER);
	const auto produce = [&] {
		for (uint32_t round = 0; round < PER_PRODUCER; ++round) {
			auto job = queue.MakeCommands(QUEUE, Stream {1, 2, 3});
			gate.Run(
			    QUEUE, [&] { return queue.Push(std::move(job)); },
			    [&](uint64_t seq) {
				    std::this_thread::yield();
				    std::scoped_lock lock(cp_mutex);
				    cp_order.push_back(seq);
			    });
		}
	};
	std::vector<std::thread> threads;
	for (uint32_t producer = 0; producer < PRODUCERS; ++producer) {
		threads.emplace_back(produce);
	}
	for (auto& thread: threads) {
		thread.join();
	}
	CHECK(cp_order.size() == size_t {PRODUCERS} * PER_PRODUCER);
	const bool increasing = std::ranges::adjacent_find(cp_order, std::greater_equal<> {}) == cp_order.end();
	CHECK(increasing);
}

// Runs the queue's whole submit path, copies and pushes and markers and drops and a cut list, for many
// rounds, and counts what the heap saw.
void CycleQueue(ScanJobQueue& queue, const std::vector<const Stream*>& lists, int rounds, ScanJob& job) {
	constexpr uint32_t QUEUE = 3;
	for (int round = 0; round < rounds; ++round) {
		for (const Stream* list: lists) {
			(void)queue.Push(queue.MakeCommands(QUEUE, *list), 1.5);
		}
		if (round % 5 == 0) {
			(void)queue.Push(ScanJobQueue::MakeReset(QUEUE));
		}
		if (round % 3 == 0) {
			(void)queue.TryPop(job);
		}
		// Slower than the pushes, so the newest list is always past it and a pop never has to wait.
		queue.NoteBegan(QUEUE, static_cast<uint64_t>(round / 4));
	}
}

// Once the queue is built the submit path allocates nothing, across every size class, a list cut by the
// copy cap, a full queue that drops and a reset marker. The counter is not stuck at zero, since a
// vector that grows in the same section is seen, and the stats show the drops and the cut did happen.
void TestNoQueueAllocationAfterWarmup() {
	const Stream small(100, 7u);
	const Stream medium(10000, 7u);
	const Stream large(100000, 7u);
	const Stream huge(SCAN_COPY_CAP_DWORDS + 5000, 7u);
	const std::vector<const Stream*> lists {&small, &medium, &large, &huge, &small};
	ScanJob job;
	// Counting starts as the constructor returns, so there is no warm-up to hide an allocation in.
	ScanJobQueue queue(8, SCAN_COPY_CAP_DWORDS);
	g_allocation_count  = 0;
	g_count_allocations = true;
	CycleQueue(queue, lists, 2050, job);
	g_count_allocations = false;
	CHECK(g_allocation_count == 0);
	const auto stats = queue.Stats();
	CHECK(stats.dropped > 0 && stats.truncated > 0 && stats.pushed > 2000);

	g_allocation_count  = 0;
	g_count_allocations = true;
	std::vector<uint32_t> grown;
	for (uint32_t index = 0; index < 100; ++index) {
		grown.push_back(index);
	}
	g_count_allocations = false;
	CHECK(g_allocation_count >= 1);
}

// What came out of running one long list through a queue.
struct QueuedOutcome {
	size_t dispatches = 0;
	bool   certain    = false;
	bool   truncated  = false;
};

// Pushes a list through a queue and the scan and says what the scan made of it.
QueuedOutcome ScanListThroughQueue(ScanJobQueue& queue, const Stream& list, const FakeMemory& memory) {
	ScanLimits slow_machine;
	slow_machine.time_cap = std::chrono::seconds(30);
	CsShadow   shadow;
	ScanResult result;
	ScanJob    job;
	(void)queue.Push(queue.MakeCommands(3, list));
	CHECK(queue.TryPop(job));
	RunScanJob(SubmitScanner(memory.Reader(), slow_machine), job, false, shadow, result.dispatches, result.stats);
	return {result.dispatches.size(), shadow.certain, job.truncated};
}

// A top level list longer than the old budget goes through the queue whole, is walked to its dispatch and
// leaves the shadow certain. The baseline is a queue whose copy cap is the old budget, which cuts the
// same list, never reaches the dispatch and leaves the shadow in doubt, so the copy cap is what is
// proven and not the stream.
void TestQueuedLongListScansCertain() {
	FakeMemory   memory;
	const Stream list = LongListEndingInDispatch(size_t {OLD_DWORD_BUDGET} + 70000);
	ScanJobQueue queue;
	const auto   whole = ScanListThroughQueue(queue, list, memory);
	CHECK(whole.dispatches == 1 && whole.certain && !whole.truncated);
	ScanJobQueue old_cap_queue(4, OLD_DWORD_BUDGET);
	const auto   cut = ScanListThroughQueue(old_cap_queue, list, memory);
	CHECK(cut.dispatches == 0 && !cut.certain && cut.truncated);
}

// The arena gets every slice back however the scope that took it ends. Nested slices stack and unwind,
// an early return releases like a normal end, and a slice too big for the arena is null and leaves the
// arena as it was. A destructor that released nothing would leave the dwords out.
void TestArenaSliceReleasesOnEveryExit() {
	constexpr size_t ARENA_DWORDS = 100;
	ScratchArena     arena;
	arena.Reserve(ARENA_DWORDS);
	{
		const ScratchSlice outer(arena, 40);
		CHECK(outer.Data() != nullptr && arena.Used() == 40);
		{
			const ScratchSlice inner(arena, 30);
			CHECK(inner.Data() == outer.Data() + 40 && arena.Used() == 70);
		}
		CHECK(arena.Used() == 40);
	}
	CHECK(arena.Used() == 0);

	const auto leave_early = [&](bool early) {
		const ScratchSlice slice(arena, 10);
		if (early) {
			return 1;
		}
		return 2;
	};
	CHECK(leave_early(true) == 1 && arena.Used() == 0);
	CHECK(leave_early(false) == 2 && arena.Used() == 0);

	const ScratchSlice held(arena, 60);
	{
		const ScratchSlice too_big(arena, ARENA_DWORDS - 59);
		CHECK(too_big.Data() == nullptr && arena.Used() == 60);
	}
	CHECK(arena.Used() == 60);
}

// What the lists a queue hands out say about their shadow, for a run of pushes. 'L' is a list and 'R' a
// reset marker, on a queue that holds three.
std::vector<bool> DoubtOfListsAfter(const char* pattern) {
	ScanJobQueue queue(3, 64);
	for (const char* step = pattern; *step != '\0'; ++step) {
		if (*step == 'R') {
			(void)queue.Push(ScanJobQueue::MakeReset(3));
		} else {
			(void)queue.Push(queue.MakeCommands(3, Stream {1}));
		}
	}
	std::vector<bool> doubts;
	ScanJob           job;
	while (queue.TryPop(job)) {
		if (job.kind == ScanJob::Kind::Commands) {
			doubts.push_back(job.shadow_uncertain);
		}
	}
	return doubts;
}

// A reset marker clears the doubt of a list dropped before it, since the reset puts the shadow right,
// and keeps the doubt of one dropped after it. Without a marker the same drop leaves the first list that
// follows in doubt, which is the baseline. A marker that cleared nothing would leave the first case in doubt.
void TestMarkerClearsOlderDoubt() {
	// The first list is dropped by the fourth push, then the marker, then lists. Nothing is in doubt.
	const std::vector<bool> after_marker {false, false};
	CHECK(DoubtOfListsAfter("LRLL") == after_marker);
	// The same drop with no marker between leaves the first list handed out in doubt.
	const std::vector<bool> no_marker {true, false, false};
	CHECK(DoubtOfListsAfter("LLLL") == no_marker);
	// The marker comes first and the dropped list is newer than it, so the doubt stays.
	const std::vector<bool> newer_drop {true, false};
	CHECK(DoubtOfListsAfter("RLLL") == newer_drop);
}

// A list that wants a large buffer when those are out gets the largest smaller one, cut and counted, and
// a larger one if that is what is free. The queue allocates nothing to serve it.
void TestBufferFallsBackToSmallerClass() {
	ScanJobQueue queue(8, SCAN_COPY_CAP_DWORDS);
	const Stream hundred_thousand(100000, 1u);
	const Stream two_hundred_thousand(200000, 1u);
	auto         large = queue.MakeCommands(1, hundred_thousand);
	auto         huge_a = queue.MakeCommands(1, two_hundred_thousand);
	auto         huge_b = queue.MakeCommands(1, two_hundred_thousand);
	CHECK(large.buffer.size_class == 2 && huge_a.buffer.size_class == 3 && huge_b.buffer.size_class == 3);
	// Every class that holds 100,000 dwords is out, so the next one takes the medium class and is cut.
	auto cut = queue.MakeCommands(1, hundred_thousand);
	CHECK(cut.buffer.size_class == 1 && cut.truncated && cut.dwords == SCAN_BUFFER_CLASS_DWORDS[1]);
	// With a huge buffer back, the same list is held whole in it.
	queue.Recycle(std::move(huge_a));
	auto whole = queue.MakeCommands(1, hundred_thousand);
	CHECK(whole.buffer.size_class == 3 && !whole.truncated && whole.dwords == 100000);
}

// A full queue of markers and lists. A marker arriving at a full queue evicts a list, which is dropped
// and counted, and when no list is left it is lost and counted, and the shadow is put in doubt in its
// place. The lists the queue hands out afterwards are in doubt, which a loss that left no doubt would not.
void TestMarkerLossIsCountedAndLeavesDoubt() {
	ScanJobQueue queue(2, 64);
	(void)queue.Push(queue.MakeCommands(3, Stream {1}));
	(void)queue.Push(queue.MakeCommands(3, Stream {1}));
	for (size_t marker = 0; marker < SCAN_MARKER_RESERVE; ++marker) {
		(void)queue.Push(ScanJobQueue::MakeReset(3));
	}
	// Two lists and the whole reserve fill the ring.
	const size_t ring_size = 2 + SCAN_MARKER_RESERVE;
	CHECK(queue.Size() == ring_size && queue.Stats().dropped == 0 && queue.Stats().markers_lost == 0);
	(void)queue.Push(ScanJobQueue::MakeReset(3));
	(void)queue.Push(ScanJobQueue::MakeReset(3));
	CHECK(queue.Size() == ring_size && queue.Stats().dropped == 2 && queue.Stats().markers_lost == 0);
	(void)queue.Push(ScanJobQueue::MakeReset(3));
	CHECK(queue.Size() == ring_size && queue.Stats().markers_lost == 1);

	ScanJob job;
	size_t  markers = 0;
	while (queue.TryPop(job)) {
		markers++;
	}
	CHECK(markers == ring_size);
	(void)queue.Push(queue.MakeCommands(3, Stream {1}));
	CHECK(queue.TryPop(job) && job.shadow_uncertain);
}

// A scan given its own storage uses it and not the calling thread's. The storage is allocated by the
// first scan, on one thread, and a scan of the same stream on another thread finds it ready and allocates
// nothing. A scan that used the thread's own would allocate again on the second thread.
void TestScanUsesCallersScratch() {
	FakeMemory memory;
	Stream     stream = SetProgram();
	Append(stream, Dispatch(1, 1, 1, 0));
	ScanScratch scratch;
	const auto  scan = [&](uint64_t& allocations) {
		std::vector<ScannedDispatch> found;
		found.reserve(SCAN_DISPATCH_CAPACITY);
		CsShadow  shadow;
		ScanStats stats;
		SubmitScanner(memory.Reader()).Scan(stream, 3, false, shadow, found, stats, scratch);
		CHECK(found.size() == 1);
		allocations = stats.scan_allocations;
	};
	uint64_t first_allocations  = 0;
	uint64_t second_allocations = 99;
	std::thread([&] { scan(first_allocations); }).join();
	std::thread([&] { scan(second_allocations); }).join();
	CHECK(first_allocations >= 1 && scratch.arena.Capacity() == SCAN_DWORD_BUDGET);
	CHECK(second_allocations == 0);
}

// Whether the dispatch of list C comes out uncertain after this flow on queue 3. A sets the program, X
// writes user data and C dispatches with no program of its own. With `refuse_x` every other buffer is
// held by jobs of queue 5 when X is copied, so X is refused and dropped, and the held jobs are given back
// before C is copied. A is handed out and walked before C is pushed, so A pops after the drop and is older
// than it.
bool DispatchAfterRefusedListIsUncertain(bool refuse_x) {
	constexpr uint32_t HELD_QUEUE = 5;
	FakeMemory         memory;
	ScanJobQueue       queue(2, 64);
	const SubmitScanner scanner(memory.Reader());
	CsShadow           shadow;
	ScanResult         result;
	std::vector<ScanJob> held;
	// A capacity of 2 has 38 buffers, and A takes the last one.
	constexpr uint32_t BUFFERS = 2 + SCAN_SMALL_BUFFER_SLACK + 1 + 1 + SCAN_HUGE_BUFFER_COUNT;
	for (uint32_t index = 0; refuse_x && index + 1 < BUFFERS; ++index) {
		held.push_back(queue.MakeCommands(HELD_QUEUE, Stream {1}));
	}
	(void)queue.Push(queue.MakeCommands(3, SetProgram()));
	auto x = queue.MakeCommands(3, SetSh(Pm4::COMPUTE_USER_DATA_0, {1u, 2u}));
	CHECK(x.copied != refuse_x);
	(void)queue.Push(std::move(x));
	for (auto& job: held) {
		queue.Recycle(std::move(job));
	}
	ScanJob job;
	CHECK(queue.TryPop(job));
	RunScanJob(scanner, job, false, shadow, result.dispatches, result.stats);
	(void)queue.Push(queue.MakeCommands(3, Dispatch(1, 1, 1, 0)));
	while (result.dispatches.empty() && queue.TryPop(job)) {
		RunScanJob(scanner, job, false, shadow, result.dispatches, result.stats);
	}
	CHECK(result.dispatches.size() == 1);
	return result.dispatches.empty() || result.dispatches[0].uncertain_program;
}

// A list older than the drop that caused a doubt carries the doubt and does not use it up, so a program
// it writes cannot make the list after the drop certain. The baseline is the same flow where X gets a
// buffer and is walked, and there the dispatch is certain. A list that cleared the doubt on the way out
// would report the dispatch certain under a program the missing list may have changed.
void TestOlderListKeepsNewerDropDoubt() {
	CHECK(!DispatchAfterRefusedListIsUncertain(false));
	CHECK(DispatchAfterRefusedListIsUncertain(true));
}

} // namespace

// Counts the allocations of the tests, for TestNoQueueAllocationAfterWarmup.
void* operator new(std::size_t size) {
	if (g_count_allocations.load(std::memory_order_relaxed)) {
		g_allocation_count.fetch_add(1, std::memory_order_relaxed);
	}
	if (void* memory = std::malloc(size == 0 ? 1 : size)) {
		return memory;
	}
	throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
	std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
	std::free(memory);
}

int main() {
	TestDispatchBehindIndirectBuffer();
	TestWaveSizeFromInitiator();
	TestUserDataCountSeparatesKeys();
	TestThreadDimensionsRoundUp();
	TestUnreadableBufferSkipped();
	TestBadBufferAddressIsUncertain();
	TestBadPacketIsUncertain();
	TestShadowPersistsAndResets();
	TestSelfReferenceTerminates();
	TestCondExecCoversBothArms();
	TestBranchWalksBothArms();
	TestBranchModeOneNeverReadsElse();
	TestBranchFallsThroughToTail();
	TestElseArmWritesAreDropped();
	TestIndirectRegisterList();
	TestBudgetBoundsReads();
	TestAsyncQueueFlag();
	TestDepthSixFoundOnlyBeyondOldLimit();
	TestChainAddsNoDepth();
	TestSettledCacheKeepsCoverage();
	TestCacheProbesAtInterval();
	TestUncertainDispatchLeavesCacheAlone();
	TestManyBuffersGrowVisitedSet();
	TestVisitedSetKeepsProgramStatesApart();
	TestRevisitDoubtsOnlyAfterKeyStateWrites();
	TestTruncatedBufferEndsUncertain();
	TestSetBaseResolvesBaseForm();
	TestResetClearsDispatchBase();
	TestCondExecRegionMakesProgramUncertain();
	TestCondExecRegionEndingTheBufferIsUncertain();
	TestCondExecRegionJudgesCalledBuffers();
	TestPredicatedPacketsOnlyDoubtWrites();
	TestOtherDoubtsMakeProgramUncertain();
	TestUncertainDispatchIsNeverCached();
	TestTimeCapStopsSlowReader();
	TestTimeCapChecksClockWhileWalking();
	TestDispatchStorageIsReused();
	TestNoAllocationAfterWarmup();
	TestBusyPostIsRememberedAndProbed();
	TestRaisedBudgetScansLongListCertain();
	TestArenaStopsGrowingAfterWarmup();
	TestVisitedSetResetKeepsOldEntriesHidden();
	TestQueueDropsOldestCommands();
	TestResetMarkersNeverDropped();
	TestDropPrefersExecutedJobs();
	TestWatermarkSkipsLateJobs();
	TestDropMarksShadowUncertain();
	TestCopyIsCapped();
	TestTruncatedCopyLeavesShadowUncertain();
	TestBuffersFitBySizeClass();
	TestStoppedQueueCopiesNothing();
	TestNoBufferDropsInsteadOfAllocating();
	TestCopyTimeHistogram();
	TestQueuedScanMatchesDirectScan();
	TestPerQueueOrderPreserved();
	TestNoQueueAllocationAfterWarmup();
	TestQueuedLongListScansCertain();
	TestArenaSliceReleasesOnEveryExit();
	TestMarkerClearsOlderDoubt();
	TestBufferFallsBackToSmallerClass();
	TestMarkerLossIsCountedAndLeavesDoubt();
	TestScanUsesCallersScratch();
	TestOlderListKeepsNewerDropDoubt();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("SubmitScanTests passed");
	return 0;
}
