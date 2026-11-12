#include "graphics/guest_gpu/command_processor/submitScan.h"

#include "graphics/guest_gpu/csRegisters.h"
#include "graphics/guest_gpu/pm4.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// The compute user data slots the processor accepts, which HwShSetCsUserSgpr bounds the same way.
constexpr uint32_t CS_USER_DATA_SLOTS = 16;
// Bits of a register offset that select how an indirect pair names it, which the processor strips.
constexpr uint32_t REGISTER_SELECTOR_MASK = 0x70000000u;
// The sentinel an indirect register list uses for an unused pair.
constexpr uint32_t INDIRECT_PAIR_SENTINEL = 0xffffffffu;
// How many register pairs one read of an indirect register list fetches.
constexpr uint32_t INDIRECT_PAIRS_PER_READ = 256;
// Packet lengths, counting the header, that the processor accepts for these packets.
constexpr uint32_t DISPATCH_DIRECT_LEN        = 5;
constexpr uint32_t DISPATCH_INDIRECT_BASE_LEN = 3;
constexpr uint32_t DISPATCH_INDIRECT_ADDR_LEN = 4;
constexpr uint32_t INDIRECT_BUFFER_LEN        = 4;
constexpr uint32_t BRANCH_LEN                 = 14;
constexpr uint32_t SET_SH_REG_INDIRECT_LEN    = 5;
constexpr uint32_t SET_BASE_LEN               = 4;
// A COND_EXEC needs its four payload words before the skip count can be read.
constexpr uint32_t COND_EXEC_MIN_LEN = 5;
// The type 2 filler word, which the processor steps over as one dword.
constexpr uint32_t FILLER_PACKET = 0x80000000u;
// The processor's reset packet, a custom code carried by IT_NOP.
constexpr uint32_t DISPATCH_RESET_PACKET = 0xC0001024u;
// The mask of the dword count of an indirect buffer, and the bit that chains rather than calls.
constexpr uint32_t IB_SIZE_MASK   = 0xfffffu;
constexpr uint32_t IB_CHAIN_BIT   = 20;
constexpr uint32_t PM4_TYPE_SHIFT = 30;
constexpr uint32_t PM4_TYPE_3     = 3;
// Bit 0 of a type 3 header marks a packet the processor skips while a predicate says so.
constexpr uint32_t PM4_PREDICATE_BIT = 1;
// IT_SET_BASE. The index must be the dispatch arguments slot and the shader type compute, and the
// processor reads the address from the second and third payload words.
constexpr uint32_t SET_BASE_INDEX_MASK    = 0xfu;
constexpr uint32_t SET_BASE_INDEX_ARGS    = 1;
constexpr uint32_t SET_BASE_TYPE_SHIFT    = 1;
constexpr uint32_t SET_BASE_TYPE_MASK     = 0x3u;
constexpr uint32_t SET_BASE_TYPE_COMPUTE  = 1;
constexpr uint64_t SET_BASE_LOW_MASK      = ~uint64_t {7};
constexpr uint32_t SET_BASE_HIGH_MASK     = 0xffffu;
constexpr uint32_t SET_BASE_HIGH_SHIFT    = 32;
// COND_EXEC keeps how many dwords it may skip in the low bits of its fourth payload word.
constexpr uint32_t COND_EXEC_COUNT_INDEX = 3;
constexpr uint32_t COND_EXEC_COUNT_MASK  = 0x3fffu;
// BRANCH names its mode in the low bits of its first payload word. Mode 1 runs the then buffer or
// nothing, and mode 2 runs the then buffer or the else buffer.
constexpr uint32_t BRANCH_MODE_MASK      = 0x3u;
constexpr uint32_t BRANCH_MODE_THEN_ONLY = 1;
// Where a branch packet keeps its then buffer (address low and high, dword count) and its else buffer,
// as indexes into the payload.
constexpr uint32_t BRANCH_THEN_ADDRESS_LOW  = 7;
constexpr uint32_t BRANCH_THEN_ADDRESS_HIGH = 8;
constexpr uint32_t BRANCH_THEN_DWORDS       = 9;
constexpr uint32_t BRANCH_ELSE_ADDRESS_LOW  = 10;
constexpr uint32_t BRANCH_ELSE_ADDRESS_HIGH = 11;
constexpr uint32_t BRANCH_ELSE_DWORDS       = 12;
// Guest addresses in packets are dword aligned, and the low two bits of the word hold flags.
constexpr uint32_t ADDRESS_ALIGN_MASK  = 0xfffffffcu;
constexpr uint32_t ADDRESS_HIGH_SHIFT  = 32;
// The mask of the pair count in an indirect register list packet.
constexpr uint32_t INDIRECT_PAIR_COUNT_MASK = 0x3fffu;
// The opcode of a type 3 header.
constexpr uint32_t PM4_OPCODE_SHIFT = 8;
constexpr uint32_t PM4_OPCODE_MASK  = 0xffu;
// A dispatch has three axes.
constexpr uint32_t DISPATCH_AXES = 3;

// How a walk of one buffer ended.
enum class WalkEnd : uint8_t {
	// The buffer ran out of packets.
	Finished,
	// A chain replaced the buffer, so the caller's packets after it never run.
	Chained,
	// The budget is spent, so no caller goes on.
	Budget,
};

// The finishing step of splitmix64, which spreads every input bit over every output bit.
uint64_t Scramble(uint64_t value) {
	value ^= value >> 30u;
	value *= 0xbf58476d1ce4e5b9ull;
	value ^= value >> 27u;
	value *= 0x94d049bb133111ebull;
	value ^= value >> 31u;
	return value;
}

// The buffers a walk has met, as a flat open-addressing table so that a walk of thousands of buffers
// allocates nothing once the table has grown. Zero marks an empty slot, so a zero key is stored as two.
// It is reset for each walk. Keys are already hashes, and the slot is chosen from their scrambled bits.
class VisitedSet {
public:
	/// Adds the key. False when it was already there, and then `first_walk_clean` says whether the walk
	/// that added it has finished and left the compute key state untouched. It is false while that walk
	/// is still going, as when a buffer calls itself.
	bool Insert(uint64_t key, bool& first_walk_clean) {
		key = Normalize(key);
		if (m_slots.empty()) {
			m_slots.assign(INITIAL_SLOTS, 0);
			m_growths++;
		}
		if ((m_count + 1) * 2 > m_slots.size()) {
			Grow();
		}
		const auto mask  = m_slots.size() - 1;
		auto       index = Scramble(key) & mask;
		while (m_slots[index] != 0) {
			if ((m_slots[index] & ~CLEAN_BIT) == key) {
				first_walk_clean = (m_slots[index] & CLEAN_BIT) != 0;
				return false;
			}
			index = (index + 1) & mask;
		}
		m_slots[index] = key;
		m_count++;
		return true;
	}

	/// Records that the walk which added the key finished without touching the key state. The bit lives
	/// in the slot itself, so it needs no storage of its own and no allocation.
	void MarkClean(uint64_t key) {
		key        = Normalize(key);
		const auto mask = m_slots.size() - 1;
		auto       index = Scramble(key) & mask;
		while (m_slots[index] != 0) {
			if ((m_slots[index] & ~CLEAN_BIT) == key) {
				m_slots[index] |= CLEAN_BIT;
				return;
			}
			index = (index + 1) & mask;
		}
	}

	/// Empties the set for the next walk. A table that was never used is not touched, and one that was
	/// is cleared whole, which is a small memset at the sizes a walk reaches.
	void Reset() {
		if (m_count != 0) {
			std::fill(m_slots.begin(), m_slots.end(), 0);
			m_count = 0;
		}
	}

	/// How many times the table was allocated or grown since this was last asked, then forgotten.
	uint64_t TakeGrowths() { return std::exchange(m_growths, 0); }

private:
	static constexpr size_t   INITIAL_SLOTS = 256;
	// The lowest bit of a stored key is the clean mark, so keys are compared without it.
	static constexpr uint64_t CLEAN_BIT = 1;

	// Clears the mark bit, and maps zero, which marks an empty slot, to a nonzero key.
	static uint64_t Normalize(uint64_t key) {
		key &= ~CLEAN_BIT;
		return key == 0 ? 2 : key;
	}

	// Doubles the table and places every key again with its mark, which is counted as an allocation.
	void Grow() {
		std::vector<uint64_t> old(m_slots.size() * 2, 0);
		old.swap(m_slots);
		m_growths++;
		const auto mask = m_slots.size() - 1;
		for (const auto stored: old) {
			if (stored == 0) {
				continue;
			}
			auto index = Scramble(stored & ~CLEAN_BIT) & mask;
			while (m_slots[index] != 0) {
				index = (index + 1) & mask;
			}
			m_slots[index] = stored;
		}
	}

	std::vector<uint64_t> m_slots;
	size_t                m_count   = 0;
	uint64_t              m_growths = 0;
};

// The most buffers that may be walked inside each other, chains included. A chain adds no nesting for
// the depth limit, but each one is a frame of the walk, so a run of them is bounded here.
constexpr uint32_t MAX_WALK_LEVELS = 256;

// The buffers a walk reads indirect buffers into, one per level of the walk. They are kept for the
// thread's life so that a walk of hundreds of small buffers allocates nothing after the first, and
// are left uninitialized, since the read fills them.
class ScratchPool {
public:
	/// A buffer of at least `dwords` dwords for the given level of the walk.
	uint32_t* Get(size_t level, size_t dwords) {
		if (m_levels.empty()) {
			m_levels.reserve(MAX_WALK_LEVELS);
			m_growths++;
		}
		if (m_levels.size() <= level) {
			m_levels.resize(level + 1u);
		}
		auto& buffer = m_levels[level];
		if (dwords > buffer.capacity) {
			buffer.capacity = std::max<size_t>(dwords, buffer.capacity * 2);
			buffer.data     = std::make_unique_for_overwrite<uint32_t[]>(buffer.capacity);
			m_growths++;
		}
		return buffer.data.get();
	}

	/// How many times a buffer was allocated or grown since this was last asked, then forgotten.
	uint64_t TakeGrowths() { return std::exchange(m_growths, 0); }

private:
	struct Buffer {
		std::unique_ptr<uint32_t[]> data;
		size_t                      capacity = 0;
	};

	std::vector<Buffer> m_levels;
	uint64_t            m_growths = 0;
};

// The state of one submission's walk.
struct Walk {
	const ScanMemoryReader&       reader;
	const ScanLimits&             limits;
	CsShadow&                     shadow;
	std::vector<ScannedDispatch>& out;
	ScanStats&                    stats;
	uint32_t                      queue;
	bool                          async_queue;
	uint32_t                      dwords_left;
	VisitedSet&                   visited;
	ScratchPool&                  scratch;
	DispatchCache*                cache;
	// How many buffers are being walked inside each other, chains included, which indexes the scratch
	// buffer the next one is read into.
	uint32_t level = 0;
	// The time the walk must end by, how many packets passed since the clock was read, and whether the
	// cap has been hit, after which every loop ends at its next look.
	std::chrono::steady_clock::time_point deadline;
	uint32_t                              packets_since_clock = 0;
	bool                                  out_of_time         = false;
};

// Stops the walk vouching for the program and counts the cause.
void NoteUncertain(Walk& walk, UncertainCause cause) {
	walk.stats.uncertain_cleared_by[static_cast<size_t>(cause)]++;
	walk.shadow.MarkUncertain();
}

// Counts a cut caused by the dword budget. What the walk did not read may have set the program, so
// the shadow can no longer vouch for it.
void NoteBudgetCut(Walk& walk) {
	walk.stats.budget_exhausted++;
	NoteUncertain(walk, UncertainCause::BudgetCut);
}

// Writes one register of a SET_SH_REG packet into the shadow the way the processor would.
void ApplyRegister(CsShadow& shadow, uint32_t reg, uint32_t value) {
	if (reg >= Pm4::COMPUTE_USER_DATA_0 && reg < Pm4::COMPUTE_USER_DATA_0 + CS_USER_DATA_SLOTS) {
		shadow.regs.cs_user_sgpr.value[reg - Pm4::COMPUTE_USER_DATA_0] = value;
		return;
	}
	// A register the compute decoder does not know is another stage's, and the walk does not care.
	if (!DecodeCsRegister(shadow.regs.cs_regs, reg, value)) {
		return;
	}
	shadow.key_state_dirty = true;
	if (reg == Pm4::COMPUTE_PGM_LO || reg == Pm4::COMPUTE_PGM_HI) {
		shadow.program_set_in_submission = true;
		shadow.NoteProgramWrite(reg == Pm4::COMPUTE_PGM_HI);
	}
}

// Hashes the registers that make up a compute program's state, which is what decides its module. The
// address comes first. The user data is left out because it differs on nearly every dispatch of the
// same program, but how many registers of it the program reads is part of the state.
uint64_t ProgramStateHash(const HW::CsStageRegisters& cs) {
	const uint32_t words[] = {
	    static_cast<uint32_t>(cs.data_addr),
	    static_cast<uint32_t>(cs.data_addr >> ADDRESS_HIGH_SHIFT),
	    cs.user_sgpr,
	    cs.float_mode,
	    cs.num_thread_x,
	    cs.num_thread_y,
	    cs.num_thread_z,
	    cs.lds_size,
	    cs.tidig_comp_cnt,
	    cs.wave_size,
	    static_cast<uint32_t>(cs.tgid_x_en) | (static_cast<uint32_t>(cs.tgid_y_en) << 1u) |
	        (static_cast<uint32_t>(cs.tgid_z_en) << 2u) | (static_cast<uint32_t>(cs.tg_size_en) << 3u) |
	        (static_cast<uint32_t>(cs.scratch_en) << 4u),
	};
	return XXH3_64bits(words, sizeof(words));
}

// Hashes what decides a dispatch's static key, which is its program state and how it is launched.
uint64_t KeyHash(const HW::CsStageRegisters& cs, bool thread_dimensions, bool async_queue) {
	const uint64_t words[] = {
	    ProgramStateHash(cs),
	    static_cast<uint64_t>(thread_dimensions) | (static_cast<uint64_t>(async_queue) << 1u),
	};
	return XXH3_64bits(words, sizeof(words));
}

// Turns the counts of a direct dispatch into group counts, as the runtime derives them.
void FillWorkgroupCounts(ScannedDispatch& dispatch, const uint32_t* counts) {
	const auto& cs = dispatch.regs.cs_regs;
	const uint32_t group_sizes[DISPATCH_AXES] = {cs.num_thread_x, cs.num_thread_y, cs.num_thread_z};
	for (uint32_t axis = 0; axis < DISPATCH_AXES; ++axis) {
		uint32_t value = counts[axis];
		if (dispatch.thread_dimensions) {
			// Threads become groups the way the runtime rounds them up.
			const auto size = std::max(group_sizes[axis], 1u);
			value           = value / size + (value % size != 0u);
		}
		dispatch.workgroup_counts[axis] = value;
	}
}

// Asks the settled-dispatch cache about a dispatch. Returns true when it is skipped. A program the
// caller has settled is not copied or posted again, except for the one dispatch in a while that the
// cache lets through to probe for a new specialization, which sets `probe`. A dispatch whose program
// is uncertain does not ask, so it cannot use up a probe that a certain dispatch is owed.
bool SkippedByCache(Walk& walk, uint64_t key_hash, bool& probe) {
	if (walk.cache == nullptr || !walk.shadow.certain) {
		return false;
	}
	switch (walk.cache->Check(key_hash)) {
		case DispatchCache::Verdict::Skip: walk.stats.dispatches_cached++; return true;
		case DispatchCache::Verdict::Probe: probe = true; return false;
		case DispatchCache::Verdict::Pass: break;
	}
	return false;
}

// Adds the dispatch a packet makes, or counts it as having no shader. The wave size is set first, as
// CommandProcessor::DispatchDirect sets it, so it persists in the shadow like the real state does.
// `counts` holds the group counts of a direct dispatch and is null for an indirect one.
void RecordDispatch(Walk& walk, uint32_t initiator, const uint32_t* counts, uint64_t args_address) {
	walk.stats.dispatches++;
	auto& regs             = walk.shadow.regs;
	regs.cs_regs.wave_size = Pm4::ComputeWaveSize(initiator);
	if (regs.cs_regs.data_addr == 0) {
		walk.stats.no_shader_state++;
		return;
	}
	if (!walk.shadow.program_set_in_submission) {
		walk.stats.program_inherited++;
	}
	if (!walk.shadow.certain) {
		walk.stats.uncertain_program++;
	}
	const bool thread_dimensions =
	    (initiator & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	const uint64_t key_hash = KeyHash(regs.cs_regs, thread_dimensions, walk.async_queue);
	bool           probe    = false;
	if (SkippedByCache(walk, key_hash, probe)) {
		return;
	}
	ScannedDispatch dispatch;
	dispatch.program_address       = regs.cs_regs.data_addr;
	dispatch.regs                  = regs;
	dispatch.initiator             = initiator;
	dispatch.thread_dimensions     = thread_dimensions;
	dispatch.force_probe           = probe;
	dispatch.indirect              = counts == nullptr;
	dispatch.async_queue           = walk.async_queue;
	dispatch.uncertain_program     = !walk.shadow.certain;
	dispatch.queue                 = walk.queue;
	dispatch.indirect_args_address = args_address;
	dispatch.key_hash              = key_hash;
	if (counts != nullptr) {
		FillWorkgroupCounts(dispatch, counts);
	}
	walk.out.push_back(std::move(dispatch));
}

// Whether the walk has used its time. The clock is read only when `force` is set, as before a read,
// or after enough packets, so the check costs almost nothing on a quiet walk. Once set it stays set.
bool OutOfTime(Walk& walk, bool force) {
	if (walk.out_of_time) {
		return true;
	}
	if (!force && ++walk.packets_since_clock < SCAN_CLOCK_CHECK_PACKETS) {
		return false;
	}
	walk.packets_since_clock = 0;
	if (std::chrono::steady_clock::now() >= walk.deadline) {
		walk.out_of_time = true;
		walk.stats.time_capped++;
		// What the walk did not reach may have changed the program.
		NoteUncertain(walk, UncertainCause::TimeCap);
	}
	return walk.out_of_time;
}

WalkEnd WalkBuffer(Walk& walk, std::span<const uint32_t> buffer, uint32_t depth, bool truncated);

// Whether a buffer the walk is about to follow is one it should not read at all, and counts why.
// The key holds the address, the size and the program state, so the same buffer called under two
// different programs is walked under both, since what it dispatches depends on the program.
// The key it entered in the visited set is left in `visit_key`.
bool ShouldSkipBuffer(Walk& walk, uint64_t address, uint32_t dword_count, uint32_t depth, bool chain,
                      uint64_t& visit_key) {
	if (address == 0 || (address & 3u) != 0) {
		walk.stats.ib_unreadable++;
		NoteUncertain(walk, UncertainCause::UnreadableBuffer);
		return true;
	}
	// A chain replaces the buffer it is in, as the processor does, so it adds no nesting. The visited
	// set and the budget bound a long run of them.
	if ((!chain && depth >= walk.limits.max_depth) || walk.level >= MAX_WALK_LEVELS) {
		walk.stats.ib_depth_limited++;
		NoteUncertain(walk, UncertainCause::DepthLimited);
		return true;
	}
	// A buffer met twice under one program in one submission is not walked again, which also ends a
	// buffer that calls itself. The repeat leaves the program in doubt only if the first walk touched
	// the compute key state or has not finished, since a buffer that wrote nothing that matters would
	// change nothing the second time either.
	const uint64_t parts[] = {address, dword_count, ProgramStateHash(walk.shadow.regs.cs_regs)};
	visit_key              = XXH3_64bits(parts, sizeof(parts));
	bool first_walk_clean  = false;
	if (!walk.visited.Insert(visit_key, first_walk_clean)) {
		walk.stats.ib_revisited++;
		if (!first_walk_clean) {
			walk.stats.ib_revisit_doubts++;
			NoteUncertain(walk, UncertainCause::Revisited);
		}
		return true;
	}
	return false;
}

// Walks a buffer that was read into the scratch, and notes in the visited set whether the walk left
// the compute key state untouched. The dirty flag is judged on its own and then merged back, so that
// a region or predicate around the call still sees what the buffer did.
WalkEnd WalkFollowedBuffer(Walk& walk, std::span<const uint32_t> buffer, uint32_t depth,
                           bool truncated, uint64_t visit_key) {
	const bool outer_dirty          = walk.shadow.key_state_dirty;
	walk.shadow.key_state_dirty     = false;
	walk.level++;
	const auto end = WalkBuffer(walk, buffer, depth, truncated);
	walk.level--;
	if (!walk.shadow.key_state_dirty) {
		walk.visited.MarkClean(visit_key);
	}
	walk.shadow.key_state_dirty = walk.shadow.key_state_dirty || outer_dirty;
	return end;
}

// Reads an indirect buffer and walks it. Returns how the walk of the parent should go on.
WalkEnd FollowIndirectBuffer(Walk& walk, uint64_t address, uint32_t dword_count, uint32_t depth,
                             bool chain) {
	if (dword_count == 0) {
		return WalkEnd::Finished;
	}
	uint64_t visit_key = 0;
	if (ShouldSkipBuffer(walk, address, dword_count, depth, chain, visit_key)) {
		return WalkEnd::Finished;
	}
	if (walk.dwords_left == 0) {
		NoteBudgetCut(walk);
		return WalkEnd::Budget;
	}
	const uint32_t readable  = std::min(dword_count, walk.dwords_left);
	const bool     truncated = readable < dword_count;
	if (truncated) {
		NoteBudgetCut(walk);
	}
	uint32_t* copy = walk.scratch.Get(walk.level, readable);
	if (OutOfTime(walk, true)) {
		return WalkEnd::Budget;
	}
	if (!walk.reader(address, copy, uint64_t {readable} * sizeof(uint32_t))) {
		walk.stats.ib_unreadable++;
		NoteUncertain(walk, UncertainCause::UnreadableBuffer);
		return WalkEnd::Finished;
	}
	walk.dwords_left -= readable;
	walk.stats.dwords_read += readable;
	walk.stats.ib_followed++;
	const uint32_t next_depth = chain ? depth : depth + 1u;
	walk.stats.depth_max      = std::max<uint64_t>(walk.stats.depth_max, next_depth);
	return WalkFollowedBuffer(walk, std::span<const uint32_t>(copy, readable), next_depth, truncated,
	                          visit_key);
}

// Applies an indirect register list (SET_SH_REG_INDIRECT), which holds offset and value pairs in
// guest memory that the walk reads in small chunks.
void ApplyIndirectRegisters(Walk& walk, const uint32_t* payload) {
	const uint64_t address = (static_cast<uint64_t>(payload[1]) << ADDRESS_HIGH_SHIFT) |
	                         (payload[0] & ADDRESS_ALIGN_MASK);
	const uint32_t pairs = payload[3] & INDIRECT_PAIR_COUNT_MASK;
	uint32_t       done  = 0;
	std::array<uint32_t, INDIRECT_PAIRS_PER_READ * 2> chunk;
	while (done < pairs) {
		const uint32_t count = std::min(pairs - done, INDIRECT_PAIRS_PER_READ);
		if (walk.dwords_left < count * 2u) {
			NoteBudgetCut(walk);
			return;
		}
		if (OutOfTime(walk, true)) {
			return;
		}
		if (!walk.reader(address + uint64_t {done} * 8u, chunk.data(), uint64_t {count} * 8u)) {
			walk.stats.register_list_unreadable++;
			NoteUncertain(walk, UncertainCause::UnreadableRegisterList);
			return;
		}
		walk.dwords_left -= count * 2u;
		walk.stats.dwords_read += count * 2u;
		for (uint32_t pair = 0; pair < count; ++pair) {
			const uint32_t raw = chunk[pair * 2u];
			if (raw == INDIRECT_PAIR_SENTINEL) {
				continue;
			}
			ApplyRegister(walk.shadow, raw & ~REGISTER_SELECTOR_MASK, chunk[pair * 2u + 1u]);
		}
		done += count;
	}
}

// The two buffers a branch packet names.
struct BranchArms {
	uint64_t then_address;
	uint32_t then_dwords;
	uint64_t else_address;
	uint32_t else_dwords;
};

// Reads the buffers out of a branch packet's payload, as CpOpBranch does.
BranchArms ReadBranchArms(const uint32_t* payload) {
	const auto join = [&](uint32_t low_index, uint32_t high_index) {
		return (payload[low_index] & ADDRESS_ALIGN_MASK) |
		       (static_cast<uint64_t>(payload[high_index]) << ADDRESS_HIGH_SHIFT);
	};
	return {join(BRANCH_THEN_ADDRESS_LOW, BRANCH_THEN_ADDRESS_HIGH), payload[BRANCH_THEN_DWORDS] & IB_SIZE_MASK,
	        join(BRANCH_ELSE_ADDRESS_LOW, BRANCH_ELSE_ADDRESS_HIGH), payload[BRANCH_ELSE_DWORDS] & IB_SIZE_MASK};
}

// A branch whose not taken path falls through to the parent's next packets. The then arm is walked so
// that what it dispatches is found, and its register writes are dropped, since the parent's tail runs
// only when the then arm did not. The tail then runs under a program the walk cannot vouch for.
WalkEnd WalkBranchThenOrNothing(Walk& walk, const BranchArms& arms, uint32_t depth) {
	const CsShadow before = walk.shadow;
	const auto     end    = FollowIndirectBuffer(walk, arms.then_address, arms.then_dwords, depth, true);
	if (end == WalkEnd::Budget) {
		return end;
	}
	walk.shadow = before;
	NoteUncertain(walk, UncertainCause::Branch);
	return WalkEnd::Finished;
}

// A mode 2 branch with an else buffer, which always chains to one of the two arms, so the parent's
// tail never runs. The else arm is walked first and its register writes are dropped, since the then
// arm starts from the state the branch met. The state after the branch is the then arm's.
WalkEnd WalkBranchElseThenThen(Walk& walk, const BranchArms& arms, uint32_t depth) {
	const CsShadow before = walk.shadow;
	const auto end = FollowIndirectBuffer(walk, arms.else_address, arms.else_dwords, depth, true);
	if (end == WalkEnd::Budget) {
		return end;
	}
	walk.shadow    = before;
	const auto then_end = FollowIndirectBuffer(walk, arms.then_address, arms.then_dwords, depth, true);
	NoteUncertain(walk, UncertainCause::Branch);
	return then_end == WalkEnd::Budget ? then_end : WalkEnd::Chained;
}

// Follows a branch packet and counts it by mode. Mode 1 has no else arm, so its else words are never
// read, and it falls through when the then arm is not taken, as does mode 2 with an empty else.
WalkEnd HandleBranch(Walk& walk, const uint32_t* payload, uint32_t depth) {
	const BranchArms arms = ReadBranchArms(payload);
	if ((payload[0] & BRANCH_MODE_MASK) == BRANCH_MODE_THEN_ONLY) {
		walk.stats.branch_mode1++;
		return WalkBranchThenOrNothing(walk, arms, depth);
	}
	walk.stats.branch_mode2++;
	if (arms.else_dwords == 0) {
		walk.stats.branch_mode2_empty_else++;
		return WalkBranchThenOrNothing(walk, arms, depth);
	}
	return WalkBranchElseThenThen(walk, arms, depth);
}

// Follows an indirect buffer packet, which is a call or a chain, or a branch with two buffers.
// Returns whether the parent goes on.
WalkEnd HandleIndirectBuffer(Walk& walk, uint32_t length, const uint32_t* payload, uint32_t depth) {
	if (length == BRANCH_LEN) {
		return HandleBranch(walk, payload, depth);
	}
	if (length != INDIRECT_BUFFER_LEN) {
		return WalkEnd::Finished;
	}
	const uint64_t address = payload[0] | (static_cast<uint64_t>(payload[1]) << ADDRESS_HIGH_SHIFT);
	const uint32_t control = payload[2];
	const bool     chain   = ((control >> IB_CHAIN_BIT) & 1u) != 0;
	const auto end = FollowIndirectBuffer(walk, address, control & IB_SIZE_MASK, depth, chain);
	if (end == WalkEnd::Budget) {
		return end;
	}
	// A chain replaces the buffer being walked, so what follows it in the parent never runs.
	return chain ? WalkEnd::Chained : WalkEnd::Finished;
}

// Applies IT_SET_BASE, which sets where compute indirect arguments are, as CpOpSetBase does. Other
// base indexes and shader types name state the walk does not keep.
void ApplySetBase(CsShadow& shadow, uint32_t header, uint32_t length, const uint32_t* payload) {
	const uint32_t shader_type = (header >> SET_BASE_TYPE_SHIFT) & SET_BASE_TYPE_MASK;
	if (length != SET_BASE_LEN || (payload[0] & SET_BASE_INDEX_MASK) != SET_BASE_INDEX_ARGS ||
	    shader_type != SET_BASE_TYPE_COMPUTE) {
		return;
	}
	shadow.dispatch_args_base =
	    (payload[1] & SET_BASE_LOW_MASK) |
	    (static_cast<uint64_t>(payload[2] & SET_BASE_HIGH_MASK) << SET_BASE_HIGH_SHIFT);
}

// Records a DISPATCH_INDIRECT. The address form names its arguments. The base form names an offset
// from the base IT_SET_BASE last set, and where none was set the address stays 0, which the processor
// would refuse to run.
void HandleDispatchIndirect(Walk& walk, uint32_t length, const uint32_t* payload) {
	if (length == DISPATCH_INDIRECT_ADDR_LEN) {
		RecordDispatch(walk, payload[2], nullptr,
		               payload[0] | (static_cast<uint64_t>(payload[1]) << ADDRESS_HIGH_SHIFT));
	} else if (length == DISPATCH_INDIRECT_BASE_LEN) {
		const uint64_t base = walk.shadow.dispatch_args_base;
		RecordDispatch(walk, payload[1], nullptr, base == 0 ? 0 : base + payload[0]);
	}
}

// The opcode of a type 3 header.
uint32_t PacketOpcode(uint32_t header) {
	return (header >> PM4_OPCODE_SHIFT) & PM4_OPCODE_MASK;
}

// Applies one packet of a buffer. Returns Finished to go on with the next packet.
WalkEnd HandlePacket(Walk& walk, uint32_t header, uint32_t length, const uint32_t* payload,
                     uint32_t depth) {
	switch (PacketOpcode(header)) {
		case Pm4::IT_NOP:
			if ((header & ~PM4_PREDICATE_BIT) == DISPATCH_RESET_PACKET) {
				walk.shadow.Reset();
				// The reset changes every register, which a region or predicate may have skipped.
				walk.shadow.key_state_dirty = true;
				walk.stats.resets++;
			}
			break;
		case Pm4::IT_SET_SH_REG:
			if (length >= 3u && payload[0] != Pm4::SH_NOP) {
				for (uint32_t index = 0; index + 2u < length; ++index) {
					ApplyRegister(walk.shadow, payload[0] + index, payload[1u + index]);
				}
			}
			break;
		case Pm4::IT_SET_SH_REG_INDIRECT:
			if (length == SET_SH_REG_INDIRECT_LEN) {
				ApplyIndirectRegisters(walk, payload);
			}
			break;
		case Pm4::IT_SET_BASE: ApplySetBase(walk.shadow, header, length, payload); break;
		case Pm4::IT_DISPATCH_DIRECT:
			if (length == DISPATCH_DIRECT_LEN) {
				RecordDispatch(walk, payload[3], payload, 0);
			}
			break;
		case Pm4::IT_DISPATCH_INDIRECT: HandleDispatchIndirect(walk, length, payload); break;
		case Pm4::IT_INDIRECT_BUFFER: return HandleIndirectBuffer(walk, length, payload, depth);
		default: break;
	}
	return WalkEnd::Finished;
}

// Applies a packet, and when the packet is predicated and wrote compute state, stops vouching for the
// program, since the processor may have skipped the write. Only a write counts, so a predicated dispatch
// or user data write leaves the program certain. The packet's own effects are applied first.
WalkEnd RunPacket(Walk& walk, uint32_t header, uint32_t length, const uint32_t* payload,
                  uint32_t depth) {
	const bool predicated  = (header & PM4_PREDICATE_BIT) != 0;
	const bool outer_dirty = walk.shadow.key_state_dirty;
	if (predicated) {
		walk.shadow.key_state_dirty = false;
	}
	const auto end = HandlePacket(walk, header, length, payload, depth);
	if (predicated) {
		if (walk.shadow.key_state_dirty) {
			NoteUncertain(walk, UncertainCause::Predicated);
		}
		walk.shadow.key_state_dirty = walk.shadow.key_state_dirty || outer_dirty;
	}
	return end;
}

// The span a COND_EXEC may skip in the buffer being walked, as an end offset, with whether the shadow
// was already dirty when the span opened, so that only writes inside the span are judged at its end.
struct CondRegion {
	size_t end         = 0;
	bool   outer_dirty = false;
};

// Where the dwords a COND_EXEC may skip end, as an offset in the buffer being walked, or 0 when the
// packet is too short to say or skips nothing. A count that runs past the buffer ends at its end.
size_t CondExecRegionEnd(uint32_t length, const uint32_t* payload, size_t at, size_t buffer_size) {
	if (length < COND_EXEC_MIN_LEN) {
		return 0;
	}
	const size_t skipped = payload[COND_EXEC_COUNT_INDEX] & COND_EXEC_COUNT_MASK;
	return skipped == 0 ? 0 : std::min(buffer_size, at + length + skipped);
}

// Opens or extends the region of a COND_EXEC. Writes are judged from the outermost opening.
void OpenRegion(Walk& walk, CondRegion& region, size_t end) {
	if (end == 0) {
		return;
	}
	if (region.end == 0) {
		region.outer_dirty          = walk.shadow.key_state_dirty;
		walk.shadow.key_state_dirty = false;
	}
	region.end = std::max(region.end, end);
}

// Ends the region. The program is no longer certain if the region wrote compute state, since the
// condition may have skipped the writes.
void CloseRegion(Walk& walk, CondRegion& region) {
	if (walk.shadow.key_state_dirty) {
		NoteUncertain(walk, UncertainCause::CondExecRegion);
	}
	walk.shadow.key_state_dirty = walk.shadow.key_state_dirty || region.outer_dirty;
	region                      = {};
}

// Whether a packet that does not fit the rest of the buffer is only the cut end of a buffer that was
// read short of its size, which the budget already counted, and not a malformed packet.
bool IsCutOffPacket(uint32_t header, uint32_t length, size_t remaining, bool truncated) {
	return truncated && (header >> PM4_TYPE_SHIFT) == PM4_TYPE_3 && length > remaining;
}

// Walks the packets of one buffer. Both arms of a COND_EXEC region are walked, and when the walk leaves
// the region the program is no longer certain if the region wrote compute state. The region is left
// open if the walk ends inside it.
WalkEnd WalkPackets(Walk& walk, std::span<const uint32_t> buffer, uint32_t depth, bool truncated,
                    CondRegion& region) {
	size_t at = 0;
	while (at < buffer.size()) {
		// A look at the clock every few packets, so a long list that reads nothing is cut too.
		if (OutOfTime(walk, false)) {
			return WalkEnd::Budget;
		}
		if (region.end != 0 && at >= region.end) {
			CloseRegion(walk, region);
		}
		const uint32_t header = buffer[at];
		if (header == FILLER_PACKET || Pm4::PacketSizeDw(header) == 1u) {
			at++;
			continue;
		}
		const uint32_t length = KYTY_PM4_LEN(header);
		if ((header >> PM4_TYPE_SHIFT) != PM4_TYPE_3 || length > buffer.size() - at) {
			if (!IsCutOffPacket(header, length, buffer.size() - at, truncated)) {
				// A packet the processor would not run as written, so nothing after it can be trusted.
				walk.stats.bad_packets++;
				NoteUncertain(walk, UncertainCause::BadPacket);
			}
			return WalkEnd::Finished;
		}
		const uint32_t* payload = buffer.data() + at + 1u;
		if (PacketOpcode(header) == Pm4::IT_COND_EXEC) {
			OpenRegion(walk, region, CondExecRegionEnd(length, payload, at, buffer.size()));
		}
		const auto end = RunPacket(walk, header, length, payload, depth);
		if (end != WalkEnd::Finished) {
			return end;
		}
		at += length;
	}
	return WalkEnd::Finished;
}

// Walks the packets of one buffer, and closes a COND_EXEC region that was still open when the walk of
// the buffer ended. `truncated` says the buffer was read short of its size.
WalkEnd WalkBuffer(Walk& walk, std::span<const uint32_t> buffer, uint32_t depth, bool truncated) {
	CondRegion region;
	const auto end = WalkPackets(walk, buffer, depth, truncated, region);
	if (region.end != 0) {
		CloseRegion(walk, region);
	}
	// The part of the buffer that was not read may rewrite what the read part set, so the program is in
	// doubt at its end whatever the read part wrote. The budget cut itself was counted when it was made.
	if (truncated) {
		NoteUncertain(walk, UncertainCause::BudgetCut);
	}
	return end;
}

} // namespace

void SubmitScanner::Scan(std::span<const uint32_t> commands, uint32_t queue, bool async_queue,
                         CsShadow& shadow, std::vector<ScannedDispatch>& dispatches,
                         ScanStats& stats, DispatchCache* cache) const {
	stats.submissions++;
	shadow.program_set_in_submission = false;
	shadow.key_state_dirty           = false;
	// One table and one set of buffers per thread, so that no walk allocates after the thread's first
	// large one.
	thread_local VisitedSet  visited;
	thread_local ScratchPool scratch;
	visited.Reset();
	const uint32_t top_level =
	    static_cast<uint32_t>(std::min<size_t>(commands.size(), m_limits.dword_budget));
	Walk walk {.reader      = m_reader,
	           .limits      = m_limits,
	           .shadow      = shadow,
	           .out         = dispatches,
	           .stats       = stats,
	           .queue       = queue,
	           .async_queue = async_queue,
	           .dwords_left = m_limits.dword_budget - top_level,
	           .visited     = visited,
	           .scratch     = scratch,
	           .cache       = cache};
	const bool truncated = top_level < commands.size();
	if (truncated) {
		NoteBudgetCut(walk);
	}
	walk.deadline              = std::chrono::steady_clock::now() + m_limits.time_cap;
	const uint64_t read_before = stats.dwords_read;
	stats.dwords_read += top_level;
	const size_t capacity_before = dispatches.capacity();
	(void)WalkBuffer(walk, commands.first(top_level), 0, truncated);
	// A walk that outgrew the reserved dispatch storage allocated, which a reused, reserved vector never
	// does. The table and buffers it reuses count their own growth.
	if (dispatches.capacity() > capacity_before) {
		stats.scan_allocations++;
	}
	stats.scan_allocations += visited.TakeGrowths() + scratch.TakeGrowths();
	stats.dwords_max = std::max(stats.dwords_max, stats.dwords_read - read_before);
}

} // namespace Libs::Graphics
