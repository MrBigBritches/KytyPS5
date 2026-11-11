#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Counts what the exec-select passes saw and changed. Each pass fills the fields that apply to it.
struct ExecSelectStats {
	// Forwarding counts selects whose false operand is another select of the same opcode. Elision
	// counts every select it looked at.
	uint32_t candidates         = 0;
	uint32_t forwarded          = 0; // selects whose false operand moved down at least one link
	uint32_t elided             = 0; // selects replaced by their true operand
	uint32_t depth_cap_hits     = 0; // implication queries that ran into the And depth cap
	uint32_t demand_truncations = 0; // conjuncts the demand analysis dropped for lack of room
};

// Most LogicalAnd levels one implication query descends. It bounds the work per query and, with
// memoisation, the work of a whole pass.
inline constexpr int MaxMaskImplicationDepth = 32;

// Decides whether one lane mask is false wherever another is false, using only operand edges that
// are exact in every lane. Results are memoised, so one object should serve a whole pass.
// The memo is keyed on instruction pointers and the operands behind them. One instance must not
// live across dead code elimination or any change to LogicalAnd operands, and every query must
// finish before any replacement is made.
class MaskImplication {
public:
	// True when subset is known to be false wherever superset is false.
	// Phi, SelectU1, LogicalNot, thread-bit compares and ConditionRef are never looked through,
	// because none of them keeps the same lane meaning across that edge.
	[[nodiscard]] bool Implies(Value subset, Value superset);

	// Number of Implies queries whose search reached the depth cap. Memoised answers do not count
	// again, so this is a lower bound.
	[[nodiscard]] uint32_t DepthCapHits() const { return m_cap_hits; }

private:
	struct Key {
		const Inst* subset;
		const Inst* superset;
		bool        operator==(const Key&) const = default;
	};
	struct KeyHash {
		size_t operator()(const Key& key) const noexcept;
	};
	// What is known about one pair. needed is the exact number of And levels a proof takes, or -1
	// while no proof is known. failed_budget is the largest budget that was searched without one.
	// Keeping both makes every answer depend only on the pair and the budget, never on query order.
	struct Entry {
		int needed        = -1;
		int failed_budget = -1;
	};

	// Returns the fewest And levels needed to prove the implication within budget, or -1.
	[[nodiscard]] int LevelsNeeded(Value subset, Value superset, int budget);
	[[nodiscard]] int LevelsThroughAnd(const Inst& conjunction, Value superset, int budget);

	std::unordered_map<Key, Entry, KeyHash> m_memo;
	bool                                    m_hit_cap  = false;
	uint32_t                                m_cap_hits = 0;
};

// Select(c, x, Select(c2, y, z)) becomes Select(c, x, z) when c2 is false wherever c is false and
// both selects have the same opcode. This is exact in every lane. Repeats down a chain. It leaves
// the bypassed selects in place for dead code elimination.
[[nodiscard]] ExecSelectStats ForwardSubsetSelects(Program& program);

// How the demand analysis treats an opcode. Every opcode belongs to exactly one class.
enum class ExecOpClass : uint8_t {
	Pure,     // each lane of the result depends only on the same lane of the operands
	Select,   // picks between two values by a U1 condition
	And,      // LogicalAnd
	Phi,      // merges values from predecessor blocks
	MemData,  // masked store or atomic, the data operands matter only under the trailing mask
	Lane,     // reads other lanes, or reads image state, so every lane of its operands matters
	Load,     // masked load, address operands matter in every lane
	Unlisted, // anything else, which must be assumed to observe every lane of every operand
};

// Class of an opcode. A value outside the table is Unlisted, which is the safe answer.
[[nodiscard]] ExecOpClass ClassifyExecOpcode(ValueOpcode opcode);

// How many classification rules name the opcode. A test requires exactly one for every opcode, so a
// new opcode must be classified on purpose.
[[nodiscard]] size_t CountExecOpcodeRules(ValueOpcode opcode);

// Replaces Select(c, new, old) by new wherever every lane that can observe the result has c true.
// A backward analysis finds, for each value, the lanes that observe it. Only compute shaders are
// changed. Every query finishes before any replacement is made, and replacement chains are
// resolved first. The replaced selects are left as dead instructions for dead code elimination.
[[nodiscard]] ExecSelectStats ElideUnobservedSelects(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR
