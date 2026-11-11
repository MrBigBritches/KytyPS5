#pragma once

#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>
#include <unordered_map>

namespace Libs::Graphics::ShaderRecompiler::IR {

// Counts what the exec-select passes saw and changed. Elision is added by a later commit.
struct ExecSelectStats {
	uint32_t candidates = 0; // selects whose false operand is another select of the same opcode
	uint32_t forwarded  = 0; // selects whose false operand moved down at least one link
	uint32_t elided     = 0;
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
};

// Select(c, x, Select(c2, y, z)) becomes Select(c, x, z) when c2 is false wherever c is false and
// both selects have the same opcode. This is exact in every lane. Repeats down a chain. It leaves
// the bypassed selects in place for dead code elimination.
[[nodiscard]] ExecSelectStats ForwardSubsetSelects(Program& program);

} // namespace Libs::Graphics::ShaderRecompiler::IR
