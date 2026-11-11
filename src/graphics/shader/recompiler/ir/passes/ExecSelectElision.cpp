#include "graphics/shader/recompiler/ir/passes/ExecSelectElision.h"

#include <algorithm>
#include <functional>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr size_t   SelectConditionArg  = 0;
constexpr size_t   SelectFalseValueArg = 2;
constexpr size_t   AndOperandCount     = 2;
constexpr size_t   HashGoldenRatio     = 0x9e3779b97f4a7c15ull;
constexpr unsigned HashShiftLeft       = 6u;
constexpr unsigned HashShiftRight      = 2u;

// True for every opcode that picks between two values by a U1 condition.
bool IsSelect(ValueOpcode opcode) {
	return opcode == ValueOpcode::SelectU32 || opcode == ValueOpcode::SelectF32 ||
	       opcode == ValueOpcode::SelectF64 || opcode == ValueOpcode::SelectU1;
}

// True when the value is the U1 immediate with the given truth value.
bool IsImmediateBool(const Value& value, bool expected) {
	return value.IsImmediate() && value.GetType() == Type::U1 && value.U1() == expected;
}

// Follows the false operand while the inner select is only reachable where the outer one is false.
Value FollowFalseChain(const Inst& outer, MaskImplication& implication) {
	const auto condition = outer.Arg(SelectConditionArg);
	auto       falseArm  = outer.Arg(SelectFalseValueArg).Resolve();
	for (;;) {
		const auto* inner = falseArm.TryInstruction();
		// Parent() is null for retained resource-planning instructions, which are not rewritten.
		if (inner == nullptr || inner->GetOpcode() != outer.GetOpcode() ||
		    inner->Parent() == nullptr)
			return falseArm;
		// The caller's implication object is reused so repeated pairs are answered from the memo.
		if (!implication.Implies(inner->Arg(SelectConditionArg), condition)) return falseArm;
		falseArm = inner->Arg(SelectFalseValueArg).Resolve();
	}
}

} // namespace

// Combines the two instruction pointers into one hash, in the style of samplerCache.h.
size_t MaskImplication::KeyHash::operator()(const Key& key) const noexcept {
	const auto first  = std::hash<const Inst*> {}(key.subset);
	const auto second = std::hash<const Inst*> {}(key.superset);
	// The golden ratio constant spreads pointer bits.
	return first ^ (second + static_cast<size_t>(HashGoldenRatio) + (first << HashShiftLeft) +
	                (first >> HashShiftRight));
}

bool MaskImplication::Implies(Value subset, Value superset) {
	return LevelsNeeded(subset, superset, MaxMaskImplicationDepth) >= 0;
}

int MaskImplication::LevelsNeeded(Value subset, Value superset, int budget) {
	subset   = subset.Resolve();
	superset = superset.Resolve();
	if (IsImmediateBool(superset, true) || IsImmediateBool(subset, false) || subset == superset)
		return 0;
	const auto* conjunction = subset.TryInstruction();
	if (conjunction == nullptr || superset.TryInstruction() == nullptr ||
	    conjunction->GetOpcode() != ValueOpcode::LogicalAnd || budget <= 0)
		return -1;
	const Key  key {conjunction, superset.TryInstruction()};
	const auto found = m_memo.find(key);
	Entry      entry = found != m_memo.end() ? found->second : Entry {};
	if (entry.needed >= 0) return entry.needed <= budget ? entry.needed : -1;
	if (entry.failed_budget >= budget) return -1;
	const int needed = LevelsThroughAnd(*conjunction, superset, budget);
	// The recursion may add entries, so the pair is written back by key and not through a held
	// reference.
	const auto latest = m_memo.find(key);
	if (latest != m_memo.end()) entry = latest->second;
	if (needed >= 0)
		entry.needed = needed;
	else
		entry.failed_budget = std::max(entry.failed_budget, budget);
	m_memo.insert_or_assign(key, entry);
	return needed;
}

// Fewest levels needed through either operand of a LogicalAnd, or -1 when neither operand proves
// it.
int MaskImplication::LevelsThroughAnd(const Inst& conjunction, Value superset, int budget) {
	// An And is false wherever either operand is false, so one operand under the superset is
	// enough. Both are searched so the fewest levels is found and the answer does not depend on
	// order.
	int best = -1;
	for (size_t operand = 0; operand < AndOperandCount; operand++) {
		const int below = LevelsNeeded(conjunction.Arg(operand), superset, budget - 1);
		if (below >= 0 && (best < 0 || below + 1 < best)) best = below + 1;
	}
	return best;
}

ExecSelectStats ForwardSubsetSelects(Program& program) {
	ExecSelectStats stats;
	MaskImplication implication;
	for (auto* block: program.blocks) {
		for (auto& inst: *block) {
			if (!IsSelect(inst.GetOpcode())) continue;
			const auto  falseArm = inst.Arg(SelectFalseValueArg).Resolve();
			const auto* first    = falseArm.TryInstruction();
			if (first == nullptr || first->GetOpcode() != inst.GetOpcode()) continue;
			stats.candidates++;
			const auto target = FollowFalseChain(inst, implication);
			if (target == falseArm) continue;
			inst.SetArg(SelectFalseValueArg, target);
			stats.forwarded++;
		}
	}
	return stats;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
