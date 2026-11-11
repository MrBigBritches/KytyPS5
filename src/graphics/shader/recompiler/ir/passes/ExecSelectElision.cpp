#include "graphics/shader/recompiler/ir/passes/ExecSelectElision.h"

#include <algorithm>
#include <array>
#include <bit>
#include <functional>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler::IR {
namespace {

constexpr size_t   SelectConditionArg  = 0;
constexpr size_t   SelectTrueValueArg  = 1;
constexpr size_t   SelectFalseValueArg = 2;
constexpr size_t   AndOperandCount     = 2;
constexpr size_t   HashGoldenRatio     = 0x9e3779b97f4a7c15ull;
constexpr unsigned HashShiftLeft       = 6u;
constexpr unsigned HashShiftRight      = 2u;
constexpr unsigned HashFoldShift       = 32u;

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
	m_hit_cap          = false;
	const bool implied = LevelsNeeded(subset, superset, MaxMaskImplicationDepth) >= 0;
	if (m_hit_cap) m_cap_hits++;
	return implied;
}

int MaskImplication::LevelsNeeded(Value subset, Value superset, int budget) {
	subset   = subset.Resolve();
	superset = superset.Resolve();
	if (IsImmediateBool(superset, true) || IsImmediateBool(subset, false) || subset == superset)
		return 0;
	const auto* conjunction = subset.TryInstruction();
	if (conjunction == nullptr || superset.TryInstruction() == nullptr ||
	    conjunction->GetOpcode() != ValueOpcode::LogicalAnd)
		return -1;
	if (budget <= 0) {
		// A proof might exist one level further down, so this answer is only "not within the cap".
		m_hit_cap = true;
		return -1;
	}
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
	stats.depth_cap_hits = implication.DepthCapHits();
	return stats;
}

namespace {

// Rule table. The classes that decide what may be dropped (Pure, Select, And, Phi, MemData) name
// every opcode exactly, so a new opcode never lands in one by the accident of its spelling. The
// cautious classes (Lane, Load, Unlisted) match by prefix, because a wrong guess there only keeps
// more values. Each opcode must match exactly one rule.
constexpr std::string_view PureNames[] = {
    "Identity",
    "UndefU1",
    "UndefU32",
    "BitCastU16F16",
    "BitCastF16U16",
    "BitCastU32F32",
    "BitCastF32U32",
    "BitCastU64F64",
    "BitCastF64U64",
    "ConvertU16U32",
    "ConvertU32U16",
    "ConvertU8U32",
    "ConvertU32U8",
    "ConvertF32F16",
    "ConvertF16F32",
    "ConvertS32F32",
    "ConvertU32F32",
    "ConvertF32S32",
    "ConvertF64S32",
    "ConvertF32F64",
    "ConvertF64F32",
    "ConvertF32U32",
    "ConvertF64U32",
    "CompositeConstructU64",
    "CompositeConstructU32x2",
    "CompositeConstructU32x3",
    "CompositeConstructF32x2",
    "CompositeConstructU32x4",
    "CompositeExtractU64",
    "CompositeExtractU32x2",
    "CompositeExtractU32x3",
    "CompositeExtractU32x4",
    "PackHalf2x16",
    "PackSnorm2x16",
    "PackUnorm2x16",
    "PackFloat2x16Rtz",
    "FPAbs32",
    "FPNeg32",
    "FPSaturate32",
    "BitFieldInsert",
    "BitFieldUExtract",
    "BitFieldSExtract",
    "IAdd32",
    "IAdd64",
    "ISub32",
    "ISub64",
    "IMul32",
    "UDiv32",
    "SMulHi",
    "UMulHi",
    "IAbs32",
    "ShiftLeftLogical32",
    "ShiftLeftLogical64",
    "ShiftRightLogical32",
    "ShiftRightLogical64",
    "ShiftRightArithmetic32",
    "ShiftRightArithmetic64",
    "BitwiseAnd32",
    "BitwiseAnd64",
    "BitwiseOr32",
    "BitwiseXor32",
    "BitwiseNot32",
    "BitReverse32",
    "BitCount32",
    "FindUMsb32",
    "FindILsb32",
    "SMin32",
    "UMin32",
    "SMax32",
    "UMax32",
    "SMinTri32",
    "UMinTri32",
    "SMaxTri32",
    "UMaxTri32",
    "SMedTri32",
    "UMedTri32",
    "SLessThan32",
    "SLessThan64",
    "SLessThanEqual64",
    "ULessThanEqual64",
    "UGreaterThanEqual64",
    "ULessThan32",
    "ULessThan64",
    "IEqual32",
    "IEqual64",
    "SLessThanEqual32",
    "ULessThanEqual32",
    "SGreaterThan32",
    "UGreaterThan32",
    "UGreaterThan64",
    "INotEqual32",
    "INotEqual64",
    "SGreaterThanEqual32",
    "UGreaterThanEqual32",
    "LogicalOr",
    "LogicalXor",
    "LogicalNot",
    "FPOrdEqual32",
    "FPOrdEqual64",
    "FPOrdLessThanEqual64",
    "FPOrdGreaterThanEqual64",
    "FPOrdNotEqual64",
    "FPOrdLessThan64",
    "FPOrdGreaterThan64",
    "FPUnordEqual64",
    "FPUnordNotEqual64",
    "FPUnordLessThan64",
    "FPUnordGreaterThan64",
    "FPUnordLessThanEqual64",
    "FPUnordGreaterThanEqual64",
    "FPUnordEqual32",
    "FPOrdNotEqual32",
    "FPUnordNotEqual32",
    "FPOrdLessThan32",
    "FPUnordLessThan32",
    "FPOrdGreaterThan32",
    "FPUnordGreaterThan32",
    "FPOrdLessThanEqual32",
    "FPUnordLessThanEqual32",
    "FPOrdGreaterThanEqual32",
    "FPUnordGreaterThanEqual32",
    "FPIsNan32",
    "FPIsNan64",
    "FPCmpClass32",
    "FPCmpClass16",
    "FPAdd32",
    "FPSub32",
    "FPFma32",
    "FPMad32",
    "FPMul32",
    "FPAdd64",
    "FPMul64",
    "FPFma64",
    "FPRecip64",
    "FPMin32",
    "FPMax32",
    "FPMin64",
    "FPMax64",
    "FPMinTri32",
    "FPMaxTri32",
    "FPMedTri32",
    "FPRecip32",
    "FPRecipIFlag32",
    "FPRecipSqrt32",
    "FPSqrt",
    "FPSin",
    "FPCos",
    "FPExp2",
    "FPLog2",
    "FPLdexp",
    "FPRoundEven32",
    "FPFloor32",
    "FPCeil32",
    "FPTrunc32",
    "FPFloor64",
    "FPCeil64",
    "FPTrunc64",
    "FPFract32",
    "FPFract64",
    "LaneId",
};
constexpr std::string_view MemDataNames[] = {
    "StoreAddressU8",     "StoreAddressU16",        "StoreAddressU32",
    "StoreBufferU8",      "StoreBufferU16",         "StoreBufferU32",
    "StoreBufferU32x2",   "StoreBufferU32x3",       "StoreBufferU32x4",
    "BufferAtomicSwap32", "BufferAtomicCmpSwap32",  "BufferAtomicSwap64",
    "BufferAtomicIAdd32", "BufferAtomicIAdd64",     "BufferAtomicISub32",
    "BufferAtomicSMin32", "BufferAtomicUMin32",     "BufferAtomicUMin64",
    "BufferAtomicSMax32", "BufferAtomicUMax32",     "BufferAtomicAnd32",
    "BufferAtomicAnd64",  "BufferAtomicOr32",       "BufferAtomicOr64",
    "BufferAtomicXor32",  "BufferAtomicFMin32",     "BufferAtomicFMax32",
    "WriteSharedU8",      "WriteSharedU16",         "WriteSharedU32",
    "WriteSharedU32x2",   "WriteSharedU32x3",       "WriteSharedU32x4",
    "SharedAtomicFMin32", "SharedAtomicFMax32",     "SharedAtomicSwap32",
    "SharedAtomicIAdd32", "SharedAtomicIAdd64",     "SharedAtomicISub32",
    "SharedAtomicInc32",  "SharedAtomicDec32",      "SharedAtomicSMin32",
    "SharedAtomicUMin32", "SharedAtomicSMax32",     "SharedAtomicUMax32",
    "SharedAtomicUMax64", "SharedAtomicCmpSwap32",  "SharedAtomicCmpSwap64",
    "SharedAtomicAnd32",  "SharedAtomicOr32",       "SharedAtomicOr64",
    "SharedAtomicXor32",  "SharedAtomicMaskedOr32",
};
constexpr std::string_view SelectNames[]  = {"SelectU1", "SelectU32", "SelectF32", "SelectF64"};
constexpr std::string_view AndNames[]     = {"LogicalAnd"};
constexpr std::string_view PhiNames[]     = {"Phi"};
constexpr std::string_view LanePrefixes[] = {
    "Ballot",     "ReadFirstLane", "ReadLane", "WriteLane", "Subgroup",     "Permlane",
    "Permute",    "Bpermute",      "Swizzle",  "Dpp",       "ConditionRef", "Wqm",
    "DataAppend", "DataConsume",   "Image",    "Bvh",       "LiveMask"};
constexpr std::string_view LoadPrefixes[]     = {"LoadAddress", "LoadBuffer", "LoadShared",
                                                 "ReadConst"};
constexpr std::string_view UnlistedPrefixes[] = {"Get",
                                                 "Set",
                                                 "Reference",
                                                 "Void",
                                                 "Mesh",
                                                 "Tessellation",
                                                 "ReadRealtimeClock",
                                                 "Make",
                                                 "Control",
                                                 "StoreCompletion",
                                                 "DataCache",
                                                 "Barrier",
                                                 "Sendmsg",
                                                 "Ttrace",
                                                 "InstPrefetch"};

struct OpcodeRule {
	ExecOpClass                       op_class;
	std::span<const std::string_view> names;
	bool                              exact;
};

const OpcodeRule OpcodeRules[] = {
    {ExecOpClass::Pure, PureNames, true},       {ExecOpClass::Select, SelectNames, true},
    {ExecOpClass::And, AndNames, true},         {ExecOpClass::Phi, PhiNames, true},
    {ExecOpClass::MemData, MemDataNames, true}, {ExecOpClass::Lane, LanePrefixes, false},
    {ExecOpClass::Load, LoadPrefixes, false},   {ExecOpClass::Unlisted, UnlistedPrefixes, false}};

// True when the opcode name is one of the rule names, or starts with one for a prefix rule.
bool RuleMatches(const OpcodeRule& rule, std::string_view name) {
	return std::ranges::any_of(rule.names, [&](std::string_view entry) {
		return rule.exact ? name == entry : name.starts_with(entry);
	});
}

// Where the data operands of a masked store or atomic begin. Everything before them addresses
// memory, and the trailing operand is the mask.
size_t FirstDataOperand(std::string_view name) {
	constexpr size_t AddressDataBegin = 3;
	constexpr size_t BufferDataBegin  = 4;
	constexpr size_t SharedDataBegin  = 1;
	if (name.starts_with("StoreAddress")) return AddressDataBegin;
	if (name.starts_with("StoreBuffer") || name.starts_with("BufferAtomic")) return BufferDataBegin;
	return SharedDataBegin;
}

} // namespace

size_t CountExecOpcodeRules(ValueOpcode opcode) {
	if (opcode >= ValueOpcode::Count) return 0;
	const auto name = ValueOpcodeName(opcode);
	return static_cast<size_t>(std::ranges::count_if(
	    OpcodeRules, [&](const OpcodeRule& rule) { return RuleMatches(rule, name); }));
}

ExecOpClass ClassifyExecOpcode(ValueOpcode opcode) {
	if (opcode >= ValueOpcode::Count) return ExecOpClass::Unlisted;
	const auto name = ValueOpcodeName(opcode);
	for (const auto& rule: OpcodeRules)
		if (RuleMatches(rule, name)) return rule.op_class;
	return ExecOpClass::Unlisted;
}

namespace {

// ClassifyExecOpcode for every opcode, worked out once. The prefix search is too slow to repeat for
// every instruction visit.
ExecOpClass CachedClass(ValueOpcode opcode) {
	static const std::vector<ExecOpClass> table = [] {
		std::vector<ExecOpClass> classes(static_cast<size_t>(ValueOpcode::Count));
		for (size_t index = 0; index < classes.size(); index++)
			classes[index] = ClassifyExecOpcode(static_cast<ValueOpcode>(index));
		return classes;
	}();
	return table[static_cast<size_t>(opcode)];
}

// Which lanes can observe a value. None means no lane, All means any lane, and Mask means only
// the lanes where every conjunct is true. A mask is a short list of U1 values rather than one value
// because an intersection such as "d and c" is not itself an IR value.
enum class DemandKind : uint8_t { None, Mask, All };
constexpr size_t MaxDemandConjuncts = 4;

struct Demand {
	DemandKind                            kind = DemandKind::None;
	std::array<Inst*, MaxDemandConjuncts> conjuncts {};
	size_t                                count = 0;

	bool operator==(const Demand&) const = default;
};

// Demand of every lane.
Demand AllLanes() {
	Demand demand;
	demand.kind = DemandKind::All;
	return demand;
}

// Demand of the lanes where one U1 value is true.
Demand MaskedBy(Inst* mask) {
	Demand demand;
	demand.kind         = DemandKind::Mask;
	demand.conjuncts[0] = mask;
	demand.count        = 1;
	return demand;
}

// Dominator tree of the program blocks as nested DFS intervals, so a dominance query is two
// comparisons. Block 0 is the entry.
class DominatorIntervals {
public:
	explicit DominatorIntervals(const BlockList& blocks): m_count(blocks.size()) {
		for (size_t index = 0; index < m_count; index++)
			m_id[blocks[index]] = static_cast<uint32_t>(index);
		const auto order = ReversePostorder(blocks);
		const auto idom  = ImmediateDominators(blocks, order);
		NumberTree(order, idom);
	}

	// True when definition dominates use and is not the same block. Unreachable blocks answer
	// false.
	[[nodiscard]] bool StrictlyDominates(const Block* definition, const Block* use) const {
		if (definition == use || definition == nullptr || use == nullptr) return false;
		const auto def = m_id.at(definition), user = m_id.at(use);
		if (!m_reached[def] || !m_reached[user]) return false;
		return m_enter[def] <= m_enter[user] && m_exit[user] <= m_exit[def];
	}

private:
	size_t                                     m_count;
	std::unordered_map<const Block*, uint32_t> m_id;
	std::vector<uint32_t>                      m_enter, m_exit;
	std::vector<bool>                          m_reached;

	// Blocks reachable from the entry, entry first, each after its dominators.
	std::vector<uint32_t> ReversePostorder(const BlockList& blocks) const {
		std::vector<uint32_t>                    order;
		std::vector<bool>                        seen(m_count, false);
		std::vector<std::pair<uint32_t, size_t>> stack {{0u, 0u}};
		seen[0] = true;
		while (!stack.empty()) {
			auto& [block, next]   = stack.back();
			const auto successors = blocks[block]->ImmSuccessors();
			if (next < successors.size()) {
				const auto target = m_id.at(successors[next++]);
				if (!seen[target]) {
					seen[target] = true;
					stack.emplace_back(target, 0u);
				}
			} else {
				order.push_back(block);
				stack.pop_back();
			}
		}
		std::ranges::reverse(order);
		return order;
	}

	// Iterative algorithm of Cooper, Harvey and Kennedy over reverse postorder.
	std::vector<uint32_t> ImmediateDominators(const BlockList&             blocks,
	                                          const std::vector<uint32_t>& order) const {
		std::vector<uint32_t> number(m_count, UINT32_MAX), idom(m_count, UINT32_MAX);
		for (size_t index = 0; index < order.size(); index++)
			number[order[index]] = static_cast<uint32_t>(index);
		idom[0]              = 0;
		const auto intersect = [&](uint32_t first, uint32_t second) {
			while (first != second) {
				while (number[first] > number[second])
					first = idom[first];
				while (number[second] > number[first])
					second = idom[second];
			}
			return first;
		};
		for (bool changed = true; changed;) {
			changed = false;
			for (size_t index = 1; index < order.size(); index++) {
				uint32_t best = UINT32_MAX;
				for (const auto* pred: blocks[order[index]]->ImmPredecessors()) {
					const auto candidate = m_id.at(pred);
					if (idom[candidate] == UINT32_MAX) continue;
					best = best == UINT32_MAX ? candidate : intersect(candidate, best);
				}
				if (best != idom[order[index]]) {
					idom[order[index]] = best;
					changed            = true;
				}
			}
		}
		return idom;
	}

	// Numbers the dominator tree in depth-first order so that a subtree is an interval.
	void NumberTree(const std::vector<uint32_t>& order, const std::vector<uint32_t>& idom) {
		std::vector<std::vector<uint32_t>> children(m_count);
		m_reached.assign(m_count, false);
		for (const auto block: order) {
			m_reached[block] = true;
			if (block != 0) children[idom[block]].push_back(block);
		}
		m_enter.assign(m_count, 0);
		m_exit.assign(m_count, 0);
		uint32_t                                 clock = 0;
		std::vector<std::pair<uint32_t, size_t>> walk {{0u, 0u}};
		m_enter[0] = clock++;
		while (!walk.empty()) {
			auto& [block, next] = walk.back();
			if (next < children[block].size()) {
				const auto child = children[block][next++];
				m_enter[child]   = clock++;
				walk.emplace_back(child, 0u);
			} else {
				m_exit[block] = clock++;
				walk.pop_back();
			}
		}
	}
};

// Selects to replace, and what the analysis saw. Holds no analysis state, so the memo is gone by
// the time any replacement is applied.
struct ElisionPlan {
	std::vector<std::pair<Inst*, Value>> replacements;
	uint32_t                             selects     = 0;
	uint32_t                             cap_hits    = 0;
	uint32_t                             truncations = 0;
};

// Maps an instruction to its position in the pass, by open addressing over a table sized once. It
// replaces std::unordered_map, whose node allocation and pointer chasing dominated the lookups.
class InstIndex {
public:
	static constexpr uint32_t NotFound = UINT32_MAX;

	explicit InstIndex(size_t count) {
		const size_t capacity =
		    std::bit_ceil(std::max<size_t>(count * LoadFactorInverse, MinCapacity));
		m_mask = capacity - 1;
		m_keys.assign(capacity, nullptr);
		m_values.assign(capacity, NotFound);
	}

	// Records the position of an instruction that is not yet in the table.
	void Insert(const Inst* inst, uint32_t position) {
		size_t slot = SlotOf(inst);
		while (m_keys[slot] != nullptr)
			slot = (slot + 1) & m_mask;
		m_keys[slot]   = inst;
		m_values[slot] = position;
	}

	// Position of the instruction, or NotFound when it is not in the program.
	[[nodiscard]] uint32_t Find(const Inst* inst) const {
		for (size_t slot = SlotOf(inst); m_keys[slot] != nullptr; slot = (slot + 1) & m_mask)
			if (m_keys[slot] == inst) return m_values[slot];
		return NotFound;
	}

private:
	static constexpr size_t LoadFactorInverse = 2;
	static constexpr size_t MinCapacity       = 16;
	static constexpr size_t PointerAlignShift = 4;

	size_t                   m_mask = 0;
	std::vector<const Inst*> m_keys;
	std::vector<uint32_t>    m_values;

	[[nodiscard]] size_t SlotOf(const Inst* inst) const {
		const auto bits = reinterpret_cast<uintptr_t>(inst) >> PointerAlignShift;
		return static_cast<size_t>((bits * HashGoldenRatio) >> HashFoldShift) & m_mask;
	}
};

// Backward demand analysis. For every instruction it finds which lanes can observe its value, then
// picks the selects whose observers all see the select condition true. It reads the program and
// never changes it. One object serves one run, because its implication memo is keyed on the
// instructions as they were when the run began.
class DemandAnalysis {
public:
	explicit DemandAnalysis(Program& program): m_program(program) {}

	// Runs the whole analysis and decides which selects can be elided.
	ElisionPlan Run() {
		size_t instruction_count = 0;
		for (const auto* block: m_program.blocks)
			instruction_count += block->Instructions().size();
		m_index.emplace(instruction_count);
		m_insts.reserve(instruction_count);
		for (auto* block: m_program.blocks)
			for (auto& inst: *block) {
				m_index->Insert(&inst, static_cast<uint32_t>(m_insts.size()));
				m_insts.push_back(&inst);
			}
		m_demand.assign(m_insts.size(), Demand {});
		m_growths.assign(m_insts.size(), 0);
		m_queued.assign(m_insts.size(), false);
		Solve();
		auto plan        = Decide();
		plan.cap_hits    = m_implication.DepthCapHits();
		plan.truncations = m_truncations;
		return plan;
	}

private:
	// A slot that has grown this many times becomes All. A slot grows rarely, since it only changes
	// to a demand that reaches lanes it did not cover, so this limit is not met by real programs.
	// It makes the end of the solve independent of how the implication proofs happen to compose.
	static constexpr uint8_t MaxSlotGrowths = 8;

	Program&                          m_program;
	MaskImplication                   m_implication;
	uint32_t                          m_truncations = 0;
	std::vector<Inst*>                m_insts;
	std::optional<InstIndex>          m_index;
	std::vector<Demand>               m_demand;
	std::vector<uint8_t>              m_growths;
	std::vector<uint32_t>             m_work;
	std::vector<bool>                 m_queued;
	std::optional<DominatorIntervals> m_dominators;

	// True when every lane in the demand has the condition true.
	bool Within(const Demand& demand, Value condition) {
		if (demand.kind != DemandKind::Mask) return false;
		condition = condition.Resolve();
		// The same value needs no implication query. Most checks during the solve are of this kind.
		for (size_t index = 0; index < demand.count; index++)
			if (Value(demand.conjuncts[index]) == condition) return true;
		for (size_t index = 0; index < demand.count; index++)
			if (m_implication.Implies(Value(demand.conjuncts[index]), condition)) return true;
		return false;
	}

	// True when every lane of inner is a lane of outer.
	bool Subset(const Demand& inner, const Demand& outer) {
		if (outer.kind == DemandKind::All || inner.kind == DemandKind::None) return true;
		if (inner.kind == DemandKind::All || outer.kind == DemandKind::None) return false;
		for (size_t index = 0; index < outer.count; index++)
			if (!Within(inner, Value(outer.conjuncts[index]))) return false;
		return true;
	}

	// Adds a conjunct to a list. Nothing is added when the list already implies it, and the
	// conjuncts it implies are removed, so a list never holds one conjunct that another makes
	// redundant. Returns false when the list is full of unrelated conjuncts.
	bool AddMinimal(Demand& list, Inst* conjunct) {
		if (Within(list, Value(conjunct))) return true;
		size_t kept = 0;
		for (size_t index = 0; index < list.count; index++)
			if (!m_implication.Implies(Value(conjunct), Value(list.conjuncts[index])))
				list.conjuncts[kept++] = list.conjuncts[index];
		std::fill(list.conjuncts.begin() + kept, list.conjuncts.end(), nullptr);
		list.count = kept;
		if (list.count == MaxDemandConjuncts) return false;
		list.conjuncts[list.count++] = conjunct;
		return true;
	}

	// Adds the conjuncts of from that also cover other. A conjunct that does not fit is dropped,
	// which only widens the result, and is counted.
	void KeepCovering(Demand& joined, const Demand& from, const Demand& other) {
		for (size_t index = 0; index < from.count; index++)
			if (Within(other, Value(from.conjuncts[index])) &&
			    !AddMinimal(joined, from.conjuncts[index]))
				m_truncations++;
	}

	// Smallest demand this representation can state that covers both. The first argument comes back
	// unchanged when the second is already inside it, so a slot changes only when it truly grows.
	// Otherwise only conjuncts that cover the other side are kept, so the result is a superset of
	// each. No shared conjunct means All.
	Demand Join(const Demand& first, const Demand& second) {
		if (first.kind == DemandKind::None) return second;
		if (second.kind == DemandKind::None) return first;
		if (first.kind == DemandKind::All || second.kind == DemandKind::All) return AllLanes();
		if (first == second || Subset(second, first)) return first;
		if (Subset(first, second)) return second;
		Demand joined;
		joined.kind = DemandKind::Mask;
		KeepCovering(joined, first, second);
		KeepCovering(joined, second, first);
		return joined.count == 0 ? AllLanes() : joined;
	}

	// The lanes of a demand that are also lanes of the mask. Dropping the mask would still be sound
	// but less precise, so a demand that has no room for another conjunct is returned as it is.
	Demand Narrow(const Demand& demand, Value mask) {
		mask = mask.Resolve();
		if (demand.kind == DemandKind::None) return demand;
		if (IsImmediateBool(mask, false)) return Demand {};
		auto* inst = mask.TryInstruction();
		// A mask that is not a program instruction cannot be proven about, so the demand is kept.
		if (inst == nullptr || inst->Parent() == nullptr) return demand;
		if (demand.kind == DemandKind::All) return MaskedBy(inst);
		Demand narrowed = demand;
		if (AddMinimal(narrowed, inst)) return narrowed;
		m_truncations++;
		return demand;
	}

	// The demand a masked memory operation puts on its data, which is its mask.
	static Demand DemandOfMask(Value mask) {
		mask = mask.Resolve();
		if (IsImmediateBool(mask, false)) return Demand {};
		auto* inst = mask.TryInstruction();
		return inst != nullptr && inst->Parent() != nullptr ? MaskedBy(inst) : AllLanes();
	}

	// Raises the demand on one operand and queues its producer when that changed anything.
	void Push(Value value, const Demand& demand) {
		auto* inst = value.Resolve().TryInstruction();
		if (inst == nullptr || demand.kind == DemandKind::None) return;
		const auto position = m_index->Find(inst);
		if (position == InstIndex::NotFound) return;
		auto& slot   = m_demand[position];
		auto  joined = Join(slot, demand);
		if (joined == slot) return;
		if (++m_growths[position] > MaxSlotGrowths) joined = AllLanes();
		slot = joined;
		if (!m_queued[position]) {
			m_queued[position] = true;
			m_work.push_back(position);
		}
	}

	// An incoming value only keeps a mask when every conjunct was fixed before the phi block was
	// entered. A conjunct computed inside the loop may differ between the iterations the phi joins.
	Demand ThroughPhi(const Demand& demand, const Inst& phi) {
		if (demand.kind != DemandKind::Mask) return demand;
		if (!m_dominators) m_dominators.emplace(m_program.blocks);
		for (size_t index = 0; index < demand.count; index++)
			if (!m_dominators->StrictlyDominates(demand.conjuncts[index]->Parent(), phi.Parent()))
				return AllLanes();
		return demand;
	}

	// Masked stores and atomics. The address, offset and resource operands (M1) and the mask are
	// observed in every lane, and the data only under the mask.
	void TransferMasked(const Inst& inst) {
		const auto   operands = inst.NumArgs();
		const auto   begin    = FirstDataOperand(ValueOpcodeName(inst.GetOpcode()));
		const auto   mask     = inst.Arg(operands - 1);
		const Demand data     = DemandOfMask(mask);
		for (size_t operand = 0; operand < begin; operand++)
			Push(inst.Arg(operand), AllLanes());
		for (size_t operand = begin; operand + 1 < operands; operand++)
			Push(inst.Arg(operand), data);
		Push(mask, AllLanes());
	}

	// Pushes what one instruction needs from its operands, given how much of it is observed.
	void Transfer(uint32_t index) {
		const auto* inst   = m_insts[index];
		const auto  demand = m_demand[index];
		const auto  count  = inst->NumArgs();
		switch (CachedClass(inst->GetOpcode())) {
			case ExecOpClass::Pure:
				for (size_t operand = 0; operand < count; operand++)
					Push(inst->Arg(operand), demand);
				return;
			case ExecOpClass::Select:
				Push(inst->Arg(SelectConditionArg), demand);
				Push(inst->Arg(SelectTrueValueArg), Narrow(demand, inst->Arg(SelectConditionArg)));
				Push(inst->Arg(SelectFalseValueArg), demand);
				return;
			case ExecOpClass::And:
				// Only the second operand is narrowed by the first. Narrowing each by the other is
				// circular and wrong where both are false and both get replaced.
				Push(inst->Arg(0), demand);
				Push(inst->Arg(1), Narrow(demand, inst->Arg(0)));
				return;
			case ExecOpClass::Phi: {
				const auto incoming = ThroughPhi(demand, *inst);
				for (size_t operand = 0; operand < count; operand++)
					Push(inst->Arg(operand), incoming);
			}
				return;
			case ExecOpClass::MemData: TransferMasked(*inst); return;
			case ExecOpClass::Lane:
			case ExecOpClass::Load:
			case ExecOpClass::Unlisted:
				for (size_t operand = 0; operand < count; operand++)
					Push(inst->Arg(operand), AllLanes());
				return;
		}
	}

	// Seeds what is observed unconditionally. Retained resource-planning instructions sit outside
	// the block list and read their operands in every lane.
	void Seed() {
		for (uint32_t index = 0; index < m_insts.size(); index++)
			for (const auto& use: m_insts[index]->Uses())
				if (use.user->Parent() == nullptr) m_demand[index] = AllLanes();
		for (auto* block: m_program.blocks) {
			Push(block->condition, AllLanes());
			Push(block->indirect_target, AllLanes());
		}
	}

	// Runs the worklist until no demand grows. Join hands the slot back unchanged unless the new
	// demand reaches lanes the slot does not cover, so a slot only changes by growing. The growth
	// is strictly upward over finitely many lists of at most MaxDemandConjuncts program values, and
	// a slot that has grown MaxSlotGrowths times becomes All, which never changes again. Each slot
	// is therefore queued a bounded number of times and the loop ends. This is a per-slot widening,
	// not a cap on the number of iterations. The worklist is a stack and is filled in program
	// order, so the last instructions come off first. Users are processed before their producers,
	// which lets a producer see its whole demand in one visit in straight-line code.
	void Solve() {
		Seed();
		for (uint32_t index = 0; index < m_insts.size(); index++) {
			m_queued[index] = true;
			m_work.push_back(index);
		}
		while (!m_work.empty()) {
			const auto index = m_work.back();
			m_work.pop_back();
			m_queued[index] = false;
			Transfer(index);
		}
	}

	// Follows replacements that are themselves replaced, so none names a select that goes dead. A
	// chain cannot be longer than the number of replacements, which bounds the walk. Returns an
	// empty value when the walk runs out of steps.
	Value ResolveReplacement(Value value, const std::vector<Value>& chosen, size_t limit) const {
		for (size_t step = 0; step <= limit; step++) {
			value                = value.Resolve();
			const auto* inst     = value.TryInstruction();
			const auto  position = inst == nullptr ? InstIndex::NotFound : m_index->Find(inst);
			if (position == InstIndex::NotFound || chosen[position].IsEmpty()) return value;
			value = chosen[position];
		}
		// Out of steps, which only a cycle of selects can cause. There is no safe replacement.
		return Value {};
	}

	// Picks every select whose observers all see the condition true, then resolves chains.
	ElisionPlan Decide() {
		ElisionPlan                          plan;
		std::vector<std::pair<Inst*, Value>> picked;
		std::vector<Value>                   chosen(m_insts.size());
		for (uint32_t index = 0; index < m_insts.size(); index++) {
			auto* inst = m_insts[index];
			if (!IsSelect(inst->GetOpcode())) continue;
			plan.selects++;
			if (!Within(m_demand[index], inst->Arg(SelectConditionArg))) continue;
			picked.emplace_back(inst, inst->Arg(SelectTrueValueArg));
			chosen[index] = inst->Arg(SelectTrueValueArg);
		}
		plan.replacements.reserve(picked.size());
		for (auto& [inst, replacement]: picked) {
			const auto resolved = ResolveReplacement(replacement, chosen, picked.size());
			if (!resolved.IsEmpty()) plan.replacements.emplace_back(inst, resolved);
		}
		return plan;
	}
};

} // namespace

ExecSelectStats ElideUnobservedSelects(Program& program) {
	ExecSelectStats stats;
	if (program.stage != ShaderType::Compute) return stats;
	// The analysis object is gone before anything is replaced, so its memo never sees changed
	// operands.
	const auto plan          = DemandAnalysis(program).Run();
	stats.candidates         = plan.selects;
	stats.depth_cap_hits     = plan.cap_hits;
	stats.demand_truncations = plan.truncations;
	for (const auto& [inst, replacement]: plan.replacements) {
		inst->ReplaceUsesWith(replacement, false);
		stats.elided++;
	}
	return stats;
}

size_t CountInstructions(const Program& program) {
	size_t count = 0;
	for (const auto* block: program.blocks)
		count += block->Instructions().size();
	return count;
}

ExecSelectDecision ClassifyExecSelectGate(const Program& program, bool opted_in) {
	if (!opted_in) return {ExecSelectGate::Off, 0};
	if (program.stage != ShaderType::Compute) return {ExecSelectGate::NotCompute, 0};
	const auto instructions = CountInstructions(program);
	return {instructions > MaxExecSelectInstructions ? ExecSelectGate::TooLarge
	                                                 : ExecSelectGate::Run,
	        instructions};
}

} // namespace Libs::Graphics::ShaderRecompiler::IR
