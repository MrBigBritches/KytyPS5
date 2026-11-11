#include "graphics/shader/storegen/candidateOrder.h"

#include "graphics/shader/storegen/generatedOrder.h"

#include <algorithm>
#include <tuple>

namespace Libs::Graphics::StoreGen {

namespace {

// The passes of the order. The head of every program comes first, the partial subsets of wide
// programs next and the base of narrow programs last.
constexpr int PASS_HEAD_AND_WIDE_BASE = 0;
constexpr int PASS_WIDE_PARTIAL       = 1;
constexpr int PASS_NARROW_BASE        = 2;

// Within the first pass, a head goes before the base of its own program.
constexpr int ROLE_HEAD = 0;
constexpr int ROLE_BASE = 1;

// Where one candidate falls in the order.
struct Placement {
	int pass        = PASS_HEAD_AND_WIDE_BASE;
	int subset_rank = 0;
	int role        = ROLE_HEAD;
};

// Works out the pass, subset rank and role of one candidate of a program.
Placement PlaceCandidate(const OrderedProgram& program, size_t head, uint32_t candidate) {
	const auto& nulls = program.candidates[candidate].nulls;
	if (candidate == head) {
		return {PASS_HEAD_AND_WIDE_BASE, 0, ROLE_HEAD};
	}
	const bool wide = program.candidates.size() > WIDE_CANDIDATES_ABOVE;
	if (nulls.empty()) {
		return wide ? Placement {PASS_HEAD_AND_WIDE_BASE, 0, ROLE_BASE}
		            : Placement {PASS_NARROW_BASE, 0, ROLE_BASE};
	}
	// A wide program's other subsets go by size across all programs, so the larger ones come first.
	return {PASS_WIDE_PARTIAL, -static_cast<int>(nulls.size()), ROLE_BASE};
}

} // namespace

size_t HeadIndex(std::span<const Candidate> candidates) {
	size_t head = 0;
	for (size_t index = 1; index < candidates.size(); index++) {
		// Strictly larger only, so the first candidate wins a tie.
		if (candidates[index].nulls.size() > candidates[head].nulls.size()) {
			head = index;
		}
	}
	return head;
}

std::vector<RankedCandidate> OrderCandidates(std::span<const OrderedProgram> programs) {
	std::vector<RankedCandidate> ranked;
	for (uint32_t program = 0; program < programs.size(); program++) {
		for (uint32_t candidate = 0; candidate < programs[program].candidates.size(); candidate++) {
			ranked.push_back({program, candidate});
		}
	}
	const auto sort_key = [&](const RankedCandidate& entry) {
		const auto& program = programs[entry.program];
		const auto  placement =
		    PlaceCandidate(program, HeadIndex(program.candidates), entry.candidate);
		return std::make_tuple(placement.pass, placement.subset_rank, program.code_dwords,
		                       entry.program, placement.role, entry.candidate);
	};
	std::ranges::sort(ranked, [&](const auto& left, const auto& right) {
		return sort_key(left) < sort_key(right);
	});
	return ranked;
}

} // namespace Libs::Graphics::StoreGen
