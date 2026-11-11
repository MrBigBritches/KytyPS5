#include "graphics/shader/storegen/candidateOrder.h"

#include <algorithm>
#include <tuple>

namespace Libs::Graphics::StoreGen {

std::vector<RankedCandidate> OrderCandidates(std::span<const OrderedProgram> programs) {
	std::vector<RankedCandidate> ranked;
	for (uint32_t program = 0; program < programs.size(); program++) {
		for (uint32_t candidate = 0; candidate < programs[program].candidates.size(); candidate++) {
			ranked.push_back({program, candidate});
		}
	}
	const auto sort_key = [&](const RankedCandidate& entry) {
		const auto& program = programs[entry.program];
		const bool  variant = !program.candidates[entry.candidate].nulls.empty();
		return std::make_tuple(variant, program.code_dwords, entry.program, entry.candidate);
	};
	std::ranges::sort(ranked, [&](const auto& left, const auto& right) {
		return sort_key(left) < sort_key(right);
	});
	return ranked;
}

} // namespace Libs::Graphics::StoreGen
