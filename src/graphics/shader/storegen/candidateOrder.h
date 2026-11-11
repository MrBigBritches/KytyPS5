#pragma once

#include "graphics/shader/storegen/computePredictor.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace Libs::Graphics::StoreGen {

/// What the order needs to know about one predicted program.
struct OrderedProgram {
	/// The size of the program's code and literals in dwords, as the scan's `code.size()`. It is
	/// the same for every candidate.
	size_t code_dwords = 0;
	/// The program's candidates, in the predictor's order.
	std::span<const Candidate> candidates;
};

/// One position in the ordering, as indices into the programs given to OrderCandidates.
struct RankedCandidate {
	uint32_t program   = 0;
	uint32_t candidate = 0;
};

/// The order in which the worker builds candidates, and so what a rank means.
///
/// A candidate that leaves every buffer bound is a base candidate, and every base candidate comes
/// before any other. Within each group the smaller program comes first, by the size of its code and
/// literals, then the program that comes first in the list, then the candidate the predictor lists
/// first. The result depends on nothing but the programs given.
/// @param programs the predicted programs, in the order of the eboot.
/// @return every candidate of every program, once, in build order.
std::vector<RankedCandidate> OrderCandidates(std::span<const OrderedProgram> programs);

} // namespace Libs::Graphics::StoreGen
