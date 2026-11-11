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

/// The head of a program, which is the candidate with the largest set of unbound buffers. The
/// candidate the predictor lists first wins a tie.
/// @param candidates the program's candidates, in the predictor's order.
/// @return the index of the head, or 0 when there are no candidates.
size_t HeadIndex(std::span<const Candidate> candidates);

/// The order in which the worker builds candidates, and so what a rank means.
///
/// The key is (pass, subset_rank, code_dwords, program index, role, candidate index). A base has no
/// unbound buffers, and a wide program has more than WIDE_CANDIDATES_ABOVE candidates.
/// Pass 0 holds the head of every program, and for a wide program its base right after, as role 0
/// and role 1. A program with one candidate appears here once. subset_rank is 0.
///
/// Pass 1 holds the other candidates of wide programs. subset_rank is minus the size of the
/// candidate's unbound set, so the larger subsets of all programs come before the smaller ones.
///
/// Pass 2 holds the base of narrow programs, which have 2 candidates. subset_rank is 0.
///
/// code_dwords is the size of the program's code and literals, as the scan's `code.size()`. It
/// belongs to the program and so is the same for every candidate of it. The smaller program comes
/// first among equals. The result depends on nothing but the programs given.
/// @param programs the predicted programs, in the order of the eboot.
/// @return every candidate of every program, once, in build order.
std::vector<RankedCandidate> OrderCandidates(std::span<const OrderedProgram> programs);

} // namespace Libs::Graphics::StoreGen
