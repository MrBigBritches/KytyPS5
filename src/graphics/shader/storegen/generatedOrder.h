#ifndef GRAPHICS_SHADER_STOREGEN_GENERATED_ORDER_H
#define GRAPHICS_SHADER_STOREGEN_GENERATED_ORDER_H

#include <cstdint>
#include <optional>

// Where a generated module stands among the variants of its program. The supervisor works it out
// as each record arrives, and the pipeline cache carries it with the module so that the warm queue
// can build the late variants of a program after the others.
namespace Libs::Graphics::StoreGen {

/// A program with more candidates than this has a long tail of variants and is called wide. It is
/// shared with the worker's candidate order, so both agree on what wide means.
constexpr uint64_t WIDE_CANDIDATES_ABOVE = 2;
/// The first Back tier of a wide program. The worker builds its head at tier 0 and its base at
/// tier 1, and both are likely to be asked for, so only the partials after them wait.
constexpr uint32_t WIDE_FIRST_BACK_TIER = 2;
/// The first Back tier of a narrow program. Its head is tier 0 and its base, built last by the
/// worker, is tier 1, which is the one build that can wait.
constexpr uint32_t NARROW_FIRST_BACK_TIER = 1;

/// Whether a module is built in the Back lane of the warm queue.
/// @param tier how many records of its program came before it.
/// @param candidates how many candidates the worker predicted for the program. An unknown count or
///        0 is never Back, which is the order a store had before this was tracked.
[[nodiscard]] constexpr bool IsBackLane(uint32_t tier, std::optional<uint64_t> candidates) {
	if (!candidates.has_value() || *candidates == 0) {
		return false;
	}
	const uint32_t first_back_tier =
	    *candidates > WIDE_CANDIDATES_ABOVE ? WIDE_FIRST_BACK_TIER : NARROW_FIRST_BACK_TIER;
	return tier >= first_back_tier;
}

/// What the supervisor says of one generated module besides its record.
struct GeneratedOrder {
	/// How many records of the same program were delivered before this one.
	uint32_t tier = 0;
	/// The program's predicted candidate count, when its prediction line was read.
	std::optional<uint64_t> candidates;
	/// IsBackLane of the two above. The name is kept from the first rule so that the reports still
	/// compare like for like. It means built in the Back lane.
	bool wide_tail = false;
	/// The module came from the submit-time prescan and not from the generator. It builds on the
	/// prescan's own warm set, which the pause for draws never holds, and the generator's rows and
	/// stall causes leave it out.
	bool prescan = false;
};

} // namespace Libs::Graphics::StoreGen

#endif
