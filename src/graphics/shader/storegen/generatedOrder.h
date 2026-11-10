#ifndef GRAPHICS_SHADER_STOREGEN_GENERATED_ORDER_H
#define GRAPHICS_SHADER_STOREGEN_GENERATED_ORDER_H

#include <cstdint>
#include <optional>

// Where a generated module stands among the variants of its program. The supervisor works it out
// as each record arrives, and the pipeline cache carries it with the module so that the warm queue
// can build the programs with many variants after the others.
namespace Libs::Graphics::StoreGen {

/// The first tier that can be wide tail. Tier 0 is the first record of a program, which a game is
/// most likely to ask for, so it is never left to the end.
constexpr uint32_t WIDE_TAIL_MIN_TIER = 1;
/// A program with more candidates than this has a long tail of variants. Programs of 2 or fewer
/// are cheap enough to build in the normal order.
constexpr uint64_t WIDE_TAIL_CANDIDATES_ABOVE = 2;

/// Whether a module belongs to the wide tail of a program.
/// @param tier how many records of its program came before it.
/// @param candidates how many candidates the worker predicted for the program. An unknown count is
///        never wide, which is the order a store had before this was tracked.
[[nodiscard]] constexpr bool IsWideTail(uint32_t tier, std::optional<uint64_t> candidates) {
	return tier >= WIDE_TAIL_MIN_TIER && candidates.has_value() &&
	       *candidates > WIDE_TAIL_CANDIDATES_ABOVE;
}

/// What the supervisor says of one generated module besides its record.
struct GeneratedOrder {
	/// How many records of the same program were delivered before this one.
	uint32_t tier = 0;
	/// The program's predicted candidate count, when its prediction line was read.
	std::optional<uint64_t> candidates;
	/// IsWideTail of the two above.
	bool wide_tail = false;
};

} // namespace Libs::Graphics::StoreGen

#endif
