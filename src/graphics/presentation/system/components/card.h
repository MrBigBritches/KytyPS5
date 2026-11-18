#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_CARD_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_CARD_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

#include <span>

namespace Libs::Graphics::SystemUi {

// A card's padding and corners, in reference pixels.
struct CardTokens {
	float padding = 16.0f;
	float radius  = RadiusScale {}.lg;
};

/// The card's tokens for `theme`: its corners come from the radius scale.
[[nodiscard]] constexpr CardTokens CardDefaults(const Theme& theme) {
	CardTokens tokens;
	tokens.radius = theme.radius.lg;
	return tokens;
}

static_assert(CardDefaults(DEFAULT_THEME).padding >= 0.0f &&
              CardDefaults(DEFAULT_THEME).radius >= 0.0f);

// A progress bar's thickness, in reference pixels. Bars share the card's header since they are
// the other plain surface the theme draws.
struct ProgressTokens {
	float height = 8.0f;
};

/// The progress bar's tokens for `theme`.
[[nodiscard]] constexpr ProgressTokens ProgressDefaults(const Theme&) {
	return {};
}

static_assert(ProgressDefaults(DEFAULT_THEME).height > 0.0f);

// A card's look, apart from what Card's layout says.
struct CardProps {
	// Fields of CardDefaults to change for this card, such as its padding.
	TokenPatch<CardTokens> tokens;
};

/// The card's look: a raised surface with a subtle border, rounded corners by `tokens` and a low
/// shadow.
[[nodiscard]] Style CardStyle(const Theme& theme, const CardTokens& tokens);

/// The theme's card as an element: a raised surface with a subtle border, rounded corners and a low
/// shadow, holding `children` inside the theme's card padding. `layout` says how the children are
/// arranged and what the card asks of its container, but its padding is replaced by the card's own,
/// so every card keeps the same margin, and debug builds assert on a non-zero padding. Use Box for
/// a surface with other padding. Returns a none handle when the box is refused, with `reason` set
/// if given.
[[nodiscard]] Element Card(UiFrame& frame, const Layout& layout, const CardProps& props = {},
                           std::span<const Element> children = {}, ElementError* reason = nullptr);

// A progress bar's value and look.
struct ProgressProps {
	// From 0 to 1. A value outside the range is clamped to it, and NaN counts as not started.
	float value = 0.0f;
	// The filled part's colour. Transparent draws only the track.
	Color fill;
	// Fields of ProgressDefaults to change for this bar, such as its height.
	TokenPatch<ProgressTokens> tokens;
};

/// A determinate progress bar as an element: a rounded track with its value of it filled. A started
/// bar always shows at least a round dot, so a little progress is still visible. `item` sizes the
/// bar in its container. A bar with no height in `item` is as thick as the theme's progress height.
/// Returns a none handle when the track is refused, with `reason` set if given.
[[nodiscard]] Element ProgressBar(UiFrame& frame, const ProgressProps& props,
                                  const ItemLayout& item = {}, ElementError* reason = nullptr);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_CARD_H_
