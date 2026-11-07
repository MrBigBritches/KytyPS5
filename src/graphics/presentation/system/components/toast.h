#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TOAST_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TOAST_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

#include <cstdint>
#include <optional>
#include <string_view>

struct ImTextureData;

namespace Libs::Graphics::SystemUi {

// A toast's sizes, in reference pixels, and how it moves, in milliseconds.
struct ToastTokens {
	float width              = 560.0f;
	float min_width          = 360.0f;
	float max_width_fraction = 0.4f; // of the content frame
	float padding            = 16.0f;
	float icon               = 80.0f;
	float icon_radius        = RadiusScale {}.sm;
	float gap                = 16.0f; // between the icon and the text
	float min_height         = 112.0f;
	float edge_bar           = BorderWidths {}.bar; // a status toast's coloured left edge
	// The trophy cup drawn before a trophy's title, and the space after it.
	float     cup_width  = 22.0f;
	float     cup_height = 21.0f;
	float     cup_gap    = 8.0f;
	float     radius     = RadiusScale {}.lg;
	Elevation elevation  = ElevationScale {}.level3;
	// How a toast moves. It slides in from `slide` reference pixels above while it fades in, and
	// back out the same way. Exits take about 65% of the entrance.
	float slide         = 32.0f;
	float enter_fade_ms = MotionDurations {}.base;
	float enter_move_ms = MotionDurations {}.slow;
	float exit_fade_ms  = MotionDurations {}.exit_fade;
	float exit_move_ms  = MotionDurations {}.base;
	// A one-off shine starts this long after the toast has faded in.
	float shine_delay_ms = 200.0f;

	/// True when the widths run from the minimum up to the width, the share of the frame is a
	/// fraction, no length or time is negative, and the padding keeps the corner clip off the
	/// content, which needs at least 0.29 times the radius.
	[[nodiscard]] constexpr bool IsValid() const {
		return min_width > 0.0f && min_width <= width && max_width_fraction > 0.0f &&
		       max_width_fraction <= 1.0f && padding >= 0.29f * radius && icon > 0.0f &&
		       gap >= 0.0f && min_height >= 0.0f && edge_bar >= 0.0f && cup_width >= 0.0f &&
		       cup_height >= 0.0f && cup_gap >= 0.0f && slide >= 0.0f && enter_fade_ms >= 0.0f &&
		       enter_move_ms >= 0.0f && exit_fade_ms >= 0.0f && exit_move_ms >= 0.0f &&
		       shine_delay_ms >= 0.0f;
	}
};

/// The toast's tokens for `theme`: its corners, edge bar, shadow and timings come from the radius,
/// border, elevation and motion scales.
[[nodiscard]] constexpr ToastTokens ToastDefaults(const Theme& theme) {
	ToastTokens tokens;
	tokens.icon_radius   = theme.radius.sm;
	tokens.edge_bar      = theme.border.bar;
	tokens.radius        = theme.radius.lg;
	tokens.elevation     = theme.elevation.level3;
	tokens.enter_fade_ms = theme.motion.base;
	tokens.enter_move_ms = theme.motion.slow;
	tokens.exit_fade_ms  = theme.motion.exit_fade;
	tokens.exit_move_ms  = theme.motion.base;
	return tokens;
}

static_assert(ToastDefaults(DEFAULT_THEME).IsValid());

/// The look of a toast's card: a card's surface and border with the toast's own corners and a
/// higher shadow, since it floats over the game.
[[nodiscard]] Style ToastStyle(const Theme& theme, const ToastTokens& tokens);

// What a toast shows: an optional icon, a title of up to two lines and a one-line subtitle. The
// text is copied into the frame when the toast is built, so it need only last until then.
struct ToastProps {
	std::string_view title;
	std::string_view subtitle;
	ImTextureData*   icon = nullptr;
	Vec2             icon_size;
	// A glow behind the icon, such as a trophy's grade colour. Transparent draws none.
	Color       icon_glow;
	// A trophy cup before the title in this colour, such as a trophy's grade. Transparent draws
	// none.
	Color title_cup;
	// A bar along the left edge, such as a status colour. Transparent draws none.
	Color edge_bar;
	// How far a one-off shine has swept across the toast, from 0 to 1, while it plays.
	std::optional<float> shine;
	// Fields of ToastDefaults to change for this toast, such as its width or its padding.
	TokenPatch<ToastTokens> tokens;
};

// The nodes of a built toast. Only the root is always there. The others are none handles for a
// part the props leave out, and they let a caller find where each part was arranged.
struct ToastParts {
	Element root;
	Element icon;
	Element cup;
	Element title;
	Element subtitle;
};

/// Builds a toast as an element tree for a screen whose content frame is `frame_width` pixels
/// wide. Measure the root with no limit and arrange it at its measured size, then pose and paint
/// it. Its width is the theme's toast width, but never wider than its share of the frame unless
/// that would make it narrower than its minimum, and never wider than the frame. It is tall enough
/// for its icon and its text, and at least the theme's minimum height.
///
/// The card is the theme's toast surface. Its content is clipped to the card inside its rounded
/// corners, apart from the icon's glow, which is wider than the padding and spills past the card
/// over the game. The edge bar is outside that clip, drawn between the rounded corners. The cup is
/// centred on the title's first line, not on the whole title. The toast's padding must stay at
/// least 0.29 times its corner radius, or the corner clip would cut into the content. Returns a
/// root that is a none handle when any part is refused, with `reason` set if given.
[[nodiscard]] ToastParts Toast(UiFrame& frame, const ToastProps& props, float frame_width,
                               ElementError* reason = nullptr);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TOAST_H_
