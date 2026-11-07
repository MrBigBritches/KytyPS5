#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TOGGLE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TOGGLE_H_

#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

#include <string_view>

namespace Libs::Graphics::SystemUi {

// A toggle's size and its knob's, in reference pixels.
struct ToggleTokens {
	float width  = 64.0f;
	float height = 36.0f;
	float knob   = 28.0f;

	/// True when the lengths are positive and the knob fits in the track.
	[[nodiscard]] constexpr bool IsValid() const {
		return width > 0.0f && height > 0.0f && knob > 0.0f && knob <= height;
	}
};

/// The toggle's tokens for `theme`. They take nothing from the theme's scales, so they are the
/// same for every theme until one asks for more.
[[nodiscard]] constexpr ToggleTokens ToggleDefaults(const Theme&) {
	return {};
}

static_assert(ToggleDefaults(DEFAULT_THEME).IsValid());

// How a toggle takes input.
struct ToggleProps {
	bool disabled = false;
	// Where Render says whether the player changed the value this frame, by confirm, a click or the
	// D-pad. It is set false when the toggle is built and Render overwrites it, so a frame that
	// never reaches Render reads false. A left that finds it already off is no change. Null for no
	// report. It must outlive the frame's Render, like the value.
	bool* changed = nullptr;
	// Fields of ToggleDefaults to change for this toggle, such as its width.
	TokenPatch<ToggleTokens> tokens;
};

/// An on and off switch as an element: a track that changes colour as the knob slides across it,
/// a hover shade, the knob, and the focus ring over all of them, made interactive. `on` is the
/// caller's value, which must outlive the frame's Render, and is read when the toggle paints and
/// changed when Render finds the player's input, before anything is painted, so the knob moves on
/// the frame of the press. Confirm or a click flips it, and while the toggle has focus the D-pad's
/// left turns it off and right turns it on without moving focus, in that order when they come on
/// one frame. A held left or right stops at a toggle rather than passing it, and a disabled toggle
/// gives the D-pad back to navigation. The knob slides over the theme's knob time with the
/// Emphasis curve. The tween is kept in the frame's store in two entries that move together, one
/// under the path id of the track layer and one under the knob's, both derived from the toggle's
/// key. A toggle that first appears is already at its end, and so is one painted without Render,
/// which gives no ids. It is the theme's size unless `item` says otherwise.
///
/// A confirm or click is reported as an activation, in RenderResult::activated and InputOf, and to
/// the callback given to OnActivate, which the toggle leaves free. A change by the D-pad is a
/// change of `on` and not an activation, and `props.changed` reports every change. `key` names the
/// toggle among its siblings, so give each its own. A disabled toggle is skipped by navigation,
/// fades with the theme's disabled opacity and never changes. Fade or move the whole toggle with
/// WithPose or the opacity of Paint. Returns a none handle when a part is refused, with `reason`
/// set if given, or when `on` is null, as NoValue.
[[nodiscard]] Element Toggle(UiFrame& frame, std::string_view key, bool* on,
                             const ToggleProps& props = {}, const ItemLayout& item = {},
                             ElementError* reason = nullptr);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TOGGLE_H_
