#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_SLIDER_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_SLIDER_H_

#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

#include <string_view>

namespace Libs::Graphics::SystemUi {

// A slider's track thickness and its knob's diameter, in reference pixels.
struct SliderTokens {
	float track = 8.0f;
	float knob  = 28.0f;

	/// True when the lengths are positive and the knob is at least as thick as the track.
	[[nodiscard]] constexpr bool IsValid() const { return track > 0.0f && knob >= track; }
};

/// The slider's tokens for `theme`. They take nothing from the theme's scales, so they are the same
/// for every theme until one asks for more.
[[nodiscard]] constexpr SliderTokens SliderDefaults(const Theme&) {
	return {};
}

static_assert(SliderDefaults(DEFAULT_THEME).IsValid());

// A slider's range and look. A range given backwards, with `min` above `max`, is put in order.
struct SliderProps {
	float min = 0.0f;
	float max = 1.0f;
	// The value moves in whole steps from `min`, and within half a step of `max` it is `max`, so
	// the range should be at least half a step wide. Zero or less lets it take any value, and the
	// gamepad then moves a hundredth of the range.
	float step = 0.01f;
	// How many steps the gamepad's L2 and R2 move at once.
	int page_steps = 10;
	// Shown at the right of the track, such as "75%". The caller formats it.
	std::string_view value_label;
	// The widest label the slider will show, such as "100%", which the label's space is sized
	// for, so the track keeps its length as the label changes. A label too wide for that space is
	// cut short with an ellipsis. Empty uses `value_label`.
	std::string_view label_reserve;
	bool             disabled = false;
	// Where Render says whether the player changed the value this frame, true when the D-pad, a
	// trigger or the mouse moved it, false when it rested or was only put onto a step. Null for no
	// report. It must outlive the frame's Render, like the value.
	bool* changed = nullptr;
	// Fields of SliderDefaults to change for this slider, such as its knob.
	TokenPatch<SliderTokens> tokens;
};

// The nodes of a built slider. The root holds the track and the label and carries the key. The
// track is the interactive node, the one the player hits and the one whose input InputOf reports,
// so it is the one to give OnActivate. The label is a none handle when the slider has none, and a
// label that the row is too short for collapses to no width and stays a node.
struct SliderParts {
	Element root;
	Element track;
	Element label;
};

/// How many steps one repeat of a held left or right moves after the direction has been held for
/// `held_seconds`: one at first, two after 0.6 seconds and five after 1.5 seconds, so a long
/// range can be crossed quickly but a short press is still precise.
[[nodiscard]] int SliderStepsForHold(float held_seconds);

/// A slider as an element: the track with its fill up to the thumb, the thumb with its shadow, and
/// the focus ring around the thumb, stacked in a track node that takes the input, and the value
/// label beside it. `value` is the caller's, which must outlive the frame's Render. Render changes
/// it before anything is painted, so the thumb moves on the frame of the input. While the slider
/// has focus the D-pad's left and right move it a step, faster the longer they are held, and L2 and
/// R2 move it a page, without moving focus. With the mouse, pressing on the track sets the value
/// there and dragging follows the pointer. The value is kept on a step and within the range, and
/// one that isn't a number becomes the minimum, even when the slider is disabled. A disabled slider
/// is skipped by navigation, gives the D-pad back, is drawn in the theme's solid disabled greys
/// with a flat knob, and never changes.
///
/// `item` sizes the slider in its container. Its height is the thumb's size unless given, and a
/// slider that is to fill its row needs a width or a grow. Only the track takes presses, so
/// pressing the label doesn't move the thumb. The label collapses to nothing when it would leave
/// the track shorter than two thumbs, whether the slider's width is explicit or comes from its
/// container. A change the player made is reported through `props.changed`, which putting a stored
/// value onto a step does not set. `props.changed` is set false when the slider is built, so a
/// frame that never reaches Render reads false. The slider uses the track's OnInput, and leaves its
/// OnActivate to the caller. Fade or move the whole slider with WithPose or the opacity of Paint.
/// `key` names the slider among its siblings. Returns a none handle as the root when a part is
/// refused, with `reason` set if given, or when `value` is null, as NoValue.
///
/// The value label is drawn by a Custom node and is not a text node, so SetTextColor can't recolour
/// it.
[[nodiscard]] SliderParts Slider(UiFrame& frame, std::string_view key, float* value,
                                 const SliderProps& props, const ItemLayout& item = {},
                                 ElementError* reason = nullptr);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_SLIDER_H_
