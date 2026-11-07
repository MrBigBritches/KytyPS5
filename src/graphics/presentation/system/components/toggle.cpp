#include "graphics/presentation/system/components/toggle.h"

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/effects/halo.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics::SystemUi {

namespace {

// Number of segments for the knob's circle, enough to look round at 4K.
constexpr int KNOB_SEGMENTS = 32;

// How far across the track the knob is, from 0 at the off end to 1 at the on end, eased toward
// where `on` says. The tween is kept in the store under the painting node's id, and starts at its
// target, so a toggle that appears already on doesn't slide on. It may overshoot 1 and dip below 0.
float KnobPosition(const CustomPaint& painting, const bool* on) {
	// A tree painted without Render has no ids, and every node would share the one entry for 0, so
	// the knob rests at its end instead of being eased.
	if (painting.id == 0) {
		return *on ? 1.0f : 0.0f;
	}
	const Track slide {painting.frame.ActiveTheme().interaction.knob_ms, Ease::Emphasis};
	return painting.frame.Store().Animate(painting.id, *on ? 1.0f : 0.0f, {slide, slide},
	                                      painting.frame.Now());
}

// Draws the track, in a colour that follows the knob without its overshoot, and the off track's
// outline, which fades as the accent fill takes over. Trivially copyable so Custom can store it.
struct TrackPaint {
	const bool* on;

	void operator()(const CustomPaint& painting) const {
		const Theme& theme   = painting.frame.ActiveTheme();
		const auto&  colors  = theme.colors;
		const Rect&  rect    = painting.painted;
		const float  radius  = rect.height / 2.0f;
		const float  on_part = ClampUnit(KnobPosition(painting, on));
		const Color  track   = Lerp(colors.control.track_off, colors.accent.fill, on_part);
		painting.list.AddRectFilled(TopLeft(rect), BottomRight(rect),
		                            ToImU32(track.MulAlpha(painting.opacity)), radius);
		// A stroke whose inner edge lies inside by its own width ends at the track's edge.
		const float width = painting.frame.Scale().Border(theme.border.hair);
		StrokeOutside(painting.list, rect, radius, -width, width,
		              colors.border.strong.MulAlpha((1.0f - on_part) * painting.opacity));
	}
};

// Draws the knob, a circle whose centre travels between the centres of the track's round ends.
struct KnobPaint {
	const bool* on;
	// The knob's diameter in reference pixels.
	float knob_size;

	void operator()(const CustomPaint& painting) const {
		const Theme& theme  = painting.frame.ActiveTheme();
		const Rect&  rect   = painting.painted;
		const float  knob   = KnobPosition(painting, on);
		const float  radius = rect.height / 2.0f;
		const float  knob_radius =
		    std::min(painting.frame.Scale().Px(knob_size), rect.height) / 2.0f;
		const float  travel = std::max(0.0f, rect.width - 2.0f * radius);
		const ImVec2 centre {rect.x + radius + travel * knob, rect.y + radius};
		painting.list.AddCircleFilled(centre, knob_radius,
		                              ToImU32(theme.colors.control.knob.MulAlpha(painting.opacity)),
		                              KNOB_SEGMENTS);
	}
};

// Applies the input Render found to the caller's value. Confirm flips it, left turns it off and
// right turns it on, in that order, and says whether that changed it. A disabled toggle changes
// nothing, and building it has already said so.
struct ApplyInput {
	bool* on;
	bool* changed;

	void operator()(const NodeInput& input) const {
		if (input.interaction.state.disabled) {
			return;
		}
		const bool before = *on;
		if (input.interaction.state.activated) {
			*on = !*on;
		}
		if (input.dpad.left) {
			*on = false;
		}
		if (input.dpad.right) {
			*on = true;
		}
		if (changed != nullptr) {
			*changed = *on != before;
		}
	}
};

} // namespace

Element Toggle(UiFrame& frame, std::string_view key, bool* on, const ToggleProps& props,
               const ItemLayout& item, ElementError* reason) {
	// A frame that does not reach Render must not leave last frame's report standing.
	if (props.changed != nullptr) {
		*props.changed = false;
	}
	if (on == nullptr) {
		RefuseBuild(ElementError::NoValue, reason);
		return {};
	}
	const Theme& theme  = frame.ActiveTheme();
	const auto   tokens = props.tokens.Apply(ToggleDefaults(theme));
	ItemLayout   sized  = item;
	if (!sized.width) {
		sized.width = Dp(tokens.width);
	}
	if (!sized.height) {
		sized.height = Dp(tokens.height);
	}
	// The layers share the toggle's rectangle and are drawn in order, the way the toggle has always
	// been: the track, the hover shade over it, the knob, and last the focus ring, so it sits over
	// the knob where they meet. The two boxes take the toggle's input from the node the player
	// hits.
	Style shade {.radius = Dp(RADIUS_FULL)};
	shade.states.hover.overlay = theme.colors.state.hover;
	Style ring {.radius = Dp(RADIUS_FULL)};
	ring.states.focus.focus_ring = true;
	const std::array<Element, 4> layers {
	    Custom(frame, {}, TrackPaint {on}, reason), Box(frame, {}, shade, {}, reason),
	    Custom(frame, {}, KnobPaint {on, tokens.knob}, reason), Box(frame, {}, ring, {}, reason)};
	const Element toggle = Box(frame, {.item = sized, .stacked = true}, {}, layers, reason);
	if (Keyed(frame, toggle, key, reason).IsNone()) {
		return {};
	}
	InteractProps interact;
	interact.disabled = props.disabled;
	interact.dpad     = DpadClaim::Once;
	if (MakeInteractive(frame, toggle, interact, reason).IsNone()) {
		return {};
	}
	return OnInput(frame, toggle, ApplyInput {on, props.changed}, reason);
}

} // namespace Libs::Graphics::SystemUi
