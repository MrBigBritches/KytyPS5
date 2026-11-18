#include "graphics/presentation/system/components/slider.h"

#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/effects/halo.h"
#include "graphics/presentation/system/effects/highlight.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace Libs::Graphics::SystemUi {

namespace {

// Number of segments for the knob's circle, enough to look round at 4K.
constexpr int KNOB_SEGMENTS = 32;
// A held direction speeds up after these many seconds, to these many steps a repeat.
constexpr float FAST_HOLD_SECONDS    = 0.6f;
constexpr int   FAST_STEPS           = 2;
constexpr float FASTEST_HOLD_SECONDS = 1.5f;
constexpr int   FASTEST_STEPS        = 5;
// A slider with no step moves a hundredth of its range per gamepad step.
constexpr float UNSTEPPED_SHARE = 0.01f;
// The label is left out rather than leave the knob less than its own width of travel, a track
// two knobs long.
constexpr float MIN_TRACK_KNOBS = 2.0f;

// A slider's range, in order, and how it steps. Plain numbers, so the callables that hold it can
// be stored in the frame.
struct Range {
	float min        = 0.0f;
	float max        = 1.0f;
	float step       = 0.01f;
	int   page_steps = 10;
};

// Where a slider's parts go within the track's rectangle `area`. The knob's centre travels from
// `left` to `right`, so the whole knob stays over the track's area.
struct Parts {
	Rect  track;
	float knob_radius = 0.0f;
	float left        = 0.0f;
	float right       = 0.0f;
};

// The parts of a slider whose track area is `area`, for a knob `knob_size` pixels across at most
// and a track `track_height` pixels thick. The track sits on a whole pixel, as a hairline outline
// on a half pixel would spread over two rows at half strength.
Parts PartsOf(const Rect& area, float knob_size, float track_height) {
	const float radius = std::min(knob_size, area.height) / 2.0f;
	const float left   = area.x + radius;
	return {
	    {area.x, SnapPixel(area.y + (area.height - track_height) / 2.0f), area.width, track_height},
	    radius,
	    left,
	    std::max(left, area.x + area.width - radius)};
}

// How far along the range `value` is, 0 for a range with no width.
float FractionOf(float value, const Range& range) {
	const float width = range.max - range.min;
	return width > 0.0f ? (value - range.min) / width : 0.0f;
}

// `value` within the range on a whole step, with NaN as the minimum and anything within half a
// step of the maximum as the maximum.
float Snap(float value, const Range& range) {
	if (std::isnan(value)) {
		return range.min;
	}
	if (range.step > 0.0f && std::isfinite(value)) {
		value = range.max - value < range.step / 2.0f
		            ? range.max
		            : range.min + std::round((value - range.min) / range.step) * range.step;
	}
	return std::clamp(value, range.min, range.max);
}

// `value` moved by the D-pad's left and right and the L2 and R2 triggers Render found this frame.
float ApplyGamepad(const NodeInput& input, float value, const Range& range) {
	const float step   = range.step > 0.0f ? range.step : (range.max - range.min) * UNSTEPPED_SHARE;
	const int   across = SliderStepsForHold(input.dpad.held_seconds);
	value +=
	    static_cast<float>((input.dpad.right ? across : 0) - (input.dpad.left ? across : 0)) * step;
	value += static_cast<float>(((input.triggers.right ? 1 : 0) - (input.triggers.left ? 1 : 0)) *
	                            range.page_steps) *
	         step;
	return value;
}

// `value` set from the mouse while the mouse holds the slider down.
float ApplyPointer(const NodeInput& input, float value, const Range& range, float knob_size) {
	if (!HeldByPointer(input.id)) {
		return value;
	}
	const Parts parts    = PartsOf(input.hit, knob_size, 0.0f);
	const float span     = parts.right - parts.left;
	const float fraction = span > 0.0f ? (PointerPosition().x - parts.left) / span : 0.0f;
	return range.min + std::clamp(fraction, 0.0f, 1.0f) * (range.max - range.min);
}

// Applies what Render found of the player's input to the caller's value, and says whether the
// player changed it. Putting a stored value onto a step is not a change the player made, so the
// change is measured from the snapped value. A disabled slider only has its value put on a step.
struct ApplyInput {
	float* value;
	Range  range;
	// The knob's size in pixels, as the theme gives it at this scale.
	float knob_size;
	// Where to say whether the player changed the value this frame, or null.
	bool* changed;

	void operator()(const NodeInput& input) const {
		const float resting = Snap(*value, range);
		// A disabled node gets no input from Render, which refuses it, so this repeats that
		// refusal. It is a redundant guard, kept so the callback is safe wherever it is called
		// from.
		if (input.interaction.state.disabled) {
			*value = resting;
			Report(false);
			return;
		}
		*value =
		    Snap(ApplyPointer(input, ApplyGamepad(input, resting, range), range, knob_size), range);
		Report(*value != resting);
	}

	// Says what happened to the caller.
	void Report(bool player_changed) const {
		if (changed != nullptr) {
			*changed = player_changed;
		}
	}
};

// The parts of the slider drawn in `painting`, for the value it shows.
struct Placed {
	Parts  parts;
	ImVec2 knob;
	Rect   around;
};

// Works out where the track and the knob are for a slider painted in `painting`.
Placed PlaceParts(const CustomPaint& painting, const float* value, const Range& range,
                  const SliderTokens& tokens) {
	const auto& scale = painting.frame.Scale();
	Placed      placed;
	placed.parts         = PartsOf(painting.painted, scale.Px(tokens.knob), scale.Px(tokens.track));
	const float fraction = FractionOf(*value, range);
	placed.knob          = {placed.parts.left + (placed.parts.right - placed.parts.left) * fraction,
	                        placed.parts.track.y + placed.parts.track.height / 2.0f};
	const float radius   = placed.parts.knob_radius;
	placed.around = {placed.knob.x - radius, placed.knob.y - radius, 2.0f * radius, 2.0f * radius};
	return placed;
}

// Draws the track with its outline, then its fill up to the knob, which covers the outline. A
// disabled slider's fill is grey. Trivially copyable so Custom can store it.
struct TrackPaint {
	const float* value;
	Range        range;
	SliderTokens tokens;
	bool         disabled;

	void operator()(const CustomPaint& painting) const {
		const Theme& theme  = painting.frame.ActiveTheme();
		const auto&  colors = theme.colors;
		const Placed placed = PlaceParts(painting, value, range, tokens);
		const Rect&  track  = placed.parts.track;
		const float  pill   = track.height / 2.0f;
		painting.list.AddRectFilled(TopLeft(track), BottomRight(track),
		                            ToImU32(colors.surface.inset.MulAlpha(painting.opacity)), pill);
		// A stroke whose inner edge lies inside by its own width ends at the track's edge.
		const float width = painting.frame.Scale().Border(theme.border.hair);
		StrokeOutside(painting.list, track, pill, -width, width,
		              colors.border.strong.MulAlpha(painting.opacity));
		const Color fill = disabled ? colors.control.fill_disabled : colors.accent.line;
		painting.list.AddRectFilled(TopLeft(track), {placed.knob.x, track.Bottom()},
		                            ToImU32(fill.MulAlpha(painting.opacity)), pill);
	}
};

// Draws the knob with its shadow, or a disabled slider's flat grey knob, which doesn't float.
struct ThumbPaint {
	const float* value;
	Range        range;
	SliderTokens tokens;
	bool         disabled;

	void operator()(const CustomPaint& painting) const {
		const Theme& theme  = painting.frame.ActiveTheme();
		const Placed placed = PlaceParts(painting, value, range, tokens);
		const float  radius = placed.parts.knob_radius;
		if (!disabled) {
			DrawShadow(painting.list, placed.around, radius, theme.elevation.level1,
			           theme.colors.effect.shadow, painting.frame.Scale(), painting.opacity);
		}
		const Color knob =
		    disabled ? theme.colors.control.knob_disabled : theme.colors.control.knob;
		painting.list.AddCircleFilled(placed.knob, radius, ToImU32(knob.MulAlpha(painting.opacity)),
		                              KNOB_SEGMENTS);
	}
};

// Draws the focus ring around the knob, as strong as the input's focus highlight, and dimmed on a
// disabled slider that still has focus, as a control can be disabled while focused.
struct RingPaint {
	const float* value;
	Range        range;
	SliderTokens tokens;
	bool         disabled;

	void operator()(const CustomPaint& painting) const {
		const Theme& theme  = painting.frame.ActiveTheme();
		const Placed placed = PlaceParts(painting, value, range, tokens);
		const float  dimmed = disabled ? theme.opacity.disabled : 1.0f;
		DrawFocusRing(painting.list, placed.around, placed.parts.knob_radius, theme,
		              painting.frame.Scale(),
		              ClampUnit(painting.input.visuals.focus) * dimmed * painting.opacity);
	}
};

// The value label's text props, in the body style at the end of its space. Tertiary when disabled,
// as the disabled text colour is too faint to read a value by.
TextProps LabelProps(const Theme& theme, std::string_view text, bool disabled) {
	return {text, theme.type.body,
	        disabled ? theme.colors.text.tertiary : theme.colors.text.secondary, TextAlign::End};
}

// The width the value label needs, sized for the widest label so the track keeps its length, or
// zero when there is no label.
float LabelWidth(const UiFrame& frame, const SliderProps& props) {
	const std::string_view widest =
	    props.label_reserve.empty() ? props.value_label : props.label_reserve;
	if (props.value_label.empty() || widest.empty()) {
		return 0.0f;
	}
	const Theme& theme = frame.ActiveTheme();
	return frame
	           .LaidOutText(LabelProps(theme, widest, props.disabled),
	                        std::numeric_limits<float>::max())
	           .size.x +
	       frame.Scale().Px(theme.space.s3);
}

// Draws the value label at the end of its space, centred top to bottom in it even when it is taller
// than the space, so it spills equally above and below. It is laid out where it is painted, in the
// width it was given, and a space with no width shows nothing. Trivially copyable so Custom can
// store it, with a view of text the frame holds.
struct LabelPaint {
	std::string_view text;
	bool             disabled;

	void operator()(const CustomPaint& painting) const {
		const Rect&       space  = painting.painted;
		const TextProps   props  = LabelProps(painting.frame.ActiveTheme(), text, disabled);
		const TextLayout& layout = painting.frame.LaidOutText(props, space.width);
		if (layout.lines.empty()) {
			return;
		}
		const Rect slot {space.x, SnapPixel(space.y + (space.height - layout.size.y) / 2.0f),
		                 space.width, layout.size.y};
		DrawText(painting.list, layout, props, slot, painting.opacity);
	}
};

// A box that doesn't fade when disabled, as a disabled slider shows it in its colours instead.
Style NoDisabledFade() {
	Style style;
	style.states.disabled.opacity = 1.0f;
	return style;
}

// The track's area, with the track, the thumb and the focus ring stacked in it. With a label it
// keeps at least two knobs of length, each twice `radius`, and the label collapses when the row is
// too short for both.
Element MakeTrack(UiFrame& frame, const float* value, const Range& range,
                  const SliderTokens& control, bool has_label, float radius, bool disabled,
                  ElementError* reason) {
	const std::array<Element, 3> layers {
	    Custom(frame, {}, TrackPaint {value, range, control, disabled}, reason),
	    Custom(frame, {}, ThumbPaint {value, range, control, disabled}, reason),
	    Custom(frame, {}, RingPaint {value, range, control, disabled}, reason)};
	ItemLayout track_item {.grow = 1.0f};
	if (has_label) {
		track_item.min_main = FromPixels(frame.Scale(), MIN_TRACK_KNOBS * 2.0f * radius);
	}
	return Box(frame, {.item = track_item, .stacked = true}, NoDisabledFade(), layers, reason);
}

// The value label `width` pixels wide, in a box of its own beside the track.
Element MakeLabel(UiFrame& frame, const SliderProps& props, float width, ElementError* reason) {
	const std::array<Element, 1> text {
	    Custom(frame, {}, LabelPaint {frame.Intern(props.value_label), props.disabled}, reason)};
	const ItemLayout item {.width = FromPixels(frame.Scale(), width), .collapses = true};
	return Box(frame, {.item = item, .stacked = true}, {}, text, reason);
}

} // namespace

int SliderStepsForHold(float held_seconds) {
	if (held_seconds >= FASTEST_HOLD_SECONDS) {
		return FASTEST_STEPS;
	}
	return held_seconds >= FAST_HOLD_SECONDS ? FAST_STEPS : 1;
}

SliderParts Slider(UiFrame& frame, std::string_view key, float* value, const SliderProps& given,
                   const ItemLayout& item, ElementError* reason) {
	const Theme& theme   = frame.ActiveTheme();
	const auto   control = given.tokens.Apply(SliderDefaults(theme));
	const auto&  scale   = frame.Scale();
	const Range  range {std::min(given.min, given.max), std::max(given.min, given.max), given.step,
                       given.page_steps};
	ItemLayout   sized = item;
	if (!sized.height) {
		sized.height = Dp(control.knob);
	}
	const float knob_size = scale.Px(control.knob);
	const float radius    = std::min(knob_size, ToPixels(scale, *sized.height)) / 2.0f;
	const float label     = LabelWidth(frame, given);

	// A frame that does not reach Render must not leave last frame's report standing, so the report
	// starts false when the slider is built and Render's callback overwrites it.
	if (given.changed != nullptr) {
		*given.changed = false;
	}
	if (value == nullptr) {
		RefuseBuild(ElementError::NoValue, reason);
		return {};
	}
	SliderParts parts;
	parts.track =
	    MakeTrack(frame, value, range, control, label > 0.0f, radius, given.disabled, reason);
	std::array<Element, 2> row {parts.track};
	size_t                 count = 1;
	if (label > 0.0f) {
		parts.label  = MakeLabel(frame, given, label, reason);
		row[count++] = parts.label;
	}
	parts.root = Box(frame, {.align = Align::Stretch, .item = sized}, {},
	                 std::span<const Element>(row).first(count), reason);
	InteractProps interact;
	interact.disabled = given.disabled;
	interact.dpad     = DpadClaim::Repeat;
	interact.triggers = true;
	if (Keyed(frame, parts.root, key, reason).IsNone() ||
	    MakeInteractive(frame, parts.track, interact, reason).IsNone() ||
	    OnInput(frame, parts.track, ApplyInput {value, range, knob_size, given.changed}, reason)
	        .IsNone()) {
		return {};
	}
	return parts;
}

} // namespace Libs::Graphics::SystemUi
