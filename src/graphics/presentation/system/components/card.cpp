#include "graphics/presentation/system/components/card.h"

#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <cassert>

namespace Libs::Graphics::SystemUi {

namespace {

// Draws the filled part of a progress bar over its track. Trivially copyable so Custom can store
// it.
struct ProgressFill {
	float progress;
	Color fill;

	void operator()(const CustomPaint& painting) const {
		const Rect& track  = painting.painted;
		const float radius = track.height / 2.0f;
		// At least as wide as the bar is tall, so the rounded ends never collapse into a sliver.
		const float width = std::min(track.width, std::max(track.width * progress, track.height));
		// The right edge is snapped, not the width, so the fill ends on a whole pixel, as layout
		// rectangles do.
		const Rect done {track.x, track.y, SnapPixel(track.x + width) - track.x, track.height};
		painting.list.AddRectFilled(TopLeft(done), BottomRight(done),
		                            ToImU32(fill.MulAlpha(painting.opacity)), radius);
	}
};

} // namespace

Style CardStyle(const Theme& theme, const CardTokens& tokens) {
	return {.fill         = theme.colors.surface.raised,
	        .border       = theme.colors.border.subtle,
	        .border_width = Dp(theme.border.hair),
	        .radius       = Dp(tokens.radius),
	        .elevation    = theme.elevation.level1};
}

Element Card(UiFrame& frame, const Layout& layout, const CardProps& props,
             std::span<const Element> children, ElementError* reason) {
	const Theme&     theme  = frame.ActiveTheme();
	const CardTokens tokens = props.tokens.Apply(CardDefaults(theme));
	// Padding is the card's own, so a caller's would be silently dropped.
	assert(layout.padding == DpInsets {} && "Card sets its own padding, use Box for other padding");
	Layout padded  = layout;
	padded.padding = DpInsets::All(Dp(tokens.padding));
	return Box(frame, padded, CardStyle(theme, tokens), children, reason);
}

Element ProgressBar(UiFrame& frame, const ProgressProps& props, const ItemLayout& item,
                    ElementError* reason) {
	ItemLayout sized = item;
	if (!sized.height) {
		sized.height = Dp(props.tokens.Apply(ProgressDefaults(frame.ActiveTheme())).height);
	}
	const Layout track_layout {.align = Align::Stretch, .item = sized};
	// A thin border, as the inset surface alone is faint against a card.
	const Theme& theme = frame.ActiveTheme();
	const Style  track_style {.fill         = theme.colors.surface.inset,
	                          .border       = theme.colors.border.standard,
	                          .border_width = Dp(theme.border.hair),
	                          .radius       = Dp(RADIUS_FULL)};
	const float  progress = ClampUnit(props.value);
	if (progress <= 0.0f) {
		return Box(frame, track_layout, track_style, {}, reason);
	}
	// The fill grows to the track's whole width and stretches to its height, so it is painted in
	// the track's own rectangle and measures its width there.
	const std::array<Element, 1> fill_node {
	    Custom(frame, ItemLayout {.grow = 1.0f}, ProgressFill {progress, props.fill}, reason)};
	return Box(frame, track_layout, track_style, fill_node, reason);
}

} // namespace Libs::Graphics::SystemUi
