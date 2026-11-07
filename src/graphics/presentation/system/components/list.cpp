#include "graphics/presentation/system/components/list.h"

#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics::SystemUi {

namespace {

// The look of a row in each state: the accent bar and tint when selected, the hover and focus
// overlays, and the focus ring.
Style RowStyle(const Theme& theme, const ListTokens& tokens, bool on_light) {
	Style style {{}, {}, Dp(), Dp(tokens.row_radius), {}};
	style.states.selected.selection = true;
	const auto& state               = theme.colors.state;
	style.states.hover.overlay      = on_light ? state.hover_on_light : state.hover;
	style.states.focus.overlay      = on_light ? state.focus_on_light : state.focus;
	style.states.focus.focus_ring   = true;
	return style;
}

// Draws a row's lines, stacked and centred top to bottom inside the row's padding. The text is laid
// out where it is painted, so a row that is not in view costs no layout, and the font is asked for
// its glyphs only as rows come into view. The frame keeps the layouts, as a row has no text node.
// Trivially copyable so Custom can store it, with views of text the frame holds.
struct RowLines {
	std::string_view primary;
	std::string_view secondary;
	bool             on_light;

	void operator()(const CustomPaint& painting) const {
		const UiContext&  context = painting.frame.Ui();
		const auto&       theme   = context.theme;
		const Rect&       rect    = painting.painted;
		const float       padding = context.scale.Px(theme.space.s4);
		const float       width   = std::max(0.0f, rect.width - 2.0f * padding);
		const auto&       text    = theme.colors.text;
		const TextProps   first_props {primary, theme.type.body,
                                     on_light ? text.on_light : text.primary};
		const TextProps   second_props {secondary, theme.type.body_small,
                                      on_light ? text.secondary_on_light : text.secondary};
		const TextLayout& first = painting.frame.LaidOutText(first_props, width);
		if (secondary.empty()) {
			const float y = SnapPixel(rect.y + (rect.height - first.size.y) / 2.0f);
			DrawText(painting.list, first, first_props, {rect.x + padding, y, width, first.size.y},
			         painting.opacity);
			return;
		}
		const TextLayout& second = painting.frame.LaidOutText(second_props, width);
		const float y = SnapPixel(rect.y + (rect.height - first.size.y - second.size.y) / 2.0f);
		DrawText(painting.list, first, first_props, {rect.x + padding, y, width, first.size.y},
		         painting.opacity);
		DrawText(painting.list, second, second_props,
		         {rect.x + padding, y + first.size.y, width, second.size.y}, painting.opacity);
	}
};

// Draws a group's heading in its row's slot, set in on both sides like a row's label and resting on
// the slot's bottom so it sits with the rows under it. Laid out where it is painted, so not while
// the list has scrolled it out of view, which Paint leaves out like a row.
struct HeadingText {
	std::string_view text;

	void operator()(const CustomPaint& painting) const {
		const UiContext&  context = painting.frame.Ui();
		const auto&       theme   = context.theme;
		const Rect&       rect    = painting.painted;
		const float       padding = context.scale.Px(theme.space.s4);
		const TextProps   props {text, theme.type.caption, theme.colors.text.secondary};
		const TextLayout& layout =
		    painting.frame.LaidOutText(props, std::max(0.0f, rect.width - 2.0f * padding));
		const float y = rect.Bottom() - layout.size.y - context.scale.Px(theme.space.s2);
		DrawText(painting.list, layout, props, {rect.x + padding, y, layout.size.x, layout.size.y},
		         painting.opacity);
	}
};

} // namespace

Element ListRow(UiFrame& frame, std::string_view key, const ListRowProps& props,
                ElementError* reason) {
	const Theme& theme = frame.ActiveTheme();
	const auto   list  = props.tokens.Apply(ListDefaults(theme));
	// The lines are drawn by one node that fills the row, over the row's own states.
	const std::array<Element, 1> lines {Custom(
	    frame, {},
	    RowLines {frame.Intern(props.primary), frame.Intern(props.secondary), props.on_light},
	    reason)};
	const Layout                 layout {
	                    .item = {.height = Dp(props.secondary.empty() ? list.row_single : list.row_double)},
	                    .stacked = true};
	const Element row = Box(frame, layout, RowStyle(theme, list, props.on_light), lines, reason);
	InteractProps interact;
	interact.disabled = props.disabled;
	interact.selected = props.selected;
	if (Keyed(frame, row, key, reason).IsNone()) {
		return {};
	}
	return MakeInteractive(frame, row, interact, reason);
}

Element ListHeading(UiFrame& frame, std::string_view key, const ListHeadingProps& props,
                    ElementError* reason) {
	const ListTokens tokens = props.tokens.Apply(ListDefaults(frame.ActiveTheme()));
	const ItemLayout item {.height = Dp(tokens.row_single)};
	const Element    heading = Custom(frame, item, HeadingText {frame.Intern(props.text)}, reason);
	return Keyed(frame, heading, key, reason);
}

Element List(UiFrame& frame, std::string_view key, std::span<const Element> rows,
             const ItemLayout& item, ElementError* reason) {
	const Theme& theme = frame.ActiveTheme();
	// Rows sit in from the list's sides by the focus ring's width, so the ring isn't clipped.
	const float inset =
	    frame.Scale().Border(theme.border.thick) + frame.Scale().Border(theme.border.thin);
	// The right also leaves a gutter for the scroll bar, whether the rows fit or not, so they don't
	// change width as a list grows past its height.
	const Dp ring    = FromPixels(frame.Scale(), inset);
	DpInsets padding = DpInsets::All(ring);
	padding.right    = Dp(ring.value + LIST_THUMB_WIDTH + LIST_THUMB_GAP);
	const Layout layout {.direction = FlexDirection::Column,
	                     .align     = Align::Stretch,
	                     .padding   = padding,
	                     .item      = item};
	Style        clip;
	clip.overflow      = Overflow::Clip;
	const Element list = Box(frame, layout, clip, rows, reason);
	if (Keyed(frame, list, key, reason).IsNone()) {
		return {};
	}
	return AsList(frame, list, reason);
}

} // namespace Libs::Graphics::SystemUi
