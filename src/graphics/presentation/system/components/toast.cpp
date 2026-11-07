#include "graphics/presentation/system/components/toast.h"

#include "graphics/presentation/system/components/card.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/trophyCup.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics::SystemUi {

namespace {

constexpr int TITLE_LINES = 2;

// The toast's width for a content frame `frame_width` pixels wide. The theme's share of the
// frame caps the width, the minimum width overrides that cap, and the frame itself caps both.
float ToastWidth(const UiContext& context, const ToastTokens& toast, float frame_width) {
	const auto& scale    = context.scale;
	const float capped   = std::min(scale.Px(toast.width), toast.max_width_fraction * frame_width);
	const float at_least = std::max(capped, scale.Px(toast.min_width));
	return SnapPixel(std::min(at_least, std::max(frame_width, 0.0f)));
}

// The title's text props, heading text in the primary colour wrapped to two lines.
TextProps TitleProps(const Theme& theme, const ToastProps& props) {
	TextProps title {props.title, theme.type.heading, theme.colors.text.primary};
	title.overflow  = TextOverflow::Wrap;
	title.max_lines = TITLE_LINES;
	return title;
}

// The subtitle's text props, one line of secondary text that shrinks before it is cut short.
TextProps SubtitleProps(const Theme& theme, const ToastProps& props) {
	TextProps subtitle {props.subtitle, theme.type.body_small, theme.colors.text.secondary};
	subtitle.shrink_to_fit = true;
	return subtitle;
}

// Insets that are only to the right, as the gap after a part that is followed by another.
DpInsets GapAfter(Dp gap) {
	return {Dp(), Dp(), gap, Dp()};
}

// Draws the bar along the card's left edge, between the rounded corners so it never pokes past the
// card's outline. Trivially copyable so Custom can store it.
struct EdgeBarPaint {
	Color color;
	// The card's corner radius and the bar's width, in reference pixels.
	float card_radius;
	float bar_width;

	void operator()(const CustomPaint& painting) const {
		const auto& scale  = painting.frame.Scale();
		const Rect& card   = painting.painted;
		const float radius = std::min(scale.Px(card_radius), card.height / 2.0f);
		const Rect  bar {card.x, card.y + radius, scale.Border(bar_width),
                        std::max(0.0f, card.height - 2.0f * radius)};
		painting.list.AddRectFilled(TopLeft(bar), BottomRight(bar),
		                            ToImU32(color.MulAlpha(painting.opacity)), bar.width / 2.0f);
	}
};

// The icon in the glow that `props` asks for, and the box that adds the gap after it.
Element IconSlot(UiFrame& frame, const ToastProps& props, const ToastTokens& toast,
                 ToastParts& parts, ElementError* reason) {
	const Theme& theme = frame.ActiveTheme();
	const Dp     side(toast.icon);
	parts.icon = Image(frame, props.icon, props.icon_size, ImageFit::Cover, Dp(toast.icon_radius),
	                   {.width = side, .height = side}, reason);
	if (props.icon_glow.a > 0) {
		// The glow is wider than the padding, and is meant to spill past the card over the game, so
		// it escapes the clip the rest of the content is under.
		Decor decor;
		decor.glow      = GlowEffect {theme.glow.trophy, props.icon_glow};
		decor.unclipped = true;
		(void)Decorate(frame, parts.icon, decor, reason);
	}
	const std::array<Element, 1> kids {parts.icon};
	return Box(frame, {.padding = GapAfter(Dp(toast.gap))}, {}, kids, reason);
}

// The cup in a slot one title line tall, so it is centred on the title's first line and stays
// there when the title wraps. The slot's width is the cup and the gap after it.
Element CupSlot(UiFrame& frame, const ToastProps& props, const ToastTokens& toast,
                const TextProps& title, ToastParts& parts, ElementError* reason) {
	const auto& scale = frame.Scale();
	// Text lays out a line at its line height or its font size, whichever is larger.
	const float line = std::max(scale.Px(title.style.line), scale.FontSize(title.style));
	parts.cup        = TrophyCup(frame, {.color = props.title_cup, .tokens = props.tokens},
	                             {.width = Dp(toast.cup_width), .height = Dp(toast.cup_height)},
	                             reason);
	const std::array<Element, 1> kids {parts.cup};
	return Box(frame,
	           {.align   = Align::Center,
	            .padding = GapAfter(Dp(toast.cup_gap)),
	            .item    = {.height = FromPixels(scale, line)}},
	           {}, kids, reason);
}

// The title, after the cup when there is one, over the subtitle. It takes the width the icon
// leaves, and the title the width the cup leaves.
Element TextBlock(UiFrame& frame, const ToastProps& props, const ToastTokens& toast,
                  ToastParts& parts, ElementError* reason) {
	const Theme&    theme = frame.ActiveTheme();
	const TextProps title = TitleProps(theme, props);
	parts.title           = Text(frame, title, {}, reason);
	parts.subtitle        = Text(frame, SubtitleProps(theme, props), {}, reason);
	std::array<Element, 2> row {parts.title};
	size_t                 count = 1;
	if (props.title_cup.a > 0) {
		row   = {CupSlot(frame, props, toast, title, parts, reason), parts.title};
		count = 2;
	}
	const std::array<Element, 2> lines {
	    Box(frame, {}, {}, std::span<const Element>(row).first(count), reason), parts.subtitle};
	return Box(
	    frame,
	    {.direction = FlexDirection::Column, .align = Align::Stretch, .item = {.grow = 1.0f}}, {},
	    lines, reason);
}

// The content inside the card's padding, centred across and clipped to the card inside its
// corners. Its least height makes the card at least the theme's minimum height.
Element Content(UiFrame& frame, const ToastProps& props, const ToastTokens& toast,
                ToastParts& parts, ElementError* reason) {
	const auto&            scale   = frame.Scale();
	const float            padding = scale.Px(toast.padding);
	const float            least   = std::max(0.0f, scale.Px(toast.min_height) - 2.0f * padding);
	std::array<Element, 3> kids {Spacer(frame, {.height = FromPixels(scale, least)}, reason)};
	size_t                 count = 1;
	if (props.icon != nullptr) {
		kids[count++] = IconSlot(frame, props, toast, parts, reason);
	}
	kids[count++] = TextBlock(frame, props, toast, parts, reason);
	const Style clip {.radius = Dp(toast.radius), .overflow = Overflow::Clip};
	return Box(frame,
	           {.align   = Align::Center,
	            .padding = DpInsets::All(Dp(toast.padding)),
	            .item    = {.grow = 1.0f}},
	           clip, std::span<const Element>(kids).first(count), reason);
}

} // namespace

Style ToastStyle(const Theme& theme, const ToastTokens& tokens) {
	Style style     = CardStyle(theme, CardDefaults(theme));
	style.radius    = Dp(tokens.radius);
	style.elevation = tokens.elevation;
	return style;
}

ToastParts Toast(UiFrame& frame, const ToastProps& props, float frame_width, ElementError* reason) {
	ToastParts             parts;
	const Theme&           theme   = frame.ActiveTheme();
	const ToastTokens      tokens  = props.tokens.Apply(ToastDefaults(theme));
	const Element          content = Content(frame, props, tokens, parts, reason);
	std::array<Element, 2> card {content};
	size_t                 count = 1;
	if (props.edge_bar.a > 0) {
		// First, so it sits before the content and its corner is the card's own.
		card  = {Custom(frame, {}, EdgeBarPaint {props.edge_bar, tokens.radius, tokens.edge_bar},
		                reason),
		         content};
		count = 2;
	}
	const Layout layout {
	    .align = Align::Stretch,
	    .item  = {.width = FromPixels(frame.Scale(), ToastWidth(frame.Ui(), tokens, frame_width))}};
	parts.root = Box(frame, layout, ToastStyle(theme, tokens),
	                 std::span<const Element>(card).first(count), reason);
	if (props.shine.has_value()) {
		Decor decor;
		decor.shine = props.shine;
		(void)Decorate(frame, parts.root, decor, reason);
	}
	return parts;
}

} // namespace Libs::Graphics::SystemUi
