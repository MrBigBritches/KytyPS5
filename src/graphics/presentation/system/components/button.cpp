#include "graphics/presentation/system/components/button.h"

#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"

#include <algorithm>
#include <array>

namespace Libs::Graphics::SystemUi {

namespace {

// The colours a variant draws with at rest. An accent button blends its own fill toward its hover
// and pressed colours, while the others take the theme's state overlays on top.
struct VariantColors {
	Color fill;
	Color border;
	Color text;
	bool  accent = false;
};

// The resting colours of `variant`.
VariantColors Colors(const Theme& theme, ButtonVariant variant) {
	const auto& colors = theme.colors;
	switch (variant) {
		case ButtonVariant::Primary: return {colors.accent.fill, {}, colors.text.on_accent, true};
		case ButtonVariant::Danger: return {colors.status.danger_fill, {}, colors.text.on_accent};
		case ButtonVariant::Ghost: return {{}, {}, colors.text.primary};
		default: return {colors.surface.overlay, colors.border.standard, colors.text.primary};
	}
}

// The resting colours of a Ghost button on a light surface.
VariantColors GhostOnLightColors(const Theme& theme) {
	return {{}, {}, theme.colors.text.on_light};
}

// The height of `size` in reference pixels.
float Height(const ButtonTokens& tokens, ButtonSize size) {
	switch (size) {
		case ButtonSize::Small: return tokens.height_small;
		case ButtonSize::Large: return tokens.height_large;
		default: return tokens.height_medium;
	}
}

// The space on each side of the label in reference pixels. Small buttons are tighter.
float Padding(const ButtonTokens& tokens, ButtonSize size) {
	return size == ButtonSize::Small ? tokens.padding_x_small : tokens.padding_x;
}

// The look of the button's surface in each state. An accent button blends its own fill toward its
// hover colour while hovered or focused, then toward its pressed colour. The others take the
// theme's overlays, each as strong as its highlight. Focus shows the ring and a press shrinks the
// surface by the theme's press scale.
Style SurfaceStyle(const Theme& theme, const ButtonTokens& tokens, const VariantColors& look) {
	Style       style {look.fill, look.border, Dp(theme.border.hair), Dp(tokens.radius), {}};
	const auto& colors = theme.colors;
	if (look.accent) {
		style.states.hover.fill   = colors.accent.fill_hover;
		style.states.focus.fill   = colors.accent.fill_hover;
		style.states.pressed.fill = colors.accent.fill_pressed;
	} else {
		style.states.hover.overlay   = colors.state.hover;
		style.states.focus.overlay   = colors.state.focus;
		style.states.pressed.overlay = colors.state.pressed;
	}
	style.states.focus.focus_ring = true;
	style.states.pressed.scale    = theme.interaction.press_scale;
	return style;
}

} // namespace

Element Button(UiFrame& frame, std::string_view key, const ButtonProps& props,
               const ItemLayout& item, ElementError* reason) {
	const Theme& theme   = frame.ActiveTheme();
	const auto   tokens  = props.tokens.Apply(ButtonDefaults(theme));
	const auto&  scale   = frame.Scale();
	const bool   shades  = props.on_light && props.variant == ButtonVariant::Ghost;
	const auto   look    = shades ? GhostOnLightColors(theme) : Colors(theme, props.variant);
	const float  padding = scale.Px(Padding(tokens, props.size));
	// The label is centred in the column, which the strut widens to the minimum width. The label is
	// never given a heavier weight in a state, so it never reflows.
	const float     least = std::max(0.0f, scale.Px(tokens.min_width) - 2.0f * padding);
	const TextProps label {props.label, theme.type.label, look.text, TextAlign::Center};
	const std::array<Element, 2> content {
	    Spacer(frame, {.width = FromPixels(scale, least)}, reason), Text(frame, label, {}, reason)};
	const Layout layout {.direction = FlexDirection::Column,
	                     .justify   = Justify::Center,
	                     .align     = Align::Stretch,
	                     .padding   = DpInsets::Symmetric(Dp(Padding(tokens, props.size)), Dp()),
	                     .item      = {.grow = 1.0f}};
	Style        surface_style = SurfaceStyle(theme, tokens, look);
	if (shades) {
		surface_style.states.hover.overlay = theme.colors.state.hover_on_light;
		surface_style.states.focus.overlay = theme.colors.state.focus_on_light;
	}
	const Element surface = Box(frame, layout, surface_style, content, reason);
	// The surface is a child of the interactive box and takes its input from it, so the box that
	// the player hits is the whole button while the surface alone shrinks when it is pressed.
	ItemLayout sized = item;
	if (!sized.height) {
		sized.height = Dp(Height(tokens, props.size));
	}
	const std::array<Element, 1> kids {surface};
	const Element                button =
	    Box(frame, {.direction = FlexDirection::Column, .align = Align::Stretch, .item = sized}, {},
	        kids, reason);
	InteractProps interact;
	interact.disabled      = props.disabled;
	interact.default_focus = props.default_focus;
	if (Keyed(frame, button, key, reason).IsNone()) {
		return {};
	}
	return MakeInteractive(frame, button, interact, reason);
}

} // namespace Libs::Graphics::SystemUi
