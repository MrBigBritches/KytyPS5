#include "graphics/presentation/system/gallery/progressPage.h"

#include "graphics/presentation/system/components/card.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/stage.h"

#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view RULE =
    "Use for work with a known end. Never let it go backwards. A started bar always shows a dot, "
    "so a little progress is still seen. Say what a colour means in words.";

constexpr float BAR_WIDTH = 380.0f;
constexpr float PERCENT   = 100.0f;

// The values that behave differently. 1% is under the bar's height in length, so it shows the dot.
constexpr std::array<float, 4> VALUES {0.0f, 0.01f, 0.5f, 1.0f};

// A fill and what it means.
struct NamedFill {
	std::string_view name;
	Color            color;
};

// One bar, and in words what it shows, so the fill doesn't rest on its colour.
Element Sample(UiFrame& frame, const NamedFill& fill, float value) {
	const Theme&                      theme = frame.ActiveTheme();
	std::array<char, sizeof(" 100%")> percent {};
	(void)std::snprintf(percent.data(), percent.size(), " %.0f%%",
	                    static_cast<double>(value * PERCENT));
	const std::string            line = std::string(fill.name) + percent.data();
	const std::array<Element, 2> parts {
	    ProgressBar(frame, {.value = value, .fill = fill.color}, {.width = Dp(BAR_WIDTH)}),
	    Text(frame, {line, theme.type.caption, theme.colors.text.primary})};
	// On a surface, so the words read on any backdrop.
	return Box(frame,
	           {.direction = FlexDirection::Column,
	            .gap       = Dp(theme.space.s2),
	            .padding   = DpInsets::All(Dp(theme.space.s3))},
	           {.fill = theme.colors.surface.raised, .radius = Dp(theme.radius.sm)}, parts);
}

// A row of one value in each fill.
Element ValueRow(UiFrame& frame, float value) {
	const Theme&                   theme  = frame.ActiveTheme();
	const auto&                    colors = theme.colors;
	const std::array<NamedFill, 3> fills {NamedFill {"Accent", colors.accent.line},
	                                      NamedFill {"Success", colors.status.success},
	                                      NamedFill {"Error", colors.status.error}};
	std::vector<Element>           cells;
	for (const NamedFill& fill: fills) {
		cells.push_back(Sample(frame, fill, value));
	}
	return Box(frame, {.gap = Dp(theme.space.s6)}, {}, cells);
}

// The page's specimens.
Element BuildSpecimens(UiFrame& frame, GalleryState& /*state*/) {
	std::vector<Element> rows;
	rows.reserve(VALUES.size());
	for (const float value: VALUES) {
		rows.push_back(ValueRow(frame, value));
	}
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s5)},
	           {}, rows);
}

} // namespace

void DrawProgressPage(UiFrame& frame, GalleryState& state, const PageEntry& page,
                      const Rect& pane) {
	ShowStagePage(frame, state, page, RULE, BuildSpecimens, pane);
}

} // namespace Libs::Graphics::SystemUi::Gallery
