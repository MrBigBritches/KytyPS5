#include "graphics/presentation/system/gallery/stage.h"

#include "graphics/presentation/system/components/slider.h"
#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/pageParts.h"
#include "graphics/presentation/system/imgui/interaction.h"
#include "graphics/presentation/system/imgui/scaledDraw.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <string>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

// The strip's opacity slider.
constexpr float OPACITY_MIN   = 0.25f;
constexpr float OPACITY_STEP  = 0.05f;
constexpr float OPACITY_WIDTH = 720.0f;
constexpr float PERCENT       = 100.0f;
// Names the specimens' controls apart from the page's, which share the frame's store and window.
constexpr std::string_view STAGE_SCOPE = "##stage";

// What the page's tree holds for the steps after it is rendered.
struct StageTree {
	Element root;
	// The strip's slider, the first control, which focus enters the page on.
	Element entry;
	// The backdrop the specimens are drawn on.
	Element panel;
};

// A number as a whole percentage.
std::string PercentText(std::string_view lead, float share) {
	constexpr size_t          LONGEST = 32;
	std::array<char, LONGEST> text {};
	(void)std::snprintf(text.data(), text.size(), "%.*s %.0f%%", static_cast<int>(lead.size()),
	                    lead.data(), static_cast<double>(share * PERCENT));
	return text.data();
}

// The strip's rows, the opacity alone, as a focused slider keeps left and right, and then the tabs
// and how far the specimens are shrunk, or that they aren't, as rounding could show 100% for a
// little.
struct StripParts {
	Element root;
	Element entry;
};

// Builds the control strip.
StripParts BuildStrip(UiFrame& frame, StageState& stage) {
	const Theme&                 theme  = frame.ActiveTheme();
	const auto&                  text   = theme.colors.text;
	const std::string            value  = PercentText("", stage.opacity);
	const SliderParts            slider = Slider(frame, "opacity", &stage.opacity,
	                                             {.min           = OPACITY_MIN,
	                                              .max           = 1.0f,
	                                              .step          = OPACITY_STEP,
	                                              .value_label   = frame.Intern(value),
	                                              .label_reserve = " 100%"},
	                                             {.width = Dp(OPACITY_WIDTH)});
	const Layout                 row {.align = Align::Center, .gap = Dp(theme.space.s4)};
	const std::array<Element, 2> opacity_parts {
	    Text(frame, {"Opacity", theme.type.body_small, text.secondary}), slider.root};
	const std::array<Element, 3> tab_parts {
	    TabBar(frame, "output", STAGE_OUTPUT_NAMES, &stage.output).element,
	    TabBar(frame, "backdrop", STAGE_BACKDROP_NAMES, &stage.backdrop).element,
	    Text(frame,
	         {stage.fit >= 1.0f ? std::string("Full size") : PercentText("Shrunk to", stage.fit),
	          theme.type.caption, text.secondary})};
	const std::array<Element, 2> rows {Box(frame, row, {}, opacity_parts),
	                                   Box(frame, row, {}, tab_parts)};
	return {Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, rows),
	        slider.track};
}

// Builds the page's tree, with a backdrop that takes all the height left.
StageTree BuildTree(UiFrame& frame, GalleryState& state, const PageEntry& page,
                    std::string_view rule, const Rect& pane) {
	const Theme& theme = frame.ActiveTheme();
	const auto   strip = BuildStrip(frame, state.stage);
	const int    backdrop =
	    std::clamp(state.stage.backdrop, 0, static_cast<int>(STAGE_BACKDROPS.size()) - 1);
	const Element                panel = Box(frame, {.item = {.grow = 1.0f}},
	                                         {.fill = STAGE_BACKDROPS[static_cast<size_t>(backdrop)]});
	const std::array<Element, 4> rows {
	    Text(frame, {page.title, theme.type.title, theme.colors.text.primary}),
	    Text(frame, {rule, theme.type.body_small, theme.colors.text.secondary}), strip.root, panel};
	const Element root = Box(frame,
	                         {.direction = FlexDirection::Column,
	                          .align     = Align::Stretch,
	                          .gap       = Dp(theme.space.s2),
	                          .item      = {.height = FromPixels(frame.Scale(), pane.height)}},
	                         {}, rows);
	return {root, strip.entry, panel};
}

// A page's live copy in a row of its own, saying what it's for in the column the matrix below has
// its rows' names in.
Element LiveRow(StageState& stage, Element live) {
	const Theme&                 theme = stage.frame.ActiveTheme();
	const std::array<Element, 2> parts {Text(stage.frame,
	                                         {"Try it", theme.type.caption, StageInk(theme, stage)},
	                                         {.width = Dp(STAGE_ROW_LABEL_WIDTH)}),
	                                    live};
	return Keyed(stage.frame,
	             Box(stage.frame, {.align = Align::Center, .gap = Dp(theme.space.s3)}, {}, parts),
	             "live row");
}

// Draws a page's live copy at the top of `area`, at its true size, kept to `panel`, and returns the
// area left for the specimens. It takes input, so it is drawn with no scaling.
Rect DrawLive(StageState& stage, Element live, const Rect& area, const Rect& panel) {
	const float unlimited = std::numeric_limits<float>::infinity();
	const Vec2  size      = Measure(stage.frame, live, {unlimited, unlimited});
	(void)Arrange(stage.frame, live, {area.x, area.y, size.x, size.y});
	ImDrawList& list = CurrentDrawList();
	list.PushClipRect({panel.x, panel.y}, {panel.Right(), panel.Bottom()}, true);
	(void)Render(stage.frame, live, list, {.opacity = stage.opacity});
	list.PopClipRect();
	const float taken = size.y + stage.frame.Scale().Px(stage.frame.ActiveTheme().space.s5);
	return {area.x, area.y + taken, area.width, area.height - taken};
}

// Draws the specimens in `area`, measured at the output chosen and shrunk to fit, about its top
// left. What they draw may reach `panel`'s edge, as a shadow or a glow spills past a specimen.
// Render, not Paint, as a specimen's pose comes from Render.
void DrawMatrix(StageState& stage, Element root, const Rect& area, const Rect& panel) {
	(void)AssignIds(stage.frame, root, CurrentIdSeed());
	const float unlimited = std::numeric_limits<float>::infinity();
	const Vec2  size      = Measure(stage.frame, root, {unlimited, unlimited});
	if (!(area.width > 0.0f) || !(area.height > 0.0f) || !(size.x > 0.0f) || !(size.y > 0.0f)) {
		// Nothing is drawn, so nothing is shrunk, whatever the strip said last.
		stage.fit = 1.0f;
		return;
	}
	stage.fit = StageFit(size, area);
	(void)Arrange(stage.frame, root, {area.x, area.y, size.x, size.y});
	ImDrawList&           list = CurrentDrawList();
	const ScaledDrawScope scaled(list, {area.x, area.y}, stage.fit, panel);
	(void)Render(stage.frame, root, list, {.opacity = stage.opacity});
}

// Draws the stage's content in `panel`, inset by `inset`, in the stage's frame at the output
// chosen, the live copy if there is one, then the specimens under it, then the overlay if there is
// one.
void DrawSpecimens(UiFrame& page_frame, GalleryState& state, StageBuild build, StageBuild live,
                   StageOverlay overlay, const Rect& panel, float inset) {
	StageState& stage  = state.stage;
	const int   output = std::clamp(stage.output, 0, static_cast<int>(STAGE_OUTPUTS.size()) - 1);
	const Vec2  screen = STAGE_OUTPUTS[static_cast<size_t>(output)];
	const ScaleContext scale(page_frame.ActiveTheme().scale, screen.x, screen.y);
	stage.frame.Begin(page_frame.ActiveTheme(), scale, page_frame.Ui().fonts, page_frame.Store(),
	                  page_frame.Now());
	const IdScope scope(STAGE_SCOPE);
	Rect          area = panel.Inset(EdgeInsets::All(inset));
	if (live != nullptr) {
		area = DrawLive(stage, LiveRow(stage, live(stage.frame, state)), area, panel);
	}
	DrawMatrix(stage, build(stage.frame, state), area, panel);
	if (overlay != nullptr) {
		overlay(stage.frame, state, panel);
	}
}

} // namespace

Color StageInk(const Theme& theme, const StageState& stage) {
	return stage.backdrop == STAGE_BACKDROP_BRIGHT ? Palette::GRAY_0 : theme.colors.text.primary;
}

float StageFit(Vec2 size, const Rect& area) {
	return std::min({1.0f, area.width / size.x, area.height / size.y});
}

void StageState::Reset() {
	output   = STAGE_OUTPUT_1080P;
	backdrop = STAGE_BACKDROP_MID;
	opacity  = 1.0f;
	fit      = 1.0f;
}

void ShowStagePage(UiFrame& frame, GalleryState& state, const PageEntry& page,
                   std::string_view rule, StageBuild build, const Rect& pane, StageBuild live,
                   StageOverlay overlay) {
	const StageTree tree = BuildTree(frame, state, page, rule, pane);
	(void)AssignIds(frame, tree.root, CurrentIdSeed());
	const float unlimited = std::numeric_limits<float>::infinity();
	(void)Measure(frame, tree.root, {pane.width, unlimited});
	(void)Arrange(frame, tree.root, pane);
	state.entry = IdOf(frame, tree.entry);
	(void)Render(frame, tree.root, CurrentDrawList());
	const float inset = frame.Scale().Px(frame.ActiveTheme().space.s4);
	DrawSpecimens(frame, state, build, live, overlay, Arranged(frame, tree.panel), inset);
}

} // namespace Libs::Graphics::SystemUi::Gallery
