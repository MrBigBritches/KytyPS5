#include "graphics/presentation/system/gallery/spacingPage.h"

#include "graphics/presentation/system/components/card.h"
#include "graphics/presentation/system/components/hud.h"
#include "graphics/presentation/system/components/slider.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/components/toggle.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/effects/halo.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/pageParts.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view RULE =
    "Use the spacing scale, with hair for strokes only. Menus live in the content frame. Anything "
    "pinned to an edge uses the safe area.";

// The Scales view, in reference pixels.
constexpr float SCALE_NAME_WIDTH  = 96.0f;
constexpr float SPACE_COLUMN      = 400.0f;
constexpr float SPACE_BAR_HEIGHT  = 24.0f;
constexpr float RADIUS_TILE       = 96.0f;
constexpr float BORDER_BOX_WIDTH  = 160.0f;
constexpr float BORDER_BOX_HEIGHT = 64.0f;
constexpr int   RADII_PER_ROW     = 3;
// The Frame view, in reference pixels. The mock is a 16:9 screen, and an ultrawide one is as wide
// and less tall in the same space.
constexpr float MOCK_WIDTH        = 880.0f;
constexpr float MOCK_HEIGHT       = 495.0f;
constexpr float LEGEND_LINE_WIDTH = 40.0f;
constexpr float LEGEND_LINE_TALL  = 20.0f;
constexpr float LEGEND_NAME_WIDTH = 150.0f;
constexpr float DIAGRAM_WIDTH     = 250.0f;
constexpr float DIAGRAM_HEIGHT    = 120.0f;
constexpr int   DIAGRAMS_PER_ROW  = 2;
// The screens the mock shows.
constexpr Vec2 STANDARD_SCREEN {1920.0f, 1080.0f};
constexpr Vec2 ULTRAWIDE_SCREEN {3440.0f, 1440.0f};
// How the safe area slider moves.
constexpr float SAFE_STEP = 0.5f;

// One length of a scale and what it is called.
template <class Scale>
struct NamedLength {
	std::string_view name;
	float Scale::* value;
};

constexpr std::array<NamedLength<SpaceScale>, 10> SPACES {
    NamedLength<SpaceScale> {"hair", &SpaceScale::hair},
    {"s1", &SpaceScale::s1},
    {"s2", &SpaceScale::s2},
    {"s3", &SpaceScale::s3},
    {"s4", &SpaceScale::s4},
    {"s5", &SpaceScale::s5},
    {"s6", &SpaceScale::s6},
    {"s7", &SpaceScale::s7},
    {"s8", &SpaceScale::s8},
    {"s9", &SpaceScale::s9}};
constexpr std::array<NamedLength<BorderWidths>, 4> BORDERS {
    NamedLength<BorderWidths> {"hair", &BorderWidths::hair},
    {"thin", &BorderWidths::thin},
    {"thick", &BorderWidths::thick},
    {"bar", &BorderWidths::bar}};

// A radius and what it is called.
struct NamedRadius {
	std::string_view name;
	float            radius;
};

// `value` with no decimals when it is whole, as a token's length usually is.
std::string LengthText(float value) {
	constexpr size_t          LONGEST = sizeof("10000.0");
	std::array<char, LONGEST> text {};
	const auto                whole = static_cast<float>(static_cast<int>(value));
	(void)std::snprintf(text.data(), text.size(), value == whole ? "%.0f" : "%.1f",
	                    static_cast<double>(value));
	return text.data();
}

// A column's heading, its contents and a caption under them.
Element ScaleColumn(UiFrame& frame, std::string_view heading, std::string_view caption,
                    Element contents) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 3> parts {
	    SectionHeading(frame, heading), contents,
	    Text(frame, {caption, theme.type.caption, theme.colors.text.secondary})};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s3)}, {}, parts);
}

// The space scale, a bar as long as each length.
Element SpaceColumn(UiFrame& frame) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> rows;
	rows.reserve(SPACES.size());
	for (const auto& named: SPACES) {
		const float                  length = theme.space.*named.value;
		const std::string            label  = std::string(named.name) + "  " + LengthText(length);
		const std::array<Element, 2> parts {
		    Text(frame, {label, theme.type.caption, theme.colors.text.secondary},
		         {.width = Dp(SCALE_NAME_WIDTH)}),
		    Box(frame, {.item = {.width = Dp(length), .height = Dp(SPACE_BAR_HEIGHT)}},
		        {.fill = theme.colors.accent.line})};
		rows.push_back(Box(frame, {.align = Align::Center}, {}, parts));
	}
	const Element bars = Box(frame,
	                         {.direction = FlexDirection::Column,
	                          .gap       = Dp(theme.space.s2),
	                          .item      = {.width = Dp(SPACE_COLUMN)}},
	                         {}, rows);
	return ScaleColumn(frame, "Space", "hair is for strokes, not layout", bars);
}

// The radius scale, a tile rounded by each, in rows.
Element RadiusColumn(UiFrame& frame) {
	const Theme&                     theme = frame.ActiveTheme();
	const auto&                      radii = theme.radius;
	const std::array<NamedRadius, 6> named {NamedRadius {"xs", radii.xs},
	                                        {"sm", radii.sm},
	                                        {"md", radii.md},
	                                        {"lg", radii.lg},
	                                        {"xl", radii.xl},
	                                        {"full", RADIUS_FULL}};
	std::vector<Element>             rows;
	std::vector<Element>             tiles;
	for (const NamedRadius& radius: named) {
		const std::array<Element, 2> parts {
		    Box(frame, {.item = {.width = Dp(RADIUS_TILE), .height = Dp(RADIUS_TILE)}},
		        {.fill         = theme.colors.accent.tint,
		         .border       = theme.colors.border.strong,
		         .border_width = Dp(theme.border.hair),
		         .radius       = Dp(radius.radius)}),
		    Text(frame, {radius.name, theme.type.caption, theme.colors.text.secondary})};
		tiles.push_back(
		    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, parts));
		if (tiles.size() == RADII_PER_ROW) {
			rows.push_back(Box(frame, {.gap = Dp(theme.space.s5)}, {}, tiles));
			tiles.clear();
		}
	}
	const Element grid =
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s4)}, {}, rows);
	return ScaleColumn(frame, "Radius", "full is half the shorter side", grid);
}

// The border widths, a box outlined in each.
Element BorderColumn(UiFrame& frame) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> rows;
	rows.reserve(BORDERS.size());
	for (const auto& named: BORDERS) {
		const float       width = theme.border.*named.value;
		const std::string label = "border." + std::string(named.name) + "  " + LengthText(width);
		const std::array<Element, 2> parts {
		    Box(frame, {.item = {.width = Dp(BORDER_BOX_WIDTH), .height = Dp(BORDER_BOX_HEIGHT)}},
		        {.border = theme.colors.accent.fill, .border_width = Dp(width)}),
		    Text(frame, {label, theme.type.caption, theme.colors.text.secondary})};
		rows.push_back(
		    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, parts));
	}
	const Element boxes =
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s2)}, {}, rows);
	return ScaleColumn(frame, "Border", "never below 1 pixel when scaled", boxes);
}

// What the theme's scale rules say, in words, with their numbers read from the theme.
Element ScaleRulesNote(UiFrame& frame) {
	const Theme&      theme = frame.ActiveTheme();
	const ScaleRules& rules = theme.scale;
	const std::string sizes =
	    "Lengths are reference pixels at " + LengthText(rules.reference_width) + " x " +
	    LengthText(rules.reference_height) + ", and the output's height and width set the scale.";
	const std::string safe = "The safe area is " + LengthText(rules.default_safe_area_percent) +
	                         "% in fullscreen, at most " + LengthText(rules.max_safe_area_percent) +
	                         "%, and never less than " + LengthText(rules.edge_min) +
	                         " pixels from an edge.";
	const std::array<Element, 3> rows {
	    SectionHeading(frame, "Scaling"),
	    Text(frame, {sizes, theme.type.caption, theme.colors.text.secondary}),
	    Text(frame, {safe, theme.type.caption, theme.colors.text.secondary})};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, rows);
}

// The Scales view.
Element ScalesView(UiFrame& frame) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 3> columns {SpaceColumn(frame), RadiusColumn(frame),
	                                      BorderColumn(frame)};
	const std::array<Element, 2> rows {Box(frame, {.gap = Dp(theme.space.s6)}, {}, columns),
	                                   ScaleRulesNote(frame)};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s4)}, {}, rows);
}

// The sample screen for the choices the view offers, as the real scale works it out.
ScaleContext SampleOutput(const ScaleRules& rules, float safe_percent, bool ultrawide) {
	const Vec2 screen = ultrawide ? ULTRAWIDE_SCREEN : STANDARD_SCREEN;
	return {rules, screen.x, screen.y, {.safe_area_percent = safe_percent}};
}

// How each area of the sample screen is outlined.
struct AreaLine {
	Color border;
	float width;
};

// The screen's outline.
AreaLine ScreenLine(const Theme& theme) {
	return {theme.colors.border.strong, theme.border.hair};
}

// The safe area's outline.
AreaLine SafeLine(const Theme& theme) {
	return {theme.colors.status.success, theme.border.thick};
}

// The content frame's outline, thinner than the safe area's so both show where they meet.
AreaLine ContentLine(const Theme& theme) {
	return {theme.colors.status.warning, theme.border.hair};
}

// Draws the sample screen and its areas to scale, at the top of the space it is given. Trivially
// copyable so Custom can store it.
struct MockPaint {
	float safe_percent;
	bool  ultrawide;

	void operator()(const CustomPaint& painting) const {
		const Theme&       theme  = painting.frame.ActiveTheme();
		const Rect&        space  = painting.painted;
		const ScaleContext output = SampleOutput(theme.scale, safe_percent, ultrawide);
		const float        shrink = space.width / output.Screen().width;
		// Whole pixels, so that each outline stays as crisp as the legend says it is.
		const auto to_mock = [&space, shrink](const Rect& rect) {
			return Rect {SnapPixel(space.x + rect.x * shrink), SnapPixel(space.y + rect.y * shrink),
			             SnapPixel(rect.width * shrink), SnapPixel(rect.height * shrink)};
		};
		const auto outline = [&](const Rect& rect, const AreaLine& line) {
			const float width = painting.frame.Scale().Border(line.width);
			StrokeOutside(painting.list, rect, 0.0f, -width, width,
			              line.border.MulAlpha(painting.opacity));
		};
		const Rect screen = to_mock(output.Screen());
		painting.list.AddRectFilled(TopLeft(screen), BottomRight(screen),
		                            ToImU32(theme.colors.surface.inset.MulAlpha(painting.opacity)));
		outline(screen, ScreenLine(theme));
		outline(to_mock(output.SafeArea()), SafeLine(theme));
		outline(to_mock(output.ContentFrame()), ContentLine(theme));
	}
};

// `rect`'s size and how far it lies from the screen's edges.
std::string AreaFigures(const Rect& rect) {
	return LengthText(rect.width) + " x " + LengthText(rect.height) + ", " + LengthText(rect.x) +
	       " from the left and " + LengthText(rect.y) + " from the top";
}

// One line of the legend, a sample of the area's outline, its name and its figures. The name and
// the thickness tell the areas apart as well as the colour does.
Element LegendRow(UiFrame& frame, std::string_view name, const AreaLine& line, const Rect& area) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 3> parts {
	    Box(frame, {.item = {.width = Dp(LEGEND_LINE_WIDTH), .height = Dp(LEGEND_LINE_TALL)}},
	        {.border = line.border, .border_width = Dp(line.width)}),
	    Text(frame, {name, theme.type.caption, theme.colors.text.primary},
	         {.width = Dp(LEGEND_NAME_WIDTH)}),
	    Text(frame, {AreaFigures(area), theme.type.caption, theme.colors.text.secondary})};
	return Box(frame, {.align = Align::Center, .gap = Dp(theme.space.s3)}, {}, parts);
}

// The controls under the mock, for the safe area and for an ultrawide screen. They are in rows of
// their own, as a focused slider keeps left and right, so a control beside it couldn't be reached.
Element FrameControls(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme = frame.ActiveTheme();
	const auto&                  text  = theme.colors.text;
	const std::string            value = LengthText(state.spacing_safe_percent) + "%";
	SliderProps                  slider {.min           = 0.0f,
	                                     .max           = theme.scale.max_safe_area_percent,
	                                     .step          = SAFE_STEP,
	                                     .value_label   = frame.Intern(value),
	                                     .label_reserve = "10.5%"};
	const Layout                 row {.align = Align::Center, .gap = Dp(theme.space.s3)};
	const std::array<Element, 2> safe_parts {
	    Text(frame, {"Safe area", theme.type.body_small, text.secondary}),
	    Slider(frame, "safe area", &state.spacing_safe_percent, slider, {.grow = 1.0f}).root};
	const std::array<Element, 2> wide_parts {
	    Toggle(frame, "ultrawide", &state.spacing_ultrawide),
	    Text(frame, {"Ultrawide", theme.type.body_small, text.secondary})};
	const std::array<Element, 2> rows {Box(frame, row, {}, safe_parts),
	                                   Box(frame, row, {}, wide_parts)};
	return Box(frame,
	           {.direction = FlexDirection::Column,
	            .align     = Align::Stretch,
	            .gap       = Dp(theme.space.s2),
	            .item      = {.width = Dp(MOCK_WIDTH)}},
	           {}, rows);
}

// The mock, its controls and the legend that gives each area's figures.
Element FrameMock(UiFrame& frame, GalleryState& state) {
	const Theme&       theme = frame.ActiveTheme();
	const ScaleContext output =
	    SampleOutput(theme.scale, state.spacing_safe_percent, state.spacing_ultrawide);
	const std::string edge_note = "At 0% an edge is still " + LengthText(theme.scale.edge_min) +
	                              " pixels away, as a TV may crop it.";
	const std::array<Element, 7> rows {
	    Custom(frame, {.width = Dp(MOCK_WIDTH), .height = Dp(MOCK_HEIGHT)},
	           MockPaint {state.spacing_safe_percent, state.spacing_ultrawide}),
	    FrameControls(frame, state),
	    LegendRow(frame, "Screen", ScreenLine(theme), output.Screen()),
	    LegendRow(frame, "Safe area", SafeLine(theme), output.SafeArea()),
	    LegendRow(frame, "Content frame", ContentLine(theme), output.ContentFrame()),
	    Text(frame, {edge_note, theme.type.caption, theme.colors.text.secondary}),
	    Text(frame, {"From 3% at 16:9 the content frame all but matches the safe area. Below that, "
	                 "or ultrawide, they differ.",
	                 theme.type.caption, theme.colors.text.secondary})};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s2)}, {}, rows);
}

// A surface's padding, tinted around the space its content gets, with its figures above it.
struct PaddingSample {
	std::string_view name;
	std::string_view note;
	float            horizontal;
	float            vertical;
};

// One padding diagram.
Element PaddingDiagram(UiFrame& frame, const PaddingSample& sample) {
	const Theme&                 theme = frame.ActiveTheme();
	const auto&                  text  = theme.colors.text;
	const std::array<Element, 1> content {
	    Box(frame, {.item = {.grow = 1.0f}}, {.fill = theme.colors.surface.raised})};
	const std::string title = std::string(sample.name) + ", " + LengthText(sample.horizontal) +
	                          " x " + LengthText(sample.vertical);
	const std::array<Element, 3> parts {
	    Text(frame, {title, theme.type.caption, text.primary}),
	    Box(frame,
	        {.align   = Align::Stretch,
	         .padding = DpInsets::Symmetric(Dp(sample.horizontal), Dp(sample.vertical)),
	         .item    = {.width = Dp(DIAGRAM_WIDTH), .height = Dp(DIAGRAM_HEIGHT)}},
	        {.fill = theme.colors.accent.tint}, content),
	    Text(frame, {sample.note, theme.type.caption, text.secondary})};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, parts);
}

// The padding each surface keeps, read from where the surface takes it.
Element PaddingDiagrams(UiFrame& frame) {
	const Theme&                       theme = frame.ActiveTheme();
	const float                        card  = CardDefaults(theme).padding;
	const float                        toast = ToastDefaults(theme).padding;
	const HudTokens                    hud   = HudDefaults(theme);
	const std::array<PaddingSample, 5> samples {
	    PaddingSample {"Card", "CardTokens::padding", card, card},
	    {"Toast", "ToastTokens::padding", toast, toast},
	    {"Dialog", "space.s6", theme.space.s6, theme.space.s6},
	    {"List row", "space.s4, its lines centred", theme.space.s4, 0.0f},
	    {"HUD", "tokens only, no component yet", hud.padding_x, hud.padding_y}};
	std::vector<Element> rows {SectionHeading(frame, "Padding")};
	std::vector<Element> cells;
	for (const PaddingSample& sample: samples) {
		cells.push_back(PaddingDiagram(frame, sample));
		if (cells.size() == DIAGRAMS_PER_ROW) {
			rows.push_back(Box(frame, {.gap = Dp(theme.space.s4)}, {}, cells));
			cells.clear();
		}
	}
	rows.push_back(Box(frame, {.gap = Dp(theme.space.s4)}, {}, cells));
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s3)}, {}, rows);
}

// The Frame view.
Element FrameView(UiFrame& frame, GalleryState& state) {
	const std::array<Element, 2> columns {FrameMock(frame, state), PaddingDiagrams(frame)};
	return Box(frame, {.gap = Dp(frame.ActiveTheme().space.s6)}, {}, columns);
}

} // namespace

void DrawSpacingPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	const PageHead head = BuildPageHead(frame, page, RULE, &state.spacing_view);
	const int view = std::clamp(state.spacing_view, 0, static_cast<int>(SPACING_VIEWS.size()) - 1);
	const std::array<Element, 2> rows {head.root,
	                                   view == 0 ? ScalesView(frame) : FrameView(frame, state)};
	const Element                root =
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s4)},
	        {}, rows);
	ShowPage(frame, state, root, head.tabs, view, pane);
}

} // namespace Libs::Graphics::SystemUi::Gallery
