#include "graphics/presentation/system/gallery/colourPage.h"

#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/pageParts.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <span>
#include <string>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view RULE =
    "Components read semantic tokens, never the palette. Re-theme by editing the palette only. "
    "Never show state by colour alone.";

// A swatch's size, in reference pixels. Twelve fit the page's width, the most a section has.
constexpr float SWATCH_WIDTH = 112.0f;
constexpr float CHIP_HEIGHT  = 64.0f;
// The contrast table's columns and row, in reference pixels.
constexpr float PAIR_WIDTH   = 460.0f;
constexpr float NUMBER_WIDTH = 150.0f;
constexpr float ROW_HEIGHT   = 30.0f;

constexpr float CHANNEL_MAX = 255.0f;
constexpr float ROUND_HALF  = 0.5f;
// The sRGB curve and luminance weights, from WCAG 2.1.
constexpr float SRGB_LINEAR_END = 0.04045f;
constexpr float SRGB_LINEAR_DIV = 12.92f;
constexpr float SRGB_OFFSET     = 0.055f;
constexpr float SRGB_DIV        = 1.055f;
constexpr float SRGB_POWER      = 2.4f;
constexpr float RED_WEIGHT      = 0.2126f;
constexpr float GREEN_WEIGHT    = 0.7152f;
constexpr float BLUE_WEIGHT     = 0.0722f;
constexpr float RATIO_OFFSET    = 0.05f;

// The frames a translucent surface is judged over, the darkest and the brightest a game can show.
constexpr Color BLACK_FRAME = Rgb(0x000000);
constexpr Color WHITE_FRAME = Rgb(0xFFFFFF);

// A colour and the name it goes by.
struct Named {
	std::string_view name;
	Color            color;
};

// A row of swatches and what the section is called.
struct Section {
	std::string_view       title;
	std::span<const Named> colors;
};

constexpr std::array<Named, 12> PALETTE_GRAY {
    Named {"gray.0", Palette::GRAY_0},     Named {"gray.50", Palette::GRAY_50},
    Named {"gray.100", Palette::GRAY_100}, Named {"gray.150", Palette::GRAY_150},
    Named {"gray.200", Palette::GRAY_200}, Named {"gray.300", Palette::GRAY_300},
    Named {"gray.400", Palette::GRAY_400}, Named {"gray.500", Palette::GRAY_500},
    Named {"gray.600", Palette::GRAY_600}, Named {"gray.700", Palette::GRAY_700},
    Named {"gray.800", Palette::GRAY_800}, Named {"gray.900", Palette::GRAY_900}};
constexpr std::array<Named, 6> PALETTE_BLUE {
    Named {"blue.300", Palette::BLUE_300}, Named {"blue.400", Palette::BLUE_400},
    Named {"blue.500", Palette::BLUE_500}, Named {"blue.550", Palette::BLUE_550},
    Named {"blue.600", Palette::BLUE_600}, Named {"blue.700", Palette::BLUE_700}};
constexpr std::array<Named, 5> PALETTE_STATUS {
    Named {"green.500", Palette::GREEN_500}, Named {"yellow.500", Palette::YELLOW_500},
    Named {"red.400", Palette::RED_400}, Named {"red.500", Palette::RED_500},
    Named {"red.600", Palette::RED_600}};
constexpr std::array<Named, 4> PALETTE_TROPHY {
    Named {"platinum", Palette::PLATINUM}, Named {"gold", Palette::GOLD},
    Named {"silver", Palette::SILVER}, Named {"bronze", Palette::BRONZE}};

// `channel` from 0..255 sRGB to linear light.
float Linear(uint8_t channel) {
	const float value = static_cast<float>(channel) / CHANNEL_MAX;
	return value <= SRGB_LINEAR_END ? value / SRGB_LINEAR_DIV
	                                : std::pow((value + SRGB_OFFSET) / SRGB_DIV, SRGB_POWER);
}

// How bright `color` is to the eye, from 0 to 1.
float Luminance(Color color) {
	return RED_WEIGHT * Linear(color.r) + GREEN_WEIGHT * Linear(color.g) +
	       BLUE_WEIGHT * Linear(color.b);
}

// `color` as #RRGGBB, followed by its opacity when it has one.
std::string HexOf(Color color) {
	constexpr size_t          LONGEST = sizeof("#RRGGBB 100%");
	std::array<char, LONGEST> text {};
	if (color.a == CHANNEL_MAX) {
		(void)std::snprintf(text.data(), text.size(), "#%02X%02X%02X", color.r, color.g, color.b);
	} else {
		constexpr float PERCENT = 100.0f;
		const int       opacity =
		    static_cast<int>(static_cast<float>(color.a) / CHANNEL_MAX * PERCENT + ROUND_HALF);
		(void)std::snprintf(text.data(), text.size(), "#%02X%02X%02X %d%%", color.r, color.g,
		                    color.b, opacity);
	}
	return text.data();
}

// A chip of `color` over a strip that is half dark and half white, so its translucency shows.
Element Chip(UiFrame& frame, Color color) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 2> halves {
	    Box(frame, {.item = {.grow = 1.0f}}, {.fill = BLACK_FRAME}),
	    Box(frame, {.item = {.grow = 1.0f}}, {.fill = WHITE_FRAME})};
	const std::array<Element, 3> layers {
	    Box(frame, {.align = Align::Stretch}, {}, halves), Box(frame, {}, {.fill = color}),
	    Box(frame, {},
	        {.border = theme.colors.border.standard, .border_width = Dp(theme.border.hair)})};
	return Box(frame,
	           {.item = {.width = Dp(SWATCH_WIDTH), .height = Dp(CHIP_HEIGHT)}, .stacked = true},
	           {}, layers);
}

// A chip with the colour's name and value under it.
Element Swatch(UiFrame& frame, const Named& named) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::string            hex   = HexOf(named.color);
	const ItemLayout             width {.width = Dp(SWATCH_WIDTH)};
	const std::array<Element, 3> lines {
	    Chip(frame, named.color),
	    Text(frame, {named.name, theme.type.caption, theme.colors.text.primary}, width),
	    Text(frame, {hex, theme.type.caption, theme.colors.text.tertiary}, width)};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, lines);
}

// A section's heading over a row of its swatches.
Element SwatchSection(UiFrame& frame, const Section& section) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> swatches;
	swatches.reserve(section.colors.size());
	for (const Named& named: section.colors) {
		swatches.push_back(Swatch(frame, named));
	}
	const std::array<Element, 2> parts {SectionHeading(frame, section.title),
	                                    Box(frame, {.gap = Dp(theme.space.s2)}, {}, swatches)};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, parts);
}

// The sections in a column.
Element SwatchSections(UiFrame& frame, std::span<const Section> sections) {
	std::vector<Element> rows;
	rows.reserve(sections.size());
	for (const Section& section: sections) {
		rows.push_back(SwatchSection(frame, section));
	}
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s3)},
	           {}, rows);
}

// The Palette view, every raw colour.
Element PaletteView(UiFrame& frame) {
	const std::array<Section, 4> sections {
	    Section {"Gray", PALETTE_GRAY}, Section {"Blue", PALETTE_BLUE},
	    Section {"Status", PALETTE_STATUS}, Section {"Trophy", PALETTE_TROPHY}};
	return SwatchSections(frame, sections);
}

// The Roles view, what each semantic token is.
Element RolesView(UiFrame& frame) {
	const auto&                  colors = frame.ActiveTheme().colors;
	const std::array<Named, 7>   surfaces {Named {"base", colors.surface.base},
                                         Named {"raised", colors.surface.raised},
                                         Named {"overlay", colors.surface.overlay},
                                         Named {"hud", colors.surface.hud},
                                         Named {"hud_strong", colors.surface.hud_strong},
                                         Named {"inset", colors.surface.inset},
                                         Named {"scrim", colors.surface.scrim}};
	const std::array<Named, 8>   text {Named {"primary", colors.text.primary},
                                     Named {"secondary", colors.text.secondary},
                                     Named {"tertiary", colors.text.tertiary},
                                     Named {"disabled", colors.text.disabled},
                                     Named {"on_accent", colors.text.on_accent},
                                     Named {"accent", colors.text.accent},
                                     Named {"on_light", colors.text.on_light},
                                     Named {"secondary_on_light", colors.text.secondary_on_light}};
	const std::array<Named, 9>   accent {Named {"fill", colors.accent.fill},
                                       Named {"fill_hover", colors.accent.fill_hover},
                                       Named {"fill_pressed", colors.accent.fill_pressed},
                                       Named {"line", colors.accent.line},
                                       Named {"tint", colors.accent.tint},
                                       Named {"track_off", colors.control.track_off},
                                       Named {"knob", colors.control.knob},
                                       Named {"fill_disabled", colors.control.fill_disabled},
                                       Named {"knob_disabled", colors.control.knob_disabled}};
	const std::array<Named, 9>   status {Named {"success", colors.status.success},
                                       Named {"success_tint", colors.status.success_tint},
                                       Named {"warning", colors.status.warning},
                                       Named {"warning_tint", colors.status.warning_tint},
                                       Named {"error", colors.status.error},
                                       Named {"error_tint", colors.status.error_tint},
                                       Named {"info", colors.status.info},
                                       Named {"info_tint", colors.status.info_tint},
                                       Named {"danger_fill", colors.status.danger_fill}};
	const std::array<Named, 11>  lines {Named {"subtle", colors.border.subtle},
                                       Named {"standard", colors.border.standard},
                                       Named {"strong", colors.border.strong},
                                       Named {"hover", colors.state.hover},
                                       Named {"focus", colors.state.focus},
                                       Named {"pressed", colors.state.pressed},
                                       Named {"ring", colors.focus.ring},
                                       Named {"keyline", colors.focus.keyline},
                                       Named {"glow", colors.focus.glow},
                                       Named {"hover_on_light", colors.state.hover_on_light},
                                       Named {"focus_on_light", colors.state.focus_on_light}};
	const std::array<Section, 5> sections {Section {"Surfaces", surfaces}, Section {"Text", text},
	                                       Section {"Accent and controls", accent},
	                                       Section {"Status", status},
	                                       Section {"Borders, state and focus", lines}};
	return SwatchSections(frame, sections);
}

// `ratio` as the table shows it.
std::string RatioText(float ratio) {
	constexpr size_t          LONGEST = sizeof("21.0:1");
	std::array<char, LONGEST> text {};
	(void)std::snprintf(text.data(), text.size(), "%.1f:1", static_cast<double>(ratio));
	return text.data();
}

// One row of the contrast table's cells, in `style`, on a box of `fill`.
struct TableCells {
	std::string_view pair;
	std::string      dark;
	std::string      white;
	std::string      target;
	std::string_view result;
	Color            pair_color;
	Color            dark_color;
	Color            white_color;
	Color            result_color;
	Color            fill;
};

// The row's cells side by side, the numbers at the end of their columns.
Element TableRow(UiFrame& frame, const TableCells& cells, const TypeStyle& style) {
	const Theme&                 theme = frame.ActiveTheme();
	const ItemLayout             number {.width = Dp(NUMBER_WIDTH)};
	const std::array<Element, 5> parts {
	    Text(frame, {cells.pair, style, cells.pair_color}, {.width = Dp(PAIR_WIDTH)}),
	    Text(frame, {cells.dark, style, cells.dark_color, TextAlign::End}, number),
	    Text(frame, {cells.white, style, cells.white_color, TextAlign::End}, number),
	    Text(frame, {cells.target, style, theme.colors.text.secondary, TextAlign::End}, number),
	    Text(frame, {cells.result, style, cells.result_color, TextAlign::End}, number)};
	return Box(frame,
	           {.align   = Align::Center,
	            .padding = DpInsets::Symmetric(Dp(theme.space.s2), Dp()),
	            .item    = {.height = Dp(ROW_HEIGHT)}},
	           {.fill = cells.fill}, parts);
}

// The cells of `row`. A failing row is tinted and says so in words, and the number that fell short
// is marked, so a fail doesn't rest on the colour alone and shows which frame decided it.
TableCells CellsOf(const Theme& theme, const ContrastRow& row) {
	const float dark   = ContrastOver(row, BLACK_FRAME);
	const float white  = ContrastOver(row, WHITE_FRAME);
	const bool  passes = ContrastPasses(row);
	const auto& text   = theme.colors.text;
	const Color error  = theme.colors.status.error;
	return {row.name,
	        RatioText(dark),
	        RatioText(white),
	        RatioText(row.target),
	        passes ? "Pass" : "Fail",
	        text.secondary,
	        dark < row.target ? error : text.secondary,
	        white < row.target ? error : text.secondary,
	        passes ? theme.colors.status.success : error,
	        passes ? Color {} : theme.colors.status.error_tint};
}

// The Contrast view, a heading row and a row for each pair.
Element ContrastView(UiFrame& frame) {
	const Theme&         theme = frame.ActiveTheme();
	const auto&          text  = theme.colors.text;
	const TableCells     heading {"Pair",       "Over black", "Over white", "Target",     "Result",
                              text.primary, text.primary, text.primary, text.primary, {}};
	std::vector<Element> rows {TableRow(frame, heading, theme.type.caption)};
	for (const ContrastRow& row: ContrastRows(theme)) {
		rows.push_back(TableRow(frame, CellsOf(theme, row), theme.type.body_small));
	}
	return Box(frame, {.direction = FlexDirection::Column}, {}, rows);
}

} // namespace

Color Composite(Color top, Color under) {
	const float alpha = static_cast<float>(top.a) / CHANNEL_MAX;
	const auto  blend = [alpha](uint8_t over, uint8_t below) {
        return static_cast<uint8_t>(static_cast<float>(over) * alpha +
		                             static_cast<float>(below) * (1.0f - alpha) + ROUND_HALF);
	};
	return {blend(top.r, under.r), blend(top.g, under.g), blend(top.b, under.b),
	        static_cast<uint8_t>(CHANNEL_MAX)};
}

float ContrastRatio(Color first, Color second) {
	const float one   = Luminance(first);
	const float other = Luminance(second);
	return (std::max(one, other) + RATIO_OFFSET) / (std::min(one, other) + RATIO_OFFSET);
}

float ContrastOver(const ContrastRow& row, Color backdrop) {
	const Color floor      = row.under.a > 0 ? Composite(row.under, backdrop) : backdrop;
	const Color background = Composite(row.background, floor);
	const Color ground =
	    row.foreground_on.a > 0 ? Composite(row.foreground_on, background) : background;
	return ContrastRatio(Composite(row.foreground, ground), background);
}

bool ContrastPasses(const ContrastRow& row) {
	return std::min(ContrastOver(row, BLACK_FRAME), ContrastOver(row, WHITE_FRAME)) >= row.target;
}

std::vector<ContrastRow> ContrastRows(const Theme& theme) {
	const auto& colors = theme.colors;
	const Color raised = colors.surface.raised;
	return {
	    {"Primary text on raised", colors.text.primary, raised},
	    {"Secondary text on raised", colors.text.secondary, raised},
	    {"Tertiary text on raised", colors.text.tertiary, raised},
	    {"Primary text on base", colors.text.primary, colors.surface.base},
	    {"Secondary text on base", colors.text.secondary, colors.surface.base},
	    {"Primary text on overlay", colors.text.primary, colors.surface.overlay},
	    {"Secondary text on HUD", colors.text.secondary, colors.surface.hud},
	    {"Secondary text on strong HUD", colors.text.secondary, colors.surface.hud_strong},
	    {"Button label on accent, large", colors.text.on_accent, colors.accent.fill,
	     CONTRAST_LARGE},
	    {"Button label on accent hover, large", colors.text.on_accent, colors.accent.fill_hover,
	     CONTRAST_LARGE},
	    {"Button label on accent pressed, large", colors.text.on_accent, colors.accent.fill_pressed,
	     CONTRAST_LARGE},
	    {"Button label on danger, large", colors.text.on_accent, colors.status.danger_fill,
	     CONTRAST_LARGE},
	    {"Success text on raised", colors.status.success, raised},
	    {"Warning text on raised", colors.status.warning, raised},
	    {"Error text on raised", colors.status.error, raised},
	    {"Info text on raised", colors.status.info, raised},
	    {"Success text on its tint, on raised", colors.status.success, colors.status.success_tint,
	     CONTRAST_TEXT, raised},
	    {"Warning text on its tint, on raised", colors.status.warning, colors.status.warning_tint,
	     CONTRAST_TEXT, raised},
	    {"Error text on its tint, on raised", colors.status.error, colors.status.error_tint,
	     CONTRAST_TEXT, raised},
	    {"Info text on its tint, on raised", colors.status.info, colors.status.info_tint,
	     CONTRAST_TEXT, raised},
	    {"Accent line on raised, not text", colors.accent.line, raised, CONTRAST_LARGE},
	    {"Toggle track when on, on raised, not text", colors.accent.fill, raised, CONTRAST_LARGE},
	    {"Knob on the off track, not text", colors.control.knob, colors.control.track_off,
	     CONTRAST_LARGE},
	    {"Toggle's off outline on raised, not text",
	     colors.border.strong,
	     raised,
	     CONTRAST_LARGE,
	     {},
	     colors.control.track_off},
	    {"Slider track's outline on raised, not text",
	     colors.border.strong,
	     raised,
	     CONTRAST_LARGE,
	     {},
	     colors.surface.inset},
	    {"Slider fill on its track, on raised, not text", colors.accent.line, colors.surface.inset,
	     CONTRAST_LARGE, raised},
	    {"Knob on the slider's track, on raised, not text", colors.control.knob,
	     colors.surface.inset, CONTRAST_LARGE, raised},
	    {"Focus ring on raised, not text", colors.focus.ring, raised, CONTRAST_LARGE},
	};
}

void DrawColourPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	const PageHead head = BuildPageHead(frame, page, RULE, &state.colour_view);
	const int view = std::clamp(state.colour_view, 0, static_cast<int>(COLOUR_VIEWS.size()) - 1);
	const Element                body = view == 0   ? PaletteView(frame)
	                                    : view == 1 ? RolesView(frame)
	                                                : ContrastView(frame);
	const std::array<Element, 2> rows {head.root, body};
	const Element                root =
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s4)},
	        {}, rows);
	ShowPage(frame, state, root, head.tabs, view, pane);
}

} // namespace Libs::Graphics::SystemUi::Gallery
