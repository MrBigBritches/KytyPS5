#include "graphics/presentation/system/gallery/typographyPage.h"

#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/toggle.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/pageParts.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <string>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view RULE =
    "Use Bold for titles, labels and headings, and Regular for reading text. Dense styles are for "
    "heads-up displays only. Plan for 35% longer text.";

// The ramp's columns, in reference pixels.
constexpr float NAME_WIDTH    = 160.0f;
constexpr float SAMPLE_WIDTH  = 420.0f;
constexpr float METRICS_WIDTH = 300.0f;
// A slot too short for the sample text, and its label's column.
constexpr float SLOT_WIDTH       = 320.0f;
constexpr float SLOT_LABEL_WIDTH = 240.0f;
constexpr int   WRAP_LINES       = 2;
// A column of numbers, and how far it is from its neighbour.
constexpr float DIGITS_WIDTH = 200.0f;

constexpr std::string_view SAMPLE      = "Trophy earned";
constexpr std::string_view LONG_SAMPLE = "The quick brown fox jumps over the lazy dog and keeps on "
                                         "running through the whole of the wood";
// Longer than the English they stand for, as translations usually are.
constexpr std::string_view GERMAN_SAMPLE  = "Geschwindigkeitsbegrenzungseinstellungen speichern";
constexpr std::string_view FINNISH_SAMPLE = "Tallennetaanko pelin edistyminen pilvipalveluun?";

// A style of the ramp and what it is called.
struct NamedStyle {
	std::string_view name;
	TypeStyle TypeRamp::* style;
};

constexpr std::array<NamedStyle, 9> RAMP {
    NamedStyle {"display", &TypeRamp::display},        NamedStyle {"title", &TypeRamp::title},
    NamedStyle {"heading", &TypeRamp::heading},        NamedStyle {"body", &TypeRamp::body},
    NamedStyle {"body_small", &TypeRamp::body_small},  NamedStyle {"label", &TypeRamp::label},
    NamedStyle {"caption", &TypeRamp::caption},        NamedStyle {"dense", &TypeRamp::dense},
    NamedStyle {"dense_small", &TypeRamp::dense_small}};

// A text colour and what it is called.
struct NamedText {
	std::string_view name;
	Color TextColors::* color;
};

constexpr std::array<NamedText, 4> TEXT_LEVELS {
    NamedText {"Primary", &TextColors::primary}, NamedText {"Secondary", &TextColors::secondary},
    NamedText {"Tertiary", &TextColors::tertiary}, NamedText {"Disabled", &TextColors::disabled}};

// An overflow mode and what it is called.
struct NamedOverflow {
	std::string_view name;
	TextOverflow     overflow;
	int              max_lines     = 0;
	bool             shrink_to_fit = false;
};

constexpr std::array<NamedOverflow, 4> OVERFLOWS {
    NamedOverflow {"Ellipsis", TextOverflow::Ellipsis}, NamedOverflow {"Clip", TextOverflow::Clip},
    NamedOverflow {"Wrap, at most 2 lines", TextOverflow::Wrap, WRAP_LINES},
    NamedOverflow {"Shrink, then ellipsis", TextOverflow::Ellipsis, 0, true}};

// A style's size, line and weight, and the least size it may shrink to.
std::string MetricsOf(const TypeStyle& style) {
	constexpr size_t          LONGEST = sizeof("000 / 000 Regular, at least 000");
	std::array<char, LONGEST> text {};
	(void)std::snprintf(text.data(), text.size(), "%.0f / %.0f %s, at least %.0f",
	                    static_cast<double>(style.size), static_cast<double>(style.line),
	                    style.weight == FontWeight::Bold ? "Bold" : "Regular",
	                    static_cast<double>(style.min_size));
	return text.data();
}

// One style of the ramp, its name, a sample and its metrics. With `line_boxes`, the sample's line
// box is tinted, so the room around the glyphs shows.
Element RampRow(UiFrame& frame, const NamedStyle& named, bool line_boxes) {
	const Theme&                 theme  = frame.ActiveTheme();
	const TypeStyle&             style  = theme.type.*named.style;
	const auto&                  colors = theme.colors;
	const Color                  tint   = line_boxes ? colors.accent.tint : Color {};
	const std::array<Element, 1> sample_text {Text(frame, {SAMPLE, style, colors.text.primary})};
	const std::array<Element, 3> cells {
	    Text(frame, {named.name, theme.type.caption, colors.text.secondary},
	         {.width = Dp(NAME_WIDTH)}),
	    Box(frame, {.item = {.width = Dp(SAMPLE_WIDTH)}}, {.fill = tint}, sample_text),
	    Text(frame, {MetricsOf(style), theme.type.caption, colors.text.tertiary},
	         {.width = Dp(METRICS_WIDTH)})};
	return Box(frame, {.align = Align::Center}, {}, cells);
}

// The Ramp view, the toggle and then each style in order.
Element RampView(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 2> toggle_parts {
	    Toggle(frame, "line boxes", &state.typography_line_boxes),
	    Text(frame, {"Show line boxes", theme.type.body, theme.colors.text.secondary})};
	std::vector<Element> rows {
	    Box(frame, {.align = Align::Center, .gap = Dp(theme.space.s3)}, {}, toggle_parts)};
	for (const NamedStyle& named: RAMP) {
		rows.push_back(RampRow(frame, named, state.typography_line_boxes));
	}
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s2)}, {}, rows);
}

// The text colours side by side, each on the raised surface it is judged on.
Element TextColoursRow(UiFrame& frame) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> samples;
	for (const NamedText& named: TEXT_LEVELS) {
		samples.push_back(
		    Text(frame, {named.name, theme.type.body, theme.colors.text.*named.color}));
	}
	return Box(frame, {.gap = Dp(theme.space.s5), .padding = DpInsets::All(Dp(theme.space.s3))},
	           {.fill = theme.colors.surface.raised}, samples);
}

// A label, and text too long for its bordered slot in the look `slot_text` asks for. The border is
// the slot's own edge, so what the text does at the edge shows.
Element SlotRow(UiFrame& frame, std::string_view label, const TextProps& slot_text) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 1> inside {Text(frame, slot_text)};
	const std::array<Element, 2> parts {
	    Text(frame, {label, theme.type.caption, theme.colors.text.secondary},
	         {.width = Dp(SLOT_LABEL_WIDTH)}),
	    Box(frame,
	        {.direction = FlexDirection::Column,
	         .align     = Align::Stretch,
	         .padding   = DpInsets::Symmetric(Dp(theme.space.s2), Dp()),
	         .item      = {.width = Dp(SLOT_WIDTH)}},
	        {.border = theme.colors.border.standard, .border_width = Dp(theme.border.hair)},
	        inside)};
	return Box(frame, {.align = Align::Center}, {}, parts);
}

// What each overflow mode does with text too long for its slot.
Element OverflowRows(UiFrame& frame) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> rows;
	for (const NamedOverflow& named: OVERFLOWS) {
		TextProps props {LONG_SAMPLE, theme.type.body, theme.colors.text.primary};
		props.overflow      = named.overflow;
		props.max_lines     = named.max_lines;
		props.shrink_to_fit = named.shrink_to_fit;
		rows.push_back(SlotRow(frame, named.name, props));
	}
	for (const std::string_view sample: {GERMAN_SAMPLE, FINNISH_SAMPLE}) {
		TextProps props {sample, theme.type.body, theme.colors.text.primary};
		props.overflow = TextOverflow::Wrap;
		rows.push_back(SlotRow(
		    frame, sample == GERMAN_SAMPLE ? "German, wrapped" : "Finnish, wrapped", props));
	}
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s2)}, {}, rows);
}

// Numbers in a column at the end of their slots, where digits that differ in width would stagger.
Element DigitRows(UiFrame& frame) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> rows;
	for (const std::string_view digits: {"1111.11", "8888.88", "1234.56"}) {
		rows.push_back(Text(frame,
		                    {digits, theme.type.body, theme.colors.text.primary, TextAlign::End},
		                    {.width = Dp(DIGITS_WIDTH)}));
	}
	return Box(frame, {.direction = FlexDirection::Column}, {}, rows);
}

// The Text view.
Element TextView(UiFrame& frame) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 6> rows {SectionHeading(frame, "Colours"),
	                                   TextColoursRow(frame),
	                                   SectionHeading(frame, "Too long for its slot"),
	                                   OverflowRows(frame),
	                                   SectionHeading(frame, "Digits line up in a column"),
	                                   DigitRows(frame)};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s2)}, {}, rows);
}

} // namespace

void DrawTypographyPage(UiFrame& frame, GalleryState& state, const PageEntry& page,
                        const Rect& pane) {
	const PageHead head = BuildPageHead(frame, page, RULE, &state.typography_view);
	const int      view =
	    std::clamp(state.typography_view, 0, static_cast<int>(TYPOGRAPHY_VIEWS.size()) - 1);
	const std::array<Element, 2> rows {head.root,
	                                   view == 0 ? RampView(frame, state) : TextView(frame)};
	const Element                root =
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s4)},
	        {}, rows);
	ShowPage(frame, state, root, head.tabs, view, pane);
}

} // namespace Libs::Graphics::SystemUi::Gallery
