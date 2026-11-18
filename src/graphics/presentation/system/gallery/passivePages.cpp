#include "graphics/presentation/system/gallery/passivePages.h"

#include "graphics/presentation/system/components/card.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/trophyCup.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/imagesPage.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/stage.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view BOX_RULE =
    "A card groups related content. Do not nest cards. Raise elevation only for things that float.";
constexpr std::string_view TEXT_RULE =
    "Truncate with an ellipsis by default. Wrap only where the content is prose. Shrink only "
    "titles "
    "in a fixed slot. Text has no shaping, and runs left to right.";
constexpr std::string_view IMAGE_RULE =
    "Cover for posters and thumbnails. A missing image draws nothing, so give it a surface behind.";
constexpr std::string_view GLYPHS_RULE =
    "Draw symbols from shapes. Do not add an icon font. Pair every symbol with a word.";

// The box and card page, in reference pixels.
constexpr float MATRIX_LABEL_WIDTH = 120.0f;
constexpr float MATRIX_CELL_WIDTH  = 200.0f;
constexpr float MATRIX_CELL_HEIGHT = 96.0f;
constexpr float CARD_COLUMN_WIDTH  = 426.0f;
constexpr float SMALL_CARD_WIDTH   = 205.0f;
constexpr float SMALL_CARD_HEIGHT  = 120.0f;

// The text page.
constexpr float            SLOT_WIDTH       = 420.0f;
constexpr float            SLOT_HEIGHT      = 44.0f;
constexpr float            TABLE_NAME_WIDTH = 120.0f;
constexpr float            TABLE_CELL_WIDTH = 96.0f;
constexpr int              LONGEST_LIMIT    = 3;
constexpr std::string_view PARAGRAPH =
    "A paragraph that says enough to show what a limit on its lines does. Each slot below holds "
    "the "
    "same words, and each is allowed one more line than the last, until the one with no limit "
    "shows "
    "all of them. Where the words are cut short, the last line ends with an ellipsis, so it is "
    "clear "
    "that there is more. A paragraph like this one belongs in a dialog or a description, where "
    "there "
    "is room to read it, and rarely in a list, a button or a toast, where a single line should say "
    "what the player needs to know.";

// The image page.
constexpr float POSTER_CARD_WIDTH   = 440.0f;
constexpr float POSTER_IMAGE_WIDTH  = 408.0f;
constexpr float POSTER_IMAGE_HEIGHT = 230.0f;
constexpr float ICON_SIZE           = 80.0f;
constexpr float THUMB_WIDTH         = 96.0f;
constexpr float THUMB_HEIGHT        = 54.0f;

// The glyphs page. A cup's shape is 22 by 21, and it stretches to whatever it is given.
constexpr float               GLYPH_LABEL_WIDTH = 160.0f;
constexpr float               GLYPH_CELL_WIDTH  = 160.0f;
constexpr float               GLYPH_CELL_HEIGHT = 96.0f;
constexpr float               STRETCHED_CUP     = 80.0f;
constexpr std::array<Vec2, 3> CUP_SIZES {Vec2 {22.0f, 21.0f}, Vec2 {40.0f, 38.0f},
                                         Vec2 {80.0f, 76.0f}};

// A column with `gap` between its rows.
Layout ColumnOf(float gap) {
	return {.direction = FlexDirection::Column, .gap = Dp(gap)};
}

// A caption in `ink`.
Element Caption(UiFrame& frame, std::string_view text, Color ink, const ItemLayout& item = {}) {
	return Text(frame, {text, frame.ActiveTheme().type.caption, ink}, item);
}

// A heading in `ink`.
Element Heading(UiFrame& frame, std::string_view text, Color ink) {
	return Text(frame, {text, frame.ActiveTheme().type.label, ink});
}

// ---- Box and card -------------------------------------------------------------------------------

// A surface or a border and what it is called.
struct NamedColor {
	std::string_view name;
	Color            color;
};

// The matrix's head, each border's name over its column.
Element MatrixHead(UiFrame& frame, std::span<const NamedColor> borders, Color ink) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> cells {Spacer(frame, {.width = Dp(MATRIX_LABEL_WIDTH)})};
	for (const NamedColor& border: borders) {
		cells.push_back(Caption(frame, border.name, ink, {.width = Dp(MATRIX_CELL_WIDTH)}));
	}
	return Box(frame, {.gap = Dp(theme.space.s4)}, {}, cells);
}

// One surface with each border. The first border is transparent, which draws none.
Element MatrixRow(UiFrame& frame, const NamedColor& surface, std::span<const NamedColor> borders,
                  Color ink) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> cells {
	    Caption(frame, surface.name, ink, {.width = Dp(MATRIX_LABEL_WIDTH)})};
	for (const NamedColor& border: borders) {
		cells.push_back(
		    Box(frame, {.item = {.width = Dp(MATRIX_CELL_WIDTH), .height = Dp(MATRIX_CELL_HEIGHT)}},
		        {.fill         = surface.color,
		         .border       = border.color,
		         .border_width = Dp(theme.border.hair),
		         .radius       = Dp(theme.radius.md)}));
	}
	return Box(frame, {.align = Align::Center, .gap = Dp(theme.space.s4)}, {}, cells);
}

// Each surface with each border.
Element SurfaceMatrix(UiFrame& frame, Color ink) {
	const Theme&                    theme  = frame.ActiveTheme();
	const auto&                     colors = theme.colors;
	const std::array<NamedColor, 4> surfaces {NamedColor {"Raised", colors.surface.raised},
	                                          {"Overlay", colors.surface.overlay},
	                                          {"HUD", colors.surface.hud},
	                                          {"Inset", colors.surface.inset}};
	const std::array<NamedColor, 4> borders {NamedColor {"No border", {}},
	                                         {"Subtle", colors.border.subtle},
	                                         {"Standard", colors.border.standard},
	                                         {"Strong", colors.border.strong}};
	std::vector<Element>            rows {MatrixHead(frame, borders, ink)};
	for (const NamedColor& surface: surfaces) {
		rows.push_back(MatrixRow(frame, surface, borders, ink));
	}
	return Box(frame, ColumnOf(theme.space.s4), {}, rows);
}

// A card of the given size holding words, or another card when `inside` is given.
Element WordCard(UiFrame& frame, const ItemLayout& item, std::span<const Element> inside) {
	const Theme& theme  = frame.ActiveTheme();
	Layout       layout = ColumnOf(theme.space.s1);
	layout.item         = item;
	return Card(frame, layout, {}, inside);
}

// A card in use, and the nesting to avoid.
Element CardColumn(UiFrame& frame, Color ink) {
	const Theme&     theme   = frame.ActiveTheme();
	const Color      on_card = theme.colors.text.primary;
	const ItemLayout small {.width = Dp(SMALL_CARD_WIDTH), .height = Dp(SMALL_CARD_HEIGHT)};
	const std::array<Element, 2> words {
	    Heading(frame, "A card", on_card),
	    Caption(frame, "Padding, corners and a low shadow", on_card)};
	const std::array<Element, 1> one {Caption(frame, "One card", on_card)};
	const std::array<Element, 1> nested {
	    WordCard(frame, {}, std::array<Element, 1> {Caption(frame, "In a card", on_card)})};
	const std::array<Element, 2> pair {WordCard(frame, small, one), WordCard(frame, small, nested)};
	const std::array<Element, 2> labels {
	    Caption(frame, "Do, one card", ink, {.width = Dp(SMALL_CARD_WIDTH)}),
	    Caption(frame, "Don't, a card in a card", ink, {.width = Dp(SMALL_CARD_WIDTH)})};
	const std::array<Element, 3> rows {WordCard(frame, {.width = Dp(CARD_COLUMN_WIDTH)}, words),
	                                   Box(frame, {.gap = Dp(theme.space.s4)}, {}, pair),
	                                   Box(frame, {.gap = Dp(theme.space.s4)}, {}, labels)};
	return Box(frame, ColumnOf(theme.space.s3), {}, rows);
}

// The box and card page's specimens.
Element BuildBox(UiFrame& frame, GalleryState& state) {
	const Color                  ink = StageInk(frame.ActiveTheme(), state.stage);
	const std::array<Element, 2> columns {SurfaceMatrix(frame, ink), CardColumn(frame, ink)};
	return Box(frame, {.gap = Dp(frame.ActiveTheme().space.s6)}, {}, columns);
}

// ---- Text ---------------------------------------------------------------------------------------

// A style's size at an output, as ScaleContext gives it, saying when its minimum holds it up.
std::string SizeText(const Theme& theme, const TypeStyle& style, Vec2 screen) {
	const ScaleContext        scale(theme.scale, screen.x, screen.y);
	const float               size    = scale.FontSize(style);
	constexpr size_t          LONGEST = sizeof("000, floor");
	std::array<char, LONGEST> text {};
	(void)std::snprintf(text.data(), text.size(),
	                    scale.Px(style.size) < style.min_size ? "%.0f, floor" : "%.0f",
	                    static_cast<double>(size));
	return text.data();
}

// The three alignments, each in a slot whose edges show where it sits.
Element AlignSlots(UiFrame& frame, Color ink) {
	const Theme& theme = frame.ActiveTheme();
	struct NamedAlign {
		std::string_view name;
		TextAlign        align;
	};
	std::vector<Element> rows {Heading(frame, "Alignment in a slot", ink)};
	for (const NamedAlign& named:
	     {NamedAlign {"Start", TextAlign::Start}, NamedAlign {"Center", TextAlign::Center},
	      NamedAlign {"End", TextAlign::End}}) {
		const std::array<Element, 1> inside {
		    Text(frame, {named.name, theme.type.body, theme.colors.text.accent, named.align})};
		rows.push_back(Box(frame,
		                   {.direction = FlexDirection::Column,
		                    .justify   = Justify::Center,
		                    .align     = Align::Stretch,
		                    .padding   = DpInsets::Symmetric(Dp(theme.space.s2), Dp()),
		                    .item      = {.width = Dp(SLOT_WIDTH), .height = Dp(SLOT_HEIGHT)}},
		                   {.fill         = theme.colors.surface.raised,
		                    .border       = theme.colors.border.strong,
		                    .border_width = Dp(theme.border.hair)},
		                   inside));
	}
	return Box(frame, ColumnOf(theme.space.s2), {}, rows);
}

// What each style really is at each output, with the output chosen in bold.
Element SizeTable(UiFrame& frame, int chosen, Color ink) {
	const Theme& theme = frame.ActiveTheme();
	struct NamedStyle {
		std::string_view name;
		TypeStyle        style;
	};
	const std::array<NamedStyle, 3> styles {NamedStyle {"body_small", theme.type.body_small},
	                                        {"caption", theme.type.caption},
	                                        {"dense_small", theme.type.dense_small}};
	TypeStyle                       bold = theme.type.caption;
	bold.weight                          = FontWeight::Bold;
	const auto cell_style                = [&](size_t output) {
        return static_cast<int>(output) == chosen ? bold : theme.type.caption;
	};
	const ItemLayout     cell {.width = Dp(TABLE_CELL_WIDTH)};
	std::vector<Element> head {Spacer(frame, {.width = Dp(TABLE_NAME_WIDTH)})};
	for (size_t output = 0; output < STAGE_OUTPUT_NAMES.size(); output++) {
		head.push_back(Text(frame, {STAGE_OUTPUT_NAMES[output], cell_style(output), ink}, cell));
	}
	std::vector<Element> rows {Heading(frame, "Sizes each output reaches", ink),
	                           Box(frame, {}, {}, head)};
	for (const NamedStyle& named: styles) {
		std::vector<Element> cells {
		    Caption(frame, named.name, ink, {.width = Dp(TABLE_NAME_WIDTH)})};
		for (size_t output = 0; output < STAGE_OUTPUTS.size(); output++) {
			cells.push_back(
			    Text(frame,
			         {SizeText(theme, named.style, STAGE_OUTPUTS[output]), cell_style(output), ink},
			         cell));
		}
		rows.push_back(Box(frame, {}, {}, cells));
	}
	return Box(frame, ColumnOf(theme.space.s2), {}, rows);
}

// One paragraph in a slot of the same width, limited to `lines`, with zero for no limit.
Element LimitedParagraph(UiFrame& frame, int lines, Color ink) {
	const Theme& theme = frame.ActiveTheme();
	TextProps    props {PARAGRAPH, theme.type.body_small, ink};
	props.overflow  = TextOverflow::Wrap;
	props.max_lines = lines;
	std::array<char, sizeof("At most 0 lines")> name {};
	(void)std::snprintf(name.data(), name.size(),
	                    lines == 1 ? "At most %d line" : "At most %d lines", lines);
	const std::array<Element, 2> parts {Caption(frame, lines > 0 ? name.data() : "No limit", ink),
	                                    Text(frame, props, {.width = Dp(SLOT_WIDTH)})};
	return Box(frame, ColumnOf(theme.space.s1), {}, parts);
}

// The text page's specimens.
Element BuildText(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme = frame.ActiveTheme();
	const Color                  ink   = StageInk(theme, state.stage);
	const std::array<Element, 2> left {AlignSlots(frame, ink),
	                                   SizeTable(frame, state.stage.output, ink)};
	std::vector<Element>         right {Heading(frame, "Limiting a paragraph's lines", ink)};
	for (int lines = 1; lines <= LONGEST_LIMIT; lines++) {
		right.push_back(LimitedParagraph(frame, lines, ink));
	}
	right.push_back(LimitedParagraph(frame, 0, ink));
	const std::array<Element, 2> columns {Box(frame, ColumnOf(theme.space.s5), {}, left),
	                                      Box(frame, ColumnOf(theme.space.s2), {}, right)};
	return Box(frame, {.gap = Dp(theme.space.s7)}, {}, columns);
}

// ---- Image --------------------------------------------------------------------------------------

// A caption giving a length's size in this output's pixels.
std::string DrawnSize(const UiFrame& frame, std::string_view what, float width, float height) {
	const auto&                                 scale = frame.Scale();
	std::array<char, sizeof(", 00000 x 00000")> size {};
	(void)std::snprintf(size.data(), size.size(), ", %.0f x %.0f",
	                    static_cast<double>(scale.Px(width)),
	                    static_cast<double>(scale.Px(height)));
	return std::string(what) + size.data();
}

// A poster card holding `picture`, which may be missing.
Element PosterCard(UiFrame& frame, const TestPicture& picture, std::string_view words) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 2> parts {
	    Image(frame, picture.texture, picture.size, ImageFit::Cover, Dp(theme.radius.sm),
	          {.width = Dp(POSTER_IMAGE_WIDTH), .height = Dp(POSTER_IMAGE_HEIGHT)}),
	    Caption(frame, words, theme.colors.text.primary, {.width = Dp(POSTER_IMAGE_WIDTH)})};
	Layout layout = ColumnOf(theme.space.s2);
	layout.item   = {.width = Dp(POSTER_CARD_WIDTH)};
	return Card(frame, layout, {}, parts);
}

// The small ways an image is used.
Element SmallImages(UiFrame& frame, const TestPicture& square, const TestPicture& wide, Color ink) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 4> parts {
	    Image(frame, square.texture, square.size, ImageFit::Cover, Dp(theme.radius.sm),
	          {.width = Dp(ICON_SIZE), .height = Dp(ICON_SIZE)}),
	    Caption(frame, DrawnSize(frame, "An icon", ICON_SIZE, ICON_SIZE), ink),
	    Image(frame, wide.texture, wide.size, ImageFit::Cover, Dp(theme.radius.xs),
	          {.width = Dp(THUMB_WIDTH), .height = Dp(THUMB_HEIGHT)}),
	    Caption(frame, DrawnSize(frame, "A row's thumbnail", THUMB_WIDTH, THUMB_HEIGHT), ink)};
	return Box(frame, ColumnOf(theme.space.s2), {}, parts);
}

// The image page's specimens.
Element BuildImage(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme  = frame.ActiveTheme();
	const TestPicture            square = SquarePattern(state);
	const TestPicture            wide   = WidePattern(state);
	const std::array<Element, 3> columns {
	    PosterCard(frame, square,
	               DrawnSize(frame, "A poster", POSTER_IMAGE_WIDTH, POSTER_IMAGE_HEIGHT)),
	    PosterCard(frame, {}, "Its image missing, the card shows through"),
	    SmallImages(frame, square, wide, StageInk(theme, state.stage))};
	return Box(frame, {.gap = Dp(theme.space.s6)}, {}, columns);
}

// ---- Glyphs -------------------------------------------------------------------------------------

// One grade's row, its word and a cup at each size, each keeping the cup's shape.
Element GradeRow(UiFrame& frame, const NamedColor& grade, Color ink) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> cells {Caption(frame, grade.name, ink, {.width = Dp(GLYPH_LABEL_WIDTH)})};
	for (const Vec2 size: CUP_SIZES) {
		const std::array<Element, 1> inside {
		    TrophyCup(frame, {.color = grade.color}, {.width = Dp(size.x), .height = Dp(size.y)})};
		cells.push_back(
		    Box(frame,
		        {.justify = Justify::Center,
		         .align   = Align::Center,
		         .item    = {.width = Dp(GLYPH_CELL_WIDTH), .height = Dp(GLYPH_CELL_HEIGHT)}},
		        {.fill = theme.colors.surface.raised}, inside));
	}
	return Box(frame, {.align = Align::Center, .gap = Dp(theme.space.s1)}, {}, cells);
}

// A cup beside its word, and one stretched out of its shape.
Element CupInUse(UiFrame& frame, Color ink) {
	const Theme&                 theme  = frame.ActiveTheme();
	const auto&                  colors = theme.colors;
	const std::array<Element, 2> line {TrophyCup(frame, {.color = colors.trophy.gold}),
	                                   Text(frame, {"Gold trophy", theme.type.label, ink})};
	const std::array<Element, 4> rows {
	    Caption(frame, "In use, beside a word", ink),
	    Box(frame, {.align = Align::Center, .gap = Dp(theme.space.s2)}, {}, line),
	    Caption(frame, "Don't, stretched out of shape", ink),
	    TrophyCup(frame, {.color = colors.status.warning},
	              {.width = Dp(STRETCHED_CUP), .height = Dp(STRETCHED_CUP)})};
	return Box(frame, ColumnOf(theme.space.s3), {}, rows);
}

// The glyphs page's specimens.
Element BuildGlyphs(UiFrame& frame, GalleryState& state) {
	const Theme&                    theme  = frame.ActiveTheme();
	const Color                     ink    = StageInk(theme, state.stage);
	const auto&                     trophy = theme.colors.trophy;
	const std::array<NamedColor, 4> grades {NamedColor {"Platinum", trophy.platinum},
	                                        {"Gold", trophy.gold},
	                                        {"Silver", trophy.silver},
	                                        {"Bronze", trophy.bronze}};
	std::vector<Element>            rows;
	for (const NamedColor& grade: grades) {
		rows.push_back(GradeRow(frame, grade, ink));
	}
	const std::array<Element, 2> columns {Box(frame, ColumnOf(theme.space.s1), {}, rows),
	                                      CupInUse(frame, ink)};
	return Box(frame, {.gap = Dp(theme.space.s6)}, {}, columns);
}

} // namespace

void DrawBoxPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, BOX_RULE, BuildBox, pane);
}

void DrawTextPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, TEXT_RULE, BuildText, pane);
}

void DrawImagePage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, IMAGE_RULE, BuildImage, pane);
}

void DrawGlyphsPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, GLYPHS_RULE, BuildGlyphs, pane);
}

} // namespace Libs::Graphics::SystemUi::Gallery
