#include "graphics/presentation/system/gallery/imagesPage.h"

#include "graphics/presentation/system/components/image.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/capture.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/pageParts.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view RULE =
    "Cover for thumbnails and posters. Contain when nothing may be cropped. Fill only for "
    "gradients. Always give a surface behind an image.";

// The test pattern, in its own pixels.
constexpr int   CHECK_SIZE     = 32;
constexpr float CROSS_HALF     = 1.5f;
constexpr float RING_THICKNESS = 6.0f;
constexpr int   NOTCH_SIZE     = 12;
constexpr int   NOTCH_GAP      = 6;
constexpr int   CORNERS        = 4;
constexpr int   CHANNELS       = 4;
// More than either pattern's longer side.
constexpr int MAX_PATTERN_SIDE = 512;

// The page's layout, in reference pixels.
constexpr float SOURCE_COLUMN = 200.0f;
constexpr float SOURCE_WIDTH  = 160.0f;
constexpr float WIDE_SLOT_W   = 320.0f;
constexpr float WIDE_SLOT_H   = 180.0f;
constexpr float SQUARE_SLOT   = 180.0f;
constexpr float FIT_COLUMN    = 360.0f;
constexpr float CORNER_TILE   = 96.0f;
constexpr float MISSING_W     = 160.0f;
constexpr float MISSING_H     = 90.0f;

// A way to fit an image and what to say about it.
struct FitChoice {
	ImageFit         fit;
	std::string_view caption;
};

constexpr std::array<FitChoice, 3> FITS {
    FitChoice {ImageFit::Fill, "Fill, stretched, so the ring is an oval"},
    FitChoice {ImageFit::Contain, "Contain, whole, with bars around it"},
    FitChoice {ImageFit::Cover, "Cover, filling the slot, its corners cropped"}};

// Whether a pixel is one of the notches. Each corner counts its notches along its top or bottom
// edge, from the corner inward, with the top left having one and the rest following clockwise.
bool IsNotch(int x, int y, int width, int height) {
	const bool left      = x < width / 2;
	const bool top       = y < height / 2;
	const int  from_side = left ? x : width - 1 - x;
	const int  from_edge = top ? y : height - 1 - y;
	if (from_edge >= NOTCH_SIZE) {
		return false;
	}
	const int corner = top ? (left ? 0 : 1) : (left ? CORNERS - 1 : 2);
	const int pitch  = NOTCH_SIZE + NOTCH_GAP;
	return from_side / pitch <= corner && from_side % pitch < NOTCH_SIZE;
}

// The pattern's colour at a pixel, each layer over the one before.
Color PatternAt(int x, int y, int width, int height) {
	const auto  wide        = static_cast<float>(width);
	const auto  tall        = static_cast<float>(height);
	const float pos_x       = static_cast<float>(x) + 0.5f;
	const float pos_y       = static_cast<float>(y) + 0.5f;
	const float radius      = std::min(wide, tall) / 2.0f;
	const float from_centre = std::hypot(pos_x - wide / 2.0f, pos_y - tall / 2.0f);
	// How far the pixel is from each diagonal, by the area its triangle makes with the corners.
	const float diagonal     = std::hypot(wide, tall);
	const float from_falling = std::fabs(pos_y * wide - pos_x * tall) / diagonal;
	const float from_rising  = std::fabs((tall - pos_y) * wide - pos_x * tall) / diagonal;
	if (IsNotch(x, y, width, height)) {
		return PATTERN_NOTCH;
	}
	if (from_centre <= radius && from_centre >= radius - RING_THICKNESS) {
		return PATTERN_RING;
	}
	if (std::min(from_falling, from_rising) <= CROSS_HALF) {
		return PATTERN_CROSS;
	}
	return (x / CHECK_SIZE + y / CHECK_SIZE) % 2 == 0 ? PATTERN_CHECK_DARK : PATTERN_CHECK_LIGHT;
}

// Encodes the pattern at a size and gives it to the store, which decodes it like any other image.
std::optional<ImageId> AddPattern(ImageStore& store, int width, int height) {
	const std::vector<std::byte> encoded = EncodePng(TestPatternRgba(width, height), width, height);
	return encoded.empty() ? std::nullopt : store.Add(encoded, MAX_PATTERN_SIDE);
}

// Loads the patterns the first time the page is drawn with a store.
void LoadPatterns(GalleryState& state) {
	if (state.images == nullptr || state.images_page.tried) {
		return;
	}
	state.images_page.tried  = true;
	state.images_page.square = AddPattern(*state.images, PATTERN_SQUARE_SIDE, PATTERN_SQUARE_SIDE);
	state.images_page.wide   = AddPattern(*state.images, PATTERN_WIDE_WIDTH, PATTERN_WIDE_HEIGHT);
}

// The picture for `id`.
TestPicture PictureOf(const GalleryState& state, const std::optional<ImageId>& id) {
	if (state.images == nullptr || !id.has_value()) {
		return {};
	}
	return {state.images->Texture(*id), state.images->Size(*id)};
}

// A slot on an inset surface, so the bars a contained image leaves show, holding `picture`. The
// outline is over the picture, so a slot's edge shows whole whatever the picture covers, and one
// with no picture shows too.
Element Slot(UiFrame& frame, const TestPicture& picture, ImageFit fit, float width, float height) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 2> layers {
	    Image(frame, picture.texture, picture.size, fit, Dp()),
	    Box(frame, {},
	        {.border = theme.colors.border.strong, .border_width = Dp(theme.border.hair)})};
	return Box(frame, {.item = {.width = Dp(width), .height = Dp(height)}, .stacked = true},
	           {.fill = theme.colors.surface.inset}, layers);
}

// One row of the Fit view, the picture as it is and then in a slot each way it can fit.
Element FitRow(UiFrame& frame, std::string_view source, const TestPicture& picture,
               float slot_width, float slot_height, float source_height) {
	const Theme&                 theme = frame.ActiveTheme();
	const auto&                  text  = theme.colors.text;
	const std::array<Element, 2> source_parts {
	    Image(frame, picture.texture, picture.size, ImageFit::Fill, Dp(),
	          {.width = Dp(SOURCE_WIDTH), .height = Dp(source_height)}),
	    Text(frame, {source, theme.type.caption, text.secondary})};
	const Layout         column {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)};
	std::vector<Element> cells {Box(frame,
	                                {.direction = FlexDirection::Column,
	                                 .gap       = Dp(theme.space.s1),
	                                 .item      = {.width = Dp(SOURCE_COLUMN)}},
	                                {}, source_parts)};
	for (const FitChoice& choice: FITS) {
		const std::array<Element, 2> parts {
		    Slot(frame, picture, choice.fit, slot_width, slot_height),
		    Text(frame, {choice.caption, theme.type.caption, text.secondary})};
		Layout cell     = column;
		cell.item.width = Dp(FIT_COLUMN);
		cells.push_back(Box(frame, cell, {}, parts));
	}
	return Box(frame, {}, {}, cells);
}

// The Fit view, a square picture in wide slots and a wide picture in square ones.
Element FitView(UiFrame& frame, GalleryState& state) {
	const float wide_source =
	    SOURCE_WIDTH * static_cast<float>(PATTERN_WIDE_HEIGHT) / PATTERN_WIDE_WIDTH;
	const std::array<Element, 2> rows {FitRow(frame, "Square, 256 x 256", SquarePattern(state),
	                                          WIDE_SLOT_W, WIDE_SLOT_H, SOURCE_WIDTH),
	                                   FitRow(frame, "Wide, 320 x 180", WidePattern(state),
	                                          SQUARE_SLOT, SQUARE_SLOT, wide_source)};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s5)},
	           {}, rows);
}

// The radius scale on the square picture, each tile named.
Element CornerTiles(UiFrame& frame, const TestPicture& picture) {
	const Theme&      theme = frame.ActiveTheme();
	const RadiusScale radii = theme.radius;
	struct NamedRadius {
		std::string_view name;
		float            radius;
	};
	const std::array<NamedRadius, 6> named {NamedRadius {"xs", radii.xs},
	                                        {"sm", radii.sm},
	                                        {"md", radii.md},
	                                        {"lg", radii.lg},
	                                        {"xl", radii.xl},
	                                        {"full", RADIUS_FULL}};
	std::vector<Element>             tiles;
	for (const NamedRadius& radius: named) {
		const std::array<Element, 2> parts {
		    Image(frame, picture.texture, picture.size, ImageFit::Cover, Dp(radius.radius),
		          {.width = Dp(CORNER_TILE), .height = Dp(CORNER_TILE)}),
		    Text(frame, {radius.name, theme.type.caption, theme.colors.text.secondary})};
		tiles.push_back(
		    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, parts));
	}
	return Box(frame, {.gap = Dp(theme.space.s4)}, {}, tiles);
}

// The Corners view, the radii and what a missing image leaves behind.
Element CornersView(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 5> rows {
	    SectionHeading(frame, "Corners"), CornerTiles(frame, SquarePattern(state)),
	    SectionHeading(frame, "No image"), Slot(frame, {}, ImageFit::Cover, MISSING_W, MISSING_H),
	    Text(frame, {"A missing image draws nothing, so what shows is the surface behind it.",
	                 theme.type.caption, theme.colors.text.secondary})};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s3)}, {}, rows);
}

} // namespace

TestPicture SquarePattern(GalleryState& state) {
	LoadPatterns(state);
	return PictureOf(state, state.images_page.square);
}

TestPicture WidePattern(GalleryState& state) {
	LoadPatterns(state);
	return PictureOf(state, state.images_page.wide);
}

std::vector<uint8_t> TestPatternRgba(int width, int height) {
	std::vector<uint8_t> rgba(static_cast<size_t>(width) * static_cast<size_t>(height) * CHANNELS);
	size_t               at = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			const Color color = PatternAt(x, y, width, height);
			rgba[at++]        = color.r;
			rgba[at++]        = color.g;
			rgba[at++]        = color.b;
			rgba[at++]        = color.a;
		}
	}
	return rgba;
}

void DrawImagesPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	const PageHead head = BuildPageHead(frame, page, RULE, &state.images_view);
	const int view = std::clamp(state.images_view, 0, static_cast<int>(IMAGES_VIEWS.size()) - 1);
	const std::array<Element, 2> rows {head.root, view == 0 ? FitView(frame, state)
	                                                        : CornersView(frame, state)};
	const Element                root =
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s4)},
	        {}, rows);
	ShowPage(frame, state, root, head.tabs, view, pane);
}

} // namespace Libs::Graphics::SystemUi::Gallery
