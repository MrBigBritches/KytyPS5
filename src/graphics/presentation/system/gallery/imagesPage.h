#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_IMAGESPAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_IMAGESPAGE_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/imgui/imageStore.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The images page's views, in tab order.
inline constexpr std::array<std::string_view, 2> IMAGES_VIEWS {"Fit", "Corners"};

// The test pattern's sizes, one square and one wide, so each can go into a slot of the other shape.
inline constexpr int PATTERN_SQUARE_SIDE = 256;
inline constexpr int PATTERN_WIDE_WIDTH  = 320;
inline constexpr int PATTERN_WIDE_HEIGHT = 180;

// The test pattern's colours.
inline constexpr Color PATTERN_CHECK_DARK  = Rgb(0x2A3140);
inline constexpr Color PATTERN_CHECK_LIGHT = Rgb(0x3A4252);
inline constexpr Color PATTERN_CROSS       = Rgb(0x4DA6FF);
inline constexpr Color PATTERN_RING        = Rgb(0xDDE1EA);
inline constexpr Color PATTERN_NOTCH       = Rgb(0xFFCF52);

// What the images page keeps from frame to frame. The images last as long as the store, as only it
// can free them, so all of this survives a reset.
struct ImagesPageState {
	std::optional<ImageId> square;
	std::optional<ImageId> wide;
	// Whether the images have been asked for, as a failure would otherwise repeat on every frame.
	bool tried = false;
};

// A picture of the test pattern, a null texture when it couldn't be had.
struct TestPicture {
	ImTextureData* texture = nullptr;
	Vec2           size {};
};

/// The square test pattern, loaded the first time it is asked for with a store.
[[nodiscard]] TestPicture SquarePattern(GalleryState& state);

/// The wide test pattern, loaded the first time it is asked for with a store.
[[nodiscard]] TestPicture WidePattern(GalleryState& state);

/// The test pattern as tightly packed RGBA, opaque. Over a checker it has a ring as tall as the
/// image, which stretching makes an oval, a cross from corner to corner, and in the corners 1, 2, 3
/// and 4 notches, clockwise from the top left, so what a crop lost can be counted. The size must be
/// positive, and at least 132 by 24 for all the notches to fit.
[[nodiscard]] std::vector<uint8_t> TestPatternRgba(int width, int height);

/// Draws the images page into `pane`. Fit shows the pattern in slots of another shape, each way an
/// image can fit. Corners shows the radius scale on an image, and what a missing image leaves.
/// Slots stay empty when `state.images` is null.
void DrawImagesPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_IMAGESPAGE_H_
