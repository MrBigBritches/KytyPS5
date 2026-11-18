#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_IMAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_IMAGE_H_

#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/geometry.h"

#include <cstdint>

struct ImDrawList;
struct ImTextureData;

namespace Libs::Graphics::SystemUi {

// How an image fits a rectangle of a different shape.
enum class ImageFit : uint8_t {
	Fill,    // stretched to the rectangle
	Contain, // whole, centred, with space left on two sides
	Cover,   // filling the rectangle, centred, with the excess cropped
};

/// Where DrawImage puts an image of `size` pixels given `rect` and `fit`. A contained image is the
/// picture alone, with the empty bars left out, and the others fill `rect`.
[[nodiscard]] Rect PlacedImageRect(Vec2 size, const Rect& rect, ImageFit fit);

/// Draws `texture`, which is `size` pixels, in `rect` with corners rounded by `radius` reference
/// pixels. `opacity` fades it with its element. A null texture draws nothing.
void DrawImage(ImDrawList& list, const UiContext& context, ImTextureData* texture, Vec2 size,
               const Rect& rect, float radius, ImageFit fit, float opacity = 1.0f);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_IMAGE_H_
