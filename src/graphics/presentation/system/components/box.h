#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_BOX_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_BOX_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/geometry.h"

struct ImDrawList;

namespace Libs::Graphics::SystemUi {

// How a box looks. Lengths are reference pixels. A transparent border or a zero border width
// draws no border, and elevation level 0 casts no shadow.
struct BoxStyle {
	Color     fill;
	float     radius = 0.0f;
	Color     border;
	float     border_width = 0.0f;
	Elevation elevation;
};

/// Draws a box filling `rect`: its shadow, its fill, then its border just inside its edge, so the
/// border never makes the box bigger. `opacity` fades all of it with its element.
void DrawBox(ImDrawList& list, const UiContext& context, const Rect& rect, const BoxStyle& style,
             float opacity = 1.0f);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_BOX_H_
