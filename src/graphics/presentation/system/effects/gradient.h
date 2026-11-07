#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_GRADIENT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_GRADIENT_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/geometry.h"

struct ImDrawList;

namespace Libs::Graphics::SystemUi {

/// Fills `rect`, with corners of `rounding` pixels, with a gradient from `top` to `bottom`, alpha
/// included. ImGui's own gradient fill can't round corners, so this fills a rounded rectangle and
/// colours each vertex by its height. Anti-aliased edges keep their soft fringe. `rect` and
/// `rounding` are in output pixels.
void DrawVerticalGradient(ImDrawList& list, const Rect& rect, float rounding, Color top,
                          Color bottom, float opacity = 1.0f);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_GRADIENT_H_
