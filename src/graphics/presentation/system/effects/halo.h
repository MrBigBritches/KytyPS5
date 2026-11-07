#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_HALO_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_HALO_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/core/theme.h"

struct ImDrawList;

// Soft shadows and glows, drawn as a mesh whose alpha fades with distance from the element. ImGui's
// draw lists have no blur, so the fade stands in for one. Rectangles and corner radii are in output
// pixels, while elevation and glow tokens are in reference pixels. A colour's own alpha
// multiplies the token's peak, so theme colours for effects are opaque.
namespace Libs::Graphics::SystemUi {

/// A shadow under `rect`, whose corners have `rounding` pixels. Draw it before the element.
/// `elevation` is in reference pixels. `opacity` fades the whole shadow, as an animation fades its
/// element. Level 0, a zero opacity or an empty colour draws nothing.
void DrawShadow(ImDrawList& list, const Rect& rect, float rounding, const Elevation& elevation,
                Color color, const ScaleContext& scale, float opacity = 1.0f);

/// A halo of `color` around `rect`, as wide as `glow.radius` reference pixels. Strongest next to
/// the element and fading outward. It is filled, so draw it behind the element.
void DrawGlow(ImDrawList& list, const Rect& rect, float rounding, const Glow& glow, Color color,
              const ScaleContext& scale, float opacity = 1.0f);

/// A halo like DrawGlow that never covers the element, for drawing on top of it, such as around a
/// focused control. It is made of thin rings, about 1.5 pixels each, from the element's edge
/// outward, fading quadratically from the peak to nothing.
void DrawGlowOutside(ImDrawList& list, const Rect& rect, float rounding, const Glow& glow,
                     Color color, const ScaleContext& scale, float opacity = 1.0f);

/// A rounded outline `thickness` pixels wide whose inner edge lies `inner` pixels outside `rect`.
void StrokeOutside(ImDrawList& list, const Rect& rect, float rounding, float inner, float thickness,
                   Color color);

/// `rect` grown by `amount` pixels on every side.
[[nodiscard]] Rect Expand(const Rect& rect, float amount);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_HALO_H_
