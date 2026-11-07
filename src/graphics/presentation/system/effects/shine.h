#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_SHINE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_SHINE_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/core/theme.h"

struct ImDrawList;

namespace Libs::Graphics::SystemUi {

/// A slanted band of light `progress` of the way across `rect`, drawn over the element. At 0 the
/// band is just off the left edge and at 1 just off the right, so a sweep runs `progress` from 0
/// to 1, usually through the Standard curve. Progress outside 0..1 draws nothing.
///
/// The band's brightness comes from the shine tokens, replacing the colour's own alpha, then
/// `opacity` fades it. It is clipped to `rect` pulled in from the corners by about 0.29 of
/// `rounding`, which puts the clip's corners on the rounded outline, so the band barely shows past
/// a rounded corner. `rect` and `rounding` are in output pixels. The list's current texture must
/// be the font atlas, as it is unless an image texture has been pushed.
void DrawShine(ImDrawList& list, const Rect& rect, float rounding, float progress,
               const ShineTokens& shine, Color color, const ScaleContext& scale,
               float opacity = 1.0f);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_SHINE_H_
