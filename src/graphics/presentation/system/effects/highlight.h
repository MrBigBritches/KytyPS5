#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_HIGHLIGHT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_HIGHLIGHT_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/core/theme.h"

struct ImDrawList;

// How focus and selection are shown. A controller drives the UI, so the focused element must be
// obvious on any game frame, light or dark.
namespace Libs::Graphics::SystemUi {

/// The focus ring around `rect`, drawn just outside it so it never covers the element. A light
/// ring with a dark keyline outside it reads on light and dark frames alike, and a soft glow
/// outside both draws the eye.
void DrawFocusRing(ImDrawList& list, const Rect& rect, float rounding, const Theme& theme,
                   const ScaleContext& scale, float opacity = 1.0f);

/// The selected state of a row or card. A faint accent fill covers it and a rounded accent bar
/// marks its left edge between the rounded corners, so selection still shows when focus is
/// elsewhere.
void DrawSelection(ImDrawList& list, const Rect& rect, float rounding, const Theme& theme,
                   const ScaleContext& scale, float opacity = 1.0f);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_EFFECTS_HIGHLIGHT_H_
