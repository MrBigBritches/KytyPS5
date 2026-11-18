#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_LAYOUT_ANCHOR_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_LAYOUT_ANCHOR_H_

#include "graphics/presentation/system/core/geometry.h"

#include <cstdint>

// Places panels such as toasts and heads-up displays against an area's corners, edges or centre.
// The area is usually the safe area or the content frame from ScaleContext.
namespace Libs::Graphics::SystemUi {

enum class Anchor : uint8_t {
	TopLeft,
	Top,
	TopRight,
	Left,
	Center,
	Right,
	BottomLeft,
	Bottom,
	BottomRight,
};

/// A rectangle of `size` placed at `anchor` inside `area`, with all four edges on whole pixels, so
/// a fractional measured size is snapped too. A panel larger than the area is still aligned the
/// same way and runs past the area's edges.
[[nodiscard]] Rect Place(const Rect& area, Vec2 size, Anchor anchor);

/// The part of `area` below `edge`, such as the space left under another panel, so panels can
/// stack without overlapping. An edge above the area leaves it whole, and one below it leaves an
/// empty rectangle at its bottom.
[[nodiscard]] Rect BelowEdge(const Rect& area, float edge);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_LAYOUT_ANCHOR_H_
