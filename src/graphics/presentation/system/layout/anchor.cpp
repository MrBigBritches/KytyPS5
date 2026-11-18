#include "graphics/presentation/system/layout/anchor.h"

#include <algorithm>

namespace Libs::Graphics::SystemUi {

namespace {

// How far along an axis an anchor sits, as a fraction from the start edge to the end edge.
constexpr float START  = 0.0f;
constexpr float MIDDLE = 0.5f;
constexpr float END    = 1.0f;

// The anchor's horizontal and vertical fractions.
Vec2 Fractions(Anchor anchor) {
	switch (anchor) {
		case Anchor::TopLeft: return {START, START};
		case Anchor::Top: return {MIDDLE, START};
		case Anchor::TopRight: return {END, START};
		case Anchor::Left: return {START, MIDDLE};
		case Anchor::Center: return {MIDDLE, MIDDLE};
		case Anchor::Right: return {END, MIDDLE};
		case Anchor::BottomLeft: return {START, END};
		case Anchor::Bottom: return {MIDDLE, END};
		case Anchor::BottomRight: return {END, END};
	}
	return {START, START};
}

} // namespace

Rect Place(const Rect& area, Vec2 size, Anchor anchor) {
	const auto  fraction = Fractions(anchor);
	const float left     = area.x + (area.width - size.x) * fraction.x;
	const float top      = area.y + (area.height - size.y) * fraction.y;
	const float x        = SnapPixel(left);
	const float y        = SnapPixel(top);
	return {x, y, SnapPixel(left + size.x) - x, SnapPixel(top + size.y) - y};
}

Rect BelowEdge(const Rect& area, float edge) {
	// Written with min and max rather than clamp, so a NaN edge leaves the area whole.
	const float top = std::max(area.y, std::min(edge, area.Bottom()));
	return {area.x, top, area.width, std::max(0.0f, area.Bottom() - top)};
}

} // namespace Libs::Graphics::SystemUi
