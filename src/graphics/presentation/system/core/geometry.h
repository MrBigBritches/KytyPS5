#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_GEOMETRY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_GEOMETRY_H_

#include <algorithm>
#include <cmath>

// Plain geometry for the system UI, in output pixels with y growing downward. It has no ImGui
// types, so layout code can be tested without a GPU.
namespace Libs::Graphics::SystemUi {

/// `value` rounded to the nearest whole pixel, with halves rounded up. Unlike std::round, which
/// rounds halves away from zero, this gives a span the same length wherever it sits, including
/// below zero while scrolling. It matches ImGui's own pixel rounding.
[[nodiscard]] inline float SnapPixel(float value) {
	constexpr float HALF = 0.5f;
	return std::floor(value + HALF);
}

// A point or a size.
struct Vec2 {
	float x = 0.0f;
	float y = 0.0f;

	constexpr bool operator==(const Vec2&) const = default;
};

// Space kept clear on each side of a rectangle, such as padding.
struct EdgeInsets {
	float left   = 0.0f;
	float top    = 0.0f;
	float right  = 0.0f;
	float bottom = 0.0f;

	/// The same inset on all four sides.
	static constexpr EdgeInsets All(float inset) { return {inset, inset, inset, inset}; }

	/// `horizontal` on the left and right, `vertical` at the top and bottom.
	static constexpr EdgeInsets Symmetric(float horizontal, float vertical) {
		return {horizontal, vertical, horizontal, vertical};
	}

	constexpr bool operator==(const EdgeInsets&) const = default;
};

// A rectangle from its top-left corner and its size.
struct Rect {
	float x      = 0.0f;
	float y      = 0.0f;
	float width  = 0.0f;
	float height = 0.0f;

	[[nodiscard]] constexpr float Right() const { return x + width; }
	[[nodiscard]] constexpr float Bottom() const { return y + height; }
	[[nodiscard]] constexpr Vec2  Position() const { return {x, y}; }
	[[nodiscard]] constexpr Vec2  Size() const { return {width, height}; }
	[[nodiscard]] constexpr Vec2  Center() const { return {x + width / 2.0f, y + height / 2.0f}; }

	/// True when `point` lies inside, counting the top and left edges but not the bottom and
	/// right ones, so neighbouring rectangles never both contain a point.
	[[nodiscard]] constexpr bool Contains(Vec2 point) const {
		return point.x >= x && point.y >= y && point.x < Right() && point.y < Bottom();
	}

	/// The rectangle shrunk by `insets`. Insets larger than the rectangle leave it empty, never
	/// negative.
	[[nodiscard]] constexpr Rect Inset(const EdgeInsets& insets) const {
		return {x + insets.left, y + insets.top, std::max(0.0f, width - insets.left - insets.right),
		        std::max(0.0f, height - insets.top - insets.bottom)};
	}

	constexpr bool operator==(const Rect&) const = default;
};

/// The largest rectangle inside `rect` that stays clear of its corners rounded by `radius`. Each
/// side moves in by radius * (1 - 1/sqrt(2)), which puts the result's corners on the rounded
/// outline, so clipping to it keeps a drawing inside the rounded shape except for slivers along
/// the straight edges near each corner.
[[nodiscard]] inline Rect InsideCorners(const Rect& rect, float radius) {
	constexpr float CORNER_INSET = 0.29289322f; // 1 - 1/sqrt(2)
	return rect.Inset(EdgeInsets::All(std::max(0.0f, radius) * CORNER_INSET));
}

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_GEOMETRY_H_
