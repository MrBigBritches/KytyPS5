#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_UNITS_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_UNITS_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/scale.h"

#include <compare>

namespace Libs::Graphics::SystemUi {

// A length in design units, the theme's reference pixels at 1080p. The element API takes these
// everywhere, so components never handle output pixels and can't mix the two. Each length is
// converted on its own and rounded to a whole pixel, as the theme's tokens always have been, so
// sums of converted lengths match what the components drew before.
struct Dp {
	float value = 0.0f;

	constexpr Dp() = default;
	explicit constexpr Dp(float length): value(length) {}

	constexpr auto operator<=>(const Dp&) const = default;

	// Lengths add and scale as plain numbers do, staying in design units.
	constexpr Dp& operator+=(Dp other) {
		value += other.value;
		return *this;
	}
	constexpr Dp& operator-=(Dp other) {
		value -= other.value;
		return *this;
	}
	friend constexpr Dp operator+(Dp a, Dp b) { return Dp(a.value + b.value); }
	friend constexpr Dp operator-(Dp a, Dp b) { return Dp(a.value - b.value); }
	friend constexpr Dp operator-(Dp length) { return Dp(-length.value); }
	friend constexpr Dp operator*(Dp length, float factor) { return Dp(length.value * factor); }
	friend constexpr Dp operator*(float factor, Dp length) { return Dp(length.value * factor); }
	friend constexpr Dp operator/(Dp length, float divisor) { return Dp(length.value / divisor); }
};

namespace Literals {

/// A length in design units, as in 16_dp.
constexpr Dp operator""_dp(long double length) {
	return Dp(static_cast<float>(length));
}

/// A whole length in design units, as in 16_dp.
constexpr Dp operator""_dp(unsigned long long length) {
	return Dp(static_cast<float>(length));
}

} // namespace Literals

// Insets on each side, in design units, such as a box's padding.
struct DpInsets {
	Dp left;
	Dp top;
	Dp right;
	Dp bottom;

	/// The same inset on all four sides.
	static constexpr DpInsets All(Dp inset) { return {inset, inset, inset, inset}; }

	/// `horizontal` on the left and right, `vertical` at the top and bottom.
	static constexpr DpInsets Symmetric(Dp horizontal, Dp vertical) {
		return {horizontal, vertical, horizontal, vertical};
	}

	constexpr bool operator==(const DpInsets&) const = default;
};

/// `length` in output pixels, rounded to a whole pixel.
[[nodiscard]] inline float ToPixels(const ScaleContext& scale, Dp length) {
	return scale.Px(length.value);
}

/// The design length that converts back to exactly `pixels` whole pixels. For a length worked out
/// in output pixels, such as one measured from text, which would otherwise be rounded a second
/// time. A fractional pixel count is rounded by the conversion, as any length is.
[[nodiscard]] inline Dp FromPixels(const ScaleContext& scale, float pixels) {
	return Dp(pixels / scale.Scale());
}

/// A border width in output pixels, never below one pixel.
[[nodiscard]] inline float ToBorderPixels(const ScaleContext& scale, Dp width) {
	return scale.Border(width.value);
}

/// `insets` in output pixels, each side rounded on its own.
[[nodiscard]] inline EdgeInsets ToPixels(const ScaleContext& scale, const DpInsets& insets) {
	return {ToPixels(scale, insets.left), ToPixels(scale, insets.top),
	        ToPixels(scale, insets.right), ToPixels(scale, insets.bottom)};
}

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_UNITS_H_
