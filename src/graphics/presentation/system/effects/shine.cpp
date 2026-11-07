#include "graphics/presentation/system/effects/shine.h"

#include "graphics/presentation/system/imgui/convert.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace Libs::Graphics::SystemUi {

namespace {

constexpr float HALF_TURN_DEGREES = 180.0f;
constexpr int   QUAD_VERTICES     = 4;
constexpr int   QUAD_INDICES      = 6;
constexpr int   STRIPS            = 3;

// The band's shape. Its bottom edge starts at `left`, its top edge is `skew` further right, and
// it is `width` wide. `white_pixel` is where the font atlas is solid white.
struct Band {
	float  left;
	float  skew;
	float  width;
	float  top;
	float  bottom;
	ImVec2 white_pixel;
};

// One slanted strip of the band, from `from` to `to` across its width, fading from `from_color`
// to `to_color`. Space for it must already be reserved.
void DrawStrip(ImDrawList& list, const Band& band, float from, float to, ImU32 from_color,
               ImU32 to_color) {
	const auto base = static_cast<ImDrawIdx>(list._VtxCurrentIdx);
	list.PrimWriteVtx({band.left + from, band.bottom}, band.white_pixel, from_color);
	list.PrimWriteVtx({band.left + to, band.bottom}, band.white_pixel, to_color);
	list.PrimWriteVtx({band.left + band.skew + to, band.top}, band.white_pixel, to_color);
	list.PrimWriteVtx({band.left + band.skew + from, band.top}, band.white_pixel, from_color);
	constexpr std::array<ImDrawIdx, QUAD_INDICES> CORNERS {0, 1, 2, 0, 2, 3};
	for (const auto corner: CORNERS) {
		list.PrimWriteIdx(static_cast<ImDrawIdx>(base + corner));
	}
}

} // namespace

void DrawShine(ImDrawList& list, const Rect& rect, float rounding, float progress,
               const ShineTokens& shine, Color color, const ScaleContext& scale, float opacity) {
	const Color bright = color.WithAlpha(shine.peak_alpha).MulAlpha(opacity);
	if (!(progress >= 0.0f && progress <= 1.0f) || rect.width <= 0.0f || rect.height <= 0.0f ||
	    bright.a == 0) {
		return;
	}
	const float width = std::max(rect.width * shine.width_fraction, scale.Px(shine.min_width));
	const float skew =
	    rect.height * std::tan(shine.angle_degrees * std::numbers::pi_v<float> / HALF_TURN_DEGREES);
	// From fully off the left edge to fully off the right.
	const float start = rect.x - width - skew;
	const Band  band {start + (rect.Right() - start) * progress,
                     skew,
                     width,
                     rect.y,
                     rect.Bottom(),
                     ImGui::GetFontTexUvWhitePixel()};
	const float plateau  = ClampUnit(shine.plateau_fraction) * width;
	const float rise_end = (width - plateau) / 2.0f;
	const ImU32 clear    = ToImU32(color.WithAlpha(0.0f));
	const ImU32 lit      = ToImU32(bright);
	const auto  clip     = InsideCorners(rect, rounding);

	list.PushClipRect(TopLeft(clip), BottomRight(clip), true);
	list.PrimReserve(STRIPS * QUAD_INDICES, STRIPS * QUAD_VERTICES);
	DrawStrip(list, band, 0.0f, rise_end, clear, lit);
	DrawStrip(list, band, rise_end, rise_end + plateau, lit, lit);
	DrawStrip(list, band, rise_end + plateau, width, lit, clear);
	list.PopClipRect();
}

} // namespace Libs::Graphics::SystemUi
