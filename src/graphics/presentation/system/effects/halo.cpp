#include "graphics/presentation/system/effects/halo.h"

#include "graphics/presentation/system/imgui/convert.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>

namespace Libs::Graphics::SystemUi {

namespace {

// The fade is split into bands about this many pixels wide, so it follows its curve as closely at
// 4K as at 1080p. The GPU blends each band's alpha between its edges, so the bands don't show.
constexpr float BAND_WIDTH = 6.0f;
constexpr int   MIN_BANDS  = 3;
constexpr int   MAX_BANDS  = 12;
// A corner of the halo's outer edge is cut into segments about this many pixels long. The edge fades
// to nothing, so it needs far fewer points than a shape with a crisp edge would.
constexpr float CORNER_SEGMENT_LENGTH = 8.0f;
constexpr int   MIN_CORNER_SEGMENTS   = 4;
constexpr int   MAX_CORNER_SEGMENTS   = 24;
constexpr int   CORNERS               = 4;
constexpr float QUARTER_TURN          = std::numbers::pi_v<float> / 2.0f;
// The most points an outline can have, one more than its segments at each corner.
constexpr int MAX_OUTLINE_POINTS = CORNERS * (MAX_CORNER_SEGMENTS + 1);
// Each band is a strip of two triangles per outline point, and the centre a fan.
constexpr int TRIANGLES_PER_QUAD   = 2;
constexpr int INDICES_PER_TRIANGLE = 3;
// Rings of an outside glow are this thin, so they blend into a halo rather than stripes.
constexpr float RING_WIDTH     = 1.5f;
constexpr int   MAX_RING_COUNT = 32;

// How many pieces of `piece` pixels fit `length` pixels, within the limits. The clamp comes before
// the conversion, since a huge length would not fit an integer. `length` must not be NaN, which the
// clamp would pass through.
int PieceCount(float length, float piece, int minimum, int maximum) {
	return static_cast<int>(std::ceil(std::clamp(length / piece, static_cast<float>(minimum),
	                                             static_cast<float>(maximum))));
}

// How strong a halo is a fraction `t` of the way out, from 1 at the element to 0 at the outer edge.
// It is a smoothstep. A straight ramp would meet the background at an angle, which the eye picks
// out as a line, while this one arrives flat at both ends.
float Falloff(float t) {
	return 1.0f - t * t * (3.0f - 2.0f * t);
}

// A rounded rectangle's outline as points that can be pushed outward together. Each point moves
// along its own direction from a fixed corner centre, so every expansion of the outline has the
// same number of points, and neighbouring rings can be joined by a strip without any gaps.
struct HaloOutline {
	std::array<ImVec2, MAX_OUTLINE_POINTS> centres;
	std::array<ImVec2, MAX_OUTLINE_POINTS> directions;
	int                                    count  = 0;
	float                                  radius = 0.0f;

	// Point `index` of the outline grown by `expansion` pixels.
	[[nodiscard]] ImVec2 At(int index, float expansion) const {
		const float reach = radius + expansion;
		return {centres[index].x + directions[index].x * reach,
		        centres[index].y + directions[index].y * reach};
	}
};

// The outline of `rect` with corners of `rounding` pixels, with corners fine enough for the outer
// edge `spread` pixels further out. It runs clockwise from the top right corner.
HaloOutline MakeOutline(const Rect& rect, float rounding, float spread) {
	HaloOutline outline;
	// Written so a NaN rounding gives no corners, as a NaN would reach the point counts through a
	// clamp. A negative size has no room for corners either.
	const float limit = std::max(0.0f, std::min(rect.width, rect.height) / 2.0f);
	outline.radius    = rounding > 0.0f ? std::min(rounding, limit) : 0.0f;
	const int segments = PieceCount((outline.radius + spread) * QUARTER_TURN, CORNER_SEGMENT_LENGTH,
	                                MIN_CORNER_SEGMENTS, MAX_CORNER_SEGMENTS);
	const std::array<ImVec2, CORNERS> corners {
	    ImVec2 {rect.Right() - outline.radius, rect.y + outline.radius},
	    ImVec2 {rect.Right() - outline.radius, rect.Bottom() - outline.radius},
	    ImVec2 {rect.x + outline.radius, rect.Bottom() - outline.radius},
	    ImVec2 {rect.x + outline.radius, rect.y + outline.radius}};
	for (int corner = 0; corner < CORNERS; corner++) {
		// The top right corner's arc starts pointing straight up, and each corner turns on from there.
		const float start = QUARTER_TURN * static_cast<float>(corner - 1);
		for (int step = 0; step <= segments; step++) {
			const float angle =
			    start + QUARTER_TURN * static_cast<float>(step) / static_cast<float>(segments);
			outline.centres[outline.count]    = corners[corner];
			outline.directions[outline.count] = {std::cos(angle), std::sin(angle)};
			outline.count++;
		}
	}
	return outline;
}

// Writes the triangles of a halo whose vertices start at `first`, ring after ring from the inside.
// The innermost ring is filled as a fan, since the element in front may be translucent, and each
// pair of neighbouring rings is joined by a strip, wrapping round to the first point.
void WriteHaloIndices(ImDrawList& list, unsigned int first, int points, int bands) {
	const auto write = [&list](unsigned int index) {
		list.PrimWriteIdx(static_cast<ImDrawIdx>(index));
	};
	for (int point = 1; point + 1 < points; point++) {
		write(first);
		write(first + point);
		write(first + point + 1);
	}
	for (int band = 0; band < bands; band++) {
		const unsigned int inner = first + static_cast<unsigned int>(band * points);
		const unsigned int outer = inner + static_cast<unsigned int>(points);
		for (int point = 0; point < points; point++) {
			const int next = (point + 1) % points;
			write(inner + point);
			write(outer + point);
			write(outer + next);
			write(inner + point);
			write(outer + next);
			write(inner + next);
		}
	}
}

// Draws a soft halo around `rect`, `spread` pixels wide and moved down by `offset_y`, as one mesh.
// It reaches `peak` times the colour's own alpha at the element and is filled under it, and alpha
// is set at each vertex, so the GPU blends it smoothly across each band. Nothing overlaps, so it
// has none of the steps that stacked shapes show, and costs one layer of overdraw.
void DrawSoftHalo(ImDrawList& list, const Rect& rect, float rounding, float spread, float offset_y,
                  float peak, Color color) {
	if (!(spread > 0.0f) || !std::isfinite(spread) || !(peak > 0.0f) || color.a == 0) {
		return;
	}
	const HaloOutline outline =
	    MakeOutline({rect.x, rect.y + offset_y, rect.width, rect.height}, rounding, spread);
	const int points       = outline.count;
	const int bands        = PieceCount(spread, BAND_WIDTH, MIN_BANDS, MAX_BANDS);
	const int fan_indices  = (points - 2) * INDICES_PER_TRIANGLE;
	const int band_indices = points * TRIANGLES_PER_QUAD * INDICES_PER_TRIANGLE;
	list.PrimReserve(fan_indices + bands * band_indices, points * (bands + 1));
	// Read after reserving, which can start a new vertex range when the list is nearly full.
	WriteHaloIndices(list, list._VtxCurrentIdx, points, bands);
	const ImVec2 white = ImGui::GetFontTexUvWhitePixel();
	for (int band = 0; band <= bands; band++) {
		const float t    = static_cast<float>(band) / static_cast<float>(bands);
		const ImU32 ring = ToImU32(color.MulAlpha(peak * Falloff(t)));
		for (int point = 0; point < points; point++) {
			list.PrimWriteVtx(outline.At(point, spread * t), white, ring);
		}
	}
}

} // namespace

Rect Expand(const Rect& rect, float amount) {
	return rect.Inset(EdgeInsets::All(-amount));
}

void StrokeOutside(ImDrawList& list, const Rect& rect, float rounding, float inner, float thickness,
                   Color color) {
	// ImGui centres strokes on their path, so the path sits half the thickness further out.
	// AddRect also pulls its path half a pixel inside the box it is given, to keep thin lines
	// crisp, so the box is grown by that half pixel too.
	constexpr float ADD_RECT_INSET = 0.5f;
	const float     half           = thickness / 2.0f;
	const auto      bounds         = Expand(rect, inner + half + ADD_RECT_INSET);
	list.AddRect(TopLeft(bounds), BottomRight(bounds), ToImU32(color), rounding + inner + half,
	             thickness);
}

void DrawGlowOutside(ImDrawList& list, const Rect& rect, float rounding, const Glow& glow,
                     Color color, const ScaleContext& scale, float opacity) {
	const float spread = scale.Px(glow.radius);
	const float peak   = ClampUnit(glow.peak_alpha) * ClampUnit(opacity);
	if (!(spread > 0.0f) || !(peak > 0.0f) || color.a == 0) {
		return;
	}
	const auto  count = static_cast<uint32_t>(PieceCount(spread, RING_WIDTH, 1, MAX_RING_COUNT));
	const float steps = static_cast<float>(count);
	const float    ring  = spread / steps;
	for (uint32_t layer = 0; layer < count; layer++) {
		// A quadratic falloff, bright at the element's edge and soft at the outside.
		const float remaining = 1.0f - static_cast<float>(layer) / steps;
		StrokeOutside(list, rect, rounding, ring * static_cast<float>(layer), ring,
		              color.MulAlpha(peak * remaining * remaining));
	}
}

void DrawShadow(ImDrawList& list, const Rect& rect, float rounding, const Elevation& elevation,
                Color color, const ScaleContext& scale, float opacity) {
	DrawSoftHalo(list, rect, rounding, scale.Px(elevation.blur), scale.Px(elevation.offset_y),
	             ClampUnit(elevation.peak_alpha) * ClampUnit(opacity), color);
}

void DrawGlow(ImDrawList& list, const Rect& rect, float rounding, const Glow& glow, Color color,
              const ScaleContext& scale, float opacity) {
	DrawSoftHalo(list, rect, rounding, scale.Px(glow.radius), 0.0f,
	             ClampUnit(glow.peak_alpha) * ClampUnit(opacity), color);
}

} // namespace Libs::Graphics::SystemUi
