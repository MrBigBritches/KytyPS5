#include "graphics/presentation/system/effects/highlight.h"

#include "graphics/presentation/system/effects/halo.h"
#include "graphics/presentation/system/imgui/convert.h"

#include <algorithm>

namespace Libs::Graphics::SystemUi {

void DrawFocusRing(ImDrawList& list, const Rect& rect, float rounding, const Theme& theme,
                   const ScaleContext& scale, float opacity) {
	const auto& focus   = theme.colors.focus;
	const float ring    = scale.Border(theme.border.thick);
	const float keyline = scale.Border(theme.border.thin);
	DrawGlowOutside(list, Expand(rect, ring + keyline), rounding + ring + keyline, theme.glow.focus,
	                focus.glow, scale, opacity);
	StrokeOutside(list, rect, rounding, ring, keyline, focus.keyline.MulAlpha(opacity));
	StrokeOutside(list, rect, rounding, 0.0f, ring, focus.ring.MulAlpha(opacity));
}

void DrawSelection(ImDrawList& list, const Rect& rect, float rounding, const Theme& theme,
                   const ScaleContext& scale, float opacity) {
	const auto& accent = theme.colors.accent;
	list.AddRectFilled(TopLeft(rect), BottomRight(rect), ToImU32(accent.tint.MulAlpha(opacity)),
	                   rounding);
	// The bar stops short of the rounded corners, so it never pokes past the element's outline.
	// ImGui caps its rounding a pixel short of half its width, which still reads as rounded ends.
	const float bar_width = std::min(scale.Border(theme.border.bar), rect.width);
	const float bar_inset = std::clamp(rounding, 0.0f, rect.height / 2.0f);
	const Rect  bar {rect.x, rect.y + bar_inset, bar_width, rect.height - 2.0f * bar_inset};
	list.AddRectFilled(TopLeft(bar), BottomRight(bar), ToImU32(accent.line.MulAlpha(opacity)),
	                   bar_width / 2.0f);
}

} // namespace Libs::Graphics::SystemUi
