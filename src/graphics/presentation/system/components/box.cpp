#include "graphics/presentation/system/components/box.h"

#include "graphics/presentation/system/effects/halo.h"
#include "graphics/presentation/system/imgui/convert.h"

#include <algorithm>

namespace Libs::Graphics::SystemUi {

void DrawBox(ImDrawList& list, const UiContext& context, const Rect& rect, const BoxStyle& style,
             float opacity) {
	const auto& scale  = context.scale;
	const float radius = std::min(scale.Px(style.radius), std::min(rect.width, rect.height) / 2.0f);
	DrawShadow(list, rect, radius, style.elevation, context.theme.colors.effect.shadow, scale,
	           opacity);
	const Color fill = style.fill.MulAlpha(opacity);
	if (fill.a > 0) {
		list.AddRectFilled(TopLeft(rect), BottomRight(rect), ToImU32(fill), radius);
	}
	const Color border = style.border.MulAlpha(opacity);
	if (style.border_width > 0.0f && border.a > 0) {
		// A stroke whose inner edge lies inside the box by its own width ends at the box's edge.
		const float width = scale.Border(style.border_width);
		StrokeOutside(list, rect, radius, -width, width, border);
	}
}

} // namespace Libs::Graphics::SystemUi
