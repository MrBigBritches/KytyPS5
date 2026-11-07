#include "graphics/presentation/system/effects/gradient.h"

#include "graphics/presentation/system/imgui/convert.h"

namespace Libs::Graphics::SystemUi {

void DrawVerticalGradient(ImDrawList& list, const Rect& rect, float rounding, Color top,
                          Color bottom, float opacity) {
	constexpr float CHANNEL_MAX = 255.0f;
	constexpr ImU32 CHANNEL     = 0xffu;
	const int       first       = list.VtxBuffer.Size;
	list.AddRectFilled(TopLeft(rect), BottomRight(rect), IM_COL32_WHITE, rounding);
	for (int index = first; index < list.VtxBuffer.Size; index++) {
		auto&       vertex   = list.VtxBuffer[index];
		const float position = rect.height > 0.0f ? (vertex.pos.y - rect.y) / rect.height : 0.0f;
		// The fill is opaque white except for the transparent anti-aliasing fringe, so the vertex's
		// own alpha says how much of the gradient's alpha it keeps.
		const float coverage =
		    static_cast<float>((vertex.col >> IM_COL32_A_SHIFT) & CHANNEL) / CHANNEL_MAX;
		vertex.col = ToImU32(Lerp(top, bottom, position).MulAlpha(coverage * opacity));
	}
}

} // namespace Libs::Graphics::SystemUi
