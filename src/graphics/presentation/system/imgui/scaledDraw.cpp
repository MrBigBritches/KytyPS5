#include "graphics/presentation/system/imgui/scaledDraw.h"

#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>

namespace Libs::Graphics::SystemUi {

namespace {

// The vertices and indices of the empty rectangle that holds a draw command open.
constexpr int HOLDING_INDICES  = 6;
constexpr int HOLDING_VERTICES = 4;

// `value` moved toward `origin` by `fit`.
float Shrink(float value, float origin, float fit) {
	return origin + fit * (value - origin);
}

// `value` moved away from `origin` by what `fit` would shrink it, so shrinking it gets it back.
float Grow(float value, float origin, float fit) {
	return origin + (value - origin) / fit;
}

} // namespace

ScaledDrawScope::ScaledDrawScope(ImDrawList& list, Vec2 origin, float fit, const Rect& visible)
    : m_list(list), m_origin(origin), m_fit(fit), m_visible(visible),
      m_first_vertex(list.VtxBuffer.Size) {
	const ImGuiWindow* window = ImGui::GetCurrentWindow();
	m_content_max             = {window->DC.CursorMaxPos.x, window->DC.CursorMaxPos.y};
	list.PushClipRect(
	    {Grow(visible.x, origin.x, fit), Grow(visible.y, origin.y, fit)},
	    {Grow(visible.Right(), origin.x, fit), Grow(visible.Bottom(), origin.y, fit)});
	// A command of its own, or the clips scaled below would be shared with what came before. An
	// empty command merges back into the one before it, so this one is given something to hold.
	list.AddDrawCmd();
	list.PrimReserve(HOLDING_INDICES, HOLDING_VERTICES);
	list.PrimRect({origin.x, origin.y}, {origin.x, origin.y}, 0);
	m_first_command = list.CmdBuffer.Size - 1;
}

ScaledDrawScope::~ScaledDrawScope() {
	for (int vertex = m_first_vertex; vertex < m_list.VtxBuffer.Size; vertex++) {
		ImVec2& pos = m_list.VtxBuffer[vertex].pos;
		pos         = {Shrink(pos.x, m_origin.x, m_fit), Shrink(pos.y, m_origin.y, m_fit)};
	}
	for (int index = m_first_command; index < m_list.CmdBuffer.Size; index++) {
		ImVec4& clip = m_list.CmdBuffer[index].ClipRect;
		clip.x = std::clamp(Shrink(clip.x, m_origin.x, m_fit), m_visible.x, m_visible.Right());
		clip.y = std::clamp(Shrink(clip.y, m_origin.y, m_fit), m_visible.y, m_visible.Bottom());
		clip.z = std::clamp(Shrink(clip.z, m_origin.x, m_fit), m_visible.x, m_visible.Right());
		clip.w = std::clamp(Shrink(clip.w, m_origin.y, m_fit), m_visible.y, m_visible.Bottom());
	}
	// After the mapping, so the command opened for what follows has the clip from before the scope.
	m_list.PopClipRect();
	ImGui::GetCurrentWindow()->DC.CursorMaxPos = {m_content_max.x, m_content_max.y};
}

} // namespace Libs::Graphics::SystemUi
