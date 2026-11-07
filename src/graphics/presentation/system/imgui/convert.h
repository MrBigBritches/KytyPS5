#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_CONVERT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_CONVERT_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/geometry.h"
#include "imgui.h"

// The one place the system UI's plain types meet ImGui's, so the rest of the framework stays free
// of ImGui headers and its packed colour layout.
namespace Libs::Graphics::SystemUi {

/// A colour packed the way ImGui expects. IM_COL32 follows the build's packing, so this stays
/// right even if IMGUI_USE_BGRA_PACKED_COLOR is ever defined.
[[nodiscard]] inline ImU32 ToImU32(Color color) {
	return IM_COL32(color.r, color.g, color.b, color.a);
}

/// A point or size as an ImGui vector.
[[nodiscard]] inline ImVec2 ToImVec2(Vec2 value) {
	return {value.x, value.y};
}

/// A rectangle's top-left corner, as ImGui's draw calls take it.
[[nodiscard]] inline ImVec2 TopLeft(const Rect& rect) {
	return {rect.x, rect.y};
}

/// A rectangle's bottom-right corner, as ImGui's draw calls take it.
[[nodiscard]] inline ImVec2 BottomRight(const Rect& rect) {
	return {rect.Right(), rect.Bottom()};
}

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_CONVERT_H_
