#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_SCALEDDRAW_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_SCALEDDRAW_H_

#include "graphics/presentation/system/core/geometry.h"

struct ImDrawList;

namespace Libs::Graphics::SystemUi {

// Draws what is added while it lives as if at another size. Everything goes into `list` as usual,
// in full-size coordinates, and on leaving, each new vertex and clip rectangle is shrunk toward
// `origin` by `fit`, which must be positive, so it lands where it should. Clips are then held to
// `visible`, which is in the final coordinates.
//
// A full-size clip is pushed for its life, as text drawn outside the current clip is culled as it
// is added, and would vanish while boxes beside it stay. It is pushed without intersecting, so that
// what lies past the window at full size still reaches the scaling. Whatever the drawing does to
// the window's content size is put back, so it can't make the window scroll.
//
// Controls registered with ImGui inside it keep their full-size rectangles, which don't match what
// is seen, so only draw things here that take no input.
class ScaledDrawScope {
public:
	/// Starts scaling what is added to `list`, about `origin` by `fit`, held to `visible`.
	ScaledDrawScope(ImDrawList& list, Vec2 origin, float fit, const Rect& visible);
	/// Scales all that was added and restores the clip and the window's content size.
	~ScaledDrawScope();
	ScaledDrawScope(const ScaledDrawScope&)            = delete;
	ScaledDrawScope& operator=(const ScaledDrawScope&) = delete;

private:
	ImDrawList& m_list;
	Vec2        m_origin;
	float       m_fit;
	Rect        m_visible;
	int         m_first_vertex  = 0;
	int         m_first_command = 0;
	Vec2        m_content_max;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_SCALEDDRAW_H_
