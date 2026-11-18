#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_POSED_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_POSED_H_

#include "graphics/presentation/system/animation/timeline.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/scale.h"

namespace Libs::Graphics::SystemUi {

/// Where an element laid out at `rect` is drawn in `pose`. The pose's offset, in reference
/// pixels, moves it, and its scale grows or shrinks it about its centre. The corner snaps to a
/// whole pixel so a resting element is sharp. Its opacity is pose.alpha.
[[nodiscard]] inline Rect Posed(const Rect& rect, const Pose& pose, const ScaleContext& scale) {
	const Vec2  centre = rect.Center();
	const float width  = rect.width * pose.scale;
	const float height = rect.height * pose.scale;
	return {SnapPixel(centre.x - width / 2.0f + scale.Px(pose.offset.x)),
	        SnapPixel(centre.y - height / 2.0f + scale.Px(pose.offset.y)), width, height};
}

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_POSED_H_
