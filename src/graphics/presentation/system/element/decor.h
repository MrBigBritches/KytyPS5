#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_DECOR_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_DECOR_H_

#include "graphics/presentation/system/animation/timeline.h"
#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include <optional>

namespace Libs::Graphics::SystemUi {

/// The point a pose scales about, as a fraction of the node's arranged rectangle. Zero is the left
/// or top edge and one the right or bottom edge.
struct Origin {
	float x = 0.5f;
	float y = 0.5f;
};

/// The middle of the node, which is where a pressed button shrinks toward.
inline constexpr Origin CENTER {0.5f, 0.5f};
/// The top-right corner, where a toast grows from.
inline constexpr Origin TOP_RIGHT {1.0f, 0.0f};

/// How a node is posed, as the element API means it. Unlike the animation Pose, whose default is
/// the invisible waiting state, a default ElementPose changes nothing. The offset is in reference
/// pixels.
struct ElementPose {
	float alpha = 1.0f;
	Vec2  offset;
	float scale = 1.0f;
};

/// A keyed animation for an element. Its start and dismissal are kept in the interaction store
/// under the element's path id, so key the element when its row can be reordered. Setting
/// `dismissed` starts the exit from wherever the element is, and holding it plays the exit once.
/// Clearing it again restarts the entrance.
struct Animate {
	Timeline timeline;
	Origin   origin    = CENTER;
	bool     dismissed = false;
};

/// The element pose a timeline pose asks for, so a pose from an animation can be applied as it is.
[[nodiscard]] inline ElementPose ToElementPose(const Pose& pose) {
	return {pose.alpha, pose.offset, pose.scale};
}

/// A glow behind an element, of the theme's glow token `glow` in `color`.
struct GlowEffect {
	Glow  glow;
	Color color;
};

/// What the player is doing to an element, which its state styles respond to. The caller supplies
/// it, from Interact or from a specimen's fixed pose. The visuals are how far each highlight has
/// come, and disabled and selected are on or off.
struct ItemInput {
	ItemVisuals visuals;
	bool        disabled = false;
	bool        selected = false;
	// Set for an input Render passed down from an interactive ancestor. The ancestor's own disabled
	// fade already multiplies down the tree, so a box that inherits the input must not fade again,
	// though a disabled state style it has of its own still applies.
	bool inherited = false;
};

/// The effects an element draws about itself, on top of its box. Each uses the element's rectangle
/// and corner radius, and fades with the opacity the element inherits. An element draws them in
/// this order. The glow comes before the box, the selection and the focus ring after it, the
/// children next, and the shine over the children. A clip on the element cuts its children but
/// not these, while a clip on an ancestor cuts them unless `unclipped` is set.
struct Decor {
	std::optional<GlowEffect> glow;
	/// How far a shine has swept across, from 0 to 1. Outside that range it draws nothing.
	std::optional<float> shine;
	/// The strength of the focus ring, from 0 to 1.
	std::optional<float> focus_ring;
	bool                 selection = false;
	/// Draws the effects outside the nearest clipping ancestor's clip, as a glow that spills past a
	/// card does. Clips further up the tree still apply.
	bool unclipped = false;
};

// What a node carries beyond the generic node when it is posed or decorated, kept in the frame's
// side vector so plain nodes pay nothing for it. The pose, the effects and the input are set by
// separate calls and none of them resets another, since Interact fills the input after layout
// and a later Decorate must not return it to resting.
struct DecorData {
	ElementPose pose;
	Origin      origin;
	Decor       effects;
	// What the element's state styles respond to.
	ItemInput input;
	// Whether SetInput gave the input, which Render's inheritance from an interactive ancestor
	// must then leave alone.
	bool input_explicit = false;
	// The keyed animation, and the pose Render found for it this frame, which sits inside the
	// explicit pose. Without Render there is no pose, and the animation changes nothing.
	std::optional<Animate> animate;
	std::optional<Pose>    animated;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_DECOR_H_
