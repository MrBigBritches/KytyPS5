#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_STATES_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_STATES_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/element/decor.h"
#include "graphics/presentation/system/element/props.h"

#include <array>
#include <cstddef>

// The merging of a box's state styles, which primitives.cpp calls when it paints. Internal to the
// element code, so components don't include it.
namespace Libs::Graphics::SystemUi {

// One overlay a state draws over the box, faded by how strong the state is.
struct StateOverlay {
	Color color;
	float weight = 0.0f;
};

// A box's look once its states are merged into its base style. Lengths are in reference pixels, as
// in Style, and the radius is not yet clamped to the painted rectangle.
struct ResolvedStyle {
	Color     fill;
	Color     border;
	float     border_width = 0.0f;
	float     radius       = 0.0f;
	Elevation elevation;
	// The factor the box and its subtree fade by, and the scale it is drawn at about its centre.
	float opacity = 1.0f;
	float scale   = 1.0f;
	// How strong the focus ring and the selection fill are, from 0 to 1. Zero draws neither.
	float focus_ring = 0.0f;
	float selection  = 0.0f;
	// The overlays to draw in order, and how many are used.
	std::array<StateOverlay, 5> overlays;
	size_t                      overlay_count = 0;
};

/// The look of a box with `style` for `input`. The states apply in the order selected, hover,
/// focus, pressed, disabled, and each moves the value it sets toward its own by the state's
/// weight, so a state at 0 changes nothing. Hover and focus that set the same value share the
/// larger of their weights, as an accent button blends its hover fill, instead of approaching it
/// twice. A disabled input fades by the theme's disabled opacity unless the disabled state sets
/// one. The elevation switches to a state's at a weight of half or more. A non-finite or
/// non-positive scale and a non-finite opacity in a state are ignored.
[[nodiscard]] ResolvedStyle ResolveStyle(const Theme& theme, const Style& style,
                                         const ItemInput& input);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_STATES_H_
