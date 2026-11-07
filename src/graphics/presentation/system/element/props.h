#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_PROPS_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_PROPS_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/units.h"
#include "graphics/presentation/system/layout/flex.h"

#include <cstdint>
#include <optional>

namespace Libs::Graphics::SystemUi {

// What a node asks of the container it sits in. Every kind of node takes one, so Box, Text and
// Image share the same fields. Lengths are in design units. Min and max limit the length along
// the container's main axis, whichever way that runs.
struct ItemLayout {
	// An explicit size replaces the node's own measurement on that axis, border and padding
	// included. A node with an explicit main-axis size, width in a row or height in a column,
	// never shrinks when its siblings overflow the row. Nodes without one shrink in proportion to
	// their size, down to their min_main.
	std::optional<Dp> width;
	std::optional<Dp> height;
	// The node's share of the container's free main-axis space, relative to its siblings.
	float             grow = 0.0f;
	std::optional<Dp> min_main;
	std::optional<Dp> max_main;
	// Overrides the container's cross-axis alignment for this node.
	std::optional<Align> align_self;
	// Gives up all its length along the main axis when the container is too short to leave the
	// other items their minimums, as a value label does when it would leave the track too short.
	// The container's length must be known, so it never collapses while measured with no limit.
	// An interactive node that has focus when it collapses loses it, since a collapsed node is not
	// registered.
	bool collapses = false;
};

// How a box lays out its children, and what the box itself asks of its own container.
struct Layout {
	FlexDirection direction = FlexDirection::Row;
	Justify       justify   = Justify::Start;
	Align         align     = Align::Start;
	// Space between neighbouring children on the main axis.
	Dp       gap;
	DpInsets padding;
	// The box's own request to its container.
	ItemLayout item;
	// Lays the children over one another instead of along an axis, each filling the box inside its
	// padding whatever length it asks for, and in order, so a later child is drawn over an earlier
	// one. The box measures as its largest child. Direction, justify, align and gap are ignored. It
	// suits parts of one control that share a rectangle, such as a toggle's track, shade, knob
	// and ring. A stacked box is not a list and does not scroll, so AsList and scrolling are not
	// supported on one, and it is not laid out by the flex solver. Last, so positional initialisers
	// that stop at the item still compile.
	bool stacked = false;
};

// How a box differs while it is in a state. Fields left unset leave the box as it is. A state set
// to a value moves the box that far toward it as the state comes on, and the colours blend by the
// state's weight. The overlay is drawn over the box at the state's weight, the scale shrinks the
// box about its centre, the focus ring and the selection show at the state's weight, and the
// elevation switches at half weight. Lengths are design units.
struct StateStyle {
	std::optional<Color>     fill;
	std::optional<Color>     border;
	std::optional<Dp>        border_width;
	std::optional<Dp>        radius;
	std::optional<Elevation> elevation;
	Color                    overlay;
	std::optional<float>     opacity;
	std::optional<float>     scale;
	bool                     focus_ring = false;
	bool                     selection  = false;
};

// The states a box responds to, applied in the order selected, hover, focus, pressed, disabled.
struct StateStyles {
	StateStyle hover;
	StateStyle focus;
	StateStyle pressed;
	StateStyle disabled;
	StateStyle selected;
};

// Whether a box's children may draw outside it. Clip keeps them inside the box's outer rectangle,
// less the corners its radius rounds off, so children may draw over the border band. That matches
// the toast today and not CSS, which clips to the padding box. The box's own shadow and effects
// are not clipped by it. Text keeps its size when a pose scales the box down, so text in a
// scaled-down clipping box can be cut at its edges, as it can in CSS.
enum class Overflow : uint8_t {
	Visible,
	Clip,
};

// How a box looks. A transparent border or a zero border width draws no border, and elevation
// level 0 casts no shadow. Lengths are design units, converted when the box is drawn.
struct Style {
	Color     fill;
	Color     border;
	Dp        border_width;
	Dp        radius;
	Elevation elevation;
	// Last, so positional initialisers that stop at the elevation still compile.
	Overflow overflow = Overflow::Visible;
	// How the box changes with the state of the control it stands for.
	StateStyles states;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_PROPS_H_
