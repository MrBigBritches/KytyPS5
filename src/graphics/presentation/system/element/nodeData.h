#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_NODE_DATA_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_NODE_DATA_H_

#include "graphics/presentation/system/components/image.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/element.h"
#include "graphics/presentation/system/element/props.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

struct ImTextureData;

namespace Libs::Graphics::SystemUi {

struct CustomPaint;
struct NodeInput;
class UiFrame;

// The function that calls a custom node's callable, which sits at `callable` in the frame's
// storage. Each callable type gets its own, made when the node is built.
using CustomInvoke = void (*)(const std::byte* callable, const CustomPaint& paint);

// The function that calls an activation callback stored at `callable`, with the activated element.
using ActivateInvoke = void (*)(const std::byte* callable, Element element);

// The function that calls an input callback stored at `callable`, with what the node found.
using InputInvoke = void (*)(const std::byte* callable, const NodeInput& input);

// The function that calls a callback stored at `callable` that runs once a tree has interacted,
// with the frame and the element it was given.
using AfterInteractInvoke = void (*)(const std::byte* callable, UiFrame& frame, Element element);

// A callback kept in the frame's callable storage, which Custom nodes share. Not set when `invoke`
// is null.
template <class Invoke>
struct StoredCallback {
	Invoke   invoke = nullptr;
	uint32_t offset = 0;
};

// What a box node keeps beyond the generic node.
struct BoxData {
	Layout layout;
	Style  style;
};

// A text layout kept from frame to frame, and what it was made from. A text is laid out at several
// widths in a frame, so one of these holds one of them.
struct CachedTextLayout {
	bool          used = false;
	TextLayoutKey key;
	// The bytes and not a view, as the frame's interned text can change under the same address.
	std::string text;
	TextLayout  layout;
	// The least width this text can take at this key, which measuring it again would only repeat.
	float min_width = 0.0f;
};

// The layouts kept for the text node at one position among the frame's texts. A node has no identity
// before Render gives it an id, so nothing but the key and bytes says a layout is right, and a tree
// that changes only misses until it settles. Text wrapped into a column sees up to four widths in a
// frame, unbounded, the room across, its final length and the width it is arranged at, and cycling
// more widths than there are entries would make every frame miss.
inline constexpr size_t TEXT_LAYOUTS_KEPT = 4;
struct TextSlot {
	std::array<CachedTextLayout, TEXT_LAYOUTS_KEPT> entries;
	// The entry the next new layout replaces.
	uint8_t next = 0;
};

// A layout kept for text laid out as it is painted, which LaidOutText finds by what it shows, as a
// list's rows come and go as it scrolls and have no place of their own.
struct PaintedText {
	TextLayoutKey key;
	std::string   text;
	TextLayout    layout;
	// The BeginCount it was last asked for at, so one no longer shown can be forgotten.
	uint32_t last_used = 0;
};

// What a text node keeps. The props view text the frame interned, and the layout it last asked for is
// an entry of its slot, by index as a later text can move the slots.
struct TextData {
	TextProps props;
	// Whether Measure has laid it out, and so whether `entry` means anything.
	bool    laidOut = false;
	uint8_t entry   = 0;
	// The width the layout was made for, so Arrange can tell when the width it gives differs.
	float laidOutWidth = 0.0f;
};

// What an image node keeps.
struct ImageData {
	ImTextureData* texture = nullptr;
	Vec2           textureSize;
	ImageFit       fit = ImageFit::Fill;
	Dp             radius;
};

// What a custom node keeps. The callable's bytes live in the frame's custom storage at `offset`.
struct CustomData {
	CustomInvoke invoke = nullptr;
	uint32_t     offset = 0;
};

// How an interactive node claims the D-pad's left and right while it has focus. Once takes one
// step for each press, Repeat repeats a held direction at the pace of focus moves, and None leaves
// them to navigation.
enum class DpadClaim : uint8_t {
	None,
	Once,
	Repeat,
};

// The key groups a node owns while Render walks it, claimed under the node's path id every frame.
// Owning the shoulders keeps the gallery's page paging and any lower bar from seeing L1 and R1, and
// owning back keeps the screen's back handling from seeing it, so a tab bar or a dialog built from
// elements gets the keys as the components do. See ClaimKeys for how the priorities settle.
struct KeyOwner {
	bool          shoulders = false;
	bool          back      = false;
	ClaimPriority priority  = ClaimPriority::Control;
};

// The presses an owning node read this frame as the owner. Each is false for a group it didn't ask
// for, or didn't win.
struct KeyPresses {
	HorizontalPresses shoulders;
	bool              back = false;
};

// What an interactive node asks of the player's input. Lengths and colours stay with the box.
struct InteractProps {
	bool      disabled      = false;
	bool      selected      = false;
	bool      default_focus = false;
	// A disabled node does not claim the D-pad, and gives back any direction it held, so navigation
	// has it.
	DpadClaim dpad          = DpadClaim::None;
	// Whether the node reads the L2 and R2 triggers while it has focus.
	bool triggers = false;
	// The key groups the node owns, which replace any given by OwnKeys.
	KeyOwner keys;
};

// What Render found out about an interactive node this frame.
struct NodeInput {
	Interaction       interaction;
	HorizontalPresses dpad;
	HorizontalPresses triggers;
	// What the node read of the key groups it owns.
	KeyPresses keys;
	// The rectangle the node was hit-tested in, its arranged one and not the posed one.
	Rect   hit;
	ItemId id = 0;
};

// A list's settings and, once Render has walked it, how far it was scrolled when drawn and how tall
// its rows were found to be. Paint reads the scroll as layout, so the children of a list sit that
// far up, and both for the scroll bar.
struct ListData {
	bool  is_list = false;
	float scroll  = 0.0f;
	float content = 0.0f;
	// Whether the D-pad's up and down scroll it, for a list with nothing in it to focus.
	bool pad_scrolls = false;
};

// A list's scroll bar, in reference pixels. It has a gutter of its own at the list's right, with a
// gap past the room the rows leave for the focus ring. It can be grabbed across the gutter, wider
// than it's drawn, without taking a click from the end of a row.
inline constexpr float LIST_THUMB_WIDTH     = 4.0f;
inline constexpr float LIST_THUMB_GAP       = 4.0f;
inline constexpr float LIST_THUMB_MIN       = 32.0f;
inline constexpr float LIST_THUMB_HIT_WIDTH = 8.0f;
// How far a notch of the mouse wheel scrolls a list, a row of one line.
inline constexpr float LIST_WHEEL_STEP = 56.0f;

/// Where a list's thumb lies at the right of `area`, `content` tall and scrolled by `scroll`, with
/// lengths in pixels, or nothing when the rows all fit. Its length is the share of the rows in
/// view, but never under `min_length`, and it travels the area as the scroll goes from none to its
/// end.
[[nodiscard]] inline std::optional<Rect> ListThumb(const Rect& area, float content, float scroll,
                                                   float width, float min_length) {
	const float hidden = content - area.height;
	if (!(hidden > 0.0f) || !(area.height > 0.0f)) {
		return std::nullopt;
	}
	const float length =
	    std::min(area.height, std::max(min_length, area.height * area.height / content));
	const float share = std::clamp(scroll / hidden, 0.0f, 1.0f);
	return Rect {area.Right() - width, area.y + (area.height - length) * share, width, length};
}

// What a node carries to take part in Render's interaction walk, in the frame's side vector.
struct InteractData {
	InteractProps props;
	bool          interactive = false;
	ListData      list;
	// Set by Specimen, which gives the node and its subtree a fixed pose and no input.
	bool                     posed = false;
	ItemVisuals              pose;
	std::optional<NodeInput> input;
	// The key groups the node owns, and what it read of them this frame. A node can own keys
	// without being interactive, so a container such as a tab bar or a dialog can hold interactive
	// children.
	KeyOwner                  keys;
	std::optional<KeyPresses> key_presses;
	// Called in Render's walk right after the node interacts, before anything is painted.
	StoredCallback<ActivateInvoke> on_activate;
	StoredCallback<InputInvoke>    on_input;
	// Called once the whole tree has interacted, before anything is painted.
	StoredCallback<AfterInteractInvoke> after_interact;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_NODE_DATA_H_
