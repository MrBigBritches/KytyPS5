#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_NODE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_NODE_H_

#include "graphics/presentation/system/core/geometry.h"

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace Libs::Graphics::SystemUi {

// What a node is. Each kind gets its data through the node's payload index.
enum class NodeKind : uint8_t {
	None,
	Box,
	Text,
	Image,
	Custom,
};

// One node of the frame's element tree. It lives in the frame's arena, which is reset by
// clearing vectors and never runs destructors, so a node must stay trivially destructible. Only
// the frame and the primitives include this header, components see Element handles alone.
struct Node {
	// The parent value of a node that has none yet.
	static constexpr uint32_t NO_PARENT = UINT32_MAX;
	// The decor value of a node with nothing beyond the generic node.
	static constexpr uint32_t NO_DECOR = UINT32_MAX;
	// The interact value of a node with no interaction record.
	static constexpr uint32_t NO_INTERACT = UINT32_MAX;

	NodeKind kind = NodeKind::None;

	// The index of the node this one is a child of. A node has at most one parent, since the
	// element API builds a tree.
	uint32_t parent = NO_PARENT;

	// The children as a run in the frame's child storage.
	uint32_t firstChild = 0;
	uint32_t childCount = 0;

	// The size the node wants with no limit on its container's main axis, set by its parent. It
	// stays what it is however much space the parent hands out.
	Vec2 natural;
	// The size with no limit on either axis, which the memo keeps for the whole pass. `natural`
	// is this size, or the size at the room across when the node is wider than that room.
	Vec2 unbounded;
	// Leaf contract: a leaf's Measure computes its natural size from its content, and never
	// returns `measured`, which may hold a length assigned in an earlier pass.
	// The size the node got once its parent assigned it a main-axis length, which is what Measure
	// last found. It differs from the natural size only when the node grew or shrank.
	Vec2 measured;
	Rect arranged;
	// The smallest main-axis size the node can take without its content spilling out, what CSS
	// calls min-content. A flex item shrinks no further than this unless it sets its own minimum.
	// For text it depends on overflow, the longest word when wrapped, the ellipsis when truncated
	// and nothing when clipped, and its height is the laid-out height.
	Vec2 minContent;
	// The space Measure was last given for this node, so a parent can tell whether the subtree
	// still holds the layout for the space it is about to assign. Measuring is the only thing that
	// writes the subtree, and any later measure with a different limit overwrites it.
	Vec2 lastLimit;
	// Set by Measure, so Arrange can refuse a tree that was never measured.
	bool isMeasured = false;
	// Set by the parent's Arrange when its flex solver collapsed this node because the room was
	// short. A collapsed node and its subtree are absent: they are not arranged again, painted or
	// walked for interaction. It is not decided by the node having no size, since a spacer or a bar
	// of no width is still there. Skipping the walk skips PlayAnimation, so a keyed animation on a
	// collapsed node, or below it, is not played that frame, and the store forgets an animation not
	// played for a frame. A node that collapses and then comes back therefore starts its entrance
	// again, as any node missing for a frame does. Arrange given the node itself clears the flag.
	bool collapsed = false;
	// Whether `natural` holds a measurement made in measure pass `naturalGeneration`, so a parent
	// that measures the node at its natural size again in that pass can reuse it instead of
	// walking its subtree. A later pass may offer different space, so it measures afresh.
	bool     hasNatural        = false;
	uint32_t naturalGeneration = 0;

	// An index into the data of this node's kind, which the primitives add.
	uint32_t payload = 0;

	// An index into the frame's item layouts, what the node asks of its container.
	uint32_t item = 0;

	// The key the caller gave the node, which names it among its siblings in place of its position.
	// A key is a string, interned in the frame, or a number, and the two never match each other.
	std::string_view key;
	uint32_t         keyNumber = 0;
	bool             hasKey    = false;
	bool             keyIsText = false;
	// The ImGui id of the node, set by AssignIds from its key or position and its parent's id. Zero
	// until then.
	uint32_t id = 0;

	// An index into the frame's decor records, or NO_DECOR for a plain node.
	uint32_t decor = NO_DECOR;

	// An index into the frame's interaction records, or NO_INTERACT for a node that takes no input.
	uint32_t interact = NO_INTERACT;
};

static_assert(std::is_trivially_destructible_v<Node>,
              "the arena is reset without running destructors");

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_NODE_H_
