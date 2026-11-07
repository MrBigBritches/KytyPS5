#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_LEAVES_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_LEAVES_H_

#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/primitives.h"

struct ImDrawList;

// The Text and Image nodes' measuring, arranging and painting, which layout.cpp and primitives.cpp
// call. Internal to the element code, so components don't include it.
namespace Libs::Graphics::SystemUi {

// What measuring a node found. The size is what it takes in the space offered, and the minimum
// is what it needs to avoid spilling its content.
struct NodeSize {
	Vec2 size;
	Vec2 minContent;
};

// Measures a text node. It lays the text out in the width on offer, so the node's own layout is
// ready to draw if the width turns out right. It computes from the content and never returns the
// node's `measured`, which may hold a length assigned in an earlier pass.
NodeSize MeasureText(UiFrame& frame, const Node& node, Vec2 max);

// Measures an image node from its explicit size and its texture's shape.
NodeSize MeasureImage(const UiFrame& frame, const Node& node);

// Lays a text node out again when the width Arrange gives it differs from the one it was
// measured at, so ellipsis and wrapping follow the final width within the same frame.
void ArrangeText(UiFrame& frame, const Node& node, const Rect& rect);

// Where a layout rectangle is drawn under `transform`. It is scaled about the layout origin and
// moved, and its corner is rounded to a whole pixel only when the scale is 1.
Rect MapRect(const Rect& rect, const PaintTransform& transform);

// The same for text, which keeps its size and moves with its centre.
Rect MapCentred(const Rect& rect, const PaintTransform& transform);

// Copies `size` bytes at `bytes` into the frame's callable storage, aligned to `alignment`, and
// returns where they sit. Custom nodes and the interaction callbacks share the storage, which Begin
// clears. The callers have checked the callable's type, and its alignment is at most the default
// new alignment, which the storage starts at.
// While Render's afterInteract hook or a callback runs it stores nothing and returns NO_CALLABLE,
// since growing the storage would move the bytes they run from. Debug builds assert instead.
uint32_t StoreCallable(UiFrame& frame, const void* bytes, size_t size, size_t alignment);

// What StoreCallable returns when it refuses.
constexpr uint32_t NO_CALLABLE = UINT32_MAX;

// The callable stored at `offset` by StoreCallable.
const std::byte* StoredBytes(const UiFrame& frame, uint32_t offset);

// Sets what the state styles of `node` respond to, unless SetInput gave it a value by hand. Boxes
// have states, and a custom node keeps the input for its callable to read in CustomPaint::input.
// Any other node is left alone. Render uses it to pass the input of an
// interactive node down its subtree.
void InheritInput(UiFrame& frame, Node& node, const ItemInput& input);

// Measures a custom node, whose size comes from its ItemLayout alone.
NodeSize MeasureCustom(const UiFrame& frame, const Node& node);

// Calls a custom node's callable with the rectangle it is painted in.
void PaintCustom(const UiFrame& frame, const Node& node, ImDrawList& list, const PaintState& state,
                 const Rect& painted);

// Draws a text node centred vertically in its arranged rectangle.
void PaintText(const UiFrame& frame, const Node& node, ImDrawList& list, const PaintState& state);

// Draws an image node in its arranged rectangle.
void PaintImage(const UiFrame& frame, const Node& node, ImDrawList& list, const PaintState& state);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_LEAVES_H_
