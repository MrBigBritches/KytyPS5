#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_FRAME_INTERNALS_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_FRAME_INTERNALS_H_

#include "graphics/presentation/system/element/frame.h"

#include <cassert>

namespace Libs::Graphics::SystemUi {

/// Makes ReportRefusal log every reason again, for tests.
void ForgetReportedRefusals();

// The frame's node storage, for the primitives and their tests. Components don't include this
// header, so they can only pass Element handles around.
struct FrameInternals {
	/// Adds a node of `kind` with copies of the `children` handles and returns its handle. It
	/// returns a none handle and changes nothing when called before Begin, or when a child is none,
	/// stale, out of range, listed twice or already has a parent. That also refuses a span taken
	/// from this frame's own child storage, since every handle in it has a parent, so the copy can
	/// never read storage it is growing. `item` is what the node asks of its container. When it
	/// refuses, `reason` is set if given, and left alone otherwise. An out parameter keeps the
	/// return a plain Element, so builders nest in a call and only callers that care look at why.
	static Element AddNode(UiFrame& frame, NodeKind kind, std::span<const Element> children = {},
	                       const ItemLayout& item = {}, ElementError* reason = nullptr) {
		return frame.AddNode(kind, children, item, reason);
	}

	/// The node `element` refers to, or null when it is none, stale or out of range. The pointer is
	/// invalidated by the next AddNode, which may move the node storage.
	static Node*       Find(UiFrame& frame, Element element) { return frame.Find(element); }
	static const Node* Find(const UiFrame& frame, Element element) { return frame.Find(element); }

	/// What the node asks of its container.
	static const ItemLayout& Item(const UiFrame& frame, const Node& node) {
		return frame.items_[node.item];
	}

	/// The data of the box nodes, indexed by a box node's payload. Boxes are added here.
	static std::vector<BoxData>&       Boxes(UiFrame& frame) { return frame.boxes_; }
	static const std::vector<BoxData>& Boxes(const UiFrame& frame) { return frame.boxes_; }

	/// The data of the text and image nodes, indexed by their payload.
	static std::vector<TextData>&        Texts(UiFrame& frame) { return frame.texts_; }
	static const std::vector<TextData>&  Texts(const UiFrame& frame) { return frame.texts_; }
	static std::vector<ImageData>&       Images(UiFrame& frame) { return frame.images_; }
	static const std::vector<ImageData>& Images(const UiFrame& frame) { return frame.images_; }

	/// The decor records of the posed nodes, indexed by a node's decor.
	static std::vector<DecorData>&       Decors(UiFrame& frame) { return frame.decor_; }
	static const std::vector<DecorData>& Decors(const UiFrame& frame) { return frame.decor_; }

	/// The custom nodes' data, indexed by their payload, and the bytes of their callables.
	static std::vector<CustomData>&       Customs(UiFrame& frame) { return frame.customs_; }
	static const std::vector<CustomData>& Customs(const UiFrame& frame) { return frame.customs_; }
	static std::vector<std::byte>&        CustomBytes(UiFrame& frame) { return frame.customBytes_; }
	static const std::vector<std::byte>&  CustomBytes(const UiFrame& frame) {
        return frame.customBytes_;
	}

	/// The interaction records of the nodes that take input or show a fixed pose.
	static std::vector<InteractData>&       Interacts(UiFrame& frame) { return frame.interacts_; }
	static const std::vector<InteractData>& Interacts(const UiFrame& frame) {
		return frame.interacts_;
	}

	/// Scratch for AssignIds, and the ids of the roots it has handed out this frame.
	static std::vector<uint32_t>&   IdScratch(UiFrame& frame) { return frame.idScratch_; }
	static std::vector<RootRecord>& RootIds(UiFrame& frame) { return frame.rootIds_; }
	/// Marks the frame as running a callback, during which nodes and their records can't be added.
	static void SetInCallback(UiFrame& frame, bool inside) { frame.inCallback_ = inside; }
	/// Marks the frame as running Render's afterInteract hook, during which nodes can't be added
	/// but their styling can change.
	static void SetInHook(UiFrame& frame, bool inside) { frame.inHook_ = inside; }
	/// Whether Render's afterInteract hook is running.
	static bool InHook(const UiFrame& frame) { return frame.inHook_; }
	/// Whether one of Render's callbacks is running.
	static bool InCallback(const UiFrame& frame) { return frame.inCallback_; }
	/// Why nothing can be added to the frame now, or nothing when it can.
	static std::optional<ElementError> BuildRefusal(const UiFrame& frame) {
		if (frame.inHook_) {
			return ElementError::BuildingInHook;
		}
		if (frame.inCallback_) {
			return ElementError::BuildingInCallback;
		}
		return std::nullopt;
	}
	/// Asserts, in debug builds, that neither a callback nor the hook is running, for the calls
	/// that add nodes, records or stored bytes. Changing a node's decor stays allowed in the hook,
	/// so that is not checked here.
	static void AssertNotInCallbackOrHook(const UiFrame& frame) {
		assert(!frame.inCallback_ && !frame.inHook_ &&
		       "callbacks and hooks must not build elements");
		(void)frame;
	}
	/// Asserts, in debug builds, that no callback is running.
	static void AssertNotInCallback(const UiFrame& frame) {
		assert(!frame.inCallback_ && "a callback must not build or change elements");
		(void)frame;
	}
	/// The row records of the lists Render walks, one for each depth of nesting, so a steady frame
	/// allocates nothing for them. Indices are used and not pointers, since a deeper list grows it.
	static std::vector<ListRows>& ListPool(UiFrame& frame) { return frame.listPool_; }
	/// The roots Render has drawn this frame.
	static std::vector<Element>& RenderedRoots(UiFrame& frame) { return frame.renderedRoots_; }

	/// The node at `index`, which must be a parent value a node holds. Walking up the tree needs
	/// this because a node records its parent as an index and not as a handle.
	static const Node& At(const UiFrame& frame, uint32_t index) { return frame.nodes_[index]; }

	/// Counts one natural measurement, which tests read to check the memo.
	static void CountNaturalMeasure(UiFrame& frame) { ++frame.naturalMeasures_; }
	/// Counts one call of Measure on a node, which tests read to check the work stays linear.
	static void CountMeasureCall(UiFrame& frame) { ++frame.measureCalls_; }
	/// Counts one text laid out again by Arrange.
	static void CountTextRelayout(UiFrame& frame) { ++frame.textRelayouts_; }
	/// Starts a new measure pass and returns its number, so natural sizes from an earlier pass
	/// are never reused.
	static uint32_t BeginMeasurePass(UiFrame& frame) { return ++frame.measureGeneration_; }
	/// The pass the frame is in.
	static uint32_t MeasurePass(const UiFrame& frame) { return frame.measureGeneration_; }

	/// Scratch for the flex solver, kept at capacity between frames. Arrange fills the items and
	/// reads the rectangles back, using the rectangles as a stack so nested boxes don't overlap.
	static std::vector<FlexItem>& FlexItems(UiFrame& frame) { return frame.flexItems_; }
	static std::vector<Rect>&     FlexRects(UiFrame& frame) { return frame.flexRects_; }
	/// The text layouts kept across frames.
	static std::vector<TextSlot>&       TextCache(UiFrame& frame) { return frame.textCache_; }
	static const std::vector<TextSlot>& TextCache(const UiFrame& frame) { return frame.textCache_; }
	/// Counts a text layout that had to be made, for tests of the cache.
	static void CountTextLayoutBuilt(UiFrame& frame) { ++frame.textLayoutsBuilt_; }
	/// One flag per child of each box being solved, a stack like the lengths and rectangles.
	static std::vector<uint8_t>& FlexCollapsed(UiFrame& frame) { return frame.flexCollapsed_; }
	/// The flex solver's working space.
	static FlexScratch& FlexWorkspace(UiFrame& frame) { return frame.flexScratch_; }
	static std::vector<float>&    FlexLengths(UiFrame& frame) { return frame.flexLengths_; }

	/// Why `element` is refused, or nothing when it is valid.
	static std::optional<ElementError> Check(const UiFrame& frame, Element element) {
		return frame.Check(element);
	}

	/// The child handles of `node`, in the order they were given. The span is valid until the next
	/// AddNode, which may move the child storage.
	static std::span<const Element> Children(const UiFrame& frame, const Node& node) {
		return frame.Children(node);
	}
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_FRAME_INTERNALS_H_
