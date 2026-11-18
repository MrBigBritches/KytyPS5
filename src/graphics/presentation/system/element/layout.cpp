#include "graphics/presentation/system/element/frameInternals.h"
#include "graphics/presentation/system/element/leaves.h"
#include "graphics/presentation/system/element/primitives.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>

namespace Libs::Graphics::SystemUi {

namespace {

constexpr float UNBOUNDED = std::numeric_limits<float>::infinity();

// Every child without a fixed main-axis size gives up length in proportion to its size when its
// row overflows, down to its own minimum, so content never runs past its box.
constexpr float CHILD_SHRINK = 1.0f;

// A length along the main axis for a direction.
float MainOf(Vec2 size, FlexDirection direction) {
	return direction == FlexDirection::Row ? size.x : size.y;
}

// A length along the cross axis for a direction.
float CrossOf(Vec2 size, FlexDirection direction) {
	return direction == FlexDirection::Row ? size.y : size.x;
}

// A size from its main and cross lengths.
Vec2 FromAxes(float main, float cross, FlexDirection direction) {
	return direction == FlexDirection::Row ? Vec2 {main, cross} : Vec2 {cross, main};
}

// The space inside a box's padding that its children may use, given the space `max` on offer to
// the box. An explicit size replaces the offer on its axis. Infinity stays infinity.
Vec2 InnerLimit(const ScaleContext& scale, const Layout& layout, const EdgeInsets& padding,
                Vec2 max) {
	if (layout.item.width) {
		max.x = ToPixels(scale, *layout.item.width);
	}
	if (layout.item.height) {
		max.y = ToPixels(scale, *layout.item.height);
	}
	return {std::max(0.0f, max.x - padding.left - padding.right),
	        std::max(0.0f, max.y - padding.top - padding.bottom)};
}

// The least main-axis length a child can take. Its own minimum when it sets one, and otherwise
// its min-content, as CSS does with min-width auto, so its content never spills. A maximum below
// the min-content still wins. Flex items and the parent's min-content both use this, so the two
// never disagree about what a child needs.
float EffectiveMinMain(const ScaleContext& scale, const ItemLayout& item, float min_content_main) {
	if (item.min_main) {
		return ToPixels(scale, *item.min_main);
	}
	const float max_main = item.max_main ? ToPixels(scale, *item.max_main) : UNBOUNDED;
	return std::min(min_content_main, max_main);
}

// The same measurement as Measure, for the recursion inside one pass. Measure itself starts a
// new pass, so it can't be used here.
Vec2 MeasureNode(UiFrame& frame, Element element, Vec2 max);

// The flex item for a child of a container laid out by `container`. Its size is the one given.
// A child that sets its own cross size is not stretched, as in CSS, so a small icon in a tall row
// keeps its size instead of filling the row.
FlexItem ToFlexItem(const ScaleContext& scale, const Layout& container, const ItemLayout& item,
                    Vec2 size, float min_content_main) {
	FlexItem flex;
	flex.size       = size;
	flex.grow       = item.grow;
	flex.max_main   = item.max_main ? ToPixels(scale, *item.max_main) : UNBOUNDED;
	flex.min_main   = EffectiveMinMain(scale, item, min_content_main);
	flex.align_self = item.align_self;
	flex.collapses  = item.collapses;
	// A fixed main-axis size is a promise, so such a child keeps it and its siblings shrink
	// instead.
	const bool explicit_main  = container.direction == FlexDirection::Row ? item.width.has_value()
	                                                                      : item.height.has_value();
	flex.shrink               = explicit_main ? 0.0f : CHILD_SHRINK;
	const bool explicit_cross = container.direction == FlexDirection::Row ? item.height.has_value()
	                                                                      : item.width.has_value();
	if (explicit_cross && flex.align_self.value_or(container.align) == Align::Stretch) {
		flex.align_self = Align::Start;
	}
	return flex;
}

// What a box's children add up to along the main axis, and the longest of them across it.
struct ChildrenSize {
	float main  = 0.0f;
	float cross = 0.0f;
};

// Measures `child` with no limit on either axis, once per pass, and keeps the result as its
// unbounded size. Keyed by the room across, the memo missed whenever a row and a column
// alternated, and the work doubled at each level of depth, so the room is left out of the key. A
// later pass may offer different space, so a result from an earlier pass is never reused.
void MeasureUnbounded(UiFrame& frame, Element child, Node& node) {
	const uint32_t pass = FrameInternals::MeasurePass(frame);
	if (node.hasNatural && node.naturalGeneration == pass) {
		return;
	}
	FrameInternals::CountNaturalMeasure(frame);
	MeasureNode(frame, child, {UNBOUNDED, UNBOUNDED});
	node.unbounded         = node.measured;
	node.hasNatural        = true;
	node.naturalGeneration = pass;
}

// Gives every child its natural size, the size it wants with no limit along the main axis. A child
// wider across than the box has room for is measured again at that room, and that is its natural
// size, so wrapped text in a column is as tall as its lines there. The room across a child can
// narrow and widen within a pass, since a box that is measured more than once may be assigned a
// longer main length the second time, which a child inside it takes as its room across. A child
// therefore records the limit it was last measured at, and AssignLengths checks it.
void MeasureNatural(UiFrame& frame, std::span<const Element> children, FlexDirection direction,
                    float cross_cap) {
	for (const Element child: children) {
		Node* node = FrameInternals::Find(frame, child);
		if (node == nullptr) {
			continue;
		}
		MeasureUnbounded(frame, child, *node);
		node->natural = node->unbounded;
		if (CrossOf(node->unbounded, direction) > cross_cap) {
			MeasureNode(frame, child, FromAxes(UNBOUNDED, cross_cap, direction));
			node->natural = node->measured;
		}
	}
}

// Fills the frame's flex item scratch with one item per child. `use_measured_cross` takes the
// cross size from the assigned measurement, as arranging does, and otherwise from the natural one.
void BuildFlexItems(UiFrame& frame, const Layout& layout, std::span<const Element> children,
                    bool use_measured_cross) {
	auto& items = FrameInternals::FlexItems(frame);
	items.clear();
	const UiFrame& view = frame;
	for (const Element child: children) {
		const Node* node = FrameInternals::Find(view, child);
		if (node == nullptr) {
			items.emplace_back();
			continue;
		}
		const Vec2 cross_source = use_measured_cross ? node->measured : node->natural;
		const Vec2 size         = FromAxes(MainOf(node->natural, layout.direction),
		                                   CrossOf(cross_source, layout.direction), layout.direction);
		items.push_back(ToFlexItem(view.Scale(), layout, FrameInternals::Item(view, *node), size,
		                           MainOf(node->minContent, layout.direction)));
	}
}

// Which layout a node's subtree already holds, if it is the one for the limit about to be assigned.
// The two differ in where the node's size comes from, since a node kept for its exact limit has
// the size of that measure, which can differ from its natural size when its length was clamped.
enum class Held {
	No,
	Exact,
	Natural,
};

// Whether `node`'s subtree already holds the layout for the limit it is about to be assigned. That
// is so when it was last measured at exactly that limit, or when its length is unchanged and it was
// last measured as MeasureNatural measures it, with no limit along the main axis and the room
// across only when it didn't fit. A subtree last measured at some other room across holds sizes for
// that room, even when the node's own natural size looks right.
Held HoldsLayout(const Node& node, FlexDirection direction, float final_main, float cross_cap) {
	const Vec2 assigned = FromAxes(final_main, cross_cap, direction);
	if (node.lastLimit.x == assigned.x && node.lastLimit.y == assigned.y) {
		return Held::Exact;
	}
	if (final_main != MainOf(node.natural, direction)) {
		return Held::No;
	}
	const float natural_cross =
	    CrossOf(node.unbounded, direction) > cross_cap ? cross_cap : UNBOUNDED;
	const Vec2 natural_limit = FromAxes(UNBOUNDED, natural_cross, direction);
	const bool natural = node.lastLimit.x == natural_limit.x && node.lastLimit.y == natural_limit.y;
	return natural ? Held::Natural : Held::No;
}

// Gives each child its final main length from the flex solver and measures it again at that
// length, so a child that grew or shrank can lay out its own content in the new space. A child
// that already holds the layout for its limit keeps it, which spares nested boxes a second pass.
// The result is the children's lengths added up, with the longest cross size.
ChildrenSize AssignLengths(UiFrame& frame, const Layout& layout, std::span<const Element> children,
                           float inner_main, float cross_cap) {
	const auto  direction = layout.direction;
	const float gap       = ToPixels(frame.Scale(), layout.gap);
	BuildFlexItems(frame, layout, children, false);
	// The lengths are a stack, so a nested box's lengths land after this box's and never overlap.
	auto&        lengths = FrameInternals::FlexLengths(frame);
	const size_t base    = lengths.size();
	lengths.resize(base + children.size());
	// The flags are a stack too, and read by index, as a nested box's solve may move them.
	auto&        flags     = FrameInternals::FlexCollapsed(frame);
	const size_t flag_base = flags.size();
	flags.resize(flag_base + children.size());
	(void)FlexMainLengths(inner_main, direction, gap, FrameInternals::FlexItems(frame),
	                      std::span<float>(lengths).subspan(base, children.size()),
	                      std::span<uint8_t>(flags).subspan(flag_base, children.size()),
	                      FrameInternals::FlexWorkspace(frame));

	// The gaps are between the children the solver left visible, and it says which it collapsed.
	ChildrenSize size;
	const auto   first_flag = flags.begin() + static_cast<std::ptrdiff_t>(flag_base);
	const size_t hidden     = static_cast<size_t>(
        std::count(first_flag, first_flag + static_cast<std::ptrdiff_t>(children.size()), uint8_t {1}));
	const size_t visible = children.size() - hidden;
	size.main            = visible > 1 ? gap * static_cast<float>(visible - 1) : 0.0f;
	for (size_t i = 0; i < children.size(); ++i) {
		const float final_main = lengths[base + i];
		const Node* node       = FrameInternals::Find(std::as_const(frame), children[i]);
		if (node == nullptr) {
			continue;
		}
		if (flags[flag_base + i] != 0) {
			// A collapsed child is absent. It keeps the cross size it naturally has, for its
			// rectangle, but it is not measured again at no length, where text would wrap a glyph
			// to a line, and it does not count toward the box's cross size.
			FrameInternals::Find(frame, children[i])->measured =
			    FromAxes(0.0f, CrossOf(node->natural, direction), direction);
			continue;
		}
		Vec2       assigned = node->natural;
		const Held held     = HoldsLayout(*node, direction, final_main, cross_cap);
		if (held == Held::No) {
			assigned = MeasureNode(frame, children[i], FromAxes(final_main, cross_cap, direction));
		} else if (held == Held::Exact) {
			// The result of that exact measure, since a clamped length can wrap the node taller
			// than its natural size.
			assigned = node->measured;
		}
		// The child takes exactly the length it was assigned, whatever it would have chosen.
		Node* mutable_node     = FrameInternals::Find(frame, children[i]);
		mutable_node->measured = FromAxes(final_main, CrossOf(assigned, direction), direction);
		size.main += final_main;
		size.cross = std::max(size.cross, CrossOf(assigned, direction));
	}
	lengths.resize(base);
	flags.resize(flag_base);
	return size;
}

// The flex style a box's layout asks for, with lengths in output pixels.
FlexStyle ToFlexStyle(const ScaleContext& scale, const Layout& layout) {
	return {.direction = layout.direction,
	        .justify   = layout.justify,
	        .align     = layout.align,
	        .gap       = ToPixels(scale, layout.gap),
	        .padding   = ToPixels(scale, layout.padding)};
}

// What a box's children need at least, along its main axis and across it. A child contributes
// its own minimum when it sets one, and its min-content otherwise. A child that collapses is
// counted as collapsed, since the box can always be as short as that, so it adds no length along
// the main axis and no gap beside it. It still counts across, since a collapsed child is as wide or
// tall as it was in the other direction.
Vec2 ChildrenMinContent(const UiFrame& frame, const Layout& layout,
                        std::span<const Element> children) {
	const auto direction = layout.direction;
	float      main      = 0.0f;
	float      cross     = 0.0f;
	size_t     counted   = 0;
	for (const Element child: children) {
		const Node* node = FrameInternals::Find(frame, child);
		if (node == nullptr) {
			continue;
		}
		const ItemLayout& item = FrameInternals::Item(frame, *node);
		cross                  = std::max(cross, CrossOf(node->minContent, direction));
		if (item.collapses) {
			continue;
		}
		main += EffectiveMinMain(frame.Scale(), item, MainOf(node->minContent, direction));
		++counted;
	}
	if (counted > 1) {
		main += ToPixels(frame.Scale(), layout.gap) * static_cast<float>(counted - 1);
	}
	return FromAxes(main, cross, direction);
}

// Measures a stacked box, whose children share its inner rectangle. Each child is measured in the
// space inside the padding, and the box is the largest of them less nothing, plus the padding. An
// explicit size on an axis replaces it, as for any box.
NodeSize MeasureStack(UiFrame& frame, const Layout& layout, std::span<const Element> children,
                      const EdgeInsets& padding, Vec2 inner) {
	Vec2 size;
	Vec2 min;
	for (const Element child: children) {
		const Vec2  found = MeasureNode(frame, child, inner);
		const Node* node  = FrameInternals::Find(std::as_const(frame), child);
		size              = {std::max(size.x, found.x), std::max(size.y, found.y)};
		if (node != nullptr) {
			min = {std::max(min.x, node->minContent.x), std::max(min.y, node->minContent.y)};
		}
	}
	for (Vec2* axes: {&size, &min}) {
		axes->x += padding.left + padding.right;
		axes->y += padding.top + padding.bottom;
	}
	const auto& scale = std::as_const(frame).Scale();
	if (layout.item.width) {
		size.x = min.x = ToPixels(scale, *layout.item.width);
	}
	if (layout.item.height) {
		size.y = min.y = ToPixels(scale, *layout.item.height);
	}
	return {size, min};
}

// Measures a box and its subtree in the space `max`. A box with an explicit size on an axis is
// that size on it, and its min-content is too.
NodeSize MeasureBox(UiFrame& frame, const Node& node, Vec2 max) {
	const Layout&    layout    = FrameInternals::Boxes(std::as_const(frame))[node.payload].layout;
	const auto&      scale     = frame.Scale();
	const EdgeInsets padding   = ToPixels(scale, layout.padding);
	const Vec2       inner     = InnerLimit(scale, layout, padding, max);
	const auto       children  = FrameInternals::Children(std::as_const(frame), node);
	const auto       cross_cap = CrossOf(inner, layout.direction);
	if (layout.stacked) {
		return MeasureStack(frame, layout, children, padding, inner);
	}

	MeasureNatural(frame, children, layout.direction, cross_cap);
	const ChildrenSize content =
	    AssignLengths(frame, layout, children, MainOf(inner, layout.direction), cross_cap);
	Vec2 min  = ChildrenMinContent(frame, layout, children);
	Vec2 size = FromAxes(content.main, content.cross, layout.direction);
	for (Vec2* axes: {&size, &min}) {
		axes->x += padding.left + padding.right;
		axes->y += padding.top + padding.bottom;
	}
	if (layout.item.width) {
		size.x = min.x = ToPixels(scale, *layout.item.width);
	}
	if (layout.item.height) {
		size.y = min.y = ToPixels(scale, *layout.item.height);
	}
	return {size, min};
}

Vec2 MeasureNode(UiFrame& frame, Element element, Vec2 max) {
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return {};
	}
	FrameInternals::CountMeasureCall(frame);
	node->isMeasured = true;
	node->lastLimit  = max;
	NodeSize found;
	switch (node->kind) {
		case NodeKind::Box: found = MeasureBox(frame, *node, max); break;
		case NodeKind::Text: found = MeasureText(frame, *node, max); break;
		case NodeKind::Image: found = MeasureImage(frame, *node); break;
		case NodeKind::Custom: found = MeasureCustom(frame, *node); break;
		case NodeKind::None: break;
	}
	node->measured   = found.size;
	node->minContent = found.minContent;
	return found.size;
}

} // namespace

Vec2 Measure(UiFrame& frame, Element element, Vec2 max) {
	(void)FrameInternals::BeginMeasurePass(frame);
	return MeasureNode(frame, element, max);
}

namespace {

bool ArrangeNode(UiFrame& frame, Element element, const Rect& rect);

// Arranges a stacked box's children, each filling the space inside the padding.
bool ArrangeStack(UiFrame& frame, const Layout& layout, std::span<const Element> children,
                  const Rect& rect) {
	const Rect inner = rect.Inset(ToPixels(std::as_const(frame).Scale(), layout.padding));
	bool       all   = true;
	for (const Element child: children) {
		if (Node* below = FrameInternals::Find(frame, child)) {
			below->collapsed = false;
		}
		all = ArrangeNode(frame, child, inner) && all;
	}
	return all;
}

// Arranges `element` and what is below it. The element's own collapsed flag is its parent's to set,
// so it is read and left alone here, and the flags of its children are set from its own solve.
bool ArrangeNode(UiFrame& frame, Element element, const Rect& rect) {
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr || !node->isMeasured) {
		return false;
	}
	node->arranged = rect;
	if (node->collapsed) {
		// Absent: the subtree keeps what it was measured as, and is neither painted nor walked.
		return true;
	}
	if (node->kind == NodeKind::Text) {
		ArrangeText(frame, *node, rect);
	}
	if (node->kind != NodeKind::Box) {
		return true;
	}
	const UiFrame& view     = frame;
	const Layout&  layout   = FrameInternals::Boxes(view)[node->payload].layout;
	const auto     children = FrameInternals::Children(view, *node);
	if (layout.stacked) {
		return ArrangeStack(frame, layout, children, rect);
	}
	BuildFlexItems(frame, layout, children, true);

	// The rectangles are a stack, so a nested solve lands after this box's and the two never
	// overlap. Children are read by index because the nested solve may grow the storage.
	auto&        rects = FrameInternals::FlexRects(frame);
	const size_t base  = rects.size();
	rects.resize(base + children.size());
	auto&        flags     = FrameInternals::FlexCollapsed(frame);
	const size_t flag_base = flags.size();
	flags.resize(flag_base + children.size());
	bool arranged = SolveFlex(rect, ToFlexStyle(view.Scale(), layout), FrameInternals::FlexItems(frame),
	                          std::span<Rect>(rects).subspan(base, children.size()),
	                          std::span<uint8_t>(flags).subspan(flag_base, children.size()),
	                          FrameInternals::FlexWorkspace(frame));
	for (size_t i = 0; i < children.size(); ++i) {
		const Rect placed = rects[base + i];
		if (Node* below = FrameInternals::Find(frame, children[i])) {
			below->collapsed = flags[flag_base + i] != 0;
		}
		arranged = ArrangeNode(frame, children[i], placed) && arranged;
	}
	rects.resize(base);
	flags.resize(flag_base);
	return arranged;
}

} // namespace

bool Arrange(UiFrame& frame, Element element, const Rect& rect) {
	// The element given is the root of this arrange, so it is laid out whatever an earlier arrange
	// of its parent decided. Only a parent collapses a child.
	if (Node* node = FrameInternals::Find(frame, element)) {
		node->collapsed = false;
	}
	return ArrangeNode(frame, element, rect);
}

} // namespace Libs::Graphics::SystemUi
