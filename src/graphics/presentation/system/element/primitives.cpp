#include "graphics/presentation/system/element/primitives.h"

#include "graphics/presentation/system/components/box.h"
#include "graphics/presentation/system/effects/halo.h"
#include "graphics/presentation/system/effects/highlight.h"
#include "graphics/presentation/system/effects/shine.h"
#include "graphics/presentation/system/element/frameInternals.h"
#include "graphics/presentation/system/element/leaves.h"
#include "graphics/presentation/system/element/states.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace Libs::Graphics::SystemUi {

namespace {

// DrawBox takes reference pixels and converts them itself, so design units pass through as they
// are and round exactly as the older components did.
BoxStyle ToBoxStyle(const ResolvedStyle& style) {
	return {style.fill, style.radius, style.border, style.border_width, style.elevation};
}

// The decor record of a node, or a plain one for a node that has none.
const DecorData& DecorOf(const UiFrame& frame, const Node& node) {
	static constexpr DecorData PLAIN {};
	return node.decor == Node::NO_DECOR ? PLAIN : FrameInternals::Decors(frame)[node.decor];
}

// The look of a node once its states are merged in. Only boxes have states, so any other node
// resolves to no change.
ResolvedStyle ResolveNode(const UiFrame& frame, const Node& node) {
	if (node.kind != NodeKind::Box) {
		return {};
	}
	return ResolveStyle(frame.ActiveTheme(), FrameInternals::Boxes(frame)[node.payload].style,
	                    DecorOf(frame, node).input);
}

// The smallest scale an animation pose is given, since an easing that overshoots could take a
// small hidden scale through zero and flip the subtree.
constexpr float MIN_ANIMATED_SCALE = 1e-3f;

// A scale about the `origin` point of `rect`, then an offset, as a transform of layout space.
PaintTransform LocalTransform(const Rect& rect, Origin origin, float scale, Vec2 offset) {
	const Vec2 point = {rect.x + origin.x * rect.width, rect.y + origin.y * rect.height};
	return {scale, {point.x * (1.0f - scale) + offset.x, point.y * (1.0f - scale) + offset.y}};
}

// The transform that does `inner` first and then `outer`. The scales multiply and the outer scale
// applies to the inner translation.
PaintTransform Beneath(const PaintTransform& outer, const PaintTransform& inner) {
	return {outer.scale * inner.scale,
	        {outer.scale * inner.translate.x + outer.translate.x,
	         outer.scale * inner.translate.y + outer.translate.y}};
}

// The transform of a node's explicit pose with its animation inside it. The animation moves the
// node in its own space first, and the explicit pose then moves the result, so a card that slides
// in can still be nudged or shrunk from outside.
PaintTransform DecorTransform(const UiFrame& frame, const Node& node, const DecorData& decor) {
	const auto&    scale  = frame.Scale();
	const Vec2     offset = {scale.Px(decor.pose.offset.x), scale.Px(decor.pose.offset.y)};
	PaintTransform local  = LocalTransform(node.arranged, decor.origin, decor.pose.scale, offset);
	if (decor.animate.has_value() && decor.animated.has_value()) {
		const Pose& moving = *decor.animated;
		const Vec2  slide  = {scale.Px(moving.offset.x), scale.Px(moving.offset.y)};
		const float size   = std::max(moving.scale, MIN_ANIMATED_SCALE);
		local = Beneath(local, LocalTransform(node.arranged, decor.animate->origin, size, slide));
	}
	return local;
}

// The opacity a node's own pose and animation give it, each clamped on its own so two negative
// alphas can't multiply into a visible one.
float DecorOpacity(const DecorData& decor) {
	float opacity = std::clamp(decor.pose.alpha, 0.0f, 1.0f);
	if (decor.animate.has_value() && decor.animated.has_value()) {
		opacity *= std::clamp(decor.animated->alpha, 0.0f, 1.0f);
	}
	return opacity;
}

// The transform of a node's subtree. The node's pose maps x to O + k * (x - O) + o in layout
// space and the parent's transform then maps that, so the scales multiply and the parent's scale
// applies to the offset. A state that scales the node does so about its centre, before the pose.
PaintTransform Compose(const UiFrame& frame, const Node& node, const PaintTransform& parent,
                       float state_scale) {
	constexpr float               IDENTITY_SCALE = 1.0f;
	std::optional<PaintTransform> local;
	if (node.decor != Node::NO_DECOR) {
		local = DecorTransform(frame, node, FrameInternals::Decors(frame)[node.decor]);
	}
	if (state_scale != IDENTITY_SCALE) {
		const PaintTransform pressed = LocalTransform(node.arranged, CENTER, state_scale, {});
		local                        = local.has_value() ? Beneath(*local, pressed) : pressed;
	}
	return local.has_value() ? Beneath(parent, *local) : parent;
}

// The state a node paints with and passes to its children. Opacity multiplies, since a fading
// parent fades its children, and so do the pose and the node's states.
PaintState Inherit(const UiFrame& frame, const Node& node, const PaintState& parent,
                   const ResolvedStyle& resolved) {
	PaintState state = parent;
	state.transform  = Compose(frame, node, parent.transform, resolved.scale);
	if (node.decor != Node::NO_DECOR) {
		state.opacity *= DecorOpacity(FrameInternals::Decors(frame)[node.decor]);
	}
	state.opacity *= resolved.opacity;
	return state;
}

// The list data of a node, or null when it is not a list.
const ListData* ListOf(const UiFrame& frame, const Node& node) {
	if (node.interact == Node::NO_INTERACT) {
		return nullptr;
	}
	const ListData& list = FrameInternals::Interacts(frame)[node.interact].list;
	return list.is_list ? &list : nullptr;
}

// `transform` for the children of `node`, which sit higher by the scroll when it is a list. The
// scroll is layout and not a pose, so it comes before the node's own transform, which then moves
// the scrolled children along with the node.
PaintTransform ForChildren(const UiFrame& frame, const Node& node,
                           const PaintTransform& transform) {
	const ListData* list = ListOf(frame, node);
	if (list == nullptr || list->scroll == 0.0f) {
		return transform;
	}
	return Beneath(transform, {1.0f, {0.0f, -list->scroll}});
}

// Whether a child of the list `list` lies wholly above or below its area once scrolled, so
// nothing of it shows.
bool WhollyHidden(const Node& list_node, const ListData& list, const Node& child) {
	const Rect& area = list_node.arranged;
	const float top  = child.arranged.y - list.scroll;
	return top + child.arranged.height <= area.y || top >= area.Bottom();
}

// The transform that applies to a node's own drawing, from the root of its tree down.
PaintTransform TransformOf(const UiFrame& frame, const Node& node, const PaintTransform& root) {
	PaintTransform parent = root;
	if (node.parent != Node::NO_PARENT) {
		const Node& above = FrameInternals::At(frame, node.parent);
		parent            = ForChildren(frame, above, TransformOf(frame, above, root));
	}
	return Compose(frame, node, parent, ResolveNode(frame, node).scale);
}

// The corner radius in output pixels of a node drawn in `painted`. A box and an image round their
// corners by their radius, clamped to half the shorter side as DrawBox clamps it, and text has no
// corners. Effects and clips both use it, so they follow the drawn corners.
float Rounding(const UiFrame& frame, const Node& node, const Rect& painted,
               const ResolvedStyle& resolved) {
	constexpr float HALF   = 0.5f;
	float           radius = 0.0f;
	switch (node.kind) {
		case NodeKind::Box: radius = frame.Scale().Px(resolved.radius); break;
		case NodeKind::Image:
			radius = frame.Scale().Px(FrameInternals::Images(frame)[node.payload].radius.value);
			break;
		case NodeKind::Text:
		case NodeKind::Custom:
		case NodeKind::None: return 0.0f;
	}
	return std::min(radius, std::min(painted.width, painted.height) * HALF);
}

// The rectangle a node's effects go around. Text keeps its size and moves with its centre, so its
// effects do too, and an image that letterboxes is only as big as the picture, so a glow follows
// the picture and not the empty bars.
Rect EffectRect(const UiFrame& frame, const Node& node, const PaintTransform& transform,
                const Rect& painted) {
	switch (node.kind) {
		case NodeKind::Text: return MapCentred(node.arranged, transform);
		case NodeKind::Image: {
			const ImageData& data = FrameInternals::Images(frame)[node.payload];
			return PlacedImageRect(data.textureSize, painted, data.fit);
		}
		case NodeKind::Box:
		case NodeKind::Custom:
		case NodeKind::None: break;
	}
	return painted;
}

// While it lives, an unclipped node's drawing is clipped by the clip that was in effect outside
// the nearest clipping ancestor, and not by that ancestor's. It pushes that clip without
// intersecting, so clips further up are still part of it, and does nothing for other nodes.
class ClipEscape {
public:
	ClipEscape(ImDrawList& list, const PaintState& state, const Decor& decor)
	    : list_(list), active_(decor.unclipped && state.outer_clip.has_value()) {
		if (active_) {
			const ClipBounds& bounds = *state.outer_clip;
			list_.PushClipRect({bounds.min.x, bounds.min.y}, {bounds.max.x, bounds.max.y}, false);
		}
	}
	~ClipEscape() {
		if (active_) {
			list_.PopClipRect();
		}
	}
	ClipEscape(const ClipEscape&)            = delete;
	ClipEscape& operator=(const ClipEscape&) = delete;

private:
	ImDrawList& list_;
	bool        active_;
};

// What the effects of one node are drawn around and how, so each effect takes one argument for it.
struct EffectContext {
	const UiFrame&       frame;
	ImDrawList&          list;
	const Decor&         decor;
	const ResolvedStyle& resolved;
	Rect                 rect;
	float                rounding;
	const PaintState&    state;
};

// The glow behind a node, drawn before its body.
void DrawGlowEffect(const EffectContext& fx) {
	if (!fx.decor.glow.has_value()) {
		return;
	}
	const ClipEscape escape(fx.list, fx.state, fx.decor);
	DrawGlow(fx.list, fx.rect, fx.rounding, fx.decor.glow->glow, fx.decor.glow->color,
	         fx.frame.Scale(), fx.state.opacity);
}

// The selection fill, from the decor or from a selected state, which sits on the node's body.
void DrawSelectionEffect(const EffectContext& fx) {
	const float strength = fx.decor.selection ? 1.0f : fx.resolved.selection;
	if (strength <= 0.0f) {
		return;
	}
	const ClipEscape escape(fx.list, fx.state, fx.decor);
	DrawSelection(fx.list, fx.rect, fx.rounding, fx.frame.ActiveTheme(), fx.frame.Scale(),
	              ClampUnit(strength) * fx.state.opacity);
}

// The overlays of the node's states, in order, each as strong as its state. Zero strengths were
// left out when the states were merged.
void DrawStateOverlays(const EffectContext& fx) {
	for (size_t i = 0; i < fx.resolved.overlay_count; ++i) {
		const StateOverlay& overlay = fx.resolved.overlays[i];
		const Color color = overlay.color.MulAlpha(ClampUnit(overlay.weight) * fx.state.opacity);
		fx.list.AddRectFilled(TopLeft(fx.rect), BottomRight(fx.rect), ToImU32(color), fx.rounding);
	}
}

// The focus ring, from the decor or from the states that ask for one, outside everything else on
// the node's body.
void DrawFocusRingEffect(const EffectContext& fx) {
	std::optional<float> ring = fx.decor.focus_ring;
	if (fx.resolved.focus_ring > 0.0f) {
		ring = std::max(ring.value_or(0.0f), fx.resolved.focus_ring);
	}
	if (!ring.has_value()) {
		return;
	}
	const ClipEscape escape(fx.list, fx.state, fx.decor);
	DrawFocusRing(fx.list, fx.rect, fx.rounding, fx.frame.ActiveTheme(), fx.frame.Scale(),
	              ClampUnit(*ring) * fx.state.opacity);
}

// The shine over a node and its children.
void DrawShineEffect(const EffectContext& fx) {
	if (!fx.decor.shine.has_value()) {
		return;
	}
	const ClipEscape escape(fx.list, fx.state, fx.decor);
	const Theme&     theme = fx.frame.ActiveTheme();
	DrawShine(fx.list, fx.rect, fx.rounding, *fx.decor.shine, theme.shine,
	          theme.colors.effect.shine, fx.frame.Scale(), fx.state.opacity);
}

// Draws what the node is in `painted`, a box, an image or text.
void PaintBody(const UiFrame& frame, const Node& node, ImDrawList& list, const PaintState& state,
               const Rect& painted, const ResolvedStyle& resolved) {
	switch (node.kind) {
		case NodeKind::Text: PaintText(frame, node, list, state); return;
		case NodeKind::Image: PaintImage(frame, node, list, state); return;
		case NodeKind::Custom: PaintCustom(frame, node, list, state, painted); return;
		case NodeKind::Box: break;
		case NodeKind::None: return;
	}
	DrawBox(list, frame.Ui(), painted, ToBoxStyle(resolved), state.opacity);
}

// Starts a clipping box's clip for its children, if it is one. The clip is the painted rectangle
// less its rounded corners, so it follows the pose, and it intersects with the clip in effect. The
// clip already in effect is recorded in `inner` first, for an unclipped descendant to return to.
bool PushChildClip(const UiFrame& frame, const Node& node, ImDrawList& list, const Rect& painted,
                   const ResolvedStyle& resolved, PaintState& inner) {
	if (node.kind != NodeKind::Box ||
	    FrameInternals::Boxes(frame)[node.payload].style.overflow != Overflow::Clip) {
		return false;
	}
	const ImVec2 outerMin = list.GetClipRectMin();
	const ImVec2 outerMax = list.GetClipRectMax();
	inner.outer_clip      = ClipBounds {{outerMin.x, outerMin.y}, {outerMax.x, outerMax.y}};
	const Rect clip       = InsideCorners(painted, Rounding(frame, node, painted, resolved));
	list.PushClipRect(TopLeft(clip), BottomRight(clip), true);
	return true;
}

// Ends the clip PushChildClip started.
void PopChildClip(ImDrawList& list, bool clipped) {
	if (clipped) {
		list.PopClipRect();
	}
}

// Sets `reason` if given and returns the none handle.
Element Refuse(ElementError* reason, ElementError why) {
	RefuseBuild(why, reason);
	return {};
}

// The node `element` refers to, for a call that changes it, or null with `reason` set when it is
// none, stale or out of range, or when a callback is running, which must not change elements. The
// hook may, as restyling is what it is for.
Node* FindOrRefuse(UiFrame& frame, Element element, ElementError* reason) {
	FrameInternals::AssertNotInCallback(frame);
	if (FrameInternals::InCallback(frame)) {
		(void)Refuse(reason, ElementError::BuildingInCallback);
		return nullptr;
	}
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		(void)Refuse(reason, FrameInternals::Check(frame, element).value_or(ElementError::None));
	}
	return node;
}

// The decor record of `node`, made when it has none, so posing and decorating share one.
DecorData& DecorRecord(UiFrame& frame, Node& node) {
	FrameInternals::AssertNotInCallback(frame);
	auto& decors = FrameInternals::Decors(frame);
	if (node.decor == Node::NO_DECOR) {
		node.decor = static_cast<uint32_t>(decors.size());
		decors.emplace_back();
	}
	return decors[node.decor];
}

// A scale that is not positive and finite would flip or lose the subtree, an offset that is not
// finite would poison every translation below it, and a NaN alpha every opacity.
bool ValidPose(const ElementPose& pose) {
	const bool movable = std::isfinite(pose.offset.x) && std::isfinite(pose.offset.y);
	return std::isfinite(pose.scale) && pose.scale > 0.0f && movable && !std::isnan(pose.alpha);
}

// An animation with an origin that is not finite, or a transition whose scale is not positive and
// finite or whose slide is not finite, would poison the transforms below it.
bool ValidAnimate(const Animate& animate) {
	const auto valid = [](const Transition& transition) {
		return std::isfinite(transition.scale) && transition.scale > 0.0f &&
		       std::isfinite(transition.slide.x) && std::isfinite(transition.slide.y);
	};
	return std::isfinite(animate.origin.x) && std::isfinite(animate.origin.y) &&
	       valid(animate.timeline.enter) && valid(animate.timeline.exit);
}

// A NaN strength would reach the opacity of every effect it feeds, and a glow whose radius or
// peak is not finite would size or fade it by nonsense.
bool ValidDecor(const Decor& decor) {
	const bool glow_ok  = !decor.glow.has_value() || (std::isfinite(decor.glow->glow.radius) &&
                                                     std::isfinite(decor.glow->glow.peak_alpha));
	const bool shine_ok = !decor.shine.has_value() || !std::isnan(*decor.shine);
	const bool ring_ok  = !decor.focus_ring.has_value() || !std::isnan(*decor.focus_ring);
	return glow_ok && shine_ok && ring_ok;
}

} // namespace

Element Box(UiFrame& frame, const Layout& layout, const Style& style,
            std::span<const Element> children, ElementError* reason) {
	const Element box =
	    FrameInternals::AddNode(frame, NodeKind::Box, children, layout.item, reason);
	if (box.IsNone()) {
		return box;
	}
	auto& boxes                               = FrameInternals::Boxes(frame);
	FrameInternals::Find(frame, box)->payload = static_cast<uint32_t>(boxes.size());
	boxes.push_back({layout, style});
	return box;
}

// Draws a list's scroll bar's thumb over its rows, if they don't all fit. Only the thumb, so a list
// that has more to show says so without a track taking the eye from its rows. Worked out from the
// list's layout and then posed, as the rows' height is in layout's terms, and comparing it with a
// posed size would show a thumb on a list that fits.
void PaintListThumb(const UiFrame& frame, const Node& node, const ListData& scrolling,
                    ImDrawList& list, const PaintState& state) {
	const auto&               scale = frame.Scale();
	const std::optional<Rect> thumb =
	    ListThumb(node.arranged, scrolling.content, scrolling.scroll, scale.Px(LIST_THUMB_WIDTH),
	              scale.Px(LIST_THUMB_MIN));
	if (thumb.has_value()) {
		const Rect posed = MapRect(*thumb, state.transform);
		list.AddRectFilled(
		    TopLeft(posed), BottomRight(posed),
		    ToImU32(frame.ActiveTheme().colors.text.tertiary.MulAlpha(state.opacity)),
		    posed.width / 2.0f);
	}
}

void Paint(const UiFrame& frame, Element element, ImDrawList& list, const PaintState& state) {
	const Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr || node->kind == NodeKind::None || node->collapsed) {
		return;
	}
	const ResolvedStyle resolved = ResolveNode(frame, *node);
	const PaintState    own      = Inherit(frame, *node, state, resolved);
	const Rect          painted  = MapRect(node->arranged, own.transform);
	const Rect          around   = EffectRect(frame, *node, own.transform, painted);
	const EffectContext fx {frame,    list,   DecorOf(frame, *node).effects,
	                        resolved, around, Rounding(frame, *node, around, resolved),
	                        own};
	// Glow first so it sits behind, then the body, then what sits on it, with the focus ring on
	// top. The shine goes last, over the children. None of these is cut by the node's own clip,
	// which starts after them.
	DrawGlowEffect(fx);
	PaintBody(frame, *node, list, own, painted, resolved);
	DrawSelectionEffect(fx);
	DrawStateOverlays(fx);
	DrawFocusRingEffect(fx);
	PaintState inner          = own;
	const bool clipped        = PushChildClip(frame, *node, list, painted, resolved, inner);
	inner.transform           = ForChildren(frame, *node, inner.transform);
	const ListData* scrolling = ListOf(frame, *node);
	for (const Element child: FrameInternals::Children(frame, *node)) {
		const Node* below = FrameInternals::Find(frame, child);
		// Children scrolled out of a list's area are not worth drawing.
		if (scrolling != nullptr && below != nullptr && WhollyHidden(*node, *scrolling, *below)) {
			continue;
		}
		Paint(frame, child, list, inner);
	}
	PopChildClip(list, clipped);
	if (scrolling != nullptr) {
		PaintListThumb(frame, *node, *scrolling, list, own);
	}
	DrawShineEffect(fx);
}

namespace {

// The paint of a spacer, which is none.
struct DrawsNothing {
	void operator()(const CustomPaint&) const {}
};

} // namespace

Element Spacer(UiFrame& frame, const ItemLayout& item, ElementError* reason) {
	return Custom(frame, item, DrawsNothing {}, reason);
}

Element WithPose(UiFrame& frame, Element element, const ElementPose& pose, Origin origin,
                 ElementError* reason) {
	Node* node = FindOrRefuse(frame, element, reason);
	if (node == nullptr) {
		return {};
	}
	if (!ValidPose(pose)) {
		return Refuse(reason, ElementError::InvalidPose);
	}
	DecorData& record = DecorRecord(frame, *node);
	record.pose       = pose;
	record.origin     = origin;
	return element;
}

Element WithAnimation(UiFrame& frame, Element element, const Animate& animate,
                      ElementError* reason) {
	Node* node = FindOrRefuse(frame, element, reason);
	if (node == nullptr) {
		return {};
	}
	if (!ValidAnimate(animate)) {
		return Refuse(reason, ElementError::InvalidPose);
	}
	DecorData& record = DecorRecord(frame, *node);
	record.animate    = animate;
	record.animated.reset();
	return element;
}

std::optional<Phase> PhaseOf(const UiFrame& frame, Element element) {
	const Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr || node->decor == Node::NO_DECOR) {
		return std::nullopt;
	}
	const DecorData& decor = FrameInternals::Decors(frame)[node->decor];
	if (!decor.animated.has_value()) {
		return std::nullopt;
	}
	return decor.animated->phase;
}

Element Decorate(UiFrame& frame, Element element, const Decor& decor, ElementError* reason) {
	Node* node = FindOrRefuse(frame, element, reason);
	if (node == nullptr) {
		return {};
	}
	if (!ValidDecor(decor)) {
		return Refuse(reason, ElementError::InvalidDecor);
	}
	DecorRecord(frame, *node).effects = decor;
	return element;
}

bool SetInput(UiFrame& frame, Element element, const ItemInput& input, ElementError* reason) {
	Node* node = FindOrRefuse(frame, element, reason);
	if (node == nullptr) {
		return false;
	}
	DecorData& record = DecorRecord(frame, *node);
	record.input      = input;
	// A caller's input is the node's own, so it gets the default disabled fade whatever it says.
	record.input.inherited = false;
	record.input_explicit  = true;
	return true;
}

bool SetTextColor(UiFrame& frame, Element element, Color color, ElementError* reason) {
	Node* node = FindOrRefuse(frame, element, reason);
	if (node == nullptr) {
		return false;
	}
	if (node->kind != NodeKind::Text) {
		(void)Refuse(reason, ElementError::NotText);
		return false;
	}
	FrameInternals::Texts(frame)[node->payload].props.color = color;
	return true;
}

void InheritInput(UiFrame& frame, Node& node, const ItemInput& input) {
	if (node.kind != NodeKind::Box && node.kind != NodeKind::Custom) {
		return;
	}
	DecorData& record = DecorRecord(frame, node);
	if (!record.input_explicit) {
		record.input           = input;
		record.input.inherited = true;
	}
}

namespace {

// The node a key goes on, or null with `reason` set when `element` is refused. A key set after
// the element has a parent could repeat a sibling's, so such an element is refused.
Node* KeyTarget(UiFrame& frame, Element element, ElementError* reason) {
	Node* node = FindOrRefuse(frame, element, reason);
	if (node == nullptr) {
		return nullptr;
	}
	if (node->parent != Node::NO_PARENT) {
		(void)Refuse(reason, ElementError::AlreadyParented);
		return nullptr;
	}
	return node;
}

// The id of `node`, from its key or else its `index` among its siblings, under `seed`. A number key
// and an index hash alike, which is why AssignIds checks the ids and not just the keys.
ItemId NodeId(const Node& node, uint32_t index, ItemId seed) {
	if (!node.hasKey) {
		return IdWithSeed(index, seed);
	}
	return node.keyIsText ? IdWithSeed(node.key, seed) : IdWithSeed(node.keyNumber, seed);
}

// Clears the ids of `element` and its subtree, which a refused AssignIds leaves behind.
void ClearIds(UiFrame& frame, Element element) {
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return;
	}
	node->id = 0;
	for (const Element child: FrameInternals::Children(std::as_const(frame), *node)) {
		ClearIds(frame, child);
	}
}

// Whether any two ids in the frame's id scratch are equal.
bool HasRepeatedId(UiFrame& frame) {
	auto& ids = FrameInternals::IdScratch(frame);
	std::sort(ids.begin(), ids.end());
	return std::adjacent_find(ids.begin(), ids.end()) != ids.end();
}

// Gives the children of `element`, whose own id is set, their ids and then theirs below. A child's
// id is its key's or its index's under its parent's. The children are given ids before any of them
// is walked so that the shared scratch can check them for a repeat, which refuses siblings that
// the raw keys would let through, such as "a###x" and "b###x" or a number key equal to an index.
std::optional<ElementError> IdChildren(UiFrame& frame, Element element) {
	const Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return std::nullopt;
	}
	const ItemId parentId = node->id;
	const auto   children = FrameInternals::Children(std::as_const(frame), *node);
	auto&        ids      = FrameInternals::IdScratch(frame);
	ids.clear();
	for (uint32_t i = 0; i < children.size(); ++i) {
		Node* child = FrameInternals::Find(frame, children[i]);
		if (child != nullptr) {
			child->id = NodeId(*child, i, parentId);
			ids.push_back(child->id);
		}
	}
	if (HasRepeatedId(frame)) {
		return ElementError::IdCollision;
	}
	for (const Element child: children) {
		if (const auto failure = IdChildren(frame, child); failure.has_value()) {
			return failure;
		}
	}
	return std::nullopt;
}

} // namespace

Element Keyed(UiFrame& frame, Element element, std::string_view key, ElementError* reason) {
	if (key.empty()) {
		return Refuse(reason, ElementError::InvalidKey);
	}
	Node* node = KeyTarget(frame, element, reason);
	if (node == nullptr) {
		return {};
	}
	// Only a key that is going to be kept is copied, so a refusal wastes no frame storage.
	node->key       = frame.Intern(key);
	node->keyIsText = true;
	node->hasKey    = true;
	return element;
}

Element Keyed(UiFrame& frame, Element element, uint32_t key, ElementError* reason) {
	Node* node = KeyTarget(frame, element, reason);
	if (node == nullptr) {
		return {};
	}
	node->keyNumber = key;
	node->keyIsText = false;
	node->hasKey    = true;
	return element;
}

// `why`, once it has been logged, for AssignIds to return.
std::optional<ElementError> Reported(ElementError why) {
	(void)ReportRefusal(why);
	return why;
}

std::optional<ElementError> AssignIds(UiFrame& frame, Element root, ItemId seed) {
	// The walk reads the ids a callback would be rewriting.
	FrameInternals::AssertNotInCallback(frame);
	if (FrameInternals::InCallback(frame)) {
		return Reported(ElementError::BuildingInCallback);
	}
	Node* node = FrameInternals::Find(frame, root);
	if (node == nullptr) {
		return Reported(FrameInternals::Check(frame, root).value_or(ElementError::None));
	}
	// A node inside a tree gets its id from its place there, and as a root it would get another.
	if (node->parent != Node::NO_PARENT) {
		return Reported(ElementError::AlreadyParented);
	}
	const ItemId id    = NodeId(*node, 0, seed);
	auto&        roots = FrameInternals::RootIds(frame);
	// The roots are searched in a line, which is fine for the few a window has.
	const auto same  = [root](const RootRecord& record) { return record.element == root; };
	const auto taken = [id](const RootRecord& record) { return record.id == id; };
	// The same root assigned again is walked again, which gives the same ids, and keeps one record.
	// A failure leaves it with neither ids nor a record.
	std::erase_if(roots, same);
	if (std::any_of(roots.begin(), roots.end(), taken)) {
		ClearIds(frame, root);
		return Reported(ElementError::IdCollision);
	}
	node->id = id;
	if (const auto failure = IdChildren(frame, root); failure.has_value()) {
		ClearIds(frame, root);
		return Reported(*failure);
	}
	roots.push_back({root, id});
	return std::nullopt;
}

ItemId IdOf(const UiFrame& frame, Element element) {
	const Node* node = FrameInternals::Find(frame, element);
	return node == nullptr ? 0 : node->id;
}

Element WithTimelinePose(UiFrame& frame, Element element, const Pose& pose, Origin origin,
                         ElementError* reason) {
	return WithPose(frame, element, ToElementPose(pose), origin, reason);
}

Rect Painted(const UiFrame& frame, Element element, const PaintTransform& root) {
	const Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return {};
	}
	const PaintTransform transform = TransformOf(frame, *node, root);
	return node->kind == NodeKind::Text ? MapCentred(node->arranged, transform)
	                                    : MapRect(node->arranged, transform);
}

Rect Arranged(const UiFrame& frame, Element element) {
	const Node* node = FrameInternals::Find(frame, element);
	return node == nullptr ? Rect {} : node->arranged;
}

std::optional<Vec2> MeasuredAt(const UiFrame& frame, Element element, Vec2 limit) {
	const Node* node = FrameInternals::Find(frame, element);
	// A node with a parent may have had its size reassigned by the parent's layout since Measure,
	// which leaves the limit it was measured in unchanged, so only a root can be trusted.
	if (node == nullptr || node->parent != Node::NO_PARENT || !node->isMeasured ||
	    node->lastLimit.x != limit.x || node->lastLimit.y != limit.y) {
		return std::nullopt;
	}
	return node->measured;
}

} // namespace Libs::Graphics::SystemUi
