#include "graphics/presentation/system/element/interact.h"

#include "graphics/presentation/system/element/frameInternals.h"
#include "graphics/presentation/system/element/leaves.h"
#include "graphics/presentation/system/imgui/interactContext.h"

#include <algorithm>
#include <utility>

namespace Libs::Graphics::SystemUi {

namespace {

// Sets `reason` if given and returns the none handle.
Element Refuse(ElementError* reason, ElementError why) {
	RefuseBuild(why, reason);
	return {};
}

// True, with `reason` set if given, when Render's hook or a callback is running, neither of which
// may add records. Debug builds assert instead.
bool RefusedInsideRender(const UiFrame& frame, ElementError* reason) {
	FrameInternals::AssertNotInCallbackOrHook(frame);
	const auto why = FrameInternals::BuildRefusal(frame);
	if (!why.has_value()) {
		return false;
	}
	(void)Refuse(reason, *why);
	return true;
}

// The interaction record of `node`, made when it has none, so MakeInteractive and Specimen share
// one.
InteractData& InteractRecord(UiFrame& frame, Node& node) {
	FrameInternals::AssertNotInCallbackOrHook(frame);
	auto& records = FrameInternals::Interacts(frame);
	if (node.interact == Node::NO_INTERACT) {
		node.interact = static_cast<uint32_t>(records.size());
		records.emplace_back();
	}
	return records[node.interact];
}

// The interaction record of `node`, or null when it has none.
const InteractData* RecordOf(const UiFrame& frame, const Node& node) {
	return node.interact == Node::NO_INTERACT ? nullptr
	                                          : &FrameInternals::Interacts(frame)[node.interact];
}

// Whether `node` is interactive.
bool IsInteractive(const UiFrame& frame, const Node& node) {
	const InteractData* record = RecordOf(frame, node);
	return record != nullptr && record->interactive;
}

// Whether any node above `node` is interactive.
bool InteractiveAbove(const UiFrame& frame, const Node& node) {
	for (uint32_t parent = node.parent; parent != Node::NO_PARENT;) {
		const Node& above = FrameInternals::At(frame, parent);
		if (IsInteractive(frame, above)) {
			return true;
		}
		parent = above.parent;
	}
	return false;
}

// Whether any node below `node` is interactive.
bool InteractiveBelow(const UiFrame& frame, const Node& node) {
	for (const Element child: FrameInternals::Children(frame, node)) {
		const Node* below = FrameInternals::Find(frame, child);
		if (below != nullptr && (IsInteractive(frame, *below) || InteractiveBelow(frame, *below))) {
			return true;
		}
	}
	return false;
}

// Calls the callbacks of an interactive node, the input one first. The record was just filled, so
// both see this frame's input. The callbacks and the input are copied out first, so nothing the
// walk holds is read after a callback runs. A callback must not build or change elements, which
// debug builds assert, since that could move the storage its own bytes sit in.
void RunCallbacks(UiFrame& frame, const InteractData& record, Element element) {
	const StoredCallback<InputInvoke>    onInput    = record.on_input;
	const StoredCallback<ActivateInvoke> onActivate = record.on_activate;
	const NodeInput                      input      = *record.input;
	FrameInternals::SetInCallback(frame, true);
	if (onInput.invoke != nullptr) {
		onInput.invoke(StoredBytes(frame, onInput.offset), input);
	}
	if (onActivate.invoke != nullptr && input.interaction.state.activated) {
		onActivate.invoke(StoredBytes(frame, onActivate.offset), element);
	}
	FrameInternals::SetInCallback(frame, false);
}

// Registers one interactive node with ImGui, records what the player is doing to it, runs its
// callbacks, and returns the node if it was activated.
Element InteractNode(UiFrame& frame, InteractContext& context, Element element,
                     float scroll_offset) {
	const Node*          node   = FrameInternals::Find(frame, element);
	InteractData&        record = FrameInternals::Interacts(frame)[node->interact];
	const InteractProps& props  = record.props;
	NodeInput            input;
	input.id  = node->id;
	input.hit = node->arranged;
	input.hit.y += scroll_offset;
	input.interaction = Interact(context, input.id, input.hit, props.disabled);
	input.keys        = record.key_presses.value_or(KeyPresses {});
	// ImGui would hand first focus to a disabled default, which then shows no focus at all.
	if (props.default_focus && !props.disabled) {
		MakeLastControlDefaultFocus();
	}
	if (props.dpad != DpadClaim::None) {
		if (props.disabled) {
			// A disabled node leaves the D-pad to navigation, even a direction held from before.
			ReleaseHorizontal(input.id);
		} else {
			input.dpad = ClaimHorizontal(input.id, props.dpad == DpadClaim::Repeat);
		}
	}
	if (props.triggers && !props.disabled) {
		input.triggers = ClaimTriggers(input.id);
	}
	(void)SetInput(frame, element, {input.interaction.visuals, props.disabled, props.selected});
	record.input = input;
	RunCallbacks(frame, record, element);
	return input.interaction.state.activated ? element : Element {};
}

// Whether `node` is a list.
bool IsList(const UiFrame& frame, const Node& node) {
	const InteractData* record = RecordOf(frame, node);
	return record != nullptr && record->list.is_list;
}

// Narrows the window's clip for hit testing to a clipping box's rectangle less its rounded corners,
// the same one Paint clips its children to but laid out and not posed, since hit tests ignore
// poses. A list clips to its exact rectangle instead, as ListScope does, so a row partly in view
// near a rounded corner can still be hovered. Returns whether it pushed one, for PopChildHitClip.
bool PushChildHitClip(const UiFrame& frame, const Node& node, float scroll_offset) {
	if (node.kind != NodeKind::Box) {
		return false;
	}
	const Style& style = FrameInternals::Boxes(frame)[node.payload].style;
	if (style.overflow != Overflow::Clip) {
		return false;
	}
	constexpr float HALF = 0.5f;
	Rect            area = node.arranged;
	area.y += scroll_offset;
	// The base radius, not the state-resolved one, as hit tests ignore the pose too. A hover that
	// changed the radius would move the hit edge and could drop the hover again, flickering.
	const float radius =
	    std::min(frame.Scale().Px(style.radius.value), std::min(area.width, area.height) * HALF);
	PushHitClip(IsList(frame, node) ? area : InsideCorners(area, radius));
	return true;
}

// Plays the keyed animation of `node`, if it has one, at the frame's time and keeps the pose for
// Paint. The store keeps the start and dismissal under the node's id. This runs before the node
// interacts, so nothing it does can move the decor record afterwards.
void PlayAnimation(UiFrame& frame, InteractContext& context, const Node& node) {
	if (node.decor == Node::NO_DECOR) {
		return;
	}
	DecorData& decor = FrameInternals::Decors(frame)[node.decor];
	if (decor.animate.has_value()) {
		decor.animated = context.store.Play(node.id, decor.animate->timeline,
		                                    decor.animate->dismissed, context.now);
	}
}

// Claims the key groups `node` owns under its path id and reads their presses as that owner. This
// runs before the node interacts, so its callbacks see them, and in tree order, so of two Control
// claims the first node keeps the group.
void ClaimNodeKeys(UiFrame& frame, const Node& node) {
	if (node.interact == Node::NO_INTERACT) {
		return;
	}
	InteractData&   record = FrameInternals::Interacts(frame)[node.interact];
	const KeyOwner& keys   = record.keys;
	record.key_presses.reset();
	if (!keys.shoulders && !keys.back) {
		return;
	}
	KeyPresses found;
	if (keys.shoulders) {
		(void)ClaimKeys(KeyGroup::Shoulders, node.id, keys.priority);
		found.shoulders = ShoulderPresses(node.id);
	}
	if (keys.back) {
		(void)ClaimKeys(KeyGroup::Back, node.id, keys.priority);
		found.back = BackKeyPressed(node.id);
	}
	record.key_presses = found;
	// Reporting a press is acting on it, so layers drawn after this node stand down. A claim alone
	// doesn't stop a legacy read that accepts any owner, which only a consumed key does, as
	// TestElementBackIsConsumedForLegacyReaders shows.
	if (found.shoulders.left || found.shoulders.right) {
		ConsumeKey(KeyGroup::Shoulders, node.id);
	}
	if (found.back) {
		ConsumeKey(KeyGroup::Back, node.id);
	}
}

// What the walk carries down the tree.
struct WalkState {
	// The fixed pose of the nearest specimen above, or null.
	const ItemVisuals* pose = nullptr;
	// The input of the nearest interactive ancestor, or null.
	const ItemInput* inherited = nullptr;
	// Where the first activation is recorded.
	RenderResult* result = nullptr;
	// How far the lists above have scrolled the node, which moves it up on the screen.
	float scroll_offset = 0.0f;
	// The index in the frame's list pool of the nearest list above, or -1.
	int list = -1;
};

// Interacts with `element` if it is interactive and returns the input to pass down its subtree,
// which is the one it found, written to `found`, or else the one it was given from above. Nesting
// is refused by MakeInteractive, so one interactive ancestor's input never replaces another's.
const ItemInput* InteractOrInherit(UiFrame& frame, InteractContext& context, Element element,
                                   const WalkState& walk, ItemInput& found) {
	Node* node = FrameInternals::Find(frame, element);
	if (!IsInteractive(frame, *node)) {
		if (walk.inherited != nullptr) {
			InheritInput(frame, *node, *walk.inherited);
		}
		return walk.inherited;
	}
	ListRows*       rows = walk.list < 0 ? nullptr : &FrameInternals::ListPool(frame)[walk.list];
	InteractContext at {context.ui, context.store, context.list, context.now, rows, walk.pose};
	const Element   activated = InteractNode(frame, at, element, walk.scroll_offset);
	if (walk.result->activated.IsNone() && !activated.IsNone()) {
		walk.result->activated = activated;
	}
	const InteractData& record = *RecordOf(frame, *FrameInternals::Find(frame, element));
	found = {record.input->interaction.visuals, record.props.disabled, record.props.selected};
	return &found;
}

// A list being walked. Its rows are recorded in the frame's pool at its depth of nesting.
struct ListRun {
	bool   active = false;
	int    index  = -1;
	ItemId id     = 0;
	Rect   area;
	float  inset = 0.0f;
	// How far its children reach, for its content when it has no rows to measure it by.
	float children_reach = 0.0f;
};

// The space a list keeps clear at its top and bottom, which is its padding, the larger of the two.
// It is what the rows sit in from the edges by, so scrolling a row into view uses the area the
// rows lay out in.
float InsetOf(const UiFrame& frame, const Node& node) {
	const DpInsets& padding = FrameInternals::Boxes(frame)[node.payload].layout.padding;
	return std::max(ToPixels(frame.Scale(), padding.top), ToPixels(frame.Scale(), padding.bottom));
}

// How far below `node`'s top its children reach, which is how tall a list's content is when none of
// it takes focus, so no rows are recorded to measure it by.
float ChildrenReach(const UiFrame& frame, const Node& node) {
	float bottom = 0.0f;
	for (const Element child: FrameInternals::Children(frame, node)) {
		if (const Node* below = FrameInternals::Find(frame, child); below != nullptr) {
			bottom = std::max(bottom, below->arranged.Bottom() - node.arranged.y);
		}
	}
	return bottom;
}

// Starts the walk of `node` as a list if it is one, and moves its children up by its scroll. Each
// depth of nesting has its own record in the pool, which grows only when a deeper list first
// appears.
void BeginList(UiFrame& frame, InteractContext& context, const Node& node, WalkState& walk,
               ListRun& run) {
	InteractData* record = node.interact == Node::NO_INTERACT
	                           ? nullptr
	                           : &FrameInternals::Interacts(frame)[node.interact];
	if (record == nullptr || !record->list.is_list) {
		return;
	}
	auto& memory = context.store.List(node.id);
	run.active   = true;
	run.id       = node.id;
	run.area     = node.arranged;
	run.area.y += walk.scroll_offset;
	// A specimen takes no input, so the pointer doesn't scroll it, and its bar is only shown.
	if (walk.pose == nullptr) {
		const auto& scale = frame.Scale();
		ScrollListWithPointer(memory, node.id, run.area,
		                      ListThumb(run.area, memory.content, memory.scroll,
		                                scale.Px(LIST_THUMB_WIDTH), scale.Px(LIST_THUMB_MIN)),
		                      scale.Px(LIST_THUMB_HIT_WIDTH), scale.Px(LIST_WHEEL_STEP));
		if (record->list.pad_scrolls) {
			ScrollListWithPad(memory, run.area, scale.Px(LIST_WHEEL_STEP));
		}
	}
	run.inset  = InsetOf(frame, node);
	run.index  = walk.list + 1;
	auto& pool = FrameInternals::ListPool(frame);
	if (pool.size() <= static_cast<size_t>(run.index)) {
		pool.resize(run.index + 1);
	}
	ListRows& rows = pool[run.index];
	rows.rows.clear();
	rows.area           = run.area;
	rows.focus_inside   = memory.inside;
	record->list.scroll  = memory.scroll;
	record->list.content = memory.content;
	walk.scroll_offset -= memory.scroll;
	walk.list = run.index;
}

// Ends a list's walk, keeping its focused row in view.
void EndList(UiFrame& frame, InteractContext& context, const ListRun& run) {
	if (!run.active) {
		return;
	}
	const ListRows& rows = FrameInternals::ListPool(frame)[run.index];
	FinishList(context.store, run.id, std::span<const RecordedRow>(rows.rows), run.area, run.inset,
	           run.children_reach);
}

// Walks `element` and what is below it in tree order, with `walk` as the state from above.
void InteractWalk(UiFrame& frame, InteractContext& context, Element element, WalkState walk) {
	const Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr || node->collapsed) {
		// A collapsed node and what is below it are absent, so nothing of them is registered with
		// ImGui and navigation never reaches them.
		return;
	}
	const InteractData* record = RecordOf(frame, *node);
	if (record != nullptr && record->posed) {
		walk.pose = &record->pose;
	}
	PlayAnimation(frame, context, *node);
	// A specimen takes no input, so it owns no keys either.
	if (walk.pose == nullptr) {
		ClaimNodeKeys(frame, *node);
	}
	ItemInput found;
	walk.inherited = InteractOrInherit(frame, context, element, walk, found);
	node           = FrameInternals::Find(frame, element);
	// Children of a clipping box can only be reached where they are drawn. The box itself sits
	// where its own parents' lists put it, and not where its own scroll moves its children.
	const bool clipped = PushChildHitClip(frame, *node, walk.scroll_offset);
	ListRun    run;
	BeginList(frame, context, *node, walk, run);
	run.children_reach = run.active ? ChildrenReach(frame, *node) : 0.0f;
	for (const Element child: FrameInternals::Children(std::as_const(frame), *node)) {
		InteractWalk(frame, context, child, walk);
	}
	EndList(frame, context, run);
	if (clipped) {
		PopHitClip();
	}
}

// The record a callback goes on, or null with `reason` set when `element` is refused. Only an
// interactive node is called back, so any other is refused.
InteractData* CallbackTarget(UiFrame& frame, Element element, ElementError* reason) {
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		(void)Refuse(reason, FrameInternals::Check(frame, element).value_or(ElementError::None));
		return nullptr;
	}
	if (!IsInteractive(frame, *node)) {
		(void)Refuse(reason, ElementError::NotInteractive);
		return nullptr;
	}
	return &FrameInternals::Interacts(frame)[node->interact];
}

} // namespace

Interaction Interact(UiFrame& frame, ItemId id, const Rect& rect, bool disabled) {
	InteractContext context {frame.Ui(), frame.Store(), CurrentDrawList(), frame.Now()};
	return Interact(context, id, rect, disabled);
}

Element MakeInteractive(UiFrame& frame, Element element, const InteractProps& props,
                        ElementError* reason) {
	if (RefusedInsideRender(frame, reason)) {
		return {};
	}
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return Refuse(reason, FrameInternals::Check(frame, element).value_or(ElementError::None));
	}
	if (InteractiveAbove(frame, *node) || InteractiveBelow(frame, *node)) {
		return Refuse(reason, ElementError::NestedInteractive);
	}
	if (IsInteractive(frame, *node)) {
		// Replacing the record would silently drop what the first call set, so a change to one
		// field has its own call, such as ClaimDpad.
		return Refuse(reason, ElementError::AlreadyInteractive);
	}
	InteractData& record = InteractRecord(frame, *node);
	record.props         = props;
	record.keys          = props.keys;
	record.interactive   = true;
	return element;
}

Element OwnKeys(UiFrame& frame, Element element, const KeyOwner& keys, ElementError* reason) {
	if (RefusedInsideRender(frame, reason)) {
		return {};
	}
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return Refuse(reason, FrameInternals::Check(frame, element).value_or(ElementError::None));
	}
	InteractRecord(frame, *node).keys = keys;
	return element;
}

std::optional<KeyPresses> KeyPressesOf(const UiFrame& frame, Element element) {
	const Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return std::nullopt;
	}
	const InteractData* record = RecordOf(frame, *node);
	return record == nullptr ? std::nullopt : record->key_presses;
}

Element ClaimDpad(UiFrame& frame, Element element, DpadClaim claim, ElementError* reason) {
	if (RefusedInsideRender(frame, reason)) {
		return {};
	}
	InteractData* record = CallbackTarget(frame, element, reason);
	if (record == nullptr) {
		return {};
	}
	// Only this field changes, so the disabled flag, the keys and the rest stay as they were set.
	record->props.dpad = claim;
	return element;
}

Element AttachActivate(UiFrame& frame, Element element, ActivateInvoke invoke, const void* bytes,
                       size_t size, size_t alignment, ElementError* reason) {
	InteractData* record = CallbackTarget(frame, element, reason);
	if (record == nullptr) {
		return {};
	}
	const uint32_t offset = StoreCallable(frame, bytes, size, alignment);
	if (offset == NO_CALLABLE) {
		return Refuse(reason, FrameInternals::BuildRefusal(frame).value_or(ElementError::None));
	}
	record->on_activate = {invoke, offset};
	return element;
}

Element AttachInput(UiFrame& frame, Element element, InputInvoke invoke, const void* bytes,
                    size_t size, size_t alignment, ElementError* reason) {
	InteractData* record = CallbackTarget(frame, element, reason);
	if (record == nullptr) {
		return {};
	}
	const uint32_t offset = StoreCallable(frame, bytes, size, alignment);
	if (offset == NO_CALLABLE) {
		return Refuse(reason, FrameInternals::BuildRefusal(frame).value_or(ElementError::None));
	}
	record->on_input = {invoke, offset};
	return element;
}

Element AttachAfterInteract(UiFrame& frame, Element element, AfterInteractInvoke invoke,
                            const void* bytes, size_t size, size_t alignment,
                            ElementError* reason) {
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return Refuse(reason, FrameInternals::Check(frame, element).value_or(ElementError::None));
	}
	// Inside Render this stores nothing, which is the refusal.
	const uint32_t offset = StoreCallable(frame, bytes, size, alignment);
	if (offset == NO_CALLABLE) {
		return Refuse(reason, FrameInternals::BuildRefusal(frame).value_or(ElementError::None));
	}
	InteractRecord(frame, *node).after_interact = {invoke, offset};
	return element;
}

Element AsScrollingText(UiFrame& frame, Element element, ElementError* reason) {
	if (AsList(frame, element, reason).IsNone()) {
		return {};
	}
	InteractRecord(frame, *FrameInternals::Find(frame, element)).list.pad_scrolls = true;
	return element;
}

Element AsList(UiFrame& frame, Element element, ElementError* reason) {
	if (RefusedInsideRender(frame, reason)) {
		return {};
	}
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return Refuse(reason, FrameInternals::Check(frame, element).value_or(ElementError::None));
	}
	const bool clips = node->kind == NodeKind::Box &&
	                   FrameInternals::Boxes(frame)[node->payload].style.overflow == Overflow::Clip;
	if (!clips) {
		return Refuse(reason, ElementError::NotClippingBox);
	}
	InteractData& record = InteractRecord(frame, *node);
	record.list.is_list  = true;
	return element;
}

Element Specimen(UiFrame& frame, Element element, const ItemVisuals& pose, ElementError* reason) {
	if (RefusedInsideRender(frame, reason)) {
		return {};
	}
	Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return Refuse(reason, FrameInternals::Check(frame, element).value_or(ElementError::None));
	}
	InteractData& record = InteractRecord(frame, *node);
	record.posed         = true;
	record.pose          = pose;
	return element;
}

namespace {

// The hook Render runs after the walk, as the template stored it.
struct HookSpec {
	AfterInteractInvoke invoke;
	const void*         bytes;
	size_t              size;
	size_t              alignment;
};

// Stores the hook's bytes and runs it in the mode where nodes can't be added. The bytes are
// stored first and read from the frame, as the other callbacks are, so the hook sees the same
// kind of copy and nothing about the caller's object matters once Render has begun.
void RunAfterInteract(UiFrame& frame, Element root, const HookSpec& hook) {
	const uint32_t offset = StoreCallable(frame, hook.bytes, hook.size, hook.alignment);
	FrameInternals::SetInHook(frame, true);
	hook.invoke(StoredBytes(frame, offset), frame, root);
	FrameInternals::SetInHook(frame, false);
}

// Runs the AfterInteract callbacks of `element` and everything below it, children first, in the
// mode where nodes can't be added, so the nodes and records this walks through stay where they are.
void RunNodeCallbacks(UiFrame& frame, Element element) {
	const Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return;
	}
	for (const Element child: FrameInternals::Children(frame, *node)) {
		RunNodeCallbacks(frame, child);
	}
	const InteractData* record = RecordOf(frame, *node);
	if (record == nullptr || record->after_interact.invoke == nullptr) {
		return;
	}
	const StoredCallback<AfterInteractInvoke> callback = record->after_interact;
	FrameInternals::SetInHook(frame, true);
	callback.invoke(StoredBytes(frame, callback.offset), frame, element);
	FrameInternals::SetInHook(frame, false);
}

// Render with an optional hook between the walk and the paint.
RenderResult RenderTree(UiFrame& frame, Element root, ImDrawList& list, const PaintState& state,
                        const HookSpec* hook) {
	// A Render inside the hook or a callback would store a hook of its own in the storage this one
	// runs from, paint out of order and end the mode early.
	FrameInternals::AssertNotInCallbackOrHook(frame);
	if (const auto why = FrameInternals::BuildRefusal(frame); why.has_value()) {
		(void)ReportRefusal(*why);
		return {{}, why};
	}
	// A second Render would register every control again, which ImGui reports as an id conflict
	// and which doubles navigation candidates and store updates.
	auto& rendered = FrameInternals::RenderedRoots(frame);
	if (std::find(rendered.begin(), rendered.end(), root) != rendered.end()) {
		(void)ReportRefusal(ElementError::AlreadyRendered);
		return {{}, ElementError::AlreadyRendered};
	}
	if (const auto error = AssignIds(frame, root, CurrentIdSeed()); error.has_value()) {
		return {{}, error};
	}
	RenderResult    result;
	InteractContext context {frame.Ui(), frame.Store(), list, frame.Now()};
	rendered.push_back(root);
	InteractWalk(frame, context, root, {.result = &result});
	RunNodeCallbacks(frame, root);
	if (hook != nullptr) {
		RunAfterInteract(frame, root, *hook);
	}
	Paint(frame, root, list, state);
	return result;
}

} // namespace

RenderResult Render(UiFrame& frame, Element root, ImDrawList& list, const PaintState& state) {
	return RenderTree(frame, root, list, state, nullptr);
}

RenderResult RenderWithHook(UiFrame& frame, Element root, ImDrawList& list, const PaintState& state,
                            AfterInteractInvoke invoke, const void* bytes, size_t size,
                            size_t alignment) {
	const HookSpec hook {invoke, bytes, size, alignment};
	return RenderTree(frame, root, list, state, &hook);
}

std::optional<NodeInput> InputOf(const UiFrame& frame, Element element) {
	const Node* node = FrameInternals::Find(frame, element);
	if (node == nullptr) {
		return std::nullopt;
	}
	const InteractData* record = RecordOf(frame, *node);
	return record == nullptr ? std::nullopt : record->input;
}

} // namespace Libs::Graphics::SystemUi
