#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_INTERACT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_INTERACT_H_

#include "graphics/presentation/system/element/element.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/primitives.h"

#include <cstddef>
#include <memory>
#include <new>
#include <optional>
#include <type_traits>

struct ImDrawList;

namespace Libs::Graphics::SystemUi {

/// Makes `element` own the key groups in `keys` while Render walks it, and returns the same
/// element, or a none handle when it is refused, with `reason` set if given. A stale or none handle
/// is refused as for Box. Unlike MakeInteractive it works on a container, since owning keys is not
/// taking input and is not nested-interactive, so a tab bar or a dialog can own the shoulders and
/// back while its tabs and buttons are interactive. Render claims the groups under the node's path
/// id with ClaimKeys every frame, in tree order, so a Control claim that comes first keeps the
/// group and a Modal claim always takes it. It reads the presses as that owner, and KeyPressesOf
/// reports them. The gallery and the components read as NO_OWNER, so they stand down. A second call
/// replaces the first, and so does MakeInteractive with its own `keys`.
Element OwnKeys(UiFrame& frame, Element element, const KeyOwner& keys,
                ElementError* reason = nullptr);

/// What `element` read of the key groups it owns this frame, or nothing when it owns none, is a
/// stale or none handle, or Render has not run. Interactive nodes find the same in InputOf.
[[nodiscard]] std::optional<KeyPresses> KeyPressesOf(const UiFrame& frame, Element element);

/// Registers a hand-made control covering `rect` with ImGui under `id`, in the window being drawn
/// into, and returns what the player is doing to it. It is for a control that is not an element,
/// such as a test's stand-in or a custom one, and follows ImGui's buttons: it activates on press
/// for the gamepad and on release for the mouse, a disabled one is skipped by navigation and never
/// activates, and `rect` is in screen pixels after the window's scrolling. Call it once a frame for
/// each control, after `frame` is begun, and between a ModalScope's construction and its end to
/// register a control inside the modal. `id` is not checked against the ids of the frame's
/// elements, nor against another call's, which ImGui reports as a conflict, so derive it from your
/// own seed, such as IdFor under an IdScope. A frame that has not been begun is not refused here,
/// since nothing of the frame is read but its theme, store and time, so begin it first.
[[nodiscard]] Interaction Interact(UiFrame& frame, ItemId id, const Rect& rect, bool disabled);

/// Makes `element` a control that takes the player's input, as props say, and returns the same
/// element, or a none handle when it is refused, with `reason` set if given. A stale or none
/// handle is refused as for Box. An interactive node inside another interactive node, or holding
/// one, is refused as NestedInteractive, since the two would fight over the same input. Set a key
/// on the element first when it needs a stable id, since Render gives it the id of its path. Its
/// state styles respond to what Render finds, through SetInput, so Render replaces any input set
/// by hand. A node that is already interactive is refused as AlreadyInteractive in every build,
/// since a second record would drop what the first set, such as its disabled flag. To change one
/// field afterwards, use the call for it, such as ClaimDpad.
Element MakeInteractive(UiFrame& frame, Element element, const InteractProps& props,
                        ElementError* reason = nullptr);

/// Changes only the D-pad claim of the interactive `element`, and returns the same element, or a
/// none handle when it is refused, with `reason` set if given. A stale or none handle is refused as
/// for Box, and a node that is not interactive as NotInteractive. Its disabled flag, keys and every
/// other setting are kept, and a disabled node still leaves the D-pad to navigation. This is how a
/// control built by a component, which has already called MakeInteractive, takes the D-pad.
Element ClaimDpad(UiFrame& frame, Element element, DpadClaim claim, ElementError* reason = nullptr);

/// Shows `element` and its subtree with the fixed highlights in `pose`, as InteractContext::pose
/// does for the older controls. The interactive nodes under it show the pose, take no input and
/// are skipped by navigation, so a gallery can show a control in every state. Returns the same
/// element, or a none handle when it is refused, with `reason` set if given.
Element Specimen(UiFrame& frame, Element element, const ItemVisuals& pose,
                 ElementError* reason = nullptr);

// Adds the callback whose bytes are `size` bytes at `bytes`, copied into the frame, and which
// `invoke` calls, to the interactive `element`. Only OnActivate and OnInput call these, and they
// have already checked the callable's type. A second call replaces the first.
Element AttachActivate(UiFrame& frame, Element element, ActivateInvoke invoke, const void* bytes,
                       size_t size, size_t alignment, ElementError* reason);
Element AttachInput(UiFrame& frame, Element element, InputInvoke invoke, const void* bytes,
                    size_t size, size_t alignment, ElementError* reason);

/// Calls `callback(element)` when Render finds `element` activated, which is on the frame it is
/// confirmed or clicked, once for each press. Render calls it right after the node interacts and
/// before anything is painted, so a callback that changes what the tree draws, such as flipping a
/// toggle's value, shows in the same frame's paint. Render still reports the activation through
/// RenderResult and InputOf as well. It returns the same element, or a none handle when it is
/// refused, with `reason` set if given. A stale or none handle is refused as for Box, and a node
/// that is not interactive as NotInteractive, so call MakeInteractive first. A second call
/// replaces the first.
///
/// A bound value belongs to the caller. A callback that changes one captures a pointer or reference
/// to it, as the older controls take `bool& on`, and the value must outlive the Render call. The
/// frame holds only the callable's bytes, copied into frame storage as for Custom, so `callback`
/// must be trivially copyable and trivially destructible and capture pointers or plain values,
/// never strings or vectors. A steady frame allocates nothing for it. A second call replaces the
/// callback but leaves the old one's bytes in frame storage until the next Begin. The callback may
/// change state the caller owns and call FocusControl, but must not build or change elements, since
/// that could move the storage it runs from or the children the walk is going through. Debug builds
/// assert on it, and in every build Box, Text, Custom, MakeInteractive, OwnKeys, ClaimDpad, AsList,
/// Specimen, OnActivate, OnInput, AfterInteract, Decorate, WithPose, WithAnimation, SetInput,
/// SetTextColor, Keyed, AssignIds and Render are refused with BuildingInCallback and change
/// nothing, and Begin does nothing. Measure and Arrange aren't refused, as they move nothing the
/// walk holds, but they leave that frame's hit rectangles and paint disagreeing, so don't call them
/// either. It runs on the render thread, once for each Render of the tree.
template <class F>
Element OnActivate(UiFrame& frame, Element element, F callback, ElementError* reason = nullptr) {
	static_assert(std::is_trivially_copyable_v<F> && std::is_trivially_destructible_v<F>,
	              "OnActivate stores its callable as raw bytes in the frame, so it must be "
	              "trivially copyable and trivially destructible. Capture pointers or plain "
	              "values, not strings, vectors or smart pointers.");
	static_assert(std::is_invocable_v<const F&, Element>,
	              "OnActivate calls its callable as a const object with the activated Element.");
	static_assert(alignof(F) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
	              "The frame's callable storage is aligned to the default new alignment, no more.");
	const ActivateInvoke invoke = [](const std::byte* callable, Element activated) {
		(*std::launder(reinterpret_cast<const F*>(callable)))(activated);
	};
	return AttachActivate(frame, element, invoke, std::addressof(callback), sizeof(F), alignof(F),
	                      reason);
}

/// Calls `callback(input)` with what Render found out about `element`, every frame the node
/// interacts and not only when it is activated, so a callback can read the D-pad presses or the
/// highlights. It is called right after the node interacts, before OnActivate's callback, with the
/// same input InputOf reports. Everything said of OnActivate about refusal, bound values, storage
/// and the callable's type holds here too, and so does the replaced callback's storage.
template <class F>
Element OnInput(UiFrame& frame, Element element, F callback, ElementError* reason = nullptr) {
	static_assert(std::is_trivially_copyable_v<F> && std::is_trivially_destructible_v<F>,
	              "OnInput stores its callable as raw bytes in the frame, so it must be "
	              "trivially copyable and trivially destructible. Capture pointers or plain "
	              "values, not strings, vectors or smart pointers.");
	static_assert(std::is_invocable_v<const F&, const NodeInput&>,
	              "OnInput calls its callable as a const object with a const NodeInput&.");
	static_assert(alignof(F) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
	              "The frame's callable storage is aligned to the default new alignment, no more.");
	const InputInvoke invoke = [](const std::byte* callable, const NodeInput& found) {
		(*std::launder(reinterpret_cast<const F*>(callable)))(found);
	};
	return AttachInput(frame, element, invoke, std::addressof(callback), sizeof(F), alignof(F),
	                   reason);
}

// The part of AfterInteract that stores the callable, once the template has checked its type.
Element AttachAfterInteract(UiFrame& frame, Element element, AfterInteractInvoke invoke,
                            const void* bytes, size_t size, size_t alignment,
                            ElementError* reason);

/// Calls `callback(frame, element)` once in each Render of the tree `element` is in, after the
/// whole tree has interacted and before anything is painted. Every interactive node's InputOf is
/// final by then, so a component can work out state from all its children's input, such as a tab
/// bar's chosen tab, and style them in the same frame with SetInput, SetTextColor, Decorate and
/// WithPose. Children run before their parents, so a node sees what the callbacks below it decided.
/// A node that layout collapsed has no input, as the walk passed it by, but its callback still runs
/// and finds InputOf empty, so that what it works out can be let go. It works on any node,
/// interactive or not, and a tree needs no more than this however many of them it holds. Returns
/// the same element, or a none handle when it is refused, with `reason` set if given. A stale or
/// none handle is refused as for Box. A second call replaces the first. Nothing calls it for a tree
/// painted without Render or when Render is refused. Everything said of OnActivate about the
/// callable's type and storage holds here, and what a callback may not do is what Render's
/// afterInteract hook may not do, as it runs in the same mode.
template <class F>
Element AfterInteract(UiFrame& frame, Element element, F callback, ElementError* reason = nullptr) {
	static_assert(std::is_trivially_copyable_v<F> && std::is_trivially_destructible_v<F>,
	              "AfterInteract stores its callable as raw bytes in the frame, so it must be "
	              "trivially copyable and trivially destructible. Capture pointers or plain "
	              "values, not strings, vectors or smart pointers.");
	static_assert(std::is_invocable_v<const F&, UiFrame&, Element>,
	              "AfterInteract calls its callable as a const object with the frame and the "
	              "element.");
	static_assert(alignof(F) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
	              "The frame's callable storage is aligned to the default new alignment, no more.");
	const AfterInteractInvoke invoke = [](const std::byte* callable, UiFrame& hooked,
	                                      Element node) {
		(*std::launder(reinterpret_cast<const F*>(callable)))(hooked, node);
	};
	return AttachAfterInteract(frame, element, invoke, std::addressof(callback), sizeof(F),
	                           alignof(F), reason);
}

/// Makes the clipping box `element` a list, which scrolls its children and remembers its place. In
/// Render, a list reads its scroll and the row that last had focus from the store under its id, so
/// give it a key when it needs a stable one. Its children sit higher by the scroll, both where
/// they are hit and where they are drawn, since the scroll is layout and not a pose. Its rows are
/// the interactive nodes below it, ideally its children, and each should be keyed by the data it
/// shows with Keyed, so that focus returns to the same data after rows are inserted or removed.
/// Focus coming back into the list from outside returns to the row that last had it. While focus
/// is outside, rows not wholly inside the list are left out of navigation, so the controls
/// around the list are not trapped by rows they cannot see. The list scrolls to keep the focused
/// row in view, less the box's padding at the top and bottom, and never past its last row. That
/// padding is also what the rows sit in from the edges by, so give the box a padding at least as
/// wide as the focus ring, and the ring is not cut. The larger of the top and bottom padding is
/// used. Children wholly scrolled out of view are not drawn, so an `unclipped` glow on such a row
/// does not show either. The list clips hit testing to its exact rectangle, as ListScope does and
/// unlike a plain clipping box, which clips it to the rectangle less its rounded corners, so a row
/// partly in view near a corner can still be hovered. That keeps hidden rows from taking the mouse.
/// Returns the same element, or a none handle when it is refused, with `reason` set if given. A
/// stale or none handle is refused as for Box, and a node that is not a box with Overflow::Clip as
/// NotClippingBox. A second call replaces the first.
Element AsList(UiFrame& frame, Element element, ElementError* reason = nullptr);

/// AsList for what has nothing in it to take focus, such as a long passage of text. With no focus
/// to follow, the D-pad's up and down scroll it while its window has focus, as well as the pointer,
/// so use it where nothing else in the window navigates up and down, as they would otherwise do
/// both. It starts at the top each time the window opens. It's refused as AsList is.
Element AsScrollingText(UiFrame& frame, Element element, ElementError* reason = nullptr);

/// What Render found.
struct RenderResult {
	/// The first interactive node, in tree order, that was activated this frame, and none if no
	/// node was. Only the first is reported, so a tree where two nodes can activate on one frame
	/// reads each node's activation from InputOf.
	Element activated;
	/// Why the tree was not drawn, or nothing. A failed Render draws and registers nothing.
	std::optional<ElementError> error;
};

/// Interacts with the tree of `root` and draws it into `list`, inside the current ImGui window.
/// Call it once the tree has been measured and arranged, with the arranged rectangle in screen
/// pixels after the window's scrolling, as Interact wants its rectangle. Call it once for each
/// root in a frame, and a second call for the same root returns AlreadyRendered and does nothing.
/// Calling AssignIds first is fine. In order it gives the tree its ids from
/// the window's id stack, as AssignIds does, and returns that error if it fails. It then walks the
/// tree in order, and for each interactive node registers a control with ImGui under the node's id
/// over its arranged rectangle, the layout one and not the posed one, claims the D-pad and the
/// triggers if asked, and sets the node's input from what the player is doing. The interaction
/// comes first so the drawing shows this frame's highlights. Right after a node interacts, its
/// OnInput and OnActivate callbacks run, so they too come before any paint. The node then passes
/// its input down its subtree, so the state styles of the boxes below it that have no input of
/// their own respond to it, and a button's label can change colour on hover. Last it paints with
/// Paint, which is unchanged. `state` is passed on to it.
RenderResult Render(UiFrame& frame, Element root, ImDrawList& list, const PaintState& state = {});

// The part of Render's overload that takes a hook, once the template has checked the callable's
// type. It copies the bytes into the frame as the other callbacks do.
RenderResult RenderWithHook(UiFrame& frame, Element root, ImDrawList& list, const PaintState& state,
                            AfterInteractInvoke invoke, const void* bytes, size_t size,
                            size_t alignment);

/// Render with `afterInteract` called once, with the frame and `root`, after the whole tree has
/// interacted, after any AfterInteract callbacks in the tree, and before anything is painted. It
/// suits a caller that holds the whole tree, and a component uses AfterInteract instead, so that
/// whoever renders it has nothing to pass on. It may style nodes as an AfterInteract callback may.
/// Unlike OnActivate's callback it runs outside callback mode, so those calls are allowed, but it
/// must not build nodes or records, with Box, Text, MakeInteractive, OnActivate and the like. Debug
/// builds assert on it. In every build Box, Text, Custom, MakeInteractive, OwnKeys, AsList,
/// Specimen, OnActivate, OnInput and AfterInteract there are refused with BuildingInHook and change
/// nothing, as is Render, and Begin does nothing. Everything said of OnActivate about the
/// callable's type and storage holds here. It is not called when Render is refused. It is an
/// overload, not an options struct, so the callable's type is deduced and every existing call stays
/// as it is.
template <class F>
RenderResult Render(UiFrame& frame, Element root, ImDrawList& list, const PaintState& state,
                    F afterInteract) {
	static_assert(std::is_trivially_copyable_v<F> && std::is_trivially_destructible_v<F>,
	              "Render stores its hook as raw bytes in the frame, so it must be trivially "
	              "copyable and trivially destructible. Capture pointers or plain values.");
	static_assert(std::is_invocable_v<const F&, UiFrame&, Element>,
	              "Render calls its hook as a const object with the frame and the root.");
	static_assert(alignof(F) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
	              "The frame's callable storage is aligned to the default new alignment, no more.");
	const AfterInteractInvoke invoke = [](const std::byte* callable, UiFrame& hooked,
	                                      Element tree) {
		(*std::launder(reinterpret_cast<const F*>(callable)))(hooked, tree);
	};
	return RenderWithHook(frame, root, list, state, invoke, std::addressof(afterInteract),
	                      sizeof(F), alignof(F));
}

/// What Render found out about `element` this frame, including whether it was activated, for every
/// interactive node and not just the first. It is nothing for a node that is not
/// interactive, a stale or none handle, and any node before Render ran.
[[nodiscard]] std::optional<NodeInput> InputOf(const UiFrame& frame, Element element);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_INTERACT_H_
