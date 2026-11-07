#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_PRIMITIVES_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_PRIMITIVES_H_

#include "graphics/presentation/system/animation/timeline.h"
#include "graphics/presentation/system/components/image.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/decor.h"
#include "graphics/presentation/system/element/element.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>

struct ImDrawList;
struct ImTextureData;

namespace Libs::Graphics::SystemUi {

/// A box with `children`, laid out by `layout` and drawn in `style`. Each child can have one
/// parent only. A refused box is a none handle, and `reason`, when given, says why. Call Measure,
/// then Arrange, then Paint on the root.
[[nodiscard]] Element Box(UiFrame& frame, const Layout& layout, const Style& style,
                          std::span<const Element> children = {}, ElementError* reason = nullptr);

/// A text node showing `props`, which is laid out where it is measured. The text is copied into
/// frame storage, so the view in `props` need only last until this call returns. Its natural size
/// is the text with no width limit, so ellipsis and clipped text keep their full width and wrapped
/// text breaks only at newlines. Given a width, it ellipsizes, wraps up to `max_lines` or shrinks
/// to fit, as its props say. Its automatic minimum width, where `item` sets no `min_main`, depends
/// on what it does when short of room. Wrapped text stops at its longest word. Ellipsis text,
/// which truncates, stops at the width of its ellipsis, so shrink to fit can still shrink the
/// font first. Clipped text has no minimum. Its minimum height is its laid-out height.
[[nodiscard]] Element Text(UiFrame& frame, const TextProps& props, const ItemLayout& item = {},
                           ElementError* reason = nullptr);

/// An image node showing `texture`, which is `texture_size` pixels, fitted to its rectangle as
/// `fit` says with corners rounded by `radius`. An explicit width and height in `item` are used as
/// given. With only one the other follows the texture's shape, and with neither the image measures
/// zero. A null texture draws nothing.
[[nodiscard]] Element Image(UiFrame& frame, ImTextureData* texture, Vec2 texture_size, ImageFit fit,
                            Dp radius, const ItemLayout& item = {}, ElementError* reason = nullptr);

/// Measures `element` and its subtree and returns the size it takes in output pixels. `max` is the
/// space on offer, and infinity on an axis means unbounded. Children are measured at their natural
/// size first, then given the lengths the flex solver gives them, growing or shrinking to fit, and
/// measured again at those lengths. The result is not clamped to `max`. A stale or none handle
/// measures to nothing.
Vec2 Measure(UiFrame& frame, Element element, Vec2 max);

/// Places `element` at `rect` and its children inside it, from the sizes Measure found. Returns
/// false for a stale or none handle anywhere in the subtree, and for a node Measure never reached.
/// Children are laid out from their natural lengths and the space `rect` gives, whatever space
/// Measure was offered, so measuring with or without a limit arranges the same.
bool Arrange(UiFrame& frame, Element element, const Rect& rect);

/// A clip rectangle as ImGui holds it, by its two corners.
struct ClipBounds {
	Vec2 min;
	Vec2 max;
};

/// Maps a point x of the layout to scale * x + translate on the output. A pose is a scale about a
/// point and an offset, and composing poses down the tree keeps that form, so a subtree carries
/// one of these instead of a list of poses.
struct PaintTransform {
	float scale = 1.0f;
	Vec2  translate;
};

// What a paint pass carries down the tree. Opacity multiplies down, and so do poses. Clip joins it
// later, so call sites pass the state on instead of a growing list of arguments.
struct PaintState {
	float          opacity = 1.0f;
	PaintTransform transform;
	// The clip that was in effect just outside the nearest clipping ancestor, which an unclipped
	// node returns to. Empty when there is no clipping ancestor.
	std::optional<ClipBounds> outer_clip;
};

/// Draws `element` and its subtree, which must have been arranged. `state.opacity` fades all of it.
void Paint(const UiFrame& frame, Element element, ImDrawList& list, const PaintState& state = {});

/// Where Arrange put `element`, or an empty rectangle for a stale or none handle.
[[nodiscard]] Rect Arranged(const UiFrame& frame, Element element);

/// The size the root `element` measured to when Measure last gave it exactly the room `limit`, or
/// nothing if it was never measured or was measured in other room, or is a stale or none handle. It
/// is for a caller that would measure the same root in the same room again and can reuse the
/// answer. It applies only to roots: a node with a parent always gets nothing, since the parent's
/// layout may have resized it since without changing the room it was measured in. Arrange and later
/// passes don't clear it.
[[nodiscard]] std::optional<Vec2> MeasuredAt(const UiFrame& frame, Element element, Vec2 limit);

/// Poses `element` and its whole subtree and returns the same element, or a none handle when it
/// is refused, with `reason` set if given. A stale or none handle is refused as for Box, and so is
/// a pose whose scale is not positive and finite, whose offset is not finite or whose alpha is
/// NaN, as InvalidPose. An alpha outside 0 to 1 is clamped to it for each element. The pose
/// maps a point x of the layout to
/// O + k * (x - O) + o, where O is the `origin` point of the element's arranged rectangle, k the
/// pose scale and o the offset in reference pixels, rounded with Px. A parent's pose applies to its
/// children after theirs, so its scale shrinks their offsets too. The pose alpha multiplies into
/// the opacity of the subtree. The pose is paint-only, so Measure and Arrange never see it. A
/// default ElementPose changes nothing. A second call replaces the first.
Element WithPose(UiFrame& frame, Element element, const ElementPose& pose, Origin origin = CENTER,
                 ElementError* reason = nullptr);

/// Plays a timeline on `element`, which is keyed by its path id, and returns the same element, or
/// a none handle when it is refused, with `reason` set if given. A stale or none handle is refused
/// as for Box, and so is an origin that is not finite or a timeline whose scale is not positive and
/// finite or whose slide is not finite, as InvalidPose. Render evaluates it at the frame's time
/// against the start and dismissal kept in the store under the element's id, and applies the pose
/// inside any pose from WithPose, so the animation moves in the element's own space and the
/// explicit pose then moves the result. Set a key first when rows can be reordered, since an index
/// path would hand one row's start to the next. A timeline changed under the same id keeps its
/// start and continues from it, so a change mid-entrance jumps to the new curve's position. The
/// first frame `dismissed` is true starts the exit once from wherever the element is, and holding
/// it does not restart it. Setting it false again after a dismissal, even once the element is
/// Gone, restarts the entrance from that frame. That includes undismissing in the middle of the
/// exit, where a half-faded element snaps to its first entering pose and fades in again, as the
/// pose is not blended from where it was. Timing comes from the frame's time alone. A node
/// that is missing for a frame is forgotten by the store and replays its entrance. Nothing is
/// applied without Render, and a second call replaces the first.
Element WithAnimation(UiFrame& frame, Element element, const Animate& animate,
                      ElementError* reason = nullptr);

/// The phase of the keyed animation of `element` as Render found it this frame, or nothing when
/// the element has none, was refused, or Render has not run.
[[nodiscard]] std::optional<Phase> PhaseOf(const UiFrame& frame, Element element);

/// Gives `element` the effects in `decor` and returns the same element, or a none handle when it
/// is refused, with `reason` set if given. A stale or none handle is refused as for Box, and a NaN
/// shine or ring strength as InvalidDecor. The effects live in the same record as the pose and the
/// input, so neither WithPose nor SetInput undoes this, and a second call replaces the first's
/// effects.
Element Decorate(UiFrame& frame, Element element, const Decor& decor,
                 ElementError* reason = nullptr);

/// The same for a pose from a timeline, which is converted as it is, alpha included. It has its
/// own name because a braced pose such as {.scale = 0.5f} would fit both pose types.
Element WithTimelinePose(UiFrame& frame, Element element, const Pose& pose, Origin origin = CENTER,
                         ElementError* reason = nullptr);

/// Sets what `element`'s state styles respond to, and nothing else. A NaN visual clamps to 0
/// through ClampUnit, so it rests. Returns false when the handle
/// is none, stale or out of range, with `reason` set if given. It leaves the pose and the effects
/// alone, and Decorate and WithPose leave the input alone, so Interact can fill it after Arrange
/// whatever was decorated before. A second call replaces the first. A node that never gets one
/// rests. An input set here wins over the one Render passes down from an interactive ancestor.
bool SetInput(UiFrame& frame, Element element, const ItemInput& input,
              ElementError* reason = nullptr);

/// Overrides the colour of the text node `element` for this frame's paint, and nothing else, so
/// a component can restyle a label once it knows its own state, such as a tab bar colouring the
/// chosen tab's label after Render's afterInteract hook has read every tab's input. The frame's
/// nodes are rebuilt every frame, so the colour lasts until the next Begin. Returns false when
/// the handle is none, stale or out of range, with `reason` set if given, and NotText when the
/// node is not a text node. A second call replaces the first.
bool SetTextColor(UiFrame& frame, Element element, Color color, ElementError* reason = nullptr);

/// Where Paint draws `element`, in output pixels, with every pose of its ancestors and its own
/// applied after `root`, and its place moved up by the scroll of every list above it, as Render
/// last found it. The corner is rounded to a whole pixel only when the total scale is 1,
/// so a resting element is sharp and a scaling one moves smoothly. Text keeps its laid-out size
/// and moves with its centre, so for text this is the arranged size at the moved centre. That
/// differs from the rectangle Paint draws when the text is taller than its rectangle, since the
/// drawn text then starts at the top and spills below. An empty rectangle for a stale or none
/// handle.
[[nodiscard]] Rect Painted(const UiFrame& frame, Element element, const PaintTransform& root = {});

/// Names `element` among its siblings with `key`, instead of its position, and returns the same
/// element, or a none handle when it is refused, with `reason` set if given. A stale or none
/// handle is refused as for Box, an empty text key as InvalidKey, and an element that already has
/// a parent as AlreadyParented, since a key set after adoption could repeat a sibling's. Set the
/// key before the element goes into a Box, which refuses children that repeat a key with
/// DuplicateKey. A key makes the element's id follow the data it shows, so reordering siblings
/// keeps each one's id, and with it its focus and animation state. The text is copied into frame
/// storage. A text key and a number key never match, and a second call replaces the first.
Element Keyed(UiFrame& frame, Element element, std::string_view key,
              ElementError* reason = nullptr);
Element Keyed(UiFrame& frame, Element element, uint32_t key, ElementError* reason = nullptr);

/// Gives `root` and everything below it an id, top down. A node's id is the ImGui id of its key,
/// or of its index among its siblings when it has none, seeded with its parent's id. The root has
/// index 0 and is seeded with `seed`, normally CurrentIdSeed(). That is what nested PushID calls
/// would give, so a component called under PushOverrideID(IdOf(node)) makes the ids it always
/// did. Call it once the tree is built, since ids depend on the whole path.
///
/// It returns nothing on success. It returns IdCollision, and leaves every node of the tree with
/// id 0, when two siblings would get the same id, or when another tree given ids this frame has
/// the same root id. The ids decide and not the raw keys, so it catches a number key equal to an
/// index, keys that differ as text but hash alike such as "a###x" and "b###x", and real hash
/// collisions, none of which the DuplicateKey check on Box can see. A stale or none root returns
/// its reason, and a node that already has a parent is refused as AlreadyParented, since it would
/// get other ids as a root than it has in its tree. The same root can be given ids again in a
/// frame and gets the same ids. When re-assigning a root fails, its old record is dropped and its
/// ids are cleared, so the root ends up with no ids and no record. A different tree in the same
/// window needs a key on its root, or a different id scope with a different seed, or its root would
/// share an id with the first and be refused.
[[nodiscard]] std::optional<ElementError> AssignIds(UiFrame& frame, Element root, ItemId seed);

/// The id AssignIds gave `element`, or 0 for a stale or none handle and for one never assigned.
[[nodiscard]] ItemId IdOf(const UiFrame& frame, Element element);

/// What a Custom node's callable is given when the node is painted.
struct CustomPaint {
	/// The list to draw into, already inside the clips of the node's ancestors.
	ImDrawList& list;
	/// Where the node is drawn in output pixels, which is its arranged rectangle with its own pose
	/// and every ancestor's applied, as Painted reports it.
	Rect painted;
	/// The transform from layout space to the output, for a callable that draws more than the one
	/// rectangle and needs to follow the pose.
	PaintTransform transform;
	/// The opacity the node inherits, its ancestors' and its own multiplied. The callable should
	/// scale the alpha of everything it draws by it.
	float opacity;
	/// The frame, for its theme and scale.
	const UiFrame& frame;
	/// The node's path id, which Render or AssignIds gave it, and 0 before that. A callable that
	/// keeps state in the frame's InteractionStore, such as a knob's tween, keys it by this, which
	/// is stable while the node keeps its key and its place in the tree.
	ItemId id;
	/// What the nearest interactive ancestor found this frame, as its boxes' state styles see it,
	/// or a resting input for a node with none. A callable that draws a state itself, such as a
	/// focus ring around a part that moves, reads the highlights here.
	ItemInput input;
};

// Adds a custom node whose callable is `size` bytes at `bytes`, which are copied into the frame,
// and which `invoke` calls. Only Custom calls this, and it has already checked the callable's
// type.
Element AddCustom(UiFrame& frame, const ItemLayout& item, CustomInvoke invoke, const void* bytes,
                  size_t size, size_t alignment, ElementError* reason);

/// A node that draws itself with `paint`, called as `paint(const CustomPaint&)` when the tree is
/// painted. It is the only sanctioned place for raw ImGui drawing outside the imgui folder, so
/// anything the other nodes can't draw goes here. Its size comes from `item` alone. Measure gives
/// it the width and height `item` sets and 0 on an axis it leaves unset, and its min-content is
/// the same, so a Custom node that should take room needs a width and height, or a grow in its
/// row. It has no children. `paint` is copied into frame storage, so it must be trivially copyable
/// and trivially destructible, which excludes captured strings and vectors. Capture pointers or
/// views to what outlives the frame, or values. Its copy is aligned, and a steady frame allocates
/// nothing for it. The callable must not keep `CustomPaint::list` or `frame` after the call, since
/// they belong to the paint pass. The call happens inside the clips of its ancestors, once for each
/// Paint of the tree. A refused node, with a null handle and `reason` set if given, is never
/// called. A pose and decor given to the node apply as for any other, and its callable draws only
/// itself.
template <class F>
Element Custom(UiFrame& frame, const ItemLayout& item, F paint, ElementError* reason = nullptr) {
	static_assert(std::is_trivially_copyable_v<F> && std::is_trivially_destructible_v<F>,
	              "Custom stores its callable as raw bytes in the frame, so it must be trivially "
	              "copyable and trivially destructible. Capture pointers or plain values, not "
	              "strings, vectors or smart pointers.");
	static_assert(std::is_invocable_v<const F&, const CustomPaint&>,
	              "Custom calls its callable as a const object with a const CustomPaint&.");
	static_assert(alignof(F) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__,
	              "Custom's storage is aligned to the default new alignment, no more.");
	const CustomInvoke invoke = [](const std::byte* callable, const CustomPaint& painting) {
		(*std::launder(reinterpret_cast<const F*>(callable)))(painting);
	};
	return AddCustom(frame, item, invoke, std::addressof(paint), sizeof(F), alignof(F), reason);
}

/// A node that takes up the room `item` gives it and draws nothing, for holding a row or column to
/// a least width or height without moving anything in it, since a node with only one length takes
/// 0 on the other. Like any Custom node it has no children. Returns a none handle when it is
/// refused, with `reason` set if given.
[[nodiscard]] Element Spacer(UiFrame& frame, const ItemLayout& item,
                             ElementError* reason = nullptr);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_PRIMITIVES_H_
