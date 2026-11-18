#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_INTERACTCONTEXT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_INTERACTCONTEXT_H_

// INTERNAL to the element layer and its tests. Besides InteractContext it holds the immediate-mode
// helpers only the element layer drives (list recording and finishing, hit clips, D-pad and trigger
// claims, the input mode), which callers reach through elements. Callers hold a UiFrame and pass
// that, never an InteractContext: the frame carries the theme, the scale, the fonts, the store and the time, and
// the draw list is the current window's. The element layer builds an InteractContext from a frame
// where it registers a control with ImGui, and the tests that register hand-made controls do the
// same through Interact(UiFrame&, ...) in element/interact.h.
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include <optional>
#include <span>
#include <vector>

struct ImDrawList;

namespace Libs::Graphics::SystemUi {

// One row of a list as it was drawn.
struct RecordedRow {
	ItemId id = 0;
	Rect   rect;
	bool   disabled = false;
};

// The rows of one list drawn this frame, where each is, and the list's visible area. While focus
// is outside the list, rows not wholly inside the area are left out of navigation, since ImGui
// scores them at their real positions, so moving into the list lands on a row in view and hidden
// rows don't trap the controls around it.
struct ListRows {
	std::vector<RecordedRow> rows;
	Rect                     area;
	bool                     focus_inside = false;
};

// What a control needs to take input and draw.
struct InteractContext {
	const UiContext&    ui;
	InteractionStore&   store;
	ImDrawList&         list;
	UiClock::time_point now;
	// Set while drawing a list's rows, which are recorded here rather than widening the window.
	ListRows* rows = nullptr;
	// Set while drawing specimens, such as a gallery's row of a control in each state, which show
	// these highlights whatever the input and can't be focused or pressed.
	const ItemVisuals* pose = nullptr;
};

/// Registers a control covering `rect` with ImGui under `id`, so gamepad navigation, the mouse and
/// focus reach it, and returns what the player is doing to it. Call it inside an ImGui window, once
/// a frame for each control, with `rect` in screen pixels after the window's scrolling, so a
/// control below the window's edge makes the window scrollable and focus can scroll to it. A
/// disabled control is skipped by navigation and never activates.
/// Activation follows ImGui's buttons, on press for the gamepad and on release for the mouse.
/// With the context's pose set, the control shows the pose, navigation skips it, and it is never
/// hovered, held or activated. The pose shows even on a disabled control. It is for specimens with
/// their own names, since a control posed while it already has focus keeps that focus, and a
/// toggle or slider would still take the D-pad.
[[nodiscard]] Interaction Interact(InteractContext& context, ItemId id, const Rect& rect,
                                   bool disabled);

/// Narrows the current window's clip rectangle to `rect`, so the controls registered until the
/// matching PopHitClip can only be hovered and pressed inside it, as controls in a clipped area.
/// It intersects with the clip already in effect. Calls must be balanced.
void PushHitClip(const Rect& rect);

/// Ends the clip PushHitClip started.
void PopHitClip();

/// The input the player used last. It starts as Pad, turns to Pointer when the mouse moves or
/// clicks, and back to Pad on the next gamepad or keyboard navigation.
[[nodiscard]] InputMode CurrentInputMode();

/// While the control `id` has focus, takes the D-pad's left and right from navigation, so they
/// adjust the control instead of moving focus, and returns which were pressed this frame. With
/// `repeat`, a held direction repeats at the pace a held D-pad moves focus. Call it every frame
/// after Interact. A direction held while focus arrives stops there, so a held D-pad scanning a row
/// stops at each such control.
[[nodiscard]] HorizontalPresses ClaimHorizontal(ItemId id, bool repeat);

/// Gives back to navigation any of the D-pad's left and right that the control `id` still owns,
/// such as when it becomes disabled while a direction is held. ClaimHorizontal does this itself
/// once the control loses focus.
void ReleaseHorizontal(ItemId id);

/// While the control `id` has focus, whether the gamepad's L2 (`left`) and R2 (`right`) triggers
/// were pressed this frame, repeating at the pace a held D-pad moves focus. Navigation doesn't
/// use them.
[[nodiscard]] HorizontalPresses ClaimTriggers(ItemId id);

/// Lets the pointer scroll a list, before its rows are drawn, so the scroll bar takes the pointer
/// ahead of a row beneath it. The wheel scrolls by `wheel_step` a notch while the pointer is over
/// `area` and the window can be hovered. Pressing the `hit` part of the bar's rectangle takes the
/// thumb, at `thumb`, and it follows until released. A press off the thumb jumps it there. It never
/// takes focus. Does nothing when `thumb` is empty, as the rows all fit.
void ScrollListWithPointer(InteractionStore::ListMemory& memory, ItemId list, const Rect& area,
                           const std::optional<Rect>& thumb, float hit_width, float wheel_step);

/// Lets the D-pad's up and down, or the arrow keys, scroll a list while its window has focus, by
/// `step` a press, repeating while held, and go back to the top when the window opens afresh. For a
/// list with no rows to focus, such as a dialog's long body, in a window with nothing else to
/// navigate up and down, as the D-pad would otherwise move focus and scroll together. It scrolls
/// nothing when the list's `memory.content` all fits in `area`.
void ScrollListWithPad(InteractionStore::ListMemory& memory, const Rect& area, float step);

/// Finishes drawing the list `list`, whose `rows` lie in `area`. When the D-pad moves focus into
/// the list from outside onto another row, focus is sent back to the row that last had it, if
/// that row is still there and enabled. A click or a deliberate focus change is left alone. While
/// a row has focus, the list scrolls to keep it inside `area`, less `inset` at the top and bottom
/// for the focus ring, and the scroll never passes the last row. Focus sent back reaches its row
/// on the next frame.
void FinishList(InteractionStore& store, ItemId list, const ListRows& rows, const Rect& area,
                float inset);

/// The same for rows kept anywhere else, such as in a frame's scratch. `at_least` is how tall the
/// content is known to be apart from its rows, for a list with none to measure.
void FinishList(InteractionStore& store, ItemId list, std::span<const RecordedRow> rows,
                const Rect& area, float inset, float at_least = 0.0f);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_INTERACTCONTEXT_H_
