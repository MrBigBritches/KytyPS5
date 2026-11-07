#include "graphics/presentation/system/imgui/interaction.h"

#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/imgui/interactContext.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <array>
#include <initializer_list>

namespace Libs::Graphics::SystemUi {

namespace {

constexpr float MS_PER_SECOND = 1000.0f;
// ImGui moves focus faster than its key repeat settings, at these shares of the delay and the
// interval (GetTypematicRepeatRate, ImGuiInputFlags_RepeatRateNavMove). Dividing by them makes a
// held D-pad move focus at exactly the theme's pace.
constexpr float NAV_MOVE_DELAY_SHARE = 0.72f;
constexpr float NAV_MOVE_RATE_SHARE  = 0.80f;
// Repeats a held key at the pace a held D-pad moves focus, the theme's own pace.
constexpr ImGuiInputFlags NAV_REPEAT =
    ImGuiInputFlags_Repeat | static_cast<ImGuiInputFlags>(ImGuiInputFlags_RepeatRateNavMove);

// The tracks each highlight moves on. Focus arrives with the enter curve and leaves faster with
// the exit curve. Hover and press ease the same both ways.
struct HighlightTracks {
	TweenTracks hover;
	TweenTracks focus;
	TweenTracks press;
};

// The highlight tracks from `theme`'s interaction tokens.
HighlightTracks Tracks(const Theme& theme) {
	const auto& tokens = theme.interaction;
	return {{{tokens.hover_ms, Ease::Standard}, {tokens.hover_ms, Ease::Standard}},
	        {{tokens.focus_in_ms, Ease::Enter}, {tokens.focus_out_ms, Ease::Exit}},
	        {{tokens.press_ms, Ease::Standard}, {tokens.press_ms, Ease::Standard}}};
}

// 1 when `on`, otherwise 0, as a tween target.
float Target(bool on) {
	return on ? 1.0f : 0.0f;
}

} // namespace

ItemId IdFor(std::string_view name) {
	IM_ASSERT(!name.empty() && "a control needs a name");
	// ImGui reads an empty range as a string ending in a zero, which a view doesn't have.
	constexpr std::string_view UNNAMED = "unnamed control";
	const std::string_view     used    = name.empty() ? UNNAMED : name;
	return ImGui::GetID(used.data(), used.data() + used.size());
}

ItemId IdWithSeed(std::string_view name, ItemId seed) {
	IM_ASSERT(!name.empty() && "an id needs a name");
	constexpr std::string_view UNNAMED = "unnamed control";
	const std::string_view     used    = name.empty() ? UNNAMED : name;
	return ImGui::GetIDWithSeed(used.data(), used.data() + used.size(), seed);
}

ItemId IdWithSeed(uint32_t number, ItemId seed) {
	return ImGui::GetIDWithSeed(static_cast<int>(number), seed);
}

ItemId CurrentIdSeed() {
	// Only reads it, as a toast is drawn with no window open, and marking the current window as
	// written to would make ImGui show its empty fallback Debug window.
	return ImGui::GetCurrentWindowRead()->IDStack.back();
}

void PushHitClip(const Rect& rect) {
	// As ImGui::PushClipRect does, but it marks the window as written to, and Render pushes this for
	// a toast's clipping box too, which would show ImGui's fallback window as CurrentIdSeed explains.
	ImGuiWindow* window = ImGui::GetCurrentWindowRead();
	window->DrawList->PushClipRect({rect.x, rect.y}, {rect.Right(), rect.Bottom()}, true);
	window->ClipRect = window->DrawList->_ClipRectStack.back();
}

void PopHitClip() {
	// As ImGui::PopClipRect does, for the same reason as PushHitClip.
	ImGuiWindow* window = ImGui::GetCurrentWindowRead();
	window->DrawList->PopClipRect();
	window->ClipRect = window->DrawList->_ClipRectStack.back();
}

Vec2 PointerPosition() {
	const ImVec2 position = ImGui::GetIO().MousePos;
	return {position.x, position.y};
}

bool HeldByPointer(ItemId id) {
	const ImGuiContext& g = *GImGui;
	return g.ActiveId == id && g.ActiveIdSource == ImGuiInputSource_Mouse;
}

IdScope::IdScope(std::string_view name) {
	const std::string_view used = name.empty() ? std::string_view {"unnamed scope"} : name;
	ImGui::PushID(used.data(), used.data() + used.size());
}

IdScope::~IdScope() {
	ImGui::PopID();
}

void MakeLastControlDefaultFocus() {
	ImGui::SetItemDefaultFocus();
}

void FocusControl(ItemId id) {
	ImGui::SetFocusID(id, ImGui::GetCurrentWindow());
}

void MoveFocus(ItemId id) {
	FocusControl(id);
	ImGui::SetNavCursorVisibleAfterMove();
}

ItemId FocusedControl() {
	return ImGui::GetFocusID();
}

NavExclusion::NavExclusion(bool excluded) {
	ImGui::PushItemFlag(ImGuiItemFlags_NoNav, excluded);
}

NavExclusion::~NavExclusion() {
	ImGui::PopItemFlag();
}

bool LeftMoveFindsNothing() {
	const ImGuiContext& g = *GImGui;
	return g.NavMoveSubmitted && g.NavMoveDir == ImGuiDir_Left && g.NavMoveResultLocal.ID == 0 &&
	       g.NavMoveResultLocalVisible.ID == 0 && g.NavMoveResultOther.ID == 0;
}

HorizontalPresses ShoulderPresses(ItemId owner) {
	return {ImGui::IsKeyPressed(ImGuiKey_GamepadL1, ImGuiInputFlags_None, owner),
	        ImGui::IsKeyPressed(ImGuiKey_GamepadR1, ImGuiInputFlags_None, owner)};
}

namespace {

static_assert(NO_OWNER == ImGuiKeyOwner_NoOwner && ANY_OWNER == ImGuiKeyOwner_Any,
              "the owner constants are ImGui's");

// The gamepad's cancel button as ImGui maps it, which confirm and back swapped moves.
ImGuiKey CancelKey() {
	return ImGui::GetIO().ConfigNavSwapGamepadButtons ? ImGuiKey_GamepadFaceDown
	                                                  : ImGuiKey_GamepadFaceRight;
}

// The keys of `group`.
std::array<ImGuiKey, 2> KeysOf(KeyGroup group) {
	if (group == KeyGroup::Shoulders) {
		return {ImGuiKey_GamepadL1, ImGuiKey_GamepadR1};
	}
	return {CancelKey(), ImGuiKey_Escape};
}

} // namespace

bool ClaimKeys(KeyGroup group, ItemId owner, ClaimPriority priority) {
	const auto keys = KeysOf(group);
	if (priority == ClaimPriority::Control) {
		// A live owner keeps its keys, so a control drawn later or further back can't steal them.
		for (const ImGuiKey key: keys) {
			const ImGuiID holder = ImGui::GetKeyOwner(key);
			if (holder != ImGuiKeyOwner_NoOwner && holder != owner) {
				return false;
			}
		}
	}
	for (const ImGuiKey key: keys) {
		// A key consumed this frame stays consumed, since SetKeyOwner would clear its lock.
		const bool locked = ImGui::GetKeyOwnerData(GImGui, key)->LockThisFrame;
		ImGui::SetKeyOwner(key, owner,
		                   locked ? ImGuiInputFlags_LockThisFrame : ImGuiInputFlags_None);
	}
	return true;
}

void ConsumeKey(KeyGroup group, ItemId actor) {
	for (const ImGuiKey key: KeysOf(group)) {
		ImGui::SetKeyOwner(key, actor, ImGuiInputFlags_LockThisFrame);
	}
}

bool FocusInCurrentWindow() {
	const ImGuiContext& g      = *GImGui;
	const ImGuiWindow*  window = g.CurrentWindow;
	return g.NavWindow != nullptr && window != nullptr &&
	       g.NavWindow->RootWindow == window->RootWindow;
}

InputMode CurrentInputMode() {
	const ImGuiContext& g = *GImGui;
	// Before the mouse has ever moved there is no position, so the UI starts in pad mode.
	return ImGui::IsMousePosValid() && !g.NavHighlightItemUnderNav ? InputMode::Pointer
	                                                               : InputMode::Pad;
}

ItemVisuals InteractionStore::Update(ItemId id, const ItemState& state, InputMode mode,
                                     const Theme& theme, UiClock::time_point now) {
	const auto tracks = Tracks(theme);
	auto&      entry  = m_entries[id];
	entry.seen        = true;
	// A disabled control shows no hover or press. It keeps its focus ring if it had focus when it
	// was disabled, since ImGui keeps focus on it, and the player must still see where focus is.
	// The control draws the ring as faded as the rest of it. Hover belongs to the pointer and the
	// ring to the pad.
	const bool live = !state.disabled;
	entry.hover.Retarget(Target(live && state.hovered && mode == InputMode::Pointer), tracks.hover,
	                     now);
	entry.focus.Retarget(Target(state.focused && mode == InputMode::Pad), tracks.focus, now);
	entry.press.Retarget(Target(live && state.held), tracks.press, now);
	return {entry.hover.Value(now), entry.focus.Value(now), entry.press.Value(now)};
}

float InteractionStore::Animate(ItemId id, float target, const TweenTracks& tracks,
                                UiClock::time_point now, bool whole_track) {
	auto& entry = m_entries[id];
	entry.seen  = true;
	if (!entry.value_started) {
		entry.value         = Tween(target);
		entry.value_started = true;
	}
	entry.value.Retarget(target, tracks, now, whole_track);
	return entry.value.Value(now);
}

Pose InteractionStore::Play(ItemId id, const Timeline& timeline, bool dismissed,
                            UiClock::time_point now) {
	auto& entry = m_entries[id];
	entry.seen  = true;
	if (!entry.played) {
		entry.played = true;
		entry.start  = now;
	}
	// A dismissal is recorded once, since moving it every frame would hold the exit at its start.
	// Taking it back starts the entrance again from now, as for a row hidden and shown again.
	if (dismissed && !entry.dismissed_at.has_value()) {
		entry.dismissed_at = now;
	} else if (!dismissed && entry.dismissed_at.has_value()) {
		entry.dismissed_at.reset();
		entry.start = now;
	}
	entry.timeline = timeline;
	return Evaluate(timeline, entry.start, entry.dismissed_at, now);
}

void InteractionStore::Collect() {
	for (auto entry = m_entries.begin(); entry != m_entries.end();) {
		if (entry->second.seen) {
			entry->second.seen = false;
			++entry;
		} else {
			entry = m_entries.erase(entry);
		}
	}
}

bool InteractionStore::Animating(UiClock::time_point now) const {
	for (const auto& [id, entry]: m_entries) {
		if (entry.hover.Moving(now) || entry.focus.Moving(now) || entry.press.Moving(now) ||
		    entry.value.Moving(now)) {
			return true;
		}
		// A timeline moves until it has settled, which is shown for good or gone.
		if (entry.played && Evaluate(entry.timeline, entry.start, entry.dismissed_at, now)
		                        .NeedsFrames(entry.timeline)) {
			return true;
		}
	}
	return false;
}

ImDrawList& CurrentDrawList() {
	return *ImGui::GetWindowDrawList();
}

Interaction Interact(InteractContext& context, ItemId id, const Rect& rect, bool disabled) {
	const ImRect   bounds {rect.x, rect.y, rect.Right(), rect.Bottom()};
	ImGuiItemFlags flags = disabled ? ImGuiItemFlags_Disabled : 0;
	ItemState      state;
	state.disabled = disabled;
	if (context.rows != nullptr) {
		// A list's row scrolls with its list, so it is recorded for the list instead.
		context.rows->rows.push_back({id, rect, disabled});
		const Rect& area = context.rows->area;
		// From outside, focus enters only on a row wholly in view.
		if (!context.rows->focus_inside && (rect.y < area.y || rect.Bottom() > area.Bottom())) {
			flags |= ImGuiItemFlags_NoNav;
		}
	} else {
		// The window's content reaches the control, so a window can scroll to one below its edge.
		ImGuiWindow* window     = ImGui::GetCurrentWindow();
		window->DC.CursorMaxPos = ImMax(window->DC.CursorMaxPos, bounds.Max);
	}
	if (context.pose != nullptr) {
		// A specimen is still registered, so its id is checked for duplicates, but takes no input.
		(void)ImGui::ItemAdd(bounds, id, nullptr, flags | ImGuiItemFlags_NoNav);
		return {state, *context.pose};
	}
	// A control outside the window's clip can still be reached by navigation, which scrolls to it,
	// but can't be hovered or pressed until it is in view.
	if (ImGui::ItemAdd(bounds, id, nullptr, flags)) {
		// ImGui never presses or hovers a disabled item. Without NoHoveredOnFocus it would report
		// the gamepad's focused item as hovered too.
		state.activated = ImGui::ButtonBehavior(bounds, id, &state.hovered, &state.held,
		                                        ImGuiButtonFlags_NoHoveredOnFocus);
		state.focused   = ImGui::GetFocusID() == id;
	}
	const auto visuals =
	    context.store.Update(id, state, CurrentInputMode(), context.ui.theme, context.now);
	return {state, visuals};
}

namespace {

// Whether `key` was pressed this frame for `id`, which owns it. On the frame focus arrives,
// navigation has already read the press as a move, so the move is cancelled and the control
// alone acts on it.
bool TakePress(ImGuiKey key, ImGuiDir direction, ItemId id, ImGuiInputFlags flags) {
	const bool pressed = ImGui::IsKeyPressed(key, flags, id);
	if (pressed && GImGui->NavMoveDir == direction) {
		ImGui::NavMoveRequestCancel();
	}
	return pressed;
}

} // namespace

void ReleaseHorizontal(ItemId id) {
	// A key held while a control loses focus stays owned until it is released, so give it back at
	// once, or the next control couldn't move focus with it.
	for (const ImGuiKey key: {ImGuiKey_GamepadDpadLeft, ImGuiKey_GamepadDpadRight}) {
		if (ImGui::GetKeyOwner(key) == id) {
			ImGui::SetKeyOwner(key, ImGuiKeyOwner_NoOwner);
		}
	}
}

HorizontalPresses ClaimHorizontal(ItemId id, bool repeat) {
	if (ImGui::GetFocusID() != id) {
		ReleaseHorizontal(id);
		return {};
	}
	// Owning the keys keeps them from navigation, which only reads keys nobody owns. Ownership
	// lasts into the next frame, so claiming them every frame while focused keeps them.
	ImGui::SetKeyOwner(ImGuiKey_GamepadDpadLeft, id);
	ImGui::SetKeyOwner(ImGuiKey_GamepadDpadRight, id);
	const ImGuiInputFlags flags = repeat ? NAV_REPEAT : ImGuiInputFlags_None;
	const bool            left  = TakePress(ImGuiKey_GamepadDpadLeft, ImGuiDir_Left, id, flags);
	const bool            right = TakePress(ImGuiKey_GamepadDpadRight, ImGuiDir_Right, id, flags);
	if (!left && !right) {
		return {};
	}
	const auto* held_key =
	    ImGui::GetKeyData(right ? ImGuiKey_GamepadDpadRight : ImGuiKey_GamepadDpadLeft);
	return {left, right, std::max(0.0f, held_key->DownDuration)};
}

VerticalPresses ClaimVertical(ItemId id) {
	constexpr std::array VERTICAL_KEYS {ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadDpadDown,
	                                    ImGuiKey_UpArrow, ImGuiKey_DownArrow};
	if (ImGui::GetFocusID() != id) {
		// As for left and right, a key held while focus leaves must not stay owned.
		for (const ImGuiKey key: VERTICAL_KEYS) {
			if (ImGui::GetKeyOwner(key) == id) {
				ImGui::SetKeyOwner(key, ImGuiKeyOwner_NoOwner);
			}
		}
		return {};
	}
	for (const ImGuiKey key: VERTICAL_KEYS) {
		ImGui::SetKeyOwner(key, id);
	}
	// Both are read, so neither is left unconsumed for navigation to act on.
	const bool pad_up   = TakePress(ImGuiKey_GamepadDpadUp, ImGuiDir_Up, id, NAV_REPEAT);
	const bool pad_down = TakePress(ImGuiKey_GamepadDpadDown, ImGuiDir_Down, id, NAV_REPEAT);
	const bool key_up   = TakePress(ImGuiKey_UpArrow, ImGuiDir_Up, id, NAV_REPEAT);
	const bool key_down = TakePress(ImGuiKey_DownArrow, ImGuiDir_Down, id, NAV_REPEAT);
	return {pad_up || key_up, pad_down || key_down};
}

HorizontalPresses ClaimTriggers(ItemId id) {
	if (ImGui::GetFocusID() != id) {
		return {};
	}
	return {ImGui::IsKeyPressed(ImGuiKey_GamepadL2, NAV_REPEAT, ImGuiKeyOwner_Any),
	        ImGui::IsKeyPressed(ImGuiKey_GamepadR2, NAV_REPEAT, ImGuiKeyOwner_Any)};
}

namespace {

// Scrolls `memory`'s list just enough to bring `row` inside `area`, preferring its top when it
// doesn't fit.
void ScrollIntoView(InteractionStore::ListMemory& memory, const Rect& row, const Rect& area) {
	float shift = 0.0f;
	if (row.Bottom() > area.Bottom()) {
		shift = row.Bottom() - area.Bottom();
	}
	// A row above the area, or one too tall to fit once its bottom is in, lines up with the top.
	if (row.y - shift < area.y) {
		shift = row.y - area.y;
	}
	memory.scroll += shift;
}

// How far below the list's top its rows reach, unscrolled. The rows were drawn at `drawn_scroll`.
float ContentHeight(std::span<const RecordedRow> rows, const Rect& area, float drawn_scroll) {
	float bottom = 0.0f;
	for (const auto& row: rows) {
		bottom = std::max(bottom, row.rect.Bottom() + drawn_scroll - area.y);
	}
	return bottom;
}

} // namespace

void ScrollListWithPointer(InteractionStore::ListMemory& memory, ItemId list, const Rect& area,
                           const std::optional<Rect>& thumb, float hit_width, float wheel_step) {
	if (!thumb.has_value()) {
		return;
	}
	const float  hidden = memory.content - area.height;
	const ImVec2 mouse  = ImGui::GetIO().MousePos;
	if (ImGui::IsWindowHovered() && ImGui::IsMouseHoveringRect(TopLeft(area), BottomRight(area)) &&
	    ImGui::GetIO().MouseWheel != 0.0f) {
		memory.scroll =
		    std::clamp(memory.scroll - ImGui::GetIO().MouseWheel * wheel_step, 0.0f, hidden);
		memory.pointer_scrolled = true;
	}
	const ItemId id = ImGui::GetIDWithSeed("scroll bar", nullptr, list);
	const Rect   hit {area.Right() - hit_width, area.y, hit_width, area.height};
	const ImRect bounds {TopLeft(hit), BottomRight(hit)};
	(void)ImGui::ItemAdd(bounds, id, nullptr, ImGuiItemFlags_NoNav);
	bool       hovered = false;
	bool       held    = false;
	const bool pressed = ImGui::ButtonBehavior(
	    bounds, id, &hovered, &held, ImGuiButtonFlags_PressedOnClick | ImGuiButtonFlags_NoNavFocus);
	if (pressed) {
		const bool on_thumb = mouse.y >= thumb->y && mouse.y <= thumb->Bottom();
		memory.thumb_grab   = on_thumb ? mouse.y - thumb->y : thumb->height / 2.0f;
	}
	const float travel = area.height - thumb->height;
	if (held && travel > 0.0f) {
		const float share       = (mouse.y - memory.thumb_grab - area.y) / travel;
		memory.scroll           = std::clamp(share, 0.0f, 1.0f) * hidden;
		memory.pointer_scrolled = true;
	}
}

void ScrollListWithPad(InteractionStore::ListMemory& memory, const Rect& area, float step) {
	// A window opening afresh starts at the top, as the store remembers lists for as long as it
	// lasts.
	if (ImGui::IsWindowAppearing()) {
		memory.scroll = 0.0f;
	}
	const float hidden = memory.content - area.height;
	if (!(hidden > 0.0f) || !ImGui::IsWindowFocused()) {
		return;
	}
	const auto pressed = [](ImGuiKey key) {
		return ImGui::IsKeyPressed(key, NAV_REPEAT, ImGuiKeyOwner_Any);
	};
	const bool up   = pressed(ImGuiKey_GamepadDpadUp) || pressed(ImGuiKey_UpArrow);
	const bool down = pressed(ImGuiKey_GamepadDpadDown) || pressed(ImGuiKey_DownArrow);
	memory.scroll =
	    std::clamp(memory.scroll + (down ? step : 0.0f) - (up ? step : 0.0f), 0.0f, hidden);
}

void FinishList(InteractionStore& store, ItemId list, const ListRows& rows, const Rect& area,
                float inset) {
	FinishList(store, list, std::span<const RecordedRow>(rows.rows), area, inset);
}

void FinishList(InteractionStore& store, ItemId list, std::span<const RecordedRow> rows,
                const Rect& area, float inset, float at_least) {
	const ImGuiContext& g      = *GImGui;
	const ItemId        focus  = g.NavId;
	auto&               memory = store.List(list);
	const auto          find   = [&rows](ItemId id) {
        return std::find_if(rows.begin(), rows.end(),
		                               [id](const RecordedRow& row) { return row.id == id; });
	};
	const auto focused    = find(focus);
	const auto remembered = find(memory.row);
	// Measured at the scroll the rows were drawn with, before any change below.
	const float content = std::max(ContentHeight(rows, area, memory.scroll), at_least);
	const bool  inside  = focused != rows.end();
	// A list left undrawn by a screen change may still say focus was inside, but focus can't be on
	// one of its rows on its first frame back, since navigation needs the rows drawn the frame
	// before. So the first frame back sets it right before the D-pad can bring focus in.
	if (inside && !memory.inside && g.NavJustMovedToId == focus && memory.row != focus &&
	    remembered != rows.end() && !remembered->disabled) {
		// The D-pad just moved focus in from outside onto another row, so send it back, and show
		// it.
		memory.pointer_scrolled = false;
		ImGui::SetFocusID(memory.row, ImGui::GetCurrentWindow());
	} else if (inside) {
		// Focus coming in, or moving to another row, takes the list back to it, as confirm
		// shouldn't act on a row out of view. Otherwise a list the mouse has scrolled stays where
		// it was put, away from the row that still has focus.
		memory.pointer_scrolled = memory.pointer_scrolled && memory.inside && memory.row == focus;
		memory.row              = focus;
		if (!memory.pointer_scrolled) {
			ScrollIntoView(
			    memory, focused->rect,
			    {area.x, area.y + inset, area.width, std::max(0.0f, area.height - 2.0f * inset)});
		}
	}
	// Never past the last row, so a list that shrank shows its rows.
	memory.scroll  = std::clamp(memory.scroll, 0.0f, std::max(0.0f, content + inset - area.height));
	memory.content = content + inset;
	memory.inside = inside;
}

bool BackKeyPressed(ItemId owner) {
	return ImGui::IsKeyPressed(CancelKey(), ImGuiInputFlags_None, owner) ||
	       ImGui::IsKeyPressed(ImGuiKey_Escape, ImGuiInputFlags_None, owner);
}

bool CancelKeepsFocus() {
	const auto& io = ImGui::GetIO();
	return !io.ConfigNavEscapeClearFocusItem && !io.ConfigNavEscapeClearFocusWindow;
}

void ConfigureNavigation(const Theme& theme) {
	auto& io          = ImGui::GetIO();
	io.KeyRepeatDelay = theme.interaction.repeat_delay_ms / NAV_MOVE_DELAY_SHARE / MS_PER_SECOND;
	io.KeyRepeatRate  = theme.interaction.repeat_rate_ms / NAV_MOVE_RATE_SHARE / MS_PER_SECOND;
	GImGui->ConfigNavWindowingWithGamepad = false;
	// Back belongs to the screen, such as to close a dialog, so it never drops focus by itself.
	io.ConfigNavEscapeClearFocusItem = false;
}

} // namespace Libs::Graphics::SystemUi
