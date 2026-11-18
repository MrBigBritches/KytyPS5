#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_INTERACTION_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_INTERACTION_H_

#include "graphics/presentation/system/animation/timeline.h"
#include "graphics/presentation/system/animation/tween.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/geometry.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

struct ImDrawList;

namespace Libs::Graphics::SystemUi {

// ImGui's identifier for an item, the same type as ImGuiID.
using ItemId = unsigned int;

/// The ImGui item id for a control named `name`, unique within the current window and ID scope.
/// A name must not be empty.
[[nodiscard]] ItemId IdFor(std::string_view name);

// A scope that makes the names of the controls drawn inside it unique to it, so two groups of
// controls can use the same names, such as two tab bars with the same tabs. Like IdFor, the name
// must not be empty, or unnamed scopes would share their names.
class IdScope {
public:
	/// Opens the scope `name` within the current one.
	explicit IdScope(std::string_view name);
	/// Closes it.
	~IdScope();
	IdScope(const IdScope&)            = delete;
	IdScope& operator=(const IdScope&) = delete;
};

// Keeps the controls drawn while it lives out of directional navigation, when `excluded` says so, so
// a move can't land on them. A click or FocusControl still reaches them.
class NavExclusion {
public:
	/// Opens the exclusion. With `excluded` false the controls are in navigation, even inside another
	/// exclusion.
	explicit NavExclusion(bool excluded);
	/// Closes it.
	~NavExclusion();
	NavExclusion(const NavExclusion&)            = delete;
	NavExclusion& operator=(const NavExclusion&) = delete;
};

/// The id `name` gets when `seed` is the innermost id of the ID stack, which is what IdFor gives
/// inside scopes whose combined id is `seed`. Chaining it down a path of names gives the id the
/// same names nested in IdScope or PushID would. A name must not be empty.
[[nodiscard]] ItemId IdWithSeed(std::string_view name, ItemId seed);

/// The same for a number, as PushID with an integer hashes it.
[[nodiscard]] ItemId IdWithSeed(uint32_t number, ItemId seed);

/// The innermost id of the current window's ID stack, which seeds the ids IdFor makes now.
[[nodiscard]] ItemId CurrentIdSeed();

/// Moves focus to the control `id` in the current window.
void FocusControl(ItemId id);

/// Moves focus to the control `id` as a D-pad move does, so its focus ring shows even while the
/// mouse is in the window, for a control that moves focus itself on a key it owns.
void MoveFocus(ItemId id);

/// Makes the control just registered the one focus lands on when its window first appears.
void MakeLastControlDefaultFocus();

/// The control that has focus, or 0 when none has.
[[nodiscard]] ItemId FocusedControl();

/// True when the D-pad asked this frame to move focus left and no control drawn so far in the
/// window lies that way, level with the focused one. ImGui only moves sideways to a control whose
/// middle 60% overlaps the focused one's top to bottom. Call it after drawing every control the
/// move could reach.
[[nodiscard]] bool LeftMoveFindsNothing();

// How the player last moved around the UI. With a gamepad or the keyboard the focused control
// shows its ring and nothing shows hover. With the mouse, hover shows and the ring hides.
enum class InputMode : uint8_t { Pad, Pointer };

/// Where the mouse is, in screen pixels.
[[nodiscard]] Vec2 PointerPosition();

/// True while the mouse, not the gamepad, holds the control `id` down, such as while dragging.
[[nodiscard]] bool HeldByPointer(ItemId id);

// What the player is doing to one control this frame. Hovered means the mouse is over it, never
// that the gamepad has focused it.
struct ItemState {
	bool hovered   = false;
	bool focused   = false;
	bool held      = false;
	bool activated = false;
	bool disabled  = false;
};

// How far each of a control's highlights has come, from 0 to 1. Hover shows only for the pointer
// and focus only for the pad, so a control draws these, not its raw state. A disabled control
// shows no hover or press, but keeps focus if it had it.
struct ItemVisuals {
	float hover = 0.0f;
	float focus = 0.0f;
	float press = 0.0f;
};

// The highlights of a view's controls, each eased by its own tweens so they fade rather than
// snap. A view keeps one store for its controls.
class InteractionStore {
public:
	/// Moves the highlights of the control `id` toward `state` at `now` and returns them.
	ItemVisuals Update(ItemId id, const ItemState& state, InputMode mode, const Theme& theme,
	                   UiClock::time_point now);
	/// The animated value of the control `id`, such as a toggle's knob, moving toward `target` on
	/// `tracks` from wherever it is. The first time a control asks, the value starts at its target,
	/// so a toggle that is already on doesn't slide on when it appears. It is forgotten with the
	/// control's highlights. Name controls by what they show, not their position in a list, or a
	/// list whose rows change would slide one row's value into the next.
	/// With `whole_track`, every move takes the track's whole duration, however short.
	float Animate(ItemId id, float target, const TweenTracks& tracks, UiClock::time_point now,
	              bool whole_track = false);
	/// Plays `timeline` for the control `id` and returns its pose at `now`. The first frame a
	/// control plays is its start, and the first frame `dismissed` is true is its dismissal, which
	/// is recorded once and not moved by later frames, so holding `dismissed` plays the exit once.
	/// A frame with `dismissed` false after a dismissal undoes it and restarts the entrance from
	/// that frame, so a row hidden and shown again enters again. A different timeline under the
	/// same id keeps the start and continues from it, so a change mid-entrance jumps to the new
	/// curve's position at the same elapsed time. Both are kept with the control, so rows named by
	/// what they show keep their own start when they are reordered. A frame that passes without the
	/// control makes Collect forget it, and it then replays from its entrance. Animating counts the
	/// timeline while it moves.
	Pose Play(ItemId id, const Timeline& timeline, bool dismissed, UiClock::time_point now);
	/// Forgets controls not updated since the last call, so one that comes back starts at rest.
	/// Call once a frame, after drawing.
	void Collect();
	/// True while any highlight is still moving at `now`, so the UI must keep drawing frames.
	[[nodiscard]] bool Animating(UiClock::time_point now) const;
	/// Remembers, for the list `list`, which row last had focus, whether focus was inside it when
	/// it was last drawn, and how far it has scrolled. Kept until the store goes, so a list
	/// returns to its row after focus has been elsewhere. Lists named from data, such as one per
	/// game, add an entry each.
	struct ListMemory {
		ItemId row    = 0;
		bool   inside = false;
		float  scroll = 0.0f;
		// How tall the rows were found to be last frame, for the scroll bar.
		float content = 0.0f;
		// Whether the mouse scrolled the list since focus last moved, so it stays where the player
		// put it rather than going back to the focused row.
		bool pointer_scrolled = false;
		// Where in the thumb a drag took hold of it, from its top.
		float thumb_grab = 0.0f;
	};
	[[nodiscard]] ListMemory& List(ItemId list) { return m_lists[list]; }

private:
	struct Entry {
		Tween hover;
		Tween focus;
		Tween press;
		Tween value;
		bool  value_started = false;
		bool  seen          = true;
		// The keyed animation Play runs, whose start and dismissal are kept here.
		bool                               played = false;
		Timeline                           timeline;
		UiClock::time_point                start;
		std::optional<UiClock::time_point> dismissed_at;
	};
	std::unordered_map<ItemId, Entry>      m_entries;
	std::unordered_map<ItemId, ListMemory> m_lists;
};

// A control's state and highlights this frame.
struct Interaction {
	ItemState   state;
	ItemVisuals visuals;
};

/// The draw list of the ImGui window being drawn into, which is where the system UI draws, so a
/// caller holding only a frame can draw hand-made shapes or pass the list to Render.
[[nodiscard]] ImDrawList& CurrentDrawList();

// Which of up and down were pressed this frame for a focused control.
struct VerticalPresses {
	bool up   = false;
	bool down = false;
};

/// While the control `id` has focus, takes up and down from navigation, on the D-pad and the arrow
/// keys, and returns which were pressed this frame, a held one repeating at the pace a held D-pad
/// moves focus. It is for a control that moves focus itself, along a column that navigation would
/// leave for something nearer to one side. It gives them back once the control loses focus. Call it
/// every frame after the control has interacted.
[[nodiscard]] VerticalPresses ClaimVertical(ItemId id);

// Which of the D-pad's left and right were pressed this frame for a focused control.
struct HorizontalPresses {
	bool left  = false;
	bool right = false;
	// How long the pressed direction has been held, in seconds, so a control can speed up. Zero
	// when neither was pressed, and right's when both were.
	float held_seconds = 0.0f;
};

/// The owner that reads a key group only while nobody owns it, as a screen-wide shortcut does.
inline constexpr ItemId NO_OWNER = ~0u;
/// The owner that reads a key group whoever owns it, which keeps the behaviour from before keys
/// had owners.
inline constexpr ItemId ANY_OWNER = 0u;

/// A group of keys that one layer of the UI can own at a time.
enum class KeyGroup : uint8_t {
	/// The gamepad's L1 and R1.
	Shoulders,
	/// The gamepad's cancel button as ImGui maps it, so swapped confirm and back are honoured, and
	/// Escape.
	Back,
};

/// How firmly a claim holds a key group. The frontmost layer wins, and within a layer the
/// first owner to claim it. A Control never takes a group from a live owner, and a Modal always
/// takes it, since it sits in front of everything else.
enum class ClaimPriority : uint8_t {
	Control,
	Modal,
};

/// Makes `owner` the owner of `group` for this frame and the next, and returns whether it owns it.
/// Keep claiming every frame to keep it. Navigation, the gallery's shortcuts and anything else
/// that reads the group as NO_OWNER stop seeing its presses, and `owner` reads them with
/// ShoulderPresses(owner) or BackKeyPressed(owner). A Control claim is refused, and returns false,
/// when another owner holds the group. A Modal claim always succeeds. Ownership lingers a frame
/// after the last claim, so a layer that has just closed still blocks the press that closed it.
/// Owning Back also stops ImGui's own cancel from running. Here that only means releasing a held
/// widget, since ConfigureNavigation turns off its clearing of focus and a modal is not closed by
/// it.
/// A Modal claim made later in a frame where the keys were consumed takes ownership and keeps
/// the lock, so the modal's own read as owner still sees the consumed press. That is harmless,
/// since the gallery's shell consumes only when focus is in its own window, and a modal has focus
/// in its own.
bool ClaimKeys(KeyGroup group, ItemId owner, ClaimPriority priority);

/// Whether the gamepad's L1 (`left`) and R1 (`right`) shoulder buttons were pressed this frame,
/// for switching tabs from anywhere on a screen. By default only a press nobody owns counts, so one
/// another control or dialog has taken by owning the key doesn't. Pass the `owner` given to
/// ClaimKeys to read the press as that owner, which counts it while that owner holds the keys and
/// also while nobody does, since ImGui's owner test passes for an unowned key. That is why whatever
/// acts on a press must ConsumeKey it, or a layer that appears in the same frame and claims
/// afterwards would act on the same press. ANY_OWNER reads it whoever owns it. Holding the
/// gamepad's menu button with L1 or R1 also switches ImGui windows, whoever owns the keys, which
/// --system-ui turns off.
[[nodiscard]] HorizontalPresses ShoulderPresses(ItemId owner = NO_OWNER);

/// Makes `actor` consume the keys of `group` for the rest of this frame, and call it when `actor`
/// acts on a press of them. Every reader but `actor`, including a layer drawn later in the frame
/// that claims the group, then sees no press, which a claim alone can't do since it only takes
/// effect for the next frame. Ownership lingers into the next frame as for ClaimKeys.
void ConsumeKey(KeyGroup group, ItemId actor);

/// Whether focus is in the window being drawn into, or one of its child windows, and not in a
/// popup such as a modal drawn over it. A screen-wide shortcut uses it to leave alone what happens
/// in a dialog in front of it.
[[nodiscard]] bool FocusInCurrentWindow();

/// True on the frame the player pressed back, the gamepad's cancel button as ImGui maps it, so
/// swapped confirm and back are honoured, or Escape. By default whoever owns the key doesn't
/// matter. Pass the `owner` given to ClaimKeys to read it as that owner, which counts it while that
/// owner holds the key and also while nobody does, so whatever acts on it must ConsumeKey it. Pass
/// NO_OWNER to read it only while nobody holds it, as a shortcut does.
[[nodiscard]] bool BackKeyPressed(ItemId owner = ANY_OWNER);

/// Whether back leaves focus where it is, which ConfigureNavigation sets up and which a screen that
/// owns back relies on. When it does not, ImGui's own cancel drops focus, and owning back would
/// take that away, so a layer that owns back checks this first.
[[nodiscard]] bool CancelKeepsFocus();

/// Applies the system UI's navigation settings to the current ImGui context, which the old overlay
/// shares. Holding a D-pad direction moves focus at the theme's repeat pace, and holding the
/// gamepad's Square button no longer opens ImGui's window switcher, so controls can use it.
/// Back no longer clears focus, so a dialog that ignores it keeps its ring.
void ConfigureNavigation(const Theme& theme);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_IMGUI_INTERACTION_H_
