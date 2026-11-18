#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_VIEWS_TROPHYTOAST_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_VIEWS_TROPHYTOAST_H_

#include "graphics/presentation/system/animation/timeline.h"
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/imgui/imageStore.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct ImDrawList;

namespace Libs::Graphics::SystemUi {

// A trophy the game unlocked, as the overlay received it.
struct TrophyUnlock {
	std::string name;
	// 1 is platinum, 2 gold, 3 silver and 4 bronze. Anything else has no grade.
	int32_t                grade = 0;
	std::vector<std::byte> icon_png;
};

/// The colour of `grade`, numbered as in TrophyUnlock, or transparent when it has none.
[[nodiscard]] Color TrophyGradeColor(const Theme& theme, int32_t grade);

// How the trophy toast is shown, from the emulator's settings.
struct TrophyToastOptions {
	// How long each toast is on screen in total, entrance and exit included.
	float seconds = 0.0f;
	// Fields of ToastDefaults to change for this toast, such as its width or when its shine starts.
	// They set the toast's timings as well as its look.
	TokenPatch<ToastTokens> tokens;
	// For a player who asks for less motion. The toast fades without sliding, as ReducedMotion
	// makes any timeline, starts leaving at the same moment, and has no shine, which is only
	// decoration.
	bool reduced_motion = false;
};

/// The toast's life for a total of `seconds` on screen. It slides down into place from above while
/// fading in, holds, then slides back up while fading out. The hold is what is left of the total
/// once the entrance and exit have taken theirs.
[[nodiscard]] Timeline ToastTimeline(const ToastTokens& toast, float seconds);

// The trophy notification. It shows one unlock at a time at the top right of its area: a toast
// with the trophy's icon, a cup and glow in its grade's colour, its name and a subtitle. It
// slides in from above, holds, and leaves the same way, and platinum and gold trophies get a
// one-off shine. Use it on the present thread.
//
// The slide and fade are the framework's keyed animation. Each toast is a tree keyed by a number
// of its own, so the frame's InteractionStore keeps its start under that key, and Render plays the
// timeline against it, which is how the view knows when the toast has left. The store therefore has
// to be the same one from frame to frame and be collected each frame, or a toast would start over
// every frame. A toast is never dismissed, it leaves when its time is up, so the rule that undoing
// a dismissal replays the entrance doesn't arise, but it would apply if a caller ever cleared one.
// A toast that a frame skips is forgotten by the store and would replay its entrance, so a toast
// must be drawn on every frame that it is shown, as the overlay does while it keeps frames going.
class TrophyToastView {
public:
	/// A view that loads icons into `images`, which must outlive it.
	explicit TrophyToastView(ImageStore& images);
	~TrophyToastView();
	TrophyToastView(const TrophyToastView&)            = delete;
	TrophyToastView& operator=(const TrophyToastView&) = delete;

	/// True when no toast is on screen, so the next unlock can be shown.
	[[nodiscard]] bool Idle() const { return !m_shown.has_value(); }

	/// Starts showing `unlock` with `subtitle`, timed by `theme`, replacing any toast still on
	/// screen. An icon that can't be decoded is left out. The entrance starts on the first Draw, so
	/// call Draw in the frame it is shown, at the same `now`, which the shine is timed from.
	void Show(TrophyUnlock unlock, std::string subtitle, const Theme& theme,
	          const TrophyToastOptions& options, UiClock::time_point now);

	/// Draws the toast at the top right of `area`, such as the safe area below other panels.
	/// `frame` must have been begun for this frame, and its time is the moment drawn. The toast is
	/// built into the frame each call, keyed, and animated and drawn by Render. Returns whether it
	/// will change without new input, so the UI must keep drawing frames. Once the toast has left,
	/// its icon is released and the view is idle. A released icon is freed only once the store
	/// collects it, so the caller must also keep drawing frames while the store has retiring
	/// images.
	bool Draw(UiFrame& frame, ImDrawList& list, const Rect& area);

private:
	struct Shown {
		TrophyUnlock            unlock;
		std::string             subtitle;
		std::optional<ImageId>  icon;
		TokenPatch<ToastTokens> tokens;
		// The toast's life, which Render plays against the store under the toast's key.
		Timeline timeline;
		// The number that keys this toast, so no two toasts share a start in the store.
		uint32_t serial = 0;
		// When it was shown, which the shine is timed from.
		UiClock::time_point start;
		// When the shine starts after `start` and how long it lasts, from the theme it was shown
		// with, so it stays in step with the timeline.
		float shine_start_ms = 0.0f;
		float shine_ms       = 0.0f;
		// Whether a grade that shines gets to, which reduced motion turns off.
		bool shines = true;
	};

	// What the shown toast draws at `now`: its text, icon, grade colours and any shine.
	[[nodiscard]] ToastProps Props(const UiContext& context, UiClock::time_point now) const;
	// Forgets the shown toast and releases its icon, so its texture is freed once unused.
	void Retire();

	ImageStore&          m_images;
	std::optional<Shown> m_shown;
	// The key of the next toast shown.
	uint32_t m_next_serial = 1;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_VIEWS_TROPHYTOAST_H_
