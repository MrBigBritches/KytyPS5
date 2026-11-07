#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_TIMELINE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_TIMELINE_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/theme.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>

// How an element enters, stays and leaves. Every pose is computed from absolute times, never by
// adding up frame times, so a game that stalls for seconds during a shader compile can't freeze
// or skip an animation. It resumes at the right point on the next frame.
namespace Libs::Graphics::SystemUi {

using UiClock = std::chrono::steady_clock;

// One animated value going from hidden to shown or back. A zero duration is instant.
struct Track {
	float duration_ms = 0.0f;
	Ease  ease        = Ease::Linear;
};

// How an element arrives or leaves. Opacity and movement run on separate tracks, since a toast
// fades in faster than it slides. Slide and scale describe the hidden end, and the shown end is
// always no offset at full size.
struct Transition {
	Track fade;
	Track motion;
	// The offset at the hidden end, in reference pixels, so ScaleContext::Px converts it.
	Vec2 slide;
	// The size at the hidden end, as a factor about the element's centre.
	float scale = 1.0f;
};

// The whole life of an element.
struct Timeline {
	// A wait before entering, such as a list row's stagger.
	float      delay_ms = 0.0f;
	Transition enter;
	// How long the element stays fully shown before leaving on its own. Infinite stays until it
	// is dismissed.
	float      hold_ms = std::numeric_limits<float>::infinity();
	Transition exit;
};

enum class Phase : uint8_t { Waiting, Entering, Shown, Exiting, Gone };

/// True when `timeline` never leaves on its own, because its hold is infinite or too long to
/// represent. Evaluate and Pose::NeedsFrames both use it, so they always agree.
[[nodiscard]] bool HoldsUntilDismissed(const Timeline& timeline);

// What to draw at one moment. The offset is in reference pixels.
struct Pose {
	Phase phase = Phase::Waiting;
	float alpha = 0.0f;
	Vec2  offset;
	float scale = 1.0f;

	/// True while the pose will change without new input, so the UI must keep drawing frames.
	/// Only an element shown until dismissed, or one that is gone, can rest.
	[[nodiscard]] bool NeedsFrames(const Timeline& timeline) const {
		return phase != Phase::Gone && !(phase == Phase::Shown && HoldsUntilDismissed(timeline));
	}
};

/// The pose at `now` of an element that started at `start` and was dismissed at `dismissed`, if
/// it was. A dismissal leaves from wherever the element is, moving toward the exit's hidden end,
/// so an interrupted entrance never jumps. A dismissal after the element has started leaving on
/// its own changes nothing. Delays and durations that are negative or not finite count as zero,
/// and a hold too long to represent counts as shown until dismissed.
[[nodiscard]] Pose Evaluate(const Timeline& timeline, UiClock::time_point start,
                            std::optional<UiClock::time_point> dismissed, UiClock::time_point now);

/// The fade-only form of `transition` for players who ask for reduced motion. Slides and scaling
/// are removed, and the fade becomes linear and no longer than `max_fade_ms`.
[[nodiscard]] Transition ReducedMotion(const Transition& transition, float max_fade_ms);

/// The reduced-motion form of a whole timeline. Both transitions only fade, and the delay is
/// removed, since a stagger is motion too. The hold grows by however much the entrance got
/// shorter, so the element starts leaving at the same moment.
[[nodiscard]] Timeline ReducedMotion(const Timeline& timeline, float max_fade_ms);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_TIMELINE_H_
