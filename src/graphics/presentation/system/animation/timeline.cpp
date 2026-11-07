#include "graphics/presentation/system/animation/timeline.h"

#include "graphics/presentation/system/animation/easing.h"

#include <algorithm>
#include <cmath>

namespace Libs::Graphics::SystemUi {

namespace {

using Milliseconds = std::chrono::duration<float, std::milli>;

// Longer than any element stays on screen, and short enough that adding it to a clock time never
// overflows. A longer hold counts as shown until dismissed.
constexpr float LONGEST_TIME_MS = 1.0e9f; // about eleven days

// How far each track has come from hidden (0) to shown (1).
struct Shown {
	float fade   = 0.0f;
	float motion = 0.0f;
};

// `value` when it is a finite number above zero, otherwise zero. Delays and durations from a
// theme or a caller can't make an element wait, enter or leave forever.
float NonNegativeFinite(float value) {
	return std::isfinite(value) && value > 0.0f ? value : 0.0f;
}

// A clock duration of `ms` milliseconds, which must be within LONGEST_TIME_MS.
UiClock::duration ToDuration(float ms) {
	return std::chrono::duration_cast<UiClock::duration>(Milliseconds(ms));
}

// Milliseconds from `from` to `to`. The difference is taken in whole clock ticks first, so it
// stays exact however long ago `from` was.
float Elapsed(UiClock::time_point from, UiClock::time_point to) {
	return std::chrono::duration_cast<Milliseconds>(to - from).count();
}

// A track's eased progress `elapsed` milliseconds in. A track without a positive duration has
// already finished.
float Progress(const Track& track, float elapsed) {
	const float duration = std::min(NonNegativeFinite(track.duration_ms), LONGEST_TIME_MS);
	return duration > 0.0f ? ApplyEase(track.ease, elapsed / duration) : 1.0f;
}

// How long a transition runs, the longer of its two tracks.
float Length(const Transition& transition) {
	return std::min(std::max(NonNegativeFinite(transition.fade.duration_ms),
	                         NonNegativeFinite(transition.motion.duration_ms)),
	                LONGEST_TIME_MS);
}

// The pose `elapsed` milliseconds into the entrance. Hidden means the transition's slide and
// scale, shown means no offset at full size. Alpha stays within 0..1 even when an overshooting
// curve drives the fade.
Pose EnteringPose(Phase phase, const Transition& enter, float elapsed) {
	const Shown shown {Progress(enter.fade, elapsed), Progress(enter.motion, elapsed)};
	const float hidden = 1.0f - shown.motion;
	return {phase,
	        std::clamp(shown.fade, 0.0f, 1.0f),
	        {enter.slide.x * hidden, enter.slide.y * hidden},
	        enter.scale + (1.0f - enter.scale) * shown.motion};
}

// The pose `elapsed` milliseconds into the exit, which left from `from`. It moves from wherever
// the element was toward the exit's hidden end, so an interrupted entrance never jumps, even when
// the exit slides or scales differently from the entrance.
Pose ExitingPose(const Transition& exit, const Pose& from, float elapsed) {
	const float fade   = Progress(exit.fade, elapsed);
	const float motion = Progress(exit.motion, elapsed);
	return {Phase::Exiting,
	        std::clamp(from.alpha * (1.0f - fade), 0.0f, 1.0f),
	        {from.offset.x + (exit.slide.x - from.offset.x) * motion,
	         from.offset.y + (exit.slide.y - from.offset.y) * motion},
	        from.scale + (exit.scale - from.scale) * motion};
}

// The fully hidden pose at the end of the exit.
Pose GonePose(const Transition& exit) {
	return {Phase::Gone, 0.0f, exit.slide, exit.scale};
}

// The hold, with NaN and negative holds counting as no hold.
float HoldMs(const Timeline& timeline) {
	return std::isnan(timeline.hold_ms) ? 0.0f : std::max(0.0f, timeline.hold_ms);
}

// When the exit begins, the earlier of the end of the hold and the dismissal, or never.
std::optional<UiClock::time_point> ExitStart(const Timeline&                    timeline,
                                             UiClock::time_point                enter_start,
                                             std::optional<UiClock::time_point> dismissed) {
	std::optional<UiClock::time_point> exit_start = dismissed;
	if (!HoldsUntilDismissed(timeline)) {
		const auto automatic = enter_start + ToDuration(Length(timeline.enter) + HoldMs(timeline));
		exit_start           = dismissed.has_value() ? std::min(*dismissed, automatic) : automatic;
	}
	return exit_start;
}

} // namespace

bool HoldsUntilDismissed(const Timeline& timeline) {
	// Written so a NaN sum also counts as never ending.
	return !(Length(timeline.enter) + HoldMs(timeline) <= LONGEST_TIME_MS);
}

Pose Evaluate(const Timeline& timeline, UiClock::time_point start,
              std::optional<UiClock::time_point> dismissed, UiClock::time_point now) {
	const float delay       = std::min(NonNegativeFinite(timeline.delay_ms), LONGEST_TIME_MS);
	const auto  enter_start = start + ToDuration(delay);
	const auto  exit_start  = ExitStart(timeline, enter_start, dismissed);
	// Dismissed before it ever appeared, so it never will.
	if (exit_start.has_value() && *exit_start < enter_start && now >= *exit_start) {
		return GonePose(timeline.exit);
	}
	if (now < enter_start) {
		// Spelled out, since an instant entrance would otherwise count as already shown.
		return {Phase::Waiting, 0.0f, timeline.enter.slide, timeline.enter.scale};
	}
	if (!exit_start.has_value() || now < *exit_start) {
		const float elapsed = Elapsed(enter_start, now);
		const auto  phase   = elapsed < Length(timeline.enter) ? Phase::Entering : Phase::Shown;
		return EnteringPose(phase, timeline.enter, elapsed);
	}
	const float into_exit = Elapsed(*exit_start, now);
	if (into_exit >= Length(timeline.exit)) {
		return GonePose(timeline.exit);
	}
	const auto from =
	    EnteringPose(Phase::Entering, timeline.enter, Elapsed(enter_start, *exit_start));
	return ExitingPose(timeline.exit, from, into_exit);
}

Transition ReducedMotion(const Transition& transition, float max_fade_ms) {
	const float fade =
	    std::min(NonNegativeFinite(transition.fade.duration_ms), NonNegativeFinite(max_fade_ms));
	return {{fade, Ease::Linear}, {}, {}, 1.0f};
}

Timeline ReducedMotion(const Timeline& timeline, float max_fade_ms) {
	const auto enter = ReducedMotion(timeline.enter, max_fade_ms);
	return {0.0f, enter, timeline.hold_ms + Length(timeline.enter) - Length(enter),
	        ReducedMotion(timeline.exit, max_fade_ms)};
}

} // namespace Libs::Graphics::SystemUi
