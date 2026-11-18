#include "graphics/presentation/system/animation/easing.h"
#include "graphics/presentation/system/animation/presence.h"
#include "graphics/presentation/system/animation/timeline.h"
#include "graphics/presentation/system/animation/tween.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

namespace {

using Libs::Graphics::SystemUi::ApplyEase;
using Libs::Graphics::SystemUi::DEFAULT_THEME;
using Libs::Graphics::SystemUi::Ease;
using Libs::Graphics::SystemUi::Evaluate;
using Libs::Graphics::SystemUi::Phase;
using Libs::Graphics::SystemUi::Pose;
using Libs::Graphics::SystemUi::Presence;
using Libs::Graphics::SystemUi::ReducedMotion;
using Libs::Graphics::SystemUi::Timeline;
using Libs::Graphics::SystemUi::Transition;
using Libs::Graphics::SystemUi::Tween;
using Libs::Graphics::SystemUi::TweenTracks;
using Libs::Graphics::SystemUi::UiClock;

constexpr float TOLERANCE = 0.001f;
constexpr float NOT_A_NUMBER = std::numeric_limits<float>::quiet_NaN();
constexpr float FOREVER = std::numeric_limits<float>::infinity();

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiAnimationTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two values agree to within rounding noise.
bool Near(float actual, float expected) { return std::fabs(actual - expected) <= TOLERANCE; }

// A moment `ms` milliseconds after an arbitrary origin. Double precision, so moments a day apart
// still land on the exact millisecond.
UiClock::time_point At(double ms) {
  return UiClock::time_point{} +
         std::chrono::duration_cast<UiClock::duration>(std::chrono::duration<double, std::milli>(ms));
}

// A toast's timeline with linear tracks, so poses are easy to work out by hand. It fades in over
// 180 ms while sliding 32 pixels down over 280 ms, holds for 1000 ms, then fades out over 120 ms
// while sliding back up over 180 ms. The exit therefore starts at 1280 and ends at 1460.
Timeline Toast(float hold_ms = 1000.0f) {
  const Transition enter{{180, Ease::Linear}, {280, Ease::Linear}, {0, -32}, 1.0f};
  const Transition exit{{120, Ease::Linear}, {180, Ease::Linear}, {0, -32}, 1.0f};
  return {0.0f, enter, hold_ms, exit};
}

// The pose of the toast timeline at `ms`, optionally dismissed at `dismissed_ms`.
Pose ToastAt(const Timeline &timeline, double ms, std::optional<double> dismissed_ms = std::nullopt) {
  std::optional<UiClock::time_point> dismissed;
  if (dismissed_ms.has_value()) {
    dismissed = At(*dismissed_ms);
  }
  return Evaluate(timeline, At(0), dismissed, At(ms));
}

// Every curve starts at 0 and ends at 1, whatever the input, and NaN counts as the start.
void TestEaseEndpoints() {
  for (const auto ease : {Ease::Linear, Ease::Enter, Ease::Exit, Ease::Standard, Ease::Emphasis}) {
    Check(ApplyEase(ease, 0.0f) == 0.0f && Near(ApplyEase(ease, 1.0f), 1.0f), "curves run from 0 to 1");
    Check(ApplyEase(ease, -1.0f) == 0.0f && Near(ApplyEase(ease, 2.0f), 1.0f), "progress outside 0..1 is clamped");
    Check(ApplyEase(ease, NOT_A_NUMBER) == 0.0f, "NaN progress counts as the start");
  }
}

// Each curve has its own shape. Linear is the baseline at the midpoint, so a curve that falls
// back to linear fails.
void TestEaseShapes() {
  Check(Near(ApplyEase(Ease::Linear, 0.5f), 0.5f), "linear baseline");
  Check(Near(ApplyEase(Ease::Enter, 0.5f), 0.875f), "enter is fast then slow");
  Check(Near(ApplyEase(Ease::Exit, 0.5f), 0.125f), "exit is slow then fast");
  Check(Near(ApplyEase(Ease::Standard, 0.25f), 0.0625f), "standard starts slowly");
  Check(Near(ApplyEase(Ease::Standard, 0.75f), 0.9375f), "standard ends slowly");
  float peak = 0.0f;
  for (int step = 0; step <= 100; step++) {
    peak = std::max(peak, ApplyEase(Ease::Emphasis, static_cast<float>(step) / 100.0f));
  }
  Check(peak > 1.05f, "emphasis overshoots past the end before settling");
}

// Every curve except emphasis only ever moves forward.
void TestEaseIsMonotonic() {
  for (const auto ease : {Ease::Linear, Ease::Enter, Ease::Exit, Ease::Standard}) {
    float previous = 0.0f;
    for (int step = 0; step <= 1000; step++) {
      const float value = ApplyEase(ease, static_cast<float>(step) / 1000.0f);
      Check(value >= previous, "the curve never moves backwards");
      previous = value;
    }
  }
}

// The toast walks through each phase. The fade finishes at 180 while the slide is still moving,
// so the two tracks really are separate, and the exit starts after the hold on its own.
void TestToastPhases() {
  const auto timeline = Toast();
  const auto start = ToastAt(timeline, 0);
  Check(start.phase == Phase::Entering && start.alpha == 0.0f && Near(start.offset.y, -32), "starts hidden above");
  const auto early = ToastAt(timeline, 90);
  Check(Near(early.alpha, 0.5f) && Near(early.offset.y, -32.0f * (1.0f - 90.0f / 280.0f)), "fade and slide run separately");
  const auto faded = ToastAt(timeline, 180);
  Check(faded.phase == Phase::Entering && faded.alpha == 1.0f && faded.offset.y < 0.0f, "faded in, still sliding");
  const auto shown = ToastAt(timeline, 280);
  Check(shown.phase == Phase::Shown && shown.alpha == 1.0f && shown.offset.y == 0.0f, "fully shown");
  Check(ToastAt(timeline, 1279).phase == Phase::Shown, "still holding just before the exit");
  const auto leaving = ToastAt(timeline, 1340);
  Check(leaving.phase == Phase::Exiting && Near(leaving.alpha, 0.5f), "fading out after the hold");
  Check(Near(leaving.offset.y, -32.0f * 60.0f / 180.0f), "sliding back up while it fades");
  Check(ToastAt(timeline, 1460).phase == Phase::Gone, "gone when the exit ends");
}

// A delay holds the element back before it enters, as a list row's stagger does.
void TestDelay() {
  auto timeline = Toast();
  timeline.delay_ms = 100;
  const auto waiting = ToastAt(timeline, 50);
  Check(waiting.phase == Phase::Waiting && waiting.alpha == 0.0f, "waits during the delay");
  Check(Near(ToastAt(timeline, 190).alpha, 0.5f), "enters after the delay");
  const Timeline instant{100.0f, {}, FOREVER, {}};
  const auto hidden = Evaluate(instant, At(0), std::nullopt, At(50));
  Check(hidden.phase == Phase::Waiting && hidden.alpha == 0.0f, "an instant entrance stays hidden during its delay");
}

// A dismissal during the entrance leaves from where the toast is. At 90 it is half faded in, so
// it must leave from half, not jump to fully shown or restart the entrance.
void TestDismissDuringEntrance() {
  const auto timeline = Toast();
  const auto at_dismissal = ToastAt(timeline, 90, 90.0f);
  Check(at_dismissal.phase == Phase::Exiting && Near(at_dismissal.alpha, 0.5f), "leaves from the current fade");
  Check(Near(ToastAt(timeline, 150, 90.0f).alpha, 0.25f), "fades out from there");
  Check(ToastAt(timeline, 270, 90.0f).phase == Phase::Gone, "gone when the exit ends");
}

// The exit uses its own transition. Here it slides out to the right while the entrance came
// from above, so an exit drawn with the entrance's slide would move up instead.
void TestExitHasItsOwnTransition() {
  auto timeline = Toast();
  timeline.exit.slide = {32, 0};
  const auto leaving = ToastAt(timeline, 1370);
  Check(Near(leaving.offset.x, 32.0f * 90.0f / 180.0f) && leaving.offset.y == 0.0f, "the exit slides its own way");
}

// An element shown until dismissed rests while shown and leaves when dismissed.
void TestHoldUntilDismissed() {
  const auto timeline = Toast(FOREVER);
  const auto resting = ToastAt(timeline, 1.0e6f);
  Check(resting.phase == Phase::Shown && !resting.NeedsFrames(timeline), "rests while shown");
  Check(ToastAt(timeline, 5060, 5000.0f).phase == Phase::Exiting, "leaves when dismissed");
  Check(ToastAt(timeline, 5180, 5000.0f).phase == Phase::Gone, "gone after its exit");
}

// A dismissal after the toast has started leaving on its own changes nothing. The pose at 1340
// is the same as without the dismissal.
void TestLateDismissal() {
  const auto timeline = Toast();
  Check(Near(ToastAt(timeline, 1340, 1400.0f).alpha, ToastAt(timeline, 1340).alpha), "a late dismissal is ignored");
}

// A dismissal during the delay means the element never appears.
void TestDismissBeforeEntering() {
  auto timeline = Toast();
  timeline.delay_ms = 100;
  Check(ToastAt(timeline, 40, 50.0f).phase == Phase::Waiting, "still waiting before the dismissal");
  Check(ToastAt(timeline, 60, 50.0f).phase == Phase::Gone, "gone without ever appearing");
}

// Zero durations are instant in both directions.
void TestInstant() {
  const Timeline timeline{0.0f, {}, 100.0f, {}};
  const auto at_start = Evaluate(timeline, At(0), std::nullopt, At(0));
  Check(at_start.phase == Phase::Shown && at_start.alpha == 1.0f, "an instant entrance is shown at once");
  Check(Evaluate(timeline, At(0), std::nullopt, At(100)).phase == Phase::Gone, "an instant exit is gone at once");
}

// Scale runs from the hidden size to full size on the motion track.
void TestScale() {
  const Timeline timeline{0.0f, {{280, Ease::Linear}, {280, Ease::Linear}, {}, 0.96f}, FOREVER, {}};
  Check(Near(Evaluate(timeline, At(0), std::nullopt, At(0)).scale, 0.96f), "starts at the hidden scale");
  Check(Near(Evaluate(timeline, At(0), std::nullopt, At(140)).scale, 0.98f), "halfway to full size");
  Check(Evaluate(timeline, At(0), std::nullopt, At(280)).scale == 1.0f, "ends at full size");
}

// Every phase that will change without input needs frames, and only resting or gone elements
// don't.
void TestNeedsFrames() {
  auto timeline = Toast();
  timeline.delay_ms = 100;
  Check(ToastAt(timeline, 50).NeedsFrames(timeline), "waiting needs frames to start");
  Check(ToastAt(timeline, 150).NeedsFrames(timeline), "entering needs frames");
  Check(ToastAt(timeline, 600).NeedsFrames(timeline), "a timed hold needs frames to end on time");
  Check(ToastAt(timeline, 1420).NeedsFrames(timeline), "exiting needs frames");
  Check(!ToastAt(timeline, 2000).NeedsFrames(timeline), "gone needs none");
}

// Reduced motion keeps only a short linear fade. The theme's limit shortens the 180 ms fade.
void TestReducedMotion() {
  const Transition full{{180, Ease::Enter}, {280, Ease::Enter}, {0, -32}, 0.96f};
  const auto reduced = ReducedMotion(full, DEFAULT_THEME.motion.reduced_fade_max);
  Check(reduced.fade.duration_ms == 120.0f && reduced.fade.ease == Ease::Linear, "a short linear fade");
  Check(reduced.motion.duration_ms == 0.0f && reduced.slide.y == 0.0f && reduced.scale == 1.0f, "no movement");
  const Timeline timeline{0.0f, reduced, FOREVER, reduced};
  const auto start = Evaluate(timeline, At(0), std::nullopt, At(0));
  Check(start.offset.y == 0.0f && start.scale == 1.0f && start.alpha == 0.0f, "it only fades");
  Check(ReducedMotion(Transition{{100, Ease::Enter}}, 120).fade.duration_ms == 100.0f, "a short fade stays short");
  Check(ReducedMotion(Transition{}, 120).fade.duration_ms == 0.0f, "an instant fade stays instant");
  auto staggered = Toast();
  staggered.delay_ms = 90;
  const auto reduced_timeline = ReducedMotion(staggered, 120);
  Check(reduced_timeline.delay_ms == 0.0f, "the stagger goes");
  Check(ToastAt(reduced_timeline, 1279).phase == Phase::Shown && ToastAt(reduced_timeline, 1281).phase == Phase::Exiting,
        "a reduced toast leaves at the same time as the full one");
}

// When the entrance and exit move differently, a dismissal mid-entrance must still leave from
// where the element is. Enter slides from 48 above and exit to 32 above. Halfway in, the element
// is 24 above, and switching to the exit's slide at the same progress would jump it to 16.
void TestInterruptedEntranceIsContinuous() {
  Timeline timeline = Toast();
  timeline.enter.slide = {0, -48};
  timeline.enter.scale = 0.96f;
  timeline.exit.scale = 0.98f;
  const auto before = ToastAt(timeline, 140);
  const auto at_dismissal = ToastAt(timeline, 140, 140.0f);
  Check(Near(before.offset.y, -24.0f) && Near(at_dismissal.offset.y, -24.0f), "the offset doesn't jump on dismissal");
  Check(Near(before.scale, 0.98f) && Near(at_dismissal.scale, 0.98f), "the scale doesn't jump on dismissal");
  const auto halfway_out = ToastAt(timeline, 230, 140.0f);
  Check(Near(halfway_out.offset.y, -28.0f), "it moves from there toward the exit's own end");
}

// Delays and durations that are not finite or are negative count as zero, so nothing waits,
// enters or leaves forever. Each case is compared with the zero value given directly.
void TestNonFiniteTimes() {
  for (const float bad : {NOT_A_NUMBER, FOREVER, -FOREVER, -50.0f}) {
    auto delayed = Toast();
    delayed.delay_ms = bad;
    Check(Near(ToastAt(delayed, 90).alpha, ToastAt(Toast(), 90).alpha), "a bad delay counts as none");
    auto fades = Toast();
    fades.enter.fade.duration_ms = bad;
    fades.exit.fade.duration_ms = bad;
    fades.exit.motion.duration_ms = bad;
    Check(ToastAt(fades, 0).alpha == 1.0f, "a bad entrance fade is instant");
    Check(ToastAt(fades, 1280).phase == Phase::Gone, "a bad exit is instant, never stuck leaving");
  }
  auto no_hold = Toast(NOT_A_NUMBER);
  Check(ToastAt(no_hold, 300).phase == Phase::Exiting, "a NaN hold counts as none");
  const auto absurd = Toast(1.0e30f);
  const auto resting = ToastAt(absurd, 1.0e7);
  Check(resting.phase == Phase::Shown && !resting.NeedsFrames(absurd), "an absurdly long hold rests until dismissed");
}

// A fade can't exceed full opacity even when an overshooting curve drives it.
void TestAlphaStaysInRange() {
  auto timeline = Toast();
  timeline.enter.fade = {600, Ease::Emphasis};
  for (int ms = 0; ms <= 600; ms += 10) {
    const float alpha = ToastAt(timeline, static_cast<float>(ms)).alpha;
    Check(alpha >= 0.0f && alpha <= 1.0f, "alpha stays within 0..1");
  }
}

// An element held open for a day still leaves smoothly. Float milliseconds a day apart are 8 ms
// apart, so working the exit out from the start time would land on the wrong frame of a 120 ms
// fade.
void TestLongLivedExitIsPrecise() {
  const auto timeline = Toast(FOREVER);
  constexpr double DAY_MS = 8.64e7;
  Check(Near(ToastAt(timeline, DAY_MS + 61.0f, DAY_MS + 1.0f).alpha, 0.5f), "a day-old element fades out on time");
}

// The container shows elements while they are on screen and removes them once they are gone.
void TestPresence() {
  Presence<std::string> toasts;
  const auto first = toasts.Add("first", Toast(), At(0));
  toasts.Add("second", Toast(FOREVER), At(0));
  Check(toasts.Update(At(100)) && toasts.Size() == 2, "entering elements need frames");
  std::vector<std::string> drawn;
  toasts.ForEach(At(100), [&](const std::string &value, const Pose &) { drawn.push_back(value); });
  Check(drawn == std::vector<std::string>{"first", "second"}, "visible elements are drawn oldest first");
  Check(toasts.Dismiss(first, At(300)) && toasts.Dismiss(first, At(400)), "dismissing twice is allowed");
  Check(!toasts.Update(At(480)) && toasts.Size() == 1, "the first is gone 180 ms after its first dismissal");
  Check(!toasts.Dismiss(first, At(500)), "a removed element can't be dismissed");
  auto delayed = Toast();
  delayed.delay_ms = 1000;
  toasts.Add("later", delayed, At(500));
  std::vector<std::string> before_delay;
  toasts.ForEach(At(600), [&](const std::string &value, const Pose &) { before_delay.push_back(value); });
  Check(before_delay == std::vector<std::string>{"second"}, "an element still in its delay isn't drawn");
  Check(toasts.Update(At(600)), "an element waiting to enter needs frames");
}

// Linear tracks that rise over 100 ms and fall over 60 ms, as a focus highlight does.
constexpr TweenTracks LINEAR_FOCUS{{100, Ease::Linear}, {60, Ease::Linear}};

// A tween follows its track's curve. Half way through a 100 ms Enter rise it is at Enter(0.5),
// 0.875, where a linear or instant tween would be at 0.5 or 1.
void TestTweenFollowsItsEase() {
  Tween tween;
  tween.Retarget(1, {{100, Ease::Enter}, {60, Ease::Exit}}, At(0));
  Check(Near(tween.Value(At(0)), 0) && Near(tween.Value(At(50)), ApplyEase(Ease::Enter, 0.5f)) &&
            Near(tween.Value(At(100)), 1),
        "a tween follows its rise curve from 0 to 1");
}

// Turning around half way starts from where the value is, and covers the half left in half the
// track's time.
void TestTweenReversal() {
  const TweenTracks even{{100, Ease::Linear}, {100, Ease::Linear}};
  Tween tween;
  tween.Retarget(1, even, At(0));
  tween.Retarget(0, even, At(50));
  Check(Near(tween.Value(At(50)), 0.5f), "turning around doesn't jump");
  Check(Near(tween.Value(At(75)), 0.25f), "it falls at the track's pace");
  Check(Near(tween.Value(At(100)), 0) && !tween.Moving(At(100)), "and arrives in half the track's time");
}

// Asking every frame for the target already set doesn't restart the move. With a curved track a
// restart would bend the path, so the value 75 ms in would no longer be Enter(0.75).
void TestTweenSameTarget() {
  const TweenTracks curved{{100, Ease::Enter}, {60, Ease::Exit}};
  Tween tween;
  tween.Retarget(1, curved, At(0));
  tween.Retarget(1, curved, At(50));
  Check(Near(tween.Value(At(75)), ApplyEase(Ease::Enter, 0.75f)), "the same target doesn't restart the move");
}

// Going down uses the fall track. A focus highlight that loses focus fades out over 60 ms, so it
// is half way at 30 ms, where the 100 ms rise track would only be at 0.7.
void TestTweenFallTrack() {
  Tween tween(1);
  tween.Retarget(0, LINEAR_FOCUS, At(0));
  Check(Near(tween.Value(At(30)), 0.5f), "going down uses the fall track");
  Check(Near(tween.Value(At(60)), 0), "and ends when it does");
}

// A tween needs frames only while it moves.
void TestTweenMoving() {
  Tween tween;
  Check(!tween.Moving(At(0)), "a tween at rest doesn't move");
  tween.Retarget(1, LINEAR_FOCUS, At(0));
  Check(tween.Moving(At(50)), "it moves on its way");
  Check(!tween.Moving(At(100)) && !tween.Moving(At(5000)), "and rests once there");
  Check(Near(tween.Value(At(5000)), 1), "a frame drawn long after, such as after a stall, finds it at its target");
}

// Targets outside 0 to 1 are clamped, and NaN counts as 0.
void TestTweenRange() {
  Tween tween(0.5f);
  tween.Retarget(2, LINEAR_FOCUS, At(0));
  Check(Near(tween.Target(), 1) && Near(tween.Value(At(1000)), 1), "a target above 1 is clamped");
  tween.Retarget(NOT_A_NUMBER, LINEAR_FOCUS, At(1000));
  Check(Near(tween.Target(), 0), "a NaN target counts as 0");
  Check(Near(Tween(-3).Value(At(0)), 0), "a start below 0 is clamped");
}

// Reduced motion makes both tracks linear and caps them, so the knob's overshooting Emphasis curve
// no longer passes 1. The full tracks are the baseline, which do overshoot.
void TestTweenReducedMotion() {
  const TweenTracks knob{{180, Ease::Emphasis}, {180, Ease::Emphasis}};
  const auto reduced = ReducedMotion(knob, 120);
  Check(reduced.rise.ease == Ease::Linear && reduced.fall.ease == Ease::Linear, "reduced tracks are linear");
  Check(Near(reduced.rise.duration_ms, 120) && Near(reduced.fall.duration_ms, 120), "and no longer than the cap");
  const auto highest = [](const TweenTracks &tracks) {
    Tween tween;
    tween.Retarget(1, tracks, At(0));
    float peak = 0;
    for (int ms = 0; ms <= 200; ms += 5) {
      peak = std::max(peak, tween.Value(At(ms)));
    }
    return peak;
  };
  Check(highest(knob) > 1.01f, "the full knob track overshoots");
  Check(highest(reduced) <= 1.0f + TOLERANCE, "the reduced one doesn't");
}

// Turning around while an overshooting knob is past its target starts from the peak, then takes
// no longer than the fall track, though the distance left is more than the whole way.
void TestTweenReversalDuringOvershoot() {
  const TweenTracks knob{{180, Ease::Emphasis}, {180, Ease::Linear}};
  Tween tween;
  tween.Retarget(1, knob, At(0));
  int peak_ms = 0;
  for (int ms = 0; ms <= 180; ms++) {
    peak_ms = tween.Value(At(ms)) > tween.Value(At(peak_ms)) ? ms : peak_ms;
  }
  const float peak = tween.Value(At(peak_ms));
  Check(peak > 1.01f, "the knob overshoots on its way");
  tween.Retarget(0, knob, At(peak_ms));
  Check(Near(tween.Value(At(peak_ms)), peak), "turning around at the peak doesn't jump");
  Check(Near(tween.Value(At(peak_ms + 180)), 0) && !tween.Moving(At(peak_ms + 180)),
        "and it arrives within the fall track's time");
}

// A track whose duration is NaN, infinite or negative is instant, rather than leaving the value
// stuck where it was. The 100 ms track in TestTweenMoving is the baseline that does move.
void TestTweenBadDurations() {
  for (const float duration : {NOT_A_NUMBER, FOREVER, -5.0f}) {
    Tween tween;
    tween.Retarget(1, {{duration, Ease::Linear}, {duration, Ease::Linear}}, At(0));
    Check(Near(tween.Value(At(0)), 1) && !tween.Moving(At(0)), "a broken track duration is instant");
  }
}

// A tween that has arrived reports exactly its target. From 0.76228011 toward 0.0021060533 the
// blend from + (to - from) x 1 rounds to 0.0021060705 in floats, a step off the target.
void TestTweenEndsExactly() {
  constexpr float FROM = 0.76228011f;
  constexpr float TO = 0.0021060533f;
  Tween tween(FROM);
  tween.Retarget(TO, LINEAR_FOCUS, At(0));
  Check(tween.Value(At(60)) == TO && tween.Value(At(5000)) == TO, "an arrived tween is exactly at its target");
}

// With the whole track, a short move still takes the track's whole duration. A move of a tenth
// takes 10 ms on a 100 ms linear track by default, the baseline, but 100 ms with the whole track.
void TestTweenWholeTrack() {
  const TweenTracks linear{{100, Ease::Linear}, {100, Ease::Linear}};
  Tween scaled;
  scaled.Retarget(0.1f, linear, At(0));
  Check(!scaled.Moving(At(20)), "by default a short move is quick");
  Tween whole;
  whole.Retarget(0.1f, linear, At(0), true);
  Check(whole.Moving(At(50)) && Near(whole.Value(At(50)), 0.05f), "with the whole track it takes the whole duration");
  Check(!whole.Moving(At(100)) && Near(whole.Value(At(100)), 0.1f), "and arrives at its end");
}

} // namespace

int main() {
  TestEaseEndpoints();
  TestEaseShapes();
  TestEaseIsMonotonic();
  TestToastPhases();
  TestDelay();
  TestDismissDuringEntrance();
  TestExitHasItsOwnTransition();
  TestHoldUntilDismissed();
  TestLateDismissal();
  TestDismissBeforeEntering();
  TestInstant();
  TestScale();
  TestNeedsFrames();
  TestReducedMotion();
  TestInterruptedEntranceIsContinuous();
  TestNonFiniteTimes();
  TestAlphaStaysInRange();
  TestLongLivedExitIsPrecise();
  TestPresence();
  TestTweenFollowsItsEase();
  TestTweenReversal();
  TestTweenSameTarget();
  TestTweenFallTrack();
  TestTweenMoving();
  TestTweenRange();
  TestTweenReducedMotion();
  TestTweenReversalDuringOvershoot();
  TestTweenBadDurations();
  TestTweenEndsExactly();
  TestTweenWholeTrack();
  std::puts("System UI animation tests passed");
  return 0;
}
