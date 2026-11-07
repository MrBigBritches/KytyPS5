#include "SystemUiNavHarness.h"

#include "graphics/presentation/system/components/slider.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/frameInternals.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>

namespace {

using namespace SystemUiTest;

constexpr float TOLERANCE = 0.001f;
// A slider 400 wide at the reference size, so its 28 pixel knob travels from x 114 to 486.
constexpr Rect SLIDER{100, 100, 400, 44};
constexpr float LEFT_END = SLIDER.x + 14;
constexpr float RIGHT_END = SLIDER.Right() - 14;
constexpr Rect NEIGHBOUR{600, 100, 120, 44};
// How far from the thumb's centre its shadow reaches at the reference size, with room to spare.
constexpr float SHADOW_REACH = 40;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiSliderTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two values agree to within `tolerance`.
bool Near(float actual, float expected, float tolerance = TOLERANCE) { return std::fabs(actual - expected) <= tolerance; }

// The vertices the view window drew in its last frame.
const ImVector<ImDrawVert> &Drawn() { return ImGui::FindWindowByName("view")->DrawList->VtxBuffer; }

// The horizontal extent of the last frame's vertices of exactly `color` within the slider's rows,
// or an empty span when there are none.
std::pair<float, float> Span(Color color) {
  float left = std::numeric_limits<float>::max(), right = -left;
  for (const auto &vertex : Drawn()) {
    if (vertex.col == ToImU32(color) && vertex.pos.y > SLIDER.y - 20 && vertex.pos.y < SLIDER.Bottom() + 20) {
      left = std::min(left, vertex.pos.x);
      right = std::max(right, vertex.pos.x);
    }
  }
  return {left, right};
}

// A slider element drawn each frame.
struct Rig {
  UiFrame frame;
  SliderParts slider;
};

// Builds a slider over `value` in `rect` and renders it in this frame's window.
void DrawSlider(InteractContext &context, Rig &rig, float *value, const SliderProps &props, const Rect &rect) {
  rig.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
  const ItemLayout size{.width = Dp(rect.width), .height = Dp(rect.height)};
  rig.slider = Slider(rig.frame, "slider", value, props, size);
  Check(!rig.slider.root.IsNone(), "the slider was built");
  const float unlimited = std::numeric_limits<float>::infinity();
  (void)Measure(rig.frame, rig.slider.root, {unlimited, unlimited});
  Check(Arrange(rig.frame, rig.slider.root, rect), "the slider was arranged");
  Check(!Render(rig.frame, rig.slider.root, context.list).error.has_value(), "the slider rendered");
}

// A slider from 0 to 100 in steps of 1, with a neighbour to its right.
struct Scene {
  Rig rig;
  float value = 50;
  SliderProps props{0, 100, 1};
  // What the slider last reported of a change by the player, and how many it has reported.
  bool changed = false;
  int changes = 0;
  bool slider_focused = false;
  bool slider_active = false;
  bool neighbour_focused = false;
};

// Draws the slider then its neighbour, recording what happened in `scene`.
auto DrawScene(Scene &scene) {
  return [&scene](InteractContext &context) {
    scene.props.changed = &scene.changed;
    DrawSlider(context, scene.rig, &scene.value, scene.props, SLIDER);
    scene.changes += scene.changed ? 1 : 0;
    const ItemId id = IdOf(scene.rig.frame, scene.rig.slider.track);
    scene.slider_focused = ImGui::GetFocusID() == id;
    scene.slider_active = ImGui::GetActiveID() == id;
    scene.neighbour_focused = Interact(context, ImGui::GetID("neighbour"), NEIGHBOUR, false).state.focused;
  };
}

// Right and left move the slider one step, keeping focus, and stop at the ends.
void TestSteps() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(Near(scene.value, 51) && scene.slider_focused && !scene.neighbour_focused, "right moves one step and keeps focus");
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(Near(scene.value, 49) && scene.changes == 3, "left moves one step back");
  scene.value = 100;
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(Near(scene.value, 100), "it stops at the top");
  scene.value = 0;
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(Near(scene.value, 0), "and at the bottom");
}

// The value changes on the frame of the press, before the slider is painted, and not a frame
// later. The change count and the drawn knob agree with it on that frame.
void TestChangesOnThePressFrame() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, true);
  nav.Frame(draw);
  Check(Near(scene.value, 51) && scene.changes == 1, "right moves the value on the frame it is pressed");
  const auto [knob_left, knob_right] = Span(DEFAULT_THEME.colors.control.knob);
  const float knob_x = LEFT_END + (RIGHT_END - LEFT_END) * 0.51f;
  Check(Near((knob_left + knob_right) / 2, knob_x, 0.5f), "and the knob is drawn there on that frame");
  nav.Key(ImGuiKey_GamepadDpadRight, false);
  nav.Frames(2, draw);
}

// The report of a change is false when the slider is built, so a frame that builds it and never
// reaches Render does not repeat the last frame's change.
void TestChangeReportClearsWhenRenderIsSkipped() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, true);
  nav.Frame(draw);
  Check(scene.changed, "the press changed the value");
  Rig skipped;
  nav.Frame([&](InteractContext &context) {
    skipped.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    (void)Slider(skipped.frame, "slider", &scene.value, scene.props);
  });
  Check(!scene.changed, "a frame that builds the slider without rendering it reports no change");
  nav.Key(ImGuiKey_GamepadDpadRight, false);
  nav.Frames(2, draw);
}

// R2 and L2 move a page of ten steps.
void TestPages() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadR2, draw);
  Check(Near(scene.value, 60), "R2 moves up a page");
  nav.Press(ImGuiKey_GamepadL2, draw);
  nav.Press(ImGuiKey_GamepadL2, draw);
  Check(Near(scene.value, 40), "L2 moves down a page");
}

// A held direction moves one step a repeat at first, two after 0.6 seconds and five after 1.5.
void TestHoldSpeeds() {
  Check(SliderStepsForHold(0.0f) == 1 && SliderStepsForHold(0.59f) == 1, "one step at first");
  Check(SliderStepsForHold(0.6f) == 2 && SliderStepsForHold(1.49f) == 2, "two after 0.6 seconds");
  Check(SliderStepsForHold(1.5f) == 5 && SliderStepsForHold(10.0f) == 5, "five after 1.5 seconds");
}

// With the system UI's navigation settings, a held right repeats at the theme's 400 ms then
// 80 ms, as focus moves. Held for 103 frames it reaches exactly 41: one step at the press and at
// 400, 480 and 560 ms, two steps a repeat from 640 ms, and five from 1.52 s. Never going past two
// steps a repeat would give 32, one step throughout 18, and ImGui's own faster pace 82.
void TestHoldSpeedsUp() {
  Scene scene;
  scene.value = 0;
  scene.props = {0, 1000, 1};
  NavHarness nav;
  ConfigureNavigation(DEFAULT_THEME);
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, true);
  nav.Frames(103, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, false);
  nav.Frame(draw);
  Check(Near(scene.value, 41), "a held right speeds up at the theme's pace");
}

// A value that isn't a number becomes the minimum.
void TestNotANumber() {
  Scene scene;
  scene.value = std::numeric_limits<float>::quiet_NaN();
  NavHarness nav;
  nav.Frame(DrawScene(scene));
  Check(scene.value == 0.0f, "a value that isn't a number becomes the minimum");
}

// A value between steps settles on the nearest whole step without counting as a change the
// player made, and a pointer between steps settles too.
void TestSnapsToSteps() {
  Scene scene;
  scene.value = 33.4f;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frame(draw);
  Check(scene.value == 33.0f && scene.changes == 0, "a stored value settles on the nearest step quietly");
  nav.MoveMouse(LEFT_END + (RIGHT_END - LEFT_END) * 0.756f, SLIDER.y + 22, draw);
  nav.MouseButton(true);
  nav.Frames(2, draw);
  nav.MouseButton(false);
  nav.Frame(draw);
  Check(scene.value == 76.0f && scene.changes == 1, "a pointer settles on the nearest step");
}

// Within half a step of the top is the top, so a step that doesn't divide the range still
// reaches it. In steps of 0.3, 0.9 is within half a step of 1, so it is 1 and stays 1 after
// right, rather than falling back to 0.9, the last whole step.
void TestReachesTheTop() {
  Scene scene;
  scene.value = 0.9f;
  scene.props = {0, 1, 0.3f};
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(Near(scene.value, 1), "a step that doesn't divide the range still reaches the top");
}

// A slider with no step moves a hundredth of its range per press.
void TestUnstepped() {
  Scene scene;
  scene.value = 0.5f;
  scene.props = {0, 1, 0};
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(Near(scene.value, 0.51f), "an unstepped slider moves a hundredth per press");
}

// A slider with no step stops at its top and its bottom, since nothing else rounds it back into the
// range. Half a hundredth below the top, a press of right would go over by half a hundredth.
void TestUnsteppedStopsAtTheEnds() {
  Scene scene;
  scene.value = 0.995f;
  scene.props = {0, 1, 0};
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(scene.value == 1.0f, "an unstepped slider stops at its top");
  scene.value = 0.005f;
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(scene.value == 0.0f, "and at its bottom");
}

// A range given backwards is put in order: from 100 down to 0 behaves as 0 to 100.
void TestBackwardsRange() {
  Scene scene;
  scene.props = {100, 0, 1};
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(Near(scene.value, 51), "a backwards range is put in order");
}

// With the mouse, pressing on the track sets the value under the pointer, and dragging follows
// it. Three quarters of the way along the knob's travel is 75.
void TestPointer() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.MoveMouse(LEFT_END + (RIGHT_END - LEFT_END) * 0.75f, SLIDER.y + 22, draw);
  nav.MouseButton(true);
  nav.Frames(2, draw);
  Check(Near(scene.value, 75), "pressing on the track sets the value there");
  nav.MoveMouse(LEFT_END + (RIGHT_END - LEFT_END) * 0.25f, SLIDER.y + 22, draw);
  nav.Frame(draw);
  Check(Near(scene.value, 25), "dragging follows the pointer");
  nav.MouseButton(false);
  nav.Frame(draw);
}

// Confirm held on the gamepad doesn't let the mouse drag the knob, though the mouse moves over
// the track while it is held.
void TestGamepadHoldIgnoresTheMouse() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frame(draw);
  Check(scene.slider_active, "confirm holds the slider down");
  nav.MoveMouse(LEFT_END + 10, SLIDER.y + 22, draw);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frame(draw);
  Check(Near(scene.value, 50), "a gamepad hold doesn't let the mouse drag the knob");
}

// The knob sits at the value, and the fill runs from the track's start to the knob. At 25 the
// knob is a quarter of the way along its travel.
void TestKnobAndFill() {
  Scene scene;
  scene.value = 25;
  NavHarness nav;
  nav.Frames(2, DrawScene(scene));
  const float knob_x = LEFT_END + (RIGHT_END - LEFT_END) * 0.25f;
  const auto [knob_left, knob_right] = Span(DEFAULT_THEME.colors.control.knob);
  Check(Near((knob_left + knob_right) / 2, knob_x, 0.5f), "the knob sits at the value");
  const auto [fill_left, fill_right] = Span(DEFAULT_THEME.colors.accent.line);
  Check(Near(fill_left, SLIDER.x, 0.5f) && Near(fill_right, knob_x, 0.5f), "the fill runs from the start to the knob");
}

// The track wears a one pixel outline along its whole length and only its height, following its
// round ends, which a square corner would leave and a wider stroke would reach further in from. It
// comes after the track's own colour, which would hide it, and before the fill, which covers it up to
// the knob.
void TestTrackOutline() {
  constexpr float TRACK_HEIGHT = 8.0f;
  const auto &theme = DEFAULT_THEME;
  const ImU32 outline = ToImU32(theme.colors.border.strong);
  const ImU32 fill = ToImU32(theme.colors.accent.line);
  const ImU32 inset = ToImU32(theme.colors.surface.inset);
  const float centre = SLIDER.y + SLIDER.height / 2;
  const auto depth_in_track = [centre](ImVec2 point) {
    const float radius = TRACK_HEIGHT / 2;
    const float nearest_x = std::clamp(point.x, SLIDER.x + radius, SLIDER.Right() - radius);
    return radius - std::hypot(point.x - nearest_x, point.y - centre);
  };
  Scene scene;
  scene.value = 25;
  NavHarness nav;
  nav.Frames(2, DrawScene(scene));
  const auto [outline_left, outline_right] = Span(theme.colors.border.strong);
  Check(Near(outline_left, SLIDER.x, 0.5f) && Near(outline_right, SLIDER.Right(), 0.5f), "the outline runs the track's length");
  float top = std::numeric_limits<float>::max(), bottom = -top;
  int last_inset = -1, first_outline = -1, last_outline = -1, first_fill = -1;
  float deepest = 0.0f;
  const auto &vertices = Drawn();
  for (int index = 0; index < vertices.Size; ++index) {
    if (vertices[index].col == outline) {
      top = std::min(top, vertices[index].pos.y);
      bottom = std::max(bottom, vertices[index].pos.y);
      first_outline = first_outline < 0 ? index : first_outline;
      last_outline = index;
      const float depth = depth_in_track(vertices[index].pos);
      Check(depth >= -0.5f, "and follows its round ends");
      deepest = std::max(deepest, depth);
    } else if (vertices[index].col == inset) {
      last_inset = index;
    } else if (vertices[index].col == fill && first_fill < 0) {
      first_fill = index;
    }
  }
  Check(Near(deepest, theme.border.hair, 0.5f), "one pixel wide");
  Check(last_inset >= 0 && first_outline > last_inset, "over the track's own colour");
  Check(Near(top, centre - TRACK_HEIGHT / 2, 0.5f) && Near(bottom, centre + TRACK_HEIGHT / 2, 0.5f), "and only its height");
  Check(last_outline >= 0 && first_fill > last_outline, "and the fill is drawn over it");
}

// A disabled slider is solid grey with no opacity change. Its fill and knob are its own greys and sit
// where an enabled one's do, its knob has no shadow, its label is the disabled text colour, and none of
// the enabled colours or any faded colour is left. The same slider enabled is the baseline, which shows
// that a shadow and those colours are there to be missing.
void TestDisabledIsGrey() {
  const auto &theme = DEFAULT_THEME;
  const auto &colors = theme.colors;
  float enabled_knob = 0.0f;
  const auto draw_with = [&](bool disabled, auto &&check) {
    Scene scene;
    scene.value = 25;
    scene.props.disabled = disabled;
    scene.props.value_label = "25%";
    NavHarness nav;
    // Past the ring's fade, whose keyline shares the shadow's hue until it settles.
    constexpr int FRAMES_TO_SETTLE = 20;
    nav.Frames(FRAMES_TO_SETTLE, DrawScene(scene));
    check();
  };
  const auto shadow_drawn = [&] {
    const auto &vertices = Drawn();
    // Everything of that hue but the track's colour and the focus ring's keyline, which share it.
    return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &vertex) {
      return (vertex.col & ~IM_COL32_A_MASK) == (ToImU32(colors.effect.shadow) & ~IM_COL32_A_MASK) && vertex.col != ToImU32(colors.surface.inset) && (vertex.col & IM_COL32_A_MASK) != (ToImU32(colors.focus.keyline) & IM_COL32_A_MASK);
    });
  };
  const auto has = [](Color color) {
    const auto [left, right] = Span(color);
    return left <= right;
  };
  draw_with(false, [&] {
    Check(has(colors.accent.line) && has(colors.control.knob) && shadow_drawn(), "enabled, it has its colours and a shadow");
    Check(!has(colors.control.fill_disabled) && !has(colors.control.knob_disabled), "and no greys");
    const auto [left, right] = Span(colors.control.knob);
    enabled_knob = (left + right) / 2;
  });
  draw_with(true, [&] {
    const auto [fill_left, fill_right] = Span(colors.control.fill_disabled);
    // Up to the track's end, as the label past it is the same grey as the knob.
    const float track_right = Span(colors.surface.inset).second;
    float knob_left = std::numeric_limits<float>::max(), knob_right = -knob_left;
    for (const auto &vertex : Drawn()) {
      if (vertex.col == ToImU32(colors.control.knob_disabled) && vertex.pos.x <= track_right) {
        knob_left = std::min(knob_left, vertex.pos.x);
        knob_right = std::max(knob_right, vertex.pos.x);
      }
    }
    Check(fill_left <= fill_right && Near(fill_left, SLIDER.x, 0.5f), "disabled, the fill is grey from the track's start");
    Check(Near((knob_left + knob_right) / 2, enabled_knob, 0.5f), "and the knob is grey, where the enabled one is");
    Check(!has(colors.accent.line) && !has(colors.control.knob), "with none of the enabled colours");
    Check(!shadow_drawn(), "no shadow");
    Check(has(colors.border.strong) && !has(colors.border.strong.MulAlpha(theme.opacity.disabled)), "the outline at full strength, not faded");
    Check(has(colors.surface.inset) && !has(colors.surface.inset.MulAlpha(theme.opacity.disabled)), "and the track too");
    // Past the track, as the knob is the same grey as the label.
    Check(Span(colors.text.tertiary).second > track_right + 1.0f && !has(colors.text.secondary) && !has(colors.text.tertiary.MulAlpha(theme.opacity.disabled)), "and the label is the tertiary colour, at full strength");
  });
}

// A slider disabled while it has focus keeps a dimmed ring, once its focus highlight has settled, so
// focus isn't lost from view, though nothing else on it fades any more.
void TestDisabledRingIsDimmed() {
  constexpr int FRAMES_TO_SETTLE = 20;
  const auto &theme = DEFAULT_THEME;
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(FRAMES_TO_SETTLE, draw);
  // Wider than the knob, which is the same white.
  constexpr float KNOB = 28.0f;
  const auto [enabled_left, enabled_right] = Span(theme.colors.focus.ring);
  Check(enabled_right - enabled_left > KNOB, "focused, the ring is at full strength");
  scene.props.disabled = true;
  nav.Frames(FRAMES_TO_SETTLE, draw);
  Check(scene.slider_focused, "disabled, it keeps focus");
  const auto [dim_left, dim_right] = Span(theme.colors.focus.ring.MulAlpha(theme.opacity.disabled));
  Check(dim_left <= dim_right, "and the ring is dimmed");
  const auto [full_left, full_right] = Span(theme.colors.focus.ring);
  Check(full_left > full_right, "with none left at full strength");
}

// A track that would centre on a half pixel sits on a whole one, or its outline would spread over two
// rows at half strength. A height one pixel more than the usual is what puts it on the half pixel.
void TestTrackOnAWholePixel() {
  constexpr float ODD_HEIGHT = 45.0f;
  const Rect odd{SLIDER.x, SLIDER.y, SLIDER.width, ODD_HEIGHT};
  Scene scene;
  NavHarness nav;
  nav.Frames(2, [&](InteractContext &context) { DrawSlider(context, scene.rig, &scene.value, scene.props, odd); });
  const ImU32 outline = ToImU32(DEFAULT_THEME.colors.border.strong);
  float top = std::numeric_limits<float>::max();
  for (const auto &vertex : Drawn()) {
    if (vertex.col == outline) {
      top = std::min(top, vertex.pos.y);
    }
  }
  Check(top < SLIDER.Bottom(), "the outline is drawn");
  // Far less than the half pixel this is about, but more than the rounding in the curves' vertices.
  constexpr float WHOLE_SLACK = 0.01f;
  Check(std::fabs(top - std::round(top)) <= WHOLE_SLACK, "and its top row is whole");
}

// A focused slider rings its knob: the ring's keyline is centred on the knob and just wider than
// it. With the neighbour drawn first and focused, there is none.
void TestFocusRing() {
  Scene focused;
  NavHarness with_focus;
  with_focus.Frames(12, DrawScene(focused));
  const auto [left, right] = Span(DEFAULT_THEME.colors.focus.keyline);
  const float knob_x = (LEFT_END + RIGHT_END) / 2;
  Check(focused.slider_focused && Near((left + right) / 2, knob_x, 1.0f) && right - left > 28 && right - left < 44,
        "a focused slider rings its knob");
  float value = 50;
  Rig behind;
  NavHarness without_focus;
  without_focus.Frames(12, [&](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("neighbour"), NEIGHBOUR, false);
    DrawSlider(context, behind, &value, {0, 100, 1}, SLIDER);
  });
  const auto [none_left, none_right] = Span(DEFAULT_THEME.colors.focus.keyline);
  Check(none_left > none_right, "an unfocused one doesn't");
}

// The focus ring is drawn after the thumb, so it sits over it where they meet. The thumb's shadow is
// its first drawing and the ring's keyline another colour, and the shadow near the thumb comes
// before the first keyline vertex.
void TestRingIsOverTheThumb() {
  Scene scene;
  NavHarness nav;
  nav.Frames(12, DrawScene(scene));
  const ImU32 shadow = ToImU32(DEFAULT_THEME.colors.effect.shadow);
  const ImU32 keyline = ToImU32(DEFAULT_THEME.colors.focus.keyline);
  const auto &vertices = Drawn();
  int first_shadow = static_cast<int>(vertices.size());
  int first_keyline = static_cast<int>(vertices.size());
  for (int i = 0; i < vertices.size(); ++i) {
    const float from_centre = std::hypot(vertices[i].pos.x - (LEFT_END + RIGHT_END) / 2, vertices[i].pos.y - (SLIDER.y + SLIDER.height / 2));
    if (vertices[i].col == keyline) {
      first_keyline = std::min(first_keyline, i);
    } else if ((vertices[i].col & 0x00ffffffu) == (shadow & 0x00ffffffu) && from_centre <= SHADOW_REACH) {
      first_shadow = std::min(first_shadow, i);
    }
  }
  Check(first_shadow < static_cast<int>(vertices.size()) && first_keyline < static_cast<int>(vertices.size()) && first_shadow < first_keyline,
        "the ring is drawn after the thumb");
}

// The label's space is sized for the widest label, so the track keeps its length as the label
// changes. Without a reserve, a longer label shortens the track.
void TestLabelKeepsTheTrack() {
  const auto track_end = [](std::string_view label, std::string_view reserve) {
    float value = 50;
    Rig rig;
    NavHarness nav;
    nav.Frames(2, [&](InteractContext &context) {
      SliderProps props{0, 100, 1};
      props.value_label = label;
      props.label_reserve = reserve;
      DrawSlider(context, rig, &value, props, SLIDER);
    });
    return Span(DEFAULT_THEME.colors.surface.inset).second;
  };
  Check(track_end("9%", "") > track_end("100%", ""), "without a reserve, a longer label shortens the track");
  Check(Near(track_end("9%", "100%"), track_end("100%", "100%")), "with one, the track keeps its length");
  Check(track_end("100%", "100%") < SLIDER.Right() - 20, "and leaves room for the label");
}

// A label too wide to leave the knob room is left out. In the full-width slider the same label
// is drawn.
void TestNarrowDropsTheLabel() {
  const auto label_drawn = [](float width) {
    float value = 50;
    Rig rig;
    NavHarness nav;
    nav.Frames(2, [&value, &rig, width](InteractContext &context) {
      SliderProps props{0, 100, 1};
      props.value_label = "100%";
      DrawSlider(context, rig, &value, props, {100, 100, width, 44});
    });
    const auto [left, right] = Span(DEFAULT_THEME.colors.text.secondary);
    return left < right;
  };
  Check(label_drawn(SLIDER.width), "a label that fits is drawn");
  Check(!label_drawn(90), "a label that doesn't fit is left out");
}

// Pressing on the value label doesn't move the knob.
void TestLabelIgnoresPresses() {
  Scene scene;
  scene.props.value_label = "50%";
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.MoveMouse(SLIDER.Right() - 5, SLIDER.y + 22, draw);
  nav.MouseButton(true);
  nav.Frames(2, draw);
  nav.MouseButton(false);
  nav.Frame(draw);
  Check(Near(scene.value, 50), "pressing the label doesn't move the knob");
}

// A slider disabled while it has focus turns grey and never changes, not by R2 while it still
// has focus, nor by the mouse. It gives the D-pad back, so right moves focus to the neighbour.
void TestDisabled() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  scene.props.disabled = true;
  nav.Press(ImGuiKey_GamepadR2, draw);
  Check(scene.slider_focused && Near(scene.value, 50), "R2 doesn't move a disabled slider that has focus");
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(scene.neighbour_focused && Near(scene.value, 50), "right moves focus on instead");
  nav.MoveMouse(LEFT_END, SLIDER.y + 22, draw);
  nav.MouseButton(true);
  nav.Frames(2, draw);
  nav.MouseButton(false);
  nav.Frame(draw);
  Check(Near(scene.value, 50) && scene.changes == 0, "nor does the mouse");
  const auto [left, right] = Span(DEFAULT_THEME.colors.control.knob_disabled);
  Check(left < right, "and its knob is grey");
}

// A slider disabled while right is held gives the key back, so the held right then moves focus
// on to the neighbour as it repeats.
void TestDisabledGivesBackAHeldKey() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, true);
  nav.Frames(2, draw);
  Check(scene.value > 50 && scene.slider_focused, "the held right moves the slider");
  scene.props.disabled = true;
  nav.Frames(40, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, false);
  nav.Frame(draw);
  Check(scene.neighbour_focused, "disabled, the held right moves focus on to the neighbour");
}

// A slider with no value to show is refused and logged, and says nothing changed. One with a value
// is the baseline that builds in the same frame.
void TestNoValueIsRefused() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    ElementError reason = ElementError::None;
    bool changed = true;
    SliderProps props;
    props.changed = &changed;
    ForgetReportedRefusals();
    const SliderParts slider = Slider(frame, "none", nullptr, props, {}, &reason);
    Check(!ReportRefusal(ElementError::NoValue), "a slider with no value is logged");
    Check(slider.root.IsNone() && reason == ElementError::NoValue, "and is a none handle that says why");
    Check(!changed, "and reports no change");
    float value = 0.5f;
    Check(!Slider(frame, "some", &value, props).root.IsNone(), "where one with a value builds");
  });
}

} // namespace

int main() {
  TestNoValueIsRefused();
  TestSteps();
  TestChangesOnThePressFrame();
  TestChangeReportClearsWhenRenderIsSkipped();
  TestPages();
  TestHoldSpeeds();
  TestHoldSpeedsUp();
  TestNotANumber();
  TestSnapsToSteps();
  TestReachesTheTop();
  TestUnstepped();
  TestUnsteppedStopsAtTheEnds();
  TestBackwardsRange();
  TestPointer();
  TestGamepadHoldIgnoresTheMouse();
  TestKnobAndFill();
  TestTrackOutline();
  TestTrackOnAWholePixel();
  TestDisabledIsGrey();
  TestDisabledRingIsDimmed();
  TestFocusRing();
  TestRingIsOverTheThumb();
  TestLabelKeepsTheTrack();
  TestNarrowDropsTheLabel();
  TestLabelIgnoresPresses();
  TestDisabled();
  TestDisabledGivesBackAHeldKey();
  std::printf("System UI slider tests passed\n");
  return 0;
}
