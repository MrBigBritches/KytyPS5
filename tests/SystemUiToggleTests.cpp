#include "SystemUiNavHarness.h"

#include "graphics/presentation/system/components/toggle.h"
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
#include <tuple>

namespace {

using namespace SystemUiTest;

constexpr float TOLERANCE = 0.5f;
// A toggle at the reference size, and a control to its right.
constexpr Rect TOGGLE{100, 100, 64, 36};
constexpr Rect NEIGHBOUR{300, 100, 120, 36};
// Where the knob's centre rests at each end: the centres of the track's round ends.
constexpr float OFF_X = TOGGLE.x + TOGGLE.height / 2;
constexpr float ON_X = TOGGLE.Right() - TOGGLE.height / 2;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiToggleTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two lengths agree to within half a pixel.
bool Near(float actual, float expected) { return std::fabs(actual - expected) <= TOLERANCE; }

// The vertices the view window drew in its last frame.
const ImVector<ImDrawVert> &Drawn() { return ImGui::FindWindowByName("view")->DrawList->VtxBuffer; }

// True when some vertex of the last frame has exactly `color`.
bool HasColor(Color color) {
  const auto &vertices = Drawn();
  return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &v) { return v.col == ToImU32(color); });
}

// How far `point` is inside the toggle's pill-shaped track, negative outside.
float DepthInTrack(ImVec2 point) {
  const float radius = TOGGLE.height / 2;
  const float centre_y = TOGGLE.y + radius;
  const float nearest_x = std::clamp(point.x, TOGGLE.x + radius, TOGGLE.Right() - radius);
  return radius - std::hypot(point.x - nearest_x, point.y - centre_y);
}

// The knob's centre across the track in the last frame. The knob is white, as is the focus ring,
// whose inner edge follows the track's outline, so only white well inside the outline counts. The
// knob sits 4 pixels inside it at the least.
float KnobX() {
  const ImU32 knob = ToImU32(DEFAULT_THEME.colors.control.knob);
  float left = std::numeric_limits<float>::max(), right = -left;
  for (const auto &vertex : Drawn()) {
    if (vertex.col == knob && DepthInTrack(vertex.pos) > 2) {
      left = std::min(left, vertex.pos.x);
      right = std::max(right, vertex.pos.x);
    }
  }
  return (left + right) / 2;
}

// Counts the activations an OnActivate callback reports.
struct CountActivations {
  int *count;
  void operator()(Element) const { ++*count; }
};

// A toggle element drawn each frame, and what it reported of its activations in three ways: the
// result of Render, InputOf and an OnActivate callback. The three must always agree.
struct Rig {
  UiFrame frame;
  Element toggle;
  int rendered = 0;
  int inputs = 0;
  int callbacks = 0;
};

// Builds a toggle over `on` in `TOGGLE` and renders it in this frame's window.
void DrawToggle(InteractContext &context, Rig &rig, bool *on, bool disabled) {
  rig.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
  const ItemLayout size{.width = Dp(TOGGLE.width), .height = Dp(TOGGLE.height)};
  rig.toggle = Toggle(rig.frame, "toggle", on, {.disabled = disabled}, size);
  Check(!rig.toggle.IsNone(), "the toggle was built");
  (void)OnActivate(rig.frame, rig.toggle, CountActivations{&rig.callbacks});
  const float unlimited = std::numeric_limits<float>::infinity();
  (void)Measure(rig.frame, rig.toggle, {unlimited, unlimited});
  Check(Arrange(rig.frame, rig.toggle, TOGGLE), "the toggle was arranged");
  const RenderResult result = Render(rig.frame, rig.toggle, context.list);
  Check(!result.error.has_value(), "the toggle rendered");
  rig.rendered += result.activated.IsNone() ? 0 : 1;
  const auto input = InputOf(rig.frame, rig.toggle);
  rig.inputs += input && input->interaction.state.activated ? 1 : 0;
}

// A toggle with its state and change count, and the control beside it.
struct Scene {
  Rig rig;
  bool on = false;
  bool disabled = false;
  int changes = 0;
  bool toggle_focused = false;
  bool neighbour_focused = false;
};

// Draws the toggle then its neighbour, recording what happened in `scene`.
auto DrawScene(Scene &scene) {
  return [&scene](InteractContext &context) {
    const bool was = scene.on;
    DrawToggle(context, scene.rig, &scene.on, scene.disabled);
    scene.changes += scene.on != was ? 1 : 0;
    scene.toggle_focused = ImGui::GetFocusID() == IdOf(scene.rig.frame, scene.rig.toggle);
    scene.neighbour_focused = Interact(context, ImGui::GetID("neighbour"), NEIGHBOUR, false).state.focused;
  };
}

// Confirm flips the toggle each press, reporting each change once.
void TestConfirmFlips() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  Check(scene.toggle_focused, "the toggle starts with focus");
  nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(scene.on && scene.changes == 1, "confirm turns it on");
  nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(!scene.on && scene.changes == 2, "and off again");
  Check(scene.rig.rendered == 2 && scene.rig.inputs == 2 && scene.rig.callbacks == 2,
        "each press is an activation, by Render, by InputOf and by the callback");
}

// While the toggle has focus, right turns it on and left off, and focus stays on it. Right again
// changes nothing. A plain control in the toggle's place is the baseline, where right moves focus
// to the neighbour.
void TestLeftAndRight() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(scene.on && scene.toggle_focused && !scene.neighbour_focused, "right turns it on and keeps focus");
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(scene.on && scene.changes == 1, "right again changes nothing");
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(!scene.on && scene.changes == 2 && scene.toggle_focused, "left turns it off");
  Check(scene.rig.rendered == 0 && scene.rig.callbacks == 0, "a change by the D-pad is not an activation");
  NavHarness baseline;
  bool neighbour_focused = false;
  const auto plain = [&neighbour_focused](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("toggle"), TOGGLE, false);
    neighbour_focused = Interact(context, ImGui::GetID("neighbour"), NEIGHBOUR, false).state.focused;
  };
  baseline.Frames(2, plain);
  baseline.Press(ImGuiKey_GamepadDpadRight, plain);
  Check(neighbour_focused, "without a toggle, right moves focus to the neighbour");
}

// The knob slides with the overshooting Emphasis curve over 180 ms. Two frames after turning on it
// is about two thirds across, and it rests at the on end afterwards.
void TestKnobSlides() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  Check(Near(KnobX(), OFF_X), "an off toggle's knob rests at the left end");
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frame(draw);
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(2, draw);
  const float moving = KnobX();
  Check(moving > OFF_X + 2 && moving < ON_X - 2, "part way through it is between the ends");
  float furthest = moving;
  for (int frame = 0; frame < 15; frame++) {
    nav.Frame(draw);
    furthest = std::max(furthest, KnobX());
  }
  Check(furthest > ON_X + 1, "on the way it springs briefly past the end");
  Check(Near(KnobX(), ON_X), "then it rests at the right end");
}

// A toggle that is on when it first appears shows its knob at the on end straight away, rather
// than sliding there.
void TestStartsWhereItIs() {
  Scene scene;
  scene.on = true;
  NavHarness nav;
  nav.Frame(DrawScene(scene));
  Check(Near(KnobX(), ON_X), "a toggle that starts on has its knob at the on end at once");
}

// The track is grey when off and the accent fill when on, each without the other.
void TestTrackColours() {
  const auto &colors = DEFAULT_THEME.colors;
  Scene off;
  NavHarness off_nav;
  off_nav.Frames(2, DrawScene(off));
  Check(HasColor(colors.control.track_off) && !HasColor(colors.accent.fill), "an off track is grey");
  Scene on;
  on.on = true;
  NavHarness on_nav;
  on_nav.Frames(2, DrawScene(on));
  Check(HasColor(colors.accent.fill) && !HasColor(colors.control.track_off), "an on track is the accent fill");
}

// The span of the last frame's vertices of exactly `color`.
struct Extent {
  float left = std::numeric_limits<float>::max();
  float top = std::numeric_limits<float>::max();
  float right = -std::numeric_limits<float>::max();
  float bottom = -std::numeric_limits<float>::max();
};

// Finds the span of `color` in the last frame.
Extent ExtentOf(Color color) {
  Extent extent;
  for (const auto &vertex : Drawn()) {
    if (vertex.col == ToImU32(color)) {
      extent.left = std::min(extent.left, vertex.pos.x);
      extent.top = std::min(extent.top, vertex.pos.y);
      extent.right = std::max(extent.right, vertex.pos.x);
      extent.bottom = std::max(extent.bottom, vertex.pos.y);
    }
  }
  return extent;
}

// True when the last frame has a white vertex fainter than the outline and not fully clear. Only the
// outline can be that, part way through its fade, once the focus ring has settled at full strength.
bool HasFadingOutline() {
  const Color outline = DEFAULT_THEME.colors.border.strong;
  const auto &vertices = Drawn();
  return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &vertex) {
    const uint32_t alpha = (vertex.col & IM_COL32_A_MASK) >> IM_COL32_A_SHIFT;
    return (vertex.col & ~IM_COL32_A_MASK) == (ToImU32(outline) & ~IM_COL32_A_MASK) && alpha > 0 && alpha < outline.a;
  });
}

// Checks the outline at rest. It is a stroke one pixel wide that follows the pill, which a square
// corner would leave and a wider stroke or a fill would reach further in from. It is drawn over the
// track's fill, which would otherwise hide it, and under the knob.
void CheckOutlineShape() {
  const auto &colors = DEFAULT_THEME.colors;
  const ImU32 outline = ToImU32(colors.border.strong);
  const auto &vertices = Drawn();
  int last_track = -1, first_outline = -1, last_outline = -1, first_knob = -1;
  float deepest = 0.0f;
  for (int index = 0; index < vertices.Size; ++index) {
    const ImU32 col = vertices[index].col;
    if (col == ToImU32(colors.control.track_off)) {
      last_track = index;
    } else if (col == outline) {
      first_outline = first_outline < 0 ? index : first_outline;
      last_outline = index;
      const float depth = DepthInTrack(vertices[index].pos);
      Check(depth >= -TOLERANCE, "the outline follows the pill");
      deepest = std::max(deepest, depth);
    } else if (col == ToImU32(colors.control.knob) && first_knob < 0) {
      first_knob = index;
    }
  }
  Check(Near(deepest, DEFAULT_THEME.border.hair), "and is one pixel wide");
  Check(last_track >= 0 && first_outline > last_track, "over the track's fill");
  Check(last_outline >= 0 && first_knob > last_outline, "and under the knob");
}

// An off track wears an outline along the toggle's edge. An on track is the baseline, as its fill
// needs none, and the outline fades between the two rather than vanishing.
void TestOutlineFollowsTheKnob() {
  const Color outline = DEFAULT_THEME.colors.border.strong;
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  constexpr int FRAMES_TO_SETTLE = 20;
  nav.Frames(FRAMES_TO_SETTLE, draw);
  const Extent extent = ExtentOf(outline);
  Check(Near(extent.left, TOGGLE.x) && Near(extent.right, TOGGLE.Right()) && Near(extent.top, TOGGLE.y) && Near(extent.bottom, TOGGLE.Bottom()), "an off track's outline lies along its edge");
  Check(!HasFadingOutline(), "and is at full strength at rest");
  CheckOutlineShape();
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frame(draw);
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(2, draw);
  Check(HasFadingOutline(), "it fades as the knob slides");
  nav.Frames(FRAMES_TO_SETTLE, draw);
  Check(!HasColor(outline) && !HasFadingOutline(), "and an on track has none");
}

// A disabled toggle's outline fades with the rest of it.
void TestDisabledOutline() {
  const auto &theme = DEFAULT_THEME;
  Scene scene;
  scene.disabled = true;
  NavHarness nav;
  nav.Frames(2, DrawScene(scene));
  Check(HasColor(theme.colors.border.strong.MulAlpha(theme.opacity.disabled)) && !HasColor(theme.colors.border.strong), "a disabled toggle's outline fades to 38%");
}

// A disabled toggle fades to 38%. TestDisabledWhileFocused covers that it never changes, since
// a disabled toggle can't otherwise get focus.
void TestDisabled() {
  Scene scene;
  scene.disabled = true;
  NavHarness nav;
  nav.Frames(2, DrawScene(scene));
  Check(HasColor(DEFAULT_THEME.colors.control.track_off.MulAlpha(DEFAULT_THEME.opacity.disabled)),
        "a disabled toggle fades to 38%");
  const auto draw = DrawScene(scene);
  nav.MoveMouse(TOGGLE.x + 10, TOGGLE.y + 10, draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(2, draw);
  Check(!scene.on && scene.changes == 0 && scene.rig.rendered == 0 && scene.rig.callbacks == 0, "and a click never changes or activates it");
}

// The press frame is the frame the value changes and the callback runs, before the toggle paints,
// so none of them is a frame late.
void TestActivationOnThePressFrame() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frame(draw);
  Check(scene.on && scene.changes == 1 && scene.rig.rendered == 1 && scene.rig.inputs == 1 && scene.rig.callbacks == 1,
        "the value, Render, InputOf and the callback all report the press on the press frame");
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(3, draw);
}

// A toggle that becomes disabled while it has focus keeps ImGui's focus, but the D-pad no longer
// changes it. Enabled, the same press turns it on.
void TestDisabledWhileFocused() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  scene.disabled = true;
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(!scene.on && scene.changes == 0, "a toggle disabled while focused ignores the D-pad");
  scene.disabled = false;
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(scene.on, "enabled again, the same press turns it on");
}

// True when some white vertex of the last frame lies outside the toggle's track, which only the
// focus ring draws.
bool WhiteOutsideTrack() {
  const ImU32 white = ToImU32(DEFAULT_THEME.colors.focus.ring);
  const auto &vertices = Drawn();
  return std::any_of(vertices.begin(), vertices.end(),
                     [&](const ImDrawVert &v) { return v.col == white && DepthInTrack(v.pos) < -1; });
}

// A focused toggle wears the focus ring around its track. With the neighbour drawn first, so it
// takes the first focus, the toggle has none.
void TestFocusRing() {
  Scene focused;
  NavHarness with_focus;
  with_focus.Frames(12, DrawScene(focused));
  Check(focused.toggle_focused && WhiteOutsideTrack(), "a focused toggle wears the focus ring");
  bool on = false;
  Rig behind;
  NavHarness without_focus;
  without_focus.Frames(12, [&](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("neighbour"), NEIGHBOUR, false);
    DrawToggle(context, behind, &on, false);
  });
  Check(!WhiteOutsideTrack(), "an unfocused toggle has none");
}

// The focus ring is drawn after the knob, so it sits over it where they meet. The knob is white
// well inside the track and the ring white outside it, and every knob vertex comes before every
// ring vertex.
void TestRingIsOverTheKnob() {
  Scene scene;
  NavHarness nav;
  nav.Frames(12, DrawScene(scene));
  const ImU32 white = ToImU32(DEFAULT_THEME.colors.focus.ring);
  const auto &vertices = Drawn();
  int last_knob = -1;
  int first_ring = static_cast<int>(vertices.size());
  for (int i = 0; i < vertices.size(); ++i) {
    if (vertices[i].col != white) {
      continue;
    }
    if (DepthInTrack(vertices[i].pos) > 2) {
      last_knob = i;
    } else if (DepthInTrack(vertices[i].pos) < -1) {
      first_ring = std::min(first_ring, i);
    }
  }
  Check(last_knob >= 0 && first_ring < static_cast<int>(vertices.size()) && last_knob < first_ring, "the ring is drawn after the knob");
}

// A toggle without focus leaves the D-pad to navigation. With the neighbour to the toggle's left
// focused, right moves focus onto the toggle and doesn't switch it on.
void TestUnfocusedLeavesTheDpad() {
  constexpr Rect LEFT_NEIGHBOUR{0, 100, 60, 36};
  bool on = false;
  bool toggle_focused = false;
  Rig rig;
  NavHarness nav;
  const auto draw = [&](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("left"), LEFT_NEIGHBOUR, false);
    DrawToggle(context, rig, &on, false);
    toggle_focused = ImGui::GetFocusID() == IdOf(rig.frame, rig.toggle);
  };
  nav.Frames(2, draw);
  Check(!toggle_focused, "the left neighbour starts with focus");
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(toggle_focused && !on, "right moves focus onto the toggle without switching it");
}

// Right pressed on the very frame focus lands on the toggle switches it on and nothing else, though
// navigation has already read that press as a move. A plain control in the toggle's place is the
// baseline, where the same press does move focus on to the neighbour.
void TestPressOnTheLandingFrame() {
  static constexpr Rect ABOVE{100, 20, 64, 36};
  const auto landing_press = [](bool with_toggle) {
    bool on = false;
    bool landed_focused = false;
    bool neighbour_focused = false;
    Rig rig;
    NavHarness nav;
    const auto draw = [&](InteractContext &context) {
      (void)Interact(context, ImGui::GetID("above"), ABOVE, false);
      if (with_toggle) {
        DrawToggle(context, rig, &on, false);
        landed_focused = ImGui::GetFocusID() == IdOf(rig.frame, rig.toggle);
      } else {
        (void)Interact(context, ImGui::GetID("toggle"), TOGGLE, false);
        landed_focused = ImGui::GetFocusID() == ImGui::GetID("toggle");
      }
      neighbour_focused = Interact(context, ImGui::GetID("neighbour"), NEIGHBOUR, false).state.focused;
    };
    nav.Frames(2, draw);
    nav.Key(ImGuiKey_GamepadDpadDown, true);
    nav.Frame(draw);
    nav.Key(ImGuiKey_GamepadDpadDown, false);
    nav.Key(ImGuiKey_GamepadDpadRight, true);
    nav.Frame(draw);
    nav.Key(ImGuiKey_GamepadDpadRight, false);
    nav.Frames(3, draw);
    return std::tuple{on, landed_focused, neighbour_focused};
  };
  const auto [plain_on, plain_stayed, plain_moved] = landing_press(false);
  Check(plain_moved && !plain_stayed, "on a plain control the landing-frame press moves focus on");
  const auto [on, stayed, moved] = landing_press(true);
  Check(on && stayed && !moved, "on a toggle it switches the toggle and focus stays");
}

// A direction held while focus leaves the toggle is given back to navigation at once. Holding
// right, the player presses up to the control above, and the held right then moves focus from
// there as a held key repeats.
void TestHeldKeyIsGivenBack() {
  static constexpr Rect ABOVE{100, 20, 64, 36};
  static constexpr Rect ABOVE_RIGHT{300, 20, 120, 36};
  bool on = false;
  bool above_right_focused = false;
  Rig rig;
  NavHarness nav;
  const auto draw = [&](InteractContext &context) {
    DrawToggle(context, rig, &on, false);
    (void)Interact(context, ImGui::GetID("above"), ABOVE, false);
    above_right_focused = Interact(context, ImGui::GetID("above right"), ABOVE_RIGHT, false).state.focused;
  };
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, true);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadUp, draw);
  nav.Frames(40, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, false);
  nav.Frame(draw);
  Check(on && above_right_focused, "the held right moves focus on from the control above");
}

// A toggle disabled while right is held gives the key back, so the held right then moves focus
// on to the neighbour as it repeats.
void TestDisabledGivesBackAHeldKey() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, true);
  nav.Frames(2, draw);
  Check(scene.on && scene.toggle_focused, "the held right turns the toggle on");
  scene.disabled = true;
  nav.Frames(40, draw);
  nav.Key(ImGuiKey_GamepadDpadRight, false);
  nav.Frame(draw);
  Check(scene.neighbour_focused, "disabled, the held right moves focus on to the neighbour");
}

// A tree painted without Render has no ids, so the knobs can't be kept apart in the store. Each
// rests at its own end all the same, an off toggle at the left and an on one at the right, rather
// than sharing one tween.
void TestPaintedWithoutRenderRestsAtItsTargets() {
  constexpr Rect SECOND{100, 200, 64, 36};
  bool off = false;
  bool on = true;
  UiFrame frame;
  NavHarness nav;
  nav.Frames(2, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const ItemLayout size{.width = Dp(TOGGLE.width), .height = Dp(TOGGLE.height)};
    const Element first = Toggle(frame, "off", &off, {}, size);
    const Element second = Toggle(frame, "on", &on, {}, size);
    const float unlimited = std::numeric_limits<float>::infinity();
    for (const auto &[element, rect] : {std::pair{first, TOGGLE}, std::pair{second, SECOND}}) {
      (void)Measure(frame, element, {unlimited, unlimited});
      Check(Arrange(frame, element, rect), "the toggle was arranged");
      Paint(frame, element, context.list);
    }
  });
  const ImU32 knob = ToImU32(DEFAULT_THEME.colors.control.knob);
  float off_left = std::numeric_limits<float>::max(), off_right = -off_left, on_left = off_left, on_right = off_right;
  for (const auto &vertex : Drawn()) {
    if (vertex.col != knob) {
      continue;
    }
    const bool in_first = vertex.pos.y >= TOGGLE.y && vertex.pos.y <= TOGGLE.Bottom();
    float &left = in_first ? off_left : on_left;
    float &right = in_first ? off_right : on_right;
    left = std::min(left, vertex.pos.x);
    right = std::max(right, vertex.pos.x);
  }
  Check(Near((off_left + off_right) / 2, OFF_X), "the off toggle's knob rests at the left end");
  Check(Near((on_left + on_right) / 2, ON_X), "and the on toggle's at the right end");
  // Nothing was kept under the key of a node with no id. A control asked for there for the first
  // time starts at its target, and one that had an entry would be at its end already.
  const Track slide{DEFAULT_THEME.interaction.knob_ms, Ease::Emphasis};
  Check(nav.store.Animate(0, 0.5f, {slide, slide}, nav.Now()) == 0.5f, "and the store has no entry under key 0");
}

// The store's animated value starts at its first target, then eases to new ones and asks for
// frames while it moves.
void TestAnimate() {
  InteractionStore store;
  const TweenTracks tracks{{100, Ease::Linear}, {100, Ease::Linear}};
  const auto at = [](int ms) { return UiClock::time_point{} + std::chrono::milliseconds(ms); };
  Check(store.Animate(1, 1, tracks, at(0)) == 1.0f && !store.Animating(at(0)), "a new value starts at its target");
  store.Animate(1, 0, tracks, at(0));
  Check(store.Animating(at(50)), "a value moving to a new target asks for frames");
  Check(std::fabs(store.Animate(1, 0, tracks, at(50)) - 0.5f) < 0.001f, "and eases there");
}

// A toggle with no value to show is refused and logged, and one with a value is the baseline that
// builds in the same frame.
void TestNoValueIsRefused() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    ElementError reason = ElementError::None;
    bool changed = true;
    ForgetReportedRefusals();
    const Element toggle = Toggle(frame, "none", nullptr, {.changed = &changed}, {}, &reason);
    Check(!ReportRefusal(ElementError::NoValue), "a toggle with no value is logged");
    Check(toggle.IsNone() && reason == ElementError::NoValue, "and is a none handle that says why");
    Check(!changed, "and reports no change");
    bool on = false;
    Check(!Toggle(frame, "some", &on).IsNone(), "where one with a value builds");
  });
}

// The change flag is set on the frame of each change, by confirm and by the D-pad, and not by a
// direction that finds the toggle already there, which is the baseline that the flag is about the
// value and not the press. A frame that builds the toggle and never renders it reads false.
void TestChangedReportsEveryChange() {
  NavHarness nav;
  UiFrame frame;
  bool on = false;
  bool changed = true;
  int changes = 0;
  const auto draw = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const ItemLayout size{.width = Dp(TOGGLE.width), .height = Dp(TOGGLE.height)};
    const Element toggle = Toggle(frame, "toggle", &on, {.changed = &changed}, size);
    const float unlimited = std::numeric_limits<float>::infinity();
    (void)Measure(frame, toggle, {unlimited, unlimited});
    Check(Arrange(frame, toggle, TOGGLE), "the toggle was arranged");
    Check(!Render(frame, toggle, context.list).error.has_value(), "the toggle rendered");
    changes += changed ? 1 : 0;
  };
  nav.Frames(2, draw);
  Check(changes == 0, "nothing has changed at rest");
  nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(on && changes == 1, "confirm is a change");
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(on && changes == 1, "right when it is already on is not");
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(!on && changes == 2, "left is");
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(on && changes == 3, "and so is right when it is off");
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    changed = true;
    (void)Toggle(frame, "toggle", &on, {.changed = &changed});
    Check(!changed, "building it clears the flag, for a frame that never renders it");
  });
}

} // namespace

int main() {
  TestChangedReportsEveryChange();
  TestNoValueIsRefused();
  TestConfirmFlips();
  TestActivationOnThePressFrame();
  TestRingIsOverTheKnob();
  TestLeftAndRight();
  TestKnobSlides();
  TestStartsWhereItIs();
  TestTrackColours();
  TestOutlineFollowsTheKnob();
  TestDisabledOutline();
  TestDisabled();
  TestDisabledWhileFocused();
  TestFocusRing();
  TestPressOnTheLandingFrame();
  TestHeldKeyIsGivenBack();
  TestDisabledGivesBackAHeldKey();
  TestUnfocusedLeavesTheDpad();
  TestPaintedWithoutRenderRestsAtItsTargets();
  TestAnimate();
  std::printf("System UI toggle tests passed\n");
  return 0;
}
