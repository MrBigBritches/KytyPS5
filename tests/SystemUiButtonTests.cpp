#include "SystemUiNavHarness.h"

#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using namespace SystemUiTest;

constexpr float TOLERANCE = 0.01f;
constexpr Rect BUTTON{400, 300, 240, 56};
// A control above the button that takes the window's first focus, so the button rests unfocused.
constexpr Rect DECOY{400, 100, 240, 56};

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiButtonTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two lengths agree to within rounding noise.
bool Near(float actual, float expected) { return std::fabs(actual - expected) <= TOLERANCE; }

// The vertices the view window drew in its last frame.
const ImVector<ImDrawVert> &Drawn() { return ImGui::FindWindowByName("view")->DrawList->VtxBuffer; }

// True when some vertex of the last frame has exactly `color`.
bool HasColor(Color color) {
  const auto &vertices = Drawn();
  return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &v) { return v.col == ToImU32(color); });
}

// The smallest rectangle around the last frame's vertices of exactly `color`.
Rect Bounds(Color color) {
  float left = std::numeric_limits<float>::max(), top = left, right = -left, bottom = -left;
  for (const auto &vertex : Drawn()) {
    if (vertex.col == ToImU32(color)) {
      left = std::min(left, vertex.pos.x);
      top = std::min(top, vertex.pos.y);
      right = std::max(right, vertex.pos.x);
      bottom = std::max(bottom, vertex.pos.y);
    }
  }
  return {left, top, right - left, bottom - top};
}

// Counts the activations an OnActivate callback reports.
struct CountActivations {
  int *count;
  void operator()(Element) const { ++*count; }
};

// A button element drawn each frame, and what it reported of its activations in three ways: the
// result of Render, InputOf and an OnActivate callback. The three must always agree.
struct Rig {
  UiFrame frame;
  Element button;
  int rendered = 0;
  int inputs = 0;
  int callbacks = 0;
};

// Builds `props` as a button in `rect` and renders it in this frame's window. A specimen shows
// the button in a fixed pose, with no input taken.
void DrawButton(InteractContext &context, Rig &rig, const ButtonProps &props, const Rect &rect, const ItemVisuals *specimen = nullptr) {
  rig.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
  rig.button = Button(rig.frame, "button", props);
  Check(!rig.button.IsNone(), "the button was built");
  (void)OnActivate(rig.frame, rig.button, CountActivations{&rig.callbacks});
  if (specimen != nullptr) {
    (void)Specimen(rig.frame, rig.button, *specimen);
  }
  const float unlimited = std::numeric_limits<float>::infinity();
  (void)Measure(rig.frame, rig.button, {unlimited, unlimited});
  Check(Arrange(rig.frame, rig.button, rect), "the button was arranged");
  const RenderResult result = Render(rig.frame, rig.button, context.list);
  Check(!result.error.has_value(), "the button rendered");
  rig.rendered += result.activated.IsNone() ? 0 : 1;
  const auto input = InputOf(rig.frame, rig.button);
  rig.inputs += input && input->interaction.state.activated ? 1 : 0;
}

// Draws a decoy control that takes first focus, then a button with `props`.
auto DecoyThen(Rig &rig, const ButtonProps &props) {
  return [&rig, props](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("decoy"), DECOY, false);
    DrawButton(context, rig, props, BUTTON);
  };
}

// Draws the button alone, so it takes the first focus.
auto Alone(Rig &rig, const ButtonProps &props) {
  return [&rig, props](InteractContext &context) { DrawButton(context, rig, props, BUTTON); };
}

// Each variant rests in its own fill. The variants are told apart by colours none of the others
// draw, so a variant mapped to the wrong colours fails.
void TestVariantFills() {
  const auto &colors = DEFAULT_THEME.colors;
  struct Case {
    ButtonVariant variant;
    Color fill;
  };
  const Case cases[] = {{ButtonVariant::Primary, colors.accent.fill},
                        {ButtonVariant::Secondary, colors.surface.overlay},
                        {ButtonVariant::Danger, colors.status.danger_fill}};
  for (const auto &[variant, fill] : cases) {
    NavHarness nav;
    Rig rig;
    nav.Frames(3, DecoyThen(rig, {"Save", variant}));
    for (const auto &[other_variant, other_fill] : cases) {
      Check(HasColor(other_fill) == (other_variant == variant), "each variant rests in its own fill and no other");
    }
  }
  NavHarness nav;
  Rig rig;
  nav.Frames(3, DecoyThen(rig, {"Cancel", ButtonVariant::Ghost}));
  for (const auto &[variant, fill] : cases) {
    Check(!HasColor(fill), "a ghost button has no fill");
  }
}

// True when some vertex of exactly `color` lies outside `rect`. The ring shares its colour with
// the label on an accent button, so only its place outside the edge tells it apart.
bool ColorOutside(Color color, const Rect &rect) {
  const auto &vertices = Drawn();
  return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &v) {
    return v.col == ToImU32(color) && (v.pos.x < rect.x - TOLERANCE || v.pos.x > rect.Right() + TOLERANCE ||
                                       v.pos.y < rect.y - TOLERANCE || v.pos.y > rect.Bottom() + TOLERANCE);
  });
}

// A focused primary button turns to its hover fill and wears the focus ring outside its edge. The
// same button unfocused, behind the decoy, is the baseline with neither.
void TestFocusedPrimary() {
  const auto &colors = DEFAULT_THEME.colors;
  NavHarness unfocused;
  Rig behind;
  unfocused.Frames(12, DecoyThen(behind, {"Save", ButtonVariant::Primary}));
  Check(!HasColor(colors.accent.fill_hover), "unfocused, it has no hover fill");
  Check(!ColorOutside(colors.focus.ring, BUTTON), "or ring");
  NavHarness focused;
  Rig alone;
  focused.Frames(12, Alone(alone, {"Save", ButtonVariant::Primary}));
  Check(HasColor(colors.accent.fill_hover), "focused, it turns to its hover fill");
  const Rect ring = Bounds(colors.focus.ring);
  Check(ColorOutside(colors.focus.ring, BUTTON) && ring.x < BUTTON.x && ring.Right() > BUTTON.Right(),
        "and its ring is outside its edge");
}

// A button posed as focused wears its ring and hover fill though the decoy has focus, so a
// gallery can show the state without input. TestFocusedPrimary's unfocused button is the baseline.
void TestPosedFocus() {
  const auto &colors = DEFAULT_THEME.colors;
  const ItemVisuals focused{0.0f, 1.0f, 0.0f};
  NavHarness nav;
  Rig rig;
  nav.Frames(3, [&](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("decoy"), DECOY, false);
    DrawButton(context, rig, {"Save", ButtonVariant::Primary}, BUTTON, &focused);
  });
  Check(ColorOutside(colors.focus.ring, BUTTON) && HasColor(colors.accent.fill_hover),
        "a button posed as focused wears its ring and hover fill");
}

// While held, the button shrinks to 98% about its centre. Released, it is back to full size.
void TestPress() {
  const auto &fill = DEFAULT_THEME.colors.accent.fill_pressed;
  NavHarness nav;
  Rig rig;
  const auto draw = Alone(rig, {"Save", ButtonVariant::Primary});
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frames(12, draw);
  const Rect pressed = Bounds(fill);
  Check(Near(pressed.width, BUTTON.width * 0.98f) && Near(pressed.x + pressed.width / 2, BUTTON.x + BUTTON.width / 2),
        "a held button shrinks to 98% about its centre in its pressed fill");
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(12, draw);
  Check(!HasColor(fill), "released, it leaves its pressed fill");
}

// Clicks the middle of the button with the mouse over the frames `draw` draws.
template <class Draw> void ClickButton(NavHarness &nav, Draw &&draw) {
  nav.MoveMouse(BUTTON.x + 10, BUTTON.y + 10, draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(2, draw);
}

// A disabled button fades to 38% and never activates, by Render, by InputOf or by its callback. The
// same button enabled, which does activate, is the baseline.
void TestDisabled() {
  const auto &fill = DEFAULT_THEME.colors.accent.fill;
  NavHarness nav;
  Rig off;
  const auto disabled = DecoyThen(off, {"Save", ButtonVariant::Primary, ButtonSize::Medium, true});
  nav.Frames(3, disabled);
  Check(HasColor(fill.MulAlpha(DEFAULT_THEME.opacity.disabled)) && !HasColor(fill), "a disabled button fades to 38%");
  ClickButton(nav, disabled);
  Check(off.rendered == 0 && off.inputs == 0 && off.callbacks == 0, "and a click never activates it");
  NavHarness enabled_nav;
  Rig on;
  const auto enabled = DecoyThen(on, {"Save", ButtonVariant::Primary});
  ClickButton(enabled_nav, enabled);
  Check(on.rendered == 1 && on.inputs == 1 && on.callbacks == 1, "the same click activates an enabled button once");
}

// On a light surface a Ghost button's label is dark, even faded, and its hover and focus shade. The same
// button without the option is the baseline, and a variant with its own surface ignores it. The ring
// has white of its own, which is why the label is only checked at rest.
void TestGhostOnLight() {
  const auto &colors = DEFAULT_THEME.colors;
  const float faded = DEFAULT_THEME.opacity.disabled;
  const auto drawn = [](ButtonVariant variant, bool disabled, bool on_light, const ItemVisuals &pose, const std::function<void()> &check) {
    NavHarness nav;
    Rig rig;
    ButtonProps props{"Save", variant, ButtonSize::Medium, disabled};
    props.on_light = on_light;
    nav.Frames(3, [&](InteractContext &context) { DrawButton(context, rig, props, BUTTON, &pose); });
    check();
  };
  drawn(ButtonVariant::Ghost, false, true, {}, [&] {
    Check(HasColor(colors.text.on_light) && !HasColor(colors.text.primary), "on a light surface a Ghost's label is dark");
  });
  drawn(ButtonVariant::Ghost, false, false, {}, [&] {
    Check(HasColor(colors.text.primary) && !HasColor(colors.text.on_light), "and without the option it's the usual colour");
  });
  drawn(ButtonVariant::Ghost, true, true, {}, [&] {
    Check(HasColor(colors.text.on_light.MulAlpha(faded)) && !HasColor(colors.text.primary.MulAlpha(faded)), "disabled, the dark label fades");
  });
  drawn(ButtonVariant::Ghost, false, true, {.hover = 1.0f}, [&] {
    Check(HasColor(colors.state.hover_on_light) && !HasColor(colors.state.hover), "its hover shades");
  });
  drawn(ButtonVariant::Ghost, false, false, {.hover = 1.0f}, [&] {
    Check(HasColor(colors.state.hover) && !HasColor(colors.state.hover_on_light), "where without the option it lightens");
  });
  drawn(ButtonVariant::Ghost, false, true, {.focus = 1.0f}, [&] {
    Check(HasColor(colors.state.focus_on_light) && !HasColor(colors.state.focus), "and so does its focus");
  });
  drawn(ButtonVariant::Ghost, false, true, {.press = 1.0f}, [&] {
    Check(HasColor(colors.state.pressed), "and a press darkens, as it always does");
  });
  drawn(ButtonVariant::Secondary, false, true, {.hover = 1.0f}, [&] {
    Check(HasColor(colors.text.primary) && HasColor(colors.state.hover) && !HasColor(colors.text.on_light), "a variant with its own surface ignores the option");
  });
}

// Sizes follow the theme at the output's scale. At 4K each reference pixel is two, so a medium
// button is 112 high, and a short label still gets the 120 reference pixel minimum, 240 wide.
Vec2 MeasureAt(float width, float height, const ButtonProps &props) {
  const ScaleContext scale(DEFAULT_THEME.scale, width, height);
  InteractionStore store;
  UiFrame frame;
  frame.Begin(DEFAULT_THEME, scale, UiFonts{}, store, UiClock::now());
  const float unlimited = std::numeric_limits<float>::infinity();
  return Measure(frame, Button(frame, "button", props), {unlimited, unlimited});
}

void TestMeasure() {
  NavHarness nav;
  nav.Frame([](InteractContext &) {
    const Vec2 ok = MeasureAt(3840, 2160, {"OK"});
    Check(Near(ok.y, 112) && Near(ok.x, 240), "a short label gets the minimum width at the size's height");
    Check(Near(MeasureAt(3840, 2160, {"OK", ButtonVariant::Secondary, ButtonSize::Small}).y, 96) &&
              Near(MeasureAt(3840, 2160, {"OK", ButtonVariant::Secondary, ButtonSize::Large}).y, 128),
          "small and large have their own heights");
    const Vec2 wide = MeasureAt(3840, 2160, {"Delete all saved games and settings"});
    Check(wide.x > 240, "a long label widens the button past the minimum");
    Check(Near(MeasureAt(3840, 2160, {"OK"}).x, MeasureAt(1920, 1080, {"OK"}).x * 2), "and the minimum doubles with the output");
  });
}

// A button marked as the default focus takes the first focus when its window appears, ahead of a
// control above it. Without the mark, the first control would have it.
void TestDefaultFocus() {
  NavHarness nav;
  Rig rig;
  ItemId focused = 0;
  nav.Frames(3, [&](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("decoy"), DECOY, false);
    DrawButton(context, rig, {"Cancel", ButtonVariant::Secondary, ButtonSize::Medium, false, true}, BUTTON);
    focused = ImGui::GetFocusID();
  });
  Check(focused == IdOf(rig.frame, rig.button), "the default button takes the first focus");
}

// Confirm activates the focused button once, on the press, and reports it by all three routes.
// A mouse click does the same, and a button that was not focused is activated by it as well.
void TestActivates() {
  NavHarness nav;
  Rig rig;
  const auto draw = Alone(rig, {"Save", ButtonVariant::Primary});
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frame(draw);
  Check(rig.rendered == 1, "confirm activates the focused button on the press");
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(3, draw);
  Check(rig.rendered == 1 && rig.inputs == 1 && rig.callbacks == 1, "once, by Render, by InputOf and by the callback");
  ClickButton(nav, draw);
  Check(rig.rendered == 2 && rig.inputs == 2 && rig.callbacks == 2, "and a click activates it again, by all three");
}

// The vertices of exactly `color` inside `rect`, as a bounding rectangle.
Rect BoundsInside(Color color, const Rect &rect) {
  float left = std::numeric_limits<float>::max(), top = left, right = -left, bottom = -left;
  for (const auto &vertex : Drawn()) {
    if (vertex.col == ToImU32(color) && vertex.pos.x >= rect.x && vertex.pos.x <= rect.Right() && vertex.pos.y >= rect.y &&
        vertex.pos.y <= rect.Bottom()) {
      left = std::min(left, vertex.pos.x);
      top = std::min(top, vertex.pos.y);
      right = std::max(right, vertex.pos.x);
      bottom = std::max(bottom, vertex.pos.y);
    }
  }
  return {left, top, right - left, bottom - top};
}

// The label sits in the middle of the button, top to bottom and side to side.
void TestLabelCentred() {
  NavHarness nav;
  Rig rig;
  nav.Frames(3, DecoyThen(rig, {"Save", ButtonVariant::Primary}));
  const Rect label = Bounds(DEFAULT_THEME.colors.text.on_accent);
  Check(label.x > BUTTON.x && label.Right() < BUTTON.Right() && label.y > BUTTON.y && label.Bottom() < BUTTON.Bottom(),
        "the label is inside the button");
  Check(std::fabs((label.y + label.height / 2) - (BUTTON.y + BUTTON.height / 2)) < 3.0f &&
            std::fabs((label.x + label.width / 2) - (BUTTON.x + BUTTON.width / 2)) < 3.0f,
        "and centred in it");
}

// A button sized to its long label keeps the whole label while pressed, though the press shrinks
// it. The mouse presses it, so no focus ring shares the label's colour.
void TestPressKeepsTheLabel() {
  const ButtonProps props{"Delete all saved games and settings", ButtonVariant::Secondary};
  NavHarness nav;
  Rig rig;
  Rect rect{};
  const auto draw = [&](InteractContext &context) {
    const Vec2 size = MeasureAt(1920, 1080, props);
    rect = {300, 300, size.x, size.y};
    (void)Interact(context, ImGui::GetID("decoy"), DECOY, false);
    DrawButton(context, rig, props, rect);
  };
  nav.Frames(2, draw);
  nav.MoveMouse(310, 320, draw);
  nav.Frames(10, draw);
  const float resting = BoundsInside(DEFAULT_THEME.colors.text.primary, rect).width;
  nav.MouseButton(true);
  nav.Frames(10, draw);
  Check(HasColor(DEFAULT_THEME.colors.state.pressed), "the button is held");
  const float pressed = BoundsInside(DEFAULT_THEME.colors.text.primary, rect).width;
  Check(std::fabs(pressed - resting) < 1.0f, "a pressed button keeps its whole label");
  nav.MouseButton(false);
  nav.Frame(draw);
}

// A focused secondary button shows the theme's focus overlay, which an unfocused one, behind the
// decoy, doesn't.
void TestSecondaryFocusOverlay() {
  const ButtonProps props{"Options", ButtonVariant::Secondary};
  NavHarness unfocused;
  Rig behind;
  unfocused.Frames(12, DecoyThen(behind, props));
  Check(!HasColor(DEFAULT_THEME.colors.state.focus), "unfocused, it has no focus overlay");
  NavHarness focused;
  Rig alone;
  focused.Frames(12, Alone(alone, props));
  const Rect overlay = Bounds(DEFAULT_THEME.colors.state.focus);
  Check(HasColor(DEFAULT_THEME.colors.state.focus) && overlay.x >= BUTTON.x && overlay.Right() <= BUTTON.Right(),
        "focused, its focus overlay covers it");
}

// A secondary button has a border, and a ghost button none.
void TestBorders() {
  NavHarness secondary;
  Rig first;
  secondary.Frames(3, DecoyThen(first, {"Options", ButtonVariant::Secondary}));
  Check(HasColor(DEFAULT_THEME.colors.border.standard), "a secondary button has a border");
  NavHarness ghost;
  Rig second;
  ghost.Frames(3, DecoyThen(second, {"Options", ButtonVariant::Ghost}));
  Check(!HasColor(DEFAULT_THEME.colors.border.standard), "a ghost button has none");
}

// A resting ghost button draws nothing but its label.
void TestGhostDrawsOnlyItsLabel() {
  NavHarness nav;
  Rig rig;
  nav.Frames(3, DecoyThen(rig, {"Cancel", ButtonVariant::Ghost}));
  const ImU32 label = ToImU32(DEFAULT_THEME.colors.text.primary);
  for (const auto &vertex : Drawn()) {
    const bool inside = vertex.pos.x >= BUTTON.x && vertex.pos.x <= BUTTON.Right() && vertex.pos.y >= BUTTON.y &&
                        vertex.pos.y <= BUTTON.Bottom();
    Check(!inside || vertex.col == label, "a resting ghost button draws only its label");
  }
}

// A long label's button is exactly the label plus the size's padding on each side: 24 pixels
// for medium and 16 for small.
void TestMeasureIsExact() {
  NavHarness nav;
  nav.Frame([](InteractContext &context) {
    const std::string_view text = "Delete all saved games and settings";
    const TextProps label{text, DEFAULT_THEME.type.label, DEFAULT_THEME.colors.text.primary, TextAlign::Center};
    const float width = LayoutText(context.ui, label, std::numeric_limits<float>::max()).size.x;
    Check(Near(MeasureAt(1920, 1080, {text}).x, width + 48), "medium adds 24 pixels each side");
    Check(Near(MeasureAt(1920, 1080, {text, ButtonVariant::Secondary, ButtonSize::Small}).x, width + 32),
          "small adds 16 pixels each side");
  });
}

// A disabled button marked as the default focus leaves first focus to the control before it.
void TestDisabledDefaultFocus() {
  NavHarness nav;
  Rig rig;
  ItemId decoy = 0;
  ItemId focused = 0;
  nav.Frames(3, [&](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("decoy"), DECOY, false);
    DrawButton(context, rig, {"Cancel", ButtonVariant::Secondary, ButtonSize::Medium, true, true}, BUTTON);
    decoy = ImGui::GetID("decoy");
    focused = ImGui::GetFocusID();
  });
  Check(focused == decoy, "a disabled default button doesn't take first focus");
}

// A button disabled while it has focus keeps its ring, faded to 38% with the rest of it, so the
// player can still see where focus is. Enabled, the same button wears the ring at full strength.
void TestDisabledKeepsAFadedRing() {
  const auto &ring = DEFAULT_THEME.colors.focus.ring;
  bool disabled = false;
  NavHarness nav;
  Rig rig;
  const auto draw = [&](InteractContext &context) {
    DrawButton(context, rig, {"Save", ButtonVariant::Primary, ButtonSize::Medium, disabled}, BUTTON);
  };
  nav.Frames(12, draw);
  Check(ColorOutside(ring, BUTTON), "an enabled focused button wears its ring");
  disabled = true;
  nav.Frames(12, draw);
  Check(ColorOutside(ring.MulAlpha(DEFAULT_THEME.opacity.disabled), BUTTON) && !ColorOutside(ring, BUTTON),
        "disabled while focused, it keeps the ring faded to 38%");
}

// Fading the whole button with its screen is the caller's: Paint's opacity fades the fill, the
// border and the label together, as the old opacity property did.
void TestFadesWithItsScreen() {
  NavHarness nav;
  Rig rig;
  const float fade = 0.5f;
  nav.Frames(3, [&](InteractContext &context) {
    rig.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element button = Button(rig.frame, "button", {"Save", ButtonVariant::Secondary});
    const float unlimited = std::numeric_limits<float>::infinity();
    (void)Measure(rig.frame, button, {unlimited, unlimited});
    (void)Arrange(rig.frame, button, BUTTON);
    (void)Render(rig.frame, button, context.list, PaintState{.opacity = fade});
  });
  const auto &colors = DEFAULT_THEME.colors;
  Check(HasColor(colors.surface.overlay.MulAlpha(fade)) && HasColor(colors.border.standard.MulAlpha(fade)) &&
            HasColor(colors.text.primary.MulAlpha(fade)),
        "the fill, the border and the label fade together");
}

} // namespace

int main() {
  TestVariantFills();
  TestFocusedPrimary();
  TestPosedFocus();
  TestPress();
  TestDisabled();
  TestGhostOnLight();
  TestMeasure();
  TestDefaultFocus();
  TestActivates();
  TestLabelCentred();
  TestPressKeepsTheLabel();
  TestSecondaryFocusOverlay();
  TestBorders();
  TestGhostDrawsOnlyItsLabel();
  TestMeasureIsExact();
  TestDisabledDefaultFocus();
  TestDisabledKeepsAFadedRing();
  TestFadesWithItsScreen();
  std::printf("System UI button tests passed\n");
  return 0;
}
