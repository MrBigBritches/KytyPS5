#include "SystemUiNavHarness.h"
#include "graphics/presentation/system/imgui/interactContext.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace SystemUiTest;

constexpr float TOLERANCE = 0.001f;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiInteractionTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two values agree to within rounding noise.
bool Near(float actual, float expected) { return std::fabs(actual - expected) <= TOLERANCE; }

// A control for a test: where it is, whether it is disabled, and what it reported last frame.
struct Control {
  const char *name;
  Rect rect;
  bool disabled = false;
  Interaction last{};
  int activations = 0;
};

// Draws every control in `controls`, recording what each reported.
auto DrawAll(std::vector<Control> &controls) {
  return [&controls](InteractContext &context) {
    for (auto &control : controls) {
      control.last = Interact(context, ImGui::GetID(control.name), control.rect, control.disabled);
      control.activations += control.last.state.activated ? 1 : 0;
    }
  };
}

// Three buttons stacked down the screen.
std::vector<Control> Column() {
  return {{"a", {100, 100, 200, 50}}, {"b", {100, 200, 200, 50}}, {"c", {100, 300, 200, 50}}};
}

// The name of the control that has focus, or "none".
const char *Focused(const std::vector<Control> &controls) {
  for (const auto &control : controls) {
    if (control.last.state.focused) {
      return control.name;
    }
  }
  return "none";
}

// A window that appears focuses its first control, so the player can act without a press first.
// The D-pad then moves focus between controls by position.
void TestDpadMovesFocus() {
  NavHarness nav;
  auto controls = Column();
  const auto draw = DrawAll(controls);
  nav.Frames(2, draw);
  Check(std::string_view(Focused(controls)) == "a", "the first control has focus when the window appears");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(std::string_view(Focused(controls)) == "b", "down moves to the control below");
  nav.Press(ImGuiKey_GamepadDpadUp, draw);
  Check(std::string_view(Focused(controls)) == "a", "up moves back");
}

// Moving right from the top left of a grid picks the control beside it, not the one below, so
// navigation follows each control's real rectangle.
void TestGridNavigation() {
  NavHarness nav;
  std::vector<Control> controls{{"top-left", {100, 100, 200, 50}},
                                {"top-right", {400, 100, 200, 50}},
                                {"bottom-left", {100, 200, 200, 50}},
                                {"bottom-right", {400, 200, 200, 50}}};
  const auto draw = DrawAll(controls);
  nav.Frames(2, draw);
  Check(std::string_view(Focused(controls)) == "top-left", "the grid starts at the top left");
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(std::string_view(Focused(controls)) == "top-right", "right moves to the control beside it");
}

// A disabled control is skipped. With the middle control disabled, down goes from the first to
// the third.
void TestDisabledIsSkipped() {
  NavHarness nav;
  auto controls = Column();
  controls[1].disabled = true;
  const auto draw = DrawAll(controls);
  nav.Frames(2, draw);
  Check(std::string_view(Focused(controls)) == "a", "the first control starts with focus");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(std::string_view(Focused(controls)) == "c", "down skips the disabled control");
}

// The confirm button activates the focused control once per press, on the press. A disabled
// control under the mouse never activates when clicked.
void TestActivation() {
  NavHarness nav;
  auto controls = Column();
  controls[2].disabled = true;
  const auto draw = DrawAll(controls);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frame(draw);
  Check(controls[0].activations == 1, "confirm activates the focused control on the press");
  // Held past the repeat delay, it still activates only once.
  nav.Frames(30, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(3, draw);
  Check(controls[0].activations == 1 && controls[1].activations == 0, "once however long it is held, and only that control");
  nav.MoveMouse(150, 320, draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(2, draw);
  Check(controls[2].activations == 0, "a disabled control never activates");
}

// A posed control shows its pose whatever the input, navigation skips it, and a click on it does
// nothing. The same column unposed is the baseline, where down reaches the middle control, a
// click on it activates it, and it shows no highlights while it doesn't have focus.
void TestPose() {
  const ItemVisuals pose{0.25f, 1.0f, 0.5f};
  for (const bool posed : {false, true}) {
    NavHarness nav;
    auto controls = Column();
    const auto draw = [&controls, &pose, posed](InteractContext &context) {
      for (size_t index = 0; index < controls.size(); index++) {
        context.pose = (posed && index == 1) ? &pose : nullptr;
        controls[index].last = Interact(context, ImGui::GetID(controls[index].name), controls[index].rect, false);
        controls[index].activations += controls[index].last.state.activated ? 1 : 0;
      }
      context.pose = nullptr;
    };
    nav.Frames(2, draw);
    const ItemVisuals &shown = controls[1].last.visuals;
    Check(posed == (Near(shown.hover, 0.25f) && Near(shown.focus, 1.0f) && Near(shown.press, 0.5f)),
          posed ? "a posed control shows its pose" : "the baseline shows no highlights");
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
    Check(std::string_view(Focused(controls)) == (posed ? "c" : "b"), posed ? "navigation skips a posed control" : "the baseline reaches it");
    nav.MoveMouse(150, 220, draw);
    nav.MouseButton(true);
    nav.Frame(draw);
    nav.MouseButton(false);
    nav.Frames(2, draw);
    Check((controls[1].activations == 0) == posed, posed ? "a click on a posed control does nothing" : "the baseline activates");
  }
}

// Holding confirm on the focused control shows its press highlight, which falls away again after
// release. The control below, not pressed, is the baseline that stays at rest.
void TestPressHighlight() {
  NavHarness nav;
  auto controls = Column();
  const auto draw = DrawAll(controls);
  nav.Frames(2, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frames(10, draw);
  Check(controls[0].last.state.held && Near(controls[0].last.visuals.press, 1), "a held control shows its press");
  Check(Near(controls[1].last.visuals.press, 0), "a control that isn't held doesn't");
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(10, draw);
  Check(Near(controls[0].last.visuals.press, 0), "the press falls away after release");
}

// The input mode follows the last input. It starts on the pad, moves to the pointer when the
// mouse moves, and back on the next D-pad press.
void TestInputMode() {
  NavHarness nav;
  auto controls = Column();
  const auto draw = DrawAll(controls);
  nav.Frames(2, draw);
  Check(CurrentInputMode() == InputMode::Pad, "the UI starts in pad mode");
  nav.MoveMouse(500, 500, draw);
  Check(CurrentInputMode() == InputMode::Pointer, "moving the mouse switches to the pointer");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(CurrentInputMode() == InputMode::Pad, "the D-pad switches back");
}

// Hover shows only for the pointer and the focus ring only for the pad, each fading in over 100
// ms. With the mouse over the focused control, hover rises and focus falls away.
void TestHighlightsFollowTheMode() {
  NavHarness nav;
  auto controls = Column();
  const auto draw = DrawAll(controls);
  nav.Frames(10, draw);
  Check(Near(controls[0].last.visuals.focus, 1) && Near(controls[0].last.visuals.hover, 0),
        "with the pad, the focused control shows its ring and no hover");
  nav.MoveMouse(150, 120, draw);
  nav.Frames(10, draw);
  Check(Near(controls[0].last.visuals.hover, 1) && Near(controls[0].last.visuals.focus, 0),
        "with the pointer over it, it shows hover and no ring");
}

// The store's highlights fade rather than snap. Half way through the 100 ms focus-in, the ring is
// partly shown and the store asks for frames, then rests once it is fully shown.
void TestHighlightsFade() {
  InteractionStore store;
  const auto at = [](double ms) {
    return UiClock::time_point{} + std::chrono::duration_cast<UiClock::duration>(std::chrono::duration<double, std::milli>(ms));
  };
  const ItemState focused{false, true, false, false, false};
  store.Update(1, ItemState{}, InputMode::Pad, DEFAULT_THEME, at(0));
  const float half = store.Update(1, focused, InputMode::Pad, DEFAULT_THEME, at(1000)).focus;
  Check(half < TOLERANCE && store.Animating(at(1050)), "focus starts fading in and the store asks for frames");
  const float later = store.Update(1, focused, InputMode::Pad, DEFAULT_THEME, at(1050)).focus;
  Check(later > 0.5f && later < 1.0f, "half way it is partly shown");
  Check(Near(store.Update(1, focused, InputMode::Pad, DEFAULT_THEME, at(1100)).focus, 1) && !store.Animating(at(1100)),
        "then fully shown, and the store rests");
}

// The store shows hover only in pointer mode, whatever it is told, so a caller that reports hover
// under the gamepad still draws none. Pointer mode is the baseline that does show it.
void TestStoreHoverNeedsThePointer() {
  const ItemState hovered{true, false, false, false, false};
  const auto later = UiClock::time_point{} + std::chrono::seconds(1);
  InteractionStore pad;
  InteractionStore pointer;
  pad.Update(1, hovered, InputMode::Pad, DEFAULT_THEME, UiClock::time_point{});
  pointer.Update(1, hovered, InputMode::Pointer, DEFAULT_THEME, UiClock::time_point{});
  Check(Near(pointer.Update(1, hovered, InputMode::Pointer, DEFAULT_THEME, later).hover, 1), "the pointer shows hover");
  Check(Near(pad.Update(1, hovered, InputMode::Pad, DEFAULT_THEME, later).hover, 0), "the pad doesn't");
}

// A disabled control shows no press even while held, but keeps its focus ring if it still has
// focus, as it does when it is disabled while focused, so the player can see where focus is. The
// control draws that ring faded with the rest of it.
void TestDisabledKeepsFocusButNoPress() {
  InteractionStore store;
  const auto now = UiClock::time_point{};
  const ItemState everything{true, true, true, false, true};
  store.Update(1, everything, InputMode::Pad, DEFAULT_THEME, now);
  const auto visuals = store.Update(1, everything, InputMode::Pad, DEFAULT_THEME, now + std::chrono::seconds(1));
  Check(Near(visuals.press, 0), "a disabled control shows no press");
  Check(Near(visuals.focus, 1), "but keeps its focus ring while it has focus");
}

// Under the pointer, an enabled control shows hover while a disabled one doesn't.
void TestDisabledShowsNoHover() {
  const ItemState hovered{true, false, false, false, false};
  const ItemState disabled_hovered{true, false, false, false, true};
  const auto later = UiClock::time_point{} + std::chrono::seconds(1);
  InteractionStore enabled;
  InteractionStore disabled;
  enabled.Update(1, hovered, InputMode::Pointer, DEFAULT_THEME, UiClock::time_point{});
  disabled.Update(1, disabled_hovered, InputMode::Pointer, DEFAULT_THEME, UiClock::time_point{});
  Check(Near(enabled.Update(1, hovered, InputMode::Pointer, DEFAULT_THEME, later).hover, 1), "an enabled control shows hover");
  Check(Near(disabled.Update(1, disabled_hovered, InputMode::Pointer, DEFAULT_THEME, later).hover, 0),
        "a disabled one doesn't");
}

// The gamepad's focus never counts as hover. ImGui itself reports the focused item as hovered
// after a D-pad move, so the move here is what would show it.
void TestFocusIsNotHover() {
  NavHarness nav;
  auto controls = Column();
  const auto draw = DrawAll(controls);
  nav.Frames(3, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(controls[1].last.state.focused && !controls[1].last.state.hovered, "a control focused by the D-pad isn't hovered");
}

// A held D-pad moves focus once, waits the theme's 400 ms, then moves every 80 ms. ImGui's own
// pace, 198 ms then 40 ms, is the baseline, which has moved further by the same time.
void TestHeldDirectionPace() {
  const auto moves_after = [](bool configured, int frames) {
    NavHarness nav;
    if (configured) {
      ConfigureNavigation(DEFAULT_THEME);
    }
    std::vector<Control> controls;
    std::vector<std::string> names;
    for (int index = 0; index < 40; index++) {
      names.push_back("row" + std::to_string(index));
    }
    for (int index = 0; index < 40; index++) {
      controls.push_back({names[index].c_str(), {100, 20.0f + 25.0f * index, 200, 20}});
    }
    const auto draw = DrawAll(controls);
    nav.Frames(2, draw);
    nav.Key(ImGuiKey_GamepadDpadDown, true);
    nav.Frames(frames, draw);
    for (int index = 0; index < 40; index++) {
      if (controls[index].last.state.focused) {
        return index;
      }
    }
    return -1;
  };
  // Frames at 60 a second, one more frame for a move to land: 367 ms is before the second move,
  // 440 ms after it, and 700 ms after the moves at 400, 480, 560 and 640.
  Check(moves_after(true, 23) == 1, "a held D-pad moves once, then waits");
  Check(moves_after(true, 27) == 2, "after 400 ms it moves again");
  Check(moves_after(true, 43) == 5, "then every 80 ms");
  Check(moves_after(false, 23) > 1, "ImGui's own pace would already have repeated");
}

// A control below the window's edge makes the window scrollable, so moving focus to it scrolls
// it into view. Without the controls reaching the window's content, the window could never
// scroll.
void TestFocusScrollsTheWindow() {
  NavHarness nav;
  std::vector<std::string> names;
  for (int index = 0; index < 20; index++) {
    names.push_back("row" + std::to_string(index));
  }
  std::vector<Control> controls;
  const auto draw = [&](InteractContext &context) {
    const float top = ImGui::GetWindowPos().y - ImGui::GetScrollY();
    controls.clear();
    for (int index = 0; index < 20; index++) {
      controls.push_back({names[index].c_str(), {100, top + 100.0f + 150.0f * index, 200, 50}});
    }
    for (auto &control : controls) {
      control.last = Interact(context, ImGui::GetID(control.name), control.rect, false);
    }
  };
  nav.Frames(2, draw);
  for (int press = 0; press < 19; press++) {
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
  }
  nav.Frames(10, draw);
  Check(controls[19].last.state.focused, "focus reaches the last control");
  Check(ImGui::FindWindowByName("view")->Scroll.y > 0.0f, "and the window has scrolled to it");
}

// A control not drawn for a frame is forgotten, so when it comes back its highlight starts at
// rest instead of where it was. The baseline keeps drawing it and stays fully shown.
void TestCollectForgets() {
  const ItemState focused{false, true, false, false, false};
  const auto at = [](int ms) { return UiClock::time_point{} + std::chrono::milliseconds(ms); };
  InteractionStore kept;
  InteractionStore dropped;
  for (auto *store : {&kept, &dropped}) {
    store->Update(1, focused, InputMode::Pad, DEFAULT_THEME, at(0));
    store->Update(1, focused, InputMode::Pad, DEFAULT_THEME, at(1000));
    store->Collect();
  }
  kept.Update(1, focused, InputMode::Pad, DEFAULT_THEME, at(1016));
  kept.Collect();
  dropped.Collect();
  Check(Near(kept.Update(1, focused, InputMode::Pad, DEFAULT_THEME, at(1032)).focus, 1), "a control drawn every frame stays shown");
  Check(dropped.Update(1, focused, InputMode::Pad, DEFAULT_THEME, at(1032)).focus < 0.5f,
        "a control missing for a frame starts again from rest");
}

// The navigation settings free Square from ImGui's window switcher. TestHeldDirectionPace covers
// their repeat pace.
void TestConfigureNavigation() {
  NavHarness nav;
  Check(GImGui->ConfigNavWindowingWithGamepad, "ImGui's window switcher starts on Square");
  ConfigureNavigation(DEFAULT_THEME);
  Check(!GImGui->ConfigNavWindowingWithGamepad, "Square no longer opens the window switcher");
}

// A harness's options reach its frames. The output size sets the display and the UI's scale, the
// loaded fonts reach the UI, and the system UI's navigation settings are applied. The changed
// size is not 16:9, so a height left at the default shows. A harness with the defaults is the
// baseline for each.
void TestHarnessOptions() {
  for (const bool changed : {false, true}) {
    HarnessOptions options;
    UiFonts loaded;
    if (changed) {
      options.width = 3440;
      options.height = 1440;
      options.load_fonts = [&loaded](ImFontAtlas &atlas) {
        loaded = {atlas.AddFontDefault(), atlas.AddFontDefault()};
        return loaded;
      };
      options.configure_navigation = true;
    }
    NavHarness nav(options);
    Rect screen;
    UiFonts fonts;
    nav.Frame([&](InteractContext &context) {
      screen = context.ui.scale.Screen();
      fonts = context.ui.fonts;
    });
    const auto &io = ImGui::GetIO();
    Check(Near(io.DisplaySize.x, changed ? 3440 : 1920) && Near(io.DisplaySize.y, changed ? 1440 : 1080) &&
              Near(screen.width, io.DisplaySize.x) && Near(screen.height, io.DisplaySize.y),
          changed ? "the output size sets the display and the scale" : "the default output is 1080p");
    const bool fonts_reach = fonts.regular != nullptr && fonts.regular == loaded.regular &&
                             fonts.bold != nullptr && fonts.bold == loaded.bold && fonts.bold != fonts.regular;
    Check(changed ? fonts_reach : (fonts.regular == nullptr && fonts.bold == nullptr),
          changed ? "both loaded fonts reach the UI" : "by default the UI has no fonts");
    Check(GImGui->ConfigNavWindowingWithGamepad != changed && io.ConfigNavEscapeClearFocusItem != changed,
          changed ? "the system UI's navigation settings are applied" : "by default ImGui's own settings stay");
  }
}

} // namespace

int main() {
  TestDpadMovesFocus();
  TestGridNavigation();
  TestDisabledIsSkipped();
  TestActivation();
  TestPose();
  TestPressHighlight();
  TestInputMode();
  TestHighlightsFollowTheMode();
  TestHighlightsFade();
  TestStoreHoverNeedsThePointer();
  TestDisabledKeepsFocusButNoPress();
  TestDisabledShowsNoHover();
  TestFocusIsNotHover();
  TestHeldDirectionPace();
  TestFocusScrollsTheWindow();
  TestCollectForgets();
  TestConfigureNavigation();
  TestHarnessOptions();
  std::printf("System UI interaction tests passed\n");
  return 0;
}
