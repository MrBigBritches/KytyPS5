#include "SystemUiNavHarness.h"
#include "SystemUiElementTestHooks.h"
#include "SystemUiTabsKit.h"

#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/element/frameInternals.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/imgui/convert.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace SystemUiTest;

constexpr Rect BAR{100, 50, 900, 56};
constexpr Rect BELOW{100, 300, 200, 56};
constexpr std::array<std::string_view, 3> LABELS{"Saves", "Auto-saves", "Settings"};
// A point over the third tab. Each tab is its bold 22 pixel label plus 24 pixels each side, so
// the third spans roughly 371 to 509.
constexpr float THIRD_TAB_X = 440;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiTabsTests: failed: %s\n", message);
    std::abort();
  }
}

// The vertices the view window drew in its last frame.
const ImVector<ImDrawVert> &Drawn() { return ImGui::FindWindowByName("view")->DrawList->VtxBuffer; }

// The horizontal extent of the last frame's vertices of exactly `color` between `top` and
// `bottom`, or an empty span when there are none.
std::pair<float, float> Span(Color color, float top, float bottom) {
  const ImU32 wanted = ToImU32(color);
  float left = std::numeric_limits<float>::max(), right = -left;
  for (const auto &vertex : Drawn()) {
    if (vertex.col == wanted && vertex.pos.y >= top && vertex.pos.y <= bottom) {
      left = std::min(left, vertex.pos.x);
      right = std::max(right, vertex.pos.x);
    }
  }
  return {left, right};
}

// The indicator's extent, in the bar's bottom rows.
std::pair<float, float> IndicatorSpan() { return Span(DEFAULT_THEME.colors.accent.line, BAR.Bottom() - 4, BAR.Bottom()); }

// A tab bar element with the shoulders and a control below it, recording what has focus.
struct Scene {
  int selected = 0;
  int changes = 0;
  bool below_first = false;
  bool below_focused = false;
  // Whether the bar is given the shoulders.
  bool shoulders = true;
  int focused_tab = -1;
  UiFrame frame;
};

// Draws the scene. With `below_first`, the control below is drawn first and takes first focus.
auto DrawScene(Scene &scene) {
  return [&scene](InteractContext &context) {
    if (scene.below_first) {
      scene.below_focused = Interact(context, ImGui::GetID("below"), BELOW, false).state.focused;
    }
    scene.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const auto outcome = DrawElementBar(scene.frame, "tabs", BAR, LABELS, scene.selected, scene.shoulders);
    scene.changes += outcome.changed ? 1 : 0;
    scene.focused_tab = outcome.focusedTab;
    if (!scene.below_first) {
      scene.below_focused = Interact(context, ImGui::GetID("below"), BELOW, false).state.focused;
    }
  };
}

// Clicks at `x`, `y`, with a frame for the hover to settle first.
template <class Draw> void Click(NavHarness &nav, float x, float y, Draw &&draw) {
  nav.MoveMouse(x, y, draw);
  nav.Frame(draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(2, draw);
}

// R1 and L1 choose the next and previous tab while focus stays on a control elsewhere, and stop at
// the ends.
void TestShoulders() {
  Scene scene;
  scene.below_first = true;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  Check(scene.below_focused, "the control below starts with focus");
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(scene.selected == 1 && scene.below_focused, "R1 chooses the next tab and focus stays put");
  nav.Press(ImGuiKey_GamepadL1, draw);
  nav.Press(ImGuiKey_GamepadL1, draw);
  Check(scene.selected == 0 && scene.changes == 2, "L1 goes back and stops at the first");
}

// A press the bar acts on is taken from a reader drawn after it in the same frame, one that takes a
// press whoever owns the key, as ImGui's own shortcuts do. Owning the key does not stop that reader,
// only the bar's consuming does. The bar steps once. The same frame without the bar is the
// baseline, where the reader sees the press.
void TestPressIsConsumed() {
  for (const bool with_bar : {true, false}) {
    Scene scene;
    scene.below_first = true;
    bool shown = false;
    bool later_saw_press = false;
    NavHarness nav;
    const auto scene_draw = DrawScene(scene);
    const auto draw = [&](InteractContext &context) {
      if (shown) {
        scene_draw(context);
      }
      later_saw_press = later_saw_press || ShoulderPresses(ANY_OWNER).right;
    };
    nav.Frames(2, draw);
    nav.Key(ImGuiKey_GamepadR1, true);
    shown = with_bar;
    nav.Frame(draw);
    nav.Key(ImGuiKey_GamepadR1, false);
    nav.Frames(3, draw);
    if (with_bar) {
      Check(scene.selected == 1 && scene.changes == 1 && !later_saw_press, "the bar stepped once and the reader after it saw no press");
    } else {
      Check(later_saw_press, "the baseline's reader sees the press");
    }
  }
}

// A bar that isn't given the shoulders leaves them alone. It doesn't step on R1, and a reader that
// takes only an unowned press still sees it, which it wouldn't if the bar had claimed the keys. The
// same bar with the shoulders is the baseline, which steps and takes the press from that reader.
void TestBarWithoutTheShoulders() {
  for (const bool shoulders : {false, true}) {
    Scene scene;
    scene.below_first = true;
    scene.shoulders = shoulders;
    bool reader_saw_press = false;
    NavHarness nav;
    const auto scene_draw = DrawScene(scene);
    const auto draw = [&](InteractContext &context) {
      scene_draw(context);
      reader_saw_press = reader_saw_press || ShoulderPresses().right;
    };
    nav.Frames(3, draw);
    nav.Press(ImGuiKey_GamepadR1, draw);
    if (shoulders) {
      Check(scene.selected == 1 && !reader_saw_press, "with the shoulders the bar steps and holds the press");
    } else {
      Check(scene.selected == 0 && reader_saw_press, "without them the bar doesn't step and leaves the press to others");
    }
  }
}

// While another control owns R1, the bar leaves it alone. TestShoulders is the baseline, the same
// scene without the owner.
void TestOwnedShoulders() {
  Scene scene;
  scene.below_first = true;
  NavHarness nav;
  const auto scene_draw = DrawScene(scene);
  const auto draw = [&scene_draw](InteractContext &context) {
    ImGui::SetKeyOwner(ImGuiKey_GamepadR1, ImGui::GetID("owner"));
    scene_draw(context);
  };
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(scene.selected == 0 && scene.changes == 0, "an owned R1 doesn't reach the bar");
}

// Coming back to a screen after another took focus, the D-pad brings focus into the bar afresh. Up
// from where the other screen's control was reaches the first tab, and the bar sends focus on to
// the chosen tab, so the choice stays.
void TestReturnToScreen() {
  Scene scene;
  scene.selected = 2;
  bool shown = true;
  NavHarness nav;
  const auto scene_draw = DrawScene(scene);
  const auto draw = [&](InteractContext &context) {
    if (shown) {
      scene_draw(context);
      return;
    }
    const ItemId other = ImGui::GetID("other screen");
    (void)Interact(context, other, BELOW, false);
    FocusControl(other);
  };
  nav.Frames(3, draw);
  Check(scene.focused_tab == 2, "focus starts on the chosen tab");
  shown = false;
  nav.Frames(3, draw);
  shown = true;
  const int changes = scene.changes;
  nav.Frame(draw);
  nav.Press(ImGuiKey_GamepadDpadUp, draw);
  nav.Frames(3, draw);
  Check(scene.focused_tab == 2 && scene.selected == 2 && scene.changes == changes, "coming back keeps the choice");
}

// Focus coming into the bar lands on the chosen tab, so the ring and the indicator agree. Then the
// D-pad's left moves focus and the choice together, and R1 brings focus along to the new tab.
void TestFocusOnTheBar() {
  Scene scene;
  scene.selected = 2;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(3, draw);
  Check(scene.focused_tab == 2 && scene.selected == 2, "first focus lands on the chosen tab");
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(scene.focused_tab == 1 && scene.selected == 1, "left moves focus and the choice together");
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(scene.focused_tab == 2 && scene.selected == 2, "R1 brings focus along to the chosen tab");
}

// A click chooses a tab. Focus starts on the control below, so the click alone chooses it, not
// focus moving between tabs.
void TestClick() {
  Scene scene;
  scene.below_first = true;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  Click(nav, THIRD_TAB_X, BAR.y + 20, draw);
  Check(scene.selected == 2, "a click chooses the tab under it");
}

// The chosen tab's label is the primary colour, wholly under the indicator, and the others are the
// secondary colour, wholly outside it.
void TestLabelColours() {
  Scene scene;
  scene.below_first = true;
  scene.selected = 1;
  NavHarness nav;
  nav.Frames(3, DrawScene(scene));
  const auto [tab_left, tab_right] = IndicatorSpan();
  const auto [primary_left, primary_right] = Span(DEFAULT_THEME.colors.text.primary, BAR.y, BAR.Bottom());
  Check(primary_left <= primary_right && primary_left >= tab_left && primary_right <= tab_right, "the chosen label is primary");
  bool secondary_outside = true;
  bool secondary_seen = false;
  for (const auto &vertex : Drawn()) {
    if (vertex.col == ToImU32(DEFAULT_THEME.colors.text.secondary) && vertex.pos.y >= BAR.y && vertex.pos.y <= BAR.Bottom()) {
      secondary_seen = true;
      secondary_outside = secondary_outside && (vertex.pos.x < tab_left || vertex.pos.x > tab_right);
    }
  }
  Check(secondary_seen && secondary_outside, "and the others secondary");
}

// Each label's text is laid out once in a frame. Arranging a label at a width other than the one it
// was measured at would lay it out again, so the frame's count of such layouts stays at none. 
void TestLabelsAreLaidOutOnce() {
  Scene scene;
  scene.below_first = true;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(3, draw);
  const auto relayouts = FrameTestHooks::TextRelayouts(scene.frame);
  Check(relayouts == 0, "no label is laid out a second time");
}

// Two bars in one tree are rendered once and need nothing more. Each decides its own choice, the
// second clamping one past its end, which only its own callback can do, and the first stepping on
// the shoulders, which the second doesn't have.
void TestTwoBarsShareOneRender() {
  NavHarness nav;
  UiFrame frame;
  int upper = 0;
  int lower = 7;
  const auto draw = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const auto a = TabBar(frame, "upper", LABELS, &upper, {.shoulders = true});
    const auto b = TabBar(frame, "lower", LABELS, &lower);
    const std::array<Element, 2> bars{a.element, b.element};
    const Element column = Box(frame, {.direction = FlexDirection::Column}, {}, bars);
    const float unlimited = std::numeric_limits<float>::infinity();
    const auto size = Measure(frame, column, {unlimited, unlimited});
    (void)Arrange(frame, column, {BAR.x, BAR.y, size.x, size.y});
    (void)Render(frame, column, context.list);
  };
  nav.Frames(3, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(upper == 1 && lower == 2, "the bar with the shoulders stepped and the other clamped and stayed");
}

// The labels take the chosen colour on the frame the choice is made, and not a frame later. The
// primary label sits where it will be after the press has settled, which is not where it was.
void TestLabelsFollowOnTheSameFrame() {
  Scene scene;
  scene.below_first = true;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(3, draw);
  const auto before = Span(DEFAULT_THEME.colors.text.primary, BAR.y, BAR.Bottom());
  nav.Key(ImGuiKey_GamepadR1, true);
  nav.Frame(draw);
  nav.Key(ImGuiKey_GamepadR1, false);
  Check(scene.selected == 1, "R1 chose the next tab on that frame");
  const auto on_the_frame = Span(DEFAULT_THEME.colors.text.primary, BAR.y, BAR.Bottom());
  nav.Frames(5, draw);
  const auto settled = Span(DEFAULT_THEME.colors.text.primary, BAR.y, BAR.Bottom());
  Check(before.first != settled.first, "the primary label moved to the new tab");
  Check(on_the_frame == settled, "and was there on the frame of the choice");
}

// The indicator slides to the newly chosen tab over 280 ms: half way it is between the tabs, it is
// still moving at 250 ms, and by 300 ms it lies under the new tab. The slide starts on the frame
// R1 is pressed, so after it 8 frames is 133 ms, 15 is 250 ms and 18 is 300 ms.
void TestIndicatorSlides() {
  Scene scene;
  scene.below_first = true;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  const auto [first_left, first_right] = IndicatorSpan();
  Check(first_left == BAR.x, "the indicator starts under the first tab");
  nav.Key(ImGuiKey_GamepadR1, true);
  nav.Frame(draw);
  nav.Key(ImGuiKey_GamepadR1, false);
  nav.Frames(8, draw);
  const auto [moving_left, moving_right] = IndicatorSpan();
  Check(moving_left > first_left && moving_left < first_right, "half way it is between the tabs");
  nav.Frames(7, draw);
  Check(IndicatorSpan().first < first_right - 0.5f, "at 250 ms it is still moving");
  nav.Frames(3, draw);
  const auto [second_left, second_right] = IndicatorSpan();
  Check(std::fabs(second_left - first_right) < 0.5f && second_right > second_left, "by 300 ms it lies under the second tab");
}

// A pose that scales the bar scales where its indicator goes, which stays under the chosen tab as
// that tab is painted. The unposed bar is the baseline, with the same tab at its arranged place.
void TestIndicatorFollowsAScaledBar() {
  const auto spans = [](float scale) {
    int selected = 1;
    NavHarness nav;
    UiFrame frame;
    std::pair<float, float> expected;
    float bottom = 0.0f;
    const auto draw = [&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const auto bar = TabBar(frame, "tabs", LABELS, &selected);
      Check(!WithPose(frame, bar.element, ElementPose{.scale = scale}).IsNone(), "the pose is accepted");
      const float unlimited = std::numeric_limits<float>::infinity();
      const auto size = Measure(frame, bar.element, {unlimited, unlimited});
      (void)Arrange(frame, bar.element, {BAR.x, BAR.y, size.x, size.y});
      (void)Render(frame, bar.element, context.list);
      const Rect tab = Painted(frame, TabOf(bar, 1));
      expected = {tab.x, tab.Right()};
      bottom = tab.Bottom();
    };
    nav.Frames(2, draw);
    // Looked for along the bottom of the tab as painted, where the indicator must be.
    constexpr float INDICATOR_ROOM = 4.0f;
    return std::pair{Span(DEFAULT_THEME.colors.accent.line, bottom - INDICATOR_ROOM, bottom), expected};
  };
  constexpr float HALF_SIZE = 0.5f;
  constexpr float SLACK = 1.0f;
  const auto [plain, plain_tab] = spans(1.0f);
  const auto [scaled, scaled_tab] = spans(HALF_SIZE);
  Check(std::fabs(plain.first - plain_tab.first) <= SLACK && std::fabs(plain.second - plain_tab.second) <= SLACK, "unposed, the indicator lies under the chosen tab");
  Check(std::fabs((scaled_tab.second - scaled_tab.first) - (plain_tab.second - plain_tab.first) * HALF_SIZE) <= SLACK, "the pose halves the tab");
  Check(std::fabs(scaled.first - scaled_tab.first) <= SLACK && std::fabs(scaled.second - scaled_tab.second) <= SLACK, "scaled, the indicator lies under the tab as painted");
}

// In a bar taller than its tabs the indicator stays along the tabs' bottom and doesn't drop to the
// bar's, which the empty stretch of the bar below it is the baseline for.
void TestIndicatorSitsUnderTheTabs() {
  constexpr float TALL_BAR = 100;
  constexpr float INDICATOR_ROOM = 4.0f;
  int selected = 0;
  NavHarness nav;
  UiFrame frame;
  float tab_bottom = 0.0f;
  float bar_bottom = 0.0f;
  nav.Frames(2, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const auto bar = TabBar(frame, "tabs", LABELS, &selected, {}, {.height = Dp(TALL_BAR)});
    const float unlimited = std::numeric_limits<float>::infinity();
    const auto size = Measure(frame, bar.element, {unlimited, unlimited});
    (void)Arrange(frame, bar.element, {BAR.x, BAR.y, size.x, size.y});
    (void)Render(frame, bar.element, context.list);
    tab_bottom = Arranged(frame, TabOf(bar, 0)).Bottom();
    bar_bottom = Arranged(frame, bar.element).Bottom();
  });
  Check(bar_bottom > tab_bottom + INDICATOR_ROOM, "the bar reaches past its tabs");
  const auto under_tabs = Span(DEFAULT_THEME.colors.accent.line, tab_bottom - INDICATOR_ROOM, tab_bottom);
  const auto under_bar = Span(DEFAULT_THEME.colors.accent.line, bar_bottom - INDICATOR_ROOM, bar_bottom);
  Check(under_tabs.first <= under_tabs.second, "the indicator lies along the tabs' bottom");
  Check(under_bar.first > under_bar.second, "and not along the bar's");
}

// In a bar narrower than its tabs, the indicator under a tab past the bar's end keeps its width.
void TestNarrowBar() {
  int selected = 2;
  NavHarness nav;
  UiFrame frame;
  nav.Frames(2, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    (void)DrawElementBar(frame, "tabs", {100, 50, 120, 56}, LABELS, selected);
  });
  const auto [left, right] = IndicatorSpan();
  Check(right - left > 50, "the indicator under a tab past the bar keeps its width");
}

// Two tabs with the same label are separate tabs: clicking the second chooses the second, and no
// two items share an id. Two bars with the same tabs are separate too.
void TestSameLabels() {
  constexpr std::array<std::string_view, 3> SAME{"Slot", "Slot", "Settings"};
  int selected = 0;
  int other = 0;
  NavHarness nav;
  UiFrame frame;
  const auto draw = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    (void)DrawElementBar(frame, "first", BAR, SAME, selected);
    (void)DrawElementBar(frame, "second", {100, 200, 900, 56}, SAME, other);
  };
  nav.Frames(2, draw);
  // Each "Slot" tab is about 104 wide, so the second spans about 204 to 308.
  Click(nav, 250, BAR.y + 20, draw);
  Check(selected == 1 && other == 0, "clicking the second of two same tabs chooses it, in that bar alone");
  // ImGui only checks the hovered item for a twin, so this checks the tab just clicked.
  Check(GImGui->DebugDrawIdConflictsId == 0, "the clicked tab shares its id with nothing");
}

// Only the bar given the shoulders answers them. R1 changes the first bar and leaves the second.
void TestOneBarHasTheShoulders() {
  int first = 0;
  int second = 0;
  NavHarness nav;
  UiFrame frame;
  const auto draw = [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    (void)DrawElementBar(frame, "first", BAR, LABELS, first, true);
    (void)DrawElementBar(frame, "second", {100, 200, 900, 56}, LABELS, second);
  };
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(first == 1 && second == 0, "only the bar with the shoulders answers R1");
}

// A choice out of range is clamped to the tabs and reported as a change. With no labels, nothing
// changes.
void TestClampAndEmpty() {
  int selected = 7;
  int changes = 0;
  NavHarness nav;
  UiFrame frame;
  nav.Frames(2, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const auto outcome = DrawElementBar(frame, "tabs", BAR, LABELS, selected);
    changes += outcome.changed ? 1 : 0;
  });
  Check(selected == 2 && changes == 1, "a choice past the end is clamped, once, as a change");
  int untouched = 4;
  bool changed = true;
  NavHarness empty;
  UiFrame empty_frame;
  empty.Frame([&](InteractContext &context) {
    empty_frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    TabBarProps props;
    props.changed = &changed;
    ElementError reason = ElementError::None;
    ForgetReportedRefusals();
    const auto bar = TabBar(empty_frame, "tabs", std::span<const std::string_view>{}, &untouched, props, {}, &reason);
    Check(!ReportRefusal(ElementError::NoLabels), "a bar with no labels is logged");
    Check(bar.element.IsNone() && reason == ElementError::NoLabels, "a bar with no labels is a none element, and says why");
  });
  Check(!changed && untouched == 4, "with no labels nothing changes");
}

// A bar with no choice to show is refused and logged as that, with no labels still the reason when
// it has neither. One with a choice is the baseline that builds in the same frame.
void TestNoChoiceIsRefused() {
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const std::array<std::string_view, 2> labels{"One", "Two"};
    ElementError reason = ElementError::None;
    ForgetReportedRefusals();
    const auto bar = TabBar(frame, "none", labels, nullptr, {}, {}, &reason);
    Check(!ReportRefusal(ElementError::NoValue), "a bar with no choice is logged");
    Check(bar.element.IsNone() && reason == ElementError::NoValue, "and is a none element that says why");
    (void)TabBar(frame, "neither", std::span<const std::string_view>{}, nullptr, {}, {}, &reason);
    Check(reason == ElementError::NoLabels, "with no labels either, that is the reason");
    int selected = 0;
    Check(!TabBar(frame, "some", labels, &selected).element.IsNone(), "where one with a choice builds");
  });
}

} // namespace

int main() {
  TestNoChoiceIsRefused();
  TestShoulders();
  TestPressIsConsumed();
  TestBarWithoutTheShoulders();
  TestOwnedShoulders();
  TestFocusOnTheBar();
  TestReturnToScreen();
  TestClick();
  TestLabelColours();
  TestLabelsAreLaidOutOnce();
  TestTwoBarsShareOneRender();
  TestLabelsFollowOnTheSameFrame();
  TestIndicatorSlides();
  TestIndicatorFollowsAScaledBar();
  TestIndicatorSitsUnderTheTabs();
  TestNarrowBar();
  TestSameLabels();
  TestOneBarHasTheShoulders();
  TestClampAndEmpty();
  std::printf("System UI tabs tests passed\n");
  return 0;
}
