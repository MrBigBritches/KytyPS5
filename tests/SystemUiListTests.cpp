#include "SystemUiNavHarness.h"

#include "graphics/presentation/system/components/list.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"

#include <algorithm>
#include <array>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace SystemUiTest;

constexpr Rect ABOVE{100, 40, 300, 56};
constexpr Rect LIST{100, 140, 300, 400};
constexpr float ROW = 56;
// A control below the list's area, far enough below it that row 9, hidden at 649 to 705, lies
// between them.
constexpr Rect BELOW{100, 720, 300, 56};
// The list's rows sit in from its edges by the focus ring, three pixels and a two pixel keyline.
constexpr float INSET = 5;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiListTests: failed: %s\n", message);
    std::abort();
  }
}

// The vertices every window drew in its last frame.
std::vector<ImDrawVert> DrawnAll() {
  std::vector<ImDrawVert> vertices;
  for (ImGuiWindow *window : GImGui->Windows) {
    vertices.insert(vertices.end(), window->DrawList->VtxBuffer.begin(), window->DrawList->VtxBuffer.end());
  }
  return vertices;
}

// True when some vertex of the last frame, in any window, has exactly `color`.
bool HasColor(Color color) {
  const auto vertices = DrawnAll();
  return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &v) { return v.col == ToImU32(color); });
}

// Counts the activations an OnActivate callback reports.
struct CountActivations {
  int *count;
  void operator()(Element) const { ++*count; }
};

// A control above a list of rows, recording which has focus and what activated.
struct Scene {
  // How many activations the rows' OnActivate callbacks counted.
  int callbacks = 0;
  int rows = 5;
  int selected = -1;
  int disabled = -1;
  // Tells the rows that what's behind them is light.
  bool on_light = false;
  // Each row's second line.
  std::string_view second;
  int focused_row = -1;
  bool above_focused = false;
  // Moves focus to the control above on the next frame, as a screen change would.
  bool focus_above = false;
  // Draws the list. A screen without it leaves it undrawn.
  bool draw_list = true;
  // Draws a control below the list, first, so it takes the first focus.
  bool below = false;
  bool below_focused = false;
  int activated = -1;
  float scroll = 0;
  std::vector<std::string> names;
  // Draws the list's rows posed like this, as a gallery's specimens.
  const ItemVisuals *pose = nullptr;
  // The frame the list is built in.
  UiFrame frame;
};

// Draws the scene's control and list.
auto DrawScene(Scene &scene) {
  for (int index = 0; index < scene.rows; index++) {
    scene.names.push_back("row " + std::to_string(index));
  }
  return [&scene](InteractContext &context) {
    if (scene.below) {
      scene.below_focused = Interact(context, ImGui::GetID("below"), BELOW, false).state.focused;
    }
    scene.above_focused = Interact(context, ImGui::GetID("above"), ABOVE, false).state.focused;
    if (scene.focus_above) {
      ImGui::SetFocusID(ImGui::GetID("above"), ImGui::GetCurrentWindow());
      scene.focus_above = false;
    }
    scene.focused_row = -1;
    if (!scene.draw_list) {
      return;
    }
    UiFrame &frame = scene.frame;
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    // How far the list has scrolled as the frame starts, which is where it draws its rows.
    scene.scroll = context.store.List(ImGui::GetID("list")).scroll;
    std::vector<Element> rows;
    for (int index = 0; index < scene.rows; index++) {
      ListRowProps props{scene.names[index], scene.second, index == scene.selected, index == scene.disabled};
      props.on_light = scene.on_light;
      rows.push_back(ListRow(frame, scene.names[index], props));
      (void)OnActivate(frame, rows.back(), CountActivations{&scene.callbacks});
    }
    const ItemLayout size{.width = Dp(LIST.width), .height = Dp(LIST.height)};
    const Element list = List(frame, "list", rows, size);
    if (scene.pose != nullptr) {
      (void)Specimen(frame, list, *scene.pose);
    }
    const float unlimited = std::numeric_limits<float>::infinity();
    (void)Measure(frame, list, {unlimited, unlimited});
    (void)Arrange(frame, list, LIST);
    Check(!Render(frame, list, context.list).error.has_value(), "the list rendered");
    for (int index = 0; index < scene.rows; index++) {
      const auto input = InputOf(frame, rows[index]);
      if (input && input->interaction.state.activated) {
        scene.activated = index;
      }
      if (input && input->interaction.state.focused) {
        scene.focused_row = index;
      }
    }
  };
}

// A heading too long for its slot is cut short inside the padding of a row's text, on both sides. It
// starts at the padding, and its end reaches past where the padding would be if it were counted
// twice, so a short heading or one with too little room could not pass.
void TestHeadingKeepsItsPadding() {
  constexpr Rect SLOT{100, 140, 300, 56};
  // A glyph's first pixel sits a little inside its advance.
  constexpr float GLYPH_SLACK = 2.0f;
  const std::string text(200, 'a');
  float padding = 0.0f;
  Color color;
  UiFrame frame;
  NavHarness nav;
  nav.Frames(2, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    padding = context.ui.scale.Px(context.ui.theme.space.s4);
    color = context.ui.theme.colors.text.secondary;
    const Element heading = ListHeading(frame, "slot", {.text = text});
    (void)Measure(frame, heading, {SLOT.width, std::numeric_limits<float>::infinity()});
    (void)Arrange(frame, heading, SLOT);
    Paint(frame, heading, context.list);
  });
  float left = std::numeric_limits<float>::max();
  float right = std::numeric_limits<float>::lowest();
  for (const ImDrawVert &vertex : DrawnAll()) {
    if (vertex.col == ToImU32(color)) {
      left = std::min(left, vertex.pos.x);
      right = std::max(right, vertex.pos.x);
    }
  }
  Check(left >= SLOT.x + padding - GLYPH_SLACK && left <= SLOT.x + padding + GLYPH_SLACK, "a heading starts at the padding");
  Check(right <= SLOT.Right() - padding, "and stops short of the padding on the right");
  Check(right >= SLOT.Right() - 2.0f * padding, "having used the line up to there");
}

// Down moves into the list and along its rows. When focus leaves the list from row 2 and comes
// back with the D-pad, it returns to row 2 rather than row 0, the row nearest the way in.
void TestRememberedRow() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  Check(scene.above_focused, "the control above starts with focus");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(scene.focused_row == 0, "down enters the list at its first row");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(scene.focused_row == 2, "down moves along the rows");
  scene.focus_above = true;
  nav.Frames(2, draw);
  Check(scene.above_focused, "focus leaves the list");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(scene.focused_row == 2, "coming back returns to the row it left");
}

// A selected row shows the accent bar while focus is elsewhere.
void TestSelectedWithoutFocus() {
  Scene scene;
  scene.selected = 3;
  NavHarness nav;
  nav.Frames(3, DrawScene(scene));
  Check(scene.above_focused && HasColor(DEFAULT_THEME.colors.accent.line), "a selected row shows its bar without focus");
  Scene plain;
  NavHarness plain_nav;
  plain_nav.Frames(3, DrawScene(plain));
  Check(!HasColor(DEFAULT_THEME.colors.accent.line), "a list with nothing selected has no bar");
}

// True when the focus ring's colour is drawn in the list's inset, left of the rows. The ring is
// the labels' colour, so only its place outside a row, within the list, tells it apart.
bool RingBesideRows() {
  const auto vertices = DrawnAll();
  return std::any_of(vertices.begin(), vertices.end(), [](const ImDrawVert &vertex) {
    return vertex.col == ToImU32(DEFAULT_THEME.colors.focus.ring) && vertex.pos.x < LIST.x + INSET - 0.5f &&
           vertex.pos.y >= LIST.y && vertex.pos.y <= LIST.Bottom();
  });
}

// Rows drawn in a posed scope show the pose and can't be reached. The same list unposed is the
// baseline, where down from the control above enters the list and no row wears a ring while the
// control above has focus.
void TestPosedRows() {
  const ItemVisuals focused{0.0f, 1.0f, 0.0f};
  for (const bool posed : {false, true}) {
    Scene scene;
    scene.pose = posed ? &focused : nullptr;
    NavHarness nav;
    const auto draw = DrawScene(scene);
    nav.Frames(3, draw);
    Check(scene.above_focused && RingBesideRows() == posed,
          posed ? "posed rows wear the ring while focus is above" : "the baseline's rows wear no ring");
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
    Check((scene.focused_row == -1) == posed, posed ? "down can't enter posed rows" : "the baseline's down enters the list");
  }
}

// True when the list's own rectangle holds a vertex of exactly `color`. ImGui's fallback window has a
// title in the same white as a row's first line, so a test of that colour has to look only here.
bool ListHolds(Color color) {
  const auto vertices = DrawnAll();
  return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &vertex) {
    return vertex.col == ToImU32(color) && vertex.pos.x >= LIST.x && vertex.pos.x <= LIST.Right() && vertex.pos.y >= LIST.y && vertex.pos.y <= LIST.Bottom();
  });
}

// On a light surface a row's lines are dark, even faded, and its hover and focus shade. The same row
// without the option is the baseline. A single row, so nothing else draws these, and the ring's white
// is why the first line's colour is only checked at rest.
void TestRowsOnLight() {
  const auto &colors = DEFAULT_THEME.colors;
  const float faded = DEFAULT_THEME.opacity.disabled;
  const auto drawn = [](bool on_light, bool disabled, const ItemVisuals *pose, const std::function<void()> &check) {
    Scene scene;
    scene.rows = 1;
    scene.second = "second";
    scene.on_light = on_light;
    scene.disabled = disabled ? 0 : -1;
    scene.pose = pose;
    NavHarness nav;
    nav.Frames(3, DrawScene(scene));
    check();
  };
  drawn(true, false, nullptr, [&] {
    Check(ListHolds(colors.text.on_light) && !ListHolds(colors.text.primary), "on a light surface the first line is dark");
    Check(ListHolds(colors.text.secondary_on_light) && !ListHolds(colors.text.secondary), "and so is the second");
  });
  drawn(false, false, nullptr, [&] {
    Check(ListHolds(colors.text.primary) && ListHolds(colors.text.secondary) && !ListHolds(colors.text.on_light) && !ListHolds(colors.text.secondary_on_light), "and without the option they're the usual colours");
  });
  drawn(true, true, nullptr, [&] {
    Check(ListHolds(colors.text.on_light.MulAlpha(faded)) && ListHolds(colors.text.secondary_on_light.MulAlpha(faded)), "disabled, the dark lines fade");
  });
  const ItemVisuals hover{.hover = 1.0f};
  const ItemVisuals focus{.focus = 1.0f};
  drawn(true, false, &hover, [&] { Check(ListHolds(colors.state.hover_on_light) && !ListHolds(colors.state.hover), "its hover shades"); });
  drawn(false, false, &hover, [&] { Check(ListHolds(colors.state.hover) && !ListHolds(colors.state.hover_on_light), "where without the option it lightens"); });
  drawn(true, false, &focus, [&] { Check(ListHolds(colors.state.focus_on_light) && !ListHolds(colors.state.focus), "and so does its focus"); });
}

// The vertices of the list's thumb, drawn in the tertiary colour at the list's right edge, where nothing
// else in these scenes draws it. Rows' second lines use their own colour.
Rect ThumbRect() {
  float left = std::numeric_limits<float>::max(), top = left, right = -left, bottom = -left;
  const ImU32 colour = ToImU32(DEFAULT_THEME.colors.text.tertiary);
  for (const ImDrawVert &vertex : DrawnAll()) {
    if (vertex.col == colour && vertex.pos.x >= LIST.Right() - 16.0f && vertex.pos.x <= LIST.Right() + 1.0f) {
      left = std::min(left, vertex.pos.x);
      top = std::min(top, vertex.pos.y);
      right = std::max(right, vertex.pos.x);
      bottom = std::max(bottom, vertex.pos.y);
    }
  }
  return {left, top, right - left, bottom - top};
}

// Rows that don't all fit show a thumb at the right edge, its length the share in view, worked out here
// apart from the code, at the top with no scroll and at the bottom at the end. Rows that do fit are the
// baseline, with none.
void TestScrollBarShowsWhenRowsOverflow() {
  constexpr float SLACK = 1.5f;
  const float row = ListDefaults(DEFAULT_THEME).row_single;
  Scene short_list;
  short_list.rows = 3;
  NavHarness short_nav;
  short_nav.Frames(3, DrawScene(short_list));
  Check(ThumbRect().width < 0.0f, "rows that fit have no thumb");
  short_list.selected = 0;
  short_nav.Frames(2, DrawScene(short_list));
  float fitting_right = 0.0f;
  for (const ImDrawVert &vertex : DrawnAll()) {
    fitting_right = vertex.col == ToImU32(DEFAULT_THEME.colors.accent.tint) ? std::max(fitting_right, vertex.pos.x) : fitting_right;
  }
  Scene scene;
  scene.rows = 20;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(3, draw);
  const Rect thumb = ThumbRect();
  const float content = row * 20.0f + 2.0f * (LIST.height - (LIST.height - 2.0f * INSET)) / 2.0f;
  Check(std::fabs(thumb.height - LIST.height * LIST.height / content) <= SLACK, "the thumb's length is the share of the rows in view");
  Check(std::fabs(thumb.y - LIST.y) <= SLACK && std::fabs(thumb.Right() - LIST.Right()) <= SLACK, "at the top right with no scroll");
  // The rows stop short of it by the ring's room and a gap, written out here apart from the code. The first
  // row is selected, so its tint shows where it ends.
  constexpr float THUMB_WIDTH = 4.0f;
  constexpr float GAP = 4.0f;
  scene.selected = 0;
  nav.Frames(2, draw);
  float row_right = 0.0f;
  for (const ImDrawVert &vertex : DrawnAll()) {
    row_right = vertex.col == ToImU32(DEFAULT_THEME.colors.accent.tint) ? std::max(row_right, vertex.pos.x) : row_right;
  }
  Check(std::fabs(row_right - (LIST.Right() - THUMB_WIDTH - GAP - INSET)) <= SLACK, "and the rows leave a gap before it");
  Check(thumb.x - row_right >= INSET + GAP - SLACK, "wide enough for a row's focus ring and a gap");
  Check(std::fabs(fitting_right - row_right) <= SLACK, "the same width as rows that all fit, so they don't move");
  scene.selected = -1;
  for (int press = 0; press < 25; press++) {
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
  }
  nav.Frames(3, draw);
  const Rect end = ThumbRect();
  Check(std::fabs(end.Bottom() - LIST.Bottom()) <= SLACK, "and at the bottom at the end");
}

// The wheel scrolls a list while the pointer is over it, and the list stays there though a row has focus,
// where without a wheel it holds that row in view. Moving focus goes back to it. With the pointer off the
// list, the baseline, the wheel does nothing. Rows that fit don't move.
void TestWheelScrollsTheList() {
  const float row = ListDefaults(DEFAULT_THEME).row_single;
  Scene scene;
  scene.rows = 20;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(3, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Frames(2, draw);
  const int focused = scene.focused_row;
  Check(focused >= 0 && scene.scroll == 0.0f, "focus starts on a row with nothing scrolled");
  nav.MoveMouse(LIST.Right() + 100.0f, LIST.y + 50.0f, draw);
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -1.0f);
  nav.Frames(3, draw);
  Check(scene.scroll == 0.0f, "the wheel off the list does nothing");
  nav.MoveMouse(LIST.x + 50.0f, LIST.y + 50.0f, draw);
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -3.0f);
  nav.Frames(5, draw);
  Check(std::fabs(scene.scroll - 3.0f * row) <= 1.0f, "over it, each notch scrolls a row");
  ImGui::GetIO().AddMouseWheelEvent(0.0f, 1.0f);
  nav.Frames(5, draw);
  Check(std::fabs(scene.scroll - 2.0f * row) <= 1.0f, "and back the other way");
  Check(scene.focused_row == focused, "with focus left on its row, now above the list");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Frames(3, draw);
  const float focused_top = LIST.y + INSET + static_cast<float>(scene.focused_row) * row - scene.scroll;
  Check(scene.focused_row == focused + 1 && focused_top >= LIST.y && focused_top + row <= LIST.Bottom(), "and moving focus brings the list back to it");
  ImGui::GetIO().AddMouseWheelEvent(0.0f, 100.0f);
  nav.Frames(5, draw);
  Check(scene.scroll == 0.0f, "stopping at the top");
  Scene fitting;
  fitting.rows = 3;
  NavHarness fitting_nav;
  const auto fitting_draw = DrawScene(fitting);
  fitting_nav.Frames(3, fitting_draw);
  fitting_nav.MoveMouse(LIST.x + 50.0f, LIST.y + 50.0f, fitting_draw);
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -3.0f);
  fitting_nav.Frames(3, fitting_draw);
  Check(fitting.scroll == 0.0f, "rows that fit don't move");
}

// Dragging the thumb scrolls the list in step, by the share of its travel, and stops when released, with
// no row under it pressed and focus left where it was. A press off the thumb jumps there.
void TestDraggingTheThumb() {
  constexpr float SLACK = 2.0f;
  const float row = ListDefaults(DEFAULT_THEME).row_single;
  Scene scene;
  scene.rows = 20;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(3, draw);
  const Rect thumb = ThumbRect();
  const float hidden = row * 20.0f + 2.0f * INSET - LIST.height;
  const float travel = LIST.height - thumb.height;
  const float x = thumb.x + thumb.width / 2.0f;
  nav.MoveMouse(x, thumb.y + 4.0f, draw);
  nav.Frame(draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MoveMouse(x, thumb.y + 4.0f + travel / 2.0f, draw);
  nav.Frames(2, draw);
  Check(std::fabs(scene.scroll - hidden / 2.0f) <= SLACK, "half the thumb's travel is half the scroll");
  nav.MouseButton(false);
  nav.Frames(2, draw);
  const float released = scene.scroll;
  nav.MoveMouse(x, LIST.y + 4.0f, draw);
  nav.Frames(2, draw);
  Check(scene.scroll == released, "and letting go stops it following");
  Check(scene.activated == -1 && scene.callbacks == 0, "and no row was pressed");
  Check(scene.above_focused && scene.focused_row == -1, "and focus stays where it was");
  // Three quarters of the way down, past the thumb, whose middle then comes to the pointer.
  const float jump_y = LIST.y + LIST.height * 0.75f;
  nav.MoveMouse(x, jump_y, draw);
  nav.Frame(draw);
  nav.MouseButton(true);
  nav.Frames(2, draw);
  nav.MouseButton(false);
  nav.Frames(2, draw);
  Check(std::fabs(scene.scroll - (jump_y - thumb.height / 2.0f - LIST.y) / travel * hidden) <= SLACK, "a press off the thumb jumps its middle there");
}

void LeaveFromRowTwo(NavHarness &nav, Scene &scene, const std::function<void(InteractContext &)> &draw);

// Focus coming back into a list goes to the row it remembers, which the wheel may have scrolled away. The
// list goes back to it, as confirm shouldn't act on a row out of view, where a list that kept focus
// stays where the mouse put it, the baseline in TestWheelScrollsTheList.
void TestFocusReturningShowsItsRow() {
  const float row = ListDefaults(DEFAULT_THEME).row_single;
  Scene scene;
  scene.rows = 20;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  LeaveFromRowTwo(nav, scene, draw);
  nav.MoveMouse(LIST.x + 50.0f, LIST.y + 50.0f, draw);
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -10.0f);
  nav.Frames(3, draw);
  const float scrolled = scene.scroll;
  Check(scrolled > row * 3.0f, "the wheel takes the remembered row out of view");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Frames(4, draw);
  const float top = LIST.y + INSET + 2.0f * row - scene.scroll;
  Check(scene.focused_row == 2 && top >= LIST.y && top + row <= LIST.Bottom(), "and focus coming back shows it");
}

// A thumb comes from the list's layout, not its posed size, so a list that fits has none at half size,
// where a list that doesn't fit is the baseline, with one at half the length. Rows of one line, so only a
// thumb draws in this colour.
void TestPosedListThumb() {
  const float row = ListDefaults(DEFAULT_THEME).row_single;
  const float fits = row * 3.0f + 2.0f * INSET;
  const auto draw_at = [&](int rows, float height, float scale) {
    NavHarness nav;
    UiFrame frame;
    std::vector<std::string> names;
    for (int index = 0; index < rows; ++index) {
      names.push_back("row " + std::to_string(index));
    }
    nav.Frames(3, [&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      std::vector<Element> elements;
      for (const std::string &name : names) {
        elements.push_back(ListRow(frame, name, {name}));
      }
      const Element list = List(frame, "list", elements, {.width = Dp(LIST.width), .height = Dp(height)});
      ElementPose pose;
      pose.scale = scale;
      (void)WithPose(frame, list, pose);
      const float unlimited = std::numeric_limits<float>::infinity();
      (void)Measure(frame, list, {unlimited, unlimited});
      (void)Arrange(frame, list, {LIST.x, LIST.y, LIST.width, height});
      (void)Render(frame, list, context.list);
    });
    return HasColor(DEFAULT_THEME.colors.text.tertiary);
  };
  Check(!draw_at(3, fits, 0.5f), "a list that fits has no thumb at half size");
  Check(draw_at(20, fits, 0.5f), "and one that doesn't has");
}

// Text with nothing in it to focus scrolls by the D-pad while its window has focus. With its window not
// focused it doesn't, as a dialog under another must stay still. Each is the other's baseline.
void TestScrollingTextNeedsFocus() {
  std::string words;
  for (int line = 0; line < 200; ++line) {
    words += "A long passage of words that wraps across the width of the area many times over. ";
  }
  const auto thumb_after_down = [&](bool focused) {
    NavHarness nav;
    UiFrame frame;
    const auto draw = [&](InteractContext &context) {
      if (!focused) {
        ImGui::SetWindowFocus(nullptr);
      }
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      TextProps props{words, DEFAULT_THEME.type.body, DEFAULT_THEME.colors.text.secondary};
      props.overflow = TextOverflow::Wrap;
      const std::array<Element, 1> inside{Text(frame, props)};
      Style clip;
      clip.overflow = Overflow::Clip;
      const Element box = Keyed(frame, Box(frame, {.direction = FlexDirection::Column, .align = Align::Stretch, .item = {.width = Dp(LIST.width), .height = Dp(LIST.height)}}, clip, inside), "words");
      (void)AsScrollingText(frame, box);
      const float unlimited = std::numeric_limits<float>::infinity();
      (void)Measure(frame, box, {LIST.width, unlimited});
      (void)Arrange(frame, box, LIST);
      (void)Render(frame, box, context.list);
    };
    nav.Frames(3, draw);
    const float before = ThumbRect().y;
    Check(ThumbRect().width > 0.0f, "text that doesn't fit has a thumb");
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
    nav.Frames(2, draw);
    return ThumbRect().y - before;
  };
  Check(thumb_after_down(true) > 0.0f, "with its window focused, down scrolls the text");
  Check(thumb_after_down(false) == 0.0f, "and with it not focused, down leaves it");
}

// A list shown as a specimen takes no input, so the wheel doesn't move it, where an unposed list is the
// baseline in TestWheelScrollsTheList. Its thumb still shows.
void TestPosedListIgnoresTheWheel() {
  const ItemVisuals resting{};
  Scene scene;
  scene.rows = 20;
  scene.pose = &resting;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(3, draw);
  Check(ThumbRect().width > 0.0f, "a posed list shows its thumb");
  nav.MoveMouse(LIST.x + 50.0f, LIST.y + 50.0f, draw);
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -3.0f);
  nav.Frames(3, draw);
  Check(scene.scroll == 0.0f, "and the wheel doesn't move it");
}

// A disabled row is skipped by the D-pad.
void TestDisabledRowSkipped() {
  Scene scene;
  scene.disabled = 1;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(scene.focused_row == 2, "down skips the disabled row");
}

// Confirm activates the focused row.
void TestActivation() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(scene.activated == 1, "confirm activates the focused row");
}

// An activation is reported by the row's input and by its callback on the frame of the press, so
// neither is a frame late, and RenderResult::activated names the row.
void TestActivationIsReportedTwoWays() {
  Scene scene;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Key(ImGuiKey_GamepadFaceDown, true);
  nav.Frame(draw);
  Check(scene.activated == 0 && scene.callbacks == 1, "the input and the callback report the press on the frame of the press");
  nav.Key(ImGuiKey_GamepadFaceDown, false);
  nav.Frames(3, draw);
  Check(scene.callbacks == 1, "and once");
}

// The template of ListOf is run once for each datum, in order, with that datum, and the rows it
// makes are the list's rows in that order, so a row shows the data it is given and never another's.
void TestListOfRunsTheTemplateOnEachDatum() {
  struct Datum {
    std::string name;
  };
  const std::vector<Datum> data{{"alpha"}, {"beta"}, {"gamma"}};
  std::vector<std::string> seen;
  NavHarness nav;
  UiFrame frame;
  nav.Frame([&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element list = ListOf(frame, "list", std::span<const Datum>(data), [&](UiFrame &built, const Datum &datum) {
      seen.push_back(datum.name);
      return ListRow(built, datum.name, {datum.name});
    }, {.width = Dp(LIST.width), .height = Dp(LIST.height)});
    Check(!list.IsNone(), "the list was built");
    const float unlimited = std::numeric_limits<float>::infinity();
    (void)Measure(frame, list, {unlimited, unlimited});
    Check(Arrange(frame, list, LIST), "the list arranges");
    Check(!Render(frame, list, context.list).error.has_value(), "the list rendered");
  });
  Check(seen == std::vector<std::string>{"alpha", "beta", "gamma"}, "the template ran on each datum, in order");
}

// A row with one line is 56 high and one with two is 80.
void TestRowHeights() {
  NavHarness nav;
  nav.Frame([](InteractContext &context) {
    UiFrame frame;
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const float unlimited = std::numeric_limits<float>::infinity();
    const Element single = ListRow(frame, "single", {"Title"});
    const Element twin = ListRow(frame, "twin", {"Title", "Detail"});
    Check(Measure(frame, single, {unlimited, unlimited}).y == 56 && Measure(frame, twin, {unlimited, unlimited}).y == 80,
          "single rows are 56 high and double rows 80");
  });
}

// One row of a list of mixed heights, which has a second line when `twin`.
struct MixedDatum {
  std::string name;
  bool twin = false;
};

// What a list of mixed-height rows laid out and which row has focus, over several frames.
struct MixedRun {
  std::vector<Rect> rects;
  int focused = -1;
  float scroll = 0;
};

// Draws `data` as rows, single ones with a title and double ones with a second line too, into LIST
// and records where each row was arranged and which has focus.
auto DrawMixed(UiFrame &frame, const std::vector<MixedDatum> &data, MixedRun &run) {
  return [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    std::vector<Element> made;
    const Element list = ListOf(frame, "list", std::span<const MixedDatum>(data), [&](UiFrame &built, const MixedDatum &datum) {
      made.push_back(ListRow(built, datum.name, {datum.name, datum.twin ? "detail" : ""}));
      return made.back();
    }, {.width = Dp(LIST.width), .height = Dp(LIST.height)});
    run.scroll = context.store.List(ImGui::GetID("list")).scroll;
    const float unlimited = std::numeric_limits<float>::infinity();
    (void)Measure(frame, list, {unlimited, unlimited});
    (void)Arrange(frame, list, LIST);
    Check(!Render(frame, list, context.list).error.has_value(), "the list rendered");
    run.rects.clear();
    run.focused = -1;
    for (size_t index = 0; index < made.size(); index++) {
      run.rects.push_back(Arranged(frame, made[index]));
      const auto input = InputOf(frame, made[index]);
      if (input && input->interaction.state.focused) {
        run.focused = static_cast<int>(index);
      }
    }
  };
}

// Rows of one line, two lines and one line stack with each row's own height, so the second begins
// 56 below the first and the third 80 below the second. A list of three single rows is the
// baseline, where the third is 112 below the first and not 136.
void TestMixedRowHeightsAccumulate() {
  const auto positions = [](bool middle_twin) {
    UiFrame frame;
    MixedRun run;
    const std::vector<MixedDatum> data{{"a"}, {"b", middle_twin}, {"c"}};
    NavHarness nav;
    nav.Frames(2, DrawMixed(frame, data, run));
    Check(run.rects.size() == 3, "three rows were laid out");
    return run.rects;
  };
  const auto mixed = positions(true);
  const auto plain = positions(false);
  Check(mixed[0].y == LIST.y + INSET && mixed[0].height == ROW, "the first row starts at the list's top and is a line high");
  Check(mixed[1].y == mixed[0].y + 56 && mixed[1].height == 80, "the double row follows it and is two lines high");
  Check(mixed[2].y == mixed[1].y + 80 && mixed[2].height == 56, "the third starts below the double row");
  Check(plain[2].y == plain[0].y + 2 * 56, "the baseline's third row is only two single rows down");
}

// Moving focus to a double row that sticks out below the list scrolls the whole row into view,
// where scrolling by one line's height would leave its second line hidden. The row sits in the
// last of eight, after six single rows, so it begins in view and ends below it.
void TestScrollingToADoubleRowShowsAllOfIt() {
  UiFrame frame;
  MixedRun run;
  std::vector<MixedDatum> data;
  for (int index = 0; index < 6; index++) {
    data.push_back({"row " + std::to_string(index)});
  }
  data.push_back({"double", true});
  data.push_back({"after"});
  NavHarness nav;
  const auto draw = DrawMixed(frame, data, run);
  nav.Frames(2, draw);
  Check(run.rects[6].Bottom() - run.scroll > LIST.Bottom(), "before scrolling, the double row sticks out below the list");
  // Nothing else takes first focus, so the list starts on its first row.
  Check(run.focused == 0, "the list starts with focus on its first row");
  for (int press = 0; press < 6; press++) {
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
  }
  nav.Frames(10, draw);
  Check(run.focused == 6, "focus reached the double row");
  const float top = run.rects[6].y - run.scroll;
  const float bottom = run.rects[6].Bottom() - run.scroll;
  Check(top >= LIST.y && bottom <= LIST.Bottom() - INSET + 0.5f, "the whole double row is in view");
}

// A row's key and its text, which change independently.
struct KeyedDatum {
  std::string key;
  std::string text;
};

// What a ListOf of keyed rows drew, where focus was, and the pixels of the frame.
struct KeyedRun {
  std::string focusedKey;
  std::vector<std::pair<ImVec2, ImU32>> shapes;
};

// Draws a control above a ListOf of `data` keyed by each datum's key and showing its text, and
// records which row's key has focus and the position and colour of every vertex drawn.
auto DrawKeyed(UiFrame &frame, const std::vector<KeyedDatum> &data, KeyedRun &run) {
  return [&](InteractContext &context) {
    (void)Interact(context, ImGui::GetID("above"), ABOVE, false);
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    std::vector<std::pair<std::string, Element>> made;
    const Element list = ListOf(frame, "list", std::span<const KeyedDatum>(data), [&](UiFrame &built, const KeyedDatum &datum) {
      const Element row = ListRow(built, datum.key, {datum.text});
      made.emplace_back(datum.key, row);
      return row;
    }, {.width = Dp(LIST.width), .height = Dp(LIST.height)});
    const float unlimited = std::numeric_limits<float>::infinity();
    (void)Measure(frame, list, {unlimited, unlimited});
    (void)Arrange(frame, list, LIST);
    Check(!Render(frame, list, context.list).error.has_value(), "the list rendered");
    run.focusedKey.clear();
    for (const auto &[key, row] : made) {
      const auto input = InputOf(frame, row);
      if (input && input->interaction.state.focused) {
        run.focusedKey = key;
      }
    }
    run.shapes.clear();
    for (const ImDrawVert &vertex : context.list.VtxBuffer) {
      run.shapes.emplace_back(vertex.pos, vertex.col);
    }
  };
}

// Whether two runs drew every vertex at the same place in the same colour.
bool SameShapes(const KeyedRun &first, const KeyedRun &second) {
  return first.shapes.size() == second.shapes.size() &&
         std::equal(first.shapes.begin(), first.shapes.end(), second.shapes.begin(), [](const auto &a, const auto &b) {
           return a.first.x == b.first.x && a.first.y == b.first.y && a.second == b.second;
         });
}

// Between frames the data changes its text and its order. Each row then draws its new text, where
// a row bound to the data it was first made from would still draw the old, and focus stays on the
// row of its key wherever that has moved to. The comparison is the same final data drawn from a
// fresh start with focus on the same key, which must match vertex for vertex. The first data drawn
// alone shows that the comparison can tell the texts apart.
void TestListOfFollowsChangedAndReorderedData() {
  const std::vector<KeyedDatum> first{{"a", "Alpha"}, {"b", "Bravo"}, {"c", "Charlie delta echo"}};
  const std::vector<KeyedDatum> second{{"c", "Cc"}, {"a", "Alpha with a much longer title"}, {"b", "B"}};
  // Long enough for every highlight to finish its tween.
  constexpr int SETTLE = 40;
  KeyedRun before, after, fresh;
  UiFrame frame;
  NavHarness nav;
  const auto draw_first = DrawKeyed(frame, first, before);
  nav.Frames(SETTLE, draw_first);
  // Down from the control above enters the list on its first row, and once more reaches the second.
  nav.Press(ImGuiKey_GamepadDpadDown, draw_first);
  nav.Press(ImGuiKey_GamepadDpadDown, draw_first);
  nav.Frames(SETTLE, draw_first);
  Check(before.focusedKey == "b", "focus starts on the row of key b");
  const auto draw_second = DrawKeyed(frame, second, after);
  nav.Frames(SETTLE, draw_second);
  Check(after.focusedKey == "b", "after the reorder, focus is still on the row of key b, now the third");

  UiFrame fresh_frame;
  NavHarness fresh_nav;
  const auto draw_fresh = DrawKeyed(fresh_frame, second, fresh);
  fresh_nav.Frames(SETTLE, draw_fresh);
  for (int press = 0; press < 3; press++) {
    fresh_nav.Press(ImGuiKey_GamepadDpadDown, draw_fresh);
  }
  fresh_nav.Frames(SETTLE, draw_fresh);
  Check(fresh.focusedKey == "b", "the baseline has focus on the row of key b too");
  Check(!SameShapes(before, after), "the changed data draws differently from the first data");
  Check(SameShapes(after, fresh), "each row draws the text it is given now");

  // The same order and keys with the first texts, which a row that kept its old text would draw.
  // It differs from the new data in nothing but the text, so the new data must not match it.
  const std::vector<KeyedDatum> old_texts{{"c", "Charlie delta echo"}, {"a", "Alpha"}, {"b", "Bravo"}};
  KeyedRun stale;
  UiFrame stale_frame;
  NavHarness stale_nav;
  const auto draw_stale = DrawKeyed(stale_frame, old_texts, stale);
  stale_nav.Frames(SETTLE, draw_stale);
  for (int press = 0; press < 3; press++) {
    stale_nav.Press(ImGuiKey_GamepadDpadDown, draw_stale);
  }
  stale_nav.Frames(SETTLE, draw_stale);
  Check(stale.focusedKey == "b" && !SameShapes(after, stale), "and not the text it had before");
}

// A double row draws its second line in the secondary colour, which a single row doesn't.
void TestSecondLine() {
  const auto draws_detail = [](bool with_detail) {
    NavHarness nav;
    UiFrame frame;
    nav.Frames(2, [&frame, with_detail](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const std::array<Element, 1> rows{ListRow(frame, "row", {"Title", with_detail ? "Detail" : ""})};
      const Element list = List(frame, "list", rows, {.width = Dp(LIST.width), .height = Dp(LIST.height)});
      const float unlimited = std::numeric_limits<float>::infinity();
      (void)Measure(frame, list, {unlimited, unlimited});
      (void)Arrange(frame, list, LIST);
      (void)Render(frame, list, context.list);
    });
    return HasColor(DEFAULT_THEME.colors.text.secondary);
  };
  Check(!draws_detail(false) && draws_detail(true), "a second line shows in the secondary colour");
}

// Focus moving down a long list scrolls it, keeping the focused row in view.
void TestScrolls() {
  Scene scene;
  scene.rows = 20;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  for (int press = 0; press < 16; press++) {
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
  }
  nav.Frames(10, draw);
  const float top = LIST.y + INSET + 15 * ROW - scene.scroll;
  Check(scene.focused_row == 15 && top >= LIST.y && top + ROW <= LIST.Bottom(),
        "focus deep in a long list scrolls it into view");
}

// A row outside the list's area can't be hovered or clicked, though it lies under the mouse. Row 9
// of 20 spans 649 to 705, wholly below the list's area, which ends at 540.
void TestHiddenRowsIgnoreTheMouse() {
  Scene scene;
  scene.rows = 20;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  // Clicks at a point, with a frame for the hover to settle before the press.
  const auto click = [&nav, &draw](float x, float y) {
    nav.MoveMouse(x, y, draw);
    nav.Frame(draw);
    nav.MouseButton(true);
    nav.Frame(draw);
    nav.MouseButton(false);
    nav.Frames(3, draw);
  };
  click(200, 670);
  // A click that reached the hidden row would focus it, and the list would scroll to it.
  Check(scene.activated == -1 && scene.focused_row == -1 && scene.scroll == 0.0f,
        "a row outside the list's area ignores the mouse");
  click(200, 180);
  Check(scene.activated == 0, "a row inside it takes the click");
}

// Leaves the list from row 2, by moving focus to the control above.
void LeaveFromRowTwo(NavHarness &nav, Scene &scene, const std::function<void(InteractContext &)> &draw) {
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(scene.focused_row == 2, "focus reaches row 2");
  scene.focus_above = true;
  nav.Frames(2, draw);
  Check(scene.above_focused, "focus leaves the list");
}

// A click onto another row stays on that row, though the list remembers row 2. Only the D-pad
// coming back in is sent to the remembered row.
void TestClickIsNotRedirected() {
  Scene scene;
  NavHarness nav;
  const std::function<void(InteractContext &)> draw = DrawScene(scene);
  LeaveFromRowTwo(nav, scene, draw);
  nav.MoveMouse(200, LIST.y + INSET + 4 * ROW + 20, draw);
  nav.Frame(draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(3, draw);
  Check(scene.focused_row == 4 && scene.activated == 4, "a click onto row 4 keeps focus on row 4");
}

// Focus isn't sent back to a remembered row that has since been disabled. Coming back in, it
// lands on row 0 instead.
void TestDisabledRowNotRestored() {
  Scene scene;
  NavHarness nav;
  const std::function<void(InteractContext &)> draw = DrawScene(scene);
  LeaveFromRowTwo(nav, scene, draw);
  scene.disabled = 2;
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(scene.focused_row == 0, "a disabled remembered row isn't restored");
}

// A screen without the list in between still counts as leaving it. The list isn't drawn for a
// few frames, then comes back, and the D-pad coming in returns to row 2.
void TestScreenChange() {
  Scene scene;
  NavHarness nav;
  const std::function<void(InteractContext &)> draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  scene.draw_list = false;
  scene.focus_above = true;
  nav.Frames(5, draw);
  scene.draw_list = true;
  nav.Frames(2, draw);
  Check(scene.above_focused, "focus is outside the list when it comes back");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(scene.focused_row == 2, "the D-pad returns to the row it left");
}

// With a control below a long list, up from that control lands on row 6, the last row in view,
// not row 9 or 10, the nearest to it, which are hidden below the list's area. The row is read on
// the frame the move lands, before the list would scroll to it.
void TestControlBelowTheList() {
  Scene scene;
  scene.rows = 20;
  scene.below = true;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  Check(scene.below_focused, "the control below starts with focus");
  nav.Key(ImGuiKey_GamepadDpadUp, true);
  nav.Frame(draw);
  nav.Key(ImGuiKey_GamepadDpadUp, false);
  nav.Frame(draw);
  Check(scene.focused_row == 6 && scene.scroll == 0.0f, "up lands on the last row wholly in view");
}

// A selected row scrolled out of the list's area isn't drawn, so its selection bar doesn't show.
// The same row in view is the baseline.
void TestHiddenRowsNotDrawn() {
  const auto bar_shown = [](int selected) {
    Scene scene;
    scene.rows = 20;
    scene.selected = selected;
    NavHarness nav;
    nav.Frames(2, DrawScene(scene));
    return HasColor(DEFAULT_THEME.colors.accent.line);
  };
  Check(bar_shown(3), "a selected row in view shows its bar");
  Check(!bar_shown(15), "one scrolled out of view isn't drawn");
}

// A list that shrinks after scrolling scrolls back, so its remaining rows show.
void TestShrinkingListScrollsBack() {
  Scene scene;
  scene.rows = 20;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  nav.Frames(2, draw);
  for (int press = 0; press < 16; press++) {
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
  }
  nav.Frames(5, draw);
  Check(scene.scroll > 0, "the long list has scrolled");
  scene.rows = 3;
  nav.Frames(2, draw);
  Check(scene.scroll == 0.0f, "shrunk to three rows, it scrolls back to show them at once");
}

} // namespace

int main() {
  TestHeadingKeepsItsPadding();
  TestRememberedRow();
  TestSelectedWithoutFocus();
  TestPosedRows();
  TestRowsOnLight();
  TestScrollBarShowsWhenRowsOverflow();
  TestWheelScrollsTheList();
  TestDraggingTheThumb();
  TestFocusReturningShowsItsRow();
  TestPosedListThumb();
  TestScrollingTextNeedsFocus();
  TestPosedListIgnoresTheWheel();
  TestDisabledRowSkipped();
  TestActivation();
  TestActivationIsReportedTwoWays();
  TestListOfRunsTheTemplateOnEachDatum();
  TestListOfFollowsChangedAndReorderedData();
  TestRowHeights();
  TestMixedRowHeightsAccumulate();
  TestScrollingToADoubleRowShowsAllOfIt();
  TestSecondLine();
  TestScrolls();
  TestHiddenRowsIgnoreTheMouse();
  TestClickIsNotRedirected();
  TestDisabledRowNotRestored();
  TestScreenChange();
  TestControlBelowTheList();
  TestShrinkingListScrollsBack();
  TestHiddenRowsNotDrawn();
  std::printf("System UI list tests passed\n");
  return 0;
}
