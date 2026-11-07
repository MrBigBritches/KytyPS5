#include "SystemUiNavHarness.h"

#include "SystemUiElementTestHooks.h"
#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/modal.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"

#include <cmath>
#include <optional>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <functional>
#include <limits>
#include <string>

namespace {

using namespace SystemUiTest;

constexpr Rect OPENER{100, 100, 240, 56};
constexpr Rect OUTSIDE{100, 900, 240, 56};
constexpr Rect CARD{660, 340, 600, 400};
constexpr Rect FIRST{720, 600, 220, 56};
constexpr Rect SECOND{980, 600, 220, 56};

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiModalTests: failed: %s\n", message);
    std::abort();
  }
}

// A button that opens a modal with two buttons, and a control outside it.
struct Scene {
  bool open = false;
  bool opener_focused = false;
  bool outside_focused = false;
  int outside_activations = 0;
  bool inside_focused = false;
  bool second_focused = false;
  bool visible = false;
  bool back_closes = true;
  // What the scrim covers, the whole output unless set.
  std::optional<Rect> area;
  // How many times the OnBack callback ran, and how many frames BackPressed was true.
  int back_callbacks = 0;
  int back_frames = 0;
  // The frame the modal's scrim and card are built in.
  UiFrame frame;
};

// Draws the scene. Back closes the modal unless `back_closes` is off, as for a dialog that must be
// answered.
auto DrawScene(Scene &scene) {
  return [&scene](InteractContext &context) {
    scene.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    if (Interact(context, ImGui::GetID("opener"), OPENER, false).state.activated) {
      scene.open = true;
    }
    scene.opener_focused = ImGui::GetFocusID() == ImGui::GetID("opener");
    const auto outside = Interact(context, ImGui::GetID("outside"), OUTSIDE, false);
    scene.outside_focused = outside.state.focused;
    scene.outside_activations += outside.state.activated ? 1 : 0;
    ModalScope modal(scene.frame, "dialog", CARD, scene.open, 1.0f, {}, scene.area);
    scene.visible = modal.Visible();
    scene.inside_focused = false;
    scene.second_focused = false;
    if (modal.Visible()) {
      const auto first = Interact(scene.frame, ImGui::GetID("first"), FIRST, false);
      const auto second = Interact(scene.frame, ImGui::GetID("second"), SECOND, false);
      scene.inside_focused = first.state.focused || second.state.focused;
      scene.second_focused = second.state.focused;
      scene.back_frames += modal.BackPressed() ? 1 : 0;
      modal.OnBack([&scene] { ++scene.back_callbacks; });
      if (modal.BackPressed() && scene.back_closes) {
        scene.open = false;
      }
    }
  };
}

// Opens the modal by confirming on the opener, which has the first focus.
void Open(NavHarness &nav, Scene &scene, const std::function<void(InteractContext &)> &draw) {
  nav.Frames(2, draw);
  Check(scene.opener_focused, "the opener starts with focus");
  nav.Press(ImGuiKey_GamepadFaceDown, draw);
  nav.Frames(3, draw);
  Check(scene.visible && scene.inside_focused, "confirm opens the modal and focus moves into it");
}

// Focus stays inside an open modal whatever the D-pad does. With it closed, down from the opener
// reaches the control outside, the baseline. ImGui keeps D-pad moves inside any focused window, so
// this guards against regressions only. TestClickOutside is what shows the dialog is modal.
void TestFocusTrap() {
  Scene closed;
  NavHarness baseline;
  const std::function<void(InteractContext &)> plain = DrawScene(closed);
  baseline.Frames(2, plain);
  baseline.Press(ImGuiKey_GamepadDpadDown, plain);
  Check(closed.outside_focused, "with no modal, down reaches the control outside");
  Scene scene;
  NavHarness nav;
  const std::function<void(InteractContext &)> draw = DrawScene(scene);
  Open(nav, scene, draw);
  for (const auto key : {ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadDpadUp, ImGuiKey_GamepadDpadLeft,
                         ImGuiKey_GamepadDpadRight, ImGuiKey_GamepadDpadDown, ImGuiKey_GamepadDpadUp}) {
    nav.Press(key, draw);
    Check(scene.inside_focused && !scene.outside_focused && !scene.opener_focused, "focus stays inside the modal");
  }
}

// Back closes the modal, and focus returns to the opener.
void TestBackRestoresFocus() {
  Scene scene;
  NavHarness nav;
  const std::function<void(InteractContext &)> draw = DrawScene(scene);
  Open(nav, scene, draw);
  nav.Key(ImGuiKey_GamepadFaceRight, true);
  nav.Frame(draw);
  Check(scene.back_frames == 1 && scene.back_callbacks == 1, "the close is reported by BackPressed and by the callback on the frame of the press");
  nav.Key(ImGuiKey_GamepadFaceRight, false);
  nav.Frames(4, draw);
  Check(!scene.visible && scene.opener_focused, "back closes the modal and focus returns to the opener");
  Check(scene.back_frames == 1 && scene.back_callbacks == 1, "and each reported it once");
}

// With confirm and back swapped, back is the bottom face button.
void TestSwappedBack() {
  Scene scene;
  NavHarness nav;
  ImGui::GetIO().ConfigNavSwapGamepadButtons = true;
  const std::function<void(InteractContext &)> draw = DrawScene(scene);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadFaceRight, draw);
  nav.Frames(3, draw);
  Check(scene.visible, "swapped, the right face button confirms");
  nav.Press(ImGuiKey_GamepadFaceDown, draw);
  nav.Frames(3, draw);
  Check(!scene.visible, "and the bottom one goes back");
}

// Calls `visit` with each vertex the modal's window drew in its last frame, in drawing order, with
// the clip rectangle it was drawn under.
template <class Visit> void ForEachModalVertex(Visit &&visit) {
  const ImGuiWindow *window = ImGui::FindWindowByName("dialog");
  Check(window != nullptr, "the modal has its own window");
  const ImDrawList &list = *window->DrawList;
  for (const ImDrawCmd &command : list.CmdBuffer) {
    for (unsigned int index = 0; index < command.ElemCount; index++) {
      visit(list.VtxBuffer[command.VtxOffset + list.IdxBuffer[command.IdxOffset + index]], command.ClipRect);
    }
  }
}

// True when `point` lies inside `clip`, so it shows.
bool Shows(const ImVec2 &point, const ImVec4 &clip) {
  return point.x >= clip.x && point.y >= clip.y && point.x <= clip.z && point.y <= clip.w;
}

// Draws the scene with the modal open for a few frames.
void DrawOpen(NavHarness &nav) {
  Scene scene;
  scene.open = true;
  nav.Frames(3, DrawScene(scene));
}

// The scrim covers the whole output, beneath the card. Its vertices span the output whatever the
// clip, so the draw commands that carry them must also clip to the whole output, not the card.
void TestScrim() {
  NavHarness nav;
  DrawOpen(nav);
  const ImVec2 display = ImGui::GetIO().DisplaySize;
  const ImU32 scrim = ToImU32(DEFAULT_THEME.colors.surface.scrim);
  bool top_left = false;
  bool bottom_right = false;
  bool clipped_to_output = true;
  ForEachModalVertex([&](const ImDrawVert &vertex, const ImVec4 &clip) {
    if (vertex.col == scrim) {
      top_left = top_left || (vertex.pos.x <= 0 && vertex.pos.y <= 0);
      bottom_right = bottom_right || (vertex.pos.x >= display.x && vertex.pos.y >= display.y);
      clipped_to_output = clipped_to_output && Shows({0, 0}, clip) && Shows(display, clip);
    }
  });
  Check(top_left && bottom_right, "the scrim covers the whole output");
  Check(clipped_to_output, "and isn't clipped to the card");
}

// Given an area, the scrim covers just that and is held to it, where TestScrim is the baseline that
// covers the whole output. Focus stays inside just the same.
void TestScrimCoversItsArea() {
  const Rect area{CARD.x - 60.0f, CARD.y - 60.0f, CARD.width + 120.0f, CARD.height + 120.0f};
  Scene scene;
  scene.area = area;
  NavHarness nav;
  const auto draw = DrawScene(scene);
  Open(nav, scene, draw);
  const ImU32 scrim = ToImU32(DEFAULT_THEME.colors.surface.scrim);
  float left = std::numeric_limits<float>::max(), top = left, right = -left, bottom = -left;
  bool held = false;
  ForEachModalVertex([&](const ImDrawVert &vertex, const ImVec4 &clip) {
    if (vertex.col == scrim) {
      left = std::min(left, vertex.pos.x);
      top = std::min(top, vertex.pos.y);
      right = std::max(right, vertex.pos.x);
      bottom = std::max(bottom, vertex.pos.y);
      held = clip.x >= area.x - 0.5f && clip.y >= area.y - 0.5f && clip.z <= area.Right() + 0.5f && clip.w <= area.Bottom() + 0.5f;
    }
  });
  Check(std::fabs(left - area.x) <= 1.0f && std::fabs(top - area.y) <= 1.0f && std::fabs(right - area.Right()) <= 1.0f && std::fabs(bottom - area.Bottom()) <= 1.0f, "the scrim covers its area and no more");
  Check(held, "and is held to it");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Frames(2, draw);
  Check(scene.inside_focused, "and focus stays inside");
}

// The scrim is drawn before the card, so the card sits on it rather than under it.
void TestScrimBeneathCard() {
  NavHarness nav;
  DrawOpen(nav);
  const ImU32 scrim = ToImU32(DEFAULT_THEME.colors.surface.scrim);
  const ImU32 card = ToImU32(DEFAULT_THEME.colors.surface.raised);
  int order = 0;
  int first_scrim = -1;
  int first_card = -1;
  ForEachModalVertex([&](const ImDrawVert &vertex, const ImVec4 &) {
    first_scrim = (vertex.col == scrim && first_scrim < 0) ? order : first_scrim;
    first_card = (vertex.col == card && first_card < 0) ? order : first_card;
    order++;
  });
  Check(first_scrim >= 0 && first_card >= 0 && first_scrim < first_card, "the scrim is drawn beneath the card");
}

// The card's shadow shows past the card's left edge. The shadow is the only thing in the shadow's
// colour drawn outside the card, apart from the scrim, which is a different alpha. The left edge is
// the one to check, as the shadow's drop puts even its innermost ring below the card, so only a
// shadow that really spreads reaches past the side.
void TestCardShadow() {
  NavHarness nav;
  DrawOpen(nav);
  const ImU32 rgb_mask = IM_COL32(0xFF, 0xFF, 0xFF, 0);
  const ImU32 shadow = ToImU32(DEFAULT_THEME.colors.effect.shadow) & rgb_mask;
  const ImU32 scrim = ToImU32(DEFAULT_THEME.colors.surface.scrim);
  bool shows_outside = false;
  ForEachModalVertex([&](const ImDrawVert &vertex, const ImVec4 &clip) {
    const bool outside = vertex.pos.x < CARD.x - 1;
    // The shadow's outer edge has no alpha, so only a vertex that has some shows it reaching this far.
    const bool visible = (vertex.col & IM_COL32_A_MASK) != 0;
    if ((vertex.col & rgb_mask) == shadow && vertex.col != scrim && outside && visible && Shows(vertex.pos, clip)) {
      shows_outside = true;
    }
  });
  Check(shows_outside, "the card's shadow shows past the card's side");
}

// A dialog taller than the output is cut to fit, so its card starts on screen.
void TestTallDialogFits() {
  constexpr float TOO_TALL = 5000;
  Rect card;
  NavHarness nav;
  nav.Frame([&card, TOO_TALL](UiFrame &frame) { card = DialogCard(frame, Dp(TOO_TALL)); });
  Check(card.y >= 0 && card.Bottom() <= ImGui::GetIO().DisplaySize.y, "a tall dialog's card fits the output");
}

// A dialog that must be answered ignores back, and with the system UI's navigation settings focus
// stays on its button. Without them, the baseline, ImGui drops focus on back.
void TestMustAnswer() {
  for (const bool configured : {false, true}) {
    Scene scene;
    scene.back_closes = false;
    NavHarness nav;
    if (configured) {
      ConfigureNavigation(DEFAULT_THEME);
    }
    const std::function<void(InteractContext &)> draw = DrawScene(scene);
    Open(nav, scene, draw);
    nav.Press(ImGuiKey_GamepadDpadRight, draw);
    nav.Press(ImGuiKey_GamepadFaceRight, draw);
    nav.Frames(2, draw);
    Check(scene.visible, "a dialog that must be answered stays open on back");
    Check(scene.second_focused == configured, configured ? "and keeps focus on its button" : "the baseline drops focus");
  }
}

// A click outside the card neither closes the modal nor reaches the control under it. Focus starts
// on the second button, since a popup closed and opened again would put it back on the first.
void TestClickOutside() {
  Scene scene;
  NavHarness nav;
  const std::function<void(InteractContext &)> draw = DrawScene(scene);
  Open(nav, scene, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(scene.second_focused, "right moves focus to the second button");
  nav.MoveMouse(OUTSIDE.x + 20, OUTSIDE.y + 20, draw);
  nav.Frame(draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(3, draw);
  Check(scene.visible && scene.second_focused && scene.outside_activations == 0,
        "a click outside doesn't close it, move its focus, or reach what's under it");
}

// ---- DialogContent -----------------------------------------------------------------------------

// A dialog's content built and rendered each frame in a rectangle of its own, with two buttons, and
// where everything landed. Independent of any other layout, it is what the claims about
// DialogContent are made on.
struct DialogScene {
  std::string title = "Delete this save?";
  std::string body = "This cannot be undone, and nothing can be brought back once it has gone from this console.";
  std::string yesLabel = "Delete";
  std::string noLabel = "Keep";
  Rect content{100, 100, 500, 300};
  int yes = 0;
  int no = 0;
  Rect titleRect, bodyRect, yesRect, noRect;
  // How many glyph quads the title and the second button's label drew. A label drawn whole has one
  // for each letter, and one cut short with an ellipsis has fewer.
  int titleGlyphs = 0;
  int noGlyphs = 0;
  // Whether the title and that label drew an ellipsis, which a text cut short by a clip does not.
  bool titleEllipsis = false;
  bool noEllipsis = false;
  UiFrame frame;
};

// Counts activations.
struct CountActivation {
  int *count;
  void operator()(Element) const { ++*count; }
};

// How many glyph quads in the primary text colour `list` drew inside `area`. Used on a title and on
// a button that has no focus ring, which is in the same colour.
int GlyphsIn(const ImDrawList &list, const Rect &area) {
  const ImU32 colour = ToImU32(DEFAULT_THEME.colors.text.primary);
  int vertices = 0;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if (vertex.col == colour && vertex.pos.x >= area.x - 1 && vertex.pos.x <= area.Right() + 1 && vertex.pos.y >= area.y - 1 && vertex.pos.y <= area.Bottom() + 1) {
      ++vertices;
    }
  }
  return vertices / 4;
}

// Whether `list` drew an ellipsis inside `area`: a glyph quad whose first corner has the texture
// coordinates of the font's ellipsis, or of a full stop, which the text falls back on. The texts the
// tests use have neither, so a quad like that is the ellipsis, and text cut by a clip has none.
bool DrewEllipsis(const ImDrawList &list, const Rect &area, const UiContext &ui, const TypeStyle &style) {
  // The font the text was drawn in, which is the theme's for its weight, or ImGui's own where the
  // harness has none, and the glyphs it really has, since a missing one would fall back on another.
  ImFont *font = ui.fonts.For(style.weight);
  font = font != nullptr ? font : ImGui::GetFont();
  ImFontBaked *baked = font->GetFontBaked(ui.scale.FontSize(style));
  const ImFontGlyph *marks[2] = {baked->FindGlyphNoFallback(0x2026), baked->FindGlyphNoFallback('.')};
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if (vertex.pos.x < area.x - 1 || vertex.pos.x > area.Right() + 1 || vertex.pos.y < area.y - 1 || vertex.pos.y > area.Bottom() + 1) {
      continue;
    }
    for (const ImFontGlyph *mark : marks) {
      if (mark != nullptr && vertex.uv.x == mark->U0 && vertex.uv.y == mark->V0) {
        return true;
      }
    }
  }
  return false;
}

// Draws the dialog scene, and notes the rectangles it laid out.
auto DrawDialogScene(DialogScene &scene) {
  return [&scene](InteractContext &context) {
    scene.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    const Element yes = Button(scene.frame, "yes", {.label = scene.yesLabel});
    const Element no = Button(scene.frame, "no", {.label = scene.noLabel});
    (void)OnActivate(scene.frame, yes, CountActivation{&scene.yes});
    (void)OnActivate(scene.frame, no, CountActivation{&scene.no});
    const std::array<Element, 2> buttons{yes, no};
    const DialogParts parts = DialogContent(scene.frame, "content", {scene.title, scene.body}, buttons);
    (void)MeasureDialog(scene.frame, parts, scene.content.width);
    (void)Arrange(scene.frame, parts.content, scene.content);
    (void)Render(scene.frame, parts.content, context.list);
    scene.titleRect = Arranged(scene.frame, parts.title);
    scene.bodyRect = Arranged(scene.frame, parts.body);
    scene.yesRect = Arranged(scene.frame, yes);
    scene.noRect = Arranged(scene.frame, no);
    scene.titleGlyphs = GlyphsIn(context.list, scene.titleRect);
    scene.noGlyphs = GlyphsIn(context.list, scene.noRect);
    scene.titleEllipsis = DrewEllipsis(context.list, scene.titleRect, context.ui, DEFAULT_THEME.type.heading);
    scene.noEllipsis = DrewEllipsis(context.list, scene.noRect, context.ui, DEFAULT_THEME.type.label);
  };
}

// A body long enough to overflow a dialog of the scene's size at any output the tests use.
std::string LongBody() {
  std::string body;
  for (int sentence = 0; sentence < 40; ++sentence) {
    body += "Every backup of the save goes with it and nothing can be brought back. ";
  }
  return body;
}

// True when `inner` lies inside `outer`, to within a hair for rounding.
bool Inside(const Rect &inner, const Rect &outer) {
  constexpr float HAIR = 0.01f;
  return inner.x >= outer.x - HAIR && inner.y >= outer.y - HAIR && inner.Right() <= outer.Right() + HAIR && inner.Bottom() <= outer.Bottom() + HAIR;
}

// What DialogContent promises when everything fits, held to the content rectangle and the theme and
// to LayoutText, and not to any other drawing of a dialog: the title is at the content's top left,
// the body starts a gap below it and has the lines LayoutText gives at that width, and the last
// button's right and bottom edges are the content's, with the theme's gap between the buttons.
void TestDialogContentSpec() {
  DialogScene scene;
  NavHarness nav;
  nav.Frames(2, DrawDialogScene(scene));
  const UiContext ui{DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, 1920, 1080), {}};
  const float gap = ui.scale.Px(DEFAULT_THEME.space.s4);
  const Rect &content = scene.content;
  Check(scene.titleRect.x == content.x && scene.titleRect.y == content.y, "the title is at the content's top left");
  Check(scene.bodyRect.x == content.x && scene.bodyRect.y == scene.titleRect.Bottom() + gap, "the body's top is the title's bottom plus the gap");
  TextProps body{scene.body, DEFAULT_THEME.type.body, DEFAULT_THEME.colors.text.secondary};
  body.overflow = TextOverflow::Wrap;
  const TextLayout expected = LayoutText(ui, body, content.width);
  Check(expected.lines.size() > 1, "the body wraps to more than one line, so the line count says something");
  Check(scene.bodyRect.height == expected.size.y, "the body has as many lines as LayoutText gives at the content's width");
  Check(scene.noRect.Right() == content.Right() && scene.noRect.Bottom() == content.Bottom(), "the last button sits on the content's right and bottom edges");
  Check(scene.noRect.y == scene.yesRect.y, "the buttons share a row");
  Check(scene.noRect.x - scene.yesRect.Right() == gap, "and are the theme's gap apart");
}

// A card sized to its dialog with FitDialogCard is as tall as the dialog needs, so a body of several
// lines gives a taller card than one of a single line, and the dialog laid out in it keeps its buttons
// on the content's bottom edge. The one-line body is the baseline.
void TestCardFitsItsDialog() {
  float heights[2] = {};
  for (const bool multiLine : {false, true}) {
    DialogScene scene;
    scene.body = multiLine ? "One sentence after another, and a good many of them, so that the body runs over several lines of the card." : "Short.";
    NavHarness nav;
    float height = 0;
    nav.Frame([&](InteractContext &context) {
      scene.frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Element yes = Button(scene.frame, "yes", {.label = "Delete"});
      const std::array<Element, 1> buttons{yes};
      const DialogParts parts = DialogContent(scene.frame, "content", {scene.title, scene.body}, buttons);
      const Rect card = FitDialogCard(scene.frame, parts);
      height = card.height;
      Check(card.height < 1080, "the card fits the output");
    });
    heights[multiLine ? 1 : 0] = height;
  }
  Check(heights[0] > 0 && heights[1] > heights[0], "a card for a body of several lines is taller than one for a single line");
}

// ModalScope::Place measures the dialog at the card's content width, so the body has the lines
// LayoutText gives at that width and not the one line it would have measured with no limit.
void TestPlaceMeasuresAtTheContentWidth() {
  UiFrame frame;
  NavHarness nav;
  float bodyHeight = 0;
  float expectedHeight = 0;
  bool placed = false;
  nav.Frames(3, [&](InteractContext &context) {
    frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    ModalScope modal(frame, "dialog", CARD, true);
    if (!modal.Visible()) {
      return;
    }
    const Element yes = Button(frame, "yes", {.label = "Delete"});
    const std::array<Element, 1> buttons{yes};
    const std::string text = "This cannot be undone, and nothing can be brought back once it has gone from this console.";
    const DialogParts parts = DialogContent(frame, "content", {"Delete this save?", text}, buttons);
    placed = modal.Place(frame, parts);
    TextProps body{text, DEFAULT_THEME.type.body, DEFAULT_THEME.colors.text.secondary};
    body.overflow = TextOverflow::Wrap;
    expectedHeight = LayoutText(context.ui, body, modal.Content().width).size.y;
    bodyHeight = Arranged(frame, parts.body).height;
  });
  Check(placed && expectedHeight > 0, "the dialog was placed in the card");
  Check(bodyHeight == expectedHeight, "the body has the lines of the card's content width");
}

// Place keeps a measurement made in the room it is laying out in. A card sized with FitDialogCard has
// already measured the dialog at its content width, so Place measures nothing more, where a card
// of another width, the baseline, makes it measure again. The count is of nodes measured, since a
// re-measure shows there. The count of texts laid out again at arrange, which is what TextRelayouts
// reports, is the same either way, two for the title and the body that stretch to the content's
// width, so skipping the measurement changes nothing about how the dialog comes out.
void TestPlaceReusesTheMeasurement() {
  uint32_t relayoutsSeen[2] = {};
  const auto check_with = [&](const char *words) {
  for (const bool fitted : {true, false}) {
    UiFrame frame;
    NavHarness nav;
    uint32_t measuredBefore = 0;
    uint32_t measuredAfter = 0;
    uint32_t relayouts = 0;
    nav.Frames(3, [&](InteractContext &context) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
      const Element yes = Button(frame, "yes", {.label = "Delete"});
      const std::array<Element, 1> buttons{yes};
      const DialogParts parts = DialogContent(frame, "content", {"Delete this save?", words}, buttons);
      // A card of the dialog's own fitted size, or the fixed one, which is not the width measured.
      const Rect card = fitted ? FitDialogCard(frame, parts) : CARD;
      ModalScope modal(frame, "dialog", card, true);
      measuredBefore = FrameTestHooks::MeasureCalls(frame);
      (void)modal.Place(frame, parts);
      measuredAfter = FrameTestHooks::MeasureCalls(frame);
      relayouts = FrameTestHooks::TextRelayouts(frame);
    });
    if (fitted) {
      Check(measuredAfter == measuredBefore, "Place measured nothing more for the card FitDialogCard sized");
    } else {
      Check(measuredAfter > measuredBefore, "Place measures a dialog that wasn't measured in the card's room");
    }
    relayoutsSeen[fitted ? 1 : 0] = relayouts;
  }
  };
  // A body that wraps reuses its measurement too, where how many lines it has can change what stretches, so
  // the count of texts is taken with one line, whose width always differs from the content's.
  check_with("This cannot be undone, and nothing can be brought back once it has gone from this console.");
  check_with("This cannot be undone.");
  Check(relayoutsSeen[0] == relayoutsSeen[1] && relayoutsSeen[0] == 2, "the same two texts, the title and the body, stretch to the content width either way");
}

// A title wider than the content is cut short on one line, with an ellipsis, so it neither wraps nor
// pushes the buttons off the card. The short title is the baseline, which keeps the same line height.
void TestLongTitleIsCutShort() {
  DialogScene shortTitle;
  DialogScene longTitle;
  shortTitle.title = "Heading";
  longTitle.title = std::string(200, 'W');
  NavHarness nav;
  nav.Frames(2, DrawDialogScene(longTitle));
  NavHarness baseline;
  baseline.Frames(2, DrawDialogScene(shortTitle));
  Check(longTitle.titleRect.height == shortTitle.titleRect.height, "a long title stays on one line");
  Check(shortTitle.titleGlyphs == 7, "a title that fits draws a glyph for each letter, which is how glyphs are counted");
  Check(longTitle.titleGlyphs > 0 && longTitle.titleGlyphs < 200, "a title too wide draws fewer glyphs than it has letters");
  Check(longTitle.titleEllipsis && !shortTitle.titleEllipsis, "and ends in an ellipsis, which a title that fits does not draw");
  Check(Inside(longTitle.titleRect, longTitle.content), "and inside the content");
  Check(longTitle.noRect.Right() == longTitle.content.Right() && longTitle.noRect.Bottom() == longTitle.content.Bottom(), "the buttons stay on the content's edges");
}

// Buttons whose labels together are wider than the card shrink to fit it and keep the last one on
// the content's right edge, which is what a dialog that must be answered needs more than full
// labels. The same buttons with short labels are the baseline, where they keep their natural width.
void TestWideButtonsShrinkToTheCard() {
  DialogScene normal;
  DialogScene wide;
  wide.yesLabel = "DeleteThisSaveAndAllOfItsBackups";
  wide.noLabel = "KeepEverythingExactlyAsItIs";
  normal.noLabel = "KeepAsIs";
  wide.content = {100, 100, 300, 300};
  normal.content = wide.content;
  NavHarness nav;
  nav.Frames(2, DrawDialogScene(wide));
  NavHarness baseline;
  baseline.Frames(2, DrawDialogScene(normal));
  Check(Inside(normal.yesRect, normal.content) && Inside(normal.noRect, normal.content), "short buttons fit");
  Check(normal.noGlyphs == 8, "a label that fits draws a glyph for each letter, which is how glyphs are counted");
  Check(wide.noGlyphs > 0 && wide.noGlyphs < 27, "a label cut to fit draws fewer glyphs than it has letters");
  Check(wide.noEllipsis && !normal.noEllipsis, "and ends in an ellipsis, which a label that fits does not draw");
  Check(wide.noRect.Right() == wide.content.Right(), "wide buttons keep the last one on the right edge");
  Check(Inside(wide.yesRect, wide.content) && Inside(wide.noRect, wide.content), "and shrink to stay inside the content");
}

// With more to say than fits, the buttons stay on the card and answer to the mouse and the D-pad, as
// they do when everything fits, which is the baseline. The body is the part that gives way.
void TestOverflowingDialogCanBeAnswered() {
  for (const bool overflowing : {false, true}) {
    DialogScene scene;
    if (overflowing) {
      scene.body = LongBody();
    }
    NavHarness nav;
    const auto draw = DrawDialogScene(scene);
    nav.Frames(3, draw);
    Check(Inside(scene.yesRect, scene.content) && Inside(scene.noRect, scene.content), overflowing ? "an overflowing dialog's buttons stay inside its content" : "the baseline's buttons are inside");
    Check(scene.noRect.Bottom() == scene.content.Bottom(), "resting on its bottom edge");
    if (overflowing) {
      Check(scene.bodyRect.Bottom() > scene.noRect.y, "the body is taller than the room it has left, so it is the part that gives way");
    }
    nav.Press(ImGuiKey_GamepadFaceDown, draw);
    Check(scene.yes == 1, "the D-pad's confirm answers the first button");
    nav.Press(ImGuiKey_GamepadDpadRight, draw);
    nav.Press(ImGuiKey_GamepadFaceDown, draw);
    Check(scene.no == 1, "right then confirm answers the second");
    const Rect target = scene.yesRect;
    nav.MoveMouse(target.x + target.width / 2, target.y + target.height / 2, draw);
    nav.Frame(draw);
    nav.MouseButton(true);
    nav.Frame(draw);
    nav.MouseButton(false);
    nav.Frames(2, draw);
    Check(scene.yes == 2, "and a click on a button answers it");
  }
}

// ---- Keys ---------------------------------------------------------------------------------------

// Whether a layer behind an open modal that claims the shoulders and back and reads them as its own
// saw R1 and back. It claims first each frame, as a bar drawn before the dialog does.
struct BehindReads {
  bool shoulders = false;
  bool back = false;
};

// Opens the modal, settles, presses R1 and then back, and says whether a Control layer behind it
// saw either. `configured` is whether the system UI's navigation settings are on, which is whether
// back keeps focus where it was.
BehindReads RunBehind(bool configured, bool modal_open) {
  Scene scene;
  scene.open = modal_open;
  scene.back_closes = false;
  BehindReads seen;
  NavHarness nav;
  if (configured) {
    ConfigureNavigation(DEFAULT_THEME);
  }
  const auto scene_draw = DrawScene(scene);
  const auto draw = [&](InteractContext &context) {
    const ItemId behind = ImGui::GetID("layer behind");
    (void)ClaimKeys(KeyGroup::Shoulders, behind, ClaimPriority::Control);
    (void)ClaimKeys(KeyGroup::Back, behind, ClaimPriority::Control);
    seen.shoulders = seen.shoulders || ShoulderPresses(behind).right;
    seen.back = seen.back || BackKeyPressed(behind);
    scene_draw(context);
  };
  nav.Frames(4, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  nav.Press(ImGuiKey_GamepadFaceRight, draw);
  return seen;
}

// An open modal takes the shoulders from a layer behind it, which claimed them first. With the modal
// closed the same layer sees R1, so the press does reach it, which is the baseline.
void TestModalTakesTheShoulders() {
  Check(RunBehind(true, false).shoulders, "with no modal, a layer behind sees R1");
  Check(!RunBehind(true, true).shoulders, "an open modal takes R1 from a layer behind it");
}

// An open modal owns back only where ImGui's cancel leaves focus alone, which is with the system UI's
// navigation settings. Without them cancel is what drops focus, and owning back would stop that, so
// the layer behind still sees back. With them it doesn't.
void TestModalOwnsBackOnlyWhenCancelKeepsFocus() {
  Check(RunBehind(true, false).back, "with no modal, a layer behind sees back");
  Check(!RunBehind(true, true).back, "with CancelKeepsFocus, an open modal owns back");
  Check(RunBehind(false, true).back, "without it, the modal leaves back alone");
}

} // namespace

int main() {
  TestFocusTrap();
  TestBackRestoresFocus();
  TestSwappedBack();
  TestScrim();
  TestScrimCoversItsArea();
  TestScrimBeneathCard();
  TestCardShadow();
  TestMustAnswer();
  TestTallDialogFits();
  TestClickOutside();
  TestDialogContentSpec();
  TestCardFitsItsDialog();
  TestPlaceMeasuresAtTheContentWidth();
  TestPlaceReusesTheMeasurement();
  TestLongTitleIsCutShort();
  TestWideButtonsShrinkToTheCard();
  TestOverflowingDialogCanBeAnswered();
  TestModalTakesTheShoulders();
  TestModalOwnsBackOnlyWhenCancelKeepsFocus();
  std::printf("System UI modal tests passed\n");
  return 0;
}
