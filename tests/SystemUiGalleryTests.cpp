#include "SystemUiNavHarness.h"
#include "SystemUiTabsKit.h"

#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/list.h"
#include "graphics/presentation/system/components/modal.h"
#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/animation/easing.h"
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/components/toggle.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/gallery/capture.h"
#include "graphics/presentation/system/gallery/effectsPage.h"
#include "graphics/presentation/system/gallery/colourPage.h"
#include "graphics/presentation/system/gallery/options.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/shell.h"
#include "graphics/presentation/system/gallery/stage.h"
#include "graphics/presentation/system/gallery/toastPage.h"
#include "graphics/presentation/system/gallery/typographyPage.h"
#include "graphics/presentation/system/imgui/interactContext.h"
#include "graphics/presentation/system/imgui/scaledDraw.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/imgui/fonts.h"
#include "graphics/presentation/system/imgui/imageStore.h"
#include "graphics/presentation/system/views/trophyToast.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <filesystem>
#include <fstream>
#include <span>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "stb_image.h"
#pragma clang diagnostic pop

namespace {

using namespace SystemUiTest;
using namespace Libs::Graphics::SystemUi::Gallery;

constexpr Rect SCREEN{0, 0, 1920, 1080};

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiGalleryTests: failed: %s\n", message);
    std::abort();
  }
}

// The page whose control had focus in the last frame, or -1.
int g_item_focused_on = -1;

// A page with two controls, named the same on every such page. "item", which focus enters the
// page on, is at the top, right of "beside", which sits lower at the pane's left edge, level with
// the sidebar's first row. ImGui's left only reaches a sidebar row from a control level with one.
void DrawItem(UiFrame &frame, GalleryState &state, const PageEntry &, const Rect &pane) {
  state.entry = IdFor("item");
  if (Interact(frame, state.entry, {pane.x + 300, pane.y, 200, 56}, false).state.focused) {
    g_item_focused_on = state.page;
  }
  (void)Interact(frame, IdFor("beside"), {pane.x, pane.y + 60, 200, 56}, false);
}

// A page with no controls.
void DrawNothing(UiFrame &, GalleryState &, const PageEntry &, const Rect &) {}

// Five pages over three groups, so the sidebar reads heading, a, b, heading, c, heading, d, e.
// Page e has no controls.
constexpr std::array TEST_PAGES{PageEntry{PageGroup::Styles, "a", "A", DrawItem}, PageEntry{PageGroup::Styles, "b", "B", DrawItem},
                                PageEntry{PageGroup::Components, "c", "C", DrawItem},
                                PageEntry{PageGroup::Patterns, "d", "D", DrawItem},
                                PageEntry{PageGroup::Patterns, "e", "E", DrawNothing}};

// Draws the gallery over the test pages.
auto DrawTestGallery(GalleryState &state) {
  return [&state](UiFrame &frame) {
    g_item_focused_on = -1;
    DrawGallery(frame, state, TEST_PAGES, SCREEN);
  };
}

// A harness with the system UI's navigation, as the gallery runs.
HarnessOptions SystemUiNavigation() {
  HarnessOptions options;
  options.configure_navigation = true;
  return options;
}

// The gallery's own pages have unique ids, titles and drawing, and list their groups in order,
// each group's pages together, with every group present.
void TestRegistry() {
  std::set<std::string> ids;
  int group = 0;
  std::set<int> groups;
  for (const auto &page : Pages()) {
    Check(ids.insert(std::string(page.id)).second, "page ids are unique");
    Check(!page.title.empty() && page.draw != nullptr, "each page has a title and draws");
    Check(static_cast<int>(page.group) >= group, "groups are listed in order, each together");
    group = static_cast<int>(page.group);
    groups.insert(group);
  }
  Check(groups.size() == 3, "every group has pages");
}

// The first row has the first focus. Down moves along the sidebar, skipping headings, and shows
// each page focus lands on.
void TestSidebarShowsFocusedPage() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  nav.Frames(2, draw);
  Check(state.page == 0 && g_item_focused_on == -1, "the first row starts with focus");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(state.page == 1, "down shows the next page");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(state.page == 2, "and skips the heading between groups");
}

// Moves focus from the first row down to page c's row.
template <class Draw> void FocusRowC(NavHarness &nav, Draw &&draw) {
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
}

// Right enters the page on its entry control, though the control lies above the row, where the
// D-pad's right alone wouldn't reach it. Back returns focus to the page's row, so down then shows
// the page after. Without back, focus would stay on the page's control, where down finds nothing.
void TestBackReturnsToRow() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  FocusRowC(nav, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(g_item_focused_on == 2, "right enters the page");
  nav.Press(ImGuiKey_GamepadFaceRight, draw);
  Check(g_item_focused_on == -1 && state.page == 2, "back leaves the page, which stays shown");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(state.page == 3, "focus was on the page's row");
}

// In the sidebar, R1 and L1 show the page after and before, stopping at the ends, and focus moves
// with them, so down afterwards continues from the new row.
void TestShoulders() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadL1, draw);
  Check(state.page == 0, "L1 stops at the first page");
  nav.Press(ImGuiKey_GamepadR1, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(state.page == 2 && g_item_focused_on == -1, "R1 shows the page after, and focus stays in the sidebar");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(state.page == 3, "focus moved with the page");
  nav.Press(ImGuiKey_GamepadR1, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(state.page == 4, "R1 stops at the last page");
}

// Inside a page, R1 takes focus along to the next page's entry control, since the old page's
// controls are gone. A page with no controls takes it to its row instead, so up then shows the
// page before.
void TestShouldersInsideAPage() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  FocusRowC(nav, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(state.page == 3 && g_item_focused_on == 3, "R1 takes focus to the next page's entry");
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(state.page == 4 && g_item_focused_on == -1, "a page with no controls");
  nav.Press(ImGuiKey_GamepadDpadUp, draw);
  Check(state.page == 3, "takes focus to its row");
}

// Each page's controls are named within it. With focus on page c's control, the page changes
// without the gallery moving focus, and page d's control of the same name doesn't inherit it.
void TestPagesHaveTheirOwnNames() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  FocusRowC(nav, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(g_item_focused_on == 2, "page c's control has focus");
  state.page = 3;
  nav.Frames(2, draw);
  Check(g_item_focused_on == -1, "page d's control of the same name has no focus");
}

// Coming back to the sidebar after R1 changed the page from inside one, focus lands on the shown
// page's row, not row a, the row nearest the way in, so the page stays. Down goes from the entry
// control to "beside", and left from there reaches row a.
void TestFocusReturnsToShownPage() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(g_item_focused_on == 0, "right enters the first page");
  nav.Press(ImGuiKey_GamepadR1, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(state.page == 2 && g_item_focused_on == 2, "R1 changes the page from inside one");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(g_item_focused_on == -1, "down moves to the control beside the sidebar");
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  nav.Frames(2, draw);
  Check(state.page == 2, "left returns to the sidebar on the shown page");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(state.page == 3, "with focus on its row");
}

// Up from the entry control, the topmost of all, finds nothing and leaves focus there, since only
// left returns to the sidebar.
void TestOtherMovesStayInThePage() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  FocusRowC(nav, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(g_item_focused_on == 2, "right enters the page");
  nav.Press(ImGuiKey_GamepadDpadUp, draw);
  Check(state.page == 2 && g_item_focused_on == 2, "up with nothing above stays on the control");
}

// Left from the entry control, with no row or control level with it, still returns to the shown
// page's row, so down then shows the page after. ImGui alone would leave focus where it was.
void TestLeftWithNothingBeside() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  FocusRowC(nav, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(g_item_focused_on == 2, "right enters the page");
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(g_item_focused_on == -1 && state.page == 2, "left leaves the page, which stays shown");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(state.page == 3, "focus was on the page's row");
}

// A click on a row shows its page. Row d is the seventh line, about 357 to 413 down.
void TestClickShowsPage() {
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  nav.Frames(2, draw);
  nav.MoveMouse(100, 385, draw);
  nav.Frame(draw);
  nav.MouseButton(true);
  nav.Frame(draw);
  nav.MouseButton(false);
  nav.Frames(2, draw);
  Check(state.page == 3, "a click on a row shows its page");
}

// At 4K the page pane still leaves room on every side for a focus ring around a control at its
// edge, the ring, its keyline and its glow.
void TestPaneLeavesRoomForTheRing() {
  HarnessOptions options;
  options.width = 3840;
  options.height = 2160;
  NavHarness nav(options);
  nav.Frame([](UiFrame &frame) {
    const auto &theme = frame.ActiveTheme();
    const auto &scale = frame.Scale();
    const float reach = scale.Border(theme.border.thick) + scale.Border(theme.border.thin) + scale.Px(theme.glow.focus.radius);
    const Rect screen = scale.Screen();
    const Rect pane = PagePane(frame, screen);
    const Rect sidebar = SidebarRect(frame, screen);
    Check(pane.x - sidebar.Right() >= reach && pane.y - screen.y >= reach && screen.Right() - pane.Right() >= reach &&
              screen.Bottom() - pane.Bottom() >= reach,
          "the pane leaves room for the ring on every side");
  });
}

// A page asked for from outside is shown, with focus on its row, so down then shows the page after.
// The request is cleared, and one past the end shows the last page. Setting the page alone, the
// baseline, is undone by the focused first row.
void TestShowRequest() {
  GalleryState plain;
  NavHarness baseline(SystemUiNavigation());
  const auto plain_draw = DrawTestGallery(plain);
  baseline.Frames(2, plain_draw);
  plain.page = 2;
  baseline.Frames(2, plain_draw);
  Check(plain.page == 0, "setting the page alone is undone by the focused row");
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawTestGallery(state);
  nav.Frames(2, draw);
  state.show = 2;
  nav.Frames(2, draw);
  Check(state.page == 2 && state.show == -1, "a page asked for is shown and the request cleared");
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(state.page == 3, "with focus on its row");
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(g_item_focused_on == 3, "right enters page d");
  state.show = 1;
  nav.Frames(2, draw);
  Check(state.page == 1 && g_item_focused_on == -1, "from inside a page, focus goes to the row, not the new page's entry");
  // Up from row b would reach page b's "beside" control, which ImGui scores nearer than row a.
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(state.page == 2, "so down shows the page after");
  state.show = 99;
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadUp, draw);
  Check(state.page == 3, "past the end shows the last page, with focus on its row");
}

// With no pages the gallery draws nothing and leaves the state alone.
void TestNoPages() {
  GalleryState state;
  state.page = 5;
  NavHarness nav;
  nav.Frame([&state](UiFrame &frame) { DrawGallery(frame, state, {}, SCREEN); });
  Check(state.page == 5, "with no pages nothing changes");
}

// A frame that is not begun again for a new ImGui frame is reported by the gallery, which debug builds
// also assert on, so the skipped Begin runs only where asserts are off. While the frame is begun every
// ImGui frame it is not reported, which is the baseline, and begun again after a skip it is clear.
void TestStaleFrameIsReported() {
  GalleryState state;
  NavHarness nav;
  UiFrame frame;
  bool begin = true;
  const auto draw = [&](InteractContext &context) {
    if (begin) {
      frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
    }
    DrawGallery(frame, state, TEST_PAGES, SCREEN);
  };
  nav.Frames(3, draw);
  Check(!state.frame_was_stale, "a frame begun for each ImGui frame is not reported");
#ifdef NDEBUG
  begin = false;
  nav.Frame(draw);
  Check(state.frame_was_stale, "a frame not begun for the next ImGui frame is reported");
  begin = true;
  nav.Frame(draw);
  Check(!state.frame_was_stale, "and begun again it is clear");
#endif
}

// Parses `args` as the command line after the program name.
ParseResult Parse(std::vector<std::string_view> args) { return ParseGalleryOptions(args); }

// Whether `text` mentions `part`, which an error must do to tell the user what was wrong.
bool Mentions(const std::string &text, const char *part) { return text.find(part) != std::string::npos; }

// With no arguments every option has its default, and nothing is an error.
void TestOptionDefaults() {
  const ParseResult none = Parse({});
  Check(none.Ok(), "no arguments is valid");
  Check(none.options.width == 1920 && none.options.height == 1080, "the default size is 1920x1080");
  Check(none.options.screenshot_dir.empty() && !none.options.help, "no screenshot folder and no help by default");
  Check(DEFAULT_WIDTH == 1920 && DEFAULT_HEIGHT == 1080, "the default constants are the size the spec gives");
}

// --screenshot stores its folder and changes nothing else. The same parse without the flag is the
// baseline, so a parser that always filled the folder, or took the folder for the size, would fail.
void TestOptionScreenshot() {
  const ParseResult with = Parse({"--screenshot", "shots/out"});
  Check(with.Ok(), "--screenshot with a folder is valid");
  Check(with.options.screenshot_dir == "shots/out", "--screenshot stores its folder");
  Check(with.options.width == 1920 && with.options.height == 1080 && !with.options.help, "--screenshot leaves the rest at their defaults");
  Check(Parse({"--width", "640"}).options.screenshot_dir.empty(), "without --screenshot the folder stays empty");
}

// --width and --height each set their own value. Giving only one leaves the other at its default,
// and the values differ from each other, so a swapped pair is caught.
void TestOptionSize() {
  const ParseResult both = Parse({"--width", "800", "--height", "600"});
  Check(both.Ok(), "--width and --height are valid");
  Check(both.options.width == 800 && both.options.height == 600, "the sizes land in their own fields");
  const ParseResult reversed = Parse({"--height", "600", "--width", "800"});
  Check(reversed.options.width == 800 && reversed.options.height == 600, "the order of the flags doesn't matter");
  const ParseResult width_only = Parse({"--width", "800"});
  Check(width_only.options.width == 800 && width_only.options.height == 1080, "--width alone leaves the height");
  const ParseResult height_only = Parse({"--height", "600"});
  Check(height_only.options.height == 600 && height_only.options.width == 1920, "--height alone leaves the width");
  Check(Parse({"--help"}).options.help && Parse({"--help"}).Ok(), "--help is valid and sets help");
}

// A flag at the end of the line with no value is an error that names the flag. The same flag with
// a value is the baseline that shows the missing value is what fails it. A flag followed by another
// flag has no value either, so it never swallows that flag.
void TestOptionMissingValue() {
  for (const char *flag : {"--screenshot", "--width", "--height"}) {
    const ParseResult missing = Parse({flag});
    Check(!missing.Ok() && Mentions(missing.error, flag) && Mentions(missing.error, "needs a value"), "a flag with no value is an error naming it");
  }
  Check(Parse({"--screenshot", "dir"}).Ok() && Parse({"--width", "5"}).Ok() && Parse({"--height", "5"}).Ok(), "the same flags with a value are valid");
  // A value that starts with "--" is another flag, so the first flag is the one missing a value.
  const ParseResult swallowed = Parse({"--width", "--height", "600"});
  Check(!swallowed.Ok() && Mentions(swallowed.error, "--width") && Mentions(swallowed.error, "needs a value"), "a flag following --width means --width has no value");
  const ParseResult folder = Parse({"--screenshot", "--width", "800"});
  Check(!folder.Ok() && Mentions(folder.error, "--screenshot") && folder.options.screenshot_dir.empty() && folder.options.width == 1920, "--width is not taken as the screenshot folder");
}

// A size that isn't a whole number above zero is an error that names the flag and the text. Each
// case would pass a parser that only checked the first character or ignored overflow.
void TestOptionBadNumber() {
  for (const char *bad : {"abc", "12x", "x12", "0", "-5", "1.5", "", " 7", "+7", "99999999999"}) {
    for (const char *flag : {"--width", "--height"}) {
      const ParseResult result = Parse({flag, bad});
      Check(!result.Ok(), "a bad size is an error");
      // Mentions with an empty value is always true, so only the flag can be checked then.
      Check(Mentions(result.error, flag) && (std::string_view(bad).empty() || Mentions(result.error, bad)), "the error names the flag and the value");
      Check(result.options.width == 1920 && result.options.height == 1080, "a bad size leaves the defaults");
    }
  }
  Check(Parse({"--width", "1"}).Ok(), "the smallest valid size is 1");
}

// An unknown flag is an error that names it, and parsing stops there so the flags after it don't
// apply. --help is the baseline for a known flag, and a bare word is unknown too.
void TestOptionUnknownFlag() {
  const ParseResult unknown = Parse({"--bogus"});
  Check(!unknown.Ok() && Mentions(unknown.error, "--bogus"), "an unknown flag is an error naming it");
  const ParseResult stray = Parse({"folder"});
  Check(!stray.Ok() && Mentions(stray.error, "folder"), "a bare word is unknown");
  const ParseResult after = Parse({"--bogus", "--width", "800"});
  Check(!after.Ok() && after.options.width == 1920, "flags after an error are not applied");
  Check(Parse({"--help"}).Ok(), "a known flag is not an error");
  Check(!Parse({"--Width", "800"}).Ok(), "flags are case sensitive");
}

// A 3x2 image with a 16 byte row pitch, 4 bytes more than a row, whose padding holds 0xEE so a
// pitch that is ignored reads it as pixels. Every channel of every pixel is different, so a
// swapped, dropped or shifted channel changes the result.
constexpr int PATTERN_WIDTH = 3;
constexpr int PATTERN_HEIGHT = 2;
constexpr size_t PATTERN_PITCH = 16;

std::vector<uint8_t> Pattern() {
  std::vector<uint8_t> pixels(PATTERN_PITCH * PATTERN_HEIGHT, 0xEE);
  uint8_t next = 1;
  for (int row = 0; row < PATTERN_HEIGHT; row++) {
    for (int byte = 0; byte < PATTERN_WIDTH * 4; byte++) {
      pixels[row * PATTERN_PITCH + byte] = next++;
    }
  }
  return pixels;
}

// With RGBA input nothing is swapped and the padding is dropped. This is the baseline the BGRA
// case differs from only in the red and blue channels.
void TestToRgba() {
  const auto pixels = Pattern();
  const auto rgba = ToRgba(pixels, PATTERN_WIDTH, PATTERN_HEIGHT, PATTERN_PITCH, false);
  const std::vector<uint8_t> expected = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  Check(rgba == expected, "RGBA input is packed tightly with the row padding dropped and no swap");

  const auto swapped = ToRgba(pixels, PATTERN_WIDTH, PATTERN_HEIGHT, PATTERN_PITCH, true);
  Check(swapped.size() == expected.size(), "BGRA output is the same size");
  for (size_t index = 0; index < expected.size(); index++) {
    uint8_t want = expected[index];
    if (index % 4 == 0) {
      want = expected[index + 2];
    } else if (index % 4 == 2) {
      want = expected[index - 2];
    }
    Check(swapped[index] == want, "BGRA swaps red and blue and keeps green and alpha");
  }
  Check(swapped != rgba, "BGRA differs from RGBA");

  Check(ToRgba(std::span<const uint8_t>(pixels).first(PATTERN_PITCH * 2 - 5), PATTERN_WIDTH, PATTERN_HEIGHT, PATTERN_PITCH, false).empty(), "a short buffer gives nothing");
  Check(ToRgba(pixels, PATTERN_WIDTH, PATTERN_HEIGHT, 8, false).empty(), "a pitch under a row gives nothing");
  Check(ToRgba(std::span<const uint8_t>(pixels).first(PATTERN_PITCH + PATTERN_WIDTH * 4), PATTERN_WIDTH, PATTERN_HEIGHT, PATTERN_PITCH, false).size() == expected.size(),
        "the last row needs no padding after it");
}

// A fresh folder for a test's files.
std::filesystem::path TempFolder(const char *name) {
  const auto folder = std::filesystem::temp_directory_path() / name;
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder);
  return folder;
}

// A PNG written and read back by a different decoder has the same size and pixels. The pixels
// are the distinct pattern, with alpha not 255, so lost alpha or channels would show.
void TestPngRoundTrip() {
  const auto folder = TempFolder("system_ui_gallery_png");
  const auto rgba = ToRgba(Pattern(), PATTERN_WIDTH, PATTERN_HEIGHT, PATTERN_PITCH, false);
  const auto path = folder / "round.png";
  Check(WritePng(path, rgba, PATTERN_WIDTH, PATTERN_HEIGHT), "the PNG is written");
  int width = 0, height = 0, channels = 0;
  const std::string text = path.string();
  uint8_t *loaded = stbi_load(text.c_str(), &width, &height, &channels, 4);
  Check(loaded != nullptr, "the PNG loads");
  Check(width == PATTERN_WIDTH && height == PATTERN_HEIGHT, "the PNG has the size written");
  Check(std::equal(rgba.begin(), rgba.end(), loaded), "the PNG has the pixels written");
  stbi_image_free(loaded);
  Check(!WritePng(folder / "bad.png", std::span<const uint8_t>(rgba).first(rgba.size() - 1), PATTERN_WIDTH, PATTERN_HEIGHT), "data of the wrong size is refused");
  Check(!std::filesystem::exists(folder / "bad.png"), "and writes nothing");
  std::filesystem::remove_all(folder);
}

// Writing where the file can't go fails. The same call into an existing folder is the baseline.
void TestPngWriteFails() {
  const auto folder = TempFolder("system_ui_gallery_png_fail");
  const auto rgba = ToRgba(Pattern(), PATTERN_WIDTH, PATTERN_HEIGHT, PATTERN_PITCH, false);
  Check(WritePng(folder / "ok.png", rgba, PATTERN_WIDTH, PATTERN_HEIGHT), "writing into an existing folder works");
  Check(!WritePng(folder / "missing" / "x.png", rgba, PATTERN_WIDTH, PATTERN_HEIGHT), "a missing folder fails");
  // A folder where the file should be can't be opened as a file.
  std::filesystem::create_directories(folder / "dir.png");
  Check(!WritePng(folder / "dir.png", rgba, PATTERN_WIDTH, PATTERN_HEIGHT), "a path that is a folder fails");
  std::filesystem::remove_all(folder);
}

// Every page has one job, in order, with a unique lower case name that ends in .png and starts
// with its group, so two pages with the same id in different groups don't collide.
void TestScreenshotJobs() {
  const auto jobs = ScreenshotJobs(Pages());
  const auto pages = Pages();
  size_t expected = 0;
  for (const PageEntry &page : pages) {
    expected += std::max<size_t>(1, page.view_names.size());
  }
  Check(jobs.size() == expected && !jobs.empty() && expected > pages.size(), "there is a job for every view of every page");
  std::set<std::string> names;
  for (size_t index = 0; index < jobs.size(); index++) {
    Check(index == 0 || jobs[index].page == jobs[index - 1].page || jobs[index].page == jobs[index - 1].page + 1, "jobs are in page order");
    Check(index == 0 || jobs[index].page != jobs[index - 1].page || jobs[index].view == jobs[index - 1].view + 1, "and a page's in view order");
    const std::string &name = jobs[index].file_name;
    Check(name.size() > 4 && name.substr(name.size() - 4) == ".png", "names end in .png");
    Check(name == ScreenshotName(pages[static_cast<size_t>(jobs[index].page)], jobs[index].view), "a job is named for its page and view");
    for (const char letter : name) {
      Check(!(letter >= 'A' && letter <= 'Z'), "names are lower case");
    }
    names.insert(name);
  }
  Check(names.size() == jobs.size(), "names are unique");
  Check(names.count("styles-colour.png") == 1 && names.count("components-button.png") == 1, "names are group then id");
  Check(names.count("styles-colour-roles.png") == 1 && names.count("styles-colour-contrast.png") == 1, "a later view adds its name");
  const PageEntry same_id_elsewhere{PageGroup::Patterns, "button", "Button", nullptr};
  Check(ScreenshotName(same_id_elsewhere) == "patterns-button.png", "the group is part of the name");
}

// ---- Key ownership ----------------------------------------------------------------------------

constexpr Rect CARD{660, 340, 600, 400};
constexpr Rect FIRST{720, 600, 220, 56};
constexpr Rect SECOND{980, 600, 220, 56};
constexpr std::array<std::string_view, 3> TAB_LABELS{"One", "Two", "Three"};

// What the key pages recorded in their last frame, and what the test asks of them.
bool g_bar_shoulders = false;
int g_tab = 0;
bool g_modal_open = false;
bool g_modal_inside = false;
bool g_foreign_open = false;
bool g_foreign_focused = false;

// A page with a tab bar, whose shoulders the test turns on or off, and a modal the test opens. The
// bar is drawn before the modal, as a page behind a dialog would be.
void DrawKeyPage(UiFrame &frame, GalleryState &state, const PageEntry &page, const Rect &pane) {
  DrawItem(frame, state, page, pane);
  (void)SystemUiTest::DrawElementBar(frame, "tabs", {pane.x + 300, pane.y + 200, 700, 56}, TAB_LABELS, g_tab, g_bar_shoulders);
  ModalScope modal(frame, "dialog", CARD, g_modal_open);
  g_modal_inside = false;
  if (modal.Visible()) {
    const auto first = Interact(frame, ImGui::GetID("first"), FIRST, false);
    const auto second = Interact(frame, ImGui::GetID("second"), SECOND, false);
    g_modal_inside = first.state.focused || second.state.focused;
    if (modal.BackPressed()) {
      g_modal_open = false;
    }
  }
}

// A page with a dialog that is not ModalScope's, which claims no keys. Only the gallery's check of
// where focus is can tell it that back and left belong to the dialog.
void DrawForeignPage(UiFrame &frame, GalleryState &state, const PageEntry &page, const Rect &pane) {
  DrawItem(frame, state, page, pane);
  if (g_foreign_open && !ImGui::IsPopupOpen("foreign")) {
    ImGui::OpenPopup("foreign");
  }
  ImGui::SetNextWindowPos({CARD.x, CARD.y});
  ImGui::SetNextWindowSize({CARD.width, CARD.height});
  g_foreign_focused = false;
  if (ImGui::BeginPopupModal("foreign", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoSavedSettings)) {
    g_foreign_focused = Interact(frame, ImGui::GetID("fx"), FIRST, false).state.focused;
    ImGui::EndPopup();
  }
}

constexpr std::array KEY_PAGES{PageEntry{PageGroup::Styles, "a", "A", DrawItem}, PageEntry{PageGroup::Components, "k", "K", DrawKeyPage},
                               PageEntry{PageGroup::Patterns, "z", "Z", DrawItem}};
constexpr std::array FOREIGN_PAGES{PageEntry{PageGroup::Styles, "a", "A", DrawItem}, PageEntry{PageGroup::Components, "f", "F", DrawForeignPage},
                                   PageEntry{PageGroup::Patterns, "z", "Z", DrawItem}};

// Draws the gallery over `pages`.
template <size_t N> auto DrawKeyGallery(GalleryState &state, const std::array<PageEntry, N> &pages) {
  return [&state, &pages](UiFrame &frame) {
    g_item_focused_on = -1;
    DrawGallery(frame, state, pages, SCREEN);
  };
}

// Resets the key pages' controls.
void ResetKeyPages() {
  g_bar_shoulders = false;
  g_tab = 0;
  g_modal_open = false;
  g_modal_inside = false;
  g_foreign_open = false;
  g_foreign_focused = false;
}

// A tab bar with the shoulders owns L1 and R1 on its page, so R1 changes only the tab. The same bar
// without the shoulders is the baseline, where R1 reaches the gallery and changes the page.
void TestBarWithShouldersOwnsThem() {
  for (const bool shoulders : {true, false}) {
    ResetKeyPages();
    g_bar_shoulders = shoulders;
    GalleryState state;
    NavHarness nav(SystemUiNavigation());
    const auto draw = DrawKeyGallery(state, KEY_PAGES);
    nav.Frames(2, draw);
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
    Check(state.page == 1, "the page with the bar is shown");
    nav.Press(ImGuiKey_GamepadR1, draw);
    if (shoulders) {
      Check(g_tab == 1 && state.page == 1, "with the shoulders, R1 changes the tab and not the page");
    } else {
      Check(g_tab == 0 && state.page == 2, "without them, R1 changes the page and not the tab");
    }
  }
}

// Opens the key page's dialog with focus on the page's entry control, which the dialog then takes.
template <class Draw> void OpenDialog(NavHarness &nav, GalleryState &state, Draw &&draw) {
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(state.page == 1 && g_item_focused_on == 1, "right enters the page");
  g_modal_open = true;
  nav.Frames(3, draw);
  Check(g_modal_inside, "the dialog takes focus");
}

// An open modal owns the shoulders over both the gallery and a bar that has them, so R1 changes
// neither the page nor the tab. Back closes it and focus returns to the page's control and not to
// the sidebar row. With the modal closed, the baseline, back from the same control goes to the row.
void TestModalOwnsShouldersAndBack() {
  ResetKeyPages();
  g_bar_shoulders = true;
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawKeyGallery(state, KEY_PAGES);
  OpenDialog(nav, state, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(state.page == 1 && g_tab == 0, "R1 changes neither the page nor the tab under the modal");
  nav.Press(ImGuiKey_GamepadFaceRight, draw);
  Check(!g_modal_open, "back closes the modal");
  Check(g_item_focused_on == 1, "and focus returns to the page's control, not the sidebar");
  ResetKeyPages();
  GalleryState closed;
  NavHarness baseline(SystemUiNavigation());
  const auto closed_draw = DrawKeyGallery(closed, KEY_PAGES);
  baseline.Frames(2, closed_draw);
  baseline.Press(ImGuiKey_GamepadDpadDown, closed_draw);
  baseline.Press(ImGuiKey_GamepadDpadRight, closed_draw);
  baseline.Press(ImGuiKey_GamepadFaceRight, closed_draw);
  Check(g_item_focused_on == -1, "with the modal closed, back returns focus to the sidebar row");
}

// Left from the modal's leftmost button finds nothing, and does not pull focus out to the sidebar.
void TestLeftMissInModalStays() {
  ResetKeyPages();
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawKeyGallery(state, KEY_PAGES);
  OpenDialog(nav, state, draw);
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(g_modal_open && g_modal_inside, "left with nothing beside leaves focus in the modal");
}

// A dialog that claims no keys is kept from the gallery's back by where focus is alone. Back and
// left leave its focus where it is.
void TestForeignDialogKeepsFocus() {
  ResetKeyPages();
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawKeyGallery(state, FOREIGN_PAGES);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadDpadDown, draw);
  nav.Press(ImGuiKey_GamepadDpadRight, draw);
  g_foreign_open = true;
  nav.Frames(3, draw);
  Check(g_foreign_focused, "the dialog has focus");
  nav.Press(ImGuiKey_GamepadFaceRight, draw);
  Check(g_foreign_focused, "back does not move focus to the sidebar");
  nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(g_foreign_focused, "and neither does a left with nothing beside");
}

// R1 from the page before a tab bar brings the page in and does not also step the bar, which is
// drawn later in the same frame and finds the key unowned. A later R1 on that page moves only the
// tab, which is the baseline.
void TestShoulderIntoTabsPageDoesNotStepTab() {
  ResetKeyPages();
  g_bar_shoulders = true;
  GalleryState state;
  NavHarness nav(SystemUiNavigation());
  const auto draw = DrawKeyGallery(state, KEY_PAGES);
  nav.Frames(2, draw);
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(state.page == 1 && g_tab == 0, "R1 into the page with the bar changes the page and not the tab");
  nav.Press(ImGuiKey_GamepadR1, draw);
  Check(state.page == 1 && g_tab == 1, "a later R1 on that page moves only the tab");
}

// Walking the sidebar of the real pages, with the real fonts, at each output. Down along every row
// shows each page in turn and up shows them back to the first, so the rows are reachable and the
// page shown follows focus whatever the output's scale. The rows of a long sidebar scroll, so this
// also holds for the pages below the first screenful.
void TestSidebarWalksTheRealPages() {
  struct Output {
    float width;
    float height;
  };
  for (const Output &output : {Output{1280, 720}, Output{1920, 1080}, Output{3840, 2160}}) {
    GalleryState state;
    const auto pages = Pages();
    const Rect screen{0, 0, output.width, output.height};
    HarnessOptions options = SystemUiNavigation();
    options.width = output.width;
    options.height = output.height;
    options.load_fonts = [output](ImFontAtlas &atlas) { return LoadPrebakedFonts(atlas, output.width, output.height); };
    NavHarness nav(options);
    const auto draw = [&](UiFrame &frame) { DrawGallery(frame, state, pages, screen); };
    nav.Frames(3, draw);
    for (size_t page = 1; page < pages.size(); ++page) {
      nav.Press(ImGuiKey_GamepadDpadDown, draw);
      Check(state.page == static_cast<int>(page), "down shows the next page, in order, to the last row");
    }
    nav.Press(ImGuiKey_GamepadDpadDown, draw);
    Check(state.page == static_cast<int>(pages.size()) - 1, "down from the last row stays on it");
    for (size_t page = pages.size() - 1; page > 0; --page) {
      nav.Press(ImGuiKey_GamepadDpadUp, draw);
      Check(state.page == static_cast<int>(page) - 1, "and up shows them back to the first");
    }
  }
}

// The icon the gallery's Toast page shows, read from the repository.
std::vector<std::byte> ReadTrophyIcon() {
  std::ifstream stream(KYTY_TEST_TROPHY_ICON, std::ios::binary);
  const std::vector<char> bytes{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
  Check(!bytes.empty(), "the sample trophy icon can be read");
  const auto *data = reinterpret_cast<const std::byte *>(bytes.data());
  return {data, data + bytes.size()};
}

// The gallery on its Toast page, with the real fonts, navigation and an image store, as the program
// runs it. With no renderer, ImGui itself destroys a released image on the next frame. Field order
// matters, as the state releases into the store and the store unregisters from the harness's context.
struct ToastPageRig {
  explicit ToastPageRig(bool with_images = true) : nav(Options()) {
    if (with_images) {
      state.images = &images;
      state.sample_icon = icon;
    }
    for (size_t page = 0; page < Pages().size(); ++page) {
      if (Pages()[page].id == "toast") {
        toast_page = static_cast<int>(page);
      }
    }
    Check(toast_page >= 0, "the gallery has a Toast page");
    // The sidebar's first row takes focus as the window appears, which would undo an earlier request.
    Frames(2);
    state.show = toast_page;
    Frames(1);
    Check(state.page == toast_page, "the Toast page is showing");
  }

  static HarnessOptions Options() {
    HarnessOptions options = SystemUiNavigation();
    options.load_fonts = [](ImFontAtlas &atlas) { return LoadPrebakedFonts(atlas, SCREEN.width, SCREEN.height); };
    return options;
  }

  auto Draw() {
    return [this](UiFrame &frame) {
      DrawGallery(frame, state, Pages(), SCREEN);
      list = ImGui::GetWindowDrawList();
    };
  }
  void Frames(int count) { nav.Frames(count, Draw()); }

  NavHarness nav;
  ImageStore images{32};
  std::vector<std::byte> icon = ReadTrophyIcon();
  GalleryState state;
  int toast_page = -1;
  // The list the last frame drew into, which stays readable until the next frame starts.
  ImDrawList *list = nullptr;
};

// A box around points, empty until a point is added.
struct Extent {
  float left = std::numeric_limits<float>::infinity();
  float top = std::numeric_limits<float>::infinity();
  float right = -std::numeric_limits<float>::infinity();
  float bottom = -std::numeric_limits<float>::infinity();

  [[nodiscard]] bool Any() const { return left <= right; }
  void Add(ImVec2 point) {
    left = std::min(left, point.x);
    top = std::min(top, point.y);
    right = std::max(right, point.x);
    bottom = std::max(bottom, point.y);
  }
};

constexpr bool Inside(const Rect &rect, ImVec2 point) {
  return point.x >= rect.x && point.x <= rect.Right() && point.y >= rect.y && point.y <= rect.Bottom();
}

// The triangles of `list` whose centre lies in `area` and whose corners all pass `wanted`, as how far
// they were drawn, and how far they show once each command's clip rectangle is applied. A clipped
// glow and an unclipped one draw the same vertices, so only the second tells them apart.
struct TriangleExtents {
  Extent drawn;
  Extent visible;
  int triangles = 0;
};

template <class Wanted> TriangleExtents TrianglesIn(const ImDrawList &list, const Rect &area, Wanted wanted) {
  TriangleExtents result;
  for (const ImDrawCmd &command : list.CmdBuffer) {
    for (unsigned int at = 0; at + 2 < command.ElemCount; at += 3) {
      const ImDrawVert *corners[3];
      for (unsigned int corner = 0; corner < 3; ++corner) {
        const unsigned int index = list.IdxBuffer[static_cast<int>(command.IdxOffset + at + corner)];
        corners[corner] = &list.VtxBuffer[static_cast<int>(command.VtxOffset + index)];
      }
      if (!wanted(command, *corners[0]) || !wanted(command, *corners[1]) || !wanted(command, *corners[2])) {
        continue;
      }
      const ImVec2 centre{(corners[0]->pos.x + corners[1]->pos.x + corners[2]->pos.x) / 3.0f,
                          (corners[0]->pos.y + corners[1]->pos.y + corners[2]->pos.y) / 3.0f};
      if (!Inside(area, centre)) {
        continue;
      }
      ++result.triangles;
      for (const ImDrawVert *vertex : corners) {
        result.drawn.Add(vertex->pos);
      }
      const ImVec4 &clip = command.ClipRect;
      if (centre.x >= clip.x && centre.x <= clip.z && centre.y >= clip.y && centre.y <= clip.w) {
        result.visible.Add(centre);
      }
    }
  }
  return result;
}

// Whether a vertex has `colour`, at any alpha, since glows and fades vary only the alpha.
bool SameColour(const ImDrawVert &vertex, Color colour) {
  return (vertex.col & ~IM_COL32_A_MASK) == (ToImU32(colour) & ~IM_COL32_A_MASK);
}

// What the page draws in `colour` around `card`, with room for a glow that spills past it.
TriangleExtents ColourAround(const ImDrawList &list, const Rect &card, Color colour) {
  constexpr float ROOM = 30.0f;
  const Rect area{card.x - ROOM, card.y - ROOM, card.width + 2.0f * ROOM, card.height + 2.0f * ROOM};
  return TrianglesIn(list, area,
                     [colour](const ImDrawCmd &, const ImDrawVert &vertex) { return SameColour(vertex, colour); });
}

// The top of the live toast's card on `stage`, or nothing when no toast is there. Only a toast's
// panel has this colour, whatever the toast's opacity, and the held toasts are below the stage.
std::optional<float> LiveCardTop(const ImDrawList &list, const Rect &stage) {
  const Color panel = ToastStyle(DEFAULT_THEME, ToastDefaults(DEFAULT_THEME)).fill;
  const auto found = TrianglesIn(
      list, stage, [panel](const ImDrawCmd &, const ImDrawVert &vertex) { return SameColour(vertex, panel); });
  if (!found.drawn.Any()) {
    return std::nullopt;
  }
  return found.drawn.top;
}

// Whether two positions are the same to within half a pixel.
bool SamePlace(float left, float right) { return std::abs(left - right) <= 0.5f; }

// Whether the live toast's card is resting at `rest`.
bool RestsAt(std::optional<float> top, float rest) { return top.has_value() && SamePlace(*top, rest); }

// How far below the stage's top the live toast rests, once its entrance is over.
constexpr float LIVE_REST = 40.0f;
// Long enough for the entrance, short of the end of the hold.
constexpr int FRAMES_TO_REST = 60;

// The held toast's glow shows past its card on both backdrops. A clipped glow draws the same
// vertices, so what shows, after the clip, is what tells them apart.
void TestToastPageHeldGlowSpills() {
  ToastPageRig rig;
  rig.Frames(3);
  const Color platinum = TrophyGradeColor(DEFAULT_THEME, 1);
  for (const Rect &card : rig.state.toast.layout.cards) {
    constexpr float EDGE = 1.0f;
    Check(ColourAround(*rig.list, card, platinum).visible.left < card.x - EDGE, "the glow shows past the card");
  }
}

// The toasts take the chosen grade's colour, and lose the last one's, so a page stuck on one grade
// or drawing every grade fails.
void TestToastPageFollowsTheGrade() {
  ToastPageRig rig;
  rig.Frames(3);
  const Color platinum = TrophyGradeColor(DEFAULT_THEME, 1);
  const Color gold = TrophyGradeColor(DEFAULT_THEME, 2);
  const auto draws = [&rig](Color colour) {
    return ColourAround(*rig.list, rig.state.toast.layout.cards[0], colour).triangles > 0;
  };
  Check(draws(platinum) && !draws(gold), "the first tab's grade is shown");
  rig.state.toast.grade = 1;
  rig.Frames(1);
  Check(draws(gold) && !draws(platinum), "and another grade replaces it");
}

// Every held toast draws the shared icon inside its card, and the live toast has one of its
// own, which is how the real view loads an unlock's icon.
void TestToastPageShowsIcons() {
  ToastPageRig rig;
  rig.Frames(3);
  Check(rig.state.toast.icon.has_value(), "the sample icon decodes");
  const ImTextureData *shared = rig.images.Texture(*rig.state.toast.icon);
  for (const Rect &card : rig.state.toast.layout.cards) {
    const auto found = TrianglesIn(*rig.list, card, [shared](const ImDrawCmd &command, const ImDrawVert &) {
      return command.TexRef._TexData == shared;
    });
    Check(found.triangles > 0, "each card draws the icon");
  }
  Check(rig.images.LiveCount() == 2, "the shared icon and the live toast's, and no more");
}

// Replay restarts a toast that has come to rest, and the same mouse moves without the click leave
// it resting, so nothing else on the page can be what moved it.
void TestToastPageReplayRestartsTheLiveToast() {
  const auto top_after = [](bool click) {
    ToastPageRig rig;
    rig.Frames(FRAMES_TO_REST);
    const Rect stage = rig.state.toast.layout.stage;
    Check(RestsAt(LiveCardTop(*rig.list, stage), stage.y + LIVE_REST), "the live toast has come to rest");
    const Rect replay = rig.state.toast.layout.replay;
    rig.nav.MoveMouse(replay.x + replay.width / 2.0f, replay.y + replay.height / 2.0f, rig.Draw());
    rig.Frames(1);
    if (click) {
      rig.nav.MouseButton(true);
      rig.Frames(1);
      rig.nav.MouseButton(false);
    } else {
      rig.Frames(1);
    }
    rig.Frames(3);
    return LiveCardTop(*rig.list, stage).value_or(0.0f) - stage.y;
  };
  Check(SamePlace(top_after(false), LIVE_REST), "without the click the toast stays at rest");
  const float replayed = top_after(true);
  Check(replayed < LIVE_REST - 0.5f && replayed > 0.0f, "with it the toast is sliding in again");
}

// Focus from the sidebar lands on the tabs, moving there changes nothing, and moving along them
// changes the grade and restarts the live toast.
void TestToastPageTabsRestartTheLiveToast() {
  ToastPageRig rig;
  rig.Frames(FRAMES_TO_REST);
  const Rect stage = rig.state.toast.layout.stage;
  const auto draw = rig.Draw();
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.page == rig.toast_page && rig.state.toast.grade == 0, "entering the page keeps the grade");
  Check(FocusedControl() == rig.state.entry && rig.state.entry != 0, "and focus is on the chosen tab");
  Check(RestsAt(LiveCardTop(*rig.list, stage), stage.y + LIVE_REST), "and the toast stays at rest");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.toast.grade == 1, "moving along the tabs chooses the next grade");
  const auto top = LiveCardTop(*rig.list, stage);
  Check(top.has_value() && *top < stage.y + LIVE_REST - 0.5f, "and the toast slides in again");
  const auto draws = [&rig, stage](Color colour) {
    return TrianglesIn(*rig.list, stage, [colour](const ImDrawCmd &, const ImDrawVert &vertex) {
             return SameColour(vertex, colour);
           }).triangles > 0;
  };
  Check(draws(TrophyGradeColor(DEFAULT_THEME, 2)), "in the chosen grade's colour");
  Check(!draws(TrophyGradeColor(DEFAULT_THEME, 1)), "and not the first grade's");
}

// The toast leaves after its four seconds, the stage stays empty for the gap, and it plays again.
void TestToastPageLoops() {
  ToastPageRig rig;
  rig.Frames(FRAMES_TO_REST);
  const Rect stage = rig.state.toast.layout.stage;
  rig.Frames(200); // past the four seconds, which are 240 frames
  Check(!LiveCardTop(*rig.list, stage).has_value(), "the toast has left");
  rig.Frames(60); // 320 frames in, still inside the gap of 90 frames after the toast left
  Check(!LiveCardTop(*rig.list, stage).has_value(), "the stage stays empty during the gap");
  rig.Frames(30);
  Check(LiveCardTop(*rig.list, stage).has_value(), "and the toast plays again");
}

// A toast the page skipped starts over when the page returns, shine included. The shine is timed
// from when the toast was shown, so a toast only redrawn would replay its entrance with no shine.
// The toast's own later frames, past the shine, are the baseline: same toast, same place.
void TestToastPageRestartsWhenShownAgain() {
  ToastPageRig rig;
  rig.Frames(180); // the shine, which ends about 1.1 seconds in, is long over
  const Rect stage = rig.state.toast.layout.stage;
  rig.state.show = 0;
  rig.Frames(5);
  rig.state.show = rig.toast_page;
  const auto triangles = [&rig, stage] {
    return TrianglesIn(*rig.list, stage, [](const ImDrawCmd &, const ImDrawVert &) { return true; }).triangles;
  };
  rig.Frames(45); // 750 ms in, the shine is under way
  const int shining = triangles();
  rig.Frames(60); // 1750 ms in, it is over
  Check(shining > triangles(), "the returning toast shines again");
}

// The live toast shows its glow past the card, on the stage alone so no held toast can satisfy it.
// The card rests a cell's padding in from the stage's left edge.
void TestToastPageLiveShowsTheChosenGlow() {
  ToastPageRig rig;
  rig.Frames(FRAMES_TO_REST);
  const Rect stage = rig.state.toast.layout.stage;
  const Color platinum = TrophyGradeColor(DEFAULT_THEME, 1);
  const auto glow = TrianglesIn(*rig.list, stage, [platinum](const ImDrawCmd &, const ImDrawVert &vertex) {
    return SameColour(vertex, platinum);
  });
  constexpr float CARD_INSET = 40.0f;
  Check(glow.visible.left < stage.x + CARD_INSET - 1.0f, "the live glow shows past the card");
}

// Replaying far more often than the store has room for never fills it, because the gallery collects
// what it released. A store nothing collects would refuse the icon, and the toast would lose it.
void TestToastPageReplaysDoNotFillTheStore() {
  ToastPageRig rig;
  rig.Frames(FRAMES_TO_REST);
  const Rect replay = rig.state.toast.layout.replay;
  rig.nav.MoveMouse(replay.x + replay.width / 2.0f, replay.y + replay.height / 2.0f, rig.Draw());
  rig.Frames(1);
  constexpr int MORE_THAN_THE_STORE_HOLDS = 40;
  for (int replayed = 0; replayed < MORE_THAN_THE_STORE_HOLDS; ++replayed) {
    rig.nav.MouseButton(true);
    rig.Frames(1);
    rig.nav.MouseButton(false);
    rig.Frames(1);
  }
  Check(rig.images.LiveCount() == 2, "every replay still gets its icon");
  Check(rig.images.RetiringCount() <= 2, "and what they released has been collected");
}

// A bad icon is tried once and not again every frame.
void TestToastPageTriesABadIconOnce() {
  ToastPageRig rig(false);
  const std::array<std::byte, 4> not_an_image{};
  rig.state.images = &rig.images;
  rig.state.sample_icon = not_an_image;
  rig.Frames(3);
  Check(!rig.state.toast.icon.has_value() && rig.state.toast.icon_tried, "the bad icon was tried and refused");
  rig.state.sample_icon = rig.icon;
  rig.Frames(1);
  Check(!rig.state.toast.icon.has_value(), "and not tried again, even once a good one would load");
  rig.state.sample_icon = {};
}

// Without an image store the page still draws its held toasts, without an icon, and skips only the live
// toast, as the tests that walk every page run it.
void TestToastPageWithoutImages() {
  ToastPageRig rig(false);
  rig.Frames(3);
  const auto &toast = rig.state.toast;
  Check(!toast.live.has_value() && !LiveCardTop(*rig.list, toast.layout.stage).has_value(), "no live toast");
  Check(ColourAround(*rig.list, toast.layout.cards[0], TrophyGradeColor(DEFAULT_THEME, 1)).triangles > 0,
        "the held toasts still draw");
  Check(rig.state.entry != 0, "and focus can still enter the page");
}

// Reset gives the toast page a fresh start for the next screenshot, so a picture doesn't depend on
// the pages before it, without loading the shared icon again.
void TestToastPageReset() {
  ToastPageRig rig;
  rig.state.toast.grade = 2;
  rig.Frames(3);
  Check(rig.images.LiveCount() == 2, "the shared icon and the live toast's");
  rig.state.Reset();
  Check(rig.state.toast.grade == 0 && !rig.state.toast.live.has_value(), "the grade and the live toast go");
  Check(rig.images.LiveCount() == 1, "which leaves only the shared icon");
  rig.state.show = rig.toast_page;
  rig.Frames(3);
  Check(rig.images.LiveCount() == 2, "and the shared icon is not loaded again");
}

// ---- Colour and Typography ----------------------------------------------------------------------

// The gallery on one of its real pages, with the real fonts and navigation.
struct StylePageRig {
  explicit StylePageRig(std::string_view id) : nav(ToastPageRig::Options()) {
    for (size_t index = 0; index < Pages().size(); ++index) {
      if (Pages()[index].id == id) {
        page = static_cast<int>(index);
      }
    }
    Check(page >= 0, "the gallery has the page");
    Frames(2);
    state.show = page;
    Frames(1);
    Check(state.page == page, "the page is showing");
  }
  auto Draw() {
    return [this](UiFrame &frame) {
      DrawGallery(frame, state, Pages(), SCREEN);
      list = ImGui::GetWindowDrawList();
    };
  }
  void Frames(int count) { nav.Frames(count, Draw()); }

  NavHarness nav;
  GalleryState state;
  int page = -1;
  ImDrawList *list = nullptr;
};

// The sidebar is this wide, and what a page draws lies to the right of its edge.
constexpr float PANE_LEFT = 320.0f;

// Whether the page draws a chip of `colour`, a rectangle of the swatches' size, which text in the
// same colour isn't, nor is the half of a chip's strip. A rectangle's third vertex is its bottom right
// corner.
bool HasChip(const ImDrawList &list, Color colour) {
  constexpr float CHIP_WIDTH = 112.0f;
  constexpr float CHIP_HEIGHT = 64.0f;
  const auto &vertices = list.VtxBuffer;
  for (int index = 0; index + 2 < vertices.Size; ++index) {
    const bool is_colour = vertices[index].pos.x > PANE_LEFT && vertices[index].col == ToImU32(colour);
    if (is_colour && std::fabs(vertices[index + 2].pos.y - vertices[index].pos.y - CHIP_HEIGHT) <= 0.5f &&
        std::fabs(vertices[index + 2].pos.x - vertices[index].pos.x - CHIP_WIDTH) <= 0.5f) {
      return true;
    }
  }
  return false;
}

// How many vertices the page, leaving out the sidebar, draws in exactly `colour`.
int PageVertices(const ImDrawList &list, Color colour) {
  int count = 0;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    count += vertex.pos.x > PANE_LEFT && vertex.col == ToImU32(colour) ? 1 : 0;
  }
  return count;
}

// The ratio is the WCAG one, checked against black and white, where it is exactly 21, and a grey that
// a calculator gives as 4.478. Order and alpha don't matter, and a colour against itself is 1. Over a
// surface, an opaque colour hides what is under it, a clear one shows it, and half shows both.
void TestContrastMaths() {
  constexpr float SLACK = 0.001f;
  const Color white = Rgb(0xFFFFFF);
  const Color black = Rgb(0x000000);
  const Color grey = Rgb(0x777777);
  Check(std::fabs(ContrastRatio(white, black) - 21.0f) <= SLACK, "white on black is 21 to 1");
  Check(std::fabs(ContrastRatio(grey, white) - 4.478f) <= SLACK, "the grey is 4.478 to 1 on white");
  Check(ContrastRatio(white, grey) == ContrastRatio(grey, white), "in either order");
  Check(ContrastRatio(grey.WithAlpha(0.2f), white) == ContrastRatio(grey, white), "whatever the alpha");
  Check(ContrastRatio(grey, grey) == 1.0f, "a colour on itself is 1 to 1");
  Check(Composite(grey, white) == grey, "an opaque colour hides what is under it");
  Check(Composite(grey.WithAlpha(0.0f), white) == white, "a clear one shows it");
  Check(Composite(Color{255, 255, 255, 128}, black) == Rgb(0x808080), "and half shows both");
  Check(Composite(Color{255, 255, 255, 128}, black).a == 255, "and what results is opaque");
}

// The table's figures, against ones worked out apart from this code. A
// surface is translucent, so a pair differs over black and white. The row with a tint is judged on
// the surface under it, which the same row without that surface doesn't reach. A pair that is fine
// over black and not over white fails, as the worse of the two decides.
void TestContrastRowsAgainstWorkedFigures() {
  constexpr float SLACK = 0.01f;
  const Color black = Rgb(0x000000);
  const Color white = Rgb(0xFFFFFF);
  const auto rows = ContrastRows(DEFAULT_THEME);
  const auto find = [&rows](std::string_view name) {
    const auto found = std::find_if(rows.begin(), rows.end(), [name](const ContrastRow &row) { return row.name == name; });
    Check(found != rows.end(), "the table has the row");
    return *found;
  };
  const ContrastRow primary = find("Primary text on raised");
  Check(std::fabs(ContrastOver(primary, black) - 17.759f) <= SLACK && std::fabs(ContrastOver(primary, white) - 13.971f) <= SLACK, "primary on raised is as worked out");
  const ContrastRow tertiary = find("Tertiary text on raised");
  Check(std::fabs(ContrastOver(tertiary, white) - 4.527f) <= SLACK && ContrastPasses(tertiary), "tertiary on raised only just passes, over white");
  const ContrastRow hud = find("Secondary text on HUD");
  Check(std::fabs(ContrastOver(hud, black) - 9.441f) <= SLACK && std::fabs(ContrastOver(hud, white) - 3.715f) <= SLACK, "secondary on the HUD is as worked out");
  Check(!ContrastPasses(hud), "and fails, as the worse of the two is under its target");
  const ContrastRow tinted = find("Error text on its tint, on raised");
  Check(std::fabs(ContrastOver(tinted, black) - 5.867f) <= SLACK && std::fabs(ContrastOver(tinted, white) - 4.627f) <= SLACK, "a tint is judged on the surface under it");
  ContrastRow bare = tinted;
  bare.under = {};
  Check(std::fabs(ContrastOver(bare, white) - ContrastOver(tinted, white)) > SLACK, "which is why that surface is given");
  ContrastRow lower = hud;
  lower.target = 3.0f;
  Check(ContrastPasses(lower), "the target decides, as the same pair passes a lower one");
  Check(find("Button label on accent, large").target == CONTRAST_LARGE && primary.target == CONTRAST_TEXT, "large text has the lower target");
  // Error and info text are lighter than their tints, so that they pass on those tints as well, and
  // the outline lets a track reach the target its fill can't.
  const auto passes_as_worked = [&](const char *name, float over_black, float over_white) {
    const ContrastRow row = find(name);
    Check(std::fabs(ContrastOver(row, black) - over_black) <= SLACK && std::fabs(ContrastOver(row, white) - over_white) <= SLACK && ContrastPasses(row), name);
  };
  passes_as_worked("Error text on raised", 7.001f, 5.508f);
  passes_as_worked("Error text on its tint, on raised", 5.867f, 4.627f);
  passes_as_worked("Info text on raised", 8.815f, 6.935f);
  passes_as_worked("Info text on its tint, on raised", 6.764f, 5.262f);
  passes_as_worked("Toggle's off outline on raised, not text", 5.395f, 4.244f);
  passes_as_worked("Slider track's outline on raised, not text", 3.639f, 3.200f);
}

// Every colour of the palette is on the Palette view, written out here apart from the page's table.
// The Contrast view is the baseline, which draws no chips.
void TestColourPalette() {
  namespace P = Libs::Graphics::SystemUi::Palette;
  constexpr std::array WANTED{P::GRAY_0, P::GRAY_50, P::GRAY_100, P::GRAY_150, P::GRAY_200, P::GRAY_300, P::GRAY_400, P::GRAY_500, P::GRAY_600,
                              P::GRAY_700, P::GRAY_800, P::GRAY_900, P::BLUE_300, P::BLUE_400, P::BLUE_500, P::BLUE_550, P::BLUE_600, P::BLUE_700,
                              P::GREEN_500, P::YELLOW_500, P::RED_400, P::RED_500, P::RED_600, P::PLATINUM, P::GOLD, P::SILVER, P::BRONZE};
  StylePageRig rig("colour");
  rig.Frames(1);
  for (const Color &colour : WANTED) {
    Check(HasChip(*rig.list, colour), "every colour of the palette has its chip");
  }
  rig.state.colour_view = 2;
  rig.Frames(1);
  Check(std::none_of(WANTED.begin(), WANTED.end(), [&](const Color &colour) { return HasChip(*rig.list, colour); }), "where the table has none");
}

// The translucent tokens are drawn on the Roles view as they are, alpha included, so the strip under
// them shows. The Palette view is the baseline, as it draws none of them.
void TestColourRoles() {
  const auto &colors = DEFAULT_THEME.colors;
  const std::array WANTED{colors.surface.base, colors.surface.raised, colors.surface.overlay, colors.surface.hud, colors.surface.hud_strong, colors.surface.inset, colors.surface.scrim,
                          colors.accent.tint, colors.status.success_tint, colors.status.warning_tint, colors.status.error_tint, colors.status.info_tint, colors.border.strong,
                          colors.state.hover, colors.state.focus, colors.state.pressed, colors.state.hover_on_light, colors.state.focus_on_light, colors.focus.keyline};
  StylePageRig rig("colour");
  rig.state.colour_view = 1;
  rig.Frames(1);
  for (const Color &colour : WANTED) {
    Check(PageVertices(*rig.list, colour) > 0, "every translucent token is drawn as it is");
  }
  Check(PageVertices(*rig.list, Rgb(0x000000)) > 0 && PageVertices(*rig.list, Rgb(0xFFFFFF)) > 0, "over a dark and a white strip");
  rig.state.colour_view = 0;
  rig.Frames(1);
  for (const Color &colour : WANTED) {
    Check(PageVertices(*rig.list, colour) == 0, "the palette draws none of them");
  }
}

// The table says Pass for each row that passes, a word of four glyphs of four vertices, so the
// vertices in that colour count the rows. A row that fails is tinted, one quad of four vertices, and
// the same colour marks the words and numbers that fell short, so there are more than its word.
void TestContrastTableReportsEveryRow() {
  constexpr int VERTICES_PER_WORD = 16;
  const auto rows = ContrastRows(DEFAULT_THEME);
  const auto passes = std::count_if(rows.begin(), rows.end(), ContrastPasses);
  const auto fails = static_cast<std::ptrdiff_t>(rows.size()) - passes;
  Check(passes > 0 && fails > 0, "the theme has rows of both");
  StylePageRig rig("colour");
  rig.state.colour_view = 2;
  rig.Frames(1);
  Check(PageVertices(*rig.list, DEFAULT_THEME.colors.status.success) == passes * VERTICES_PER_WORD, "each row that passes says so");
  constexpr int VERTICES_PER_QUAD = 4;
  Check(PageVertices(*rig.list, DEFAULT_THEME.colors.status.error_tint) == fails * VERTICES_PER_QUAD, "each that fails is tinted");
  Check(PageVertices(*rig.list, DEFAULT_THEME.colors.status.error) > fails * VERTICES_PER_WORD, "and says so, with the number that fell short marked");
  // The two words have as many glyphs, so it is the glyphs that say which is which, compared with the
  // words drawn by themselves. Each word ends its row, so the last vertices in a colour are its word.
  const auto word_uvs = [](const ImDrawList &list, Color colour) {
    std::vector<std::pair<float, float>> uvs;
    for (const ImDrawVert &vertex : list.VtxBuffer) {
      if (vertex.pos.x > PANE_LEFT && vertex.col == ToImU32(colour)) {
        uvs.push_back({vertex.uv.x, vertex.uv.y});
      }
    }
    return std::vector<std::pair<float, float>>(uvs.end() - VERTICES_PER_WORD, uvs.end());
  };
  const auto &status = DEFAULT_THEME.colors.status;
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    const auto expected = [&](std::string_view word, Color colour) {
      const TextProps props{word, DEFAULT_THEME.type.body_small, colour};
      ImDrawList alone(ImGui::GetDrawListSharedData());
      alone._ResetForNewFrame();
      alone.PushClipRectFullScreen();
      DrawText(alone, LayoutText(frame.Ui(), props, std::numeric_limits<float>::infinity()), props, {PANE_LEFT + 1.0f, 0.0f, 100.0f, 30.0f});
      return word_uvs(alone, colour);
    };
    Check(word_uvs(*rig.list, status.success) == expected("Pass", status.success), "a row that passes says Pass");
    Check(word_uvs(*rig.list, status.error) == expected("Fail", status.error), "and one that fails says Fail");
  });
}

// Every view of the style pages fits the page at 1080p, as only the List page may scroll.
void TestStylePagesFit() {
  for (const std::string_view id : {"colour", "typography", "spacing", "images", "effects"}) {
    StylePageRig rig(id);
    const PageEntry &page = Pages()[static_cast<size_t>(rig.page)];
    Check(!page.view_names.empty(), "the page has views");
    for (int view = 0; view < static_cast<int>(page.view_names.size()); ++view) {
      rig.state.*page.view = view;
      rig.state.typography_line_boxes = true;
      rig.Frames(1);
      float bottom = 0.0f;
      float right = 0.0f;
      for (const ImDrawVert &vertex : rig.list->VtxBuffer) {
        if (vertex.pos.x > PANE_LEFT) {
          bottom = std::max(bottom, vertex.pos.y);
          right = std::max(right, vertex.pos.x);
        }
      }
      // The images page's Corners view is short, but a third of the way down is still past the page's
      // head, so its body has been drawn.
      const float floor = id == "images" ? SCREEN.height / 3.0f : SCREEN.height / 2.0f;
      Check(bottom > floor && bottom <= SCREEN.Bottom() && right <= SCREEN.Right(), "the view is drawn and fits");
    }
  }
}

// Up and down stay in the sidebar whatever lies near it, on the arrow keys as on the D-pad, and a
// page's tabs lie nearer the first row than the next. A step shows the focus ring even with the mouse
// in the window, which only a move the gamepad or keys made does.
void TestSidebarStepsOnArrowsAndShowsTheRing() {
  StylePageRig rig("colour");
  ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
  auto draw = rig.Draw();
  rig.Frames(1);
  const int first = rig.page;
  rig.nav.Press(ImGuiKey_DownArrow, draw);
  Check(rig.state.page == first + 1, "the down arrow shows the next page");
  rig.nav.Press(ImGuiKey_UpArrow, draw);
  Check(rig.state.page == first, "and the up arrow the one before");
  // Twice, as ImGui goes by how far the mouse moved, which the first position gives it nothing of.
  rig.nav.MoveMouse(1000.0f, 900.0f, draw);
  rig.nav.MoveMouse(1010.0f, 900.0f, draw);
  Check(CurrentInputMode() == InputMode::Pointer, "the mouse moving makes it the pointer's turn");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(rig.state.page == first + 1, "and a step still moves");
  Check(CurrentInputMode() == InputMode::Pad, "and brings the ring back");
}

// Focus enters on the tabs, and moving along them chooses the view, which draws on that frame. A
// reset puts every page's choice back, and the toggle's too.
void TestViewTabsChooseAndReset() {
  StylePageRig rig("colour");
  auto draw = rig.Draw();
  rig.Frames(1);
  Check(rig.state.entry != 0, "the page gives focus somewhere to enter");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.colour_view == 0, "entering stays on the chosen view");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.colour_view == 1, "and moving along the tabs chooses the next");
  Check(PageVertices(*rig.list, DEFAULT_THEME.colors.surface.scrim) > 0, "which draws");
  rig.state.colour_view = 2;
  rig.state.typography_view = 1;
  rig.state.typography_line_boxes = true;
  rig.state.Reset();
  Check(rig.state.colour_view == 0 && rig.state.typography_view == 0 && !rig.state.typography_line_boxes, "a reset puts them back");
}

// Down from the page's last control stays there, though rows of the sidebar lie below it. A click on a
// row still leaves the page, and from the row, down moves along the rows again.
void TestMovesDownStayInThePage() {
  constexpr float ROW_X = 160.0f;
  constexpr float IMAGES_ROW_Y = 274.0f;
  StylePageRig rig("typography");
  auto draw = rig.Draw();
  const int page = rig.page;
  rig.Frames(1);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(FocusedControl() == rig.state.entry, "right enters the page on its tabs");
  rig.Frames(3);
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  const ItemId toggle = FocusedControl();
  Check(toggle != 0 && toggle != rig.state.entry && rig.state.page == page, "down reaches the toggle under the tabs");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(FocusedControl() == toggle && rig.state.page == page, "and from there stays on it, rather than go to a row below");
  rig.nav.MoveMouse(ROW_X, IMAGES_ROW_Y, draw);
  rig.nav.MoveMouse(ROW_X + 1.0f, IMAGES_ROW_Y, draw);
  rig.nav.MouseButton(true);
  rig.nav.Frame(draw);
  rig.nav.MouseButton(false);
  rig.Frames(2);
  const int images = page + 2;
  Check(rig.state.page == images, "a click on a row still shows its page");
  Check(ImGui::GetCurrentContext()->NavIdIsAlive, "and the focused row stays known to navigation, which would otherwise lose where focus is");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(rig.state.page == images + 1, "and down moves along the rows from there");
}

// Left from a real page's first tab, with nothing further left, returns to the page's row, so down
// shows the page after. The page stays shown throughout.
void TestLeftLeavesARealPage() {
  StylePageRig rig("typography");
  auto draw = rig.Draw();
  const int page = rig.page;
  rig.Frames(1);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(FocusedControl() == rig.state.entry, "right enters the page");
  rig.nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  rig.Frames(1);
  Check(FocusedControl() != rig.state.entry && rig.state.page == page, "left leaves the page, which stays shown");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(rig.state.page == page + 1, "focus was on the page's row");
}

// ---- Spacing and layout -------------------------------------------------------------------------

// The first plain rectangle of `colour` on the page that is this big, if there is one. A rectangle's
// third vertex is its bottom right corner.
std::optional<Rect> FindQuad(const ImDrawList &list, Color colour, float width, float height) {
  constexpr float SLACK = 0.5f;
  const auto &vertices = list.VtxBuffer;
  for (int index = 0; index + 2 < vertices.Size; ++index) {
    const ImVec2 first = vertices[index].pos;
    const ImVec2 third = vertices[index + 2].pos;
    if (first.x > PANE_LEFT && vertices[index].col == ToImU32(colour) && vertices[index + 2].col == vertices[index].col &&
        std::fabs(third.x - first.x - width) <= SLACK && std::fabs(third.y - first.y - height) <= SLACK) {
      return Rect{first.x, first.y, third.x - first.x, third.y - first.y};
    }
  }
  return std::nullopt;
}

// The texture coordinates of what the page draws in `colour`, which say which glyphs they are.
std::vector<std::pair<float, float>> GlyphUvs(const ImDrawList &list, Color colour) {
  std::vector<std::pair<float, float>> uvs;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if (vertex.pos.x > PANE_LEFT && vertex.col == ToImU32(colour)) {
      uvs.push_back({vertex.uv.x, vertex.uv.y});
    }
  }
  return uvs;
}

// Where the page draws `text` among what it draws in `colour`, compared glyph by glyph with the text
// drawn by itself, or nothing when it doesn't.
std::optional<std::ptrdiff_t> TextAt(const ImDrawList &page, UiFrame &frame, std::string_view text, const TypeStyle &style, Color colour) {
  const TextProps props{text, style, colour};
  ImDrawList alone(ImGui::GetDrawListSharedData());
  alone._ResetForNewFrame();
  alone.PushClipRectFullScreen();
  const float unlimited = std::numeric_limits<float>::infinity();
  DrawText(alone, LayoutText(frame.Ui(), props, unlimited), props, {PANE_LEFT + 1.0f, 0.0f, SCREEN.width, SCREEN.height});
  const auto wanted = GlyphUvs(alone, colour);
  const auto drawn = GlyphUvs(page, colour);
  const auto found = std::search(drawn.begin(), drawn.end(), wanted.begin(), wanted.end());
  if (wanted.empty() || found == drawn.end()) {
    return std::nullopt;
  }
  return found - drawn.begin();
}

// Whether the page draws `text` at all.
bool DrawsText(const ImDrawList &page, UiFrame &frame, std::string_view text, const TypeStyle &style, Color colour) {
  return TextAt(page, frame, text, style, colour).has_value();
}

// Each length of the space scale has a bar of just its length and a label giving its name and length,
// written out here apart from the page's table. The Frame view is the baseline, which draws no bars.
void TestSpacingBars() {
  constexpr float BAR_HEIGHT = 24.0f;
  constexpr std::array WANTED{2.0f, 4.0f, 8.0f, 12.0f, 16.0f, 24.0f, 32.0f, 48.0f, 64.0f, 96.0f};
  const Color bar = DEFAULT_THEME.colors.accent.line;
  StylePageRig rig("spacing");
  rig.Frames(1);
  for (const float length : WANTED) {
    Check(FindQuad(*rig.list, bar, length, BAR_HEIGHT).has_value(), "each length has its bar");
  }
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    for (const char *label : {"hair  2", "s1  4", "s2  8", "s3  12", "s4  16", "s5  24", "s6  32", "s7  48", "s8  64", "s9  96", "border.hair  1", "border.thin  2", "border.thick  3", "border.bar  4"}) {
      Check(DrawsText(*rig.list, frame, label, DEFAULT_THEME.type.caption, DEFAULT_THEME.colors.text.secondary), "each length's label gives its name and length");
    }
  });
  rig.state.spacing_view = 1;
  rig.Frames(1);
  for (const float length : WANTED) {
    Check(!FindQuad(*rig.list, bar, length, BAR_HEIGHT).has_value(), "the frame view has none");
  }
}

// Each radius tile starts its top edge as far in as its radius. Full is half its side, less the one
// pixel ImGui's rounded path takes off. The tiles lie in two rows of three, in the scale's order, with
// these written out here apart from the theme.
void TestSpacingRadii() {
  constexpr float TILE = 96.0f;
  constexpr float PITCH = TILE + 24.0f;
  constexpr int PER_ROW = 3;
  constexpr std::array WANTED{4.0f, 8.0f, 12.0f, 16.0f, 24.0f, 47.0f};
  StylePageRig rig("spacing");
  rig.Frames(1);
  std::vector<ImVec2> points;
  for (const ImDrawVert &vertex : rig.list->VtxBuffer) {
    if (vertex.pos.x > PANE_LEFT && vertex.col == ToImU32(DEFAULT_THEME.colors.accent.tint)) {
      points.push_back(vertex.pos);
    }
  }
  Check(!points.empty(), "the tiles are drawn");
  float left = SCREEN.Right(), top = SCREEN.Bottom();
  for (const ImVec2 &point : points) {
    left = std::min(left, point.x);
    top = std::min(top, point.y);
  }
  for (size_t tile = 0; tile < WANTED.size(); ++tile) {
    const int column = static_cast<int>(tile) % PER_ROW;
    const bool second_row = static_cast<int>(tile) >= PER_ROW;
    const float tile_left = left + PITCH * static_cast<float>(column);
    float tile_top = SCREEN.Bottom(), top_start = SCREEN.Right();
    for (const ImVec2 &point : points) {
      const bool in_tile = point.x >= tile_left - 0.5f && point.x <= tile_left + TILE + 0.5f && (point.y > top + TILE + 0.5f) == second_row;
      tile_top = in_tile ? std::min(tile_top, point.y) : tile_top;
    }
    for (const ImVec2 &point : points) {
      const bool in_tile = point.x >= tile_left - 0.5f && point.x <= tile_left + TILE + 0.5f && (point.y > top + TILE + 0.5f) == second_row;
      top_start = in_tile && point.y <= tile_top + 0.25f ? std::min(top_start, point.x) : top_start;
    }
    Check(std::fabs(top_start - tile_left - WANTED[tile]) <= 0.25f, "each tile is rounded by its radius");
  }
}

// Each border box's stroke reaches in from its sides by the border's width, written out here apart from
// the theme. A stroke has vertices only at its corners, so none lies further in than that. The boxes
// lie one under another, so a vertex past a box's height begins the next.
void TestSpacingBorders() {
  constexpr float BOX_HEIGHT = 64.0f;
  constexpr float BOX_WIDTH = 160.0f;
  constexpr std::array WANTED{1.0f, 2.0f, 3.0f, 4.0f};
  StylePageRig rig("spacing");
  rig.Frames(1);
  const ImU32 stroke = ToImU32(DEFAULT_THEME.colors.accent.fill);
  std::vector<std::vector<ImVec2>> boxes;
  for (const ImDrawVert &vertex : rig.list->VtxBuffer) {
    if (vertex.pos.x <= PANE_LEFT || vertex.col != stroke) {
      continue;
    }
    if (boxes.empty() || vertex.pos.y > boxes.back().front().y + BOX_HEIGHT + 0.5f) {
      boxes.emplace_back();
    }
    boxes.back().push_back(vertex.pos);
  }
  Check(boxes.size() == WANTED.size(), "each width has its box");
  for (size_t box = 0; box < WANTED.size(); ++box) {
    float left = SCREEN.Right();
    for (const ImVec2 &point : boxes[box]) {
      left = std::min(left, point.x);
    }
    float width = 0.0f;
    for (const ImVec2 &point : boxes[box]) {
      width = std::max(width, std::min(point.x - left, left + BOX_WIDTH - point.x));
    }
    Check(std::fabs(width - WANTED[box]) <= 0.25f, "and its stroke is that wide");
  }
}

// The span of what the page draws in `colour` above `bottom`, leaving out the legend under the mock.
struct PageSpan {
  float left = SCREEN.Right();
  float top = SCREEN.Bottom();
  float right = 0.0f;
  float bottom = 0.0f;
};
PageSpan SpanAbove(const ImDrawList &list, Color colour, float limit) {
  PageSpan span;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if (vertex.pos.x > PANE_LEFT && vertex.pos.y <= limit && vertex.col == ToImU32(colour)) {
      span.left = std::min(span.left, vertex.pos.x);
      span.top = std::min(span.top, vertex.pos.y);
      span.right = std::max(span.right, vertex.pos.x);
      span.bottom = std::max(span.bottom, vertex.pos.y);
    }
  }
  return span;
}

// The legend's figures for `rect`, as the test works them out from the real scale.
std::string FiguresOf(const Rect &rect) {
  std::array<char, 80> text{};
  (void)std::snprintf(text.data(), text.size(), "%.0f x %.0f, %.0f from the left and %.0f from the top", static_cast<double>(rect.width), static_cast<double>(rect.height),
                      static_cast<double>(rect.x), static_cast<double>(rect.y));
  return text.data();
}

// Checks the legend gives each area's figures, in the order of the names beside them, as each line's
// name is fixed, so the order ties the figures to their names. Two areas can have the same figures, when
// the search finds the first for both. The notes say what 0% means, and when the two frames look alike.
void CheckFrameLegend(const ImDrawList &list, UiFrame &frame, const ScaleContext &real) {
  const auto &caption = DEFAULT_THEME.type.caption;
  const Color secondary = DEFAULT_THEME.colors.text.secondary;
  const auto screen_at = TextAt(list, frame, FiguresOf(real.Screen()), caption, secondary);
  const auto safe_at = TextAt(list, frame, FiguresOf(real.SafeArea()), caption, secondary);
  const auto content_at = TextAt(list, frame, FiguresOf(real.ContentFrame()), caption, secondary);
  Check(screen_at && safe_at && content_at, "the legend gives each area's figures");
  Check(*screen_at < *safe_at && *safe_at <= *content_at, "in the order of the names beside them");
  Check(DrawsText(list, frame, "At 0% an edge is still 32 pixels away, as a TV may crop it.", caption, secondary), "with the note that says what 0% means");
  Check(DrawsText(list, frame, "From 3% at 16:9 the content frame all but matches the safe area. Below that, or ultrawide, they differ.", caption, secondary), "and when the two look alike");
}

// The mock draws the areas the real ScaleContext gives for each choice, shrunk to its width, and the
// legend gives the same figures. The choices differ in what they move, so one fixed picture can't
// pass, and the ultrawide screen is the one where the content frame is far narrower than the safe area.
void TestFrameMockFollowsTheRealScale() {
  constexpr float MOCK_WIDTH = 880.0f;
  constexpr float SLACK = 1.5f;
  const auto &colors = DEFAULT_THEME.colors;
  struct Choice {
    float percent;
    bool ultrawide;
  };
  StylePageRig rig("spacing");
  rig.state.spacing_view = 1;
  for (const Choice &choice : {Choice{3.5f, false}, Choice{0.0f, false}, Choice{10.0f, false}, Choice{3.5f, true}}) {
    rig.state.spacing_safe_percent = choice.percent;
    rig.state.spacing_ultrawide = choice.ultrawide;
    rig.Frames(1);
    const float width = choice.ultrawide ? 3440.0f : 1920.0f;
    const float height = choice.ultrawide ? 1440.0f : 1080.0f;
    const ScaleContext real(DEFAULT_THEME.scale, width, height, {.safe_area_percent = choice.percent});
    const float shrink = MOCK_WIDTH / width;
    const auto screen = FindQuad(*rig.list, colors.surface.inset, MOCK_WIDTH, height * shrink);
    Check(screen.has_value(), "the screen is to scale");
    const auto matches = [&](const PageSpan &span, const Rect &area) {
      return std::fabs(span.left - (screen->x + area.x * shrink)) <= SLACK && std::fabs(span.top - (screen->y + area.y * shrink)) <= SLACK &&
             std::fabs(span.right - (screen->x + area.Right() * shrink)) <= SLACK && std::fabs(span.bottom - (screen->y + area.Bottom() * shrink)) <= SLACK;
    };
    Check(matches(SpanAbove(*rig.list, colors.status.success, screen->Bottom()), real.SafeArea()), "the safe area is where the scale puts it");
    Check(matches(SpanAbove(*rig.list, colors.status.warning, screen->Bottom()), real.ContentFrame()), "and the content frame");
    Check(choice.ultrawide == (real.ContentFrame().width < real.SafeArea().width - 100.0f), "which only an ultrawide screen narrows by this much");
    rig.nav.Frame([&](UiFrame &frame) {
      rig.Draw()(frame);
      CheckFrameLegend(*rig.list, frame, real);
    });
  }
  // The note says the two match at 16:9 from 3% up and not below, at each step the slider can take.
  constexpr float STEP = 0.5f;
  constexpr float MATCHES_FROM = 3.0f;
  constexpr float ALL_BUT = 2.0f;
  for (float percent = 0.0f; percent <= DEFAULT_THEME.scale.max_safe_area_percent; percent += STEP) {
    const ScaleContext screen(DEFAULT_THEME.scale, 1920.0f, 1080.0f, {.safe_area_percent = percent});
    const bool matches = std::fabs(screen.SafeArea().width - screen.ContentFrame().width) <= ALL_BUT;
    Check(matches == (percent >= MATCHES_FROM), "the note is right about when the frames match");
  }
  const ScaleContext none(DEFAULT_THEME.scale, 1920.0f, 1080.0f, {.safe_area_percent = 0.0f});
  Check(none.SafeArea().y == DEFAULT_THEME.scale.edge_min, "the note is right that 0% still keeps the minimum distance");
}

// Up to down, the D-pad reaches each control, and each changes what the mock is drawn from. A control
// beside the slider couldn't be reached, as a focused slider keeps left and right.
void TestFrameControls() {
  StylePageRig rig("spacing");
  auto draw = rig.Draw();
  rig.Frames(2);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.spacing_view == 0, "entering stays on the view");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.spacing_view == 1, "and moving along the tabs chooses the next");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.spacing_safe_percent == 4.0f, "down reaches the slider, and right moves it half a percent");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(!rig.state.spacing_ultrawide, "down reaches the toggle without changing it");
  rig.nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(rig.state.spacing_ultrawide, "which confirm turns on");
  rig.state.Reset();
  Check(rig.state.spacing_view == 0 && rig.state.spacing_safe_percent == DEFAULT_THEME.scale.default_safe_area_percent && !rig.state.spacing_ultrawide, "a reset puts them back");
}

// Each padding diagram insets the content's surface from the tint by that surface's padding, written
// out here apart from the components, in the order of the diagrams.
void TestPaddingDiagrams() {
  constexpr float WIDTH = 250.0f;
  constexpr float HEIGHT = 120.0f;
  struct Padding {
    const char *title;
    float across;
    float down;
  };
  constexpr std::array WANTED{Padding{"Card, 16 x 16", 16, 16}, Padding{"Toast, 16 x 16", 16, 16}, Padding{"Dialog, 32 x 32", 32, 32}, Padding{"List row, 16 x 0", 16, 0},
                              Padding{"HUD, 12 x 8", 12, 8}};
  const auto &colors = DEFAULT_THEME.colors;
  StylePageRig rig("spacing");
  rig.state.spacing_view = 1;
  rig.Frames(1);
  const auto &vertices = rig.list->VtxBuffer;
  size_t found = 0;
  constexpr int LAST_OF_THE_TWO_QUADS = 7;
  for (int index = 0; index + LAST_OF_THE_TWO_QUADS < vertices.Size; ++index) {
    const bool is_tint = vertices[index].pos.x > PANE_LEFT && vertices[index].col == ToImU32(colors.accent.tint);
    const bool is_inner = vertices[index + 4].col == ToImU32(colors.surface.raised);
    if (!is_tint || !is_inner) {
      continue;
    }
    Check(found < WANTED.size(), "there are only the five");
    const ImVec2 outer = vertices[index].pos, outer_end = vertices[index + 2].pos, inner = vertices[index + 4].pos, inner_end = vertices[index + 6].pos;
    Check(std::fabs(outer_end.x - outer.x - WIDTH) <= 0.5f && std::fabs(outer_end.y - outer.y - HEIGHT) <= 0.5f, "each diagram has the same size");
    Check(std::fabs(inner.x - outer.x - WANTED[found].across) <= 0.5f && std::fabs(outer_end.x - inner_end.x - WANTED[found].across) <= 0.5f, "its padding across");
    Check(std::fabs(inner.y - outer.y - WANTED[found].down) <= 0.5f && std::fabs(outer_end.y - inner_end.y - WANTED[found].down) <= 0.5f, "and down");
    ++found;
    index += LAST_OF_THE_TWO_QUADS;
  }
  Check(found == WANTED.size(), "every surface has its diagram");
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    for (const Padding &padding : WANTED) {
      Check(DrawsText(*rig.list, frame, padding.title, DEFAULT_THEME.type.caption, colors.text.primary), "and says what it shows");
    }
  });
}

// ---- Effects ------------------------------------------------------------------------------------

// Whether a vertex has `colour`'s red, green and blue, at any alpha, as a glow or shadow fades its own.
bool SameHue(const ImDrawVert &vertex, Color colour) {
  return (vertex.col & ~IM_COL32_A_MASK) == (ToImU32(colour) & ~IM_COL32_A_MASK);
}

// How far a shadow or glow reaches past a card.
struct Reach {
  float left = 0.0f;
  float down = 0.0f;
};

// How far past `card_left` to the left, or past `card_bottom` downward, the page draws `colour` near a
// card, which lies between `card_left` and `card_left + card_width`, and between the given tops.
Reach ReachOf(const ImDrawList &list, Color colour, float card_left, float card_width, float top, float card_bottom, float room_across, float room_down) {
  Reach reach;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    const bool near_card = vertex.pos.x > PANE_LEFT && vertex.pos.x >= card_left - room_across && vertex.pos.x <= card_left + card_width + room_across && vertex.pos.y >= top - room_down &&
                           vertex.pos.y <= card_bottom + room_down;
    if (near_card && SameHue(vertex, colour)) {
      reach.left = std::max(reach.left, card_left - vertex.pos.x);
      reach.down = std::max(reach.down, vertex.pos.y - card_bottom);
    }
  }
  return reach;
}

// Checks the page draws each of `lines`, each after the one before.
void CheckInOrder(const ImDrawList &page, UiFrame &frame, std::initializer_list<const char *> lines, const TypeStyle &style, Color colour) {
  std::ptrdiff_t last = -1;
  for (const char *line : lines) {
    const auto at = TextAt(page, frame, line, style, colour);
    // The line's own words as the message, so a failure says which one is missing or out of order.
    Check(at && *at > last, line);
    last = *at;
  }
}

// Each level's shadow reaches below its card by its blur and offset, written out here apart from the
// theme, and level 0 has none. Each glow reaches out by its width in its own colour, and the card with
// no glow is the baseline. Their captions say the same figures.
void TestEffectsDepth() {
  constexpr float LEVEL_LEFT = 384.0f;
  constexpr float LEVEL_PITCH = 264.0f;
  constexpr float LEVEL_WIDTH = 240.0f;
  constexpr float LEVEL_TOP = 292.0f;
  constexpr float LEVEL_BOTTOM = 404.0f;
  constexpr float BACKDROP_WIDTH = 1360.0f;
  constexpr float BACKDROP_HEIGHT = 240.0f;
  constexpr float NEXT_SHADOW_SPREAD = 50.0f;
  constexpr float GLOW_LEFT = 384.0f;
  constexpr float GLOW_PITCH = 288.0f;
  constexpr float GLOW_TOP = 552.0f;
  constexpr float GLOW_BOTTOM = 648.0f;
  constexpr float GLOW_ROOM = 34.0f;
  constexpr float SLACK = 2.0f;
  const auto &colors = DEFAULT_THEME.colors;
  constexpr std::array WANTED_SHADOWS{0.0f, 10.0f, 22.0f, 38.0f, 64.0f};
  StylePageRig rig("effects");
  rig.Frames(1);
  for (size_t level = 0; level < WANTED_SHADOWS.size(); ++level) {
    const float left = LEVEL_LEFT + LEVEL_PITCH * static_cast<float>(level);
    // Only the card's left part, as the next card's deeper shadow spreads over its right end.
    const Reach reach = ReachOf(*rig.list, colors.effect.shadow, left, LEVEL_WIDTH - NEXT_SHADOW_SPREAD, LEVEL_TOP, LEVEL_BOTTOM, 0.0f, WANTED_SHADOWS.back() + SLACK);
    Check(std::fabs(reach.down - WANTED_SHADOWS[level]) <= SLACK, "each level's shadow reaches as far as its blur and offset");
  }
  const auto backdrop = FindQuad(*rig.list, Palette::GRAY_500, BACKDROP_WIDTH, BACKDROP_HEIGHT);
  Check(backdrop && backdrop->Bottom() >= LEVEL_BOTTOM + WANTED_SHADOWS.back(), "the backdrop has room for the deepest shadow");
  const std::array<std::pair<Color, float>, 3> glows{std::pair{colors.focus.glow, 12.0f}, std::pair{colors.accent.glow, 24.0f}, std::pair{colors.trophy.gold, 32.0f}};
  for (size_t glow = 0; glow < glows.size(); ++glow) {
    const float left = GLOW_LEFT + GLOW_PITCH * static_cast<float>(glow + 1);
    // Just past the widest glow, and short of what the card before spreads to its right.
    const Reach reach = ReachOf(*rig.list, glows[glow].first, left, LEVEL_WIDTH, GLOW_TOP, GLOW_BOTTOM, GLOW_ROOM, GLOW_ROOM);
    Check(std::fabs(reach.left - glows[glow].second) <= SLACK, "each glow reaches out by its width, in its colour");
  }
  Check(ReachOf(*rig.list, colors.focus.glow, GLOW_LEFT, LEVEL_WIDTH, GLOW_TOP, GLOW_BOTTOM, GLOW_ROOM, GLOW_ROOM).left == 0.0f, "and the card with no glow has none");
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    const auto &caption = DEFAULT_THEME.type.caption;
    CheckInOrder(*rig.list, frame, {"Level 0", "Level 1", "Level 2", "Level 3", "Level 4", "No glow", "Focus", "Emphasis", "Trophy"}, DEFAULT_THEME.type.label, colors.text.primary);
    for (const char *line : {"flat, blur 0, y 0, 0.00", "cards in lists, blur 8, y 2, 0.30", "a focused card, blur 16, y 6, 0.40", "toasts, dropdowns, blur 28, y 10, 0.45", "modals, blur 48, y 16, 0.50",
                             "0 wide, 0.00 at its peak", "12 wide, 0.35 at its peak", "24 wide, 0.45 at its peak", "32 wide, 0.30 at its peak"}) {
      Check(DrawsText(*rig.list, frame, line, caption, colors.text.secondary), "each says its figures");
    }
  });
}

// The middle of the shine's band on a card, from its vertices, or nothing with no band. A shine draws
// in a clip of its own, just inside the card, which keeps out what the next card's band spreads to,
// and the vertices run past the clip, so they place the whole band.
std::optional<float> BandCentre(const ImDrawList &list, float left, float width, float top, float bottom) {
  constexpr float CLIP_INSET = 8.0f;
  float first = std::numeric_limits<float>::max(), last = -first;
  for (const ImDrawCmd &command : list.CmdBuffer) {
    const bool on_card = std::fabs(command.ClipRect.x - left) <= CLIP_INSET && std::fabs(command.ClipRect.z - left - width) <= CLIP_INSET && command.ClipRect.y >= top && command.ClipRect.w <= bottom;
    for (unsigned at = 0; on_card && at < command.ElemCount; ++at) {
      const ImDrawVert &vertex = list.VtxBuffer[static_cast<int>(command.VtxOffset + list.IdxBuffer[static_cast<int>(command.IdxOffset + at)])];
      if (SameHue(vertex, DEFAULT_THEME.colors.effect.shine)) {
        first = std::min(first, vertex.pos.x);
        last = std::max(last, vertex.pos.x);
      }
    }
  }
  return last >= first ? std::optional<float>((first + last) / 2.0f) : std::nullopt;
}

// Each held frame's band lies further along than the last, by steps the Standard curve makes uneven,
// so the middle step is the longest, and each caption states it, worked out here apart from the code.
// A band stepping evenly would rise in order too, so the steps are what show the curve. The tokens
// line isn't cut off. The live card sweeps and then rests, so over a whole loop its band is both there and gone.
void TestEffectsShine() {
  constexpr float HELD_LEFT = 352.0f;
  constexpr float HELD_PITCH = 232.0f;
  constexpr float HELD_WIDTH = 220.0f;
  constexpr float HELD_TOP = 434.0f;
  constexpr float HELD_BOTTOM = 544.0f;
  constexpr float LIVE_LEFT = 352.0f;
  constexpr float LIVE_WIDTH = 480.0f;
  constexpr float LIVE_TOP = 188.0f;
  constexpr float LIVE_BOTTOM = 348.0f;
  constexpr int HELD_FRAMES = 6;
  StylePageRig rig("effects");
  rig.state.effects_view = 1;
  rig.Frames(1);
  std::vector<float> places;
  for (int held = 0; held < HELD_FRAMES; ++held) {
    const float left = HELD_LEFT + HELD_PITCH * static_cast<float>(held);
    const auto centre = BandCentre(*rig.list, left, HELD_WIDTH, HELD_TOP, HELD_BOTTOM);
    Check(centre.has_value(), "each frame shows the band");
    const float across = (*centre - left) / HELD_WIDTH;
    Check(places.empty() || across > places.back(), "further along than the last");
    places.push_back(across);
  }
  // The curve's middle step is about 1.4 times its neighbours, and even steps are equal but for rounding.
  constexpr float CLEARLY_LONGER = 1.2f;
  const float middle_step = places[3] - places[2];
  Check(middle_step > (places[2] - places[1]) * CLEARLY_LONGER && middle_step > (places[4] - places[3]) * CLEARLY_LONGER, "by steps the curve makes longest in the middle");
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    const auto &caption = DEFAULT_THEME.type.caption;
    const Color colour = DEFAULT_THEME.colors.text.secondary;
    for (const char *line : {"175 ms, at 0.06", "245 ms, at 0.17", "315 ms, at 0.36", "385 ms, at 0.64", "455 ms, at 0.83", "525 ms, at 0.94"}) {
      Check(DrawsText(*rig.list, frame, line, caption, colour), "each says when it is and how far the curve has come");
    }
    Check(DrawsText(*rig.list, frame, "700 ms on the Standard curve, leaning 20 degrees, 30% of the width and at least 80, 0.14 at its brightest", caption, colour), "and the tokens are given whole");
  });
  constexpr int FRAMES_IN_A_LOOP = 150;
  bool swept = false, rested = false;
  for (int step = 0; step < FRAMES_IN_A_LOOP; ++step) {
    rig.Frames(1);
    const bool band = BandCentre(*rig.list, LIVE_LEFT, LIVE_WIDTH, LIVE_TOP, LIVE_BOTTOM).has_value();
    swept = swept || band;
    rested = rested || !band;
  }
  Check(swept && rested, "the live card sweeps and rests");
}

// With reduced motion the Shine view has no shine, live or held, over a whole loop, where without it
// both show one, as TestEffectsShine shows. The view's toggle is reached by the D-pad, and the frames
// and live card say why they are empty.
void TestEffectsShineReduced() {
  constexpr float HELD_LEFT = 352.0f;
  constexpr float HELD_PITCH = 232.0f;
  constexpr float HELD_WIDTH = 220.0f;
  constexpr float HELD_TOP = 434.0f;
  constexpr float HELD_BOTTOM = 544.0f;
  constexpr float LIVE_LEFT = 352.0f;
  constexpr float LIVE_WIDTH = 480.0f;
  constexpr float LIVE_TOP = 188.0f;
  constexpr float LIVE_BOTTOM = 348.0f;
  constexpr int HELD_FRAMES = 6;
  constexpr int FRAMES_IN_A_LOOP = 150;
  StylePageRig rig("effects");
  auto draw = rig.Draw();
  rig.Frames(2);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.effects_view == 1, "the tabs reach the Shine view");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  rig.nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(rig.state.effects_reduced_motion, "and down reaches its toggle, which confirm turns on");
  for (int step = 0; step < FRAMES_IN_A_LOOP; ++step) {
    rig.Frames(1);
    Check(!BandCentre(*rig.list, LIVE_LEFT, LIVE_WIDTH, LIVE_TOP, LIVE_BOTTOM).has_value(), "the live card never sweeps");
  }
  for (int held = 0; held < HELD_FRAMES; ++held) {
    Check(!BandCentre(*rig.list, HELD_LEFT + HELD_PITCH * static_cast<float>(held), HELD_WIDTH, HELD_TOP, HELD_BOTTOM).has_value(), "and no frame holds a band");
  }
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    Check(DrawsText(*rig.list, frame, "No shine with reduced motion", DEFAULT_THEME.type.label, DEFAULT_THEME.colors.text.primary), "the card says why");
    Check(DrawsText(*rig.list, frame, "No shine", DEFAULT_THEME.type.caption, DEFAULT_THEME.colors.text.secondary), "and so do the frames");
    Check(!DrawsText(*rig.list, frame, "175 ms, at 0.06", DEFAULT_THEME.type.caption, DEFAULT_THEME.colors.text.secondary), "in place of their times");
  });
}

// How the Motion view's plots and bars are laid out, written out here apart from the page.
constexpr float MOTION_PLOT_WIDTH = 240.0f;
constexpr float MOTION_PLOT_HEIGHT = 160.0f;
constexpr float MOTION_PLOT_PITCH = 256.0f;
constexpr float MOTION_PLOT_TOP_VALUE = 1.2f;
constexpr std::array MOTION_EASES{Ease::Linear, Ease::Enter, Ease::Exit, Ease::Standard, Ease::Emphasis};

// Checks each curve's dots lie where ApplyEase puts them at even steps of time, and with reduced motion
// on the straight line the real rule gives. Each bar is as long as its duration, written out here apart
// from the theme, and with reduced motion none is longer than the cap or keeps its full length.
void CheckMotionView(const ImDrawList &list, bool reduced) {
  constexpr int DOTS = 10;
  constexpr float BAR_HEIGHT = 8.0f;
  constexpr float BAR_PER_MS = 0.5f;
  constexpr float CAP_MS = 120.0f;
  constexpr std::array DURATIONS{100.0f, 180.0f, 280.0f, 420.0f, 600.0f, 700.0f, 120.0f, 1600.0f};
  const auto &colors = DEFAULT_THEME.colors;
  const auto first = FindQuad(list, colors.surface.inset, MOTION_PLOT_WIDTH, MOTION_PLOT_HEIGHT);
  Check(first.has_value(), "the plots are drawn");
  for (size_t plot = 0; plot < MOTION_EASES.size(); ++plot) {
    const float left = first->x + MOTION_PLOT_PITCH * static_cast<float>(plot);
    const Ease ease = reduced ? Ease::Linear : MOTION_EASES[plot];
    for (int dot = 0; dot <= DOTS; ++dot) {
      const float share = static_cast<float>(dot) / DOTS;
      const ImVec2 wanted{left + MOTION_PLOT_WIDTH * share, first->Bottom() - MOTION_PLOT_HEIGHT * ApplyEase(ease, share) / MOTION_PLOT_TOP_VALUE};
      const bool found = std::any_of(list.VtxBuffer.begin(), list.VtxBuffer.end(), [&](const ImDrawVert &vertex) {
        return vertex.col == ToImU32(colors.accent.line) && std::hypot(vertex.pos.x - wanted.x, vertex.pos.y - wanted.y) <= 4.0f;
      });
      Check(found, "each dot lies on its curve");
    }
  }
  for (const float full : DURATIONS) {
    const float lasts = reduced ? std::min(full, CAP_MS) : full;
    Check(FindQuad(list, colors.accent.fill, lasts * BAR_PER_MS, BAR_HEIGHT).has_value(), "each bar is as long as it lasts");
    if (lasts < full) {
      Check(!FindQuad(list, colors.accent.fill, full * BAR_PER_MS, BAR_HEIGHT).has_value(), "and with reduced motion none keeps its full length");
    }
  }
}

// Checks only Emphasis rises past the line at 1.
void CheckOnlyEmphasisOvershoots(const ImDrawList &list) {
  const auto &colors = DEFAULT_THEME.colors;
  const Rect first = *FindQuad(list, colors.surface.inset, MOTION_PLOT_WIDTH, MOTION_PLOT_HEIGHT);
  const float line_y = first.Bottom() - MOTION_PLOT_HEIGHT / MOTION_PLOT_TOP_VALUE;
  for (size_t plot = 0; plot < MOTION_EASES.size(); ++plot) {
    const float left = first.x + MOTION_PLOT_PITCH * static_cast<float>(plot);
    const bool above = std::any_of(list.VtxBuffer.begin(), list.VtxBuffer.end(), [&](const ImDrawVert &vertex) {
      return vertex.col == ToImU32(colors.accent.line) && vertex.pos.x >= left && vertex.pos.x <= left + MOTION_PLOT_WIDTH && vertex.pos.y < line_y - 6.0f && vertex.pos.y >= first.y;
    });
    Check(above == (MOTION_EASES[plot] == Ease::Emphasis), "only Emphasis rises past 1");
  }
}

// The page's curves and durations, drawn and named in order, in both modes. The toggle is reached by
// the D-pad. A name is tied to its figures by order, as a swap would keep every bar and dot.
void TestEffectsMotion() {
  StylePageRig rig("effects");
  auto draw = rig.Draw();
  rig.Frames(2);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.effects_view == 0, "entering stays on the view");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.effects_view == 2, "and moving along the tabs chooses the next");
  rig.Frames(2);
  CheckMotionView(*rig.list, false);
  CheckOnlyEmphasisOvershoots(*rig.list);
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    const auto &caption = DEFAULT_THEME.type.caption;
    const Color colour = DEFAULT_THEME.colors.text.secondary;
    CheckInOrder(*rig.list, frame, {"Linear", "Enter", "Exit", "Standard", "Emphasis"}, caption, colour);
    CheckInOrder(*rig.list, frame,
                 {"fast", "100 ms", "base", "180 ms", "slow", "280 ms", "slower", "420 ms", "emphasis", "600 ms", "shine", "700 ms", "exit_fade", "120 ms", "pulse", "1600 ms"}, caption, colour);
  });
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  rig.nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(rig.state.effects_reduced_motion, "the D-pad reaches the toggle, and confirm turns it on");
  rig.Frames(1);
  CheckMotionView(*rig.list, true);
  rig.state.Reset();
  Check(rig.state.effects_view == 0 && !rig.state.effects_reduced_motion, "a reset puts them back");
}

// The loop's position holds to a fraction of a millisecond when the clock has run for days, as a
// machine's boot clock has. A float count of milliseconds would be off by tens of them by then.
void TestLoopHoldsOnALongUptime() {
  using namespace std::chrono;
  constexpr float PERIOD = 2000.0f;
  constexpr float SLACK = 0.01f;
  constexpr int DAYS = 6;
  const auto start = UiClock::time_point(duration_cast<UiClock::duration>(hours(24 * DAYS)));
  for (const int offset : {0, 123, 777, 1999}) {
    const auto now = start + duration_cast<UiClock::duration>(milliseconds(offset));
    const double wanted = std::fmod(static_cast<double>(duration_cast<milliseconds>(now.time_since_epoch()).count()), static_cast<double>(PERIOD));
    Check(std::fabs(LoopMs(now, PERIOD) - static_cast<float>(wanted)) <= SLACK, "the loop's position is exact after days of uptime");
  }
  Check(std::fabs(LoopMs(UiClock::time_point(duration_cast<UiClock::duration>(milliseconds(2500))), PERIOD) - 500.0f) <= SLACK, "and it wraps at the period");
}

// ---- Scaled drawing and the stage ----------------------------------------------------------------

// The clip of the draw command that holds the first vertex of `colour`.
ImVec4 ClipOf(const ImDrawList &list, Color colour) {
  for (const ImDrawCmd &command : list.CmdBuffer) {
    for (unsigned at = 0; at < command.ElemCount; ++at) {
      if (list.VtxBuffer[static_cast<int>(command.VtxOffset + list.IdxBuffer[static_cast<int>(command.IdxOffset + at)])].col == ToImU32(colour)) {
        return command.ClipRect;
      }
    }
  }
  Check(false, "the colour was drawn");
  return {};
}

// The box around everything the list holds in `colour` on the page.
PageSpan SpanOf(const ImDrawList &list, Color colour, float below = 0.0f) {
  PageSpan span;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if (vertex.pos.x > PANE_LEFT && vertex.pos.y >= below && vertex.col == ToImU32(colour)) {
      span.left = std::min(span.left, vertex.pos.x);
      span.top = std::min(span.top, vertex.pos.y);
      span.right = std::max(span.right, vertex.pos.x);
      span.bottom = std::max(span.bottom, vertex.pos.y);
    }
  }
  return span;
}

// Whether two clips are the same.
bool SameClip(const ImVec4 &first, const ImVec4 &second) {
  constexpr float SLACK = 0.01f;
  return std::fabs(first.x - second.x) <= SLACK && std::fabs(first.y - second.y) <= SLACK && std::fabs(first.z - second.z) <= SLACK && std::fabs(first.w - second.w) <= SLACK;
}

// What's drawn inside the scope shrinks toward its origin, its clips with it, held to what's visible,
// and what's drawn before and after doesn't move or change clip. Unscaled drawing of the same things is
// the baseline, so the sums below are the only way it can pass. Text past the window still arrives, as
// its clip is the scope's, and the window's content size is put back.
void TestScaledDrawScope() {
  constexpr float FIT = 0.5f;
  constexpr Vec2 ORIGIN{400.0f, 400.0f};
  constexpr Rect VISIBLE{400.0f, 400.0f, 300.0f, 600.0f};
  constexpr Color BEFORE = Rgb(0x112233), INSIDE = Rgb(0x223344), INNER = Rgb(0x334455), WIDE = Rgb(0x445566), AFTER = Rgb(0x556677), FAR_TEXT = Rgb(0x667788);
  // Past the window at full size, which would cull it, and inside the visible area once shrunk.
  constexpr float FAR_Y = 1200.0f;
  StylePageRig rig("colour");
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    ImDrawList &list = *ImGui::GetWindowDrawList();
    const ImVec4 outer = list._CmdHeader.ClipRect;
    const float content_x = ImGui::GetCurrentWindow()->DC.CursorMaxPos.x;
    list.AddRectFilled({500.0f, 500.0f}, {510.0f, 510.0f}, ToImU32(BEFORE));
    {
      const ScaledDrawScope scope(list, ORIGIN, FIT, VISIBLE);
      list.AddRectFilled({600.0f, 600.0f}, {640.0f, 640.0f}, ToImU32(INSIDE));
      list.PushClipRect({500.0f, 500.0f}, {600.0f, 600.0f}, true);
      list.AddRectFilled({520.0f, 520.0f}, {540.0f, 540.0f}, ToImU32(INNER));
      list.PopClipRect();
      list.PushClipRect({0.0f, 0.0f}, {5000.0f, 5000.0f}, false);
      list.AddRectFilled({420.0f, 420.0f}, {440.0f, 440.0f}, ToImU32(WIDE));
      list.PopClipRect();
      list.AddText(frame.Ui().fonts.regular, 20.0f, {420.0f, FAR_Y}, ToImU32(FAR_TEXT), "far");
      ImGui::GetCurrentWindow()->DC.CursorMaxPos.x = 9000.0f;
    }
    list.AddRectFilled({520.0f, 500.0f}, {530.0f, 510.0f}, ToImU32(AFTER));
    const auto near = [](float actual, float wanted) { return std::fabs(actual - wanted) <= 0.01f; };
    const auto is_at = [&](Color colour, float left, float top, float right, float bottom) {
      const PageSpan span = SpanOf(list, colour);
      return near(span.left, left) && near(span.top, top) && near(span.right, right) && near(span.bottom, bottom);
    };
    Check(is_at(BEFORE, 500.0f, 500.0f, 510.0f, 510.0f), "what came before stays");
    Check(is_at(INSIDE, 500.0f, 500.0f, 520.0f, 520.0f), "what's inside shrinks toward the origin");
    Check(is_at(AFTER, 520.0f, 500.0f, 530.0f, 510.0f), "and what comes after stays");
    Check(SameClip(ClipOf(list, BEFORE), outer), "what came before keeps its clip");
    Check(SameClip(ClipOf(list, AFTER), outer), "and so does what comes after");
    Check(SameClip(ClipOf(list, INNER), {450.0f, 450.0f, 500.0f, 500.0f}), "a clip inside shrinks with it");
    Check(SameClip(ClipOf(list, WIDE), {VISIBLE.x, VISIBLE.y, VISIBLE.Right(), VISIBLE.Bottom()}), "and one that ignores those around it is held to what's visible");
    Check(PageVertices(list, FAR_TEXT) > 0, "text past the window at full size isn't culled");
    Check(ImGui::GetCurrentWindow()->DC.CursorMaxPos.x == content_x, "and the window's content size is put back");
  });
}

// When the scope's full-size clip happens to be the clip already in use, what came before must still
// keep it, in a draw command of its own, or scaling the scope's clips would move theirs too. A clip
// pushed and popped with nothing drawn, as an empty clipped child does, would merge that command back
// into the one before, and what's drawn next would then be left out of the scaling.
void TestScaledDrawScopeWithTheSameClip() {
  constexpr float FIT = 0.5f;
  constexpr Vec2 ORIGIN{0.0f, 0.0f};
  constexpr Color BEFORE = Rgb(0x778899), INSIDE = Rgb(0x8899AA);
  StylePageRig rig("colour");
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    ImDrawList &list = *ImGui::GetWindowDrawList();
    const ImVec4 outer = list._CmdHeader.ClipRect;
    list.AddRectFilled({500.0f, 500.0f}, {510.0f, 510.0f}, ToImU32(BEFORE));
    {
      const Rect shrunk{outer.x * FIT, outer.y * FIT, (outer.z - outer.x) * FIT, (outer.w - outer.y) * FIT};
      const ScaledDrawScope scope(list, ORIGIN, FIT, shrunk);
      list.PushClipRect({10.0f, 10.0f}, {20.0f, 20.0f}, true);
      list.PopClipRect();
      list.AddRectFilled({600.0f, 600.0f}, {640.0f, 640.0f}, ToImU32(INSIDE));
    }
    Check(SameClip(ClipOf(list, BEFORE), outer), "what came before keeps its clip");
    Check(SameClip(ClipOf(list, INSIDE), {outer.x * FIT, outer.y * FIT, outer.z * FIT, outer.w * FIT}), "and what's inside has the shrunk one");
  });
}

// Specimens shrink by whichever side is tightest, and never grow. Worked out here apart from the code.
void TestStageFit() {
  constexpr Rect AREA{100.0f, 100.0f, 1000.0f, 500.0f};
  Check(StageFit({500.0f, 250.0f}, AREA) == 1.0f, "what fits isn't grown");
  Check(StageFit({2000.0f, 250.0f}, AREA) == 0.5f, "too wide shrinks by the width");
  Check(StageFit({500.0f, 2000.0f}, AREA) == 0.25f, "too tall shrinks by the height");
  Check(StageFit({2000.0f, 2000.0f}, AREA) == 0.25f, "and both by the tighter");
}

// A component page on its stage, the Progress page unless asked for another.
struct StageRig {
  explicit StageRig(std::string_view page = "progress") : rig(page) { rig.Frames(2); }
  // The backdrop's panel.
  [[nodiscard]] PageSpan Panel() const { return SpanOf(*rig.list, STAGE_BACKDROPS[static_cast<size_t>(rig.state.stage.backdrop)]); }
  StylePageRig rig;
};

// Each output lays the specimens out at its own scale, so a bar's length follows it, and they are
// shrunk only when too big, until they just fit. The page's own parts don't change with the output,
// as the panel staying put shows. Lengths are worked out here apart from the code.
void TestStageOutputs() {
  constexpr float INSET = 16.0f;
  constexpr float EDGE_SLACK = 2.0f;
  struct Output {
    int index;
    float bar;
    bool shrunk;
  };
  const auto &colors = DEFAULT_THEME.colors;
  StageRig stage;
  auto &rig = stage.rig;
  const PageSpan panel = stage.Panel();
  for (const Output &output : {Output{0, 253.0f, false}, Output{1, 380.0f, false}, Output{2, 760.0f, true}, Output{3, 507.0f, true}}) {
    rig.state.stage.output = output.index;
    rig.Frames(2);
    const float fit = rig.state.stage.fit;
    Check((fit < 1.0f) == output.shrunk, "the specimens are shrunk only when they're too big");
    const PageSpan fill = SpanOf(*rig.list, colors.status.success);
    Check(std::fabs(fill.right - fill.left - output.bar * fit) <= EDGE_SLACK, "a bar's length follows the output's scale and the fit");
    const PageSpan now = stage.Panel();
    Check(now.left == panel.left && now.top == panel.top && now.right == panel.right && now.bottom == panel.bottom, "and the page around them doesn't change");
    // Effects may spill to the panel's edge, past the inset that specimens keep.
    Check(SameClip(ClipOf(*rig.list, colors.status.success), {panel.left, panel.top, panel.right, panel.bottom}), "what they draw is held to the panel");
    const PageSpan cells = SpanOf(*rig.list, colors.surface.raised);
    Check(cells.left >= panel.left + INSET - 0.5f && cells.right <= panel.right - INSET + 0.5f && cells.bottom <= panel.bottom - INSET + 0.5f, "they stay in the panel");
    if (output.shrunk) {
      rig.nav.Frame([&](UiFrame &frame) {
        rig.Draw()(frame);
        std::array<char, 32> words{};
        (void)std::snprintf(words.data(), words.size(), "Shrunk to %.0f%%", static_cast<double>(rig.state.stage.fit * 100.0f));
        Check(DrawsText(*rig.list, frame, words.data(), DEFAULT_THEME.type.caption, colors.text.secondary), "and the strip says how far");
      });
      Check(std::fabs(cells.right - (panel.right - INSET)) <= EDGE_SLACK || std::fabs(cells.bottom - (panel.bottom - INSET)) <= EDGE_SLACK, "and shrinking stops once they fit");
    }
  }
}

// The opacity reaches the specimens and not the page's own controls. Each backdrop fills the panel in
// its colour, and the others aren't drawn.
void TestStageOpacityAndBackdrop() {
  const auto &colors = DEFAULT_THEME.colors;
  StageRig stage;
  auto &rig = stage.rig;
  Check(PageVertices(*rig.list, colors.status.success) > 0, "at full opacity a fill is its own colour");
  rig.state.stage.opacity = 0.5f;
  rig.Frames(2);
  Check(PageVertices(*rig.list, colors.status.success) == 0 && PageVertices(*rig.list, colors.status.success.MulAlpha(0.5f)) > 0, "the opacity fades it");
  // Only in the slider's row, as the page's title and chosen tabs are the same white.
  constexpr float SLIDER_ROW_TOP = 110.0f;
  constexpr float SLIDER_ROW_BOTTOM = 150.0f;
  const bool knob_is_whole = std::any_of(rig.list->VtxBuffer.begin(), rig.list->VtxBuffer.end(), [&](const ImDrawVert &vertex) {
    return vertex.pos.x > PANE_LEFT && vertex.pos.y >= SLIDER_ROW_TOP && vertex.pos.y <= SLIDER_ROW_BOTTOM && vertex.col == ToImU32(colors.control.knob);
  });
  Check(knob_is_whole, "and leaves the page's slider alone");
  for (int backdrop = 0; backdrop < static_cast<int>(STAGE_BACKDROPS.size()); ++backdrop) {
    rig.state.stage.backdrop = backdrop;
    rig.Frames(1);
    for (int other = 0; other < static_cast<int>(STAGE_BACKDROPS.size()); ++other) {
      const int found = PageVertices(*rig.list, STAGE_BACKDROPS[static_cast<size_t>(other)]);
      Check((found > 0) == (other == backdrop), "only the chosen backdrop is drawn");
    }
  }
  rig.state.stage.output = 2;
  rig.state.stage.fit = 0.5f;
  rig.state.Reset();
  Check(rig.state.stage.output == STAGE_OUTPUT_1080P && rig.state.stage.backdrop == STAGE_BACKDROP_MID && rig.state.stage.opacity == 1.0f && rig.state.stage.fit == 1.0f, "a reset puts them back");
}

// Focus enters on the opacity slider, whose left and right change it, and down reaches the tabs. Moving
// along the output tabs chooses the next, and past the last it carries on to the backdrop tabs, the
// first press landing on the chosen one and the second moving along.
void TestStageControls() {
  StageRig stage;
  auto &rig = stage.rig;
  auto draw = rig.Draw();
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(FocusedControl() == rig.state.entry && rig.state.entry != 0, "right enters on the slider");
  rig.nav.Press(ImGuiKey_GamepadDpadLeft, draw);
  Check(std::fabs(rig.state.stage.opacity - 0.95f) <= 0.001f && FocusedControl() == rig.state.entry, "left turns the opacity down a step");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(FocusedControl() != rig.state.entry, "down reaches the tabs");
  Check(rig.state.stage.output == STAGE_OUTPUT_1080P, "landing on the chosen one");
  const int last = static_cast<int>(STAGE_OUTPUTS.size()) - 1;
  for (int output = STAGE_OUTPUT_1080P + 1; output <= last; ++output) {
    rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
    Check(rig.state.stage.output == output, "moving along them chooses each output");
  }
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.stage.output == last && rig.state.stage.backdrop == STAGE_BACKDROP_MID, "past the last, focus lands on the chosen backdrop");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.stage.output == last && rig.state.stage.backdrop == STAGE_BACKDROP_MID + 1, "and moving along chooses the next");
}

// Each bar says what it shows in words, row by row, and how far along it is: nothing at 0%, the dot at
// 1%, half, and all of it, in each fill. Worked out here apart from the code.
void TestProgressSpecimens() {
  constexpr float DOT = 8.0f;
  constexpr float HALF = 190.0f;
  constexpr float WHOLE = 380.0f;
  const auto &colors = DEFAULT_THEME.colors;
  StageRig stage;
  auto &rig = stage.rig;
  const float panel_top = stage.Panel().top;
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    Check(DrawsText(*rig.list, frame, "Full size", DEFAULT_THEME.type.caption, colors.text.secondary), "nothing shrunk says so");
  });
  for (const Color fill : {colors.accent.line, colors.status.success, colors.status.error}) {
    std::vector<std::pair<float, float>> rows;
    std::vector<float> tops;
    for (const ImDrawVert &vertex : rig.list->VtxBuffer) {
      if (vertex.pos.x <= PANE_LEFT || vertex.pos.y < panel_top || vertex.col != ToImU32(fill)) {
        continue;
      }
      if (tops.empty() || vertex.pos.y > tops.back() + DOT + 1.0f) {
        tops.push_back(vertex.pos.y);
        rows.push_back({vertex.pos.x, vertex.pos.x});
      }
      rows.back().first = std::min(rows.back().first, vertex.pos.x);
      rows.back().second = std::max(rows.back().second, vertex.pos.x);
    }
    Check(rows.size() == 3, "a bar at 0% has no fill, and the others have");
    Check(std::fabs(rows[0].second - rows[0].first - DOT) <= 1.0f, "at 1% it's the dot");
    Check(std::fabs(rows[1].second - rows[1].first - HALF) <= 1.0f, "at 50% it's half");
    Check(std::fabs(rows[2].second - rows[2].first - WHOLE) <= 1.0f, "and at 100% all of it");
  }
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    CheckInOrder(*rig.list, frame, {"Accent 0%", "Success 0%", "Error 0%", "Accent 1%", "Success 1%", "Error 1%", "Accent 50%", "Success 50%", "Error 50%", "Accent 100%", "Success 100%", "Error 100%"}, DEFAULT_THEME.type.caption, colors.text.primary);
  });
}

// ---- Interactive component pages ------------------------------------------------------------------

// How far a stage's specimens or live copy sit inside its panel.
constexpr float STAGE_INSET = 16.0f;
constexpr int STAGE_DARK = 0;

// Clicks the page at a place.
void ClickAt(StylePageRig &rig, float across, float down) {
  auto draw = rig.Draw();
  rig.nav.MoveMouse(across, down, draw);
  rig.nav.Frame(draw);
  rig.nav.MouseButton(true);
  rig.nav.Frame(draw);
  rig.nav.MouseButton(false);
  rig.nav.Frames(2, draw);
}

// The box around what the panel holds in `colour` from `top` down as far as `bottom`.
PageSpan SpanBetween(const ImDrawList &list, Color colour, float top, float bottom) {
  PageSpan span;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if (vertex.pos.x > PANE_LEFT && vertex.pos.y >= top && vertex.pos.y <= bottom && vertex.col == ToImU32(colour)) {
      span.left = std::min(span.left, vertex.pos.x);
      span.top = std::min(span.top, vertex.pos.y);
      span.right = std::max(span.right, vertex.pos.x);
      span.bottom = std::max(span.bottom, vertex.pos.y);
    }
  }
  return span;
}

// Whether the span holds anything, as an empty one would pass any check of what's before what.
bool Holds(const PageSpan &span) { return span.right > span.left && span.bottom > span.top; }

// That the words a stage's matrix says sit on its backdrop in this order, on each backdrop, in the ink
// that reads on it, so a column or a row in the wrong place fails, and so do words that ignore the
// backdrop. Which backdrop was chosen is put back.
void CheckStageWords(StylePageRig &rig, std::initializer_list<const char *> lines) {
  const int chosen = rig.state.stage.backdrop;
  for (int backdrop = 0; backdrop < static_cast<int>(STAGE_BACKDROPS.size()); ++backdrop) {
    rig.state.stage.backdrop = backdrop;
    rig.nav.Frame([&](UiFrame &frame) {
      rig.Draw()(frame);
      CheckInOrder(*rig.list, frame, lines, DEFAULT_THEME.type.caption, StageInk(DEFAULT_THEME, rig.state.stage));
    });
  }
  rig.state.stage.backdrop = chosen;
  rig.Frames(1);
}

// At 4K the matrix shrinks to fit and the live toggle above it doesn't, measured against 1080p, where
// neither does. Worked out here apart from the code. It takes the stage's opacity, and a click where
// it's drawn.
void TestStageLiveCopy() {
  constexpr float EDGE_SLACK = 1.5f;
  constexpr int OUTPUT_4K = 2;
  constexpr float SCALE_4K = 2.0f;
  const auto &theme = DEFAULT_THEME;
  const auto &control = theme.colors.control;
  const ToggleTokens tokens = ToggleDefaults(theme);
  StageRig stage("toggle");
  auto &rig = stage.rig;
  const PageSpan panel = stage.Panel();
  const auto live_track = [&](float scale) { return SpanBetween(*rig.list, control.track_off, panel.top, panel.top + STAGE_INSET + tokens.height * scale + 1.0f); };
  const PageSpan full = live_track(1.0f);
  Check(rig.state.stage.fit == 1.0f && std::fabs(full.right - full.left - tokens.width) <= EDGE_SLACK, "at 1080p the live toggle is its own size");
  Check(std::fabs(full.left - panel.left - STAGE_INSET - 120.0f - theme.space.s3) <= EDGE_SLACK && std::fabs(full.top - panel.top - STAGE_INSET) <= EDGE_SLACK, "at the panel's top, past the room for its label");
  rig.state.stage.output = OUTPUT_4K;
  rig.Frames(2);
  const float fit = rig.state.stage.fit;
  Check(fit < 1.0f, "at 4K the matrix is too wide");
  const PageSpan big = live_track(SCALE_4K);
  Check(std::fabs(big.right - big.left - tokens.width * SCALE_4K) <= EDGE_SLACK, "the live toggle isn't shrunk");
  // The Disabled column's off toggle, which only a specimen draws.
  const PageSpan specimen = SpanOf(*rig.list, control.track_off.MulAlpha(theme.opacity.disabled), panel.top);
  Check(std::fabs(specimen.right - specimen.left - tokens.width * SCALE_4K * fit) <= EDGE_SLACK, "and the specimens are");
  Check(specimen.top > big.bottom, "and they sit below it");
  const PageSpan now = stage.Panel();
  Check(now.left == panel.left && now.top == panel.top && now.right == panel.right && now.bottom == panel.bottom, "and the page around them doesn't change");

  rig.state.stage.opacity = 0.5f;
  rig.Frames(2);
  const PageSpan dimmed = SpanBetween(*rig.list, control.track_off.MulAlpha(0.5f), panel.top, panel.top + STAGE_INSET + tokens.height * SCALE_4K + 1.0f);
  Check(dimmed.right > dimmed.left, "the live toggle takes the stage's opacity too");
  rig.state.stage.opacity = 1.0f;
  rig.Frames(2);

  ClickAt(rig, big.left + 8.0f, (big.top + big.bottom) / 2.0f);
  Check(rig.state.interactive.live_toggle, "a click where the live toggle is drawn turns it on");
}

// At 1080p, where a specimen's rectangle is where it's drawn, a click on a specimen does nothing. The
// pad reaches the live toggle below the tabs, and confirm changes it, and its focus never reaches a
// specimen. A page nothing has been clicked on, as a click leaves focus in the page.
void TestStageLiveCopyInput() {
  const auto &theme = DEFAULT_THEME;
  const auto &control = theme.colors.control;
  StageRig fresh("toggle");
  const PageSpan panel = fresh.Panel();
  const PageSpan row = SpanOf(*fresh.rig.list, control.track_off.MulAlpha(theme.opacity.disabled), panel.top);
  Check(Holds(row), "the Off row is drawn");
  // The Default column's toggle in the Off row, which is the first control across that row.
  ClickAt(fresh.rig, panel.left + STAGE_INSET + 120.0f + theme.space.s3 + 8.0f, (row.top + row.bottom) / 2.0f);
  Check(!fresh.rig.state.interactive.off && !fresh.rig.state.interactive.live_toggle, "and a click on a specimen changes nothing");
  fresh.rig.state.Reset();

  auto draw = fresh.rig.Draw();
  fresh.rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  fresh.rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  const ItemId tabs = FocusedControl();
  fresh.rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  const ItemId live = FocusedControl();
  Check(live != tabs, "the pad reaches the live toggle below the tabs");
  fresh.rig.nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(fresh.rig.state.interactive.live_toggle, "and confirm turns it on");
  fresh.rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  Check(FocusedControl() == live, "and no specimen below it takes focus");
}

// With no room left under the live copy nothing is drawn, so the strip mustn't go on saying the
// specimens were shrunk, as it did when there was room. A short screen is what takes the room away.
void TestStageWithNoRoomIsNotShrunk() {
  constexpr int OUTPUT_4K = 2;
  constexpr float SHORT_SCREEN = 500.0f;
  GalleryState state;
  NavHarness nav(ToastPageRig::Options());
  Rect screen = SCREEN;
  const auto draw = [&](UiFrame &frame) { DrawGallery(frame, state, Pages(), screen); };
  nav.Frames(2, draw);
  for (size_t index = 0; index < Pages().size(); ++index) {
    state.show = Pages()[index].id == "list" ? static_cast<int>(index) : state.show;
  }
  nav.Frames(2, draw);
  state.stage.output = OUTPUT_4K;
  nav.Frames(2, draw);
  Check(state.stage.fit < 1.0f, "with room the specimens are shrunk");
  screen.height = SHORT_SCREEN;
  nav.Frames(2, draw);
  Check(state.stage.fit == 1.0f, "and with none the figure doesn't stay behind");
}

// The Disabled column fades, in the rows from Primary to Danger, and comes after the posed ones, of
// which only the Focused column has the ring. The live row above is enabled and unposed, so these
// colours can only come from the matrix.
void TestButtonPage() {
  const auto &theme = DEFAULT_THEME;
  const auto &colors = theme.colors;
  const float faded = theme.opacity.disabled;
  StageRig stage("button");
  auto &rig = stage.rig;
  CheckStageWords(rig, {"Try it", "Default", "Hover", "Focused", "Pressed", "Disabled", "Primary", "Secondary", "Ghost", "Danger"});
  const PageSpan panel = stage.Panel();
  const PageSpan primary = SpanOf(*rig.list, colors.accent.fill.MulAlpha(faded), panel.top);
  const PageSpan danger = SpanOf(*rig.list, colors.status.danger_fill.MulAlpha(faded), panel.top);
  Check(Holds(primary) && Holds(danger), "a disabled button is faded");
  Check(primary.bottom < danger.top, "from the Primary row down to the Danger row");
  const PageSpan live = SpanOf(*rig.list, colors.accent.fill, panel.top);
  Check(std::fabs(live.top - panel.top - STAGE_INSET) <= 1.0f, "the live Primary is at the top");
  // Danger's fill, the first of it being the live one, as nothing outside the panel draws it.
  Check(SameClip(ClipOf(*rig.list, colors.status.danger_fill), {panel.left, panel.top, panel.right, panel.bottom}), "and the live row is held to the panel");
  // The keyline, as only a focus ring draws it, where the ring's own white is also the labels'.
  const PageSpan ring = SpanOf(*rig.list, colors.focus.keyline, panel.top);
  Check(Holds(ring) && ring.left > live.left + 170.0f && ring.right < primary.left, "and only a column between the first and the Disabled one has the focus ring");
}

// On the Bright backdrop a Ghost button's label is dark and its Hover column shades, in the matrix and
// the live row. On Mid, the baseline, neither happens. Secondary's label stays white either way, as it
// has its own surface. TestDialogSays covers a Ghost on a dialog's dark card.
void TestGhostOnBrightBackdrop() {
  const auto &theme = DEFAULT_THEME;
  const auto &colors = theme.colors;
  StageRig stage("button");
  auto &rig = stage.rig;
  const auto look = [&](bool dark_label, bool shade) {
    rig.nav.Frame([&](UiFrame &frame) {
      rig.Draw()(frame);
      Check(DrawsText(*rig.list, frame, "Ghost", theme.type.label, colors.text.on_light) == dark_label, "a Ghost's label is dark only on Bright");
      Check(DrawsText(*rig.list, frame, "Ghost", theme.type.label, colors.text.primary) != dark_label, "and white otherwise");
      Check(DrawsText(*rig.list, frame, "Secondary", theme.type.label, colors.text.primary), "while Secondary's stays white");
    });
    Check((PageVertices(*rig.list, colors.state.hover_on_light) > 0) == shade, "and its hover shades only on Bright");
  };
  look(false, false);
  const int light_on_mid = PageVertices(*rig.list, colors.state.hover);
  rig.state.stage.backdrop = STAGE_BACKDROP_BRIGHT;
  rig.Frames(1);
  look(true, true);
  // Secondary's hover is still the light one, so there's less of it, not none.
  Check(PageVertices(*rig.list, colors.state.hover) < light_on_mid, "in place of the light hover");
  // Past the "Try it" beside it, which is the same dark ink, and above the matrix.
  const PageSpan panel = stage.Panel();
  const PageSpan live_dark = SpanBetween(*rig.list, colors.text.on_light, panel.top, panel.top + STAGE_INSET + ButtonDefaults(theme).height_medium);
  Check(live_dark.right > panel.left + STAGE_INSET + 120.0f + theme.space.s3, "and the live Ghost's label is dark too");
}

// Off and on rows, each with a faded Disabled toggle after the Focused one's ring. Only a specimen
// draws these, as the live toggle is enabled.
void TestTogglePage() {
  const auto &theme = DEFAULT_THEME;
  const auto &colors = theme.colors;
  const float faded = theme.opacity.disabled;
  StageRig stage("toggle");
  auto &rig = stage.rig;
  CheckStageWords(rig, {"Try it", "Now off", "Default", "Hover", "Focused", "Pressed", "Disabled", "Off", "On"});
  rig.state.interactive.live_toggle = true;
  CheckStageWords(rig, {"Try it", "Now on", "Default"});
  rig.state.interactive.live_toggle = false;
  rig.Frames(2);
  const float top = stage.Panel().top;
  const PageSpan faded_off = SpanOf(*rig.list, colors.control.track_off.MulAlpha(faded), top);
  const PageSpan faded_on = SpanOf(*rig.list, colors.accent.fill.MulAlpha(faded), top);
  Check(Holds(faded_off) && Holds(faded_on), "a disabled toggle is faded in each row");
  Check(faded_off.bottom < faded_on.top, "off above on");
  const PageSpan ring = SpanOf(*rig.list, colors.focus.keyline, top);
  Check(Holds(ring) && ring.right < faded_off.left && ring.bottom >= faded_on.top, "and the Focused column shows its ring in both rows, before the Disabled one");
  rig.state.interactive.live_toggle = true;
  rig.state.Reset();
  Check(!rig.state.interactive.live_toggle, "a reset puts the live toggle back");
}

// Three values in each state, and one disabled column, which is grey, with nothing faded. The enabled
// columns before it have the accent fill. It's the slider's own value that the live copy changes, and
// a reset puts it back. On the dark backdrop, as Mid is the grey fill's own colour.
void TestSliderPage() {
  const auto &theme = DEFAULT_THEME;
  const auto &colors = theme.colors;
  StageRig stage("slider");
  auto &rig = stage.rig;
  CheckStageWords(rig, {"Try it", "Default", "Hover", "Focused", "Pressed", "Disabled", "0%", "50%", "100%"});
  // A value no specimen has, so only the live slider can say it. Not 0%, as that's also the end of 50%.
  constexpr float LIVE_VALUE = 0.37f;
  rig.state.interactive.live_slider = LIVE_VALUE;
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    Check(DrawsText(*rig.list, frame, "37%", theme.type.body, colors.text.secondary), "the live slider says its value");
    Check(DrawsText(*rig.list, frame, "50%", theme.type.body, colors.text.secondary), "and an enabled specimen says its own");
    for (const char *value : {"50%", "100%"}) {
      Check(DrawsText(*rig.list, frame, value, theme.type.body, colors.text.tertiary), "and the disabled column says its own, in the tertiary colour");
    }
  });
  rig.state.interactive.live_slider = 0.5f;
  rig.state.stage.backdrop = STAGE_DARK;
  rig.Frames(2);
  const PageSpan panel = stage.Panel();
  const PageSpan knob = SpanOf(*rig.list, colors.control.knob_disabled, panel.top);
  const PageSpan fill = SpanOf(*rig.list, colors.control.fill_disabled, panel.top);
  const PageSpan accent = SpanOf(*rig.list, colors.accent.line, panel.top);
  Check(knob.right > knob.left && fill.right > fill.left, "the disabled column has its grey knob and fill");
  Check(accent.right > accent.left && accent.right < knob.left && accent.right < fill.left, "and the enabled columns before it have the accent fill");
  const PageSpan ring = SpanOf(*rig.list, colors.focus.keyline, panel.top);
  Check(ring.right > ring.left && ring.right < knob.left, "and the focused column shows its ring before the disabled one");
  Check(PageVertices(*rig.list, colors.control.fill_disabled.MulAlpha(theme.opacity.disabled)) == 0 && PageVertices(*rig.list, colors.accent.line.MulAlpha(theme.opacity.disabled)) == 0, "and nothing is faded");
  const PageSpan live = SpanBetween(*rig.list, colors.accent.line, panel.top, panel.top + STAGE_INSET + 40.0f);
  ClickAt(rig, live.right + 100.0f, (live.top + live.bottom) / 2.0f);
  Check(rig.state.interactive.live_slider > 0.5f, "a click past the live slider's knob moves it up");
  const auto &values = rig.state.interactive.slider_values;
  Check(values[0] == 0.0f && values[1] == 0.5f && values[2] == 1.0f, "and no specimen's value changes");
  rig.state.Reset();
  Check(rig.state.interactive.live_slider == 0.5f, "a reset puts it back");
}

// Each kind of row as it is and disabled, with its words in their colours. The live list's selection
// follows a click.
void TestListPage() {
  const auto &theme = DEFAULT_THEME;
  const auto &colors = theme.colors;
  const float faded = theme.opacity.disabled;
  const float row = ListDefaults(theme).row_single;
  StageRig stage("list");
  auto &rig = stage.rig;
  CheckStageWords(rig, {"Try it", "Default", "Disabled", "Single", "Double", "Selected"});
  const PageSpan panel = stage.Panel();
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    for (const char *line : {"One line", "Two lines", "The current choice"}) {
      Check(DrawsText(*rig.list, frame, line, theme.type.body, colors.text.primary), line);
      Check(DrawsText(*rig.list, frame, line, theme.type.body, colors.text.primary.MulAlpha(faded)), line);
    }
    Check(DrawsText(*rig.list, frame, "With a smaller second", theme.type.body_small, colors.text.secondary), "the second line is its own colour");
    Check(DrawsText(*rig.list, frame, "With a smaller second", theme.type.body_small, colors.text.secondary.MulAlpha(faded)), "and it fades");
  });
  const PageSpan faded_tint = SpanOf(*rig.list, colors.accent.tint.MulAlpha(faded), panel.top);
  Check(Holds(faded_tint), "the disabled selected row keeps its selection, faded");
  // Below the live list, whose first row is selected too.
  const PageSpan matrix_tint = SpanBetween(*rig.list, colors.accent.tint, panel.top + STAGE_INSET * 2.0f + row, panel.bottom);
  Check(Holds(matrix_tint) && matrix_tint.right < faded_tint.left, "and the Default column's selected row is before it, whole");

  const float live_bottom = panel.top + STAGE_INSET * 2.0f + row;
  const PageSpan first = SpanBetween(*rig.list, colors.accent.tint, panel.top, live_bottom);
  Check(rig.state.interactive.live_row == 0 && Holds(first), "the live list starts on its first row");
  Check(faded_tint.top > first.bottom, "above the matrix");
  ClickAt(rig, first.left + 40.0f, (first.top + first.bottom) / 2.0f + row * 2.0f);
  Check(rig.state.interactive.live_row == 2, "a click chooses the row under it");
  const PageSpan chosen = SpanBetween(*rig.list, colors.accent.tint, panel.top, live_bottom + row * 2.0f);
  Check(std::fabs(chosen.top - first.top - row * 2.0f) <= 1.0f, "and the selection moves there");
  rig.state.Reset();
  Check(rig.state.interactive.live_row == 0, "a reset puts it back");
}

// On the Bright backdrop the matrix's rows and the live list's are dark, each in the words only it says.
// On Mid, the baseline, they're the usual colours. Faded rows follow, and a shading hover isn't posed
// here, as this page's matrix has no Hover column.
void TestRowsOnBrightBackdrop() {
  const auto &theme = DEFAULT_THEME;
  const auto &text = theme.colors.text;
  const float faded = theme.opacity.disabled;
  StageRig stage("list");
  auto &rig = stage.rig;
  const auto look = [&](Color first, Color second) {
    rig.nav.Frame([&](UiFrame &frame) {
      rig.Draw()(frame);
      Check(DrawsText(*rig.list, frame, "Row 1", theme.type.body, first), "the live list's rows are in the backdrop's colour");
      Check(DrawsText(*rig.list, frame, "One line", theme.type.body, first), "and the matrix's");
      Check(DrawsText(*rig.list, frame, "One line", theme.type.body, first.MulAlpha(faded)), "and its disabled ones, faded");
      Check(DrawsText(*rig.list, frame, "With a smaller second", theme.type.body_small, second), "and their second lines");
    });
  };
  look(text.primary, text.secondary);
  rig.state.stage.backdrop = STAGE_BACKDROP_BRIGHT;
  look(text.on_light, text.secondary_on_light);
}

// A live bar whose chosen tab follows a click, and the words that say why nothing is compared.
void TestTabsPage() {
  const auto &colors = DEFAULT_THEME.colors;
  StageRig stage("tabs");
  auto &rig = stage.rig;
  CheckStageWords(rig, {"Try it", "Tabs have no disabled state."});
  const PageSpan panel = stage.Panel();
  const PageSpan first = SpanOf(*rig.list, colors.accent.line, panel.top);
  Check(first.right > first.left && rig.state.interactive.live_tab == 0, "the first tab is underlined");
  ClickAt(rig, first.right + 30.0f, first.top - 20.0f);
  Check(rig.state.interactive.live_tab == 1, "a click chooses the second");
  // The underline slides, so it's measured once it has arrived.
  constexpr int FRAMES_TO_SETTLE = 60;
  rig.Frames(FRAMES_TO_SETTLE);
  const PageSpan second = SpanOf(*rig.list, colors.accent.line, panel.top);
  Check(second.left >= first.right, "and the underline moves there");
  auto draw = rig.Draw();
  rig.nav.Press(ImGuiKey_GamepadR1, draw);
  Check(rig.state.interactive.live_tab == 2, "and R1 moves on, as the page's rule says");
  rig.nav.Press(ImGuiKey_GamepadL1, draw);
  Check(rig.state.interactive.live_tab == 1, "and L1 back");
}

// ---- Pattern pages --------------------------------------------------------------------------------

// The views of the dialogs page, written out here apart from the page.
constexpr int DIALOG_CONFIRM = 1;
constexpr int DIALOG_PROGRESS = 2;
constexpr int DIALOG_SAVING = 3;
constexpr int DIALOG_ERROR = 4;
constexpr int DIALOG_LONG = 5;

// The draw list of the dialog on top, as a dialog is a window of its own.
const ImDrawList &DialogList() {
  const ImGuiWindow *window = ImGui::GetTopMostPopupModal();
  Check(window != nullptr, "a dialog is open");
  return *window->DrawList;
}

// The dialogs page with `view` open, settled.
struct DialogRig {
  explicit DialogRig(int view) : stage("dialogs") {
    stage.rig.state.dialogs_view = view;
    stage.rig.Frames(3);
  }
  void Press(ImGuiKey key) {
    auto draw = stage.rig.Draw();
    stage.rig.nav.Press(key, draw);
    stage.rig.Frames(2);
  }
  // Moves the work's clock on, as if it had been running this long.
  void Wait(int seconds) {
    stage.rig.state.dialogs.opened -= std::chrono::seconds(seconds);
    stage.rig.Frames(2);
  }
  StageRig stage;
};

// A dialog says its title in the heading style and its body, on the raised surface at a modal's width, with
// an enabled Cancel beside its Danger answer. That Cancel is a Ghost on a dark card, so it stays white on
// the bright backdrop, though a Ghost directly on that backdrop goes dark.
void TestDialogSays() {
  const auto &theme = DEFAULT_THEME;
  const auto &colors = theme.colors;
  DialogRig dialog(DIALOG_CONFIRM);
  auto &rig = dialog.stage.rig;
  const auto check_words = [&] {
    rig.nav.Frame([&](UiFrame &frame) {
      rig.Draw()(frame);
      const auto &list = DialogList();
      Check(TextAt(list, frame, "Delete this save?", theme.type.heading, colors.text.primary).has_value(), "the dialog has its title");
      Check(TextAt(list, frame, "This can't be undone.", theme.type.body, colors.text.secondary).has_value(), "and its words");
      Check(TextAt(list, frame, "Cancel", theme.type.label, colors.text.primary).has_value(), "and an enabled Cancel, white");
    });
  };
  check_words();
  const PageSpan card = SpanOf(DialogList(), colors.surface.raised);
  Check(Holds(card) && std::fabs(card.right - card.left - DialogDefaults(theme).width_medium) <= 1.5f, "on the raised surface, at a modal's width");
  Check(Holds(SpanOf(DialogList(), colors.status.danger_fill)), "and its Delete is the Danger button");
  rig.state.stage.backdrop = STAGE_BACKDROP_BRIGHT;
  rig.Frames(2);
  check_words();
}

// A dialog overlays just the panel: its scrim covers exactly that, its card is in the middle of it, at
// the output's true size, not shrunk, and it takes the stage's opacity. A dialog at 1080p is the
// baseline for the size at 4K. Worked out here apart from the code.
void TestDialogOverlaysThePanel() {
  constexpr int OUTPUT_4K = 2;
  constexpr float SLACK = 1.5f;
  const auto &colors = DEFAULT_THEME.colors;
  const float medium = DialogDefaults(DEFAULT_THEME).width_medium;
  DialogRig dialog(DIALOG_CONFIRM);
  auto &rig = dialog.stage.rig;
  const PageSpan panel = dialog.stage.Panel();
  const auto check_card = [&](float width) {
    const PageSpan scrim = SpanOf(DialogList(), colors.surface.scrim);
    Check(std::fabs(scrim.left - panel.left) <= SLACK && std::fabs(scrim.top - panel.top) <= SLACK && std::fabs(scrim.right - panel.right) <= SLACK && std::fabs(scrim.bottom - panel.bottom) <= SLACK, "the scrim covers the panel and no more");
    const PageSpan card = SpanOf(DialogList(), colors.surface.raised);
    Check(std::fabs(card.right - card.left - width) <= SLACK, "the card is its true size at the output");
    Check(std::fabs((card.left + card.right) / 2.0f - (panel.left + panel.right) / 2.0f) <= SLACK && std::fabs((card.top + card.bottom) / 2.0f - (panel.top + panel.bottom) / 2.0f) <= SLACK, "in the middle of the panel");
  };
  check_card(medium);
  Check(PageVertices(*rig.list, colors.surface.scrim) == 0, "the page beneath has no scrim of its own");
  rig.state.stage.output = OUTPUT_4K;
  rig.Frames(2);
  check_card(medium * 2.0f);
  rig.state.stage.opacity = 0.5f;
  rig.Frames(2);
  Check(Holds(SpanOf(DialogList(), colors.surface.scrim.MulAlpha(0.5f))), "the dialog takes the stage's opacity");
  rig.state.stage.opacity = 1.0f;
  rig.Frames(2);
  // A click on what's drawn at 4K lands, as a dialog is at true size, with no shrinking to throw it off.
  const PageSpan danger = SpanOf(DialogList(), colors.status.danger_fill);
  Check(Holds(danger), "the card has its Delete");
  ClickAt(rig, (danger.left + danger.right) / 2.0f, (danger.top + danger.bottom) / 2.0f);
  Check(rig.state.dialogs.answer == "Deleted", "and a click where Delete is drawn answers it");
}

// The pad opens a dialog from its button. Focus starts on Cancel, and Back cancels too, with focus going
// back to the button that opened it. Delete takes a move to the right, the baseline, which shows the
// press answers for whichever has focus. R1 does nothing under a dialog and changes page without one.
void TestConfirmDialog() {
  StageRig stage("dialogs");
  auto &rig = stage.rig;
  auto draw = rig.Draw();
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  const ItemId opener = FocusedControl();
  rig.nav.Press(ImGuiKey_GamepadFaceDown, draw);
  rig.Frames(3);
  Check(rig.state.dialogs_view == DIALOG_CONFIRM, "confirm on the first button opens the first dialog");
  const int page = rig.state.page;
  rig.nav.Press(ImGuiKey_GamepadR1, draw);
  rig.Frames(2);
  Check(rig.state.dialogs_view == DIALOG_CONFIRM && rig.state.page == page, "R1 does nothing under it");
  rig.nav.Press(ImGuiKey_GamepadFaceRight, draw);
  rig.Frames(3);
  Check(rig.state.dialogs_view == 0 && rig.state.dialogs.answer == "Cancelled", "back cancels");
  Check(FocusedControl() == opener, "and focus returns to the button");
  rig.nav.Press(ImGuiKey_GamepadR1, draw);
  rig.Frames(2);
  Check(rig.state.page != page, "where without a dialog R1 changes page");
  DialogRig on_cancel(DIALOG_CONFIRM);
  on_cancel.Press(ImGuiKey_GamepadFaceDown);
  Check(on_cancel.stage.rig.state.dialogs.answer == "Cancelled" && on_cancel.stage.rig.state.dialogs_view == 0, "confirm on the first focus cancels");
  DialogRig on_delete(DIALOG_CONFIRM);
  on_delete.Press(ImGuiKey_GamepadDpadRight);
  on_delete.Press(ImGuiKey_GamepadFaceDown);
  Check(on_delete.stage.rig.state.dialogs.answer == "Deleted", "and a move to the right then confirm deletes");
}

// Cancelling work asks first, by the button and by Back, and the question takes Back and the first focus
// back to the work, which is still going. Only the Danger answer stops it. The work finishing closes both.
void TestProgressAsksBeforeStopping() {
  const auto question = [] { return ImGui::GetTopMostPopupModal()->Name; };
  DialogRig dialog(DIALOG_PROGRESS);
  auto &state = dialog.stage.rig.state;
  const std::string work = question();
  dialog.Press(ImGuiKey_GamepadFaceDown);
  Check(state.dialogs_view == DIALOG_PROGRESS && state.dialogs.asking_to_stop && question() != work, "Cancel asks in a dialog of its own");
  dialog.Press(ImGuiKey_GamepadFaceDown);
  Check(state.dialogs_view == DIALOG_PROGRESS && !state.dialogs.asking_to_stop, "and the first focus keeps going");
  dialog.stage.rig.Frames(2);
  Check(question() == work, "back in the work's dialog");
  dialog.Press(ImGuiKey_GamepadFaceRight);
  Check(state.dialogs.asking_to_stop && state.dialogs_view == DIALOG_PROGRESS, "back asks too");
  dialog.Press(ImGuiKey_GamepadFaceRight);
  Check(!state.dialogs.asking_to_stop && state.dialogs_view == DIALOG_PROGRESS, "and back again keeps going");
  dialog.stage.rig.Frames(2);
  dialog.Press(ImGuiKey_GamepadFaceDown);
  dialog.Press(ImGuiKey_GamepadDpadRight);
  dialog.Press(ImGuiKey_GamepadFaceDown);
  Check(state.dialogs_view == 0 && state.dialogs.answer == "Cancelled", "cancelling the install stops it");
  Check(ImGui::GetTopMostPopupModal() == nullptr, "and closes both dialogs");
  DialogRig finishing(DIALOG_PROGRESS);
  finishing.Press(ImGuiKey_GamepadFaceDown);
  Check(finishing.stage.rig.state.dialogs.asking_to_stop, "while it asks");
  finishing.Wait(60);
  Check(finishing.stage.rig.state.dialogs_view == 0 && finishing.stage.rig.state.dialogs.answer == "Finished" && ImGui::GetTopMostPopupModal() == nullptr, "work that finishes closes both");
  finishing.stage.rig.state.dialogs_view = DIALOG_PROGRESS;
  finishing.stage.rig.Frames(3);
  Check(!finishing.stage.rig.state.dialogs.asking_to_stop && std::string(ImGui::GetTopMostPopupModal()->Name).find("ask") == std::string::npos, "and opening it again doesn't ask");
}

// The bar shows how far along the work is, half way at half its time and a quarter at a quarter, of a
// dialog's content width at 1080p. Worked out here apart from the code.
void TestProgressBarFollowsTheWork() {
  constexpr float INSTALL_SECONDS = 8.0f;
  constexpr float PADDING = 32.0f;
  constexpr float SLACK = 16.0f;
  const auto &colors = DEFAULT_THEME.colors;
  const float track = DialogDefaults(DEFAULT_THEME).width_medium - 2.0f * PADDING;
  for (const int seconds : {2, 4}) {
    DialogRig dialog(DIALOG_PROGRESS);
    dialog.Wait(seconds);
    const PageSpan fill = SpanOf(DialogList(), colors.accent.line);
    Check(std::fabs(fill.right - fill.left - track * static_cast<float>(seconds) / INSTALL_SECONDS) <= SLACK, "the bar is as far along as the work");
  }
}

// In the frame a dialog closes in, ImGui still draws it, so it must show nothing of itself, as its card
// would otherwise flash at the size of the panel. The dialog's own frames are the baseline, which do draw
// both, and it's a frame where the list is drawn afresh, as a dialog left open would keep what it had.
void TestClosingDialogDrawsNothing() {
  const auto &colors = DEFAULT_THEME.colors;
  DialogRig dialog(DIALOG_CONFIRM);
  auto &rig = dialog.stage.rig;
  const ImDrawList &list = DialogList();
  Check(Holds(SpanOf(list, colors.surface.scrim)) && Holds(SpanOf(list, colors.surface.raised)), "an open dialog draws its scrim and card");
  rig.state.dialogs_view = DIALOG_ERROR;
  rig.Frames(1);
  Check(!Holds(SpanOf(list, colors.surface.scrim)) && !Holds(SpanOf(list, colors.surface.raised)), "and the one closing as another opens draws neither");
  const ImDrawList &error = DialogList();
  rig.Frames(2);
  rig.state.dialogs_view = 0;
  rig.Frames(1);
  Check(!Holds(SpanOf(error, colors.surface.scrim)) && !Holds(SpanOf(error, colors.surface.raised)), "nor does one closing with nothing to follow");
  DialogRig work(DIALOG_PROGRESS);
  work.Press(ImGuiKey_GamepadFaceDown);
  const ImDrawList &question = DialogList();
  Check(Holds(SpanOf(question, colors.surface.raised)), "the question draws its card while it asks");
  work.stage.rig.state.dialogs.asking_to_stop = false;
  work.stage.rig.Frames(1);
  Check(!Holds(SpanOf(question, colors.surface.raised)), "and nothing in the frame it closes in");
}

// Work that can't be stopped has no Cancel and ignores Back and confirm, where the cancelable dialog is
// the baseline in TestProgressAsksBeforeStopping, and it closes itself when done.
void TestSavingCannotBeStopped() {
  const auto &theme = DEFAULT_THEME;
  DialogRig dialog(DIALOG_SAVING);
  auto &rig = dialog.stage.rig;
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    Check(TextAt(DialogList(), frame, "Saving", theme.type.heading, theme.colors.text.primary).has_value(), "the dialog says what it's doing");
    Check(!TextAt(DialogList(), frame, "Cancel", theme.type.label, theme.colors.text.primary).has_value(), "with no Cancel");
  });
  const int page = rig.state.page;
  dialog.Press(ImGuiKey_GamepadFaceRight);
  dialog.Press(ImGuiKey_GamepadFaceDown);
  Check(rig.state.dialogs_view == DIALOG_SAVING && !rig.state.dialogs.asking_to_stop, "back and confirm do nothing");
  Check(rig.state.page == page, "and back doesn't reach the page beneath");
  dialog.Wait(60);
  Check(rig.state.dialogs_view == 0 && rig.state.dialogs.answer == "Finished", "and it closes itself when done");
}

// Work that closes itself gives focus back to the button that opened it, as an answer does.
void TestFinishedWorkReturnsFocus() {
  StageRig stage("dialogs");
  auto &rig = stage.rig;
  auto draw = rig.Draw();
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  const ItemId opener = FocusedControl();
  rig.nav.Press(ImGuiKey_GamepadFaceDown, draw);
  rig.Frames(3);
  Check(rig.state.dialogs_view == DIALOG_PROGRESS, "the second button opens the work");
  Check(FocusedControl() != opener, "and focus goes into it");
  rig.state.dialogs.opened -= std::chrono::seconds(60);
  rig.Frames(4);
  Check(rig.state.dialogs_view == 0 && FocusedControl() == opener, "finishing gives focus back to the button");
}

// A screenshot run resets the state with a dialog open and goes on to another page, which never draws the
// dialog to close it. That page must still take the pad, as a stranded modal would block it. A dialog
// open on its own page is the baseline, where the same press doesn't reach the sidebar.
void TestStrandedDialogDoesntBlockThePage() {
  StageRig blocked("dialogs");
  auto &blocked_rig = blocked.rig;
  auto blocked_draw = blocked_rig.Draw();
  blocked_rig.state.dialogs_view = DIALOG_CONFIRM;
  blocked_rig.Frames(3);
  const int blocked_page = blocked_rig.state.page;
  blocked_rig.nav.Press(ImGuiKey_GamepadDpadDown, blocked_draw);
  blocked_rig.Frames(2);
  Check(blocked_rig.state.page == blocked_page && blocked_rig.state.dialogs_view == DIALOG_CONFIRM, "with a dialog open, down doesn't reach the sidebar");
  DialogRig dialog(DIALOG_LONG);
  auto &rig = dialog.stage.rig;
  auto draw = rig.Draw();
  rig.state.Reset();
  size_t tabs = 0;
  for (size_t index = 0; index < Pages().size(); ++index) {
    tabs = Pages()[index].id == "tabs" ? index : tabs;
  }
  rig.state.show = static_cast<int>(tabs);
  rig.Frames(8);
  Check(rig.state.page == static_cast<int>(tabs), "another page shows");
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  rig.Frames(2);
  Check(rig.state.page == static_cast<int>(tabs) + 1, "and the pad still moves down the sidebar");
}

// The error's detail sits in its quiet colour, focus starts on Retry, and Close is a move left.
void TestErrorDialog() {
  const auto &theme = DEFAULT_THEME;
  DialogRig dialog(DIALOG_ERROR);
  auto &rig = dialog.stage.rig;
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    Check(TextAt(DialogList(), frame, "Error code CE-10005-6", theme.type.caption, theme.colors.text.tertiary).has_value(), "the detail is small and quiet");
  });
  dialog.Press(ImGuiKey_GamepadFaceDown);
  Check(rig.state.dialogs.answer == "Retry" && rig.state.dialogs_view == 0, "focus starts on Retry");
  DialogRig closing(DIALOG_ERROR);
  closing.Press(ImGuiKey_GamepadDpadLeft);
  closing.Press(ImGuiKey_GamepadFaceDown);
  Check(closing.stage.rig.state.dialogs.answer == "Closed", "and Close is a move left");
}

// More text than fits stops at the panel, with its button still on the card, where a short dialog's card
// is the baseline that fits its text. Worked out here from the panel apart from the code.
void TestLongDialogGivesWay() {
  constexpr float MARGIN = 16.0f;
  const auto &colors = DEFAULT_THEME.colors;
  DialogRig dialog(DIALOG_LONG);
  const PageSpan panel = dialog.stage.Panel();
  const PageSpan card = SpanOf(DialogList(), colors.surface.raised);
  Check(std::fabs((card.bottom - card.top) - (panel.bottom - panel.top - 2.0f * MARGIN)) <= 1.5f, "a long body stops the card at the panel's margin");
  // The focused Primary's fill, as focus starts on it, once its highlight has settled.
  constexpr int FRAMES_TO_SETTLE = 60;
  dialog.stage.rig.Frames(FRAMES_TO_SETTLE);
  const PageSpan button = SpanOf(DialogList(), colors.accent.fill_hover);
  Check(Holds(button) && button.bottom <= card.bottom, "with its button still on the card");
  DialogRig short_one(DIALOG_CONFIRM);
  const PageSpan short_card = SpanOf(DialogList(), colors.surface.raised);
  Check(short_card.bottom - short_card.top < (card.bottom - card.top) / 2.0f, "where a short dialog's card fits its text");
}

// The thumb of the dialog on top, in the tertiary colour only it draws in this dialog.
PageSpan DialogThumb() { return SpanOf(DialogList(), DEFAULT_THEME.colors.text.tertiary); }

// A long body scrolls by the D-pad, repeating no further than its end, and back, with focus staying on the
// button, and by the wheel. A short dialog is the baseline, with no thumb and nothing to move. The buttons
// stay where they are throughout, so an answer can always be reached.
void TestLongDialogScrolls() {
  constexpr float SLACK = 1.5f;
  constexpr int FRAMES_TO_SETTLE = 60;
  const auto &colors = DEFAULT_THEME.colors;
  DialogRig short_one(DIALOG_CONFIRM);
  Check(!Holds(DialogThumb()), "a short dialog has no thumb");
  DialogRig dialog(DIALOG_LONG);
  auto &rig = dialog.stage.rig;
  auto draw = rig.Draw();
  rig.Frames(FRAMES_TO_SETTLE);
  const ItemId close = FocusedControl();
  const PageSpan start = DialogThumb();
  const PageSpan button = SpanOf(DialogList(), colors.accent.fill_hover);
  Check(Holds(start) && Holds(button), "a long one has a thumb and its button");
  dialog.Press(ImGuiKey_GamepadDpadDown);
  const PageSpan after = DialogThumb();
  Check(after.top > start.top + SLACK, "down scrolls it");
  Check(FocusedControl() == close, "with focus staying on the button");
  const PageSpan moved = SpanOf(DialogList(), colors.accent.fill_hover);
  Check(std::fabs(moved.top - button.top) <= SLACK, "and the button staying put");
  dialog.Press(ImGuiKey_GamepadDpadUp);
  Check(std::fabs(DialogThumb().top - start.top) <= SLACK, "up scrolls it back");
  rig.nav.MoveMouse((button.left + button.right) / 2.0f, start.top + 40.0f, draw);
  rig.nav.Frame(draw);
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -2.0f);
  rig.Frames(4);
  Check(DialogThumb().top > start.top + SLACK, "and the wheel scrolls it");
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -1000.0f);
  rig.Frames(4);
  const float bottom = DialogThumb().bottom;
  Check(bottom <= button.top, "stopping at its end, above the button");
  dialog.Press(ImGuiKey_GamepadDpadDown);
  Check(std::fabs(DialogThumb().bottom - bottom) <= SLACK, "where down goes no further");
  rig.Frames(2);
  rig.nav.Press(ImGuiKey_GamepadFaceDown, draw);
  rig.Frames(3);
  Check(rig.state.dialogs_view == 0 && rig.state.dialogs.answer == "Closed", "and confirm still answers it");
  rig.state.dialogs_view = DIALOG_LONG;
  rig.Frames(3);
  Check(std::fabs(DialogThumb().top - start.top) <= SLACK, "and opening it again starts at the top, not where it was left");
}

// The page says what each dialog is for and the last answer. Changing the view closes one dialog and opens
// another, and no view closes it. A reset puts the page's state back.
void TestDialogsPageViews() {
  StageRig stage("dialogs");
  auto &rig = stage.rig;
  CheckStageWords(rig, {"Try it", "Confirm delete: Danger asks first, with focus on Cancel.", "Long text: A long body scrolls, and the buttons stay."});
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    Check(DrawsText(*rig.list, frame, "Last answer: None yet", DEFAULT_THEME.type.caption, StageInk(DEFAULT_THEME, rig.state.stage)), "and says no dialog has been answered");
  });
  rig.state.dialogs_view = DIALOG_CONFIRM;
  rig.Frames(3);
  const std::string first = ImGui::GetTopMostPopupModal()->Name;
  rig.state.dialogs_view = DIALOG_ERROR;
  rig.Frames(3);
  Check(first != ImGui::GetTopMostPopupModal()->Name, "another view opens its own dialog");
  rig.state.dialogs_view = 0;
  rig.Frames(3);
  Check(ImGui::GetTopMostPopupModal() == nullptr, "and none closes it");
  rig.state.dialogs_view = DIALOG_ERROR;
  rig.state.dialogs.answer = "Retry";
  rig.state.Reset();
  Check(rig.state.dialogs_view == 0 && rig.state.dialogs.answer == "None yet", "a reset puts them back");
}

// The vertices the page draws in the card's own colour, whatever its alpha, inside a column of the page.
struct CardSeen {
  bool seen = false;
  int alpha = 0;
  float top = 0.0f;
};
CardSeen CardIn(const ImDrawList &list, float left, float right, float below, float above) {
  const ImU32 rgb = ToImU32(DEFAULT_THEME.colors.surface.raised) & ~IM_COL32_A_MASK;
  CardSeen card;
  card.top = SCREEN.Bottom();
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    const bool inside = vertex.pos.x >= left && vertex.pos.x <= right && vertex.pos.y >= below && vertex.pos.y <= above;
    if (inside && (vertex.col & ~IM_COL32_A_MASK) == rgb && (vertex.col >> IM_COL32_A_SHIFT) != 0) {
      card.seen = true;
      card.alpha = std::max(card.alpha, static_cast<int>(vertex.col >> IM_COL32_A_SHIFT));
      card.top = std::min(card.top, vertex.pos.y);
    }
  }
  return card;
}

// Where the Notification page's frozen frames start across, past the room for their row's name.
constexpr float NOTIFICATION_LEFT = 368.0f;
constexpr float FIRST_CELL = NOTIFICATION_LEFT + 120.0f + 12.0f;

// The bottom of the Notification page's kinds of toast, the only cards left of its first frozen frame.
float NotificationKindsBottom(const ImDrawList &list) {
  float bottom = 0.0f;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if (vertex.pos.x >= NOTIFICATION_LEFT && vertex.pos.x < FIRST_CELL && vertex.col == ToImU32(DEFAULT_THEME.colors.surface.raised)) {
      bottom = std::max(bottom, vertex.pos.y);
    }
  }
  return bottom;
}

// Each frozen frame is the toast's real pose at its moment, from the same timeline the view plays, worked
// out from it here. The exit's moments are from where its hold ends, by the tokens. Frames don't move as
// the clock runs, unlike the live toast. The kinds of toast are above the rows, and left of the first
// frame, which tells where the rows start. The resting frame is the baseline for how far one has slid.
void TestNotificationFrames() {
  constexpr float SECONDS = 4.0f;
  constexpr float CELL_PITCH = 280.0f + 12.0f;
  constexpr float CELL_WIDTH = 280.0f;
  constexpr float SLACK = 1.5f;
  const auto &theme = DEFAULT_THEME;
  const ToastTokens tokens = ToastDefaults(theme);
  const Timeline timeline = ToastTimeline(tokens, SECONDS);
  StageRig stage("notification");
  auto &rig = stage.rig;
  Check(rig.state.stage.fit == 1.0f, "the page fits at 1080p");
  CheckStageWords(rig, {"Enter", "100 ms", "180 ms", "280 ms", "Exit", "40 ms", "120 ms"});
  const float kinds_bottom = NotificationKindsBottom(*rig.list);
  Check(kinds_bottom > 0.0f, "the kinds of toast are above the rows");
  // A row's height, its time's caption and gap, the slide's room, the card, then the gap to the next.
  constexpr float row_height = 194.0f;
  const float enter_top = kinds_bottom + 1.0f;
  const float exit_top = enter_top + row_height;
  const float exit_start = SECONDS * 1000.0f - std::max(tokens.exit_fade_ms, tokens.exit_move_ms);
  const auto pose_at = [&](float ms) {
    const UiClock::time_point start{};
    return Evaluate(timeline, start, std::nullopt, start + std::chrono::duration_cast<UiClock::duration>(std::chrono::duration<float, std::milli>(ms)));
  };
  const auto card_at = [&](size_t cell, float top) {
    const float left = FIRST_CELL + CELL_PITCH * static_cast<float>(cell);
    return CardIn(*rig.list, left, left + CELL_WIDTH, top, top + row_height - 1.0f);
  };
  const auto check_row = [&](float top, float from, const std::array<float, 4> &moments) {
    for (size_t cell = 0; cell < moments.size(); ++cell) {
      const CardSeen card = card_at(cell, top);
      const Pose pose = pose_at(from + moments[cell]);
      const int wanted = static_cast<int>(std::lround(static_cast<float>(ToImU32(theme.colors.surface.raised) >> IM_COL32_A_SHIFT) * pose.alpha));
      Check(card.seen == (wanted > 0), "a frame shows a card exactly when its pose can be seen");
      if (card.seen) {
        Check(std::abs(card.alpha - wanted) <= 1, "each frame's card has its pose's alpha");
      }
    }
  };
  check_row(enter_top, 0.0f, {0.0f, 100.0f, 180.0f, 280.0f});
  check_row(exit_top, exit_start, {0.0f, 40.0f, 80.0f, 120.0f});
  const CardSeen sliding = card_at(1, enter_top);
  const CardSeen resting = card_at(3, enter_top);
  Check(sliding.seen && resting.seen, "the slide's frames are drawn");
  Check(std::fabs(sliding.top - resting.top - pose_at(100.0f).offset.y) <= SLACK && pose_at(100.0f).offset.y < -1.0f, "and one is as far up its slide as its pose says");
  Check(std::fabs(card_at(0, exit_top).top - resting.top - row_height) <= SLACK, "and the exit starts from where the toast rests");
  rig.Frames(60);
  const CardSeen later = card_at(1, enter_top);
  Check(later.alpha == sliding.alpha && later.top == sliding.top, "and it stays frozen as the clock runs");
}

// Each status toast has its own bar, thin, with its title beside it, and the trophy's toast is there too
// with none. The frozen frames below use one of the bar colours, so this looks only at the kinds. The
// sample icon loads here even if the Toast page was never shown, as pictures would otherwise depend on
// the order of the pages.
void TestNotificationKinds() {
  const auto &theme = DEFAULT_THEME;
  const auto &status = theme.colors.status;
  StageRig stage("notification");
  auto &rig = stage.rig;
  const float top = stage.Panel().top;
  const float bottom = NotificationKindsBottom(*rig.list);
  const float bar_width = ToastDefaults(theme).edge_bar;
  float last_left = 0.0f;
  float last_top = 0.0f;
  float last_bottom = 0.0f;
  for (const Color colour : {status.success, status.warning, status.error, status.info}) {
    const PageSpan bar = SpanBetween(*rig.list, colour, top, bottom);
    Check(Holds(bar) && bar.right - bar.left <= bar_width + 1.0f, "each status toast has its own thin bar");
    Check(bar.top > last_top || bar.left > last_left, "in its own place");
    last_left = bar.left;
    last_top = bar.top;
    last_bottom = bar.bottom;
  }
  // The trophy's toast is the row below them, with no bar of any status colour.
  for (const Color colour : {status.success, status.warning, status.error, status.info}) {
    Check(!Holds(SpanBetween(*rig.list, colour, last_bottom + 1.0f, bottom)), "and the trophy's toast has none");
  }
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    for (const char *name : {"Download complete", "Couldn't connect", "Update available", "Master Alchemist"}) {
      Check(DrawsText(*rig.list, frame, name, theme.type.heading, theme.colors.text.primary), name);
    }
  });
  ToastPageRig icons;
  icons.state.toast.icon.reset();
  icons.state.toast.icon_tried = false;
  for (size_t page = 0; page < Pages().size(); ++page) {
    icons.state.show = Pages()[page].id == "notification" ? static_cast<int>(page) : icons.state.show;
  }
  icons.Frames(2);
  Check(icons.state.toast.icon.has_value(), "the notification page loads the sample icon for itself");
}

// Says it's a mock, on every backdrop. Each line has its word in its own colour, Bad's being the status
// colour and not the chart's. The panel's padding is the HUD tokens', written out here apart from them.
// The graph's bars differ in height and colour, the chart's red only ever being a bar, and they take the
// stage's opacity, as they're painted by hand. The tile has its caption and value.
void TestMetricPage() {
  constexpr float PADDING_X = 12.0f;
  constexpr float PADDING_Y = 8.0f;
  constexpr float SLACK = 1.0f;
  const auto &theme = DEFAULT_THEME;
  const auto &colors = theme.colors;
  StageRig stage("metric");
  auto &rig = stage.rig;
  CheckStageWords(rig, {"Mock, as there's no HUD component yet. Colour never stands alone, as each line carries a word."});
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    Check(DrawsText(*rig.list, frame, "60.0 FPS  Good", theme.type.dense, colors.chart.good), "good has its word in its colour");
    Check(DrawsText(*rig.list, frame, "42.0 FPS  Slow", theme.type.dense, colors.chart.medium), "and medium");
    Check(DrawsText(*rig.list, frame, "18.0 FPS  Bad", theme.type.dense, colors.status.error), "and bad, in the status colour");
    Check(DrawsText(*rig.list, frame, "Frame time", theme.type.caption, colors.text.secondary), "the tile has its caption");
    Check(DrawsText(*rig.list, frame, "16.7 ms", theme.type.heading, colors.text.primary), "and its value");
  });
  const float top = stage.Panel().top;
  const PageSpan panel = SpanOf(*rig.list, colors.surface.hud, top);
  const PageSpan graph = SpanOf(*rig.list, colors.surface.inset, top);
  Check(Holds(panel) && Holds(graph), "the panel holds a graph");
  Check(std::fabs(graph.left - panel.left - PADDING_X) <= SLACK && std::fabs(graph.top - panel.top - PADDING_Y) <= SLACK && std::fabs(panel.right - graph.right - PADDING_X) <= SLACK, "inside the HUD's padding");
  const PageSpan good = SpanBetween(*rig.list, colors.chart.good, graph.top, graph.bottom);
  const PageSpan medium = SpanBetween(*rig.list, colors.chart.medium, graph.top, graph.bottom);
  const PageSpan bad = SpanBetween(*rig.list, colors.chart.bad, graph.top, graph.bottom);
  Check(Holds(good) && Holds(medium) && Holds(bad), "the graph has bars of each colour");
  Check(bad.top < medium.top && medium.top < good.top, "and the worse a frame, the taller its bar");
  Check(std::fabs(good.bottom - graph.bottom) <= SLACK && std::fabs(bad.bottom - graph.bottom) <= SLACK, "standing on the graph's floor");
  Check(SpanBetween(*rig.list, colors.chart.bad, graph.bottom + 1.0f, SCREEN.Bottom()).right < graph.left, "and the chart's red is only ever a bar");
  Check(Holds(SpanOf(*rig.list, colors.surface.raised, top)), "the tile is a card");
  rig.state.stage.opacity = 0.5f;
  rig.Frames(2);
  Check(Holds(SpanOf(*rig.list, colors.chart.bad.MulAlpha(0.5f), top)) && !Holds(SpanOf(*rig.list, colors.chart.bad, top)), "the bars take the stage's opacity");
  Check(Holds(SpanOf(*rig.list, colors.surface.inset.MulAlpha(0.5f), top)), "and so does the graph's ground");
}

// The sidebar's rows run past the screen, so the wheel scrolls it to reach the last page and click it. The
// same click with no wheel is the baseline, landing on an earlier row, as the sidebar hasn't moved. Clicking
// a row moves focus there, so the sidebar stays with it.
void TestSidebarScrollsToItsLastPage() {
  constexpr float NEAR_BOTTOM = 30.0f;
  const int last = static_cast<int>(Pages().size()) - 1;
  const auto click_near_bottom = [&](bool wheel) {
    StylePageRig rig("colour");
    auto draw = rig.Draw();
    rig.nav.MoveMouse(100.0f, 300.0f, draw);
    rig.nav.Frame(draw);
    if (wheel) {
      ImGui::GetIO().AddMouseWheelEvent(0.0f, -100.0f);
      rig.nav.Frames(4, draw);
    }
    rig.nav.MoveMouse(100.0f, SCREEN.Bottom() - NEAR_BOTTOM, draw);
    rig.nav.Frame(draw);
    rig.nav.MouseButton(true);
    rig.nav.Frame(draw);
    rig.nav.MouseButton(false);
    rig.nav.Frames(3, draw);
    return rig.state.page;
  };
  Check(click_near_bottom(false) < last, "without the wheel the last row is out of reach");
  Check(click_near_bottom(true) == last, "and the wheel brings it into reach of a click");
}

// Whether the sidebar draws `text`, by its glyphs in order, as TextAt does for the page.
bool SidebarDraws(const ImDrawList &page, UiFrame &frame, std::string_view text, const TypeStyle &style, Color colour) {
  const TextProps props{text, style, colour};
  ImDrawList alone(ImGui::GetDrawListSharedData());
  alone._ResetForNewFrame();
  alone.PushClipRectFullScreen();
  const float unlimited = std::numeric_limits<float>::infinity();
  DrawText(alone, LayoutText(frame.Ui(), props, unlimited), props, {PANE_LEFT + 1.0f, 0.0f, SCREEN.width, SCREEN.height});
  const auto wanted = GlyphUvs(alone, colour);
  std::vector<std::pair<float, float>> drawn;
  for (const ImDrawVert &vertex : page.VtxBuffer) {
    if (vertex.pos.x < PANE_LEFT && vertex.col == ToImU32(colour)) {
      drawn.push_back({vertex.uv.x, vertex.uv.y});
    }
  }
  return !wanted.empty() && std::search(drawn.begin(), drawn.end(), wanted.begin(), wanted.end()) != drawn.end();
}

// With focus in a page, the wheel can take the sidebar's row for that page out of view. Back sends focus
// straight to that row, so the sidebar goes back to it, as confirm shouldn't act on a row out of view. A
// row out of view isn't drawn, so its word shows whether it's in view, with the same word gone after the
// wheel as the baseline. The focus ring wouldn't do, as the pointer having moved hides it.
void TestBackShowsTheSidebarsRow() {
  constexpr int FRAMES_TO_SETTLE = 30;
  const auto &theme = DEFAULT_THEME;
  const auto &list_word = theme.colors.text.primary;
  StylePageRig rig("colour");
  auto draw = rig.Draw();
  const auto row_drawn = [&] {
    bool drawn = false;
    rig.nav.Frame([&](UiFrame &frame) {
      rig.Draw()(frame);
      drawn = SidebarDraws(*rig.list, frame, "Colour", theme.type.body, list_word);
    });
    return drawn;
  };
  rig.Frames(FRAMES_TO_SETTLE);
  Check(row_drawn(), "the page's row starts in view");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  rig.Frames(FRAMES_TO_SETTLE);
  rig.nav.MoveMouse(100.0f, 300.0f, draw);
  rig.nav.Frame(draw);
  ImGui::GetIO().AddMouseWheelEvent(0.0f, -100.0f);
  rig.Frames(4);
  Check(!row_drawn(), "the wheel takes it out of view, with focus in the page");
  rig.nav.Press(ImGuiKey_GamepadFaceRight, draw);
  rig.Frames(FRAMES_TO_SETTLE);
  Check(row_drawn(), "and back shows it again");
}

// Every page says how to use what it shows, under its title. Written out here for the rules that carry a
// decision, as those are what a change could quietly lose, and a page drawing its rule is what's checked.
void TestPagesSayTheirRules() {
  struct PageRule {
    std::string_view page;
    const char *rule;
  };
  constexpr std::array RULES{
      PageRule{"toast", "Use for temporary event messages that do not require a response."},
      PageRule{"button", "One Primary per screen. Ghost for Cancel. Danger only for data loss, with focus on Cancel. Activates on press. A disabled button fades."},
      PageRule{"toggle", "For an on or off setting that applies at once. A disabled toggle fades."},
      PageRule{"slider", "For a range the player tunes by feel. Show the value. A disabled slider is solid grey."},
      PageRule{"list", "For choosing among a few to many rows. Selected shows the current choice. Focus shows where the pad is. A disabled row fades. A list that doesn't fit shows a scroll bar."},
      PageRule{"dialogs", "Use when the game cannot go on until the player answers. Focus stays inside. Back cancels, unless the work can't be stopped. A long body scrolls."},
      PageRule{"notification", "Use for status that can be ignored. It never takes focus. One shows at a time, and others wait their turn."},
  };
  for (const PageRule &item : RULES) {
    StylePageRig rig(item.page);
    rig.nav.Frame([&](UiFrame &frame) {
      rig.Draw()(frame);
      Check(DrawsText(*rig.list, frame, item.rule, DEFAULT_THEME.type.body_small, DEFAULT_THEME.colors.text.secondary), item.rule);
    });
  }
}

// ---- Images -------------------------------------------------------------------------------------

// The pattern's pixel at a place.
Color PatternPixel(const std::vector<uint8_t> &rgba, int width, int x, int y) {
  const size_t at = (static_cast<size_t>(y) * static_cast<size_t>(width) + static_cast<size_t>(x)) * 4;
  return {rgba[at], rgba[at + 1], rgba[at + 2], rgba[at + 3]};
}

// How many separate runs of the notch colour lie along a row of the pattern, from `first` to `last`.
int NotchRuns(const std::vector<uint8_t> &rgba, int width, int y, int first, int last) {
  int runs = 0;
  bool inside = false;
  for (int x = first; x <= last; ++x) {
    const bool notch = PatternPixel(rgba, width, x, y) == PATTERN_NOTCH;
    runs += notch && !inside ? 1 : 0;
    inside = notch;
  }
  return runs;
}

// The corners have 1, 2, 3 and 4 notches clockwise from the top left, so each crop can be counted. The
// ring reaches the top and bottom at the middle, and on the square the sides too, and every pixel is
// opaque. The pattern is the same at either size, which only the shape changes.
void TestPatternMarks() {
  for (const auto [width, height] : {std::pair{PATTERN_SQUARE_SIDE, PATTERN_SQUARE_SIDE}, std::pair{PATTERN_WIDE_WIDTH, PATTERN_WIDE_HEIGHT}}) {
    const auto rgba = TestPatternRgba(width, height);
    Check(rgba.size() == static_cast<size_t>(width) * static_cast<size_t>(height) * 4, "the pattern is the size asked for");
    const int middle = width / 2;
    Check(NotchRuns(rgba, width, 0, 0, middle - 1) == 1, "the top left has one notch");
    Check(NotchRuns(rgba, width, 0, middle, width - 1) == 2, "the top right has two");
    Check(NotchRuns(rgba, width, height - 1, middle, width - 1) == 3, "the bottom right has three");
    Check(NotchRuns(rgba, width, height - 1, 0, middle - 1) == 4, "the bottom left has four");
    Check(PatternPixel(rgba, width, middle, 1) == PATTERN_RING && PatternPixel(rgba, width, middle, height - 2) == PATTERN_RING, "the ring reaches the top and the bottom");
    bool opaque = true;
    bool has_cross = false;
    bool has_both_checks = false;
    bool has_dark = false;
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const Color pixel = PatternPixel(rgba, width, x, y);
        opaque = opaque && pixel.a == 255;
        has_cross = has_cross || pixel == PATTERN_CROSS;
        has_dark = has_dark || pixel == PATTERN_CHECK_DARK;
        has_both_checks = has_both_checks || (has_dark && pixel == PATTERN_CHECK_LIGHT);
      }
    }
    Check(opaque, "every pixel is opaque");
    Check(has_cross && has_both_checks, "over a cross and a checker");
  }
  const auto square = TestPatternRgba(PATTERN_SQUARE_SIDE, PATTERN_SQUARE_SIDE);
  Check(PatternPixel(square, PATTERN_SQUARE_SIDE, 1, PATTERN_SQUARE_SIDE / 2) == PATTERN_RING, "and on the square the ring reaches the sides too");
  Check(PatternPixel(square, PATTERN_SQUARE_SIDE, PATTERN_SQUARE_SIDE / 2 + 20, PATTERN_SQUARE_SIDE / 2 + 20) == PATTERN_CROSS, "with the cross through the middle");
}

// What EncodePng gives decodes back to the same pixels, which a store relies on. A size that doesn't
// match the data gives nothing.
void TestEncodePngRoundTrip() {
  const auto rgba = TestPatternRgba(PATTERN_WIDE_WIDTH, PATTERN_WIDE_HEIGHT);
  const auto encoded = EncodePng(rgba, PATTERN_WIDE_WIDTH, PATTERN_WIDE_HEIGHT);
  Check(!encoded.empty(), "the pattern encodes");
  int width = 0, height = 0, channels = 0;
  unsigned char *decoded = stbi_load_from_memory(reinterpret_cast<const unsigned char *>(encoded.data()), static_cast<int>(encoded.size()), &width, &height, &channels, 4);
  Check(decoded != nullptr && width == PATTERN_WIDE_WIDTH && height == PATTERN_WIDE_HEIGHT, "and decodes at its size");
  Check(std::equal(rgba.begin(), rgba.end(), decoded), "to the same pixels");
  stbi_image_free(decoded);
  Check(EncodePng(rgba, PATTERN_WIDE_WIDTH + 1, PATTERN_WIDE_HEIGHT).empty(), "a size that doesn't match gives nothing");
  Check(EncodePng(rgba, 0, PATTERN_WIDE_HEIGHT).empty(), "nor does an empty one");
}

// The images page on its real rig, with a store of its own.
struct ImagesRig {
  explicit ImagesRig(std::string_view page = "images") : rig(page) { rig.state.images = &store; }
  StylePageRig rig;
  ImageStore store{8};
};

// The vertices the page draws from `texture`, in order, each once.
std::vector<ImDrawVert> VerticesOf(const ImDrawList &list, ImTextureData *texture) {
  std::vector<ImDrawVert> vertices;
  std::set<unsigned> seen;
  for (const ImDrawCmd &command : list.CmdBuffer) {
    if (command.TexRef._TexData != texture) {
      continue;
    }
    for (unsigned at = 0; at < command.ElemCount; ++at) {
      const unsigned vertex = command.VtxOffset + list.IdxBuffer[static_cast<int>(command.IdxOffset + at)];
      if (seen.insert(vertex).second) {
        vertices.push_back(list.VtxBuffer[static_cast<int>(vertex)]);
      }
    }
  }
  return vertices;
}

// The rectangle in `vertices` of this size showing this part of its image in the page's `column` of
// fits, if any. The first and third corners of a rectangle are its top left and bottom right. A
// column's slot starts at its left, and a smaller picture is centred in the slot.
bool HasImageQuad(const std::vector<ImDrawVert> &vertices, float left, float width, float height, ImVec2 uv_min, ImVec2 uv_max) {
  constexpr float SLACK = 0.5f;
  constexpr float UV_SLACK = 0.001f;
  for (size_t index = 0; index + 2 < vertices.size(); ++index) {
    const ImDrawVert &first = vertices[index];
    const ImDrawVert &third = vertices[index + 2];
    if (std::fabs(first.pos.x - left) <= SLACK && std::fabs(third.pos.x - first.pos.x - width) <= SLACK && std::fabs(third.pos.y - first.pos.y - height) <= SLACK && std::fabs(first.uv.x - uv_min.x) <= UV_SLACK &&
        std::fabs(first.uv.y - uv_min.y) <= UV_SLACK && std::fabs(third.uv.x - uv_max.x) <= UV_SLACK && std::fabs(third.uv.y - uv_max.y) <= UV_SLACK) {
      return true;
    }
  }
  return false;
}

// Each way of fitting puts the picture where the sums below, done apart from the code, say.
void CheckFitsOfBothPictures(ImagesRig &images) {
  constexpr float CROP = 0.21875f;
  constexpr float SOURCE_LEFT = 352.0f;
  constexpr float FILL_LEFT = SOURCE_LEFT + 200.0f;
  constexpr float CONTAIN_LEFT = FILL_LEFT + 360.0f;
  constexpr float COVER_LEFT = CONTAIN_LEFT + 360.0f;
  constexpr float CENTRED_ACROSS = (320.0f - 180.0f) / 2.0f;
  const ImVec2 whole_min{0.0f, 0.0f};
  const ImVec2 whole_max{1.0f, 1.0f};
  const auto &page = images.rig.state.images_page;
  Check(images.store.Size(*page.square).x == 256.0f && images.store.Size(*page.square).y == 256.0f, "the square one is square");
  Check(images.store.Size(*page.wide).x == 320.0f && images.store.Size(*page.wide).y == 180.0f, "the wide one is wide");
  const auto square = VerticesOf(*images.rig.list, images.store.Texture(*page.square));
  const auto wide = VerticesOf(*images.rig.list, images.store.Texture(*page.wide));
  Check(HasImageQuad(square, SOURCE_LEFT, 160.0f, 160.0f, whole_min, whole_max), "the square picture as it is");
  Check(HasImageQuad(square, FILL_LEFT, 320.0f, 180.0f, whole_min, whole_max), "stretched to fill");
  Check(HasImageQuad(square, CONTAIN_LEFT + CENTRED_ACROSS, 180.0f, 180.0f, whole_min, whole_max), "whole when contained, centred");
  Check(HasImageQuad(square, COVER_LEFT, 320.0f, 180.0f, {0.0f, CROP}, {1.0f, 1.0f - CROP}), "its middle when covering");
  Check(HasImageQuad(wide, SOURCE_LEFT, 160.0f, 90.0f, whole_min, whole_max), "the wide picture as it is");
  Check(HasImageQuad(wide, FILL_LEFT, 180.0f, 180.0f, whole_min, whole_max), "stretched to fill");
  Check(HasImageQuad(wide, CONTAIN_LEFT, 180.0f, 101.25f, whole_min, whole_max), "whole when contained");
  Check(HasImageQuad(wide, COVER_LEFT, 180.0f, 180.0f, {CROP, 0.0f}, {1.0f - CROP, 1.0f}), "its middle when covering");
  float contained_top = SCREEN.Bottom(), filled_top = SCREEN.Bottom();
  for (const ImDrawVert &vertex : wide) {
    filled_top = std::fabs(vertex.pos.x - FILL_LEFT) <= 0.5f ? std::min(filled_top, vertex.pos.y) : filled_top;
    contained_top = std::fabs(vertex.pos.x - CONTAIN_LEFT) <= 0.5f ? std::min(contained_top, vertex.pos.y) : contained_top;
  }
  Check(std::fabs(contained_top - filled_top - (180.0f - 101.25f) / 2.0f) <= 1.0f, "and centred down its slot");
}

// The captions are in the order of the columns, and each slot is outlined after its picture is drawn, or
// the picture would hide the outline.
void CheckFitCaptionsAndOutlines(ImagesRig &images) {
  constexpr int SLOTS = 6;
  constexpr int VERTICES_PER_OUTLINE = 16;
  images.rig.nav.Frame([&](UiFrame &frame) {
    images.rig.Draw()(frame);
    CheckInOrder(*images.rig.list, frame, {"Fill, stretched, so the ring is an oval", "Contain, whole, with bars around it", "Cover, filling the slot, its corners cropped"}, DEFAULT_THEME.type.caption, DEFAULT_THEME.colors.text.secondary);
  });
  const ImDrawList &list = *images.rig.list;
  Check(PageVertices(list, DEFAULT_THEME.colors.border.strong) == SLOTS * VERTICES_PER_OUTLINE, "each slot has its outline");
  int last_outline = -1;
  for (int index = 0; index < list.VtxBuffer.Size; ++index) {
    last_outline = list.VtxBuffer[index].col == ToImU32(DEFAULT_THEME.colors.border.strong) && list.VtxBuffer[index].pos.x > PANE_LEFT ? index : last_outline;
  }
  unsigned last_picture = 0;
  for (const ImDrawCmd &command : list.CmdBuffer) {
    if (command.TexRef._TexData == images.store.Texture(*images.rig.state.images_page.wide)) {
      for (unsigned at = 0; at < command.ElemCount; ++at) {
        last_picture = std::max<unsigned>(last_picture, command.VtxOffset + list.IdxBuffer[static_cast<int>(command.IdxOffset + at)]);
      }
    }
  }
  Check(last_picture > 0 && last_outline > static_cast<int>(last_picture), "over the pictures");
}

// Each way of fitting puts the picture where the sums below, done apart from the code, say. A square
// picture in a 320 x 180 slot shows whole at 180 x 180 when contained, centred across the slot, and
// covering it shows the middle 180/320 of its height. A 320 x 180 picture in a 180 x 180 slot shows whole
// at 180 x 101.25 when contained, centred down the slot, and covering it shows the middle 180/320 of its
// width. The picture as it is shows in the first column, and the fits in columns of their own, in the
// order of their captions, so the position ties each picture to its caption.
void TestFitsShowTheRightPart() {
  ImagesRig images;
  images.rig.Frames(1);
  Check(images.rig.state.images_page.square.has_value() && images.rig.state.images_page.wide.has_value(), "the page loads both patterns");
  CheckFitsOfBothPictures(images);
  CheckFitCaptionsAndOutlines(images);
  const int loaded = static_cast<int>(images.store.LiveCount());
  images.rig.Frames(3);
  Check(static_cast<int>(images.store.LiveCount()) == loaded && loaded == 2, "and the patterns load only once");
}

// A slot keeps its inset surface behind a contained picture, which is what shows the bars. Without a
// store the page still draws its slots, and they stay empty.
void TestSlotsWithoutAStore() {
  constexpr float SLOT_WIDTH = 320.0f;
  constexpr float SLOT_HEIGHT = 180.0f;
  StylePageRig rig("images");
  rig.Frames(1);
  Check(FindQuad(*rig.list, DEFAULT_THEME.colors.surface.inset, SLOT_WIDTH, SLOT_HEIGHT).has_value(), "the slot is drawn with no store");
  Check(PageVertices(*rig.list, DEFAULT_THEME.colors.border.strong) > 0, "and outlined, which nothing else on this page is");
  Check(!rig.state.images_page.tried, "and nothing was tried without one");
}

// Each tile on the Corners view starts its top edge as far in as its radius, as on the spacing page, in
// the scale's order. The Fit view is the baseline, whose pictures are all square-cornered. A missing
// image leaves only its surface, and that surface is the 160 x 90 slot.
void TestCornersView() {
  constexpr float TILE = 96.0f;
  constexpr float PITCH = TILE + 16.0f;
  constexpr std::array WANTED{4.0f, 8.0f, 12.0f, 16.0f, 24.0f, 47.0f};
  ImagesRig images;
  images.rig.state.images_view = 1;
  images.rig.Frames(1);
  Check(images.rig.state.images_page.square.has_value() && images.rig.state.images_page.wide.has_value(), "the page loads both patterns");
  const auto vertices = VerticesOf(*images.rig.list, images.store.Texture(*images.rig.state.images_page.square));
  Check(!vertices.empty(), "the tiles are drawn");
  float left = SCREEN.Right(), top = SCREEN.Bottom();
  for (const ImDrawVert &vertex : vertices) {
    left = std::min(left, vertex.pos.x);
    top = std::min(top, vertex.pos.y);
  }
  for (size_t tile = 0; tile < WANTED.size(); ++tile) {
    const float tile_left = left + PITCH * static_cast<float>(tile);
    float start = SCREEN.Right();
    for (const ImDrawVert &vertex : vertices) {
      const bool in_tile = vertex.pos.x >= tile_left - 0.5f && vertex.pos.x <= tile_left + TILE + 0.5f;
      start = in_tile && vertex.pos.y <= top + 0.25f ? std::min(start, vertex.pos.x) : start;
    }
    Check(std::fabs(start - tile_left - WANTED[tile]) <= 0.25f, "each tile is rounded by its radius");
  }
  Check(FindQuad(*images.rig.list, DEFAULT_THEME.colors.surface.inset, 160.0f, 90.0f).has_value(), "a missing image leaves its surface");
  Check(VerticesOf(*images.rig.list, images.store.Texture(*images.rig.state.images_page.wide)).empty(), "and this view draws no wide picture");
  images.rig.nav.Frame([&](UiFrame &frame) {
    images.rig.Draw()(frame);
    Check(DrawsText(*images.rig.list, frame, "A missing image draws nothing, so what shows is the surface behind it.", DEFAULT_THEME.type.caption, DEFAULT_THEME.colors.text.secondary), "and says so");
  });
}

// The tabs choose the view, and a reset goes back to the first while the store keeps its images, which
// aren't loaded a second time.
void TestImagesViewsAndReset() {
  ImagesRig images;
  auto &rig = images.rig;
  auto draw = rig.Draw();
  rig.Frames(2);
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.images_view == 0, "entering stays on the view");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  Check(rig.state.images_view == 1, "and moving along the tabs chooses the next");
  rig.state.Reset();
  Check(rig.state.images_view == 0, "a reset puts it back");
  rig.state.show = rig.page;
  rig.Frames(3);
  Check(images.store.LiveCount() == 2, "and the patterns aren't loaded again");
}

// ---- Passive component pages ----------------------------------------------------------------------

// The box around what the page draws in `colour` in the given part of the page.
PageSpan SpanWithin(const ImDrawList &list, Color colour, float left, float right, float top = 0.0f, float bottom = SCREEN.Bottom()) {
  PageSpan span;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    if (vertex.col == ToImU32(colour) && vertex.pos.x >= left && vertex.pos.x <= right && vertex.pos.y >= top && vertex.pos.y <= bottom) {
      span.left = std::min(span.left, vertex.pos.x);
      span.top = std::min(span.top, vertex.pos.y);
      span.right = std::max(span.right, vertex.pos.x);
      span.bottom = std::max(span.bottom, vertex.pos.y);
    }
  }
  return span;
}

// Words on the backdrop read on each one, as worked out by the contrast sum the Colour page uses. White
// would disappear on Bright, and dark on Dark, so one colour for all can't pass.
void TestStageInkReadsOnEachBackdrop() {
  StageState stage;
  for (int backdrop = 0; backdrop < static_cast<int>(STAGE_BACKDROPS.size()); ++backdrop) {
    stage.backdrop = backdrop;
    Check(ContrastRatio(StageInk(DEFAULT_THEME, stage), STAGE_BACKDROPS[static_cast<size_t>(backdrop)]) >= CONTRAST_TEXT, "words on the backdrop can be read");
  }
}

// The row names and column names are in order, tying each to what it labels, the card and its nesting
// are named, and the inner card really is one, with its border inside the outer's. On the bright backdrop
// words on it turn dark, and words on a card don't.
void CheckBoxPageWords(StylePageRig &rig) {
  constexpr float NESTING_CARD_LEFT = 1605.0f;
  constexpr float NESTING_CARD_RIGHT = 1810.0f;
  // Past the outer cards' round corners, whose vertices reach in by their radius.
  constexpr float PAST_THE_CORNERS = 17.0f;
  const auto &colors = DEFAULT_THEME.colors;
  const auto &caption = DEFAULT_THEME.type.caption;
  const bool has_inner_border = std::any_of(rig.list->VtxBuffer.begin(), rig.list->VtxBuffer.end(), [&](const ImDrawVert &vertex) {
    return vertex.col == ToImU32(colors.border.subtle) && vertex.pos.x > NESTING_CARD_LEFT + PAST_THE_CORNERS && vertex.pos.x < NESTING_CARD_RIGHT - PAST_THE_CORNERS;
  });
  Check(has_inner_border, "the card in a card is a card");
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    CheckInOrder(*rig.list, frame, {"No border", "Subtle", "Standard", "Strong", "Raised", "Overlay", "HUD", "Inset", "Padding, corners and a low shadow", "One card", "In a card", "Do, one card", "Don't, a card in a card"}, caption, colors.text.primary);
  });
  rig.state.stage.backdrop = STAGE_BACKDROP_BRIGHT;
  rig.Frames(1);
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    Check(DrawsText(*rig.list, frame, "Raised", caption, Palette::GRAY_0), "on the bright backdrop, words on it turn dark");
    Check(DrawsText(*rig.list, frame, "One card", caption, colors.text.primary), "and words on a card stay as they are");
  });
}

// Each surface fills a row of four cells, and each border outlines one column of them, the first having
// none. Positions are written out here apart from the page.
void TestBoxPage() {
  constexpr float CELLS_LEFT = 504.0f;
  constexpr float CELL_WIDTH = 200.0f;
  constexpr float CELL_HEIGHT = 96.0f;
  constexpr float PITCH = 216.0f;
  constexpr float ROW_PITCH = 112.0f;
  constexpr float SLACK = 1.0f;
  constexpr int CELLS = 4;
  const auto &colors = DEFAULT_THEME.colors;
  StageRig stage("box");
  auto &rig = stage.rig;
  const float cells_right = CELLS_LEFT + PITCH * CELLS;
  const PageSpan first_row = SpanWithin(*rig.list, colors.surface.raised, CELLS_LEFT, cells_right);
  const float rows_top = first_row.top;
  const std::array surfaces{colors.surface.raised, colors.surface.overlay, colors.surface.hud, colors.surface.inset};
  for (size_t row = 0; row < surfaces.size(); ++row) {
    const float top = rows_top + ROW_PITCH * static_cast<float>(row);
    const PageSpan span = SpanWithin(*rig.list, surfaces[row], CELLS_LEFT, cells_right, top, top + CELL_HEIGHT + SLACK);
    Check(std::fabs(span.left - CELLS_LEFT) <= SLACK && std::fabs(span.right - (CELLS_LEFT + PITCH * (CELLS - 1) + CELL_WIDTH)) <= SLACK, "each surface fills a row of cells");
    Check(std::fabs(span.top - top) <= SLACK && std::fabs(span.bottom - span.top - CELL_HEIGHT) <= SLACK, "in its own row");
    // Its top edge starts as far in as the corner's radius.
    float top_edge_start = SCREEN.Right();
    for (const ImDrawVert &vertex : rig.list->VtxBuffer) {
      if (vertex.col == ToImU32(surfaces[row]) && vertex.pos.x >= CELLS_LEFT && vertex.pos.x <= CELLS_LEFT + CELL_WIDTH && vertex.pos.y <= top + 0.25f && vertex.pos.y >= top - 0.25f) {
        top_edge_start = std::min(top_edge_start, vertex.pos.x);
      }
    }
    Check(std::fabs(top_edge_start - CELLS_LEFT - DEFAULT_THEME.radius.md) <= 0.25f, "with rounded corners");
  }
  const std::array borders{colors.border.subtle, colors.border.standard, colors.border.strong};
  for (size_t border = 0; border < borders.size(); ++border) {
    const float left = CELLS_LEFT + PITCH * static_cast<float>(border + 1);
    // Inside the panel, as the page's own slider has the strong outline too.
    const PageSpan span = SpanWithin(*rig.list, borders[border], CELLS_LEFT, cells_right, stage.Panel().top);
    Check(std::fabs(span.left - left) <= SLACK && std::fabs(span.right - (left + CELL_WIDTH)) <= SLACK, "each border outlines one column");
    Check(std::fabs(span.bottom - span.top - (ROW_PITCH * (CELLS - 1) + CELL_HEIGHT)) <= SLACK, "from top to bottom");
  }
  CheckBoxPageWords(rig);
}

// The words in each alignment's slot sit where it says, worked out here apart from the code.
void CheckAlignedWords(const ImDrawList &list) {
  constexpr float SLOT_LEFT = 368.0f;
  constexpr float SLOT_WIDTH = 420.0f;
  constexpr float PADDING = 8.0f;
  constexpr float SLACK = 3.0f;
  const Color ink = DEFAULT_THEME.colors.text.accent;
  const PageSpan all = SpanOf(list, ink, 212.0f);
  Check(all.right > all.left, "the alignment words are drawn");
  constexpr float ROW_PITCH = 52.0f;
  const auto word = [&](int row) { return SpanWithin(list, ink, SLOT_LEFT, SLOT_LEFT + SLOT_WIDTH, all.top + ROW_PITCH * static_cast<float>(row) - 6.0f, all.top + ROW_PITCH * static_cast<float>(row) + 30.0f); };
  Check(std::fabs(word(0).left - (SLOT_LEFT + PADDING)) <= SLACK, "Start sits at the slot's start");
  Check(std::fabs((word(1).left + word(1).right) / 2.0f - (SLOT_LEFT + SLOT_WIDTH / 2.0f)) <= SLACK, "Center in its middle");
  Check(std::fabs(word(2).right - (SLOT_LEFT + SLOT_WIDTH - PADDING)) <= SLACK, "and End at its end");
}

// The table gives what each style really reaches at each output, worked out here by hand, floors marked,
// row by row. The output chosen is bold, and bold moves with it. The paragraph's limits hold, each with
// its caption, and the words are the same in each.
void TestTextPage() {
  const auto &colors = DEFAULT_THEME.colors;
  StageRig stage("text");
  auto &rig = stage.rig;
  CheckAlignedWords(*rig.list);
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    const auto &caption = DEFAULT_THEME.type.caption;
    TypeStyle bold = caption;
    bold.weight = FontWeight::Bold;
    CheckInOrder(*rig.list, frame, {"body_small", "14, floor", "40", "27", "caption", "12, floor", "32", "21", "dense_small", "11, floor", "26", "17"}, caption, colors.text.primary);
    CheckInOrder(*rig.list, frame, {"1080p", "20", "16", "13"}, bold, colors.text.primary);
    CheckInOrder(*rig.list, frame, {"At most 1 line", "At most 2 lines", "At most 3 lines", "No limit"}, caption, colors.text.primary);
    Check(!DrawsText(*rig.list, frame, "At most 1 lines", caption, colors.text.primary), "with one line said as one");
  });
  rig.state.stage.output = 0;
  rig.Frames(2);
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    TypeStyle bold = DEFAULT_THEME.type.caption;
    bold.weight = FontWeight::Bold;
    // Laid out in the stage's frame, as its text is at that output's size.
    CheckInOrder(*rig.list, rig.state.stage.frame, {"720p", "14, floor", "12, floor", "11, floor"}, bold, colors.text.primary);
  });
}

// The paragraphs show as many lines as they're allowed, the last cut with an ellipsis, compared glyph by
// glyph with the same words laid out by themselves under each limit, and each differs from the rest, so
// none can pass as another. The one with no limit shows all of it.
void TestTextPageLineLimits() {
  constexpr float WIDTH = 420.0f;
  const std::string_view words =
      "A paragraph that says enough to show what a limit on its lines does. Each slot below holds the same words, and each is allowed one more line than the last, until the one with no limit shows all of them. "
      "Where the words are cut short, the last line ends with an ellipsis, so it is clear that there is more. A paragraph like this one belongs in a dialog or a description, where there is room to read it, and "
      "rarely in a list, a button or a toast, where a single line should say what the player needs to know.";
  const auto &colors = DEFAULT_THEME.colors;
  StageRig stage("text");
  auto &rig = stage.rig;
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    std::vector<std::ptrdiff_t> places;
    for (const int lines : {1, 2, 3, 0}) {
      TextProps props{words, DEFAULT_THEME.type.body_small, colors.text.primary};
      props.overflow = TextOverflow::Wrap;
      props.max_lines = lines;
      ImDrawList alone(ImGui::GetDrawListSharedData());
      alone._ResetForNewFrame();
      alone.PushClipRectFullScreen();
      const TextLayout layout = LayoutText(frame.Ui(), props, WIDTH);
      Check(lines == 0 ? static_cast<int>(layout.lines.size()) > 3 : static_cast<int>(layout.lines.size()) == lines, "the limit gives that many lines, and no limit gives more");
      DrawText(alone, layout, props, {PANE_LEFT + 1.0f, 0.0f, WIDTH, SCREEN.height});
      const auto wanted = GlyphUvs(alone, colors.text.primary);
      const auto drawn = GlyphUvs(*rig.list, colors.text.primary);
      const auto found = std::search(drawn.begin(), drawn.end(), wanted.begin(), wanted.end());
      Check(!wanted.empty() && found != drawn.end(), "each paragraph shows just those lines");
      places.push_back(found - drawn.begin());
    }
    Check(std::is_sorted(places.begin(), places.end()), "in order of their limits");
  });
}

// The box around the vertices at or right of `from`, and how far along the top edge the first starts.
struct ImageSpan {
  float left = SCREEN.Right();
  float right = 0.0f;
  float top = SCREEN.Bottom();
  float bottom = 0.0f;
  float uv_top = 1.0f;
  float top_edge_start = SCREEN.Right();
};
ImageSpan ImageSpanOf(const std::vector<ImDrawVert> &vertices, float from, float to) {
  ImageSpan span;
  for (const ImDrawVert &vertex : vertices) {
    if (vertex.pos.x >= from && vertex.pos.x <= to) {
      span.left = std::min(span.left, vertex.pos.x);
      span.right = std::max(span.right, vertex.pos.x);
      span.top = std::min(span.top, vertex.pos.y);
      span.bottom = std::max(span.bottom, vertex.pos.y);
      span.uv_top = std::min(span.uv_top, vertex.uv.y);
    }
  }
  for (const ImDrawVert &vertex : vertices) {
    if (vertex.pos.x >= from && vertex.pos.x <= to && vertex.pos.y <= span.top + 0.25f) {
      span.top_edge_start = std::min(span.top_edge_start, vertex.pos.x);
    }
  }
  return span;
}

// Where the poster ends and the icon begins on the page, as the card with no picture lies between.
constexpr float IMAGE_PAGE_SPLIT = 900.0f;

// A poster covers its slot, showing the middle of the square picture, and the small images are the sizes
// their captions say, each with its own corners. A card with no picture draws none and keeps its words.
// Worked out here apart from the code.
void TestImagePage() {
  constexpr float SLACK = 1.0f;
  constexpr float CORNER_SLACK = 0.25f;
  const auto &radius = DEFAULT_THEME.radius;
  ImagesRig images("image");
  auto &rig = images.rig;
  rig.Frames(2);
  const auto &page = rig.state.images_page;
  Check(page.square.has_value() && page.wide.has_value(), "the page loads the patterns it shows");
  const auto square = VerticesOf(*rig.list, images.store.Texture(*page.square));
  const auto wide = VerticesOf(*rig.list, images.store.Texture(*page.wide));
  const ImageSpan poster = ImageSpanOf(square, 0.0f, IMAGE_PAGE_SPLIT);
  Check(std::fabs(poster.right - poster.left - 408.0f) <= SLACK && std::fabs(poster.bottom - poster.top - 230.0f) <= SLACK, "the poster is its size");
  Check(std::fabs(poster.uv_top - (1.0f - 230.0f / 408.0f) / 2.0f) <= 0.01f, "and covers it, showing the middle of the picture");
  Check(std::fabs(poster.top_edge_start - poster.left - radius.sm) <= CORNER_SLACK, "with small corners");
  const ImageSpan icon = ImageSpanOf(square, IMAGE_PAGE_SPLIT, SCREEN.Right());
  Check(std::fabs(icon.right - icon.left - 80.0f) <= SLACK && std::fabs(icon.bottom - icon.top - 80.0f) <= SLACK, "the icon is 80 x 80");
  Check(std::fabs(icon.top_edge_start - icon.left - radius.sm) <= CORNER_SLACK, "with small corners");
  const ImageSpan thumb = ImageSpanOf(wide, 0.0f, SCREEN.Right());
  Check(std::fabs(thumb.right - thumb.left - 96.0f) <= SLACK && std::fabs(thumb.bottom - thumb.top - 54.0f) <= SLACK, "the thumbnail is 96 x 54");
  Check(std::fabs(thumb.top_edge_start - thumb.left - radius.xs) <= CORNER_SLACK, "with the smallest");
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    CheckInOrder(*rig.list, frame, {"A poster, 408 x 230", "Its image missing, the card shows through", "An icon, 80 x 80", "A row's thumbnail, 96 x 54"}, DEFAULT_THEME.type.caption, DEFAULT_THEME.colors.text.primary);
  });
}

// At a larger output, what's too big shrinks, and an image shrinks with the rest, as it's drawn from a
// texture of its own, in draw commands of their own. Its command keeps to the panel, and its caption
// gives its size at that output.
void TestImagesShrinkOnTheStage() {
  constexpr float POSTER_AT_4K = 816.0f;
  constexpr float SLACK = 2.0f;
  ImagesRig images("image");
  auto &rig = images.rig;
  rig.state.stage.output = 2;
  rig.Frames(3);
  const float fit = rig.state.stage.fit;
  Check(fit < 1.0f, "at 4K the page is shrunk");
  const auto &page = rig.state.images_page;
  ImTextureData *const texture = images.store.Texture(*page.square);
  // The poster is the first thing drawn from this texture, in the order the vertices came.
  float left = SCREEN.Right(), right = 0.0f;
  for (const ImDrawVert &vertex : VerticesOf(*rig.list, texture)) {
    if (vertex.pos.x > left + POSTER_AT_4K * fit + SLACK) {
      break;
    }
    left = std::min(left, vertex.pos.x);
    right = std::max(right, vertex.pos.x);
  }
  Check(std::fabs(right - left - POSTER_AT_4K * fit) <= SLACK, "the poster is its size at that output, shrunk");
  const PageSpan panel = SpanOf(*rig.list, STAGE_BACKDROPS[static_cast<size_t>(rig.state.stage.backdrop)]);
  for (const ImDrawCmd &command : rig.list->CmdBuffer) {
    if (command.TexRef._TexData == texture) {
      Check(SameClip(command.ClipRect, {panel.left, panel.top, panel.right, panel.bottom}), "and it keeps to the panel");
    }
  }
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    // Laid out in the stage's frame, as its text is at that output's size.
    Check(DrawsText(*rig.list, rig.state.stage.frame, "A poster, 816 x 460", DEFAULT_THEME.type.caption, DEFAULT_THEME.colors.text.primary), "and the captions follow the output");
  });
}

// Each grade has a cup at each size in its own colour, keeping the cup's shape, in the order of the words
// beside them. The cup stretched out of shape is square, where the others aren't. Sizes are written out
// here apart from the page.
void TestGlyphsPage() {
  constexpr float FIRST_CELL = 532.0f;
  constexpr float CELL_PITCH = 164.0f;
  constexpr float CELL_WIDTH = 160.0f;
  constexpr std::array WIDTHS{22.0f, 40.0f, 80.0f};
  constexpr std::array HEIGHTS{21.0f, 38.0f, 76.0f};
  constexpr float SLACK = 2.0f;
  // The handles' stroke is centred on the cup's edge, so it reaches a little past its box.
  constexpr float HANDLE_REACH = 1.1f;
  const auto &colors = DEFAULT_THEME.colors;
  StageRig stage("glyphs");
  auto &rig = stage.rig;
  const std::array grades{colors.trophy.platinum, colors.trophy.gold, colors.trophy.silver, colors.trophy.bronze};
  for (const Color grade : grades) {
    for (size_t size = 0; size < WIDTHS.size(); ++size) {
      const float left = FIRST_CELL + CELL_PITCH * static_cast<float>(size);
      const PageSpan cup = SpanWithin(*rig.list, grade, left, left + CELL_WIDTH);
      const float width = cup.right - cup.left;
      Check(width >= WIDTHS[size] && width <= WIDTHS[size] * HANDLE_REACH && std::fabs(cup.bottom - cup.top - HEIGHTS[size]) <= SLACK, "each grade has a cup at each size");
    }
  }
  // The cup in use lies past the grid's last cell, at its natural size.
  const PageSpan in_use = SpanWithin(*rig.list, colors.trophy.gold, FIRST_CELL + CELL_PITCH * static_cast<float>(WIDTHS.size()), SCREEN.Right());
  Check(in_use.right - in_use.left >= WIDTHS[0] && in_use.right - in_use.left <= WIDTHS[0] * HANDLE_REACH && std::fabs(in_use.bottom - in_use.top - HEIGHTS[0]) <= SLACK, "the cup in use is at its natural size");
  const PageSpan stretched = SpanOf(*rig.list, colors.status.warning);
  Check(std::fabs(stretched.bottom - stretched.top - 80.0f) <= SLACK && stretched.right - stretched.left >= 80.0f && stretched.right - stretched.left <= 80.0f * HANDLE_REACH, "the stretched cup is as tall as its box is wide");
  rig.nav.Frame([&](UiFrame &frame) {
    rig.Draw()(frame);
    CheckInOrder(*rig.list, frame, {"Platinum", "Gold", "Silver", "Bronze", "In use, beside a word", "Don't, stretched out of shape"}, DEFAULT_THEME.type.caption, colors.text.primary);
    Check(DrawsText(*rig.list, frame, "Gold trophy", DEFAULT_THEME.type.label, colors.text.primary), "the cup in use has its word");
  });
}

// A key held as focus leaves the sidebar goes back to navigation, or the page couldn't use it. Right
// enters the page with down still held, which the focused row owned until then.
void TestHeldKeyIsGivenBackLeavingTheSidebar() {
  StylePageRig rig("colour");
  auto draw = rig.Draw();
  rig.Frames(1);
  rig.nav.Key(ImGuiKey_GamepadDpadDown, true);
  rig.nav.Frame(draw);
  rig.nav.Frame(draw);
  Check(ImGui::GetKeyOwner(ImGuiKey_GamepadDpadDown) != ImGuiKeyOwner_NoOwner, "the focused row owns the held key");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  rig.nav.Frame(draw);
  Check(FocusedControl() == rig.state.entry, "right enters the page with the key still held");
  Check(ImGui::GetKeyOwner(ImGuiKey_GamepadDpadDown) == ImGuiKeyOwner_NoOwner, "which gives the key back");
  rig.nav.Key(ImGuiKey_GamepadDpadDown, false);
}

// With the toggle on, each style's line box is tinted, one quad a style, and each is as tall as that
// style's line in the theme, written out here. With it off, none is. A press on the toggle turns it on.
void TestTypographyLineBoxes() {
  const auto &ramp = DEFAULT_THEME.type;
  std::multiset<float> wanted{ramp.display.line, ramp.title.line, ramp.heading.line, ramp.body.line, ramp.body_small.line,
                              ramp.label.line, ramp.caption.line, ramp.dense.line, ramp.dense_small.line};
  StylePageRig rig("typography");
  auto draw = rig.Draw();
  rig.Frames(1);
  const Color tint = DEFAULT_THEME.colors.accent.tint;
  Check(PageVertices(*rig.list, tint) == 0, "with the toggle off nothing is tinted");
  rig.nav.Press(ImGuiKey_GamepadDpadRight, draw);
  rig.nav.Press(ImGuiKey_GamepadDpadDown, draw);
  rig.nav.Press(ImGuiKey_GamepadFaceDown, draw);
  Check(rig.state.typography_line_boxes, "a press on the toggle turns it on");
  rig.Frames(1);
  std::multiset<float> heights;
  float top = 0.0f;
  int corner = 0;
  for (const ImDrawVert &vertex : rig.list->VtxBuffer) {
    if (vertex.pos.x > PANE_LEFT && vertex.col == ToImU32(tint)) {
      if (corner == 0) {
        top = vertex.pos.y;
      } else if (corner == 2) {
        heights.insert(vertex.pos.y - top);
      }
      corner = (corner + 1) % 4;
    }
  }
  Check(heights == wanted, "each style has a line box of its own height");
}

// The heights of the lines of primary text right of `left`. A glyph is four vertices, and the spans
// of the glyphs on a line overlap, while those of two lines don't.
std::vector<float> PrimaryLineHeights(const ImDrawList &list, float left) {
  constexpr int VERTICES_PER_GLYPH = 4;
  std::vector<std::pair<float, float>> spans;
  const auto &vertices = list.VtxBuffer;
  for (int index = 0; index + VERTICES_PER_GLYPH <= vertices.Size;) {
    if (vertices[index].pos.x > left && vertices[index].col == ToImU32(DEFAULT_THEME.colors.text.primary)) {
      spans.push_back({std::min(vertices[index].pos.y, vertices[index + 2].pos.y), std::max(vertices[index].pos.y, vertices[index + 2].pos.y)});
      index += VERTICES_PER_GLYPH;
    } else {
      ++index;
    }
  }
  std::sort(spans.begin(), spans.end());
  std::vector<float> heights;
  std::pair<float, float> line = spans.empty() ? std::pair{0.0f, 0.0f} : spans.front();
  for (const auto &span : spans) {
    if (span.first > line.second) {
      heights.push_back(line.second - line.first);
      line = span;
    }
    line.second = std::max(line.second, span.second);
  }
  heights.push_back(line.second - line.first);
  return heights;
}

// On the Text view the clip mode cuts at its slot, which only a command with that clip shows, and the
// four text levels are drawn, on the raised surface. The Ramp view is the baseline for the clip.
void TestTypographyOverflowModes() {
  // Narrower than the 320 pixel slot by its padding and border, and wider than any text the ramp clips.
  constexpr float LEAST_CLIP = 280.0f;
  constexpr float MOST_CLIP = 320.0f;
  const auto slot_clips = [](const ImDrawList &list) {
    return std::count_if(list.CmdBuffer.begin(), list.CmdBuffer.end(), [](const ImDrawCmd &command) {
      const float width = command.ClipRect.z - command.ClipRect.x;
      return command.ClipRect.x >= PANE_LEFT && command.ElemCount > 0 && width >= LEAST_CLIP && width <= MOST_CLIP;
    });
  };
  StylePageRig rig("typography");
  rig.Frames(1);
  Check(slot_clips(*rig.list) == 0, "the ramp has no slot to clip to");
  rig.state.typography_view = 1;
  rig.Frames(1);
  Check(slot_clips(*rig.list) == 1, "only the clip mode cuts at its slot");
  // The levels are looked for on the raised surface, as the page's title and rule are drawn in two of
  // these colours too.
  const auto &colors = DEFAULT_THEME.colors;
  float surface_top = SCREEN.height;
  float surface_bottom = 0.0f;
  for (const ImDrawVert &vertex : rig.list->VtxBuffer) {
    if (vertex.pos.x > PANE_LEFT && vertex.col == ToImU32(colors.surface.raised)) {
      surface_top = std::min(surface_top, vertex.pos.y);
      surface_bottom = std::max(surface_bottom, vertex.pos.y);
    }
  }
  Check(surface_bottom > surface_top, "the raised surface is drawn");
  for (const Color &level : {colors.text.primary, colors.text.secondary, colors.text.tertiary, colors.text.disabled}) {
    const auto &vertices = rig.list->VtxBuffer;
    const bool found = std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &vertex) {
      return vertex.pos.x > PANE_LEFT && vertex.col == ToImU32(level) && vertex.pos.y >= surface_top && vertex.pos.y <= surface_bottom;
    });
    Check(found, "each text level is drawn on it");
  }
  // The slots' column, right of the page's headings and labels. Wrapping shows as more lines than
  // slots, and shrinking as a line shorter than the rest.
  constexpr float SLOTS_LEFT = 592.0f;
  constexpr int SLOT_LINES = 9;
  constexpr float SHRUNK = 0.95f;
  const std::vector<float> line_heights = PrimaryLineHeights(*rig.list, SLOTS_LEFT);
  Check(static_cast<int>(line_heights.size()) == SLOT_LINES, "the wrapped rows take two lines each and the others one");
  const auto [shortest, tallest] = std::minmax_element(line_heights.begin(), line_heights.end());
  Check(*shortest < *tallest * SHRUNK, "and the shrunk row is smaller than the rest");
}

} // namespace

int main() {
  TestRegistry();
  TestSidebarWalksTheRealPages();
  TestToastPageHeldGlowSpills();
  TestToastPageFollowsTheGrade();
  TestToastPageShowsIcons();
  TestToastPageReplayRestartsTheLiveToast();
  TestToastPageTabsRestartTheLiveToast();
  TestToastPageLoops();
  TestToastPageRestartsWhenShownAgain();
  TestToastPageLiveShowsTheChosenGlow();
  TestToastPageReplaysDoNotFillTheStore();
  TestToastPageTriesABadIconOnce();
  TestToastPageWithoutImages();
  TestToastPageReset();
  TestBarWithShouldersOwnsThem();
  TestModalOwnsShouldersAndBack();
  TestLeftMissInModalStays();
  TestForeignDialogKeepsFocus();
  TestShoulderIntoTabsPageDoesNotStepTab();
  TestSidebarShowsFocusedPage();
  TestBackReturnsToRow();
  TestShoulders();
  TestShouldersInsideAPage();
  TestPagesHaveTheirOwnNames();
  TestFocusReturnsToShownPage();
  TestLeftWithNothingBeside();
  TestOtherMovesStayInThePage();
  TestClickShowsPage();
  TestPaneLeavesRoomForTheRing();
  TestNoPages();
  TestStaleFrameIsReported();
  TestShowRequest();
  TestOptionDefaults();
  TestOptionScreenshot();
  TestOptionSize();
  TestOptionMissingValue();
  TestOptionBadNumber();
  TestOptionUnknownFlag();
  TestToRgba();
  TestPngRoundTrip();
  TestPngWriteFails();
  TestScreenshotJobs();
  TestContrastMaths();
  TestContrastRowsAgainstWorkedFigures();
  TestColourPalette();
  TestColourRoles();
  TestContrastTableReportsEveryRow();
  TestStylePagesFit();
  TestViewTabsChooseAndReset();
  TestHeldKeyIsGivenBackLeavingTheSidebar();
  TestSidebarStepsOnArrowsAndShowsTheRing();
  TestTypographyLineBoxes();
  TestTypographyOverflowModes();
  TestMovesDownStayInThePage();
  TestLeftLeavesARealPage();
  TestSpacingBars();
  TestSpacingRadii();
  TestSpacingBorders();
  TestFrameMockFollowsTheRealScale();
  TestFrameControls();
  TestPaddingDiagrams();
  TestEffectsDepth();
  TestEffectsShine();
  TestEffectsShineReduced();
  TestEffectsMotion();
  TestLoopHoldsOnALongUptime();
  TestScaledDrawScope();
  TestScaledDrawScopeWithTheSameClip();
  TestStageFit();
  TestStageOutputs();
  TestStageOpacityAndBackdrop();
  TestStageControls();
  TestProgressSpecimens();
  TestStageLiveCopy();
  TestStageLiveCopyInput();
  TestStageWithNoRoomIsNotShrunk();
  TestButtonPage();
  TestGhostOnBrightBackdrop();
  TestTogglePage();
  TestSliderPage();
  TestListPage();
  TestRowsOnBrightBackdrop();
  TestTabsPage();
  TestDialogSays();
  TestDialogOverlaysThePanel();
  TestConfirmDialog();
  TestProgressAsksBeforeStopping();
  TestProgressBarFollowsTheWork();
  TestClosingDialogDrawsNothing();
  TestSavingCannotBeStopped();
  TestFinishedWorkReturnsFocus();
  TestStrandedDialogDoesntBlockThePage();
  TestErrorDialog();
  TestLongDialogGivesWay();
  TestLongDialogScrolls();
  TestDialogsPageViews();
  TestNotificationFrames();
  TestNotificationKinds();
  TestMetricPage();
  TestSidebarScrollsToItsLastPage();
  TestBackShowsTheSidebarsRow();
  TestPagesSayTheirRules();
  TestStageInkReadsOnEachBackdrop();
  TestBoxPage();
  TestTextPage();
  TestTextPageLineLimits();
  TestImagePage();
  TestImagesShrinkOnTheStage();
  TestGlyphsPage();
  TestPatternMarks();
  TestEncodePngRoundTrip();
  TestFitsShowTheRightPart();
  TestSlotsWithoutAStore();
  TestCornersView();
  TestImagesViewsAndReset();
  std::printf("System UI gallery tests passed\n");
  return 0;
}
