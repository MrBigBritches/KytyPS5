#include "SystemUiElementTestHooks.h"
#include "SystemUiNavHarness.h"
#include "SystemUiDrawingKit.h"
#include "graphics/presentation/system/animation/posed.h"
#include "graphics/presentation/system/components/box.h"
#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/list.h"
#include "graphics/presentation/system/components/image.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/units.h"
#include "graphics/presentation/system/effects/halo.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/effects/highlight.h"
#include "graphics/presentation/system/effects/shine.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/frameInternals.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/fonts.h"
#include "graphics/presentation/system/layout/flex.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics::SystemUi;
using namespace Libs::Graphics::SystemUi::Literals;
using SystemUiTest::NavHarness;
using SystemUiTest::SameCommands;

constexpr float TOLERANCE = 0.001f;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiElementTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two lengths agree to within rounding noise.
bool Near(float actual, float expected) { return std::fabs(actual - expected) <= TOLERANCE; }

constexpr float WIDTH_720P = 1280;
constexpr float HEIGHT_720P = 720;

// A 720p output, where the scale is 2/3 and rounding each length shows. At 1080p the scale is 1,
// so converting with or without rounding would give the same numbers.
ScaleContext Scale720() { return ScaleContext(DEFAULT_THEME.scale, WIDTH_720P, HEIGHT_720P); }

// Design units convert as the theme's tokens do, each length rounded to a whole pixel. 16 dp at
// 2/3 is 10.67 unrounded, so only rounding gives 11.
void TestToPixelsRounds() {
  const auto scale = Scale720();
  Check(Near(scale.Scale(), 2.0f / 3.0f), "720p is two thirds of 1080p");
  Check(Near(ToPixels(scale, 16_dp), scale.Px(16)) && Near(ToPixels(scale, 16_dp), 11), "16 dp is 11 px at 720p");
  Check(!Near(16 * scale.Scale(), 11), "and unrounded it would not be");
}

// FromPixels gives the design length that converts back to the same whole pixels, at scales where
// a plain division could land a hair short and round down. Every pixel count up to 4000 at 720p
// and at an odd user scale round-trips. A length of 11 px unconverted as dp would be 7 px, so the
// conversion is not the identity.
void TestFromPixelsRoundTrips() {
  constexpr int MAX_PIXELS = 4000;
  constexpr float ODD_USER_SCALE = 1.37f;
  const ScaleContext odd(DEFAULT_THEME.scale, WIDTH_720P, HEIGHT_720P, {.user_scale = ODD_USER_SCALE});
  for (const ScaleContext &scale : {Scale720(), odd}) {
    for (int pixels = 0; pixels <= MAX_PIXELS; ++pixels) {
      Check(ToPixels(scale, FromPixels(scale, static_cast<float>(pixels))) == static_cast<float>(pixels), "a pixel count survives the round trip");
    }
  }
  Check(!Near(ToPixels(Scale720(), Dp(11)), 11), "while the same number as dp is not the same pixels");
}

// A border never drops below one pixel. Half a dp at 2/3 rounds to 0, which the border width
// lifts back to 1. A plain length of half a dp is 0, the baseline. Above the minimum a border
// follows its width, so 3 dp is 2 px.
void TestBorderPixels() {
  const auto scale = Scale720();
  Check(Near(ToPixels(scale, 0.5_dp), 0), "half a dp is no pixels as a length");
  Check(Near(ToBorderPixels(scale, 0.5_dp), 1), "but one pixel as a border");
  Check(Near(ToBorderPixels(scale, 3_dp), scale.Px(3)) && Near(scale.Px(3), 2), "a wider border follows its width");
}

// Insets convert each side on its own, so sides of different sizes keep their own rounding.
void TestInsets() {
  const auto scale = Scale720();
  const auto insets = ToPixels(scale, DpInsets{4_dp, 16_dp, 24_dp, 32_dp});
  Check(Near(insets.left, scale.Px(4)) && Near(insets.top, scale.Px(16)) && Near(insets.right, scale.Px(24)) &&
            Near(insets.bottom, scale.Px(32)),
        "each side converts on its own");
  Check(DpInsets::All(8_dp) == DpInsets{8_dp, 8_dp, 8_dp, 8_dp}, "All fills every side");
  Check(DpInsets::Symmetric(8_dp, 4_dp) == DpInsets{8_dp, 4_dp, 8_dp, 4_dp}, "Symmetric pairs the sides");
}

// Design units add, subtract, scale and compare as lengths.
void TestArithmetic() {
  static_assert(16_dp + 8_dp == 24_dp);
  static_assert(16_dp - 8_dp == 8_dp);
  static_assert(-(8_dp) == Dp(-8));
  static_assert(8_dp * 2.0f == 16_dp && 2.0f * 8_dp == 16_dp);
  static_assert(16_dp / 2.0f == 8_dp);
  static_assert(8_dp < 16_dp && 16_dp > 8_dp);
  static_assert(1.5_dp == Dp(1.5f));
  Dp length = 8_dp;
  length += 8_dp;
  Check(length == 16_dp, "+= adds");
  length -= 4_dp;
  Check(length == 12_dp, "-= subtracts");
}

// A frame with the objects it borrows, so each test can start frames without repeating them.
struct FrameRig {
  Theme theme = DEFAULT_THEME;
  ScaleContext scale;
  UiFonts fonts;
  InteractionStore store;
  UiFrame frame;

  // The rig defaults to 720p, where a scale of 2/3 shows rounding.
  explicit FrameRig(float width = WIDTH_720P, float height = HEIGHT_720P)
      : scale(DEFAULT_THEME.scale, width, height) {}

  void Begin() { frame.Begin(theme, scale, fonts, store, UiClock::now()); }
};

constexpr int BUILD_COUNT = 200;
constexpr size_t LABEL_REPEAT = 40;

// Adds BUILD_COUNT nodes, each adopting up to two of the nodes that have no parent yet, and
// interns a label for each. A node has one parent, so each is adopted once. The labels total about
// 8 KB, enough to span more than one text chunk. Returns the last node's handle, the tree's root.
Element BuildTree(UiFrame &frame) {
  std::vector<Element> roots;
  for (int i = 0; i < BUILD_COUNT; ++i) {
    const std::string label = "label " + std::to_string(i) + std::string(LABEL_REPEAT, 'x');
    (void)frame.Intern(label);
    const size_t adopted = std::min<size_t>(2, roots.size());
    const Element node = FrameInternals::AddNode(frame, NodeKind::Box,
                                                 std::span<const Element>(roots).last(adopted));
    Check(!node.IsNone(), "the tree builds without a refused node");
    roots.resize(roots.size() - adopted);
    roots.push_back(node);
  }
  return roots.back();
}

// Beginning a frame keeps the storage's capacity, so a steady-state frame allocates nothing. The
// first build must have grown it from empty, or an unchanged size would prove nothing. Checking
// right after Begin catches storage that is freed and then regrown to the same size by the next
// build. The second build is checked for content, so an empty frame can't pass either.
void TestBeginKeepsCapacity() {
  FrameRig rig;
  rig.Begin();
  Check(rig.frame.ReservedBytes() == 0, "a new frame reserves nothing");
  const Element last = BuildTree(rig.frame);
  const size_t grown = rig.frame.ReservedBytes();
  Check(grown > 0, "the first build grew the storage");
  Check(FrameInternals::Find(rig.frame, last) != nullptr, "the first build produced its nodes");

  rig.Begin();
  Check(rig.frame.ReservedBytes() == grown, "Begin keeps the capacity");
  const Element lastAgain = BuildTree(rig.frame);
  Check(FrameInternals::Find(rig.frame, lastAgain) != nullptr, "the second build produced its nodes");
  Check(rig.frame.ReservedBytes() == grown, "a repeat build needs no more storage");
}

// A handle from an earlier frame is stale. The new frame has a node at the same index, so range
// can't be what rejects it, and a fresh handle at that index is found.
void TestStaleHandle() {
  FrameRig rig;
  rig.Begin();
  const Element old = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  Check(FrameInternals::Find(rig.frame, old) != nullptr, "a handle is found in its own frame");

  rig.Begin();
  const Element fresh = FrameInternals::AddNode(rig.frame, NodeKind::Text);
  Check(FrameInternals::Find(rig.frame, fresh) != nullptr, "a fresh handle is found");
  Check(FrameInternals::Find(rig.frame, fresh)->kind == NodeKind::Text, "and it is the new node");
  Check(FrameInternals::Find(rig.frame, old) == nullptr, "the old handle at the same index is not");
  Check(FrameInternals::Check(rig.frame, old) == ElementError::Stale, "and the reason is staleness");
}

// A handle past the last node is out of range, and the last node is still found.
void TestOutOfRangeHandle() {
  FrameRig rig;
  rig.Begin();
  (void)FrameInternals::AddNode(rig.frame, NodeKind::Box);
  const uint32_t serial = FrameTestHooks::Serial(rig.frame);
  Check(FrameInternals::Find(rig.frame, FrameTestHooks::Forge(0, serial)) != nullptr, "the last node is in range");
  const Element beyond = FrameTestHooks::Forge(1, serial);
  Check(FrameInternals::Find(rig.frame, beyond) == nullptr, "one past the last node is not");
  Check(FrameInternals::Check(rig.frame, beyond) == ElementError::OutOfRange, "and the reason is range");
}

// Children come back as the handles given, in order, and each node sees only its own run.
void TestChildrenKeepOrder() {
  FrameRig rig;
  rig.Begin();
  const Element a = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  const Element b = FrameInternals::AddNode(rig.frame, NodeKind::Text);
  const Element c = FrameInternals::AddNode(rig.frame, NodeKind::Image);
  const std::array<Element, 3> first{a, b, c};
  const Element d = FrameInternals::AddNode(rig.frame, NodeKind::Image);
  const Element e = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  const std::array<Element, 2> second{d, e};
  const Element parent = FrameInternals::AddNode(rig.frame, NodeKind::Box, first);
  const Element other = FrameInternals::AddNode(rig.frame, NodeKind::Box, second);
  const Element childless = FrameInternals::AddNode(rig.frame, NodeKind::Box);

  const auto kids = FrameInternals::Children(rig.frame, *FrameInternals::Find(rig.frame, parent));
  Check(kids.size() == 3, "the parent has three children");
  Check(FrameInternals::Find(rig.frame, kids[0])->kind == NodeKind::Box, "the first child is first");
  Check(FrameInternals::Find(rig.frame, kids[1])->kind == NodeKind::Text, "the second child is second");
  Check(FrameInternals::Find(rig.frame, kids[2])->kind == NodeKind::Image, "the third child is third");

  const auto otherKids = FrameInternals::Children(rig.frame, *FrameInternals::Find(rig.frame, other));
  Check(otherKids.size() == 2 && FrameInternals::Find(rig.frame, otherKids[0])->kind == NodeKind::Image &&
            FrameInternals::Find(rig.frame, otherKids[1])->kind == NodeKind::Box,
        "a later node sees its own run, not the earlier one");
  Check(FrameInternals::Children(rig.frame, *FrameInternals::Find(rig.frame, childless)).empty(),
        "a node given no children has none");
}

// Text interned early still reads correctly after later text needs new chunks. Each string is
// 3000 bytes, so two do not fit one 4 KB chunk, and one of 10000 bytes needs a chunk of its own.
// The views must point at the frame's copy, or they would pass by echoing the source. Chunks are
// kept and reset by Begin, so the same short text lands at the same address next frame. One
// growing std::string would have reallocated during the growth, so its first byte would sit
// elsewhere. That makes the address a deterministic check that needs no help from the allocator.
void TestInternSurvivesGrowth() {
  constexpr size_t MEDIUM = 3000;
  constexpr size_t HUGE_TEXT = 10000;
  FrameRig rig;
  rig.Begin();
  const std::string early = "early text";
  const std::string_view firstFrameView = rig.frame.Intern(early);
  Check(firstFrameView == early && firstFrameView.data() != early.data(), "an interned view is a copy");

  const size_t before = rig.frame.ReservedBytes();
  std::vector<std::string> sources;
  std::vector<std::string_view> views;
  for (const char fill : {'a', 'b', 'c'}) {
    sources.emplace_back(MEDIUM, fill);
    views.push_back(rig.frame.Intern(sources.back()));
  }
  sources.emplace_back(HUGE_TEXT, 'd');
  views.push_back(rig.frame.Intern(sources.back()));
  Check(rig.frame.ReservedBytes() >= before + MEDIUM + HUGE_TEXT, "the later text needed new chunks");

  Check(firstFrameView == early, "the early view is intact after growth");
  for (size_t i = 0; i < views.size(); ++i) {
    Check(views[i] == sources[i], "each later view is intact too");
  }
  Check(rig.frame.Intern("").empty(), "empty text interns to nothing");

  const char *firstByte = firstFrameView.data();
  rig.Begin();
  const std::string_view nextFrameView = rig.frame.Intern(early);
  Check(nextFrameView == early, "the text reads correctly in the next frame");
  Check(nextFrameView.data() == firstByte, "the first chunk did not move during growth");
}

// A default handle is none, and one a frame made is not. A live frame doesn't find a none handle
// even though it has a node at index 0.
void TestDefaultElementIsNone() {
  Check(Element().IsNone(), "a default handle is none");
  FrameRig rig;
  rig.Begin();
  const Element made = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  Check(!made.IsNone(), "a made handle is not none");
  Check(FrameInternals::Find(rig.frame, Element()) == nullptr, "a none handle finds no node");
  Check(FrameInternals::Find(rig.frame, made) != nullptr, "while the node at index 0 exists");
}

// When the frame serial wraps, it skips zero, which belongs to the none handle. Without the skip
// a none handle would match the frame and find node 0. The made handle is the baseline that the
// frame still works after the wrap.
void TestSerialWrapSkipsNone() {
  FrameRig rig;
  rig.Begin();
  FrameTestHooks::SetSerial(rig.frame, UINT32_MAX);
  rig.Begin();
  const Element made = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  Check(FrameTestHooks::Serial(rig.frame) != 0, "the wrapped serial is not the none serial");
  Check(FrameInternals::Find(rig.frame, made) != nullptr, "a handle made after the wrap is found");
  Check(FrameInternals::Find(rig.frame, Element()) == nullptr, "a none handle still finds nothing");
}

// A node has one parent. A second parent is refused with a none handle, the first parent keeps
// its children, the contested child still names the first parent, and the refusal leaves no
// half-built state, so the uncontested child of the refused parent is free to be adopted again.
// A child listed twice and a span from the frame's own child storage are refused the same way.
void TestOneParentPerNode() {
  FrameRig rig;
  rig.Begin();
  const Element shared = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  const Element spare = FrameInternals::AddNode(rig.frame, NodeKind::Text);
  const std::array<Element, 1> firstKids{shared};
  const Element first = FrameInternals::AddNode(rig.frame, NodeKind::Box, firstKids);
  Check(!first.IsNone(), "the first parent is accepted");

  const std::array<Element, 2> secondKids{spare, shared};
  const Element second = FrameInternals::AddNode(rig.frame, NodeKind::Box, secondKids);
  Check(second.IsNone(), "a second parent for the same child is refused");
  const auto kids = FrameInternals::Children(rig.frame, *FrameInternals::Find(rig.frame, first));
  Check(kids.size() == 1 && FrameInternals::Find(rig.frame, kids[0]) == FrameInternals::Find(rig.frame, shared),
        "the first parent keeps its child");
  constexpr uint32_t FIRST_INDEX = 2; // shared and spare were added before it
  Check(FrameInternals::Find(rig.frame, shared)->parent == FIRST_INDEX, "the child still names the first parent");
  Check(FrameInternals::Find(rig.frame, spare)->parent == Node::NO_PARENT, "the refusal left the other child free");

  const std::array<Element, 1> spareKids{spare};
  Check(!FrameInternals::AddNode(rig.frame, NodeKind::Box, spareKids).IsNone(), "so it can be adopted afterwards");

  const Element loose = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  const std::array<Element, 2> twice{loose, loose};
  Check(FrameInternals::AddNode(rig.frame, NodeKind::Box, twice).IsNone(), "a child listed twice is refused");
  Check(FrameInternals::Find(rig.frame, loose)->parent == Node::NO_PARENT, "and stays free");

  const auto ownStorage = FrameInternals::Children(rig.frame, *FrameInternals::Find(rig.frame, first));
  Check(FrameInternals::AddNode(rig.frame, NodeKind::Box, ownStorage).IsNone(),
        "a span from the frame's own child storage is refused");
  Check(FrameInternals::Children(rig.frame, *FrameInternals::Find(rig.frame, first)).size() == 1,
        "and the storage is unchanged");
}

// A child that is none or from an earlier frame is refused, and the valid child listed before it
// is left free. Without the check the lookup would return null and be dereferenced.
void TestInvalidChildRefused() {
  FrameRig rig;
  rig.Begin();
  const Element oldChild = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  rig.Begin();
  // A free node at the stale handle's index, so only the serial can tell the two apart.
  (void)FrameInternals::AddNode(rig.frame, NodeKind::Text);
  const Element valid = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  const std::array<Element, 1> onlyStale{oldChild};
  Check(FrameInternals::AddNode(rig.frame, NodeKind::Box, onlyStale).IsNone(), "a stale child is refused");
  const std::array<Element, 2> withNone{valid, Element()};
  Check(FrameInternals::AddNode(rig.frame, NodeKind::Box, withNone).IsNone(), "a none child is refused");
  Check(FrameInternals::Find(rig.frame, valid)->parent == Node::NO_PARENT, "and the valid child stays free");
  const std::array<Element, 1> justValid{valid};
  Check(!FrameInternals::AddNode(rig.frame, NodeKind::Box, justValid).IsNone(), "so it can still be adopted");
}

// Using a frame before Begin gives none handles, and a none handle finds nothing.
void TestBeforeBegin() {
  FrameRig rig;
  const Element early = FrameInternals::AddNode(rig.frame, NodeKind::Box);
  Check(early.IsNone(), "AddNode before Begin gives a none handle");
  Check(FrameInternals::Find(rig.frame, early) == nullptr, "and a none handle finds nothing");
  rig.Begin();
  Check(!FrameInternals::AddNode(rig.frame, NodeKind::Box).IsNone(), "while the same call after Begin works");
}

// True when Begin accepts a theme passed as `ThemeArg`.
template <typename ThemeArg>
concept BeginAccepts = requires(UiFrame &frame, ThemeArg &&theme, const ScaleContext &scale, const UiFonts &fonts,
                                InteractionStore &store, UiClock::time_point now) {
  frame.Begin(std::forward<ThemeArg>(theme), scale, fonts, store, now);
};

// A temporary theme would dangle, so Begin must not accept one. A named theme is the baseline
// that shows the check fails on the temporary and not on some other argument.
static_assert(BeginAccepts<Theme &>, "Begin accepts a named theme");
static_assert(BeginAccepts<const Theme &>, "Begin accepts a named const theme");
static_assert(!BeginAccepts<Theme>, "Begin refuses a temporary theme");

// The three outputs the box tests run at. At 1080p the scale is 1, so a missing or doubled
// conversion goes unseen there and shows at the other two.
struct Output {
  float width;
  float height;
};
constexpr std::array<Output, 3> OUTPUTS{{{1280, 720}, {1920, 1080}, {3840, 2160}}};
constexpr float INFINITE_SPACE = std::numeric_limits<float>::infinity();
constexpr float ONE_PIXEL = 1.0f;

// An ImGui context for draw lists, with anti-aliasing off so shapes have exact edges.
class ImGuiScope {
public:
  // Fonts load only when asked for, since the box tests draw no text and don't need them.
  explicit ImGuiScope(bool withFonts = false) {
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.DisplaySize = {3840, 2160};
    io.DeltaTime = 1.0f / 60.0f;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.IniFilename = nullptr;
    ImGui::GetStyle().AntiAliasedFill = false;
    ImGui::GetStyle().AntiAliasedLines = false;
    if (withFonts) {
      fonts = LoadUiFonts(*io.Fonts, KYTY_TEST_FONT_REGULAR, KYTY_TEST_FONT_BOLD);
    }
    ImGui::NewFrame();
  }
  UiFonts fonts;
  ~ImGuiScope() {
    ImGui::EndFrame();
    ImGui::DestroyContext();
  }
  ImGuiScope(const ImGuiScope &) = delete;
  ImGuiScope &operator=(const ImGuiScope &) = delete;
};

// A draw list of its own that shares the context's draw list data, so two lists can be compared.
std::unique_ptr<ImDrawList> NewDrawList() {
  auto list = std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData());
  list->_ResetForNewFrame();
  list->PushClipRectFullScreen();
  return list;
}

// True when two lists hold the same vertices, UVs, colours and indices, exactly.
bool SameDrawing(const ImDrawList &a, const ImDrawList &b) {
  if (a.VtxBuffer.Size != b.VtxBuffer.Size || a.IdxBuffer.Size != b.IdxBuffer.Size) {
    return false;
  }
  for (int i = 0; i < a.VtxBuffer.Size; ++i) {
    const ImDrawVert &left = a.VtxBuffer[i];
    const ImDrawVert &right = b.VtxBuffer[i];
    if (left.pos.x != right.pos.x || left.pos.y != right.pos.y || left.uv.x != right.uv.x ||
        left.uv.y != right.uv.y || left.col != right.col) {
      return false;
    }
  }
  for (int i = 0; i < a.IdxBuffer.Size; ++i) {
    if (a.IdxBuffer[i] != b.IdxBuffer[i]) {
      return false;
    }
  }
  return true;
}

// True when two rectangles agree edge by edge to within `slack`.
bool NearRect(const Rect &a, const Rect &b, float slack = ONE_PIXEL) {
  return std::fabs(a.x - b.x) <= slack && std::fabs(a.y - b.y) <= slack && std::fabs(a.width - b.width) <= slack &&
         std::fabs(a.height - b.height) <= slack;
}

// A style with every facet on, so dropping any one of them changes the drawing.
Style FullStyle() { return {Rgb(0x112233), Rgb(0xFFFFFF), 2_dp, 12_dp, DEFAULT_THEME.elevation.level1}; }

// The older DrawBox style for the same look, written with plain numbers. DrawBox takes reference
// pixels, and 12 and 2 are FullStyle's radius and border width.
BoxStyle FullBoxStyle() { return {Rgb(0x112233), 12.0f, Rgb(0xFFFFFF), 2.0f, DEFAULT_THEME.elevation.level1}; }

// Builds a childless box in a fresh frame, arranges it at `rect` and paints it into `list`.
void PaintBox(FrameRig &rig, const Style &style, const Rect &rect, ImDrawList &list, float opacity = 1.0f) {
  rig.Begin();
  const Element box = Box(rig.frame, {}, style);
  (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, box, rect), "the box arranges");
  Paint(rig.frame, box, list, {.opacity = opacity});
}

// A box paints exactly what DrawBox paints at the same rectangle, at every output size. The
// fill-only box is the baseline that shows the border and shadow add vertices, so a box missing
// them can't pass by drawing nothing extra. Passing the radius already in pixels would round
// twice, which the 720p and 4K outputs show and 1080p hides.
void TestBoxMatchesDrawBox() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 200, 100};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    auto expected = NewDrawList();
    rig.Begin();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    auto actual = NewDrawList();
    PaintBox(rig, FullStyle(), rect, *actual);
    Check(expected->VtxBuffer.Size > 0, "the reference drew something");
    Check(SameDrawing(*actual, *expected), "a box draws what DrawBox draws");

    Style fillOnly = FullStyle();
    fillOnly.border_width = 0_dp;
    fillOnly.elevation = {};
    auto plain = NewDrawList();
    PaintBox(rig, fillOnly, rect, *plain);
    Check(plain->VtxBuffer.Size < actual->VtxBuffer.Size, "the border and shadow add vertices");
    Check(!SameDrawing(*plain, *expected), "so a fill-only box differs from the full one");
  }
}

// Opacity fades the box and its children, as it does for DrawBox. The half-opacity drawing differs
// from the opaque one, or the comparison would hold with opacity ignored on both sides.
void TestPaintPassesOpacity() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 200, 100};
  FrameRig rig(1920, 1080);
  auto opaque = NewDrawList();
  PaintBox(rig, FullStyle(), rect, *opaque);
  auto faded = NewDrawList();
  PaintBox(rig, FullStyle(), rect, *faded, 0.5f);
  auto expected = NewDrawList();
  DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle(), 0.5f);
  Check(SameDrawing(*faded, *expected), "a faded box matches a faded DrawBox");
  Check(!SameDrawing(*faded, *opaque), "and differs from the opaque one");
}

// Children paint after their box and inside it, each as a box of its own, at the same opacity.
void TestPaintReachesChildren() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Style childStyle{Rgb(0xAA5500), {}, 0_dp, 4_dp, {}};
  const Layout fixed{.item = {.width = 100_dp, .height = 40_dp}};
  const std::array<Element, 1> kids{Box(rig.frame, fixed, childStyle)};
  const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, FullStyle(), kids);
  (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
  const Rect rect{100, 50, 300, 200};
  Check(Arrange(rig.frame, parent, rect), "the tree arranges");
  auto actual = NewDrawList();
  Paint(rig.frame, parent, *actual, {.opacity = 0.75f});

  auto expected = NewDrawList();
  DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle(), 0.75f);
  DrawBox(*expected, rig.frame.Ui(), Arranged(rig.frame, kids[0]), {Rgb(0xAA5500), 4.0f, {}, 0.0f, {}}, 0.75f);
  Check(SameDrawing(*actual, *expected), "the box and then its child are drawn");
  auto parentOnly = NewDrawList();
  DrawBox(*parentOnly, rig.frame.Ui(), rect, FullBoxStyle(), 0.75f);
  Check(actual->VtxBuffer.Size > parentOnly->VtxBuffer.Size, "the child adds vertices to its parent's");
}

// Padding insets the child by the converted padding. The child is a fixed 100 by 40 dp box, so
// its size is also checked. With no padding it sits at the rectangle's corner, the baseline that
// shows the inset comes from the padding and not from the placement.
void TestPaddingInsetsChild() {
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    const Rect rect{20, 30, 600, 300};
    const auto build = [&](Dp padding) {
      rig.Begin();
      const Layout fixed{.item = {.width = 100_dp, .height = 40_dp}};
      const std::array<Element, 1> kids{Box(rig.frame, fixed, {})};
      const Element root = Box(rig.frame, {.padding = DpInsets::All(padding)}, {}, kids);
      (void)Measure(rig.frame, root, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, root, rect), "the padded box arranges");
      return Arranged(rig.frame, kids[0]);
    };
    const float inset = ToPixels(rig.scale, 16_dp);
    const Rect size{0, 0, ToPixels(rig.scale, 100_dp), ToPixels(rig.scale, 40_dp)};
    const Rect child = build(16_dp);
    const Rect inner = rect.Inset(EdgeInsets::All(inset));
    Check(NearRect(child, {inner.x, inner.y, size.width, size.height}), "the child sits inside the padding");
    const Rect bare = build(0_dp);
    Check(NearRect(bare, {rect.x, rect.y, size.width, size.height}), "without padding it sits at the corner");
    Check(std::fabs(child.x - bare.x - inset) <= ONE_PIXEL && inset > 0, "so the padding is what moved it");
  }
}

// Checks a row's two children against a hand-built SolveFlex of the same row, in the same units.
void CheckRowAgainstSolver(const FrameRig &rig, const std::array<Element, 2> &kids, const Rect &rect,
                           const DpInsets &padding) {
  const auto &scale = rig.scale;
  const FlexStyle style{.direction = FlexDirection::Row,
                        .justify = Justify::Start,
                        .align = Align::Center,
                        .gap = ToPixels(scale, 10_dp),
                        .padding = ToPixels(scale, padding)};
  const std::vector<FlexItem> items = {{.size = {ToPixels(scale, 100_dp), ToPixels(scale, 40_dp)}},
                                       {.size = {0, ToPixels(scale, 20_dp)}, .grow = 1.0f}};
  const auto expected = SolveFlex(rect, style, items);
  Check(Arranged(rig.frame, kids[0]) == expected[0], "the fixed box lands where the solver puts it");
  Check(Arranged(rig.frame, kids[1]) == expected[1], "and the growing one fills the rest");
  Check(expected[1].width > 0 && expected[0].y > rect.y + style.padding.top,
        "the baseline grows and centres, so those inputs matter");
}

// A row of a fixed box and a growing one, with gap, padding, centred alignment and a container
// wider and taller than its content, laid out exactly as a hand-built SolveFlex does. The
// container's height of 100 dp against 40 dp of content makes centring show. Measured with the
// width on offer, the grower already holds its share. Measured with no limit it holds nothing, so
// only the solver's grow can fill the rectangle. Both must end the same.
void TestRowMatchesHandBuiltFlex() {
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const DpInsets padding{8_dp, 6_dp, 12_dp, 10_dp};
    const Layout fixed{.item = {.width = 100_dp, .height = 40_dp}};
    const Layout grower{.item = {.height = 20_dp, .grow = 1.0f}};
    const std::array<Element, 2> kids{Box(rig.frame, fixed, {}), Box(rig.frame, grower, {})};
    const Layout rowLayout{.align = Align::Center, .gap = 10_dp, .padding = padding};
    const Element root = Box(rig.frame, rowLayout, {}, kids);
    const Rect rect{10, 20, ToPixels(rig.scale, 600_dp), ToPixels(rig.scale, 100_dp)};
    for (const float offered : {rect.width, INFINITE_SPACE}) {
      (void)Measure(rig.frame, root, {offered, rect.height});
      Check(Arrange(rig.frame, root, rect), "the row arranges");
      CheckRowAgainstSolver(rig, kids, rect, padding);
    }
  }
}

// Measuring and arranging a tree reserves scratch space on the first frame and keeps it on the
// next, so layout allocates nothing in steady state. The reserve before Arrange is the baseline
// that shows the solver's scratch is part of what is kept.
void TestLayoutScratchKeepsCapacity() {
  FrameRig rig;
  const auto build = [&]() {
    rig.Begin();
    const Layout cell{.item = {.width = 20_dp, .height = 20_dp, .grow = 1.0f}};
    const std::array<Element, 3> kids{Box(rig.frame, cell, {}), Box(rig.frame, cell, {}), Box(rig.frame, cell, {})};
    return Box(rig.frame, {.gap = 4_dp}, {}, kids);
  };
  const Rect rect{0, 0, 400, 100};
  Element root = build();
  const size_t beforeLayout = rig.frame.ReservedBytes();
  (void)Measure(rig.frame, root, {rect.width, rect.height});
  Check(Arrange(rig.frame, root, rect), "the tree arranges");
  const size_t afterLayout = rig.frame.ReservedBytes();
  Check(afterLayout > beforeLayout, "arranging reserved scratch space");

  root = build();
  Check(rig.frame.ReservedBytes() == afterLayout, "Begin keeps the scratch space");
  (void)Measure(rig.frame, root, {rect.width, rect.height});
  Check(Arrange(rig.frame, root, rect), "the second tree arranges");
  Check(rig.frame.ReservedBytes() == afterLayout, "and a repeat layout needs no more");
}

// A row of two growers whose natural widths are not zero, so growing is what splits the space and
// not a zero start that any share would satisfy. A 200 px row with naturals of 100 and 0 gives 150
// and 50, and with 150 and 0 gives 175 and 25, as SolveFlex does. Measured with the space given
// and measured with none, the row arranges the same. The measured width of the assigned lengths
// is checked too, since that is what a child's own content lays out in.
void TestGrowersWithNaturalSizes() {
  FrameRig rig(1920, 1080);
  constexpr float ROW_WIDTH = 200;
  struct Case {
    Dp natural;
    float first;
    float second;
  };
  for (const Case &item : {Case{100_dp, 150, 50}, Case{150_dp, 175, 25}}) {
    rig.Begin();
    const Layout first{.item = {.width = item.natural, .height = 20_dp, .grow = 1.0f}};
    const Layout second{.item = {.height = 20_dp, .grow = 1.0f}};
    const std::array<Element, 2> kids{Box(rig.frame, first, {}), Box(rig.frame, second, {})};
    const Element row = Box(rig.frame, {}, {}, kids);
    const Rect rect{0, 0, ROW_WIDTH, 40};
    const std::vector<FlexItem> items = {{.size = {item.natural.value, 20}, .grow = 1.0f}, {.size = {0, 20}, .grow = 1.0f}};
    const auto expected = SolveFlex(rect, {}, items);
    Check(expected[0].width == item.first && expected[1].width == item.second, "the solver gives the stated split");
    for (const float offered : {ROW_WIDTH, INFINITE_SPACE}) {
      (void)Measure(rig.frame, row, {offered, 40});
      Check(Arrange(rig.frame, row, rect), "the row arranges");
      Check(Arranged(rig.frame, kids[0]).width == item.first, "the first grower gets its split");
      Check(Arranged(rig.frame, kids[1]).width == item.second, "the second grower gets its split");
      Check(Arranged(rig.frame, kids[1]).x == expected[1].x, "and starts where the solver puts it");
    }
    (void)Measure(rig.frame, row, {ROW_WIDTH, 40});
    Check(FrameInternals::Find(rig.frame, kids[0])->measured.x == item.first, "the assigned length is the split");
    Check(FrameInternals::Find(rig.frame, kids[0])->natural.x == item.natural.value, "and the natural width is kept");
  }
}

// A box whose natural width is `width` because of its content, with no main-axis size of its own,
// so it may shrink. A box given the width directly would be fixed and would not.
Element Content(FrameRig &rig, Dp width, std::optional<Dp> minMain) {
  const Layout fixed{.item = {.width = width, .height = 20_dp}};
  const std::array<Element, 1> inner{Box(rig.frame, fixed, {})};
  // The minimum defaults to zero here, so these children shrink below their content. The tests
  // using this helper are about the shrink maths, and a box's automatic minimum would stop it.
  return Box(rig.frame, {.item = {.height = 20_dp, .min_main = minMain.value_or(0_dp)}}, {}, inner);
}

// A child with a fixed width keeps it when its row overflows, and its sibling without one shrinks
// to what is left. The same row with the first child's width coming from its content is the
// baseline, where both shrink, so the fixed width is what held the first one.
void TestFixedWidthChildDoesNotShrink() {
  FrameRig rig(1920, 1080);
  const auto widths = [&](bool fixedFirst) {
    rig.Begin();
    const Layout fixed{.item = {.width = 150_dp, .height = 20_dp}};
    const std::array<Element, 2> kids{fixedFirst ? Box(rig.frame, fixed, {}) : Content(rig, 150_dp, std::nullopt),
                                      Content(rig, 150_dp, std::nullopt)};
    const Element row = Box(rig.frame, {}, {}, kids);
    (void)Measure(rig.frame, row, {200, 40});
    Check(Arrange(rig.frame, row, {0, 0, 200, 40}), "the row arranges");
    return std::array<float, 2>{Arranged(rig.frame, kids[0]).width, Arranged(rig.frame, kids[1]).width};
  };
  Check((widths(false) == std::array<float, 2>{100, 100}), "without a fixed width both shrink");
  Check((widths(true) == std::array<float, 2>{150, 50}), "a fixed width holds and its sibling gives way");
  // The automatic minimum would hold a fixed width on its own, so a minimum of zero shows that
  // the fixed width itself stops the shrinking.
  rig.Begin();
  const Layout loose{.item = {.width = 150_dp, .height = 20_dp, .min_main = 0_dp}};
  const std::array<Element, 2> kids{Box(rig.frame, loose, {}), Content(rig, 150_dp, std::nullopt)};
  const Element row = Box(rig.frame, {}, {}, kids);
  (void)Measure(rig.frame, row, {200, 40});
  Check(Arrange(rig.frame, row, {0, 0, 200, 40}), "the row arranges");
  Check(Arranged(rig.frame, kids[0]).width == 150, "a fixed width holds even with a minimum of zero");
}

// Children that overflow their row shrink in proportion to their length, down to their own
// minimums. Two 150 px children in 200 px take 100 each. With a minimum of 120 on the second, it
// stops there and the first takes the rest. A row with room is the baseline that shrinks nothing.
void TestOverflowShrinks() {
  FrameRig rig(1920, 1080);
  const auto arrange = [&](float rowWidth, std::optional<Dp> secondMin) {
    rig.Begin();
    const std::array<Element, 2> kids{Content(rig, 150_dp, std::nullopt), Content(rig, 150_dp, secondMin)};
    const Element row = Box(rig.frame, {}, {}, kids);
    (void)Measure(rig.frame, row, {rowWidth, 40});
    Check(Arrange(rig.frame, row, {0, 0, rowWidth, 40}), "the row arranges");
    return std::array<float, 2>{Arranged(rig.frame, kids[0]).width, Arranged(rig.frame, kids[1]).width};
  };
  Check((arrange(300, std::nullopt) == std::array<float, 2>{150, 150}), "with room nothing shrinks");
  Check((arrange(200, std::nullopt) == std::array<float, 2>{100, 100}), "overflow shrinks both evenly");
  Check((arrange(200, 120_dp) == std::array<float, 2>{80, 120}), "a minimum stops one and the other gives more");
}

// Arranging starts from each child's natural length and not from the length Measure assigned, so
// a row measured in 200 px and arranged in 300 px lays out as if it had been measured in 300. The
// two 150 px children shrank to 100 each in the 200 px measure, and must be 150 again with room.
void TestArrangeUsesNaturalLengths() {
  FrameRig rig(1920, 1080);
  rig.Begin();
  const std::array<Element, 2> kids{Content(rig, 150_dp, std::nullopt), Content(rig, 150_dp, std::nullopt)};
  const Element row = Box(rig.frame, {}, {}, kids);
  (void)Measure(rig.frame, row, {200, 40});
  Check(FrameInternals::Find(rig.frame, kids[0])->measured.x == 100, "the 200 px measure shrank the child");
  Check(Arrange(rig.frame, row, {0, 0, 300, 40}), "the row arranges");
  Check(Arranged(rig.frame, kids[0]).width == 150 && Arranged(rig.frame, kids[1]).width == 150,
        "and the 300 px arrangement is from natural lengths");
}

// The lengths scratch is a stack, so measuring a nested box again does not overwrite the lengths
// its parent is still reading. The first child grows, so it is measured again, and its own two
// children write their lengths while the parent still has the second child's 50 px to read.
void TestNestedMeasuresDoNotClobberLengths() {
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Layout small{.item = {.width = 10_dp, .height = 10_dp}};
  const std::array<Element, 2> inner{Box(rig.frame, small, {}), Box(rig.frame, small, {})};
  const Layout grower{.item = {.height = 10_dp, .grow = 1.0f}};
  const Layout fixed{.item = {.width = 50_dp, .height = 10_dp}};
  const std::array<Element, 2> kids{Box(rig.frame, grower, {}, inner), Box(rig.frame, fixed, {})};
  const Element row = Box(rig.frame, {}, {}, kids);
  const Vec2 size = Measure(rig.frame, row, {300, 40});
  Check(FrameInternals::Find(rig.frame, kids[1])->measured.x == 50, "the second child keeps its 50 px");
  Check(size.x == 300, "and the row adds up to the space");
}

// A child's own maximum limits what its children get. The box is capped at 200 dp in a 400 dp row,
// so its growing child shares 200 dp and not 400. Without the cap the same child gets the row.
void TestChildLimitsReachGrandchildren() {
  FrameRig rig(1920, 1080);
  const auto measureGrower = [&](std::optional<Dp> cap) {
    rig.Begin();
    const Layout fixed{.item = {.width = 50_dp, .height = 10_dp}};
    const Layout grower{.item = {.height = 10_dp, .grow = 1.0f}};
    const std::array<Element, 2> inner{Box(rig.frame, fixed, {}), Box(rig.frame, grower, {})};
    const Layout capped{.item = {.height = 10_dp, .grow = 1.0f, .max_main = cap}};
    const std::array<Element, 1> outer{Box(rig.frame, capped, {}, inner)};
    const Element row = Box(rig.frame, {}, {}, outer);
    (void)Measure(rig.frame, row, {400, 40});
    return std::array<float, 2>{FrameInternals::Find(rig.frame, outer[0])->measured.x,
                                FrameInternals::Find(rig.frame, inner[1])->measured.x};
  };
  Check((measureGrower(200_dp) == std::array<float, 2>{200, 150}), "a capped box limits its grower to what is left");
  Check((measureGrower(std::nullopt) == std::array<float, 2>{400, 350}), "without the cap the grower gets the row");
}

// A child with its own cross size is not stretched, so a 24 dp icon in a stretched 64 dp row stays
// 24 dp. A sibling without a size is the baseline that does stretch, so the row is stretching.
void TestExplicitCrossIsNotStretched() {
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Layout icon{.item = {.width = 24_dp, .height = 24_dp}};
  const Layout label{.item = {.width = 40_dp}};
  const std::array<Element, 2> kids{Box(rig.frame, icon, {}), Box(rig.frame, label, {})};
  const Element row = Box(rig.frame, {.align = Align::Stretch}, {}, kids);
  (void)Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, row, {0, 0, 200, 64}), "the row arranges");
  Check(Arranged(rig.frame, kids[0]).height == 24, "the icon keeps its height");
  Check(Arranged(rig.frame, kids[1]).height == 64, "while the label stretches");
}

// A row of two columns, each holding two children. Laying out the first column solves a nested
// row of rectangles, which must not overwrite the outer row's rectangle for the second column.
void TestNestedSolvesDoNotClobber() {
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Layout cell{.item = {.width = 30_dp, .height = 10_dp}};
  const Layout column{.direction = FlexDirection::Column, .gap = 2_dp};
  const std::array<Element, 2> left{Box(rig.frame, cell, {}), Box(rig.frame, cell, {})};
  const std::array<Element, 2> right{Box(rig.frame, cell, {}), Box(rig.frame, cell, {})};
  const std::array<Element, 2> columns{Box(rig.frame, column, {}, left), Box(rig.frame, column, {}, right)};
  const Element row = Box(rig.frame, {.gap = 10_dp}, {}, columns);
  (void)Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, row, {0, 0, 200, 100}), "the tree arranges");
  Check(NearRect(Arranged(rig.frame, columns[0]), {0, 0, 30, 22}, 0), "the first column is at the start");
  Check(NearRect(Arranged(rig.frame, columns[1]), {40, 0, 30, 22}, 0), "the second column follows the gap");
  Check(NearRect(Arranged(rig.frame, left[1]), {0, 12, 30, 10}, 0), "the first column's second child");
  Check(NearRect(Arranged(rig.frame, right[0]), {40, 0, 30, 10}, 0), "the second column's first child");
  Check(NearRect(Arranged(rig.frame, right[1]), {40, 12, 30, 10}, 0), "the second column's second child");
}

// Arrange refuses a tree that was never measured, and accepts it once it has been.
void TestArrangeNeedsMeasure() {
  FrameRig rig;
  rig.Begin();
  const Element box = Box(rig.frame, {}, {});
  Check(!Arrange(rig.frame, box, {0, 0, 10, 10}), "an unmeasured box is refused");
  (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, box, {0, 0, 10, 10}), "a measured box arranges");
}

// A child's own align_self and max_main reach the solver. A sibling without them is the baseline
// that stays at the start and keeps growing, so the overrides can't be what the container does
// anyway. The row is measured with no limit, so the grower's cap is the solver's to apply.
void TestArrangeHonoursItemOverrides() {
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const auto px = [&](Dp length) { return ToPixels(rig.scale, length); };
    const Layout lowered{.item = {.width = 50_dp, .height = 20_dp, .align_self = Align::End}};
    const Layout plain{.item = {.width = 50_dp, .height = 20_dp}};
    const Layout capped{.item = {.height = 20_dp, .grow = 1.0f, .max_main = 150_dp}};
    const Layout free{.item = {.height = 20_dp, .grow = 1.0f}};
    const std::array<Element, 3> kids{Box(rig.frame, lowered, {}), Box(rig.frame, plain, {}), Box(rig.frame, capped, {})};
    const Element row = Box(rig.frame, {}, {}, kids);
    const Rect rect{0, 0, px(600_dp), px(100_dp)};
    (void)Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, row, rect), "the row arranges");
    Check(NearRect(Arranged(rig.frame, kids[0]), {0, px(100_dp) - px(20_dp), px(50_dp), px(20_dp)}),
          "align_self puts the first child at the bottom");
    Check(Arranged(rig.frame, kids[1]).y == 0, "while its sibling stays at the top");
    Check(Arranged(rig.frame, kids[2]).width == px(150_dp), "a capped grower stops at its maximum");

    rig.Begin();
    const std::array<Element, 1> open{Box(rig.frame, free, {})};
    const Element openRow = Box(rig.frame, {}, {}, open);
    (void)Measure(rig.frame, openRow, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, openRow, rect), "the open row arranges");
    Check(Arranged(rig.frame, open[0]).width == rect.width, "without a maximum the same grower fills the row");
  }
}

// Boxes nested in boxes are all arranged, each inside the padding of the one around it.
void TestArrangeRecurses() {
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Layout leafLayout{.item = {.width = 20_dp, .height = 20_dp}};
  const std::array<Element, 1> leaf{Box(rig.frame, leafLayout, {})};
  const std::array<Element, 1> middle{Box(rig.frame, {.padding = DpInsets::All(6_dp)}, {}, leaf)};
  const Element root = Box(rig.frame, {.padding = DpInsets::All(10_dp)}, {}, middle);
  (void)Measure(rig.frame, root, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, root, {100, 100, 200, 200}), "the tree arranges");
  Check(NearRect(Arranged(rig.frame, leaf[0]), {116, 116, 20, 20}), "the grandchild is inside both paddings");
  Check(NearRect(Arranged(rig.frame, middle[0]), {110, 110, 32, 32}), "and the middle box is inside the root's padding");
}

// A box wants its children's main lengths, the gaps and its padding, and the largest cross length
// plus padding. A column swaps the axes. Every size is built from converted tokens at 720p.
void TestMeasureIntrinsic() {
  FrameRig rig;
  const auto &scale = rig.scale;
  const auto px = [&](Dp length) { return ToPixels(scale, length); };
  rig.Begin();
  const Layout small{.item = {.width = 30_dp, .height = 10_dp}};
  const Layout large{.item = {.width = 50_dp, .height = 20_dp}};
  const std::array<Element, 2> rowKids{Box(rig.frame, small, {}), Box(rig.frame, large, {})};
  const Element row = Box(rig.frame, {.gap = 4_dp, .padding = DpInsets::All(5_dp)}, {}, rowKids);
  const Vec2 rowSize = Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Near(rowSize.x, px(30_dp) + px(50_dp) + px(4_dp) + 2 * px(5_dp)), "a row adds widths, a gap and padding");
  Check(Near(rowSize.y, px(20_dp) + 2 * px(5_dp)), "and takes the tallest child plus padding");

  const std::array<Element, 2> columnKids{Box(rig.frame, small, {}), Box(rig.frame, large, {})};
  const Layout columnLayout{.direction = FlexDirection::Column, .gap = 4_dp, .padding = DpInsets::All(5_dp)};
  const Element column = Box(rig.frame, columnLayout, {}, columnKids);
  const Vec2 columnSize = Measure(rig.frame, column, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Near(columnSize.x, px(50_dp) + 2 * px(5_dp)), "a column takes the widest child plus padding");
  Check(Near(columnSize.y, px(10_dp) + px(20_dp) + px(4_dp) + 2 * px(5_dp)), "and adds heights and a gap");

  const std::array<Element, 2> wideKids{Box(rig.frame, small, {}), Box(rig.frame, large, {})};
  const Layout wide{.gap = 4_dp, .padding = DpInsets::All(5_dp), .item = {.width = 200_dp}};
  const Vec2 wideSize = Measure(rig.frame, Box(rig.frame, wide, {}, wideKids), {INFINITE_SPACE, INFINITE_SPACE});
  Check(Near(wideSize.x, px(200_dp)), "an explicit width replaces the measured one");
  Check(Near(wideSize.y, rowSize.y), "and leaves the other axis alone");
}

// A child's minimum and maximum limit its main length in its container. The child is 10 dp wide
// and the container's width is the child's plus nothing else, so the baseline without limits
// measures 10 dp. Each limit then moves it, and 90 dp shrinks to the maximum.
void TestMeasureClampsMainAxis() {
  FrameRig rig;
  const auto px = [&](Dp length) { return ToPixels(rig.scale, length); };
  const auto measureRow = [&](Dp width, std::optional<Dp> minMain, std::optional<Dp> maxMain) {
    rig.Begin();
    const Layout child{.item = {.width = width, .height = 10_dp, .min_main = minMain, .max_main = maxMain}};
    const std::array<Element, 1> kids{Box(rig.frame, child, {})};
    return Measure(rig.frame, Box(rig.frame, {}, {}, kids), {INFINITE_SPACE, INFINITE_SPACE});
  };
  Check(Near(measureRow(10_dp, std::nullopt, std::nullopt).x, px(10_dp)), "without limits the child keeps its width");
  Check(Near(measureRow(10_dp, 40_dp, std::nullopt).x, px(40_dp)), "a minimum lifts a narrow child");
  Check(Near(measureRow(90_dp, std::nullopt, 20_dp).x, px(20_dp)), "a maximum caps a wide child");
  Check(Near(measureRow(90_dp, std::nullopt, std::nullopt).x, px(90_dp)), "which is 90 dp without the maximum");
}

// A growing child is measured with its share of what is on offer, so a row given 600 dp fills it.
// Measured with no limit the same row is only as wide as its content. A growing child measured
// without a limit would make both rows the same width.
void TestMeasureSharesSpaceWithGrowers() {
  FrameRig rig;
  const auto px = [&](Dp length) { return ToPixels(rig.scale, length); };
  const auto measureRow = [&](float offered) {
    rig.Begin();
    const Layout fixed{.item = {.width = 100_dp, .height = 40_dp}};
    const Layout grower{.item = {.height = 20_dp, .grow = 1.0f}};
    const std::array<Element, 2> kids{Box(rig.frame, fixed, {}), Box(rig.frame, grower, {})};
    return Measure(rig.frame, Box(rig.frame, {.gap = 10_dp}, {}, kids), {offered, INFINITE_SPACE});
  };
  Check(Near(measureRow(px(600_dp)).x, px(600_dp)), "a row offered 600 dp fills it");
  Check(Near(measureRow(INFINITE_SPACE).x, px(100_dp) + px(10_dp)), "an unbounded row is as wide as its content");
}

// A refused box reports why through the reason, and leaves the reason alone when it succeeds.
void TestBoxReportsRefusal() {
  const auto sentinel = ElementError::OutOfRange;
  FrameRig rig;
  ElementError reason = sentinel;
  (void)Box(rig.frame, {}, {}, {}, &reason);
  Check(reason == ElementError::NotBegun, "a box before Begin is not begun");

  rig.Begin();
  reason = sentinel;
  const Element child = Box(rig.frame, {}, {}, {}, &reason);
  Check(!child.IsNone() && reason == sentinel, "a good box leaves the reason alone");
  const std::array<Element, 1> kids{child};
  const Element first = Box(rig.frame, {}, {}, kids, &reason);
  Check(!first.IsNone() && reason == sentinel, "so does a good parent");

  Check(Box(rig.frame, {}, {}, kids, &reason).IsNone() && reason == ElementError::AlreadyParented,
        "a second parent is refused for already having one");
  const std::array<Element, 1> none{Element()};
  Check(Box(rig.frame, {}, {}, none, &reason).IsNone() && reason == ElementError::None, "a none child is refused");
  rig.Begin();
  Check(Box(rig.frame, {}, {}, kids, &reason).IsNone() && reason == ElementError::Stale, "a stale child is refused");
  Check(Box(rig.frame, {}, {}).IsNone() == false, "while a box with no reason asked for still builds");
}

// Body text in white, the common fixture for the text tests.
TextProps Body(std::string_view text) { return {text, DEFAULT_THEME.type.body, Rgb(0xFFFFFF)}; }

// A frame at `output` with the test fonts, begun and ready to build in.
struct TextRig {
  FrameRig rig;
  TextRig(const ImGuiScope &imgui, const Output &output) : rig(output.width, output.height) {
    rig.fonts = imgui.fonts;
    rig.Begin();
  }
};

// Measures the row offered `offer`, arranges it at `rect` and paints it into a new list.
std::unique_ptr<ImDrawList> LayOut(FrameRig &rig, Element row, Vec2 offer, const Rect &rect) {
  (void)Measure(rig.frame, row, offer);
  Check(Arrange(rig.frame, row, rect), "the row arranges");
  auto list = NewDrawList();
  Paint(rig.frame, row, *list);
  return list;
}

// Text in a slot narrower than it wants is ellipsized to the slot, exactly as LayoutText and
// DrawText do it. Text with no limit is the baseline, which keeps its full width and so draws
// more vertices, so the slot is what cut it.
void TestTextMatchesLayoutAndDraw() {
  ImGuiScope imgui(true);
  for (const Output &output : OUTPUTS) {
    TextRig text(imgui, output);
    FrameRig &rig = text.rig;
    const TextProps props = Body("A fairly long title that cannot fit its slot");
    // The node must keep its own copy of the text, so the caller's string is overwritten once the
    // node is built. A node that only viewed it would then measure and draw the filler.
    std::string owned(props.text);
    TextProps viewing = props;
    viewing.text = owned;
    const float slot = ToPixels(rig.scale, 200_dp);
    const std::array<Element, 1> kids{Text(rig.frame, viewing)};
    std::fill(owned.begin(), owned.end(), 'x');
    const Element row = Box(rig.frame, {}, {}, kids);
    const auto actual = LayOut(rig, row, {slot, INFINITE_SPACE}, {10, 20, slot, 100});

    const TextLayout cut = LayoutText(rig.frame.Ui(), props, slot);
    auto expected = NewDrawList();
    DrawText(*expected, cut, props, {10, 20, slot, cut.size.y});
    Check(SameDrawing(*actual, *expected), "text draws what LayoutText and DrawText draw");
    Check(Arranged(rig.frame, kids[0]).width == slot, "and the slot is its width");

    const TextLayout whole = LayoutText(rig.frame.Ui(), props, INFINITE_SPACE);
    auto unlimited = NewDrawList();
    DrawText(*unlimited, whole, props, {10, 20, whole.size.x, whole.size.y});
    Check(whole.size.x > slot && unlimited->VtxBuffer.Size > actual->VtxBuffer.Size,
          "the same text with no limit keeps its width and draws more");
  }
}

// A wrapped text with max_lines draws the lines LayoutText gives at the slot's width. The text
// needs more than two lines there, so ignoring max_lines draws more vertices than the layout.
void TestTextWrapsToMaxLines() {
  ImGuiScope imgui(true);
  for (const Output &output : OUTPUTS) {
    TextRig text(imgui, output);
    FrameRig &rig = text.rig;
    TextProps props = Body("one two three four five six seven eight nine ten eleven twelve");
    props.overflow = TextOverflow::Wrap;
    props.max_lines = 2;
    const float slot = ToPixels(rig.scale, 200_dp);
    const std::array<Element, 1> kids{Text(rig.frame, props)};
    const Element row = Box(rig.frame, {}, {}, kids);
    const auto actual = LayOut(rig, row, {slot, INFINITE_SPACE}, {10, 20, slot, 200});

    const TextLayout wrapped = LayoutText(rig.frame.Ui(), props, slot);
    auto expected = NewDrawList();
    DrawText(*expected, wrapped, props, {10, 20, slot, wrapped.size.y});
    Check(wrapped.lines.size() == 2, "the layout stops at two lines");
    Check(Arranged(rig.frame, kids[0]).height == wrapped.size.y, "and the text is as tall as those two lines");
    Check(SameDrawing(*actual, *expected), "wrapped text draws the lines LayoutText gives");
    TextProps open = props;
    open.max_lines = 0;
    Check(LayoutText(rig.frame.Ui(), open, slot).lines.size() > 2, "without the limit it would take more lines");
  }
}

// Measured with no limit and arranged in a narrower rectangle, text is laid out again at the new
// width in the same frame, and centred vertically in the stretched rectangle. Skipping the layout
// would draw the full-width text, and the unstretched rectangle would draw it at the top.
void TestTextRelayoutsAndCentres() {
  ImGuiScope imgui(true);
  for (const Output &output : OUTPUTS) {
    TextRig text(imgui, output);
    FrameRig &rig = text.rig;
    const TextProps props = Body("A fairly long title that cannot fit its slot");
    const float slot = ToPixels(rig.scale, 200_dp);
    const float height = ToPixels(rig.scale, 64_dp);
    const std::array<Element, 1> kids{Text(rig.frame, props)};
    const Element row = Box(rig.frame, {.align = Align::Stretch}, {}, kids);
    const auto actual = LayOut(rig, row, {INFINITE_SPACE, INFINITE_SPACE}, {10, 20, slot, height});
    Check(Arranged(rig.frame, kids[0]).height == height, "the stretched rectangle is the row's height");

    const TextLayout cut = LayoutText(rig.frame.Ui(), props, slot);
    auto expected = NewDrawList();
    DrawText(*expected, cut, props, {10, 20 + (height - cut.size.y) / 2.0f, slot, cut.size.y});
    Check(SameDrawing(*actual, *expected), "the text is laid out again at the final width and centred");
  }
}

// A texture of 200 by 100 for the image tests, which only need its handle and shape.
struct TestTexture {
  ImTextureData data;
  TestTexture() { data.Create(ImTextureFormat_RGBA32, 200, 100); }
};

// An image draws what DrawImage draws in its rectangle, contained and covered, with the radius in
// design units rounded once. At 1080p a radius already in pixels would look the same, so 720p and
// 4K tell them apart. Contain and Cover differ from each other, or the comparison proves little.
void TestImageMatchesDrawImage() {
  ImGuiScope imgui;
  TestTexture texture;
  for (const Output &output : OUTPUTS) {
    std::unique_ptr<ImDrawList> drawn[2];
    int which = 0;
    for (const ImageFit fit : {ImageFit::Contain, ImageFit::Cover}) {
      TextRig text(imgui, output);
      FrameRig &rig = text.rig;
      const std::array<Element, 1> kids{
          Image(rig.frame, &texture.data, {200, 100}, fit, 12_dp, {.width = 100_dp, .height = 100_dp})};
      const Element row = Box(rig.frame, {}, {}, kids);
      const float side = ToPixels(rig.scale, 100_dp);
      const auto actual = LayOut(rig, row, {INFINITE_SPACE, INFINITE_SPACE}, {10, 20, side * 3, side * 2});
      auto expected = NewDrawList();
      DrawImage(*expected, rig.frame.Ui(), &texture.data, {200, 100}, {10, 20, side, side}, 12.0f, fit);
      Check(expected->VtxBuffer.Size > 0 && SameDrawing(*actual, *expected), "an image draws what DrawImage draws");
      drawn[which++] = NewDrawList();
      DrawImage(*drawn[which - 1], rig.frame.Ui(), &texture.data, {200, 100}, {10, 20, side, side}, 12.0f, fit);
    }
    Check(!SameDrawing(*drawn[0], *drawn[1]), "contain and cover draw differently");
  }
}

// With one axis set an image takes the other from the texture's shape, 200 by 100, so 64 dp wide
// is 32 dp tall and 32 dp tall is 64 dp wide. With neither it measures nothing.
void TestImageFollowsAspect() {
  ImGuiScope imgui;
  TestTexture texture;
  for (const Output &output : OUTPUTS) {
    TextRig text(imgui, output);
    FrameRig &rig = text.rig;
    const auto measure = [&](const ItemLayout &item) {
      const Element image = Image(rig.frame, &texture.data, {200, 100}, ImageFit::Contain, 0_dp, item);
      return Measure(rig.frame, image, {INFINITE_SPACE, INFINITE_SPACE});
    };
    const Vec2 wide = measure({.width = 64_dp});
    Check(wide.x == ToPixels(rig.scale, 64_dp) && std::fabs(wide.y - ToPixels(rig.scale, 32_dp)) <= ONE_PIXEL,
          "64 dp wide is 32 dp tall");
    const Vec2 tall = measure({.height = 32_dp});
    Check(tall.y == ToPixels(rig.scale, 32_dp) && std::fabs(tall.x - ToPixels(rig.scale, 64_dp)) <= ONE_PIXEL,
          "32 dp tall is 64 dp wide");
    const Vec2 both = measure({.width = 50_dp, .height = 50_dp});
    Check(both.x == ToPixels(rig.scale, 50_dp) && both.y == both.x, "two sizes are used as given");
    const Vec2 neither = measure({});
    Check(neither.x == 0 && neither.y == 0, "with no size an image measures zero");
  }
}

// A row with a fixed icon, growing text and an image against a hand-built SolveFlex, within a
// pixel. The row is wider and taller than its content and centres its children, so a child that
// ignores its item layout or is measured wrongly lands somewhere else.
void TestMixedRowMatchesSolver() {
  ImGuiScope imgui(true);
  TestTexture texture;
  for (const Output &output : OUTPUTS) {
    TextRig text(imgui, output);
    FrameRig &rig = text.rig;
    const auto &scale = rig.scale;
    const TextProps props = Body("Now playing");
    const Layout icon{.item = {.width = 24_dp, .height = 24_dp}};
    const std::array<Element, 3> kids{
        Box(rig.frame, icon, {}), Text(rig.frame, props, {.grow = 1.0f}),
        Image(rig.frame, &texture.data, {200, 100}, ImageFit::Contain, 0_dp, {.height = 40_dp})};
    const Layout rowLayout{.align = Align::Center, .gap = 8_dp, .padding = DpInsets::All(4_dp)};
    const Element row = Box(rig.frame, rowLayout, {}, kids);
    const Rect rect{10, 20, ToPixels(scale, 600_dp), ToPixels(scale, 80_dp)};
    (void)Measure(rig.frame, row, {rect.width, rect.height});
    Check(Arrange(rig.frame, row, rect), "the row arranges");

    const TextLayout natural = LayoutText(rig.frame.Ui(), props, INFINITE_SPACE);
    const float imageWidth = SnapPixel(ToPixels(scale, 40_dp) * 2);
    const std::vector<FlexItem> items = {{.size = {ToPixels(scale, 24_dp), ToPixels(scale, 24_dp)}},
                                         {.size = natural.size, .grow = 1.0f},
                                         {.size = {imageWidth, ToPixels(scale, 40_dp)}}};
    const FlexStyle style{.align = Align::Center,
                          .gap = ToPixels(scale, 8_dp),
                          .padding = EdgeInsets::All(ToPixels(scale, 4_dp))};
    const auto expected = SolveFlex(rect, style, items);
    for (size_t i = 0; i < kids.size(); ++i) {
      Check(NearRect(Arranged(rig.frame, kids[i]), expected[i]), "each child lands where the solver puts it");
    }
    Check(expected[1].width > natural.size.x, "the text grew, so growing was part of the layout");
  }
}

// Wrapped text in an overflowing row stops shrinking at its longest word, and a short-worded text
// shrinks to what is left. The same text given a smaller minimum is not held, so the word is what
// held it. The sibling has a fixed width and so never shrinks.
void TestTextKeepsLongestWord() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const auto textWidth = [&](std::string_view words, std::optional<Dp> minMain) {
    rig.Begin();
    TextProps wrapped = Body(words);
    wrapped.overflow = TextOverflow::Wrap;
    const std::array<Element, 2> kids{Text(rig.frame, wrapped, {.min_main = minMain}),
                                      Box(rig.frame, {.item = {.width = 150_dp, .height = 20_dp}}, {})};
    const Element row = Box(rig.frame, {}, {}, kids);
    (void)Measure(rig.frame, row, {200, 60});
    Check(Arrange(rig.frame, row, {0, 0, 200, 60}), "the row arranges");
    return Arranged(rig.frame, kids[0]).width;
  };
  const std::string_view longWord = "supercalifragilisticexpialidocious";
  const float wordWidth = LayoutText(rig.frame.Ui(), Body(longWord), INFINITE_SPACE).size.x;
  Check(wordWidth > 50, "the word is wider than the room left");
  Check(std::fabs(textWidth("a short supercalifragilisticexpialidocious", std::nullopt) - wordWidth) <= ONE_PIXEL,
        "long-worded text stops at its longest word");
  Check(textWidth("a short supercalifragilisticexpialidocious", 0_dp) == 50, "unless its minimum is set lower");
  Check(textWidth("a b c d e f g h i j k l m n", std::nullopt) == 50, "short words shrink to what is left");
}

// One-line text that truncates or clips has no word to protect, so in the same overflowing row it
// shrinks below its longest word. The ellipsis text ends in the ellipsis and the clipped text does
// not, and both draw what LayoutText gives at the width the row assigns. The same text as Wrap is
// the baseline that stops at its longest word, so the word is what the overflow mode removes.
void TestTruncatingTextShrinksBelowWord() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  constexpr float ROOM = 50;
  const std::string_view sentence = "a short supercalifragilisticexpialidocious";
  const auto lay = [&](TextOverflow overflow) {
    rig.Begin();
    TextProps props = Body(sentence);
    props.overflow = overflow;
    const std::array<Element, 2> kids{Text(rig.frame, props),
                                      Box(rig.frame, {.item = {.width = 150_dp, .height = 20_dp}}, {})};
    const Element row = Box(rig.frame, {}, {}, kids);
    const auto actual = LayOut(rig, row, {200, 60}, {0, 0, 200, 60});
    const float width = Arranged(rig.frame, kids[0]).width;
    if (overflow == TextOverflow::Wrap) {
      return width;
    }
    const TextLayout layout = LayoutText(rig.frame.Ui(), props, ROOM);
    auto expected = NewDrawList();
    DrawText(*expected, layout, props, {0, 0, ROOM, layout.size.y});
    Check(width == ROOM && SameDrawing(*actual, *expected), "the text shrinks to the room and draws as LayoutText does");
    const std::string &shown = layout.lines.front().text;
    const bool endsInEllipsis = shown.size() >= 3 && shown.compare(shown.size() - 3, 3, "\xE2\x80\xA6") == 0;
    Check(endsInEllipsis == (overflow == TextOverflow::Ellipsis), "only the ellipsis text ends in an ellipsis");
    return width;
  };
  const float word = LayoutText(rig.frame.Ui(), Body("supercalifragilisticexpialidocious"), INFINITE_SPACE).size.x;
  Check(std::fabs(lay(TextOverflow::Wrap) - word) <= ONE_PIXEL && word > ROOM, "wrapped text stops at its longest word");
  Check(lay(TextOverflow::Ellipsis) == ROOM, "ellipsis text truncates below it");
  Check(lay(TextOverflow::Clip) == ROOM, "and clipped text does too");
}

// Ellipsis text stops shrinking at the width of its ellipsis, so with less room than that it still
// shows the mark. Clipped text takes whatever is left, the baseline with no minimum at all.
void TestEllipsisTextKeepsItsMark() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  constexpr float ROOM = 5;
  const auto width = [&](TextOverflow overflow) {
    rig.Begin();
    TextProps props = Body("A long title");
    props.overflow = overflow;
    const std::array<Element, 2> kids{Text(rig.frame, props),
                                      Box(rig.frame, {.item = {.width = 195_dp, .height = 20_dp}}, {})};
    const Element row = Box(rig.frame, {}, {}, kids);
    (void)Measure(rig.frame, row, {200, 60});
    Check(Arrange(rig.frame, row, {0, 0, 200, 60}), "the row arranges");
    return Arranged(rig.frame, kids[0]).width;
  };
  const float mark = LayoutText(rig.frame.Ui(), Body("\xE2\x80\xA6"), INFINITE_SPACE).size.x;
  Check(mark > ROOM, "the mark is wider than the room");
  Check(std::fabs(width(TextOverflow::Ellipsis) - mark) <= ONE_PIXEL, "ellipsis text keeps room for its mark");
  Check(width(TextOverflow::Clip) == ROOM, "clipped text takes what is left");
}

// A box without a size of its own stops shrinking at its content, here a 150 dp child, so the
// child does not spill out of a box that shrank around it. Giving the box a minimum of zero is the
// baseline where it does shrink, to the 100 dp left beside its fixed sibling.
void TestBoxKeepsItsContent() {
  FrameRig rig(1920, 1080);
  const auto boxWidth = [&](std::optional<Dp> minMain) {
    rig.Begin();
    const std::array<Element, 1> inner{Box(rig.frame, {.item = {.width = 150_dp, .height = 20_dp}}, {})};
    const std::array<Element, 2> kids{Box(rig.frame, {.item = {.height = 20_dp, .min_main = minMain}}, {}, inner),
                                      Box(rig.frame, {.item = {.width = 100_dp, .height = 20_dp}}, {})};
    const Element row = Box(rig.frame, {}, {}, kids);
    (void)Measure(rig.frame, row, {200, 40});
    Check(Arrange(rig.frame, row, {0, 0, 200, 40}), "the row arranges");
    return Arranged(rig.frame, kids[0]).width;
  };
  Check(boxWidth(std::nullopt) == 150, "a box keeps room for its 150 dp child");
  Check(boxWidth(0_dp) == 100, "and with a minimum of zero it shrinks to the room left");
}

// An image with only a height keeps the width its shape gives it when its row overflows, where an
// image given a minimum of zero is the baseline that shrinks to the room left beside a fixed box.
void TestImageKeepsItsSize() {
  ImGuiScope imgui;
  TestTexture texture;
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const auto imageWidth = [&](std::optional<Dp> minMain) {
    rig.Begin();
    const std::array<Element, 2> kids{
        Image(rig.frame, &texture.data, {200, 100}, ImageFit::Contain, 0_dp, {.height = 40_dp, .min_main = minMain}),
        Box(rig.frame, {.item = {.width = 150_dp, .height = 20_dp}}, {})};
    const Element row = Box(rig.frame, {}, {}, kids);
    (void)Measure(rig.frame, row, {200, 60});
    Check(Arrange(rig.frame, row, {0, 0, 200, 60}), "the row arranges");
    return Arranged(rig.frame, kids[0]).width;
  };
  Check(imageWidth(std::nullopt) == 80, "an image keeps the 80 dp its shape gives it");
  Check(imageWidth(0_dp) == 50, "and with a minimum of zero it shrinks to the room left");
}

// Opacity reaches text and images as it does DrawText and DrawImage. The faded drawing differs
// from the opaque one, or the comparison would hold with opacity ignored on both sides.
void TestLeavesFade() {
  ImGuiScope imgui(true);
  TestTexture texture;
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const TextProps props = Body("Fade me");
  const std::array<Element, 2> kids{
      Text(rig.frame, props),
      Image(rig.frame, &texture.data, {200, 100}, ImageFit::Fill, 0_dp, {.width = 40_dp, .height = 40_dp})};
  const Element row = Box(rig.frame, {}, {}, kids);
  (void)Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
  const Rect rect{10, 20, 300, 60};
  Check(Arrange(rig.frame, row, rect), "the row arranges");
  auto faded = NewDrawList();
  Paint(rig.frame, row, *faded, {.opacity = 0.5f});
  auto opaque = NewDrawList();
  Paint(rig.frame, row, *opaque);
  // The expected drawing uses the rectangles Arrange gave, since this test is about opacity and
  // not placement, which the other tests check against independent numbers.
  auto expected = NewDrawList();
  const Node *textNode = FrameInternals::Find(rig.frame, kids[0]);
  const TextLayout layout = LayoutText(rig.frame.Ui(), props, INFINITE_SPACE);
  DrawText(*expected, layout, props, textNode->arranged, 0.5f);
  DrawImage(*expected, rig.frame.Ui(), &texture.data, {200, 100}, Arranged(rig.frame, kids[1]), 0.0f, ImageFit::Fill,
            0.5f);
  Check(SameDrawing(*faded, *expected), "faded text and image match the faded components");
  Check(!SameDrawing(*faded, *opaque), "and differ from the opaque drawing");
}

// Natural sizes are measured once per node and pass, however deep the tree. Every level here grows
// in a wide row, so every level measures its children again at a new length. Without the memo that
// doubles the work at each level, and the count would far exceed the number of nodes.
void TestNaturalMeasureIsMemoised() {
  constexpr int DEPTH = 10;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Layout fixed{.item = {.width = 10_dp, .height = 10_dp}};
  const Layout level{.item = {.height = 10_dp, .grow = 1.0f}};
  int nodes = 0;
  Element inner = Box(rig.frame, level, {});
  ++nodes;
  for (int i = 0; i < DEPTH; ++i) {
    const std::array<Element, 2> kids{Box(rig.frame, fixed, {}), inner};
    inner = Box(rig.frame, level, {}, kids);
    nodes += 2;
  }
  const std::array<Element, 1> top{inner};
  const Element root = Box(rig.frame, {}, {}, top);
  (void)Measure(rig.frame, root, {1000, 100});
  Check(FrameTestHooks::NaturalMeasures(rig.frame) == static_cast<uint32_t>(nodes),
        "every node below the root is measured naturally once");
  Check(FrameInternals::Find(rig.frame, inner)->measured.x == 1000, "and the tree still grows to fit");
}

// The elements of the tree used to check a second measure: a row holding a box that holds wrapped
// text, beside a fixed box.
struct WrapTree {
  Element row;
  Element box;
  Element text;
};

// Builds the tree in `rig`.
WrapTree BuildWrapTree(FrameRig &rig) {
  TextProps props = Body("one two three four five six seven eight nine ten");
  props.overflow = TextOverflow::Wrap;
  const std::array<Element, 1> inner{Text(rig.frame, props)};
  const Element box = Box(rig.frame, {}, {}, inner);
  const std::array<Element, 2> kids{box, Box(rig.frame, {.item = {.width = 150_dp, .height = 20_dp}}, {})};
  return {Box(rig.frame, {}, {}, kids), box, inner[0]};
}

// Measuring a tree at one width and then at another gives what a fresh frame measured at the
// second width alone gives. At 200 px the text is squeezed to its longest word and wraps onto many
// lines, and at 1000 px it fits on one. A second pass that kept the first pass state would leave
// the text tall and outside its box.
void TestSecondMeasureMatchesFresh() {
  ImGuiScope imgui(true);
  TextRig again(imgui, OUTPUTS[1]);
  const WrapTree reused = BuildWrapTree(again.rig);
  (void)Measure(again.rig.frame, reused.row, {200, 60});
  const float squeezed = FrameInternals::Find(again.rig.frame, reused.text)->measured.y;
  (void)Measure(again.rig.frame, reused.row, {1000, 60});
  Check(Arrange(again.rig.frame, reused.row, {0, 0, 1000, 60}), "the reused tree arranges");

  TextRig fresh(imgui, OUTPUTS[1]);
  const WrapTree once = BuildWrapTree(fresh.rig);
  (void)Measure(fresh.rig.frame, once.row, {1000, 60});
  Check(Arrange(fresh.rig.frame, once.row, {0, 0, 1000, 60}), "the fresh tree arranges");

  const float oneLine = FrameInternals::Find(fresh.rig.frame, once.text)->measured.y;
  Check(squeezed > 2 * oneLine, "the first pass wrapped the text onto several lines");
  Check(Arranged(again.rig.frame, reused.text) == Arranged(fresh.rig.frame, once.text),
        "the text is placed as in a fresh frame");
  Check(Arranged(again.rig.frame, reused.box) == Arranged(fresh.rig.frame, once.box), "and so is its box");
  const Rect box = Arranged(again.rig.frame, reused.box);
  const Rect inside = Arranged(again.rig.frame, reused.text);
  Check(inside.Bottom() <= box.Bottom() && inside.Right() <= box.Right(), "the text sits inside its box");
}

// A minimum is the same number whether the parent asks what a child needs or lays it out. A
// child with a min-content of 150 dp and a maximum of 50 dp needs 50 dp, so its parent needs 50 dp
// too and stays at 50 dp in a row too short for it. A parent that counted the full 150 dp would
// refuse to shrink below that and grow instead.
void TestMinimumHonoursMaximumInParent() {
  FrameRig rig(1920, 1080);
  rig.Begin();
  const std::array<Element, 1> content{Box(rig.frame, {.item = {.width = 150_dp, .height = 20_dp}}, {})};
  const std::array<Element, 1> capped{Box(rig.frame, {.item = {.height = 20_dp, .max_main = 50_dp}}, {}, content)};
  const std::array<Element, 2> kids{Box(rig.frame, {.item = {.height = 20_dp}}, {}, capped),
                                    Box(rig.frame, {.item = {.width = 80_dp, .height = 20_dp}}, {})};
  const Element row = Box(rig.frame, {}, {}, kids);
  (void)Measure(rig.frame, row, {100, 40});
  Check(Arrange(rig.frame, row, {0, 0, 100, 40}), "the row arranges");
  Check(Arranged(rig.frame, kids[0]).width == 50, "the parent keeps the capped 50 dp");
}

// A box minimum comes from its content through two levels of boxes, with the padding of each
// added. The outer box holds a middle box with 10 dp padding around a 150 dp child, and has 5 dp
// padding of its own, so it needs 180 dp. Given a minimum of zero it shrinks to the 100 dp left.
void TestMinContentThroughNestedBoxes() {
  FrameRig rig(1920, 1080);
  const auto outerWidth = [&](std::optional<Dp> minMain) {
    rig.Begin();
    const std::array<Element, 1> leaf{Box(rig.frame, {.item = {.width = 150_dp, .height = 20_dp}}, {})};
    const std::array<Element, 1> middle{Box(rig.frame, {.padding = DpInsets::All(10_dp)}, {}, leaf)};
    const Layout outer{.padding = DpInsets::All(5_dp), .item = {.min_main = minMain}};
    const std::array<Element, 2> kids{Box(rig.frame, outer, {}, middle),
                                      Box(rig.frame, {.item = {.width = 100_dp, .height = 20_dp}}, {})};
    const Element row = Box(rig.frame, {}, {}, kids);
    (void)Measure(rig.frame, row, {200, 60});
    Check(Arrange(rig.frame, row, {0, 0, 200, 60}), "the row arranges");
    return Arranged(rig.frame, kids[0]).width;
  };
  Check(outerWidth(std::nullopt) == 180, "the outer box needs its content and both paddings");
  Check(outerWidth(0_dp) == 100, "and with a minimum of zero it shrinks to the room left");
}

// An image whose texture has no size measures and paints without trouble. With one axis set there
// is no shape to follow, so the other axis is zero, and with both set it is drawn as DrawImage
// draws it, with no division by the empty texture.
void TestImageWithEmptyTexture() {
  ImGuiScope imgui;
  TestTexture texture;
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const Element wideOnly = Image(rig.frame, &texture.data, {0, 0}, ImageFit::Contain, 4_dp, {.width = 40_dp});
  const Vec2 size = Measure(rig.frame, wideOnly, {INFINITE_SPACE, INFINITE_SPACE});
  Check(size.x == 40 && size.y == 0, "one axis alone gives nothing to follow");

  const std::array<Element, 1> kids{
      Image(rig.frame, &texture.data, {0, 0}, ImageFit::Contain, 4_dp, {.width = 40_dp, .height = 40_dp})};
  const Element row = Box(rig.frame, {}, {}, kids);
  const auto actual = LayOut(rig, row, {INFINITE_SPACE, INFINITE_SPACE}, {10, 20, 100, 100});
  auto expected = NewDrawList();
  DrawImage(*expected, rig.frame.Ui(), &texture.data, {0, 0}, {10, 20, 40, 40}, 4.0f, ImageFit::Contain);
  Check(SameDrawing(*actual, *expected), "an image with an empty texture draws as DrawImage does");
}

// Text narrower than its limit is not laid out again by Arrange. A short text in a 400 px column
// is measured at that width and arranged at its own, which the layout already has. The baseline is
// a long text measured with no limit and arranged in a narrow row, which does need the new layout.
void TestShortTextIsNotRelaidOut() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const std::array<Element, 1> shortText{Text(rig.frame, Body("Hi"))};
  const Element column = Box(rig.frame, {.direction = FlexDirection::Column}, {}, shortText);
  (void)Measure(rig.frame, column, {400, INFINITE_SPACE});
  Check(Arrange(rig.frame, column, {0, 0, 400, 200}), "the column arranges");
  Check(FrameTestHooks::TextRelayouts(rig.frame) == 0, "a short text is not laid out again");

  rig.Begin();
  const std::array<Element, 1> longText{Text(rig.frame, Body("A fairly long title that cannot fit its slot"))};
  const Element row = Box(rig.frame, {}, {}, longText);
  (void)Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, row, {0, 0, 200, 60}), "the row arranges");
  Check(FrameTestHooks::TextRelayouts(rig.frame) == 1, "while a text cut to a new width is laid out again");
}

// MeasuredAt answers for a root and not for a node with a parent. Two children of a row that flex
// shrinks to fit are given new sizes by the row's layout, which leaves the room they were measured in
// as it was, so they must not report the size they measured to. The root, which nothing resizes,
// does report it, and in other room it reports nothing.
void TestMeasuredAtIsForRoots() {
  FrameRig rig;
  rig.Begin();
  const std::array<Element, 1> insideA{Custom(rig.frame, {.width = 80_dp, .height = 20_dp}, [](const CustomPaint &) {})};
  const std::array<Element, 1> insideB{Custom(rig.frame, {.width = 80_dp, .height = 20_dp}, [](const CustomPaint &) {})};
  const Element first = Box(rig.frame, {}, {}, insideA);
  const Element second = Box(rig.frame, {}, {}, insideB);
  const std::array<Element, 2> kids{first, second};
  const Element row = Box(rig.frame, {}, {}, kids);
  const Vec2 room{INFINITE_SPACE, INFINITE_SPACE};
  const Vec2 wanted = Measure(rig.frame, row, room);
  Check(Arrange(rig.frame, row, {0, 0, 100, 20}), "the row arranges narrower than its children want");
  Check(Arranged(rig.frame, first).width < 80.0f, "so flex shrinks a child");
  const auto measured = MeasuredAt(rig.frame, row, room);
  Check(measured && measured->x == wanted.x && measured->y == wanted.y, "the root reports what it measured to in the room it measured in");
  Check(!MeasuredAt(rig.frame, row, {100, INFINITE_SPACE}), "and nothing in other room");
  Check(!MeasuredAt(rig.frame, first, room) && !MeasuredAt(rig.frame, first, {INFINITE_SPACE, INFINITE_SPACE}), "a child that flex resized reports nothing");
}

// A text with an explicit width is laid out in that width in Measure, and Arrange gives it the same
// width, so it is not laid out again even when the text is much narrower. TestShortTextIsNotRelaidOut
// is the baseline that shows the count does move when a text is given a new width.
void TestExplicitWidthTextIsLaidOutOnce() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const std::array<Element, 1> label{Text(rig.frame, Body("Hi"), {.width = 200_dp})};
  const Element column = Box(rig.frame, {.direction = FlexDirection::Column, .align = Align::Center}, {}, label);
  (void)Measure(rig.frame, column, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, column, {0, 0, 400, 200}), "the column arranges");
  Check(FrameTestHooks::TextRelayouts(rig.frame) == 0, "a text with an explicit width is laid out once");
}

// Wrapped text in a column takes the column width, so it wraps and is as tall as its lines there.
// Its natural width is one long line, wider than the column, which is why the column measures it
// again at the room it has. The text is also checked against LayoutText at that width.
void TestColumnWrapsTextToItsWidth() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  TextProps props = Body("one two three four five six seven eight nine ten");
  props.overflow = TextOverflow::Wrap;
  const std::array<Element, 1> kids{Text(rig.frame, props)};
  const Element column = Box(rig.frame, {.direction = FlexDirection::Column}, {}, kids);
  const auto actual = LayOut(rig, column, {200, INFINITE_SPACE}, {0, 0, 200, 400});
  const TextLayout wrapped = LayoutText(rig.frame.Ui(), props, 200);
  Check(wrapped.lines.size() > 1, "the text needs several lines in 200 px");
  Check(Arranged(rig.frame, kids[0]).height == wrapped.size.y, "and the text is as tall as those lines");
  auto expected = NewDrawList();
  DrawText(*expected, wrapped, props, {0, 0, wrapped.size.x, wrapped.size.y});
  Check(SameDrawing(*actual, *expected), "and draws what LayoutText gives at that width");
}

// A column with padding holding wrapped text reports the height of the wrapped lines plus its
// padding. The text is one line at its unbounded width, so a column that sized itself from that
// would be too short. A sibling below it in an outer column starts under the text and not inside
// it. The text needs at least three lines at the width, so a one-line height would not pass.
void TestWrappedColumnReportsItsHeight() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  TextProps props = Body("one two three four five six seven eight nine ten eleven twelve thirteen fourteen");
  props.overflow = TextOverflow::Wrap;
  const float pad = ToPixels(rig.scale, 10_dp);
  const std::array<Element, 1> inner{Text(rig.frame, props)};
  const Layout column{.direction = FlexDirection::Column, .padding = DpInsets::All(10_dp)};
  const Element wrapped = Box(rig.frame, column, {}, inner);
  const std::array<Element, 2> stacked{wrapped, Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, {})};
  const Element outer = Box(rig.frame, {.direction = FlexDirection::Column}, {}, stacked);

  const Vec2 size = Measure(rig.frame, outer, {200, INFINITE_SPACE});
  const TextLayout lines = LayoutText(rig.frame.Ui(), props, 200 - 2 * pad);
  Check(lines.lines.size() >= 3, "the text needs three lines at the column width");
  const float columnHeight = FrameInternals::Find(rig.frame, wrapped)->measured.y;
  Check(columnHeight == lines.size.y + 2 * pad, "the column is as tall as the wrapped lines plus padding");
  Check(size.y == columnHeight + 20, "and the outer column adds its sibling below that");

  Check(Arrange(rig.frame, outer, {0, 0, 200, 600}), "the outer column arranges");
  const Rect textRect = Arranged(rig.frame, inner[0]);
  Check(textRect.height == lines.size.y, "the text keeps its wrapped height");
  Check(Arranged(rig.frame, stacked[1]).y >= textRect.Bottom() + pad, "the sibling starts below the wrapped text");
}

// A column measured narrow and then wide matches a fresh frame measured wide. At 200 px its box
// is wider than the room and is measured at that room, with its text wrapped. At 1000 px it fits,
// so its text must be laid out at its unbounded width again, or the box would keep the height of
// the wrapped lines.
void TestWrappedColumnMeasuredTwice() {
  ImGuiScope imgui(true);
  struct Tree {
    Element outer;
    Element box;
    Element text;
  };
  const auto build = [](FrameRig &rig) {
    TextProps props = Body("one two three four five six seven eight nine ten eleven twelve thirteen fourteen");
    props.overflow = TextOverflow::Wrap;
    const std::array<Element, 1> inner{Text(rig.frame, props)};
    const std::array<Element, 1> boxes{Box(rig.frame, {.direction = FlexDirection::Column, .padding = DpInsets::All(10_dp)}, {}, inner)};
    return Tree{Box(rig.frame, {.direction = FlexDirection::Column}, {}, boxes), boxes[0], inner[0]};
  };
  TextRig again(imgui, OUTPUTS[1]);
  const Tree reused = build(again.rig);
  (void)Measure(again.rig.frame, reused.outer, {200, INFINITE_SPACE});
  const float narrow = FrameInternals::Find(again.rig.frame, reused.box)->measured.y;
  (void)Measure(again.rig.frame, reused.outer, {1000, INFINITE_SPACE});
  Check(Arrange(again.rig.frame, reused.outer, {0, 0, 1000, 600}), "the reused tree arranges");

  TextRig fresh(imgui, OUTPUTS[1]);
  const Tree once = build(fresh.rig);
  (void)Measure(fresh.rig.frame, once.outer, {1000, INFINITE_SPACE});
  Check(Arrange(fresh.rig.frame, once.outer, {0, 0, 1000, 600}), "the fresh tree arranges");
  Check(narrow > FrameInternals::Find(fresh.rig.frame, once.box)->measured.y, "the narrow pass wrapped the text");
  Check(Arranged(again.rig.frame, reused.box) == Arranged(fresh.rig.frame, once.box), "the box matches a fresh frame");
  Check(Arranged(again.rig.frame, reused.text) == Arranged(fresh.rig.frame, once.text), "and so does the text");
  Check(FrameInternals::Find(again.rig.frame, reused.box)->measured.y ==
            FrameInternals::Find(fresh.rig.frame, once.box)->measured.y,
        "and the box measures the same height");
}

// A tree that alternates rows and columns, ten levels deep, with every level growing so that it is
// measured again at its assigned length. The measure calls stay linear in the node count. A node
// is measured naturally once per pass, and again at its assigned length only in the one measure of
// its parent that has room to assign, so the bound is two per node plus the root. Keyed by the
// cross room, which a row inside a column and a column inside a row offer differently each time,
// the memo missed and this tree took 156 calls for its 21 nodes.
void TestAlternatingTreeStaysLinear() {
  constexpr int DEPTH = 10;
  constexpr uint32_t MEASURES_PER_NODE = 2;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Layout fixed{.item = {.width = 10_dp, .height = 10_dp}};
  uint32_t nodes = 0;
  Element inner = Box(rig.frame, {.item = {.grow = 1.0f}}, {});
  ++nodes;
  for (int i = 0; i < DEPTH; ++i) {
    const std::array<Element, 2> kids{Box(rig.frame, fixed, {}), inner};
    const FlexDirection direction = i % 2 == 0 ? FlexDirection::Column : FlexDirection::Row;
    inner = Box(rig.frame, {.direction = direction, .item = {.grow = 1.0f}}, {}, kids);
    nodes += 2;
  }
  const std::array<Element, 1> top{inner};
  const Element root = Box(rig.frame, {}, {}, top);
  (void)Measure(rig.frame, root, {1000, 1000});
  const uint32_t calls = FrameTestHooks::MeasureCalls(rig.frame);
  Check(calls <= MEASURES_PER_NODE * nodes + 1, "measure calls stay linear in the number of nodes");
  Check(calls > nodes, "and the tree really was measured again at assigned lengths");
}

// The grow share of a wrapped child counts its height at the room across, not its one-line height.
// A column holds wrapped text beside an empty box, both growing. The text is three or more lines
// tall at the column width and one line unbounded, so it starts at the taller height and the two
// split the space left. Taking the one-line height as its natural size would hand the text a
// bigger share of that space, by the difference between the two heights divided by two. The text
// sets a minimum of zero, since its automatic minimum is the wrapped height and would lift the
// one-line size back up to the same number, hiding the difference.
void TestGrowShareUsesWrappedHeight() {
  constexpr float COLUMN_WIDTH = 200;
  constexpr float LINES_OF_ROOM = 10;
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  TextProps props = Body("one two three four five six seven eight nine ten eleven twelve thirteen fourteen");
  props.overflow = TextOverflow::Wrap;
  const float wrapped = LayoutText(rig.frame.Ui(), props, COLUMN_WIDTH).size.y;
  const float oneLine = LayoutText(rig.frame.Ui(), props, INFINITE_SPACE).size.y;
  Check(wrapped >= 3 * oneLine, "the text takes three lines or more at the column width");
  const float length = LINES_OF_ROOM * wrapped;

  const std::array<Element, 2> kids{Text(rig.frame, props, {.grow = 1.0f, .min_main = 0_dp}), Box(rig.frame, {.item = {.grow = 1.0f}}, {})};
  const Element column = Box(rig.frame, {.direction = FlexDirection::Column}, {}, kids);
  (void)Measure(rig.frame, column, {COLUMN_WIDTH, length});
  Check(Arrange(rig.frame, column, {0, 0, COLUMN_WIDTH, length}), "the column arranges");
  const float expected = (length + wrapped) / 2;
  Check(Near(Arranged(rig.frame, kids[0]).height, expected), "the text gets half of the space left past its wrapped height");
  Check(Near(Arranged(rig.frame, kids[1]).height, length - expected), "and the empty box gets the other half");
  Check(!Near(expected, (length + oneLine) / 2), "which differs from the share its one-line height would give");
}

// The pieces of a tree whose middle row is measured at two different heights in one pass.
struct LimitTree {
  Element root;
  Element row;
  Element box;
  Element column;
  Element shrinker;
};

// Builds a row holding wrapped text beside a 40 dp wide box. The box holds a column, and that holds
// a child with a minimum of zero around a 100 dp block, so the child is as tall as the room the
// row offers up to 100 dp. With `nested` the row is wrapped in what makes it measure twice. It is
// the second child of a column 100 dp tall, below a 40 dp box, and that column sits in a row beside
// a 100 dp wide box. The row is then offered 60 dp of height when the outer row measures at its
// natural width, and more once its wrapped text is taller. Without it the row is the root.
LimitTree BuildLimitTree(UiFrame &frame, const TextProps &props, bool nested) {
  const std::array<Element, 1> block{Box(frame, {.item = {.width = 10_dp, .height = 100_dp}}, {})};
  const std::array<Element, 1> shrink{Box(frame, {.item = {.min_main = 0_dp}}, {}, block)};
  const std::array<Element, 1> column{Box(frame, {.direction = FlexDirection::Column}, {}, shrink)};
  const std::array<Element, 1> box{Box(frame, {.item = {.width = 40_dp}}, {}, column)};
  const std::array<Element, 2> rowKids{Text(frame, props), box[0]};
  const Element row = Box(frame, {}, {}, rowKids);
  LimitTree tree{row, row, box[0], column[0], shrink[0]};
  if (nested) {
    const std::array<Element, 2> stack{Box(frame, {.item = {.width = 10_dp, .height = 40_dp}}, {}), row};
    const std::array<Element, 1> outer{
        Box(frame, {.direction = FlexDirection::Column, .item = {.height = 100_dp}}, {}, stack)};
    const std::array<Element, 2> top{outer[0], Box(frame, {.item = {.width = 100_dp, .height = 10_dp}}, {})};
    tree.root = Box(frame, {}, {}, top);
  }
  return tree;
}

// A subtree that was measured at a smaller room across and then a larger one is laid out as a fresh
// frame measured once at the larger one. The row is offered 60 dp of height the first time, so its
// shrinking child gets 60 dp, and 160 dp the second time, when its text wraps and is taller. That
// is more than the 100 dp the block needs, so the child must take 100 dp. The box in the middle
// has the same length both times, so a layout that only compared lengths kept the 60 dp.
void TestSubtreeMatchesFreshAfterLimitWidens() {
  constexpr float OUTER_WIDTH = 300;
  constexpr float OUTER_HEIGHT = 400;
  constexpr float BLOCK_HEIGHT = 100;
  constexpr float FIRST_OFFER = 60;
  ImGuiScope imgui(true);
  TextProps props = Body("one two three four five six seven eight nine ten eleven twelve thirteen fourteen");
  props.overflow = TextOverflow::Wrap;
  TextRig again(imgui, OUTPUTS[1]);
  const LimitTree reused = BuildLimitTree(again.rig.frame, props, true);
  (void)Measure(again.rig.frame, reused.root, {OUTER_WIDTH, OUTER_HEIGHT});
  Check(Arrange(again.rig.frame, reused.root, {0, 0, OUTER_WIDTH, OUTER_HEIGHT}), "the nested tree arranges");
  const Rect row = Arranged(again.rig.frame, reused.row);
  Check(row.height > BLOCK_HEIGHT && row.height > FIRST_OFFER, "the row ends up taller than the block and its first offer");

  TextRig fresh(imgui, OUTPUTS[1]);
  const LimitTree once = BuildLimitTree(fresh.rig.frame, props, false);
  (void)Measure(fresh.rig.frame, once.root, {row.width, row.height});
  Check(Arrange(fresh.rig.frame, once.root, {0, 0, row.width, row.height}), "the fresh row arranges");
  Check(Arranged(fresh.rig.frame, once.shrinker).height == BLOCK_HEIGHT, "a fresh row gives the block its full height");
  for (const Element LimitTree::*part : {&LimitTree::box, &LimitTree::column, &LimitTree::shrinker}) {
    const Rect expected = Arranged(fresh.rig.frame, once.*part);
    const Rect actual = Arranged(again.rig.frame, reused.*part);
    Check(actual.width == expected.width && actual.height == expected.height, "the subtree matches a fresh frame");
  }
}

// A chain of boxes alternating column and row, each holding a long wrapped text beside the next
// box. The root is a box around the chain, and `nodes` counts every node in it. Boxes and texts
// with `shrinkable` take a minimum of zero, so each level squeezes its children as well as wraps.
Element BuildOverflowChain(UiFrame &frame, int depth, bool shrinkable, uint32_t &nodes) {
  TextProps props = Body("one two three four five six seven eight nine ten eleven twelve thirteen fourteen");
  props.overflow = TextOverflow::Wrap;
  ItemLayout item;
  if (shrinkable) {
    item.min_main = 0_dp;
  }
  Element inner = Text(frame, props, item);
  nodes = 1;
  for (int level = 0; level < depth; ++level) {
    const std::array<Element, 2> kids{Text(frame, props, item), inner};
    const FlexDirection direction = level % 2 == 0 ? FlexDirection::Column : FlexDirection::Row;
    inner = Box(frame, {.direction = direction, .item = item}, {}, kids);
    nodes += 2;
  }
  const std::array<Element, 1> top{inner};
  ++nodes;
  return Box(frame, {}, {}, top);
}

// How many times Measure ran on a node for a chain, and how many nodes the chain has.
struct MeasureCount {
  uint32_t calls;
  uint32_t nodes;
};

// Measures a chain of `depth` levels in a 200 px wide root that is `height` tall.
MeasureCount CountOverflowMeasures(int depth, float height, bool shrinkable) {
  constexpr float ROOT_WIDTH = 200;
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  uint32_t nodes = 0;
  const Element root = BuildOverflowChain(text.rig.frame, depth, shrinkable, nodes);
  (void)Measure(text.rig.frame, root, {ROOT_WIDTH, height});
  return {FrameTestHooks::MeasureCalls(text.rig.frame), nodes};
}

// Content that overflows at every level keeps the measure calls in step with the node count. The
// text can't shrink below its lines, so a box is measured once unbounded and at most once more at
// the room it is assigned, which makes two calls per node. The extra two allow for the root, which
// the caller measures, and its one child, which has its own limit from the root.
void TestOverflowingTreeStaysLinear() {
  constexpr uint32_t MEASURES_PER_NODE = 2;
  constexpr uint32_t ROOT_ALLOWANCE = 2;
  for (const int depth: {4, 10, 20}) {
    for (const float height: {100.0f, 600.0f}) {
      const MeasureCount count = CountOverflowMeasures(depth, height, false);
      Check(count.calls <= MEASURES_PER_NODE * count.nodes + ROOT_ALLOWANCE,
            "measure calls stay linear when content overflows");
      Check(count.calls > count.nodes, "and the tree really was measured again at assigned sizes");
    }
  }
}

// When every level can also shrink below its content the calls are no longer linear. A box is
// measured once for each distinct room its ancestors offer it, and every ancestor can add one or
// two. The cost then stays within one measure per node per level of depth, which is what this
// bounds. It is far under the doubling at each level that a missing memo gives.
void TestShrinkingOverflowIsBoundedByDepth() {
  constexpr uint32_t LINEAR_MEASURES_PER_NODE = 2;
  for (const int depth: {4, 10, 20}) {
    const MeasureCount count = CountOverflowMeasures(depth, 100.0f, true);
    Check(count.calls <= count.nodes * static_cast<uint32_t>(depth), "measure calls stay within nodes times depth");
    Check(count.calls > LINEAR_MEASURES_PER_NODE * count.nodes, "and the shrinking tree really costs more than the linear one");
  }
}

// The pieces of a row that holds a pinned wrapped text beside an image, inside a fixed-height row.
struct PinnedTree {
  Element root;
  Element row;
  Element text;
};

// Builds a row 40 dp tall holding a row of a text pinned to 100 dp wide and a 60 dp tall image.
// With `capped` the row sits in a root only 30 px tall, which is less than the row's 40 dp, so the
// root measures the row twice with the same height, once unbounded and once capped. Without it the
// row is the root.
PinnedTree BuildPinnedTree(UiFrame &frame, TestTexture &texture, const TextProps &props, bool capped) {
  const Element text = Text(frame, props, {.max_main = 100_dp});
  const std::array<Element, 2> inner{text, Image(frame, &texture.data, {200, 100}, ImageFit::Contain, 0_dp,
                                                  {.width = 30_dp, .height = 60_dp})};
  const std::array<Element, 1> middle{Box(frame, {}, {}, inner)};
  const Element row = Box(frame, {.item = {.height = 40_dp}}, {}, middle);
  PinnedTree tree{row, row, text};
  if (capped) {
    const std::array<Element, 1> kids{row};
    tree.root = Box(frame, {}, {}, kids);
  }
  return tree;
}

// A child whose length a maximum clamps wraps taller than its one-line natural size, and keeps that
// height when its parent is measured a second time with the same limit. The text is over 100 dp
// wide naturally and pinned to 100 dp, so it wraps. A layout that took its natural height when the
// limit matched would give it one line, and the other lines would spill out of its rectangle.
void TestPinnedTextKeepsWrappedHeight() {
  constexpr float PINNED_WIDTH = 100;
  constexpr float OUTER_WIDTH = 600;
  constexpr float OUTER_HEIGHT = 30;
  ImGuiScope imgui(true);
  TestTexture texture;
  TextProps props = Body("alpha beta gamma delta");
  props.overflow = TextOverflow::Wrap;
  TextRig again(imgui, OUTPUTS[1]);
  const PinnedTree reused = BuildPinnedTree(again.rig.frame, texture, props, true);
  (void)Measure(again.rig.frame, reused.root, {OUTER_WIDTH, OUTER_HEIGHT});
  Check(Arrange(again.rig.frame, reused.root, {0, 0, OUTER_WIDTH, OUTER_HEIGHT}), "the tree arranges");
  const TextLayout wrapped = LayoutText(again.rig.frame.Ui(), props, PINNED_WIDTH);
  const TextLayout oneLine = LayoutText(again.rig.frame.Ui(), props, INFINITE_SPACE);
  Check(wrapped.size.y > oneLine.size.y, "the pinned text wraps");
  Check(Arranged(again.rig.frame, reused.text).height == wrapped.size.y, "the text keeps its wrapped height");

  const Rect row = Arranged(again.rig.frame, reused.row);
  TextRig fresh(imgui, OUTPUTS[1]);
  const PinnedTree once = BuildPinnedTree(fresh.rig.frame, texture, props, false);
  (void)Measure(fresh.rig.frame, once.root, {row.width, row.height});
  Check(Arrange(fresh.rig.frame, once.root, {0, 0, row.width, row.height}), "the fresh row arranges");
  const Rect expected = Arranged(fresh.rig.frame, once.text);
  const Rect actual = Arranged(again.rig.frame, reused.text);
  Check(actual.width == expected.width && actual.height == expected.height, "and matches a fresh frame");
}

// Builds a childless box in a fresh frame, poses it when `pose` is given, arranges it at `rect` and
// paints it into `list` with `opacity`. Returns the rectangle Painted reports for it.
Rect PaintPosedBox(FrameRig &rig, const Rect &rect, const std::optional<ElementPose> &pose, Origin origin, ImDrawList &list,
                   float opacity = 1.0f) {
  rig.Begin();
  const Element box = Box(rig.frame, {}, FullStyle());
  if (pose) {
    Check(!WithPose(rig.frame, box, *pose, origin).IsNone(), "the pose is accepted");
  }
  (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, box, rect), "the box arranges");
  Paint(rig.frame, box, list, {.opacity = opacity});
  return Painted(rig.frame, box);
}

// A posed box draws what DrawBox draws at Posed, to the pixel, at every output size. Posed is the
// older rule that moves a resting element, with the offset in reference pixels and the corner
// rounded. The unposed drawing is the baseline, since a box that ignored its pose would draw it
// and nothing else in the comparison would notice.
void TestPosedBoxMatchesPosed() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 201, 100};
  const ElementPose element{.offset = {12, -7}};
  const Pose pose{.phase = Phase::Shown, .alpha = 1.0f, .offset = element.offset, .scale = element.scale};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    auto actual = NewDrawList();
    const Rect painted = PaintPosedBox(rig, rect, element, CENTER, *actual);
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), Posed(rect, pose, rig.scale), FullBoxStyle());
    Check(SameDrawing(*actual, *expected), "a posed box draws what DrawBox draws at Posed");
    Check(painted == Posed(rect, pose, rig.scale), "and Painted reports the same rectangle");

    auto resting = NewDrawList();
    (void)PaintPosedBox(rig, rect, std::nullopt, CENTER, *resting);
    Check(!SameDrawing(*actual, *resting), "which differs from the unposed box");
  }
}

// A box scaled by a half about its centre is drawn at half size around the same centre, and the
// corner is not rounded, since a rounded corner would jump while a pose animates. The rectangle is
// 201 wide so the scaled corner lands on a quarter pixel, where rounding would show. The same pose
// about the top-right corner puts the box elsewhere, so the origin matters.
void TestScaledPoseKeepsFractions() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 201, 100};
  const ElementPose pose{.scale = 0.5f};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    auto actual = NewDrawList();
    const Rect painted = PaintPosedBox(rig, rect, pose, CENTER, *actual);
    const Rect centred{150.25f, 75, 100.5f, 50};
    Check(painted == centred, "the scaled box shrinks about its centre without rounding");
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), centred, FullBoxStyle());
    Check(SameDrawing(*actual, *expected), "and is drawn there");

    auto corner = NewDrawList();
    const Rect topRight = PaintPosedBox(rig, rect, pose, TOP_RIGHT, *corner);
    Check(topRight == Rect{200.5f, 50, 100.5f, 50}, "about the top-right corner the box shrinks toward that corner");
    Check(!SameDrawing(*corner, *actual), "which draws differently from the centred one");
  }
}

// A child of a parent scaled by a half about its top-right corner is drawn where the mapping puts
// it, and its own offset is halved with it. The child is posed too, with an offset of 20 reference
// pixels, which a parent scale that did not reach the child's offset would leave at full size. The
// same parent pose about the centre is the baseline, since a mapping that ignored the origin would
// draw both alike.
void TestParentPoseReachesChild() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 300, 200};
  const Style childStyle{Rgb(0xAA5500), {}, 0_dp, 4_dp, {}};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    const auto draw = [&](Origin origin, Rect &childPainted, Rect &childArranged) {
      rig.Begin();
      const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, childStyle)};
      const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, FullStyle(), kids);
      Check(!WithPose(rig.frame, parent, ElementPose{.scale = 0.5f}, origin).IsNone(), "the parent pose is accepted");
      Check(!WithPose(rig.frame, kids[0], ElementPose{.offset = {20, 0}}).IsNone(), "and so is the child's");
      (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, parent, rect), "the tree arranges");
      auto list = NewDrawList();
      Paint(rig.frame, parent, *list);
      childPainted = Painted(rig.frame, kids[0]);
      childArranged = Arranged(rig.frame, kids[0]);
      return list;
    };
    Rect painted, arranged, centredPainted, centredArranged;
    const auto actual = draw(TOP_RIGHT, painted, arranged);
    const float offset = rig.scale.Px(20);
    const Rect expectedChild{0.5f * (arranged.x + offset) + 0.5f * rect.Right(), 0.5f * arranged.y + 0.5f * rect.y,
                             arranged.width / 2, arranged.height / 2};
    Check(painted == expectedChild, "the child lands where the parent's mapping puts it");

    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), {0.5f * rect.x + 0.5f * rect.Right(), rect.y, rect.width / 2, rect.height / 2},
            FullBoxStyle());
    DrawBox(*expected, rig.frame.Ui(), expectedChild, {Rgb(0xAA5500), 4.0f, {}, 0.0f, {}});
    Check(SameDrawing(*actual, *expected), "and the parent and child are drawn there");

    const auto centred = draw(CENTER, centredPainted, centredArranged);
    Check(!(centredPainted == painted) && !SameDrawing(*centred, *actual), "the same pose about the centre differs");
  }
}

// A parent at alpha 0.5 painted at an opacity of 0.5 draws at 0.25, whatever the output size. Alphas
// that added would give 1, and the half-opacity and half-alpha drawings alone are the baselines
// that show each factor counts.
void TestAlphaMultipliesIntoOpacity() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 200, 100};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    auto actual = NewDrawList();
    (void)PaintPosedBox(rig, rect, ElementPose{.alpha = 0.5f}, CENTER, *actual, 0.5f);
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle(), 0.25f);
    Check(SameDrawing(*actual, *expected), "alpha and opacity multiply");

    auto alphaOnly = NewDrawList();
    (void)PaintPosedBox(rig, rect, ElementPose{.alpha = 0.5f}, CENTER, *alphaOnly);
    auto opacityOnly = NewDrawList();
    (void)PaintPosedBox(rig, rect, ElementPose{}, CENTER, *opacityOnly, 0.5f);
    Check(!SameDrawing(*actual, *alphaOnly) && !SameDrawing(*actual, *opacityOnly), "each factor changes the drawing");
  }
}

// A pose is paint-only. The sizes Measure returns and the rectangles Arrange gives are the same
// with the pose and without it, at every output size, for a parent and its child. The posed tree
// does paint differently, or the comparison would hold if the pose were never stored.
void TestPoseLeavesLayoutAlone() {
  ImGuiScope imgui;
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    const auto lay = [&](bool posed, Vec2 &size, Rect &parentRect, Rect &childRect) {
      rig.Begin();
      const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, FullStyle())};
      const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, FullStyle(), kids);
      if (posed) {
        (void)WithPose(rig.frame, parent, ElementPose{.offset = {30, 10}, .scale = 0.5f}, TOP_RIGHT);
        (void)WithPose(rig.frame, kids[0], ElementPose{.offset = {5, 5}, .scale = 2.0f});
      }
      size = Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, parent, {10, 20, size.x, size.y}), "the tree arranges");
      parentRect = Arranged(rig.frame, parent);
      childRect = Arranged(rig.frame, kids[0]);
      auto list = NewDrawList();
      Paint(rig.frame, parent, *list);
      return list;
    };
    Vec2 plainSize, posedSize;
    Rect plainParent, posedParent, plainChild, posedChild;
    const auto plain = lay(false, plainSize, plainParent, plainChild);
    const auto posed = lay(true, posedSize, posedParent, posedChild);
    Check(plainSize == posedSize, "Measure ignores the pose");
    Check(plainParent == posedParent && plainChild == posedChild, "and so does Arrange");
    Check(!SameDrawing(*plain, *posed), "though the painting differs");
  }
}

// The smallest and largest x of the vertices in `list`, which for a line of text spans its glyphs.
Vec2 HorizontalExtent(const ImDrawList &list) {
  Vec2 extent{INFINITE_SPACE, -INFINITE_SPACE};
  for (int i = 0; i < list.VtxBuffer.Size; ++i) {
    extent.x = std::min(extent.x, list.VtxBuffer[i].pos.x);
    extent.y = std::max(extent.y, list.VtxBuffer[i].pos.x);
  }
  return extent;
}

// Posed text keeps its glyph sizes, since scaling a font would blur it, and moves with its centre.
// Scaled by a half about the top-right corner, its centre goes where the mapping sends it and its
// rectangle keeps the laid-out size. The unposed text is the baseline for both the position, which
// must change, and the width of the glyphs, which must not.
void TestPosedTextKeepsGlyphSize() {
  ImGuiScope imgui(true);
  const ElementPose pose{.offset = {16, 8}, .scale = 0.5f};
  for (const Output &output : OUTPUTS) {
    TextRig text(imgui, output);
    FrameRig &rig = text.rig;
    const TextProps props = Body("A short title");
    const auto draw = [&](bool posed, Rect &painted) {
      rig.Begin();
      const Element label = Text(rig.frame, props);
      if (posed) {
        Check(!WithPose(rig.frame, label, pose, TOP_RIGHT).IsNone(), "the pose is accepted");
      }
      const Vec2 size = Measure(rig.frame, label, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, label, {40, 60, size.x, size.y}), "the text arranges");
      auto list = NewDrawList();
      Paint(rig.frame, label, *list);
      painted = Painted(rig.frame, label);
      return list;
    };
    Rect plainRect, posedRect;
    const auto plain = draw(false, plainRect);
    const auto posed = draw(true, posedRect);
    Check(posedRect.width == plainRect.width && posedRect.height == plainRect.height, "the text keeps its size");

    const Vec2 centre = plainRect.Center();
    const float offset = rig.scale.Px(16);
    const float offsetY = rig.scale.Px(8);
    const Vec2 mapped{plainRect.Right() + 0.5f * (centre.x - plainRect.Right()) + offset,
                      plainRect.y + 0.5f * (centre.y - plainRect.y) + offsetY};
    Check(Near(posedRect.Center().x, mapped.x) && Near(posedRect.Center().y, mapped.y), "its centre follows the mapping");

    // The drawing is exactly the text drawn at its own size around the moved centre. Scaling the
    // rectangle instead, as boxes are, would move the corner and change the position.
    const TextLayout layout = LayoutText(rig.frame.Ui(), props, INFINITE_SPACE);
    const Vec2 origin{plainRect.x + plainRect.width, plainRect.y};
    const Vec2 movedCentre{centre.x * 0.5f + (origin.x * 0.5f + offset), centre.y * 0.5f + (origin.y * 0.5f + offsetY)};
    auto expected = NewDrawList();
    DrawText(*expected, layout, props,
             {movedCentre.x - plainRect.width / 2.0f, movedCentre.y - plainRect.height / 2.0f, plainRect.width,
              plainRect.height});
    Check(SameDrawing(*posed, *expected), "the text is drawn at its own size around the moved centre");

    const Vec2 plainSpan = HorizontalExtent(*plain);
    const Vec2 posedSpan = HorizontalExtent(*posed);
    Check(Near(posedSpan.y - posedSpan.x, plainSpan.y - plainSpan.x), "the glyphs keep their width");
    Check(!Near(posedSpan.x, plainSpan.x), "while the text moves");
  }
}

// A default element pose changes nothing, at every output size. Its drawing is compared with the
// unposed node for a box with a child and for text, since a default that faded or moved anything
// would show in one of them. The timeline's own default is the invisible waiting state, which is
// the reason the element API has its own type, so it is drawn as a baseline that must differ.
void TestDefaultElementPoseChangesNothing() {
  ImGuiScope imgui(true);
  for (const Output &output : OUTPUTS) {
    TextRig text(imgui, output);
    FrameRig &rig = text.rig;
    const auto draw = [&](std::optional<Pose> pose, bool element) {
      rig.Begin();
      const std::array<Element, 2> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, FullStyle()),
                                        Text(rig.frame, Body("A label"))};
      const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, FullStyle(), kids);
      if (element) {
        Check(!WithPose(rig.frame, parent, ElementPose{}).IsNone(), "a default element pose is accepted");
        Check(!WithPose(rig.frame, kids[1], ElementPose{}).IsNone(), "on text too");
      } else if (pose) {
        Check(!WithTimelinePose(rig.frame, parent, *pose).IsNone(), "a timeline pose is accepted");
      }
      const Vec2 size = Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, parent, {10, 20, size.x, size.y}), "the tree arranges");
      auto list = NewDrawList();
      Paint(rig.frame, parent, *list);
      return list;
    };
    const auto unposed = draw(std::nullopt, false);
    const auto defaulted = draw(std::nullopt, true);
    Check(SameDrawing(*unposed, *defaulted), "a default element pose draws like the unposed tree");
    const auto waiting = draw(Pose{}, false);
    Check(!SameDrawing(*unposed, *waiting), "while a default timeline pose does not");
  }
}

// A timeline pose given to WithPose is applied as it is, so it draws like the element pose with
// the same alpha, offset and scale.
void TestTimelinePoseConverts() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 200, 100};
  FrameRig rig(1920, 1080);
  const Pose timeline{.phase = Phase::Shown, .alpha = 0.5f, .offset = {12, -7}, .scale = 0.5f};
  auto fromTimeline = NewDrawList();
  rig.Begin();
  const Element first = Box(rig.frame, {}, FullStyle());
  Check(!WithTimelinePose(rig.frame, first, timeline, TOP_RIGHT).IsNone(), "the timeline pose is accepted");
  (void)Measure(rig.frame, first, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, first, rect), "the box arranges");
  Paint(rig.frame, first, *fromTimeline);

  auto fromElement = NewDrawList();
  (void)PaintPosedBox(rig, rect, ElementPose{.alpha = 0.5f, .offset = {12, -7}, .scale = 0.5f}, TOP_RIGHT, *fromElement);
  Check(SameDrawing(*fromTimeline, *fromElement), "and draws like the same element pose");
  auto resting = NewDrawList();
  (void)PaintPosedBox(rig, rect, std::nullopt, TOP_RIGHT, *resting);
  Check(!SameDrawing(*fromTimeline, *resting), "which differs from the unposed box");
}

// A scale that is not positive and finite, or a NaN alpha, is refused with InvalidPose and leaves
// the node as it was. Each refusal is a different input, so a check that only caught one kind
// would pass the others through. A pose that is fine is accepted without touching the reason.
void TestInvalidPoseRefused() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  const std::array<ElementPose, 9> bad{ElementPose{.scale = nan},  ElementPose{.scale = infinity},
                                       ElementPose{.scale = 0.0f}, ElementPose{.scale = -1.0f},
                                       ElementPose{.alpha = nan},  ElementPose{.alpha = nan, .scale = 0.0f},
                                       ElementPose{.offset = {nan, 0}}, ElementPose{.offset = {0, infinity}},
                                       ElementPose{.offset = {-infinity, nan}}};
  for (const ElementPose &pose : bad) {
    rig.Begin();
    const Element box = Box(rig.frame, {}, FullStyle());
    ElementError reason = ElementError::None;
    Check(WithPose(rig.frame, box, pose, CENTER, &reason).IsNone(), "an invalid pose is refused");
    Check(reason == ElementError::InvalidPose, "with the invalid pose reason");
    (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, box, {0, 0, 50, 50}), "the box arranges");
    auto refused = NewDrawList();
    Paint(rig.frame, box, *refused);
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), {0, 0, 50, 50}, FullBoxStyle());
    Check(SameDrawing(*refused, *expected), "and the node is left unposed");
  }
  rig.Begin();
  const Element fine = Box(rig.frame, {}, FullStyle());
  ElementError reason = ElementError::Stale;
  Check(!WithPose(rig.frame, fine, ElementPose{.alpha = 0.0f, .scale = 0.25f}, CENTER, &reason).IsNone(),
        "a faded, small pose is accepted");
  Check(reason == ElementError::Stale, "and leaves the reason alone");
  Check(WithPose(rig.frame, Element(), ElementPose{}, CENTER, &reason).IsNone() && reason == ElementError::None,
        "a none handle is refused as for Box");
}

// Poses whose scales cancel to within rounding still rest on whole pixels. A parent at 0.8 and a
// child at 1.25001 multiply to a hair above 1, which an exact comparison with 1 would treat as
// scaling and leave the corner on a fraction. The setup checks the product is not exactly 1.
void TestComposedUnitScaleStillSnaps() {
  constexpr float PARENT_SCALE = 0.8f;
  constexpr float CHILD_SCALE = 1.25001f;
  constexpr float SCALE_TOLERANCE = 1e-5f;
  const float product = PARENT_SCALE * CHILD_SCALE;
  Check(product != 1.0f && std::fabs(product - 1.0f) < SCALE_TOLERANCE, "the scales cancel but not exactly");
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 101_dp, .height = 41_dp}}, FullStyle())};
  const Element parent = Box(rig.frame, {.padding = DpInsets::All(13_dp)}, FullStyle(), kids);
  Check(!WithPose(rig.frame, parent, ElementPose{.scale = PARENT_SCALE}).IsNone(), "the parent pose is accepted");
  Check(!WithPose(rig.frame, kids[0], ElementPose{.offset = {7, 3}, .scale = CHILD_SCALE}).IsNone(), "and so is the child's");
  const Vec2 size = Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, parent, {10, 20, size.x, size.y}), "the tree arranges");
  const Rect painted = Painted(rig.frame, kids[0]);
  Check(painted.x == SnapPixel(painted.x) && painted.y == SnapPixel(painted.y), "the child's corner is on a whole pixel");
  const Rect arranged = Arranged(rig.frame, kids[0]);
  Check(!(painted.x == arranged.x && painted.y == arranged.y), "after the poses really moved it");
}

// Alpha reaches the children. A parent at alpha 0.5 painted at opacity 0.5 draws its child at 0.25
// too. The baseline is the child drawn at the parent's incoming opacity, which is what passing the
// state past the parent would give.
void TestAlphaAndPosesReachDescendants() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 300, 200};
  const Style childStyle{Rgb(0xAA5500), {}, 0_dp, 4_dp, {}};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, childStyle)};
    const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, FullStyle(), kids);
    Check(!WithPose(rig.frame, parent, ElementPose{.alpha = 0.5f}).IsNone(), "the parent is faded");
    (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, parent, rect), "the tree arranges");
    auto actual = NewDrawList();
    Paint(rig.frame, parent, *actual, {.opacity = 0.5f});
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle(), 0.25f);
    DrawBox(*expected, rig.frame.Ui(), Arranged(rig.frame, kids[0]), {Rgb(0xAA5500), 4.0f, {}, 0.0f, {}}, 0.25f);
    Check(SameDrawing(*actual, *expected), "the child draws at the product of the alpha and the opacity");
    auto parentOpacity = NewDrawList();
    DrawBox(*parentOpacity, rig.frame.Ui(), rect, FullBoxStyle(), 0.25f);
    DrawBox(*parentOpacity, rig.frame.Ui(), Arranged(rig.frame, kids[0]), {Rgb(0xAA5500), 4.0f, {}, 0.0f, {}}, 0.5f);
    Check(!SameDrawing(*actual, *parentOpacity), "and not at the incoming opacity");
  }
}

// Three nested poses with scales of 0.5, 4 and 0.25 about different origins and offsets. Each
// rectangle only comes out right if the scales multiply and the offsets compose through the
// parents' scales, and the alphas multiply to an eighth. No prefix of the scales multiplies to 1,
// which would snap, so the hand-computed corners need no rounding.
void TestThreeLevelPosesCompose() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 400, 300};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const std::array<Element, 1> inner{Box(rig.frame, {.item = {.width = 40_dp, .height = 20_dp}}, FullStyle())};
    const std::array<Element, 1> middle{Box(rig.frame, {.padding = DpInsets::All(12_dp)}, FullStyle(), inner)};
    const Element outer = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, FullStyle(), middle);
    const ElementPose poses[3] = {{.alpha = 0.5f, .offset = {16, 0}, .scale = 0.5f},
                                  {.alpha = 0.5f, .offset = {0, 8}, .scale = 4.0f},
                                  {.alpha = 0.5f, .offset = {8, 8}, .scale = 0.25f}};
    const Origin origins[3] = {TOP_RIGHT, CENTER, Origin{0.0f, 0.0f}};
    const Element nodes[3] = {outer, middle[0], inner[0]};
    for (int level = 0; level < 3; ++level) {
      Check(!WithPose(rig.frame, nodes[level], poses[level], origins[level]).IsNone(), "each level is posed");
    }
    (void)Measure(rig.frame, outer, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, outer, rect), "the tree arranges");
    auto actual = NewDrawList();
    Paint(rig.frame, outer, *actual);

    // Maps a corner through one level's pose, about that level's own rectangle.
    const auto through = [&](int level, Vec2 point) {
      const Rect r = Arranged(rig.frame, nodes[level]);
      const Vec2 o{r.x + origins[level].x * r.width, r.y + origins[level].y * r.height};
      const ElementPose &p = poses[level];
      return Vec2{o.x + p.scale * (point.x - o.x) + rig.scale.Px(p.offset.x),
                  o.y + p.scale * (point.y - o.y) + rig.scale.Px(p.offset.y)};
    };
    auto expected = NewDrawList();
    float alpha = 1.0f;
    for (int level = 0; level < 3; ++level) {
      const Rect r = Arranged(rig.frame, nodes[level]);
      Vec2 corner = r.Position();
      float scale = 1.0f;
      for (int up = level; up >= 0; --up) {
        corner = through(up, corner);
        scale *= poses[up].scale;
      }
      alpha *= poses[level].alpha;
      DrawBox(*expected, rig.frame.Ui(), {corner.x, corner.y, r.width * scale, r.height * scale}, FullBoxStyle(), alpha);
    }
    Check(SameDrawing(*actual, *expected), "every level lands where the composed mapping puts it");
    Check(alpha == 0.125f, "and the alphas multiplied to an eighth");
  }
}

// The full box style with the given overflow.
Style WithOverflow(Overflow overflow) {
  Style style = FullStyle();
  style.overflow = overflow;
  return style;
}

// The clip a box of `rect` keeps its children in, worked out by hand from the 12 reference pixel
// radius of FullStyle, clamped to half the shorter side as DrawBox clamps it.
Rect HandClip(const FrameRig &rig, const Rect &rect) {
  return InsideCorners(rect, std::min(rig.scale.Px(12.0f), std::min(rect.width, rect.height) / 2.0f));
}

// Pushes `clip` on `list` the way a clipping box does.
void PushClip(ImDrawList &list, const Rect &clip) {
  list.PushClipRect({clip.x, clip.y}, {clip.Right(), clip.Bottom()}, true);
}

// A clipping box draws its own box, then its children inside the clip of its rectangle less the
// rounded corners, then lifts the clip. The same tree set to Visible draws the same triangles with
// no clip, so only the commands tell them apart, and that is the baseline.
void TestClipKeepsChildrenInside() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 300, 200};
  const Style childStyle{Rgb(0xAA5500), {}, 0_dp, 4_dp, {}};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    const auto draw = [&](Overflow overflow, Rect &childRect) {
      rig.Begin();
      const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, childStyle)};
      const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, WithOverflow(overflow), kids);
      (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, parent, rect), "the tree arranges");
      childRect = Arranged(rig.frame, kids[0]);
      auto list = NewDrawList();
      Paint(rig.frame, parent, *list);
      return list;
    };
    Rect childRect, visibleChild;
    const auto clipped = draw(Overflow::Clip, childRect);
    const auto visible = draw(Overflow::Visible, visibleChild);

    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    PushClip(*expected, HandClip(rig, rect));
    DrawBox(*expected, rig.frame.Ui(), childRect, {Rgb(0xAA5500), 4.0f, {}, 0.0f, {}});
    expected->PopClipRect();
    Check(SameDrawing(*clipped, *expected) && SameCommands(*clipped, *expected), "the child is drawn inside the clip");
    Check(SameDrawing(*clipped, *visible), "the clip leaves the triangles alone");
    Check(!SameCommands(*clipped, *visible), "while an unclipped box has other commands");
  }
}

// A clip inside a clip intersects with it. The inner box is wider than the outer one, which does
// not shrink it, so its own clip reaches past the outer clip. Without the intersection the
// grandchild would be clipped to the wider rectangle.
void TestNestedClipsIntersect() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 240, 160};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const std::array<Element, 1> grand{Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, FullStyle())};
    const std::array<Element, 1> inner{Box(rig.frame, {.item = {.width = 400_dp, .height = 60_dp}}, WithOverflow(Overflow::Clip), grand)};
    const Element outer = Box(rig.frame, {}, WithOverflow(Overflow::Clip), inner);
    (void)Measure(rig.frame, outer, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, outer, rect), "the tree arranges");
    auto actual = NewDrawList();
    Paint(rig.frame, outer, *actual);

    const Rect innerRect = Arranged(rig.frame, inner[0]);
    const Rect outerClip = HandClip(rig, rect);
    Check(HandClip(rig, innerRect).Right() > outerClip.Right(), "the inner clip reaches past the outer one");
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    PushClip(*expected, outerClip);
    DrawBox(*expected, rig.frame.Ui(), innerRect, FullBoxStyle());
    PushClip(*expected, HandClip(rig, innerRect));
    DrawBox(*expected, rig.frame.Ui(), Arranged(rig.frame, grand[0]), FullBoxStyle());
    expected->PopClipRect();
    expected->PopClipRect();
    Check(SameDrawing(*actual, *expected) && SameCommands(*actual, *expected), "the clips nest as hand-built ones do");
    const ImDrawCmd *last = nullptr;
    for (const ImDrawCmd &cmd : actual->CmdBuffer) {
      last = cmd.ElemCount > 0 ? &cmd : last;
    }
    Check(last != nullptr && last->ClipRect.z == outerClip.Right(), "and the grandchild is clipped at the outer clip's edge");
  }
}

// The clip follows the posed rectangle. A box scaled by a half about its top-right corner clips its
// child to the smaller rectangle, with the child mapped there too. The unposed box is the
// baseline, whose clip is the larger rectangle.
void TestClipFollowsPose() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 300, 200};
  const Style childStyle{Rgb(0xAA5500), {}, 0_dp, 4_dp, {}};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    const auto draw = [&](bool posed, Rect &childRect) {
      rig.Begin();
      const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, childStyle)};
      const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, WithOverflow(Overflow::Clip), kids);
      if (posed) {
        Check(!WithPose(rig.frame, parent, {.scale = 0.5f}, TOP_RIGHT).IsNone(), "the pose is accepted");
      }
      (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, parent, rect), "the tree arranges");
      childRect = Arranged(rig.frame, kids[0]);
      auto list = NewDrawList();
      Paint(rig.frame, parent, *list);
      return list;
    };
    Rect arranged, unposedChild;
    const auto posed = draw(true, arranged);
    const auto unposed = draw(false, unposedChild);

    const Rect shrunk{0.5f * rect.x + 0.5f * rect.Right(), rect.y, rect.width / 2, rect.height / 2};
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), shrunk, FullBoxStyle());
    PushClip(*expected, HandClip(rig, shrunk));
    DrawBox(*expected, rig.frame.Ui(),
            {0.5f * arranged.x + 0.5f * rect.Right(), 0.5f * arranged.y + 0.5f * rect.y, arranged.width / 2, arranged.height / 2},
            {Rgb(0xAA5500), 4.0f, {}, 0.0f, {}});
    expected->PopClipRect();
    Check(SameDrawing(*posed, *expected) && SameCommands(*posed, *expected), "the clip is the posed rectangle's");
    Check(!SameCommands(*posed, *unposed), "and differs from the unposed clip");
  }
}

// The box's own shadow is drawn before the clip starts, so it is not clipped by it. The first
// command holds everything DrawBox draws, shadow included, under the clip the list started with,
// and the child's command is the one under the box's clip.
void TestClipLeavesOwnShadowAlone() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 300, 200};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, FullStyle())};
    const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, WithOverflow(Overflow::Clip), kids);
    (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, parent, rect), "the tree arranges");
    auto actual = NewDrawList();
    Paint(rig.frame, parent, *actual);

    auto boxOnly = NewDrawList();
    DrawBox(*boxOnly, rig.frame.Ui(), rect, FullBoxStyle());
    Style flat = FullStyle();
    flat.elevation = {};
    auto noShadow = NewDrawList();
    PaintBox(rig, flat, rect, *noShadow);
    Check(boxOnly->IdxBuffer.Size > noShadow->IdxBuffer.Size, "the box casts a shadow");
    const ImDrawCmd &first = actual->CmdBuffer[0];
    Check(first.ElemCount == static_cast<unsigned>(boxOnly->IdxBuffer.Size), "the first command holds the box and its shadow");
    Check(first.ClipRect.x == boxOnly->CmdBuffer[0].ClipRect.x && first.ClipRect.z == boxOnly->CmdBuffer[0].ClipRect.z,
          "under the clip the list started with");
    Check(actual->CmdBuffer[1].ClipRect.x > first.ClipRect.x, "and the child's command is clipped");
  }
}

// A braced pose such as {.scale = 0.5f} names the element pose, since the timeline pose has its
// own function, and draws like the spelled-out element pose.
void TestBracedPoseCompiles() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 200, 100};
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Element box = Box(rig.frame, {}, FullStyle());
  Check(!WithPose(rig.frame, box, {.scale = 0.5f}).IsNone(), "a braced pose is accepted");
  (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, box, rect), "the box arranges");
  auto braced = NewDrawList();
  Paint(rig.frame, box, *braced);
  auto spelled = NewDrawList();
  (void)PaintPosedBox(rig, rect, ElementPose{.scale = 0.5f}, CENTER, *spelled);
  Check(SameDrawing(*braced, *spelled), "and draws like the element pose");
}

// An alpha outside 0 to 1 is clamped for each element. A child at alpha 2 under a parent at 0.5
// is no more opaque than 0.5, where an unclamped product would give 1. A parent and child both at
// alpha -1 draw nothing, where unclamped alphas would multiply to a fully visible 1. Opacities of
// 0 and 1 for the same boxes are the references, and they differ from each other.
void TestAlphaIsClamped() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 300, 200};
  const Style childStyle{Rgb(0xAA5500), {}, 0_dp, 4_dp, {}};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    const auto draw = [&](float parentAlpha, float childAlpha, Rect &childRect) {
      rig.Begin();
      const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, childStyle)};
      const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, FullStyle(), kids);
      Check(!WithPose(rig.frame, parent, {.alpha = parentAlpha}).IsNone(), "the parent pose is accepted");
      Check(!WithPose(rig.frame, kids[0], {.alpha = childAlpha}).IsNone(), "and so is the child's");
      (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, parent, rect), "the tree arranges");
      childRect = Arranged(rig.frame, kids[0]);
      auto list = NewDrawList();
      Paint(rig.frame, parent, *list);
      return list;
    };
    const auto reference = [&](float parentOpacity, float childOpacity, const Rect &childRect) {
      auto list = NewDrawList();
      DrawBox(*list, rig.frame.Ui(), rect, FullBoxStyle(), parentOpacity);
      DrawBox(*list, rig.frame.Ui(), childRect, {Rgb(0xAA5500), 4.0f, {}, 0.0f, {}}, childOpacity);
      return list;
    };
    Rect childRect;
    const auto over = draw(0.5f, 2.0f, childRect);
    Check(SameDrawing(*over, *reference(0.5f, 0.5f, childRect)), "alpha 2 under a half draws at a half");
    const auto negative = draw(-1.0f, -1.0f, childRect);
    Check(SameDrawing(*negative, *reference(0.0f, 0.0f, childRect)), "two alphas of -1 draw nothing");
    Check(!SameDrawing(*reference(0.0f, 0.0f, childRect), *reference(1.0f, 1.0f, childRect)), "unlike the opaque drawing");
  }
}

// Builds a childless 200 by 100 box in a fresh frame, poses it when `pose` is given, decorates it
// with `decor`, arranges it at `rect` and paints it into `list` with `opacity`.
void PaintDecorated(FrameRig &rig, const Rect &rect, const Decor &decor, const std::optional<ElementPose> &pose,
                    ImDrawList &list, float opacity = 1.0f, Overflow overflow = Overflow::Visible) {
  rig.Begin();
  const Element box = Box(rig.frame, {}, WithOverflow(overflow));
  Check(!Decorate(rig.frame, box, decor).IsNone(), "the decor is accepted");
  if (pose) {
    Check(!WithPose(rig.frame, box, *pose).IsNone(), "the pose is accepted");
  }
  (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, box, rect), "the box arranges");
  Paint(rig.frame, box, list, {.opacity = opacity});
}

// The corner radius FullStyle's box has in `rect` on this output, in pixels.
float BoxRounding(const FrameRig &rig, const Rect &rect) {
  return std::min(rig.scale.Px(12.0f), std::min(rect.width, rect.height) / 2.0f);
}

constexpr float EFFECT_PROGRESS = 0.5f;
constexpr float RING_STRENGTH = 0.8f;
const Color EFFECT_COLOR = Rgb(0x3399FF);

// Each effect draws what the older function draws at the box's rectangle and radius, and the box
// without the effect is the baseline, since a node that dropped the effect would draw that and
// nothing else in the comparison would notice. The glow comes before the box, the rest after it.
void TestEachEffectMatchesItsFunction() {
  ImGuiScope imgui(true);
  const Rect rect{100, 50, 200, 100};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    auto plain = NewDrawList();
    PaintDecorated(rig, rect, {}, std::nullopt, *plain);
    const Theme &theme = rig.frame.ActiveTheme();
    const float rounding = BoxRounding(rig, rect);

    auto glow = NewDrawList();
    PaintDecorated(rig, rect, {.glow = GlowEffect{theme.glow.focus, EFFECT_COLOR}}, std::nullopt, *glow);
    auto expected = NewDrawList();
    DrawGlow(*expected, rect, rounding, theme.glow.focus, EFFECT_COLOR, rig.scale);
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    Check(SameDrawing(*glow, *expected) && SameCommands(*glow, *expected), "a glow is drawn before the box");
    Check(!SameDrawing(*glow, *plain), "which the plain box lacks");

    auto selection = NewDrawList();
    PaintDecorated(rig, rect, {.selection = true}, std::nullopt, *selection);
    expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    DrawSelection(*expected, rect, rounding, theme, rig.scale);
    Check(SameDrawing(*selection, *expected), "a selection is drawn after the box");
    Check(!SameDrawing(*selection, *plain), "which the plain box lacks");

    auto ring = NewDrawList();
    PaintDecorated(rig, rect, {.focus_ring = RING_STRENGTH}, std::nullopt, *ring);
    expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    DrawFocusRing(*expected, rect, rounding, theme, rig.scale, RING_STRENGTH);
    Check(SameDrawing(*ring, *expected), "a focus ring is drawn after the box at its strength");
    Check(!SameDrawing(*ring, *plain), "which the plain box lacks");

    auto shine = NewDrawList();
    PaintDecorated(rig, rect, {.shine = EFFECT_PROGRESS}, std::nullopt, *shine);
    expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    DrawShine(*expected, rect, rounding, EFFECT_PROGRESS, theme.shine, theme.colors.effect.shine, rig.scale);
    Check(SameDrawing(*shine, *expected) && SameCommands(*shine, *expected), "a shine is drawn after the box");
    Check(!SameDrawing(*shine, *plain), "which the plain box lacks");
  }
}

// All the effects on a clipping box with a child draw in the planned order, glow, box, selection,
// focus ring, then the clip around the child, then the shine. Everything of the box's own is
// outside its clip, shine included, so SameCommands finds the clip only around the child. The
// selection and ring overlap the box, so swapping them would change the vertices.
void TestEffectsDrawInOrder() {
  ImGuiScope imgui(true);
  const Rect rect{100, 50, 300, 200};
  const Style childStyle{Rgb(0xAA5500), {}, 0_dp, 4_dp, {}};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 40_dp}}, childStyle)};
    const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, WithOverflow(Overflow::Clip), kids);
    const Decor decor{.glow = GlowEffect{DEFAULT_THEME.glow.focus, EFFECT_COLOR},
                      .shine = EFFECT_PROGRESS,
                      .focus_ring = RING_STRENGTH,
                      .selection = true};
    Check(!Decorate(rig.frame, parent, decor).IsNone(), "the decor is accepted");
    (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, parent, rect), "the tree arranges");
    auto actual = NewDrawList();
    Paint(rig.frame, parent, *actual);

    const Theme &theme = rig.frame.ActiveTheme();
    const float rounding = BoxRounding(rig, rect);
    auto expected = NewDrawList();
    DrawGlow(*expected, rect, rounding, theme.glow.focus, EFFECT_COLOR, rig.scale);
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    DrawSelection(*expected, rect, rounding, theme, rig.scale);
    DrawFocusRing(*expected, rect, rounding, theme, rig.scale, RING_STRENGTH);
    PushClip(*expected, HandClip(rig, rect));
    DrawBox(*expected, rig.frame.Ui(), Arranged(rig.frame, kids[0]), {Rgb(0xAA5500), 4.0f, {}, 0.0f, {}});
    expected->PopClipRect();
    DrawShine(*expected, rect, rounding, EFFECT_PROGRESS, theme.shine, theme.colors.effect.shine, rig.scale);
    Check(SameDrawing(*actual, *expected) && SameCommands(*actual, *expected), "the effects draw in the planned order");
  }
}

// Effects fade with the alpha a node inherits, so a node at alpha 0.5 painted at opacity 0.5 draws
// each of them at 0.25, and the ring's strength scales by it too. The plain 0.5 drawing is the
// baseline that shows the alpha reached the effects and not only the box.
void TestEffectsMultiplyAlpha() {
  ImGuiScope imgui(true);
  const Rect rect{100, 50, 200, 100};
  const Decor decor{.glow = GlowEffect{DEFAULT_THEME.glow.focus, EFFECT_COLOR},
                    .shine = EFFECT_PROGRESS,
                    .focus_ring = RING_STRENGTH,
                    .selection = true};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    auto actual = NewDrawList();
    PaintDecorated(rig, rect, decor, ElementPose{.alpha = 0.5f}, *actual, 0.5f);
    const Theme &theme = rig.frame.ActiveTheme();
    const float rounding = BoxRounding(rig, rect);
    constexpr float QUARTER = 0.25f;
    auto expected = NewDrawList();
    DrawGlow(*expected, rect, rounding, theme.glow.focus, EFFECT_COLOR, rig.scale, QUARTER);
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle(), QUARTER);
    DrawSelection(*expected, rect, rounding, theme, rig.scale, QUARTER);
    DrawFocusRing(*expected, rect, rounding, theme, rig.scale, RING_STRENGTH * QUARTER);
    DrawShine(*expected, rect, rounding, EFFECT_PROGRESS, theme.shine, theme.colors.effect.shine, rig.scale, QUARTER);
    Check(SameDrawing(*actual, *expected), "every effect draws at the product of alpha and opacity");

    auto opacityOnly = NewDrawList();
    PaintDecorated(rig, rect, decor, std::nullopt, *opacityOnly, 0.5f);
    Check(!SameDrawing(*actual, *opacityOnly), "which is not the half-opacity drawing");
  }
}

// Posing and decorating share a record, so neither undoes the other whichever comes first.
void TestDecorateKeepsPose() {
  ImGuiScope imgui(true);
  const Rect rect{100, 50, 201, 100};
  const Decor decor{.glow = GlowEffect{DEFAULT_THEME.glow.focus, EFFECT_COLOR}};
  FrameRig rig(1920, 1080);
  const ElementPose pose{.offset = {12, -7}, .scale = 0.5f};
  const auto draw = [&](bool decorateFirst) {
    rig.Begin();
    const Element box = Box(rig.frame, {}, FullStyle());
    if (decorateFirst) {
      (void)Decorate(rig.frame, box, decor);
    }
    (void)WithPose(rig.frame, box, pose);
    if (!decorateFirst) {
      (void)Decorate(rig.frame, box, decor);
    }
    (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, box, rect), "the box arranges");
    auto list = NewDrawList();
    Paint(rig.frame, box, *list);
    return list;
  };
  const auto first = draw(true);
  const auto second = draw(false);
  Check(SameDrawing(*first, *second), "decorating then posing draws as posing then decorating");
  auto posedOnly = NewDrawList();
  (void)PaintPosedBox(rig, rect, pose, CENTER, *posedOnly);
  Check(!SameDrawing(*first, *posedOnly), "and the glow is part of it");
  Check(first->VtxBuffer.Size > posedOnly->VtxBuffer.Size, "as extra vertices");
}

// A NaN strength is refused with InvalidDecor, and a stale handle as for Box. Each strength is
// tried alone so a check on one would not cover the other.
void TestInvalidDecorRefused() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Element box = Box(rig.frame, {}, FullStyle());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  for (const Decor &bad : {Decor{.shine = nan}, Decor{.focus_ring = nan}}) {
    ElementError reason = ElementError::None;
    Check(Decorate(rig.frame, box, bad, &reason).IsNone() && reason == ElementError::InvalidDecor, "a NaN strength is refused");
  }
  ElementError reason = ElementError::None;
  Check(!Decorate(rig.frame, box, Decor{.shine = 0.5f, .focus_ring = 1.0f}, &reason).IsNone() && reason == ElementError::None,
        "valid strengths are accepted without touching the reason");
  const Element old = box;
  rig.Begin();
  Check(Decorate(rig.frame, old, {}, &reason).IsNone() && reason == ElementError::Stale, "a stale handle is refused");
}

// A box too short for its radius clamps the clip's rounding as it clamps the drawn corners. At 4K
// the 12 dp radius is 24 px, and in a box 16 px tall only 8 px of it fit, so a clip rounded by the
// full 24 px would pull each side in by far more and leave a sliver. The clamped clip is the hand
// one, which clamps explicitly.
void TestShortBoxClipClampsRadius() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 300, 16};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const std::array<Element, 1> kids{Box(rig.frame, {.item = {.width = 100_dp, .height = 8_dp}}, FullStyle())};
    const Element parent = Box(rig.frame, {}, WithOverflow(Overflow::Clip), kids);
    (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, parent, rect), "the tree arranges");
    auto actual = NewDrawList();
    Paint(rig.frame, parent, *actual);
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    PushClip(*expected, InsideCorners(rect, std::min(rig.scale.Px(12.0f), rect.height / 2.0f)));
    DrawBox(*expected, rig.frame.Ui(), Arranged(rig.frame, kids[0]), FullBoxStyle());
    expected->PopClipRect();
    Check(SameDrawing(*actual, *expected) && SameCommands(*actual, *expected), "the clip uses the clamped radius");
    if (output.height > 1080) {
      Check(rig.scale.Px(12.0f) > rect.height / 2.0f, "so the clamp really mattered at this output");
    }
  }
}

constexpr float ICON_RADIUS_DP = 8;

// What the icon-glow tests share, a clipping card holding an image with a glow, optionally under
// another clipping box. Returns the list and the arranged rectangles.
struct IconScene {
  std::unique_ptr<ImDrawList> list;
  Rect card;
  Rect icon;
};

IconScene PaintIconScene(FrameRig &rig, TestTexture &texture, const Rect &outerRect, const Rect &cardRect, bool unclipped,
                         bool outerClip) {
  rig.Begin();
  const Element image = Image(rig.frame, &texture.data, {200, 100}, ImageFit::Cover, Dp(ICON_RADIUS_DP),
                              {.width = 40_dp, .height = 40_dp});
  Check(!Decorate(rig.frame, image, {.glow = GlowEffect{rig.frame.ActiveTheme().glow.trophy, EFFECT_COLOR}, .unclipped = unclipped}).IsNone(),
        "the glow is accepted");
  const std::array<Element, 1> inCard{image};
  const Element card = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, WithOverflow(Overflow::Clip), inCard);
  Element root = card;
  Rect rootRect = cardRect;
  if (outerClip) {
    const std::array<Element, 1> inOuter{card};
    root = Box(rig.frame, {}, WithOverflow(Overflow::Clip), inOuter);
    rootRect = outerRect;
  }
  (void)Measure(rig.frame, root, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, root, rootRect), "the scene arranges");
  IconScene scene{NewDrawList(), Arranged(rig.frame, card), Arranged(rig.frame, image)};
  Paint(rig.frame, root, *scene.list);
  return scene;
}

// A glow on an image inside a clipping card draws a glow clipped to the card: the same
// triangles, in the card's clip. A tree clips everything in the card, so the image is under the
// clip too, which a clip around the whole icon leaves unchanged as the icon sits inside the card.
// The commands are the hand-built tree sequence, and the glow command's clip is InsideCorners of
// the card.
void TestGlowInClippingCardMatchesToast() {
  ImGuiScope imgui;
  TestTexture texture;
  const Rect cardRect{100, 50, 300, 120};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    IconScene scene = PaintIconScene(rig, texture, {}, cardRect, false, false);
    const Theme &theme = rig.frame.ActiveTheme();
    const float radius = rig.scale.Px(ICON_RADIUS_DP);
    const Rect inside = HandClip(rig, cardRect);

    auto toast = NewDrawList();
    DrawBox(*toast, rig.frame.Ui(), cardRect, FullBoxStyle());
    PushClip(*toast, inside);
    DrawGlow(*toast, scene.icon, radius, theme.glow.trophy, EFFECT_COLOR, rig.scale);
    toast->PopClipRect();
    DrawImage(*toast, rig.frame.Ui(), &texture.data, {200, 100}, scene.icon, ICON_RADIUS_DP, ImageFit::Cover);
    Check(SameDrawing(*scene.list, *toast), "the triangles are the toast's");

    auto tree = NewDrawList();
    DrawBox(*tree, rig.frame.Ui(), cardRect, FullBoxStyle());
    PushClip(*tree, inside);
    DrawGlow(*tree, scene.icon, radius, theme.glow.trophy, EFFECT_COLOR, rig.scale);
    DrawImage(*tree, rig.frame.Ui(), &texture.data, {200, 100}, scene.icon, ICON_RADIUS_DP, ImageFit::Cover);
    tree->PopClipRect();
    Check(SameCommands(*scene.list, *tree), "and the commands are the tree sequence");
    Check(scene.list->CmdBuffer[1].ClipRect.x == static_cast<float>(inside.x), "with the glow under the card's clip");
  }
}

// An unclipped glow returns to the clip outside the card, so it draws as the toast's glow does,
// in the clip the list started with. It escapes only the nearest clipping box, so
// with a clip above the card the glow still gets that one. The clipped glow is the baseline.
void TestUnclippedGlowEscapesNearestClip() {
  ImGuiScope imgui;
  TestTexture texture;
  const Rect outerRect{60, 30, 380, 200};
  const Rect cardRect{100, 50, 300, 120};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    IconScene alone = PaintIconScene(rig, texture, {}, cardRect, true, false);
    const Theme &theme = rig.frame.ActiveTheme();
    const float radius = rig.scale.Px(ICON_RADIUS_DP);
    const Rect inside = HandClip(rig, cardRect);

    auto toast = NewDrawList();
    DrawBox(*toast, rig.frame.Ui(), cardRect, FullBoxStyle());
    DrawGlow(*toast, alone.icon, radius, theme.glow.trophy, EFFECT_COLOR, rig.scale);
    DrawImage(*toast, rig.frame.Ui(), &texture.data, {200, 100}, alone.icon, ICON_RADIUS_DP, ImageFit::Cover);
    Check(SameDrawing(*alone.list, *toast), "the triangles are the toast's own");
    const ImVec4 started = toast->CmdBuffer[0].ClipRect;
    Check(alone.list->CmdBuffer[0].ClipRect.x == started.x && alone.list->CmdBuffer[0].ClipRect.z == started.z,
          "and the box is under the list's own clip");
    Check(alone.list->CmdBuffer[0].ElemCount == toast->CmdBuffer[0].ElemCount,
          "and so is the glow, which left the card's clip, since the first command holds both");
    Check(alone.list->CmdBuffer[0].ElemCount > 0, "and that command is not empty");

    IconScene clipped = PaintIconScene(rig, texture, {}, cardRect, false, false);
    Check(!SameCommands(*alone.list, *clipped.list), "unlike the clipped glow");

    IconScene nested = PaintIconScene(rig, texture, outerRect, cardRect, true, true);
    const Rect outside = HandClip(rig, outerRect);
    auto tree = NewDrawList();
    DrawBox(*tree, rig.frame.Ui(), outerRect, FullBoxStyle());
    PushClip(*tree, outside);
    DrawBox(*tree, rig.frame.Ui(), nested.card, FullBoxStyle());
    PushClip(*tree, HandClip(rig, nested.card));
    tree->PushClipRect({outside.x, outside.y}, {outside.Right(), outside.Bottom()}, false);
    DrawGlow(*tree, nested.icon, radius, theme.glow.trophy, EFFECT_COLOR, rig.scale);
    tree->PopClipRect();
    DrawImage(*tree, rig.frame.Ui(), &texture.data, {200, 100}, nested.icon, ICON_RADIUS_DP, ImageFit::Cover);
    tree->PopClipRect();
    tree->PopClipRect();
    Check(SameDrawing(*nested.list, *tree) && SameCommands(*nested.list, *tree),
          "with a clip above, the glow escapes the card's clip only");
    (void)inside;
  }
}

// True when two lists hold the same triangles to within `slack` pixels in position, and exactly in
// colour, texture coordinates and indices. A pressed button is scaled by two different but
// equivalent sums of floats, so its corners can differ in the last digits.
bool NearDrawing(const ImDrawList &a, const ImDrawList &b, float slack) {
  if (a.VtxBuffer.Size != b.VtxBuffer.Size || a.IdxBuffer.Size != b.IdxBuffer.Size) {
    return false;
  }
  for (int i = 0; i < a.VtxBuffer.Size; ++i) {
    const ImDrawVert &left = a.VtxBuffer[i];
    const ImDrawVert &right = b.VtxBuffer[i];
    if (std::fabs(left.pos.x - right.pos.x) > slack || std::fabs(left.pos.y - right.pos.y) > slack ||
        left.uv.x != right.uv.x || left.uv.y != right.uv.y || left.col != right.col) {
      return false;
    }
  }
  for (int i = 0; i < a.IdxBuffer.Size; ++i) {
    if (a.IdxBuffer[i] != b.IdxBuffer[i]) {
      return false;
    }
  }
  return true;
}

constexpr float STATE_SLACK = 0.001f;

// Posed text keeps its size and moves with its centre, and so do the effects around it. A glow, a
// selection and a focus ring on text scaled to a half about its top-right corner go around the
// glyphs where they are drawn, which is the moved centre at the laid-out size. The same effects
// around the scaled box, which is what a plain rectangle mapping would give, are the baseline.
void TestEffectsFollowPosedText() {
  ImGuiScope imgui(true);
  const ElementPose pose{.offset = {16, 8}, .scale = 0.5f};
  const Decor decor{.glow = GlowEffect{DEFAULT_THEME.glow.focus, EFFECT_COLOR}, .focus_ring = RING_STRENGTH, .selection = true};
  for (const Output &output : OUTPUTS) {
    TextRig text(imgui, output);
    FrameRig &rig = text.rig;
    const TextProps props = Body("A short title");
    rig.Begin();
    const Element label = Text(rig.frame, props);
    Check(!WithPose(rig.frame, label, pose, TOP_RIGHT).IsNone() && !Decorate(rig.frame, label, decor).IsNone(), "text is posed and decorated");
    const Vec2 size = Measure(rig.frame, label, {INFINITE_SPACE, INFINITE_SPACE});
    const Rect rect{40, 60, size.x, size.y};
    Check(Arrange(rig.frame, label, rect), "the text arranges");
    auto actual = NewDrawList();
    Paint(rig.frame, label, *actual);

    const Theme &theme = rig.frame.ActiveTheme();
    const Vec2 centre = rect.Center();
    const float offsetX = rig.scale.Px(16), offsetY = rig.scale.Px(8);
    const Vec2 moved{centre.x * 0.5f + (rect.Right() * 0.5f + offsetX), centre.y * 0.5f + (rect.y * 0.5f + offsetY)};
    const Rect around{moved.x - rect.width / 2.0f, moved.y - rect.height / 2.0f, rect.width, rect.height};
    const TextLayout layout = LayoutText(rig.frame.Ui(), props, INFINITE_SPACE);
    const auto draw = [&](const Rect &effectRect) {
      auto list = NewDrawList();
      DrawGlow(*list, effectRect, 0.0f, theme.glow.focus, EFFECT_COLOR, rig.scale);
      DrawText(*list, layout, props, around);
      DrawSelection(*list, effectRect, 0.0f, theme, rig.scale);
      DrawFocusRing(*list, effectRect, 0.0f, theme, rig.scale, RING_STRENGTH);
      return list;
    };
    Check(SameDrawing(*actual, *draw(around)), "the effects go around the glyphs where they are drawn");
    const Rect scaledBox{rect.x * 0.5f + (rect.Right() * 0.5f + offsetX), rect.y * 0.5f + (rect.y * 0.5f + offsetY), rect.width * 0.5f,
                         rect.height * 0.5f};
    Check(!SameDrawing(*actual, *draw(scaledBox)), "and not around the scaled box");
  }
}

// An image that letterboxes puts its glow around the picture, not the empty bars. A 200 by 100
// texture contained in a square is half as tall as the square, so the glow follows that rectangle,
// with the corners rounded by the radius clamped to it. The glow around the whole square is the
// baseline.
void TestImageEffectsUsePlacedRect() {
  ImGuiScope imgui;
  TestTexture texture;
  const Rect rect{100, 50, 100, 100};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const Element image = Image(rig.frame, &texture.data, {200, 100}, ImageFit::Contain, 8_dp, {.width = 100_dp, .height = 100_dp});
    const GlowEffect glow{DEFAULT_THEME.glow.focus, EFFECT_COLOR};
    Check(!Decorate(rig.frame, image, {.glow = glow}).IsNone(), "the glow is accepted");
    (void)Measure(rig.frame, image, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, image, rect), "the image arranges");
    auto actual = NewDrawList();
    Paint(rig.frame, image, *actual);
    const auto draw = [&](const Rect &around) {
      auto list = NewDrawList();
      DrawGlow(*list, around, std::min(rig.scale.Px(8.0f), std::min(around.width, around.height) / 2.0f), glow.glow, glow.color, rig.scale);
      DrawImage(*list, rig.frame.Ui(), &texture.data, {200, 100}, rect, 8.0f, ImageFit::Contain);
      return list;
    };
    const Rect placed{100, 75, 100, 50};
    Check(SameDrawing(*actual, *draw(placed)), "the glow follows the picture");
    Check(!SameDrawing(*actual, *draw(rect)), "and not the empty bars");
  }
}

// A glow whose radius or peak is not finite is refused with InvalidDecor, each alone, and a sound
// glow is accepted.
void TestInvalidGlowRefused() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Element box = Box(rig.frame, {}, FullStyle());
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  for (const Glow &bad : {Glow{nan, 0.3f}, Glow{infinity, 0.3f}, Glow{12.0f, nan}}) {
    ElementError reason = ElementError::None;
    Check(Decorate(rig.frame, box, {.glow = GlowEffect{bad, EFFECT_COLOR}}, &reason).IsNone() && reason == ElementError::InvalidDecor,
          "a glow that is not finite is refused");
  }
  Check(!Decorate(rig.frame, box, {.glow = GlowEffect{DEFAULT_THEME.glow.focus, EFFECT_COLOR}}).IsNone(), "a sound glow is accepted");
}

// The look of a button-like box, as button.cpp draws a secondary or an accent button. The fill and
// border are the variant's, the hover, focus and pressed overlays are the theme's, and the press
// scale and the ring come from the states. An accent button has no overlays, and its fill blends
// to the hover and pressed colours instead.
struct ButtonLook {
  bool accent;
  Color fill;
  Color border;
};

ButtonLook SecondaryLook(const Theme &theme) { return {false, theme.colors.surface.overlay, theme.colors.border.standard}; }
ButtonLook PrimaryLook(const Theme &theme) { return {true, theme.colors.accent.fill, {}}; }

Style ButtonLikeStyle(const Theme &theme, const ButtonLook &look) {
  Style style{look.fill, look.border, Dp(theme.border.hair), Dp(ButtonDefaults(theme).radius), {}};
  const auto &colors = theme.colors;
  if (look.accent) {
    style.states.hover.fill = colors.accent.fill_hover;
    style.states.focus.fill = colors.accent.fill_hover;
    style.states.pressed.fill = colors.accent.fill_pressed;
  } else {
    style.states.hover.overlay = colors.state.hover;
    style.states.focus.overlay = colors.state.focus;
    style.states.pressed.overlay = colors.state.pressed;
  }
  style.states.focus.focus_ring = true;
  style.states.pressed.scale = theme.interaction.press_scale;
  return style;
}

// button.cpp's drawing of a button in `rect`, copied from its DrawButton: the box in the pressed
// rectangle, the theme's overlays for a button that is not an accent one, then the focus ring.
void ReplicaButton(ImDrawList &list, const UiContext &context, const Rect &rect, const ButtonLook &look, ItemVisuals visuals,
                   bool disabled) {
  const auto &theme = context.theme;
  const auto &tokens = ButtonDefaults(theme);
  const float opacity = disabled ? theme.opacity.disabled : 1.0f;
  visuals = {ClampUnit(visuals.hover), ClampUnit(visuals.focus), ClampUnit(visuals.press)};
  const float scale = 1.0f - (1.0f - theme.interaction.press_scale) * visuals.press;
  const float width = rect.width * scale;
  const float height = rect.height * scale;
  const Rect drawn{rect.x + (rect.width - width) / 2.0f, rect.y + (rect.height - height) / 2.0f, width, height};
  const float radius = std::min(context.scale.Px(tokens.radius), std::min(drawn.width, drawn.height) / 2.0f);
  Color fill = look.fill;
  if (look.accent) {
    const Color raised = Lerp(theme.colors.accent.fill, theme.colors.accent.fill_hover, std::max(visuals.hover, visuals.focus));
    fill = Lerp(raised, theme.colors.accent.fill_pressed, visuals.press);
  }
  DrawBox(list, context, drawn, {fill, tokens.radius, look.border, theme.border.hair, {}}, opacity);
  if (!look.accent) {
    const auto &state = theme.colors.state;
    const std::pair<Color, float> overlays[] = {{state.hover, visuals.hover}, {state.focus, visuals.focus}, {state.pressed, visuals.press}};
    for (const auto &[color, amount] : overlays) {
      if (amount > 0.0f) {
        list.AddRectFilled({drawn.x, drawn.y}, {drawn.Right(), drawn.Bottom()}, ToImU32(color.MulAlpha(amount * opacity)), radius);
      }
    }
  }
  DrawFocusRing(list, drawn, radius, theme, context.scale, visuals.focus * opacity);
}

// Paints a childless box of `style` in `rect` with `input`, into `list`.
void PaintStyled(FrameRig &rig, const Style &style, const Rect &rect, const ItemInput &input, ImDrawList &list) {
  rig.Begin();
  const Element box = Box(rig.frame, {}, style);
  Check(SetInput(rig.frame, box, input), "the input is accepted");
  (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, box, rect), "the box arranges");
  Paint(rig.frame, box, list);
}

// The inputs the state tests run, from nothing to every highlight at once.
struct StateCase {
  ItemInput input;
  bool pressed;
};

const std::array<StateCase, 7> STATE_CASES{{{{{0.0f, 0.0f, 0.0f}, false, false}, false},
                                            {{{0.3f, 0.0f, 0.0f}, false, false}, false},
                                            {{{0.0f, 0.6f, 0.0f}, false, false}, false},
                                            {{{0.0f, 0.0f, 0.8f}, false, false}, true},
                                            {{{0.3f, 0.6f, 0.8f}, false, false}, true},
                                            {{{0.0f, 0.0f, 0.0f}, true, false}, false},
                                            {{{0.0f, 0.6f, 0.0f}, true, false}, false}}};

// A button-like box in each state draws what a copy of button.cpp's drawing does, for the
// secondary and the accent look, at every output size. The cases cover hover alone, focus alone,
// press alone, all three at once, and a disabled button, with and without the focus it kept. The
// resting button is the baseline each differs from. The press scale is a different sum of floats
// from the button's, so pressed cases agree to a thousandth of a pixel, the others exactly.
void TestButtonLikeStatesMatchReplica() {
  ImGuiScope imgui;
  const Rect rect{400, 300, 240, 56};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const Theme &theme = rig.frame.ActiveTheme();
    for (const ButtonLook &look : {SecondaryLook(theme), PrimaryLook(theme)}) {
      const Style style = ButtonLikeStyle(theme, look);
      auto resting = NewDrawList();
      PaintStyled(rig, style, rect, STATE_CASES[0].input, *resting);
      for (const StateCase &item : STATE_CASES) {
        auto actual = NewDrawList();
        PaintStyled(rig, style, rect, item.input, *actual);
        auto expected = NewDrawList();
        ReplicaButton(*expected, rig.frame.Ui(), rect, look, item.input.visuals, item.input.disabled);
        const float slack = item.pressed ? STATE_SLACK : 0.0f;
        Check(NearDrawing(*actual, *expected, slack), "the box draws what the button does");
        const bool isResting = &item == &STATE_CASES[0];
        Check(isResting || !NearDrawing(*actual, *resting, STATE_SLACK), "which differs from the resting button");
      }
    }
  }
}

// Lays `element` out in `rect` as a specimen posed at `visuals`, renders it into `context`'s list and
// returns the vertices it drew. A specimen takes no input and shows the pose, which is how the real
// controls are put in a state without a player to put them there.
template <class Make>
std::vector<ImDrawVert> RenderPosed(InteractContext &context, UiFrame &frame, const ItemVisuals &visuals, const Rect &rect, Make &&make) {
  frame.Begin(context.ui.theme, context.ui.scale, context.ui.fonts, context.store, context.now);
  const Element element = make(frame);
  Check(!element.IsNone(), "the component was built");
  Check(!Specimen(frame, element, visuals).IsNone(), "and posed");
  (void)Measure(frame, element, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(frame, element, rect), "and arranged");
  const int before = context.list.VtxBuffer.Size;
  Check(!Render(frame, element, context.list).error.has_value(), "and rendered");
  return std::vector<ImDrawVert>(context.list.VtxBuffer.begin() + before, context.list.VtxBuffer.end());
}

// The same through the real Button element: its vertices for a posed button are the box's. The
// button draws its label after the ring, so the box's vertices are the front of its slice of the
// list.
void TestButtonStatesMatchRealButton() {
  const Rect rect{400, 300, 240, 56};
  for (const bool accent : {false, true}) {
    for (const StateCase &item : STATE_CASES) {
      NavHarness nav;
      UiFrame frame;
      nav.Frame([&](InteractContext &context) {
        FrameRig rig(1920, 1080);
        const auto look = accent ? PrimaryLook(DEFAULT_THEME) : SecondaryLook(DEFAULT_THEME);
        const std::vector<ImDrawVert> real = RenderPosed(context, frame, item.input.visuals, rect, [&](UiFrame &built) {
          return Button(built, "button", {.label = "Save", .variant = accent ? ButtonVariant::Primary : ButtonVariant::Secondary, .disabled = item.input.disabled});
        });
        auto element = NewDrawList();
        PaintStyled(rig, ButtonLikeStyle(DEFAULT_THEME, look), rect, item.input, *element);
        const int count = element->VtxBuffer.Size;
        Check(count > 0 && static_cast<int>(real.size()) >= count, "the button drew the box");
        for (int i = 0; i < count; ++i) {
          const ImDrawVert &theirs = real[i];
          const ImDrawVert &mine = element->VtxBuffer[i];
          const float slack = item.pressed ? STATE_SLACK : 0.0f;
          Check(std::fabs(theirs.pos.x - mine.pos.x) <= slack && std::fabs(theirs.pos.y - mine.pos.y) <= slack &&
                    theirs.uv.x == mine.uv.x && theirs.uv.y == mine.uv.y && theirs.col == mine.col,
                "the real button's vertices are the box's");
        }
      });
    }
  }
}

// A list row's look: no fill, the row radius, a selection fill when selected, and the hover and
// focus overlays with the ring, as list.cpp draws a row.
Style RowLikeStyle(const Theme &theme) {
  Style style{{}, {}, 0_dp, Dp(ListDefaults(theme).row_radius), {}};
  style.states.selected.selection = true;
  style.states.hover.overlay = theme.colors.state.hover;
  style.states.focus.overlay = theme.colors.state.focus;
  style.states.focus.focus_ring = true;
  return style;
}

// The same through the real ListRow element, selected or not, hovered, focused or both, and
// disabled. The unselected row is the baseline that shows the selection fill is part of the drawing.
void TestRowStatesMatchRealRow() {
  const Rect rect{400, 300, 600, 64};
  struct Case {
    ItemVisuals visuals;
    bool selected;
    bool disabled;
  };
  const Case cases[] = {{{0.0f, 0.0f, 0.0f}, true, false}, {{0.3f, 0.0f, 0.0f}, true, false}, {{0.0f, 0.6f, 0.0f}, true, false},
                        {{0.3f, 0.6f, 0.0f}, true, false}, {{0.3f, 0.6f, 0.0f}, false, false}, {{0.0f, 0.6f, 0.0f}, true, true}};
  std::vector<std::vector<ImDrawVert>> drawn;
  for (const Case &item : cases) {
    NavHarness nav;
    UiFrame frame;
    nav.Frame([&](InteractContext &context) {
      FrameRig rig(1920, 1080);
      const std::vector<ImDrawVert> real = RenderPosed(context, frame, item.visuals, rect, [&](UiFrame &built) {
        return ListRow(built, "row", {"Primary", "", item.selected, item.disabled});
      });
      auto element = NewDrawList();
      PaintStyled(rig, RowLikeStyle(DEFAULT_THEME), rect, {item.visuals, item.disabled, item.selected}, *element);
      const int count = element->VtxBuffer.Size;
      Check(count > 0 && static_cast<int>(real.size()) >= count, "the row drew the box");
      std::vector<ImDrawVert> mine(element->VtxBuffer.begin(), element->VtxBuffer.end());
      for (int i = 0; i < count; ++i) {
        Check(real[i].pos.x == mine[i].pos.x && real[i].pos.y == mine[i].pos.y && real[i].uv.x == mine[i].uv.x && real[i].uv.y == mine[i].uv.y &&
                  real[i].col == mine[i].col,
              "the real row's vertices are the box's");
      }
      drawn.push_back(mine);
    });
  }
  Check(drawn[3].size() > drawn[4].size(), "a selected row draws more than the same row unselected");
}

// The other state rules, each with its own baseline. A hovered box blends its fill by the hover
// weight, so half hover is the halfway colour. Elevation holds the base until the weight reaches
// half and then switches. A disabled state with its own opacity uses that instead of the theme's.
// A scale or opacity that is not finite is ignored rather than spreading.
void TestStateRules() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 200, 100};
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Theme &theme = rig.frame.ActiveTheme();
  const Color base = Rgb(0x112233);
  const Color hover = Rgb(0x884400);
  Style style{base, {}, 0_dp, 8_dp, theme.elevation.level1};
  style.states.hover.fill = hover;
  style.states.hover.elevation = theme.elevation.level3;
  const auto drawn = [&](float weight) {
    auto list = NewDrawList();
    PaintStyled(rig, style, rect, {{weight, 0.0f, 0.0f}, false, false}, *list);
    return list;
  };
  const auto box = [&](Color fill, const Elevation &elevation, float opacity = 1.0f) {
    auto list = NewDrawList();
    DrawBox(*list, rig.frame.Ui(), rect, {fill, 8.0f, {}, 0.0f, elevation}, opacity);
    return list;
  };
  Check(SameDrawing(*drawn(0.4f), *box(Lerp(base, hover, 0.4f), theme.elevation.level1)), "below half the fill blends and the elevation holds");
  Check(SameDrawing(*drawn(0.5f), *box(Lerp(base, hover, 0.5f), theme.elevation.level3)), "at half the elevation switches");
  Check(!SameDrawing(*drawn(0.4f), *drawn(0.5f)), "so the two differ");

  Style disabledStyle{base, {}, 0_dp, 8_dp, {}};
  auto themeDisabled = NewDrawList();
  PaintStyled(rig, disabledStyle, rect, {{}, true, false}, *themeDisabled);
  Check(SameDrawing(*themeDisabled, *box(base, {}, theme.opacity.disabled)), "disabled fades by the theme's opacity");
  disabledStyle.states.disabled.opacity = 0.5f;
  auto ownDisabled = NewDrawList();
  PaintStyled(rig, disabledStyle, rect, {{}, true, false}, *ownDisabled);
  Check(SameDrawing(*ownDisabled, *box(base, {}, 0.5f)), "a disabled state's own opacity replaces it");

  Style nonsense{base, {}, 0_dp, 8_dp, {}};
  nonsense.states.pressed.scale = std::numeric_limits<float>::quiet_NaN();
  nonsense.states.pressed.opacity = std::numeric_limits<float>::infinity();
  auto ignored = NewDrawList();
  PaintStyled(rig, nonsense, rect, {{0.0f, 0.0f, 1.0f}, false, false}, *ignored);
  Check(SameDrawing(*ignored, *box(base, {})), "a non-finite scale and opacity are ignored");
}

// A row selected and then hovered or focused, against the real ListRow element at every output
// size. The selection fill is drawn first and the overlays and ring over it, so a row that drew them
// in another order would match no case. Each case is also compared with the same row unselected,
// which has no selection fill, so the fill is part of what matches.
void TestSelectedRowOverlaysMatchRealRow() {
  const Rect rect{400, 300, 600, 64};
  const ItemVisuals cases[] = {{0.3f, 0.0f, 0.0f}, {0.0f, 0.6f, 0.0f}, {0.3f, 0.6f, 0.0f}};
  for (const Output &output : OUTPUTS) {
    for (const ItemVisuals &visuals : cases) {
      SystemUiTest::HarnessOptions options;
      options.width = output.width;
      options.height = output.height;
      NavHarness nav(options);
      UiFrame frame;
      nav.Frame([&](InteractContext &context) {
        FrameRig rig(output.width, output.height);
        const std::vector<ImDrawVert> real = RenderPosed(context, frame, visuals, rect, [&](UiFrame &built) {
          return ListRow(built, "row", {"Primary", "", true, false});
        });
        auto selected = NewDrawList();
        PaintStyled(rig, RowLikeStyle(DEFAULT_THEME), rect, {visuals, false, true}, *selected);
        auto unselected = NewDrawList();
        PaintStyled(rig, RowLikeStyle(DEFAULT_THEME), rect, {visuals, false, false}, *unselected);
        const int count = selected->VtxBuffer.Size;
        Check(count > unselected->VtxBuffer.Size, "the selected row draws the fill the unselected one lacks");
        Check(static_cast<int>(real.size()) >= count, "the row drew the box");
        for (int i = 0; i < count; ++i) {
          const ImDrawVert &mine = selected->VtxBuffer[i];
          Check(real[i].pos.x == mine.pos.x && real[i].pos.y == mine.pos.y && real[i].uv.x == mine.uv.x && real[i].uv.y == mine.uv.y &&
                    real[i].col == mine.col,
                "the real row's vertices are the selected box's");
        }
      });
    }
  }
}

// The states draw in the order selected, hover, focus. A selected state with an overlay of its own
// draws it between the selection fill and the hover overlay, and the ring last. The hand-built list
// draws them in that order, so a merge that put selected anywhere else is a different list.
void TestSelectedOverlayDrawsFirst() {
  ImGuiScope imgui;
  const Rect rect{400, 300, 600, 64};
  const Color selectedOverlay = Rgb(0xFF8800);
  const ItemVisuals visuals{0.3f, 0.6f, 0.0f};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    rig.Begin();
    const Theme &theme = rig.frame.ActiveTheme();
    Style style = RowLikeStyle(theme);
    style.states.selected.overlay = selectedOverlay;
    auto actual = NewDrawList();
    PaintStyled(rig, style, rect, {visuals, false, true}, *actual);

    const float radius = std::min(rig.scale.Px(ListDefaults(theme).row_radius), std::min(rect.width, rect.height) / 2.0f);
    const std::pair<Color, float> overlays[] = {
        {selectedOverlay, 1.0f}, {theme.colors.state.hover, visuals.hover}, {theme.colors.state.focus, visuals.focus}};
    const auto handBuilt = [&](std::array<int, 3> order) {
      auto list = NewDrawList();
      DrawSelection(*list, rect, radius, theme, rig.scale, 1.0f);
      for (const int index : order) {
        list->AddRectFilled({rect.x, rect.y}, {rect.Right(), rect.Bottom()}, ToImU32(overlays[index].first.MulAlpha(overlays[index].second)), radius);
      }
      DrawFocusRing(*list, rect, radius, theme, rig.scale, visuals.focus);
      return list;
    };
    const auto expected = handBuilt({0, 1, 2});
    Check(SameDrawing(*actual, *expected) && SameCommands(*actual, *expected), "the selected overlay comes before hover and focus");
    Check(!SameDrawing(*actual, *handBuilt({1, 2, 0})), "and the selected overlay last draws differently");
  }
}

// SetInput changes the input and nothing else, and Decorate the effects and nothing else, so
// the two calls can come in either order. The row is selected and hovered, and glows. The
// baselines are the same row with only the glow and with only the input. The first has resting
// states, the second no glow, so a call that undid the other would match one of them.
void TestSetInputAndDecorateAreIndependent() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 400, 64};
  const ItemInput input{{0.4f, 0.0f, 0.0f}, false, true};
  const Decor decor{.glow = GlowEffect{DEFAULT_THEME.glow.focus, EFFECT_COLOR}};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    const auto draw = [&](bool setInput, bool decorate, bool inputFirst) {
      rig.Begin();
      const Element box = Box(rig.frame, {}, RowLikeStyle(rig.frame.ActiveTheme()));
      for (const bool inputStep : {inputFirst, !inputFirst}) {
        if (inputStep && setInput) {
          Check(SetInput(rig.frame, box, input), "the input is accepted");
        }
        if (!inputStep && decorate) {
          Check(!Decorate(rig.frame, box, decor).IsNone(), "the decor is accepted");
        }
      }
      (void)Measure(rig.frame, box, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, box, rect), "the box arranges");
      auto list = NewDrawList();
      Paint(rig.frame, box, *list);
      return list;
    };
    const auto inputThenDecor = draw(true, true, true);
    const auto decorThenInput = draw(true, true, false);
    const auto glowOnly = draw(false, true, true);
    const auto inputOnly = draw(true, false, true);
    Check(SameDrawing(*inputThenDecor, *decorThenInput) && SameCommands(*inputThenDecor, *decorThenInput), "either order draws the same");
    Check(!SameDrawing(*inputThenDecor, *glowOnly), "the input is part of the drawing");
    Check(!SameDrawing(*inputThenDecor, *inputOnly), "and so is the glow");
    Check(inputThenDecor->VtxBuffer.Size > inputOnly->VtxBuffer.Size && inputThenDecor->VtxBuffer.Size > glowOnly->VtxBuffer.Size,
          "which draws more than either part");
  }
}

// SetInput refuses a stale or none handle with its reason, as the other element calls do, and
// makes no record for it.
void TestSetInputRefusals() {
  FrameRig rig;
  rig.Begin();
  const Element old = Box(rig.frame, {}, FullStyle());
  rig.Begin();
  ElementError reason = ElementError::OutOfRange;
  Check(!SetInput(rig.frame, old, {}, &reason) && reason == ElementError::Stale, "a stale handle is refused");
  Check(!SetInput(rig.frame, Element{}, {}, &reason) && reason == ElementError::None, "and so is a none handle");
  Check(FrameInternals::Decors(rig.frame).empty(), "neither made a record");
}

// What a test's Custom callable saw, kept outside the frame and reached through a pointer.
struct CustomSeen {
  int calls = 0;
  Rect painted;
  PaintTransform transform;
  float opacity = -1.0f;
  const ImDrawList *list = nullptr;
  const UiFrame *frame = nullptr;
  // Where the callable ran from. A test reads the address back instead of testing it in the callable,
  // where the compiler may assume a member function's object is aligned and fold the test away.
  uintptr_t address = 0;
  uint32_t magic = 0;
};

constexpr uint32_t CUSTOM_FILL = 0xFFC86432;

// A Custom node that records what it was given and fills the rectangle it was told to paint.
Element RecordingCustom(UiFrame &frame, const ItemLayout &item, CustomSeen *seen) {
  return Custom(frame, item, [seen](const CustomPaint &paint) {
    ++seen->calls;
    seen->painted = paint.painted;
    seen->transform = paint.transform;
    seen->opacity = paint.opacity;
    seen->list = &paint.list;
    seen->frame = &paint.frame;
    paint.list.AddRectFilled({paint.painted.x, paint.painted.y}, {paint.painted.Right(), paint.painted.Bottom()}, CUSTOM_FILL);
  });
}

// A Custom node is called once, with the rectangle it is painted in and the opacity multiplied
// down to it. The tree poses both the parent and the node and fades the whole paint, so the
// arranged rectangle and each opacity on the way differ from the painted rectangle and the
// product. The baseline is the same tree unposed and opaque, which is called once with the
// arranged rectangle and an opacity of 1.
void TestCustomReceivesPaintedRectAndOpacity() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 400, 300};
  constexpr float PAINT_OPACITY = 0.8f;
  constexpr float PARENT_ALPHA = 0.5f;
  constexpr float NODE_ALPHA = 0.75f;
  const ElementPose parentPose{.alpha = PARENT_ALPHA, .offset = {10, -6}, .scale = 0.5f};
  const ElementPose nodePose{.alpha = NODE_ALPHA, .offset = {4, 2}, .scale = 0.8f};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    // Paints the tree, and reports the node's arranged and painted rectangles and the list used.
    const auto run = [&](bool posed, float opacity, CustomSeen &seen, Rect &arranged, Rect &painted) {
      rig.Begin();
      const std::array<Element, 1> kids{RecordingCustom(rig.frame, {.width = 100_dp, .height = 40_dp}, &seen)};
      const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, FullStyle(), kids);
      if (posed) {
        Check(!WithPose(rig.frame, parent, parentPose).IsNone() && !WithPose(rig.frame, kids[0], nodePose).IsNone(), "the poses are accepted");
      }
      (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, parent, rect), "the tree arranges");
      arranged = Arranged(rig.frame, kids[0]);
      painted = Painted(rig.frame, kids[0]);
      auto list = NewDrawList();
      Paint(rig.frame, parent, *list, {.opacity = opacity});
      return list;
    };
    CustomSeen base, posed;
    Rect baseArranged, basePainted, arranged, painted;
    const auto baseList = run(false, 1.0f, base, baseArranged, basePainted);
    Check(base.calls == 1 && base.frame == &rig.frame && base.list == baseList.get(), "the unposed node is called once");
    Check(NearRect(base.painted, baseArranged, TOLERANCE) && Near(base.opacity, 1.0f) && Near(base.transform.scale, 1.0f),
          "unposed and opaque it gets its arranged rectangle and full opacity");

    const auto posedList = run(true, PAINT_OPACITY, posed, arranged, painted);
    Check(posed.calls == 1 && posed.list == posedList.get(), "the posed node is called once, with the list being painted");
    Check(NearRect(arranged, baseArranged, TOLERANCE) && !NearRect(painted, arranged, 1.0f), "the pose moves the painted rectangle and not the layout");
    Check(NearRect(posed.painted, painted, TOLERANCE), "it gets the posed rectangle");
    Check(Near(posed.transform.scale, 0.5f * 0.8f), "and the composed transform");
    constexpr float EXPECTED_OPACITY = PAINT_OPACITY * PARENT_ALPHA * NODE_ALPHA;
    Check(Near(posed.opacity, EXPECTED_OPACITY), "and the opacity multiplied down to it");
    Check(!Near(posed.opacity, PAINT_OPACITY * PARENT_ALPHA) && !Near(posed.opacity, PAINT_OPACITY), "which is neither the incoming opacity nor the parent's");
  }
}

// A Custom node whose parent was refused is never called, and a refusal leaves it free for another
// parent, which then calls it once. A Custom node built before Begin is refused and stores nothing.
void TestCustomRefusalsDoNotCall() {
  ImGuiScope imgui;
  FrameRig rig;
  CustomSeen early;
  const Element refusedEarly = RecordingCustom(rig.frame, {}, &early);
  Check(refusedEarly.IsNone() && early.calls == 0, "a Custom node before Begin is refused");

  rig.Begin();
  CustomSeen seen;
  const Element custom = RecordingCustom(rig.frame, {.width = 50_dp, .height = 20_dp}, &seen);
  Check(!custom.IsNone(), "the node is made");
  const std::array<Element, 2> twice{custom, custom};
  ElementError reason = ElementError::None;
  const Element refused = Box(rig.frame, {}, FullStyle(), twice, &reason);
  Check(refused.IsNone() && reason == ElementError::AlreadyParented, "a parent listing it twice is refused");
  auto list = NewDrawList();
  (void)Measure(rig.frame, refused, {INFINITE_SPACE, INFINITE_SPACE});
  Paint(rig.frame, refused, *list);
  Check(seen.calls == 0 && list->VtxBuffer.Size == 0, "painting the refused parent calls nothing");

  const std::array<Element, 1> once{custom};
  const Element parent = Box(rig.frame, {}, FullStyle(), once);
  Check(!parent.IsNone(), "a later parent takes the node");
  (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, parent, {0, 0, 200, 100}), "the tree arranges");
  Paint(rig.frame, parent, *list);
  Check(seen.calls == 1, "which calls it once");
}

// A Custom node under a clipping ancestor draws inside the clip. The hand-built list pushes the
// clip of the box less its corners and fills the rectangle the node was painted in. The same tree
// without the clip has the same triangles and other commands, so only SameCommands sees the clip.
void TestCustomPaintsInsideAncestorClip() {
  ImGuiScope imgui;
  const Rect rect{100, 50, 300, 200};
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    const auto draw = [&](Overflow overflow, Rect &childRect) {
      rig.Begin();
      CustomSeen seen;
      const std::array<Element, 1> kids{RecordingCustom(rig.frame, {.width = 100_dp, .height = 40_dp}, &seen)};
      const Element parent = Box(rig.frame, {.padding = DpInsets::All(16_dp)}, WithOverflow(overflow), kids);
      (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, parent, rect), "the tree arranges");
      childRect = Arranged(rig.frame, kids[0]);
      auto list = NewDrawList();
      Paint(rig.frame, parent, *list);
      Check(seen.calls == 1, "the node is called once");
      return list;
    };
    Rect childRect, visibleChild;
    const auto clipped = draw(Overflow::Clip, childRect);
    const auto visible = draw(Overflow::Visible, visibleChild);
    auto expected = NewDrawList();
    DrawBox(*expected, rig.frame.Ui(), rect, FullBoxStyle());
    PushClip(*expected, HandClip(rig, rect));
    expected->AddRectFilled({childRect.x, childRect.y}, {childRect.Right(), childRect.Bottom()}, CUSTOM_FILL);
    expected->PopClipRect();
    Check(SameDrawing(*clipped, *expected) && SameCommands(*clipped, *expected), "the node draws inside the clip");
    Check(SameDrawing(*clipped, *visible) && !SameCommands(*clipped, *visible), "which the unclipped tree lacks");
  }
}

// An item that collapses gets no width when its row cannot leave the others their minimums, both
// when the row's width is its own and when the row is only arranged in a rect, and keeps its width
// otherwise. A row measured with no limit collapses nothing, and the same row without the flag is
// the baseline that keeps the item.
void TestCollapsingItem() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  const auto widthOfItem = [&](float rowWidth, bool collapses, bool explicitWidth) {
    rig.Begin();
    const std::array<Element, 2> kids{Box(rig.frame, {.item = {.grow = 1.0f, .min_main = 56_dp}}, FullStyle()),
                                      Box(rig.frame, {.item = {.width = 60_dp, .collapses = collapses}}, FullStyle())};
    Layout layout{};
    if (explicitWidth) {
      layout.item.width = Dp(rowWidth);
    }
    const Element row = Box(rig.frame, layout, {}, kids);
    (void)Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, row, {0, 0, rowWidth, 40}), "the row arranges");
    return Arranged(rig.frame, kids[1]).width;
  };
  Check(Near(widthOfItem(116, true, false), 60.0f), "a row that fits keeps the item");
  Check(Near(widthOfItem(115, true, false), 0.0f), "a row that is a pixel short collapses it");
  Check(Near(widthOfItem(115, true, true), 0.0f), "also when the row has a width of its own");
  Check(Near(widthOfItem(115, false, false), 60.0f), "while the same item that does not collapse is kept");
}

// A row with an item that collapses needs no room for that item. Its min-content is the other
// children's minimums with one gap between them, so a parent can shrink it until the item goes. A
// row with 50 wide children either side of a 60 wide collapsing one and a gap of 10 needs 110 and not
// 180. Laid out in 110 it gives the item no width and puts the outer children one gap apart, and the
// same row without the flag is the baseline that needs the 180.
void TestCollapsingItemAddsNoMinimum() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  const auto build = [&](bool collapses) {
    rig.Begin();
    const std::array<Element, 3> kids{Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, FullStyle()),
                                      Box(rig.frame, {.item = {.width = 60_dp, .height = 20_dp, .collapses = collapses}}, FullStyle()),
                                      Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, FullStyle())};
    const Element row = Box(rig.frame, {.gap = 10_dp}, {}, kids);
    (void)Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
    return std::pair{row, kids};
  };
  const auto [row, kids] = build(true);
  Check(Near(FrameInternals::Find(rig.frame, row)->minContent.x, 110.0f), "a collapsing item adds neither its width nor its gap to the minimum");
  Check(Arrange(rig.frame, row, {0, 0, 110, 20}), "the row arranges in its minimum");
  Check(Near(Arranged(rig.frame, kids[1]).width, 0.0f) && Near(Arranged(rig.frame, kids[2]).x, 60.0f), "the item collapses and leaves one gap");
  const auto [plain, plainKids] = build(false);
  (void)plainKids;
  Check(Near(FrameInternals::Find(rig.frame, plain)->minContent.x, 180.0f), "while the same row without the flag needs all of it");
}

// The measure and the layout agree about an item whose length is zero for another reason than being
// collapsed. A row of 50 wide children either side of one that collapses but is limited to no width,
// with a gap of 10, is 120 wide however it is measured: the limited item is not collapsed, so its gaps
// stay, and the row ends where its last child does.
void TestMeasureAgreesWithLayoutForALimitedItem() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const std::array<Element, 3> kids{Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, FullStyle()),
                                    Box(rig.frame, {.item = {.width = 60_dp, .height = 20_dp, .min_main = 0_dp, .max_main = 0_dp, .collapses = true}}, FullStyle()),
                                    Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, FullStyle())};
  const Element row = Box(rig.frame, {.gap = 10_dp}, {}, kids);
  const Vec2 measured = Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, row, {0, 0, measured.x, 20}), "the row arranges in its measured size");
  Check(Near(measured.x, 120.0f) && Near(Arranged(rig.frame, kids[2]).Right(), measured.x), "and ends where its last child does");
}

// A collapsing item first beside a growing one, in a row 100 wide with a gap of 10, leaves the grower
// the whole row from its start to its end. The row measured in that width is 100 wide, so it does not
// overflow, and the layout puts the grower where the measure found room for it.
void TestCollapsingFirstBesideAGrower() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const std::array<Element, 2> kids{Box(rig.frame, {.item = {.width = 60_dp, .height = 20_dp, .collapses = true}}, FullStyle()),
                                    Box(rig.frame, {.item = {.height = 20_dp, .grow = 1.0f, .min_main = 56_dp}}, FullStyle())};
  const Element row = Box(rig.frame, {.gap = 10_dp}, {}, kids);
  const Vec2 measured = Measure(rig.frame, row, {100, INFINITE_SPACE});
  Check(Arrange(rig.frame, row, {0, 0, 100, 20}), "the row arranges");
  Check(Near(measured.x, 100.0f), "the row measured in 100 is 100 wide and does not overflow");
  Check(Near(Arranged(rig.frame, kids[1]).x, 0.0f) && Near(Arranged(rig.frame, kids[1]).Right(), 100.0f), "and the grower fills it from its start");
  Check(Near(Arranged(rig.frame, kids[0]).width, 0.0f), "while the collapsing item has no width");
}

// Draws `props` as a tree of its own in `frame`, in `limit` pixels, begun at `scale`, into a new list.
std::unique_ptr<ImDrawList> DrawOneText(UiFrame &frame, FrameRig &rig, const ScaleContext &scale, const TextProps &props, float limit) {
  frame.Begin(rig.theme, scale, rig.fonts, rig.store, UiClock::now());
  const Element text = Text(frame, props);
  const Vec2 size = Measure(frame, text, {limit, INFINITE_SPACE});
  Check(Arrange(frame, text, {0, 0, limit, size.y}), "the text arranges");
  auto list = NewDrawList();
  Paint(frame, text, *list);
  return list;
}

constexpr float TEXT_CACHE_ROOM = 300.0f;
// Longer than a chunk of the frame's interned text.
constexpr size_t TEXT_CHUNK_TEST_BYTES = 5000;
constexpr const char *CACHED_TEXT = "Slayer of the Seven Sleeping Giants of the Frozen Northern Wastes";

// A text drawn again exactly as before reuses its layout and makes no new one, and draws what a frame
// that has never seen it draws, which is the baseline, as that frame has nothing to reuse.
void TestKeptTextLayoutIsReused() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  TextProps props{CACHED_TEXT, rig.theme.type.heading, Rgb(0xFFFFFF)};
  props.overflow = TextOverflow::Wrap;
  (void)DrawOneText(rig.frame, rig, rig.scale, props, TEXT_CACHE_ROOM);
  Check(FrameTestHooks::TextLayoutsBuilt(rig.frame) > 0, "a first draw makes its layouts");
  const auto again = DrawOneText(rig.frame, rig, rig.scale, props, TEXT_CACHE_ROOM);
  Check(FrameTestHooks::TextLayoutsBuilt(rig.frame) == 0, "drawing it again makes none");
  Check(again->VtxBuffer.Size > 0, "and it did draw");
  UiFrame fresh;
  const auto baseline = DrawOneText(fresh, rig, rig.scale, props, TEXT_CACHE_ROOM);
  Check(FrameTestHooks::TextLayoutsBuilt(fresh) > 0 && SameDrawing(*again, *baseline), "and draws what a fresh frame does");
}

// A change to anything a layout depends on draws what a frame that never saw the old text draws, and
// each change really does change the drawing, so the comparison can see it. Text of the same length
// shows it is the bytes that count, as the frame interns them at the same address. Colour and alignment
// aren't in the key, so they need no new layout, and still have to show.
void TestTextLayoutFollowsEveryInput() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const auto &type = rig.theme.type;
  TextProps base{CACHED_TEXT, type.heading, Rgb(0xFFFFFF)};
  base.overflow = TextOverflow::Wrap;
  base.max_lines = 2;
  struct Change {
    const char *name;
    TextProps props;
    float limit = TEXT_CACHE_ROOM;
    ScaleContext scale;
    bool new_layout = true;
  };
  const auto with = [&](const char *name, auto &&change) {
    Change made{name, base, TEXT_CACHE_ROOM, rig.scale};
    change(made);
    return made;
  };
  const std::string same_length = std::string(CACHED_TEXT).replace(0, 1, "Z");
  const Change changes[] = {
      with("the text", [&](Change &c) { c.props.text = same_length; }),
      with("the weight", [&](Change &c) { c.props.style.weight = FontWeight::Regular; }),
      with("the size", [&](Change &c) { c.props.style.size = 18.0f; }),
      with("the line height", [&](Change &c) { c.props.style.line = 60.0f; }),
      with("the width", [&](Change &c) { c.limit = 180.0f; }),
      with("the lines allowed", [&](Change &c) { c.props.max_lines = 1; }),
      with("the overflow", [&](Change &c) { c.props.overflow = TextOverflow::Ellipsis; }),
      with("the scale", [&](Change &c) { c.scale = ScaleContext(rig.theme.scale, 3840, 2160); }),
      with("the colour", [&](Change &c) {
        c.props.color = Rgb(0xFF0000);
        c.new_layout = false;
      }),
      with("the alignment", [&](Change &c) {
        c.props.align = TextAlign::End;
        c.new_layout = false;
      }),
  };
  for (const Change &change : changes) {
    const auto settled = DrawOneText(rig.frame, rig, rig.scale, base, TEXT_CACHE_ROOM);
    UiFrame fresh_base;
    Check(SameDrawing(*settled, *DrawOneText(fresh_base, rig, rig.scale, base, TEXT_CACHE_ROOM)), "going back to a kept layout draws it right");
    const auto changed = DrawOneText(rig.frame, rig, change.scale, change.props, change.limit);
    Check((FrameTestHooks::TextLayoutsBuilt(rig.frame) > 0) == change.new_layout, change.name);
    UiFrame fresh;
    const auto baseline = DrawOneText(fresh, rig, change.scale, change.props, change.limit);
    Check(!SameDrawing(*settled, *baseline), change.name);
    Check(SameDrawing(*changed, *baseline), change.name);
  }
  // Shrinking only shows on one line too long for its room.
  TextProps single{CACHED_TEXT, type.heading, Rgb(0xFFFFFF)};
  TextProps shrinking = single;
  shrinking.shrink_to_fit = true;
  const auto plain = DrawOneText(rig.frame, rig, rig.scale, single, TEXT_CACHE_ROOM);
  const auto shrunk = DrawOneText(rig.frame, rig, rig.scale, shrinking, TEXT_CACHE_ROOM);
  UiFrame fresh;
  const auto baseline = DrawOneText(fresh, rig, rig.scale, shrinking, TEXT_CACHE_ROOM);
  Check(!SameDrawing(*plain, *baseline) && SameDrawing(*shrunk, *baseline), "shrinking to fit");
  // The density picks the glyphs measured. The font holds it, and nothing here changes it otherwise.
  (void)DrawOneText(rig.frame, rig, rig.scale, base, TEXT_CACHE_ROOM);
  (void)DrawOneText(rig.frame, rig, rig.scale, base, TEXT_CACHE_ROOM);
  Check(FrameTestHooks::TextLayoutsBuilt(rig.frame) == 0, "the same density reuses the layout");
  const float density = rig.fonts.bold->CurrentRasterizerDensity;
  rig.fonts.bold->CurrentRasterizerDensity = density * 2.0f;
  (void)DrawOneText(rig.frame, rig, rig.scale, base, TEXT_CACHE_ROOM);
  const bool rebuilt = FrameTestHooks::TextLayoutsBuilt(rig.frame) > 0;
  rig.fonts.bold->CurrentRasterizerDensity = density;
  Check(rebuilt, "and another density doesn't");
}

// A text with an explicit width is laid out in that width alone, so Arrange asks for nothing new and the
// layout Measure chose is the one drawn, whichever of its kept entries that is.
void TestExplicitWidthDrawsItsOwnEntry() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  TextProps props{CACHED_TEXT, rig.theme.type.heading, Rgb(0xFFFFFF)};
  props.overflow = TextOverflow::Wrap;
  const auto draw = [&](UiFrame &frame, float width) {
    frame.Begin(rig.theme, rig.scale, rig.fonts, rig.store, UiClock::now());
    const Element node = Text(frame, props, {.width = FromPixels(rig.scale, width)});
    const Vec2 size = Measure(frame, node, {width, INFINITE_SPACE});
    Check(Arrange(frame, node, {0, 0, width, size.y}), "the text arranges");
    auto list = NewDrawList();
    Paint(frame, node, *list);
    return list;
  };
  for (const float width : {300.0f, 180.0f, 300.0f, 180.0f}) {
    const auto kept = draw(rig.frame, width);
    UiFrame fresh;
    Check(kept->VtxBuffer.Size > 0 && SameDrawing(*kept, *draw(fresh, width)), "each width draws its own layout");
  }
  Check(FrameTestHooks::TextLayoutsBuilt(rig.frame) == 0, "from a layout it already has");
}

// Draws two texts in a column in `frame`, in order, and leaves out the second when it is empty.
std::unique_ptr<ImDrawList> DrawTextColumn(UiFrame &frame, FrameRig &rig, std::string_view first, std::string_view second) {
  frame.Begin(rig.theme, rig.scale, rig.fonts, rig.store, UiClock::now());
  const TypeStyle style = rig.theme.type.heading;
  std::array<Element, 2> kids{Text(frame, {first, style, Rgb(0xFFFFFF)})};
  size_t count = 1;
  if (!second.empty()) {
    kids[count++] = Text(frame, {second, style, Rgb(0xFFFFFF)});
  }
  const Element column = Box(frame, {.direction = FlexDirection::Column, .align = Align::Stretch}, {}, std::span<const Element>(kids).first(count));
  const Vec2 size = Measure(frame, column, {TEXT_CACHE_ROOM, INFINITE_SPACE});
  Check(Arrange(frame, column, {0, 0, TEXT_CACHE_ROOM, size.y}), "the column arranges");
  auto list = NewDrawList();
  Paint(frame, column, *list);
  return list;
}

// Layouts are kept by a text's place among the frame's texts, and not by the text, so texts that swap or
// go away must not pick up each other's layouts.
void TestTextLayoutsFollowTheirText() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  struct Step {
    const char *first;
    const char *second;
  };
  for (const Step &step : {Step{"Alpha", "A much longer second line"}, Step{"A much longer second line", "Alpha"}, Step{"Alpha", ""},
                           Step{"A much longer second line", ""}, Step{"Alpha", "A much longer second line"}}) {
    const auto kept = DrawTextColumn(rig.frame, rig, step.first, step.second);
    UiFrame fresh;
    const auto baseline = DrawTextColumn(fresh, rig, step.first, step.second);
    Check(kept->VtxBuffer.Size > 0 && SameDrawing(*kept, *baseline), "each text draws as itself whatever was in its place");
  }
}

// More widths than a text keeps layouts for still each draw right, as old ones are replaced, and a text
// that goes back to one it has kept makes no new layout.
void TestTextLayoutsAreReplacedWhenFull() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  TextProps props{CACHED_TEXT, rig.theme.type.heading, Rgb(0xFFFFFF)};
  props.overflow = TextOverflow::Wrap;
  for (const float room : {120.0f, 160.0f, 200.0f, 240.0f, 280.0f, 320.0f, 360.0f, 120.0f, 200.0f}) {
    const auto kept = DrawOneText(rig.frame, rig, rig.scale, props, room);
    UiFrame fresh;
    const auto baseline = DrawOneText(fresh, rig, rig.scale, props, room);
    Check(SameDrawing(*kept, *baseline), "each width draws right");
  }
  (void)DrawOneText(rig.frame, rig, rig.scale, props, 500.0f);
  (void)DrawOneText(rig.frame, rig, rig.scale, props, 500.0f);
  Check(FrameTestHooks::TextLayoutsBuilt(rig.frame) == 0, "and a width it still has makes no new layout");
}

// A text that was built but never measured has nothing to draw, however much its slot holds from a
// frame before, as what it holds may be another text's.
void TestUnmeasuredTextDrawsNothing() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const TextProps props{CACHED_TEXT, rig.theme.type.heading, Rgb(0xFFFFFF)};
  const auto drawn = DrawOneText(rig.frame, rig, rig.scale, props, TEXT_CACHE_ROOM);
  Check(drawn->VtxBuffer.Size > 0, "measured, it draws");
  rig.Begin();
  const Element unmeasured = Text(rig.frame, props);
  auto list = NewDrawList();
  Paint(rig.frame, unmeasured, *list);
  Check(list->VtxBuffer.Size == 0, "never measured, it draws nothing");
}

// Whether two layouts say the same thing, which is what drawing them reads.
bool SameLayout(const TextLayout &a, const TextLayout &b) {
  if (a.font != b.font || a.font_size != b.font_size || a.line_height != b.line_height || a.size.x != b.size.x ||
      a.size.y != b.size.y || a.lines.size() != b.lines.size()) {
    return false;
  }
  for (size_t index = 0; index < a.lines.size(); ++index) {
    if (a.lines[index].text != b.lines[index].text || a.lines[index].width != b.lines[index].width) {
      return false;
    }
  }
  return true;
}

// LaidOutText gives what LayoutText gives, makes it once however often it is asked, across frames, and
// a different width or text of the same length is a layout of its own. What it has already returned
// stays right while it makes more in the same frame, as a row reads two at once.
void TestLaidOutTextIsKept() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  UiFrame &frame = rig.frame;
  TextProps props{CACHED_TEXT, rig.theme.type.heading, Rgb(0xFFFFFF)};
  props.overflow = TextOverflow::Wrap;
  rig.Begin();
  const UiContext &ui = frame.Ui();
  const TextLayout &first = frame.LaidOutText(props, TEXT_CACHE_ROOM);
  Check(FrameTestHooks::TextLayoutsBuilt(frame) == 1 && !first.lines.empty(), "the first ask makes a layout");
  Check(SameLayout(first, LayoutText(ui, props, TEXT_CACHE_ROOM)), "which is LayoutText's");
  const TextLayout &narrow = frame.LaidOutText(props, 150.0f);
  Check(FrameTestHooks::TextLayoutsBuilt(frame) == 2 && SameLayout(narrow, LayoutText(ui, props, 150.0f)), "another width is a layout of its own");
  Check(!SameLayout(first, narrow) && SameLayout(first, LayoutText(ui, props, TEXT_CACHE_ROOM)), "and the first stays right beside it");
  Check(&frame.LaidOutText(props, TEXT_CACHE_ROOM) == &first && FrameTestHooks::TextLayoutsBuilt(frame) == 2, "asking again makes nothing");
  rig.Begin();
  Check(SameLayout(frame.LaidOutText(props, TEXT_CACHE_ROOM), LayoutText(frame.Ui(), props, TEXT_CACHE_ROOM)) && FrameTestHooks::TextLayoutsBuilt(frame) == 0, "nor does the next frame");
  std::string other(CACHED_TEXT);
  other[0] = 'Z';
  TextProps changed = props;
  changed.text = other;
  Check(SameLayout(frame.LaidOutText(changed, TEXT_CACHE_ROOM), LayoutText(frame.Ui(), changed, TEXT_CACHE_ROOM)) && FrameTestHooks::TextLayoutsBuilt(frame) == 1, "text of the same length is its own");
  TextProps coloured = props;
  coloured.color = Rgb(0xFF0000);
  coloured.align = TextAlign::End;
  (void)frame.LaidOutText(coloured, TEXT_CACHE_ROOM);
  Check(FrameTestHooks::TextLayoutsBuilt(frame) == 1, "colour and alignment need no new layout, as DrawText reads those itself");
}

// Text that hashes to a kept layout's place is checked against it, so it gets its own layout and not
// the other's, whether it is the bytes or the key that differ.
void TestLaidOutTextChecksWhatItFinds() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  TextProps props{CACHED_TEXT, rig.theme.type.heading, Rgb(0xFFFFFF)};
  for (const bool bytes : {true, false}) {
    rig.Begin();
    (void)rig.frame.LaidOutText(props, TEXT_CACHE_ROOM);
    rig.Begin();
    (void)rig.frame.LaidOutText(props, TEXT_CACHE_ROOM);
    Check(FrameTestHooks::TextLayoutsBuilt(rig.frame) == 0, "with nothing in its way the layout is reused");
    FrameTestHooks::CollidePaintedTexts(rig.frame, bytes);
    (void)rig.frame.LaidOutText(props, TEXT_CACHE_ROOM);
    Check(FrameTestHooks::TextLayoutsBuilt(rig.frame) == 1, bytes ? "other bytes are not reused" : "another key is not reused");
  }
}

// A layout nothing has asked for in a while is forgotten, so scrolling through a long list can't keep
// its rows forever, and one asked for every frame is never made again.
void TestLaidOutTextForgetsWhatIsUnused() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const TextProps kept{"Asked for every frame", rig.theme.type.body, Rgb(0xFFFFFF)};
  const TextProps dropped{"Asked for once", rig.theme.type.body, Rgb(0xFFFFFF)};
  rig.Begin();
  (void)rig.frame.LaidOutText(kept, TEXT_CACHE_ROOM);
  (void)rig.frame.LaidOutText(dropped, TEXT_CACHE_ROOM);
  Check(FrameTestHooks::PaintedTexts(rig.frame) == 2, "both are kept at first");
  uint32_t rebuilt = 0;
  constexpr int LONG_ENOUGH = 800;
  for (int at = 0; at < LONG_ENOUGH; ++at) {
    rig.Begin();
    (void)rig.frame.LaidOutText(kept, TEXT_CACHE_ROOM);
    rebuilt += FrameTestHooks::TextLayoutsBuilt(rig.frame);
  }
  Check(FrameTestHooks::PaintedTexts(rig.frame) == 1, "the one nothing asks for is forgotten");
  Check(rebuilt == 0, "and the one still asked for never had to be made again");
}

// A C string has the text's bytes and its end, stays where it is while more is interned, and an empty
// one is still a string. Intern and it share storage without either overwriting the other.
void TestInternCString() {
  FrameRig rig;
  rig.Begin();
  const char *first = rig.frame.InternCString("first");
  const std::string_view viewed = rig.frame.Intern("viewed");
  const char *empty = rig.frame.InternCString({});
  const std::string big(TEXT_CHUNK_TEST_BYTES, 'x');
  const char *large = rig.frame.InternCString(big);
  Check(std::string_view(first) == "first", "it ends where the text does");
  Check(viewed == "viewed", "and doesn't overwrite what is interned after it");
  Check(empty != nullptr && std::string_view(empty).empty(), "an empty text is an empty string");
  Check(std::string_view(large) == big, "and text longer than a chunk is whole");
}

// Counts how many times it is painted.
struct PaintCount {
  int *count;
  void operator()(const CustomPaint &) const { ++*count; }
};

// A box's collapse flags are its own however deep it sits, and survive its children solving theirs. A
// nested row that collapses its tall child is as high as its plain one, and doesn't paint that child,
// and a row whose plain child is itself a box collapses its second child. Rooms wide enough to keep
// them are the baseline, showing it is the collapse that did this.
void TestCollapseInsideANestedBox() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  const ItemLayout tall{.width = 60_dp, .height = 100_dp, .collapses = true};
  const ItemLayout plain{.width = 80_dp, .height = 20_dp, .min_main = 80_dp};
  auto list = NewDrawList();
  struct Found {
    float height;
    int painted;
  };
  const auto nestedCollapse = [&](float room) {
    rig.Begin();
    int painted = 0;
    const std::array<Element, 2> kids{Box(rig.frame, {.item = plain}, FullStyle()), Custom(rig.frame, tall, PaintCount{&painted})};
    // A column's automatic minimum still counts a collapsed child's height, as flex.h says, so the row
    // opts out of it to show its own height.
    const std::array<Element, 1> row{Box(rig.frame, {.item = {.min_main = 0_dp}}, {}, kids)};
    const Element root = Box(rig.frame, {.direction = FlexDirection::Column, .align = Align::Stretch}, {}, row);
    const Vec2 size = Measure(rig.frame, root, {room, INFINITE_SPACE});
    Check(Arrange(rig.frame, root, {0, 0, room, size.y}), "the nested row arranges");
    Paint(rig.frame, root, *list);
    return Found{size.y, painted};
  };
  const Found kept = nestedCollapse(400);
  Check(Near(kept.height, 100.0f) && kept.painted == 1, "with room the tall child counts and paints");
  const Found gone = nestedCollapse(100);
  Check(Near(gone.height, 20.0f), "a nested row that collapses it is only as high as its plain child");
  Check(gone.painted == 0, "and doesn't paint it");
  // A box's first measure happens before its parent has any flags on the stack. Only a child measured
  // again at the length its parent gave it solves on top of them, as this row does when it shrinks.
  const auto shrunkCollapse = [&](float room) {
    rig.Begin();
    int painted = 0;
    const std::array<Element, 2> kids{Box(rig.frame, {.item = plain}, FullStyle()), Custom(rig.frame, tall, PaintCount{&painted})};
    const std::array<Element, 1> row{Box(rig.frame, {}, {}, kids)};
    const Element root = Box(rig.frame, {}, {}, row);
    const Vec2 size = Measure(rig.frame, root, {room, INFINITE_SPACE});
    Check(Arrange(rig.frame, root, {0, 0, room, size.y}), "the shrunk row arranges");
    Paint(rig.frame, root, *list);
    return Found{size.y, painted};
  };
  const Found shrunkKept = shrunkCollapse(400);
  Check(Near(shrunkKept.height, 100.0f) && shrunkKept.painted == 1, "a row with room doesn't shrink its child");
  const Found shrunkGone = shrunkCollapse(100);
  Check(Near(shrunkGone.height, 20.0f) && shrunkGone.painted == 0, "and one that does collapses inside it when measured again");
  const auto outerCollapse = [&](float room) {
    rig.Begin();
    int painted = 0;
    const std::array<Element, 1> inner{Box(rig.frame, {.item = plain}, FullStyle())};
    const std::array<Element, 2> kids{Box(rig.frame, {}, {}, inner), Custom(rig.frame, tall, PaintCount{&painted})};
    const Element root = Box(rig.frame, {}, {}, kids);
    const Vec2 size = Measure(rig.frame, root, {room, INFINITE_SPACE});
    Check(Arrange(rig.frame, root, {0, 0, room, size.y}), "the outer row arranges");
    Paint(rig.frame, root, *list);
    return Found{size.y, painted};
  };
  const Found outerKept = outerCollapse(400);
  Check(Near(outerKept.height, 100.0f) && outerKept.painted == 1, "an outer row with room keeps its second child");
  const Found outerGone = outerCollapse(100);
  Check(Near(outerGone.height, 20.0f) && outerGone.painted == 0, "and collapses it once its first child's box has solved its own");
}

// A collapsed child is absent from its row's measure. A row with a plain child 20 high and a collapsing
// one 100 high, measured in a width that collapses it, is 20 high, while the same row measured wide
// enough to keep it is the 100 of its tall child. A collapsing text with no length of its own, in a
// row whose other child is 10 high, stays one line high when it collapses, and is not measured again
// at no width, where it would wrap a glyph to a line and grow tall.
void TestCollapsedChildIsAbsentFromTheMeasure() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  const auto heightIn = [&](float room) {
    rig.Begin();
    const std::array<Element, 2> kids{Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, FullStyle()),
                                      Box(rig.frame, {.item = {.width = 60_dp, .height = 100_dp, .collapses = true}}, FullStyle())};
    const Element row = Box(rig.frame, {}, {}, kids);
    return Measure(rig.frame, row, {room, INFINITE_SPACE}).y;
  };
  Check(Near(heightIn(100), 20.0f), "a collapsed child adds nothing to the row's height");
  Check(Near(heightIn(200), 100.0f), "while the same row wide enough to keep it is as tall as it");

  rig.Begin();
  TextProps props = Body("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda");
  props.overflow = TextOverflow::Wrap;
  const float oneLine = LayoutText(rig.frame.Ui(), props, INFINITE_SPACE).size.y;
  const std::array<Element, 2> kids{Box(rig.frame, {.item = {.width = 50_dp, .height = 10_dp}}, FullStyle()),
                                    Text(rig.frame, props, {.collapses = true})};
  const Element row = Box(rig.frame, {.align = Align::Start}, {}, kids);
  const Vec2 measured = Measure(rig.frame, row, {100, INFINITE_SPACE});
  Check(Arrange(rig.frame, row, {0, 0, 100, 400}), "the row arranges");
  Check(Near(measured.y, 10.0f), "a collapsed text adds nothing to the height either");
  Check(Near(Arranged(rig.frame, kids[1]).width, 0.0f) && Near(Arranged(rig.frame, kids[1]).height, oneLine),
        "and its rectangle is no wider than nothing and one line high instead of wrapped at no width");
  // What is painted is the plain child alone, so the text drew nothing, not a column of glyphs.
  auto both = NewDrawList();
  Paint(rig.frame, row, *both);
  rig.Begin();
  const std::array<Element, 1> alone{Box(rig.frame, {.item = {.width = 50_dp, .height = 10_dp}}, FullStyle())};
  const Element only = Box(rig.frame, {.align = Align::Start}, {}, alone);
  (void)Measure(rig.frame, only, {100, INFINITE_SPACE});
  Check(Arrange(rig.frame, only, {0, 0, 100, 400}), "the plain child alone arranges");
  auto reference = NewDrawList();
  Paint(rig.frame, only, *reference);
  Check(both->VtxBuffer.Size > 0 && both->VtxBuffer.Size == reference->VtxBuffer.Size, "and the collapsed text paints nothing");
}

// Collapsed nodes are not painted. A row of a plain child and, collapsing, a wrapping text and a box
// with a border and a shadow, arranged in a width that collapses them, draws exactly what the plain
// child draws alone, with no glyphs, no shadow and no border. The same row arranged wide draws more,
// so the nodes do have something to draw. A collapsed text is not laid out again for its empty
// rectangle either, which would cost a layout and could not be seen once it is not painted, so the
// count of text layouts after arranging is zero, where the same text that does not collapse and is
// squeezed into the same width is laid out again.
void TestCollapsedNodesAreNotPainted() {
  ImGuiScope imgui(true);
  TextRig text(imgui, OUTPUTS[1]);
  FrameRig &rig = text.rig;
  TextProps props = Body("alpha beta gamma delta epsilon zeta eta theta");
  props.overflow = TextOverflow::Wrap;
  const Style plain{Rgb(0x112233), {}, 0_dp, 0_dp, {}};
  struct Drawn {
    int vertices;
    uint32_t relayouts;
  };
  const auto draw = [&](float room, bool withOthers, bool othersCollapse, float measuredIn = 0.0f) {
    rig.Begin();
    std::vector<Element> kids{Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, plain)};
    if (withOthers) {
      kids.push_back(Text(rig.frame, props, {.collapses = othersCollapse}));
      kids.push_back(Box(rig.frame, {.item = {.width = 60_dp, .height = 20_dp, .collapses = othersCollapse}}, FullStyle()));
    }
    const Element row = Box(rig.frame, {}, {}, kids);
    (void)Measure(rig.frame, row, {measuredIn > 0.0f ? measuredIn : room, INFINITE_SPACE});
    Check(Arrange(rig.frame, row, {0, 0, room, 40}), "the row arranges");
    auto list = NewDrawList();
    Paint(rig.frame, row, *list);
    return Drawn{list->VtxBuffer.Size, FrameTestHooks::TextRelayouts(rig.frame)};
  };
  const Drawn alone = draw(100, false, false);
  const Drawn collapsed = draw(100, true, true);
  const Drawn wide = draw(1000, true, true);
  // The text that does not collapse, measured wide and arranged narrow, is laid out again, which
  // shows the count is not dead.
  const Drawn squeezed = draw(100, true, false, 1000);
  Check(alone.vertices > 0 && collapsed.vertices == alone.vertices, "collapsed nodes add no vertices");
  Check(wide.vertices > alone.vertices, "while the same row arranged wide draws them");
  Check(collapsed.relayouts == 0, "and a collapsed text is not laid out again");
  Check(squeezed.relayouts > 0, "where a text that does not collapse is, when arranged narrower than measured");
}

// Arranging a node given directly lays it out and paints it, whatever an earlier arrange of its
// parent decided. A row arranged short collapses its second child, which then paints nothing. Arranged
// by itself the child is no longer collapsed, takes its rectangle and paints, since only a parent
// collapses a child.
void TestArrangingACollapsedNodeDirectlyLaysItOut() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Style fill{Rgb(0x112233), {}, 0_dp, 0_dp, {}};
  const std::array<Element, 2> kids{Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, fill),
                                    Box(rig.frame, {.item = {.width = 60_dp, .height = 20_dp, .collapses = true}}, fill)};
  const Element row = Box(rig.frame, {}, {}, kids);
  (void)Measure(rig.frame, row, {100, INFINITE_SPACE});
  Check(Arrange(rig.frame, row, {0, 0, 100, 20}), "the row arranges");
  auto collapsed = NewDrawList();
  Paint(rig.frame, kids[1], *collapsed);
  Check(collapsed->VtxBuffer.Size == 0, "collapsed through its parent, the child paints nothing");
  Check(Arrange(rig.frame, kids[1], {200, 0, 60, 20}), "arranged by itself it lays out");
  auto alone = NewDrawList();
  Paint(rig.frame, kids[1], *alone);
  Check(alone->VtxBuffer.Size > 0 && Near(Arranged(rig.frame, kids[1]).width, 60.0f), "and paints in the rectangle it was given");
}

// A collapsing item still counts across.

// A collapsing item still counts across. A column holding a collapsing item 80 wide and a plain one 30
// wide needs 80 across, though the collapsing item needs no height, and a column of the plain one
// alone is the baseline that needs 30.
void TestCollapsingItemKeepsItsCrossMinimum() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  const auto build = [&](bool withWide) {
    rig.Begin();
    std::vector<Element> kids{Box(rig.frame, {.item = {.width = 30_dp, .height = 20_dp}}, FullStyle())};
    if (withWide) {
      kids.push_back(Box(rig.frame, {.item = {.width = 80_dp, .height = 20_dp, .collapses = true}}, FullStyle()));
    }
    const Element column = Box(rig.frame, {.direction = FlexDirection::Column}, {}, kids);
    (void)Measure(rig.frame, column, {INFINITE_SPACE, INFINITE_SPACE});
    return FrameInternals::Find(rig.frame, column)->minContent;
  };
  const Vec2 with = build(true);
  Check(Near(with.x, 80.0f) && Near(with.y, 20.0f), "a collapsing item keeps its width across, and adds no height");
  Check(Near(build(false).x, 30.0f), "while the plain item alone needs only its own");
}

// A stacked box lays its children over one another. It measures as its largest child plus its
// padding, each child fills the space inside the padding whatever length it asks for, an explicit
// size replaces the measure, and a later child is drawn over an earlier one. A column of the same
// children is the baseline that puts them one below the other.
void TestStackedBoxOverlaysItsChildren() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  const Style first{Rgb(0xAA0000), {}, 0_dp, 0_dp, {}};
  const Style second{Rgb(0x00BB00), {}, 0_dp, 0_dp, {}};
  const auto build = [&](bool stacked) {
    rig.Begin();
    const std::array<Element, 2> kids{Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, first),
                                      Box(rig.frame, {.item = {.width = 80_dp, .height = 10_dp}}, second)};
    Layout layout{.padding = DpInsets::All(5_dp)};
    layout.stacked = stacked;
    layout.direction = FlexDirection::Column;
    return std::pair{Box(rig.frame, layout, {}, kids), kids};
  };
  const auto [stack, kids] = build(true);
  const Vec2 size = Measure(rig.frame, stack, {INFINITE_SPACE, INFINITE_SPACE});
  const Vec2 plain_min = FrameInternals::Find(rig.frame, stack)->minContent;
  Check(Near(size.x, 90.0f) && Near(size.y, 30.0f), "a stack measures as its largest child plus the padding");
  Check(Arrange(rig.frame, stack, {100, 100, 90, 30}), "it arranges");
  for (const Element kid : kids) {
    Check(Near(Arranged(rig.frame, kid).x, 105.0f) && Near(Arranged(rig.frame, kid).y, 105.0f) &&
              Near(Arranged(rig.frame, kid).width, 80.0f) && Near(Arranged(rig.frame, kid).height, 20.0f),
          "each child fills the space inside the padding");
  }
  auto list = NewDrawList();
  Paint(rig.frame, stack, *list);
  Check(list->VtxBuffer.Size == 8 && list->VtxBuffer[0].col == ToImU32(first.fill) && list->VtxBuffer[4].col == ToImU32(second.fill),
        "the later child is drawn over the earlier");
  const auto [column, columnKids] = build(false);
  (void)Measure(rig.frame, column, {INFINITE_SPACE, INFINITE_SPACE});
  Check(Arrange(rig.frame, column, {100, 100, 90, 40}), "the baseline column arranges");
  Check(Arranged(rig.frame, columnKids[1]).y > Arranged(rig.frame, columnKids[0]).y, "while a column puts the second below the first");
  rig.Begin();
  const std::array<Element, 2> sized_kids{Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, first),
                                          Box(rig.frame, {.item = {.width = 80_dp, .height = 10_dp}}, second)};
  Layout sized_layout{.padding = DpInsets::All(5_dp), .item = {.width = 200_dp, .height = 100_dp}};
  sized_layout.stacked = true;
  const Element sized = Box(rig.frame, sized_layout, {}, sized_kids);
  const Vec2 sized_size = Measure(rig.frame, sized, {INFINITE_SPACE, INFINITE_SPACE});
  const Vec2 sized_min = FrameInternals::Find(rig.frame, sized)->minContent;
  Check(Near(sized_size.x, 200.0f) && Near(sized_size.y, 100.0f) && Near(sized_min.x, 200.0f) && Near(sized_min.y, 100.0f),
        "an explicit size replaces both the measure and the min-content of a stack");
  Check(Near(plain_min.x, 90.0f) && Near(plain_min.y, 30.0f), "which the same stack without one keeps as its largest child's");
}

// A spacer takes the lengths it is given, 0 on an axis it is given none, draws nothing, and holds a
// column to its width without moving the label beside it. A column of the label alone is the
// baseline that is narrower.
void TestSpacerTakesRoomAndDrawsNothing() {
  ImGuiScope imgui;
  FrameRig rig(1920, 1080);
  rig.Begin();
  const Element wide = Spacer(rig.frame, {.width = 300_dp});
  const Element tall = Spacer(rig.frame, {.height = 40_dp});
  Check(Near(Measure(rig.frame, wide, {INFINITE_SPACE, INFINITE_SPACE}).x, 300.0f) &&
            Near(Measure(rig.frame, wide, {INFINITE_SPACE, INFINITE_SPACE}).y, 0.0f),
        "a spacer with a width is that wide and 0 high");
  Check(Near(Measure(rig.frame, tall, {INFINITE_SPACE, INFINITE_SPACE}).y, 40.0f), "and one with a height that high");
  auto list = NewDrawList();
  Check(Arrange(rig.frame, tall, {0, 0, 10, 40}), "it arranges");
  Paint(rig.frame, tall, *list);
  Check(list->VtxBuffer.Size == 0, "and draws nothing");
  const auto label = [&] { return Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, FullStyle()); };
  const std::array<Element, 1> alone{label()};
  const Element baseline = Box(rig.frame, {.direction = FlexDirection::Column}, {}, alone);
  const std::array<Element, 2> held{Spacer(rig.frame, {.width = 300_dp}), label()};
  const Element column = Box(rig.frame, {.direction = FlexDirection::Column}, {}, held);
  Check(Near(Measure(rig.frame, baseline, {INFINITE_SPACE, INFINITE_SPACE}).x, 50.0f), "a column of the label alone is as wide as it");
  Check(Near(Measure(rig.frame, column, {INFINITE_SPACE, INFINITE_SPACE}).x, 300.0f), "a spacer holds a column to its width");
}

// A Custom node's size comes from its ItemLayout alone. In a row with a box, one node with both
// lengths, one with a width only and a growing one with neither, each takes the lengths it is
// given, 0 on an axis it is not, and the grower takes what is left. A second row differs only in
// the first node's width, and the nodes after it move by the difference.
void TestCustomSizeFollowsItemLayout() {
  ImGuiScope imgui;
  for (const Output &output : OUTPUTS) {
    FrameRig rig(output.width, output.height);
    using Kids = std::array<Element, 4>;
    const auto lay = [&](Dp firstWidth, Kids &kids, std::array<CustomSeen, 3> &seen) {
      rig.Begin();
      kids = {Box(rig.frame, {.item = {.width = 50_dp, .height = 20_dp}}, FullStyle()),
              RecordingCustom(rig.frame, {.width = firstWidth, .height = 30_dp}, &seen[0]),
              RecordingCustom(rig.frame, {.width = 40_dp}, &seen[1]),
              RecordingCustom(rig.frame, {.grow = 1.0f}, &seen[2])};
      const Element row = Box(rig.frame, {}, FullStyle(), kids);
      (void)Measure(rig.frame, row, {INFINITE_SPACE, INFINITE_SPACE});
      Check(Arrange(rig.frame, row, {0, 0, ToPixels(rig.scale, 400_dp), ToPixels(rig.scale, 100_dp)}), "the row arranges");
    };
    const auto at = [&](const Kids &kids, int index) { return Arranged(rig.frame, kids[index]); };
    Kids kids, wider;
    std::array<CustomSeen, 3> seen, seenWider;
    lay(80_dp, kids, seen);
    const float boxWidth = ToPixels(rig.scale, 50_dp);
    const float firstWidth = ToPixels(rig.scale, 80_dp);
    const float secondWidth = ToPixels(rig.scale, 40_dp);
    const float rowWidth = ToPixels(rig.scale, 400_dp);
    Check(Near(at(kids, 1).x, boxWidth) && Near(at(kids, 1).width, firstWidth) && Near(at(kids, 1).height, ToPixels(rig.scale, 30_dp)),
          "a node with both lengths takes them");
    Check(Near(at(kids, 2).x, boxWidth + firstWidth) && Near(at(kids, 2).width, secondWidth) && Near(at(kids, 2).height, 0.0f),
          "a node with a width only is that wide and 0 high");
    const float used = boxWidth + firstWidth + secondWidth;
    Check(Near(at(kids, 3).x, used) && Near(at(kids, 3).width, rowWidth - used) && Near(at(kids, 3).height, 0.0f),
          "a grower with no lengths takes the rest and no height");
    const Vec2 alone = Measure(rig.frame, kids[2], {INFINITE_SPACE, INFINITE_SPACE});
    Check(Near(alone.x, secondWidth) && Near(alone.y, 0.0f), "Measure gives the lengths set and 0 for the rest");

    lay(100_dp, wider, seenWider);
    const float shift = ToPixels(rig.scale, 100_dp) - firstWidth;
    Check(shift > 0.0f && Near(at(wider, 2).x, boxWidth + firstWidth + shift), "widening the first node moves the next one by the difference");
    Check(Near(at(wider, 3).width, rowWidth - used - shift), "and takes it from the grower");
  }
}

// A callable survives the storage growing, and its copy is aligned. One probe goes first and
// another after 63 one byte callables, so the first is stored and then moved when the storage
// reallocates, and the second lands at an offset that is not aligned until it is rounded up. Both
// run at paint time and report the magic number they were built with and where they ran from.
// The second frame is built the same way and the reserved bytes do not move.
void TestCustomStorageIsAlignedAndSteady() {
  ImGuiScope imgui;
  constexpr uint32_t FIRST_MAGIC = 0xA1B2C3D4;
  constexpr uint32_t LAST_MAGIC = 0x0F1E2D3C;
  struct Probe {
    alignas(16) float padding[4];
    CustomSeen *seen;
    uint32_t magic;
    void operator()(const CustomPaint &) const {
      ++seen->calls;
      seen->magic = magic;
      seen->address = reinterpret_cast<uintptr_t>(this);
    }
  };
  static_assert(alignof(Probe) == 16 && std::is_trivially_copyable_v<Probe>);
  FrameRig rig;
  size_t reserved = 0;
  for (int frame = 0; frame < 2; ++frame) {
    rig.Begin();
    CustomSeen first, last;
    constexpr int FILLER_COUNT = 63;
    std::vector<Element> kids;
    kids.push_back(Custom(rig.frame, {.width = 10_dp, .height = 10_dp}, Probe{{}, &first, FIRST_MAGIC}));
    for (int i = 0; i < FILLER_COUNT; ++i) {
      kids.push_back(Custom(rig.frame, {.width = 10_dp, .height = 10_dp}, [](const CustomPaint &) {}));
    }
    kids.push_back(Custom(rig.frame, {.width = 10_dp, .height = 10_dp}, Probe{{}, &last, LAST_MAGIC}));
    const Element parent = Box(rig.frame, {}, FullStyle(), kids);
    (void)Measure(rig.frame, parent, {INFINITE_SPACE, INFINITE_SPACE});
    Check(Arrange(rig.frame, parent, {0, 0, 200, 100}), "the tree arranges");
    auto list = NewDrawList();
    Paint(rig.frame, parent, *list);
    Check(first.calls == 1 && first.magic == FIRST_MAGIC, "the first callable survives the storage moving");
    Check(last.calls == 1 && last.magic == LAST_MAGIC, "and so does the last");
    Check(first.address % alignof(Probe) == 0 && last.address % alignof(Probe) == 0, "both run from aligned copies");
    if (frame == 0) {
      reserved = rig.frame.ReservedBytes();
    } else {
      Check(rig.frame.ReservedBytes() == reserved, "a steady frame reserves nothing more");
    }
  }
}

} // namespace

int main() {
  TestEffectsFollowPosedText();
  TestImageEffectsUsePlacedRect();
  TestInvalidGlowRefused();
  TestButtonLikeStatesMatchReplica();
  TestButtonStatesMatchRealButton();
  TestRowStatesMatchRealRow();
  TestStateRules();
  TestSelectedRowOverlaysMatchRealRow();
  TestSelectedOverlayDrawsFirst();
  TestSetInputAndDecorateAreIndependent();
  TestSetInputRefusals();
  TestCustomReceivesPaintedRectAndOpacity();
  TestCustomRefusalsDoNotCall();
  TestCustomPaintsInsideAncestorClip();
  TestSpacerTakesRoomAndDrawsNothing();
  TestCollapsingItem();
  TestCollapsingItemAddsNoMinimum();
  TestMeasureAgreesWithLayoutForALimitedItem();
  TestCollapsingItemKeepsItsCrossMinimum();
  TestCollapsedChildIsAbsentFromTheMeasure();
  TestCollapseInsideANestedBox();
  TestKeptTextLayoutIsReused();
  TestTextLayoutFollowsEveryInput();
  TestExplicitWidthDrawsItsOwnEntry();
  TestTextLayoutsFollowTheirText();
  TestTextLayoutsAreReplacedWhenFull();
  TestUnmeasuredTextDrawsNothing();
  TestLaidOutTextIsKept();
  TestLaidOutTextChecksWhatItFinds();
  TestLaidOutTextForgetsWhatIsUnused();
  TestInternCString();
  TestCollapsedNodesAreNotPainted();
  TestArrangingACollapsedNodeDirectlyLaysItOut();
  TestCollapsingFirstBesideAGrower();
  TestStackedBoxOverlaysItsChildren();
  TestCustomSizeFollowsItemLayout();
  TestCustomStorageIsAlignedAndSteady();
  TestEachEffectMatchesItsFunction();
  TestEffectsDrawInOrder();
  TestEffectsMultiplyAlpha();
  TestDecorateKeepsPose();
  TestInvalidDecorRefused();
  TestShortBoxClipClampsRadius();
  TestGlowInClippingCardMatchesToast();
  TestUnclippedGlowEscapesNearestClip();
  TestClipKeepsChildrenInside();
  TestNestedClipsIntersect();
  TestClipFollowsPose();
  TestClipLeavesOwnShadowAlone();
  TestBracedPoseCompiles();
  TestAlphaIsClamped();
  TestDefaultElementPoseChangesNothing();
  TestTimelinePoseConverts();
  TestInvalidPoseRefused();
  TestComposedUnitScaleStillSnaps();
  TestAlphaAndPosesReachDescendants();
  TestThreeLevelPosesCompose();
  TestPosedBoxMatchesPosed();
  TestScaledPoseKeepsFractions();
  TestParentPoseReachesChild();
  TestAlphaMultipliesIntoOpacity();
  TestPoseLeavesLayoutAlone();
  TestPosedTextKeepsGlyphSize();
  TestPinnedTextKeepsWrappedHeight();
  TestToPixelsRounds();
  TestFromPixelsRoundTrips();
  TestBorderPixels();
  TestInsets();
  TestArithmetic();
  TestBeginKeepsCapacity();
  TestStaleHandle();
  TestOutOfRangeHandle();
  TestChildrenKeepOrder();
  TestInternSurvivesGrowth();
  TestDefaultElementIsNone();
  TestSerialWrapSkipsNone();
  TestOneParentPerNode();
  TestInvalidChildRefused();
  TestBeforeBegin();
  TestBoxMatchesDrawBox();
  TestPaintPassesOpacity();
  TestPaintReachesChildren();
  TestPaddingInsetsChild();
  TestRowMatchesHandBuiltFlex();
  TestArrangeHonoursItemOverrides();
  TestLayoutScratchKeepsCapacity();
  TestGrowersWithNaturalSizes();
  TestOverflowShrinks();
  TestFixedWidthChildDoesNotShrink();
  TestArrangeUsesNaturalLengths();
  TestNestedMeasuresDoNotClobberLengths();
  TestChildLimitsReachGrandchildren();
  TestExplicitCrossIsNotStretched();
  TestNestedSolvesDoNotClobber();
  TestArrangeNeedsMeasure();
  TestTextMatchesLayoutAndDraw();
  TestTextWrapsToMaxLines();
  TestTextRelayoutsAndCentres();
  TestImageMatchesDrawImage();
  TestImageFollowsAspect();
  TestMixedRowMatchesSolver();
  TestTextKeepsLongestWord();
  TestTruncatingTextShrinksBelowWord();
  TestEllipsisTextKeepsItsMark();
  TestBoxKeepsItsContent();
  TestImageKeepsItsSize();
  TestLeavesFade();
  TestNaturalMeasureIsMemoised();
  TestSecondMeasureMatchesFresh();
  TestMinimumHonoursMaximumInParent();
  TestMinContentThroughNestedBoxes();
  TestImageWithEmptyTexture();
  TestShortTextIsNotRelaidOut();
  TestExplicitWidthTextIsLaidOutOnce();
  TestMeasuredAtIsForRoots();
  TestColumnWrapsTextToItsWidth();
  TestWrappedColumnReportsItsHeight();
  TestWrappedColumnMeasuredTwice();
  TestAlternatingTreeStaysLinear();
  TestGrowShareUsesWrappedHeight();
  TestSubtreeMatchesFreshAfterLimitWidens();
  TestOverflowingTreeStaysLinear();
  TestShrinkingOverflowIsBoundedByDepth();
  TestArrangeRecurses();
  TestMeasureIntrinsic();
  TestMeasureClampsMainAxis();
  TestMeasureSharesSpaceWithGrowers();
  TestBoxReportsRefusal();
  std::printf("System UI element tests passed\n");
  return 0;
}
