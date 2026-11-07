#include "SystemUiDrawingKit.h"
#include "SystemUiToastKit.h"
#include "graphics/presentation/system/animation/posed.h"
#include "graphics/presentation/system/components/box.h"
#include "graphics/presentation/system/components/card.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/trophyCup.h"
#include "graphics/presentation/system/components/image.h"
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/imgui/fonts.h"
#include "graphics/presentation/system/imgui/imageStore.h"

#include "imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <set>
#include <tuple>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics::SystemUi;
using SystemUiTest::ContainsDrawing;
using SystemUiTest::NearDrawing;
using SystemUiTest::NearGeometry;
using SystemUiTest::SameCommands;

constexpr const char *REGULAR_FONT = KYTY_TEST_FONT_REGULAR;
constexpr const char *BOLD_FONT = KYTY_TEST_FONT_BOLD;
constexpr float TOLERANCE = 0.01f;
// ImGui pulls a stroke's bottom-right corner in by 0.49 rather than 0.5.
constexpr float STROKE_SLACK = 0.02f;
constexpr Rect PANEL{100, 100, 200, 100};

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiComponentsTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two values agree to within `slack`.
bool Near(float actual, float expected, float slack = TOLERANCE) { return std::fabs(actual - expected) <= slack; }

// True when every edge of two rectangles agrees to within `slack`.
bool Near(const Rect &actual, const Rect &expected, float slack = TOLERANCE) {
  return Near(actual.x, expected.x, slack) && Near(actual.y, expected.y, slack) &&
         Near(actual.width, expected.width, slack) && Near(actual.height, expected.height, slack);
}

// One ImGui frame with no renderer and the theme's fonts loaded. Anti-aliasing is off so shapes
// have exact edges.
class Frame {
public:
  Frame() {
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.DisplaySize = {3840, 2160};
    io.DeltaTime = 1.0f / 60.0f;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.IniFilename = nullptr;
    ImGui::GetStyle().AntiAliasedFill = false;
    ImGui::GetStyle().AntiAliasedLines = false;
    fonts = LoadUiFonts(*io.Fonts, REGULAR_FONT, BOLD_FONT);
    ImGui::NewFrame();
  }
  ~Frame() {
    ImGui::EndFrame();
    ImGui::DestroyContext();
  }
  Frame(const Frame &) = delete;
  Frame &operator=(const Frame &) = delete;

  ImDrawList &List() { return *ImGui::GetForegroundDrawList(); }
  int Mark() { return List().VtxBuffer.Size; }
  std::vector<ImDrawVert> Since(int mark) {
    const auto &buffer = List().VtxBuffer;
    return {buffer.begin() + mark, buffer.end()};
  }
  UiContext Context(float width = 1920, float height = 1080) const {
    return {DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, width, height), fonts};
  }

  UiFonts fonts;
};

// The smallest rectangle around `vertices`, optionally only those of colour `only`.
Rect Bounds(const std::vector<ImDrawVert> &vertices, std::optional<ImU32> only = std::nullopt) {
  float left = std::numeric_limits<float>::max(), top = left, right = -left, bottom = -left;
  for (const auto &vertex : vertices) {
    if (only.has_value() && vertex.col != *only) {
      continue;
    }
    left = std::min(left, vertex.pos.x);
    top = std::min(top, vertex.pos.y);
    right = std::max(right, vertex.pos.x);
    bottom = std::max(bottom, vertex.pos.y);
  }
  return {left, top, right - left, bottom - top};
}

// True when some vertex has colour `color`.
bool HasColor(const std::vector<ImDrawVert> &vertices, Color color) {
  return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &v) { return v.col == ToImU32(color); });
}

// A `width` by `height` PNG, encoded with stb so the store decodes a real file.
std::vector<std::byte> Png(int width, int height) {
  std::vector<unsigned char> pixels(static_cast<size_t>(width * height * 4), 200);
  int length = 0;
  unsigned char *encoded = stbi_write_png_to_mem(pixels.data(), width * 4, width, height, 4, &length);
  std::vector<std::byte> bytes(reinterpret_cast<std::byte *>(encoded), reinterpret_cast<std::byte *>(encoded) + length);
  STBIW_FREE(encoded);
  return bytes;
}

// A draw list of its own that shares the context's draw list data, so two lists can be compared.
std::unique_ptr<ImDrawList> NewDrawList() {
  auto list = std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData());
  list->_ResetForNewFrame();
  list->PushClipRectFullScreen();
  return list;
}

// The vertices of a list, as the Frame helpers return them.
std::vector<ImDrawVert> VerticesOf(const ImDrawList &list) { return {list.VtxBuffer.begin(), list.VtxBuffer.end()}; }

struct Output {
  float width;
  float height;
  // Where a card's child lands in PANEL, for the outputs that TestCard checks.
  Rect cardContent = {};
};
// The outputs the components are held to. 720p rounds each length, 1080p is the reference and 4K doubles it.
constexpr std::array<Output, 3> OUTPUTS{{{1280, 720}, {1920, 1080}, {3840, 2160}}};

// A UiFrame at a given output size, with the objects it borrows.
struct ElementRig {
  ElementRig(const Frame &imgui, const Output &output)
      : scale(DEFAULT_THEME.scale, output.width, output.height), fonts(imgui.fonts) {}
  void Begin() { ui.Begin(theme, scale, fonts, store, UiClock::now()); }

  // The theme the frame uses, the default unless a test sets another before it begins.
  Theme theme = DEFAULT_THEME;
  ScaleContext scale;
  UiFonts fonts;
  InteractionStore store;
  UiFrame ui;
};

// Measures `root`, places it at `rect`, and paints it at `opacity` into a new list.
std::unique_ptr<ImDrawList> DrawElement(ElementRig &rig, Element root, const Rect &rect, float opacity = 1.0f) {
  Check(!root.IsNone(), "the element was built");
  Measure(rig.ui, root, {rect.width, rect.height});
  Check(Arrange(rig.ui, root, rect), "the element was arranged");
  auto list = NewDrawList();
  Paint(rig.ui, root, *list, PaintState{.opacity = opacity});
  return list;
}

// A box fills its rectangle, its border lies just inside its edge, and its shadow spreads outside.
// Without a border width or elevation, neither is drawn.
void TestBox() {
  Frame frame;
  const auto context = frame.Context();
  const BoxStyle style{Rgb(0x112233), 12, Rgb(0xFFFFFF), 2, DEFAULT_THEME.elevation.level1};
  int mark = frame.Mark();
  DrawBox(frame.List(), context, PANEL, style);
  auto vertices = frame.Since(mark);
  Check(Near(Bounds(vertices, ToImU32(style.fill)), PANEL), "the fill covers the box exactly");
  Check(Near(Bounds(vertices, ToImU32(style.border)), PANEL, STROKE_SLACK), "the border ends at the box's edge");
  Check(Bounds(vertices).x < PANEL.x - 1.0f, "the shadow spreads outside the box");
  mark = frame.Mark();
  DrawBox(frame.List(), context, PANEL, {Rgb(0x112233), 12, Rgb(0xFFFFFF), 0, {}});
  vertices = frame.Since(mark);
  Check(!HasColor(vertices, style.border) && Near(Bounds(vertices), PANEL), "no border width and no elevation draw neither");
}

// Opacity fades the fill and border of a box together.
void TestBoxFades() {
  Frame frame;
  const auto context = frame.Context();
  const BoxStyle style{Rgb(0x112233), 12, Rgb(0xFFFFFF), 2, {}};
  const int mark = frame.Mark();
  DrawBox(frame.List(), context, PANEL, style, 0.5f);
  const auto vertices = frame.Since(mark);
  Check(HasColor(vertices, style.fill.MulAlpha(0.5f)) && HasColor(vertices, style.border.MulAlpha(0.5f)),
        "fill and border fade together");
}

constexpr Color CHILD_FILL = Rgb(0x336699);
constexpr float CHILD_RADIUS = 4.0f;

// A card and the child inside it, which fills the card's content space. The child is a none handle
// for an empty card.
struct BuiltCard {
  Element card;
  Element child;
};
BuiltCard BuildCard(ElementRig &rig, bool withChild) {
  BuiltCard built;
  if (!withChild) {
    built.card = Card(rig.ui, {.align = Align::Stretch});
    return built;
  }
  built.child = Box(rig.ui, {.item = {.grow = 1.0f}}, {.fill = CHILD_FILL, .radius = Dp(CHILD_RADIUS)});
  const std::array<Element, 1> kids{built.child};
  built.card = Card(rig.ui, {.align = Align::Stretch}, {}, kids);
  return built;
}

// A card is the theme's raised surface with a subtle border, and its content sits inside its 16
// pixel padding.
void TestCard() {
  Frame frame;
  const std::array<Output, 2> outputs{Output{1920, 1080, {116, 116, 168, 68}}, Output{3840, 2160, {132, 132, 136, 36}}};
  for (const Output &output : outputs) {
    ElementRig rig(frame, output);
    rig.Begin();
    const BuiltCard built = BuildCard(rig, true);
    const auto list = DrawElement(rig, built.card, PANEL);
    const auto vertices = VerticesOf(*list);
    Check(HasColor(vertices, DEFAULT_THEME.colors.surface.raised), "a card is filled with the raised surface");
    Check(HasColor(vertices, DEFAULT_THEME.colors.border.subtle), "a card has the subtle border");
    Check(Near(Arranged(rig.ui, built.child), output.cardContent), "the content sits inside the card's padding, which doubles at 4K");
  }
}

// The width of the part of the bar drawn in `fill`, or 0 when nothing is.
float FillWidth(ElementRig &rig, const Rect &bar, float value, Color fill) {
  rig.Begin();
  const auto list = DrawElement(rig, ProgressBar(rig.ui, {.value = value, .fill = fill}), bar);
  const auto vertices = VerticesOf(*list);
  return HasColor(vertices, fill) ? Bounds(vertices, ToImU32(fill)).width : 0.0f;
}

// The fill covers the given share of the track, at least a round dot once started, and nothing
// before starting. Values outside 0..1 are clamped, and NaN counts as not started.
void TestProgressBar() {
  Frame frame;
  ElementRig rig(frame, Output{1920, 1080});
  const Rect bar{100, 100, 200, 8};
  const Color fill = DEFAULT_THEME.colors.accent.line;
  Check(Near(FillWidth(rig, bar, 0.5f, fill), 100.0f), "half the bar is filled at 0.5");
  Check(Near(FillWidth(rig, bar, 0.01f, fill), 8.0f), "a little progress still shows a round dot");
  Check(FillWidth(rig, bar, 0.0f, fill) == 0.0f && FillWidth(rig, bar, std::numeric_limits<float>::quiet_NaN(), fill) == 0.0f,
        "nothing before starting");
  Check(Near(FillWidth(rig, bar, 1.5f, fill), 200.0f), "progress past the end fills the bar exactly");
  // On a bar narrower than it is tall, the round dot would be wider than the bar, so the bar's
  // own width caps it.
  Check(FillWidth(rig, Rect{100, 100, 6, 8}, 0.5f, fill) <= 6.0f + TOLERANCE, "the fill never passes the bar");
  rig.Begin();
  const auto track = DrawElement(rig, ProgressBar(rig.ui, {.value = 0.0f, .fill = fill}), bar);
  Check(Near(Bounds(VerticesOf(*track), ToImU32(DEFAULT_THEME.colors.surface.inset)), bar), "the track fills the bar");
  // A thin border around the whole track, as the inset surface alone is faint against a card. It's a
  // colour only this draws here, and it's drawn whatever the progress, under the fill where they meet.
  constexpr float BORDER_SLACK = 1.0f;
  const ImU32 border = ToImU32(DEFAULT_THEME.colors.border.standard);
  Check(Near(Bounds(VerticesOf(*track), border), bar, BORDER_SLACK), "with a thin border around it");
  rig.Begin();
  const auto partway = DrawElement(rig, ProgressBar(rig.ui, {.value = 0.5f, .fill = fill}), bar);
  Check(Near(Bounds(VerticesOf(*partway), border), bar, BORDER_SLACK), "and it's still drawn with a fill");
}

// A height no theme token has, so a bar given it is not mistaken for the default.
constexpr float EXPLICIT_BAR_HEIGHT = 20;

// A bar given no height is as thick as the theme's progress height, which scales with the output.
void TestProgressBarDefaultHeight() {
  Frame frame;
  for (const Output &output : {OUTPUTS[1], OUTPUTS[2]}) {
    ElementRig rig(frame, output);
    rig.Begin();
    const Element bar = ProgressBar(rig.ui, {.value = 0.5f, .fill = Rgb(0x33CC66)});
    const float expected = rig.scale.Px(ProgressDefaults(DEFAULT_THEME).height);
    Check(Near(Measure(rig.ui, bar, {400, 1000}).y, expected), "a bar with no height takes the theme's progress height");
    Check(Near(expected, ProgressDefaults(DEFAULT_THEME).height * rig.scale.Scale()), "which scales with the output");
  }
  ElementRig rig(frame, OUTPUTS[1]);
  rig.Begin();
  const Element tall = ProgressBar(rig.ui, {.value = 0.5f, .fill = Rgb(0x33CC66)}, {.height = Dp(EXPLICIT_BAR_HEIGHT)});
  Check(Near(Measure(rig.ui, tall, {400, 1000}).y, EXPLICIT_BAR_HEIGHT), "and an explicit height wins");
}

// The fill ends on a whole pixel, as a layout rectangle does. Half of 201 is 100.5, so an unsnapped
// edge would sit on a half pixel, which a within-a-pixel comparison can't tell from the snapped
// one. This checks the edge itself.
void TestProgressFillEdgeSnaps() {
  Frame frame;
  ElementRig rig(frame, Output{1920, 1080});
  const Rect bar{100, 100, 201, 8};
  const Color fill = Rgb(0x33CC66);
  rig.Begin();
  const auto actual = DrawElement(rig, ProgressBar(rig.ui, {.value = 0.5f, .fill = fill}), bar);
  const Rect filled = Bounds(VerticesOf(*actual), ToImU32(fill));
  const float right = filled.x + filled.width;
  Check(right == std::floor(right) && Near(right, 201.0f), "the fill's right edge is a whole pixel");
}

// The store decodes real images, refuses data it can't read, oversized images and images past
// its capacity, and keeps a released image's slot until the renderer has destroyed it.
void TestImageStore() {
  Frame frame;
  ImageStore store(2);
  const auto icon = store.Add(Png(64, 32), 512);
  Check(icon.has_value() && store.Texture(*icon) != nullptr, "a PNG is decoded and registered");
  Check(store.Size(*icon).x == 64.0f && store.Size(*icon).y == 32.0f, "its size is the image's");
  Check(!store.Add(Png(600, 10), 512).has_value(), "an image larger than the limit is refused");
  Check(!store.Add(Png(4100, 1), 100000).has_value(), "no image may be wider than 4096, whatever the limit asked");
  const std::vector<std::byte> garbage(100, std::byte{7});
  Check(!store.Add(garbage, 512).has_value(), "unreadable data is refused");
  const auto second = store.Add(Png(8, 8), 512);
  Check(second.has_value() && !store.Add(Png(8, 8), 512).has_value(), "the store refuses images past its capacity");
  auto *texture = store.Texture(*icon);
  store.Release(*icon);
  Check(store.Texture(*icon) == nullptr && texture->WantDestroyNextFrame, "a released image is queued for destruction");
  Check(store.LiveCount() == 1 && store.RetiringCount() == 1, "it waits for the renderer");
  store.Collect();
  Check(store.RetiringCount() == 1 && !store.Add(Png(8, 8), 512).has_value(), "its slot stays taken until destroyed");
  // A real frame. Rendering copies the registered textures into the renderer's list, and the next
  // frame destroys the released one, which was never uploaded, without needing a renderer.
  ImGui::Render();
  const auto &listed = ImGui::GetPlatformIO().Textures;
  Check(listed.contains(texture), "rendering lists the released texture for the renderer");
  ImGui::NewFrame();
  Check(texture->Status == ImTextureStatus_Destroyed, "the next frame destroys the released texture");
  store.Collect();
  Check(!listed.contains(texture), "collecting takes the texture off the renderer's list");
  Check(store.RetiringCount() == 0 && store.Add(Png(8, 8), 512).has_value(), "a destroyed image frees its slot");
}

// The texture handle each image vertex carries, for checking which image was drawn.
ImTextureID Texture(const ImDrawList &list) { return list.CmdBuffer.back().TexRef.GetTexID(); }

// An image fills its rectangle, or keeps its shape inside it, or crops to cover it. A wide image
// contained in a square is centred with space above and below, and covering shows its middle.
void TestImage() {
  Frame frame;
  const auto context = frame.Context();
  ImageStore store(4);
  const auto id = *store.Add(Png(200, 100), 512);
  const Rect square{100, 100, 100, 100};
  int mark = frame.Mark();
  DrawImage(frame.List(), context, store.Texture(id), store.Size(id), square, 0, ImageFit::Fill);
  Check(Near(Bounds(frame.Since(mark)), square), "a filled image covers its rectangle");
  mark = frame.Mark();
  DrawImage(frame.List(), context, store.Texture(id), store.Size(id), square, 0, ImageFit::Contain);
  Check(Near(Bounds(frame.Since(mark)), {100, 125, 100, 50}), "a contained image keeps its shape, centred");
  mark = frame.Mark();
  DrawImage(frame.List(), context, store.Texture(id), store.Size(id), square, 0, ImageFit::Cover);
  auto vertices = frame.Since(mark);
  float uv_left = 1, uv_right = 0;
  for (const auto &vertex : vertices) {
    uv_left = std::min(uv_left, vertex.uv.x);
    uv_right = std::max(uv_right, vertex.uv.x);
  }
  Check(Near(Bounds(vertices), square) && Near(uv_left, 0.25f) && Near(uv_right, 0.75f),
        "a covering image fills the rectangle with the middle of itself");
  mark = frame.Mark();
  DrawImage(frame.List(), context, store.Texture(id), store.Size(id), square, 0, ImageFit::Fill, 0.5f);
  vertices = frame.Since(mark);
  Check(HasColor(vertices, Rgb(0xFFFFFF).MulAlpha(0.5f)), "opacity fades the image");
  mark = frame.Mark();
  DrawImage(frame.List(), context, nullptr, {}, square, 0, ImageFit::Fill);
  Check(frame.Since(mark).empty(), "no texture draws nothing");
}

// Posing moves an element by its offset in output pixels and scales it about its centre, on
// whole pixels. At 4K a 32 pixel offset moves it 64.
void TestPosed() {
  const ScaleContext uhd(DEFAULT_THEME.scale, 3840, 2160);
  const Rect rect{100, 100, 200, 100};
  Pose pose;
  pose.offset = {0, -32};
  Check(Near(Posed(rect, pose, uhd), {100, 36, 200, 100}), "the offset scales with the output");
  pose = {};
  pose.scale = 0.9f;
  Check(Near(Posed(rect, pose, uhd), {110, 105, 180, 90}), "scaling keeps the centre");
}

// The number of draw commands clipped to `clip`, for telling whether a draw added one.
int ClipCount(const ImDrawList &list, const Rect &clip) {
  int count = 0;
  for (const auto &command : list.CmdBuffer) {
    count += Near(command.ClipRect.x, clip.x) && Near(command.ClipRect.y, clip.y) && Near(command.ClipRect.z, clip.Right()) &&
                     Near(command.ClipRect.w, clip.Bottom())
                 ? 1
                 : 0;
  }
  return count;
}

constexpr uint64_t HASH_SEED = 14695981039346656037ull;
constexpr uint64_t HASH_PRIME = 1099511628211ull;

// A fingerprint of a drawing's vertices and clips, so a test can tell that two drawings are different.
uint64_t Fingerprint(const ImDrawList &list) {
  uint64_t hash = HASH_SEED;
  for (const ImDrawVert &vertex : list.VtxBuffer) {
    for (const float value : {vertex.pos.x, vertex.pos.y}) {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      hash = (hash ^ bits) * HASH_PRIME;
    }
    hash = (hash ^ vertex.col) * HASH_PRIME;
  }
  // The clips, since two toasts can draw the same shapes and differ only in where they are cut.
  for (const ImDrawCmd &command : list.CmdBuffer) {
    for (const float value : {command.ClipRect.x, command.ClipRect.y, command.ClipRect.z, command.ClipRect.w}) {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      hash = (hash ^ bits) * HASH_PRIME;
    }
  }
  return hash;
}

// A toast built, measured with no limit and arranged where tests draw it.
struct PlacedToast {
  ToastParts parts;
  Rect placed;
};
constexpr Vec2 TOAST_ORIGIN{1000, 50};
constexpr float UNLIMITED = std::numeric_limits<float>::infinity();
// The toast title's line height at 1080p.
constexpr float TITLE_LINE_1080P = 34;

PlacedToast PlaceToast(ElementRig &rig, const ToastProps &props, float frameWidth = 1920) {
  rig.Begin();
  PlacedToast built{Toast(rig.ui, props, frameWidth), {}};
  Check(!built.parts.root.IsNone(), "the toast was built");
  const Vec2 size = Measure(rig.ui, built.parts.root, {frameWidth, UNLIMITED});
  built.placed = {TOAST_ORIGIN.x, TOAST_ORIGIN.y, size.x, size.y};
  Check(Arrange(rig.ui, built.parts.root, built.placed), "the toast was arranged");
  return built;
}

// Paints a placed toast at `opacity` into a new list.
std::unique_ptr<ImDrawList> PaintToast(const ElementRig &rig, const PlacedToast &toast, float opacity = 1.0f) {
  auto list = NewDrawList();
  Paint(rig.ui, toast.parts.root, *list, PaintState{.opacity = opacity});
  return list;
}

// `part` relative to the toast's top-left corner.
Rect Within(const Rect &part, const Rect &toast) { return {part.x - toast.x, part.y - toast.y, part.width, part.height}; }

// The title's props as the toast sets them, for asking the text layout what it did.
TextProps WrappedTitle(const ToastProps &props) {
  TextProps title{props.title, DEFAULT_THEME.type.heading, DEFAULT_THEME.colors.text.primary};
  title.overflow = TextOverflow::Wrap;
  title.max_lines = 2;
  return title;
}

// The clip of the first draw command holding a vertex of colour `rgb`, alpha ignored. With
// `glyphs`, only vertices that sample a glyph count, since a plain shape in the same colour, such
// as a white border, would otherwise be found first.
std::optional<ImVec4> ClipOfColour(const ImDrawList &list, ImU32 rgb, bool glyphs = false) {
  for (const ImDrawCmd &command : list.CmdBuffer) {
    for (unsigned i = 0; i < command.ElemCount; i++) {
      const ImDrawVert &vertex = list.VtxBuffer[command.VtxOffset + list.IdxBuffer[command.IdxOffset + i]];
      const bool plain = vertex.uv.x == list._Data->TexUvWhitePixel.x && vertex.uv.y == list._Data->TexUvWhitePixel.y;
      if ((vertex.col & SystemUiTest::RGB_MASK) == (rgb & SystemUiTest::RGB_MASK) && !(glyphs && plain)) {
        return command.ClipRect;
      }
    }
  }
  return std::nullopt;
}

// The clip of the last draw command that draws anything.
std::optional<ImVec4> LastClip(const ImDrawList &list) {
  for (int i = list.CmdBuffer.Size - 1; i >= 0; --i) {
    if (list.CmdBuffer[i].ElemCount > 0) {
      return list.CmdBuffer[i].ClipRect;
    }
  }
  return std::nullopt;
}

// True when `clip` is the rectangle `rect`.
bool IsClip(const std::optional<ImVec4> &clip, const Rect &rect) {
  return clip.has_value() && Near(clip->x, rect.x) && Near(clip->y, rect.y) && Near(clip->z, rect.Right()) &&
         Near(clip->w, rect.Bottom());
}

// A toast at 1080p is 560 wide, with an 80 pixel icon 16 in from the left and the text 16 beyond
// it. A long title wraps to two lines at most, and the toast is at least 112 tall.
void TestToastLayout() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  ImageStore store(4);
  const auto icon = *store.Add(Png(64, 64), 512);
  ToastProps props{SystemUiTest::LONG_TITLE, "Gold trophy"};
  props.icon = store.Texture(icon);
  props.icon_size = store.Size(icon);
  const PlacedToast toast = PlaceToast(rig, props);
  const Rect &placed = toast.placed;
  Check(placed.width == 560.0f && placed.height >= 112.0f, "the toast is the theme's size");
  const Rect icon_rect = Within(Arranged(rig.ui, toast.parts.icon), placed);
  Check(Near(icon_rect, {16, (placed.height - 80) / 2, 80, 80}, 0.5f), "the icon sits in the padding, centred");
  const Rect title = Within(Arranged(rig.ui, toast.parts.title), placed);
  Check(title.x == 112.0f && title.width == 432.0f, "the text starts past the icon and its gap");
  TextProps unlimited = WrappedTitle(props);
  unlimited.max_lines = 0;
  const TextLayout uncut = LayoutText(rig.ui.Ui(), unlimited, 432);
  Check(uncut.lines.size() > 2, "the title would need more than two lines");
  const TextLayout cut = LayoutText(rig.ui.Ui(), WrappedTitle(props), title.width);
  Check(cut.lines.size() == 2 && Near(title.height, 2 * cut.line_height), "a long title is cut to two lines");
  const auto &last_line = cut.lines.back().text;
  Check(last_line.size() >= 3 && last_line.compare(last_line.size() - 3, 3, "\xE2\x80\xA6") == 0,
        "the second line ends with an ellipsis");
  const Rect subtitle = Within(Arranged(rig.ui, toast.parts.subtitle), placed);
  Check(subtitle.y == title.Bottom(), "the subtitle sits under the title");
  const PlacedToast short_toast = PlaceToast(rig, ToastProps{"Saved", "Slot 1"});
  Check(Within(Arranged(rig.ui, short_toast.parts.title), short_toast.placed).x == 16.0f, "without an icon the text starts in the padding");
  Check(short_toast.placed.height == 112.0f, "a short toast is raised to the minimum height");
  Check(Within(Arranged(rig.ui, short_toast.parts.title), short_toast.placed).y == 25.0f, "the text block is centred in the toast's height");
  ElementRig uhd_rig(frame, OUTPUTS[2]);
  const PlacedToast uhd = PlaceToast(uhd_rig, props, 3840);
  Check(uhd.placed.width == 1120.0f && Arranged(uhd_rig.ui, uhd.parts.icon).width == 160.0f &&
            Within(Arranged(uhd_rig.ui, uhd.parts.title), uhd.placed).x == 224.0f,
        "at 4K the toast, its icon, padding and gap all double");
}

// The width caps at 40% of the frame, but never below the 360 minimum, and never past the frame.
void TestToastWidth() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  const ToastProps props{"Saved", "Slot 1"};
  Check(PlaceToast(rig, props, 1000).placed.width == 400.0f, "40% of a narrow frame caps the width");
  Check(PlaceToast(rig, props, 600).placed.width == 360.0f, "the minimum overrides the cap");
  Check(PlaceToast(rig, props, 300).placed.width == 300.0f, "the frame caps the minimum");
}

// The index of the first vertex a draw command uses, for comparing what was drawn first.
unsigned MinimumVertex(const ImDrawList &list, const ImDrawCmd &command) {
  unsigned lowest = std::numeric_limits<unsigned>::max();
  for (unsigned i = 0; i < command.ElemCount; i++) {
    lowest = std::min(lowest, command.VtxOffset + list.IdxBuffer[static_cast<int>(command.IdxOffset + i)]);
  }
  return lowest;
}

// True when some draw command clips to `clip`.
bool ClipsTo(const ImDrawList &list, const Rect &clip) {
  for (const auto &command : list.CmdBuffer) {
    if (Near(command.ClipRect.x, clip.x) && Near(command.ClipRect.y, clip.y) && Near(command.ClipRect.z, clip.Right()) &&
        Near(command.ClipRect.w, clip.Bottom())) {
      return true;
    }
  }
  return false;
}

// How far the toast grows in TestToastDraw's scaled card.
constexpr float TOAST_GROWN = 1.1f;
// The least a title must move when the card grows for the move to be told from none, in pixels.
constexpr float TITLE_PULL_LEAST = 2.0f;
constexpr float EDGE_BAR_WIDTH = 4;
constexpr float CARD_RADIUS = 16;

// The card fills the placed rectangle. An edge bar runs between its rounded corners and a shine is
// clipped inside them, each only when asked for. The text keeps its size when the card is drawn
// larger, as a scaling animation does. The edge bar is outside the clip that holds the content,
// which the card's own fill and the bar share and the title does not.
void TestToastDraw() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  ToastProps props{"Saved", "Slot 1"};
  const PlacedToast plain = PlaceToast(rig, props);
  const Rect placed = plain.placed;
  const Color red = Rgb(0xFF0000);
  const auto drawn = PaintToast(rig, plain);
  auto vertices = VerticesOf(*drawn);
  Check(Near(Bounds(vertices, ToImU32(DEFAULT_THEME.colors.surface.raised)), placed), "the card fills the placed rectangle");
  Check(!HasColor(vertices, red), "no edge bar unless asked for");
  const auto title = Bounds(vertices, ToImU32(DEFAULT_THEME.colors.text.primary));
  const Rect inside = InsideCorners(placed, CARD_RADIUS);
  Check(IsClip(ClipOfColour(*drawn, ToImU32(DEFAULT_THEME.colors.text.primary), true), inside), "the content is clipped inside the card's corners");
  Check(!WithPose(rig.ui, plain.parts.root, ElementPose{.scale = TOAST_GROWN}).IsNone(), "the pose is accepted");
  const Rect title_rect = Arranged(rig.ui, plain.parts.title);
  const auto grown = VerticesOf(*PaintToast(rig, plain));
  const ImU32 primary = ToImU32(DEFAULT_THEME.colors.text.primary);
  Check(Near(Bounds(grown, primary).width, title.width), "in a larger card the text keeps its size");
  // The pose scales about the card's centre. The fill grows by the scale and stays centred, and the
  // title moves with the scaled position of its own centre without growing.
  const Rect fill = Bounds(grown, ToImU32(DEFAULT_THEME.colors.surface.raised));
  Check(Near(fill.width, placed.width * TOAST_GROWN, 1.0f) && Near(fill.height, placed.height * TOAST_GROWN, 1.0f), "the card grows by the pose's scale");
  Check(Near(fill.Center().x, placed.Center().x, 1.0f) && Near(fill.Center().y, placed.Center().y, 1.0f), "about its centre");
  const Vec2 pull{(title_rect.Center().x - placed.Center().x) * (TOAST_GROWN - 1.0f),
                  (title_rect.Center().y - placed.Center().y) * (TOAST_GROWN - 1.0f)};
  Check(std::fabs(pull.x) > TITLE_PULL_LEAST, "the title is far enough from the centre to be seen moving");
  const Vec2 before = Bounds(vertices, primary).Center();
  const Vec2 after = Bounds(grown, primary).Center();
  Check(Near(after.x, before.x + pull.x, 1.0f) && Near(after.y, before.y + pull.y, 1.0f), "and the title's centre lands on the scaled point");
  props.edge_bar = red;
  const PlacedToast unshone = PlaceToast(rig, props);
  const int unshone_vertices = PaintToast(rig, unshone)->VtxBuffer.Size;
  props.shine = 0.5f;
  const PlacedToast barred = PlaceToast(rig, props);
  const auto with_bar = PaintToast(rig, barred);
  Check(Near(Bounds(VerticesOf(*with_bar), ToImU32(red)), {placed.x, placed.y + CARD_RADIUS, EDGE_BAR_WIDTH, placed.height - 2 * CARD_RADIUS}),
        "the edge bar is 4 pixels wide and runs between the corners");
  Check(!IsClip(ClipOfColour(*with_bar, ToImU32(red)), inside), "the edge bar is outside the content's clip");
  Check(ClipOfColour(*with_bar, ToImU32(red))->z == ClipOfColour(*with_bar, ToImU32(DEFAULT_THEME.colors.surface.raised))->z,
        "and in the clip the card's own fill is drawn in");
  // The shine is drawn last, over the content. ImGui merges its commands into the content's when
  // they share a clip, so it shows as added vertices in a last command clipped inside the corners.
  Check(with_bar->VtxBuffer.Size > unshone_vertices && IsClip(LastClip(*with_bar), inside),
        "the shine is clipped inside the rounded corners");
}

// A trophy's glow is drawn before its icon, so the icon covers it, and is clipped inside the card
// so it doesn't spill over the game.
void TestToastIconGlow() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  ImageStore store(4);
  const auto icon = *store.Add(Png(64, 64), 512);
  ToastProps props{"Gold trophy", "Master Alchemist"};
  props.icon = store.Texture(icon);
  props.icon_size = store.Size(icon);
  props.icon_glow = DEFAULT_THEME.colors.trophy.gold;
  const PlacedToast toast = PlaceToast(rig, props);
  const auto list = PaintToast(rig, toast);
  unsigned last_glow = 0;
  bool glow = false;
  for (int i = 0; i < list->VtxBuffer.Size; i++) {
    if ((list->VtxBuffer[i].col & SystemUiTest::RGB_MASK) == (ToImU32(props.icon_glow) & SystemUiTest::RGB_MASK)) {
      glow = true;
      last_glow = static_cast<unsigned>(i);
    }
  }
  unsigned first_image = std::numeric_limits<unsigned>::max();
  for (const auto &command : list->CmdBuffer) {
    if (command.TexRef._TexData == props.icon) {
      first_image = std::min(first_image, MinimumVertex(*list, command));
    }
  }
  Check(glow && first_image != std::numeric_limits<unsigned>::max(), "the glow and the icon are drawn");
  Check(last_glow < first_image, "the glow is drawn before the icon");
}

// A trophy cup sits before the title, which moves right by the cup's 22 pixels and an 8 pixel gap
// and gets narrower by the same. The subtitle stays where it was. Without a cup, the title starts
// at the text's edge, which is the baseline. The cup is centred on the title's first line, so a
// title that wraps to two lines leaves it beside the first, not between the two.
void TestToastCup() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  const auto &tokens = ToastDefaults(DEFAULT_THEME);
  const float cup_width = rig.scale.Px(tokens.cup_width);
  const float cup_height = rig.scale.Px(tokens.cup_height);
  const float cup_room = cup_width + rig.scale.Px(tokens.cup_gap);
  const ToastProps plain{"Master Alchemist", "Trophy earned!"};
  ToastProps wrapped = plain;
  wrapped.title = SystemUiTest::WRAPPING_TITLE;
  // Read before the next toast is placed, which starts a new frame and makes these handles stale.
  const PlacedToast without = PlaceToast(rig, plain);
  const Rect title_without = Arranged(rig.ui, without.parts.title);
  const bool has_cup_without = !without.parts.cup.IsNone();
  const Rect wrapped_without = Arranged(rig.ui, PlaceToast(rig, wrapped).parts.title);
  auto props = plain;
  props.title_cup = DEFAULT_THEME.colors.trophy.gold;
  const PlacedToast with = PlaceToast(rig, props);
  const Rect title = Arranged(rig.ui, with.parts.title);
  Check(title.x == title_without.x + cup_room, "the title makes room for the cup");
  Check(Arranged(rig.ui, with.parts.subtitle).x == title_without.x, "the subtitle stays at the text's edge");
  const Rect cup = Arranged(rig.ui, with.parts.cup);
  Check(cup.x == title_without.x && cup.width == cup_width && cup.height == cup_height, "the cup is the token's size");
  Check(Near(cup.y + cup.height / 2.0f, title.y + TITLE_LINE_1080P / 2.0f, 0.5f), "the cup is centred on the title's first line");
  const auto drawn = VerticesOf(*PaintToast(rig, with));
  Check(Near(Bounds(drawn, ToImU32(props.title_cup)), cup, 1.0f), "the cup is drawn in its place in the grade's colour");
  Check(!has_cup_without, "without a cup there is no space for one");
  wrapped.title_cup = props.title_cup;
  const PlacedToast two_lines = PlaceToast(rig, wrapped);
  const Rect long_title = Arranged(rig.ui, two_lines.parts.title);
  const Rect long_cup = Arranged(rig.ui, two_lines.parts.cup);
  Check(long_title.width == wrapped_without.width - cup_room, "a title that fills its line gets narrower by the cup's room");
  Check(Near(long_title.height, 2 * TITLE_LINE_1080P), "the long title wraps to two lines");
  Check(Near(long_cup.y + long_cup.height / 2.0f, long_title.y + TITLE_LINE_1080P / 2.0f, 0.5f), "the cup stays on the first of two lines");
  Check(!Near(long_cup.y + long_cup.height / 2.0f, long_title.y + long_title.height / 2.0f, 2.0f), "and not on the middle of the whole title");
}

// A toast with a cup and no title keeps the line the cup sits on, so it is one title line taller
// than the same toast without the cup. The default theme's minimum height hides it, because the
// content is far shorter, so the test lowers the minimum to nothing. A title has that line already,
// so there the cup changes nothing.
void TestToastEmptyTitleWithCup() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  ToastProps props{"", "Trophy earned!"};
  props.tokens = TokenPatch<ToastTokens> {}.With<&ToastTokens::min_height>(0.0f);
  const float without_cup = PlaceToast(rig, props).placed.height;
  props.title_cup = DEFAULT_THEME.colors.trophy.gold;
  Check(Near(PlaceToast(rig, props).placed.height, without_cup + TITLE_LINE_1080P), "with a cup it is one title line taller");
  props.title = "Saved";
  const float titled_with_cup = PlaceToast(rig, props).placed.height;
  props.title_cup = Color{};
  Check(Near(PlaceToast(rig, props).placed.height, titled_with_cup), "while a title brings the heights back together");
}

constexpr float CUP_FADE = 0.5f;
// How far the cup's handle strokes may spill past its rectangle, about 1.3 pixels at the size the
// test draws it at, with room to spare.
constexpr float CUP_SPILL_SLACK = 3.0f;

// The cup glyph fills its rectangle: its handles reach the sides and its base the bottom. Opacity
// fades it, and a transparent colour draws nothing.
void TestCupGlyph() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  const Rect rect{100, 100, 44, 42};
  const Color gold = DEFAULT_THEME.colors.trophy.gold;
  rig.Begin();
  const auto plain = DrawElement(rig, TrophyCup(rig.ui, {.color = gold}), rect);
  // The handles are strokes, which reach half their width past their outline, so at this size the
  // cup spills over each side by about 1.3 pixels.
  Check(Near(Bounds(VerticesOf(*plain)), rect, CUP_SPILL_SLACK), "the cup fills its rectangle");
  rig.Begin();
  const auto faded = DrawElement(rig, TrophyCup(rig.ui, {.color = gold}), rect, CUP_FADE);
  Check(HasColor(VerticesOf(*faded), gold.MulAlpha(CUP_FADE)), "opacity fades the cup");
  rig.Begin();
  const auto none = DrawElement(rig, TrophyCup(rig.ui, {.color = Color{}}), rect);
  Check(none->VtxBuffer.Size == 0, "a transparent cup draws nothing");
}

const Color CUP_GRADES[] = {DEFAULT_THEME.colors.trophy.gold, DEFAULT_THEME.colors.trophy.silver,
                            DEFAULT_THEME.colors.trophy.bronze, DEFAULT_THEME.colors.trophy.platinum};
constexpr float CUP_POSE_SCALE = 0.5f;
constexpr Vec2 CUP_POSE_OFFSET{30, 12};
// A cup in a box is sized in design units, so it lands on whole pixels at every output.
constexpr float CUP_BOX_WIDTH = 24;
constexpr float CUP_BOX_HEIGHT = 23;
// Space on offer when measuring a cup, more than it could take.
constexpr float CUP_PROBE_SPACE = 1000;

// A cup drawn by itself at `rect`, sized as the cup in BuildCupInBox is, with nothing above it, for
// holding a cup under a parent to.
std::unique_ptr<ImDrawList> BareCup(ElementRig &rig, const Rect &rect, Color color, float opacity = 1.0f) {
  rig.Begin();
  return DrawElement(rig, TrophyCup(rig.ui, {.color = color}, {.width = Dp(CUP_BOX_WIDTH), .height = Dp(CUP_BOX_HEIGHT)}), rect, opacity);
}

// The cup's parts, each where the 22 by 21 grid puts it, at sizes that stretch the grid each way.
// Written out by hand, as every other cup test compares one drawn cup with another, which a change
// of shape would pass. Corners and strokes follow the smaller of the two stretches.
void TestCupShape() {
  struct GridPart {
    float left;
    float top;
    float right;
    float bottom;
  };
  struct Stretch {
    float x;
    float y;
  };
  constexpr GridPart BOWL{3, 0, 19, 13};
  constexpr GridPart LEFT_HANDLE{0, 2, 5, 8};
  constexpr GridPart RIGHT_HANDLE{17, 2, 22, 8};
  constexpr GridPart STEM{9.5f, 13, 12.5f, 18};
  constexpr GridPart BASE{6, 18, 16, 21};
  constexpr float GRID_WIDTH = 22;
  constexpr float GRID_HEIGHT = 21;
  constexpr float CORNER = 2.0f;
  constexpr float HANDLE_STROKE = 1.8f;
  constexpr float BASE_CORNER = 1.5f;
  constexpr float ORIGIN = 100;
  constexpr float EXACT = 0.0f;
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  const Color gold = DEFAULT_THEME.colors.trophy.gold;
  const ImU32 packed = ToImU32(gold);
  for (const Stretch &stretch : {Stretch{2, 2}, Stretch{2, 4}, Stretch{4, 2}}) {
    const float unit = std::min(stretch.x, stretch.y);
    const auto top_left = [&](const GridPart &part) { return ImVec2{ORIGIN + part.left * stretch.x, ORIGIN + part.top * stretch.y}; };
    const auto bottom_right = [&](const GridPart &part) { return ImVec2{ORIGIN + part.right * stretch.x, ORIGIN + part.bottom * stretch.y}; };
    auto expected = NewDrawList();
    expected->AddRectFilled(top_left(BOWL), bottom_right(BOWL), packed, CORNER * unit, ImDrawFlags_RoundCornersBottom);
    expected->AddRect(top_left(LEFT_HANDLE), bottom_right(LEFT_HANDLE), packed, CORNER * unit, HANDLE_STROKE * unit, ImDrawFlags_RoundCornersLeft);
    expected->AddRect(top_left(RIGHT_HANDLE), bottom_right(RIGHT_HANDLE), packed, CORNER * unit, HANDLE_STROKE * unit, ImDrawFlags_RoundCornersRight);
    expected->AddRectFilled(top_left(STEM), bottom_right(STEM), packed);
    expected->AddRectFilled(top_left(BASE), bottom_right(BASE), packed, BASE_CORNER * unit);
    const Rect rect{ORIGIN, ORIGIN, GRID_WIDTH * stretch.x, GRID_HEIGHT * stretch.y};
    const auto actual = BareCup(rig, rect, gold);
    Check(actual->VtxBuffer.Size > 0 && NearDrawing(*actual, *expected, EXACT), "the cup's parts are where the grid puts them");
  }
}

// The cup draws in the colour it is given, at every output, for each grade colour and at two sizes.
// It differs from the same cup in the next grade's colour, so a cup that took its colour from
// somewhere else could not pass.
void TestCupTakesItsGradeColour() {
  Frame frame;
  const Rect sizes[] = {Rect{100, 100, 22, 21}, Rect{50, 60, 64, 62}};
  const size_t grades = std::size(CUP_GRADES);
  for (const Output &output : OUTPUTS) {
    for (const Rect &rect : sizes) {
      for (size_t grade = 0; grade < grades; ++grade) {
        ElementRig rig(frame, output);
        const auto actual = BareCup(rig, rect, CUP_GRADES[grade]);
        const auto wrong = BareCup(rig, rect, CUP_GRADES[(grade + 1) % grades]);
        Check(HasColor(VerticesOf(*actual), CUP_GRADES[grade]), "the cup draws in its grade's colour");
        Check(!NearDrawing(*actual, *wrong), "and differs from a cup of another grade");
      }
    }
  }
}

// A cup inside a box of its own.
struct CupInBox {
  Element root;
  Element cup;
};
CupInBox BuildCupInBox(ElementRig &rig, Color color) {
  const Element cup = TrophyCup(rig.ui, {.color = color}, {.width = Dp(CUP_BOX_WIDTH), .height = Dp(CUP_BOX_HEIGHT)});
  const std::array<Element, 1> kids{cup};
  return {Box(rig.ui, {}, {}, kids), cup};
}

// A cup under a parent faded to half draws what a bare cup faded to half does, with half the vertex
// alpha, and differs from the unfaded one. The bare cups differ from this one only in having no parent.
void TestCupFadesWithParent() {
  Frame frame;
  const Rect parentRect{100, 100, 200, 100};
  for (const Output &output : OUTPUTS) {
    ElementRig rig(frame, output);
    rig.Begin();
    const CupInBox built = BuildCupInBox(rig, CUP_GRADES[0]);
    const auto actual = DrawElement(rig, built.root, parentRect, CUP_FADE);
    const Rect placed = Arranged(rig.ui, built.cup);
    const auto expected = BareCup(rig, placed, CUP_GRADES[0], CUP_FADE);
    const auto wrong = BareCup(rig, placed, CUP_GRADES[0]);
    Check(actual->VtxBuffer.Size > 0, "the faded cup is not empty");
    Check(HasColor(VerticesOf(*actual), CUP_GRADES[0].MulAlpha(CUP_FADE)), "its vertices carry half the alpha");
    Check(NearDrawing(*actual, *expected) && SameCommands(*actual, *expected), "and it matches a bare cup at half opacity");
    Check(!NearDrawing(*actual, *wrong), "and differs from the unfaded cup");
  }
}

// A cup under a posed parent is drawn where the pose puts it, which is a bare cup at the painted
// rectangle. That rectangle differs from the arranged one, so a cup that ignored the pose, or drew
// at the layout rectangle, could not pass.
void TestCupFollowsPose() {
  Frame frame;
  const Rect parentRect{100, 100, 200, 100};
  for (const Output &output : OUTPUTS) {
    ElementRig rig(frame, output);
    rig.Begin();
    const CupInBox built = BuildCupInBox(rig, CUP_GRADES[0]);
    Check(!WithPose(rig.ui, built.root, ElementPose{.offset = CUP_POSE_OFFSET, .scale = CUP_POSE_SCALE}).IsNone(), "the pose is accepted");
    const auto actual = DrawElement(rig, built.root, parentRect);
    const Rect painted = Painted(rig.ui, built.cup);
    const Rect arranged = Arranged(rig.ui, built.cup);
    Check(!Near(painted, arranged, 1.0f), "the pose moves the cup");
    const auto expected = BareCup(rig, painted, CUP_GRADES[0]);
    const auto wrong = BareCup(rig, arranged, CUP_GRADES[0]);
    Check(actual->VtxBuffer.Size > 0, "the posed cup is not empty");
    Check(NearDrawing(*actual, *expected) && SameCommands(*actual, *expected), "the cup is drawn at the painted rectangle");
    Check(!NearDrawing(*actual, *wrong), "and not at the arranged one");
  }
}

// A cup with no size is the toast's cup size, 22 by 21 design units, whatever the output.
void TestCupDefaultSize() {
  Frame frame;
  for (const Output &output : OUTPUTS) {
    ElementRig rig(frame, output);
    rig.Begin();
    const Vec2 size = Measure(rig.ui, TrophyCup(rig.ui, {.color = CUP_GRADES[0]}), {CUP_PROBE_SPACE, CUP_PROBE_SPACE});
    const auto &toast = ToastDefaults(DEFAULT_THEME);
    Check(Near(size.x, rig.scale.Px(toast.cup_width)) && Near(size.y, rig.scale.Px(toast.cup_height)), "a bare cup takes the toast's cup size");
  }
}

// The icon's glow spills past the card. It is the trophy glow's full 32 pixels, which reach past the
// card's 16 pixel padding, and it is drawn outside the clip the rest of the content is under, so
// that clip can't cut it off at the card. The same 32 pixels also tell it from a narrower glow.
void TestToastGlowSpills() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  ImageStore store(4);
  const auto icon = *store.Add(Png(64, 64), 512);
  ToastProps props{"Gold trophy", "Master Alchemist"};
  props.icon = store.Texture(icon);
  props.icon_size = store.Size(icon);
  props.icon_glow = Rgb(0x00FF00);
  const PlacedToast toast = PlaceToast(rig, props);
  const auto list = PaintToast(rig, toast);
  float left = std::numeric_limits<float>::max();
  for (const auto &vertex : list->VtxBuffer) {
    if ((vertex.col & SystemUiTest::RGB_MASK) == (ToImU32(props.icon_glow) & SystemUiTest::RGB_MASK)) {
      left = std::min(left, vertex.pos.x);
    }
  }
  const Rect icon_rect = Arranged(rig.ui, toast.parts.icon);
  Check(Near(left, icon_rect.x - 32.0f), "the glow is the full 32 wide");
  Check(left < toast.placed.x, "which reaches past the card");
  const Rect inside = InsideCorners(toast.placed, CARD_RADIUS);
  const auto clip = ClipOfColour(*list, ToImU32(props.icon_glow));
  Check(clip.has_value() && !IsClip(clip, inside), "and isn't clipped to the card");
}

// Opacity fades every part of a toast together: its shadow, card, edge bar, glow, title,
// subtitle and shine.
void TestToastFades() {
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  ImageStore store(4);
  const auto icon = *store.Add(Png(64, 64), 512);
  ToastProps props{"Gold trophy", "Master Alchemist"};
  props.icon = store.Texture(icon);
  props.icon_size = store.Size(icon);
  props.icon_glow = Rgb(0x00FF00);
  props.edge_bar = Rgb(0xFF0000);
  props.shine = 0.5f;
  const auto vertices = VerticesOf(*PaintToast(rig, PlaceToast(rig, props), 0.5f));
  const auto &colors = DEFAULT_THEME.colors;
  // Part way out, an unfaded halo has this same alpha, so the unfaded peak has to be gone as well.
  Check(HasColor(vertices, colors.effect.shadow.MulAlpha(0.45f * 0.5f)), "the shadow fades");
  Check(!HasColor(vertices, colors.effect.shadow.MulAlpha(0.45f)), "and none of it stays at full strength");
  Check(HasColor(vertices, colors.surface.raised.MulAlpha(0.5f)), "the card fades");
  Check(HasColor(vertices, props.edge_bar.MulAlpha(0.5f)), "the edge bar fades");
  Check(HasColor(vertices, props.icon_glow.MulAlpha(0.30f * 0.5f)), "the glow fades");
  Check(!HasColor(vertices, props.icon_glow.MulAlpha(0.30f)), "and none of it stays at full strength");
  Check(HasColor(vertices, colors.text.primary.MulAlpha(0.5f)), "the title fades");
  Check(HasColor(vertices, colors.text.secondary.MulAlpha(0.5f)), "the subtitle fades");
  Check(HasColor(vertices, colors.effect.shine.WithAlpha(0.14f).MulAlpha(0.5f)), "the shine fades");
}

constexpr float PROOF_NARROW_FRAME = 600;

// A wide icon fills its square by covering it, with the rounded corners of the toast's tokens, at
// every output, which is the same image drawn with those settings. A stretched icon or a
// square-cornered one draws something else, so the comparison can tell them apart.
void TestToastIconFit() {
  constexpr int WIDE_ICON_WIDTH = 128;
  constexpr int WIDE_ICON_HEIGHT = 64;
  constexpr int LARGEST_ICON_SIDE = 512;
  constexpr int STORE_SLOTS = 4;
  Frame frame;
  ImageStore store(STORE_SLOTS);
  const auto icon = *store.Add(Png(WIDE_ICON_WIDTH, WIDE_ICON_HEIGHT), LARGEST_ICON_SIDE);
  ToastProps props{"Gold trophy", "Master Alchemist"};
  props.icon = store.Texture(icon);
  props.icon_size = store.Size(icon);
  for (const Output &output : OUTPUTS) {
    ElementRig rig(frame, output);
    const PlacedToast toast = PlaceToast(rig, props);
    const auto actual = PaintToast(rig, toast);
    const Rect rect = Arranged(rig.ui, toast.parts.icon);
    // In design units, as DrawImage scales it.
    const float radius = ToastDefaults(DEFAULT_THEME).icon_radius;
    const auto drawn = [&](float corner, ImageFit fit) {
      auto list = NewDrawList();
      DrawImage(*list, rig.ui.Ui(), props.icon, props.icon_size, rect, corner, fit);
      return list;
    };
    Check(ContainsDrawing(*actual, *drawn(radius, ImageFit::Cover)), "the icon covers its square with rounded corners");
    Check(!ContainsDrawing(*actual, *drawn(radius, ImageFit::Fill)), "and isn't stretched");
    Check(!ContainsDrawing(*actual, *drawn(0.0f, ImageFit::Cover)), "and has its corners");
  }
}

// A subtitle a little too long for its line shrinks to fit, so every letter is drawn, four vertices
// each. One far too long can't shrink enough and is cut, which shows the count can tell the two apart.
void TestToastSubtitleShrinks() {
  constexpr size_t VERTICES_PER_LETTER = 4;
  constexpr size_t FAR_TOO_LONG = 3;
  Frame frame;
  ElementRig rig(frame, OUTPUTS[1]);
  const ToastProps plain{"Saved", "Slot 1"};
  const float room = Arranged(rig.ui, PlaceToast(rig, plain).parts.subtitle).width;
  Check(room > 0.0f, "the subtitle has a line to fit");
  const Color color = DEFAULT_THEME.colors.text.secondary;
  std::string subtitle;
  while (LayoutText(rig.ui.Ui(), TextProps{subtitle, DEFAULT_THEME.type.body_small, color}, UNLIMITED).size.x <= room) {
    subtitle += 'a';
  }
  const auto letters_drawn = [&](const std::string &text) {
    const ToastProps props{"Saved", text};
    size_t vertices = 0;
    for (const ImDrawVert &vertex : VerticesOf(*PaintToast(rig, PlaceToast(rig, props)))) {
      vertices += vertex.col == ToImU32(color) ? 1 : 0;
    }
    return vertices;
  };
  Check(letters_drawn(subtitle) == subtitle.size() * VERTICES_PER_LETTER, "a subtitle just past its line draws every letter");
  std::string far_too_long;
  for (size_t i = 0; i < FAR_TOO_LONG; ++i) {
    far_too_long += subtitle;
  }
  Check(letters_drawn(far_too_long) < far_too_long.size() * VERTICES_PER_LETTER, "and one far past it is cut");
}

// A toast's style is a card's surface and border with the corners its tokens give. The defaults of
// the two happen to share a radius, so the test asks for another. The tests that draw a toast
// check its shadow.
void TestToastStyleTakesItsTokens() {
  constexpr float OTHER_RADIUS = 5.0f;
  ToastTokens tokens = ToastDefaults(DEFAULT_THEME);
  tokens.radius = OTHER_RADIUS;
  const Style toast = ToastStyle(DEFAULT_THEME, tokens);
  const Style card = CardStyle(DEFAULT_THEME, CardDefaults(DEFAULT_THEME));
  Check(card.radius.value != OTHER_RADIUS, "a card has other corners");
  Check(toast.radius.value == OTHER_RADIUS, "and a toast has the corners its tokens give");
  Check(ToImU32(toast.fill) == ToImU32(card.fill) && ToImU32(toast.border) == ToImU32(card.border), "on a card's surface and border");
}

// Every variant of the toast (plain, icon, a glow, edge bar with shine, a long title, a wrapped title
// with a cup) at every output, two frame widths and two fades. Each draws something, clips as
// ToastClipFault requires, and differs from the same toast with another title, so a toast that drew
// nothing or only its card could not pass. Every variant draws its own picture.
void TestToastVariants() {
  Frame frame;
  ImageStore store(4);
  const auto icon = *store.Add(Png(64, 64), 512);
  const std::vector<ToastProps> variants = SystemUiTest::ToastVariants(store.Texture(icon));
  std::set<uint64_t> distinct;
  for (const Output &output : OUTPUTS) {
    for (const ToastProps &props : variants) {
      ToastProps other = props;
      other.title = "A different title";
      for (const float frameWidth : {output.width, PROOF_NARROW_FRAME}) {
        ElementRig rig(frame, output);
        for (const float opacity : {1.0f, 0.5f}) {
          const PlacedToast toast = PlaceToast(rig, props, frameWidth);
          const auto actual = PaintToast(rig, toast, opacity);
          Check(actual->VtxBuffer.Size > 0, "the toast is not empty");
          SystemUiTest::ToastClipExpect expect;
          expect.card = toast.placed;
          expect.radius = std::min(rig.scale.Px(ToastDefaults(DEFAULT_THEME).radius), toast.placed.height / 2.0f);
          expect.has_icon = props.icon != nullptr;
          expect.edge_bar = props.edge_bar;
          expect.icon_glow = props.icon_glow;
          expect.shine = props.shine.has_value();
          const char *fault = SystemUiTest::ToastClipFault(*actual, expect);
          Check(fault == nullptr, fault != nullptr ? fault : "");
          const auto wrong = PaintToast(rig, PlaceToast(rig, other, frameWidth), opacity);
          Check(!NearGeometry(*actual, *wrong), "and differs from one with another title");
          if (frameWidth == output.width && opacity == 1.0f && output.width == OUTPUTS[1].width) {
            distinct.insert(Fingerprint(*actual));
          }
        }
      }
    }
  }
  Check(distinct.size() == variants.size(), "and each variant draws a toast of its own");
}

} // namespace

int main() {
  TestBox();
  TestBoxFades();
  TestCard();
  TestProgressBar();
  TestProgressFillEdgeSnaps();
  TestProgressBarDefaultHeight();
  TestImageStore();
  TestImage();
  TestPosed();
  TestToastLayout();
  TestToastWidth();
  TestToastDraw();
  TestToastIconGlow();
  TestToastFades();
  TestToastCup();
  TestToastVariants();
  TestToastStyleTakesItsTokens();
  TestToastIconFit();
  TestToastSubtitleShrinks();
  TestToastEmptyTitleWithCup();
  TestCupGlyph();
  TestCupShape();
  TestCupTakesItsGradeColour();
  TestCupFadesWithParent();
  TestCupFollowsPose();
  TestCupDefaultSize();
  TestToastGlowSpills();
  std::puts("System UI components tests passed");
  return 0;
}
