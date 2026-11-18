#include "SystemUiDrawingKit.h"
#include "SystemUiToastKit.h"
#include "graphics/presentation/system/animation/easing.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/effects/shine.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/imgui/fonts.h"
#include "graphics/presentation/system/imgui/imageStore.h"
#include "graphics/presentation/system/views/trophyToast.h"

#include "imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics::SystemUi;

constexpr const char *REGULAR_FONT = KYTY_TEST_FONT_REGULAR;
constexpr const char *BOLD_FONT = KYTY_TEST_FONT_BOLD;
constexpr float TOLERANCE = 0.01f;
// Four seconds on screen: a 280 ms entrance, a 180 ms exit, so the exit starts at 3820.
constexpr float SECONDS = 4.0f;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiViewsTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two values agree to within rounding noise.
bool Near(float actual, float expected) { return std::fabs(actual - expected) <= TOLERANCE; }

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
  // The store the views' animations are kept in, the same one from frame to frame, collected after
  // each, as the overlay does.
  InteractionStore interactions;
};

// A moment `ms` milliseconds after an arbitrary origin.
UiClock::time_point At(double ms) {
  return UiClock::time_point{} +
         std::chrono::duration_cast<UiClock::duration>(std::chrono::duration<double, std::milli>(ms));
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

// The smallest rectangle around the vertices of colour `only`.
Rect Bounds(const std::vector<ImDrawVert> &vertices, ImU32 only) {
  float left = std::numeric_limits<float>::max(), top = left, right = -left, bottom = -left;
  for (const auto &vertex : vertices) {
    if (vertex.col == only) {
      left = std::min(left, vertex.pos.x);
      top = std::min(top, vertex.pos.y);
      right = std::max(right, vertex.pos.x);
      bottom = std::max(bottom, vertex.pos.y);
    }
  }
  return {left, top, right - left, bottom - top};
}

// True when some vertex has colour `color`.
bool HasColor(const std::vector<ImDrawVert> &vertices, Color color) {
  return std::any_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &v) { return v.col == ToImU32(color); });
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

// A gold trophy with an icon.
TrophyUnlock Gold() { return {"Master Alchemist", 2, Png(64, 64)}; }

// The options for a four-second toast.
TrophyToastOptions Options() { return {SECONDS}; }

// The toast's card where it rests: the top right of the 1080p safe area, which is 32 pixels in.
Rect Resting(float height) { return {1920.0f - 32.0f - 560.0f, 32.0f, 560.0f, height}; }

// Draws the view at `ms` and returns the vertices it added, and whether it needs more frames.
std::pair<std::vector<ImDrawVert>, bool> DrawAt(Frame &frame, TrophyToastView &view, double ms,
                                                std::optional<UiContext> context = std::nullopt) {
  const auto used = context.value_or(frame.Context());
  const int mark = frame.Mark();
  UiFrame ui;
  ui.Begin(used.theme, used.scale, used.fonts, frame.interactions, At(ms));
  const bool more = view.Draw(ui, frame.List(), used.scale.SafeArea());
  frame.interactions.Collect();
  return {frame.Since(mark), more};
}

// Shows `unlock` at `ms` and draws the frame it is shown in, as the overlay does, so the toast's
// entrance starts then. What that first frame drew is not wanted.
void ShowAt(Frame &frame, TrophyToastView &view, TrophyUnlock unlock, const char *subtitle, const Theme &theme,
            const TrophyToastOptions &options, double ms) {
  view.Show(std::move(unlock), subtitle, theme, options, At(ms));
  (void)DrawAt(frame, view, ms);
}

// The top of the card's own fill among `vertices`, matched by colour whatever its fading alpha.
// The shadow always reaches above the card, so only the fill shows where the card itself is.
float CardTop(const std::vector<ImDrawVert> &vertices) {
  const auto rgb = ToImU32(DEFAULT_THEME.colors.surface.raised) & SystemUiTest::RGB_MASK;
  float top = std::numeric_limits<float>::max();
  for (const auto &vertex : vertices) {
    if ((vertex.col & SystemUiTest::RGB_MASK) == rgb) {
      top = std::min(top, vertex.pos.y);
    }
  }
  return top;
}

// The toast slides down into place from above while it fades in, rests at the top right of the
// safe area, then leaves when its four seconds are up. A view with nothing to show draws nothing
// and needs no frames.
void TestToastLife() {
  Frame frame;
  ImageStore store(4);
  TrophyToastView view(store);
  Check(view.Idle() && !DrawAt(frame, view, 0).second, "an idle view draws nothing");
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 0);
  const auto raised = DEFAULT_THEME.colors.surface.raised;
  const auto [entering, entering_more] = DrawAt(frame, view, 100);
  Check(entering_more && !HasColor(entering, raised), "entering, the card is still fading in");
  const auto [resting, resting_more] = DrawAt(frame, view, 1000);
  const auto card = Bounds(resting, ToImU32(raised));
  Check(resting_more && Near(card.x, Resting(card.height).x) && Near(card.y, 32.0f) && Near(card.width, 560.0f),
        "it rests at the top right of the safe area");
  Check(CardTop(DrawAt(frame, view, 140).first) < 32.0f - 1.0f, "entering, the card is still above where it rests");
  // Just before the exit the card is still fully opaque where it rests. 80 ms into the exit it is
  // fading and moving back up, so the exit starts at 3820 and leaves upwards.
  const auto [holding, holding_more] = DrawAt(frame, view, 3819);
  Check(holding_more && HasColor(holding, raised) && Near(CardTop(holding), 32.0f), "it holds in place until its exit");
  const auto leaving = DrawAt(frame, view, 3900).first;
  Check(!HasColor(leaving, raised) && CardTop(leaving) < 32.0f - 1.0f, "it fades out upwards");
  const auto [gone, gone_more] = DrawAt(frame, view, 4001);
  Check(gone.empty() && !gone_more && view.Idle(), "after four seconds it is gone and the view is idle");
}

// A toast is drawn with no window open, as the overlay draws it, so the current window is ImGui's
// fallback, which ImGui shows as an empty Debug window once something marks it as written to. A toast
// really drawn leaves it unmarked as it enters, rests and leaves, with and without reduced motion, which
// draws different things. Marking it by hand is the baseline, which shows the flag is what this reads.
void TestToastLeavesTheFallbackWindowHidden() {
  constexpr std::array<double, 3> PHASES{100.0, 1000.0, 3900.0};
  for (const bool reduced : {false, true}) {
    Frame frame;
    ImageStore store(4);
    TrophyToastView view(store);
    const ImGuiWindow *fallback = ImGui::GetCurrentWindowRead();
    Check(fallback != nullptr && fallback->IsFallbackWindow && !fallback->WriteAccessed, "with no window open, the fallback is current and unmarked");
    TrophyToastOptions options = Options();
    options.reduced_motion = reduced;
    ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, options, 0);
    Check(!fallback->WriteAccessed, "showing a toast leaves the fallback window hidden");
    for (const double phase : PHASES) {
      Check(!DrawAt(frame, view, phase).first.empty(), "the toast is drawn");
      Check(!fallback->WriteAccessed, "and so does drawing it, whatever its phase");
    }
    (void)ImGui::GetCurrentWindow();
    Check(fallback->WriteAccessed, "where writing to it would show it");
  }
}

// The toast's icon is loaded into the store and released once the toast has left, so its texture
// is freed after the frames using it.
void TestIconLifetime() {
  Frame frame;
  ImageStore store(4);
  TrophyToastView view(store);
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 0);
  Check(store.LiveCount() == 1, "showing a toast loads its icon");
  DrawAt(frame, view, 4001);
  Check(store.LiveCount() == 0 && store.RetiringCount() == 1, "a toast that has left releases its icon");
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 5000);
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 5100);
  Check(store.LiveCount() == 1 && store.RetiringCount() == 2, "replacing a toast releases the old icon");
  ShowAt(frame, view, {"Broken icon", 2, std::vector<std::byte>(16, std::byte{1})}, "Trophy earned!", DEFAULT_THEME, Options(), 6000);
  Check(store.LiveCount() == 0, "an icon that can't be decoded is left out");
  Check(!DrawAt(frame, view, 7000).first.empty(), "a toast without an icon still shows");
}

// The grade's colour tints the cup and the glow. Only the cup is drawn in the exact colour, so
// each grade's toast has its own colour and none of the other three. A trophy without a grade gets
// none of them.
void TestGradeColours() {
  Frame frame;
  ImageStore store(8); // five toasts, never collected
  TrophyToastView view(store);
  const auto &trophy = DEFAULT_THEME.colors.trophy;
  const Color colours[] = {trophy.platinum, trophy.gold, trophy.silver, trophy.bronze};
  double ms = 0;
  for (int32_t grade = 1; grade <= 4; ++grade, ms += 5000) {
    ShowAt(frame, view, {"Trophy", grade, Png(64, 64)}, "Trophy earned!", DEFAULT_THEME, Options(), ms);
    const auto drawn = DrawAt(frame, view, ms + 1000).first;
    for (int32_t other = 1; other <= 4; ++other) {
      Check(HasColor(drawn, colours[other - 1]) == (other == grade), "each grade has its own colour and no other");
    }
  }
  ShowAt(frame, view, {"Ungraded", 0, Png(64, 64)}, "Trophy earned!", DEFAULT_THEME, Options(), ms);
  const auto plain = DrawAt(frame, view, ms + 1000).first;
  Check(!plain.empty(), "an ungraded trophy still shows");
  for (const auto colour : colours) {
    Check(!HasColor(plain, colour), "an ungraded trophy has no cup");
  }
}

// Platinum and gold toasts get one shine, 200 ms after fading in, lasting 700 ms. The shine adds
// vertices to the toast, clipped inside the card's corners, so a frame with it draws more than the
// same frame without. Before the sweep, after it, and on a silver toast, the count matches the
// baseline.
void TestShineTiming() {
  const auto vertices = [](int32_t grade, double ms) {
    Frame frame;
    ImageStore store(4);
    TrophyToastView view(store);
    ShowAt(frame, view, {"Trophy", grade, Png(64, 64)}, "Trophy earned!", DEFAULT_THEME, Options(), 0);
    return DrawAt(frame, view, ms).first.size();
  };
  // 300 ms is before the shine, which starts 180 + 200 ms in.
  const auto baseline = vertices(2, 300);
  Check(vertices(2, 730) > baseline, "gold shines 200 ms after fading in");
  Check(vertices(1, 730) > baseline, "platinum shines");
  Check(vertices(2, 1200) == baseline, "the shine has finished after 700 ms");
  Check(vertices(3, 730) == baseline, "silver never shines");
  Check(vertices(4, 730) == baseline, "bronze never shines");
}

// With reduced motion a toast that would shine doesn't, as its count stays at the baseline where the
// same toast without it has more, and it doesn't slide, as its top stays put while the same toast
// without it moves. The same toast is drawn either way, as without the shine it matches the one
// without reduced motion, so an empty toast can't pass.
void TestReducedMotionHasNoShineOrSlide() {
  constexpr double EARLY_MS = 50.0;
  constexpr double BEFORE_SHINE_MS = 300.0;
  constexpr double DURING_SHINE_MS = 730.0;
  constexpr double SETTLED_MS = 1500.0;
  const auto draw = [](bool reduced, double ms) {
    Frame frame;
    ImageStore store(4);
    TrophyToastView view(store);
    TrophyToastOptions options = Options();
    options.reduced_motion = reduced;
    ShowAt(frame, view, {"Trophy", 2, Png(64, 64)}, "Trophy earned!", DEFAULT_THEME, options, 0);
    return DrawAt(frame, view, ms).first;
  };
  Check(draw(false, DURING_SHINE_MS).size() > draw(false, BEFORE_SHINE_MS).size(), "without it, gold shines");
  Check(draw(true, DURING_SHINE_MS).size() == draw(false, BEFORE_SHINE_MS).size(), "with it, the same toast is drawn with no shine");
  Check(CardTop(draw(false, EARLY_MS)) < CardTop(draw(false, SETTLED_MS)) - 1.0f, "without it, the toast slides in");
  Check(Near(CardTop(draw(true, EARLY_MS)), CardTop(draw(true, SETTLED_MS))), "with it, the toast only fades");
}

// The shine is timed by the tokens the toast was shown with, so it stays in step with the
// entrance even when a frame is drawn with another theme. Here the shine is delayed to 780 ms by
// the options' tokens, while the frames use the default tokens, whose shine would already be
// sweeping at 730 ms.
void TestShineFollowsShownTheme() {
  TrophyToastOptions late = Options();
  late.tokens = TokenPatch<ToastTokens> {}.With<&ToastTokens::shine_delay_ms>(600.0f);
  const auto vertices = [&](double ms) {
    Frame frame;
    ImageStore store(4);
    TrophyToastView view(store);
    ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, late, 0);
    return DrawAt(frame, view, ms).first.size();
  };
  const auto baseline = vertices(300);
  Check(vertices(730) == baseline, "the shown theme's later shine hasn't started");
  Check(vertices(1000) > baseline, "the shown theme's shine sweeps after its own delay");
}

// The toast is built from the theme each frame is drawn with, even at the same output and with the
// same options. A theme with larger large corners, one of the scales the toast's tokens are built
// from, draws other vertices than the default theme at the same scale, and drawing the default
// theme again gives the first drawing back, so the difference is the theme's.
void TestThemeChange() {
  Frame frame;
  ImageStore store(4);
  TrophyToastView view(store);
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 0);
  constexpr double SETTLED_MS = 1100.0;
  const auto same = [](const std::vector<ImDrawVert> &a, const std::vector<ImDrawVert> &b) {
    if (a.size() != b.size()) {
      return false;
    }
    for (size_t index = 0; index < a.size(); ++index) {
      if (a[index].pos.x != b[index].pos.x || a[index].pos.y != b[index].pos.y || a[index].col != b[index].col) {
        return false;
      }
    }
    return true;
  };
  const auto first = DrawAt(frame, view, SETTLED_MS).first;
  Theme rounder = DEFAULT_THEME;
  rounder.radius.lg = 40.0f;
  const UiContext context{rounder, ScaleContext(DEFAULT_THEME.scale, 1920, 1080), frame.fonts};
  const auto changed = DrawAt(frame, view, SETTLED_MS, context).first;
  const auto again = DrawAt(frame, view, SETTLED_MS).first;
  Check(!first.empty() && same(first, again), "the default theme draws the same toast each time");
  Check(!same(first, changed), "a theme with other corners draws another toast at the same scale");
}

// The layout follows the output. At 4K the card is twice as wide and sits twice as far in.
void TestScaleChange() {
  Frame frame;
  ImageStore store(4);
  TrophyToastView view(store);
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 0);
  DrawAt(frame, view, 1000);
  const auto uhd = frame.Context(3840, 2160);
  const auto card = Bounds(DrawAt(frame, view, 1100, uhd).first, ToImU32(DEFAULT_THEME.colors.surface.raised));
  Check(Near(card.width, 1120.0f) && Near(card.x, 3840.0f - 64.0f - 1120.0f), "at 4K the card doubles");
}

// The layout also follows the width the toast may use. At the minimum scale, an 800 pixel wide
// output caps the toast at 40% of its content frame, while a 1200 pixel one leaves it at its own
// 373 pixel width, though both have the same scale.
void TestWidthChange() {
  Frame frame;
  ImageStore store(4);
  TrophyToastView view(store);
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 0);
  const auto raised = ToImU32(DEFAULT_THEME.colors.surface.raised);
  const auto narrow = frame.Context(800, 1080);
  const auto wide = frame.Context(1200, 1080);
  Check(narrow.scale.Scale() == wide.scale.Scale(), "both outputs have the same scale");
  const float narrow_width = Bounds(DrawAt(frame, view, 1000, narrow).first, raised).width;
  const float wide_width = Bounds(DrawAt(frame, view, 1100, wide).first, raised).width;
  Check(narrow_width < wide_width - 1.0f, "a wider output rebuilds the layout at its own width");
}

// The layout follows the scale even when the width it may use doesn't change. In fullscreen at 4K
// the 3.5% safe area, not the scaled edge distance, sets the content frame, so raising the user
// scale from 1 to 1.1 grows the toast from 1120 to 1232 pixels on an identical frame. At 1.1 the
// scaled edge distance, 70.4 pixels, is still under 3.5% of the height, 75.6.
void TestUserScaleChange() {
  Frame frame;
  ImageStore store(4);
  TrophyToastView view(store);
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 0);
  const auto at = [&](float user_scale) {
    return UiContext{DEFAULT_THEME,
                     ScaleContext(DEFAULT_THEME.scale, 3840, 2160, {.user_scale = user_scale, .safe_area_percent = 3.5f}),
                     frame.fonts};
  };
  const auto normal = at(1.0f);
  const auto larger = at(1.1f);
  Check(normal.scale.ContentFrame() == larger.scale.ContentFrame(), "both scales leave the same content frame");
  const auto raised = ToImU32(DEFAULT_THEME.colors.surface.raised);
  Check(Near(Bounds(DrawAt(frame, view, 1000, normal).first, raised).width, 1120.0f), "at the normal scale it is 1120");
  Check(Near(Bounds(DrawAt(frame, view, 1100, larger).first, raised).width, 1232.0f), "a larger scale rebuilds it at 1232");
}

// The clip of the first draw command holding a vertex of colour `rgb`, alpha ignored.
std::optional<ImVec4> ClipOfColour(const ImDrawList &list, ImU32 rgb) {
  for (const ImDrawCmd &command : list.CmdBuffer) {
    for (unsigned i = 0; i < command.ElemCount; i++) {
      const ImDrawVert &vertex = list.VtxBuffer[command.VtxOffset + list.IdxBuffer[command.IdxOffset + i]];
      if ((vertex.col & SystemUiTest::RGB_MASK) == (rgb & SystemUiTest::RGB_MASK)) {
        return command.ClipRect;
      }
    }
  }
  return std::nullopt;
}

// A trophy's glow spills past the card, so it is drawn outside the card's clip. The glow is the first
// gold shape drawn, before the cup, and the cup, which is under the clip, is the baseline.
void TestGlowEscapesTheCard() {
  const Color gold = DEFAULT_THEME.colors.trophy.gold;
  Frame frame;
  ImageStore store(4);
  TrophyToastView view(store);
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, {SECONDS}, 0);
  DrawAt(frame, view, 300);
  const auto glow = ClipOfColour(frame.List(), ToImU32(gold));
  Check(glow.has_value(), "the glow is drawn");
  const float inset = 16.0f * 0.29289322f;
  Check(!Near(glow->x, Resting(0).x + inset), "it is not clipped to the card");
  const auto &commands = frame.List().CmdBuffer;
  const bool any_card_clip = std::any_of(commands.begin(), commands.end(), [&](const ImDrawCmd &command) {
    return Near(command.ClipRect.x, Resting(0).x + inset);
  });
  Check(any_card_clip, "though the card's clip is there for the rest of the content");
}

// A toast that replaces another, even one of the same grade, starts its own entrance, since each is
// keyed by a number of its own and the store keeps its start under that key. The first toast is in
// the middle of its hold when the second is shown, so a second that shared the first's key would
// carry on from the hold and be fully visible at once, where its own entrance has the card still
// fading in. The first toast is the baseline, which is fully visible in the hold.
void TestReplacementStartsItsOwnEntrance() {
  Frame frame;
  ImageStore store(4);
  TrophyToastView view(store);
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 0);
  const auto raised = DEFAULT_THEME.colors.surface.raised;
  Check(HasColor(DrawAt(frame, view, 1500).first, raised), "the first toast's card is visible in its hold");
  ShowAt(frame, view, Gold(), "Trophy earned!", DEFAULT_THEME, Options(), 1500);
  const auto entering = DrawAt(frame, view, 1540).first;
  Check(!entering.empty() && !HasColor(entering, raised), "the toast that replaced it is still fading in");
  Check(HasColor(DrawAt(frame, view, 2400).first, raised), "and is visible once its entrance is over");
}

// The multiplier of the 64 bit FNV hash that tells two pictures apart.
constexpr uint64_t HASH_PRIME = 1099511628211ull;

constexpr double FRAME_MS = 1000.0 / 60.0;
constexpr double PROOF_END_MS = SECONDS * 1000.0 + 100.0;
// The least number of frames in a run that must draw something, so a run of empty frames can't pass.
constexpr size_t LEAST_DRAWN_FRAMES = 150;
// The entrance and exit each take several frames, so each holds at least this many pictures when it
// moves. The exit shows the fewest, seven, as the toast has faded out before its 180 ms are over.
constexpr size_t LEAST_PICTURES = 5;
// When the entrance ends and the exit starts, for a toast of SECONDS.
constexpr double ENTRANCE_END_MS = 280.0;
constexpr double EXIT_START_MS = 3820.0;
struct ProofOutput {
  float width;
  float height;
};
constexpr ProofOutput PROOF_OUTPUTS[] = {{1280, 720}, {1920, 1080}, {3840, 2160}};
// A hold frame, in the middle of the toast's life.
constexpr double HOLD_MS = 1500.0;

// What one toast life drew. How many frames drew something, how many different pictures its entrance
// and exit held, and whether the view still asked for frames at the end.
struct ProofRun {
  size_t drawn = 0;
  size_t entrance_pictures = 0;
  size_t exit_pictures = 0;
  bool wants_frames_at_end = true;
  // Frames whose clips were checked, which are those that drew the card.
  size_t clipped = 0;
};

// Plays one toast at the output `output`, checking its clips on every frame of its entrance, hold
// and exit that drew the card, and that it asks for frames until its exit has begun.
ProofRun PlayToast(Frame &frame, const ProofOutput &output, const std::string &name, int32_t grade, bool withIcon) {
  ImageStore store(2);
  TrophyToastView view(store);
  view.Show({name, grade, withIcon ? Png(64, 64) : std::vector<std::byte>{}}, "Trophy earned!", DEFAULT_THEME, {SECONDS}, At(0));
  const UiContext context = frame.Context(output.width, output.height);
  // The store the view's animation is kept in, the same one for the whole life of the toast.
  InteractionStore interactions;
  ProofRun run;
  std::set<uint64_t> entrance_pictures;
  std::set<uint64_t> exit_pictures;
  for (double ms = 0; ms <= PROOF_END_MS; ms += FRAME_MS) {
    auto actual = std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData());
    actual->_ResetForNewFrame();
    actual->PushClipRectFullScreen();
    UiFrame ui;
    ui.Begin(context.theme, context.scale, context.fonts, interactions, At(ms));
    run.wants_frames_at_end = view.Draw(ui, *actual, context.scale.SafeArea());
    interactions.Collect();
    Check(run.wants_frames_at_end || ms >= EXIT_START_MS, "the view asks for frames until its exit");
    if (const auto card = SystemUiTest::DrawnCard(*actual)) {
      SystemUiTest::ToastClipExpect expect;
      expect.card = *card;
      expect.radius = std::min(context.scale.Px(ToastDefaults(DEFAULT_THEME).radius), card->height / 2.0f);
      expect.has_icon = withIcon;
      expect.icon_glow = grade == 2 ? DEFAULT_THEME.colors.trophy.gold : DEFAULT_THEME.colors.trophy.silver;
      // The shine sweeps for a gold or platinum trophy once the toast has faded in and waited. The
      // frames at the edges of the sweep draw none, so only the middle of it is asserted.
      const auto &toast = ToastDefaults(DEFAULT_THEME);
      const double sweeping = ms - (toast.enter_fade_ms + toast.shine_delay_ms);
      expect.shine = (grade == 1 || grade == 2) && sweeping > FRAME_MS && sweeping < DEFAULT_THEME.motion.shine - FRAME_MS;
      const char *fault = SystemUiTest::ToastClipFault(*actual, expect);
      Check(fault == nullptr, fault != nullptr ? fault : "");
      ++run.clipped;
    }
    run.drawn += actual->VtxBuffer.Size > 0 ? 1 : 0;
    uint64_t hash = actual->VtxBuffer.Size;
    for (const ImDrawVert &vertex : actual->VtxBuffer) {
      hash = hash * HASH_PRIME ^ static_cast<uint64_t>(vertex.col) ^ static_cast<uint64_t>(vertex.pos.y * 64.0f);
    }
    if (ms < ENTRANCE_END_MS) {
      entrance_pictures.insert(hash);
    } else if (ms >= EXIT_START_MS && actual->VtxBuffer.Size > 0) {
      exit_pictures.insert(hash);
    }
  }
  run.entrance_pictures = entrance_pictures.size();
  run.exit_pictures = exit_pictures.size();
  return run;
}

// A gold toast called `name` as drawn `ms` after it was shown. The toast starts when it is first
// drawn, so it is drawn in the frame it is shown, and then at `ms`.
std::unique_ptr<ImDrawList> PictureAt(Frame &frame, const std::string &name, double ms) {
  ImageStore store(2);
  TrophyToastView view(store);
  view.Show({name, 2, Png(64, 64)}, "Trophy earned!", DEFAULT_THEME, Options(), At(0));
  const UiContext context = frame.Context();
  InteractionStore interactions;
  {
    UiFrame first;
    first.Begin(context.theme, context.scale, context.fonts, interactions, At(0));
    ImDrawList discarded(ImGui::GetDrawListSharedData());
    discarded._ResetForNewFrame();
    discarded.PushClipRectFullScreen();
    view.Draw(first, discarded, context.scale.SafeArea());
    interactions.Collect();
  }
  UiFrame ui;
  ui.Begin(context.theme, context.scale, context.fonts, interactions, At(ms));
  auto list = std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData());
  list->_ResetForNewFrame();
  list->PushClipRectFullScreen();
  view.Draw(ui, *list, context.scale.SafeArea());
  return list;
}

// Every frame of the toast's life has the clips it should, at 720p, 1080p and 4K, with and without an
// icon, with a short and a long wrapped title, and for a gold trophy, which shines, and a silver one,
// which does not. The entrance and the exit each move, the hold draws, and the view asks for frames
// until it has gone, so a view that drew nothing, stood still or stopped early could not pass. The
// same name draws the same picture twice and another name draws something else, so the name is what
// tells them apart.
void TestViewPlaysItsWholeLife() {
  Frame frame;
  const std::string longName = SystemUiTest::LONG_TITLE;
  for (const ProofOutput &output : PROOF_OUTPUTS) {
    for (const bool withIcon : {false, true}) {
      for (const std::string &name : {std::string("Master Alchemist"), longName}) {
        for (const int32_t grade : {2, 3}) {
          const ProofRun run = PlayToast(frame, output, name, grade, withIcon);
          Check(run.drawn >= LEAST_DRAWN_FRAMES, "most frames of the run drew something");
          Check(run.clipped >= LEAST_DRAWN_FRAMES, "and the clips of most frames were checked");
          Check(run.entrance_pictures >= LEAST_PICTURES, "and the picture moved through the entrance");
          Check(run.exit_pictures >= LEAST_PICTURES, "and through the exit");
          Check(!run.wants_frames_at_end, "and the view asks for no frames once it has gone");
        }
      }
    }
  }
  const auto actual = PictureAt(frame, "Master Alchemist", HOLD_MS);
  Check(actual->VtxBuffer.Size > 0, "the toast draws at its hold");
  Check(SystemUiTest::NearGeometry(*actual, *PictureAt(frame, "Master Alchemist", HOLD_MS)), "the same name draws the same toast");
  Check(!SystemUiTest::NearGeometry(*actual, *PictureAt(frame, "Another Name", HOLD_MS)), "a toast with another name draws something else");
}

// A quarter of the way through its time, the shine is where the eased curve puts it and not a
// quarter of the way across. Both are the shine drawn over the same card, so only the curve differs.
void TestShineIsEased() {
  constexpr float SWEEP = 0.25f;
  Frame frame;
  const UiContext context = frame.Context();
  const auto &toast = ToastDefaults(DEFAULT_THEME);
  const double ms = toast.enter_fade_ms + toast.shine_delay_ms + SWEEP * DEFAULT_THEME.motion.shine;
  const auto actual = PictureAt(frame, "Master Alchemist", ms);
  const auto card = SystemUiTest::DrawnCard(*actual);
  Check(card.has_value(), "the card is drawn");
  const float rounding = std::min(context.scale.Px(toast.radius), card->height / 2.0f);
  const auto shine = [&](float progress) {
    auto list = std::make_unique<ImDrawList>(ImGui::GetDrawListSharedData());
    list->_ResetForNewFrame();
    list->PushClipRectFullScreen();
    DrawShine(*list, *card, rounding, progress, DEFAULT_THEME.shine, DEFAULT_THEME.colors.effect.shine, context.scale);
    return list;
  };
  Check(SystemUiTest::ContainsDrawing(*actual, *shine(ApplyEase(Ease::Standard, SWEEP))), "the shine follows the eased curve");
  Check(!SystemUiTest::ContainsDrawing(*actual, *shine(SWEEP)), "and not a straight line");
}

} // namespace

int main() {
  TestToastLife();
  TestToastLeavesTheFallbackWindowHidden();
  TestIconLifetime();
  TestGradeColours();
  TestShineTiming();
  TestShineFollowsShownTheme();
  TestReducedMotionHasNoShineOrSlide();
  TestShineIsEased();
  TestThemeChange();
  TestScaleChange();
  TestWidthChange();
  TestUserScaleChange();
  TestGlowEscapesTheCard();
  TestReplacementStartsItsOwnEntrance();
  TestViewPlaysItsWholeLife();
  std::puts("System UI views tests passed");
  return 0;
}
