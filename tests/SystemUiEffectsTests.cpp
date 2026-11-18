#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/effects/gradient.h"
#include "graphics/presentation/system/effects/halo.h"
#include "graphics/presentation/system/effects/highlight.h"
#include "graphics/presentation/system/effects/shine.h"
#include "graphics/presentation/system/imgui/convert.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numbers>
#include <optional>
#include <set>
#include <vector>

namespace {

using Libs::Graphics::SystemUi::Color;
using Libs::Graphics::SystemUi::DEFAULT_THEME;
using Libs::Graphics::SystemUi::DrawFocusRing;
using Libs::Graphics::SystemUi::DrawGlow;
using Libs::Graphics::SystemUi::DrawGlowOutside;
using Libs::Graphics::SystemUi::DrawSelection;
using Libs::Graphics::SystemUi::DrawShadow;
using Libs::Graphics::SystemUi::DrawShine;
using Libs::Graphics::SystemUi::DrawVerticalGradient;
using Libs::Graphics::SystemUi::Glow;
using Libs::Graphics::SystemUi::Lerp;
using Libs::Graphics::SystemUi::Rect;
using Libs::Graphics::SystemUi::Rgb;
using Libs::Graphics::SystemUi::ScaleContext;
using Libs::Graphics::SystemUi::ToImU32;

constexpr float TOLERANCE = 0.01f;
// ImGui pulls a stroke's bottom-right corner in by 0.49 rather than 0.5, so strokes measure
// 0.01 wider than their box.
constexpr float STROKE_SLACK = 0.02f;
constexpr Rect PANEL{100, 100, 200, 100};
constexpr Rect CARD{100, 100, 400, 100};
constexpr Color BLUE = Rgb(0x2391FF);
constexpr Color WHITE = Rgb(0xFFFFFF);

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiEffectsTests: failed: %s\n", message);
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

// One ImGui frame with no renderer, enough to record draw lists. Anti-aliasing is off unless
// asked for, so shapes have exact edges to measure.
class Frame {
public:
  explicit Frame(bool anti_aliased = false) {
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.DisplaySize = {3840, 2160};
    io.DeltaTime = 1.0f / 60.0f;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.IniFilename = nullptr; // keep the test from writing imgui.ini
    ImGui::GetStyle().AntiAliasedFill = anti_aliased;
    ImGui::GetStyle().AntiAliasedLines = anti_aliased;
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
  int IndexMark() { return List().IdxBuffer.Size; }

  // The vertices added since `mark`.
  std::vector<ImDrawVert> Since(int mark) {
    const auto &buffer = List().VtxBuffer;
    return {buffer.begin() + mark, buffer.end()};
  }

  // The centre of every triangle drawn since `index_mark`. A filled shape has vertices only on its
  // outline, so whether it covers a point shows in its triangles, not its vertices.
  std::vector<ImVec2> TriangleCentres(int index_mark) {
    const auto &indices = List().IdxBuffer;
    const auto &vertices = List().VtxBuffer;
    std::vector<ImVec2> centres;
    for (int i = index_mark; i + 2 < indices.Size; i += 3) {
      const auto &a = vertices[indices[i]].pos, &b = vertices[indices[i + 1]].pos, &c = vertices[indices[i + 2]].pos;
      centres.push_back({(a.x + b.x + c.x) / 3.0f, (a.y + b.y + c.y) / 3.0f});
    }
    return centres;
  }

  // The clip rectangles of the draw commands recorded so far.
  std::vector<Rect> ClipRects() {
    std::vector<Rect> clips;
    for (const auto &command : List().CmdBuffer) {
      clips.push_back({command.ClipRect.x, command.ClipRect.y, command.ClipRect.z - command.ClipRect.x,
                       command.ClipRect.w - command.ClipRect.y});
    }
    return clips;
  }
};

// The alpha byte of a packed colour.
int Alpha(ImU32 color) { return static_cast<int>((color >> IM_COL32_A_SHIFT) & 0xffu); }

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

// True when `point` lies inside `rect` with corners rounded by `radius`, by more than a hair. A
// shape drawn just outside the element follows its rounded corners, which lie inside the square
// box, so coverage has to be measured against the rounded shape.
bool InsideRounded(const Rect &rect, float radius, ImVec2 point) {
  constexpr float HAIR = 0.01f;
  const float nearest_x = std::clamp(point.x, rect.x + radius, rect.Right() - radius);
  const float nearest_y = std::clamp(point.y, rect.y + radius, rect.Bottom() - radius);
  const float dx = point.x - nearest_x, dy = point.y - nearest_y;
  const bool in_box = point.x > rect.x + HAIR && point.x < rect.Right() - HAIR && point.y > rect.y + HAIR &&
                      point.y < rect.Bottom() - HAIR;
  return in_box && std::sqrt(dx * dx + dy * dy) < radius - HAIR;
}

// True when any triangle drawn since `index_mark` covers the rounded panel.
bool CoversPanel(Frame &frame, int index_mark) {
  const auto centres = frame.TriangleCentres(index_mark);
  return std::any_of(centres.begin(), centres.end(), [](ImVec2 centre) { return InsideRounded(PANEL, 12, centre); });
}

// How far `point` lies outside `rect` with corners rounded by `radius`, or 0 inside it.
float DistanceOutside(const Rect &rect, float radius, ImVec2 point) {
  const float nearest_x = std::clamp(point.x, rect.x + radius, rect.Right() - radius);
  const float nearest_y = std::clamp(point.y, rect.y + radius, rect.Bottom() - radius);
  const float dx = point.x - nearest_x, dy = point.y - nearest_y;
  return std::max(0.0f, std::sqrt(dx * dx + dy * dy) - radius);
}

// The total area of the triangles drawn since `index_mark`. A halo drawn once covers its outer shape
// exactly once, so its triangles add up to that shape's area. Stacked or doubled shapes add up to
// more, which is what shows as steps, and a gap adds up to less.
float TriangleArea(Frame &frame, int index_mark) {
  const auto &indices = frame.List().IdxBuffer;
  const auto &vertices = frame.List().VtxBuffer;
  float area = 0.0f;
  for (int i = index_mark; i + 2 < indices.Size; i += 3) {
    const auto &a = vertices[indices[i]].pos, &b = vertices[indices[i + 1]].pos, &c = vertices[indices[i + 2]].pos;
    area += std::fabs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2.0f;
  }
  return area;
}

// The area of `rect` with corners rounded by `radius`.
float RoundedArea(const Rect &rect, float radius) {
  return rect.width * rect.height - (4.0f - std::numbers::pi_v<float>) * radius * radius;
}

// What a halo should be, in output pixels: `base` is the element's rectangle with any shadow's drop.
struct HaloExpectation {
  Rect base;
  float radius;
  float spread;
  float peak;
};

// Checks the fade itself. Each vertex's alpha is fixed by its distance from the element, full at the
// element, none at the outer edge, and easing flat into both ends. The halo takes many levels of
// alpha on the way, and its triangles cover its outer shape exactly once. A stack of flat shapes,
// which is what used to show as rings, fails every part of this.
int CheckHaloFade(Frame &frame, int mark, int index_mark, const HaloExpectation &halo) {
  const auto vertices = frame.Since(mark);
  Check(!vertices.empty(), "the halo is drawn");
  std::set<int> levels;
  for (const auto &vertex : vertices) {
    const float t = std::min(DistanceOutside(halo.base, halo.radius, vertex.pos) / halo.spread, 1.0f);
    const float expected = 255.0f * halo.peak * (1.0f - t * t * (3.0f - 2.0f * t));
    Check(std::fabs(static_cast<float>(Alpha(vertex.col)) - expected) <= 2.0f, "alpha follows the distance");
    levels.insert(Alpha(vertex.col));
  }
  Check(levels.count(0) == 1, "it fades to nothing at the outer edge");
  Check(levels.count(static_cast<int>(255.0f * halo.peak + 0.5f)) == 1, "and reaches its peak at the element");
  // Enough for the straight segments that stand in for round corners, and less than a whole band or edge.
  constexpr float AREA_SLACK = 0.003f;
  const Rect outer{halo.base.x - halo.spread, halo.base.y - halo.spread, halo.base.width + 2.0f * halo.spread,
                   halo.base.height + 2.0f * halo.spread};
  const float shape = RoundedArea(outer, halo.radius + halo.spread);
  Check(std::fabs(TriangleArea(frame, index_mark) / shape - 1.0f) <= AREA_SLACK, "it covers its shape exactly once");
  return static_cast<int>(levels.size());
}

// The 1080p scale most tests draw at.
ScaleContext FullHd() { return ScaleContext(DEFAULT_THEME.scale, 1920, 1080); }

// A level 3 shadow spreads 28 pixels around the panel and sits 10 pixels lower, fading from its 0.45
// peak. Level 0 and zero opacity are the baseline that draws nothing.
void TestShadow() {
  Frame frame;
  const auto shadow = DEFAULT_THEME.colors.effect.shadow;
  const auto &elevation = DEFAULT_THEME.elevation;
  int mark = frame.Mark();
  int index_mark = frame.IndexMark();
  DrawShadow(frame.List(), PANEL, 16, elevation.level0, shadow, FullHd());
  DrawShadow(frame.List(), PANEL, 16, elevation.level3, shadow, FullHd(), 0.0f);
  Check(frame.Since(mark).empty(), "level 0 and zero opacity draw nothing");
  DrawShadow(frame.List(), PANEL, 16, elevation.level3, shadow, FullHd());
  Check(Near(Bounds(frame.Since(mark)), {72, 82, 256, 156}), "the shadow spreads 28 pixels and drops 10");
  (void)CheckHaloFade(frame, mark, index_mark, {{100, 110, 200, 100}, 16, 28, 0.45f});
}

// At 4K the shadow spreads and drops twice as far, with more bands than at 1080p, so it follows its
// curve as closely. Opacity fades the shadow, not its size.
void TestShadowScalesAndFades() {
  Frame frame;
  const ScaleContext ultra_hd(DEFAULT_THEME.scale, 3840, 2160);
  int mark = frame.Mark();
  int index_mark = frame.IndexMark();
  DrawShadow(frame.List(), PANEL, 16, DEFAULT_THEME.elevation.level3, DEFAULT_THEME.colors.effect.shadow, ultra_hd,
             0.5f);
  Check(Near(Bounds(frame.Since(mark)), {44, 64, 312, 212}), "at 4K the shadow spreads 56 pixels and drops 20");
  const int ultra_levels = CheckHaloFade(frame, mark, index_mark, {{100, 120, 200, 100}, 16, 56, 0.225f});
  mark = frame.Mark();
  index_mark = frame.IndexMark();
  DrawShadow(frame.List(), PANEL, 16, DEFAULT_THEME.elevation.level3, DEFAULT_THEME.colors.effect.shadow, FullHd(),
             0.5f);
  const int full_levels = CheckHaloFade(frame, mark, index_mark, {{100, 110, 200, 100}, 16, 28, 0.225f});
  Check(ultra_levels > full_levels, "the wider halo at 4K has more bands");
}

// A glow spreads evenly on all sides without dropping, in its own colour, and opacity fades it.
// A filled glow covers the element, which is the baseline proving the coverage check can see a
// filled shape before it is used to show the outside glow and focus ring don't cover.
void TestGlow() {
  Frame frame;
  const auto &glow = DEFAULT_THEME.glow.focus;
  int mark = frame.Mark();
  int index_mark = frame.IndexMark();
  DrawGlow(frame.List(), PANEL, 12, glow, BLUE, FullHd());
  const auto vertices = frame.Since(mark);
  Check(Near(Bounds(vertices), {88, 88, 224, 124}), "the focus glow spreads 12 pixels on every side");
  (void)CheckHaloFade(frame, mark, index_mark, {PANEL, 12, 12, 0.35f});
  Check((vertices.front().col & 0x00ffffffu) == (ToImU32(BLUE) & 0x00ffffffu), "the glow keeps its colour");
  Check(CoversPanel(frame, index_mark), "the filled glow covers the element, as the coverage check can see");
  mark = frame.Mark();
  index_mark = frame.IndexMark();
  DrawGlow(frame.List(), PANEL, 12, glow, BLUE, FullHd(), 0.5f);
  (void)CheckHaloFade(frame, mark, index_mark, {PANEL, 12, 12, 0.175f});
}

// Corners rounder than the element can hold are limited to it, as ImGui does for a filled shape, and
// a square element still gets round outer corners. An empty element and an unusable spread are safe.
void TestHaloShapes() {
  Frame frame;
  const Glow wide{20.0f, 0.5f};
  int mark = frame.Mark();
  int index_mark = frame.IndexMark();
  DrawGlow(frame.List(), PANEL, 500, wide, BLUE, FullHd());
  (void)CheckHaloFade(frame, mark, index_mark, {PANEL, 50, 20, 0.5f});
  mark = frame.Mark();
  index_mark = frame.IndexMark();
  DrawGlow(frame.List(), PANEL, 0, wide, BLUE, FullHd());
  (void)CheckHaloFade(frame, mark, index_mark, {PANEL, 0, 20, 0.5f});
  mark = frame.Mark();
  DrawGlow(frame.List(), {100, 100, 0, 0}, 0, wide, BLUE, FullHd());
  Check(!frame.Since(mark).empty(), "an empty element still has a halo around its point");
  mark = frame.Mark();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float infinity = std::numeric_limits<float>::infinity();
  DrawGlow(frame.List(), PANEL, 12, Glow{nan, 0.5f}, BLUE, FullHd());
  DrawGlow(frame.List(), PANEL, 12, Glow{infinity, 0.5f}, BLUE, FullHd());
  DrawGlow(frame.List(), PANEL, 12, Glow{0.0f, 0.5f}, BLUE, FullHd());
  Check(frame.Since(mark).empty(), "a spread that is zero, infinite or not a number draws nothing");
  // A NaN corner would otherwise reach the point count and make ImGui reserve a negative size, which
  // cuts indices off whatever was drawn before. So it must add a whole halo with square corners.
  mark = frame.Mark();
  index_mark = frame.IndexMark();
  DrawGlow(frame.List(), PANEL, nan, wide, BLUE, FullHd());
  Check(frame.List().IdxBuffer.Size > index_mark, "a corner that is not a number still adds to the list");
  (void)CheckHaloFade(frame, mark, index_mark, {PANEL, 0, 20, 0.5f});
  mark = frame.Mark();
  index_mark = frame.IndexMark();
  DrawGlow(frame.List(), {100, 100, -40, 50}, 12, wide, BLUE, FullHd());
  Check(frame.List().IdxBuffer.Size > index_mark && !frame.Since(mark).empty(), "a negative size is safe too");
}

// The outside glow is eight 1.5 pixel rings from the element's edge outward, fading
// quadratically, so it reads as a halo rather than stripes. It never covers the element.
void TestGlowOutside() {
  Frame frame;
  const int mark = frame.Mark();
  const int index_mark = frame.IndexMark();
  DrawGlowOutside(frame.List(), PANEL, 12, DEFAULT_THEME.glow.focus, BLUE, FullHd());
  const auto vertices = frame.Since(mark);
  for (int ring = 0; ring < 8; ring++) {
    const float remaining = 1.0f - static_cast<float>(ring) / 8.0f;
    const auto color = BLUE.MulAlpha(0.35f * remaining * remaining);
    const auto bounds = Bounds(vertices, ToImU32(color));
    Check(Near(bounds.x, 100.0f - 1.5f * static_cast<float>(ring + 1), STROKE_SLACK), "each ring is 1.5 pixels wide");
  }
  Check(!CoversPanel(frame, index_mark), "the outside glow never covers the element");
}

// Each vertex of the gradient is coloured by its height, alpha included, so a rounded corner
// part way down gets the colour of that height. Half opacity halves every alpha.
void TestGradient() {
  Frame frame;
  const Color top = Rgb(0xFF0000).WithAlpha(1.0f), bottom = Rgb(0x0000FF).WithAlpha(0.2f);
  for (const float opacity : {1.0f, 0.5f}) {
    const int mark = frame.Mark();
    DrawVerticalGradient(frame.List(), PANEL, 16, top, bottom, opacity);
    const auto vertices = frame.Since(mark);
    Check(!vertices.empty(), "the gradient is drawn");
    for (const auto &vertex : vertices) {
      const float t = (vertex.pos.y - PANEL.y) / PANEL.height;
      Check(vertex.col == ToImU32(Lerp(top, bottom, t).MulAlpha(opacity)), "each vertex has the colour of its height");
    }
  }
}

// With anti-aliasing on, the fill's soft fringe stays transparent, or the edges would turn hard,
// and the vertices on the outline still get the gradient.
void TestGradientKeepsFringe() {
  Frame frame(true);
  const Color top = Rgb(0xFF0000), bottom = Rgb(0x0000FF);
  const int mark = frame.Mark();
  DrawVerticalGradient(frame.List(), PANEL, 16, top, bottom);
  int fringe = 0, outline = 0;
  for (const auto &vertex : frame.Since(mark)) {
    // Anti-aliasing puts the outline half a pixel inside the rounded shape and the fringe half a
    // pixel outside it, including around the corners, which lie inside the square box.
    if (!InsideRounded(PANEL, 16, vertex.pos)) {
      fringe++;
      Check(Alpha(vertex.col) == 0, "the anti-aliasing fringe stays transparent");
    } else {
      outline++;
      Check(vertex.col == ToImU32(Lerp(top, bottom, (vertex.pos.y - PANEL.y) / PANEL.height)),
            "outline vertices get the gradient with anti-aliasing on");
    }
  }
  Check(fringe > 0 && outline > 0, "anti-aliasing added a fringe and an outline to check");
}

// Halfway through, the shine's band sits in the middle of its sweep. It is 120 wide (30% of
// 400), leans 36.4 pixels over 100, and sweeps from fully off the left to fully off the right.
// It peaks at the theme's alpha with transparent edges and a bright plateau in the middle.
void TestShine() {
  Frame frame;
  const float skew = 100.0f * std::tan(20.0f * std::numbers::pi_v<float> / 180.0f);
  const float start = CARD.x - 120.0f - skew;
  const float left = start + (CARD.Right() - start) * 0.5f;
  const int mark = frame.Mark();
  DrawShine(frame.List(), CARD, 0, 0.5f, DEFAULT_THEME.shine, WHITE, FullHd());
  const auto vertices = frame.Since(mark);
  Check(Near(Bounds(vertices), {left, 100, 120 + skew, 100}), "the band is halfway across and leans right");
  const int peak = static_cast<int>(0.14f * 255.0f + 0.5f);
  int brightest = 0, faintest = 255;
  bool plateau_start = false, plateau_end = false;
  for (const auto &vertex : vertices) {
    brightest = std::max(brightest, Alpha(vertex.col));
    faintest = std::min(faintest, Alpha(vertex.col));
    // The middle 20% of the band, 24 of its 120 pixels, is the plateau, so along the bottom the
    // bright part runs from 48 to 72 pixels in.
    if (Alpha(vertex.col) == peak && Near(vertex.pos.y, CARD.Bottom())) {
      plateau_start |= Near(vertex.pos.x, left + 48.0f);
      plateau_end |= Near(vertex.pos.x, left + 72.0f);
    }
  }
  Check(brightest == peak && faintest == 0, "the band peaks at the theme's alpha and fades to nothing");
  Check(plateau_start && plateau_end, "the band has a bright plateau in the middle");
  const auto clips = frame.ClipRects();
  Check(std::any_of(clips.begin(), clips.end(), [](const Rect &clip) { return Near(clip, CARD); }),
        "with square corners the band is clipped to the element");
}

// The sweep starts fully off the left edge and ends fully off the right, and progress outside
// 0..1 or zero opacity draws nothing.
void TestShineEnds() {
  Frame frame;
  int mark = frame.Mark();
  DrawShine(frame.List(), CARD, 0, 0.0f, DEFAULT_THEME.shine, WHITE, FullHd());
  Check(Bounds(frame.Since(mark)).Right() <= CARD.x + TOLERANCE, "at the start the band is off the left edge");
  mark = frame.Mark();
  DrawShine(frame.List(), CARD, 0, 1.0f, DEFAULT_THEME.shine, WHITE, FullHd());
  Check(Bounds(frame.Since(mark)).x >= CARD.Right() - TOLERANCE, "at the end the band is off the right edge");
  mark = frame.Mark();
  for (const float outside : {-0.1f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
    DrawShine(frame.List(), CARD, 0, outside, DEFAULT_THEME.shine, WHITE, FullHd());
  }
  DrawShine(frame.List(), CARD, 0, 0.5f, DEFAULT_THEME.shine, WHITE, FullHd(), 0.0f);
  Check(frame.Since(mark).empty(), "progress outside 0..1 or zero opacity draws nothing");
}

// On a narrow card the band keeps its 80 pixel minimum instead of 30% of the width, opacity
// fades its peak, and with rounded corners it is clipped 0.29 of the radius in from each side,
// which puts the clip's corners on the rounded outline.
void TestShineOptions() {
  Frame frame;
  const Rect narrow{100, 100, 100, 100};
  const float skew = 100.0f * std::tan(20.0f * std::numbers::pi_v<float> / 180.0f);
  const int mark = frame.Mark();
  DrawShine(frame.List(), narrow, 0, 0.5f, DEFAULT_THEME.shine, WHITE, FullHd(), 0.5f);
  const auto vertices = frame.Since(mark);
  Check(Near(Bounds(vertices).width, 80 + skew), "a narrow card's band keeps its minimum width");
  int brightest = 0;
  for (const auto &vertex : vertices) {
    brightest = std::max(brightest, Alpha(vertex.col));
  }
  Check(brightest == WHITE.WithAlpha(0.14f).MulAlpha(0.5f).a, "half opacity halves the band's peak");
  DrawShine(frame.List(), CARD, 16, 0.5f, DEFAULT_THEME.shine, WHITE, FullHd());
  const float inset = 16.0f * (1.0f - std::numbers::sqrt2_v<float> / 2.0f);
  const Rect expected{CARD.x + inset, CARD.y + inset, CARD.width - 2 * inset, CARD.height - 2 * inset};
  const auto clips = frame.ClipRects();
  Check(std::any_of(clips.begin(), clips.end(), [&](const Rect &clip) { return Near(clip, expected); }),
        "rounded corners pull the clip in to the outline");
}

// The focus ring sits outside the element. The 3 pixel ring's inner edge is the element's edge,
// the 2 pixel dark keyline is outside it, and the glow is outside both, at full strength next to
// the keyline. None of them covers the element.
void TestFocusRing() {
  Frame frame;
  const auto &focus = DEFAULT_THEME.colors.focus;
  const int mark = frame.Mark();
  const int index_mark = frame.IndexMark();
  DrawFocusRing(frame.List(), PANEL, 12, DEFAULT_THEME, FullHd());
  const auto vertices = frame.Since(mark);
  Check(Near(Bounds(vertices, ToImU32(focus.ring)), {97, 97, 206, 106}, STROKE_SLACK),
        "the ring is 3 pixels just outside");
  Check(Near(Bounds(vertices, ToImU32(focus.keyline)), {95, 95, 210, 110}, STROKE_SLACK),
        "the keyline is 2 pixels outside the ring");
  Check(Near(Bounds(vertices), {83, 83, 234, 134}, STROKE_SLACK), "the glow spreads 12 pixels outside both");
  const auto inner_glow = Bounds(vertices, ToImU32(focus.glow.MulAlpha(0.35f)));
  Check(Near(inner_glow.x, 93.5f, STROKE_SLACK), "the glow starts at the focus peak right outside the keyline");
  Check(!CoversPanel(frame, index_mark), "nothing is drawn over the element");
}

// Opacity fades every part of the ring together.
void TestFocusRingFades() {
  Frame frame;
  const auto &focus = DEFAULT_THEME.colors.focus;
  const int mark = frame.Mark();
  DrawFocusRing(frame.List(), PANEL, 12, DEFAULT_THEME, FullHd(), 0.5f);
  const auto vertices = frame.Since(mark);
  Check(HasColor(vertices, focus.ring.MulAlpha(0.5f)), "the ring is drawn at half opacity");
  Check(HasColor(vertices, focus.keyline.MulAlpha(0.5f)), "the keyline is drawn at half opacity");
  Check(HasColor(vertices, focus.glow.MulAlpha(0.35f * 0.5f)), "the glow is drawn at half opacity");
}

// Selection fills the element faintly and marks its left edge with a 4 pixel bar that stops at
// the 12 pixel corners, so it never pokes past the rounded outline. Opacity fades both.
void TestSelection() {
  Frame frame;
  const auto &accent = DEFAULT_THEME.colors.accent;
  int mark = frame.Mark();
  DrawSelection(frame.List(), PANEL, 12, DEFAULT_THEME, FullHd());
  auto vertices = frame.Since(mark);
  Check(Near(Bounds(vertices, ToImU32(accent.tint)), PANEL), "the tint fills the element");
  Check(Near(Bounds(vertices, ToImU32(accent.line)), {100, 112, 4, 76}), "the bar runs between the corners");
  mark = frame.Mark();
  DrawSelection(frame.List(), PANEL, 12, DEFAULT_THEME, FullHd(), 0.5f);
  vertices = frame.Since(mark);
  Check(HasColor(vertices, accent.tint.MulAlpha(0.5f)) && HasColor(vertices, accent.line.MulAlpha(0.5f)),
        "selection fades with opacity");
}

// Effect colours in the theme are opaque, since glows and shadows take their strength from their
// tokens. A translucent colour would apply the alpha twice.
static_assert(DEFAULT_THEME.colors.focus.glow.a == 255 && DEFAULT_THEME.colors.accent.glow.a == 255,
              "glow colours are opaque");
static_assert(DEFAULT_THEME.colors.effect.shadow.a == 255, "the shadow colour is opaque");

// Colour helpers the effects rely on. MulAlpha scales the existing alpha rather than replacing
// it, and Lerp blends every channel, alpha included.
static_assert(Rgb(0x123456).WithAlpha(0.5f).MulAlpha(0.5f).a == 64, "MulAlpha scales the existing alpha");
static_assert(Rgb(0x123456).MulAlpha(2.0f).a == 255 && Rgb(0x123456).MulAlpha(-1.0f).a == 0, "the factor is clamped");
static_assert(Lerp(Color{0, 0, 0, 0}, Color{200, 100, 50, 255}, 0.5f) == Color{100, 50, 25, 128}, "every channel blends");
static_assert(Lerp(Color{0, 0, 0, 0}, Color{200, 100, 50, 255}, 2.0f) == Color{200, 100, 50, 255}, "t is clamped");

} // namespace

int main() {
  TestShadow();
  TestShadowScalesAndFades();
  TestGlow();
  TestHaloShapes();
  TestGlowOutside();
  TestGradient();
  TestGradientKeepsFringe();
  TestShine();
  TestShineEnds();
  TestShineOptions();
  TestFocusRing();
  TestFocusRingFades();
  TestSelection();
  std::puts("System UI effects tests passed");
  return 0;
}
