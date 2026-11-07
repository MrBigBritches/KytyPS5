#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/core/theme.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>

namespace {

using Libs::Graphics::SystemUi::Color;
using Libs::Graphics::SystemUi::DEFAULT_THEME;
using Libs::Graphics::SystemUi::Rect;
using Libs::Graphics::SystemUi::Rgb;
using Libs::Graphics::SystemUi::ScaleContext;
using Libs::Graphics::SystemUi::ScaleOptions;
using Libs::Graphics::SystemUi::TypeStyle;

constexpr auto &RULES = DEFAULT_THEME.scale;
constexpr float TOLERANCE = 0.01f;
constexpr float NOT_A_NUMBER = std::numeric_limits<float>::quiet_NaN();
constexpr float POSITIVE_INFINITY = std::numeric_limits<float>::infinity();

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiCoreTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two lengths agree to within rounding noise.
bool Near(float actual, float expected) { return std::fabs(actual - expected) <= TOLERANCE; }

// True when every edge of two rectangles agrees to within rounding noise.
bool Near(const Rect &actual, const Rect &expected) {
  return Near(actual.x, expected.x) && Near(actual.y, expected.y) && Near(actual.width, expected.width) &&
         Near(actual.height, expected.height);
}

// True when `inner` lies entirely inside `outer`.
bool Contains(const Rect &outer, const Rect &inner) {
  return inner.x >= outer.x && inner.y >= outer.y && inner.x + inner.width <= outer.x + outer.width &&
         inner.y + inner.height <= outer.y + outer.height;
}

// The scale the default theme gives an output of `width` by `height`.
float ScaleAt(float width, float height, ScaleOptions options = {}) {
  return ScaleContext(RULES, width, height, options).Scale();
}

// The common outputs land on the scales the design names. 8K would be 4 unclamped, and 640x360
// would be 1/3, so these two also show both clamps are applied.
void TestScaleAtCommonOutputs() {
  Check(Near(ScaleAt(1920, 1080), 1.0f), "1080p is the reference scale");
  Check(Near(ScaleAt(1280, 720), 2.0f / 3.0f), "720p scales by two thirds");
  Check(Near(ScaleAt(2560, 1440), 4.0f / 3.0f), "1440p scales by four thirds");
  Check(Near(ScaleAt(3840, 2160), 2.0f), "4K scales by two");
  Check(Near(ScaleAt(7680, 4320), 3.0f), "8K is clamped to the maximum scale");
  Check(Near(ScaleAt(640, 360), 2.0f / 3.0f), "a tiny output is clamped to the minimum scale");
}

// The smaller axis ratio wins. Taking the larger one would give 1.79 for the ultrawide and 1.0
// for the narrow window, so each case tells min from max.
void TestSmallerAxisWins() {
  Check(Near(ScaleAt(3440, 1440), 4.0f / 3.0f), "height drives an ultrawide output");
  Check(Near(ScaleAt(1440, 1080), 0.75f), "width drives a narrow window");
}

// The user scale multiplies the output scale, within its range. The baseline is the same output
// with the default user scale.
void TestUserScale() {
  Check(Near(ScaleAt(1920, 1080, {.user_scale = 1.0f}), 1.0f), "baseline without a user scale");
  Check(Near(ScaleAt(1920, 1080, {.user_scale = 1.25f}), 1.25f), "the user scale multiplies");
  Check(Near(ScaleAt(1920, 1080, {.user_scale = 2.0f}), 1.5f), "the user scale is capped");
  Check(Near(ScaleAt(1920, 1080, {.user_scale = 0.5f}), 0.8f), "the user scale has a floor");
}

// Lengths round to whole pixels, and a border never vanishes. At 720p a 0.7 pixel border scales
// to 0.47, which rounds to zero without the one pixel floor.
void TestPixelsAndBorders() {
  const ScaleContext hd(RULES, 1280, 720);
  Check(Near(hd.Px(22.0f), 15.0f), "22 reference pixels round to 15 at 720p");
  Check(Near(hd.Px(16.0f), 11.0f), "16 reference pixels round to 11 at 720p");
  Check(Near(hd.Border(3.0f), 2.0f), "a 3 pixel border scales to 2 at 720p");
  Check(Near(hd.Border(0.7f), 1.0f), "a border never scales below one pixel");
}

// Each style keeps its minimum size. The 1080p baseline shows the same styles at full size, and
// body text at 720p shows the floor leaves larger sizes alone.
void TestFontFloors() {
  const ScaleContext full(RULES, 1920, 1080);
  const ScaleContext hd(RULES, 1280, 720);
  const auto &type = DEFAULT_THEME.type;
  Check(Near(full.FontSize(type.dense_small), 13.0f), "baseline dense text at 1080p");
  Check(Near(hd.FontSize(type.dense_small), 11.0f), "dense text stays at 11 pixels at 720p");
  Check(Near(hd.FontSize(type.caption), 12.0f), "captions stay at 12 pixels at 720p");
  Check(Near(hd.FontSize(type.body), 15.0f), "body text above its floor just scales");
}

// With no percentage the inset is the minimum edge distance. With 3.5% it is the larger
// percentage, 67.2 by 37.8, snapped to whole pixels. A percentage above the maximum is capped.
void TestSafeArea() {
  const auto edge_only = ScaleContext(RULES, 1920, 1080).SafeArea();
  Check(Near(edge_only, {32, 32, 1856, 1016}), "the minimum edge distance applies with no percentage");
  const auto tv = ScaleContext(RULES, 1920, 1080, {.safe_area_percent = 3.5f}).SafeArea();
  Check(Near(tv, {67, 38, 1786, 1004}), "a 3.5% safe area insets by the percentage on whole pixels");
  const auto capped = ScaleContext(RULES, 1920, 1080, {.safe_area_percent = 50.0f}).SafeArea();
  Check(Near(capped, {192, 108, 1536, 864}), "the safe area percentage is capped at 10%");
}

// On an ultrawide output the side inset is a percentage of the 16:9 width, 2560, giving 89.6,
// not of the full 3440, which would give 120.4. The minimum edge distance scales too, to 42.67
// at a scale of 4/3, which snaps to 43.
void TestUltrawideSafeArea() {
  const auto tv = ScaleContext(RULES, 3440, 1440, {.safe_area_percent = 3.5f}).SafeArea();
  Check(Near(tv, {90, 50, 3260, 1340}), "ultrawide side insets use the 16:9 width");
  const auto edge_only = ScaleContext(RULES, 3440, 1440).SafeArea();
  Check(Near(edge_only, {43, 43, 3354, 1354}), "the edge distance scales");
}

// On an ultrawide output the content frame is the safe area narrowed to 16:9, 2407 pixels wide
// and centred on whole pixels. This is the baseline for the narrow cases, where the same rule
// must leave the safe area whole.
void TestContentFrame() {
  const ScaleContext wide(RULES, 3440, 1440);
  const auto frame = wide.ContentFrame();
  Check(Near(frame, {516, 43, 2407, 1354}), "an ultrawide frame is the 16:9 middle of the safe area");
  const auto area = wide.SafeArea();
  const float left = frame.x - area.x;
  const float right = area.x + area.width - (frame.x + frame.width);
  Check(std::fabs(left - right) <= 1.0f, "the ultrawide frame is centred to within a pixel");
  Check(Contains(area, frame), "the ultrawide frame lies inside the safe area");
}

// Narrower outputs keep their full height, so a 4:3 window or a portrait screen gets the whole
// safe area for menus instead of a letterboxed 16:9 strip.
void TestNarrowContentFrame() {
  const ScaleContext four_three(RULES, 1440, 1080);
  Check(Near(four_three.ContentFrame(), four_three.SafeArea()), "a 4:3 frame is the whole safe area");
  const ScaleContext portrait(RULES, 1080, 1440);
  Check(Near(portrait.ContentFrame(), portrait.SafeArea()), "a portrait frame is the whole safe area");
}

// A minimised window reports an empty output. The scale falls back to the minimum and every area
// is empty, rather than negative or NaN.
void TestEmptyOutput() {
  const ScaleContext empty(RULES, 0, 0, {.safe_area_percent = 3.5f});
  Check(Near(empty.Scale(), 2.0f / 3.0f), "an empty output uses the minimum scale");
  Check(empty.SafeArea().width == 0.0f && empty.SafeArea().height == 0.0f, "an empty safe area");
  Check(empty.ContentFrame().width == 0.0f && empty.ContentFrame().height == 0.0f, "an empty frame");
  const ScaleContext negative(RULES, -10, 50);
  Check(negative.Screen().width == 0.0f, "a negative size is empty");
}

// Values that are not finite fall back instead of spreading NaN. Each case is compared with the
// same output given the fallback value directly.
void TestNonFiniteInputs() {
  Check(Near(ScaleAt(1920, 1080, {.user_scale = NOT_A_NUMBER}), 1.0f), "a NaN user scale means 1");
  Check(Near(ScaleAt(1920, 1080, {.user_scale = POSITIVE_INFINITY}), 1.0f), "an infinite user scale means 1");
  const auto nan_percent = ScaleContext(RULES, 1920, 1080, {.safe_area_percent = NOT_A_NUMBER});
  Check(Near(nan_percent.SafeArea(), ScaleContext(RULES, 1920, 1080).SafeArea()), "a NaN percentage means 0");
  const ScaleContext nan_width(RULES, NOT_A_NUMBER, 1080);
  Check(Near(nan_width.Scale(), 2.0f / 3.0f), "a NaN width gives the minimum scale");
  Check(nan_width.Screen().width == 0.0f, "a NaN width is empty");
  const ScaleContext infinite_width(RULES, POSITIVE_INFINITY, 1080);
  Check(infinite_width.Screen().width == 0.0f, "an infinite width is empty");
}

// Colour helpers. The channels must not swap, and alpha is clamped and rounded. These are
// compile-time checks because an unclamped alpha overflows the byte, which is undefined
// behaviour that a runtime check could see pass by chance, while constant evaluation rejects it.
static_assert(Rgb(0x2391FF) == Color{0x23, 0x91, 0xFF, 0xFF}, "Rgb keeps the channel order");
static_assert(Rgb(0x2391FF).WithAlpha(0.5f).a == 128, "half alpha rounds to 128");
static_assert(Rgb(0x2391FF).WithAlpha(2.0f).a == 255, "alpha above one is clamped");
static_assert(Rgb(0x2391FF).WithAlpha(-1.0f).a == 0, "alpha below zero is clamped");
static_assert(Rgb(0x2391FF).WithAlpha(NOT_A_NUMBER).a == 0, "NaN alpha is transparent");

// Perceived brightness, enough to order the text levels.
constexpr int Brightness(const Color &color) {
  constexpr int RED_WEIGHT = 299, GREEN_WEIGHT = 587, BLUE_WEIGHT = 114;
  return RED_WEIGHT * color.r + GREEN_WEIGHT * color.g + BLUE_WEIGHT * color.b;
}

// True when every value is positive and larger than the one before it.
constexpr bool Increasing(std::initializer_list<float> values) {
  float previous = 0.0f;
  for (const float value : values) {
    if (value <= previous) {
      return false;
    }
    previous = value;
  }
  return true;
}

// True when every style's floor is positive and no larger than its size, and its line height
// fits its size.
constexpr bool FloorsFit(std::initializer_list<TypeStyle> styles) {
  for (const auto &style : styles) {
    if (style.min_size <= 0.0f || style.min_size > style.size || style.line < style.size) {
      return false;
    }
  }
  return true;
}

// Invariants a re-theme must keep. Each scale grows strictly, text levels fade in order, and
// every type style's floor is positive and no larger than its size.
constexpr auto &THEME = DEFAULT_THEME;
static_assert(Increasing({THEME.space.hair, THEME.space.s1, THEME.space.s2, THEME.space.s3, THEME.space.s4,
                          THEME.space.s5, THEME.space.s6, THEME.space.s7, THEME.space.s8, THEME.space.s9}),
              "the spacing scale grows strictly");
static_assert(Increasing({THEME.radius.xs, THEME.radius.sm, THEME.radius.md, THEME.radius.lg, THEME.radius.xl}),
              "the radius scale grows strictly");
static_assert(Increasing({THEME.border.hair, THEME.border.thin, THEME.border.thick, THEME.border.bar}),
              "the border widths grow strictly");
static_assert(Increasing({THEME.elevation.level1.blur, THEME.elevation.level2.blur, THEME.elevation.level3.blur,
                          THEME.elevation.level4.blur}),
              "higher elevations cast wider shadows");
static_assert(Brightness(THEME.colors.text.primary) > Brightness(THEME.colors.text.secondary) &&
                  Brightness(THEME.colors.text.secondary) > Brightness(THEME.colors.text.tertiary) &&
                  Brightness(THEME.colors.text.tertiary) > Brightness(THEME.colors.text.disabled),
              "text levels fade in order");
static_assert(FloorsFit({THEME.type.display, THEME.type.title, THEME.type.heading, THEME.type.body,
                         THEME.type.body_small, THEME.type.label, THEME.type.caption, THEME.type.dense,
                         THEME.type.dense_small}),
              "every type style has a positive floor no larger than its size");
static_assert(RULES.min_scale < 1.0f && RULES.max_scale > 1.0f && RULES.min_user_scale <= 1.0f &&
                  RULES.max_user_scale >= 1.0f,
              "the reference scale and the default user scale are inside their ranges");

} // namespace

int main() {
  TestScaleAtCommonOutputs();
  TestSmallerAxisWins();
  TestUserScale();
  TestPixelsAndBorders();
  TestFontFloors();
  TestSafeArea();
  TestUltrawideSafeArea();
  TestContentFrame();
  TestNarrowContentFrame();
  TestEmptyOutput();
  TestNonFiniteInputs();
  std::puts("System UI core tests passed");
  return 0;
}
