#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/imgui/fonts.h"

#include "imgui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

using Libs::Graphics::SystemUi::DEFAULT_THEME;
using Libs::Graphics::SystemUi::DrawText;
using Libs::Graphics::SystemUi::FontWeight;
using Libs::Graphics::SystemUi::LayoutText;
using Libs::Graphics::SystemUi::LoadUiFonts;
using Libs::Graphics::SystemUi::MeasureTextLine;
using Libs::Graphics::SystemUi::Rect;
using Libs::Graphics::SystemUi::Rgb;
using Libs::Graphics::SystemUi::ScaleContext;
using Libs::Graphics::SystemUi::TextAlign;
using Libs::Graphics::SystemUi::TextLayout;
using Libs::Graphics::SystemUi::TextOverflow;
using Libs::Graphics::SystemUi::TextProps;
using Libs::Graphics::SystemUi::ToImU32;
using Libs::Graphics::SystemUi::UiContext;
using Libs::Graphics::SystemUi::UiFonts;

constexpr const char *REGULAR_FONT = KYTY_TEST_FONT_REGULAR;
constexpr const char *BOLD_FONT = KYTY_TEST_FONT_BOLD;
constexpr const char *ELLIPSIS = "\xE2\x80\xA6";
constexpr const char *SENTENCE = "The quick brown fox jumps over the lazy dog";
constexpr float TOLERANCE = 0.01f;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiTextTests: failed: %s\n", message);
    std::abort();
  }
}

// One ImGui frame with no renderer and the theme's fonts loaded, enough to measure and record
// text. Anti-aliasing is off so glyph quads have exact positions.
class Frame {
public:
  explicit Frame(const char *regular = REGULAR_FONT, const char *bold = BOLD_FONT) {
    ImGui::CreateContext();
    auto &io = ImGui::GetIO();
    io.DisplaySize = {3840, 2160};
    io.DeltaTime = 1.0f / 60.0f;
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.IniFilename = nullptr;
    fonts = LoadUiFonts(*io.Fonts, regular, bold);
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

  UiFonts fonts;
};

// A context for the theme at `width` by `height` with `fonts`.
UiContext Context(const UiFonts &fonts, float width = 1920, float height = 1080) {
  return {DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, width, height), fonts};
}

// Body text of `text` with the other options left at their defaults.
TextProps Body(std::string_view text) { return {text, DEFAULT_THEME.type.body, Rgb(0xFFFFFF)}; }

// True when `text` ends with `suffix`.
bool EndsWith(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() && text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// True when `text` is well-formed UTF-8, so no multi-byte character was cut in half.
bool ValidUtf8(const std::string &text) {
  for (size_t i = 0; i < text.size();) {
    const auto lead = static_cast<unsigned char>(text[i]);
    const size_t length = lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;
    if ((lead & 0xC0) == 0x80 || i + length > text.size()) {
      return false;
    }
    for (size_t k = 1; k < length; k++) {
      if ((static_cast<unsigned char>(text[i + k]) & 0xC0) != 0x80) {
        return false;
      }
    }
    i += length;
  }
  return true;
}

// The smallest x and largest right edge of `vertices`.
std::pair<float, float> Span(const std::vector<ImDrawVert> &vertices) {
  float left = std::numeric_limits<float>::max(), right = -left;
  for (const auto &vertex : vertices) {
    left = std::min(left, vertex.pos.x);
    right = std::max(right, vertex.pos.x);
  }
  return {left, right};
}

// Both weights load. A missing bold leaves it null, and asking for bold then gives the regular
// font. The fallback hook runs once after each font that loads and never for one that doesn't.
void TestFontsLoad() {
  Frame frame;
  Check(frame.fonts.regular != nullptr && frame.fonts.bold != nullptr, "both weights load");
  Check(frame.fonts.regular != frame.fonts.bold, "the weights are different fonts");
  // The hook records how many fonts the atlas has when it runs. Merging after the font is added
  // sees one more than before, while merging too early would merge into the previous font.
  auto &atlas = *ImGui::GetIO().Fonts;
  const int before = atlas.Fonts.Size;
  std::vector<int> seen;
  const auto missing = LoadUiFonts(atlas, REGULAR_FONT, "missing-bold.ttf", [&] { seen.push_back(atlas.Fonts.Size); });
  Check(missing.bold == nullptr && missing.For(FontWeight::Bold) == missing.regular, "a missing bold uses regular");
  Check(seen == std::vector<int>{before + 1}, "fallbacks merge once, right after each font that loads");
}

// Each style picks its weight and pixel size. Bold label text is wider than regular body text of
// the same size, which shows the weight is really used. At 720p dense text keeps its 11 pixel
// floor.
void TestStyleChoosesFontAndSize() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const auto body = LayoutText(context, Body("Settings"), 1000);
  Check(body.font == frame.fonts.regular && body.font_size == 22.0f && body.line_height == 32.0f,
        "body text is regular, 22 on 32");
  TextProps label{"Settings", DEFAULT_THEME.type.label, Rgb(0xFFFFFF)};
  const auto bold = LayoutText(context, label, 1000);
  Check(bold.font == frame.fonts.bold && bold.size.x > body.size.x, "label text is bold and wider");
  TextProps dense{"60 fps", DEFAULT_THEME.type.dense_small, Rgb(0xFFFFFF)};
  Check(LayoutText(Context(frame.fonts, 1280, 720), dense, 1000).font_size == 11.0f,
        "dense text keeps its floor at 720p");
}

// Text that fits is left alone. Text that doesn't is cut at a character with the font's ellipsis,
// keeping as much of its start as fits.
void TestEllipsis() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const auto roomy = LayoutText(context, Body(SENTENCE), 2000);
  Check(roomy.lines.size() == 1 && roomy.lines[0].text == SENTENCE, "text that fits is unchanged");
  const auto cut = LayoutText(context, Body(SENTENCE), 150);
  const auto &shown = cut.lines[0].text;
  Check(cut.lines.size() == 1 && cut.lines[0].width <= 150.0f, "cut text fits its space");
  Check(EndsWith(shown, ELLIPSIS), "cut text ends with an ellipsis");
  const auto kept = shown.substr(0, shown.size() - std::string(ELLIPSIS).size());
  Check(!kept.empty() && std::string(SENTENCE).rfind(kept, 0) == 0, "cut text keeps the start of the original");
}

// A title that is a little too wide shrinks to fit without being cut. One far too wide shrinks to
// 85% and is then cut. Without shrinking, the slightly wide one is cut at full size.
void TestShrinkToFit() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const float full = LayoutText(context, Body(SENTENCE), 5000).size.x;
  auto title = Body(SENTENCE);
  title.shrink_to_fit = true;
  const auto snug = LayoutText(context, title, full * 0.95f);
  Check(snug.lines[0].text == SENTENCE && snug.font_size < 22.0f && snug.font_size >= 22.0f * 0.85f,
        "a slightly wide title shrinks instead of being cut");
  Check(snug.size.x <= full * 0.95f + 0.5f, "the shrunk title fits");
  const auto squeezed = LayoutText(context, title, full * 0.5f);
  Check(std::fabs(squeezed.font_size - 22.0f * 0.85f) < TOLERANCE && EndsWith(squeezed.lines[0].text, ELLIPSIS),
        "a far too wide title shrinks to 85% and is cut");
  // Shrinking measures at a fractional size, and rounding can leave the result a hair wider than
  // its space. A title that only needed shrinking must still never be cut, at any width.
  for (const char *text : {SENTENCE, "Trophy unlocked: Master of Alchemy"}) {
    auto sweep = Body(text);
    sweep.shrink_to_fit = true;
    const float natural = LayoutText(context, Body(text), 5000).size.x;
    for (int percent = 86; percent < 100; percent++) {
      const auto fitted = LayoutText(context, sweep, natural * static_cast<float>(percent) / 100.0f);
      Check(fitted.lines[0].text == text, "a title that only needs shrinking is never cut");
    }
  }
  // Text up to half a pixel wider than its space still fits, so a rounding error at a fractional
  // size never cuts it. Without that allowance, a quarter pixel too narrow would cut it short.
  const auto overhang = LayoutText(context, Body(SENTENCE), full - 0.25f);
  Check(overhang.lines[0].text == SENTENCE, "text a hair wider than its space still fits");
  const auto unshrunk = LayoutText(context, Body(SENTENCE), full * 0.95f);
  Check(unshrunk.font_size == 22.0f && EndsWith(unshrunk.lines[0].text, ELLIPSIS), "without shrinking it is cut");
}

// Wrapped text breaks at spaces into lines that fit, losing no words, and explicit line breaks
// start new lines, including empty ones.
void TestWrap() {
  Frame frame;
  const auto context = Context(frame.fonts);
  auto paragraph = Body(SENTENCE);
  paragraph.overflow = TextOverflow::Wrap;
  const auto wrapped = LayoutText(context, paragraph, 200);
  Check(wrapped.lines.size() > 1, "a long sentence wraps");
  std::string joined;
  for (const auto &line : wrapped.lines) {
    Check(line.width <= 200.0f, "every wrapped line fits");
    joined += (joined.empty() ? "" : " ") + line.text;
  }
  Check(joined == SENTENCE, "wrapping loses no words");
  Check(wrapped.size.y == 32.0f * static_cast<float>(wrapped.lines.size()), "the height is one line height per line");
  paragraph.text = "first\n\nthird";
  const auto breaks = LayoutText(context, paragraph, 2000);
  Check(breaks.lines.size() == 3 && breaks.lines[1].text.empty(), "line breaks start new lines, even empty ones");
}

// Past the line limit, the last line shown is cut short with an ellipsis.
void TestMaxLines() {
  Frame frame;
  const auto context = Context(frame.fonts);
  auto paragraph = Body(SENTENCE);
  paragraph.overflow = TextOverflow::Wrap;
  paragraph.max_lines = 2;
  const auto limited = LayoutText(context, paragraph, 150);
  Check(limited.lines.size() == 2, "only two lines are kept");
  Check(EndsWith(limited.lines[1].text, ELLIPSIS) && limited.lines[1].width <= 150.0f, "the last line is cut short");
}

// A word longer than a line is broken where it reaches the edge rather than overflowing.
void TestLongWordWraps() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const std::string word = "Supercalifragilisticexpialidocious";
  auto props = Body(word);
  props.overflow = TextOverflow::Wrap;
  const auto layout = LayoutText(context, props, 60);
  std::string joined;
  for (const auto &line : layout.lines) {
    Check(line.width <= 60.0f, "every piece of the long word fits");
    joined += line.text;
  }
  Check(layout.lines.size() > 1 && joined == word, "the long word is broken without losing letters");
}

// Cutting text short never splits a multi-byte character. Each width cuts at a different point,
// and dropping single bytes would leave half a character at some of them.
void TestEllipsisKeepsUtf8Whole() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const std::string accented = "\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9\xC3\xA9";
  const int full = static_cast<int>(LayoutText(context, Body(accented), 5000).size.x);
  Check(full > 31, "the accented text is wide enough to cut at many widths");
  for (int width = 30; width < full; width++) {
    const auto cut = LayoutText(context, Body(accented), static_cast<float>(width));
    Check(EndsWith(cut.lines[0].text, ELLIPSIS) && ValidUtf8(cut.lines[0].text), "the cut text is whole UTF-8");
  }
}

// A blank line that falls exactly on the line limit is cut short like any other, without
// reading outside the text.
void TestBlankLineAtLimit() {
  Frame frame;
  const auto context = Context(frame.fonts);
  auto props = Body("Title\n\nBody text");
  props.overflow = TextOverflow::Wrap;
  props.max_lines = 2;
  const auto layout = LayoutText(context, props, 2000);
  Check(layout.lines.size() == 2 && layout.lines[0].text == "Title", "the first line is kept");
  Check(layout.lines[1].text == ELLIPSIS, "the blank last line shows that text was dropped");
}

// The last line under a limit holds its own paragraph, not the ones after it, and is marked as
// cut even when that paragraph fits, since later paragraphs were dropped.
void TestDroppedParagraphsAreMarked() {
  Frame frame;
  const auto context = Context(frame.fonts);
  auto props = Body("One\nTwo\nThree");
  props.overflow = TextOverflow::Wrap;
  props.max_lines = 2;
  const auto layout = LayoutText(context, props, 2000);
  Check(layout.lines.size() == 2 && layout.lines[1].text == std::string("Two") + ELLIPSIS,
        "the second line is its own paragraph, marked as cut");
  const auto one_line = LayoutText(context, Body("Title\nSecond line"), 2000);
  Check(one_line.lines.size() == 1 && one_line.lines[0].text == std::string("Title") + ELLIPSIS,
        "one-line text shows that its later lines were dropped");
}

// With no room, ellipsis and wrapped text show nothing rather than spilling out of their slot.
// Clipped text keeps its whole line, which drawing clips.
void TestNoRoom() {
  Frame frame;
  const auto context = Context(frame.fonts);
  Check(LayoutText(context, Body(SENTENCE), 0).lines.empty(), "ellipsis text in no room shows nothing");
  auto wrapped = Body(SENTENCE);
  wrapped.overflow = TextOverflow::Wrap;
  Check(LayoutText(context, wrapped, 0).lines.empty(), "wrapped text in no room shows nothing");
  auto clipped = Body(SENTENCE);
  clipped.overflow = TextOverflow::Clip;
  Check(LayoutText(context, clipped, 0).lines.size() == 1, "clipped text keeps its line");
  Check(LayoutText(context, clipped, 80).lines[0].text == SENTENCE, "clipped text is never cut short");
}

// A font without an ellipsis character gets three dots instead. ImGui's built-in font, used when
// the theme's fonts are missing, has none.
void TestDotsWithoutEllipsisGlyph() {
  Frame frame("missing-regular.ttf", "missing-bold.ttf");
  const auto context = Context(frame.fonts);
  Check(!ImGui::GetFont()->IsGlyphInFont(0x2026), "the fallback font has no ellipsis character");
  const auto cut = LayoutText(context, Body(SENTENCE), 120);
  Check(EndsWith(cut.lines[0].text, "..."), "text is cut short with three dots");
}

// Glyphs sit in the middle of their line height. Body text is 22 on a 32 line, so its glyphs start
// 5 pixels lower than the same text in a style whose line is exactly its size.
void TestVerticalCentring() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const Rect rect{100, 100, 400, 40};
  auto body = Body("Hi");
  auto tight = body;
  tight.style.line = tight.style.size;
  float tops[2] = {};
  int index = 0;
  for (const auto *props : {&body, &tight}) {
    const int mark = frame.Mark();
    DrawText(frame.List(), LayoutText(context, *props, rect.width), *props, rect);
    float top = std::numeric_limits<float>::max();
    for (const auto &vertex : frame.Since(mark)) {
      top = std::min(top, vertex.pos.y);
    }
    tops[index++] = top;
  }
  Check(std::fabs(tops[0] - tops[1] - 5.0f) < TOLERANCE, "glyphs are centred in the line height");
}

// Alignment moves the line by the spare space, on whole pixels. Start is the baseline that the
// others are measured against.
void TestAlignment() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const Rect rect{100, 100, 400, 40};
  float lefts[3] = {};
  const TextAlign aligns[3] = {TextAlign::Start, TextAlign::Center, TextAlign::End};
  float width = 0;
  for (int i = 0; i < 3; i++) {
    auto props = Body("Hi");
    props.align = aligns[i];
    const auto layout = LayoutText(context, props, rect.width);
    width = layout.lines[0].width;
    const int mark = frame.Mark();
    DrawText(frame.List(), layout, props, rect);
    lefts[i] = Span(frame.Since(mark)).first;
  }
  const float spare = rect.width - width;
  Check(std::fabs((lefts[1] - lefts[0]) - std::floor(spare / 2.0f + 0.5f)) < TOLERANCE, "center moves by half the spare");
  Check(std::fabs((lefts[2] - lefts[0]) - std::floor(spare + 0.5f)) < TOLERANCE, "end moves by all of the spare");
}

// Opacity fades the glyphs, clipped text is clipped to its rectangle, and ellipsis text isn't.
void TestDrawOpacityAndClip() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const Rect rect{100, 100, 80, 40};
  auto props = Body(SENTENCE);
  props.color = Rgb(0x123456);
  const int mark = frame.Mark();
  DrawText(frame.List(), LayoutText(context, props, rect.width), props, rect, 0.5f);
  const auto vertices = frame.Since(mark);
  Check(!vertices.empty() && std::all_of(vertices.begin(), vertices.end(), [&](const ImDrawVert &v) {
          return v.col == ToImU32(props.color.MulAlpha(0.5f));
        }),
        "every glyph is drawn at half opacity");
  // Ellipsis text is already cut to fit, so it isn't clipped.
  for (const auto &command : frame.List().CmdBuffer) {
    Check(command.ClipRect.z != rect.Right(), "ellipsis text is not clipped");
  }
  const auto commands_before = frame.List().CmdBuffer.Size;
  props.overflow = TextOverflow::Clip;
  DrawText(frame.List(), LayoutText(context, props, rect.width), props, rect);
  bool clipped = false;
  for (const auto &command : frame.List().CmdBuffer) {
    clipped |= command.ClipRect.x == rect.x && command.ClipRect.z == rect.Right();
  }
  Check(clipped && frame.List().CmdBuffer.Size > commands_before, "clipped text is clipped to its rectangle");
}

// Empty text lays out to nothing and draws nothing. Without the theme's fonts, text still lays
// out in ImGui's own font.
void TestEmptyAndMissingFonts() {
  Frame frame("missing-regular.ttf", "missing-bold.ttf");
  const auto context = Context(frame.fonts);
  const auto empty = LayoutText(context, Body(""), 100);
  Check(empty.lines.empty() && empty.size.x == 0.0f && empty.size.y == 0.0f, "empty text has no size");
  const int mark = frame.Mark();
  DrawText(frame.List(), empty, Body(""), {0, 0, 100, 40});
  Check(frame.Since(mark).empty(), "empty text draws nothing");
  const auto fallback = LayoutText(context, Body("Hello"), 100);
  Check(std::fabs(MeasureTextLine(context, Body("Hello")) - fallback.size.x) <= TOLERANCE, "a line is measured in the font it falls back to");
  Check(fallback.font == ImGui::GetFont() && fallback.size.x > 0.0f, "missing fonts fall back to ImGui's font");
}

// A line measured without a layout is as wide as the layout of the same text says, in each weight's
// font and size. Trailing spaces don't count, and those inside the text do. With no text it is 0.
void TestMeasureTextLine() {
  Frame frame;
  const auto context = Context(frame.fonts);
  const auto &type = DEFAULT_THEME.type;
  for (const auto &style : {type.body, type.heading, type.caption}) {
    const auto props = [&](const char *text) { return TextProps{text, style, Rgb(0xFFFFFF)}; };
    const float saved = MeasureTextLine(context, props("Saved"));
    Check(saved > 0.0f, "a line has a width");
    Check(std::fabs(saved - LayoutText(context, props("Saved"), std::numeric_limits<float>::infinity()).size.x) <= TOLERANCE, "the same as its layout");
    Check(MeasureTextLine(context, props("Saved   ")) == saved, "trailing spaces don't count");
    Check(MeasureTextLine(context, props("Sa ved")) > saved, "spaces inside the text do");
    Check(MeasureTextLine(context, props("")) == 0.0f, "no text has no width");
    Check(MeasureTextLine(context, TextProps{{}, style, Rgb(0xFFFFFF)}) == 0.0f, "nor a label never given any, which has no data");
    Check(MeasureTextLine(context, props("   ")) == 0.0f, "and nor do spaces alone");
  }
}

} // namespace

int main() {
  TestMeasureTextLine();
  TestFontsLoad();
  TestStyleChoosesFontAndSize();
  TestEllipsis();
  TestShrinkToFit();
  TestWrap();
  TestMaxLines();
  TestLongWordWraps();
  TestEllipsisKeepsUtf8Whole();
  TestBlankLineAtLimit();
  TestDroppedParagraphsAreMarked();
  TestNoRoom();
  TestDotsWithoutEllipsisGlyph();
  TestVerticalCentring();
  TestAlignment();
  TestDrawOpacityAndClip();
  TestEmptyAndMissingFonts();
  std::puts("System UI text tests passed");
  return 0;
}
