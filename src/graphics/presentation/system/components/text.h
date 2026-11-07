#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TEXT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TEXT_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/geometry.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

struct ImDrawList;
struct ImFont;

namespace Libs::Graphics::SystemUi {

enum class TextAlign : uint8_t { Start, Center, End };

// What happens to text wider than its space.
enum class TextOverflow : uint8_t {
	Ellipsis, // one line, cut short with an ellipsis
	Clip,     // one line, cut off at the edge
	Wrap,     // as many lines as needed, up to max_lines, the last cut short with an ellipsis
};

// What to show. The text is only viewed, so whoever builds the props must keep it alive until the
// text is drawn. Keep owned strings in the view, not in props kept across frames.
struct TextProps {
	std::string_view text;
	TypeStyle        style;
	Color            color;
	TextAlign        align    = TextAlign::Start;
	TextOverflow     overflow = TextOverflow::Ellipsis;
	// For wrapped text, the most lines to show. Zero means no limit. When lines are dropped, the
	// last one shown ends with an ellipsis.
	int max_lines = 0;
	// For one-line text, shrink the font down to 85% before cutting it short, as a title in a
	// fixed-width slot does.
	bool shrink_to_fit = false;
};

// Text broken into the lines it will be drawn as.
struct TextLayout {
	struct Line {
		std::string text;
		float       width = 0.0f;
	};
	std::vector<Line> lines;
	ImFont*           font        = nullptr;
	float             font_size   = 0.0f;
	float             line_height = 0.0f;
	// The width of the widest line by the height of all lines.
	Vec2 size;
};

// Everything LayoutText's result depends on apart from the text itself, as it resolves them, so two
// calls with an equal key and equal text bytes give the same layout. Colour and alignment aren't in
// it, as DrawText reads those from the props. Theme and scale reach it through what they resolve
// to.
struct TextLayoutKey {
	ImFont*      font        = nullptr;
	float        font_size   = 0.0f;
	float        line_height = 0.0f;
	// The width laid out in, with every width that means the same thing made equal.
	float width = 0.0f;
	// The density the font bakes at, which decides the glyphs measured.
	float        density       = 0.0f;
	int          max_lines     = 0;
	TextOverflow overflow      = TextOverflow::Ellipsis;
	bool         shrink_to_fit = false;

	friend bool operator==(const TextLayoutKey&, const TextLayoutKey&) = default;
};

/// The key of laying `props` out in `max_width` pixels. Call it on the present thread, like
/// LayoutText.
[[nodiscard]] TextLayoutKey MakeTextLayoutKey(const UiContext& context, const TextProps& props,
                                              float max_width);

/// Lays `props` out in `max_width` pixels. It picks the font and its size from the style, breaks
/// wrapped text into lines, and shortens text that doesn't fit, marking with an ellipsis any text
/// it drops. Measure with this, place the result, then draw the same layout. A slot with no width
/// shows nothing unless the text is clipped. Call it on the present thread only, since measuring
/// can add glyphs to ImGui's shared font atlas.
[[nodiscard]] TextLayout LayoutText(const UiContext& context, const TextProps& props,
                                    float max_width);

/// How wide `props` is as one line at its style's size, trailing spaces left off, without laying it
/// out, or 0 with no font. Call it on the present thread, like LayoutText.
[[nodiscard]] float MeasureTextLine(const UiContext& context, const TextProps& props);

/// Draws `layout` in `rect`, each line aligned as `props` says and centred in its line height.
/// Clipped text is clipped to `rect`. `opacity` fades it with its element. Glyphs go left to right
/// in the order of the text, with no shaping and no reordering, so Arabic has to be shaped before
/// it gets here or its letters come out unjoined and in the wrong order. The overlay does that for
/// its trophy subtitle, in PrepareTrophySubtitle, and a trophy's name from a game gets no such
/// help.
void DrawText(ImDrawList& list, const TextLayout& layout, const TextProps& props, const Rect& rect,
              float opacity = 1.0f);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TEXT_H_
