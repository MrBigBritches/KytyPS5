#include "graphics/presentation/system/gallery/effectsPage.h"

#include "graphics/presentation/system/animation/easing.h"
#include "graphics/presentation/system/animation/tween.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/toggle.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/pageParts.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view RULE = "Raise elevation only for what floats. Glow marks an arrival or "
                                  "focus. Motion explains a change, "
                                  "and is never the only sign of it.";

// The Depth view, in reference pixels.
constexpr float LEVEL_CARD_WIDTH  = 240.0f;
constexpr float LEVEL_CARD_HEIGHT = 112.0f;
constexpr float GLOW_CARD_WIDTH   = 240.0f;
constexpr float GLOW_CARD_HEIGHT  = 96.0f;
// A mid grey, on which a shadow and the raised cards both show.
constexpr Color DEPTH_BACKDROP = Palette::GRAY_500;

// The Shine view. The live card rests between sweeps, so each can be seen as a separate event.
constexpr float LIVE_CARD_WIDTH  = 480.0f;
constexpr float LIVE_CARD_HEIGHT = 160.0f;
constexpr float HELD_CARD_WIDTH  = 220.0f;
constexpr float HELD_CARD_HEIGHT = 110.0f;
constexpr float SHINE_REST_MS    = 1300.0f;
// A sweep past its end, which draws nothing.
constexpr float NO_SWEEP = 2.0f;
// The shares of the sweep the held frames show. A sweep starts and ends with the band off the card,
// so these keep to the part where it can be seen.
constexpr std::array<float, 6> HELD_SHARES {0.25f, 0.35f, 0.45f, 0.55f, 0.65f, 0.75f};

// The Motion view.
constexpr float PLOT_WIDTH    = 240.0f;
constexpr float PLOT_HEIGHT   = 160.0f;
constexpr int   PLOT_DOTS     = 10;
constexpr float DOT_RADIUS    = 3.0f;
constexpr float MARKER_RADIUS = 6.0f;
constexpr int   DOT_SEGMENTS  = 12;
// A plot reaches past 1, so the Emphasis curve's overshoot shows.
constexpr float PLOT_TOP_VALUE = 1.2f;
constexpr float PLOT_LOOP_MS   = 1600.0f;
// A duration's bar is this long for each millisecond, so the longest fits the page.
constexpr float BAR_PER_MS       = 0.5f;
constexpr float BAR_HEIGHT       = 8.0f;
constexpr float BAR_NAME_WIDTH   = 120.0f;
constexpr float BAR_REST_MS      = 600.0f;
constexpr float BAR_COLUMN_WIDTH = 860.0f;
constexpr float PERCENT          = 100.0f;

// An elevation level and what it is for.
struct NamedLevel {
	std::string_view name;
	Elevation ElevationScale::* level;
	std::string_view            use;
};

constexpr std::array<NamedLevel, 5> LEVELS {
    NamedLevel {"Level 0", &ElevationScale::level0, "flat"},
    NamedLevel {"Level 1", &ElevationScale::level1, "cards in lists"},
    NamedLevel {"Level 2", &ElevationScale::level2, "a focused card"},
    NamedLevel {"Level 3", &ElevationScale::level3, "toasts, dropdowns"},
    NamedLevel {"Level 4", &ElevationScale::level4, "modals"}};

// A glow, its colour and what it is for.
struct NamedGlow {
	std::string_view name;
	Glow             glow;
	Color            color;
};

// An easing curve and its name.
struct NamedEase {
	std::string_view name;
	Ease             ease;
};

constexpr std::array<NamedEase, 5> EASES {
    NamedEase {"Linear", Ease::Linear}, NamedEase {"Enter", Ease::Enter},
    NamedEase {"Exit", Ease::Exit}, NamedEase {"Standard", Ease::Standard},
    NamedEase {"Emphasis", Ease::Emphasis}};

// A duration and its name.
struct NamedDuration {
	std::string_view name;
	float MotionDurations::* duration;
};

constexpr std::array<NamedDuration, 8> DURATIONS {
    NamedDuration {"fast", &MotionDurations::fast},
    NamedDuration {"base", &MotionDurations::base},
    NamedDuration {"slow", &MotionDurations::slow},
    NamedDuration {"slower", &MotionDurations::slower},
    NamedDuration {"emphasis", &MotionDurations::emphasis},
    NamedDuration {"shine", &MotionDurations::shine},
    NamedDuration {"exit_fade", &MotionDurations::exit_fade},
    NamedDuration {"pulse", &MotionDurations::pulse}};

// Text from a printf format, cut short if it runs past the limit, which is more than any caption.
template <class... Args>
std::string Formatted(const char* format, Args... args) {
	constexpr size_t          LONGEST = 160;
	std::array<char, LONGEST> text {};
	(void)std::snprintf(text.data(), text.size(), format, args...);
	return text.data();
}

// A card with a name and a detail inside it, in `style`.
Element NamedCard(UiFrame& frame, Style style, std::string_view name, const std::string& detail,
                  float width, float height) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 2> lines {
	    Text(frame, {name, theme.type.label, theme.colors.text.primary}),
	    Text(frame, {detail, theme.type.caption, theme.colors.text.secondary})};
	style.fill   = theme.colors.surface.raised;
	style.radius = Dp(theme.radius.md);
	return Box(frame,
	           {.direction = FlexDirection::Column,
	            .justify   = Justify::Center,
	            .gap       = Dp(theme.space.s1),
	            .padding   = DpInsets::All(Dp(theme.space.s3)),
	            .item      = {.width = Dp(width), .height = Dp(height)}},
	           style, lines);
}

// The elevation levels on a mid backdrop with room for the deepest shadow.
Element LevelRow(UiFrame& frame) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> cards;
	cards.reserve(LEVELS.size());
	for (const NamedLevel& named: LEVELS) {
		const Elevation&  level = theme.elevation.*named.level;
		const std::string detail =
		    Formatted("blur %.0f, y %.0f, %.2f", static_cast<double>(level.blur),
		              static_cast<double>(level.offset_y), static_cast<double>(level.peak_alpha));
		const std::string line = std::string(named.use) + ", " + detail;
		cards.push_back(NamedCard(frame, {.elevation = level}, named.name, line, LEVEL_CARD_WIDTH,
		                          LEVEL_CARD_HEIGHT));
	}
	return Box(frame,
	           {.gap     = Dp(theme.space.s5),
	            .padding = DpInsets::Symmetric(Dp(theme.space.s6), Dp(theme.space.s8))},
	           {.fill = DEPTH_BACKDROP}, cards);
}

// The glows, on the page's own dark ground where a glow shows best, and one with none to compare.
Element GlowRow(UiFrame& frame) {
	const Theme&                   theme = frame.ActiveTheme();
	const std::array<NamedGlow, 4> glows {
	    NamedGlow {"No glow", {}, {}},
	    NamedGlow {"Focus", theme.glow.focus, theme.colors.focus.glow},
	    NamedGlow {"Emphasis", theme.glow.emphasis, theme.colors.accent.glow},
	    NamedGlow {"Trophy", theme.glow.trophy, theme.colors.trophy.gold}};
	std::vector<Element> cards;
	cards.reserve(glows.size());
	for (const NamedGlow& named: glows) {
		const std::string detail =
		    Formatted("%.0f wide, %.2f at its peak", static_cast<double>(named.glow.radius),
		              static_cast<double>(named.glow.peak_alpha));
		const Element card =
		    NamedCard(frame, {}, named.name, detail, GLOW_CARD_WIDTH, GLOW_CARD_HEIGHT);
		Decor decor;
		if (named.glow.radius > 0.0f) {
			decor.glow = GlowEffect {named.glow, named.color};
		}
		cards.push_back(Decorate(frame, card, decor));
	}
	return Box(frame, {.gap = Dp(theme.space.s7), .padding = DpInsets::All(Dp(theme.space.s6))}, {},
	           cards);
}

// The Depth view.
Element DepthView(UiFrame& frame) {
	const std::array<Element, 4> rows {SectionHeading(frame, "Elevation"), LevelRow(frame),
	                                   SectionHeading(frame, "Glow"), GlowRow(frame)};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s3)},
	           {}, rows);
}

// A card with a shine `sweep` of the way across, and any text it holds.
Element ShineCard(UiFrame& frame, float width, float height, float sweep,
                  std::span<const Element> inside = {}) {
	const Theme&  theme = frame.ActiveTheme();
	const Element card =
	    Box(frame,
	        {.direction = FlexDirection::Column,
	         .justify   = Justify::Center,
	         .padding   = DpInsets::All(Dp(theme.space.s4)),
	         .item      = {.width = Dp(width), .height = Dp(height)}},
	        {.fill = theme.colors.surface.raised, .radius = Dp(theme.radius.md)}, inside);
	return Decorate(frame, card, {.shine = sweep});
}

// The frames held along the sweep, each saying when it is and how far the curve has come. The curve
// makes the steps uneven, which is what the frames are there to show. With reduced motion there is
// no sweep to show, so they stand empty and say so.
Element HeldFrames(UiFrame& frame, bool reduced) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> cells;
	cells.reserve(HELD_SHARES.size());
	for (const float share: HELD_SHARES) {
		const float       sweep = ApplyEase(Ease::Standard, share);
		const std::string line =
		    reduced ? std::string("No shine")
		            : Formatted("%.0f ms, at %.2f", static_cast<double>(share * theme.motion.shine),
		                        static_cast<double>(sweep));
		const std::array<Element, 2> parts {
		    ShineCard(frame, HELD_CARD_WIDTH, HELD_CARD_HEIGHT, reduced ? NO_SWEEP : sweep),
		    Text(frame, {line, theme.type.caption, theme.colors.text.secondary})};
		cells.push_back(
		    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, parts));
	}
	return Box(frame, {.gap = Dp(theme.space.s3)}, {}, cells);
}

// The toggle for reduced motion, and what it does on this view.
Element ReducedToggle(UiFrame& frame, GalleryState& state, std::string_view words) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 2> parts {
	    Toggle(frame, "reduced motion", &state.effects_reduced_motion),
	    Text(frame, {words, theme.type.body_small, theme.colors.text.secondary})};
	return Box(frame, {.align = Align::Center, .gap = Dp(theme.space.s3)}, {}, parts);
}

// The Shine view, a sweep that repeats, the frames along it and its tokens in words. Reduced motion
// has no shine at all, as it is only decoration. The toggle is last, so the rest stays where it is.
Element ShineView(UiFrame& frame, GalleryState& state) {
	const bool         reduced = state.effects_reduced_motion;
	const Theme&       theme   = frame.ActiveTheme();
	const ShineTokens& tokens  = theme.shine;
	const float        at      = LoopMs(frame.Now(), theme.motion.shine + SHINE_REST_MS);
	// Past the sweep's end draws nothing, which is the rest between sweeps.
	const float sweep =
	    at <= theme.motion.shine ? ApplyEase(Ease::Standard, at / theme.motion.shine) : NO_SWEEP;
	const std::array<Element, 1> inside {
	    Text(frame, {reduced ? "No shine with reduced motion" : "Sweeps once, then rests",
	                 theme.type.label, theme.colors.text.primary})};
	const std::string tokens_line = Formatted(
	    "%.0f ms on the Standard curve, leaning %.0f degrees, %.0f%% of the width and at least "
	    "%.0f, "
	    "%.2f at its brightest",
	    static_cast<double>(theme.motion.shine), static_cast<double>(tokens.angle_degrees),
	    static_cast<double>(tokens.width_fraction * PERCENT), static_cast<double>(tokens.min_width),
	    static_cast<double>(tokens.peak_alpha));
	const std::array<Element, 5> rows {
	    ShineCard(frame, LIVE_CARD_WIDTH, LIVE_CARD_HEIGHT, reduced ? NO_SWEEP : sweep, inside),
	    Text(frame, {tokens_line, theme.type.caption, theme.colors.text.secondary}),
	    SectionHeading(frame, "Along the sweep"), HeldFrames(frame, reduced),
	    ReducedToggle(frame, state, "Reduced motion, no shine")};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s3)}, {}, rows);
}

// The eased value `ease` gives at a share of the way along, as a point on a plot in `rect`.
ImVec2 PlotPoint(const Rect& rect, Ease ease, float share) {
	return {rect.x + rect.width * share,
	        rect.Bottom() - rect.height * ApplyEase(ease, share) / PLOT_TOP_VALUE};
}

// Draws one curve as dots at even steps of time, so bunching shows where it is slow, the lines for
// its start and end, and a marker running along it. Trivially copyable so Custom can store it.
struct PlotPaint {
	Ease  ease;
	float loop_share;

	void operator()(const CustomPaint& painting) const {
		const auto& colors = painting.frame.ActiveTheme().colors;
		const auto& scale  = painting.frame.Scale();
		const Rect& rect   = painting.painted;
		ImDrawList& list   = painting.list;
		const auto  packed = [&painting](Color color) {
            return ToImU32(color.MulAlpha(painting.opacity));
		};
		list.AddRectFilled(TopLeft(rect), BottomRight(rect), packed(colors.surface.inset));
		const float top_line = rect.Bottom() - rect.height / PLOT_TOP_VALUE;
		list.AddLine({rect.x, top_line}, {rect.Right(), top_line}, packed(colors.border.strong));
		for (int dot = 0; dot <= PLOT_DOTS; dot++) {
			const float share = static_cast<float>(dot) / PLOT_DOTS;
			list.AddCircleFilled(PlotPoint(rect, ease, share), scale.Px(DOT_RADIUS),
			                     packed(colors.accent.line), DOT_SEGMENTS);
		}
		list.AddCircleFilled(PlotPoint(rect, ease, loop_share), scale.Px(MARKER_RADIUS),
		                     packed(colors.control.knob), DOT_SEGMENTS);
	}
};

// The curves, each named, plotted over time. With reduced motion each is the straight line the
// real rule gives it.
Element CurveRow(UiFrame& frame, bool reduced) {
	const Theme&         theme = frame.ActiveTheme();
	const float          loop  = LoopMs(frame.Now(), PLOT_LOOP_MS) / PLOT_LOOP_MS;
	std::vector<Element> cells;
	cells.reserve(EASES.size());
	for (const NamedEase& named: EASES) {
		const Track track {PLOT_LOOP_MS, named.ease};
		const Ease  ease =
            reduced
		         ? ReducedMotion(TweenTracks {track, track}, theme.motion.reduced_fade_max).rise.ease
		         : named.ease;
		const std::array<Element, 2> parts {
		    Custom(frame, {.width = Dp(PLOT_WIDTH), .height = Dp(PLOT_HEIGHT)},
		           PlotPaint {ease, loop}),
		    Text(frame, {named.name, theme.type.caption, theme.colors.text.secondary})};
		cells.push_back(
		    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, parts));
	}
	return Box(frame, {.gap = Dp(theme.space.s4)}, {}, cells);
}

// Draws a duration's bar with a dot running along it for as long as the duration lasts, then
// resting. Trivially copyable so Custom can store it.
struct BarPaint {
	float bar_width;
	float dot_share;

	void operator()(const CustomPaint& painting) const {
		const auto& colors = painting.frame.ActiveTheme().colors;
		const auto& scale  = painting.frame.Scale();
		const Rect& rect   = painting.painted;
		const float width  = scale.Px(bar_width);
		const float height = scale.Px(BAR_HEIGHT);
		const float y      = rect.y + (rect.height - height) / 2.0f;
		painting.list.AddRectFilled({rect.x, y}, {rect.x + width, y + height},
		                            ToImU32(colors.accent.fill.MulAlpha(painting.opacity)));
		painting.list.AddCircleFilled(
		    {rect.x + width * dot_share, y + height / 2.0f}, scale.Px(MARKER_RADIUS),
		    ToImU32(colors.control.knob.MulAlpha(painting.opacity)), DOT_SEGMENTS);
	}
};

// The durations, each as a bar as long as it lasts. With reduced motion each is capped by the real
// rule, and a line says so.
Element DurationRows(UiFrame& frame, bool reduced) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> rows;
	rows.reserve(DURATIONS.size());
	for (const NamedDuration& named: DURATIONS) {
		const float full = theme.motion.*named.duration;
		const Track track {full, Ease::Linear};
		const float lasts =
		    reduced ? ReducedMotion(TweenTracks {track, track}, theme.motion.reduced_fade_max)
		                  .rise.duration_ms
		            : full;
		const float       dot  = std::min(1.0f, LoopMs(frame.Now(), lasts + BAR_REST_MS) / lasts);
		const std::string line = Formatted("%.0f ms", static_cast<double>(lasts));
		const std::array<Element, 3> parts {
		    Text(frame, {named.name, theme.type.caption, theme.colors.text.secondary},
		         {.width = Dp(BAR_NAME_WIDTH)}),
		    Text(frame, {line, theme.type.caption, theme.colors.text.secondary},
		         {.width = Dp(BAR_NAME_WIDTH)}),
		    Custom(frame, {.width = Dp(BAR_COLUMN_WIDTH), .height = Dp(theme.space.s5)},
		           BarPaint {lasts * BAR_PER_MS, dot})};
		rows.push_back(Box(frame, {.align = Align::Center}, {}, parts));
	}
	return Box(frame, {.direction = FlexDirection::Column}, {}, rows);
}

// The Motion view.
Element MotionView(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme   = frame.ActiveTheme();
	const bool                   reduced = state.effects_reduced_motion;
	const std::array<Element, 5> rows {
	    ReducedToggle(frame, state, "Reduced motion, each curve straight and each duration capped"),
	    SectionHeading(frame, "Curves"), CurveRow(frame, reduced),
	    SectionHeading(frame, "Durations"), DurationRows(frame, reduced)};
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s2)}, {}, rows);
}

} // namespace

float LoopMs(UiClock::time_point now, float period_ms) {
	const auto since = std::chrono::duration<double, std::milli>(now.time_since_epoch());
	return static_cast<float>(std::fmod(since.count(), static_cast<double>(period_ms)));
}

void DrawEffectsPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	const PageHead head = BuildPageHead(frame, page, RULE, &state.effects_view);
	const int view = std::clamp(state.effects_view, 0, static_cast<int>(EFFECTS_VIEWS.size()) - 1);
	const Element                body = view == 0   ? DepthView(frame)
	                                    : view == 1 ? ShineView(frame, state)
	                                                : MotionView(frame, state);
	const std::array<Element, 2> rows {head.root, body};
	const Element                root =
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(frame.ActiveTheme().space.s4)},
	        {}, rows);
	ShowPage(frame, state, root, head.tabs, view, pane);
}

} // namespace Libs::Graphics::SystemUi::Gallery
