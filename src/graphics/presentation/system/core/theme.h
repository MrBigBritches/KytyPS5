#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_THEME_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_THEME_H_

#include "graphics/presentation/system/core/color.h"

#include <cstdint>
#include <string_view>
#include <type_traits>

// The system UI theme. Values are in three tiers.
//  - The palette holds raw colours with no meaning. A re-theme only edits the palette.
//  - Semantic tokens give each colour a role, such as a surface level or a text level.
//  - Component tokens size and style one kind of component. They are not here: each component
//    declares its own tokens struct in its header, with an XDefaults(theme) function that builds
//    it from the tiers above, and takes a TokenPatch to change a few fields.
// Components read semantic tokens and their own tokens, never the palette.
//
// Lengths are reference pixels at 1920x1080. ScaleContext converts them for the actual output.
namespace Libs::Graphics::SystemUi {

// Tier 1. The neutral ramp is a cool grey-blue that sits on most game art. Where a value
// equals today's overlay, the note says so, so ports keep that look.
namespace Palette {
inline constexpr Color GRAY_0   = Rgb(0x05070B);
inline constexpr Color GRAY_50  = Rgb(0x0E1117);
inline constexpr Color GRAY_100 = Rgb(0x151922);
inline constexpr Color GRAY_150 = Rgb(0x181A20); // today's trophy toast panel
inline constexpr Color GRAY_200 = Rgb(0x1E232E);
inline constexpr Color GRAY_300 = Rgb(0x2A3140);
inline constexpr Color GRAY_400 = Rgb(0x3A4252);
inline constexpr Color GRAY_500 = Rgb(0x5A6478);
inline constexpr Color GRAY_600 = Rgb(0x8A93A6);
inline constexpr Color GRAY_700 = Rgb(0xB0B4C0); // today's trophy toast subtitle
inline constexpr Color GRAY_800 = Rgb(0xDDE1EA);
inline constexpr Color GRAY_900 = Rgb(0xFFFFFF);

inline constexpr Color BLUE_300   = Rgb(0x7DBBFF);
inline constexpr Color BLUE_400   = Rgb(0x4DA6FF);
inline constexpr Color BLUE_500   = Rgb(0x2391FF); // today's touch cursor blue
inline constexpr Color BLUE_550   = Rgb(0x1D80EE);
inline constexpr Color BLUE_600   = Rgb(0x1670D6);
inline constexpr Color BLUE_700   = Rgb(0x1259B0);
inline constexpr Color GREEN_500  = Rgb(0x60DC78); // today's performance overlay "good"
inline constexpr Color YELLOW_500 = Rgb(0xE6C33C); // today's performance overlay "medium"
inline constexpr Color RED_400    = Rgb(0xFF7A72);
inline constexpr Color RED_500    = Rgb(0xEB5046); // today's performance overlay "bad"
inline constexpr Color RED_600    = Rgb(0xC93A32);
inline constexpr Color PLATINUM   = Rgb(0xA1E1F5); // today's trophy grade colours
inline constexpr Color GOLD       = Rgb(0xFFCF52);
inline constexpr Color SILVER     = Rgb(0xD3DCE8);
inline constexpr Color BRONZE     = Rgb(0xD38E5B);
} // namespace Palette

// Tier 1 opacities. The colour tokens below take their alpha from here, so a re-theme or a
// high-contrast variant changes each one in one place.
namespace Opacity {
inline constexpr float HOVER           = 0.06f;
inline constexpr float FOCUS           = 0.10f;
inline constexpr float BORDER_SUBTLE   = 0.11f;
inline constexpr float TINT            = 0.16f;
inline constexpr float PRESSED         = 0.20f;
inline constexpr float BORDER_STANDARD = 0.20f;
inline constexpr float INSET           = 0.35f;
inline constexpr float DISABLED        = 0.38f;
inline constexpr float BORDER_STRONG   = 0.40f;
inline constexpr float KEYLINE         = 0.60f;
inline constexpr float SCRIM           = 0.62f;
inline constexpr float HUD             = 0.72f;
inline constexpr float HUD_STRONG      = 0.84f;
inline constexpr float PANEL           = 0.92f;
inline constexpr float OVERLAY         = 0.96f;
} // namespace Opacity

// Tier 2. Surfaces are translucent over the game, so their alpha is part of the token.
struct SurfaceColors {
	Color base    = Palette::GRAY_100.WithAlpha(Opacity::PANEL);   // full-screen menus
	Color raised  = Palette::GRAY_150.WithAlpha(Opacity::PANEL);   // cards, toasts, dialogs
	Color overlay = Palette::GRAY_200.WithAlpha(Opacity::OVERLAY); // popups above raised surfaces
	Color hud     = Palette::GRAY_50.WithAlpha(Opacity::HUD); // non-interactive heads-up display
	Color hud_strong = Palette::GRAY_50.WithAlpha(Opacity::HUD_STRONG);
	Color inset = Palette::GRAY_0.WithAlpha(Opacity::INSET); // wells, graph backgrounds, tracks
	Color scrim = Palette::GRAY_0.WithAlpha(Opacity::SCRIM); // behind modals
};

// Text levels, from most to least prominent. The gallery's Colour page shows the contrast of each.
struct TextColors {
	Color primary   = Palette::GRAY_900;
	Color secondary = Palette::GRAY_700;
	Color tertiary  = Palette::GRAY_600;
	Color disabled  = Palette::GRAY_500;
	Color on_accent = Palette::GRAY_900;
	Color accent    = Palette::BLUE_500;
	// For words on a light surface, as every other text colour here is for a dark one.
	Color on_light           = Palette::GRAY_0;
	Color secondary_on_light = Palette::GRAY_400;
};

struct AccentColors {
	Color fill         = Palette::BLUE_600;
	Color fill_hover   = Palette::BLUE_550;
	Color fill_pressed = Palette::BLUE_700;
	Color line         = Palette::BLUE_500; // selection bars, progress fill
	Color tint         = Palette::BLUE_500.WithAlpha(Opacity::TINT); // selected row background
	// Opaque, since a glow takes its strength from its Glow token.
	Color glow = Palette::BLUE_500;
};

struct StatusColors {
	Color success = Palette::GREEN_500;
	Color warning = Palette::YELLOW_500;
	// A step lighter than their tints, as these are read as text, on a tint or a raised surface.
	Color error       = Palette::RED_400;
	Color info        = Palette::BLUE_300;
	Color danger_fill = Palette::RED_600;
	// Faint backgrounds for badges and toast bars in each status colour.
	Color success_tint = Palette::GREEN_500.WithAlpha(Opacity::TINT);
	Color warning_tint = Palette::YELLOW_500.WithAlpha(Opacity::TINT);
	Color error_tint   = Palette::RED_500.WithAlpha(Opacity::TINT);
	Color info_tint    = Palette::BLUE_400.WithAlpha(Opacity::TINT);
};

// Thresholds in graphs and the performance overlay.
struct ChartColors {
	Color good    = Palette::GREEN_500;
	Color medium  = Palette::YELLOW_500;
	Color bad     = Palette::RED_500;
	Color neutral = Palette::GRAY_600;
};

// The focus ring is a light stroke with a dark keyline and a soft glow, so it reads on light
// and dark game frames alike.
struct FocusColors {
	Color ring    = Palette::GRAY_900;
	Color keyline = Palette::GRAY_0.WithAlpha(Opacity::KEYLINE);
	// Opaque, since a glow takes its strength from its Glow token.
	Color glow = Palette::BLUE_500;
};

struct BorderColors {
	Color subtle   = Palette::GRAY_900.WithAlpha(Opacity::BORDER_SUBTLE);
	Color standard = Palette::GRAY_900.WithAlpha(Opacity::BORDER_STANDARD);
	Color strong   = Palette::GRAY_900.WithAlpha(Opacity::BORDER_STRONG);
};

// Overlays drawn on top of a control to show its state.
struct StateColors {
	Color hover   = Palette::GRAY_900.WithAlpha(Opacity::HOVER);
	Color focus   = Palette::GRAY_900.WithAlpha(Opacity::FOCUS);
	Color pressed = Palette::GRAY_0.WithAlpha(Opacity::PRESSED);
	// For what sits on a light surface, where a light overlay wouldn't show. A press already darkens.
	Color hover_on_light = Palette::GRAY_0.WithAlpha(Opacity::HOVER);
	Color focus_on_light = Palette::GRAY_0.WithAlpha(Opacity::FOCUS);
};

struct TrophyColors {
	Color platinum = Palette::PLATINUM;
	Color gold     = Palette::GOLD;
	Color silver   = Palette::SILVER;
	Color bronze   = Palette::BRONZE;
};

// Effects take their alpha from elevation and effect tokens.
struct EffectColors {
	Color shadow = Palette::GRAY_0;
	Color shine  = Palette::GRAY_900;
};

// Switches and sliders. A toggle's track is this grey when off and the accent fill when on, under
// a white knob. An unfilled track is too close to the surface behind it to read by its fill, so it
// also has a border.strong outline. A disabled slider is solid grey rather than faded, so it can't
// be taken for an enabled one on any game frame, with no blue in its fill and no white in its knob.
struct ControlColors {
	Color track_off     = Palette::GRAY_400;
	Color knob          = Palette::GRAY_900;
	Color fill_disabled = Palette::GRAY_500;
	Color knob_disabled = Palette::GRAY_600;
};

struct SemanticColors {
	SurfaceColors surface;
	TextColors    text;
	AccentColors  accent;
	StatusColors  status;
	FocusColors   focus;
	BorderColors  border;
	StateColors   state;
	TrophyColors  trophy;
	ChartColors   chart;
	EffectColors  effect;
	ControlColors control;
};

// A spacing scale on a 4-pixel grid.
struct SpaceScale {
	float hair = 2.0f;
	float s1   = 4.0f;
	float s2   = 8.0f;
	float s3   = 12.0f;
	float s4   = 16.0f;
	float s5   = 24.0f;
	float s6   = 32.0f;
	float s7   = 48.0f;
	float s8   = 64.0f;
	float s9   = 96.0f;
};

// A radius that rounds a box's corners fully, whatever its size. Drawing clamps a radius to half
// the box's shorter side, so any value past the largest box does it. It is large and finite and not
// infinity, since blending infinite radii between states gives inf minus inf, which is NaN.
// Blending it with a small radius keeps the corners round almost until the end of the blend, then
// snaps to the small one.
inline constexpr float RADIUS_FULL = 10000.0f;

// Corner radii. A fully rounded shape uses half its shorter side instead of a token.
struct RadiusScale {
	float xs = 4.0f;
	float sm = 8.0f;
	float md = 12.0f;
	float lg = 16.0f;
	float xl = 24.0f;
};

// Border widths. ScaleContext never lets a scaled border drop below one pixel.
struct BorderWidths {
	float hair  = 1.0f;
	float thin  = 2.0f; // selected outline, focus keyline
	float thick = 3.0f; // focus ring
	float bar   = 4.0f; // selection bar
};

// Opacities that components apply on top of a colour, such as dimming a disabled control.
struct OpacityScale {
	float hover    = Opacity::HOVER;
	float focus    = Opacity::FOCUS;
	float pressed  = Opacity::PRESSED;
	float disabled = Opacity::DISABLED;
};

enum class FontWeight : uint8_t { Regular, Bold };

// One text style. `min_size` is the smallest pixel size it may shrink to at low resolutions.
struct TypeStyle {
	float      size     = 0.0f;
	float      line     = 0.0f;
	FontWeight weight   = FontWeight::Regular;
	float      min_size = 0.0f;
};

// The type ramp. Body styles stay at 14 pixels or more, captions at 12 and dense heads-up text
// at 11.
struct TypeRamp {
	TypeStyle display {44.0f, 52.0f, FontWeight::Bold, 14.0f};
	TypeStyle title {32.0f, 40.0f, FontWeight::Bold, 14.0f};
	TypeStyle heading {26.0f, 34.0f, FontWeight::Bold, 14.0f};
	TypeStyle body {22.0f, 32.0f, FontWeight::Regular, 14.0f};
	TypeStyle body_small {20.0f, 28.0f, FontWeight::Regular, 14.0f};
	TypeStyle label {22.0f, 28.0f, FontWeight::Bold, 14.0f};
	TypeStyle caption {16.0f, 22.0f, FontWeight::Regular, 12.0f};
	TypeStyle dense {15.0f, 20.0f, FontWeight::Regular, 11.0f};
	TypeStyle dense_small {13.0f, 18.0f, FontWeight::Regular, 11.0f};
};

// Font files, relative to the emulator's assets folder.
struct FontFiles {
	std::string_view regular = "fonts/NotoSans-Regular.ttf";
	std::string_view bold    = "fonts/NotoSans-Bold.ttf";
};

// A soft shadow that fades outward from under an element.
struct Elevation {
	float blur       = 0.0f;
	float offset_y   = 0.0f;
	float peak_alpha = 0.0f;
};

// Level 0 is flat. Higher levels sit nearer the viewer.
struct ElevationScale {
	Elevation level0 {};
	Elevation level1 {8.0f, 2.0f, 0.30f};    // cards in lists
	Elevation level2 {16.0f, 6.0f, 0.40f};   // a focused card
	Elevation level3 {28.0f, 10.0f, 0.45f};  // toasts, dropdowns
	Elevation level4 {48.0f, 16.0f, 0.50f}; // modals
};

// A soft halo around an element, `radius` reference pixels wide. It reaches `peak_alpha` next to the
// element and fades toward the outside.
struct Glow {
	float radius     = 0.0f;
	float peak_alpha = 0.0f;
};

struct GlowScale {
	Glow focus {12.0f, 0.35f};    // around the focused control
	Glow emphasis {24.0f, 0.45f}; // a success or an arrival
	Glow trophy {32.0f, 0.30f};   // around a trophy toast, in the grade colour
};

// A soft band of light that sweeps across a surface once, leaning right.
struct ShineTokens {
	float angle_degrees    = 20.0f; // from vertical
	float width_fraction   = 0.3f;  // of the element's width
	float min_width        = 80.0f; // reference pixels
	float peak_alpha       = 0.14f;
	float plateau_fraction = 0.2f; // of the band, at full brightness in the middle
};

enum class Ease : uint8_t {
	Linear,   // spinners, indeterminate progress
	Enter,    // things arriving, 1 - (1 - t)^3
	Exit,     // things leaving, t^3
	Standard, // moves, resizes, scrolling and the shine sweep, cubic in and out
	Emphasis, // a success pop or a toggle knob, slightly overshooting
};

// Durations in milliseconds. Exits take about 65% of the matching enter.
struct MotionDurations {
	float instant  = 0.0f;
	float fast     = 100.0f;
	float base     = 180.0f;
	float slow     = 280.0f;
	float slower   = 420.0f;
	float emphasis = 600.0f;
	float shine    = 700.0f;
	// A quick fade out, for a toast leaving once its time is up.
	float exit_fade = 120.0f;
	float pulse     = 1600.0f;
	// The longest fade when the player asks for reduced motion.
	float reduced_fade_max = 120.0f;
};

// How reference pixels map to the output. The scale is
// clamp(min(width / reference_width, height / reference_height) * user scale, min, max).
struct ScaleRules {
	float reference_width  = 1920.0f;
	float reference_height = 1080.0f;
	float min_scale        = 2.0f / 3.0f; // 720p
	float max_scale        = 3.0f;        // 8K
	float min_user_scale   = 0.8f;
	float max_user_scale   = 1.5f;
	// The safe area keeps content away from the screen edges, as TVs may crop them. Its inset is
	// a percentage of the screen, but never less than `edge_min`.
	float default_safe_area_percent = 3.5f;
	float max_safe_area_percent     = 10.0f;
	float edge_min                  = 32.0f;
	// Menus and dialogs sit in a frame no wider than this aspect, centred in the safe area, so
	// they don't stretch across ultrawide screens.
	float content_aspect = 16.0f / 9.0f;

	// True when the rules can produce a finite scale. A theme must pass this before use.
	[[nodiscard]] constexpr bool IsValid() const {
		return reference_width > 0.0f && reference_height > 0.0f && min_scale > 0.0f &&
		       min_scale <= max_scale && min_user_scale > 0.0f &&
		       min_user_scale <= max_user_scale && max_safe_area_percent >= 0.0f &&
		       edge_min >= 0.0f && content_aspect > 0.0f;
	}
};

// How controls respond to focus, pointer hover and presses, and how a held direction repeats. Focus
// arrives a little slower than it leaves, so moving along a row reads as one highlight travelling.
struct InteractionTokens {
	float focus_in_ms  = MotionDurations {}.fast;
	float focus_out_ms = 60.0f;
	float hover_ms     = MotionDurations {}.fast;
	float press_ms     = MotionDurations {}.fast;
	// A pressed control shrinks slightly about its centre.
	float press_scale = 0.98f;
	// A toggle's knob, and a tab bar's indicator moving to the chosen tab.
	float knob_ms      = MotionDurations {}.base;
	float indicator_ms = MotionDurations {}.slow;
	// A held direction waits this long, then repeats at this interval.
	float repeat_delay_ms = 400.0f;
	float repeat_rate_ms  = 80.0f;
	// The smallest targets, in output pixels after scaling, not reference pixels. Gamepad
	// targets need more room than pointer targets.
	float min_gamepad_target = 44.0f;
	float min_pointer_target = 32.0f;
};

// Everything a component needs to look consistent. DEFAULT_THEME is the only theme for now.
struct Theme {
	SemanticColors  colors;
	SpaceScale      space;
	RadiusScale     radius;
	BorderWidths    border;
	OpacityScale    opacity;
	TypeRamp        type;
	FontFiles       fonts;
	ElevationScale  elevation;
	GlowScale       glow;
	ShineTokens     shine;
	MotionDurations motion;
	ScaleRules      scale;
	// Added last, so themes built field by field keep their order.
	InteractionTokens interaction;
};

inline constexpr Theme DEFAULT_THEME {};

// A runtime theme is built by copying DEFAULT_THEME and changing fields, so the tree must stay a
// plain aggregate. A component's own sizes live with the component, in a tokens struct built from
// the theme by its XDefaults function, so changing a scale on a copy changes the components that
// use it.
static_assert(std::is_aggregate_v<Theme> && std::is_trivially_copyable_v<Theme>);
static_assert(DEFAULT_THEME.scale.IsValid());

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_THEME_H_
