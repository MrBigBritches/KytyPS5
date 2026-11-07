#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_STAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_STAGE_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/element/frame.h"

#include <array>
#include <string_view>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The outputs a component page can show its specimens at, in tab order, and their sizes.
inline constexpr std::array<std::string_view, 4> STAGE_OUTPUT_NAMES {"720p", "1080p", "4K",
                                                                     "Ultrawide"};
inline constexpr std::array<Vec2, 4> STAGE_OUTPUTS {Vec2 {1280.0f, 720.0f}, Vec2 {1920.0f, 1080.0f},
                                                    Vec2 {3840.0f, 2160.0f},
                                                    Vec2 {3440.0f, 1440.0f}};
inline constexpr int                 STAGE_OUTPUT_1080P = 1;

// The room a stage's rows keep for the words before them, in reference pixels, so what follows
// lines up.
inline constexpr float STAGE_ROW_LABEL_WIDTH = 120.0f;

// The backdrops a component page can show its specimens on, in tab order, and their colours.
inline constexpr std::array<std::string_view, 3> STAGE_BACKDROP_NAMES {"Dark", "Mid", "Bright"};
inline constexpr std::array<Color, 3> STAGE_BACKDROPS {Palette::GRAY_50, Palette::GRAY_500,
                                                       Palette::GRAY_800};
inline constexpr int                  STAGE_BACKDROP_MID    = 1;
inline constexpr int                  STAGE_BACKDROP_BRIGHT = 2;

// What a component page's control strip chose, and the frame its specimens are built in. A second
// frame, as a frame has one scale, and the specimens' differs from the page's.
struct StageState {
	UiFrame frame;
	int     output   = STAGE_OUTPUT_1080P;
	int     backdrop = STAGE_BACKDROP_MID;
	float   opacity  = 1.0f;
	// How far the specimens were shrunk to fit on the last frame, which the strip shows.
	float fit = 1.0f;

	/// Puts the choices back as they start.
	void Reset();
};

// Builds a page's specimens in the stage's frame, at the output chosen, from what the page keeps.
// Handles from the page's own frame mean nothing in it, so a builder must take none. A live copy is
// built the same way. Key the root of each, as they share the frame.
using StageBuild = Element (*)(UiFrame& frame, GalleryState& state);
// Draws what lies over a page's stage in the stage's frame, at the output chosen, with `panel` the
// backdrop's place in output pixels. It renders for itself, at true size, after the specimens.
using StageOverlay = void (*)(UiFrame& frame, GalleryState& state, const Rect& panel);

/// The colour of words that sit straight on the backdrop chosen, which has to read on all of them.
[[nodiscard]] Color StageInk(const Theme& theme, const StageState& stage);

/// How far to shrink specimens of `size` to fit `area`, by whichever side is tightest, and never
/// past their own size, as they show what they look like at that output.
[[nodiscard]] float StageFit(Vec2 size, const Rect& area);

/// Shows a component page in `pane`. Its title, `rule`, the control strip, and below them the
/// specimens `build` makes, laid out at the output chosen, at the opacity chosen, on the backdrop
/// chosen, and shrunk to fit when they're too big, never grown. A `live` copy, if given, takes
/// input, so it sits above them at the output's true size and is never shrunk, as a shrunk control
/// would not be where the pointer finds it. An `overlay`, if given, is drawn over them all.
void ShowStagePage(UiFrame& frame, GalleryState& state, const PageEntry& page,
                   std::string_view rule, StageBuild build, const Rect& pane,
                   StageBuild live = nullptr, StageOverlay overlay = nullptr);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_STAGE_H_
