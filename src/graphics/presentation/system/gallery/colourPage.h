#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_COLOURPAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_COLOURPAGE_H_

#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/element/frame.h"

#include <array>
#include <string_view>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The colour page's views, in tab order.
inline constexpr std::array<std::string_view, 3> COLOUR_VIEWS {"Palette", "Roles", "Contrast"};

// The contrast a pair of colours should reach. Reading text needs 4.5 to 1, while large text and
// parts of a control that aren't text need 3 to 1, as WCAG 2.1 asks.
inline constexpr float CONTRAST_TEXT  = 4.5f;
inline constexpr float CONTRAST_LARGE = 3.0f;

// One pair of the contrast table. Surfaces are translucent over the game, so the pair is judged
// over a dark and over a white frame, and the worse of the two decides.
struct ContrastRow {
	std::string_view name;
	Color            foreground;
	Color            background;
	float            target = CONTRAST_TEXT;
	// What lies under `background`, for a tint on a surface. Transparent for nothing.
	Color under;
	// What `foreground` is drawn on when that isn't `background`, for an outline on the edge of a
	// track, judged against the surface around the track. Transparent for nothing.
	Color foreground_on;
};

/// `top` drawn over `under`, which is treated as opaque, blended in sRGB as the renderer does.
[[nodiscard]] Color Composite(Color top, Color under);

/// The WCAG contrast ratio of two colours, from 1 to 21. Their alphas are ignored.
[[nodiscard]] float ContrastRatio(Color first, Color second);

/// The ratio `row` reaches on a frame of the opaque colour `backdrop`.
[[nodiscard]] float ContrastOver(const ContrastRow& row, Color backdrop);

/// Whether `row` reaches its target over both a black and a white frame.
[[nodiscard]] bool ContrastPasses(const ContrastRow& row);

/// The pairs the theme's contrast is judged on.
[[nodiscard]] std::vector<ContrastRow> ContrastRows(const Theme& theme);

/// Draws the colour page into `pane`. Palette shows the raw colours, Roles what each semantic token
/// is over a dark and a white strip, and Contrast a table computed from the theme.
void DrawColourPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_COLOURPAGE_H_
