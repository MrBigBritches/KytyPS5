#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_TYPOGRAPHYPAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_TYPOGRAPHYPAGE_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"

#include <array>
#include <string_view>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The typography page's views, in tab order.
inline constexpr std::array<std::string_view, 2> TYPOGRAPHY_VIEWS {"Ramp", "Text"};

/// Draws the typography page into `pane`. Ramp shows every style of the type ramp, with a toggle
/// that shows the line box each one takes. Text shows the text colours, what each overflow mode
/// does with text too long for its slot, longer languages and digits in a column. The gallery loads
/// no CJK font, so no CJK sample is shown, as it would only draw the font's missing glyph.
void DrawTypographyPage(UiFrame& frame, GalleryState& state, const PageEntry& page,
                        const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_TYPOGRAPHYPAGE_H_
