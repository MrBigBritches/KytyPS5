#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_SPACINGPAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_SPACINGPAGE_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"

#include <array>
#include <string_view>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The spacing page's views, in tab order.
inline constexpr std::array<std::string_view, 2> SPACING_VIEWS {"Scales", "Frame"};

/// Draws the spacing and layout page into `pane`. Scales shows the space, radius and border scales
/// and the rules that scale them for an output. Frame shows the safe area and content frame of a
/// sample screen, worked out by the real ScaleContext, with controls for the safe area and for an
/// ultrawide screen, and the padding each kind of surface keeps.
void DrawSpacingPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_SPACINGPAGE_H_
