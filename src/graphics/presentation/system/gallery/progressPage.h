#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PROGRESSPAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PROGRESSPAGE_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

/// Draws the progress page into `pane`, with bars at the values that behave differently, not
/// started, barely started, part way and done, in each fill the bars come in.
void DrawProgressPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PROGRESSPAGE_H_
