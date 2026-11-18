#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_INTERACTIVEPAGES_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_INTERACTIVEPAGES_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

/// Draws the button page into `pane`, each variant in each state, and live buttons to try.
void DrawButtonPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

/// Draws the toggle page into `pane`, off and on in each state, and a live toggle to try.
void DrawTogglePage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

/// Draws the slider page into `pane`, three values in each state, and a live slider to try.
void DrawSliderPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

/// Draws the list page into `pane`, a row of each kind as it is and disabled, and a live list to
/// try.
void DrawListPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

/// Draws the tabs page into `pane`, a live tab bar and a note that tabs have no disabled state.
void DrawTabsPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_INTERACTIVEPAGES_H_
