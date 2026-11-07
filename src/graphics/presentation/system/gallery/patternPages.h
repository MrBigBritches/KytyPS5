#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PATTERNPAGES_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PATTERNPAGES_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"

#include <array>
#include <string_view>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The dialogs page's views, each a dialog open over the stage, after the page with none. A
// screenshot run takes each of them, so it captures the real dialogs.
inline constexpr std::array<std::string_view, 6> DIALOG_VIEWS {"Closed", "Confirm", "Progress",
                                                               "Saving", "Error",   "Long"};

/// Draws the dialogs page into `pane`, buttons that open each kind of dialog over the stage's
/// panel, at the output chosen, and what each kind is for.
void DrawDialogsPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

/// Draws the notification page into `pane`, each kind of toast, and a toast's entrance and exit
/// frozen at moments in them.
void DrawNotificationPage(UiFrame& frame, GalleryState& state, const PageEntry& page,
                          const Rect& pane);

/// Draws the metric page into `pane`, a labelled mock of a heads-up panel and a stat tile, as no
/// component draws them yet.
void DrawMetricPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PATTERNPAGES_H_
