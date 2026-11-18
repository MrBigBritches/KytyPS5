#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_SHELL_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_SHELL_H_

#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include <span>

namespace Libs::Graphics::SystemUi::Gallery {

/// The sidebar's rectangle along the left of `screen`.
[[nodiscard]] Rect SidebarRect(const UiFrame& frame, const Rect& screen);

/// The space a page draws in, beside the sidebar and inset from it and from the screen's edges by
/// the page margin, which leaves room for the focus ring around a control at the pane's edge.
[[nodiscard]] Rect PagePane(const UiFrame& frame, const Rect& screen);

/// Collects `state.images`, if any, so its caller needn't. Draws the gallery filling `screen` in the
/// current window and takes its input. `frame` is the
/// caller's, begun for this frame, and the sidebar and the page are built in it. Debug builds
/// assert that it was begun again since the gallery last drew in an earlier ImGui frame, and every
/// build records the result in `state.frame_was_stale`. The sidebar lists `pages` under their
/// groups' headings, and the page `state.page` draws beside it,
/// its controls named within the page. Focus moving along the sidebar shows the page it lands on,
/// and focus coming into the sidebar lands on the page shown. Right on a row moves focus to the
/// page's entry control, and left from a page control with nothing beside it returns to the page's
/// row. L1 and R1 show the page before and after, and from inside a page take focus along to the
/// new page's entry control, or its row when it has none. Back from a page returns focus to its
/// row. Does nothing with no pages. The window's navigation must be set up with
/// ConfigureNavigation, or ImGui clears focus on back before the gallery sees it.
void DrawGallery(UiFrame& frame, GalleryState& state, std::span<const PageEntry> pages,
                 const Rect& screen);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_SHELL_H_
