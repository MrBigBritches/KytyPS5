#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PAGEPARTS_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PAGEPARTS_H_

#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"

#include <string_view>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The top of a page with views: its title, its usage rule and the tabs that choose a view.
struct PageHead {
	Element       root;
	TabBarElement tabs;
};

/// Builds the head of `page`, with `rule` under its title and a tab for each of its views, choosing
/// into `view`. `view` must outlive the frame's Render.
[[nodiscard]] PageHead BuildPageHead(UiFrame& frame, const PageEntry& page, std::string_view rule,
                                     int* view);

/// A section's heading, in the page's label style.
[[nodiscard]] Element SectionHeading(UiFrame& frame, std::string_view text);

/// Lays `root` out at the top left of `pane`, at the pane's width, makes the chosen tab of `tabs`
/// the control focus enters the page on, and renders it.
void ShowPage(UiFrame& frame, GalleryState& state, Element root, const TabBarElement& tabs,
              int chosen, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PAGEPARTS_H_
