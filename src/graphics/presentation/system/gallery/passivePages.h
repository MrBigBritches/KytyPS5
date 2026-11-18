#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PASSIVEPAGES_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PASSIVEPAGES_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

/// Draws the box and card page into `pane`, each surface with each border, and a card in use, with
/// the nesting to avoid.
void DrawBoxPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

/// Draws the text page into `pane`, covering what the Typography page doesn't. Text aligned in a
/// slot, the size each style really reaches at each output, and a line limit on a paragraph.
void DrawTextPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

/// Draws the image page into `pane`, covering what the Images page doesn't. Images as they are
/// used, in a poster card, as an icon and as a row's thumbnail, and a card with its image missing.
void DrawImagePage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

/// Draws the glyphs page into `pane`, the trophy cup in each grade at each size, in use beside a
/// word, and stretched out of shape.
void DrawGlyphsPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PASSIVEPAGES_H_
