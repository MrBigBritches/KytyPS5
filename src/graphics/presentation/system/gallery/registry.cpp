#include "graphics/presentation/system/gallery/colourPage.h"
#include "graphics/presentation/system/gallery/effectsPage.h"
#include "graphics/presentation/system/gallery/imagesPage.h"
#include "graphics/presentation/system/gallery/interactivePages.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/passivePages.h"
#include "graphics/presentation/system/gallery/patternPages.h"
#include "graphics/presentation/system/gallery/progressPage.h"
#include "graphics/presentation/system/gallery/spacingPage.h"
#include "graphics/presentation/system/gallery/toastPage.h"
#include "graphics/presentation/system/gallery/typographyPage.h"

#include <array>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

// Every page, in sidebar order.
constexpr std::array PAGES {
    PageEntry {PageGroup::Styles, "colour", "Colour", DrawColourPage, &GalleryState::colour_view,
               COLOUR_VIEWS},
    PageEntry {PageGroup::Styles, "typography", "Typography", DrawTypographyPage,
               &GalleryState::typography_view, TYPOGRAPHY_VIEWS},
    PageEntry {PageGroup::Styles, "spacing", "Spacing and layout", DrawSpacingPage,
               &GalleryState::spacing_view, SPACING_VIEWS},
    PageEntry {PageGroup::Styles, "images", "Images", DrawImagesPage, &GalleryState::images_view,
               IMAGES_VIEWS},
    PageEntry {PageGroup::Styles, "effects", "Effects", DrawEffectsPage,
               &GalleryState::effects_view, EFFECTS_VIEWS},
    PageEntry {PageGroup::Components, "box", "Box and card", DrawBoxPage},
    PageEntry {PageGroup::Components, "text", "Text", DrawTextPage},
    PageEntry {PageGroup::Components, "image", "Image", DrawImagePage},
    PageEntry {PageGroup::Components, "glyphs", "Glyphs", DrawGlyphsPage},
    PageEntry {PageGroup::Components, "progress", "Progress", DrawProgressPage},
    PageEntry {PageGroup::Components, "toast", "Toast", DrawToastPage},
    PageEntry {PageGroup::Components, "button", "Button", DrawButtonPage},
    PageEntry {PageGroup::Components, "toggle", "Toggle", DrawTogglePage},
    PageEntry {PageGroup::Components, "slider", "Slider", DrawSliderPage},
    PageEntry {PageGroup::Components, "list", "List", DrawListPage},
    PageEntry {PageGroup::Components, "tabs", "Tabs", DrawTabsPage},
    PageEntry {PageGroup::Patterns, "dialogs", "Dialogs", DrawDialogsPage,
               &GalleryState::dialogs_view, DIALOG_VIEWS},
    PageEntry {PageGroup::Patterns, "notification", "Notification", DrawNotificationPage},
    PageEntry {PageGroup::Patterns, "metric", "Metric", DrawMetricPage},
};

} // namespace

std::span<const PageEntry> Pages() {
	return PAGES;
}

std::string_view GroupTitle(PageGroup group) {
	switch (group) {
		case PageGroup::Styles: return "Styles";
		case PageGroup::Components: return "Components";
		case PageGroup::Patterns: return "Patterns";
	}
	return {};
}

} // namespace Libs::Graphics::SystemUi::Gallery
