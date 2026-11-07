#include "graphics/presentation/system/gallery/pageParts.h"

#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include <array>
#include <limits>

namespace Libs::Graphics::SystemUi::Gallery {

PageHead BuildPageHead(UiFrame& frame, const PageEntry& page, std::string_view rule, int* view) {
	const Theme&    theme = frame.ActiveTheme();
	const TextProps title {page.title, theme.type.title, theme.colors.text.primary};
	const TextProps note {rule, theme.type.body_small, theme.colors.text.secondary};
	PageHead        head;
	head.tabs = TabBar(frame, "views", page.view_names, view);
	const std::array<Element, 3> rows {Text(frame, title), Text(frame, note), head.tabs.element};
	head.root =
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s2)}, {}, rows);
	return head;
}

Element SectionHeading(UiFrame& frame, std::string_view text) {
	const Theme& theme = frame.ActiveTheme();
	return Text(frame, {text, theme.type.label, theme.colors.text.primary});
}

void ShowPage(UiFrame& frame, GalleryState& state, Element root, const TabBarElement& tabs,
              int chosen, const Rect& pane) {
	(void)AssignIds(frame, root, CurrentIdSeed());
	const Vec2 size = Measure(frame, root, {pane.width, std::numeric_limits<float>::infinity()});
	(void)Arrange(frame, root, {pane.x, pane.y, pane.width, size.y});
	state.entry = IdOf(frame, TabOf(tabs, chosen));
	(void)Render(frame, root, CurrentDrawList());
}

} // namespace Libs::Graphics::SystemUi::Gallery
