#include "graphics/presentation/system/gallery/shell.h"

#include "graphics/presentation/system/components/list.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/imgui/interaction.h"
#include "imgui.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

// The sidebar's width at the reference output, room for the longest page title.
constexpr float SIDEBAR_WIDTH = 320.0f;
// The scope the sidebar's rows are named in, so a page's controls can share their names.
constexpr std::string_view SIDEBAR_SCOPE = "##sidebar";
constexpr std::string_view SIDEBAR_LIST  = "##pages";

// One line of the sidebar, a group's heading or a page's row.
struct SidebarLine {
	PageGroup group = PageGroup::Styles;
	// The page's index, or -1 for a heading.
	int page = -1;
};

// The sidebar's lines, each group's heading before its first page.
std::vector<SidebarLine> SidebarLines(std::span<const PageEntry> pages) {
	std::vector<SidebarLine> lines;
	for (int index = 0; index < static_cast<int>(pages.size()); index++) {
		if (index == 0 || pages[index].group != pages[index - 1].group) {
			lines.push_back({pages[index].group, -1});
		}
		lines.push_back({pages[index].group, index});
	}
	return lines;
}

// What the sidebar's rows did this frame.
struct RowsInput {
	// The page whose row has focus, or -1.
	int focused = -1;
	// Right was pressed on the focused row, asking to enter its page.
	bool enter = false;
	// How many rows up or down the focused row was asked to move, as -1, 0 or 1.
	int step = 0;
};

// The sidebar's list, built and laid out, with each page's row.
struct SidebarTree {
	Element              list;
	std::vector<Element> pageRows;
};

// Builds the sidebar's list, a heading before each group's first page and a row for every page,
// with a row for every page that claims left and right, so the focused one keeps them from
// navigation. The rows are keyed by their page, and the headings by their group.
SidebarTree BuildList(UiFrame& frame, const GalleryState& state, std::span<const PageEntry> pages,
                      const Rect& area) {
	SidebarTree          tree;
	std::vector<Element> rows;
	tree.pageRows.resize(pages.size());
	for (const SidebarLine& line: SidebarLines(pages)) {
		if (line.page < 0) {
			rows.push_back(
			    ListHeading(frame, std::string("##heading") + std::string(GroupTitle(line.group)),
			                {.text = GroupTitle(line.group)}));
			continue;
		}
		const Element row = ListRow(frame, pages[line.page].id,
		                            {pages[line.page].title, {}, line.page == state.page});
		// Left and right belong to the focused row, since its page's controls may lie above or
		// below it, where right wouldn't reach them.
		(void)ClaimDpad(frame, row, DpadClaim::Once);
		rows.push_back(row);
		tree.pageRows[line.page] = row;
	}
	const ItemLayout size {.width  = FromPixels(frame.Scale(), area.width),
	                       .height = FromPixels(frame.Scale(), area.height)};
	tree.list = List(frame, SIDEBAR_LIST, rows, size);
	return tree;
}

// Builds, names and lays out the sidebar's list inside `area`. The ids are given here, before
// Render, so the shell can focus a page's row by the id the tree gave it.
SidebarTree PrepareList(UiFrame& frame, GalleryState& state, std::span<const PageEntry> pages,
                        const Rect& area) {
	SidebarTree tree = BuildList(frame, state, pages, area);
	(void)AssignIds(frame, tree.list, CurrentIdSeed());
	const float unlimited = std::numeric_limits<float>::infinity();
	(void)Measure(frame, tree.list, {unlimited, unlimited});
	(void)Arrange(frame, tree.list, area);
	return tree;
}

// Draws the prepared sidebar rows and takes their input. A click focuses a row too, so the caller
// shows the page of whichever row has focus. The focused row keeps left and right from navigation,
// since its page's controls may lie above or below it, where right wouldn't reach them. It keeps up
// and down too, as a page's control can lie nearer than the next row, and navigation would go
// there. While focus is in the page the rows are out of navigation, or down from a control with a
// row nearer than anything below it would leave the page and show another. Left still returns, as
// the gallery does that itself when its move finds nothing.
RowsInput DrawRows(UiFrame& frame, const SidebarTree& tree, bool focus_in_page) {
	{
		const NavExclusion out_of_navigation(focus_in_page);
		(void)Render(frame, tree.list, CurrentDrawList());
	}
	RowsInput input;
	for (int page = 0; page < static_cast<int>(tree.pageRows.size()); page++) {
		const auto found = InputOf(frame, tree.pageRows[page]);
		if (!found) {
			continue;
		}
		const VerticalPresses vertical = ClaimVertical(found->id);
		if (found->interaction.state.focused) {
			input.focused = page;
			input.enter   = found->dpad.right;
			input.step    = (vertical.down ? 1 : 0) - (vertical.up ? 1 : 0);
		}
	}
	return input;
}

// What the sidebar asks of the page drawn after it.
struct SidebarInput {
	// Right on a row asked to enter the page.
	bool enter = false;
	// L1 or R1 changed the page while focus was in the old one, whose controls are gone, so focus
	// must move into the new page or to its row.
	bool refocus = false;
	// The shown page's row.
	ItemId row = 0;
	// Focus is on a control of the page, so the rows were out of navigation this frame.
	bool focus_in_page = false;
};

// What the shortcuts decided, before any row has an id.
struct ShortcutResult {
	// The page whose row should take focus now, or -1.
	int focus_page = -1;
	// L1 or R1 changed the page while focus was in the old page.
	bool refocus = false;
};

// Moves to a page asked for from outside, or the page before or after on L1 or R1, and says whether
// focus should go to the shown page's row. A page asked for from outside always takes focus to its
// row. Back from a page returns focus to its row. Focus moves while focus is in the sidebar too,
// so it follows the page. The caller does the focusing, once the rows have ids.
ShortcutResult ApplyShortcuts(GalleryState& state, std::span<const PageEntry> pages, bool inside) {
	const auto presses   = ShoulderPresses();
	const int  last      = static_cast<int>(pages.size()) - 1;
	const bool requested = state.show >= 0;
	const int  stepped =
	    std::clamp(state.page + (presses.right ? 1 : 0) - (presses.left ? 1 : 0), 0, last);
	const int moved = requested ? std::clamp(state.show, 0, last) : stepped;
	// Back belongs to the gallery only while focus is in its own window, and while nobody in front
	// of it, such as a dialog, owns the key.
	const bool back =
	    !inside && FocusedControl() != 0 && FocusInCurrentWindow() && BackKeyPressed(NO_OWNER);
	state.show = -1;
	// Acting on a press takes it from the page drawn after the shell, so a tab bar on the page
	// just shown can't step on the same R1 that brought the page in.
	if (presses.left || presses.right) {
		ConsumeKey(KeyGroup::Shoulders, IdFor(SIDEBAR_LIST));
	}
	if (back) {
		ConsumeKey(KeyGroup::Back, IdFor(SIDEBAR_LIST));
	}
	ShortcutResult result;
	if ((moved != state.page && inside) || back || requested) {
		result.focus_page = moved;
	}
	result.refocus = moved != state.page && !inside && FocusedControl() != 0 && !requested;
	state.page     = moved;
	return result;
}

// Whether focus is on a control of the page. A dialog in front of the page keeps its own focus,
// which isn't this window's.
bool FocusInPage(const UiFrame& frame, const SidebarTree& tree) {
	const ItemId focused = FocusedControl();
	return focused != 0 && FocusInCurrentWindow() &&
	       std::none_of(tree.pageRows.begin(), tree.pageRows.end(),
	                    [&frame, focused](Element row) { return IdOf(frame, row) == focused; });
}

// Draws the sidebar's background and rows and takes their input.
SidebarInput DrawSidebar(UiFrame& frame, GalleryState& state, std::span<const PageEntry> pages,
                         const Rect& rect) {
	const auto& theme = frame.ActiveTheme();
	CurrentDrawList().AddRectFilled(TopLeft(rect), BottomRight(rect),
	                                ToImU32(theme.colors.surface.base));
	const IdScope scope(SIDEBAR_SCOPE);
	auto&         memory = frame.Store().List(IdFor(SIDEBAR_LIST));
	const bool    inside = memory.inside;
	SidebarInput  result;
	const auto    shortcuts = ApplyShortcuts(state, pages, inside);
	result.refocus          = shortcuts.refocus;
	const Rect        area  = rect.Inset(EdgeInsets::All(frame.Scale().Px(theme.space.s4)));
	const SidebarTree tree  = PrepareList(frame, state, pages, area);
	if (shortcuts.focus_page >= 0) {
		FocusControl(IdOf(frame, tree.pageRows[shortcuts.focus_page]));
	}
	if (!inside) {
		// The list returns focus to the row it remembers, so focus coming in lands on the page
		// shown rather than the row nearest the way in.
		memory.row = IdOf(frame, tree.pageRows[state.page]);
	}
	result.focus_in_page = FocusInPage(frame, tree);
	const auto input     = DrawRows(frame, tree, result.focus_in_page);
	if (inside && input.focused >= 0) {
		state.page = input.focused;
	}
	if (input.focused >= 0 && input.step != 0) {
		const int last = static_cast<int>(pages.size()) - 1;
		MoveFocus(IdOf(frame, tree.pageRows[std::clamp(input.focused + input.step, 0, last)]));
	}
	result.enter = input.enter;
	result.row   = IdOf(frame, tree.pageRows[state.page]);
	return result;
}

} // namespace

Rect SidebarRect(const UiFrame& frame, const Rect& screen) {
	return {screen.x, screen.y, std::min(screen.width, frame.Scale().Px(SIDEBAR_WIDTH)),
	        screen.height};
}

Rect PagePane(const UiFrame& frame, const Rect& screen) {
	const float margin  = frame.Scale().Px(frame.ActiveTheme().space.s6);
	const Rect  sidebar = SidebarRect(frame, screen);
	const float x       = sidebar.Right() + margin;
	return {x, screen.y + margin, std::max(0.0f, screen.Right() - margin - x),
	        std::max(0.0f, screen.height - 2.0f * margin)};
}

void DrawGallery(UiFrame& frame, GalleryState& state, std::span<const PageEntry> pages,
                 const Rect& screen) {
	if (pages.empty()) {
		return;
	}
	// A page replays its toast for as long as the gallery runs, and each one retires an image.
	if (state.images != nullptr) {
		state.images->Collect();
	}
	// A new ImGui frame needs a frame begun again. One drawn with last frame's tree, theme and time
	// would draw and act on what was already there, so it is recorded, and debug builds stop here.
	state.frame_was_stale = ImGui::GetFrameCount() != state.drawn_in_imgui_frame &&
	                        frame.BeginCount() == state.drawn_begin_count;
	assert(!state.frame_was_stale && "the frame was not begun for this ImGui frame");
	state.drawn_in_imgui_frame = ImGui::GetFrameCount();
	state.drawn_begin_count    = frame.BeginCount();
	state.page                 = std::clamp(state.page, 0, static_cast<int>(pages.size()) - 1);
	const auto    sidebar      = DrawSidebar(frame, state, pages, SidebarRect(frame, screen));
	const auto&   page         = pages[state.page];
	const IdScope scope(page.id);
	state.entry = 0;
	if (page.draw != nullptr) {
		page.draw(frame, state, page, PagePane(frame, screen));
	}
	if ((sidebar.enter || sidebar.refocus) && state.entry != 0) {
		FocusControl(state.entry);
	} else if (sidebar.refocus || (sidebar.focus_in_page && LeftMoveFindsNothing())) {
		// Left from a page control with nothing level to its left still returns to the sidebar,
		// but not from a dialog in front of the page, which keeps its own focus.
		FocusControl(sidebar.row);
	}
}

} // namespace Libs::Graphics::SystemUi::Gallery
