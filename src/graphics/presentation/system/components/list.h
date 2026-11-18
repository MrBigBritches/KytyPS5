#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_LIST_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_LIST_H_

#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

#include <span>
#include <string_view>
#include <vector>

namespace Libs::Graphics::SystemUi {

// A list row's heights and corners, in reference pixels.
struct ListTokens {
	float row_single = 56.0f;
	float row_double = 80.0f;
	float row_radius = RadiusScale {}.md;

	/// True when a row of two lines is at least as tall as one of a single line.
	[[nodiscard]] constexpr bool IsValid() const {
		return row_single > 0.0f && row_single <= row_double && row_radius >= 0.0f;
	}
};

/// The list's tokens for `theme`: a row's corners come from the radius scale.
[[nodiscard]] constexpr ListTokens ListDefaults(const Theme& theme) {
	ListTokens tokens;
	tokens.row_radius = theme.radius.md;
	return tokens;
}

static_assert(ListDefaults(DEFAULT_THEME).IsValid());

// A list row's content and state.
struct ListRowProps {
	std::string_view primary;
	// A second, smaller line under the first. Empty for a single-line row.
	std::string_view secondary;
	// The current choice, shown whether or not the row has focus.
	bool selected = false;
	bool disabled = false;
	// Fields of ListDefaults to change for this row, such as its height.
	TokenPatch<ListTokens> tokens;
	// A row has no surface of its own, so it needs telling when what's behind it is light. Its
	// lines are then dark and its overlays shade.
	bool on_light = false;
};

/// A list row as an element: a box the height of one line, or two when it has a second line,
/// made interactive, with its lines stacked and centred top to bottom in the theme's row padding.
/// A selected row shows an accent bar and tint even without focus, a focused row shows the focus
/// overlay and ring, and a hovered one the hover overlay. A disabled row fades, is skipped by
/// navigation and never activates. `key` names the row among its siblings, so key it by the data
/// it shows and its focus and place survive the rows being reordered. The text is copied into the
/// frame. The lines are drawn by one custom node inside the row, which lays its text out when it
/// is painted. A list leaves out the rows scrolled out of its area when it paints, so the text of
/// a row out of view is never laid out, and never measured, which a text node would do whenever the
/// row was arranged. The text is not a text node, so SetTextColor can't recolour it. Give
/// the row an OnActivate callback to act on it. Render also reports the first row activated in a
/// frame through RenderResult::activated, and InputOf reports each row's. Returns a none handle
/// when a part is refused, with `reason` set if given.
[[nodiscard]] Element ListRow(UiFrame& frame, std::string_view key, const ListRowProps& props,
                              ElementError* reason = nullptr);

// A group's heading in a list.
struct ListHeadingProps {
	std::string_view text;
	// Fields of ListDefaults to change for this heading, such as its height.
	TokenPatch<ListTokens> tokens;
};

/// A group's heading in a list: a line of small text over the row's height, resting on its bottom
/// and set in like a row's label, with no input. It is for a list that sorts its rows into groups.
/// It is one custom node, laid out when painted, so as for a row its text is never laid out while
/// the list has scrolled it out of view. `key` names it among its siblings. Returns a none handle
/// when it is refused.
[[nodiscard]] Element ListHeading(UiFrame& frame, std::string_view key,
                                  const ListHeadingProps& props, ElementError* reason = nullptr);

/// A scrolling list of `rows`, which are interactive nodes such as ListRow makes, stacked in a
/// column and clipped to the list's area. The list scrolls to keep the focused row in view and
/// remembers its scroll and the row that last had focus under its key, so focus coming back into
/// the list from outside lands on that row. While focus is outside, rows not wholly in view are
/// left out of navigation, and rows outside the area can't be hovered or pressed, but the D-pad
/// still reaches them once focus is inside. The rows sit in from the list's edges by the focus
/// ring's width, so the ring isn't clipped, and further on the right to leave a gutter for the
/// scroll bar, which shows when they don't all fit. `item` sizes the list in its container, and it
/// needs a height to scroll within. The list is a clipping box made a list with AsList, so
/// everything said there holds. Returns a none handle when it is refused, with `reason` set if
/// given.
[[nodiscard]] Element List(UiFrame& frame, std::string_view key, std::span<const Element> rows,
                           const ItemLayout& item = {}, ElementError* reason = nullptr);

/// A list with one row for each of `data`, each made by the template `row(frame, datum)`, which
/// returns the element for that datum, such as a ListRow keyed by the datum's name, or a
/// ListHeading. The template runs again for every datum every frame, so a row always shows the
/// data it is given now and never a stale copy. The data must outlive the call, and the text a row
/// takes need not, since the frame copies it. The rows are gathered on the frame's ElementStack, so
/// a steady frame allocates nothing for them. Otherwise as List.
template <class Data, class RowTemplate>
[[nodiscard]] Element ListOf(UiFrame& frame, std::string_view key, std::span<const Data> data,
                             RowTemplate&& row, const ItemLayout& item = {},
                             ElementError* reason = nullptr) {
	std::vector<Element>& stack = frame.ElementStack();
	const size_t          first = stack.size();
	for (const Data& datum: data) {
		// The row is made before it is pushed, so a list the template builds inside it has finished
		// with the stack, and put it back, by then.
		const Element made = row(frame, datum);
		stack.push_back(made);
	}
	const Element list =
	    List(frame, key, std::span<const Element>(stack).subspan(first), item, reason);
	stack.resize(first);
	return list;
}

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_LIST_H_
