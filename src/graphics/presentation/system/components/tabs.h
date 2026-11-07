#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TABS_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TABS_H_

#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

#include <cstddef>
#include <span>
#include <string_view>

namespace Libs::Graphics::SystemUi {

// A tab bar's height and the thickness of its indicator, in reference pixels.
struct TabTokens {
	float height    = 56.0f;
	float indicator = 3.0f;

	/// True when both lengths are positive and the indicator fits in the bar.
	[[nodiscard]] constexpr bool IsValid() const {
		return height > 0.0f && indicator > 0.0f && indicator <= height;
	}
};

/// The tab bar's tokens for `theme`. A bar is as tall as a medium button, which is the size of the
/// targets it sits among, so its height follows the button's tokens. It is defined in tabs.cpp, so
/// this header needs nothing of the button's.
[[nodiscard]] TabTokens TabDefaults(const Theme& theme);

static_assert(TabTokens {}.IsValid());

// A tab bar's input and how it reports a change.
struct TabBarProps {
	// Lets the gamepad's L1 and R1 choose tabs from anywhere on the screen. Give it to one bar per
	// screen. A control or dialog that owns L1 and R1 still keeps them from the bar. The bar owns
	// the shoulders only while it takes input, so a bar shown as a Specimen leaves them alone.
	bool shoulders = false;
	// Where Render says whether the choice changed this frame. It is set false when the bar is
	// built and Render overwrites it. A choice out of range that is clamped counts as a change.
	// Left null when the caller doesn't need it.
	bool* changed = nullptr;
	// Fields of TabDefaults to change for this bar, such as its height.
	TokenPatch<TabTokens> tokens;
};

/// A tab bar. Its handles and pointers are valid until the next Begin of the frame it was built in,
/// so build it, render it and drop it within one frame.
struct TabBarElement {
	/// The bar, or a none handle when it is refused or has no labels.
	Element element;
	/// The handles of its tabs, which TabOf reads.
	const std::byte* tabs  = nullptr;
	int              count = 0;
};

/// A row of tabs named `labels` as an element, with an accent bar sliding under the chosen tab.
/// `selected` is the caller's choice, which must outlive Render, and is clamped to the tabs and
/// changed when Render finds the player's input. The bar is a stacked box of three layers, an
/// empty one that gives it `item.width` at least, the tabs in a row, and the indicator over them.
/// Its width is the larger of `item.width` and the tabs, so tabs wider than the width asked for run
/// past it and the indicator stretches over all of them, and arrange it in the width Measure gave.
/// Its height is the theme's medium button unless `item` says otherwise.
///
/// Each tab is an interactive box named by its place, so tabs may share labels, and `key` names the
/// bar among its siblings. While a tab has focus, the D-pad's left and right move focus and the
/// choice together, confirm or a click chooses a tab, and focus coming into the bar lands on the
/// chosen tab. With the shoulders prop, L1 and R1 choose the tab before and after, and a press the
/// bar acts on is taken from everything drawn after it, so a page paging on the same press doesn't
/// step the tab too. A tab bar never claims the D-pad, since navigation moves focus between tabs.
///
/// Render decides the choice once every tab has taken its input, so the labels and the indicator
/// follow it on the frame it is made, and two bars in one tree need nothing more. A change is
/// reported through `props.changed` on that frame, and an AfterInteract callback on a node above
/// the bar sees it before anything is painted. Returns the bar, and a none element, with `reason`
/// set if given, when a part is refused. With no labels it returns a none element and sets `reason`
/// to NoLabels, and with a null `selected` to NoValue.
[[nodiscard]] TabBarElement TabBar(UiFrame& frame, std::string_view key,
                                   std::span<const std::string_view> labels, int* selected,
                                   const TabBarProps& props = {}, const ItemLayout& item = {},
                                   ElementError* reason = nullptr);

/// The interactive box of the tab at `index` in `bar`, for InputOf and the like, or a none handle
/// when the bar was refused or `index` is out of range. Valid until the next Begin.
[[nodiscard]] Element TabOf(const TabBarElement& bar, int index);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TABS_H_
