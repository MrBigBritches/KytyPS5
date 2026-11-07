#include "graphics/presentation/system/components/tabs.h"

#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/leaves.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/imgui/interaction.h"
#include "imgui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <vector>

namespace Libs::Graphics::SystemUi {

namespace {

// The handles of a bar's parts, kept as bytes in frame storage so a trivially copyable callable
// can reach them. Tab `index` is entry 2 * index and its label the entry after. A handle is copied
// out of the bytes, which need not be aligned for it.
Element EntryAt(const std::byte* table, int entry) {
	Element found;
	std::memcpy(static_cast<void*>(&found), table + static_cast<size_t>(entry) * sizeof(Element),
	            sizeof(Element));
	return found;
}

// The tab at `index`.
Element TabAt(const std::byte* table, int index) {
	return EntryAt(table, 2 * index);
}

// The label inside the tab at `index`.
Element LabelAt(const std::byte* table, int index) {
	return EntryAt(table, 2 * index + 1);
}

static_assert(std::is_trivially_copyable_v<Element>, "Tab handles are stored as raw bytes");

// Draws the accent indicator along the bar's bottom. Its edges are shares of the tabs' whole
// width, so it stretches between tabs of different widths and is never cut off when the tabs run
// past the bar. Each edge is eased toward the chosen tab's, and a bar that appears starts at its
// target. Trivially copyable so Custom can store it.
struct IndicatorPaint {
	const std::byte* tabs;
	int              count;
	const int*       selected;
	TabTokens        tokens;

	void operator()(const CustomPaint& painting) const {
		const Theme& theme = painting.frame.ActiveTheme();
		// Where each tab is drawn, as a pose that scales the bar moves and sizes them too.
		const auto painted = [&](int index) {
			return MapRect(Arranged(painting.frame, TabAt(tabs, index)), painting.transform);
		};
		const Rect first  = painted(0);
		const Rect last   = painted(count - 1);
		const Rect chosen = painted(std::clamp(*selected, 0, count - 1));
		const Rect& bar   = painting.painted;
		const float span  = std::max({1.0f, bar.width, last.Right() - first.x});
		float       left  = (chosen.x - first.x) / span;
		float       right = (chosen.Right() - first.x) / span;
		// A tree painted without Render has no ids, so nothing can keep a tween and the indicator
		// rests under the chosen tab.
		if (painting.id != 0) {
			const Track       track {theme.interaction.indicator_ms, Ease::Standard};
			const TweenTracks slide {track, track};
			auto&             store = painting.frame.Store();
			left  = store.Animate(IdWithSeed("##indicator left", painting.id), left, slide,
			                      painting.frame.Now(), true);
			right = store.Animate(IdWithSeed("##indicator right", painting.id), right, slide,
			                      painting.frame.Now(), true);
		}
		const float thickness = painting.frame.Scale().Px(tokens.indicator);
		const float bottom    = first.Bottom();
		painting.list.AddRectFilled(
		    {bar.x + left * span, bottom - thickness}, {bar.x + right * span, bottom},
		    ToImU32(theme.colors.accent.line.MulAlpha(painting.opacity)), thickness / 2.0f);
	}
};

// The colour of a label, the chosen tab's or the others'.
Color LabelColor(const Theme& theme, bool chosen) {
	return chosen ? theme.colors.text.primary : theme.colors.text.secondary;
}

// One tab, an interactive box of `width` whole pixels with its label centred in it. The label is
// as wide as its text rounded up to a whole pixel, which is never less than the text, so it can't
// be cut short with an ellipsis. The tab shades on hover and shows the focus ring, which the label
// is drawn over. Sets `label_out` to the label, and returns a none handle when a part is refused.
Element MakeTab(UiFrame& frame, const TabTokens& tokens, int index, std::string_view label,
                bool chosen, float width, float natural, Element& label_out, ElementError* reason) {
	const Theme& theme = frame.ActiveTheme();
	Style        style {.radius = Dp(theme.radius.sm)};
	style.states.hover.overlay    = theme.colors.state.hover;
	style.states.focus.focus_ring = true;
	const TextProps props {label, theme.type.label, LabelColor(theme, chosen), TextAlign::Center};
	label_out =
	    Text(frame, props, {.width = FromPixels(frame.Scale(), std::ceil(natural))}, reason);
	const std::array<Element, 1> content {label_out};
	const Layout                 layout {
	                    .direction = FlexDirection::Column,
	                    .justify   = Justify::Center,
	                    .align     = Align::Center,
	                    .item = {.width = FromPixels(frame.Scale(), width), .height = Dp(tokens.height)}};
	const Element tab = Box(frame, layout, style, content, reason);
	if (Keyed(frame, tab, static_cast<uint32_t>(index), reason).IsNone()) {
		return {};
	}
	return MakeInteractive(frame, tab, {}, reason);
}

// Builds a tab for each of `labels`, pushing the tab and then its label on the frame's stack, where
// they stay for the caller, which also gives the stack back. Returns false when one is refused.
bool PushTabs(UiFrame& frame, const TabTokens& tokens, std::span<const std::string_view> labels,
              int chosen, ElementError* reason) {
	const Theme&          theme = frame.ActiveTheme();
	std::vector<Element>& stack = frame.ElementStack();
	// The tabs lie along fractional edges, each tab its label's width plus the padding on both
	// sides. Tab i takes the whole pixels between the snapped edges i and i + 1, so a tab is never
	// off its exact place by as much as a pixel, however many tabs come before it.
	const float padding = frame.Scale().Px(theme.space.s5);
	float       edge    = 0.0f;
	for (int index = 0; index < static_cast<int>(labels.size()); index++) {
		Element         label;
		const TextProps measured {labels[index], theme.type.label, {}, TextAlign::Center};
		const float     natural   = MeasureTextLine(frame.Ui(), measured);
		const float     next_edge = edge + natural + 2.0f * padding;
		const float     width     = SnapPixel(next_edge) - SnapPixel(edge);
		edge                      = next_edge;
		const Element tab = MakeTab(frame, tokens, index, labels[index], index == chosen, width,
		                            natural, label, reason);
		if (tab.IsNone()) {
			return false;
		}
		stack.push_back(tab);
		stack.push_back(label);
	}
	return true;
}

// The table of handles a bar's callables reach and the row of tabs.
struct BuiltTabs {
	std::byte* table = nullptr;
	Element    row;
};

// Stores the table from the tabs and labels on the frame's stack from `base`, and builds the row.
// The row's children are the tabs again on the stack, which the caller gives back.
BuiltTabs MakeTabRow(UiFrame& frame, size_t base, int count, ElementError* reason) {
	std::vector<Element>& stack    = frame.ElementStack();
	const size_t          children = stack.size();
	for (int index = 0; index < count; index++) {
		stack.push_back(stack[base + 2 * static_cast<size_t>(index)]);
	}
	std::byte* table =
	    frame.StoreBytes(&stack[base], 2 * static_cast<size_t>(count) * sizeof(Element));
	const auto tabs = std::span<const Element>(stack).subspan(children, static_cast<size_t>(count));
	return {table, Box(frame, {.direction = FlexDirection::Row}, {}, tabs, reason)};
}

// What a bar asks of its container, `item` with the width left to the bar's first layer and the
// height of `tokens` when it gives none.
ItemLayout BarItem(const ItemLayout& item, const TabTokens& tokens) {
	ItemLayout bar_item = item;
	bar_item.width      = {};
	if (!bar_item.height) {
		bar_item.height = Dp(tokens.height);
	}
	return bar_item;
}

// What decides a bar's choice once every tab has taken its input. Focus moving between tabs, a
// confirm or a click, and the shoulders when the bar has them, choose a tab. Focus coming into the
// bar, or a choice by the shoulders while a tab has focus, is put on the chosen tab. The labels
// take the chosen colour and the change is reported, all before anything is painted. Trivially
// copyable so AfterInteract can store it.
struct TabChoice {
	const std::byte* tabs      = nullptr;
	int              count     = 0;
	int*             selected  = nullptr;
	bool*            changed   = nullptr;
	bool             shoulders = false;

	void operator()(UiFrame& frame, Element bar) const;
};

// What the tabs' input chose, from the choice given. Focus moving between tabs chooses the tab it
// lands on. A confirm or click chooses its tab, the last of them when several come on one frame.
// Returns the tab focus is on, or -1.
int ApplyTabInput(const UiFrame& frame, const TabChoice& choice, ItemId bar, int& selected) {
	int focused = -1;
	for (int index = 0; index < choice.count; index++) {
		const auto input = InputOf(frame, TabAt(choice.tabs, index));
		if (!input) {
			continue;
		}
		selected = input->interaction.state.activated ? index : selected;
		focused  = input->interaction.state.focused ? index : focused;
	}
	const auto& memory = frame.Store().List(bar);
	if (focused >= 0 && memory.inside && IdOf(frame, TabAt(choice.tabs, focused)) != memory.row) {
		selected = focused;
	}
	return focused;
}

void TabChoice::operator()(UiFrame& frame, Element bar) const {
	const ItemId bar_id = IdOf(frame, bar);
	const int    given  = *selected;
	const int    last   = count - 1;
	*selected           = std::clamp(*selected, 0, last);
	int focused         = ApplyTabInput(frame, *this, bar_id, *selected);
	if (shoulders) {
		// Read as the bar's own, since Render claimed the shoulders for it, and took the press from
		// anything drawn after the bar this frame when it reported it.
		const auto presses = KeyPressesOf(frame, bar);
		if (presses && (presses->shoulders.left || presses->shoulders.right)) {
			*selected = std::clamp(*selected + (presses->shoulders.right ? 1 : 0) -
			                           (presses->shoulders.left ? 1 : 0),
			                       0, last);
		}
	}
	if (focused >= 0 && focused != *selected) {
		// Remembered on the chosen tab, so focus arriving there isn't read as a move.
		FocusControl(IdOf(frame, TabAt(tabs, *selected)));
		focused = *selected;
	}
	auto& memory       = frame.Store().List(bar_id);
	memory.inside      = focused >= 0;
	memory.row         = focused >= 0 ? IdOf(frame, TabAt(tabs, focused)) : 0;
	const Theme& theme = frame.ActiveTheme();
	for (int index = 0; index < count; index++) {
		(void)SetTextColor(frame, LabelAt(tabs, index), LabelColor(theme, index == *selected));
	}
	const bool moved = *selected != given;
	if (changed != nullptr) {
		*changed = moved;
	}
}

} // namespace

Element TabOf(const TabBarElement& bar, int index) {
	if (bar.element.IsNone() || index < 0 || index >= bar.count) {
		return {};
	}
	return TabAt(bar.tabs, index);
}

TabTokens TabDefaults(const Theme& theme) {
	TabTokens tokens;
	tokens.height = ButtonDefaults(theme).height_medium;
	return tokens;
}

TabBarElement TabBar(UiFrame& frame, std::string_view key, std::span<const std::string_view> labels,
                     int* selected, const TabBarProps& props, const ItemLayout& item,
                     ElementError* reason) {
	if (props.changed != nullptr) {
		*props.changed = false;
	}
	if (labels.empty() || selected == nullptr) {
		RefuseBuild(labels.empty() ? ElementError::NoLabels : ElementError::NoValue, reason);
		return {};
	}
	const Theme&    theme  = frame.ActiveTheme();
	const TabTokens tokens = props.tokens.Apply(TabDefaults(theme));
	const int       count  = static_cast<int>(labels.size());
	const int       chosen = std::clamp(*selected, 0, count - 1);
	// The table of tab and label handles on the frame's stack, then the tabs again for the row's
	// children, all given back once the row is built.
	std::vector<Element>& stack = frame.ElementStack();
	const size_t          base  = stack.size();
	if (!PushTabs(frame, tokens, labels, chosen, reason)) {
		stack.resize(base);
		return {};
	}
	const BuiltTabs built = MakeTabRow(frame, base, count, reason);
	stack.resize(base);
	// The first layer only widens the bar to the width asked for, and the last is drawn over the
	// tabs.
	const ItemLayout             width_only {.width = item.width};
	const std::array<Element, 3> layers {
	    Spacer(frame, width_only, reason), built.row,
	    Custom(frame, {}, IndicatorPaint {built.table, count, selected, tokens}, reason)};
	const Element bar =
	    Box(frame, {.item = BarItem(item, tokens), .stacked = true}, {}, layers, reason);
	if (Keyed(frame, bar, key, reason).IsNone()) {
		return {};
	}
	if (props.shoulders && OwnKeys(frame, bar, {.shoulders = true}, reason).IsNone()) {
		return {};
	}
	const TabChoice choice {built.table, count, selected, props.changed, props.shoulders};
	if (AfterInteract(frame, bar, choice, reason).IsNone()) {
		return {};
	}
	return {bar, built.table, count};
}

} // namespace Libs::Graphics::SystemUi
