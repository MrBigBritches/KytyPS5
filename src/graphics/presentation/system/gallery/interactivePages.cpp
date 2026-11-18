#include "graphics/presentation/system/gallery/interactivePages.h"

#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/list.h"
#include "graphics/presentation/system/components/slider.h"
#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/toggle.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/stage.h"

#include <array>
#include <cmath>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view BUTTON_RULE =
    "One Primary per screen. Ghost for Cancel. Danger only for data loss, with focus on Cancel. "
    "Activates on press. A disabled button fades.";
constexpr std::string_view TOGGLE_RULE =
    "For an on or off setting that applies at once. A disabled toggle fades.";
constexpr std::string_view SLIDER_RULE =
    "For a range the player tunes by feel. Show the value. A disabled slider is solid grey.";
constexpr std::string_view LIST_RULE =
    "For choosing among a few to many rows. Selected shows the current choice. Focus shows where "
    "the pad is. A disabled row fades. A list that doesn't fit shows a scroll bar.";
constexpr std::string_view TABS_RULE =
    "For peer views of one subject, three to six of them. L1 and R1 switch.";

// The matrices, in reference pixels.
constexpr float BUTTON_CELL       = 170.0f;
constexpr float TOGGLE_CELL       = 170.0f;
constexpr float SLIDER_CELL       = 260.0f;
constexpr float LIVE_SLIDER_WIDTH = 400.0f;
constexpr float LIST_CELL         = 440.0f;
constexpr float LIVE_LIST_WIDTH   = 560.0f;
constexpr int   LIVE_LIST_ROWS    = 4;
constexpr float SLIDER_STEP       = 0.01f;
constexpr float PERCENT           = 100.0f;
// The value label's room, for the widest it says.
constexpr std::string_view LABEL_RESERVE = "100%";

// A state a control is posed in, and whether it is disabled.
struct PosedState {
	std::string_view name;
	ItemVisuals      pose;
	bool             disabled = false;
};

constexpr std::array<PosedState, 5> STATES {
    PosedState {"Default", {}}, PosedState {"Hover", {.hover = 1.0f}},
    PosedState {"Focused", {.focus = 1.0f}}, PosedState {"Pressed", {.press = 1.0f}},
    PosedState {"Disabled", {}, true}};
// A row's hover, focus and press are shown by the live list, so its matrix has only these.
constexpr std::array<PosedState, 2> LIST_STATES {PosedState {"Default", {}},
                                                 PosedState {"Disabled", {}, true}};

constexpr std::array<std::string_view, 3> LIVE_TAB_NAMES {"First", "Second", "Third"};

// A caption in `ink`.
Element Words(UiFrame& frame, std::string_view text, Color ink, const ItemLayout& item = {}) {
	return Text(frame, {text, frame.ActiveTheme().type.caption, ink}, item);
}

// A column of `rows` with `gap` between them.
Element Column(UiFrame& frame, std::span<const Element> rows, float gap) {
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(gap)}, {}, rows);
}

// A row of the matrix, its name and its cells.
Element MatrixRow(UiFrame& frame, std::string_view name, std::span<const Element> cells,
                  Color ink) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> parts {Words(frame, name, ink, {.width = Dp(STAGE_ROW_LABEL_WIDTH)})};
	parts.insert(parts.end(), cells.begin(), cells.end());
	return Box(frame, {.align = Align::Center, .gap = Dp(theme.space.s3)}, {}, parts);
}

// The matrix's head, each state's name over its column.
Element MatrixHead(UiFrame& frame, std::span<const PosedState> states, float cell, Color ink) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> parts {Spacer(frame, {.width = Dp(STAGE_ROW_LABEL_WIDTH)})};
	for (const PosedState& state: states) {
		parts.push_back(Words(frame, state.name, ink, {.width = Dp(cell)}));
	}
	return Box(frame, {.gap = Dp(theme.space.s3)}, {}, parts);
}

// A matrix's rows, keyed as a root, as a page's live copy is a root of the same frame too.
Element Specimens(UiFrame& frame, std::span<const Element> rows) {
	return Keyed(frame, Column(frame, rows, frame.ActiveTheme().space.s3), "specimens");
}

// ---- Button -------------------------------------------------------------------------------------

// A variant and what it is called.
struct NamedVariant {
	std::string_view name;
	ButtonVariant    variant;
};

constexpr std::array<NamedVariant, 4> VARIANTS {
    NamedVariant {"Primary", ButtonVariant::Primary},
    NamedVariant {"Secondary", ButtonVariant::Secondary},
    NamedVariant {"Ghost", ButtonVariant::Ghost}, NamedVariant {"Danger", ButtonVariant::Danger}};

// Whether the backdrop chosen is the light one, which a Ghost button has to be told.
bool OnLightBackdrop(const GalleryState& state) {
	return state.stage.backdrop == STAGE_BACKDROP_BRIGHT;
}

// The button page's specimens.
Element BuildButtons(UiFrame& frame, GalleryState& state) {
	const Color          ink = StageInk(frame.ActiveTheme(), state.stage);
	std::vector<Element> rows {MatrixHead(frame, STATES, BUTTON_CELL, ink)};
	for (const NamedVariant& named: VARIANTS) {
		std::vector<Element> cells;
		for (const PosedState& posed: STATES) {
			const ButtonProps props {.label    = named.name,
			                         .variant  = named.variant,
			                         .disabled = posed.disabled,
			                         .on_light = OnLightBackdrop(state)};
			const Element     button = Button(frame, posed.name, props, {.width = Dp(BUTTON_CELL)});
			cells.push_back(Specimen(frame, button, posed.pose));
		}
		rows.push_back(Keyed(frame, MatrixRow(frame, named.name, cells, ink), named.name));
	}
	return Specimens(frame, rows);
}

// Buttons to try.
Element LiveButtons(UiFrame& frame, GalleryState& state) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> buttons;
	for (const NamedVariant& named: VARIANTS) {
		buttons.push_back(Button(
		    frame, named.name,
		    {.label = named.name, .variant = named.variant, .on_light = OnLightBackdrop(state)}));
	}
	return Keyed(frame, Box(frame, {.gap = Dp(theme.space.s3)}, {}, buttons), "live");
}

// ---- Toggle -------------------------------------------------------------------------------------

// One row of toggles at `on`, which nothing changes, as a specimen takes no input.
Element ToggleRow(UiFrame& frame, std::string_view name, bool* on, Color ink) {
	std::vector<Element> cells;
	for (const PosedState& posed: STATES) {
		const std::array<Element, 2> parts {
		    Toggle(frame, posed.name, on, {.disabled = posed.disabled}),
		    Spacer(frame, {.width = Dp(TOGGLE_CELL - ToggleDefaults(frame.ActiveTheme()).width)})};
		const Element cell = Box(frame, {}, {}, parts);
		cells.push_back(Specimen(frame, cell, posed.pose));
	}
	return Keyed(frame, MatrixRow(frame, name, cells, ink), name);
}

// The toggle page's specimens.
Element BuildToggles(UiFrame& frame, GalleryState& state) {
	const Color                  ink = StageInk(frame.ActiveTheme(), state.stage);
	const std::array<Element, 3> rows {MatrixHead(frame, STATES, TOGGLE_CELL, ink),
	                                   ToggleRow(frame, "Off", &state.interactive.off, ink),
	                                   ToggleRow(frame, "On", &state.interactive.on, ink)};
	return Specimens(frame, rows);
}

// A toggle to try.
Element LiveToggle(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 2> parts {
	    Toggle(frame, "live toggle", &state.interactive.live_toggle),
	    Words(frame, state.interactive.live_toggle ? "Now on" : "Now off",
	          StageInk(theme, state.stage))};
	return Keyed(frame, Box(frame, {.align = Align::Center, .gap = Dp(theme.space.s3)}, {}, parts),
	             "live");
}

// ---- Slider -------------------------------------------------------------------------------------

// A share of the way along a slider, in words.
std::string PercentOf(float value) {
	return std::to_string(std::lround(value * PERCENT)) + "%";
}

// The slider page's specimens, a row for each value, each saying it.
Element BuildSliders(UiFrame& frame, GalleryState& state) {
	const Color          ink = StageInk(frame.ActiveTheme(), state.stage);
	std::vector<Element> rows {MatrixHead(frame, STATES, SLIDER_CELL, ink)};
	for (float& value: state.interactive.slider_values) {
		const std::string    name = PercentOf(value);
		std::vector<Element> cells;
		for (const PosedState& posed: STATES) {
			const SliderProps props {.min           = 0.0f,
			                         .max           = 1.0f,
			                         .step          = SLIDER_STEP,
			                         .value_label   = name,
			                         .label_reserve = LABEL_RESERVE,
			                         .disabled      = posed.disabled};
			const SliderParts slider =
			    Slider(frame, posed.name, &value, props, {.width = Dp(SLIDER_CELL)});
			cells.push_back(Specimen(frame, slider.root, posed.pose));
		}
		rows.push_back(Keyed(frame, MatrixRow(frame, name, cells, ink), name));
	}
	return Specimens(frame, rows);
}

// A slider to try.
Element LiveSlider(UiFrame& frame, GalleryState& state) {
	const std::string percent = PercentOf(state.interactive.live_slider);
	const SliderProps props {.min           = 0.0f,
	                         .max           = 1.0f,
	                         .step          = SLIDER_STEP,
	                         .value_label   = percent,
	                         .label_reserve = LABEL_RESERVE};
	return Keyed(frame,
	             Slider(frame, "live slider", &state.interactive.live_slider, props,
	                    {.width = Dp(LIVE_SLIDER_WIDTH)})
	                 .root,
	             "live");
}

// ---- List ---------------------------------------------------------------------------------------

// A kind of row and what it is called.
struct NamedRow {
	std::string_view name;
	std::string_view primary;
	std::string_view secondary;
	bool             selected = false;
};

constexpr std::array<NamedRow, 3> ROWS {NamedRow {"Single", "One line", {}},
                                        NamedRow {"Double", "Two lines", "With a smaller second"},
                                        NamedRow {"Selected", "The current choice", {}, true}};

// The list page's specimens.
Element BuildListRows(UiFrame& frame, GalleryState& state) {
	const Color          ink = StageInk(frame.ActiveTheme(), state.stage);
	std::vector<Element> rows {MatrixHead(frame, LIST_STATES, LIST_CELL, ink)};
	for (const NamedRow& kind: ROWS) {
		std::vector<Element> cells;
		for (const PosedState& posed: LIST_STATES) {
			ListRowProps props {kind.primary, kind.secondary, kind.selected, posed.disabled};
			props.on_light = OnLightBackdrop(state);
			const std::array<Element, 1> inside {ListRow(frame, posed.name, props)};
			const Element                cell = Box(frame,
			                                        {.direction = FlexDirection::Column,
			                                         .align     = Align::Stretch,
			                                         .item      = {.width = Dp(LIST_CELL)}},
			                                        {}, inside);
			cells.push_back(Specimen(frame, cell, posed.pose));
		}
		rows.push_back(Keyed(frame, MatrixRow(frame, kind.name, cells, ink), kind.name));
	}
	return Specimens(frame, rows);
}

// Chooses the row that was activated. Trivially copyable so OnActivate can store it.
struct ChooseRow {
	int* chosen;
	int  row;

	void operator()(Element /*element*/) const { *chosen = row; }
};

// A list to try, which remembers the row chosen.
Element LiveList(UiFrame& frame, GalleryState& state) {
	const ListTokens     tokens = ListDefaults(frame.ActiveTheme());
	std::vector<Element> rows;
	for (int index = 0; index < LIVE_LIST_ROWS; index++) {
		const std::string name = "Row " + std::to_string(index + 1);
		ListRowProps      props {name, {}, state.interactive.live_row == index};
		props.on_light    = OnLightBackdrop(state);
		const Element row = ListRow(frame, name, props);
		rows.push_back(OnActivate(frame, row, ChooseRow {&state.interactive.live_row, index}));
	}
	// The list's own insets are the focus ring's, so a little more than its rows' height holds
	// them.
	constexpr float RING_ROOM = 16.0f;
	return List(frame, "live", rows,
	            {.width  = Dp(LIVE_LIST_WIDTH),
	             .height = Dp(tokens.row_single * static_cast<float>(LIVE_LIST_ROWS) + RING_ROOM)});
}

// ---- Tabs ---------------------------------------------------------------------------------------

// What the page says in place of a matrix.
Element BuildTabsNote(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 1> rows {
	    Words(frame, "Tabs have no disabled state.", StageInk(theme, state.stage))};
	return Specimens(frame, rows);
}

// A tab bar to try.
Element LiveTabs(UiFrame& frame, GalleryState& state) {
	return Keyed(
	    frame,
	    TabBar(frame, "tabs", LIVE_TAB_NAMES, &state.interactive.live_tab, {.shoulders = true})
	        .element,
	    "live");
}

} // namespace

void DrawButtonPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, BUTTON_RULE, BuildButtons, pane, LiveButtons);
}

void DrawTogglePage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, TOGGLE_RULE, BuildToggles, pane, LiveToggle);
}

void DrawSliderPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, SLIDER_RULE, BuildSliders, pane, LiveSlider);
}

void DrawListPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, LIST_RULE, BuildListRows, pane, LiveList);
}

void DrawTabsPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, TABS_RULE, BuildTabsNote, pane, LiveTabs);
}

} // namespace Libs::Graphics::SystemUi::Gallery
