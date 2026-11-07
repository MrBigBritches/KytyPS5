#include "graphics/presentation/system/gallery/toastPage.h"

#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/imgui/interaction.h"
#include "imgui.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

// The grid's columns and the room around a toast, in reference pixels. A cell leaves 40 pixels each
// side of the 560 pixel toast and 32 above and below, so the glow's 16 pixel spill and the toast's
// shadow both show inside the cell.
constexpr float LABEL_WIDTH    = 240.0f;
constexpr float CELL_WIDTH     = 640.0f;
constexpr float CELL_PADDING_X = 40.0f;
constexpr float CELL_PADDING_Y = 32.0f;
// The stage holds the toast's 32 pixel slide above where it rests, and its shadow below.
constexpr float STAGE_HEIGHT = 192.0f;
constexpr float STAGE_TOP    = 40.0f;
// A caption fits the label column in this many lines, and stays shorter than a stage.
constexpr int CAPTION_LINES = 4;
// The live toast holds long enough to judge its glow and shine, then the empty stage makes the next
// entrance read as a new event.
constexpr float LIVE_SECONDS  = 4.0f;
constexpr auto  REPLAY_GAP    = std::chrono::milliseconds(1500);
constexpr int   MAX_ICON_SIDE = 512;

// The trophy grades in tab order. libTrophy numbers them from 1 in the same order.
constexpr std::array<std::string_view, 4> GRADE_NAMES {"Platinum", "Gold", "Silver", "Bronze"};

constexpr std::string_view RULE =
    "Use for temporary event messages that do not require a response.";
// Stands in for the subtitle a game's language would give.
constexpr std::string_view SAMPLE_SUBTITLE = "Trophy earned!";

// The backdrops in column order. Dark is what a player mostly sees. Bright is the worst case for
// the toast's dark panel and glow, so a faint glow there is a finding and not a fault.
constexpr std::array<Color, TOAST_BACKDROPS> BACKDROPS {Palette::GRAY_50, Palette::GRAY_800};
constexpr std::array<std::string_view, TOAST_BACKDROPS> BACKDROP_NAMES {"Dark backdrop",
                                                                        "Bright backdrop"};

// The title every toast on the page shows for `grade`.
std::string SampleTitle(int grade) {
	return std::string(GRADE_NAMES[grade]) + " sample trophy";
}

// The page's elements that are needed again after they are built.
struct PageTree {
	Element       root;
	TabBarElement tabs;
	Element       replay;
	Element       stage;
	std::array<Element, TOAST_BACKDROPS> cards {};
};

// A row of the grid's label column, a bold name with a wrapped caption under it.
Element LabelColumn(UiFrame& frame, std::string_view label, std::string_view caption) {
	const Theme&          theme = frame.ActiveTheme();
	const TextProps       name {label, theme.type.label, theme.colors.text.primary};
	const TextProps       note {caption,          theme.type.caption,
                          theme.colors.text.tertiary, TextAlign::Start,
                          TextOverflow::Wrap, CAPTION_LINES};
	std::array<Element, 2> lines {Text(frame, name), Text(frame, note)};
	return Box(frame,
	           {.direction = FlexDirection::Column,
	            .align     = Align::Stretch,
	            .gap       = Dp(theme.space.s1),
	            .padding   = {Dp(), Dp(), Dp(theme.space.s4), Dp()},
	            .item      = {.width = Dp(LABEL_WIDTH)}},
	           {}, lines);
}

// The tabs and the Replay button, with the button at the far end of the grid.
Element BuildControls(UiFrame& frame, ToastPageState& state, PageTree& tree) {
	tree.tabs = TabBar(frame, "grades", GRADE_NAMES, &state.grade, {.changed = &state.grade_changed});
	tree.replay =
	    OnActivate(frame, Button(frame, "replay", {.label = "Replay"}),
	               [pressed = &state.replay_pressed](Element /*button*/) { *pressed = true; });
	const std::array<Element, 3> parts {tree.tabs.element, Spacer(frame, {.grow = 1.0f}),
	                                    tree.replay};
	return Box(frame,
	           {.item = {.width = Dp(LABEL_WIDTH + TOAST_BACKDROPS * CELL_WIDTH)}}, {}, parts);
}

// The Live row, its label and the empty stage the view draws on afterwards. The stage lines up with
// the dark column below, so the live toast sits exactly over the held one.
Element BuildLiveRow(UiFrame& frame, PageTree& tree) {
	const Theme& theme = frame.ActiveTheme();
	tree.stage         = Box(frame,
	                         {.item = {.width = Dp(CELL_WIDTH), .height = Dp(STAGE_HEIGHT)}},
	                         {BACKDROPS[0], theme.colors.border.standard, Dp(theme.border.hair), Dp()});
	const std::array<Element, 2> parts {
	    LabelColumn(frame, "Live", "Slides in, fades and shines, then plays again."),
	    tree.stage};
	return Box(frame, {}, {}, parts);
}

// The header over the held toasts, each backdrop's name over its column.
Element BuildHeader(UiFrame& frame) {
	const Theme&         theme = frame.ActiveTheme();
	std::vector<Element> parts {Spacer(frame, {.width = Dp(LABEL_WIDTH)})};
	for (const std::string_view name: BACKDROP_NAMES) {
		const TextProps heading {name, theme.type.label, theme.colors.text.primary,
		                         TextAlign::Center};
		parts.push_back(Text(frame, heading, {.width = Dp(CELL_WIDTH)}));
	}
	return Box(frame, {.align = Align::Center}, {}, parts);
}

// The toast held still, with no shine, on each backdrop, to judge its glow and shadow.
Element BuildHeldRow(UiFrame& frame, const GalleryState& state, PageTree& tree) {
	const Theme&         theme = frame.ActiveTheme();
	const std::string    title = SampleTitle(state.toast.grade);
	std::vector<Element> parts {
	    LabelColumn(frame, "Held still", "32 px glow spills 16 px past the card. No shine.")};
	for (size_t backdrop = 0; backdrop < TOAST_BACKDROPS; backdrop++) {
		const Element card =
		    Toast(frame, SampleToast(frame, state, title), frame.Scale().ContentFrame().width).root;
		tree.cards[backdrop] = card;
		const std::array<Element, 1> content {card};
		parts.push_back(
		    Box(frame,
		        {.justify = Justify::Center,
		         .align   = Align::Center,
		         .padding = DpInsets::Symmetric(Dp(CELL_PADDING_X), Dp(CELL_PADDING_Y)),
		         .item    = {.width = Dp(CELL_WIDTH)}},
		        {BACKDROPS[backdrop], theme.colors.border.standard, Dp(theme.border.hair), Dp()},
		        content));
	}
	return Box(frame, {}, {}, parts);
}

// Builds the page apart from the live toast, which the view builds for itself.
PageTree BuildPage(UiFrame& frame, GalleryState& state, const PageEntry& page) {
	const Theme& theme = frame.ActiveTheme();
	PageTree     tree;
	const TextProps title {page.title, theme.type.title, theme.colors.text.primary};
	const TextProps              note {RULE, theme.type.body_small, theme.colors.text.secondary};
	const std::array<Element, 6> rows {
	    Text(frame, title),        Text(frame, note),  BuildControls(frame, state.toast, tree),
	    BuildLiveRow(frame, tree), BuildHeader(frame), BuildHeldRow(frame, state, tree)};
	tree.root = Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s4)}, {}, rows);
	return tree;
}

// Notes where the page's parts landed.
ToastPageLayout LayoutOf(const UiFrame& frame, const PageTree& tree) {
	ToastPageLayout layout;
	layout.stage  = Arranged(frame, tree.stage);
	layout.replay = Arranged(frame, tree.replay);
	for (size_t backdrop = 0; backdrop < TOAST_BACKDROPS; backdrop++) {
		layout.cards[backdrop] = Arranged(frame, tree.cards[backdrop]);
	}
	return layout;
}

// Starts the live toast again for the chosen grade.
void ShowLive(UiFrame& frame, GalleryState& state) {
	TrophyUnlock unlock {SampleTitle(state.toast.grade), static_cast<int32_t>(state.toast.grade + 1),
	                     {state.sample_icon.begin(), state.sample_icon.end()}};
	state.toast.live->Show(std::move(unlock), std::string(SAMPLE_SUBTITLE), frame.ActiveTheme(),
	                       {LIVE_SECONDS}, frame.Now());
}

// Plays the real trophy toast on the stage. It starts again when the page comes back into view, as
// the store forgets a toast a frame skips, and when the player asks or changes the grade. Once it
// has left, the stage stays empty for a moment before it plays again.
void DrawLive(UiFrame& frame, GalleryState& state, const Rect& stage) {
	if (state.images == nullptr) {
		return;
	}
	ToastPageState& toast = state.toast;
	if (!toast.live.has_value()) {
		toast.live.emplace(*state.images);
	}
	const int  imgui_frame = ImGui::GetFrameCount();
	const bool came_back   = toast.drawn_in_imgui_frame != imgui_frame - 1;
	const bool due         = toast.live->Idle() && frame.Now() >= toast.next_show;
	toast.drawn_in_imgui_frame = imgui_frame;
	if (came_back || due || toast.replay_pressed || toast.grade_changed) {
		ShowLive(frame, state);
	}
	const auto& scale = frame.Scale();
	const Rect  area  = stage.Inset(
        {scale.Px(CELL_PADDING_X), scale.Px(STAGE_TOP), scale.Px(CELL_PADDING_X), 0.0f});
	const bool was_shown = !toast.live->Idle();
	(void)toast.live->Draw(frame, CurrentDrawList(), area);
	if (was_shown && toast.live->Idle()) {
		toast.next_show = frame.Now() + std::chrono::duration_cast<UiClock::duration>(REPLAY_GAP);
	}
}

} // namespace

void ToastPageState::Reset() {
	grade                = 0;
	live.reset();
	next_show            = {};
	drawn_in_imgui_frame = -1;
	replay_pressed       = false;
	grade_changed        = false;
	layout               = {};
}

ToastProps SampleToast(const UiFrame& frame, const GalleryState& state, const std::string& title) {
	ToastProps props {title, SAMPLE_SUBTITLE};
	if (state.images != nullptr && state.toast.icon.has_value()) {
		props.icon      = state.images->Texture(*state.toast.icon);
		props.icon_size = state.images->Size(*state.toast.icon);
	}
	const Color grade = TrophyGradeColor(frame.ActiveTheme(), state.toast.grade + 1);
	props.icon_glow   = grade;
	props.title_cup   = grade;
	return props;
}

void LoadSampleIcon(GalleryState& state) {
	ToastPageState& toast = state.toast;
	// Once only, since a bad icon would otherwise fail and log on every frame.
	if (state.images != nullptr && !toast.icon_tried && !state.sample_icon.empty()) {
		toast.icon_tried = true;
		toast.icon       = state.images->Add(state.sample_icon, MAX_ICON_SIDE);
	}
}

void DrawToastPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ToastPageState& toast = state.toast;
	toast.grade           = std::clamp(toast.grade, 0, static_cast<int>(GRADE_NAMES.size()) - 1);
	toast.replay_pressed  = false;
	LoadSampleIcon(state);
	const PageTree tree = BuildPage(frame, state, page);
	(void)AssignIds(frame, tree.root, CurrentIdSeed());
	const float unlimited = std::numeric_limits<float>::infinity();
	const Vec2  size      = Measure(frame, tree.root, {unlimited, unlimited});
	(void)Arrange(frame, tree.root, {pane.x, pane.y, size.x, size.y});
	// Focus coming in from the sidebar lands on the chosen grade.
	state.entry = IdOf(frame, TabOf(tree.tabs, toast.grade));
	(void)Render(frame, tree.root, CurrentDrawList());
	toast.layout = LayoutOf(frame, tree);
	DrawLive(frame, state, toast.layout.stage);
}

} // namespace Libs::Graphics::SystemUi::Gallery
