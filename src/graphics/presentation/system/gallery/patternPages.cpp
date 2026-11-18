#include "graphics/presentation/system/gallery/patternPages.h"

#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/card.h"
#include "graphics/presentation/system/components/hud.h"
#include "graphics/presentation/system/components/modal.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/stage.h"
#include "graphics/presentation/system/gallery/toastPage.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "graphics/presentation/system/layout/flex.h"
#include "graphics/presentation/system/views/trophyToast.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

namespace {

constexpr std::string_view METRIC_RULE = "Use for read-only numbers. Always include a word with "
                                         "the colour. Use the secondary text colour "
                                         "or brighter on a HUD.";
constexpr std::string_view NOTIFICATION_RULE = "Use for status that can be ignored. It never takes "
                                               "focus. One shows at a time, and others wait "
                                               "their turn.";
constexpr std::string_view DIALOGS_RULE =
    "Use when the game cannot go on until the player answers. Focus stays inside. Back cancels, "
    "unless the work can't be stopped. A long body scrolls.";

// The views, as indexes into DIALOG_VIEWS.
constexpr int VIEW_CLOSED   = 0;
constexpr int VIEW_CONFIRM  = 1;
constexpr int VIEW_PROGRESS = 2;
constexpr int VIEW_SAVING   = 3;
constexpr int VIEW_ERROR    = 4;
constexpr int VIEW_LONG     = 5;

// How far a dialog stays from the panel's edges, in pixels on screen. It isn't scaled, as the panel
// doesn't grow with the output chosen, and a margin that did would squeeze a 4K card.
constexpr float PANEL_MARGIN = 16.0f;
// How long the sample work takes, in seconds.
constexpr float INSTALL_SECONDS = 8.0f;
constexpr float SAVE_SECONDS    = 5.0f;
// How many times the long dialog says its sentence, more than a card on any output holds.
constexpr int LONG_TEXT_REPEATS = 40;

constexpr std::string_view ANSWER_CANCELLED = "Cancelled";
constexpr std::string_view ANSWER_DELETED   = "Deleted";
constexpr std::string_view ANSWER_FINISHED  = "Finished";
constexpr std::string_view ANSWER_RETRY     = "Retry";
constexpr std::string_view ANSWER_CLOSED    = "Closed";

// An opener and what it says about its dialog.
struct DialogOpener {
	std::string_view label;
	int              view;
	std::string_view note;
};

constexpr std::array<DialogOpener, 5> OPENERS {
    DialogOpener {"Confirm delete", VIEW_CONFIRM, "Danger asks first, with focus on Cancel."},
    DialogOpener {
        "Progress", VIEW_PROGRESS,
        "Work that can be stopped. Cancelling asks first, and the work goes on meanwhile."},
    DialogOpener {
        "Saving", VIEW_SAVING,
        "Work that can't be stopped. No Cancel, Back does nothing, and it closes itself."},
    DialogOpener {"Error with detail", VIEW_ERROR,
                  "Detail sits below the answer, small and quiet."},
    DialogOpener {"Long text", VIEW_LONG, "A long body scrolls, and the buttons stay."}};

// The name of the popup a view opens, one each, so opening one never finds another's state.
const char* PopupName(int view) {
	switch (view) {
		case VIEW_CONFIRM: return "confirm";
		case VIEW_PROGRESS: return "progress";
		case VIEW_SAVING: return "saving";
		case VIEW_ERROR: return "error";
		default: return "long text";
	}
}

// Opens a dialog when its button is activated. Trivially copyable so OnActivate can store it.
struct OpenDialog {
	int* view;
	int  chosen;

	void operator()(Element /*element*/) const { *view = chosen; }
};

// Whether `button` was activated in the Render just done.
bool Activated(const UiFrame& frame, Element button) {
	const auto input = InputOf(frame, button);
	return input && input->interaction.state.activated;
}

// Ends the dialog that's showing with `answer`.
void Answer(GalleryState& state, std::string_view answer) {
	state.dialogs.answer = answer;
	state.dialogs_view   = VIEW_CLOSED;
}

// The card for `parts`, sized as a real dialog's at the output chosen, but in the middle of `panel`
// and held inside it, as the panel is what this dialog overlays.
Rect PanelCard(UiFrame& frame, const DialogParts& parts, const Rect& panel) {
	const Rect  fitted = FitDialogCard(frame, parts);
	const float width  = std::min(fitted.width, std::max(0.0f, panel.width - 2.0f * PANEL_MARGIN));
	const float height =
	    std::min(fitted.height, std::max(0.0f, panel.height - 2.0f * PANEL_MARGIN));
	return {SnapPixel(panel.x + (panel.width - width) / 2.0f),
	        SnapPixel(panel.y + (panel.height - height) / 2.0f), width, height};
}

// A dialog's content with something beneath its body, which DialogContent's text-only body can't
// have, and `buttons` along the bottom if there are any. MeasureDialog and Place read only
// `content`.
DialogParts DialogWith(UiFrame& frame, std::string_view title, std::string_view body, Element extra,
                       std::span<const Element> buttons) {
	const Theme& theme = frame.ActiveTheme();
	const Dp     gap {theme.space.s4};
	TextProps    wrapped {body, theme.type.body, theme.colors.text.secondary};
	wrapped.overflow = TextOverflow::Wrap;
	std::vector<Element> rows {Text(frame, {title, theme.type.heading, theme.colors.text.primary}),
	                           Text(frame, wrapped), extra};
	if (!buttons.empty()) {
		rows.push_back(Box(frame, {.justify = Justify::End, .gap = gap}, {}, buttons));
	}
	DialogParts parts;
	parts.content =
	    Keyed(frame,
	          Box(frame, {.direction = FlexDirection::Column, .align = Align::Stretch, .gap = gap},
	              {}, rows),
	          "content");
	return parts;
}

// The dialog that asks before stopping work, inside the dialog of the work it would stop, as a
// dialog opened from another has to be. It stays while `open` is false for the frame it closes in.
// Back is the outer dialog's to read, which has already taken it.
void AskToStop(UiFrame& frame, GalleryState& state, const Rect& panel, float opacity) {
	auto&         dialogs = state.dialogs;
	const Element keep    = Button(
        frame, "keep",
        {.label = "Keep installing", .variant = ButtonVariant::Ghost, .default_focus = true});
	const Element stop =
	    Button(frame, "stop", {.label = "Cancel install", .variant = ButtonVariant::Danger});
	const std::array<Element, 2> buttons {keep, stop};
	const DialogParts            parts = DialogContent(
        frame, "ask", {"Cancel the install?", "What's installed so far will be removed."}, buttons);
	const Rect card = PanelCard(frame, parts, panel);
	const bool open = dialogs.asking_to_stop;
	ModalScope modal(frame, "ask to stop", card, open, open ? opacity : 0.0f, {}, panel);
	dialogs.asked_to_stop = open;
	if (!modal.Visible() || !open) {
		return;
	}
	(void)modal.Place(frame, parts);
	(void)Render(frame, parts.content, modal.List(), {.opacity = opacity});
	if (Activated(frame, keep)) {
		dialogs.asking_to_stop = false;
	}
	if (Activated(frame, stop)) {
		Answer(state, ANSWER_CANCELLED);
	}
}

// How far along the sample work is, from when its dialog opened, where 1 is finished.
float WorkDone(const UiFrame& frame, const GalleryState& state, float seconds) {
	const std::chrono::duration<float> elapsed = frame.Now() - state.dialogs.opened;
	return elapsed.count() / seconds;
}

// A dialog over `panel` showing work's progress, which closes itself when done. When `can_stop`, it
// has a Cancel, which Back presses too, and asks before stopping, with the work going on meanwhile.
// Otherwise nothing stops it, which is why it has no buttons and ignores Back.
void ShowWork(UiFrame& frame, GalleryState& state, const Rect& panel, float opacity,
              const char* name, std::string_view title, std::string_view body, float seconds,
              bool can_stop) {
	auto&         dialogs = state.dialogs;
	const float   done    = WorkDone(frame, state, seconds);
	const Element bar     = ProgressBar(
        frame, {.value = done, .fill = frame.ActiveTheme().colors.accent.line}, {.grow = 0.0f});
	std::vector<Element> buttons;
	if (can_stop) {
		buttons.push_back(
		    Button(frame, "cancel",
		           {.label = "Cancel", .variant = ButtonVariant::Ghost, .default_focus = true}));
	}
	const DialogParts parts = DialogWith(frame, title, body, bar, buttons);
	ModalScope        modal(frame, name, PanelCard(frame, parts, panel), true, opacity, {}, panel);
	if (!modal.Visible()) {
		return;
	}
	(void)modal.Place(frame, parts);
	(void)Render(frame, parts.content, modal.List(), {.opacity = opacity});
	if (can_stop) {
		if (modal.BackPressed()) {
			dialogs.asking_to_stop = !dialogs.asking_to_stop;
		} else if (Activated(frame, buttons[0])) {
			dialogs.asking_to_stop = true;
		}
		if (dialogs.asking_to_stop || dialogs.asked_to_stop) {
			AskToStop(frame, state, panel, opacity);
		}
	}
	if (done >= 1.0f) {
		Answer(state, ANSWER_FINISHED);
	}
}

// A dialog over `panel` with `parts` in it, which Back answers with `on_back`. Returns whether it's
// showing, so the caller can read what was pressed.
bool ShowAnswerable(UiFrame& frame, GalleryState& state, const Rect& panel, float opacity,
                    const char* name, const DialogParts& parts, std::string_view on_back) {
	ModalScope modal(frame, name, PanelCard(frame, parts, panel), true, opacity, {}, panel);
	if (!modal.Visible()) {
		return false;
	}
	(void)modal.Place(frame, parts);
	(void)Render(frame, parts.content, modal.List(), {.opacity = opacity});
	if (modal.BackPressed()) {
		Answer(state, on_back);
		return false;
	}
	return true;
}

// The confirmation dialog, whose first focus is on Cancel, as the other answer loses data.
void ShowConfirm(UiFrame& frame, GalleryState& state, const Rect& panel, float opacity) {
	const Element cancel =
	    Button(frame, "cancel",
	           {.label = "Cancel", .variant = ButtonVariant::Ghost, .default_focus = true});
	const Element delete_it =
	    Button(frame, "delete", {.label = "Delete", .variant = ButtonVariant::Danger});
	const std::array<Element, 2> buttons {cancel, delete_it};
	const DialogParts            parts =
	    DialogContent(frame, "confirm", {"Delete this save?", "This can't be undone."}, buttons);
	if (!ShowAnswerable(frame, state, panel, opacity, PopupName(VIEW_CONFIRM), parts,
	                    ANSWER_CANCELLED)) {
		return;
	}
	if (Activated(frame, cancel)) {
		Answer(state, ANSWER_CANCELLED);
	} else if (Activated(frame, delete_it)) {
		Answer(state, ANSWER_DELETED);
	}
}

// The error dialog, with a detail line under what to do about it.
void ShowError(UiFrame& frame, GalleryState& state, const Rect& panel, float opacity) {
	const Theme&  theme = frame.ActiveTheme();
	const Element retry =
	    Button(frame, "retry",
	           {.label = "Retry", .variant = ButtonVariant::Primary, .default_focus = true});
	const Element close =
	    Button(frame, "close", {.label = "Close", .variant = ButtonVariant::Ghost});
	const std::array<Element, 2> buttons {close, retry};
	const Element                detail =
	    Text(frame, {"Error code CE-10005-6", theme.type.caption, theme.colors.text.tertiary});
	const DialogParts parts =
	    DialogWith(frame, "Couldn't load the save",
	               "Check that the storage is connected, then try again.", detail, buttons);
	if (!ShowAnswerable(frame, state, panel, opacity, PopupName(VIEW_ERROR), parts,
	                    ANSWER_CLOSED)) {
		return;
	}
	if (Activated(frame, retry)) {
		Answer(state, ANSWER_RETRY);
	} else if (Activated(frame, close)) {
		Answer(state, ANSWER_CLOSED);
	}
}

// A dialog with more to say than any card holds, which is DialogContent's to cut short.
void ShowLongText(UiFrame& frame, GalleryState& state, const Rect& panel, float opacity) {
	std::string body;
	for (int repeat = 0; repeat < LONG_TEXT_REPEATS; repeat++) {
		body += "This game saves on its own, and shows an icon while it does. ";
	}
	const Element close =
	    Button(frame, "close",
	           {.label = "Close", .variant = ButtonVariant::Primary, .default_focus = true});
	const std::array<Element, 1> buttons {close};
	const DialogParts parts = DialogContent(frame, "long", {"About autosave", body}, buttons);
	if (ShowAnswerable(frame, state, panel, opacity, PopupName(VIEW_LONG), parts, ANSWER_CLOSED) &&
	    Activated(frame, close)) {
		Answer(state, ANSWER_CLOSED);
	}
}

// Shows the dialog of the view chosen, over `panel`, in the stage's frame, so it's at the output
// chosen. One no longer chosen is closed first, with nothing of it showing in the frame it closes
// in.
void DrawDialog(UiFrame& frame, GalleryState& state, const Rect& panel) {
	auto&     dialogs = state.dialogs;
	const int view    = state.dialogs_view;
	if (dialogs.showing != VIEW_CLOSED && dialogs.showing != view) {
		const ModalScope closing(frame, PopupName(dialogs.showing), panel, false, 0.0f, {}, panel);
	}
	if (dialogs.showing != view) {
		dialogs.opened         = frame.Now();
		dialogs.asking_to_stop = false;
		dialogs.asked_to_stop  = false;
	}
	dialogs.showing     = view;
	const float opacity = state.stage.opacity;
	switch (view) {
		case VIEW_CONFIRM: ShowConfirm(frame, state, panel, opacity); break;
		case VIEW_PROGRESS:
			ShowWork(frame, state, panel, opacity, PopupName(view), "Installing update",
			         "Keep your console on.", INSTALL_SECONDS, true);
			break;
		case VIEW_SAVING:
			ShowWork(frame, state, panel, opacity, PopupName(view), "Saving",
			         "Don't turn off your console.", SAVE_SECONDS, false);
			break;
		case VIEW_ERROR: ShowError(frame, state, panel, opacity); break;
		case VIEW_LONG: ShowLongText(frame, state, panel, opacity); break;
		default: break;
	}
}

// The buttons that open each dialog.
Element BuildOpeners(UiFrame& frame, GalleryState& state) {
	std::vector<Element> buttons;
	for (const DialogOpener& opener: OPENERS) {
		const Element button = Button(frame, opener.label, {.label = opener.label});
		buttons.push_back(OnActivate(frame, button, OpenDialog {&state.dialogs_view, opener.view}));
	}
	return Keyed(frame, Box(frame, {.gap = Dp(frame.ActiveTheme().space.s3)}, {}, buttons), "live");
}

// What each dialog is for, and how the last one was answered.
Element BuildNotes(UiFrame& frame, GalleryState& state) {
	const Theme&         theme = frame.ActiveTheme();
	const Color          ink   = StageInk(theme, state.stage);
	std::vector<Element> rows;
	for (const DialogOpener& opener: OPENERS) {
		const std::string line = std::string(opener.label) + ": " + std::string(opener.note);
		rows.push_back(Text(frame, {line, theme.type.caption, ink}));
	}
	const std::string answer = "Last answer: " + std::string(state.dialogs.answer);
	rows.push_back(Text(frame, {answer, theme.type.caption, ink}));
	return Keyed(
	    frame,
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s2)}, {}, rows),
	    "specimens");
}

// ---- Notification -------------------------------------------------------------------------------

// A status toast and what it says.
struct StatusSample {
	std::string_view title;
	std::string_view subtitle;
	Color StatusColors::* bar;
};

constexpr std::array<StatusSample, 4> STATUS_SAMPLES {
    StatusSample {"Download complete", "Saved to your library.", &StatusColors::success},
    StatusSample {"Low storage", "Under 2 GB left.", &StatusColors::warning},
    StatusSample {"Couldn't connect", "Check your network.", &StatusColors::error},
    StatusSample {"Update available", "Version 1.04 is ready.", &StatusColors::info}};

// How long the frozen frames' toast is on screen, which puts its exit in the same place each time.
constexpr float FROZEN_SECONDS = 4.0f;
constexpr float MS_PER_SECOND  = 1000.0f;
// The frozen frames' width, narrower than a toast so a row of four fits. They aren't scaled, as a
// pose's scale doesn't reach text.
constexpr float FROZEN_WIDTH = 280.0f;
// Moments in the toast's entrance and exit, in milliseconds. The exit's stop where its fade ends,
// as there's nothing to see after that.
constexpr std::array<float, 4> ENTER_MOMENTS {0.0f, 100.0f, 180.0f, 280.0f};
constexpr std::array<float, 4> EXIT_MOMENTS {0.0f, 40.0f, 80.0f, 120.0f};
constexpr std::string_view     KIND_TITLE = "Master Alchemist";

// A plain status toast frozen `ms` into its life, which the clock has nothing to do with. Plain, as
// an icon wouldn't leave its words room at this width. It's in a cell with room above for the
// slide.
Element FrozenToast(UiFrame& frame, const ToastTokens& tokens, float ms) {
	using Milliseconds = std::chrono::duration<float, std::milli>;
	const UiClock::time_point start {};
	const auto moment = std::chrono::duration_cast<UiClock::duration>(Milliseconds(ms));
	const Pose pose =
	    Evaluate(ToastTimeline(tokens, FROZEN_SECONDS), start, std::nullopt, start + moment);
	const StatusSample& sample = STATUS_SAMPLES[1];
	ToastProps          props {sample.title, sample.subtitle};
	props.edge_bar = frame.ActiveTheme().colors.status.*sample.bar;
	props.tokens   = TokenPatch<ToastTokens> {}
	                   .With<&ToastTokens::width>(FROZEN_WIDTH)
	                   .With<&ToastTokens::min_width>(FROZEN_WIDTH);
	const Element toast = Toast(frame, props, frame.Scale().ContentFrame().width).root;
	(void)WithTimelinePose(frame, toast, pose);
	const std::array<Element, 1> inside {toast};
	return Box(frame, {.padding = {.top = Dp(tokens.slide)}, .item = {.width = Dp(FROZEN_WIDTH)}},
	           {}, inside);
}

// A row of frozen frames with a name, each frame under its time.
Element FrozenRow(UiFrame& frame, GalleryState& state, std::string_view name, float from,
                  std::span<const float> moments) {
	const Theme&         theme  = frame.ActiveTheme();
	const ToastTokens    tokens = ToastDefaults(theme);
	const Color          ink    = StageInk(theme, state.stage);
	std::vector<Element> cells {
	    Text(frame, {name, theme.type.caption, ink}, {.width = Dp(STAGE_ROW_LABEL_WIDTH)})};
	for (const float moment: moments) {
		const std::string            label = std::to_string(std::lround(moment)) + " ms";
		const std::array<Element, 2> parts {Text(frame, {label, theme.type.caption, ink}),
		                                    FrozenToast(frame, tokens, from + moment)};
		cells.push_back(Keyed(
		    frame,
		    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s1)}, {}, parts),
		    label));
	}
	return Keyed(frame, Box(frame, {.gap = Dp(theme.space.s3)}, {}, cells), name);
}

// The status toasts and a trophy's, two to a row.
Element BuildToastKinds(UiFrame& frame, GalleryState& state) {
	const Theme&         theme = frame.ActiveTheme();
	const float          width = frame.Scale().ContentFrame().width;
	std::vector<Element> toasts;
	for (const StatusSample& sample: STATUS_SAMPLES) {
		ToastProps props {sample.title, sample.subtitle};
		props.edge_bar = theme.colors.status.*sample.bar;
		toasts.push_back(Toast(frame, props, width).root);
	}
	toasts.push_back(Toast(frame, SampleToast(frame, state, std::string(KIND_TITLE)), width).root);
	std::vector<Element> rows;
	for (size_t index = 0; index < toasts.size(); index += 2) {
		const size_t count = std::min<size_t>(2, toasts.size() - index);
		rows.push_back(Box(frame, {.gap = Dp(theme.space.s4)}, {},
		                   std::span<const Element>(toasts).subspan(index, count)));
	}
	return Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s4)}, {}, rows);
}

// The notification page's specimens, the kinds of toast, then its entrance and exit, frozen.
Element BuildNotifications(UiFrame& frame, GalleryState& state) {
	LoadSampleIcon(state);
	const Theme&      theme  = frame.ActiveTheme();
	const ToastTokens tokens = ToastDefaults(theme);
	// The exit starts where the timeline's hold ends, so its frames are from there.
	const float exit_starts =
	    FROZEN_SECONDS * MS_PER_SECOND - std::max(tokens.exit_fade_ms, tokens.exit_move_ms);
	const std::array<Element, 3> rows {BuildToastKinds(frame, state),
	                                   FrozenRow(frame, state, "Enter", 0.0f, ENTER_MOMENTS),
	                                   FrozenRow(frame, state, "Exit", exit_starts, EXIT_MOMENTS)};
	return Keyed(
	    frame,
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s5)}, {}, rows),
	    "specimens");
}

// ---- Metric -------------------------------------------------------------------------------------

// The mock's sizes, in reference pixels.
constexpr float HUD_WIDTH    = 360.0f;
constexpr float GRAPH_HEIGHT = 48.0f;
constexpr float TILE_WIDTH   = 200.0f;

// A sample frame's share of the graph's height, and which of the chart's colours it's drawn in,
// written out so the picture never changes. Taller is slower.
constexpr std::array<float, 12> GRAPH_SAMPLES {0.25f, 0.30f, 0.25f, 0.35f, 0.30f, 0.55f,
                                               0.60f, 0.30f, 0.25f, 0.95f, 0.40f, 0.25f};
constexpr float                 GRAPH_MEDIUM_FROM = 0.5f;
constexpr float                 GRAPH_BAD_FROM    = 0.9f;
// The gap between the bars in pixels, so they read as separate frames.
constexpr float BAR_GAP = 1.0f;

// Draws the graph, a bar for each sample standing on the inset's floor. Trivially copyable so
// Custom can store it.
struct GraphPaint {
	void operator()(const CustomPaint& painting) const {
		const auto& colors = painting.frame.ActiveTheme().colors;
		const Rect& rect   = painting.painted;
		painting.list.AddRectFilled(TopLeft(rect), BottomRight(rect),
		                            ToImU32(colors.surface.inset.MulAlpha(painting.opacity)));
		const float pitch = rect.width / static_cast<float>(GRAPH_SAMPLES.size());
		for (size_t index = 0; index < GRAPH_SAMPLES.size(); index++) {
			const float sample = GRAPH_SAMPLES[index];
			const Color colour = sample >= GRAPH_BAD_FROM      ? colors.chart.bad
			                     : sample >= GRAPH_MEDIUM_FROM ? colors.chart.medium
			                                                   : colors.chart.good;
			const float left   = rect.x + pitch * static_cast<float>(index);
			painting.list.AddRectFilled({left, rect.Bottom() - rect.height * sample},
			                            {left + pitch - BAR_GAP, rect.Bottom()},
			                            ToImU32(colour.MulAlpha(painting.opacity)));
		}
	}
};

// The heads-up panel, a graph above lines that each say in words what their colour says. Bad's word
// is the status colour, as the chart's own red is for bars, and darker still over a bright game.
Element HudMock(UiFrame& frame) {
	const Theme&                 theme = frame.ActiveTheme();
	const auto&                  type  = theme.type;
	const auto&                  chart = theme.colors.chart;
	const HudTokens              hud   = HudDefaults(theme);
	const std::array<Element, 4> rows {
	    Custom(frame, {.height = Dp(GRAPH_HEIGHT)}, GraphPaint {}),
	    Text(frame, {"60.0 FPS  Good", type.dense, chart.good}),
	    Text(frame, {"42.0 FPS  Slow", type.dense, chart.medium}),
	    Text(frame, {"18.0 FPS  Bad", type.dense, theme.colors.status.error})};
	return Box(frame,
	           {.direction = FlexDirection::Column,
	            .align     = Align::Stretch,
	            .gap       = Dp(theme.space.s2),
	            .padding   = DpInsets::Symmetric(Dp(hud.padding_x), Dp(hud.padding_y)),
	            .item      = {.width = Dp(HUD_WIDTH)}},
	           {.fill = theme.colors.surface.hud, .radius = Dp(hud.radius)}, rows);
}

// A tile that shows one number and what it is.
Element StatTile(UiFrame& frame) {
	const Theme&                 theme = frame.ActiveTheme();
	const auto&                  text  = theme.colors.text;
	const std::array<Element, 2> rows {
	    Text(frame, {"Frame time", theme.type.caption, text.secondary}),
	    Text(frame, {"16.7 ms", theme.type.heading, text.primary})};
	return Box(frame,
	           {.direction = FlexDirection::Column,
	            .gap       = Dp(theme.space.s1),
	            .padding   = DpInsets::All(Dp(theme.space.s4)),
	            .item      = {.width = Dp(TILE_WIDTH)}},
	           {.fill = theme.colors.surface.raised, .radius = Dp(theme.radius.md)}, rows);
}

// The metric page's specimens, said to be a mock, as no component draws these yet.
Element BuildMetric(UiFrame& frame, GalleryState& state) {
	const Theme&                 theme = frame.ActiveTheme();
	const std::array<Element, 2> mocks {HudMock(frame), StatTile(frame)};
	const std::array<Element, 2> rows {
	    Text(frame,
	         {"Mock, as there's no HUD component yet. Colour never stands alone, as each line "
	          "carries a word.",
	          theme.type.caption, StageInk(theme, state.stage)}),
	    Box(frame, {.gap = Dp(theme.space.s4)}, {}, mocks)};
	return Keyed(
	    frame,
	    Box(frame, {.direction = FlexDirection::Column, .gap = Dp(theme.space.s4)}, {}, rows),
	    "specimens");
}

} // namespace

void DrawDialogsPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, DIALOGS_RULE, BuildNotes, pane, BuildOpeners, DrawDialog);
}

void DrawNotificationPage(UiFrame& frame, GalleryState& state, const PageEntry& page,
                          const Rect& pane) {
	ShowStagePage(frame, state, page, NOTIFICATION_RULE, BuildNotifications, pane);
}

void DrawMetricPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane) {
	ShowStagePage(frame, state, page, METRIC_RULE, BuildMetric, pane);
}

} // namespace Libs::Graphics::SystemUi::Gallery
