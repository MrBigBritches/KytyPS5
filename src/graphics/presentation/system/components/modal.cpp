#include "graphics/presentation/system/components/modal.h"

#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"
#include "imgui.h"
#include "imgui_internal.h"

#include <algorithm>
#include <array>
#include <limits>

namespace Libs::Graphics::SystemUi {

namespace {

// Opens or closes the popup `name` as `open` asks, and begins it with its window filling `card`.
// Returns whether it is showing. The window has no background or padding of its own, since the
// modal draws its scrim and card itself.
bool BeginModal(const char* name, const Rect& card, bool open) {
	if (open && !ImGui::IsPopupOpen(name)) {
		ImGui::OpenPopup(name);
	}
	ImGui::SetNextWindowPos(TopLeft(card));
	ImGui::SetNextWindowSize({card.width, card.height});
	ImGui::PushStyleColor(ImGuiCol_ModalWindowDimBg, 0u);
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {0.0f, 0.0f});
	ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
	const bool visible = ImGui::BeginPopupModal(
	    name, nullptr,
	    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings |
	        ImGuiWindowFlags_NoBackground);
	ImGui::PopStyleVar(2);
	ImGui::PopStyleColor();
	if (visible && !open) {
		// It still ends this frame, so its contents may be drawn once more as it closes.
		ImGui::CloseCurrentPopup();
	}
	return visible;
}

// Makes the dialog in front the owner of the shoulders and back, whatever any control behind it
// claimed, so no tab bar or page switch fires under it, and sets `back` from its own read of back.
// Back is only owned where ImGui's own cancel would leave focus alone. Otherwise cancel is what
// drops focus from the dialog, and taking it away would change that, so back is read whoever owns
// it and the screen's shortcut is kept out by its focus check instead.
void ClaimForDialog(bool& back) {
	const ItemId owner = ImGui::GetCurrentWindow()->ID;
	(void)ClaimKeys(KeyGroup::Shoulders, owner, ClaimPriority::Modal);
	const bool keeps_focus = CancelKeepsFocus();
	if (keeps_focus) {
		(void)ClaimKeys(KeyGroup::Back, owner, ClaimPriority::Modal);
	}
	back = BackKeyPressed(keeps_focus ? owner : ANY_OWNER);
	if (back && keeps_focus) {
		// Closing on back takes the press from everything else this frame. Where ImGui's cancel
		// drops focus it is left alone, since consuming would stop that too.
		ConsumeKey(KeyGroup::Back, owner);
	}
}

// Builds the scrim over `area` and the dialog card as elements and paints them into the modal's
// window, the scrim first.
void DrawScrimAndCard(UiFrame& frame, ImDrawList& list, const Rect& card, const Rect& area,
                      float opacity, const DialogTokens& dialog) {
	const auto&   theme = frame.ActiveTheme();
	const auto&   scale = frame.Scale();
	Style         scrim_style {.fill = theme.colors.surface.scrim};
	const Element scrim = Box(frame,
	                          {.item = {.width  = FromPixels(scale, area.width),
	                                    .height = FromPixels(scale, area.height)}},
	                          scrim_style, {});
	const Style   card_style {theme.colors.surface.raised, theme.colors.border.subtle,
                            Dp(theme.border.hair), Dp(dialog.radius), dialog.elevation};
	const Element face = Box(frame,
	                         {.item = {.width  = FromPixels(scale, card.width),
	                                   .height = FromPixels(scale, card.height)}},
	                         card_style, {});
	// Both have explicit sizes, so measuring them in unlimited room only records those sizes.
	const float unlimited = std::numeric_limits<float>::infinity();
	(void)Measure(frame, scrim, {unlimited, unlimited});
	(void)Measure(frame, face, {unlimited, unlimited});
	(void)Arrange(frame, scrim, area);
	(void)Arrange(frame, face, card);
	// The modal's window clips to the card, so the clip widens to the area for the scrim and for
	// the card's shadow, which spreads past the card.
	list.PushClipRect({area.x, area.y}, {area.Right(), area.Bottom()}, false);
	Paint(frame, scrim, list, {.opacity = opacity});
	Paint(frame, face, list, {.opacity = opacity});
	list.PopClipRect();
}

} // namespace

Rect DialogCard(const UiFrame& owner, Dp height, const TokenPatch<DialogTokens>& patch) {
	const UiContext& context = owner.Ui();
	const auto&      theme   = context.theme;
	const auto       dialog  = patch.Apply(DialogDefaults(theme));
	const Rect       frame   = context.scale.ContentFrame();
	const Rect       output  = context.scale.Screen();
	const float      margin  = context.scale.Px(theme.space.s6);
	const float      width   = std::min(context.scale.Px(dialog.width_medium),
	                                    std::max(0.0f, frame.width - 2.0f * margin));
	const float      fitted =
	    std::min(ToPixels(context.scale, height), std::max(0.0f, frame.height - 2.0f * margin));
	return {SnapPixel(output.x + (output.width - width) / 2.0f),
	        SnapPixel(output.y + (output.height - fitted) / 2.0f), width, fitted};
}

ModalScope::ModalScope(UiFrame& frame, std::string_view id, const Rect& card, bool open,
                       float opacity, const TokenPatch<DialogTokens>& tokens,
                       const std::optional<Rect>& area)
    : m_card(card), m_visible(BeginModal(frame.InternCString(id), card, open)),
      m_list(&CurrentDrawList()), m_padding(frame.Scale().Px(frame.ActiveTheme().space.s6)) {
	if (!m_visible) {
		return;
	}
	const ImVec2 display = ImGui::GetIO().DisplaySize;
	DrawScrimAndCard(frame, *m_list, card, area.value_or(Rect {0.0f, 0.0f, display.x, display.y}),
	                 opacity, tokens.Apply(DialogDefaults(frame.ActiveTheme())));
	if (open) {
		ClaimForDialog(m_back);
	}
}

ModalScope::~ModalScope() {
	if (m_visible) {
		ImGui::EndPopup();
	}
}

DialogParts DialogContent(UiFrame& frame, std::string_view key, const DialogProps& props,
                          std::span<const Element> buttons, const ItemLayout& item,
                          ElementError* reason) {
	const Theme&    theme = frame.ActiveTheme();
	const Dp        gap {theme.space.s4};
	const TextProps title {props.title, theme.type.heading, theme.colors.text.primary};
	TextProps       body {props.body, theme.type.body, theme.colors.text.secondary};
	body.overflow = TextOverflow::Wrap;
	DialogParts parts;
	parts.title = Text(frame, title, {}, reason);
	parts.body  = Text(frame, body, {}, reason);
	// The body sits in a clipping box that takes the room the title and the buttons leave and may
	// shrink to nothing, so a dialog with more to say than fits scrolls its body and the buttons
	// stay on the card. It keeps a gutter for its scroll bar whether it needs one or not, so its
	// lines don't wrap differently as it grows.
	Style clip;
	clip.overflow = Overflow::Clip;
	const std::array<Element, 1> inside {parts.body};
	const Element                bodyBox = Box(frame,
	                                           {.direction = FlexDirection::Column,
	                                            .align     = Align::Stretch,
	                                            .padding = {.right = Dp(LIST_THUMB_WIDTH + LIST_THUMB_GAP)},
	                                            .item = {.grow = 1.0f, .min_main = Dp(0.0f)}},
	                                           clip, inside, reason);
	(void)AsScrollingText(frame, bodyBox, reason);
	// The buttons keep their natural length, so it is the body that gives way.
	const Element row =
	    Box(frame, {.direction = FlexDirection::Row, .justify = Justify::End, .gap = gap}, {},
	        buttons, reason);
	const std::array<Element, 3> column {parts.title, bodyBox, row};
	const Element                content =
	    Box(frame,
	        {.direction = FlexDirection::Column, .align = Align::Stretch, .gap = gap, .item = item},
	        {}, column, reason);
	parts.content = Keyed(frame, content, key, reason);
	return parts;
}

Vec2 MeasureDialog(UiFrame& frame, const DialogParts& parts, float width) {
	const Vec2 room {width, std::numeric_limits<float>::infinity()};
	// A dialog already measured in this room, such as by FitDialogCard for the card Place is now
	// laying it out in, keeps its answer.
	if (const auto measured = MeasuredAt(frame, parts.content, room)) {
		return *measured;
	}
	return Measure(frame, parts.content, room);
}

Rect FitDialogCard(UiFrame& frame, const DialogParts& parts,
                   const TokenPatch<DialogTokens>& tokens) {
	const UiContext& ui      = frame.Ui();
	const float      padding = ui.scale.Px(ui.theme.space.s6);
	// The width DialogCard gives does not depend on the height asked for.
	const float width  = DialogCard(frame, Dp(0.0f), tokens).width - 2.0f * padding;
	const Vec2  wanted = MeasureDialog(frame, parts, width);
	return DialogCard(frame, FromPixels(ui.scale, wanted.y + 2.0f * padding), tokens);
}

bool ModalScope::Place(UiFrame& frame, const DialogParts& parts) const {
	const Rect content = Content();
	(void)MeasureDialog(frame, parts, content.width);
	return Arrange(frame, parts.content, content);
}

Rect ModalScope::Content() const {
	return m_card.Inset(EdgeInsets::All(m_padding));
}

} // namespace Libs::Graphics::SystemUi
