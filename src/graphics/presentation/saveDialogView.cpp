#include "graphics/presentation/saveDialogView.h"

#include "common/logging/log.h"
#include "imgui_internal.h"
#include "libs/controller.h"
#include "stb_image.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace Libs::Graphics {

SaveDialogView::SaveDialogView(RetireTexture retire_texture)
    : m_retire_texture(std::move(retire_texture)) {}

void SaveDialogView::SetFonts(ImFont* regular, ImFont* bold) {
	m_font       = regular;
	m_title_font = bold != nullptr ? bold : regular;
}

void SaveDialogView::UnregisterTextures() {
	for (const auto& icon: m_icons) {
		if (icon.texture != nullptr) {
			ImGui::UnregisterUserTexture(icon.texture.get());
		}
	}
}

bool SaveDialogView::Frame(const Snapshot* open, bool& focus_pending, ImVec2 display,
                           float delta_time, Clock::time_point now) {
	Update(open, delta_time, now);
	const bool shown = m_shown_valid && m_window_alpha > 0.0f;
	if (shown) {
		// Only the open dialog's own content takes input.
		Draw(m_shown, display, open != nullptr && m_shown.generation == open->generation,
		     focus_pending);
	} else if (open == nullptr) {
		m_shown       = {};
		m_shown_valid = false;
		if (!m_icons.empty()) {
			RetireIcons();
		}
	}
	return shown;
}

void SaveDialogView::Update(const Snapshot* open, float delta_time, Clock::time_point now) {
	if (open != nullptr) {
		// When a progress view has lasted long enough to show its bar.
		if (open->view == View::Progress) {
			if (m_progress_generation != open->generation) {
				m_progress_generation = open->generation;
				m_progress_since      = now;
			}
			m_progress_bar = now - m_progress_since >= PROGRESS_BAR_DELAY;
		} else {
			m_progress_generation = 0;
			m_progress_bar        = false;
		}
	}
	const float fade_step = delta_time / FADE_SECONDS;
	// How long the shown content stays once its dialog has closed: a finished save or load a
	// moment, any other dialog briefly in case the game opens the next one straight away.
	const auto hold_for = [now](const Snapshot& shown) {
		return now + (shown.view == View::Progress ? FINISHED_HOLD : CLOSE_GRACE);
	};
	if (open != nullptr) {
		if (!m_shown_valid || m_window_alpha <= 0.0f) {
			// Opening from nothing: the window fades in with its content.
			m_shown         = *open;
			m_shown_valid   = true;
			m_next_valid    = false;
			m_content_alpha = 1.0f;
		}
		if (open->generation == m_shown.generation) {
			m_shown         = *open;
			m_next_valid    = false;
			m_content_alpha = std::min(m_content_alpha + fade_step, 1.0f);
		} else {
			// Another dialog followed: the shown content fades out, then this one in.
			m_next       = *open;
			m_next_valid = true;
		}
		m_window_alpha = std::min(m_window_alpha + fade_step, 1.0f);
		m_hold_until   = {};
	} else if (m_was_open && m_shown_valid && !m_next_valid) {
		m_hold_until = hold_for(m_shown);
	}
	m_was_open = open != nullptr;
	if (m_next_valid) {
		// Finish a content change even if the next dialog has closed already (a quick save's
		// progress view), so it is still seen, and held, before the window goes.
		m_content_alpha -= fade_step;
		if (m_content_alpha <= 0.0f) {
			m_content_alpha = 0.0f;
			m_shown         = m_next;
			m_next_valid    = false;
			if (open == nullptr) {
				m_hold_until = hold_for(m_shown);
			}
		}
	} else if (open == nullptr) {
		m_content_alpha = std::min(m_content_alpha + fade_step, 1.0f);
	}
	const bool holding = open == nullptr && (m_next_valid || now < m_hold_until);
	if (open == nullptr && !holding) {
		m_window_alpha = std::max(m_window_alpha - fade_step, 0.0f);
	}
}

void SaveDialogView::FeedControls(ImGuiIO& io) {
	const auto input  = Controller::GetSystemInput();
	const auto button = [&input](uint32_t mask) { return (input.buttons & mask) != 0; };
	io.AddKeyEvent(ImGuiKey_GamepadFaceDown, button(Controller::PAD_BUTTON_CROSS));
	io.AddKeyEvent(ImGuiKey_GamepadFaceRight, button(Controller::PAD_BUTTON_CIRCLE));
	io.AddKeyEvent(ImGuiKey_GamepadDpadUp,
	               button(Controller::PAD_BUTTON_UP) || input.left_y < 64);
	io.AddKeyEvent(ImGuiKey_GamepadDpadDown,
	               button(Controller::PAD_BUTTON_DOWN) || input.left_y > 192);
	io.AddKeyEvent(ImGuiKey_GamepadDpadLeft,
	               button(Controller::PAD_BUTTON_LEFT) || input.left_x < 64);
	io.AddKeyEvent(ImGuiKey_GamepadDpadRight,
	               button(Controller::PAD_BUTTON_RIGHT) || input.left_x > 192);
}

// Word-wrapped text with each line centered in the content region.
void SaveDialogView::TextCenteredWrapped(const std::string& text) {
	const float left  = ImGui::GetCursorPosX();
	const float avail = std::max(ImGui::GetContentRegionAvail().x, 1.0f);
	ImFont*     font  = ImGui::GetFont();
	const float size  = ImGui::GetFontSize();
	const char* begin = text.c_str();
	const char* end   = begin + text.size();
	std::vector<std::pair<const char*, const char*>> lines;
	while (begin < end) {
		const char* line_end = font->CalcWordWrapPosition(size, begin, end, avail);
		if (line_end == begin) {
			line_end = begin + 1; // a single glyph wider than the line
		}
		const char* newline = static_cast<const char*>(
		    std::memchr(begin, '\n', static_cast<size_t>(line_end - begin)));
		if (newline != nullptr) {
			line_end = newline;
		}
		lines.emplace_back(begin, line_end);
		begin = line_end;
		while (begin < end && (*begin == ' ' || *begin == '\n')) {
			begin++;
		}
	}
	// Lines sit as close as wrapped text; the usual spacing follows the last one.
	const ImVec2 spacing = ImGui::GetStyle().ItemSpacing;
	for (size_t i = 0; i < lines.size(); i++) {
		const bool last = i + 1 == lines.size();
		ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {spacing.x, last ? spacing.y : 0.0f});
		const float width = ImGui::CalcTextSize(lines[i].first, lines[i].second).x;
		ImGui::SetCursorPosX(left + std::max((avail - width) * 0.5f, 0.0f));
		ImGui::TextUnformatted(lines[i].first, lines[i].second);
		ImGui::PopStyleVar();
	}
}

void SaveDialogView::Draw(const Snapshot& snapshot, ImVec2 display, bool interactive,
                          bool& focus_pending) {
	const float  scale = std::max(std::min(display.x / 1280.0f, display.y / 720.0f), 0.5f);
	const bool   list  = snapshot.view == View::List;
	if (snapshot.generation != m_icons_generation) {
		RetireIcons();
		m_icons_generation = snapshot.generation;
	}
	// The window and the title follow the window's fade; the content (below) also its own.
	ImGui::PushStyleVar(ImGuiStyleVar_Alpha, m_window_alpha);
	ImGui::GetBackgroundDrawList()->AddRectFilled({0.0f, 0.0f}, display,
	                                              ImGui::GetColorU32(IM_COL32(0, 0, 0, 160)));
	ImGui::SetNextWindowPos({display.x * 0.5f, display.y * 0.5f}, ImGuiCond_Always,
	                        {0.5f, 0.5f});
	// One size for the list and the messages about a save (such as confirming an overwrite).
	const ImVec2 size {std::min(860.0f * scale, display.x - 32.0f),
	                   std::min(600.0f * scale, display.y - 32.0f)};
	ImGui::SetNextWindowSize({std::max(size.x, 1.0f), std::max(size.y, 1.0f)},
	                         ImGuiCond_Always);
	const bool first = interactive && focus_pending;
	if (first) {
		ImGui::SetNextWindowFocus();
	}
	ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, {24.0f * scale, 20.0f * scale});
	ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, {12.0f * scale, 12.0f * scale});
	ImGui::PushFont(m_font, 18.0f * scale);
	// The list takes no ImGui navigation (see below); messages use it for their buttons.
	const ImGuiWindowFlags flags =
	    ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
	    ImGuiWindowFlags_NoSavedSettings |
	    (list ? ImGuiWindowFlags_NoNavInputs : ImGuiWindowFlags_None);
	// ImGui's own focus frame never shows; rows and buttons draw their own.
	ImGui::PushStyleColor(ImGuiCol_NavCursor, IM_COL32(0, 0, 0, 0));
	ImGui::Begin("##SaveDataDialog", nullptr, flags);
	ImGui::PushFont(m_title_font, 26.0f * scale);
	ImGui::TextUnformatted(snapshot.disp_type == 2   ? "Load"
	                       : snapshot.disp_type == 3 ? "Delete"
	                                                 : "Save");
	ImGui::PopFont();
	ImGui::PushStyleVar(ImGuiStyleVar_Alpha, m_window_alpha * m_content_alpha);

	Action action = Action::None;
	size_t chosen = 0;
	if (list) {
		// The scrollbar sits on the dialog's own background, with a gap before the items.
		ImGui::PushStyleColor(ImGuiCol_ScrollbarBg, IM_COL32(0, 0, 0, 0));
		// The list moves its selection itself rather than through ImGui's navigation, which
		// stepped out of the list at its ends (flickering the highlight, and leaving a
		// frame around the whole list) and, with the list flattened into the dialog
		// window, did not move at all. The selection always stays on an item and stops at
		// the first and last ones.
		ImGui::BeginChild("##SaveList", {0.0f, 0.0f}, ImGuiChildFlags_None,
		                  ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_NoNavFocus);
		// The same gap as between two items.
		const float scrollbar_gap =
		    ImGui::GetScrollMaxY() > 0.0f ? ImGui::GetStyle().ItemSpacing.y : 0.0f;
		const size_t count = snapshot.items.size();
		bool         moved = false;
		if (first) {
			m_list_selected = snapshot.focus;
		} else if (interactive && count != 0) {
			if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadUp, true) && m_list_selected > 0) {
				m_list_selected--;
				moved = true;
			}
			if (ImGui::IsKeyPressed(ImGuiKey_GamepadDpadDown, true) &&
			    m_list_selected + 1 < count) {
				m_list_selected++;
				moved = true;
			}
			if (ImGui::IsKeyPressed(ImGuiKey_GamepadFaceDown, false)) {
				action = Action::Select;
				chosen = std::min(m_list_selected, count - 1);
			}
		}
		if (count != 0) {
			m_list_selected = std::min(m_list_selected, count - 1);
		}
		for (size_t i = 0; i < count; ++i) {
			DrawItem(snapshot.items[i], scale, snapshot.item_style, i == m_list_selected,
			             scrollbar_gap);
			if (i == m_list_selected) {
				if (first) {
					ImGui::SetScrollHereY(0.5f);
				} else if (moved) {
					ImGui::ScrollToItem(ImGuiScrollFlags_KeepVisibleEdgeY);
				}
			}
		}
		ImGui::EndChild();
		ImGui::PopStyleColor();
	} else {
		for (const auto& item: snapshot.items) {
			if (!item.is_new || snapshot.items.size() == 1) {
				DrawItem(item, scale, snapshot.item_style, false);
				break;
			}
		}
		// Three times the usual gap above the message (two item spacings, as with
		// ImGui::Spacing(), become six).
		ImGui::Dummy({0.0f, ImGui::GetStyle().ItemSpacing.y * 4.0f});
		if (!snapshot.message.empty()) {
			if (snapshot.error_message) {
				ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 120, 120, 255));
			}
			// The size of a save's title, in the regular weight, centered line by line.
			ImGui::PushFont(m_font, 22.0f * scale);
			TextCenteredWrapped(snapshot.message);
			ImGui::PopFont();
			if (snapshot.error_message) {
				ImGui::PopStyleColor();
			}
		}
		// Quick saves and loads show no bar; one appears only for a longer operation.
		if (snapshot.view == View::Progress && m_progress_bar) {
			// Twice the usual gap below the message: an empty item adds one more spacing.
			ImGui::Dummy({0.0f, 0.0f});
			ImGui::PushStyleColor(ImGuiCol_PlotHistogram, ACCENT);
			ImGui::ProgressBar(static_cast<float>(snapshot.progress) / 100.0f,
			                   {-1.0f, 12.0f * scale}, "");
			ImGui::PopStyleColor();
		}
		const auto buttons = snapshot.buttons;
		std::vector<std::pair<const char*, Action>> row;
		if (buttons == Buttons::YesNo) {
			row = {{"No", Action::No}, {"Yes", Action::Ok}};
		} else if (buttons == Buttons::OkCancel) {
			row = {{"OK", Action::Ok}, {"Cancel", Action::Cancel}};
		} else if (buttons == Buttons::Ok) {
			row = {{"OK", Action::Ok}};
		} else if (snapshot.can_cancel && snapshot.view == View::Message) {
			row = {{"Cancel", Action::Cancel}};
		}
		if (!row.empty()) {
			ImGui::Spacing();
			// The buttons sit at the bottom of the fixed-size window.
			const float bottom = ImGui::GetWindowHeight() - ImGui::GetStyle().WindowPadding.y -
			                     40.0f * scale;
			if (ImGui::GetCursorPosY() < bottom) {
				ImGui::SetCursorPosY(bottom);
			}
			const float width = std::min(140.0f * scale, ImGui::GetContentRegionAvail().x);
			const float total = width * static_cast<float>(row.size()) +
			                    ImGui::GetStyle().ItemSpacing.x *
			                        static_cast<float>(row.size() - 1);
			ImGui::SetCursorPosX(std::max((ImGui::GetWindowSize().x - total) * 0.5f, 0.0f));
			// Yes or OK has the initial focus, wherever it sits in the row; else the first
			// button. Focus is set explicitly: the dialog window can stay open from the
			// previous dialog (while it fades), so ImGui would not pick a default itself.
			const bool has_ok = std::any_of(row.begin(), row.end(), [](const auto& b) {
				return b.second == Action::Ok;
			});
			for (size_t i = 0; i < row.size(); ++i) {
				if (i != 0) {
					ImGui::SameLine();
				}
				if (first && (has_ok ? row[i].second == Action::Ok : i == 0)) {
					ImGui::SetKeyboardFocusHere();
				}
				// No and Cancel are grey; Yes and OK are the accent blue.
				const bool secondary =
				    row[i].second == Action::No || row[i].second == Action::Cancel;
				const ImU32 face = secondary ? IM_COL32(72, 72, 78, 255) : ACCENT;
				ImGui::PushStyleColor(ImGuiCol_Button, face);
				ImGui::PushStyleColor(ImGuiCol_ButtonHovered, secondary
				                                                  ? IM_COL32(88, 88, 94, 255)
				                                                  : IM_COL32(80, 185, 255, 255));
				ImGui::PushStyleColor(ImGuiCol_ButtonActive, secondary
				                                                 ? IM_COL32(60, 60, 66, 255)
				                                                 : IM_COL32(20, 135, 220, 255));
				if (ImGui::Button(row[i].first, {width, 40.0f * scale})) {
					action = row[i].second;
				}
				ImGui::PopStyleColor(3);
				// ImGui's own focus outline (hidden above) keeps the button's corner radius
				// while sitting outside it, leaving wider gaps at the corners; this one is
				// concentric, in the button's own color.
				if (ImGui::IsItemFocused()) {
					const ImU32     outline   = ImGui::GetColorU32(face);
					constexpr float thickness = 2.0f;
					constexpr float distance  = 4.0f; // as ImGui places it
					const ImVec2    min       = ImGui::GetItemRectMin();
					const ImVec2    max       = ImGui::GetItemRectMax();
					ImGui::GetWindowDrawList()->AddRect(
					    {min.x - distance, min.y - distance}, {max.x + distance, max.y + distance},
					    outline, ImGui::GetStyle().FrameRounding + distance, 0, thickness);
				}
			}
		}
	}
	// Circle backs out: No for a yes/no question, OK for a plain message.
	if (action == Action::None && ImGui::IsKeyPressed(ImGuiKey_GamepadFaceRight, false)) {
		if (list) {
			action = snapshot.can_cancel ? Action::Cancel : Action::None;
		} else if (snapshot.buttons == Buttons::YesNo) {
			action = Action::No;
		} else if (snapshot.buttons == Buttons::Ok) {
			action = Action::Ok;
		} else if (snapshot.can_cancel && snapshot.view == View::Message) {
			action = Action::Cancel;
		}
	}
	if (interactive) {
		focus_pending = false;
	} else {
		action = Action::None;
	}
	ImGui::PopStyleVar(); // content alpha
	ImGui::End();
	ImGui::PopStyleColor(); // NavCursor
	ImGui::PopFont();
	ImGui::PopStyleVar(3);
	switch (action) {
		case Action::Select: Dialog::SaveDataDialog::HostSelect(snapshot.generation, chosen); break;
		case Action::Ok: Dialog::SaveDataDialog::HostButton(snapshot.generation, 1); break;
		case Action::No: Dialog::SaveDataDialog::HostButton(snapshot.generation, 2); break;
		case Action::Cancel: Dialog::SaveDataDialog::HostCancel(snapshot.generation); break;
		case Action::None: break;
	}
}

// Decodes a save thumbnail once per dialog; null if the item has none or it is not a PNG.
ImTextureData* SaveDialogView::IconTexture(const Item& item) {
	if (!item.icon_png) {
		return nullptr;
	}
	for (const auto& icon: m_icons) {
		if (icon.png == item.icon_png) {
			return icon.texture.get();
		}
	}
	if (m_icons.size() >= MAX_ICONS) {
		return nullptr; // keeps within the descriptor pool; the row draws without a picture
	}
	Icon icon {item.icon_png, nullptr};
	const auto& encoded = *item.icon_png;
	int         width   = 0;
	int         height  = 0;
	if (!encoded.empty() &&
	    encoded.size() <= static_cast<size_t>(std::numeric_limits<int>::max()) &&
	    stbi_info_from_memory(encoded.data(), static_cast<int>(encoded.size()), &width, &height,
	                          nullptr) &&
	    width > 0 && height > 0 && width <= 2048 && height <= 2048) {
		auto* pixels = stbi_load_from_memory(encoded.data(), static_cast<int>(encoded.size()),
		                                     &width, &height, nullptr, 4);
		if (pixels != nullptr) {
			icon.texture = std::make_unique<ImTextureData>();
			icon.texture->Create(ImTextureFormat_RGBA32, width, height);
			std::memcpy(icon.texture->Pixels, pixels,
			            static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
			icon.texture->UseColors = true;
			stbi_image_free(pixels);
			ImGui::RegisterUserTexture(icon.texture.get());
		}
	}
	if (icon.texture == nullptr) {
		LOGF("SaveDataDialog: could not decode a save icon (%zu bytes)\n", encoded.size());
	}
	m_icons.push_back(std::move(icon));
	return m_icons.back().texture.get();
}

// Hands the thumbnails to the backend for destruction once in-flight frames are done.
void SaveDialogView::RetireIcons() {
	for (auto& icon: m_icons) {
		if (icon.texture != nullptr) {
			icon.texture->WantDestroyNextFrame = true;
			m_retire_texture(std::move(icon.texture));
		}
	}
	m_icons.clear();
}

// U+3000 (ideographic space), which games use to align columns, as U+2003 (em space, the same
// width) when the dialog font and its fallbacks lack it.
std::string SaveDialogView::FullWidthSpaceFallback(const std::string& text) {
	if (!m_full_width_space_checked) {
		m_full_width_space_checked = true;
		m_full_width_space_missing =
		    m_font != nullptr && !m_font->IsGlyphInFont(static_cast<ImWchar>(0x3000));
	}
	if (!m_full_width_space_missing || text.find("\xE3\x80\x80") == std::string::npos) {
		return text;
	}
	std::string result;
	result.reserve(text.size());
	for (size_t i = 0; i < text.size(); ++i) {
		if (text.compare(i, 3, "\xE3\x80\x80") == 0) {
			result += "\xE2\x80\x83";
			i += 2;
		} else {
			result += text[i];
		}
	}
	return result;
}

// One save: its thumbnail, then the title and the lines `style` (SceSaveDataDialogItemStyle)
// asks for. The default style puts the date beside the title and the size beside the
// subtitle, right-aligned. `selected` draws the row highlighted; the list handles its input.
void SaveDialogView::DrawItem(const Item& item, float scale, uint32_t style, bool selected,
                              float right_inset) {
	const float  width   = ImGui::GetContentRegionAvail().x - right_inset;
	const float  padding = 12.0f * scale;
	auto*        texture = IconTexture(item);
	// Without a thumbnail (none stored, or not a PNG) the row shows an empty 16:9 box.
	const float  icon_h  = 90.0f * scale;
	const float  icon_w  = texture != nullptr ? icon_h * static_cast<float>(texture->Width) /
	                                              static_cast<float>(texture->Height)
	                                          : icon_h * 16.0f / 9.0f;
	const float  text_x  = padding + icon_w + padding;
	const auto   title_size = 22.0f * scale;
	const auto   body_size  = 17.0f * scale;

	const std::string title    = FullWidthSpaceFallback(item.title);
	const std::string subtitle = FullWidthSpaceFallback(item.subtitle);
	std::string       date_size = item.date;
	if (!item.size.empty()) {
		date_size += (date_size.empty() ? "" : "    ") + item.size;
	}
	// The lines under the title, in the requested order.
	std::vector<std::pair<const std::string*, ImU32>> lines;
	const auto add = [&lines](const std::string& text, ImU32 color) {
		if (!text.empty()) {
			lines.emplace_back(&text, color);
		}
	};
	const ImU32 subtitle_color = IM_COL32(210, 210, 220, 255);
	const ImU32 info_color     = IM_COL32(170, 170, 185, 255);
	const bool  right_aligned  = style != 1 && style != 2;
	if (right_aligned) {
		if (!subtitle.empty() || !item.size.empty()) {
			lines.emplace_back(&subtitle, subtitle_color);
		}
	} else if (style == 1) {
		add(subtitle, subtitle_color);
		add(date_size, info_color);
	} else if (style == 2) {
		add(date_size, info_color);
	}

	ImGui::PushFont(m_title_font, title_size);
	ImFont*     title_font = ImGui::GetFont();
	const float title_h    = ImGui::GetTextLineHeight();
	ImGui::PopFont();
	ImGui::PushFont(nullptr, body_size);
	const float line_h = ImGui::GetTextLineHeight();
	ImGui::PopFont();
	const float text_h = title_h + line_h * static_cast<float>(lines.size());
	const float height = padding * 2.0f + std::max(text_h, icon_h);

	const ImVec2 origin = ImGui::GetCursorScreenPos();
	ImGui::Dummy({width, height});
	auto* draw = ImGui::GetWindowDrawList();
	if (selected) {
		// A light accent tint inside an accent border.
		draw->AddRectFilled(origin, {origin.x + width, origin.y + height},
		                    ImGui::GetColorU32(IM_COL32(35, 163, 255, 24)), 6.0f * scale);
		const float inset = 1.0f * scale;
		draw->AddRect({origin.x + inset, origin.y + inset},
		              {origin.x + width - inset, origin.y + height - inset},
		              ImGui::GetColorU32(ACCENT), 6.0f * scale, 0, 2.0f * scale);
	} else {
		draw->AddRect(origin, {origin.x + width, origin.y + height},
		              ImGui::GetColorU32(IM_COL32(255, 255, 255, 40)), 6.0f * scale);
	}
	const ImVec2 icon_min {origin.x + padding, origin.y + (height - icon_h) * 0.5f};
	const ImVec2 icon_max {icon_min.x + icon_w, icon_min.y + icon_h};
	if (texture != nullptr) {
		draw->AddImage(texture->GetTexRef(), icon_min, icon_max, {0.0f, 0.0f}, {1.0f, 1.0f},
		               ImGui::GetColorU32(IM_COL32(255, 255, 255, 255)));
	} else {
		draw->AddRectFilled(icon_min, icon_max, ImGui::GetColorU32(IM_COL32(64, 64, 70, 255)));
	}
	ImVec2 pos {origin.x + text_x, origin.y + padding};
	// Right-aligned date and size; text on their line is clipped short of them.
	const float right = origin.x + width - padding;
	const auto  right_text = [&](const std::string& text, float y, float row_h) {
		if (text.empty()) {
			return right;
		}
		ImGui::PushFont(nullptr, body_size);
		const float text_w = ImGui::CalcTextSize(text.c_str()).x;
		ImGui::PopFont();
		draw->AddText(nullptr, body_size, {right - text_w, y + (row_h - line_h) * 0.5f},
		              ImGui::GetColorU32(info_color), text.c_str());
		return right - text_w - padding * 2.0f;
	};
	const float title_right = right_aligned ? right_text(item.date, pos.y, title_h) : right;
	const ImVec4 title_clip {pos.x, pos.y, title_right, pos.y + title_h};
	draw->AddText(title_font, title_size, pos,
	              ImGui::GetColorU32(IM_COL32(255, 255, 255, 255)),
	              title.c_str(), nullptr, 0.0f, &title_clip);
	pos.y += title_h;
	for (const auto& [text, color]: lines) {
		const float line_right =
		    right_aligned ? right_text(item.size, pos.y, line_h) : right;
		const ImVec4 clip {pos.x, pos.y, line_right, pos.y + line_h};
		draw->AddText(nullptr, body_size, pos, ImGui::GetColorU32(color), text->c_str(), nullptr,
		              0.0f, &clip);
		pos.y += line_h;
	}
}

} // namespace Libs::Graphics
