#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SAVEDIALOGVIEW_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SAVEDIALOGVIEW_H_

#include "common/common.h"
#include "imgui.h"
#include "libs/dialog.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Libs::Graphics {

// The save data dialog in the system overlay: the game's save or load list and its messages,
// operated with the controls as the emulator maps them for the game. The window fades in and
// out; when one dialog follows another (the list, then the overwrite question) the window stays
// while its content fades out and the next fades in.
class SaveDialogView final {
public:
	using Clock    = std::chrono::steady_clock;
	using Snapshot = Dialog::SaveDataDialog::HostSnapshot;
	// Takes a thumbnail texture the view no longer draws, to destroy once frames using it are
	// done.
	using RetireTexture = std::function<void(std::unique_ptr<ImTextureData>)>;

	explicit SaveDialogView(RetireTexture retire_texture);
	~SaveDialogView() = default;
	KYTY_CLASS_NO_COPY(SaveDialogView);

	// The fonts the dialog draws with; a null bold font falls back to the regular one, and a
	// null regular font to ImGui's default.
	void SetFonts(ImFont* regular, ImFont* bold);

	// Before ImGui::NewFrame while the game has the dialog open: the mapped controls as ImGui
	// gamepad keys. Cross confirms, Circle backs out, and the D-pad or left stick moves.
	static void FeedControls(ImGuiIO& io);

	// Advances the fades and draws the dialog. `open` is the dialog the game has open, or null.
	// `focus_pending` is set when a new overlay session starts and cleared once the view has
	// placed its initial focus. Returns whether the dialog is on screen: open, fading out, or
	// holding a finished save or load.
	bool Frame(const Snapshot* open, bool& focus_pending, ImVec2 display, float delta_time,
	           Clock::time_point now);

	// Unregisters the thumbnails before the ImGui context is destroyed.
	void UnregisterTextures();

private:
	using Item    = Dialog::SaveDataDialog::HostItem;
	using View    = Dialog::SaveDataDialog::HostView;
	using Buttons = Dialog::SaveDataDialog::HostButtons;

	enum class Action { None, Select, Ok, No, Cancel };

	struct Icon {
		std::shared_ptr<const std::vector<uint8_t>> png;
		std::unique_ptr<ImTextureData>              texture;
	};

	static constexpr float FADE_SECONDS       = 0.175f;
	static constexpr auto  PROGRESS_BAR_DELAY = std::chrono::milliseconds(500);
	static constexpr auto  FINISHED_HOLD      = std::chrono::milliseconds(1000);
	static constexpr auto  CLOSE_GRACE        = std::chrono::milliseconds(100);
	static constexpr ImU32 ACCENT             = IM_COL32(0x23, 0xA3, 0xFF, 0xFF); // #23a3ff
	static constexpr size_t MAX_ICONS         = 200; // within the overlay's descriptor pool

	void Update(const Snapshot* open, float delta_time, Clock::time_point now);
	void Draw(const Snapshot& snapshot, ImVec2 display, bool interactive, bool& focus_pending);
	void DrawItem(const Item& item, float scale, uint32_t style, bool selected,
	              float right_inset = 0.0f);
	static void    TextCenteredWrapped(const std::string& text);
	ImTextureData* IconTexture(const Item& item);
	void           RetireIcons();
	std::string    FullWidthSpaceFallback(const std::string& text);

	RetireTexture m_retire_texture;
	ImFont*       m_font       = nullptr;
	ImFont*       m_title_font = nullptr;
	bool          m_full_width_space_checked = false;
	bool          m_full_width_space_missing = false;

	// Thumbnails, decoded once per dialog.
	std::vector<Icon> m_icons;
	uint64_t          m_icons_generation = 0;

	// What is on screen, and the fades.
	Snapshot          m_shown; // the dialog drawn
	bool              m_shown_valid = false;
	Snapshot          m_next; // a dialog waiting for the content change
	bool              m_next_valid    = false;
	bool              m_was_open      = false;
	float             m_window_alpha  = 0.0f;
	float             m_content_alpha = 1.0f;
	Clock::time_point m_hold_until {};

	// The list's selection, and when a progress view has lasted long enough to show its bar.
	size_t            m_list_selected       = 0;
	uint64_t          m_progress_generation = 0;
	Clock::time_point m_progress_since {};
	bool              m_progress_bar = false;
};

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_PRESENTATION_SAVEDIALOGVIEW_H_ */
