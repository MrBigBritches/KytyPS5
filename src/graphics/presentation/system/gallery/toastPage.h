#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_TOASTPAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_TOASTPAGE_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/imgui/imageStore.h"
#include "graphics/presentation/system/views/trophyToast.h"

#include <array>
#include <cstddef>
#include <optional>
#include <string>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The held toast is shown once on each backdrop.
inline constexpr size_t TOAST_BACKDROPS = 2;

// Where the toast page last put its parts, in output pixels, so a test can find them.
struct ToastPageLayout {
	// The dark stage the live toast plays on.
	Rect stage;
	Rect replay;
	// The held toast's card on each backdrop.
	std::array<Rect, TOAST_BACKDROPS> cards {};
};

// What the toast page keeps from frame to frame.
struct ToastPageState {
	// The chosen grade, as an index into the grade tabs.
	int grade = 0;
	// The icon the held toasts share, loaded when first needed. It lasts as long as the store,
	// so it survives Reset.
	std::optional<ImageId> icon;
	// Whether the icon has been asked for. It survives Reset too, as the same bytes would fail again.
	bool icon_tried = false;
	// The toast that plays on the stage, made when the page is first drawn with an image store.
	std::optional<TrophyToastView> live;
	// When the live toast plays next, once the last one has left.
	UiClock::time_point next_show {};
	// The ImGui frame the page last drew in, to notice the page being shown again.
	int drawn_in_imgui_frame = -1;
	// What the page's controls found this frame, which they write during Render.
	bool replay_pressed = false;
	bool grade_changed  = false;
	ToastPageLayout layout;

	/// Puts the page back as it starts, keeping the icon, which only the store can free.
	void Reset();
};

/// What a sample trophy toast shows for the toast page's chosen grade, with its icon once loaded.
[[nodiscard]] ToastProps SampleToast(const UiFrame& frame, const GalleryState& state,
                                     const std::string& title);

/// Loads the sample icon if it hasn't been tried yet, for any page that shows a sample toast, so
/// what that page shows doesn't depend on the toast page having been shown first.
void LoadSampleIcon(GalleryState& state);

/// Draws the toast page into `pane`, with grade tabs and a Replay button, the real trophy toast
/// playing on a dark stage, and the toast held still on a dark and a bright backdrop. The
/// live toast is skipped when `state.images` is null, and every toast goes without an icon when
/// that or `state.sample_icon` gives none.
void DrawToastPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_TOASTPAGE_H_
