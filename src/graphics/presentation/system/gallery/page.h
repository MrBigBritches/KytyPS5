#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PAGE_H_

#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/gallery/imagesPage.h"
#include "graphics/presentation/system/gallery/stage.h"
#include "graphics/presentation/system/gallery/toastPage.h"
#include "graphics/presentation/system/imgui/imageStore.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace Libs::Graphics::SystemUi::Gallery {

// The sidebar's sections, in the order they are listed.
enum class PageGroup : uint8_t { Styles, Components, Patterns };

// The values of the interactive pages' specimens, which take no input, and their live copies.
struct InteractivePagesState {
	bool                 off = false;
	bool                 on  = true;
	std::array<float, 3> slider_values {0.0f, 0.5f, 1.0f};
	bool                 live_toggle = false;
	float                live_slider = 0.5f;
	int                  live_row    = 0;
	int                  live_tab    = 0;
};

// What the dialogs page keeps from frame to frame, as dialogs_view chooses which is open.
struct DialogsPageState {
	// The view whose dialog was open last frame, to close it and to know when another opens.
	int showing = 0;
	// When the dialog showing opened, which its sample work is timed from.
	UiClock::time_point opened {};
	// Whether the work's dialog is asking before it stops, this frame and the last, as a dialog
	// needs one more frame to close in.
	bool asking_to_stop = false;
	bool asked_to_stop  = false;
	// How the last dialog was answered.
	std::string_view answer = "None yet";
};

// What the gallery shows, kept across frames.
struct GalleryState {
	// The page shown, as an index into the page list.
	int page = 0;
	// The control focus enters the page on from the sidebar, which the page sets as it draws. Zero
	// while the page has no controls.
	ItemId entry = 0;
	// A page to show at the request of something outside the gallery, such as a screenshot run
	// stepping through every page. Setting `page` alone isn't enough, since the sidebar shows the
	// page of whichever row has focus. The gallery shows it on the next frame, with focus on its
	// row, and clears the request. Negative while nothing is asked.
	int show = -1;
	// What the last draw saw, to catch a frame that was not begun again for a new ImGui frame: the
	// ImGui frame it drew in and the frame's BeginCount then. The counter is kept here and not in
	// UiFrame so that the frame needs no ImGui, which its tests run without. The count starts at
	// what a frame never begun reports, so the first draw from such a frame is caught too.
	int      drawn_in_imgui_frame = -1;
	uint32_t drawn_begin_count    = UiFrame {}.BeginCount();
	// Whether the last draw found the frame not begun again for a new ImGui frame, the fault that
	// debug builds assert on. It is set in every build, so a test can check it.
	bool frame_was_stale = false;
	// Where pages load images, which outlives this state, and a sample trophy icon as an encoded
	// image, which the host supplies. A page does without images while these are empty, as it does
	// in a test that isn't about them.
	ImageStore*                images = nullptr;
	std::span<const std::byte> sample_icon;
	// What the toast page keeps from frame to frame.
	ToastPageState toast;
	// What the images page keeps from frame to frame.
	ImagesPageState images_page;
	// What the component pages' control strip chose, and the frame their specimens are built in.
	StageState stage;
	// Which view each page with views shows, as an index into its view names.
	int colour_view     = 0;
	int typography_view = 0;
	int spacing_view    = 0;
	int images_view     = 0;
	int effects_view    = 0;
	int dialogs_view    = 0;
	// Whether the typography page shows each style's line box.
	bool typography_line_boxes = false;
	// Whether the effects page's Shine and Motion views apply the reduced-motion rule.
	bool effects_reduced_motion = false;
	// What the dialogs page keeps from frame to frame.
	DialogsPageState dialogs;
	// What the interactive pages keep from frame to frame.
	InteractivePagesState interactive;
	// The sample screen on the spacing page's Frame view.
	float spacing_safe_percent = ScaleRules {}.default_safe_area_percent;
	bool  spacing_ultrawide    = false;

	/// Puts the gallery back as it starts, on the first page with nothing asked for. What the host
	/// supplied stays.
	void Reset() {
		page  = 0;
		entry = 0;
		show  = -1;
		toast.Reset();
		stage.Reset();
		colour_view            = 0;
		typography_view        = 0;
		typography_line_boxes  = false;
		spacing_view           = 0;
		images_view            = 0;
		effects_view           = 0;
		effects_reduced_motion = false;
		interactive            = {};
		dialogs                = {};
		dialogs_view           = 0;
		spacing_safe_percent   = ScaleRules {}.default_safe_area_percent;
		spacing_ultrawide      = false;
	}
};

struct PageEntry;

// Draws `page`'s content into `pane`, the space beside the sidebar.
// `frame` is the frame the gallery is drawn in, begun by the caller, which the page builds its
// elements in and registers its controls through.
using PageDraw = void (*)(UiFrame& frame, GalleryState& state, const PageEntry& page,
                          const Rect& pane);

// One page of the gallery.
struct PageEntry {
	PageGroup group = PageGroup::Styles;
	// Names the page uniquely. Its controls are named within it, so two pages can use the same
	// control names.
	std::string_view id;
	std::string_view title;
	PageDraw         draw = nullptr;
	// For a page that shows one of several views, where the state keeps the choice and what the
	// views are called. A screenshot run takes each of them.
	int GalleryState::*               view = nullptr;
	std::span<const std::string_view> view_names;
};

/// The gallery's pages in sidebar order, each group's pages together.
[[nodiscard]] std::span<const PageEntry> Pages();

/// The heading shown over `group`'s pages in the sidebar.
[[nodiscard]] std::string_view GroupTitle(PageGroup group);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_PAGE_H_
