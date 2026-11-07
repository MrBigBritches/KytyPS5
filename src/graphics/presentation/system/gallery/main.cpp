// The System UI gallery as a program. It shows the design system's pages in a window of its own,
// so the owner can review them without running a game.
//
// SDL3/SDL_main.h is deliberately not included. The emulator doesn't use it either, so main is a
// plain main on every platform and SDL doesn't rename it.

#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/gallery/capture.h"
#include "graphics/presentation/system/gallery/options.h"
#include "graphics/presentation/system/gallery/page.h"
#include "graphics/presentation/system/gallery/shell.h"
#include "graphics/presentation/system/gallery/vulkanHost.h"
#include "graphics/presentation/system/imgui/fonts.h"
#include "graphics/presentation/system/imgui/imageStore.h"
#include "graphics/presentation/system/imgui/interaction.h"
#include "imgui.h"
#include "imgui_impl_sdl3.h"

#include <SDL3/SDL.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

// The image store decodes PNG and JPEG with stb_image, which this program gets nowhere else.
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#include "stb_image.h"
#pragma clang diagnostic pop

namespace {

using namespace Libs::Graphics::SystemUi;
using namespace Libs::Graphics::SystemUi::Gallery;

constexpr const char* WINDOW_TITLE = "System UI gallery";
// Next to the executable, where the build copies the Noto Sans files for the emulator too.
constexpr const char* FONT_FOLDER  = "assets/fonts/";
constexpr const char* FONT_REGULAR = "NotoSans-Regular.ttf";
constexpr const char* FONT_BOLD    = "NotoSans-Bold.ttf";
// The icon the emulator's test toasts use, which the build copies next to the executable too.
constexpr const char* SAMPLE_ICON = "assets/images/hidden-trophy.png";
// How long to sleep while the window is minimised, so it doesn't spin a core.
constexpr uint32_t IDLE_DELAY_MS = 10;
// Screenshots advance time by this much per frame, as a 60 Hz display would, so every run
// animates identically.
constexpr double SCREENSHOT_FRAME_SECONDS = 1.0 / 60.0;
constexpr double MS_PER_SECOND            = 1000.0;
// Extra frames past the slowest motion, so rounding and a late start can't leave a highlight
// mid-way.
constexpr int SETTLE_MARGIN_FRAMES = 6;
// The slowest motion that stops by itself in the interaction tokens, in milliseconds. These are
// the focus, hover, press, knob and tab indicator tweens, and the indicator is the longest.
constexpr float SLOWEST_SETTLING_MS =
    std::max({DEFAULT_THEME.interaction.focus_in_ms, DEFAULT_THEME.interaction.focus_out_ms,
              DEFAULT_THEME.interaction.hover_ms, DEFAULT_THEME.interaction.press_ms,
              DEFAULT_THEME.interaction.knob_ms, DEFAULT_THEME.interaction.indicator_ms});
// Frames to run on each page before capturing it. Looping effects such as the halo pulse never
// settle, and are captured at whatever phase the fixed clock gives them.
constexpr int SETTLE_FRAMES =
    static_cast<int>(SLOWEST_SETTLING_MS / (SCREENSHOT_FRAME_SECONDS * MS_PER_SECOND)) + 1 +
    SETTLE_MARGIN_FRAMES;
// The side of the hidden window a screenshot run opens, in pixels.
constexpr int HIDDEN_WINDOW_SIZE = 256;
// The usage exit code, apart from a failure while running.
constexpr int EXIT_USAGE = 2;

// The background behind the pages. GRAY_50 is the darkest neutral that isn't near black, so the
// base surface (GRAY_100) and the raised surface (GRAY_150) read as lifted off it, as they do
// over a dark game. GRAY_0 would make them look flat against the background.
constexpr Color BACKGROUND = Palette::GRAY_50;

// Loads the system UI's fonts from beside the executable. Says so on stderr when any is missing,
// and then adds ImGui's default font so text still shows.
UiFonts LoadFonts(ImGuiIO& io) {
	const char*       base_path = SDL_GetBasePath();
	const std::string folder    = std::string(base_path != nullptr ? base_path : "") + FONT_FOLDER;
	const UiFonts     fonts     = LoadUiFonts(*io.Fonts, folder + FONT_REGULAR, folder + FONT_BOLD);
	if (fonts.regular == nullptr || fonts.bold == nullptr) {
		std::fprintf(stderr,
		             "system_ui_gallery: a font is missing from %s\n"
		             "Build the system_ui_gallery target to copy them there.\n",
		             folder.c_str());
	}
	// One font is enough, as UiFonts::For uses it for the other weight. ImGui's default is only
	// needed when both are missing.
	if (fonts.regular == nullptr && fonts.bold == nullptr) {
		io.Fonts->AddFontDefault();
	}
	return fonts;
}

// Reads the sample trophy icon from beside the executable. Says so on stderr when it can't, and the
// gallery then shows its toasts without an icon. SDL gives the folder as UTF-8, which the path must
// be told on Windows.
std::vector<std::byte> LoadSampleIcon() {
	const char* base_path = SDL_GetBasePath();
	const auto  file = std::filesystem::path(reinterpret_cast<const char8_t*>(
	                      base_path != nullptr ? base_path : "")) /
	                  SAMPLE_ICON;
	std::ifstream stream(file, std::ios::binary);
	if (!stream) {
		std::fprintf(stderr, "system_ui_gallery: could not read %s\n", file.string().c_str());
		return {};
	}
	const std::vector<char> bytes {std::istreambuf_iterator<char>(stream),
	                               std::istreambuf_iterator<char>()};
	const auto*             data = reinterpret_cast<const std::byte*>(bytes.data());
	return {data, data + bytes.size()};
}

// Draws one frame of the gallery filling the display, in a window that has no chrome, as the
// system UI's views do.
void DrawFrame(UiFrame& frame, InteractionStore& store, GalleryState& state, const UiFonts& fonts,
               UiClock::time_point now) {
	const ImVec2 size = ImGui::GetIO().DisplaySize;
	ImGui::SetNextWindowPos({0, 0});
	ImGui::SetNextWindowSize(size);
	ImGui::Begin("view", nullptr,
	             ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
	                 ImGuiWindowFlags_NoSavedSettings);
	frame.Begin(DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, size.x, size.y), fonts, store,
	            now);
	DrawGallery(frame, state, Pages(), {0, 0, size.x, size.y});
	ImGui::End();
}

// Handles pending window events. Returns false once the window was asked to close.
bool PollEvents(SDL_Window* window) {
	bool      running = true;
	SDL_Event event;
	while (SDL_PollEvent(&event)) {
		ImGui_ImplSDL3_ProcessEvent(&event);
		if (event.type == SDL_EVENT_QUIT || (event.type == SDL_EVENT_WINDOW_CLOSE_REQUESTED &&
		                                     event.window.windowID == SDL_GetWindowID(window))) {
			running = false;
		}
	}
	return running;
}

// Runs frames until the window closes. Returns a failure code when Vulkan can not continue.
int RunLoop(SDL_Window* window, VulkanHost& host, const UiFonts& fonts, ImageStore& images,
            std::span<const std::byte> sample_icon) {
	InteractionStore store;
	GalleryState     state;
	UiFrame          frame;
	state.images      = &images;
	state.sample_icon = sample_icon;
	while (PollEvents(window)) {
		const FrameResult began = host.BeginFrame();
		if (began == FrameResult::Fatal) {
			return EXIT_FAILURE;
		}
		if (began == FrameResult::Skip) {
			SDL_Delay(IDLE_DELAY_MS);
			continue;
		}
		ImGui_ImplSDL3_NewFrame();
		ImGui::NewFrame();
		DrawFrame(frame, store, state, fonts, UiClock::now());
		ImGui::Render();
		const FrameResult rendered = host.Render(ImGui::GetDrawData(), BACKGROUND);
		// Controls not drawn this frame are forgotten, so they start at rest if they return.
		store.Collect();
		if (rendered == FrameResult::Fatal) {
			return EXIT_FAILURE;
		}
	}
	return EXIT_SUCCESS;
}

// What a screenshot run needs from frame to frame.
struct ScreenshotRun {
	VulkanHost&         host;
	const UiFonts&      fonts;
	InteractionStore    store;
	GalleryState        state;
	UiFrame             frame;
	ImVec2              size;
	UiClock::time_point now {};
};

// Runs one frame of the gallery at the fixed size and time step, and draws it into ImGui's draw
// data without sending it to the GPU. Mouse and gamepad input are never fed in, so the input mode
// stays Pad.
void StepScreenshotFrame(ScreenshotRun& run) {
	ImGuiIO& io                = ImGui::GetIO();
	io.DisplaySize             = run.size;
	io.DisplayFramebufferScale = {1.0f, 1.0f};
	io.DeltaTime               = static_cast<float>(SCREENSHOT_FRAME_SECONDS);
	run.host.BeginCaptureFrame();
	ImGui::NewFrame();
	DrawFrame(run.frame, run.store, run.state, run.fonts, run.now);
	ImGui::Render();
	run.store.Collect();
	run.now += std::chrono::duration_cast<UiClock::duration>(
	    std::chrono::duration<double>(SCREENSHOT_FRAME_SECONDS));
}

// Shows `job`'s page, lets its motion settle, and writes its PNG into `folder`.
bool ScreenshotPage(ScreenshotRun& run, const ScreenshotJob& job,
                    const std::filesystem::path& folder) {
	// Each page starts from the same clock, store and state, so its picture depends only on that
	// page. Otherwise adding or reordering pages would change every later PNG.
	run.now   = {};
	run.store = {};
	run.state.Reset();
	run.state.show = job.page;
	if (const auto view = Pages()[static_cast<size_t>(job.page)].view; view != nullptr) {
		run.state.*view = job.view;
	}
	for (int frame = 0; frame < SETTLE_FRAMES; frame++) {
		StepScreenshotFrame(run);
	}
	CapturedImage image;
	if (!run.host.Capture(ImGui::GetDrawData(), BACKGROUND, image)) {
		std::fprintf(stderr, "system_ui_gallery: could not render %s\n", job.file_name.c_str());
		return false;
	}
	const auto rgba = ToRgba(image.pixels, image.width, image.height, image.row_pitch, image.bgra);
	if (rgba.empty() || !WritePng(folder / job.file_name, rgba, image.width, image.height)) {
		std::fprintf(stderr, "system_ui_gallery: could not write %s\n", job.file_name.c_str());
		return false;
	}
	std::printf("%s\n", (folder / job.file_name).string().c_str());
	return true;
}

// Writes a PNG of every page into the folder asked for, at exactly the size asked for. Returns
// a failure code when any page could not be written.
int RunScreenshots(VulkanHost& host, const UiFonts& fonts, ImageStore& images,
                   std::span<const std::byte> sample_icon, const GalleryOptions& options) {
	const std::filesystem::path folder = options.screenshot_dir;
	std::error_code             error;
	std::filesystem::create_directories(folder, error);
	if (error) {
		std::fprintf(stderr, "system_ui_gallery: could not create %s (%s)\n",
		             folder.string().c_str(), error.message().c_str());
		return EXIT_FAILURE;
	}
	if (!host.CreateCaptureTarget(options.width, options.height)) {
		return EXIT_FAILURE;
	}
	ScreenshotRun run {
	    host, fonts, {},
	    {},   {},    {static_cast<float>(options.width), static_cast<float>(options.height)}};
	run.state.images      = &images;
	run.state.sample_icon = sample_icon;
	// Baking every glyph first makes the pictures independent of which text a page asks for first,
	// so a change to what gets measured can't shift pixels around glyph edges.
	PrebakeGlyphs(fonts, DEFAULT_THEME.type,
	              ScaleContext(DEFAULT_THEME.scale, static_cast<float>(options.width),
	                           static_cast<float>(options.height)));
	bool all_written = true;
	for (const ScreenshotJob& job: ScreenshotJobs(Pages())) {
		all_written = ScreenshotPage(run, job, folder) && all_written;
	}
	return all_written ? EXIT_SUCCESS : EXIT_FAILURE;
}

// Starts ImGui's SDL and Vulkan backends and runs the gallery in `window`. `sdl_backend_up` is
// set once the SDL backend needs shutting down.
int RunWithImGui(SDL_Window* window, VulkanHost& host, ImageStore& images,
                 const GalleryOptions& options, bool& sdl_backend_up) {
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad | ImGuiConfigFlags_NavEnableKeyboard;
	// The gamepad only activates a control while ImGui's navigation cursor shows.
	io.ConfigNavCursorVisibleAlways = true;
	io.IniFilename                  = nullptr;
	ConfigureNavigation(DEFAULT_THEME);

	const bool screenshots = !options.screenshot_dir.empty();
	// A screenshot run takes no window input, so it leaves the SDL backend off.
	sdl_backend_up = !screenshots && ImGui_ImplSDL3_InitForVulkan(window);
	if ((!screenshots && !sdl_backend_up) || !host.InitImGuiBackend()) {
		std::fprintf(stderr, "system_ui_gallery: could not start the ImGui backends\n");
		return EXIT_FAILURE;
	}
	const UiFonts                fonts       = LoadFonts(io);
	const std::vector<std::byte> sample_icon = LoadSampleIcon();
	if (screenshots) {
		// Screenshots with a fallback font would not show the real design, so a missing font fails.
		const bool fonts_loaded = fonts.regular != nullptr && fonts.bold != nullptr;
		return fonts_loaded ? RunScreenshots(host, fonts, images, sample_icon, options) : EXIT_FAILURE;
	}
	return RunLoop(window, host, fonts, images, sample_icon);
}

// Brings Vulkan and ImGui up for `window`, runs the gallery and tears both down in the order
// the backends need, with ImGui's context still alive while the Vulkan backend shuts down.
int RunWindow(SDL_Window* window, const GalleryOptions& options) {
	VulkanHost host;
	if (!host.Init(window)) {
		return EXIT_FAILURE;
	}
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	bool sdl_backend_up = false;
	// The Vulkan backend reads the store's textures as it shuts down, and the store unregisters them
	// from the ImGui context, so it has to go between the two.
	std::optional<ImageStore> images;
	images.emplace(GALLERY_IMAGE_CAPACITY);
	const int result = RunWithImGui(window, host, *images, options, sdl_backend_up);
	host.Shutdown();
	images.reset();
	if (sdl_backend_up) {
		ImGui_ImplSDL3_Shutdown();
	}
	ImGui::DestroyContext();
	return result;
}

// Opens the window at the size asked for and runs the gallery in it.
int Run(const GalleryOptions& options) {
	if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) {
		std::fprintf(stderr, "system_ui_gallery: SDL_Init failed (%s)\n", SDL_GetError());
		return EXIT_FAILURE;
	}
	// A screenshot run draws offscreen, so its window stays hidden and never appears.
	const bool            screenshots = !options.screenshot_dir.empty();
	const SDL_WindowFlags hidden      = screenshots ? SDL_WINDOW_HIDDEN : 0;
	// The screenshots never use the window's swapchain, so it stays small. That keeps a size
	// beyond the device's limits from failing in the swapchain instead of in the capture check.
	const int   width  = screenshots ? HIDDEN_WINDOW_SIZE : options.width;
	const int   height = screenshots ? HIDDEN_WINDOW_SIZE : options.height;
	SDL_Window* window = SDL_CreateWindow(WINDOW_TITLE, width, height,
	                                      SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | hidden);
	int         result = EXIT_FAILURE;
	if (window == nullptr) {
		std::fprintf(stderr, "system_ui_gallery: could not open a window (%s)\n", SDL_GetError());
	} else {
		result = RunWindow(window, options);
		SDL_DestroyWindow(window);
	}
	SDL_Quit();
	return result;
}

} // namespace

/// Reads the command line and runs the gallery.
int main(int argc, char** argv) {
	std::vector<std::string_view> args(argv + (argc > 0 ? 1 : 0), argv + argc);
	const ParseResult             parsed = ParseGalleryOptions(args);
	if (!parsed.Ok()) {
		std::fprintf(stderr, "system_ui_gallery: %s\n\n%s", parsed.error.c_str(),
		             std::string(GalleryUsage()).c_str());
		return EXIT_USAGE;
	}
	if (parsed.options.help) {
		std::fputs(std::string(GalleryUsage()).c_str(), stdout);
		return EXIT_SUCCESS;
	}
	return Run(parsed.options);
}
