#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_

#include <SDL3/SDL.h>

#include "common/threads.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineLibrary.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Libs::Graphics {

class Presenter;
class RenderContext;

struct SurfaceCapabilities {
	vk::SurfaceCapabilitiesKHR        capabilities {};
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR>   present_modes;
};

struct WindowLoopState {
	SDL_Event             event {};
	bool                  need_exit = false;
	std::atomic<uint64_t> presented_frames {0};
};

// Queries what --graphics-pipeline-library needs from a physical device. The flag is read from the
// config. The query is its own, so a device's shared feature and property chains stay as they were,
// and with the flag off nothing is asked.
[[nodiscard]] PipelineLibrarySupport
QueryPipelineLibrarySupport(vk::PhysicalDevice                          physical_device,
                            const std::vector<vk::ExtensionProperties>& available);

// Decides whether pipelines may be built from libraries, and if so adds the two extensions to
// `device_extensions`. The caller then also enables the graphicsPipelineLibrary feature. Only a
// device meeting every requirement gets them, so any other takes the path it always took. The answer
// is logged and goes to the first-run report. The device creation and the tests call this one.
[[nodiscard]] bool EnablePipelineLibrary(vk::PhysicalDevice                          physical_device,
                                         const std::vector<vk::ExtensionProperties>& available,
                                         std::vector<const char*>&                   device_extensions);

struct WindowContext {
	WindowContext();
	~WindowContext();
	KYTY_CLASS_NO_COPY(WindowContext);

	[[nodiscard]] static vk::PhysicalDeviceVulkan11Features RequiredVulkan11Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan12Features RequiredVulkan12Features() noexcept;
	[[nodiscard]] static vk::PhysicalDeviceVulkan13Features RequiredVulkan13Features() noexcept;
	[[nodiscard]] static uint32_t InitialWindowFlags(bool fullscreen) noexcept;
	void                                                    CreateVulkan();
	void                                                    RecreateSurface();
	void                                                    RefreshSurfaceCapabilities();
	void                                                    UpdateIcon();
	void UpdateTitle(uint64_t frame_num, double current_fps);
	void                                                    Resize(int width, int height);
	void ProcessWindowEvent(const SDL_WindowEvent& event);
	void ProcessDisplayEvent(const SDL_DisplayEvent& event);
	void ProcessEvent(double time_seconds);
	void Run();

	GraphicContext                 graphic_ctx;
	SDL_Window*                    window        = nullptr;
	vk::SurfaceKHR                 surface       = nullptr;
	SurfaceCapabilities            surface_capabilities;
	std::unique_ptr<RenderContext> render_context;
	std::unique_ptr<Presenter>     presenter;
	WindowLoopState                loop;

	Common::Mutex mutex;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_WINDOWINTERNAL_H_
