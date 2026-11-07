#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_VULKANHOST_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_VULKANHOST_H_

// These match host_gpu/vulkanCommon.h, so Vulkan is called through function pointers loaded at
// run time and the gallery needs no Vulkan import library.
#if defined(VULKAN_H_) && !defined(VK_NO_PROTOTYPES)
#error "vulkan.h was included before VK_NO_PROTOTYPES was defined"
#endif
#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#ifndef VULKAN_HPP_DISPATCH_LOADER_DYNAMIC
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#endif
#ifndef VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL
#define VULKAN_HPP_ENABLE_DYNAMIC_LOADER_TOOL 1
#endif
#ifndef VULKAN_HPP_NO_CONSTRUCTORS
#define VULKAN_HPP_NO_CONSTRUCTORS
#endif
#ifndef VULKAN_HPP_NO_EXCEPTIONS
#define VULKAN_HPP_NO_EXCEPTIONS
#endif

#include "graphics/presentation/system/core/color.h"
#include "imgui.h"
#include "imgui_impl_vulkan.h"

#include <cstdint>
#include <memory>
#include <vulkan/vulkan.hpp>

struct SDL_Window;

namespace Libs::Graphics::SystemUi::Gallery {

// Descriptors for images the gallery's pages load, such as sample art, on top of the backend's own
// font textures. The gallery's image store holds no more than this.
inline constexpr uint32_t GALLERY_IMAGE_CAPACITY = 64;

/// Owns the Vulkan instance, device, queue, surface and swapchain for one SDL window, and draws
/// ImGui's output into it. Every failure is written to stderr and reported by a false return, as
/// exceptions are off. The swapchain format is 8-bit UNORM and presenting is FIFO.
/// What a frame step did. Skip means this frame wasn't drawn but the next may be, as when the
/// window is minimised or the swapchain needs rebuilding. Fatal means Vulkan can not continue and
/// the caller should stop.
enum class FrameResult : uint8_t { Ok, Skip, Fatal };

struct CapturedImage;
class OffscreenTarget;

class VulkanHost {
public:
	VulkanHost();
	~VulkanHost();
	VulkanHost(const VulkanHost&)            = delete;
	VulkanHost& operator=(const VulkanHost&) = delete;

	/// Brings Vulkan up for `window`, which must have been created with SDL_WINDOW_VULKAN, and
	/// builds its swapchain at the window's pixel size. Safe to call once.
	[[nodiscard]] bool Init(SDL_Window* window);

	/// Starts ImGui's Vulkan backend. Needs a current ImGui context and a successful Init.
	[[nodiscard]] bool InitImGuiBackend();

	/// Prepares a frame, rebuilding the swapchain first if the window was resized or the last
	/// present found it out of date. Call before ImGui::NewFrame. Returns false when the window
	/// has no area, as when minimised, and the frame should be skipped.
	[[nodiscard]] FrameResult BeginFrame();

	/// Draws `draw_data` over a `clear` background and presents it. An out of date swapchain
	/// skips the frame and is rebuilt by the next BeginFrame.
	[[nodiscard]] FrameResult Render(ImDrawData* draw_data, const Color& clear);

	/// Makes an offscreen image of `width` by `height` for Capture, in the swapchain's format.
	/// Needs a successful InitImGuiBackend, whose pipeline it must match.
	[[nodiscard]] bool CreateCaptureTarget(int width, int height);

	/// Prepares ImGui's Vulkan backend for a frame drawn with Capture. Unlike BeginFrame it
	/// ignores the window, which may be hidden. Call before ImGui::NewFrame.
	void BeginCaptureFrame();

	/// Draws `draw_data` over `clear` into the capture target, waits for the GPU, and reads the
	/// pixels into `out`. Nothing is presented.
	[[nodiscard]] bool Capture(ImDrawData* draw_data, const Color& clear, CapturedImage& out);

	/// Waits for the GPU to go idle, then releases everything. The ImGui context that
	/// InitImGuiBackend used must still be current. Safe to call more than once.
	void Shutdown();

private:
	bool        CreateInstance();
	bool        PickDevice();
	bool        CreateDevice();
	bool        CreateSurface();
	bool        CreateSwapchain(int width, int height);
	bool        BuildSwapchain(int width, int height);
	bool        ResizeSwapchain(int width, int height);
	bool        LoadImGuiFunctions();
	FrameResult AcquireImage(const ImGui_ImplVulkanH_Frame*& frame);
	FrameResult RecordAndSubmit(const ImGui_ImplVulkanH_Frame& frame, ImDrawData* draw_data);
	FrameResult Present();

	SDL_Window*                      m_window = nullptr;
	vk::Instance                     m_instance;
	vk::PhysicalDevice               m_physical_device;
	vk::Device                       m_device;
	vk::Queue                        m_queue;
	vk::SurfaceKHR                   m_surface;
	uint32_t                         m_queue_family = 0;
	ImGui_ImplVulkanH_Window         m_window_data;
	bool                             m_swapchain_built = false;
	bool                             m_imgui_backend   = false;
	int                              m_built_width     = 0;
	int                              m_built_height    = 0;
	bool                             m_rebuild         = false;
	std::unique_ptr<OffscreenTarget> m_capture;
};

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_VULKANHOST_H_
