#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_OFFSCREEN_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_OFFSCREEN_H_

#include "graphics/presentation/system/gallery/capture.h"
#include "graphics/presentation/system/gallery/vulkanHost.h"

namespace Libs::Graphics::SystemUi::Gallery {

/// An image of a fixed size that ImGui can draw into and the CPU can read back, independent of
/// any window or swapchain. Its render pass matches the one ImGui's pipeline was built for, which
/// needs the same colour format and sample count.
class OffscreenTarget {
public:
	OffscreenTarget() = default;
	~OffscreenTarget();
	OffscreenTarget(const OffscreenTarget&)            = delete;
	OffscreenTarget& operator=(const OffscreenTarget&) = delete;

	/// Makes the image, render pass, framebuffer, readback buffer and command objects for a
	/// `width` by `height` image of `format`. Failures are written to stderr.
	[[nodiscard]] bool Create(vk::PhysicalDevice physical_device, vk::Device device,
	                          uint32_t queue_family, vk::Format format, int width, int height);

	/// Draws `draw_data` over `clear`, waits for the GPU, and reads the result into `out`.
	[[nodiscard]] bool Capture(vk::Queue queue, ImDrawData* draw_data, const Color& clear,
	                           CapturedImage& out);

	/// Releases everything. Safe to call more than once, and the device must be idle.
	void Destroy();

private:
	bool CreateImage(vk::PhysicalDevice physical_device);
	bool CreateBuffer(vk::PhysicalDevice physical_device);
	bool CreatePassAndFramebuffer();
	bool CreateCommands(uint32_t queue_family);
	bool Record(ImDrawData* draw_data, const Color& clear);

	vk::Device       m_device;
	vk::Format       m_format = vk::Format::eUndefined;
	int              m_width  = 0;
	int              m_height = 0;
	vk::Image        m_image;
	vk::DeviceMemory m_image_memory;
	vk::ImageView    m_view;
	vk::RenderPass   m_render_pass;
	vk::Framebuffer  m_framebuffer;
	vk::Buffer       m_buffer;
	vk::DeviceMemory m_buffer_memory;
	void*            m_mapped      = nullptr;
	size_t           m_buffer_size = 0;
	// Whether the readback memory is coherent. When it isn't, the CPU must invalidate it first.
	bool              m_coherent = true;
	vk::CommandPool   m_pool;
	vk::CommandBuffer m_commands;
	vk::Fence         m_fence;
};

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_OFFSCREEN_H_
