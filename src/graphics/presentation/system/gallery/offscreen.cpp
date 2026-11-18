#include "graphics/presentation/system/gallery/offscreen.h"

#include <cstdio>
#include <cstring>

namespace Libs::Graphics::SystemUi::Gallery {
namespace {

constexpr size_t   BYTES_PER_PIXEL = 4;
constexpr float    CHANNEL_MAX     = 255.0f;
constexpr float    OPAQUE_ALPHA    = 1.0f;
constexpr uint64_t NO_TIMEOUT      = UINT64_MAX;

// Writes `what` to stderr when `result` isn't success, and returns whether it was.
bool Succeeded(vk::Result result, const char* what) {
	if (result == vk::Result::eSuccess) {
		return true;
	}
	std::fprintf(stderr, "system_ui_gallery: could not %s (%s)\n", what,
	             vk::to_string(result).c_str());
	return false;
}

// The index of a memory type that `type_bits` allows and that has all of `wanted`, or -1.
int FindMemoryType(vk::PhysicalDevice physical_device, uint32_t type_bits,
                   vk::MemoryPropertyFlags wanted) {
	vk::PhysicalDeviceMemoryProperties properties {};
	physical_device.getMemoryProperties(&properties);
	for (uint32_t index = 0; index < properties.memoryTypeCount; index++) {
		const bool allowed = (type_bits & (1u << index)) != 0;
		if (allowed && (properties.memoryTypes[index].propertyFlags & wanted) == wanted) {
			return static_cast<int>(index);
		}
	}
	return -1;
}

// Allocates memory for `requirements` with the properties `wanted`.
bool Allocate(vk::Device device, vk::PhysicalDevice physical_device,
              const vk::MemoryRequirements& requirements, vk::MemoryPropertyFlags wanted,
              vk::DeviceMemory& memory) {
	const int type = FindMemoryType(physical_device, requirements.memoryTypeBits, wanted);
	if (type < 0) {
		std::fprintf(stderr, "system_ui_gallery: no suitable memory type for the screenshot\n");
		return false;
	}
	vk::MemoryAllocateInfo info {};
	info.allocationSize  = requirements.size;
	info.memoryTypeIndex = static_cast<uint32_t>(type);
	return Succeeded(device.allocateMemory(&info, nullptr, &memory), "allocate memory");
}

} // namespace

OffscreenTarget::~OffscreenTarget() {
	Destroy();
}

bool OffscreenTarget::Create(vk::PhysicalDevice physical_device, vk::Device device,
                             uint32_t queue_family, vk::Format format, int width, int height) {
	m_device = device;
	m_format = format;
	m_width  = width;
	m_height = height;
	return CreateImage(physical_device) && CreateBuffer(physical_device) &&
	       CreatePassAndFramebuffer() && CreateCommands(queue_family);
}

bool OffscreenTarget::CreateImage(vk::PhysicalDevice physical_device) {
	vk::ImageCreateInfo info {};
	info.imageType = vk::ImageType::e2D;
	info.format    = m_format;
	info.extent = vk::Extent3D {static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height), 1};
	info.mipLevels   = 1;
	info.arrayLayers = 1;
	info.samples     = vk::SampleCountFlagBits::e1;
	info.tiling      = vk::ImageTiling::eOptimal;
	info.usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc;
	info.initialLayout = vk::ImageLayout::eUndefined;
	if (!Succeeded(m_device.createImage(&info, nullptr, &m_image), "create the screenshot image")) {
		return false;
	}
	vk::MemoryRequirements requirements {};
	m_device.getImageMemoryRequirements(m_image, &requirements);
	if (!Allocate(m_device, physical_device, requirements, vk::MemoryPropertyFlagBits::eDeviceLocal,
	              m_image_memory) ||
	    !Succeeded(m_device.bindImageMemory(m_image, m_image_memory, 0), "bind image memory")) {
		return false;
	}
	vk::ImageViewCreateInfo view {};
	view.image            = m_image;
	view.viewType         = vk::ImageViewType::e2D;
	view.format           = m_format;
	view.subresourceRange = vk::ImageSubresourceRange {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
	return Succeeded(m_device.createImageView(&view, nullptr, &m_view), "create the image view");
}

bool OffscreenTarget::CreateBuffer(vk::PhysicalDevice physical_device) {
	// Rows are copied tightly packed, so the readback pitch is exactly width * 4.
	m_buffer_size = static_cast<size_t>(m_width) * static_cast<size_t>(m_height) * BYTES_PER_PIXEL;
	vk::BufferCreateInfo info {};
	info.size  = m_buffer_size;
	info.usage = vk::BufferUsageFlagBits::eTransferDst;
	if (!Succeeded(m_device.createBuffer(&info, nullptr, &m_buffer),
	               "create the readback buffer")) {
		return false;
	}
	vk::MemoryRequirements requirements {};
	m_device.getBufferMemoryRequirements(m_buffer, &requirements);
	// Cached memory reads fast on the CPU, which matters for tens of megabytes per 4K page, but
	// needs an invalidate after the GPU writes. Coherent memory is the fallback.
	constexpr auto CACHED =
	    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached;
	constexpr auto COHERENT =
	    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
	const bool cached = FindMemoryType(physical_device, requirements.memoryTypeBits, CACHED) >= 0;
	m_coherent        = !cached;
	if (!Allocate(m_device, physical_device, requirements, cached ? CACHED : COHERENT,
	              m_buffer_memory) ||
	    !Succeeded(m_device.bindBufferMemory(m_buffer, m_buffer_memory, 0), "bind buffer memory")) {
		return false;
	}
	return Succeeded(m_device.mapMemory(m_buffer_memory, 0, VK_WHOLE_SIZE, {}, &m_mapped),
	                 "map the readback buffer");
}

bool OffscreenTarget::CreatePassAndFramebuffer() {
	// Same format and samples as the window's pass, which is what makes ImGui's pipeline usable
	// here. Only the final layout differs, as the image is copied rather than presented.
	vk::AttachmentDescription attachment {};
	attachment.format         = m_format;
	attachment.samples        = vk::SampleCountFlagBits::e1;
	attachment.loadOp         = vk::AttachmentLoadOp::eClear;
	attachment.storeOp        = vk::AttachmentStoreOp::eStore;
	attachment.stencilLoadOp  = vk::AttachmentLoadOp::eDontCare;
	attachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	attachment.initialLayout  = vk::ImageLayout::eUndefined;
	attachment.finalLayout    = vk::ImageLayout::eTransferSrcOptimal;
	vk::AttachmentReference reference {0, vk::ImageLayout::eColorAttachmentOptimal};
	vk::SubpassDescription  subpass {};
	subpass.pipelineBindPoint    = vk::PipelineBindPoint::eGraphics;
	subpass.colorAttachmentCount = 1;
	subpass.pColorAttachments    = &reference;
	// Makes the colour writes visible to the copy that follows the pass.
	vk::SubpassDependency dependency {};
	dependency.srcSubpass    = 0;
	dependency.dstSubpass    = VK_SUBPASS_EXTERNAL;
	dependency.srcStageMask  = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependency.dstStageMask  = vk::PipelineStageFlagBits::eTransfer;
	dependency.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
	dependency.dstAccessMask = vk::AccessFlagBits::eTransferRead;
	vk::RenderPassCreateInfo pass {};
	pass.attachmentCount = 1;
	pass.pAttachments    = &attachment;
	pass.subpassCount    = 1;
	pass.pSubpasses      = &subpass;
	pass.dependencyCount = 1;
	pass.pDependencies   = &dependency;
	if (!Succeeded(m_device.createRenderPass(&pass, nullptr, &m_render_pass),
	               "create the screenshot render pass")) {
		return false;
	}
	vk::FramebufferCreateInfo framebuffer {};
	framebuffer.renderPass      = m_render_pass;
	framebuffer.attachmentCount = 1;
	framebuffer.pAttachments    = &m_view;
	framebuffer.width           = static_cast<uint32_t>(m_width);
	framebuffer.height          = static_cast<uint32_t>(m_height);
	framebuffer.layers          = 1;
	return Succeeded(m_device.createFramebuffer(&framebuffer, nullptr, &m_framebuffer),
	                 "create the screenshot framebuffer");
}

bool OffscreenTarget::CreateCommands(uint32_t queue_family) {
	vk::CommandPoolCreateInfo pool {};
	pool.queueFamilyIndex = queue_family;
	if (!Succeeded(m_device.createCommandPool(&pool, nullptr, &m_pool), "create a command pool")) {
		return false;
	}
	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = 1;
	vk::FenceCreateInfo fence {};
	return Succeeded(m_device.allocateCommandBuffers(&allocate, &m_commands),
	                 "allocate a command buffer") &&
	       Succeeded(m_device.createFence(&fence, nullptr, &m_fence), "create a fence");
}

bool OffscreenTarget::Record(ImDrawData* draw_data, const Color& clear) {
	vk::CommandBufferBeginInfo begin {};
	begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
	if (!Succeeded(m_commands.begin(&begin), "begin the command buffer")) {
		return false;
	}
	vk::ClearValue clear_value {};
	clear_value.color.float32 = {{static_cast<float>(clear.r) / CHANNEL_MAX,
	                              static_cast<float>(clear.g) / CHANNEL_MAX,
	                              static_cast<float>(clear.b) / CHANNEL_MAX, OPAQUE_ALPHA}};
	vk::RenderPassBeginInfo pass {};
	pass.renderPass  = m_render_pass;
	pass.framebuffer = m_framebuffer;
	pass.renderArea.extent =
	    vk::Extent2D {static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height)};
	pass.clearValueCount = 1;
	pass.pClearValues    = &clear_value;
	m_commands.beginRenderPass(&pass, vk::SubpassContents::eInline);
	ImGui_ImplVulkan_RenderDrawData(draw_data, m_commands);
	m_commands.endRenderPass();

	vk::BufferImageCopy copy {};
	copy.imageSubresource = vk::ImageSubresourceLayers {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
	copy.imageExtent =
	    vk::Extent3D {static_cast<uint32_t>(m_width), static_cast<uint32_t>(m_height), 1};
	m_commands.copyImageToBuffer(m_image, vk::ImageLayout::eTransferSrcOptimal, m_buffer, 1, &copy);
	// Makes the copy's writes available to the host read after the fence.
	vk::MemoryBarrier barrier {};
	barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
	barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
	m_commands.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
	                           vk::PipelineStageFlagBits::eHost, {}, 1, &barrier, 0, nullptr, 0,
	                           nullptr);
	return Succeeded(m_commands.end(), "end the command buffer");
}

bool OffscreenTarget::Capture(vk::Queue queue, ImDrawData* draw_data, const Color& clear,
                              CapturedImage& out) {
	if (!Succeeded(m_device.resetCommandPool(m_pool, {}), "reset the command pool") ||
	    !Record(draw_data, clear)) {
		return false;
	}
	vk::SubmitInfo submit {};
	submit.commandBufferCount = 1;
	submit.pCommandBuffers    = &m_commands;
	if (!Succeeded(queue.submit(1, &submit, m_fence), "submit the screenshot") ||
	    !Succeeded(m_device.waitForFences(1, &m_fence, VK_TRUE, NO_TIMEOUT),
	               "wait for the screenshot") ||
	    !Succeeded(m_device.resetFences(1, &m_fence), "reset the fence")) {
		return false;
	}
	if (!m_coherent) {
		vk::MappedMemoryRange range {};
		range.memory = m_buffer_memory;
		range.offset = 0;
		range.size   = VK_WHOLE_SIZE;
		if (!Succeeded(m_device.invalidateMappedMemoryRanges(1, &range),
		               "invalidate the readback")) {
			return false;
		}
	}
	out.pixels.resize(m_buffer_size);
	std::memcpy(out.pixels.data(), m_mapped, m_buffer_size);
	out.width     = m_width;
	out.height    = m_height;
	out.row_pitch = static_cast<size_t>(m_width) * BYTES_PER_PIXEL;
	out.bgra      = m_format == vk::Format::eB8G8R8A8Unorm;
	return true;
}

void OffscreenTarget::Destroy() {
	if (!m_device) {
		return;
	}
	m_device.destroyFence(m_fence);
	m_device.destroyCommandPool(m_pool);
	m_device.destroyFramebuffer(m_framebuffer);
	m_device.destroyRenderPass(m_render_pass);
	m_device.destroyImageView(m_view);
	m_device.destroyImage(m_image);
	m_device.freeMemory(m_image_memory);
	if (m_mapped != nullptr) {
		m_device.unmapMemory(m_buffer_memory);
		m_mapped = nullptr;
	}
	m_device.destroyBuffer(m_buffer);
	m_device.freeMemory(m_buffer_memory);
	m_device = nullptr;
}

} // namespace Libs::Graphics::SystemUi::Gallery
