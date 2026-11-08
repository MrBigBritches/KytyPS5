#ifndef KYTY_TESTS_COMMANDSTREAMRECORDER_H_
#define KYTY_TESTS_COMMANDSTREAMRECORDER_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

// Records the Vulkan commands a test records, as text rows. The driver is still called, so the
// commands run. Handles are numbered in the order they first appear, or take a name the test gave
// them, so two runs of the same scene give the same rows whatever the driver handed out.
namespace CommandStream {

struct Recorder {
  static inline bool active = false;
  static inline std::vector<std::string> rows;
  static inline std::map<uint64_t, std::string> labels;
  static inline std::map<std::string, uint32_t> counts;

  // Forgets the rows and every handle seen so far, and starts or stops recording.
  static void Reset(bool on) {
    rows.clear();
    labels.clear();
    counts.clear();
    active = on;
  }

  // Gives a handle a name that stands for what owns it, instead of its place in the stream.
  static void Name(const void *handle, const std::string &label) {
    labels[reinterpret_cast<uint64_t>(handle)] = label;
  }

  static std::string Label(const char *kind, uint64_t raw) {
    if (raw == 0) return "null";
    const auto found = labels.find(raw);
    if (found != labels.end()) return found->second;
    auto label = std::string(kind) + "#" + std::to_string(counts[kind]++);
    labels.emplace(raw, label);
    return label;
  }
};

#define KYTY_STREAM_HANDLE(Type, Kind) \
  inline std::string FormatArg(Type handle) { \
    return Recorder::Label(Kind, reinterpret_cast<uint64_t>(handle)); \
  }
KYTY_STREAM_HANDLE(VkBuffer, "buffer")
KYTY_STREAM_HANDLE(VkImage, "image")
KYTY_STREAM_HANDLE(VkImageView, "view")
KYTY_STREAM_HANDLE(VkSampler, "sampler")
KYTY_STREAM_HANDLE(VkPipeline, "pipeline")
KYTY_STREAM_HANDLE(VkPipelineLayout, "layout")
KYTY_STREAM_HANDLE(VkDescriptorSet, "set")
KYTY_STREAM_HANDLE(VkQueryPool, "pool")
#undef KYTY_STREAM_HANDLE

// The command buffer says nothing about the scene, so it is left out of every row.
inline std::string FormatArg(VkCommandBuffer) { return {}; }

template <typename T>
  requires(std::is_arithmetic_v<T> || std::is_enum_v<T>)
std::string FormatArg(T value) {
  if constexpr (std::is_floating_point_v<T>) {
    return std::to_string(static_cast<double>(value));
  } else if constexpr (std::is_enum_v<T>) {
    return std::to_string(static_cast<int64_t>(value));
  } else {
    return std::to_string(value);
  }
}

// A data pointer is not followed here. Commands whose data matters are written out below.
template <typename T>
  requires std::is_pointer_v<T>
std::string FormatArg(T pointer) {
  return pointer == nullptr ? "null" : "*";
}

inline std::string Join(const char *name, std::initializer_list<std::string> parts) {
  std::string row = std::string(name) + "(";
  bool first = true;
  for (const auto &part : parts) {
    if (part.empty()) continue;
    row += (first ? "" : ", ") + part;
    first = false;
  }
  return row + ")";
}

inline void Add(std::string row) {
  if (Recorder::active) Recorder::rows.push_back(std::move(row));
}

// Wraps one entry of the default dispatcher whose arguments are all plain values or handles.
template <auto Member, typename Fn> struct Hook;
template <auto Member, typename... Args>
struct Hook<Member, void(VKAPI_PTR *)(Args...)> {
  static inline void(VKAPI_PTR *real)(Args...) = nullptr;
  static inline const char *name = nullptr;

  static void VKAPI_CALL Call(Args... args) {
    if (Recorder::active) Add(Join(name, {FormatArg(args)...}));
    real(args...);
  }
  static void Install(const char *entry) {
    auto &table = VULKAN_HPP_DEFAULT_DISPATCHER;
    name = entry;
    real = table.*Member;
    table.*Member = &Call;
  }
  static void Remove() { VULKAN_HPP_DEFAULT_DISPATCHER.*Member = real; }
};

inline std::string Floats(const float *values, uint32_t count) {
  std::string text;
  for (uint32_t i = 0; i < count; ++i) text += (i ? " " : "") + std::to_string(values[i]);
  return text;
}

inline std::string Bytes(const void *data, uint32_t size) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string text;
  const auto *bytes = static_cast<const uint8_t *>(data);
  for (uint32_t i = 0; i < size; ++i) {
    text += digits[bytes[i] >> 4];
    text += digits[bytes[i] & 15];
  }
  return text;
}

inline std::string DescribeWrite(const VkWriteDescriptorSet &write) {
  std::string text = "bind=" + std::to_string(write.dstBinding) + " " +
                     vk::to_string(static_cast<vk::DescriptorType>(write.descriptorType)) +
                     " n=" + std::to_string(write.descriptorCount);
  for (uint32_t i = 0; i < write.descriptorCount; ++i) {
    if (write.pImageInfo != nullptr) {
      const auto &info = write.pImageInfo[i];
      text += " [view=" + FormatArg(info.imageView) + " sampler=" + FormatArg(info.sampler) +
              " layout=" + vk::to_string(static_cast<vk::ImageLayout>(info.imageLayout)) + "]";
    } else if (write.pBufferInfo != nullptr) {
      const auto &info = write.pBufferInfo[i];
      text += " [buffer=" + FormatArg(info.buffer) + " +" + std::to_string(info.offset) + "/" +
              (info.range == VK_WHOLE_SIZE ? std::string("whole") : std::to_string(info.range)) +
              "]";
    }
  }
  return text;
}

inline std::string DescribeAttachment(const VkRenderingAttachmentInfo &info) {
  return "{view=" + FormatArg(info.imageView) +
         " layout=" + vk::to_string(static_cast<vk::ImageLayout>(info.imageLayout)) +
         " load=" + vk::to_string(static_cast<vk::AttachmentLoadOp>(info.loadOp)) +
         " store=" + vk::to_string(static_cast<vk::AttachmentStoreOp>(info.storeOp)) +
         " depth=" + std::to_string(info.clearValue.depthStencil.depth) +
         " stencil=" + std::to_string(info.clearValue.depthStencil.stencil) + "}";
}

inline std::string DescribeDependency(const VkDependencyInfo &info) {
  std::string text;
  for (uint32_t i = 0; i < info.memoryBarrierCount; ++i) {
    const auto &barrier = info.pMemoryBarriers[i];
    text += " mem{" + vk::to_string(static_cast<vk::PipelineStageFlags2>(barrier.srcStageMask)) +
            "/" + vk::to_string(static_cast<vk::AccessFlags2>(barrier.srcAccessMask)) + " -> " +
            vk::to_string(static_cast<vk::PipelineStageFlags2>(barrier.dstStageMask)) + "/" +
            vk::to_string(static_cast<vk::AccessFlags2>(barrier.dstAccessMask)) + "}";
  }
  for (uint32_t i = 0; i < info.bufferMemoryBarrierCount; ++i) {
    const auto &barrier = info.pBufferMemoryBarriers[i];
    text += " buf{" + FormatArg(barrier.buffer) + " +" + std::to_string(barrier.offset) + "/" +
            std::to_string(barrier.size) + " " +
            vk::to_string(static_cast<vk::AccessFlags2>(barrier.srcAccessMask)) + " -> " +
            vk::to_string(static_cast<vk::AccessFlags2>(barrier.dstAccessMask)) + "}";
  }
  for (uint32_t i = 0; i < info.imageMemoryBarrierCount; ++i) {
    const auto &barrier = info.pImageMemoryBarriers[i];
    const auto &range = barrier.subresourceRange;
    text += " img{" + FormatArg(barrier.image) + " " +
            vk::to_string(static_cast<vk::ImageLayout>(barrier.oldLayout)) + " -> " +
            vk::to_string(static_cast<vk::ImageLayout>(barrier.newLayout)) + " " +
            vk::to_string(static_cast<vk::PipelineStageFlags2>(barrier.srcStageMask)) + "/" +
            vk::to_string(static_cast<vk::AccessFlags2>(barrier.srcAccessMask)) + " -> " +
            vk::to_string(static_cast<vk::PipelineStageFlags2>(barrier.dstStageMask)) + "/" +
            vk::to_string(static_cast<vk::AccessFlags2>(barrier.dstAccessMask)) + " " +
            vk::to_string(static_cast<vk::ImageAspectFlags>(range.aspectMask)) + " mip " +
            std::to_string(range.baseMipLevel) + "+" + std::to_string(range.levelCount) +
            " layer " + std::to_string(range.baseArrayLayer) + "+" +
            std::to_string(range.layerCount) + "}";
  }
  return text;
}

// Hand-written wrappers for the commands whose data matters, which a plain hook would hide.
struct PushConstantsHook {
  static inline PFN_vkCmdPushConstants real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, VkPipelineLayout layout,
                              VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                              const void *values) {
    Add("vkCmdPushConstants(" + FormatArg(layout) + ", " +
        vk::to_string(static_cast<vk::ShaderStageFlags>(stages)) + ", +" +
        std::to_string(offset) + "/" + std::to_string(size) + ", " + Bytes(values, size) + ")");
    real(cb, layout, stages, offset, size, values);
  }
};

struct PushDescriptorHook {
  static inline PFN_vkCmdPushDescriptorSetKHR real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, VkPipelineBindPoint point,
                              VkPipelineLayout layout, uint32_t set, uint32_t count,
                              const VkWriteDescriptorSet *writes) {
    std::string text = "vkCmdPushDescriptorSet(" +
                       vk::to_string(static_cast<vk::PipelineBindPoint>(point)) + ", " +
                       FormatArg(layout) + ", set " + std::to_string(set);
    for (uint32_t i = 0; i < count; ++i) text += "; " + DescribeWrite(writes[i]);
    Add(text + ")");
    real(cb, point, layout, set, count, writes);
  }
};

struct BindDescriptorSetsHook {
  static inline PFN_vkCmdBindDescriptorSets real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, VkPipelineBindPoint point,
                              VkPipelineLayout layout, uint32_t first, uint32_t count,
                              const VkDescriptorSet *sets, uint32_t dynamic_count,
                              const uint32_t *dynamic_offsets) {
    std::string text = "vkCmdBindDescriptorSets(" +
                       vk::to_string(static_cast<vk::PipelineBindPoint>(point)) + ", " +
                       FormatArg(layout) + ", first " + std::to_string(first);
    for (uint32_t i = 0; i < count; ++i) text += " " + FormatArg(sets[i]);
    Add(text + ", dynamic " + std::to_string(dynamic_count) + ")");
    real(cb, point, layout, first, count, sets, dynamic_count, dynamic_offsets);
  }
};

struct BindVertexBuffersHook {
  static inline PFN_vkCmdBindVertexBuffers2 real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, uint32_t first, uint32_t count,
                              const VkBuffer *buffers, const VkDeviceSize *offsets,
                              const VkDeviceSize *sizes, const VkDeviceSize *strides) {
    std::string text = "vkCmdBindVertexBuffers2(first " + std::to_string(first);
    for (uint32_t i = 0; i < count; ++i) {
      text += "; " + FormatArg(buffers[i]) + " +" + std::to_string(offsets[i]) + "/" +
              (sizes ? std::to_string(sizes[i]) : std::string("whole")) +
              (strides ? " stride " + std::to_string(strides[i]) : std::string());
    }
    Add(text + ")");
    real(cb, first, count, buffers, offsets, sizes, strides);
  }
};

struct BeginRenderingHook {
  static inline PFN_vkCmdBeginRendering real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, const VkRenderingInfo *info) {
    std::string text = "vkCmdBeginRendering(" + std::to_string(info->renderArea.extent.width) +
                       "x" + std::to_string(info->renderArea.extent.height) + " layers " +
                       std::to_string(info->layerCount);
    for (uint32_t i = 0; i < info->colorAttachmentCount; ++i) {
      text += " color" + std::to_string(i) + DescribeAttachment(info->pColorAttachments[i]);
    }
    if (info->pDepthAttachment) text += " depth" + DescribeAttachment(*info->pDepthAttachment);
    if (info->pStencilAttachment) text += " stencil" + DescribeAttachment(*info->pStencilAttachment);
    Add(text + ")");
    real(cb, info);
  }
};

struct PipelineBarrier2Hook {
  static inline PFN_vkCmdPipelineBarrier2 real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, const VkDependencyInfo *info) {
    Add("vkCmdPipelineBarrier2(" + DescribeDependency(*info) + " )");
    real(cb, info);
  }
};

struct PipelineBarrierHook {
  static inline PFN_vkCmdPipelineBarrier real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, VkPipelineStageFlags src,
                              VkPipelineStageFlags dst, VkDependencyFlags flags,
                              uint32_t memory_count, const VkMemoryBarrier *memory,
                              uint32_t buffer_count, const VkBufferMemoryBarrier *buffers,
                              uint32_t image_count, const VkImageMemoryBarrier *images) {
    Add("vkCmdPipelineBarrier(" + vk::to_string(static_cast<vk::PipelineStageFlags>(src)) +
        " -> " + vk::to_string(static_cast<vk::PipelineStageFlags>(dst)) + " memory " +
        std::to_string(memory_count) + " buffers " + std::to_string(buffer_count) + " images " +
        std::to_string(image_count) + ")");
    real(cb, src, dst, flags, memory_count, memory, buffer_count, buffers, image_count, images);
  }
};

struct ViewportsHook {
  static inline PFN_vkCmdSetViewportWithCount real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, uint32_t count, const VkViewport *viewports) {
    std::string text = "vkCmdSetViewportWithCount(";
    for (uint32_t i = 0; i < count; ++i) {
      const float values[] = {viewports[i].x,     viewports[i].y,        viewports[i].width,
                              viewports[i].height, viewports[i].minDepth, viewports[i].maxDepth};
      text += (i ? "; " : "") + Floats(values, 6);
    }
    Add(text + ")");
    real(cb, count, viewports);
  }
};

struct ScissorsHook {
  static inline PFN_vkCmdSetScissorWithCount real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, uint32_t count, const VkRect2D *scissors) {
    std::string text = "vkCmdSetScissorWithCount(";
    for (uint32_t i = 0; i < count; ++i) {
      text += std::string(i ? "; " : "") + std::to_string(scissors[i].offset.x) + "," +
              std::to_string(scissors[i].offset.y) + " " +
              std::to_string(scissors[i].extent.width) + "x" +
              std::to_string(scissors[i].extent.height);
    }
    Add(text + ")");
    real(cb, count, scissors);
  }
};

struct BlendConstantsHook {
  static inline PFN_vkCmdSetBlendConstants real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, const float constants[4]) {
    Add("vkCmdSetBlendConstants(" + Floats(constants, 4) + ")");
    real(cb, constants);
  }
};

struct ColorWriteEnableHook {
  static inline PFN_vkCmdSetColorWriteEnableEXT real = nullptr;
  static void VKAPI_CALL Call(VkCommandBuffer cb, uint32_t count, const VkBool32 *enables) {
    std::string text = "vkCmdSetColorWriteEnableEXT(";
    for (uint32_t i = 0; i < count; ++i) text += (i ? " " : "") + std::to_string(enables[i]);
    Add(text + ")");
    real(cb, count, enables);
  }
};

// Puts the recording hooks in the default dispatcher for as long as it lives. Every command the
// renderer records between the draws and the barriers they need goes through these entries, and
// each row names the command and the values the scene decides.
class Scope {
public:
  Scope() {
    auto &table = VULKAN_HPP_DEFAULT_DISPATCHER;
#define KYTY_STREAM_MANUAL(Hook, Entry) \
  Hook::real = table.Entry; \
  table.Entry = &Hook::Call
    KYTY_STREAM_MANUAL(PushConstantsHook, vkCmdPushConstants);
    KYTY_STREAM_MANUAL(PushDescriptorHook, vkCmdPushDescriptorSetKHR);
    KYTY_STREAM_MANUAL(BindDescriptorSetsHook, vkCmdBindDescriptorSets);
    KYTY_STREAM_MANUAL(BindVertexBuffersHook, vkCmdBindVertexBuffers2);
    KYTY_STREAM_MANUAL(BeginRenderingHook, vkCmdBeginRendering);
    KYTY_STREAM_MANUAL(PipelineBarrier2Hook, vkCmdPipelineBarrier2);
    KYTY_STREAM_MANUAL(PipelineBarrierHook, vkCmdPipelineBarrier);
    KYTY_STREAM_MANUAL(ViewportsHook, vkCmdSetViewportWithCount);
    KYTY_STREAM_MANUAL(ScissorsHook, vkCmdSetScissorWithCount);
    KYTY_STREAM_MANUAL(BlendConstantsHook, vkCmdSetBlendConstants);
    KYTY_STREAM_MANUAL(ColorWriteEnableHook, vkCmdSetColorWriteEnableEXT);
#undef KYTY_STREAM_MANUAL
#define KYTY_STREAM_PLAIN(Entry) \
  Hook<&vk::detail::DispatchLoaderDynamic::Entry, decltype(table.Entry)>::Install(#Entry)
    KYTY_STREAM_PLAIN(vkCmdEndRendering);
    KYTY_STREAM_PLAIN(vkCmdBindPipeline);
    KYTY_STREAM_PLAIN(vkCmdBindIndexBuffer);
    KYTY_STREAM_PLAIN(vkCmdDraw);
    KYTY_STREAM_PLAIN(vkCmdDrawIndexed);
    KYTY_STREAM_PLAIN(vkCmdDrawIndirect);
    KYTY_STREAM_PLAIN(vkCmdDrawIndexedIndirect);
    KYTY_STREAM_PLAIN(vkCmdDrawMeshTasksEXT);
    KYTY_STREAM_PLAIN(vkCmdSetDepthTestEnable);
    KYTY_STREAM_PLAIN(vkCmdSetDepthWriteEnable);
    KYTY_STREAM_PLAIN(vkCmdSetDepthCompareOp);
    KYTY_STREAM_PLAIN(vkCmdSetDepthBoundsTestEnable);
    KYTY_STREAM_PLAIN(vkCmdSetDepthBounds);
    KYTY_STREAM_PLAIN(vkCmdSetDepthBiasEnable);
    KYTY_STREAM_PLAIN(vkCmdSetDepthBias);
    KYTY_STREAM_PLAIN(vkCmdSetStencilTestEnable);
    KYTY_STREAM_PLAIN(vkCmdSetStencilOp);
    KYTY_STREAM_PLAIN(vkCmdSetStencilReference);
    KYTY_STREAM_PLAIN(vkCmdSetStencilWriteMask);
    KYTY_STREAM_PLAIN(vkCmdSetStencilCompareMask);
    KYTY_STREAM_PLAIN(vkCmdSetCullMode);
    KYTY_STREAM_PLAIN(vkCmdSetFrontFace);
    KYTY_STREAM_PLAIN(vkCmdSetLineWidth);
    KYTY_STREAM_PLAIN(vkCmdSetAttachmentFeedbackLoopEnableEXT);
    KYTY_STREAM_PLAIN(vkCmdBeginQuery);
    KYTY_STREAM_PLAIN(vkCmdEndQuery);
    KYTY_STREAM_PLAIN(vkCmdFillBuffer);
#undef KYTY_STREAM_PLAIN
  }
  ~Scope() {
    auto &table = VULKAN_HPP_DEFAULT_DISPATCHER;
#define KYTY_STREAM_MANUAL(Hook, Entry) table.Entry = Hook::real
    KYTY_STREAM_MANUAL(PushConstantsHook, vkCmdPushConstants);
    KYTY_STREAM_MANUAL(PushDescriptorHook, vkCmdPushDescriptorSetKHR);
    KYTY_STREAM_MANUAL(BindDescriptorSetsHook, vkCmdBindDescriptorSets);
    KYTY_STREAM_MANUAL(BindVertexBuffersHook, vkCmdBindVertexBuffers2);
    KYTY_STREAM_MANUAL(BeginRenderingHook, vkCmdBeginRendering);
    KYTY_STREAM_MANUAL(PipelineBarrier2Hook, vkCmdPipelineBarrier2);
    KYTY_STREAM_MANUAL(PipelineBarrierHook, vkCmdPipelineBarrier);
    KYTY_STREAM_MANUAL(ViewportsHook, vkCmdSetViewportWithCount);
    KYTY_STREAM_MANUAL(ScissorsHook, vkCmdSetScissorWithCount);
    KYTY_STREAM_MANUAL(BlendConstantsHook, vkCmdSetBlendConstants);
    KYTY_STREAM_MANUAL(ColorWriteEnableHook, vkCmdSetColorWriteEnableEXT);
#undef KYTY_STREAM_MANUAL
#define KYTY_STREAM_PLAIN(Entry) \
  Hook<&vk::detail::DispatchLoaderDynamic::Entry, decltype(table.Entry)>::Remove()
    KYTY_STREAM_PLAIN(vkCmdEndRendering);
    KYTY_STREAM_PLAIN(vkCmdBindPipeline);
    KYTY_STREAM_PLAIN(vkCmdBindIndexBuffer);
    KYTY_STREAM_PLAIN(vkCmdDraw);
    KYTY_STREAM_PLAIN(vkCmdDrawIndexed);
    KYTY_STREAM_PLAIN(vkCmdDrawIndirect);
    KYTY_STREAM_PLAIN(vkCmdDrawIndexedIndirect);
    KYTY_STREAM_PLAIN(vkCmdDrawMeshTasksEXT);
    KYTY_STREAM_PLAIN(vkCmdSetDepthTestEnable);
    KYTY_STREAM_PLAIN(vkCmdSetDepthWriteEnable);
    KYTY_STREAM_PLAIN(vkCmdSetDepthCompareOp);
    KYTY_STREAM_PLAIN(vkCmdSetDepthBoundsTestEnable);
    KYTY_STREAM_PLAIN(vkCmdSetDepthBounds);
    KYTY_STREAM_PLAIN(vkCmdSetDepthBiasEnable);
    KYTY_STREAM_PLAIN(vkCmdSetDepthBias);
    KYTY_STREAM_PLAIN(vkCmdSetStencilTestEnable);
    KYTY_STREAM_PLAIN(vkCmdSetStencilOp);
    KYTY_STREAM_PLAIN(vkCmdSetStencilReference);
    KYTY_STREAM_PLAIN(vkCmdSetStencilWriteMask);
    KYTY_STREAM_PLAIN(vkCmdSetStencilCompareMask);
    KYTY_STREAM_PLAIN(vkCmdSetCullMode);
    KYTY_STREAM_PLAIN(vkCmdSetFrontFace);
    KYTY_STREAM_PLAIN(vkCmdSetLineWidth);
    KYTY_STREAM_PLAIN(vkCmdSetAttachmentFeedbackLoopEnableEXT);
    KYTY_STREAM_PLAIN(vkCmdBeginQuery);
    KYTY_STREAM_PLAIN(vkCmdEndQuery);
    KYTY_STREAM_PLAIN(vkCmdFillBuffer);
#undef KYTY_STREAM_PLAIN
    Recorder::Reset(false);
  }
  Scope(const Scope &) = delete;
  Scope &operator=(const Scope &) = delete;
};

} // namespace CommandStream

#endif // KYTY_TESTS_COMMANDSTREAMRECORDER_H_
