#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_VULKANVALIDATION_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_VULKANVALIDATION_H_

#include "graphics/host_gpu/vulkanCommon.h"

#include <cstdint>
#include <string>
#include <vector>

// A view of the Vulkan validation layer's messages for tests. The debug callback counts every
// validation warning and error. In the emulator the first error still ends the process, and a test
// turns that off so it can read the count instead.
namespace Libs::Graphics::VulkanValidation {

/// The most message texts kept, so a flood of the same hazard does not use up memory.
inline constexpr size_t MAX_KEPT_MESSAGES = 32;

/// The start of the message id of every synchronization hazard the layer reports.
inline constexpr const char* HAZARD_ID_PREFIX = "SYNC-HAZARD";

/// Validation warnings and errors the callback has seen since the last reset.
[[nodiscard]] uint64_t MessageCount();

/// How many of them were synchronization hazards, told by message id and not by wording.
[[nodiscard]] uint64_t HazardCount();

/// The first MAX_KEPT_MESSAGES message texts seen since the last reset.
[[nodiscard]] std::vector<std::string> Messages();

/// Zeroes the count and forgets the kept texts.
void ResetMessages();

/// Makes validation errors count and be kept instead of ending the process. For tests only.
void SetCountOnly(bool count_only);

/// The debug messenger callback, for a test instance to register.
[[nodiscard]] vk::PFN_DebugUtilsMessengerCallbackEXT Callback();

} // namespace Libs::Graphics::VulkanValidation

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_WINDOW_VULKANVALIDATION_H_
