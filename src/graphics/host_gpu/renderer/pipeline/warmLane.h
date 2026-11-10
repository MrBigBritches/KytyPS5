#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WARMLANE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WARMLANE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"
#include "graphics/shader/storegen/generatedOrder.h"

namespace Libs::Graphics {

/// The warm queue lane a generated module's build is posted on. Only the wide tail waits in the
/// Back lane, so it builds last and everything else keeps the order it was posted in. Generated
/// modules exist only when the generator runs, so the main store's warm-up never comes here. Kept
/// apart from the cache so that a test can check the rule without a device.
/// @param order what the supervisor said of the module.
[[nodiscard]] constexpr PipelineBuildQueue::Lane WarmLaneOf(const StoreGen::GeneratedOrder& order) {
	return order.wide_tail ? PipelineBuildQueue::Lane::Back : PipelineBuildQueue::Lane::Front;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WARMLANE_H_
