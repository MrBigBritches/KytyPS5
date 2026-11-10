#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WARMLANE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WARMLANE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"
#include "graphics/shader/storegen/generatedOrder.h"

namespace Libs::Graphics {

/// The warm queue lane a generated module's build is posted on. Only the wide tail waits in the
/// Back lane, and only when --warm-wide-tail-last is on, so everything else keeps the order it was
/// posted in. Kept apart from the cache so that a test can check the rule without a device.
/// @param wide_tail_last whether --warm-wide-tail-last is on.
/// @param order what the supervisor said of the module.
[[nodiscard]] constexpr PipelineBuildQueue::Lane WarmLaneOf(bool wide_tail_last,
                                                            const StoreGen::GeneratedOrder& order) {
	return wide_tail_last && order.wide_tail ? PipelineBuildQueue::Lane::Back
	                                         : PipelineBuildQueue::Lane::Front;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_WARMLANE_H_
