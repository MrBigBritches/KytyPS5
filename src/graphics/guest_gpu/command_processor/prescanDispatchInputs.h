#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PRESCAN_DISPATCH_INPUTS_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PRESCAN_DISPATCH_INPUTS_H

#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/shader/storegen/prescanInputs.h"

// The adapter from what the submit scanner records to what the pre-scan's input info is built from.
// It lives on the command processor's side so the shader store code needs no scanner header.
namespace Libs::Graphics {

/// The dispatch inputs of a scanned dispatch. The scanner already set the registers' wave size from
/// the initiator and rounded thread counts up to groups, as the command processor does.
[[nodiscard]] inline StoreGen::PrescanDispatchInputs PrescanInputsOf(const ScannedDispatch& dispatch) {
	return {.cs_regs               = dispatch.regs.cs_regs,
	        .async_compute         = dispatch.async_queue,
	        .thread_dimensions     = dispatch.thread_dimensions,
	        .indirect              = dispatch.indirect,
	        .workgroup_counts      = dispatch.workgroup_counts,
	        .indirect_args_address = dispatch.indirect_args_address};
}

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_PRESCAN_DISPATCH_INPUTS_H
