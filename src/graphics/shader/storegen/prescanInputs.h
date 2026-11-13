#ifndef GRAPHICS_SHADER_STOREGEN_PRESCAN_INPUTS_H
#define GRAPHICS_SHADER_STOREGEN_PRESCAN_INPUTS_H

#include "graphics/guest_gpu/hardwareContext.h"

#include <array>
#include <cstdint>

// The dispatch inputs the pre-scan records, kept apart from computeTranslation.h so that code which
// only carries them, such as the Stream worker's frames, does not pull in the recompiler headers.
namespace Libs::Graphics::StoreGen {

/// What the runtime takes from one compute dispatch besides the shader map, as a submit scan saw
/// it. PrescanInputsOf in the command processor's prescanDispatchInputs.h fills it from a
/// ScannedDispatch. The caller fills every field.
struct PrescanDispatchInputs {
	/// The compute registers at the dispatch. Their wave size must be the dispatch initiator's, as
	/// the command processor writes it into the registers before it dispatches.
	HW::CsStageRegisters cs_regs;
	/// Whether the dispatch is on an async compute queue.
	bool async_compute;
	/// Whether the initiator asks for thread dimensions rather than group counts.
	bool thread_dimensions;
	/// Whether the dispatch came from an indirect packet.
	bool indirect;
	/// The group counts of a direct dispatch, already rounded up from threads as the runtime does.
	std::array<uint32_t, 3> workgroup_counts;
	/// The address of an indirect dispatch's arguments, 0 for a direct one.
	uint64_t indirect_args_address;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_PRESCAN_INPUTS_H
