#ifndef GRAPHICS_SHADER_STOREGEN_COMPUTE_STATIC_KEY_H
#define GRAPHICS_SHADER_STOREGEN_COMPUTE_STATIC_KEY_H

#include "graphics/shader/shader.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace Libs::Graphics::StoreGen {

/// The host values that the static key depends on and that only the device knows.
/// The caller fills every field.
struct ComputeKeyDevice {
	/// What the runtime sets from SupportsComputeWave64() in GetComputeProgram, 64 or 32.
	uint32_t host_subgroup_size;
	/// The device's maxComputeSharedMemorySize limit, which decides lds_storage.
	uint32_t max_compute_shared_memory_bytes;
};

/// The key fields that depend on how a dispatch is issued, not on the shader header.
/// The runtime takes them from the dispatch packet and the queue, in renderCompute.cpp and
/// graphicsRun.cpp. The caller fills every field.
struct ComputeDispatchInputs {
	/// The wave size, 64 or 32. The runtime takes it from the dispatch initiator.
	uint32_t wave_size;
	/// Whether the dispatch uses thread dimensions rather than group counts.
	bool dispatch_thread_dimensions;
	/// Whether the dispatch is on an async compute queue.
	bool async_compute;
};

/// The dispatch inputs assumed for every Motor compute program. The header does not record them.
constexpr ComputeDispatchInputs MOTOR_DISPATCH {64, false, false};

/// The static inputs of one compute program.
struct ComputeStaticKey {
	/// The static part of the runtime's input info, which is what its key is built from. The
	/// per-dispatch group counts and indirect address are not filled.
	ShaderComputeInputInfo input_info;
	/// The program's user SGPR count, which is rsrc2.user_sgpr and sizes the user data.
	uint32_t user_data_count = 0;
	/// The words BuildStageStaticKey makes from input_info.
	std::vector<uint32_t> state;
};

/// Builds the static key of a compute program from its shader header alone.
/// The SH registers in the header go through the same decoder as the PM4 handler uses, and the
/// input info through the same function the runtime uses, so the key is what the runtime would
/// build for a game that sends those registers.
/// @param header   the .shader_header section.
/// @param device   the host values the key depends on.
/// @param dispatch the dispatch-dependent key fields, such as MOTOR_DISPATCH.
/// @return nothing when the header is too short, its register table is missing or out of range,
///         or it has no COMPUTE_PGM_RSRC1 or COMPUTE_PGM_RSRC2 entry.
std::optional<ComputeStaticKey> BuildComputeStaticKey(std::span<const uint8_t>     header,
                                                      const ComputeKeyDevice&      device,
                                                      const ComputeDispatchInputs& dispatch);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_COMPUTE_STATIC_KEY_H
