#ifndef GRAPHICS_GUEST_GPU_GUEST_GPU_OPTIONS_H
#define GRAPHICS_GUEST_GPU_GUEST_GPU_OPTIONS_H

#include "graphics/guest_gpu/command_processor/prescanScanner.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/shader/storegen/prescanTransport.h"

#include <chrono>

// What a GuestGpu is made with. It has a header of its own so that graphicsRun.h and renderContext.h
// can name it without pulling in the prescan's transport and scanner, which only the files that make a
// GuestGpu need.
namespace Libs::Graphics {

/// How the GuestGpu runs. Every field defaults to what the emulator does without any option, so a
/// default-made value changes nothing.
struct GuestGpuOptions {
	/// Runs the submit-time compute prescan. It reads each command list as the game submits it and
	/// builds the compute shaders it is about to run. Off by default.
	bool prescan_compute_builds = false;
	/// Makes the prescan's transport. Empty means the Stream worker in a child process, which is what a
	/// real run uses. Tests pass the in-process one.
	StoreGen::PrescanTransportFactory prescan_transport;
	/// The reader the scanner reads indirect buffers with, which must never fault. The scanner wraps it
	/// in ChunkedScanRead itself, so this is the inner reader. Empty means
	/// LibKernel::Memory::TryReadBacking. Tests pass one that reads host memory or blocks.
	ScanMemoryReader prescan_reader;
	/// How old a command list may be when the scanner reaches it and still be walked. Tests lengthen it,
	/// since a scanner thread below normal priority can be starved on a loaded machine and would then
	/// skip the lists the test is about.
	std::chrono::microseconds prescan_lag_bound = PRESCAN_LAG_BOUND;
	/// Leaves the scanner made but not started, so a test can let the GPU thread begin a list before
	/// the scanner reaches it. GuestGpuTestAccess starts it.
	bool defer_scanner_start_for_test = false;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_GUEST_GPU_OPTIONS_H
