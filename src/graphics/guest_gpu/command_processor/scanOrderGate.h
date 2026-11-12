#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_ORDER_GATE_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_ORDER_GATE_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <utility>

// Keeps the order a queue's submissions are scanned in the same as the order the command processor
// runs them. A scan and a push of the same list are two steps, and two threads submitting to one queue
// could otherwise interleave them, so that the scan sees the lists in one order and the processor in
// another and the register shadow is left wrong. Production code and the tests both go through it, so
// the ordering a test proves is the ordering the emulator uses.
namespace Libs::Graphics {

/// How many queues have a lock, a sequence number and a watermark of their own. A queue id past it shares
/// the last, and GuestGpu checks that it has no more.
inline constexpr size_t SCAN_MAX_QUEUES = 64;

class ScanOrderGate {
public:
	/// Runs both steps of a submission to `queue` while holding the queue's lock. `scan` runs first and
	/// returns the sequence number the scan gave the submission, 0 for none, and `push` runs second and
	/// receives it, to put the submission on the command processor's queue. A different queue's
	/// submissions are not held up. The lock is taken before any lock `push` takes, so `push` may take
	/// the command processor's queue lock, and nothing under either may wait on this gate.
	/// @return the sequence number `scan` returned.
	template <typename ScanStep, typename PushStep>
	uint64_t Run(uint32_t queue, ScanStep&& scan, PushStep&& push) {
		std::scoped_lock lock(m_locks[std::min<size_t>(queue, SCAN_MAX_QUEUES - 1)]);
		const uint64_t   seq = std::forward<ScanStep>(scan)();
		std::forward<PushStep>(push)(seq);
		return seq;
	}

private:
	std::array<std::mutex, SCAN_MAX_QUEUES> m_locks;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_ORDER_GATE_H
