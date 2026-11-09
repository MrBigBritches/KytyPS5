#ifndef KYTY_COMMON_THREAD_PRIORITY_H_
#define KYTY_COMMON_THREAD_PRIORITY_H_

#include <optional>

// Lowers the priority of one thread, for background work that must lose to the guest's threads
// without starving. It lives apart from threads.h, which holds the mutex and condition variable
// wrappers, so that the operating system headers stay in one file.
namespace Common {

/// What a refused priority change reports, for a caller that logs it.
struct PriorityFailure {
	/// The system call that failed. Null when the platform has no priority support at all.
	const char* api = nullptr;
	/// The error the call reported (errno, GetLastError or the QoS error).
	long code = 0;
};

/// One platform step of the change. It returns nothing when the system accepted the change.
/// Production uses SystemLowerStep. A test passes its own to make the system refuse.
using PriorityStep = std::optional<PriorityFailure> (*)();

/// The platform's way of lowering the calling thread. Below normal on Windows, a nice value of 5
/// above the thread's current one (at most 19) on Linux, and the utility QoS class on macOS.
[[nodiscard]] std::optional<PriorityFailure> SystemLowerStep();

/// Lowers the priority of the calling thread only, never the process. Returns false when the system
/// refused, and the thread then keeps its old priority. A caller that wants to log a refusal passes
/// `failure`, which is filled only on a refusal.
bool LowerCurrentThreadPriority(PriorityFailure* failure = nullptr);

/// LowerCurrentThreadPriority with the platform step replaced, so a test can inject a refusal.
bool LowerCurrentThreadPriorityWith(PriorityStep step, PriorityFailure* failure = nullptr);

} // namespace Common

#endif /* KYTY_COMMON_THREAD_PRIORITY_H_ */
