#ifndef KYTY_COMMON_THREAD_PRIORITY_H_
#define KYTY_COMMON_THREAD_PRIORITY_H_

#include <cstdint>
#include <limits>
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

/// One platform step below SystemLowerStep (experimental, for the graphics prescan's workers, which must
/// lose to the compute prescan's). Lowest on Windows, one nice value more than SystemLowerStep on
/// Linux (at most 19), and the same utility QoS class on macOS, which has no finer step. Elsewhere it
/// refuses with no api, so a caller has nothing to log.
[[nodiscard]] std::optional<PriorityFailure> SystemLowestStep();

/// Lowers the calling thread by SystemLowestStep. The same contract as LowerCurrentThreadPriority.
bool LowerCurrentThreadPriorityToLowest(PriorityFailure* failure = nullptr);

/// What CurrentThreadSystemPriority returns when the system could not read the priority. It is the
/// value of Windows' THREAD_PRIORITY_ERROR_RETURN and lies outside every platform's range, so
/// SetCurrentThreadSystemPriority refuses it rather than set some clamped level.
inline constexpr int UNREAD_THREAD_PRIORITY = std::numeric_limits<int>::max();

/// The calling thread's priority as the system reports it: GetThreadPriority on Windows and the nice
/// value on Linux. UNREAD_THREAD_PRIORITY when the read fails, and 0 elsewhere, where there is nothing
/// to read.
[[nodiscard]] int CurrentThreadSystemPriority();

/// Sets the calling thread back to `priority`, a value CurrentThreadSystemPriority returned on this
/// thread, so a thread that was raised for a while returns to exactly the level it had. SetThreadPriority
/// on Windows and the thread's nice value on Linux. It refuses UNREAD_THREAD_PRIORITY, and elsewhere
/// it does nothing.
/// @return whether the system accepted the change.
bool SetCurrentThreadSystemPriority(int priority);

/// The calling thread's id as the system knows it, which BoostThreadToNormal takes. The thread id on
/// Windows and the kernel thread id on Linux. 0 where neither exists, which BoostThreadToNormal
/// refuses.
[[nodiscard]] uint64_t CurrentThreadSystemId();

/// Raises another thread to normal priority, so a thread that waits on its work is not held up by its
/// lowered priority. Normal priority on Windows. On Linux it is the process's own nice value, read once,
/// and only a thread below it is raised, so a boost never lowers a thread that already runs above it.
/// An unprivileged process may be refused there. Elsewhere it does nothing and returns false. The
/// thread drops back itself once the waited-on work is done, with SetCurrentThreadSystemPriority.
/// @return whether the system raised the thread.
bool BoostThreadToNormal(uint64_t thread_id);

/// Raises another thread to `priority`, a value CurrentThreadSystemPriority returned, and only if it
/// runs below that now, so a thread is never raised past the caller whose priority was read (experimental).
/// GetThreadPriority and SetThreadPriority on Windows, and the nice value on Linux, where lower is
/// higher. Elsewhere it does nothing.
/// @return whether the thread was raised. False when it already ran at `priority` or above.
bool BoostThreadToPriority(uint64_t thread_id, int priority);

} // namespace Common

#endif /* KYTY_COMMON_THREAD_PRIORITY_H_ */
