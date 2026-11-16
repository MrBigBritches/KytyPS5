#include "common/threadPriority.h"

#include "common/common.h"

#include <algorithm>
#include <cerrno>

#if defined(__APPLE__)
#include <pthread/qos.h>
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace Common {

namespace {
// How much nicer than the thread's current value the thread becomes, so it loses to the guest
// threads without starving.
[[maybe_unused]] constexpr int LOW_PRIORITY_NICE = 5;
// The largest nice value Linux accepts.
[[maybe_unused]] constexpr int MAX_NICE = 19;

#if !defined(__APPLE__) && KYTY_PLATFORM == KYTY_PLATFORM_LINUX
// The nice value the process runs at, which is normal priority for its threads. A process started
// under nice runs every thread there, so nice 0 would be a raise it may not be allowed, and above
// its own threads. It is read once, on whichever thread boosts first, and it is the main thread's
// nice value, since PRIO_PROCESS with the process id names the thread whose id that is. 0 if even
// that read fails.
int ProcessBaseNice() {
	static const int base = [] {
		errno          = 0;
		const int nice = getpriority(PRIO_PROCESS, static_cast<id_t>(getpid()));
		return (nice == -1 && errno != 0) ? 0 : nice;
	}();
	return base;
}
#endif
} // namespace

std::optional<PriorityFailure> SystemLowerStep() {
#if defined(__APPLE__)
	const int error = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
	if (error != 0) {
		return PriorityFailure {"pthread_set_qos_class_self_np", error};
	}
	return std::nullopt;
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) == 0) {
		return PriorityFailure {"SetThreadPriority", static_cast<long>(GetLastError())};
	}
	return std::nullopt;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// On Linux the nice value belongs to the thread id, so this leaves the rest of the process
	// alone. The value is absolute, so it is built from the current one and a process that already
	// runs niced is not asked for something it may not do.
	const auto thread_id = static_cast<id_t>(syscall(SYS_gettid));
	// -1 is a valid nice value, so only errno says whether the read failed.
	errno             = 0;
	const int current = getpriority(PRIO_PROCESS, thread_id);
	if (current == -1 && errno != 0) {
		return PriorityFailure {"getpriority", errno};
	}
	if (setpriority(PRIO_PROCESS, thread_id, std::min(current + LOW_PRIORITY_NICE, MAX_NICE)) !=
	    0) {
		return PriorityFailure {"setpriority", errno};
	}
	return std::nullopt;
#else
	// No way to lower a thread here. The refusal carries no api, so a caller has nothing to log.
	return PriorityFailure {};
#endif
}

std::optional<PriorityFailure> SystemLowestStep() {
#if defined(__APPLE__)
	return SystemLowerStep();
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) == 0) {
		return PriorityFailure {"SetThreadPriority", static_cast<long>(GetLastError())};
	}
	return std::nullopt;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// Built from the current value like SystemLowerStep, one nice value further, so a thread lowered
	// this way loses to one lowered by SystemLowerStep from the same start.
	const auto thread_id = static_cast<id_t>(syscall(SYS_gettid));
	errno                = 0;
	const int current    = getpriority(PRIO_PROCESS, thread_id);
	if (current == -1 && errno != 0) {
		return PriorityFailure {"getpriority", errno};
	}
	if (setpriority(PRIO_PROCESS, thread_id, std::min(current + LOW_PRIORITY_NICE + 1, MAX_NICE)) != 0) {
		return PriorityFailure {"setpriority", errno};
	}
	return std::nullopt;
#else
	return PriorityFailure {};
#endif
}

bool LowerCurrentThreadPriorityToLowest(PriorityFailure* failure) {
	return LowerCurrentThreadPriorityWith(&SystemLowestStep, failure);
}

int CurrentThreadSystemPriority() {
#if defined(__APPLE__)
	return 0;
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// A failed read already returns the sentinel, since the two are the same value.
	static_assert(THREAD_PRIORITY_ERROR_RETURN == UNREAD_THREAD_PRIORITY);
	return GetThreadPriority(GetCurrentThread());
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// -1 is a valid nice value, so only errno says whether the read failed.
	errno             = 0;
	const int  nice   = getpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)));
	const bool failed = nice == -1 && errno != 0;
	return failed ? UNREAD_THREAD_PRIORITY : nice;
#else
	return 0;
#endif
}

// The sentinel is refused before the system sees it, since Linux would clamp it to nice 19 and
// accept, which would leave the thread far lower than any level it ever had.
bool SetCurrentThreadSystemPriority(int priority) {
	if (priority == UNREAD_THREAD_PRIORITY) {
		return false;
	}
#if defined(__APPLE__)
	return false;
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return SetThreadPriority(GetCurrentThread(), priority) != 0;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// The nice value belongs to the thread id, so this leaves the rest of the process alone.
	return setpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)), priority) == 0;
#else
	return false;
#endif
}

bool LowerCurrentThreadPriorityWith(PriorityStep step, PriorityFailure* failure) {
	const auto refused = step();
	if (!refused) {
		return true;
	}
	if (failure != nullptr) {
		*failure = *refused;
	}
	return false;
}

bool LowerCurrentThreadPriority(PriorityFailure* failure) {
	return LowerCurrentThreadPriorityWith(&SystemLowerStep, failure);
}

uint64_t CurrentThreadSystemId() {
#if defined(__APPLE__)
	return 0;
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return GetCurrentThreadId();
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	return static_cast<uint64_t>(syscall(SYS_gettid));
#else
	return 0;
#endif
}

// macOS has no call that sets another thread's QoS class, so the boost is a no-op there. On Linux it
// goes through BoostThreadToPriority, which compares before it sets, so a thread already above the
// process's nice value is never pulled down to it.
bool BoostThreadToNormal(uint64_t thread_id) {
	if (thread_id == 0) {
		return false;
	}
#if defined(__APPLE__)
	return false;
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE thread = OpenThread(THREAD_SET_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(thread_id));
	if (thread == nullptr) {
		return false;
	}
	const bool raised = SetThreadPriority(thread, THREAD_PRIORITY_NORMAL) != 0;
	CloseHandle(thread);
	return raised;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	return BoostThreadToPriority(thread_id, ProcessBaseNice());
#else
	return false;
#endif
}

bool BoostThreadToPriority(uint64_t thread_id, int priority) {
	if (thread_id == 0) {
		return false;
	}
#if defined(__APPLE__)
	(void)priority;
	return false;
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE thread = OpenThread(THREAD_SET_LIMITED_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION, FALSE,
	                           static_cast<DWORD>(thread_id));
	if (thread == nullptr) {
		return false;
	}
	const int  current = GetThreadPriority(thread);
	const bool raised  = current != THREAD_PRIORITY_ERROR_RETURN && current < priority &&
	                    SetThreadPriority(thread, priority) != 0;
	CloseHandle(thread);
	return raised;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	errno             = 0;
	const int current = getpriority(PRIO_PROCESS, static_cast<id_t>(thread_id));
	if ((current == -1 && errno != 0) || current <= priority) {
		return false;
	}
	return setpriority(PRIO_PROCESS, static_cast<id_t>(thread_id), priority) == 0;
#else
	(void)priority;
	return false;
#endif
}

} // namespace Common
