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

} // namespace Common
