// Tests the helper that lowers one thread's priority. A thread that never calls the helper is the
// baseline in each check, since a thread could report a lowered priority for another reason, such
// as a runner that starts everything at a low priority.
#include "common/common.h"
#include "common/threadPriority.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string_view>
#include <thread>

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

namespace {

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "ThreadPriorityTests:%d: %s\n", __LINE__, #condition);            \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

// The priority of the calling thread in the units of this platform. It is the Windows thread
// priority, the Linux nice value of the thread or the macOS QoS class.
int CurrentThreadPriority() {
#if defined(__APPLE__)
	return static_cast<int>(qos_class_self());
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return GetThreadPriority(GetCurrentThread());
#else
	return getpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)));
#endif
}

// What a lowered thread reports, given what an ordinary one reports. These are spelled out here and
// do not use the helper's own constants, so a wrong level in the helper is caught.
int LoweredPriority(int ordinary) {
#if defined(__APPLE__)
	(void)ordinary;
	return static_cast<int>(QOS_CLASS_UTILITY);
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	(void)ordinary;
	return THREAD_PRIORITY_BELOW_NORMAL;
#else
	constexpr int EXPECTED_NICE_STEP = 5;
	constexpr int MAX_NICE           = 19;
	return std::min(ordinary + EXPECTED_NICE_STEP, MAX_NICE);
#endif
}

// Runs work on a new thread and waits for it, so a lowered priority never reaches the test's own.
void OnNewThread(const std::function<void()>& work) {
	std::thread thread(work);
	thread.join();
}

// The helper lowers the thread that calls it, to the platform's level, and no other thread. The
// baseline thread does not call it, and a thread started afterwards by the unchanged caller still
// reports the ordinary value.
void TestLowersTheCallingThread() {
	const int caller_before = CurrentThreadPriority();
	int       baseline      = -1;
	OnNewThread([&] { baseline = CurrentThreadPriority(); });
#if !defined(__APPLE__)
	// A new thread starts at its creator's level. On macOS a new pthread may report another QoS
	// class than the main thread, so there threads are only compared with other new threads.
	CHECK(baseline == caller_before);
#endif
	// A runner already at the floor could not tell a lowered thread from an ordinary one. CI
	// runners start at nice 0.
	CHECK(LoweredPriority(baseline) != baseline);

	int                     before_call = -1;
	int                     after_call  = -1;
	bool                    succeeded   = false;
	Common::PriorityFailure failure {"untouched", -7};
	OnNewThread([&] {
		before_call = CurrentThreadPriority();
		succeeded   = Common::LowerCurrentThreadPriority(&failure);
		after_call  = CurrentThreadPriority();
	});
	CHECK(succeeded);
	CHECK(before_call == baseline);
	CHECK(after_call == LoweredPriority(baseline));
	// A success leaves the failure alone, so an old value cannot be mistaken for a new refusal.
	CHECK(std::string_view(failure.api) == "untouched" && failure.code == -7);

	int after_helper_thread = -1;
	OnNewThread([&] { after_helper_thread = CurrentThreadPriority(); });
	CHECK(after_helper_thread == baseline);
	CHECK(CurrentThreadPriority() == caller_before);

	// The failure pointer is optional.
	OnNewThread([&] { CHECK(Common::LowerCurrentThreadPriority()); });
}

std::optional<Common::PriorityFailure> RefusingStep() {
	return Common::PriorityFailure {"injected_call", 4242};
}

std::optional<Common::PriorityFailure> AcceptingStep() {
	return std::nullopt;
}

std::optional<Common::PriorityFailure> RefusingStepWithoutApi() {
	return Common::PriorityFailure {};
}

// A refusing step is reported with its call and code, and the thread keeps its priority. The
// accepting step is the baseline that differs only in the answer, so a helper that always says
// "refused" or always says "lowered" fails one of the two.
void TestRefusalIsReported() {
	OnNewThread([] {
		const int before = CurrentThreadPriority();

		Common::PriorityFailure refused;
		CHECK(!Common::LowerCurrentThreadPriorityWith(&RefusingStep, &refused));
		CHECK(refused.api != nullptr && std::string_view(refused.api) == "injected_call");
		CHECK(refused.code == 4242);

		Common::PriorityFailure accepted {"untouched", 1};
		CHECK(Common::LowerCurrentThreadPriorityWith(&AcceptingStep, &accepted));
		CHECK(std::string_view(accepted.api) == "untouched" && accepted.code == 1);

		// A refusal with no failure pointer, and one that names no call, are still reported.
		CHECK(!Common::LowerCurrentThreadPriorityWith(&RefusingStep));
		Common::PriorityFailure nameless {"untouched", 1};
		CHECK(!Common::LowerCurrentThreadPriorityWith(&RefusingStepWithoutApi, &nameless));
		CHECK(nameless.api == nullptr);

		// The injected steps never touch the system, so the priority did not move.
		CHECK(CurrentThreadPriority() == before);
	});
}

} // namespace

int main() {
	TestLowersTheCallingThread();
	TestRefusalIsReported();
	std::printf("ThreadPriorityTests: ok\n");
	return 0;
}
