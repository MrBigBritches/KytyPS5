// Tests the helper that lowers one thread's priority, its lowest step and the boosts that raise
// another thread. A thread that never calls the helper is the baseline in each check, since a
// thread could report a lowered priority for another reason, such as a runner that starts
// everything at a low priority.
#include "common/common.h"
#include "common/threadPriority.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <mutex>
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

// What a thread lowered by the lowest step reports, given what an ordinary one reports. Spelled out
// like LoweredPriority. macOS has no finer step, so there it is the same class.
int LowestPriority(int ordinary) {
#if defined(__APPLE__)
	return LoweredPriority(ordinary);
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	(void)ordinary;
	return THREAD_PRIORITY_LOWEST;
#else
	constexpr int EXPECTED_LOWEST_NICE_STEP = 6;
	constexpr int MAX_NICE                  = 19;
	return std::min(ordinary + EXPECTED_LOWEST_NICE_STEP, MAX_NICE);
#endif
}

// The lowest step lowers the calling thread one step past the ordinary helper's. Both are measured
// from the same baseline thread, so a lowest step that does what the ordinary one does fails.
void TestLowestStepGoesOneFurther() {
	int baseline = -1;
	OnNewThread([&] { baseline = CurrentThreadPriority(); });

	int                     lowered         = -1;
	int                     lowest          = -1;
	int                     lowest_reported = -1;
	bool                    succeeded       = false;
	Common::PriorityFailure failure {"untouched", -7};
	OnNewThread([&] {
		CHECK(Common::LowerCurrentThreadPriority());
		lowered = CurrentThreadPriority();
	});
	OnNewThread([&] {
		succeeded       = Common::LowerCurrentThreadPriorityToLowest(&failure);
		lowest          = CurrentThreadPriority();
		lowest_reported = Common::CurrentThreadSystemPriority();
	});
	CHECK(succeeded);
	CHECK(lowered == LoweredPriority(baseline));
	CHECK(lowest == LowestPriority(baseline));
#if !defined(__APPLE__)
	// Otherwise the two expectations could not tell the steps apart. CI runners start at nice 0.
	CHECK(lowest != lowered);
	// Read on a lowered thread, so a reader that returns the ordinary value regardless fails.
	CHECK(lowest_reported == lowest);
#else
	// macOS reports 0 there, as TestBoostsDoNothingOnMacOS checks.
	(void)lowest_reported;
#endif
	CHECK(std::string_view(failure.api) == "untouched" && failure.code == -7);
	OnNewThread([] { CHECK(!Common::SystemLowestStep().has_value()); });
}

#if defined(__APPLE__)
// macOS has no system thread id to hand over and no way to set another thread's QoS class, so both
// boosts refuse and do nothing.
void TestBoostsDoNothingOnMacOS() {
	CHECK(Common::CurrentThreadSystemId() == 0);
	CHECK(Common::CurrentThreadSystemPriority() == 0);
	CHECK(!Common::BoostThreadToNormal(1));
	CHECK(!Common::BoostThreadToPriority(1, 0));
}
#else
// What a thread raised by BoostThreadToNormal reports.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
constexpr int NORMAL_PRIORITY = THREAD_PRIORITY_NORMAL;
#else
constexpr int NORMAL_PRIORITY = 0;
#endif

// Another thread's priority read from outside it, in the units of CurrentThreadPriority.
int ThreadPriorityOf(uint64_t thread_id) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE thread =
	    OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, static_cast<DWORD>(thread_id));
	CHECK(thread != nullptr);
	const int priority = GetThreadPriority(thread);
	CloseHandle(thread);
	return priority;
#else
	return getpriority(PRIO_PROCESS, static_cast<id_t>(thread_id));
#endif
}

// A thread that lowers itself by the lowest step and then waits, so the test can boost it from
// outside and read it back. Asked to, it lowers itself again, the way a boosted thread drops back
// once the work it was raised for is done.
class LoweredThread {
public:
	LoweredThread(): m_thread([this] { Run(); }) {
		std::unique_lock lock(m_mutex);
		m_changed.wait(lock, [this] { return m_ready; });
	}
	~LoweredThread() {
		{
			const std::lock_guard lock(m_mutex);
			m_quit = true;
		}
		m_changed.notify_all();
		m_thread.join();
	}
	LoweredThread(const LoweredThread&)            = delete;
	LoweredThread& operator=(const LoweredThread&) = delete;

	[[nodiscard]] uint64_t Id() const { return m_id; }
	[[nodiscard]] bool     Lowered() const { return m_lowered; }

	// Has the thread lower itself again and returns whether that succeeded.
	bool LowerAgain() {
		std::unique_lock lock(m_mutex);
		const int        wanted = ++m_lower_requests;
		m_changed.notify_all();
		m_changed.wait(lock, [&] { return m_lowers_done == wanted; });
		return m_lowered;
	}

private:
	void Run() {
		const bool       lowered = Common::LowerCurrentThreadPriorityToLowest();
		std::unique_lock lock(m_mutex);
		m_lowered = lowered;
		m_id      = Common::CurrentThreadSystemId();
		m_ready   = true;
		m_changed.notify_all();
		while (!m_quit) {
			m_changed.wait(lock, [this] { return m_quit || m_lower_requests > m_lowers_done; });
			if (m_lower_requests > m_lowers_done) {
				m_lowered = Common::LowerCurrentThreadPriorityToLowest();
				m_lowers_done++;
				m_changed.notify_all();
			}
		}
	}

	std::mutex              m_mutex;
	std::condition_variable m_changed;
	bool                    m_ready          = false;
	bool                    m_quit           = false;
	bool                    m_lowered        = false;
	uint64_t                m_id             = 0;
	int                     m_lower_requests = 0;
	int                     m_lowers_done    = 0;
	// Declared last, so it starts after the members it uses.
	std::thread m_thread;
};

// Whether a boost the system refused is allowed to skip the rest of a check. Linux refuses an
// unprivileged process a lower nice value, as the helper documents. Windows lets a process raise
// its own threads within its class, so a refusal there fails.
bool RefusedRaiseMaySkip() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return false;
#else
	std::puts("Skipping the rest of a boost check, the system refused to raise the thread");
	return true;
#endif
}

// BoostThreadToPriority raises a lowered thread to the caller's priority, and the thread drops back
// when it lowers itself again. The baseline is the same thread read before the boost. Asking for a
// priority the thread already has, or one below it, changes nothing and returns false, so a boost
// that sets the target without comparing fails, and so does one that raises the caller instead.
void TestBoostToPriorityRaisesAndTheThreadDropsBack() {
	const int caller = Common::CurrentThreadSystemPriority();
	CHECK(caller == CurrentThreadPriority());
	const int lowest = LowestPriority(caller);
	CHECK(lowest != caller);

	LoweredThread worker;
	CHECK(worker.Lowered());
	CHECK(worker.Id() != 0 && worker.Id() != Common::CurrentThreadSystemId());
	CHECK(ThreadPriorityOf(worker.Id()) == lowest);
	CHECK(!Common::BoostThreadToPriority(worker.Id(), lowest));
	CHECK(ThreadPriorityOf(worker.Id()) == lowest);

	if (!Common::BoostThreadToPriority(worker.Id(), caller)) {
		CHECK(ThreadPriorityOf(worker.Id()) == lowest);
		CHECK(RefusedRaiseMaySkip());
		return;
	}
	CHECK(ThreadPriorityOf(worker.Id()) == caller);
	CHECK(!Common::BoostThreadToPriority(worker.Id(), caller));
	CHECK(!Common::BoostThreadToPriority(worker.Id(), lowest));
	CHECK(ThreadPriorityOf(worker.Id()) == caller);

	CHECK(worker.LowerAgain());
	CHECK(ThreadPriorityOf(worker.Id()) == lowest);
}

// BoostThreadToNormal raises a lowered thread to normal priority. The baseline is the same thread
// before the call. Thread id 0 names no thread, and both boosts refuse it.
void TestBoostToNormal() {
	CHECK(!Common::BoostThreadToNormal(0));
	CHECK(!Common::BoostThreadToPriority(0, NORMAL_PRIORITY));

	LoweredThread worker;
	const int     before = ThreadPriorityOf(worker.Id());
	CHECK(before != NORMAL_PRIORITY);
	if (!Common::BoostThreadToNormal(worker.Id())) {
		CHECK(ThreadPriorityOf(worker.Id()) == before);
		CHECK(RefusedRaiseMaySkip());
		return;
	}
	CHECK(ThreadPriorityOf(worker.Id()) == NORMAL_PRIORITY);
}
#endif

} // namespace

int main() {
	TestLowersTheCallingThread();
	TestRefusalIsReported();
	TestLowestStepGoesOneFurther();
#if defined(__APPLE__)
	TestBoostsDoNothingOnMacOS();
#else
	TestBoostToPriorityRaisesAndTheThreadDropsBack();
	TestBoostToNormal();
#endif
	std::printf("ThreadPriorityTests: ok\n");
	return 0;
}
