#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"

#include "common/config.h"
#include "common/profiler.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <thread>
#include <utility>

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

namespace Libs::Graphics {

namespace {
// How much nicer than the thread's current value the build threads become, so they lose to the
// guest threads without starving.
[[maybe_unused]] constexpr int LOW_PRIORITY_NICE = 5;
// The largest nice value Linux accepts.
[[maybe_unused]] constexpr int MAX_NICE = 19;

// The hint is only worth reporting once per process, however many workers it fails on.
std::atomic_flag g_priority_failure_logged;

// Reports a refused priority hint the first time it happens.
void LogPriorityFailure(const char* api, long code) {
	if (!g_priority_failure_logged.test_and_set()) {
		std::printf("PipelineBuildQueue: %s failed with %ld, workers keep normal priority\n", api, code);
	}
}

// Lowers the priority of the calling thread only, to below normal on Windows, by a few nice
// steps on Linux and to the utility QoS class on macOS. Returns false when the system refused, and
// the caller carries on at its old priority.
bool LowerCurrentThreadPriority() {
#if defined(__APPLE__)
	const int error = pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
	if (error != 0) {
		LogPriorityFailure("pthread_set_qos_class_self_np", error);
		return false;
	}
	return true;
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) == 0) {
		LogPriorityFailure("SetThreadPriority", static_cast<long>(GetLastError()));
		return false;
	}
	return true;
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// On Linux the nice value belongs to the thread id, so this leaves the rest of the process alone.
	// The value is absolute, so it is built from the current one and a process that already runs
	// niced is not asked for something it may not do.
	const auto thread_id = static_cast<id_t>(syscall(SYS_gettid));
	// -1 is a valid nice value, so only errno says whether the read failed.
	errno                = 0;
	const int current    = getpriority(PRIO_PROCESS, thread_id);
	if (current == -1 && errno != 0) {
		LogPriorityFailure("getpriority", errno);
		return false;
	}
	if (setpriority(PRIO_PROCESS, thread_id, std::min(current + LOW_PRIORITY_NICE, MAX_NICE)) != 0) {
		LogPriorityFailure("setpriority", errno);
		return false;
	}
	return true;
#else
	return false;
#endif
}
} // namespace

uint32_t PipelineBuildWorkerCountFor(uint32_t hardware_threads) {
	return std::clamp(std::max(1u, hardware_threads) / 2u, 1u, PIPELINE_BUILD_WORKERS);
}

uint32_t PipelineBuildWorkerCount() {
	return PipelineBuildWorkerCountFor(std::thread::hardware_concurrency());
}

PipelineBuildQueue::PipelineBuildQueue(uint32_t worker_count, PipelineBuildQueueOptions options)
    : m_low_priority(options.low_priority) {
	m_workers.reserve(worker_count);
	for (uint32_t index = 0; index < worker_count; index++) {
		m_workers.emplace_back([this](std::stop_token stop) { WorkerLoop(stop); });
	}
}

PipelineBuildQueue::~PipelineBuildQueue() {
	Stop();
}

PipelineBuildQueue::JobPtr PipelineBuildQueue::Post(Common::UniqueFunction<void>&& work) {
	JobPtr job;
	{
		std::scoped_lock lock(m_mutex);
		// Checked before the work is moved, so a refused caller still owns it.
		if (m_stopping.load(std::memory_order_relaxed)) {
			return nullptr;
		}
		job = std::make_shared<Job>(std::move(work));
		m_jobs.push_back(job);
	}
	m_ready.notify_one();
	return job;
}

bool PipelineBuildQueue::RunIfQueued(Job& job) {
	// Only the thread that wins this exchange runs the work, so a worker and a caller of Acquire
	// can never both run it.
	auto expected = State::Queued;
	if (!job.m_state.compare_exchange_strong(expected, State::Running, std::memory_order_acq_rel,
	                                         std::memory_order_acquire)) {
		return false;
	}
	RunClaimed(job);
	return true;
}

void PipelineBuildQueue::RunClaimed(Job& job) {
	job.m_work();
	// Frees what the work captured, which a job nobody drops would otherwise keep.
	job.m_work = Common::UniqueFunction<void>();
	job.m_state.store(State::Done, std::memory_order_release);
	job.m_finished.count_down();
}

PipelineBuildQueue::Acquired PipelineBuildQueue::Acquire(Job& job) {
	if (RunIfQueued(job)) {
		return Acquired::RanInline;
	}
	if (job.GetState() == State::Done) {
		return Acquired::AlreadyDone;
	}
	job.m_finished.wait();
	return Acquired::Waited;
}

void PipelineBuildQueue::WorkerLoop(std::stop_token stop) {
	KYTY_PROFILER_THREAD("PipelineBuild");
	// A refused hint is logged inside and must not stop the worker.
	if (m_low_priority) {
		LowerCurrentThreadPriority();
	}
	std::unique_lock lock(m_mutex);
	// Once stop is requested the wait returns false only when the queue is empty, and Stop empties
	// it first.
	while (m_ready.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
		const auto job = std::move(m_jobs.front());
		m_jobs.pop_front();
		m_active++;
		lock.unlock();
		RunIfQueued(*job);
		lock.lock();
		m_active--;
	}
}

uint32_t PipelineBuildQueue::Stop() {
	uint32_t waited_for = 0;
	{
		std::scoped_lock lock(m_mutex);
		if (m_stopping.exchange(true, std::memory_order_acq_rel)) {
			return 0;
		}
		m_jobs.clear();
		waited_for = m_active;
	}
	for (auto& worker: m_workers) {
		worker.request_stop();
	}
	m_workers.clear();
	return waited_for;
}

} // namespace Libs::Graphics
