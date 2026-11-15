#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"

#include "common/config.h"
#include "common/profiler.h"
#include "common/threadPriority.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <thread>
#include <utility>

namespace Libs::Graphics {

namespace {
// The hint is only worth reporting once per process, however many workers it fails on.
std::atomic_flag g_priority_failure_logged;

// Reports a refused priority hint the first time it happens.
void LogPriorityFailure(const char* api, long code) {
	if (!g_priority_failure_logged.test_and_set()) {
		std::printf("PipelineBuildQueue: %s failed with %ld, workers keep normal priority\n", api, code);
	}
}

// Lowers the priority of the calling thread through the shared helper, to the lowest step when asked.
// A refusal is logged once and the caller carries on at its old priority.
bool LowerCurrentThreadPriority(bool lowest) {
	Common::PriorityFailure failure;
	if (lowest ? Common::LowerCurrentThreadPriorityToLowest(&failure) : Common::LowerCurrentThreadPriority(&failure)) {
		return true;
	}
	// A platform without priority support names no call, and nothing was logged for it before.
	if (failure.api != nullptr) {
		LogPriorityFailure(failure.api, failure.code);
	}
	return false;
}
} // namespace

uint32_t PipelineBuildWorkerCountFor(uint32_t hardware_threads, uint32_t worker_cap) {
	return std::clamp(std::max(1u, hardware_threads) / 2u, 1u, std::max(1u, worker_cap));
}

uint32_t PipelineBuildWorkerCount(uint32_t worker_cap) {
	return PipelineBuildWorkerCountFor(std::thread::hardware_concurrency(), worker_cap);
}

PipelineBuildQueue::PipelineBuildQueue(uint32_t worker_count, PipelineBuildQueueOptions options)
    : m_low_priority(options.low_priority), m_lowest_priority(options.low_priority && options.lowest_priority) {
	m_workers.reserve(worker_count);
	for (uint32_t index = 0; index < worker_count; index++) {
		m_workers.emplace_back([this](std::stop_token stop) { WorkerLoop(stop); });
	}
}

PipelineBuildQueue::~PipelineBuildQueue() {
	Stop();
}

PipelineBuildQueue::JobPtr PipelineBuildQueue::Post(Common::UniqueFunction<void>&& work,
                                                    Lane                           lane) {
	JobPtr job;
	{
		std::scoped_lock lock(m_mutex);
		// Checked before the work is moved, so a refused caller still owns it.
		if (m_stopping.load(std::memory_order_relaxed)) {
			return nullptr;
		}
		job = std::make_shared<Job>(std::move(work));
		(lane == Lane::Back ? m_back_jobs : m_jobs).push_back(job);
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
		LowerCurrentThreadPriority(m_lowest_priority);
	}
	std::unique_lock lock(m_mutex);
	while (!stop.stop_requested()) {
		if (!HasJobsLocked()) {
			// Woken by a post to either lane or by the stop.
			m_ready.wait(lock, stop, [this] { return HasJobsLocked(); });
		} else if (PausedLocked()) {
			// A job waits and the queue is paused. Resume, the deadline and the stop all end the wait, and
			// so does a pause that moved the deadline, since the wait runs to a copy of the old one.
			const auto deadline = m_pause_deadline;
			m_ready.wait_until(lock, stop, deadline, [this, deadline] {
				return !PausedLocked() || m_pause_deadline != deadline;
			});
		} else {
			const auto job = PopNextLocked();
			m_active++;
			lock.unlock();
			RunIfQueued(*job);
			lock.lock();
			m_active--;
		}
	}
}

bool PipelineBuildQueue::HasJobsLocked() const {
	return !m_jobs.empty() || !m_back_jobs.empty();
}

PipelineBuildQueue::JobPtr PipelineBuildQueue::PopNextLocked() {
	// The one place that decides which job runs next. A later admit check for Back jobs, and a cap
	// on how many of them run at once, belongs here. The caller tests the pause for both lanes.
	// Such a check must also change the workers' wait to "a job that may run now", since Back can
	// then hold jobs that none may take, and a wait on HasJobsLocked would spin.
	auto& lane = m_jobs.empty() ? m_back_jobs : m_jobs;
	auto  job  = std::move(lane.front());
	lane.pop_front();
	return job;
}

bool PipelineBuildQueue::PausedLocked() const {
	return m_paused && Clock::now() < m_pause_deadline;
}

void PipelineBuildQueue::Pause(Clock::time_point deadline) {
	{
		std::scoped_lock lock(m_mutex);
		deadline         = std::min(deadline, Clock::now() + MAX_PAUSE);
		m_pause_deadline = PausedLocked() ? std::min(m_pause_deadline, deadline) : deadline;
		m_paused         = true;
	}
	// Workers already waiting on a later deadline must wait on this one.
	m_ready.notify_all();
}

void PipelineBuildQueue::Resume() {
	{
		std::scoped_lock lock(m_mutex);
		m_paused = false;
	}
	m_ready.notify_all();
}

bool PipelineBuildQueue::Paused() const {
	std::scoped_lock lock(m_mutex);
	return PausedLocked();
}

uint32_t PipelineBuildQueue::Stop() {
	uint32_t waited_for = 0;
	{
		std::scoped_lock lock(m_mutex);
		if (m_stopping.exchange(true, std::memory_order_acq_rel)) {
			return 0;
		}
		m_jobs.clear();
		m_back_jobs.clear();
		waited_for = m_active;
	}
	for (auto& worker: m_workers) {
		worker.request_stop();
	}
	m_workers.clear();
	return waited_for;
}

} // namespace Libs::Graphics
