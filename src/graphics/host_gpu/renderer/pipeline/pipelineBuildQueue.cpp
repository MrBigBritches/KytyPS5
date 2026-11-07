#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"

#include "common/profiler.h"

#include <algorithm>
#include <thread>
#include <utility>

namespace Libs::Graphics {

uint32_t PipelineBuildWorkerCountFor(uint32_t hardware_threads) {
	return std::clamp(std::max(1u, hardware_threads) / 2u, 1u, PIPELINE_BUILD_WORKERS);
}

uint32_t PipelineBuildWorkerCount() {
	return PipelineBuildWorkerCountFor(std::thread::hardware_concurrency());
}

PipelineBuildQueue::PipelineBuildQueue(uint32_t worker_count) {
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
