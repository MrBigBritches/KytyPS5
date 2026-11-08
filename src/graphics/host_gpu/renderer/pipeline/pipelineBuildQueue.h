#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINEBUILDQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINEBUILDQUEUE_H_

#include "common/common.h"
#include "common/uniqueFunction.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <latch>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace Libs::Graphics {
/// The most threads that build pipelines in the background, a few so that a burst of misses cannot
/// take every core from the game. A small machine starts fewer.
inline constexpr uint32_t PIPELINE_BUILD_WORKERS = 4;

/// The most build workers while --deferred-pipeline-draws is on. Its draws wait for their pipelines
/// together, so a burst of misses finishes sooner with more of them building at once.
inline constexpr uint32_t DEFERRED_PIPELINE_BUILD_WORKERS = 6;

/// How many build workers to start on this machine. At most `worker_cap`, and at most half the
/// hardware threads, so a small machine keeps room for the guest threads. Never below one.
uint32_t PipelineBuildWorkerCount(uint32_t worker_cap = PIPELINE_BUILD_WORKERS);

/// The worker count for a machine with the given number of hardware threads, zero meaning unknown.
uint32_t PipelineBuildWorkerCountFor(uint32_t hardware_threads,
                                     uint32_t worker_cap = PIPELINE_BUILD_WORKERS);

/// Options for a build queue. The defaults give the queue its original behaviour.
struct PipelineBuildQueueOptions {
	/// Runs each worker thread below normal priority, so builds yield to the guest and render
	/// threads. Opt in only, and it never changes the priority of the process. This is below normal
	/// on Windows, a higher nice value on Linux and the utility QoS class on macOS. A refusal by
	/// the system is logged once per process and the worker runs on at normal priority. On Linux
	/// and macOS threads that the driver creates from a worker inherit the lowered setting, and on
	/// Apple Silicon the utility class may put the workers on efficiency cores.
	bool low_priority = false;
};

/// Runs jobs on a few worker threads in the order they were posted. It knows nothing about what a
/// job does. Each job is run exactly once, by a worker or by whoever needs the result first.
class PipelineBuildQueue {
public:
	/// Queued until someone claims the job, Running while its work executes, Done after.
	enum class State : uint8_t { Queued, Running, Done };

	/// How Acquire got the job's result.
	enum class Acquired : uint8_t {
		/// The job was still queued, so the caller ran it.
		RanInline,
		/// A worker was running it, and the caller waited.
		Waited,
		/// It had finished already.
		AlreadyDone,
	};

	/// One posted piece of work. Whoever moves it from Queued to Running owns running it.
	class Job {
	public:
		explicit Job(Common::UniqueFunction<void>&& work): m_work(std::move(work)) {}
		KYTY_CLASS_NO_COPY(Job);

		/// The state now. Done means the work's effects are visible to the reader.
		[[nodiscard]] State GetState() const noexcept {
			return m_state.load(std::memory_order_acquire);
		}

	private:
		friend class PipelineBuildQueue;

		Common::UniqueFunction<void> m_work;
		std::atomic<State>           m_state {State::Queued};
		// Released when the work is done, for a caller that finds the job Running.
		std::latch                   m_finished {1};
	};

	using JobPtr = std::shared_ptr<Job>;

	/// Starts `worker_count` threads, configured by `options`.
	explicit PipelineBuildQueue(uint32_t worker_count, PipelineBuildQueueOptions options = {});
	/// Stops the queue first if the owner did not.
	~PipelineBuildQueue();
	KYTY_CLASS_NO_COPY(PipelineBuildQueue);

	/// Queues work behind everything posted before it. Returns null after Stop, and then the work
	/// has not been taken.
	[[nodiscard]] JobPtr Post(Common::UniqueFunction<void>&& work);

	/// How many worker threads were started.
	[[nodiscard]] uint32_t WorkerCount() const noexcept { return static_cast<uint32_t>(m_workers.size()); }

	/// Makes sure the job is done and says how. A queued job is claimed and run on the calling
	/// thread, so a worker can never also run it, and a running one is waited for.
	Acquired Acquire(Job& job);

	/// Drops the queued jobs, waits for the running ones and joins the workers. A dropped job
	/// stays Queued and never runs. Returns how many jobs were running or about to run, which is
	/// what the call had to wait for. Calling it again does nothing and returns 0.
	uint32_t Stop();

	/// The clock a pause deadline is read from.
	using Clock = std::chrono::steady_clock;

	/// The furthest ahead a pause may reach. A deadline past it is cut back, so the clock arithmetic
	/// cannot overflow when a caller passes the end of time.
	static constexpr std::chrono::hours MAX_PAUSE {1};

	/// Makes the workers take no new job until Resume, or until `deadline` passes, whichever comes
	/// first. Jobs that are running finish, and a job someone Acquires still runs on that caller. A
	/// pause that is already in force keeps the earlier of the two deadlines, so asking again cannot
	/// hold the queue longer.
	void Pause(Clock::time_point deadline);

	/// Ends a pause at once and wakes every worker. Does nothing when the queue is not paused.
	void Resume();

	/// True while a pause is in force: asked for and neither resumed nor past its deadline.
	[[nodiscard]] bool Paused() const;

	/// True once Stop has begun.
	[[nodiscard]] bool Stopping() const noexcept {
		return m_stopping.load(std::memory_order_acquire);
	}

private:
	void WorkerLoop(std::stop_token stop);
	/// Runs the job if nobody has claimed it yet.
	static bool RunIfQueued(Job& job);
	/// Runs the work of a job this thread has just moved to Running, then marks it Done.
	static void RunClaimed(Job& job);
	/// Whether a pause is in force. The caller holds m_mutex.
	[[nodiscard]] bool PausedLocked() const;

	mutable std::mutex          m_mutex;
	std::condition_variable_any m_ready;
	std::deque<JobPtr>          m_jobs;
	// Jobs a worker has taken from the queue and not finished. Guarded by m_mutex.
	uint32_t                    m_active = 0;
	// A pause asked for and not resumed. It is in force only until m_pause_deadline. Guarded by m_mutex.
	bool                        m_paused = false;
	Clock::time_point           m_pause_deadline;
	std::atomic<bool>           m_stopping {false};
	// Set in the init list and only read by the workers, so it needs no lock.
	const bool                  m_low_priority;
	// Declared last, so the workers join before the rest is destroyed.
	std::vector<std::jthread>   m_workers;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINEBUILDQUEUE_H_
