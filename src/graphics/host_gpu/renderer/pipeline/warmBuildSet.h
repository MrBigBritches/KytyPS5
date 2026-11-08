#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WARMBUILDSET_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WARMBUILDSET_H_

#include "common/common.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Libs::Graphics {
/// The most threads that build warm-up pipelines. Offline measurement (A0) found that 4 workers
/// cost 1.10x the summed compile time of building one at a time and 8 workers cost 1.34x, so the
/// warm-up keeps to 4. It is its own constant on purpose and is not tied to the async queue's cap.
/// A cap shared by the warm and async queues is deferred to the step that warms graphics pipelines.
inline constexpr uint32_t WARM_UP_WORKERS = 4;

/// How many warm-up workers to start on a machine with the given number of hardware threads, zero
/// meaning unknown. At most WARM_UP_WORKERS and at most half the hardware threads, so a small
/// machine keeps room for the guest. Never below one.
constexpr uint32_t WarmUpWorkerCount(uint32_t hardware_threads) {
	const uint32_t half = hardware_threads / 2;
	const uint32_t most = half < WARM_UP_WORKERS ? half : WARM_UP_WORKERS;
	return most < 1 ? 1 : most;
}

/// Builds that start before anything asks for them and are taken by key when something does. Each
/// build writes into storage that only its job owns, so a worker never touches the structure the
/// owner reads, and the owner adopts the result by moving it out. The owner is one thread, which
/// starts the builds before it begins to look them up, and it alone calls every member except the
/// work given to Start. The owner must call Stop before the set is destroyed, because a finished
/// build nobody took still owns what it created and only Stop hands that back to be freed.
template <typename Key, typename Result> class WarmBuildSet {
public:
	/// Fills the result of one build. It runs on a worker, or on the thread that takes the build.
	using Work = Common::UniqueFunction<void, Result&>;

	/// A build handed over to its taker, and how the taker got it.
	struct Taken {
		Result                         result;
		PipelineBuildQueue::Acquired how;
	};

	/// What is left when the set stops.
	struct Stopped {
		/// Finished builds that nobody took, which still own whatever they created.
		std::vector<Result> unadopted;
		/// How many builds were running, which the stop had to wait for.
		uint32_t            waited_for = 0;
	};

	/// Starts the workers of the queue the builds run on.
	WarmBuildSet(uint32_t worker_count, PipelineBuildQueueOptions options): m_queue(worker_count, options) {}
	KYTY_CLASS_NO_COPY(WarmBuildSet);

	/// Queues the build under the key. Returns false and drops the work when the key is already
	/// pending or the set has stopped.
	bool Start(const Key& key, Work work) {
		if (m_pending.contains(key)) {
			return false;
		}
		auto slot = std::make_shared<Result>();
		auto job  = m_queue.Post([slot, work = std::move(work)] { work(*slot); });
		if (job == nullptr) {
			return false;
		}
		m_pending.emplace(key, Entry {std::move(slot), std::move(job)});
		return true;
	}

	/// True while a build is pending under the key. Lets a caller skip the cost of a take that
	/// would find nothing.
	[[nodiscard]] bool IsPending(const Key& key) const { return m_pending.contains(key); }

	/// True when the build under the key has finished, so taking it neither runs nor waits.
	[[nodiscard]] bool IsDone(const Key& key) const {
		const auto found = m_pending.find(key);
		return found != m_pending.end() &&
		       found->second.job->GetState() == PipelineBuildQueue::State::Done;
	}

	/// Takes the build under the key if one is pending. A build that has not started is run here by
	/// the caller, a running one is waited for, and a finished one is handed over at once. Returns
	/// nothing when the key is not pending, and then the caller builds it as it would without a set.
	[[nodiscard]] std::optional<Taken> Take(const Key& key) {
		const auto found = m_pending.find(key);
		if (found == m_pending.end()) {
			return std::nullopt;
		}
		// Kept past the erase below, since the entry owns the only other handle.
		const auto entry = found->second;
		const auto how   = m_queue.Acquire(*entry.job);
		m_pending.erase(found);
		return Taken {std::move(*entry.slot), how};
	}

	/// How many builds are pending, started and not yet taken.
	[[nodiscard]] size_t Pending() const noexcept { return m_pending.size(); }

	/// Drops the builds that never started, waits for the running ones and hands back the finished
	/// ones nobody took. Nothing runs after it returns. Calling it again returns nothing.
	Stopped Stop() {
		Stopped stopped;
		stopped.waited_for = m_queue.Stop();
		for (auto& pending: m_pending) {
			// A build that never ran created nothing, so only the finished ones have anything to free.
			if (pending.second.job->GetState() == PipelineBuildQueue::State::Done) {
				stopped.unadopted.push_back(std::move(*pending.second.slot));
			}
		}
		m_pending.clear();
		return stopped;
	}

	/// Holds the workers from new builds until Resume or the deadline. A build that someone takes
	/// still runs on the taker, and a running one finishes.
	void Pause(PipelineBuildQueue::Clock::time_point deadline) { m_queue.Pause(deadline); }
	/// Lets the workers take builds again.
	void Resume() { m_queue.Resume(); }
	/// True while a pause is in force.
	[[nodiscard]] bool Paused() const { return m_queue.Paused(); }

	/// True once the set has begun to stop.
	[[nodiscard]] bool Stopping() const noexcept { return m_queue.Stopping(); }

	/// The queue the builds run on, for a test that holds its workers.
	[[nodiscard]] PipelineBuildQueue& Queue() noexcept { return m_queue; }

private:
	struct Entry {
		std::shared_ptr<Result>       slot;
		PipelineBuildQueue::JobPtr job;
	};

	PipelineBuildQueue                  m_queue;
	std::unordered_map<Key, Entry> m_pending;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_WARMBUILDSET_H_
