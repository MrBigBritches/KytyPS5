#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <latch>
#include <memory>
#include <thread>
#include <vector>

#include "common/config.h"
#if defined(__APPLE__)
#include <pthread/qos.h>
#define KYTY_TEST_HAS_PRIORITY 1
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#define KYTY_TEST_HAS_PRIORITY 1
#elif KYTY_PLATFORM == KYTY_PLATFORM_LINUX
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>
#define KYTY_TEST_HAS_PRIORITY 1
#endif

namespace {
using namespace Libs::Graphics;
using namespace std::chrono_literals;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "PipelineBuildQueueTests:%d: %s\n", __LINE__, #condition);        \
			std::abort();                                                                          \
		}                                                                                          \
	} while (false)

constexpr auto WAIT_LIMIT = 10s;

// Spins until the condition holds, and fails the test if it never does.
template <typename Condition> void WaitFor(Condition condition) {
	const auto deadline = std::chrono::steady_clock::now() + WAIT_LIMIT;
	while (!condition()) {
		CHECK(std::chrono::steady_clock::now() < deadline);
		std::this_thread::yield();
	}
}

// A job that holds its worker until released, so a test decides exactly what is queued behind it.
struct Blocker {
	std::atomic<bool>     started {false};
	std::atomic<bool>     release {false};
	std::atomic<uint32_t> runs {0};
	std::atomic<bool>     finished {false};

	Common::UniqueFunction<void> Work() {
		return [this] {
			runs++;
			started = true;
			WaitFor([this] { return release.load(); });
			finished = true;
		};
	}
};

// Counts its runs and notes which thread ran them.
struct Counted {
	std::atomic<uint32_t> runs {0};
	std::thread::id       ran_on;

	Common::UniqueFunction<void> Work() {
		return [this] {
			ran_on = std::this_thread::get_id();
			runs++;
		};
	}
};

// Runs a job that is queued behind the stale entry of an earlier job, so the earlier entry has been
// taken from the queue by the time this returns.
void DrainQueue(PipelineBuildQueue& queue) {
	Counted sentinel;
	auto    job = queue.Post(sentinel.Work());
	CHECK(job != nullptr);
	WaitFor([&] { return job->GetState() == PipelineBuildQueue::State::Done; });
}

// A job moves through its states, and a worker runs it once when nobody claims it first. This is
// the baseline for the claim tests below, which differ only in the claim.
void TestWorkerRunsJob() {
	PipelineBuildQueue queue(1);
	Blocker            blocker;
	auto               holding = queue.Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });
	CHECK(holding->GetState() == PipelineBuildQueue::State::Running);

	Counted counted;
	auto    job = queue.Post(counted.Work());
	CHECK(job->GetState() == PipelineBuildQueue::State::Queued);
	blocker.release = true;
	WaitFor([&] { return job->GetState() == PipelineBuildQueue::State::Done; });
	CHECK(counted.runs == 1);
	CHECK(counted.ran_on != std::this_thread::get_id());
	CHECK(holding->GetState() == PipelineBuildQueue::State::Done);
}

// A queued job that the caller needs is run by the caller, and the worker that later reaches its
// entry in the queue must not run it again.
void TestInlineClaimRunsOnce() {
	PipelineBuildQueue queue(1);
	Blocker            blocker;
	auto               holding = queue.Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });

	Counted counted;
	auto    job = queue.Post(counted.Work());
	CHECK(queue.Acquire(*job) == PipelineBuildQueue::Acquired::RanInline);
	CHECK(counted.runs == 1);
	CHECK(counted.ran_on == std::this_thread::get_id());
	CHECK(job->GetState() == PipelineBuildQueue::State::Done);

	// The worker is free after this, and it takes the stale entry of the claimed job on the way.
	blocker.release = true;
	DrainQueue(queue);
	CHECK(counted.runs == 1);
	// A second acquire finds it finished and runs nothing.
	CHECK(queue.Acquire(*job) == PipelineBuildQueue::Acquired::AlreadyDone);
	CHECK(counted.runs == 1);
}

// A job a worker is running is waited for, not run a second time, and the waiter sees its effects.
void TestAcquireWaitsForRunningJob() {
	PipelineBuildQueue queue(1);
	Blocker            blocker;
	auto               job = queue.Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });

	std::atomic<bool>                  returned {false};
	std::atomic<bool>                  saw_finished {false};
	PipelineBuildQueue::Acquired       how = PipelineBuildQueue::Acquired::RanInline;
	std::thread                        waiter([&] {
        how          = queue.Acquire(*job);
        saw_finished = blocker.finished.load();
        returned     = true;
    });
	// The job is held, so a waiter that returns now did not wait.
	std::this_thread::sleep_for(100ms);
	CHECK(!returned);
	blocker.release = true;
	waiter.join();
	CHECK(how == PipelineBuildQueue::Acquired::Waited);
	CHECK(saw_finished);
	CHECK(blocker.runs == 1);
}

// Callers and workers race for many jobs, and each job still runs exactly once.
void TestRaceRunsEveryJobOnce() {
	constexpr size_t JOB_COUNT = 20000;
	PipelineBuildQueue queue(PIPELINE_BUILD_WORKERS);
	// Every worker is held while the jobs are posted, so all of them start racing together.
	std::vector<std::unique_ptr<Blocker>> holds;
	for (uint32_t worker = 0; worker < PIPELINE_BUILD_WORKERS; worker++) {
		holds.push_back(std::make_unique<Blocker>());
		(void)queue.Post(holds.back()->Work());
	}
	for (const auto& hold: holds) {
		WaitFor([&] { return hold->started.load(); });
	}
	std::vector<std::unique_ptr<Counted>> counters;
	std::vector<PipelineBuildQueue::JobPtr> jobs;
	for (size_t index = 0; index < JOB_COUNT; index++) {
		counters.push_back(std::make_unique<Counted>());
		jobs.push_back(queue.Post(counters.back()->Work()));
	}
	for (const auto& hold: holds) {
		hold->release = true;
	}
	// The workers take jobs from the front and the caller from the back, so they meet in between.
	for (auto job = jobs.rbegin(); job != jobs.rend(); ++job) {
		queue.Acquire(**job);
	}
	DrainQueue(queue);
	for (size_t index = 0; index < JOB_COUNT; index++) {
		CHECK(counters[index]->runs == 1);
	}
}

// Several threads claim one queued job at the same instant, and exactly one of them runs it. The
// workers are held, so only the claimers can run anything.
void TestSimultaneousClaims() {
	constexpr size_t   ROUNDS   = 3000;
	constexpr uint32_t CLAIMERS = 4;
	PipelineBuildQueue queue(PIPELINE_BUILD_WORKERS);
	std::vector<std::unique_ptr<Blocker>> holds;
	for (uint32_t worker = 0; worker < PIPELINE_BUILD_WORKERS; worker++) {
		holds.push_back(std::make_unique<Blocker>());
		(void)queue.Post(holds.back()->Work());
	}
	for (const auto& hold: holds) {
		WaitFor([&] { return hold->started.load(); });
	}
	std::vector<std::unique_ptr<Counted>>   counters;
	std::vector<PipelineBuildQueue::JobPtr> jobs;
	for (size_t round = 0; round < ROUNDS; round++) {
		counters.push_back(std::make_unique<Counted>());
		jobs.push_back(queue.Post(counters.back()->Work()));
	}
	std::atomic<size_t>                  released {0};
	std::vector<std::atomic<uint32_t>>   arrived(ROUNDS);
	std::vector<std::thread>             claimers;
	for (uint32_t claimer = 0; claimer < CLAIMERS; claimer++) {
		claimers.emplace_back([&] {
			for (size_t round = 0; round < ROUNDS; round++) {
				WaitFor([&] { return released.load() > round; });
				queue.Acquire(*jobs[round]);
				arrived[round]++;
			}
		});
	}
	for (size_t round = 0; round < ROUNDS; round++) {
		released = round + 1;
		WaitFor([&] { return arrived[round].load() == CLAIMERS; });
	}
	for (auto& claimer: claimers) {
		claimer.join();
	}
	for (const auto& hold: holds) {
		hold->release = true;
	}
	for (const auto& counted: counters) {
		CHECK(counted->runs == 1);
	}
}

// Stop drops the jobs that never started, waits for the one that is running, tells the caller how
// many it waited for, frees the dropped work, and may be called again.
void TestStop() {
	PipelineBuildQueue queue(1);
	Blocker            blocker;
	auto               holding = queue.Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });

	Counted counted;
	auto    token = std::make_shared<int>(0);
	auto    dropped = queue.Post([token, &counted] { counted.runs++; });
	auto    other   = queue.Post(counted.Work());
	CHECK(token.use_count() == 2);

	uint32_t    waited_for = 0;
	std::thread stopper([&] { waited_for = queue.Stop(); });
	WaitFor([&] { return queue.Stopping(); });
	// The running job holds Stop in its join.
	CHECK(holding->GetState() == PipelineBuildQueue::State::Running);
	blocker.release = true;
	stopper.join();

	CHECK(waited_for == 1);
	CHECK(holding->GetState() == PipelineBuildQueue::State::Done && blocker.finished);
	CHECK(dropped->GetState() == PipelineBuildQueue::State::Queued);
	CHECK(other->GetState() == PipelineBuildQueue::State::Queued);
	CHECK(counted.runs == 0);
	// Our handle is the only one left to the dropped job, so the work it captured dies with it.
	dropped.reset();
	CHECK(token.use_count() == 1);
	CHECK(queue.Stop() == 0);
	// A refused post leaves the caller its work, which still runs.
	auto refused = counted.Work();
	CHECK(queue.Post(std::move(refused)) == nullptr);
	CHECK(static_cast<bool>(refused));
	refused();
	CHECK(counted.runs == 1);
}

// Without Stop the same queued jobs run, so only the drop explains the result of TestStop.
void TestQueuedJobsRunWithoutStop() {
	PipelineBuildQueue queue(1);
	Blocker            blocker;
	auto               holding = queue.Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });
	Counted first;
	Counted second;
	auto    first_job  = queue.Post(first.Work());
	auto    second_job = queue.Post(second.Work());
	blocker.release    = true;
	WaitFor([&] { return second_job->GetState() == PipelineBuildQueue::State::Done; });
	CHECK(first.runs == 1 && second.runs == 1);
	CHECK(first_job->GetState() == PipelineBuildQueue::State::Done);
}

// The worker count is at most four and leaves half the hardware threads to the guest, but never
// drops below one.
void TestWorkerCount() {
	struct Case {
		uint32_t hardware, workers;
	};
	for (const auto test: {Case{0, 1}, Case{1, 1}, Case{2, 1}, Case{3, 1}, Case{4, 2}, Case{6, 3},
	                       Case{8, 4}, Case{16, 4}, Case{128, 4}}) {
		CHECK(PipelineBuildWorkerCountFor(test.hardware) == test.workers);
	}
	const auto here = PipelineBuildWorkerCount();
	CHECK(here >= 1 && here <= PIPELINE_BUILD_WORKERS);
}

// A queue that is destroyed with work still queued stops itself instead of hanging or running it.
void TestDestructorStops() {
	Counted     counted;
	Blocker     blocker;
	std::thread releaser;
	{
		PipelineBuildQueue queue(1);
		auto               holding = queue.Post(blocker.Work());
		WaitFor([&] { return blocker.started.load(); });
		auto queued = queue.Post(counted.Work());
		releaser    = std::thread([&] {
            std::this_thread::sleep_for(50ms);
            blocker.release = true;
        });
		// Leaving the scope destroys the queue, which has to wait for the held job.
	}
	releaser.join();
	CHECK(blocker.finished);
	CHECK(counted.runs == 0);
}

#ifdef KYTY_TEST_HAS_PRIORITY
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

// What a lowered worker reports, given what an ordinary thread reports.
int LoweredPriority(int ordinary) {
#if defined(__APPLE__)
	(void)ordinary;
	return static_cast<int>(QOS_CLASS_UTILITY);
#elif KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	(void)ordinary;
	return THREAD_PRIORITY_BELOW_NORMAL;
#else
	// These mirror LOW_PRIORITY_NICE and MAX_NICE in pipelineBuildQueue.cpp.
	constexpr int EXPECTED_NICE_STEP = 5;
	constexpr int MAX_NICE           = 19;
	return std::min(ordinary + EXPECTED_NICE_STEP, MAX_NICE);
#endif
}

// Returns the priority each worker of the queue reports. Every job waits for all the others, so
// no worker can take two of them and each worker is seen exactly once.
std::vector<int> WorkerThreadPriorities(PipelineBuildQueue& queue) {
	const uint32_t   worker_count = queue.WorkerCount();
	std::vector<int> priorities(worker_count, -1);
	std::latch       all_started(worker_count);
	std::vector<PipelineBuildQueue::JobPtr> jobs;
	for (uint32_t index = 0; index < worker_count; index++) {
		jobs.push_back(queue.Post([&, index] {
			priorities[index] = CurrentThreadPriority();
			all_started.arrive_and_wait();
		}));
	}
	WaitFor([&] {
		return std::all_of(jobs.begin(), jobs.end(), [](const auto& job) {
			return job->GetState() == PipelineBuildQueue::State::Done;
		});
	});
	return priorities;
}

// The low priority option lowers every worker thread and nothing else. The default queue is the
// baseline that differs only in the option, since a worker could differ for another reason.
void TestLowPriorityWorkers() {
	constexpr uint32_t WORKERS         = 3;
	const int          caller_priority = CurrentThreadPriority();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto process_class = GetPriorityClass(GetCurrentProcess());
#endif
	int ordinary = 0;
	{
		PipelineBuildQueue queue(WORKERS);
		for (const int priority: WorkerThreadPriorities(queue)) {
			CHECK(priority == caller_priority);
			ordinary = priority;
		}
	}
	// A runner already at the lowest priority could not tell a lowered worker from an ordinary one.
	CHECK(LoweredPriority(ordinary) != ordinary);
	{
		PipelineBuildQueue queue(WORKERS, PipelineBuildQueueOptions {.low_priority = true});
		const auto         priorities = WorkerThreadPriorities(queue);
		CHECK(priorities.size() == WORKERS);
		for (const int priority: priorities) {
			CHECK(priority == LoweredPriority(ordinary));
		}
	}
	CHECK(CurrentThreadPriority() == caller_priority);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	CHECK(GetPriorityClass(GetCurrentProcess()) == process_class);
#endif
}
#endif

// A queue with the option set still runs its jobs.
void TestLowPriorityQueueRunsJobs() {
	PipelineBuildQueue queue(1, PipelineBuildQueueOptions {.low_priority = true});
	Counted            counted;
	auto               job = queue.Post(counted.Work());
	WaitFor([&] { return job->GetState() == PipelineBuildQueue::State::Done; });
	CHECK(counted.runs == 1);
}
} // namespace

int main() {
#ifdef KYTY_TEST_HAS_PRIORITY
	TestLowPriorityWorkers();
#endif
	TestLowPriorityQueueRunsJobs();
	TestWorkerRunsJob();
	TestInlineClaimRunsOnce();
	TestAcquireWaitsForRunningJob();
	TestRaceRunsEveryJobOnce();
	TestSimultaneousClaims();
	TestStop();
	TestQueuedJobsRunWithoutStop();
	TestWorkerCount();
	TestDestructorStops();
	std::puts("Pipeline build queue tests passed");
}
