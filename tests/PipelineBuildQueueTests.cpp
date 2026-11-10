#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"
#include "graphics/host_gpu/renderer/pipeline/warmBuildSet.h"
#include "graphics/host_gpu/renderer/pipeline/warmLane.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <latch>
#include <memory>
#include <mutex>
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
	// Deferred pipeline draws raise the cap to six, and the half of the hardware threads still
	// bounds it.
	struct CappedCase {
		uint32_t hardware, cap, workers;
	};
	for (const auto test: {CappedCase{8, DEFERRED_PIPELINE_BUILD_WORKERS, 4},
	                       CappedCase{12, DEFERRED_PIPELINE_BUILD_WORKERS, 6},
	                       CappedCase{32, DEFERRED_PIPELINE_BUILD_WORKERS, 6},
	                       CappedCase{32, PIPELINE_BUILD_WORKERS, 4},
	                       CappedCase{32, 0, 1}}) {
		CHECK(PipelineBuildWorkerCountFor(test.hardware, test.cap) == test.workers);
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
	// These mirror LOW_PRIORITY_NICE and MAX_NICE in src/common/threadPriority.cpp.
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

// What a warm build produces. The resource stands for the Vulkan objects a pipeline owns, which
// only a build that ran has made.
struct Built {
	std::shared_ptr<int> resource;
	std::thread::id      ran_on;
};

using Warm = WarmBuildSet<uint32_t, Built>;

// One warm build, counting its compiles. A held build waits for `release` after it has made its
// resource, so a test decides when it finishes.
struct WarmProbe {
	std::atomic<uint32_t> builds {0};
	std::atomic<bool>     started {false};
	std::atomic<bool>     release {false};
	std::shared_ptr<int>  token = std::make_shared<int>(0);

	Warm::Work Work(bool hold) {
		return [this, hold](Built& built) {
			builds++;
			built.resource = token;
			built.ran_on   = std::this_thread::get_id();
			started        = true;
			if (hold) {
				WaitFor([this] { return release.load(); });
			}
		};
	}
};

// What a lookup in the pipeline cache does with a key, in miniature. A pending warm build is taken
// and anything else is built here, so the number of builds says whether the lookup checked.
Built LookupOrBuild(Warm& set, uint32_t key, WarmProbe& fallback,
                    PipelineBuildQueue::Acquired* how = nullptr) {
	if (auto taken = set.Take(key)) {
		if (how != nullptr) {
			*how = taken->how;
		}
		return std::move(taken->result);
	}
	Built built;
	fallback.Work(false)(built);
	return built;
}

// Holds the only worker of the set until released, so what is started behind it stays queued.
PipelineBuildQueue::JobPtr HoldWorker(Warm& set, Blocker& blocker) {
	auto holding = set.Queue().Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });
	return holding;
}

// A lookup that meets a warm build a worker is running waits for it and adopts its result. The
// build is not made a second time. The baseline looks up a key that nobody started, which has to
// build, so a count of one can only come from the warm build.
void TestWarmLookupAdoptsRunningBuild() {
	constexpr uint32_t KEY = 7;
	// The probes outlive the set, whose worker reads them until it is joined.
	WarmProbe  warm;
	WarmProbe  fallback;
	WarmProbe  other_fallback;
	Warm       set(1, {});
	CHECK(set.Start(KEY, warm.Work(true)));
	WaitFor([&] { return warm.started.load(); });
	CHECK(set.IsPending(KEY));

	Built       adopted;
	auto        how = PipelineBuildQueue::Acquired::AlreadyDone;
	std::thread::id looker_id;
	std::thread looker([&] {
		looker_id = std::this_thread::get_id();
		adopted   = LookupOrBuild(set, KEY, fallback, &how);
	});
	// The build is held, so a lookup that returns now did not wait for it.
	std::this_thread::sleep_for(100ms);
	CHECK(set.IsPending(KEY));
	warm.release = true;
	looker.join();
	CHECK(warm.builds == 1);
	CHECK(fallback.builds == 0);
	CHECK(adopted.resource == warm.token);
	// The looker waited for the worker's build. A lookup that ran or redid it would say otherwise.
	CHECK(how == PipelineBuildQueue::Acquired::Waited);
	CHECK(adopted.ran_on != looker_id && adopted.ran_on != std::this_thread::get_id());
	CHECK(!set.IsPending(KEY));
	CHECK(set.Pending() == 0);

	const auto other = LookupOrBuild(set, KEY + 1, other_fallback);
	CHECK(other_fallback.builds == 1);
	CHECK(other.resource == other_fallback.token);
}

// A warm build no worker has started is run by the lookup, once. The worker that later reaches it
// finds it taken. The baseline lets the worker run an identical build, which differs only in that
// nobody looked it up.
void TestWarmLookupClaimsQueuedBuild() {
	constexpr uint32_t KEY = 3;
	{
		Blocker   blocker;
		WarmProbe warm;
		Warm      set(1, {});
		const auto holding = HoldWorker(set, blocker);
		CHECK(set.Start(KEY, warm.Work(false)));
		blocker.release = true;
		// The one worker is free again only when the build is done, which the sentinel waits for.
		DrainQueue(set.Queue());
		CHECK(warm.builds == 1);
		const auto taken = set.Take(KEY);
		CHECK(taken.has_value());
		CHECK(taken->how == PipelineBuildQueue::Acquired::AlreadyDone);
		CHECK(taken->result.ran_on != std::this_thread::get_id());
	}
	Blocker   blocker;
	WarmProbe warm;
	Warm      set(1, {});
	const auto holding = HoldWorker(set, blocker);
	CHECK(set.Start(KEY, warm.Work(false)));
	auto taken = set.Take(KEY);
	CHECK(taken.has_value());
	CHECK(taken->how == PipelineBuildQueue::Acquired::RanInline);
	CHECK(taken->result.ran_on == std::this_thread::get_id());
	CHECK(warm.builds == 1);
	blocker.release = true;
	DrainQueue(set.Queue());
	CHECK(warm.builds == 1);
}

// Starting a key that is already pending, or after the stop, takes no work and runs nothing.
void TestWarmStartRefusals() {
	constexpr uint32_t KEY = 1;
	Blocker   blocker;
	WarmProbe first;
	WarmProbe second;
	WarmProbe late;
	Warm      set(1, {});
	const auto holding = HoldWorker(set, blocker);
	CHECK(set.Start(KEY, first.Work(false)));
	CHECK(!set.Start(KEY, second.Work(false)));
	CHECK(set.Pending() == 1);
	blocker.release = true;
	CHECK(set.Take(KEY).has_value());
	CHECK(first.builds == 1 && second.builds == 0);
	(void)set.Stop();
	CHECK(!set.Start(KEY + 1, late.Work(false)));
	CHECK(set.Pending() == 0);
}

// Stopping with builds in every state waits for the running one, hands back the resources of the
// finished ones that nobody took, runs none of the queued ones and may be done twice. The baseline
// does not stop, and then every queued build runs, so the drop is the stop's doing.
void TestWarmStopWithPendingBuilds() {
	constexpr uint32_t DONE = 1, RUNNING = 2, QUEUED = 3;
	{
		WarmProbe done, running, queued;
		Warm      set(1, {});
		CHECK(set.Start(DONE, done.Work(false)));
		CHECK(set.Start(RUNNING, running.Work(false)));
		CHECK(set.Start(QUEUED, queued.Work(false)));
		WaitFor([&] { return queued.builds.load() == 1; });
		CHECK(done.builds == 1 && running.builds == 1);
	}
	WarmProbe done, running, queued;
	Warm      set(1, {});
	CHECK(set.Start(DONE, done.Work(false)));
	// The finished build is not taken before the stop, so only the stop can reach it.
	WaitFor([&] { return done.builds.load() == 1; });
	CHECK(set.Start(RUNNING, running.Work(true)));
	WaitFor([&] { return running.started.load(); });
	CHECK(set.Start(QUEUED, queued.Work(false)));
	CHECK(done.token.use_count() == 2);

	Warm::Stopped stopped;
	std::thread   stopper([&] { stopped = set.Stop(); });
	WaitFor([&] { return set.Stopping(); });
	// The running build holds the stop until it is released.
	CHECK(set.Pending() == 3);
	running.release = true;
	stopper.join();
	CHECK(stopped.waited_for == 1);
	CHECK(stopped.unadopted.size() == 2);
	CHECK(queued.builds == 0);
	CHECK(queued.token.use_count() == 1);
	CHECK(set.Pending() == 0);
	CHECK(set.Stop().unadopted.empty());
}

// A set that is destroyed after it stopped has nothing left to wait for, and a stop with nothing
// pending returns nothing.
void TestWarmStopWithNothingPending() {
	Warm set(2, {});
	const auto stopped = set.Stop();
	CHECK(stopped.unadopted.empty() && stopped.waited_for == 0);
}

// IsDone is true only for a build that has finished and is still pending. It is false while the
// build is queued, while it runs, after it was taken and for a key nobody started, and true again
// once the held build is released and drained, so neither an always-true nor an always-false
// answer can pass.
void TestWarmIsDone() {
	constexpr uint32_t QUEUED = 1, RUNNING = 2, UNKNOWN = 99;
	Blocker   blocker;
	WarmProbe queued;
	WarmProbe running;
	Warm      set(1, {});
	const auto holding = HoldWorker(set, blocker);
	CHECK(set.Start(QUEUED, queued.Work(false)));
	CHECK(!set.IsDone(QUEUED));
	CHECK(!set.IsDone(UNKNOWN));
	blocker.release = true;
	DrainQueue(set.Queue());
	CHECK(set.IsDone(QUEUED));

	CHECK(set.Start(RUNNING, running.Work(true)));
	WaitFor([&] { return running.started.load(); });
	CHECK(!set.IsDone(RUNNING));
	CHECK(set.IsDone(QUEUED));
	running.release = true;
	DrainQueue(set.Queue());
	CHECK(set.IsDone(RUNNING));

	CHECK(set.Take(QUEUED).has_value());
	CHECK(!set.IsDone(QUEUED));
	CHECK(set.IsDone(RUNNING));
}

// The warm-up starts four workers on a big machine, half the hardware threads on a small one and
// never fewer than one. It does not follow the async queue's cap, which a later change may move.
void TestWarmUpWorkerCount() {
	static_assert(WarmUpWorkerCount(0) == 1);
	static_assert(WarmUpWorkerCount(1) == 1);
	static_assert(WarmUpWorkerCount(2) == 1);
	static_assert(WarmUpWorkerCount(7) == 3);
	static_assert(WarmUpWorkerCount(8) == 4);
	static_assert(WarmUpWorkerCount(64) == 4);
	CHECK(WarmUpWorkerCount(std::thread::hardware_concurrency()) >= 1);
}
} // namespace

// How long a test watches a paused job to see that it stays queued. No signal says that a worker
// did not start something, so this is the one place a short fixed wait is the only way to look.
constexpr auto QUIET_WINDOW = 100ms;

// True when the job is still queued after the quiet window.
bool StaysQueued(const PipelineBuildQueue::Job& job) {
	const auto end = std::chrono::steady_clock::now() + QUIET_WINDOW;
	while (std::chrono::steady_clock::now() < end) {
		if (job.GetState() != PipelineBuildQueue::State::Queued) {
			return false;
		}
		std::this_thread::yield();
	}
	return true;
}

constexpr auto FAR_FUTURE = 1h;

// Makes sure the workers are awake and idle, so a job posted next is taken at once unless a pause
// stops it. Without this the first job could be slow only because the thread had not started.
void WarmWorker(PipelineBuildQueue& queue) {
	DrainQueue(queue);
}

// Test 12. A paused queue still gives a job to whoever acquires it, and the caller runs it inline.
// The same job on an unpaused queue is run by the worker, which is the baseline: the pause is the
// only difference between them.
void TestPausedQueueRunsAcquiredJobInline() {
	{
		PipelineBuildQueue queue(1);
		WarmWorker(queue);
		Counted counted;
		auto    job = queue.Post(counted.Work());
		WaitFor([&] { return job->GetState() == PipelineBuildQueue::State::Done; });
		CHECK(queue.Acquire(*job) == PipelineBuildQueue::Acquired::AlreadyDone);
		CHECK(counted.ran_on != std::this_thread::get_id());
	}
	PipelineBuildQueue queue(1);
	WarmWorker(queue);
	queue.Pause(std::chrono::steady_clock::now() + FAR_FUTURE);
	CHECK(queue.Paused());
	Counted counted;
	auto    job = queue.Post(counted.Work());
	CHECK(StaysQueued(*job));
	CHECK(queue.Acquire(*job) == PipelineBuildQueue::Acquired::RanInline);
	CHECK(counted.runs == 1 && counted.ran_on == std::this_thread::get_id());
}

// A job that notes when and where it ran, for the tests of a deadline. They read the time the job
// ran at, which a window of watching could not tell from a job that was never started.
struct Timed {
	std::atomic<bool>                      ran {false};
	std::chrono::steady_clock::time_point ran_at;
	std::thread::id                        ran_on;

	Common::UniqueFunction<void> Work() {
		return [this] {
			ran_at = std::chrono::steady_clock::now();
			ran_on = std::this_thread::get_id();
			ran    = true;
		};
	}
};

// A paused queue starts no job before its deadline, and starts it once the deadline has passed, with
// no Resume. The job says when it ran, so the test needs no window of watching. The unpaused run in
// the inline test above is the baseline for starting at once.
void TestPausedQueueWaitsForDeadline() {
	PipelineBuildQueue queue(1);
	WarmWorker(queue);
	constexpr auto PAUSE = 300ms;
	const auto     start = std::chrono::steady_clock::now();
	queue.Pause(start + PAUSE);
	Timed timed;
	auto  job = queue.Post(timed.Work());
	WaitFor([&] { return job->GetState() == PipelineBuildQueue::State::Done; });
	CHECK(timed.ran_at >= start + PAUSE);
	CHECK(timed.ran_on != std::this_thread::get_id());
	CHECK(!queue.Paused());
}

// Resume ends a pause at once and wakes workers that are waiting on the far deadline. The jobs run
// on workers long before the deadline, and without the Resume they stay queued.
void TestResumeWakesWorkers() {
	PipelineBuildQueue queue(2);
	WarmWorker(queue);
	queue.Pause(std::chrono::steady_clock::now() + FAR_FUTURE);
	Counted first;
	Counted second;
	auto    first_job  = queue.Post(first.Work());
	auto    second_job = queue.Post(second.Work());
	CHECK(StaysQueued(*first_job) && StaysQueued(*second_job));
	queue.Resume();
	CHECK(!queue.Paused());
	WaitFor([&] {
		return first_job->GetState() == PipelineBuildQueue::State::Done &&
		       second_job->GetState() == PipelineBuildQueue::State::Done;
	});
	CHECK(first.ran_on != std::this_thread::get_id() && second.ran_on != std::this_thread::get_id());
}

// A job that is running when the pause starts finishes, and the job queued behind it waits for the
// resume. Without the pause the second job would start as soon as the first ends.
void TestPausedQueueLetsRunningJobFinish() {
	PipelineBuildQueue queue(1);
	Blocker            blocker;
	auto               running = queue.Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });
	Counted counted;
	auto    waiting = queue.Post(counted.Work());
	queue.Pause(std::chrono::steady_clock::now() + FAR_FUTURE);
	blocker.release = true;
	WaitFor([&] { return running->GetState() == PipelineBuildQueue::State::Done; });
	CHECK(blocker.finished.load());
	CHECK(StaysQueued(*waiting));
	queue.Resume();
	WaitFor([&] { return waiting->GetState() == PipelineBuildQueue::State::Done; });
	CHECK(counted.runs == 1);
}

// Pausing again while paused keeps the earlier deadline, so repeated pauses cannot hold the queue
// for longer than the first one asked. In either order the job runs after the short deadline and
// well before the far one. Asking for the short deadline second reaches workers that are already
// waiting on the far one, which the first order does not.
void TestRepeatedPauseKeepsEarlierDeadline() {
	for (const bool long_first: {false, true}) {
		PipelineBuildQueue queue(1);
		WarmWorker(queue);
		const auto start = std::chrono::steady_clock::now();
		Timed      timed;
		PipelineBuildQueue::JobPtr job;
		if (long_first) {
			queue.Pause(start + FAR_FUTURE);
			// The worker has a job and is waiting on the far deadline when the short one arrives.
			job = queue.Post(timed.Work());
			CHECK(StaysQueued(*job));
			queue.Pause(start + 300ms);
		} else {
			queue.Pause(start + 300ms);
			queue.Pause(start + FAR_FUTURE);
			job = queue.Post(timed.Work());
		}
		WaitFor([&] { return job->GetState() == PipelineBuildQueue::State::Done; });
		CHECK(timed.ran_at >= start + 300ms);
		CHECK(timed.ran_at < start + 5s);
		CHECK(timed.ran_on != std::this_thread::get_id());
	}
}

// A deadline at the end of time is cut back to the longest pause, so the clock arithmetic cannot
// overflow, and the queue is still held and still resumes.
void TestPauseClampsFarDeadline() {
	PipelineBuildQueue queue(1);
	WarmWorker(queue);
	queue.Pause(std::chrono::steady_clock::time_point::max());
	CHECK(queue.Paused());
	Timed timed;
	auto  job = queue.Post(timed.Work());
	CHECK(StaysQueued(*job));
	queue.Resume();
	WaitFor([&] { return job->GetState() == PipelineBuildQueue::State::Done; });
	CHECK(timed.ran_on != std::this_thread::get_id());
}

// Stopping a paused queue neither hangs nor runs what was queued, and the destructor does the same.
void TestStopPausedQueue() {
	Counted                    counted;
	PipelineBuildQueue::JobPtr job;
	{
		PipelineBuildQueue queue(2);
		WarmWorker(queue);
		queue.Pause(std::chrono::steady_clock::now() + FAR_FUTURE);
		job = queue.Post(counted.Work());
		CHECK(StaysQueued(*job));
		CHECK(queue.Stop() == 0);
	}
	CHECK(job->GetState() == PipelineBuildQueue::State::Queued && counted.runs == 0);
	{
		PipelineBuildQueue queue(1);
		queue.Pause(std::chrono::steady_clock::now() + FAR_FUTURE);
		(void)queue.Post(counted.Work());
	}
	CHECK(counted.runs == 0);
}

// A warm set that is paused hands a build that no worker started to the caller who takes it, and a
// resumed set runs the next one on a worker. The first take, from an unpaused set, is the baseline.
void TestPausedWarmSetTakesInline() {
	Warm      set(1, {});
	WarmProbe probe;
	CHECK(set.Start(1, probe.Work(false)));
	WaitFor([&] { return set.IsDone(1); });
	auto taken = set.Take(1);
	CHECK(taken && taken->how == PipelineBuildQueue::Acquired::AlreadyDone);
	CHECK(taken->result.ran_on != std::this_thread::get_id());

	set.Pause(std::chrono::steady_clock::now() + FAR_FUTURE);
	CHECK(set.Paused());
	CHECK(set.Start(2, probe.Work(false)));
	auto inline_taken = set.Take(2);
	CHECK(inline_taken && inline_taken->how == PipelineBuildQueue::Acquired::RanInline);
	CHECK(inline_taken->result.ran_on == std::this_thread::get_id());
	set.Resume();
	CHECK(!set.Paused());
	CHECK(set.Start(3, probe.Work(false)));
	WaitFor([&] { return set.IsDone(3); });
	auto resumed = set.Take(3);
	CHECK(resumed && resumed->result.ran_on != std::this_thread::get_id());
	(void)set.Stop();
}

using Lane = PipelineBuildQueue::Lane;

// Notes the order in which numbered jobs ran, from any thread.
struct RunOrder {
	std::mutex       mutex;
	std::vector<int> ids;

	Common::UniqueFunction<void> Work(int id) {
		return [this, id] {
			std::scoped_lock lock(mutex);
			ids.push_back(id);
		};
	}
};

// One numbered job and the lane it is posted to.
struct Posting {
	int  id;
	Lane lane;
};

// Posts the jobs in order behind a held worker, releases it and returns the order they ran in. The
// only worker runs one job at a time, so the order is the order the queue chose.
std::vector<int> RunOrderOf(const std::vector<Posting>& postings) {
	PipelineBuildQueue queue(1);
	Blocker            blocker;
	auto               holding = queue.Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });
	RunOrder                                order;
	std::vector<PipelineBuildQueue::JobPtr> jobs;
	for (const auto& posting: postings) {
		jobs.push_back(queue.Post(order.Work(posting.id), posting.lane));
		CHECK(jobs.back() != nullptr);
	}
	blocker.release = true;
	WaitFor([&] {
		return std::all_of(jobs.begin(), jobs.end(), [](const auto& job) {
			return job->GetState() == PipelineBuildQueue::State::Done;
		});
	});
	return order.ids;
}

// Each lane is first in, first out. The posts alternate between the lanes, so the order 1 to 6 can
// only come from two separate lines that each keep their order and are read Front first. The same
// posts all on Front are the baseline, which run in the order posted.
void TestLanesAreFifoEach() {
	const std::vector<int> expected {1, 2, 3, 4, 5, 6};
	const auto             split = RunOrderOf({{4, Lane::Back},
	                                           {1, Lane::Front},
	                                           {5, Lane::Back},
	                                           {2, Lane::Front},
	                                           {6, Lane::Back},
	                                           {3, Lane::Front}});
	CHECK(split == expected);
	const std::vector<int> posted {4, 1, 5, 2, 6, 3};
	const auto             merged = RunOrderOf({{4, Lane::Front},
	                                            {1, Lane::Front},
	                                            {5, Lane::Front},
	                                            {2, Lane::Front},
	                                            {6, Lane::Front},
	                                            {3, Lane::Front}});
	CHECK(merged == posted);
	// A lane on its own keeps the order it was posted in.
	const std::vector<int> back_only {1, 2, 3, 4};
	const auto             alone =
	    RunOrderOf({{1, Lane::Back}, {2, Lane::Back}, {3, Lane::Back}, {4, Lane::Back}});
	CHECK(alone == back_only);
}

// A Back job posted first runs after the Front jobs posted later. All on Front they run as posted.
void TestBackRunsAfterFront() {
	constexpr int X = 1, Y = 2, Z = 3;
	const auto    back_first = RunOrderOf({{X, Lane::Back}, {Y, Lane::Front}, {Z, Lane::Front}});
	const std::vector<int> expected_back_last {Y, Z, X};
	CHECK(back_first == expected_back_last);
	const auto all_front = RunOrderOf({{X, Lane::Front}, {Y, Lane::Front}, {Z, Lane::Front}});
	const std::vector<int> expected_posted {X, Y, Z};
	CHECK(all_front == expected_posted);
}

// With nothing on Front the Back jobs all run, and on idle workers with no one holding them.
void TestBackJobsAreNotStarved() {
	PipelineBuildQueue queue(2);
	WarmWorker(queue);
	constexpr size_t                        BACK_JOBS = 5;
	std::vector<std::unique_ptr<Counted>>   counters;
	std::vector<PipelineBuildQueue::JobPtr> jobs;
	for (size_t index = 0; index < BACK_JOBS; index++) {
		counters.push_back(std::make_unique<Counted>());
		jobs.push_back(queue.Post(counters.back()->Work(), Lane::Back));
	}
	WaitFor([&] {
		return std::all_of(jobs.begin(), jobs.end(), [](const auto& job) {
			return job->GetState() == PipelineBuildQueue::State::Done;
		});
	});
	for (const auto& counted: counters) {
		CHECK(counted->runs == 1 && counted->ran_on != std::this_thread::get_id());
	}
}

// A pause holds a job of either lane until Resume. The Front lane is the baseline, and the Back job
// is alone in its queue, so nothing on Front could be what keeps the worker waiting.
void TestPauseHoldsBothLanes() {
	for (const Lane lane: {Lane::Front, Lane::Back}) {
		PipelineBuildQueue queue(1);
		WarmWorker(queue);
		queue.Pause(std::chrono::steady_clock::now() + FAR_FUTURE);
		Counted counted;
		auto    job = queue.Post(counted.Work(), lane);
		CHECK(StaysQueued(*job));
		CHECK(counted.runs == 0);
		queue.Resume();
		WaitFor([&] { return job->GetState() == PipelineBuildQueue::State::Done; });
		CHECK(counted.runs == 1 && counted.ran_on != std::this_thread::get_id());
	}
}

// Stop drops the queued jobs of either lane and frees the work they captured, with the Front lane
// as the baseline. The weak handle expires only when the queue let go of the job, so it is read
// while the queue is still alive.
void TestStopDropsBothLanes() {
	for (const Lane lane: {Lane::Front, Lane::Back}) {
		PipelineBuildQueue queue(1);
		Blocker            blocker;
		auto               holding = queue.Post(blocker.Work());
		WaitFor([&] { return blocker.started.load(); });

		Counted                  counted;
		auto                     token   = std::make_shared<int>(0);
		const std::weak_ptr<int> watch   = token;
		auto                     dropped = queue.Post([token, &counted] { counted.runs++; }, lane);
		token.reset();
		CHECK(!watch.expired());

		std::thread stopper([&] { (void)queue.Stop(); });
		WaitFor([&] { return queue.Stopping(); });
		blocker.release = true;
		stopper.join();

		CHECK(dropped->GetState() == PipelineBuildQueue::State::Queued);
		CHECK(counted.runs == 0);
		dropped.reset();
		CHECK(watch.expired());
	}
}

// The warm set passes the lane on. The Back build is started first and the Front build second, and
// the Front one is built first. Both on Front are built in the order started.
void TestWarmSetLaneOrder() {
	constexpr uint32_t K1 = 1, K2 = 2;
	for (const Lane first_lane: {Lane::Back, Lane::Front}) {
		RunOrder   order;
		Blocker    blocker;
		Warm       set(1, {});
		const auto holding = HoldWorker(set, blocker);
		CHECK(set.Start(K1, [&order](Built&) { order.Work(1)(); }, first_lane));
		CHECK(set.Start(K2, [&order](Built&) { order.Work(2)(); }, Lane::Front));
		blocker.release = true;
		WaitFor([&] { return set.IsDone(K1) && set.IsDone(K2); });
		const std::vector<int> expected =
		    first_lane == Lane::Back ? std::vector<int> {2, 1} : std::vector<int> {1, 2};
		CHECK(order.ids == expected);
		(void)set.Stop();
	}
}

// A Back job that a caller claims while Front is not empty runs once, on the caller. The worker
// reaches its stale entry later and skips it. A sentinel behind it on Back says the entry was
// taken.
void TestInlineClaimOfBackJobRunsOnce() {
	PipelineBuildQueue queue(1);
	Blocker            blocker;
	auto               holding = queue.Post(blocker.Work());
	WaitFor([&] { return blocker.started.load(); });

	Counted front;
	Counted back;
	auto    front_job = queue.Post(front.Work(), Lane::Front);
	auto    back_job  = queue.Post(back.Work(), Lane::Back);
	CHECK(front_job->GetState() == PipelineBuildQueue::State::Queued);
	CHECK(queue.Acquire(*back_job) == PipelineBuildQueue::Acquired::RanInline);
	CHECK(back.runs == 1 && back.ran_on == std::this_thread::get_id());
	CHECK(front.runs == 0);

	blocker.release = true;
	Counted sentinel;
	auto    last = queue.Post(sentinel.Work(), Lane::Back);
	WaitFor([&] { return last->GetState() == PipelineBuildQueue::State::Done; });
	CHECK(front.runs == 1);
	CHECK(back.runs == 1);
	CHECK(queue.Acquire(*back_job) == PipelineBuildQueue::Acquired::AlreadyDone);
	CHECK(back.runs == 1);
}

// A generated module's build goes on the back lane exactly when the module is wide tail. The rows
// differ only in the wide flag, so a rule that ignores it gives a wrong lane in one of them. An
// unknown candidate count is never wide, so it stays in front.
void TestWarmLaneOf() {
	struct Row {
		bool                    wide_tail;
		std::optional<uint64_t> candidates;
		Lane                    expected;
	};
	const Row rows[] = {
	    {true, 3, Lane::Back},
	    {false, 3, Lane::Front},
	    {false, std::nullopt, Lane::Front},
	};
	for (const auto& row: rows) {
		const StoreGen::GeneratedOrder order {
		    .tier = 1, .candidates = row.candidates, .wide_tail = row.wide_tail};
		CHECK(WarmLaneOf(order) == row.expected);
	}
}

int main() {
	TestWarmLaneOf();
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
	TestWarmLookupAdoptsRunningBuild();
	TestWarmLookupClaimsQueuedBuild();
	TestWarmStartRefusals();
	TestWarmStopWithPendingBuilds();
	TestWarmStopWithNothingPending();
	TestWarmIsDone();
	TestWarmUpWorkerCount();
	TestPausedQueueRunsAcquiredJobInline();
	TestPausedQueueWaitsForDeadline();
	TestResumeWakesWorkers();
	TestPausedQueueLetsRunningJobFinish();
	TestRepeatedPauseKeepsEarlierDeadline();
	TestStopPausedQueue();
	TestPauseClampsFarDeadline();
	TestPausedWarmSetTakesInline();
	TestLanesAreFifoEach();
	TestBackRunsAfterFront();
	TestBackJobsAreNotStarved();
	TestPauseHoldsBothLanes();
	TestStopDropsBothLanes();
	TestWarmSetLaneOrder();
	TestInlineClaimOfBackJobRunsOnce();
	std::puts("Pipeline build queue tests passed");
}
