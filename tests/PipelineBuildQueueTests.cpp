#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"
#include "graphics/host_gpu/renderer/pipeline/warmBuildSet.h"

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
	TestWarmLookupAdoptsRunningBuild();
	TestWarmLookupClaimsQueuedBuild();
	TestWarmStartRefusals();
	TestWarmStopWithPendingBuilds();
	TestWarmStopWithNothingPending();
	TestWarmIsDone();
	TestWarmUpWorkerCount();
	std::puts("Pipeline build queue tests passed");
}
