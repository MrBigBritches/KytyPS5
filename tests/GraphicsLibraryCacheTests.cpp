// Tests the cache that shares graphics pipeline libraries. No device is needed, since the libraries are
// made by functions the tests give it, which count what they make and hand out fake handles.

#include "PipelinePartsFixtures.h"
#include "common/gpuStall.h"
#include "common/threadPriority.h"
#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace Libs::Graphics;
using namespace PartFixtures;

namespace {

// How long a fake build takes in the thread tests. A second build of the same library needs a window
// to start in, and the window is this long.
constexpr std::chrono::microseconds kSlowBuild(3000);
constexpr unsigned kThreadCount       = 16;
constexpr unsigned kPipelinesPerThread = 24;
// The distinct values each part key takes across the pipelines of the thread test.
constexpr unsigned kDistinctVertexShaders = 5;
constexpr unsigned kDistinctPixelShaders  = 7;
constexpr unsigned kDistinctBlends        = 3;
// How long a test waits for something that must happen before it calls it stuck.
constexpr std::chrono::seconds kStuck(5);

// Stops the test with a message when a check fails.
void Check(bool passed, const std::string& what) {
	if (!passed) {
		std::fprintf(stderr, "GraphicsLibraryCacheTests: failed: %s\n", what.c_str());
		std::abort();
	}
}

// The number a fake handle stands for.
uint64_t Number(vk::Pipeline pipeline) {
	return reinterpret_cast<uint64_t>(static_cast<VkPipeline>(pipeline));
}

// Stands in for the driver. Every build makes a new numbered pipeline, and the builds are counted by part.
struct FakeDriver {
	std::atomic<uint64_t>                                next {1};
	std::array<std::atomic<uint32_t>, LIBRARY_PART_COUNT> builds {};
	std::chrono::microseconds                            delay {0};
	std::mutex                                           mutex;
	std::vector<uint64_t>                                destroyed;

	LibraryHandles Build(LibraryPart part) {
		builds[static_cast<std::size_t>(part)]++;
		if (delay.count() != 0) {
			std::this_thread::sleep_for(delay);
		}
		const auto number = next++;
		return {vk::Pipeline(reinterpret_cast<VkPipeline>(number)), nullptr};
	}

	[[nodiscard]] uint32_t Builds(LibraryPart part) const {
		return builds[static_cast<std::size_t>(part)].load();
	}

	void Destroy(const LibraryHandles& handles) {
		std::scoped_lock lock(mutex);
		destroyed.push_back(Number(handles.pipeline));
	}
};

GraphicsLibraryCache MakeCache(FakeDriver& driver) {
	return GraphicsLibraryCache([&](const LibraryHandles& handles) { driver.Destroy(handles); });
}

// Asks for the four libraries of a pipeline, as the library path does, and returns what it got.
std::array<LibraryHandles, LIBRARY_PART_COUNT> GetAll(GraphicsLibraryCache& cache, FakeDriver& driver,
                                                      Pipeline& pipeline) {
	const auto stages = pipeline.Stages();
	return {
	    cache.GetOrBuild(MakeVertexInputKey(pipeline.key, stages),
	                     [&] { return driver.Build(LibraryPart::VertexInput); }),
	    cache.GetOrBuild(MakePreRasterKey(pipeline.key, stages),
	                     [&] { return driver.Build(LibraryPart::PreRaster); }),
	    cache.GetOrBuild(MakeFragmentKey(pipeline.key, stages),
	                     [&] { return driver.Build(LibraryPart::Fragment); }),
	    cache.GetOrBuild(MakeFragmentOutputKey(pipeline.key, stages),
	                     [&] { return driver.Build(LibraryPart::Output); }),
	};
}

// A pipeline that differs from the baseline in its vertex shader only.
Pipeline OtherVertexShader(uint64_t id) {
	Pipeline pipeline;
	FillBaseline(pipeline);
	pipeline.key.stage_ids[VERTEX_STAGE_INDEX] = id;
	pipeline.vertex.id                         = id;
	return pipeline;
}

// A pipeline that differs from the baseline in its pixel shader and one blend factor.
Pipeline OtherPixelShaderAndBlend(uint64_t id) {
	Pipeline pipeline;
	FillBaseline(pipeline);
	pipeline.key.stage_ids[PIXEL_STAGE_INDEX]  = id;
	pipeline.pixel.id                          = id;
	pipeline.key.static_params.color_srcblend[0] = 7;
	return pipeline;
}

// Two pipelines with the same vertex shader share one pre-rasterization library, and the baseline is
// two vertex shaders, which make two. Every other library follows its own key too. What else could
// pass the shared case is a cache that shares everything, and the baseline pipeline C shows it does
// not, since it has a library of its own for the vertex shader and shares the rest.
void TestDedup() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	Pipeline   baseline;
	FillBaseline(baseline);
	auto same_vertex_shader = OtherPixelShaderAndBlend(0x7777);
	auto other_vertex       = OtherVertexShader(0x8888);

	const auto first = GetAll(cache, driver, baseline);
	const auto second = GetAll(cache, driver, same_vertex_shader);
	Check(driver.Builds(LibraryPart::PreRaster) == 1 && first[1].pipeline == second[1].pipeline,
	      "two pipelines with one vertex shader did not share a pre-rasterization library");
	Check(driver.Builds(LibraryPart::Fragment) == 2 && first[2].pipeline != second[2].pipeline,
	      "two pixel shaders shared a fragment library");
	Check(driver.Builds(LibraryPart::Output) == 2 && first[3].pipeline != second[3].pipeline,
	      "two blends shared an output library");
	Check(driver.Builds(LibraryPart::VertexInput) == 1 && first[0].pipeline == second[0].pipeline,
	      "pipelines with the same vertex input did not share a library");

	const auto third = GetAll(cache, driver, other_vertex);
	Check(driver.Builds(LibraryPart::PreRaster) == 2 && third[1].pipeline != first[1].pipeline,
	      "two vertex shaders shared a pre-rasterization library");
	Check(third[0].pipeline == first[0].pipeline && third[2].pipeline == first[2].pipeline &&
	          third[3].pipeline == first[3].pipeline,
	      "a vertex shader change split a library that does not read the vertex shader");
	Check(cache.Size(LibraryPart::PreRaster) == 2 && cache.Size(LibraryPart::Fragment) == 2 &&
	          cache.Size(LibraryPart::Output) == 2 && cache.Size(LibraryPart::VertexInput) == 1,
	      "the cache holds another number of libraries than were built");
	// Hits are the requests an existing library answered. Three pipelines asked for four libraries each,
	// and seven were built (1 vertex input, 2 of each other part), so five requests were hits.
	const auto hits = cache.Counts(LibraryPart::PreRaster).hits + cache.Counts(LibraryPart::Fragment).hits +
	                  cache.Counts(LibraryPart::Output).hits + cache.Counts(LibraryPart::VertexInput).hits;
	Check(hits == 5, "the hits do not match the requests an existing library answered");
	std::printf("GraphicsLibraryCacheTests: dedup ok\n");
}

// The pipeline a thread asks for at a step of its walk. The threads walk the same shaders from
// different starting points, so they meet on the same libraries.
Pipeline ConcurrentPipeline(unsigned step) {
	Pipeline pipeline;
	FillBaseline(pipeline);
	const uint64_t vertex_id                   = 0x100 + step % kDistinctVertexShaders;
	pipeline.key.stage_ids[VERTEX_STAGE_INDEX] = vertex_id;
	pipeline.vertex.id                         = vertex_id;
	const uint64_t pixel_id                    = 0x200 + step % kDistinctPixelShaders;
	pipeline.key.stage_ids[PIXEL_STAGE_INDEX]  = pixel_id;
	pipeline.pixel.id                          = pixel_id;
	pipeline.key.static_params.color_srcblend[0] = static_cast<uint8_t>(step % kDistinctBlends);
	return pipeline;
}

// What the threads of the concurrency test saw of the libraries they were given.
struct ThreadSightings {
	std::mutex                              mutex;
	std::set<std::pair<unsigned, uint64_t>> pre_raster;
	std::set<uint64_t>                      handles;
	std::atomic<unsigned>                   missing {0};
};

// One thread's walk. Each library handle it gets is kept, so the test can tell whether two threads
// were given different libraries for the same key.
void WalkConcurrentPipelines(GraphicsLibraryCache& cache, FakeDriver& driver, unsigned thread,
                             ThreadSightings& sightings) {
	for (unsigned index = 0; index < kPipelinesPerThread; index++) {
		auto       pipeline  = ConcurrentPipeline(thread + index);
		const auto libraries = GetAll(cache, driver, pipeline);
		std::scoped_lock lock(sightings.mutex);
		for (const auto& library: libraries) {
			sightings.missing += library.pipeline == nullptr ? 1u : 0u;
			sightings.handles.insert(Number(library.pipeline));
		}
		sightings.pre_raster.insert({static_cast<unsigned>(pipeline.vertex.id), Number(libraries[1].pipeline)});
	}
}

// Many threads build pipelines that share libraries, with a slow build so that threads meet on the
// same library. Each distinct library is built exactly once, and every thread gets the handle that
// one build made. Counting alone could pass if a build raced and the second result was dropped, so the
// handles are compared too, and the fake driver numbers every build.
void TestConcurrentBuilds() {
	FakeDriver driver;
	driver.delay = kSlowBuild;
	auto                     cache = MakeCache(driver);
	ThreadSightings          sightings;
	std::vector<std::thread> threads;
	for (unsigned thread = 0; thread < kThreadCount; thread++) {
		threads.emplace_back([&, thread] { WalkConcurrentPipelines(cache, driver, thread, sightings); });
	}
	for (auto& thread: threads) thread.join();
	Check(sightings.missing == 0, "a thread got no library");
	Check(driver.Builds(LibraryPart::PreRaster) == kDistinctVertexShaders,
	      "a pre-rasterization library was built more or fewer times than it has keys");
	Check(driver.Builds(LibraryPart::Fragment) == kDistinctPixelShaders,
	      "a fragment library was built more or fewer times than it has keys");
	Check(driver.Builds(LibraryPart::Output) == kDistinctBlends,
	      "an output library was built more or fewer times than it has keys");
	Check(driver.Builds(LibraryPart::VertexInput) == 1, "the vertex input library was built twice");
	Check(sightings.pre_raster.size() == kDistinctVertexShaders,
	      "a vertex shader was given two different pre-rasterization libraries");
	const unsigned built = kDistinctVertexShaders + kDistinctPixelShaders + kDistinctBlends + 1;
	Check(sightings.handles.size() == built, "the threads saw another number of libraries than were built");
	const auto requests = kThreadCount * kPipelinesPerThread * LIBRARY_PART_COUNT;
	uint64_t   total_built = 0, total_hits = 0;
	for (const auto part: {LibraryPart::VertexInput, LibraryPart::PreRaster, LibraryPart::Fragment,
	                       LibraryPart::Output}) {
		total_built += cache.Counts(part).built;
		total_hits += cache.Counts(part).hits;
	}
	Check(total_built == built && total_built + total_hits == requests,
	      "built and hits do not add up to the requests");
	std::printf("GraphicsLibraryCacheTests: threads ok\n");
}

// Waits for a flag with a limit, so a test that hangs fails.
bool WaitFor(const std::atomic<bool>& flag) {
	const auto limit = std::chrono::steady_clock::now() + kStuck;
	while (!flag.load() && std::chrono::steady_clock::now() < limit) {
		std::this_thread::yield();
	}
	return flag.load();
}

// A thread that needs a library another thread is making waits for that build and does not make a
// second one. A thread that needs another library does not wait at all, so builds run side by side
// and the cache never holds a lock while one runs. The first thread is held inside its build, which
// is how a long compile looks to the others.
void TestWaitsForRunningBuildOnly() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	Pipeline   slow, fast;
	FillBaseline(slow);
	fast = OtherVertexShader(0x9999);
	std::atomic<bool> slow_started {false}, release {false}, fast_done {false}, waiter_done {false};
	const auto        slow_key = MakePreRasterKey(slow.key, slow.Stages());
	const auto        fast_key = MakePreRasterKey(fast.key, fast.Stages());

	std::thread builder([&] {
		(void)cache.GetOrBuild(slow_key, [&] {
			slow_started = true;
			while (!release.load()) std::this_thread::yield();
			return driver.Build(LibraryPart::PreRaster);
		});
	});
	Check(WaitFor(slow_started), "the slow build never started");
	std::thread other([&] {
		(void)cache.GetOrBuild(fast_key, [&] { return driver.Build(LibraryPart::PreRaster); });
		fast_done = true;
	});
	// The other library is finished while the first build is still held.
	Check(WaitFor(fast_done), "a build of another library waited for a running build");
	Check(driver.Builds(LibraryPart::PreRaster) == 1, "the other library was not built alone");
	LibraryHandles waited;
	std::atomic<bool> waiter_calling {false};
	std::thread       waiter([&] {
        waiter_calling = true;
        waited         = cache.GetOrBuild(slow_key, [&] { return driver.Build(LibraryPart::PreRaster); });
        waiter_done    = true;
    });
	// The waiter is about to ask. The sleep gives it time to reach the wait, and it must still be
	// waiting, with nothing built for it.
	Check(WaitFor(waiter_calling), "the waiter never started");
	std::this_thread::sleep_for(std::chrono::milliseconds(50));
	Check(!waiter_done.load(), "a thread did not wait for the library being built");
	Check(driver.Builds(LibraryPart::PreRaster) == 1, "a waiting thread built the library itself");
	release = true;
	builder.join();
	other.join();
	waiter.join();
	Check(waiter_done.load() && waited.pipeline != nullptr &&
	          driver.Builds(LibraryPart::PreRaster) == 2,
	      "the waiting thread did not get the finished library");
	const auto again = cache.GetOrBuild(slow_key, [&] { return driver.Build(LibraryPart::PreRaster); });
	Check(again.pipeline == waited.pipeline, "the waiter and a later request got different libraries");
	std::printf("GraphicsLibraryCacheTests: wait ok\n");
}

// The pipeline row of a thread that builds a library holds the build as built time, and the row of a thread
// that waits for it holds the wait as waited time and builds nothing. The build is held for a while, so
// both figures are well above zero. What else could pass is a row that files the wait as a build, which
// shows as a built count on the waiter.
void TestRowsSeparateBuildAndWait() {
	using Common::FirstRunReport::LibraryKind;
	using Common::FirstRunReport::LibraryPipelineScope;
	// The build is held long enough that a waiter which starts a little late on a slow machine still
	// waits for most of it. The waiter is checked against half of the shorter bound below.
	constexpr auto   hold     = std::chrono::milliseconds(200);
	constexpr double at_least = 100.0;
	constexpr double waiter_share = at_least / 2;
	Common::FirstRunReport::Configure(true, false);
	Common::FirstRunReport::Global().Reset();
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	Pipeline   pipeline;
	FillBaseline(pipeline);
	const auto        key = MakePreRasterKey(pipeline.key, pipeline.Stages());
	std::atomic<bool> building {false};
	std::thread       builder([&] {
        LibraryPipelineScope scope(1, 1, false);
        (void)cache.GetOrBuild(key, [&] {
            building = true;
            std::this_thread::sleep_for(hold);
            return driver.Build(LibraryPart::PreRaster);
        });
        scope.Commit();
    });
	Check(WaitFor(building), "the build never started");
	std::thread waiter([&] {
		LibraryPipelineScope scope(2, 2, false);
		(void)cache.GetOrBuild(key, [&] { return driver.Build(LibraryPart::PreRaster); });
		scope.Commit();
	});
	builder.join();
	waiter.join();
	const auto rows = Common::FirstRunReport::Global().LibraryPipelines();
	Common::FirstRunReport::Configure(false, false);
	Check(rows.size() == 2, "there was not a row for each thread");
	const auto& built  = rows[0].vs == 1 ? rows[0] : rows[1];
	const auto& waited = rows[0].vs == 1 ? rows[1] : rows[0];
	Check(built.built_count == 1 && built.built_ms >= at_least && built.waited_ms == 0.0,
	      "the thread that built did not get the time as built time");
	Check(waited.built_count == 0 && waited.built_ms == 0.0 && waited.waited_ms >= waiter_share,
	      "the thread that waited got the wait as built time, or no wait");
	std::printf("GraphicsLibraryCacheTests: rows ok\n");
}

// A refused library is kept, so the key is not tried again, and the pipeline that needed it is built
// whole. The baseline is a library that was made, which is built once and counted as built.
void TestRefusalIsKept() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	Pipeline   refused, made;
	FillBaseline(refused);
	made = OtherVertexShader(0xaaaa);
	std::atomic<uint32_t> attempts {0};
	const auto refuse = [&] {
		attempts++;
		return LibraryHandles {};
	};
	const auto key = MakePreRasterKey(refused.key, refused.Stages());
	Check(cache.GetOrBuild(key, refuse).pipeline == nullptr, "a refused library had a handle");
	Check(cache.GetOrBuild(key, refuse).pipeline == nullptr, "a refused library got a handle later");
	Check(attempts == 1, "a refused library was tried again");
	const auto counts = cache.Counts(LibraryPart::PreRaster);
	Check(counts.failed == 1 && counts.built == 0 && counts.hits == 0 && counts.refused_hits == 1,
	      "a refused library was counted as built, or its second request was not a refused hit");
	const auto good = cache.GetOrBuild(MakePreRasterKey(made.key, made.Stages()),
	                                   [&] { return driver.Build(LibraryPart::PreRaster); });
	Check(good.pipeline != nullptr && cache.Counts(LibraryPart::PreRaster).built == 1,
	      "a library that was made was not counted as built");
	std::printf("GraphicsLibraryCacheTests: refusal ok\n");
}

// The cache destroys each library it holds once, when it is destroyed, and a refused library has
// nothing to destroy. Nothing is destroyed earlier, since a linked pipeline needs its libraries.
void TestDestroyOnce() {
	FakeDriver driver;
	std::set<uint64_t> made;
	{
		auto     cache = MakeCache(driver);
		Pipeline a, b, c;
		FillBaseline(a);
		b = OtherVertexShader(0xbbbb);
		c = OtherPixelShaderAndBlend(0xcccc);
		for (auto* pipeline: {&a, &b, &c}) {
			for (const auto& library: GetAll(cache, driver, *pipeline)) made.insert(Number(library.pipeline));
		}
		(void)cache.GetOrBuild(MakePreRasterKey(OtherVertexShader(0xdddd).key, a.Stages()),
		                       [] { return LibraryHandles {}; });
		Check(driver.destroyed.empty(), "a library was destroyed before the cache was");
	}
	std::set<uint64_t> destroyed(driver.destroyed.begin(), driver.destroyed.end());
	Check(driver.destroyed.size() == destroyed.size(), "a library was destroyed twice");
	Check(destroyed == made, "the destroyed libraries are not the ones that were made");
	std::printf("GraphicsLibraryCacheTests: destroy ok\n");
}

// The counters of links and fallbacks start at zero and count only their own event.
void TestLinkAndFallbackCounts() {
	using Common::FirstRunReport::FallbackKind;
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	Check(cache.Links() == 0 && cache.Fallbacks(FallbackKind::Mesh) == 0, "a new cache counted something");
	cache.NoteLink(0.5);
	cache.NoteLink(0.5);
	Check(cache.LiveLinks() == 2, "two links were not live");
	cache.NoteUnlinked();
	cache.NoteUnlinked();
	Check(cache.LiveLinks() == 0 && cache.Links() == 2, "destroying pipelines changed the link count");
	cache.NoteFallback(FallbackKind::Tess);
	Check(cache.Links() == 2 && cache.Fallbacks(FallbackKind::Tess) == 1 &&
	          cache.Fallbacks(FallbackKind::CreateFailed) == 0 && cache.Fallbacks(FallbackKind::Mesh) == 0 &&
	          cache.Fallbacks(FallbackKind::RectList) == 0,
	      "a link or fallback was counted under another count");
	std::printf("GraphicsLibraryCacheTests: counts ok\n");
}

// A fragment key for the prebuilt tests, of a pixel shader id with a sample count.
FragmentKey PrebuiltKey(uint64_t pixel_id, uint32_t samples = 1) {
	FragmentKey key;
	key.pixel_id = pixel_id;
	key.samples  = samples;
	return key;
}

// A prebuilt fragment library answers the first ordinary request for its key, which is counted once
// as a first hit, and is not built again. The baseline is the same two requests after an ordinary
// build, which count no first hit, so only the prebuild's mark makes the count.
void TestPrebuiltFirstHit() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	const auto build = [&] { return driver.Build(LibraryPart::Fragment); };
	Check(cache.GetOrBuildPrebuilt(PrebuiltKey(5), build), "the prebuild did not build");
	Check(!cache.GetOrBuildPrebuilt(PrebuiltKey(5), build), "a second prebuild built again");
	(void)cache.GetOrBuild(PrebuiltKey(5), build);
	(void)cache.GetOrBuild(PrebuiltKey(5), build);
	auto counts = cache.PrebuiltCounts();
	Check(driver.Builds(LibraryPart::Fragment) == 1, "a prebuilt library was built again");
	Check(counts.built == 1 && counts.already_there == 1 && counts.first_hits == 1 && counts.waits == 0,
	      "the prebuilt library's first hit was not counted once");

	FakeDriver plain_driver;
	auto       plain = MakeCache(plain_driver);
	const auto plain_build = [&] { return plain_driver.Build(LibraryPart::Fragment); };
	(void)plain.GetOrBuild(PrebuiltKey(5), plain_build);
	(void)plain.GetOrBuild(PrebuiltKey(5), plain_build);
	counts = plain.PrebuiltCounts();
	Check(counts.first_hits == 0 && counts.built == 0, "an ordinary library counted as prebuilt");
	std::printf("GraphicsLibraryCacheTests: prebuilt first hit ok\n");
}

// An ordinary request that meets a prebuild still running waits for it in the entry's once flag, and
// is counted as a wait with its time, not as a first hit. The prebuild is held on a promise until the
// request has had time to arrive, and the request's own build never runs.
void TestPrebuiltWaitIsCounted() {
	FakeDriver         driver;
	auto               cache = MakeCache(driver);
	std::promise<void> release;
	auto               released = release.get_future().share();
	std::atomic<bool>  started {false};
	std::thread prebuild([&] {
		(void)cache.GetOrBuildPrebuilt(PrebuiltKey(9), [&] {
			started = true;
			released.wait();
			return driver.Build(LibraryPart::Fragment);
		});
	});
	const auto deadline = std::chrono::steady_clock::now() + kStuck;
	while (!started && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::yield();
	}
	Check(started, "the prebuild never started");
	std::thread request([&] { (void)cache.GetOrBuild(PrebuiltKey(9), [&] { return driver.Build(LibraryPart::Fragment); }); });
	// Long enough for the request to reach the once flag on a loaded machine, so it really waits.
	constexpr std::chrono::milliseconds kRequestArrival(100);
	std::this_thread::sleep_for(kRequestArrival);
	release.set_value();
	prebuild.join();
	request.join();
	const auto counts = cache.PrebuiltCounts();
	Check(driver.Builds(LibraryPart::Fragment) == 1, "the waiting request built the library itself");
	Check(counts.waits == 1 && counts.first_hits == 0 && counts.wait_ms > 0.0 && counts.wait_max_ms > 0.0,
	      "a request that waited on a prebuild was not counted as a wait");
	std::printf("GraphicsLibraryCacheTests: prebuilt wait ok\n");
}

// What the priority hooks of the boost tests saw.
struct BoostLog {
	std::atomic<uint64_t> next_thread {100};
	std::atomic<uint64_t> builder {0};
	std::atomic<uint64_t> boosted_thread {0};
	std::atomic<uint32_t> boosts {0};
	std::atomic<uint32_t> lowers {0};
};

// Hooks that give each thread that asks a made-up id and record boosts and drops, so no thread's
// real priority changes.
GraphicsLibraryCache::PriorityHooks LoggingHooks(BoostLog& log) {
	return {.current_thread = [&log] { return log.next_thread++; },
	        .boost          = [&log](uint64_t thread) {
		        log.boosted_thread = thread;
		        log.boosts++;
		        return true;
	        },
	        .restore = [&log](int) { log.lowers++; }};
}

// Runs a prebuild held on a promise and an ordinary request of the same key, and releases the prebuild
// once the request has had time to reach the once flag. With `request_first` false the prebuild is
// released and joined before the request, so the request meets a finished entry.
void RunPrebuildAndRequest(GraphicsLibraryCache& cache, FakeDriver& driver, BoostLog& log, bool request_first) {
	std::promise<void> release;
	auto               released = release.get_future().share();
	std::atomic<bool>  started {false};
	std::thread prebuild([&] {
		(void)cache.GetOrBuildPrebuilt(PrebuiltKey(11), [&] {
			log.builder = log.next_thread.load() - 1;
			started     = true;
			released.wait();
			return driver.Build(LibraryPart::Fragment);
		});
	});
	const auto deadline = std::chrono::steady_clock::now() + kStuck;
	while (!started && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::yield();
	}
	Check(started, "the prebuild never started");
	const auto request_once = [&] { (void)cache.GetOrBuild(PrebuiltKey(11), [&] { return driver.Build(LibraryPart::Fragment); }); };
	if (!request_first) {
		release.set_value();
		prebuild.join();
		request_once();
		return;
	}
	std::thread request(request_once);
	constexpr std::chrono::milliseconds kRequestArrival(100);
	std::this_thread::sleep_for(kRequestArrival);
	release.set_value();
	prebuild.join();
	request.join();
}

// An ordinary request that meets a prebuild still running raises the prebuild's thread before it
// waits, once, and the prebuild drops back when its build is over. The baseline request comes after
// the prebuild finished and boosts nothing, so only waiting on a running prebuild makes the boost.
void TestWaitingRequestBoostsPrebuild() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	BoostLog   log;
	cache.SetPriorityHooksForTest(LoggingHooks(log));
	RunPrebuildAndRequest(cache, driver, log, true);
	Check(log.boosts == 1 && log.boosted_thread == log.builder && log.lowers == 1,
	      "a request waiting on a running prebuild did not boost its thread once, or it never dropped back");
	Check(cache.PrebuiltCounts().boosts == 1 && cache.PrebuiltCounts().waits == 1,
	      "the boost or the wait was not counted");
	Check(driver.Builds(LibraryPart::Fragment) == 1, "the waiting request built its own copy");

	FakeDriver after_driver;
	auto       after = MakeCache(after_driver);
	BoostLog   after_log;
	after.SetPriorityHooksForTest(LoggingHooks(after_log));
	RunPrebuildAndRequest(after, after_driver, after_log, false);
	Check(after_log.boosts == 0 && after_log.lowers == 0 && after.PrebuiltCounts().boosts == 0,
	      "a request after the prebuild finished boosted it");
	std::printf("GraphicsLibraryCacheTests: prebuilt boost ok\n");
}

// With three prebuilds held and a request waiting on each, only MAX_CONCURRENT_BOOSTS of them are
// boosted, the third waiter is refused and waits unboosted, and the peak is the cap. The single-wait
// test above is the baseline, where the one waiter is boosted, so the refusal comes from the cap.
void TestBoostsAreCapped() {
	constexpr uint32_t kPrebuilds = 3;
	FakeDriver         driver;
	auto               cache = MakeCache(driver);
	BoostLog           log;
	cache.SetPriorityHooksForTest(LoggingHooks(log));
	std::promise<void>       release;
	auto                     released = release.get_future().share();
	std::atomic<uint32_t>    started {0};
	std::vector<std::thread> threads;
	for (uint32_t index = 0; index < kPrebuilds; ++index) {
		threads.emplace_back([&, index] {
			(void)cache.GetOrBuildPrebuilt(PrebuiltKey(30 + index), [&] {
				started++;
				released.wait();
				return driver.Build(LibraryPart::Fragment);
			});
		});
	}
	const auto deadline = std::chrono::steady_clock::now() + kStuck;
	while (started < kPrebuilds && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::yield();
	}
	Check(started == kPrebuilds, "the prebuilds never started");
	for (uint32_t index = 0; index < kPrebuilds; ++index) {
		threads.emplace_back([&, index] {
			(void)cache.GetOrBuild(PrebuiltKey(30 + index), [&] { return driver.Build(LibraryPart::Fragment); });
		});
	}
	constexpr std::chrono::milliseconds kRequestArrival(100);
	std::this_thread::sleep_for(kRequestArrival);
	release.set_value();
	for (auto& thread: threads) {
		thread.join();
	}
	const auto counts = cache.PrebuiltCounts();
	Check(counts.boosts == MAX_CONCURRENT_BOOSTS && counts.boosts_refused_cap == 1 &&
	          counts.boosts_peak == MAX_CONCURRENT_BOOSTS && log.lowers == MAX_CONCURRENT_BOOSTS,
	      "more prebuilds were boosted at once than the cap allows");
	Check(driver.Builds(LibraryPart::Fragment) == kPrebuilds, "a waiter built its own copy");
	std::printf("GraphicsLibraryCacheTests: boost cap ok\n");
}

// What a cache with every boost slot taken by two ordinary waiters did for a third waiter, which is a
// deferred build worker when `blocking`, and how many boosts went to the waiter's own priority.
struct CapFullOutcome {
	PrebuiltLibraryCounts counts;
	uint32_t              boost_tos = 0;
	uint32_t              lowers    = 0;
};

CapFullOutcome RunThirdWaiterWithCapFull(bool blocking) {
	constexpr uint32_t kPrebuilds  = 3;
	constexpr uint32_t kFirstKey   = 40;
	FakeDriver         driver;
	auto               cache = MakeCache(driver);
	BoostLog           log;
	std::atomic<uint32_t> boost_tos {0};
	auto                  hooks = LoggingHooks(log);
	hooks.current_priority      = [] { return 0; };
	hooks.boost_to              = [&](uint64_t thread, int) {
		log.boosted_thread = thread;
		log.boosts++;
		boost_tos++;
		return true;
	};
	cache.SetPriorityHooksForTest(hooks);
	std::promise<void>       release;
	auto                     released = release.get_future().share();
	std::atomic<uint32_t>    started {0};
	std::vector<std::thread> threads;
	for (uint32_t index = 0; index < kPrebuilds; ++index) {
		threads.emplace_back([&, index] {
			(void)cache.GetOrBuildPrebuilt(PrebuiltKey(kFirstKey + index), [&] {
				started++;
				released.wait();
				return driver.Build(LibraryPart::Fragment);
			});
		});
	}
	const auto wait_for = [](const auto& done) {
		const auto deadline = std::chrono::steady_clock::now() + kStuck;
		while (!done() && std::chrono::steady_clock::now() < deadline) {
			std::this_thread::yield();
		}
		return done();
	};
	Check(wait_for([&] { return started == kPrebuilds; }), "the prebuilds never started");
	const auto request = [&](uint32_t index, bool as_deferred) {
		return std::thread([&, index, as_deferred] {
			if (as_deferred) {
				Common::GpuStall::SetThreadRole(Common::GpuStall::ThreadRole::DeferredBuild);
			}
			(void)cache.GetOrBuild(PrebuiltKey(kFirstKey + index), [&] { return driver.Build(LibraryPart::Fragment); });
		});
	};
	threads.push_back(request(0, false));
	threads.push_back(request(1, false));
	Check(wait_for([&] { return log.boosts == MAX_CONCURRENT_BOOSTS; }), "the ordinary waiters did not fill the cap");
	threads.push_back(request(2, blocking));
	constexpr std::chrono::milliseconds kRequestArrival(100);
	std::this_thread::sleep_for(kRequestArrival);
	release.set_value();
	for (auto& thread: threads) {
		thread.join();
	}
	return {cache.PrebuiltCounts(), boost_tos.load(), log.lowers.load()};
}

// A deferred build worker that waits on a running prebuild while every boost slot is taken still
// boosts it, to its own priority, and it drops back at its end. The baseline's third waiter is an
// ordinary one, which the cap refuses, so the boost past the cap comes from the waiter blocking a draw.
void TestBlockingWaiterBoostsPastCap() {
	const auto blocking = RunThirdWaiterWithCapFull(true);
	Check(blocking.counts.boosts == MAX_CONCURRENT_BOOSTS + 1 && blocking.counts.boosts_blocking == 1 &&
	          blocking.boost_tos == 1 && blocking.counts.boosts_refused_cap == 0 &&
	          blocking.counts.boosts_peak == MAX_CONCURRENT_BOOSTS + 1 && blocking.lowers == MAX_CONCURRENT_BOOSTS + 1,
	      "a deferred build worker waiting with the cap full did not boost the prebuild to its own priority");
	const auto ordinary = RunThirdWaiterWithCapFull(false);
	Check(ordinary.counts.boosts == MAX_CONCURRENT_BOOSTS && ordinary.counts.boosts_refused_cap == 1 &&
	          ordinary.boost_tos == 0 && ordinary.counts.boosts_blocking == 0,
	      "an ordinary waiter with the cap full was not refused");
	std::printf("GraphicsLibraryCacheTests: blocking boost ok\n");
}

// A prebuilt pre-rasterization library answers the first ordinary request for its key, counted once
// as a first hit of that part and not of the fragment part, and is not built again. The baseline is an
// ordinary build of another vertex shader's key, which counts no first hit.
void TestPreRasterPrebuiltFirstHit() {
	FakeDriver   driver;
	auto         cache = MakeCache(driver);
	const auto   build = [&] { return driver.Build(LibraryPart::PreRaster); };
	PreRasterKey prebuilt_key;
	prebuilt_key.stage_ids[0] = 7;
	PreRasterKey plain_key;
	plain_key.stage_ids[0] = 8;
	Check(cache.GetOrBuildPrebuilt(prebuilt_key, build), "the pre-raster prebuild did not build");
	(void)cache.GetOrBuild(prebuilt_key, build);
	(void)cache.GetOrBuild(prebuilt_key, build);
	(void)cache.GetOrBuild(plain_key, build);
	(void)cache.GetOrBuild(plain_key, build);
	const auto counts = cache.PrebuiltCounts(LibraryPart::PreRaster);
	Check(driver.Builds(LibraryPart::PreRaster) == 2, "a prebuilt pre-raster library was built again");
	Check(counts.built == 1 && counts.first_hits == 1 && counts.waits == 0,
	      "the prebuilt pre-raster library's first hit was not counted once");
	Check(cache.PrebuiltCounts(LibraryPart::Fragment).first_hits == 0, "a pre-raster hit was counted as fragment");
	std::printf("GraphicsLibraryCacheTests: pre-raster prebuilt first hit ok\n");
}

// An ordinary request that builds a fragment library for a prebuilt pixel shader under another key,
// here another sample count, is counted. The baseline builds one for a pixel shader that was never
// prebuilt, which counts nothing.
void TestPrebuiltKeyDiffers() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	const auto build = [&] { return driver.Build(LibraryPart::Fragment); };
	(void)cache.GetOrBuildPrebuilt(PrebuiltKey(5), build);
	(void)cache.GetOrBuild(PrebuiltKey(6, 4), build);
	Check(cache.PrebuiltCounts().key_differs == 0, "a pixel shader never prebuilt counted as differing");
	(void)cache.GetOrBuild(PrebuiltKey(5, 4), build);
	Check(cache.PrebuiltCounts().key_differs == 1, "a prebuilt pixel shader under another key was not counted");
	std::printf("GraphicsLibraryCacheTests: prebuilt key differs ok\n");
}

// How long a held prebuild waits before it is let go, so an ordinary request started just before has
// reached the entry's once flag even on a loaded machine.
constexpr std::chrono::milliseconds kHeldPrebuildArrival(100);

// Holds a prebuild of `key` until an ordinary request for the same key has had time to reach the
// entry's once flag, so the request waits on the prebuild, then lets both finish.
void WaitOnHeldPrebuild(GraphicsLibraryCache& cache, FakeDriver& driver, const FragmentKey& key) {
	std::promise<void> release;
	auto               released = release.get_future().share();
	std::atomic<bool>  started {false};
	std::thread prebuild([&] {
		(void)cache.GetOrBuildPrebuilt(key, [&] {
			started = true;
			released.wait();
			return driver.Build(LibraryPart::Fragment);
		});
	});
	Check(WaitFor(started), "the held prebuild never started");
	std::thread request([&] { (void)cache.GetOrBuild(key, [&] { return driver.Build(LibraryPart::Fragment); }); });
	std::this_thread::sleep_for(kHeldPrebuildArrival);
	release.set_value();
	prebuild.join();
	request.join();
}

// A request that waited on a prebuild claims its entry, so a later request that finds it finished is
// not counted as a first hit. The baseline is the same later request on a prebuild nobody waited on,
// which is counted as the first hit, so only the wait's claim keeps the count at zero.
void TestWaitClaimsThePrebuilt() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	WaitOnHeldPrebuild(cache, driver, PrebuiltKey(21));
	(void)cache.GetOrBuild(PrebuiltKey(21), [&] { return driver.Build(LibraryPart::Fragment); });
	const auto counts = cache.PrebuiltCounts();
	Check(counts.waits == 1 && counts.first_hits == 0, "a request after a wait on a prebuild counted as its first hit");

	FakeDriver plain_driver;
	auto       plain = MakeCache(plain_driver);
	const auto plain_build = [&] { return plain_driver.Build(LibraryPart::Fragment); };
	(void)plain.GetOrBuildPrebuilt(PrebuiltKey(21), plain_build);
	(void)plain.GetOrBuild(PrebuiltKey(21), plain_build);
	Check(plain.PrebuiltCounts().first_hits == 1, "a request on a prebuild nobody waited on was not its first hit");
	std::printf("GraphicsLibraryCacheTests: wait claims the prebuilt ok\n");
}

// An ordinary request that builds a pre-rasterization library for a prebuilt vertex shader under
// another key, here another polygon mode, is counted under that part. The baselines are a vertex
// shader that was never prebuilt and a fragment library of a pixel shader that has the prebuilt
// vertex shader's id, neither of which counts, so only the prebuilt shader's id within its own part
// makes the count.
void TestPreRasterKeyDiffers() {
	constexpr uint64_t kPrebuiltVertex = 7;
	constexpr uint64_t kOtherVertex    = 8;
	FakeDriver         driver;
	auto               cache        = MakeCache(driver);
	const auto         raster_build = [&] { return driver.Build(LibraryPart::PreRaster); };
	PreRasterKey       prebuilt_key;
	prebuilt_key.stage_ids[0] = kPrebuiltVertex;
	PreRasterKey other_mode   = prebuilt_key;
	other_mode.polygon_mode   = vk::PolygonMode::eLine;
	PreRasterKey other_vertex = other_mode;
	other_vertex.stage_ids[0] = kOtherVertex;
	(void)cache.GetOrBuildPrebuilt(prebuilt_key, raster_build);
	(void)cache.GetOrBuild(other_vertex, raster_build);
	(void)cache.GetOrBuild(PrebuiltKey(kPrebuiltVertex), [&] { return driver.Build(LibraryPart::Fragment); });
	Check(cache.PrebuiltCounts(LibraryPart::PreRaster).key_differs == 0 &&
	          cache.PrebuiltCounts(LibraryPart::Fragment).key_differs == 0,
	      "a vertex shader never prebuilt, or a pixel shader with the same id, counted as differing");
	(void)cache.GetOrBuild(other_mode, raster_build);
	Check(cache.PrebuiltCounts(LibraryPart::PreRaster).key_differs == 1 &&
	          cache.PrebuiltCounts(LibraryPart::Fragment).key_differs == 0,
	      "a prebuilt vertex shader under another key was not counted as a pre-raster key differs");
	std::printf("GraphicsLibraryCacheTests: pre-raster key differs ok\n");
}

// A prebuild is counted among the cache's built libraries like any build, so the library totals the
// report takes from the cache include it, and the request it answers counts as a hit. The request
// builds nothing itself, so only the prebuild can make the built count.
void TestPrebuildCountsAsBuilt() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	const auto build = [&] { return driver.Build(LibraryPart::Fragment); };
	(void)cache.GetOrBuildPrebuilt(PrebuiltKey(23), build);
	(void)cache.GetOrBuild(PrebuiltKey(23), build);
	const auto counts = cache.Counts(LibraryPart::Fragment);
	Check(driver.Builds(LibraryPart::Fragment) == 1 && counts.built == 1 && counts.hits == 1,
	      "a prebuild was not counted as a built library");
	std::printf("GraphicsLibraryCacheTests: prebuild counts as built ok\n");
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// The priorities RunRealBoost read, the prebuild thread's after it lowered itself, while its build ran
// boosted and once the prebuild returned, and the waiter's own.
struct DropBackLevels {
	int lowered = 0;
	int during  = 0;
	int after   = 0;
	int waiter  = 0;
};

// Runs one prebuild with the cache's own priority hooks, on a thread that first lowers itself as the
// graphics prescan's library workers do, and one request that waits on it. The waiter blocks draws
// when `blocking` and lowers itself one ordinary step first when `waiter_lowered`. The build is held
// until the boost is counted, so `during` is read while the thread runs boosted. `lower_builder` is how
// the prebuild's thread lowers itself, the lowest step by default as the prescan's workers do.
using LowerStep = bool (*)(Common::PriorityFailure*);
DropBackLevels RunRealBoost(bool blocking, bool waiter_lowered,
                            LowerStep lower_builder = &Common::LowerCurrentThreadPriorityToLowest) {
	constexpr uint64_t kKey = 50;
	FakeDriver         driver;
	auto               cache = MakeCache(driver);
	DropBackLevels     levels;
	std::promise<void> release;
	auto               released = release.get_future().share();
	std::atomic<bool>  started {false};
	std::thread        prebuild([&] {
		(void)lower_builder(nullptr);
		levels.lowered = Common::CurrentThreadSystemPriority();
		(void)cache.GetOrBuildPrebuilt(PrebuiltKey(kKey), [&] {
			started = true;
			released.wait();
			levels.during = Common::CurrentThreadSystemPriority();
			return driver.Build(LibraryPart::Fragment);
		});
		levels.after = Common::CurrentThreadSystemPriority();
	});
	Check(WaitFor(started), "the prebuild never started");
	std::thread request([&] {
		if (waiter_lowered) {
			(void)Common::LowerCurrentThreadPriority();
		}
		if (blocking) {
			Common::GpuStall::SetThreadRole(Common::GpuStall::ThreadRole::DeferredBuild);
		}
		levels.waiter = Common::CurrentThreadSystemPriority();
		(void)cache.GetOrBuild(PrebuiltKey(kKey), [&] { return driver.Build(LibraryPart::Fragment); });
	});
	const auto deadline = std::chrono::steady_clock::now() + kStuck;
	while (cache.PrebuiltCounts().boosts == 0 && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::yield();
	}
	release.set_value();
	prebuild.join();
	request.join();
	Check(cache.PrebuiltCounts().boosts == 1, "the waiting request never boosted the prebuild");
	return levels;
}

// A boosted prebuild drops back to exactly the level its thread was lowered to, whatever level the
// boost raised it to. An ordinary waiter raises it to normal priority, and a blocking waiter that runs
// one step lowered raises it only to that step, so the two runs boost to different levels. Each run
// also checks that the boost reached the waiter's level, so a thread that never moved cannot pass as
// one that dropped back. Windows only, where the levels are absolute and the boost is accepted. Linux
// works the drop out from the boosted nice value and usually refuses an unprivileged raise, and macOS
// has no boost.
void TestBoostDropsBackToTheLoweredLevel() {
	for (const bool blocking: {false, true}) {
		const auto levels = RunRealBoost(blocking, blocking);
		Check(levels.during == levels.waiter && levels.during != levels.lowered,
		      "the boost did not raise the prebuild to the waiter's level");
		Check(levels.after == levels.lowered, "a boosted prebuild did not drop back to the level it was lowered to");
	}
	std::printf("GraphicsLibraryCacheTests: boost drop-back ok\n");
}

// A boosted prebuild whose thread runs one ordinary step lowered, not at the lowest step, drops back to
// that step. The baseline is the same run with the lowest-step builder, which drops back to lowest, so
// the two runs differ only in the builder's level. A drop that always takes the lowest step passes the
// baseline and lands at lowest here. Each run also checks that the boost moved the thread, so a thread
// that never left its level cannot pass. Windows only, for the reasons TestBoostDropsBackToTheLoweredLevel gives.
void TestBoostRestoresTheBuilderLevel() {
	const auto baseline = RunRealBoost(false, false);
	Check(baseline.lowered == THREAD_PRIORITY_LOWEST && baseline.during != baseline.lowered &&
	          baseline.after == THREAD_PRIORITY_LOWEST,
	      "the lowest-step baseline did not boost and drop back to lowest");
	const auto lowered = RunRealBoost(false, false, &Common::LowerCurrentThreadPriority);
	Check(lowered.lowered == THREAD_PRIORITY_BELOW_NORMAL && lowered.during != lowered.lowered,
	      "the below-normal builder was not boosted");
	Check(lowered.after == lowered.lowered, "a boosted prebuild did not drop back to its below-normal level");
	std::printf("GraphicsLibraryCacheTests: boost restores the builder level ok\n");
}
#endif

// A boost gives back its slot and its underrun tag when the prebuild drops back. One more prebuild than
// MAX_CONCURRENT_BOOSTS is held and waited on in turn, each over before the next starts, so only one
// runs boosted at a time and each should be boosted. A slot kept after the drop refuses the last one.
// The first round reads the tag while its prebuild runs boosted, one above the count before, so a tag
// never noted stays level. After the last round the count is back where it began, so a tag never given
// back, or a drop that takes nothing away, leaves it raised.
void TestBoostGivesBackItsSlotAndTag() {
	constexpr uint64_t kFirstKey = 60;
	FakeDriver         driver;
	auto               cache = MakeCache(driver);
	BoostLog           log;
	cache.SetPriorityHooksForTest(LoggingHooks(log));
	const uint32_t before     = Common::FirstRunReport::PrescanGraphicsBoosted();
	bool           tag_raised = false;
	for (uint64_t index = 0; index <= MAX_CONCURRENT_BOOSTS; ++index) {
		const auto         key = PrebuiltKey(kFirstKey + index);
		std::promise<void> release;
		auto               released = release.get_future().share();
		std::atomic<bool>  started {false};
		std::thread        prebuild([&] {
			(void)cache.GetOrBuildPrebuilt(key, [&] {
				started = true;
				released.wait();
				return driver.Build(LibraryPart::Fragment);
			});
		});
		Check(WaitFor(started), "the prebuild never started");
		std::thread request([&] { (void)cache.GetOrBuild(key, [&] { return driver.Build(LibraryPart::Fragment); }); });
		// The tag is noted just after the boost is counted, so the first round waits for both.
		const auto boosted = [&] {
			return cache.PrebuiltCounts().boosts == index + 1 &&
			       (index != 0 || Common::FirstRunReport::PrescanGraphicsBoosted() == before + 1);
		};
		const auto deadline = std::chrono::steady_clock::now() + kStuck;
		while (!boosted() && std::chrono::steady_clock::now() < deadline) {
			std::this_thread::yield();
		}
		tag_raised = tag_raised || (index == 0 && boosted());
		release.set_value();
		prebuild.join();
		request.join();
	}
	const auto counts = cache.PrebuiltCounts();
	Check(counts.boosts == MAX_CONCURRENT_BOOSTS + 1 && counts.boosts_refused_cap == 0 && counts.boosts_peak == 1 &&
	          log.lowers == MAX_CONCURRENT_BOOSTS + 1,
	      "a boost kept its slot after its prebuild dropped back");
	Check(tag_raised && Common::FirstRunReport::PrescanGraphicsBoosted() == before,
	      "the boosted tag did not count a running boost or did not give it back at the drop");
	std::printf("GraphicsLibraryCacheTests: boost gives back its slot and tag ok\n");
}

// What one round of TestThrowingPrebuildDropsBack saw.
struct ThrowingRound {
	bool threw      = false;
	bool tag_raised = false;
};

// Runs one prebuild of `key` that is held until a waiting request has boosted it and then throws, as a
// build that fails partway would. The prebuild's thread catches the throw, as a prescan worker's job
// loop would, and the waiter, still in the entry's once flag, is handed the flag and builds the library
// itself. The round waits for the boost and for the underrun tag it notes, one above `tag_start`.
ThrowingRound RunThrowingBoostedPrebuild(GraphicsLibraryCache& cache, FakeDriver& driver, const FragmentKey& key,
                                         uint64_t boosts_before, uint32_t tag_start) {
	ThrowingRound      round;
	std::promise<void> release;
	auto               released = release.get_future().share();
	std::atomic<bool>  started {false};
	std::thread        prebuild([&] {
		try {
			(void)cache.GetOrBuildPrebuilt(key, [&]() -> LibraryHandles {
				started = true;
				released.wait();
				throw std::runtime_error("the prebuild failed partway");
			});
		} catch (const std::runtime_error&) {
			round.threw = true;
		}
	});
	Check(WaitFor(started), "the throwing prebuild never started");
	std::thread request([&] { (void)cache.GetOrBuild(key, [&] { return driver.Build(LibraryPart::Fragment); }); });
	const auto boosted = [&] {
		return cache.PrebuiltCounts().boosts == boosts_before + 1 &&
		       Common::FirstRunReport::PrescanGraphicsBoosted() == tag_start + 1;
	};
	const auto deadline = std::chrono::steady_clock::now() + kStuck;
	while (!boosted() && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::yield();
	}
	round.tag_raised = boosted();
	release.set_value();
	prebuild.join();
	request.join();
	return round;
}

// A prebuild whose build throws after a waiting request boosted it still drops back and gives back its
// boost slot and its underrun tag. One more round than MAX_CONCURRENT_BOOSTS runs, each over before the
// next starts, so a slot a throw kept makes the cap refuse the last round's boost, and a kept tag stays
// raised. TestBoostGivesBackItsSlotAndTag is the baseline, the same rounds with builds that return, so
// only the throw is new here. Each round also checks that the tag rose while boosted, so a tag never
// noted cannot pass as one given back. The waiters build each library, which is what the once flag
// does after a throw with or without the boost.
// A throwing std::call_once must hand the flag to a waiter, which libstdc++ does not do on every
// target (GCC PR66146). x86_64 glibc is expected to be fine, and the first Linux CI run is the
// evidence. A hang there ends at the test's ctest TIMEOUT.
void TestThrowingPrebuildDropsBack() {
	constexpr uint64_t kFirstKey = 70;
	constexpr uint64_t kRounds   = MAX_CONCURRENT_BOOSTS + 1;
	FakeDriver         driver;
	auto               cache = MakeCache(driver);
	BoostLog           log;
	cache.SetPriorityHooksForTest(LoggingHooks(log));
	const uint32_t tag_before        = Common::FirstRunReport::PrescanGraphicsBoosted();
	bool           every_round_threw = true;
	bool           every_tag_raised  = true;
	for (uint64_t index = 0; index < kRounds; ++index) {
		const auto round = RunThrowingBoostedPrebuild(cache, driver, PrebuiltKey(kFirstKey + index), index,
		                                              Common::FirstRunReport::PrescanGraphicsBoosted());
		every_round_threw = every_round_threw && round.threw;
		every_tag_raised  = every_tag_raised && round.tag_raised;
	}
	const auto counts = cache.PrebuiltCounts();
	Check(every_round_threw && every_tag_raised && counts.boosts == kRounds,
	      "a throwing prebuild was not boosted with its tag raised, or its throw never reached its thread");
	Check(log.lowers == kRounds && counts.boosts_refused_cap == 0,
	      "a throwing prebuild did not drop back or kept its boost slot");
	Check(Common::FirstRunReport::PrescanGraphicsBoosted() == tag_before, "a throwing prebuild kept its boosted tag");
	Check(driver.Builds(LibraryPart::Fragment) == kRounds && cache.Counts(LibraryPart::Fragment).built == kRounds,
	      "a waiter did not build the library after the prebuild it waited on threw");
	std::printf("GraphicsLibraryCacheTests: throwing prebuild drops back ok\n");
}

// Hooks with no restore still finish a boosted prebuild, rather than end the process by calling an
// empty function from the drop-back guard's destructor, and the drop still gives back the tag. The
// baseline is TestWaitingRequestBoostsPrebuild, the same run with LoggingHooks, which drops back once.
// Here nothing is called to drop back, and the tag coming back shows the rest of the drop still ran. The
// boost is checked as well, since a run that never boosted would never reach the drop at all.
void TestBoostWithoutRestoreHookFinishes() {
	FakeDriver driver;
	auto       cache = MakeCache(driver);
	BoostLog   log;
	auto       hooks = LoggingHooks(log);
	hooks.restore    = nullptr;
	cache.SetPriorityHooksForTest(hooks);
	const uint32_t tag_before = Common::FirstRunReport::PrescanGraphicsBoosted();
	// stderr is unbuffered, so if the drop-back ends the process this line still names the test.
	std::fputs("GraphicsLibraryCacheTests: boost without a restore hook\n", stderr);
	RunPrebuildAndRequest(cache, driver, log, true);
	Check(log.boosts == 1 && cache.PrebuiltCounts().boosts == 1, "the waiting request did not boost the prebuild");
	Check(Common::FirstRunReport::PrescanGraphicsBoosted() == tag_before,
	      "a prebuild boosted under hooks with no restore did not finish its drop-back");
	Check(driver.Builds(LibraryPart::Fragment) == 1, "the waiting request built its own copy");
	std::printf("GraphicsLibraryCacheTests: boost without a restore hook finishes ok\n");
}

} // namespace

int main() {
	TestDedup();
	TestConcurrentBuilds();
	TestWaitsForRunningBuildOnly();
	TestRowsSeparateBuildAndWait();
	TestRefusalIsKept();
	TestDestroyOnce();
	TestLinkAndFallbackCounts();
	TestPrebuiltFirstHit();
	TestPrebuiltWaitIsCounted();
	TestPrebuiltKeyDiffers();
	TestWaitingRequestBoostsPrebuild();
	TestBoostsAreCapped();
	TestBlockingWaiterBoostsPastCap();
	TestPreRasterPrebuiltFirstHit();
	TestWaitClaimsThePrebuilt();
	TestPreRasterKeyDiffers();
	TestPrebuildCountsAsBuilt();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TestBoostDropsBackToTheLoweredLevel();
	TestBoostRestoresTheBuilderLevel();
#endif
	TestBoostGivesBackItsSlotAndTag();
	TestThrowingPrebuildDropsBack();
	TestBoostWithoutRestoreHookFinishes();
	std::printf("GraphicsLibraryCacheTests: ok\n");
	return 0;
}
