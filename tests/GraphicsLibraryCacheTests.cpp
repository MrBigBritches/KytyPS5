// Tests the cache that shares graphics pipeline libraries. No device is needed, since the libraries are
// made by functions the tests give it, which count what they make and hand out fake handles.

#include "PipelinePartsFixtures.h"
#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <mutex>
#include <set>
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

} // namespace

int main() {
	TestDedup();
	TestConcurrentBuilds();
	TestWaitsForRunningBuildOnly();
	TestRowsSeparateBuildAndWait();
	TestRefusalIsKept();
	TestDestroyOnce();
	TestLinkAndFallbackCounts();
	std::printf("GraphicsLibraryCacheTests: ok\n");
	return 0;
}
