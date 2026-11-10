#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GENERATIONSTATE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GENERATIONSTATE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/storegen/generatedDrainPlanner.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <list>
#include <mutex>
#include <optional>
#include <unordered_set>
#include <vector>

// The parts of the pipeline cache that take in generated shader modules. They are defined apart
// from pipelineCache.h so that only the files that use them see them. A cache made without
// generation holds a null pointer to the state, and nothing more.
namespace Libs::Graphics {

// A module made on the generator's thread, waiting to be taken into the program cache.
struct PipelineCache::GeneratedModule {
	ProgramKey key;
	// The plan of the program, decoded for the first record of each program key only. Later records
	// of the program carry the hash and no plan, since the inbox is first in, first out and the
	// record with the plan is always taken first.
	std::optional<ShaderRecompiler::IR::ResourcePlan> plan;
	// The hash of the plan bytes of the record, which is what the tripwire compares.
	uint64_t     plan_hash = 0;
	ShaderModule module;
	// When the generator's thread posted it, for the report's drain lag.
	std::chrono::steady_clock::time_point posted;
};

struct PipelineCache::GenerationState {
	// The generator's thread fills the inbox under the mutex and sets m_has_generated, and the GPU
	// thread clears the flag before it takes the inbox, as for the published builds. A list, so
	// that both sides only splice under the lock, which is constant time and allocates nothing: the
	// producer makes its node before it takes the lock, and the GPU thread swaps the whole list
	// out.
	std::mutex                 mutex;
	std::list<GeneratedModule> inbox;
	// Set once, under the mutex, so that nothing is posted after the stop destroyed the inbox. It
	// is atomic because the GPU thread reads it without the lock, when it takes a module in.
	std::atomic<bool> stopped {false};
	// The program keys whose plan was decoded. Only the generator's thread touches it.
	std::unordered_set<ProgramKey, ProgramKeyHash> decoded_keys;
	// The GPU thread's own part. A drain moves the inbox into the backlog and takes a few at a
	// time.
	std::deque<GeneratedModule>     backlog;
	StoreGen::GeneratedDrainPlanner planner;

	std::atomic<uint64_t> posted {0};
	std::atomic<uint64_t> created {0};
	std::atomic<uint64_t> destroyed {0};
	std::atomic<uint64_t> bad_plans {0};
	std::atomic<uint64_t> undrained_at_stop {0};
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GENERATIONSTATE_H_
