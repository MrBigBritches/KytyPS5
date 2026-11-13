#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GENERATIONSTATE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GENERATIONSTATE_H_

#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/shader/storegen/generatedDrainPlanner.h"
#include "graphics/shader/storegen/generatedOrder.h"
#include "graphics/shader/storegen/generationRows.h"
#include "graphics/shader/storegen/storeSupervisor.h"

#include <atomic>
#include <chrono>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
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
	// The tier and candidate count the supervisor gave the record. They fill the report's row, and
	// the wide flag puts the warm build on the back lane.
	StoreGen::GeneratedOrder order;
};

// What one producer of generated modules keeps while it makes them on its own thread. The generator
// and the prescan's preparer each have one, so MakeGeneratedModule never shares a set between two
// threads.
struct PipelineCache::GeneratedIntake {
	// Each module carries its program's plan. The prescan's modules do, so a drain never has to
	// guess whether the program the plan was decoded for still exists.
	bool always_plan = false;
	// The program keys whose plan was decoded. Only the producer's thread touches it, and a
	// producer with always_plan never fills it. The counters every producer adds to are atomics in
	// the state, which two threads may add to at once.
	std::unordered_set<ProgramKey, ProgramKeyHash> decoded_keys;
};

struct PipelineCache::GenerationState {
	// A copy of a generated module would silently drop its retained record, so the backlogs and the
	// inbox must move them, which containers do only when the move cannot throw. It is checked here,
	// since the state is nested in the cache and may name the cache's private types.
	static_assert(std::is_nothrow_move_constructible_v<GeneratedModule>);

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
	// The producers' own states. The generator's thread uses the first and the preparer's the
	// second.
	GeneratedIntake generator;
	GeneratedIntake prescan {.always_plan = true};
	// The GPU thread's own part. A drain moves the inbox into the backlogs and takes a few at a
	// time. Prescan modules have their own backlog, which is taken first, since the scan found them
	// about to run and a burst of generator modules must not hold them back.
	std::deque<GeneratedModule>     backlog;
	std::deque<GeneratedModule>     prescan_backlog;
	StoreGen::GeneratedDrainPlanner planner;
	// The prescan's own planner. Each planner remembers which programs it trusts, so with one shared
	// planner a prescan module with a bad plan would make the generator's modules of that program
	// refused for good, and each producer's figures would hold the other's modules.
	StoreGen::GeneratedDrainPlanner prescan_planner;
	// The store records of adopted prescan modules that no lookup has used yet. The GPU thread owns
	// it, like the backlogs. The records belong to their modules, and the list is released by
	// StopStoreGeneration, so destroying the program cache after the state touches no list.
	RetainedRecordList retained_records;

	// What the drain last counted, copied under the mutex at the end of each drain so the report's
	// writer can read it from another thread. The planners' own counters belong to the GPU thread.
	StoreGen::GeneratedDrainCounters published_drain;
	StoreGen::GeneratedDrainCounters published_prescan_drain;
	uint64_t                         published_backlog = 0;
	// The prescan's record counts, copied with the drain's, and once more at the stop, only with
	// the prescan on. The list itself belongs to the GPU thread.
	RetainedRecordCounters published_prescan_records;

	// The rows of the report, all guarded by the mutex above. They live until the cache is
	// destroyed, which is after the warm queue has stopped, and StopStoreGeneration never clears
	// them, so a build that is still running may stamp its row after the stop. A deque keeps the
	// address of a row that a queued build holds. A push may allocate a chunk, which is short and
	// rare, and the lock is never held across anything else.
	std::deque<StoreGen::GenRow>  generated_rows;
	std::deque<StoreGen::DropRow> refused_rows;
	std::deque<StoreGen::MissRow> miss_rows;
	// Whether the report was given a source, which StopStoreGeneration must then replace.
	bool report_registered = false;
	// Whether the report was given the prescan's source, which StopStoreGeneration must then
	// replace. Like report_registered it is a plain bool of the thread that owns the cache, which
	// sets it and stops the generation, so the mutex does not guard it.
	bool prescan_report_registered = false;
	// Whether the prescan's preparer posts into this intake, which the generation's report marks on
	// its drain figures. Set once by EnablePrescan and read by the report's writer on its own
	// thread.
	std::atomic<bool> includes_prescan {false};

	// Every producer adds to these, so the handles made balance the handles destroyed once nothing
	// waits, whoever made them.
	std::atomic<uint64_t> posted {0};
	std::atomic<uint64_t> created {0};
	std::atomic<uint64_t> destroyed {0};
	std::atomic<uint64_t> bad_plans {0};
	std::atomic<uint64_t> undrained_at_stop {0};

	// The modules both backlogs hold. GPU thread only.
	[[nodiscard]] size_t BacklogSize() const { return backlog.size() + prescan_backlog.size(); }

	// The generator, which posts into this state. It is declared last so it is destroyed first, and
	// StopStoreGeneration stops and joins it before anything else, so nothing posts after the rest
	// is gone.
	std::unique_ptr<StoreGen::StoreSupervisor> supervisor;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_GENERATIONSTATE_H_
