#include "common/childProcess.h"
#include "common/emulatorConfig.h"
#include "common/firstRunReport.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/pipeline/generationState.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/warmLane.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/ebootSource.h"
#include "graphics/shader/storegen/generationReport.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <algorithm>
#include <chrono>
#include <span>
#include <tuple>
#include <xxhash.h>

// The intake of modules that the shader store generator made. The generator's thread decodes each
// record and makes its Vulkan shader module, and posts it. The GPU thread takes it into the program
// cache at the top of a shader lookup and queues its build on the warm queue, in the order the
// modules were posted.
namespace Libs::Graphics {

namespace {

using Clock = std::chrono::steady_clock;

// The hash that stands for a plan, from the bytes a record holds it as.
uint64_t HashPlanBytes(std::span<const uint8_t> bytes) {
	return XXH3_64bits(bytes.data(), bytes.size());
}

// The number by which the planner knows a program, which is its address. A program never moves,
// since the program cache is a node map and a program leaves it only when it is erased.
uint64_t ProgramNumber(const void* program) {
	return reinterpret_cast<uintptr_t>(program);
}

// A time of the steady clock on the clock of the first-run report, which counts nanoseconds from
// its epoch.
int64_t ToNanoseconds(Clock::time_point time) {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

} // namespace

void PipelineCache::EnableStoreGeneration() {
	if (m_generation == nullptr) {
		m_generation = std::make_unique<GenerationState>();
	}
}

// The report asks for the figures from its own thread, at exit, which may come without any
// destructor running. StopStoreGeneration takes the source away again, and the source must not
// outlive this cache.
void PipelineCache::RegisterGenerationReport() {
	m_generation->report_registered = true;
	Common::FirstRunReport::Global().SetGenerationSource(
	    [this] { return BuildGenerationReport(false); });
}

void PipelineCache::StartStoreGeneration(const std::filesystem::path& store_path,
                                         std::span<const uint8_t>     header) {
	EnableStoreGeneration();
	StoreGen::SupervisorConfig config;
	config.request.store_header.assign(header.begin(), header.end());
	config.request.host_subgroup_size = HostComputeSubgroupSize(m_graphics);
	config.request.eboot_path         = StoreGen::Utf8FromPath(StoreGen::EbootPath());
	// The output is next to the store and named for this process, so two emulators of one title do
	// not write one file.
	const auto stem       = store_path.stem().string();
	const auto pid        = std::to_string(Common::CurrentProcessId());
	config.output_path    = store_path.parent_path() / (stem + ".generated." + pid + ".tmp");
	config.generated_path = store_path.parent_path() / (stem + ".generated");
	config.sink = [this](DecodedModuleRecord&& record, const StoreGen::GeneratedOrder& order) {
		PostGeneratedModule(std::move(record), order);
	};
	config.log = [](std::string_view text) { Log::WriteToConsoleAndLog(std::string(text) + "\n"); };
	if (m_supervisor_tweak) {
		m_supervisor_tweak(config);
	}
	m_generation->supervisor = std::make_unique<StoreGen::StoreSupervisor>(std::move(config));
	m_generation->supervisor->Start();
	RegisterGenerationReport();
	Log::WriteToConsoleAndLog("Shader store generation: started\n");
}

// Runs on the generator's thread. It touches no cache map, only the device, which creates shader
// modules from any thread, and the state of the intake that only this thread uses.
std::optional<PipelineCache::GeneratedModule>
PipelineCache::MakeGeneratedModule(DecodedModuleRecord&&           record,
                                   const StoreGen::GeneratedOrder& order) {
	auto&           state = *m_generation;
	GeneratedModule item;
	item.order     = order;
	item.plan_hash = HashPlanBytes(record.plan);
	FillKeyFromRecord(record, item.key);
	// The plan is decoded for the first record of a program only. Decoding it is the costly part of
	// a record, and the same program's other records carry the same plan.
	if (!state.decoded_keys.contains(item.key)) {
		ShaderRecompiler::IR::ResourcePlan plan;
		if (!ShaderRecompiler::IR::LoadResourcePlan(record.plan, plan)) {
			state.bad_plans++;
			return std::nullopt;
		}
		item.plan = std::move(plan);
		state.decoded_keys.insert(item.key);
	}
	// The SPIR-V is taken out first, so nothing reads the record after it is moved.
	const auto spirv = std::move(record.spirv);
	FillModuleFromRecord(std::move(record), item.module);
	item.module.module     = CompileSPV(spirv, m_graphics.device);
	item.module.gpl_module = CompileSplitPixelModule(m_graphics, item.key.stage, spirv);
	state.created += (item.module.module ? 1 : 0) + (item.module.gpl_module ? 1 : 0);
	item.posted = Clock::now();
	return item;
}

// Makes the module of a record on the generator's thread and hands it to the GPU thread. The order
// travels with it, for the report's rows and for the lane of the module's warm build.
void PipelineCache::PostGeneratedModule(DecodedModuleRecord&&           record,
                                        const StoreGen::GeneratedOrder& order) {
	if (m_generation == nullptr) {
		return;
	}
	if (auto item = MakeGeneratedModule(std::move(record), order)) {
		PushGenerated(std::move(*item));
	}
}

void PipelineCache::PushGenerated(GeneratedModule&& item) {
	auto& state = *m_generation;
	// The node is made before the lock is taken, so the lock only splices it in and the GPU thread
	// never waits for an allocation.
	std::list<GeneratedModule> node;
	node.push_back(std::move(item));
	{
		std::scoped_lock lock(state.mutex);
		if (!state.stopped.load(std::memory_order_relaxed)) {
			state.inbox.splice(state.inbox.end(), node);
			state.posted++;
			// Set after the push and under the lock, so a drain that clears it before it takes the
			// inbox cannot miss this module.
			m_has_generated.store(true, std::memory_order_relaxed);
			return;
		}
	}
	DestroyGeneratedHandles(node.front());
}

void PipelineCache::DestroyGeneratedHandles(GeneratedModule& item) {
	auto& state = *m_generation;
	if (item.module.module) {
		m_graphics.device.destroyShaderModule(item.module.module, nullptr);
		state.destroyed++;
	}
	if (item.module.gpl_module) {
		m_graphics.device.destroyShaderModule(item.module.gpl_module, nullptr);
		state.destroyed++;
	}
	item.module.module     = nullptr;
	item.module.gpl_module = nullptr;
}

// No GpuStall scope is taken around the drain on purpose. A drain is neither a compile nor a wait:
// it moves a few modules between maps and queues their builds, which takes microseconds, and the
// budget below keeps even a slow plan to about a millisecond. A scope would count such a drain as a
// stall of the GPU thread and could make guest audio hold for nothing.
void PipelineCache::DrainGenerated() {
	auto&                       state = *m_generation;
	const auto                  start = Clock::now();
	const StoreGen::DrainBudget budget;
	// Cleared before the inbox is taken, so a module posted after the take sets it again.
	m_has_generated.store(false, std::memory_order_relaxed);
	// The whole inbox is swapped out under the lock, which takes constant time however many modules
	// wait, and moved to the backlog after it is released.
	std::list<GeneratedModule> taken;
	{
		std::scoped_lock lock(state.mutex);
		taken.swap(state.inbox);
	}
	for (auto& item: taken) {
		state.backlog.push_back(std::move(item));
	}
	size_t                   handled = 0;
	std::chrono::nanoseconds oldest_wait {};
	while (!state.backlog.empty() && budget.Allows(handled, Clock::now() - start)) {
		auto item = std::move(state.backlog.front());
		state.backlog.pop_front();
		oldest_wait = std::max(
		    oldest_wait, std::chrono::duration_cast<std::chrono::nanoseconds>(start - item.posted));
		const auto item_start = Clock::now();
		AdoptGenerated(std::move(item));
		// The time budget cannot cut one module short, so the longest one is kept to show whether a
		// big plan pushes a drain over it.
		state.planner.NoteItem(Clock::now() - item_start);
		handled++;
	}
	// What the budget left stays in the backlog, and the next lookup goes on with it.
	if (!state.backlog.empty()) {
		m_has_generated.store(true, std::memory_order_relaxed);
	}
	state.planner.NoteDrain(Clock::now() - start, oldest_wait);
	// Handed to the report's writer. A copy of a few counters under the mutex that a post already
	// takes, so it costs the GPU thread no more than a post costs the generator.
	{
		std::scoped_lock lock(state.mutex);
		state.published_drain   = state.planner.Counters();
		state.published_backlog = state.backlog.size();
	}
}

void PipelineCache::AdoptGenerated(GeneratedModule&& item) {
	auto&                state = *m_generation;
	StoreGen::DrainFacts facts;
	facts.stopped = state.stopped.load(std::memory_order_relaxed) || m_warm_compute == nullptr ||
	                m_warm_compute->Stopping();
	facts.plan_hash                                                           = item.plan_hash;
	Program*                                                          program = nullptr;
	std::unordered_map<ProgramKey, Program, ProgramKeyHash>::iterator entry;
	if (!facts.stopped) {
		bool inserted             = false;
		std::tie(entry, inserted) = m_program_cache.try_emplace(item.key);
		program                   = &entry->second;
		facts.program             = ProgramNumber(program);
		facts.program_inserted    = inserted;
		if (inserted) {
			facts.plan_installed = item.plan.has_value();
			if (item.plan) {
				program->resource_plan = std::move(*item.plan);
			}
		} else {
			facts.id_present =
			    std::ranges::any_of(program->modules, [&](const ShaderModule& module) {
				    return module.id == item.module.id;
			    });
		}
	}
	const auto verdict = state.planner.Judge(facts, [&] {
		return HashPlanBytes(ShaderRecompiler::IR::SaveResourcePlan(program->resource_plan));
	});
	switch (verdict) {
		case StoreGen::DrainVerdict::Adopt: {
			StoreGen::GenRow* row = NoteAdopted(item);
			// A deque, so the address that the queued build holds stays valid.
			program->modules.push_back(std::move(item.module));
			// The wide tail goes to the back lane, so it never delays the other builds.
			StartWarmCompute(program->modules.back(), row,
			                 WarmLaneOf(Config::WarmWideTailLastEnabled(), item.order));
			return;
		}
		case StoreGen::DrainVerdict::DropFailedInstall:
			// The program was made for this module and has no plan, so it goes again, as a store
			// record whose plan cannot be loaded does.
			m_program_cache.erase(entry);
			break;
		default: break;
	}
	NoteRefused(item, verdict);
	DestroyGeneratedHandles(item);
}

// Adds the row of a module that was taken in, and returns it for the warm build to stamp. The row
// is pushed under the state's mutex, which the report's writer also takes to copy the rows.
StoreGen::GenRow* PipelineCache::NoteAdopted(const GeneratedModule& item) {
	auto&            state   = *m_generation;
	const int64_t    posted  = ToNanoseconds(item.posted);
	const int64_t    adopted = Common::FirstRunReport::NowNs();
	std::scoped_lock lock(state.mutex);
	auto& row = state.generated_rows.emplace_back(item.module.id, item.key.hash, posted, adopted);
	row.tier  = item.order.tier;
	row.wide_tail  = item.order.wide_tail;
	row.candidates = item.order.candidates;
	return &row;
}

// Adds the row of a module that the drain refused. Only the drain's refusals are rows. A module
// that never reached the drain because the generator was stopped is counted as undrained instead.
void PipelineCache::NoteRefused(const GeneratedModule& item, StoreGen::DrainVerdict verdict) {
	using StoreGen::DrainVerdict;
	using StoreGen::DropVerdict;
	const DropVerdict named =
	    verdict == DrainVerdict::DropDuplicate       ? DropVerdict::Duplicate
	    : verdict == DrainVerdict::DropPlanMismatch  ? DropVerdict::PlanMismatch
	    : verdict == DrainVerdict::DropFailedInstall ? DropVerdict::FailedInstall
	                                                 : DropVerdict::DroppedStopped;
	// The time is taken before the lock, as a miss's is, so both are on the same footing.
	const int64_t    refused_at = Common::FirstRunReport::NowNs();
	auto&            state      = *m_generation;
	std::scoped_lock lock(state.mutex);
	state.refused_rows.push_back({item.module.id, named, refused_at});
}

void PipelineCache::NoteGeneratedMiss(uint64_t code_hash, uint64_t id, bool has_calls,
                                      bool skip_dispatch) {
	if (m_generation == nullptr) {
		return;
	}
	// The time is taken before the lock, so a wait for it is not counted as part of the miss.
	const StoreGen::MissRow row {.code_hash     = code_hash,
	                             .id            = id,
	                             .at_ns         = Common::FirstRunReport::NowNs(),
	                             .has_calls     = has_calls,
	                             .skip_dispatch = skip_dispatch};
	std::scoped_lock        lock(m_generation->mutex);
	m_generation->miss_rows.push_back(row);
}

// A program that generation made was translated again by the runtime, which happens for a module
// the generator did not predict. A program's plan depends only on its code, so the two plans must
// be the same, and a difference means the generator and the runtime disagree. It is only counted.
void PipelineCache::CheckGeneratedPlan(Program&                                 program,
                                       const ShaderRecompiler::TranslateResult& translated) {
	(void)m_generation->planner.CheckReverse(ProgramNumber(&program), [&] {
		return HashPlanBytes(ShaderRecompiler::IR::SaveResourcePlan(
		    ShaderRecompiler::IR::ExtractResourcePlan(translated.program)));
	});
}

void PipelineCache::StopStoreGeneration() {
	if (m_generation == nullptr) {
		return;
	}
	auto& state = *m_generation;
	// The generator is stopped and joined first: it ends its child process on its own thread, and
	// it is the only thing that posts into the state below. The destructor of the supervisor would
	// do the same, but the order here does not depend on it.
	if (state.supervisor) {
		state.supervisor->Stop();
	}
	std::list<GeneratedModule> inbox;
	{
		std::scoped_lock lock(state.mutex);
		state.stopped.store(true, std::memory_order_relaxed);
		inbox.swap(state.inbox);
	}
	std::vector<GeneratedModule> undrained;
	for (auto& item: inbox) {
		undrained.push_back(std::move(item));
	}
	for (auto& item: state.backlog) {
		undrained.push_back(std::move(item));
	}
	state.backlog.clear();
	state.undrained_at_stop += undrained.size();
	for (auto& item: undrained) {
		DestroyGeneratedHandles(item);
	}
	m_has_generated.store(false, std::memory_order_relaxed);
	// The final figures are taken now that nothing posts and nothing drains, and handed to the
	// report in one step that also takes the source away. The writer may be calling the source at
	// this moment, which is safe since the state still exists. After this it never calls it again,
	// so the state can be destroyed. No lock is held across the two steps, since building the
	// figures takes the state's mutex and handing them over takes the report's.
	if (state.report_registered) {
		state.report_registered = false;
		Common::FirstRunReport::Global().SetGenerationFinal(BuildGenerationReport(true));
	}
}

// Copies the cache's rows under the state's mutex, which is the only place the writer reads them.
StoreGen::GenerationRowInputs PipelineCache::SnapshotGenerationRows() const {
	auto&                         state = *m_generation;
	StoreGen::GenerationRowInputs rows;
	std::scoped_lock              lock(state.mutex);
	for (const auto& row: state.generated_rows) {
		rows.generated.push_back(StoreGen::SnapshotOf(row));
	}
	rows.refused.assign(state.refused_rows.begin(), state.refused_rows.end());
	rows.misses.assign(state.miss_rows.begin(), state.miss_rows.end());
	return rows;
}

Common::FirstRunReport::ShaderStoreGenerationReport
PipelineCache::BuildGenerationReport(bool final_figures) const {
	auto&                            state = *m_generation;
	StoreGen::GenerationCounts       counts;
	StoreGen::GeneratedDrainCounters drain;
	{
		std::scoped_lock lock(state.mutex);
		drain = final_figures ? state.planner.Counters() : state.published_drain;
		counts.pending =
		    state.inbox.size() + (final_figures ? state.backlog.size() : state.published_backlog);
	}
	counts.posted            = state.posted;
	counts.handles_created   = state.created;
	counts.handles_destroyed = state.destroyed;
	counts.bad_plans         = state.bad_plans;
	counts.undrained_at_stop = state.undrained_at_stop;
	StoreGen::SupervisorStats stats;
	if (state.supervisor) {
		stats = state.supervisor->Stats();
	}
	auto rows = SnapshotGenerationRows();
	// The report's own rows are copied after the state's lock is released, so the two are never
	// held together.
	rows.compiles = Common::FirstRunReport::Global().Compiles();
	rows.takes    = Common::FirstRunReport::Global().WarmTakes();
	return StoreGen::MakeGenerationReport(stats, drain, counts, rows);
}

std::optional<StoreGenerationStats> PipelineCache::GenerationStats() const {
	if (m_generation == nullptr) {
		return std::nullopt;
	}
	auto&                state = *m_generation;
	StoreGenerationStats stats;
	stats.drain             = state.planner.Counters();
	stats.posted            = state.posted;
	stats.handles_created   = state.created;
	stats.handles_destroyed = state.destroyed;
	stats.bad_plans         = state.bad_plans;
	stats.undrained_at_stop = state.undrained_at_stop;
	// The inbox is the generator's thread's to fill, so it is read under the lock. The backlog is
	// the GPU thread's own, which is why the whole count is only exact when nothing drains
	// meanwhile.
	{
		std::scoped_lock lock(state.mutex);
		stats.pending = state.inbox.size();
	}
	stats.pending += state.backlog.size();
	return stats;
}

} // namespace Libs::Graphics
