#include "common/childProcess.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/renderer/pipeline/generationState.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/prescanPreparer.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/moduleId.h"
#include "kernel/memory.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

// The pipeline cache's side of the submit-time compute prescan. It makes the preparer with the host
// values this device translates with, and gives it the intake that generated modules use, so the
// GPU thread takes a prescan module in at the top of a shader lookup and builds its pipeline on the
// prescan's own warm set.
namespace Libs::Graphics {

namespace {

// Reads guest memory for the preparer through the backing alias, which never faults and never waits
// for a readback, so it is safe off the GPU thread. The runtime's own reader is for the GPU thread
// only.
bool ReadPrescanMemory(void* /*userdata*/, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() &&
	       LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes());
}

// The transport of a real run, the Stream worker in a child process. Its request file and progress
// log go in a directory no other user can write to, under this process's id, so two emulators
// never share a log and a crash is never blamed on the other's request. The prescan runs without a
// shader store too, so there is no store directory to put them next to. The files that dead runs
// left there are removed first. The transport is written to `made` when the preparer makes it, so
// the report can read its supervisor's counts.
// Returns nothing when there is no such directory.
std::optional<StoreGen::PrescanTransportFactory>
HelperTransportFactory(StoreGen::SupervisorPrescanTransport** made) {
	const auto directory = StoreGen::PrescanHelperDirectory();
	if (!directory) {
		return std::nullopt;
	}
	const uint32_t pid = Common::CurrentProcessId();
	(void)StoreGen::RemoveStalePrescanFiles(
	    *directory, pid, &Common::ProcessIsRunning,
	    [](std::string_view line) { Log::WriteToConsoleAndLog(std::string(line) + "\n"); });
	StoreGen::StreamSupervisorConfig config;
	config.files_base = StoreGen::PrescanHelperFilesBase(*directory, pid);
	return StoreGen::SupervisorPrescanTransportFactory(std::move(config), made);
}

} // namespace

// The host values are the ones the store header records and GetComputeProgram reads, so the helper
// translates exactly as the runtime would on this device and with this Config.
PrescanPreparerConfig
PipelineCache::MakePrescanConfig(StoreGen::PrescanTransportFactory make_transport) {
	const uint32_t max_shared =
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	PrescanPreparerConfig config;
	config.lookup      = [](uint64_t address) { return ShaderTryGetMapped(address); };
	config.read_memory = ReadPrescanMemory;
	config.host        = {.structured_jump_tables = Config::StructuredJumpTablesEnabled(),
	                      .profile                = SpirvProfile(m_graphics),
	                      .max_shared_bytes       = max_shared};
	// The same source as the generator's request and GetComputeProgram.
	config.device         = {.host_subgroup_size              = HostComputeSubgroupSize(m_graphics),
	                         .max_compute_shared_memory_bytes = max_shared};
	config.make_transport = std::move(make_transport);
	// Prescan modules persist only with a shader store. Without one the record bytes are left with the
	// preparer, which frees them, so the GPU thread never frees them at adoption. The store is opened
	// by WarmUp, which runs before EnablePrescan, so whether there is one is known here and never
	// changes after.
	const bool keep_records = static_cast<bool>(m_store);
	config.sink = [this, keep_records](DecodedModuleRecord&& record, std::vector<uint8_t>&& record_bytes) {
		PostPrescanModule(std::move(record), keep_records ? std::move(record_bytes) : std::vector<uint8_t> {});
	};
	// EnablePrescan runs after WarmUp, so the seed holds every record the shader store loaded, with
	// what the generator adopted so far. Taken here and not as each record loads, so a run without
	// the prescan computes no keys at all.
	config.known_modules = KnownModuleKeys();
	return config;
}

// The module's specialization is the one it was compiled with, and the key's fields are the ones
// its id was made from, so this is the key the preparer makes for the same module before compiling.
uint64_t PipelineCache::SkipKeyOf(const ProgramKey& key, const ShaderModule& module) {
	return ShaderModuleSkipKey(key.stage, key.hash, key.user_data_count, key.static_state,
	                           module.specialization);
}

// Every module counts, whoever made it, since the drain refuses a duplicate of any of them. Only
// compute modules without GPU-selected calls, since the prescan builds nothing else. The helper
// refuses a program with calls at its Plan, so no snapshot of one ever asks for its key. The keys
// are counted first, so the store's thousands of modules fill the vector without regrowing it.
std::vector<uint64_t> PipelineCache::KnownModuleKeys() const {
	size_t count = 0;
	for (const auto& [key, program]: m_program_cache) {
		count += key.stage == ShaderType::Compute ? program.modules.size() : 0;
	}
	std::vector<uint64_t> keys;
	keys.reserve(count);
	for (const auto& [key, program]: m_program_cache) {
		if (key.stage != ShaderType::Compute) {
			continue;
		}
		for (const auto& module: program.modules) {
			if (module.function_code.empty()) {
				keys.push_back(SkipKeyOf(key, module));
			}
		}
	}
	return keys;
}

// A stopped preparer reads no more reports, so the key is not even hashed then. A module with
// GPU-selected calls stays out, as it does from the seed.
void PipelineCache::NoteKnownModule(const ProgramKey& key, const ShaderModule& module) {
	if (m_prescan == nullptr || key.stage != ShaderType::Compute || !module.function_code.empty() ||
	    m_prescan->Stopped()) {
		return;
	}
	m_prescan->NoteKnownModule(SkipKeyOf(key, module));
}

void PipelineCache::EnablePrescan(StoreGen::PrescanTransportFactory make_transport) {
	if (m_prescan != nullptr) {
		return;
	}
	// The factory writes the helper's transport here while the preparer is made below. A
	// preparer that failed destroyed its transport, so what an earlier failed call left is
	// cleared first. The factory writes a member and not a local, since the preparer keeps the
	// factory.
	m_prescan_supervisor = nullptr;
	if (!make_transport) {
		auto helper = HelperTransportFactory(&m_prescan_supervisor);
		if (!helper) {
			Log::WriteToConsoleAndLog("Prescan: there is no private directory for the helper's "
			                          "files, so the prescan is off\n");
			return;
		}
		make_transport = std::move(*helper);
	}
	// The intake exists only with a generation state. Its report is not registered here, since the
	// generator did not run.
	EnableStoreGeneration();
	m_prescan_handoff = std::make_unique<PrescanHandoff>();
	m_prescan_compute = std::make_unique<WarmBuildSet<uint64_t, Pipeline>>(
	    PRESCAN_BUILD_WORKERS, PipelineBuildQueueOptions {.low_priority = true});
	auto config = MakePrescanConfig(std::move(make_transport));
	if (m_prescan_tweak) {
		m_prescan_tweak(config);
	}
	m_prescan = std::make_unique<PrescanPreparer>(*m_prescan_handoff, std::move(config));
	// Marked before the preparer runs, so no module of it reaches the intake unmarked.
	m_generation->includes_prescan.store(true, std::memory_order_relaxed);
	m_prescan->Start();
	Log::WriteToConsoleAndLog("Prescan: the preparer started\n");
	// Registered last, so a throw from any step above leaves the report without a source that reads
	// this cache. StopStoreGeneration replaces it with the final figures, and the destructor always
	// runs that before anything the source reads is destroyed.
	m_generation->prescan_report_registered = true;
	Common::FirstRunReport::Global().SetPrescanCacheSource(
	    [this] { return BuildPrescanReport(false); });
}

void PipelineCache::StopPrescan() {
	if (m_prescan != nullptr) {
		m_prescan->Stop();
	}
}

Common::FirstRunReport::PrescanDrainReport
PrescanDrainReportOf(const StoreGen::GeneratedDrainCounters& prescan,
                     const StoreGen::GeneratedDrainCounters& both) {
	return {.adopted         = prescan.adopted,
	        .duplicates      = prescan.duplicates,
	        .plan_mismatch   = prescan.plan_mismatch,
	        .failed_installs = prescan.failed_installs,
	        .dropped_stopped = prescan.dropped_stopped,
	        .item_max_ms     = prescan.item_max_ms,
	        .both_calls      = both.drain_calls,
	        .both_over_1ms   = both.drain_over_1ms,
	        .both_over_4ms   = both.drain_over_4ms,
	        .both_max_ms     = both.drain_max_ms,
	        .both_total_ms   = both.drain_total_ms,
	        .both_lag_ms_max = both.drain_lag_ms_max};
}

Common::FirstRunReport::PrescanBuildsReport PrescanBuildsReportOf(const PrescanBuildStats& builds) {
	return {.builds        = builds.builds,
	        .worker_builds = builds.worker_builds,
	        .in_scope      = builds.in_scope,
	        .in_hold       = builds.in_hold,
	        .total_ms      = builds.total_ms,
	        .in_scope_ms   = builds.in_scope_ms,
	        .in_hold_ms    = builds.in_hold_ms};
}

Common::FirstRunReport::PrescanRecordsReport
PrescanRecordsReportOf(const RetainedRecordCounters& records) {
	return {.stored_first_use    = records.stored_first_use,
	        .retained_peak_bytes = records.retained_peak_bytes,
	        .dropped_over_cap    = records.dropped_over_cap,
	        .unused_at_exit      = records.unused_at_exit};
}

// The state's lock is taken alone and released before the preparer's and the supervisor's own locks
// are taken, so no two of them are ever held together. The preparer and its transport outlive its
// stop, so a source called after StopPrescan still reads live objects.
Common::FirstRunReport::PrescanCacheReport
PipelineCache::BuildPrescanReport(bool final_figures) const {
	// Only a cache whose prescan started registers the source or hands over final figures.
	EXIT_IF(m_prescan == nullptr);
	auto&                            state = *m_generation;
	StoreGen::GeneratedDrainCounters prescan_drain;
	StoreGen::GeneratedDrainCounters both_drain;
	RetainedRecordCounters           records;
	{
		std::scoped_lock lock(state.mutex);
		prescan_drain =
		    final_figures ? state.prescan_planner.Counters() : state.published_prescan_drain;
		both_drain = final_figures ? state.planner.Counters() : state.published_drain;
		records    = state.published_prescan_records;
	}
	Common::FirstRunReport::PrescanCacheReport report;
	const auto                                 counters = m_prescan->Counters();
	report.state                                        = PrescanStateReportOf(counters.state);
	report.preparer                                     = PrescanPreparerReportOf(counters);
	if (m_prescan_supervisor != nullptr) {
		report.child = PrescanChildReportOf(m_prescan_supervisor->Stats());
	}
	report.drain                        = PrescanDrainReportOf(prescan_drain, both_drain);
	report.builds                       = PrescanBuildsReportOf(PrescanBuilds());
	report.holds.in_flight_at_miss_only = PrescanInFlightAtMissOnly();
	report.records                      = PrescanRecordsReportOf(records);
	return report;
}

// The stop has released the list, so the unused count is final now. The copy goes through the same
// published field the drain fills, under the same lock, so the live and final figures come one way.
// The hand-over takes the report's lock after the state's is released. The flag is cleared only
// once the report holds the final figures, so a throw while they are built or handed over leaves
// the source registered, and the destructor's StopStoreGeneration tries again before anything the
// source reads is gone.
void PipelineCache::FinishPrescanReport() {
	if (m_prescan == nullptr) {
		return;
	}
	auto& state = *m_generation;
	{
		std::scoped_lock lock(state.mutex);
		state.published_prescan_records = state.retained_records.Counters();
	}
	if (state.prescan_report_registered) {
		auto final_figures = BuildPrescanReport(true);
		Common::FirstRunReport::Global().SetPrescanCacheFinal(std::move(final_figures));
		state.prescan_report_registered = false;
	}
}

// The hand-off is tested first, so a run without the prescan pays one null test. The unit is made
// by the one function the preparer makes it with, from the fields of the key the lookup missed.
void PipelineCache::NoteInlineComputeMiss() {
	if (m_prescan_handoff == nullptr || !GuestGpu::IsGpuThread()) {
		return;
	}
	m_inline_compute_unit = PrescanUnitKey(m_lookup_key.hash, m_lookup_key.user_data_count,
	                                       m_lookup_key.static_state);
	m_inline_compute_in_flight = m_prescan_handoff->in_flight.Contains(m_inline_compute_unit);
	if (m_inline_miss_hook) {
		m_inline_miss_hook();
	}
}

// Only a miss that was noted has a unit, so a compile on another thread keeps no id.
void PipelineCache::NoteInlineComputeModule(uint64_t module_id) noexcept {
	if (m_inline_compute_unit != 0) {
		m_inline_compute_id = module_id;
	}
}

// The id check keeps the note of one lookup from naming another module's compile, such as one whose
// lookup missed but whose pipeline was already made. Both reads are relaxed loads of the table, so
// the GPU thread never waits on the preparer. The table is read even when the miss already said in
// flight, so a stop the miss alone accounts for can be counted apart.
Common::GpuStall::HoldCause PipelineCache::InlineComputeCause(uint64_t module_id) noexcept {
	const bool noted = m_prescan_handoff != nullptr && m_inline_compute_unit != 0 &&
	                   module_id == m_inline_compute_id;
	const bool at_scope = noted && m_prescan_handoff->in_flight.Contains(m_inline_compute_unit);
	const bool at_miss  = noted && m_inline_compute_in_flight;
	if (at_miss && !at_scope) {
		m_in_flight_at_miss_only.fetch_add(1, std::memory_order_relaxed);
	}
	ResetInlineCompute();
	return at_miss || at_scope ? Common::GpuStall::HoldCause::PrescanInFlight
	                           : Common::GpuStall::HoldCause::RequiredCompute;
}

} // namespace Libs::Graphics
