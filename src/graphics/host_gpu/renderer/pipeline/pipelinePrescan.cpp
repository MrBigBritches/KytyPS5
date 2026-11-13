#include "common/childProcess.h"
#include "common/emulatorConfig.h"
#include "common/logging/log.h"
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
// left there are removed first.
// Returns nothing when there is no such directory.
std::optional<StoreGen::PrescanTransportFactory> HelperTransportFactory() {
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
	return StoreGen::SupervisorPrescanTransportFactory(std::move(config));
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
	if (!make_transport) {
		auto helper = HelperTransportFactory();
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
	m_prescan->Start();
	Log::WriteToConsoleAndLog("Prescan: the preparer started\n");
}

void PipelineCache::StopPrescan() {
	if (m_prescan != nullptr) {
		m_prescan->Stop();
	}
}

} // namespace Libs::Graphics
