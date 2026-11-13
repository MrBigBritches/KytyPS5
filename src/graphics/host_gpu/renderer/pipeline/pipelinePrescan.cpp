#include "common/emulatorConfig.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/pipeline/generationState.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/pipeline/prescanPreparer.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "kernel/memory.h"

#include <span>
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
	config.sink = [this](DecodedModuleRecord&& record) { PostPrescanModule(std::move(record)); };
	return config;
}

void PipelineCache::EnablePrescan(StoreGen::PrescanTransportFactory make_transport) {
	if (m_prescan != nullptr) {
		return;
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
