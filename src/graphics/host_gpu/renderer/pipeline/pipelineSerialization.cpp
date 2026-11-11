#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/childProcess.h"
#include "common/emulatorConfig.h"
#include "common/firstRunReport.h"
#include "common/logging/log.h"
#include "common/parallelFor.h"
#include "common/stringUtils.h"
#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/ebootSource.h"
#include "graphics/shader/storegen/storeGenerationGate.h"
#include "graphics/shader/storegen/storeSupervisor.h"

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <ranges>

// The shader store: a header, then zstd-compressed records of everything a run created first.
// Records are only appended, so a store cut short by a crash keeps every complete record.
namespace Libs::Graphics {

namespace {

using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

// Changes with every build, so records never meet code that encodes them differently. Zero when
// the executable cannot be found.
uint64_t BuildStamp() {
	std::error_code error;
	const auto      time =
	    std::filesystem::last_write_time(Common::ExecutableStatPath(), error).time_since_epoch().count();
	return error ? 0 : static_cast<uint64_t>(time);
}

// What every record depends on besides its own fields; empty when the build is unknown.
std::vector<uint8_t> StoreHeader(const GraphicContext& graphics) {
	const auto stamp = BuildStamp();
	if (stamp == 0) {
		return {};
	}
	Writer header;
	// The jump-table mode changes recompiler output, so a store written in the other mode
	// is discarded.
	IoAll(header, stamp, Config::StructuredJumpTablesEnabled(), SpirvProfile(graphics),
	      graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize);
	return std::move(header.bytes);
}

// Why --generate-shader-store cannot run for this store, or nothing when it can.
std::optional<std::string> StoreGenerationReason(const GraphicContext&          graphics,
                                                 const std::filesystem::path& path) {
	return StoreGen::StoreGenerationAvailability(
	    {.background_warm_up_enabled = Config::BackgroundWarmUpEnabled(),
	     .shader_store_path_empty    = path.empty(),
	     .store_header_empty         = StoreHeader(graphics).empty(),
	     .eboot_path                 = StoreGen::EbootPath(),
	     .executable_path            = StoreGen::WorkerProgramPath()});
}

} // namespace

void ShaderStoreWriter::Open(const std::filesystem::path& path, std::span<const uint8_t> header,
                             uint64_t kept) {
	Common::File::CreateDirectories(path.parent_path());
	if (kept == 0) {
		if (m_file.Create(path)) {
			m_file.Write(header.data(), static_cast<uint32_t>(header.size()));
		}
	} else if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
		m_file.Truncate(kept);
		m_file.Seek(kept);
	}
	if (m_file.IsInvalid()) {
		return;
	}
	m_thread = std::jthread([this](std::stop_token stop) {
		std::unique_lock lock(m_mutex);
		// Once stop is requested the wait returns false only when the queue is empty.
		while (m_ready.wait(lock, stop, [this] { return !m_jobs.empty(); })) {
			const auto job = std::move(m_jobs.front());
			m_jobs.pop();
			lock.unlock();
			job();
			lock.lock();
		}
	});
}

void ShaderStoreWriter::Append(std::vector<uint8_t>&& record) {
	Post([this, record = std::move(record)] {
		const auto bytes = FrameStoreRecord(record);
		// A failed write leaves a partial record; the next run drops it and everything after it.
		m_file.Write(bytes.data(), static_cast<uint32_t>(bytes.size()));
	});
}

void ShaderStoreWriter::Post(Common::UniqueFunction<void>&& job) {
	{
		std::scoped_lock lock(m_mutex);
		m_jobs.push(std::move(job));
	}
	m_ready.notify_one();
}

void PipelineCache::AddComputeWarmUp(const ModuleMap&                             modules,
                                      std::vector<Common::UniqueFunction<void>>& jobs) {
	// Read once. With the flag off no warm queue exists and boot builds every compute pipeline.
	if (Config::BackgroundWarmUpEnabled()) {
		const uint32_t workers =
		    ChooseWarmUpWorkerCount(m_warm_worker_count_override, Config::GetWarmWorkers(),
		                            std::thread::hardware_concurrency());
		m_warm_compute = std::make_unique<WarmBuildSet<uint64_t, Pipeline>>(
		    workers, PipelineBuildQueueOptions {.low_priority = true});
		// The machine block of the report shows how many workers the warm-up runs.
		Common::FirstRunReport::SetWarmWorkerCount(workers);
		// The queue runs jobs in the order they are posted. A store is appended in the order the
		// title first used each shader, and an offline generator writes the likely ones first, so
		// the record order puts the builds the game needs soonest at the front. A module with
		// several records is started at its first, since Start refuses a module already pending.
		for (const auto id: m_module_record_order) {
			const auto found = modules.find(id);
			if (found != modules.end() && found->second != nullptr &&
			    found->second->info.stage == ShaderType::Compute) {
				// Boot does not wait for this one. The first lookup claims it or the queue gets to it.
				StartWarmCompute(*found->second);
			}
		}
		return;
	}
	// A compute pipeline depends only on its module.
	for (const auto* module: modules | std::views::values) {
		if (module == nullptr || module->info.stage != ShaderType::Compute) {
			continue;
		}
		jobs.emplace_back([this, module, &pipeline = m_compute_pipelines[module->id]] {
			CreatePipelineInternal(m_graphics, pipeline, *module, m_driver_cache);
		});
	}
}

void PipelineCache::WarmUp(const std::filesystem::path& path) {
	// Asked before the early return, since an empty store path is one of the reasons. With the flag
	// off none of this runs and nothing is logged.
	const bool                 generation_wanted = Config::GenerateShaderStoreEnabled();
	std::optional<std::string> generation_off_reason;
	if (generation_wanted) {
		generation_off_reason = StoreGenerationReason(m_graphics, path);
		if (generation_off_reason) {
			Log::WriteToConsoleAndLog(
			    fmt::format("Shader store generation: inactive ({})\n", *generation_off_reason));
			// The report says so too, so a run that asked for generation and did not get it shows
			// why. A run without the flag never reaches this, and has no block.
			Common::FirstRunReport::Global().SetGenerationInactive(*generation_off_reason);
		}
	}
	if (path.empty()) {
		return;
	}
	const auto header = StoreHeader(m_graphics);
	if (header.empty()) {
		return;
	}
	const auto           start = std::chrono::steady_clock::now();
	// Compiles during the replay are boot work, not waiting inside the game.
	Common::FirstRunReport::SetPhase(Common::FirstRunReport::Phase::Boot);
	std::vector<uint8_t> bytes;
	if (Common::File file;
	    Common::File::IsFileExisting(path) && file.Open(path, Common::File::Mode::Read)) {
		bytes.resize(file.Size());
		uint32_t read = 0;
		file.Read(bytes.data(), static_cast<uint32_t>(bytes.size()), &read);
		bytes.resize(read);
	}
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: compiling shaders and pipelines from {} ({:.1f} MB) with {} threads\n",
	    Common::PathToString(path), static_cast<double>(bytes.size()) / 1'000'000.0,
	    Common::ParallelThreads()));
	auto       records = std::span<const uint8_t>(bytes);
	const bool reusable =
	    records.size() >= header.size() && std::ranges::equal(records.first(header.size()), header);
	if (reusable) {
		records = records.subspan(header.size());
		while (LoadRecord(records)) {
		}
	}
	// A damaged record ends the store; the records before it stay.
	m_store.Open(path, header, reusable ? bytes.size() - records.size() : 0);

	const auto modules = MapModules();
	std::vector<Common::UniqueFunction<void>> jobs;
	jobs.reserve(m_graphics_pipelines.size() + modules.size());
	for (auto& node: m_graphics_pipelines) {
		jobs.emplace_back([this, &modules, &node] {
			CreateGraphicsPipeline(node.second, node.first, modules);
		});
	}
	AddComputeWarmUp(modules, jobs);
	std::vector<uint64_t>().swap(m_module_record_order);
	// The player's own records are queued by now, so what the generator makes comes behind them.
	if (generation_wanted && !generation_off_reason) {
		StartStoreGeneration(path, header);
	}
	Common::ParallelFor(jobs.size(), [&](size_t i) { jobs[i](); });
	// A key with a stage missing from the store keeps no pipeline.
	std::erase_if(m_graphics_pipelines,
	              [](const auto& entry) { return entry.second.pipeline == nullptr; });
	if (!m_graphics_pipelines.empty() || !m_compute_pipelines.empty()) {
		CheckpointDriverCache();
	}
	const auto warm_up_ms = std::chrono::duration<double, std::milli>(
	                            std::chrono::steady_clock::now() - start)
	                            .count();
	// Pipelines still building in the background count as loaded, since the store holds them.
	const size_t pipeline_count = m_graphics_pipelines.size() + m_compute_pipelines.size() +
	                              (m_warm_compute != nullptr ? m_warm_compute->Pending() : 0);
	Common::FirstRunReport::SetBoot({.warm_up_ms = warm_up_ms,
	                                 .programs   = static_cast<uint32_t>(m_program_cache.size()),
	                                 .modules    = static_cast<uint32_t>(modules.size() - 1),
	                                 .pipelines  = static_cast<uint32_t>(pipeline_count)});
	Common::FirstRunReport::SetPhase(Common::FirstRunReport::Phase::Play);
	Log::WriteToConsoleAndLog(fmt::format(
	    "Shader cache: loaded {} programs, {} modules, {} pipelines in {} ms\n",
	    m_program_cache.size(), modules.size() - 1, pipeline_count,
	    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() -
	                                                          start)
	        .count()));
}

bool PipelineCache::LoadRecord(std::span<const uint8_t>& records) {
	const auto unframed = UnframeStoreRecord(records);
	if (!unframed) {
		return false;
	}
	Reader     reader {unframed->raw};
	auto       type     = StoreRecord::Module;
	const auto complete = [&] { return reader.ok && reader.bytes.empty(); };
	IoAll(reader, type);
	switch (type) {
		case StoreRecord::Module: {
			DecodedModuleRecord decoded;
			DecodeModuleRecord(reader, decoded);
			if (!complete() || !LoadModuleRecord(std::move(decoded))) {
				return false;
			}
			break;
		}
		case StoreRecord::GraphicsPipeline: {
			GraphicsPipelineKey key;
			reader.Raw(&key, sizeof(key));
			if (!complete()) {
				return false;
			}
			// WarmUp creates the pipeline once every record is loaded.
			m_graphics_pipelines.try_emplace(key);
			break;
		}
		case StoreRecord::TessellationLayout: {
			uint64_t                    key = 0;
			ShaderTessellationInputInfo layout;
			IoAll(reader, key, layout);
			if (!complete()) {
				return false;
			}
			m_tess_layouts.try_emplace(key, layout);
			break;
		}
		default: return false;
	}
	records = records.subspan(unframed->frame_bytes);
	return true;
}

// Adds a module read from the store to its program, which a first record of the program creates.
bool PipelineCache::LoadModuleRecord(DecodedModuleRecord&& decoded) {
	ProgramKey key;
	FillKeyFromRecord(decoded, key);
	const auto [program, inserted] = m_program_cache.try_emplace(std::move(key));
	if (inserted &&
	    !ShaderRecompiler::IR::LoadResourcePlan(decoded.plan, program->second.resource_plan)) {
		m_program_cache.erase(program);
		return false;
	}
	// The handles come from the SPIR-V, and stored programs have no function code.
	// The SPIR-V is taken out first, so nothing reads the record after it is moved.
	const auto   spirv = std::move(decoded.spirv);
	ShaderModule module;
	FillModuleFromRecord(std::move(decoded), module);
	module.module     = CompileSPV(spirv, m_graphics.device);
	module.gpl_module = CompileSplitPixelModule(m_graphics, program->first.stage, spirv);
	m_module_record_order.push_back(module.id);
	program->second.modules.push_back(std::move(module));
	return true;
}

// Records are encoded here, since programs and modules change later.
void PipelineCache::StoreModule(const ProgramKey& key, const Program& program,
                                const ShaderModule& module, const std::vector<uint32_t>& spirv) {
	if (!m_store) {
		return;
	}
	std::vector<int32_t> formats;
	m_store.Append(
	    EncodeModuleRecord(RecordInputsOf(key, program.resource_plan, module, formats, spirv)));
}

void PipelineCache::StorePipeline(const GraphicsPipelineKey& key) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, StoreRecord::GraphicsPipeline);
	record.Raw(&key, sizeof(key));
	m_store.Append(std::move(record.bytes));
}

void PipelineCache::StoreLayout(uint64_t key, const ShaderTessellationInputInfo& layout) {
	if (!m_store) {
		return;
	}
	Writer record;
	IoAll(record, StoreRecord::TessellationLayout, key, layout);
	m_store.Append(std::move(record.bytes));
}

PipelineCache::ModuleMap PipelineCache::MapModules() const {
	ModuleMap modules {{0, nullptr}};
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
			modules.emplace(module.id, &module);
		}
	}
	return modules;
}

bool PipelineCache::CreateGraphicsPipeline(Pipeline& pipeline, const GraphicsPipelineKey& key,
                                           const ModuleMap& modules) {
	std::array<const ShaderModule*, 4> stages {};
	for (size_t i = 0; i < stages.size(); i++) {
		const auto found = modules.find(key.stage_ids[i]);
		if (found == modules.end()) {
			return false;
		}
		stages[i] = found->second;
	}
	CreatePipelineInternal(m_graphics, m_set_layouts, pipeline, key, stages, m_driver_cache,
	                       m_libraries.get());
	return true;
}

} // namespace Libs::Graphics
