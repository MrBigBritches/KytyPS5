#include "graphics/host_gpu/renderer/pipeline/prescanGraphics.h"

#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/firstRunReport.h"
#include "common/gpuStall.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/command_processor/graphicsScanShadow.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/generationState.h"
#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineParts.h"
#include "graphics/host_gpu/renderer/pipeline/prescanHeapProbe.h"
#include "graphics/host_gpu/renderer/pipeline/prescanValidate.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/moduleId.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/computeCompile.h"
#include "kernel/memory.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

namespace IR = ShaderRecompiler::IR;
using Clock  = std::chrono::steady_clock;

// The most input semantics ShaderCustomInterpolationMask reads, one per bit of its mask.
constexpr uint32_t MAX_MASK_SEMANTICS = 32;
// Microseconds in a millisecond, for the counters, which are kept in whole microseconds.
constexpr double US_PER_MS = 1000.0;
// Nanoseconds in a microsecond and in a second, for the busy time and its timeline.
constexpr uint64_t NS_PER_US     = 1000;
constexpr int64_t  NS_PER_SECOND = 1'000'000'000;

// The timeline second of a time on the NowNs clock, counted from the epoch, or nothing before the
// epoch, without one, or past the timeline's end.
std::optional<size_t> TimelineSecond(int64_t epoch_ns, int64_t at_ns) {
	if (epoch_ns == Common::FirstRunReport::NO_TIME_NS || at_ns < epoch_ns) {
		return std::nullopt;
	}
	const auto second = static_cast<uint64_t>((at_ns - epoch_ns) / NS_PER_SECOND);
	return second < GRAPHICS_PRESCAN_TIMELINE_SECONDS ? std::optional(static_cast<size_t>(second)) : std::nullopt;
}

// Reads guest memory through the backing alias, which never faults and never waits for a readback,
// so it is safe on a worker. The same reader the compute prescan's preparer uses.
bool ReadGraphicsPrescanMemory(void* /*userdata*/, uint64_t address, std::span<uint32_t> values) {
	return !values.empty() && LibKernel::Memory::TryReadBacking(address, values.data(), values.size_bytes());
}

// Raises an atomic maximum. Any thread may race on it, and a lost race retries.
void RaiseMax(std::atomic<uint64_t>& maximum, uint64_t value) {
	uint64_t seen = maximum.load(std::memory_order_relaxed);
	while (seen < value && !maximum.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
	}
}

uint64_t WholeMicros(Clock::duration duration) {
	return static_cast<uint64_t>(
	    std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::microseconds>(duration).count()));
}

// What a translate job knows of its program once it has read it.
struct GraphicsUnit {
	std::vector<uint32_t> code;
	uint64_t              code_hash       = 0;
	uint32_t              user_data_count = 0;
	ShaderPixelInputInfo  info;
	std::vector<uint32_t> static_state;
	uint32_t              push_cursor = 0;
	uint64_t              program_key = 0;
};

// Copies the program and checks it still hashes as the shader map says, since the game may have put
// another program at the address since the scan.
std::optional<GraphicsPrescanSkip> ReadCode(const ScannedDraw& draw, const ShaderMapSnapshot& snapshot,
                                            GraphicsUnit& unit) {
	unit.code.resize(snapshot.code_size_bytes / sizeof(uint32_t));
	if (unit.code.empty() || !ReadGraphicsPrescanMemory(nullptr, draw.ps_program_address, unit.code)) {
		return GraphicsPrescanSkip::CodeUnreadable;
	}
	if (XXH3_64bits(unit.code.data(), snapshot.code_size_bytes) != snapshot.hash) {
		return GraphicsPrescanSkip::CodeChanged;
	}
	unit.code_hash = snapshot.hash;
	return std::nullopt;
}

// Adds what the scanner could not know to its input info: the scratch size from the shader map and
// the custom interpolation mask from the input semantics, read as GetGraphicsPrograms reads them.
std::optional<GraphicsPrescanSkip> FinishInputInfo(const ScannedDraw& draw, const ShaderMapSnapshot& snapshot,
                                                   GraphicsUnit& unit) {
	unit.info                     = draw.ps_info;
	unit.info.scratch_size_dwords = snapshot.scratch_size_dwords;
	const uint32_t count          = std::min(snapshot.num_input_semantics, MAX_MASK_SEMANTICS);
	std::array<ShaderSemantic, MAX_MASK_SEMANTICS> semantics {};
	if (count != 0 && snapshot.input_semantics_address != 0) {
		static_assert(sizeof(ShaderSemantic) == sizeof(uint32_t), "a semantic is one dword");
		std::span<uint32_t> words(reinterpret_cast<uint32_t*>(semantics.data()), count);
		if (!ReadGraphicsPrescanMemory(nullptr, snapshot.input_semantics_address, words)) {
			return GraphicsPrescanSkip::SemanticsUnreadable;
		}
	}
	const auto read_count = snapshot.input_semantics_address != 0 ? count : 0u;
	unit.info.custom_interpolation_mask =
	    ShaderCustomInterpolationMask(std::span<const ShaderSemantic>(semantics.data(), read_count),
	                                  unit.info.input_num);
	return std::nullopt;
}

// Reads everything the GPU thread's lookup of the draw's pixel shader would key it by. The vertex
// stage's kind and the draw's user clip planes decide where the pixel shader's push data starts, as
// in GetGraphicsPrograms. The object primitive id, which the scanner skipped, adds nothing.
std::optional<GraphicsPrescanSkip> ReadGraphicsUnit(const ScannedDraw& draw, GraphicsUnit& unit) {
	const auto snapshot = ShaderTryGetMapped(draw.ps_program_address);
	if (!snapshot) {
		return GraphicsPrescanSkip::CodeUnmapped;
	}
	unit.user_data_count = draw.ps_regs.ps_regs.rsrc2.user_sgpr;
	if (unit.user_data_count > HW::UserSgprInfo::SGPRS_MAX) {
		return GraphicsPrescanSkip::TooManyUserData;
	}
	if (const auto skip = ReadCode(draw, *snapshot, unit)) {
		return skip;
	}
	if (const auto skip = FinishInputInfo(draw, *snapshot, unit)) {
		return skip;
	}
	const auto vertex = ShaderTryGetMapped(draw.vs_program_address);
	if (!vertex) {
		return GraphicsPrescanSkip::VertexUnmapped;
	}
	const bool mesh_active = VertexStageIsMesh(draw.shader_stages, vertex->type, draw.gs_lds_size);
	unit.push_cursor       = ScannedDrawPushCursor(draw, mesh_active);
	BuildStageStaticKey(unit.info, unit.static_state);
	unit.program_key = GraphicsProgramKeyHash(unit.code_hash, unit.user_data_count, unit.static_state);
	return std::nullopt;
}

// What a translate job compiled.
struct PixelPrescanBuild {
	std::optional<GraphicsPrescanSkip> skip;
	bool                               heap_image = false;
	IR::ResourcePlan                   plan;
	IR::ResourceSpecialization         specialization;
	IR::CompiledShaderInfo             info;
	std::vector<uint32_t>              spirv;
	uint64_t                           id = 0;
};

// The options GetProgram translates a pixel shader with, from the cached host values.
ShaderRecompiler::CompileOptions PixelOptions(GraphicsUnit& unit, std::span<const uint32_t> user_data,
                                              const StoreGen::HostValues& host, bool emit_debug_info) {
	ShaderRecompiler::CompileOptions options {.wave_size = unit.info.wave_size};
	options.stage                  = ShaderType::Pixel;
	options.input_info.pixel       = &unit.info;
	options.shader_hash            = unit.code_hash;
	options.user_data              = user_data;
	options.dump_ir                = false;
	options.emit_debug_info        = emit_debug_info;
	options.structured_jump_tables = host.structured_jump_tables;
	options.profile                = &host.profile;
	return options;
}

} // namespace

// What a compile needs of its program besides the options, the same for each stage. It is outside the
// anonymous namespace so the cache's vertex member can take it.
struct GraphicsStageSource {
	ShaderType                   stage = ShaderType::Pixel;
	std::span<const uint32_t>    code;
	uint64_t                     code_hash       = 0;
	uint32_t                     user_data_count = 0;
	const std::vector<uint32_t>& static_state;
	uint64_t                     shader_base = 0;
	std::span<const uint32_t>    user_data;
	uint32_t                     push_cursor = 0;
	/// GraphicsProgramKeyHash of the code hash, user data count and static key, for the second dedupe.
	uint64_t program_key = 0;
};

namespace {

// Specializes a plan with a walk that never faults and never waits. The image heap belongs to the
// GPU thread, so the walk gets a probe that declines every table, as the compute preparer's does, and
// a draw whose heap the GPU thread keeps resident is counted and built anyway.
std::optional<GraphicsPrescanSkip> MaterializeGraphics(const GraphicsStageSource& source, const IR::ResourcePlan& plan,
                                                       PixelPrescanBuild& build) {
	PrescanHeapProbe     heap;
	const IR::SrtRuntime runtime {.user_data                  = source.user_data,
	                              .shader_base                = source.shader_base,
	                              .read_specialization_memory = ReadGraphicsPrescanMemory,
	                              .image_heap                 = &heap};
	IR::ResourceSnapshot resources;
	uint32_t             skipped = 0;
	const bool built = IR::MaterializeResources(plan, runtime, resources, build.specialization,
	                                            {.non_faulting = true, .skipped_loads = &skipped});
	build.heap_image = heap.Offered();
	if (skipped != 0) {
		return GraphicsPrescanSkip::MaterializeFaulted;
	}
	if (!built) {
		return GraphicsPrescanSkip::MaterializeFailed;
	}
	return ValidateGraphicsSpecialization(source.stage, plan, build.specialization);
}

uint64_t SpecializationHash(const IR::ResourceSpecialization& specialization);

// The keys of the second dedupe. A plan belongs to a stage's program key, and a module to that with
// the push start the compile is given and the specialization, which decide its id.
uint64_t PlanKey(const GraphicsStageSource& source) {
	return source.program_key ^ (static_cast<uint64_t>(source.stage) << 56u);
}

uint64_t BuildTupleKey(const GraphicsStageSource& source, const IR::ResourceSpecialization& specialization) {
	const std::array<uint64_t, 4> words {static_cast<uint64_t>(source.stage), source.program_key, source.push_cursor,
	                                     SpecializationHash(specialization)};
	return XXH3_64bits(words.data(), sizeof(words));
}

// Specializes with the plan an earlier job of the program key kept, before anything is translated,
// and claims the module. A job whose module was claimed already stops as deduped_late.
std::optional<GraphicsPrescanSkip> ClaimWithKeptPlan(const GraphicsStageSource& source, GraphicsBuildDedupe::KeptPlan& kept,
                                                     GraphicsBuildDedupe& dedupe, PixelPrescanBuild& build) {
	std::optional<GraphicsPrescanSkip> skip;
	{
		std::scoped_lock lock(kept.mutex);
		skip = MaterializeGraphics(source, kept.plan, build);
	}
	if (!skip && !dedupe.Claim(BuildTupleKey(source, build.specialization))) {
		skip = GraphicsPrescanSkip::DedupedLate;
	}
	return skip;
}

// Prepares and translates, rejectably, and extracts the plan, and a second copy to keep when the
// program key has none kept yet.
std::optional<GraphicsPrescanSkip> TranslateStage(const GraphicsStageSource& source,
                                                  const ShaderRecompiler::CompileOptions& options, GraphicsRejectTally& rejects,
                                                  std::optional<ShaderRecompiler::TranslateResult>& translated,
                                                  PixelPrescanBuild& build, std::optional<IR::ResourcePlan>& to_keep) {
	bool       has_calls     = false;
	const bool translated_ok = RunRejectable(rejects, [&] {
		const auto prepared = ShaderRecompiler::PrepareShaderSource(source.code, options);
		has_calls           = prepared.unsupported_call_pc != UINT32_MAX || !prepared.calls.empty();
		if (!has_calls) {
			translated = ShaderRecompiler::TranslateProgram(prepared.decoded, options);
			build.plan = IR::ExtractResourcePlan(translated->program);
			if (to_keep) {
				to_keep = IR::ExtractResourcePlan(translated->program);
			}
		}
	});
	if (!translated_ok) {
		return GraphicsPrescanSkip::RecompilerRejected;
	}
	return has_calls ? std::optional(GraphicsPrescanSkip::HasCalls) : std::nullopt;
}

// Translates and compiles a program as GetProgram and CompileModule do, after every check that stands
// in for one of their EXITs, and works out the module id the GPU thread would compute. Translation and
// compilation run rejectable, so an EXIT the checks missed drops the job, but the walk between them
// does not, since it runs generated code. A program key seen before is specialized with its kept plan
// and its module claimed before translation, so a repeat costs one walk. A new one is claimed after
// its own walk, and keeps its plan for the next.
PixelPrescanBuild CompileStage(const GraphicsStageSource& source, const ShaderRecompiler::CompileOptions& options,
                               GraphicsRejectTally& rejects, GraphicsBuildDedupe& dedupe) {
	PixelPrescanBuild                                build;
	std::optional<ShaderRecompiler::TranslateResult> translated;
	auto*                                            kept = dedupe.FindPlan(PlanKey(source));
	if (kept != nullptr && (build.skip = ClaimWithKeptPlan(source, *kept, dedupe, build))) {
		return build;
	}
	std::optional<IR::ResourcePlan> to_keep;
	if (kept == nullptr) {
		to_keep.emplace();
	}
	if ((build.skip = TranslateStage(source, options, rejects, translated, build, to_keep))) {
		return build;
	}
	if (kept == nullptr) {
		if ((build.skip = MaterializeGraphics(source, build.plan, build))) {
			return build;
		}
		dedupe.KeepPlan(PlanKey(source), std::move(*to_keep));
		if (!dedupe.Claim(BuildTupleKey(source, build.specialization))) {
			build.skip = GraphicsPrescanSkip::DedupedLate;
			return build;
		}
	}
	const auto& listed = translated->program.info;
	if (StoreGen::ProgramExceedsSpecialization(listed.buffers.size(), listed.images.size(),
	                                           listed.samplers.size(), build.specialization)) {
		build.skip = GraphicsPrescanSkip::ExceedsSpecialization;
		return build;
	}
	const bool compiled = RunRejectable(rejects, [&] {
		auto result = ShaderRecompiler::CompileProgram(std::move(*translated), options, build.specialization,
		                                               source.push_cursor);
		build.info  = std::move(result.program).TakeCompiledInfo();
		build.spirv = std::move(result.spirv);
	});
	if (!compiled) {
		build.skip = GraphicsPrescanSkip::RecompilerRejected;
		return build;
	}
	build.id = ShaderModuleId(source.stage, source.code_hash, source.user_data_count, source.static_state,
	                             build.info.bindings.push_data_start_dword, build.specialization, {});
	return build;
}

// The pixel shader's program as the compile and the record see it.
GraphicsStageSource PixelSource(const ScannedDraw& draw, const GraphicsUnit& unit) {
	return {.stage           = ShaderType::Pixel,
	        .code            = unit.code,
	        .code_hash       = unit.code_hash,
	        .user_data_count = unit.user_data_count,
	        .static_state    = unit.static_state,
	        .shader_base     = draw.ps_program_address,
	        .user_data       = std::span<const uint32_t>(draw.ps_regs.ps_user_sgpr.value, unit.user_data_count),
	        .push_cursor     = unit.push_cursor,
	        .program_key     = unit.program_key};
}

// The pixel shader's compile, with the options GetProgram gives a pixel shader.
PixelPrescanBuild CompileGraphicsUnit(const ScannedDraw& draw, GraphicsUnit& unit,
                                      const StoreGen::HostValues& host, bool emit_debug_info,
                                      GraphicsRejectTally& rejects, GraphicsBuildDedupe& dedupe) {
	const auto source = PixelSource(draw, unit);
	return CompileStage(source, PixelOptions(unit, source.user_data, host, emit_debug_info), rejects, dedupe);
}

// The record the GPU thread's intake decodes, made directly instead of encoded and decoded again.
DecodedModuleRecord MakeRecord(const GraphicsStageSource& source, const PixelPrescanBuild& build,
                               std::span<const int32_t> vertex_formats) {
	DecodedModuleRecord record;
	record.stage           = source.stage;
	record.hash            = source.code_hash;
	record.user_data_count = source.user_data_count;
	record.static_state    = source.static_state;
	record.plan            = IR::SaveResourcePlan(build.plan);
	record.id              = build.id;
	record.info            = build.info;
	record.specialization  = build.specialization;
	record.vertex_formats.assign(vertex_formats.begin(), vertex_formats.end());
	record.spirv           = build.spirv;
	return record;
}

// The store bytes of the module, which StoreModule would write, so its first use stores it.
std::vector<uint8_t> EncodeRecord(const GraphicsStageSource& source, const PixelPrescanBuild& build,
                                  std::span<const int32_t> vertex_formats) {
	return EncodeModuleRecord({.stage           = source.stage,
	                           .hash            = source.code_hash,
	                           .user_data_count = source.user_data_count,
	                           .static_state    = source.static_state,
	                           .plan            = build.plan,
	                           .id              = build.id,
	                           .info            = build.info,
	                           .specialization  = build.specialization,
	                           .vertex_formats  = vertex_formats,
	                           .rect_parameters = {},
	                           .mesh_task       = false,
	                           .spirv           = build.spirv});
}

// The bucket of a histogram by powers of two that a whole number of milliseconds falls in.
size_t MillisecondBucket(uint64_t micros) {
	const auto millis = static_cast<uint64_t>(static_cast<double>(micros) / US_PER_MS);
	return std::min<size_t>(std::bit_width(millis), GRAPHICS_PRESCAN_HISTOGRAM_BUCKETS - 1);
}

// The nearest-rank percentile of the posted-after-offer histogram, as the top of the bucket the rank
// falls in, capped at the maximum, so it is an upper bound.
double HistogramPercentileMs(const GraphicsPrescanCounters& counters, double fraction) {
	uint64_t total = 0;
	for (const auto& bucket: counters.posted_after_offer_ms) {
		total += bucket.load(std::memory_order_relaxed);
	}
	const double max_ms = static_cast<double>(counters.posted_after_offer_max_us.load()) / US_PER_MS;
	if (total == 0) {
		return 0.0;
	}
	const auto rank = std::max<uint64_t>(1, static_cast<uint64_t>(fraction * static_cast<double>(total) + 0.999999));
	uint64_t   seen = 0;
	for (size_t bucket = 0; bucket + 1 < GRAPHICS_PRESCAN_HISTOGRAM_BUCKETS; ++bucket) {
		seen += counters.posted_after_offer_ms[bucket].load(std::memory_order_relaxed);
		if (seen >= rank) {
			return std::min(static_cast<double>((uint64_t {1} << bucket) - 1), max_ms);
		}
	}
	return max_ms;
}

double ToMs(const std::atomic<uint64_t>& micros) {
	return static_cast<double>(micros.load(std::memory_order_relaxed)) / US_PER_MS;
}

double Count(const std::atomic<uint64_t>& value) {
	return static_cast<double>(value.load(std::memory_order_relaxed));
}

// The names the report gives the skip and miss classes.
const char* SkipName(GraphicsPrescanSkip skip) {
	switch (skip) {
		case GraphicsPrescanSkip::CodeUnmapped: return "code_unmapped";
		case GraphicsPrescanSkip::TooManyUserData: return "too_many_user_data";
		case GraphicsPrescanSkip::CodeUnreadable: return "code_unreadable";
		case GraphicsPrescanSkip::CodeChanged: return "code_changed";
		case GraphicsPrescanSkip::SemanticsUnreadable: return "semantics_unreadable";
		case GraphicsPrescanSkip::VertexUnmapped: return "vertex_unmapped";
		case GraphicsPrescanSkip::HasCalls: return "has_calls";
		case GraphicsPrescanSkip::MaterializeFaulted: return "materialize_faulted";
		case GraphicsPrescanSkip::MaterializeFailed: return "materialize_failed";
		case GraphicsPrescanSkip::InvalidDescriptor: return "invalid_descriptor";
		case GraphicsPrescanSkip::ExceedsSpecialization: return "exceeds_specialization";
		case GraphicsPrescanSkip::VertexIndirect: return "vertex_indirect";
		case GraphicsPrescanSkip::VertexMesh: return "vertex_mesh";
		case GraphicsPrescanSkip::VertexMetadata: return "vertex_metadata";
		case GraphicsPrescanSkip::VertexTables: return "vertex_tables";
		case GraphicsPrescanSkip::InvalidBufferDescriptor: return "invalid_buffer_descriptor";
		case GraphicsPrescanSkip::InvalidVertexFormat: return "invalid_vertex_format";
		case GraphicsPrescanSkip::InvalidVertexFetch: return "invalid_vertex_fetch";
		case GraphicsPrescanSkip::RecompilerRejected: return "recompiler_rejected";
		case GraphicsPrescanSkip::DedupedLate: return "deduped_late";
		case GraphicsPrescanSkip::VertexNoPixel: return "vertex_no_pixel";
		case GraphicsPrescanSkip::Count: break;
	}
	return "unknown";
}

const char* PrebuildSkipName(GraphicsPrebuildSkip skip) {
	switch (skip) {
		case GraphicsPrebuildSkip::LibrariesOff: return "libraries_off";
		case GraphicsPrebuildSkip::NoHeapLayout: return "no_heap_layout";
		case GraphicsPrebuildSkip::NoSplitModule: return "no_split_module";
		case GraphicsPrebuildSkip::LayoutLimits: return "layout_limits";
		case GraphicsPrebuildSkip::AlreadyThere: return "already_there";
		case GraphicsPrebuildSkip::Count: break;
	}
	return "unknown";
}

const char* MissName(GraphicsPrescanMiss miss) {
	switch (miss) {
		case GraphicsPrescanMiss::Late: return "late";
		case GraphicsPrescanMiss::IdDiffers: return "id_differs";
		case GraphicsPrescanMiss::Unpredicted: return "unpredicted";
		case GraphicsPrescanMiss::Count: break;
	}
	return "unknown";
}

// A short fingerprint of a specialization, the module id of an otherwise empty key, which hashes every
// field the way the real id does.
uint64_t SpecializationHash(const IR::ResourceSpecialization& specialization) {
	static const std::vector<uint32_t> no_state;
	return ShaderModuleId(ShaderType::Pixel, 0, 0, no_state, 0, specialization, {});
}

size_t SpecializationSize(const IR::ResourceSpecialization& specialization) {
	return specialization.buffers.size() + specialization.images.size() + specialization.samplers.size();
}

// The fields of one buffer entry that differ, by name.
std::string BufferFieldsDiffer(const IR::ResourceSpecialization::Buffer& gpu,
                               const IR::ResourceSpecialization::Buffer& built) {
	std::string fields;
	const auto  add = [&](bool differs, const char* name) {
		if (differs) {
			fields += fields.empty() ? name : std::string("+") + name;
		}
	};
	add(gpu.packed_stride != built.packed_stride, "stride");
	add(gpu.descriptor_format != built.descriptor_format, "format");
	add(gpu.descriptor_swizzle != built.descriptor_swizzle, "swizzle");
	add(gpu.zero_stride_oob != built.zero_stride_oob, "zero_stride_oob");
	add(gpu.scalar_empty != built.scalar_empty, "scalar_empty");
	add(gpu.indirect_root != built.indirect_root || gpu.indirect_mapping_offset != built.indirect_mapping_offset ||
	        gpu.indirect_search_iterations != built.indirect_search_iterations,
	    "indirect");
	return fields;
}

// The fields of one image entry that differ, by name.
std::string ImageFieldsDiffer(const IR::ResourceSpecialization::Image& gpu, const IR::ResourceSpecialization::Image& built) {
	std::string fields;
	const auto  add = [&](bool differs, const char* name) {
		if (differs) {
			fields += fields.empty() ? name : std::string("+") + name;
		}
	};
	add(gpu.numeric_class != built.numeric_class, "numeric_class");
	add(gpu.dimension != built.dimension, "dimension");
	add(gpu.mip_count != built.mip_count, "mips");
	add(gpu.conversion_format != built.conversion_format, "conversion");
	add(gpu.shader_swizzle != built.shader_swizzle, "swizzle");
	add(gpu.srgb_decode != built.srgb_decode, "srgb");
	add(gpu.cube != built.cube, "cube");
	add(gpu.skip_degamma != built.skip_degamma, "degamma");
	add(gpu.fmask != built.fmask, "fmask");
	add(gpu.heap != built.heap, "heap");
	add(gpu.indirect_root != built.indirect_root || gpu.indirect_mapping_offset != built.indirect_mapping_offset ||
	        gpu.indirect_search_iterations != built.indirect_search_iterations,
	    "indirect");
	return fields;
}

// Which entry kind differs first between the GPU's specialization and a built one: a count, or the
// first entry of a kind whose fields differ, named field by field.
std::string SpecializationDiff(const IR::ResourceSpecialization& gpu, const IR::ResourceSpecialization& built) {
	if (gpu.buffers.size() != built.buffers.size() || gpu.images.size() != built.images.size() ||
	    gpu.samplers.size() != built.samplers.size()) {
		return fmt::format("counts b{}/{} i{}/{} s{}/{}", gpu.buffers.size(), built.buffers.size(), gpu.images.size(),
		                   built.images.size(), gpu.samplers.size(), built.samplers.size());
	}
	for (size_t index = 0; index < gpu.buffers.size(); ++index) {
		if (!(gpu.buffers[index] == built.buffers[index])) {
			return fmt::format("buffer[{}] {}", index, BufferFieldsDiffer(gpu.buffers[index], built.buffers[index]));
		}
	}
	for (size_t index = 0; index < gpu.images.size(); ++index) {
		if (!(gpu.images[index] == built.images[index])) {
			return fmt::format("image[{}] {}", index, ImageFieldsDiffer(gpu.images[index], built.images[index]));
		}
	}
	for (size_t index = 0; index < gpu.samplers.size(); ++index) {
		if (!(gpu.samplers[index] == built.samplers[index])) {
			return fmt::format("sampler[{}] indirect", index);
		}
	}
	return "same";
}

// One log line of an id_differs miss: the program key, the GPU module's push start and specialization,
// then each module the prescan built for the key with how its specialization differs.
std::string IdDiffersLine(uint64_t program_key, const ShaderModule& module, std::span<const GraphicsPrescanVariant> built) {
	std::string line = fmt::format("Graphics prescan id_differs: program_key={:016x} gpu id={:016x} push_start={} "
	                               "spec={:08x}/{}",
	                               program_key, module.id, module.info.bindings.push_data_start_dword,
	                               static_cast<uint32_t>(SpecializationHash(module.specialization)),
	                               SpecializationSize(module.specialization));
	for (const auto& variant: built) {
		line += fmt::format(" | built id={:016x} push_start={} spec={:08x}/{} diff={}", variant.id, variant.push_start,
		                    static_cast<uint32_t>(SpecializationHash(variant.specialization)),
		                    SpecializationSize(variant.specialization),
		                    SpecializationDiff(module.specialization, variant.specialization));
	}
	if (built.empty()) {
		line += " | no built module kept";
	}
	return line + "\n";
}

// The vertex stage's groups of the block: its modules, misses and skips, and the pre-rasterization
// libraries built ahead, with the library cache's prebuilt counts for that part.
std::vector<Common::FirstRunReport::PrescanGraphicsReport::Group>
VertexGroups(const GraphicsVertexCounters& counts, const PrebuiltLibraryCounts& pre_raster) {
	using Group          = Common::FirstRunReport::PrescanGraphicsReport::Group;
	const auto number    = [](const std::atomic<uint64_t>& value) { return static_cast<double>(value.load()); };
	const auto us_to_ms  = [](const std::atomic<uint64_t>& value) { return static_cast<double>(value.load()) / US_PER_MS; };
	const double adopted = number(counts.adopted);
	const double used    = number(counts.used);
	Group modules {"vs_modules",
	               {{"posted", number(counts.posted)},
	                {"translate_ms", us_to_ms(counts.translate_us)},
	                {"adopted", adopted},
	                {"used", used},
	                {"unused", std::max(0.0, adopted - used)},
	                {"late_duplicates", number(counts.late_duplicates)}}};
	Group misses {"vs_misses",
	              {{"id_differs_push_start", number(counts.id_differs_push_start)},
	               {"id_differs_specialization", number(counts.id_differs_specialization)},
	               {"id_differs_other", number(counts.id_differs_other)}}};
	for (size_t miss = 0; miss < counts.misses.size(); ++miss) {
		misses.figures.emplace_back(MissName(static_cast<GraphicsPrescanMiss>(miss)), number(counts.misses[miss]));
	}
	Group skipped {"vs_skipped", {}};
	for (size_t reason = 0; reason < counts.skipped.size(); ++reason) {
		skipped.figures.emplace_back(SkipName(static_cast<GraphicsPrescanSkip>(reason)), number(counts.skipped[reason]));
	}
	Group libraries {"pre_raster",
	                 {{"prebuilt", number(counts.prebuilt)},
	                  {"prebuild_ms", us_to_ms(counts.prebuild_us)},
	                  {"prebuild_in_scope", number(counts.prebuild_in_scope)},
	                  {"prebuild_in_hold", number(counts.prebuild_in_hold)},
	                  {"provoking_unsupported", number(counts.prebuild_provoking_unsupported)},
	                  {"deduped_late", number(counts.prebuild_deduped_late)},
	                  {"first_hits", static_cast<double>(pre_raster.first_hits)},
	                  {"waits", static_cast<double>(pre_raster.waits)},
	                  {"wait_ms", pre_raster.wait_ms},
	                  {"wait_max_ms", pre_raster.wait_max_ms},
	                  {"key_differs", static_cast<double>(pre_raster.key_differs)}}};
	for (size_t reason = 0; reason < counts.prebuild_skipped.size(); ++reason) {
		libraries.figures.emplace_back(std::string("skipped_") + PrebuildSkipName(static_cast<GraphicsPrebuildSkip>(reason)),
		                               number(counts.prebuild_skipped[reason]));
	}
	return {std::move(modules), std::move(misses), std::move(skipped), std::move(libraries)};
}

// The recompiler EXITs the workers caught, one figure per source line named <file>_<line> by the
// file's base name, the first few lines only, after the count of distinct lines.
Common::FirstRunReport::PrescanGraphicsReport::Group RejectedGroup(const GraphicsRejectTally& tally) {
	constexpr size_t max_sites = 8;
	const auto       sites     = tally.Sites();
	Common::FirstRunReport::PrescanGraphicsReport::Group group {"recompiler_rejected",
	                                                            {{"sites", static_cast<double>(sites.size())}}};
	for (size_t index = 0; index < std::min(sites.size(), max_sites); ++index) {
		auto name = sites[index].where;
		if (const auto slash = name.find_last_of("/\\"); slash != std::string::npos) {
			name.erase(0, slash + 1);
		}
		std::ranges::replace(name, ':', '_');
		group.figures.emplace_back(std::move(name), static_cast<double>(sites[index].count));
	}
	return group;
}

// Whether a stall scope or a hold of guest audio overlapped a prebuild, sampled as the compute
// prescan's builds sample it, without the stall mutex.
struct StallMarks {
	bool     scope_active = false;
	bool     hold_active  = false;
	uint64_t scopes       = 0;
	uint64_t holds        = 0;

	static StallMarks Take() {
		StallMarks marks;
		marks.hold_active  = Common::GpuStall::HoldInProgress();
		marks.holds        = Common::GpuStall::HoldCount();
		marks.scopes       = Common::GpuStall::ScopeCount();
		marks.scope_active = Common::GpuStall::AnyScopeActive();
		return marks;
	}
};

} // namespace

// The count is taken with a compare and swap, so two jobs never both take the last slot. A job that
// finds the cap full sleeps a step and looks again rather than waiting on a condition, which keeps
// the release a plain decrement. The cap is the governor's at each look, so a throttle signal stops
// new starts at once and jobs already running finish.
GraphicsJobSlot::GraphicsJobSlot(std::atomic<uint32_t>& running, GraphicsPrescanCounters& counters,
                                 const PipelineBuildQueue& queue)
    : m_running(running), m_counters(counters) {
	// The epoch is read before any mutex, so the report's lock is never taken inside one.
	const int64_t epoch_ns = Common::FirstRunReport::Global().EpochNs();
	bool          capped   = false;
	bool          paused   = false;
	while (!queue.Stopping()) {
		const uint32_t cap  = counters.cap_governor.CapAt(Common::FirstRunReport::NowNs(),
		                                                  Common::GpuStall::LastThrottleSignalNs(), epoch_ns);
		uint32_t       seen = m_running.load(std::memory_order_relaxed);
		if (seen < cap && m_running.compare_exchange_weak(seen, seen + 1, std::memory_order_acq_rel)) {
			m_held = true;
			break;
		}
		if (seen >= cap) {
			capped = true;
			paused = paused || cap < GRAPHICS_PRESCAN_MAX_RUNNING_JOBS;
			std::this_thread::sleep_for(GRAPHICS_PRESCAN_ADMIT_STEP);
		}
	}
	counters.capped_waits.fetch_add(capped ? 1 : 0, std::memory_order_relaxed);
	counters.paused_starts.fetch_add(paused ? 1 : 0, std::memory_order_relaxed);
	if (!m_held) {
		return;
	}
	const int64_t    now_ns   = Common::FirstRunReport::NowNs();
	std::scoped_lock lock(counters.busy_mutex);
	if (counters.busy_jobs++ == 0) {
		counters.busy_since    = Clock::now();
		counters.busy_since_ns = now_ns;
	}
	if (const auto second = TimelineSecond(epoch_ns, now_ns)) {
		auto& peak = counters.timeline_peak[*second];
		peak       = static_cast<uint8_t>(std::max<uint32_t>(peak, std::min<uint32_t>(counters.busy_jobs, UINT8_MAX)));
	}
}

// A slot that was never held, because the queue stopped, gives nothing back.
GraphicsJobSlot::~GraphicsJobSlot() {
	if (!m_held) {
		return;
	}
	const int64_t epoch_ns = Common::FirstRunReport::Global().EpochNs();
	const int64_t now_ns   = Common::FirstRunReport::NowNs();
	{
		std::scoped_lock lock(m_counters.busy_mutex);
		if (--m_counters.busy_jobs == 0) {
			m_counters.busy_ns += WholeMicros(Clock::now() - m_counters.busy_since) * NS_PER_US;
			AddBusyToTimeline(m_counters, epoch_ns, m_counters.busy_since_ns, now_ns);
		}
	}
	m_running.fetch_sub(1, std::memory_order_acq_rel);
}

namespace {
// Adds an interval on the NowNs clock to the timeline seconds it covers from the epoch, split at each
// second boundary, and marks those seconds in `peak` when given. An interval that began before the
// epoch counts from it, and the part past the timeline's end is dropped.
void AddIntervalToSeconds(std::array<uint32_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS>& seconds_us,
                          std::array<uint8_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS>* peak, int64_t epoch_ns,
                          int64_t begin_ns, int64_t end_ns) {
	if (epoch_ns == Common::FirstRunReport::NO_TIME_NS || end_ns <= begin_ns) {
		return;
	}
	int64_t at_ns = std::max(begin_ns, epoch_ns);
	while (at_ns < end_ns) {
		const auto second = TimelineSecond(epoch_ns, at_ns);
		if (!second) {
			return;
		}
		const int64_t second_end_ns = epoch_ns + static_cast<int64_t>(*second + 1) * NS_PER_SECOND;
		const int64_t piece_end_ns  = std::min(end_ns, second_end_ns);
		seconds_us[*second] += static_cast<uint32_t>((piece_end_ns - at_ns) / static_cast<int64_t>(NS_PER_US));
		if (peak != nullptr) {
			(*peak)[*second] = std::max<uint8_t>((*peak)[*second], 1);
		}
		at_ns = piece_end_ns;
	}
}
} // namespace

void AddBusyToTimeline(GraphicsPrescanCounters& counters, int64_t epoch_ns, int64_t begin_ns, int64_t end_ns) {
	AddIntervalToSeconds(counters.timeline_busy_us, &counters.timeline_peak, epoch_ns, begin_ns, end_ns);
}

// A new signal adds the part of its window that the window in force did not cover yet, so the reduced
// time is the union of the windows, and so is its timeline.
uint32_t GraphicsCapGovernor::CapAt(int64_t now_ns, int64_t signal_ns, int64_t epoch_ns) {
	std::scoped_lock lock(m_mutex);
	if (signal_ns != 0 && signal_ns != m_seen_signal_ns) {
		m_seen_signal_ns         = signal_ns;
		const int64_t new_until  = signal_ns + GRAPHICS_PRESCAN_REDUCED_CAP_NS;
		const int64_t added_from = std::max(signal_ns, m_reduced_until);
		if (new_until > added_from) {
			m_scheduled_ns += new_until - added_from;
			AddIntervalToSeconds(m_timeline_us, nullptr, epoch_ns, added_from, new_until);
		}
		m_reduced_until = std::max(m_reduced_until, new_until);
		m_triggers++;
	}
	return now_ns < m_reduced_until ? GRAPHICS_PRESCAN_REDUCED_RUNNING_JOBS : GRAPHICS_PRESCAN_MAX_RUNNING_JOBS;
}

uint64_t GraphicsCapGovernor::Triggers() const {
	std::scoped_lock lock(m_mutex);
	return m_triggers;
}

// The part of the window still ahead of now is not reduced time yet.
double GraphicsCapGovernor::ReducedMs(int64_t now_ns) const {
	constexpr double ns_per_ms = 1e6;
	std::scoped_lock lock(m_mutex);
	const int64_t    ahead_ns = std::clamp<int64_t>(m_reduced_until - now_ns, 0, m_scheduled_ns);
	return static_cast<double>(m_scheduled_ns - ahead_ns) / ns_per_ms;
}

std::array<uint32_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS> GraphicsCapGovernor::ReducedTimelineUs() const {
	std::scoped_lock lock(m_mutex);
	return m_timeline_us;
}

// The open interval is added to a copy, so reading the timeline changes nothing.
Common::FirstRunReport::PrescanGraphicsReport::Group GraphicsBusyTimeline(const GraphicsPrescanCounters& counters,
                                                                          int64_t epoch_ns, int64_t now_ns) {
	Common::FirstRunReport::PrescanGraphicsReport::Group group {"busy_timeline", {}};
	const auto last = TimelineSecond(epoch_ns, now_ns);
	if (epoch_ns == Common::FirstRunReport::NO_TIME_NS || now_ns < epoch_ns) {
		return group;
	}
	const size_t seconds = last ? *last + 1 : GRAPHICS_PRESCAN_TIMELINE_SECONDS;
	GraphicsPrescanCounters copy;
	{
		std::scoped_lock lock(counters.busy_mutex);
		copy.timeline_busy_us = counters.timeline_busy_us;
		copy.timeline_peak    = counters.timeline_peak;
		if (counters.busy_jobs != 0) {
			AddBusyToTimeline(copy, epoch_ns, counters.busy_since_ns, now_ns);
		}
	}
	const auto          reduced_us = counters.cap_governor.ReducedTimelineUs();
	std::vector<double> busy_ms(seconds);
	std::vector<double> peak(seconds);
	std::vector<double> reduced_ms(seconds);
	for (size_t second = 0; second < seconds; second++) {
		busy_ms[second]    = static_cast<double>(copy.timeline_busy_us[second]) / US_PER_MS;
		peak[second]       = static_cast<double>(copy.timeline_peak[second]);
		reduced_ms[second] = static_cast<double>(reduced_us[second]) / US_PER_MS;
	}
	group.series.emplace_back("busy_ms", std::move(busy_ms));
	group.series.emplace_back("peak_running", std::move(peak));
	group.series.emplace_back("reduced_ms", std::move(reduced_ms));
	return group;
}

// The interval still open when it is read counts up to now.
double GraphicsBusyFraction(const GraphicsPrescanCounters& counters) {
	const auto       now = Clock::now();
	std::scoped_lock lock(counters.busy_mutex);
	uint64_t         busy_ns = counters.busy_ns;
	if (counters.busy_jobs != 0) {
		busy_ns += WholeMicros(now - counters.busy_since) * NS_PER_US;
	}
	const uint64_t total_ns = WholeMicros(now - counters.started) * NS_PER_US;
	return total_ns == 0 ? 0.0 : static_cast<double>(busy_ns) / static_cast<double>(total_ns);
}

// The specialization is the one the walk read at worker time, which may be staler than the draw's, so
// each check is the predicate the compile itself EXITs on. ADD_TID is judged over every buffer, which
// is wider than info collection, which judges only the buffers an operation reaches.
std::optional<GraphicsPrescanSkip>
ValidateGraphicsSpecialization(ShaderType stage, const ShaderRecompiler::IR::ResourcePlan& plan,
                               const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	for (const auto& buffer: specialization.buffers) {
		if (ShaderRecompiler::IR::BufferAddTidUnsupported(stage, buffer.packed_stride)) {
			return GraphicsPrescanSkip::InvalidBufferDescriptor;
		}
	}
	if (FindInvalidDescriptor(plan, specialization)) {
		return GraphicsPrescanSkip::InvalidDescriptor;
	}
	return std::nullopt;
}

void GraphicsRejectTally::Note(const char* file, int line, const std::string& text) {
	auto where = fmt::format("{}:{}", file != nullptr ? file : "?", line);
	std::scoped_lock lock(m_mutex);
	const auto site = std::ranges::find(m_sites, where, &Site::where);
	if (site != m_sites.end()) {
		site->count++;
		return;
	}
	Log::Write({}, fmt::format("Graphics prescan recompiler_rejected at {}: {}\n", where, text));
	m_sites.push_back({std::move(where), text, 1});
}

std::vector<GraphicsRejectTally::Site> GraphicsRejectTally::Sites() const {
	std::scoped_lock lock(m_mutex);
	return m_sites;
}

GraphicsBuildDedupe::KeptPlan* GraphicsBuildDedupe::FindPlan(uint64_t plan_key) {
	std::scoped_lock lock(m_mutex);
	const auto       found = m_plans.find(plan_key);
	return found != m_plans.end() ? found->second.get() : nullptr;
}

void GraphicsBuildDedupe::KeepPlan(uint64_t plan_key, ShaderRecompiler::IR::ResourcePlan&& plan) {
	auto kept  = std::make_unique<KeptPlan>();
	kept->plan = std::move(plan);
	std::scoped_lock lock(m_mutex);
	m_plans.try_emplace(plan_key, std::move(kept));
}

bool GraphicsBuildDedupe::Claim(uint64_t key) {
	std::scoped_lock lock(m_mutex);
	return m_claimed.insert(key).second;
}

// The counts go up before the post, so a worker that starts the job at once never takes them below
// zero, and the total is reserved with one increment, so two posters never both take the last place.
bool PostBounded(PipelineBuildQueue& queue, GraphicsQueueBound& bound, PipelineBuildQueue::Lane lane,
                 Common::UniqueFunction<void> work) {
	// A Head job waits at the head of the front lane, so it is counted with the front lane.
	const auto index = lane == PipelineBuildQueue::Lane::Back ? 1u : 0u;
	if (bound.pending.fetch_add(1, std::memory_order_relaxed) >= GRAPHICS_PRESCAN_QUEUE_CAP) {
		bound.pending.fetch_sub(1, std::memory_order_relaxed);
		bound.dropped_full.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const uint32_t depth = bound.depth[index].fetch_add(1, std::memory_order_relaxed) + 1;
	RaiseMax(bound.depth_max[index], depth);
	auto job = queue.Post(
	    [&bound, index, work = std::move(work)]() mutable {
		    bound.depth[index].fetch_sub(1, std::memory_order_relaxed);
		    bound.pending.fetch_sub(1, std::memory_order_relaxed);
		    work();
	    },
	    lane);
	if (job == nullptr) {
		bound.depth[index].fetch_sub(1, std::memory_order_relaxed);
		bound.pending.fetch_sub(1, std::memory_order_relaxed);
		return false;
	}
	return true;
}

// Only ExitRejected is caught, so any other exception still reaches the queue as before.
bool RunRejectable(GraphicsRejectTally& tally, const std::function<void()>& step) {
	try {
		const Common::ExitRejectScope scope;
		step();
		return true;
	} catch (const Common::ExitRejected& rejected) {
		tally.Note(rejected.file, rejected.line, rejected.text);
		return false;
	}
}

uint64_t GraphicsProgramKeyHash(uint64_t code_hash, uint32_t user_data_count,
                                std::span<const uint32_t> static_state) {
	// The user data count sits in the top byte of the seed, which the code hash's own bits fill below.
	constexpr uint32_t COUNT_SHIFT = 56;
	return XXH3_64bits_withSeed(static_state.data(), static_state.size_bytes(),
	                            code_hash ^ (uint64_t {user_data_count} << COUNT_SHIFT));
}

namespace {

// The vertex tables one draw's vertex shader reads, copied through the reader. The attribute table is
// read up to the largest semantic and the buffer table up to the largest buffer an attribute names.
struct VertexTables {
	std::array<ShaderSemantic, ShaderVertexInputInfo::RES_MAX> semantics {};
	uint32_t                                                    semantic_count = 0;
	// A semantic is 8 bits, and a buffer index 5 bits with four words per buffer.
	std::array<uint32_t, 256>                                  attrib {};
	std::array<uint32_t, ShaderVertexInputInfo::RES_MAX * 4>   buffer {};
};

// A 64-bit table pointer from two user registers, as ShaderGetStaticVertexInputInfo builds it.
uint64_t UserPointer(const HW::UserSgprInfo& user, int reg) {
	return static_cast<uint64_t>(user.value[reg]) | (static_cast<uint64_t>(user.value[reg + 1]) << 32u);
}

// Copies the semantics and both tables. False where a read fails or a pointer is null, which the GPU
// thread refuses too.
bool ReadVertexTables(const ScannedDraw& draw, const ShaderMapSnapshot& vertex, GraphicsPrescanReader read,
                      void* userdata, VertexTables& tables) {
	constexpr uint32_t index_mask    = 0x1fu;
	constexpr uint32_t buffer_words  = 4;
	tables.semantic_count            = vertex.num_input_semantics;
	const auto& user                 = draw.vs_regs.gs_user_sgpr;
	const uint64_t attrib_address    = UserPointer(user, vertex.vertex_attrib_reg);
	const uint64_t buffer_address    = UserPointer(user, vertex.vertex_buffer_reg);
	auto semantic_words = std::span<uint32_t>(reinterpret_cast<uint32_t*>(tables.semantics.data()), tables.semantic_count);
	if (attrib_address == 0 || buffer_address == 0 || vertex.input_semantics_address == 0 ||
	    !read(userdata, vertex.input_semantics_address, semantic_words)) {
		return false;
	}
	uint32_t max_semantic = 0;
	for (uint32_t i = 0; i < tables.semantic_count; i++) {
		max_semantic = std::max<uint32_t>(max_semantic, tables.semantics[i].semantic);
	}
	if (!read(userdata, attrib_address, std::span<uint32_t>(tables.attrib.data(), max_semantic + 1u))) {
		return false;
	}
	uint32_t max_index = 0;
	for (uint32_t i = 0; i < tables.semantic_count; i++) {
		max_index = std::max(max_index, tables.attrib[tables.semantics[i].semantic] & index_mask);
	}
	return read(userdata, buffer_address, std::span<uint32_t>(tables.buffer.data(), (max_index + 1u) * buffer_words));
}

// The clip space and user clip planes GetGraphicsPrograms puts on the last pre-rasterization stage.
void AddClipState(const ScannedDraw& draw, std::array<uint32_t, 2> max_viewport, ShaderVertexInputInfo& info) {
	// GetGraphicsPrograms caps the clip extent at the guest's largest viewport.
	constexpr uint32_t max_guest_viewport = 16384;
	constexpr float    half               = 0.5f;
	if (draw.clip_control.clip_disable) {
		auto& clip = info.clip_space;
		for (size_t axis = 0; axis < 2; axis++) {
			clip.scale[axis]       = draw.viewport_scale[axis];
			clip.offset[axis]      = draw.viewport_offset[axis];
			clip.half_extent[axis] = static_cast<float>(std::min(max_viewport[axis], max_guest_viewport)) * half;
		}
		clip.enabled = true;
	} else if (draw.user_clip_planes != 0) {
		info.user_clip_planes = draw.user_clip_planes;
	}
}

// Whether every attribute's buffer can be fetched by TranslateEmbeddedFetch without EXIT. A fetch reads
// at most four components and which ones is only known after translation, so all four selectors are
// judged. That rejects only a buffer with a reserved selector (2 or 3) on a component no fetch reads.
// The pre-rasterization key of a vertex module for a draw, as FillStaticKey and MakePreRasterKey make
// it for a draw with no tessellation and no rect list, which are both skipped.
PreRasterKey ScannedPreRasterKey(uint64_t id, const ScannedDraw& draw) {
	const auto&  clip = draw.clip_control;
	const auto&  mode = draw.mode_control;
	PreRasterKey key;
	key.stage_ids[0]        = id;
	key.negative_one_to_one = !clip.dx_clip_space;
	key.depth_clip_enable   = clip.IsZClipEnabled();
	key.provoking_vtx_last  = mode.provoking_vtx_last;
	key.polygon_mode        = PrescanPolygonMode(mode, mode.cull_front, mode.cull_back);
	return key;
}

bool EmbeddedFetchesValid(const ShaderVertexInputInfo& info) {
	constexpr uint32_t max_fetch_components = 4;
	for (int index = 0; index < info.resources_num; index++) {
		if (!ShaderRecompiler::EmbeddedFetchFormatValid(info.resources[index], max_fetch_components)) {
			return false;
		}
	}
	return true;
}

} // namespace

std::optional<GraphicsPrescanSkip> ScannedVertexInputInfo(const ScannedDraw& draw, const ShaderMapSnapshot& vertex,
                                                          std::array<uint32_t, 2> max_viewport,
                                                          GraphicsPrescanReader read, void* userdata,
                                                          ShaderVertexInputInfo& info) {
	// VGT_SHADER_STAGES_EN bit 22 runs the vertex stage in wave32, and LDS_SIZE counts 128 dwords.
	constexpr uint32_t wave32_stage_bit = 0x00400000u;
	constexpr uint32_t lds_granule      = 128;
	if (vertex.vertex_error) {
		return GraphicsPrescanSkip::VertexMetadata;
	}
	auto tables = std::make_unique<VertexTables>();
	if (vertex.vertex_attrib_reg >= 0) {
		if (vertex.num_input_semantics == 0 || vertex.num_input_semantics > ShaderVertexInputInfo::RES_MAX) {
			return GraphicsPrescanSkip::VertexMetadata;
		}
		if (!ReadVertexTables(draw, vertex, read, userdata, *tables)) {
			return GraphicsPrescanSkip::VertexTables;
		}
	}
	HW::ShaderRegisters sh {};
	sh.m_paClVsOutCntl = draw.pa_cl_vs_out_cntl;
	if (!ShaderTryStaticVertexInputInfo(sh, vertex.scratch_size_dwords, vertex.vertex_attrib_reg,
	                                    vertex.vertex_buffer_reg,
	                                    std::span<const ShaderSemantic>(tables->semantics.data(), tables->semantic_count),
	                                    tables->attrib.data(), tables->buffer.data(), info)) {
		return GraphicsPrescanSkip::VertexTables;
	}
	// The attribute widths and formats come from the tables, so one VertexInputFormats would EXIT on
	// is rejected before anything is translated.
	if (!PrescanVertexInputsSupported(info)) {
		return GraphicsPrescanSkip::InvalidVertexFormat;
	}
	if (!EmbeddedFetchesValid(info)) {
		return GraphicsPrescanSkip::InvalidVertexFetch;
	}
	info.wave_size      = (draw.shader_stages & wave32_stage_bit) != 0 ? 32u : 64u;
	info.lds_size_dwords = draw.gs_lds_size * lds_granule;
	AddClipState(draw, max_viewport, info);
	return std::nullopt;
}

// The report's source is registered last, so a throw from any step before leaves no source that
// reads this cache. StopStoreGeneration hands over the final figures.
GraphicsDrawSink PipelineCache::GraphicsPrescanSink() {
	if (m_generation == nullptr) {
		return {};
	}
	if (m_graphics_prescan == nullptr) {
		auto state = std::make_unique<GraphicsPrescanState>(GRAPHICS_PRESCAN_TRANSLATE_WORKERS,
		                                                    GRAPHICS_PRESCAN_LIBRARY_WORKERS);
		state->host            = {.structured_jump_tables = Config::StructuredJumpTablesEnabled(),
		                          .profile                = SpirvProfile(m_graphics),
		                          .max_shared_bytes =
		                              m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize};
		state->emit_debug_info = Config::GraphicsDebugDumpEnabled();
		state->keep_records    = static_cast<bool>(m_store);
		m_graphics_prescan     = std::move(state);
		m_graphics_prescan_report_registered = true;
		Common::FirstRunReport::Global().SetGraphicsPrescanSource(
		    Common::FirstRunReport::GraphicsPrescanPart::Cache, [this] { return BuildGraphicsPrescanReport(); });
	}
	return [this](std::span<const ScannedDraw> draws) { OfferGraphicsDraws(draws); };
}

uint64_t PipelineCache::GraphicsPrescanDrawsOffered() const noexcept {
	return m_graphics_prescan == nullptr ? 0 : m_graphics_prescan->counters.offered.load(std::memory_order_relaxed);
}

// Each half's key is queued once over the run, so a draw drawn every frame costs a set lookup or two
// after its first. A draw whose halves are both new gets one pixel job, which queues the vertex job
// when it has the bindings the vertex shader's push start follows. A draw with only a new vertex half
// gets a vertex job alone, which finds the bindings of its pixel half's earlier job. Pixel jobs go in
// the front lane and vertex jobs in the back, so pixel work never waits behind vertex work. The jobs
// copy the draw, since the span is only valid during the call.
void PipelineCache::OfferGraphicsDraws(std::span<const ScannedDraw> draws) {
	auto& state     = *m_graphics_prescan;
	auto& counters  = state.counters;
	for (const auto& draw: draws) {
		counters.offered.fetch_add(1, std::memory_order_relaxed);
		const bool new_ps = draw.offer_ps && state.offered_keys.insert(draw.key_hash).second;
		counters.deduped.fetch_add(draw.offer_ps && !new_ps ? 1 : 0, std::memory_order_relaxed);
		bool new_vs = false;
		if (draw.offer_vs && draw.vs_key_hash != 0) {
			counters.vs_offered.fetch_add(1, std::memory_order_relaxed);
			new_vs = state.offered_vs_keys.insert(draw.vs_key_hash).second;
			counters.vs_deduped.fetch_add(new_vs ? 0 : 1, std::memory_order_relaxed);
		}
		if (new_ps) {
			const auto offered = Clock::now();
			const bool queued  = PostBounded(state.translate, state.translate_bound, PipelineBuildQueue::Lane::Front,
			                                 [this, draw, offered, new_vs] { RunGraphicsPrescanJob(draw, offered, new_vs); });
			(queued ? counters.queued : counters.not_queued).fetch_add(1, std::memory_order_relaxed);
		} else if (new_vs) {
			PostVertexJob(draw, std::nullopt);
		}
	}
}

// Queues a vertex job behind every pixel job.
void PipelineCache::PostVertexJob(const ScannedDraw& draw, std::optional<ShaderRecompiler::IR::Bindings> ps_bindings) {
	auto& state = *m_graphics_prescan;
	const bool queued = PostBounded(state.translate, state.translate_bound, PipelineBuildQueue::Lane::Back,
	                                [this, draw, ps_bindings = std::move(ps_bindings)] { RunVertexJob(draw, ps_bindings); });
	state.counters.vs_queued.fetch_add(queued ? 1 : 0, std::memory_order_relaxed);
}

// A vertex job given no bindings takes those its pixel half's job kept. Without them the push start
// is unknown, so the job stops as vertex_no_pixel. The pixel shader's cursor is the one a draw whose
// vertex stage is not a mesh shader gets, since RunVertexPrescan skips mesh stages.
void PipelineCache::RunVertexJob(const ScannedDraw& draw, const std::optional<ShaderRecompiler::IR::Bindings>& ps_bindings) {
	auto& state = *m_graphics_prescan;
	std::optional<ShaderRecompiler::IR::Bindings> bindings = ps_bindings;
	if (!bindings) {
		std::scoped_lock lock(state.ps_bindings_mutex);
		if (const auto found = state.ps_bindings.find(draw.key_hash); found != state.ps_bindings.end()) {
			bindings = found->second;
		}
	}
	if (!bindings) {
		state.counters.vertex.skipped[static_cast<size_t>(GraphicsPrescanSkip::VertexNoPixel)].fetch_add(
		    1, std::memory_order_relaxed);
		return;
	}
	const GraphicsJobSlot slot(state.running_jobs, state.counters, state.translate);
	const Common::FirstRunReport::PrescanGraphicsRunningScope running;
	RunVertexPrescan(draw, *bindings, ScannedDrawPushCursor(draw, false));
}

// The program key is published as busy while the job translates, and as settled once it built, so a
// pixel shader the GPU thread compiles meanwhile is told from one the prescan never tried. A job that
// stops short clears it again, unless it stopped because the same module was built or claimed.
void PipelineCache::RunGraphicsPrescanJob(const ScannedDraw& draw, Clock::time_point offered, bool chain_vertex) {
	auto& state    = *m_graphics_prescan;
	auto& counters = state.counters;
	const GraphicsJobSlot slot(state.running_jobs, counters, state.translate);
	const Common::FirstRunReport::PrescanGraphicsRunningScope running;
	const auto                                                start = Clock::now();
	GraphicsUnit                                              unit;
	auto                                                      skip = ReadGraphicsUnit(draw, unit);
	PixelPrescanBuild                                             build;
	if (!skip) {
		state.program_keys.Publish(unit.program_key, PrescanFeedback::Answer::Busy);
		build = CompileGraphicsUnit(draw, unit, state.host, state.emit_debug_info, state.rejects, state.dedupe);
		skip  = build.skip;
	}
	if (skip) {
		const bool built_elsewhere = *skip == GraphicsPrescanSkip::DedupedLate;
		state.program_keys.Publish(unit.program_key,
		                           built_elsewhere ? PrescanFeedback::Answer::Settled : PrescanFeedback::Answer::None);
		counters.skipped[static_cast<size_t>(*skip)].fetch_add(1, std::memory_order_relaxed);
		// The vertex half was offered with this pixel half, so it goes too, and a later vertex job finds
		// the bindings if another job of the same module keeps them.
		if (chain_vertex) {
			PostVertexJob(draw, std::nullopt);
		}
		return;
	}
	counters.heap_image.fetch_add(build.heap_image ? 1 : 0, std::memory_order_relaxed);
	{
		std::scoped_lock lock(state.variants_mutex);
		state.variants[unit.program_key].push_back(
		    {build.id, build.info.bindings.push_data_start_dword, build.specialization});
	}
	state.program_keys.Publish(unit.program_key, PrescanFeedback::Answer::Settled);
	state.built_ids.Publish(build.id, PrescanFeedback::Answer::Busy);
	const auto source = PixelSource(draw, unit);
	auto       bytes  = state.keep_records ? EncodeRecord(source, build, {}) : std::vector<uint8_t> {};
	PostGraphicsPrescanModule(MakeRecord(source, build, {}), std::move(bytes));
	const auto posted = Clock::now();
	counters.posted.fetch_add(1, std::memory_order_relaxed);
	const auto after_offer_us = WholeMicros(posted - offered);
	counters.posted_after_offer_ms[MillisecondBucket(after_offer_us)].fetch_add(1, std::memory_order_relaxed);
	RaiseMax(counters.posted_after_offer_max_us, after_offer_us);
	const auto translate_us = WholeMicros(posted - start);
	counters.translate_us.fetch_add(translate_us, std::memory_order_relaxed);
	RaiseMax(counters.translate_max_us, translate_us);
	// The vertex shader's push data follows the pixel shader's, as GetGraphicsPrograms advances one
	// cursor over both, so the pixel shader's bindings are kept for its vertex jobs.
	const auto ps_bindings = build.info.bindings;
	{
		std::scoped_lock lock(state.ps_bindings_mutex);
		state.ps_bindings.insert_or_assign(draw.key_hash, ps_bindings);
	}
	(void)PostBounded(state.libraries, state.libraries_bound, PipelineBuildQueue::Lane::Front,
	                  [this, id = build.id, spirv = std::move(build.spirv), info = std::move(build.info),
	                   a2c = draw.alpha_to_coverage] { RunFragmentPrebuild(id, spirv, info, a2c); });
	if (chain_vertex) {
		PostVertexJob(draw, ps_bindings);
	}
}

// The vertex shader's half of a job, after the pixel shader posted. It is skipped for indirect draws
// and mesh stages, and otherwise built with the options GetProgram gives a vertex shader, at the push
// cursor the pixel shader left. Its pre-rasterization library is then queued like the fragment one.
void PipelineCache::RunVertexPrescan(const ScannedDraw& draw, const ShaderRecompiler::IR::Bindings& ps_bindings,
                                     uint32_t ps_cursor) {
	auto&      state  = *m_graphics_prescan;
	auto&      counts = state.counters.vertex;
	const auto start  = Clock::now();
	const auto skip   = [&](GraphicsPrescanSkip reason) {
		counts.skipped[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
	};
	const auto vertex = ShaderTryGetMapped(draw.vs_program_address);
	if (draw.indirect || !vertex) {
		return skip(draw.indirect ? GraphicsPrescanSkip::VertexIndirect : GraphicsPrescanSkip::CodeUnmapped);
	}
	if (VertexStageIsMesh(draw.shader_stages, vertex->type, draw.gs_lds_size)) {
		return skip(GraphicsPrescanSkip::VertexMesh);
	}
	const uint32_t user_data_count = draw.vs_regs.gs_regs.rsrc2.user_sgpr;
	std::vector<uint32_t> code(vertex->code_size_bytes / sizeof(uint32_t));
	if (user_data_count > HW::UserSgprInfo::SGPRS_MAX) {
		return skip(GraphicsPrescanSkip::TooManyUserData);
	}
	if (code.empty() || !ReadGraphicsPrescanMemory(nullptr, draw.vs_program_address, code) ||
	    XXH3_64bits(code.data(), vertex->code_size_bytes) != vertex->hash) {
		return skip(GraphicsPrescanSkip::CodeChanged);
	}
	const auto& limits = m_graphics.GetPhysicalDeviceProperties().limits;
	ShaderVertexInputInfo info;
	if (const auto refused = ScannedVertexInputInfo(draw, *vertex, {limits.maxViewportDimensions[0], limits.maxViewportDimensions[1]},
	                                                ReadGraphicsPrescanMemory, nullptr, info)) {
		return skip(*refused);
	}
	std::vector<uint32_t> static_state;
	BuildStageStaticKey(info, static_state);
	uint32_t cursor = ps_cursor;
	ps_bindings.AdvancePushData(cursor);
	const GraphicsStageSource source {.stage           = ShaderType::Vertex,
	                          .code            = code,
	                          .code_hash       = vertex->hash,
	                          .user_data_count = user_data_count,
	                          .static_state    = static_state,
	                          .shader_base     = draw.vs_program_address,
	                          .user_data       = std::span<const uint32_t>(draw.vs_regs.gs_user_sgpr.value, user_data_count),
	                          .push_cursor     = cursor,
	                          .program_key     = GraphicsProgramKeyHash(vertex->hash, user_data_count, static_state)};
	PostVertexBuild(draw, source, info, start);
}

// Compiles the vertex shader with GetProgram's vertex options, posts it with its attribute formats and
// queues its pre-rasterization library.
void PipelineCache::PostVertexBuild(const ScannedDraw& draw, const GraphicsStageSource& source,
                                    const ShaderVertexInputInfo& info, Clock::time_point start) {
	auto&      state       = *m_graphics_prescan;
	auto&      counts      = state.counters.vertex;
	const auto program_key = source.program_key;
	ShaderRecompiler::CompileOptions options {.wave_size = info.wave_size};
	options.stage                  = info.logical_stage;
	options.input_info.vertex      = &info;
	options.user_data_base         = 8;
	options.shader_hash            = source.code_hash;
	options.user_data              = source.user_data;
	options.dump_ir                = false;
	options.emit_debug_info        = state.emit_debug_info;
	options.structured_jump_tables = state.host.structured_jump_tables;
	options.profile                = &state.host.profile;
	state.program_keys.Publish(program_key, PrescanFeedback::Answer::Busy);
	auto build = CompileStage(source, options, state.rejects, state.dedupe);
	if (build.skip) {
		const bool built_elsewhere = *build.skip == GraphicsPrescanSkip::DedupedLate;
		state.program_keys.Publish(program_key,
		                           built_elsewhere ? PrescanFeedback::Answer::Settled : PrescanFeedback::Answer::None);
		counts.skipped[static_cast<size_t>(*build.skip)].fetch_add(1, std::memory_order_relaxed);
		return;
	}
	std::vector<int32_t> formats;
	for (const auto format: PrescanVertexInputFormats(info, build.info)) {
		formats.push_back(static_cast<int32_t>(format));
	}
	{
		std::scoped_lock lock(state.variants_mutex);
		state.variants[program_key].push_back({build.id, build.info.bindings.push_data_start_dword, build.specialization});
	}
	state.program_keys.Publish(program_key, PrescanFeedback::Answer::Settled);
	state.built_ids.Publish(build.id, PrescanFeedback::Answer::Busy);
	auto bytes = state.keep_records ? EncodeRecord(source, build, formats) : std::vector<uint8_t> {};
	PostGraphicsPrescanModule(MakeRecord(source, build, formats), std::move(bytes));
	counts.posted.fetch_add(1, std::memory_order_relaxed);
	counts.translate_us.fetch_add(WholeMicros(Clock::now() - start), std::memory_order_relaxed);
	// A library of the same key built or queued already is not queued again, before any module or
	// layout is made for it.
	constexpr uint64_t pre_raster_domain = 0x5052455241535445ull;
	if (!state.dedupe.Claim(PartKeyHash<PreRasterKey> {}(ScannedPreRasterKey(build.id, draw)) ^ pre_raster_domain)) {
		counts.prebuild_deduped_late.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	(void)PostBounded(state.libraries, state.libraries_bound, PipelineBuildQueue::Lane::Back,
	                  [this, id = build.id, spirv = std::move(build.spirv), compiled = std::move(build.info), draw] {
		                  RunPreRasterPrebuild(id, spirv, compiled, draw);
	                  });
}

// The pre-rasterization key is made as FillStaticKey and MakePreRasterKey make it for a draw with no
// tessellation and no rect list, which are both skipped. The vertex module here has no handle of its
// own, since the library build uses only the plain module, made privately and destroyed after.
void PipelineCache::RunPreRasterPrebuild(uint64_t id, const std::vector<uint32_t>& spirv,
                                         const ShaderRecompiler::IR::CompiledShaderInfo& info, const ScannedDraw& draw) {
	auto&      counts = m_graphics_prescan->counters.vertex;
	const auto skip   = [&](GraphicsPrebuildSkip reason) {
		counts.prebuild_skipped[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
	};
	const GraphicsJobSlot slot(m_graphics_prescan->running_jobs, m_graphics_prescan->counters, m_graphics_prescan->libraries);
	const Common::FirstRunReport::PrescanGraphicsRunningScope running;
	if (!m_graphics.graphics_pipeline_library_enabled || m_libraries == nullptr) {
		return skip(GraphicsPrebuildSkip::LibrariesOff);
	}
	if (m_graphics.image_heap_layout == nullptr) {
		return skip(GraphicsPrebuildSkip::NoHeapLayout);
	}
	if (draw.mode_control.provoking_vtx_last && !m_graphics.provoking_vertex_last_enabled) {
		counts.prebuild_provoking_unsupported.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	ShaderModule vertex {.id = id, .module = CompileSPV(spirv, m_graphics.device), .info = info};
	const auto   layout = VertexSplitSetLayout(m_graphics, m_set_layouts, vertex);
	if (layout) {
		const auto key    = ScannedPreRasterKey(id, draw);
		const auto before = StallMarks::Take();
		const auto begin  = Clock::now();
		if (PrebuildPreRasterLibrary(m_graphics, *m_libraries, *layout, vertex, key, m_driver_cache)) {
			const auto after = StallMarks::Take();
			counts.prebuilt.fetch_add(1, std::memory_order_relaxed);
			counts.prebuild_us.fetch_add(WholeMicros(Clock::now() - begin), std::memory_order_relaxed);
			counts.prebuild_in_scope.fetch_add(
			    before.scope_active || after.scope_active || after.scopes != before.scopes ? 1 : 0, std::memory_order_relaxed);
			counts.prebuild_in_hold.fetch_add(
			    before.hold_active || after.hold_active || after.holds != before.holds ? 1 : 0, std::memory_order_relaxed);
			m_driver_cache_dirty.store(true, std::memory_order_relaxed);
		} else {
			skip(GraphicsPrebuildSkip::AlreadyThere);
		}
	} else {
		skip(GraphicsPrebuildSkip::LayoutLimits);
	}
	m_graphics.device.destroyShaderModule(vertex.module, nullptr);
}

// Several translate workers post at once. That is safe: the intake always carries the plan, so its
// decoded key set is never touched, its counters are atomics, and PushGenerated splices under the
// state's mutex.
void PipelineCache::PostGraphicsPrescanModule(DecodedModuleRecord&& record, std::vector<uint8_t>&& record_bytes) {
	if (m_generation == nullptr) {
		return;
	}
	if (auto item = MakeGeneratedModule(std::move(record), {.prescan = true}, m_generation->prescan_graphics)) {
		if (!record_bytes.empty()) {
			item->module.retained.record = std::make_unique<RetainedRecord>(std::move(record_bytes));
		}
		PushGenerated(std::move(*item));
	}
}

// The library is keyed as a single-sample draw with this alpha to coverage would key it, see
// FillStaticKey. The split module is private to this job and destroyed after, which Vulkan allows
// once the library is made.
void PipelineCache::RunFragmentPrebuild(uint64_t id, const std::vector<uint32_t>& spirv,
                                        const ShaderRecompiler::IR::CompiledShaderInfo& info,
                                        bool alpha_to_coverage) {
	auto&      counters = m_graphics_prescan->counters;
	const auto skip     = [&](GraphicsPrebuildSkip reason) {
		counters.prebuild_skipped[static_cast<size_t>(reason)].fetch_add(1, std::memory_order_relaxed);
	};
	const GraphicsJobSlot slot(m_graphics_prescan->running_jobs, counters, m_graphics_prescan->libraries);
	const Common::FirstRunReport::PrescanGraphicsRunningScope running;
	if (!m_graphics.graphics_pipeline_library_enabled || m_libraries == nullptr) {
		return skip(GraphicsPrebuildSkip::LibrariesOff);
	}
	if (m_graphics.image_heap_layout == nullptr) {
		return skip(GraphicsPrebuildSkip::NoHeapLayout);
	}
	ShaderModule pixel {.id = id, .gpl_module = CompileSplitPixelModule(m_graphics, ShaderType::Pixel, spirv),
	                    .info = info};
	if (pixel.gpl_module == nullptr) {
		return skip(GraphicsPrebuildSkip::NoSplitModule);
	}
	const auto layout = PixelSplitSetLayout(m_graphics, m_set_layouts, pixel);
	if (layout) {
		const FragmentKey key {.pixel_id = id, .samples = 1, .alpha_to_coverage_enable = alpha_to_coverage};
		const auto        before = StallMarks::Take();
		const auto        start  = Clock::now();
		const bool built = PrebuildFragmentLibrary(m_graphics, *m_libraries, *layout, pixel, key, m_driver_cache);
		const auto after = StallMarks::Take();
		if (built) {
			counters.prebuilt.fetch_add(1, std::memory_order_relaxed);
			counters.prebuild_us.fetch_add(WholeMicros(Clock::now() - start), std::memory_order_relaxed);
			const bool in_scope = before.scope_active || after.scope_active || after.scopes != before.scopes;
			const bool in_hold  = before.hold_active || after.hold_active || after.holds != before.holds;
			counters.prebuild_in_scope.fetch_add(in_scope ? 1 : 0, std::memory_order_relaxed);
			counters.prebuild_in_hold.fetch_add(in_hold ? 1 : 0, std::memory_order_relaxed);
			m_driver_cache_dirty.store(true, std::memory_order_relaxed);
		} else {
			skip(GraphicsPrebuildSkip::AlreadyThere);
		}
	} else {
		skip(GraphicsPrebuildSkip::LayoutLimits);
	}
	m_graphics.device.destroyShaderModule(pixel.gpl_module, nullptr);
}

// The set is the GPU thread's, so a lookup on another thread, such as a boot-time one, is left out.
// Pixel and vertex modules each have their own set and counts, and other stages none.
void PipelineCache::NoteGraphicsPrescanBind(uint64_t id, ShaderType stage) {
	if (m_graphics_prescan == nullptr || (stage != ShaderType::Pixel && stage != ShaderType::Vertex) ||
	    !GuestGpu::IsGpuThread()) {
		return;
	}
	auto&      state  = *m_graphics_prescan;
	const bool vertex = stage == ShaderType::Vertex;
	auto&      set    = vertex ? state.adopted_vertex : state.adopted;
	if (!set.empty() && set.erase(id) != 0) {
		(vertex ? state.counters.vertex.used : state.counters.used).fetch_add(1, std::memory_order_relaxed);
	}
}

void PipelineCache::NoteGraphicsPrescanAdopted(uint64_t id, ShaderType stage) {
	if (m_graphics_prescan == nullptr) {
		return;
	}
	auto&      state  = *m_graphics_prescan;
	const bool vertex = stage == ShaderType::Vertex;
	(vertex ? state.adopted_vertex : state.adopted).insert(id);
	(vertex ? state.counters.vertex.adopted : state.counters.adopted).fetch_add(1, std::memory_order_relaxed);
}

void PipelineCache::NoteGraphicsPrescanRefused(bool duplicate, ShaderType stage) {
	if (m_graphics_prescan == nullptr || !duplicate) {
		return;
	}
	auto& counters = m_graphics_prescan->counters;
	(stage == ShaderType::Vertex ? counters.vertex.late_duplicates : counters.late_duplicates)
	    .fetch_add(1, std::memory_order_relaxed);
}

// Only relaxed loads of the worker's tables and one hash of the key, on a compile that is happening
// anyway. A built id or a program still being translated means the module came too late, a program
// that built under another id means the two disagreed on the specialization or the push start.
void PipelineCache::NoteGraphicsPrescanMiss(const ProgramKey& key, const ShaderModule& module, uint64_t code_address) {
	if (m_graphics_prescan == nullptr || (key.stage != ShaderType::Pixel && key.stage != ShaderType::Vertex)) {
		return;
	}
	auto&      state       = *m_graphics_prescan;
	const auto program_key = GraphicsProgramKeyHash(key.hash, key.user_data_count, key.static_state);
	const auto answer      = state.program_keys.Read(program_key);
	GraphicsPrescanMiss miss = GraphicsPrescanMiss::Unpredicted;
	if (state.built_ids.Read(module.id) != PrescanFeedback::Answer::None || answer == PrescanFeedback::Answer::Busy) {
		miss = GraphicsPrescanMiss::Late;
	} else if (answer == PrescanFeedback::Answer::Settled) {
		miss = GraphicsPrescanMiss::IdDiffers;
		NoteGraphicsPrescanIdDiffers(program_key, module);
	}
	auto& misses = key.stage == ShaderType::Vertex ? state.counters.vertex.misses : state.counters.misses;
	misses[static_cast<size_t>(miss)].fetch_add(1, std::memory_order_relaxed);
}

// A built module that matches the GPU's in specialization but not push start points at the push
// start, one that matches in push start but not specialization points at the specialization, and
// anything else, such as both differing or no module kept, is other.
void PipelineCache::NoteGraphicsPrescanIdDiffers(uint64_t program_key, const ShaderModule& module) {
	auto&                               state = *m_graphics_prescan;
	std::vector<GraphicsPrescanVariant> built;
	{
		std::scoped_lock lock(state.variants_mutex);
		if (const auto found = state.variants.find(program_key); found != state.variants.end()) {
			built = found->second;
		}
	}
	const uint32_t push_start = module.info.bindings.push_data_start_dword;
	bool           push_only  = false;
	bool           spec_only  = false;
	for (const auto& variant: built) {
		const bool same_spec = variant.specialization == module.specialization;
		push_only            = push_only || (same_spec && variant.push_start != push_start);
		spec_only            = spec_only || (!same_spec && variant.push_start == push_start);
	}
	auto&      counters = state.counters;
	const bool vertex   = module.info.stage == ShaderType::Vertex;
	auto& push = vertex ? counters.vertex.id_differs_push_start : counters.id_differs_push_start;
	auto& spec = vertex ? counters.vertex.id_differs_specialization : counters.id_differs_specialization;
	auto& other = vertex ? counters.vertex.id_differs_other : counters.id_differs_other;
	(push_only ? push : spec_only ? spec : other).fetch_add(1, std::memory_order_relaxed);
	Log::Write({}, IdDiffersLine(program_key, module, built));
}

// Translate first, so no new library job is queued behind the stop, then the libraries.
void PipelineCache::StopGraphicsPrescan() {
	if (m_graphics_prescan != nullptr) {
		(void)m_graphics_prescan->translate.Stop();
		(void)m_graphics_prescan->libraries.Stop();
	}
}

Common::FirstRunReport::PrescanGraphicsReport PipelineCache::BuildGraphicsPrescanReport() const {
	using Group          = Common::FirstRunReport::PrescanGraphicsReport::Group;
	const auto& state    = *m_graphics_prescan;
	const auto& counters = state.counters;
	Group worker {"worker",
	              {{"offered", Count(counters.offered)},
	               {"deduped", Count(counters.deduped)},
	               {"queued", Count(counters.queued)},
	               {"not_queued", Count(counters.not_queued)},
	               {"vs_offered", Count(counters.vs_offered)},
	               {"vs_deduped", Count(counters.vs_deduped)},
	               {"vs_queued", Count(counters.vs_queued)},
	               {"translate_front_depth_max", Count(state.translate_bound.depth_max[0])},
	               {"translate_back_depth_max", Count(state.translate_bound.depth_max[1])},
	               {"translate_dropped_full", Count(state.translate_bound.dropped_full)},
	               {"libraries_front_depth_max", Count(state.libraries_bound.depth_max[0])},
	               {"libraries_back_depth_max", Count(state.libraries_bound.depth_max[1])},
	               {"libraries_dropped_full", Count(state.libraries_bound.dropped_full)},
	               {"capped_waits", Count(counters.capped_waits)},
	               {"paused_starts", Count(counters.paused_starts)},
	               {"reduced_cap_ms", counters.cap_governor.ReducedMs(Common::FirstRunReport::NowNs())},
	               {"reduced_cap_triggers", static_cast<double>(counters.cap_governor.Triggers())},
	               {"graphics_busy_fraction", GraphicsBusyFraction(counters)},
	               {"posted", Count(counters.posted)},
	               {"heap_image", Count(counters.heap_image)},
	               {"translate_ms", ToMs(counters.translate_us)},
	               {"translate_max_ms", ToMs(counters.translate_max_us)},
	               {"posted_after_offer_p50_ms", HistogramPercentileMs(counters, 0.5)},
	               {"posted_after_offer_p90_ms", HistogramPercentileMs(counters, 0.9)},
	               {"posted_after_offer_max_ms", ToMs(counters.posted_after_offer_max_us)}}};
	Group skipped {"skipped", {}};
	for (size_t reason = 0; reason < counters.skipped.size(); ++reason) {
		skipped.figures.emplace_back(SkipName(static_cast<GraphicsPrescanSkip>(reason)), Count(counters.skipped[reason]));
	}
	const double adopted = Count(counters.adopted);
	const double used    = Count(counters.used);
	Group modules {"modules",
	               {{"adopted", adopted},
	                {"used", used},
	                {"unused", std::max(0.0, adopted - used)},
	                {"late_duplicates", Count(counters.late_duplicates)}}};
	Group misses {"misses", {}};
	for (size_t miss = 0; miss < counters.misses.size(); ++miss) {
		misses.figures.emplace_back(MissName(static_cast<GraphicsPrescanMiss>(miss)), Count(counters.misses[miss]));
	}
	misses.figures.emplace_back("id_differs_push_start", Count(counters.id_differs_push_start));
	misses.figures.emplace_back("id_differs_specialization", Count(counters.id_differs_specialization));
	misses.figures.emplace_back("id_differs_other", Count(counters.id_differs_other));
	const auto prebuilt = m_libraries != nullptr ? m_libraries->PrebuiltCounts() : PrebuiltLibraryCounts {};
	Group      libraries {"libraries",
	                      {{"prebuilt", Count(counters.prebuilt)},
	                       {"prebuild_ms", ToMs(counters.prebuild_us)},
	                       {"prebuild_in_scope", Count(counters.prebuild_in_scope)},
	                       {"prebuild_in_hold", Count(counters.prebuild_in_hold)},
	                       {"first_hits", static_cast<double>(prebuilt.first_hits)},
	                       {"waits", static_cast<double>(prebuilt.waits)},
	                       {"wait_ms", prebuilt.wait_ms},
	                       {"wait_max_ms", prebuilt.wait_max_ms},
	                       {"key_differs", static_cast<double>(prebuilt.key_differs)}}};
	for (size_t reason = 0; reason < counters.prebuild_skipped.size(); ++reason) {
		libraries.figures.emplace_back(std::string("skipped_") + PrebuildSkipName(static_cast<GraphicsPrebuildSkip>(reason)),
		                               Count(counters.prebuild_skipped[reason]));
	}
	Common::FirstRunReport::PrescanGraphicsReport report {
	    {std::move(worker), std::move(skipped), std::move(modules), std::move(misses), std::move(libraries)}};
	const auto pre_raster =
	    m_libraries != nullptr ? m_libraries->PrebuiltCounts(LibraryPart::PreRaster) : PrebuiltLibraryCounts {};
	for (auto& group: VertexGroups(counters.vertex, pre_raster)) {
		report.groups.push_back(std::move(group));
	}
	report.groups.push_back(RejectedGroup(m_graphics_prescan->rejects));
	report.groups.push_back(
	    GraphicsBusyTimeline(counters, Common::FirstRunReport::Global().EpochNs(), Common::FirstRunReport::NowNs()));
	return report;
}

// Called by StopStoreGeneration, after both queues are joined and before the library cache goes, so
// the final figures are whole.
void PipelineCache::FinishGraphicsPrescanReport() {
	if (m_graphics_prescan == nullptr || !m_graphics_prescan_report_registered) {
		return;
	}
	auto final_figures = BuildGraphicsPrescanReport();
	Common::FirstRunReport::Global().SetGraphicsPrescanFinal(Common::FirstRunReport::GraphicsPrescanPart::Cache,
	                                                       std::move(final_figures));
	m_graphics_prescan_report_registered = false;
}

} // namespace Libs::Graphics
