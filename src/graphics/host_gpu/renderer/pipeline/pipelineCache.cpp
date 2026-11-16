#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/firstRunReport.h"
#include "common/gpuStall.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/image/textureCommon.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/generationState.h"
#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"
#include "graphics/host_gpu/renderer/pipeline/pixelInputs.h"
#include "graphics/host_gpu/renderer/pipeline/prescanGraphics.h"
#include "graphics/host_gpu/renderer/pipeline/prescanPreparer.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/shader/moduleId.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/Tessellation.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/storegen/generationRows.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <optional>
#include <ranges>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// PA_SU_SC_MODE_CNTL.POLY_MODE disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

void DumpShader(std::span<const uint32_t> words, const std::string& name) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	const auto path = Config::GetShaderLogFolder() / name;
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(words.data(), words.size_bytes());
}

void ValidateShaderSpirv(const char* stage, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF("%s\n", text.c_str());
	EXIT("ShaderRecompiler %s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s", stage,
	     shader_hash, messages.c_str());
}

// Formats with a narrower host variant drop the components the shader does not read.
vk::Format NarrowInputFormat(vk::Format format, uint32_t used_components) {
	switch (format) {
		case vk::Format::eR32G32B32A32Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR32Sfloat;
				case 2: return vk::Format::eR32G32Sfloat;
				case 3: return vk::Format::eR32G32B32Sfloat;
				default: return format;
			}
		case vk::Format::eR32G32B32Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR32Sfloat;
				case 2: return vk::Format::eR32G32Sfloat;
				default: return format;
			}
		case vk::Format::eR16G16B16A16Sfloat:
			switch (used_components) {
				case 1: return vk::Format::eR16Sfloat;
				case 2: return vk::Format::eR16G16Sfloat;
				default: return format;
			}
		case vk::Format::eR8G8B8A8Unorm:
			switch (used_components) {
				case 1: return vk::Format::eR8Unorm;
				case 2: return vk::Format::eR8G8Unorm;
				default: return format;
			}
		case vk::Format::eR8G8B8A8Snorm:
			return used_components == 2 ? vk::Format::eR8G8Snorm : format;
		case vk::Format::eR8G8B8A8Uint:
			switch (used_components) {
				case 1: return vk::Format::eR8Uint;
				case 2: return vk::Format::eR8G8Uint;
				default: return format;
			}
		default: return format;
	}
}

// The raw formats InputFormat maps by hand, which VulkanFormat does not know.
constexpr uint32_t RAW_FORMAT_16_16_SINT = 113;
constexpr uint32_t RAW_FORMAT_121       = 121;

// Whether InputFormat has a host format for the attribute, which it EXITs without.
bool InputFormatKnown(const ShaderBufferResource& resource) {
	if (resource.RawFormat() == RAW_FORMAT_16_16_SINT || resource.RawFormat() == RAW_FORMAT_121) {
		return true;
	}
	return VulkanFormat(resource.Format()) != vk::Format::eUndefined &&
	       ShaderRecompiler::Format::GetFormatInfo(resource.Format()).component_count != 0;
}

vk::Format InputFormat(const ShaderBufferResource& resource, uint32_t used_components) {
	switch (resource.RawFormat()) {
		// Temporary: invalid PS5 buffer format 113 (VertexAttribFormat::k16_16SInt).
		case 113: return NarrowInputFormat(vk::Format::eR32G32B32A32Sfloat, used_components);
		case 121: return vk::Format::eR16G16Sfloat;
		default: break;
	}
	if (!InputFormatKnown(resource)) {
		EXIT("unknown vertex format: fmt = %u\n", resource.RawFormat());
	}
	return NarrowInputFormat(VulkanFormat(resource.Format()), used_components);
}

// Whether VertexInputFormats can run on the info without EXIT, which it does on an attribute that is
// not one to four registers wide or whose format has no host format.
bool VertexInputsSupported(const ShaderVertexInputInfo& input_info) {
	constexpr int max_registers = 4;
	for (int index = 0; index < input_info.resources_num; index++) {
		const auto registers_num = input_info.resources_dst[index].registers_num;
		if (registers_num < 1 || registers_num > max_registers || !InputFormatKnown(input_info.resources[index])) {
			return false;
		}
	}
	return true;
}

// The host format of each vertex input attribute, narrowed to the components the stage reads.
std::vector<vk::Format> VertexInputFormats(const ShaderVertexInputInfo&                    input_info,
                                           const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	std::vector<vk::Format> formats(input_info.resources_num);
	for (int index = 0; index < input_info.resources_num; index++) {
		const auto registers_num = input_info.resources_dst[index].registers_num;
		EXIT_IF(registers_num < 1 || registers_num > 4);
		const auto fetched = program.info.vertex_fetch_components[index];
		formats[index] = InputFormat(input_info.resources[index],
		                             fetched > 0 ? fetched : static_cast<uint32_t>(registers_num));
	}
	return formats;
}

constexpr uint32_t AmdVendorId   = 0x1002u;
constexpr uint32_t AppleVendorId = 0x106Bu;

// Why the opt-in does or does not reach the shaders.
enum class HostFloatMinMaxVerdict { FlagOff, Enabled, AmdNative, Apple, MissingFloat32Property };

// Decides whether the opt-in reaches the shaders on this device.
HostFloatMinMaxVerdict HostFloatMinMaxVerdictFor(const GraphicContext& graphics, bool flag_on) {
	if (!flag_on) {
		return HostFloatMinMaxVerdict::FlagOff;
	}
	const auto vendor = graphics.physical_device_properties.vendorID;
	if (vendor == AmdVendorId) {
		return HostFloatMinMaxVerdict::AmdNative;
	}
	// MoltenVK may report the vendor of the GPU underneath, so its driver id counts as well.
	if (vendor == AppleVendorId || graphics.moltenvk_driver) {
		return HostFloatMinMaxVerdict::Apple;
	}
	// The SignedZeroInfNanPreserve mode the shaders request is only honoured with the property.
	if (!graphics.float32_signed_zero_inf_nan_preserve) {
		return HostFloatMinMaxVerdict::MissingFloat32Property;
	}
	return HostFloatMinMaxVerdict::Enabled;
}

// Tells once per process whether the opt-in took effect, so a run can be checked from its log.
void LogHostFloatMinMaxVerdict(HostFloatMinMaxVerdict verdict) {
	// Returning before the once-flag keeps a flag-off call from using up the one log line.
	if (verdict == HostFloatMinMaxVerdict::FlagOff) {
		return;
	}
	static std::once_flag logged;
	std::call_once(logged, [verdict] {
		switch (verdict) {
			case HostFloatMinMaxVerdict::Enabled:
				LOGF("Native float min/max: host min/max enabled\n");
				break;
			case HostFloatMinMaxVerdict::AmdNative:
				LOGF("Native float min/max: host min/max not needed, AMD already uses it\n");
				break;
			case HostFloatMinMaxVerdict::Apple:
				LOGF("Native float min/max: host min/max disabled, Apple/MoltenVK is not supported\n");
				break;
			case HostFloatMinMaxVerdict::MissingFloat32Property:
				LOGF("Native float min/max: host min/max disabled, the device lacks "
				     "shaderSignedZeroInfNanPreserveFloat32\n");
				break;
			case HostFloatMinMaxVerdict::FlagOff: break; // Handled above.
		}
	});
}

} // namespace

// The graphics prescan's worker uses the GPU thread's own two, so its vertex records and its
// pre-rasterization keys are made the same way.
std::vector<vk::Format> PrescanVertexInputFormats(const ShaderVertexInputInfo&                    input_info,
                                                  const ShaderRecompiler::IR::CompiledShaderInfo& program) {
	return VertexInputFormats(input_info, program);
}

bool PrescanVertexInputsSupported(const ShaderVertexInputInfo& input_info) {
	return VertexInputsSupported(input_info);
}

vk::PolygonMode PrescanPolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	return ResolvePolygonMode(mode, cull_front, cull_back);
}

uint32_t HostComputeSubgroupSize(const GraphicContext& graphics) {
	return graphics.SupportsComputeWave64() ? 64u : 32u;
}

ShaderRecompiler::Spirv::Profile SpirvProfile(const GraphicContext& graphics) {
	const bool flag_on = Config::NativeFloatMinMaxEnabled();
	const auto verdict = HostFloatMinMaxVerdictFor(graphics, flag_on);
	LogHostFloatMinMaxVerdict(verdict);
	// AMD GPUs run NMin/NMax as v_min/v_max, which give the guest's NaN and signed-zero
	// results. GLSL.std.450 alone does not order -0 below +0.
	return {.maintenance8           = graphics.maintenance8_enabled,
	        .native_float_min_max   = graphics.physical_device_properties.vendorID == AmdVendorId,
	        .host_float_min_max     = verdict == HostFloatMinMaxVerdict::Enabled,
	        .device_clock_period_ps = graphics.device_clock_period_ps,
	        .robust_buffer_access   = graphics.robust_buffer_access_enabled,
	        .r8_srgb_views          = graphics.r8_srgb_views,
	        .r8g8_srgb_views        = graphics.r8g8_srgb_views,
	        .select_guarded_loads   = Config::SelectGuardedLoadsEnabled(),
	        .single_compare_record_guard = Config::SingleCompareRecordGuardEnabled(),
	        .elide_exec_selects = Config::ElideExecSelectsEnabled()};
}

std::filesystem::path ShaderStorePath() {
	if (KYTY_BUILD != KYTY_BUILD_RELEASE || !Config::ShaderCacheEnabled() ||
	    Config::GraphicsDebugDumpEnabled()) {
		return {};
	}
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return {};
	}
	return std::filesystem::path("_PipelineCache") / (title_id + ".shaders");
}

bool GraphicsPipelineKey::operator==(const GraphicsPipelineKey& other) const {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	return XXH3_64bits(&key, sizeof(key));
}

std::span<const uint32_t> PipelineCache::Program::FunctionCode() const {
	return call_source
	    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
	    : std::span<const uint32_t>{};
}

bool PipelineCache::Program::RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
	const auto revision = call_source->revision;
	ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
	resource_plan.source_reads = call_source->reads;
	return revision != call_source->revision;
}

template <bool HasCalls>
ShaderModule* PipelineCache::Program::FindModule(const ShaderRecompiler::IR::SrtRuntime& runtime,
                                                 uint32_t push_data_cursor) {
	EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(resource_plan, runtime, resources,
	                                                    specialization));
	for (auto& candidate: modules) {
		const auto& layout = candidate.info.bindings;
		if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
		        push_data_cursor, layout.ShaderDataDwords()) ||
		    candidate.specialization != specialization) {
			continue;
		}
		if constexpr (HasCalls) {
			if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
		}
		return &candidate;
	}
	return nullptr;
}

uint64_t PipelineCache::ModuleId(const ProgramKey& key, uint32_t push_data_start_dword,
                                 const ShaderRecompiler::IR::ResourceSpecialization& specialization,
                                 std::span<const uint32_t>                           function_code) {
	return ShaderModuleId(key.stage, key.hash, key.user_data_count, key.static_state,
	                      push_data_start_dword, specialization, function_code);
}

ShaderModule& PipelineCache::CompileModule(Program& program, const ProgramKey& key,
                                           ShaderRecompiler::CompileOptions  options,
                                           ShaderRecompiler::TranslateResult translated,
                                           uint32_t                          push_data_cursor) {
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
	                                         Common::DebugCounters::Counter::ShaderCompileNs,
	                                         Common::DebugCounters::Peak::ShaderCompileNs);
	const auto profile = SpirvProfile(m_graphics);
	options.profile    = &profile;
	auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
	                                               program.specialization, push_data_cursor);
	auto info = std::move(result.program).TakeCompiledInfo();

	const auto function_code = program.FunctionCode();
	const auto id = ModuleId(key, info.bindings.push_data_start_dword, program.specialization, function_code);

	const auto* stage = ShaderRecompiler::StageName(key.stage);
	DumpShader(result.spirv, fmt::format("{}_{:016x}_{:016x}.spv", stage, key.hash, id));
	ValidateShaderSpirv(stage, key.hash, result.spirv, profile.maintenance8);
	auto& module = program.modules.emplace_back(ShaderModule {
	    .id             = id,
	    .module         = CompileSPV(result.spirv, m_graphics.device),
	    .gpl_module     = CompileSplitPixelModule(m_graphics, key.stage, result.spirv),
	    .info           = std::move(info),
	    .specialization = program.specialization,
	    .function_code  = {function_code.begin(), function_code.end()},
	});
	m_compiled[static_cast<size_t>(key.stage)]++;
	// Guest geometry shaders are compiled through the host mesh stage.
	std::printf("Shaders: VS %u | PS %u | CS %u | GS %u | LS %u | HS %u | TES %u\n",
	            m_compiled[static_cast<size_t>(ShaderType::Vertex)],
	            m_compiled[static_cast<size_t>(ShaderType::Pixel)],
	            m_compiled[static_cast<size_t>(ShaderType::Compute)],
	            m_compiled[static_cast<size_t>(ShaderType::Mesh)],
	            m_compiled[static_cast<size_t>(ShaderType::Local)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationControl)],
	            m_compiled[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
	// The inputs of these come from the stage's static key.
	if (key.stage == ShaderType::Vertex || key.stage == ShaderType::Local) {
		module.vertex_formats = VertexInputFormats(*options.input_info.vertex, module.info);
	} else if (key.stage == ShaderType::Pixel &&
	           options.input_info.pixel->parameter_mode == ShaderPixelParameterMode::Rectangle) {
		module.rect_parameters = RectListParameters(*options.input_info.pixel, module.info);
	} else if (key.stage == ShaderType::Mesh) {
		module.mesh_task = options.input_info.vertex->indirect_draw;
	}
	// Programs with GPU-selected calls depend on their callees' current code.
	if (program.call_source == nullptr) {
		StoreModule(key, program, module, result.spirv);
	}
	return module;
}

template <typename InputInfo>
const ShaderModule* PipelineCache::GetProgram(const ShaderParams& params, InputInfo& input_info,
                                              uint32_t& push_data_cursor) {
	// A compute lookup forgets the last inline compile it noted, so a hit never inherits a note.
	if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
		ResetInlineCompute();
	}
	// Modules that the shader store generator made are taken in first, so a lookup finds what is
	// there. Without generation the first test fails. With it and nothing posted the second does,
	// so the cost of an empty inbox is one relaxed load. The GPU thread test is last, since it is
	// the dearest, and it keeps any other thread that reaches a lookup from draining.
	if (m_generation && m_has_generated.load(std::memory_order_relaxed) &&
	    GuestGpu::IsGpuThread()) {
		DrainGenerated();
	}
	const auto user_data = std::span(params.user_data).first(params.user_data_count);
	ShaderRecompiler::IR::SrtRuntime runtime {
	    .user_data                  = user_data,
	    .shader_base                = params.Base(),
	    .read_specialization_memory = ReadShaderGuestMemory,
	    .image_heap                 = m_image_heap,
	};
	ShaderRecompiler::CompileOptions options {.wave_size = input_info.wave_size};
	if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
		options.stage            = input_info.logical_stage;
		options.input_info.vertex = &input_info;
		options.user_data_base   = 8;
		if (options.stage == ShaderType::Mesh ||
		    options.stage == ShaderType::TessellationControl) {
			options.user_data_base = 0;
			options.wave_size =
			    options.stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
		}
	} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
		options.stage            = ShaderType::Pixel;
		options.input_info.pixel = &input_info;
	} else {
		static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
		options.stage              = ShaderType::Compute;
		options.input_info.compute = &input_info;
		runtime.workgroup_counts   = input_info.workgroup_counts;
		runtime.indirect_args_address = input_info.indirect_args_address;
	}
	m_lookup_key.stage           = options.stage;
	m_lookup_key.hash            = params.hash;
	m_lookup_key.user_data_count = params.user_data_count;
	BuildStageStaticKey(input_info, m_lookup_key.static_state);
	auto entry = m_program_cache.find(m_lookup_key);
	const auto bind = [&](ShaderModule& module) {
		// A prescan module's record goes to the store at its first use, so a guess the game never
		// runs leaves nothing in the store.
		StoreOnFirstUse(entry->second, module);
		if constexpr (!std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			NoteGraphicsPrescanBind(module.id, entry->first.stage);
		}
		input_info.stage = {.module = &module, .resources = &entry->second.resources};
		module.info.bindings.AdvancePushData(push_data_cursor);
		return &module;
	};
	bool source_changed = false;
	if (entry != m_program_cache.end()) {
		auto& cached = entry->second;
		if (cached.skip_dispatch) return nullptr;
		ShaderModule* module = nullptr;
		if (cached.call_source) {
			source_changed = cached.RefreshCallSource(runtime);
			if (!source_changed) module = cached.FindModule<true>(runtime, push_data_cursor);
		} else {
			module = cached.FindModule<false>(runtime, push_data_cursor);
		}
		if (module != nullptr) return bind(*module);
	}
	if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
		NoteInlineComputeMiss(input_info);
	}

	// Everything from here to the driver's pipeline compile is translation, SPIR-V emission and
	// module creation. It blocks the GPU thread only when this runs on it.
	Common::FirstRunReport::TranslateScope translate_timer(
	    options.stage == ShaderType::Compute ? Common::FirstRunReport::Kind::Compute
	                                         : Common::FirstRunReport::Kind::Graphics,
	    ShaderRecompiler::StageName(options.stage), params.hash, GuestGpu::IsGpuThread());
	options.shader_hash     = params.hash;
	options.user_data       = user_data;
	options.back_code       = params.back_code;
	options.dump_ir         = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
	options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
	options.structured_jump_tables = Config::StructuredJumpTablesEnabled();
	ShaderRecompiler::ShaderSource source;
	if (entry == m_program_cache.end()) {
		DumpShader(params.code, fmt::format("original/{}_{:016x}.bin",
		                                    ShaderRecompiler::StageName(options.stage), params.hash));
		source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		if (source.unsupported_call_pc != UINT32_MAX) {
			PipelineCacheLog("!!! TEMPORARY COMPUTE DISPATCH SKIP !!! shader=0x{:016x} pc=0x{:08x} "
			                 "reason=unsupported GPU-selected callable target; rendering is incomplete",
			                 params.hash, source.unsupported_call_pc);
			m_program_cache.emplace(m_lookup_key, Program {.skip_dispatch = true});
			if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
				NoteGeneratedMiss(params.hash, 0, false, true);
			}
			return nullptr;
		}
	}
	auto* current = entry == m_program_cache.end() ? &source : entry->second.call_source.get();
	ShaderRecompiler::TranslateResult translated;
	if (current != nullptr) {
		if (!current->calls.empty()) {
			if (entry == m_program_cache.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
			translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
			translated.program.source_reads = current->reads;
		} else {
			translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
		}
	} else {
		translated = ShaderRecompiler::TranslateProgram(params.code, options);
	}
	if (entry == m_program_cache.end()) {
		auto retained = !source.calls.empty()
		                    ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source))
		                    : nullptr;
		entry = m_program_cache.emplace(m_lookup_key, Program {
		    .call_source   = std::move(retained),
		    .resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program),
		}).first;
		auto& program = entry->second;
		EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
		    program.resource_plan, runtime, program.resources, program.specialization));
	} else if (source_changed) {
		entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
		if (auto* module = entry->second.FindModule<true>(runtime, push_data_cursor)) {
			return bind(*module);
		}
	} else if (m_generation != nullptr && GuestGpu::IsGpuThread()) {
		// The runtime translated a program that already exists and no module of it fits, so a
		// program that generation made is checked against the runtime's own plan. The planner is the
		// GPU thread's, so no other thread makes the check.
		CheckGeneratedPlan(entry->second, translated);
	}
	auto& compiled = CompileModule(entry->second, entry->first, options, std::move(translated),
	                               push_data_cursor);
	if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
		NoteGeneratedMiss(params.hash, compiled.id, entry->second.call_source != nullptr, false);
		NoteInlineComputeModule(compiled.id);
	} else {
		NoteGraphicsPrescanMiss(entry->first, compiled, params.Base());
	}
	return bind(compiled);
}

// Opens the driver cache and, with --async-pipelines or --deferred-pipeline-draws on, starts the
// build workers.
PipelineCache::PipelineCache(GraphicContext&                         graphics,
                             ShaderRecompiler::IR::ImageHeapSource* image_heap)
    : m_graphics(graphics), m_set_layouts(graphics.device),
      m_libraries(std::make_unique<GraphicsLibraryCache>(graphics.device)), m_image_heap(image_heap),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	// Read once. With both flags off no thread exists and every miss builds on the calling thread.
	// Deferred draws wait for their builds together, so they get more workers. With both flags on
	// the skips share that larger pool.
	if (Config::AsyncPipelinesEnabled() || Config::DeferredPipelineDrawsEnabled()) {
		const uint32_t worker_cap = Config::DeferredPipelineDrawsEnabled()
		                                ? DEFERRED_PIPELINE_BUILD_WORKERS
		                                : PIPELINE_BUILD_WORKERS;
		m_build_queue = std::make_unique<PipelineBuildQueue>(PipelineBuildWorkerCount(worker_cap));
	}
}

// Stops the builds, saves the driver cache and destroys every pipeline, layout and module.
PipelineCache::~PipelineCache() {
	// Generated modules that no drain took still own Vulkan handles, which go before the device does.
	StopStoreGeneration();
	// For a cache made without a render context, which has no one to stop the builds earlier.
	StopBuilds();
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& pipeline: pipelines | std::views::values) {
			DestroyPipelineObjects(m_graphics.device, pipeline);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	// Every pipeline linked from a library is gone, so the libraries may go. The layouts they were made
	// with are the set layouts cache's, which is destroyed after this body.
	m_libraries.reset();
	for (const auto& program: m_program_cache | std::views::values) {
		for (const auto& module: program.modules) {
			m_graphics.device.destroyShaderModule(module.module, nullptr);
			m_graphics.device.destroyShaderModule(module.gpl_module, nullptr);
		}
	}
	m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	// A cold run must neither read a warm driver cache nor leave its own behind for the next run.
	// It still gets an empty cache object, as a true first run does, so its timings match. With no
	// path set, Save has nowhere to write.
	const bool cold = Common::FirstRunReport::ColdEnabled();
	if (cold) {
		PipelineCacheLog("Vulkan pipeline cache: starting empty and unsaved (--cold-pipelines)");
	} else {
		m_driver_cache_path = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	}
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = !cold && Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else if (!cold) {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr || m_driver_cache_path.empty()) {
		return;
	}
	m_driver_cache_dirty = true;
	if (std::chrono::steady_clock::now() >= m_next_driver_cache_save) {
		Save();
	}
}

void PipelineCache::Save() {
	if (m_driver_cache == nullptr || m_driver_cache_path.empty() || !m_driver_cache_dirty) {
		return;
	}
	m_next_driver_cache_save = std::chrono::steady_clock::now() + DriverCacheSaveInterval;
	// Cleared before the data is read, so a worker that adds to the cache while this runs leaves it
	// dirty for the next save. A failure below sets it again.
	m_driver_cache_dirty = false;

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		m_driver_cache_dirty = true;
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		m_driver_cache_dirty = true;
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		m_driver_cache_dirty = true;
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
}

// The pixel input helpers count targets on their own, so they must count what the renderer does.
static_assert(PIXEL_COLOR_TARGETS == RENDER_COLOR_ATTACHMENTS_MAX);

void PipelineCache::GetGraphicsPrograms(const CommandBuffer& command, DrawRenderState& state) {
	const auto& context           = command.GetRegisters();
	const auto& user_config       = command.GetUserConfig();
	const auto& sh                = context.GetShaderRegisters();
	const auto& vertex_regs       = command.GetShaders().GetVs();
	const auto& pixel_regs        = command.GetShaders().GetPs();
	auto&       vertex_info       = state.vertex_info;
	auto&       pixel_info        = state.ps_input_info;
	const bool  object_primitive_id = context.HasObjectPrimitiveIdOutput();
	// The graphics prescan works these out with the same helpers, so its keys match these.
	const uint32_t color_output_mask = PixelColorOutputMask(context);
	const bool     pixel_active      = PixelStageActive(context, pixel_regs, color_output_mask);
	state.ps_active = pixel_active;
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
		// Reflection decodes both programs; the key also covers the VGT fields in tess.
		const uint64_t layout_key[] = {vertex_params[0].hash, vertex_params[1].hash,
		                               uint64_t {sh.m_vgtLsHsConfig & 0xfff00u} << 32u |
		                                   sh.m_vgtTfParam};
		const auto [layout, inserted] = m_tess_layouts.try_emplace(
		    XXH3_64bits(layout_key, sizeof(layout_key)), vertex_info[0].tess);
		if (inserted) {
			ShaderRecompiler::AnalyzeTessellationPrograms(vertex_params[0].code,
			                                              vertex_params[1].code, layout->second);
			StoreLayout(layout->first, layout->second);
		}
		for (auto& stage: vertex_info) {
			stage.tess = layout->second;
		}
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	if (!tess_active && !mesh_active) {
		vertex_info[0].indirect_offsets = state.indirect_offsets;
		vertex_info[0].indirect_draw    = state.indirect_draw;
	}
	if (object_primitive_id) {
		// PA_CL_OBJPRIM_ID_CNTL = 2: VGT_OBJECT_ID plus the pipelined 32-bit primitive ID.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: dropping mesh draws the host does not support (shader 0x{:016x}).\n",
				    vertex_params[0].hash));
			}
			return;
		}
		// A task shader launches each record's groups and passes its draw parameters in a
		// payload beside the LDS, the allocation pair and the Layer array; without room for
		// it, the CPU reads the records.
		if (state.indirect_draw &&
		    (mesh.lds_size_dwords + 2u + mesh.max_vertices +
		     ShaderRecompiler::IR::PushData::MeshTaskPayloadDwordCount) * sizeof(uint32_t) <=
		        limits.maxMeshPayloadAndSharedMemorySize) {
			vertex_info[0].indirect_offsets = state.indirect_offsets;
			vertex_info[0].indirect_draw    = true;
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		const auto target_export_mapping = PixelTargetExportMappingOf(context, color_output_mask);
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		PackedUnormRejection rejected;
		const PixelDrawShape shape {.primitive_type      = user_config.GetPrimType(),
		                            .object_primitive_id = object_primitive_id,
		                            .mesh_active         = mesh_active,
		                            .tess_active         = tess_active};
		if (FinishPixelInputInfo(context, shape, pixel_info, &rejected) != PixelInputStatus::Ok) {
			EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
			     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
			     rejected.slot, rejected.export_index, rejected.mode, rejected.target_mask,
			     rejected.shader_mask, rejected.mapping, static_cast<uint32_t>(rejected.blend));
		}
	}
	auto&       last_stage   = vertex_info[tess_active ? 2u : 0u];
	const auto& clip_control = context.GetClipControl();
	if (clip_control.clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = last_stage.clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	} else if (clip_control.user_clip_planes != 0) {
		// CLIP_DISABLE also disables user planes. How they combine with shader clip and cull
		// distances is unverified; draw push constants only reach VS/mesh and PS, and the
		// generated RECTLIST stages do not forward clip distances.
		EXIT_NOT_IMPLEMENTED((last_stage.pa_cl_vs_out_cntl & 0xffffu) != 0 || tess_active ||
		                     Prospero::IsRectList(user_config.GetPrimType()));
		last_stage.user_clip_planes = clip_control.user_clip_planes;
	}
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id, last_stage.user_clip_planes);
	if (pixel_active) {
		const auto& pixel_module = *GetProgram(pixel_params, pixel_info, push_data_cursor);
		for (const auto& output: pixel_module.info.info.outputs) {
			if (output.kind == ShaderRecompiler::IR::StageOutputKind::Mrt ||
			    output.kind == ShaderRecompiler::IR::StageOutputKind::ObjectPrimitiveId) {
				state.mrt_mask |= 1u << output.location;
			}
		}
		state.mrt_mask &= color_output_mask;
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		GetProgram(vertex_params[i], vertex_info[i], push_data_cursor);
	}
}

const ShaderModule* PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                                     ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = HostComputeSubgroupSize(m_graphics);
	const auto        params      = PrepareProgram(regs, input_info);
	input_info.lds_storage = ComputeLdsNeedsStorage(
	    input_info.lds_size_dwords,
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize);
	uint32_t          push_data_cursor = 0;
	return GetProgram(params, input_info, push_data_cursor);
}

namespace {

// Fills the blend fields of one color target's key. A mapping the host cannot blend directly drops
// the blend, with a warning the first time.
void FillBlendKey(PipelineStaticParameters& params, uint32_t slot, const HW::BlendControl& control,
                  const Prospero::ColorComponentMapping& mapping,
                  ShaderAlphaBlendSource alpha_source) {
	if (params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
	    ClassifyBlendMapping(control, mapping) != BlendMappingSupport::Direct) {
		params.blend_enable[slot] = false;
		static std::atomic_bool warned = false;
		if (!warned.exchange(true, std::memory_order_relaxed)) {
			Log::WriteToConsoleAndLog(fmt::format(
			    "Warning: blending disabled for unsupported color mapping "
			    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
			    slot, mapping.packed, control.color_srcblend, control.color_destblend,
			    control.alpha_srcblend, control.alpha_destblend,
			    control.separate_alpha_blend ? 1 : 0));
		}
	}
	if (!params.blend_enable[slot]) {
		return;
	}
	auto blend = control;
	switch (alpha_source) {
		case ShaderAlphaBlendSource::SourceAlpha:
			blend.color_srcblend       = RemapSourceAlphaFactor(blend.color_srcblend);
			blend.color_destblend      = RemapSourceAlphaFactor(blend.color_destblend);
			blend.separate_alpha_blend = false;
			break;
		case ShaderAlphaBlendSource::SourceAlphaOne:
		case ShaderAlphaBlendSource::SourceAlphaZero:
			// The second source carries the mapped source factor; its alpha stays logical Sa.
			blend.color_srcblend  = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
			blend.color_destblend = static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
			blend.separate_alpha_blend = false;
			break;
		case ShaderAlphaBlendSource::None: break;
	}
	params.color_srcblend[slot]       = blend.color_srcblend;
	params.color_comb_fcn[slot]       = blend.color_comb_fcn;
	params.color_destblend[slot]      = blend.color_destblend;
	params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
	if (blend.separate_alpha_blend) {
		params.alpha_srcblend[slot]  = blend.alpha_srcblend;
		params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
		params.alpha_destblend[slot] = blend.alpha_destblend;
	}
}

// Fills the key's color attachments and returns their sample count, or 0 with none.
uint32_t FillColorKey(GraphicsPipelineKey& key, const HW::Context& ctx,
                      const DrawRenderState& state) {
	auto&    static_params      = key.static_params;
	auto&    rendering          = key.rendering;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < state.color_count; i++) {
		const auto& color = state.color_info[i];
		const auto  slot  = color.target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!color.image_id || color.desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = color.export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), slot));
		if (color.desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = color.desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = color.desc.info.samples;
		} else if (attachment_samples != color.desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, color.desc.info.samples);
		}
		const auto& control = ctx.GetBlendControl(slot);
		auto        alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && state.ps_active) {
			alpha_source = state.ps_input_info.alpha_blend_source;
		}
		static_params.blend_enable[slot] =
		    control.enable && !ctx.GetRenderTarget(slot).info.blend_bypass;
		FillBlendKey(static_params, slot, control, color.export_mapping, alpha_source);
	}
	return attachment_samples;
}

// Whether the draw has a depth or stencil attachment.
bool HasDepthAttachment(const RenderDepthInfo& depth) {
	return depth.desc.view_info.format != vk::Format::eUndefined &&
	       static_cast<bool>(depth.image_id);
}

// Fills the key's depth and stencil formats and returns the sample count shared with the colors.
uint32_t FillDepthKey(GraphicsPipelineKey& key, const RenderDepthInfo& depth,
                      uint32_t attachment_samples) {
	if (!HasDepthAttachment(depth)) {
		return attachment_samples;
	}
	auto&      rendering = key.rendering;
	const auto aspects   = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
	rendering.depth_format = aspects & vk::ImageAspectFlagBits::eDepth
	                             ? depth.desc.view_info.format
	                             : vk::Format::eUndefined;
	rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
	                               ? depth.desc.view_info.format
	                               : vk::Format::eUndefined;
	if (attachment_samples == 0) {
		return depth.desc.info.samples;
	}
	if (attachment_samples != depth.desc.info.samples) {
		EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
		     depth.desc.info.samples);
	}
	return attachment_samples;
}

// Fills the key's vertex bindings and attributes from the vertex stage's fetches.
void FillVertexInputKey(GraphicsPipelineKey& key, const ShaderVertexInputInfo& vs_input_info) {
	EXIT_IF(vs_input_info.buffers_num < 0 ||
	        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
	        vs_input_info.resources_num < 0 ||
	        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
	key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
	key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
	for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
		const auto& buffer = vs_input_info.buffers[binding];
		key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
		                                      .instance = buffer.fetch_index != 0};
	}
	for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
		const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
		EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
		key.vertex_input.attributes[attribute] = {
		    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
		                                    vs_input_info.buffers[binding].addr),
		    .binding = static_cast<uint8_t>(binding),
		};
	}
}

// The stages of a draw in key order, with null for an absent one.
std::array<const ShaderModule*, 4> DrawStages(const DrawRenderState& state) {
	std::array<const ShaderModule*, 4> stages {};
	const auto                         vertex_info = state.VertexStages();
	for (uint32_t i = 0; i < vertex_info.size(); i++) {
		stages[i] = vertex_info[i].stage.module;
	}
	if (state.ps_active) {
		stages[3] = state.ps_input_info.stage.module;
	}
	return stages;
}

// Fills the key's fixed-function state from the registers and the draw.
void FillStaticKey(PipelineStaticParameters& params, const CommandBuffer& command,
                   const DrawRenderState& state, uint32_t attachment_samples, bool mesh) {
	const auto& ctx          = command.GetRegisters();
	const auto& mc           = ctx.GetModeControl();
	const auto& clip_control = ctx.GetClipControl();
	params.negative_one_to_one      = !clip_control.dx_clip_space;
	params.depth_clip_enable        = clip_control.IsZClipEnabled();
	params.topology                 = state.topology;
	params.primitive_restart_enable = state.primitive_restart_enable;
	params.samples                  = attachment_samples;
	params.sample_shading_enable =
	    state.ps_active && attachment_samples > 1 && state.ps_input_info.ps_sample_shading;
	params.alpha_to_coverage_enable = state.ps_active && ctx.GetShaderRegisters().AlphaToMask();
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	params.provoking_vtx_last = mc.provoking_vtx_last;
	params.polygon_mode =
	    ResolvePolygonMode(mc, !rect_list && mc.cull_front, !rect_list && mc.cull_back);
	const auto vertex_info = state.VertexStages();
	if (vertex_info.size() == 3) {
		params.patch_control_points = vertex_info.front().tess.input_control_points;
	} else if (!mesh && params.topology == vk::PrimitiveTopology::ePatchList) {
		// Rectangle lists expand through generated tessellation stages.
		params.patch_control_points = 3;
	}
}

// Prints the stage inputs of a draw whose pipeline is about to be created, when dumps are on.
void DumpPipelineInputs(const DrawRenderState& state) {
	if (!graphics_debug_dump_enabled()) {
		return;
	}
	ShaderDbgDumpInputInfo(state.VertexStages().front());
	if (state.ps_active) {
		ShaderDbgDumpInputInfo(state.ps_input_info);
	}
}

} // namespace

GraphicsPipelineKey PipelineCache::MakeGraphicsPipelineKey(const CommandBuffer&   command,
                                                           const DrawRenderState& state,
                                                           const Stages&          stages) const {
	const auto  vertex_info = state.VertexStages();
	const auto& ctx         = command.GetRegisters();
	const bool  mesh        = stages[0]->info.stage == ShaderType::Mesh;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < stages.size(); i++) {
		if (stages[i] != nullptr) {
			key.stage_ids[i] = stages[i]->id;
		}
	}
	auto attachment_samples = FillColorKey(key, ctx, state);
	attachment_samples      = FillDepthKey(key, state.depth_info, attachment_samples);
	if (state.color_count == 0 && !HasDepthAttachment(state.depth_info)) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	FillStaticKey(key.static_params, command, state, attachment_samples, mesh);
	if (!mesh) {
		FillVertexInputKey(key, vertex_info.front());
	}
	return key;
}

PipelineCache::Pipeline& PipelineCache::CreateGraphicsPipelineHere(Pipeline&                  slot,
                                                                   const GraphicsPipelineKey& key,
                                                                   const Stages&              stages,
                                                                   Common::GpuStall::HoldCause cause) {
	{
		// A compile on the GPU command thread stops all guest rendering, so guest audio may wait
		// for it. One on any other thread leaves the game running and must not hold audio.
		std::optional<Common::GpuStall::Scope> stall;
		if (GuestGpu::IsGpuThread()) {
			stall.emplace(cause);
		}
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, m_set_layouts, slot, key, stages, m_driver_cache,
		                       m_libraries.get());
	}
	StorePipeline(key);
	CheckpointDriverCache();
	return slot;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(const CommandBuffer&   command,
                                                            const DrawRenderState& state) {
	return *AcquireGraphicsPipeline(command, state, nullptr);
}

PipelineCache::Pipeline* PipelineCache::AcquireGraphicsPipeline(const CommandBuffer&   command,
                                                                const DrawRenderState& state,
                                                                const PipelineSkipPolicy* policy,
                                                                DeferredDraw* deferred) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);
	const auto stages = DrawStages(state);
	const auto key    = MakeGraphicsPipelineKey(command, state, stages);
	if (m_build_queue != nullptr) {
		// The change in the report's blocked graphics time across the call is what a refused draw
		// waited, whether it compiled here or waited for a worker.
		const double blocked_before = Common::FirstRunReport::BlockedMs(Common::FirstRunReport::Kind::Graphics);
		// The report flag is read once, so the clock and the row agree even if it changes meanwhile,
		// and the clock is read only for the report.
		MissedDraw missed {.timed = Common::FirstRunReport::Enabled()};
		auto*      pipeline = AcquireBuiltPipeline(key, stages, state, policy, deferred, missed);
		// A draw that could not be deferred and got its pipeline waited for it. One that a skip took
		// has no pipeline and waited for nothing.
		if (missed.timed && missed.blocker != nullptr && pipeline != nullptr) {
			NoteIneligibleWait(missed, stages);
		}
		// Refusals are the S2 rules' own, so they are only kept when skips are on, and a deferred
		// draw waited for nothing.
		const bool deferred_now = deferred != nullptr && deferred->active;
		if (Config::AsyncPipelinesEnabled() && !deferred_now && policy != nullptr &&
		    policy->Refusal() != nullptr) {
			Common::FirstRunReport::AddRefusal(
			    policy->Refusal(),
			    Common::FirstRunReport::BlockedMs(Common::FirstRunReport::Kind::Graphics) - blocked_before);
		}
		return pipeline;
	}

	auto [iter, inserted] = m_graphics_pipelines.try_emplace(key);
	if (!inserted) {
		return &iter->second;
	}
	DumpPipelineInputs(state);
	return &CreateGraphicsPipelineHere(iter->second, key, stages,
	                                   Common::GpuStall::HoldCause::RequiredGraphics);
}

void PipelineCache::NoteIneligibleWait(const MissedDraw& missed, const Stages& stages) {
	// The vertex stage is first and the pixel stage last, as the compile rows name them.
	constexpr size_t VERTEX_STAGE = 0;
	constexpr size_t PIXEL_STAGE  = 3;
	const auto*      pixel        = stages[PIXEL_STAGE];
	const double     waited_ms =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - missed.start).count();
	// A draw refused for its thread may be off the GPU thread, where its wait stops nothing.
	Common::FirstRunReport::AddDeferredIneligibleWait({.reason     = missed.blocker,
	                                                   .vs         = stages[VERTEX_STAGE]->id,
	                                                   .ps         = pixel != nullptr ? pixel->id : 0,
	                                                   .waited_ms  = waited_ms,
	                                                   .gpu_thread = GuestGpu::IsGpuThread()});
}

// Starts the clock for a build requested now.
PipelineCache::GraphicsBuild::GraphicsBuild(const GraphicsPipelineKey& request_key,
                                            const Stages&              request_stages)
    : key(request_key), stages(request_stages), requested(std::chrono::steady_clock::now()) {}

const char* PipelineCache::DeferBlocker(const GraphicsPipelineKey& key, const Stages& stages,
                                        const PipelineSkipPolicy* policy) const {
	// A shared heap set is bound in the buffer a draw is met in, which the hole's buffer lacks.
	const bool heap = std::ranges::any_of(stages, [](const ShaderModule* stage) {
		return stage != nullptr && stage->info.info.uses_image_heap;
	});
	if (heap) {
		return "image_heap";
	}
	// A hole writes its descriptors when it is recorded, so a set would be allocated and written
	// during the resolve. That is opt in, and without it only pushed descriptors are deferred. The
	// layout code predicts which this draw gets, since its layout does not exist before the build.
	// With the split layout every draw whose VS has bindings gets a set, so the key is asked too.
	if (!Config::DeferredNonPushDrawsEnabled() &&
	    !PredictPushDescriptors(m_graphics, key, stages)) {
		return "non_push";
	}
	return policy != nullptr ? policy->DeferBlocker() : "unsupported";
}

void PipelineCache::NoteDeferredWouldSkip(const std::shared_ptr<GraphicsBuild>& build, const char* refusal) {
	m_would_skip.push_back({build, std::chrono::steady_clock::now(), refusal});
}

// The name a refusal is counted under when the policy refused without naming its rule.
static constexpr const char* UNNAMED_REFUSAL = "unnamed";

// A deferred draw the rules allowed is one a skip would have taken. One they refused, asked only with
// --async-pipelines, is counted under the rule, so the report says why deferral was the only way.
const char* PipelineCache::NoteDeferredSkipVerdict(const std::shared_ptr<GraphicsBuild>& build,
                                                   const PipelineSkipPolicy* policy, bool may_skip) {
	if (may_skip) {
		NoteDeferredWouldSkip(build);
		return Common::FirstRunReport::HOLE_VERDICT_ALLOWED.data();
	}
	if (Config::AsyncPipelinesEnabled() && policy != nullptr) {
		const char* refusal = policy->Refusal() != nullptr ? policy->Refusal() : UNNAMED_REFUSAL;
		NoteDeferredWouldSkip(build, refusal);
		return refusal;
	}
	return Common::FirstRunReport::HOLE_VERDICT_NOT_ASKED.data();
}

// The policy's refusing targets, only where NoteDeferredSkipVerdict takes the policy's own refusal as
// the verdict, so a hole never carries targets that explain another verdict.
static std::shared_ptr<const Common::FirstRunReport::SkipTargets> RefusedTargetsOf(const PipelineSkipPolicy* policy,
                                                                                   bool may_skip) {
	if (may_skip || !Config::AsyncPipelinesEnabled() || policy == nullptr) {
		return nullptr;
	}
	return policy->RefusedTargets();
}

void PipelineCache::ReportDeferredWouldSkip() {
	for (const auto& noted: m_would_skip) {
		// The build reports its readiness as a time since it was requested, which a draw met after
		// the request did not wait for. A build done before the draw was deferred waited for nothing.
		const auto ready_at = noted.build->requested +
		                      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
		                          std::chrono::duration<double, std::milli>(noted.build->ready_ms));
		const double waited_ms =
		    std::max(0.0, std::chrono::duration<double, std::milli>(ready_at - noted.deferred_at).count());
		if (noted.refusal != nullptr) {
			Common::FirstRunReport::AddDeferredRefusedSkip(noted.refusal, waited_ms);
		} else {
			Common::FirstRunReport::AddDeferredWouldSkip(waited_ms);
		}
	}
	m_would_skip.clear();
}

DrawFate PipelineCache::ChooseMissedDrawFate(const GraphicsPipelineKey& key, const Stages& stages,
                                             const PipelineSkipPolicy* policy,
                                             const DeferredDraw* deferred, bool& may_skip,
                                             const char*& blocker) {
	// The S2 rules are only asked when skips are on. A flag-only run must neither skip nor pay for
	// their history checks.
	may_skip = Config::AsyncPipelinesEnabled() && policy != nullptr && policy->MaySkip();
	const bool wants_defer = deferred != nullptr && Config::DeferredPipelineDrawsEnabled();
	blocker                = wants_defer ? DeferBlocker(key, stages, policy) : nullptr;
	// The counter says the draw could not be deferred, so a draw that a skip takes instead is in it.
	if (blocker != nullptr) {
		Common::FirstRunReport::AddDeferredIneligible(blocker);
	}
	return ChooseDrawFate({.defer_allowed = wants_defer && blocker == nullptr,
	                       .skip_allowed  = may_skip,
	                       .prefer_skip   = Config::AsyncPipelinesPreferSkip()});
}

PipelineCache::Pipeline* PipelineCache::AcquireBuiltPipeline(const GraphicsPipelineKey& key,
                                                             const Stages&              stages,
                                                             const DrawRenderState&     state,
                                                             const PipelineSkipPolicy*  policy,
                                                             DeferredDraw*              deferred,
                                                             MissedDraw&                missed) {
	// The one cost a pipeline that is already built pays for background builds.
	if (m_has_published.load(std::memory_order_relaxed)) {
		DrainPublishedBuilds();
	}
	if (const auto found = m_graphics_pipelines.find(key); found != m_graphics_pipelines.end()) {
		return &found->second;
	}
	auto pending = m_pending_graphics.find(key);
	if (pending != m_pending_graphics.end() &&
	    pending->second.build->completed.load(std::memory_order_acquire)) {
		return &AdoptBuild(pending->second.build);
	}
	// The wait of a draw that cannot be deferred starts here, after the drain above, which serves
	// other keys.
	if (missed.timed) {
		missed.start = std::chrono::steady_clock::now();
	}
	bool       may_skip = false;
	const auto fate     = ChooseMissedDrawFate(key, stages, policy, deferred, may_skip, missed.blocker);
	// A draw that could not be deferred is the one NoteIneligibleWait reports, so its wait gets that
	// cause. Any other draw that must wait for its pipeline needs it to render, so it is required.
	const auto wait_cause = missed.blocker != nullptr ? Common::GpuStall::HoldCause::IneligibleDraw
	                                                  : Common::GpuStall::HoldCause::RequiredGraphics;
	if (pending == m_pending_graphics.end()) {
		DumpPipelineInputs(state);
		if (fate == DrawFate::Block) {
			return &CreateGraphicsPipelineHere(m_graphics_pipelines[key], key, stages, wait_cause);
		}
		RequestBuild(key, stages);
		pending = m_pending_graphics.find(key);
	}
	if (fate == DrawFate::Defer) {
		deferred->active = true;
		deferred->key    = key;
		// A draw that S2 would have dropped is deferred instead, which costs it a wait, and one S2
		// refused had no other way.
		deferred->skip_verdict = NoteDeferredSkipVerdict(pending->second.build, policy, may_skip);
		deferred->skip_targets = RefusedTargetsOf(policy, may_skip);
		return nullptr;
	}
	if (fate == DrawFate::Skip) {
		pending->second.build->skipped_draws++;
		return nullptr;
	}
	return &FinishBuild(pending->second.build, pending->second.job, wait_cause);
}

PipelineCache::Pipeline PipelineCache::AcquireDeferredPipeline(const GraphicsPipelineKey& key,
                                                               double&                    waited_ms) {
	// No drain here. Taking published builds checkpoints the driver cache, which must not run under
	// the audio hold. AcquireDeferredPipelines drains before its scope, and a build published since
	// is taken below, when its key is acquired.
	// Adopting a build twice is fatal, so a key that is already built is only looked up. A drain
	// or a blocking draw of the same key may have taken the build while the hole waited.
	if (const auto found = m_graphics_pipelines.find(key); found != m_graphics_pipelines.end()) {
		return found->second;
	}
	const auto pending = m_pending_graphics.find(key);
	EXIT_IF(pending == m_pending_graphics.end());
	return FinishBuild(pending->second.build, pending->second.job,
	                   Common::GpuStall::HoldCause::DeferredDrawResolve, &waited_ms);
}

std::vector<PipelineCache::Pipeline>
PipelineCache::AcquireDeferredPipelines(std::span<const GraphicsPipelineKey> keys, bool hold_audio,
                                        DeferredAcquireStats& stats) {
	using Clock = std::chrono::steady_clock;
	std::vector<Pipeline> built;
	built.reserve(keys.size());
	std::unordered_set<GraphicsPipelineKey, GraphicsPipelineKeyHash> distinct;
	double                                                          waited_ms = 0.0;
	// Taking the builds that are already published may save the driver cache, so it happens before
	// the scope and counts toward save_ms rather than the audio hold.
	const auto drain_start = Clock::now();
	if (m_has_published.load(std::memory_order_relaxed)) {
		DrainPublishedBuilds();
	}
	const auto start = Clock::now();
	{
		// One scope for the whole wait, so guest audio holds once however many keys there are.
		// Recording the holes and saving the driver cache are outside it.
		std::optional<Common::GpuStall::Scope> stall;
		if (hold_audio) {
			stall.emplace(Common::GpuStall::HoldCause::DeferredDrawResolve);
		}
		// A resolve is one pause of the GPU thread, so it is one sample of its longest hitch.
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::Count,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		// Only the first-run report reads the critical key, so without it no clock is read per key.
		const bool find_critical = Common::FirstRunReport::Enabled();
		double     critical_ms   = 0.0;
		for (const auto& key: keys) {
			distinct.insert(key);
			if (find_critical) {
				AcquireTimedDeferredPipeline(key, waited_ms, built, critical_ms, stats);
			} else {
				built.push_back(AcquireDeferredPipeline(key, waited_ms));
			}
		}
	}
	const auto waited = Clock::now();
	stats.distinct_keys = static_cast<uint32_t>(distinct.size());
	stats.wait_ms       = std::chrono::duration<double, std::milli>(waited - start).count();
	// Only the time spent waiting for a running worker is added to the blocked time, since builds run
	// here record themselves as compiles.
	if (waited_ms > 0.0) {
		Common::FirstRunReport::AddWait(Common::FirstRunReport::Kind::Graphics, waited_ms);
	}
	ReportDeferredWouldSkip();
	CheckpointDriverCache();
	stats.save_ms = std::chrono::duration<double, std::milli>(Clock::now() - waited).count() +
	                std::chrono::duration<double, std::milli>(start - drain_start).count();
	return built;
}

void PipelineCache::AcquireTimedDeferredPipeline(const GraphicsPipelineKey& key, double& waited_ms,
                                                 std::vector<Pipeline>& built, double& critical_ms,
                                                 DeferredAcquireStats& stats) {
	using Clock = std::chrono::steady_clock;
	// Read before the acquire, which adopts the build and erases it from the pending map.
	const bool had_build = m_pending_graphics.contains(key);
	const auto before    = Clock::now();
	built.push_back(AcquireDeferredPipeline(key, waited_ms));
	const double took_ms = std::chrono::duration<double, std::milli>(Clock::now() - before).count();
	// Keys are acquired in hole order, so the key with the longest own acquire is the one the
	// resolve was stuck on.
	if (!had_build || took_ms <= critical_ms) {
		return;
	}
	critical_ms        = took_ms;
	stats.critical_key = key;
	stats.has_critical = true;
}

void PipelineCache::RequestBuild(const GraphicsPipelineKey& key, const Stages& stages) {
	auto build = std::make_shared<GraphicsBuild>(key, stages);
	auto job   = m_build_queue->Post([this, build] { RunBuild(build); });
	EXIT_IF(job == nullptr);
	m_pending_graphics.emplace(key, PendingBuild {std::move(build), std::move(job)});
	// Stored when asked for, so a pipeline still pending at exit is rebuilt by the next run.
	StorePipeline(key);
}

void PipelineCache::RunBuild(const std::shared_ptr<GraphicsBuild>& build) {
	// On a worker the compile records itself as worker time, and nothing here counts toward the
	// GPU thread's compile counters.
	CreatePipelineInternal(m_graphics, m_set_layouts, build->pipeline, build->key, build->stages,
	                       m_driver_cache, m_libraries.get());
	build->ready_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
	                                                            build->requested)
	                      .count();
	m_driver_cache_dirty.store(true, std::memory_order_relaxed);
	build->completed.store(true, std::memory_order_release);
	{
		std::scoped_lock lock(m_published_mutex);
		m_published.push_back(build);
	}
	m_has_published.store(true, std::memory_order_relaxed);
}

namespace {

// Makes sure a build job is done. Returns how long this thread waited for a worker that was running
// it, or nothing when the job was already done or this thread ran it, which counts as a compile.
std::optional<double> AcquireBuildJob(PipelineBuildQueue& queue, PipelineBuildQueue::Job& job) {
	const auto start = std::chrono::steady_clock::now();
	const auto how   = queue.Acquire(job);
	if (how == PipelineBuildQueue::Acquired::RanInline) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::PipelinesCreated);
	} else if (how == PipelineBuildQueue::Acquired::Waited) {
		return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	}
	return std::nullopt;
}

} // namespace

PipelineCache::Pipeline& PipelineCache::FinishBuild(const std::shared_ptr<GraphicsBuild>& build,
                                                    const PipelineBuildQueue::JobPtr&      job,
                                                    Common::GpuStall::HoldCause            wait_cause,
                                                    double*                                resolve_waited_ms) {
	// Keeps the build alive across the adoption below, which erases it from the pending map.
	const auto keep_alive = build;
	if (resolve_waited_ms != nullptr) {
		// The resolve holds the stall scope and does the timing and the save for all its keys.
		*resolve_waited_ms += AcquireBuildJob(*m_build_queue, *job).value_or(0.0);
		EXIT_IF(!keep_alive->completed.load(std::memory_order_acquire));
		return AdoptBuild(keep_alive);
	}
	{
		// Both ways of getting the pipeline stop the GPU thread, so guest audio may wait for them.
		std::optional<Common::GpuStall::Scope> stall;
		if (GuestGpu::IsGpuThread()) {
			stall.emplace(wait_cause);
		}
		// Running the compile here and waiting for a worker both stop the GPU thread, so both
		// count as pipeline time and toward its longest hitch.
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::Count,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		if (const auto waited_ms = AcquireBuildJob(*m_build_queue, *job)) {
			Common::FirstRunReport::AddWait(Common::FirstRunReport::Kind::Graphics, *waited_ms);
			PipelineCacheLog("Pipeline build: waited {:.1f} ms for a running background compile",
			                 *waited_ms);
		}
	}
	EXIT_IF(!keep_alive->completed.load(std::memory_order_acquire));
	CheckpointDriverCache();
	return AdoptBuild(keep_alive);
}

PipelineCache::Pipeline& PipelineCache::AdoptBuild(std::shared_ptr<GraphicsBuild> build) {
	EXIT_IF(build->adopted);
	build->adopted = true;
	Common::FirstRunReport::AddReady(Common::FirstRunReport::Kind::Graphics, build->key.stage_ids[0],
	                                 build->key.stage_ids[3], build->ready_ms,
	                                 build->skipped_draws);
	// Copy the key first, since erasing the pending entry may release the build.
	const auto key        = build->key;
	auto [iter, inserted] = m_graphics_pipelines.try_emplace(key, build->pipeline);
	EXIT_IF(!inserted);
	m_pending_graphics.erase(key);
	return iter->second;
}

void PipelineCache::DrainPublishedBuilds() {
	// Clears the flag before taking the list, so a build published after the take sets it again.
	m_has_published.store(false, std::memory_order_relaxed);
	std::vector<std::shared_ptr<GraphicsBuild>> finished;
	{
		std::scoped_lock lock(m_published_mutex);
		finished.swap(m_published);
	}
	for (const auto& build: finished) {
		// The GPU thread may have taken it already by waiting for it.
		if (!build->adopted) {
			AdoptBuild(build);
		}
	}
	if (!finished.empty()) {
		CheckpointDriverCache();
	}
}

void PipelineCache::StartWarmCompute(const ShaderModule& module, StoreGen::GenRow* row,
                                     PipelineBuildQueue::Lane lane) {
	// The module lives in the program cache, which outlasts the queue because StopBuilds runs
	// first. A refusal leaves the module to build on first use, as it would without the warm-up.
	// The row is in the generation state, which outlasts the queue for the same reason.
	auto build = [this, &module, row, lane](Pipeline& pipeline) {
		// The rows of worker compiles show how many warm builds were running when they began, and
		// on which lane. The report has its own lane type because common cannot see the queue's.
		// A job the GPU thread takes inline before a worker starts it still counts on its lane,
		// which is fine for a figure of contention.
		const Common::FirstRunReport::WarmBuildScope running(
		    lane == PipelineBuildQueue::Lane::Back ? Common::FirstRunReport::WarmLane::Back
		                                           : Common::FirstRunReport::WarmLane::Front);
		const auto build_start = std::chrono::steady_clock::now();
		CreatePipelineInternal(m_graphics, pipeline, module, m_driver_cache);
		if (row != nullptr) {
			const auto build_end = std::chrono::steady_clock::now();
			// The build time goes first, so a reader that sees the time of the finish sees both.
			row->build_ms.store(
			    std::chrono::duration<double, std::milli>(build_end - build_start).count(),
			    std::memory_order_release);
			row->ready_ns.store(
			    std::chrono::duration_cast<std::chrono::nanoseconds>(build_end.time_since_epoch())
			        .count(),
			    std::memory_order_release);
		}
		// Marked here because this build adds to the driver cache without a checkpoint of its own.
		m_driver_cache_dirty.store(true, std::memory_order_relaxed);
	};
	// Only the generator passes a row, so the row marks the module as generated for the stall cause.
	if (m_warm_compute->Start(module.id, std::move(build), lane) && row != nullptr) {
		m_generated_warm_ids.insert(module.id);
	}
}

namespace {

/// Records a dispatch's first use of a warm-up module for the report. The report lives in common,
/// which cannot see the queue, so it has its own names for how the build was handed over.
void RecordWarmTake(uint64_t id, PipelineBuildQueue::Acquired how, double stopped_ms, int64_t at_ns,
                    bool paused = false) {
	using Common::FirstRunReport::WarmTakeHow;
	using Acquired               = PipelineBuildQueue::Acquired;
	const WarmTakeHow report_how = how == Acquired::AlreadyDone ? WarmTakeHow::Done
	                               : how == Acquired::Waited    ? WarmTakeHow::Waited
	                                                            : WarmTakeHow::Inline;
	Common::FirstRunReport::AddWarmTake(
	    {.id = id, .how = report_how, .stopped_ms = stopped_ms, .at_ns = at_ns, .paused = paused});
}

} // namespace

Common::GpuStall::HoldCause WarmTakeCause(PipelineBuildQueue::Acquired how, bool paused,
                                          WarmOrigin origin) {
	using Common::GpuStall::HoldCause;
	const bool waited = how == PipelineBuildQueue::Acquired::Waited;
	// The prescan's set is never paused, so the pause says nothing about it. A build no worker had
	// started was still queued on the prescan's lane when the GPU thread ran it. PrescanInFlight is
	// kept for a request still in the helper, which is a different miss.
	if (origin == WarmOrigin::Prescan) {
		return waited ? HoldCause::PrescanWaited : HoldCause::PrescanQueued;
	}
	const bool generated = origin == WarmOrigin::Generated;
	if (waited) {
		return generated ? HoldCause::GeneratedWaited : HoldCause::WarmWaited;
	}
	// Generated and paused are both told apart, so neither wins. The cost of the pause is the two
	// paused causes added, and the generator's misses are the two generated inline causes added.
	if (generated) {
		return paused ? HoldCause::GeneratedInlinePaused : HoldCause::GeneratedInline;
	}
	return paused ? HoldCause::WarmInlinePaused : HoldCause::WarmInline;
}

// The stall state a prescan build reads at its start and end. Only lock-free reads, so a worker
// never takes the stall mutex the GPU thread and the audio threads share.
struct PipelineCache::StallSample {
	bool     hold_active  = false;
	uint64_t holds        = 0;
	uint64_t scopes       = 0;
	bool     scope_active = false;

	// The two pairs are read in opposite orders because their counts move at opposite ends. The hold
	// count goes up when a hold ends, so the hold flag is read first, and a hold that ends between
	// the reads is already in the count. The Scope count goes up when a Scope begins, so the Scope
	// count is read first, and a Scope that begins between the reads is still active at the flag.
	// Read the other way round, a Scope that began between them and ended before the build did
	// would show in neither sample. A test passes between_scope_reads to open a Scope just there.
	static StallSample Take(const std::function<void()>* between_scope_reads = nullptr) {
		StallSample sample;
		sample.hold_active = Common::GpuStall::HoldInProgress();
		sample.holds       = Common::GpuStall::HoldCount();
		sample.scopes      = Common::GpuStall::ScopeCount();
		if (between_scope_reads != nullptr && *between_scope_reads) {
			(*between_scope_reads)();
		}
		sample.scope_active = Common::GpuStall::AnyScopeActive();
		return sample;
	}
};

// The prescan's builds never wait on the pause and never take a generator row. A build that runs on
// the GPU thread is the stop itself and not work beside it, so only a worker's build is judged, and
// only a worker's build counts toward the prescan_running underrun tag. That unit opens first, so
// the test hook already sees it.
void PipelineCache::StartPrescanCompute(const ShaderModule& module) {
	auto build = [this, &module](Pipeline& pipeline) {
		const bool on_worker = !GuestGpu::IsGpuThread();
		std::optional<Common::FirstRunReport::PrescanRunningScope> running;
		if (on_worker) {
			running.emplace();
		}
		const auto before = StallSample::Take(&m_prescan_sample_hook);
		const auto start  = std::chrono::steady_clock::now();
		if (m_prescan_build_hook) {
			m_prescan_build_hook();
		}
		CreatePipelineInternal(m_graphics, pipeline, module, m_driver_cache);
		const auto elapsed = std::chrono::steady_clock::now() - start;
		// Marked here because this build adds to the driver cache without a checkpoint of its own.
		m_driver_cache_dirty.store(true, std::memory_order_relaxed);
		CountPrescanBuild(on_worker, before, elapsed);
	};
	(void)m_prescan_compute->Start(module.id, std::move(build), PipelineBuildQueue::Lane::Front);
}

// A worker's build overlapped a stop when a Scope was active at its start or end or one began while
// it ran. It overlapped a hold of guest audio when a hold was in progress at its start or end or
// one ended while it ran. The hold is judged from HoldInProgress and HoldCount and never from
// IsStalled, which takes the stall mutex.
void PipelineCache::CountPrescanBuild(bool on_worker, const StallSample& before,
                                      std::chrono::steady_clock::duration elapsed) {
	const auto after      = StallSample::Take();
	const auto elapsed_us = static_cast<uint64_t>(
	    std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
	m_prescan_builds.builds.fetch_add(1, std::memory_order_relaxed);
	if (!on_worker) {
		return;
	}
	// Worker-only, since these are what in_scope, in_hold and their times are measured against.
	m_prescan_builds.worker_builds.fetch_add(1, std::memory_order_relaxed);
	m_prescan_builds.total_us.fetch_add(elapsed_us, std::memory_order_relaxed);
	if (before.scope_active || after.scope_active || after.scopes != before.scopes) {
		m_prescan_builds.in_scope.fetch_add(1, std::memory_order_relaxed);
		m_prescan_builds.in_scope_us.fetch_add(elapsed_us, std::memory_order_relaxed);
	}
	if (before.hold_active || after.hold_active || after.holds != before.holds) {
		m_prescan_builds.in_hold.fetch_add(1, std::memory_order_relaxed);
		m_prescan_builds.in_hold_us.fetch_add(elapsed_us, std::memory_order_relaxed);
	}
}

// in_scope_ms and in_hold_ms are the whole time of each build that overlapped a stop or a hold, not
// only the overlap, so they are upper bounds on the contention the stop met.
PrescanBuildStats PipelineCache::PrescanBuilds() const {
	constexpr double MS_PER_US = 1e-3;
	const auto       to_ms     = [](const std::atomic<uint64_t>& micros) {
		return static_cast<double>(micros.load(std::memory_order_relaxed)) * MS_PER_US;
	};
	return {.builds        = m_prescan_builds.builds.load(std::memory_order_relaxed),
	        .worker_builds = m_prescan_builds.worker_builds.load(std::memory_order_relaxed),
	        .in_scope      = m_prescan_builds.in_scope.load(std::memory_order_relaxed),
	        .in_hold       = m_prescan_builds.in_hold.load(std::memory_order_relaxed),
	        .total_ms      = to_ms(m_prescan_builds.total_us),
	        .in_scope_ms   = to_ms(m_prescan_builds.in_scope_us),
	        .in_hold_ms    = to_ms(m_prescan_builds.in_hold_us)};
}

WarmBuildSet<uint64_t, PipelineCache::Pipeline>* PipelineCache::WarmSetPending(uint64_t id) const {
	if (m_prescan_compute != nullptr && m_prescan_compute->IsPending(id)) {
		return m_prescan_compute.get();
	}
	if (m_warm_compute != nullptr && m_warm_compute->IsPending(id)) {
		return m_warm_compute.get();
	}
	return nullptr;
}

PipelineCache::Pipeline* PipelineCache::TakeWarmCompute(const ShaderComputeInputInfo& input_info) {
	const auto id  = input_info.stage.module->id;
	auto*      set = WarmSetPending(id);
	if (set == nullptr) {
		return nullptr;
	}
	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}
	// A finished build is just handed over. Once the warm-up is over this is the usual case, and it
	// needs neither the stall nor a timed sample.
	std::optional<WarmBuildSet<uint64_t, Pipeline>::Taken> taken;
	if (set->IsDone(id)) {
		taken = set->Take(id);
		EXIT_IF(!taken);
		// The clock is read only for the report, so a run without it does exactly what it did.
		const int64_t at_ns = Common::FirstRunReport::Enabled() ? Common::FirstRunReport::NowNs()
		                                                        : Common::FirstRunReport::NO_TIME_NS;
		RecordWarmTake(id, taken->how, 0.0, at_ns);
	} else {
		taken = TakeUnfinishedWarmCompute(id, *set);
	}
	EXIT_IF(!taken);
	// The preparer backs off a program whose modules go unused, so it hears of each one taken. Its
	// report ring has one producer, the GPU thread, so a take from any other thread is not told.
	if (set == m_prescan_compute.get() && m_prescan != nullptr && GuestGpu::IsGpuThread()) {
		m_prescan->NoteTaken(id);
	}
	auto [iter, inserted] = m_compute_pipelines.try_emplace(id, std::move(taken->result));
	EXIT_IF(!inserted);
	CheckpointDriverCache();
	return &iter->second;
}

std::optional<WarmBuildSet<uint64_t, PipelineCache::Pipeline>::Taken>
PipelineCache::TakeUnfinishedWarmCompute(uint64_t id, WarmBuildSet<uint64_t, Pipeline>& set) {
	// Looked up before the Scope, so no part of it counts toward a hold. The id set belongs to the
	// GPU thread, so another thread, which makes no Scope, does not read it. A prescan module is told
	// by its set, and its id is never in m_generated_warm_ids.
	const bool on_gpu_thread = GuestGpu::IsGpuThread();
	const auto origin        = &set == m_prescan_compute.get() ? WarmOrigin::Prescan
	                           : on_gpu_thread && m_generated_warm_ids.contains(id)
	                               ? WarmOrigin::Generated
	                               : WarmOrigin::Store;
	// Running the compile here and waiting for a worker both stop the GPU thread, so guest audio
	// may wait for them and both count toward its longest hitch.
	std::optional<Common::GpuStall::Scope> stall;
	if (on_gpu_thread) {
		// What the dispatch waits for is known only once Take says whether it ran the build or waited
		// for a worker, so the cause is corrected below, before the Scope ends.
		stall.emplace(Common::GpuStall::HoldCause::WarmInline);
	}
	Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::Count,
	                                         Common::DebugCounters::Counter::PipelineCreateNs,
	                                         Common::DebugCounters::Peak::PipelineCreateNs);
	const auto start  = std::chrono::steady_clock::now();
	const bool paused = set.Paused();
	auto       taken  = set.Take(id);
	EXIT_IF(!taken);
	if (stall) {
		stall->Reclassify(WarmTakeCause(taken->how, paused, origin));
	}
	const auto stopped_ms = std::chrono::duration<double, std::milli>(
	                            std::chrono::steady_clock::now() - start)
	                            .count();
	NoteWarmComputeStop(taken->how, stopped_ms, paused);
	// The row's time is the start of this stop, on the steady clock that NowNs reads, so it and
	// stopped_ms come from the same reading.
	const int64_t start_ns =
	    std::chrono::duration_cast<std::chrono::nanoseconds>(start.time_since_epoch()).count();
	RecordWarmTake(id, taken->how, stopped_ms, start_ns, paused);
	return taken;
}

void PipelineCache::NoteWarmComputeStop(PipelineBuildQueue::Acquired how, double stopped_ms,
                                        bool queue_held) {
	if (how == PipelineBuildQueue::Acquired::AlreadyDone) {
		return;
	}
	// A stop as long as the audio hold threshold holds guest audio, if audio was waiting, so those
	// are counted apart. Every build taken here is counted, held or not, so the arms of a run have
	// figures to compare.
	const bool over = stopped_ms >= std::chrono::duration<double, std::milli>(
	                                    Common::GpuStall::GPU_STALL_AUDIO_HOLD_THRESHOLD)
	                                    .count();
	const bool waited = how == PipelineBuildQueue::Acquired::Waited;
	Common::FirstRunReport::AddWarmCompute(waited, stopped_ms, over);
	if (waited) {
		Common::FirstRunReport::AddWait(Common::FirstRunReport::Kind::Compute, stopped_ms);
		// Lets a hold of guest audio be matched to the wait, and shows what a worker at low
		// priority costs the GPU thread.
		PipelineCacheLog("Pipeline build: waited {:.1f} ms for a running warm-up compute build",
		                 stopped_ms);
		return;
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::PipelinesCreated);
	if (queue_held) {
		// No worker had started this build while the queue was held, so the GPU thread ran it. The
		// pause may not be the only reason, which is why this is an upper bound.
		m_warm_inline_during_pause++;
		Common::FirstRunReport::AddDeferredWarmInline(over);
		PipelineCacheLog("Pipeline build: a warm-up compute build ran inline while the queue was held, "
		                 "{:.1f} ms{} ({} so far)",
		                 stopped_ms, over ? ", over the audio hold threshold" : "",
		                 m_warm_inline_during_pause);
	}
}

// The prescan's set is left running on purpose. The audio results the prescan was approved on were
// measured with its lane exempt, and its modules are ones a scanned submission is about to use, so
// holding them would only turn them into inline compiles on the GPU thread. Its builds that overlap
// a stop of the GPU thread are counted in PrescanBuilds instead.
void PipelineCache::PauseWarmBuilds() {
	if (m_warm_compute == nullptr || m_warm_paused || !Config::BackgroundWarmUpEnabled() ||
	    !Config::DeferredPipelineDrawsEnabled() || !Config::PauseWarmUpForDrawsEnabled()) {
		return;
	}
	m_warm_paused = true;
	m_warm_compute->Pause(PipelineBuildQueue::Clock::now() + WARM_UP_PAUSE_LIMIT);
}

void PipelineCache::ResumeWarmBuilds() {
	if (!m_warm_paused) {
		return;
	}
	m_warm_paused = false;
	if (m_warm_compute != nullptr) {
		m_warm_compute->Resume();
	}
}

void PipelineCache::StopWarmSet(std::unique_ptr<WarmBuildSet<uint64_t, Pipeline>>& set) {
	if (set == nullptr) {
		return;
	}
	auto stopped = set->Stop();
	if (stopped.waited_for != 0) {
		PipelineCacheLog("Pipeline build: waiting for {} running warm-up compiles", stopped.waited_for);
	}
	// Nothing runs now. A finished build nobody took still owns its Vulkan objects, and a build
	// that never ran owns none. Each finished one marked the driver cache when it was built, so the
	// save that follows keeps its work.
	for (const auto& pipeline: stopped.unadopted) {
		DestroyPipelineObjects(m_graphics.device, pipeline);
	}
	set.reset();
}

void PipelineCache::StopWarmBuilds() {
	// Stopping ends a pause, since the workers wake for the stop.
	m_warm_paused = false;
	StopWarmSet(m_prescan_compute);
	StopWarmSet(m_warm_compute);
}

void PipelineCache::StopBuilds() {
	// The generator goes first: it posts modules whose builds belong to the warm queue, and the
	// child process it owns must be gone before the queue is.
	StopStoreGeneration();
	StopWarmBuilds();
	if (m_build_queue == nullptr) {
		return;
	}
	if (const auto running = m_build_queue->Stop(); running != 0) {
		PipelineCacheLog("Pipeline build: waiting for {} running background compiles", running);
	}
	// Nothing runs now. A finished build nobody took still owns its Vulkan objects, and a build
	// that never ran owns none. A pipeline linked from libraries is destroyed on its own here, which
	// is fine, since the libraries stay in the cache until the destructor and may serve later builds.
	for (const auto& pending: m_pending_graphics | std::views::values) {
		const auto& pipeline = pending.build->pipeline;
		DestroyPipelineObjects(m_graphics.device, pipeline);
	}
	m_pending_graphics.clear();
	{
		std::scoped_lock lock(m_published_mutex);
		m_published.clear();
	}
	m_has_published.store(false, std::memory_order_relaxed);
	m_build_queue.reset();
}

PipelineCache::Pipeline& PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	const auto& module = *input_info.stage.module;

	// A warm-up build for this module must be taken before a second one is made.
	if (auto* warm = TakeWarmCompute(input_info)) {
		return *warm;
	}

	auto [iter, inserted] = m_compute_pipelines.try_emplace(module.id);
	auto& pipeline        = iter->second;
	if (!inserted) {
		return pipeline;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	{
		// A compile on the GPU command thread stops all guest rendering, so guest audio may wait
		// for it. One on any other thread leaves the game running and must not hold audio. The cause
		// is picked before the Scope opens, from lock-free reads only, so the GPU thread never waits
		// on the prescan to name it.
		std::optional<Common::GpuStall::Scope> stall;
		if (GuestGpu::IsGpuThread()) {
			stall.emplace(InlineComputeCause(module.id));
			if (m_inline_scope_hook) {
				m_inline_scope_hook();
			}
		}
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, pipeline, module, m_driver_cache);
	}
	CheckpointDriverCache();

	return pipeline;
}
} // namespace Libs::Graphics
