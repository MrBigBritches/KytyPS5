#ifndef GRAPHICS_SHADER_SHADER_STORE_RECORD_H
#define GRAPHICS_SHADER_SHADER_STORE_RECORD_H

#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/rectListShader.h"
#include "graphics/shader/shader.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

// The byte format of the shader store, apart from its header. This code reads no device, no
// PipelineCache and no Config, so the pipeline cache and an offline generator share it and write
// the same bytes by construction. Anything changed here changes the store format, so the golden
// bytes in ShaderStoreRecordTests must keep passing.
namespace Libs::Graphics {

namespace ShaderRecompiler::IR {
struct Reader;
} // namespace ShaderRecompiler::IR

/// What a store record holds, which is its first byte.
enum class StoreRecord : uint8_t { Module, GraphicsPipeline, TessellationLayout };

/// Precedes each record's zstd frame in the store.
struct StoreFrame {
	uint32_t compressed_size = 0;
	uint32_t raw_size        = 0;
	/// XXH3 of the compressed bytes.
	uint64_t checksum = 0;
};

/// The zstd level of every record. Changing it changes the stored bytes but not their meaning.
constexpr int STORE_COMPRESSION_LEVEL = 3;

/// Compresses a record and puts a StoreFrame in front of it, as it is appended to the store.
/// @param record the uncompressed record, from EncodeModuleRecord for example.
[[nodiscard]] std::vector<uint8_t> FrameStoreRecord(std::span<const uint8_t> record);

/// A record taken out of its frame.
struct UnframedRecord {
	std::vector<uint8_t> raw;
	/// The size of the frame in the store, which is where the next one starts.
	size_t frame_bytes = 0;
};

/// Reads the first frame of `records`. It returns nothing when the frame is incomplete, when its
/// checksum or sizes are wrong, or when it does not decompress, so a damaged store ends there.
/// @param records the store after its header, or after the frames already read.
[[nodiscard]] std::optional<UnframedRecord> UnframeStoreRecord(std::span<const uint8_t> records);

/// Everything the store keeps of a compiled module, as views. The referenced values must outlive
/// the call.
struct ModuleRecordInputs {
	ShaderType stage           = ShaderType::Unknown;
	uint64_t   hash            = 0;
	uint32_t   user_data_count = 0;
	/// The program key's static state words.
	const std::vector<uint32_t>& static_state;
	/// The program's resource plan, saved with SaveResourcePlan.
	const ShaderRecompiler::IR::ResourcePlan& plan;
	/// The module id, from ShaderModuleId.
	uint64_t                                            id = 0;
	const ShaderRecompiler::IR::CompiledShaderInfo&     info;
	const ShaderRecompiler::IR::ResourceSpecialization& specialization;
	/// The raw values of the vk::Format of each vertex input attribute, empty outside VS and LS.
	std::span<const int32_t>           vertex_formats;
	std::span<const RectListParameter> rect_parameters;
	bool                               mesh_task = false;
	std::span<const uint32_t>          spirv;
};

/// Builds the bytes of a module record, which is what PipelineCache::StoreModule appends. The store
/// does not keep a module's function code, since only programs without GPU-selected calls are
/// stored.
[[nodiscard]] std::vector<uint8_t> EncodeModuleRecord(const ModuleRecordInputs& inputs);

/// A module record read back. The plan is kept as bytes, for LoadResourcePlan.
struct DecodedModuleRecord {
	ShaderType                                   stage           = ShaderType::Unknown;
	uint64_t                                     hash            = 0;
	uint32_t                                     user_data_count = 0;
	std::vector<uint32_t>                        static_state;
	std::vector<uint8_t>                         plan;
	uint64_t                                     id = 0;
	ShaderRecompiler::IR::CompiledShaderInfo     info;
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	std::vector<int32_t>                         vertex_formats;
	std::vector<RectListParameter>               rect_parameters;
	bool                                         mesh_task = false;
	std::vector<uint32_t>                        spirv;
};

/// Reads the fields of a module record that follow its StoreRecord byte. The caller checks that the
/// reader is ok and has nothing left.
void DecodeModuleRecord(ShaderRecompiler::IR::Reader& reader, DecodedModuleRecord& record);

/// Whether a vertex format type is an enum over int32_t, which is what the record keeps. Reading
/// and writing both rely on it, so a host format that changed its size would fail to compile.
template <class Format>
constexpr bool IS_RAW_INT_FORMAT =
    std::is_enum_v<Format> && std::is_same_v<std::underlying_type_t<Format>, int32_t>;

/// Fills a module of the pipeline cache from a decoded record. The handles, the function code and
/// anything else a stored module does not hold are left alone. The module's fields are all bound
/// here, so a field added to it fails to compile until it is stored or skipped on purpose.
/// @tparam Module a ShaderModule, or a type with the same nine fields in the same order.
template <class Module>
void FillModuleFromRecord(DecodedModuleRecord&& record, Module& module) {
	auto& [id, handle, gpl_handle, info, specialization, function_code, vertex_formats,
	       rect_parameters, mesh_task] = module;
	using Format = typename std::decay_t<decltype(vertex_formats)>::value_type;
	static_assert(IS_RAW_INT_FORMAT<Format>);
	id              = record.id;
	info            = std::move(record.info);
	specialization  = std::move(record.specialization);
	rect_parameters = std::move(record.rect_parameters);
	mesh_task       = record.mesh_task;
	vertex_formats.clear();
	for (const auto raw: record.vertex_formats) {
		vertex_formats.push_back(static_cast<Format>(raw));
	}
}

/// Fills a program key from a decoded record. The key's fields are all bound, like the module's.
/// @tparam Key a ProgramKey, or a type with the same four fields in the same order.
template <class Key>
void FillKeyFromRecord(DecodedModuleRecord& record, Key& key) {
	auto& [stage, hash, user_data_count, static_state] = key;
	stage                                              = record.stage;
	hash                                               = record.hash;
	user_data_count                                    = record.user_data_count;
	static_state                                       = std::move(record.static_state);
}

/// The inputs of EncodeModuleRecord for a program key and a module of the pipeline cache. Both are
/// bound field by field, so a field added to either fails to compile until it is stored or skipped.
/// @param formats receives the raw host formats and must outlive the returned inputs.
/// @param plan    the plan of the program the module belongs to.
/// @param spirv   the module's SPIR-V.
template <class Key, class Module>
ModuleRecordInputs RecordInputsOf(const Key& key, const ShaderRecompiler::IR::ResourcePlan& plan,
                                  const Module& module, std::vector<int32_t>& formats,
                                  std::span<const uint32_t> spirv) {
	const auto& [stage, hash, user_data_count, static_state] = key;
	const auto& [id, handle, gpl_handle, info, specialization, function_code, vertex_formats,
	             rect_parameters, mesh_task]                 = module;
	using Format = typename std::decay_t<decltype(vertex_formats)>::value_type;
	static_assert(IS_RAW_INT_FORMAT<Format>);
	formats.clear();
	for (const auto format: vertex_formats) {
		formats.push_back(static_cast<int32_t>(format));
	}
	return {.stage           = stage,
	        .hash            = hash,
	        .user_data_count = user_data_count,
	        .static_state    = static_state,
	        .plan            = plan,
	        .id              = id,
	        .info            = info,
	        .specialization  = specialization,
	        .vertex_formats  = formats,
	        .rect_parameters = rect_parameters,
	        .mesh_task       = mesh_task,
	        .spirv           = spirv};
}

} // namespace Libs::Graphics

#endif // GRAPHICS_SHADER_SHADER_STORE_RECORD_H
