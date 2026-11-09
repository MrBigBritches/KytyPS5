#include "graphics/shader/shaderStoreRecord.h"

#include "common/assert.h"
#include "graphics/shader/recompiler/ir/Serialize.h"

#include <cstring>
#include <xxhash.h>
#include <zstd.h>

namespace Libs::Graphics {

namespace {

using ShaderRecompiler::IR::IoAll;
using ShaderRecompiler::IR::Reader;
using ShaderRecompiler::IR::Writer;

// A span is written like the vector it views, a count and then the elements, so a store written
// from spans reads back into vectors.
template <class T>
void IoSpan(Writer& writer, std::span<const T> elements) {
	IoAll(writer, static_cast<uint32_t>(elements.size()));
	for (const auto& element: elements) {
		IoAll(writer, element);
	}
}

// Elements that are plain scalars and enums are copied in one piece, as Io does for a vector.
template <class T>
requires(std::is_arithmetic_v<T> || std::is_enum_v<T>)
void IoSpan(Writer& writer, std::span<const T> elements) {
	IoAll(writer, static_cast<uint32_t>(elements.size()));
	writer.Raw(elements.data(), elements.size_bytes());
}

} // namespace

std::vector<uint8_t> FrameStoreRecord(std::span<const uint8_t> record) {
	std::vector<uint8_t> bytes(sizeof(StoreFrame) + ZSTD_compressBound(record.size()));
	const auto           size =
	    ZSTD_compress(bytes.data() + sizeof(StoreFrame), bytes.size() - sizeof(StoreFrame),
	                  record.data(), record.size(), STORE_COMPRESSION_LEVEL);
	EXIT_IF(ZSTD_isError(size));
	const StoreFrame frame {.compressed_size = static_cast<uint32_t>(size),
	                        .raw_size        = static_cast<uint32_t>(record.size()),
	                        .checksum = XXH3_64bits(bytes.data() + sizeof(StoreFrame), size)};
	std::memcpy(bytes.data(), &frame, sizeof(frame));
	bytes.resize(sizeof(StoreFrame) + size);
	return bytes;
}

std::optional<UnframedRecord> UnframeStoreRecord(std::span<const uint8_t> records) {
	StoreFrame frame;
	if (records.size() < sizeof(frame)) {
		return std::nullopt;
	}
	std::memcpy(&frame, records.data(), sizeof(frame));
	if (records.size() - sizeof(frame) < frame.compressed_size) {
		return std::nullopt;
	}
	const auto compressed = records.subspan(sizeof(frame), frame.compressed_size);
	// The checksum does not cover the frame, so raw_size is untrusted. Comparing it with the size
	// zstd recorded also bounds the allocation below.
	if (XXH3_64bits(compressed.data(), compressed.size()) != frame.checksum ||
	    ZSTD_getFrameContentSize(compressed.data(), compressed.size()) != frame.raw_size) {
		return std::nullopt;
	}
	UnframedRecord result {.raw = std::vector<uint8_t>(frame.raw_size)};
	if (ZSTD_decompress(result.raw.data(), result.raw.size(), compressed.data(),
	                    compressed.size()) != result.raw.size()) {
		return std::nullopt;
	}
	result.frame_bytes = sizeof(frame) + compressed.size();
	return result;
}

std::vector<uint8_t> EncodeModuleRecord(const ModuleRecordInputs& inputs) {
	Writer record;
	IoAll(record, StoreRecord::Module);
	IoAll(record, inputs.stage, inputs.hash, inputs.user_data_count, inputs.static_state);
	IoAll(record, ShaderRecompiler::IR::SaveResourcePlan(inputs.plan));
	IoAll(record, inputs.id, inputs.info, inputs.specialization);
	IoSpan(record, inputs.vertex_formats);
	IoSpan(record, inputs.rect_parameters);
	IoAll(record, inputs.mesh_task);
	IoSpan(record, inputs.spirv);
	return std::move(record.bytes);
}

void DecodeModuleRecord(Reader& reader, DecodedModuleRecord& record) {
	auto& [stage, hash, user_data_count, static_state, plan, id, info, specialization,
	       vertex_formats, rect_parameters, mesh_task, spirv] = record;
	IoAll(reader, stage, hash, user_data_count, static_state, plan, id, info, specialization,
	      vertex_formats, rect_parameters, mesh_task, spirv);
}

} // namespace Libs::Graphics
