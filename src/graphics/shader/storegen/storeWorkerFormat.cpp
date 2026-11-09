#include "graphics/shader/storegen/storeWorkerFormat.h"

#include "common/file.h"
#include "graphics/shader/recompiler/ir/Serialize.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <xxhash.h>

namespace Libs::Graphics::StoreGen {

namespace IR = ShaderRecompiler::IR;

std::filesystem::path PathFromUtf8(const std::string& text) {
	return std::filesystem::path(std::u8string(text.begin(), text.end()));
}

std::string Utf8FromPath(const std::filesystem::path& path) {
	const auto text = path.u8string();
	return {text.begin(), text.end()};
}

std::vector<uint8_t> EncodeWorkerRequest(const WorkerRequest& request) {
	IR::Writer writer;
	auto       magic   = WORKER_REQUEST_MAGIC;
	auto       version = WORKER_REQUEST_VERSION;
	// Copies, because Io takes its values by reference and the request is const.
	auto fields = request;
	IR::IoAll(writer, magic, version, fields.store_header, fields.host_subgroup_size,
	          fields.eboot_path, fields.output_path, fields.progress_path, fields.generated_path,
	          fields.kept, fields.start_rank, fields.skip_hashes, fields.failed_hashes,
	          fields.parent_pid, fields.mode, fields.generator_version, fields.expected_order);
	return std::move(writer.bytes);
}

std::optional<WorkerRequest> DecodeWorkerRequest(std::span<const uint8_t> bytes) {
	IR::Reader    reader {bytes};
	uint32_t      magic   = 0;
	uint32_t      version = 0;
	WorkerRequest request;
	IR::IoAll(reader, magic, version);
	if (!reader.ok || magic != WORKER_REQUEST_MAGIC || version != WORKER_REQUEST_VERSION) {
		return std::nullopt;
	}
	IR::IoAll(reader, request.store_header, request.host_subgroup_size, request.eboot_path,
	          request.output_path, request.progress_path, request.generated_path, request.kept,
	          request.start_rank, request.skip_hashes, request.failed_hashes, request.parent_pid,
	          request.mode, request.generator_version, request.expected_order);
	// A mode byte is checked after the read, since an enum read takes any value.
	const bool known_mode =
	    request.mode == WorkerMode::Generate || request.mode == WorkerMode::Verify;
	if (!reader.ok || !reader.bytes.empty() || !known_mode) {
		return std::nullopt;
	}
	return request;
}

bool WriteWorkerRequestFile(const std::filesystem::path& path, const WorkerRequest& request) {
	const auto bytes = EncodeWorkerRequest(request);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file;
	if (!file.Create(path)) {
		return false;
	}
	uint32_t written = 0;
	file.Write(bytes.data(), static_cast<uint32_t>(bytes.size()), &written);
	return written == bytes.size() && file.Flush();
}

std::optional<WorkerRequest> ReadWorkerRequestFile(const std::filesystem::path& path) {
	Common::File file;
	if (!file.Open(path, Common::File::Mode::Read)) {
		return std::nullopt;
	}
	const auto bytes = file.ReadWholeBuffer();
	return DecodeWorkerRequest({reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size()});
}

EbootIdentity ComputeEbootIdentity(uint64_t eboot_bytes, std::span<const ScannedProgram> programs) {
	std::vector<uint64_t> hashes;
	hashes.reserve(programs.size());
	for (const auto& program: programs) {
		hashes.push_back(program.code_hash);
	}
	std::ranges::sort(hashes);
	return {eboot_bytes, XXH3_64bits(hashes.data(), hashes.size() * sizeof(uint64_t))};
}

std::vector<uint8_t> EncodeGeneratedHeader(std::span<const uint8_t> store_header,
                                           uint32_t                 generator_version,
                                           const EbootIdentity&     identity) {
	IR::Writer writer;
	writer.Raw(store_header.data(), store_header.size());
	// Copies, because Io takes its values by reference.
	auto size = identity.size;
	auto hash = identity.program_hash;
	IR::IoAll(writer, generator_version, size, hash);
	return std::move(writer.bytes);
}

std::optional<GeneratedHeader> ParseGeneratedHeader(std::span<const uint8_t> file,
                                                    size_t                   store_header_bytes) {
	if (file.size() < store_header_bytes ||
	    file.size() - store_header_bytes < GENERATED_HEADER_TAIL_BYTES) {
		return std::nullopt;
	}
	GeneratedHeader header;
	header.store_header.assign(file.begin(),
	                           file.begin() + static_cast<ptrdiff_t>(store_header_bytes));
	IR::Reader reader {file.subspan(store_header_bytes, GENERATED_HEADER_TAIL_BYTES)};
	IR::IoAll(reader, header.generator_version, header.identity.size, header.identity.program_hash);
	header.bytes = store_header_bytes + GENERATED_HEADER_TAIL_BYTES;
	return reader.ok ? std::optional<GeneratedHeader>(std::move(header)) : std::nullopt;
}

} // namespace Libs::Graphics::StoreGen
