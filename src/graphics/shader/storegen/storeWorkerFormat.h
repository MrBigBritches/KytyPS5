#ifndef GRAPHICS_SHADER_STOREGEN_STORE_WORKER_FORMAT_H
#define GRAPHICS_SHADER_STOREGEN_STORE_WORKER_FORMAT_H

#include "graphics/shader/storegen/ebootShaderScan.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

// The bytes the shader store worker and its supervisor exchange through files, which are the
// request that starts a worker and the header of the file it generates. Neither needs a graphics
// device or Config.
namespace Libs::Graphics::StoreGen {

/// What a worker does with its request.
enum class WorkerMode : uint8_t {
	Generate, ///< Generate the records into the output file.
	Verify,   ///< Exit at once with WORKER_EXIT_UP_TO_DATE when the finished file is current.
	/// Serve StreamRequest frames from the pipe the request names, as streamWorker.h describes.
	Stream,
};

/// The version of the worker's rules, which are the prediction, the candidate order and the record
/// encoding it drives. It is part of the generated file's header, so changing any of them means
/// raising it, and an older file is generated again.
constexpr uint32_t SHADER_STORE_GENERATOR_VERSION = 2;

/// Starts every request file, and holds "WSRQ".
constexpr uint32_t WORKER_REQUEST_MAGIC = 0x51525357;
/// The layout of the request file. A reader refuses any other value. Version 3 added the stream
/// pipe name.
constexpr uint32_t WORKER_REQUEST_VERSION = 3;

/// Everything a worker needs, so it reads no device and no Config. The supervisor writes it to a
/// file and passes the file's path on the worker's command line.
struct WorkerRequest {
	/// The bytes of the pipeline cache's StoreHeader, which hold the build stamp, the jump table
	/// mode, the SPIR-V profile and the shared memory limit. The worker decodes the values it
	/// translates with from them, and writes them first in the generated file.
	std::vector<uint8_t> store_header;
	/// The host's subgroup size, which the compute static key depends on.
	uint32_t host_subgroup_size = 0;
	/// The eboot, as a path Common::File reads. Inside a .zar it is `<host.zar>!/eboot.bin`.
	std::string eboot_path;
	/// The file the records are appended to, which is the supervisor's temporary file.
	std::string output_path;
	/// The log the worker appends its progress lines to.
	std::string progress_path;
	/// The finished generated file, read by Verify mode only.
	std::string generated_path;
	/// How many bytes of the output file are good. Zero makes a new file with its header, and more
	/// keeps that many bytes and appends after them, so a restarted worker continues the file.
	uint64_t kept = 0;
	/// The first candidate rank to generate, with the ranks as the worker's ordering gives them.
	uint64_t start_rank = 0;
	/// Programs left out before their candidates are predicted, for a program whose prediction
	/// itself crashed. No rank is ever given to such a program, so leaving it out moves nothing
	/// that a supervisor has already seen.
	std::vector<uint64_t> skip_hashes;
	/// Programs whose candidates keep their ranks and are not translated, for a program that
	/// crashed or hung while one of its candidates was built. The ranks of the others stay put, so
	/// a restart can continue at the next rank. A supervisor also adds the hash of every `drop`
	/// line of the progress log here when it restarts, since the restart would drop that program
	/// again at the same candidate and would build it up to there for nothing.
	std::vector<uint64_t> failed_hashes;
	/// The `order` hash of the run that wrote the kept bytes, from WorkerResult::order_hash. When
	/// kept is above zero and this is not zero, a worker whose ordering hashes differently exits
	/// with WORKER_EXIT_ORDER_MISMATCH. Skip hashes are only safe to change before any `start` line
	/// was seen, since they change the ordering. Failed hashes never do.
	uint64_t expected_order = 0;
	/// The supervisor's process id, for the worker's watchdog.
	uint32_t   parent_pid = 0;
	WorkerMode mode       = WorkerMode::Generate;
	/// The SHADER_STORE_GENERATOR_VERSION of the supervisor, which the worker writes into the file.
	uint32_t generator_version = SHADER_STORE_GENERATOR_VERSION;
	/// Stream mode only, the name of the pipe to connect to, from StreamPipeListener::Name. It
	/// travels in the request rather than on the command line so the request stays the one thing a
	/// worker reads, next to the parent_pid that the pipe's server must have.
	std::string stream_pipe;
};

/// A path from the UTF-8 text a request holds.
[[nodiscard]] std::filesystem::path PathFromUtf8(const std::string& text);

/// The UTF-8 text of a path, as a request holds it.
[[nodiscard]] std::string Utf8FromPath(const std::filesystem::path& path);

/// The bytes of a request, as the request file holds them.
[[nodiscard]] std::vector<uint8_t> EncodeWorkerRequest(const WorkerRequest& request);

/// Reads a request.
/// @return nothing when the bytes are cut short, have trailing bytes, hold another magic or
///         version, hold a mode that does not exist, or are in Stream mode with no pipe name.
[[nodiscard]] std::optional<WorkerRequest> DecodeWorkerRequest(std::span<const uint8_t> bytes);

/// Writes a request file, replacing the file if it exists.
/// @return false when the file could not be written whole.
[[nodiscard]] bool WriteWorkerRequestFile(const std::filesystem::path& path,
                                          const WorkerRequest&         request);

/// Reads and decodes a request file.
/// @return nothing when the file is missing or DecodeWorkerRequest refuses it.
[[nodiscard]] std::optional<WorkerRequest> ReadWorkerRequestFile(const std::filesystem::path& path);

/// What a game's eboot is known by. A changed eboot gives another identity, so a generated file
/// made from the old one is not reused.
struct EbootIdentity {
	/// The size of the eboot file in bytes.
	uint64_t size = 0;
	/// XXH3 of the compute programs' code hashes, sorted ascending, as 64-bit little endian words.
	uint64_t program_hash = 0;

	bool operator==(const EbootIdentity&) const = default;
};

/// The identity of an eboot from its size and its scanned programs. The order of the programs
/// does not matter, since the hashes are sorted first.
[[nodiscard]] EbootIdentity ComputeEbootIdentity(uint64_t                        eboot_bytes,
                                                 std::span<const ScannedProgram> programs);

/// The bytes after the StoreHeader in a generated file, which are the generator version and then
/// the eboot identity's size and program hash.
constexpr size_t GENERATED_HEADER_TAIL_BYTES =
    sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint64_t);

/// The header of a generated file, which is the StoreHeader, the generator version and the eboot
/// identity in that order. The records follow it. The StoreHeader comes first so the supervisor
/// checks it the way it checks the main store's.
[[nodiscard]] std::vector<uint8_t> EncodeGeneratedHeader(std::span<const uint8_t> store_header,
                                                         uint32_t                 generator_version,
                                                         const EbootIdentity&     identity);

/// What a generated file's header says.
struct GeneratedHeader {
	std::vector<uint8_t> store_header;
	uint32_t             generator_version = 0;
	EbootIdentity        identity;
	/// The size of the whole header, which is where the first record starts.
	size_t bytes = 0;
};

/// Reads the header of a generated file.
/// @param file               the file, or its first bytes.
/// @param store_header_bytes the size of the StoreHeader the file was written with.
/// @return nothing when the file is shorter than its header.
[[nodiscard]] std::optional<GeneratedHeader> ParseGeneratedHeader(std::span<const uint8_t> file,
                                                                  size_t store_header_bytes);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_STORE_WORKER_FORMAT_H
