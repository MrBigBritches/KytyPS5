#ifndef GRAPHICS_SHADER_STOREGEN_STREAM_WORKER_H
#define GRAPHICS_SHADER_STOREGEN_STREAM_WORKER_H

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/storegen/computeTranslation.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"
#include "graphics/shader/storegen/streamProtocol.h"
#include "graphics/shader/storegen/streamTranslationCache.h"

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

// The Stream mode of the shader store worker. The emulator's pre-scan sends it compute programs
// through a pipe as the game submits them, and it answers with each program's resource plan and
// then with the modules the parent asks for. Like the Generate mode it reads no device and no
// Config, so it runs in a child process where a crash or an EXIT in the recompiler cannot reach the
// emulator.
namespace Libs::Graphics::StoreGen {

/// How long the worker tries to connect to its pipe. The parent creates the listener before it
/// starts the child, so a long wait means the parent is gone or stuck.
constexpr std::chrono::milliseconds STREAM_CONNECT_TIMEOUT {5000};

/// How many requests may wait for a translation thread. Past it the reader stops reading until a
/// thread takes one, so a parent that sends faster than the worker answers fills the pipe and is
/// held back, instead of the worker's memory growing without bound.
constexpr uint32_t STREAM_MAX_QUEUED_JOBS = 64;

/// The largest program code a request may carry. Guest compute programs are tens of kilobytes, so
/// a megabyte is far past any real one while still bounding what a bad request can make the
/// worker copy and translate.
constexpr size_t MAX_STREAM_CODE_BYTES = size_t {1} << 20u;

/// The largest encoded specialization a Compile request may carry. A real one is a few dozen bytes
/// per resource, so this leaves room for hundreds of resources.
constexpr size_t MAX_STREAM_SPECIALIZATION_BYTES = size_t {64} * 1024u;

/// What a Stream worker has done, for the parent's report.
struct StreamWorkerStats {
	/// Requests the reader decoded and queued.
	uint64_t requests = 0;
	/// Plan and Compile requests answered, whatever the status.
	uint64_t plans    = 0;
	uint64_t compiles = 0;
	/// Answers by status, besides Ok.
	uint64_t has_calls = 0;
	uint64_t failed    = 0;
	/// Answers too large to send, which went out as Failed with no payload.
	uint64_t refused_replies = 0;
	/// Requests a stopping worker gave up, either dropped from the queue unanswered or answered
	/// Dropped.
	uint64_t dropped = 0;
	/// The process's priority as the system reports it, from ProcessPriorityName.
	std::string           priority;
	TranslationCacheStats cache;
};

/// What a Stream worker run ended with.
struct StreamWorkerResult {
	int               exit_code = 0;
	StreamWorkerStats stats;
};

/// The settings and test hooks of a Stream worker. A real worker passes none.
struct StreamWorkerOptions {
	uint32_t                  translation_threads = STREAM_TRANSLATION_THREADS;
	uint64_t                  cache_bytes         = STREAM_IR_CACHE_BYTES;
	std::chrono::milliseconds connect_timeout     = STREAM_CONNECT_TIMEOUT;
	uint32_t                  max_queued_jobs     = STREAM_MAX_QUEUED_JOBS;
	/// The largest encoded specialization a Compile may carry. A test lowers it to just under a
	/// real one, since a valid specialization as large as the default is hard to build.
	size_t max_specialization_bytes = MAX_STREAM_SPECIALIZATION_BYTES;
	/// Called with a request's seq after its `start` line is in the log and before its work.
	std::function<void(uint64_t seq)> before_work;
	/// Called on the translating thread just before a program is translated.
	std::function<void(const TranslationKey& key)> on_translate;
	/// Called when a thread has to wait for a program another thread holds, as
	/// TranslationCache's on_wait is.
	std::function<void()> on_cache_wait;
	/// Called with each answer before the worker checks that it can be sent. A test uses it to make
	/// an answer too large.
	std::function<void(StreamResponse& response)> before_send;
};

/// One Stream worker run. The request's mode is Stream.
///
/// Run connects to the request's stream_pipe, retrying while the pipe is busy for up to the
/// connect timeout, and serves nothing unless the pipe's server is the request's parent_pid. One
/// thread reads request frames and queues them, and the translation threads answer them in any
/// order, each answer carrying its request's seq. At most max_queued_jobs requests wait, and past
/// that the reader stops reading, so the pipe holds the parent back. The answers go out one at a
/// time, since the channel allows one writer. When the parent closes the pipe, or a write to it
/// fails, the worker stops, and requests still queued are dropped unanswered.
///
/// A Plan request translates the program, keeps the translation in the cache and answers with
/// SaveResourcePlan bytes. A Compile request takes the translation from the cache, or makes it,
/// compiles it with the request's specialization and answers with EncodeModuleRecord bytes. A
/// program with GPU-selected calls is answered HasCalls. A request is answered Failed when its code
/// is over MAX_STREAM_CODE_BYTES or does not hash to its code hash, its user data count is not the
/// registers' user SGPR count, its dispatch cannot be prescanned, or its specialization is over
/// max_specialization_bytes, does not decode or does not cover the program. Every answer
/// past those first checks echoes the static key words the worker built from the request's
/// registers. An answer too large for a frame goes out as Failed with no payload, so the parent is
/// never left waiting for it. A request that a stopping worker had already taken is answered
/// Dropped.
///
/// The progress log gets `priority <name>` first, then `start <seq> <hash>` before a request's
/// work and `done <seq> <status>` after it, with the status ok, calls, failed or dropped. A queued
/// request dropped at the end gets a `done <seq> dropped` line and no start line. A crash is blamed
/// on the requests with a `start` line and no `done` line. The last line is `summary
/// exit=<code>` with the counts, unless the process died.
class StreamWorker {
public:
	StreamWorker(const WorkerRequest& request, StreamWorkerOptions options = {});
	~StreamWorker();
	StreamWorker(const StreamWorker&)            = delete;
	StreamWorker& operator=(const StreamWorker&) = delete;

	/// Serves the pipe until the parent closes it or something fails.
	/// @return a WORKER_EXIT_ value. DONE after a clean close, PIPE_FAILED when a write failed
	///         while the parent had not closed the pipe, PROTOCOL_ERROR after a frame that was not
	///         a request or a stream that ended inside a frame.
	int Run();

	/// What the worker has done so far. Any thread may call it, during Run or after it.
	[[nodiscard]] StreamWorkerStats Stats() const;

	/// The state of a run. Defined in the source file.
	struct Impl;

private:
	std::unique_ptr<Impl> m_impl;
};

/// Runs a Stream worker request to the end, in this process.
StreamWorkerResult RunStreamWorker(const WorkerRequest& request, StreamWorkerOptions options = {});

/// Answers one request the way a Stream worker does, on the calling thread and without the
/// worker's translation cache, so each request translates its program afresh. The checks, the
/// translation and the compile are the worker's own, so an answer equals the child's. The
/// pre-scan's in-process transport uses it in tests, and it never runs in the emulator, since a
/// recompiler EXIT here would end the process.
/// @param request the request, with its code in request.code.
/// @param host    the host values the worker would decode from its store header.
/// @param device  the host subgroup size and the shared memory limit.
[[nodiscard]] StreamResponse AnswerStreamRequest(const StreamRequest& request, const HostValues& host,
                                                 const ComputeKeyDevice& device);

/// The bytes of a specialization, as a Compile request carries it.
[[nodiscard]] std::vector<uint8_t>
EncodeSpecialization(const ShaderRecompiler::IR::ResourceSpecialization& specialization);

/// Reads a specialization from a Compile request.
/// @return nothing when the bytes are cut short, have bytes left over, or hold a flag byte that is
///         neither 0 nor 1.
[[nodiscard]] std::optional<ShaderRecompiler::IR::ResourceSpecialization>
DecodeSpecialization(std::span<const uint8_t> bytes);

/// The priority this process runs at, as a word for the report. On Windows it is the priority class,
/// such as below_normal, and on POSIX it is the nice value of the calling thread, such as nice_10.
[[nodiscard]] std::string ProcessPriorityName();

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_STREAM_WORKER_H
