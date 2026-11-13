#include "graphics/shader/storegen/streamWorker.h"

#include "common/logging/log.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/storegen/appendFile.h"
#include "graphics/shader/storegen/computeCompile.h"
#include "graphics/shader/storegen/computeTranslation.h"
#include "graphics/shader/storegen/scopeExit.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/streamChannel.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fmt/format.h>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>
#include <xxhash.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <sys/resource.h>
#endif

namespace Libs::Graphics::StoreGen {

namespace IR = ShaderRecompiler::IR;

namespace {

// The requests the reader has taken and no translation thread has started yet. It holds a limited
// number, and a reader that finds it full waits, so a parent that sends too fast is held back by
// the pipe rather than growing the worker's memory.
class JobQueue {
public:
	explicit JobQueue(size_t limit): m_limit(std::max<size_t>(limit, 1)) {}

	// Adds a request, waiting while the queue is full, and wakes one translation thread for it.
	// Returns false when the queue was closed first, and the request is then dropped.
	bool Push(StreamRequest request) {
		{
			std::unique_lock lock(m_mutex);
			m_room.wait(lock, [&] { return m_closed || m_jobs.size() < m_limit; });
			if (m_closed) {
				return false;
			}
			m_jobs.push_back(std::move(request));
		}
		m_ready.notify_one();
		return true;
	}

	// Waits for a request and makes room for the reader. Returns nothing once the queue is closed,
	// even with requests left, since the parent that would read their answers has gone.
	std::optional<StreamRequest> Pop() {
		std::optional<StreamRequest> request;
		{
			std::unique_lock lock(m_mutex);
			m_ready.wait(lock, [&] { return m_closed || !m_jobs.empty(); });
			if (m_closed) {
				return std::nullopt;
			}
			request = std::move(m_jobs.front());
			m_jobs.pop_front();
		}
		m_room.notify_one();
		return request;
	}

	// Closes the queue, wakes every waiting thread and returns the seqs of the requests dropped. A
	// second call drops nothing more.
	std::vector<uint64_t> Close() {
		std::vector<uint64_t> dropped;
		{
			std::lock_guard lock(m_mutex);
			m_closed = true;
			for (const auto& job: m_jobs) {
				dropped.push_back(job.seq);
			}
			m_jobs.clear();
		}
		m_ready.notify_all();
		m_room.notify_all();
		return dropped;
	}

private:
	std::mutex                m_mutex;
	std::condition_variable   m_ready;
	std::condition_variable   m_room;
	std::deque<StreamRequest> m_jobs;
	size_t                    m_limit  = 1;
	bool                      m_closed = false;
};

// What every kind of request needs before its work, made once from the request.
struct PreparedRequest {
	ShaderComputeInputInfo info {};
	TranslationKey         key;
	std::vector<uint32_t>  code;
};

// Whether a request's code and counts can be used. The code is size checked before it is hashed,
// and hashed as the shader map hashes it, so a module is never filed under another program's hash.
// The user data count must be the registers' user SGPR count, which is where the runtime takes it
// from, since a parent that sent two different values has a bug a module would hide.
bool RequestIsWhole(const StreamRequest& request) {
	return !request.code.empty() && request.code.size() <= MAX_STREAM_CODE_BYTES &&
	       request.code.size() % sizeof(uint32_t) == 0 &&
	       XXH3_64bits(request.code.data(), request.code.size()) == request.code_hash &&
	       request.user_data_count == request.dispatch.cs_regs.user_sgpr;
}

// Checks a request and builds what the work needs.
// Returns nothing when RequestIsWhole refuses it or the dispatch is one the pre-scan cannot build
// an input info for.
std::optional<PreparedRequest> PrepareRequest(const StreamRequest&    request,
                                              const ComputeKeyDevice& device) {
	if (!RequestIsWhole(request)) {
		return std::nullopt;
	}
	const auto info = MakePrescanInputInfo(request.dispatch, request.scratch_size_dwords, device);
	if (!info) {
		return std::nullopt;
	}
	PreparedRequest prepared;
	prepared.info                = *info;
	prepared.key.code_hash       = request.code_hash;
	prepared.key.user_data_count = request.user_data_count;
	BuildStageStaticKey(prepared.info, prepared.key.static_state);
	prepared.code.resize(request.code.size() / sizeof(uint32_t));
	std::memcpy(prepared.code.data(), request.code.data(), request.code.size());
	return prepared;
}

// The unit a prepared request names, for the shared compile path.
ComputeUnit UnitOf(const PreparedRequest& prepared) {
	return {prepared.code, prepared.key.code_hash, prepared.key.user_data_count,
	        prepared.key.static_state};
}

// An answer to a request, which echoes its seq, its kind and the static key words.
StreamResponse MakeResponse(const StreamRequest& request, StreamStatus status,
                            std::vector<uint32_t> static_state, std::vector<uint8_t> payload = {}) {
	return {.seq          = request.seq,
	        .kind         = request.kind,
	        .status       = status,
	        .payload      = std::move(payload),
	        .static_state = std::move(static_state)};
}

// The word a `done` line gives for an answer's status.
const char* StatusWord(StreamStatus status) {
	switch (status) {
		case StreamStatus::Ok: return "ok";
		case StreamStatus::HasCalls: return "calls";
		case StreamStatus::Dropped: return "dropped";
		case StreamStatus::Failed: return "failed";
	}
	return "unknown";
}

// Reads one field the way Io writes it. A flag is read as a byte that must be 0 or 1, since a bool
// read from any other byte value would be undefined behaviour.
template <class T> void ReadCheckedField(IR::Reader& reader, bool& flags_ok, T& value) {
	if constexpr (std::is_same_v<T, bool>) {
		uint8_t byte = 0;
		IR::Io(reader, byte);
		flags_ok = flags_ok && byte <= 1;
		value    = byte != 0;
	} else {
		IR::Io(reader, value);
	}
}

// Reads fields in order, as ReadCheckedField does.
template <class... Fields>
void ReadCheckedFields(IR::Reader& reader, bool& flags_ok, Fields&... fields) {
	(ReadCheckedField(reader, flags_ok, fields), ...);
}

// Reads a list the way Io writes one, which is a count and then each element. The reader bounds
// the count by the bytes left, so a damaged count cannot allocate much.
template <class Element, class ReadOne>
void ReadCheckedList(IR::Reader& reader, std::vector<Element>& list, ReadOne read_one) {
	uint32_t count = 0;
	IR::Io(reader, count);
	list.resize(reader.Count(count));
	for (auto& element: list) {
		read_one(element);
	}
}

} // namespace

struct StreamWorker::Impl {
	Impl(const WorkerRequest& worker_request, StreamWorkerOptions worker_options)
	    : request(worker_request), options(std::move(worker_options)),
	      cache(options.cache_bytes, options.on_cache_wait), jobs(options.max_queued_jobs) {}

	int  Run();
	int  ConnectAndServe();
	int  Serve();
	int  ReadRequests();
	int  ExitCode(int reader_code) const;
	void Stop();
	void LogDropped(const std::vector<uint64_t>& seqs);
	void WorkLoop();
	bool Answer(const StreamRequest& job);
	void Send(const StreamResponse& response);

	StreamResponse Handle(const StreamRequest& job);
	StreamResponse Plan(const StreamRequest& job, const PreparedRequest& prepared);
	StreamResponse Compile(const StreamRequest& job, const PreparedRequest& prepared);
	std::optional<ShaderRecompiler::TranslateResult> Translate(const PreparedRequest& prepared);

	void              Count(const StreamRequest& job, const StreamResponse& response, bool refused);
	bool              Line(const std::string& text);
	void              Summarize(int code);
	StreamWorkerStats Snapshot() const;

	WorkerRequest       request;
	StreamWorkerOptions options;
	TranslationCache    cache;
	JobQueue            jobs;
	HostValues          host;
	ComputeKeyDevice    device {};

	AppendFile progress;
	std::mutex progress_mutex;

	std::optional<StreamChannel> channel;
	std::mutex                   send_mutex;
	/// Set once the worker is stopping, so a translation thread starts no new work.
	std::atomic<bool> stopping {false};
	/// Set when a progress line could not be written, or a write to the pipe failed.
	std::atomic<bool> output_failed {false};
	std::atomic<bool> send_failed {false};
	/// Set when the reader saw the parent close the pipe before the worker began to stop.
	std::atomic<bool> parent_closed {false};

	mutable std::mutex stats_mutex;
	StreamWorkerStats  stats;
};

// Checks the request, opens the log and serves the pipe. The summary closes a log that was opened.
int StreamWorker::Impl::Run() {
	const auto decoded = DecodeHostHeader(request.store_header);
	if (!decoded || request.host_subgroup_size == 0 || request.stream_pipe.empty()) {
		return WORKER_EXIT_BAD_REQUEST;
	}
	host   = *decoded;
	device = {request.host_subgroup_size, host.max_shared_bytes};
	if (!progress.OpenAppend(PathFromUtf8(request.progress_path))) {
		return WORKER_EXIT_OUTPUT_FAILED;
	}
	const auto priority = ProcessPriorityName();
	{
		std::lock_guard lock(stats_mutex);
		stats.priority = priority;
	}
	// The class the system really gave the process, which the spawner asked for and the report
	// shows, since a parent at idle class makes its children idle too.
	if (!Line("priority " + priority)) {
		return WORKER_EXIT_OUTPUT_FAILED;
	}
	const int code = ConnectAndServe();
	Summarize(code);
	return code;
}

// Connects to the pipe and serves it, but only when the server is the parent that started this
// worker, so a process that took the pipe's name first is never sent a module.
int StreamWorker::Impl::ConnectAndServe() {
	auto pipe = ConnectStreamPipeWithin(request.stream_pipe, options.connect_timeout);
	if (!pipe) {
		LOGF("Stream worker: could not connect to the pipe %s\n", request.stream_pipe.c_str());
		return WORKER_EXIT_PIPE_FAILED;
	}
	if (!pipe->IsPeerProcess(request.parent_pid)) {
		LOGF("Stream worker: the pipe's server is not process %u, so nothing is served\n",
		     request.parent_pid);
		return WORKER_EXIT_PEER_MISMATCH;
	}
	channel.emplace(std::move(*pipe));
	return Serve();
}

// Runs the translation threads while this thread reads requests. Once reading ends the worker is
// stopped, then the threads are joined and the pipe is closed, in the order StreamPipe requires.
// The guard stops the worker on an exception too, since a join would otherwise wait for ever on a
// thread waiting for a request or for room to write.
int StreamWorker::Impl::Serve() {
	std::vector<std::jthread> threads;
	ScopeExit                 stop_before_join([this] { Stop(); });
	const uint32_t            count = std::max(options.translation_threads, 1u);
	threads.reserve(count);
	for (uint32_t index = 0; index < count; index++) {
		threads.emplace_back([this] { WorkLoop(); });
	}
	const int code = ReadRequests();
	Stop();
	stop_before_join.Dismiss();
	threads.clear();
	channel->Close();
	return ExitCode(code);
}

// The run's exit code. A log that failed comes first, then a write that failed while the parent
// had not closed the pipe, and otherwise what the reader saw. If the parent closes while a write is
// in progress and the write fails before the reader sees Closed, the exit is still PIPE_FAILED,
// since WriteAll cannot tell those apart, so C9 ignores the exit code of a child whose pipe it
// closed.
int StreamWorker::Impl::ExitCode(int reader_code) const {
	if (output_failed) {
		return WORKER_EXIT_OUTPUT_FAILED;
	}
	if (send_failed && !parent_closed) {
		return WORKER_EXIT_PIPE_FAILED;
	}
	return reader_code;
}

// Stops the worker. The queue is closed first, so a reader waiting for room wakes and no thread
// takes another request, and then the pipe is interrupted, so a blocked read or write wakes. Any
// thread may call it, any number of times.
void StreamWorker::Impl::Stop() {
	stopping = true;
	LogDropped(jobs.Close());
	channel->Interrupt();
}

// Logs and counts the requests dropped unanswered. They have no start line, so a crash is never
// blamed on them.
void StreamWorker::Impl::LogDropped(const std::vector<uint64_t>& seqs) {
	for (const auto seq: seqs) {
		(void)Line(fmt::format("done {} dropped", seq));
	}
	std::lock_guard lock(stats_mutex);
	stats.dropped += seqs.size();
}

// Reads request frames and queues them until the parent closes the pipe or the worker stops. A
// stream that ends inside a frame, or a frame that is not a request, leaves the stream out of
// step for good, so either ends the run.
int StreamWorker::Impl::ReadRequests() {
	for (;;) {
		const auto read = channel->ReadFrame();
		if (read.status == StreamReadStatus::Closed) {
			parent_closed = !stopping;
			return WORKER_EXIT_DONE;
		}
		if (read.status == StreamReadStatus::Truncated) {
			LOGF("Stream worker: the stream ended inside a frame\n");
			return WORKER_EXIT_PROTOCOL_ERROR;
		}
		auto job = read.status == StreamReadStatus::Frame ? DecodeStreamRequest(read.frame)
		                                                  : std::nullopt;
		if (!job) {
			LOGF("Stream worker: a frame was not a request, so the stream is given up\n");
			return WORKER_EXIT_PROTOCOL_ERROR;
		}
		const uint64_t seq = job->seq;
		if (!jobs.Push(std::move(*job))) {
			// The queue closed while the reader waited for room, so the request is given up like
			// the ones the close dropped.
			LogDropped({seq});
			return WORKER_EXIT_DONE;
		}
		std::lock_guard lock(stats_mutex);
		stats.requests++;
	}
}

// Answers requests until the queue closes. A log that cannot be written cannot blame a crash, so
// that stops the whole worker rather than serving on without it.
void StreamWorker::Impl::WorkLoop() {
	while (auto job = jobs.Pop()) {
		if (!Answer(*job)) {
			LOGF("Stream worker: the progress log could not be written, so the worker stops\n");
			output_failed = true;
			Stop();
			return;
		}
	}
}

// Logs a request's start, does its work, logs its end and sends the answer. The end is logged
// before the send, so once the parent has an answer its `done` line is already in the log. An
// answer too large for a frame is turned into Failed with no payload first, so the parent always
// gets an answer and never waits out its deadline for one. StreamResponseEncodes only adds up
// sizes, so the answer is still encoded once, by the send.
// Returns false when the log could not be written.
bool StreamWorker::Impl::Answer(const StreamRequest& job) {
	if (stopping) {
		LogDropped({job.seq});
		return true;
	}
	if (!Line(fmt::format("start {} {:016x}", job.seq, job.code_hash))) {
		return false;
	}
	if (options.before_work) {
		options.before_work(job.seq);
	}
	auto response = Handle(job);
	if (options.before_send) {
		options.before_send(response);
	}
	const bool refused = !StreamResponseEncodes(response);
	if (refused) {
		LOGF("Stream worker: the answer to request %llu does not fit in a frame, so it is sent as "
		     "failed\n",
		     static_cast<unsigned long long>(job.seq));
		response.status = StreamStatus::Failed;
		response.payload.clear();
	}
	Count(job, response, refused);
	if (!Line(fmt::format("done {} {}", job.seq, StatusWord(response.status)))) {
		return false;
	}
	Send(response);
	return true;
}

// Sends one answer. The channel allows one writer, so the threads take turns. A write that failed
// may have left half a frame in the pipe, so nothing more can be sent, and the worker stops. A
// write that fails once the worker is already stopping failed because Stop interrupted the pipe,
// so it is not counted as a failed write, and the reason the worker stopped keeps its exit code.
void StreamWorker::Impl::Send(const StreamResponse& response) {
	std::lock_guard lock(send_mutex);
	if (send_failed) {
		return;
	}
	if (!channel->Send(response)) {
		if (!stopping) {
			send_failed = true;
		}
		Stop();
	}
}

// Does a request's work, after the checks every kind shares.
StreamResponse StreamWorker::Impl::Handle(const StreamRequest& job) {
	const auto prepared = PrepareRequest(job, device);
	if (!prepared) {
		return MakeResponse(job, StreamStatus::Failed, {});
	}
	return job.kind == StreamRequestKind::Plan ? Plan(job, *prepared) : Compile(job, *prepared);
}

// Answers a Plan request with the program's resource plan. The translation stays in the cache for
// the Compile that follows, and the lease is let go before the answer is sent, so a Compile that
// arrives as soon as the parent has the plan finds it. A thread that waited for the program while
// the worker began to stop gives the request up.
StreamResponse StreamWorker::Impl::Plan(const StreamRequest& job, const PreparedRequest& prepared) {
	auto lease = cache.Acquire(prepared.key, TranslationUse::Plan);
	if (stopping) {
		return MakeResponse(job, StreamStatus::Dropped, prepared.key.static_state);
	}
	if (lease.Result() == nullptr) {
		auto translated = Translate(prepared);
		if (!translated) {
			return MakeResponse(job, StreamStatus::HasCalls, prepared.key.static_state);
		}
		const auto bytes = EstimateTranslationBytes(*translated);
		lease.Put(std::move(*translated), bytes);
	}
	auto plan = IR::SaveResourcePlan(IR::ExtractResourcePlan(lease.Result()->program));
	lease.Release();
	return MakeResponse(job, StreamStatus::Ok, prepared.key.static_state, std::move(plan));
}

// Answers a Compile request with the module's record, built through the path the generator uses.
// The translation is taken out of the cache, or made while the program is held, so no other thread
// translates it at the same time, and one made here is consumed by the compile without entering
// the cache.
StreamResponse StreamWorker::Impl::Compile(const StreamRequest&   job,
                                           const PreparedRequest& prepared) {
	const auto& state = prepared.key.static_state;
	const auto  specialization =
        job.specialization.size() <= options.max_specialization_bytes
	         ? DecodeSpecialization(job.specialization)
	         : std::nullopt;
	if (!specialization) {
		return MakeResponse(job, StreamStatus::Failed, state);
	}
	std::optional<ShaderRecompiler::TranslateResult> translated;
	{
		auto lease = cache.Acquire(prepared.key, TranslationUse::Compile);
		if (stopping) {
			return MakeResponse(job, StreamStatus::Dropped, state);
		}
		translated = lease.Take();
		if (!translated) {
			translated = Translate(prepared);
		}
	}
	if (!translated) {
		return MakeResponse(job, StreamStatus::HasCalls, state);
	}
	TranslationSetup setup(prepared.info, prepared.key.user_data_count, prepared.key.code_hash, host);
	auto build = CompileComputeUnit(std::move(*translated), UnitOf(prepared), setup, *specialization);
	if (build.status != ComputeBuildStatus::Built) {
		return MakeResponse(job, StreamStatus::Failed, state);
	}
	return MakeResponse(job, StreamStatus::Ok, state, std::move(build.record));
}

// Translates a program with zeroed user data, as the generator does.
// Returns nothing for a program with GPU-selected calls.
std::optional<ShaderRecompiler::TranslateResult>
StreamWorker::Impl::Translate(const PreparedRequest& prepared) {
	if (options.on_translate) {
		options.on_translate(prepared.key);
	}
	TranslationSetup setup(prepared.info, prepared.key.user_data_count, prepared.key.code_hash, host);
	return TranslateComputeUnit(UnitOf(prepared), setup);
}

// Counts an answer by its kind and status.
void StreamWorker::Impl::Count(const StreamRequest& job, const StreamResponse& response,
                               bool refused) {
	std::lock_guard lock(stats_mutex);
	(job.kind == StreamRequestKind::Plan ? stats.plans : stats.compiles)++;
	stats.has_calls += response.status == StreamStatus::HasCalls ? 1 : 0;
	stats.failed += response.status == StreamStatus::Failed ? 1 : 0;
	stats.dropped += response.status == StreamStatus::Dropped ? 1 : 0;
	stats.refused_replies += refused ? 1 : 0;
}

// Writes one progress line. The translation threads share the log, so they take turns, and each
// line goes out in one write so lines never interleave.
bool StreamWorker::Impl::Line(const std::string& text) {
	std::lock_guard lock(progress_mutex);
	return progress.Line(text);
}

// Writes the counts as the log's last line. Nothing can be done about a log that fails to write at
// this point, so it is best effort.
void StreamWorker::Impl::Summarize(int code) {
	const auto  snapshot = Snapshot();
	const auto& cached   = snapshot.cache;
	(void)Line(fmt::format(
	    "summary exit={} requests={} plans={} compiles={} with_calls={} failed={} refused={} "
	    "dropped={} cache_hits={} cache_plan_misses={} cache_compile_misses={} "
	    "compile_evictions={} cap_evictions={} cache_waits={} cache_peak_bytes={} priority={}",
	    code, snapshot.requests, snapshot.plans, snapshot.compiles, snapshot.has_calls,
	    snapshot.failed, snapshot.refused_replies, snapshot.dropped, cached.hits, cached.plan_misses,
	    cached.compile_misses, cached.compile_evictions, cached.cap_evictions, cached.waits,
	    cached.peak_bytes, snapshot.priority));
}

// The counts so far with the cache's own.
StreamWorkerStats StreamWorker::Impl::Snapshot() const {
	StreamWorkerStats snapshot;
	{
		std::lock_guard lock(stats_mutex);
		snapshot = stats;
	}
	snapshot.cache = cache.Stats();
	return snapshot;
}

StreamWorker::StreamWorker(const WorkerRequest& request, StreamWorkerOptions options)
    : m_impl(std::make_unique<Impl>(request, std::move(options))) {}

StreamWorker::~StreamWorker() = default;

int StreamWorker::Run() {
	return m_impl->Run();
}

StreamWorkerStats StreamWorker::Stats() const {
	return m_impl->Snapshot();
}

StreamWorkerResult RunStreamWorker(const WorkerRequest& request, StreamWorkerOptions options) {
	StreamWorker       worker(request, std::move(options));
	StreamWorkerResult result;
	result.exit_code = worker.Run();
	result.stats     = worker.Stats();
	return result;
}

// The worker's Handle, Plan and Compile without the cache. A Plan translates and extracts the plan,
// and a Compile decodes the specialization and builds through the generator's path. The size cap
// on the specialization is the worker's default, since there is no options object here.
StreamResponse AnswerStreamRequest(const StreamRequest& request, const HostValues& host,
                                   const ComputeKeyDevice& device) {
	const auto prepared = PrepareRequest(request, device);
	if (!prepared) {
		return MakeResponse(request, StreamStatus::Failed, {});
	}
	const auto&      state = prepared->key.static_state;
	TranslationSetup setup(prepared->info, prepared->key.user_data_count, prepared->key.code_hash,
	                       host);
	if (request.kind == StreamRequestKind::Plan) {
		auto translated = TranslateComputeUnit(UnitOf(*prepared), setup);
		if (!translated) {
			return MakeResponse(request, StreamStatus::HasCalls, state);
		}
		auto plan = IR::SaveResourcePlan(IR::ExtractResourcePlan(translated->program));
		return MakeResponse(request, StreamStatus::Ok, state, std::move(plan));
	}
	const auto specialization = request.specialization.size() <= MAX_STREAM_SPECIALIZATION_BYTES
	                                ? DecodeSpecialization(request.specialization)
	                                : std::nullopt;
	if (!specialization) {
		return MakeResponse(request, StreamStatus::Failed, state);
	}
	auto translated = TranslateComputeUnit(UnitOf(*prepared), setup);
	if (!translated) {
		return MakeResponse(request, StreamStatus::HasCalls, state);
	}
	auto build = CompileComputeUnit(std::move(*translated), UnitOf(*prepared), setup, *specialization);
	if (build.status != ComputeBuildStatus::Built) {
		return MakeResponse(request, StreamStatus::Failed, state);
	}
	return MakeResponse(request, StreamStatus::Ok, state, std::move(build.record));
}

std::vector<uint8_t> EncodeSpecialization(const IR::ResourceSpecialization& specialization) {
	IR::Writer writer;
	// A copy, because Io takes its values by reference.
	auto copy = specialization;
	IR::IoAll(writer, copy);
	return std::move(writer.bytes);
}

// Reads the bytes EncodeSpecialization writes, with every flag byte checked as HeaderFlagsAreBooleans
// checks the store header's. Each struct is bound field by field in the order Serialize.h binds it,
// so a field added to one fails to compile here until it is listed, and the names are only places.
std::optional<IR::ResourceSpecialization> DecodeSpecialization(std::span<const uint8_t> bytes) {
	IR::Reader                 reader {bytes};
	bool                       flags_ok = true;
	IR::ResourceSpecialization specialization;
	ReadCheckedList(reader, specialization.buffers, [&](auto& buffer) {
		auto& [a, b, c, d, e, f, g, h] = buffer;
		ReadCheckedFields(reader, flags_ok, a, b, c, d, e, f, g, h);
	});
	ReadCheckedList(reader, specialization.images, [&](auto& image) {
		auto& [a, b, c, d, e, f, g, h, i, j, k, l, m] = image;
		ReadCheckedFields(reader, flags_ok, a, b, c, d, e, f, g, h, i, j, k, l, m);
	});
	ReadCheckedList(reader, specialization.samplers, [&](auto& sampler) {
		auto& [a, b, c] = sampler;
		ReadCheckedFields(reader, flags_ok, a, b, c);
	});
	if (!reader.ok || !reader.bytes.empty() || !flags_ok) {
		return std::nullopt;
	}
	return specialization;
}

std::string ProcessPriorityName() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	switch (GetPriorityClass(GetCurrentProcess())) {
		case IDLE_PRIORITY_CLASS: return "idle";
		case BELOW_NORMAL_PRIORITY_CLASS: return "below_normal";
		case NORMAL_PRIORITY_CLASS: return "normal";
		case ABOVE_NORMAL_PRIORITY_CLASS: return "above_normal";
		case HIGH_PRIORITY_CLASS: return "high";
		case REALTIME_PRIORITY_CLASS: return "realtime";
		default: return "unknown";
	}
#else
	// getpriority returns -1 for a nice value of -1 too, so errno tells a failure apart.
	errno           = 0;
	const int value = getpriority(PRIO_PROCESS, 0);
	if (value == -1 && errno != 0) {
		return "unknown";
	}
	return fmt::format("nice_{}", value);
#endif
}

} // namespace Libs::Graphics::StoreGen
