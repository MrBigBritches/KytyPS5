#include "graphics/shader/storegen/streamSupervisor.h"

#include "common/childProcess.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/storeSupervisor.h"
#include "graphics/shader/storegen/streamChannel.h"
#include "graphics/shader/storegen/workerProcess.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <iterator>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics::StoreGen {

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

static_assert(STREAM_MAX_RESTARTS == GENERATION_MAX_RESTARTS,
              "the stream lane absorbs as many failures as the generator");

namespace {

/// How many bytes of the progress log one read takes.
constexpr size_t PROGRESS_READ_CHUNK_BYTES = 64 * 1024;
/// The percentiles the report gives for the round trips.
constexpr double RTT_P50 = 0.5;
constexpr double RTT_P90 = 0.9;
/// How long one wait for a child's exit lasts before the stop flag is looked at again.
constexpr std::chrono::milliseconds STOP_CHECK_SLICE {10};
/// The seed of a program's retry key, so it never equals a request's key by its layout.
constexpr uint64_t PROGRAM_KEY_SEED = 0x50524F47;

// One request the supervisor holds, waiting or sent. It is always in exactly one of the two
// containers, pending or outstanding, and never in both.
struct Entry {
	/// The request, without its code.
	StreamRequest request;
	/// The program's code, which a Plan and the Compiles of one program share. Never null.
	std::shared_ptr<const std::vector<uint8_t>> code;
	uint64_t                                    key         = 0;
	uint64_t                                    program_key = 0;
	uint64_t                                    tag         = 0;
	/// When it was last sent, which its deadline and its round trip are measured from.
	Clock::time_point sent {};
	/// Whether it was running with others when a child died. Such a request is sent alone, so the
	/// next failure while it runs can be blamed on it.
	bool suspect = false;
};

// What the owner thread does next.
enum class Work : uint8_t { Stop, Connect, Ended, Deadline, Send, Check };

// Why a connection's reading ended, as the reader saw it.
enum class ReaderEnd : uint8_t {
	Running,       ///< Still reading.
	Closed,        ///< The pipe ended, between frames or inside one, or was interrupted.
	ProtocolError, ///< A frame header was refused or an answer did not decode.
};

double MillisecondsBetween(Clock::time_point from, Clock::time_point to) {
	return std::chrono::duration<double, std::milli>(to - from).count();
}

// The size of a file, or 0 when it is not there, which is where a new spawn's lines begin.
uint64_t FileSizeOrZero(const fs::path& path) {
	Common::File file;
	return file.Open(path, Common::File::Mode::Read) ? file.Size() : 0;
}

// The completion of a request that was not answered.
StreamCompletion Unanswered(const Entry& entry, StreamOutcome outcome) {
	StreamCompletion completion;
	completion.seq     = entry.request.seq;
	completion.kind    = entry.request.kind;
	completion.outcome = outcome;
	completion.tag     = entry.tag;
	return completion;
}

// The code of a request submitted with none, shared by all of them.
std::shared_ptr<const std::vector<uint8_t>> EmptyCode() {
	static const auto empty = std::make_shared<const std::vector<uint8_t>>();
	return empty;
}

// Takes one complete line of a Stream worker's log into what the spawn's progress says. Only the
// seqs asked for are kept, so a long session's log costs no memory here.
void TakeStreamLine(std::string_view line, std::span<const uint64_t> seqs,
                    StreamSpawnProgress& progress) {
	if (!line.empty() && line.back() == '\r') {
		line.remove_suffix(1);
	}
	const auto parsed = line.empty() ? std::nullopt : ParseProgressLine(line);
	if (!parsed) {
		return;
	}
	const bool asked = std::ranges::find(seqs, parsed->rank_or_count) != seqs.end();
	if (parsed->kind == ProgressKind::Priority) {
		progress.priority = parsed->priority;
	} else if (parsed->kind == ProgressKind::Start && asked) {
		progress.started.push_back(parsed->rank_or_count);
	} else if (parsed->kind == ProgressKind::Done && asked) {
		progress.done.push_back(parsed->rank_or_count);
	}
}

// Splits the bytes of a log into lines, holding a line until its newline arrives. A line longer
// than any the worker writes is skipped up to its newline, so no piece of it is read as a line.
class LineSplitter {
public:
	template <class OnLine> void Feed(std::string_view bytes, OnLine&& on_line) {
		if (m_skipping) {
			const auto newline = bytes.find('\n');
			if (newline == std::string_view::npos) {
				return;
			}
			bytes.remove_prefix(newline + 1);
			m_skipping = false;
		}
		m_held.append(bytes);
		size_t begin = 0;
		for (auto end = m_held.find('\n'); end != std::string::npos;
		     end      = m_held.find('\n', begin)) {
			on_line(std::string_view(m_held).substr(begin, end - begin));
			begin = end + 1;
		}
		m_held.erase(0, begin);
		if (m_held.size() > MAX_PROGRESS_LINE_BYTES) {
			m_held.clear();
			m_skipping = true;
		}
	}

private:
	std::string m_held;
	bool        m_skipping = false;
};

} // namespace

void StreamRttHistogram::Add(double milliseconds) {
	m_buckets[BucketOf(milliseconds)]++;
	m_count++;
	m_max_ms = std::max(m_max_ms, milliseconds);
}

double StreamRttHistogram::Percentile(double fraction) const {
	if (m_count == 0) {
		return 0.0;
	}
	// Nearest rank, so the median of two is the first.
	const auto rank = std::max<uint64_t>(
	    1, static_cast<uint64_t>(std::ceil(fraction * static_cast<double>(m_count))));
	uint64_t seen = 0;
	for (size_t bucket = 0; bucket < BUCKETS; bucket++) {
		seen += m_buckets[bucket];
		if (seen >= rank) {
			return std::min(UpperBoundOf(bucket), m_max_ms);
		}
	}
	return m_max_ms;
}

// The bucket a round trip counts in. Below the split the whole milliseconds are the bucket, and
// above it each coarse bucket covers COARSE_WIDTH_MS, with the last taking everything longer.
size_t StreamRttHistogram::BucketOf(double milliseconds) {
	if (!(milliseconds > 0.0)) {
		return 0;
	}
	if (milliseconds < FINE_LIMIT_MS) {
		return static_cast<size_t>(milliseconds);
	}
	const auto coarse = static_cast<size_t>((milliseconds - FINE_LIMIT_MS) / COARSE_WIDTH_MS);
	return std::min(FINE_LIMIT_MS + coarse, BUCKETS - 1);
}

// The longest round trip a bucket holds, which a percentile reports.
double StreamRttHistogram::UpperBoundOf(size_t bucket) {
	if (bucket < FINE_LIMIT_MS) {
		return static_cast<double>(bucket + 1);
	}
	return static_cast<double>(FINE_LIMIT_MS + (bucket - FINE_LIMIT_MS + 1) * COARSE_WIDTH_MS);
}

uint64_t StreamRetryKey(const StreamRequest& request) {
	// The fixed fields are copied one by one into bytes, so no padding is hashed.
	std::array<uint8_t, sizeof(uint8_t) + sizeof(uint64_t) + sizeof(uint32_t)> head {};
	const auto kind = static_cast<uint8_t>(request.kind);
	std::memcpy(head.data(), &kind, sizeof(kind));
	std::memcpy(head.data() + sizeof(kind), &request.code_hash, sizeof(request.code_hash));
	std::memcpy(head.data() + sizeof(kind) + sizeof(request.code_hash), &request.user_data_count,
	            sizeof(request.user_data_count));
	const uint64_t head_hash = XXH3_64bits(head.data(), head.size());
	return XXH3_64bits_withSeed(request.specialization.data(), request.specialization.size(),
	                            head_hash);
}

uint64_t StreamProgramRetryKey(const StreamRequest& request) {
	std::array<uint8_t, sizeof(uint64_t) + sizeof(uint32_t)> program {};
	std::memcpy(program.data(), &request.code_hash, sizeof(request.code_hash));
	std::memcpy(program.data() + sizeof(request.code_hash), &request.user_data_count,
	            sizeof(request.user_data_count));
	return XXH3_64bits_withSeed(program.data(), program.size(), PROGRAM_KEY_SEED);
}

StreamSpawnProgress ReadStreamSpawnProgress(const fs::path& path, uint64_t offset,
                                            std::span<const uint64_t> seqs) {
	StreamSpawnProgress progress;
	Common::File        file;
	if (!file.Open(path, Common::File::Mode::Read)) {
		return progress;
	}
	const uint64_t size = file.Size();
	if (size <= offset || !file.Seek(offset)) {
		return progress;
	}
	std::vector<char> chunk(PROGRESS_READ_CHUNK_BYTES);
	LineSplitter      lines;
	for (uint64_t at = offset; at < size;) {
		const auto want = static_cast<uint32_t>(std::min<uint64_t>(size - at, chunk.size()));
		uint32_t   got  = 0;
		file.Read(chunk.data(), want, &got);
		if (got == 0) {
			break;
		}
		at += got;
		lines.Feed(std::string_view(chunk.data(), got),
		           [&](std::string_view line) { TakeStreamLine(line, seqs, progress); });
	}
	// What the splitter still holds has no newline. The child died while it wrote that line, so
	// the line is ignored, and a request whose `done` line was torn still counts as running.
	return progress;
}

struct StreamSupervisor::Impl {
	explicit Impl(StreamSupervisorConfig supplied);

	// ---- The owner thread. ----
	void                              Run();
	Work                              NextWork();
	[[nodiscard]] std::optional<Work> ReadyWorkLocked(Clock::time_point now) const;
	[[nodiscard]] bool                CanSendLocked() const;
	[[nodiscard]] Clock::time_point   OldestDeadlineLocked() const;
	void                              Shutdown();

	// ---- Starting a child and connecting. ----
	void                      Connect();
	bool                      StartChild(const std::string& pipe_name);
	std::optional<StreamPipe> AcceptChild(StreamPipeListener& listener);
	void                      BeginConnection(StreamPipe accepted);
	void                      FailSpawn();

	// ---- Sending and reading. ----
	std::optional<Clock::time_point> SendWhatFits();
	std::optional<Clock::time_point> MoveNextToSendLocked(std::vector<StreamCompletion>& unsent);
	void                             ReadAnswers();
	void                             Deliver(StreamResponse&& response);
	void                             FinishReading(ReaderEnd end);
	[[nodiscard]] bool               ReaderEndIs(ReaderEnd end) const;
	[[nodiscard]] bool               ConnectionEndedLocked() const;

	// ---- A connection that ended. ----
	void EndConnection(bool hang, Clock::time_point not_before = Clock::time_point::min());
	std::optional<int> CloseConnection(bool kill_now);
	std::optional<int> WaitForChildExit(std::chrono::milliseconds limit);
	void               WaitForReaderEnd(std::chrono::milliseconds limit);
	void               NoteFailure(bool hang, std::optional<int> own_exit);
	void               HandleProtocolError();
	void               RecoverOutstanding(bool hang, Clock::time_point kill_time);
	std::vector<bool>  RunningFlags(const std::vector<Entry>& lost) const;
	void               Requeue(std::vector<Entry> lost, bool count_resent);
	std::vector<StreamCompletion> SweepBlockedPending();
	StreamCompletion   Blame(const Entry& entry);
	void               CountRestart();
	void               TurnLaneOff(const std::string& reason);

	// ---- Shared helpers. ----
	[[nodiscard]] StreamSubmitStatus RefusalLocked(uint64_t key, uint64_t program_key) const;
	[[nodiscard]] bool               IsBlockedLocked(uint64_t key, uint64_t program_key) const;
	[[nodiscard]] std::vector<Entry> TakeAllLocked();
	[[nodiscard]] bool               IsStopping() const;
	void                             Complete(StreamCompletion&& completion);
	void                             NoteCallbackFailure(const char* what);
	void                             CompleteDropped(std::vector<Entry>& entries);
	void                             InterruptPipe();

	StreamSupervisorConfig config;
	fs::path               progress_path;
	fs::path               request_path;
	WorkerRequest          request;

	// Shared by the owner, the reader, Submit and Stop.
	mutable std::mutex           mutex;
	std::condition_variable      wake;
	std::deque<Entry>            pending;
	std::vector<Entry>           outstanding;
	std::unordered_set<uint64_t> never_retry;
	std::unordered_set<uint64_t> never_retry_programs;
	uint64_t                     next_seq   = 1;
	bool                         stopping   = false;
	bool                         lane_off   = false;
	// Why the reader stopped, which only the reader sets. It is kept apart from a failed write, so
	// the drain still waits for the reader and a bad frame it finds later is not hidden.
	ReaderEnd reader_end = ReaderEnd::Running;
	// Whether a write to the current connection failed, which only the owner sets.
	bool                  write_failed = false;
	StreamSupervisorStats stats;
	StreamRttHistogram           plan_rtt;
	StreamRttHistogram           compile_rtt;

	// Keeps two outcomes from reaching the callback at once.
	std::mutex callback_mutex;
	// Guards whether the pipe exists, so Stop can interrupt it from its own thread while the owner
	// makes or destroys it. Reads and writes go on without it, as StreamPipe allows.
	std::mutex                pipe_mutex;
	std::optional<StreamPipe> pipe;
	// Keeps two Stops from joining the owner at once.
	std::mutex stop_mutex;

	// Owned by the owner thread.
	std::unique_ptr<Common::ChildProcess> child;
	std::thread                           reader;
	std::vector<uint8_t>                  send_frame;
	uint64_t                              spawn_offset = 0;
	bool                                  connected    = false;
	bool                                  first_spawn  = true;

	// Started last, once everything it reads is made.
	std::thread owner;
};

StreamSupervisor::Impl::Impl(StreamSupervisorConfig supplied): config(std::move(supplied)) {
	const auto base = fs::absolute(config.files_base);
	progress_path   = fs::path(base).concat(".progress");
	request_path    = fs::path(base).concat(".request");
	if (config.program.empty()) {
		config.program = WorkerProgramPath();
	}
	if (config.arguments_before_request.empty()) {
		config.arguments_before_request = {WORKER_COMMAND_LINE_FLAG};
	}
	config.max_outstanding = std::max(config.max_outstanding, 1u);
	request                = config.request;
	request.mode           = WorkerMode::Stream;
	request.progress_path  = Utf8FromPath(progress_path);
	request.parent_pid     = Common::CurrentProcessId();
	outstanding.reserve(config.max_outstanding);
	stats.pipe_names.reserve(config.max_restarts + 1);
	owner = std::thread([this] { Run(); });
}

// The loop of the owner thread. Each turn does one thing, and a stop ends the loop at the next
// turn, which Stop makes come at once.
void StreamSupervisor::Impl::Run() {
	for (;;) {
		switch (NextWork()) {
			case Work::Stop: Shutdown(); return;
			case Work::Connect: Connect(); break;
			case Work::Ended: EndConnection(false); break;
			case Work::Deadline: EndConnection(true); break;
			case Work::Send:
				if (const auto deadline = SendWhatFits()) {
					EndConnection(true, *deadline);
				}
				break;
			case Work::Check:
				if (child->Poll()) {
					EndConnection(false);
				}
				break;
		}
	}
}

// Waits for the next thing to do. A connected owner also wakes at the oldest deadline, and at the
// liveness interval to look at the child.
Work StreamSupervisor::Impl::NextWork() {
	std::unique_lock lock(mutex);
	for (;;) {
		if (const auto work = ReadyWorkLocked(Clock::now())) {
			return *work;
		}
		if (!connected) {
			wake.wait(lock);
			continue;
		}
		const auto until = std::min(OldestDeadlineLocked(), Clock::now() + STREAM_LIVENESS_CHECK);
		if (wake.wait_until(lock, until) == std::cv_status::timeout &&
		    !ReadyWorkLocked(Clock::now())) {
			return Work::Check;
		}
	}
}

// What is ready to be done, in the order it matters. A stop comes first, then a connection to make,
// then one that ended, then a deadline that passed, then requests to send.
std::optional<Work> StreamSupervisor::Impl::ReadyWorkLocked(Clock::time_point now) const {
	if (stopping) {
		return Work::Stop;
	}
	if (!connected) {
		return !pending.empty() && !lane_off ? std::optional(Work::Connect) : std::nullopt;
	}
	if (ConnectionEndedLocked()) {
		return Work::Ended;
	}
	if (OldestDeadlineLocked() <= now) {
		return Work::Deadline;
	}
	return CanSendLocked() ? std::optional(Work::Send) : std::nullopt;
}

// Whether the next waiting request may be sent. A suspect goes alone, so nothing is sent with it
// and it waits for the requests already out.
bool StreamSupervisor::Impl::CanSendLocked() const {
	if (pending.empty() || outstanding.size() >= config.max_outstanding) {
		return false;
	}
	if (std::ranges::any_of(outstanding, &Entry::suspect)) {
		return false;
	}
	return !pending.front().suspect || outstanding.empty();
}

// The earliest moment an outstanding request passes its deadline, or the largest time point when
// none is out.
Clock::time_point StreamSupervisor::Impl::OldestDeadlineLocked() const {
	auto oldest = Clock::time_point::max();
	for (const auto& entry: outstanding) {
		oldest = std::min(oldest, entry.sent + config.request_deadline);
	}
	return oldest;
}

// Ends the child without waiting for it to close the pipe, and gives up every request left.
void StreamSupervisor::Impl::Shutdown() {
	if (connected) {
		(void)CloseConnection(true);
	} else if (child) {
		child->Terminate();
	}
	std::vector<Entry> left;
	{
		std::lock_guard lock(mutex);
		left = TakeAllLocked();
	}
	CompleteDropped(left);
}

// Starts a child on a new pipe and waits for it to connect. A new name each time keeps a child of
// an earlier spawn, which may still be starting, from taking the new pipe.
void StreamSupervisor::Impl::Connect() {
	auto listener = StreamPipeListener::Create(STREAM_PIPE_TAG);
	if (!listener) {
		TurnLaneOff("pipe_failed");
		return;
	}
	if (!StartChild(listener->Name())) {
		return;
	}
	auto accepted = AcceptChild(*listener);
	if (accepted) {
		BeginConnection(std::move(*accepted));
	} else if (!IsStopping()) {
		FailSpawn();
	}
}

// Writes the request for this spawn and starts the child. The log offset is taken before the child
// starts, since every line it writes belongs to this spawn. A failure here is the system's and a
// restart would meet it again, so the lane turns off.
bool StreamSupervisor::Impl::StartChild(const std::string& pipe_name) {
	if (first_spawn) {
		// The files of an earlier session under the same name explain nothing about this one.
		std::error_code ignored;
		const auto out_path = fs::path(progress_path).concat(".out");
		for (const auto& stale: {progress_path, request_path, out_path}) {
			fs::remove(stale, ignored);
		}
		first_spawn = false;
	}
	request.stream_pipe = pipe_name;
	if (!WriteWorkerRequestFile(request_path, request)) {
		TurnLaneOff("request_write_failed");
		return false;
	}
	spawn_offset                       = FileSizeOrZero(progress_path);
	child                              = std::make_unique<Common::ChildProcess>();
	std::vector<std::string> arguments = config.arguments_before_request;
	arguments.push_back(Utf8FromPath(request_path));
	if (!child->Start(config.program, arguments)) {
		child.reset();
		TurnLaneOff("spawn_failed");
		return false;
	}
	std::lock_guard lock(mutex);
	stats.spawns++;
	stats.last_pid = child->Pid();
	stats.pipe_names.push_back(pipe_name);
	return true;
}

// Waits for the child to connect, in short slices, looking at the child and the stop flag between
// them. Accept cannot tell its timeout from an error it returns from at once, so Accepts that did
// not wait are counted, and too many give the spawn up before its deadline.
std::optional<StreamPipe> StreamSupervisor::Impl::AcceptChild(StreamPipeListener& listener) {
	const auto deadline = Clock::now() + config.spawn_deadline;
	uint32_t   instant  = 0;
	while (!IsStopping() && Clock::now() < deadline && instant < STREAM_ACCEPT_INSTANT_LIMIT) {
		if (child->Poll()) {
			return std::nullopt;
		}
		const auto began = Clock::now();
		auto       pipe  = listener.Accept(STREAM_ACCEPT_SLICE, child->Pid());
		if (pipe) {
			return pipe;
		}
		instant += Clock::now() - began < STREAM_ACCEPT_INSTANT ? 1 : 0;
	}
	return std::nullopt;
}

// Starts the reader on a new connection. The child wrote its priority before it connected, so the
// log has it now.
void StreamSupervisor::Impl::BeginConnection(StreamPipe accepted) {
	{
		std::lock_guard lock(pipe_mutex);
		pipe.emplace(std::move(accepted));
	}
	{
		std::lock_guard lock(mutex);
		reader_end   = ReaderEnd::Running;
		write_failed = false;
	}
	connected     = true;
	reader        = std::thread([this] { ReadAnswers(); });
	auto progress = ReadStreamSpawnProgress(progress_path, spawn_offset, {});
	if (!progress.priority.empty()) {
		std::lock_guard lock(mutex);
		stats.priority = std::move(progress.priority);
	}
}

// A child that never connected. It is ended and counted like a crash, since a restart may work.
void StreamSupervisor::Impl::FailSpawn() {
	child->Terminate();
	{
		std::lock_guard lock(mutex);
		stats.failed_spawns++;
	}
	LOGF("Prescan helper: the helper did not connect within %lld ms\n",
	     static_cast<long long>(config.spawn_deadline.count()));
	CountRestart();
}

// Sends waiting requests while they fit. A write waits for room in the pipe no longer than the
// oldest outstanding request may go unanswered, so a child that stopped reading is caught by the
// same deadline as one that stopped answering. A request blamed since it was queued is not sent.
// Returns the deadline of a write that timed out, which the caller ends as a hang no earlier than
// that deadline, or nothing when no write did.
std::optional<Clock::time_point> StreamSupervisor::Impl::SendWhatFits() {
	std::vector<StreamCompletion>    unsent;
	std::optional<Clock::time_point> timed_out;
	for (;;) {
		std::optional<Clock::time_point> deadline;
		{
			std::lock_guard lock(mutex);
			deadline = MoveNextToSendLocked(unsent);
		}
		if (!deadline) {
			break;
		}
		const auto written = pipe->WriteAllBy(send_frame, *deadline);
		if (written == StreamWriteStatus::Written) {
			continue;
		}
		if (written == StreamWriteStatus::TimedOut) {
			timed_out = deadline;
		} else {
			// The pipe is broken or interrupted, which the reader sees too. Marking it here saves
			// waiting for the reader to notice.
			std::lock_guard lock(mutex);
			write_failed = true;
		}
		break;
	}
	wake.notify_all();
	for (auto& completion: unsent) {
		Complete(std::move(completion));
	}
	return timed_out;
}

// Moves the next request that may be sent into outstanding and encodes it into the send buffer.
// The encoding happens here, under the lock, because the request lives in exactly one container at
// a time, and once it is in outstanding the reader may complete and erase it before the owner
// could read it. Requests that are blamed or do not encode are set aside for their outcomes.
// Returns the deadline of the write, or nothing when there is nothing to send.
std::optional<Clock::time_point>
StreamSupervisor::Impl::MoveNextToSendLocked(std::vector<StreamCompletion>& unsent) {
	while (!stopping && !ConnectionEndedLocked() && CanSendLocked()) {
		Entry entry = std::move(pending.front());
		pending.pop_front();
		// A backstop. The sweep after a blame normally leaves no blocked request waiting, so this
		// is not reached, and no test reaches it.
		const bool blamed = IsBlockedLocked(entry.key, entry.program_key);
		if (blamed || !EncodeStreamRequest(entry.request, *entry.code, send_frame)) {
			(blamed ? stats.blamed : stats.dropped)++;
			const auto outcome = blamed ? StreamOutcome::Blamed : StreamOutcome::Dropped;
			unsent.push_back(Unanswered(entry, outcome));
			continue;
		}
		entry.sent = Clock::now();
		outstanding.push_back(std::move(entry));
		stats.peak_outstanding =
		    std::max(stats.peak_outstanding, static_cast<uint32_t>(outstanding.size()));
		return OldestDeadlineLocked();
	}
	return std::nullopt;
}

// The reader thread. It reads until the pipe ends or a frame is bad, then tells the owner why. A
// pipe that ends inside a frame is a child that died while it wrote, so it counts as closed. A
// refused header or an answer that does not decode is a protocol error.
void StreamSupervisor::Impl::ReadAnswers() {
	StreamFrameReader frames;
	auto              end = ReaderEnd::Closed;
	for (;;) {
		const auto read = frames.ReadFrame(*pipe);
		if (read.status == StreamReadStatus::BadHeader) {
			end = ReaderEnd::ProtocolError;
			break;
		}
		if (read.status != StreamReadStatus::Frame) {
			break;
		}
		auto response = DecodeStreamResponse(read.frame);
		if (!response) {
			end = ReaderEnd::ProtocolError;
			break;
		}
		if (config.after_reply_read) {
			config.after_reply_read(*response);
		}
		Deliver(std::move(*response));
	}
	if (end == ReaderEnd::ProtocolError) {
		LOGF("Prescan helper: a frame from the helper was bad, so the connection is given up\n");
	}
	FinishReading(end);
}

// Completes the request an answer belongs to. An answer read after Stop began is destroyed, and
// its request is Dropped with the rest, so the caller sees one outcome and never a late module.
void StreamSupervisor::Impl::Deliver(StreamResponse&& response) {
	StreamCompletion completion;
	{
		std::lock_guard lock(mutex);
		if (stopping) {
			stats.late_replies++;
			return;
		}
		const auto found = std::ranges::find_if(
		    outstanding, [&](const Entry& entry) { return entry.request.seq == response.seq; });
		if (found == outstanding.end()) {
			stats.unexpected_replies++;
			return;
		}
		completion         = Unanswered(*found, StreamOutcome::Answered);
		completion.rtt_ms  = MillisecondsBetween(found->sent, Clock::now());
		auto& histogram = completion.kind == StreamRequestKind::Plan ? plan_rtt : compile_rtt;
		histogram.Add(completion.rtt_ms);
		stats.answered++;
		outstanding.erase(found);
	}
	// A slot is free for the owner to fill.
	wake.notify_all();
	completion.response = std::move(response);
	Complete(std::move(completion));
}

// The reader's last word. It records why reading stopped and wakes the owner.
void StreamSupervisor::Impl::FinishReading(ReaderEnd end) {
	{
		std::lock_guard lock(mutex);
		reader_end = end;
	}
	wake.notify_all();
}

// Whether the reader stopped for the given reason.
bool StreamSupervisor::Impl::ReaderEndIs(ReaderEnd end) const {
	std::lock_guard lock(mutex);
	return reader_end == end;
}

// Whether the connection is over, because the reader stopped or a write failed.
bool StreamSupervisor::Impl::ConnectionEndedLocked() const {
	return reader_end != ReaderEnd::Running || write_failed;
}

// Ends a connection that broke, or whose request passed its deadline, and recovers its requests.
// The kill time is never before not_before, which a write that timed out passes as its deadline,
// so the request whose deadline the write waited for always counts as overdue. A bad frame the
// reader saw first makes the close kill the child at once. A bad frame is a protocol error only
// when the child was still alive at the close, since a child that died after it wrote one crashed,
// and its death explains the frame. A stop that came meanwhile leaves the requests to Shutdown.
void StreamSupervisor::Impl::EndConnection(bool hang, Clock::time_point not_before) {
	const auto kill_time = std::max(Clock::now(), not_before);
	const auto own_exit  = CloseConnection(hang || ReaderEndIs(ReaderEnd::ProtocolError));
	if (IsStopping()) {
		return;
	}
	if (ReaderEndIs(ReaderEnd::ProtocolError) && !own_exit) {
		HandleProtocolError();
		return;
	}
	NoteFailure(hang, own_exit);
	RecoverOutstanding(hang, kill_time);
	CountRestart();
}

// Closes the connection in the order StreamPipe needs, which is to interrupt, end the child, join
// the reader, then close. A child whose pipe ended on its own gets a moment to exit, so its exit
// code can be logged, and the reader a moment to deliver the answers it left in the pipe. A child
// the supervisor kills is not waited for, nor is any child once Stop began. It is only polled, so a
// child that had already exited by itself still shows its code, and the kill's own code is never
// kept. Returns the exit code of a child that ended on its own.
std::optional<int> StreamSupervisor::Impl::CloseConnection(bool kill_now) {
	std::optional<int> own_exit;
	if (kill_now) {
		own_exit = child->Poll();
	} else if (!IsStopping()) {
		own_exit = WaitForChildExit(STREAM_EXIT_WAIT);
		WaitForReaderEnd(STREAM_EXIT_WAIT);
	}
	InterruptPipe();
	child->Terminate();
	if (reader.joinable()) {
		reader.join();
	}
	{
		std::lock_guard lock(pipe_mutex);
		pipe->Close();
		pipe.reset();
	}
	connected = false;
	return own_exit;
}

// Waits up to limit for the child to exit, in short slices that look at the stop flag, so a Stop
// that comes meanwhile is not held up by the wait.
// Returns the exit code, or nothing when the child still runs or Stop began.
std::optional<int> StreamSupervisor::Impl::WaitForChildExit(std::chrono::milliseconds limit) {
	const auto until = Clock::now() + limit;
	for (;;) {
		if (const auto code = child->Wait(STOP_CHECK_SLICE)) {
			return code;
		}
		if (IsStopping() || Clock::now() >= until) {
			return std::nullopt;
		}
	}
}

// Waits for the reader to reach the end of the pipe, or for a stop, for at most limit.
void StreamSupervisor::Impl::WaitForReaderEnd(std::chrono::milliseconds limit) {
	std::unique_lock lock(mutex);
	wake.wait_for(lock, limit, [&] { return stopping || reader_end != ReaderEnd::Running; });
}

// Counts a crash or a hang and logs it. A connection that ended while it was meant to be live is a
// crash whatever the code, since requests were left unanswered. A hung child's exit is the kill's,
// so it is not counted.
void StreamSupervisor::Impl::NoteFailure(bool hang, std::optional<int> own_exit) {
	size_t open = 0;
	{
		std::lock_guard lock(mutex);
		open = outstanding.size();
		if (hang) {
			stats.hangs++;
		} else {
			stats.crashes++;
			stats.last_crash_exit = own_exit;
		}
	}
	if (hang) {
		LOGF("Prescan helper: a request passed its deadline, so the helper was ended with %zu "
		     "requests unanswered\n",
		     open);
	} else {
		LOGF("Prescan helper: the helper's pipe ended with %zu requests unanswered, exit %s\n",
		     open, own_exit ? std::to_string(*own_exit).c_str() : "unknown");
	}
}

// A bad frame is not a crash and blames no request, since no request is known to have caused it.
// The same build's next child would send the same frame, so the lane turns off, which drops the
// requests put back here.
void StreamSupervisor::Impl::HandleProtocolError() {
	std::vector<Entry> lost;
	{
		std::lock_guard lock(mutex);
		stats.protocol_errors++;
		lost.swap(outstanding);
	}
	Requeue(std::move(lost), false);
	TurnLaneOff("protocol_error");
}

// Finds which of the dead child's requests could explain its failure and decides each one's fate.
// After a crash that is every running request, and after a hang only the running ones that were
// overdue when the child was killed, since a request sent later cannot have made it pass a
// deadline. One candidate is blamed. Other running requests become suspects, as all of them do
// when several are candidates, to be sent one at a time. With no candidate nothing is blamed and
// nothing becomes a suspect. Everything not blamed, including a request that has a `done` line and
// no answer, is sent again, except that after a blame every waiting request the blame keeps out
// completes as Blamed at once.
void StreamSupervisor::Impl::RecoverOutstanding(bool hang, Clock::time_point kill_time) {
	std::vector<Entry> lost;
	{
		std::lock_guard lock(mutex);
		lost.swap(outstanding);
	}
	if (lost.empty()) {
		return;
	}
	const auto running   = RunningFlags(lost);
	const auto candidate = [&](size_t index) {
		return running[index] &&
		       (!hang || lost[index].sent + config.request_deadline <= kill_time);
	};
	size_t candidates = 0;
	size_t culprit    = 0;
	for (size_t index = 0; index < lost.size(); index++) {
		if (candidate(index)) {
			candidates++;
			culprit = index;
		}
	}
	for (size_t index = 0; index < lost.size() && candidates > 0; index++) {
		const bool is_culprit = candidates == 1 && index == culprit;
		lost[index].suspect   = lost[index].suspect || (running[index] && !is_culprit);
	}
	std::optional<StreamCompletion> blamed;
	if (candidates == 1) {
		blamed = Blame(lost[culprit]);
		lost.erase(lost.begin() + static_cast<std::ptrdiff_t>(culprit));
	}
	Requeue(std::move(lost), true);
	// The sweep comes after the requeue, so a request of the blamed program that the dead child
	// held unstarted is caught as well as one that was waiting here.
	auto swept = blamed ? SweepBlockedPending() : std::vector<StreamCompletion> {};
	if (blamed) {
		Complete(std::move(*blamed));
	}
	for (auto& completion: swept) {
		Complete(std::move(completion));
	}
}

// Takes every waiting request whose key or program a blame has just kept out, and returns their
// Blamed outcomes for the caller to hand over outside the lock. Without it such a request would
// start a new child only to be refused at the send.
std::vector<StreamCompletion> StreamSupervisor::Impl::SweepBlockedPending() {
	std::vector<StreamCompletion> swept;
	std::lock_guard               lock(mutex);
	for (auto entry = pending.begin(); entry != pending.end();) {
		if (!IsBlockedLocked(entry->key, entry->program_key)) {
			++entry;
			continue;
		}
		swept.push_back(Unanswered(*entry, StreamOutcome::Blamed));
		stats.blamed++;
		entry = pending.erase(entry);
	}
	return swept;
}

// Which of the lost requests were running by the spawn's log, a `start` line and no `done` line.
std::vector<bool> StreamSupervisor::Impl::RunningFlags(const std::vector<Entry>& lost) const {
	std::vector<uint64_t> seqs;
	seqs.reserve(lost.size());
	for (const auto& entry: lost) {
		seqs.push_back(entry.request.seq);
	}
	const auto        progress = ReadStreamSpawnProgress(progress_path, spawn_offset, seqs);
	std::vector<bool> running;
	running.reserve(lost.size());
	for (const auto seq: seqs) {
		running.push_back(std::ranges::find(progress.started, seq) != progress.started.end() &&
		                  std::ranges::find(progress.done, seq) == progress.done.end());
	}
	return running;
}

// Puts the dead child's requests back at the head of the queue, the suspects first, each group in
// the order they were queued, so the oldest work goes first and the suspects go alone.
void StreamSupervisor::Impl::Requeue(std::vector<Entry> lost, bool count_resent) {
	if (lost.empty()) {
		return;
	}
	std::ranges::sort(lost, [](const Entry& left, const Entry& right) {
		if (left.suspect != right.suspect) {
			return left.suspect;
		}
		return left.request.seq < right.request.seq;
	});
	std::lock_guard lock(mutex);
	stats.resent += count_resent ? lost.size() : 0;
	pending.insert(pending.begin(), std::make_move_iterator(lost.begin()),
	               std::make_move_iterator(lost.end()));
}

// Blames a request that alone could explain its child's failure. Its key stays in never_retry for
// the session, so neither it nor a copy of it is sent again. A blamed Compile also keeps its
// program out, since another specialization of it would most likely fail the same way.
StreamCompletion StreamSupervisor::Impl::Blame(const Entry& entry) {
	{
		std::lock_guard lock(mutex);
		never_retry.insert(entry.key);
		if (entry.request.kind == StreamRequestKind::Compile) {
			never_retry_programs.insert(entry.program_key);
		}
		stats.never_retry          = static_cast<uint32_t>(never_retry.size());
		stats.never_retry_programs = static_cast<uint32_t>(never_retry_programs.size());
		stats.blamed++;
	}
	LOGF("Prescan helper: request %llu for program %016llx alone could explain the helper's "
	     "failure, so it is never sent again\n",
	     static_cast<unsigned long long>(entry.request.seq),
	     static_cast<unsigned long long>(entry.request.code_hash));
	return Unanswered(entry, StreamOutcome::Blamed);
}

// Counts one failure. Past max_restarts the lane turns off for the session. The child itself is
// started again only when a request waits. A failure seen while Stop runs is the stop's doing, so
// it is not counted.
void StreamSupervisor::Impl::CountRestart() {
	{
		std::lock_guard lock(mutex);
		if (stopping) {
			return;
		}
		stats.failures++;
		if (stats.failures <= config.max_restarts) {
			return;
		}
	}
	TurnLaneOff("restarts_capped");
}

// Turns the lane off for the session. Every request it holds is Dropped and Submit refuses new
// ones, so the game compiles them itself as it would without the pre-scan. Once Stop began,
// Shutdown drops the requests instead, and the lane is not marked off for a reason it never had.
void StreamSupervisor::Impl::TurnLaneOff(const std::string& reason) {
	std::vector<Entry> dropped;
	{
		std::lock_guard lock(mutex);
		if (lane_off || stopping) {
			return;
		}
		lane_off              = true;
		stats.lane_off        = true;
		stats.lane_off_reason = reason;
		dropped               = TakeAllLocked();
	}
	LOGF("Prescan helper: the lane is off for this session (%s)\n", reason.c_str());
	CompleteDropped(dropped);
}

// Why Submit refuses a request, or Queued when it does not.
StreamSubmitStatus StreamSupervisor::Impl::RefusalLocked(uint64_t key, uint64_t program_key) const {
	if (stopping) {
		return StreamSubmitStatus::Stopped;
	}
	if (lane_off) {
		return StreamSubmitStatus::LaneOff;
	}
	if (IsBlockedLocked(key, program_key)) {
		return StreamSubmitStatus::NeverRetry;
	}
	if (pending.size() >= config.max_pending) {
		return StreamSubmitStatus::QueueFull;
	}
	return StreamSubmitStatus::Queued;
}

// Whether a request's key, or its program's, was blamed.
bool StreamSupervisor::Impl::IsBlockedLocked(uint64_t key, uint64_t program_key) const {
	return never_retry.contains(key) || never_retry_programs.contains(program_key);
}

// Takes every request the supervisor holds, outstanding ones first, and counts them as dropped.
std::vector<Entry> StreamSupervisor::Impl::TakeAllLocked() {
	std::vector<Entry> taken;
	taken.swap(outstanding);
	taken.insert(taken.end(), std::make_move_iterator(pending.begin()),
	             std::make_move_iterator(pending.end()));
	pending.clear();
	stats.dropped += taken.size();
	return taken;
}

// Whether Stop has begun.
bool StreamSupervisor::Impl::IsStopping() const {
	std::lock_guard lock(mutex);
	return stopping;
}

// Hands one outcome to the callback, one at a time. The callback must not throw, and one that does
// is logged and counted, so a caller's bug cannot end the supervisor's thread and the process.
void StreamSupervisor::Impl::Complete(StreamCompletion&& completion) {
	if (!config.on_complete) {
		return;
	}
	std::lock_guard lock(callback_mutex);
	try {
		config.on_complete(std::move(completion));
	} catch (const std::exception& error) {
		NoteCallbackFailure(error.what());
	} catch (...) {
		NoteCallbackFailure("an unknown exception");
	}
}

// Logs and counts a callback that threw.
void StreamSupervisor::Impl::NoteCallbackFailure(const char* what) {
	LOGF("Prescan helper: the completion callback threw %s\n", what);
	std::lock_guard lock(mutex);
	stats.callback_failures++;
}

// Hands a Dropped outcome to the callback for each request.
void StreamSupervisor::Impl::CompleteDropped(std::vector<Entry>& entries) {
	for (const auto& entry: entries) {
		Complete(Unanswered(entry, StreamOutcome::Dropped));
	}
}

// Wakes any read or write blocked on the pipe, when there is one. Any thread may call it.
void StreamSupervisor::Impl::InterruptPipe() {
	std::lock_guard lock(pipe_mutex);
	if (pipe) {
		pipe->Interrupt();
	}
}

StreamSupervisor::StreamSupervisor(StreamSupervisorConfig config)
    : m_impl(std::make_unique<Impl>(std::move(config))) {}

StreamSupervisor::~StreamSupervisor() {
	Stop();
}

StreamSubmitResult StreamSupervisor::Submit(StreamRequest request, uint64_t tag) {
	auto code = std::make_shared<const std::vector<uint8_t>>(std::move(request.code));
	return Submit(std::move(request), std::move(code), tag);
}

StreamSubmitResult StreamSupervisor::Submit(StreamRequest                               request,
                                            std::shared_ptr<const std::vector<uint8_t>> code,
                                            uint64_t                                    tag) {
	// The shared copy is the one sent, so the request's own is let go.
	request.code                   = {};
	const auto         key         = StreamRetryKey(request);
	const auto         program_key = StreamProgramRetryKey(request);
	StreamSubmitResult result;
	{
		std::lock_guard lock(m_impl->mutex);
		result.status = m_impl->RefusalLocked(key, program_key);
		if (result.status != StreamSubmitStatus::Queued) {
			m_impl->stats.refused++;
			return result;
		}
		result.seq  = m_impl->next_seq++;
		request.seq = result.seq;
		m_impl->pending.push_back({.request     = std::move(request),
		                           .code        = code ? std::move(code) : EmptyCode(),
		                           .key         = key,
		                           .program_key = program_key,
		                           .tag         = tag});
		m_impl->stats.submitted++;
	}
	m_impl->wake.notify_all();
	return result;
}

// Sets the stop flag and interrupts the pipe here, so an owner blocked in a write wakes, then joins
// the owner, which ends the child, joins the reader and drops what is left.
void StreamSupervisor::Stop() {
	std::lock_guard stop_lock(m_impl->stop_mutex);
	{
		std::lock_guard lock(m_impl->mutex);
		m_impl->stopping = true;
	}
	m_impl->wake.notify_all();
	m_impl->InterruptPipe();
	if (m_impl->owner.joinable()) {
		m_impl->owner.join();
	}
}

bool StreamSupervisor::Stopping() const {
	return m_impl->IsStopping();
}

StreamSupervisorStats StreamSupervisor::Stats() const {
	std::lock_guard lock(m_impl->mutex);
	auto            copy = m_impl->stats;
	for (auto [histogram, summary]: {std::pair {&m_impl->plan_rtt, &copy.plan_rtt},
	                                 std::pair {&m_impl->compile_rtt, &copy.compile_rtt}}) {
		summary->count  = histogram->Count();
		summary->p50_ms = histogram->Percentile(RTT_P50);
		summary->p90_ms = histogram->Percentile(RTT_P90);
		summary->max_ms = histogram->MaxMs();
	}
	return copy;
}

} // namespace Libs::Graphics::StoreGen
