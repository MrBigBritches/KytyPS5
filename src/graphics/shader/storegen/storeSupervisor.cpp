#include "graphics/shader/storegen/storeSupervisor.h"

#include "common/childProcess.h"
#include "common/file.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/storegen/frameTailer.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/workerProcess.h"

#include <algorithm>
#include <condition_variable>
#include <fmt/format.h>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <unordered_set>
#include <vector>

namespace Libs::Graphics::StoreGen {

namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

namespace {

// How many polls in a row may read nothing before the poll after a child's exit gives up waiting
// for the tailer to settle.
constexpr int STUCK_POLL_LIMIT = 3;

constexpr int EXIT_CODE_DONE = 0;
// Exit codes of the worker that need no restart, because no restart would change them.
constexpr int EXIT_CODE_ORDER_MISMATCH = WORKER_EXIT_ORDER_MISMATCH;

// What a worker run ended in and what the supervisor does about it.
struct Verdict {
	enum class Action : uint8_t { Finish, Fatal, Fresh, Resume };
	Action      action = Action::Fatal;
	std::string reason;
	FailureKind kind = FailureKind::Crash;
	// Whether the last line's program is blamed although the run starts over. It is for a worker
	// that crashed and left a bad file, where the crash is still the program's.
	bool blame = false;
};

// The failure kind of a tailer status that ended a run.
FailureKind KindOfStatus(TailStatus status) {
	return status == TailStatus::Truncated ? FailureKind::Truncated : FailureKind::BadOutput;
}

// The name of a progress line's kind, for the failures a report lists.
const char* KindName(ProgressKind kind) {
	switch (kind) {
		case ProgressKind::Predict: return "predict";
		case ProgressKind::Order: return "order";
		case ProgressKind::Start: return "start";
		case ProgressKind::Drop: return "drop";
		case ProgressKind::Summary: return "summary";
	}
	return "";
}

// Adds a hash to a list that holds each once.
void AddUnique(std::vector<uint64_t>& list, uint64_t hash) {
	if (std::ranges::find(list, hash) == list.end()) {
		list.push_back(hash);
	}
}

} // namespace

std::filesystem::path WorkerProgramPath() {
#if defined(__linux__)
	return Common::ExecutableStatPath();
#else
	return Common::ExecutablePath();
#endif
}

// Everything one supervisor owns. The thread's own state is touched by that thread alone, and the
// statistics are shared under the mutex.
struct StoreSupervisor::Impl {
	explicit Impl(SupervisorConfig supplied);

	// ---- The thread. ----
	void               Run(std::stop_token token);
	bool               Begin();
	void               Iterate();
	void               WaitInterval();
	[[nodiscard]] bool Running() const;
	void               LowerPriority();

	// ---- Following the child. ----
	bool               Spawn();
	void               Deliver(std::vector<TailedFrame>&& frames);
	void               DeliverRecord(const TailedFrame& frame);
	void               CatchUp();
	[[nodiscard]] bool TimedOut(ProgressClock::time_point now) const;
	std::optional<int> TerminateChild();

	// ---- Deciding. ----
	void                  Conclude(int exit_code, bool killed);
	[[nodiscard]] Verdict Judge(int exit_code, bool killed) const;
	void                  Restart(const Verdict& verdict);
	void                  BlameLastLine();
	void                  PlanResume();
	void                  AbsorbDrops();
	bool                  PrepareFresh();
	void                  RecordFailure(const Verdict& verdict, int exit_code);

	// ---- The shared state. ----
	template <typename Change>
	void Update(Change&& change) {
		std::scoped_lock lock(mutex);
		change(stats);
	}
	void Log(const std::string& text) const {
		if (config.log) {
			config.log(text);
		}
	}
	void End(SupervisorOutcome outcome, const std::string& reason);

	SupervisorConfig        config;
	fs::path                progress_path;
	fs::path                request_path;
	mutable std::mutex      mutex;
	std::condition_variable end_changed;
	SupervisorStats         stats;
	std::jthread            thread;
	bool                    started = false;

	// Owned by the thread.
	std::stop_token                       stop;
	std::unique_ptr<Common::ChildProcess> child;
	std::unique_ptr<FrameTailer>          tailer;
	std::unique_ptr<FileTailPoller>       poller;
	ProgressLogReader                     progress;
	WorkerRequest                         request;
	std::unordered_set<uint64_t>          seen_ids;
	Clock::time_point                     thread_start;
	Clock::time_point                     spawn_time;
};

StoreSupervisor::Impl::Impl(SupervisorConfig supplied)
    : config(std::move(supplied)),
      progress_path(fs::absolute(config.output_path).concat(".progress")),
      request_path(fs::absolute(config.output_path).concat(".request")), progress(progress_path) {
	config.output_path = fs::absolute(config.output_path);
	if (config.program.empty()) {
		config.program = WorkerProgramPath();
	}
	if (config.arguments_before_request.empty()) {
		config.arguments_before_request = {WORKER_COMMAND_LINE_FLAG};
	}
	request                = config.request;
	request.output_path    = Utf8FromPath(config.output_path);
	request.progress_path  = Utf8FromPath(progress_path);
	request.eboot_path     = Utf8FromPath(fs::absolute(PathFromUtf8(request.eboot_path)));
	request.generated_path = config.generated_path.empty()
	                             ? std::string()
	                             : Utf8FromPath(fs::absolute(config.generated_path));
	request.parent_pid     = Common::CurrentProcessId();
	request.mode           = WorkerMode::Generate;
}

void StoreSupervisor::Impl::End(SupervisorOutcome outcome, const std::string& reason) {
	{
		std::scoped_lock lock(mutex);
		if (stats.outcome != SupervisorOutcome::Running) {
			return;
		}
		stats.outcome         = outcome;
		stats.inactive_reason = reason;
	}
	end_changed.notify_all();
}

bool StoreSupervisor::Impl::Running() const {
	std::scoped_lock lock(mutex);
	return stats.outcome == SupervisorOutcome::Running;
}

// Lowers this thread below the guest's. The helper does not log, so a refusal is logged here, once,
// since it is asked once.
void StoreSupervisor::Impl::LowerPriority() {
	Common::PriorityFailure failure;
	if (!Common::LowerCurrentThreadPriorityWith(config.priority_step, &failure) &&
	    failure.api != nullptr) {
		Log(fmt::format("Shader store generation: {} failed with {}, the supervisor keeps normal "
		                "priority",
		                failure.api, failure.code));
	}
}

// Sleeps until the next poll, and wakes at once for a stop.
void StoreSupervisor::Impl::WaitInterval() {
	std::mutex                  idle;
	std::condition_variable_any wake;
	std::unique_lock            lock(idle);
	wake.wait_for(lock, stop, config.poll_interval, [] { return false; });
}

void StoreSupervisor::Impl::Run(std::stop_token token) {
	stop         = token;
	thread_start = Clock::now();
	LowerPriority();
	if (Begin()) {
		while (!stop.stop_requested() && Running()) {
			WaitInterval();
			if (stop.stop_requested()) {
				break;
			}
			Iterate();
		}
	}
	// A stop ends the child here, on the thread that owns it.
	TerminateChild();
	End(SupervisorOutcome::Stopped, "");
	// Destroyed on this thread, which is the one that spawned it.
	child.reset();
}

// Clears what an earlier process with the same id may have left, makes the tailer and starts the
// first worker. Returns false when that failed, and the outcome says why.
bool StoreSupervisor::Impl::Begin() {
	Common::File::CreateDirectories(config.output_path.parent_path());
	std::error_code ignored;
	for (const auto& stale: {request_path, progress_path, config.output_path,
	                         fs::path(progress_path).concat(".out")}) {
		fs::remove(stale, ignored);
	}
	tailer = std::make_unique<FrameTailer>(request.store_header, request.generator_version,
	                                       std::nullopt);
	poller = std::make_unique<FileTailPoller>(config.output_path, *tailer);
	return Spawn();
}

// Writes the request and starts the worker. The progress log's spawn offset is taken before the
// child exists, since a line it writes at once must count.
bool StoreSupervisor::Impl::Spawn() {
	if (!WriteWorkerRequestFile(request_path, request)) {
		End(SupervisorOutcome::Fatal, "request_write_failed");
		return false;
	}
	const auto now = Clock::now();
	progress.BeginSpawn(now);
	child                              = std::make_unique<Common::ChildProcess>();
	std::vector<std::string> arguments = config.arguments_before_request;
	arguments.push_back(Utf8FromPath(request_path));
	if (!child->Start(config.program, arguments)) {
		End(SupervisorOutcome::Fatal, "spawn_failed");
		return false;
	}
	spawn_time = Clock::now();
	Update([&](SupervisorStats& s) {
		if (s.spawns == 0) {
			s.spawn_ms =
			    std::chrono::duration<double, std::milli>(spawn_time - thread_start).count();
		}
		s.spawns++;
		s.last_pid = child->Pid();
		s.requests.push_back(request);
	});
	return true;
}

// Whether the child has been silent for too long. Before its first line it has the startup time,
// and after that the time of one candidate.
bool StoreSupervisor::Impl::TimedOut(ProgressClock::time_point now) const {
	if (progress.LinesSinceSpawn() == 0) {
		return now - spawn_time > config.startup_timeout;
	}
	return now - progress.LastLineTime() > config.candidate_timeout;
}

// Ends the child and says how it ended. The wait is timed, and a slow one is logged.
std::optional<int> StoreSupervisor::Impl::TerminateChild() {
	if (!child) {
		return std::nullopt;
	}
	const auto begin = Clock::now();
	child->Terminate();
	const auto waited = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
	Update(
	    [&](SupervisorStats& s) { s.terminate_wait_ms = std::max(s.terminate_wait_ms, waited); });
	if (waited > static_cast<double>(GENERATION_TERMINATE_LOG_THRESHOLD.count())) {
		Log(fmt::format("Shader store generation: ending the worker took {:.0f} ms", waited));
	}
	return child->Poll();
}

// One look at the child and its files, and the decision that follows from it.
void StoreSupervisor::Impl::Iterate() {
	// Read before the files, since bytes written just before the exit must still be read after it.
	const auto exited = child->Poll();
	const auto now    = ProgressClock::now();
	progress.Poll(now);
	Deliver(poller->Poll(exited.has_value()));
	if (exited) {
		Conclude(*exited, false);
		return;
	}
	const auto status = tailer->Status();
	if (status != TailStatus::Waiting) {
		// The child lives and its output is bad for good.
		const auto code = TerminateChild();
		Conclude(code.value_or(0), true);
		return;
	}
	if (TimedOut(now)) {
		const auto code = TerminateChild();
		Conclude(code.value_or(0), true);
	}
}

// Reads what a dead child left: the rest of the progress log, and the output until the tailer has
// a final answer. A blame is only fair on all of it.
void StoreSupervisor::Impl::CatchUp() {
	progress.PollToEnd(ProgressClock::now());
	int stuck = 0;
	while (!tailer->IsFinal() && stuck < STUCK_POLL_LIMIT && !stop.stop_requested()) {
		const auto before  = poller->BytesRead();
		auto       frames  = poller->Poll(true);
		const bool nothing = frames.empty() && poller->BytesRead() == before;
		Deliver(std::move(frames));
		stuck = nothing ? stuck + 1 : 0;
	}
}

void StoreSupervisor::Impl::Deliver(std::vector<TailedFrame>&& frames) {
	for (const auto& frame: frames) {
		DeliverRecord(frame);
	}
}

// Decodes one frame, and gives its module to the sink unless an earlier record had its id. A fresh
// restart sends the records again, so the ids are what makes them arrive once.
void StoreSupervisor::Impl::DeliverRecord(const TailedFrame& frame) {
	using ShaderRecompiler::IR::IoAll;
	using ShaderRecompiler::IR::Reader;
	Update([](SupervisorStats& s) { s.frames_delivered++; });
	Reader reader {frame.raw};
	auto   type = StoreRecord::GraphicsPipeline;
	IoAll(reader, type);
	DecodedModuleRecord decoded;
	if (reader.ok && type == StoreRecord::Module) {
		DecodeModuleRecord(reader, decoded);
	}
	if (!reader.ok || !reader.bytes.empty() || type != StoreRecord::Module ||
	    decoded.stage != ShaderType::Compute) {
		Update([](SupervisorStats& s) { s.bad_records++; });
		return;
	}
	if (!seen_ids.insert(decoded.id).second) {
		Update([](SupervisorStats& s) { s.duplicates++; });
		return;
	}
	if (config.sink) {
		config.sink(std::move(decoded));
	}
	Update([](SupervisorStats& s) { s.posted++; });
}

// What the end of a worker run means, by the plan's restart contract.
Verdict StoreSupervisor::Impl::Judge(int exit_code, bool killed) const {
	using Action = Verdict::Action;
	auto status  = tailer->Status();
	// A tailer that is still waiting after the child is gone has a tail that will not complete.
	if (status == TailStatus::Waiting) {
		status = TailStatus::DamagedTail;
	}
	if (!killed && exit_code == EXIT_CODE_DONE) {
		if (status == TailStatus::Complete) {
			return {Action::Finish, "", FailureKind::Crash};
		}
		if (status != TailStatus::HeaderError) {
			return {Action::Fresh, "", FailureKind::BadOutput};
		}
	}
	// Verify mode is not asked for yet, so exit 10 is no answer to anything. The others mean the
	// request, the eboot or the disk is wrong, which another run would meet again.
	if (!killed &&
	    (exit_code == WORKER_EXIT_UP_TO_DATE ||
	     (exit_code >= WORKER_EXIT_BAD_REQUEST && exit_code <= WORKER_EXIT_OUTPUT_FAILED))) {
		return {Action::Fatal, fmt::format("worker_exit_{}", exit_code), FailureKind::Crash};
	}
	if (status == TailStatus::HeaderError) {
		return {Action::Fatal, "bad_output", FailureKind::BadOutput};
	}
	if (!killed && exit_code == EXIT_CODE_ORDER_MISMATCH) {
		return {Action::Fresh, "", FailureKind::OrderMismatch};
	}
	if (status == TailStatus::Truncated || status == TailStatus::DamagedFrame ||
	    status == TailStatus::FrameTooLarge) {
		// A worker that crashed (it was not killed, and did not end by itself with 0 or 14) and
		// left a bad file is blamed all the same, or the fresh run would crash at the same program.
		const bool crashed =
		    !killed && exit_code != EXIT_CODE_DONE && exit_code != EXIT_CODE_ORDER_MISMATCH;
		return {Action::Fresh, "", KindOfStatus(status), crashed};
	}
	// A crash or a hang, blamed on what the log's last line names.
	if (progress.LinesSinceSpawn() == 0) {
		return {Action::Fatal, "child_silent", FailureKind::Crash};
	}
	return {Action::Resume, "", killed ? FailureKind::Hang : FailureKind::Crash};
}

void StoreSupervisor::Impl::Conclude(int exit_code, bool killed) {
	CatchUp();
	const auto verdict = Judge(exit_code, killed);
	if (verdict.action == Verdict::Action::Finish) {
		End(SupervisorOutcome::Finished, "");
		return;
	}
	RecordFailure(verdict, exit_code);
	if (verdict.action == Verdict::Action::Fatal) {
		End(SupervisorOutcome::Fatal, verdict.reason);
		return;
	}
	// A stop that came while the log was read ends it here. A worker started now would be spawned
	// after the stop, and the thread would end with it running.
	if (stop.stop_requested()) {
		return;
	}
	Restart(verdict);
}

void StoreSupervisor::Impl::RecordFailure(const Verdict& verdict, int exit_code) {
	SupervisorFailure failure;
	failure.exit_code = exit_code;
	failure.kind      = verdict.kind;
	if (const auto& last = progress.LastLine()) {
		failure.hash      = last->hash;
		failure.rank      = last->kind == ProgressKind::Start ? last->rank_or_count : 0;
		failure.last_line = KindName(last->kind);
	}
	Update([&](SupervisorStats& s) { s.failures.push_back(failure); });
}

// Every program that a `drop` line gave up on stays out of the next run, except for the drop of a
// record that was only too big, which belongs to that one record and not to its program.
void StoreSupervisor::Impl::AbsorbDrops() {
	for (const auto& drop: progress.DropsSinceSpawn()) {
		if (drop.reason != ProgressDropReason::FrameTooLarge) {
			AddUnique(request.failed_hashes, drop.hash);
		}
	}
}

// Puts the blame of a crash or a hang on the program of the last line: a program that crashed
// while it was predicted is skipped, which changes the ordering, and one that crashed while a
// candidate was built is failed, which keeps every rank. The run goes on after the last start line.
void StoreSupervisor::Impl::BlameLastLine() {
	if (const auto& last = progress.LastLine()) {
		if (last->kind == ProgressKind::Predict) {
			AddUnique(request.skip_hashes, last->hash);
		} else if (last->kind == ProgressKind::Start) {
			AddUnique(request.failed_hashes, last->hash);
		}
	}
}

// Where a resumed run goes on: after the last start line, with the ordering the log last gave.
void StoreSupervisor::Impl::PlanResume() {
	request.start_rank     = progress.AnyStartSeen() ? progress.LastStart()->rank_or_count + 1 : 0;
	request.expected_order = progress.LatestOrder() ? progress.LatestOrder()->hash : 0;
}

// Cuts the output file, which the old run may have left, and starts the tailer and the logs over.
// The cut comes first: the new poller must never see the old file, or the worker making a new one
// would look like a file cut back. Returns false when the file could not be cut.
bool StoreSupervisor::Impl::PrepareFresh() {
	std::error_code error;
	fs::remove(config.output_path, error);
	if (fs::exists(config.output_path, error)) {
		std::error_code resize_error;
		fs::resize_file(config.output_path, 0, resize_error);
		if (resize_error) {
			End(SupervisorOutcome::Fatal, "output_reset_failed");
			return false;
		}
	}
	poller.reset();
	tailer = std::make_unique<FrameTailer>(request.store_header, request.generator_version,
	                                       std::nullopt);
	poller = std::make_unique<FileTailPoller>(config.output_path, *tailer);
	progress.BeginFreshRun(ProgressClock::now());
	request.kept           = 0;
	request.start_rank     = 0;
	request.expected_order = 0;
	return true;
}

// Starts the worker again, from where it was or from nothing, unless the cap is reached.
void StoreSupervisor::Impl::Restart(const Verdict& verdict) {
	AbsorbDrops();
	if (verdict.action == Verdict::Action::Resume) {
		BlameLastLine();
		PlanResume();
	} else if (verdict.blame) {
		BlameLastLine();
	}
	if (stats.restarts >= config.max_restarts) {
		End(SupervisorOutcome::RestartsCapped, "restarts_capped");
		return;
	}
	Update([](SupervisorStats& s) { s.restarts++; });
	if (verdict.action == Verdict::Action::Fresh) {
		if (!PrepareFresh()) {
			return;
		}
	} else {
		poller->RestartFrom();
		request.kept = tailer->GoodOffset();
	}
	Spawn();
}

StoreSupervisor::StoreSupervisor(SupervisorConfig config)
    : m_impl(std::make_unique<Impl>(std::move(config))) {}

StoreSupervisor::~StoreSupervisor() {
	Stop();
}

void StoreSupervisor::Start() {
	if (m_impl->started) {
		return;
	}
	m_impl->started = true;
	m_impl->thread =
	    std::jthread([impl = m_impl.get()](std::stop_token token) { impl->Run(token); });
}

void StoreSupervisor::Stop() {
	if (!m_impl->thread.joinable()) {
		return;
	}
	m_impl->thread.request_stop();
	m_impl->thread.join();
}

bool StoreSupervisor::WaitForEnd(std::chrono::milliseconds timeout) {
	std::unique_lock lock(m_impl->mutex);
	return m_impl->end_changed.wait_for(
	    lock, timeout, [&] { return m_impl->stats.outcome != SupervisorOutcome::Running; });
}

SupervisorStats StoreSupervisor::Stats() const {
	std::scoped_lock lock(m_impl->mutex);
	return m_impl->stats;
}

} // namespace Libs::Graphics::StoreGen
