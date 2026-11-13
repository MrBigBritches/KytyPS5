#include "graphics/shader/storegen/storeSupervisor.h"

#include "common/childProcess.h"
#include "common/file.h"
#include "common/firstRunReport.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/storegen/frameTailer.h"
#include "graphics/shader/storegen/generatedOrder.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/workerProcess.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <condition_variable>
#include <deque>
#include <fmt/format.h>
#include <fstream>
#include <mutex>
#include <optional>
#include <stop_token>
#include <thread>
#include <unordered_map>
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
	// Verified is a Verify worker's exit 10, which says the generated file is current. It is not
	// a finish yet, since the file has to be loaded whole as well.
	enum class Action : uint8_t { Finish, Verified, Fatal, Fresh, Resume };
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
		case ProgressKind::Prediction: return "prediction";
		case ProgressKind::Done: return "done";
		case ProgressKind::Priority: return "priority";
		case ProgressKind::Summary: return "summary";
	}
	return "";
}

// The last max_bytes of a file as it is, or nothing when it cannot be read. Only the end is read,
// since the file can be large and the line that explains a crash is the last one.
std::string ReadFileTail(const fs::path& path, size_t max_bytes) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	if (!file) {
		return {};
	}
	const auto  size  = static_cast<uint64_t>(file.tellg());
	const auto  count = std::min<uint64_t>(size, max_bytes);
	std::string tail(static_cast<size_t>(count), '\0');
	file.seekg(static_cast<std::streamoff>(size - count));
	file.read(tail.data(), static_cast<std::streamsize>(count));
	tail.resize(static_cast<size_t>(std::max<std::streamsize>(file.gcount(), 0)));
	return tail;
}

// How good a peak memory source is. Higher is better.
int PeakSourceRank(Common::PeakMemorySource source) {
	switch (source) {
		case Common::PeakMemorySource::WorkingSet: return 2;
		case Common::PeakMemorySource::WorkingSetSampled: return 1;
		case Common::PeakMemorySource::Commit: return 0;
	}
	return -1;
}

// Milliseconds between two times of the steady clock.
double MillisecondsBetween(Clock::time_point from, Clock::time_point to) {
	return std::chrono::duration<double, std::milli>(to - from).count();
}

// A time of the steady clock as nanoseconds since its epoch, as the first-run report's NowNs reads.
int64_t NanosecondsOf(Clock::time_point time) {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

// Adds a hash to a list that holds each once.
void AddUnique(std::vector<uint64_t>& list, uint64_t hash) {
	if (std::ranges::find(list, hash) == list.end()) {
		list.push_back(hash);
	}
}

// Streams the generated file of an earlier run, one read at a time. The file is complete, since
// only a finished run renames its output to it, so nothing writes to it and a bad item in it is
// final. It is opened for each read and closed again, so nothing holds it when a finish replaces
// it.
class GeneratedFileLoader {
public:
	enum class Phase : uint8_t {
		Loading,  // More frames are queued or still to be read.
		Complete, // Every frame was handed over and the file ended after one.
		Damaged,  // The file is bad from some point on. What came before it was handed over.
	};

	GeneratedFileLoader(std::vector<uint8_t> store_header, uint32_t generator_version,
	                    const fs::path& path, uint64_t read_bytes)
	    : m_tailer(std::move(store_header), generator_version, std::nullopt),
	      m_poller(path, m_tailer, read_bytes) {}
	GeneratedFileLoader(const GeneratedFileLoader&)            = delete;
	GeneratedFileLoader& operator=(const GeneratedFileLoader&) = delete;

	// Whether the file's header is the expected one, by the stamp and the generator version. The
	// eboot identity is not asked, since the worker's Verify answers that.
	[[nodiscard]] bool HeaderAccepted() const { return m_tailer.Identity().has_value(); }

	[[nodiscard]] Phase GetPhase() const {
		if (!m_pending.empty()) {
			return Phase::Loading;
		}
		if (m_stalled) {
			return Phase::Damaged;
		}
		switch (m_tailer.Status()) {
			case TailStatus::Waiting: return Phase::Loading;
			case TailStatus::Complete: return Phase::Complete;
			default: return Phase::Damaged;
		}
	}

	// Whether the next frame needs a read of the file first.
	[[nodiscard]] bool NeedsRead() const {
		return m_pending.empty() && GetPhase() == Phase::Loading;
	}

	// Reads the next part of the file and queues the frames that became complete. A bad item is
	// held once and read again, which judges it. One that is read again from the same good frame
	// twice is larger than a read, which no read can judge, so the file is given up as damaged.
	void ReadSlice() {
		const auto rewinds_before = m_poller.Rewinds();
		for (auto& frame: m_poller.Poll(true)) {
			m_pending.push_back(std::move(frame));
		}
		if (m_poller.Rewinds() == rewinds_before) {
			return;
		}
		const auto good      = m_tailer.GoodOffset();
		m_rewinds_at_good    = good == m_rewind_good_offset ? m_rewinds_at_good + 1 : 1;
		m_rewind_good_offset = good;
		if (m_rewinds_at_good >= GENERATION_LOADER_REWIND_LIMIT) {
			m_stalled = true;
		}
	}

	// The next queued frame, or nothing.
	std::optional<TailedFrame> TakeFrame() {
		if (m_pending.empty()) {
			return std::nullopt;
		}
		auto frame = std::move(m_pending.front());
		m_pending.pop_front();
		return frame;
	}

private:
	FrameTailer             m_tailer;
	FileTailPoller          m_poller;
	std::deque<TailedFrame> m_pending;
	uint64_t                m_rewind_good_offset = 0;
	uint32_t                m_rewinds_at_good    = 0;
	bool                    m_stalled            = false;
};

// What follows the title in the name of a temporary file, then the process id, then one of the
// suffixes. The temporary file itself and the three helper files of its run.
constexpr std::string_view                GENERATED_INFIX   = ".generated.";
constexpr std::array<std::string_view, 4> RUN_FILE_SUFFIXES = {".tmp", ".tmp.progress",
                                                               ".tmp.request", ".tmp.progress.out"};

} // namespace

std::optional<uint32_t> StaleFilePid(std::string_view name, std::string_view title) {
	// The title and the infix are matched as plain text, so a title with dots or other characters
	// that a pattern would read specially is still taken literally.
	if (!name.starts_with(title)) {
		return std::nullopt;
	}
	name.remove_prefix(title.size());
	if (!name.starts_with(GENERATED_INFIX)) {
		return std::nullopt;
	}
	name.remove_prefix(GENERATED_INFIX.size());
	uint32_t   pid    = 0;
	const auto digits = std::from_chars(name.data(), name.data() + name.size(), pid);
	// No digits, a sign or a number too large for an id is not one of ours. from_chars reads no
	// sign for an unsigned type, so a leading minus gives no digits.
	if (digits.ec != std::errc() || digits.ptr == name.data()) {
		return std::nullopt;
	}
	const std::string_view rest(digits.ptr,
	                            static_cast<size_t>(name.data() + name.size() - digits.ptr));
	if (std::ranges::find(RUN_FILE_SUFFIXES, rest) == RUN_FILE_SUFFIXES.end()) {
		return std::nullopt;
	}
	return pid;
}

StaleCleanup RemoveStaleGenerationFiles(const fs::path& folder, std::string_view title,
                                        uint32_t                                     current_pid,
                                        const std::function<bool(uint32_t)>&         is_running,
                                        const std::function<void(std::string_view)>& log) {
	return RemoveDeadProcessFiles(
	    folder, current_pid,
	    [title](std::string_view name) { return StaleFilePid(name, title); }, is_running, log,
	    "Shader store generation");
}

StaleCleanup
RemoveDeadProcessFiles(const fs::path& folder, uint32_t current_pid,
                       const std::function<std::optional<uint32_t>(std::string_view)>& pid_of,
                       const std::function<bool(uint32_t)>&                            is_running,
                       const std::function<void(std::string_view)>&                    log,
                       std::string_view                                                who) {
	StaleCleanup      result;
	std::error_code   error;
	const std::string prefix = std::string(who) + ": ";
	// A plain directory_iterator reads the folder itself and never descends, so a matching name in
	// a subfolder is out of reach.
	for (fs::directory_iterator it(folder, error), end; !error && it != end; it.increment(error)) {
		const auto name = Utf8FromPath(it->path().filename());
		const auto pid  = pid_of(name);
		if (!pid || *pid == current_pid || is_running(*pid)) {
			continue;
		}
		// symlink_status looks at the link itself, so a link is never a regular file here and is
		// kept, as is a directory with a matching name.
		std::error_code status_error;
		const auto      status = fs::symlink_status(it->path(), status_error);
		if (status_error || !fs::is_regular_file(status)) {
			continue;
		}
		std::error_code remove_error;
		if (fs::remove(it->path(), remove_error) && !remove_error) {
			result.removed++;
			if (log) {
				log(prefix + "removed the stale file " + name);
			}
		} else {
			result.failed++;
			if (log) {
				log(prefix + "could not remove the stale file " + name + ", " +
				    remove_error.message());
			}
		}
	}
	return result;
}

bool PeakReplaces(const Common::PeakMemory& held, const Common::PeakMemory& candidate) {
	if (candidate.source != held.source) {
		return PeakSourceRank(candidate.source) > PeakSourceRank(held.source);
	}
	return candidate.bytes > held.bytes;
}

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
	bool Spawn();
	void Deliver(std::vector<TailedFrame>&& frames);
	// Whether a record came from the worker's output or from the earlier generated file.
	enum class RecordSource : uint8_t { Worker, Loaded };
	void                         DeliverRecord(const TailedFrame& frame, RecordSource source);
	void                         HandleRecord(const TailedFrame& frame, RecordSource source);
	[[nodiscard]] GeneratedOrder OrderOf(uint64_t program_hash, RecordSource source);
	void                         CatchUp();
	[[nodiscard]] bool           TimedOut(ProgressClock::time_point now) const;
	std::optional<int>           TerminateChild();

	// ---- Loading the generated file of an earlier run. ----
	void               OpenLoader();
	void               AdvanceLoader(std::chrono::milliseconds budget);
	[[nodiscard]] bool LoaderLoading() const;
	void               SettleVerified();
	void               RespawnAfterDamage();

	// ---- Deciding. ----
	void                  Conclude(int exit_code, bool killed);
	void                  Finish(bool verified_current);
	void                  EndLoader();
	GeneratedPersistence  Persist();
	void                  RemoveStaleFiles();
	void                  RemoveOwnHelperFiles();
	void                  NoteExistingFile();
	void                  NoteSource(bool verified_current);
	[[nodiscard]] Verdict Judge(int exit_code, bool killed) const;
	void                  Restart(const Verdict& verdict);
	void                  BlameLastLine();
	void                  PlanResume();
	void                  AbsorbDrops();
	bool                  PrepareFresh();
	void                  RecordFailure(const Verdict& verdict, int exit_code);

	// ---- What a spawn leaves in the statistics. ----
	void NoteProgress();
	void CollectSpawnStats();
	void CollectDrops();
	void CollectSummary();
	void RecordUsage();
	void PublishHashLists();

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
	// The earlier generated file being loaded. Empty when there is none, when its header was not
	// the expected one, and after a finish gave up on it.
	std::unique_ptr<GeneratedFileLoader> loader;
	// Whether the header of the earlier file was the expected one. It stays set when the loader is
	// gone, since the source of the run depends on it.
	bool header_accepted = false;
	// Whether a Verify worker said the file is current and the supervisor waits for it to load.
	bool verified = false;
	// Set before a worker is started for a damaged file, and checked first, so it happens once. It
	// cannot be reached today, since RespawnAfterDamage and Restart both set Generate, nothing sets
	// Verify again, and Judge gives Verified only in Verify mode. It stays as a backstop against a
	// respawn loop if a later change lets a restart verify again.
	bool                         regenerated_after_damage = false;
	ProgressLogReader            progress;
	WorkerRequest                request;
	std::unordered_set<uint64_t> seen_ids;
	// How many records of each program were delivered. Like seen_ids it is never cleared, so a
	// fresh restart that sends records again does not give them a lower tier.
	std::unordered_map<uint64_t, uint32_t> delivered_per_program;
	Clock::time_point                      thread_start;
	Clock::time_point                      spawn_time;
	Clock::time_point                      first_spawn_time;
	// What the statistics of the current spawn have taken already, so ending a spawn in two places
	// counts it once.
	bool   stats_collected     = false;
	size_t predicted_published = 0;
	// The prediction rows last copied into the statistics, to see when they change.
	std::vector<ProgressPrediction> predictions_published;
	std::optional<double>           last_cpu_ms;
};

StoreSupervisor::Impl::Impl(SupervisorConfig supplied)
    : config(std::move(supplied)),
      progress_path(fs::absolute(config.output_path).concat(".progress")),
      request_path(fs::absolute(config.output_path).concat(".request")), progress(progress_path) {
	config.output_path = fs::absolute(config.output_path);
	if (!config.generated_path.empty()) {
		config.generated_path = fs::absolute(config.generated_path);
	}
	if (!config.process_running) {
		config.process_running = &Common::ProcessIsRunning;
	}
	if (!config.rename_file) {
		config.rename_file = [](const fs::path& from, const fs::path& to) {
			return Common::File::RenameFile(from, to);
		};
	}
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
	Common::FirstRunReport::SetGeneratorChildAlive(false);
	{
		std::scoped_lock lock(mutex);
		if (stats.outcome != SupervisorOutcome::Running) {
			return;
		}
		stats.outcome         = outcome;
		stats.inactive_reason = reason;
		const auto now        = Clock::now();
		stats.ended_ms        = MillisecondsBetween(thread_start, now);
		if (outcome == SupervisorOutcome::Finished) {
			stats.finished_ms = stats.ended_ms;
			stats.finished_ns = NanosecondsOf(now);
		}
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
	// A stop ends the child here, on the thread that owns it. What the spawn used and wrote is
	// counted after it ends, unless a verdict has counted it already.
	TerminateChild();
	CollectSpawnStats();
	End(SupervisorOutcome::Stopped, "");
	// Destroyed on this thread, which is the one that spawned it.
	child.reset();
}

// Clears what an earlier process with the same id may have left, makes the tailer and starts the
// first worker. Returns false when that failed, and the outcome says why.
bool StoreSupervisor::Impl::Begin() {
	Common::File::CreateDirectories(config.output_path.parent_path());
	RemoveStaleFiles();
	std::error_code ignored;
	for (const auto& stale: {request_path, progress_path, config.output_path,
	                         fs::path(progress_path).concat(".out")}) {
		fs::remove(stale, ignored);
	}
	tailer = std::make_unique<FrameTailer>(request.store_header, request.generator_version,
	                                       std::nullopt);
	poller = std::make_unique<FileTailPoller>(config.output_path, *tailer);
	OpenLoader();
	return Spawn();
}

// Starts loading the generated file of an earlier run, when there is one. The first reads go on
// until the header is judged, since that decides what the first worker is asked to do. A header
// that is not the expected one, by the stamp or the version, makes the file ignored: the worker
// then generates, as it does when there is no file. Otherwise the worker runs in Verify mode while
// the records load.
void StoreSupervisor::Impl::OpenLoader() {
	std::error_code error;
	if (config.generated_path.empty() || !fs::is_regular_file(config.generated_path, error)) {
		return;
	}
	loader = std::make_unique<GeneratedFileLoader>(request.store_header, request.generator_version,
	                                               config.generated_path, config.loader_read_bytes);
	while (!loader->HeaderAccepted() && loader->GetPhase() == GeneratedFileLoader::Phase::Loading) {
		loader->ReadSlice();
	}
	if (!loader->HeaderAccepted()) {
		Log("Shader store generation: the generated file of an earlier run is not for this build, "
		    "so it is ignored");
		loader.reset();
		return;
	}
	header_accepted = true;
	request.mode    = WorkerMode::Verify;
	// Provisional, so a run that is stopped during or after the load still says it loaded. The
	// finish refines it to loaded_regenerated when the worker writes a new file.
	Update([](SupervisorStats& s) { s.source = GenerationSource::Loaded; });
}

bool StoreSupervisor::Impl::LoaderLoading() const {
	return loader && loader->GetPhase() == GeneratedFileLoader::Phase::Loading;
}

// Gives the load of the earlier file about one slice of work: it reads when no frame is queued and
// hands frames on, through the same decode, id check and sink as the worker's records, and stops
// at the first record that ends the slice. At least one record goes on each call, so a slice that
// is shorter than a record still makes progress.
void StoreSupervisor::Impl::AdvanceLoader(std::chrono::milliseconds budget) {
	if (!loader) {
		return;
	}
	const auto deadline = Clock::now() + budget;
	do {
		if (loader->NeedsRead()) {
			loader->ReadSlice();
		}
		if (const auto frame = loader->TakeFrame()) {
			DeliverRecord(*frame, RecordSource::Loaded);
		}
	} while (LoaderLoading() && Clock::now() < deadline && !stop.stop_requested());
}

// What follows a Verify worker's exit 10 once the earlier file has loaded as far as it can. The
// worker has said that the file is current, which is only worth anything if it loaded whole.
void StoreSupervisor::Impl::SettleVerified() {
	AdvanceLoader(config.loader_slice);
	if (LoaderLoading()) {
		return;
	}
	const bool whole = loader && loader->GetPhase() == GeneratedFileLoader::Phase::Complete;
	switch (DecideAfterVerify(whole ? LoaderEnd::Complete : LoaderEnd::Damaged,
	                          regenerated_after_damage)) {
		case VerifiedAction::Finish: Finish(true); break;
		case VerifiedAction::Respawn: RespawnAfterDamage(); break;
		case VerifiedAction::Fatal: End(SupervisorOutcome::Fatal, "bad_output"); break;
	}
}

// The file the worker called current is damaged, so it is not trusted. A worker is started again in
// Generate mode, which writes a new file. It is not a restart, since nothing failed. The flag is
// set first, so that nothing can bring the supervisor here twice.
void StoreSupervisor::Impl::RespawnAfterDamage() {
	regenerated_after_damage = true;
	Update([](SupervisorStats& s) { s.regenerated_after_damage = true; });
	Log("Shader store generation: the generated file of an earlier run is damaged, so it is "
	    "generated again");
	verified     = false;
	request.mode = WorkerMode::Generate;
	if (PrepareFresh()) {
		Spawn();
	}
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
	spawn_time      = Clock::now();
	stats_collected = false;
	last_cpu_ms.reset();
	Update([&](SupervisorStats& s) {
		if (s.spawns == 0) {
			first_spawn_time = spawn_time;
			s.spawn_ms       = MillisecondsBetween(thread_start, spawn_time);
		}
		s.spawns++;
		s.last_pid = child->Pid();
		s.requests.push_back(request);
	});
	// The rows of worker compiles in the report show whether this process was running.
	Common::FirstRunReport::SetGeneratorChildAlive(true);
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
	// A worker that said the file is current has exited, and only the load is left to wait for.
	if (verified) {
		SettleVerified();
		return;
	}
	AdvanceLoader(config.loader_slice);
	// Read before the files, since bytes written just before the exit must still be read after it.
	const auto exited = child->Poll();
	const auto now    = ProgressClock::now();
	// The progress log is polled before the records, and a record's candidate count is looked up in
	// the log when the record is delivered. The worker writes all its prediction lines before it
	// opens the output, so a poll that reads a record almost always read its line too. A line
	// written between the two reads is missed by the records that follow it, which then count as
	// not wide.
	progress.Poll(now);
	NoteProgress();
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
	NoteProgress();
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
		DeliverRecord(frame, RecordSource::Worker);
	}
}

// Decodes one frame, and gives its module to the sink unless an earlier record had its id. A fresh
// restart sends the records again, so the ids are what makes them arrive once.
void StoreSupervisor::Impl::DeliverRecord(const TailedFrame& frame, RecordSource source) {
	const auto begin = Clock::now();
	HandleRecord(frame, source);
	const auto end = Clock::now();
	Update([&](SupervisorStats& s) {
		s.decode_ms_max = std::max(s.decode_ms_max, MillisecondsBetween(begin, end));
	});
}

// Where a new record stands among its program's, and how many candidates the program has. It runs
// after the id check, so a record the restart sent again was dropped before it could count. The
// counter is kept across fresh restarts for the same reason as the ids. A record of the earlier
// file has no prediction line of this run, so it has no count and is never in the Back lane.
GeneratedOrder StoreSupervisor::Impl::OrderOf(uint64_t program_hash, RecordSource source) {
	GeneratedOrder order;
	order.tier = delivered_per_program[program_hash]++;
	if (source == RecordSource::Worker) {
		order.candidates = progress.CandidatesOf(program_hash);
	}
	order.wide_tail = IsBackLane(order.tier, order.candidates);
	return order;
}

// The work of DeliverRecord, which it times.
void StoreSupervisor::Impl::HandleRecord(const TailedFrame& frame, RecordSource source) {
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
	const auto order = OrderOf(decoded.hash, source);
	if (config.sink) {
		config.sink(std::move(decoded), order);
	}
	const auto since_start = MillisecondsBetween(thread_start, Clock::now());
	Update([&](SupervisorStats& s) {
		s.posted++;
		if (!s.first_record_ms) {
			s.first_record_ms = since_start;
		}
		s.last_record_ms = since_start;
	});
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
	// Exit 10 is Verify mode's answer, and only for a worker that was asked to verify a file whose
	// header was accepted. From a Generate worker it means something is wrong, like the other
	// codes below, which say the request, the eboot or the disk is wrong. Another run would meet
	// that again.
	if (!killed && exit_code == WORKER_EXIT_UP_TO_DATE && request.mode == WorkerMode::Verify) {
		return {Action::Verified, "", FailureKind::Crash};
	}
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
	const auto concluded = Clock::now();
	CatchUp();
	CollectSpawnStats();
	const auto verdict = Judge(exit_code, killed);
	if (verdict.action == Verdict::Action::Finish) {
		Finish(false);
		return;
	}
	if (verdict.action == Verdict::Action::Verified) {
		Update(
		    [&](SupervisorStats& s) { s.verify_ms = MillisecondsBetween(spawn_time, concluded); });
		verified = true;
		SettleVerified();
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

// Finishes a run whose worker exited with 0 and left a complete file, in the order that keeps the
// old generated file safe. The loader is ended first, so nothing still reads a file the rename
// replaces. The rename comes next, since only a worker that has exited has closed its file. The
// outcome is set last, so a reader that sees Finished also sees what became of the file. A stop
// that comes while the loader ends renames nothing, and the thread ends as Stopped. A run whose
// worker found the earlier file current has written no file, so there is nothing to rename and the
// earlier file is what persists. The helper files of the run are deleted after a rename that worked
// and after such a loaded finish, and are kept after a failed rename.
void StoreSupervisor::Impl::Finish(bool verified_current) {
	EndLoader();
	if (stop.stop_requested()) {
		return;
	}
	auto persisted = GeneratedPersistence::Existing;
	if (verified_current) {
		NoteExistingFile();
	} else {
		persisted = Persist();
	}
	NoteSource(verified_current);
	// A failed rename keeps the helper files with the temporary file, which is what explains it,
	// and so does a run that renamed nothing because it had no generated path.
	if (persisted == GeneratedPersistence::Renamed || persisted == GeneratedPersistence::Existing) {
		RemoveOwnHelperFiles();
	}
	End(SupervisorOutcome::Finished, "");
}

// The loader step of a finish. The load of the earlier file is allowed to reach its end for a
// bounded time, since the generation that just ended took far longer, and is abandoned after that.
// Slices follow each other with no wait between them, and a stop ends the wait at once. The
// configured hook runs after.
void StoreSupervisor::Impl::EndLoader() {
	if (LoaderLoading()) {
		const auto deadline = Clock::now() + config.loader_end_wait;
		while (LoaderLoading() && !stop.stop_requested() && Clock::now() < deadline) {
			AdvanceLoader(config.loader_slice);
		}
		if (LoaderLoading() && !stop.stop_requested()) {
			Update([](SupervisorStats& s) { s.loader_abandoned = true; });
			Log("Shader store generation: the load of the earlier generated file was given up, "
			    "since it did not end in time");
			// Nothing reads the file after this, so the rename may replace it.
			loader.reset();
		}
	}
	if (config.end_loader) {
		config.end_loader(stop, config.loader_end_wait);
	}
}

// The size of the generated file on disk, or zero when it cannot be read.
static uint64_t GeneratedFileSize(const fs::path& path) {
	std::error_code error;
	const auto      size = fs::file_size(path, error);
	return error ? 0 : static_cast<uint64_t>(size);
}

// Records that the worker found the earlier file current and left it as it was.
void StoreSupervisor::Impl::NoteExistingFile() {
	const auto bytes = GeneratedFileSize(config.generated_path);
	Update([&](SupervisorStats& s) {
		s.persisted  = GeneratedPersistence::Existing;
		s.file_bytes = bytes;
	});
}

// Says where the modules came from, which a finish knows: a worker that verified the file loaded
// it, one that wrote a new file after the earlier file's header was accepted regenerated it, and
// any other run generated.
void StoreSupervisor::Impl::NoteSource(bool verified_current) {
	const auto source = verified_current  ? GenerationSource::Loaded
	                    : header_accepted ? GenerationSource::LoadedRegenerated
	                                      : GenerationSource::Generated;
	Update([&](SupervisorStats& s) { s.source = source; });
}

// Renames the finished temporary file to the generated file and records the result. A failure keeps
// the temporary file where it is, so the next boot finds nothing to load and generates again. An
// empty generated_path (only a test leaves it so) renames nothing, and the run still ends Finished
// with persisted left at NotFinished. Returns what persisted became.
GeneratedPersistence StoreSupervisor::Impl::Persist() {
	if (config.generated_path.empty()) {
		return GeneratedPersistence::NotFinished;
	}
	std::string error;
	if (!config.rename_file(config.output_path, config.generated_path)) {
		// File::RenameFile deletes the destination first, so a failure can leave neither the old
		// generated file nor the new one, and the report shows which of them exist.
		std::error_code ignored;
		error = fmt::format("rename of {} to {} failed, source {}, destination {}",
		                    Utf8FromPath(config.output_path), Utf8FromPath(config.generated_path),
		                    fs::exists(config.output_path, ignored) ? "kept" : "missing",
		                    fs::exists(config.generated_path, ignored) ? "present" : "absent");
		Log("Shader store generation: " + error);
	}
	// A failed rename leaves no new generated file, so the size is only taken after a good one.
	const auto bytes = error.empty() ? GeneratedFileSize(config.generated_path) : 0;
	const auto result =
	    error.empty() ? GeneratedPersistence::Renamed : GeneratedPersistence::RenameFailed;
	Update([&](SupervisorStats& s) {
		s.persisted     = result;
		s.persist_error = std::move(error);
		s.file_bytes    = bytes;
	});
	return result;
}

// Deletes the files of dead runs of this title that sit in the store's folder, once, before the
// first worker is started and before the generated file is read. It does nothing without a
// generated path, which is where the title is taken from.
void StoreSupervisor::Impl::RemoveStaleFiles() {
	constexpr std::string_view GENERATED_SUFFIX = ".generated";
	const auto                 name             = Utf8FromPath(config.generated_path.filename());
	if (config.generated_path.empty() || !name.ends_with(GENERATED_SUFFIX)) {
		return;
	}
	const auto title   = std::string_view(name).substr(0, name.size() - GENERATED_SUFFIX.size());
	const auto cleaned = RemoveStaleGenerationFiles(
	    config.output_path.parent_path(), title, Common::CurrentProcessId(), config.process_running,
	    [this](std::string_view line) { Log(std::string(line)); });
	Update([&](SupervisorStats& s) {
		s.files_removed += cleaned.removed;
		s.remove_failures += cleaned.failed;
	});
}

// Deletes this run's progress log, request file and worker text file once the run has succeeded,
// since they only explain a run that went wrong. The worker has exited by now, so nothing holds the
// text file, but a delete that still fails (a scanner, or a permission) is logged by name and
// counted, and the file is left for the next boot to remove as the file of a dead process.
void StoreSupervisor::Impl::RemoveOwnHelperFiles() {
	const fs::path helpers[] = {progress_path, request_path,
	                            fs::path(progress_path).concat(".out")};
	for (const auto& helper: helpers) {
		std::error_code error;
		if (!fs::exists(helper, error)) {
			continue;
		}
		if (fs::remove(helper, error) && !error) {
			Update([](SupervisorStats& s) { s.files_removed++; });
			continue;
		}
		Update([](SupervisorStats& s) { s.remove_failures++; });
		Log("Shader store generation: could not remove " + Utf8FromPath(helper) + ", " +
		    error.message());
	}
}

void StoreSupervisor::Impl::RecordFailure(const Verdict& verdict, int exit_code) {
	SupervisorFailure failure;
	failure.exit_code = exit_code;
	failure.kind      = verdict.kind;
	if (const auto& last = progress.LastLine()) {
		// A crash after a program's `prediction` line and before `order` is in no program, so it
		// names none. BlameLastLine ignores that line for the same reason.
		failure.hash      = last->kind == ProgressKind::Prediction ? 0 : last->hash;
		failure.rank      = last->kind == ProgressKind::Start ? last->rank_or_count : 0;
		failure.last_line = KindName(last->kind);
	}
	failure.cpu_ms = last_cpu_ms;
	failure.out_tail =
	    ReadFileTail(fs::path(progress_path).concat(".out"), GENERATION_OUT_TAIL_BYTES);
	Update([&](SupervisorStats& s) {
		failure.spawn = s.spawns;
		s.failures.push_back(failure);
	});
}

// Takes what a spawn leaves into the statistics after it ended: the program of each `predict` line,
// the drops, and what the process used. It does it once for each spawn, and only after the child
// has ended, since the usage is only there then.
void StoreSupervisor::Impl::CollectSpawnStats() {
	if (stats_collected || !child) {
		return;
	}
	stats_collected = true;
	progress.PollToEnd(ProgressClock::now());
	NoteProgress();
	CollectDrops();
	CollectSummary();
	RecordUsage();
}

// Publishes the predicted programs, and times the first `predict` line against the first spawn.
void StoreSupervisor::Impl::NoteProgress() {
	const auto& predicted = progress.PredictedHashes();
	const auto  first     = progress.FirstPredictTime();
	// A program predicted again changes a row and not the count, so the rows are compared whole.
	const bool predictions_changed = progress.Predictions() != predictions_published;
	if (predicted.size() == predicted_published && !first && !predictions_changed) {
		return;
	}
	Update([&](SupervisorStats& s) {
		if (predicted.size() != predicted_published) {
			s.predicted_hashes = predicted;
		}
		if (predictions_changed) {
			s.predictions.clear();
			for (const auto& row: progress.Predictions()) {
				s.predictions.push_back({row.hash, row.outcome, row.candidates});
			}
		}
		if (first && !s.child_open_ms) {
			s.child_open_ms    = MillisecondsBetween(first_spawn_time, *first);
			s.first_predict_ns = NanosecondsOf(*first);
		}
	});
	predicted_published   = predicted.size();
	predictions_published = progress.Predictions();
}

// Adds the drops of this spawn to the list of all of them.
void StoreSupervisor::Impl::CollectDrops() {
	const auto& drops = progress.DropsSinceSpawn();
	if (drops.empty()) {
		return;
	}
	Update([&](SupervisorStats& s) {
		for (const auto& drop: drops) {
			s.dropped.push_back({.hash      = drop.hash,
			                     .rank      = drop.rank_or_count,
			                     .reason    = drop.reason,
			                     .module_id = drop.module_id});
		}
	});
}

// Keeps the fields of the summary line of this spawn, if it wrote one. A later spawn's summary
// replaces an earlier one, since only the last run is the one that ended the generation.
void StoreSupervisor::Impl::CollectSummary() {
	const auto& summary = progress.Summary();
	if (!summary) {
		return;
	}
	Update([&](SupervisorStats& s) { s.worker_summary = summary->fields; });
}

// Keeps what the ended child used. A figure the system did not give is left out, not zeroed.
void StoreSupervisor::Impl::RecordUsage() {
	const auto usage = child->Usage();
	if (!usage) {
		return;
	}
	SupervisorSpawnUsage spawn;
	spawn.cpu_ms = usage->cpu_user_ms + usage->cpu_kernel_ms;
	spawn.peak   = usage->peak;
	last_cpu_ms  = spawn.cpu_ms;
	Update([&](SupervisorStats& s) {
		spawn.spawn = s.spawns;
		s.spawn_usage.push_back(spawn);
		if (spawn.peak && (!s.peak_memory || PeakReplaces(*s.peak_memory, *spawn.peak))) {
			s.peak_memory = spawn.peak;
		}
	});
}

// Copies the programs the next run leaves out into the statistics.
void StoreSupervisor::Impl::PublishHashLists() {
	Update([&](SupervisorStats& s) {
		s.failed_hashes = request.failed_hashes;
		s.skip_hashes   = request.skip_hashes;
	});
}

// Every program that a `drop` line gave up on stays out of the next run, except for the drop of a
// record that was only too big, which belongs to that one record and not to its program. The drop
// of a head with a wrong push start is the same, since the program's other candidates are built.
void StoreSupervisor::Impl::AbsorbDrops() {
	for (const auto& drop: progress.DropsSinceSpawn()) {
		if (drop.reason != ProgressDropReason::FrameTooLarge &&
		    drop.reason != ProgressDropReason::PushStartHead) {
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
	// Only the first spawn verifies. A worker that started again has failed once, and the file it
	// would find current is not the question any more.
	request.mode = WorkerMode::Generate;
	AbsorbDrops();
	if (verdict.action == Verdict::Action::Resume) {
		BlameLastLine();
		PlanResume();
	} else if (verdict.blame) {
		BlameLastLine();
	}
	PublishHashLists();
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
