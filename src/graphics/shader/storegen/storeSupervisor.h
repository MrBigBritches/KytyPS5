#ifndef GRAPHICS_SHADER_STOREGEN_STORE_SUPERVISOR_H
#define GRAPHICS_SHADER_STOREGEN_STORE_SUPERVISOR_H

#include "common/childProcess.h"
#include "common/threadPriority.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Runs a shader store worker in a child process and follows what it writes. One thread owns the
// child, the files and the decisions, and hands every record it takes from the worker's output to a
// sink. It needs no graphics device, so the pipeline cache gives it a sink that makes the modules.
namespace Libs::Graphics::StoreGen {

/// How often the supervisor looks at the child and its files. A record only matters at the next
/// shader lookup of the game, and the warm queue is minutes deep, so a slow poll costs nothing, and
/// each poll opens two files, which a real time virus scanner may charge for.
constexpr std::chrono::milliseconds GENERATION_POLL_INTERVAL {50};
/// How long a child may take to write its first progress line, which covers starting the process,
/// opening the eboot and the scan, on a loaded machine.
constexpr std::chrono::seconds GENERATION_STARTUP_TIMEOUT {180};
/// How long a child may stay on one progress line. The slowest translation of a real game takes
/// about a second.
constexpr std::chrono::seconds GENERATION_CANDIDATE_TIMEOUT {60};
/// How many times a worker is started again before generation gives up and keeps what it has.
constexpr uint32_t GENERATION_MAX_RESTARTS = 5;
/// A stop that takes the child longer than this to end is logged.
constexpr std::chrono::milliseconds GENERATION_TERMINATE_LOG_THRESHOLD {100};
/// How much of the end of the worker's text file a failure keeps. The file holds what the
/// recompiler printed, and the line that explains a crash is the last one. A real file can be
/// hundreds of kilobytes.
constexpr size_t GENERATION_OUT_TAIL_BYTES = 2048;

/// How the supervisor ended.
enum class SupervisorOutcome : uint8_t {
	Running,        ///< Not ended.
	Finished,       ///< The worker finished and every record was taken.
	Fatal,          ///< Stopped for a reason that a restart cannot mend. See inactive_reason.
	RestartsCapped, ///< Restarted GENERATION_MAX_RESTARTS times and gave up. What it got is kept.
	Stopped,        ///< Told to stop.
};

/// What ended a worker run that did not finish.
enum class FailureKind : uint8_t { Crash, Hang, OrderMismatch, BadOutput, Truncated };

/// One run of the worker that ended badly.
struct SupervisorFailure {
	/// The program the last progress line named, or zero.
	uint64_t hash = 0;
	/// The candidate rank of that line when it was a start line.
	uint64_t    rank      = 0;
	int         exit_code = 0;
	FailureKind kind      = FailureKind::Crash;
	/// The kind of the last progress line, as text, or empty when the log had none.
	std::string last_line;
	/// Which spawn ended, counting from 1 as `spawns` does.
	uint32_t spawn = 0;
	/// The processor time the worker had used when it ended, user and kernel together, when the
	/// system said. A worker that was ended for a timeout shows what it spent before that.
	std::optional<double> cpu_ms;
	/// The last GENERATION_OUT_TAIL_BYTES of the worker's text file at that time, or empty when
	/// there is none.
	std::string out_tail;
};

/// A program that a `drop` line gave up on.
struct SupervisorDrop {
	uint64_t           hash   = 0;
	uint64_t           rank   = 0;
	ProgressDropReason reason = ProgressDropReason::PushStart;
};

/// What a program's prediction came to, from its latest `prediction` line.
struct SupervisorPrediction {
	uint64_t          hash       = 0;
	PredictionOutcome outcome    = PredictionOutcome::Predicted;
	uint64_t          candidates = 0;
};

/// What one run of the worker used, once it had ended.
struct SupervisorSpawnUsage {
	/// User and kernel time together.
	double cpu_ms = 0.0;
	/// Which spawn this is, counting from 1 as `spawns` does, so a usage entry can be joined with
	/// the failure of the same spawn.
	uint32_t spawn = 0;
	/// Peak memory with the source that measured it. Missing when the system gave none, which is
	/// not zero. See Common::PeakMemorySource for what each source means.
	std::optional<Common::PeakMemory> peak;
};

/// What a supervisor did, for the tests and the report.
struct SupervisorStats {
	SupervisorOutcome outcome = SupervisorOutcome::Running;
	/// Why generation is off, when the outcome is Fatal or RestartsCapped. The names are the
	/// report's: request_write_failed, spawn_failed, output_reset_failed, restarts_capped,
	/// worker_exit_<n>, child_silent and bad_output.
	std::string inactive_reason;
	uint32_t    spawns   = 0;
	uint32_t    restarts = 0;
	/// Frames the tailer handed over, and what became of them.
	uint32_t frames_delivered = 0;
	uint32_t bad_records      = 0;
	uint32_t duplicates       = 0;
	uint32_t posted           = 0;
	/// Process id of the last child.
	uint32_t                       last_pid = 0;
	std::vector<SupervisorFailure> failures;
	/// The request of each spawn, as the worker read it.
	std::vector<WorkerRequest> requests;
	/// From the start of the thread to the first successful spawn.
	double spawn_ms = 0.0;
	/// The longest Terminate took.
	double terminate_wait_ms = 0.0;
	/// From the start of the thread to the end of the run, whatever the outcome was (Finished,
	/// Fatal, RestartsCapped or Stopped). Zero while it runs.
	double ended_ms = 0.0;
	/// The same moment, but only when the outcome is Finished, so the worker really completed.
	/// Zero for every other outcome. A comparison that cuts a run at the end of generation uses
	/// this one, and ended_ms for when the supervisor stopped doing anything.
	double finished_ms = 0.0;
	/// From the first spawn to the first `predict` line, as the supervisor's poll saw it, so it is
	/// exact to about a poll interval. Missing until a `predict` line arrives. It covers starting
	/// the process, opening the eboot and the scan.
	std::optional<double> child_open_ms;
	/// The same two moments on the steady clock, as nanoseconds since its epoch, which is the clock
	/// the first-run report's NowNs reads. first_predict_ns is when the first `predict` line was
	/// found, so it is exact to about a poll interval. finished_ns is set only for a run that
	/// finished, which is the moment from which every record the worker wrote has been posted.
	std::optional<int64_t> first_predict_ns;
	std::optional<int64_t> finished_ns;
	/// From the start of the thread to the first and the last record given to the sink.
	std::optional<double> first_record_ms;
	std::optional<double> last_record_ms;
	/// The longest one record took on the supervisor's thread: decoding it and the sink, which for
	/// the pipeline cache makes the shader modules.
	double decode_ms_max = 0.0;
	/// The program of every `predict` line, each once, over all spawns.
	std::vector<uint64_t> predicted_hashes;
	/// The latest `prediction` line of each program, over all spawns, in the order the programs
	/// were first seen. A program in predicted_hashes with no row here was cut off while it was
	/// predicted or was not reached by a worker that wrote the line.
	std::vector<SupervisorPrediction> predictions;
	/// Every `drop` line, over all spawns, in order.
	std::vector<SupervisorDrop> dropped;
	/// The programs the supervisor kept out of the next run: `failed_hashes` of the request, which
	/// keep their ranks, and `skip_hashes`, which change the ordering.
	std::vector<uint64_t> failed_hashes;
	std::vector<uint64_t> skip_hashes;
	/// The fields of the last `summary` line a worker wrote, in the order it wrote them, `exit`
	/// first. Empty when no worker finished far enough to write one.
	std::vector<std::pair<std::string, std::string>> worker_summary;
	/// One entry for each spawn whose usage the system gave, in order.
	std::vector<SupervisorSpawnUsage> spawn_usage;
	/// The largest peak memory of the spawns, among those measured by one source. Missing when no
	/// spawn had one.
	std::optional<Common::PeakMemory> peak_memory;
};

/// Whether a measured peak replaces the held one. Sources rank WorkingSet, then WorkingSetSampled,
/// then Commit, since the first is the system's own figure for resident memory and the last counts
/// memory that was never used. A better source replaces the held one whatever the sizes are, and
/// the same source replaces it only with a larger value.
[[nodiscard]] bool PeakReplaces(const Common::PeakMemory& held,
                                const Common::PeakMemory& candidate);

/// What a supervisor is made with.
struct SupervisorConfig {
	/// The request the worker gets. Its store header, host subgroup size, eboot path and generator
	/// version are used as given, and the supervisor sets the files, the restart fields and the
	/// pid.
	WorkerRequest request;
	/// The output file of the worker, which is the temporary of the generated file. The progress
	/// log, the request file and the worker's text file are next to it, and are named from it.
	std::filesystem::path output_path;
	/// The finished generated file, which only Verify mode reads.
	std::filesystem::path generated_path;
	/// The program to spawn. Empty means the running emulator, WorkerProgramPath().
	std::filesystem::path program;
	/// The command line words before the request file's path. Empty means the worker flag of the
	/// emulator. A test names its own fake worker here.
	std::vector<std::string>  arguments_before_request;
	std::chrono::milliseconds poll_interval     = GENERATION_POLL_INTERVAL;
	std::chrono::milliseconds startup_timeout   = GENERATION_STARTUP_TIMEOUT;
	std::chrono::milliseconds candidate_timeout = GENERATION_CANDIDATE_TIMEOUT;
	uint32_t                  max_restarts      = GENERATION_MAX_RESTARTS;
	/// Receives each new module record, on the supervisor's thread, after its id was checked
	/// against the ones already given. It must not stop the supervisor that calls it, since Stop
	/// joins the very thread the sink runs on.
	std::function<void(DecodedModuleRecord&&)> sink;
	/// Receives the lines to log, such as a refused priority change or a slow stop.
	std::function<void(std::string_view)> log;
	/// How the supervisor lowers its own thread. A test replaces it to make the system refuse.
	Common::PriorityStep priority_step = &Common::SystemLowerStep;
};

/// The program a worker is started from: the running emulator. Linux gives the path whose file is
/// the running image even after a rebuild replaced it, so the child is the build whose stamp is in
/// the request.
[[nodiscard]] std::filesystem::path WorkerProgramPath();

/// Follows one worker, and starts it again by the rules of the plan's restart contract.
///
/// The thread that runs it lives until the worker is done or Stop, since on Linux a child's death
/// signal follows the thread that spawned it. The supervisor lowers that thread's priority, polls
/// every poll_interval, takes the records the worker appends, blames a crash or a hang on the last
/// program the progress log named, and gives up for good when a restart cannot help.
///
/// Stop ends the child with Terminate on the supervisor's own thread, so the owner of the child is
/// always one thread. It takes no GpuStall scope, since it runs on no thread that renders.
class StoreSupervisor {
public:
	explicit StoreSupervisor(SupervisorConfig config);
	/// Stops the thread when it still runs.
	~StoreSupervisor();
	StoreSupervisor(const StoreSupervisor&)            = delete;
	StoreSupervisor& operator=(const StoreSupervisor&) = delete;

	/// Starts the thread. Does nothing after the first call.
	void Start();

	/// Asks the thread to end, ends the child, and joins. It returns within about the child's
	/// terminate time. Nothing happens when the thread is not running.
	void Stop();

	/// Waits until the outcome is no longer Running.
	/// @return whether it ended within the timeout.
	[[nodiscard]] bool WaitForEnd(std::chrono::milliseconds timeout);

	/// A copy of what was counted so far. It is final after Stop, which joins the thread. After
	/// WaitForEnd alone the thread may still be tidying up, though the outcome is set.
	[[nodiscard]] SupervisorStats Stats() const;

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_STORE_SUPERVISOR_H
