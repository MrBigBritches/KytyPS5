#ifndef GRAPHICS_SHADER_STOREGEN_STORE_SUPERVISOR_H
#define GRAPHICS_SHADER_STOREGEN_STORE_SUPERVISOR_H

#include "common/threadPriority.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
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
};

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
