#ifndef GRAPHICS_SHADER_STOREGEN_STREAM_SUPERVISOR_H
#define GRAPHICS_SHADER_STOREGEN_STREAM_SUPERVISOR_H

#include "graphics/shader/storegen/storeWorkerFormat.h"
#include "graphics/shader/storegen/streamProtocol.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

// The parent side of the Stream worker. It starts the worker in a child process on the first
// request, sends it Plan and Compile requests through a pipe and hands each answer to a callback.
// When the child crashes or hangs it finds the requests that were running from the progress log,
// blames one only when it was the only one that can explain the failure, and sends the rest to a
// new child. After too many failures the lane is off for the session. It needs no graphics device.
// The pre-scan wires it in C11, and nothing calls it before that.
namespace Libs::Graphics::StoreGen {

/// How many requests one child may have sent and unanswered. One per translation thread keeps them
/// all busy, and one more lets the next request wait in the child rather than in the parent. More
/// would only queue in the child, where a deadline measured from the send would time the queue and
/// not the work.
constexpr uint32_t STREAM_MAX_OUTSTANDING = STREAM_TRANSLATION_THREADS + 1;
/// How long a request may go unanswered from its send before the child is taken as hung and killed.
/// The slowest translation of a real game takes about a second, so this leaves room for a loaded
/// machine and a child below normal priority while a real hang still ends within one scene. A send
/// that cannot finish because the child stopped reading is held to the same deadline.
constexpr std::chrono::seconds STREAM_REQUEST_DEADLINE {25};
/// How long a child may take from its start to its connect. The emulator binary is large and the
/// child reads its request and opens its log first, and it tries to connect for up to
/// STREAM_CONNECT_TIMEOUT itself, so this covers both on a loaded machine.
constexpr std::chrono::seconds STREAM_SPAWN_DEADLINE {15};
/// How long one Accept waits. The supervisor looks at the child and the stop flag between two, so a
/// stop or a child that died before it connected is seen within this.
constexpr std::chrono::milliseconds STREAM_ACCEPT_SLICE {50};
/// An Accept that returns sooner than this did not wait, which happens when the pipe has an error
/// the listener cannot clear. A few can be a client of another process.
constexpr std::chrono::milliseconds STREAM_ACCEPT_INSTANT {5};
/// How many Accepts in one spawn may return without waiting before the spawn is given up as failed,
/// rather than spinning until the spawn deadline.
constexpr uint32_t STREAM_ACCEPT_INSTANT_LIMIT = 20;
/// How many failures (crashes, hangs and spawns that never connected) the lane absorbs before the
/// next one turns it off for the session. It is the generator's GENERATION_MAX_RESTARTS, which the
/// source checks, since a child that keeps failing costs the game the same in both.
constexpr uint32_t STREAM_MAX_RESTARTS = 5;
/// How many requests may wait for a free slot. Each holds its program's code, tens of kilobytes,
/// and a request that waited behind hundreds of others would be answered long after the game
/// needed it.
constexpr size_t STREAM_MAX_PENDING = 256;
/// How long the supervisor waits for a child whose pipe ended on its own to exit, so its exit code
/// can be logged, and then for the reader to drain what a dead child left in the pipe. A child that
/// crashed has exited before its pipe ends.
constexpr std::chrono::milliseconds STREAM_EXIT_WAIT {500};
/// How often a connected supervisor with nothing else to do looks at whether its child is alive. A
/// dead child also ends its pipe, which wakes the supervisor at once, so this is only a backstop.
constexpr std::chrono::milliseconds STREAM_LIVENESS_CHECK {250};
/// The tag of the supervisor's pipe names.
constexpr const char* STREAM_PIPE_TAG = "prescan";

/// What became of a request that Submit queued.
enum class StreamOutcome : uint8_t {
	Answered, ///< The child answered. The response says how.
	Blamed,   ///< The child died while this request was the only one that could explain it, so it
	          ///< is never sent again in this session.
	Dropped,  ///< Given up unanswered, by Stop or because the lane turned off.
};

/// Why Submit did or did not queue a request.
enum class StreamSubmitStatus : uint8_t {
	Queued,     ///< Queued. The callback gets its outcome, once.
	Stopped,    ///< Stop has begun.
	LaneOff,    ///< The lane is off for the session, which the caller treats as Dropped.
	NeverRetry, ///< A request with the same key, or a Compile of the same program, was blamed.
	QueueFull,  ///< STREAM_MAX_PENDING requests are already waiting.
};

/// What Submit did.
struct StreamSubmitResult {
	StreamSubmitStatus status = StreamSubmitStatus::Stopped;
	/// The seq given to the request, when it was queued. Seqs start at 1.
	uint64_t seq = 0;
};

/// One request's outcome, as the callback gets it.
struct StreamCompletion {
	uint64_t          seq     = 0;
	StreamRequestKind kind    = StreamRequestKind::Plan;
	StreamOutcome     outcome = StreamOutcome::Dropped;
	/// The caller's own value from Submit, echoed unchanged. The callback can run before Submit
	/// returns the seq, so a caller matches by this.
	uint64_t tag = 0;
	/// The child's answer, when the outcome is Answered.
	StreamResponse response;
	/// From the last send of the request to its answer, when the outcome is Answered.
	double rtt_ms = 0.0;
};

/// A histogram of round trips in milliseconds, with no allocation. Below the split each millisecond
/// has a bucket, and above it each bucket is coarser, up to past the request deadline. A time past
/// the last bucket counts in it, and the exact maximum is kept apart.
class StreamRttHistogram {
public:
	/// Where the buckets stop being one millisecond wide.
	static constexpr uint32_t FINE_LIMIT_MS = 1024;
	/// The width of the coarse buckets.
	static constexpr uint32_t COARSE_WIDTH_MS = 64;
	/// Where the coarse buckets end, a little past the deadline.
	static constexpr uint32_t COARSE_LIMIT_MS = 32768;
	static constexpr size_t   BUCKETS =
	    FINE_LIMIT_MS + (COARSE_LIMIT_MS - FINE_LIMIT_MS) / COARSE_WIDTH_MS;

	/// Counts one round trip.
	void Add(double milliseconds);
	/// The time that the given fraction of round trips took no longer than, nearest rank. It is the
	/// upper bound of the bucket that holds the rank, and never more than the maximum. 0 with none.
	[[nodiscard]] double Percentile(double fraction) const;
	/// The longest round trip.
	[[nodiscard]] double MaxMs() const { return m_max_ms; }
	/// How many round trips were counted.
	[[nodiscard]] uint64_t Count() const { return m_count; }

private:
	[[nodiscard]] static size_t BucketOf(double milliseconds);
	[[nodiscard]] static double UpperBoundOf(size_t bucket);

	std::array<uint32_t, BUCKETS> m_buckets {};
	uint64_t                      m_count  = 0;
	double                        m_max_ms = 0.0;
};

/// The round trips of one request kind, for the report.
struct StreamRttSummary {
	uint64_t count  = 0;
	double   p50_ms = 0.0;
	double   p90_ms = 0.0;
	double   max_ms = 0.0;
};

/// What a stream supervisor did, for the tests and the report.
struct StreamSupervisorStats {
	/// Children started, and spawns that never connected.
	uint32_t spawns        = 0;
	uint32_t failed_spawns = 0;
	/// Connections that ended while they were meant to be live.
	uint32_t crashes = 0;
	/// Children killed because a request passed its deadline, or a send to them could not finish
	/// by it.
	uint32_t hangs = 0;
	/// Connections given up because a frame from the child was bad. One turns the lane off.
	uint32_t protocol_errors = 0;
	/// Failures counted against max_restarts. It counts crashes, hangs and failed spawns, not the
	/// children started after them, which happen only when a request waits.
	uint32_t failures = 0;
	/// Request keys blamed for a crash, which is the size of the never_retry set.
	uint32_t never_retry = 0;
	/// Programs whose Compile was blamed, so no request of theirs is sent again.
	uint32_t never_retry_programs = 0;
	/// Whether the lane is off for the session, and why. The reasons are restarts_capped,
	/// protocol_error, pipe_failed, request_write_failed and spawn_failed.
	bool        lane_off = false;
	std::string lane_off_reason;
	/// Requests queued, and their outcomes so far.
	uint64_t submitted = 0;
	uint64_t answered  = 0;
	uint64_t blamed    = 0;
	uint64_t dropped   = 0;
	/// Requests sent again to a new child.
	uint64_t resent = 0;
	/// Requests refused by Submit, for any reason.
	uint64_t refused = 0;
	/// Answers read after Stop began, which are destroyed unread.
	uint64_t late_replies = 0;
	/// Answers whose seq was not outstanding.
	uint64_t unexpected_replies = 0;
	/// Calls of on_complete that threw. The exception is logged and the outcome counts as given.
	uint64_t callback_failures = 0;
	/// The most requests one child had outstanding at once.
	uint32_t peak_outstanding = 0;
	/// The priority the child reported in its progress log, such as below_normal. Empty until a
	/// child connected.
	std::string priority;
	/// The pipe name of each spawn, in order. A failed spawn has one too. Bounded by the restarts.
	std::vector<std::string> pipe_names;
	/// The process id of the last child.
	uint32_t last_pid = 0;
	/// The exit code of the last child whose pipe ended on its own, which a crash leaves behind.
	std::optional<int> last_crash_exit;
	StreamRttSummary   plan_rtt;
	StreamRttSummary   compile_rtt;
};

/// What a stream supervisor is made with.
struct StreamSupervisorConfig {
	/// The request every child gets. Its store header and host subgroup size are used as given, and
	/// the supervisor sets the mode, the progress path, the pipe name and the parent pid.
	WorkerRequest request;
	/// Where the supervisor's files go, as a path without a suffix. The progress log is this with
	/// `.progress` added and the request file with `.request`.
	std::filesystem::path files_base;
	/// The program to start. Empty means the running emulator, WorkerProgramPath().
	std::filesystem::path program;
	/// The command line words before the request file's path. Empty means the worker flag of the
	/// emulator. A test names its own fake worker here.
	std::vector<std::string>  arguments_before_request;
	uint32_t                  max_outstanding  = STREAM_MAX_OUTSTANDING;
	size_t                    max_pending      = STREAM_MAX_PENDING;
	std::chrono::milliseconds request_deadline = STREAM_REQUEST_DEADLINE;
	std::chrono::milliseconds spawn_deadline   = STREAM_SPAWN_DEADLINE;
	uint32_t                  max_restarts     = STREAM_MAX_RESTARTS;
	/// Receives every outcome of a queued request, exactly once each. It runs on one of the
	/// supervisor's threads and never on two at once, and nothing is called after Stop returns.
	/// It must return quickly and may call Submit. Its rules are these.
	/// - It must not call Stop or destroy the supervisor.
	/// - No thread may call Stop while it holds a lock this callback takes, since Stop waits for a
	///   callback in progress.
	/// - It must not throw. An exception is caught, logged and counted in callback_failures, and
	///   the outcome counts as given.
	std::function<void(StreamCompletion&&)> on_complete;
	/// A test hook, called on the reader thread with each answer it decoded and before the answer
	/// is matched to its request. A test holds it there to make an answer arrive after Stop began.
	/// Empty in production.
	std::function<void(const StreamResponse&)> after_reply_read;
};

/// The key a blamed request is kept under. It is the kind, the code hash, the user data count and
/// the specialization, and leaves out the registers and the dispatch size on purpose. A program
/// that crashed the child would likely crash it again under other registers, and each crash spends
/// one of the few restarts of the session.
[[nodiscard]] uint64_t StreamRetryKey(const StreamRequest& request);

/// The key a program is kept under once one of its Compiles was blamed, which is the code hash and
/// the user data count. Another specialization of the same program would most likely crash the
/// child the same way, so every request of the program is refused after that.
[[nodiscard]] uint64_t StreamProgramRetryKey(const StreamRequest& request);

/// What the progress log says about some requests of one spawn.
struct StreamSpawnProgress {
	/// The seqs that have a `start` line, and those that have a `done` line, among those asked for.
	std::vector<uint64_t> started;
	std::vector<uint64_t> done;
	/// The name from the `priority` line, or empty.
	std::string priority;
};

/// Reads the progress log of one spawn, from offset to its end, for the given seqs. A last line
/// with no newline is a write the child's death tore, so it is ignored, whatever it would say.
[[nodiscard]] StreamSpawnProgress ReadStreamSpawnProgress(const std::filesystem::path& path,
                                                          uint64_t                     offset,
                                                          std::span<const uint64_t>    seqs);

/// Runs a Stream worker in a child process for the pre-scan and keeps it running.
///
/// One owner thread holds the child, starts it on the first request, sends the requests as the one
/// writer, and enforces the deadlines, including on a send that a child which stopped reading
/// cannot take. One reader thread per connection drains and decodes the answers and hands each to
/// the callback straight away, so an answer never waits for the owner. At most max_outstanding
/// requests are sent and unanswered. Past that they wait in the supervisor, up to max_pending, and
/// Submit refuses more, so the caller is never blocked.
///
/// When the pipe ends while the connection was meant to be live, or a request passes its deadline
/// and the child is killed, the supervisor reads the spawn's progress log. A request with a `start`
/// line and no `done` line was running. After a crash every running request is a candidate for
/// the blame, and after a hang only the running requests that were overdue are. When exactly one
/// is, it is blamed. It completes as Blamed and its key goes into a never_retry set for the
/// session, with its program's key when it is a Compile. Other running requests become suspects,
/// and so do all of them when several are candidates. The new child gets the suspects one at a
/// time, so a request is blamed only when it fails while it is the only one that can explain the
/// failure. Every other request unanswered by the dead child, including one that has a `done` line
/// and no answer, is sent again. A bad frame from a child that is still alive is no crash and
/// blames nothing. It turns the lane off, since the next child would send the same. A bad frame
/// from a child that had died by the close is a crash, which the death explains. A write that
/// times out ends the child as a hang no earlier than the deadline it waited for, so the request
/// that set that deadline counts as overdue. The exit code of a child the
/// supervisor ended itself is never counted. After max_restarts failures the next one turns the
/// lane off for the session, and every waiting and future request is Dropped or refused.
///
/// Stop is prompt with a hung child. It never waits for the child to close the pipe, since on
/// Windows the child closes it only after it has joined its threads.
class StreamSupervisor {
public:
	/// Starts the owner thread, which starts no child until the first request.
	explicit StreamSupervisor(StreamSupervisorConfig config);
	/// Stops when Stop was not called.
	~StreamSupervisor();
	StreamSupervisor(const StreamSupervisor&)            = delete;
	StreamSupervisor& operator=(const StreamSupervisor&) = delete;

	/// Queues a request with its own code, which moves into a copy the supervisor shares. Its seq
	/// is replaced by the supervisor's own. Any thread may call it.
	/// @param tag any value, echoed in the request's completion.
	[[nodiscard]] StreamSubmitResult Submit(StreamRequest request, uint64_t tag = 0);

	/// Queues a request whose code is shared, so a Plan and the Compiles of one program hold one
	/// copy. The request's own code is not read.
	/// @param code the program's code. Null is taken as empty.
	/// @param tag  any value, echoed in the request's completion.
	[[nodiscard]] StreamSubmitResult Submit(StreamRequest                               request,
	                                        std::shared_ptr<const std::vector<uint8_t>> code,
	                                        uint64_t                                    tag = 0);

	/// Kills the child, completes every request still queued or outstanding as Dropped and joins
	/// the threads. Any thread but the callback's may call it, any number of times.
	void Stop();

	/// Whether Stop has begun.
	[[nodiscard]] bool Stopping() const;

	/// A copy of what was counted so far. It is final after Stop.
	[[nodiscard]] StreamSupervisorStats Stats() const;

	/// The state of a supervisor. Defined in the source file.
	struct Impl;

private:
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_STREAM_SUPERVISOR_H
