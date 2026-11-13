#ifndef GRAPHICS_SHADER_STOREGEN_PRESCAN_TRANSPORT_H
#define GRAPHICS_SHADER_STOREGEN_PRESCAN_TRANSPORT_H

#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/computeTranslation.h"
#include "graphics/shader/storegen/storeSupervisor.h"
#include "graphics/shader/storegen/streamProtocol.h"
#include "graphics/shader/storegen/streamSupervisor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <vector>

// How the pre-scan's preparer reaches whatever translates and compiles its programs. The preparer
// sends Plan and Compile requests and gets each outcome back through a callback, whatever does the
// work. The real one is the Stream worker in a child process, behind the StreamSupervisor, so a
// crash in the recompiler never reaches the emulator. Tests use the in-process one, which runs the
// worker's own answer on the caller's thread.
namespace Libs::Graphics::StoreGen {

/// Receives every outcome of a request the transport queued, exactly once. It must only queue the
/// outcome and wake its owner. The supervisor's request deadline includes the time its reader spends
/// in the callback, so slow work here would make a healthy child look hung.
using PrescanCompletionHandler = std::function<void(StreamCompletion&&)>;

/// Sends the pre-scan's requests. The preparer is its only caller.
class PrescanTransport {
public:
	virtual ~PrescanTransport() = default;

	/// Queues a request whose code is shared, so a program's Plan and Compiles hold one copy and
	/// the caller keeps its own. The request's own code is not read. The outcome may reach the
	/// handler before this returns, so the caller matches outcomes by `tag` and not by the seq.
	///
	/// It never blocks and never waits for the helper. It only queues the request, or refuses it,
	/// and returns. The preparer's Stop joins the preparer's thread before it stops the transport,
	/// so a Submit that waited for a hung helper would hold Stop up for ever. The in-process
	/// transport is the one exception, since it answers on the caller's thread, and it exists only
	/// for tests.
	/// @return Queued, or why the request was refused. A refused request gets no outcome.
	[[nodiscard]] virtual StreamSubmitResult
	Submit(StreamRequest request, std::shared_ptr<const std::vector<uint8_t>> code, uint64_t tag) = 0;

	/// Gives up every request still queued, which completes as Dropped, and refuses new ones. No
	/// outcome reaches the handler after it returns. The caller must not hold a lock the handler
	/// takes, since this may wait for a handler in progress. The preparer calls it from its own
	/// thread when it fails, or after that thread is joined, so it never runs beside a Submit.
	/// Calling it again does nothing.
	virtual void Stop() = 0;
};

/// Makes a transport that sends its outcomes to the handler. The preparer makes its transport with
/// it, since the handler belongs to the preparer and the supervisor takes it at construction.
/// @param host   the host values translation depends on, from the running device and Config.
/// @param device the host subgroup size and the shared memory limit, from the same device.
using PrescanTransportFactory = std::function<std::unique_ptr<PrescanTransport>(
    const HostValues& host, const ComputeKeyDevice& device, PrescanCompletionHandler handler)>;

/// Answers each request with AnswerStreamRequest on the thread that submits it, and hands the
/// outcome to the handler before Submit returns. It exists for tests, which need the preparer's
/// whole path without a child process, and it never runs in the emulator, since a recompiler EXIT
/// would end the process. A test can hold Compile requests back, which stands for a child that has
/// not answered yet.
class InProcessPrescanTransport final: public PrescanTransport {
public:
	InProcessPrescanTransport(const HostValues& host, const ComputeKeyDevice& device,
	                          PrescanCompletionHandler handler);

	[[nodiscard]] StreamSubmitResult Submit(StreamRequest                               request,
	                                        std::shared_ptr<const std::vector<uint8_t>> code,
	                                        uint64_t                                    tag) override;
	void                             Stop() override;

	/// While set, Compile requests are kept unanswered until ReleaseHeld. Plans are still answered.
	void SetHoldCompiles(bool hold);
	/// How many Compile requests are held.
	[[nodiscard]] size_t Held() const;
	/// Answers the held requests on the calling thread, in the order they came.
	void ReleaseHeld();
	/// How many requests of each kind were queued, answered or held.
	[[nodiscard]] uint64_t Plans() const { return m_plans.load(std::memory_order_relaxed); }
	[[nodiscard]] uint64_t Compiles() const { return m_compiles.load(std::memory_order_relaxed); }

private:
	// A request kept back, with what its outcome needs.
	struct HeldRequest {
		StreamRequest request;
		uint64_t      tag = 0;
	};

	// Answers one request and hands the outcome to the handler.
	void Answer(const StreamRequest& request, uint64_t tag);

	HostValues               m_host;
	ComputeKeyDevice         m_device {};
	PrescanCompletionHandler m_handler;
	mutable std::mutex       m_mutex;
	std::deque<HeldRequest>  m_held;
	uint64_t                 m_next_seq = 1;
	bool                     m_hold     = false;
	bool                     m_stopped  = false;
	std::atomic<uint64_t>    m_plans {0};
	std::atomic<uint64_t>    m_compiles {0};
};

/// A factory for InProcessPrescanTransport. When `made` is given it receives a pointer to the
/// transport, which lives as long as the preparer that made it, so a test can hold its Compiles.
[[nodiscard]] PrescanTransportFactory
InProcessPrescanTransportFactory(InProcessPrescanTransport** made = nullptr);

/// Sends the pre-scan's requests to the Stream worker in a child process, through a
/// StreamSupervisor, which is what a real run uses. A crash or a hang in the recompiler then costs
/// the helper and never the emulator.
///
/// The supervisor's outcomes and refusals are the transport's own, so each passes through as it
/// is. The outcomes are Answered, Blamed and Dropped, and Submit gives Queued, Stopped, LaneOff,
/// NeverRetry or QueueFull. The preparer gives them their meaning, which is that Blamed and
/// NeverRetry retire the program with no fallback, LaneOff turns the pre-scan off for the session,
/// and the rest leave the module to the GPU thread.
///
/// Submit only queues, as the supervisor's Submit does, so it never waits for the child. The
/// handler is the supervisor's callback, so it runs on a supervisor thread and must only queue and
/// wake. Stop is the supervisor's, so it may be called any number of times, and no outcome reaches
/// the handler after the first call returns.
///
/// The first Stop also removes the supervisor's request file, progress log and the child's text
/// output, which only explain a session that went wrong. They are kept, and their place is logged,
/// when a child failed, a request was blamed or the lane turned off, and the next run's start-up
/// cleanup removes them.
class SupervisorPrescanTransport final: public PrescanTransport {
public:
	/// Starts the supervisor, which starts no child until the first request.
	/// @param config  the supervisor's settings. Its request's store header, host subgroup size and
	///                on_complete are replaced with ones made from host, device and handler.
	/// @param host    the host values translation depends on, from the running device and Config.
	/// @param device  the host subgroup size and the shared memory limit, from the same device.
	/// @param handler gets every outcome.
	/// @throws std::invalid_argument when device's shared memory limit is not host's. The child
	///         takes its limit from the header's host values, so it would build other modules.
	SupervisorPrescanTransport(StreamSupervisorConfig config, const HostValues& host,
	                           const ComputeKeyDevice& device, PrescanCompletionHandler handler);
	/// Stops, which also settles the files.
	~SupervisorPrescanTransport() override;
	SupervisorPrescanTransport(const SupervisorPrescanTransport&)            = delete;
	SupervisorPrescanTransport& operator=(const SupervisorPrescanTransport&) = delete;

	[[nodiscard]] StreamSubmitResult Submit(StreamRequest                               request,
	                                        std::shared_ptr<const std::vector<uint8_t>> code,
	                                        uint64_t                                    tag) override;
	void                             Stop() override;

	/// What the supervisor counted so far. It is final after Stop.
	[[nodiscard]] StreamSupervisorStats Stats() const { return m_supervisor.Stats(); }

private:
	// Removes the files, or keeps them and logs where they are, once the child is gone.
	void SettleFiles() noexcept;

	// Made before the supervisor, from the same files_base, which the supervisor's config takes.
	StreamSupervisorFiles m_files;
	StreamSupervisor      m_supervisor;
	std::once_flag        m_files_settled;
};

/// A directory for the helper's files that no other user can write to, and that holds nothing but
/// those files, so the start-up cleanup never looks at anything else. The request file names the
/// pipe the child connects to and where it writes, so another user who could rewrite it could
/// steer the child. It is always a `kyty_prescan` directory. On Windows that is in the user's own
/// temporary directory. On POSIX it is in XDG_RUNTIME_DIR, or else in `kyty-<uid>` in the
/// temporary directory. Each POSIX directory is made with mode 0700 and used only when it is a real
/// directory, not a link, owned by this user and closed to everyone else.
/// @return nothing when no such directory can be had, and the pre-scan then stays off.
[[nodiscard]] std::optional<std::filesystem::path> PrescanHelperDirectory();

/// The files_base of the helper of process `pid`, `kyty_prescan_<pid>` in the directory.
[[nodiscard]] std::filesystem::path PrescanHelperFilesBase(const std::filesystem::path& directory,
                                                           uint32_t                     pid);

/// The process id in the name of one of the helper's files, `kyty_prescan_<pid>` followed by
/// `.request`, `.progress` or `.progress.out`. The whole name has to match.
[[nodiscard]] std::optional<uint32_t> PrescanHelperFilePid(std::string_view name);

/// Removes the helper files that dead runs left in the directory, as RemoveDeadProcessFiles does.
/// @param is_running tells whether a process id belongs to a running process. Production passes
///                   Common::ProcessIsRunning, and a test passes its own so it needs no dead pid.
StaleCleanup RemoveStalePrescanFiles(const std::filesystem::path& directory, uint32_t current_pid,
                                     const std::function<bool(uint32_t)>&         is_running,
                                     const std::function<void(std::string_view)>& log);

/// A factory for SupervisorPrescanTransport.
/// @param config the supervisor's settings besides what the transport fills in. A real run sets only
///               files_base, and a test also names its own worker program and short limits.
/// @param made   when given, receives a pointer to the transport, which lives as long as the
///               preparer that made it, so a test can read its supervisor's counts.
[[nodiscard]] PrescanTransportFactory
SupervisorPrescanTransportFactory(StreamSupervisorConfig       config,
                                  SupervisorPrescanTransport** made = nullptr);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_PRESCAN_TRANSPORT_H
