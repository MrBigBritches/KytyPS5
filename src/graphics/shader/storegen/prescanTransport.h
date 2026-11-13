#ifndef GRAPHICS_SHADER_STOREGEN_PRESCAN_TRANSPORT_H
#define GRAPHICS_SHADER_STOREGEN_PRESCAN_TRANSPORT_H

#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/computeTranslation.h"
#include "graphics/shader/storegen/streamProtocol.h"
#include "graphics/shader/storegen/streamSupervisor.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
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
	/// @return Queued, or why the request was refused. A refused request gets no outcome.
	[[nodiscard]] virtual StreamSubmitResult
	Submit(StreamRequest request, std::shared_ptr<const std::vector<uint8_t>> code, uint64_t tag) = 0;

	/// Gives up every request still queued, which completes as Dropped, and refuses new ones. The
	/// caller must not hold a lock the handler takes, since this may wait for a handler in progress.
	/// The preparer calls it from its own thread when it fails, or after that thread is joined, so it
	/// never runs beside a Submit. Calling it again does nothing.
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

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_PRESCAN_TRANSPORT_H
