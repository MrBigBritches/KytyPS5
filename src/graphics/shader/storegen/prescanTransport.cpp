#include "graphics/shader/storegen/prescanTransport.h"

#include "graphics/shader/storegen/streamWorker.h"

#include <utility>

namespace Libs::Graphics::StoreGen {

InProcessPrescanTransport::InProcessPrescanTransport(const HostValues&        host,
                                                     const ComputeKeyDevice&  device,
                                                     PrescanCompletionHandler handler)
    : m_host(host), m_device(device), m_handler(std::move(handler)) {}

// The code is copied into the request, since the worker's answer reads it from there. The seq is
// given under the lock, and the answer runs outside it, so a handler that submits again cannot
// deadlock on it.
StreamSubmitResult InProcessPrescanTransport::Submit(StreamRequest                               request,
                                                     std::shared_ptr<const std::vector<uint8_t>> code,
                                                     uint64_t                                    tag) {
	if (code != nullptr) {
		request.code = *code;
	}
	{
		std::lock_guard lock(m_mutex);
		if (m_stopped) {
			return {StreamSubmitStatus::Stopped, 0};
		}
		(request.kind == StreamRequestKind::Plan ? m_plans : m_compiles)
		    .fetch_add(1, std::memory_order_relaxed);
		request.seq = m_next_seq++;
		if (m_hold && request.kind == StreamRequestKind::Compile) {
			const uint64_t seq = request.seq;
			m_held.push_back({std::move(request), tag});
			return {StreamSubmitStatus::Queued, seq};
		}
	}
	const uint64_t seq = request.seq;
	Answer(request, tag);
	return {StreamSubmitStatus::Queued, seq};
}

// Held requests complete as Dropped, as the supervisor's outstanding ones do when it stops.
void InProcessPrescanTransport::Stop() {
	std::deque<HeldRequest> held;
	{
		std::lock_guard lock(m_mutex);
		m_stopped = true;
		held.swap(m_held);
	}
	for (auto& entry: held) {
		StreamCompletion completion;
		completion.seq     = entry.request.seq;
		completion.kind    = entry.request.kind;
		completion.outcome = StreamOutcome::Dropped;
		completion.tag     = entry.tag;
		m_handler(std::move(completion));
	}
}

void InProcessPrescanTransport::SetHoldCompiles(bool hold) {
	std::lock_guard lock(m_mutex);
	m_hold = hold;
}

size_t InProcessPrescanTransport::Held() const {
	std::lock_guard lock(m_mutex);
	return m_held.size();
}

void InProcessPrescanTransport::ReleaseHeld() {
	std::deque<HeldRequest> held;
	{
		std::lock_guard lock(m_mutex);
		held.swap(m_held);
	}
	for (const auto& entry: held) {
		Answer(entry.request, entry.tag);
	}
}

void InProcessPrescanTransport::Answer(const StreamRequest& request, uint64_t tag) {
	StreamCompletion completion;
	completion.seq      = request.seq;
	completion.kind     = request.kind;
	completion.outcome  = StreamOutcome::Answered;
	completion.tag      = tag;
	completion.response = AnswerStreamRequest(request, m_host, m_device);
	m_handler(std::move(completion));
}

PrescanTransportFactory InProcessPrescanTransportFactory(InProcessPrescanTransport** made) {
	return [made](const HostValues& host, const ComputeKeyDevice& device,
	              PrescanCompletionHandler handler) -> std::unique_ptr<PrescanTransport> {
		auto transport = std::make_unique<InProcessPrescanTransport>(host, device, std::move(handler));
		if (made != nullptr) {
			*made = transport.get();
		}
		return transport;
	};
}

} // namespace Libs::Graphics::StoreGen
