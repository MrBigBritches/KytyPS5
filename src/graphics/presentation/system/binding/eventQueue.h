#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_EVENTQUEUE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_EVENTQUEUE_H_

#include "graphics/presentation/system/binding/uiActivity.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <utility>
#include <vector>

namespace Libs::Graphics::SystemUi {

// One-off events from emulator threads to the system UI, such as a trophy unlock. The queue is
// bounded so a game that floods it can't grow it without limit. When it is full the oldest event
// is dropped and counted. Size it for the largest burst its events can come in, since the present
// thread only drains it when it draws a frame.
template <class T>
class EventQueue {
public:
	/// A queue holding at most `capacity` events, at least one. Wakes `activity`, when given, on
	/// every push.
	explicit EventQueue(size_t capacity, UiActivity* activity = nullptr)
	    : m_capacity(capacity > 0 ? capacity : 1), m_activity(activity) {}

	/// Adds an event. Safe from any thread. Avoid calling it while holding the presenter's lock, to
	/// keep the present thread from waiting.
	void Push(T event) {
		{
			std::lock_guard lock(m_mutex);
			if (m_events.size() == m_capacity) {
				m_events.pop_front();
				m_dropped.fetch_add(1, std::memory_order_relaxed);
			}
			m_events.push_back(std::move(event));
			m_pending.store(true, std::memory_order_release);
		}
		if (m_activity != nullptr) {
			m_activity->Wake();
		}
	}

	/// Moves every queued event out, oldest first. Checks a flag before locking, so an empty
	/// queue costs one atomic load per frame.
	[[nodiscard]] std::vector<T> Drain() {
		if (!m_pending.load(std::memory_order_acquire)) {
			return {};
		}
		std::deque<T> taken;
		{
			// Swapped out under the lock and converted after, so producers wait as little as
			// possible.
			std::lock_guard lock(m_mutex);
			taken.swap(m_events);
			m_pending.store(false, std::memory_order_release);
		}
		return {std::make_move_iterator(taken.begin()), std::make_move_iterator(taken.end())};
	}

	/// True when events are waiting, without taking the lock.
	[[nodiscard]] bool HasPending() const { return m_pending.load(std::memory_order_acquire); }

	/// How many events were dropped because the queue was full.
	[[nodiscard]] uint64_t Dropped() const { return m_dropped.load(std::memory_order_relaxed); }

private:
	const size_t          m_capacity;
	UiActivity*           m_activity;
	std::mutex            m_mutex;
	std::deque<T>         m_events;
	std::atomic<bool>     m_pending {false};
	std::atomic<uint64_t> m_dropped {0};
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_EVENTQUEUE_H_
