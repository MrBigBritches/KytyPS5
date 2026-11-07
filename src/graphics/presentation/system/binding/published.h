#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_PUBLISHED_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_PUBLISHED_H_

#include "graphics/presentation/system/binding/uiActivity.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>

// State that emulator threads publish and the system UI reads. The UI draws on the present
// thread and must not lock emulator data while it draws, so it keeps its own copy and refreshes
// it once per frame, only when the publisher has changed it.
namespace Libs::Graphics::SystemUi {

// A value one or more threads publish. The value is copied under a small lock, and an atomic
// revision lets readers see a change without taking the lock.
template <class T>
class Published {
public:
	/// Wakes `activity`, when given, on every publish, so the UI redraws even while the game has
	/// stopped presenting frames.
	explicit Published(UiActivity* activity = nullptr): m_activity(activity) {}

	/// Replaces the value. Safe from any thread. Every publish wakes the UI, so publish only real
	/// changes. Avoid calling it while holding the presenter's lock, to keep the present thread
	/// from waiting.
	void Publish(T value) {
		{
			std::lock_guard lock(m_mutex);
			m_value = std::move(value);
			m_revision.fetch_add(1, std::memory_order_release);
		}
		if (m_activity != nullptr) {
			m_activity->Wake();
		}
	}

	/// The number of publishes so far, without taking the lock.
	[[nodiscard]] uint64_t Revision() const { return m_revision.load(std::memory_order_acquire); }

	/// A copy of the value with the revision it was published at, taken together under the lock
	/// so they always match.
	[[nodiscard]] std::pair<T, uint64_t> Read() const {
		std::lock_guard lock(m_mutex);
		return {m_value, m_revision.load(std::memory_order_relaxed)};
	}

private:
	mutable std::mutex    m_mutex;
	T                     m_value {};
	std::atomic<uint64_t> m_revision {0};
	UiActivity*           m_activity = nullptr;
};

// The UI's copy of a published value. Refresh it once per frame on the present thread, then read
// Value freely while drawing.
template <class T>
class Binding {
public:
	/// Binds to `source`, which must outlive the binding.
	explicit Binding(const Published<T>& source): m_source(&source) {}

	/// Copies the published value when it has changed since the last refresh. Returns whether it
	/// did, so a view can skip work that depends only on the value.
	bool Refresh() {
		if (m_source->Revision() == m_revision) {
			return false;
		}
		auto [value, revision] = m_source->Read();
		m_value                = std::move(value);
		m_revision             = revision;
		return true;
	}

	/// The value as of the last refresh. Default constructed before the first publish.
	[[nodiscard]] const T& Value() const { return m_value; }

	/// The revision of the value as of the last refresh.
	[[nodiscard]] uint64_t Revision() const { return m_revision; }

private:
	const Published<T>* m_source;
	T                   m_value {};
	uint64_t            m_revision = 0;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_PUBLISHED_H_
