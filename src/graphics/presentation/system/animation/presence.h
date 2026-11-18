#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_PRESENCE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_PRESENCE_H_

#include "graphics/presentation/system/animation/timeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

namespace Libs::Graphics::SystemUi {

// Elements that come and go with their own timelines, such as toasts. A view adds an element
// when its data arrives, asks Update each frame whether to keep drawing, and draws each element
// at its current pose. Elements remove themselves once they are gone.
template <class T>
class Presence {
public:
	using Id = uint64_t;

	/// Starts showing `value` with `timeline` from `now`. Returns an id for dismissing it.
	Id Add(T value, const Timeline& timeline, UiClock::time_point now) {
		m_entries.push_back({++m_last_id, std::move(value), timeline, now, std::nullopt});
		return m_last_id;
	}

	/// Starts the element leaving at `now`. A second dismissal keeps the first, so the exit never
	/// restarts. Returns false when no element has `id`, for example because it already left.
	bool Dismiss(Id id, UiClock::time_point now) {
		const auto entry = std::find_if(m_entries.begin(), m_entries.end(),
		                                [id](const Entry& entry) { return entry.id == id; });
		if (entry == m_entries.end()) {
			return false;
		}
		entry->dismissed = entry->dismissed.value_or(now);
		return true;
	}

	/// Removes the elements that are gone at `now`, and returns whether any remaining element
	/// will change without new input, so the UI must keep drawing frames.
	bool Update(UiClock::time_point now) {
		std::erase_if(m_entries, [now](const Entry& entry) {
			return PoseOf(entry, now).phase == Phase::Gone;
		});
		return std::any_of(m_entries.begin(), m_entries.end(), [now](const Entry& entry) {
			return PoseOf(entry, now).NeedsFrames(entry.timeline);
		});
	}

	/// Calls `draw(value, pose)` for every element visible at `now`, oldest first. Elements still
	/// waiting to enter, or already gone, are skipped.
	template <class Draw>
	void ForEach(UiClock::time_point now, Draw&& draw) const {
		for (const auto& entry: m_entries) {
			const auto pose = PoseOf(entry, now);
			if (pose.phase != Phase::Waiting && pose.phase != Phase::Gone) {
				draw(entry.value, pose);
			}
		}
	}

	/// The number of elements not yet removed by Update, including ones still leaving.
	[[nodiscard]] size_t Size() const { return m_entries.size(); }

private:
	struct Entry {
		Id                                 id;
		T                                  value;
		Timeline                           timeline;
		UiClock::time_point                start;
		std::optional<UiClock::time_point> dismissed;
	};

	// An entry's pose at `now`.
	static Pose PoseOf(const Entry& entry, UiClock::time_point now) {
		return Evaluate(entry.timeline, entry.start, entry.dismissed, now);
	}

	std::vector<Entry> m_entries;
	Id                 m_last_id = 0;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ANIMATION_PRESENCE_H_
