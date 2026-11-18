#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_UIACTIVITY_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_UIACTIVITY_H_

#include <atomic>
#include <cstdint>

namespace Libs::Graphics::SystemUi {

// Whether the system UI needs another frame. The overlay is only redrawn when it is active or
// its revision changes, and the game may stop presenting frames, for example while a shader
// compiles. Without this, an animation would freeze halfway and new data would wait for the
// game's next frame.
//
// The present thread brackets each UI frame with BeginFrame and EndFrame, and anything still
// animating calls RequestFrame in between. Any thread calls Wake when new data arrives.
class UiActivity {
public:
	/// Present thread, before drawing. Starts collecting whether this frame needs another.
	void BeginFrame() { m_frame_requested.store(false, std::memory_order_relaxed); }

	/// Present thread, while drawing. Something is still moving, so draw another frame.
	void RequestFrame() { m_frame_requested.store(true, std::memory_order_relaxed); }

	/// Present thread, after drawing. Publishes whether the frame asked for another.
	void EndFrame() {
		m_active.store(m_frame_requested.load(std::memory_order_relaxed),
		               std::memory_order_release);
	}

	/// Any thread. New data arrived, so the UI must draw at least one more frame.
	void Wake() { m_revision.fetch_add(1, std::memory_order_acq_rel); }

	/// Any thread. True while the last frame asked for another.
	[[nodiscard]] bool Active() const { return m_active.load(std::memory_order_acquire); }

	/// Any thread. Changes whenever Wake is called.
	[[nodiscard]] uint64_t Revision() const { return m_revision.load(std::memory_order_acquire); }

private:
	// The present thread's current frame. Atomic only so that a misplaced call from another
	// thread is a harmless race rather than undefined behaviour.
	std::atomic<bool>     m_frame_requested {false};
	std::atomic<bool>     m_active {false};
	std::atomic<uint64_t> m_revision {0};
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_BINDING_UIACTIVITY_H_
