#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_ELEMENT_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_ELEMENT_H_

#include <cstdint>
#include <type_traits>

namespace Libs::Graphics::SystemUi {

struct FrameTestHooks;

// A handle to one node of the frame's element tree. It holds the node's index and the serial of
// the frame that made it, so a handle kept past Begin() is recognised as stale instead of
// pointing at whatever node now sits at that index. Components pass handles around by value and
// never see the node itself. Only the frame and the primitives can make a valid one.
class Element {
public:
	/// The handle that refers to nothing, which is also what a default-constructed one is.
	constexpr Element() = default;

	/// True for a handle that refers to no node.
	[[nodiscard]] constexpr bool IsNone() const { return serial_ == NONE_SERIAL; }

	/// Whether two handles refer to the same node of the same frame.
	friend constexpr bool operator==(const Element&, const Element&) = default;

private:
	friend class UiFrame;
	friend struct FrameTestHooks;

	// The frame serial starts above this, so a default handle never matches a live frame.
	static constexpr uint32_t NONE_SERIAL = 0;

	constexpr Element(uint32_t index, uint32_t serial): index_(index), serial_(serial) {}

	uint32_t index_  = 0;
	uint32_t serial_ = NONE_SERIAL;
};

static_assert(sizeof(Element) == 8, "an Element is passed by value, so it stays two 32-bit words");
static_assert(std::is_trivially_copyable_v<Element>);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_ELEMENT_H_
