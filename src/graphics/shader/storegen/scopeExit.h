#ifndef GRAPHICS_SHADER_STOREGEN_SCOPE_EXIT_H
#define GRAPHICS_SHADER_STOREGEN_SCOPE_EXIT_H

#include <utility>

namespace Libs::Graphics::StoreGen {

/// Runs a call when it goes out of scope, by a return or an exception, unless it was dismissed. The
/// Stream worker uses it so a lock is let go, a waiter is woken and threads are stopped on every
/// way out of a function.
template <class Call> class ScopeExit {
public:
	explicit ScopeExit(Call call): m_call(std::move(call)) {}
	~ScopeExit() {
		if (m_armed) {
			m_call();
		}
	}
	ScopeExit(const ScopeExit&)            = delete;
	ScopeExit& operator=(const ScopeExit&) = delete;

	/// Keeps the call from running.
	void Dismiss() { m_armed = false; }

private:
	Call m_call;
	bool m_armed = true;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_SCOPE_EXIT_H
