#ifndef KYTY_COMMON_ASSERT_H_
#define KYTY_COMMON_ASSERT_H_

#include "common/common.h"
#include "common/logging/log.h"

#include <string>
#include <string_view>

namespace Common {

[[noreturn]] void DbgExitHandler(char const* file, int line, std::string_view text);
[[noreturn]] void DbgExitIfHandler(char const* expr, char const* file, int line);
[[noreturn]] void DbgNotImplementedHandler(char const* expr, char const* file, int line);

/// What the three handlers above throw instead of exiting while an ExitRejectScope is active on the
/// calling thread (experimental).
struct ExitRejected {
	const char* file = nullptr;
	int         line = 0;
	std::string text;
};

/// Makes EXIT, EXIT_IF and EXIT_NOT_IMPLEMENTED on this thread throw ExitRejected instead of ending the
/// process while it lives (experimental). The graphics prescan's worker holds one around translation and
/// compilation, whose inputs come from guest memory read ahead of the draw, so a check that fails
/// there drops one job instead of the run. It must never be held where a throw cannot unwind, such as
/// through generated code or a noexcept frame. Scopes nest.
///
/// An EXIT reached while an exception thrown inside the innermost scope is unwinding stays fatal,
/// because a destructor run by that unwinding cannot throw again without ending the process through
/// std::terminate. A scope entered inside such a destructor covers that destructor as usual.
/// Destructors that can EXIT hold an ExitRejectPause, so an EXIT in them is the same clean fatal exit
/// with or without a scope.
class ExitRejectScope {
public:
	ExitRejectScope() noexcept;
	~ExitRejectScope();
	ExitRejectScope(const ExitRejectScope&)            = delete;
	ExitRejectScope& operator=(const ExitRejectScope&) = delete;

private:
	int m_outer_uncaught = 0; ///< The enclosing scope's exception count, put back when this one ends.
};

/// Leaves EXIT fatal on this thread while it lives, even under an ExitRejectScope, and puts the
/// scopes back when it ends (experimental). Destructors that can EXIT hold one, since a throw out of
/// them would reach a noexcept frame and end the process through std::terminate instead of EXIT.
class ExitRejectPause {
public:
	ExitRejectPause() noexcept;
	~ExitRejectPause();
	ExitRejectPause(const ExitRejectPause&)            = delete;
	ExitRejectPause& operator=(const ExitRejectPause&) = delete;

private:
	uint32_t m_paused_depth = 0; ///< How many scopes were in force when the pause began.
};

} // namespace Common

#ifndef KYTY_FINAL
#define EXIT_IF(x) ((x) ? Common::DbgExitIfHandler(#x, __FILE__, __LINE__) : (void)0)
#else
#define EXIT_IF(x) ((void)(false && (x)))
#endif
#define EXIT(...) Common::DbgExitHandler(__FILE__, __LINE__, ::fmt::sprintf(__VA_ARGS__))
#define EXIT_NOT_IMPLEMENTED(x)                                                                    \
	((x) ? Common::DbgNotImplementedHandler(#x, __FILE__, __LINE__) : (void)0)

#endif /* KYTY_COMMON_ASSERT_H_ */
