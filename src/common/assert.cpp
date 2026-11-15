#include "common/assert.h"

#include "common/subsystems.h"
#include "kytyGitVersion.h"

#include <cstdio>
#include <cstdlib>
#include <fmt/format.h>
#include <string>
#include <utility>

namespace Common {

[[noreturn]] static void DbgReport(const char* title, std::string_view text, const char* file,
                                   int line) {
	Log::WriteToConsoleAndLog(fmt::format("--- Build ---\n{}\n{}\n{} in {}:{}\n", KYTY_BUILD_LABEL,
	                                      title, text, file, line));
	Subsystems::EmergencyShutdownActive();
	std::fflush(nullptr);
	std::_Exit(321);
}

namespace {
// How many ExitRejectScopes the calling thread holds.
thread_local uint32_t t_exit_reject_depth = 0;

// Throws in place of exiting while the calling thread holds a scope, and otherwise returns.
void ThrowIfRejecting(const char* file, int line, std::string text) {
	if (t_exit_reject_depth != 0) {
		throw ExitRejected {file, line, std::move(text)};
	}
}
} // namespace

ExitRejectScope::ExitRejectScope() noexcept {
	t_exit_reject_depth++;
}

ExitRejectScope::~ExitRejectScope() {
	t_exit_reject_depth--;
}

void DbgExitIfHandler(const char* expr, const char* file, int line) {
	ThrowIfRejecting(file, line, fmt::format("Error: condition ({}) is true", expr));
	DbgReport("--- Fatal Error ---", fmt::format("Error: condition ({}) is true", expr), file,
	          line);
}

void DbgNotImplementedHandler(const char* expr, const char* file, int line) {
	ThrowIfRejecting(file, line, fmt::format("Not implemented ({})", expr));
	DbgReport("--- Fatal Error ---", fmt::format("Not implemented ({})", expr), file, line);
}

void DbgExitHandler(const char* file, int line, std::string_view text) {
	ThrowIfRejecting(file, line, std::string(text));
	DbgReport("--- Error ---", text, file, line);
}

} // namespace Common
