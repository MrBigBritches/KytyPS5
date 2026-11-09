#include "common/crashUi.h"

#include "common/common.h"

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <stdlib.h>
#else
#include <sys/resource.h>
#endif

namespace Common {

void DisableCrashUi() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
	_set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#else
	const rlimit no_core {0, 0};
	setrlimit(RLIMIT_CORE, &no_core);
#endif
}

} // namespace Common
