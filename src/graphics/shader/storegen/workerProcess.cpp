#include "graphics/shader/storegen/workerProcess.h"

#include "common/childProcess.h"
#include "common/common.h"
#include "common/crashUi.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <cstdio>
#include <string>

namespace Libs::Graphics::StoreGen {

// The file is opened once first, because a failed freopen closes the stream it was given, so a
// file that cannot be opened leaves the streams as they are and the worker still works. It appends,
// since a restarted worker writes to the same file and an earlier crash's text matters.
void RedirectWorkerOutputTo(const std::string& progress_path) {
	const auto text_path = PathFromUtf8(progress_path + ".out");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const auto native = text_path.wstring();
	std::FILE* probe  = _wfopen(native.c_str(), L"a");
	if (probe == nullptr) {
		return;
	}
	std::fclose(probe);
	(void)_wfreopen(native.c_str(), L"a", stdout);
	(void)_wfreopen(native.c_str(), L"a", stderr);
#else
	std::FILE* probe = std::fopen(text_path.c_str(), "a");
	if (probe == nullptr) {
		return;
	}
	std::fclose(probe);
	(void)std::freopen(text_path.c_str(), "a", stdout);
	(void)std::freopen(text_path.c_str(), "a", stderr);
#endif
}

int RunShaderStoreWorkerProcess(const std::filesystem::path& request_file) {
	const auto request = ReadWorkerRequestFile(request_file);
	if (!request) {
		return WORKER_EXIT_BAD_REQUEST;
	}
	Common::DisableCrashUi();
	RedirectWorkerOutputTo(request->progress_path);
	Common::ChildProcess::EnterChildMode(request->parent_pid);
	return RunShaderStoreWorker(*request).exit_code;
}

} // namespace Libs::Graphics::StoreGen
