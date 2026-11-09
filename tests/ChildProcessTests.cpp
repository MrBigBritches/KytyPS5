#include "common/childProcess.h"
#include "common/config.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#else
#include <csignal>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif

// The test binary starts itself as the child, with --fake-child <mode>, so the same code runs
// on every operating system without a second helper executable.
namespace {

using Common::ChildProcess;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

constexpr auto CHILD_FINISH_TIMEOUT  = 20s;
// How fast Terminate must finish, and how long a killed process may take to disappear.
constexpr auto GONE_DEADLINE         = 2s;
// A watchdog polls every 500 ms and loaded CI machines can be slow, so parent death gets longer.
constexpr auto DEATH_DEADLINE        = 5s;
constexpr auto FILE_WAIT_TIMEOUT     = 20s;
constexpr auto POLL_PAUSE            = 5ms;
constexpr int  EXIT_CODE_SEVEN       = 7;
constexpr int  EXIT_CODE_BIG         = 321;
[[maybe_unused]] constexpr int POSIX_EXIT_CODE_MASK = 0xFF;
constexpr auto HANG_SLEEP            = 1s;
// Exit codes of the fake children for failures, apart from the codes the tests look for.
constexpr int  FAKE_PARENT_SPAWN_FAILED = 3;
constexpr int  FAKE_PARENT_NOT_RELEASED = 4;
constexpr int  UNKNOWN_FAKE_MODE        = 99;
// Same buffer size BuildStamp has always used for _NSGetExecutablePath.
[[maybe_unused]] constexpr size_t MAC_EXECUTABLE_PATH_BUFFER = 4096;
// The lowest scheduling priority a POSIX process can have, expressed as the rank PriorityRank uses.
[[maybe_unused]] constexpr int LOWEST_POSIX_RANK = -19;
// The rank of below normal in PriorityRank on Windows.
[[maybe_unused]] constexpr int WINDOWS_BELOW_NORMAL_RANK = 1;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

// Writes through a temporary name so a reader never sees half a file.
void WriteTextFile(const fs::path& path, const std::string& text) {
	const fs::path temporary = path.string() + ".part";
	{
		std::ofstream file(temporary, std::ios::binary);
		file << text;
	}
	fs::rename(temporary, path);
}

// The path as UTF-8, matching what ChildProcess expects for its arguments.
std::string Utf8(const fs::path& path) {
	const auto text = path.u8string();
	return {reinterpret_cast<const char*>(text.data()), text.size()};
}

// Reads a whole file, or an empty string when it does not exist.
std::string ReadTextFile(const fs::path& path) {
	std::ifstream      file(path, std::ios::binary);
	std::ostringstream text;
	text << file.rdbuf();
	return text.str();
}

// Waits for a file to appear. Files are written by rename, so an existing file is complete.
bool WaitForFile(const fs::path& path, std::chrono::milliseconds timeout) {
	const auto deadline = std::chrono::steady_clock::now() + timeout;
	while (!fs::exists(path)) {
		if (std::chrono::steady_clock::now() >= deadline) {
			return false;
		}
		std::this_thread::sleep_for(POLL_PAUSE);
	}
	return true;
}

// A rank where a higher number means a higher scheduling priority, comparable across OSes.
int PriorityRank() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	switch (GetPriorityClass(GetCurrentProcess())) {
		case IDLE_PRIORITY_CLASS: return 0;
		case BELOW_NORMAL_PRIORITY_CLASS: return 1;
		case NORMAL_PRIORITY_CLASS: return 2;
		case ABOVE_NORMAL_PRIORITY_CLASS: return 3;
		case HIGH_PRIORITY_CLASS: return 4;
		default: return 5;
	}
#else
	// A larger nice value means the process is scheduled less.
	return -getpriority(PRIO_PROCESS, 0);
#endif
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// A zombie still answers kill(pid, 0), but it has finished running and only waits to be reaped.
// Linux shows that in /proc, and the check matters when the reaper (init) is missing.
bool IsZombie(uint32_t pid) {
	std::ifstream stat("/proc/" + std::to_string(pid) + "/stat");
	std::string   line;
	if (!std::getline(stat, line)) {
		return false;
	}
	const auto close_paren = line.rfind(')');
	return close_paren != std::string::npos && close_paren + 2 < line.size() && line[close_paren + 2] == 'Z';
}
#endif

// True while the process exists and has not finished. A zombie counts as finished, so
// ExistsAtAll is the check that catches a missing reap.
bool IsAlive(uint32_t pid) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
	if (process == nullptr) {
		return false;
	}
	const bool running = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
	CloseHandle(process);
	return running;
#else
	return kill(static_cast<pid_t>(pid), 0) == 0 && !IsZombie(pid);
#endif
}

// True while the process still has an entry in the system, running or not. A finished child
// that was not reaped (POSIX) or not closed (Windows, once every handle is gone) is the thing
// this tells apart from a properly collected one.
bool ExistsAtAll(uint32_t pid) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
	if (process == nullptr) {
		return false;
	}
	CloseHandle(process);
	return true;
#else
	return kill(static_cast<pid_t>(pid), 0) == 0;
#endif
}

// The system may drop a finished process a moment after its last handle closed.
bool WaitUntilForgotten(uint32_t pid, std::chrono::milliseconds deadline) {
	const auto limit = std::chrono::steady_clock::now() + deadline;
	while (ExistsAtAll(pid)) {
		if (std::chrono::steady_clock::now() >= limit) {
			return false;
		}
		std::this_thread::sleep_for(POLL_PAUSE);
	}
	return true;
}

// Waits until the process has finished or is gone.
bool WaitUntilGone(uint32_t pid, std::chrono::milliseconds deadline) {
	const auto limit = std::chrono::steady_clock::now() + deadline;
	while (IsAlive(pid)) {
		if (std::chrono::steady_clock::now() >= limit) {
			return false;
		}
		std::this_thread::sleep_for(POLL_PAUSE);
	}
	return true;
}

// ---- child side ----

[[noreturn]] void HangForever() {
	while (true) {
		std::this_thread::sleep_for(HANG_SLEEP);
	}
}

// args are the words after --fake-child.
int RunFakeChild(const std::vector<std::string>& args);

// The child that stands in for the parent of a hanging grandchild. It reports once the grandchild
// runs, waits for the test to confirm that, and then dies without any cleanup.
int RunFakeParent(const fs::path& work_dir, const std::string& grandchild_mode) {
	ChildProcess grandchild;
	const auto   ready = work_dir / "grandchild.pid";
	const std::vector<std::string> arguments {"--fake-child", grandchild_mode, std::to_string(Common::CurrentProcessId()),
	                                          Utf8(ready)};
	if (!grandchild.Start(Common::ExecutablePath(), arguments) || !WaitForFile(ready, FILE_WAIT_TIMEOUT)) {
		return FAKE_PARENT_SPAWN_FAILED;
	}
	WriteTextFile(work_dir / "parent.ready", "1");
	if (!WaitForFile(work_dir / "parent.go", FILE_WAIT_TIMEOUT)) {
		return FAKE_PARENT_NOT_RELEASED;
	}
	// No destructors and no Terminate, as in a crash.
	std::_Exit(0);
}

int RunFakeChild(const std::vector<std::string>& args) {
	const std::string mode = args.empty() ? "" : args[0];
	if (mode == "exit7") {
		return EXIT_CODE_SEVEN;
	}
	if (mode == "exit321") {
		std::_Exit(EXIT_CODE_BIG);
	}
	if (mode == "fake-parent" && args.size() >= 3) {
		return RunFakeParent(args[1], args[2]);
	}
	// <mode> <parent pid> <pid file>. Each mode arms a different set of guards, so a test can
	// show that one guard alone ends the child.
	// hang: none, so only what the spawn gives (the job on Windows).
	// hang-deathsig: only the death signal (Linux).
	// hang-watchdog-only: only the watchdog.
	// hang-watched: all of EnterChildMode.
	if (mode.starts_with("hang") && args.size() >= 3) {
		const auto parent_pid = static_cast<uint32_t>(std::stoul(args[1]));
		if (mode == "hang-deathsig") {
			ChildProcess::EnterChildMode(parent_pid, {.lower_priority = false, .death_signal = true, .watchdog = false});
		} else if (mode == "hang-watchdog-only") {
			ChildProcess::EnterChildMode(parent_pid, {.lower_priority = false, .death_signal = false, .watchdog = true});
		} else if (mode == "hang-watched") {
			ChildProcess::EnterChildMode(parent_pid);
		}
		WriteTextFile(args[2], std::to_string(Common::CurrentProcessId()));
		HangForever();
	}
	// priority <parent pid> <enter child mode 0|1> <output file>
	if (mode == "priority" && args.size() >= 4) {
		if (args[2] == "1") {
			ChildProcess::EnterChildMode(static_cast<uint32_t>(std::stoul(args[1])));
		}
		WriteTextFile(args[3], std::to_string(PriorityRank()));
		return 0;
	}
	return UNKNOWN_FAKE_MODE;
}

// ---- test side ----

// A fresh empty directory for one test, under a directory named for this process.
fs::path MakeWorkDir(const char* name) {
	const fs::path dir = fs::temp_directory_path() / ("kyty_child_process_" + std::to_string(Common::CurrentProcessId())) / name;
	fs::remove_all(dir);
	fs::create_directories(dir);
	return dir;
}

// Starts this binary as a fake child in the given mode with the given arguments.
bool SpawnFake(ChildProcess& child, std::vector<std::string> mode_arguments) {
	mode_arguments.insert(mode_arguments.begin(), "--fake-child");
	return child.Start(Common::ExecutablePath(), mode_arguments);
}

// A child that runs and exits by itself.
void TestExitCodes() {
	ChildProcess seven;
	CHECK(SpawnFake(seven, {"exit7"}));
	const auto seven_code = seven.Wait(CHILD_FINISH_TIMEOUT);
	// The child could also fail to start and report some other code, which a wrong number catches.
	CHECK(seven_code.has_value() && *seven_code == EXIT_CODE_SEVEN);
	CHECK(seven.Poll() == seven_code);

	ChildProcess big;
	CHECK(SpawnFake(big, {"exit321"}));
	const auto big_code = big.Wait(CHILD_FINISH_TIMEOUT);
	// POSIX keeps only the low 8 bits of an exit status.
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	const int expected_big = EXIT_CODE_BIG;
#else
	const int expected_big = EXIT_CODE_BIG & POSIX_EXIT_CODE_MASK;
#endif
	CHECK(big_code.has_value() && *big_code == expected_big);
}

// A child that has not finished must not report an exit code.
void TestPollWhileRunning() {
	const auto    dir = MakeWorkDir("poll");
	ChildProcess  child;
	const fs::path ready = dir / "child.pid";
	CHECK(SpawnFake(child, {"hang", std::to_string(Common::CurrentProcessId()), Utf8(ready)}));
	CHECK(WaitForFile(ready, FILE_WAIT_TIMEOUT));
	CHECK(!child.Poll().has_value());
	CHECK(!child.Wait(50ms).has_value());
	CHECK(child.Pid() == std::stoul(ReadTextFile(ready)));
}

// Terminate must end a hanging child promptly and leave nothing behind to reap.
void TestTerminateHangingChild() {
	const auto    dir = MakeWorkDir("terminate");
	ChildProcess  child;
	const fs::path ready = dir / "child.pid";
	CHECK(SpawnFake(child, {"hang", std::to_string(Common::CurrentProcessId()), Utf8(ready)}));
	// The pid file shows the child really started, so a spawn that never ran cannot pass.
	CHECK(WaitForFile(ready, FILE_WAIT_TIMEOUT));
	const uint32_t pid = child.Pid();
	CHECK(IsAlive(pid));
	const auto begin = std::chrono::steady_clock::now();
	child.Terminate();
	CHECK(std::chrono::steady_clock::now() - begin < GONE_DEADLINE);
	CHECK(WaitUntilGone(pid, GONE_DEADLINE));
	CHECK(child.Poll().has_value());
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	// Reaped means the pid no longer exists, where an unreaped child would remain a zombie.
	CHECK(WaitUntilForgotten(pid, GONE_DEADLINE));
#endif
}

// A finished child leaves no process entry once the object is gone, and a running one is killed.
void TestDestructor() {
	const auto dir = MakeWorkDir("destructor");
	uint32_t   finished_pid = 0;
	{
		ChildProcess child;
		CHECK(SpawnFake(child, {"exit7"}));
		finished_pid = child.Pid();
		CHECK(child.Wait(CHILD_FINISH_TIMEOUT).has_value());
	}
	CHECK(WaitUntilForgotten(finished_pid, GONE_DEADLINE));

	uint32_t       hanging_pid = 0;
	const fs::path ready       = dir / "child.pid";
	{
		ChildProcess child;
		CHECK(SpawnFake(child, {"hang", std::to_string(Common::CurrentProcessId()), Utf8(ready)}));
		CHECK(WaitForFile(ready, FILE_WAIT_TIMEOUT));
		hanging_pid = child.Pid();
		CHECK(IsAlive(hanging_pid));
	}
	CHECK(WaitUntilGone(hanging_pid, GONE_DEADLINE));
	CHECK(WaitUntilForgotten(hanging_pid, GONE_DEADLINE));
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Handles that are never closed show up as a growing handle count.
void TestNoHandleLeak() {
	constexpr int SPAWN_CYCLES = 12;
	auto handle_count = [] {
		DWORD count = 0;
		GetProcessHandleCount(GetCurrentProcess(), &count);
		return count;
	};
	{
		ChildProcess warm_up;
		CHECK(SpawnFake(warm_up, {"exit7"}));
		CHECK(warm_up.Wait(CHILD_FINISH_TIMEOUT).has_value());
	}
	const DWORD before = handle_count();
	for (int cycle = 0; cycle < SPAWN_CYCLES; ++cycle) {
		ChildProcess child;
		CHECK(SpawnFake(child, {"exit7"}));
		CHECK(child.Wait(CHILD_FINISH_TIMEOUT).has_value());
	}
	CHECK(handle_count() <= before);
}
#endif

// Ends a process by pid, for cleaning up a grandchild the test could not otherwise reach.
void KillProcess(uint32_t pid) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	if (HANDLE process = OpenProcess(PROCESS_TERMINATE, FALSE, pid)) {
		TerminateProcess(process, 1);
		CloseHandle(process);
	}
#else
	kill(static_cast<pid_t>(pid), SIGKILL);
#endif
}

// Kills a grandchild when a test returns early, so a failure cannot leave it running. Disarmed
// once the grandchild is known to be gone, since its pid could then belong to someone else.
struct GrandchildCleanup {
	uint32_t pid   = 0;
	bool     armed = true;
	~GrandchildCleanup() {
		if (armed && pid != 0) {
			KillProcess(pid);
		}
	}
};

// Starts a fake parent that spawns a hanging grandchild and then dies without cleanup, and
// reports whether the grandchild disappeared.
bool GrandchildDiesWithParent(const char* name, const std::string& grandchild_mode) {
	const auto                     dir = MakeWorkDir(name);
	ChildProcess                   parent;
	const std::vector<std::string> arguments {"fake-parent", Utf8(dir), grandchild_mode};
	if (!SpawnFake(parent, arguments) || !WaitForFile(dir / "grandchild.pid", FILE_WAIT_TIMEOUT)) {
		return false;
	}
	// Read as soon as it exists, so every later failure path can still clean the grandchild up.
	GrandchildCleanup cleanup;
	cleanup.pid = static_cast<uint32_t>(std::stoul(ReadTextFile(dir / "grandchild.pid")));
	if (!WaitForFile(dir / "parent.ready", FILE_WAIT_TIMEOUT)) {
		return false;
	}
	// The grandchild has to be running now, or its absence later proves nothing.
	if (cleanup.pid == parent.Pid() || !IsAlive(cleanup.pid) || !IsAlive(parent.Pid())) {
		return false;
	}
	WriteTextFile(dir / "parent.go", "1");
	const auto parent_code = parent.Wait(CHILD_FINISH_TIMEOUT);
	// The parent must have exited by itself, so the grandchild was not killed by this test.
	if (!parent_code || *parent_code != 0) {
		return false;
	}
	cleanup.armed = !WaitUntilGone(cleanup.pid, DEATH_DEADLINE);
	return !cleanup.armed;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// Only the watchdog can end this child. It is started directly by the test, so the test's own
// job keeps the child alive and nothing but the death of the watched process remains.
void TestWatchdogAloneOnWindows() {
	const auto     dir = MakeWorkDir("watchdog_windows");
	ChildProcess   target;
	ChildProcess   watcher;
	const fs::path target_file  = dir / "target.pid";
	const fs::path watcher_file = dir / "watcher.pid";
	CHECK(SpawnFake(target, {"hang", std::to_string(Common::CurrentProcessId()), Utf8(target_file)}));
	CHECK(WaitForFile(target_file, FILE_WAIT_TIMEOUT));
	CHECK(SpawnFake(watcher, {"hang-watchdog-only", std::to_string(target.Pid()), Utf8(watcher_file)}));
	CHECK(WaitForFile(watcher_file, FILE_WAIT_TIMEOUT));
	// Alive while the target lives, so its end later is caused by the target dying.
	CHECK(!watcher.Wait(500ms).has_value());
	target.Terminate();
	// The watchdog ends the process with exit code 1, so any other code is a crash, not the watchdog.
	constexpr int WATCHDOG_EXIT_CODE = 1;
	CHECK(watcher.Wait(DEATH_DEADLINE) == WATCHDOG_EXIT_CODE);
}
#endif

// Each guard is shown alone where the platform has it, then all together.
void TestParentDeathKillsChild() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// No EnterChildMode, so only the job can end this grandchild.
	CHECK(GrandchildDiesWithParent("parent_death_job", "hang"));
	TestWatchdogAloneOnWindows();
#else
	CHECK(GrandchildDiesWithParent("parent_death_watchdog_only", "hang-watchdog-only"));
#ifdef __linux__
	CHECK(GrandchildDiesWithParent("parent_death_deathsig", "hang-deathsig"));
#endif
#endif
	CHECK(GrandchildDiesWithParent("parent_death_all", "hang-watched"));
}

// The baseline is this process, which runs at the default priority. A child that skips
// EnterChildMode shows what the spawn alone gives.
void TestChildPriorityIsLower() {
	const auto dir  = MakeWorkDir("priority");
	const int  mine = PriorityRank();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// A parent that is already low cannot be told apart from its child.
	if (mine <= WINDOWS_BELOW_NORMAL_RANK) {
		std::puts("Skipping the priority check, the test already runs at a low priority");
		return;
	}
	constexpr bool SPAWN_SETS_PRIORITY = true;
#else
	if (mine <= LOWEST_POSIX_RANK) {
		std::puts("Skipping the priority check, the test already has the lowest priority");
		return;
	}
	// POSIX spawn cannot set a priority, so the child has to do it itself.
	constexpr bool SPAWN_SETS_PRIORITY = false;
#endif

	ChildProcess   entered;
	const fs::path entered_file = dir / "entered.txt";
	CHECK(SpawnFake(entered, {"priority", std::to_string(Common::CurrentProcessId()), "1", Utf8(entered_file)}));
	CHECK(entered.Wait(CHILD_FINISH_TIMEOUT) == 0);
	CHECK(fs::exists(entered_file));
	CHECK(std::stoi(ReadTextFile(entered_file)) < mine);

	// Where the spawn sets nothing, skipping EnterChildMode leaves the child equal to the parent,
	// which shows the drop above comes from that call and not from the environment.
	ChildProcess   plain;
	const fs::path plain_file = dir / "plain.txt";
	CHECK(SpawnFake(plain, {"priority", std::to_string(Common::CurrentProcessId()), "0", Utf8(plain_file)}));
	CHECK(plain.Wait(CHILD_FINISH_TIMEOUT) == 0);
	const int plain_rank = std::stoi(ReadTextFile(plain_file));
	CHECK(SPAWN_SETS_PRIORITY ? plain_rank < mine : plain_rank == mine);
}

// Arguments with spaces, quotes and trailing backslashes must reach the child unchanged.
void TestQuoting() {
	CHECK(Common::QuoteWindowsArgument("plain") == "plain");
	CHECK(Common::QuoteWindowsArgument("") == "\"\"");
	CHECK(Common::QuoteWindowsArgument("a b") == "\"a b\"");
	CHECK(Common::QuoteWindowsArgument("say \"hi\"") == "\"say \\\"hi\\\"\"");
	CHECK(Common::QuoteWindowsArgument("C:\\dir with space\\") == "\"C:\\dir with space\\\\\"");
	CHECK(Common::QuoteWindowsArgument("back\\\\\"q") == "\"back\\\\\\\\\\\"q\"");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// The parse that matters is the one the child's runtime does.
	const std::vector<std::string> samples {"plain", "", "a b", "say \"hi\"", "C:\\dir with space\\", "back\\\\\"q", "tab\there"};
	for (const auto& sample: samples) {
		const std::string line = "prog " + Common::QuoteWindowsArgument(sample);
		const int         wide_size = MultiByteToWideChar(CP_UTF8, 0, line.c_str(), -1, nullptr, 0);
		std::wstring      wide(static_cast<size_t>(wide_size), L'\0');
		MultiByteToWideChar(CP_UTF8, 0, line.c_str(), -1, wide.data(), wide_size);
		int      count = 0;
		LPWSTR*  parsed = CommandLineToArgvW(wide.c_str(), &count);
		CHECK(parsed != nullptr && count == 2);
		if (parsed != nullptr && count == 2) {
			std::string back(static_cast<size_t>(WideCharToMultiByte(CP_UTF8, 0, parsed[1], -1, nullptr, 0, nullptr, nullptr)), '\0');
			WideCharToMultiByte(CP_UTF8, 0, parsed[1], -1, back.data(), static_cast<int>(back.size()), nullptr, nullptr);
			back.resize(back.size() - 1);
			CHECK(back == sample);
		}
		LocalFree(parsed);
	}
#endif
}

// The lookup BuildStamp used before ExecutableStatPath existed. The shader store stamp is the
// last write time of this path, so the shared lookup has to keep naming the same file.
fs::path LegacyStampPath() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	std::array<wchar_t, 32768> name {};
	GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	return name.data();
#elif defined(__APPLE__)
	std::array<char, MAC_EXECUTABLE_PATH_BUFFER> name {};
	auto                   size = static_cast<uint32_t>(name.size());
	_NSGetExecutablePath(name.data(), &size);
	return name.data();
#else
	return "/proc/self/exe";
#endif
}

// The executable path must name the running binary, since every spawn depends on it.
void TestExecutablePath() {
	const fs::path path = Common::ExecutablePath();
	CHECK(path.is_absolute());
	CHECK(fs::exists(path));
	std::error_code error;
	CHECK(fs::equivalent(path, Common::ExecutableStatPath(), error));
	// The stamp is a time value, so equal times from the old and new lookups mean an equal stamp.
	const auto legacy_time = fs::last_write_time(LegacyStampPath(), error).time_since_epoch().count();
	CHECK(!error);
	const auto shared_time = fs::last_write_time(Common::ExecutableStatPath(), error).time_since_epoch().count();
	CHECK(!error);
	CHECK(legacy_time == shared_time);
	CHECK(legacy_time != 0);
}

} // namespace

int main(int argc, char** argv) {
	if (argc >= 2 && std::string(argv[1]) == "--fake-child") {
		return RunFakeChild(std::vector<std::string>(argv + 2, argv + argc));
	}
	TestQuoting();
	TestExecutablePath();
	TestExitCodes();
	TestPollWhileRunning();
	TestTerminateHangingChild();
	TestDestructor();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TestNoHandleLeak();
#endif
	TestChildPriorityIsLower();
	TestParentDeathKillsChild();
	std::error_code error;
	fs::remove_all(fs::temp_directory_path() / ("kyty_child_process_" + std::to_string(Common::CurrentProcessId())), error);
	if (g_failures == 0) {
		std::puts("Child process tests passed");
	}
	return g_failures == 0 ? 0 : 1;
}
