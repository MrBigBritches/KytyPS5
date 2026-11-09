#include "common/childProcess.h"

#include "common/config.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <system_error>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <csignal>
#include <cstdlib>
#include <fcntl.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <crt_externs.h>
#include <mach-o/dyld.h>
#else
#include <sys/prctl.h>
extern char** environ;
#endif
#endif

namespace Common {

namespace {

// How long Terminate waits for the system to finish the child. A killed process normally
// disappears within a few milliseconds.
constexpr std::chrono::milliseconds TERMINATE_GRACE {2000};

// The buffer _NSGetExecutablePath fills on macOS. Kept as it was in BuildStamp so the shader
// store stamp does not change.
[[maybe_unused]] constexpr size_t MAC_EXECUTABLE_PATH_BUFFER = 4096;

// The path as UTF-8, which is what the Windows command line is built from.
[[maybe_unused]] std::string PathToUtf8(const std::filesystem::path& path) {
	const auto text = path.u8string();
	return {reinterpret_cast<const char*>(text.data()), text.size()};
}

} // namespace

std::string QuoteWindowsArgument(std::string_view argument) {
	if (!argument.empty() && argument.find_first_of(" \t\n\v\"") == std::string_view::npos) {
		return std::string(argument);
	}
	std::string quoted = "\"";
	// Backslashes only matter in front of a quote, where each one must be doubled. They are
	// held back until the next character shows what follows them.
	size_t pending_backslashes = 0;
	for (const char character: argument) {
		if (character == '\\') {
			++pending_backslashes;
			continue;
		}
		if (character == '"') {
			quoted.append((pending_backslashes * 2) + 1, '\\');
		} else {
			quoted.append(pending_backslashes, '\\');
		}
		quoted.push_back(character);
		pending_backslashes = 0;
	}
	// The closing quote follows, so trailing backslashes are doubled as well.
	quoted.append(pending_backslashes * 2, '\\');
	quoted.push_back('"');
	return quoted;
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

namespace {

constexpr size_t MAX_WINDOWS_PATH = 32768;

// The exit code Windows reports for a child that Terminate killed.
constexpr unsigned TERMINATED_EXIT_CODE = 1;

// Converts UTF-8 to the UTF-16 the Windows API takes.
std::wstring Widen(const std::string& utf8) {
	if (utf8.empty()) {
		return {};
	}
	const int length = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
	std::wstring wide(static_cast<size_t>(std::max(length, 0)), L'\0');
	MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), length);
	return wide;
}

// Watches a handle to the parent and ends this process when it signals. The handle comes from a
// pid, which Windows may reuse once the parent is gone. That only matters if the parent died
// before the handle was opened, and the job object covers that case.
void WatchParent(HANDLE parent) {
	WaitForSingleObject(parent, INFINITE);
	TerminateProcess(GetCurrentProcess(), TERMINATED_EXIT_CODE);
}

} // namespace

std::filesystem::path ExecutablePath() {
	std::array<wchar_t, MAX_WINDOWS_PATH> name {};
	GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
	return name.data();
}

std::filesystem::path ExecutableStatPath() {
	return ExecutablePath();
}

uint32_t CurrentProcessId() {
	return GetCurrentProcessId();
}

struct ChildProcess::Impl {
	HANDLE             process = nullptr;
	HANDLE             job     = nullptr;
	DWORD              pid     = 0;
	std::optional<int> exit_code;

	// Closing the job handle is what kills the child when this process dies without cleanup.
	void Release() {
		if (process != nullptr) {
			CloseHandle(process);
			process = nullptr;
		}
		if (job != nullptr) {
			CloseHandle(job);
			job = nullptr;
		}
	}

	// A job that kills its members when its last handle closes, which Windows also does when
	// this process is terminated.
	bool CreateKillOnCloseJob() {
		job = CreateJobObjectW(nullptr, nullptr);
		if (job == nullptr) {
			return false;
		}
		JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
		limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
		return SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) != FALSE;
	}
};

ChildProcess::ChildProcess(): m_impl(std::make_unique<Impl>()) {}

ChildProcess::~ChildProcess() {
	Terminate();
	m_impl->Release();
}

bool ChildProcess::Start(const std::filesystem::path& program, std::span<const std::string> arguments) {
	if (m_impl->process != nullptr || program.empty()) {
		return false;
	}
	std::wstring command_line = Widen(QuoteWindowsArgument(PathToUtf8(program)));
	for (const auto& argument: arguments) {
		command_line += L' ';
		command_line += Widen(QuoteWindowsArgument(argument));
	}
	if (!m_impl->CreateKillOnCloseJob()) {
		m_impl->Release();
		return false;
	}
	STARTUPINFOW        startup = {sizeof(startup)};
	PROCESS_INFORMATION created {};
	// A fixed below normal class would put the child above an idle parent.
	const DWORD priority =
	    GetPriorityClass(GetCurrentProcess()) == IDLE_PRIORITY_CLASS ? IDLE_PRIORITY_CLASS : BELOW_NORMAL_PRIORITY_CLASS;
	// Suspended, so the child cannot spawn its own children before it is in the job.
	const DWORD flags = CREATE_SUSPENDED | CREATE_NO_WINDOW | priority;
	if (CreateProcessW(program.c_str(), command_line.data(), nullptr, nullptr, FALSE, flags, nullptr, nullptr,
	                   &startup, &created) == FALSE) {
		m_impl->Release();
		return false;
	}
	m_impl->process = created.hProcess;
	m_impl->pid     = created.dwProcessId;
	if (AssignProcessToJobObject(m_impl->job, created.hProcess) == FALSE) {
		TerminateProcess(created.hProcess, TERMINATED_EXIT_CODE);
		CloseHandle(created.hThread);
		m_impl->Release();
		m_impl->pid = 0;
		return false;
	}
	ResumeThread(created.hThread);
	CloseHandle(created.hThread);
	return true;
}

uint32_t ChildProcess::Pid() const {
	return m_impl->pid;
}

std::optional<int> ChildProcess::Wait(std::chrono::milliseconds timeout) {
	if (m_impl->exit_code || m_impl->process == nullptr) {
		return m_impl->exit_code;
	}
	// INFINITE itself would never time out, so it is left out of the range.
	const auto milliseconds = std::clamp<long long>(timeout.count(), 0, static_cast<long long>(INFINITE) - 1);
	if (WaitForSingleObject(m_impl->process, static_cast<DWORD>(milliseconds)) != WAIT_OBJECT_0) {
		return std::nullopt;
	}
	DWORD code = 0;
	GetExitCodeProcess(m_impl->process, &code);
	m_impl->exit_code = static_cast<int>(code);
	return m_impl->exit_code;
}

std::optional<int> ChildProcess::Poll() {
	return Wait(std::chrono::milliseconds {0});
}

void ChildProcess::Terminate() {
	if (m_impl->process == nullptr || m_impl->exit_code) {
		return;
	}
	TerminateProcess(m_impl->process, TERMINATED_EXIT_CODE);
	(void)Wait(TERMINATE_GRACE);
}

void ChildProcess::EnterChildMode(uint32_t parent_pid, ChildGuards guards) {
	// The class was set at spawn and the job is the death signal, so only the watchdog is armed here.
	if (!guards.watchdog) {
		return;
	}
	HANDLE parent = OpenProcess(SYNCHRONIZE, FALSE, parent_pid);
	if (parent == nullptr) {
		// No parent to watch means it is already gone, or the pid was wrong.
		TerminateProcess(GetCurrentProcess(), TERMINATED_EXIT_CODE);
		return;
	}
	std::thread(WatchParent, parent).detach();
}

#else

namespace {

// How often Wait looks again, as there is no handle to block on.
constexpr std::chrono::milliseconds POLL_INTERVAL {2};

// How often the watchdog checks that the parent is still there.
constexpr std::chrono::milliseconds WATCHDOG_INTERVAL {500};

// Added to the nice value of a child. The kernel clamps the result to 19.
constexpr int CHILD_NICE_INCREMENT = 10;
constexpr int MAX_NICE             = 19;

// Shell convention for a process that died from a signal.
constexpr int SIGNAL_EXIT_CODE_BASE = 128;

// Ends the process once the parent is gone. Re-parenting changes getppid, which also covers a
// parent that died before this thread started.
void WatchParent(pid_t parent) {
	while (getppid() == parent) {
		std::this_thread::sleep_for(WATCHDOG_INTERVAL);
	}
	_Exit(SIGNAL_EXIT_CODE_BASE + SIGKILL);
}

// Lowers the priority relative to the current one, so a parent that is already niced is not
// made higher than its child. A failure leaves the priority alone, which is harmless.
void LowerPriority() {
	errno             = 0;
	const int current = getpriority(PRIO_PROCESS, 0);
	if (current == -1 && errno != 0) {
		return;
	}
	setpriority(PRIO_PROCESS, 0, std::min(current + CHILD_NICE_INCREMENT, MAX_NICE));
}

// Asks the kernel to kill this process when the parent dies (Linux only). Returns false when
// the parent is already gone, which the kernel never reports. A prctl failure is tolerated
// because the watchdog covers it.
bool ArmDeathSignal(pid_t parent) {
#ifdef __linux__
	prctl(PR_SET_PDEATHSIG, SIGKILL);
#endif
	return getppid() == parent;
}

// Everything the child must not inherit from this process. Returns the spawn result.
int SpawnPosix(pid_t& pid, const char* program, char* const* argv) {
	posix_spawn_file_actions_t actions;
	posix_spawnattr_t          attributes;
	posix_spawn_file_actions_init(&actions);
	posix_spawnattr_init(&attributes);
	// No stdio to talk to, and no descriptor of ours left in the child where the system allows.
	posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
	posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
	posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
	short flags = POSIX_SPAWN_SETSIGMASK | POSIX_SPAWN_SETSIGDEF;
#ifdef __APPLE__
	flags |= POSIX_SPAWN_CLOEXEC_DEFAULT;
#elif defined(__GLIBC_PREREQ)
#if __GLIBC_PREREQ(2, 34)
	posix_spawn_file_actions_addclosefrom_np(&actions, STDERR_FILENO + 1);
#endif
	// Older libc cannot close the rest, so descriptors without close-on-exec stay open.
#endif
	// The caller may have signals blocked or ignored, and the child would inherit both. The two
	// signals that cannot be changed are left out of the reset set.
	sigset_t no_signals;
	sigset_t all_signals;
	sigemptyset(&no_signals);
	sigfillset(&all_signals);
	sigdelset(&all_signals, SIGKILL);
	sigdelset(&all_signals, SIGSTOP);
	posix_spawnattr_setsigmask(&attributes, &no_signals);
	posix_spawnattr_setsigdefault(&attributes, &all_signals);
	posix_spawnattr_setflags(&attributes, flags);
#ifdef __APPLE__
	char** environment = *_NSGetEnviron();
#else
	char** environment = environ;
#endif
	const int result = posix_spawn(&pid, program, &actions, &attributes, argv, environment);
	posix_spawnattr_destroy(&attributes);
	posix_spawn_file_actions_destroy(&actions);
	return result;
}

} // namespace

std::filesystem::path ExecutablePath() {
#ifdef __APPLE__
	std::array<char, MAC_EXECUTABLE_PATH_BUFFER> name {};
	auto                   size = static_cast<uint32_t>(name.size());
	_NSGetExecutablePath(name.data(), &size);
	return name.data();
#else
	std::error_code error;
	auto            resolved = std::filesystem::read_symlink("/proc/self/exe", error);
	return error ? std::filesystem::path("/proc/self/exe") : resolved;
#endif
}

std::filesystem::path ExecutableStatPath() {
#ifdef __APPLE__
	return ExecutablePath();
#else
	return "/proc/self/exe";
#endif
}

uint32_t CurrentProcessId() {
	return static_cast<uint32_t>(getpid());
}

struct ChildProcess::Impl {
	pid_t              pid = -1;
	std::optional<int> exit_code;
};

ChildProcess::ChildProcess(): m_impl(std::make_unique<Impl>()) {}

ChildProcess::~ChildProcess() {
	Terminate();
	// A child that ignored the kill for the whole grace period is still reaped, so no zombie
	// outlives this object.
	if (m_impl->pid > 0 && !m_impl->exit_code) {
		int status = 0;
		while (waitpid(m_impl->pid, &status, 0) < 0 && errno == EINTR) {}
	}
}

bool ChildProcess::Start(const std::filesystem::path& program, std::span<const std::string> arguments) {
	if (m_impl->pid > 0 || program.empty()) {
		return false;
	}
	std::vector<std::string> storage;
	storage.push_back(program.string());
	storage.insert(storage.end(), arguments.begin(), arguments.end());
	std::vector<char*> argv;
	for (auto& text: storage) {
		argv.push_back(text.data());
	}
	argv.push_back(nullptr);

	pid_t pid = -1;
	if (SpawnPosix(pid, program.c_str(), argv.data()) != 0) {
		return false;
	}
	m_impl->pid = pid;
	return true;
}

uint32_t ChildProcess::Pid() const {
	return m_impl->pid > 0 ? static_cast<uint32_t>(m_impl->pid) : 0;
}

std::optional<int> ChildProcess::Poll() {
	if (m_impl->exit_code || m_impl->pid <= 0) {
		return m_impl->exit_code;
	}
	int   status = 0;
	pid_t result = 0;
	do {
		result = waitpid(m_impl->pid, &status, WNOHANG);
	} while (result < 0 && errno == EINTR);
	if (result == 0) {
		return std::nullopt;
	}
	if (result < 0) {
		// Someone else reaped it, so the real code is lost.
		m_impl->exit_code = -1;
	} else if (WIFEXITED(status)) {
		m_impl->exit_code = WEXITSTATUS(status);
	} else {
		m_impl->exit_code = SIGNAL_EXIT_CODE_BASE + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
	}
	return m_impl->exit_code;
}

std::optional<int> ChildProcess::Wait(std::chrono::milliseconds timeout) {
	const auto deadline = std::chrono::steady_clock::now() + timeout;
	while (true) {
		if (auto code = Poll()) {
			return code;
		}
		if (std::chrono::steady_clock::now() >= deadline) {
			return std::nullopt;
		}
		std::this_thread::sleep_for(POLL_INTERVAL);
	}
}

void ChildProcess::Terminate() {
	if (m_impl->pid <= 0 || m_impl->exit_code) {
		return;
	}
	kill(m_impl->pid, SIGKILL);
	(void)Wait(TERMINATE_GRACE);
}

void ChildProcess::EnterChildMode(uint32_t parent_pid, ChildGuards guards) {
	if (guards.lower_priority) {
		LowerPriority();
	}
	if (guards.death_signal && !ArmDeathSignal(static_cast<pid_t>(parent_pid))) {
		_Exit(SIGNAL_EXIT_CODE_BASE + SIGKILL);
	}
	if (guards.watchdog) {
		std::thread(WatchParent, static_cast<pid_t>(parent_pid)).detach();
	}
}

#endif

} // namespace Common
