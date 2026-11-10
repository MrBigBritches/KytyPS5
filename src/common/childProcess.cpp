#include "common/childProcess.h"

#include "common/config.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <fstream>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
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

// What the system said about a finished child, before a memory measure is chosen.
struct CapturedUsage {
	double                  user_ms   = 0.0;
	double                  kernel_ms = 0.0;
	std::optional<uint64_t> working_set;
	std::optional<uint64_t> commit;
	std::optional<uint64_t> sampled;
};

// Chooses the memory figure. Auto prefers the figure the system measured by itself, then the
// commit, then the sample. A named choice returns that figure or none, never another one.
std::optional<PeakMemory> PickMemory(const CapturedUsage& captured, PeakMemorySelect select) {
	const auto make = [](const std::optional<uint64_t>& bytes,
	                     PeakMemorySource               source) -> std::optional<PeakMemory> {
		if (!bytes) {
			return std::nullopt;
		}
		return PeakMemory {.bytes = *bytes, .source = source};
	};
	switch (select) {
		case PeakMemorySelect::WorkingSet:
			return make(captured.working_set, PeakMemorySource::WorkingSet);
		case PeakMemorySelect::Commit: return make(captured.commit, PeakMemorySource::Commit);
		case PeakMemorySelect::WorkingSetSampled:
			return make(captured.sampled, PeakMemorySource::WorkingSetSampled);
		case PeakMemorySelect::Auto: break;
	}
	if (auto best = make(captured.working_set, PeakMemorySource::WorkingSet)) {
		return best;
	}
	if (auto best = make(captured.sampled, PeakMemorySource::WorkingSetSampled)) {
		return best;
	}
	return make(captured.commit, PeakMemorySource::Commit);
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

// Windows counts processor time in ticks of 100 ns, so ten thousand of them make a millisecond.
constexpr double TICKS_PER_MILLISECOND = 10000.0;

// The processor time of a FILETIME duration, in milliseconds.
double DurationMs(const FILETIME& time) {
	ULARGE_INTEGER ticks;
	ticks.LowPart  = time.dwLowDateTime;
	ticks.HighPart = time.dwHighDateTime;
	return static_cast<double>(ticks.QuadPart) / TICKS_PER_MILLISECOND;
}

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

bool ProcessIsRunning(uint32_t pid) {
	// Only the query right is asked for, which is the one that most processes allow.
	HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pid);
	if (process == nullptr) {
		return !WindowsOpenErrorMeansGone(GetLastError());
	}
	DWORD      code   = 0;
	const BOOL gotten = GetExitCodeProcess(process, &code);
	CloseHandle(process);
	// An exit code that could not be read says nothing, so the process counts as running. A handle
	// stays valid after the process ended, which is why the exit code is what tells.
	return !gotten || code == STILL_ACTIVE;
}

struct ChildProcess::Impl {
	HANDLE             process = nullptr;
	HANDLE             job     = nullptr;
	DWORD              pid     = 0;
	std::optional<int> exit_code;
	std::optional<CapturedUsage> usage;

	// Reads what the finished child used. The process handle is open here, and so is the job, which
	// keeps the totals of a member that has ended. Both memory figures are kept, so each source can
	// be asked for alone.
	void CaptureUsage() {
		CapturedUsage captured;
		bool          have_cpu = false;
		FILETIME      creation {};
		FILETIME      exited {};
		FILETIME      kernel {};
		FILETIME      user {};
		if (GetProcessTimes(process, &creation, &exited, &kernel, &user) != FALSE) {
			captured.user_ms   = DurationMs(user);
			captured.kernel_ms = DurationMs(kernel);
			have_cpu           = true;
		}
		PROCESS_MEMORY_COUNTERS memory {};
		if (GetProcessMemoryInfo(process, &memory, sizeof(memory)) != FALSE) {
			captured.working_set = memory.PeakWorkingSetSize;
		}
		JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits {};
		if (job != nullptr &&
		    QueryInformationJobObject(job, JobObjectExtendedLimitInformation, &limits,
		                              sizeof(limits), nullptr) != FALSE) {
			captured.commit = limits.PeakProcessMemoryUsed;
		}
		JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting {};
		if (!have_cpu && job != nullptr &&
		    QueryInformationJobObject(job, JobObjectBasicAccountingInformation, &accounting,
		                              sizeof(accounting), nullptr) != FALSE) {
			captured.user_ms =
			    static_cast<double>(accounting.TotalUserTime.QuadPart) / TICKS_PER_MILLISECOND;
			captured.kernel_ms =
			    static_cast<double>(accounting.TotalKernelTime.QuadPart) / TICKS_PER_MILLISECOND;
			have_cpu = true;
		}
		if (have_cpu || captured.working_set || captured.commit) {
			usage = captured;
		}
	}

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
	m_impl->CaptureUsage();
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

#ifdef __APPLE__
// What one unit of ru_maxrss is, in bytes. macOS counts bytes. Linux counts kilobytes, but its
// ru_maxrss is not used (see PeakMemorySource::WorkingSetSampled), so only macOS reads it.
constexpr uint64_t RU_MAXRSS_UNIT_BYTES = 1;
#endif
// The unit of the VmHWM line in /proc/<pid>/status, which is kilobytes.
[[maybe_unused]] constexpr uint64_t PROC_STATUS_UNIT_BYTES = 1024;

constexpr double MS_PER_SECOND       = 1000.0;
constexpr double MICROSECONDS_PER_MS = 1000.0;

// The processor time of a timeval, in milliseconds.
double DurationMs(const timeval& time) {
	return (static_cast<double>(time.tv_sec) * MS_PER_SECOND) +
	       (static_cast<double>(time.tv_usec) / MICROSECONDS_PER_MS);
}

#ifdef __linux__
// The high-water mark of the resident memory of a live process, from its status file, or nothing
// when the file or the line is not there (the process ended, or the kernel does not say).
std::optional<uint64_t> ReadPeakResidentBytes(pid_t pid) {
	std::ifstream     status("/proc/" + std::to_string(pid) + "/status");
	std::string       line;
	const std::string key = "VmHWM:";
	while (std::getline(status, line)) {
		if (!line.starts_with(key)) {
			continue;
		}
		// from_chars, because exceptions are off in the POSIX builds. It stops at " kB".
		const char* begin = line.data() + key.size();
		const char* end   = line.data() + line.size();
		while (begin != end && (*begin == ' ' || *begin == '\t')) {
			++begin;
		}
		uint64_t   kilobytes = 0;
		const auto parsed    = std::from_chars(begin, end, kilobytes);
		if (parsed.ec != std::errc {} || parsed.ptr == begin) {
			return std::nullopt;
		}
		return kilobytes * PROC_STATUS_UNIT_BYTES;
	}
	return std::nullopt;
}
#endif

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

bool ProcessIsRunning(uint32_t pid) {
	// An id above the range of pid_t would wrap to a negative one, and kill would then signal a
	// process group. Nothing is gone that cannot be asked about.
	if (pid > static_cast<uint32_t>(INT32_MAX)) {
		return true;
	}
	if (kill(static_cast<pid_t>(pid), 0) == 0) {
		return true;
	}
	return !PosixKillErrorMeansGone(errno);
}

struct ChildProcess::Impl {
	pid_t                        pid = -1;
	std::optional<int>           exit_code;
	std::optional<CapturedUsage> usage;
	// The largest sample of the child's resident peak, taken while it ran (Linux).
	std::optional<uint64_t> sampled_peak;

	// Takes one sample of the live child. A smaller value than before is ignored, since the
	// high-water mark only grows and a smaller read would be a different process.
	void SamplePeak() {
#ifdef __linux__
		if (const auto bytes = ReadPeakResidentBytes(pid)) {
			sampled_peak = std::max(sampled_peak.value_or(0), *bytes);
		}
#endif
	}

	// Keeps what the kernel said the child used, from the wait4 that reaped it. The processor
	// times are the kernel's. The memory is ru_maxrss on macOS only, and the samples on Linux.
	void CaptureUsage(const rusage& used) {
		CapturedUsage captured;
		captured.user_ms   = DurationMs(used.ru_utime);
		captured.kernel_ms = DurationMs(used.ru_stime);
#ifdef __APPLE__
		captured.working_set = static_cast<uint64_t>(used.ru_maxrss) * RU_MAXRSS_UNIT_BYTES;
#else
		captured.sampled = sampled_peak;
#endif
		usage = captured;
	}
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
	// Sampled before the wait, while the process can still be read. One that ends in between keeps
	// the previous sample.
	m_impl->SamplePeak();
	int    status = 0;
	pid_t  result = 0;
	rusage used {};
	do {
		result = wait4(m_impl->pid, &status, WNOHANG, &used);
	} while (result < 0 && errno == EINTR);
	if (result == 0) {
		return std::nullopt;
	}
	if (result < 0) {
		// Someone else reaped it, so the real code is lost.
		m_impl->exit_code = -1;
	} else {
		if (WIFEXITED(status)) {
			m_impl->exit_code = WEXITSTATUS(status);
		} else {
			m_impl->exit_code =
			    SIGNAL_EXIT_CODE_BASE + (WIFSIGNALED(status) ? WTERMSIG(status) : 0);
		}
		// The rusage belongs to this child only when wait4 returned its pid.
		if (result == m_impl->pid) {
			m_impl->CaptureUsage(used);
		}
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

std::optional<ChildUsage> ChildProcess::Usage(PeakMemorySelect select) const {
	if (!m_impl->usage) {
		return std::nullopt;
	}
	ChildUsage usage;
	usage.cpu_user_ms   = m_impl->usage->user_ms;
	usage.cpu_kernel_ms = m_impl->usage->kernel_ms;
	usage.peak          = PickMemory(*m_impl->usage, select);
	return usage;
}

} // namespace Common
