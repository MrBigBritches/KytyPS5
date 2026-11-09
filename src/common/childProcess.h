#ifndef KYTY_COMMON_CHILD_PROCESS_H_
#define KYTY_COMMON_CHILD_PROCESS_H_

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

// Runs the emulator's own binary as a low priority helper that cannot outlive its parent.
// Output travels through files, so the child gets no pipes, no console and no inherited handles.
namespace Common {

/// Absolute path of the running executable, suitable for a spawn call. Linux resolves the
/// /proc/self/exe link. Windows and macOS return what the system reports, and macOS does not
/// resolve symlinks. Empty when the operating system cannot tell.
[[nodiscard]] std::filesystem::path ExecutablePath();

/// A path whose last write time is the running executable's, even after a rebuild replaced the
/// file on disk. Linux keeps the replaced file reachable through /proc/self/exe only. Elsewhere
/// this is ExecutablePath().
[[nodiscard]] std::filesystem::path ExecutableStatPath();

/// Process id of the caller, in the form the child side needs for EnterChildMode.
[[nodiscard]] uint32_t CurrentProcessId();

/// Quotes one argument so CommandLineToArgvW and the Microsoft C runtime parse it back unchanged.
[[nodiscard]] std::string QuoteWindowsArgument(std::string_view argument);

/// The protections a child arms on itself in EnterChildMode. Each can be turned off alone so a
/// test can prove the others are not what ends the child. Production code uses the defaults.
struct ChildGuards {
	bool lower_priority = true; ///< POSIX only. Windows sets the class at spawn.
	bool death_signal   = true; ///< Linux PR_SET_PDEATHSIG, plus a check that the parent is alive.
	bool watchdog       = true; ///< A thread that ends the process once the parent is gone.
};

/// One spawned helper process. Not thread safe: one owner thread calls every method. On POSIX
/// the pid can be reused by an unrelated process once Poll or Wait has returned a code, so do
/// not signal a pid taken from Pid() after that. Program and arguments are UTF-8. The
/// destructor terminates a child that is still running and releases every operating system
/// handle, so a ChildProcess cannot leak a zombie.
class ChildProcess {
public:
	ChildProcess();
	~ChildProcess();
	ChildProcess(const ChildProcess&)            = delete;
	ChildProcess& operator=(const ChildProcess&) = delete;
	ChildProcess(ChildProcess&&)                 = delete;
	ChildProcess& operator=(ChildProcess&&)      = delete;

	/// Starts program with the given arguments (argv[0] is program itself). The child runs at
	/// below normal priority (idle when this process is idle) without a window and without
	/// inherited handles. POSIX children also get /dev/null for stdio, default signal
	/// dispositions, and, where the system supports it, every other descriptor closed. On
	/// Windows it is also bound to a job that kills it when this process goes away. Returns
	/// false when the spawn failed or when a child is already started.
	///
	/// On Linux the death signal armed by EnterChildMode follows the thread that called Start,
	/// so call Start from a thread that lives as long as the child is wanted.
	[[nodiscard]] bool Start(const std::filesystem::path& program, std::span<const std::string> arguments);

	/// Process id of the child, or 0 before a successful Start.
	[[nodiscard]] uint32_t Pid() const;

	/// Returns the exit code when the child has finished, without blocking. A child killed by a
	/// signal reports 128 plus the signal number. POSIX exit codes are limited to 8 bits.
	/// On Windows a child that Terminate killed reports 1, which cannot be told apart from a
	/// child that exited with 1 by itself.
	[[nodiscard]] std::optional<int> Poll();

	/// Waits up to timeout for the child to finish and returns its exit code if it did.
	[[nodiscard]] std::optional<int> Wait(std::chrono::milliseconds timeout);

	/// Kills the child if it is still running and waits briefly for the system to reap it.
	void Terminate();

	/// Called by the child itself, first thing in its worker mode and before it creates any
	/// thread, because Linux nice values and the death signal are per thread. Lowers its
	/// priority on POSIX, arms PR_SET_PDEATHSIG on Linux, and starts a thread that ends the
	/// process once the parent is gone. The watchdog is the only protection on macOS. Failures
	/// of setpriority and prctl are tolerated, since the watchdog still covers the parent dying.
	static void EnterChildMode(uint32_t parent_pid, ChildGuards guards = {});

private:
	struct Impl;
	std::unique_ptr<Impl> m_impl;
};

} // namespace Common

#endif /* KYTY_COMMON_CHILD_PROCESS_H_ */
