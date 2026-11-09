#ifndef GRAPHICS_SHADER_STOREGEN_PROGRESS_LOG_H
#define GRAPHICS_SHADER_STOREGEN_PROGRESS_LOG_H

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Reads the progress log of a shader store worker, which the supervisor in another process uses to
// blame a crash or a hang on the program the worker was at. The log is one text line for each step,
// written by the worker as shaderStoreWorker.h describes. It needs no graphics device.
namespace Libs::Graphics::StoreGen {

/// What a progress line says.
enum class ProgressKind : uint8_t {
	Predict, ///< `predict <hash>`, before a program is predicted.
	Order,   ///< `order <count> <hash>`, once the candidate ordering is known.
	Start,   ///< `start <rank> <hash>`, before a candidate is translated.
	Drop,    ///< `drop <rank> <hash> <reason>`, when a program's candidates are given up.
	Summary, ///< `summary key=value ...`, the last line of a run. Keep it the last kind.
};

/// How many kinds there are, which sizes the reader's counts.
constexpr size_t PROGRESS_KIND_COUNT = static_cast<size_t>(ProgressKind::Summary) + 1;
static_assert(ProgressKind::Predict < ProgressKind::Order &&
                  ProgressKind::Order < ProgressKind::Start &&
                  ProgressKind::Start < ProgressKind::Drop &&
                  ProgressKind::Drop < ProgressKind::Summary,
              "Summary must be the last kind, since it sizes the counts");

/// Why a `drop` line was written.
enum class ProgressDropReason : uint8_t {
	PushStart,     ///< `push_start predicted=<n> actual=<n>`
	ResourceCount, ///< `resource_count`
	FrameTooLarge, ///< `frame_too_large`, which drops one candidate and not its program.
};

/// One parsed line. Only the fields of its kind mean anything.
struct ProgressLine {
	ProgressKind kind = ProgressKind::Predict;
	/// The program's code hash. Predict, Start, Drop, and Order (where it is the order hash).
	uint64_t hash = 0;
	/// The candidate rank for Start and Drop, and the number of candidates for Order.
	uint64_t rank_or_count = 0;
	/// Drop only.
	ProgressDropReason reason = ProgressDropReason::PushStart;
	/// Drop with PushStart only, the push data start the predictor assumed and the module has.
	uint32_t predicted_push = 0;
	uint32_t actual_push    = 0;
	/// Summary only. `exit=` and the rest of the fields in the order they were written, as text.
	int                                              exit_code = 0;
	std::vector<std::pair<std::string, std::string>> fields;

	bool operator==(const ProgressLine&) const = default;
};

/// Reads one complete line, without its newline.
/// @return nothing when the line is not one the worker writes: an unknown word, a missing or extra
///         field, a number that is not a number, a hash that is not 16 hex digits, or a reason that
///         is not one of the worker's.
[[nodiscard]] std::optional<ProgressLine> ParseProgressLine(std::string_view line);

/// The longest line a reader holds while it waits for the newline. The worker's longest line is the
/// summary, a few hundred bytes, so a longer one is not a line of the worker's.
constexpr size_t MAX_PROGRESS_LINE_BYTES = 16 * 1024;

/// The most bytes one Poll reads, which leaves the rest for the next one.
constexpr uint64_t MAX_PROGRESS_POLL_BYTES = 1024 * 1024;

/// The clock that a reader is given its times from.
using ProgressClock = std::chrono::steady_clock;

/// Follows a worker's progress log for a supervisor. The log is appended across the spawns of one
/// run, so a reader counts only what a spawn appended: BeginSpawn names where that spawn's lines
/// start. What the supervisor needs across spawns, such as the latest `order` line and whether any
/// `start` line was ever seen, is kept apart from what belongs to one spawn.
///
/// One thread owns a reader, like the supervisor's other pieces. The file is opened for each Poll
/// and closed again, so the worker's writes are never blocked. The reader takes its times from the
/// caller, so a test needs no real clock.
class ProgressLogReader {
public:
	/// @param path the log, which need not exist yet.
	explicit ProgressLogReader(std::filesystem::path path);

	/// Starts counting a spawn. Its lines are the ones from the log's present size on, which is the
	/// offset a supervisor records before it spawns. A log that is not there yet starts at zero.
	/// Call it after the previous child was reaped, since a child still writing would put lines on
	/// either side of the offset. The caller must also have caught up first, with PollToEnd, since
	/// the jump skips what was not read yet, and the dead child's last line is the one that blames
	/// a crash.
	/// @param now when the spawn starts, which is what LastLineTime gives until a line arrives.
	void BeginSpawn(ProgressClock::time_point now);

	/// Starts a run again from nothing, as the supervisor does for a fresh restart. What BeginSpawn
	/// keeps across spawns, which is the latest `order` line and the last `start` line, belongs to
	/// the ordering that the fresh run throws away, so it is cleared too. The same advice about
	/// catching up holds.
	void BeginFreshRun(ProgressClock::time_point now);

	/// Starts counting a spawn at a given offset of the log. For a supervisor that saved the
	/// offset, and for tests.
	void BeginSpawnAt(uint64_t offset, ProgressClock::time_point now);

	/// Reads what was appended since the last Poll and parses the complete lines. A last line with
	/// no newline yet is held until the next Poll finds it complete. A missing log means nothing
	/// yet.
	/// @param now when this Poll runs, taken as the time of every line that arrives in it.
	/// @return how many lines of the worker's it found.
	size_t Poll(ProgressClock::time_point now);

	/// Polls until the log is read to its end, which a log bigger than one Poll reads needs. It
	/// stops when a poll reached the size it saw, so a log that still grows is read to where it was
	/// when the last poll began. It also stops when a poll made no progress, meaning the offset did
	/// not move and the log was not reset, as when the file cannot be read on, so it never spins.
	/// @return how many lines of the worker's it found.
	size_t PollToEnd(ProgressClock::time_point now);

	/// Whether the last Poll read up to the size of the log it saw.
	[[nodiscard]] bool CaughtUp() const { return m_caught_up; }

	/// How many times the log was found shorter than what was read, which means another file took
	/// its place. It is then read from its start.
	[[nodiscard]] size_t LogResets() const { return m_resets; }

	/// How many lines of a kind were found since BeginSpawn.
	[[nodiscard]] size_t KindCount(ProgressKind kind) const {
		return m_kind_counts[static_cast<size_t>(kind)];
	}

	/// Where the next Poll reads from.
	[[nodiscard]] uint64_t Offset() const { return m_offset; }

	/// Lines of the worker's found since BeginSpawn.
	[[nodiscard]] size_t LinesSinceSpawn() const { return m_lines_since_spawn; }
	/// The last of them.
	[[nodiscard]] const std::optional<ProgressLine>& LastLine() const { return m_last_line; }
	/// When the last of them arrived, or when the spawn began while there is none. A supervisor
	/// compares it with its timeouts.
	[[nodiscard]] ProgressClock::time_point LastLineTime() const { return m_last_line_time; }
	/// Whether a `start` line came since BeginSpawn.
	[[nodiscard]] bool StartSeenSinceSpawn() const { return m_start_since_spawn; }
	/// Whether a `start` line ever came, in any spawn. A skipped program after that changes the
	/// ranks, so it matters to the restart contract.
	[[nodiscard]] bool AnyStartSeen() const { return m_last_start.has_value(); }
	/// The last `start` line of the whole log.
	[[nodiscard]] const std::optional<ProgressLine>& LastStart() const { return m_last_start; }
	/// The `drop` lines since BeginSpawn, in order.
	[[nodiscard]] const std::vector<ProgressLine>& DropsSinceSpawn() const { return m_drops; }
	/// The latest `order` line of the whole log.
	[[nodiscard]] const std::optional<ProgressLine>& LatestOrder() const { return m_order; }
	/// The `summary` line of this spawn, when it arrived.
	[[nodiscard]] const std::optional<ProgressLine>& Summary() const { return m_summary; }
	/// Lines that were complete but not a line of the worker's, since the reader was made.
	[[nodiscard]] size_t MalformedLines() const { return m_malformed; }

private:
	void TakeLine(std::string_view line, ProgressClock::time_point now);

	std::filesystem::path m_path;
	uint64_t              m_offset = 0;
	/// The bytes of a last line that has no newline yet.
	std::string m_partial;
	/// Whether the bytes up to the next newline belong to a line that was too long.
	bool m_skipping = false;

	size_t                      m_lines_since_spawn = 0;
	std::optional<ProgressLine> m_last_line;
	ProgressClock::time_point   m_last_line_time {};
	bool                        m_start_since_spawn = false;
	std::vector<ProgressLine>   m_drops;
	std::optional<ProgressLine> m_summary;
	std::optional<ProgressLine> m_order;
	std::optional<ProgressLine> m_last_start;
	size_t                      m_malformed = 0;
	size_t                      m_resets    = 0;
	bool                        m_caught_up = true;
	size_t                      m_kind_counts[PROGRESS_KIND_COUNT] {};
};

/// Where a reader stands, for judging whether a poll made progress. The offset and the number of
/// resets together change whenever a poll read anything or found the log replaced.
using ProgressMark = std::pair<uint64_t, size_t>;

/// The loop of PollToEnd, apart from the reader so that a test can give it a poll that is stuck. It
/// calls `poll` until `caught_up` says so, or until a call leaves `mark` as it was.
/// @return the sum of what `poll` returned.
size_t PollWhileProgressing(const std::function<size_t()>&       poll,
                            const std::function<bool()>&         caught_up,
                            const std::function<ProgressMark()>& mark);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_PROGRESS_LOG_H
