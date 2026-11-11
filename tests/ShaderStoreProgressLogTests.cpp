// Tests the reader of a shader store worker's progress log. The table of lines here is written by
// hand, so it pins what the parser takes each line to mean, and it cannot notice that the worker's
// own format changed. ShaderStoreDispatchTests does that, by reading the logs of real worker runs.
// Each check names what else could satisfy it and compares against a baseline that differs in one
// change.
#include "common/childProcess.h"
#include "graphics/shader/storegen/progressLog.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace Libs::Graphics::StoreGen;
using namespace std::chrono_literals;
namespace fs = std::filesystem;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

// Hashes and ranks that differ from one another, so a parser that mixes two fields up fails.
constexpr uint64_t HASH_A    = 0x0123456789abcdefull;
constexpr uint64_t HASH_B    = 0xfedcba9876543210ull;
constexpr uint64_t HASH_C    = 0x00000000000000a5ull;
constexpr uint64_t ORDER_ONE = 0x1111222233334444ull;
// The id a dropped head carries.
constexpr uint64_t HEAD_ID   = 0x0123456789abcdefull;
constexpr uint64_t ORDER_TWO = 0x5555666677778888ull;
constexpr uint64_t RANK_FIVE = 5;
constexpr uint64_t RANK_NINE = 9;
constexpr uint64_t COUNT_7   = 7;
constexpr uint32_t PREDICTED = 3;
constexpr uint32_t ACTUAL    = 4;
// Longer than the longest line a reader holds, and than the chunks it reads in, so the line is cut
// across several reads.
constexpr size_t TOO_LONG_BYTES = 200 * 1024;
// Enough lines of 25 bytes to be more than one poll reads.
constexpr size_t LARGE_LOG_LINES = 50000;

// The hash as the worker writes it, 16 hex digits.
std::string Hex(uint64_t value) {
	char text[20];
	std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
	return text;
}

// A directory for the files of one test, which is removed with it.
class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		m_path             = fs::temp_directory_path() /
		         ("kyty_progress_log_tests_" + std::to_string(Common::CurrentProcessId()) + "_" +
		          std::to_string(counter++));
		fs::remove_all(m_path);
		fs::create_directories(m_path);
	}
	~TempDirectory() {
		std::error_code ignored;
		fs::remove_all(m_path, ignored);
	}
	TempDirectory(const TempDirectory&)            = delete;
	TempDirectory& operator=(const TempDirectory&) = delete;

	fs::path operator/(const std::string& name) const { return m_path / name; }

private:
	fs::path m_path;
};

void AppendText(const fs::path& path, std::string_view text) {
	std::ofstream out(path, std::ios::binary | std::ios::app);
	out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

uint64_t SizeOf(const fs::path& path) {
	std::error_code error;
	const auto      size = fs::file_size(path, error);
	return error ? 0 : size;
}

// The first time of each test, and a step after it.
const ProgressClock::time_point T0 = ProgressClock::time_point {} + 1000s;
const ProgressClock::time_point T1 = T0 + 10s;
const ProgressClock::time_point T2 = T0 + 20s;
const ProgressClock::time_point T3 = T0 + 30s;

ProgressLine Predict(uint64_t hash) {
	ProgressLine line;
	line.kind = ProgressKind::Predict;
	line.hash = hash;
	return line;
}

ProgressLine WithRank(ProgressKind kind, uint64_t rank_or_count, uint64_t hash) {
	ProgressLine line;
	line.kind          = kind;
	line.rank_or_count = rank_or_count;
	line.hash          = hash;
	return line;
}

ProgressLine Prediction(uint64_t hash, PredictionOutcome outcome, uint64_t candidates) {
	auto line    = WithRank(ProgressKind::Prediction, candidates, hash);
	line.outcome = outcome;
	return line;
}

ProgressLine Drop(uint64_t rank, uint64_t hash, ProgressDropReason reason) {
	auto line   = WithRank(ProgressKind::Drop, rank, hash);
	line.reason = reason;
	return line;
}

// A row of the table of lines: the text and what it parses to.
struct LineCase {
	std::string  text;
	ProgressLine expected;
};

// Every kind of line the worker writes, with different values in each, parses to exactly its
// fields. A parser that returned the right kind with the wrong field, or filled a field from the
// wrong word, fails the comparison of the whole struct.
void TestEveryLineKind() {
	auto push           = Drop(RANK_FIVE, HASH_B, ProgressDropReason::PushStart);
	push.predicted_push = PREDICTED;
	push.actual_push    = ACTUAL;
	// The same line with the head word is the same drop with another reason, and so is the same
	// line with its id, which only the head's line carries.
	auto head              = push;
	head.reason            = ProgressDropReason::PushStartHead;
	auto head_with_id      = head;
	head_with_id.module_id = HEAD_ID;
	ProgressLine summary;
	summary.kind      = ProgressKind::Summary;
	summary.exit_code = 0;
	summary.fields    = {
        {"exit", "0"}, {"scanned", "12"}, {"worst_ms", "340"}, {"order", Hex(ORDER_ONE)}};

	const std::vector<LineCase> cases {
	    {"predict " + Hex(HASH_A), Predict(HASH_A)},
	    {"start 5 " + Hex(HASH_B), WithRank(ProgressKind::Start, RANK_FIVE, HASH_B)},
	    {"order 7 " + Hex(ORDER_ONE), WithRank(ProgressKind::Order, COUNT_7, ORDER_ONE)},
	    {"drop 5 " + Hex(HASH_B) + " push_start predicted=3 actual=4", push},
	    {"drop 5 " + Hex(HASH_B) + " push_start_head predicted=3 actual=4 id=" + Hex(HEAD_ID),
	     head_with_id},
	    {"drop 9 " + Hex(HASH_C) + " resource_count",
	     Drop(RANK_NINE, HASH_C, ProgressDropReason::ResourceCount)},
	    {"drop 9 " + Hex(HASH_C) + " frame_too_large",
	     Drop(RANK_NINE, HASH_C, ProgressDropReason::FrameTooLarge)},
	    {"summary exit=0 scanned=12 worst_ms=340 order=" + Hex(ORDER_ONE), summary},
	    {"prediction " + Hex(HASH_A) + " predicted 7",
	     Prediction(HASH_A, PredictionOutcome::Predicted, COUNT_7)},
	    {"prediction " + Hex(HASH_B) + " calls 0", Prediction(HASH_B, PredictionOutcome::Calls, 0)},
	    {"prediction " + Hex(HASH_C) + " unclassified 0",
	     Prediction(HASH_C, PredictionOutcome::Unclassified, 0)},
	    {"prediction " + Hex(HASH_A) + " unreadable 0",
	     Prediction(HASH_A, PredictionOutcome::Unreadable, 0)},
	    {"prediction " + Hex(HASH_B) + " problems 0",
	     Prediction(HASH_B, PredictionOutcome::Problems, 0)},
	};
	for (const auto& row: cases) {
		const auto parsed = ParseProgressLine(row.text);
		CHECK(parsed.has_value() && *parsed == row.expected);
	}
}

// A malformed line next to the one change that makes it a line of the worker's. The twin parses and
// the bad one does not, so the line is refused for the one thing that differs and not for
// something else in it.
void TestMalformedLines() {
	struct BadCase {
		std::string bad;
		std::string good;
	};
	const auto                 hash = Hex(HASH_A);
	const std::vector<BadCase> cases {
	    {"", "predict " + hash},
	    {"bogus " + hash, "predict " + hash},
	    {"predict", "predict " + hash},
	    {"predict " + hash.substr(0, 15), "predict " + hash},
	    {"predict " + hash + "0", "predict " + hash},
	    {"predict " + hash + " extra", "predict " + hash},
	    {" predict " + hash, "predict " + hash},
	    {"predict  " + hash, "predict " + hash},
	    {"predict " + std::string(16, 'g'), "predict " + hash},
	    {"start x " + hash, "start 1 " + hash},
	    {"start -1 " + hash, "start 1 " + hash},
	    {"start 1 " + hash + " extra", "start 1 " + hash},
	    {"start 1", "start 1 " + hash},
	    {"order 7", "order 7 " + hash},
	    {"drop 1 " + hash, "drop 1 " + hash + " resource_count"},
	    {"drop 1 " + hash + " weird", "drop 1 " + hash + " resource_count"},
	    {"drop 1 " + hash + " resource_count extra", "drop 1 " + hash + " resource_count"},
	    {"drop 1 " + hash + " frame_too_large extra", "drop 1 " + hash + " frame_too_large"},
	    {"drop 1 " + hash + " push_start", "drop 1 " + hash + " push_start predicted=1 actual=2"},
	    {"drop 1 " + hash + " push_start predicted=1",
	     "drop 1 " + hash + " push_start predicted=1 actual=2"},
	    {"drop 1 " + hash + " push_start predicted=x actual=2",
	     "drop 1 " + hash + " push_start predicted=1 actual=2"},
	    {"drop 1 " + hash + " push_start actual=2 predicted=1",
	     "drop 1 " + hash + " push_start predicted=1 actual=2"},
	    {"drop 1 " + hash + " push_start_head predicted=1 actual=2",
	     "drop 1 " + hash + " push_start_head predicted=1 actual=2 id=" + hash},
	    {"drop 1 " + hash + " push_start_head predicted=1 actual=2 id=zz",
	     "drop 1 " + hash + " push_start_head predicted=1 actual=2 id=" + hash},
	    {"drop 1 " + hash + " push_start predicted=1 actual=2 id=" + hash,
	     "drop 1 " + hash + " push_start predicted=1 actual=2"},
	    {"prediction " + hash + " predicted", "prediction " + hash + " predicted 2"},
	    {"prediction " + hash + " 2", "prediction " + hash + " predicted 2"},
	    {"prediction predicted 2", "prediction " + hash + " predicted 2"},
	    {"prediction " + hash + " maybe 2", "prediction " + hash + " predicted 2"},
	    {"prediction " + hash + " Predicted 2", "prediction " + hash + " predicted 2"},
	    {"prediction " + hash + " calls two", "prediction " + hash + " calls 2"},
	    {"prediction " + hash + " unclassified -1", "prediction " + hash + " unclassified 1"},
	    {"prediction " + hash.substr(0, 15) + " unreadable 0",
	     "prediction " + hash + " unreadable 0"},
	    {"prediction " + hash + " problems 0 extra", "prediction " + hash + " problems 0"},
	    {"prediction " + hash + " problems 0x", "prediction " + hash + " problems 0"},
	    {"summary", "summary exit=0"},
	    {"summary exit=x", "summary exit=0"},
	    {"summary scanned=1", "summary exit=0 scanned=1"},
	    {"summary exit=0 oops", "summary exit=0 oops=1"},
	};
	for (const auto& row: cases) {
		CHECK(!ParseProgressLine(row.bad).has_value());
		CHECK(ParseProgressLine(row.good).has_value());
	}

	// Through a reader: bad lines are counted and skipped, the good ones around them are read, and
	// a line of binary junk and one far too long for the worker never crash it.
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	std::string   text = "predict " + hash + "\n";
	text += "not a line\n";
	text += std::string("\x01\x02\xff\xfe junk\n", 10);
	text += std::string(TOO_LONG_BYTES, 'x') + "\n";
	text += "start 3 " + hash + "\n";
	AppendText(path, text);
	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	CHECK(reader.Poll(T1) == 2);
	CHECK(reader.LinesSinceSpawn() == 2);
	CHECK(reader.MalformedLines() == 3);
	CHECK(reader.LastLine().has_value() && reader.LastLine()->kind == ProgressKind::Start);
	CHECK(reader.LastLine()->rank_or_count == 3);
}

// A last line with no newline is held and is not a line until the newline arrives. The baseline is
// the same bytes written whole, which gives the line at the first poll.
void TestPartialLineIsHeld() {
	TempDirectory directory;
	const auto    path  = directory / "log.txt";
	const auto    start = "start 5 " + Hex(HASH_B);
	AppendText(path, "predict " + Hex(HASH_A) + "\n");
	AppendText(path, start.substr(0, 3));
	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	CHECK(reader.Poll(T1) == 1);
	CHECK(reader.LastLine() && reader.LastLine()->kind == ProgressKind::Predict);
	CHECK(!reader.StartSeenSinceSpawn());
	// Cut mid-way through a token, and then at the end of the whole line but before its newline.
	AppendText(path, start.substr(3, 6));
	CHECK(reader.Poll(T2) == 0);
	CHECK(reader.LinesSinceSpawn() == 1 && !reader.StartSeenSinceSpawn());
	AppendText(path, start.substr(9));
	CHECK(reader.Poll(T2) == 0);
	CHECK(!reader.StartSeenSinceSpawn());
	AppendText(path, "\n");
	CHECK(reader.Poll(T3) == 1);
	CHECK(reader.StartSeenSinceSpawn());
	CHECK(reader.LastLine() &&
	      *reader.LastLine() == WithRank(ProgressKind::Start, RANK_FIVE, HASH_B));
	// All the bytes were read by the polls, so none is read twice.
	CHECK(reader.Offset() == SizeOf(path));
	CHECK(reader.Poll(T3) == 0 && reader.LinesSinceSpawn() == 2);

	// The same bytes in one piece are a line at the first poll.
	const auto whole = directory / "whole.txt";
	AppendText(whole, "predict " + Hex(HASH_A) + "\n" + start + "\n");
	ProgressLogReader whole_reader(whole);
	whole_reader.BeginSpawnAt(0, T0);
	CHECK(whole_reader.Poll(T1) == 2);

	// A carriage return before the newline is not part of the line.
	const auto crlf = directory / "crlf.txt";
	AppendText(crlf, start + "\r\n");
	ProgressLogReader crlf_reader(crlf);
	crlf_reader.BeginSpawnAt(0, T0);
	CHECK(crlf_reader.Poll(T1) == 1 && crlf_reader.MalformedLines() == 0);
}

// A spawn counts only the lines appended after its offset. The baseline is a reader that begins at
// offset 0 on the same log, which counts every line.
void TestPerSpawnOffset() {
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	AppendText(path, "predict " + Hex(HASH_A) + "\norder 7 " + Hex(ORDER_ONE) + "\n");
	AppendText(path, "start 0 " + Hex(HASH_A) + "\n");
	AppendText(path, "drop 0 " + Hex(HASH_A) + " resource_count\n");
	AppendText(path, "start 1 " + Hex(HASH_B) + "\n");

	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	CHECK(reader.Poll(T1) == 5);
	CHECK(reader.AnyStartSeen() && reader.StartSeenSinceSpawn());
	CHECK(reader.DropsSinceSpawn().size() == 1);

	// The first child crashed in the middle of writing a line. The second spawn begins after it.
	AppendText(path, "start 2 " + Hex(HASH_C).substr(0, 5));
	reader.BeginSpawn(T2);
	CHECK(reader.Offset() == SizeOf(path));
	AppendText(path, "predict " + Hex(HASH_C) + "\n");
	CHECK(reader.Poll(T3) == 1);
	// What belongs to this spawn is only the new line.
	CHECK(reader.LinesSinceSpawn() == 1);
	CHECK(reader.LastLine() && *reader.LastLine() == Predict(HASH_C));
	CHECK(!reader.StartSeenSinceSpawn());
	CHECK(reader.DropsSinceSpawn().empty());
	CHECK(!reader.Summary().has_value());
	CHECK(reader.MalformedLines() == 0);
	// What the whole log has taught is kept.
	CHECK(reader.AnyStartSeen());
	CHECK(reader.LastStart() && reader.LastStart()->rank_or_count == 1 &&
	      reader.LastStart()->hash == HASH_B);
	CHECK(reader.LatestOrder() && reader.LatestOrder()->hash == ORDER_ONE);

	// The baseline: the same log read from the start of it counts the five lines of the first
	// spawn, and the torn line and the new spawn's first line together are one line that is not the
	// worker's.
	ProgressLogReader everything(path);
	everything.BeginSpawnAt(0, T0);
	CHECK(everything.Poll(T1) == 5);
	CHECK(everything.MalformedLines() == 1);
	CHECK(everything.DropsSinceSpawn().size() == 1);
}

// Each reason of a drop line is read as its own, with its own numbers, and only the drops of this
// spawn are given. frame_too_large is told apart from the other two, since a supervisor leaves it
// out of the failed hashes.
void TestDropReasons() {
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	AppendText(path, "drop 1 " + Hex(HASH_A) + " resource_count\n");
	ProgressLogReader reader(path);
	reader.BeginSpawn(T0);
	CHECK(reader.DropsSinceSpawn().empty());
	AppendText(path, "drop 2 " + Hex(HASH_A) + " push_start predicted=3 actual=4\n");
	AppendText(path, "drop 3 " + Hex(HASH_B) + " resource_count\n");
	AppendText(path, "drop 4 " + Hex(HASH_C) + " frame_too_large\n");
	CHECK(reader.Poll(T1) == 3);
	const auto& drops = reader.DropsSinceSpawn();
	CHECK(drops.size() == 3);
	if (drops.size() == 3) {
		CHECK(drops[0].reason == ProgressDropReason::PushStart && drops[0].rank_or_count == 2);
		CHECK(drops[0].hash == HASH_A && drops[0].predicted_push == PREDICTED &&
		      drops[0].actual_push == ACTUAL);
		CHECK(drops[1].reason == ProgressDropReason::ResourceCount && drops[1].hash == HASH_B &&
		      drops[1].rank_or_count == 3);
		CHECK(drops[2].reason == ProgressDropReason::FrameTooLarge && drops[2].hash == HASH_C &&
		      drops[2].rank_or_count == 4);
	}
	// A new spawn does not carry them.
	reader.BeginSpawn(T2);
	CHECK(reader.DropsSinceSpawn().empty());
}

// A log that is not there is nothing yet, whatever the reader does with it, and its lines count
// from offset 0 when it appears. The baseline is a log that is there.
void TestMissingFile() {
	TempDirectory     directory;
	const auto        path = directory / "later.txt";
	ProgressLogReader reader(path);
	reader.BeginSpawn(T0);
	CHECK(reader.Offset() == 0);
	CHECK(reader.Poll(T1) == 0);
	CHECK(reader.LinesSinceSpawn() == 0 && !reader.LastLine().has_value());
	CHECK(reader.LastLineTime() == T0);
	AppendText(path, "predict " + Hex(HASH_A) + "\n");
	CHECK(reader.Poll(T2) == 1);
	CHECK(reader.LastLine() && *reader.LastLine() == Predict(HASH_A));

	ProgressLogReader present(path);
	present.BeginSpawnAt(0, T0);
	CHECK(present.Poll(T1) == 1);
}

// The time of the last line is the poll's time when a line arrived, the spawn's time while none
// has, and it does not move for a poll that found nothing or only lines that are not the worker's.
// The supervisor's 180 s and 60 s timeouts are measured from it.
void TestLastLineTime() {
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	AppendText(path, "");
	ProgressLogReader reader(path);
	reader.BeginSpawn(T0);
	CHECK(reader.LastLineTime() == T0);
	CHECK(reader.Poll(T1) == 0);
	CHECK(reader.LastLineTime() == T0);
	AppendText(path, "garbage line\n");
	CHECK(reader.Poll(T2) == 0);
	CHECK(reader.MalformedLines() == 1);
	CHECK(reader.LastLineTime() == T0);
	AppendText(path, "predict " + Hex(HASH_A) + "\n");
	CHECK(reader.Poll(T3) == 1);
	CHECK(reader.LastLineTime() == T3);
	// A new spawn starts its clock again.
	reader.BeginSpawn(T3 + 5s);
	CHECK(reader.LastLineTime() == T3 + 5s);
}

// The latest order line wins, the summary belongs to its spawn, and the last start line of the
// whole log is kept for a spawn that has none of its own.
void TestOrderSummaryAndStart() {
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	AppendText(path, "order 7 " + Hex(ORDER_ONE) + "\n");
	AppendText(path, "start 4 " + Hex(HASH_A) + "\n");
	AppendText(path, "order 9 " + Hex(ORDER_TWO) + "\n");
	AppendText(path, "summary exit=321 scanned=2 order=" + Hex(ORDER_TWO) + "\n");
	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	CHECK(reader.Poll(T1) == 4);
	CHECK(reader.LatestOrder() && reader.LatestOrder()->rank_or_count == 9 &&
	      reader.LatestOrder()->hash == ORDER_TWO);
	CHECK(reader.Summary() && reader.Summary()->exit_code == 321);
	CHECK(reader.LastLine() && reader.LastLine()->kind == ProgressKind::Summary);
	reader.BeginSpawn(T2);
	CHECK(!reader.Summary().has_value());
	CHECK(!reader.StartSeenSinceSpawn() && reader.AnyStartSeen());
	CHECK(reader.LastStart() && reader.LastStart()->rank_or_count == 4);
	CHECK(reader.LatestOrder() && reader.LatestOrder()->hash == ORDER_TWO);
}

// A poll reads a bounded part of a big log, and the next poll goes on from where it stopped, so no
// line is lost or read twice at the cut. The baseline is the same log through a smaller one that
// one poll takes whole.
void TestLargeLogIsReadInPolls() {
	TempDirectory directory;
	const auto    path = directory / "big.txt";
	std::string   text;
	for (size_t i = 0; i < LARGE_LOG_LINES; i++) {
		text += "predict " + Hex(HASH_A + i) + "\n";
	}
	AppendText(path, text);
	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	const auto first = reader.Poll(T1);
	CHECK(first > 0 && first < LARGE_LOG_LINES);
	CHECK(reader.Offset() <= MAX_PROGRESS_POLL_BYTES);
	size_t total = first;
	for (int polls = 0; polls < 10 && total < LARGE_LOG_LINES; polls++) {
		total += reader.Poll(T2);
	}
	CHECK(total == LARGE_LOG_LINES);
	CHECK(reader.MalformedLines() == 0);
	CHECK(reader.LastLine() && reader.LastLine()->hash == HASH_A + LARGE_LOG_LINES - 1);

	const auto small = directory / "small.txt";
	AppendText(small, text.substr(0, 1000 * 25));
	ProgressLogReader small_reader(small);
	small_reader.BeginSpawnAt(0, T0);
	CHECK(small_reader.Poll(T1) == 1000);
}

// A fresh run forgets what the spawns of the old run taught, which a spawn keeps. The baseline is
// BeginSpawn on the same log, which keeps all three.
void TestFreshRunForgets() {
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	AppendText(path, "order 7 " + Hex(ORDER_ONE) + "\nstart 4 " + Hex(HASH_A) + "\n");
	ProgressLogReader spawned(path);
	spawned.BeginSpawnAt(0, T0);
	CHECK(spawned.Poll(T1) == 2);
	spawned.BeginSpawn(T2);
	CHECK(spawned.AnyStartSeen() && spawned.LastStart().has_value() &&
	      spawned.LatestOrder().has_value());

	ProgressLogReader fresh(path);
	fresh.BeginSpawnAt(0, T0);
	CHECK(fresh.Poll(T1) == 2);
	fresh.BeginFreshRun(T2);
	CHECK(!fresh.AnyStartSeen());
	CHECK(!fresh.LastStart().has_value());
	CHECK(!fresh.LatestOrder().has_value());
	CHECK(fresh.LinesSinceSpawn() == 0 && fresh.LastLineTime() == T2);
	// It still begins after what the log held, so the old lines are not read again.
	CHECK(fresh.Offset() == SizeOf(path));
	AppendText(path, "predict " + Hex(HASH_C) + "\n");
	CHECK(fresh.Poll(T3) == 1 && !fresh.AnyStartSeen());
}

// The prediction rows outlive a spawn and a fresh run, and the latest row of a program wins while
// it keeps the place of the first. The baselines are the order row, which a fresh run does forget,
// and a second program whose row is not replaced.
void TestPredictionsKept() {
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	AppendText(path, "predict " + Hex(HASH_A) + "\nprediction " + Hex(HASH_A) + " calls 0\n" +
	                     "predict " + Hex(HASH_B) + "\nprediction " + Hex(HASH_B) +
	                     " predicted 3\norder 3 " + Hex(ORDER_ONE) + "\n");
	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	CHECK(reader.PollToEnd(T1) == 5 && reader.MalformedLines() == 0);
	const std::vector<ProgressPrediction> first {{HASH_A, PredictionOutcome::Calls, 0},
	                                             {HASH_B, PredictionOutcome::Predicted, 3}};
	CHECK(reader.Predictions() == first);
	CHECK(reader.KindCount(ProgressKind::Prediction) == 2);

	// A new spawn counts its own lines, and still holds the rows.
	reader.BeginSpawn(T2);
	CHECK(reader.KindCount(ProgressKind::Prediction) == 0 && reader.Predictions() == first);

	// A program predicted again in the next spawn takes the new row in its old place.
	AppendText(path, "prediction " + Hex(HASH_A) + " predicted 5\n");
	CHECK(reader.PollToEnd(T3) == 1);
	const std::vector<ProgressPrediction> replaced {{HASH_A, PredictionOutcome::Predicted, 5},
	                                                {HASH_B, PredictionOutcome::Predicted, 3}};
	CHECK(reader.Predictions() == replaced);

	// A fresh run forgets the order and keeps the rows.
	ProgressLogReader fresh(path);
	fresh.BeginSpawnAt(0, T0);
	fresh.PollToEnd(T1);
	CHECK(fresh.LatestOrder().has_value());
	fresh.BeginFreshRun(T2);
	CHECK(!fresh.LatestOrder().has_value());
	CHECK(fresh.Predictions() == replaced);
	AppendText(path, "prediction " + Hex(HASH_C) + " problems 0\nprediction " + Hex(HASH_B) +
	                     " unreadable 0\n");
	CHECK(fresh.PollToEnd(T3) == 2);
	const std::vector<ProgressPrediction> after_fresh {{HASH_A, PredictionOutcome::Predicted, 5},
	                                                   {HASH_B, PredictionOutcome::Unreadable, 0},
	                                                   {HASH_C, PredictionOutcome::Problems, 0}};
	CHECK(fresh.Predictions() == after_fresh);
}

// CandidatesOf gives the count of a program's latest row across spawns, and keeps it through a
// fresh run. Program B is never predicted again, so it shows that the answer for A changed because
// of the later row and not because every count moved. A program with no row has no count, which a
// count of 0 is not.
void TestCandidatesOf() {
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	AppendText(path, "predict " + Hex(HASH_A) + "\nprediction " + Hex(HASH_A) + " predicted 3\n" +
	                     "predict " + Hex(HASH_B) + "\nprediction " + Hex(HASH_B) +
	                     " predicted 2\n");
	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	reader.PollToEnd(T1);
	CHECK(reader.CandidatesOf(HASH_A) == std::optional<uint64_t> {3});
	CHECK(reader.CandidatesOf(HASH_B) == std::optional<uint64_t> {2});
	CHECK(!reader.CandidatesOf(HASH_C).has_value());

	// The second spawn predicts A again with another count, and the latest row wins.
	reader.BeginSpawn(T2);
	AppendText(path, "prediction " + Hex(HASH_A) + " predicted 5\n");
	reader.PollToEnd(T3);
	CHECK(reader.CandidatesOf(HASH_A) == std::optional<uint64_t> {5});
	CHECK(reader.CandidatesOf(HASH_B) == std::optional<uint64_t> {2});

	// A fresh run keeps the rows, and still has no row for C.
	reader.BeginFreshRun(T3);
	CHECK(reader.CandidatesOf(HASH_A) == std::optional<uint64_t> {5});
	CHECK(reader.CandidatesOf(HASH_B) == std::optional<uint64_t> {2});
	CHECK(!reader.CandidatesOf(HASH_C).has_value());
}

// PollToEnd reads a log of more than one poll whole, where Poll stops at its bound, and says that
// it caught up. It is what a supervisor calls before BeginSpawn, so that a dead child's last line
// is not skipped.
void TestPollToEndCatchesUp() {
	TempDirectory directory;
	const auto    path = directory / "big.txt";
	std::string   text;
	for (size_t i = 0; i < LARGE_LOG_LINES; i++) {
		text += "predict " + Hex(HASH_A + i) + "\n";
	}
	AppendText(path, text);
	ProgressLogReader single(path);
	single.BeginSpawnAt(0, T0);
	single.Poll(T1);
	CHECK(!single.CaughtUp());

	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	CHECK(reader.PollToEnd(T1) == LARGE_LOG_LINES);
	CHECK(reader.CaughtUp() && reader.Offset() == SizeOf(path));
	CHECK(reader.LastLine() && reader.LastLine()->hash == HASH_A + LARGE_LOG_LINES - 1);
	// The caller can now begin a spawn and lose nothing.
	reader.BeginSpawn(T2);
	CHECK(reader.Offset() == SizeOf(path));
	CHECK(reader.KindCount(ProgressKind::Predict) == 0);
}

// A log that is shorter than what was read is another file. Its lines are read from its start and
// the held half line of the old one is dropped. The baseline is a log that only grew.
void TestShrunkLogIsReadAgain() {
	TempDirectory directory;
	const auto    path = directory / "log.txt";
	AppendText(path, "predict " + Hex(HASH_A) + "\nstart 1 " + Hex(HASH_B) + "\nstart 2 " +
	                     Hex(HASH_C).substr(0, 4));
	ProgressLogReader reader(path);
	reader.BeginSpawnAt(0, T0);
	CHECK(reader.Poll(T1) == 2);
	CHECK(reader.LogResets() == 0);

	// Another file takes its place, shorter than what was read.
	{
		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		out << "order 3 " << Hex(ORDER_TWO) << "\n";
	}
	AppendText(path, "start 8 " + Hex(HASH_B) + "\n");
	CHECK(SizeOf(path) < reader.Offset());
	CHECK(reader.Poll(T2) == 2);
	CHECK(reader.LogResets() == 1);
	CHECK(reader.MalformedLines() == 0);
	CHECK(reader.LatestOrder() && reader.LatestOrder()->hash == ORDER_TWO);
	CHECK(reader.LastLine() && *reader.LastLine() == WithRank(ProgressKind::Start, 8, HASH_B));
	CHECK(reader.Offset() == SizeOf(path));

	// A log that only grows is no reset.
	AppendText(path, "start 9 " + Hex(HASH_C) + "\n");
	CHECK(reader.Poll(T3) == 1 && reader.LogResets() == 1);
}

// The loop of PollToEnd ends for a poll that is stuck, one that is not caught up and moves nothing.
// The fake counts its calls and claims to be caught up after a hundred, so a loop that did not stop
// would still end the test, with too many calls. The baseline is a poll that does move, which the
// loop calls until it is caught up.
void TestPollToEndStopsWithoutProgress() {
	constexpr int GIVE_UP_AFTER = 100;
	int           stuck_calls   = 0;
	const auto    stuck_found   = PollWhileProgressing(
        [&] {
            stuck_calls++;
            return size_t {0};
        },
        [&] { return stuck_calls >= GIVE_UP_AFTER; }, [] { return ProgressMark {7, 0}; });
	CHECK(stuck_calls == 1 && stuck_found == 0);

	// A reset counts as progress, though the offset did not move.
	int    reset_calls = 0;
	size_t resets      = 0;
	PollWhileProgressing(
	    [&] {
		    reset_calls++;
		    resets++;
		    return size_t {0};
	    },
	    [&] { return reset_calls >= 3; }, [&] { return ProgressMark {7, resets}; });
	CHECK(reset_calls == 3);

	int        moving_calls = 0;
	uint64_t   position     = 0;
	const auto moving_found = PollWhileProgressing(
	    [&] {
		    moving_calls++;
		    position += 10;
		    return size_t {2};
	    },
	    [&] { return moving_calls >= 4; }, [&] { return ProgressMark {position, 0}; });
	CHECK(moving_calls == 4 && moving_found == 8);
}

} // namespace

int main() {
	TestEveryLineKind();
	TestMalformedLines();
	TestPartialLineIsHeld();
	TestPerSpawnOffset();
	TestDropReasons();
	TestMissingFile();
	TestLastLineTime();
	TestOrderSummaryAndStart();
	TestLargeLogIsReadInPolls();
	TestFreshRunForgets();
	TestPredictionsKept();
	TestCandidatesOf();
	TestPollToEndCatchesUp();
	TestShrunkLogIsReadAgain();
	TestPollToEndStopsWithoutProgress();
	if (g_failures == 0) {
		std::puts("Shader store progress log tests passed");
	}
	return g_failures == 0 ? 0 : 1;
}
