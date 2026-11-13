// Tests the parent side of the Stream worker, which starts the worker in a child process, sends it
// requests, and recovers from a crash or a hang. The worker is a fake. This test binary starts
// itself with --fake-stream-worker <mode> <request file>, connects to the pipe its request names,
// writes the progress lines a Stream worker writes, and answers, crashes, hangs or misbehaves as
// its mode says. That makes a crash on one request repeatable, which a real worker cannot be made
// to do, and it needs no recompiler and no device. The real worker over a real pipe is tested in
// ShaderStoreStreamWorkerTests and ShaderStoreDispatchTests. Each check names what else could
// satisfy it and compares against a baseline that differs only in the behaviour under test.
#include "common/childProcess.h"
#include "common/file.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"
#include "graphics/shader/storegen/streamChannel.h"
#include "graphics/shader/storegen/streamProtocol.h"
#include "graphics/shader/storegen/streamSupervisor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics::StoreGen;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using Clock  = std::chrono::steady_clock;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

/// The command line word that makes this binary a fake worker.
constexpr const char* FAKE_FLAG = "--fake-stream-worker";
/// The exit code of a fake that crashes on purpose, which no real worker code uses.
constexpr int FAKE_CRASH_EXIT = 77;
/// How long a fake sleeps where it is meant to hang, longer than any test waits for it.
constexpr auto FAKE_HANG_TIME = 20s;
/// How long the fake tries to connect, as a real worker does.
constexpr auto FAKE_CONNECT_TIMEOUT = 5000ms;
/// How long a batching fake waits for more requests after the first, and how many it waits for.
/// The crash test sends that many, so the fake has them all running when it crashes.
constexpr auto   FAKE_BATCH_WINDOW = 200ms;
constexpr size_t FAKE_BATCH_SIZE   = 3;
/// The priority the fake writes, which only a supervisor that read the log can report.
constexpr const char* FAKE_PRIORITY = "fake_priority";
/// What the bad_frame fake sends instead of an answer, bytes that are no frame header.
constexpr uint8_t GARBAGE_BYTE  = 0xEE;
constexpr size_t  GARBAGE_BYTES = 16;
/// How long the fake that answers slowly takes for each request in the cap test.
constexpr uint64_t SLOW_ANSWER_MS = 100;
/// How many requests the cap test sends, well past the cap.
constexpr uint64_t CAP_TEST_REQUESTS = 10;
/// The code size of the blocked write test, far past what a pipe buffers, so the write blocks.
constexpr size_t BLOCKING_CODE_BYTES = size_t {4} << 20u;
/// The short deadlines the tests use in place of the real ones.
constexpr auto SHORT_DEADLINE = 1000ms;
/// How long the overdue test waits before its second request, half the deadline, so the second is
/// running but not overdue when the first passes its deadline.
constexpr auto HALF_DEADLINE = 500ms;
/// How much earlier than a deadline a clock may see it end, and how much later a loaded machine may
/// take to act on it.
constexpr auto DEADLINE_EARLY = 100ms;
constexpr auto DEADLINE_SLACK = 5000ms;
/// How long a test waits for outcomes that should come within a few spawns.
constexpr auto OUTCOME_LIMIT = 30s;
/// How long Stop may take with a hung child.
constexpr auto STOP_LIMIT = 1000ms;
/// How long a test waits to see that nothing more happens.
constexpr auto QUIET_TIME = 500ms;
/// How long the run may take before the watchdog fails it.
constexpr auto TEST_TIMEOUT = 280s;
/// How often a test looks at a file or a count while it waits.
constexpr auto POLL_SLICE = 10ms;
/// The restart cap of the cap test.
constexpr uint32_t SMALL_RESTART_CAP = 2;
/// Code hashes of the test's requests. Each request gets its own, so their retry keys differ.
constexpr uint64_t HASH_BASE = 0xC0DE0000;
/// The seq the crash test crashes on, and the new request it sends after the first crash.
constexpr uint64_t CRASH_SEQ = 2;
constexpr uint64_t LATE_SEQ  = FAKE_BATCH_SIZE + 1;
/// The tags of the tag test, one per outcome.
constexpr uint64_t TAG_ANSWERED = 0xA11;
constexpr uint64_t TAG_BLAMED   = 0xB1A;
constexpr uint64_t TAG_DROPPED  = 0xD20;

// ---- Files. ----

// A directory for the files of one test, which is removed with it.
class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		const auto name    = "kyty_stream_supervisor_tests_" +
		                  std::to_string(Common::CurrentProcessId()) + "_" +
		                  std::to_string(counter++);
		m_path = fs::temp_directory_path() / name;
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

// The lines of a text file, read through Common::File so a writer can keep it open. A last line
// with no newline is kept as it is.
std::vector<std::string> ReadLines(const fs::path& path) {
	Common::File file;
	if (!file.Open(path, Common::File::Mode::Read)) {
		return {};
	}
	const auto               bytes = file.ReadWholeBuffer();
	const std::string        text(reinterpret_cast<const char*>(bytes.data()), bytes.size());
	std::istringstream       in(text);
	std::vector<std::string> lines;
	for (std::string line; std::getline(in, line);) {
		lines.push_back(line);
	}
	return lines;
}

// How many of the lines start with the prefix.
size_t CountStarting(const std::vector<std::string>& lines, std::string_view prefix) {
	return static_cast<size_t>(
	    std::ranges::count_if(lines, [&](const auto& line) { return line.starts_with(prefix); }));
}

// ---- The fake worker. ----

// What the fake is asked to do, `name` or `name:value`.
struct FakeMode {
	std::string name;
	uint64_t    value = 0;
};

// Splits a mode word into its name and the number after a colon, which is 0 when there is none.
FakeMode ParseFakeMode(std::string_view text) {
	FakeMode   mode;
	const auto colon = text.find(':');
	mode.name        = std::string(text.substr(0, colon));
	if (colon != std::string_view::npos) {
		mode.value = std::stoull(std::string(text.substr(colon + 1)));
	}
	return mode;
}

// The fake's progress log. Every line is flushed at once, so a crash right after it leaves it in
// the file, as the real worker's unbuffered writes do.
class FakeLog {
public:
	explicit FakeLog(const fs::path& path): m_file(path, std::ios::binary | std::ios::app) {}

	void Line(const std::string& text) { Raw(text + "\n"); }
	void Raw(const std::string& text) {
		std::lock_guard lock(m_mutex);
		m_file << text;
		m_file.flush();
	}

private:
	std::mutex    m_mutex;
	std::ofstream m_file;
};

// The fake's end of the pipe. It keeps the pipe itself rather than a StreamChannel, so the
// bad_frame mode can write bytes that are no frame. One thread reads and one writes.
class FakeConnection {
public:
	explicit FakeConnection(StreamPipe pipe): m_pipe(std::move(pipe)) {}

	StreamReadResult ReadFrame() { return m_reader.ReadFrame(m_pipe); }
	bool             Send(const StreamResponse& response) {
        return EncodeStreamResponse(response, m_buffer) && m_pipe.WriteAll(m_buffer);
	}
	bool SendRaw(std::span<const uint8_t> bytes) { return m_pipe.WriteAll(bytes); }
	void Interrupt() { m_pipe.Interrupt(); }
	void Close() { m_pipe.Close(); }

private:
	StreamPipe           m_pipe;
	StreamFrameReader    m_reader;
	std::vector<uint8_t> m_buffer;
};

// The requests the fake's reader has taken and its serving thread has not.
class FakeInbox {
public:
	void Push(StreamRequest request) {
		std::lock_guard lock(m_mutex);
		m_requests.push_back(std::move(request));
		m_changed.notify_all();
	}
	void Close() {
		std::lock_guard lock(m_mutex);
		m_closed = true;
		m_changed.notify_all();
	}
	// Waits for a request, and in a batching mode for up to FAKE_BATCH_SIZE within the window.
	// Returns nothing once the pipe is closed.
	std::vector<StreamRequest> Take(bool batched) {
		std::unique_lock lock(m_mutex);
		m_changed.wait(lock, [&] { return m_closed || !m_requests.empty(); });
		if (batched) {
			m_changed.wait_for(lock, FAKE_BATCH_WINDOW,
			                   [&] { return m_closed || m_requests.size() >= FAKE_BATCH_SIZE; });
		}
		std::vector<StreamRequest> taken(std::make_move_iterator(m_requests.begin()),
		                                 std::make_move_iterator(m_requests.end()));
		m_requests.clear();
		return m_closed ? std::vector<StreamRequest> {} : taken;
	}

private:
	std::mutex                m_mutex;
	std::condition_variable   m_changed;
	std::deque<StreamRequest> m_requests;
	bool                      m_closed = false;
};

// Whether the batch holds a request with this seq.
bool HasSeq(const std::vector<StreamRequest>& batch, uint64_t seq) {
	return std::ranges::any_of(batch, [&](const auto& request) { return request.seq == seq; });
}

// Creates a marker file, which tells a later spawn of the same test that it is not the first.
void Touch(const fs::path& path) {
	std::ofstream(path, std::ios::binary).put('1');
}

// The fake's reader. Like the real worker it logs a start line for a request, here as soon as it
// arrives, so every request it holds counts as running.
void ReadFakeRequests(FakeConnection& connection, FakeLog& log, FakeInbox& inbox) {
	for (;;) {
		const auto read    = connection.ReadFrame();
		auto       request = read.status == StreamReadStatus::Frame
		                         ? DecodeStreamRequest(read.frame)
		                         : std::nullopt;
		if (!request) {
			inbox.Close();
			return;
		}
		log.Line(fmt::format("start {} {:016x}", request->seq, request->code_hash));
		inbox.Push(std::move(*request));
	}
}

// Answers one request as a real worker does, with its `done` line before the answer, and with the
// request's code as the payload, so a test sees which code was sent. The first spawn of the
// done_no_reply and torn_done modes dies between the line and the answer, with the line whole or
// torn.
bool AnswerFake(const FakeMode& mode, bool first, const fs::path& marker,
                FakeConnection& connection, FakeLog& log, const StreamRequest& request) {
	const auto done = fmt::format("done {} ok", request.seq);
	if (first && (mode.name == "done_no_reply" || mode.name == "torn_done")) {
		mode.name == "done_no_reply" ? log.Line(done) : log.Raw(done);
		Touch(marker);
		std::_Exit(FAKE_CRASH_EXIT);
	}
	if (mode.name == "answer") {
		std::this_thread::sleep_for(std::chrono::milliseconds(mode.value));
	}
	log.Line(done);
	StreamResponse response;
	response.seq     = request.seq;
	response.kind    = request.kind;
	response.status  = StreamStatus::Ok;
	response.payload = request.code;
	return connection.Send(response);
}

// Serves requests until the pipe closes. The crash and hang modes batch, so the requests sent
// together are all running when the fake crashes or hangs on one of them. The bad_frame mode
// writes bytes that are no frame in place of its first answer, then keeps the pipe open.
void ServeFake(const FakeMode& mode, bool first, const fs::path& marker,
               FakeConnection& connection, FakeLog& log, FakeInbox& inbox) {
	const bool batched = mode.name == "crash_with_others" || mode.name == "hang";
	for (;;) {
		const auto batch = inbox.Take(batched);
		if (batch.empty()) {
			return;
		}
		if (mode.name == "crash_with_others" && HasSeq(batch, mode.value)) {
			std::_Exit(FAKE_CRASH_EXIT);
		}
		if ((mode.name == "hang" && HasSeq(batch, mode.value)) || mode.name == "bad_frame") {
			if (mode.name == "bad_frame") {
				const std::vector<uint8_t> garbage(GARBAGE_BYTES, GARBAGE_BYTE);
				(void)connection.SendRaw(garbage);
			}
			std::this_thread::sleep_for(FAKE_HANG_TIME);
			return;
		}
		for (const auto& request: batch) {
			if (!AnswerFake(mode, first, marker, connection, log, request)) {
				return;
			}
		}
	}
}

// The first spawn of the read_one mode. It takes one request and logs its start, then freezes and
// reads no more, so the request stays running and the parent's next large write fills the pipe.
// Later spawns of the mode answer normally.
int ReadOneAndFreeze(FakeConnection& connection, FakeLog& log, const fs::path& marker) {
	const auto read    = connection.ReadFrame();
	const auto request = read.status == StreamReadStatus::Frame ? DecodeStreamRequest(read.frame)
	                                                            : std::nullopt;
	if (request) {
		log.Line(fmt::format("start {} {:016x}", request->seq, request->code_hash));
	}
	Touch(marker);
	std::this_thread::sleep_for(FAKE_HANG_TIME);
	return WORKER_EXIT_DONE;
}

// What this binary does when it is started as a fake worker. The marker file next to the log tells
// a later spawn of the same test that it is not the first. The no_read mode connects and then never
// reads, as a frozen child would, so the parent's writes fill the pipe, and the read_one mode does
// the same after it took one request.
int RunFakeStreamWorker(std::string_view mode_text, const fs::path& request_file) {
	const auto request = ReadWorkerRequestFile(request_file);
	if (!request) {
		return WORKER_EXIT_BAD_REQUEST;
	}
	Common::ChildProcess::EnterChildMode(request->parent_pid);
	const auto mode   = ParseFakeMode(mode_text);
	const auto marker = PathFromUtf8(request->progress_path + ".marker");
	const bool first  = !fs::exists(marker);
	FakeLog    log(PathFromUtf8(request->progress_path));
	log.Line(std::string("priority ") + FAKE_PRIORITY);
	if (mode.name == "never_connect" || (mode.name == "never_connect_once" && first)) {
		Touch(marker);
		std::this_thread::sleep_for(FAKE_HANG_TIME);
		return WORKER_EXIT_PIPE_FAILED;
	}
	auto pipe = ConnectStreamPipeWithin(request->stream_pipe, FAKE_CONNECT_TIMEOUT);
	if (!pipe) {
		return WORKER_EXIT_PIPE_FAILED;
	}
	if (mode.name == "crash_at_connect") {
		std::_Exit(FAKE_CRASH_EXIT);
	}
	if (mode.name == "no_read") {
		std::this_thread::sleep_for(FAKE_HANG_TIME);
		return WORKER_EXIT_DONE;
	}
	FakeConnection connection(std::move(*pipe));
	if (mode.name == "read_one" && first) {
		return ReadOneAndFreeze(connection, log, marker);
	}
	FakeInbox      inbox;
	std::thread    reader([&] { ReadFakeRequests(connection, log, inbox); });
	ServeFake(mode, first, marker, connection, log, inbox);
	connection.Interrupt();
	reader.join();
	connection.Close();
	return WORKER_EXIT_DONE;
}

// ---- The supervisor under test. ----

// Collects the outcomes the callback gets.
class Outcomes {
public:
	void Add(StreamCompletion&& completion) {
		std::lock_guard lock(m_mutex);
		m_list.push_back(std::move(completion));
		m_changed.notify_all();
	}
	// Waits until at least count outcomes arrived.
	bool WaitFor(size_t count, std::chrono::milliseconds limit) {
		std::unique_lock lock(m_mutex);
		return m_changed.wait_for(lock, limit, [&] { return m_list.size() >= count; });
	}
	// A copy of every outcome so far, in the order they arrived.
	std::vector<StreamCompletion> List() const {
		std::lock_guard lock(m_mutex);
		return m_list;
	}
	// The completion of a seq, when it has exactly one.
	std::optional<StreamCompletion> CompletionOf(uint64_t seq) const {
		std::lock_guard                 lock(m_mutex);
		std::optional<StreamCompletion> found;
		size_t                          count = 0;
		for (const auto& completion: m_list) {
			if (completion.seq == seq) {
				found = completion;
				count++;
			}
		}
		return count == 1 ? found : std::nullopt;
	}
	// The outcome of a seq, when it has exactly one.
	std::optional<StreamOutcome> Of(uint64_t seq) const {
		const auto completion = CompletionOf(seq);
		return completion ? std::optional(completion->outcome) : std::nullopt;
	}
	// Where a seq's outcome came in the order of arrival.
	size_t IndexOf(uint64_t seq) const {
		std::lock_guard lock(m_mutex);
		const auto found = std::ranges::find_if(
		    m_list, [&](const auto& completion) { return completion.seq == seq; });
		return static_cast<size_t>(found - m_list.begin());
	}

private:
	mutable std::mutex            m_mutex;
	std::condition_variable       m_changed;
	std::vector<StreamCompletion> m_list;
};

// A config that starts this binary as a fake in the given mode, with its files in the directory.
StreamSupervisorConfig MakeConfig(const TempDirectory& directory, const std::string& mode,
                                  Outcomes& outcomes) {
	StreamSupervisorConfig config;
	config.files_base               = directory / "prescan";
	config.program                  = Common::ExecutablePath();
	config.arguments_before_request = {FAKE_FLAG, mode};
	config.on_complete = [&outcomes](StreamCompletion&& completion) {
		outcomes.Add(std::move(completion));
	};
	return config;
}

// Where the supervisor of a test keeps its progress log.
fs::path ProgressPathOf(const TempDirectory& directory) {
	return directory / "prescan.progress";
}

// A request the fake answers. Its code hash makes its retry key its own, and its code holds the
// index and kind, so an answer's payload shows which request's code was sent.
StreamRequest MakeRequest(uint64_t index, StreamRequestKind kind = StreamRequestKind::Plan) {
	StreamRequest request;
	request.kind      = kind;
	request.code_hash = HASH_BASE + index;
	request.code      = {static_cast<uint8_t>(index), 0xC0, 0xDE, static_cast<uint8_t>(kind)};
	return request;
}

// Waits until the log has at least count lines starting with prefix.
bool WaitForLines(const fs::path& path, std::string_view prefix, size_t count) {
	const auto until = Clock::now() + OUTCOME_LIMIT;
	while (Clock::now() < until) {
		if (CountStarting(ReadLines(path), prefix) >= count) {
			return true;
		}
		std::this_thread::sleep_for(POLL_SLICE);
	}
	return false;
}

// Waits until the supervisor has counted at least count crashes.
bool WaitForCrashes(const StreamSupervisor& supervisor, uint32_t count) {
	const auto until = Clock::now() + OUTCOME_LIMIT;
	while (Clock::now() < until) {
		if (supervisor.Stats().crashes >= count) {
			return true;
		}
		std::this_thread::sleep_for(POLL_SLICE);
	}
	return false;
}

// Whether the late request never started while another request was running in the same spawn. A
// spawn begins at its priority line, and a request runs from its start line to its done line.
bool LateRequestRanAlone(const std::vector<std::string>& lines) {
	std::set<std::string> running;
	bool                  alone = true;
	for (const auto& line: lines) {
		std::istringstream words(line);
		std::string        word;
		std::string        seq;
		words >> word >> seq;
		if (word == "priority") {
			running.clear();
		} else if (word == "start") {
			alone = alone && (seq != std::to_string(LATE_SEQ) || running.empty());
			running.insert(seq);
		} else if (word == "done") {
			running.erase(seq);
		}
	}
	return alone;
}

// ---- Tests. ----

// Percentiles come from the bucket upper bounds and never pass the maximum. A histogram that kept
// only the maximum would give 99.5 for all three, and one with coarse buckets would round 49.5 up
// past 50.
void TestRttHistogram() {
	// The values are 0.5, 1.5 and on to 99.5 ms, so the median is 49.5 in the bucket below 50.
	constexpr int    VALUES  = 100;
	constexpr double HALF    = 0.5;
	constexpr double P50     = 0.5;
	constexpr double P90     = 0.9;
	constexpr double ALL     = 1.0;
	constexpr double LARGEST = VALUES - HALF;
	// A coarse value whose bucket's upper bound is past it, so only the maximum caps it.
	constexpr double SLOW_MS = 5000.0;
	StreamRttHistogram histogram;
	for (int value = 0; value < VALUES; value++) {
		histogram.Add(value + HALF);
	}
	CHECK(histogram.Count() == VALUES);
	CHECK(histogram.Percentile(P50) == 50.0 && histogram.Percentile(P90) == 90.0);
	CHECK(histogram.MaxMs() == LARGEST && histogram.Percentile(ALL) == LARGEST);
	StreamRttHistogram slow;
	slow.Add(SLOW_MS);
	CHECK(slow.Percentile(P50) == SLOW_MS);
	CHECK(StreamRttHistogram {}.Percentile(P50) == 0.0);
}

// No child exists before the first request. Then a Plan and a Compile that share one copy of their
// code are answered by one child, the priority comes from the log and each kind's round trip is
// counted apart. A supervisor that spawned at once would show a spawn before the request. The
// fake answers with the code it was sent, so a supervisor that sent the request's own emptied
// code fails the payload check. The priority could only be empty or the fake's.
void TestLazySpawnAnswersAndReadsPriority() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(MakeConfig(directory, "answer", outcomes));
	std::this_thread::sleep_for(QUIET_TIME);
	CHECK(supervisor.Stats().spawns == 0);
	const auto code    = std::make_shared<const std::vector<uint8_t>>(MakeRequest(1).code);
	const auto plan    = supervisor.Submit(MakeRequest(1), code);
	const auto compile = supervisor.Submit(MakeRequest(1, StreamRequestKind::Compile), code);
	CHECK(plan.status == StreamSubmitStatus::Queued);
	CHECK(compile.status == StreamSubmitStatus::Queued);
	CHECK(outcomes.WaitFor(2, OUTCOME_LIMIT));
	CHECK(outcomes.Of(plan.seq) == StreamOutcome::Answered);
	CHECK(outcomes.Of(compile.seq) == StreamOutcome::Answered);
	const auto list = outcomes.List();
	CHECK(std::ranges::all_of(list, [&](const auto& c) { return c.response.payload == *code; }));
	const auto stats = supervisor.Stats();
	CHECK(stats.spawns == 1 && stats.crashes == 0 && stats.priority == FAKE_PRIORITY);
	CHECK(stats.plan_rtt.count == 1 && stats.compile_rtt.count == 1);
	supervisor.Stop();
}

// With a slow child, no more than the cap are ever sent and unanswered. The fake logs a start line
// as each request arrives and a done line before each answer, so the most starts ahead of dones is
// what the child really held. All ten are answered, so the cap delays and drops nothing. Without
// the cap the child holds all ten, and a cap that was too small would hold fewer than four.
void TestOutstandingCapHolds() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(
	    MakeConfig(directory, fmt::format("answer:{}", SLOW_ANSWER_MS), outcomes));
	for (uint64_t index = 0; index < CAP_TEST_REQUESTS; index++) {
		CHECK(supervisor.Submit(MakeRequest(index)).status == StreamSubmitStatus::Queued);
	}
	CHECK(outcomes.WaitFor(CAP_TEST_REQUESTS, OUTCOME_LIMIT));
	supervisor.Stop();
	int64_t held = 0;
	int64_t most = 0;
	for (const auto& line: ReadLines(ProgressPathOf(directory))) {
		held += line.starts_with("start ") ? 1 : line.starts_with("done ") ? -1 : 0;
		most = std::max(most, held);
	}
	CHECK(most == STREAM_MAX_OUTSTANDING);
	CHECK(supervisor.Stats().peak_outstanding == STREAM_MAX_OUTSTANDING);
	CHECK(supervisor.Stats().answered == CAP_TEST_REQUESTS);
}

// Three requests run together and the child crashes on the second. Three were running, so none is
// blamed then. The new child gets them one at a time, and only #2 crashes it while alone, so only
// #2 is blamed, after #1 was answered, and #3 is answered after it. A supervisor that blamed every
// running request would blame all three at the first crash. A copy of #2 is refused afterwards.
// A fourth request queued after the first crash waits for the suspects. It never starts while
// another request runs in the same spawn. Without the suspect gate it would go out with the last
// suspect, #3, since the suspects go in seq order.
void TestCrashBlamesRequest() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(
	    MakeConfig(directory, fmt::format("crash_with_others:{}", CRASH_SEQ), outcomes));
	for (uint64_t index = 1; index <= FAKE_BATCH_SIZE; index++) {
		CHECK(supervisor.Submit(MakeRequest(index)).seq == index);
	}
	CHECK(WaitForCrashes(supervisor, 1));
	CHECK(supervisor.Submit(MakeRequest(LATE_SEQ)).seq == LATE_SEQ);
	CHECK(outcomes.WaitFor(LATE_SEQ, OUTCOME_LIMIT));
	CHECK(outcomes.Of(1) == StreamOutcome::Answered);
	CHECK(outcomes.Of(CRASH_SEQ) == StreamOutcome::Blamed);
	CHECK(outcomes.Of(FAKE_BATCH_SIZE) == StreamOutcome::Answered);
	CHECK(outcomes.Of(LATE_SEQ) == StreamOutcome::Answered);
	CHECK(outcomes.IndexOf(1) < outcomes.IndexOf(CRASH_SEQ) &&
	      outcomes.IndexOf(CRASH_SEQ) < outcomes.IndexOf(FAKE_BATCH_SIZE));
	const auto stats = supervisor.Stats();
	CHECK(stats.crashes == 2 && stats.spawns == 3 && stats.never_retry == 1 && stats.blamed == 1);
	// The first spawn ran the first three, and each was sent again alone.
	const auto lines = ReadLines(ProgressPathOf(directory));
	for (uint64_t seq = 1; seq <= FAKE_BATCH_SIZE; seq++) {
		CHECK(CountStarting(lines, fmt::format("start {} ", seq)) == 2);
	}
	CHECK(LateRequestRanAlone(lines));
	CHECK(supervisor.Submit(MakeRequest(CRASH_SEQ)).status == StreamSubmitStatus::NeverRetry);
	supervisor.Stop();
}

// The child wrote the request's done line and died before the answer. The request is not running
// by its log, so nothing is blamed and the next child answers it. The torn test below is the same
// death with the done line cut off, which blames the request, so only the done line differs.
void TestDoneButNoReplyIsResent() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(MakeConfig(directory, "done_no_reply", outcomes));
	const auto       seq = supervisor.Submit(MakeRequest(1)).seq;
	CHECK(outcomes.WaitFor(1, OUTCOME_LIMIT));
	CHECK(outcomes.Of(seq) == StreamOutcome::Answered);
	const auto stats = supervisor.Stats();
	CHECK(stats.crashes == 1 && stats.never_retry == 0 && stats.spawns == 2 && stats.resent == 1);
	supervisor.Stop();
}

// The child died while it wrote the done line, so the log ends in `done 1 ok` with no newline. The
// torn line is ignored, so the request was running alone and is blamed, with no second spawn. A
// reader that took the torn line would count the request done and resend it, and the next child
// would answer it.
void TestTornLastLineIsIgnored() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(MakeConfig(directory, "torn_done", outcomes));
	const auto       seq = supervisor.Submit(MakeRequest(1)).seq;
	CHECK(outcomes.WaitFor(1, OUTCOME_LIMIT));
	CHECK(outcomes.Of(seq) == StreamOutcome::Blamed);
	const auto lines = ReadLines(ProgressPathOf(directory));
	CHECK(!lines.empty() && lines.back() == "done 1 ok");
	const auto stats = supervisor.Stats();
	CHECK(stats.crashes == 1 && stats.never_retry == 1 && stats.spawns == 1);
	supervisor.Stop();
}

// A request that hangs is killed at its deadline, counted as a hang and not a crash, and blamed,
// since it was running alone. It completes no sooner than the deadline and well before the
// fake's hang ends, which a supervisor without deadlines would wait for. The next request is
// answered by a new child.
void TestDeadlineKillsHungRequest() {
	TempDirectory directory;
	Outcomes      outcomes;
	auto          config    = MakeConfig(directory, "hang:1", outcomes);
	config.request_deadline = SHORT_DEADLINE;
	StreamSupervisor supervisor(std::move(config));
	const auto       began = Clock::now();
	CHECK(supervisor.Submit(MakeRequest(1)).seq == 1);
	CHECK(outcomes.WaitFor(1, SHORT_DEADLINE + DEADLINE_SLACK));
	const auto took = Clock::now() - began;
	CHECK(took >= SHORT_DEADLINE - DEADLINE_EARLY);
	CHECK(outcomes.Of(1) == StreamOutcome::Blamed);
	CHECK(supervisor.Submit(MakeRequest(2)).seq == 2);
	CHECK(outcomes.WaitFor(2, OUTCOME_LIMIT));
	CHECK(outcomes.Of(2) == StreamOutcome::Answered);
	const auto stats = supervisor.Stats();
	CHECK(stats.hangs == 1 && stats.crashes == 0 && stats.spawns == 2);
	supervisor.Stop();
}

// Two requests are running when the first passes its deadline, and the second was sent half a
// deadline later, so it is not overdue. Only the first can explain the hang, so it is blamed at
// once, and the second becomes a suspect that the next child answers alone. One hang and two
// spawns. A supervisor that took every running request as a candidate would make both suspects,
// so the first would hang a second time alone, for two hangs and three spawns.
void TestHangBlamesOnlyOverdue() {
	TempDirectory directory;
	Outcomes      outcomes;
	auto          config    = MakeConfig(directory, "hang:1", outcomes);
	config.request_deadline = SHORT_DEADLINE;
	StreamSupervisor supervisor(std::move(config));
	(void)supervisor.Submit(MakeRequest(1));
	CHECK(WaitForLines(ProgressPathOf(directory), "start 1 ", 1));
	std::this_thread::sleep_for(HALF_DEADLINE);
	(void)supervisor.Submit(MakeRequest(2));
	CHECK(WaitForLines(ProgressPathOf(directory), "start 2 ", 1));
	CHECK(outcomes.WaitFor(2, OUTCOME_LIMIT));
	CHECK(outcomes.Of(1) == StreamOutcome::Blamed && outcomes.Of(2) == StreamOutcome::Answered);
	const auto stats = supervisor.Stats();
	CHECK(stats.hangs == 1 && stats.spawns == 2 && stats.never_retry == 1);
	supervisor.Stop();
}

// A child that connects and never reads lets the parent's write fill the pipe and block. The write
// gives up at the request's deadline, which counts as a hang. Nothing was running, so nothing is
// blamed, and with no restarts left the lane turns off and the request is Dropped. A write with no
// deadline would block until the fake's hang ends, far past the limit here.
void TestBlockedWriteIsAHang() {
	TempDirectory directory;
	Outcomes      outcomes;
	auto          config    = MakeConfig(directory, "no_read", outcomes);
	config.request_deadline = SHORT_DEADLINE;
	config.max_restarts     = 0;
	StreamSupervisor supervisor(std::move(config));
	auto             request = MakeRequest(1);
	request.code.assign(BLOCKING_CODE_BYTES, GARBAGE_BYTE);
	const auto began = Clock::now();
	const auto seq   = supervisor.Submit(std::move(request)).seq;
	CHECK(outcomes.WaitFor(1, SHORT_DEADLINE + DEADLINE_SLACK));
	CHECK(Clock::now() - began >= SHORT_DEADLINE - DEADLINE_EARLY);
	CHECK(outcomes.Of(seq) == StreamOutcome::Dropped);
	const auto stats = supervisor.Stats();
	CHECK(stats.hangs == 1 && stats.crashes == 0 && stats.blamed == 0);
	CHECK(stats.lane_off && stats.lane_off_reason == "restarts_capped");
	supervisor.Stop();
}

// A child that took one request and froze leaves it running, and the parent's next large write
// fills the pipe and blocks until the running request's deadline. The write's timeout ends the
// child as a hang no earlier than that deadline, so the running request is overdue, is the only
// candidate and is blamed. The large request was never read, so it is sent again and the next
// child answers it, with one hang and two spawns. A kill time taken before the deadline would make
// no request a candidate, so the first request would be sent again too and answered, and the
// frozen child's hang would never be blamed.
void TestBlockedWriteBlamesRunningRequest() {
	TempDirectory directory;
	Outcomes      outcomes;
	auto          config    = MakeConfig(directory, "read_one", outcomes);
	config.request_deadline = SHORT_DEADLINE;
	StreamSupervisor supervisor(std::move(config));
	(void)supervisor.Submit(MakeRequest(1));
	CHECK(WaitForLines(ProgressPathOf(directory), "start 1 ", 1));
	auto large = MakeRequest(2);
	large.code.assign(BLOCKING_CODE_BYTES, GARBAGE_BYTE);
	(void)supervisor.Submit(std::move(large));
	CHECK(outcomes.WaitFor(2, OUTCOME_LIMIT));
	CHECK(outcomes.Of(1) == StreamOutcome::Blamed && outcomes.Of(2) == StreamOutcome::Answered);
	const auto stats = supervisor.Stats();
	CHECK(stats.hangs == 1 && stats.crashes == 0 && stats.spawns == 2 && stats.never_retry == 1);
	supervisor.Stop();
}

// A callback that throws on its first call is caught and counted, and the supervisor goes on, so
// the next request is still Answered. An exception that left the reader thread would end the
// process at once through std::terminate.
void TestThrowingCallbackIsContained() {
	TempDirectory     directory;
	Outcomes          outcomes;
	std::atomic<bool> thrown {false};
	auto              config = MakeConfig(directory, "answer", outcomes);
	// Throws on the first outcome only, and collects the rest.
	config.on_complete = [&](StreamCompletion&& completion) {
		if (!thrown.exchange(true)) {
			throw std::runtime_error("a test callback that throws once");
		}
		outcomes.Add(std::move(completion));
	};
	StreamSupervisor supervisor(std::move(config));
	(void)supervisor.Submit(MakeRequest(1));
	const auto until = Clock::now() + OUTCOME_LIMIT;
	while (supervisor.Stats().callback_failures == 0 && Clock::now() < until) {
		std::this_thread::sleep_for(POLL_SLICE);
	}
	const auto second = supervisor.Submit(MakeRequest(2)).seq;
	CHECK(outcomes.WaitFor(1, OUTCOME_LIMIT));
	CHECK(outcomes.Of(second) == StreamOutcome::Answered);
	CHECK(supervisor.Stats().callback_failures == 1 && supervisor.Stats().answered == 2);
	supervisor.Stop();
}

// Two Compiles of one program run together and the child crashes on the first, so both become
// suspects. The first then crashes the child alone and is blamed, which keeps its program out. The
// second is still waiting by then, and the sweep after the blame completes it as Blamed at once,
// so it starts only in the first spawn and no third child is started for it. Without the sweep a
// third child would be started, and the send path's backstop would then refuse the request, for
// three spawns.
void TestPendingCompileOfBlamedProgramIsBlamed() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(MakeConfig(directory, "crash_with_others:1", outcomes));
	auto             first  = MakeRequest(1, StreamRequestKind::Compile);
	auto             second = MakeRequest(1, StreamRequestKind::Compile);
	first.specialization    = {1};
	second.specialization   = {2};
	CHECK(supervisor.Submit(first).seq == 1);
	CHECK(supervisor.Submit(second).seq == 2);
	CHECK(outcomes.WaitFor(2, OUTCOME_LIMIT));
	CHECK(outcomes.Of(1) == StreamOutcome::Blamed && outcomes.Of(2) == StreamOutcome::Blamed);
	CHECK(CountStarting(ReadLines(ProgressPathOf(directory)), "start 2 ") == 1);
	const auto stats = supervisor.Stats();
	CHECK(stats.crashes == 2 && stats.spawns == 2 && stats.never_retry_programs == 1);
	// No child is started later for the swept request either.
	std::this_thread::sleep_for(QUIET_TIME);
	CHECK(supervisor.Stats().spawns == 2);
	supervisor.Stop();
}

// A child that sends bytes that are no frame breaks the protocol. That is no crash and blames
// nothing, though the request was running, and the lane turns off, since every child of this build
// would do the same. A supervisor that took it as a crash would blame the running request and
// start another child.
void TestProtocolErrorTurnsLaneOff() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(MakeConfig(directory, "bad_frame", outcomes));
	const auto       seq = supervisor.Submit(MakeRequest(1)).seq;
	CHECK(outcomes.WaitFor(1, OUTCOME_LIMIT));
	CHECK(outcomes.Of(seq) == StreamOutcome::Dropped);
	const auto stats = supervisor.Stats();
	CHECK(stats.lane_off && stats.lane_off_reason == "protocol_error");
	CHECK(stats.protocol_errors == 1 && stats.crashes == 0 && stats.blamed == 0);
	CHECK(stats.never_retry == 0 && stats.spawns == 1);
	supervisor.Stop();
}

// A Compile that crashed the child alone is blamed, and a Compile of the same program with another
// specialization is refused, since it would most likely crash the child too. The baseline is a
// Compile of another program, which is queued and answered, so the refusal is the program's.
void TestBlamedCompileBlocksProgram() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(MakeConfig(directory, "crash_with_others:1", outcomes));
	auto             first = MakeRequest(1, StreamRequestKind::Compile);
	first.specialization   = {1};
	CHECK(supervisor.Submit(first).seq == 1);
	CHECK(outcomes.WaitFor(1, OUTCOME_LIMIT));
	CHECK(outcomes.Of(1) == StreamOutcome::Blamed);
	auto second           = MakeRequest(1, StreamRequestKind::Compile);
	second.specialization = {2};
	CHECK(supervisor.Submit(second).status == StreamSubmitStatus::NeverRetry);
	const auto other = supervisor.Submit(MakeRequest(2, StreamRequestKind::Compile));
	CHECK(other.status == StreamSubmitStatus::Queued);
	CHECK(outcomes.WaitFor(2, OUTCOME_LIMIT));
	CHECK(outcomes.Of(other.seq) == StreamOutcome::Answered);
	CHECK(supervisor.Stats().never_retry_programs == 1);
	supervisor.Stop();
}

// Each outcome carries the tag its request was submitted with, Answered, Blamed and Dropped alike.
// Each tag is its own, so an outcome that carried another request's tag fails as well as one that
// carried none.
void TestTagIsEchoed() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor crashing(
	    MakeConfig(directory, fmt::format("crash_with_others:{}", CRASH_SEQ), outcomes));
	const auto answered = crashing.Submit(MakeRequest(1), TAG_ANSWERED).seq;
	CHECK(outcomes.WaitFor(1, OUTCOME_LIMIT));
	const auto blamed = crashing.Submit(MakeRequest(CRASH_SEQ), TAG_BLAMED).seq;
	CHECK(outcomes.WaitFor(2, OUTCOME_LIMIT));
	crashing.Stop();
	const auto answered_completion = outcomes.CompletionOf(answered);
	const auto blamed_completion   = outcomes.CompletionOf(blamed);
	CHECK(answered_completion && answered_completion->outcome == StreamOutcome::Answered &&
	      answered_completion->tag == TAG_ANSWERED);
	CHECK(blamed_completion && blamed_completion->outcome == StreamOutcome::Blamed &&
	      blamed_completion->tag == TAG_BLAMED);

	TempDirectory    hung_directory;
	Outcomes         hung_outcomes;
	StreamSupervisor hung(MakeConfig(hung_directory, "hang:1", hung_outcomes));
	const auto       dropped = hung.Submit(MakeRequest(1), TAG_DROPPED).seq;
	CHECK(WaitForLines(ProgressPathOf(hung_directory), "start 1 ", 1));
	hung.Stop();
	const auto dropped_completion = hung_outcomes.CompletionOf(dropped);
	CHECK(dropped_completion && dropped_completion->outcome == StreamOutcome::Dropped &&
	      dropped_completion->tag == TAG_DROPPED);
}

// The exit of a child the supervisor ended itself is never counted. A hang ends the child by a
// kill, and that adds no crash and keeps no exit code. The baseline is a child that crashes by
// itself on the same single request, whose exit is counted and kept. A live child that Stop ends
// is checked as well, as a record that Stop keeps no code, though no code path could make it fail,
// since Stop never counts a failure.
void TestExitCodeIgnoredAfterParentClose() {
	TempDirectory directory;
	Outcomes      outcomes;
	auto          hung_config    = MakeConfig(directory, "hang:1", outcomes);
	hung_config.request_deadline = SHORT_DEADLINE;
	StreamSupervisor hung(std::move(hung_config));
	(void)hung.Submit(MakeRequest(1));
	CHECK(outcomes.WaitFor(1, SHORT_DEADLINE + DEADLINE_SLACK));
	hung.Stop();
	CHECK(hung.Stats().hangs == 1 && hung.Stats().crashes == 0 && !hung.Stats().last_crash_exit);

	TempDirectory    live_directory;
	Outcomes         live_outcomes;
	StreamSupervisor live(MakeConfig(live_directory, "answer", live_outcomes));
	(void)live.Submit(MakeRequest(1));
	CHECK(live_outcomes.WaitFor(1, OUTCOME_LIMIT));
	live.Stop();
	CHECK(live.Stats().crashes == 0 && !live.Stats().last_crash_exit);

	TempDirectory    crash_directory;
	Outcomes         crash_outcomes;
	StreamSupervisor crashed(MakeConfig(crash_directory, "crash_with_others:1", crash_outcomes));
	(void)crashed.Submit(MakeRequest(1));
	CHECK(crash_outcomes.WaitFor(1, OUTCOME_LIMIT));
	crashed.Stop();
	CHECK(crashed.Stats().crashes == 1 && crashed.Stats().last_crash_exit == FAKE_CRASH_EXIT);
}

// A child that crashes as soon as it connects never starts a request, so nothing is blamed and
// both requests are sent again each time. After the cap the lane turns off. Both complete as
// Dropped before Stop and Submit refuses, which a supervisor that kept restarting would never do.
// The spawn count is read again after a pause as a record that nothing more started, though that
// check could not catch a late spawn, since no request waits by then.
void TestRestartCapTurnsLaneOff() {
	TempDirectory directory;
	Outcomes      outcomes;
	auto          config = MakeConfig(directory, "crash_at_connect", outcomes);
	config.max_restarts  = SMALL_RESTART_CAP;
	StreamSupervisor supervisor(std::move(config));
	(void)supervisor.Submit(MakeRequest(1));
	(void)supervisor.Submit(MakeRequest(2));
	CHECK(outcomes.WaitFor(2, OUTCOME_LIMIT));
	CHECK(outcomes.Of(1) == StreamOutcome::Dropped && outcomes.Of(2) == StreamOutcome::Dropped);
	auto stats = supervisor.Stats();
	CHECK(stats.lane_off && stats.lane_off_reason == "restarts_capped");
	CHECK(stats.spawns == SMALL_RESTART_CAP + 1 && stats.failures == SMALL_RESTART_CAP + 1);
	CHECK(stats.never_retry == 0);
	CHECK(supervisor.Submit(MakeRequest(3)).status == StreamSubmitStatus::LaneOff);
	std::this_thread::sleep_for(QUIET_TIME);
	stats = supervisor.Stats();
	CHECK(stats.spawns == SMALL_RESTART_CAP + 1);
	supervisor.Stop();
}

// Stop with three requests running in a hung child returns within a second, and each request is
// Dropped exactly once, with nothing after Stop. The fake keeps its pipe open while it hangs, as a
// Windows worker does while it joins its threads, so a Stop that waited for the child to close the
// pipe would take the whole hang.
void TestStopIsPrompt() {
	TempDirectory    directory;
	Outcomes         outcomes;
	StreamSupervisor supervisor(MakeConfig(directory, "hang:1", outcomes));
	for (uint64_t index = 1; index <= FAKE_BATCH_SIZE; index++) {
		(void)supervisor.Submit(MakeRequest(index));
	}
	CHECK(WaitForLines(ProgressPathOf(directory), "start ", FAKE_BATCH_SIZE));
	const auto began = Clock::now();
	supervisor.Stop();
	CHECK(Clock::now() - began < STOP_LIMIT);
	CHECK(outcomes.List().size() == FAKE_BATCH_SIZE);
	for (uint64_t seq = 1; seq <= FAKE_BATCH_SIZE; seq++) {
		CHECK(outcomes.Of(seq) == StreamOutcome::Dropped);
	}
	std::this_thread::sleep_for(QUIET_TIME);
	CHECK(outcomes.List().size() == FAKE_BATCH_SIZE);
	CHECK(supervisor.Stats().crashes == 0 && supervisor.Stats().hangs == 0);
}

// An answer that the reader holds until Stop has begun is destroyed, and its request is Dropped,
// not Answered. Without the hook the same request is Answered, as the lazy spawn test shows.
void TestLateReplyAfterStopIsDropped() {
	TempDirectory                  directory;
	Outcomes                       outcomes;
	std::atomic<StreamSupervisor*> watched {nullptr};
	std::atomic<bool>              read {false};
	auto                           config = MakeConfig(directory, "answer", outcomes);
	// Holds the reader on the answer until Stop has begun on the test's thread.
	config.after_reply_read = [&](const StreamResponse&) {
		read             = true;
		const auto limit = Clock::now() + OUTCOME_LIMIT;
		while (Clock::now() < limit && (watched == nullptr || !watched.load()->Stopping())) {
			std::this_thread::sleep_for(POLL_SLICE);
		}
	};
	StreamSupervisor supervisor(std::move(config));
	watched          = &supervisor;
	const auto seq   = supervisor.Submit(MakeRequest(1)).seq;
	const auto until = Clock::now() + OUTCOME_LIMIT;
	while (!read && Clock::now() < until) {
		std::this_thread::sleep_for(POLL_SLICE);
	}
	CHECK(read);
	supervisor.Stop();
	CHECK(outcomes.Of(seq) == StreamOutcome::Dropped);
	CHECK(supervisor.Stats().late_replies == 1 && supervisor.Stats().answered == 0);
}

// A child that never connects fails its spawn at the spawn deadline, not sooner and not at the end
// of its hang, which is what a spawn without a deadline would wait for. With no restarts left the
// lane turns off and the request is Dropped.
void TestSpawnThatNeverConnectsFails() {
	TempDirectory directory;
	Outcomes      outcomes;
	auto          config  = MakeConfig(directory, "never_connect", outcomes);
	config.spawn_deadline = SHORT_DEADLINE;
	config.max_restarts   = 0;
	StreamSupervisor supervisor(std::move(config));
	const auto       began = Clock::now();
	const auto       seq   = supervisor.Submit(MakeRequest(1)).seq;
	CHECK(outcomes.WaitFor(1, SHORT_DEADLINE + DEADLINE_SLACK));
	CHECK(Clock::now() - began >= SHORT_DEADLINE - DEADLINE_EARLY);
	CHECK(outcomes.Of(seq) == StreamOutcome::Dropped);
	const auto stats = supervisor.Stats();
	CHECK(stats.failed_spawns == 1 && stats.lane_off && stats.spawns == 1);
	supervisor.Stop();
}

// The first child never connects, and the second, started on a new pipe, answers. Each spawn's pipe
// has its own name. The first listener was never accepted, so a supervisor that kept it would
// give the second child the same name, which would still connect.
void TestRespawnUsesNewPipeName() {
	TempDirectory directory;
	Outcomes      outcomes;
	auto          config  = MakeConfig(directory, "never_connect_once", outcomes);
	config.spawn_deadline = SHORT_DEADLINE;
	StreamSupervisor supervisor(std::move(config));
	const auto       seq = supervisor.Submit(MakeRequest(1)).seq;
	CHECK(outcomes.WaitFor(1, OUTCOME_LIMIT));
	CHECK(outcomes.Of(seq) == StreamOutcome::Answered);
	const auto stats = supervisor.Stats();
	CHECK(stats.failed_spawns == 1 && stats.spawns == 2 && stats.pipe_names.size() == 2);
	CHECK(stats.pipe_names.size() == 2 && !stats.pipe_names[0].empty() &&
	      stats.pipe_names[0] != stats.pipe_names[1]);
	supervisor.Stop();
}

// Fails the run when the tests take far longer than they should.
void StartWatchdog() {
	std::thread([] {
		std::this_thread::sleep_for(TEST_TIMEOUT);
		std::fprintf(stderr, "shader_store_stream_supervisor_tests: timed out\n");
		std::_Exit(1);
	}).detach();
}

} // namespace

int main(int argc, char** argv) {
	if (argc >= 4 && std::string_view(argv[1]) == FAKE_FLAG) {
		return RunFakeStreamWorker(argv[2], PathFromUtf8(argv[3]));
	}
	StartWatchdog();
	TestRttHistogram();
	TestLazySpawnAnswersAndReadsPriority();
	TestOutstandingCapHolds();
	TestCrashBlamesRequest();
	TestDoneButNoReplyIsResent();
	TestTornLastLineIsIgnored();
	TestDeadlineKillsHungRequest();
	TestHangBlamesOnlyOverdue();
	TestBlockedWriteIsAHang();
	TestBlockedWriteBlamesRunningRequest();
	TestThrowingCallbackIsContained();
	TestPendingCompileOfBlamedProgramIsBlamed();
	TestProtocolErrorTurnsLaneOff();
	TestBlamedCompileBlocksProgram();
	TestTagIsEchoed();
	TestExitCodeIgnoredAfterParentClose();
	TestRestartCapTurnsLaneOff();
	TestStopIsPrompt();
	TestLateReplyAfterStopIsDropped();
	TestSpawnThatNeverConnectsFails();
	TestRespawnUsesNewPipeName();
	if (g_failures == 0) {
		std::puts("Shader store stream supervisor tests passed");
	}
	return g_failures == 0 ? 0 : 1;
}
