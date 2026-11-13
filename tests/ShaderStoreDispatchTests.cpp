// Tests that the emulator binary runs as a shader store worker when it is started with
// --shader-store-worker. The tests start the real kyty_emulator through ChildProcess, so they prove
// the dispatch in Main and the order of the steps in RunShaderStoreWorkerProcess. Each check names
// what else could satisfy it and compares against a baseline that differs only in the behaviour
// under test. The worker's own results are tested in ShaderStoreWorkerTests.
#include "ShaderStoreEbootFixtures.h"
#include "common/childProcess.h"
#include "common/common.h"
#include "common/crashUi.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"
#include "graphics/shader/storegen/streamChannel.h"
#include "graphics/shader/storegen/workerProcess.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <xxhash.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/stat.h>
#include <sys/syscall.h>
#endif

#ifndef KYTY_EMULATOR_PATH
#error "KYTY_EMULATOR_PATH names the kyty_emulator executable and is set by CMake"
#endif

namespace {

using namespace Libs::Graphics::StoreGen;
using namespace StoreGenFixtures;
using namespace std::chrono_literals;
using Common::ChildProcess;
namespace fs = std::filesystem;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

// A worker is quick on these tiny ebootes. The limit is for a loaded runner, and a worker still
// alive after it is killed and counts as a failure.
constexpr auto WORKER_LIMIT = 60s;
// How long a crash may take to end the process once the crash UI is off.
constexpr auto PROMPT_EXIT = 10s;
// What a worker ends with when its guards find the parent gone, from EnterChildMode. Windows ends
// it with 1 and POSIX with 128 + SIGKILL.
[[maybe_unused]] constexpr int GONE_PARENT_WINDOWS = 1;
[[maybe_unused]] constexpr int GONE_PARENT_POSIX   = 137;
// What a crashed process reports. Windows gives the exception code and 3 for abort. POSIX gives
// 128 plus the signal number.
[[maybe_unused]] constexpr int SEGFAULT_WINDOWS  = static_cast<int>(0xC0000005);
[[maybe_unused]] constexpr int ABORT_WINDOWS     = 3;
[[maybe_unused]] constexpr int SIGNAL_BASE       = 128;
[[maybe_unused]] constexpr int SIGABRT_NUMBER    = 6;
[[maybe_unused]] constexpr int SIGSEGV_NUMBER    = 11;
[[maybe_unused]] constexpr int POSIX_STATUS_MASK = 0xFF;
constexpr int                  FAKE_MODE_UNKNOWN = 99;
// The extra words that the recompiler cannot take: a VOPC with an SDWA operand, as in a real game.
constexpr uint32_t SDWA_WORD_FIRST  = 0x7df2d4f9;
constexpr uint32_t SDWA_WORD_SECOND = 0x86060000;
// What a previous worker left in the text file, to see that a new one appends.
constexpr std::string_view EARLIER_TEXT = "EARLIER RUN\n";
// What the recompiler's EXIT prints first.
constexpr std::string_view EXIT_TEXT_MARK = "--- Build ---";
// A line the recompiler writes to stdout while it translates.
constexpr std::string_view RECOMPILER_LOG_MARK = "ShaderRecompiler CS phase";
// The size of an eboot that holds no shader.
constexpr size_t TINY_EBOOT_BYTES = 64;
// How long the Stream smoke test waits for the worker to connect, and for its answer.
constexpr auto STREAM_CONNECT_LIMIT = 30s;
constexpr auto STREAM_ANSWER_LIMIT  = 60s;

// A directory for the files of one test, which is removed with it.
class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		m_path             = fs::temp_directory_path() /
		         ("kyty_store_dispatch_tests_" + std::to_string(Common::CurrentProcessId()) + "_" +
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

void WriteFile(const fs::path& path, const Bytes& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()),
	          static_cast<std::streamsize>(bytes.size()));
}

void WriteText(const fs::path& path, std::string_view text) {
	WriteFile(path, Bytes(text.begin(), text.end()));
}

Bytes ReadFile(const fs::path& path) {
	std::ifstream in(path, std::ios::binary);
	return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string ReadText(const fs::path& path) {
	const auto bytes = ReadFile(path);
	return std::string(bytes.begin(), bytes.end());
}

// The StoreHeader bytes, written as PipelineCache writes them.
Bytes MakeStoreHeader() {
	IR::Writer                       writer;
	auto                             jump_tables = false;
	ShaderRecompiler::Spirv::Profile profile;
	uint32_t                         shared = MAX_SHARED_BYTES;
	IR::IoAll(writer, BUILD_STAMP, jump_tables, profile, shared);
	return std::move(writer.bytes);
}

// A request with its files in the directory, started by this process.
WorkerRequest MakeRequest(const TempDirectory& directory, const std::string& tag,
                          const fs::path& eboot) {
	WorkerRequest request;
	request.store_header       = MakeStoreHeader();
	request.host_subgroup_size = HOST_SUBGROUP_SIZE;
	request.eboot_path         = Utf8FromPath(eboot);
	request.output_path        = Utf8FromPath(directory / (tag + ".generated.tmp"));
	request.progress_path      = Utf8FromPath(directory / (tag + ".progress"));
	request.generated_path     = Utf8FromPath(directory / (tag + ".generated"));
	request.parent_pid         = Common::CurrentProcessId();
	return request;
}

// Writes the request and returns its file.
fs::path WriteRequest(const TempDirectory& directory, const std::string& tag,
                      const WorkerRequest& request) {
	const auto path = directory / (tag + ".request");
	CHECK(WriteWorkerRequestFile(path, request));
	return path;
}

// The file that a worker appends what it prints to.
fs::path TextFileOf(const WorkerRequest& request) {
	return PathFromUtf8(request.progress_path + ".out");
}

// Starts a process through ChildProcess and returns its exit code. One that is still running after
// the limit is killed and gives nothing.
std::optional<int> RunToEnd(const fs::path& program, const std::vector<std::string>& arguments,
                            std::chrono::milliseconds limit) {
	ChildProcess child;
	if (!child.Start(program, arguments)) {
		CHECK(false && "the process did not start");
		return std::nullopt;
	}
	const auto code = child.Wait(limit);
	if (!code) {
		child.Terminate();
		CHECK(false && "the process did not end in time");
	}
	return code;
}

// Starts the real emulator binary as a worker, with the given words after the request.
std::optional<int> RunEmulatorWorker(const fs::path&                 request_file,
                                     const std::vector<std::string>& extra_words = {}) {
	std::vector<std::string> arguments {WORKER_COMMAND_LINE_FLAG, Utf8FromPath(request_file)};
	arguments.insert(arguments.end(), extra_words.begin(), extra_words.end());
	return RunToEnd(KYTY_EMULATOR_PATH, arguments, WORKER_LIMIT);
}

// What a POSIX status keeps of a code. Windows keeps all of it.
int StatusOf(int code) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	return code;
#else
	return code & POSIX_STATUS_MASK;
#endif
}

// Reads a real worker's progress log from its first byte with the reader the supervisor uses. A
// line the reader does not take is counted, so none may be: this is what notices that the worker's
// log format and the parser have drifted apart.
ProgressLogReader ReadRealLog(const WorkerRequest& request) {
	ProgressLogReader reader(PathFromUtf8(request.progress_path));
	reader.BeginSpawnAt(0, ProgressClock::now());
	reader.PollToEnd(ProgressClock::now());
	CHECK(reader.MalformedLines() == 0);
	return reader;
}

// A request that cannot be read is answered with 11, and a missing request file too.
void TestBadRequest() {
	TempDirectory directory;
	WriteText(directory / "garbage.request", "this is not a request");
	CHECK(RunEmulatorWorker(directory / "garbage.request") == WORKER_EXIT_BAD_REQUEST);
	CHECK(RunEmulatorWorker(directory / "absent.request") == WORKER_EXIT_BAD_REQUEST);
}

// The request with an eboot that is there is the baseline for the missing one, so that the 12 is
// the worker's answer to the eboot and not to something else in the request.
void TestMissingEboot() {
	TempDirectory directory;
	auto          missing = MakeRequest(directory, "missing", directory / "no_such_file.eboot");
	CHECK(RunEmulatorWorker(WriteRequest(directory, "missing", missing)) ==
	      WORKER_EXIT_EBOOT_UNREADABLE);

	WriteFile(directory / "junk.eboot", Bytes(TINY_EBOOT_BYTES, 0));
	const auto present = MakeRequest(directory, "present", directory / "junk.eboot");
	CHECK(RunEmulatorWorker(WriteRequest(directory, "present", present)) == WORKER_EXIT_DONE);
}

// Words after the request are never read as emulator options. Parsing them would fail on the
// unknown option, with 1, before the worker ran, and --help would print and give 0.
void TestJunkArgumentsAreIgnored() {
	TempDirectory directory;
	const auto    request = MakeRequest(directory, "junk", directory / "no_such_file.eboot");
	const auto    file    = WriteRequest(directory, "junk", request);
	CHECK(RunEmulatorWorker(file, {"--bogus", "--game", "nowhere", "--help"}) ==
	      WORKER_EXIT_EBOOT_UNREADABLE);
}

// An eboot with no shader in it is a whole run with nothing to do. The file it leaves is the
// generated header, and nothing else could write exactly those bytes. The text file of an earlier
// run is still there with its text first, which only an append leaves.
void TestMinimalRequestFinishes() {
	TempDirectory directory;
	WriteFile(directory / "tiny.eboot", Bytes(TINY_EBOOT_BYTES, 0));
	const auto request = MakeRequest(directory, "tiny", directory / "tiny.eboot");
	WriteText(TextFileOf(request), EARLIER_TEXT);

	CHECK(RunEmulatorWorker(WriteRequest(directory, "tiny", request)) == WORKER_EXIT_DONE);
	const auto expected = EncodeGeneratedHeader(request.store_header, request.generator_version,
	                                            ComputeEbootIdentity(TINY_EBOOT_BYTES, {}));
	CHECK(ReadFile(PathFromUtf8(request.output_path)) == expected);
	CHECK(ReadText(TextFileOf(request)).starts_with(EARLIER_TEXT));
	// No program means no predict, start or drop line, and the run still ends with its summary.
	const auto log = ReadRealLog(request);
	CHECK(log.KindCount(ProgressKind::Summary) == 1 && log.Summary() &&
	      log.Summary()->exit_code == 0);
	CHECK(log.KindCount(ProgressKind::Predict) == 0 && log.KindCount(ProgressKind::Start) == 0);
}

// What the worker prints lands in the text file, after what an earlier run left, and so does the
// text of a recompiler EXIT. The same program without the unsupported words is the baseline: it
// ends with 0 and prints the recompiler's log but no EXIT text, so the text is the crash's.
void TestRecompilerExitIsCaptured() {
	TempDirectory directory;
	auto          crashing     = ProgramSmall();
	crashing.extra_words       = {SDWA_WORD_FIRST, SDWA_WORD_SECOND};
	const fs::path crash_eboot = directory / "crash.eboot";
	WriteFile(crash_eboot, BuildEboot({crashing}));
	const auto crash = MakeRequest(directory, "crash", crash_eboot);
	WriteText(TextFileOf(crash), EARLIER_TEXT);
	const auto code = RunEmulatorWorker(WriteRequest(directory, "crash", crash));
	CHECK(code && StatusOf(*code) == StatusOf(WORKER_EXIT_RECOMPILER_EXIT));
	const auto text = ReadText(TextFileOf(crash));
	CHECK(text.starts_with(EARLIER_TEXT));
	CHECK(text.find(EXIT_TEXT_MARK) != std::string::npos);
	CHECK(text.find(RECOMPILER_LOG_MARK) != std::string::npos);
	// The program is translated while it is predicted, so the run ends at its predict line, which
	// is the last line and the one that blames it. It never got to an order, a start or the
	// summary.
	const auto crash_log = ReadRealLog(crash);
	CHECK(crash_log.KindCount(ProgressKind::Predict) >= 1);
	CHECK(crash_log.LastLine() && crash_log.LastLine()->kind == ProgressKind::Predict);
	CHECK(crash_log.KindCount(ProgressKind::Order) == 0 && !crash_log.LatestOrder());
	CHECK(crash_log.KindCount(ProgressKind::Start) == 0 && !crash_log.LastStart());
	CHECK(crash_log.KindCount(ProgressKind::Summary) == 0 && !crash_log.Summary());
	// A crash while predicting leaves no `prediction` line, which is written after it, so the
	// blame stays on the predict line.
	CHECK(crash_log.KindCount(ProgressKind::Prediction) == 0 && crash_log.Predictions().empty());

	const fs::path plain_eboot = directory / "plain.eboot";
	WriteFile(plain_eboot, BuildEboot({ProgramSmall()}));
	const auto plain = MakeRequest(directory, "plain", plain_eboot);
	WriteText(TextFileOf(plain), EARLIER_TEXT);
	CHECK(RunEmulatorWorker(WriteRequest(directory, "plain", plain)) == WORKER_EXIT_DONE);
	const auto plain_text = ReadText(TextFileOf(plain));
	CHECK(plain_text.starts_with(EARLIER_TEXT));
	// The recompiler logs to stdout as it works, so a worker whose output was not redirected would
	// leave none of this in the file. The EXIT text is the crash's alone.
	CHECK(plain_text.find(RECOMPILER_LOG_MARK) != std::string::npos);
	CHECK(plain_text.find(EXIT_TEXT_MARK) == std::string::npos);
	// The same program without the crash gets as far as the summary.
	const auto plain_log = ReadRealLog(plain);
	CHECK(plain_log.KindCount(ProgressKind::Predict) >= 1 &&
	      plain_log.KindCount(ProgressKind::Order) >= 1);
	CHECK(plain_log.KindCount(ProgressKind::Start) >= 1);
	CHECK(plain_log.KindCount(ProgressKind::Summary) == 1 && plain_log.Summary());
	// One program, predicted, and its candidate count is the one the ordering holds. The crashing
	// run above has no such row, so the row is the worker's answer and not a leftover.
	CHECK(plain_log.KindCount(ProgressKind::Prediction) == 1 &&
	      plain_log.Predictions().size() == 1);
	CHECK(plain_log.LatestOrder().has_value());
	if (plain_log.Predictions().size() == 1 && plain_log.LatestOrder()) {
		CHECK(plain_log.Predictions()[0].outcome == PredictionOutcome::Predicted);
		CHECK(plain_log.Predictions()[0].candidates == plain_log.LatestOrder()->rank_or_count);
		CHECK(plain_log.Predictions()[0].candidates >= 1);
	}
}

// A worker whose parent is already gone ends at once, from EnterChildMode, and never runs. The
// baseline is the same request started by a live parent, which runs and answers 12 for its missing
// eboot. A worker that skipped the guards would answer 12 here too.
void TestDeadParentEndsTheWorker() {
	TempDirectory directory;
	uint32_t      dead_pid = 0;
	{
		ChildProcess                   quick;
		const std::vector<std::string> arguments {"--fake-mode", "exit"};
		CHECK(quick.Start(Common::ExecutablePath(), arguments));
		dead_pid = quick.Pid();
		CHECK(quick.Wait(WORKER_LIMIT).has_value());
	}
	auto request       = MakeRequest(directory, "orphan", directory / "no_such_file.eboot");
	request.parent_pid = dead_pid;
	const auto code    = RunEmulatorWorker(WriteRequest(directory, "orphan", request));
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	constexpr int EXPECTED = GONE_PARENT_WINDOWS;
#else
	constexpr int EXPECTED = GONE_PARENT_POSIX;
#endif
	CHECK(code && StatusOf(*code) == EXPECTED);
	CHECK(code != WORKER_EXIT_EBOOT_UNREADABLE);
}

// Starts this test binary in a mode of its own and returns its exit code.
std::optional<int> RunFakeMode(const std::string& mode, const std::string& argument = {}) {
	std::vector<std::string> arguments {"--fake-mode", mode};
	if (!argument.empty()) {
		arguments.push_back(argument);
	}
	return RunToEnd(Common::ExecutablePath(), arguments, PROMPT_EXIT);
}

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
// Raises the soft core file size limit to the hard one, so that the children below start with a
// limit that is not zero and the zero that DisableCrashUi sets is a difference that the call made.
// A runner whose hard limit is zero has no such baseline, and the checks that need it are skipped.
bool RaiseCoreLimitForBaseline() {
	rlimit limit {};
	if (getrlimit(RLIMIT_CORE, &limit) != 0 || limit.rlim_max == 0) {
		return false;
	}
	limit.rlim_cur = limit.rlim_max;
	return setrlimit(RLIMIT_CORE, &limit) == 0;
}

// Whether this process raised its core limit, so its children start with one that is not zero.
bool g_core_baseline = false;
#endif

// DisableCrashUi turns the reporting off for the process. The state is read inside the process
// after the call, and the mode that never calls it is the baseline for what a start leaves.
void TestCrashUiState() {
	TempDirectory directory;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// A child inherits the error mode, and a shell or a CI agent may have set it, so this process
	// clears it for the two children below to have a baseline that is really off.
	const UINT inherited = SetErrorMode(0);
#endif
	CHECK(RunFakeMode("state-off", Utf8FromPath(directory / "off.txt")) == 0);
	CHECK(RunFakeMode("state-on", Utf8FromPath(directory / "on.txt")) == 0);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// Back as it was, so the crash tests are not shown a dialog if DisableCrashUi stopped working.
	SetErrorMode(inherited);
#endif
	const auto off = ReadText(directory / "off.txt");
	const auto on  = ReadText(directory / "on.txt");
	CHECK(!off.empty() && !on.empty());
	// "disabled" is what the mode writes when the crash UI is off.
	CHECK(on == "disabled");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// With the inherited mode cleared the difference is the call.
	CHECK(off == "enabled");
#else
	// With the soft core limit raised the difference is the call. A runner whose hard limit is zero
	// cannot show it.
	if (g_core_baseline) {
		CHECK(off == "enabled");
	} else {
		std::puts("The core limit baseline is skipped, since the hard limit is zero.");
	}
#endif
}

// A real access violation and a real abort end the process promptly once the crash UI is off.
void TestCrashEndsPromptly() {
	const auto null_code  = RunFakeMode("null");
	const auto abort_code = RunFakeMode("abort");
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	CHECK(null_code == SEGFAULT_WINDOWS);
	CHECK(abort_code == ABORT_WINDOWS);
#else
	CHECK(null_code == SIGNAL_BASE + SIGSEGV_NUMBER);
	CHECK(abort_code == SIGNAL_BASE + SIGABRT_NUMBER);
#endif
}

#if defined(__linux__)
// The nice value of a thread, from field 19 of its stat file. The command name is in parentheses
// and may hold spaces, so the fields are counted after the last parenthesis. They start at field 3,
// the state.
std::optional<int> NiceOfTask(const fs::path& stat_file) {
	const auto text = ReadText(stat_file);
	const auto last = text.rfind(')');
	if (last == std::string::npos) {
		return std::nullopt;
	}
	constexpr int NICE_FIELD                = 19;
	constexpr int FIRST_FIELD_AFTER_COMMAND = 3;
	std::string   rest                      = text.substr(last + 1);
	size_t        at                        = 0;
	for (int field = FIRST_FIELD_AFTER_COMMAND; field <= NICE_FIELD; field++) {
		at = rest.find_first_not_of(' ', at);
		if (at == std::string::npos) {
			return std::nullopt;
		}
		const auto end = rest.find(' ', at);
		if (field == NICE_FIELD) {
			return std::stoi(rest.substr(at, end - at));
		}
		at = end;
	}
	return std::nullopt;
}

// The soft limit of a process on the core file size, from its limits file. Nothing when the file
// or the line cannot be read.
std::optional<unsigned long long> CoreLimitOf(uint32_t pid) {
	const auto        text  = ReadText(fs::path("/proc") / std::to_string(pid) / "limits");
	const std::string label = "Max core file size";
	const auto        at    = text.find(label);
	if (at == std::string::npos) {
		return std::nullopt;
	}
	const auto first = text.find_first_not_of(' ', at + label.size());
	if (first == std::string::npos) {
		return std::nullopt;
	}
	// "unlimited" is no number, and stands for a limit that is not zero.
	if (!std::isdigit(static_cast<unsigned char>(text[first]))) {
		return std::numeric_limits<unsigned long long>::max();
	}
	return std::stoull(text.substr(first));
}

// Every thread of the worker is niced at least ten steps above the spawner's own thread, or 19.
// The worker is parked on an eboot that is a FIFO nobody writes, so it sits in the open call after
// its guards were armed. Its main thread has the lowered value from EnterChildMode, and the
// watchdog thread inherits it. A worker that skipped EnterChildMode keeps the spawner's value.
// The same parked worker has been through DisableCrashUi, so its core limit is zero while this
// process's raised one is not, which is what shows that the worker made the call.
void TestEveryThreadIsNiced() {
	TempDirectory directory;
	const auto    fifo = directory / "parked.eboot";
	CHECK(mkfifo(fifo.c_str(), 0600) == 0);
	const auto request = MakeRequest(directory, "parked", fifo);
	const auto file    = WriteRequest(directory, "parked", request);

	constexpr int NICE_STEP = 10;
	constexpr int MAX_NICE  = 19;
	const int     spawner   = getpriority(PRIO_PROCESS, static_cast<id_t>(syscall(SYS_gettid)));
	const int     minimum   = std::min(spawner + NICE_STEP, MAX_NICE);

	ChildProcess                   child;
	const std::vector<std::string> arguments {WORKER_COMMAND_LINE_FLAG, Utf8FromPath(file)};
	CHECK(child.Start(KYTY_EMULATOR_PATH, arguments));
	const fs::path tasks = fs::path("/proc") / std::to_string(child.Pid()) / "task";
	// The main thread and the watchdog make two, and the watchdog starts after the priority change.
	constexpr size_t PARKED_THREADS = 2;
	const auto       deadline       = std::chrono::steady_clock::now() + WORKER_LIMIT;
	size_t           count          = 0;
	while (std::chrono::steady_clock::now() < deadline) {
		count = 0;
		std::error_code error;
		for (const auto& entry: fs::directory_iterator(tasks, error)) {
			(void)entry;
			count++;
		}
		if (count >= PARKED_THREADS || child.Poll()) {
			break;
		}
		std::this_thread::sleep_for(10ms);
	}
	CHECK(!child.Poll().has_value());
	CHECK(count == PARKED_THREADS);
	std::error_code error;
	for (const auto& entry: fs::directory_iterator(tasks, error)) {
		const auto nice = NiceOfTask(entry.path() / "stat");
		CHECK(nice && *nice >= minimum);
	}
	if (g_core_baseline) {
		const auto core = CoreLimitOf(child.Pid());
		CHECK(core && *core == 0);
	} else {
		std::puts("The worker's core limit check is skipped, since the hard limit is zero.");
	}
	child.Terminate();
}
#endif

// A Plan request for the small fixture program, with registers written by hand. Any registers a
// dispatch could have will do, since the smoke test only needs the worker to translate it.
StreamRequest SmokePlanRequest() {
	const auto    words = BuildCode(ProgramSmall());
	StreamRequest request;
	request.seq       = 1;
	request.kind      = StreamRequestKind::Plan;
	const auto* bytes = reinterpret_cast<const uint8_t*>(words.data());
	request.code.assign(bytes, bytes + words.size() * sizeof(uint32_t));
	request.code_hash                 = XXH3_64bits(request.code.data(), request.code.size());
	auto& regs                        = request.dispatch.cs_regs;
	regs.num_thread_x                 = THREADS_X;
	regs.num_thread_y                 = 1;
	regs.num_thread_z                 = 1;
	regs.wave_size                    = static_cast<uint8_t>(MOTOR_DISPATCH.wave_size);
	regs.user_sgpr                    = USER_SGPR_COUNT;
	regs.tgid_x_en                    = true;
	request.dispatch.workgroup_counts = {1, 1, 1};
	request.user_data_count           = USER_SGPR_COUNT;
	return request;
}

// Reads one answer, and ends the child to free the read when none comes in time.
std::optional<StreamResponse> ReadAnswer(StreamChannel& channel, ChildProcess& child) {
	std::promise<std::optional<StreamResponse>> done;
	auto                                        answer = done.get_future();
	std::thread                                 reader([&] {
		const auto read = channel.ReadFrame();
		done.set_value(read.status == StreamReadStatus::Frame ? DecodeStreamResponse(read.frame)
		                                                      : std::nullopt);
	});
	if (answer.wait_for(STREAM_ANSWER_LIMIT) != std::future_status::ready) {
		child.Terminate();
		channel.Interrupt();
	}
	reader.join();
	return answer.get();
}

// The emulator binary started in Stream mode connects to this process's pipe, finds that the
// pipe's server is the parent its request names, answers a Plan and ends with 0 once the pipe
// closes. A worker that asked for the wrong end's process id would see its own, refuse the pipe
// and end with WORKER_EXIT_PEER_MISMATCH, and the Accept here takes only the child's own id, so
// both ends are checked across two real processes. An Ok answer with plan bytes is something only a
// worker that translated the program can send. The log opens with the priority the child runs at,
// which on Windows is the below normal class that ChildProcess gives it, or idle under an idle
// parent.
void TestStreamWorkerAnswersOverPipe() {
	TempDirectory directory;
	auto          listener = StreamPipeListener::Create("dispatch");
	CHECK(listener.has_value());
	if (!listener) {
		return;
	}
	auto request        = MakeRequest(directory, "stream", directory / "unused.eboot");
	request.mode        = WorkerMode::Stream;
	request.stream_pipe = listener->Name();
	const auto                     file = WriteRequest(directory, "stream", request);
	ChildProcess                   child;
	const std::vector<std::string> arguments {WORKER_COMMAND_LINE_FLAG, Utf8FromPath(file)};
	CHECK(child.Start(KYTY_EMULATOR_PATH, arguments));
	auto pipe = listener->Accept(STREAM_CONNECT_LIMIT, child.Pid());
	CHECK(pipe.has_value());
	if (pipe) {
		StreamChannel channel(std::move(*pipe));
		const auto    plan = SmokePlanRequest();
		CHECK(channel.Send(plan));
		const auto answer = ReadAnswer(channel, child);
		CHECK(answer && answer->seq == plan.seq && answer->status == StreamStatus::Ok &&
		      !answer->payload.empty());
		channel.Close();
	}
	const auto code = child.Wait(WORKER_LIMIT);
	if (!code) {
		child.Terminate();
	}
	CHECK(code == WORKER_EXIT_DONE);
	const auto log = ReadText(PathFromUtf8(request.progress_path));
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// ChildProcess gives a child idle class instead when the parent itself runs at idle.
	CHECK(log.starts_with("priority below_normal\n") || log.starts_with("priority idle\n"));
#else
	CHECK(log.starts_with("priority nice_"));
#endif
	CHECK(log.find("done 1 ok\n") != std::string::npos);
}

// What this binary does when started with --fake-mode, which is how the crash tests get a process
// of their own. It answers FAKE_MODE_UNKNOWN to a mode it does not know.
int RunFakeModeMain(const std::vector<std::string>& args) {
	if (args.empty()) {
		return FAKE_MODE_UNKNOWN;
	}
	const auto& mode = args[0];
	if (mode == "exit") {
		return 0;
	}
	if (mode == "state-on" || mode == "state-off") {
		if (mode == "state-on") {
			Common::DisableCrashUi();
		}
		bool disabled = false;
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
		const UINT wanted = SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX;
		disabled          = (GetErrorMode() & wanted) == wanted;
#else
		rlimit limit {};
		disabled = getrlimit(RLIMIT_CORE, &limit) == 0 && limit.rlim_cur == 0;
#endif
		if (args.size() > 1) {
			WriteText(PathFromUtf8(args[1]), disabled ? "disabled" : "enabled");
		}
		return 0;
	}
	if (mode == "null" || mode == "abort") {
		Common::DisableCrashUi();
		if (mode == "abort") {
			std::abort();
		}
		volatile int* nothing = nullptr;
		*nothing              = 1;
		return 0;
	}
	return FAKE_MODE_UNKNOWN;
}

} // namespace

int main(int argc, char** argv) {
	if (argc >= 3 && std::string_view(argv[1]) == "--fake-mode") {
		return RunFakeModeMain(std::vector<std::string>(argv + 2, argv + argc));
	}
#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
	g_core_baseline = RaiseCoreLimitForBaseline();
#endif
	TestBadRequest();
	TestMissingEboot();
	TestJunkArgumentsAreIgnored();
	TestMinimalRequestFinishes();
	TestRecompilerExitIsCaptured();
	TestDeadParentEndsTheWorker();
	TestCrashUiState();
	TestCrashEndsPromptly();
	TestStreamWorkerAnswersOverPipe();
#if defined(__linux__)
	TestEveryThreadIsNiced();
#endif
	if (g_failures == 0) {
		std::puts("Shader store dispatch tests passed");
	}
	return g_failures == 0 ? 0 : 1;
}
