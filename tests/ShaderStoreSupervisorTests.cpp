// Tests the supervisor of a shader store worker. The worker is a fake: this test binary starts
// itself with --fake-store-worker <mode> <request file>, writes the files a worker writes, and
// fails in the way its mode says. That runs the same on every operating system and needs no game
// and no device. Each check names what else could satisfy it and compares against a baseline that
// differs only in the behaviour under test.
#include "common/childProcess.h"
#include "common/file.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/generationReport.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/storeGenerationGate.h"
#include "graphics/shader/storegen/storeSupervisor.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <csignal>
#include <unistd.h>
#endif

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;
using namespace std::chrono_literals;
namespace fs = std::filesystem;
using Bytes  = std::vector<uint8_t>;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

// ---- The fake worker's world: three programs of two candidates each. ----

constexpr uint64_t PROGRAM_BASE      = 0xABC000;
constexpr uint64_t ID_BASE           = 5000;
constexpr size_t   PROGRAMS          = 3;
constexpr size_t   CANDIDATES_EACH   = 2;
constexpr uint64_t DROPPED_HASH      = 0xD0D0;
constexpr uint64_t TOO_LARGE_HASH    = 0xF7F7;
constexpr uint64_t PUSH_DROPPED_HASH = 0xE1E1;
constexpr uint32_t GENERATOR_VERSION = 9;
constexpr size_t   STORE_HEADER_SIZE = 16;
constexpr uint64_t EBOOT_SIZE        = 1234;
constexpr uint64_t IDENTITY_HASH     = 0xFEED;
constexpr int      CRASH_EXIT        = 321;
// The stale file that the stale_gone fake looks for, and its exit when the file is still there.
constexpr const char* STALE_PROBE_NAME       = "title.generated.4243.tmp";
constexpr int         STALE_STILL_THERE_EXIT = WORKER_EXIT_EBOOT_UNREADABLE;
// How long the fake sleeps where it is meant to hang, which is longer than any test waits.
constexpr int HANG_SECONDS = 30;
// The rank at which crash_damaged_start crashes, which is the first candidate of program 1.
constexpr uint64_t CRASH_DAMAGED_RANK = 2;
// The name that the test's sink adds to the output path for its marker file, and how long the
// truncating fake waits for it, in steps of 10 ms.
constexpr const char* TWO_RECORDS_MARKER = ".two";
constexpr int         MARKER_WAIT_STEPS  = 6000;
// How long the delayed fake takes to create its file, which is many polls of the supervisor.
constexpr auto CREATE_DELAY = 2000ms;
// How long the new file stays at its header.
constexpr auto HEADER_PAUSE = 1000ms;
// Lines of 25 bytes that are more than the 1 MiB that one poll of the log reads.
constexpr size_t NOISE_LINES = 130000;
// How much processor time the burning fakes spend, and the least a report must then show. The
// baseline pays only for starting the process, which stays under the baseline limit.
constexpr double BURN_CPU_MS          = 400.0;
constexpr double BURN_MIN_REPORTED_MS = 250.0;
constexpr double BASELINE_MAX_CPU_MS  = 150.0;
// The memory the alloc fake takes, and the threshold that a measured peak must pass. A child that
// took nothing stays far below it.
constexpr uint64_t ALLOC_MEGABYTES      = 64;
constexpr uint64_t BYTES_PER_MEGABYTE   = 1024 * 1024;
constexpr uint64_t ALLOC_MIN_PEAK_BYTES = 48 * BYTES_PER_MEGABYTE;
constexpr size_t   PAGE_STRIDE_BYTES    = 4096;
// The text file of the out mode is larger than what a failure keeps, with a marker at each end so
// that the part that was kept can be told.
constexpr size_t      OUT_FILE_BYTES = 6000;
constexpr const char* OUT_HEAD_MARK  = "HEAD-MARK";
constexpr const char* OUT_TAIL_MARK  = "TAIL-MARK";
// How long the slow fake waits before its first predict line, and how long a slow sink takes.
constexpr int SLOW_OPEN_MS = 600;
// How much of the 600 ms delay the supervisor's view of it may lose, to poll granularity and a
// loaded machine. The slow run must show at least the delay less this over the normal run.
constexpr int    OPEN_MARGIN_MS   = 200;
constexpr auto   SLOW_SINK_DELAY  = 200ms;
constexpr double SLOW_SINK_MIN_MS = 150.0;

uint64_t ProgramHash(size_t program) {
	return PROGRAM_BASE + program;
}
uint64_t ModuleIdOf(size_t program, size_t candidate) {
	return ID_BASE + program * 10 + candidate;
}

// One candidate of the fake's ordering.
struct FakeCandidate {
	size_t   program;
	size_t   index;
	uint64_t hash;
	uint64_t id;
};

// The ordering for a skip set, which changes with it, as a real worker's does.
std::vector<FakeCandidate> FakeOrder(const std::vector<uint64_t>& skip) {
	std::vector<FakeCandidate> order;
	for (size_t program = 0; program < PROGRAMS; program++) {
		if (std::ranges::find(skip, ProgramHash(program)) != skip.end()) {
			continue;
		}
		for (size_t index = 0; index < CANDIDATES_EACH; index++) {
			order.push_back({program, index, ProgramHash(program), ModuleIdOf(program, index)});
		}
	}
	return order;
}

uint64_t OrderHashOf(const std::vector<FakeCandidate>& order) {
	uint64_t hash = 0x9E3779B97F4A7C15ull;
	for (const auto& candidate: order) {
		hash = (hash ^ candidate.id) * 0x100000001B3ull;
	}
	return hash;
}

Bytes StoreHeader(uint8_t seed = 0xA0) {
	Bytes header(STORE_HEADER_SIZE);
	for (size_t i = 0; i < header.size(); i++) {
		header[i] = static_cast<uint8_t>(seed + i);
	}
	return header;
}

Bytes GeneratedHeaderBytes(const WorkerRequest& request, uint8_t seed = 0xA0) {
	(void)request;
	return EncodeGeneratedHeader(StoreHeader(seed), GENERATOR_VERSION,
	                             {.size = EBOOT_SIZE, .program_hash = IDENTITY_HASH});
}

// A module record that the supervisor decodes: a compute module with this id.
Bytes ModuleRecord(uint64_t id, uint64_t hash, ShaderType stage = ShaderType::Compute) {
	const std::vector<uint32_t>                        static_state;
	const ShaderRecompiler::IR::ResourcePlan           plan;
	const ShaderRecompiler::IR::CompiledShaderInfo     info;
	const ShaderRecompiler::IR::ResourceSpecialization specialization;
	const std::vector<uint32_t> spirv {0x07230203u, static_cast<uint32_t>(id)};
	return EncodeModuleRecord({.stage           = stage,
	                           .hash            = hash,
	                           .user_data_count = 0,
	                           .static_state    = static_state,
	                           .plan            = plan,
	                           .id              = id,
	                           .info            = info,
	                           .specialization  = specialization,
	                           .vertex_formats  = {},
	                           .rect_parameters = {},
	                           .mesh_task       = false,
	                           .spirv           = spirv});
}

// ---- The fake worker. ----

// A progress log and the output file, written the way the worker writes them.
class FakeFiles {
public:
	explicit FakeFiles(const WorkerRequest& request)
	    : m_request(request), m_log(PathFromUtf8(request.progress_path), std::ios::app) {}

	void Line(const std::string& text) {
		m_log << text << '\n';
		m_log.flush();
	}
	static std::string Hex(uint64_t value) {
		char text[20];
		std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
		return text;
	}

	// Starts the output as a worker does: a new file with its header, or the kept bytes of one.
	void OpenOutput(const Bytes& header) {
		const auto path = PathFromUtf8(m_request.output_path);
		if (m_request.kept == 0) {
			m_out.open(path, std::ios::binary | std::ios::trunc);
			Write(header);
			return;
		}
		fs::resize_file(path, m_request.kept);
		m_out.open(path, std::ios::binary | std::ios::app);
	}
	void Write(const Bytes& bytes) {
		m_out.write(reinterpret_cast<const char*>(bytes.data()),
		            static_cast<std::streamsize>(bytes.size()));
		m_out.flush();
	}
	void Frame(const Bytes& record) { Write(FrameStoreRecord(record)); }
	// Appends text to the log in one write, as a worker that logs a lot in a burst would.
	void RawLine(const std::string& text) {
		m_log << text;
		m_log.flush();
	}
	void Close() { m_out.close(); }

private:
	WorkerRequest m_request;
	std::ofstream m_log;
	std::ofstream m_out;
};

void SleepSeconds(int seconds) {
	std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

// The processor time this process has used, in milliseconds.
double OwnCpuMs() {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	FILETIME creation {};
	FILETIME exited {};
	FILETIME kernel {};
	FILETIME user {};
	GetProcessTimes(GetCurrentProcess(), &creation, &exited, &kernel, &user);
	constexpr double TICKS_PER_MILLISECOND = 10000.0;
	const auto       ticks                 = [](const FILETIME& time) {
        return static_cast<double>((static_cast<uint64_t>(time.dwHighDateTime) << 32) |
		                                                 time.dwLowDateTime);
	};
	return (ticks(kernel) + ticks(user)) / TICKS_PER_MILLISECOND;
#else
	timespec now {};
	clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &now);
	constexpr double MS_PER_SECOND      = 1000.0;
	constexpr double NANOSECONDS_PER_MS = 1e6;
	return (static_cast<double>(now.tv_sec) * MS_PER_SECOND) +
	       (static_cast<double>(now.tv_nsec) / NANOSECONDS_PER_MS);
#endif
}

// Spins until this process has used the given processor time. It reads its own clock, so a loaded
// machine only makes it take longer in wall time.
void BurnCpu(double milliseconds) {
	const double       start = OwnCpuMs();
	volatile uint64_t  sink  = 0;
	constexpr uint64_t BATCH = 100000;
	while (OwnCpuMs() - start < milliseconds) {
		for (uint64_t i = 0; i < BATCH; ++i) {
			sink = sink + i;
		}
	}
}

// Where the alloc fake keeps its block. A volatile pointer stops the compiler from dropping an
// allocation that nothing reads.
char* volatile g_block = nullptr;

// Takes memory and makes every page resident.
void TakeResidentMemory(uint64_t megabytes) {
	const uint64_t bytes = megabytes * BYTES_PER_MEGABYTE;
	g_block              = static_cast<char*>(std::malloc(bytes));
	if (g_block == nullptr) {
		return;
	}
	for (uint64_t offset = 0; offset < bytes; offset += PAGE_STRIDE_BYTES) {
		g_block[offset] = static_cast<char>(offset);
	}
}

// Appends to the text file of the worker, which the worker process writes its output to.
void AppendWorkerOutText(const WorkerRequest& request, const std::string& text) {
	std::ofstream(PathFromUtf8(request.progress_path + ".out"), std::ios::app | std::ios::binary)
	    << text;
}

// A text file larger than what a failure keeps, with a marker at the very start and at the very
// end, and filler between.
std::string LargeOutText() {
	std::string text = OUT_HEAD_MARK;
	text.append(OUT_FILE_BYTES - std::string(OUT_HEAD_MARK).size() -
	                std::string(OUT_TAIL_MARK).size(),
	            '.');
	text += OUT_TAIL_MARK;
	return text;
}

// The index of this spawn, kept in a file next to the request, which a worker restarted by the
// supervisor can tell apart from the first.
int SpawnIndex(const WorkerRequest& request) {
	const auto path  = PathFromUtf8(request.progress_path + ".spawns");
	int        count = 0;
	if (std::ifstream in(path); in) {
		in >> count;
	}
	std::ofstream(path, std::ios::trunc) << count + 1;
	return count;
}

// Splits `name:a:b` at the colons.
std::vector<std::string> SplitMode(const std::string& mode) {
	std::vector<std::string> words;
	size_t                   begin = 0;
	while (true) {
		const auto end = mode.find(':', begin);
		words.push_back(mode.substr(begin, end == std::string::npos ? end : end - begin));
		if (end == std::string::npos) {
			return words;
		}
		begin = end + 1;
	}
}

bool Contains(const std::vector<uint64_t>& list, uint64_t value) {
	return std::ranges::find(list, value) != list.end();
}

// What a fake worker does, apart from the modes that end before it has a request to read.
struct FakeRun {
	WorkerRequest              request;
	std::vector<std::string>   mode;
	int                        spawn = 0;
	std::vector<FakeCandidate> order;
	FakeFiles                  files;

	FakeRun(const WorkerRequest& r, std::vector<std::string> m, int s)
	    : request(r), mode(std::move(m)), spawn(s), order(FakeOrder(r.skip_hashes)), files(r) {}

	[[nodiscard]] bool     Is(const char* name) const { return mode[0] == name; }
	[[nodiscard]] uint64_t Arg(size_t index) const {
		return index < mode.size() ? std::stoull(mode[index]) : 0;
	}
};

// A frame that does not unframe, and a good one after it, in one write each, so that the damage has
// bytes after it and is final.
void FakeWriteDamage(FakeRun& run) {
	auto bad = FrameStoreRecord(ModuleRecord(ModuleIdOf(9, 0), ProgramHash(9)));
	bad[sizeof(StoreFrame) + 2] ^= 0xFF;
	run.files.Write(bad);
	run.files.Frame(ModuleRecord(ModuleIdOf(9, 1), ProgramHash(9)));
}

// The `prediction` line of a program. Program 0 is predicted, with a count of candidates that
// grows with the spawn so that a later spawn's row can be told from an earlier one. Program 1 has
// calls and program 2 has problems, so each outcome word reaches the statistics.
std::string FakePredictionLine(size_t program, int spawn) {
	static const char* const OUTCOMES[PROGRAMS] = {"predicted", "calls", "problems"};
	const uint64_t           candidates = program == 0 ? 1 + static_cast<uint64_t>(spawn) : 0;
	return "prediction " + FakeFiles::Hex(ProgramHash(program)) + " " + OUTCOMES[program] + " " +
	       std::to_string(candidates);
}

// Whether no program after this one is left in by the skip list.
bool IsLastUnskipped(const std::vector<uint64_t>& skipped, size_t program) {
	for (size_t later = program + 1; later < PROGRAMS; later++) {
		if (!Contains(skipped, ProgramHash(later))) {
			return false;
		}
	}
	return true;
}

// The predict lines, each followed by its prediction line, and the crash at a predict line when the
// mode says so. The crash comes before the prediction line, as the worker's does.
void FakePredict(FakeRun& run) {
	for (size_t program = 0; program < PROGRAMS; program++) {
		const auto hash = ProgramHash(program);
		if (Contains(run.request.skip_hashes, hash)) {
			continue;
		}
		// After a fresh restart this mode predicts only program 0 again, so the others show whether
		// the rows of the first spawn were kept.
		if (run.Is("fresh_predictions") && run.spawn >= 1 && program != 0) {
			continue;
		}
		run.files.Line("predict " + FakeFiles::Hex(hash));
		if (run.Is("crash_predict") && run.Arg(1) == program) {
			std::_Exit(CRASH_EXIT);
		}
		if (run.Is("crash_damaged_predict") && run.spawn == 0 && program == 1) {
			std::_Exit(CRASH_EXIT);
		}
		// This mode ends at the predict line of the last program the request leaves in, which is
		// where a crash while predicting that program would end it.
		if (run.Is("exit_after_line") && IsLastUnskipped(run.request.skip_hashes, program)) {
			std::_Exit(static_cast<int>(run.Arg(1)));
		}
		run.files.Line(FakePredictionLine(program, run.spawn));
	}
}

// The drop lines of the drops mode, which come before the crash it ends in.
void FakeDrops(FakeRun& run) {
	run.files.Line("drop 1 " + FakeFiles::Hex(DROPPED_HASH) + " resource_count");
	run.files.Line("drop 1 " + FakeFiles::Hex(TOO_LARGE_HASH) + " frame_too_large");
	run.files.Line("drop 1 " + FakeFiles::Hex(PUSH_DROPPED_HASH) +
	               " push_start predicted=3 actual=4");
}

// What goes wrong at the start of a candidate, by mode. Returns true when the run ends there.
bool FakeFaultAtStart(FakeRun& run, uint64_t rank, const FakeCandidate& candidate) {
	if (run.Is("crash_start") && run.Arg(1) == rank) {
		std::_Exit(CRASH_EXIT);
	}
	if (run.Is("drops_crash") && run.Arg(1) == rank) {
		std::_Exit(CRASH_EXIT);
	}
	if (run.Is("crash_start_out") && run.Arg(1) == rank) {
		AppendWorkerOutText(run.request, LargeOutText());
		std::_Exit(CRASH_EXIT);
	}
	// Takes memory in the first run only, and crashes at the rank like crash_start.
	if (run.Is("alloc_crash_start") && run.Arg(1) == rank) {
		if (run.spawn == 0) {
			TakeResidentMemory(ALLOC_MEGABYTES);
		}
		std::_Exit(CRASH_EXIT);
	}
	if (run.Is("burn_hang_start") && run.Arg(1) == rank) {
		BurnCpu(BURN_CPU_MS);
		SleepSeconds(HANG_SECONDS);
		return true;
	}
	if (run.Is("hang_start") && run.Arg(1) == rank) {
		SleepSeconds(run.Arg(2) != 0 ? static_cast<int>(run.Arg(2)) : HANG_SECONDS);
		return true;
	}
	(void)candidate;
	return false;
}

// What goes wrong after a record was written, by mode. Returns true when the run ends there.
bool FakeFaultAfterRecord(FakeRun& run, uint64_t rank) {
	// The first run of this mode ends after two records, with the code that asks for a fresh start.
	if ((run.Is("order14") || run.Is("order14_delay")) && run.spawn == 0 && rank == 1) {
		std::_Exit(WORKER_EXIT_ORDER_MISMATCH);
	}
	if (run.Is("damaged") && run.spawn == 0 && rank == 0) {
		auto bad = FrameStoreRecord(ModuleRecord(ModuleIdOf(9, 0), ProgramHash(9)));
		bad[sizeof(StoreFrame) + 2] ^= 0xFF;
		run.files.Write(bad);
		run.files.Frame(ModuleRecord(ModuleIdOf(9, 1), ProgramHash(9)));
		SleepSeconds(HANG_SECONDS);
		return true;
	}
	if (run.Is("truncated") && run.spawn == 0 && rank == 1) {
		// The supervisor's poll reads the two records before the cut. The cut leaves the header
		// only.
		// The test's sink writes the marker once it holds both records, so the cut cannot come
		// before the supervisor has read them, whatever the load.
		const auto marker = PathFromUtf8(run.request.output_path + TWO_RECORDS_MARKER);
		for (int waited = 0; waited < MARKER_WAIT_STEPS && !fs::exists(marker); waited++) {
			std::this_thread::sleep_for(10ms);
		}
		fs::resize_file(PathFromUtf8(run.request.output_path), GeneratedHeaderBytes({}).size());
		SleepSeconds(HANG_SECONDS);
		return true;
	}
	return false;
}

// Writes the record of one candidate, or garbage in its place for the bad record mode.
void FakeWriteRecord(FakeRun& run, uint64_t rank, const FakeCandidate& candidate) {
	// A module record that is cut short, and one with bytes after its end.
	if (run.Is("malformed_record") && (rank == 1 || rank == 3)) {
		auto record = ModuleRecord(candidate.id, candidate.hash);
		if (rank == 1) {
			record.resize(record.size() / 2);
		} else {
			record.insert(record.end(), {7, 7, 7});
		}
		run.files.Frame(record);
		return;
	}
	if (run.Is("vertex_record") && rank == 1) {
		run.files.Frame(ModuleRecord(candidate.id, candidate.hash, ShaderType::Vertex));
		return;
	}
	if (run.Is("bad_record") && rank == 1) {
		run.files.Frame(Bytes {1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
		return;
	}
	run.files.Frame(ModuleRecord(candidate.id, candidate.hash));
}

// The candidates from the request's start rank, which a real worker walks in order. A candidate of
// a failed program keeps its rank and is not built.
int FakeBuild(FakeRun& run) {
	// This mode crashes at the start line of the rank it is asked to begin at, whatever the request
	// says about failed programs, so that every restart meets a crash.
	if (run.Is("crash_always") || run.Is("crash_always_out")) {
		if (run.Is("crash_always_out")) {
			AppendWorkerOutText(run.request, "worker text before the crash\n");
		}
		const auto rank = run.request.start_rank % run.order.size();
		run.files.Line("start " + std::to_string(run.request.start_rank) + " " +
		               FakeFiles::Hex(run.order[rank].hash));
		std::_Exit(CRASH_EXIT);
	}
	for (uint64_t rank = run.request.start_rank; rank < run.order.size(); rank++) {
		const auto& candidate = run.order[rank];
		if (Contains(run.request.failed_hashes, candidate.hash)) {
			continue;
		}
		if ((run.Is("drops_crash") || run.Is("drops_normal")) && rank == 1 + 0) {
			FakeDrops(run);
		}
		if (run.Is("crash_damaged_start") && run.spawn == 0 && rank == CRASH_DAMAGED_RANK) {
			FakeWriteDamage(run);
		}
		if (run.Is("noisy_crash") && run.spawn == 0 && rank == run.Arg(1)) {
			// More log than one poll of the supervisor reads, with the line that matters last.
			std::string noise;
			for (size_t i = 0; i < NOISE_LINES; i++) {
				noise += "predict " + FakeFiles::Hex(ProgramHash(0)) + "\n";
			}
			run.files.RawLine(noise);
		}
		run.files.Line("start " + std::to_string(rank) + " " + FakeFiles::Hex(candidate.hash));
		if (run.Is("noisy_crash") && run.spawn == 0 && rank == run.Arg(1)) {
			std::_Exit(CRASH_EXIT);
		}
		if (run.Is("crash_damaged_start") && run.spawn == 0 && rank == CRASH_DAMAGED_RANK) {
			std::_Exit(CRASH_EXIT);
		}
		if (FakeFaultAtStart(run, rank, candidate)) {
			return 0;
		}
		FakeWriteRecord(run, rank, candidate);
		if (FakeFaultAfterRecord(run, rank)) {
			return 0;
		}
	}
	run.files.Line("summary exit=0 candidates=" + std::to_string(run.order.size()));
	return 0;
}

// Opens the output after the modes that delay or spoil it have had their say.
void FakeOpenOutput(FakeRun& run) {
	if (run.Is("order14_delay") && run.spawn >= 1) {
		std::this_thread::sleep_for(CREATE_DELAY);
	}
	Bytes header = GeneratedHeaderBytes(run.request);
	if (run.Is("header_error") && run.spawn == 0) {
		header = GeneratedHeaderBytes(run.request, 0x11);
	}
	run.files.OpenOutput(header);
	// The new file stays at its header for a while, which is less than the file it replaces, so a
	// poller that still held the old file would find it cut back. A fast fake could grow the file
	// past the old size between two polls and hide that.
	if (run.Is("order14_delay") && run.spawn >= 1) {
		std::this_thread::sleep_for(HEADER_PAUSE);
	}
	if (run.Is("header_error") && run.spawn == 0) {
		run.files.Frame(ModuleRecord(ModuleIdOf(0, 0), ProgramHash(0)));
		SleepSeconds(HANG_SECONDS);
		std::_Exit(0);
	}
}

// Whether a file starts with these bytes. A file that is missing or shorter does not.
bool FileBeginsWith(const fs::path& path, const Bytes& prefix) {
	std::ifstream in(path, std::ios::binary);
	Bytes         head(prefix.size());
	in.read(reinterpret_cast<char*>(head.data()), static_cast<std::streamsize>(head.size()));
	return in && static_cast<size_t>(in.gcount()) == prefix.size() && head == prefix;
}

// The whole fake. A mode such as `exit:3` ends before it does anything else.
int RunFakeWorker(const std::string& mode_text, const fs::path& request_file) {
	const auto request = ReadWorkerRequestFile(request_file);
	if (!request) {
		return WORKER_EXIT_BAD_REQUEST;
	}
	auto      words = SplitMode(mode_text);
	const int spawn = SpawnIndex(*request);
	FakeRun   run(*request, words, spawn);
	// This mode writes its text file and then runs as a normal worker, which leaves the file for a
	// finish to delete. It comes first so that a Verify answer leaves it too.
	if (run.Is("out_normal")) {
		AppendWorkerOutText(run.request, "worker text of a good run\n");
	}
	// This mode ends fatally when a stale file of a dead process is still in the folder at its
	// start, which shows the cleanup came before the spawn.
	if (run.Is("stale_gone") &&
	    fs::exists(PathFromUtf8(run.request.output_path).parent_path() / STALE_PROBE_NAME)) {
		return STALE_STILL_THERE_EXIT;
	}
	if (run.Is("exit")) {
		return static_cast<int>(run.Arg(1));
	}
	if (run.Is("sleep_silent")) {
		SleepSeconds(HANG_SECONDS);
		return 0;
	}
	// A worker asked to verify leaves a file that begins with its own header as it is, and answers
	// with exit 10 without a progress line, as the real one does. Any other file is generated over.
	if (request->mode == WorkerMode::Verify &&
	    FileBeginsWith(PathFromUtf8(request->generated_path), GeneratedHeaderBytes(*request))) {
		return WORKER_EXIT_UP_TO_DATE;
	}
	// This mode has left a damaged file behind before it crashes at a predict line.
	if (run.Is("crash_damaged_predict") && run.spawn == 0) {
		run.files.OpenOutput(GeneratedHeaderBytes(run.request));
		FakeWriteDamage(run);
	}
	// These two wait or burn before the first predict line, which is what the time to it measures.
	if (run.Is("slow_open")) {
		std::this_thread::sleep_for(std::chrono::milliseconds(run.Arg(1)));
	}
	if (run.Is("burn_normal")) {
		BurnCpu(BURN_CPU_MS);
	}
	FakePredict(run);
	// The first spawn of this mode asks for a fresh start once it has predicted everything.
	if (run.Is("fresh_predictions") && run.spawn == 0) {
		return WORKER_EXIT_ORDER_MISMATCH;
	}
	// This one crashes between its last prediction line and the order line.
	if (run.Is("crash_before_order") && run.spawn == 0) {
		std::_Exit(CRASH_EXIT);
	}
	if (run.Is("exit_after_line")) {
		return static_cast<int>(run.Arg(1));
	}
	const auto order_hash = OrderHashOf(run.order);
	run.files.Line("order " + std::to_string(run.order.size()) + " " + FakeFiles::Hex(order_hash));
	// A resumed run whose ordering is not the one the supervisor saw, by the real worker's guard.
	if (run.request.kept > 0 && run.request.expected_order != 0 &&
	    run.request.expected_order != order_hash) {
		return WORKER_EXIT_ORDER_MISMATCH;
	}
	FakeOpenOutput(run);
	const int code = FakeBuild(run);
	// This mode has written everything, summary line included, and then stays alive for a while
	// with the output file still open, as a worker that is slow to exit would.
	if (run.Is("linger")) {
		std::this_thread::sleep_for(std::chrono::milliseconds(run.Arg(1)));
	}
	return code;
}

// ---- The harness. ----

class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		m_path             = fs::temp_directory_path() /
		         ("kyty_store_supervisor_tests_" + std::to_string(Common::CurrentProcessId()) +
		          "_" + std::to_string(counter++));
		fs::remove_all(m_path);
		fs::create_directories(m_path);
	}
	~TempDirectory() {
		std::error_code ignored;
		fs::remove_all(m_path, ignored);
	}
	TempDirectory(const TempDirectory&)            = delete;
	TempDirectory& operator=(const TempDirectory&) = delete;
	fs::path       operator/(const std::string& name) const { return m_path / name; }

private:
	fs::path m_path;
};

constexpr auto TEST_POLL = 10ms;
// How often the supervisor polls in the tests that must not meet the child half way. The child
// writes its damage and crashes within a few milliseconds, so a poll in that gap is unlikely.
constexpr auto SLOW_POLL  = 1000ms;
constexpr auto END_LIMIT  = 20s;
constexpr auto STOP_LIMIT = 3s;

// What a run of the supervisor over the fake gave.
struct Outcome {
	SupervisorStats          stats;
	std::vector<uint64_t>    ids;
	std::vector<std::string> log;
	bool                     ended = false;
};

// The sink and the log of one run, shared with the supervisor's thread.
struct Collected {
	std::mutex               mutex;
	std::vector<uint64_t>    ids;
	std::vector<std::string> log;
};

SupervisorConfig MakeConfig(const TempDirectory& directory, const std::string& mode,
                            Collected& collected) {
	SupervisorConfig config;
	config.request.store_header       = StoreHeader();
	config.request.host_subgroup_size = 32;
	config.request.eboot_path         = Utf8FromPath(directory / "game.eboot");
	config.request.generator_version  = GENERATOR_VERSION;
	config.output_path                = directory / "title.generated.1.tmp";
	config.generated_path             = directory / "title.generated";
	config.program                    = Common::ExecutablePath();
	config.arguments_before_request   = {"--fake-store-worker", mode};
	config.poll_interval              = TEST_POLL;
	config.startup_timeout            = 20s;
	config.candidate_timeout          = 20s;
	const auto marker                 = fs::absolute(config.output_path).concat(TWO_RECORDS_MARKER);
	config.sink                       = [&collected, marker](DecodedModuleRecord&& record) {
        std::scoped_lock lock(collected.mutex);
        collected.ids.push_back(record.id);
        if (collected.ids.size() == 2) {
            std::ofstream(marker) << "two";
        }
	};
	config.log = [&collected](std::string_view text) {
		std::scoped_lock lock(collected.mutex);
		collected.log.emplace_back(text);
	};
	return config;
}

// Runs a supervisor until it ends, with the settings a test changes.
Outcome Run(const std::string& mode, const std::function<void(SupervisorConfig&)>& tweak = {}) {
	TempDirectory directory;
	Collected     collected;
	auto          config = MakeConfig(directory, mode, collected);
	if (tweak) {
		tweak(config);
	}
	Outcome outcome;
	{
		StoreSupervisor supervisor(std::move(config));
		supervisor.Start();
		outcome.ended = supervisor.WaitForEnd(END_LIMIT);
		supervisor.Stop();
		outcome.stats = supervisor.Stats();
	}
	std::scoped_lock lock(collected.mutex);
	outcome.ids = collected.ids;
	outcome.log = collected.log;
	return outcome;
}

// The ids of the candidates of some programs, in the order a fake worker writes them.
std::vector<uint64_t> IdsOf(const std::vector<size_t>& programs) {
	std::vector<uint64_t> ids;
	for (const auto program: programs) {
		for (size_t index = 0; index < CANDIDATES_EACH; index++) {
			ids.push_back(ModuleIdOf(program, index));
		}
	}
	return ids;
}

std::vector<uint64_t> SortedIds(std::vector<uint64_t> ids) {
	std::ranges::sort(ids);
	return ids;
}

// ---- The tests. ----

// A worker that does everything right is followed to the end: every record once, in its order, one
// spawn and no failure. Every other test differs from this one in what the worker does.
void TestNormalRun() {
	// The eboot is given as a relative path, so that an absolute one in the request is the
	// supervisor's doing.
	const auto run =
	    Run("normal", [](SupervisorConfig& config) { config.request.eboot_path = "game.eboot"; });
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.spawns == 1 && run.stats.restarts == 0 && run.stats.failures.empty());
	CHECK(run.ids == IdsOf({0, 1, 2}));
	CHECK(run.stats.posted == 6 && run.stats.duplicates == 0 && run.stats.bad_records == 0);
	// The request carries absolute paths, this process as the parent, and nothing to resume.
	CHECK(run.stats.requests.size() == 1);
	if (!run.stats.requests.empty()) {
		const auto& request = run.stats.requests[0];
		CHECK(fs::path(PathFromUtf8(request.output_path)).is_absolute());
		CHECK(fs::path(PathFromUtf8(request.progress_path)).is_absolute());
		CHECK(fs::path(PathFromUtf8(request.eboot_path)).is_absolute());
		CHECK(request.parent_pid == Common::CurrentProcessId());
		CHECK(request.kept == 0 && request.start_rank == 0 && request.skip_hashes.empty() &&
		      request.failed_hashes.empty() && request.expected_order == 0);
		CHECK(request.mode == WorkerMode::Generate);
	}
}

// A crash while a program is predicted skips that program. The restart finds no start line yet, so
// it resumes from nothing, and the other programs' records arrive. Without the skip the worker
// would crash at the same program again.
void TestPredictCrashSkipsTheProgram() {
	const auto run = Run("crash_predict:1");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.restarts == 1 && run.stats.spawns == 2);
	CHECK(run.stats.failures.size() == 1);
	if (run.stats.failures.size() == 1) {
		CHECK(run.stats.failures[0].kind == FailureKind::Crash);
		CHECK(run.stats.failures[0].last_line == "predict" &&
		      run.stats.failures[0].hash == ProgramHash(1));
	}
	CHECK(run.stats.requests.size() == 2);
	if (run.stats.requests.size() == 2) {
		const auto& second = run.stats.requests[1];
		CHECK(second.skip_hashes == std::vector<uint64_t> {ProgramHash(1)});
		CHECK(second.failed_hashes.empty());
		CHECK(second.kept == 0 && second.start_rank == 0);
	}
	CHECK(run.ids == IdsOf({0, 2}));
}

// A crash at a start line fails that program and keeps the ranks, so the run goes on at the next
// rank. start_rank is exactly the crashed rank plus one, and what is kept is the file up to the
// last good record.
void TestStartCrashFailsTheProgram() {
	constexpr uint64_t CRASH_RANK = 2;
	const auto         run        = Run("crash_start:2");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.restarts == 1 && run.stats.spawns == 2);
	CHECK(run.stats.requests.size() == 2);
	if (run.stats.requests.size() == 2) {
		const auto& second = run.stats.requests[1];
		CHECK(second.start_rank == CRASH_RANK + 1);
		CHECK(second.failed_hashes == std::vector<uint64_t> {ProgramHash(1)});
		CHECK(second.skip_hashes.empty());
		CHECK(second.expected_order == OrderHashOf(FakeOrder({})));
		uint64_t kept = GeneratedHeaderBytes({}).size();
		for (const auto id: {ModuleIdOf(0, 0), ModuleIdOf(0, 1)}) {
			kept += FrameStoreRecord(ModuleRecord(id, ProgramHash(0))).size();
		}
		CHECK(second.kept == kept);
	}
	CHECK(run.stats.failures.size() == 1 && run.stats.failures[0].rank == CRASH_RANK &&
	      run.stats.failures[0].last_line == "start");
	// Program 1 is left out, and the rest arrives once each.
	CHECK(run.ids == IdsOf({0, 2}));
	CHECK(run.stats.duplicates == 0);
}

// A worker that stays on one line is ended by the candidate timeout and blamed like a crash. With
// no timeout it would hang for the whole wait, which is the baseline.
void TestHangIsKilled() {
	const auto run =
	    Run("hang_start:2", [](SupervisorConfig& config) { config.candidate_timeout = 2s; });
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.failures.size() == 1);
	if (run.stats.failures.size() == 1) {
		CHECK(run.stats.failures[0].kind == FailureKind::Hang);
		CHECK(run.stats.failures[0].hash == ProgramHash(1));
	}
	CHECK(run.stats.requests.size() == 2 &&
	      run.stats.requests[1].failed_hashes == std::vector<uint64_t> {ProgramHash(1)});
	CHECK(run.ids == IdsOf({0, 2}));
}

// A worker with no progress line gets the startup time and then the supervisor ends it for good. A
// worker that ends without one is silent too. Neither can be blamed on a program.
void TestSilentChild() {
	const auto hung =
	    Run("sleep_silent", [](SupervisorConfig& config) { config.startup_timeout = 500ms; });
	CHECK(hung.ended && hung.stats.outcome == SupervisorOutcome::Fatal &&
	      hung.stats.inactive_reason == "child_silent");
	CHECK(hung.stats.spawns == 1);
	const auto crashed = Run("exit:3");
	CHECK(crashed.ended && crashed.stats.outcome == SupervisorOutcome::Fatal &&
	      crashed.stats.inactive_reason == "child_silent");
	CHECK(crashed.stats.spawns == 1);
	// The same exit with a line before it is a crash that can be blamed, and is restarted.
	// Each crash skips the program it was at, until none is left to predict and the worker is
	// silent, which is the end of that run and not the start of it.
	const auto blamed = Run("exit_after_line:3");
	CHECK(blamed.stats.outcome == SupervisorOutcome::Fatal &&
	      blamed.stats.inactive_reason == "child_silent");
	// Three crashes, each at a predict line, and then the silent run that ends it.
	CHECK(blamed.stats.restarts == PROGRAMS && blamed.stats.failures.size() == PROGRAMS + 1);
	CHECK(std::ranges::count_if(blamed.stats.failures, [](const SupervisorFailure& failure) {
		      return failure.last_line == "predict";
	      }) == static_cast<long>(PROGRAMS));
	CHECK(blamed.stats.failures.back().last_line.empty());
}

// Exit 14 starts again from nothing, so the file is new and the records come again. Each arrives
// once. The baseline is the same ids without a restart.
void TestOrderMismatchRestartsFresh() {
	const auto run = Run("order14");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.restarts == 1 && run.stats.failures.size() == 1 &&
	      run.stats.failures[0].kind == FailureKind::OrderMismatch);
	CHECK(run.stats.requests.size() == 2);
	if (run.stats.requests.size() == 2) {
		const auto& second = run.stats.requests[1];
		CHECK(second.kept == 0 && second.start_rank == 0 && second.expected_order == 0);
	}
	// Two records came twice and were not given twice.
	CHECK(run.stats.duplicates == 2 && run.stats.posted == 6);
	CHECK(run.ids == IdsOf({0, 1, 2}));
}

// The programs of drop lines stay out of the next run, but a record that was only too big does not
// take its program with it.
void TestDropLinesFeedFailedHashes() {
	const auto run = Run("drops_crash:4");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.requests.size() == 2);
	if (run.stats.requests.size() == 2) {
		const auto& failed = run.stats.requests[1].failed_hashes;
		CHECK(Contains(failed, DROPPED_HASH));
		CHECK(Contains(failed, PUSH_DROPPED_HASH));
		CHECK(Contains(failed, ProgramHash(2)));
		CHECK(!Contains(failed, TOO_LARGE_HASH));
	}
}

// A fresh restart cuts the output file before the new tailer exists. The worker of the second run
// takes its time to create the file, so a poller that saw the old file would find it cut back
// when the new one is made, and ask for another restart. One restart is the answer.
void TestFreshRestartCutsTheFile() {
	const auto run = Run("order14_delay");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.restarts == 1);
	CHECK(std::ranges::none_of(run.stats.failures, [](const SupervisorFailure& failure) {
		return failure.kind == FailureKind::Truncated;
	}));
	CHECK(run.ids == IdsOf({0, 1, 2}));
}

// A damaged frame followed by more bytes, with the worker alive, ends the worker and starts over
// from nothing, since the file cannot be trusted past its good frames and a rank cannot be told
// from an offset.
void TestDamagedFrameRestartsFresh() {
	const auto run = Run("damaged");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.restarts == 1 && run.stats.failures.size() == 1 &&
	      run.stats.failures[0].kind == FailureKind::BadOutput);
	CHECK(run.stats.requests.size() == 2 && run.stats.requests[1].kept == 0 &&
	      run.stats.requests[1].start_rank == 0);
	CHECK(run.ids == IdsOf({0, 1, 2}));
}

// A file cut back below what was handed over cannot be resumed, so it is a fresh start.
void TestTruncatedRestartsFresh() {
	const auto run = Run("truncated");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.restarts == 1 && run.stats.failures.size() == 1 &&
	      run.stats.failures[0].kind == FailureKind::Truncated);
	CHECK(run.stats.requests.size() == 2 && run.stats.requests[1].kept == 0);
	CHECK(run.ids == IdsOf({0, 1, 2}));
}

// A header that is not the expected one is final: no restart would write another.
void TestHeaderErrorIsFatal() {
	const auto run = Run("header_error");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Fatal &&
	      run.stats.inactive_reason == "bad_output");
	CHECK(run.stats.spawns == 1 && run.stats.restarts == 0);
}

// The worker's own codes that mean the request, the eboot or the disk is wrong end generation, and
// so does exit 10, which only Verify mode answers with. Exit 14 is the baseline: it restarts.
void TestFatalExitCodes() {
	for (const int code: {WORKER_EXIT_UP_TO_DATE, WORKER_EXIT_BAD_REQUEST,
	                      WORKER_EXIT_EBOOT_UNREADABLE, WORKER_EXIT_OUTPUT_FAILED}) {
		const auto run = Run("exit_after_line:" + std::to_string(code));
		CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Fatal);
		CHECK(run.stats.inactive_reason == "worker_exit_" + std::to_string(code));
		CHECK(run.stats.spawns == 1);
	}
}

// A worker that crashes at the next rank every time is started again five times and no more.
void TestRestartCap() {
	const auto run = Run("crash_always");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::RestartsCapped &&
	      run.stats.inactive_reason == "restarts_capped");
	CHECK(run.stats.restarts == GENERATION_MAX_RESTARTS &&
	      run.stats.spawns == GENERATION_MAX_RESTARTS + 1);
	// A lower cap is obeyed too.
	const auto lower =
	    Run("crash_always", [](SupervisorConfig& config) { config.max_restarts = 2; });
	CHECK(lower.stats.restarts == 2 && lower.stats.spawns == 3);
}

// Whether a process is still there.
bool ProcessExists(uint32_t pid) {
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	HANDLE handle = OpenProcess(SYNCHRONIZE, FALSE, pid);
	if (handle == nullptr) {
		return false;
	}
	const bool running = WaitForSingleObject(handle, 0) == WAIT_TIMEOUT;
	CloseHandle(handle);
	return running;
#else
	return kill(static_cast<pid_t>(pid), 0) == 0;
#endif
}

// A stop with the worker alive ends the worker and returns at once, not when the worker would have
// finished on its own, which is far later. It is the supervisor's own thread that ends the child.
void TestStopWithLiveChild() {
	TempDirectory   directory;
	Collected       collected;
	auto            config = MakeConfig(directory, "hang_start:0:10", collected);
	StoreSupervisor supervisor(std::move(config));
	supervisor.Start();
	const auto deadline = std::chrono::steady_clock::now() + END_LIMIT;
	while (supervisor.Stats().spawns == 0 && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::sleep_for(5ms);
	}
	// Let the child reach its hang, so that a stop meets a worker that is running.
	std::this_thread::sleep_for(300ms);
	const auto pid   = supervisor.Stats().last_pid;
	const auto begin = std::chrono::steady_clock::now();
	supervisor.Stop();
	const auto took = std::chrono::steady_clock::now() - begin;
	CHECK(pid != 0 && took < STOP_LIMIT);
	CHECK(supervisor.Stats().outcome == SupervisorOutcome::Stopped);
	// The worker is gone, which does not depend on who reaped it.
	const auto gone_deadline = std::chrono::steady_clock::now() + 5s;
	while (ProcessExists(pid) && std::chrono::steady_clock::now() < gone_deadline) {
		std::this_thread::sleep_for(20ms);
	}
	CHECK(!ProcessExists(pid));
}

// A frame that is not a compute module record is counted and skipped, and the records after it
// still arrive.
void TestBadRecordIsSkipped() {
	const auto run = Run("bad_record");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.bad_records == 1 && run.stats.frames_delivered == 6);
	CHECK(run.ids.size() == 5);
}

// A worker that crashed and left a bad file starts over fresh, and the crash is still blamed on the
// program of the last line, or the fresh run would crash at the same one. The poll is slow so that
// the supervisor finds the worker already dead and does not end it for the bad file itself. A
// predict line puts the program in skip_hashes and a start line in failed_hashes.
void TestCrashWithDamagedFileIsBlamed() {
	const auto predict = Run("crash_damaged_predict",
	                         [](SupervisorConfig& config) { config.poll_interval = SLOW_POLL; });
	CHECK(predict.ended && predict.stats.outcome == SupervisorOutcome::Finished);
	CHECK(predict.stats.restarts == 1 && predict.stats.requests.size() == 2);
	CHECK(!predict.stats.failures.empty() &&
	      predict.stats.failures[0].kind == FailureKind::BadOutput &&
	      predict.stats.failures[0].exit_code == CRASH_EXIT);
	if (predict.stats.requests.size() == 2) {
		const auto& second = predict.stats.requests[1];
		CHECK(second.skip_hashes == std::vector<uint64_t> {ProgramHash(1)});
		CHECK(second.failed_hashes.empty() && second.kept == 0 && second.start_rank == 0);
	}
	CHECK(predict.ids == IdsOf({0, 2}));

	const auto start = Run("crash_damaged_start",
	                       [](SupervisorConfig& config) { config.poll_interval = SLOW_POLL; });
	CHECK(start.ended && start.stats.outcome == SupervisorOutcome::Finished);
	CHECK(start.stats.restarts == 1 && start.stats.requests.size() == 2);
	CHECK(!start.stats.failures.empty() && start.stats.failures[0].kind == FailureKind::BadOutput &&
	      start.stats.failures[0].exit_code == CRASH_EXIT);
	if (start.stats.requests.size() == 2) {
		const auto& second = start.stats.requests[1];
		CHECK(second.failed_hashes == std::vector<uint64_t> {ProgramHash(1)});
		CHECK(second.skip_hashes.empty() && second.kept == 0 && second.start_rank == 0);
	}
	CHECK(start.ids == IdsOf({0, 2}));
}

// A record of another stage than compute is not a module the store generation makes, so it is
// counted and skipped like garbage is. The records after it still arrive.
void TestOtherStageIsSkipped() {
	const auto run = Run("vertex_record");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.bad_records == 1 && run.ids.size() == 5);
	CHECK(std::ranges::find(run.ids, ModuleIdOf(0, 1)) == run.ids.end());
}

// A module record that is cut short, or has bytes after its end, is no record. Each is counted and
// skipped, and the records around them arrive. The stage and type checks alone would let a cut
// record of a compute module through.
void TestMalformedRecordsAreSkipped() {
	const auto run = Run("malformed_record");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.bad_records == 2 && run.ids.size() == 4);
	CHECK(std::ranges::find(run.ids, ModuleIdOf(0, 1)) == run.ids.end());
	CHECK(std::ranges::find(run.ids, ModuleIdOf(1, 1)) == run.ids.end());
}

// The blame is judged on the whole log. A worker that writes more log than one poll reads and
// crashes at the line it ends with must be blamed on that line, and not on the last one that was
// read. Failed hashes hold its program, and the skip list holds nothing.
void TestBlameReadsTheWholeLog() {
	const auto run = Run("noisy_crash:4");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.requests.size() == 2);
	if (run.stats.requests.size() == 2) {
		CHECK(run.stats.requests[1].failed_hashes == std::vector<uint64_t> {ProgramHash(2)});
		CHECK(run.stats.requests[1].skip_hashes.empty());
		CHECK(run.stats.requests[1].start_rank == 5);
	}
}

// A refused priority change is logged once and the run goes on, which is its baseline.
void TestPriorityRefusalIsLogged() {
	const auto run = Run("normal", [](SupervisorConfig& config) {
		config.priority_step = [] {
			return std::optional<Common::PriorityFailure> {Common::PriorityFailure {"injected", 7}};
		};
	});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	const auto lines = std::ranges::count_if(run.log, [](const std::string& text) {
		return text.find("injected failed with 7") != std::string::npos;
	});
	CHECK(lines == 1);
	const auto plain = Run("normal");
	CHECK(plain.log.empty());
}

// When the file cannot be cut a fresh restart ends with the reason. Here the output path is a
// directory with something in it, which neither a remove nor a resize can turn into an empty file.
void TestOutputResetFailure() {
	const auto run = Run("exit_after_line:14", [](SupervisorConfig& config) {
		fs::create_directories(config.output_path);
		std::ofstream(config.output_path / "inside.txt") << "x";
	});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Fatal &&
	      run.stats.inactive_reason == "output_reset_failed");
}

// The gate gives the first reason that applies, in a fixed order, and nothing when none does. Each
// row changes one input of the passing one.
StoreGenerationInputs GoodGateInputs(const TempDirectory& directory) {
	StoreGenerationInputs ok;
	ok.background_warm_up_enabled = true;
	ok.shader_store_path_empty    = false;
	ok.store_header_empty         = false;
	ok.eboot_path                 = directory / "game.eboot";
	ok.executable_path            = Common::ExecutablePath();
	return ok;
}

// Each row changes one input of the inputs that pass, and gives its own reason.
void TestGateSingleReasons() {
	TempDirectory directory;
	{
		std::ofstream(directory / "game.eboot") << "x";
	}
	const auto ok = GoodGateInputs(directory);
	CHECK(!StoreGenerationAvailability(ok).has_value());

	auto no_warm                       = ok;
	no_warm.background_warm_up_enabled = false;
	CHECK(StoreGenerationAvailability(no_warm) == "needs_background_warm_up");
	auto no_store                    = ok;
	no_store.shader_store_path_empty = true;
	CHECK(StoreGenerationAvailability(no_store) == "no_shader_store");
	auto no_stamp               = ok;
	no_stamp.store_header_empty = true;
	CHECK(StoreGenerationAvailability(no_stamp) == "no_build_stamp");
	auto no_eboot       = ok;
	no_eboot.eboot_path = directory / "absent.eboot";
	CHECK(StoreGenerationAvailability(no_eboot) == "no_eboot");
	no_eboot.eboot_path.clear();
	CHECK(StoreGenerationAvailability(no_eboot) == "no_eboot");
	auto no_exe            = ok;
	no_exe.executable_path = directory / "absent.exe";
	CHECK(StoreGenerationAvailability(no_exe) == "no_executable");
	no_exe.executable_path.clear();
	CHECK(StoreGenerationAvailability(no_exe) == "no_executable");
}

// Two things wrong at once give the earlier reason, whichever way a later check was moved.
void TestGatePrecedence() {
	TempDirectory directory;
	{
		std::ofstream(directory / "game.eboot") << "x";
	}
	const auto ok                      = GoodGateInputs(directory);
	auto       stamp_and_eboot         = ok;
	stamp_and_eboot.store_header_empty = true;
	stamp_and_eboot.eboot_path         = directory / "absent.eboot";
	CHECK(StoreGenerationAvailability(stamp_and_eboot) == "no_build_stamp");
	auto eboot_and_exe            = ok;
	eboot_and_exe.eboot_path      = directory / "absent.eboot";
	eboot_and_exe.executable_path = directory / "absent.exe";
	CHECK(StoreGenerationAvailability(eboot_and_exe) == "no_eboot");
	auto warm_and_store                       = ok;
	warm_and_store.background_warm_up_enabled = false;
	warm_and_store.shader_store_path_empty    = true;
	CHECK(StoreGenerationAvailability(warm_and_store) == "needs_background_warm_up");
	auto store_and_stamp                    = ok;
	store_and_stamp.shader_store_path_empty = true;
	store_and_stamp.store_header_empty      = true;
	CHECK(StoreGenerationAvailability(store_and_stamp) == "no_shader_store");
	// With everything wrong the first reason is given, and mending it gives the next.
	StoreGenerationInputs nothing;
	CHECK(StoreGenerationAvailability(nothing) == "needs_background_warm_up");
	nothing.background_warm_up_enabled = true;
	CHECK(StoreGenerationAvailability(nothing) == "no_shader_store");
}

// ---- What a run leaves in the statistics. ----

// The times of a run. The baseline is the normal worker, and the slow one differs only in the time
// before its first predict line.
void TestTimingStats() {
	const auto normal = Run("normal");
	CHECK(normal.ended && normal.stats.outcome == SupervisorOutcome::Finished);
	const auto& stats = normal.stats;
	CHECK(stats.finished_ms > 0.0 && stats.finished_ms >= stats.spawn_ms);
	// A finished run ends where it finishes.
	CHECK(stats.ended_ms == stats.finished_ms);
	CHECK(stats.child_open_ms.has_value());
	CHECK(stats.first_record_ms.has_value() && stats.last_record_ms.has_value());
	if (stats.child_open_ms && stats.first_record_ms && stats.last_record_ms) {
		CHECK(*stats.first_record_ms <= *stats.last_record_ms);
		CHECK(*stats.last_record_ms <= stats.finished_ms);
		// The first record comes after the first predict line, and neither is after the end.
		CHECK(stats.spawn_ms + *stats.child_open_ms <= stats.finished_ms);
	}
	const auto slow = Run("slow_open:" + std::to_string(SLOW_OPEN_MS));
	CHECK(slow.ended && slow.stats.outcome == SupervisorOutcome::Finished);
	// Compared with the normal run, so a slow machine that stretches both cannot fail it.
	CHECK(slow.stats.child_open_ms.has_value() && stats.child_open_ms.has_value() &&
	      *slow.stats.child_open_ms >= *stats.child_open_ms + (SLOW_OPEN_MS - OPEN_MARGIN_MS));
	// The first record cannot come before the first predict line, so it shows the delay as well.
	CHECK(slow.stats.first_record_ms.has_value() &&
	      *slow.stats.first_record_ms >= SLOW_OPEN_MS * 0.8);
}

// finished_ms is zero while the supervisor runs and is set by the end, and nothing is known of the
// child's open time before a predict line arrives.
void TestFinishedMsOnlyAfterTheEnd() {
	TempDirectory   directory;
	Collected       collected;
	auto            config = MakeConfig(directory, "sleep_silent", collected);
	StoreSupervisor supervisor(std::move(config));
	// The report's rows of worker compiles say whether the worker was alive, which is the
	// supervisor's to say. It is not before the start, is while the worker runs, and is not after
	// the end.
	Common::FirstRunReport::SetGeneratorChildAlive(false);
	supervisor.Start();
	std::this_thread::sleep_for(300ms);
	CHECK(Common::FirstRunReport::GeneratorChildAlive());
	const auto running = supervisor.Stats();
	CHECK(running.outcome == SupervisorOutcome::Running);
	CHECK(running.finished_ms == 0.0 && running.ended_ms == 0.0 &&
	      !running.child_open_ms.has_value());
	supervisor.Stop();
	CHECK(!Common::FirstRunReport::GeneratorChildAlive());
	const auto stopped = supervisor.Stats();
	CHECK(stopped.outcome == SupervisorOutcome::Stopped);
	// The stop came after the 300 ms above, so the end cannot be earlier than that. The run did not
	// finish, so finished_ms stays zero, which tells it from ended_ms.
	CHECK(stopped.ended_ms >= 250.0);
	CHECK(stopped.finished_ms == 0.0);
	// A run that ends for good has an end and no finish.
	Common::FirstRunReport::SetGeneratorChildAlive(true);
	const auto fatal = Run("exit:3");
	CHECK(fatal.stats.outcome == SupervisorOutcome::Fatal);
	CHECK(!Common::FirstRunReport::GeneratorChildAlive());
	CHECK(fatal.stats.ended_ms > 0.0 && fatal.stats.finished_ms == 0.0);
}

// The longest record is the one with the slow sink. The baseline is the same run with the normal
// sink.
void TestDecodeMsMax() {
	const auto fast = Run("normal");
	CHECK(fast.stats.decode_ms_max < SLOW_SINK_MIN_MS);
	const auto slow = Run("normal", [](SupervisorConfig& config) {
		config.sink = [](DecodedModuleRecord&& record) {
			if (record.id == ModuleIdOf(1, 0)) {
				std::this_thread::sleep_for(SLOW_SINK_DELAY);
			}
		};
	});
	CHECK(slow.stats.outcome == SupervisorOutcome::Finished);
	CHECK(slow.stats.decode_ms_max >= SLOW_SINK_MIN_MS);
}

// The programs of the predict lines, of the drops, and of the lists that the next run leaves out.
// Each differs from the normal run in one thing.
void TestHashLists() {
	const auto normal = Run("normal");
	CHECK(normal.stats.predicted_hashes ==
	      (std::vector<uint64_t> {ProgramHash(0), ProgramHash(1), ProgramHash(2)}));
	CHECK(normal.stats.dropped.empty() && normal.stats.failed_hashes.empty() &&
	      normal.stats.skip_hashes.empty());

	// A crash at a predict line skips its program. The predicted list holds it once, though the
	// two spawns both looked at program 0.
	const auto skipped = Run("crash_predict:1");
	CHECK(skipped.stats.skip_hashes == std::vector<uint64_t> {ProgramHash(1)});
	CHECK(skipped.stats.failed_hashes.empty());
	CHECK(skipped.stats.predicted_hashes ==
	      (std::vector<uint64_t> {ProgramHash(0), ProgramHash(1), ProgramHash(2)}));

	// A crash at a start line fails its program and skips none.
	const auto failed = Run("crash_start:2");
	CHECK(failed.stats.failed_hashes == std::vector<uint64_t> {ProgramHash(1)});
	CHECK(failed.stats.skip_hashes.empty());
}

// The prediction rows reach the statistics, with the outcome and the count of each program. The
// baseline is the run's predicted_hashes, which has the same programs and none of the rest, so a
// row cannot be told from a bare predict line by the hash alone.
void TestPredictionRows() {
	const auto                              normal = Run("normal");
	const std::vector<SupervisorPrediction> expected {
	    {ProgramHash(0), PredictionOutcome::Predicted, 1},
	    {ProgramHash(1), PredictionOutcome::Calls, 0},
	    {ProgramHash(2), PredictionOutcome::Problems, 0}};
	CHECK(normal.stats.outcome == SupervisorOutcome::Finished);
	CHECK(normal.stats.predictions.size() == expected.size());
	for (size_t i = 0; i < expected.size() && i < normal.stats.predictions.size(); i++) {
		const auto& row = normal.stats.predictions[i];
		CHECK(row.hash == expected[i].hash && row.outcome == expected[i].outcome &&
		      row.candidates == expected[i].candidates);
	}

	// A resume after a crash at a start line predicts every program again in the second spawn, and
	// the latest row of program 0 is that spawn's, with one more candidate. The blame is unchanged.
	const auto resumed = Run("crash_start:2");
	CHECK(resumed.stats.spawns == 2 && resumed.stats.predictions.size() == 3);
	if (resumed.stats.predictions.size() == 3) {
		CHECK(resumed.stats.predictions[0].candidates == 2);
	}
	CHECK(resumed.stats.failed_hashes == std::vector<uint64_t> {ProgramHash(1)});
	CHECK(resumed.stats.skip_hashes.empty());

	// A crash while predicting program 1 gives it no row, since the line comes after, and it is
	// blamed all the same: skipped, with `predict` as the last line.
	const auto crashed = Run("crash_predict:1");
	CHECK(crashed.stats.skip_hashes == std::vector<uint64_t> {ProgramHash(1)});
	CHECK(crashed.stats.failures.size() == 1 && crashed.stats.failures[0].last_line == "predict");
	bool has_one = false;
	for (const auto& row: crashed.stats.predictions) {
		has_one = has_one || row.hash == ProgramHash(1);
	}
	CHECK(!has_one && crashed.stats.predictions.size() == 2);
}

// A crash after the last `prediction` line and before `order` is in no program. It blames nothing,
// where the last `predict` line used to be the last line and blamed its program. The baseline is
// the crash at a predict line above, which still blames.
void TestCrashBeforeOrderBlamesNothing() {
	const auto run = Run("crash_before_order");
	CHECK(run.stats.outcome == SupervisorOutcome::Finished && run.stats.restarts == 1);
	CHECK(run.stats.failures.size() == 1);
	if (run.stats.failures.size() == 1) {
		CHECK(run.stats.failures[0].kind == FailureKind::Crash);
		CHECK(run.stats.failures[0].last_line == "prediction" && run.stats.failures[0].hash == 0);
	}
	CHECK(run.stats.skip_hashes.empty() && run.stats.failed_hashes.empty());
	CHECK(run.stats.requests.size() == 2 && run.stats.requests[1].skip_hashes.empty() &&
	      run.stats.requests[1].failed_hashes.empty());
	CHECK(run.ids == Run("normal").ids);
}

// A fresh restart keeps the rows of the first spawn that the second one does not repeat, and the
// latest row of a program that it does repeat wins. The baseline is the first spawn alone, which
// the failure record's spawn number shows was a fresh restart's cause.
void TestPredictionRowsSurviveAFreshRestart() {
	const auto run = Run("fresh_predictions");
	CHECK(run.stats.outcome == SupervisorOutcome::Finished && run.stats.spawns == 2);
	CHECK(run.stats.failures.size() == 1 &&
	      run.stats.failures[0].kind == FailureKind::OrderMismatch);
	CHECK(run.stats.requests.size() == 2 && run.stats.requests[1].kept == 0);
	CHECK(run.stats.predictions.size() == 3);
	if (run.stats.predictions.size() == 3) {
		// The second spawn's row, in the place of the first.
		CHECK(run.stats.predictions[0].hash == ProgramHash(0) &&
		      run.stats.predictions[0].outcome == PredictionOutcome::Predicted &&
		      run.stats.predictions[0].candidates == 2);
		// Rows of the first spawn, which the fresh run did not write again.
		CHECK(run.stats.predictions[1].hash == ProgramHash(1) &&
		      run.stats.predictions[1].outcome == PredictionOutcome::Calls);
		CHECK(run.stats.predictions[2].hash == ProgramHash(2) &&
		      run.stats.predictions[2].outcome == PredictionOutcome::Problems);
	}
}

// The drops of a run that finishes are listed too. They are in no restart's request, so the list of
// failed programs stays empty, which tells the list of drops from it.
void TestDropsAreListedForAFinishedRun() {
	const auto run = Run("drops_normal");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.failed_hashes.empty());
	CHECK(run.stats.dropped.size() == 3);
	if (run.stats.dropped.size() == 3) {
		CHECK(run.stats.dropped[0].hash == DROPPED_HASH &&
		      run.stats.dropped[0].reason == ProgressDropReason::ResourceCount);
		CHECK(run.stats.dropped[1].hash == TOO_LARGE_HASH &&
		      run.stats.dropped[1].reason == ProgressDropReason::FrameTooLarge);
		CHECK(run.stats.dropped[2].hash == PUSH_DROPPED_HASH &&
		      run.stats.dropped[2].reason == ProgressDropReason::PushStart);
		CHECK(run.stats.dropped[0].rank == 1);
	}
	// A run that crashes after drops lists them once, and not again for the restart's request.
	const auto crashed = Run("drops_crash:4");
	CHECK(crashed.stats.dropped.size() == 3);
}

// Each spawn that ended has one usage entry. The burning worker is credited with its processor
// time, and the normal one is not, which is the baseline that only differs in that.
void TestSpawnUsageCpu() {
	const auto normal = Run("normal");
	CHECK(normal.stats.spawn_usage.size() == normal.stats.spawns);
	CHECK(!normal.stats.spawn_usage.empty() && normal.stats.spawn_usage[0].spawn == 1);
	const auto burned = Run("burn_normal");
	CHECK(burned.ended && burned.stats.outcome == SupervisorOutcome::Finished);
	CHECK(burned.stats.spawn_usage.size() == 1);
	if (normal.stats.spawn_usage.size() == 1 && burned.stats.spawn_usage.size() == 1) {
		CHECK(normal.stats.spawn_usage[0].cpu_ms < BASELINE_MAX_CPU_MS);
		CHECK(burned.stats.spawn_usage[0].cpu_ms >= BURN_MIN_REPORTED_MS);
	}
	// One restart means two spawns and two entries.
	const auto crashed = Run("crash_start:2");
	CHECK(crashed.stats.spawns == 2 && crashed.stats.spawn_usage.size() == 2);
	// The entries and the failure carry the number of their spawn, so they can be joined.
	if (crashed.stats.spawn_usage.size() == 2 && crashed.stats.failures.size() == 1) {
		CHECK(crashed.stats.spawn_usage[0].spawn == 1 && crashed.stats.spawn_usage[1].spawn == 2);
		CHECK(crashed.stats.failures[0].spawn == 1);
	}
}

// A peak of a better source replaces the held one, whatever the sizes, and the same source only
// with a larger value. A commit figure is not comparable with a working set.
void TestPeakSourceRank() {
	using Common::PeakMemory;
	using Common::PeakMemorySource;
	constexpr uint64_t SMALL = 10 * BYTES_PER_MEGABYTE;
	constexpr uint64_t LARGE = 100 * BYTES_PER_MEGABYTE;
	const PeakMemory   commit_large {LARGE, PeakMemorySource::Commit};
	const PeakMemory   commit_small {SMALL, PeakMemorySource::Commit};
	const PeakMemory   set_small {SMALL, PeakMemorySource::WorkingSet};
	const PeakMemory   set_large {LARGE, PeakMemorySource::WorkingSet};
	const PeakMemory   sampled_small {SMALL, PeakMemorySource::WorkingSetSampled};
	// A commit peak followed by a working set peak gives the working set one, though it is smaller.
	CHECK(PeakReplaces(commit_large, set_small));
	// The other way round keeps the working set one, though the commit is larger.
	CHECK(!PeakReplaces(set_small, commit_large));
	CHECK(PeakReplaces(sampled_small, set_small) && !PeakReplaces(set_small, sampled_small));
	CHECK(PeakReplaces(commit_large, sampled_small) && !PeakReplaces(sampled_small, commit_large));
	// The same source takes the larger value only.
	CHECK(PeakReplaces(set_small, set_large) && !PeakReplaces(set_large, set_small));
	CHECK(PeakReplaces(commit_small, commit_large) && !PeakReplaces(commit_large, commit_small));
	CHECK(!PeakReplaces(set_large, set_large));
}

// A worker that is ended for a hang has its processor time in the failure. The baseline hangs the
// same way without burning.
void TestFailureCpuAtATimeout() {
	const auto set_timeout = [](SupervisorConfig& config) { config.candidate_timeout = 2s; };
	const auto idle        = Run("hang_start:2", set_timeout);
	const auto burned      = Run("burn_hang_start:2", set_timeout);
	CHECK(idle.stats.failures.size() == 1 && burned.stats.failures.size() == 1);
	if (idle.stats.failures.size() == 1 && burned.stats.failures.size() == 1) {
		const auto& idle_failure   = idle.stats.failures[0];
		const auto& burned_failure = burned.stats.failures[0];
		CHECK(idle_failure.kind == FailureKind::Hang && burned_failure.kind == FailureKind::Hang);
		CHECK(idle_failure.cpu_ms.has_value() && *idle_failure.cpu_ms < BASELINE_MAX_CPU_MS);
		CHECK(burned_failure.cpu_ms.has_value() && *burned_failure.cpu_ms >= BURN_MIN_REPORTED_MS);
	}
}

// The text file's tail is kept by a failure, bounded, from its end. The baseline crashes the same
// way and has no text file.
void TestFailureOutTail() {
	const auto plain = Run("crash_start:2");
	CHECK(plain.stats.failures.size() == 1 && plain.stats.failures[0].out_tail.empty());
	const auto with_text = Run("crash_start_out:2");
	CHECK(with_text.stats.failures.size() == 1);
	if (with_text.stats.failures.size() == 1) {
		const auto& tail = with_text.stats.failures[0].out_tail;
		CHECK(tail.size() == GENERATION_OUT_TAIL_BYTES);
		CHECK(tail.ends_with(OUT_TAIL_MARK));
		CHECK(tail.find(OUT_HEAD_MARK) == std::string::npos);
	}
}

// The largest peak of the spawns is kept, and not the last one. The first run takes 64 MB and the
// restart takes none. Linux samples a live child, which a worker this short may not give, so only
// the other systems are held to a figure.
void TestPeakMemoryIsTheLargest() {
	const auto run = Run("alloc_crash_start:2");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.spawn_usage.size() == 2);
#ifndef __linux__
	if (run.stats.spawn_usage.size() == 2) {
		const auto& first  = run.stats.spawn_usage[0];
		const auto& second = run.stats.spawn_usage[1];
		CHECK(first.peak.has_value() && second.peak.has_value());
		if (first.peak && second.peak && run.stats.peak_memory) {
			CHECK(first.peak->bytes >= ALLOC_MIN_PEAK_BYTES);
			CHECK(second.peak->bytes < ALLOC_MIN_PEAK_BYTES);
			CHECK(run.stats.peak_memory->bytes == first.peak->bytes);
			CHECK(run.stats.peak_memory->source == first.peak->source);
		} else {
			CHECK(false);
		}
	}
#endif
	// A run with no spawn that took memory has a peak below the threshold, so the figure above is
	// the allocation and not the process.
	const auto normal = Run("normal");
#ifndef __linux__
	CHECK(normal.stats.peak_memory.has_value() &&
	      normal.stats.peak_memory->bytes < ALLOC_MIN_PEAK_BYTES);
#endif
	CHECK(normal.stats.spawn_usage.size() == 1);
}

// ---- The generation block of the first-run report. ----

// The worker's summary line, which the supervisor keeps from the last run that wrote one. The
// baseline is a run whose worker wrote none.
void TestWorkerSummaryFields() {
	const auto run = Run("normal");
	CHECK(run.stats.worker_summary ==
	      (std::vector<std::pair<std::string, std::string>> {{"exit", "0"}, {"candidates", "6"}}));
	const auto silent = Run("exit:3");
	CHECK(silent.stats.worker_summary.empty());
	// A crashed first run wrote none, and the run that finished wrote one, which is what stays.
	const auto restarted = Run("crash_start:2");
	CHECK(restarted.stats.outcome == SupervisorOutcome::Finished);
	CHECK(!restarted.stats.worker_summary.empty());
}

// Each enumerated name of the report, which compare scripts depend on.
void TestReportNames() {
	CHECK(std::string(OutcomeName(SupervisorOutcome::Running)) == "running");
	CHECK(std::string(OutcomeName(SupervisorOutcome::Finished)) == "finished");
	CHECK(std::string(OutcomeName(SupervisorOutcome::Fatal)) == "fatal");
	CHECK(std::string(OutcomeName(SupervisorOutcome::RestartsCapped)) == "restarts_capped");
	CHECK(std::string(OutcomeName(SupervisorOutcome::Stopped)) == "stopped");
	CHECK(std::string(FailureKindName(FailureKind::Crash)) == "crash");
	CHECK(std::string(FailureKindName(FailureKind::Hang)) == "hang");
	CHECK(std::string(FailureKindName(FailureKind::OrderMismatch)) == "order_mismatch");
	CHECK(std::string(FailureKindName(FailureKind::BadOutput)) == "bad_output");
	CHECK(std::string(FailureKindName(FailureKind::Truncated)) == "truncated");
	CHECK(std::string(DropReasonName(ProgressDropReason::PushStart)) == "push_start");
	CHECK(std::string(DropReasonName(ProgressDropReason::ResourceCount)) == "resource_count");
	CHECK(std::string(DropReasonName(ProgressDropReason::FrameTooLarge)) == "frame_too_large");
	CHECK(std::string(PeakSourceName(Common::PeakMemorySource::WorkingSet)) == "working_set");
	CHECK(std::string(PeakSourceName(Common::PeakMemorySource::WorkingSetSampled)) ==
	      "working_set_sampled");
	CHECK(std::string(PeakSourceName(Common::PeakMemorySource::Commit)) == "commit");
}

// A finished run gives an active block with both times, and every ending that is not a finish gives
// ended_ms without finished_ms. Each differs from the finished run in the outcome only.
void TestReportFromFinishedAndEndedRuns() {
	const auto finished = Run("normal");
	const auto block    = MakeSupervisorReport(finished.stats);
	CHECK(block.active && block.inactive_reason.empty() && block.outcome == "finished");
	CHECK(block.finished_ms.has_value() && block.ended_ms.has_value() &&
	      *block.finished_ms == *block.ended_ms && *block.finished_ms > 0.0);
	// The finish on the report's clock is carried over as it is, and is missing when there was
	// none.
	CHECK(block.finished_ns.has_value() && block.finished_ns == finished.stats.finished_ns);
	CHECK(block.spawns == 1 && block.restarts == 0 && block.failures.empty());
	CHECK(block.posted == finished.stats.posted && block.posted == 6);
	CHECK(block.first_record_ms.has_value() && block.last_record_ms.has_value() &&
	      block.child_open_ms.has_value());
	CHECK(block.predicted_programs == PROGRAMS);
	CHECK(block.worker == finished.stats.worker_summary && !block.worker.empty());
	CHECK(block.spawn_usage.size() == 1 && block.spawn_usage[0].spawn == 1);

	// Fatal: no restart can help. The block is inactive with the reason, and the run has an end.
	const auto fatal       = Run("exit:3");
	const auto fatal_block = MakeSupervisorReport(fatal.stats);
	CHECK(!fatal_block.active && fatal_block.inactive_reason == "child_silent" &&
	      fatal_block.outcome == "fatal");
	CHECK(!fatal_block.finished_ms.has_value() && fatal_block.ended_ms.has_value() &&
	      !fatal_block.finished_ns.has_value());

	// Capped: the cap is reached and what was delivered is kept.
	const auto capped =
	    Run("crash_always", [](SupervisorConfig& config) { config.max_restarts = 2; });
	const auto capped_block = MakeSupervisorReport(capped.stats);
	CHECK(capped_block.outcome == "restarts_capped" && !capped_block.active &&
	      capped_block.inactive_reason == "restarts_capped");
	CHECK(!capped_block.finished_ms.has_value() && capped_block.ended_ms.has_value());
	CHECK(capped_block.restarts == 2 && capped_block.failures.size() == 3);

	// Running and then stopped: nothing is over while it runs, and a stop is an end and no finish.
	TempDirectory   directory;
	Collected       collected;
	StoreSupervisor supervisor(MakeConfig(directory, "sleep_silent", collected));
	supervisor.Start();
	std::this_thread::sleep_for(200ms);
	const auto running = MakeSupervisorReport(supervisor.Stats());
	CHECK(running.active && running.outcome == "running");
	CHECK(!running.finished_ms.has_value() && !running.ended_ms.has_value());
	supervisor.Stop();
	const auto stopped = MakeSupervisorReport(supervisor.Stats());
	CHECK(stopped.active && stopped.outcome == "stopped");
	CHECK(!stopped.finished_ms.has_value() && stopped.ended_ms.has_value() &&
	      !stopped.finished_ns.has_value());
}

// The rows of a failure and of the drops, with the names the report uses.
void TestReportRows() {
	const auto crashed = Run("crash_start_out:2");
	const auto block   = MakeSupervisorReport(crashed.stats);
	CHECK(block.failures.size() == 1);
	if (block.failures.size() == 1) {
		const auto& failure = block.failures[0];
		CHECK(failure.kind == "crash" && failure.last_line == "start" && failure.rank == 2);
		CHECK(failure.hash == ProgramHash(1) && failure.spawn == 1);
		CHECK(failure.cpu_ms.has_value());
		CHECK(failure.out_tail == crashed.stats.failures[0].out_tail && !failure.out_tail.empty());
	}
	CHECK(block.failed_hashes == std::vector<uint64_t> {ProgramHash(1)} &&
	      block.skip_hashes.empty());
	CHECK(block.spawn_usage.size() == 2 && block.spawn_usage[1].spawn == 2);

	const auto hung =
	    Run("hang_start:2", [](SupervisorConfig& config) { config.candidate_timeout = 2s; });
	const auto hung_block = MakeSupervisorReport(hung.stats);
	CHECK(hung_block.failures.size() == 1 && hung_block.failures[0].kind == "hang");

	const auto dropped       = Run("drops_normal");
	const auto dropped_block = MakeSupervisorReport(dropped.stats);
	CHECK(dropped_block.dropped.size() == 3);
	if (dropped_block.dropped.size() == 3) {
		CHECK(dropped_block.dropped[0].reason == "resource_count" &&
		      dropped_block.dropped[0].hash == DROPPED_HASH && dropped_block.dropped[0].rank == 1);
		CHECK(dropped_block.dropped[1].reason == "frame_too_large");
		CHECK(dropped_block.dropped[2].reason == "push_start");
	}
	// The peak comes with the name of its source, where the system measured one.
#ifndef __linux__
	CHECK(block.peak_bytes.has_value() && !block.peak_source.empty());
	CHECK(block.peak_source == PeakSourceName(crashed.stats.peak_memory->source));
	CHECK(!block.spawn_usage.empty() && block.spawn_usage[0].peak_bytes.has_value() &&
	      !block.spawn_usage[0].peak_source.empty());
#endif
}

// Every counter has its own place in the block: each is given a different number, so two that were
// swapped cannot both match.
void TestDrainFiguresAreMapped() {
	GeneratedDrainCounters drain;
	drain.drain_calls           = 1;
	drain.drain_over_1ms        = 2;
	drain.drain_over_4ms        = 3;
	drain.drain_max_ms          = 4.5;
	drain.drain_total_ms        = 5.5;
	drain.drain_lag_ms_max      = 6.5;
	drain.item_max_ms           = 7.5;
	drain.adopted               = 8;
	drain.duplicates            = 9;
	drain.plan_mismatch         = 10;
	drain.plan_mismatch_reverse = 11;
	drain.failed_installs       = 12;
	drain.dropped_stopped       = 13;
	GenerationCounts counts;
	counts.posted            = 14;
	counts.handles_created   = 15;
	counts.handles_destroyed = 16;
	counts.bad_plans         = 17;
	counts.undrained_at_stop = 18;
	counts.pending           = 19;
	const auto  block        = MakeGenerationReport(SupervisorStats {}, drain, counts);
	const auto& out          = block.drain;
	CHECK(out.calls == 1 && out.over_1ms == 2 && out.over_4ms == 3);
	CHECK(out.max_ms == 4.5 && out.total_ms == 5.5 && out.lag_ms_max == 6.5 &&
	      out.item_max_ms == 7.5);
	CHECK(out.adopted == 8 && out.duplicates == 9 && out.plan_mismatch == 10);
	CHECK(out.plan_mismatch_reverse == 11 && out.failed_installs == 12 &&
	      out.dropped_stopped == 13);
	CHECK(out.posted == 14 && out.handles_created == 15 && out.handles_destroyed == 16);
	CHECK(out.bad_plans == 17 && out.undrained_at_stop == 18 && out.pending == 19);
}

// ---- Finishing: the rename of the generated file. ----

// How long the lingering fake stays alive after it wrote everything, which is far more than the
// settle time below and short enough to wait out.
constexpr int      LINGER_MS          = 2000;
constexpr auto     LINGER_SETTLE      = 400ms;
constexpr auto     WAIT_STEP          = 10ms;
constexpr auto     WAIT_LIMIT         = 10s;
constexpr auto     STOP_AFTER_START   = 300ms;
constexpr auto     HANG_KILL_TIMEOUT  = 300ms;
constexpr uint32_t REPEATED_CRASH_CAP = 2;

// A file's bytes, or nothing when it does not exist.
std::optional<Bytes> ReadBytes(const fs::path& path) {
	std::ifstream in(path, std::ios::binary);
	if (!in) {
		return std::nullopt;
	}
	return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The bytes a fake worker that finishes leaves in its output: the header and then one frame for
// each candidate, in the order it writes them.
Bytes ExpectedOutput() {
	Bytes bytes = GeneratedHeaderBytes({});
	for (const auto& candidate: FakeOrder({})) {
		const auto frame = FrameStoreRecord(ModuleRecord(candidate.id, candidate.hash));
		bytes.insert(bytes.end(), frame.begin(), frame.end());
	}
	return bytes;
}

// Waits until the supervisor has taken every record of the fake.
bool WaitForAllRecords(StoreSupervisor& supervisor) {
	for (auto waited = 0ms; waited < WAIT_LIMIT; waited += WAIT_STEP) {
		if (supervisor.Stats().posted == PROGRAMS * CANDIDATES_EACH) {
			return true;
		}
		std::this_thread::sleep_for(WAIT_STEP);
	}
	return false;
}

// What the loader and rename hooks saw, in the order they were called.
struct FinishEvents {
	std::mutex                mutex;
	std::vector<std::string>  order;
	std::chrono::milliseconds loader_wait {0};

	void Add(const std::string& name) {
		std::scoped_lock lock(mutex);
		order.push_back(name);
	}
	std::vector<std::string> Order() {
		std::scoped_lock lock(mutex);
		return order;
	}
};

// What a run of a finish test is given.
struct FinishOptions {
	bool with_loader  = false;
	bool rename_works = true;
	// A file that is in the generated file's place when the run starts.
	std::optional<std::string> old_generated;
	// Stops the run this long after it started instead of waiting for its end.
	std::optional<std::chrono::milliseconds> stop_after;
	// Stops the run once the worker has written everything and has had time to be found alive.
	bool                                   stop_after_records = false;
	std::function<void(SupervisorConfig&)> tweak;
};

// Records a call of the loader step and of the rename. The rename does what the supervisor's
// own does, or fails, as the options say.
void InstallFinishHooks(SupervisorConfig& config, FinishEvents& events,
                        const FinishOptions& options) {
	if (options.with_loader) {
		config.end_loader = [&events](std::stop_token, std::chrono::milliseconds wait) {
			events.loader_wait = wait;
			events.Add("loader");
		};
	}
	config.rename_file = [&events, works = options.rename_works](const fs::path& from,
	                                                             const fs::path& to) {
		events.Add("rename");
		return works && Common::File::RenameFile(from, to);
	};
}

// What a finish test looks at once the run is over, taken before the folder is gone.
struct FinishRun {
	SupervisorStats          stats;
	bool                     ended = false;
	std::vector<std::string> events;
	std::optional<Bytes>     generated;
	std::optional<Bytes>     temporary;
	std::vector<std::string> log;
};

// Runs the supervisor over a fake with the hooks installed and reads the two files at the end.
FinishRun RunFinish(const std::string& mode, const FinishOptions& options = {}) {
	TempDirectory directory;
	Collected     collected;
	auto          config = MakeConfig(directory, mode, collected);
	FinishEvents  events;
	InstallFinishHooks(config, events, options);
	if (options.tweak) {
		options.tweak(config);
	}
	if (options.old_generated) {
		std::ofstream(directory / "title.generated", std::ios::binary) << *options.old_generated;
	}
	FinishRun result;
	{
		StoreSupervisor supervisor(std::move(config));
		supervisor.Start();
		if (options.stop_after_records) {
			CHECK(WaitForAllRecords(supervisor));
			std::this_thread::sleep_for(LINGER_SETTLE);
		} else if (options.stop_after) {
			std::this_thread::sleep_for(*options.stop_after);
		} else {
			result.ended = supervisor.WaitForEnd(END_LIMIT);
		}
		supervisor.Stop();
		result.stats = supervisor.Stats();
	}
	result.events    = events.Order();
	result.generated = ReadBytes(directory / "title.generated");
	result.temporary = ReadBytes(directory / "title.generated.1.tmp");
	std::scoped_lock lock(collected.mutex);
	result.log = collected.log;
	return result;
}

// A finished worker's file is renamed to the generated file, over an older one, and the temporary
// is gone. The bytes are the worker's own, which nothing else in the folder could be. The loader
// step runs first when there is one and the run is the same without it.
void TestFinishRenamesTheFile() {
	const auto plain = RunFinish("normal", {.old_generated = "OLD"});
	CHECK(plain.ended && plain.stats.outcome == SupervisorOutcome::Finished);
	CHECK(plain.stats.persisted == GeneratedPersistence::Renamed &&
	      plain.stats.persist_error.empty());
	CHECK(plain.generated == ExpectedOutput());
	CHECK(!plain.temporary.has_value());
	// No loader gives no loader step, and the rename is all there is.
	CHECK(plain.events == std::vector<std::string> {"rename"});

	const auto loaded = RunFinish("normal", {.with_loader = true});
	CHECK(loaded.stats.outcome == SupervisorOutcome::Finished);
	CHECK(loaded.events == (std::vector<std::string> {"loader", "rename"}));
	CHECK(loaded.generated == ExpectedOutput() && !loaded.temporary.has_value());
}

// The worker has written the whole file and its summary and is still alive. Nothing is renamed
// then, since the worker may still hold the file, and the rename comes after its exit. The same
// run after the exit is the baseline: it differs only in the child having ended.
void TestNoRenameWhileTheChildIsAlive() {
	TempDirectory directory;
	Collected     collected;
	auto          config = MakeConfig(directory, "linger:" + std::to_string(LINGER_MS), collected);
	FinishEvents  events;
	InstallFinishHooks(config, events, {.with_loader = true});
	StoreSupervisor supervisor(std::move(config));
	supervisor.Start();
	CHECK(WaitForAllRecords(supervisor));
	std::this_thread::sleep_for(LINGER_SETTLE);
	// Every record is in, the file is complete and the summary is written, so the only thing that
	// holds the finish back is that the child lives.
	CHECK(supervisor.Stats().outcome == SupervisorOutcome::Running);
	CHECK(events.Order().empty());
	CHECK(!ReadBytes(directory / "title.generated").has_value());
	CHECK(ReadBytes(directory / "title.generated.1.tmp") == ExpectedOutput());
	CHECK(supervisor.Stats().persisted == GeneratedPersistence::NotFinished);

	CHECK(supervisor.WaitForEnd(END_LIMIT));
	supervisor.Stop();
	CHECK(supervisor.Stats().outcome == SupervisorOutcome::Finished);
	CHECK(events.Order() == (std::vector<std::string> {"loader", "rename"}));
	CHECK(ReadBytes(directory / "title.generated") == ExpectedOutput());
}

// A run that does not finish leaves no generated file and calls neither the loader step nor the
// rename. Each case ends a different way and the finished run above is their baseline.
void TestNoRenameWithoutAFinish() {
	const auto check_unfinished = [](const FinishRun& run, SupervisorOutcome outcome) {
		CHECK(run.stats.outcome == outcome);
		CHECK(run.stats.persisted == GeneratedPersistence::NotFinished);
		CHECK(run.events.empty());
		CHECK(!run.generated.has_value());
	};
	// Killed for a hang, with no restart left. The partial file is still there as a temporary.
	const auto killed = RunFinish("hang_start:3", {.with_loader = true, .tweak = [](auto& config) {
		                                               config.candidate_timeout = HANG_KILL_TIMEOUT;
		                                               config.max_restarts      = 0;
	                                               }});
	check_unfinished(killed, SupervisorOutcome::RestartsCapped);
	CHECK(killed.stats.failures.size() == 1 && killed.stats.failures[0].kind == FailureKind::Hang);
	CHECK(killed.temporary.has_value());
	// Crashes up to the cap.
	const auto capped = RunFinish("crash_always", {.with_loader = true, .tweak = [](auto& config) {
		                                               config.max_restarts = REPEATED_CRASH_CAP;
	                                               }});
	check_unfinished(capped, SupervisorOutcome::RestartsCapped);
	CHECK(capped.stats.failures.size() == REPEATED_CRASH_CAP + 1);
	// A fatal exit.
	check_unfinished(RunFinish("exit:3", {.with_loader = true}), SupervisorOutcome::Fatal);
	// Told to stop while the child works, and told to stop after it wrote everything and before it
	// exited.
	check_unfinished(
	    RunFinish("sleep_silent", {.with_loader = true, .stop_after = STOP_AFTER_START}),
	    SupervisorOutcome::Stopped);
	const auto stopped_late = RunFinish("linger:" + std::to_string(LINGER_MS),
	                                    {.with_loader = true, .stop_after_records = true});
	check_unfinished(stopped_late, SupervisorOutcome::Stopped);
	CHECK(stopped_late.temporary == ExpectedOutput());
}

// A rename that fails is reported with its reason, keeps the temporary file with the worker's
// bytes, and leaves no generated file. The run is the successful one with the rename replaced.
void TestRenameFailureKeepsTheTemporary() {
	const auto failed = RunFinish("normal", {.with_loader = true, .rename_works = false});
	CHECK(failed.ended && failed.stats.outcome == SupervisorOutcome::Finished);
	CHECK(failed.stats.persisted == GeneratedPersistence::RenameFailed);
	CHECK(failed.stats.persist_error.find("title.generated") != std::string::npos);
	CHECK(failed.stats.persist_error.find("source kept") != std::string::npos);
	CHECK(failed.events == (std::vector<std::string> {"loader", "rename"}));
	CHECK(!failed.generated.has_value());
	CHECK(failed.temporary == ExpectedOutput());
	CHECK(std::ranges::any_of(failed.log, [&](const std::string& line) {
		return line.find(failed.stats.persist_error) != std::string::npos;
	}));
	// The report carries the same two facts, which the successful run does not.
	const auto block = MakeSupervisorReport(failed.stats);
	CHECK(block.persisted == "rename_failed" && block.persist_error == failed.stats.persist_error);
	const auto good = MakeSupervisorReport(RunFinish("normal").stats);
	CHECK(good.persisted == "renamed" && good.persist_error.empty());
}

// A stop that comes while the loader step waits ends the wait at once, and renames nothing. The
// hook gets the stop token and the bounded wait, so a loader can check the one and honour the
// other. A loader that returns at once is the baseline, which renames.
void TestStopDuringTheLoaderStep() {
	TempDirectory     directory;
	Collected         collected;
	auto              config = MakeConfig(directory, "normal", collected);
	FinishEvents      events;
	std::atomic<bool> entered {false};
	InstallFinishHooks(config, events, {});
	config.end_loader = [&](std::stop_token token, std::chrono::milliseconds wait) {
		events.loader_wait = wait;
		entered            = true;
		std::mutex                  idle;
		std::condition_variable_any wake;
		std::unique_lock            lock(idle);
		wake.wait_for(lock, token, wait, [] { return false; });
	};
	StoreSupervisor supervisor(std::move(config));
	supervisor.Start();
	for (auto waited = 0ms; waited < WAIT_LIMIT && !entered; waited += WAIT_STEP) {
		std::this_thread::sleep_for(WAIT_STEP);
	}
	CHECK(entered);
	const auto begin = std::chrono::steady_clock::now();
	supervisor.Stop();
	// The wait is 5 s, so a prompt return shows the token was checked.
	CHECK(std::chrono::steady_clock::now() - begin < STOP_LIMIT);
	CHECK(events.loader_wait == GENERATION_LOADER_END_WAIT);
	const auto stats = supervisor.Stats();
	CHECK(stats.outcome == SupervisorOutcome::Stopped);
	CHECK(stats.persisted == GeneratedPersistence::NotFinished);
	CHECK(events.Order().empty());
	CHECK(!ReadBytes(directory / "title.generated").has_value());
}

// persisted is carried to the report whatever the outcome is, so a stop that came after a rename
// shows as stopped with renamed. A run that did not finish says not_finished.
void TestPersistedInTheReport() {
	SupervisorStats stats;
	stats.outcome              = SupervisorOutcome::Stopped;
	stats.persisted            = GeneratedPersistence::Renamed;
	const auto stopped_renamed = MakeSupervisorReport(stats);
	CHECK(stopped_renamed.outcome == "stopped" && stopped_renamed.persisted == "renamed");
	stats.persisted = GeneratedPersistence::NotFinished;
	CHECK(MakeSupervisorReport(stats).persisted == "not_finished");
	CHECK(std::string(PersistedName(GeneratedPersistence::RenameFailed)) == "rename_failed");
}

// ---- Loading the generated file of an earlier run, and verifying it. ----

// The records of the earlier file. Their ids are far from the fake worker's, so a record that
// reaches the sink with one of these ids came from the loader and from nothing else.
constexpr uint64_t LOADED_ID_BASE      = 9000;
constexpr size_t   LOADED_RECORDS      = 4;
constexpr size_t   LONG_LOAD_RECORDS   = 60;
constexpr uint64_t LOADED_HASH_BASE    = 0x7000;
constexpr auto     RECORD_SINK_DELAY   = 15ms;
constexpr auto     PROMPT_STOP_LIMIT   = 1s;
constexpr auto     STOP_DURING_LOAD_AT = 300ms;
constexpr auto     TINY_END_WAIT       = 50ms;
// A read this small cannot hold one frame of the fake, which makes a bad frame larger than a read.
constexpr uint64_t TINY_READ_BYTES = 40;
// The version a file of another generator has.
constexpr uint32_t OTHER_VERSION = GENERATOR_VERSION + 1;
// The seed of a store header of another build.
constexpr uint8_t OTHER_STAMP_SEED = 0x11;

std::vector<uint64_t> LoadedIds(size_t count) {
	std::vector<uint64_t> ids;
	for (size_t i = 0; i < count; i++) {
		ids.push_back(LOADED_ID_BASE + i);
	}
	return ids;
}

// What identifies the eboot of a file, and the same with another size, which the fake worker calls
// out of date.
constexpr EbootIdentity CURRENT_IDENTITY {.size = EBOOT_SIZE, .program_hash = IDENTITY_HASH};
constexpr EbootIdentity STALE_IDENTITY {.size = EBOOT_SIZE + 1, .program_hash = IDENTITY_HASH};

// The header of an earlier generated file, which the options change in the way a test needs.
struct EarlierFile {
	uint8_t       stamp_seed = 0xA0;
	uint32_t      version    = GENERATOR_VERSION;
	EbootIdentity identity   = CURRENT_IDENTITY;
};

// A generated file as an earlier run left it: the header and one frame for each id.
Bytes EarlierFileBytes(const std::vector<uint64_t>& ids, const EarlierFile& file = {}) {
	Bytes bytes = EncodeGeneratedHeader(StoreHeader(file.stamp_seed), file.version, file.identity);
	for (const auto id: ids) {
		const auto frame = FrameStoreRecord(ModuleRecord(id, LOADED_HASH_BASE + id));
		bytes.insert(bytes.end(), frame.begin(), frame.end());
	}
	return bytes;
}

// What a boot over an earlier file is given.
struct BootOptions {
	std::string mode = "normal";
	// The bytes that are in the generated file's place when the run starts.
	std::optional<Bytes> earlier;
	// How long each record takes in the sink, for a test that needs the load to be slow.
	std::chrono::milliseconds                          sink_delay {0};
	std::optional<std::chrono::milliseconds>           stop_after;
	std::function<void(SupervisorConfig&, Collected&)> tweak;
};

// What a boot left, taken before the folder is gone.
struct BootRun {
	SupervisorStats           stats;
	bool                      ended = false;
	std::vector<uint64_t>     ids;
	std::optional<Bytes>      generated;
	std::optional<Bytes>      temporary;
	std::vector<std::string>  log;
	std::chrono::milliseconds stop_took {0};
};

// Runs the supervisor over the fake with an earlier file in place, and reads the files at the end.
BootRun RunBoot(const BootOptions& options = {}) {
	TempDirectory directory;
	Collected     collected;
	auto          config = MakeConfig(directory, options.mode, collected);
	if (options.sink_delay > 0ms) {
		config.sink = [&collected, delay = options.sink_delay](DecodedModuleRecord&& record) {
			std::this_thread::sleep_for(delay);
			std::scoped_lock lock(collected.mutex);
			collected.ids.push_back(record.id);
		};
	}
	if (options.tweak) {
		options.tweak(config, collected);
	}
	if (options.earlier) {
		std::ofstream(directory / "title.generated", std::ios::binary)
		    .write(reinterpret_cast<const char*>(options.earlier->data()),
		           static_cast<std::streamsize>(options.earlier->size()));
	}
	BootRun result;
	{
		StoreSupervisor supervisor(std::move(config));
		supervisor.Start();
		if (options.stop_after) {
			std::this_thread::sleep_for(*options.stop_after);
		} else {
			result.ended = supervisor.WaitForEnd(END_LIMIT);
		}
		const auto begin = std::chrono::steady_clock::now();
		supervisor.Stop();
		result.stop_took = std::chrono::duration_cast<std::chrono::milliseconds>(
		    std::chrono::steady_clock::now() - begin);
		result.stats = supervisor.Stats();
	}
	result.generated = ReadBytes(directory / "title.generated");
	result.temporary = ReadBytes(directory / "title.generated.1.tmp");
	std::scoped_lock lock(collected.mutex);
	result.ids = collected.ids;
	result.log = collected.log;
	return result;
}

// The ids a boot should have given, each once, from several lists.
std::vector<uint64_t> UnionOf(std::vector<uint64_t> first, const std::vector<uint64_t>& second) {
	first.insert(first.end(), second.begin(), second.end());
	std::ranges::sort(first);
	first.erase(std::unique(first.begin(), first.end()), first.end());
	return first;
}

// A boot over a current earlier file loads its records, and the worker that is started in Verify
// mode answers exit 10 and writes nothing. One spawn, no failure, nothing renamed, and the file is
// the earlier one byte for byte. The ids are the file's own, which the fake worker never writes.
// Without an earlier file the same boot generates, which is the baseline for every field.
void TestSecondBootLoadsTheFile() {
	const auto file = EarlierFileBytes(LoadedIds(LOADED_RECORDS));
	const auto run  = RunBoot({.earlier = file});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.ids == LoadedIds(LOADED_RECORDS));
	CHECK(run.stats.spawns == 1 && run.stats.restarts == 0 && run.stats.failures.empty());
	CHECK(run.stats.requests.size() == 1 && run.stats.requests[0].mode == WorkerMode::Verify);
	CHECK(run.stats.source == GenerationSource::Loaded);
	CHECK(run.stats.persisted == GeneratedPersistence::Existing && run.stats.persist_error.empty());
	CHECK(run.stats.file_bytes == file.size());
	CHECK(run.stats.verify_ms.has_value() && *run.stats.verify_ms > 0.0);
	CHECK(!run.stats.loader_abandoned && !run.stats.regenerated_after_damage);
	CHECK(run.stats.posted == LOADED_RECORDS && run.stats.duplicates == 0);
	CHECK(run.generated == file && !run.temporary.has_value());

	const auto baseline = RunBoot();
	CHECK(baseline.stats.outcome == SupervisorOutcome::Finished);
	CHECK(baseline.ids == IdsOf({0, 1, 2}));
	CHECK(baseline.stats.requests.size() == 1 &&
	      baseline.stats.requests[0].mode == WorkerMode::Generate);
	CHECK(baseline.stats.source == GenerationSource::Generated);
	CHECK(baseline.stats.persisted == GeneratedPersistence::Renamed);
	CHECK(!baseline.stats.verify_ms.has_value());
	CHECK(baseline.generated == ExpectedOutput());
}

// A Verify worker's exit 10 is not the end while records of the file are still to load. The sink
// takes a while for each record, so the worker answers long before the load is done. A run that
// ended at the exit would have fewer than all of them.
void TestExit10WaitsForTheLoad() {
	const auto run = RunBoot({.earlier    = EarlierFileBytes(LoadedIds(LONG_LOAD_RECORDS)),
	                          .sink_delay = RECORD_SINK_DELAY});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.ids == LoadedIds(LONG_LOAD_RECORDS));
	CHECK(run.stats.source == GenerationSource::Loaded && run.stats.spawns == 1);
	// The answer came while the load was far from done, which is what the wait is for.
	CHECK(run.stats.verify_ms.has_value() && run.stats.last_record_ms.has_value() &&
	      *run.stats.last_record_ms > *run.stats.verify_ms);
}

// A file that is not for this build is ignored, by the stamp or by the version: the worker is asked
// to generate, no record of the file reaches the sink, and the file is replaced. Each differs from
// the current file of the test above only in that header field.
void TestHeaderMismatchIsIgnored() {
	const std::vector<EarlierFile> others = {{.stamp_seed = OTHER_STAMP_SEED},
	                                         {.version = OTHER_VERSION}};
	for (const auto& other: others) {
		const auto run = RunBoot({.earlier = EarlierFileBytes(LoadedIds(LOADED_RECORDS), other)});
		CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
		CHECK(run.ids == IdsOf({0, 1, 2}));
		CHECK(run.stats.spawns == 1 && run.stats.requests[0].mode == WorkerMode::Generate);
		CHECK(run.stats.source == GenerationSource::Generated);
		CHECK(run.stats.persisted == GeneratedPersistence::Renamed);
		CHECK(run.generated == ExpectedOutput());
		CHECK(!run.stats.verify_ms.has_value());
	}
	// A file too short for a header, and an empty one, are ignored too.
	for (const size_t bytes: {size_t {3}, size_t {0}}) {
		Bytes cut = EarlierFileBytes(LoadedIds(LOADED_RECORDS));
		cut.resize(bytes);
		const auto run = RunBoot({.earlier = cut});
		CHECK(run.stats.outcome == SupervisorOutcome::Finished && run.ids == IdsOf({0, 1, 2}));
		CHECK(run.stats.source == GenerationSource::Generated);
	}
}

// An earlier file of another eboot has a good header, so its records load, and the worker finds its
// identity out of date and generates a new file. The records the two have in common reach the sink
// once, the file is replaced, and the source says both happened. The records are not lost to the
// replacement, and the loader is not what the child's output is compared with.
void TestStaleIdentityRegenerates() {
	auto ids = LoadedIds(LOADED_RECORDS);
	// One record that the worker writes again.
	ids.push_back(ModuleIdOf(0, 0));
	const auto run = RunBoot({.earlier = EarlierFileBytes(ids, {.identity = STALE_IDENTITY})});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.spawns == 1 && run.stats.restarts == 0 && run.stats.failures.empty());
	CHECK(run.stats.requests[0].mode == WorkerMode::Verify);
	CHECK(run.stats.source == GenerationSource::LoadedRegenerated);
	CHECK(run.stats.persisted == GeneratedPersistence::Renamed);
	CHECK(run.generated == ExpectedOutput() && !run.temporary.has_value());
	CHECK(!run.stats.verify_ms.has_value());
	// Every id once. The id that both wrote is the one duplicate.
	const auto expected = UnionOf(IdsOf({0, 1, 2}), LoadedIds(LOADED_RECORDS));
	CHECK(SortedIds(run.ids) == expected && run.ids.size() == expected.size());
	CHECK(run.stats.duplicates == 1);
	CHECK(run.stats.file_bytes == ExpectedOutput().size());
}

// A damaged file that a worker calls current is not trusted. The worker is started once more, in
// Generate mode, and that is not a restart. The frames before the damage were loaded and the ones
// after it were not. A file cut off in the middle of a frame is the same, and the undamaged file of
// the first test is the baseline.
void TestDamagedFileWithExit10Regenerates() {
	const auto good = EarlierFileBytes(LoadedIds(LOADED_RECORDS));
	const auto header_bytes =
	    EncodeGeneratedHeader(StoreHeader(), GENERATOR_VERSION, CURRENT_IDENTITY).size();
	const auto frame_bytes =
	    FrameStoreRecord(ModuleRecord(LOADED_ID_BASE, LOADED_HASH_BASE + LOADED_ID_BASE)).size();
	// The second frame is broken and bytes follow it, so one record is good. The other file lacks
	// the end of the last frame, so three are.
	struct DamagedCase {
		Bytes  file;
		size_t good_records;
	};
	Bytes mid_damage = good;
	mid_damage[header_bytes + frame_bytes + sizeof(StoreFrame) + 2] ^= 0xFF;
	Bytes cut_tail = good;
	cut_tail.resize(good.size() - 3);
	for (const auto& damaged: {DamagedCase {mid_damage, 1}, DamagedCase {cut_tail, 3}}) {
		const auto run = RunBoot({.earlier = damaged.file});
		CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
		CHECK(run.stats.spawns == 2 && run.stats.restarts == 0 && run.stats.failures.empty());
		CHECK(run.stats.requests.size() == 2 && run.stats.requests[0].mode == WorkerMode::Verify &&
		      run.stats.requests[1].mode == WorkerMode::Generate);
		CHECK(run.stats.regenerated_after_damage);
		CHECK(run.stats.source == GenerationSource::LoadedRegenerated);
		CHECK(run.stats.persisted == GeneratedPersistence::Renamed);
		CHECK(run.generated == ExpectedOutput());
		// The frames before the damage were loaded, the ones after it were not, and the new
		// generation follows. The worker of the first spawn answered at once and wrote nothing.
		auto expected = LoadedIds(damaged.good_records);
		for (const auto id: IdsOf({0, 1, 2})) {
			expected.push_back(id);
		}
		CHECK(run.ids == expected);
	}
}

// The respawn for a damaged file happens once. A worker that answers exit 10 whatever it is asked
// is the case that could repeat it: its second answer comes in Generate mode, where it is fatal, so
// there are two spawns and no third. The decision itself has the flag first, so a damaged file that
// meets it a second time is fatal, and a whole file never is.
void TestDamagedRespawnHappensOnce() {
	auto damaged = EarlierFileBytes(LoadedIds(LOADED_RECORDS));
	damaged.resize(damaged.size() - 3);
	const auto run = RunBoot({.mode = "exit:10", .earlier = damaged});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Fatal);
	CHECK(run.stats.inactive_reason == "worker_exit_10");
	CHECK(run.stats.spawns == 2 && run.stats.restarts == 0 && run.stats.regenerated_after_damage);
	CHECK(run.stats.requests.size() == 2 && run.stats.requests[1].mode == WorkerMode::Generate);

	CHECK(DecideAfterVerify(LoaderEnd::Complete, false) == VerifiedAction::Finish);
	CHECK(DecideAfterVerify(LoaderEnd::Damaged, false) == VerifiedAction::Respawn);
	CHECK(DecideAfterVerify(LoaderEnd::Damaged, true) == VerifiedAction::Fatal);
	CHECK(DecideAfterVerify(LoaderEnd::Complete, true) == VerifiedAction::Finish);
}

// Exit 10 is fatal when the worker was not asked to verify, and a whole earlier file does not
// change that. The same answer to a Verify request is the done case of the first test.
void TestExit10IsFatalInGenerateMode() {
	const auto run = RunBoot({.mode = "exit:10"});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Fatal);
	CHECK(run.stats.inactive_reason == "worker_exit_10" && run.stats.spawns == 1);
	const auto ignored = RunBoot(
	    {.mode    = "exit:10",
	     .earlier = EarlierFileBytes(LoadedIds(LOADED_RECORDS), {.stamp_seed = OTHER_STAMP_SEED})});
	CHECK(ignored.stats.outcome == SupervisorOutcome::Fatal &&
	      ignored.stats.inactive_reason == "worker_exit_10");
	const auto verified = RunBoot({.mode = "exit:10", .earlier = EarlierFileBytes(LoadedIds(1))});
	CHECK(verified.stats.outcome == SupervisorOutcome::Finished);
}

// A worker that fails after a Verify spawn is started again in Generate mode, since the question of
// whether the file is current is over. The baseline has no file, and both its spawns generate.
void TestFirstRestartSwitchesToGenerate() {
	const auto run = RunBoot(
	    {.mode    = "crash_start:2",
	     .earlier = EarlierFileBytes(LoadedIds(LOADED_RECORDS), {.identity = STALE_IDENTITY})});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.requests.size() == 2);
	if (run.stats.requests.size() == 2) {
		CHECK(run.stats.requests[0].mode == WorkerMode::Verify);
		CHECK(run.stats.requests[1].mode == WorkerMode::Generate);
	}
	CHECK(run.stats.restarts == 1 && !run.stats.regenerated_after_damage);
	const auto baseline = RunBoot({.mode = "crash_start:2"});
	CHECK(baseline.stats.requests.size() == 2 &&
	      baseline.stats.requests[0].mode == WorkerMode::Generate &&
	      baseline.stats.requests[1].mode == WorkerMode::Generate);
}

// A stop during the load returns promptly. The sink takes 15 ms for a record, so the whole load
// takes far longer than the limit, and a load that did not give way at the stop would show in the
// time. The load had started and was not done, and the file is as it was.
void TestStopDuringTheLoad() {
	const auto file = EarlierFileBytes(LoadedIds(LONG_LOAD_RECORDS));
	const auto run  = RunBoot({.mode       = "sleep_silent",
	                           .earlier    = file,
	                           .sink_delay = RECORD_SINK_DELAY,
	                           .stop_after = STOP_DURING_LOAD_AT});
	CHECK(run.stop_took < PROMPT_STOP_LIMIT);
	CHECK(run.stats.outcome == SupervisorOutcome::Stopped);
	CHECK(!run.ids.empty() && run.ids.size() < LONG_LOAD_RECORDS);
	CHECK(run.stats.persisted == GeneratedPersistence::NotFinished);
	CHECK(run.generated == file);
	// The same run left to itself loads every record, so the stop is what cut it short.
	const auto whole =
	    RunBoot({.mode = "normal", .earlier = file, .sink_delay = RECORD_SINK_DELAY});
	CHECK(whole.ids == LoadedIds(LONG_LOAD_RECORDS));
}

// The finish lets a load that is behind reach its end before it renames. The earlier file is of an
// out of date eboot and is long, so the child finishes while records are still to load, which the
// sink notes when the child's last record arrives. At the rename every record of the file has
// arrived. A rename taken before the loader ended would see fewer.
void TestFinishEndsTheLoaderBeforeTheRename() {
	std::atomic<size_t> loaded_at_last_generated {0};
	std::atomic<size_t> loaded_at_rename {0};
	const auto          last_generated = ModuleIdOf(PROGRAMS - 1, CANDIDATES_EACH - 1);
	const auto          run            = RunBoot(
        {.earlier = EarlierFileBytes(LoadedIds(LONG_LOAD_RECORDS), {.identity = STALE_IDENTITY}),
	                         .tweak = [&](SupervisorConfig& config, Collected& collected) {
             config.sink = [&, &c = collected](DecodedModuleRecord&& record) {
                 std::this_thread::sleep_for(RECORD_SINK_DELAY);
                 std::scoped_lock lock(c.mutex);
                 c.ids.push_back(record.id);
                 if (record.id == last_generated) {
                     loaded_at_last_generated = static_cast<size_t>(std::ranges::count_if(
                         c.ids, [](uint64_t id) { return id >= LOADED_ID_BASE; }));
                 }
             };
             config.rename_file = [&, &c = collected](const fs::path& from, const fs::path& to) {
                 {
                     std::scoped_lock lock(c.mutex);
                     loaded_at_rename = static_cast<size_t>(std::ranges::count_if(
                         c.ids, [](uint64_t id) { return id >= LOADED_ID_BASE; }));
                 }
                 return Common::File::RenameFile(from, to);
             };
         }});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.persisted == GeneratedPersistence::Renamed && !run.stats.loader_abandoned);
	// The premise: the child's last record came while the load was far from done.
	CHECK(loaded_at_last_generated.load() < LONG_LOAD_RECORDS);
	CHECK(loaded_at_rename.load() == LONG_LOAD_RECORDS);
	CHECK(run.generated == ExpectedOutput());
}

// A load that cannot end within the wait is abandoned, counted, and the rename goes on. persisted
// stays renamed, since it is the new file that is on disk. The run of the test above, with the
// wait long enough, is the baseline: it abandons nothing.
void TestAbandonedLoader() {
	const auto earlier =
	    EarlierFileBytes(LoadedIds(LONG_LOAD_RECORDS), {.identity = STALE_IDENTITY});
	const auto run = RunBoot({.earlier    = earlier,
	                          .sink_delay = RECORD_SINK_DELAY * 3,
	                          .tweak      = [](SupervisorConfig& config, Collected&) {
                                  config.loader_end_wait = TINY_END_WAIT;
	                          }});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.loader_abandoned);
	CHECK(run.stats.persisted == GeneratedPersistence::Renamed);
	CHECK(run.generated == ExpectedOutput() && !run.temporary.has_value());
	CHECK(run.stats.source == GenerationSource::LoadedRegenerated);
	CHECK(std::ranges::count_if(run.ids, [](uint64_t id) { return id >= LOADED_ID_BASE; }) <
	      static_cast<std::ptrdiff_t>(LONG_LOAD_RECORDS));
	const auto patient = RunBoot({.earlier = earlier, .sink_delay = RECORD_SINK_DELAY});
	CHECK(patient.stats.outcome == SupervisorOutcome::Finished && !patient.stats.loader_abandoned);
}

// A bad frame that is larger than one read of the file can never be judged by a read, so the loader
// gives the file up as damaged and does not read it again for ever. A worker that calls the file
// current then regenerates it. With the default read size the same file is judged by the first
// reads, which is the baseline: it is damaged in the same way and no stall is involved.
void TestBadFrameLargerThanAReadIsDamaged() {
	auto       file = EarlierFileBytes(LoadedIds(LOADED_RECORDS));
	const auto header_bytes =
	    EncodeGeneratedHeader(StoreHeader(), GENERATOR_VERSION, CURRENT_IDENTITY).size();
	file[header_bytes + sizeof(StoreFrame) + 2] ^= 0xFF;
	const auto run = RunBoot({.earlier = file, .tweak = [](SupervisorConfig& config, Collected&) {
		                          config.loader_read_bytes = TINY_READ_BYTES;
	                          }});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.regenerated_after_damage && run.stats.spawns == 2);
	CHECK(run.generated == ExpectedOutput());
	const auto baseline = RunBoot({.earlier = file});
	CHECK(baseline.stats.regenerated_after_damage && baseline.stats.spawns == 2);
	// The small reads still load a good file, one frame at a time, so the tiny window is not what
	// breaks the loader.
	const auto good = RunBoot({.earlier = EarlierFileBytes(LoadedIds(LOADED_RECORDS)),
	                           .tweak   = [](SupervisorConfig& config, Collected&) {
                                   config.loader_read_bytes = TINY_READ_BYTES;
	                           }});
	CHECK(good.ids == LoadedIds(LOADED_RECORDS) && good.stats.source == GenerationSource::Loaded);
}

// A run that is stopped after an earlier file was accepted says it loaded, since the game may end
// before the run does. The child sleeps so the run cannot finish. The same stop with no file is the
// baseline, which generated nothing and says generated.
void TestStoppedRunReportsLoaded() {
	const auto loaded = RunBoot({.mode       = "sleep_silent",
	                             .earlier    = EarlierFileBytes(LoadedIds(LOADED_RECORDS)),
	                             .stop_after = STOP_DURING_LOAD_AT});
	CHECK(loaded.stats.outcome == SupervisorOutcome::Stopped);
	CHECK(loaded.stats.source == GenerationSource::Loaded);
	CHECK(loaded.ids == LoadedIds(LOADED_RECORDS));
	const auto baseline = RunBoot({.mode = "sleep_silent", .stop_after = STOP_DURING_LOAD_AT});
	CHECK(baseline.stats.outcome == SupervisorOutcome::Stopped);
	CHECK(baseline.stats.source == GenerationSource::Generated);
	// A file that is not accepted is no load.
	const auto ignored = RunBoot(
	    {.mode    = "sleep_silent",
	     .earlier = EarlierFileBytes(LoadedIds(LOADED_RECORDS), {.stamp_seed = OTHER_STAMP_SEED}),
	     .stop_after = STOP_DURING_LOAD_AT});
	CHECK(ignored.stats.source == GenerationSource::Generated);
}

// The report carries the new figures by name, and an existing file is said to be existing.
void TestLoadFiguresInTheReport() {
	CHECK(std::string(PersistedName(GeneratedPersistence::Existing)) == "existing");
	CHECK(std::string(SourceName(GenerationSource::Generated)) == "generated");
	CHECK(std::string(SourceName(GenerationSource::Loaded)) == "loaded");
	CHECK(std::string(SourceName(GenerationSource::LoadedRegenerated)) == "loaded_regenerated");
	const auto run   = RunBoot({.earlier = EarlierFileBytes(LoadedIds(LOADED_RECORDS))});
	const auto block = MakeSupervisorReport(run.stats);
	CHECK(block.source == "loaded" && block.persisted == "existing");
	CHECK(block.verify_ms == run.stats.verify_ms && block.verify_ms.has_value());
	CHECK(block.file_bytes == run.stats.file_bytes && block.file_bytes > 0);
	CHECK(!block.loader_abandoned && !block.regenerated_after_damage);
	SupervisorStats stats;
	stats.loader_abandoned         = true;
	stats.regenerated_after_damage = true;
	const auto flagged             = MakeSupervisorReport(stats);
	CHECK(flagged.loader_abandoned && flagged.regenerated_after_damage);
	CHECK(flagged.source == "generated" && !flagged.verify_ms.has_value());
}

// ---- Removing the helper files of dead and of finished runs. ----

// Process ids of the fixtures. The injected check calls LIVE_PID running and every other id dead,
// so no real dead process is needed. OWN_PID is what the direct calls give as the current process.
constexpr uint32_t DEAD_PID = 4243;
constexpr uint32_t LIVE_PID = 4242;
constexpr uint32_t OWN_PID  = 7777;
// The suffixes of one run's files.
const std::vector<std::string> RUN_SUFFIXES = {".tmp", ".tmp.progress", ".tmp.request",
                                               ".tmp.progress.out"};

// Makes a small file, and the folders above it.
void Touch(const fs::path& path) {
	fs::create_directories(path.parent_path());
	std::ofstream(path, std::ios::binary) << "x";
}

// The four files of one run of a title and a pid.
std::vector<std::string> RunFileNames(const std::string& title, uint32_t pid) {
	std::vector<std::string> names;
	for (const auto& suffix: RUN_SUFFIXES) {
		names.push_back(title + ".generated." + std::to_string(pid) + suffix);
	}
	return names;
}

// Every file and folder below a folder, as paths relative to it. Links are listed and not followed.
std::set<std::string> ListFolder(const fs::path& folder) {
	std::set<std::string> names;
	std::error_code       error;
	for (fs::recursive_directory_iterator it(folder, error), end; !error && it != end;
	     it.increment(error)) {
		names.insert(it->path().lexically_relative(folder).generic_string());
	}
	return names;
}

bool OnlyLivePidRuns(uint32_t pid) {
	return pid == LIVE_PID;
}
bool EverythingRuns(uint32_t) {
	return true;
}
// How long the slow check takes for a dead pid. A cleanup that came after the first spawn would
// finish after the worker had looked, so the delay makes the order observable.
constexpr auto SLOW_CHECK_DELAY = 400ms;
bool           SlowCheckLivePidRuns(uint32_t pid) {
    if (pid != LIVE_PID) {
        std::this_thread::sleep_for(SLOW_CHECK_DELAY);
    }
    return pid == LIVE_PID;
}

// The name of a file is a stale one of the title only when all of it matches, with the title taken
// literally, a numeric pid and one of the four suffixes. Each near miss differs from a good name in
// one place.
void TestStaleFileNames() {
	CHECK(StaleFilePid("title.generated.4243.tmp", "title") == DEAD_PID);
	CHECK(StaleFilePid("title.generated.4243.tmp.progress", "title") == DEAD_PID);
	CHECK(StaleFilePid("title.generated.4243.tmp.request", "title") == DEAD_PID);
	CHECK(StaleFilePid("title.generated.4243.tmp.progress.out", "title") == DEAD_PID);
	const std::vector<std::string> near_misses = {
	    "title.generated",
	    "title.shaders",
	    "title.bin",
	    "title2.generated.4243.tmp",
	    "xtitle.generated.4243.tmp",
	    "title.generated.abc.tmp",
	    "title.generated.-5.tmp",
	    "title.generated..tmp",
	    "title.generated.4243",
	    "title.generated.4243.tmp.progress.bak",
	    "title.generated.4243.tmp.progress.out.bak",
	    "title.generated.4243.tmp.requests",
	    "title.generated.4243.tmpx",
	    "title.generated.4243x.tmp",
	    "title.generated.99999999999999999999.tmp",
	    "Title.generated.4243.tmp",
	    "title_generated.4243.tmp",
	};
	for (const auto& name: near_misses) {
		CHECK(!StaleFilePid(name, "title").has_value());
	}
	// A title with characters that a pattern would read specially is still taken literally.
	CHECK(StaleFilePid("a+b(1).generated.7.tmp", "a+b(1)") == 7);
	CHECK(!StaleFilePid("aab(1).generated.7.tmp", "a+b(1)").has_value());
	CHECK(!StaleFilePid("a.b.generated.7.tmp", "axb").has_value());
}

// Creates the files of the cleanup test in a folder, and tells what is meant to go and what stays.
struct StaleFixture {
	std::vector<std::string> doomed;
	std::vector<std::string> kept;
	bool                     symlink_made = false;
};

StaleFixture MakeStaleFixture(const fs::path& folder) {
	StaleFixture fixture;
	fixture.doomed = RunFileNames("title", DEAD_PID);
	for (const auto& name: fixture.doomed) {
		Touch(folder / name);
	}
	// What is never deleted: the live run's four files, the current process's, the generated file,
	// the main store, another title that this one is a prefix of, names that do not match in full
	// and a matching name in a subfolder.
	fixture.kept = RunFileNames("title", LIVE_PID);
	for (const auto& name: RunFileNames("title", OWN_PID)) {
		fixture.kept.push_back(name);
	}
	for (const auto& name: RunFileNames("title2", DEAD_PID)) {
		fixture.kept.push_back(name);
	}
	for (const auto& name:
	     {"title.generated", "title.shaders", "title.bin", "title.generated.abc.tmp",
	      "title.generated.4243.tmp.progress.bak", "sub/title.generated.4243.tmp",
	      "sub/title.generated.4243.tmp.request"}) {
		fixture.kept.push_back(name);
	}
	for (const auto& name: fixture.kept) {
		Touch(folder / name);
	}
	fixture.kept.push_back("sub");
	// A directory with a matching name, with a file in it that goes with it if it were removed.
	fs::create_directories(folder / "title.generated.4244.tmp");
	Touch(folder / "title.generated.4244.tmp" / "inner.txt");
	fixture.kept.push_back("title.generated.4244.tmp");
	fixture.kept.push_back("title.generated.4244.tmp/inner.txt");
	// A link with a matching name, to a file outside the folder. Creating one may be refused.
	std::error_code error;
	Touch(folder.parent_path() / "link_target.txt");
	fs::create_symlink(folder.parent_path() / "link_target.txt",
	                   folder / "title.generated.4245.tmp", error);
	fixture.symlink_made = !error;
	if (fixture.symlink_made) {
		fixture.kept.push_back("title.generated.4245.tmp");
	}
	return fixture;
}

// The cleaner removes the four kinds of dead pid and nothing else. The folder holds the same names
// beside them for every case that must stay: a live pid, the current pid (which the check calls
// dead, so only the current pid rule keeps it), other titles, near misses, a directory, a link and
// a subfolder.
void TestStaleCleanupRemovesOnlyDeadRuns() {
	TempDirectory directory;
	const auto    folder  = (directory / "x").parent_path() / "store";
	const auto    fixture = MakeStaleFixture(folder);
	if (!fixture.symlink_made) {
		std::fprintf(stderr, "skipped the symlink case, the system refused to create a link\n");
	}
	std::vector<std::string> log;
	const auto               result =
	    RemoveStaleGenerationFiles(folder, "title", OWN_PID, OnlyLivePidRuns,
	                               [&](std::string_view line) { log.emplace_back(line); });
	CHECK(result.removed == fixture.doomed.size() && result.failed == 0);
	for (const auto& name: fixture.doomed) {
		CHECK(!fs::exists(folder / name));
		CHECK(std::ranges::any_of(
		    log, [&](const std::string& line) { return line.find(name) != std::string::npos; }));
	}
	for (const auto& name: fixture.kept) {
		std::error_code error;
		CHECK(fs::exists(fs::symlink_status(folder / name, error)));
	}
	CHECK(log.size() == fixture.doomed.size());
	// A link's target is as it was, so nothing followed the link.
	CHECK(fs::exists(folder.parent_path() / "link_target.txt"));
	// The folder holds exactly what was meant to stay and nothing more.
	std::set<std::string> expected(fixture.kept.begin(), fixture.kept.end());
	CHECK(ListFolder(folder) == expected);
}

// What a run in a folder left behind.
struct FolderRun {
	SupervisorStats          stats;
	bool                     ended = false;
	std::set<std::string>    remaining;
	std::vector<std::string> log;
	// The names that were there when the loader step ran, which is the last moment before the files
	// of a finished run are deleted.
	std::set<std::string> at_loader_end;
	// The same at the moment of the rename.
	std::set<std::string> at_rename;
};

// What a run in a folder is given.
struct FolderOptions {
	std::function<void(const fs::path&)>                    setup;
	std::function<bool(uint32_t)>                           process_running = EverythingRuns;
	std::optional<Bytes>                                    earlier;
	bool                                                    rename_works = true;
	std::function<void(SupervisorConfig&, const fs::path&)> tweak;
	std::optional<std::chrono::milliseconds>                stop_after;
};

// Runs the supervisor over the fake in a folder of its own, listing the folder at the loader step
// and the rename and at the end.
FolderRun RunInFolder(const std::string& mode, const FolderOptions& options = {}) {
	TempDirectory directory;
	Collected     collected;
	auto          config = MakeConfig(directory, mode, collected);
	const auto    folder = (directory / "x").parent_path();
	FolderRun     result;
	std::mutex    seen_mutex;
	config.process_running = options.process_running;
	config.end_loader      = [&](std::stop_token, std::chrono::milliseconds) {
        std::scoped_lock lock(seen_mutex);
        result.at_loader_end = ListFolder(folder);
	};
	config.rename_file = [&, works = options.rename_works](const fs::path& from,
	                                                       const fs::path& to) {
		{
			std::scoped_lock lock(seen_mutex);
			result.at_rename = ListFolder(folder);
		}
		return works && Common::File::RenameFile(from, to);
	};
	if (options.tweak) {
		options.tweak(config, folder);
	}
	if (options.setup) {
		options.setup(folder);
	}
	if (options.earlier) {
		std::ofstream(folder / "title.generated", std::ios::binary)
		    .write(reinterpret_cast<const char*>(options.earlier->data()),
		           static_cast<std::streamsize>(options.earlier->size()));
	}
	{
		StoreSupervisor supervisor(std::move(config));
		supervisor.Start();
		if (options.stop_after) {
			std::this_thread::sleep_for(*options.stop_after);
		} else {
			result.ended = supervisor.WaitForEnd(END_LIMIT);
		}
		supervisor.Stop();
		result.stats = supervisor.Stats();
	}
	result.remaining = ListFolder(folder);
	std::scoped_lock lock(collected.mutex);
	result.log = collected.log;
	return result;
}

// The names of this run's three helper files, as the supervisor names them.
const std::vector<std::string> OWN_HELPERS = {"title.generated.1.tmp.progress",
                                              "title.generated.1.tmp.request",
                                              "title.generated.1.tmp.progress.out"};

bool HasAll(const std::set<std::string>& names, const std::vector<std::string>& wanted) {
	return std::ranges::all_of(wanted, [&](const auto& name) { return names.contains(name); });
}
bool HasNone(const std::set<std::string>& names, const std::vector<std::string>& unwanted) {
	return std::ranges::none_of(unwanted, [&](const auto& name) { return names.contains(name); });
}

// At the start of a run the files of a dead process are gone, and the files of a live one are not.
// The first worker finds the stale file already gone, which only a cleanup before the spawn gives,
// and so does the first record of an earlier file given to the sink. The live pid is the baseline
// of the same folder, the worker's check differs from it only in the pid.
void TestStaleCleanupAtStart() {
	const auto dead_names = RunFileNames("title", DEAD_PID);
	const auto live_names = RunFileNames("title", LIVE_PID);
	const auto run        = RunInFolder("stale_gone", {.setup =
	                                                       [&](const fs::path& folder) {
                                                        for (const auto& name: dead_names) {
                                                            Touch(folder / name);
                                                        }
                                                        for (const auto& name: live_names) {
                                                            Touch(folder / name);
                                                        }
                                                    },
	                                                   .process_running = SlowCheckLivePidRuns});
	// The worker saw no stale file, so it was not fatal, and it finished and renamed.
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.files_removed >= dead_names.size());
	CHECK(HasNone(run.remaining, dead_names));
	CHECK(HasAll(run.remaining, live_names));
	CHECK(std::ranges::any_of(run.log, [](const std::string& line) {
		return line.find("title.generated.4243.tmp.progress.out") != std::string::npos;
	}));
	// The same folder with the pid called running keeps the stale file, and the worker then ends
	// fatally with the exit that says so, which is what shows the check is the worker's own.
	const auto kept = RunInFolder(
	    "stale_gone", {.setup = [&](const fs::path& folder) { Touch(folder / STALE_PROBE_NAME); }});
	CHECK(kept.stats.outcome == SupervisorOutcome::Fatal &&
	      kept.stats.inactive_reason == "worker_exit_" + std::to_string(STALE_STILL_THERE_EXIT));
	CHECK(kept.remaining.contains(STALE_PROBE_NAME) && kept.stats.files_removed == 0);

	// Before the load: the first record of the earlier file reaches the sink with the stale file
	// already gone.
	std::atomic<bool> stale_seen_at_first_record {true};
	std::atomic<bool> got_record {false};
	const auto        loaded = RunInFolder(
        "normal",
        {.setup = [&](const fs::path& folder) { Touch(folder / STALE_PROBE_NAME); },
	            .process_running = OnlyLivePidRuns,
	            .earlier         = EarlierFileBytes(LoadedIds(LOADED_RECORDS)),
	            .tweak =
	                [&](SupervisorConfig& config, const fs::path& folder) {
                 config.sink = [&, probe = folder / STALE_PROBE_NAME](DecodedModuleRecord&&) {
                     if (!got_record.exchange(true)) {
                         stale_seen_at_first_record = fs::exists(probe);
                     }
                 };
             }});
	CHECK(loaded.stats.source == GenerationSource::Loaded && got_record);
	CHECK(!stale_seen_at_first_record);
}

// A successful generation removes its three helper files after the rename, and not before. The
// listing at the rename shows they were there to remove, and the end shows they are gone and the
// generated file is in place. The text file is written by the worker only in this mode.
void TestFinishRemovesOwnHelpers() {
	const auto run = RunInFolder("out_normal");
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.persisted == GeneratedPersistence::Renamed);
	CHECK(HasAll(run.at_rename, OWN_HELPERS));
	CHECK(run.remaining.contains("title.generated"));
	CHECK(HasNone(run.remaining, OWN_HELPERS));
	CHECK(!run.remaining.contains("title.generated.1.tmp"));
	CHECK(run.stats.files_removed == OWN_HELPERS.size() && run.stats.remove_failures == 0);
	CHECK(MakeSupervisorReport(run.stats).files_removed == OWN_HELPERS.size());

	// The same run whose rename fails keeps all three, and the temporary file. Differing only in
	// the rename, it shows the success is what removes them.
	const auto failed = RunInFolder("out_normal", {.rename_works = false});
	CHECK(failed.stats.persisted == GeneratedPersistence::RenameFailed);
	CHECK(HasAll(failed.remaining, OWN_HELPERS));
	CHECK(failed.remaining.contains("title.generated.1.tmp") &&
	      !failed.remaining.contains("title.generated"));
	CHECK(failed.stats.files_removed == 0);
}

// A run that fails keeps its files for the next boot to remove as the files of a dead process.
// Each ends in a way that leaves the request file at least, and the capped one leaves all three.
void TestFailedRunKeepsItsFiles() {
	const auto capped = RunInFolder(
	    "crash_always_out",
	    {.tweak = [](SupervisorConfig& config, const fs::path&) { config.max_restarts = 2; }});
	CHECK(capped.stats.outcome == SupervisorOutcome::RestartsCapped);
	CHECK(capped.stats.failures.size() == 3 && capped.stats.failures[0].exit_code == CRASH_EXIT);
	CHECK(HasAll(capped.remaining, OWN_HELPERS));
	CHECK(capped.stats.files_removed == 0);

	const auto fatal = RunInFolder("exit:3");
	CHECK(fatal.stats.outcome == SupervisorOutcome::Fatal);
	CHECK(fatal.remaining.contains("title.generated.1.tmp.request"));
	CHECK(fatal.stats.files_removed == 0);

	const auto stopped = RunInFolder("sleep_silent", {.stop_after = STOP_AFTER_START});
	CHECK(stopped.stats.outcome == SupervisorOutcome::Stopped);
	CHECK(stopped.remaining.contains("title.generated.1.tmp.request"));
	CHECK(stopped.stats.files_removed == 0);
}

// A finish of a loaded file has no temporary file and no rename, and Verify still left the text
// file, so it goes with the request file. The listing at the loader step shows the text file was
// there, which the worker wrote in this mode, and the generated file is untouched.
void TestLoadedFinishRemovesTheOutFile() {
	const auto file = EarlierFileBytes(LoadedIds(LOADED_RECORDS));
	const auto run  = RunInFolder("out_normal", {.earlier = file});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.source == GenerationSource::Loaded &&
	      run.stats.persisted == GeneratedPersistence::Existing);
	CHECK(run.at_loader_end.contains("title.generated.1.tmp.progress.out"));
	CHECK(run.at_loader_end.contains("title.generated.1.tmp.request"));
	CHECK(HasNone(run.remaining, OWN_HELPERS));
	CHECK(run.remaining.contains("title.generated"));
	CHECK(run.stats.files_removed >= 2 && run.stats.remove_failures == 0);
}

// A helper that cannot be deleted is logged by name and counted, the other helpers still go, and
// the run still finishes. The worker text file is replaced by a folder with a file in it just
// before the deletion, which no system lets fs::remove delete, and the same run without it is the
// baseline that removes all three.
void TestHelperRemovalFailureIsLogged() {
	const auto run =
	    RunInFolder("normal", {.tweak = [](SupervisorConfig& config, const fs::path& folder) {
		                config.end_loader = [folder](std::stop_token, std::chrono::milliseconds) {
			                Touch(folder / "title.generated.1.tmp.progress.out" / "held.txt");
		                };
	                }});
	CHECK(run.ended && run.stats.outcome == SupervisorOutcome::Finished);
	CHECK(run.stats.persisted == GeneratedPersistence::Renamed);
	CHECK(run.stats.remove_failures == 1);
	CHECK(run.remaining.contains("title.generated.1.tmp.progress.out/held.txt"));
	CHECK(!run.remaining.contains("title.generated.1.tmp.request"));
	CHECK(std::ranges::any_of(run.log, [](const std::string& line) {
		return line.find("could not remove") != std::string::npos &&
		       line.find("title.generated.1.tmp.progress.out") != std::string::npos;
	}));
	const auto baseline = RunInFolder("normal");
	CHECK(baseline.stats.remove_failures == 0);
}

} // namespace

int main(int argc, char** argv) {
	if (argc >= 4 && std::string_view(argv[1]) == "--fake-store-worker") {
		return RunFakeWorker(argv[2], PathFromUtf8(argv[3]));
	}
	TestGateSingleReasons();
	TestGatePrecedence();
	TestNormalRun();
	TestPredictCrashSkipsTheProgram();
	TestStartCrashFailsTheProgram();
	TestHangIsKilled();
	TestSilentChild();
	TestOrderMismatchRestartsFresh();
	TestDropLinesFeedFailedHashes();
	TestFreshRestartCutsTheFile();
	TestDamagedFrameRestartsFresh();
	TestTruncatedRestartsFresh();
	TestHeaderErrorIsFatal();
	TestFatalExitCodes();
	TestRestartCap();
	TestStopWithLiveChild();
	TestBadRecordIsSkipped();
	TestCrashWithDamagedFileIsBlamed();
	TestOtherStageIsSkipped();
	TestMalformedRecordsAreSkipped();
	TestBlameReadsTheWholeLog();
	TestPriorityRefusalIsLogged();
	TestOutputResetFailure();
	TestTimingStats();
	TestFinishedMsOnlyAfterTheEnd();
	TestDecodeMsMax();
	TestHashLists();
	TestPredictionRows();
	TestPredictionRowsSurviveAFreshRestart();
	TestCrashBeforeOrderBlamesNothing();
	TestDropsAreListedForAFinishedRun();
	TestSpawnUsageCpu();
	TestPeakSourceRank();
	TestFailureCpuAtATimeout();
	TestFailureOutTail();
	TestPeakMemoryIsTheLargest();
	TestWorkerSummaryFields();
	TestReportNames();
	TestReportFromFinishedAndEndedRuns();
	TestReportRows();
	TestDrainFiguresAreMapped();
	TestFinishRenamesTheFile();
	TestNoRenameWhileTheChildIsAlive();
	TestNoRenameWithoutAFinish();
	TestRenameFailureKeepsTheTemporary();
	TestStopDuringTheLoaderStep();
	TestPersistedInTheReport();
	TestSecondBootLoadsTheFile();
	TestExit10WaitsForTheLoad();
	TestHeaderMismatchIsIgnored();
	TestStaleIdentityRegenerates();
	TestDamagedFileWithExit10Regenerates();
	TestDamagedRespawnHappensOnce();
	TestExit10IsFatalInGenerateMode();
	TestFirstRestartSwitchesToGenerate();
	TestStopDuringTheLoad();
	TestFinishEndsTheLoaderBeforeTheRename();
	TestAbandonedLoader();
	TestBadFrameLargerThanAReadIsDamaged();
	TestStoppedRunReportsLoaded();
	TestLoadFiguresInTheReport();
	TestStaleFileNames();
	TestStaleCleanupRemovesOnlyDeadRuns();
	TestStaleCleanupAtStart();
	TestFinishRemovesOwnHelpers();
	TestFailedRunKeepsItsFiles();
	TestLoadedFinishRemovesTheOutFile();
	TestHelperRemovalFailureIsLogged();
	if (g_failures == 0) {
		std::puts("Shader store supervisor tests passed");
	}
	return g_failures == 0 ? 0 : 1;
}

// The typed IR's implementation, which the module record's plan needs. It is amalgamated here to
// keep this target small, like shader_store_record_tests.
#include "../src/graphics/shader/recompiler/ir/Block.cpp"
#include "../src/graphics/shader/recompiler/ir/Program.cpp"
#include "../src/graphics/shader/recompiler/ir/Type.cpp"
#include "../src/graphics/shader/recompiler/ir/Value.cpp"
#include "../src/graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
