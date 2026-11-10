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
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <optional>
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
	if (run.Is("crash_always")) {
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

// The whole fake. A mode such as `exit:3` ends before it does anything else.
int RunFakeWorker(const std::string& mode_text, const fs::path& request_file) {
	const auto request = ReadWorkerRequestFile(request_file);
	if (!request) {
		return WORKER_EXIT_BAD_REQUEST;
	}
	auto      words = SplitMode(mode_text);
	const int spawn = SpawnIndex(*request);
	FakeRun   run(*request, words, spawn);
	if (run.Is("exit")) {
		return static_cast<int>(run.Arg(1));
	}
	if (run.Is("sleep_silent")) {
		SleepSeconds(HANG_SECONDS);
		return 0;
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
	return FakeBuild(run);
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
	supervisor.Start();
	std::this_thread::sleep_for(300ms);
	const auto running = supervisor.Stats();
	CHECK(running.outcome == SupervisorOutcome::Running);
	CHECK(running.finished_ms == 0.0 && running.ended_ms == 0.0 &&
	      !running.child_open_ms.has_value());
	supervisor.Stop();
	const auto stopped = supervisor.Stats();
	CHECK(stopped.outcome == SupervisorOutcome::Stopped);
	// The stop came after the 300 ms above, so the end cannot be earlier than that. The run did not
	// finish, so finished_ms stays zero, which tells it from ended_ms.
	CHECK(stopped.ended_ms >= 250.0);
	CHECK(stopped.finished_ms == 0.0);
	// A run that ends for good has an end and no finish.
	const auto fatal = Run("exit:3");
	CHECK(fatal.stats.outcome == SupervisorOutcome::Fatal);
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
	CHECK(!fatal_block.finished_ms.has_value() && fatal_block.ended_ms.has_value());

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
	CHECK(!stopped.finished_ms.has_value() && stopped.ended_ms.has_value());
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
