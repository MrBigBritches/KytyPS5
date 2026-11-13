// Tests the prescan's preparer over real programs, translated in this process. The in-process
// transport answers with the Stream worker's own code, so a Plan and a Compile here are the child's.
// A scripted transport wraps the same answers when a test needs an outcome a healthy translation
// never gives, such as a blamed request or a corrupted static key. Every check that proves a
// behaviour has a baseline that differs only in that behaviour, as the comments say.
//
// The supervised tests go through the real StreamSupervisor and SupervisorPrescanTransport to a
// child process. The child is this binary started with --fake-prescan-worker <mode> <request
// file>, which runs the real Stream worker with a hook that crashes, hangs or exits early as the
// mode says, so a crash on one request is repeatable. One smoke test starts the real emulator
// binary in its worker mode instead.
#include "SrtWalkerTestMemory.h"
#include "common/childProcess.h"
#include "common/crashUi.h"
#include "common/file.h"
#include "graphics/host_gpu/renderer/pipeline/prescanPreparer.h"
#include "graphics/shader/storegen/prescanTransport.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"
#include "graphics/shader/storegen/streamWorker.h"
#include "graphics/shader/storegen/workerProcess.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xxhash.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

// ---- Values of the host and the programs. ----
constexpr uint32_t HOST_SUBGROUP_SIZE = 32;
constexpr uint32_t MAX_SHARED_BYTES   = 32768;
constexpr uint32_t WAVE_SIZE          = 32;
constexpr uint32_t SCRATCH_DWORDS     = 0;
// The two strides of one program's two variants. A buffer's stride is part of its specialization.
constexpr uint32_t STRIDE_A = 4;
constexpr uint32_t STRIDE_B = 16;
// Any address and size for a buffer descriptor, which nothing here dereferences.
constexpr uint32_t BUFFER_BASE    = 0x10000;
constexpr uint32_t BUFFER_RECORDS = 64;
constexpr uint32_t STRIDE_SHIFT   = 16;
constexpr uint32_t FORMAT_SHIFT   = 12;
// The scanner's key of a dispatch, which the answers are published under.
constexpr uint64_t KEY       = 0x1234ABCD5678EF00ull;
constexpr uint64_t OTHER_KEY = 0x7777AAAA5555CC00ull;
// The key of a repeat dispatch, answered busy while its program's snapshot is open.
constexpr uint64_t REPEAT_KEY = 0x3333BBBB9999DD00ull;
// The key of a third program's dispatch.
constexpr uint64_t THIRD_KEY = 0x5555EEEE1111FF00ull;
// How many passes a test drives the preparer for before it gives up, so a preparer that keeps
// finding work fails the check instead of looping for ever. Three passes finish a dispatch.
constexpr int MAX_PASSES = 16;
// How long the threaded tests wait for the preparer, and how long Stop may take.
constexpr std::chrono::seconds      WAIT_LIMIT {10};
constexpr std::chrono::milliseconds STOP_LIMIT {1000};
constexpr std::chrono::milliseconds POLL_SLICE {5};
// How long the threaded test gives a stopped preparer to show that it still takes items.
constexpr std::chrono::milliseconds SETTLE_TIME {200};
// The run's watchdog.
constexpr std::chrono::seconds TEST_TIMEOUT {240};
// How long a supervised test waits for the helper, which has to start a child first.
constexpr std::chrono::seconds HELPER_WAIT_LIMIT {30};
// How long a test waits to see that nothing more arrives.
constexpr std::chrono::milliseconds QUIET_TIME {300};
// The thread count of the second unit of a program. Thread counts are in the static key, so the
// same program with this count is another unit.
constexpr uint32_t OTHER_UNIT_THREADS = 2;

// ---- The helper in a child process. ----
// The command line word that makes this binary a fake helper.
constexpr const char* FAKE_FLAG = "--fake-prescan-worker";
// The exit code of a fake that crashes on purpose, which no real worker code uses.
constexpr int FAKE_CRASH_EXIT = 77;
// The seq of a program's first Compile. Its Plan is seq 1, and the preparer sends the Compile only
// once the plan has come.
constexpr uint64_t FIRST_COMPILE_SEQ = 2;
// How long the hanging fake sleeps, longer than any test waits for it.
constexpr std::chrono::seconds FAKE_HANG_TIME {20};
// The tag of the stop test's request.
constexpr uint64_t STOP_TEST_TAG = 0x5709;
// How many dispatches the supervised test sends, and how many answers an answering helper gives
// for them. A and B each need a Plan and a Compile, and C, of A's unit, only a Compile.
constexpr size_t   SUPERVISED_DISPATCHES = 3;
constexpr uint64_t SUPERVISED_ANSWERS    = 5;
// How many dispatches of one program the supervised lane-off test sends after the first.
constexpr int LANE_OFF_REPEATS = 2;
// How many times a test directory's removal is tried. A child that was just ended may hold a file
// a moment longer on Windows.
constexpr int REMOVE_TRIES = 2;

// ---- Shader encodings, spelled out with plain shifts. ----
constexpr uint32_t VOP1_PREFIX          = 0x3f;
constexpr uint32_t VOP1_MOV_B32         = 0x01;
constexpr uint32_t INLINE_INTEGER_BASE  = 128;
constexpr uint32_t STORED_VALUE         = 7;
constexpr uint32_t VALUE_VGPR           = 1;
constexpr uint32_t MUBUF_PREFIX         = 0x38;
constexpr uint32_t MUBUF_STORE_FORMAT_X = 0x04;
constexpr uint32_t MUBUF_STORE_DWORD    = 0x1c;
constexpr uint32_t MUBUF_OFFEN_BIT      = 12;
constexpr uint32_t MUBUF_SOFFSET_ZERO   = 128;
constexpr uint32_t SMEM_PREFIX          = 0x3d;
constexpr uint32_t SMEM_LOAD_DWORDX4    = 0x02;
constexpr uint32_t SGPR_NULL            = 125;
constexpr uint32_t S_WAITCNT_ALL        = 0xbf8c0000u;
constexpr uint32_t S_ENDPGM             = 0xbf810000u;
constexpr uint32_t SGPRS_PER_DESCRIPTOR = 4;
// The SRT program loads its descriptor into s[4:7] from the pointer in s[0:1].
constexpr uint32_t LOADED_DESCRIPTOR_SGPR = 4;
constexpr uint32_t POINTER_SGPRS          = 2;

const ComputeKeyDevice DEVICE {HOST_SUBGROUP_SIZE, MAX_SHARED_BYTES};

// ---- Programs. ----

// A program as the shader map and the scanner would see it.
struct Program {
	std::vector<uint32_t> code;
	uint32_t              user_sgprs = 0;
};

uint32_t MovValue() {
	return (VOP1_PREFIX << 25u) | (VALUE_VGPR << 17u) | (VOP1_MOV_B32 << 9u) |
	       (INLINE_INTEGER_BASE + STORED_VALUE);
}

void AppendStore(std::vector<uint32_t>& code, uint32_t opcode, uint32_t descriptor_sgpr) {
	code.push_back((MUBUF_PREFIX << 26u) | (opcode << 18u) | (1u << MUBUF_OFFEN_BIT));
	code.push_back((MUBUF_SOFFSET_ZERO << 24u) | ((descriptor_sgpr / SGPRS_PER_DESCRIPTOR) << 16u) |
	               (VALUE_VGPR << 8u));
}

// Stores a constant through the buffer whose descriptor is in user SGPRs 0 to 3, with a plain
// dword store or a formatted one.
Program DirectProgram(uint32_t store_opcode) {
	Program program;
	program.code.push_back(MovValue());
	AppendStore(program.code, store_opcode, 0);
	program.code.push_back(S_ENDPGM);
	program.user_sgprs = SGPRS_PER_DESCRIPTOR;
	return program;
}

// Loads the descriptor from the address in user SGPRs 0 and 1, then stores through it, so the
// walker reads guest memory for it.
Program SrtProgram() {
	Program program;
	program.code.push_back((SMEM_PREFIX << 26u) | (SMEM_LOAD_DWORDX4 << 18u) |
	                       (LOADED_DESCRIPTOR_SGPR << 6u));
	program.code.push_back(SGPR_NULL << 25u);
	program.code.push_back(S_WAITCNT_ALL);
	program.code.push_back(MovValue());
	AppendStore(program.code, MUBUF_STORE_DWORD, LOADED_DESCRIPTOR_SGPR);
	program.code.push_back(S_ENDPGM);
	program.user_sgprs = POINTER_SGPRS;
	return program;
}

uint64_t AddressOf(const Program& program) {
	return reinterpret_cast<uint64_t>(program.code.data());
}

// A buffer descriptor with a stride, a destination select and a 32-bit unsigned format.
std::array<uint32_t, SGPRS_PER_DESCRIPTOR> Descriptor(uint32_t stride, uint32_t dst_sel) {
	return {BUFFER_BASE, stride << STRIDE_SHIFT, BUFFER_RECORDS,
	        dst_sel | (static_cast<uint32_t>(Prospero::BufferFormat::k32UInt) << FORMAT_SHIFT)};
}

// The ring item the scanner would make for one dispatch of the program.
PrescanItem ItemFor(const Program& program, std::span<const uint32_t> user_data, uint64_t key) {
	PrescanItem item;
	item.inputs.cs_regs.num_thread_x = 1;
	item.inputs.cs_regs.num_thread_y = 1;
	item.inputs.cs_regs.num_thread_z = 1;
	item.inputs.cs_regs.wave_size    = WAVE_SIZE;
	item.inputs.cs_regs.user_sgpr    = program.user_sgprs;
	item.inputs.workgroup_counts     = {1, 1, 1};
	std::copy(user_data.begin(), user_data.end(), item.user_data.value);
	item.program_address = AddressOf(program);
	item.key_hash        = key;
	item.seq             = 1;
	item.user_data_count = program.user_sgprs;
	return item;
}

// ---- Guest memory. ----

// When set, the reader throws, which stands for a failure in the middle of preparation.
std::atomic<bool> g_reader_throws {false};
// When set, the harness's sink throws, which stands for a failure while a module is adopted.
std::atomic<bool> g_sink_throws {false};
// Reads that start at this address are counted, so a test can see how often a program's code is
// copied. Zero counts nothing.
std::atomic<uint64_t> g_counted_address {0};
std::atomic<uint32_t> g_counted_reads {0};

// Reads host memory as guest memory, since the test programs live in host memory.
bool ReadHost(void* /*userdata*/, uint64_t address, std::span<uint32_t> values) {
	if (g_reader_throws.load()) {
		throw std::runtime_error("test reader failure");
	}
	if (address != 0 && address == g_counted_address.load()) {
		g_counted_reads++;
	}
	std::memcpy(values.data(), reinterpret_cast<const void*>(address), values.size_bytes());
	return true;
}

// One page of memory that can be made unreadable, so a walker load from it faults.
class ProtectedPage {
public:
	static constexpr size_t SIZE = 4096;

	ProtectedPage() {
#if defined(_WIN32)
		m_memory = VirtualAlloc(nullptr, SIZE, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
#else
		m_memory = mmap(nullptr, SIZE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
	}
	~ProtectedPage() {
#if defined(_WIN32)
		VirtualFree(m_memory, 0, MEM_RELEASE);
#else
		munmap(m_memory, SIZE);
#endif
	}
	ProtectedPage(const ProtectedPage&)            = delete;
	ProtectedPage& operator=(const ProtectedPage&) = delete;

	[[nodiscard]] uint32_t* Words() { return static_cast<uint32_t*>(m_memory); }
	[[nodiscard]] uint64_t  Address() const { return reinterpret_cast<uint64_t>(m_memory); }
	// Takes away every access to the page.
	void Guard() {
#if defined(_WIN32)
		DWORD previous = 0;
		VirtualProtect(m_memory, SIZE, PAGE_NOACCESS, &previous);
#else
		mprotect(m_memory, SIZE, PROT_NONE);
#endif
	}

private:
	void* m_memory = nullptr;
};

// ---- A transport with scripted outcomes. ----

// Answers like the in-process transport, unless the script says otherwise. A request can be refused
// at Submit, completed Blamed or Dropped, answered with another status and no payload, or answered
// with its echoed static key corrupted. The Compile script applies to the first compile_scripted
// Compiles, and later ones are answered plainly.
struct Script {
	StreamSubmitStatus          plan_submit  = StreamSubmitStatus::Queued;
	StreamOutcome               plan_outcome = StreamOutcome::Answered;
	std::optional<StreamStatus> plan_status;
	bool                        corrupt_plan_key = false;
	StreamSubmitStatus          compile_submit   = StreamSubmitStatus::Queued;
	StreamOutcome               compile_outcome  = StreamOutcome::Answered;
	std::optional<StreamStatus> compile_status;
	bool                        corrupt_compile_key = false;
	uint32_t                    compile_scripted    = UINT32_MAX;
	// The first plans_held Plans are queued and left unanswered, whatever plan_submit says, and Stop
	// completes them as Dropped, as the supervisor does with what it still holds.
	uint32_t plans_held = 0;
	// The first compiles_held Compiles are queued and left unanswered, and Stop answers them, as an
	// answer the supervisor read just before it stopped would reach the callback.
	uint32_t compiles_held = 0;
	// Called at each Submit, before the request is answered.
	std::function<void(const StreamRequest&)> on_submit;
	// What reached Submit, by kind, and what was answered by translation.
	uint64_t plan_attempts    = 0;
	uint64_t compile_attempts = 0;
	uint64_t translated       = 0;
};

class ScriptedTransport final: public PrescanTransport {
public:
	ScriptedTransport(Script& script, PrescanCompletionHandler handler)
	    : m_script(script), m_handler(std::move(handler)) {}

	StreamSubmitResult Submit(StreamRequest request, std::shared_ptr<const std::vector<uint8_t>> code,
	                          uint64_t tag) override {
		if (m_script.on_submit) {
			m_script.on_submit(request);
		}
		const bool plan     = request.kind == StreamRequestKind::Plan;
		const bool scripted = plan || m_compiles_seen++ < m_script.compile_scripted;
		(plan ? m_script.plan_attempts : m_script.compile_attempts)++;
		uint32_t& held_so_far = plan ? m_plans_held_so_far : m_compiles_held_so_far;
		if (held_so_far < (plan ? m_script.plans_held : m_script.compiles_held)) {
			held_so_far++;
			request.code = *code;
			request.seq  = m_next_seq++;
			m_held.push_back({.request = std::move(request), .tag = tag, .answer = !plan});
			return {StreamSubmitStatus::Queued, m_held.back().request.seq};
		}
		const auto submit = !scripted ? StreamSubmitStatus::Queued
		                    : plan    ? m_script.plan_submit
		                              : m_script.compile_submit;
		if (submit != StreamSubmitStatus::Queued) {
			return {submit, 0};
		}
		request.code = *code;
		request.seq  = m_next_seq++;
		StreamCompletion completion {.seq = request.seq, .kind = request.kind, .tag = tag};
		completion.outcome = !scripted ? StreamOutcome::Answered
		                     : plan    ? m_script.plan_outcome
		                               : m_script.compile_outcome;
		if (completion.outcome == StreamOutcome::Answered) {
			Answer(request, plan, scripted, completion.response);
		}
		m_handler(std::move(completion));
		return {StreamSubmitStatus::Queued, request.seq};
	}
	// Completes the held Plans as Dropped and answers the held Compiles, once.
	void Stop() override {
		auto held = std::move(m_held);
		m_held.clear();
		for (const auto& entry: held) {
			StreamCompletion completion {.seq     = entry.request.seq,
			                             .kind    = entry.request.kind,
			                             .outcome = StreamOutcome::Dropped,
			                             .tag     = entry.tag};
			if (entry.answer) {
				completion.outcome = StreamOutcome::Answered;
				Answer(entry.request, false, false, completion.response);
			}
			m_handler(std::move(completion));
		}
	}

private:
	// Translates the request in this process and applies the script's changes to the answer.
	void Answer(const StreamRequest& request, bool plan, bool scripted, StreamResponse& response) {
		m_script.translated++;
		response = AnswerStreamRequest(request, HostValues {}, DEVICE);
		const auto& status  = plan ? m_script.plan_status : m_script.compile_status;
		const bool  corrupt = plan ? m_script.corrupt_plan_key : m_script.corrupt_compile_key;
		if (scripted && status) {
			response.status = *status;
			response.payload.clear();
		}
		if (scripted && corrupt && !response.static_state.empty()) {
			response.static_state[0] ^= 1u;
		}
	}

	// A request kept back until Stop, which answers it or drops it.
	struct HeldRequest {
		StreamRequest request;
		uint64_t      tag    = 0;
		bool          answer = false;
	};

	Script&                  m_script;
	PrescanCompletionHandler m_handler;
	uint64_t                 m_next_seq             = 1;
	uint32_t                 m_compiles_seen        = 0;
	uint32_t                 m_plans_held_so_far    = 0;
	uint32_t                 m_compiles_held_so_far = 0;
	std::vector<HeldRequest> m_held;
};

PrescanTransportFactory ScriptedFactory(Script& script) {
	return [&script](const HostValues&, const ComputeKeyDevice&, PrescanCompletionHandler handler)
	           -> std::unique_ptr<PrescanTransport> {
		return std::make_unique<ScriptedTransport>(script, std::move(handler));
	};
}

// ---- The preparer under test. ----

// A preparer over a shader map of test programs, host memory and a sink that keeps what it gets.
// The hand-off is on the heap, since its ring is large. A supervised harness passes host values
// whose shared memory limit is the device's, since its child takes the limit from them.
struct Harness {
	explicit Harness(PrescanTransportFactory factory = {}, PrescanMapLookup lookup = {},
	                 const HostValues& host = {}) {
		PrescanPreparerConfig config;
		config.lookup = std::move(lookup);
		if (!config.lookup) {
			config.lookup = [this](uint64_t address) {
				const auto found = map.find(address);
				return found == map.end() ? std::nullopt : std::optional(found->second);
			};
		}
		config.read_memory    = ReadHost;
		config.host           = host;
		config.device         = DEVICE;
		config.make_transport = factory ? std::move(factory) : InProcessPrescanTransportFactory(&transport);
		config.sink           = [this](DecodedModuleRecord&& record, std::vector<uint8_t>&& /*bytes*/) {
			if (g_sink_throws.load()) {
				throw std::runtime_error("test sink failure");
			}
			adopted.push_back(std::move(record));
		};
		preparer              = std::make_unique<PrescanPreparer>(*handoff, std::move(config));
	}

	// Puts the program in the shader map, hashed as ShaderMapUserData hashes it.
	void Map(const Program& program) {
		const auto bytes = static_cast<uint32_t>(program.code.size() * sizeof(uint32_t));
		map[AddressOf(program)] = {.code_size_bytes     = bytes,
		                           .scratch_size_dwords = SCRATCH_DWORDS,
		                           .hash                = XXH3_64bits(program.code.data(), bytes)};
	}
	void Push(const PrescanItem& item) { CHECK(handoff->ring.TryPush(item)); }
	// Runs passes until one finds nothing to do.
	void Drive() {
		for (int pass = 0; pass < MAX_PASSES && preparer->RunOnce(); ++pass) {
		}
	}
	[[nodiscard]] PrescanPreparerCounters Counters() const { return preparer->Counters(); }

	std::unique_ptr<PrescanHandoff>                   handoff = std::make_unique<PrescanHandoff>();
	std::unordered_map<uint64_t, ShaderMapSnapshot>   map;
	std::vector<DecodedModuleRecord>                  adopted;
	InProcessPrescanTransport*                        transport = nullptr;
	std::unique_ptr<PrescanPreparer>                  preparer;
};

// Waits for a condition, polling, up to the limit.
bool WaitFor(const std::function<bool()>& condition,
             std::chrono::milliseconds limit = WAIT_LIMIT) {
	const auto until = std::chrono::steady_clock::now() + limit;
	while (!condition()) {
		if (std::chrono::steady_clock::now() >= until) {
			return false;
		}
		std::this_thread::sleep_for(POLL_SLICE);
	}
	return true;
}

// Runs passes on the test's thread until the condition holds. A helper in a child answers on the
// supervisor's threads, so its outcomes arrive between passes rather than within one.
bool DriveUntil(Harness& harness, const std::function<bool()>& condition) {
	return WaitFor(
	    [&] {
		    (void)harness.preparer->RunOnce();
		    return condition();
	    },
	    HELPER_WAIT_LIMIT);
}

// ---- The fake helper and its supervisor. ----

// What this binary does when a supervisor starts it as a fake helper. It runs the real Stream
// worker, so every answer is the child's own, and the mode only stages a failure. crash_compile
// dies at the program's first Compile, hang never finishes a request, exit_at_once exits before it
// connects, and answer changes nothing.
int RunFakeHelper(const std::string& mode, const std::filesystem::path& request_file) {
	const auto request = ReadWorkerRequestFile(request_file);
	if (!request) {
		return WORKER_EXIT_BAD_REQUEST;
	}
	// The same start as RunShaderStoreWorkerProcess, so a staged crash shows no dialog and the
	// child's text goes where the real child's does.
	Common::DisableCrashUi();
	RedirectWorkerOutputTo(request->progress_path);
	Common::ChildProcess::EnterChildMode(request->parent_pid);
	if (mode == "exit_at_once") {
		return WORKER_EXIT_PIPE_FAILED;
	}
	StreamWorkerOptions options;
	options.before_work = [mode](uint64_t seq) {
		if (mode == "crash_compile" && seq == FIRST_COMPILE_SEQ) {
			std::_Exit(FAKE_CRASH_EXIT);
		}
		if (mode == "hang") {
			std::this_thread::sleep_for(FAKE_HANG_TIME);
		}
	};
	return RunStreamWorker(*request, std::move(options)).exit_code;
}

// A directory for one test's helper files, which is removed with it.
class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		const auto name    = "kyty_prescan_preparer_tests_" +
		                  std::to_string(Common::CurrentProcessId()) + "_" + std::to_string(counter++);
		m_path = std::filesystem::temp_directory_path() / name;
		std::filesystem::remove_all(m_path);
		std::filesystem::create_directories(m_path);
	}
	// A child that was just ended may hold a file a moment longer on Windows, so a failed removal is
	// tried once more after a short wait.
	~TempDirectory() {
		for (int attempt = 0; attempt < REMOVE_TRIES; ++attempt) {
			std::error_code error;
			std::filesystem::remove_all(m_path, error);
			if (!error) {
				return;
			}
			std::this_thread::sleep_for(QUIET_TIME);
		}
	}
	TempDirectory(const TempDirectory&)            = delete;
	TempDirectory& operator=(const TempDirectory&) = delete;

	[[nodiscard]] std::filesystem::path Base() const { return m_path / "prescan"; }

private:
	std::filesystem::path m_path;
};

// The host values of a supervised harness, whose shared memory limit is the device's. The module id
// does not cover the profile, so a supervised test sets a profile flag that changes the test
// program's SPIR-V and compares the SPIR-V itself. The flag can be left off for the baseline that
// shows it does change the SPIR-V.
HostValues SupervisedHost(bool profile_flag = true) {
	HostValues host;
	host.max_shared_bytes             = MAX_SHARED_BYTES;
	host.profile.robust_buffer_access = profile_flag;
	return host;
}

// The supervisor's settings for this binary as a fake helper in the given mode.
StreamSupervisorConfig FakeHelperConfig(const TempDirectory& directory, const std::string& mode) {
	StreamSupervisorConfig config;
	config.files_base               = directory.Base();
	config.program                  = Common::ExecutablePath();
	config.arguments_before_request = {FAKE_FLAG, mode};
	return config;
}

// ---- Tests. ----

// A program's first variant sends one Plan and one Compile, and its second variant sends a Compile
// and no Plan, since the plan is kept per program. The baseline is the first variant, which must
// send exactly one Plan, so a preparer that never sends one cannot pass. The mutation is to drop
// the plan after each Compile. The second variant then sends a second Plan.
void TestSecondVariantSkipsPlan() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	Harness    harness;
	harness.Map(program);
	harness.Push(ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY));
	harness.Drive();
	CHECK(harness.transport->Plans() == 1);
	CHECK(harness.transport->Compiles() == 1);
	CHECK(harness.adopted.size() == 1);

	harness.Push(ItemFor(program, Descriptor(STRIDE_B, DstSel(4, 5, 6, 7)), KEY));
	harness.Drive();
	CHECK(harness.transport->Plans() == 1);
	CHECK(harness.transport->Compiles() == 2);
	CHECK(harness.Counters().variant_modules == 2);
	CHECK(harness.adopted.size() == 2 && harness.adopted[0].id != harness.adopted[1].id);
	std::puts("SecondVariantSkipsPlan ok");
}

// A formatted buffer with a reserved destination select is refused before it is sent, and its twin
// with a valid select, the only difference, is sent. Compiles are held, so neither is built, which
// keeps a mutant from reaching the emitter's EXIT. The mutation is to skip the validation. The
// refused one is then sent.
void TestInvalidDescriptorNeverSent() {
	const auto program = DirectProgram(MUBUF_STORE_FORMAT_X);
	for (const bool reserved: {false, true}) {
		Harness harness;
		harness.Map(program);
		harness.transport->SetHoldCompiles(true);
		const auto select = reserved ? DstSel(2, 5, 6, 7) : DstSel(4, 5, 6, 7);
		harness.Push(ItemFor(program, Descriptor(STRIDE_A, select), KEY));
		harness.Drive();
		const auto counters = harness.Counters();
		const auto refused  = counters.invalid[static_cast<size_t>(DescriptorInvalid::ReservedDstSel)];
		CHECK(harness.transport->Plans() == 1);
		CHECK(harness.transport->Compiles() == (reserved ? 0u : 1u));
		CHECK(refused == (reserved ? 1u : 0u));
		CHECK(counters.variant_modules == (reserved ? 0u : 1u));
	}
	std::puts("InvalidDescriptorNeverSent ok");
}

// A walk that faults on its table skips the load, and the snapshot then gives no variant and sends
// no Compile. The baseline is the same page left readable, which gives one variant and one Compile.
// The mutation is to walk without non_faulting. The handler then completes the load as zero on
// this thread, as a GPU-thread walk would retry it, and the guess becomes a variant that is sent.
void TestSkippedSrtLoadSendsNothing() {
	const FakeGuestMemory memory;
	const auto            program = SrtProgram();
	for (const bool guarded: {false, true}) {
		ProtectedPage page;
		const auto    descriptor = Descriptor(STRIDE_A, DstSel(4, 5, 6, 7));
		std::copy(descriptor.begin(), descriptor.end(), page.Words());
		const std::array<uint32_t, POINTER_SGPRS> pointer {static_cast<uint32_t>(page.Address()),
		                                                   static_cast<uint32_t>(page.Address() >> 32u)};
		Harness harness;
		harness.Map(program);
		harness.transport->SetHoldCompiles(true);
		if (guarded) {
			page.Guard();
		}
		harness.Push(ItemFor(program, pointer, KEY));
		harness.Drive();
		const auto counters = harness.Counters();
		CHECK(harness.transport->Plans() == 1);
		CHECK(counters.srt_faults == (guarded ? 1u : 0u));
		CHECK(counters.variant_modules == (guarded ? 0u : 1u));
		CHECK(harness.transport->Compiles() == (guarded ? 0u : 1u));
	}
	std::puts("SkippedSrtLoadSendsNothing ok");
}

// What one scripted case of a test ended with.
struct CaseResult {
	Script                  script;
	PrescanPreparerCounters counters;
	size_t                  adopted = 0;
};

// Runs the script over the given items, one drive each, and keeps what it ended with.
CaseResult RunScripted(Script script, const Program& program,
                       const std::vector<PrescanItem>& items) {
	Harness harness(ScriptedFactory(script));
	harness.Map(program);
	for (const auto& item: items) {
		harness.Push(item);
		harness.Drive();
	}
	return {script, harness.Counters(), harness.adopted.size()};
}

// A blamed Plan retires the program, so a second dispatch of it sends nothing, and nothing is
// translated, specialized or adopted here instead. The baseline is a dropped Plan, which differs
// only in the outcome and is sent again for the second dispatch. A refusal for good at Submit
// retires the program as well. The mutations are to treat Blamed as Dropped, and the blamed
// program is sent again, and to answer a blamed Plan in this process instead, and the program is
// specialized and adopted.
void TestBlamedNeverFallsBack() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	const auto item    = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY);
	const std::vector<PrescanItem> twice {item, item};
	const auto dropped = RunScripted({.plan_outcome = StreamOutcome::Dropped}, program, twice);
	CHECK(dropped.script.plan_attempts == 2 && dropped.script.translated == 0);
	CHECK(dropped.counters.dropped == 2);

	for (const auto& script: {Script {.plan_outcome = StreamOutcome::Blamed},
	                          Script {.plan_submit = StreamSubmitStatus::NeverRetry}}) {
		const auto result = RunScripted(script, program, twice);
		CHECK(result.script.plan_attempts == 1 && result.script.compile_attempts == 0);
		CHECK(result.script.translated == 0 && result.counters.state.capped == 1);
		CHECK(result.adopted == 0 && result.counters.variant_modules == 0);
		CHECK(result.counters.blamed + result.counters.never_retry == 1);
	}
	std::puts("BlamedNeverFallsBack ok");
}

// A Compile blamed while a later dispatch of the program was already admitted retires that dispatch
// before it sends anything, since the blame is read in the same pass, after the admission. The
// passes are driven one at a time so the second dispatch is admitted before the blame is read. The
// baseline is the same flow with the Compile answered, where the second dispatch, another
// specialization, is sent. The mutation is to drop the blocked check when a snapshot is prepared.
// The second dispatch is then sent for a blamed program.
void TestBlamedCompileRetiresAdmittedPost() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	for (const bool blamed: {false, true}) {
		Script script {.compile_outcome = blamed ? StreamOutcome::Blamed : StreamOutcome::Answered};
		Harness harness(ScriptedFactory(script));
		harness.Map(program);
		harness.Push(ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY));
		(void)harness.preparer->RunOnce();
		(void)harness.preparer->RunOnce();
		CHECK(script.compile_attempts == 1);
		harness.Push(ItemFor(program, Descriptor(STRIDE_B, DstSel(4, 5, 6, 7)), KEY));
		harness.Drive();
		const auto counters = harness.Counters();
		CHECK(script.compile_attempts == (blamed ? 1u : 2u));
		CHECK(counters.blocked == (blamed ? 1u : 0u) && counters.blamed == (blamed ? 1u : 0u));
	}
	std::puts("BlamedCompileRetiresAdmittedPost ok");
}

// Repeats of a dispatch in one batch are all answered before the first is prepared, so the scanner
// sees Busy at once and does not fill the ring with more. The check reads the answer at the first
// send. The baseline is the answer before the pass, which is none. The mutation is to prepare each
// item as soon as it is admitted. The first send then comes before the repeats were answered.
void TestRepeatsAnsweredBeforePreparing() {
	const auto                              program = DirectProgram(MUBUF_STORE_DWORD);
	const auto                              user    = Descriptor(STRIDE_A, DstSel(4, 5, 6, 7));
	Script                                  script;
	std::optional<PrescanFeedback::Answer>  at_first_send;
	Harness                                 harness(ScriptedFactory(script));
	script.on_submit = [&](const StreamRequest&) {
		if (!at_first_send) {
			at_first_send = harness.handoff->feedback.Read(KEY);
		}
	};
	harness.Map(program);
	constexpr int repeats = 3;
	for (int index = 0; index < repeats; ++index) {
		harness.Push(ItemFor(program, user, KEY));
	}
	CHECK(harness.handoff->feedback.Read(KEY) == PrescanFeedback::Answer::None);
	harness.Drive();
	const auto counters = harness.Counters();
	CHECK(at_first_send == PrescanFeedback::Answer::Busy);
	CHECK(counters.items == repeats && counters.state.posted == 1 && counters.state.busy == repeats - 1);
	PrescanItem left;
	CHECK(!harness.handoff->ring.TryPop(left));
	std::puts("RepeatsAnsweredBeforePreparing ok");
}

// An error between Admit and an outcome gives the snapshot back, so the program can be posted
// again. The pass rethrows it to its caller. The baseline is the same item after the reader
// recovers, which must be posted again, not answered busy. The mutation is to leave the admission
// open when the preparation throws. The second item is then busy.
void TestErrorGivesTheSnapshotBack() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	const auto user    = Descriptor(STRIDE_A, DstSel(4, 5, 6, 7));
	Harness    harness;
	harness.Map(program);
	harness.Push(ItemFor(program, user, KEY));
	g_reader_throws = true;
	bool threw      = false;
	try {
		(void)harness.preparer->RunOnce();
	} catch (const std::runtime_error&) {
		threw = true;
	}
	g_reader_throws = false;
	CHECK(threw);
	harness.Push(ItemFor(program, user, KEY));
	harness.Drive();
	const auto counters = harness.Counters();
	CHECK(counters.state.posted == 2 && counters.state.busy == 0);
	CHECK(harness.adopted.size() == 1);
	std::puts("ErrorGivesTheSnapshotBack ok");
}

// On its own thread, an error ends the preparer cleanly. It reports the failure, takes no more
// items, and Stop returns at once. The baseline is a preparer whose lookup does not throw, which
// takes both items. The mutation is to remove the thread's try and catch. The error then ends the
// process. The hand-off's preparer_stopped is set by the failure and by Stop, and not before. The
// mutations are to drop either store, which fails the check before Stop or the one after it.
void TestErrorStopsThePrescan() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	const auto user    = Descriptor(STRIDE_A, DstSel(4, 5, 6, 7));
	for (const bool failing: {false, true}) {
		std::atomic<bool> throw_next {failing};
		Harness*          self = nullptr;
		Harness           harness({}, [&](uint64_t address) -> std::optional<ShaderMapSnapshot> {
            if (throw_next.exchange(false)) {
                throw std::runtime_error("test lookup failure");
            }
            const auto found = self->map.find(address);
            return found == self->map.end() ? std::nullopt : std::optional(found->second);
        });
		self = &harness;
		harness.Map(program);
		harness.preparer->Start();
		harness.Push(ItemFor(program, user, KEY));
		WakePreparer(*harness.handoff);
		CHECK(WaitFor([&] { return harness.Counters().items == 1; }));
		CHECK(harness.Counters().failed == failing);
		harness.Push(ItemFor(program, user, OTHER_KEY));
		WakePreparer(*harness.handoff);
		if (failing) {
			std::this_thread::sleep_for(SETTLE_TIME);
			CHECK(harness.Counters().items == 1);
		} else {
			CHECK(WaitFor([&] { return harness.Counters().items == 2; }));
		}
		// The scanner is told the prescan is off by the failure, and by Stop in either case.
		CHECK(harness.handoff->preparer_stopped.load() == failing);
		const auto start = std::chrono::steady_clock::now();
		harness.preparer->Stop();
		CHECK(std::chrono::steady_clock::now() - start < STOP_LIMIT);
		CHECK(harness.handoff->preparer_stopped.load());
	}
	std::puts("ErrorStopsThePrescan ok");
}

// A Compile answer whose static key words are not the preparer's is counted and not adopted. The
// baseline is the same answer uncorrupted, which is adopted. The mutation is to skip the check.
// The corrupt answer is then adopted.
void TestKeyMismatchCounted() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	const auto user    = Descriptor(STRIDE_A, DstSel(4, 5, 6, 7));
	for (const bool corrupt: {false, true}) {
		Script script {.corrupt_compile_key = corrupt};
		Harness harness(ScriptedFactory(script));
		harness.Map(program);
		harness.Push(ItemFor(program, user, KEY));
		harness.Drive();
		const auto counters = harness.Counters();
		CHECK(script.compile_attempts == 1);
		CHECK(counters.key_mismatch == (corrupt ? 1u : 0u));
		CHECK(harness.adopted.size() == (corrupt ? 0u : 1u));
	}
	std::puts("KeyMismatchCounted ok");
}

// After PRESCAN_UNTAKEN_LIMIT modules of a program were built and none taken, a further dispatch is
// answered Settled and sends nothing, while a probe still goes through. The baseline is the same
// dispatch after the GPU thread reported one module taken, which is prepared and sent. The
// mutation is to drop the back-off check. The dispatch past the limit is then sent.
void TestUntakenModulesBackOff() {
	const auto                    program = DirectProgram(MUBUF_STORE_DWORD);
	const std::array<uint32_t, PRESCAN_UNTAKEN_LIMIT> built_strides {4, 16, 32};
	constexpr uint32_t                                NEXT_STRIDE  = 64;
	constexpr uint32_t                                PROBE_STRIDE = 80;
	for (const bool one_taken: {true, false}) {
		Harness harness;
		harness.Map(program);
		for (uint32_t index = 0; index < PRESCAN_UNTAKEN_LIMIT; ++index) {
			harness.Push(ItemFor(program, Descriptor(built_strides[index], DstSel(4, 5, 6, 7)), KEY));
			harness.Drive();
		}
		CHECK(harness.adopted.size() == PRESCAN_UNTAKEN_LIMIT);
		if (one_taken) {
			harness.preparer->NoteTaken(harness.adopted[0].id);
		}
		harness.Push(ItemFor(program, Descriptor(NEXT_STRIDE, DstSel(4, 5, 6, 7)), OTHER_KEY));
		harness.Drive();
		CHECK(harness.Counters().backed_off == (one_taken ? 0u : 1u));
		CHECK(harness.transport->Compiles() == PRESCAN_UNTAKEN_LIMIT + (one_taken ? 1u : 0u));
		CHECK(harness.handoff->feedback.Read(OTHER_KEY) ==
		      (one_taken ? PrescanFeedback::Answer::None : PrescanFeedback::Answer::Settled));
		auto probe        = ItemFor(program, Descriptor(PROBE_STRIDE, DstSel(4, 5, 6, 7)), OTHER_KEY);
		probe.force_probe = true;
		harness.Push(probe);
		harness.Drive();
		CHECK(harness.transport->Compiles() == PRESCAN_UNTAKEN_LIMIT + (one_taken ? 2u : 1u));
	}
	std::puts("UntakenModulesBackOff ok");
}

// A module the drain refused as a late duplicate counts as used, since the guess was right and only
// the answer was late. Three duplicates reported leave the program free of the back-off, and the
// next dispatch is sent. The baseline is the same three modules with nothing reported, which backs
// the program off. The mutation is to ignore duplicate reports. The program is then backed off.
void TestLateDuplicatesAreUsed() {
	const auto                                        program = DirectProgram(MUBUF_STORE_DWORD);
	const std::array<uint32_t, PRESCAN_UNTAKEN_LIMIT> built_strides {4, 16, 32};
	constexpr uint32_t                                NEXT_STRIDE = 64;
	for (const bool reported: {false, true}) {
		Harness harness;
		harness.Map(program);
		for (const uint32_t stride: built_strides) {
			harness.Push(ItemFor(program, Descriptor(stride, DstSel(4, 5, 6, 7)), KEY));
			harness.Drive();
		}
		CHECK(harness.adopted.size() == PRESCAN_UNTAKEN_LIMIT);
		for (const auto& record: harness.adopted) {
			if (reported) {
				harness.preparer->NoteDuplicate(record.id);
			}
		}
		harness.Push(ItemFor(program, Descriptor(NEXT_STRIDE, DstSel(4, 5, 6, 7)), OTHER_KEY));
		harness.Drive();
		const auto counters = harness.Counters();
		CHECK(counters.late_duplicates == (reported ? PRESCAN_UNTAKEN_LIMIT : 0u));
		CHECK(counters.backed_off == (reported ? 0u : 1u));
		CHECK(harness.transport->Compiles() == PRESCAN_UNTAKEN_LIMIT + (reported ? 1u : 0u));
	}
	std::puts("LateDuplicatesAreUsed ok");
}

// A Compile refused for now, or dropped by the helper, is forgotten, so the same specialization
// from a later dispatch is sent again and built. The baseline is a Compile the helper failed,
// which differs only in the outcome and is not sent again, since the child would fail again. The
// mutations are to drop the Forget call in Send, and the refused one is not sent again, and to drop
// it on a dropped Compile, with the same result.
void TestTransientRefusalsForgotten() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	const auto item    = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY);
	const std::vector<PrescanItem> twice {item, item};
	const auto failed =
	    RunScripted({.compile_status = StreamStatus::Failed, .compile_scripted = 1}, program, twice);
	CHECK(failed.script.compile_attempts == 1 && failed.adopted == 0);
	CHECK(failed.counters.compile_failed == 1 && failed.counters.forgotten == 0);
	const Script refused {.compile_submit = StreamSubmitStatus::QueueFull, .compile_scripted = 1};
	const Script dropped {.compile_outcome = StreamOutcome::Dropped, .compile_scripted = 1};
	for (const auto& script: {refused, dropped}) {
		const auto result = RunScripted(script, program, twice);
		CHECK(result.script.compile_attempts == 2 && result.adopted == 1);
		CHECK(result.counters.forgotten == 1 && result.counters.variant_modules == 2);
	}
	std::puts("TransientRefusalsForgotten ok");
}

// A Compile refused for good blocks the program, so the next dispatch, another specialization, is
// retired and sends nothing. The baseline is a refusal for now, which differs only in the status and
// lets the next dispatch be sent. The mutation is to not block on the refusal. The next dispatch is
// then sent.
void TestCompileNeverRetryBlocks() {
	const auto                     program = DirectProgram(MUBUF_STORE_DWORD);
	const std::vector<PrescanItem> items {
	    ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY),
	    ItemFor(program, Descriptor(STRIDE_B, DstSel(4, 5, 6, 7)), KEY)};
	for (const auto status: {StreamSubmitStatus::QueueFull, StreamSubmitStatus::NeverRetry}) {
		const bool never = status == StreamSubmitStatus::NeverRetry;
		const auto result =
		    RunScripted({.compile_submit = status, .compile_scripted = 1}, program, items);
		CHECK(result.script.compile_attempts == (never ? 1u : 2u));
		CHECK(result.counters.never_retry == (never ? 1u : 0u));
		CHECK(result.counters.blocked == (never ? 1u : 0u));
		CHECK(result.adopted == (never ? 0u : 1u));
	}
	std::puts("CompileNeverRetryBlocks ok");
}

// A Plan answered with calls retires the program, so a second dispatch sends nothing. The baseline is
// a Plan the helper dropped, which differs only in the status and is asked for again. The mutation
// is to treat calls like a drop. The Plan is then sent again.
void TestPlanHasCallsRetires() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	const auto item    = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY);
	const std::vector<PrescanItem> twice {item, item};
	for (const auto status: {StreamStatus::Dropped, StreamStatus::HasCalls}) {
		const bool calls  = status == StreamStatus::HasCalls;
		const auto result = RunScripted({.plan_status = status}, program, twice);
		CHECK(result.script.plan_attempts == (calls ? 1u : 2u));
		CHECK(result.counters.has_calls == (calls ? 1u : 0u));
		CHECK(result.counters.state.capped == (calls ? 1u : 0u));
		CHECK(result.script.compile_attempts == 0);
	}
	std::puts("PlanHasCallsRetires ok");
}

// A Plan whose echoed static key is not the preparer's is counted and retires the program, so no
// Compile is sent. The baseline is the same Plan uncorrupted, which is used and leads to a Compile.
// The mutation is to skip the check on the Plan. Its plan is then used and a Compile sent.
void TestPlanKeyMismatchRetires() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	const auto item    = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY);
	const std::vector<PrescanItem> twice {item, item};
	for (const bool corrupt: {false, true}) {
		const auto result = RunScripted({.corrupt_plan_key = corrupt}, program, twice);
		CHECK(result.counters.key_mismatch == (corrupt ? 1u : 0u));
		CHECK(result.counters.state.capped == (corrupt ? 1u : 0u));
		CHECK(result.script.compile_attempts == (corrupt ? 0u : 1u));
		CHECK(result.adopted == (corrupt ? 0u : 1u));
		CHECK(result.script.plan_attempts == 1);
	}
	std::puts("PlanKeyMismatchRetires ok");
}

// Two units of one program, which differ only in their thread count, copy the program's code once,
// since the second shares the first one's copy. Both send a Plan, which carries the code, and both
// are built, so a second unit that sent nothing cannot pass. The baseline is the first unit alone,
// which reads the code once. The mutation is to drop the sharing in CopyCode. The second unit then
// reads the code again.
void TestUnitsShareProgramCode() {
	const auto program = DirectProgram(MUBUF_STORE_DWORD);
	Harness    harness;
	harness.Map(program);
	g_counted_address = AddressOf(program);
	g_counted_reads   = 0;
	const auto first  = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY);
	harness.Push(first);
	harness.Drive();
	CHECK(g_counted_reads == 1 && harness.transport->Plans() == 1);
	auto second                        = first;
	second.inputs.cs_regs.num_thread_x = OTHER_UNIT_THREADS;
	second.key_hash                    = OTHER_KEY;
	harness.Push(second);
	harness.Drive();
	CHECK(harness.transport->Plans() == 2 && harness.adopted.size() == 2);
	CHECK(g_counted_reads == 1);
	g_counted_address = 0;
	std::puts("UnitsShareProgramCode ok");
}

// A send refused because the helper's lane is off for the session turns the prescan off. Two
// programs and a repeat of the second are admitted in one pass, the repeat answered busy. The first
// program's Plan meets LaneOff, so the second is never sent and is given back, which takes back the
// repeat's busy answer. The preparer sets preparer_stopped for the scanner and admits nothing more,
// so a second round reaches neither the state nor the transport. The baseline is QueueFull, a
// refusal for now that differs only in the status, where every program of both rounds is admitted
// and sent. The mutations are to drop the TurnOff call in Send, which admits and sends like the
// baseline, to drop the lane-off check from the PrepareReady loop, which sends the second program,
// and to drop the give-back after the loop, which leaves the repeat busy.
void TestLaneOffTurnsPrescanOff() {
	constexpr uint32_t PROGRAMS_PER_ROUND = 2;
	constexpr uint32_t ROUNDS             = 2;
	constexpr uint32_t SENT_WHEN_OFF      = 1;
	const auto         first              = DirectProgram(MUBUF_STORE_DWORD);
	const auto         other              = DirectProgram(MUBUF_STORE_FORMAT_X);
	const auto         user               = Descriptor(STRIDE_A, DstSel(4, 5, 6, 7));
	for (const auto status: {StreamSubmitStatus::QueueFull, StreamSubmitStatus::LaneOff}) {
		const bool     off  = status == StreamSubmitStatus::LaneOff;
		const uint32_t sent = off ? SENT_WHEN_OFF : PROGRAMS_PER_ROUND * ROUNDS;
		Script         script {.plan_submit = status};
		Harness        harness(ScriptedFactory(script));
		harness.Map(first);
		harness.Map(other);
		harness.Push(ItemFor(first, user, KEY));
		harness.Push(ItemFor(other, user, OTHER_KEY));
		harness.Push(ItemFor(other, user, REPEAT_KEY));
		harness.Drive();
		harness.Push(ItemFor(first, user, KEY));
		harness.Push(ItemFor(other, user, OTHER_KEY));
		harness.Drive();
		const auto counters = harness.Counters();
		CHECK(script.plan_attempts == sent && counters.refused_sends == sent);
		CHECK(counters.state.posted == (off ? PROGRAMS_PER_ROUND : PROGRAMS_PER_ROUND * ROUNDS));
		CHECK(counters.state.busy == 1);
		CHECK(harness.handoff->feedback.Read(REPEAT_KEY) == PrescanFeedback::Answer::None);
		CHECK(counters.lane_off == off);
		CHECK(harness.handoff->preparer_stopped.load() == off);
	}
	std::puts("LaneOffTurnsPrescanOff ok");
}

// A Plan already sent when the lane turns off still gets its outcome. The supervisor marks its
// lane off before it hands over the Dropped outcomes of what it holds, so the preparer stops the
// transport, which delivers them, and takes them before it ends. Program A's Plan is held by the
// transport and A waits for it, with a repeat of A answered busy. Program B's Plan then meets
// LaneOff. A's Dropped outcome gives A back to the state, which closes its admission and takes
// back the repeat's busy answer. The baseline is B refused with QueueFull, which differs only in
// the status. There the prescan stays on, A still waits, and the repeat stays busy. The mutations
// are to drop the transport's Stop and to drop the drain after it. Either leaves the repeat busy.
void TestLaneOffSettlesParkedPlan() {
	// A's Plan is the one held, so both programs send a Plan, and only A's repeat is answered busy.
	constexpr uint32_t HELD_PLANS    = 1;
	constexpr uint64_t PLANS_SENT    = 2;
	constexpr uint64_t BUSY_ANSWERS  = 1;
	const auto         first         = DirectProgram(MUBUF_STORE_DWORD);
	const auto         other         = DirectProgram(MUBUF_STORE_FORMAT_X);
	const auto         user          = Descriptor(STRIDE_A, DstSel(4, 5, 6, 7));
	for (const auto status: {StreamSubmitStatus::QueueFull, StreamSubmitStatus::LaneOff}) {
		const bool off = status == StreamSubmitStatus::LaneOff;
		Script     script {.plan_submit = status, .plans_held = HELD_PLANS};
		Harness    harness(ScriptedFactory(script));
		harness.Map(first);
		harness.Map(other);
		harness.Push(ItemFor(first, user, KEY));
		harness.Push(ItemFor(first, user, REPEAT_KEY));
		harness.Push(ItemFor(other, user, OTHER_KEY));
		harness.Drive();
		const auto counters = harness.Counters();
		CHECK(script.plan_attempts == PLANS_SENT && counters.state.busy == BUSY_ANSWERS);
		CHECK(counters.lane_off == off && counters.dropped == (off ? 1u : 0u));
		CHECK(harness.handoff->feedback.Read(REPEAT_KEY) ==
		      (off ? PrescanFeedback::Answer::None : PrescanFeedback::Answer::Busy));
	}
	std::puts("LaneOffSettlesParkedPlan ok");
}

// A settle whose drain throws still gives back the snapshots the loop never reached. Program A's
// Compile is held by the transport, and every Plan after it meets LaneOff. Then B, C and a repeat of
// C, answered busy, are admitted in one pass. B's Plan turns the prescan off, so C is never
// prepared. The settle's Stop answers A's Compile and the drain adopts it, where the sink throws.
// The error reaches the pass's caller, and C is given back first, which takes back the repeat's
// busy answer. The baseline is the same run with a sink that does not throw, which adopts A and
// gives C back too. The mutation is to call the settle after PrepareReady's try. The throw then
// skips the give-back and the repeat stays busy.
void TestSettleThrowGivesBack() {
	const auto                                program_a = DirectProgram(MUBUF_STORE_DWORD);
	const auto                                program_b = DirectProgram(MUBUF_STORE_FORMAT_X);
	const auto                                program_c = SrtProgram();
	const auto                                user      = Descriptor(STRIDE_A, DstSel(4, 5, 6, 7));
	const std::array<uint32_t, POINTER_SGPRS> no_table {};
	for (const bool throwing: {false, true}) {
		Script script {.compiles_held = 1};
		// Every Plan sent after A's Compile meets LaneOff.
		script.on_submit = [&script](const StreamRequest& request) {
			if (request.kind == StreamRequestKind::Compile) {
				script.plan_submit = StreamSubmitStatus::LaneOff;
			}
		};
		Harness harness(ScriptedFactory(script));
		for (const auto* program: {&program_a, &program_b, &program_c}) {
			harness.Map(*program);
		}
		harness.Push(ItemFor(program_a, user, KEY));
		harness.Drive();
		CHECK(script.compile_attempts == 1);
		harness.Push(ItemFor(program_b, user, OTHER_KEY));
		harness.Push(ItemFor(program_c, no_table, THIRD_KEY));
		harness.Push(ItemFor(program_c, no_table, REPEAT_KEY));
		g_sink_throws = throwing;
		bool threw    = false;
		try {
			(void)harness.preparer->RunOnce();
		} catch (const std::runtime_error&) {
			threw = true;
		}
		g_sink_throws = false;
		CHECK(threw == throwing && harness.adopted.size() == (throwing ? 0u : 1u));
		// A pass that throws publishes no counters, so the busy answer is read in the baseline,
		// whose pass is the same up to the sink.
		CHECK(throwing || harness.Counters().state.busy == 1);
		CHECK(harness.handoff->feedback.Read(REPEAT_KEY) == PrescanFeedback::Answer::None);
	}
	std::puts("SettleThrowGivesBack ok");
}

// What a supervised run of three dispatches ended with.
struct SupervisedRun {
	PrescanPreparerCounters counters;
	StreamSupervisorStats   stats;
	std::vector<uint64_t>   ids;
	// The SPIR-V of the first module adopted, or empty.
	std::vector<uint32_t> first_spirv;
	bool                  in_time = true;
	// Whether the helper's progress log existed before the preparer's Stop and after it.
	bool log_before_stop = false;
	bool log_after_stop  = false;
};

// The three dispatches of the supervised test. B is another unit of A's program and C is A's unit
// with another specialization.
std::array<PrescanItem, SUPERVISED_DISPATCHES> SupervisedItems(const Program& program) {
	const auto a                  = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY);
	auto       b                  = a;
	b.inputs.cs_regs.num_thread_x = OTHER_UNIT_THREADS;
	b.key_hash                    = OTHER_KEY;
	const auto c                  = ItemFor(program, Descriptor(STRIDE_B, DstSel(4, 5, 6, 7)), KEY);
	return {a, b, c};
}

// Sends A, B and C through a fake helper in the given mode, waiting after each for what that mode
// makes of it. A helper that answers adopts each one. One that crashes on A's Compile blames it,
// refuses B's Plan for good at Submit, and has the preparer retire C before it sends anything. It
// notes whether the helper's progress log was there before Stop and after it.
SupervisedRun RunSupervised(const std::string& mode, const Program& program) {
	const bool crash = mode == "crash_compile";
	const std::array<std::function<bool(const PrescanPreparerCounters&)>, SUPERVISED_DISPATCHES>
	    crashed {[](const PrescanPreparerCounters& counters) { return counters.blamed == 1; },
	             [](const PrescanPreparerCounters& counters) { return counters.never_retry == 1; },
	             [](const PrescanPreparerCounters& counters) { return counters.blocked == 1; }};
	TempDirectory               directory;
	SupervisorPrescanTransport* transport = nullptr;
	SupervisedRun               run;
	Harness harness(SupervisorPrescanTransportFactory(FakeHelperConfig(directory, mode), &transport),
	                {}, SupervisedHost());
	harness.Map(program);
	const auto items = SupervisedItems(program);
	for (size_t index = 0; index < items.size(); ++index) {
		harness.Push(items[index]);
		run.in_time = DriveUntil(harness, [&] {
			              return crash ? crashed[index](harness.Counters())
			                           : harness.adopted.size() == index + 1;
		              }) && run.in_time;
	}
	run.counters = harness.Counters();
	run.stats    = transport->Stats();
	for (const auto& record: harness.adopted) {
		run.ids.push_back(record.id);
	}
	run.first_spirv     = harness.adopted.empty() ? std::vector<uint32_t> {} : harness.adopted[0].spirv;
	const auto progress = StreamSupervisorFilesOf(directory.Base()).progress;
	run.log_before_stop = std::filesystem::exists(progress);
	harness.preparer->Stop();
	run.log_after_stop = std::filesystem::exists(progress);
	return run;
}

// The module this process builds for one dispatch through the in-process transport, with or
// without the supervised profile flag, or an empty record when none was built.
DecodedModuleRecord ReferenceModule(const Program& program, const PrescanItem& item,
                                    bool profile_flag) {
	Harness reference({}, {}, SupervisedHost(profile_flag));
	reference.Map(program);
	reference.Push(item);
	reference.Drive();
	return reference.adopted.size() == 1 ? reference.adopted[0] : DecodedModuleRecord {};
}

// Through the real supervisor and the real Stream worker in a child. A helper that answers adopts
// A, B and C, five answers from one child, and A's module is the one this process builds for A,
// SPIR-V and all. The id covers the device, but not the profile, so the SPIR-V is compared too. The
// profile flag the host values set changes A's SPIR-V, as the baseline built without it shows, so a
// child that got other host values builds other SPIR-V. The helper's files are gone after Stop.
//
// The same dispatches with a child that crashes on A's Compile are the baseline that differs only
// in the crash. A is Blamed, its program blocked and nothing is adopted, so the blame never falls
// back to a build here. B's Plan is refused with NeverRetry at Submit, so no second child starts
// for it, and C is retired before any send. The helper's files are kept after a blame.
//
// The mutations are to pass the child twice the host subgroup size, which adopts nothing since
// every answer's key is not the preparer's, to encode only the shared memory limit in the
// child's header, which builds other SPIR-V, to map NeverRetry to QueueFull in the transport,
// which never counts B as refused for good, and to keep the files after every Stop.
void TestSupervisedHelperOutcomes() {
	const auto program  = DirectProgram(MUBUF_STORE_DWORD);
	const auto item     = SupervisedItems(program)[0];
	const auto expected = ReferenceModule(program, item, true);
	const auto plain    = ReferenceModule(program, item, false);
	CHECK(!expected.spirv.empty() && !plain.spirv.empty() && expected.spirv != plain.spirv);

	const auto answered = RunSupervised("answer", program);
	CHECK(answered.in_time && answered.ids.size() == SUPERVISED_DISPATCHES);
	CHECK(!answered.ids.empty() && answered.ids[0] == expected.id);
	CHECK(answered.first_spirv == expected.spirv);
	CHECK(answered.counters.blamed == 0 && answered.counters.never_retry == 0);
	CHECK(answered.counters.blocked == 0);
	CHECK(answered.stats.spawns == 1 && answered.stats.answered == SUPERVISED_ANSWERS);
	CHECK(answered.log_before_stop && !answered.log_after_stop);

	const auto crashed = RunSupervised("crash_compile", program);
	CHECK(crashed.in_time && crashed.ids.empty() && crashed.counters.adopted == 0);
	CHECK(crashed.counters.blamed == 1 && crashed.counters.never_retry == 1);
	CHECK(crashed.counters.blocked == 1);
	CHECK(crashed.counters.plan_requests == 1 && crashed.counters.compile_requests == 1);
	CHECK(crashed.stats.spawns == 1 && crashed.stats.refused == 1);
	CHECK(crashed.stats.never_retry_programs == 1);
	CHECK(crashed.log_before_stop && crashed.log_after_stop);
	std::puts("SupervisedHelperOutcomes ok");
}

// A helper that exits before it connects, with no restart allowed, turns the supervisor's lane off
// at once, and its Plan comes back Dropped. The next send is refused with LaneOff, which turns the
// prescan off, and a third dispatch reaches no Submit. One child is ever started. The mutation is to
// map LaneOff to QueueFull in the transport. The prescan then stays on, and the third dispatch is
// sent and refused too.
void TestSupervisedLaneOff() {
	const auto    program = DirectProgram(MUBUF_STORE_DWORD);
	const auto    item    = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY);
	TempDirectory directory;
	auto          config = FakeHelperConfig(directory, "exit_at_once");
	config.max_restarts  = 0;
	SupervisorPrescanTransport* transport = nullptr;
	Harness harness(SupervisorPrescanTransportFactory(std::move(config), &transport), {},
	                SupervisedHost());
	harness.Map(program);
	harness.Push(item);
	CHECK(DriveUntil(harness, [&] { return harness.Counters().dropped == 1; }));
	for (int repeat = 0; repeat < LANE_OFF_REPEATS; ++repeat) {
		harness.Push(item);
		harness.Drive();
	}
	const auto counters = harness.Counters();
	const auto stats    = transport->Stats();
	CHECK(counters.lane_off && harness.handoff->preparer_stopped.load());
	CHECK(counters.plan_requests == 1 && counters.refused_sends == 1);
	CHECK(stats.lane_off && stats.spawns == 1 && stats.refused == 1);
	harness.preparer->Stop();
	std::puts("SupervisedLaneOff ok");
}

// Whether a file holds the text. It reads through Common::File, so a writer may keep it open.
bool FileContains(const std::filesystem::path& path, std::string_view text) {
	Common::File file;
	if (!file.Open(path, Common::File::Mode::Read)) {
		return false;
	}
	const auto bytes = file.ReadWholeBuffer();
	return std::string_view(reinterpret_cast<const char*>(bytes.data()), bytes.size()).find(text) !=
	       std::string_view::npos;
}

// A Plan request for the program as the preparer would send it, with its code shared.
std::pair<StreamRequest, std::shared_ptr<const std::vector<uint8_t>>>
PlanRequestFor(const Program& program) {
	const auto    bytes = program.code.size() * sizeof(uint32_t);
	StreamRequest request;
	request.kind            = StreamRequestKind::Plan;
	request.code_hash       = XXH3_64bits(program.code.data(), bytes);
	request.dispatch        = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY).inputs;
	request.user_data_count = program.user_sgprs;
	const auto* first       = reinterpret_cast<const uint8_t*>(program.code.data());
	return {request, std::make_shared<const std::vector<uint8_t>>(first, first + bytes)};
}

// The supervised transport's Stop completes the request a hung child is running as Dropped, with
// its tag, before it returns. The test waits until the child was started and logged the request's
// start, so the request is the hung child's and not one still waiting in the supervisor. A second
// Stop adds nothing, and nor does a wait after it. The mutation is to make the transport's Stop do
// nothing. The request then has no outcome when Stop returns.
void TestSupervisedStopIsFinal() {
	const auto                    program = DirectProgram(MUBUF_STORE_DWORD);
	TempDirectory                 directory;
	std::mutex                    mutex;
	std::vector<StreamCompletion> outcomes;
	// The outcomes so far, read under the lock the handler takes.
	const auto count = [&] {
		std::lock_guard lock(mutex);
		return outcomes.size();
	};
	const auto keep = [&](StreamCompletion&& completion) {
		std::lock_guard lock(mutex);
		outcomes.push_back(std::move(completion));
	};
	SupervisorPrescanTransport transport(FakeHelperConfig(directory, "hang"), SupervisedHost(),
	                                     DEVICE, keep);
	const auto [request, code] = PlanRequestFor(program);
	CHECK(transport.Submit(request, code, STOP_TEST_TAG).status == StreamSubmitStatus::Queued);
	const auto progress = StreamSupervisorFilesOf(directory.Base()).progress;
	CHECK(WaitFor([&] { return transport.Stats().spawns == 1; }, HELPER_WAIT_LIMIT));
	CHECK(WaitFor([&] { return FileContains(progress, "start 1 "); }, HELPER_WAIT_LIMIT));
	CHECK(count() == 0);
	transport.Stop();
	CHECK(count() == 1 && outcomes[0].outcome == StreamOutcome::Dropped &&
	      outcomes[0].tag == STOP_TEST_TAG);
	transport.Stop();
	std::this_thread::sleep_for(QUIET_TIME);
	CHECK(count() == 1);
	std::puts("SupervisedStopIsFinal ok");
}

// A device whose shared memory limit is not the host values' is refused when the transport is
// made, since the child would take its limit from the host values and build other modules. The
// baseline is every other supervised test, which makes the transport with the two equal. The
// mutation is to drop the check, and the transport is then made.
void TestMismatchedDeviceRefused() {
	TempDirectory          directory;
	const ComputeKeyDevice smaller {HOST_SUBGROUP_SIZE, MAX_SHARED_BYTES / 2};
	bool                   refused = false;
	try {
		SupervisorPrescanTransport transport(FakeHelperConfig(directory, "answer"), SupervisedHost(),
		                                     smaller, [](StreamCompletion&&) {});
	} catch (const std::invalid_argument&) {
		refused = true;
	}
	CHECK(refused);
	std::puts("MismatchedDeviceRefused ok");
}

// Only whole helper file names give a process id. The other names are a prefix of one, another
// suffix, no digits, or a sign, which a looser parse would take. The helper's directory is its own
// `kyty_prescan` directory and exists, so the start-up cleanup scans only the helper's files and
// never the whole temporary directory. A directory function that gave the temporary directory
// itself, as the first version did, fails the name check.
void TestHelperFileNames() {
	const auto directory = PrescanHelperDirectory();
	CHECK(directory && directory->filename() == "kyty_prescan" &&
	      std::filesystem::is_directory(*directory));
	constexpr uint32_t PID = 4321;
	CHECK(PrescanHelperFilePid("kyty_prescan_4321.request") == PID);
	CHECK(PrescanHelperFilePid("kyty_prescan_4321.progress") == PID);
	CHECK(PrescanHelperFilePid("kyty_prescan_4321.progress.out") == PID);
	CHECK(!PrescanHelperFilePid("kyty_prescan_4321"));
	CHECK(!PrescanHelperFilePid("kyty_prescan_4321.progress.old"));
	CHECK(!PrescanHelperFilePid("kyty_prescan_.request"));
	CHECK(!PrescanHelperFilePid("kyty_prescan_-4.request"));
	CHECK(!PrescanHelperFilePid("other_4321.request"));
	CHECK(PrescanHelperFilesBase("dir", PID).filename() == "kyty_prescan_4321");
	std::puts("HelperFileNames ok");
}

// The helper files of a dead run are removed at start, while those of this run, of a run still
// going, and a file of another name stay. The fake process check calls only DEAD_PID dead, so the
// test needs no real dead process. The files sit in a `kyty_prescan` folder as the real ones do,
// and a dead run's file in the folder above stays, since only the helper's own folder is scanned.
// The mutation is to drop the process check, and the live run's file then goes too.
void TestStaleHelperFilesRemoved() {
	constexpr uint32_t CURRENT_PID = 100;
	constexpr uint32_t LIVE_PID    = 200;
	constexpr uint32_t DEAD_PID    = 300;
	constexpr uint32_t DEAD_FILES  = 2;
	TempDirectory      directory;
	const auto         above  = directory.Base().parent_path();
	const auto         folder = above / "kyty_prescan";
	std::filesystem::create_directories(folder);
	// Makes the file of a run with this pid and suffix in a folder and returns its path.
	const auto touch = [](const std::filesystem::path& where, uint32_t pid, const std::string& suffix) {
		auto path = std::filesystem::path(PrescanHelperFilesBase(where, pid)).concat(suffix);
		std::ofstream(path, std::ios::binary).put('1');
		return path;
	};
	const auto dead_request  = touch(folder, DEAD_PID, ".request");
	const auto dead_output   = touch(folder, DEAD_PID, ".progress.out");
	const auto live_progress = touch(folder, LIVE_PID, ".progress");
	const auto own_progress  = touch(folder, CURRENT_PID, ".progress");
	const auto unrelated     = touch(folder, DEAD_PID, ".keep");
	const auto outside       = touch(above, DEAD_PID, ".request");
	const auto cleaned       = RemoveStalePrescanFiles(
        folder, CURRENT_PID, [&](uint32_t pid) { return pid != DEAD_PID; }, {});
	CHECK(cleaned.removed == DEAD_FILES && cleaned.failed == 0);
	CHECK(!std::filesystem::exists(dead_request) && !std::filesystem::exists(dead_output));
	CHECK(std::filesystem::exists(live_progress) && std::filesystem::exists(own_progress));
	CHECK(std::filesystem::exists(unrelated) && std::filesystem::exists(outside));
	std::puts("StaleHelperFilesRemoved ok");
}

#ifdef KYTY_EMULATOR_PATH
// The real helper, the emulator binary in its worker mode, answers a program's Plan and Compile
// through the supervisor, and the module it built is the one this process builds for the same
// dispatch, id and SPIR-V. A child given another device builds another id, one given another
// profile builds other SPIR-V, as TestSupervisedHelperOutcomes shows for this flag, and one that
// never answered adopts nothing, so only the whole real path passes.
void TestRealHelperBuildsSameModule() {
	const auto             program  = DirectProgram(MUBUF_STORE_DWORD);
	const auto             item     = ItemFor(program, Descriptor(STRIDE_A, DstSel(4, 5, 6, 7)), KEY);
	const auto             expected = ReferenceModule(program, item, true);
	TempDirectory          directory;
	StreamSupervisorConfig config;
	config.files_base                     = directory.Base();
	config.program                        = PathFromUtf8(KYTY_EMULATOR_PATH);
	SupervisorPrescanTransport* transport = nullptr;
	Harness harness(SupervisorPrescanTransportFactory(std::move(config), &transport), {},
	                SupervisedHost());
	harness.Map(program);
	harness.Push(item);
	CHECK(DriveUntil(harness, [&] { return harness.adopted.size() == 1; }));
	CHECK(!expected.spirv.empty() && harness.adopted.size() == 1 &&
	      harness.adopted[0].id == expected.id && harness.adopted[0].spirv == expected.spirv);
	CHECK(transport->Stats().spawns == 1 && transport->Stats().crashes == 0);
	harness.preparer->Stop();
	std::puts("RealHelperBuildsSameModule ok");
}
#endif

} // namespace

int main(int argc, char** argv) {
	if (argc >= 4 && std::string_view(argv[1]) == FAKE_FLAG) {
		return RunFakeHelper(argv[2], PathFromUtf8(argv[3]));
	}
	// A hang fails the run instead of holding ctest until its timeout.
	std::thread([] {
		std::this_thread::sleep_for(TEST_TIMEOUT);
		std::fprintf(stderr, "PrescanPreparerTests: the watchdog ran out\n");
		std::_Exit(1);
	}).detach();
	TestSecondVariantSkipsPlan();
	TestInvalidDescriptorNeverSent();
	TestSkippedSrtLoadSendsNothing();
	TestBlamedNeverFallsBack();
	TestRepeatsAnsweredBeforePreparing();
	TestErrorGivesTheSnapshotBack();
	TestErrorStopsThePrescan();
	TestKeyMismatchCounted();
	TestUntakenModulesBackOff();
	TestBlamedCompileRetiresAdmittedPost();
	TestLateDuplicatesAreUsed();
	TestTransientRefusalsForgotten();
	TestCompileNeverRetryBlocks();
	TestPlanHasCallsRetires();
	TestPlanKeyMismatchRetires();
	TestUnitsShareProgramCode();
	TestLaneOffTurnsPrescanOff();
	TestLaneOffSettlesParkedPlan();
	TestSettleThrowGivesBack();
	TestSupervisedHelperOutcomes();
	TestSupervisedLaneOff();
	TestSupervisedStopIsFinal();
	TestMismatchedDeviceRefused();
	TestHelperFileNames();
	TestStaleHelperFilesRemoved();
#ifdef KYTY_EMULATOR_PATH
	TestRealHelperBuildsSameModule();
#endif
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("PrescanPreparerTests passed");
	return 0;
}
