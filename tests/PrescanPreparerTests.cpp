// Tests the prescan's preparer over real programs, translated in this process. The in-process
// transport answers with the Stream worker's own code, so a Plan and a Compile here are the child's.
// A scripted transport wraps the same answers when a test needs an outcome a healthy translation
// never gives, such as a blamed request or a corrupted static key. Every check that proves a
// behaviour has a baseline that differs only in that behaviour, as the comments say.
#include "SrtWalkerTestMemory.h"
#include "graphics/host_gpu/renderer/pipeline/prescanPreparer.h"
#include "graphics/shader/storegen/prescanTransport.h"
#include "graphics/shader/storegen/streamWorker.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
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

// Reads host memory as guest memory, since the test programs live in host memory.
bool ReadHost(void* /*userdata*/, uint64_t address, std::span<uint32_t> values) {
	if (g_reader_throws.load()) {
		throw std::runtime_error("test reader failure");
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
	void Stop() override {}

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

	Script&                  m_script;
	PrescanCompletionHandler m_handler;
	uint64_t                 m_next_seq      = 1;
	uint32_t                 m_compiles_seen = 0;
};

PrescanTransportFactory ScriptedFactory(Script& script) {
	return [&script](const HostValues&, const ComputeKeyDevice&, PrescanCompletionHandler handler)
	           -> std::unique_ptr<PrescanTransport> {
		return std::make_unique<ScriptedTransport>(script, std::move(handler));
	};
}

// ---- The preparer under test. ----

// A preparer over a shader map of test programs, host memory and a sink that keeps what it gets.
// The hand-off is on the heap, since its ring is large.
struct Harness {
	explicit Harness(PrescanTransportFactory factory = {}, PrescanMapLookup lookup = {}) {
		PrescanPreparerConfig config;
		config.lookup = std::move(lookup);
		if (!config.lookup) {
			config.lookup = [this](uint64_t address) {
				const auto found = map.find(address);
				return found == map.end() ? std::nullopt : std::optional(found->second);
			};
		}
		config.read_memory    = ReadHost;
		config.device         = DEVICE;
		config.make_transport = factory ? std::move(factory) : InProcessPrescanTransportFactory(&transport);
		config.sink           = [this](DecodedModuleRecord&& record) { adopted.push_back(std::move(record)); };
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

// Waits for a condition, polling, up to the wait limit.
bool WaitFor(const std::function<bool()>& condition) {
	const auto until = std::chrono::steady_clock::now() + WAIT_LIMIT;
	while (!condition()) {
		if (std::chrono::steady_clock::now() >= until) {
			return false;
		}
		std::this_thread::sleep_for(POLL_SLICE);
	}
	return true;
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
// process.
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
		const auto start = std::chrono::steady_clock::now();
		harness.preparer->Stop();
		CHECK(std::chrono::steady_clock::now() - start < STOP_LIMIT);
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

} // namespace

int main() {
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
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("PrescanPreparerTests passed");
	return 0;
}
