// Tests the Stream mode of the shader store worker, which is the loop that answers Plan and Compile
// requests through a pipe, its cache of translated programs, and its progress lines. The worker
// runs on a thread of this process and talks to the test through a real pipe, and it translates the
// tiny programs of the synthetic eboot for real, so no game file and no device are needed. Each
// check names what else could satisfy it and compares against a baseline that differs only in the
// behaviour under test. The emulator binary in Stream mode is tested in ShaderStoreDispatchTests.
#include "ShaderStoreEbootFixtures.h"
#include "common/childProcess.h"
#include "common/file.h"
#include "graphics/guest_gpu/csRegisters.h"
#include "graphics/shader/moduleId.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/computeTranslation.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/streamChannel.h"
#include "graphics/shader/storegen/streamTranslationCache.h"
#include "graphics/shader/storegen/streamWorker.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <xxhash.h>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;
using namespace StoreGenFixtures;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

/// How long the whole run may take before the watchdog fails it, so a read that never returns
/// fails the run rather than hanging it.
constexpr std::chrono::seconds TEST_TIMEOUT {280};
/// How long the test waits for the worker to connect, and for one answer. The programs are tiny,
/// so the answer limit is for a loaded runner.
constexpr std::chrono::milliseconds ACCEPT_TIMEOUT {10000};
constexpr std::chrono::milliseconds REPLY_TIMEOUT {60000};
/// How long the refused reply test waits for the Failed answer. A worker that sent nothing would
/// otherwise make the test wait the full answer limit.
constexpr std::chrono::milliseconds REFUSED_REPLY_TIMEOUT {10000};
/// How long the first translation of the concurrency test holds on for the second request.
constexpr std::chrono::milliseconds HOLD_LIMIT {10000};
/// The seq the progress test watches.
constexpr uint64_t WATCHED_SEQ = 7;
/// The class level cache test's cap, and the size each entry claims, so two fit and three do not.
constexpr uint64_t SMALL_CAP_BYTES   = 100;
constexpr uint64_t SMALL_ENTRY_BYTES = 40;
/// A byte that fills the oversize payload, any value will do.
constexpr uint8_t OVERSIZE_FILL = 0xAB;
/// How often a test looks at the log while it waits for a line.
constexpr std::chrono::milliseconds POLL_SLICE {10};
/// How long a worker with a missing pipe may take to give up, well under its connect timeout.
constexpr std::chrono::milliseconds PROMPT_LIMIT {2000};
/// A flag byte that is neither 0 nor 1.
constexpr uint8_t NOT_A_FLAG = 2;
/// How long the queue test waits for the read count to stop changing.
constexpr std::chrono::milliseconds SETTLE_TIME {300};
/// The queue test's queue limit, and how many requests it sends, well past the limit.
constexpr uint32_t SMALL_QUEUE         = 4;
constexpr uint64_t QUEUE_TEST_REQUESTS = 10;

const ComputeKeyDevice DEVICE {HOST_SUBGROUP_SIZE, MAX_SHARED_BYTES};

// ---- Files. ----

// A directory for the files of one test, which is removed with it.
class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		m_path             = fs::temp_directory_path() /
		         ("kyty_stream_worker_tests_" + std::to_string(Common::CurrentProcessId()) + "_" +
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

// Reads a whole file through Common::File, which opens it so the worker can still write to it.
Bytes ReadWholeFile(const fs::path& path) {
	Common::File file;
	if (!file.Open(path, Common::File::Mode::Read)) {
		return {};
	}
	const auto bytes = file.ReadWholeBuffer();
	return {reinterpret_cast<const uint8_t*>(bytes.data()),
	        reinterpret_cast<const uint8_t*>(bytes.data()) + bytes.size()};
}

void WriteWholeFile(const fs::path& path, const Bytes& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()),
	          static_cast<std::streamsize>(bytes.size()));
}

// The lines of a text file.
std::vector<std::string> ReadLines(const fs::path& path) {
	const auto               bytes = ReadWholeFile(path);
	std::istringstream       in(std::string(bytes.begin(), bytes.end()));
	std::vector<std::string> lines;
	for (std::string line; std::getline(in, line);) {
		lines.push_back(line);
	}
	return lines;
}

// Where a line is in a log, or the log's size when it is not there.
size_t IndexOf(const std::vector<std::string>& lines, const std::string& line) {
	return static_cast<size_t>(std::ranges::find(lines, line) - lines.begin());
}

// Whether any line of a log starts with the text.
bool AnyStartsWith(const std::vector<std::string>& lines, const std::string& text) {
	return std::ranges::any_of(lines, [&](const std::string& line) { return line.starts_with(text); });
}

// ---- Requests and programs. ----

// The StoreHeader bytes, written as PipelineCache writes them.
Bytes MakeStoreHeader() {
	IR::Writer                       writer;
	uint64_t                         stamp       = BUILD_STAMP;
	auto                             jump_tables = false;
	ShaderRecompiler::Spirv::Profile profile;
	uint32_t                         shared = MAX_SHARED_BYTES;
	IR::IoAll(writer, stamp, jump_tables, profile, shared);
	return std::move(writer.bytes);
}

// The host values the worker decodes from that header.
HostValues Host() {
	const auto host = DecodeHostHeader(MakeStoreHeader());
	CHECK(host.has_value());
	return host.value_or(HostValues {});
}

// A fixture program as the pre-scan would send it, with its registers as the header sets them,
// and its static key built the generator's way, from the header, to compare the worker with.
struct StreamProgram {
	ScannedProgram        scanned;
	PrescanDispatchInputs dispatch {};
	ComputeStaticKey      key;
};

// The compute registers a header's SH register table sets, as the command processor would hold
// them for a Motor dispatch of the program.
HW::CsStageRegisters RegistersOf(const Bytes& header) {
	HW::CsStageRegisters regs;
	regs.wave_size = static_cast<uint8_t>(MOTOR_DISPATCH.wave_size);
	uint64_t relative = 0;
	uint8_t  count    = 0;
	CHECK(Layout::Read(header, Layout::OFFSET_SH_REGISTERS, relative) &&
	      Layout::Read(header, Layout::OFFSET_NUM_SH_REGISTERS, count));
	for (uint64_t index = 0; index < count; index++) {
		const uint64_t at = Layout::OFFSET_SH_REGISTERS + relative + index * Layout::SH_REGISTER_SIZE;
		uint32_t       register_offset = 0;
		uint32_t       value           = 0;
		CHECK(Layout::Read(header, at, register_offset) &&
		      Layout::Read(header, at + sizeof(uint32_t), value));
		(void)DecodeCsRegister(regs, register_offset, value);
	}
	return regs;
}

// Builds one program's eboot, scans it, and reads its registers and key from the header.
StreamProgram MakeStreamProgram(const ProgramSpec& spec) {
	StreamProgram program;
	const auto    scan = ScanEbootShaders(BuildEboot({spec}));
	CHECK(scan.programs.size() == 1);
	if (scan.programs.empty()) {
		return program;
	}
	program.scanned                   = scan.programs[0];
	program.dispatch.cs_regs          = RegistersOf(program.scanned.header);
	program.dispatch.workgroup_counts = {1, 1, 1};
	const auto key = BuildComputeStaticKey(program.scanned.header, DEVICE, MOTOR_DISPATCH);
	CHECK(key.has_value());
	program.key = key.value_or(ComputeStaticKey {});
	return program;
}

// A program with no buffer and no user data, whose module has no push data at all.
ProgramSpec ProgramWithoutResources() {
	ProgramSpec spec;
	spec.store = false;
	return spec;
}

// A Plan request for a program.
StreamRequest PlanRequest(const StreamProgram& program, uint64_t seq) {
	StreamRequest request;
	request.seq       = seq;
	request.kind      = StreamRequestKind::Plan;
	request.code_hash = program.scanned.code_hash;
	const auto* code  = reinterpret_cast<const uint8_t*>(program.scanned.code.data());
	request.code.assign(code, code + program.scanned.code.size() * sizeof(uint32_t));
	request.dispatch            = program.dispatch;
	request.user_data_count     = program.key.user_data_count;
	request.scratch_size_dwords = SCRATCH_DWORDS;
	return request;
}

// A Compile request for a program with a specialization.
StreamRequest CompileRequest(const StreamProgram& program, uint64_t seq,
                             const IR::ResourceSpecialization& specialization) {
	auto request           = PlanRequest(program, seq);
	request.kind           = StreamRequestKind::Compile;
	request.specialization = EncodeSpecialization(specialization);
	return request;
}

// The candidates the predictor gives a program, from the facts its spec says the code has.
std::vector<Candidate> CandidatesOf(const ProgramSpec& spec, const StreamProgram& program) {
	const auto slots   = DecodeHeaderSlots(program.scanned.header);
	const auto records = DecodeSl00Resources(program.scanned.metadata);
	CHECK(slots && records);
	if (!slots || !records) {
		return {};
	}
	const auto facts = FactsOf(spec);
	return PredictCandidates({program.scanned.code_hash, program.key.user_data_count,
	                          program.key.state, facts, *slots, *records, NullPolicy::Subsets})
	    .candidates;
}

// A module record read back from its bytes.
std::optional<DecodedModuleRecord> DecodeRecord(const Bytes& bytes) {
	IR::Reader reader {bytes};
	auto       type = StoreRecord::GraphicsPipeline;
	IR::IoAll(reader, type);
	DecodedModuleRecord record;
	DecodeModuleRecord(reader, record);
	if (type != StoreRecord::Module || !reader.ok || !reader.bytes.empty()) {
		return std::nullopt;
	}
	return record;
}

// ---- A worker on a thread. ----

// A Stream worker running on a thread of this process, connected to a listener the test holds.
// Finish, or the destructor, closes the test's end and joins the worker. A worker made with a raw
// end gives the test the bare pipe, so it can write bytes that are not a frame.
class InProcessWorker {
public:
	InProcessWorker(const TempDirectory& directory, StreamWorkerOptions options = {},
	                std::optional<uint32_t> parent_pid = std::nullopt, bool raw_end = false) {
		auto listener = StreamPipeListener::Create("worker-test");
		CHECK(listener.has_value());
		m_request.store_header       = MakeStoreHeader();
		m_request.host_subgroup_size = HOST_SUBGROUP_SIZE;
		m_request.progress_path      = Utf8FromPath(directory / "stream.progress");
		m_request.parent_pid         = parent_pid.value_or(Common::CurrentProcessId());
		m_request.mode               = WorkerMode::Stream;
		m_request.stream_pipe        = listener ? listener->Name() : std::string("missing");
		m_worker = std::make_unique<StreamWorker>(m_request, std::move(options));
		m_thread = std::thread([this] { m_exit_code = m_worker->Run(); });
		if (listener) {
			if (auto pipe = listener->Accept(ACCEPT_TIMEOUT, Common::CurrentProcessId())) {
				if (raw_end) {
					m_raw.emplace(std::move(*pipe));
				} else {
					m_channel.emplace(std::move(*pipe));
				}
			}
		}
	}
	~InProcessWorker() { (void)Finish(); }
	InProcessWorker(const InProcessWorker&)            = delete;
	InProcessWorker& operator=(const InProcessWorker&) = delete;

	[[nodiscard]] bool Connected() const { return m_channel.has_value() || m_raw.has_value(); }

	// Writes bytes on a raw end, as they are.
	bool WriteRaw(std::span<const uint8_t> bytes) { return m_raw && m_raw->WriteAll(bytes); }

	bool Send(const StreamRequest& request) { return m_channel && m_channel->Send(request); }

	// Reads one answer, or nothing when none came in time or the worker closed the pipe. A read
	// that times out interrupts the test's end, so no thread is left blocked on it.
	std::optional<StreamResponse> Read(std::chrono::milliseconds timeout) {
		if (!m_channel) {
			return std::nullopt;
		}
		std::promise<std::optional<StreamResponse>> done;
		auto                                        answer = done.get_future();
		std::thread reader([&] {
			const auto read = m_channel->ReadFrame();
			done.set_value(read.status == StreamReadStatus::Frame ? DecodeStreamResponse(read.frame)
			                                                      : std::nullopt);
		});
		if (answer.wait_for(timeout) != std::future_status::ready) {
			m_channel->Interrupt();
		}
		reader.join();
		return answer.get();
	}

	// Closes the test's end, which the worker reads as the parent going away, and waits for it.
	int Finish() {
		if (m_channel) {
			m_channel->Close();
			m_channel.reset();
		}
		m_raw.reset();
		if (m_thread.joinable()) {
			m_thread.join();
		}
		return m_exit_code;
	}

	[[nodiscard]] StreamWorkerStats Stats() const { return m_worker->Stats(); }
	[[nodiscard]] fs::path ProgressPath() const { return PathFromUtf8(m_request.progress_path); }
	[[nodiscard]] std::vector<std::string> Log() const { return ReadLines(ProgressPath()); }

private:
	WorkerRequest                 m_request;
	std::unique_ptr<StreamWorker> m_worker;
	std::thread                   m_thread;
	std::optional<StreamChannel>  m_channel;
	std::optional<StreamPipe>     m_raw;
	int                           m_exit_code = -1;
};

// Sends one request and reads its answer.
std::optional<StreamResponse> RoundTrip(InProcessWorker& worker, const StreamRequest& request) {
	CHECK(worker.Send(request));
	return worker.Read(REPLY_TIMEOUT);
}

// Whether an answer is the Ok answer of a request.
bool IsOkAnswer(const std::optional<StreamResponse>& answer, const StreamRequest& request) {
	return answer && answer->seq == request.seq && answer->kind == request.kind &&
	       answer->status == StreamStatus::Ok && !answer->payload.empty();
}

// ---- Plan and Compile. ----

// The plan bytes the runtime's path gives a program, made here with options set by hand, zeroed
// user data and the input info of the generator's header key, so none of it is the worker's code.
Bytes InProcessPlan(const StreamProgram& program) {
	const auto                       host = Host();
	ShaderComputeInputInfo           info = program.key.input_info;
	std::vector<uint32_t>            user_data(program.key.user_data_count, 0u);
	ShaderRecompiler::CompileOptions options;
	options.wave_size              = info.wave_size;
	options.stage                  = ShaderType::Compute;
	options.input_info.compute     = &info;
	options.shader_hash            = program.scanned.code_hash;
	options.dump_ir                = false;
	options.structured_jump_tables = host.structured_jump_tables;
	options.user_data              = user_data;
	options.profile                = &host.profile;
	const auto source     = ShaderRecompiler::PrepareShaderSource(program.scanned.code, options);
	auto       translated = ShaderRecompiler::TranslateProgram(source.decoded, options);
	return IR::SaveResourcePlan(IR::ExtractResourcePlan(translated.program));
}

// A Plan answer holds the bytes an in-process ExtractResourcePlan gives the same program, and the
// static key it echoes is the generator's key from the header, though the worker built it from
// the registers alone. Both would agree if both were empty, so the plan must load and list a
// buffer. A worker that answered every Plan with the same bytes would pass too, so the worker's
// answer for the big program must differ from its answer for the small one, and match the big
// program's own in-process plan.
void TestStreamPlanMatchesInProcess() {
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	const auto small   = MakeStreamProgram(ProgramSmall());
	const auto request = PlanRequest(small, 1);
	const auto answer  = RoundTrip(worker, request);
	CHECK(IsOkAnswer(answer, request));
	if (!answer) {
		return;
	}
	CHECK(answer->static_state == small.key.state);
	const auto expected = InProcessPlan(small);
	CHECK(answer->payload == expected);
	IR::ResourcePlan loaded;
	CHECK(IR::LoadResourcePlan(answer->payload, loaded) && !loaded.info.buffers.empty());
	// The first of the main programs is the big one, which has other buffers than the small one.
	const auto big         = MakeStreamProgram(MainPrograms().front());
	const auto big_request = PlanRequest(big, 2);
	const auto big_answer  = RoundTrip(worker, big_request);
	CHECK(IsOkAnswer(big_answer, big_request));
	CHECK(big_answer && big_answer->payload != answer->payload);
	CHECK(big_answer && big_answer->payload == InProcessPlan(big));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// The record the Generate worker writes for a one-program eboot's first candidate.
Bytes GeneratedRecord(const TempDirectory& directory, const ProgramSpec& spec) {
	const auto eboot = directory / "generate.eboot";
	WriteWholeFile(eboot, BuildEboot({spec}));
	WorkerRequest request;
	request.store_header       = MakeStoreHeader();
	request.host_subgroup_size = HOST_SUBGROUP_SIZE;
	request.eboot_path         = Utf8FromPath(eboot);
	request.output_path        = Utf8FromPath(directory / "generate.out");
	request.progress_path      = Utf8FromPath(directory / "generate.progress");
	request.generated_path     = Utf8FromPath(directory / "generate.generated");
	request.parent_pid         = PARENT_PID;
	const auto result          = RunShaderStoreWorker(request);
	CHECK(result.exit_code == WORKER_EXIT_DONE && result.records_written == 1);
	const auto file   = ReadWholeFile(PathFromUtf8(request.output_path));
	const auto header = request.store_header.size() + GENERATED_HEADER_TAIL_BYTES;
	const auto frame  = file.size() > header
	                        ? UnframeStoreRecord(std::span<const uint8_t>(file).subspan(header))
	                        : std::nullopt;
	CHECK(frame.has_value());
	return frame ? frame->raw : Bytes {};
}

// A Compile answer's record has the id the predictor gives the candidate, which the generator's
// tests hold to the runtime, and it is byte for byte the record the Generate worker writes for the
// same candidate. Both programs could share a wrong push start, since a program with buffers starts
// at 0, so a program without resources is compiled too. Its module has no push data, and its id
// must be the one that start gives, which differs from the id a start of 0 gives.
void TestStreamCompileIdMatchesRuntime() {
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	const auto small      = MakeStreamProgram(ProgramSmall());
	const auto candidates = CandidatesOf(ProgramSmall(), small);
	CHECK(candidates.size() == 1);
	if (candidates.empty()) {
		return;
	}
	// The same id the generator's tests expect for the candidate, so both tests hold one value.
	CHECK(Pick(PredictedFor(ProgramSmall(), small.scanned), {}).id == candidates[0].id);
	const auto request = CompileRequest(small, 1, candidates[0].specialization);
	const auto answer  = RoundTrip(worker, request);
	CHECK(IsOkAnswer(answer, request));
	const auto record = answer ? DecodeRecord(answer->payload) : std::nullopt;
	CHECK(record && record->id == candidates[0].id && record->static_state == small.key.state);
	CHECK(answer && answer->payload == GeneratedRecord(directory, ProgramSmall()));

	const auto                       bare = MakeStreamProgram(ProgramWithoutResources());
	const IR::ResourceSpecialization none;
	const auto bare_request = CompileRequest(bare, 2, none);
	const auto bare_answer  = RoundTrip(worker, bare_request);
	CHECK(IsOkAnswer(bare_answer, bare_request));
	const auto bare_record = bare_answer ? DecodeRecord(bare_answer->payload) : std::nullopt;
	const auto rule_id = ShaderModuleId(ShaderType::Compute, bare.scanned.code_hash,
	                                    bare.key.user_data_count, bare.key.state,
	                                    IR::PushData::NoStart, none, {});
	CHECK(bare_record && bare_record->info.bindings.push_data_start_dword == IR::PushData::NoStart);
	CHECK(bare_record && bare_record->id == rule_id);
	CHECK(rule_id != ShaderModuleId(ShaderType::Compute, bare.scanned.code_hash,
	                                bare.key.user_data_count, bare.key.state, 0, none, {}));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// A program with GPU-selected calls is answered HasCalls, for a Plan and for a Compile, with no
// payload and with its static key, and the log never says such a request was answered ok. A worker
// that answered everything HasCalls would pass that, so the same program without the call is
// answered Ok, and the log says so for it.
void TestHasCallsRefused() {
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	auto calls_spec  = ProgramSmall();
	calls_spec.calls = true;
	const auto calls = MakeStreamProgram(calls_spec);
	const auto plain = MakeStreamProgram(ProgramSmall());
	const auto spec  = CandidatesOf(ProgramSmall(), plain);
	CHECK(!spec.empty());
	for (const auto& request:
	     {PlanRequest(calls, 1),
	      CompileRequest(calls, 2, spec.empty() ? IR::ResourceSpecialization {} : spec[0].specialization)}) {
		const auto answer = RoundTrip(worker, request);
		CHECK(answer && answer->seq == request.seq && answer->status == StreamStatus::HasCalls);
		CHECK(answer && answer->payload.empty() && answer->static_state == calls.key.state);
	}
	const auto plain_request = PlanRequest(plain, 3);
	CHECK(IsOkAnswer(RoundTrip(worker, plain_request), plain_request));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
	const auto log = worker.Log();
	CHECK(IndexOf(log, "done 1 calls") < log.size() && IndexOf(log, "done 2 calls") < log.size());
	CHECK(!AnyStartsWith(log, "done 1 ok") && !AnyStartsWith(log, "done 2 ok"));
	CHECK(IndexOf(log, "done 3 ok") < log.size());
	CHECK(worker.Stats().has_calls == 2);
}

// ---- The cache. ----

// A Plan and then a Compile of one program translate it once, and the Compile is a cache hit. A
// counter that never counted would pass that, so a second program's Plan must translate again and
// be a miss. A cache that handed the Compile some other program's IR would pass the counts, so
// the record compiled from the cached IR must be byte for byte the Generate worker's record.
void TestPlanThenCompileTranslatesOnce() {
	TempDirectory       directory;
	std::atomic<int>    translations {0};
	StreamWorkerOptions options;
	options.on_translate = [&](const TranslationKey&) { translations++; };
	InProcessWorker worker(directory, std::move(options));
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	const auto spec  = CandidatesOf(ProgramSmall(), small);
	CHECK(!spec.empty());
	if (spec.empty()) {
		return;
	}
	CHECK(IsOkAnswer(RoundTrip(worker, PlanRequest(small, 1)), PlanRequest(small, 1)));
	const auto compile = CompileRequest(small, 2, spec[0].specialization);
	const auto answer  = RoundTrip(worker, compile);
	CHECK(IsOkAnswer(answer, compile));
	CHECK(answer && answer->payload == GeneratedRecord(directory, ProgramSmall()));
	CHECK(translations == 1);
	const auto after_compile = worker.Stats().cache;
	CHECK(after_compile.hits == 1 && after_compile.plan_misses == 1 &&
	      after_compile.compile_misses == 0);
	const auto big = MakeStreamProgram(ProgramBig());
	CHECK(IsOkAnswer(RoundTrip(worker, PlanRequest(big, 3)), PlanRequest(big, 3)));
	CHECK(translations == 2);
	CHECK(worker.Stats().cache.plan_misses == 2);
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// A Plan leaves the program in the cache and its Compile takes it out. Counting the eviction alone
// would pass the counts, so the entry's bytes must be gone too, and a second Compile of the program
// must translate it again.
void TestEvictedAfterCompile() {
	TempDirectory       directory;
	std::atomic<int>    translations {0};
	StreamWorkerOptions options;
	options.on_translate = [&](const TranslationKey&) { translations++; };
	InProcessWorker worker(directory, std::move(options));
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	const auto spec  = CandidatesOf(ProgramSmall(), small);
	CHECK(!spec.empty());
	if (spec.empty()) {
		return;
	}
	CHECK(IsOkAnswer(RoundTrip(worker, PlanRequest(small, 1)), PlanRequest(small, 1)));
	const auto after_plan = worker.Stats().cache;
	CHECK(after_plan.entries == 1 && after_plan.bytes > 0 && after_plan.compile_evictions == 0);
	const auto compile = CompileRequest(small, 2, spec[0].specialization);
	CHECK(IsOkAnswer(RoundTrip(worker, compile), compile));
	const auto after_compile = worker.Stats().cache;
	CHECK(after_compile.entries == 0 && after_compile.bytes == 0);
	CHECK(after_compile.compile_evictions == 1);
	const auto again = CompileRequest(small, 3, spec[0].specialization);
	CHECK(IsOkAnswer(RoundTrip(worker, again), again));
	CHECK(translations == 2);
	// The second Compile's translation is counted as a Compile miss, the cost the cache did not save.
	CHECK(worker.Stats().cache.compile_misses == 1);
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// What the concurrency test's hooks share.
struct ConcurrencyProbe {
	std::mutex              mutex;
	std::condition_variable changed;
	int                     translations = 0;
	bool                    waited       = false;
};

// Two Plans of one program in flight at once translate it once, and the second is a hit. The first
// translation holds on until the second request is either waiting for the program or translating
// it too, so the test does not depend on timing. Without the guard the second translates and the
// count is two. The two answers must also carry equal plans.
void TestConcurrentSameKeyTranslatesOnce() {
	TempDirectory       directory;
	ConcurrencyProbe    probe;
	StreamWorkerOptions options;
	options.on_cache_wait = [&] {
		std::lock_guard lock(probe.mutex);
		probe.waited = true;
		probe.changed.notify_all();
	};
	options.on_translate = [&](const TranslationKey&) {
		std::unique_lock lock(probe.mutex);
		const int        mine = ++probe.translations;
		probe.changed.notify_all();
		if (mine == 1) {
			probe.changed.wait_for(lock, HOLD_LIMIT,
			                       [&] { return probe.waited || probe.translations >= 2; });
		}
	};
	InProcessWorker worker(directory, std::move(options));
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	CHECK(worker.Send(PlanRequest(small, 1)) && worker.Send(PlanRequest(small, 2)));
	const auto first  = worker.Read(REPLY_TIMEOUT);
	const auto second = worker.Read(REPLY_TIMEOUT);
	CHECK(first && second && first->status == StreamStatus::Ok && second->status == StreamStatus::Ok);
	CHECK(first && second && first->seq != second->seq && first->payload == second->payload);
	{
		std::lock_guard lock(probe.mutex);
		CHECK(probe.translations == 1 && probe.waited);
	}
	const auto cache = worker.Stats().cache;
	CHECK(cache.hits == 1 && cache.plan_misses == 1 && cache.waits == 1);
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// A key of the class level cache test.
TranslationKey KeyOf(uint64_t code_hash) {
	return {code_hash, USER_SGPR_COUNT, {1, 2, 3}};
}

// Puts an empty translation of the claimed size in the cache under a key.
void Insert(TranslationCache& cache, uint64_t code_hash) {
	auto lease = cache.Acquire(KeyOf(code_hash), TranslationUse::Plan);
	lease.Put(ShaderRecompiler::TranslateResult {}, SMALL_ENTRY_BYTES);
}

// Whether the cache holds a key. Looking makes the entry the most recently used.
bool Holds(TranslationCache& cache, uint64_t code_hash) {
	return cache.Acquire(KeyOf(code_hash), TranslationUse::Plan).Hit();
}

// Over the cap the least recently used entry goes. In insertion order that is the first one put
// in. A cache that dropped the newest, or the first one put in whatever was used since, would pass
// that, so a second cache uses its first entry again before the third goes in, and then the second
// entry is the one that goes.
void TestCapEvictsOldest() {
	TranslationCache inserted(SMALL_CAP_BYTES);
	Insert(inserted, 1);
	Insert(inserted, 2);
	Insert(inserted, 3);
	const auto stats = inserted.Stats();
	CHECK(stats.entries == 2 && stats.bytes == 2 * SMALL_ENTRY_BYTES && stats.cap_evictions == 1);
	CHECK(stats.peak_bytes == 3 * SMALL_ENTRY_BYTES);
	CHECK(!Holds(inserted, 1));
	CHECK(Holds(inserted, 2) && Holds(inserted, 3));

	TranslationCache used(SMALL_CAP_BYTES);
	Insert(used, 1);
	Insert(used, 2);
	CHECK(Holds(used, 1));
	Insert(used, 3);
	CHECK(!Holds(used, 2));
	CHECK(Holds(used, 1) && Holds(used, 3));
}

// ---- Answers, the log and the peer. ----

// An answer too large for a frame goes out as Failed with no payload and the request's seq and
// static key, so the parent is not left waiting. A worker that failed every request would pass, so
// the next Plan of the same program, which the hook leaves alone, must be answered Ok. A worker
// that sent nothing makes the first read time out, which fails rather than hangs.
void TestOversizeReplySentAsFailed() {
	TempDirectory       directory;
	StreamWorkerOptions options;
	options.before_send = [](StreamResponse& response) {
		if (response.seq == 1) {
			response.payload.assign(size_t {MAX_STREAM_PAYLOAD_BYTES} + 1, OVERSIZE_FILL);
		}
	};
	InProcessWorker worker(directory, std::move(options));
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	CHECK(worker.Send(PlanRequest(small, 1)));
	const auto refused = worker.Read(REFUSED_REPLY_TIMEOUT);
	CHECK(refused && refused->seq == 1 && refused->status == StreamStatus::Failed);
	CHECK(refused && refused->payload.empty() && refused->static_state == small.key.state);
	if (refused) {
		const auto request = PlanRequest(small, 2);
		CHECK(IsOkAnswer(RoundTrip(worker, request), request));
	}
	CHECK(worker.Stats().refused_replies == 1);
	worker.Finish();
	CHECK(IndexOf(worker.Log(), "done 1 failed") < worker.Log().size());
}

// A request's `start` line is in the log before its work begins and its `done` line only after it.
// The hook reads the log at the start of the work, where the start line must be and the done line
// must not, and the finished log has the start line before the done line. The log opens with the
// priority line, ends with the summary, and its start line is one the supervisor's parser reads,
// with the seq where a rank would be and the code hash.
void TestProgressLinesBracketWork() {
	TempDirectory            directory;
	std::vector<std::string> at_start;
	fs::path                 log_path;
	StreamWorkerOptions      options;
	options.before_work = [&](uint64_t seq) {
		if (seq == WATCHED_SEQ) {
			at_start = ReadLines(log_path);
		}
	};
	log_path = directory / "stream.progress";
	InProcessWorker worker(directory, std::move(options));
	CHECK(worker.Connected() && worker.ProgressPath() == log_path);
	const auto small   = MakeStreamProgram(ProgramSmall());
	const auto request = PlanRequest(small, WATCHED_SEQ);
	CHECK(IsOkAnswer(RoundTrip(worker, request), request));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
	const auto start = fmt::format("start {} {:016x}", WATCHED_SEQ, small.scanned.code_hash);
	const auto done  = fmt::format("done {} ok", WATCHED_SEQ);
	CHECK(IndexOf(at_start, start) < at_start.size());
	CHECK(!AnyStartsWith(at_start, fmt::format("done {} ", WATCHED_SEQ)));
	const auto log = worker.Log();
	CHECK(IndexOf(log, start) < IndexOf(log, done) && IndexOf(log, done) < log.size());
	CHECK(!log.empty() && log.front().starts_with("priority "));
	CHECK(log.back().starts_with("summary exit=0 "));
	const auto parsed = ParseProgressLine(start);
	CHECK(parsed && parsed->kind == ProgressKind::Start && parsed->rank_or_count == WATCHED_SEQ &&
	      parsed->hash == small.scanned.code_hash);
}

// A worker told that its parent is another process than the pipe's server serves nothing and ends
// with WORKER_EXIT_PEER_MISMATCH, and its log has no start line. A worker that refused every pipe
// would pass that, so the same worker told this process's id answers a Plan.
void TestPeerMismatchRefused() {
	const auto small   = MakeStreamProgram(ProgramSmall());
	const auto request = PlanRequest(small, 1);
	{
		TempDirectory   directory;
		InProcessWorker worker(directory, {}, Common::CurrentProcessId() + 1);
		if (worker.Connected()) {
			(void)worker.Send(request);
			CHECK(!worker.Read(REFUSED_REPLY_TIMEOUT).has_value());
		}
		CHECK(worker.Finish() == WORKER_EXIT_PEER_MISMATCH);
		CHECK(!AnyStartsWith(worker.Log(), "start "));
		CHECK(worker.Stats().requests == 0);
	}
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	CHECK(IsOkAnswer(RoundTrip(worker, request), request));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// ---- Failures and untrusted input. ----

// Whether an answer is the Failed answer of a request, with no payload and the given static key.
bool IsFailedAnswer(const std::optional<StreamResponse>& answer, const StreamRequest& request,
                    const std::vector<uint32_t>& static_state) {
	return answer && answer->seq == request.seq && answer->status == StreamStatus::Failed &&
	       answer->payload.empty() && answer->static_state == static_state;
}

// Whether a line that starts with the text reaches the log within the answer limit. The worker
// writes its lines straight to the file, so this only waits for it to get there.
bool WaitForLine(const InProcessWorker& worker, const std::string& text) {
	const auto deadline = std::chrono::steady_clock::now() + REPLY_TIMEOUT;
	while (std::chrono::steady_clock::now() < deadline) {
		if (AnyStartsWith(worker.Log(), text)) {
			return true;
		}
		std::this_thread::sleep_for(POLL_SLICE);
	}
	return false;
}

// A Plan whose code hash is not its code's is answered Failed with no static key, since the worker
// stops before it builds one. A worker that failed every request would pass, so the unchanged
// request must be answered Ok.
void TestWrongCodeHashFailed() {
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	auto       wrong = PlanRequest(small, 1);
	wrong.code_hash ^= 1u;
	CHECK(IsFailedAnswer(RoundTrip(worker, wrong), wrong, {}));
	const auto right = PlanRequest(small, 2);
	CHECK(IsOkAnswer(RoundTrip(worker, right), right));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// A request whose user data count is not its registers' user SGPR count is answered Failed. The
// runtime takes the count from the registers, so the worker refuses the mismatch rather than pick
// one. The baseline is the same request with the counts equal, which is answered Ok.
void TestUserDataCountMismatchFailed() {
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	auto       wrong = PlanRequest(small, 1);
	wrong.user_data_count++;
	CHECK(IsFailedAnswer(RoundTrip(worker, wrong), wrong, {}));
	const auto right = PlanRequest(small, 2);
	CHECK(right.user_data_count == right.dispatch.cs_regs.user_sgpr);
	CHECK(IsOkAnswer(RoundTrip(worker, right), right));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// Writes one frame on a raw end, waits for the worker to finish with it, and returns the exit code.
// A frame that ends the run makes the worker write its summary by itself, and a good one makes it
// log the request done, so the test's end is closed only after the worker has acted on the frame.
int RunOneRawFrame(const Bytes& frame, const std::string& wait_for) {
	TempDirectory   directory;
	InProcessWorker worker(directory, {}, std::nullopt, true);
	CHECK(worker.Connected());
	CHECK(worker.WriteRaw(frame));
	CHECK(WaitForLine(worker, wait_for));
	return worker.Finish();
}

// A frame with a bad magic, or a frame of the response kind, ends the run with
// WORKER_EXIT_PROTOCOL_ERROR, since the stream is out of step. A worker that ended every run that
// way would pass, so the same request frame unchanged is answered and the run ends with DONE.
void TestBadFramesEndTheRun() {
	const auto small = MakeStreamProgram(ProgramSmall());
	Bytes      good;
	CHECK(EncodeStreamRequest(PlanRequest(small, 1), good));
	auto bad_magic = good;
	bad_magic[0] ^= 0xFFu;
	CHECK(RunOneRawFrame(bad_magic, "summary ") == WORKER_EXIT_PROTOCOL_ERROR);
	Bytes response;
	CHECK(EncodeStreamResponse(StreamResponse {.seq = 1}, response));
	CHECK(RunOneRawFrame(response, "summary ") == WORKER_EXIT_PROTOCOL_ERROR);
	CHECK(RunOneRawFrame(good, "done 1 ok") == WORKER_EXIT_DONE);
}

// A Compile whose specialization has one byte too many is answered Failed. The baseline is the same
// bytes without it, which are answered Ok.
void TestTrailingSpecializationByteFailed() {
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	const auto spec  = CandidatesOf(ProgramSmall(), small);
	CHECK(!spec.empty());
	if (spec.empty()) {
		return;
	}
	auto longer = CompileRequest(small, 1, spec[0].specialization);
	longer.specialization.push_back(0);
	CHECK(IsFailedAnswer(RoundTrip(worker, longer), longer, small.key.state));
	const auto exact = CompileRequest(small, 2, spec[0].specialization);
	CHECK(IsOkAnswer(RoundTrip(worker, exact), exact));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// A Compile whose specialization lists fewer resources than the program is answered Failed, where
// ApplyResourceSpecialization would have ended the process. The baseline is the predicted
// specialization of the same program, which is answered Ok.
void TestResourceCountMismatchFailed() {
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	const auto spec  = CandidatesOf(ProgramSmall(), small);
	CHECK(!spec.empty() && !spec[0].specialization.buffers.empty());
	if (spec.empty()) {
		return;
	}
	const auto short_of = CompileRequest(small, 1, IR::ResourceSpecialization {});
	CHECK(IsFailedAnswer(RoundTrip(worker, short_of), short_of, small.key.state));
	const auto whole = CompileRequest(small, 2, spec[0].specialization);
	CHECK(IsOkAnswer(RoundTrip(worker, whole), whole));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// A worker given a pipe name that no listener has gives up at once with WORKER_EXIT_PIPE_FAILED,
// rather than after its connect timeout, since a missing pipe is not worth another try. The time
// limit is well under the timeout, so a worker that retried every failure would miss it.
void TestMissingPipeExitsPromptly() {
	TempDirectory directory;
	auto          gone = StreamPipeListener::Create("worker-gone");
	CHECK(gone.has_value());
	WorkerRequest request;
	request.store_header       = MakeStoreHeader();
	request.host_subgroup_size = HOST_SUBGROUP_SIZE;
	request.progress_path      = Utf8FromPath(directory / "gone.progress");
	request.parent_pid         = Common::CurrentProcessId();
	request.mode               = WorkerMode::Stream;
	request.stream_pipe        = gone ? gone->Name() : std::string("missing");
	gone.reset();
	const auto start  = std::chrono::steady_clock::now();
	const auto result = RunStreamWorker(request);
	CHECK(result.exit_code == WORKER_EXIT_PIPE_FAILED);
	CHECK(std::chrono::steady_clock::now() - start < PROMPT_LIMIT);
	CHECK(PROMPT_LIMIT < STREAM_CONNECT_TIMEOUT);
}

// Code padded past MAX_STREAM_CODE_BYTES is answered Failed. The padding follows the program's
// end, so it is never run, and the baseline is the same program padded to exactly the limit,
// which is answered Ok, so the size alone is what is refused.
void TestOversizeCodeFailed() {
	TempDirectory   directory;
	InProcessWorker worker(directory);
	CHECK(worker.Connected());
	const auto small  = MakeStreamProgram(ProgramSmall());
	const auto padded = [&](uint64_t seq, size_t bytes) {
		auto request = PlanRequest(small, seq);
		request.code.resize(bytes, 0);
		request.code_hash = XXH3_64bits(request.code.data(), request.code.size());
		return request;
	};
	const auto over = padded(1, MAX_STREAM_CODE_BYTES + sizeof(uint32_t));
	CHECK(IsFailedAnswer(RoundTrip(worker, over), over, {}));
	const auto at_limit = padded(2, MAX_STREAM_CODE_BYTES);
	CHECK(IsOkAnswer(RoundTrip(worker, at_limit), at_limit));
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// A specialization encoded past the worker's limit is answered Failed. A valid specialization as
// large as the default limit is hard to build, since extra entries must be indirect children, so
// the limit is set one byte under the predicted specialization of a real program. The baseline is
// the same request to a worker whose limit is exactly its size, which is answered Ok, so the size
// alone is what is refused. The default limit is the named constant.
void TestOversizeSpecializationFailed() {
	CHECK(StreamWorkerOptions {}.max_specialization_bytes == MAX_STREAM_SPECIALIZATION_BYTES);
	const auto small = MakeStreamProgram(ProgramSmall());
	const auto spec  = CandidatesOf(ProgramSmall(), small);
	CHECK(!spec.empty());
	if (spec.empty()) {
		return;
	}
	const auto request = CompileRequest(small, 1, spec[0].specialization);
	CHECK(request.specialization.size() > 1);
	for (const bool fits: {false, true}) {
		TempDirectory       directory;
		StreamWorkerOptions options;
		options.max_specialization_bytes = request.specialization.size() - (fits ? 0 : 1);
		InProcessWorker worker(directory, std::move(options));
		CHECK(worker.Connected());
		const auto answer = RoundTrip(worker, request);
		CHECK(fits ? IsOkAnswer(answer, request) : IsFailedAnswer(answer, request, small.key.state));
		CHECK(worker.Finish() == WORKER_EXIT_DONE);
	}
}

// The index of the one byte two encodings differ in, or nothing when they differ in another way.
std::optional<size_t> OnlyDifference(const Bytes& first, const Bytes& second) {
	if (first.size() != second.size()) {
		return std::nullopt;
	}
	std::optional<size_t> found;
	for (size_t index = 0; index < first.size(); index++) {
		if (first[index] != second[index]) {
			if (found) {
				return std::nullopt;
			}
			found = index;
		}
	}
	return found;
}

// A flag byte of a specialization that is neither 0 nor 1 is refused, for a buffer flag and an
// image flag. The byte is found as the one byte a flag set and cleared changes, and the baselines
// are those two encodings, which decode to the flag's two values.
void TestSpecializationFlagMustBeBoolean() {
	IR::ResourceSpecialization set;
	set.buffers.resize(1);
	set.images.resize(1);
	set.buffers[0].zero_stride_oob = true;
	set.images[0].heap             = true;
	for (const bool image: {false, true}) {
		auto cleared = set;
		(image ? cleared.images[0].heap : cleared.buffers[0].zero_stride_oob) = false;
		const auto set_bytes     = EncodeSpecialization(set);
		const auto cleared_bytes = EncodeSpecialization(cleared);
		const auto flag          = OnlyDifference(set_bytes, cleared_bytes);
		CHECK(flag.has_value());
		CHECK(DecodeSpecialization(set_bytes) == set);
		CHECK(DecodeSpecialization(cleared_bytes) == cleared);
		if (flag) {
			auto other  = set_bytes;
			other[*flag] = NOT_A_FLAG;
			CHECK(!DecodeSpecialization(other).has_value());
		}
	}
}

// What the queue test's hook shares, a gate the one translation thread waits at.
struct QueueGate {
	std::mutex              mutex;
	std::condition_variable opened;
	bool                    open = false;
};

// The number of requests the worker has queued, once it has stopped changing for a settle time.
// The count is of requests that went into the queue, not of frames read, so a reader waiting for
// room with one more frame in hand does not count it. It first waits for one request, so a worker
// that has not read anything yet is not mistaken for one that has settled.
uint64_t SettledRequests(const InProcessWorker& worker) {
	const auto deadline = std::chrono::steady_clock::now() + REPLY_TIMEOUT;
	while (worker.Stats().requests < 1 && std::chrono::steady_clock::now() < deadline) {
		std::this_thread::sleep_for(POLL_SLICE);
	}
	uint64_t seen = worker.Stats().requests;
	while (std::chrono::steady_clock::now() < deadline) {
		std::this_thread::sleep_for(SETTLE_TIME);
		const uint64_t now = worker.Stats().requests;
		if (now == seen) {
			return now;
		}
		seen = now;
	}
	return seen;
}

// With its one translation thread held, the worker reads one request for the thread and the queue's
// worth more, then stops reading, so the rest stay in the pipe. A worker that never read would
// pass the count alone, so once the thread is let go every request must be answered Ok.
void TestQueueCapHoldsTheReader() {
	TempDirectory       directory;
	QueueGate           gate;
	StreamWorkerOptions options;
	options.translation_threads = 1;
	options.max_queued_jobs     = SMALL_QUEUE;
	options.before_work         = [&](uint64_t) {
		std::unique_lock lock(gate.mutex);
		gate.opened.wait_for(lock, HOLD_LIMIT, [&] { return gate.open; });
	};
	InProcessWorker worker(directory, std::move(options));
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	for (uint64_t seq = 1; seq <= QUEUE_TEST_REQUESTS; seq++) {
		CHECK(worker.Send(PlanRequest(small, seq)));
	}
	CHECK(SettledRequests(worker) == SMALL_QUEUE + 1);
	{
		std::lock_guard lock(gate.mutex);
		gate.open = true;
	}
	gate.opened.notify_all();
	for (uint64_t answered = 0; answered < QUEUE_TEST_REQUESTS; answered++) {
		const auto answer = worker.Read(REPLY_TIMEOUT);
		CHECK(answer && answer->status == StreamStatus::Ok);
	}
	CHECK(worker.Stats().requests == QUEUE_TEST_REQUESTS);
	CHECK(worker.Finish() == WORKER_EXIT_DONE);
}

// A bad frame that arrives while a request is in work ends the run with WORKER_EXIT_PROTOCOL_ERROR,
// even though the busy thread's answer then fails to send because stopping interrupted the pipe. A
// worker that counted that send as a failed write would end with WORKER_EXIT_PIPE_FAILED instead.
// The first request is held at the start of its work and a second waits in the queue, so the
// second's `done 2 dropped` line shows that the reader gave up and the worker began to stop before
// the first is let go. The baseline is TestBadFramesEndTheRun, where no request was in work.
void TestBadFrameWhileBusyKeepsProtocolError() {
	TempDirectory       directory;
	QueueGate           gate;
	StreamWorkerOptions options;
	options.translation_threads = 1;
	options.before_work         = [&](uint64_t) {
		std::unique_lock lock(gate.mutex);
		gate.opened.wait_for(lock, HOLD_LIMIT, [&] { return gate.open; });
	};
	InProcessWorker worker(directory, std::move(options), std::nullopt, true);
	CHECK(worker.Connected());
	const auto small = MakeStreamProgram(ProgramSmall());
	Bytes      first;
	Bytes      second;
	CHECK(EncodeStreamRequest(PlanRequest(small, 1), first));
	CHECK(EncodeStreamRequest(PlanRequest(small, 2), second));
	CHECK(worker.WriteRaw(first));
	CHECK(WaitForLine(worker, "start 1 "));
	CHECK(worker.WriteRaw(second));
	auto bad_magic = first;
	bad_magic[0] ^= 0xFFu;
	CHECK(worker.WriteRaw(bad_magic));
	CHECK(WaitForLine(worker, "done 2 dropped"));
	{
		std::lock_guard lock(gate.mutex);
		gate.open = true;
	}
	gate.opened.notify_all();
	CHECK(WaitForLine(worker, "summary "));
	CHECK(worker.Finish() == WORKER_EXIT_PROTOCOL_ERROR);
}

// Fails the run when the tests take far longer than they should, since a read that never returns
// would otherwise hang it.
void StartWatchdog() {
	std::thread([] {
		std::this_thread::sleep_for(TEST_TIMEOUT);
		std::fprintf(stderr, "shader_store_stream_worker_tests: timed out\n");
		std::_Exit(1);
	}).detach();
}

} // namespace

int main() {
	StartWatchdog();
	TestStreamPlanMatchesInProcess();
	TestStreamCompileIdMatchesRuntime();
	TestHasCallsRefused();
	TestPlanThenCompileTranslatesOnce();
	TestEvictedAfterCompile();
	TestConcurrentSameKeyTranslatesOnce();
	TestCapEvictsOldest();
	TestOversizeReplySentAsFailed();
	TestProgressLinesBracketWork();
	TestPeerMismatchRefused();
	TestWrongCodeHashFailed();
	TestUserDataCountMismatchFailed();
	TestBadFramesEndTheRun();
	TestTrailingSpecializationByteFailed();
	TestResourceCountMismatchFailed();
	TestMissingPipeExitsPromptly();
	TestOversizeCodeFailed();
	TestOversizeSpecializationFailed();
	TestSpecializationFlagMustBeBoolean();
	TestQueueCapHoldsTheReader();
	TestBadFrameWhileBusyKeepsProtocolError();
	if (g_failures != 0) {
		std::fprintf(stderr, "shader_store_stream_worker_tests: %d failures\n", g_failures);
		return 1;
	}
	std::printf("shader_store_stream_worker_tests: all passed\n");
	return 0;
}
