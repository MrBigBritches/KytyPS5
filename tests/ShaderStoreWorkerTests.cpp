// Tests the shader store worker, which turns a game's eboot into store records without a graphics
// device. The eboot is built here from a few tiny compute programs, which the worker translates for
// real, so no game file is needed. Each check names what else could satisfy it and compares
// against a baseline that differs only in the behaviour under test.
#include "ShaderStoreEbootFixtures.h"
#include "common/childProcess.h"
#include "common/file.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/moduleId.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/candidateOrder.h"
#include "graphics/shader/storegen/computePredictor.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/ebootShaderScan.h"
#include "graphics/shader/storegen/headerSlots.h"
#include "graphics/shader/storegen/progressLog.h"
#include "graphics/shader/storegen/shaderHeaderLayout.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/sl00Metadata.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
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

// ---- Files. ----

// A directory for the files of one test, which is removed with it.
class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		m_path             = fs::temp_directory_path() /
		         ("kyty_store_worker_tests_" + std::to_string(Common::CurrentProcessId()) + "_" +
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

// Reads a whole file through Common::File, which opens it so that the worker can still write to it.
// A std::ifstream does not, on Windows, while the worker holds the file.
Bytes ReadFile(const fs::path& path) {
	Common::File file;
	if (!file.Open(path, Common::File::Mode::Read)) {
		return {};
	}
	const auto bytes = file.ReadWholeBuffer();
	return {reinterpret_cast<const uint8_t*>(bytes.data()),
	        reinterpret_cast<const uint8_t*>(bytes.data()) + bytes.size()};
}

// Writes a whole file, replacing it.
void WriteFile(const fs::path& path, const Bytes& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()),
	          static_cast<std::streamsize>(bytes.size()));
}

// The lines of a text file.
std::vector<std::string> ReadLines(const fs::path& path) {
	const auto               bytes = ReadFile(path);
	std::istringstream       in(std::string(bytes.begin(), bytes.end()));
	std::vector<std::string> lines;
	for (std::string line; std::getline(in, line);) {
		lines.push_back(line);
	}
	return lines;
}

// The words of a line.
std::vector<std::string> Words(const std::string& line) {
	std::istringstream       stream(line);
	std::vector<std::string> words;
	for (std::string word; stream >> word;) {
		words.push_back(word);
	}
	return words;
}

// A hash as the worker's log writes it, which is 16 hex digits.
std::string HashText(uint64_t value) {
	char text[20];
	std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
	return text;
}

// ---- Running a worker. ----

// The StoreHeader bytes, written as PipelineCache writes them.
Bytes MakeStoreHeader(uint64_t stamp = BUILD_STAMP) {
	IR::Writer                       writer;
	auto                             jump_tables = false;
	ShaderRecompiler::Spirv::Profile profile;
	uint32_t                         shared = MAX_SHARED_BYTES;
	IR::IoAll(writer, stamp, jump_tables, profile, shared);
	return std::move(writer.bytes);
}

// A request that writes into the given directory. The eboot goes to a file there.
WorkerRequest MakeRequest(const TempDirectory& directory, const Bytes& eboot,
                          const std::string& tag = "run") {
	WorkerRequest request;
	request.store_header       = MakeStoreHeader();
	request.host_subgroup_size = HOST_SUBGROUP_SIZE;
	WriteFile(directory / (tag + ".eboot"), eboot);
	request.eboot_path     = Utf8FromPath(directory / (tag + ".eboot"));
	request.output_path    = Utf8FromPath(directory / (tag + ".out"));
	request.progress_path  = Utf8FromPath(directory / (tag + ".progress"));
	request.generated_path = Utf8FromPath(directory / (tag + ".generated"));
	request.parent_pid     = PARENT_PID;
	return request;
}

// What a run left behind.
struct RunOutput {
	WorkerResult             result;
	Bytes                    file;
	std::vector<std::string> log;
};

// Runs the worker in this process and collects what it left in its files.
RunOutput Run(const WorkerRequest& request, const WorkerHooks& hooks = {}) {
	RunOutput run;
	run.result = RunShaderStoreWorker(request, hooks);
	run.file   = ReadFile(PathFromUtf8(request.output_path));
	run.log    = ReadLines(PathFromUtf8(request.progress_path));
	return run;
}

// The log of a worker run, read from its first byte by the reader that the supervisor uses. The
// worker's own writer made it, so a line that the parser does not take is a format that drifted
// from the parser, and none may be.
ProgressLogReader ReadRealLog(const WorkerRequest& request) {
	ProgressLogReader reader(PathFromUtf8(request.progress_path));
	reader.BeginSpawnAt(0, ProgressClock::now());
	reader.PollToEnd(ProgressClock::now());
	CHECK(reader.MalformedLines() == 0);
	return reader;
}

// A record of the output, with the bytes it was framed from.
struct StoredRecord {
	DecodedModuleRecord record;
	Bytes               raw;
};

// Reads the frames after the header until the file ends. `exact` tells that every byte was used.
std::vector<StoredRecord> ReadRecords(const Bytes& file, size_t header_bytes, bool& exact) {
	std::vector<StoredRecord> records;
	size_t                    at = header_bytes;
	exact                        = file.size() >= header_bytes;
	while (exact && at < file.size()) {
		const auto frame = UnframeStoreRecord(std::span<const uint8_t>(file).subspan(at));
		if (!frame) {
			exact = false;
			break;
		}
		IR::Reader reader {frame->raw};
		auto       type = StoreRecord::GraphicsPipeline;
		IR::IoAll(reader, type);
		StoredRecord stored;
		DecodeModuleRecord(reader, stored.record);
		exact      = exact && type == StoreRecord::Module && reader.ok && reader.bytes.empty();
		stored.raw = frame->raw;
		records.push_back(std::move(stored));
		at += frame->frame_bytes;
	}
	return records;
}

// The size of the generated header for the requests above.
size_t HeaderBytes(const WorkerRequest& request) {
	return request.store_header.size() + GENERATED_HEADER_TAIL_BYTES;
}

// The expected records of the main programs, in the order the rule gives. Pass 0 holds the head of
// every program, smallest program first, and the base of the wide big program right after its head.
// Pass 1 holds the big program's other subsets, and pass 2 the base of the narrow middle program.
std::vector<Expected> ExpectedOrder(const Bytes& eboot) {
	const auto scan   = ScanEbootShaders(eboot);
	const auto big    = PredictedFor(ProgramBig(), scan.programs.at(0));
	const auto middle = PredictedFor(ProgramMiddle(), scan.programs.at(1));
	const auto small  = PredictedFor(ProgramSmall(), scan.programs.at(2));
	return {Pick(small, {}), Pick(middle, {0}), Pick(big, {0, 1}), Pick(big, {}),
	        Pick(big, {0}),  Pick(big, {1}),    Pick(middle, {})};
}

// The ids of the records, in file order.
std::vector<uint64_t> IdsOf(const std::vector<StoredRecord>& records) {
	std::vector<uint64_t> ids;
	for (const auto& stored: records) {
		ids.push_back(stored.record.id);
	}
	return ids;
}

// The code hashes of the records, in file order.
std::vector<uint64_t> IdsOfHashes(const std::vector<StoredRecord>& records) {
	std::vector<uint64_t> hashes;
	for (const auto& stored: records) {
		hashes.push_back(stored.record.hash);
	}
	return hashes;
}

// The ids the predictor expects, in the order given.
std::vector<uint64_t> IdsOf(const std::vector<Expected>& expected) {
	std::vector<uint64_t> ids;
	for (const auto& candidate: expected) {
		ids.push_back(candidate.id);
	}
	return ids;
}

// The `start` lines of a log, as "rank hash".
std::vector<std::string> StartLines(const std::vector<std::string>& log) {
	std::vector<std::string> starts;
	for (const auto& line: log) {
		const auto words = Words(line);
		if (words.size() == 3 && words[0] == "start") {
			starts.push_back(words[1] + " " + words[2]);
		}
	}
	return starts;
}

// ---- Request and header formats. ----

// A request with every field set to a value of its own, for the format tests.
WorkerRequest FullRequest() {
	WorkerRequest request;
	request.store_header       = MakeStoreHeader();
	request.host_subgroup_size = HOST_SUBGROUP_SIZE;
	request.eboot_path         = "D:/Games/PPSA00000.zar!/eboot.bin";
	request.output_path        = "cache/title.generated.123.tmp";
	request.progress_path      = "cache/title.generated.123.tmp.progress";
	request.generated_path     = "cache/title.generated";
	request.kept               = 0x123456789ull;
	request.start_rank         = 77;
	request.skip_hashes        = {0x1111222233334444ull, 5};
	request.failed_hashes      = {0xfedcba9876543210ull};
	request.parent_pid         = PARENT_PID;
	request.mode               = WorkerMode::Verify;
	request.generator_version  = 9;
	request.expected_order     = 0x0123456789abcdefull;
	return request;
}

// Whether every field of two requests is equal.
bool SameRequest(const WorkerRequest& a, const WorkerRequest& b) {
	return a.store_header == b.store_header && a.host_subgroup_size == b.host_subgroup_size &&
	       a.eboot_path == b.eboot_path && a.output_path == b.output_path &&
	       a.progress_path == b.progress_path && a.generated_path == b.generated_path &&
	       a.kept == b.kept && a.start_rank == b.start_rank && a.skip_hashes == b.skip_hashes &&
	       a.failed_hashes == b.failed_hashes && a.parent_pid == b.parent_pid && a.mode == b.mode &&
	       a.generator_version == b.generator_version && a.expected_order == b.expected_order;
}

// Every field of a request comes back, and each way a request can be wrong is refused.
void TestRequestFormat() {
	const auto request = FullRequest();
	const auto bytes   = EncodeWorkerRequest(request);
	const auto decoded = DecodeWorkerRequest(bytes);
	CHECK(decoded && SameRequest(*decoded, request));
	// The baseline differs in one field each time, so a decoder that lost a field is caught.
	WorkerRequest other = request;
	other.start_rank    = 78;
	CHECK(!SameRequest(other, request));
	other      = request;
	other.mode = WorkerMode::Generate;
	CHECK(!SameRequest(other, request));

	// Every proper prefix is refused, and so is a trailing byte.
	for (size_t length = 0; length < bytes.size(); length++) {
		CHECK(!DecodeWorkerRequest(std::span<const uint8_t>(bytes).first(length)));
	}
	auto longer = bytes;
	longer.push_back(0);
	CHECK(!DecodeWorkerRequest(longer));
	// A wrong magic or version, and a mode that does not exist.
	auto bad_magic = bytes;
	bad_magic[0] ^= 0xFF;
	CHECK(!DecodeWorkerRequest(bad_magic));
	auto bad_version = bytes;
	bad_version[sizeof(uint32_t)] ^= 0xFF;
	CHECK(!DecodeWorkerRequest(bad_version));
	WorkerRequest bad_mode = request;
	bad_mode.mode          = static_cast<WorkerMode>(7);
	CHECK(!DecodeWorkerRequest(EncodeWorkerRequest(bad_mode)));
	// A count that claims more elements than there are bytes cannot allocate them.
	auto hostile = bytes;
	Put<uint32_t>(hostile, 2 * sizeof(uint32_t), 0xFFFFFFFFu);
	CHECK(!DecodeWorkerRequest(hostile));
}

// A request survives a file, and the worker entry point tells a missing file, a damaged one and a
// good one with a missing eboot apart.
void TestRequestFile() {
	TempDirectory directory;
	const auto    request = FullRequest();
	CHECK(WriteWorkerRequestFile(directory / "request.bin", request));
	const auto read = ReadWorkerRequestFile(directory / "request.bin");
	CHECK(read && SameRequest(*read, request));
	CHECK(!ReadWorkerRequestFile(directory / "missing.bin"));

	CHECK(RunShaderStoreWorkerFromFile(directory / "missing.bin") == WORKER_EXIT_BAD_REQUEST);
	WriteFile(directory / "damaged.bin", Bytes {1, 2, 3});
	CHECK(RunShaderStoreWorkerFromFile(directory / "damaged.bin") == WORKER_EXIT_BAD_REQUEST);
	// This request is well formed, so it reaches the eboot, which does not exist.
	CHECK(RunShaderStoreWorkerFromFile(directory / "request.bin") == WORKER_EXIT_EBOOT_UNREADABLE);
	// A store header that cannot be decoded is a bad request, not an eboot problem.
	auto broken_header         = request;
	broken_header.store_header = {1, 2, 3};
	CHECK(WriteWorkerRequestFile(directory / "broken.bin", broken_header));
	CHECK(RunShaderStoreWorkerFromFile(directory / "broken.bin") == WORKER_EXIT_BAD_REQUEST);
	// A flag byte that is neither 0 nor 1 is refused, though the header is the right size. The
	// baseline is the good header, which gets as far as the missing eboot.
	auto flag_header                           = request;
	flag_header.store_header[sizeof(uint64_t)] = 2;
	CHECK(RunShaderStoreWorker(flag_header).exit_code == WORKER_EXIT_BAD_REQUEST);
	CHECK(RunShaderStoreWorker(request).exit_code == WORKER_EXIT_EBOOT_UNREADABLE);
}

// The generated header is the store header, the version and the identity, and the identity does not
// depend on the order of the programs.
void TestGeneratedHeader() {
	const auto                  store_header = MakeStoreHeader();
	std::vector<ScannedProgram> programs(3);
	programs[0].code_hash                  = 0x30;
	programs[1].code_hash                  = 0x10;
	programs[2].code_hash                  = 0x20;
	const auto                    identity = ComputeEbootIdentity(1000, programs);
	const std::array<uint64_t, 3> sorted {0x10, 0x20, 0x30};
	CHECK(identity.size == 1000);
	CHECK(identity.program_hash == XXH3_64bits(sorted.data(), sizeof sorted));
	// Programs in another order, a changed hash and a changed size each give what they should.
	std::vector<ScannedProgram> shuffled {programs[2], programs[0], programs[1]};
	CHECK(ComputeEbootIdentity(1000, shuffled) == identity);
	auto changed         = programs;
	changed[0].code_hash = 0x31;
	CHECK(!(ComputeEbootIdentity(1000, changed) == identity));
	CHECK(!(ComputeEbootIdentity(1001, programs) == identity));

	const auto header = EncodeGeneratedHeader(store_header, 5, identity);
	CHECK(header.size() == store_header.size() + GENERATED_HEADER_TAIL_BYTES);
	CHECK(std::equal(store_header.begin(), store_header.end(), header.begin()));
	auto parsed = ParseGeneratedHeader(header, store_header.size());
	CHECK(parsed && parsed->store_header == store_header && parsed->generator_version == 5 &&
	      parsed->identity == identity && parsed->bytes == header.size());
	CHECK(!ParseGeneratedHeader(std::span<const uint8_t>(header).first(header.size() - 1),
	                            store_header.size()));
	// Another version or identity gives other bytes, so a byte compare tells them apart.
	CHECK(EncodeGeneratedHeader(store_header, 6, identity) != header);
	CHECK(EncodeGeneratedHeader(store_header, 5, ComputeEbootIdentity(1001, programs)) != header);
}

// ---- Worker tests. ----

// The records are the predicted candidates in the rule's order, each one decodes, and its id is the
// predicted id and also the id the runtime would compute from the record's own module.
void TestRecordsInRankOrder() {
	TempDirectory directory;
	const auto    eboot    = BuildEboot(MainPrograms());
	const auto    request  = MakeRequest(directory, eboot);
	const auto    run      = Run(request);
	const auto    expected = ExpectedOrder(eboot);
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(expected.size() == 7);

	// The header is the request's store header, the version and the identity of this eboot.
	const auto scan   = ScanEbootShaders(eboot);
	const auto header = ParseGeneratedHeader(run.file, request.store_header.size());
	CHECK(header && header->store_header == request.store_header);
	CHECK(header && header->generator_version == SHADER_STORE_GENERATOR_VERSION);
	CHECK(header && header->identity == ComputeEbootIdentity(eboot.size(), scan.programs));

	bool       exact   = false;
	const auto records = ReadRecords(run.file, HeaderBytes(request), exact);
	CHECK(exact);
	CHECK(records.size() == expected.size());
	// The order of the ids is the order of the rule. An eboot order walk would list the big program
	// first, and the old base first order would put the middle program's base before the big head.
	CHECK(IdsOf(records) == IdsOf(expected));
	std::vector<uint64_t> distinct = IdsOf(records);
	std::ranges::sort(distinct);
	CHECK(std::ranges::adjacent_find(distinct) == distinct.end());
	for (size_t index = 0; index < std::min(records.size(), expected.size()); index++) {
		const auto& record = records[index].record;
		CHECK(record.stage == ShaderType::Compute && record.hash == expected[index].hash);
		// This uses the worker's own formula, so it only shows the record agrees with itself.
		// The comparison with the predicted ids above is the real check.
		CHECK(record.id == ShaderModuleId(record.stage, record.hash, record.user_data_count,
		                                  record.static_state,
		                                  record.info.bindings.push_data_start_dword,
		                                  record.specialization, {}));
		CHECK(!record.spirv.empty());
	}
	CHECK(run.result.records_written == 7 && run.result.candidates == 7);
	CHECK(run.result.push_start_mismatches == 0 && run.result.drops.empty());
	CHECK(run.result.programs_scanned == 3 && run.result.programs_predicted == 3);
}

// The log has a start line for every rank, in order, each before its record, and the summary last.
void TestProgressLog() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot);
	const auto    run     = Run(request);
	const auto    scan    = ScanEbootShaders(eboot);
	// Programs by rank, which are small, middle, big, big, big, big, middle.
	const std::array<size_t, 7> program_of_rank {2, 1, 0, 0, 0, 0, 1};
	std::vector<std::string>    expected_starts;
	for (size_t rank = 0; rank < program_of_rank.size(); rank++) {
		expected_starts.push_back(std::to_string(rank) + " " +
		                          HashText(scan.programs[program_of_rank[rank]].code_hash));
	}
	CHECK(StartLines(run.log) == expected_starts);
	// The programs are logged before they are predicted, and the summary closes the log.
	size_t predicts = 0;
	for (const auto& line: run.log) {
		predicts += line.starts_with("predict ") ? 1 : 0;
	}
	CHECK(predicts == 3);
	CHECK(!run.log.empty() && run.log.back().starts_with("summary "));

	// While a candidate is translated its start line is the log's last line, and the output holds
	// only the records before it. A line written after the translation would not be there yet.
	TempDirectory second;
	const auto    hooked_request = MakeRequest(second, eboot);
	size_t        checked        = 0;
	bool          ordered        = true;
	WorkerHooks   hooks;
	hooks.before_candidate = [&](uint64_t rank) {
		const auto log     = ReadLines(PathFromUtf8(hooked_request.progress_path));
		const auto output  = ReadFile(PathFromUtf8(hooked_request.output_path));
		bool       exact   = false;
		const auto records = ReadRecords(output, HeaderBytes(hooked_request), exact);
		ordered            = ordered && !log.empty() && StartLines(log).size() == rank + 1 &&
		          Words(log.back()).size() == 3 && Words(log.back())[0] == "start" &&
		          Words(log.back())[1] == std::to_string(rank) && exact && records.size() == rank;
		checked++;
	};
	const auto hooked = Run(hooked_request, hooks);
	CHECK(hooked.result.exit_code == WORKER_EXIT_DONE);
	CHECK(checked == 7 && ordered);
}

// A worker started at a rank writes the records from that rank on, the same bytes a full run wrote.
void TestStartRank() {
	TempDirectory directory;
	const auto    eboot = BuildEboot(MainPrograms());
	const auto    full  = Run(MakeRequest(directory, eboot, "full"));
	bool          exact = false;
	const auto all = ReadRecords(full.file, HeaderBytes(MakeRequest(directory, eboot, "x")), exact);
	constexpr uint64_t START = 3;
	auto               from  = MakeRequest(directory, eboot, "from");
	from.start_rank          = START;
	const auto partial       = Run(from);
	const auto tail          = ReadRecords(partial.file, HeaderBytes(from), exact);
	CHECK(exact && all.size() == 7 && tail.size() == 7 - START);
	for (size_t index = 0; index < tail.size() && index + START < all.size(); index++) {
		CHECK(tail[index].raw == all[index + START].raw);
	}
	CHECK(partial.result.candidates_before_start == START && partial.result.candidates == 7);
	// The log starts at the start rank and counts up from there.
	const auto starts = StartLines(partial.log);
	CHECK(starts.size() == 7 - START && !starts.empty() && starts.front().starts_with("3 "));
	CHECK(starts.back().starts_with("6 "));
}

// A full run of the main programs, which the skip tests compare against.
struct FullRun {
	Bytes                     eboot;
	uint64_t                  big = 0;
	WorkerRequest             request;
	RunOutput                 run;
	std::vector<StoredRecord> records;
};

FullRun MakeFullRun(const TempDirectory& directory) {
	FullRun full;
	full.eboot   = BuildEboot(MainPrograms());
	full.big     = ScanEbootShaders(full.eboot).programs[0].code_hash;
	full.request = MakeRequest(directory, full.eboot, "full");
	full.run     = Run(full.request);
	bool exact   = false;
	full.records = ReadRecords(full.run.file, HeaderBytes(full.request), exact);
	return full;
}

// How many lines of a log start with the text.
size_t CountLines(const std::vector<std::string>& log, const std::string& prefix) {
	return static_cast<size_t>(std::ranges::count_if(
	    log, [&](const std::string& line) { return line.starts_with(prefix); }));
}

// A program in the request's skip hashes is never predicted, so it has no ranks at all. The other
// programs come out as the bytes they have in a full run, in the same order among themselves.
void TestSkipHashes() {
	TempDirectory directory;
	const auto    full = MakeFullRun(directory);
	// The big program's records are in the baseline, so it is the skip that removes them.
	CHECK(full.records.size() == 7 &&
	      CountLines(full.run.log, "predict " + HashText(full.big)) == 1);
	CHECK(std::ranges::count(IdsOfHashes(full.records), full.big) == 4);

	auto skip        = MakeRequest(directory, full.eboot, "skip");
	skip.skip_hashes = {full.big};
	const auto run   = Run(skip);
	bool       exact = false;
	const auto kept  = ReadRecords(run.file, HeaderBytes(skip), exact);
	// Four of the seven candidates are the big program's, so three records are left.
	CHECK(exact && kept.size() == 3 && run.result.programs_skipped_by_request == 1);
	CHECK(run.result.candidates == 3 && run.result.records_written == 3);
	CHECK(std::ranges::count(IdsOfHashes(kept), full.big) == 0);
	if (kept.size() == 3 && full.records.size() == 7) {
		CHECK(kept[0].raw == full.records[0].raw && kept[1].raw == full.records[1].raw &&
		      kept[2].raw == full.records[6].raw);
	}
	// Left out before prediction, so it is not even translated for the prediction.
	CHECK(CountLines(run.log, "predict " + HashText(full.big)) == 0);
}

// A program in failed_hashes keeps its ranks and is not built, so the ranks of the other programs
// do not move and a restart can go on at the next rank.
void TestFailedHashes() {
	TempDirectory directory;
	const auto    full = MakeFullRun(directory);

	auto failed          = MakeRequest(directory, full.eboot, "failed");
	failed.failed_hashes = {full.big};
	const auto run       = Run(failed);
	bool       exact     = false;
	const auto rest      = ReadRecords(run.file, HeaderBytes(failed), exact);
	CHECK(exact && rest.size() == 3 && run.result.candidates_failed_earlier == 4);
	CHECK(run.result.candidates == 7 && run.result.records_written == 3);
	// The ranks of the others are what they were, which are 0, 1 and 6.
	const auto full_starts = StartLines(full.run.log);
	const auto starts      = StartLines(run.log);
	CHECK(starts.size() == 3 && full_starts.size() == 7);
	if (starts.size() == 3 && full_starts.size() == 7) {
		CHECK(starts[0] == full_starts[0] && starts[1] == full_starts[1] &&
		      starts[2] == full_starts[6]);
	}
	if (rest.size() == 3 && full.records.size() == 7) {
		CHECK(rest[0].raw == full.records[0].raw && rest[1].raw == full.records[1].raw &&
		      rest[2].raw == full.records[6].raw);
	}
	// A failed program is still predicted, which is what keeps its ranks.
	CHECK(CountLines(run.log, "predict " + HashText(full.big)) == 1);
}

// A restarted worker keeps the good bytes of the file and carries on, so the file ends up the one
// an uninterrupted run makes. A kept file that does not start with this header is refused.
void TestKeptAppends() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot, "full");
	const auto    full    = Run(request);
	bool          exact   = false;
	const auto    records = ReadRecords(full.file, HeaderBytes(request), exact);
	// Cut the file after three records, and leave half of the fourth behind as a crash would.
	size_t cut = HeaderBytes(request);
	for (size_t index = 0; index < 3; index++) {
		cut += FrameStoreRecord(records[index].raw).size();
	}
	Bytes torn(full.file.begin(), full.file.begin() + static_cast<ptrdiff_t>(cut + TORN_BYTES));
	auto  resume = MakeRequest(directory, eboot, "resume");
	WriteFile(PathFromUtf8(resume.output_path), torn);
	resume.kept           = cut;
	resume.start_rank     = 3;
	resume.expected_order = full.result.order_hash;
	const auto resumed    = Run(resume);
	CHECK(resumed.result.exit_code == WORKER_EXIT_DONE);
	CHECK(resumed.file == full.file);

	// The same bytes with another stamp in front do not match this request's header.
	auto wrong = MakeRequest(directory, eboot, "wrong");
	WriteFile(PathFromUtf8(wrong.output_path), torn);
	wrong.store_header = MakeStoreHeader(OTHER_BUILD_STAMP);
	wrong.kept         = cut;
	wrong.start_rank   = 3;
	CHECK(RunShaderStoreWorker(wrong).exit_code == WORKER_EXIT_OUTPUT_FAILED);
	// A file shorter than what is said to be kept is refused too.
	auto short_file = MakeRequest(directory, eboot, "short");
	WriteFile(PathFromUtf8(short_file.output_path), torn);
	short_file.kept = cut + PAST_THE_FILE;
	CHECK(RunShaderStoreWorker(short_file).exit_code == WORKER_EXIT_OUTPUT_FAILED);
}

// A request in Verify mode against the finished file of the main programs.
struct VerifySetup {
	Bytes    eboot;
	fs::path finished;
};

VerifySetup MakeFinishedFile(const TempDirectory& directory) {
	VerifySetup setup;
	setup.eboot = BuildEboot(MainPrograms());
	auto make   = MakeRequest(directory, setup.eboot, "make");
	CHECK(Run(make).result.exit_code == WORKER_EXIT_DONE);
	// The finished file is what the supervisor would have renamed the output to.
	setup.finished = directory / "title.generated";
	WriteFile(setup.finished, ReadFile(PathFromUtf8(make.output_path)));
	return setup;
}

WorkerRequest VerifyRequest(const TempDirectory& directory, const VerifySetup& setup,
                            const Bytes& eboot, const std::string& tag) {
	auto request           = MakeRequest(directory, eboot, tag);
	request.mode           = WorkerMode::Verify;
	request.generated_path = Utf8FromPath(setup.finished);
	return request;
}

// Verify mode leaves a current file alone, and writes nothing.
void TestVerifyUpToDate() {
	TempDirectory directory;
	const auto    setup   = MakeFinishedFile(directory);
	const auto    request = VerifyRequest(directory, setup, setup.eboot, "same");
	const auto    run     = Run(request);
	CHECK(run.result.exit_code == WORKER_EXIT_UP_TO_DATE);
	CHECK(!fs::exists(PathFromUtf8(request.output_path)) && run.result.records_written == 0);
	CHECK(run.log.empty());

	// The same programs in another order are the same eboot size and the same identity, so only the
	// identity decides, and it does not depend on the order.
	auto swapped = MainPrograms();
	std::swap(swapped[1], swapped[2]);
	const auto reordered = VerifyRequest(directory, setup, BuildEboot(swapped), "reordered");
	CHECK(Run(reordered).result.exit_code == WORKER_EXIT_UP_TO_DATE);

	// Generate mode ignores a current file, and so does Verify mode when there is no file.
	auto generate           = MakeRequest(directory, setup.eboot, "generate");
	generate.generated_path = request.generated_path;
	CHECK(Run(generate).result.exit_code == WORKER_EXIT_DONE);
	auto none = MakeRequest(directory, setup.eboot, "none");
	none.mode = WorkerMode::Verify;
	CHECK(Run(none).result.exit_code == WORKER_EXIT_DONE);
}

// Verify mode generates again when the eboot, the host values or the generator version changed. The
// baseline for each is the unchanged request above, which is up to date.
void TestVerifyRegenerates() {
	TempDirectory directory;
	const auto    setup = MakeFinishedFile(directory);
	CHECK(Run(VerifyRequest(directory, setup, setup.eboot, "same")).result.exit_code ==
	      WORKER_EXIT_UP_TO_DATE);

	// Another eboot gives another identity. One program changed is enough.
	auto changed_programs            = MainPrograms();
	changed_programs[2].padding_nops = 1;
	const auto changed = VerifyRequest(directory, setup, BuildEboot(changed_programs), "changed");
	const auto regenerated = Run(changed);
	CHECK(regenerated.result.exit_code == WORKER_EXIT_DONE);
	CHECK(regenerated.result.records_written == 7 && fs::exists(PathFromUtf8(changed.output_path)));
	const auto header = ParseGeneratedHeader(regenerated.file, changed.store_header.size());
	CHECK(header && header->identity == regenerated.result.identity);

	auto stamp         = VerifyRequest(directory, setup, setup.eboot, "stamp");
	stamp.store_header = MakeStoreHeader(OTHER_BUILD_STAMP);
	CHECK(Run(stamp).result.exit_code == WORKER_EXIT_DONE);
	auto version              = VerifyRequest(directory, setup, setup.eboot, "version");
	version.generator_version = SHADER_STORE_GENERATOR_VERSION + 1;
	CHECK(Run(version).result.exit_code == WORKER_EXIT_DONE);
}

// Programs the store cannot hold are counted by what kept them out, and never reach the output.
void TestProgramsLeftOut() {
	TempDirectory directory;
	ProgramSpec   calls          = ProgramSmall();
	calls.calls                  = true;
	ProgramSpec derived          = ProgramSmall();
	derived.derived_descriptor   = true;
	derived.padding_nops         = 1;
	ProgramSpec odd              = ProgramMiddle();
	odd.loads[0].no_rule         = true;
	ProgramSpec unreadable       = ProgramSmall();
	unreadable.without_registers = true;
	unreadable.padding_nops      = 2;
	auto good                    = ProgramSmall();
	good.padding_nops            = 3;
	const auto  eboot            = BuildEboot({calls, derived, odd, unreadable, good});
	const auto  request          = MakeRequest(directory, eboot);
	const auto  run              = Run(request);
	const auto& result           = run.result;
	CHECK(result.exit_code == WORKER_EXIT_DONE);
	CHECK(result.programs_scanned == 5 && result.programs_predicted == 1);
	CHECK(result.programs_with_calls == 1);
	CHECK(result.programs_unclassified == 1);
	CHECK(result.programs_with_problems == 1);
	CHECK(result.programs_unreadable == 1);
	CHECK(result.candidates == 1 && result.records_written == 1);
	bool exact = false;
	CHECK(ReadRecords(run.file, HeaderBytes(request), exact).size() == 1 && exact);
}

// The `prediction` lines of a log, as the words of each line.
std::vector<std::vector<std::string>> PredictionLines(const std::vector<std::string>& log) {
	std::vector<std::vector<std::string>> lines;
	for (const auto& line: log) {
		const auto words = Words(line);
		if (!words.empty() && words[0] == "prediction") {
			lines.push_back(words);
		}
	}
	return lines;
}

// Whether every `predict` line of a log is followed at once by the `prediction` line of the same
// program. A line that came later or not at all would leave a crash blamed on the wrong step.
bool PredictionFollowsPredict(const std::vector<std::string>& log) {
	size_t pairs = 0;
	for (size_t index = 0; index < log.size(); index++) {
		const auto words = Words(log[index]);
		if (words.size() != 2 || words[0] != "predict") {
			continue;
		}
		const auto next =
		    index + 1 < log.size() ? Words(log[index + 1]) : std::vector<std::string> {};
		if (next.size() != 4 || next[0] != "prediction" || next[1] != words[1]) {
			return false;
		}
		pairs++;
	}
	return pairs > 0;
}

// Every program gets one `prediction` line, with the outcome the result counts it under. The
// programs are of every kind the fixture makes. The unreadable one has no static key, so it takes
// the early exit of the loop, which is the path that a line written only after the prediction would
// miss. The others do not name the outcome by accident, since each is a program of its own kind
// and the four outcomes are counted apart in the result.
void TestPredictionLinePerProgram() {
	TempDirectory directory;
	ProgramSpec   calls          = ProgramSmall();
	calls.calls                  = true;
	ProgramSpec derived          = ProgramSmall();
	derived.derived_descriptor   = true;
	derived.padding_nops         = 1;
	ProgramSpec odd              = ProgramMiddle();
	odd.loads[0].no_rule         = true;
	ProgramSpec unreadable       = ProgramSmall();
	unreadable.without_registers = true;
	unreadable.padding_nops      = 2;
	auto good                    = ProgramSmall();
	good.padding_nops            = 3;
	const auto eboot             = BuildEboot({calls, derived, odd, unreadable, good});
	const auto scan              = ScanEbootShaders(eboot);
	CHECK(scan.programs.size() == 5);
	const auto request = MakeRequest(directory, eboot);
	const auto run     = Run(request);
	const auto lines   = PredictionLines(run.log);
	CHECK(lines.size() == run.result.programs_scanned && PredictionFollowsPredict(run.log));
	const std::array<std::pair<const char*, uint64_t>, 5> expected = {
	    {{"calls", 0}, {"unclassified", 0}, {"problems", 0}, {"unreadable", 0}, {"predicted", 1}}};
	for (size_t index = 0; index < expected.size() && index < scan.programs.size(); index++) {
		const auto& line = index < lines.size() ? lines[index] : std::vector<std::string> {};
		CHECK(line.size() == 4 && line[1] == HashText(scan.programs[index].code_hash));
		CHECK(line.size() == 4 && line[2] == expected[index].first);
		CHECK(line.size() == 4 && line[3] == std::to_string(expected[index].second));
	}
	// The totals by outcome are the result's counters, so no line is in the wrong class.
	std::unordered_map<std::string, uint32_t> by_outcome;
	uint64_t                                  candidates = 0;
	for (const auto& line: lines) {
		by_outcome[line.size() == 4 ? line[2] : ""]++;
		candidates += line.size() == 4 ? std::stoull(line[3]) : 0;
	}
	CHECK(by_outcome["predicted"] == run.result.programs_predicted);
	CHECK(by_outcome["calls"] == run.result.programs_with_calls);
	CHECK(by_outcome["unclassified"] == run.result.programs_unclassified);
	CHECK(by_outcome["unreadable"] == run.result.programs_unreadable);
	CHECK(by_outcome["problems"] == run.result.programs_with_problems);
	CHECK(candidates == run.result.candidates);

	// The same log through the reader the supervisor uses.
	const auto log = ReadRealLog(request);
	CHECK(log.KindCount(ProgressKind::Prediction) == 5 && log.Predictions().size() == 5);
	if (log.Predictions().size() == 5) {
		CHECK(log.Predictions()[3].outcome == PredictionOutcome::Unreadable);
		CHECK(log.Predictions()[3].hash == scan.programs[3].code_hash);
		CHECK(log.Predictions()[4].outcome == PredictionOutcome::Predicted &&
		      log.Predictions()[4].candidates == 1);
	}

	// A program the request skips is never predicted, so it has neither line. The baseline is the
	// run above, where the same program has both.
	auto skipping         = MakeRequest(directory, eboot, "skipping");
	skipping.skip_hashes  = {scan.programs[4].code_hash};
	const auto skipped    = Run(skipping);
	const auto kept_lines = PredictionLines(skipped.log);
	CHECK(skipped.result.programs_skipped_by_request == 1 && kept_lines.size() == 4);
	for (const auto& line: kept_lines) {
		CHECK(line.size() == 4 && line[1] != HashText(scan.programs[4].code_hash));
	}
}

// A program with several candidates reports how many, which is the count that the ordering holds
// for it. The baseline is a count of one, as the program above has, so a line that always said 1
// fails.
void TestPredictionCandidateCounts() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    scan    = ScanEbootShaders(eboot);
	const auto    request = MakeRequest(directory, eboot);
	const auto    run     = Run(request);
	const auto    lines   = PredictionLines(run.log);
	CHECK(lines.size() == scan.programs.size() && run.result.candidates == MAIN_CANDIDATES);
	uint64_t total   = 0;
	uint64_t largest = 0;
	for (const auto& line: lines) {
		CHECK(line.size() == 4 && line[2] == "predicted");
		const uint64_t count = line.size() == 4 ? std::stoull(line[3]) : 0;
		total += count;
		largest = std::max(largest, count);
	}
	CHECK(total == MAIN_CANDIDATES && largest > 1);
	const auto log = ReadRealLog(request);
	CHECK(log.Predictions().size() == scan.programs.size());
	uint64_t reader_total = 0;
	for (const auto& row: log.Predictions()) {
		reader_total += row.candidates;
	}
	CHECK(reader_total == MAIN_CANDIDATES);
}

// A program with no buffer that reads user data. The rule says a program with no buffers has no
// push data, but the module keeps the user data it reads there, so it starts at 0.
ProgramSpec ProgramUserDataOnly() {
	ProgramSpec spec;
	spec.store          = false;
	spec.user_data_only = true;
	return spec;
}

// The same program with an output buffer. The rule says a program with buffers starts at 0, which
// is right, so nothing differs from the program above except the buffer.
ProgramSpec ProgramUserDataWithBuffer() {
	ProgramSpec spec = ProgramUserDataOnly();
	spec.store       = true;
	spec.unused_records.push_back(
	    {Sl00Class::ReadOnly, UNUSED_REGISTER, SGPR_SECOND, "gUnused", false});
	return spec;
}

// The check on the push data start drops a program whose module starts elsewhere than the rule
// said. Here the rule is wrong for real, since the program has no buffer and still has push data.
void TestPushStartTripwireOnRealMiss() {
	TempDirectory directory;
	const auto    right_eboot = BuildEboot({ProgramUserDataWithBuffer()});
	const auto    right       = Run(MakeRequest(directory, right_eboot, "right"));
	CHECK(right.result.exit_code == WORKER_EXIT_DONE);
	CHECK(right.result.push_start_mismatches == 0 && right.result.drops.empty());
	CHECK(right.result.records_written == 1);

	const auto wrong_eboot   = BuildEboot({ProgramUserDataOnly()});
	const auto wrong_request = MakeRequest(directory, wrong_eboot, "wrong");
	const auto wrong         = Run(wrong_request);
	const auto scan          = ScanEbootShaders(wrong_eboot);
	CHECK(wrong.result.exit_code == WORKER_EXIT_DONE);
	CHECK(wrong.result.candidates == 1 && wrong.result.programs_predicted == 1);
	CHECK(wrong.result.push_start_mismatches == 1 && wrong.result.drops.size() == 1);
	if (wrong.result.drops.size() == 1) {
		const auto& drop = wrong.result.drops[0];
		CHECK(drop.reason == DropReason::PushStartMismatch && drop.rank == 0);
		CHECK(drop.hash == scan.programs[0].code_hash);
		// The rule said none, and the module starts at the first dword.
		CHECK(drop.predicted_push == IR::PushData::NoStart && drop.actual_push == 0);
	}
	// A record filed under the id the runtime would not compute is never found, so none is kept.
	bool       exact   = false;
	const auto records = ReadRecords(wrong.file, HeaderBytes(wrong_request), exact);
	CHECK(exact && records.empty() && wrong.result.records_written == 0);
	CHECK(wrong.result.candidates_dropped == 1);
	bool logged = false;
	for (const auto& line: wrong.log) {
		logged =
		    logged || line.find("push_start predicted=4294967295 actual=0") != std::string::npos;
	}
	CHECK(logged);
	// The same run's log, read as the supervisor reads it: one drop, with the numbers the result
	// has.
	const auto log = ReadRealLog(wrong_request);
	CHECK(log.KindCount(ProgressKind::Drop) == 1 && log.DropsSinceSpawn().size() == 1);
	if (log.DropsSinceSpawn().size() == 1) {
		const auto& drop = log.DropsSinceSpawn()[0];
		CHECK(drop.reason == ProgressDropReason::PushStart && drop.rank_or_count == 0);
		CHECK(drop.hash == scan.programs[0].code_hash);
		CHECK(drop.predicted_push == IR::PushData::NoStart && drop.actual_push == 0);
	}
	// The run with no drop has none in its log either.
	CHECK(ReadRealLog(MakeRequest(directory, right_eboot, "right")).KindCount(ProgressKind::Drop) ==
	      0);
}

// A prediction made wrong on purpose drops the program, and only that program. Its later candidates
// are not built, though they would be right, and the other programs come out as in a full run.
void TestPushStartTripwireDropsTheRest() {
	TempDirectory directory;
	const auto    eboot            = BuildEboot(MainPrograms());
	const auto    scan             = ScanEbootShaders(eboot);
	const auto    big              = scan.programs[0].code_hash;
	const auto    baseline_request = MakeRequest(directory, eboot, "baseline");
	const auto    baseline         = Run(baseline_request);
	bool          exact            = false;
	const auto    all = ReadRecords(baseline.file, HeaderBytes(baseline_request), exact);
	CHECK(baseline.result.push_start_mismatches == 0 && all.size() == 7);

	constexpr uint32_t WRONG_START = 7;
	WorkerHooks        hooks;
	hooks.after_prediction = [&](uint64_t hash, std::vector<Candidate>& candidates) {
		if (hash == big) candidates.front().push_start = WRONG_START;
	};
	const auto request = MakeRequest(directory, eboot, "wrong");
	const auto run     = Run(request, hooks);
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(run.result.push_start_mismatches == 1 && run.result.drops.size() == 1);
	if (run.result.drops.size() == 1) {
		const auto& drop = run.result.drops[0];
		CHECK(drop.hash == big && drop.rank == 3);
		CHECK(drop.predicted_push == WRONG_START && drop.actual_push == 0);
	}
	// Seven candidates. The wrong start is on the big program's base, which is rank 3 now that its
	// head, rank 2, comes first and is written. The base and the ranks 4 and 5 are dropped, and
	// those two would not trip on their own. Rank 6 is another program and is written.
	CHECK(run.result.candidates == 7 && run.result.records_written == 4);
	CHECK(run.result.candidates_dropped == 3);
	const auto records = ReadRecords(run.file, HeaderBytes(request), exact);
	CHECK(exact && records.size() == 4 && all.size() == 7);
	if (records.size() == 4 && all.size() == 7) {
		CHECK(records[0].raw == all[0].raw && records[1].raw == all[1].raw &&
		      records[2].raw == all[2].raw && records[3].raw == all[6].raw);
	}
	// Ranks 4 and 5 have no start line, so nothing was translated for them. Ranks 0 to 3 and 6 do.
	CHECK(StartLines(run.log).size() == 4 + 1);
	const auto log = ReadRealLog(request);
	CHECK(log.KindCount(ProgressKind::Drop) == 1 && log.DropsSinceSpawn().size() == 1);
	if (log.DropsSinceSpawn().size() == 1) {
		const auto& drop = log.DropsSinceSpawn()[0];
		CHECK(drop.reason == ProgressDropReason::PushStart && drop.rank_or_count == 3);
		CHECK(drop.hash == big && drop.predicted_push == WRONG_START && drop.actual_push == 0);
	}
}

// A restart with nothing left to write still cuts the file back to the kept bytes, so a torn record
// does not stay at its end. The baseline is the torn file itself, which is longer.
void TestKeptTruncatesTornBytes() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot, "full");
	const auto    full    = Run(request);
	bool          exact   = false;
	const auto    records = ReadRecords(full.file, HeaderBytes(request), exact);
	size_t        cut     = HeaderBytes(request);
	for (size_t index = 0; index < 3; index++) {
		cut += FrameStoreRecord(records[index].raw).size();
	}
	Bytes torn(full.file.begin(), full.file.begin() + static_cast<ptrdiff_t>(cut + TORN_BYTES));
	auto  resume = MakeRequest(directory, eboot, "resume");
	WriteFile(PathFromUtf8(resume.output_path), torn);
	resume.kept           = cut;
	resume.start_rank     = MAIN_CANDIDATES;
	resume.expected_order = full.result.order_hash;
	const auto resumed    = Run(resume);
	CHECK(resumed.result.exit_code == WORKER_EXIT_DONE && resumed.result.records_written == 0);
	CHECK(torn.size() == cut + TORN_BYTES);
	CHECK(resumed.file.size() == cut);
	CHECK(std::equal(resumed.file.begin(), resumed.file.end(), full.file.begin()));
}

// A restart that computes another ordering than the run it continues exits without touching the
// kept bytes. The baseline is the same restart with the right ordering, which goes on.
void TestOrderGuard() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot, "full");
	const auto    full    = Run(request);
	CHECK(full.result.order_hash != 0);
	bool       exact   = false;
	const auto records = ReadRecords(full.file, HeaderBytes(request), exact);
	size_t     cut     = HeaderBytes(request);
	for (size_t index = 0; index < 3; index++) {
		cut += FrameStoreRecord(records[index].raw).size();
	}
	const Bytes kept(full.file.begin(), full.file.begin() + static_cast<ptrdiff_t>(cut));

	auto wrong = MakeRequest(directory, eboot, "wrong");
	WriteFile(PathFromUtf8(wrong.output_path), kept);
	wrong.kept           = cut;
	wrong.start_rank     = 3;
	wrong.expected_order = full.result.order_hash ^ 1;
	const auto refused   = Run(wrong);
	CHECK(refused.result.exit_code == WORKER_EXIT_ORDER_MISMATCH);
	CHECK(refused.file == kept && refused.result.records_written == 0);

	auto right = MakeRequest(directory, eboot, "right");
	WriteFile(PathFromUtf8(right.output_path), kept);
	right.kept           = cut;
	right.start_rank     = 3;
	right.expected_order = full.result.order_hash;
	CHECK(Run(right).result.exit_code == WORKER_EXIT_DONE);
	// A restart that keeps no bytes but starts past rank 0 is guarded too.
	auto ranked_only           = MakeRequest(directory, eboot, "ranked_only");
	ranked_only.start_rank     = 3;
	ranked_only.expected_order = full.result.order_hash ^ 1;
	CHECK(Run(ranked_only).result.exit_code == WORKER_EXIT_ORDER_MISMATCH);
	// A new file has nothing to disagree with, and an expected order of zero means no expectation.
	auto fresh           = MakeRequest(directory, eboot, "fresh");
	fresh.expected_order = full.result.order_hash ^ 1;
	CHECK(Run(fresh).result.exit_code == WORKER_EXIT_DONE);
}

// Each program's `predict` line is the log's last line when its prediction starts.
void TestPredictLineComesFirst() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot);
	size_t        checked = 0;
	bool          ordered = true;
	WorkerHooks   hooks;
	hooks.before_prediction = [&](uint64_t hash) {
		const auto log = ReadLines(PathFromUtf8(request.progress_path));
		ordered        = ordered && !log.empty() && log.back() == "predict " + HashText(hash);
		checked++;
	};
	CHECK(Run(request, hooks).result.exit_code == WORKER_EXIT_DONE);
	CHECK(checked == 3 && ordered);
}

// The worker goes on past a drop. This covers a crash later in a run that has already dropped a
// program. The supervisor restarts with the dropped program in failed_hashes and the rank after
// the drop, and the file ends up as the run that went on. The baseline is the same restart without
// failed_hashes, which builds the program again.
void TestRestartAfterDrop() {
	TempDirectory directory;
	const auto    eboot = BuildEboot(MainPrograms());
	const auto    big   = ScanEbootShaders(eboot).programs[0].code_hash;
	WorkerHooks   hooks;
	hooks.after_prediction = [&](uint64_t hash, std::vector<Candidate>& candidates) {
		if (hash == big) candidates.front().push_start = 7;
	};
	const auto request = MakeRequest(directory, eboot, "first");
	const auto first   = Run(request, hooks);
	CHECK(first.result.drops.size() == 1 && first.result.records_written == 4);
	if (first.result.drops.size() != 1) return;
	const auto& drop    = first.result.drops[0];
	bool        exact   = false;
	const auto  records = ReadRecords(first.file, HeaderBytes(request), exact);
	// The bytes the supervisor kept when it saw the drop line, which are the records before it. A
	// rank past the records read would index out of range, so it fails here instead.
	CHECK(drop.rank <= records.size());
	if (drop.rank > records.size()) {
		return;
	}
	size_t cut = HeaderBytes(request);
	for (size_t index = 0; index < drop.rank; index++) {
		cut += FrameStoreRecord(records[index].raw).size();
	}
	const Bytes kept(first.file.begin(), first.file.begin() + static_cast<ptrdiff_t>(cut));

	auto restart = MakeRequest(directory, eboot, "restart");
	WriteFile(PathFromUtf8(restart.output_path), kept);
	restart.kept           = cut;
	restart.start_rank     = drop.rank + 1;
	restart.expected_order = first.result.order_hash;
	restart.failed_hashes  = {drop.hash};
	const auto restarted   = Run(restart);
	CHECK(restarted.result.exit_code == WORKER_EXIT_DONE);
	CHECK(restarted.file == first.file);

	auto again = MakeRequest(directory, eboot, "again");
	WriteFile(PathFromUtf8(again.output_path), kept);
	again.kept           = cut;
	again.start_rank     = drop.rank + 1;
	again.expected_order = first.result.order_hash;
	const auto rebuilt   = Run(again);
	CHECK(rebuilt.result.exit_code == WORKER_EXIT_DONE && rebuilt.file != first.file);
}

// A specialization with an extra buffer is not dropped, since indirect children are appended to it.
// Only a program with more buffers than the specialization is. The fixture has no program with an
// image or a sampler, so those two comparisons are not exercised here.
void TestExtraSpecializationBufferIsKept() {
	TempDirectory directory;
	const auto    eboot = BuildEboot(MainPrograms());
	WorkerHooks   hooks;
	hooks.after_prediction = [&](uint64_t, std::vector<Candidate>& candidates) {
		auto& buffers = candidates.front().specialization.buffers;
		auto  extra   = buffers.back();
		// A child that points at the first buffer, as an indirect child does.
		extra.indirect_root = 0;
		buffers.push_back(extra);
	};
	const auto run = Run(MakeRequest(directory, eboot), hooks);
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(run.result.resource_count_mismatches == 0);
	CHECK(std::ranges::none_of(run.result.drops, [](const WorkerDrop& drop) {
		return drop.reason == DropReason::ResourceCountMismatch;
	}));
	// Every candidate is written, so none was dropped for another reason either.
	CHECK(run.result.records_written == MAIN_CANDIDATES && run.result.drops.empty());
}

// The comparison that guards a specialization, with hand built specializations. Each kind has a
// program that lists more than the specialization, which is refused, and the baseline with equal
// counts, which is not. A specialization with an extra entry of every kind is accepted.
void TestProgramExceedsSpecialization() {
	const auto make = [](size_t buffers, size_t images, size_t samplers) {
		IR::ResourceSpecialization specialization;
		specialization.buffers.resize(buffers);
		specialization.images.resize(images);
		specialization.samplers.resize(samplers);
		return specialization;
	};
	constexpr size_t COUNT = 2;
	const auto       equal = make(COUNT, COUNT, COUNT);
	CHECK(!ProgramExceedsSpecialization(COUNT, COUNT, COUNT, equal));
	CHECK(ProgramExceedsSpecialization(COUNT + 1, COUNT, COUNT, equal));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT + 1, COUNT, equal));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT, COUNT + 1, equal));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT, COUNT, make(COUNT - 1, COUNT, COUNT)));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT, COUNT, make(COUNT, COUNT - 1, COUNT)));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT, COUNT, make(COUNT, COUNT, COUNT - 1)));
	CHECK(
	    !ProgramExceedsSpecialization(COUNT, COUNT, COUNT, make(COUNT + 1, COUNT + 1, COUNT + 1)));
	CHECK(!ProgramExceedsSpecialization(0, 0, 0, make(0, 0, 0)));
}

// A translation that lists another number of buffers than the prediction drops the program, as a
// wrong push start does. The prediction is made wrong by taking a buffer off the base candidate.
void TestResourceCountMismatch() {
	TempDirectory directory;
	const auto    eboot = BuildEboot(MainPrograms());
	const auto    big   = ScanEbootShaders(eboot).programs[0].code_hash;
	WorkerHooks   hooks;
	hooks.after_prediction = [&](uint64_t hash, std::vector<Candidate>& candidates) {
		auto& buffers = candidates.front().specialization.buffers;
		if (hash == big) buffers.erase(buffers.begin());
	};
	const auto request = MakeRequest(directory, eboot);
	const auto run     = Run(request, hooks);
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(run.result.resource_count_mismatches == 1 && run.result.push_start_mismatches == 0);
	CHECK(run.result.drops.size() == 1);
	if (run.result.drops.size() == 1) {
		CHECK(run.result.drops[0].reason == DropReason::ResourceCountMismatch);
		CHECK(run.result.drops[0].hash == big && run.result.drops[0].rank == 3);
	}
	// The big program's head is written first. Its base, which the hook broke, and the two
	// partials after it are dropped, and the other four candidates are written.
	CHECK(run.result.records_written == 4 && run.result.candidates_dropped == 3);
	CHECK(StartLines(run.log).size() == 5);
	bool logged = false;
	for (const auto& line: run.log) {
		logged = logged || line.ends_with("resource_count");
	}
	CHECK(logged);
	const auto log = ReadRealLog(request);
	CHECK(log.KindCount(ProgressKind::Drop) == 1 && log.DropsSinceSpawn().size() == 1);
	if (log.DropsSinceSpawn().size() == 1) {
		const auto& drop = log.DropsSinceSpawn()[0];
		CHECK(drop.reason == ProgressDropReason::ResourceCount && drop.rank_or_count == 3);
		CHECK(drop.hash == big);
	}
}

// A record whose compressed size is over the frame cap is dropped alone, with a log line and a
// count, and the other candidates of its program are still written. The cap is lowered through the
// hook to one byte under the largest record, so only the largest records trip it. The baseline is
// the same run with the cap at the largest record, which drops nothing, so a worker that dropped
// for another reason would show there too.
void TestOversizeRecordIsDroppedAlone() {
	TempDirectory directory;
	const auto    eboot            = BuildEboot(MainPrograms());
	const auto    baseline_request = MakeRequest(directory, eboot, "baseline");
	const auto    baseline         = Run(baseline_request);
	bool          exact            = false;
	const auto    all = ReadRecords(baseline.file, HeaderBytes(baseline_request), exact);
	CHECK(baseline.result.records_written == 7 && baseline.result.frame_too_large_drops == 0);
	size_t largest = 0;
	for (const auto& stored: all) {
		largest = std::max(largest, FrameStoreRecord(stored.raw).size() - sizeof(StoreFrame));
	}
	WorkerHooks at_cap;
	at_cap.max_frame_compressed_bytes = static_cast<uint32_t>(largest);
	const auto same                   = Run(MakeRequest(directory, eboot, "at_cap"), at_cap);
	CHECK(same.result.records_written == 7 && same.result.frame_too_large_drops == 0);

	WorkerHooks hooks;
	hooks.max_frame_compressed_bytes = static_cast<uint32_t>(largest - 1);
	const auto request               = MakeRequest(directory, eboot, "lowered");
	const auto run                   = Run(request, hooks);
	size_t     oversize              = 0;
	for (const auto& stored: all) {
		oversize += FrameStoreRecord(stored.raw).size() - sizeof(StoreFrame) > largest - 1;
	}
	CHECK(oversize >= 1 && oversize < all.size());
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(run.result.frame_too_large_drops == oversize && run.result.drops.size() == oversize);
	// Only the oversize candidates are gone. A drop of the rest of their program would take more.
	CHECK(run.result.records_written == all.size() - oversize);
	CHECK(run.result.candidates_dropped == oversize);
	CHECK(run.result.push_start_mismatches == 0 && run.result.resource_count_mismatches == 0);
	const auto records = ReadRecords(run.file, HeaderBytes(request), exact);
	CHECK(exact && records.size() == all.size() - oversize);
	for (const auto& drop: run.result.drops) {
		CHECK(drop.reason == DropReason::FrameTooLarge);
	}
	size_t logged = 0;
	for (const auto& line: run.log) {
		logged += line.ends_with("frame_too_large");
	}
	CHECK(logged == oversize);
	// Every candidate still gets its start line, since the size is known only after translating.
	CHECK(StartLines(run.log).size() == all.size());
	const auto log = ReadRealLog(request);
	CHECK(log.KindCount(ProgressKind::Drop) == oversize);
	CHECK(log.DropsSinceSpawn().size() == run.result.drops.size());
	for (size_t i = 0; i < log.DropsSinceSpawn().size() && i < run.result.drops.size(); i++) {
		const auto& drop = log.DropsSinceSpawn()[i];
		CHECK(drop.reason == ProgressDropReason::FrameTooLarge);
		CHECK(drop.rank_or_count == run.result.drops[i].rank &&
		      drop.hash == run.result.drops[i].hash);
	}
	// The run at the cap, which dropped nothing, has no drop line.
	CHECK(ReadRealLog(MakeRequest(directory, eboot, "at_cap")).KindCount(ProgressKind::Drop) == 0);
}

// Candidates with the given unbound sets, in the order given.
std::vector<Candidate> CandidatesWithNulls(const std::vector<std::vector<uint32_t>>& null_sets) {
	std::vector<Candidate> candidates(null_sets.size());
	for (size_t index = 0; index < null_sets.size(); index++) {
		candidates[index].nulls = null_sets[index];
	}
	return candidates;
}

// Whether the ranked candidates are exactly the expected ones, in order.
bool RankedIs(const std::vector<RankedCandidate>& ranked,
              const std::vector<RankedCandidate>& expected) {
	return std::ranges::equal(ranked, expected, [](const auto& left, const auto& right) {
		return left.program == right.program && left.candidate == right.candidate;
	});
}

// T1. P has 4 candidates and is wide, Q has 2 and is narrow, and R has one. Every head comes first
// with a wide base right after its head, then the wide partials, and last the narrow base. The
// baseline is P cut to 2 candidates, which makes it narrow, and its base then goes to the end.
void TestOrderHeadsBasesAndPartials() {
	const auto                        wide   = CandidatesWithNulls({{}, {0}, {1}, {0, 1}});
	const auto                        narrow = CandidatesWithNulls({{}, {0}});
	const auto                        lone   = CandidatesWithNulls({{}});
	const std::vector<OrderedProgram> programs {{10, wide}, {20, narrow}, {30, lone}};
	CHECK(RankedIs(OrderCandidates(programs),
	               {{0, 3}, {0, 0}, {1, 1}, {2, 0}, {0, 1}, {0, 2}, {1, 0}}));

	const auto                        cut = CandidatesWithNulls({{}, {0, 1}});
	const std::vector<OrderedProgram> baseline {{10, cut}, {20, narrow}, {30, lone}};
	CHECK(RankedIs(OrderCandidates(baseline), {{0, 1}, {1, 1}, {2, 0}, {0, 0}, {1, 0}}));
}

// T2. The wide partials are ordered by the size of their unbound set across all programs. S is the
// smaller program, so only the subset rank puts P's two buffer partials before S's single ones, and
// only a descending rank puts P's single ones after S's.
void TestOrderPartialsBySubsetSize() {
	const auto small = CandidatesWithNulls({{}, {0, 1, 2}, {0}, {1}});
	const auto large = CandidatesWithNulls({{}, {0, 1, 2}, {0, 1}, {0, 2}, {0}, {1}, {2}});
	const std::vector<OrderedProgram> programs {{10, small}, {20, large}};
	CHECK(RankedIs(
	    OrderCandidates(programs),
	    {{0, 1}, {0, 0}, {1, 1}, {1, 0}, {1, 2}, {1, 3}, {0, 2}, {0, 3}, {1, 4}, {1, 5}, {1, 6}}));
}

// T3. A Coupled like program has 2 candidates, all bound and all null. Its head is in pass 0 and
// its base is last. The baseline has a third candidate, which makes it wide and moves the base up
// to follow its head.
void TestOrderCoupledLikeProgram() {
	const auto                        coupled = CandidatesWithNulls({{}, {0, 1}});
	const auto                        lone    = CandidatesWithNulls({{}});
	const std::vector<OrderedProgram> programs {{10, coupled}, {20, lone}};
	CHECK(RankedIs(OrderCandidates(programs), {{0, 1}, {1, 0}, {0, 0}}));

	const auto                        wide = CandidatesWithNulls({{}, {0}, {0, 1}});
	const std::vector<OrderedProgram> baseline {{10, wide}, {20, lone}};
	CHECK(RankedIs(OrderCandidates(baseline), {{0, 2}, {0, 0}, {1, 0}, {0, 1}}));
}

// T4. A program with one candidate is built once, in pass 0. Treated as a narrow base it would come
// after B's variant instead.
void TestOrderLoneCandidateOnce() {
	const auto                        lone   = CandidatesWithNulls({{}});
	const auto                        narrow = CandidatesWithNulls({{}, {0}});
	const std::vector<OrderedProgram> programs {{10, lone}, {20, narrow}};
	const auto                        ranked = OrderCandidates(programs);
	CHECK(RankedIs(ranked, {{0, 0}, {1, 1}, {1, 0}}));
}

// T5. The head is the largest unbound set, and the first listed wins a tie. For {}, {0}, {1} that
// is {0}, and a head taken as the last index would be {1}.
void TestOrderHeadTieRule() {
	const auto trimmed = CandidatesWithNulls({{}, {0}, {1}});
	CHECK(HeadIndex(trimmed) == 1);
	CHECK(HeadIndex(CandidatesWithNulls({{}, {0, 1}})) == 1);
	CHECK(HeadIndex(CandidatesWithNulls({{0, 1}, {}, {0, 1}})) == 0);
	CHECK(HeadIndex(CandidatesWithNulls({{}})) == 0);
	const std::vector<OrderedProgram> programs {{10, trimmed}};
	CHECK(RankedIs(OrderCandidates(programs), {{0, 1}, {0, 0}, {0, 2}}));
}

// The key uses the size of the program, which every candidate of it shares. The first program is
// the larger and is listed first, so the program index cannot stand in for the size. The sizes
// decide the heads of pass 0, with each base after its own head, and the partials of pass 1 tie on
// subset rank so that the size alone orders them. A Candidate has no size of its own to use
// instead.
void TestOrderUsesProgramSize() {
	const auto                        wide = CandidatesWithNulls({{}, {0, 1}, {0}});
	const std::vector<OrderedProgram> programs {{30, wide}, {10, wide}};
	CHECK(RankedIs(OrderCandidates(programs), {{1, 1}, {1, 0}, {0, 1}, {0, 0}, {1, 2}, {0, 2}}));
}

// The pure order of candidates. The programs are listed out of size order so that neither the
// program index nor the candidate index can stand in for the size, and a program with one candidate
// shows that a lone candidate is built once, as a head. The heads go first by program size, then
// the narrow bases by program size. Two programs share a size, so only the program index orders
// them.
void TestOrderCandidatesPure() {
	const std::vector<Candidate> with_variant(2);
	std::vector<Candidate>       two = with_variant;
	two[1].nulls                     = {0};
	const std::vector<Candidate> lone(1);

	const std::vector<OrderedProgram>  programs {{30, two}, {10, lone}, {20, two}, {20, two}};
	const auto                         ranked = OrderCandidates(programs);
	const std::vector<RankedCandidate> expected {{1, 0}, {2, 1}, {3, 1}, {0, 1},
	                                             {2, 0}, {3, 0}, {0, 0}};
	CHECK(ranked.size() == expected.size());
	for (size_t i = 0; i < ranked.size() && i < expected.size(); i++) {
		CHECK(ranked[i].program == expected[i].program &&
		      ranked[i].candidate == expected[i].candidate);
	}
	CHECK(OrderCandidates({}).empty());

	// With the program index in the key every key is unique, so there is one right order
	// whatever the sort, and a stable sort of the indices by size is that order. Without the
	// index, programs of one size tie, and std::ranges::sort can then scramble them. A run this
	// long with sizes that cycle is partitioned rather than insertion sorted, which did scramble
	// them on MSVC, but a sort that happened to keep ties in order would let that mutation pass.
	constexpr size_t            TIED_PROGRAMS = 64;
	constexpr size_t            SIZE_CLASSES  = 5;
	constexpr size_t            SIZE_STRIDE   = 3;
	std::vector<OrderedProgram> tied;
	std::vector<uint32_t>       reference;
	for (uint32_t i = 0; i < TIED_PROGRAMS; i++) {
		tied.push_back({1 + (i * SIZE_STRIDE) % SIZE_CLASSES, lone});
		reference.push_back(i);
	}
	std::ranges::stable_sort(reference, [&](uint32_t left, uint32_t right) {
		return tied[left].code_dwords < tied[right].code_dwords;
	});
	const auto tied_ranked = OrderCandidates(tied);
	CHECK(tied_ranked.size() == TIED_PROGRAMS);
	for (size_t i = 0; i < tied_ranked.size() && i < reference.size(); i++) {
		CHECK(tied_ranked[i].program == reference[i] && tied_ranked[i].candidate == 0);
	}
}

} // namespace

int main() {
	TestOrderCandidatesPure();
	TestOrderHeadsBasesAndPartials();
	TestOrderPartialsBySubsetSize();
	TestOrderCoupledLikeProgram();
	TestOrderLoneCandidateOnce();
	TestOrderHeadTieRule();
	TestOrderUsesProgramSize();
	TestRequestFormat();
	TestRequestFile();
	TestGeneratedHeader();
	TestRecordsInRankOrder();
	TestProgressLog();
	TestStartRank();
	TestSkipHashes();
	TestFailedHashes();
	TestKeptAppends();
	TestVerifyUpToDate();
	TestVerifyRegenerates();
	TestProgramsLeftOut();
	TestPredictionLinePerProgram();
	TestPredictionCandidateCounts();
	TestPushStartTripwireOnRealMiss();
	TestPushStartTripwireDropsTheRest();
	TestKeptTruncatesTornBytes();
	TestOrderGuard();
	TestPredictLineComesFirst();
	TestRestartAfterDrop();
	TestResourceCountMismatch();
	TestProgramExceedsSpecialization();
	TestExtraSpecializationBufferIsKept();
	TestOversizeRecordIsDroppedAlone();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("shader store worker tests passed");
	return 0;
}
