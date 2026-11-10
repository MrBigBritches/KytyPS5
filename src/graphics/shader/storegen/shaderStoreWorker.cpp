#include "graphics/shader/storegen/shaderStoreWorker.h"

#include "common/file.h"
#include "graphics/shader/moduleId.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/computePredictor.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/ebootShaderScan.h"
#include "graphics/shader/storegen/headerSlots.h"
#include "graphics/shader/storegen/resourceKeys.h"
#include "graphics/shader/storegen/sl00Metadata.h"

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <xxhash.h>

namespace Libs::Graphics::StoreGen {

namespace {

namespace IR = ShaderRecompiler::IR;

using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start) {
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// The values of the store header that translation depends on.
struct HostValues {
	bool                             structured_jump_tables = false;
	ShaderRecompiler::Spirv::Profile profile;
	uint32_t                         max_shared_bytes = 0;
};

// Whether every flag byte of a store header is 0 or 1. The header is read once more as plain bytes
// in the order the fields are written, which is the build stamp, the jump table flag, the profile
// (three flags, the device clock period and four flags) and the shared memory limit. A flag read
// into a bool from any other byte value would be undefined behaviour. If the profile gains a
// field this read comes up short and the header is refused, which fails every test at once.
bool HeaderFlagsAreBooleans(std::span<const uint8_t> bytes) {
	IR::Reader reader {bytes};
	uint64_t   stamp       = 0;
	uint8_t    jump_tables = 0;
	uint8_t    profile_flags[7] {};
	uint32_t   clock_period = 0;
	uint32_t   shared       = 0;
	IR::IoAll(reader, stamp, jump_tables, profile_flags[0], profile_flags[1], profile_flags[2],
	          clock_period, profile_flags[3], profile_flags[4], profile_flags[5], profile_flags[6],
	          shared);
	bool booleans = jump_tables <= 1;
	for (const auto flag: profile_flags) {
		booleans = booleans && flag <= 1;
	}
	return reader.ok && reader.bytes.empty() && booleans;
}

// Reads the store header the way PipelineCache writes it. The build stamp comes first and does not
// affect translation.
std::optional<HostValues> DecodeHostHeader(std::span<const uint8_t> bytes) {
	if (!HeaderFlagsAreBooleans(bytes)) {
		return std::nullopt;
	}
	IR::Reader reader {bytes};
	uint64_t   stamp = 0;
	HostValues host;
	IR::IoAll(reader, stamp, host.structured_jump_tables, host.profile, host.max_shared_bytes);
	if (!reader.ok || !reader.bytes.empty()) {
		return std::nullopt;
	}
	return host;
}

// The translation options of one program, set as the runtime sets them for a compute program. The
// options point into this object, so it stays where it is.
class TranslationSetup {
public:
	TranslationSetup(const ComputeStaticKey& key, uint64_t code_hash, const HostValues& host)
	    : m_info(key.input_info), m_user_data(key.user_data_count, 0u) {
		m_options.wave_size              = m_info.wave_size;
		m_options.stage                  = ShaderType::Compute;
		m_options.input_info.compute     = &m_info;
		m_options.shader_hash            = code_hash;
		m_options.dump_ir                = false;
		m_options.structured_jump_tables = host.structured_jump_tables;
		m_options.user_data              = m_user_data;
		m_options.profile                = &host.profile;
	}
	TranslationSetup(const TranslationSetup&)            = delete;
	TranslationSetup& operator=(const TranslationSetup&) = delete;

	ShaderRecompiler::CompileOptions& Options() { return m_options; }

private:
	ShaderComputeInputInfo           m_info;
	std::vector<uint32_t>            m_user_data;
	ShaderRecompiler::CompileOptions m_options;
};

// How far a program got toward having candidates.
enum class PredictKind {
	Predicted,
	HasCalls,
	Unclassified,
	Unreadable,
	Problems,
};

// The word a `prediction` line gives for what became of a program.
const char* PredictOutcomeWord(PredictKind kind) {
	switch (kind) {
		case PredictKind::Predicted: return "predicted";
		case PredictKind::HasCalls: return "calls";
		case PredictKind::Unclassified: return "unclassified";
		case PredictKind::Unreadable: return "unreadable";
		case PredictKind::Problems: return "problems";
	}
	return "";
}

// Translates a program once to list its resources, then predicts its candidates.
PredictKind PredictProgram(const ScannedProgram& scanned, const ComputeStaticKey& key,
                           const HostValues& host, std::vector<Candidate>& candidates) {
	TranslationSetup setup(key, scanned.code_hash, host);
	const auto       source = ShaderRecompiler::PrepareShaderSource(scanned.code, setup.Options());
	// The store holds only programs without GPU-selected calls.
	if (source.unsupported_call_pc != UINT32_MAX || !source.calls.empty()) {
		return PredictKind::HasCalls;
	}
	auto       translated = ShaderRecompiler::TranslateProgram(source.decoded, setup.Options());
	const auto facts      = ExtractResourceFacts(translated.program);
	if (!facts) {
		return PredictKind::Unclassified;
	}
	const auto slots   = DecodeHeaderSlots(scanned.header);
	const auto records = DecodeSl00Resources(scanned.metadata);
	if (!slots || !records) {
		return PredictKind::Unreadable;
	}
	auto prediction = PredictCandidates({scanned.code_hash, key.user_data_count, key.state, *facts,
	                                     *slots, *records, NullPolicy::Subsets});
	candidates      = std::move(prediction.candidates);
	return candidates.empty() ? PredictKind::Problems : PredictKind::Predicted;
}

// A program with candidates to build.
struct PredictedProgram {
	const ScannedProgram*  scanned = nullptr;
	ComputeStaticKey       key;
	std::vector<Candidate> candidates;
	// The request says this program crashed before, so its ranks stay and it is not built.
	bool failed_earlier = false;
};

// One position in the ordering.
struct RankedCandidate {
	uint32_t program   = 0;
	uint32_t candidate = 0;
};

// The ordering that RunShaderStoreWorker documents.
std::vector<RankedCandidate> OrderCandidates(const std::vector<PredictedProgram>& programs) {
	std::vector<RankedCandidate> ranked;
	for (uint32_t program = 0; program < programs.size(); program++) {
		for (uint32_t candidate = 0; candidate < programs[program].candidates.size(); candidate++) {
			ranked.push_back({program, candidate});
		}
	}
	const auto sort_key = [&](const RankedCandidate& entry) {
		const auto& program = programs[entry.program];
		const bool  variant = !program.candidates[entry.candidate].nulls.empty();
		return std::make_tuple(variant, program.scanned->code.size(), entry.program,
		                       entry.candidate);
	};
	std::ranges::sort(ranked, [&](const auto& left, const auto& right) {
		return sort_key(left) < sort_key(right);
	});
	return ranked;
}

// XXH3 of the candidate ids in rank order, which names the ordering. A restart compares it to be
// sure that a rank still means the candidate it meant.
uint64_t OrderHash(const std::vector<PredictedProgram>& programs,
                   const std::vector<RankedCandidate>&  ranked) {
	std::vector<uint64_t> ids;
	ids.reserve(ranked.size());
	for (const auto& entry: ranked) {
		ids.push_back(programs[entry.program].candidates[entry.candidate].id);
	}
	return XXH3_64bits(ids.data(), ids.size() * sizeof(uint64_t));
}

// The result of building one candidate.
struct BuiltCandidate {
	std::optional<DropReason> drop;
	std::vector<uint8_t>      record;
	uint32_t                  actual_push = 0;
};

// Translates a candidate the way the runtime would, compiles it with its specialization and encodes
// the record. The module id comes from the compiled module's own push data start, as the runtime
// computes it, so the record is found only when the prediction was right. A prediction that was
// wrong is reported and the record is left out.
BuiltCandidate BuildCandidate(const PredictedProgram& program, const Candidate& candidate,
                              const HostValues& host) {
	const auto&      scanned = *program.scanned;
	TranslationSetup setup(program.key, scanned.code_hash, host);
	const auto       source = ShaderRecompiler::PrepareShaderSource(scanned.code, setup.Options());
	auto       translated   = ShaderRecompiler::TranslateProgram(source.decoded, setup.Options());
	const auto plan         = IR::ExtractResourcePlan(translated.program);
	// A specialization may have more entries than the program, since indirect children are appended
	// to it, but never fewer. The check is the one ApplyResourceSpecialization makes with EXIT.
	const auto& info_before = translated.program.info;
	if (ProgramExceedsSpecialization(info_before.buffers.size(), info_before.images.size(),
	                                 info_before.samplers.size(), candidate.specialization)) {
		return {DropReason::ResourceCountMismatch, {}, 0};
	}
	auto       result = ShaderRecompiler::CompileProgram(std::move(translated), setup.Options(),
	                                                     candidate.specialization, 0);
	const auto info   = std::move(result.program).TakeCompiledInfo();
	const auto push   = info.bindings.push_data_start_dword;
	if (push != candidate.push_start) {
		return {DropReason::PushStartMismatch, {}, push};
	}
	const auto id =
	    ShaderModuleId(ShaderType::Compute, scanned.code_hash, program.key.user_data_count,
	                   program.key.state, push, candidate.specialization, {});
	return {std::nullopt,
	        EncodeModuleRecord({.stage           = ShaderType::Compute,
	                            .hash            = scanned.code_hash,
	                            .user_data_count = program.key.user_data_count,
	                            .static_state    = program.key.state,
	                            .plan            = plan,
	                            .id              = id,
	                            .info            = info,
	                            .specialization  = candidate.specialization,
	                            .vertex_formats  = {},
	                            .rect_parameters = {},
	                            .mesh_task       = false,
	                            .spirv           = result.spirv}),
	        push};
}

// A file that is appended to. Common::File writes straight to the operating system on every
// platform, so a write is visible to another process as soon as it returns and a crash of this
// process loses nothing written before it. A flush would only add FlushFileBuffers or fsync, which
// protects against power loss, and this file can be generated again. The start line test reads
// the log through another handle, so it would catch a worker that held its lines back.
class AppendFile {
public:
	// Opens a log, which is appended to when it exists.
	bool OpenAppend(const std::filesystem::path& path) {
		Common::File::CreateDirectories(path.parent_path());
		if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
			return m_file.Seek(m_file.Size());
		}
		return m_file.Create(path);
	}

	// Starts the output. With nothing kept it is a new file that begins with the header. Otherwise
	// the first `kept` bytes stay, after a check that they begin with the same header, and the
	// writes go after them.
	bool OpenOutput(const std::filesystem::path& path, std::span<const uint8_t> header,
	                uint64_t kept) {
		Common::File::CreateDirectories(path.parent_path());
		if (kept == 0) {
			return m_file.Create(path) && Append(header);
		}
		if (!m_file.Open(path, Common::File::Mode::ReadWrite) || kept < header.size() ||
		    m_file.Size() < kept) {
			return false;
		}
		std::vector<uint8_t> existing(header.size());
		m_file.Read(existing.data(), static_cast<uint32_t>(existing.size()));
		return std::ranges::equal(existing, header) && m_file.Truncate(kept) && m_file.Seek(kept);
	}

	// Writes bytes. Returns false when they were not all written.
	bool Append(std::span<const uint8_t> bytes) {
		uint32_t written = 0;
		m_file.Write(bytes.data(), static_cast<uint32_t>(bytes.size()), &written);
		return written == bytes.size();
	}

	// Writes one text line.
	bool Line(const std::string& text) {
		const auto line = text + "\n";
		return Append({reinterpret_cast<const uint8_t*>(line.data()), line.size()});
	}

private:
	Common::File m_file;
};

// Whether the finished file begins with exactly the header this run would write.
bool FileBeginsWith(const std::filesystem::path& path, std::span<const uint8_t> header) {
	Common::File file;
	if (!file.Open(path, Common::File::Mode::Read) || file.Size() < header.size()) {
		return false;
	}
	std::vector<uint8_t> existing(header.size());
	file.Read(existing.data(), static_cast<uint32_t>(existing.size()));
	return std::ranges::equal(existing, header);
}

// One run of a worker, in the order of the steps.
class Worker {
public:
	Worker(const WorkerRequest& request, WorkerResult& result, const WorkerHooks& hooks)
	    : m_request(request), m_result(result), m_hooks(hooks) {}

	// Runs every step and returns the exit code. The summary line closes a log that was opened.
	int Run() {
		const auto start  = Clock::now();
		const int  code   = Execute();
		m_result.total_ms = MillisecondsSince(start);
		if (m_log_open) {
			Summarize(code);
		}
		return code;
	}

private:
	int  Execute();
	int  Generate(const ScanResult& scan, const HostValues& host, std::span<const uint8_t> header);
	bool LoadScan(ScanResult& scan, uint64_t& eboot_bytes);
	bool Predict(const ScanResult& scan, const HostValues& host);
	void CountPrediction(PredictKind kind, uint64_t code_hash, std::vector<Candidate>& candidates);
	bool LogPrediction(uint64_t code_hash, PredictKind kind, size_t candidates);
	void BuildRanks(const std::vector<RankedCandidate>& ranked, const HostValues& host, int& code);
	bool BuildOne(uint64_t rank, const RankedCandidate& entry, const HostValues& host);
	bool Drop(uint64_t rank, const PredictedProgram& program, const BuiltCandidate& built,
	          const Candidate& candidate);
	bool DropOversizeRecord(uint64_t rank, const PredictedProgram& program,
	                        const Candidate& candidate);
	void Summarize(int code);

	const WorkerRequest&          m_request;
	WorkerResult&                 m_result;
	const WorkerHooks&            m_hooks;
	AppendFile                    m_progress;
	AppendFile                    m_output;
	bool                          m_log_open = false;
	std::vector<PredictedProgram> m_programs;
	std::unordered_set<uint64_t>  m_dropped;
};

// Reads the eboot whole and scans it. The path may name a member of an archive. The eboot's bytes
// are freed before this returns, since the scan keeps its own copies.
bool Worker::LoadScan(ScanResult& scan, uint64_t& eboot_bytes) {
	Common::File file;
	if (!file.Open(PathFromUtf8(m_request.eboot_path), Common::File::Mode::Read)) {
		return false;
	}
	const uint64_t size = file.Size();
	if (size > UINT32_MAX) {
		return false;
	}
	std::vector<uint8_t> eboot(size);
	uint32_t             read = 0;
	file.Read(eboot.data(), static_cast<uint32_t>(size), &read);
	if (read != size) {
		return false;
	}
	scan        = ScanEbootShaders(eboot);
	eboot_bytes = size;
	return true;
}

// Predicts every program that the request does not leave out, and counts the ones that give none.
// Each program is logged first, since translating it for the prediction can end the process.
bool Worker::Predict(const ScanResult& scan, const HostValues& host) {
	const std::unordered_set<uint64_t> skipped(m_request.skip_hashes.begin(),
	                                           m_request.skip_hashes.end());
	const std::unordered_set<uint64_t> failed(m_request.failed_hashes.begin(),
	                                          m_request.failed_hashes.end());
	const ComputeKeyDevice             device {m_request.host_subgroup_size, host.max_shared_bytes};
	for (const auto& scanned: scan.programs) {
		if (skipped.contains(scanned.code_hash)) {
			m_result.programs_skipped_by_request++;
			continue;
		}
		// Logged before anything of the program runs, so a crash is blamed on it.
		if (!m_progress.Line(fmt::format("predict {:016x}", scanned.code_hash))) {
			return false;
		}
		if (m_hooks.before_prediction) {
			m_hooks.before_prediction(scanned.code_hash);
		}
		const auto key = BuildComputeStaticKey(scanned.header, device, MOTOR_DISPATCH);
		if (!key) {
			// No key means no prediction was tried, but the program still gets its line.
			m_result.programs_unreadable++;
			if (!LogPrediction(scanned.code_hash, PredictKind::Unreadable, 0)) {
				return false;
			}
			continue;
		}
		PredictedProgram program {&scanned, *key, {}, failed.contains(scanned.code_hash)};
		const auto       kind = PredictProgram(scanned, *key, host, program.candidates);
		CountPrediction(kind, scanned.code_hash, program.candidates);
		// Only a predicted program has candidates to report, whatever a rejection left behind.
		const size_t candidates = kind == PredictKind::Predicted ? program.candidates.size() : 0;
		if (!LogPrediction(scanned.code_hash, kind, candidates)) {
			return false;
		}
		if (kind == PredictKind::Predicted) {
			m_programs.push_back(std::move(program));
		}
	}
	return true;
}

// Counts what became of a predicted program in the result, and gives a test its hook.
void Worker::CountPrediction(PredictKind kind, uint64_t code_hash,
                             std::vector<Candidate>& candidates) {
	switch (kind) {
		case PredictKind::Predicted:
			m_result.programs_predicted++;
			if (m_hooks.after_prediction) {
				m_hooks.after_prediction(code_hash, candidates);
			}
			break;
		case PredictKind::HasCalls: m_result.programs_with_calls++; break;
		case PredictKind::Unclassified: m_result.programs_unclassified++; break;
		case PredictKind::Unreadable: m_result.programs_unreadable++; break;
		case PredictKind::Problems: m_result.programs_with_problems++; break;
	}
}

// Writes the line that says what came of a program's prediction. It goes in after the hook above,
// so a crash in either still leaves `predict` as the last line. Returns false when the log could
// not be written.
bool Worker::LogPrediction(uint64_t code_hash, PredictKind kind, size_t candidates) {
	return m_progress.Line(
	    fmt::format("prediction {:016x} {} {}", code_hash, PredictOutcomeWord(kind), candidates));
}

// Records why a program's remaining candidates are dropped, in the result and in the log. Returns
// false when the log could not be written.
bool Worker::Drop(uint64_t rank, const PredictedProgram& program, const BuiltCandidate& built,
                  const Candidate& candidate) {
	const auto hash = program.scanned->code_hash;
	WorkerDrop drop {hash, rank, *built.drop, candidate.push_start, built.actual_push};
	bool       logged = false;
	if (*built.drop == DropReason::PushStartMismatch) {
		m_result.push_start_mismatches++;
		logged = m_progress.Line(fmt::format("drop {} {:016x} push_start predicted={} actual={}",
		                                     rank, hash, drop.predicted_push, drop.actual_push));
	} else {
		m_result.resource_count_mismatches++;
		logged = m_progress.Line(fmt::format("drop {} {:016x} resource_count", rank, hash));
	}
	m_result.drops.push_back(drop);
	m_dropped.insert(hash);
	return logged;
}

// Drops one candidate whose record no reader of the file would accept. Unlike the other drops it
// does not leave out the rest of the program, since the size belongs to this module alone. Returns
// false when the log could not be written.
bool Worker::DropOversizeRecord(uint64_t rank, const PredictedProgram& program,
                                const Candidate& candidate) {
	const auto hash = program.scanned->code_hash;
	m_result.candidates_dropped++;
	m_result.frame_too_large_drops++;
	m_result.drops.push_back({hash, rank, DropReason::FrameTooLarge, candidate.push_start, 0});
	return m_progress.Line(fmt::format("drop {} {:016x} frame_too_large", rank, hash));
}

// Builds one candidate and appends its record. Returns false when the output or the log could not
// be written.
bool Worker::BuildOne(uint64_t rank, const RankedCandidate& entry, const HostValues& host) {
	const auto& program   = m_programs[entry.program];
	const auto& candidate = program.candidates[entry.candidate];
	const auto  hash      = program.scanned->code_hash;
	// The line goes in first, so a crash while translating is blamed on this rank.
	if (!m_progress.Line(fmt::format("start {} {:016x}", rank, hash))) {
		return false;
	}
	if (m_hooks.before_candidate) {
		m_hooks.before_candidate(rank);
	}
	const auto begin = Clock::now();
	auto       built = BuildCandidate(program, candidate, host);
	const auto ms    = MillisecondsSince(begin);
	if (ms > m_result.worst_candidate_ms) {
		m_result.worst_candidate_ms   = ms;
		m_result.worst_candidate_rank = rank;
		m_result.worst_candidate_hash = hash;
	}
	if (built.drop) {
		m_result.candidates_dropped++;
		return Drop(rank, program, built, candidate);
	}
	const auto framed = FrameStoreRecord(built.record);
	if (framed.size() - sizeof(StoreFrame) > m_hooks.max_frame_compressed_bytes) {
		return DropOversizeRecord(rank, program, candidate);
	}
	if (!m_output.Append(framed)) {
		return false;
	}
	m_result.records_written++;
	m_result.record_bytes += framed.size();
	return true;
}

// Builds the candidates from the start rank on, in rank order.
void Worker::BuildRanks(const std::vector<RankedCandidate>& ranked, const HostValues& host,
                        int& code) {
	for (uint64_t rank = 0; rank < ranked.size() && code == WORKER_EXIT_DONE; rank++) {
		const auto& program = m_programs[ranked[rank].program];
		if (rank < m_request.start_rank) {
			m_result.candidates_before_start++;
		} else if (program.failed_earlier) {
			m_result.candidates_failed_earlier++;
		} else if (m_dropped.contains(program.scanned->code_hash)) {
			m_result.candidates_dropped++;
		} else if (!BuildOne(rank, ranked[rank], host)) {
			code = WORKER_EXIT_OUTPUT_FAILED;
		}
	}
}

// Writes the counts as the log's last line. Nothing can be done about a log that fails to write
// at this point, so it is best effort.
void Worker::Summarize(int code) {
	m_progress.Line(fmt::format(
	    "summary exit={} scanned={} predicted={} skipped_by_request={} candidates={} written={} "
	    "record_bytes={} dropped={} failed_earlier={} before_start={} push_start_mismatches={} "
	    "resource_count_mismatches={} frame_too_large_drops={} with_calls={} unclassified={} "
	    "unreadable={} with_problems={} "
	    "worst_ms={:.0f} worst_rank={} worst_hash={:016x} predict_ms={:.0f} total_ms={:.0f} "
	    "order={:016x}",
	    code, m_result.programs_scanned, m_result.programs_predicted,
	    m_result.programs_skipped_by_request, m_result.candidates, m_result.records_written,
	    m_result.record_bytes, m_result.candidates_dropped, m_result.candidates_failed_earlier,
	    m_result.candidates_before_start, m_result.push_start_mismatches,
	    m_result.resource_count_mismatches, m_result.frame_too_large_drops,
	    m_result.programs_with_calls, m_result.programs_unclassified, m_result.programs_unreadable,
	    m_result.programs_with_problems, m_result.worst_candidate_ms, m_result.worst_candidate_rank,
	    m_result.worst_candidate_hash, m_result.predict_ms, m_result.total_ms,
	    m_result.order_hash));
}

// The steps that need only the request and the eboot, up to the first one that fails.
int Worker::Execute() {
	const auto host = DecodeHostHeader(m_request.store_header);
	if (!host || m_request.host_subgroup_size == 0) {
		return WORKER_EXIT_BAD_REQUEST;
	}
	ScanResult scan;
	uint64_t   eboot_bytes = 0;
	if (!LoadScan(scan, eboot_bytes)) {
		return WORKER_EXIT_EBOOT_UNREADABLE;
	}
	m_result.programs_scanned = static_cast<uint32_t>(scan.programs.size());
	m_result.identity         = ComputeEbootIdentity(eboot_bytes, scan.programs);
	const auto header = EncodeGeneratedHeader(m_request.store_header, m_request.generator_version,
	                                          m_result.identity);
	// An up to date file is left as it is, and nothing is opened.
	if (m_request.mode == WorkerMode::Verify &&
	    FileBeginsWith(PathFromUtf8(m_request.generated_path), header)) {
		return WORKER_EXIT_UP_TO_DATE;
	}
	if (!m_progress.OpenAppend(PathFromUtf8(m_request.progress_path))) {
		return WORKER_EXIT_OUTPUT_FAILED;
	}
	m_log_open = true;
	return Generate(scan, *host, header);
}

// Predicts, orders and builds. The output file is not opened until the ordering is known, so a
// restart that expected another ordering leaves the kept bytes as they are.
int Worker::Generate(const ScanResult& scan, const HostValues& host,
                     std::span<const uint8_t> header) {
	const auto predict_start = Clock::now();
	if (!Predict(scan, host)) {
		return WORKER_EXIT_OUTPUT_FAILED;
	}
	m_result.predict_ms = MillisecondsSince(predict_start);
	const auto ranked   = OrderCandidates(m_programs);
	m_result.candidates = ranked.size();
	m_result.order_hash = OrderHash(m_programs, ranked);
	if (!m_progress.Line(fmt::format("order {} {:016x}", ranked.size(), m_result.order_hash))) {
		return WORKER_EXIT_OUTPUT_FAILED;
	}
	// A restart is a request that keeps bytes or starts past rank 0.
	if ((m_request.kept > 0 || m_request.start_rank > 0) && m_request.expected_order != 0 &&
	    m_request.expected_order != m_result.order_hash) {
		return WORKER_EXIT_ORDER_MISMATCH;
	}
	if (!m_output.OpenOutput(PathFromUtf8(m_request.output_path), header, m_request.kept)) {
		return WORKER_EXIT_OUTPUT_FAILED;
	}
	int code = WORKER_EXIT_DONE;
	BuildRanks(ranked, host, code);
	return code;
}

} // namespace

bool ProgramExceedsSpecialization(size_t buffers, size_t images, size_t samplers,
                                  const IR::ResourceSpecialization& specialization) {
	return buffers > specialization.buffers.size() || images > specialization.images.size() ||
	       samplers > specialization.samplers.size();
}

WorkerResult RunShaderStoreWorker(const WorkerRequest& request, const WorkerHooks& hooks) {
	WorkerResult result;
	result.exit_code = Worker(request, result, hooks).Run();
	return result;
}

int RunShaderStoreWorkerFromFile(const std::filesystem::path& request_file) {
	const auto request = ReadWorkerRequestFile(request_file);
	return request ? RunShaderStoreWorker(*request).exit_code : WORKER_EXIT_BAD_REQUEST;
}

} // namespace Libs::Graphics::StoreGen
