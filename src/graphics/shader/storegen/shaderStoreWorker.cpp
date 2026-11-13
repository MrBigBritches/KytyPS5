#include "graphics/shader/storegen/shaderStoreWorker.h"

#include "common/file.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/appendFile.h"
#include "graphics/shader/storegen/candidateOrder.h"
#include "graphics/shader/storegen/computeCompile.h"
#include "graphics/shader/storegen/computePredictor.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/computeTranslation.h"
#include "graphics/shader/storegen/ebootShaderScan.h"
#include "graphics/shader/storegen/headerSlots.h"
#include "graphics/shader/storegen/resourceKeys.h"
#include "graphics/shader/storegen/sl00Metadata.h"
#include "graphics/shader/storegen/streamWorker.h"

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

using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start) {
	return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

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
	TranslationSetup  setup(key.input_info, key.user_data_count, scanned.code_hash, host);
	const ComputeUnit unit {scanned.code, scanned.code_hash, key.user_data_count, key.state};
	// The store holds only programs without GPU-selected calls, which the shared path refuses.
	const auto translated = TranslateComputeUnit(unit, setup);
	if (!translated) {
		return PredictKind::HasCalls;
	}
	const auto facts = ExtractResourceFacts(translated->program);
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

// The ordering that RunShaderStoreWorker documents, over the programs that were predicted.
std::vector<RankedCandidate> OrderPredicted(const std::vector<PredictedProgram>& programs) {
	std::vector<OrderedProgram> ordered;
	ordered.reserve(programs.size());
	for (const auto& program: programs) {
		ordered.push_back({program.scanned->code.size(), program.candidates});
	}
	return OrderCandidates(ordered);
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
	// The id the runtime computes for the module as it really compiled, set for a push start
	// mismatch only.
	uint64_t module_id = 0;
};

// Translates a candidate the way the runtime would, compiles it with its specialization and encodes
// the record, all through the compile path the Stream worker shares. The module id comes from the
// compiled module's own push data start, as the runtime computes it, so the record is found only
// when the prediction was right. A prediction that was wrong is reported and the record is left
// out.
BuiltCandidate BuildCandidate(const PredictedProgram& program, const Candidate& candidate,
                              const HostValues& host) {
	const auto&       scanned = *program.scanned;
	TranslationSetup  setup(program.key.input_info, program.key.user_data_count, scanned.code_hash,
	                        host);
	const ComputeUnit unit {scanned.code, scanned.code_hash, program.key.user_data_count,
	                        program.key.state};
	auto              built = BuildComputeUnit(unit, setup, candidate.specialization);
	if (built.status == ComputeBuildStatus::ResourceCountMismatch) {
		return {DropReason::ResourceCountMismatch, {}, 0};
	}
	if (built.push_start != candidate.push_start) {
		// The runtime's id carries the real start, so this is the id a miss on the candidate has.
		return {DropReason::PushStartMismatch, {}, built.push_start, built.id};
	}
	return {std::nullopt, std::move(built.record), built.push_start};
}

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
	bool DropHead(uint64_t rank, const PredictedProgram& program, const BuiltCandidate& built,
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

// Whether a candidate is the head of a program that has others besides it. A program with one
// candidate has nothing else to keep, so its head is dropped with the program as before.
bool IsHeadOfSeveral(const PredictedProgram& program, uint32_t candidate_index) {
	return program.candidates.size() >= 2 && HeadIndex(program.candidates) == candidate_index;
}

// Drops the head of a program whose push start was wrong, and nothing else of the program. The
// program is not added to m_dropped, so its base and partials are still built. Returns false when
// the log could not be written.
bool Worker::DropHead(uint64_t rank, const PredictedProgram& program, const BuiltCandidate& built,
                      const Candidate& candidate) {
	const auto hash = program.scanned->code_hash;
	m_result.push_start_mismatches++;
	m_result.drops.push_back({hash, rank, DropReason::HeadPushStartMismatch, candidate.push_start,
	                          built.actual_push, built.module_id});
	return m_progress.Line(
	    fmt::format("drop {} {:016x} push_start_head predicted={} actual={} id={:016x}", rank, hash,
	                candidate.push_start, built.actual_push, built.module_id));
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
		if (*built.drop == DropReason::PushStartMismatch &&
		    IsHeadOfSeveral(program, entry.candidate)) {
			return DropHead(rank, program, built, candidate);
		}
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
	const auto ranked   = OrderPredicted(m_programs);
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

WorkerResult RunShaderStoreWorker(const WorkerRequest& request, const WorkerHooks& hooks) {
	WorkerResult result;
	// A Stream request serves a pipe and has none of the Generate steps, so it leaves every
	// Generate figure at zero.
	if (request.mode == WorkerMode::Stream) {
		result.exit_code = RunStreamWorker(request).exit_code;
		return result;
	}
	result.exit_code = Worker(request, result, hooks).Run();
	return result;
}

int RunShaderStoreWorkerFromFile(const std::filesystem::path& request_file) {
	const auto request = ReadWorkerRequestFile(request_file);
	return request ? RunShaderStoreWorker(*request).exit_code : WORKER_EXIT_BAD_REQUEST;
}

} // namespace Libs::Graphics::StoreGen
