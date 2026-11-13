#ifndef GRAPHICS_SHADER_STOREGEN_SHADER_STORE_WORKER_H
#define GRAPHICS_SHADER_STOREGEN_SHADER_STORE_WORKER_H

#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/computeCompile.h"
#include "graphics/shader/storegen/computePredictor.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <vector>

// The worker that generates a game's compute shader store. It reads the eboot, predicts the modules
// the game will ask for, and translates each into a store record. It needs no graphics device, no
// window and no Config, so it runs in a child process of the emulator, where a crash or an exit in
// the recompiler cannot reach the emulator. Everything it writes goes to files, which are the
// output file of framed records and a progress log.
namespace Libs::Graphics::StoreGen {

/// The worker finished the whole ordering, or the part from the start rank.
constexpr int WORKER_EXIT_DONE = 0;
/// Verify mode found a finished file made from this eboot, with these host values and this
/// generator version, so nothing was written.
constexpr int WORKER_EXIT_UP_TO_DATE = 10;
/// The request could not be read, or its store header could not be decoded.
constexpr int WORKER_EXIT_BAD_REQUEST = 11;
/// The eboot could not be read whole.
constexpr int WORKER_EXIT_EBOOT_UNREADABLE = 12;
/// The output file or the progress log could not be opened or written.
constexpr int WORKER_EXIT_OUTPUT_FAILED = 13;
/// A restart expected another candidate ordering than this run computed, so the ranks of the old
/// run mean other candidates now. Nothing was written to the output file.
constexpr int WORKER_EXIT_ORDER_MISMATCH = 14;
/// A Stream worker could not connect to its pipe in time.
constexpr int WORKER_EXIT_PIPE_FAILED = 15;
/// A Stream worker found that the server end of its pipe is not the process that started it, so it
/// served nothing.
constexpr int WORKER_EXIT_PEER_MISMATCH = 16;
/// A Stream worker read a frame that is not a request it can decode, so the stream is out of step.
constexpr int WORKER_EXIT_PROTOCOL_ERROR = 17;
/// What the recompiler's EXIT leaves behind. DbgReport calls _Exit(321), which no worker code
/// chooses. Windows keeps the whole value, but POSIX keeps 8 bits of an exit status, so there it
/// reads as 65 (321 & 0xFF). Windows also gives 3 to abort and terminate, and 1 to an uncaught
/// exception, which is why no code above is below 10. Only DONE and UP_TO_DATE mean the worker did
/// not crash. A supervisor treats every other code as a failure and does not compare for 321. A
/// worker killed by a signal reports 128 plus the signal.
constexpr int WORKER_EXIT_RECOMPILER_EXIT = 321;

/// Why a program's remaining candidates were dropped.
enum class DropReason : uint8_t {
	/// The compiled module's push data start was not the one the predictor assumed, so the id the
	/// worker would store is not the id the runtime computes and the module would never be found.
	PushStartMismatch,
	/// The translated program lists more buffers, images or samplers than the specialization.
	ResourceCountMismatch,
	/// The candidate's record is over MAX_STORE_FRAME_COMPRESSED_BYTES, which no reader accepts.
	/// Only this candidate is dropped, since the size belongs to one module and not to its
	/// program, so a supervisor leaves this drop's hash out of the failed hashes.
	FrameTooLarge,
	/// The push data start was wrong on the head of a program with 2 or more candidates. Only the
	/// head is dropped, so the base and the partials are still built, and a supervisor leaves this
	/// drop's hash out of the failed hashes. The all-null is the candidate where specialization can
	/// make buffers non-live, and the head is built first, so losing the whole program would lose
	/// the base that the old order always built.
	HeadPushStartMismatch,
};

/// One program the worker stopped generating, and the evidence.
struct WorkerDrop {
	uint64_t   hash   = 0;
	uint64_t   rank   = 0;
	DropReason reason = DropReason::PushStartMismatch;
	/// PushStartMismatch only, the start the predictor assumed and the start the module has.
	uint32_t predicted_push = 0;
	uint32_t actual_push    = 0;
	/// HeadPushStartMismatch only, the id the runtime computes for the candidate, which uses the
	/// start the module has. A runtime miss with this id is the dropped head.
	uint64_t module_id = 0;
};

/// What a worker run did. The supervisor in another process reads the same figures from the last
/// line of the progress log.
struct WorkerResult {
	int           exit_code = WORKER_EXIT_DONE;
	EbootIdentity identity;

	/// Programs found by the scan, and what became of each.
	uint32_t programs_scanned   = 0;
	uint32_t programs_predicted = 0;
	/// Left out by the request, before prediction.
	uint32_t programs_skipped_by_request = 0;
	/// Programs that call GPU-selected functions, which the store does not hold.
	uint32_t programs_with_calls = 0;
	/// Programs whose descriptors the route K classifier could not name.
	uint32_t programs_unclassified = 0;
	/// Programs with no static key, no readable header slots or no readable sl00 metadata.
	uint32_t programs_unreadable = 0;
	/// Programs with a buffer or image outside the predictor's rules, or with no candidate at all.
	uint32_t programs_with_problems = 0;

	/// Candidates in the ordering, which is what the ranks count.
	uint64_t candidates              = 0;
	uint64_t candidates_before_start = 0;
	/// Candidates of programs the request listed as failed.
	uint64_t candidates_failed_earlier = 0;
	/// Candidates not built because an earlier candidate of the program tripped a check.
	uint64_t candidates_dropped = 0;
	uint64_t records_written    = 0;
	uint64_t record_bytes       = 0;

	/// Programs dropped by each check. Candidates whose record was over the frame cap are counted
	/// in frame_too_large_drops, and each is also counted in candidates_dropped.
	uint32_t                push_start_mismatches     = 0;
	uint32_t                resource_count_mismatches = 0;
	uint32_t                frame_too_large_drops     = 0;
	std::vector<WorkerDrop> drops;

	/// XXH3 of the candidate ids in rank order, and the same value the log's `order` line holds. A
	/// supervisor keeps it and passes it back as `expected_order` on a restart.
	uint64_t order_hash = 0;

	double   predict_ms           = 0;
	double   total_ms             = 0;
	double   worst_candidate_ms   = 0;
	uint64_t worst_candidate_rank = 0;
	uint64_t worst_candidate_hash = 0;
};

/// Calls a test can make the worker run at fixed points, so that it can look at the files while the
/// worker is between two steps. A real worker passes none.
struct WorkerHooks {
	/// Called with the rank of a candidate after its start line is in the log and before the
	/// candidate is translated.
	std::function<void(uint64_t rank)> before_candidate;
	/// Called with a program's code hash after its `predict` line is in the log and before its
	/// static key is built.
	std::function<void(uint64_t code_hash)> before_prediction;
	/// Called with a program's code hash and its candidates once they are predicted. A test uses it
	/// to make a prediction wrong on purpose.
	std::function<void(uint64_t code_hash, std::vector<Candidate>& candidates)> after_prediction;
	/// The largest compressed record the worker writes. A test lowers it.
	uint32_t max_frame_compressed_bytes = MAX_STORE_FRAME_COMPRESSED_BYTES;
};

/// Runs a worker request to the end, in this process.
///
/// The order of the candidates is the rule the supervisor and the tests rely on, and
/// OrderCandidates documents its key. The head of a program is the candidate with the most unbound
/// buffers. Pass 0 builds every head, and a wide program's base right after its head. Pass 1 builds
/// the other candidates of wide programs, the larger unbound sets of all programs first. Pass 2
/// builds the base of narrow programs. Within a pass the smaller program comes first, by the size
/// of its code and literals, then the program that comes first in the eboot. A rank is a position
/// in this ordering, so it does not depend on the start rank, and not on the request's failed
/// programs. It does depend on the request's skip hashes.
///
/// Before it predicts a program the worker appends `predict <hash>` to the progress log, and after
/// it, on every path that ends the program's prediction, `prediction <hash> <outcome>
/// <candidates>`. The outcome is predicted, calls, unclassified, unreadable or problems, matching
/// the result's programs_ counters, and the candidates are the program's count, which is 0 unless
/// it is predicted. A program the request skips gets neither line. After the ordering is known it
/// appends `order <count> <hash>`, and then, for each candidate from the start rank, `start <rank>
/// <hash>` before it translates the candidate. It then appends one framed record to the output
/// file. Every write goes straight to the operating system, so a crash is blamed on the last start
/// line. A kept file is only truncated and extended after the ordering matched the request's
/// `expected_order`. A candidate whose module has another push data start than predicted is not
/// written, the rest of its program is dropped, and the drop is counted and logged as `drop <rank>
/// <hash> ...`. A supervisor that restarts the worker adds the hash of each `drop` line to the
/// request's failed_hashes, so the restart does not build the program again. The exception is a
/// `frame_too_large` drop, which drops one candidate only and leaves the program's others to be
/// built. So is a `push_start_head` drop, which is a wrong push start on the head of a program with
/// 2 or more candidates. It ends with `id=<hex>`, the id the runtime would compute.
/// The last line of the log is `summary` with the counts, whenever the log was opened.
///
/// The recompiler can end the process through EXIT, so a caller that must survive that runs this
/// in a child process.
///
/// A request in Stream mode does none of this. It runs RunStreamWorker, as streamWorker.h
/// describes, and only the result's exit_code is set.
/// @return the result. Its exit_code is a WORKER_EXIT_ value.
WorkerResult RunShaderStoreWorker(const WorkerRequest& request, const WorkerHooks& hooks = {});

/// What a worker process does with its command line, which is to read the request file and run
/// it. This takes the place of a worker's main.
/// @return the process's exit code, WORKER_EXIT_BAD_REQUEST when the request is unreadable.
int RunShaderStoreWorkerFromFile(const std::filesystem::path& request_file);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_SHADER_STORE_WORKER_H
