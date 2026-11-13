#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_DIFFERENTIAL_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_DIFFERENTIAL_H

#include "common/firstRunReport.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

// A measurement that checks whether the submit scanner predicts what the command processor runs. It is
// off unless --scan-differential asks for it, and then it scans every submission on the thread that
// enqueues it, which costs that thread the walk. It changes nothing the guest sees.
namespace Libs::Graphics {

/// How many records past the cursor a record with an exact key is preferred in. The scan walks both arms
/// of a COND_EXEC and of a branch, so its list holds records the processor never runs, and an arm may
/// hold hundreds of dispatches. Finding the program has no bound at all, since a bound would turn one
/// long skipped arm into a run of false misses. Only the preference for an exact key is limited.
inline constexpr size_t DIFFERENTIAL_EXACT_WINDOW = 64;

/// One dispatch the scan predicted, reduced to what the comparison needs.
struct ScannedRecord {
	uint64_t program_address = 0;
	uint64_t key_hash        = 0;
	/// Whether the scan could not vouch for the program, see ScannedDispatch::uncertain_program.
	bool uncertain = false;
};

/// One dispatch the command processor ran.
struct ExecutedDispatch {
	uint64_t program_address = 0;
	uint64_t key_hash        = 0;
};

/// What the comparison made of one executed dispatch.
enum class DifferentialOutcome : uint8_t { Matched, KeyMismatch, Missing };

/// The verdict on one executed dispatch. scanned_key is 0 when it is Missing.
struct DifferentialVerdict {
	DifferentialOutcome outcome     = DifferentialOutcome::Missing;
	bool                uncertain   = false;
	uint64_t            scanned_key = 0;
};

/// Aligns the executed dispatches of one submission with its scanned records, one at a time. Each
/// executed dispatch takes the first scanned record at or after the cursor that has the same program
/// address, found through an index of each program's records, and the cursor moves past it. The records
/// it steps over stay unconsumed, which is how a skipped arm shows up as extra. Matching by ordinal
/// instead would drift by one for every record a skipped arm left in the list.
/// The one refinement is for records the scan was unsure of. If the first same-program record has a
/// different key, a later one with the exact key is taken instead, provided every same-program record
/// stepped over to reach it was uncertain and it lies within DIFFERENTIAL_EXACT_WINDOW records. A
/// certain record with the wrong key is a real mismatch and is never skipped to find a better one.
class DispatchMatcher {
public:
	/// `tail_uncertain` says the walk ended unable to vouch for the program, which covers what runs after the
	/// last record. A missing dispatch has no record of its own to carry uncertainty, so it takes that of the
	/// records beside the cursor, and past the last record it takes the tail flag as well.
	DispatchMatcher(std::vector<ScannedRecord> scanned, bool tail_uncertain);

	/// Classifies the next executed dispatch and consumes the record it took, if any.
	[[nodiscard]] DifferentialVerdict Match(const ExecutedDispatch& executed);

	/// The records no executed dispatch took, counted by certain and uncertain. Call it when the
	/// submission has ended.
	[[nodiscard]] Common::FirstRunReport::ScanOutcomeCount Unconsumed() const;

private:
	// Picks the record an executed dispatch takes among the candidates of its program, starting at the
	// first one at or after the cursor.
	[[nodiscard]] size_t PickCandidate(const std::vector<size_t>& candidates, size_t first,
	                                   uint64_t key_hash) const;
	// Whether a dispatch with no record is in doubt, judged from the records around the cursor.
	[[nodiscard]] bool MissingIsUncertain() const;

	std::vector<ScannedRecord> m_scanned;
	std::vector<bool>          m_consumed;
	// For each program address, the indexes of its records in ascending order.
	std::unordered_map<uint64_t, std::vector<size_t>> m_by_program;
	size_t                                            m_cursor = 0;
	bool                                              m_tail_uncertain;
};

/// The result of aligning a whole submission at once.
struct DifferentialClassification {
	Common::FirstRunReport::ScanOutcomeTotals totals;
};

/// Aligns the executed dispatches of one submission with its scanned records and counts the classes. It
/// is the whole comparison as a pure function, so tests need no renderer, and the live differential runs
/// the same matcher one dispatch at a time.
[[nodiscard]] DifferentialClassification ClassifyDispatches(std::span<const ScannedRecord>   scanned,
                                                            std::span<const ExecutedDispatch> executed,
                                                            bool tail_uncertain);

/// Adds one verdict to the totals.
void AddVerdict(Common::FirstRunReport::ScanOutcomeTotals& totals, const DifferentialVerdict& verdict);

/// A histogram of scan times in whole microseconds, read back as percentiles. A time past its last bucket
/// counts in that bucket, and the exact maximum is kept apart.
class ScanTimeHistogram {
public:
	/// The longest time, in microseconds, that has a bucket of its own.
	static constexpr uint64_t BUCKETS = 4096;

	/// Records one scan that took `microseconds`.
	void Add(uint64_t microseconds);
	/// The time that `fraction` of the scans took no longer than, or 0 with none. Nearest rank.
	[[nodiscard]] uint64_t Percentile(double fraction) const;
	/// The longest scan.
	[[nodiscard]] uint64_t Max() const { return m_max; }

private:
	std::vector<uint32_t> m_buckets;
	uint64_t              m_count = 0;
	uint64_t              m_max   = 0;
};

/// Scans each submission as it is enqueued and compares what the command processor then runs.
/// Enqueue threads call ScanSubmission and ResetShadow. The GPU thread calls OnCpDispatch and
/// FinishSubmission. Report threads call Snapshot. Scanning is serialized by one mutex, which keeps the
/// shadows whole but does not order scans against queueing. A caller that needs the scan order to be the
/// queue order must serialize the scan and the push itself, as GuestGpu::Enqueue does through its
/// ScanOrderGate. The records are guarded by a second mutex that no scan holds.
class ScanDifferential {
public:
	/// Makes the differential when `enabled`, and returns null otherwise, so a run without the flag
	/// allocates nothing for it.
	[[nodiscard]] static std::unique_ptr<ScanDifferential> Create(bool enabled, uint32_t queue_count,
	                                                              ScanMemoryReader reader,
	                                                              ScanLimits limits = {});

	ScanDifferential(uint32_t queue_count, ScanMemoryReader reader, ScanLimits limits);

	/// Scans a command list that is about to be queued and keeps its dispatches under the sequence
	/// number it returns. The first number of a queue is 1, so 0 means no scan.
	[[nodiscard]] uint64_t ScanSubmission(uint32_t queue, std::span<const uint32_t> commands);

	/// Forgets the compute registers of a queue, where the command processor forgets its own.
	void ResetShadow(uint32_t queue);

	/// Compares one dispatch the command processor is about to run with its submission's records. It
	/// takes the processor's own compute registers, with the wave size already set from the initiator as
	/// the processor sets it, and hashes them with the function the scanner uses. A dispatch with no
	/// program is left out, as the scan leaves it out.
	void OnCpDispatch(uint32_t queue, uint64_t seq, const HW::CsStageRegisters& cs, uint32_t initiator,
	                  bool async_queue);

	/// Ends a submission, which counts the records no dispatch took as extra. A submission that the
	/// processor blocked is not finished until it completes.
	void FinishSubmission(uint32_t queue, uint64_t seq);

	/// What has been counted so far.
	[[nodiscard]] Common::FirstRunReport::ScanDifferentialReport Snapshot() const;

	/// How many submissions are scanned and not yet finished, for tests.
	[[nodiscard]] size_t OpenSubmissions() const;

private:
	struct QueueState {
		CsShadow                                       shadow;
		uint64_t                                       next_seq = 1;
		std::unordered_map<uint64_t, DispatchMatcher>  open;
		ScanStats                                      stats;
		ScanTimeHistogram                              timing;
		uint64_t                                       cp_dispatches = 0;
		/// Reused by every scan of the queue, so a scan does not allocate it again.
		std::vector<ScannedDispatch> scratch;
	};

	void Record(uint32_t queue, uint64_t seq, DispatchMatcher matcher, const ScanStats& stats,
	            uint64_t microseconds);

	SubmitScanner m_scanner;
	// Held for a whole scan, so only enqueuing threads wait on it. It guards each queue's shadow and
	// scratch, and next_seq.
	std::mutex m_scan_mutex;
	// The walk's table and arena, which every scan uses under the scan mutex, so the arena exists once and
	// not for each thread that submits.
	ScanScratch m_scratch;
	// Guards everything else that a queue holds, and the totals and samples.
	mutable std::mutex                                     m_data_mutex;
	std::vector<QueueState>                                m_queues;
	Common::FirstRunReport::ScanOutcomeTotals              m_totals;
	uint64_t                                               m_untracked = 0;
	std::vector<Common::FirstRunReport::ScanDifferentialSample> m_samples;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_DIFFERENTIAL_H
