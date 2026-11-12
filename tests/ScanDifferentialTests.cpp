// Tests the scan differential, which checks whether the submit scanner predicts what the command
// processor runs. The alignment is a pure function, so most checks need no scan at all. The live checks
// scan a real stream and feed the processor side registers that were decoded on their own. Each check that
// proves a behaviour has a baseline that differs only in that behaviour, as the comments say.
#include "common/firstRunReport.h"
#include "graphics/guest_gpu/command_processor/scanDifferential.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/guest_gpu/csRegisters.h"
#include "graphics/guest_gpu/pm4.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

namespace {

using namespace Libs::Graphics;
using Common::FirstRunReport::ScanDifferentialReport;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Stream = std::vector<uint32_t>;

constexpr uint64_t PROGRAM_A = 0x0000004567890100ull;
constexpr uint64_t PROGRAM_B = 0x0000004567891100ull;
constexpr uint64_t PROGRAM_C = 0x0000004567892100ull;
constexpr uint64_t KEY_ONE   = 0x1111111111111111ull;
constexpr uint64_t KEY_TWO   = 0x2222222222222222ull;
constexpr uint64_t KEY_THREE = 0x3333333333333333ull;
// A program address no scanned record has.
constexpr uint64_t DEAD_PROGRAM = 0x0000004567893100ull;
// The distance between two program addresses a test makes up, which the register encoding keeps whole.
constexpr uint64_t PROGRAM_STRIDE = 0x100;
// The first payload word of a COND_EXEC, which the walk ignores.
constexpr uint32_t COND_EXEC_FIRST_PAYLOAD = 0x1000u;
constexpr uint32_t USER_SGPRS        = 2;
constexpr uint32_t RSRC2_USER_SHIFT  = 1;
constexpr uint32_t RSRC2_TGID_X_EN   = 1u << 7u;
constexpr uint32_t THREADS_X         = 64;
constexpr uint32_t TEST_QUEUE        = 3;
constexpr uint32_t QUEUE_COUNT       = 8;

ScannedRecord Record(uint64_t program, uint64_t key, bool uncertain = false) {
	return {program, key, uncertain};
}

ExecutedDispatch Executed(uint64_t program, uint64_t key) {
	return {program, key};
}

// Both arms of a skipped region are in the scanned list, and the processor ran only one. The baseline
// executes every record, which leaves nothing extra, so the extra count is caused by the skipped record
// and by nothing else. Matching by ordinal would pair the second executed dispatch with the skipped
// record, so this check fails for that matcher.
void TestSupersetWithSkippedArms() {
	// Each program has its own key, so a matcher that paired records by position would read the second
	// executed dispatch against the skipped record as a key mismatch and not as a lucky match.
	const std::vector<ScannedRecord> scanned {Record(PROGRAM_A, KEY_ONE), Record(PROGRAM_B, KEY_TWO),
	                                          Record(PROGRAM_C, KEY_THREE)};
	const std::vector<ExecutedDispatch> skipped {Executed(PROGRAM_A, KEY_ONE), Executed(PROGRAM_C, KEY_THREE)};
	const auto result = ClassifyDispatches(scanned, skipped, false);
	CHECK(result.totals.matched.Total() == 2);
	CHECK(result.totals.extra.Total() == 1);
	CHECK(result.totals.key_mismatch.Total() == 0 && result.totals.missing.Total() == 0);
	CHECK(result.totals.MismatchRate() == 0.0);

	const std::vector<ExecutedDispatch> all {Executed(PROGRAM_A, KEY_ONE), Executed(PROGRAM_B, KEY_TWO),
	                                         Executed(PROGRAM_C, KEY_THREE)};
	const auto baseline = ClassifyDispatches(scanned, all, false);
	CHECK(baseline.totals.matched.Total() == 3 && baseline.totals.extra.Total() == 0);
}

// The program is found and the key differs. The baseline is the same pair with equal keys, which is
// matched, so only the key decides the class. Nothing else counts the dispatch, since extra is 0.
void TestKeyDifferenceIsMismatch() {
	const std::vector<ScannedRecord> scanned {Record(PROGRAM_A, KEY_ONE)};
	const auto differs = ClassifyDispatches(scanned, std::vector {Executed(PROGRAM_A, KEY_TWO)}, false);
	CHECK(differs.totals.key_mismatch.Total() == 1 && differs.totals.matched.Total() == 0);
	CHECK(differs.totals.extra.Total() == 0 && differs.totals.missing.Total() == 0);
	CHECK(differs.totals.MismatchRate() == 1.0);

	const auto same = ClassifyDispatches(scanned, std::vector {Executed(PROGRAM_A, KEY_ONE)}, false);
	CHECK(same.totals.matched.Total() == 1 && same.totals.key_mismatch.Total() == 0);
	CHECK(same.totals.MismatchRate() == 0.0);
}

// A program the scan never listed is missing, and the record the scan did list is left over as extra. The
// baseline executes the listed program, which is matched. There is no distance bound on finding the
// program. More than the exact-key window of skipped records of other programs before it must still
// match, since an arm can hold hundreds of dispatches. The baseline with no records before it matches too,
// and a program that appears only before the cursor is missing, so the cursor moves forward only.
void TestAbsentProgramIsMissing() {
	const std::vector<ScannedRecord> scanned {Record(PROGRAM_A, KEY_ONE)};
	const auto absent = ClassifyDispatches(scanned, std::vector {Executed(PROGRAM_B, KEY_ONE)}, false);
	CHECK(absent.totals.missing.Total() == 1 && absent.totals.matched.Total() == 0);
	CHECK(absent.totals.extra.Total() == 1);
	const auto present = ClassifyDispatches(scanned, std::vector {Executed(PROGRAM_A, KEY_ONE)}, false);
	CHECK(present.totals.missing.Total() == 0 && present.totals.matched.Total() == 1);

	constexpr size_t SKIPPED_ARM_RECORDS = 3 * DIFFERENTIAL_EXACT_WINDOW;
	std::vector<ScannedRecord> padded(SKIPPED_ARM_RECORDS, Record(PROGRAM_B, KEY_ONE));
	padded.push_back(Record(PROGRAM_A, KEY_ONE));
	const auto distant = ClassifyDispatches(padded, std::vector {Executed(PROGRAM_A, KEY_ONE)}, false);
	CHECK(distant.totals.matched.Total() == 1 && distant.totals.missing.Total() == 0);
	CHECK(distant.totals.extra.Total() == SKIPPED_ARM_RECORDS);
	padded.erase(padded.begin(), padded.begin() + SKIPPED_ARM_RECORDS);
	const auto adjacent = ClassifyDispatches(padded, std::vector {Executed(PROGRAM_A, KEY_ONE)}, false);
	CHECK(adjacent.totals.matched.Total() == 1 && adjacent.totals.extra.Total() == 0);

	// A record behind the cursor is spent. The second dispatch of the program has nothing left.
	const auto twice = ClassifyDispatches(scanned, std::vector {Executed(PROGRAM_A, KEY_ONE),
	                                                            Executed(PROGRAM_A, KEY_ONE)}, false);
	CHECK(twice.totals.matched.Total() == 1 && twice.totals.missing.Total() == 1);
}

// When the scan was unsure of two records of one program, the executed dispatch takes the one with its
// key, and the other is left over as extra. The baseline marks the same records certain, and then the
// first record is a real mismatch that is not skipped for a better one, since that would hide mismatches
// behind later identical keys. Taking the first same-program record always would fail the first case.
void TestExactKeyPreferredOnlyAcrossUncertainRecords() {
	const auto records = [](bool uncertain) {
		return std::vector<ScannedRecord> {Record(PROGRAM_A, KEY_ONE, uncertain),
		                                   Record(PROGRAM_A, KEY_TWO, uncertain)};
	};
	const std::vector<ExecutedDispatch> executed {Executed(PROGRAM_A, KEY_TWO)};
	const auto unsure = ClassifyDispatches(records(true), executed, false);
	CHECK(unsure.totals.matched.Total() == 1 && unsure.totals.extra.Total() == 1);
	CHECK(unsure.totals.key_mismatch.Total() == 0);
	const auto sure = ClassifyDispatches(records(false), executed, false);
	CHECK(sure.totals.key_mismatch.Total() == 1 && sure.totals.matched.Total() == 0);

	// The window is measured from the first candidate, not from the cursor, so a long skipped arm of
	// uncertain records before a two-arm pair does not close it. The pair is the same in every case.
	const auto pair_after = [&](size_t between, size_t before) {
		std::vector<ScannedRecord> list(before, Record(PROGRAM_B, KEY_ONE, true));
		list.push_back(Record(PROGRAM_A, KEY_ONE, true));
		list.insert(list.end(), between, Record(PROGRAM_B, KEY_ONE, true));
		list.push_back(Record(PROGRAM_A, KEY_TWO, true));
		return ClassifyDispatches(list, executed, false).totals;
	};
	const auto prefixed = pair_after(0, 3 * DIFFERENTIAL_EXACT_WINDOW);
	CHECK(prefixed.matched.Total() == 1 && prefixed.key_mismatch.Total() == 0);

	// The second candidate at exactly the window distance from the first is out of reach, so the first is
	// taken and is a mismatch. One record nearer is in reach and matches.
	const auto at_boundary = pair_after(DIFFERENTIAL_EXACT_WINDOW - 1, 0);
	CHECK(at_boundary.key_mismatch.Total() == 1 && at_boundary.matched.Total() == 0);
	const auto in_reach = pair_after(DIFFERENTIAL_EXACT_WINDOW - 2, 0);
	CHECK(in_reach.matched.Total() == 1 && in_reach.key_mismatch.Total() == 0);
}

// Matched, mismatched and extra take the flag of the record they involve. Each uncertain count is paired
// with a baseline where the same records are certain, so the split cannot be a constant.
void TestUncertainSplit() {
	const auto records = [](bool uncertain) {
		return std::vector<ScannedRecord> {Record(PROGRAM_A, KEY_ONE, uncertain),
		                                   Record(PROGRAM_B, KEY_ONE, uncertain),
		                                   Record(PROGRAM_C, KEY_ONE, uncertain)};
	};
	const std::vector<ExecutedDispatch> executed {Executed(PROGRAM_A, KEY_ONE), Executed(PROGRAM_B, KEY_TWO)};
	const auto unsure = ClassifyDispatches(records(true), executed, false);
	CHECK(unsure.totals.matched.certain == 0 && unsure.totals.matched.uncertain == 1);
	CHECK(unsure.totals.key_mismatch.certain == 0 && unsure.totals.key_mismatch.uncertain == 1);
	CHECK(unsure.totals.extra.certain == 0 && unsure.totals.extra.uncertain == 1);
	CHECK(unsure.totals.MismatchRate() == 0.5 && unsure.totals.CertainMismatchRate() == 0.0);

	const auto sure = ClassifyDispatches(records(false), executed, false);
	CHECK(sure.totals.matched.certain == 1 && sure.totals.matched.uncertain == 0);
	CHECK(sure.totals.key_mismatch.certain == 1 && sure.totals.key_mismatch.uncertain == 0);
	CHECK(sure.totals.extra.certain == 1 && sure.totals.extra.uncertain == 0);
	CHECK(sure.totals.CertainMismatchRate() == 0.5);
}

// A missing dispatch has no record, so it takes the doubt of the records beside the cursor, and past the
// last record also the walk's tail doubt. Each case has a baseline that flips the one thing under test.
void TestMissingUncertaintyFollowsRecordsAtCursor() {
	const auto missing_in = [](const std::vector<ScannedRecord>& scanned,
	                           const std::vector<ExecutedDispatch>& executed, bool tail_uncertain) {
		return ClassifyDispatches(scanned, executed, tail_uncertain).totals.missing;
	};
	const std::vector<ExecutedDispatch> past_end {Executed(PROGRAM_A, KEY_ONE), Executed(PROGRAM_B, KEY_ONE),
	                                              Executed(DEAD_PROGRAM, KEY_ONE)};
	// The record before the cursor was the last one run, so its doubt carries to the miss.
	const auto last_unsure = missing_in({Record(PROGRAM_A, KEY_ONE, false), Record(PROGRAM_B, KEY_ONE, true)},
	                                    past_end, false);
	CHECK(last_unsure.uncertain == 1 && last_unsure.certain == 0);
	// An uncertain record that is not beside the cursor, with a certain tail, leaves the miss certain.
	const auto earlier_unsure = missing_in({Record(PROGRAM_A, KEY_ONE, true), Record(PROGRAM_B, KEY_ONE, false)},
	                                       past_end, false);
	CHECK(earlier_unsure.certain == 1 && earlier_unsure.uncertain == 0);
	// The same list with an uncertain tail is uncertain.
	const auto tail_unsure = missing_in({Record(PROGRAM_A, KEY_ONE, true), Record(PROGRAM_B, KEY_ONE, false)},
	                                    past_end, true);
	CHECK(tail_unsure.uncertain == 1 && tail_unsure.certain == 0);

	// A record at the cursor that the miss did not take also carries doubt. The baseline is it certain.
	const std::vector<ExecutedDispatch> before_end {Executed(PROGRAM_A, KEY_ONE), Executed(DEAD_PROGRAM, KEY_ONE)};
	const auto at_unsure = missing_in({Record(PROGRAM_A, KEY_ONE, false), Record(PROGRAM_B, KEY_ONE, true)},
	                                  before_end, false);
	CHECK(at_unsure.uncertain == 1 && at_unsure.certain == 0);
	const auto at_sure = missing_in({Record(PROGRAM_A, KEY_ONE, false), Record(PROGRAM_B, KEY_ONE, false)},
	                                before_end, false);
	CHECK(at_sure.certain == 1 && at_sure.uncertain == 0);

	// With no records at all the tail decides.
	const std::vector<ExecutedDispatch> lone {Executed(DEAD_PROGRAM, KEY_ONE)};
	CHECK(ClassifyDispatches({}, lone, true).totals.missing.uncertain == 1);
	CHECK(ClassifyDispatches({}, lone, false).totals.missing.certain == 1);
}

// The stream that makes program `address` current with `threads_x` threads in x.
Stream SetProgram(uint64_t address, uint32_t threads_x = THREADS_X) {
	const auto set_sh = [](uint32_t reg, const Stream& values) {
		Stream packet {KYTY_PM4(2 + values.size(), Pm4::IT_SET_SH_REG, 0), reg};
		packet.insert(packet.end(), values.begin(), values.end());
		return packet;
	};
	Stream stream;
	const auto append = [&](const Stream& more) { stream.insert(stream.end(), more.begin(), more.end()); };
	append(set_sh(Pm4::COMPUTE_PGM_LO,
	              {static_cast<uint32_t>(address >> 8u), static_cast<uint32_t>(address >> 40u)}));
	append(set_sh(Pm4::COMPUTE_PGM_RSRC1, {0u, (USER_SGPRS << RSRC2_USER_SHIFT) | RSRC2_TGID_X_EN}));
	append(set_sh(Pm4::COMPUTE_NUM_THREAD_X, {threads_x, 1u, 1u}));
	return stream;
}

Stream Dispatch(uint32_t mode = 0) {
	return {KYTY_PM4(5, Pm4::IT_DISPATCH_DIRECT, 0), 1u, 1u, 1u, mode};
}

Stream CondExec(uint32_t skipped) {
	return {KYTY_PM4(5, Pm4::IT_COND_EXEC, 0), COND_EXEC_FIRST_PAYLOAD, 0u, 0u, skipped};
}

// What the processor holds after a game wrote the same registers as SetProgram, decoded on its own and
// not taken from the scan. `threads_x` stands in for a register the processor and the scan disagree on.
HW::CsStageRegisters ProcessorRegisters(uint64_t address, uint32_t initiator, uint32_t threads_x = THREADS_X) {
	HW::CsStageRegisters cs;
	const auto           write = [&](uint32_t reg, uint32_t value) { CHECK(DecodeCsRegister(cs, reg, value)); };
	write(Pm4::COMPUTE_PGM_LO, static_cast<uint32_t>(address >> 8u));
	write(Pm4::COMPUTE_PGM_HI, static_cast<uint32_t>(address >> 40u));
	write(Pm4::COMPUTE_PGM_RSRC1, 0u);
	write(Pm4::COMPUTE_PGM_RSRC2, (USER_SGPRS << RSRC2_USER_SHIFT) | RSRC2_TGID_X_EN);
	write(Pm4::COMPUTE_NUM_THREAD_X, threads_x);
	write(Pm4::COMPUTE_NUM_THREAD_Y, 1u);
	write(Pm4::COMPUTE_NUM_THREAD_Z, 1u);
	cs.wave_size = Pm4::ComputeWaveSize(initiator);
	return cs;
}

std::unique_ptr<ScanDifferential> MakeLive() {
	return ScanDifferential::Create(true, QUEUE_COUNT, [](uint64_t, void*, uint64_t) { return false; });
}

// The scanner's key for a stream equals the shared key function applied to the registers the dispatch
// carries, and equals the processor side's when it holds the same registers. The baseline is a processor
// whose registers differ in one thread count, which must read as a key mismatch of the same program, so
// the key is not constant and covers that register. Dropping a register from the key on one side only
// would fail the first check.
void TestSharedKeyMatchesBothSides() {
	Stream commands = SetProgram(PROGRAM_A);
	const Stream dispatch = Dispatch();
	commands.insert(commands.end(), dispatch.begin(), dispatch.end());

	std::vector<ScannedDispatch> scanned;
	ScanStats                    stats;
	CsShadow                     shadow;
	SubmitScanner([](uint64_t, void*, uint64_t) { return false; })
	    .Scan(commands, TEST_QUEUE, true, shadow, scanned, stats);
	CHECK(scanned.size() == 1);
	if (scanned.empty()) {
		return;
	}
	CHECK(scanned[0].key_hash == ComputeDispatchKeyHash(scanned[0].regs.cs_regs, false, true));

	for (const uint32_t threads_x: {THREADS_X, THREADS_X * 2}) {
		auto live = MakeLive();
		CHECK(live != nullptr);
		if (!live) {
			return;
		}
		const auto seq = live->ScanSubmission(TEST_QUEUE, commands);
		live->OnCpDispatch(TEST_QUEUE, seq, ProcessorRegisters(PROGRAM_A, 0, threads_x), 0, true);
		live->FinishSubmission(TEST_QUEUE, seq);
		const auto report = live->Snapshot();
		if (threads_x == THREADS_X) {
			CHECK(report.totals.matched.Total() == 1 && report.totals.key_mismatch.Total() == 0);
		} else {
			CHECK(report.totals.key_mismatch.Total() == 1 && report.totals.matched.Total() == 0);
			CHECK(report.samples.size() == 1 && report.samples[0].kind == "key_mismatch");
		}
	}
}

// A real scan of a stream with a skippable region. The scan lists both programs and the processor runs
// the first only, so the second is extra and nothing mismatches. A second submission then inherits the
// program. The scan did not lose it, since the walk applied the skippable region's writes and so thinks
// program B is current. It records the inherited dispatch under PROGRAM_B, so the processor's dispatch of
// program A is missing and that record is extra, and both are uncertain because the region left the
// program in doubt. The baseline is the first submission alone, which has no mismatch at all.
void TestLiveScanSkippedRegionAndInheritedProgram() {
	const Stream second = [] {
		Stream stream = SetProgram(PROGRAM_B);
		const Stream dispatch = Dispatch();
		stream.insert(stream.end(), dispatch.begin(), dispatch.end());
		return stream;
	}();
	Stream commands = SetProgram(PROGRAM_A);
	const Stream first_dispatch = Dispatch();
	commands.insert(commands.end(), first_dispatch.begin(), first_dispatch.end());
	const Stream region = CondExec(static_cast<uint32_t>(second.size()));
	commands.insert(commands.end(), region.begin(), region.end());
	commands.insert(commands.end(), second.begin(), second.end());

	auto live = MakeLive();
	CHECK(live != nullptr);
	if (!live) {
		return;
	}
	const auto seq = live->ScanSubmission(TEST_QUEUE, commands);
	CHECK(seq == 1);
	live->OnCpDispatch(TEST_QUEUE, seq, ProcessorRegisters(PROGRAM_A, 0), 0, true);
	CHECK(live->OpenSubmissions() == 1);
	// A submission still open is visible in the report, and its records are not yet counted as extra.
	CHECK(live->Snapshot().open_submissions == 1 && live->Snapshot().totals.extra.Total() == 0);
	live->FinishSubmission(TEST_QUEUE, seq);
	CHECK(live->OpenSubmissions() == 0);
	auto report = live->Snapshot();
	CHECK(report.totals.matched.Total() == 1 && report.totals.extra.Total() == 1);
	CHECK(report.totals.key_mismatch.Total() == 0 && report.totals.missing.Total() == 0);
	CHECK(report.totals.MismatchRate() == 0.0);
	CHECK(report.queues.size() == 1 && report.queues[0].queue == TEST_QUEUE);
	if (report.queues.size() == 1) {
		CHECK(report.queues[0].dispatches == 2 && report.queues[0].cp_dispatches == 1);
		const auto cause = std::ranges::find_if(report.queues[0].uncertain_cleared_by,
		                                        [](const auto& entry) { return entry.first == "cond_exec_region"; });
		CHECK(cause != report.queues[0].uncertain_cleared_by.end() && cause->second == 1);
	}

	const Stream inherit = Dispatch();
	const auto   next    = live->ScanSubmission(TEST_QUEUE, inherit);
	CHECK(next == 2);
	live->OnCpDispatch(TEST_QUEUE, next, ProcessorRegisters(PROGRAM_A, 0), 0, true);
	live->FinishSubmission(TEST_QUEUE, next);
	report = live->Snapshot();
	CHECK(report.totals.missing.uncertain == 1 && report.totals.missing.certain == 0);
	CHECK(report.totals.extra.uncertain == 1 && report.totals.extra.Total() == 2);
	CHECK(report.open_submissions == 0);
	CHECK(report.samples.size() == 1 && report.samples[0].kind == "missing" && report.samples[0].seq == 2);
}

// Only eight samples are kept, in the order they happened. The first is the first mismatch. The scan of
// the submission found nothing and ended certain, so every miss is a certain one, which a counter that
// treated every miss as doubtful would not give.
void TestSamplesAreCapped() {
	auto live = MakeLive();
	CHECK(live != nullptr);
	if (!live) {
		return;
	}
	const auto seq = live->ScanSubmission(TEST_QUEUE, SetProgram(PROGRAM_A));
	for (uint32_t count = 0; count < 3 * Common::FirstRunReport::SCAN_DIFFERENTIAL_SAMPLE_LIMIT; ++count) {
		live->OnCpDispatch(TEST_QUEUE, seq, ProcessorRegisters(PROGRAM_B + count * PROGRAM_STRIDE, 0), 0, true);
	}
	const auto report = live->Snapshot();
	CHECK(report.samples.size() == Common::FirstRunReport::SCAN_DIFFERENTIAL_SAMPLE_LIMIT);
	CHECK(report.totals.missing.Total() == 3 * Common::FirstRunReport::SCAN_DIFFERENTIAL_SAMPLE_LIMIT);
	CHECK(report.totals.missing.certain == 3 * Common::FirstRunReport::SCAN_DIFFERENTIAL_SAMPLE_LIMIT);
	if (!report.samples.empty()) {
		CHECK(report.samples[0].program == PROGRAM_B);
	}
}

// A walk that ends in a skippable region that wrote the program is in doubt about what follows its last
// record, so a dispatch past that record counts as an uncertain miss. The baseline is the same stream
// without the region, which ends certain and gives a certain miss, so the tail doubt is the only difference.
void TestLiveTailDoubtReachesMissing() {
	const auto miss_after = [](bool with_region) {
		Stream commands = SetProgram(PROGRAM_A);
		const Stream dispatch = Dispatch();
		commands.insert(commands.end(), dispatch.begin(), dispatch.end());
		if (with_region) {
			const Stream skipped = SetProgram(PROGRAM_B);
			const Stream region  = CondExec(static_cast<uint32_t>(skipped.size()));
			commands.insert(commands.end(), region.begin(), region.end());
			commands.insert(commands.end(), skipped.begin(), skipped.end());
		}
		auto live = MakeLive();
		CHECK(live != nullptr);
		if (!live) {
			return Common::FirstRunReport::ScanOutcomeCount {};
		}
		const auto seq = live->ScanSubmission(TEST_QUEUE, commands);
		live->OnCpDispatch(TEST_QUEUE, seq, ProcessorRegisters(PROGRAM_A, 0), 0, true);
		live->OnCpDispatch(TEST_QUEUE, seq, ProcessorRegisters(DEAD_PROGRAM, 0), 0, true);
		return live->Snapshot().totals.missing;
	};
	const auto doubtful = miss_after(true);
	CHECK(doubtful.uncertain == 1 && doubtful.certain == 0);
	const auto clean = miss_after(false);
	CHECK(clean.certain == 1 && clean.uncertain == 0);
}

// A dispatch of a submission that was never scanned is counted apart and moves no class, and one with no
// program is left out entirely.
void TestUntrackedAndProgramlessDispatches() {
	auto live = MakeLive();
	CHECK(live != nullptr);
	if (!live) {
		return;
	}
	live->OnCpDispatch(TEST_QUEUE, 99, ProcessorRegisters(PROGRAM_A, 0), 0, true);
	live->OnCpDispatch(TEST_QUEUE, 99, HW::CsStageRegisters {}, 0, true);
	const auto report = live->Snapshot();
	CHECK(report.untracked_cp_dispatches == 1);
	// An untracked dispatch is not in any queue's figure either, so the queue columns agree with the total.
	CHECK(report.queues.empty());
	CHECK(report.totals.CpDispatches() == 0);
}

// The percentile is a time a scan really took. The baseline is the maximum, kept apart, which a long scan
// past the last bucket reports exactly.
void TestHistogramPercentiles() {
	ScanTimeHistogram histogram;
	CHECK(histogram.Percentile(0.5) == 0 && histogram.Max() == 0);
	for (uint64_t micros = 1; micros <= 100; ++micros) {
		histogram.Add(micros);
	}
	CHECK(histogram.Percentile(0.50) == 50);
	CHECK(histogram.Percentile(0.99) == 99);
	histogram.Add(ScanTimeHistogram::BUCKETS * 3);
	CHECK(histogram.Max() == ScanTimeHistogram::BUCKETS * 3);
	CHECK(histogram.Percentile(1.0) == ScanTimeHistogram::BUCKETS * 3);
}

// With the flag off nothing is built, so no record can exist, and a report with no source has no block.
// The baselines are the same calls with the flag on, which build the object and write the block, so the
// absence is caused by the flag and not by the calls failing.
void TestFlagOffDoesNothing() {
	CHECK(ScanDifferential::Create(false, QUEUE_COUNT, [](uint64_t, void*, uint64_t) { return false; }) == nullptr);
	CHECK(MakeLive() != nullptr);

	Common::FirstRunReport::Report off;
	const auto without = nlohmann::json::parse(off.ToJson({}, 0, 1.0, 0.0, {}));
	CHECK(!without.contains("scan_differential"));

	Common::FirstRunReport::Report on;
	on.SetScanDifferentialSource([] { return ScanDifferentialReport {}; });
	const auto with = nlohmann::json::parse(on.ToJson({}, 0, 1.0, 0.0, {}));
	CHECK(with.contains("scan_differential"));
	// A source that was replaced by final figures still writes the block, and a reset removes it again.
	on.SetScanDifferentialFinal({});
	CHECK(nlohmann::json::parse(on.ToJson({}, 0, 1.0, 0.0, {})).contains("scan_differential"));
	on.Reset();
	CHECK(!nlohmann::json::parse(on.ToJson({}, 0, 1.0, 0.0, {})).contains("scan_differential"));
}

// The block carries the totals split by certainty, the rate the gate reads, and a column row for the
// queue. Extra is outside the rate, which the 5 extra records would change if they counted. The
// baseline is a block with only matches, whose rate is 0.
void TestReportBlockFigures() {
	ScanDifferentialReport block;
	block.totals.matched.certain      = 5;
	block.totals.matched.uncertain    = 1;
	block.totals.key_mismatch.uncertain = 1;
	block.totals.missing.certain      = 1;
	block.totals.extra.certain        = 5;
	block.open_submissions            = 2;
	block.samples.push_back({.kind = "missing", .queue = 3, .seq = 9, .program = PROGRAM_A,
	                         .scanned_key = 0, .executed_key = KEY_ONE});
	Common::FirstRunReport::ScanQueueReport row;
	row.queue               = 3;
	row.scan_p99_us         = 77;
	row.uncertain_cleared_by = {{"branch", 4}};
	block.queues.push_back(row);

	Common::FirstRunReport::Report report;
	report.SetScanDifferentialSource([block] { return block; });
	const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["scan_differential"];
	CHECK(json["cp_dispatches"] == 8);
	CHECK(json["matched"]["certain"] == 5 && json["matched"]["uncertain"] == 1);
	CHECK(json["key_mismatch"]["uncertain"] == 1 && json["missing"]["certain"] == 1);
	CHECK(json["extra"]["certain"] == 5);
	CHECK(json["open_submissions"] == 2);
	CHECK(json["mismatch_rate"] == 0.25);
	CHECK(json["samples"].size() == 1 && json["samples"][0]["kind"] == "missing");
	CHECK(json["samples"][0]["program"] == "0000004567890100");
	CHECK(json["queues"].size() == 1 && json["queues"][0]["scan_p99_us"] == 77);
	CHECK(json["queues"][0]["uncertain_cleared_by"]["branch"] == 4);

	Common::FirstRunReport::Report clean;
	ScanDifferentialReport         only_matches;
	only_matches.totals.matched.certain = 4;
	clean.SetScanDifferentialSource([only_matches] { return only_matches; });
	CHECK(nlohmann::json::parse(clean.ToJson({}, 0, 1.0, 0.0, {}))["scan_differential"]["mismatch_rate"] == 0.0);
}

} // namespace

int main() {
	TestSupersetWithSkippedArms();
	TestKeyDifferenceIsMismatch();
	TestAbsentProgramIsMissing();
	TestExactKeyPreferredOnlyAcrossUncertainRecords();
	TestUncertainSplit();
	TestMissingUncertaintyFollowsRecordsAtCursor();
	TestSharedKeyMatchesBothSides();
	TestLiveScanSkippedRegionAndInheritedProgram();
	TestSamplesAreCapped();
	TestLiveTailDoubtReachesMissing();
	TestUntrackedAndProgramlessDispatches();
	TestHistogramPercentiles();
	TestFlagOffDoesNothing();
	TestReportBlockFigures();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("ScanDifferentialTests passed");
	return 0;
}
