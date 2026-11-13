#include "graphics/guest_gpu/command_processor/scanDifferential.h"

#include "graphics/guest_gpu/pm4.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

namespace Libs::Graphics {

namespace {

// Queue 0 is the graphics queue and every other queue is async compute, the way the command processor
// tells them apart.
constexpr uint32_t GRAPHICS_QUEUE = 0;
// The first sequence number of a queue. 0 is kept to mean that a submission was not scanned.
constexpr uint64_t FIRST_SEQUENCE = 1;
// The two fractions the report asks of the scan times.
constexpr double MEDIAN_FRACTION     = 0.50;
constexpr double TAIL_FRACTION       = 0.99;
constexpr uint64_t NANOS_PER_MICRO   = 1000;

using Common::FirstRunReport::ScanOutcomeCount;
using Common::FirstRunReport::ScanOutcomeTotals;

// Which half of a class a dispatch counts in.
void AddTo(ScanOutcomeCount& count, bool uncertain) {
	(uncertain ? count.uncertain : count.certain)++;
}

} // namespace

DispatchMatcher::DispatchMatcher(std::vector<ScannedRecord> scanned, bool tail_uncertain)
    : m_scanned(std::move(scanned)), m_consumed(m_scanned.size(), false), m_tail_uncertain(tail_uncertain) {
	// Filled in ascending order, so every program's list is sorted.
	for (size_t index = 0; index < m_scanned.size(); ++index) {
		m_by_program[m_scanned[index].program_address].push_back(index);
	}
}

size_t DispatchMatcher::PickCandidate(const std::vector<size_t>& candidates, size_t first,
                                      uint64_t key_hash) const {
	if (m_scanned[candidates[first]].key_hash == key_hash) {
		return candidates[first];
	}
	// Step over a same-program record only when the scan was unsure of it, and only within the window.
	for (size_t next = first + 1; next < candidates.size(); ++next) {
		const bool stepped_over_uncertain = m_scanned[candidates[next - 1]].uncertain;
		if (!stepped_over_uncertain || candidates[next] >= candidates[first] + DIFFERENTIAL_EXACT_WINDOW) {
			break;
		}
		if (m_scanned[candidates[next]].key_hash == key_hash) {
			return candidates[next];
		}
	}
	return candidates[first];
}

bool DispatchMatcher::MissingIsUncertain() const {
	// The records just before and at the cursor are what the scan thought was running around here.
	const bool before_uncertain = m_cursor > 0 && m_scanned[m_cursor - 1].uncertain;
	if (m_cursor < m_scanned.size()) {
		return before_uncertain || m_scanned[m_cursor].uncertain;
	}
	// Past the last record, what the scan could not see is the walk's tail doubt.
	return before_uncertain || m_tail_uncertain;
}

DifferentialVerdict DispatchMatcher::Match(const ExecutedDispatch& executed) {
	const auto found = m_by_program.find(executed.program_address);
	if (found != m_by_program.end()) {
		const auto& candidates = found->second;
		// The first record of this program at or after the cursor.
		const auto first = std::lower_bound(candidates.begin(), candidates.end(), m_cursor);
		if (first != candidates.end()) {
			const size_t index =
			    PickCandidate(candidates, static_cast<size_t>(first - candidates.begin()), executed.key_hash);
			m_consumed[index]  = true;
			m_cursor           = index + 1;
			const auto& record = m_scanned[index];
			return {.outcome     = record.key_hash == executed.key_hash ? DifferentialOutcome::Matched
			                                                            : DifferentialOutcome::KeyMismatch,
			        .uncertain   = record.uncertain,
			        .scanned_key = record.key_hash};
		}
	}
	return {.outcome = DifferentialOutcome::Missing, .uncertain = MissingIsUncertain(), .scanned_key = 0};
}

ScanOutcomeCount DispatchMatcher::Unconsumed() const {
	ScanOutcomeCount count;
	for (size_t index = 0; index < m_scanned.size(); ++index) {
		if (!m_consumed[index]) {
			AddTo(count, m_scanned[index].uncertain);
		}
	}
	return count;
}

void AddVerdict(ScanOutcomeTotals& totals, const DifferentialVerdict& verdict) {
	switch (verdict.outcome) {
		case DifferentialOutcome::Matched: AddTo(totals.matched, verdict.uncertain); break;
		case DifferentialOutcome::KeyMismatch: AddTo(totals.key_mismatch, verdict.uncertain); break;
		case DifferentialOutcome::Missing: AddTo(totals.missing, verdict.uncertain); break;
	}
}

DifferentialClassification ClassifyDispatches(std::span<const ScannedRecord>    scanned,
                                              std::span<const ExecutedDispatch> executed,
                                              bool                              tail_uncertain) {
	DispatchMatcher            matcher({scanned.begin(), scanned.end()}, tail_uncertain);
	DifferentialClassification result;
	for (const auto& dispatch: executed) {
		AddVerdict(result.totals, matcher.Match(dispatch));
	}
	const auto extra         = matcher.Unconsumed();
	result.totals.extra.certain   = extra.certain;
	result.totals.extra.uncertain = extra.uncertain;
	return result;
}

void ScanTimeHistogram::Add(uint64_t microseconds) {
	if (m_buckets.empty()) {
		m_buckets.assign(BUCKETS + 1, 0);
	}
	m_buckets[std::min(microseconds, BUCKETS)]++;
	m_count++;
	m_max = std::max(m_max, microseconds);
}

uint64_t ScanTimeHistogram::Percentile(double fraction) const {
	if (m_count == 0) {
		return 0;
	}
	// Nearest rank, so the answer is a time that a scan really took.
	const auto rank = std::max<uint64_t>(
	    1, static_cast<uint64_t>(std::ceil(fraction * static_cast<double>(m_count))));
	uint64_t   seen = 0;
	for (uint64_t bucket = 0; bucket < m_buckets.size(); ++bucket) {
		seen += m_buckets[bucket];
		if (seen >= rank) {
			// The last bucket holds every longer scan, so it answers with the exact maximum.
			return bucket == BUCKETS ? m_max : bucket;
		}
	}
	return m_max;
}

std::unique_ptr<ScanDifferential> ScanDifferential::Create(bool enabled, uint32_t queue_count,
                                                           ScanMemoryReader reader,
                                                           ScanLimits limits) {
	if (!enabled) {
		return nullptr;
	}
	return std::make_unique<ScanDifferential>(queue_count, std::move(reader), limits);
}

ScanDifferential::ScanDifferential(uint32_t queue_count, ScanMemoryReader reader, ScanLimits limits)
    : m_scanner(std::move(reader), limits), m_queues(queue_count) {
	for (auto& queue: m_queues) {
		queue.next_seq = FIRST_SEQUENCE;
	}
}

uint64_t ScanDifferential::ScanSubmission(uint32_t queue, std::span<const uint32_t> commands) {
	if (queue >= m_queues.size()) {
		return 0;
	}
	std::lock_guard scan_lock(m_scan_mutex);
	auto&           state = m_queues[queue];
	if (state.scratch.capacity() == 0) {
		state.scratch.reserve(SCAN_DISPATCH_CAPACITY);
	}
	state.scratch.clear();
	ScanStats  stats;
	const auto started = std::chrono::steady_clock::now();
	m_scanner.Scan(commands, queue, queue != GRAPHICS_QUEUE, state.shadow, state.scratch, stats, m_scratch);
	const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
	                       std::chrono::steady_clock::now() - started)
	                       .count();

	std::vector<ScannedRecord> records;
	records.reserve(state.scratch.size());
	for (const auto& dispatch: state.scratch) {
		records.push_back({dispatch.program_address, dispatch.key_hash, dispatch.uncertain_program});
	}
	const uint64_t seq = state.next_seq++;
	// The matcher and its index are built here, before the data lock, so the GPU thread never waits on
	// the build. The walk's tail doubt is whether the shadow ended unable to vouch for its program.
	Record(queue, seq, DispatchMatcher(std::move(records), !state.shadow.certain), stats,
	       static_cast<uint64_t>(nanos) / NANOS_PER_MICRO);
	return seq;
}

// Stores a scanned submission's matcher under its sequence number, and adds the walk's counts and time
// to its queue. Takes the data lock, which the scan itself never holds.
void ScanDifferential::Record(uint32_t queue, uint64_t seq, DispatchMatcher matcher,
                              const ScanStats& stats, uint64_t microseconds) {
	std::lock_guard lock(m_data_mutex);
	auto&           state = m_queues[queue];
	state.stats.Add(stats);
	state.timing.Add(microseconds);
	state.open.emplace(seq, std::move(matcher));
}

void ScanDifferential::ResetShadow(uint32_t queue) {
	if (queue >= m_queues.size()) {
		return;
	}
	std::lock_guard scan_lock(m_scan_mutex);
	m_queues[queue].shadow.Reset();
}

void ScanDifferential::OnCpDispatch(uint32_t queue, uint64_t seq, const HW::CsStageRegisters& cs,
                                    uint32_t initiator, bool async_queue) {
	if (cs.data_addr == 0) {
		return;
	}
	const bool thread_dimensions =
	    (initiator & Pm4::COMPUTE_DISPATCH_INITIATOR_USE_THREAD_DIMENSIONS) != 0;
	const ExecutedDispatch executed {.program_address = cs.data_addr,
	                                 .key_hash = ComputeDispatchKeyHash(cs, thread_dimensions, async_queue)};
	std::lock_guard lock(m_data_mutex);
	if (queue >= m_queues.size()) {
		m_untracked++;
		return;
	}
	auto&      state = m_queues[queue];
	const auto found = state.open.find(seq);
	if (found == state.open.end()) {
		m_untracked++;
		return;
	}
	// Counted only once the submission is known, so the queue columns add up to the top-level figure.
	state.cp_dispatches++;
	const auto verdict = found->second.Match(executed);
	AddVerdict(m_totals, verdict);
	if (verdict.outcome != DifferentialOutcome::Matched &&
	    m_samples.size() < Common::FirstRunReport::SCAN_DIFFERENTIAL_SAMPLE_LIMIT) {
		m_samples.push_back({.kind = verdict.outcome == DifferentialOutcome::Missing ? "missing"
		                                                                              : "key_mismatch",
		                     .queue        = queue,
		                     .seq          = seq,
		                     .program      = executed.program_address,
		                     .scanned_key  = verdict.scanned_key,
		                     .executed_key = executed.key_hash});
	}
}

void ScanDifferential::FinishSubmission(uint32_t queue, uint64_t seq) {
	std::lock_guard lock(m_data_mutex);
	if (queue >= m_queues.size()) {
		return;
	}
	auto& open  = m_queues[queue].open;
	const auto found = open.find(seq);
	if (found == open.end()) {
		return;
	}
	const auto extra = found->second.Unconsumed();
	m_totals.extra.certain += extra.certain;
	m_totals.extra.uncertain += extra.uncertain;
	open.erase(found);
}

size_t ScanDifferential::OpenSubmissions() const {
	std::lock_guard lock(m_data_mutex);
	size_t          total = 0;
	for (const auto& queue: m_queues) {
		total += queue.open.size();
	}
	return total;
}

Common::FirstRunReport::ScanDifferentialReport ScanDifferential::Snapshot() const {
	std::lock_guard                                lock(m_data_mutex);
	Common::FirstRunReport::ScanDifferentialReport report;
	report.totals                  = m_totals;
	report.untracked_cp_dispatches = m_untracked;
	report.samples                 = m_samples;
	for (const auto& queue: m_queues) {
		report.open_submissions += queue.open.size();
	}
	for (uint32_t id = 0; id < m_queues.size(); ++id) {
		const auto& state = m_queues[id];
		// A queue nobody submitted to has no row.
		if (state.stats.submissions == 0 && state.cp_dispatches == 0) {
			continue;
		}
		Common::FirstRunReport::ScanQueueReport row;
		row.queue                   = id;
		row.submissions             = state.stats.submissions;
		row.cp_dispatches           = state.cp_dispatches;
		row.dispatches              = state.stats.dispatches;
		row.uncertain               = state.stats.uncertain_program;
		row.branch_mode1            = state.stats.branch_mode1;
		row.branch_mode2            = state.stats.branch_mode2;
		row.branch_mode2_empty_else = state.stats.branch_mode2_empty_else;
		row.ib_revisit_doubts       = state.stats.ib_revisit_doubts;
		row.bad_packets             = state.stats.bad_packets;
		row.scan_allocations        = state.stats.scan_allocations;
		row.scan_p50_us             = state.timing.Percentile(MEDIAN_FRACTION);
		row.scan_p99_us             = state.timing.Percentile(TAIL_FRACTION);
		row.scan_max_us             = state.timing.Max();
		for (size_t cause = 0; cause < state.stats.uncertain_cleared_by.size(); ++cause) {
			row.uncertain_cleared_by.emplace_back(
			    UncertainCauseName(static_cast<UncertainCause>(cause)),
			    state.stats.uncertain_cleared_by[cause]);
		}
		report.queues.push_back(std::move(row));
	}
	return report;
}

} // namespace Libs::Graphics
