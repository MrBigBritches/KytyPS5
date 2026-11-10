#include "graphics/shader/storegen/generatedDrainPlanner.h"

namespace Libs::Graphics::StoreGen {

namespace {

constexpr double SLOW_DRAIN_MS      = 1.0;
constexpr double VERY_SLOW_DRAIN_MS = 4.0;

double Milliseconds(std::chrono::nanoseconds duration) {
	return std::chrono::duration<double, std::milli>(duration).count();
}

} // namespace

DrainVerdict GeneratedDrainPlanner::Judge(const DrainFacts&                facts,
                                          const std::function<uint64_t()>& existing_plan_hash) {
	if (facts.stopped) {
		m_counters.dropped_stopped++;
		return DrainVerdict::DropStopped;
	}
	if (facts.program_inserted) {
		if (!facts.plan_installed) {
			m_counters.failed_installs++;
			m_known.erase(facts.program);
			return DrainVerdict::DropFailedInstall;
		}
		m_known[facts.program] = {
		    .hash = facts.plan_hash, .trusted = true, .from_generation = true};
		m_counters.adopted++;
		return DrainVerdict::Adopt;
	}
	auto known = m_known.find(facts.program);
	if (known == m_known.end()) {
		// The first module that meets a program that was there before. Its plan is compared once.
		const auto existing = existing_plan_hash();
		known               = m_known
		            .emplace(facts.program,
		                     Known {.hash = existing, .trusted = existing == facts.plan_hash})
		            .first;
	}
	// A later module of the program is checked against what the program was found to have, so a
	// module of the same program with another plan is caught as well.
	if (!known->second.trusted || known->second.hash != facts.plan_hash) {
		m_counters.plan_mismatch++;
		return DrainVerdict::DropPlanMismatch;
	}
	if (facts.id_present) {
		m_counters.duplicates++;
		return DrainVerdict::DropDuplicate;
	}
	m_counters.adopted++;
	return DrainVerdict::Adopt;
}

ReverseVerdict
GeneratedDrainPlanner::CheckReverse(uint64_t                         program,
                                    const std::function<uint64_t()>& runtime_plan_hash) {
	const auto known = m_known.find(program);
	if (known == m_known.end() || !known->second.from_generation || known->second.reverse_checked) {
		return ReverseVerdict::NotApplicable;
	}
	known->second.reverse_checked = true;
	if (runtime_plan_hash() != known->second.hash) {
		m_counters.plan_mismatch_reverse++;
		return ReverseVerdict::Mismatch;
	}
	return ReverseVerdict::Match;
}

void GeneratedDrainPlanner::NoteItem(std::chrono::nanoseconds duration) {
	const double ms        = Milliseconds(duration);
	m_counters.item_max_ms = ms > m_counters.item_max_ms ? ms : m_counters.item_max_ms;
}

void GeneratedDrainPlanner::NoteDrain(std::chrono::nanoseconds duration,
                                      std::chrono::nanoseconds oldest_wait) {
	const double ms = Milliseconds(duration);
	m_counters.drain_calls++;
	m_counters.drain_total_ms += ms;
	m_counters.drain_max_ms = ms > m_counters.drain_max_ms ? ms : m_counters.drain_max_ms;
	m_counters.drain_over_1ms += ms > SLOW_DRAIN_MS;
	m_counters.drain_over_4ms += ms > VERY_SLOW_DRAIN_MS;
	const double lag = Milliseconds(oldest_wait);
	m_counters.drain_lag_ms_max =
	    lag > m_counters.drain_lag_ms_max ? lag : m_counters.drain_lag_ms_max;
}

} // namespace Libs::Graphics::StoreGen
