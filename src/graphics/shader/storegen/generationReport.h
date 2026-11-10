#ifndef GRAPHICS_SHADER_STOREGEN_GENERATION_REPORT_H
#define GRAPHICS_SHADER_STOREGEN_GENERATION_REPORT_H

#include "common/firstRunReport.h"
#include "graphics/shader/storegen/generatedDrainPlanner.h"
#include "graphics/shader/storegen/generationRows.h"
#include "graphics/shader/storegen/missClassifier.h"
#include "graphics/shader/storegen/storeSupervisor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

// Turns what the supervisor and the drain counted into the first-run report's generation block.
// It needs no device, so a test can build a block from made-up figures.
namespace Libs::Graphics::StoreGen {

/// Nanoseconds in a millisecond, for the rows that carry times on the NowNs clock.
constexpr double ROW_NS_PER_MS = 1.0e6;

/// The counts of the pipeline cache's intake that the drain planner does not hold.
struct GenerationCounts {
	uint64_t posted            = 0;
	uint64_t handles_created   = 0;
	uint64_t handles_destroyed = 0;
	uint64_t bad_plans         = 0;
	uint64_t undrained_at_stop = 0;
	uint64_t pending           = 0;
};

/// The name of an outcome in the report.
[[nodiscard]] inline const char* OutcomeName(SupervisorOutcome outcome) {
	switch (outcome) {
		case SupervisorOutcome::Running: return "running";
		case SupervisorOutcome::Finished: return "finished";
		case SupervisorOutcome::Fatal: return "fatal";
		case SupervisorOutcome::RestartsCapped: return "restarts_capped";
		case SupervisorOutcome::Stopped: return "stopped";
	}
	return "";
}

/// The name of a failure kind in the report.
[[nodiscard]] inline const char* FailureKindName(FailureKind kind) {
	switch (kind) {
		case FailureKind::Crash: return "crash";
		case FailureKind::Hang: return "hang";
		case FailureKind::OrderMismatch: return "order_mismatch";
		case FailureKind::BadOutput: return "bad_output";
		case FailureKind::Truncated: return "truncated";
	}
	return "";
}

/// The name of a drop reason in the report.
[[nodiscard]] inline const char* DropReasonName(ProgressDropReason reason) {
	switch (reason) {
		case ProgressDropReason::PushStart: return "push_start";
		case ProgressDropReason::ResourceCount: return "resource_count";
		case ProgressDropReason::FrameTooLarge: return "frame_too_large";
	}
	return "";
}

/// The name of a peak memory source in the report.
[[nodiscard]] inline const char* PeakSourceName(Common::PeakMemorySource source) {
	switch (source) {
		case Common::PeakMemorySource::WorkingSet: return "working_set";
		case Common::PeakMemorySource::WorkingSetSampled: return "working_set_sampled";
		case Common::PeakMemorySource::Commit: return "commit";
	}
	return "";
}

/// The name of a prediction outcome in the report, which is the word of the worker's line.
[[nodiscard]] inline const char* PredictionOutcomeName(PredictionOutcome outcome) {
	switch (outcome) {
		case PredictionOutcome::Predicted: return "predicted";
		case PredictionOutcome::Calls: return "calls";
		case PredictionOutcome::Unclassified: return "unclassified";
		case PredictionOutcome::Unreadable: return "unreadable";
		case PredictionOutcome::Problems: return "problems";
	}
	return "";
}

/// Copies the supervisor's statistics into the report's plain figures. finished_ms is given only
/// for a run that finished, and ended_ms for every run that is over. A supervisor that ended for a
/// reason that stops generation makes the block inactive, with that reason.
[[nodiscard]] inline Common::FirstRunReport::ShaderStoreGenerationReport
MakeSupervisorReport(const SupervisorStats& stats) {
	Common::FirstRunReport::ShaderStoreGenerationReport block;
	const bool stopped_for_good = stats.outcome == SupervisorOutcome::Fatal ||
	                              stats.outcome == SupervisorOutcome::RestartsCapped;
	block.active          = !stopped_for_good;
	block.inactive_reason = stopped_for_good ? stats.inactive_reason : std::string();
	block.outcome         = OutcomeName(stats.outcome);
	block.spawns          = stats.spawns;
	block.restarts        = stats.restarts;
	block.spawn_ms        = stats.spawn_ms;
	block.child_open_ms   = stats.child_open_ms;
	block.first_record_ms = stats.first_record_ms;
	block.last_record_ms  = stats.last_record_ms;
	if (stats.outcome == SupervisorOutcome::Finished) {
		block.finished_ms = stats.finished_ms;
	}
	if (stats.outcome != SupervisorOutcome::Running) {
		block.ended_ms = stats.ended_ms;
	}
	block.decode_ms_max      = stats.decode_ms_max;
	block.terminate_wait_ms  = stats.terminate_wait_ms;
	block.frames_delivered   = stats.frames_delivered;
	block.bad_records        = stats.bad_records;
	block.duplicates         = stats.duplicates;
	block.posted             = stats.posted;
	block.predicted_programs = static_cast<uint32_t>(stats.predicted_hashes.size());
	for (const auto& failure: stats.failures) {
		block.failures.push_back({.hash      = failure.hash,
		                          .rank      = failure.rank,
		                          .exit_code = failure.exit_code,
		                          .kind      = FailureKindName(failure.kind),
		                          .last_line = failure.last_line,
		                          .spawn     = failure.spawn,
		                          .cpu_ms    = failure.cpu_ms,
		                          .out_tail  = failure.out_tail});
	}
	for (const auto& usage: stats.spawn_usage) {
		Common::FirstRunReport::GenerationSpawnUsage row {.spawn  = usage.spawn,
		                                                  .cpu_ms = usage.cpu_ms};
		if (usage.peak) {
			row.peak_bytes  = usage.peak->bytes;
			row.peak_source = PeakSourceName(usage.peak->source);
		}
		block.spawn_usage.push_back(std::move(row));
	}
	if (stats.peak_memory) {
		block.peak_bytes  = stats.peak_memory->bytes;
		block.peak_source = PeakSourceName(stats.peak_memory->source);
	}
	for (const auto& drop: stats.dropped) {
		block.dropped.push_back(
		    {.hash = drop.hash, .rank = drop.rank, .reason = DropReasonName(drop.reason)});
	}
	block.failed_hashes = stats.failed_hashes;
	block.skip_hashes   = stats.skip_hashes;
	block.worker        = stats.worker_summary;
	for (const auto& row: stats.predictions) {
		block.predictions.push_back({.hash       = row.hash,
		                             .outcome    = PredictionOutcomeName(row.outcome),
		                             .candidates = row.candidates});
	}
	return block;
}

/// Copies the drain's counters and the intake's counts into the report's figures.
inline void FillDrainReport(Common::FirstRunReport::GenerationDrain& out,
                            const GeneratedDrainCounters& drain, const GenerationCounts& counts) {
	out.calls                 = drain.drain_calls;
	out.over_1ms              = drain.drain_over_1ms;
	out.over_4ms              = drain.drain_over_4ms;
	out.max_ms                = drain.drain_max_ms;
	out.total_ms              = drain.drain_total_ms;
	out.lag_ms_max            = drain.drain_lag_ms_max;
	out.item_max_ms           = drain.item_max_ms;
	out.adopted               = drain.adopted;
	out.duplicates            = drain.duplicates;
	out.plan_mismatch         = drain.plan_mismatch;
	out.plan_mismatch_reverse = drain.plan_mismatch_reverse;
	out.failed_installs       = drain.failed_installs;
	out.dropped_stopped       = drain.dropped_stopped;
	out.posted                = counts.posted;
	out.handles_created       = counts.handles_created;
	out.handles_destroyed     = counts.handles_destroyed;
	out.bad_plans             = counts.bad_plans;
	out.undrained_at_stop     = counts.undrained_at_stop;
	out.pending               = counts.pending;
}

/// What the rows of the report are made from. The pipeline cache copies each part under its own
/// lock and the report's own rows, so building the block holds no lock of the cache.
struct GenerationRowInputs {
	std::vector<GenRowSnapshot>                       generated;
	std::vector<DropRow>                              refused;
	std::vector<MissRow>                              misses;
	std::vector<Common::FirstRunReport::CompileEntry> compiles;
	std::vector<Common::FirstRunReport::WarmTake>     takes;
};

/// The classifier's lookups, from the supervisor's statistics and the rows of the cache.
[[nodiscard]] inline MissContext MakeMissContext(const SupervisorStats&     stats,
                                                 const GenerationRowInputs& rows) {
	MissContext context;
	context.first_predict_ns = stats.first_predict_ns;
	context.finished_ns      = stats.finished_ns;
	for (const auto& row: rows.generated) {
		context.generated_by_id.emplace(row.id, row);
	}
	for (const auto& row: rows.refused) {
		context.refused_by_id.emplace(row.id, row.verdict);
	}
	for (const auto& row: stats.predictions) {
		context.prediction_by_hash[row.hash] = {row.outcome, row.candidates};
	}
	for (const auto& drop: stats.dropped) {
		context.dropped_hashes.insert(drop.hash);
	}
	// A program that crashed in prediction is in skip_hashes and one that crashed in a build is in
	// failed_hashes, and neither was served.
	context.failed_hashes.insert(stats.failed_hashes.begin(), stats.failed_hashes.end());
	context.failed_hashes.insert(stats.skip_hashes.begin(), stats.skip_hashes.end());
	return context;
}

/// Copies the totals of the classes into the report, every class in the classifier's order even
/// when it is empty, so the shape of the block does not depend on the run.
inline void FillMissReport(Common::FirstRunReport::GenerationMisses& out,
                           const MissSummary&                        summary) {
	out.total     = summary.total;
	out.create_ms = summary.create_ms;
	for (size_t index = 0; index < MISS_CLASS_COUNT; index++) {
		out.classes.push_back({.name      = MissClassName(static_cast<MissClass>(index)),
		                       .count     = summary.classes[index].count,
		                       .create_ms = summary.classes[index].create_ms});
	}
	for (size_t verdict = 0; verdict < summary.refused_by_verdict.size(); verdict++) {
		out.refused.emplace_back(DropVerdictName(static_cast<DropVerdict>(verdict)),
		                         summary.refused_by_verdict[verdict]);
	}
	out.wrong_spec_programs     = summary.wrong_spec_programs;
	out.wrong_spec_max_variants = summary.wrong_spec_max_variants;
}

/// The name of how a first use got its build.
[[nodiscard]] inline const char* WarmTakeHowName(Common::FirstRunReport::WarmTakeHow how) {
	using Common::FirstRunReport::WarmTakeHow;
	return how == WarmTakeHow::Done ? "done" : how == WarmTakeHow::Waited ? "waited" : "inline";
}

/// One row for each module taken in, joined with its first use. A module no dispatch took is
/// unused, and its build time is what generation spent on it for nothing.
inline void FillModuleRows(Common::FirstRunReport::ShaderStoreGenerationReport& block,
                           const GenerationRowInputs&                           rows) {
	std::unordered_map<uint64_t, const Common::FirstRunReport::WarmTake*> first_use;
	for (const auto& take: rows.takes) {
		first_use.emplace(take.id, &take);
	}
	for (const auto& module: rows.generated) {
		Common::FirstRunReport::GeneratedModuleRow row {.id         = module.id,
		                                                .code_hash  = module.code_hash,
		                                                .posted_ns  = module.posted_ns,
		                                                .adopted_ns = module.adopted_ns,
		                                                .ready_ns   = module.ready_ns,
		                                                .build_ms   = module.build_ms,
		                                                .how        = "unused"};
		if (const auto take = first_use.find(module.id); take != first_use.end()) {
			row.how = WarmTakeHowName(take->second->how);
			if (take->second->at_ns != Common::FirstRunReport::NO_TIME_NS) {
				row.first_use_ns = take->second->at_ns;
				if (module.ready_ns) {
					row.lead_ms =
					    static_cast<double>(*row.first_use_ns - *module.ready_ns) / ROW_NS_PER_MS;
				}
			}
		} else if (module.build_ms) {
			block.unused_build_ms += *module.build_ms;
		}
		block.generated_modules.push_back(std::move(row));
	}
}

/// One row for each first use that stopped a dispatch for more than SLOW_TAKE_MS and was not a
/// finished build. A take of a generated module also shows when it was posted, taken in and built.
inline void FillSlowTakes(Common::FirstRunReport::ShaderStoreGenerationReport& block,
                          const GenerationRowInputs&                           rows) {
	std::unordered_map<uint64_t, const GenRowSnapshot*> generated;
	for (const auto& module: rows.generated) {
		generated.emplace(module.id, &module);
	}
	for (const auto& take: rows.takes) {
		if (take.how == Common::FirstRunReport::WarmTakeHow::Done ||
		    take.stopped_ms <= Common::FirstRunReport::SLOW_TAKE_MS) {
			continue;
		}
		Common::FirstRunReport::GenerationSlowTake row {.id         = take.id,
		                                                .how        = WarmTakeHowName(take.how),
		                                                .stopped_ms = take.stopped_ms,
		                                                .at_ns      = take.at_ns,
		                                                .paused     = take.paused};
		if (const auto module = generated.find(take.id); module != generated.end()) {
			row.generated  = true;
			row.posted_ns  = module->second->posted_ns;
			row.adopted_ns = module->second->adopted_ns;
			row.ready_ns   = module->second->ready_ns;
		}
		block.slow_takes.push_back(std::move(row));
	}
}

/// Fills the rows of the block: the generated modules, the misses by class and the slow takes.
inline void FillRowsReport(Common::FirstRunReport::ShaderStoreGenerationReport& block,
                           const SupervisorStats& stats, const GenerationRowInputs& rows) {
	FillModuleRows(block, rows);
	FillSlowTakes(block, rows);
	FillMissReport(block.misses,
	               SummarizeMisses(rows.misses, MakeMissContext(stats, rows), rows.compiles));
}

/// The whole generation block from the supervisor's statistics and the drain's figures.
[[nodiscard]] inline Common::FirstRunReport::ShaderStoreGenerationReport
MakeGenerationReport(const SupervisorStats& stats, const GeneratedDrainCounters& drain,
                     const GenerationCounts& counts) {
	auto block = MakeSupervisorReport(stats);
	FillDrainReport(block.drain, drain, counts);
	return block;
}

/// The same block with its rows, which the pipeline cache gives from its own state.
[[nodiscard]] inline Common::FirstRunReport::ShaderStoreGenerationReport
MakeGenerationReport(const SupervisorStats& stats, const GeneratedDrainCounters& drain,
                     const GenerationCounts& counts, const GenerationRowInputs& rows) {
	auto block = MakeGenerationReport(stats, drain, counts);
	FillRowsReport(block, stats, rows);
	return block;
}

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_GENERATION_REPORT_H
