#ifndef GRAPHICS_SHADER_STOREGEN_MISS_CLASSIFIER_H
#define GRAPHICS_SHADER_STOREGEN_MISS_CLASSIFIER_H

#include "common/firstRunReport.h"
#include "graphics/shader/storegen/generationRows.h"
#include "graphics/shader/storegen/progressLog.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// Says why a compute translation that the runtime had to do was not served by generation. It is a
// pure function of rows and needs no device, so the report's writer runs it at report time and a
// test runs it over made-up rows.
namespace Libs::Graphics::StoreGen {

/// The reasons, in the order the classifier tries them. The first that applies wins.
enum class MissClass : uint8_t {
	/// The miss came before the worker had written its first `predict` line, so it is a matter of
	/// how long the worker took to start, and no prediction could have served it.
	BeforePrediction,
	/// The module exists in the generator's output, but it was posted after the miss. It is also
	/// the class of a module the drain refused as a Duplicate after the miss, since the runtime
	/// built it first and the generator's copy then arrived late.
	NotYetGenerated,
	/// It was posted before the miss and was taken in after it, so it waited in the inbox or the
	/// backlog.
	InInbox,
	/// It was taken in before the miss, which contradicts the id contract and should not happen.
	AdoptedUnmatched,
	/// The generator made it and the drain refused it. A Duplicate refusal after the miss is
	/// NotYetGenerated instead. Every other verdict stays here whatever the time, since it is not
	/// about lateness, as does a Duplicate before the miss or with no time recorded.
	DroppedAtDrain,
	/// The runtime skips the program, which the worker leaves out by design.
	SkipDispatch,
	/// The runtime's program has GPU-selected calls, which the worker leaves out by design.
	HasCalls,
	/// The worker rejected the program for calls, descriptors it could not name, an unreadable
	/// header or a resource outside its rules.
	RejectedCalls,
	RejectedUnclassified,
	RejectedUnreadable,
	RejectedProblems,
	/// The worker gave up on the program after a push start or resource count check.
	ProgramDropped,
	/// A worker crashed or hung on the program, so the supervisor kept it out of later runs.
	ProgramFailed,
	/// The program was predicted and the worker had not finished when the miss came, so the
	/// candidate may simply not have been written yet. Finished means the supervisor's outcome is
	/// Finished, so after a Fatal, Capped or Stopped run every predicted program that was missed
	/// is counted here and none is wrong_spec.
	GenerationIncomplete,
	/// The program was predicted and generation had finished, so none of the worker's variants was
	/// the one the game used.
	WrongSpec,
	/// The worker never named the program.
	Unscanned,
};

/// How many classes there are, which sizes the totals.
constexpr size_t MISS_CLASS_COUNT = static_cast<size_t>(MissClass::Unscanned) + 1;

/// The name of a class in the report.
[[nodiscard]] inline const char* MissClassName(MissClass miss_class) {
	switch (miss_class) {
		case MissClass::BeforePrediction: return "before_prediction";
		case MissClass::NotYetGenerated: return "not_yet_generated";
		case MissClass::InInbox: return "in_inbox";
		case MissClass::AdoptedUnmatched: return "adopted_unmatched";
		case MissClass::DroppedAtDrain: return "dropped_at_drain";
		case MissClass::SkipDispatch: return "skip_dispatch";
		case MissClass::HasCalls: return "has_calls";
		case MissClass::RejectedCalls: return "rejected_calls";
		case MissClass::RejectedUnclassified: return "rejected_unclassified";
		case MissClass::RejectedUnreadable: return "rejected_unreadable";
		case MissClass::RejectedProblems: return "rejected_problems";
		case MissClass::ProgramDropped: return "program_dropped";
		case MissClass::ProgramFailed: return "program_failed";
		case MissClass::GenerationIncomplete: return "generation_incomplete";
		case MissClass::WrongSpec: return "wrong_spec";
		case MissClass::Unscanned: return "unscanned";
	}
	return "";
}

/// What a worker's prediction line said about a program.
struct PredictionFact {
	PredictionOutcome outcome    = PredictionOutcome::Predicted;
	uint64_t          candidates = 0;
};

/// Everything the classifier looks up. Build it once and classify every miss with it.
struct MissContext {
	/// When the worker's first `predict` line was found. Missing means none was, so every miss is
	/// before the prediction.
	std::optional<int64_t> first_predict_ns;
	/// When the worker finished. Missing means it has not, or ended some other way.
	std::optional<int64_t>                       finished_ns;
	std::unordered_map<uint64_t, GenRowSnapshot> generated_by_id;
	std::unordered_map<uint64_t, DropVerdict>    refused_by_id;
	/// When the drain refused each id, for the ids that have a time. An id in refused_by_id with no
	/// entry here has no refusal time.
	std::unordered_map<uint64_t, int64_t>        refused_ns_by_id;
	std::unordered_map<uint64_t, PredictionFact> prediction_by_hash;
	std::unordered_set<uint64_t>                 dropped_hashes;
	/// Programs that a worker crashed or hung on, whether in prediction (skipped) or in a build
	/// (failed).
	std::unordered_set<uint64_t> failed_hashes;
};

/// The class of one miss, by the order of MissClass.
[[nodiscard]] inline MissClass ClassifyMiss(const MissRow& miss, const MissContext& context) {
	if (!context.first_predict_ns || miss.at_ns < *context.first_predict_ns) {
		return MissClass::BeforePrediction;
	}
	if (const auto row = context.generated_by_id.find(miss.id);
	    miss.id != 0 && row != context.generated_by_id.end()) {
		if (row->second.posted_ns > miss.at_ns) {
			return MissClass::NotYetGenerated;
		}
		return row->second.adopted_ns > miss.at_ns ? MissClass::InInbox
		                                           : MissClass::AdoptedUnmatched;
	}
	if (miss.id != 0 && context.refused_by_id.contains(miss.id)) {
		// A Duplicate after the miss means the runtime had built the module before the drain saw
		// the generator's copy, so the module was generated too late. Other verdicts are not about
		// lateness, and a refusal before the miss or with no time is a module the drain lost.
		const auto refused_at = context.refused_ns_by_id.find(miss.id);
		if (context.refused_by_id.at(miss.id) == DropVerdict::Duplicate &&
		    refused_at != context.refused_ns_by_id.end() && refused_at->second > miss.at_ns) {
			return MissClass::NotYetGenerated;
		}
		return MissClass::DroppedAtDrain;
	}
	if (miss.skip_dispatch) {
		return MissClass::SkipDispatch;
	}
	if (miss.has_calls) {
		return MissClass::HasCalls;
	}
	const auto prediction = context.prediction_by_hash.find(miss.code_hash);
	if (prediction != context.prediction_by_hash.end() &&
	    prediction->second.outcome != PredictionOutcome::Predicted) {
		switch (prediction->second.outcome) {
			case PredictionOutcome::Calls: return MissClass::RejectedCalls;
			case PredictionOutcome::Unclassified: return MissClass::RejectedUnclassified;
			case PredictionOutcome::Unreadable: return MissClass::RejectedUnreadable;
			default: return MissClass::RejectedProblems;
		}
	}
	if (context.dropped_hashes.contains(miss.code_hash)) {
		return MissClass::ProgramDropped;
	}
	if (context.failed_hashes.contains(miss.code_hash)) {
		return MissClass::ProgramFailed;
	}
	if (prediction == context.prediction_by_hash.end()) {
		return MissClass::Unscanned;
	}
	if (!context.finished_ns || miss.at_ns < *context.finished_ns) {
		return MissClass::GenerationIncomplete;
	}
	return MissClass::WrongSpec;
}

/// A class's count, and the time the GPU thread spent creating the modules of its misses.
struct MissClassTotals {
	uint64_t count     = 0;
	double   create_ms = 0.0;
};

/// How many late modules a summary lists. There are few in a real run, and the cap keeps a run that
/// has thousands from filling the report. The rest are only counted.
constexpr size_t LATE_MODULE_LIMIT = 64;

/// How the generator delivered a module that the runtime needed first.
enum class LateVia : uint8_t {
	/// The drain adopted it, and the time is when the generator posted it.
	Adopted,
	/// The drain refused it as a Duplicate, and the time is when it refused.
	Refused,
};

/// One module that the runtime needed before the generator delivered it, with the time of its first
/// NotYetGenerated miss. A module that missed more than once has one row, and misses counts them.
/// The time of delivery is what tells the worker's generation order how far behind the game it was.
struct LateModule {
	uint64_t id           = 0;
	uint64_t code_hash    = 0;
	int64_t  miss_ns      = NO_ROW_TIME_NS;
	int64_t  delivered_ns = NO_ROW_TIME_NS;
	LateVia  via          = LateVia::Adopted;
	/// What the GPU thread spent creating the module for its first miss. Later misses of the same
	/// id add nothing, since the compile time goes to the first.
	double create_ms = 0.0;
	/// How many NotYetGenerated misses the module had.
	uint64_t misses = 1;
	/// When the GPU thread took an adopted module in, so the drain's lag is delivered to adopted.
	/// Missing for a refused module, which was never adopted.
	std::optional<int64_t> adopted_ns;
};

/// When and how the generator delivered the module of a miss that ClassifyMiss called
/// NotYetGenerated. It follows the same lookups in the same order, so the two cannot disagree on
/// which row applies.
struct LateDeliveryInfo {
	int64_t                delivered_ns = NO_ROW_TIME_NS;
	LateVia                via          = LateVia::Adopted;
	std::optional<int64_t> adopted_ns;
};

/// Looks up the delivery of the module of a NotYetGenerated miss.
[[nodiscard]] inline LateDeliveryInfo LateDelivery(const MissRow&     miss,
                                                   const MissContext& context) {
	if (const auto row = context.generated_by_id.find(miss.id);
	    row != context.generated_by_id.end()) {
		return {row->second.posted_ns, LateVia::Adopted, row->second.adopted_ns};
	}
	return {context.refused_ns_by_id.at(miss.id), LateVia::Refused, std::nullopt};
}

/// Which late modules are listed or counted as overflow, by id, so a repeat miss of an id is not a
/// second module.
struct LateTracker {
	std::unordered_map<uint64_t, size_t> listed_index;
	std::unordered_set<uint64_t>         overflow_ids;
};

/// The misses of a run by class.
struct MissSummary {
	std::array<MissClassTotals, MISS_CLASS_COUNT> classes;
	uint64_t                                      total     = 0;
	double                                        create_ms = 0.0;
	/// How the misses of DroppedAtDrain were refused, by verdict in DropVerdict order. Refusals
	/// that were a Duplicate after their miss are classed NotYetGenerated, so Duplicate counts
	/// only those before the miss or untimed, and the other verdicts count in full.
	std::array<uint64_t, 4> refused_by_verdict {};
	/// Distinct programs among the WrongSpec misses, and the most modules of one of them. A few
	/// programs with many variants point at a specification rule, and many programs at a general
	/// miss.
	uint64_t wrong_spec_programs     = 0;
	uint64_t wrong_spec_max_variants = 0;
	/// The modules of the NotYetGenerated misses, one each, in the order of their first miss, up to
	/// LATE_MODULE_LIMIT, and how many more modules there were.
	std::vector<LateModule> late_modules;
	uint64_t                late_overflow = 0;
};

/// The time that the GPU thread's compute pipeline compiles took, by module id. The miss's own
/// compile is the row with its id.
[[nodiscard]] inline std::unordered_map<uint64_t, double>
GpuComputeCreateMsById(const std::vector<Common::FirstRunReport::CompileEntry>& compiles) {
	std::unordered_map<uint64_t, double> by_id;
	for (const auto& compile: compiles) {
		if (compile.kind == Common::FirstRunReport::Kind::Compute &&
		    compile.thread == Common::FirstRunReport::Thread::Gpu) {
			by_id[compile.id] += compile.create_ms;
		}
	}
	return by_id;
}

/// Lists the module of a NotYetGenerated miss in the summary. A module already listed only has its
/// miss count raised, so the first miss keeps the times and compile time. Once the list is full a
/// new module is counted once as overflow, and its repeats are not counted again.
inline void AddLateModule(MissSummary& summary, LateTracker& tracker, const MissRow& miss,
                          const MissContext& context, double create_ms) {
	if (const auto listed = tracker.listed_index.find(miss.id);
	    listed != tracker.listed_index.end()) {
		summary.late_modules[listed->second].misses++;
		return;
	}
	if (summary.late_modules.size() >= LATE_MODULE_LIMIT) {
		if (tracker.overflow_ids.insert(miss.id).second) {
			summary.late_overflow++;
		}
		return;
	}
	const auto delivery = LateDelivery(miss, context);
	tracker.listed_index.emplace(miss.id, summary.late_modules.size());
	summary.late_modules.push_back({.id           = miss.id,
	                                .code_hash    = miss.code_hash,
	                                .miss_ns      = miss.at_ns,
	                                .delivered_ns = delivery.delivered_ns,
	                                .via          = delivery.via,
	                                .create_ms    = create_ms,
	                                .adopted_ns   = delivery.adopted_ns});
}

/// Classifies every miss and totals the classes. A module's compile time goes to the first miss
/// with its id only, so two rows of one id do not count it twice. The module of each
/// NotYetGenerated miss is also listed once, with the time it was delivered.
[[nodiscard]] inline MissSummary
SummarizeMisses(const std::vector<MissRow>& misses, const MissContext& context,
                const std::vector<Common::FirstRunReport::CompileEntry>& compiles) {
	MissSummary summary;
	LateTracker late_tracker;
	auto        create_ms = GpuComputeCreateMsById(compiles);
	std::unordered_map<uint64_t, std::unordered_set<uint64_t>> wrong_spec_ids;
	for (const auto& miss: misses) {
		const auto miss_class = ClassifyMiss(miss, context);
		auto&      totals     = summary.classes[static_cast<size_t>(miss_class)];
		totals.count++;
		summary.total++;
		double miss_create_ms = 0.0;
		if (const auto found = create_ms.find(miss.id); miss.id != 0 && found != create_ms.end()) {
			miss_create_ms = found->second;
			totals.create_ms += found->second;
			summary.create_ms += found->second;
			create_ms.erase(found);
		}
		if (miss_class == MissClass::NotYetGenerated) {
			AddLateModule(summary, late_tracker, miss, context, miss_create_ms);
		}
		if (miss_class == MissClass::DroppedAtDrain) {
			summary.refused_by_verdict[static_cast<size_t>(context.refused_by_id.at(miss.id))]++;
		}
		if (miss_class == MissClass::WrongSpec) {
			wrong_spec_ids[miss.code_hash].insert(miss.id);
		}
	}
	summary.wrong_spec_programs = wrong_spec_ids.size();
	for (const auto& [hash, ids]: wrong_spec_ids) {
		summary.wrong_spec_max_variants =
		    std::max<uint64_t>(summary.wrong_spec_max_variants, ids.size());
	}
	return summary;
}

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_MISS_CLASSIFIER_H
