// Tests the classifier that says why a runtime translation was not served by generation, and the
// rows of the report that are made from the pipeline cache's rows. Both are pure, so no device is
// needed. Each check names what else could pass it and compares against a baseline that differs in
// one thing, and the order in which the classifier tries the classes is tested with rows that would
// satisfy two classes at once.
#include "graphics/shader/storegen/generationReport.h"
#include "graphics/shader/storegen/missClassifier.h"

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>

namespace {

using namespace Libs::Graphics::StoreGen;
namespace Report = Common::FirstRunReport;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

constexpr uint64_t PROGRAM       = 0xA0A0A0A0A0A0A0A0ull;
constexpr uint64_t OTHER_PROGRAM = 0xB0B0B0B0B0B0B0B0ull;
constexpr uint64_t MODULE_ID     = 7;
// A run in nanoseconds. The worker's first predict line comes at 1000, it finishes at 5000, and the
// miss is at 3000, in between.
constexpr int64_t FIRST_PREDICT = 1000;
constexpr int64_t FINISHED      = 5000;
constexpr int64_t MISS_AT       = 3000;

MissRow Miss() {
	return {.code_hash = PROGRAM, .id = MODULE_ID, .at_ns = MISS_AT};
}

// A context where the worker is running between its first predict line and its end, and has said
// nothing about any program.
MissContext Context() {
	MissContext context;
	context.first_predict_ns = FIRST_PREDICT;
	context.finished_ns      = FINISHED;
	return context;
}

void Predicted(MissContext& context, uint64_t hash = PROGRAM,
               PredictionOutcome outcome = PredictionOutcome::Predicted) {
	context.prediction_by_hash[hash] = {outcome, outcome == PredictionOutcome::Predicted ? 2u : 0u};
}

GenRowSnapshot GenRowAt(int64_t posted, int64_t adopted) {
	return {.id = MODULE_ID, .code_hash = PROGRAM, .posted_ns = posted, .adopted_ns = adopted};
}

MissClass Of(const MissRow& miss, const MissContext& context) {
	return ClassifyMiss(miss, context);
}

// One class and the one change that moves a miss out of it. The baseline is the same miss with that
// change undone, which is in the next class that applies, so a classifier that always answered the
// class would fail the baseline.
struct Row {
	const char*                                 name;
	std::function<void(MissRow&, MissContext&)> setup;
	MissClass                                   expected;
	std::function<void(MissRow&, MissContext&)> undo;
	MissClass                                   baseline;
};

void TestEveryClassAgainstItsBaseline() {
	const std::vector<Row> rows {
	    {"before_prediction", [](MissRow& m, MissContext&) { m.at_ns = FIRST_PREDICT - 1; },
	     MissClass::BeforePrediction, [](MissRow& m, MissContext&) { m.at_ns = FIRST_PREDICT; },
	     MissClass::Unscanned},
	    {"before_prediction with no predict line",
	     [](MissRow&, MissContext& c) { c.first_predict_ns.reset(); }, MissClass::BeforePrediction,
	     [](MissRow&, MissContext& c) { c.first_predict_ns = FIRST_PREDICT; },
	     MissClass::Unscanned},
	    {"not_yet_generated",
	     [](MissRow&, MissContext& c) {
		     c.generated_by_id[MODULE_ID] = GenRowAt(MISS_AT + 1, MISS_AT + 2);
	     },
	     MissClass::NotYetGenerated,
	     [](MissRow&, MissContext& c) {
		     c.generated_by_id[MODULE_ID] = GenRowAt(MISS_AT - 1, MISS_AT + 2);
	     },
	     MissClass::InInbox},
	    {"in_inbox",
	     [](MissRow&, MissContext& c) {
		     c.generated_by_id[MODULE_ID] = GenRowAt(MISS_AT - 1, MISS_AT + 1);
	     },
	     MissClass::InInbox,
	     [](MissRow&, MissContext& c) {
		     c.generated_by_id[MODULE_ID] = GenRowAt(MISS_AT - 2, MISS_AT - 1);
	     },
	     MissClass::AdoptedUnmatched},
	    {"adopted_unmatched",
	     [](MissRow&, MissContext& c) {
		     c.generated_by_id[MODULE_ID] = GenRowAt(MISS_AT - 2, MISS_AT - 1);
	     },
	     MissClass::AdoptedUnmatched, [](MissRow&, MissContext& c) { c.generated_by_id.clear(); },
	     MissClass::Unscanned},
	    {"dropped_at_drain",
	     [](MissRow&, MissContext& c) { c.refused_by_id[MODULE_ID] = DropVerdict::PlanMismatch; },
	     MissClass::DroppedAtDrain, [](MissRow&, MissContext& c) { c.refused_by_id.clear(); },
	     MissClass::Unscanned},
	    {"skip_dispatch",
	     [](MissRow& m, MissContext&) {
		     m.skip_dispatch = true;
		     m.id            = 0;
	     },
	     MissClass::SkipDispatch, [](MissRow& m, MissContext&) { m.skip_dispatch = false; },
	     MissClass::Unscanned},
	    {"has_calls", [](MissRow& m, MissContext&) { m.has_calls = true; }, MissClass::HasCalls,
	     [](MissRow& m, MissContext&) { m.has_calls = false; }, MissClass::Unscanned},
	    {"rejected_calls",
	     [](MissRow&, MissContext& c) { Predicted(c, PROGRAM, PredictionOutcome::Calls); },
	     MissClass::RejectedCalls, [](MissRow&, MissContext& c) { Predicted(c); },
	     MissClass::GenerationIncomplete},
	    {"rejected_unclassified",
	     [](MissRow&, MissContext& c) { Predicted(c, PROGRAM, PredictionOutcome::Unclassified); },
	     MissClass::RejectedUnclassified, [](MissRow&, MissContext& c) { Predicted(c); },
	     MissClass::GenerationIncomplete},
	    {"rejected_unreadable",
	     [](MissRow&, MissContext& c) { Predicted(c, PROGRAM, PredictionOutcome::Unreadable); },
	     MissClass::RejectedUnreadable, [](MissRow&, MissContext& c) { Predicted(c); },
	     MissClass::GenerationIncomplete},
	    {"rejected_problems",
	     [](MissRow&, MissContext& c) { Predicted(c, PROGRAM, PredictionOutcome::Problems); },
	     MissClass::RejectedProblems, [](MissRow&, MissContext& c) { Predicted(c); },
	     MissClass::GenerationIncomplete},
	    {"program_dropped",
	     [](MissRow&, MissContext& c) {
		     Predicted(c);
		     c.dropped_hashes.insert(PROGRAM);
	     },
	     MissClass::ProgramDropped, [](MissRow&, MissContext& c) { c.dropped_hashes.clear(); },
	     MissClass::GenerationIncomplete},
	    {"program_failed",
	     [](MissRow&, MissContext& c) {
		     Predicted(c);
		     c.failed_hashes.insert(PROGRAM);
	     },
	     MissClass::ProgramFailed, [](MissRow&, MissContext& c) { c.failed_hashes.clear(); },
	     MissClass::GenerationIncomplete},
	    {"program_failed in prediction, which leaves no prediction row",
	     [](MissRow&, MissContext& c) { c.failed_hashes.insert(PROGRAM); },
	     MissClass::ProgramFailed, [](MissRow&, MissContext& c) { c.failed_hashes.clear(); },
	     MissClass::Unscanned},
	    {"generation_incomplete before the finish", [](MissRow&, MissContext& c) { Predicted(c); },
	     MissClass::GenerationIncomplete, [](MissRow& m, MissContext&) { m.at_ns = FINISHED + 1; },
	     MissClass::WrongSpec},
	    {"generation_incomplete with no finish",
	     [](MissRow& m, MissContext& c) {
		     Predicted(c);
		     c.finished_ns.reset();
		     m.at_ns = FINISHED + 1;
	     },
	     MissClass::GenerationIncomplete,
	     [](MissRow&, MissContext& c) { c.finished_ns = FINISHED; }, MissClass::WrongSpec},
	    {"wrong_spec",
	     [](MissRow& m, MissContext& c) {
		     Predicted(c);
		     m.at_ns = FINISHED + 1;
	     },
	     MissClass::WrongSpec, [](MissRow&, MissContext& c) { c.prediction_by_hash.clear(); },
	     MissClass::Unscanned},
	    {"unscanned", [](MissRow&, MissContext&) {}, MissClass::Unscanned,
	     [](MissRow&, MissContext& c) { Predicted(c); }, MissClass::GenerationIncomplete},
	};
	for (const auto& row: rows) {
		auto miss    = Miss();
		auto context = Context();
		row.setup(miss, context);
		if (Of(miss, context) != row.expected) {
			std::fprintf(stderr, "class row '%s': got %s, wanted %s\n", row.name,
			             MissClassName(Of(miss, context)), MissClassName(row.expected));
			++g_failures;
		}
		row.undo(miss, context);
		if (Of(miss, context) != row.baseline) {
			std::fprintf(stderr, "baseline of '%s': got %s, wanted %s\n", row.name,
			             MissClassName(Of(miss, context)), MissClassName(row.baseline));
			++g_failures;
		}
	}
}

// A miss that two classes would both claim goes to the earlier one. Each row sets up a miss for the
// earlier class and the later, and the baseline removes the earlier, which then gives the later.
void TestPrecedence() {
	// before_prediction beats a row that would be adopted_unmatched.
	{
		auto miss                          = Miss();
		auto context                       = Context();
		miss.at_ns                         = FIRST_PREDICT - 1;
		context.generated_by_id[MODULE_ID] = GenRowAt(0, 1);
		CHECK(Of(miss, context) == MissClass::BeforePrediction);
		miss.at_ns = MISS_AT;
		CHECK(Of(miss, context) == MissClass::AdoptedUnmatched);
	}
	// A generated row beats a refusal of the same id.
	{
		auto miss                          = Miss();
		auto context                       = Context();
		context.generated_by_id[MODULE_ID] = GenRowAt(MISS_AT - 1, MISS_AT + 1);
		context.refused_by_id[MODULE_ID]   = DropVerdict::Duplicate;
		CHECK(Of(miss, context) == MissClass::InInbox);
		context.generated_by_id.clear();
		CHECK(Of(miss, context) == MissClass::DroppedAtDrain);
	}
	// A refusal beats has_calls, and skip_dispatch beats has_calls.
	{
		auto miss                        = Miss();
		auto context                     = Context();
		miss.has_calls                   = true;
		context.refused_by_id[MODULE_ID] = DropVerdict::Duplicate;
		CHECK(Of(miss, context) == MissClass::DroppedAtDrain);
		context.refused_by_id.clear();
		CHECK(Of(miss, context) == MissClass::HasCalls);
		miss.skip_dispatch = true;
		CHECK(Of(miss, context) == MissClass::SkipDispatch);
	}
	// has_calls beats a rejected prediction, which beats a dropped program, which beats a failed
	// one, which beats an unfinished generation.
	{
		auto miss    = Miss();
		auto context = Context();
		Predicted(context, PROGRAM, PredictionOutcome::Problems);
		context.dropped_hashes.insert(PROGRAM);
		context.failed_hashes.insert(PROGRAM);
		miss.has_calls = true;
		CHECK(Of(miss, context) == MissClass::HasCalls);
		miss.has_calls = false;
		CHECK(Of(miss, context) == MissClass::RejectedProblems);
		Predicted(context);
		CHECK(Of(miss, context) == MissClass::ProgramDropped);
		context.dropped_hashes.clear();
		CHECK(Of(miss, context) == MissClass::ProgramFailed);
		context.failed_hashes.clear();
		CHECK(Of(miss, context) == MissClass::GenerationIncomplete);
	}
	// A miss of another program is unaffected by the first program's rows.
	{
		auto miss    = Miss();
		auto context = Context();
		Predicted(context, OTHER_PROGRAM, PredictionOutcome::Calls);
		context.dropped_hashes.insert(OTHER_PROGRAM);
		CHECK(Of(miss, context) == MissClass::Unscanned);
	}
}

// The classes are totalled with the time of the compile of each miss's module. The compile of a
// worker, of graphics and of another module does not count, and a module counts once.
void TestSummary() {
	auto context = Context();
	Predicted(context);
	std::vector<MissRow> misses;
	misses.push_back(Miss());
	auto second = Miss();
	second.id   = MODULE_ID + 1;
	misses.push_back(second);
	auto third  = Miss();
	third.id    = MODULE_ID + 2;
	third.at_ns = FINISHED + 1;
	misses.push_back(third);
	auto repeat = Miss();
	misses.push_back(repeat);

	const auto row = [](Report::Kind kind, Report::Thread thread, uint64_t id, double ms) {
		Report::CompileEntry entry;
		entry.kind      = kind;
		entry.thread    = thread;
		entry.id        = id;
		entry.create_ms = ms;
		return entry;
	};
	const std::vector<Report::CompileEntry> compiles {
	    row(Report::Kind::Compute, Report::Thread::Gpu, MODULE_ID, 10.0),
	    row(Report::Kind::Compute, Report::Thread::Worker, MODULE_ID, 1000.0),
	    row(Report::Kind::Graphics, Report::Thread::Gpu, MODULE_ID, 2000.0),
	    row(Report::Kind::Compute, Report::Thread::Gpu, MODULE_ID + 2, 5.0),
	    row(Report::Kind::Compute, Report::Thread::Gpu, 999, 4000.0),
	};
	const auto summary    = SummarizeMisses(misses, context, compiles);
	const auto incomplete = static_cast<size_t>(MissClass::GenerationIncomplete);
	const auto wrong_spec = static_cast<size_t>(MissClass::WrongSpec);
	CHECK(summary.total == 4);
	CHECK(summary.classes[incomplete].count == 3);
	CHECK(summary.classes[wrong_spec].count == 1);
	// The first miss takes the 10 ms and the repeat of its id takes nothing. The second has no
	// compile row, and the third has 5 ms.
	CHECK(summary.classes[incomplete].create_ms == 10.0);
	CHECK(summary.classes[wrong_spec].create_ms == 5.0);
	CHECK(summary.create_ms == 15.0);

	// The variants of wrong_spec: two modules of one program and one of another.
	std::vector<MissRow> wrong;
	for (const auto& [hash, id]:
	     {std::pair {PROGRAM, 1ull}, {PROGRAM, 2ull}, {OTHER_PROGRAM, 3ull}, {PROGRAM, 2ull}}) {
		MissRow miss {.code_hash = hash, .id = id, .at_ns = FINISHED + 1};
		wrong.push_back(miss);
	}
	auto wrong_context = Context();
	Predicted(wrong_context, PROGRAM);
	Predicted(wrong_context, OTHER_PROGRAM);
	const auto spec = SummarizeMisses(wrong, wrong_context, {});
	CHECK(spec.classes[wrong_spec].count == 4);
	// The same module twice is one variant, so the first program has two and the other one.
	CHECK(spec.wrong_spec_programs == 2 && spec.wrong_spec_max_variants == 2);

	// The refusals of dropped_at_drain by verdict.
	auto refusing              = Context();
	refusing.refused_by_id[10] = DropVerdict::Duplicate;
	refusing.refused_by_id[11] = DropVerdict::PlanMismatch;
	refusing.refused_by_id[12] = DropVerdict::PlanMismatch;
	std::vector<MissRow> refused;
	for (uint64_t id: {10ull, 11ull, 12ull}) {
		refused.push_back({.code_hash = PROGRAM, .id = id, .at_ns = MISS_AT});
	}
	const auto refusals = SummarizeMisses(refused, refusing, {});
	CHECK(refusals.refused_by_verdict[static_cast<size_t>(DropVerdict::Duplicate)] == 1);
	CHECK(refusals.refused_by_verdict[static_cast<size_t>(DropVerdict::PlanMismatch)] == 2);
	CHECK(refusals.refused_by_verdict[static_cast<size_t>(DropVerdict::FailedInstall)] == 0);
}

Report::WarmTake Take(uint64_t id, Report::WarmTakeHow how, double stopped_ms, int64_t at_ns,
                      bool paused = false) {
	Report::WarmTake take;
	take.id         = id;
	take.how        = how;
	take.stopped_ms = stopped_ms;
	take.at_ns      = at_ns;
	take.paused     = paused;
	return take;
}

GenRowSnapshot Module(uint64_t id, int64_t posted, int64_t adopted, std::optional<int64_t> ready,
                      std::optional<double> built) {
	return {.id         = id,
	        .code_hash  = PROGRAM,
	        .posted_ns  = posted,
	        .adopted_ns = adopted,
	        .ready_ns   = ready,
	        .build_ms   = built};
}

// Each module taken in gets a row joined with its first use. A module no dispatch took is unused
// and its build time is the block's unused_build_ms, and a module that was taken adds nothing to
// it.
void TestModuleRows() {
	using How              = Report::WarmTakeHow;
	constexpr int64_t   MS = 1'000'000;
	GenerationRowInputs rows;
	rows.generated = {Module(1, 1 * MS, 2 * MS, 10 * MS, 4.0), // taken after it was ready
	                  Module(2, 1 * MS, 2 * MS, 10 * MS, 6.0), // taken before it was ready
	                  Module(3, 1 * MS, 2 * MS, 12 * MS, 8.0), // never taken
	                  Module(4, 1 * MS, 2 * MS, std::nullopt, std::nullopt), // never built
	                  Module(5, 1 * MS, 2 * MS, 11 * MS, 16.0)};             // taken with no time
	rows.takes     = {Take(1, How::Done, 0.0, 30 * MS), Take(2, How::Inline, 400.0, 7 * MS),
	                  Take(5, How::Waited, 1.0, Report::NO_TIME_NS)};
	SupervisorStats stats;
	auto            block = MakeSupervisorReport(stats);
	FillModuleRows(block, rows);
	CHECK(block.generated_modules.size() == 5);
	const auto& modules = block.generated_modules;
	CHECK(modules[0].how == "done" && modules[0].first_use_ns == 30 * MS);
	CHECK(modules[0].lead_ms && *modules[0].lead_ms == 20.0);
	CHECK(modules[1].how == "inline" && modules[1].lead_ms && *modules[1].lead_ms == -3.0);
	CHECK(modules[2].how == "unused" && !modules[2].first_use_ns && !modules[2].lead_ms);
	CHECK(modules[3].how == "unused" && !modules[3].build_ms && !modules[3].ready_ns);
	CHECK(modules[4].how == "waited" && !modules[4].first_use_ns && !modules[4].lead_ms);
	// Only the unused module with a build counts: 8, and not the 4 + 6 + 16 of the taken ones.
	CHECK(block.unused_build_ms == 8.0);
	CHECK(modules[0].id == 1 && modules[0].code_hash == PROGRAM && modules[0].posted_ns == 1 * MS &&
	      modules[0].adopted_ns == 2 * MS && modules[0].build_ms && *modules[0].build_ms == 4.0);
}

// A first use that stopped a dispatch for more than 250 ms and that was not a finished build gets a
// row. The rows that do not are at 250 exactly, below it, and done.
void TestSlowTakes() {
	using How              = Report::WarmTakeHow;
	constexpr int64_t   MS = 1'000'000;
	GenerationRowInputs rows;
	rows.generated = {Module(1, 1 * MS, 2 * MS, 9 * MS, 3.0)};
	rows.takes = {Take(1, How::Waited, 300.0, 20 * MS, true), Take(2, How::Inline, 250.1, 21 * MS),
	              Take(3, How::Inline, 250.0, 22 * MS), Take(4, How::Waited, 249.0, 23 * MS),
	              Take(5, How::Done, 900.0, 24 * MS)};
	SupervisorStats stats;
	auto            block = MakeSupervisorReport(stats);
	FillSlowTakes(block, rows);
	CHECK(block.slow_takes.size() == 2);
	if (block.slow_takes.size() == 2) {
		const auto& generated = block.slow_takes[0];
		CHECK(generated.id == 1 && generated.generated && generated.how == "waited" &&
		      generated.paused && generated.stopped_ms == 300.0 && generated.at_ns == 20 * MS);
		CHECK(generated.posted_ns == 1 * MS && generated.adopted_ns == 2 * MS &&
		      generated.ready_ns == 9 * MS);
		const auto& player = block.slow_takes[1];
		CHECK(player.id == 2 && !player.generated && player.how == "inline" && !player.paused);
		CHECK(!player.posted_ns && !player.adopted_ns && !player.ready_ns);
	}
}

// The whole block carries the prediction rows and every class in the classifier's order, empty or
// not, so the shape of the block does not depend on the run.
void TestBlockShape() {
	SupervisorStats stats;
	stats.predictions      = {{PROGRAM, PredictionOutcome::Predicted, 3},
	                          {OTHER_PROGRAM, PredictionOutcome::Unreadable, 0}};
	stats.first_predict_ns = FIRST_PREDICT;
	stats.finished_ns      = FINISHED;
	GenerationRowInputs rows;
	rows.misses      = {Miss()};
	const auto block = MakeGenerationReport(stats, {}, {}, rows);
	CHECK(block.predictions.size() == 2 && block.predictions[0].hash == PROGRAM &&
	      block.predictions[0].outcome == "predicted" && block.predictions[0].candidates == 3);
	CHECK(block.predictions[1].outcome == "unreadable" && block.predictions[1].candidates == 0);
	CHECK(block.misses.classes.size() == MISS_CLASS_COUNT && block.misses.total == 1);
	CHECK(block.misses.classes[0].name == "before_prediction");
	CHECK(block.misses.classes.back().name == "unscanned");
	// The miss at 3000 is of a program the worker predicted, and it came before the worker
	// finished.
	uint64_t counted = 0;
	for (const auto& miss_class: block.misses.classes) {
		counted += miss_class.count;
		CHECK(miss_class.count == (miss_class.name == "generation_incomplete" ? 1u : 0u));
	}
	CHECK(counted == 1);
	CHECK(block.misses.refused.size() == 4 && block.misses.refused[0].first == "plan_mismatch");
}

// The classifier's lookups are made from the supervisor's statistics. A program that a worker
// crashed on while predicting is in skip_hashes and one that crashed in a build is in
// failed_hashes, and both are program_failed. The baseline is the same miss with a statistics that
// name neither.
void TestContextFromStats() {
	SupervisorStats stats;
	stats.first_predict_ns = FIRST_PREDICT;
	stats.finished_ns      = FINISHED;
	stats.predictions      = {{PROGRAM, PredictionOutcome::Calls, 0},
	                          {OTHER_PROGRAM, PredictionOutcome::Predicted, 2}};
	GenerationRowInputs rows;
	rows.generated = {GenRowAt(MISS_AT - 1, MISS_AT + 1)};
	rows.refused   = {{.id = MODULE_ID + 1, .verdict = DropVerdict::PlanMismatch}};
	auto context   = MakeMissContext(stats, rows);
	CHECK(context.first_predict_ns == FIRST_PREDICT && context.finished_ns == FINISHED);
	CHECK(context.generated_by_id.contains(MODULE_ID) &&
	      context.refused_by_id.contains(MODULE_ID + 1));
	CHECK(context.prediction_by_hash.at(PROGRAM).outcome == PredictionOutcome::Calls &&
	      context.prediction_by_hash.at(OTHER_PROGRAM).candidates == 2);
	CHECK(context.dropped_hashes.empty() && context.failed_hashes.empty());

	const uint64_t crashed_predicting = 0xC1;
	const uint64_t crashed_building   = 0xC2;
	const uint64_t given_up           = 0xC3;
	stats.skip_hashes                 = {crashed_predicting};
	stats.failed_hashes               = {crashed_building};
	stats.dropped = {{.hash = given_up, .rank = 1, .reason = ProgressDropReason::PushStart}};
	context       = MakeMissContext(stats, rows);
	const auto of = [&](uint64_t hash) {
		MissRow miss {.code_hash = hash, .id = 99, .at_ns = MISS_AT};
		return ClassifyMiss(miss, context);
	};
	CHECK(of(crashed_predicting) == MissClass::ProgramFailed);
	CHECK(of(crashed_building) == MissClass::ProgramFailed);
	CHECK(of(given_up) == MissClass::ProgramDropped);
	CHECK(of(0xC4) == MissClass::Unscanned);
}

} // namespace

int main() {
	TestEveryClassAgainstItsBaseline();
	TestPrecedence();
	TestSummary();
	TestContextFromStats();
	TestModuleRows();
	TestSlowTakes();
	TestBlockShape();
	if (g_failures == 0) {
		std::puts("Generation miss classifier tests passed");
	}
	return g_failures == 0 ? 0 : 1;
}
