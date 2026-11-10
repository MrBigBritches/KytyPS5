// Tests the decisions made when generated modules are taken into the pipeline cache: the tripwire
// on plans in both directions, the dedupe, and the budget of one drain. The planner needs no
// device, so nothing here does. Each check names what else could pass it and compares against a
// baseline that differs only in the behaviour under test.
#include "graphics/shader/storegen/generatedDrainPlanner.h"

#include <chrono>
#include <cstdint>
#include <cstdio>

namespace {

using namespace Libs::Graphics::StoreGen;
using namespace std::chrono_literals;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

constexpr uint64_t PROGRAM_ONE = 0x1001;
constexpr uint64_t PROGRAM_TWO = 0x1002;
constexpr uint64_t PLAN_A      = 0xAAAA000011112222ull;
constexpr uint64_t PLAN_B      = 0xBBBB000033334444ull;
// The hash of the empty plan, which a program that skips its dispatch has.
constexpr uint64_t EMPTY_PLAN = 0xE0E0E0E0E0E0E0E0ull;

// A module of a program, with a plan hash, that the cache would take in.
DrainFacts Facts(uint64_t program, uint64_t plan_hash, bool inserted = false,
                 bool id_present = false) {
	DrainFacts facts;
	facts.program          = program;
	facts.plan_hash        = plan_hash;
	facts.program_inserted = inserted;
	facts.id_present       = id_present;
	return facts;
}

// A source of the existing program's plan hash that counts how often it was asked.
struct ExistingPlan {
	uint64_t                  hash  = PLAN_A;
	int                       calls = 0;
	std::function<uint64_t()> Source() {
		return [this] {
			calls++;
			return hash;
		};
	}
};

// The budget stops a drain at its count or its time, takes the first module whatever the budget is,
// and takes more while both are in hand. Each row changes one of the two from a row that is
// allowed.
void TestBudget() {
	const DrainBudget budget;
	CHECK(budget.max_items == GENERATION_DRAIN_BUDGET &&
	      budget.max_time == GENERATION_DRAIN_TIME_BUDGET);
	CHECK(budget.Allows(1, 0ns));
	CHECK(budget.Allows(GENERATION_DRAIN_BUDGET - 1, GENERATION_DRAIN_TIME_BUDGET - 1ns));
	// The count is spent, with time left.
	CHECK(!budget.Allows(GENERATION_DRAIN_BUDGET, 0ns));
	// The time is spent, with count left.
	CHECK(!budget.Allows(1, GENERATION_DRAIN_TIME_BUDGET));
	CHECK(!budget.Allows(1, GENERATION_DRAIN_TIME_BUDGET + 1ms));
	// The first module is taken even when no time is allowed, and the second is not.
	const DrainBudget none {.max_items = 8, .max_time = 0ns};
	CHECK(none.Allows(0, 5ms));
	CHECK(!none.Allows(1, 0ns));
	const DrainBudget one {.max_items = 1, .max_time = 1s};
	CHECK(one.Allows(0, 0ns) && !one.Allows(1, 0ns));
}

// A new program takes its first module in, and the planner then holds its plan. A program whose
// plan could not be installed is not taken in and is forgotten, so the next module meets it as one
// that was there before. The stopped cache takes nothing and asks nothing.
void TestNewProgram() {
	GeneratedDrainPlanner planner;
	ExistingPlan          existing;
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_A, true), existing.Source()) ==
	      DrainVerdict::Adopt);
	CHECK(existing.calls == 0 && planner.Counters().adopted == 1);
	// Another module of the program with the same plan is taken in, with another plan it is not.
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_A), existing.Source()) == DrainVerdict::Adopt);
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_B), existing.Source()) ==
	      DrainVerdict::DropPlanMismatch);
	CHECK(existing.calls == 0);

	auto failed           = Facts(PROGRAM_TWO, PLAN_A, true);
	failed.plan_installed = false;
	CHECK(planner.Judge(failed, existing.Source()) == DrainVerdict::DropFailedInstall);
	CHECK(planner.Counters().failed_installs == 1);
	// Forgotten: the next module of the program asks for the plan of the program that is there.
	CHECK(planner.Judge(Facts(PROGRAM_TWO, PLAN_A), existing.Source()) == DrainVerdict::Adopt);
	CHECK(existing.calls == 1);

	auto stopped    = Facts(PROGRAM_ONE, PLAN_A);
	stopped.stopped = true;
	CHECK(planner.Judge(stopped, existing.Source()) == DrainVerdict::DropStopped);
	CHECK(planner.Counters().dropped_stopped == 1 && existing.calls == 1);
}

// The tripwire for a program that was there before: the first module compares its plan with the
// program's, once, and the verdict holds for the rest. The matching case is the baseline for the
// mismatching one, which differs only in the hash of the module.
void TestPlanTripwire() {
	GeneratedDrainPlanner planner;
	ExistingPlan          existing;
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_A), existing.Source()) == DrainVerdict::Adopt);
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_A), existing.Source()) == DrainVerdict::Adopt);
	// The existing plan was asked for once, for two modules.
	CHECK(existing.calls == 1);

	ExistingPlan other;
	CHECK(planner.Judge(Facts(PROGRAM_TWO, PLAN_B), other.Source()) ==
	      DrainVerdict::DropPlanMismatch);
	CHECK(planner.Counters().plan_mismatch == 1);
	// The verdict holds: a later module of the program is dropped, even one with the plan the
	// program has, and the program's plan is not asked for again.
	CHECK(planner.Judge(Facts(PROGRAM_TWO, PLAN_B), other.Source()) ==
	      DrainVerdict::DropPlanMismatch);
	CHECK(planner.Judge(Facts(PROGRAM_TWO, PLAN_A), other.Source()) ==
	      DrainVerdict::DropPlanMismatch);
	CHECK(other.calls == 1 && planner.Counters().plan_mismatch == 3);
}

// A program that skips its dispatch has the empty plan, so a generated module for it never matches
// and is dropped, which is right, since the runtime never runs such a program. The baseline is the
// same module with a program whose plan it has.
void TestSkipDispatchProgramAlwaysMismatches() {
	GeneratedDrainPlanner planner;
	ExistingPlan          skipping {.hash = EMPTY_PLAN};
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_A), skipping.Source()) ==
	      DrainVerdict::DropPlanMismatch);
	ExistingPlan normal {.hash = PLAN_A};
	CHECK(planner.Judge(Facts(PROGRAM_TWO, PLAN_A), normal.Source()) == DrainVerdict::Adopt);
}

// A module with an id that the program already holds is a duplicate. The same module with a new id
// is taken in, so it is the id that decides, and a mismatch is judged before it.
void TestDuplicate() {
	GeneratedDrainPlanner planner;
	ExistingPlan          existing;
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_A, false, true), existing.Source()) ==
	      DrainVerdict::DropDuplicate);
	CHECK(planner.Counters().duplicates == 1 && planner.Counters().adopted == 0);
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_A, false, false), existing.Source()) ==
	      DrainVerdict::Adopt);
	CHECK(planner.Judge(Facts(PROGRAM_ONE, PLAN_B, false, true), existing.Source()) ==
	      DrainVerdict::DropPlanMismatch);
	CHECK(planner.Counters().duplicates == 1);
}

// The check in the other direction runs once, for a program generation made and the runtime then
// translated, and counts a plan that is not the one the program was made with. A program that was
// there before, one the planner does not know, and one that was forgotten are not checked.
void TestReverseCheck() {
	GeneratedDrainPlanner planner;
	ExistingPlan          existing;
	(void)planner.Judge(Facts(PROGRAM_ONE, PLAN_A, true), existing.Source());
	int        calls   = 0;
	const auto runtime = [&](uint64_t hash) {
		return [&calls, hash] {
			calls++;
			return hash;
		};
	};
	CHECK(planner.CheckReverse(PROGRAM_ONE, runtime(PLAN_A)) == ReverseVerdict::Match);
	CHECK(calls == 1 && planner.Counters().plan_mismatch_reverse == 0);
	// Once: a second look is not made, whatever the runtime says.
	CHECK(planner.CheckReverse(PROGRAM_ONE, runtime(PLAN_B)) == ReverseVerdict::NotApplicable);
	CHECK(calls == 1 && planner.Counters().plan_mismatch_reverse == 0);

	(void)planner.Judge(Facts(PROGRAM_TWO, PLAN_A, true), existing.Source());
	CHECK(planner.CheckReverse(PROGRAM_TWO, runtime(PLAN_B)) == ReverseVerdict::Mismatch);
	CHECK(planner.Counters().plan_mismatch_reverse == 1);

	// A program that was there before is not one that generation made.
	const uint64_t before = 0x2001;
	(void)planner.Judge(Facts(before, PLAN_A), existing.Source());
	const int calls_before = calls;
	CHECK(planner.CheckReverse(before, runtime(PLAN_B)) == ReverseVerdict::NotApplicable);
	CHECK(planner.CheckReverse(0x3001, runtime(PLAN_B)) == ReverseVerdict::NotApplicable);
	CHECK(calls == calls_before);
	// A program whose plan could not be installed is erased by the cache, so it is not checked, and
	// one that is made again is.
	auto failed           = Facts(PROGRAM_TWO, PLAN_A, true);
	failed.plan_installed = false;
	CHECK(planner.Judge(failed, existing.Source()) == DrainVerdict::DropFailedInstall);
	CHECK(planner.CheckReverse(PROGRAM_TWO, runtime(PLAN_B)) == ReverseVerdict::NotApplicable);
	(void)planner.Judge(Facts(PROGRAM_TWO, PLAN_B, true), existing.Source());
	CHECK(planner.CheckReverse(PROGRAM_TWO, runtime(PLAN_B)) == ReverseVerdict::Match);
}

// The report's figures: how many drains there were and how many were slow, the longest, and how
// long the oldest module waited. The slowness rows are on both sides of the 1 and 4 ms lines.
void TestDrainCounters() {
	GeneratedDrainPlanner planner;
	planner.NoteDrain(500us, 3ms);
	planner.NoteDrain(2ms, 10ms);
	planner.NoteDrain(5ms, 1ms);
	const auto& counters = planner.Counters();
	CHECK(counters.drain_calls == 3);
	CHECK(counters.drain_over_1ms == 2 && counters.drain_over_4ms == 1);
	CHECK(counters.drain_max_ms > 4.9 && counters.drain_max_ms < 5.1);
	CHECK(counters.drain_total_ms > 7.4 && counters.drain_total_ms < 7.6);
	CHECK(counters.drain_lag_ms_max > 9.9 && counters.drain_lag_ms_max < 10.1);
}

// The longest module is kept, whatever order they come in, and it is separate from the drain's own
// time, which a drain of many short modules can exceed.
void TestItemMax() {
	GeneratedDrainPlanner planner;
	planner.NoteItem(300us);
	planner.NoteItem(2ms);
	planner.NoteItem(700us);
	CHECK(planner.Counters().item_max_ms > 1.9 && planner.Counters().item_max_ms < 2.1);
	CHECK(planner.Counters().drain_max_ms == 0.0 && planner.Counters().drain_calls == 0);
}

} // namespace

int main() {
	TestBudget();
	TestNewProgram();
	TestPlanTripwire();
	TestSkipDispatchProgramAlwaysMismatches();
	TestDuplicate();
	TestReverseCheck();
	TestDrainCounters();
	TestItemMax();
	if (g_failures == 0) {
		std::puts("Generated drain planner tests passed");
	}
	return g_failures == 0 ? 0 : 1;
}
