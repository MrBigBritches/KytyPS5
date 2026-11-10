#ifndef GRAPHICS_SHADER_STOREGEN_GENERATED_DRAIN_PLANNER_H
#define GRAPHICS_SHADER_STOREGEN_GENERATED_DRAIN_PLANNER_H

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <unordered_map>

// The decisions the GPU thread makes when it takes generated modules into the pipeline cache. They
// are kept apart from the cache, which needs a device, so that tests run them on every machine. The
// cache gives the planner the facts of each module and does what it answers.
namespace Libs::Graphics::StoreGen {

/// How many generated modules one drain takes at most. The GPU thread drains at the top of every
/// shader lookup, so a batch only has to be short, and the rest waits for the next lookup.
constexpr size_t GENERATION_DRAIN_BUDGET = 8;
/// How long one drain may run before it stops, however many modules are left. Audio waits on the
/// GPU thread, so a drain must stay short even when a module is slow to take in.
constexpr std::chrono::nanoseconds GENERATION_DRAIN_TIME_BUDGET = std::chrono::milliseconds(1);

/// The limits of one drain.
struct DrainBudget {
	size_t                   max_items = GENERATION_DRAIN_BUDGET;
	std::chrono::nanoseconds max_time  = GENERATION_DRAIN_TIME_BUDGET;

	/// Whether a drain that took `handled` modules in `elapsed` may take another. The first module
	/// is always taken, so a drain makes progress however short the time budget is.
	[[nodiscard]] bool Allows(size_t handled, std::chrono::nanoseconds elapsed) const {
		return handled == 0 || (handled < max_items && elapsed < max_time);
	}
};

/// What the cache does with a generated module.
enum class DrainVerdict : uint8_t {
	/// Add it to its program and start its warm build.
	Adopt,
	/// The program already holds a module with this id.
	DropDuplicate,
	/// The module's plan is not the program's. It and every other module of the program that has
	/// the same difference are dropped.
	DropPlanMismatch,
	/// The program is new and its plan could not be installed. The cache erases the program.
	DropFailedInstall,
	/// The cache has stopped, so nothing is taken in any more.
	DropStopped,
};

/// What the cache knows of a module that is about to be taken in.
struct DrainFacts {
	/// The cache stopped, or has no warm queue to start the build on.
	bool stopped = false;
	/// Names the program in the cache. The planner only compares it, so any stable number does.
	uint64_t program = 0;
	/// The program was made for this module, which means it did not exist before.
	bool program_inserted = false;
	/// For a new program, whether its plan was installed.
	bool plan_installed = true;
	/// The hash of the plan bytes the module's record held.
	uint64_t plan_hash = 0;
	/// Whether the program already holds a module with the module's id.
	bool id_present = false;
};

/// What the runtime's own translation says about a program that generation made.
enum class ReverseVerdict : uint8_t {
	/// The program was not made by generation, or this check already ran for it.
	NotApplicable,
	Match,
	Mismatch,
};

/// What the drains did, for the report.
struct GeneratedDrainCounters {
	uint64_t drain_calls           = 0;
	uint64_t drain_over_1ms        = 0;
	uint64_t drain_over_4ms        = 0;
	double   drain_max_ms          = 0.0;
	double   drain_total_ms        = 0.0;
	double   drain_lag_ms_max      = 0.0;
	double   item_max_ms           = 0.0;
	uint64_t adopted               = 0;
	uint64_t duplicates            = 0;
	uint64_t plan_mismatch         = 0;
	uint64_t plan_mismatch_reverse = 0;
	uint64_t failed_installs       = 0;
	uint64_t dropped_stopped       = 0;
};

/// Decides what happens to each generated module, and keeps what it learned of each program.
///
/// A program is known to it once a generated module met it. The first time, the plan of the program
/// is compared with the module's, which is the plan tripwire: a program's plan depends only on its
/// code and its user data count, so two plans that differ mean the generator and the runtime
/// disagree, and the generator's module is dropped. That is the check in one direction. The other
/// is CheckReverse, for a program that generation made and that the runtime then translated itself.
class GeneratedDrainPlanner {
public:
	/// Decides one module.
	/// @param facts              what the cache knows.
	/// @param existing_plan_hash gives the hash of the plan of an existing program. It is called at
	///                           most once for a program, since the answer is kept.
	[[nodiscard]] DrainVerdict Judge(const DrainFacts&                facts,
	                                 const std::function<uint64_t()>& existing_plan_hash);

	/// Compares the plan that the runtime translated for a program with the one the program was
	/// made with, once for a program generation made.
	/// @param runtime_plan_hash gives the hash of the runtime's plan. It is called only when the
	/// check
	///                          is due.
	[[nodiscard]] ReverseVerdict CheckReverse(uint64_t                         program,
	                                          const std::function<uint64_t()>& runtime_plan_hash);

	/// Notes the time one module took to be taken in. The time budget cannot cut a module short, so
	/// the longest one shows whether a big plan pushes a drain over the budget.
	void NoteItem(std::chrono::nanoseconds duration);

	/// Notes one drain, the time it took, and how long its oldest module had waited.
	void NoteDrain(std::chrono::nanoseconds duration, std::chrono::nanoseconds oldest_wait);

	[[nodiscard]] const GeneratedDrainCounters& Counters() const { return m_counters; }

private:
	struct Known {
		/// The hash of the plan the program has, as the generated modules must have it.
		uint64_t hash = 0;
		/// Whether the modules of the program are taken in. False after a first mismatch.
		bool trusted = true;
		/// Whether generation made the program, which is when the reverse check applies.
		bool from_generation = false;
		bool reverse_checked = false;
	};

	std::unordered_map<uint64_t, Known> m_known;
	GeneratedDrainCounters              m_counters;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_GENERATED_DRAIN_PLANNER_H
