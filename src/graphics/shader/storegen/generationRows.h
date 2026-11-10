#ifndef GRAPHICS_SHADER_STOREGEN_GENERATION_ROWS_H
#define GRAPHICS_SHADER_STOREGEN_GENERATION_ROWS_H

#include "common/firstRunReport.h"

#include <atomic>
#include <cstdint>
#include <optional>

// The rows that the pipeline cache keeps about generated modules, for the first-run report. They
// are plain data and need no device, so the report's classifier and its tests can use them. Times
// are in nanoseconds on the steady clock, which is the clock the first-run report's NowNs reads.
namespace Libs::Graphics::StoreGen {

/// A time that was not taken.
inline constexpr int64_t NO_ROW_TIME_NS = Common::FirstRunReport::NO_TIME_NS;

/// One module that the drain took in, kept for as long as the cache lives. The warm build of the
/// module stamps ready_ns and build_ms from a worker's thread, so those two are atomic, and the row
/// is never moved, which is why the cache keeps rows in a deque.
struct GenRow {
	GenRow(uint64_t module_id, uint64_t program_hash, int64_t posted, int64_t adopted)
	    : id(module_id), code_hash(program_hash), posted_ns(posted), adopted_ns(adopted) {}

	uint64_t id        = 0;
	uint64_t code_hash = 0;
	/// When the generator's thread posted it, and when the GPU thread took it in.
	int64_t posted_ns  = NO_ROW_TIME_NS;
	int64_t adopted_ns = NO_ROW_TIME_NS;
	/// When a worker finished its pipeline, or NO_ROW_TIME_NS while none has.
	std::atomic<int64_t> ready_ns {NO_ROW_TIME_NS};
	/// How long that build took, or a negative number while none has finished.
	std::atomic<double> build_ms {-1.0};
	/// Where the module stood among its program's variants when the supervisor delivered it. They
	/// are written once at adoption, before any worker can see the row, so they need no atomics.
	uint32_t                tier      = 0;
	bool                    wide_tail = false;
	std::optional<uint64_t> candidates;
};

/// A copy of a GenRow that can be handed to another thread.
struct GenRowSnapshot {
	uint64_t                id         = 0;
	uint64_t                code_hash  = 0;
	int64_t                 posted_ns  = NO_ROW_TIME_NS;
	int64_t                 adopted_ns = NO_ROW_TIME_NS;
	std::optional<int64_t>  ready_ns;
	std::optional<double>   build_ms;
	uint32_t                tier      = 0;
	bool                    wide_tail = false;
	std::optional<uint64_t> candidates;
};

/// Copies a row. The two atomics are read separately, so a build that finishes meanwhile may show
/// one without the other, and a reader treats each on its own.
[[nodiscard]] inline GenRowSnapshot SnapshotOf(const GenRow& row) {
	GenRowSnapshot copy {.id         = row.id,
	                     .code_hash  = row.code_hash,
	                     .posted_ns  = row.posted_ns,
	                     .adopted_ns = row.adopted_ns,
	                     .tier       = row.tier,
	                     .wide_tail  = row.wide_tail,
	                     .candidates = row.candidates};
	if (const auto ready = row.ready_ns.load(std::memory_order_acquire); ready != NO_ROW_TIME_NS) {
		copy.ready_ns = ready;
	}
	if (const auto built = row.build_ms.load(std::memory_order_acquire); built >= 0.0) {
		copy.build_ms = built;
	}
	return copy;
}

/// Why the drain refused a module that the generator had posted.
enum class DropVerdict : uint8_t {
	PlanMismatch,
	DroppedStopped,
	FailedInstall,
	Duplicate,
};

/// The name of a drop verdict in the report.
[[nodiscard]] inline const char* DropVerdictName(DropVerdict verdict) {
	switch (verdict) {
		case DropVerdict::PlanMismatch: return "plan_mismatch";
		case DropVerdict::DroppedStopped: return "dropped_stopped";
		case DropVerdict::FailedInstall: return "failed_install";
		case DropVerdict::Duplicate: return "duplicate";
	}
	return "";
}

/// A module the drain refused. It was never adopted, so it has no GenRow. at_ns is when the drain
/// refused it, on the clock of MissRow::at_ns, so the classifier can tell a refusal that came after
/// the miss (the runtime built the module first) from one that came before it (a real loss).
struct DropRow {
	uint64_t    id      = 0;
	DropVerdict verdict = DropVerdict::Duplicate;
	int64_t     at_ns   = NO_ROW_TIME_NS;
};

/// One compute translation that the runtime had to do after generation was on, which means the
/// store held no module that fit. id is 0 for a program that is skipped and so has no module.
struct MissRow {
	uint64_t code_hash = 0;
	uint64_t id        = 0;
	int64_t  at_ns     = NO_ROW_TIME_NS;
	/// The runtime's program has GPU-selected calls, which the worker leaves out by design.
	bool has_calls = false;
	/// The runtime skips the program's dispatches, which the worker also leaves out.
	bool skip_dispatch = false;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_GENERATION_ROWS_H
