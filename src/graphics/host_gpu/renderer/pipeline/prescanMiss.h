#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANMISS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANMISS_H_

#include "common/gpuStall.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/host_gpu/renderer/pipeline/prescanHandoff.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>

// Why the GPU thread compiled a compute pipeline itself while the submit-time prescan was on. Each
// such compile gets exactly one reason, picked from lock-free reads at the lookup's miss and when
// the compile's Scope opens, and the reason names the Scope's hold cause.
namespace Libs::Graphics {

/// Why the prescan did not deliver a compute module that the GPU thread then compiled itself. The
/// values are in precedence order, so the first one that applies is the reason. The three
/// NeverPosted reasons with a suffix are attributes of the dispatch the GPU thread ran, which the
/// scan may not reach or the preparer may not see, and not proof that this is why it was missed.
enum class PrescanMiss : uint8_t {
	/// No miss was noted for the module, because the lookup that led to the compile found it made.
	/// That is a module an earlier lookup compiled without its pipeline, or one made on another
	/// thread.
	ModulePresent,
	/// The preparer had a request out for the unit at the miss or when the Scope opened.
	InFlight,
	/// The prescan was off for the rest of the run at the miss, after a stop, a lane that turned
	/// off or a failure. A verdict it left is history then, so it does not count as a refusal.
	Stopped,
	/// The live preparer had a verdict for the unit at the miss and nothing outstanding for it.
	Refused,
	/// The dispatch ran in more nested indirect buffers than the scan follows.
	NeverPostedDeep,
	/// The dispatch read its group counts from memory. One with thread dimensions is run as a
	/// direct dispatch, so it never reads as indirect.
	NeverPostedIndirect,
	/// The dispatch ran on an async compute queue.
	NeverPostedAsync,
	/// None of the above. The walk's level limit, its dword budget and its time cap can each leave
	/// a shallow dispatch unscanned, and it then reads as this. So does a dispatch the preparer
	/// dropped before it had a unit, such as one whose program was not in the shader map.
	NeverPosted,
};
/// How many reasons there are.
inline constexpr size_t PRESCAN_MISS_COUNT = static_cast<size_t>(PrescanMiss::NeverPosted) + 1;

/// The reasons' names in the report, in the enum's order.
inline constexpr std::array<const char*, PRESCAN_MISS_COUNT> PRESCAN_MISS_NAMES {
    "module_present", "in_flight", "stopped", "refused", "never_posted_deep",
    "never_posted_indirect", "never_posted_async", "never_posted"};

/// How many verdicts a refused miss is split by. Every verdict but None, which never refuses.
inline constexpr size_t PRESCAN_REFUSED_VERDICT_COUNT =
    static_cast<size_t>(PRESCAN_UNIT_VERDICT_LAST);

/// The verdicts' names in the report, in the enum's order from Admitted.
inline constexpr std::array<const char*, PRESCAN_REFUSED_VERDICT_COUNT>
    PRESCAN_REFUSED_VERDICT_NAMES {"admitted", "deferred", "built", "rejected",
                                   "indirect_args", "retired", "compile_failed", "capped",
                                   "settled", "backed_off", "code_unreadable", "code_changed"};

/// Whether a name table's entry is missing. The tables are sized by the counts, so a name left out
/// would be a null entry and not a compile error, which the asserts below turn it into.
[[nodiscard]] constexpr bool IsNullName(const char* name) noexcept {
	return name == nullptr;
}
static_assert(std::ranges::none_of(PRESCAN_MISS_NAMES, IsNullName), "every reason needs a name");
static_assert(std::ranges::none_of(PRESCAN_REFUSED_VERDICT_NAMES, IsNullName),
              "every refusing verdict needs a name");

// The refused counts start at Admitted, so None must be the value below every verdict.
static_assert(static_cast<size_t>(PrescanUnitVerdict::None) == 0 &&
                  static_cast<size_t>(PrescanUnitVerdict::Admitted) == 1,
              "the refused counts' slots assume None is 0 and Admitted is 1");

/// The slot of a verdict in the refused counts. None has no slot, so never pass it.
[[nodiscard]] constexpr size_t RefusedVerdictIndex(PrescanUnitVerdict verdict) noexcept {
	return static_cast<size_t>(verdict) - 1;
}

/// The deepest buffer stack in which the scan still finds a dispatch. The walk counts the top-level
/// list as depth 0 and follows a call up to its depth limit, while the command processor's stack
/// holds the top-level list as one entry, so the stack is one deeper than the walk.
inline constexpr uint32_t PRESCAN_SCAN_REACH_STACK_DEPTH = ScanLimits {}.max_depth + 1;

/// Whether a dispatch run with this many buffers on the command processor's stack is past the
/// scan's reach. 0 means there was no stack to read, which is never deep.
[[nodiscard]] constexpr bool IsDeepDispatch(uint32_t stack_depth) noexcept {
	return stack_depth > PRESCAN_SCAN_REACH_STACK_DEPTH;
}

/// The reason of a noted miss whose unit was not in flight, from what the GPU thread read at the
/// miss. A stopped prescan comes first, since any verdict it left is history. A verdict comes
/// next, since the preparer saw the unit whatever the dispatch looked like. Then the dispatch's
/// attributes, deepest first.
[[nodiscard]] constexpr PrescanMiss ClassifyUnposted(PrescanUnitVerdict verdict, bool stopped,
                                                     bool deep, bool indirect,
                                                     bool async) noexcept {
	if (stopped) {
		return PrescanMiss::Stopped;
	}
	if (verdict != PrescanUnitVerdict::None) {
		return PrescanMiss::Refused;
	}
	if (deep) {
		return PrescanMiss::NeverPostedDeep;
	}
	if (indirect) {
		return PrescanMiss::NeverPostedIndirect;
	}
	return async ? PrescanMiss::NeverPostedAsync : PrescanMiss::NeverPosted;
}

/// The reason of a compile, given whether its miss was noted, whether the unit was in flight at the
/// miss or the Scope, and what ClassifyUnposted said at the miss.
[[nodiscard]] constexpr PrescanMiss ClassifyMiss(bool noted, bool in_flight,
                                                 PrescanMiss unposted) noexcept {
	if (!noted) {
		return PrescanMiss::ModulePresent;
	}
	return in_flight ? PrescanMiss::InFlight : unposted;
}

/// The hold cause a reason charges. ModulePresent, Stopped and plain NeverPosted keep
/// RequiredCompute, so their hold time cannot be told apart, only their counts.
[[nodiscard]] constexpr Common::GpuStall::HoldCause HoldCauseOf(PrescanMiss miss) noexcept {
	using Common::GpuStall::HoldCause;
	// No default, so a reason added without a cause is a warning.
	switch (miss) {
		case PrescanMiss::ModulePresent: return HoldCause::RequiredCompute;
		case PrescanMiss::InFlight: return HoldCause::PrescanInFlight;
		case PrescanMiss::Stopped: return HoldCause::RequiredCompute;
		case PrescanMiss::Refused: return HoldCause::PrescanRefused;
		case PrescanMiss::NeverPostedDeep: return HoldCause::PrescanNeverPostedDeep;
		case PrescanMiss::NeverPostedIndirect: return HoldCause::PrescanNeverPostedIndirect;
		case PrescanMiss::NeverPostedAsync: return HoldCause::PrescanNeverPostedAsync;
		case PrescanMiss::NeverPosted: return HoldCause::RequiredCompute;
	}
	// Only a value outside the enum gets here, and Other is the cause a finished run must not have.
	return HoldCause::Other;
}

/// The GPU thread's own compute pipeline compiles with the prescan on, by reason, as read at one
/// moment. refused_by splits the refused ones by the unit's verdict. built_undrained counts the
/// refused ones whose verdict was Built while the generated modules' inbox was not empty at the
/// miss, which is an upper bound on the misses a drain at the miss could have saved, since the
/// generator posts into the same inbox.
struct PrescanMissCounts {
	std::array<uint64_t, PRESCAN_MISS_COUNT>            by_reason {};
	std::array<uint64_t, PRESCAN_REFUSED_VERDICT_COUNT> refused_by {};
	uint64_t                                            built_undrained = 0;

	/// Every inline compile counted, one per pipeline Scope, which is the sum of the reasons.
	[[nodiscard]] uint64_t InlineCompiles() const noexcept {
		return std::accumulate(by_reason.begin(), by_reason.end(), uint64_t {0});
	}
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANMISS_H_
