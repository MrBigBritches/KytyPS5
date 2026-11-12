#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SUBMIT_SCAN_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SUBMIT_SCAN_H

#include "graphics/guest_gpu/command_processor/scanScratch.h"
#include "graphics/guest_gpu/hardwareContext.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

// A read-only walk over a PM4 command list, to learn which compute programs it is about to dispatch.
// It never calls into the command processor and never dereferences guest memory other than the
// top-level list it is given. Every indirect buffer is read through a reader that cannot fault.
namespace Libs::Graphics {

/// Reads guest memory without ever faulting or waiting. Returns false when any byte of the range is
/// unreadable. Production passes Memory::TryReadBacking.
using ScanMemoryReader = std::function<bool(uint64_t address, void* destination, uint64_t bytes)>;

/// How many indirect buffers may be nested. Four missed most dispatches of a game whose engine nests
/// its command lists deeper, so it is 16, and the visited set and the budget still bound a walk.
inline constexpr uint32_t SCAN_MAX_IB_DEPTH = 16;
/// How many dwords one submission may read in all, which is also the size of the scratch arena a
/// scanning thread keeps, so it is 4 MB per scanning thread. It was 131072 when the scan ran on the
/// submitting thread, and a game's top level lists ran into it, which left their tails unwalked and the
/// program in doubt. The time cap still bounds how long a walk may take. Under --scan-differential the
/// scan still runs on the submitting thread, so the raise lets it run there up to the 1 ms time cap.
inline constexpr uint32_t SCAN_DWORD_BUDGET = 1u << 20;

/// A small cache of the dispatches the caller has settled, so that the walk skips them without copying
/// the registers or taking the posting lock. It is direct mapped on the dispatch's key hash and is
/// used under the scanner's per-queue mutex, so it needs no lock of its own. A remembered key is let
/// through once in `probe_interval` hits, as a probe, so a program that later specializes differently
/// is still seen, and the caller is told so that it may bypass its own throttle for that dispatch.
class DispatchCache {
public:
	enum class Verdict : uint8_t { Pass, Skip, Probe };

	/// Makes an empty cache that lets a remembered key through as a probe once in `probe_interval` hits.
	explicit DispatchCache(uint32_t probe_interval) : m_probe_interval(probe_interval) {}

	/// Says what to do with a dispatch. Skip means it is remembered and not due. Probe means it was
	/// remembered and is due, and is forgotten until the caller remembers it again.
	[[nodiscard]] Verdict Check(uint64_t key_hash) {
		auto& slot = m_slots[key_hash % SLOTS];
		if (key_hash == 0 || slot.key != key_hash) {
			return Verdict::Pass;
		}
		if (++slot.hits < m_probe_interval) {
			return Verdict::Skip;
		}
		slot = {};
		return Verdict::Probe;
	}

	/// What posting a dispatch answered, as far as the cache cares.
	enum class PostAnswer : uint8_t { Posted, Busy, Settled, Contended, Other };

	/// Remembers the dispatch when its post says a repeat adds nothing, because the program is settled or
	/// capped, or a snapshot of it is already in flight. Posted, contended and the rest are left alone, since a
	/// later dispatch has something to do. A dispatch whose program was uncertain is never remembered,
	/// because its key may name a program the walk only guessed at. Returns whether it was remembered.
	bool Note(uint64_t key_hash, PostAnswer answer, bool uncertain_program) {
		if (uncertain_program || (answer != PostAnswer::Settled && answer != PostAnswer::Busy)) {
			return false;
		}
		Remember(key_hash);
		return key_hash != 0;
	}

	/// Remembers a dispatch that the caller found settled, so that the next ones are skipped.
	void Remember(uint64_t key_hash) {
		if (key_hash != 0) {
			m_slots[key_hash % SLOTS] = {key_hash, 0};
		}
	}

private:
	static constexpr size_t SLOTS = 1024;

	struct Slot {
		uint64_t key  = 0;
		uint32_t hits = 0;
	};

	std::array<Slot, SLOTS> m_slots {};
	uint32_t                m_probe_interval;
};

/// The longest one submission's walk may run before it stops, which keeps a huge or slow list from
/// holding the scanning thread. It is checked every SCAN_CLOCK_CHECK_PACKETS packets and before each
/// read, so a single read that blocks is still outside it.
inline constexpr std::chrono::microseconds SCAN_TIME_CAP {1000};
/// How many packets pass between looks at the clock.
inline constexpr uint32_t SCAN_CLOCK_CHECK_PACKETS = 128;

/// How many dispatches one scan's storage holds without growing. A scanning thread reserves it once and
/// reuses it, so a scan allocates nothing from the heap. A larger scan grows it, and that is counted.
inline constexpr size_t SCAN_DISPATCH_CAPACITY = 1024;

/// The bounds of one walk.
struct ScanLimits {
	/// How many indirect buffers may be nested.
	uint32_t max_depth = SCAN_MAX_IB_DEPTH;
	/// How many dwords one submission may read in all, indirect buffers included. A budget stops a
	/// huge or looping list from costing the scanning thread more than a small, fixed amount.
	uint32_t dword_budget = SCAN_DWORD_BUDGET;
	/// The time cap of one walk.
	std::chrono::microseconds time_cap = SCAN_TIME_CAP;
};

/// Why a walk stopped vouching for the program, for the counts in ScanStats.
enum class UncertainCause : uint8_t {
	CondExecRegion,
	Branch,
	Predicated,
	TimeCap,
	BudgetCut,
	UnreadableBuffer,
	DepthLimited,
	Revisited,
	BadPacket,
	UnreadableRegisterList,
	Count,
};

/// Hashes what decides a dispatch's static key, which is the program address and every register that
/// decides its module, and how the dispatch is launched. The user data is left out. The scanner and the
/// command processor both call this, so a difference between their keys is a difference in their register
/// state and never in how they hash it.
[[nodiscard]] uint64_t ComputeDispatchKeyHash(const HW::CsStageRegisters& cs, bool thread_dimensions,
                                              bool async_queue);

/// The compute shader registers as the walk has seen them. It is the per-queue, persistent twin of the
/// command processor's compute state, so it is reset where the processor resets (a flip suspend point
/// and the dispatch reset packet) and nowhere else.
struct CsShadow {
	HW::ComputeShaderInfo regs;
	/// Where the compute indirect arguments are, as IT_SET_BASE last set it, or 0 if never. The
	/// command processor adds the offset of a DISPATCH_INDIRECT in its base form to it.
	uint64_t dispatch_args_base = 0;
	/// Set when the current submission wrote the program address, to tell a dispatch that inherits its
	/// shader from an earlier submission from one that sets it itself.
	bool program_set_in_submission = false;
	/// Whether the program in `regs` is known to be the one the command processor holds. It clears when
	/// the walk passes something that may have skipped or replaced a write to compute state (a COND_EXEC
	/// region or a predicated packet that wrote some, a branch, an unread or cut-off buffer) and sets
	/// again once both halves of the program address have been written. A dispatch is still reported
	/// without it, and carries the doubt.
	bool certain = true;
	/// Set by anything that changed or may have changed the compute registers that decide a module, which
	/// is a write to any compute register but user data, a reset, or any doubt. A COND_EXEC region or a
	/// predicated packet clears certain only if it set this, so one that touched nothing is harmless.
	bool key_state_dirty = false;
	/// Whether each half of the program address was written since the last doubt.
	bool pgm_lo_written = false;
	bool pgm_hi_written = false;

	/// Forgets every register, as the command processor's reset does.
	void Reset() { *this = CsShadow {}; }

	/// Records that the walk can no longer vouch for the program, as when a queued list was dropped
	/// before it was walked.
	void MarkUncertain() {
		certain         = false;
		key_state_dirty = true;
		pgm_lo_written  = false;
		pgm_hi_written  = false;
	}

	/// Notes a write of one half of the program address. The program is certain again once the walk has
	/// seen both halves since the last doubt, since one half alone leaves the other stale.
	void NoteProgramWrite(bool high_half) {
		(high_half ? pgm_hi_written : pgm_lo_written) = true;
		if (pgm_lo_written && pgm_hi_written) {
			certain = true;
		}
	}
};

/// One dispatch the walk found, with everything the runtime would use to build its module.
struct ScannedDispatch {
	/// The guest address of the program, which is the address its shader map entry is keyed by.
	uint64_t program_address = 0;
	/// The compute registers and user data at the dispatch. The wave size is the dispatch
	/// initiator's, set exactly as the command processor sets it before it dispatches.
	HW::ComputeShaderInfo regs;
	/// The dispatch initiator word.
	uint32_t initiator = 0;
	/// Whether the dispatch counts threads rather than groups.
	bool thread_dimensions = false;
	/// Whether the dispatch came from an indirect packet.
	bool indirect = false;
	/// Whether the queue is an async compute queue.
	bool async_queue = false;
	/// Whether the walk could not vouch for the program, see CsShadow::certain.
	bool uncertain_program = false;
	/// The queue the submission was made on.
	uint32_t queue = 0;
	/// The group counts of a direct dispatch, as the runtime derives them. Zero for an indirect one.
	std::array<uint32_t, 3> workgroup_counts {};
	/// The address of the arguments of an indirect dispatch, which is the packet's own address or the
	/// tracked base plus the packet's offset. 0 when the base form met no IT_SET_BASE.
	uint64_t indirect_args_address = 0;
	/// A hash of the program address and every register that decides the module's static key, which
	/// includes how many user data registers the program reads. The user data itself is left out on
	/// purpose, since it differs on nearly every dispatch.
	uint64_t key_hash = 0;
	/// The dispatch was let through by the settled-dispatch cache as a probe, so the poster should not
	/// throttle it again.
	bool force_probe = false;
};

/// What one walk counted. Every field is added to, never reset, so a caller sums many walks.
struct ScanStats {
	uint64_t submissions = 0;
	uint64_t dispatches  = 0;
	/// Dispatches with no program address in the shadow, as after a reset.
	uint64_t no_shader_state = 0;
	/// Dispatches that found their program set by an earlier submission.
	uint64_t program_inherited = 0;
	/// Dispatches reported while the shadow's program was uncertain, see CsShadow::certain.
	uint64_t uncertain_program = 0;
	uint64_t ib_followed       = 0;
	uint64_t ib_unreadable     = 0;
	uint64_t ib_depth_limited  = 0;
	uint64_t ib_revisited      = 0;
	uint64_t budget_exhausted  = 0;
	uint64_t bad_packets       = 0;
	uint64_t resets            = 0;
	uint64_t dwords_read       = 0;
	/// The most dwords one submission read, to show how near the budget gets.
	uint64_t dwords_max = 0;
	/// Dispatches the settled-dispatch cache skipped before they were copied.
	uint64_t dispatches_cached = 0;
	/// Walks stopped by the time cap.
	uint64_t time_capped = 0;
	/// How many times storage a walk reuses had to grow, which is a heap allocation on the scanning
	/// thread. That is the dispatch vector, the visited table and the buffer scratch. It is 0 once
	/// they have been through one walk of the same size.
	uint64_t scan_allocations = 0;
	/// Dispatches whose program had a build in flight, remembered so that repeats are skipped.
	uint64_t cache_remembered_busy = 0;
	/// The deepest indirect buffer nesting a walk reached.
	uint64_t depth_max = 0;
	/// Indirect register lists that could not be read, counted apart from indirect buffers.
	uint64_t register_list_unreadable = 0;
	/// Branch packets by mode. Mode 2 counts every mode 2 branch, and the empty else count is the part of
	/// them whose else buffer has no dwords, which the processor treats like mode 1 when not taken.
	uint64_t branch_mode1           = 0;
	uint64_t branch_mode2           = 0;
	uint64_t branch_mode2_empty_else = 0;
	/// Revisited buffers that left the program in doubt, the part of ib_revisited whose first walk wrote
	/// compute key state or was still going.
	uint64_t ib_revisit_doubts = 0;
	/// How many times each cause was hit, indexed by UncertainCause. A hit counts even when the program
	/// was already uncertain, so the sum is not the number of times `certain` went from true to false.
	std::array<uint64_t, static_cast<size_t>(UncertainCause::Count)> uncertain_cleared_by {};

	/// Adds another walk's counts to these.
	void Add(const ScanStats& other) {
		submissions += other.submissions;
		dispatches += other.dispatches;
		no_shader_state += other.no_shader_state;
		program_inherited += other.program_inherited;
		uncertain_program += other.uncertain_program;
		ib_followed += other.ib_followed;
		ib_unreadable += other.ib_unreadable;
		ib_depth_limited += other.ib_depth_limited;
		ib_revisited += other.ib_revisited;
		budget_exhausted += other.budget_exhausted;
		bad_packets += other.bad_packets;
		resets += other.resets;
		dwords_read += other.dwords_read;
		dwords_max = std::max(dwords_max, other.dwords_max);
		dispatches_cached += other.dispatches_cached;
		time_capped += other.time_capped;
		scan_allocations += other.scan_allocations;
		cache_remembered_busy += other.cache_remembered_busy;
		depth_max = std::max(depth_max, other.depth_max);
		register_list_unreadable += other.register_list_unreadable;
		ib_revisit_doubts += other.ib_revisit_doubts;
		branch_mode1 += other.branch_mode1;
		branch_mode2 += other.branch_mode2;
		branch_mode2_empty_else += other.branch_mode2_empty_else;
		for (size_t cause = 0; cause < uncertain_cleared_by.size(); ++cause) {
			uncertain_cleared_by[cause] += other.uncertain_cleared_by[cause];
		}
	}
};

/// Walks a command list and reports the compute dispatches in it.
class SubmitScanner {
public:
	explicit SubmitScanner(ScanMemoryReader reader, ScanLimits limits = {})
	    : m_reader(std::move(reader)), m_limits(limits) {}

	/// Walks one command list, applying its compute register writes to the shadow and appending each
	/// dispatch that has a program. Both arms of a COND_EXEC are covered, since the walk is linear and
	/// does not know the condition. A branch is walked as the processor may run it. A mode 2 branch
	/// with an else buffer walks the else arm and then the then arm, and chains, so the parent's tail
	/// is not walked. A mode 1 branch, or a mode 2 one with an empty else, runs the then arm or nothing,
	/// so the walk takes the then arm, drops its register writes and carries on with the parent's tail
	/// under an uncertain program. The else buffer of a mode 1 branch is never read. An indirect buffer
	/// that cannot be read is skipped and counted.
	/// The walk keeps its tables in thread local storage, so Scan is not reentrant on one thread. A
	/// reader must not call it.
	/// @param commands    the top-level list, which the caller guarantees is readable.
	/// @param queue       the id of the queue the list was submitted on.
	/// @param async_queue whether that queue is an async compute queue.
	void Scan(std::span<const uint32_t> commands, uint32_t queue, bool async_queue, CsShadow& shadow,
	          std::vector<ScannedDispatch>& dispatches, ScanStats& stats,
	          DispatchCache* cache = nullptr) const;

	/// The same walk using the caller's storage in place of the calling thread's own. The caller must not
	/// use one ScanScratch from two walks at once.
	void Scan(std::span<const uint32_t> commands, uint32_t queue, bool async_queue, CsShadow& shadow,
	          std::vector<ScannedDispatch>& dispatches, ScanStats& stats, ScanScratch& scratch,
	          DispatchCache* cache = nullptr) const;

private:
	ScanMemoryReader m_reader;
	ScanLimits       m_limits;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SUBMIT_SCAN_H
