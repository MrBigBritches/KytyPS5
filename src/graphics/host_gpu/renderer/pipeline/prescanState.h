#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANSTATE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANSTATE_H_

#include "graphics/host_gpu/renderer/pipeline/prescanHandoff.h"
#include "graphics/host_gpu/renderer/pipeline/prescanVariants.h"

#include <algorithm>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// The preparer's side of the prescan. It decides which scanned dispatches are prepared, using a
// variant tracker per unit, and publishes what it decided for the scanner to read. Only the
// preparer thread touches it, so it has no lock. The scanner reaches it only through the
// PrescanHandoff.
namespace Libs::Graphics {

/// What the preparer's state counted. Posted, busy, capped and settled split the items it was asked
/// about by the gate's answer.
struct PrescanCounters {
	/// Items whose snapshot the preparer was told to prepare.
	uint64_t posted = 0;
	/// Items of a unit that already had a snapshot in flight.
	uint64_t busy = 0;
	/// Items of a unit at its variant cap, or retired.
	uint64_t capped = 0;
	/// Items of a settled unit that were not its probe.
	uint64_t settled = 0;
	/// Items whose program the walk could not vouch for. They are counted under a gate answer too.
	uint64_t uncertain = 0;
	/// Dispatches the scanner dropped because the ring was full. They never reached the preparer.
	uint64_t dropped_full = 0;
};

/// Decides, for each item the preparer takes from the ring, whether its snapshot is prepared.
/// `Spec` is the specialization a snapshot gives, as in VariantTracker.
template <class Spec>
class PrescanState {
public:
	using Tracker = VariantTracker<Spec>;
	using Gate    = typename Tracker::Gate;
	using Outcome = typename Tracker::Outcome;

	/// The gate's answer for one item and the unit it was counted under. When the gate is Post the
	/// preparer passes it back with exactly one of NoteSnapshot, NoteRejected, NoteDropped or
	/// Retire, which uses it up and takes back the busy answers given while it was open.
	struct Admission {
		uint64_t unit = 0;
		Gate     gate = Gate::Settled;
		/// Which post of the unit this is, so that an admission from an earlier post, or one
		/// already used, is told apart from the open one. 0 when the gate was not Post.
		uint64_t generation = 0;
	};

	/// Makes the state over the hand-off it answers through. The hand-off must outlive it.
	explicit PrescanState(PrescanHandoff& handoff): m_handoff(handoff) {}

	/// Asks the gate about one item and publishes its answer for the item's key. The unit is the
	/// program's code hash, its user data count and `static_state`, which is the static key the
	/// runtime would build. The caller must first look the program up in the shader map, fill in
	/// the item's code hash, and skip an item whose program is not in the map. Passing a code hash
	/// of 0 is a programming error.
	/// @throws std::invalid_argument when the item's code hash is 0.
	[[nodiscard]] Admission Admit(const PrescanItem& item, std::span<const uint32_t> static_state) {
		if (item.code_hash == 0) {
			throw std::invalid_argument("PrescanState::Admit needs the program's code hash");
		}
		const uint64_t unit = PrescanUnitKey(item.code_hash, item.user_data_count, static_state);
		const Gate     gate = m_tracker.TryPost(unit, item.force_probe);
		m_counters.uncertain += item.uncertain_program ? 1 : 0;
		Count(gate);
		m_handoff.feedback.Publish(item.key_hash, AnswerFor(gate));
		uint64_t generation = 0;
		if (gate == Gate::Post) {
			generation         = ++m_last_generation;
			m_open_posts[unit] = generation;
		} else if (gate == Gate::Busy) {
			NoteBusyKey(unit, item.key_hash);
		}
		return {unit, gate, generation};
	}

	/// Records the specialization the posted snapshot gave. New means the preparer builds a module
	/// for it.
	/// @throws std::logic_error when the admission was not posted or was already used.
	[[nodiscard]] Outcome NoteSnapshot(const Admission& admission, const Spec& spec) {
		RequirePosted(admission);
		return m_tracker.NoteSnapshot(admission.unit, spec);
	}

	/// Records that the posted snapshot gave a specialization that cannot be built, which counts
	/// towards settling.
	/// @throws std::logic_error when the admission was not posted or was already used.
	void NoteRejected(const Admission& admission) {
		RequirePosted(admission);
		m_tracker.NoteRejected(admission.unit);
	}

	/// Records that the posted snapshot ended with no specialization, so the unit may be posted
	/// again.
	/// @throws std::logic_error when the admission was not posted or was already used.
	void NoteDropped(const Admission& admission) {
		RequirePosted(admission);
		m_tracker.NoteDropped(admission.unit);
	}

	/// Stops the unit of the posted snapshot from being posted again, as for a program that can
	/// never be prepared.
	/// @throws std::logic_error when the admission was not posted or was already used.
	void Retire(const Admission& admission) {
		RequirePosted(admission);
		m_tracker.Retire(admission.unit);
	}

	/// Forgets a specialization a snapshot of the unit recorded with NoteSnapshot that was never
	/// built, because its send was refused for now or the helper dropped it, so a later snapshot that
	/// gives it is sent again. It needs no open admission, since the snapshot's outcome was given.
	void Forget(uint64_t unit, const Spec& spec) { m_tracker.Forget(unit, spec); }

	/// A copy of the counts, with the ring's drops read at the time of the call.
	[[nodiscard]] PrescanCounters Counters() const {
		PrescanCounters counters = m_counters;
		counters.dropped_full    = m_handoff.ring.Dropped();
		return counters;
	}

	/// The tracker, for the per-unit figures of the report.
	[[nodiscard]] const Tracker& Variants() const { return m_tracker; }

private:
	// What the scanner should do with later dispatches of a key, given the gate's answer for one of
	// them. A capped unit is as finished as a settled one, and a posted one wants more dispatches.
	static PrescanFeedback::Answer AnswerFor(Gate gate) {
		switch (gate) {
			case Gate::Busy: return PrescanFeedback::Answer::Busy;
			case Gate::Capped:
			case Gate::Settled: return PrescanFeedback::Answer::Settled;
			case Gate::Post: break;
		}
		return PrescanFeedback::Answer::None;
	}

	// Adds one item to the counter of its gate answer.
	void Count(Gate gate) {
		switch (gate) {
			case Gate::Post: m_counters.posted++; break;
			case Gate::Busy: m_counters.busy++; break;
			case Gate::Capped: m_counters.capped++; break;
			case Gate::Settled: m_counters.settled++; break;
		}
	}

	// Uses up the admission of the unit's open post, and refuses one that is not that, which is an
	// item that was never posted, an outcome already given, or an admission from an earlier post of
	// the unit. Any of those would clear another snapshot's in-flight mark or count a snapshot
	// twice.
	void RequirePosted(const Admission& admission) {
		const auto open = m_open_posts.find(admission.unit);
		if (admission.gate != Gate::Post || open == m_open_posts.end() ||
		    open->second != admission.generation) {
			throw std::logic_error("PrescanState outcome for an admission that is not open");
		}
		m_open_posts.erase(open);
		ClearBusyAnswers(admission.unit);
	}

	// Remembers that `key_hash` was answered busy because the unit's post was open, so the answer
	// can be taken back when the post closes.
	void NoteBusyKey(uint64_t unit, uint64_t key_hash) {
		if (key_hash == 0) {
			return;
		}
		auto& keys = m_busy_keys[unit];
		if (std::find(keys.begin(), keys.end(), key_hash) == keys.end()) {
			keys.push_back(key_hash);
		}
	}

	// Takes back the busy answers given while the unit's post was open. The scanner never caches a
	// dispatch whose program it could not vouch for, so such a key is never probed and never asks
	// again, and a busy answer left behind would skip it for good. Clearing only touches a slot the
	// key still owns.
	void ClearBusyAnswers(uint64_t unit) {
		const auto found = m_busy_keys.find(unit);
		if (found == m_busy_keys.end()) {
			return;
		}
		for (const uint64_t key_hash: found->second) {
			m_handoff.feedback.Publish(key_hash, PrescanFeedback::Answer::None);
		}
		m_busy_keys.erase(found);
	}

	PrescanHandoff& m_handoff;
	Tracker         m_tracker;
	PrescanCounters m_counters;
	// The generation of each unit's post that still waits for its outcome. Only units with a
	// snapshot in flight are here, so it stays as small as the work in hand.
	std::unordered_map<uint64_t, uint64_t> m_open_posts;
	// The keys answered busy for each unit with an open post. A unit is only answered busy while
	// its post is open, so this holds no more units than m_open_posts.
	std::unordered_map<uint64_t, std::vector<uint64_t>> m_busy_keys;
	// The last generation handed out. Generations start at 1, so 0 never matches an open post.
	uint64_t m_last_generation = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANSTATE_H_
