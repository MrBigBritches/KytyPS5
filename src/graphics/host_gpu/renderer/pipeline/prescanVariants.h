#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANVARIANTS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANVARIANTS_H_

#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include <xxhash.h>

// Decides which dispatch snapshots of a program the prescan prepares. The user data of a dispatch is
// pointers that change every frame, so a snapshot cannot be told from another by its values. What
// tells them apart is the specialization the preparer derives from them. The submitting side is
// gated so that only one snapshot of a program is in flight, and the preparer reports the
// specialization each one gave. A specialization seen before is not built again. Kept apart from the
// cache and generic over the specialization type, so a test can run it without a device.
namespace Libs::Graphics {

/// The most distinct specializations the prescan builds for one program.
inline constexpr uint32_t PRESCAN_VARIANT_CAP = 8;
/// How many snapshots in a row may give nothing new before a program is called settled.
inline constexpr uint32_t PRESCAN_STALE_LIMIT = 16;
/// Once settled, one dispatch in this many is posted again as a probe, since a later scene may bring
/// descriptors that specialize differently.
inline constexpr uint32_t PRESCAN_PROBE_INTERVAL = 256;

/// Names the program a dispatch runs, as the variant tracker counts it. Two copies of one shader at
/// different addresses share a code hash, and registers the static key ignores do not split it.
/// The user-data count is part of it because the runtime's module id has it and the static key does
/// not, so two dispatches of one program with different counts never build the same module and must
/// not share a variant budget.
/// @param code_hash        the hash of the program's code.
/// @param user_data_count  how many user SGPRs the dispatch loads.
/// @param static_state     the static key words from BuildStageStaticKey.
/// @return the unit key that VariantTracker takes.
[[nodiscard]] inline uint64_t PrescanUnitKey(uint64_t code_hash, uint32_t user_data_count,
                                             std::span<const uint32_t> static_state) {
	// The program and its count seed the hash of the static key, so no buffer is built to join them.
	const uint64_t program[] = {code_hash, user_data_count};
	const auto     seed      = XXH3_64bits(program, sizeof(program));
	return XXH3_64bits_withSeed(static_state.data(), static_state.size_bytes(), seed);
}

/// Tracks the specializations each program's snapshots gave, and gates which dispatches are posted.
/// It is not thread safe, so its one owner serialises every call.
template <class Spec> class VariantTracker {
public:
	/// What the submitting side does with a dispatch of a program.
	enum class Gate : uint8_t { Post, Busy, Capped, Settled };
	/// What the preparer found in a snapshot's specialization.
	enum class Outcome : uint8_t { New, Known, Capped };

	/// Decides what to do with one dispatch of `unit`. `probe` says an earlier cache already throttled
	/// this dispatch and let it through as its probe, so a settled program is posted. Post means the
	/// caller must post it, and the program then has a snapshot in flight until NoteSnapshot,
	/// NoteRejected, NoteDropped or Retire.
	[[nodiscard]] Gate TryPost(uint64_t unit, bool probe = false) {
		auto& entry = m_units[unit];
		if (entry.in_flight) {
			return Gate::Busy;
		}
		if (entry.capped) {
			return Gate::Capped;
		}
		if (entry.stale >= PRESCAN_STALE_LIMIT) {
			if (!probe && ++entry.skipped < PRESCAN_PROBE_INTERVAL) {
				return Gate::Settled;
			}
			entry.skipped = 0;
		}
		entry.in_flight = true;
		return Gate::Post;
	}

	/// Records that the preparer derived `spec` from the in-flight snapshot. New means it differs from
	/// every one built so far and is now recorded, so the caller builds a module for it.
	[[nodiscard]] Outcome NoteSnapshot(uint64_t unit, const Spec& spec) {
		auto& entry     = m_units[unit];
		entry.in_flight = false;
		for (const auto& known: entry.specs) {
			if (known == spec) {
				entry.stale++;
				return Outcome::Known;
			}
		}
		// Only reached if a caller notes a snapshot it never posted, since a capped program is not posted.
		if (entry.specs.size() >= PRESCAN_VARIANT_CAP) {
			entry.capped = true;
			return Outcome::Capped;
		}
		entry.specs.push_back(spec);
		entry.stale = 0;
		if (entry.specs.size() >= PRESCAN_VARIANT_CAP) {
			entry.capped = true;
			m_capped++;
		}
		return Outcome::New;
	}

	/// Records that the snapshot gave a specialization that could not be built. It counts as one that
	/// gave nothing new, so a program that keeps doing it settles instead of being posted on every
	/// dispatch.
	void NoteRejected(uint64_t unit) {
		auto& entry     = m_units[unit];
		entry.in_flight = false;
		entry.stale++;
	}

	/// Records that the in-flight snapshot ended without a specialization, so the program may be
	/// posted again.
	void NoteDropped(uint64_t unit) { m_units[unit].in_flight = false; }

	/// Stops a program that can never be prepared, such as one with GPU-selected calls, from being
	/// posted again. It does not count as having reached the cap.
	void Retire(uint64_t unit) {
		auto& entry     = m_units[unit];
		entry.in_flight = false;
		entry.capped    = true;
		entry.retired   = true;
	}

	/// Forgets a specialization that NoteSnapshot recorded but that was never built, because its
	/// send was refused for now or the helper dropped it. The next snapshot that gives it is then New
	/// again and is sent again. A program the cap closed reopens, and a retired one stays retired.
	void Forget(uint64_t unit, const Spec& spec) {
		const auto found = m_units.find(unit);
		if (found == m_units.end()) {
			return;
		}
		auto&      entry   = found->second;
		const auto removed = std::erase(entry.specs, spec);
		if (removed != 0 && entry.capped && !entry.retired && entry.specs.size() < PRESCAN_VARIANT_CAP) {
			entry.capped = false;
			m_capped--;
		}
	}

	/// How many programs reached the cap.
	[[nodiscard]] uint64_t CappedUnits() const { return m_capped; }

	/// Whether `unit` was retired, which a Capped gate answer does not tell apart from the cap.
	[[nodiscard]] bool IsRetired(uint64_t unit) const {
		const auto found = m_units.find(unit);
		return found != m_units.end() && found->second.retired;
	}

	/// How many programs are settled now. It walks every program, so it is meant for the end-of-run
	/// report and not for each dispatch.
	[[nodiscard]] uint64_t SettledUnits() const {
		uint64_t count = 0;
		for (const auto& unit: m_units) {
			count += unit.second.stale >= PRESCAN_STALE_LIMIT ? 1 : 0;
		}
		return count;
	}

	/// How many programs have been seen.
	[[nodiscard]] uint64_t Units() const { return m_units.size(); }

private:
	struct Entry {
		bool              in_flight = false;
		bool              capped    = false;
		bool              retired   = false;
		uint32_t          stale     = 0;
		uint32_t          skipped   = 0;
		std::vector<Spec> specs;
	};

	std::unordered_map<uint64_t, Entry> m_units;
	uint64_t                            m_capped = 0;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANVARIANTS_H_
