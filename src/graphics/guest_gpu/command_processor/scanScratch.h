#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_SCRATCH_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_SCRATCH_H

#include "common/assert.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// The storage a scanning thread reuses from walk to walk. Both pieces are sized once and then never
// grow in steady state, so a walk allocates nothing, and each has a small test hook since the walk
// that uses them keeps them out of reach.
namespace Libs::Graphics {

/// The finishing step of splitmix64, which spreads every input bit over every output bit.
[[nodiscard]] inline uint64_t ScanScramble(uint64_t value) {
	value ^= value >> 30u;
	value *= 0xbf58476d1ce4e5b9ull;
	value ^= value >> 27u;
	value *= 0x94d049bb133111ebull;
	value ^= value >> 31u;
	return value;
}

/// The buffers a walk has met, as a flat open-addressing table so that a walk of thousands of buffers
/// allocates nothing once the table has grown. Each slot carries the generation it was written in, and
/// a slot counts as full only in the current generation, so Reset is one increment and the table is
/// never cleared after a walk that grew it large. Keys are already hashes, and the slot is chosen from
/// their scrambled bits.
/// A slot is 16 bytes, an 8 byte key and a 4 byte generation with padding, which was kept over a separate
/// generation array for the locality of one probe. The table holds at most half full, and a walk visits
/// at most one buffer per dword of its budget, so the worst case is 2^21 slots, 32 MiB, plus the old
/// table while it grows. The time cap ends a walk long before that, and a real walk visits thousands.
class VisitedSet {
public:
	/// Adds the key. False when it was already there, and then `first_walk_clean` says whether the walk
	/// that added it has finished and left the compute key state untouched. It is false while that walk
	/// is still going, as when a buffer calls itself.
	bool Insert(uint64_t key, bool& first_walk_clean) {
		key = Normalize(key);
		if (m_slots.empty()) {
			m_slots.assign(INITIAL_SLOTS, Slot {});
			m_growths++;
		}
		if ((m_count + 1) * 2 > m_slots.size()) {
			Grow();
		}
		const auto mask  = m_slots.size() - 1;
		auto       index = ScanScramble(key) & mask;
		while (IsFull(m_slots[index])) {
			if ((m_slots[index].key & ~CLEAN_BIT) == key) {
				first_walk_clean = (m_slots[index].key & CLEAN_BIT) != 0;
				return false;
			}
			index = (index + 1) & mask;
		}
		m_slots[index] = {key, m_generation};
		m_count++;
		return true;
	}

	/// Records that the walk which added the key finished without touching the key state. The bit lives
	/// in the slot itself, so it needs no storage of its own and no allocation.
	void MarkClean(uint64_t key) {
		// A key being marked was inserted, so the table exists.
		EXIT_IF(m_slots.empty());
		key             = Normalize(key);
		const auto mask = m_slots.size() - 1;
		auto       index = ScanScramble(key) & mask;
		while (IsFull(m_slots[index])) {
			if ((m_slots[index].key & ~CLEAN_BIT) == key) {
				m_slots[index].key |= CLEAN_BIT;
				return;
			}
			index = (index + 1) & mask;
		}
	}

	/// Empties the set for the next walk by starting a new generation, which touches no slot. Only when
	/// the generation counter wraps, once in four billion walks, is the table cleared, since a slot
	/// written a whole cycle ago would otherwise look current.
	void Reset() {
		m_count = 0;
		if (++m_generation == 0) {
			std::ranges::fill(m_slots, Slot {});
			m_generation = 1;
		}
	}

	/// How many times the table was allocated or grown since this was last asked, then forgotten.
	uint64_t TakeGrowths() { return std::exchange(m_growths, 0); }

	/// For tests. Whether any slot still holds the key from any generation, which shows that Reset
	/// left the old entries in place.
	[[nodiscard]] bool HoldsStaleEntry(uint64_t key) const {
		key = Normalize(key);
		return std::ranges::any_of(m_slots, [key](const Slot& slot) {
			return slot.generation != 0 && (slot.key & ~CLEAN_BIT) == key;
		});
	}

	/// For tests. Jumps the generation counter, to reach its wrap without four billion walks.
	void SetGenerationForTest(uint32_t generation) { m_generation = generation; }

private:
	static constexpr size_t   INITIAL_SLOTS = 256;
	// The lowest bit of a stored key is the clean mark, so keys are compared without it.
	static constexpr uint64_t CLEAN_BIT = 1;

	struct Slot {
		uint64_t key        = 0;
		// Zero is never a live generation, so a fresh slot is empty in all of them.
		uint32_t generation = 0;
	};

	// Clears the mark bit, so a key and its marked self are the same key.
	static uint64_t Normalize(uint64_t key) { return key & ~CLEAN_BIT; }

	[[nodiscard]] bool IsFull(const Slot& slot) const { return slot.generation == m_generation; }

	// Doubles the table and places every key of the current generation again with its mark, which is
	// counted as an allocation.
	void Grow() {
		std::vector<Slot> old(m_slots.size() * 2);
		old.swap(m_slots);
		m_growths++;
		const auto mask = m_slots.size() - 1;
		for (const auto& stored: old) {
			if (!IsFull(stored)) {
				continue;
			}
			auto index = ScanScramble(stored.key & ~CLEAN_BIT) & mask;
			while (IsFull(m_slots[index])) {
				index = (index + 1) & mask;
			}
			m_slots[index] = stored;
		}
	}

	std::vector<Slot> m_slots;
	size_t            m_count      = 0;
	uint32_t          m_generation = 1;
	uint64_t          m_growths    = 0;
};

/// One block of dwords that a walk hands out in a stack order, as slices that live only while the
/// buffer read into them is being walked. A walk reads at most its dword budget in all, so an arena of
/// the budget's size is an exact bound and holds every slice live at once, and it is allocated once. It
/// is left uninitialized, since each read fills the slice it was given.
class ScratchArena {
public:
	/// Makes sure the arena holds at least `dwords`, which allocates only when it is larger than ever
	/// before. Call it between walks, when no slice is out.
	void Reserve(size_t dwords) {
		// A reallocation would leave live slices pointing at freed memory.
		EXIT_IF(m_used != 0);
		if (dwords > m_capacity) {
			m_data     = std::make_unique_for_overwrite<uint32_t[]>(dwords);
			m_capacity = dwords;
			m_growths++;
		}
	}

	/// Hands out the next `dwords` dwords, or null when they do not fit, which a walk within its budget
	/// never meets.
	[[nodiscard]] uint32_t* Take(size_t dwords) {
		if (dwords > m_capacity - m_used) {
			return nullptr;
		}
		uint32_t* slice = m_data.get() + m_used;
		m_used += dwords;
		return slice;
	}

	/// Where the arena stands, to give back every slice taken since with Release.
	[[nodiscard]] size_t Mark() const { return m_used; }

	/// Gives back the slices taken since `mark`.
	void Release(size_t mark) { m_used = mark; }

	/// How many dwords are out now, for tests.
	[[nodiscard]] size_t Used() const { return m_used; }

	/// How many dwords the arena holds, for tests.
	[[nodiscard]] size_t Capacity() const { return m_capacity; }

	/// How many times the arena was allocated since this was last asked, then forgotten.
	uint64_t TakeGrowths() { return std::exchange(m_growths, 0); }

private:
	std::unique_ptr<uint32_t[]> m_data;
	size_t                      m_capacity = 0;
	size_t                      m_used     = 0;
	uint64_t                    m_growths  = 0;
};

/// A slice of the arena that is given back when it goes out of scope, so that every way out of a
/// function that read into it releases it.
class ScratchSlice {
public:
	/// Takes `dwords` dwords from the arena, and gives them back with everything taken after them when
	/// the slice is destroyed. If they do not fit, Data is null and the arena is left as it was.
	ScratchSlice(ScratchArena& arena, size_t dwords)
	    : m_arena(arena), m_mark(arena.Mark()), m_data(arena.Take(dwords)) {}
	~ScratchSlice() { m_arena.Release(m_mark); }
	ScratchSlice(const ScratchSlice&)            = delete;
	ScratchSlice& operator=(const ScratchSlice&) = delete;

	/// The slice's dwords, or null when the arena had no room.
	[[nodiscard]] uint32_t* Data() const { return m_data; }

private:
	ScratchArena& m_arena;
	size_t        m_mark;
	uint32_t*     m_data;
};

/// The storage one scan reuses. A scanning thread keeps one, and a caller that scans from many threads
/// under a lock of its own may keep one for all of them, so that the arena exists once and not per thread.
struct ScanScratch {
	VisitedSet   visited;
	ScratchArena arena;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_SCAN_SCRATCH_H
