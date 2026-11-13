#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANHANDOFF_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANHANDOFF_H_

#include "common/spscRing.h"
#include "graphics/guest_gpu/command_processor/prescanDispatchInputs.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/shader/storegen/prescanInputs.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <type_traits>

// The lock-free hand-off between the prescan's scanner thread and its preparer thread. The scanner
// copies each dispatch it found into a bounded ring and never waits, locks or allocates. The
// preparer owns every piece of state that decides what to build, and tells the scanner which
// dispatches it no longer needs through a small table of answers that the scanner only reads.
namespace Libs::Graphics {

/// How many scanned dispatches may wait for the preparer. A full ring drops the newest one and
/// counts it. The bound is the prototype's request queue's, but that queue only held dispatches
/// its gate had already let through, while the ring takes them before the gate. What keeps the
/// ring from filling with repeats is the scanner's cache, which remembers a posted key as busy.
inline constexpr size_t PRESCAN_RING_CAPACITY = 4096;
/// The alignment of a ring item, so that no two slots share a cache line and the producer writing
/// one slot never evicts the slot the consumer is reading. It is the ring's own padding, which
/// also covers Apple's 128-byte lines, and the item's fields still round to 256 bytes.
inline constexpr size_t PRESCAN_ITEM_ALIGNMENT = Common::SPSC_FALSE_SHARING_BYTES;
/// How many dispatch keys the answer table holds. It is direct mapped, so two keys that share a
/// slot replace each other, which only costs a repeat post.
inline constexpr size_t PRESCAN_FEEDBACK_SLOTS = 4096;

/// One scanned dispatch, as the preparer needs it. It is copied whole into the ring, so it holds no
/// pointers and nothing that allocates.
struct alignas(PRESCAN_ITEM_ALIGNMENT) PrescanItem {
	/// What the runtime takes from the dispatch besides the shader map.
	StoreGen::PrescanDispatchInputs inputs {};
	/// The user data at the dispatch. Materialization follows its pointers to the descriptors.
	HW::UserSgprInfo user_data;
	/// The guest address of the program, which its shader map entry is keyed by.
	uint64_t program_address = 0;
	/// The shader map's hash of the program, or 0 when the scanner did not look it up. The preparer
	/// fills it in before it asks the state about the item.
	uint64_t code_hash = 0;
	/// The scanner's dispatch key, see ScannedDispatch::key_hash. Answers are keyed by it.
	uint64_t key_hash = 0;
	/// The number the job queue gave the submission in its queue's order.
	uint64_t seq = 0;
	/// The queue the submission was made on.
	uint32_t queue = 0;
	/// How many user SGPRs the dispatch loads, which is part of the runtime's module id.
	uint32_t user_data_count = 0;
	/// Whether the walk could not vouch for the program, see CsShadow::certain.
	bool uncertain_program = false;
	/// Whether the scanner's cache let the dispatch through as a probe of a settled program.
	bool force_probe = false;
};
static_assert(std::is_trivially_copyable_v<PrescanItem>,
              "a ring item must copy without allocating");
static_assert(sizeof(PrescanItem) % PRESCAN_ITEM_ALIGNMENT == 0,
              "ring slots must fill whole cache lines");

/// Builds the ring item for a dispatch the walk found in the job numbered `seq`. The code hash is
/// left unknown, since the scanner does not read the shader map.
[[nodiscard]] inline PrescanItem MakePrescanItem(const ScannedDispatch& dispatch, uint64_t seq) {
	PrescanItem item;
	item.inputs            = PrescanInputsOf(dispatch);
	item.user_data         = dispatch.regs.cs_user_sgpr;
	item.program_address   = dispatch.program_address;
	item.key_hash          = dispatch.key_hash;
	item.seq               = seq;
	item.queue             = dispatch.queue;
	item.user_data_count   = dispatch.regs.cs_regs.user_sgpr;
	item.uncertain_program = dispatch.uncertain_program;
	item.force_probe       = dispatch.force_probe;
	return item;
}

/// What the preparer last decided about each dispatch key, for the scanner to read. Only the
/// preparer writes it and any thread may read it. Each slot is one atomic word holding the key and
/// its answer together, so a reader never pairs one key with another key's answer.
class PrescanFeedback {
public:
	/// What the preparer last said about a key. None means it still wants dispatches of it.
	enum class Answer : uint8_t { None = 0, Busy = 1, Settled = 2 };

	/// Records the answer for `key_hash`. None clears the key's own answer and leaves a different
	/// key that shares the slot alone. Key 0 means no key and is ignored. Only the preparer may
	/// call it.
	void Publish(uint64_t key_hash, Answer answer) noexcept {
		if (key_hash == 0) {
			return;
		}
		auto& slot = m_slots[key_hash & SLOT_MASK];
		// Relaxed is enough because the word carries everything a reader needs and no other memory
		// is published through it. A reader that sees an older answer only posts or skips one
		// dispatch more.
		if (answer == Answer::None) {
			if (Matches(slot.load(std::memory_order_relaxed), key_hash)) {
				slot.store(0, std::memory_order_relaxed);
			}
			return;
		}
		slot.store((key_hash & ~ANSWER_MASK) | static_cast<uint64_t>(answer),
		           std::memory_order_relaxed);
	}

	/// The answer last published for `key_hash`, or None when there is none. Any thread may call
	/// it.
	[[nodiscard]] Answer Read(uint64_t key_hash) const noexcept {
		if (key_hash == 0) {
			return Answer::None;
		}
		const uint64_t word = m_slots[key_hash & SLOT_MASK].load(std::memory_order_relaxed);
		return Matches(word, key_hash) ? static_cast<Answer>(word & ANSWER_MASK) : Answer::None;
	}

	/// The answer as the scanner's settled-dispatch cache takes it.
	[[nodiscard]] static DispatchCache::PostAnswer ForCache(Answer answer) noexcept {
		switch (answer) {
			case Answer::Busy: return DispatchCache::PostAnswer::Busy;
			case Answer::Settled: return DispatchCache::PostAnswer::Settled;
			case Answer::None: break;
		}
		return DispatchCache::PostAnswer::Other;
	}

private:
	// The answer sits in the key's two lowest bits. Keys that share a slot already agree on every
	// bit the slot index uses, which includes those two, so dropping them never makes two keys look
	// alike.
	static constexpr uint64_t ANSWER_MASK = 3;
	static constexpr uint64_t SLOT_MASK   = PRESCAN_FEEDBACK_SLOTS - 1;
	static_assert(std::has_single_bit(PRESCAN_FEEDBACK_SLOTS) && SLOT_MASK >= ANSWER_MASK,
	              "the slot index must cover the bits the answer replaces");
	// A slot that took a lock would let the preparer hold the scanner up, which the hand-off
	// forbids.
	static_assert(std::atomic<uint64_t>::is_always_lock_free, "the answer table must be lock-free");

	// Whether a slot's word belongs to `key_hash`.
	static constexpr bool Matches(uint64_t word, uint64_t key_hash) noexcept {
		return (word & ~ANSWER_MASK) == (key_hash & ~ANSWER_MASK);
	}

	std::array<std::atomic<uint64_t>, PRESCAN_FEEDBACK_SLOTS> m_slots {};
};

/// Everything the scanner thread and the preparer thread share. The scanner is the ring's one
/// producer and the preparer its one consumer and the answer table's one writer. The owner keeps it
/// alive until both threads are joined.
struct PrescanHandoff {
	Common::SpscRing<PrescanItem, PRESCAN_RING_CAPACITY> ring;
	PrescanFeedback                                      feedback;
	/// Bumped by everything that gives the preparer work, which is the scanner after it pushed,
	/// the helper's completion callback after it queued an answer, and shutdown. The preparer reads
	/// it before it drains and then waits for it to change, so a bump that lands during the drain
	/// is never missed. See WakePreparer.
	std::atomic<uint32_t> wake_epoch {0};
	/// Set by the preparer, with release, once the prescan is off for the rest of the run, because
	/// the helper's lane turned off, the preparer failed or it was stopped. Nothing drains the ring
	/// after that, so the scanner reads it with acquire once per submission and skips the walk.
	std::atomic<bool> preparer_stopped {false};
	// A counter that took a lock would let a producer wait on the preparer, which the hand-off
	// forbids. This checks the counter only. Whether a platform's notify_one takes a lock of its
	// own is not checked here.
	static_assert(std::atomic<uint32_t>::is_always_lock_free, "the wake word must be lock-free");
};

/// Tells the preparer that new work was published. The caller must publish first, with release or
/// under a mutex the preparer takes, so the preparer that sees the new epoch also sees the work.
/// The scanner calls it once per walked submission and not per item, since each call may enter the
/// kernel.
inline void WakePreparer(PrescanHandoff& handoff) noexcept {
	handoff.wake_epoch.fetch_add(1, std::memory_order_release);
	handoff.wake_epoch.notify_one();
}

/// What became of a dispatch the scanner offered to the preparer.
enum class PrescanOffer : uint8_t {
	/// It is in the ring, and the scanner's cache remembers the key as busy.
	Posted,
	/// The preparer had already said it needs no more dispatches of the key, so it was not sent,
	/// and the scanner's cache now remembers the key.
	Skipped,
	/// The ring was full, so it was dropped and the ring counted it. The key is not remembered,
	/// so a later dispatch of it may be sent.
	Dropped,
};

/// Hands one dispatch from the scanner thread to the preparer. A key the preparer answered Busy or
/// Settled is not sent. A key that is sent is busy until the preparer answers it, so the scanner's
/// cache remembers it either way, and later dispatches of it are skipped during the walk until the
/// cache lets one through as a probe. A probe is always sent. A dispatch whose program the walk
/// could not vouch for follows the same answers, since they were given for the same key, but
/// `cache` never remembers it, because its key may name a program the walk only guessed at. It
/// never waits, locks or allocates. Only the scanner thread may call it.
[[nodiscard]] inline PrescanOffer OfferToPreparer(PrescanHandoff& handoff, DispatchCache& cache,
                                                  const PrescanItem& item) noexcept {
	if (!item.force_probe) {
		const auto answer = handoff.feedback.Read(item.key_hash);
		if (answer != PrescanFeedback::Answer::None) {
			(void)cache.Note(item.key_hash, PrescanFeedback::ForCache(answer),
			                 item.uncertain_program);
			return PrescanOffer::Skipped;
		}
	}
	if (!handoff.ring.TryPush(item)) {
		return PrescanOffer::Dropped;
	}
	// Until the preparer answers, every repeat of the key would only be answered busy, so the cache
	// keeps them out of the ring, with the same probe a busy answer gets.
	(void)cache.Note(item.key_hash, DispatchCache::PostAnswer::Busy, item.uncertain_program);
	return PrescanOffer::Posted;
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANHANDOFF_H_
