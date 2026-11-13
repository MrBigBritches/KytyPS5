#ifndef EMULATOR_SRC_COMMON_SPSCRING_H_
#define EMULATOR_SRC_COMMON_SPSCRING_H_

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

// A bounded first-in, first-out queue for exactly one producer thread and one consumer thread. It
// never locks, never waits and never allocates once it is built. A full ring refuses the item and
// counts the refusal, so a slow consumer can never hold the producer up.
namespace Common {

/// How far apart the producer's and the consumer's indices are kept, so that neither thread's
/// writes evict the line the other is using. It is two 64-byte lines, because x86 prefetches lines
/// in adjacent pairs and Apple's cores use 128-byte lines.
inline constexpr size_t SPSC_FALSE_SHARING_BYTES = 128;

/// A ring of `Capacity` items, which must be a power of two so that an index wraps with a mask.
/// Items are copied in and out whole, so they must be trivially copyable, which also means a copy
/// cannot allocate.
template <class Item, size_t Capacity>
class SpscRing {
	static_assert(Capacity >= 2 && std::has_single_bit(Capacity),
	              "the capacity must be a power of two");
	static_assert(std::is_trivially_copyable_v<Item>, "an item must copy without running code");

public:
	/// How many items the ring holds at once.
	static constexpr size_t CAPACITY = Capacity;

	/// Builds the ring and its storage, which is the only allocation it ever makes.
	SpscRing(): m_slots(std::make_unique<Item[]>(Capacity)) {}

	SpscRing(const SpscRing&)            = delete;
	SpscRing& operator=(const SpscRing&) = delete;

	/// Queues a copy of `item`, or refuses it and counts the refusal when the ring is full. Only
	/// the producer thread may call it.
	/// @return whether the item was queued.
	[[nodiscard]] bool TryPush(const Item& item) noexcept {
		const uint64_t tail = m_producer.tail.load(std::memory_order_relaxed);
		if (IsFull(tail, m_producer.head_seen)) {
			// The head the producer remembers may be stale, so the real one is read before
			// refusing. The acquire pairs with the consumer's release in TryPop, so a slot it freed
			// is no longer being read.
			m_producer.head_seen = m_consumer.head.load(std::memory_order_acquire);
			if (IsFull(tail, m_producer.head_seen)) {
				// Only the producer writes the count, so a plain add is enough and needs no locked
				// instruction.
				m_producer.dropped.store(m_producer.dropped.load(std::memory_order_relaxed) + 1,
				                         std::memory_order_relaxed);
				return false;
			}
		}
		m_slots[tail & INDEX_MASK] = item;
		// The release makes the slot's contents visible before the new tail, so a consumer that
		// sees the tail also sees the whole item.
		m_producer.tail.store(tail + 1, std::memory_order_release);
		return true;
	}

	/// Takes the oldest queued item into `out`. Only the consumer thread may call it.
	/// @return false when the ring is empty, leaving `out` untouched.
	[[nodiscard]] bool TryPop(Item& out) noexcept {
		const uint64_t head = m_consumer.head.load(std::memory_order_relaxed);
		if (head == m_consumer.tail_seen) {
			// The acquire pairs with the producer's release in TryPush, so the slot is read only
			// after it was written.
			m_consumer.tail_seen = m_producer.tail.load(std::memory_order_acquire);
			if (head == m_consumer.tail_seen) {
				return false;
			}
		}
		out = m_slots[head & INDEX_MASK];
		// The release keeps the read of the slot ahead of handing it back to the producer.
		m_consumer.head.store(head + 1, std::memory_order_release);
		return true;
	}

	/// How many items TryPush refused because the ring was full. Any thread may read it, and a
	/// reader on another thread may see a slightly older count.
	[[nodiscard]] uint64_t Dropped() const noexcept {
		return m_producer.dropped.load(std::memory_order_relaxed);
	}

private:
	static constexpr uint64_t INDEX_MASK = Capacity - 1;

	// Whether a ring whose producer is at `tail` and consumer at `head` has no free slot. The
	// indices only grow and never wrap in practice, so their difference is the number of queued
	// items.
	static constexpr bool IsFull(uint64_t tail, uint64_t head) noexcept {
		return tail - head >= Capacity;
	}

	// What the producer writes. The head it last read saves it a look at the consumer's line on
	// every push.
	struct alignas(SPSC_FALSE_SHARING_BYTES) ProducerSide {
		std::atomic<uint64_t> tail {0};
		uint64_t              head_seen = 0;
		std::atomic<uint64_t> dropped {0};
	};

	// What the consumer writes, with the tail it last read for the same reason.
	struct alignas(SPSC_FALSE_SHARING_BYTES) ConsumerSide {
		std::atomic<uint64_t> head {0};
		uint64_t              tail_seen = 0;
	};

	ProducerSide            m_producer;
	ConsumerSide            m_consumer;
	std::unique_ptr<Item[]> m_slots;
};

} // namespace Common

#endif // EMULATOR_SRC_COMMON_SPSCRING_H_
