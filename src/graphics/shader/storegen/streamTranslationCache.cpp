#include "graphics/shader/storegen/streamTranslationCache.h"

#include "graphics/shader/recompiler/ir/Block.h"
#include "graphics/shader/storegen/scopeExit.h"

#include <algorithm>
#include <utility>
#include <xxhash.h>

namespace Libs::Graphics::StoreGen {

namespace IR = ShaderRecompiler::IR;

namespace {

/// The two links a std::list node adds to each instruction it holds.
constexpr uint64_t LIST_NODE_LINK_BYTES = 2 * sizeof(void*);
/// What one instruction takes in a block's or a plan's list.
constexpr uint64_t INSTRUCTION_BYTES = sizeof(IR::Inst) + LIST_NODE_LINK_BYTES;
/// Where the user data count goes in the hash seed, above the bits the code hash varies most in.
constexpr uint32_t USER_DATA_SEED_SHIFT = 32;

} // namespace

size_t TranslationKeyHash::operator()(const TranslationKey& key) const {
	const uint64_t seed = key.code_hash ^ (static_cast<uint64_t>(key.user_data_count)
	                                       << USER_DATA_SEED_SHIFT);
	return static_cast<size_t>(XXH3_64bits_withSeed(
	    key.static_state.data(), key.static_state.size() * sizeof(uint32_t), seed));
}

uint64_t EstimateTranslationBytes(const ShaderRecompiler::TranslateResult& result) {
	const auto& program = result.program;
	uint64_t    bytes   = sizeof(ShaderRecompiler::TranslateResult) +
	                 result.decoded_dump.capacity() + result.cfg_dump.capacity() +
	                 program.value_storage.size() * INSTRUCTION_BYTES;
	// Each block also carries its SSA register tables, which is why the block itself is counted.
	for (const auto& block: program.block_storage) {
		bytes += sizeof(IR::Block) + block->Instructions().size() * INSTRUCTION_BYTES;
	}
	return bytes;
}

TranslationCache::TranslationCache(uint64_t cap_bytes, std::function<void()> on_wait)
    : m_cap(cap_bytes), m_on_wait(std::move(on_wait)) {}

TranslationCache::Lease TranslationCache::Acquire(const TranslationKey& key, TranslationUse use) {
	std::unique_lock lock(m_mutex);
	if (m_held.contains(key)) {
		m_stats.waits++;
		if (m_on_wait) {
			m_on_wait();
		}
		m_released.wait(lock, [&] { return !m_held.contains(key); });
	}
	m_held.insert(key);
	std::unique_ptr<ShaderRecompiler::TranslateResult> result;
	uint64_t                                           bytes = 0;
	if (const auto found = m_entries.find(key); found != m_entries.end()) {
		result = std::move(found->second.result);
		bytes  = found->second.bytes;
		m_recency.erase(found->second.recency);
		m_entries.erase(found);
		m_stats.bytes -= bytes;
		m_stats.entries--;
		m_stats.hits++;
	} else {
		(use == TranslationUse::Plan ? m_stats.plan_misses : m_stats.compile_misses)++;
	}
	return {this, key, std::move(result), bytes};
}

TranslationCacheStats TranslationCache::Stats() const {
	std::lock_guard lock(m_mutex);
	return m_stats;
}

// Puts back what a lease still holds, as the most recently used entry, then trims the cache to its
// cap and lets the next waiter in. The hold is let go and the waiters woken on every way out, even
// an allocation that throws, so no thread waits for ever on a program nobody holds. Whatever the
// trim drops is freed after the lock is let go, since freeing a large program takes a while and
// other threads may be waiting for the lock. The guards are declared so that they run in that
// order, the hold first, then the unlock, then the wake, then the frees.
void TranslationCache::Release(const TranslationKey&                              key,
                               std::unique_ptr<ShaderRecompiler::TranslateResult> result,
                               uint64_t bytes, bool taken) {
	std::vector<std::unique_ptr<ShaderRecompiler::TranslateResult>> dropped;
	const ScopeExit wake([this] { m_released.notify_all(); });
	std::lock_guard lock(m_mutex);
	const ScopeExit unhold([&] { m_held.erase(key); });
	if (taken) {
		m_stats.compile_evictions++;
	}
	if (result) {
		Insert(key, std::move(result), bytes);
	}
	TrimToCap(dropped);
}

// Adds an entry as the most recently used. The entry goes in first and its recency key after, and
// an entry whose key could not be added is taken out again, so every entry has a key and the trim
// can always find it.
void TranslationCache::Insert(const TranslationKey&                              key,
                              std::unique_ptr<ShaderRecompiler::TranslateResult> result,
                              uint64_t                                           bytes) {
	const auto [entry, added] =
	    m_entries.emplace(key, Entry {std::move(result), bytes, m_recency.end()});
	if (!added) {
		return;
	}
	ScopeExit unlist([this, at = entry] { m_entries.erase(at); });
	entry->second.recency = m_recency.insert(m_recency.begin(), key);
	unlist.Dismiss();
	m_stats.bytes += bytes;
	m_stats.entries++;
	m_stats.peak_bytes = std::max(m_stats.peak_bytes, m_stats.bytes);
}

// Drops the least recently used entries until the bytes are within the cap, handing their programs
// to the caller to free once the lock is let go.
void TranslationCache::TrimToCap(
    std::vector<std::unique_ptr<ShaderRecompiler::TranslateResult>>& dropped) {
	while (m_stats.bytes > m_cap && !m_recency.empty()) {
		const auto oldest = m_entries.find(m_recency.back());
		// Handed over first, so an allocation that throws here leaves the counts as they were.
		dropped.push_back(std::move(oldest->second.result));
		m_stats.bytes -= oldest->second.bytes;
		m_stats.entries--;
		m_stats.cap_evictions++;
		m_entries.erase(oldest);
		m_recency.pop_back();
	}
}

TranslationCache::Lease::Lease(TranslationCache* cache, TranslationKey key,
                               std::unique_ptr<ShaderRecompiler::TranslateResult> result,
                               uint64_t                                           bytes)
    : m_cache(cache), m_key(std::move(key)), m_result(std::move(result)), m_bytes(bytes),
      m_hit(m_result != nullptr) {}

TranslationCache::Lease::Lease(Lease&& other) noexcept
    : m_cache(std::exchange(other.m_cache, nullptr)), m_key(std::move(other.m_key)),
      m_result(std::move(other.m_result)), m_bytes(other.m_bytes), m_hit(other.m_hit),
      m_taken(other.m_taken) {}

TranslationCache::Lease::~Lease() {
	Release();
}

void TranslationCache::Lease::Put(ShaderRecompiler::TranslateResult result, uint64_t bytes) {
	m_result = std::make_unique<ShaderRecompiler::TranslateResult>(std::move(result));
	m_bytes  = bytes;
}

std::optional<ShaderRecompiler::TranslateResult> TranslationCache::Lease::Take() {
	if (!m_result) {
		return std::nullopt;
	}
	m_taken    = true;
	auto owned = std::exchange(m_result, nullptr);
	return std::move(*owned);
}

void TranslationCache::Lease::Release() {
	if (m_cache != nullptr) {
		std::exchange(m_cache, nullptr)->Release(m_key, std::move(m_result), m_bytes, m_taken);
	}
}

} // namespace Libs::Graphics::StoreGen
