#ifndef GRAPHICS_SHADER_STOREGEN_STREAM_TRANSLATION_CACHE_H
#define GRAPHICS_SHADER_STOREGEN_STREAM_TRANSLATION_CACHE_H

#include "graphics/shader/recompiler/ShaderRecompiler.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// The Stream worker's cache of translated programs. A program's first variant needs a Plan and then
// a Compile, and translation is most of the work of both, so the Plan's translation is kept for
// the Compile instead of being made twice. Compiling consumes the IR, which cannot be copied, so a
// Compile takes its entry out of the cache.
namespace Libs::Graphics::StoreGen {

/// The byte cap of a Stream worker's cache. Plans and their Compiles arrive close together, so the
/// cache only has to hold the programs between the two, and this leaves room for a burst of first
/// variants while keeping the child's memory bounded.
constexpr uint64_t STREAM_IR_CACHE_BYTES = 256ull * 1024ull * 1024ull;

/// What a translation is made from besides the code, which the code hash stands for. Translation
/// reads the input info only through what the static key holds, and the user data only through its
/// count, so equal keys give equal IR.
struct TranslationKey {
	/// The hash of the program's code, which stands for the code itself.
	uint64_t code_hash = 0;
	/// How many user data registers the program reads, which sizes its user data.
	uint32_t user_data_count = 0;
	/// The static key words from BuildStageStaticKey, which hold every input info field that
	/// translation reads.
	std::vector<uint32_t> static_state;

	bool operator==(const TranslationKey&) const = default;
};

/// Hashes a TranslationKey for the cache's tables.
struct TranslationKeyHash {
	size_t operator()(const TranslationKey& key) const;
};

/// What the cache has done since it was made, for the worker's summary and the report.
struct TranslationCacheStats {
	/// Leases that found the program translated.
	uint64_t hits = 0;
	/// Leases that did not, after which the holder translates, split by what the lease was for. A
	/// Compile miss means the Plan's translation was gone, so it is the cost the cache failed to save.
	uint64_t plan_misses    = 0;
	uint64_t compile_misses = 0;
	/// Entries a Compile took out, which is where a program's IR normally ends.
	uint64_t compile_evictions = 0;
	/// Entries dropped, oldest first, to get back under the byte cap.
	uint64_t cap_evictions = 0;
	/// Leases that had to wait for another thread to finish with the same program.
	uint64_t waits = 0;
	/// Entries and their estimated bytes now, leaving out programs held by a lease.
	uint64_t entries = 0;
	uint64_t bytes   = 0;
	/// The most estimated bytes held at once.
	uint64_t peak_bytes = 0;
};

/// What a lease is taken for, which only decides which miss count a miss goes to.
enum class TranslationUse : uint8_t {
	Plan,    ///< Reads the translation and puts it back.
	Compile, ///< Takes the translation out to compile it.
};

/// A rough size of a translated program in memory, counting its blocks and instructions, which
/// are most of it. It sizes the cache against its cap and need not be exact.
[[nodiscard]] uint64_t EstimateTranslationBytes(const ShaderRecompiler::TranslateResult& result);

/// A bounded, least recently used cache of translated programs, which threads share. One thread at
/// a time holds a program through a Lease, so a program is never translated by two threads at once,
/// and a second thread that asks for it waits until the first is done with it.
class TranslationCache {
public:
	class Lease;

	/// @param cap_bytes the estimated bytes the entries may hold. Above it the least recently used
	///                  entries go first.
	/// @param on_wait   called, with the cache's lock held, when a lease has to wait. A test uses
	///                  it to know that the second thread is really waiting.
	explicit TranslationCache(uint64_t cap_bytes, std::function<void()> on_wait = {});
	TranslationCache(const TranslationCache&)            = delete;
	TranslationCache& operator=(const TranslationCache&) = delete;

	/// Takes a program for this thread alone, waiting while another thread holds it. A cached
	/// translation leaves the cache for as long as the lease holds it, so no other thread reads it
	/// meanwhile, which matters because a plan's walker scratch is written through a const program.
	[[nodiscard]] Lease Acquire(const TranslationKey& key, TranslationUse use);

	/// The counts and sizes so far.
	[[nodiscard]] TranslationCacheStats Stats() const;

private:
	struct Entry {
		std::unique_ptr<ShaderRecompiler::TranslateResult> result;
		uint64_t                                           bytes = 0;
		std::list<TranslationKey>::iterator                recency;
	};

	void Release(const TranslationKey& key, std::unique_ptr<ShaderRecompiler::TranslateResult> result,
	             uint64_t bytes, bool taken);
	void Insert(const TranslationKey& key, std::unique_ptr<ShaderRecompiler::TranslateResult> result,
	            uint64_t bytes);
	void TrimToCap(std::vector<std::unique_ptr<ShaderRecompiler::TranslateResult>>& dropped);

	mutable std::mutex      m_mutex;
	std::condition_variable m_released;
	std::unordered_map<TranslationKey, Entry, TranslationKeyHash> m_entries;
	/// The keys of the entries, most recently used first.
	std::list<TranslationKey>                              m_recency;
	std::unordered_set<TranslationKey, TranslationKeyHash> m_held;
	uint64_t                                               m_cap = 0;
	std::function<void()>                                  m_on_wait;
	TranslationCacheStats                                  m_stats;
};

/// One thread's hold on one program of a TranslationCache. Ending the lease, by Release or by its
/// destructor, puts a translation it still holds back as the most recently used entry and lets the
/// next thread that waits for the program have it.
class TranslationCache::Lease {
public:
	Lease(Lease&& other) noexcept;
	Lease& operator=(Lease&&)      = delete;
	Lease(const Lease&)            = delete;
	Lease& operator=(const Lease&) = delete;
	~Lease();

	/// Whether the program was in the cache when the lease began.
	[[nodiscard]] bool Hit() const { return m_hit; }

	/// The translation the lease holds, or null when there is none.
	[[nodiscard]] ShaderRecompiler::TranslateResult* Result() { return m_result.get(); }

	/// Gives the lease a translation the holder made, to go into the cache when the lease ends.
	/// @param bytes its estimated size, from EstimateTranslationBytes.
	void Put(ShaderRecompiler::TranslateResult result, uint64_t bytes);

	/// Takes the translation out for a Compile, so it does not go back into the cache.
	/// @return nothing when the lease holds none.
	[[nodiscard]] std::optional<ShaderRecompiler::TranslateResult> Take();

	/// Ends the lease now. Later calls do nothing.
	void Release();

private:
	friend class TranslationCache;
	Lease(TranslationCache* cache, TranslationKey key,
	      std::unique_ptr<ShaderRecompiler::TranslateResult> result, uint64_t bytes);

	TranslationCache*                                  m_cache = nullptr;
	TranslationKey                                     m_key;
	std::unique_ptr<ShaderRecompiler::TranslateResult> m_result;
	uint64_t                                           m_bytes = 0;
	bool                                               m_hit   = false;
	bool                                               m_taken = false;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_STREAM_TRANSLATION_CACHE_H
