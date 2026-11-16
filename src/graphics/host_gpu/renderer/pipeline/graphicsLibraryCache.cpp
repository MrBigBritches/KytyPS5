#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"

#include "common/threadPriority.h"

#include <thread>

namespace Libs::Graphics {

namespace {

std::atomic<LibraryFaultInjector> g_fault_injector {nullptr};
std::atomic<uint64_t>             g_total_links {0};

// The report's name for a part, which has the same order.
Common::FirstRunReport::LibraryKind ReportKind(LibraryPart part) {
	return static_cast<Common::FirstRunReport::LibraryKind>(part);
}

} // namespace

GraphicsLibraryCache::GraphicsLibraryCache(DestroyFunction destroy)
    : m_destroy(std::move(destroy)),
      m_priority {.current_thread   = &Common::CurrentThreadSystemId,
                  .boost            = &Common::BoostThreadToNormal,
                  .lower            = [] { (void)Common::LowerCurrentThreadPriorityToLowest(); },
                  .current_priority = &Common::CurrentThreadSystemPriority,
                  .boost_to         = &Common::BoostThreadToPriority} {}

// Only the request that moves the prebuild from Running to Boosting asks the system, so a prebuild is
// boosted at most once however many requests wait on it. A refused boost goes back to Running, and the
// next waiting request may try again.
// At most MAX_CONCURRENT_BOOSTS prebuilds run boosted at once, so boosts cannot crowd out the game's
// own threads. A slot is reserved before the system is asked and given back if it refuses. A waiter
// that finds every slot taken waits unboosted, and its entry goes back to Running for a later waiter.
// A blocking waiter is not held to the cap, since the draw it holds up waits on this one prebuild,
// but it raises the prebuild only to its own priority, so it never puts the prebuild above itself.
// It still takes a slot, so the boosts it makes keep other waiters within the cap.
void GraphicsLibraryCache::TryBoostPrebuild(Entry& entry, LibraryPart part, bool blocking) {
	uint8_t expected = BOOST_RUNNING;
	if (!entry.boost_state.compare_exchange_strong(expected, BOOST_BOOSTING, std::memory_order_acq_rel)) {
		return;
	}
	const uint64_t active = m_active_boosts.fetch_add(1, std::memory_order_acq_rel) + 1;
	if (!blocking && active > MAX_CONCURRENT_BOOSTS) {
		m_active_boosts.fetch_sub(1, std::memory_order_acq_rel);
		Prebuilt(part).boosts_refused_cap.fetch_add(1, std::memory_order_relaxed);
		entry.boost_state.store(BOOST_RUNNING, std::memory_order_release);
		return;
	}
	const uint64_t builder = entry.builder_thread.load(std::memory_order_relaxed);
	const bool     to_own  = blocking && m_priority.boost_to && m_priority.current_priority;
	const bool     raised  = to_own ? m_priority.boost_to(builder, m_priority.current_priority()) : m_priority.boost(builder);
	if (raised) {
		Prebuilt(part).boosts.fetch_add(1, std::memory_order_relaxed);
		Prebuilt(part).boosts_blocking.fetch_add(blocking ? 1 : 0, std::memory_order_relaxed);
		RaiseMaxCount(Prebuilt(part).boosts_peak, active);
		Common::FirstRunReport::NotePrescanGraphicsBoosted(true);
	} else {
		m_active_boosts.fetch_sub(1, std::memory_order_acq_rel);
	}
	entry.boost_state.store(raised ? BOOST_BOOSTED : BOOST_RUNNING, std::memory_order_release);
}

// Raises an atomic maximum. Any thread may race on it, and a lost race retries.
void GraphicsLibraryCache::RaiseMaxCount(std::atomic<uint64_t>& maximum, uint64_t value) {
	uint64_t seen = maximum.load(std::memory_order_relaxed);
	while (seen < value && !maximum.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
	}
}

// Runs on the prebuild's thread after its build. A boost still being asked for is waited out, which
// is one system call, so the thread never finishes between a request's check and its boost and then
// keeps normal priority for the rest of its jobs.
void GraphicsLibraryCache::FinishPrebuildBoost(Entry& entry) {
	uint8_t state = entry.boost_state.load(std::memory_order_acquire);
	while (true) {
		if (state == BOOST_BOOSTING) {
			std::this_thread::yield();
			state = entry.boost_state.load(std::memory_order_acquire);
			continue;
		}
		if (entry.boost_state.compare_exchange_weak(state, BOOST_FINISHED, std::memory_order_acq_rel)) {
			break;
		}
	}
	if (state == BOOST_BOOSTED) {
		m_priority.lower();
		m_active_boosts.fetch_sub(1, std::memory_order_acq_rel);
		Common::FirstRunReport::NotePrescanGraphicsBoosted(false);
	}
}

GraphicsLibraryCache::~GraphicsLibraryCache() {
	// A linked pipeline is in use as long as its libraries are, so every one must be destroyed first.
	EXIT_IF(m_live_links.load() != 0);
	DestroyTable(m_vertex_input);
	DestroyTable(m_pre_raster);
	DestroyTable(m_fragment);
	DestroyTable(m_output);
}

// Destroys the libraries of one table. A refused library holds null handles, since its layout was
// destroyed when it was refused, so only libraries that were made are destroyed.
template <class Key>
void GraphicsLibraryCache::DestroyTable(Table<Key>& table) {
	std::scoped_lock lock(table.mutex);
	for (auto& entry: table.entries) {
		const auto& handles = entry.second.handles;
		if (handles.pipeline != nullptr || handles.layout != nullptr) {
			m_destroy(handles);
		}
	}
	table.entries.clear();
}

std::size_t GraphicsLibraryCache::Size(LibraryPart part) const {
	const auto count = [](const auto& table) {
		std::scoped_lock lock(table.mutex);
		return table.entries.size();
	};
	switch (part) {
		case LibraryPart::VertexInput: return count(m_vertex_input);
		case LibraryPart::PreRaster: return count(m_pre_raster);
		case LibraryPart::Fragment: return count(m_fragment);
		case LibraryPart::Output: return count(m_output);
	}
	return 0;
}

LibraryCounts GraphicsLibraryCache::Counts(LibraryPart part) const {
	const auto index = static_cast<std::size_t>(part);
	return {m_built[index].load(), m_hits[index].load(), m_failed[index].load(),
	        m_refused_hits[index].load()};
}

void GraphicsLibraryCache::NoteBuilt(LibraryPart part, bool succeeded, double ms) {
	const auto index = static_cast<std::size_t>(part);
	if (succeeded) {
		m_built[index]++;
		Common::FirstRunReport::AddLibraryBuilt(ReportKind(part), ms);
		Common::FirstRunReport::LibraryPipelineScope::NoteBuilt(ReportKind(part), ms);
	} else {
		m_failed[index]++;
		Common::FirstRunReport::AddLibraryFailed(ReportKind(part));
	}
}

void GraphicsLibraryCache::NoteHit(LibraryPart part, bool usable) {
	if (usable) {
		m_hits[static_cast<std::size_t>(part)]++;
		Common::FirstRunReport::AddLibraryHit(ReportKind(part));
	} else {
		m_refused_hits[static_cast<std::size_t>(part)]++;
		Common::FirstRunReport::AddLibraryRefusedHit(ReportKind(part));
	}
}

void GraphicsLibraryCache::NoteLink(double ms) {
	Common::FirstRunReport::LibraryPipelineScope::NoteLink(ms);
	m_links++;
	m_live_links++;
	g_total_links++;
	Common::FirstRunReport::AddLibraryLink(ms);
}

void GraphicsLibraryCache::NoteUnlinked() {
	EXIT_IF(m_live_links.load() == 0);
	m_live_links--;
}

uint64_t GraphicsLibraryCache::LiveLinks() const {
	return m_live_links.load();
}

uint64_t GraphicsLibraryCache::Links() const {
	return m_links.load();
}

void GraphicsLibraryCache::NoteFallback(Common::FirstRunReport::FallbackKind reason) {
	m_fallbacks[static_cast<std::size_t>(reason)]++;
	Common::FirstRunReport::AddLibraryFallback(reason);
}

uint64_t GraphicsLibraryCache::Fallbacks(Common::FirstRunReport::FallbackKind reason) const {
	return m_fallbacks[static_cast<std::size_t>(reason)].load();
}

// Every request that waited on a prebuild counts as a wait. The first request that found it finished
// counts as its first hit, and a wait claims the entry too, so a later finished request is not taken
// for a first hit.
void GraphicsLibraryCache::NotePrebuiltUse(Entry& entry, LibraryPart part, bool waited, double waited_ms) {
	if (waited) {
		constexpr double US_PER_MS = 1000.0;
		const auto       waited_us = static_cast<uint64_t>(waited_ms * US_PER_MS);
		entry.prebuilt_claimed.store(true, std::memory_order_relaxed);
		Prebuilt(part).waits.fetch_add(1, std::memory_order_relaxed);
		Prebuilt(part).wait_us.fetch_add(waited_us, std::memory_order_relaxed);
		uint64_t seen = Prebuilt(part).wait_max_us.load(std::memory_order_relaxed);
		while (seen < waited_us &&
		       !Prebuilt(part).wait_max_us.compare_exchange_weak(seen, waited_us, std::memory_order_relaxed)) {
		}
		return;
	}
	if (!entry.prebuilt_claimed.exchange(true, std::memory_order_relaxed)) {
		Prebuilt(part).first_hits.fetch_add(1, std::memory_order_relaxed);
	}
}

void GraphicsLibraryCache::NotePrebuiltStage(LibraryPart part, uint64_t stage_id) {
	std::scoped_lock lock(m_prebuilt_pixels_mutex);
	m_prebuilt_ids[static_cast<std::size_t>(part)].insert(stage_id);
}

void GraphicsLibraryCache::NoteBuiltHere(LibraryPart part, uint64_t stage_id) {
	std::scoped_lock lock(m_prebuilt_pixels_mutex);
	if (m_prebuilt_ids[static_cast<std::size_t>(part)].contains(stage_id)) {
		Prebuilt(part).key_differs.fetch_add(1, std::memory_order_relaxed);
	}
}

PrebuiltLibraryCounts GraphicsLibraryCache::PrebuiltCounts(LibraryPart part) const {
	constexpr double MS_PER_US = 1e-3;
	return {.built         = Prebuilt(part).built.load(std::memory_order_relaxed),
	        .already_there = Prebuilt(part).already_there.load(std::memory_order_relaxed),
	        .first_hits    = Prebuilt(part).first_hits.load(std::memory_order_relaxed),
	        .waits         = Prebuilt(part).waits.load(std::memory_order_relaxed),
	        .wait_ms       = static_cast<double>(Prebuilt(part).wait_us.load(std::memory_order_relaxed)) * MS_PER_US,
	        .wait_max_ms = static_cast<double>(Prebuilt(part).wait_max_us.load(std::memory_order_relaxed)) * MS_PER_US,
	        .key_differs = Prebuilt(part).key_differs.load(std::memory_order_relaxed),
	        .boosts      = Prebuilt(part).boosts.load(std::memory_order_relaxed),
	        .boosts_refused_cap = Prebuilt(part).boosts_refused_cap.load(std::memory_order_relaxed),
	        .boosts_peak        = Prebuilt(part).boosts_peak.load(std::memory_order_relaxed),
	        .boosts_blocking    = Prebuilt(part).boosts_blocking.load(std::memory_order_relaxed)};
}

uint64_t TotalLinkedPipelines() {
	return g_total_links.load();
}

void SetLibraryFaultInjector(LibraryFaultInjector injector) {
	g_fault_injector.store(injector, std::memory_order_release);
}

bool LibraryStepShouldFail(LibraryStep step) {
	const auto injector = g_fault_injector.load(std::memory_order_acquire);
	return injector != nullptr && injector(step);
}

} // namespace Libs::Graphics
