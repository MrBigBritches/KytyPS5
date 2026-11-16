#include "graphics/host_gpu/renderer/pipeline/graphicsLibraryCache.h"

namespace Libs::Graphics {

namespace {

std::atomic<LibraryFaultInjector> g_fault_injector {nullptr};
std::atomic<uint64_t>             g_total_links {0};

// The report's name for a part, which has the same order.
Common::FirstRunReport::LibraryKind ReportKind(LibraryPart part) {
	return static_cast<Common::FirstRunReport::LibraryKind>(part);
}

} // namespace

GraphicsLibraryCache::GraphicsLibraryCache(DestroyFunction destroy) : m_destroy(std::move(destroy)) {}

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
	        .key_differs = Prebuilt(part).key_differs.load(std::memory_order_relaxed)};
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
