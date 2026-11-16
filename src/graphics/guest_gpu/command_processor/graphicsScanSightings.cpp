#include "graphics/guest_gpu/command_processor/graphicsScanSightings.h"

namespace Libs::Graphics {

namespace {

// The bit that marks the vertex stage's key, above any guest address.
constexpr uint64_t VERTEX_SIGHTING_BIT = uint64_t {1} << 63u;

} // namespace

uint64_t SightingKey(SightingStage stage, uint64_t address) {
	return stage == SightingStage::Vertex ? address | VERTEX_SIGHTING_BIT : address;
}

const char* ScanVerdictName(ScanVerdict verdict) {
	switch (verdict) {
		case ScanVerdict::Recorded: return "recorded";
		case ScanVerdict::Cached: return "cached";
		case ScanVerdict::NoPixelStage: return "no_pixel_stage";
		case ScanVerdict::PsUnwritten: return "ps_unwritten";
		case ScanVerdict::SkippedTessellation: return "skipped_tessellation";
		case ScanVerdict::SkippedRectList: return "skipped_rect_list";
		case ScanVerdict::SkippedObjectId: return "skipped_object_id";
		case ScanVerdict::SkippedClipPlanes: return "skipped_clip_planes";
		case ScanVerdict::UnsupportedInputs: return "unsupported_inputs";
		case ScanVerdict::UnsupportedTargetFormat: return "unsupported_target_format";
		case ScanVerdict::UnsupportedPackedUnorm: return "unsupported_packed_unorm";
		case ScanVerdict::Count: break;
	}
	return "unknown";
}

std::vector<std::string> ScanVerdictNames(uint32_t verdicts) {
	std::vector<std::string> names;
	for (uint32_t index = 0; index < static_cast<uint32_t>(ScanVerdict::Count); index++) {
		const auto verdict = static_cast<ScanVerdict>(index);
		if ((verdicts & ScanVerdictBit(verdict)) != 0) {
			names.emplace_back(ScanVerdictName(verdict));
		}
	}
	return names;
}

ShaderSighting* GraphicsShaderSightings::Entry(SightingStage stage, uint64_t address, int64_t now_ns) {
	const auto key   = SightingKey(stage, address);
	auto       found = m_sightings.find(key);
	if (found != m_sightings.end()) {
		return &found->second;
	}
	if (m_sightings.size() >= GRAPHICS_SIGHTING_CAP) {
		m_dropped++;
		return nullptr;
	}
	return &m_sightings.emplace(key, ShaderSighting {.first_seen_ns = now_ns}).first->second;
}

void GraphicsShaderSightings::NoteScan(SightingStage stage, uint64_t address, ScanVerdict verdict, int64_t now_ns) {
	if (address == 0) {
		return;
	}
	std::scoped_lock lock(m_mutex);
	if (auto* sighting = Entry(stage, address, now_ns)) {
		sighting->scan_verdicts |= ScanVerdictBit(verdict);
	}
}

// A repeat only fills an empty job, so a later deduped draw never hides what the first one of its key
// met.
void GraphicsShaderSightings::NoteJob(SightingStage stage, uint64_t address, const char* outcome, SightingKind kind,
                                      int64_t now_ns) {
	if (address == 0 || outcome == nullptr) {
		return;
	}
	std::scoped_lock lock(m_mutex);
	auto*            sighting = Entry(stage, address, now_ns);
	if (sighting == nullptr) {
		return;
	}
	if (kind != SightingKind::Repeat || sighting->job == nullptr) {
		sighting->job = outcome;
	}
	if (kind == SightingKind::Queued && sighting->offered_ns == SIGHTING_NO_TIME) {
		sighting->offered_ns = now_ns;
	} else if (kind == SightingKind::Posted) {
		sighting->posted_ns = now_ns;
	}
}

std::optional<ShaderSighting> GraphicsShaderSightings::Find(SightingStage stage, uint64_t address) const {
	std::scoped_lock lock(m_mutex);
	const auto       found = m_sightings.find(SightingKey(stage, address));
	if (found == m_sightings.end()) {
		return std::nullopt;
	}
	return found->second;
}

size_t GraphicsShaderSightings::Kept() const {
	std::scoped_lock lock(m_mutex);
	return m_sightings.size();
}

uint64_t GraphicsShaderSightings::Dropped() const {
	std::scoped_lock lock(m_mutex);
	return m_dropped;
}

} // namespace Libs::Graphics
