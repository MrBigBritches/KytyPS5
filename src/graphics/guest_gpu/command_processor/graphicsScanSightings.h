#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GRAPHICS_SCAN_SIGHTINGS_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GRAPHICS_SCAN_SIGHTINGS_H

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// What the graphics prescan did with the draws of each shader code address, kept only for the
// first-run report's miss rows (experimental). The scanner's walk notes its verdicts on the draws and the
// pipeline cache notes what its sink and workers did with them, so a pixel or vertex shader the GPU
// thread later compiles itself can say whether the prescan ever met it, and how far it got.
namespace Libs::Graphics {

/// The most shader addresses the sightings keep, about 1.6 MB when full. A new address past it is
/// counted as dropped. Only a run with the first-run report keeps any.
inline constexpr size_t GRAPHICS_SIGHTING_CAP = 16384;

/// A time that did not happen, on the first-run report's NowNs clock.
inline constexpr int64_t SIGHTING_NO_TIME = std::numeric_limits<int64_t>::min();

/// Which stage an address is a sighting of. One program at one address has one entry per stage.
enum class SightingStage : uint8_t { Pixel, Vertex };

/// The walk's verdicts on a draw, each a bit of a sighting. A shader several draws share, such as a
/// vertex shader of both depth-only and colour passes, can collect several, and every one stays.
/// Recorded is first, so a list of names shows it first. The names are those of the walk's counts.
enum class ScanVerdict : uint8_t {
	Recorded,
	Cached,
	NoPixelStage,
	PsUnwritten,
	SkippedTessellation,
	SkippedRectList,
	SkippedObjectId,
	SkippedClipPlanes,
	UnsupportedInputs,
	UnsupportedTargetFormat,
	UnsupportedPackedUnorm,
	Count,
};

/// The bit of one verdict in a sighting's scan_verdicts.
[[nodiscard]] constexpr uint32_t ScanVerdictBit(ScanVerdict verdict) {
	return uint32_t {1} << static_cast<uint32_t>(verdict);
}

/// The report's name of one verdict.
[[nodiscard]] const char* ScanVerdictName(ScanVerdict verdict);

/// The names of the verdicts set in `verdicts`, in ScanVerdict order.
[[nodiscard]] std::vector<std::string> ScanVerdictNames(uint32_t verdicts);

/// How a job outcome is taken. A Repeat, such as a draw the sink's dedupe already had, only says the
/// same draw came again, so it never replaces an outcome that is there. Queued also stamps the first
/// time a job was queued, and Posted the last time one posted a module.
enum class SightingKind : uint8_t { Verdict, Repeat, Queued, Posted };

/// What the sightings remember of one address. scan_verdicts has a bit for each walk verdict seen, and
/// job is the pipeline cache's last outcome, a static name or null for none.
struct ShaderSighting {
	int64_t     first_seen_ns = SIGHTING_NO_TIME;
	uint32_t    scan_verdicts = 0;
	const char* job           = nullptr;
	int64_t     offered_ns    = SIGHTING_NO_TIME;
	int64_t     posted_ns     = SIGHTING_NO_TIME;
};

/// The sightings, bounded at GRAPHICS_SIGHTING_CAP addresses. Thread safe. The scanner thread, the
/// translate workers and the GPU thread all use it, each note under one short lock.
class GraphicsShaderSightings {
public:
	/// Adds the walk's verdict for the address at `now_ns`. An address 0 is never kept.
	void NoteScan(SightingStage stage, uint64_t address, ScanVerdict verdict, int64_t now_ns);
	/// Notes what the sink or a worker did with the address at `now_ns`, see SightingKind. An address
	/// 0 or a null outcome is never kept.
	void NoteJob(SightingStage stage, uint64_t address, const char* outcome, SightingKind kind, int64_t now_ns);
	/// What is remembered of the address, or nothing when it was never noted or did not fit.
	[[nodiscard]] std::optional<ShaderSighting> Find(SightingStage stage, uint64_t address) const;
	/// How many addresses are kept, and how many new ones were turned away past the cap.
	[[nodiscard]] size_t   Kept() const;
	[[nodiscard]] uint64_t Dropped() const;

private:
	// The entry of the address, made at `now_ns` when there is room, or null. The lock is held.
	[[nodiscard]] ShaderSighting* Entry(SightingStage stage, uint64_t address, int64_t now_ns);

	mutable std::mutex                           m_mutex;
	std::unordered_map<uint64_t, ShaderSighting> m_sightings;
	uint64_t                                     m_dropped = 0;
};

/// The key one stage's address is kept under. Guest addresses fit in 48 bits, so the top bit tells
/// the vertex stage's entry from the pixel stage's.
[[nodiscard]] uint64_t SightingKey(SightingStage stage, uint64_t address);

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GRAPHICS_SCAN_SIGHTINGS_H
