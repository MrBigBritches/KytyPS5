#include "common/firstRunReport.h"

#include "common/config.h"
#include "common/file.h"
#include "common/gpuStall.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstring>
#include <exception>
#include <fmt/format.h>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <random>
#include <string_view>
#include <thread>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// psapi.h needs windows.h first.
#include <psapi.h>
#if defined(_M_X64) || defined(_M_IX86)
#include <intrin.h>
#endif
#else
#include <sys/resource.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#endif

namespace Common::FirstRunReport {

double ElapsedMs(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
	    .count();
}

namespace {

constexpr double   ROUNDING_SCALE       = 1000.0;
constexpr uint32_t SPIRV_MAGIC          = 0x07230203;
constexpr size_t   SPIRV_HEADER_WORDS   = 5;
constexpr uint32_t OP_SOURCE_EXTENSION  = 4;
// Opcodes of the instructions that come before the debug section, in module order.
constexpr std::array<uint32_t, 7> OPCODES_BEFORE_DEBUG {
    10,  // OpExtension
    11,  // OpExtInstImport
    14,  // OpMemoryModel
    15,  // OpEntryPoint
    16,  // OpExecutionMode
    17,  // OpCapability
    331, // OpExecutionModeId
};
// The stamp is 11 characters and a terminating zero, which fill exactly three words.
constexpr size_t   STAMP_TEXT_CHARS = 11;
constexpr size_t   STAMP_TEXT_WORDS = 3;
constexpr uint32_t STAMP_WORD_COUNT = 1 + STAMP_TEXT_WORDS;
constexpr unsigned BITS_PER_STAMP_CHAR = 5;
// Every character is nonzero, so the string cannot end before the zero the stamp ends with.
constexpr std::string_view STAMP_ALPHABET = "0123456789abcdefghijklmnopqrstuv";
constexpr uint64_t         MIX_ONE        = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t         MIX_TWO        = 0x94D049BB133111EBull;

constexpr std::string_view REPORT_DIRECTORY = "_FirstRunReports";
constexpr int              JSON_INDENT      = 1;
constexpr double           NS_PER_MS        = 1.0e6;

int64_t ToNs(std::chrono::steady_clock::time_point point) {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(point.time_since_epoch()).count();
}

double Round3(double value) {
	return std::round(value * ROUNDING_SCALE) / ROUNDING_SCALE;
}

std::string Hex(uint64_t value) {
	return fmt::format("{:016x}", value);
}

// The splitmix64 finalizer, which spreads a value over all 64 bits.
uint64_t Mix(uint64_t value) {
	value = (value ^ (value >> 30u)) * MIX_ONE;
	value = (value ^ (value >> 27u)) * MIX_TWO;
	return value ^ (value >> 31u);
}

bool IsHeaderOpcode(uint32_t opcode) {
	return std::ranges::find(OPCODES_BEFORE_DEBUG, opcode) != OPCODES_BEFORE_DEBUG.end();
}

// State for the file written at exit.
struct Output {
	std::mutex                            mutex;
	Metadata                              metadata;
	std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
};

Output& GetOutput() {
	static Output& output = *new Output;
	return output;
}

std::atomic<bool>     g_written {false};
std::atomic<bool>     g_partial {false};
std::atomic<uint64_t> g_run_nonce {0};
std::atomic<uint64_t> g_last_submission {0};

} // namespace

void Configure(bool report_enabled, bool cold_enabled) {
	Detail::g_report_enabled.store(report_enabled, std::memory_order_relaxed);
	Detail::g_cold_enabled.store(cold_enabled, std::memory_order_relaxed);
	if (cold_enabled) {
		// Random and time based, so no other run repeats it and no cache can have seen it.
		const uint64_t seed = (static_cast<uint64_t>(std::random_device {}()) << 32u) ^
		                      std::random_device {}() ^
		                      static_cast<uint64_t>(
		                          std::chrono::steady_clock::now().time_since_epoch().count());
		g_run_nonce.store(Mix(seed), std::memory_order_relaxed);
	}
}

void Report::SetPhase(Phase phase) {
	std::lock_guard lock(m_mutex);
	m_phase = phase;
}

void Report::FlushPendingPauseLocked(int64_t current_flip) {
	if (m_has_pending_pause && m_pending_pause.flip != current_flip) {
		m_pauses.push_back(m_pending_pause);
		m_has_pending_pause = false;
	}
}

void Report::OnFlip(int64_t flip) {
	std::lock_guard lock(m_mutex);
	m_flip = flip;
	m_flips++;
	m_submissions_per_flip.push_back(m_submissions_this_flip);
	m_submissions_this_flip = 0;
	m_deferred_resolves_per_flip.push_back(m_deferred_resolves_this_flip);
	m_deferred_resolves_this_flip = 0;
	// Blocking in an earlier flip is over once the game moves on, so its pause is complete.
	FlushPendingPauseLocked(flip);
}

void Report::SetBoot(const BootInfo& boot) {
	std::lock_guard lock(m_mutex);
	m_boot = boot;
}

BootInfo Report::Boot() const {
	std::lock_guard lock(m_mutex);
	return m_boot;
}

void Report::SetPipelineLibrary(const PipelineLibraryInfo& info) {
	std::lock_guard lock(m_mutex);
	m_pipeline_library = info;
}

PipelineLibraryInfo Report::PipelineLibrary() const {
	std::lock_guard lock(m_mutex);
	return m_pipeline_library;
}

void Report::AddLibraryBuilt(LibraryKind kind, double ms) {
	std::lock_guard lock(m_mutex);
	auto& stats = m_library_stats.libraries[static_cast<size_t>(kind)];
	stats.built++;
	stats.ms += ms;
}

void Report::AddLibraryHit(LibraryKind kind) {
	std::lock_guard lock(m_mutex);
	m_library_stats.libraries[static_cast<size_t>(kind)].hits++;
}

void Report::AddLibraryFailed(LibraryKind kind) {
	std::lock_guard lock(m_mutex);
	m_library_stats.libraries[static_cast<size_t>(kind)].failed++;
}

void Report::AddLibraryRefusedHit(LibraryKind kind) {
	std::lock_guard lock(m_mutex);
	m_library_stats.libraries[static_cast<size_t>(kind)].refused_hits++;
}

void Report::AddLibraryLink(double ms) {
	std::lock_guard lock(m_mutex);
	m_library_stats.links++;
	m_library_stats.link_ms_total += ms;
	m_library_stats.link_ms_max = std::max(m_library_stats.link_ms_max, ms);
}

void Report::AddLibraryPipeline(const LibraryPipelineRow& row) {
	std::lock_guard lock(m_mutex);
	m_library_pipelines.push_back(row);
	m_library_pipelines.back().flip = m_flip;
}

std::vector<LibraryPipelineRow> Report::LibraryPipelines() const {
	std::lock_guard lock(m_mutex);
	return m_library_pipelines;
}

void Report::AddLibraryFallback(FallbackKind kind) {
	std::lock_guard lock(m_mutex);
	m_library_stats.fallback_full[static_cast<size_t>(kind)]++;
}

PipelineLibraryStats Report::LibraryStats() const {
	std::lock_guard lock(m_mutex);
	return m_library_stats;
}

void Report::AddBlockedLocked(Kind kind, double ms) {
	m_blocked_ms[static_cast<size_t>(kind)] += ms;
	FlushPendingPauseLocked(m_flip);
	if (!m_has_pending_pause) {
		m_pending_pause     = {.flip = m_flip};
		m_has_pending_pause = true;
	}
	m_pending_pause.ms += ms;
	m_pending_pause.events++;
}

void Report::AddTranslate(Kind kind, std::string_view stage, uint64_t hash, double ms,
                          Thread thread) {
	std::lock_guard lock(m_mutex);
	m_translate_ms += ms;
	m_translates.push_back({.stage = std::string(stage), .hash = hash, .ms = ms});
	// A worker thread leaves the GPU thread running, so only its own time is blocked.
	if (thread == Thread::Gpu && m_phase == Phase::Play) {
		AddBlockedLocked(kind, ms);
		if (kind == Kind::Graphics) {
			m_translate_gpu_graphics_ms += ms;
		}
	}
}

void Report::BeginSubmission(const SubmissionStamp& stamp, bool count) {
	std::lock_guard lock(m_mutex);
	m_stamp = stamp;
	if (count) {
		m_submissions++;
		m_submissions_this_flip++;
	}
}

void Report::EndSubmission() {
	std::lock_guard lock(m_mutex);
	m_stamp.id = NO_SUBMISSION;
}

void Report::SetEpoch(int64_t epoch_ns) {
	std::lock_guard lock(m_mutex);
	m_epoch_ns = epoch_ns;
}

int64_t Report::EpochNs() const {
	std::lock_guard lock(m_mutex);
	return m_epoch_ns;
}

void Report::AddCompile(Kind kind, uint64_t id, uint64_t id_ps, double create_ms, Thread thread,
                        int64_t start_ns, CompileContext context) {
	std::lock_guard lock(m_mutex);
	CompileEntry    entry {.kind      = kind,
	                       .id        = id,
	                       .id_ps     = id_ps,
	                       .create_ms = create_ms,
	                       .phase     = m_phase,
	                       .thread    = thread,
	                       .flip      = m_flip};
	// A worker row shows when it began and what it met, which the GPU rows get from their
	// submission.
	if (thread == Thread::Worker) {
		entry.at_ns        = start_ns;
		entry.warm_running = context.warm_running;
		entry.warm_front   = context.warm_front;
		entry.warm_back    = context.warm_back;
		entry.child_alive  = context.child_alive;
	}
	// Only the GPU thread works through submissions, so a worker row never takes the current one.
	if (thread == Thread::Gpu && m_phase == Phase::Play && m_stamp.id != NO_SUBMISSION &&
	    start_ns != NO_TIME_NS) {
		entry.submission = m_stamp.id;
		entry.queue      = m_stamp.queue;
		entry.start_ns   = start_ns;
		entry.visible_ms = static_cast<double>(start_ns - m_stamp.enqueued_ns) / NS_PER_MS;
		entry.picked_ms  = static_cast<double>(start_ns - m_stamp.picked_ns) / NS_PER_MS;
	}
	m_compiles.push_back(entry);
	if (thread == Thread::Gpu && m_phase == Phase::Play) {
		AddBlockedLocked(kind, create_ms);
	}
}

void Report::AddAudioUnderrun() noexcept {
	// The count is read before the increment, so a long run of underruns past the capacity does not
	// keep moving the index.
	if (m_underrun_next.load(std::memory_order_relaxed) >= AUDIO_UNDERRUN_CAPACITY) {
		m_underrun_overflow.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	const size_t slot = m_underrun_next.fetch_add(1, std::memory_order_relaxed);
	if (slot >= AUDIO_UNDERRUN_CAPACITY) {
		// Several calls passed the check together and took the last slots.
		m_underrun_overflow.fetch_add(1, std::memory_order_relaxed);
		return;
	}
	// The counts go in first and the time publishes the slot with release, so a reader that sees
	// the time also sees them. All of it stays lock-free, with no allocation and no logging.
	constexpr uint32_t count_limit = std::numeric_limits<uint16_t>::max();
	m_underrun_warm_front[slot].store(
	    static_cast<uint16_t>(std::min(WarmFrontBuildsRunning(), count_limit)),
	    std::memory_order_relaxed);
	m_underrun_warm_back[slot].store(
	    static_cast<uint16_t>(std::min(WarmBackBuildsRunning(), count_limit)),
	    std::memory_order_relaxed);
	m_underrun_child_alive[slot].store(GeneratorChildAlive() ? 1 : 0, std::memory_order_relaxed);
	m_underrun_prescan_running[slot].store(
	    static_cast<uint16_t>(std::min(PrescanRunning(), count_limit)), std::memory_order_relaxed);
	m_underrun_ns[slot].store(NowNs(), std::memory_order_release);
}

AudioUnderruns Report::AudioUnderrunsSoFar() const {
	AudioUnderruns result;
	result.overflow = m_underrun_overflow.load(std::memory_order_relaxed);
	const size_t taken =
	    std::min(m_underrun_next.load(std::memory_order_relaxed), AUDIO_UNDERRUN_CAPACITY);
	for (size_t slot = 0; slot < taken; slot++) {
		// Acquire pairs with the release store of the time, which follows the counts.
		const auto time = m_underrun_ns[slot].load(std::memory_order_acquire);
		if (time != 0) {
			result.at_ns.push_back(time);
			result.warm_front.push_back(
			    m_underrun_warm_front[slot].load(std::memory_order_relaxed));
			result.warm_back.push_back(m_underrun_warm_back[slot].load(std::memory_order_relaxed));
			result.child_alive.push_back(
			    m_underrun_child_alive[slot].load(std::memory_order_relaxed));
			result.prescan_running.push_back(
			    m_underrun_prescan_running[slot].load(std::memory_order_relaxed));
		}
	}
	return result;
}

void Report::AddWait(Kind kind, double ms) {
	std::lock_guard lock(m_mutex);
	// The compile itself is recorded by the worker that ran it, as worker time.
	if (m_phase == Phase::Play) {
		AddBlockedLocked(kind, ms);
	}
}

void Report::AddReady(Kind kind, uint64_t id, uint64_t id_ps, double ready_ms,
                      uint32_t skipped_draws) {
	std::lock_guard lock(m_mutex);
	m_readies.push_back({.kind           = kind,
	                     .id             = id,
	                     .id_ps          = id_ps,
	                     .ready_ms       = ready_ms,
	                     .skipped_draws  = skipped_draws});
}

void Report::AddSkip(SkipKind kind, std::string_view target_class) {
	std::lock_guard lock(m_mutex);
	m_skips[static_cast<size_t>(kind)]++;
	const auto found = std::ranges::find(m_skips_by_target, target_class,
	                                     &std::pair<std::string, uint32_t>::first);
	if (found == m_skips_by_target.end()) {
		m_skips_by_target.emplace_back(std::string(target_class), 1);
	} else {
		found->second++;
	}
}

void Report::AddSampledAfterSkip(std::string_view target_class, uint32_t draws) {
	std::lock_guard lock(m_mutex);
	const auto found = std::ranges::find(m_sampled_after_skip, target_class,
	                                     &std::pair<std::string, uint32_t>::first);
	if (found == m_sampled_after_skip.end()) {
		m_sampled_after_skip.emplace_back(std::string(target_class), draws);
	} else {
		found->second += draws;
	}
}

void Report::AddViewHistoryCount(ViewHistoryCount which) {
	std::lock_guard lock(m_mutex);
	m_view_history[static_cast<size_t>(which)]++;
}

void Report::AddColorCompletedFrames(uint32_t targets) {
	std::lock_guard lock(m_mutex);
	m_color_completed_frames += targets;
}

void Report::AddRefusal(std::string_view reason, double blocked_ms) {
	std::lock_guard lock(m_mutex);
	const auto found = std::ranges::find(m_refusals, reason, &Refusal::reason);
	if (found == m_refusals.end()) {
		m_refusals.push_back({std::string(reason), 1, blocked_ms});
	} else {
		found->draws++;
		found->blocked_ms += blocked_ms;
	}
}

void Report::AddDeferredResolve(const DeferredResolve& resolve) {
	std::lock_guard lock(m_mutex);
	m_deferred_resolves.push_back(resolve);
	m_deferred_resolves.back().flip       = m_flip;
	m_deferred_resolves.back().submission = m_stamp.id;
	m_deferred_resolves_this_flip++;
}

void Report::AddDeferredIneligible(std::string_view reason) {
	std::lock_guard lock(m_mutex);
	const auto found = std::ranges::find(m_deferred_ineligible, reason,
	                                     &std::pair<std::string, uint32_t>::first);
	if (found == m_deferred_ineligible.end()) {
		m_deferred_ineligible.emplace_back(std::string(reason), 1);
	} else {
		found->second++;
	}
}

void Report::AddDeferredIneligibleWait(const IneligibleWait& wait) {
	std::lock_guard lock(m_mutex);
	m_deferred_ineligible_waits.push_back(wait);
	m_deferred_ineligible_waits.back().flip = m_flip;
	const auto found = std::ranges::find(m_deferred_ineligible_waited_ms, wait.reason,
	                                     &std::pair<std::string, double>::first);
	if (found == m_deferred_ineligible_waited_ms.end()) {
		m_deferred_ineligible_waited_ms.emplace_back(wait.reason, wait.waited_ms);
	} else {
		found->second += wait.waited_ms;
	}
}

void Report::AddDeferredWouldSkip(double waited_ms) {
	std::lock_guard lock(m_mutex);
	m_deferred_would_skip_draws++;
	m_deferred_would_skip_waited_ms += waited_ms;
}

// A resolve has a handful of verdicts at most, so a linear search keeps them in first-met order.
void AddHoleVerdicts(DeferredResolve& row, std::span<const HoleVerdict> holes) {
	for (const auto& hole: holes) {
		const std::string_view name = hole.verdict != nullptr ? std::string_view(hole.verdict) : HOLE_VERDICT_UNKNOWN;
		auto found = std::ranges::find_if(row.refused, [&](const auto& entry) { return entry.first == name; });
		if (found == row.refused.end()) {
			row.refused.emplace_back(std::string(name), 0);
			found = std::prev(row.refused.end());
		}
		found->second++;
		if (hole.critical && row.critical_refusal.empty()) {
			row.critical_refusal = std::string(name);
			// From the same hole as the refusal, so the targets always explain the named rule.
			if (hole.targets != nullptr) {
				row.critical_targets = *hole.targets;
			}
		}
	}
}

void Report::AddPrescanMissRow(const PrescanMissRow& row) {
	std::lock_guard lock(m_mutex);
	if (m_prescan_miss_rows.size() >= PRESCAN_MISS_ROW_LIMIT) {
		m_prescan_miss_rows_dropped++;
		return;
	}
	m_prescan_miss_rows.push_back(row);
	m_prescan_miss_rows.back().flip = m_flip;
}

std::vector<PrescanMissRow> Report::PrescanMissRows() const {
	std::lock_guard lock(m_mutex);
	return m_prescan_miss_rows;
}

uint64_t Report::PrescanMissRowsDropped() const {
	std::lock_guard lock(m_mutex);
	return m_prescan_miss_rows_dropped;
}

void Report::AddDeferredRefusedSkip(std::string_view reason, double waited_ms) {
	std::lock_guard lock(m_mutex);
	auto            found = m_deferred_refused_skip.find(reason);
	if (found == m_deferred_refused_skip.end()) {
		found = m_deferred_refused_skip.emplace(std::string(reason), DeferredRefusal {}).first;
	}
	found->second.draws++;
	found->second.waited_ms += waited_ms;
}

void Report::AddDeferredWarmInline(bool over_hold_threshold) {
	std::lock_guard lock(m_mutex);
	m_deferred_warm_inline++;
	if (over_hold_threshold) {
		m_deferred_warm_inline_over++;
	}
}

void Report::AddWarmCompute(bool waited, double ms, bool over_hold_threshold) {
	std::lock_guard lock(m_mutex);
	if (waited) {
		m_warm_waited_count++;
		m_warm_waited_ms += ms;
		m_warm_waited_over += over_hold_threshold ? 1u : 0u;
	} else {
		m_warm_inline_total++;
		m_warm_inline_total_ms += ms;
		m_warm_inline_total_over += over_hold_threshold ? 1u : 0u;
	}
}

void Report::AddWarmTake(const WarmTake& take) {
	std::lock_guard lock(m_mutex);
	m_warm_takes.push_back(take);
	m_warm_takes.back().flip = m_flip;
}

double Report::BlockedMs(Kind kind) const {
	std::lock_guard lock(m_mutex);
	return m_blocked_ms[static_cast<size_t>(kind)];
}

void Report::AddStampFailure() {
	std::lock_guard lock(m_mutex);
	m_stamp_failures++;
}

void Report::SetGenerationInactive(std::string reason) {
	std::lock_guard lock(m_generation_mutex);
	m_generation_inactive = std::move(reason);
}

void Report::SetGenerationSource(std::function<ShaderStoreGenerationReport()> source) {
	std::lock_guard lock(m_generation_mutex);
	m_generation_source = std::move(source);
	m_generation_final.reset();
}

void Report::SetGenerationFinal(ShaderStoreGenerationReport final_figures) {
	std::lock_guard lock(m_generation_mutex);
	m_generation_source = nullptr;
	m_generation_final  = std::move(final_figures);
}

void Report::SetScanDifferentialSource(std::function<ScanDifferentialReport()> source) {
	std::lock_guard lock(m_scan_differential_mutex);
	m_scan_differential_source = std::move(source);
	m_scan_differential_final.reset();
}

void Report::SetScanDifferentialFinal(ScanDifferentialReport final_figures) {
	std::lock_guard lock(m_scan_differential_mutex);
	m_scan_differential_source = nullptr;
	m_scan_differential_final  = std::move(final_figures);
}

void Report::SetPrescanScanSource(std::function<PrescanScannerReport()> source) {
	std::lock_guard lock(m_prescan_scan_mutex);
	m_prescan_scan_source = std::move(source);
	m_prescan_scan_final.reset();
}

void Report::SetPrescanScanFinal(PrescanScannerReport final_figures) {
	std::lock_guard lock(m_prescan_scan_mutex);
	m_prescan_scan_source = nullptr;
	m_prescan_scan_final  = std::move(final_figures);
}

std::optional<PrescanScannerReport> Report::PrescanScanFigures() const {
	std::lock_guard lock(m_prescan_scan_mutex);
	if (m_prescan_scan_source) {
		return m_prescan_scan_source();
	}
	return m_prescan_scan_final;
}

bool Report::HoldsPrescanScanSource() const {
	std::lock_guard lock(m_prescan_scan_mutex);
	return static_cast<bool>(m_prescan_scan_source);
}

bool Report::HoldsPrescanScanFinal() const {
	std::lock_guard lock(m_prescan_scan_mutex);
	return m_prescan_scan_final.has_value();
}

void Report::SetPrescanCacheSource(std::function<PrescanCacheReport()> source) {
	std::lock_guard lock(m_prescan_cache_mutex);
	m_prescan_cache_source = std::move(source);
	m_prescan_cache_final.reset();
}

void Report::SetPrescanCacheFinal(PrescanCacheReport final_figures) {
	std::lock_guard lock(m_prescan_cache_mutex);
	m_prescan_cache_source = nullptr;
	m_prescan_cache_final  = std::move(final_figures);
}

std::optional<PrescanCacheReport> Report::PrescanCacheFigures() const {
	std::lock_guard lock(m_prescan_cache_mutex);
	if (m_prescan_cache_source) {
		return m_prescan_cache_source();
	}
	return m_prescan_cache_final;
}

bool Report::HoldsPrescanCacheSource() const {
	std::lock_guard lock(m_prescan_cache_mutex);
	return static_cast<bool>(m_prescan_cache_source);
}

bool Report::HoldsPrescanCacheFinal() const {
	std::lock_guard lock(m_prescan_cache_mutex);
	return m_prescan_cache_final.has_value();
}

void Report::SetGraphicsPrescanSource(GraphicsPrescanPart part,
                                      std::function<PrescanGraphicsReport()> source) {
	const auto      index = static_cast<size_t>(part);
	std::lock_guard lock(m_graphics_prescan_mutex);
	m_graphics_prescan_source[index] = std::move(source);
	m_graphics_prescan_final[index].reset();
}

void Report::SetGraphicsPrescanFinal(GraphicsPrescanPart part, PrescanGraphicsReport final_figures) {
	const auto      index = static_cast<size_t>(part);
	std::lock_guard lock(m_graphics_prescan_mutex);
	m_graphics_prescan_source[index] = nullptr;
	m_graphics_prescan_final[index]  = std::move(final_figures);
}

bool Report::HoldsGraphicsPrescanSource(GraphicsPrescanPart part) const {
	std::lock_guard lock(m_graphics_prescan_mutex);
	return static_cast<bool>(m_graphics_prescan_source[static_cast<size_t>(part)]);
}

std::optional<PrescanGraphicsReport> Report::GraphicsPrescanFigures() const {
	std::lock_guard                      lock(m_graphics_prescan_mutex);
	std::optional<PrescanGraphicsReport> merged;
	for (size_t part = 0; part < GRAPHICS_PRESCAN_PART_COUNT; part++) {
		std::optional<PrescanGraphicsReport> figures;
		if (m_graphics_prescan_source[part]) {
			figures = m_graphics_prescan_source[part]();
		} else {
			figures = m_graphics_prescan_final[part];
		}
		if (!figures) {
			continue;
		}
		if (!merged) {
			merged.emplace();
		}
		for (auto& group: figures->groups) {
			merged->groups.push_back(std::move(group));
		}
	}
	return merged;
}

// Each pair is cleared under its own lock, one lock at a time, as every writer of it takes them.
void Report::ResetSources() {
	{
		std::lock_guard lock(m_scan_differential_mutex);
		m_scan_differential_source = nullptr;
		m_scan_differential_final.reset();
	}
	{
		std::lock_guard lock(m_prescan_scan_mutex);
		m_prescan_scan_source = nullptr;
		m_prescan_scan_final.reset();
	}
	{
		std::lock_guard lock(m_prescan_cache_mutex);
		m_prescan_cache_source = nullptr;
		m_prescan_cache_final.reset();
	}
	{
		std::lock_guard lock(m_graphics_prescan_mutex);
		for (size_t part = 0; part < GRAPHICS_PRESCAN_PART_COUNT; part++) {
			m_graphics_prescan_source[part] = nullptr;
			m_graphics_prescan_final[part].reset();
		}
	}
	std::lock_guard lock(m_generation_mutex);
	m_generation_inactive.reset();
	m_generation_source = nullptr;
	m_generation_final.reset();
}

void Report::Reset() {
	ResetSources();
	std::lock_guard lock(m_mutex);
	m_phase             = Phase::Play;
	m_flip              = 0;
	m_flips             = 0;
	m_stamp             = {};
	m_epoch_ns          = NO_TIME_NS;
	m_submissions       = 0;
	m_submissions_this_flip = 0;
	m_submissions_per_flip.clear();
	m_stamp_failures    = 0;
	m_boot              = {};
	m_pipeline_library  = {};
	m_library_stats     = {};
	m_library_pipelines.clear();
	m_blocked_ms        = {};
	m_translate_ms      = 0.0;
	m_translate_gpu_graphics_ms = 0.0;
	m_translates.clear();
	m_compiles.clear();
	m_readies.clear();
	m_pauses.clear();
	m_has_pending_pause = false;
	m_pending_pause     = {};
	m_skips             = {};
	m_skips_by_target.clear();
	m_sampled_after_skip.clear();
	m_view_history = {};
	m_color_completed_frames = 0;
	m_refusals.clear();
	m_deferred_resolves.clear();
	m_deferred_resolves_per_flip.clear();
	m_deferred_resolves_this_flip   = 0;
	m_deferred_ineligible.clear();
	m_deferred_ineligible_waits.clear();
	m_prescan_miss_rows.clear();
	m_prescan_miss_rows_dropped = 0;
	m_deferred_ineligible_waited_ms.clear();
	m_deferred_would_skip_draws     = 0;
	m_deferred_would_skip_waited_ms = 0.0;
	m_deferred_refused_skip.clear();
	m_deferred_warm_inline          = 0;
	m_deferred_warm_inline_over     = 0;
	m_warm_inline_total             = 0;
	m_warm_inline_total_ms          = 0.0;
	m_warm_inline_total_over        = 0;
	m_warm_waited_count             = 0;
	m_warm_waited_ms                = 0.0;
	m_warm_waited_over              = 0;
	m_warm_takes.clear();
	for (auto& slot: m_underrun_ns) {
		slot.store(0, std::memory_order_relaxed);
	}
	for (size_t slot = 0; slot < AUDIO_UNDERRUN_CAPACITY; slot++) {
		m_underrun_warm_front[slot].store(0, std::memory_order_relaxed);
		m_underrun_warm_back[slot].store(0, std::memory_order_relaxed);
		m_underrun_child_alive[slot].store(0, std::memory_order_relaxed);
		m_underrun_prescan_running[slot].store(0, std::memory_order_relaxed);
	}
	m_underrun_next.store(0, std::memory_order_relaxed);
	m_underrun_overflow.store(0, std::memory_order_relaxed);
}

std::vector<Pause> Report::AllPausesLocked() const {
	auto pauses = m_pauses;
	if (m_has_pending_pause) {
		pauses.push_back(m_pending_pause);
	}
	return pauses;
}

std::vector<CompileEntry> Report::Compiles() const {
	std::lock_guard lock(m_mutex);
	return m_compiles;
}

std::vector<ReadyEntry> Report::Readies() const {
	std::lock_guard lock(m_mutex);
	return m_readies;
}

std::vector<WarmTake> Report::WarmTakes() const {
	std::lock_guard lock(m_mutex);
	return m_warm_takes;
}

namespace {

// Index of the histogram bucket a pause falls in, or the bucket count for one under the first
// bound. The highest bound the pause reaches picks the bucket.
size_t PauseBucket(double ms) {
	size_t bucket = PAUSE_BUCKET_BOUNDS_MS.size();
	for (size_t index = 0; index < PAUSE_BUCKET_BOUNDS_MS.size(); index++) {
		if (ms >= static_cast<double>(PAUSE_BUCKET_BOUNDS_MS[index])) {
			bucket = index;
		}
	}
	return bucket;
}

// Percentiles are whole numbers out of this scale.
constexpr uint32_t PERCENT_SCALE = 100;

// Percentiles that the submit lead summary reports.
constexpr uint32_t SUBMIT_LEAD_P10 = 10;
constexpr uint32_t SUBMIT_LEAD_P50 = 50;
constexpr uint32_t SUBMIT_LEAD_P90 = 90;

// The nearest-rank percentile of an ascending list that is not empty. The rank is the smallest
// 1-based position that covers the percentile of the samples, so p50 of four values is the second.
// The rank is clamped to the list, so a percentile above 100 reads the last value.
double NearestRank(const std::vector<double>& ascending, uint32_t percentile) {
	const size_t count = ascending.size();
	const size_t rank  = (count * percentile + PERCENT_SCALE - 1) / PERCENT_SCALE;
	return ascending[std::clamp<size_t>(rank, 1, count) - 1];
}

// Summarizes the lead of the compute compiles of guest shaders that ran inside a submission in
// play. The lead is the row's visible_ms, which AddCompile took from the stamp of the submission
// being processed at the time, so a later submission cannot change it. AddCompile gives only
// GPU-thread rows in play a submission, so the submission test also leaves out worker and warm-up
// rows. The renderer's own helper pipelines are left out by id, since no guest dispatch waits on them.
SubmitLeadSummary SummarizeSubmitLeads(const std::vector<CompileEntry>& compiles) {
	std::vector<double> leads;
	for (const auto& compile: compiles) {
		if (compile.kind == Kind::Compute && compile.submission != NO_SUBMISSION &&
		    !IsHelperId(compile.id)) {
			leads.push_back(compile.visible_ms);
		}
	}
	SubmitLeadSummary summary;
	if (leads.empty()) {
		return summary;
	}
	std::ranges::sort(leads);
	summary.n      = static_cast<uint32_t>(leads.size());
	summary.p10_ms = NearestRank(leads, SUBMIT_LEAD_P10);
	summary.p50_ms = NearestRank(leads, SUBMIT_LEAD_P50);
	summary.p90_ms = NearestRank(leads, SUBMIT_LEAD_P90);
	summary.max_ms = leads.back();
	return summary;
}

} // namespace

void Report::CopyDeferredLocked(Summary& summary) const {
	summary.deferred_resolves          = m_deferred_resolves;
	summary.deferred_resolves_per_flip = m_deferred_resolves_per_flip;
	// The flip still running has not been pushed by OnFlip, so its resolves are added last.
	if (m_deferred_resolves_this_flip != 0) {
		summary.deferred_resolves_per_flip.push_back(m_deferred_resolves_this_flip);
	}
	summary.deferred_ineligible                      = m_deferred_ineligible;
	summary.deferred_ineligible_waits                = m_deferred_ineligible_waits;
	summary.deferred_ineligible_waited_ms            = m_deferred_ineligible_waited_ms;
	summary.deferred_would_skip_draws                = m_deferred_would_skip_draws;
	summary.deferred_would_skip_waited_ms            = m_deferred_would_skip_waited_ms;
	summary.deferred_refused_skip                    = m_deferred_refused_skip;
	summary.deferred_warm_inline                     = m_deferred_warm_inline;
	summary.deferred_warm_inline_over_hold_threshold = m_deferred_warm_inline_over;
	summary.warm_inline_total                        = m_warm_inline_total;
	summary.warm_inline_total_ms                     = m_warm_inline_total_ms;
	summary.warm_inline_total_over_hold_threshold    = m_warm_inline_total_over;
	summary.warm_waited_count                        = m_warm_waited_count;
	summary.warm_waited_ms                           = m_warm_waited_ms;
	summary.warm_waited_over_hold_threshold          = m_warm_waited_over;
}

Summary Report::Summarize() const {
	std::lock_guard lock(m_mutex);
	Summary         summary;
	summary.blocked_graphics_ms = m_blocked_ms[static_cast<size_t>(Kind::Graphics)];
	summary.blocked_compute_ms  = m_blocked_ms[static_cast<size_t>(Kind::Compute)];
	summary.translate_ms        = m_translate_ms;
	summary.translate_gpu_graphics_ms = m_translate_gpu_graphics_ms;
	summary.flips               = m_flips;
	summary.submissions         = m_submissions;
	summary.submissions_per_flip = m_submissions_per_flip;
	summary.stamp_failures      = m_stamp_failures;
	summary.skips               = m_skips;
	summary.skips_by_target     = m_skips_by_target;
	summary.sampled_after_skip_by_target = m_sampled_after_skip;
	summary.view_history                 = m_view_history;
	summary.color_completed_frames       = m_color_completed_frames;
	summary.refusals                     = m_refusals;
	CopyDeferredLocked(summary);
	summary.submit_leads = SummarizeSubmitLeads(m_compiles);

	summary.worst_translate = m_translates;
	std::ranges::stable_sort(summary.worst_translate,
	                         [](const auto& left, const auto& right) { return left.ms > right.ms; });
	summary.worst_translate.resize(std::min(summary.worst_translate.size(), WORST_TRANSLATE_LIMIT));

	auto pauses         = AllPausesLocked();
	summary.pause_count = static_cast<uint32_t>(pauses.size());
	for (const auto& pause: pauses) {
		summary.longest_pause_ms = std::max(summary.longest_pause_ms, pause.ms);
		const size_t bucket = PauseBucket(pause.ms);
		if (bucket == PAUSE_BUCKET_BOUNDS_MS.size()) {
			summary.pauses_under_first_bound++;
		} else {
			summary.pause_histogram[bucket]++;
		}
	}
	std::ranges::stable_sort(pauses,
	                         [](const auto& left, const auto& right) { return left.ms > right.ms; });
	pauses.resize(std::min(pauses.size(), LONGEST_PAUSE_LIMIT));
	summary.longest_pauses = std::move(pauses);
	return summary;
}

namespace {

nlohmann::json CompileToJson(const CompileEntry& compile, int64_t epoch_ns) {
	nlohmann::json row = {{"kind", compile.kind == Kind::Graphics ? "graphics" : "compute"}};
	if (compile.kind == Kind::Graphics) {
		row["vs"] = Hex(compile.id);
		row["ps"] = Hex(compile.id_ps);
	} else {
		row["id"] = Hex(compile.id);
	}
	row["create_ms"] = Round3(compile.create_ms);
	row["phase"]     = compile.phase == Phase::Boot ? "boot" : "play";
	row["thread"]    = compile.thread == Thread::Gpu ? "gpu" : "worker";
	row["flip"]      = compile.flip;
	// A worker row shows when it began and what it met. The other rows get their time from the
	// submission below, and are left unchanged.
	if (compile.thread == Thread::Worker && compile.at_ns != NO_TIME_NS) {
		const int64_t base  = epoch_ns == NO_TIME_NS ? 0 : epoch_ns;
		row["at_ms"]        = Round3(static_cast<double>(compile.at_ns - base) / NS_PER_MS);
		row["warm_running"] = compile.warm_running;
		row["warm_front"]   = compile.warm_front;
		row["warm_back"]    = compile.warm_back;
		row["child_alive"]  = compile.child_alive;
	}
	// Left out when unset, so older readers see no new keys.
	if (compile.submission != NO_SUBMISSION) {
		row["submission"] = compile.submission;
		row["queue"]      = compile.queue;
		row["visible_ms"] = Round3(compile.visible_ms);
		row["picked_ms"]  = Round3(compile.picked_ms);
		// With no epoch the time is on the raw steady clock, which only tests see.
		const int64_t base = epoch_ns == NO_TIME_NS ? 0 : epoch_ns;
		row["start_ms"]    = Round3(static_cast<double>(compile.start_ns - base) / NS_PER_MS);
	}
	return row;
}

// Writes one ready row.
nlohmann::json ReadyToJson(const ReadyEntry& ready) {
	return {{"kind", ready.kind == Kind::Graphics ? "graphics" : "compute"},
	        {"vs", Hex(ready.id)},
	        {"ps", Hex(ready.id_ps)},
	        {"ready_ms", Round3(ready.ready_ms)},
	        {"skipped_draws", ready.skipped_draws}};
}

nlohmann::json HistogramToJson(const Summary& summary) {
	nlohmann::json histogram = {
	    {fmt::format("under_{}", PAUSE_BUCKET_BOUNDS_MS.front()), summary.pauses_under_first_bound}};
	for (size_t index = 0; index < PAUSE_BUCKET_BOUNDS_MS.size(); index++) {
		histogram[std::to_string(PAUSE_BUCKET_BOUNDS_MS[index])] = summary.pause_histogram[index];
	}
	return histogram;
}

nlohmann::json PlayToJson(const Summary& summary) {
	nlohmann::json worst = nlohmann::json::array();
	for (const auto& entry: summary.worst_translate) {
		worst.push_back({{"stage", entry.stage}, {"hash", Hex(entry.hash)}, {"ms", Round3(entry.ms)}});
	}
	const double blocked_total = summary.blocked_graphics_ms + summary.blocked_compute_ms;
	return {{"blocked_ms",
	         {{"graphics", Round3(summary.blocked_graphics_ms)},
	          {"compute", Round3(summary.blocked_compute_ms)},
	          {"total", Round3(blocked_total)}}},
	        {"translate_ms", Round3(summary.translate_ms)},
	        {"translate_gpu_graphics_ms", Round3(summary.translate_gpu_graphics_ms)},
	        {"worst_translate", worst}};
}

// Writes the submit lead summary. With no samples only n is written, since a zero percentile would
// read as a real lead of zero.
nlohmann::json SubmitLeadsToJson(const SubmitLeadSummary& leads) {
	if (leads.n == 0) {
		return {{"n", 0}};
	}
	return {{"n", leads.n},
	        {"p10_ms", Round3(leads.p10_ms)},
	        {"p50_ms", Round3(leads.p50_ms)},
	        {"p90_ms", Round3(leads.p90_ms)},
	        {"max_ms", Round3(leads.max_ms)}};
}

nlohmann::json PausesToJson(const Summary& summary) {
	nlohmann::json top = nlohmann::json::array();
	for (const auto& pause: summary.longest_pauses) {
		top.push_back({{"flip", pause.flip}, {"ms", Round3(pause.ms)}, {"contents", pause.events}});
	}
	return {{"count", summary.pause_count},
	        {"pause_histogram", HistogramToJson(summary)},
	        {"longest_pause_ms", Round3(summary.longest_pause_ms)},
	        {"top", top}};
}

// Writes the view-history counts, one key per ViewHistoryCount in its order.
nlohmann::json ViewHistoryToJson(const std::array<uint64_t, VIEW_HISTORY_COUNT>& counts) {
	constexpr std::array<std::string_view, VIEW_HISTORY_COUNT> NAMES {
	    "depth_layer", "depth_mip", "color_layer", "color_mip", "tables_full", "notes_not_fitted"};
	nlohmann::json result = nlohmann::json::object();
	for (size_t index = 0; index < VIEW_HISTORY_COUNT; index++) {
		result[std::string(NAMES[index])] = counts[index];
	}
	return result;
}

nlohmann::json SkipsToJson(const Summary& summary) {
	nlohmann::json by_target = nlohmann::json::object();
	for (const auto& [target, count]: summary.skips_by_target) {
		by_target[target] = count;
	}
	nlohmann::json refused = nlohmann::json::object();
	for (const auto& entry: summary.refusals) {
		refused[entry.reason] = {{"draws", entry.draws}, {"blocked_ms", Round3(entry.blocked_ms)}};
	}
	nlohmann::json sampled = nlohmann::json::object();
	for (const auto& [target, count]: summary.sampled_after_skip_by_target) {
		sampled[target] = count;
	}
	return {{"draw", summary.skips[static_cast<size_t>(SkipKind::Draw)]},
	        {"dispatch", summary.skips[static_cast<size_t>(SkipKind::Dispatch)]},
	        {"by_target_class", by_target},
	        {"sampled_after_skip_by_target_class", sampled},
	        {"refused", refused},
	        {"view_history", ViewHistoryToJson(summary.view_history)},
	        {"color_completed_frames", summary.color_completed_frames}};
}

// Writes one row per draw that could not be deferred and waited for its pipeline.
nlohmann::json IneligibleWaitsToJson(const std::vector<IneligibleWait>& waits) {
	nlohmann::json rows = nlohmann::json::array();
	for (const auto& wait: waits) {
		rows.push_back({{"reason", wait.reason},
		                {"flip", wait.flip},
		                {"vs", Hex(wait.vs)},
		                {"ps", Hex(wait.ps)},
		                {"waited_ms", Round3(wait.waited_ms)},
		                {"gpu_thread", wait.gpu_thread}});
	}
	return rows;
}

// Writes the deferred draws the S2 rules refused, one object per rule with its draws and waits.
nlohmann::json RefusedDeferredToJson(const std::map<std::string, DeferredRefusal, std::less<>>& refused) {
	nlohmann::json result = nlohmann::json::object();
	for (const auto& [reason, refusal]: refused) {
		result[reason] = {{"draws", refusal.draws}, {"waited_ms", Round3(refusal.waited_ms)}};
	}
	return result;
}

// Writes the waits of the draws that could not be deferred, summed per reason.
nlohmann::json IneligibleWaitedMsToJson(const std::vector<std::pair<std::string, double>>& totals) {
	nlohmann::json waited = nlohmann::json::object();
	for (const auto& [reason, ms]: totals) {
		waited[reason] = Round3(ms);
	}
	return waited;
}

// Writes a resolve's holes per skip verdict as one object, a verdict to its hole count.
nlohmann::json ResolveRefusedToJson(const std::vector<std::pair<std::string, uint32_t>>& refused) {
	nlohmann::json result = nlohmann::json::object();
	for (const auto& [verdict, holes]: refused) {
		result[verdict] = holes;
	}
	return result;
}

// Writes one refusing target of a resolve row, as SkipTargetFacts describes it.
nlohmann::json SkipTargetToJson(const SkipTargetFacts& facts) {
	return {{"width", facts.width},
	        {"height", facts.height},
	        {"layers", facts.layers},
	        {"levels", facts.levels},
	        {"format", facts.format},
	        {"view", facts.view},
	        {"history", facts.history},
	        {"draws_this_frame", facts.draws_this_frame},
	        {"draws_last_frame", facts.draws_last_frame},
	        {"draws_frame_before_last", facts.draws_frame_before_last},
	        {"cleared_this_frame", facts.cleared_this_frame},
	        {"cleared_last_frame", facts.cleared_last_frame},
	        {"latest_attached_ago", facts.latest_attached_ago},
	        {"previous_attached_ago", facts.previous_attached_ago}};
}

// Adds a resolve row's skip verdicts, its critical key's stage ids when it had one, its critical
// refusal and that hole's targets when it kept any.
void AddVerdictFields(nlohmann::json& row, const DeferredResolve& resolve) {
	if (resolve.has_critical) {
		row["critical_vs"] = Hex(resolve.critical_vs);
		row["critical_ps"] = Hex(resolve.critical_ps);
	}
	row["refused"]          = ResolveRefusedToJson(resolve.refused);
	row["critical_refusal"] = resolve.critical_refusal.empty() ? "none" : resolve.critical_refusal;
	// Only a row whose critical hole kept its targets has them, so other rows read as they did.
	if (resolve.critical_targets) {
		auto color    = SkipTargetToJson(resolve.critical_targets->color);
		color["slot"] = resolve.critical_targets->color_slot;
		row["critical_color_target"] = std::move(color);
		row["critical_depth_target"] = SkipTargetToJson(resolve.critical_targets->depth);
	}
}

// Writes the deferred-draw figures, which are empty or zero unless --deferred-pipeline-draws ran.
nlohmann::json DeferredToJson(const Summary& summary) {
	nlohmann::json resolves = nlohmann::json::array();
	for (const auto& resolve: summary.deferred_resolves) {
		nlohmann::json row = {{"flip", resolve.flip},
		                      {"trigger", resolve.trigger},
		                      {"holes", resolve.holes},
		                      {"distinct_keys", resolve.distinct_keys},
		                      {"resolve_wait_ms", Round3(resolve.resolve_wait_ms)},
		                      {"record_ms", Round3(resolve.record_ms)},
		                      {"pre_resolve_ms", Round3(resolve.pre_resolve_ms)},
		                      {"save_ms", Round3(resolve.save_ms)},
		                      {"hold_count_change", resolve.hold_count_change},
		                      {"window_hold_count_change", resolve.window_hold_count_change},
		                      {"held_ms", Round3(resolve.held_ms)},
		                      {"window_held_ms", Round3(resolve.window_held_ms)}};
		// Left out when unset, as in a compile row.
		if (resolve.submission != NO_SUBMISSION) {
			row["submission"] = resolve.submission;
		}
		AddVerdictFields(row, resolve);
		resolves.push_back(std::move(row));
	}
	nlohmann::json ineligible = nlohmann::json::object();
	for (const auto& [reason, count]: summary.deferred_ineligible) {
		ineligible[reason] = count;
	}
	return {{"resolves", resolves},
	        {"resolves_per_flip", summary.deferred_resolves_per_flip},
	        {"ineligible", ineligible},
	        {"ineligible_waited_ms", IneligibleWaitedMsToJson(summary.deferred_ineligible_waited_ms)},
	        {"ineligible_waits", IneligibleWaitsToJson(summary.deferred_ineligible_waits)},
	        {"s2_would_skip",
	         {{"draws", summary.deferred_would_skip_draws},
	          {"waited_ms", Round3(summary.deferred_would_skip_waited_ms)}}},
	        {"s2_refused_deferred", RefusedDeferredToJson(summary.deferred_refused_skip)},
	        {"warm_inline",
	         {{"during_pause", summary.deferred_warm_inline},
	          {"over_hold_threshold", summary.deferred_warm_inline_over_hold_threshold},
	          {"total", summary.warm_inline_total},
	          {"total_ms", Round3(summary.warm_inline_total_ms)},
	          {"total_over_hold_threshold", summary.warm_inline_total_over_hold_threshold}}},
	        {"warm_waited",
	         {{"count", summary.warm_waited_count},
	          {"ms", Round3(summary.warm_waited_ms)},
	          {"over_hold_threshold", summary.warm_waited_over_hold_threshold}}}};
}

// Writes one row per first use of a warm-up module, in the order they happened.
nlohmann::json WarmTakesToJson(const std::vector<WarmTake>& takes, int64_t epoch_ns) {
	nlohmann::json rows = nlohmann::json::array();
	for (const auto& take: takes) {
		const char* how = take.how == WarmTakeHow::Done     ? "done"
		                  : take.how == WarmTakeHow::Waited ? "waited"
		                                                    : "inline";
		nlohmann::json row = {{"id", Hex(take.id)},
		                      {"how", how},
		                      {"flip", take.flip},
		                      {"stopped_ms", Round3(take.stopped_ms)}};
		// Counted from the run start like a compile's start_ms, and left out when no time was given.
		if (take.at_ns != NO_TIME_NS) {
			const int64_t base = epoch_ns == NO_TIME_NS ? 0 : epoch_ns;
			row["at_ms"]       = Round3(static_cast<double>(take.at_ns - base) / NS_PER_MS);
		}
		rows.push_back(std::move(row));
	}
	return rows;
}

// A worker summary value is a number when it reads as one, and text otherwise.
nlohmann::json SummaryValueToJson(const std::string& name, const std::string& value) {
	// The worker writes hashes as 16 hex digits, and one made only of decimal digits must not turn
	// into a number, or the same key would change type between runs and lose its leading zeros.
	if (name == "order" || name.ends_with("_hash")) {
		return value;
	}
	int64_t     number = 0;
	const auto* end    = value.data() + value.size();
	const auto  parsed = std::from_chars(value.data(), end, number);
	if (parsed.ec == std::errc() && parsed.ptr == end) {
		return number;
	}
	return value;
}

nlohmann::json HashListToJson(const std::vector<uint64_t>& hashes) {
	nlohmann::json list = nlohmann::json::array();
	for (const auto hash: hashes) {
		list.push_back(Hex(hash));
	}
	return list;
}

nlohmann::json GenerationFailureToJson(const GenerationFailure& failure) {
	nlohmann::json row = {{"hash", Hex(failure.hash)},      {"rank", failure.rank},
	                      {"exit_code", failure.exit_code}, {"kind", failure.kind},
	                      {"last_line", failure.last_line}, {"spawn", failure.spawn},
	                      {"out_tail", failure.out_tail}};
	if (failure.cpu_ms) {
		row["cpu_ms"] = Round3(*failure.cpu_ms);
	}
	return row;
}

nlohmann::json GenerationSpawnUsageToJson(const GenerationSpawnUsage& usage) {
	nlohmann::json row = {{"spawn", usage.spawn}, {"cpu_ms", Round3(usage.cpu_ms)}};
	if (usage.peak_bytes) {
		row["peak_bytes"]  = *usage.peak_bytes;
		row["peak_source"] = usage.peak_source;
	}
	return row;
}

nlohmann::json GenerationDrainToJson(const GenerationDrain& drain) {
	return {{"calls", drain.calls},
	        {"over_1ms", drain.over_1ms},
	        {"over_4ms", drain.over_4ms},
	        {"max_ms", Round3(drain.max_ms)},
	        {"total_ms", Round3(drain.total_ms)},
	        {"lag_ms_max", Round3(drain.lag_ms_max)},
	        {"item_max_ms", Round3(drain.item_max_ms)},
	        {"adopted", drain.adopted},
	        {"duplicates", drain.duplicates},
	        {"plan_mismatch", drain.plan_mismatch},
	        {"plan_mismatch_reverse", drain.plan_mismatch_reverse},
	        {"failed_installs", drain.failed_installs},
	        {"dropped_stopped", drain.dropped_stopped},
	        {"posted", drain.posted},
	        {"handles_created", drain.handles_created},
	        {"handles_destroyed", drain.handles_destroyed},
	        {"bad_plans", drain.bad_plans},
	        {"undrained_at_stop", drain.undrained_at_stop},
	        {"pending", drain.pending}};
}

// Sets a key only when the figure is there, so a missing time is missing and not zero.
void SetOptional(nlohmann::json& object, const char* key, const std::optional<double>& value) {
	if (value) {
		object[key] = Round3(*value);
	}
}

// The supervisor's part of the block: what the child did, over its whole run.
nlohmann::json GenerationSupervisorToJson(const ShaderStoreGenerationReport& block) {
	nlohmann::json supervisor = {{"outcome", block.outcome},
	                             {"spawns", block.spawns},
	                             {"restarts", block.restarts},
	                             {"restarts_capped", block.outcome == "restarts_capped"},
	                             {"spawn_ms", Round3(block.spawn_ms)},
	                             {"decode_ms_max", Round3(block.decode_ms_max)},
	                             {"terminate_wait_ms", Round3(block.terminate_wait_ms)},
	                             {"frames_delivered", block.frames_delivered},
	                             {"bad_records", block.bad_records},
	                             {"duplicates", block.duplicates},
	                             {"posted", block.posted},
	                             {"predicted_programs", block.predicted_programs}};
	SetOptional(supervisor, "child_open_ms", block.child_open_ms);
	SetOptional(supervisor, "first_record_ms", block.first_record_ms);
	SetOptional(supervisor, "last_record_ms", block.last_record_ms);
	SetOptional(supervisor, "finished_ms", block.finished_ms);
	SetOptional(supervisor, "ended_ms", block.ended_ms);
	if (block.peak_bytes) {
		supervisor["peak_bytes"]  = *block.peak_bytes;
		supervisor["peak_source"] = block.peak_source;
	}
	supervisor["failures"] = nlohmann::json::array();
	for (const auto& failure: block.failures) {
		supervisor["failures"].push_back(GenerationFailureToJson(failure));
	}
	supervisor["spawn_usage"] = nlohmann::json::array();
	for (const auto& usage: block.spawn_usage) {
		supervisor["spawn_usage"].push_back(GenerationSpawnUsageToJson(usage));
	}
	supervisor["dropped"] = nlohmann::json::array();
	for (const auto& drop: block.dropped) {
		nlohmann::json row = {
		    {"hash", Hex(drop.hash)}, {"rank", drop.rank}, {"reason", drop.reason}};
		if (drop.module_id) {
			row["module_id"] = Hex(*drop.module_id);
		}
		supervisor["dropped"].push_back(std::move(row));
	}
	supervisor["failed_hashes"] = HashListToJson(block.failed_hashes);
	supervisor["skip_hashes"]   = HashListToJson(block.skip_hashes);
	return supervisor;
}

// A time on the NowNs clock as milliseconds from the run start, as a compile's start_ms is.
double RunMs(int64_t time_ns, int64_t epoch_ns) {
	const int64_t base = epoch_ns == NO_TIME_NS ? 0 : epoch_ns;
	return Round3(static_cast<double>(time_ns - base) / NS_PER_MS);
}

// Sets a key to a time from the run start, only when the time is there.
void SetRunMs(nlohmann::json& object, const char* key, const std::optional<int64_t>& time_ns,
              int64_t epoch_ns) {
	if (time_ns) {
		object[key] = RunMs(*time_ns, epoch_ns);
	}
}

nlohmann::json GeneratedModuleToJson(const GeneratedModuleRow& module, int64_t epoch_ns) {
	nlohmann::json row = {{"id", Hex(module.id)},
	                      {"code_hash", Hex(module.code_hash)},
	                      {"posted_ms", RunMs(module.posted_ns, epoch_ns)},
	                      {"adopted_ms", RunMs(module.adopted_ns, epoch_ns)},
	                      {"how", module.how},
	                      {"tier", module.tier},
	                      {"wide_tail", module.wide_tail}};
	// Left out when unknown, like the block's other optional figures, so a count of 0 stays apart.
	if (module.candidates) {
		row["candidates"] = *module.candidates;
	}
	SetRunMs(row, "ready_ms", module.ready_ns, epoch_ns);
	SetRunMs(row, "first_use_ms", module.first_use_ns, epoch_ns);
	SetOptional(row, "build_ms", module.build_ms);
	SetOptional(row, "lead_ms", module.lead_ms);
	return row;
}

nlohmann::json GenerationSlowTakeToJson(const GenerationSlowTake& take, int64_t epoch_ns) {
	nlohmann::json row = {{"id", Hex(take.id)},
	                      {"generated", take.generated},
	                      {"how", take.how},
	                      {"stopped_ms", Round3(take.stopped_ms)},
	                      {"paused", take.paused}};
	if (take.at_ns != NO_TIME_NS) {
		row["at_ms"] = RunMs(take.at_ns, epoch_ns);
	}
	SetRunMs(row, "posted_ms", take.posted_ns, epoch_ns);
	SetRunMs(row, "adopted_ms", take.adopted_ns, epoch_ns);
	SetRunMs(row, "ready_ms", take.ready_ns, epoch_ns);
	return row;
}

// One module the runtime needed before the generator delivered it, with both times on the run's
// clock.
nlohmann::json GenerationLateModuleToJson(const GenerationLateModule& late, int64_t epoch_ns) {
	nlohmann::json row = {{"id", Hex(late.id)},
	                      {"code_hash", Hex(late.code_hash)},
	                      {"miss_ms", RunMs(late.miss_ns, epoch_ns)},
	                      {"delivered_ms", RunMs(late.delivered_ns, epoch_ns)},
	                      {"via", late.via},
	                      {"create_ms", Round3(late.create_ms)},
	                      {"misses", late.misses}};
	SetRunMs(row, "adopted_ms", late.adopted_ns, epoch_ns);
	return row;
}

nlohmann::json GenerationMissesToJson(const GenerationMisses& misses, int64_t epoch_ns) {
	nlohmann::json classes = nlohmann::json::object();
	for (const auto& miss_class: misses.classes) {
		classes[miss_class.name] = {{"count", miss_class.count},
		                            {"create_ms", Round3(miss_class.create_ms)}};
	}
	nlohmann::json refused = nlohmann::json::object();
	for (const auto& [verdict, count]: misses.refused) {
		refused[verdict] = count;
	}
	nlohmann::json late_modules = nlohmann::json::array();
	for (const auto& late: misses.late_modules) {
		late_modules.push_back(GenerationLateModuleToJson(late, epoch_ns));
	}
	return {{"total", misses.total},
	        {"create_ms", Round3(misses.create_ms)},
	        {"classes", classes},
	        {"refused", refused},
	        {"wrong_spec_programs", misses.wrong_spec_programs},
	        {"wrong_spec_max_variants", misses.wrong_spec_max_variants},
	        {"late_modules", late_modules},
	        {"late_overflow", misses.late_overflow}};
}

// The rows of the block: what the worker said of each program, each module taken in, the misses and
// the slow first uses.
void GenerationRowsToJson(nlohmann::json& json, const ShaderStoreGenerationReport& block,
                          int64_t epoch_ns) {
	nlohmann::json predictions = nlohmann::json::array();
	for (const auto& prediction: block.predictions) {
		predictions.push_back({{"hash", Hex(prediction.hash)},
		                       {"outcome", prediction.outcome},
		                       {"candidates", prediction.candidates}});
	}
	json["predictions"]    = std::move(predictions);
	nlohmann::json modules = nlohmann::json::array();
	for (const auto& module: block.generated_modules) {
		modules.push_back(GeneratedModuleToJson(module, epoch_ns));
	}
	json["generated_modules"] = std::move(modules);
	json["unused_build_ms"]   = Round3(block.unused_build_ms);
	json["runtime_misses"]    = GenerationMissesToJson(block.misses, epoch_ns);
	nlohmann::json slow       = nlohmann::json::array();
	for (const auto& take: block.slow_takes) {
		slow.push_back(GenerationSlowTakeToJson(take, epoch_ns));
	}
	json["slow_takes"] = std::move(slow);
}

// The whole block of an active generation.
nlohmann::json GenerationToJson(const ShaderStoreGenerationReport& block, int64_t epoch_ns) {
	nlohmann::json json = {{"active", block.active}};
	if (!block.inactive_reason.empty()) {
		json["inactive_reason"] = block.inactive_reason;
	}
	json["persisted"] = block.persisted;
	if (!block.persist_error.empty()) {
		json["persist_error"] = block.persist_error;
	}
	json["source"]                   = block.source;
	json["file_bytes"]               = block.file_bytes;
	json["files_removed"]            = block.files_removed;
	json["remove_failures"]          = block.remove_failures;
	json["loader_abandoned"]         = block.loader_abandoned;
	json["regenerated_after_damage"] = block.regenerated_after_damage;
	SetOptional(json, "verify_ms", block.verify_ms);
	json["supervisor"] = GenerationSupervisorToJson(block);
	// The finish on the run's clock, so it can be set against an underrun's at_ms. finished_ms is a
	// duration from the supervisor thread's start and can't be. Written only for a finished run.
	SetRunMs(json["supervisor"], "finished_at_ms", block.finished_ns, epoch_ns);
	nlohmann::json worker = nlohmann::json::object();
	for (const auto& [name, value]: block.worker) {
		worker[name] = SummaryValueToJson(name, value);
	}
	json["worker"] = worker;
	json["drain"]  = GenerationDrainToJson(block.drain);
	// Only with the prescan on, so the block of a run without it is unchanged.
	if (block.drain.includes_prescan) {
		json["drain"]["includes_prescan"] = true;
	}
	GenerationRowsToJson(json, block, epoch_ns);
	return json;
}

// Writes one class of the scan differential as its certain and uncertain halves.
nlohmann::json ScanCountToJson(const ScanOutcomeCount& count) {
	return {{"certain", count.certain}, {"uncertain", count.uncertain}};
}

// Writes one example of a dispatch the scan got wrong, with the hashes as fixed width hex.
nlohmann::json ScanSampleToJson(const ScanDifferentialSample& sample) {
	return {{"kind", sample.kind},
	        {"queue", sample.queue},
	        {"seq", sample.seq},
	        {"program", Hex(sample.program)},
	        {"scanned_key", Hex(sample.scanned_key)},
	        {"executed_key", Hex(sample.executed_key)}};
}

// Writes the scan columns of one queue, with the causes of uncertainty as an object by name.
nlohmann::json ScanQueueToJson(const ScanQueueReport& queue) {
	nlohmann::json causes = nlohmann::json::object();
	for (const auto& [name, count]: queue.uncertain_cleared_by) {
		causes[name] = count;
	}
	return {{"queue", queue.queue},
	        {"submissions", queue.submissions},
	        {"cp_dispatches", queue.cp_dispatches},
	        {"dispatches", queue.dispatches},
	        {"uncertain", queue.uncertain},
	        {"branch_mode1", queue.branch_mode1},
	        {"branch_mode2", queue.branch_mode2},
	        {"branch_mode2_empty_else", queue.branch_mode2_empty_else},
	        {"uncertain_cleared_by", causes},
	        {"ib_revisit_doubts", queue.ib_revisit_doubts},
	        {"bad_packets", queue.bad_packets},
	        {"scan_allocations", queue.scan_allocations},
	        {"scan_p50_us", queue.scan_p50_us},
	        {"scan_p99_us", queue.scan_p99_us},
	        {"scan_max_us", queue.scan_max_us}};
}

// The whole scan_differential block. mismatch_rate is what the G0 gate reads.
nlohmann::json ScanDifferentialToJson(const ScanDifferentialReport& block) {
	nlohmann::json samples = nlohmann::json::array();
	for (const auto& sample: block.samples) {
		samples.push_back(ScanSampleToJson(sample));
	}
	nlohmann::json queues = nlohmann::json::array();
	for (const auto& queue: block.queues) {
		queues.push_back(ScanQueueToJson(queue));
	}
	const auto& totals = block.totals;
	return {{"cp_dispatches", totals.CpDispatches()},
	        {"matched", ScanCountToJson(totals.matched)},
	        {"key_mismatch", ScanCountToJson(totals.key_mismatch)},
	        {"missing", ScanCountToJson(totals.missing)},
	        {"extra", ScanCountToJson(totals.extra)},
	        {"mismatch_rate", totals.MismatchRate()},
	        {"mismatch_rate_certain", totals.CertainMismatchRate()},
	        {"untracked_cp_dispatches", block.untracked_cp_dispatches},
	        {"open_submissions", block.open_submissions},
	        {"samples", samples},
	        {"queues", queues}};
}

// Writes one of the prescan's times as its percentiles and maximum.
nlohmann::json PrescanTimeToJson(const PrescanTimeFigures& time) {
	return {{"p50", time.p50}, {"p90", time.p90}, {"p99", time.p99}, {"max", time.max}};
}

// Writes what the prescan's walks counted, with the causes of uncertainty as an object by name.
nlohmann::json PrescanScanToJson(const PrescanScanReport& scan) {
	nlohmann::json causes = nlohmann::json::object();
	for (const auto& [name, count]: scan.uncertain_cleared_by) {
		causes[name] = count;
	}
	return {{"submissions", scan.submissions},
	        {"dispatches", scan.dispatches},
	        {"no_shader_state", scan.no_shader_state},
	        {"program_inherited", scan.program_inherited},
	        {"uncertain_program", scan.uncertain_program},
	        {"ib_followed", scan.ib_followed},
	        {"ib_unreadable", scan.ib_unreadable},
	        {"ib_depth_limited", scan.ib_depth_limited},
	        {"ib_revisited", scan.ib_revisited},
	        {"budget_exhausted", scan.budget_exhausted},
	        {"bad_packets", scan.bad_packets},
	        {"resets", scan.resets},
	        {"dwords_read", scan.dwords_read},
	        {"dwords_max", scan.dwords_max},
	        {"dispatches_cached", scan.dispatches_cached},
	        {"time_capped", scan.time_capped},
	        {"scan_allocations", scan.scan_allocations},
	        {"cache_remembered_busy", scan.cache_remembered_busy},
	        {"depth_max", scan.depth_max},
	        {"register_list_unreadable", scan.register_list_unreadable},
	        {"branch_mode1", scan.branch_mode1},
	        {"branch_mode2", scan.branch_mode2},
	        {"branch_mode2_empty_else", scan.branch_mode2_empty_else},
	        {"ib_revisit_doubts", scan.ib_revisit_doubts},
	        {"uncertain_cleared_by", causes},
	        {"scan_us", PrescanTimeToJson(scan.scan_us)},
	        {"scan_at_least_1024_us", scan.scan_at_least_1024_us}};
}

// Writes what the prescan's job queue counted and the times around it.
nlohmann::json PrescanQueueToJson(const PrescanQueueReport& queue) {
	return {{"pushed", queue.pushed},
	        {"dropped", queue.dropped},
	        {"dropped_no_buffer", queue.dropped_no_buffer},
	        {"late_skipped", queue.late_skipped},
	        {"markers_lost", queue.markers_lost},
	        {"truncated", queue.truncated},
	        {"copied_dwords", queue.copied_dwords},
	        {"peak", queue.peak},
	        {"copy_us", PrescanTimeToJson(queue.copy_us)},
	        {"push_us", PrescanTimeToJson(queue.push_us)},
	        {"lag_ms", PrescanTimeToJson(queue.lag_ms)}};
}

// The scanner's part of the prescan block. The preparer's figures join it as further keys.
nlohmann::json PrescanScannerToJson(const PrescanScannerReport& block) {
	return {{"scan", PrescanScanToJson(block.scan)},
	        {"queue", PrescanQueueToJson(block.queue)},
	        {"thread",
	         {{"jobs", block.jobs},
	          {"lag_skipped", block.lag_skipped},
	          {"stopped_by_preparer", block.stopped_by_preparer},
	          {"wakes", block.wakes}}},
	        {"reads",
	         {{"calls", block.read_calls}, {"max_us", block.read_max_us}, {"over_1ms", block.read_over_1ms}}},
	        {"offer",
	         {{"posted", block.offers_posted}, {"skipped", block.offers_skipped}, {"dropped", block.offers_dropped}}}};
}

// Writes how the preparer's state answered the dispatches it was asked about.
nlohmann::json PrescanStateToJson(const PrescanStateReport& state) {
	return {{"posted", state.posted},
	        {"busy", state.busy},
	        {"capped", state.capped},
	        {"settled", state.settled},
	        {"uncertain", state.uncertain},
	        {"dropped_full", state.dropped_full}};
}

// Writes what the preparer counted before it sent anything, with the refusals as an object by
// reason.
nlohmann::json PrescanPreparerIntakeToJson(const PrescanPreparerReport& preparer) {
	nlohmann::json invalid = nlohmann::json::object();
	for (const auto& [name, count]: preparer.invalid) {
		invalid[name] = count;
	}
	return {{"items", preparer.items},
	        {"no_map_entry", preparer.no_map_entry},
	        {"zero_size", preparer.zero_size},
	        {"indirect_thread_dims", preparer.indirect_thread_dims},
	        {"bad_item", preparer.bad_item},
	        {"backed_off", preparer.backed_off},
	        {"blocked", preparer.blocked},
	        {"plan_requests", preparer.plan_requests},
	        {"compile_requests", preparer.compile_requests},
	        {"code_unreadable", preparer.code_unreadable},
	        {"code_changed", preparer.code_changed},
	        {"plan_failed", preparer.plan_failed},
	        {"has_calls", preparer.has_calls},
	        {"srt_faults", preparer.srt_faults},
	        {"materialize_failed", preparer.materialize_failed},
	        {"indirect_args_unknown", preparer.indirect_args_unknown},
	        {"invalid", invalid}};
}

// Writes what became of the preparer's variants and requests, joined to its intake figures.
nlohmann::json PrescanPreparerToJson(const PrescanPreparerReport& preparer) {
	auto json = PrescanPreparerIntakeToJson(preparer);
	json.update(nlohmann::json {{"variant_modules", preparer.variant_modules},
	                            {"known_variants", preparer.known_variants},
	                            {"capped_variants", preparer.capped_variants},
	                            {"known_skipped", preparer.known_skipped},
	                            {"heap_image_variants", preparer.heap_image_variants},
	                            {"covered_variants", preparer.covered_variants},
	                            {"known_modules", preparer.known_modules},
	                            {"known_dropped", preparer.known_dropped},
	                            {"refused_sends", preparer.refused_sends},
	                            {"never_retry", preparer.never_retry},
	                            {"blamed", preparer.blamed},
	                            {"dropped", preparer.dropped},
	                            {"compile_failed", preparer.compile_failed},
	                            {"key_mismatch", preparer.key_mismatch},
	                            {"decode_failed", preparer.decode_failed},
	                            {"record_mismatch", preparer.record_mismatch},
	                            {"adopted", preparer.adopted},
	                            {"taken", preparer.taken},
	                            {"late_duplicates", preparer.late_duplicates},
	                            {"taken_dropped", preparer.taken_dropped},
	                            {"plans_dropped", preparer.plans_dropped},
	                            {"forgotten", preparer.forgotten},
	                            {"verdicts_evicted", preparer.verdicts_evicted},
	                            {"failed", preparer.failed},
	                            {"lane_off", preparer.lane_off}});
	return json;
}

// Writes one kind of round trip to the helper.
nlohmann::json PrescanRttToJson(const PrescanRttFigures& rtt) {
	return {{"count", rtt.count},
	        {"p50", Round3(rtt.p50_ms)},
	        {"p90", Round3(rtt.p90_ms)},
	        {"max", Round3(rtt.max_ms)}};
}

// Writes what the supervisor of the helper's child process counted.
nlohmann::json PrescanChildToJson(const PrescanChildReport& child) {
	return {{"spawns", child.spawns},
	        {"failed_spawns", child.failed_spawns},
	        {"crashes", child.crashes},
	        {"hangs", child.hangs},
	        {"protocol_errors", child.protocol_errors},
	        {"failures", child.failures},
	        {"never_retry", child.never_retry},
	        {"never_retry_programs", child.never_retry_programs},
	        {"lane_off", child.lane_off},
	        {"lane_off_reason", child.lane_off_reason},
	        {"submitted", child.submitted},
	        {"answered", child.answered},
	        {"blamed", child.blamed},
	        {"dropped", child.dropped},
	        {"resent", child.resent},
	        {"refused", child.refused},
	        {"late_replies", child.late_replies},
	        {"unexpected_replies", child.unexpected_replies},
	        {"callback_failures", child.callback_failures},
	        {"peak_outstanding", child.peak_outstanding},
	        {"priority", child.priority},
	        {"plan_rtt_ms", PrescanRttToJson(child.plan_rtt)},
	        {"compile_rtt_ms", PrescanRttToJson(child.compile_rtt)}};
}

// Writes how the drain treated the prescan's modules. The timing sits in an object named for both
// producers, since one drain takes the generator's modules and the prescan's under one budget, and
// it is the same timing the generator's block gives as its own.
nlohmann::json PrescanDrainToJson(const PrescanDrainReport& drain) {
	return {{"adopted", drain.adopted},
	        {"duplicates", drain.duplicates},
	        {"plan_mismatch", drain.plan_mismatch},
	        {"failed_installs", drain.failed_installs},
	        {"dropped_stopped", drain.dropped_stopped},
	        {"item_max_ms", Round3(drain.item_max_ms)},
	        {"both_producers",
	         {{"calls", drain.both_calls},
	          {"over_1ms", drain.both_over_1ms},
	          {"over_4ms", drain.both_over_4ms},
	          {"max_ms", Round3(drain.both_max_ms)},
	          {"total_ms", Round3(drain.both_total_ms)},
	          {"lag_ms_max", Round3(drain.both_lag_ms_max)}}}};
}

// Writes the prescan's builds. The overlap figures sit in an object named for what they are, upper
// bounds that nothing may gate on, apart from the counts and the worker time they are measured
// against.
nlohmann::json PrescanBuildsToJson(const PrescanBuildsReport& builds) {
	return {{"builds", builds.builds},
	        {"worker_builds", builds.worker_builds},
	        {"total_ms", Round3(builds.total_ms)},
	        {"upper_bounds",
	         {{"in_scope", builds.in_scope},
	          {"in_scope_ms", Round3(builds.in_scope_ms)},
	          {"in_hold", builds.in_hold},
	          {"in_hold_ms", Round3(builds.in_hold_ms)}}}};
}

// Writes the GPU thread's own compute compiles by reason. The reasons and the verdicts are objects
// by name, and the one upper bound sits in an object named for what it is, so nothing gates on it.
nlohmann::json PrescanMissesToJson(const PrescanMissesReport& misses) {
	nlohmann::json by_reason = nlohmann::json::object();
	for (const auto& [name, count]: misses.by_reason) {
		by_reason[name] = count;
	}
	nlohmann::json refused_by = nlohmann::json::object();
	for (const auto& [name, count]: misses.refused_by) {
		refused_by[name] = count;
	}
	return {{"inline_compiles", misses.inline_compiles},
	        {"by_reason", by_reason},
	        {"refused_by", refused_by},
	        {"upper_bounds", {{"built_undrained", misses.built_undrained}}}};
}

// Adds the pipeline cache's part to the prescan block, next to whatever keys the scanner gave it.
void AddPrescanCacheToJson(const PrescanCacheReport& figures, nlohmann::json& block) {
	block["state"]    = PrescanStateToJson(figures.state);
	block["preparer"] = PrescanPreparerToJson(figures.preparer);
	if (figures.child) {
		block["child"] = PrescanChildToJson(*figures.child);
	}
	block["drain"]   = PrescanDrainToJson(figures.drain);
	block["builds"]  = PrescanBuildsToJson(figures.builds);
	// The GPU thread's own compiles, kept apart from the prescan's builds.
	block["holds"]   = {{"in_flight_at_miss_only", figures.holds.in_flight_at_miss_only}};
	block["misses"]  = PrescanMissesToJson(figures.misses);
	block["records"] = {{"stored_first_use", figures.records.stored_first_use},
	                    {"retained_peak_bytes", figures.records.retained_peak_bytes},
	                    {"dropped_over_cap", figures.records.dropped_over_cap},
	                    {"unused_at_exit", figures.records.unused_at_exit}};
}

// The whole prescan block, from either part or both, or nothing when neither registered. The
// scanner's keys and the cache's keys never share a name, so the two only add to one object.
std::optional<nlohmann::json> PrescanBlockToJson(const std::optional<PrescanScannerReport>& scanner,
                                                 const std::optional<PrescanCacheReport>&   cache) {
	if (!scanner && !cache) {
		return std::nullopt;
	}
	auto block = scanner ? PrescanScannerToJson(*scanner) : nlohmann::json::object();
	if (cache) {
		AddPrescanCacheToJson(*cache, block);
	}
	return block;
}

// The prescan_graphics block, one object per group. A whole number is written as an integer, so a
// count reads as a count, and any other as a number rounded to three places.
nlohmann::json GraphicsPrescanToJson(const PrescanGraphicsReport& figures) {
	// Doubles hold integers exactly up to 2^53.
	constexpr double exact_integer_limit = 9007199254740992.0;
	nlohmann::json   block               = nlohmann::json::object();
	for (const auto& group: figures.groups) {
		auto& object = block[group.name];
		object       = nlohmann::json::object();
		const auto number = [&](double value) {
			const bool whole = value >= 0.0 && value < exact_integer_limit && value == std::floor(value);
			return whole ? nlohmann::json(static_cast<uint64_t>(value)) : nlohmann::json(Round3(value));
		};
		for (const auto& [name, value]: group.figures) {
			object[name] = number(value);
		}
		for (const auto& [name, values]: group.series) {
			auto& array = object[name];
			array       = nlohmann::json::array();
			for (const double value: values) {
				array.push_back(number(value));
			}
		}
	}
	return block;
}

// The graphics prescan's miss rows (experimental). An ago time that did not happen is left out.
nlohmann::json PrescanMissRowsToJson(const std::vector<PrescanMissRow>& rows, int64_t epoch_ns) {
	nlohmann::json result = nlohmann::json::array();
	for (const auto& entry: rows) {
		nlohmann::json row = {{"stage", entry.stage}, {"id", Hex(entry.id)},   {"hash", Hex(entry.hash)},
		                      {"kind", entry.kind},   {"flip", entry.flip},    {"seen", entry.seen},
		                      {"scan", entry.scan},   {"job", entry.job}};
		if (entry.at_ns != NO_TIME_NS) {
			row["at_ms"] = RunMs(entry.at_ns, epoch_ns);
		}
		for (const auto& [name, ms]: {std::pair {"first_seen_ago_ms", entry.first_seen_ago_ms},
		                              std::pair {"offered_ago_ms", entry.offered_ago_ms},
		                              std::pair {"posted_ago_ms", entry.posted_ago_ms}}) {
			if (ms >= 0.0) {
				row[name] = Round3(ms);
			}
		}
		result.push_back(std::move(row));
	}
	return result;
}

// The guest audio underruns with each one's time from the run start. count includes the ones that
// came past the capacity and have no time.
nlohmann::json AudioUnderrunsToJson(const AudioUnderruns& underruns, int64_t epoch_ns) {
	nlohmann::json times = nlohmann::json::array();
	const int64_t  base  = epoch_ns == NO_TIME_NS ? 0 : epoch_ns;
	for (const auto at_ns: underruns.at_ns) {
		times.push_back(Round3(static_cast<double>(at_ns - base) / NS_PER_MS));
	}
	// The arrays are parallel to at_ms. child_alive is written as 0 or 1, and prescan_running is a
	// count of pre-scan work in progress.
	return {{"count", underruns.at_ns.size() + underruns.overflow},
	        {"overflow", underruns.overflow},
	        {"at_ms", times},
	        {"warm_front", underruns.warm_front},
	        {"warm_back", underruns.warm_back},
	        {"child_alive", underruns.child_alive},
	        {"prescan_running", underruns.prescan_running}};
}

// The audio held time by cause. Each cause is written, even at 0, so a gate can read every one
// without checking that it exists. warm_inline_paused is a cause of its own and not a part of
// warm_inline, so the figures add up to the whole held time.
nlohmann::json HeldByCauseToJson(const GpuStall::HoldBreakdown& holds) {
	nlohmann::json held_ms    = nlohmann::json::object();
	nlohmann::json hold_count = nlohmann::json::object();
	for (size_t cause = 0; cause < GpuStall::HOLD_CAUSE_COUNT; cause++) {
		const char* name = GpuStall::HoldCauseName(static_cast<GpuStall::HoldCause>(cause));
		held_ms[name]    = Round3(static_cast<double>(holds.cause_ns[cause]) / NS_PER_MS);
		hold_count[name] = holds.cause_count[cause];
	}
	return {{"held_ms", held_ms}, {"holds", hold_count}};
}

// The lengths of the holds in fixed buckets, for display. bounds_ms are the upper edges, and buckets
// has one more entry for the holds past the last edge. The percentiles here are bucket upper bounds
// and are named so, since they can be hundreds of ms above the truth. A gate reads
// audio_holds_exact instead.
nlohmann::json HoldHistogramToJson(const GpuStall::HoldHistogram& histogram) {
	return {{"count", histogram.count},
	        {"p50_bound_ms", Round3(histogram.PercentileMs(0.5))},
	        {"p95_bound_ms", Round3(histogram.PercentileMs(0.95))},
	        {"max_ms", Round3(histogram.MaxMs())},
	        {"bounds_ms", GpuStall::HOLD_BUCKET_BOUNDS_MS},
	        {"buckets", histogram.buckets}};
}

// The exact nearest-rank figures of the hold lengths. count is the holds stored, and overflow the
// ones past the capacity, which the percentiles leave out. max_ms covers every hold.
nlohmann::json HoldExactToJson(const GpuStall::HoldExact& exact) {
	return {{"count", exact.count},
	        {"overflow", exact.overflow},
	        {"p50_ms", Round3(exact.p50_ms)},
	        {"p95_ms", Round3(exact.p95_ms)},
	        {"max_ms", Round3(exact.max_ms)}};
}

nlohmann::json MachineToJson(const MachineInfo& info) {
	return {{"cpu_model", info.cpu_model},
	        {"logical_cores", info.logical_cores},
	        {"warm_workers", info.warm_workers},
	        {"ram_bytes", info.ram_bytes}};
}

// The emulator's own peaks, with a figure the system did not give left out.
nlohmann::json ProcessToJson(const MachineInfo& info) {
	nlohmann::json process = nlohmann::json::object();
	if (info.peak_working_set_bytes) {
		process["peak_working_set_bytes"] = *info.peak_working_set_bytes;
	}
	if (info.peak_commit_bytes) {
		process["peak_commit_bytes"] = *info.peak_commit_bytes;
	}
	return process;
}

nlohmann::json BootToJson(const BootInfo& boot) {
	return {{"warm_up_ms", Round3(boot.warm_up_ms)},
	        {"programs", boot.programs},
	        {"modules", boot.modules},
	        {"pipelines", boot.pipelines}};
}

// The names of the library kinds and fallback reasons in the JSON, in enum order.
constexpr std::array<const char*, LIBRARY_KIND_COUNT> LIBRARY_KIND_NAMES {
    "vertex_input", "pre_raster", "fragment", "output"};
constexpr std::array<const char*, FALLBACK_KIND_COUNT> FALLBACK_KIND_NAMES {
    "rect_list", "tess", "mesh", "create_failed"};

// The sum and the largest of one figure of the rows.
nlohmann::json SumAndMax(const std::vector<LibraryPipelineRow>& rows, double LibraryPipelineRow::*field) {
	double sum = 0.0, largest = 0.0;
	for (const auto& row: rows) {
		sum += row.*field;
		largest = std::max(largest, row.*field);
	}
	return {{"sum", Round3(sum)}, {"max", Round3(largest)}};
}

// The summary of some library pipeline rows. built_both_stage_libraries counts the pipelines whose thread
// made the pre-rasterization and the fragment shader library itself, one after the other. A thread that
// built one stage library and waited for the other shows up as a row with one built kind and a
// non-zero waited_ms, and is not counted there.
nlohmann::json LibraryPipelinesSummary(const std::vector<LibraryPipelineRow>& rows) {
	constexpr uint32_t both_stages = (1u << static_cast<uint32_t>(LibraryKind::PreRaster)) |
	                                 (1u << static_cast<uint32_t>(LibraryKind::Fragment));
	uint32_t both = 0, any = 0;
	for (const auto& row: rows) {
		both += (row.built_kinds & both_stages) == both_stages ? 1u : 0u;
		any += row.built_count != 0 ? 1u : 0u;
	}
	return {{"count", rows.size()},
	        {"built_ms", SumAndMax(rows, &LibraryPipelineRow::built_ms)},
	        {"waited_ms", SumAndMax(rows, &LibraryPipelineRow::waited_ms)},
	        {"link_ms", SumAndMax(rows, &LibraryPipelineRow::link_ms)},
	        {"total_ms", SumAndMax(rows, &LibraryPipelineRow::total_ms)},
	        {"built_any_library", any},
	        {"built_both_stage_libraries", both}};
}

// The rows and their summary, under graphics_pipeline_library.pipelines.
nlohmann::json LibraryPipelinesToJson(const std::vector<LibraryPipelineRow>& rows) {
	nlohmann::json list = nlohmann::json::array();
	for (const auto& row: rows) {
		nlohmann::json kinds = nlohmann::json::array();
		for (size_t index = 0; index < LIBRARY_KIND_COUNT; index++) {
			if ((row.built_kinds & (1u << index)) != 0) kinds.push_back(LIBRARY_KIND_NAMES[index]);
		}
		list.push_back({{"vs", Hex(row.vs)},
		                {"ps", Hex(row.ps)},
		                {"flip", row.flip},
		                {"thread", row.thread == Thread::Gpu ? "gpu" : "worker"},
		                {"built_ms", Round3(row.built_ms)},
		                {"built_count", row.built_count},
		                {"built_kinds", kinds},
		                {"waited_ms", Round3(row.waited_ms)},
		                {"link_ms", Round3(row.link_ms)},
		                {"total_ms", Round3(row.total_ms)}});
	}
	// Only the GPU thread's time blocks play, so its rows are summed apart from the workers'.
	std::vector<LibraryPipelineRow> gpu, worker;
	for (const auto& row: rows) (row.thread == Thread::Gpu ? gpu : worker).push_back(row);
	return {{"summary",
	         {{"all", LibraryPipelinesSummary(rows)},
	          {"gpu", LibraryPipelinesSummary(gpu)},
	          {"worker", LibraryPipelinesSummary(worker)}}},
	        {"rows", list}};
}

// The graphics_pipeline_library section, which says whether libraries were on and what they did.
nlohmann::json PipelineLibraryToJson(const PipelineLibraryInfo&  info,
                                     const PipelineLibraryStats& stats,
                                     const std::vector<LibraryPipelineRow>& rows) {
	nlohmann::json libraries = nlohmann::json::object();
	for (size_t index = 0; index < LIBRARY_KIND_COUNT; index++) {
		const auto& kind = stats.libraries[index];
		libraries[LIBRARY_KIND_NAMES[index]] = {{"built", kind.built},
		                                        {"hits", kind.hits},
		                                        {"failed", kind.failed},
		                                        {"refused_hits", kind.refused_hits},
		                                        {"ms", Round3(kind.ms)}};
	}
	nlohmann::json fallback = nlohmann::json::object();
	for (size_t index = 0; index < FALLBACK_KIND_COUNT; index++) {
		fallback[FALLBACK_KIND_NAMES[index]] = stats.fallback_full[index];
	}
	return {{"active", info.active},
	        {"inactive_reason", info.inactive_reason},
	        {"libraries", libraries},
	        {"links", {{"count", stats.links},
	                   {"ms_total", Round3(stats.link_ms_total)},
	                   {"ms_max", Round3(stats.link_ms_max)}}},
	        {"fallback_full", fallback},
	        {"pipelines", LibraryPipelinesToJson(rows)}};
}

} // namespace

std::string Report::ToJson(const Metadata& metadata, uint64_t audio_holds, double run_seconds,
                           double audio_held_ms, const GpuStall::HoldBreakdown& holds) const {
	const auto summary  = Summarize();
	const auto compiles = Compiles();
	const auto boot     = Boot();
	const auto library  = PipelineLibrary();
	int64_t    epoch_ns = NO_TIME_NS;
	{
		std::lock_guard lock(m_mutex);
		epoch_ns = m_epoch_ns;
	}

	// The generation block exists only when the flag asked for it. The source is called with the
	// generation lock held, so the owner cannot destroy what it reads meanwhile, and no other lock
	// of the report is held then.
	std::optional<nlohmann::json> generation;
	{
		std::lock_guard lock(m_generation_mutex);
		if (m_generation_source) {
			generation = GenerationToJson(m_generation_source(), epoch_ns);
		} else if (m_generation_final) {
			generation = GenerationToJson(*m_generation_final, epoch_ns);
		} else if (m_generation_inactive) {
			generation =
			    nlohmann::json {{"active", false}, {"inactive_reason", *m_generation_inactive}};
		}
	}

	// The scan differential block exists only when the flag registered a source or final figures. The
	// source is called with its lock held, so the owner cannot destroy what it reads meanwhile.
	std::optional<nlohmann::json> scan_differential;
	{
		std::lock_guard lock(m_scan_differential_mutex);
		if (m_scan_differential_source) {
			scan_differential = ScanDifferentialToJson(m_scan_differential_source());
		} else if (m_scan_differential_final) {
			scan_differential = ScanDifferentialToJson(*m_scan_differential_final);
		}
	}
	// The prescan block exists only when a scanner or a pipeline cache registered its figures.
	auto prescan = PrescanBlockToJson(PrescanScanFigures(), PrescanCacheFigures());
	// The prescan_graphics block exists only when the scanner or the cache registered a part of it.
	const auto graphics_prescan = GraphicsPrescanFigures();

	// Read once, so the two blocks that come from it agree.
	const auto machine = ReadMachineInfo();

	nlohmann::json flags = nlohmann::json::object();
	for (const auto& [name, on]: metadata.flags) {
		flags[name] = on;
	}
	nlohmann::json compile_rows = nlohmann::json::array();
	for (const auto& compile: compiles) {
		compile_rows.push_back(CompileToJson(compile, epoch_ns));
	}
	nlohmann::json ready_rows = nlohmann::json::array();
	for (const auto& ready: Readies()) {
		ready_rows.push_back(ReadyToJson(ready));
	}
	nlohmann::json report = {
	    {"title_id", metadata.title_id},
	    {"build", metadata.build},
	    {"flags", flags},
	    {"cold", metadata.cold},
	    {"cold_stamp_failures", summary.stamp_failures},
	    {"partial", metadata.partial},
	    {"boot", BootToJson(boot)},
	    {"graphics_pipeline_library",
	     PipelineLibraryToJson(library, LibraryStats(), LibraryPipelines())},
	    {"play", PlayToJson(summary)},
	    {"submit_lead_summary", SubmitLeadsToJson(summary.submit_leads)},
	    {"pauses", PausesToJson(summary)},
	    {"compiles", compile_rows},
	    {"ready", ready_rows},
	    {"warm_takes", WarmTakesToJson(WarmTakes(), epoch_ns)},
	    {"skips", SkipsToJson(summary)},
	    {"deferred_draws", DeferredToJson(summary)},
	    {"audio_holds", audio_holds},
	    {"audio_held_ms", Round3(audio_held_ms)},
	    {"audio_held_by_cause", HeldByCauseToJson(holds)},
	    {"audio_hold_histogram", HoldHistogramToJson(holds.histogram)},
	    {"audio_holds_exact", HoldExactToJson(holds.exact)},
	    {"audio_underruns", AudioUnderrunsToJson(AudioUnderrunsSoFar(), epoch_ns)},
	    {"machine", MachineToJson(machine)},
	    {"process", ProcessToJson(machine)},
	    {"flips", summary.flips},
	    {"submissions",
	     {{"total", summary.submissions}, {"per_flip", summary.submissions_per_flip}}},
	    {"run_seconds", Round3(run_seconds)},
	};
	if (generation) {
		report["shader_store_generation"] = std::move(*generation);
	}
	if (scan_differential) {
		report["scan_differential"] = std::move(*scan_differential);
	}
	if (prescan) {
		report["prescan"] = std::move(*prescan);
	}
	// The miss rows join the prescan_graphics block, which they make when no part registered one.
	const auto miss_rows = PrescanMissRows();
	if (graphics_prescan || !miss_rows.empty()) {
		auto block = graphics_prescan ? GraphicsPrescanToJson(*graphics_prescan) : nlohmann::json::object();
		block["miss_rows"]         = PrescanMissRowsToJson(miss_rows, epoch_ns);
		block["miss_rows_dropped"] = PrescanMissRowsDropped();
		report["prescan_graphics"] = std::move(block);
	}
	// The title id is not filtered, so bytes that are not UTF-8 are replaced instead of throwing.
	return report.dump(JSON_INDENT, ' ', false, nlohmann::json::error_handler_t::replace);
}

Report& Global() {
	// Never destroyed, so a late record at process exit finds it alive.
	static Report& report = *new Report;
	return report;
}

void Begin(Metadata metadata) {
	auto&           output = GetOutput();
	std::lock_guard lock(output.mutex);
	output.metadata = std::move(metadata);
	output.start    = std::chrono::steady_clock::now();
	Global().SetEpoch(ToNs(output.start));
}

void MarkPartial() {
	g_partial.store(true, std::memory_order_relaxed);
}

void RecordStampFailure() {
	static std::atomic<bool> logged {false};
	Global().AddStampFailure();
	if (!logged.exchange(true)) {
		Log::WriteToConsoleAndLog(
		    "WARNING: --cold-pipelines could not stamp a SPIR-V module, so it compiled unstamped "
		    "and a cache may serve it. The report counts these in cold_stamp_failures.\n");
	}
}

std::filesystem::path ReportPath(std::string_view title_id,
                                 std::chrono::system_clock::time_point when) {
	using namespace std::chrono;
	const auto seconds_total = floor<seconds>(when);
	const auto days_total    = floor<days>(seconds_total);
	const year_month_day civil {days_total};
	const hh_mm_ss       clock {seconds_total - days_total};
	// The title id is not filtered, so anything that is not safe in a file name becomes an underscore.
	std::string safe_title(title_id);
	std::ranges::replace_if(
	    safe_title,
	    [](unsigned char letter) {
		    return std::isalnum(letter) == 0 && letter != '-' && letter != '_';
	    },
	    '_');
	const auto name = fmt::format("{}-{:04}{:02}{:02}-{:02}{:02}{:02}.json",
	                              safe_title.empty() ? "UNKNOWN" : safe_title,
	                              static_cast<int>(civil.year()),
	                              static_cast<unsigned>(civil.month()),
	                              static_cast<unsigned>(civil.day()), clock.hours().count(),
	                              clock.minutes().count(), clock.seconds().count());
	return std::filesystem::path(REPORT_DIRECTORY) / Common::PathFromUtf8(name);
}

namespace {

// Writes through a .tmp file and a rename, so a crash or a full disk never leaves a report that
// looks complete. Returns false when anything failed.
bool WriteWhole(const std::filesystem::path& path, const std::string& text) {
	auto temporary = path;
	temporary += ".tmp";
	uint32_t written = 0;
	{
		// The one-argument constructor creates the file. Mode::Write only opens an existing one.
		File file(temporary);
		if (file.IsInvalid()) {
			return false;
		}
		file.Write(text.data(), static_cast<uint32_t>(text.size()), &written);
		file.Close();
	}
	if (written != text.size() || !File::RenameFile(temporary, path)) {
		return false;
	}
	return true;
}

void WriteReportUnguarded() {
	auto&    output = GetOutput();
	Metadata metadata;
	double   run_seconds = 0.0;
	{
		std::lock_guard lock(output.mutex);
		metadata    = output.metadata;
		run_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - output.start)
		                  .count();
	}
	metadata.partial = g_partial.load(std::memory_order_relaxed);
	// The breakdown is read before the total, and a hold adds to the total before its cause, so a
	// hold that ends meanwhile can only leave the causes short of the total and never over it.
	const auto breakdown = GpuStall::Breakdown();
	const auto holds     = GpuStall::HoldCount();
	const auto held_ms   = static_cast<double>(GpuStall::HeldNs()) / NS_PER_MS;
	const auto text      = Global().ToJson(metadata, holds, run_seconds, held_ms, breakdown);
	const auto path = ReportPath(metadata.title_id, std::chrono::system_clock::now());
	File::CreateDirectories(path.parent_path());
	if (WriteWhole(path, text)) {
		Log::WriteToConsoleAndLog(fmt::format("First-run report: wrote {}\n", PathToString(path)));
	} else {
		Log::WriteToConsoleAndLog(
		    fmt::format("First-run report: cannot write {}\n", PathToString(path)));
	}
}

} // namespace

void WriteReportOnce() {
	if (!Enabled() || g_written.exchange(true)) {
		return;
	}
	// This runs from exit handlers, where an exception would end the process in a crash.
	try {
		WriteReportUnguarded();
	} catch (...) {
		// Logging can throw too, and nothing may escape an exit handler, so it is guarded as well.
		try {
			Log::WriteToConsoleAndLog("First-run report: failed with an exception, no report written\n");
		} catch (...) {
		}
	}
}

int64_t NowNs() {
	return ToNs(std::chrono::steady_clock::now());
}

namespace {

constexpr uint64_t KILOBYTE = 1024;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// The processor's name from the brand string of cpuid, which is the same text as the registry
// holds. Other architectures give nothing, and the report says unknown.
std::string ProcessorName() {
#if defined(_M_X64) || defined(_M_IX86)
	constexpr int FIRST_BRAND_LEAF = 0x80000002;
	constexpr int BRAND_LEAVES     = 3;
	int           max_extended[4]  = {};
	__cpuid(max_extended, static_cast<int>(0x80000000));
	if (static_cast<unsigned>(max_extended[0]) < static_cast<unsigned>(FIRST_BRAND_LEAF + 2)) {
		return {};
	}
	char brand[BRAND_LEAVES * 16 + 1] = {};
	for (int leaf = 0; leaf < BRAND_LEAVES; leaf++) {
		int registers[4] = {};
		__cpuid(registers, FIRST_BRAND_LEAF + leaf);
		std::memcpy(brand + leaf * 16, registers, sizeof(registers));
	}
	std::string name(brand);
	// The text is padded with spaces on the left.
	const auto first = name.find_first_not_of(' ');
	return first == std::string::npos ? std::string() : name.substr(first);
#else
	return {};
#endif
}

uint64_t InstalledMemoryBytes() {
	MEMORYSTATUSEX status {};
	status.dwLength = sizeof(status);
	return GlobalMemoryStatusEx(&status) != FALSE ? status.ullTotalPhys : 0;
}

void ReadProcessPeaks(MachineInfo& info) {
	PROCESS_MEMORY_COUNTERS counters {};
	if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters)) != FALSE) {
		info.peak_working_set_bytes = counters.PeakWorkingSetSize;
		// The peak of the pagefile usage is the process's peak commit.
		info.peak_commit_bytes = counters.PeakPagefileUsage;
	}
}
#else
// The text after the colon of the first line of /proc/cpuinfo that starts with one of the keys.
std::string ProcessorName() {
#ifdef __APPLE__
	char   brand[256] = {};
	size_t size       = sizeof(brand);
	if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) {
		return brand;
	}
	return {};
#else
	std::ifstream cpuinfo("/proc/cpuinfo");
	std::string   line;
	while (std::getline(cpuinfo, line)) {
		for (const std::string_view key: {"model name", "Model", "Hardware", "cpu model"}) {
			if (!line.starts_with(key)) {
				continue;
			}
			const auto colon = line.find(':');
			if (colon == std::string::npos) {
				continue;
			}
			const auto first = line.find_first_not_of(" \t", colon + 1);
			if (first != std::string::npos) {
				return line.substr(first);
			}
		}
	}
	return {};
#endif
}

uint64_t InstalledMemoryBytes() {
#ifdef __APPLE__
	uint64_t memory = 0;
	size_t   size   = sizeof(memory);
	return sysctlbyname("hw.memsize", &memory, &size, nullptr, 0) == 0 ? memory : 0;
#else
	const long pages     = sysconf(_SC_PHYS_PAGES);
	const long page_size = sysconf(_SC_PAGE_SIZE);
	return pages > 0 && page_size > 0
	           ? static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size)
	           : 0;
#endif
}

// The value of a line of /proc/self/status such as "VmHWM:   1234 kB", in bytes.
std::optional<uint64_t> StatusKilobytes(std::string_view key) {
#ifdef __APPLE__
	(void)key;
	return std::nullopt;
#else
	std::ifstream status("/proc/self/status");
	std::string   line;
	while (std::getline(status, line)) {
		if (!line.starts_with(key)) {
			continue;
		}
		const char* begin = line.data() + key.size();
		const char* end   = line.data() + line.size();
		while (begin != end && (*begin == ' ' || *begin == '\t')) {
			++begin;
		}
		uint64_t   kilobytes = 0;
		const auto parsed    = std::from_chars(begin, end, kilobytes);
		if (parsed.ec != std::errc {} || parsed.ptr == begin) {
			return std::nullopt;
		}
		return kilobytes * KILOBYTE;
	}
	return std::nullopt;
#endif
}

void ReadProcessPeaks(MachineInfo& info) {
#ifdef __APPLE__
	rusage used {};
	if (getrusage(RUSAGE_SELF, &used) == 0) {
		// macOS counts ru_maxrss in bytes. It has no peak commit to give.
		info.peak_working_set_bytes = static_cast<uint64_t>(used.ru_maxrss);
	}
#else
	// VmHWM is the peak resident size. Linux has no peak commit to give, since VmPeak is the peak
	// of the address space the process mapped, which for the guest's reservation is far more than
	// it ever committed. So the peak commit is left out, as it is on macOS.
	info.peak_working_set_bytes = StatusKilobytes("VmHWM:");
#endif
}
#endif

} // namespace

MachineInfo ReadMachineInfo() {
	MachineInfo info;
	if (auto name = ProcessorName(); !name.empty()) {
		info.cpu_model = std::move(name);
	}
	info.logical_cores = std::thread::hardware_concurrency();
	info.ram_bytes     = InstalledMemoryBytes();
	info.warm_workers  = Detail::g_warm_workers.load(std::memory_order_relaxed);
	ReadProcessPeaks(info);
	return info;
}

uint64_t NextSubmissionId() {
	return Enabled() ? g_last_submission.fetch_add(1, std::memory_order_relaxed) + 1
	                 : NO_SUBMISSION;
}

SubmissionStamp SubmissionStamp::New(uint32_t queue) {
	SubmissionStamp stamp;
	stamp.id = NextSubmissionId();
	if (stamp.id != NO_SUBMISSION) {
		stamp.enqueued_ns = NowNs();
		stamp.queue       = queue;
	}
	return stamp;
}

SubmissionScope::SubmissionScope(SubmissionStamp& stamp, bool is_commands)
    : m_active(stamp.id != NO_SUBMISSION && Enabled()) {
	if (!m_active) {
		return;
	}
	if (stamp.picked_ns == NO_TIME_NS) {
		stamp.picked_ns = NowNs();
	}
	const bool count = is_commands && !stamp.counted;
	stamp.counted    = stamp.counted || is_commands;
	Global().BeginSubmission(stamp, count);
}

SubmissionScope::~SubmissionScope() {
	if (m_active) {
		Global().EndSubmission();
	}
}

namespace {

// The pipeline being linked on this thread, or null. Its row is the scope's own member.
thread_local LibraryPipelineScope* t_library_pipeline = nullptr;

} // namespace

LibraryPipelineScope::LibraryPipelineScope(uint64_t vs, uint64_t ps, bool on_gpu_thread)
    : m_active(Enabled()) {
	if (!m_active) {
		return;
	}
	m_row.vs     = vs;
	m_row.ps     = ps;
	m_row.thread = on_gpu_thread ? Thread::Gpu : Thread::Worker;
	m_start      = std::chrono::steady_clock::now();
	m_previous   = t_library_pipeline;
	t_library_pipeline = this;
}

LibraryPipelineScope::~LibraryPipelineScope() {
	if (m_active) {
		t_library_pipeline = m_previous;
	}
}

void LibraryPipelineScope::Commit() {
	if (m_active) {
		m_row.total_ms = ElapsedMs(m_start);
		Global().AddLibraryPipeline(m_row);
	}
}

void LibraryPipelineScope::NoteBuilt(LibraryKind kind, double ms) {
	if (t_library_pipeline != nullptr) {
		auto& row = t_library_pipeline->m_row;
		row.built_ms += ms;
		row.built_count++;
		row.built_kinds |= 1u << static_cast<uint32_t>(kind);
	}
}

void LibraryPipelineScope::NoteWaited(double ms) {
	if (t_library_pipeline != nullptr) {
		t_library_pipeline->m_row.waited_ms += ms;
	}
}

void LibraryPipelineScope::NoteLink(double ms) {
	if (t_library_pipeline != nullptr) {
		t_library_pipeline->m_row.link_ms += ms;
	}
}

TranslateScope::TranslateScope(Kind kind, std::string_view stage, uint64_t hash, bool on_gpu_thread)
    : m_active(Enabled()), m_kind(kind), m_thread(on_gpu_thread ? Thread::Gpu : Thread::Worker),
      m_hash(hash) {
	if (m_active) {
		m_stage = std::string(stage);
		m_start = std::chrono::steady_clock::now();
	}
}

TranslateScope::~TranslateScope() {
	if (m_active) {
		Global().AddTranslate(m_kind, m_stage, m_hash, ElapsedMs(m_start), m_thread);
	}
}

CompileScope::CompileScope(Kind kind, uint64_t id, uint64_t id_ps, bool on_gpu_thread)
    : m_active(Enabled()), m_kind(kind), m_thread(on_gpu_thread ? Thread::Gpu : Thread::Worker),
      m_id(id), m_id_ps(id_ps) {
	if (m_active) {
		m_start    = std::chrono::steady_clock::now();
		m_start_ns = ToNs(m_start);
		m_context  = {.warm_running = WarmBuildsRunning(),
		              .warm_front   = WarmFrontBuildsRunning(),
		              .warm_back    = WarmBackBuildsRunning(),
		              .child_alive  = GeneratorChildAlive()};
	}
}

CompileScope::~CompileScope() {
	if (m_active) {
		Global().AddCompile(m_kind, m_id, m_id_ps, ElapsedMs(m_start), m_thread, m_start_ns,
		                    m_context);
	}
}

std::vector<uint32_t> StampSpirv(std::span<const uint32_t> code, uint64_t nonce) {
	if (code.size() < SPIRV_HEADER_WORDS || code[0] != SPIRV_MAGIC) {
		return {};
	}
	// Walk past the instructions that must precede the debug section. The stamp goes before the
	// first instruction that is none of them, or at the end if the module has nothing else.
	size_t insert_at = SPIRV_HEADER_WORDS;
	while (insert_at < code.size()) {
		const uint32_t word_count = code[insert_at] >> 16u;
		if (word_count == 0 || insert_at + word_count > code.size()) {
			return {};
		}
		if (!IsHeaderOpcode(code[insert_at] & 0xFFFFu)) {
			break;
		}
		insert_at += word_count;
	}
	// Characters are packed little endian, and the zero bytes after them end the string.
	std::array<char, STAMP_TEXT_WORDS * sizeof(uint32_t)> text {};
	for (size_t index = 0; index < STAMP_TEXT_CHARS; index++) {
		text[index] = STAMP_ALPHABET[(nonce >> (index * BITS_PER_STAMP_CHAR)) &
		                             (STAMP_ALPHABET.size() - 1)];
	}
	std::array<uint32_t, STAMP_WORD_COUNT> stamp {};
	stamp[0] = (STAMP_WORD_COUNT << 16u) | OP_SOURCE_EXTENSION;
	for (size_t index = 0; index < STAMP_TEXT_WORDS; index++) {
		for (size_t byte = 0; byte < sizeof(uint32_t); byte++) {
			stamp[1 + index] |= static_cast<uint32_t>(static_cast<unsigned char>(
			                        text[index * sizeof(uint32_t) + byte]))
			                    << (byte * 8u);
		}
	}
	std::vector<uint32_t> stamped;
	stamped.reserve(code.size() + stamp.size());
	stamped.insert(stamped.end(), code.begin(), code.begin() + static_cast<ptrdiff_t>(insert_at));
	stamped.insert(stamped.end(), stamp.begin(), stamp.end());
	stamped.insert(stamped.end(), code.begin() + static_cast<ptrdiff_t>(insert_at), code.end());
	return stamped;
}

uint64_t StampNonce() {
	return g_run_nonce.load(std::memory_order_relaxed);
}

} // namespace Common::FirstRunReport
