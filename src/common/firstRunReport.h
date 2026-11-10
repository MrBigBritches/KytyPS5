#ifndef KYTY_COMMON_FIRST_RUN_REPORT_H_
#define KYTY_COMMON_FIRST_RUN_REPORT_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Measurement of a title's first run, when shaders and pipelines are still compiling. Two opt-in
// flags drive it. --first-run-report writes one JSON file at clean exit, and --cold-pipelines
// stamps every SPIR-V module so no driver cache can serve it. With both off, each call site costs
// one relaxed atomic load.
namespace Common::FirstRunReport {

enum class Kind : uint8_t { Graphics, Compute };
enum class Phase : uint8_t { Boot, Play };
enum class Thread : uint8_t { Gpu, Worker };
enum class SkipKind : uint8_t { Draw, Dispatch };

/// Pause lengths in ms that start a histogram bucket. A pause counts in the highest bucket whose
/// bound it reaches, and a shorter one counts as under the first bound.
inline constexpr std::array<uint32_t, 7> PAUSE_BUCKET_BOUNDS_MS {100, 250, 500, 1000, 2500, 5000, 10000};
inline constexpr size_t WORST_TRANSLATE_LIMIT = 10;
inline constexpr size_t LONGEST_PAUSE_LIMIT   = 20;

/// Submission id of a compile that no guest submission caused. Ids start at 1.
inline constexpr uint64_t NO_SUBMISSION = 0;
/// Start time that is not known, so the row gets no submission or timing fields.
inline constexpr int64_t NO_TIME_NS = INT64_MIN;

/// What the report keeps about one guest submission, made when it is enqueued and carried with it
/// through every resume. Times are on the NowNs clock.
struct SubmissionStamp {
	uint64_t id          = NO_SUBMISSION;
	// When the guest thread enqueued it. A blocked submission goes back to its queue with this
	// unchanged, so later resumes do not move it.
	int64_t  enqueued_ns = 0;
	// When the GPU thread first picked it up. Set once, by the first SubmissionScope.
	int64_t  picked_ns   = NO_TIME_NS;
	// The queue it was enqueued on, where 0 is graphics and the others are compute.
	uint32_t queue       = 0;
	// True once a command list has been counted, so a resume is not counted again.
	bool     counted     = false;

	/// A stamp for a submission enqueued now, or an inert one (id NO_SUBMISSION) with the report off.
	[[nodiscard]] static SubmissionStamp New(uint32_t queue);
};

// Ids for pipelines the renderer builds for itself rather than from a guest shader. The top bits
// tag the helper and the rest are zero, which a hashed module id is very unlikely to match.
inline constexpr uint64_t HELPER_ID_TILER = 0xFFFF000100000000ull;
inline constexpr uint64_t HELPER_ID_BLIT  = 0xFFFF000200000000ull;
inline constexpr uint64_t HELPER_ID_PSSR  = 0xFFFF000300000000ull;

namespace Detail {
inline std::atomic<bool> g_report_enabled {false};
inline std::atomic<bool> g_cold_enabled {false};
} // namespace Detail

/// True when --first-run-report is on.
[[nodiscard]] inline bool Enabled() noexcept {
	return Detail::g_report_enabled.load(std::memory_order_relaxed);
}

/// True when --cold-pipelines is on.
[[nodiscard]] inline bool ColdEnabled() noexcept {
	return Detail::g_cold_enabled.load(std::memory_order_relaxed);
}

/// Turns the report and cold stamping on or off. Called when the emulator config loads.
void Configure(bool report_enabled, bool cold_enabled);

/// One shader translation that missed the program cache.
struct TranslateEntry {
	std::string stage;
	uint64_t    hash = 0;
	double      ms   = 0.0;
};

/// What a pipeline creation met when it began, for the rows of worker compiles. warm_running is how
/// many background warm-up builds were running, and child_alive whether the shader store
/// generator's worker process was, so a row can be compared across runs with the generator on and
/// off.
struct CompileContext {
	uint32_t warm_running = 0;
	bool     child_alive  = false;
};

/// One pipeline creation. create_ms is the whole creation function, which includes layouts and
/// helper modules as well as the driver call. For graphics id is the vertex stage id and id_ps the
/// pixel stage id, which is 0 when there is none. For compute id is the module id.
/// submission is the guest submission being processed on the GPU thread when the compile began, or
/// NO_SUBMISSION for worker compiles, boot warm-up and compiles outside any submission. The fields
/// after it are meaningful only when it is set. queue is the submission's queue. visible_ms is the
/// compile start minus the time the submission was first enqueued. It includes time the submission
/// sat in its queue, waited on labels, or was blocked and resumed, and the GPU thread could not
/// have looked at it during a compile or a wait. picked_ms is the compile start minus the time the
/// GPU thread first picked the submission up. start_ns is the compile start on the NowNs clock.
struct CompileEntry {
	Kind     kind      = Kind::Graphics;
	uint64_t id        = 0;
	uint64_t id_ps     = 0;
	double   create_ms = 0.0;
	Phase    phase     = Phase::Play;
	Thread   thread    = Thread::Gpu;
	int64_t  flip      = 0;
	uint64_t submission = NO_SUBMISSION;
	uint32_t queue      = 0;
	double   visible_ms = 0.0;
	double   picked_ms  = 0.0;
	int64_t  start_ns   = NO_TIME_NS;
	/// When a worker compile began, on the NowNs clock, and what it met. Only the rows of worker
	/// compiles have them. start_ns stays for the rows that join a submission.
	int64_t  at_ns        = NO_TIME_NS;
	uint32_t warm_running = 0;
	bool     child_alive  = false;
};

/// One background pipeline build the GPU thread took. ready_ms runs from the request to the end of
/// the compile, so it includes the time in the queue, and skipped_draws counts the draws that went
/// without the pipeline meanwhile.
struct ReadyEntry {
	Kind     kind           = Kind::Graphics;
	uint64_t id             = 0;
	uint64_t id_ps          = 0;
	double   ready_ms       = 0.0;
	uint32_t skipped_draws  = 0;
};

/// Draws that missed their pipeline and were refused a skip for one reason, with the time they
/// waited for it.
struct Refusal {
	std::string reason;
	uint32_t    draws      = 0;
	double      blocked_ms = 0.0;
};

/// One resolve of the draws deferred by --deferred-pipeline-draws. All times are in ms.
/// pre_resolve_ms runs from the first hole to the start of the resolve, resolve_wait_ms is the wait
/// for the builds, record_ms the recording of the holes and save_ms the driver cache save inside the
/// resolve. hold_count_change is how many holds of guest audio the resolve added, and
/// window_hold_count_change how many were added from its first hole until the resolve ended, so
/// the time before the resolve counts too. held_ms and window_held_ms are the same two windows
/// measured in milliseconds that guest audio was held, which resolve_wait_ms is not. A stall is
/// counted in the window where it ends, so a resolve nested inside an outer scope shows 0 for
/// itself and the outer stall shows when that ends. flip and
/// submission are stamped by the report when the resolve is added, as for CompileEntry, and trigger
/// names what forced the resolve (submit, fault, stream_wrap, cap, shutdown, readback, label,
/// flip, unmap or slice_end).
struct DeferredResolve {
	int64_t     flip              = 0;
	uint64_t    submission        = NO_SUBMISSION;
	std::string trigger;
	uint32_t    holes             = 0;
	uint32_t    distinct_keys     = 0;
	double      resolve_wait_ms   = 0.0;
	double      record_ms         = 0.0;
	double      pre_resolve_ms    = 0.0;
	double      save_ms           = 0.0;
	uint64_t    hold_count_change = 0;
	uint64_t    window_hold_count_change = 0;
	double      held_ms           = 0.0;
	double      window_held_ms    = 0.0;
};

/// One draw that --deferred-pipeline-draws could not defer and that then waited for its pipeline,
/// either compiling it or waiting for a worker. A draw that an S2 skip took instead waited for
/// nothing and has no row. vs and ps are the stage ids, as in a compile row. waited_ms runs from the
/// miss to the pipeline's return, so it also holds the store and driver cache checkpoint that
/// follow a compile, and the input dump when dumps are on. gpu_thread is false for a draw met off
/// the GPU thread, whose wait stopped no guest rendering. flip is stamped by the report when the
/// row is added.
struct IneligibleWait {
	std::string reason;
	uint64_t    vs         = 0;
	uint64_t    ps         = 0;
	double      waited_ms  = 0.0;
	bool        gpu_thread = true;
	int64_t     flip       = 0;
};

/// How a dispatch got a module the background warm-up had queued, the first time it used it.
enum class WarmTakeHow : uint8_t {
	// A worker had already built it.
	Done,
	// A worker was building it, and the dispatch waited.
	Waited,
	// No worker had started it, so the dispatch built it itself.
	Inline,
};

/// The first use of one module the background warm-up had queued. at_ns is when the dispatch asked
/// for it, before any wait, on the NowNs clock, or NO_TIME_NS when no time was taken. stopped_ms is
/// how long the dispatch was stopped for it, which is about 0 for a finished build. It is not
/// exactly 0 when a worker finished just as the dispatch began to wait. Comparing the rows of a
/// shader's variants shows how far ahead of one variant another was first used. flip is stamped by
/// the report when the row is added.
struct WarmTake {
	uint64_t    id         = 0;
	WarmTakeHow how        = WarmTakeHow::Done;
	double      stopped_ms = 0.0;
	int64_t     at_ns      = NO_TIME_NS;
	int64_t     flip       = 0;
	/// Whether the warm queue was held back by a pause when the dispatch asked. Only a take that
	/// stopped the dispatch reads it, and it is not written to the warm_takes rows.
	bool paused = false;
};

/// One run of the shader store worker that ended badly, as the report lists it.
struct GenerationFailure {
	uint64_t    hash      = 0;
	uint64_t    rank      = 0;
	int         exit_code = 0;
	std::string kind;
	std::string last_line;
	/// Which spawn ended, counting from 1, so the row joins with its usage row.
	uint32_t spawn = 0;
	/// The worker's processor time when it ended, when the system said.
	std::optional<double> cpu_ms;
	/// The end of the worker's text file at that time.
	std::string out_tail;
};

/// What one spawn of the worker used.
struct GenerationSpawnUsage {
	uint32_t                spawn  = 0;
	double                  cpu_ms = 0.0;
	std::optional<uint64_t> peak_bytes;
	/// What peak_bytes measures: working_set, working_set_sampled or commit. Empty without a peak.
	std::string peak_source;
};

/// A program that the worker gave up on.
struct GenerationDrop {
	uint64_t    hash = 0;
	uint64_t    rank = 0;
	std::string reason;
};

/// What a worker's `prediction` line said about a program.
struct GenerationPrediction {
	uint64_t hash = 0;
	/// predicted, calls, unclassified, unreadable or problems.
	std::string outcome;
	uint64_t    candidates = 0;
};

/// One generated module the drain took in. Times are on the NowNs clock, and a time that did not
/// happen is missing. lead_ms is first_use minus ready, so a negative number means the dispatch got
/// there before a worker had finished the build. how is done, waited or inline, from the first use
/// of the module, and unused when no dispatch took it.
struct GeneratedModuleRow {
	uint64_t               id         = 0;
	uint64_t               code_hash  = 0;
	int64_t                posted_ns  = NO_TIME_NS;
	int64_t                adopted_ns = NO_TIME_NS;
	std::optional<int64_t> ready_ns;
	std::optional<int64_t> first_use_ns;
	std::optional<double>  build_ms;
	std::optional<double>  lead_ms;
	std::string            how;
};

/// The misses of one class.
struct GenerationMissClass {
	std::string name;
	uint64_t    count     = 0;
	double      create_ms = 0.0;
};

/// The compute translations that the runtime did with generation on, by the reason generation did
/// not serve them.
struct GenerationMisses {
	uint64_t                         total     = 0;
	double                           create_ms = 0.0;
	std::vector<GenerationMissClass> classes;
	/// How the drain refused the modules of the dropped_at_drain class, by verdict name.
	std::vector<std::pair<std::string, uint64_t>> refused;
	uint64_t                                      wrong_spec_programs     = 0;
	uint64_t                                      wrong_spec_max_variants = 0;
};

/// A first use that stopped the dispatch for long, with what is known of the module's life. The
/// generated times are missing for a module that generation did not make.
struct GenerationSlowTake {
	uint64_t               id        = 0;
	bool                   generated = false;
	std::string            how;
	double                 stopped_ms = 0.0;
	int64_t                at_ns      = NO_TIME_NS;
	bool                   paused     = false;
	std::optional<int64_t> posted_ns;
	std::optional<int64_t> adopted_ns;
	std::optional<int64_t> ready_ns;
};

/// A take that stopped the dispatch for at least this long is listed in the generation block.
inline constexpr double SLOW_TAKE_MS = 250.0;

/// The most audio underruns the report keeps a time for. Past it they are only counted.
inline constexpr size_t AUDIO_UNDERRUN_CAPACITY = 8192;

/// The guest audio underruns of the run: how many were stored, how many came past the capacity, and
/// when each stored one happened on the NowNs clock, in the order they were stored.
struct AudioUnderruns {
	uint64_t             overflow = 0;
	std::vector<int64_t> at_ns;
};

/// What the report says about the machine and the emulator's own process. A figure the system did
/// not give is missing, or "unknown" for the processor.
struct MachineInfo {
	std::string             cpu_model     = "unknown";
	uint32_t                logical_cores = 0;
	uint64_t                ram_bytes     = 0;
	uint32_t                warm_workers  = 0;
	std::optional<uint64_t> peak_working_set_bytes;
	std::optional<uint64_t> peak_commit_bytes;
};

/// What the GPU thread's drain of generated modules counted.
struct GenerationDrain {
	uint64_t calls                 = 0;
	uint64_t over_1ms              = 0;
	uint64_t over_4ms              = 0;
	double   max_ms                = 0.0;
	double   total_ms              = 0.0;
	double   lag_ms_max            = 0.0;
	double   item_max_ms           = 0.0;
	uint64_t adopted               = 0;
	uint64_t duplicates            = 0;
	uint64_t plan_mismatch         = 0;
	uint64_t plan_mismatch_reverse = 0;
	uint64_t failed_installs       = 0;
	uint64_t dropped_stopped       = 0;
	uint64_t posted                = 0;
	uint64_t handles_created       = 0;
	uint64_t handles_destroyed     = 0;
	uint64_t bad_plans             = 0;
	uint64_t undrained_at_stop     = 0;
	uint64_t pending               = 0;
};

/// The report's view of --generate-shader-store: plain numbers and text, since this library cannot
/// see the generator. The pipeline cache builds it, from a copy of what the supervisor and the
/// drain counted.
struct ShaderStoreGenerationReport {
	/// False when the supervisor ended for a reason that stops generation (outcome fatal or
	/// restarts_capped), and then inactive_reason says why.
	bool        active = true;
	std::string inactive_reason;
	/// What became of the finished file: renamed, rename_failed or not_finished. It is the field to
	/// trust for what is on disk, since a run reported as stopped can still have renamed its file.
	std::string persisted = "not_finished";
	/// Why the rename failed. Empty for any other value of persisted.
	std::string persist_error;
	/// running, finished, fatal, restarts_capped or stopped.
	std::string outcome;
	uint32_t    spawns   = 0;
	uint32_t    restarts = 0;
	double      spawn_ms = 0.0;
	/// Missing until the first predict line, the first record and the last record arrive.
	std::optional<double> child_open_ms;
	std::optional<double> first_record_ms;
	std::optional<double> last_record_ms;
	/// Set only when the worker finished, and the end of the run for any outcome except running.
	std::optional<double> finished_ms;
	/// The moment of the finish on the report's clock (NowNs), written as finished_at_ms from the
	/// run start like an underrun's at_ms. finished_ms counts from the supervisor thread's start
	/// instead, which is later than the run start, so only this one can cut a window of the run in
	/// two. Set only when the worker finished.
	std::optional<int64_t>            finished_ns;
	std::optional<double>             ended_ms;
	double                            decode_ms_max      = 0.0;
	double                            terminate_wait_ms  = 0.0;
	uint32_t                          frames_delivered   = 0;
	uint32_t                          bad_records        = 0;
	uint32_t                          duplicates         = 0;
	uint32_t                          posted             = 0;
	uint32_t                          predicted_programs = 0;
	std::vector<GenerationFailure>    failures;
	std::vector<GenerationSpawnUsage> spawn_usage;
	/// The largest peak memory of the spawns, with its source. Missing when none was measured.
	std::optional<uint64_t>     peak_bytes;
	std::string                 peak_source;
	std::vector<GenerationDrop> dropped;
	std::vector<uint64_t>       failed_hashes;
	std::vector<uint64_t>       skip_hashes;
	/// The fields of the worker's summary line, in the order it wrote them. Empty when none came.
	std::vector<std::pair<std::string, std::string>> worker;
	GenerationDrain                                  drain;
	/// The latest `prediction` line of each program, in the order the programs were first seen.
	std::vector<GenerationPrediction> predictions;
	/// One row for each module taken in, and the build time of those that no dispatch took.
	std::vector<GeneratedModuleRow> generated_modules;
	double                          unused_build_ms = 0.0;
	GenerationMisses                misses;
	std::vector<GenerationSlowTake> slow_takes;
};

/// The GPU thread's blocking within one guest flip, merged into a single pause.
struct Pause {
	int64_t  flip   = 0;
	double   ms     = 0.0;
	uint32_t events = 0;
};

/// What the boot-time warm-up replayed from the shader store.
struct BootInfo {
	double   warm_up_ms = 0.0;
	uint32_t programs   = 0;
	uint32_t modules    = 0;
	uint32_t pipelines  = 0;
};

/// Whether graphics pipelines may be built from libraries on this device. --graphics-pipeline-library
/// in the flags only says it was asked for. inactive_reason names the first requirement the device
/// missed, and is empty when active or when the device was never asked.
struct PipelineLibraryInfo {
	bool        active = false;
	std::string inactive_reason;
};

/// The four libraries a graphics pipeline is linked from, in pipeline order.
enum class LibraryKind : uint8_t { VertexInput, PreRaster, Fragment, Output };
inline constexpr size_t LIBRARY_KIND_COUNT = 4;

/// Why a pipeline was built whole although --graphics-pipeline-library was on. The first three are
/// pipelines libraries do not cover, and CreateFailed is a driver refusal of a library or link.
enum class FallbackKind : uint8_t { RectList, Tess, Mesh, CreateFailed };
inline constexpr size_t FALLBACK_KIND_COUNT = 4;

/// What one kind of library did. built counts libraries made, hits counts requests a finished
/// library answered (including one still being made by another thread), failed counts libraries the
/// driver refused, refused_hits counts requests that an entry the driver refused answered, and ms is
/// the time spent making the built ones. ms is the CPU figure for library work. The compile rows of
/// pipelines built from libraries also hold the time they waited for other threads' libraries and
/// the link, so summing worker rows over-counts CPU when libraries are shared.
struct LibraryKindStats {
	uint32_t built        = 0;
	uint32_t hits         = 0;
	uint32_t failed       = 0;
	uint32_t refused_hits = 0;
	double   ms           = 0.0;
};

/// What building pipelines from libraries did over the run. Links are the fast links that joined
/// four libraries into a pipeline, and fallback_full counts pipelines built whole, by reason.
struct PipelineLibraryStats {
	std::array<LibraryKindStats, LIBRARY_KIND_COUNT> libraries {};
	uint32_t                                         links         = 0;
	double                                           link_ms_total = 0.0;
	double                                           link_ms_max   = 0.0;
	std::array<uint32_t, FALLBACK_KIND_COUNT>        fallback_full {};
};

/// Milliseconds since a time on the steady clock.
[[nodiscard]] double ElapsedMs(std::chrono::steady_clock::time_point start);

/// One pipeline linked from libraries, with where its time went. built_ms is the time this thread
/// spent making libraries for this pipeline, built_count of them, and built_kinds has a bit for each
/// LibraryKind among them. waited_ms is the time spent waiting for libraries other threads were
/// making, and link_ms the fast link. total_ms is the whole create, layouts included, so it is at
/// least the sum of the three. A pipeline built whole after a refusal has no row.
struct LibraryPipelineRow {
	uint64_t vs           = 0;
	uint64_t ps           = 0;
	int64_t  flip         = 0;
	Thread   thread       = Thread::Worker;
	double   built_ms     = 0.0;
	double   waited_ms    = 0.0;
	double   link_ms      = 0.0;
	double   total_ms     = 0.0;
	uint32_t built_count  = 0;
	uint32_t built_kinds  = 0;
};

/// Identity of the run, written at the top of the file.
struct Metadata {
	std::string                                title_id;
	std::string                                build;
	std::vector<std::pair<std::string, bool>> flags;
	bool                                       cold = false;
	// True when the process ended early to relaunch itself, so the play is split across reports.
	bool                                       partial = false;
};

/// Figures worked out from the raw records, kept apart from JSON so tests can read them.
struct Summary {
	double                                                   blocked_graphics_ms = 0.0;
	double                                                   blocked_compute_ms  = 0.0;
	double                                                   translate_ms        = 0.0;
	// The part of translate_ms that blocked the GPU thread in play for graphics shaders, which is
	// in blocked_graphics_ms with the compiles and waits. The refusal times plus this reconcile
	// with blocked_graphics_ms, up to compiles and waits that no refused draw made.
	double                                                   translate_gpu_graphics_ms = 0.0;
	std::vector<TranslateEntry>                              worst_translate;
	std::array<uint32_t, PAUSE_BUCKET_BOUNDS_MS.size()>      pause_histogram {};
	uint32_t                                                 pauses_under_first_bound = 0;
	double                                                   longest_pause_ms         = 0.0;
	std::vector<Pause>                                       longest_pauses;
	uint32_t                                                 pause_count = 0;
	uint32_t                                                 flips       = 0;
	// Command submissions the GPU thread began, counted once each even when one was resumed.
	uint32_t                                                 submissions = 0;
	// Entry F counts the submissions begun while the flip counter was F, so entry 0 is before the first flip and
	// the counts run from flip 0. Only finished flips are listed. The flip still running is only
	// in the total.
	std::vector<uint32_t>                                    submissions_per_flip;
	uint32_t                                                 stamp_failures = 0;
	std::array<uint32_t, 2>                                  skips {};
	std::vector<std::pair<std::string, uint32_t>>            skips_by_target;
	std::vector<std::pair<std::string, uint32_t>>            sampled_after_skip_by_target;
	std::vector<Refusal>                                     refusals;
	// Filled only by --deferred-pipeline-draws. Empty or zero otherwise.
	std::vector<DeferredResolve>                             deferred_resolves;
	// Entry F counts the resolves that ended while the flip counter was F. Unlike
	// submissions_per_flip, the flip still running is appended when it has resolves, so the list can
	// be one longer than flips. Each resolve's own flip field is exact.
	std::vector<uint32_t>                                    deferred_resolves_per_flip;
	// Draws that --deferred-pipeline-draws could not defer, counted per reason. A draw that an S2 skip
	// then took is counted too, since the count says it was ineligible and not what became of it.
	std::vector<std::pair<std::string, uint32_t>>            deferred_ineligible;
	// The draws above that waited for their pipeline, one row each in the order they happened, and
	// their waits summed per reason. They show which reasons the draws that waited came from. A draw
	// that S2 also refused is in skips.refused too, so the two must not be added together.
	std::vector<IneligibleWait>                              deferred_ineligible_waits;
	std::vector<std::pair<std::string, double>>              deferred_ineligible_waited_ms;
	// Deferred draws that the S2 skip rules would have allowed to go without a pipeline. waited_ms
	// is the time from the draw's hole creation until its build is ready, counted once per draw, so
	// it compares with skips.refused.*.blocked_ms.
	uint32_t                                                 deferred_would_skip_draws = 0;
	double                                                   deferred_would_skip_waited_ms = 0.0;
	// Warm-up compute builds the GPU thread ran itself while --pause-warm-up-for-draws held the
	// queue. That is an upper bound on what the pause cost, since a worker that was busy might not
	// have reached the build either. The second figure counts those that ran as long as a hold of
	// guest audio takes to start.
	// These two are the JSON keys warm_inline.during_pause and warm_inline.over_hold_threshold. Every
	// over_hold_threshold figure means long enough to hold audio, if audio was waiting.
	uint32_t                                                 deferred_warm_inline = 0;
	uint32_t                                                 deferred_warm_inline_over_hold_threshold = 0;
	// Every warm-up compute build a dispatch took before it finished, held or not, so the arms of
	// a run can be compared. They are the JSON keys warm_inline.total, total_ms and
	// total_over_hold_threshold, and warm_waited.count, ms and over_hold_threshold.
	uint32_t                                                 warm_inline_total = 0;
	double                                                   warm_inline_total_ms = 0.0;
	uint32_t                                                 warm_inline_total_over_hold_threshold = 0;
	uint32_t                                                 warm_waited_count = 0;
	double                                                   warm_waited_ms = 0.0;
	uint32_t                                                 warm_waited_over_hold_threshold = 0;
};

/// The raw records of one run. Thread safe, and it never reads a clock, so tests drive it with
/// made-up durations.
class Report {
public:
	/// Sets the phase that later records are tagged with.
	void SetPhase(Phase phase);
	/// Notes the guest flip the next records belong to, and counts the flip.
	void OnFlip(int64_t flip);
	/// Stores what the boot warm-up did.
	void SetBoot(const BootInfo& boot);
	/// Stores whether the device builds pipelines from libraries.
	void SetPipelineLibrary(const PipelineLibraryInfo& info);
	/// What SetPipelineLibrary stored.
	[[nodiscard]] PipelineLibraryInfo PipelineLibrary() const;
	/// Counts a library made in `ms`.
	void AddLibraryBuilt(LibraryKind kind, double ms);
	/// Counts a request that an existing library answered.
	void AddLibraryHit(LibraryKind kind);
	/// Counts a library the driver refused.
	void AddLibraryFailed(LibraryKind kind);
	/// Counts a request that a refused library answered.
	void AddLibraryRefusedHit(LibraryKind kind);
	/// Counts one fast link and how long it took.
	void AddLibraryLink(double ms);
	/// Counts a pipeline built whole although libraries were asked for.
	void AddLibraryFallback(FallbackKind kind);
	/// What the Add calls above recorded.
	[[nodiscard]] PipelineLibraryStats LibraryStats() const;
	/// Records one pipeline linked from libraries. The flip field is replaced by the current flip.
	void AddLibraryPipeline(const LibraryPipelineRow& row);
	/// Copy of the library pipeline rows, for tests.
	[[nodiscard]] std::vector<LibraryPipelineRow> LibraryPipelines() const;
	/// Records a miss-path translation. Only GPU-thread time during play counts as blocked.
	void AddTranslate(Kind kind, std::string_view stage, uint64_t hash, double ms, Thread thread);
	/// Records a driver compile. Only GPU-thread time during play counts as blocked. start_ns is
	/// when the compile began, on the NowNs clock, and links GPU-thread rows to the submission.
	/// Without it the row is left unlinked.
	void AddCompile(Kind kind, uint64_t id, uint64_t id_ps, double create_ms, Thread thread,
	                int64_t start_ns = NO_TIME_NS, CompileContext context = {});
	/// Records one guest audio underrun at the time now, which is the NowNs clock. It takes no
	/// lock, allocates nothing and never logs, since it is called from the guest's audio submit. A
	/// slot is reserved with one relaxed increment, and an underrun past AUDIO_UNDERRUN_CAPACITY
	/// only adds to the overflow count, so the index cannot wrap.
	void AddAudioUnderrun() noexcept;
	/// The underruns recorded so far. Slots that a racing call reserved and has not written yet are
	/// left out.
	[[nodiscard]] AudioUnderruns AudioUnderrunsSoFar() const;
	/// Notes the submission the GPU thread is processing. count is true when it should join the
	/// submission totals, which the caller does once per command list.
	void BeginSubmission(const SubmissionStamp& stamp, bool count);
	/// Sets the point in the run that start_ms is counted from, on the NowNs clock.
	void SetEpoch(int64_t epoch_ns);
	/// Notes that the GPU thread is outside any submission.
	void EndSubmission();
	/// Records GPU-thread time spent waiting for a compile that a worker was running. It counts as
	/// blocked in play, though no compile ran on the GPU thread.
	void AddWait(Kind kind, double ms);
	/// Records how long a background pipeline took from request to ready.
	void AddReady(Kind kind, uint64_t id, uint64_t id_ps, double ready_ms, uint32_t skipped_draws);
	/// Records a skipped draw or dispatch against a target class.
	void AddSkip(SkipKind kind, std::string_view target_class);
	/// Records skipped draws whose target was then read as a texture, against a target class.
	void AddSampledAfterSkip(std::string_view target_class, uint32_t draws);
	/// Records a draw that missed its pipeline and was refused a skip, and how long it waited.
	void AddRefusal(std::string_view reason, double blocked_ms);
	/// Records one resolve of deferred draws.
	void AddDeferredResolve(const DeferredResolve& resolve);
	/// Records a draw that --deferred-pipeline-draws could not defer, against the reason.
	void AddDeferredIneligible(std::string_view reason);
	/// Records a draw that could not be deferred and waited for its pipeline, and adds the wait to
	/// its reason's total. The flip field is ignored and replaced by the current flip.
	void AddDeferredIneligibleWait(const IneligibleWait& wait);
	/// Records a deferred draw that S2 would have skipped, and how long it waited for its pipeline.
	void AddDeferredWouldSkip(double waited_ms);
	/// Records a warm-up compute build that ran inline while the warm-up queue was held, and whether
	/// it ran long enough to hold guest audio.
	void AddDeferredWarmInline(bool over_hold_threshold);
	/// Records a warm-up compute build a dispatch took before it finished: one it ran itself, or one
	/// it waited for on a worker. `ms` is how long the dispatch was stopped, and the last argument
	/// whether that was long enough to hold guest audio.
	void AddWarmCompute(bool waited, double ms, bool over_hold_threshold);
	/// Records the first use of a module the background warm-up had queued. The flip field is
	/// ignored and replaced by the current flip.
	void AddWarmTake(const WarmTake& take);
	/// The GPU-thread time blocked so far in play for the kind.
	[[nodiscard]] double BlockedMs(Kind kind) const;
	/// Records a module that --cold-pipelines could not stamp, so it compiled unstamped.
	void AddStampFailure();
	/// Forgets everything recorded. For tests, which share the process-wide report. It must not run
	/// at the same time as AddAudioUnderrun, which writes the underrun slots without a lock.
	void Reset();
	/// What the boot warm-up stored.
	[[nodiscard]] BootInfo Boot() const;

	/// Works out the totals, histogram and top lists from what was recorded so far.
	[[nodiscard]] Summary Summarize() const;
	/// Renders the whole report as JSON text. `audio_held_ms` is the whole run's guest audio held
	/// time, which also catches holds that land outside every deferred-draw window.
	[[nodiscard]] std::string ToJson(const Metadata& metadata, uint64_t audio_holds,
	                                 double run_seconds, double audio_held_ms) const;
	/// Copy of the compile rows, for tests.
	[[nodiscard]] std::vector<CompileEntry> Compiles() const;
	/// Copy of the ready rows, for tests.
	[[nodiscard]] std::vector<ReadyEntry> Readies() const;
	/// Copy of the warm-up take rows, for tests.
	[[nodiscard]] std::vector<WarmTake> WarmTakes() const;

	/// Says generation was asked for and refused, so the report has a block with active false and
	/// this reason. With nothing said, the report has no generation block at all, which is how a
	/// run without the flag looks.
	void SetGenerationInactive(std::string reason);
	/// Registers the function the writer calls for the figures of a running generation. The writer
	/// holds the generation lock while it calls, so the owner clears the source with
	/// SetGenerationFinal before what the function reads is destroyed. The function must not call
	/// back into the report's generation functions, and takes no lock the owner holds while it
	/// calls them.
	void SetGenerationSource(std::function<ShaderStoreGenerationReport()> source);
	/// Stores the final figures and removes the source, in one step under the generation lock, so
	/// the writer sees either the source or the final figures and never a source that is gone.
	void SetGenerationFinal(ShaderStoreGenerationReport final_figures);

private:
	// Moves the pause being merged to the finished ones when it belongs to another flip. Caller
	// holds the lock.
	void FlushPendingPauseLocked(int64_t current_flip);
	// Adds GPU-thread blocking to the pause of the current flip. Caller holds the lock.
	void AddBlockedLocked(Kind kind, double ms);
	// Copies the deferred-draw and warm-up figures into a summary. Caller holds the lock.
	void CopyDeferredLocked(Summary& summary) const;
	// Pauses including the one still being merged. Caller holds the lock.
	[[nodiscard]] std::vector<Pause> AllPausesLocked() const;

	// Guards the three generation members. The writer holds it while it calls the source, and the
	// source reads this report's rows through Compiles() and WarmTakes(), which take m_mutex. So the
	// order is this mutex first, then m_mutex or the owner's own mutex, and never the reverse.
	// Nothing that holds m_mutex calls into the generation members, so it cannot deadlock.
	mutable std::mutex                           m_generation_mutex;
	std::optional<std::string>                   m_generation_inactive;
	std::function<ShaderStoreGenerationReport()> m_generation_source;
	std::optional<ShaderStoreGenerationReport>   m_generation_final;

	mutable std::mutex           m_mutex;
	Phase                        m_phase = Phase::Play;
	int64_t                      m_flip  = 0;
	uint32_t                     m_flips = 0;
	SubmissionStamp              m_stamp;
	int64_t                      m_epoch_ns = NO_TIME_NS;
	uint32_t                     m_submissions = 0;
	uint32_t                     m_submissions_this_flip = 0;
	std::vector<uint32_t>        m_submissions_per_flip;
	uint32_t                     m_stamp_failures = 0;
	BootInfo                     m_boot;
	PipelineLibraryInfo          m_pipeline_library;
	PipelineLibraryStats         m_library_stats;
	std::vector<LibraryPipelineRow> m_library_pipelines;
	std::array<double, 2>        m_blocked_ms {};
	double                       m_translate_ms = 0.0;
	double                       m_translate_gpu_graphics_ms = 0.0;
	std::vector<TranslateEntry>  m_translates;
	std::vector<CompileEntry>    m_compiles;
	std::vector<ReadyEntry>      m_readies;
	std::vector<Pause>           m_pauses;
	bool                         m_has_pending_pause = false;
	Pause                        m_pending_pause;
	std::array<uint32_t, 2>      m_skips {};
	std::vector<std::pair<std::string, uint32_t>> m_skips_by_target;
	std::vector<std::pair<std::string, uint32_t>> m_sampled_after_skip;
	std::vector<Refusal>                          m_refusals;
	std::vector<DeferredResolve>                  m_deferred_resolves;
	std::vector<uint32_t>                         m_deferred_resolves_per_flip;
	uint32_t                                      m_deferred_resolves_this_flip = 0;
	std::vector<std::pair<std::string, uint32_t>> m_deferred_ineligible;
	std::vector<IneligibleWait>                   m_deferred_ineligible_waits;
	std::vector<std::pair<std::string, double>>   m_deferred_ineligible_waited_ms;
	uint32_t                                      m_deferred_would_skip_draws = 0;
	double                                        m_deferred_would_skip_waited_ms = 0.0;
	uint32_t                                      m_deferred_warm_inline = 0;
	uint32_t                                      m_deferred_warm_inline_over = 0;
	uint32_t                                      m_warm_inline_total = 0;
	double                                        m_warm_inline_total_ms = 0.0;
	uint32_t                                      m_warm_inline_total_over = 0;
	uint32_t                                      m_warm_waited_count = 0;
	double                                        m_warm_waited_ms = 0.0;
	uint32_t                                      m_warm_waited_over = 0;
	std::vector<WarmTake>                         m_warm_takes;

	// The audio underruns, written without a lock from the audio submit. A slot holds 0 until its
	// time is stored, and a time is never 0 on the steady clock.
	std::array<std::atomic<int64_t>, AUDIO_UNDERRUN_CAPACITY> m_underrun_ns {};
	std::atomic<size_t>                                       m_underrun_next {0};
	std::atomic<uint64_t>                                     m_underrun_overflow {0};
};

/// The report of this process.
Report& Global();

/// Records the identity written to the file, and starts the run clock.
void Begin(Metadata metadata);

/// Writes the report once into _FirstRunReports beside the working directory. Later calls do
/// nothing, so every clean exit path may call it. Does nothing unless the report is on.
void WriteReportOnce();

/// Where a report for this title and time would go, as <dir>/<TITLE_ID>-<UTC yyyymmdd-hhmmss>.json.
[[nodiscard]] std::filesystem::path ReportPath(std::string_view title_id,
                                               std::chrono::system_clock::time_point when);

/// Marks the report as partial because the process is about to relaunch itself.
void MarkPartial();

/// Notes that a module could not be stamped. Logs the first one.
void RecordStampFailure();

/// Nanoseconds on the steady clock that the report times use.
[[nodiscard]] int64_t NowNs();

/// A fresh submission id, or NO_SUBMISSION with the report off.
[[nodiscard]] uint64_t NextSubmissionId();

/// Marks the GPU thread as processing one submission for the scope. Inert when the stamp is
/// NO_SUBMISSION or the report is off. The first scope for a stamp sets its picked_ns. A command
/// list is counted once, on the first scope that has is_commands set.
class SubmissionScope {
public:
	SubmissionScope(SubmissionStamp& stamp, bool is_commands);
	~SubmissionScope();
	SubmissionScope(const SubmissionScope&)            = delete;
	SubmissionScope& operator=(const SubmissionScope&) = delete;

private:
	bool m_active;
};

/// Records one guest audio underrun, when the report is on. With it off the cost is the one relaxed
/// load of Enabled.
inline void AddAudioUnderrun() noexcept {
	if (Enabled()) {
		Global().AddAudioUnderrun();
	}
}

namespace Detail {
inline std::atomic<int32_t>  g_warm_running {0};
inline std::atomic<bool>     g_child_alive {false};
inline std::atomic<uint32_t> g_warm_workers {0};
} // namespace Detail

/// How many background warm-up builds are running now.
[[nodiscard]] inline uint32_t WarmBuildsRunning() noexcept {
	const auto running = Detail::g_warm_running.load(std::memory_order_relaxed);
	return running > 0 ? static_cast<uint32_t>(running) : 0;
}

/// Counts one warm-up build as running for as long as it lives, when the report is on. A worker
/// makes it around the build, and the rows of worker compiles read the count when they begin.
class WarmBuildScope {
public:
	WarmBuildScope() noexcept: m_active(Enabled()) {
		if (m_active) {
			Detail::g_warm_running.fetch_add(1, std::memory_order_relaxed);
		}
	}
	~WarmBuildScope() {
		if (m_active) {
			Detail::g_warm_running.fetch_sub(1, std::memory_order_relaxed);
		}
	}
	WarmBuildScope(const WarmBuildScope&)            = delete;
	WarmBuildScope& operator=(const WarmBuildScope&) = delete;

private:
	bool m_active;
};

/// Says whether the shader store generator is running a worker. The supervisor sets it when a
/// worker starts and when the run ends, so it also holds in the short gap between a worker that
/// ended and the restart that follows. A row of a worker compile reads it.
inline void SetGeneratorChildAlive(bool alive) noexcept {
	Detail::g_child_alive.store(alive, std::memory_order_relaxed);
}
[[nodiscard]] inline bool GeneratorChildAlive() noexcept {
	return Detail::g_child_alive.load(std::memory_order_relaxed);
}

/// Says how many workers the background warm-up runs, which the machine block shows.
inline void SetWarmWorkerCount(uint32_t workers) noexcept {
	Detail::g_warm_workers.store(workers, std::memory_order_relaxed);
}

/// Reads the machine and the emulator's own process now: the processor, the core count, the
/// installed memory, the warm-up worker count and the process's peak working set and commit.
[[nodiscard]] MachineInfo ReadMachineInfo();

/// Notes that the guest reached a new flip.
inline void OnFlip(int64_t flip) {
	if (Enabled()) {
		Global().OnFlip(flip);
	}
}

/// Sets the phase that later records are tagged with.
inline void SetPhase(Phase phase) {
	if (Enabled()) {
		Global().SetPhase(phase);
	}
}

/// Stores what the boot warm-up did.
inline void SetBoot(const BootInfo& boot) {
	if (Enabled()) {
		Global().SetBoot(boot);
	}
}

/// Stores whether the device builds pipelines from libraries, and if not, why not.
inline void SetPipelineLibrary(bool active, std::string_view inactive_reason) {
	if (Enabled()) {
		Global().SetPipelineLibrary({.active = active, .inactive_reason = std::string(inactive_reason)});
	}
}

/// Records a skipped draw or dispatch. The counts stay zero until a strategy calls this.
inline void AddSkip(SkipKind kind, std::string_view target_class) {
	if (Enabled()) {
		Global().AddSkip(kind, target_class);
	}
}

/// The GPU-thread time blocked so far for the kind, or zero with the report off.
[[nodiscard]] inline double BlockedMs(Kind kind) {
	return Enabled() ? Global().BlockedMs(kind) : 0.0;
}

/// Records a draw that missed its pipeline and was refused a skip, and how long it waited.
inline void AddRefusal(std::string_view reason, double blocked_ms) {
	if (Enabled()) {
		Global().AddRefusal(reason, blocked_ms);
	}
}

/// Records one resolve of deferred draws.
inline void AddDeferredResolve(const DeferredResolve& resolve) {
	if (Enabled()) {
		Global().AddDeferredResolve(resolve);
	}
}

/// Records a warm-up compute build that ran inline while the warm-up queue was held.
inline void AddDeferredWarmInline(bool over_hold_threshold) {
	if (Enabled()) {
		Global().AddDeferredWarmInline(over_hold_threshold);
	}
}

/// Records a warm-up compute build a dispatch ran itself or waited for.
inline void AddWarmCompute(bool waited, double ms, bool over_hold_threshold) {
	if (Enabled()) {
		Global().AddWarmCompute(waited, ms, over_hold_threshold);
	}
}

/// Records the first use of a module the background warm-up had queued.
inline void AddWarmTake(const WarmTake& take) {
	if (Enabled()) {
		Global().AddWarmTake(take);
	}
}

/// Records a draw that could not be deferred, against the reason.
inline void AddDeferredIneligible(std::string_view reason) {
	if (Enabled()) {
		Global().AddDeferredIneligible(reason);
	}
}

/// Records a draw that could not be deferred and waited for its pipeline.
inline void AddDeferredIneligibleWait(const IneligibleWait& wait) {
	if (Enabled()) {
		Global().AddDeferredIneligibleWait(wait);
	}
}

/// Records a deferred draw that S2 would have skipped, and how long it waited for its pipeline.
inline void AddDeferredWouldSkip(double waited_ms) {
	if (Enabled()) {
		Global().AddDeferredWouldSkip(waited_ms);
	}
}

/// Records skipped draws whose target was then read as a texture, a sign that a skip left damage.
inline void AddSampledAfterSkip(std::string_view target_class, uint32_t draws) {
	if (Enabled()) {
		Global().AddSampledAfterSkip(target_class, draws);
	}
}

/// Records GPU-thread time spent waiting for a compile that a worker was running.
inline void AddWait(Kind kind, double ms) {
	if (Enabled()) {
		Global().AddWait(kind, ms);
	}
}

/// Records how long a background pipeline took from request to ready.
inline void AddReady(Kind kind, uint64_t id, uint64_t id_ps, double ready_ms,
                     uint32_t skipped_draws) {
	if (Enabled()) {
		Global().AddReady(kind, id, id_ps, ready_ms, skipped_draws);
	}
}

/// Counts a library made in `ms`.
inline void AddLibraryBuilt(LibraryKind kind, double ms) {
	if (Enabled()) {
		Global().AddLibraryBuilt(kind, ms);
	}
}

/// Counts a request that an existing library answered.
inline void AddLibraryHit(LibraryKind kind) {
	if (Enabled()) {
		Global().AddLibraryHit(kind);
	}
}

/// Counts a library the driver refused.
inline void AddLibraryFailed(LibraryKind kind) {
	if (Enabled()) {
		Global().AddLibraryFailed(kind);
	}
}

/// Counts a request that a refused library answered.
inline void AddLibraryRefusedHit(LibraryKind kind) {
	if (Enabled()) {
		Global().AddLibraryRefusedHit(kind);
	}
}

/// Counts one fast link and how long it took.
inline void AddLibraryLink(double ms) {
	if (Enabled()) {
		Global().AddLibraryLink(ms);
	}
}

/// Records one pipeline linked from libraries.
inline void AddLibraryPipeline(const LibraryPipelineRow& row) {
	if (Enabled()) {
		Global().AddLibraryPipeline(row);
	}
}

/// Counts a pipeline built whole although libraries were asked for.
inline void AddLibraryFallback(FallbackKind kind) {
	if (Enabled()) {
		Global().AddLibraryFallback(kind);
	}
}

/// Times one pipeline built from libraries, on the thread that builds it. While it exists, the
/// library cache adds to its row the libraries this thread makes, the time it waits for libraries
/// other threads make, and the link. Commit records the row, so a pipeline that falls back to a
/// whole build leaves none. With the report off it reads no clock and every call does nothing.
class LibraryPipelineScope {
public:
	LibraryPipelineScope(uint64_t vs, uint64_t ps, bool on_gpu_thread);
	~LibraryPipelineScope();
	LibraryPipelineScope(const LibraryPipelineScope&)            = delete;
	LibraryPipelineScope& operator=(const LibraryPipelineScope&) = delete;

	/// Records the row with the time since construction as total_ms.
	void Commit();
	/// Adds a library this thread made for the pipeline it is building, if it is building one.
	static void NoteBuilt(LibraryKind kind, double ms);
	/// Adds time spent waiting for a library another thread made.
	static void NoteWaited(double ms);
	/// Adds the time of the link.
	static void NoteLink(double ms);

private:
	bool                                  m_active;
	LibraryPipelineRow                    m_row;
	std::chrono::steady_clock::time_point m_start;
	LibraryPipelineScope*                 m_previous = nullptr;
};

/// Times a shader translation from construction to destruction and records it. Declare it where
/// the cache miss begins. Costs one relaxed load when the report is off.
class TranslateScope {
public:
	TranslateScope(Kind kind, std::string_view stage, uint64_t hash, bool on_gpu_thread);
	~TranslateScope();
	TranslateScope(const TranslateScope&)            = delete;
	TranslateScope& operator=(const TranslateScope&) = delete;

private:
	bool                                  m_active;
	Kind                                  m_kind;
	Thread                                m_thread;
	uint64_t                              m_hash;
	std::string                           m_stage;
	std::chrono::steady_clock::time_point m_start;
};

/// Times a whole pipeline creation function, not only the driver call inside it, from
/// construction to destruction and records it. For a pipeline built from libraries that includes
/// the time spent waiting for libraries other threads were making and the link, so worker rows
/// over-count CPU when libraries are shared. The library figures hold the CPU time.
class CompileScope {
public:
	CompileScope(Kind kind, uint64_t id, uint64_t id_ps, bool on_gpu_thread);
	~CompileScope();
	CompileScope(const CompileScope&)            = delete;
	CompileScope& operator=(const CompileScope&) = delete;

private:
	bool                                  m_active;
	Kind                                  m_kind;
	Thread                                m_thread;
	uint64_t                              m_id;
	uint64_t                              m_id_ps;
	std::chrono::steady_clock::time_point m_start;
	int64_t                               m_start_ns = NO_TIME_NS;
	CompileContext                        m_context;
};

/// Returns a copy of a SPIR-V module with an OpSourceExtension carrying nonce placed first in the
/// debug section, so its bytes differ from every other run. Returns an empty vector when the
/// input is not a well-formed module, and the caller should then use the original.
[[nodiscard]] std::vector<uint32_t> StampSpirv(std::span<const uint32_t> code, uint64_t nonce);

/// The nonce for this run. It is picked when --cold-pipelines is configured, so every module of a
/// run carries it and helper shaders that are rebuilt within a run still match each other, as they
/// would in a true first run. Another run, or another Configure, picks another.
[[nodiscard]] uint64_t StampNonce();

} // namespace Common::FirstRunReport

#endif /* KYTY_COMMON_FIRST_RUN_REPORT_H_ */
