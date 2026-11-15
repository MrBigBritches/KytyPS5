#ifndef KYTY_COMMON_FIRST_RUN_REPORT_H_
#define KYTY_COMMON_FIRST_RUN_REPORT_H_

#include "common/gpuStall.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <map>
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
/// The view-history counts (--async-pipelines-prefer-skip). The first four count targets of draws the
/// skip rule allowed through a partial view's own history, by depth or colour and layer or mip view.
/// TablesFull counts images whose range table was first found full, NotesNotFitted the draws whose
/// range found no slot.
enum class ViewHistoryCount : uint8_t { DepthLayer, DepthMip, ColorLayer, ColorMip, TablesFull, NotesNotFitted, Count };
inline constexpr size_t VIEW_HISTORY_COUNT = static_cast<size_t>(ViewHistoryCount::Count);

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

/// True for the id of a pipeline the renderer builds for itself, which no guest shader caused.
/// The match is exact so a hashed guest id that shares only the tag bits still counts as a guest.
[[nodiscard]] constexpr bool IsHelperId(uint64_t id) noexcept {
	return id == HELPER_ID_TILER || id == HELPER_ID_BLIT || id == HELPER_ID_PSSR;
}

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
/// off. warm_front and warm_back split the warm builds by queue lane. The three counts are separate
/// relaxed loads, so warm_running may briefly differ from warm_front + warm_back. A build outside
/// both lanes is in warm_running only.
struct CompileContext {
	uint32_t warm_running = 0;
	uint32_t warm_front   = 0;
	uint32_t warm_back    = 0;
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
	uint32_t warm_front   = 0;
	uint32_t warm_back    = 0;
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

/// One render target of a draw the skip rule refused for its colour history, kept for the report
/// (diagnostic). The size, layers, levels and format are the image's, and `view` says whether the draw
/// viewed it whole, as a layer, as a mip or outside it. The draw counts and clears come from the
/// history the rule judged, the image's own for a whole view and its range's for a partial one, which
/// `history` names as image, range or none. FrameHistory keeps only the last two frames that had a
/// draw, and the count of the frame right before the earlier of them, so the frame the target was
/// first attached cannot be told. latest_attached_ago and
/// previous_attached_ago say how many frames ago those two were, -1 when the frame is not known.
/// draws_frame_before_last is the draws two frames back, which with draws_last_frame are the inputs
/// of the completed-frames colour rule of --async-pipelines-prefer-skip.
struct SkipTargetFacts {
	uint32_t    width                 = 0;
	uint32_t    height                = 0;
	uint32_t    layers                = 0;
	uint32_t    levels                = 0;
	std::string format;
	std::string view;
	std::string history;
	uint32_t    draws_this_frame      = 0;
	uint32_t    draws_last_frame      = 0;
	uint32_t    draws_frame_before_last = 0;
	bool        cleared_this_frame    = false;
	bool        cleared_last_frame    = false;
	int64_t     latest_attached_ago   = -1;
	int64_t     previous_attached_ago = -1;
};

/// The colour target that refused a draw for its colour history, and the draw's depth target, so the
/// two can be compared. color_slot is the refusing target's hardware slot.
struct SkipTargets {
	SkipTargetFacts color;
	SkipTargetFacts depth;
	uint32_t        color_slot = 0;
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
	/// The vertex and pixel stage ids of the critical key, which name the shaders a stalled resolve
	/// waited on. Set only when has_critical says the resolve had a critical key.
	bool        has_critical      = false;
	uint64_t    critical_vs       = 0;
	uint64_t    critical_ps       = 0;
	/// The skip verdict of each hole this resolve recorded, counted per verdict in the order first met.
	/// A verdict is the S2 rule that refused the hole's skip, or one of the HOLE_VERDICT names. The
	/// critical key is the one with a build whose own acquire took longest, and the critical refusal
	/// is the verdict of the first hole of that key, empty when unknown.
	std::vector<std::pair<std::string, uint32_t>> refused;
	std::string                                   critical_refusal;
	/// The targets of the hole the critical refusal came from, when that hole kept them, which is only
	/// a color_history refusal with the report on.
	std::optional<SkipTargets> critical_targets;
};

/// A hole whose draw S2 allowed to skip but which was deferred, under the default policy B.
inline constexpr std::string_view HOLE_VERDICT_ALLOWED = "allowed";
/// A hole deferred with skips not in play, so the S2 rules were never asked.
inline constexpr std::string_view HOLE_VERDICT_NOT_ASKED = "not_asked";
/// A hole that reached the resolve without a verdict.
inline constexpr std::string_view HOLE_VERDICT_UNKNOWN = "unknown";

/// One hole of a resolve as AddHoleVerdicts counts it. `verdict` is a static name or null for none,
/// `critical` says the hole's key is the one the resolve waited on longest, and `targets` are the
/// refusing targets the hole kept, or null.
struct HoleVerdict {
	const char*        verdict  = nullptr;
	bool               critical = false;
	const SkipTargets* targets  = nullptr;
};

/// Counts each hole under its verdict in the row, a null one as HOLE_VERDICT_UNKNOWN, and takes the
/// first critical hole's verdict as the row's critical refusal, and its targets when it kept any.
void AddHoleVerdicts(DeferredResolve& row, std::span<const HoleVerdict> holes);

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
	/// The id the runtime computes for the dropped module, given for a head drop only. It joins the
	/// drop to the runtime miss of that module.
	std::optional<uint64_t> module_id;
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
/// of the module, and unused when no dispatch took it. tier is how many records of the same program
/// came before it, wide_tail says it is built in the Back lane, and candidates is the
/// program's predicted variant count, missing when no prediction line was read.
struct GeneratedModuleRow {
	uint64_t                id         = 0;
	uint64_t                code_hash  = 0;
	int64_t                 posted_ns  = NO_TIME_NS;
	int64_t                 adopted_ns = NO_TIME_NS;
	std::optional<int64_t>  ready_ns;
	std::optional<int64_t>  first_use_ns;
	std::optional<double>   build_ms;
	std::optional<double>   lead_ms;
	std::string             how;
	uint32_t                tier      = 0;
	bool                    wide_tail = false;
	std::optional<uint64_t> candidates;
};

/// The misses of one class.
struct GenerationMissClass {
	std::string name;
	uint64_t    count     = 0;
	double      create_ms = 0.0;
};

/// A module that the runtime needed before the generator delivered it, one row per module. The
/// times are on the NowNs clock. miss_ns and create_ms are of its first miss, and misses counts all
/// its not_yet_generated misses. via says whether the delivery was an adoption (delivered_ns is the
/// post, and adopted_ns is when the GPU thread took it in) or a refusal as a duplicate
/// (delivered_ns is the refusal, and there is no adopted_ns).
struct GenerationLateModule {
	uint64_t               id           = 0;
	uint64_t               code_hash    = 0;
	int64_t                miss_ns      = NO_TIME_NS;
	int64_t                delivered_ns = NO_TIME_NS;
	std::string            via;
	double                 create_ms = 0.0;
	uint64_t               misses    = 1;
	std::optional<int64_t> adopted_ns;
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
	/// One row for each module with a not_yet_generated miss, up to a cap, and how many modules
	/// were left out.
	std::vector<GenerationLateModule> late_modules;
	uint64_t                          late_overflow = 0;
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
/// when each stored one happened on the NowNs clock, in the order they were stored. warm_front,
/// warm_back and child_alive are parallel to at_ns and hold what was running at each underrun. They
/// are separate relaxed loads, so they may briefly disagree with the total warm count.
struct AudioUnderruns {
	uint64_t              overflow = 0;
	std::vector<int64_t>  at_ns;
	std::vector<uint16_t> warm_front;
	std::vector<uint16_t> warm_back;
	std::vector<uint8_t>  child_alive;
	/// How much pre-scan work was in progress at each underrun, as a count of pre-scan builds
	/// compiling plus child requests outstanding. It is kept apart from child_alive so pre-scan
	/// activity is never excused as the generator's. Saturates at the uint16 range, and is parallel
	/// to at_ns.
	std::vector<uint16_t> prescan_running;
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
	/// Whether the submit-time prescan posted into the same intake, so the drain's timing and the
	/// intake's counts above cover its modules too. Only true is written, so a run without the
	/// prescan writes the block as it always did.
	bool includes_prescan = false;
};

/// The report's view of --generate-shader-store: plain numbers and text, since this library cannot
/// see the generator. The pipeline cache builds it, from a copy of what the supervisor and the
/// drain counted.
struct ShaderStoreGenerationReport {
	/// False when the supervisor ended for a reason that stops generation (outcome fatal or
	/// restarts_capped), and then inactive_reason says why.
	bool        active = true;
	std::string inactive_reason;
	/// What became of the finished file: renamed, rename_failed, existing (an earlier file was
	/// current and was left) or not_finished. It is the field to
	/// trust for what is on disk, since a run reported as stopped can still have renamed its file.
	std::string persisted = "not_finished";
	/// Why the rename failed. Empty for any other value of persisted.
	std::string persist_error;
	/// Where the modules came from: generated, loaded or loaded_regenerated. Loaded from the moment
	/// an earlier file is accepted, and settled when the run finishes.
	std::string source = "generated";
	/// From the spawn of a Verify worker to its answer. Missing when none answered.
	std::optional<double> verify_ms;
	/// Whether a finish gave up on the load of the earlier file, and whether a damaged earlier file
	/// made the supervisor start a worker again in Generate mode.
	bool loader_abandoned         = false;
	bool regenerated_after_damage = false;
	/// The size of the generated file once the run finished, or zero.
	uint64_t file_bytes = 0;
	/// Helper and temporary files of dead runs that were deleted at the start, and of this run
	/// after it succeeded, and the deletions that failed.
	uint32_t files_removed   = 0;
	uint32_t remove_failures = 0;
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

/// How far ahead of the GPU thread's inline compute compiles of guest shaders their submission was
/// enqueued. Helper pipelines are left out. Each lead is a compile row's visible_ms, so it is
/// measured from the enqueue of the submission that holds the dispatch, never from a later one.
/// It bounds how much time a helper process has to prepare a module before the dispatch needs it. Percentiles are nearest rank, and all figures
/// are 0 when n is 0.
struct SubmitLeadSummary {
	uint32_t n      = 0;
	double   p10_ms = 0.0;
	double   p50_ms = 0.0;
	double   p90_ms = 0.0;
	double   max_ms = 0.0;
};

/// How many dispatches landed in one class of the scan differential, split by whether the scan could
/// vouch for the program of the dispatch it was compared with.
struct ScanOutcomeCount {
	uint64_t certain   = 0;
	uint64_t uncertain = 0;

	/// Both halves.
	[[nodiscard]] uint64_t Total() const { return certain + uncertain; }
};

/// The classes of the scan differential, over every submission. The first three count dispatches the
/// command processor ran. Extra counts scanned records that no executed dispatch matched, so it is not
/// part of the dispatches.
struct ScanOutcomeTotals {
	ScanOutcomeCount matched;
	ScanOutcomeCount key_mismatch;
	ScanOutcomeCount missing;
	ScanOutcomeCount extra;

	/// The dispatches the command processor ran and the differential compared.
	[[nodiscard]] uint64_t CpDispatches() const {
		return matched.Total() + key_mismatch.Total() + missing.Total();
	}
	/// The share of those dispatches the scan got wrong, which is key_mismatch and missing together. 0
	/// when there were none. This is what the G0 gate reads.
	[[nodiscard]] double MismatchRate() const {
		const auto total = CpDispatches();
		return total == 0 ? 0.0
		                  : static_cast<double>(key_mismatch.Total() + missing.Total()) /
		                        static_cast<double>(total);
	}
	/// The same share counting only dispatches whose scanned program was certain.
	[[nodiscard]] double CertainMismatchRate() const {
		const auto total = matched.certain + key_mismatch.certain + missing.certain;
		return total == 0 ? 0.0
		                  : static_cast<double>(key_mismatch.certain + missing.certain) /
		                        static_cast<double>(total);
	}
};

/// One dispatch the scan got wrong, kept as an example. kind is "key_mismatch" or "missing", and
/// scanned_key is 0 for a missing one.
struct ScanDifferentialSample {
	std::string kind;
	uint32_t    queue        = 0;
	uint64_t    seq          = 0;
	uint64_t    program      = 0;
	uint64_t    scanned_key  = 0;
	uint64_t    executed_key = 0;
};

/// What the scan differential saw on one queue. The scan columns come from the scanner's own counts.
struct ScanQueueReport {
	uint32_t queue                   = 0;
	uint64_t submissions             = 0;
	uint64_t cp_dispatches           = 0;
	uint64_t dispatches              = 0;
	uint64_t uncertain               = 0;
	uint64_t branch_mode1            = 0;
	uint64_t branch_mode2            = 0;
	uint64_t branch_mode2_empty_else = 0;
	uint64_t ib_revisit_doubts       = 0;
	uint64_t bad_packets             = 0;
	uint64_t scan_allocations        = 0;
	uint64_t scan_p50_us             = 0;
	uint64_t scan_p99_us             = 0;
	uint64_t scan_max_us             = 0;
	/// How many times each cause made the program uncertain, by the cause's name.
	std::vector<std::pair<std::string, uint64_t>> uncertain_cleared_by;
};

/// The report's view of --scan-differential, as plain numbers since this library cannot see the scanner.
struct ScanDifferentialReport {
	ScanOutcomeTotals totals;
	/// Dispatches the command processor ran that had no scan record of their submission at all.
	uint64_t                             untracked_cp_dispatches = 0;
	/// Scanned submissions the command processor had not finished, as when one was still blocked at
	/// shutdown. Their unmatched records are not counted as extra.
	uint64_t                             open_submissions        = 0;
	std::vector<ScanDifferentialSample>  samples;
	std::vector<ScanQueueReport>         queues;
};

/// The most mismatch and missing examples the scan differential keeps.
inline constexpr size_t SCAN_DIFFERENTIAL_SAMPLE_LIMIT = 8;

/// One of the prescan's times, read from a histogram by powers of two and an exact maximum. A
/// percentile is nearest rank and is the top of the bucket its rank falls in, but never above the
/// maximum, so it is an upper bound less than twice the true figure. The exception is the last bucket,
/// which holds every larger value and answers with the maximum. All are 0 when nothing was timed.
struct PrescanTimeFigures {
	uint64_t p50 = 0;
	uint64_t p90 = 0;
	uint64_t p99 = 0;
	uint64_t max = 0;
};

/// What the prescan's walks counted, summed over every walk. The names are those of the scanner's
/// own counts.
struct PrescanScanReport {
	uint64_t submissions              = 0;
	uint64_t dispatches               = 0;
	uint64_t no_shader_state          = 0;
	uint64_t program_inherited        = 0;
	uint64_t uncertain_program        = 0;
	uint64_t ib_followed              = 0;
	uint64_t ib_unreadable            = 0;
	uint64_t ib_depth_limited         = 0;
	uint64_t ib_revisited             = 0;
	uint64_t budget_exhausted         = 0;
	uint64_t bad_packets              = 0;
	uint64_t resets                   = 0;
	uint64_t dwords_read              = 0;
	uint64_t dwords_max               = 0;
	uint64_t dispatches_cached        = 0;
	uint64_t time_capped              = 0;
	uint64_t scan_allocations         = 0;
	uint64_t cache_remembered_busy    = 0;
	uint64_t depth_max                = 0;
	uint64_t register_list_unreadable = 0;
	uint64_t branch_mode1             = 0;
	uint64_t branch_mode2             = 0;
	uint64_t branch_mode2_empty_else  = 0;
	uint64_t ib_revisit_doubts        = 0;
	/// How many times each cause made the program uncertain, by the cause's name.
	std::vector<std::pair<std::string, uint64_t>> uncertain_cleared_by;
	/// How long each walk took, in microseconds.
	PrescanTimeFigures scan_us;
	/// Walks of 1024 microseconds or more. The histogram's buckets split at powers of two, so this is
	/// the exact count nearest to the walks over one millisecond that it can give.
	uint64_t scan_at_least_1024_us = 0;
};

/// What the prescan's job queue counted, and the times of the copy, the push and the wait for the
/// scanner thread.
struct PrescanQueueReport {
	uint64_t pushed            = 0;
	uint64_t dropped           = 0;
	uint64_t dropped_no_buffer = 0;
	uint64_t late_skipped      = 0;
	uint64_t markers_lost      = 0;
	uint64_t truncated         = 0;
	uint64_t copied_dwords     = 0;
	uint64_t peak              = 0;
	/// The copy of each list on the submitting thread, the wait for the queue's lock included.
	PrescanTimeFigures copy_us;
	/// The push of each job inside the order gate, the wait for the queue's lock included.
	PrescanTimeFigures push_us;
	/// How old each list was when the scanner thread took it.
	PrescanTimeFigures lag_ms;
};

/// The report's view of the prescan's scanner thread, as plain numbers since this library cannot see
/// the scanner.
struct PrescanScannerReport {
	PrescanScanReport  scan;
	PrescanQueueReport queue;
	/// Jobs the thread finished, lists older than the lag bound it did not walk, and 1 once it stopped
	/// because the preparer had.
	uint64_t jobs                = 0;
	uint64_t lag_skipped         = 0;
	uint64_t stopped_by_preparer = 0;
	uint64_t wakes               = 0;
	/// The calls the walks made to the reader of guest memory, the longest, and those over a millisecond.
	uint64_t read_calls    = 0;
	uint64_t read_max_us   = 0;
	uint64_t read_over_1ms = 0;
	/// What became of the dispatches offered to the preparer.
	uint64_t offers_posted  = 0;
	uint64_t offers_skipped = 0;
	uint64_t offers_dropped = 0;
};

/// One part of the prescan_graphics block (experimental), as named groups of named numbers, so the
/// prescan's figures can change without this library knowing each one. A whole number is written
/// as an integer and any other as a number rounded to three places.
struct PrescanGraphicsReport {
	struct Group {
		std::string                                  name;
		std::vector<std::pair<std::string, double>> figures;
		/// Named arrays of the group, such as a timeline, written as JSON arrays beside the figures.
		std::vector<std::pair<std::string, std::vector<double>>> series;
	};
	std::vector<Group> groups;
};

/// The most graphics prescan miss rows the report keeps. Past it a miss is only counted as dropped.
inline constexpr size_t PRESCAN_MISS_ROW_LIMIT = 512;

/// The scan or job outcome of a miss row whose shader address the prescan has no sighting of, because
/// the walk never reached a draw that used it or the sightings were not kept.
inline constexpr std::string_view PRESCAN_OUTCOME_UNKNOWN = "unknown";
/// The scan or job outcome of a miss row whose address was sighted, but not by that side.
inline constexpr std::string_view PRESCAN_OUTCOME_NONE = "none";

/// One shader the GPU thread compiled itself although the graphics prescan was on (experimental). stage
/// is ps or vs, id the module id as ps and vs are written elsewhere, hash the program's code hash and
/// kind late, id_differs or unpredicted. Every kind carries what the prescan sighted of the shader's
/// code address, not only unpredicted. seen says it sighted a draw using that address, scan lists
/// every verdict the walk gave such draws, empty when only the pipeline cache noted it and just
/// PRESCAN_OUTCOME_UNKNOWN when nothing was sighted, and job is what the pipeline cache last did with
/// one. The ago times run back from the miss to the first sighting, the first queued job and the last
/// posted module of that address, and are -1 when that did not happen. flip is stamped by the report
/// when the row is added.
struct PrescanMissRow {
	std::string              stage;
	uint64_t                 id   = 0;
	uint64_t                 hash = 0;
	std::string              kind;
	int64_t                  at_ns = NO_TIME_NS;
	bool                     seen  = false;
	std::vector<std::string> scan {std::string(PRESCAN_OUTCOME_UNKNOWN)};
	std::string              job = std::string(PRESCAN_OUTCOME_UNKNOWN);
	double                   first_seen_ago_ms = -1.0;
	double                   offered_ago_ms    = -1.0;
	double                   posted_ago_ms     = -1.0;
	int64_t                  flip              = 0;
};

/// Who fills a part of the prescan_graphics block. The scanner knows the draws it walked and offered,
/// and the pipeline cache what its workers and the GPU thread made of them.
enum class GraphicsPrescanPart : uint8_t { Scanner, Cache, Count };
inline constexpr size_t GRAPHICS_PRESCAN_PART_COUNT = static_cast<size_t>(GraphicsPrescanPart::Count);

/// How the preparer's state answered the dispatches it was asked about. The names are those of the
/// state's own counts.
struct PrescanStateReport {
	uint64_t posted       = 0;
	uint64_t busy         = 0;
	uint64_t capped       = 0;
	uint64_t settled      = 0;
	uint64_t uncertain    = 0;
	uint64_t dropped_full = 0;
};

/// What the preparer counted, under the names of its own counts.
struct PrescanPreparerReport {
	uint64_t items                 = 0;
	uint64_t no_map_entry          = 0;
	uint64_t zero_size             = 0;
	uint64_t indirect_thread_dims  = 0;
	uint64_t bad_item              = 0;
	uint64_t backed_off            = 0;
	uint64_t blocked               = 0;
	uint64_t plan_requests         = 0;
	uint64_t compile_requests      = 0;
	uint64_t code_unreadable       = 0;
	uint64_t code_changed          = 0;
	uint64_t plan_failed           = 0;
	uint64_t has_calls             = 0;
	uint64_t srt_faults            = 0;
	uint64_t materialize_failed    = 0;
	uint64_t indirect_args_unknown = 0;
	/// Snapshots refused before they were sent, by the name of the reason.
	std::vector<std::pair<std::string, uint64_t>> invalid;
	uint64_t variant_modules     = 0;
	uint64_t known_variants      = 0;
	uint64_t capped_variants     = 0;
	uint64_t known_skipped       = 0;
	uint64_t heap_image_variants = 0;
	/// The new variants the prescan can stand for, which leaves out those whose image table the GPU
	/// thread's heap may take, since the runtime then builds a module the preparer's never matches.
	uint64_t covered_variants = 0;
	uint64_t known_modules    = 0;
	uint64_t known_dropped    = 0;
	uint64_t refused_sends    = 0;
	uint64_t never_retry      = 0;
	uint64_t blamed           = 0;
	uint64_t dropped          = 0;
	uint64_t compile_failed   = 0;
	uint64_t key_mismatch     = 0;
	uint64_t decode_failed    = 0;
	uint64_t record_mismatch  = 0;
	uint64_t adopted          = 0;
	uint64_t taken            = 0;
	uint64_t late_duplicates  = 0;
	uint64_t taken_dropped    = 0;
	uint64_t plans_dropped    = 0;
	uint64_t forgotten        = 0;
	uint64_t verdicts_evicted = 0;
	bool     failed           = false;
	bool     lane_off         = false;
};

/// The round trips of one kind of request to the helper, in milliseconds.
struct PrescanRttFigures {
	uint64_t count  = 0;
	double   p50_ms = 0.0;
	double   p90_ms = 0.0;
	double   max_ms = 0.0;
};

/// What the supervisor of the helper's child process counted. The pipe names and process ids are
/// left out, since they only tell one run's processes apart.
struct PrescanChildReport {
	uint64_t          spawns               = 0;
	uint64_t          failed_spawns        = 0;
	uint64_t          crashes              = 0;
	uint64_t          hangs                = 0;
	uint64_t          protocol_errors      = 0;
	uint64_t          failures             = 0;
	uint64_t          never_retry          = 0;
	uint64_t          never_retry_programs = 0;
	bool              lane_off             = false;
	std::string       lane_off_reason;
	uint64_t          submitted          = 0;
	uint64_t          answered           = 0;
	uint64_t          blamed             = 0;
	uint64_t          dropped            = 0;
	uint64_t          resent             = 0;
	uint64_t          refused            = 0;
	uint64_t          late_replies       = 0;
	uint64_t          unexpected_replies = 0;
	uint64_t          callback_failures  = 0;
	uint64_t          peak_outstanding   = 0;
	std::string       priority;
	PrescanRttFigures plan_rtt;
	PrescanRttFigures compile_rtt;
};

/// How the drain of generated modules treated the prescan's modules. The drain's timing belongs to
/// no producer, since one drain takes both producers' modules under one budget, so it is given for
/// both together, and it is the same timing the generator's block shows when that block exists.
struct PrescanDrainReport {
	uint64_t adopted = 0;
	/// The drain's refusals as duplicates. The preparer's late_duplicates counts the notes of them
	/// that reached it, which a full ring or a stop can drop, so the two need not be equal.
	uint64_t duplicates      = 0;
	uint64_t plan_mismatch   = 0;
	uint64_t failed_installs = 0;
	uint64_t dropped_stopped = 0;
	double   item_max_ms     = 0.0;
	/// The timing of every drain, of the generator's modules and the prescan's alike.
	uint64_t both_calls      = 0;
	uint64_t both_over_1ms   = 0;
	uint64_t both_over_4ms   = 0;
	double   both_max_ms     = 0.0;
	double   both_total_ms   = 0.0;
	double   both_lag_ms_max = 0.0;
};

/// What the prescan's pipeline builds did. in_scope, in_scope_ms, in_hold and in_hold_ms are upper
/// bounds on the overlap with a stop of the GPU thread and with a hold of guest audio, and nothing
/// may gate on them. They count worker builds only, so worker_builds and total_ms are what they are
/// measured against, while builds also counts the builds the GPU thread ran itself. The final
/// figures are taken when the pipeline cache stops its intake, before its warm sets stop, so builds
/// still running then are left out, at most PRESCAN_BUILD_WORKERS of them. Those finish during
/// teardown, and their overlaps would have nothing to do with gameplay.
struct PrescanBuildsReport {
	uint64_t builds        = 0;
	uint64_t worker_builds = 0;
	uint64_t in_scope      = 0;
	uint64_t in_hold       = 0;
	double   total_ms      = 0.0;
	double   in_scope_ms   = 0.0;
	double   in_hold_ms    = 0.0;
};

/// What the prescan did to the GPU thread's own stops. in_flight_at_miss_only counts the GPU
/// thread's own compiles charged to PrescanInFlight only by the check at the lookup's miss. They
/// are inline compiles and not prescan builds, so they are kept apart from the builds.
struct PrescanHoldsReport {
	uint64_t in_flight_at_miss_only = 0;
};

/// Why the GPU thread compiled compute pipelines itself with the prescan on, by name, with every
/// name present even at 0. These count compiles, one per pipeline Scope and not per translation,
/// while the holds by cause count holds of guest audio. A cause's holds are at most its compiles,
/// and for RequiredCompute those are module_present + stopped + never_posted. Nested Scopes charge
/// a hold to the oldest one's cause. inline_compiles is the sum of by_reason. never_posted_deep,
/// never_posted_indirect and never_posted_async are attributes of the dispatch, not proof of why it
/// was missed.
///
/// How to read a run. A run without the prescan charges every inline compute hold to
/// required_compute, so a prescan run is compared with it through the sum of required_compute,
/// prescan_in_flight, prescan_refused and the three prescan_never_posted causes, never through
/// required_compute alone. RequiredCompute time stands for never-posted modules only when stopped
/// is 0 and module_present is small, since those share the cause. Verdicts are per unit, so until
/// misses record whether the scanner saw the dispatch, refused includes a specialization the
/// scanner never posted of a unit it did post. It also includes a unit between its plan's answer
/// and its compile's send, which reads admitted. A large verdicts_evicted overstates never_posted
/// and understates refused. refused_by keeps the first throttle a unit met, since a throttle never
/// replaces another. built_undrained is an upper bound on the refused misses with a built verdict
/// whose module was still in the generated inbox, since the generator posts there too.
struct PrescanMissesReport {
	uint64_t                                      inline_compiles = 0;
	std::vector<std::pair<std::string, uint64_t>> by_reason;
	std::vector<std::pair<std::string, uint64_t>> refused_by;
	uint64_t                                      built_undrained = 0;
};

/// What became of the store records of the prescan's modules, which wait for each module's first
/// use. While the cache runs these are the figures the last drain published, so a first use after
/// the last drain shows only once a later drain, or the stop, publishes again.
struct PrescanRecordsReport {
	uint64_t stored_first_use    = 0;
	uint64_t retained_peak_bytes = 0;
	uint64_t dropped_over_cap    = 0;
	uint64_t unused_at_exit      = 0;
};

/// The pipeline cache's part of the prescan block, as plain numbers since this library cannot see
/// the cache. The child figures exist only when the helper runs in a child process.
struct PrescanCacheReport {
	PrescanStateReport                state;
	PrescanPreparerReport             preparer;
	std::optional<PrescanChildReport> child;
	PrescanDrainReport                drain;
	PrescanBuildsReport               builds;
	PrescanHoldsReport                holds;
	PrescanMissesReport               misses;
	PrescanRecordsReport              records;
};

/// Deferred draws one S2 rule refused, and their waits for their pipelines summed.
struct DeferredRefusal {
	uint32_t draws     = 0;
	double   waited_ms = 0.0;
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
	// The view-history counts, indexed by ViewHistoryCount.
	std::array<uint64_t, VIEW_HISTORY_COUNT>                 view_history {};
	// Colour targets of allowed draws that passed only on their two completed frames.
	uint64_t                                                 color_completed_frames = 0;
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
	// With --async-pipelines, deferred draws the S2 rules refused, by the refusing rule, with the same
	// wait as above. They show why a deferral could not have been a skip.
	std::map<std::string, DeferredRefusal, std::less<>>      deferred_refused_skip;
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
	// Lead of the inline compute compiles in play, which the JSON key submit_lead_summary shows.
	SubmitLeadSummary                                        submit_leads;
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
	/// only adds to the overflow count, so the index cannot wrap. The warm builds running per lane
	/// and the generator's liveness are stored in the slot before its time is published with
	/// release.
	void AddAudioUnderrun() noexcept;
	/// The underruns recorded so far. Slots that a racing call reserved and has not written yet are
	/// left out, and only a slot with a time has its counts read.
	[[nodiscard]] AudioUnderruns AudioUnderrunsSoFar() const;
	/// Notes the submission the GPU thread is processing. count is true when it should join the
	/// submission totals, which the caller does once per command list.
	void BeginSubmission(const SubmissionStamp& stamp, bool count);
	/// Sets the point in the run that start_ms is counted from, on the NowNs clock.
	void SetEpoch(int64_t epoch_ns);
	/// The run's epoch on the NowNs clock, or NO_TIME_NS before SetEpoch, for a source that keeps its
	/// own times from the run start.
	[[nodiscard]] int64_t EpochNs() const;
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
	/// Adds one to a view-history count.
	void AddViewHistoryCount(ViewHistoryCount which);
	/// Counts `targets` colour targets of an allowed draw that passed only on their two completed
	/// frames, the rule --async-pipelines-prefer-skip adds, where the rule of this frame and the last
	/// refused them.
	void AddColorCompletedFrames(uint32_t targets);
	/// Records a draw that missed its pipeline and was refused a skip, and how long it waited.
	void AddRefusal(std::string_view reason, double blocked_ms);
	/// Records one resolve of deferred draws.
	void AddDeferredResolve(const DeferredResolve& resolve);
	/// Records a draw that --deferred-pipeline-draws could not defer, against the reason.
	void AddDeferredIneligible(std::string_view reason);
	/// Records a draw that could not be deferred and waited for its pipeline, and adds the wait to
	/// its reason's total. The flip field is ignored and replaced by the current flip.
	void AddDeferredIneligibleWait(const IneligibleWait& wait);
	/// Adds a graphics prescan miss row and stamps its flip, or past PRESCAN_MISS_ROW_LIMIT rows only
	/// counts it as dropped.
	void AddPrescanMissRow(const PrescanMissRow& row);
	/// The graphics prescan miss rows so far, in the order they were added, and how many were dropped.
	[[nodiscard]] std::vector<PrescanMissRow> PrescanMissRows() const;
	[[nodiscard]] uint64_t                    PrescanMissRowsDropped() const;
	/// Records a deferred draw that S2 would have skipped, and how long it waited for its pipeline.
	void AddDeferredWouldSkip(double waited_ms);
	/// Records a deferred draw that S2 refused under `reason`, and how long it waited for its pipeline.
	void AddDeferredRefusedSkip(std::string_view reason, double waited_ms);
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
	/// time, which also catches holds that land outside every deferred-draw window. It comes from
	/// the hold tracker's own counter, and `holds` is the same time split by cause, so a reader can
	/// check that the causes add up to it. A caller with no breakdown passes an empty one and gets
	/// zero in every cause.
	[[nodiscard]] std::string ToJson(const Metadata& metadata, uint64_t audio_holds,
	                                 double run_seconds, double audio_held_ms,
	                                 const GpuStall::HoldBreakdown& holds) const;
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

	/// Registers the function the writer calls for the scan_differential block. With no source and no
	/// final figures the report has no such block, which is how a run without --scan-differential looks.
	/// The writer holds the scan differential lock while it calls, so the owner replaces the source with
	/// SetScanDifferentialFinal before what the function reads is destroyed. The function must not call
	/// back into these functions.
	void SetScanDifferentialSource(std::function<ScanDifferentialReport()> source);
	/// Stores the final figures and removes the source, in one step under the lock.
	void SetScanDifferentialFinal(ScanDifferentialReport final_figures);

	/// Registers the function the writer calls for the scanner part of the prescan block. With no
	/// source and no final figures the report has no prescan block, which is how a run without the
	/// prescan looks. The writer holds the prescan scan lock while it calls, so the owner replaces the
	/// source with SetPrescanScanFinal before what the function reads is destroyed. The function must
	/// not call back into these functions.
	void SetPrescanScanSource(std::function<PrescanScannerReport()> source);
	/// Stores the final figures and removes the source, in one step under the lock.
	void SetPrescanScanFinal(PrescanScannerReport final_figures);
	/// Whether a scanner source is registered, so a test can tell that its owner handed over before it
	/// was destroyed without calling a source that may be gone.
	[[nodiscard]] bool HoldsPrescanScanSource() const;
	/// Whether final scanner figures are stored.
	[[nodiscard]] bool HoldsPrescanScanFinal() const;

	/// Registers the function the writer calls for the pipeline cache's part of the prescan block.
	/// There is one slot, so a second cache's source replaces the first's, and either cache's final
	/// figures then replace whichever source is held. Either part alone makes the block, and with
	/// both their keys share it. The writer holds the prescan cache lock while it calls, so the
	/// owner replaces the source with SetPrescanCacheFinal before what the function reads is
	/// destroyed. The function must not call back into these functions.
	void SetPrescanCacheSource(std::function<PrescanCacheReport()> source);
	/// Stores the final figures and removes the source, in one step under the lock.
	void SetPrescanCacheFinal(PrescanCacheReport final_figures);
	/// Whether a pipeline cache source is registered, so a test can tell that its owner handed over
	/// without calling a source that may be gone.
	[[nodiscard]] bool HoldsPrescanCacheSource() const;
	/// Whether final pipeline cache figures are stored.
	[[nodiscard]] bool HoldsPrescanCacheFinal() const;

	/// Registers the function the writer calls for one part of the prescan_graphics block (experimental).
	/// With no part registered the report has no such block. The writer holds the block's lock while
	/// it calls, so the owner replaces the source with SetGraphicsPrescanFinal before what the
	/// function reads is destroyed. The function must not call back into these functions.
	void SetGraphicsPrescanSource(GraphicsPrescanPart part, std::function<PrescanGraphicsReport()> source);
	/// Stores the final figures of a part and removes its source, in one step under the lock.
	void SetGraphicsPrescanFinal(GraphicsPrescanPart part, PrescanGraphicsReport final_figures);
	/// Whether a part has a source registered, for tests.
	[[nodiscard]] bool HoldsGraphicsPrescanSource(GraphicsPrescanPart part) const;

private:
	// The scanner's figures from the source or the final ones, or nothing when there are neither. The
	// source is called with the prescan scan lock held.
	[[nodiscard]] std::optional<PrescanScannerReport> PrescanScanFigures() const;
	// The pipeline cache's figures from the source or the final ones, or nothing when there are
	// neither. The source is called with the prescan cache lock held.
	[[nodiscard]] std::optional<PrescanCacheReport> PrescanCacheFigures() const;
	// The groups of every registered part of the prescan_graphics block in part order, or nothing
	// when no part has a source or final figures. Sources are called with the block's lock held.
	[[nodiscard]] std::optional<PrescanGraphicsReport> GraphicsPrescanFigures() const;
	// Removes every registered source and its final figures, and the generation's inactive reason,
	// for Reset.
	void ResetSources();
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

	// Guards the two scan differential members. It is taken on its own, never with m_mutex held.
	mutable std::mutex                           m_scan_differential_mutex;
	std::function<ScanDifferentialReport()>      m_scan_differential_source;
	std::optional<ScanDifferentialReport>        m_scan_differential_final;

	// Guards the two prescan scan members. It is taken on its own, never with another lock of the
	// report held, and the source takes only the scanner's own locks, which never wait on the report.
	mutable std::mutex                           m_prescan_scan_mutex;
	std::function<PrescanScannerReport()>        m_prescan_scan_source;
	std::optional<PrescanScannerReport>          m_prescan_scan_final;

	// Guards the two prescan cache members. It is taken on its own, never with another lock of the
	// report held. The source takes the cache's generation state lock, the preparer's published
	// lock and the helper supervisor's lock, and none of their holders ever waits on the report.
	mutable std::mutex                           m_prescan_cache_mutex;
	std::function<PrescanCacheReport()>          m_prescan_cache_source;
	std::optional<PrescanCacheReport>            m_prescan_cache_final;

	// Guards the prescan_graphics parts. It is taken on its own, never with another lock of the report
	// held, and the sources read only atomics and their owners' own stats locks.
	mutable std::mutex m_graphics_prescan_mutex;
	std::array<std::function<PrescanGraphicsReport()>, GRAPHICS_PRESCAN_PART_COUNT> m_graphics_prescan_source;
	std::array<std::optional<PrescanGraphicsReport>, GRAPHICS_PRESCAN_PART_COUNT>   m_graphics_prescan_final;

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
	std::array<uint64_t, VIEW_HISTORY_COUNT>      m_view_history {};
	uint64_t                                      m_color_completed_frames = 0;
	std::vector<Refusal>                          m_refusals;
	std::vector<DeferredResolve>                  m_deferred_resolves;
	std::vector<uint32_t>                         m_deferred_resolves_per_flip;
	uint32_t                                      m_deferred_resolves_this_flip = 0;
	std::vector<std::pair<std::string, uint32_t>> m_deferred_ineligible;
	std::vector<IneligibleWait>                   m_deferred_ineligible_waits;
	std::vector<PrescanMissRow>                   m_prescan_miss_rows;
	uint64_t                                      m_prescan_miss_rows_dropped = 0;
	std::vector<std::pair<std::string, double>>   m_deferred_ineligible_waited_ms;
	uint32_t                                      m_deferred_would_skip_draws = 0;
	double                                        m_deferred_would_skip_waited_ms = 0.0;
	std::map<std::string, DeferredRefusal, std::less<>> m_deferred_refused_skip;
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
	// What was running at each underrun, stored before its time so that a reader of the time sees
	// them. Warm counts saturate at the uint16 range.
	std::array<std::atomic<uint16_t>, AUDIO_UNDERRUN_CAPACITY> m_underrun_warm_front {};
	std::array<std::atomic<uint16_t>, AUDIO_UNDERRUN_CAPACITY> m_underrun_warm_back {};
	std::array<std::atomic<uint8_t>, AUDIO_UNDERRUN_CAPACITY>  m_underrun_child_alive {};
	std::array<std::atomic<uint16_t>, AUDIO_UNDERRUN_CAPACITY> m_underrun_prescan_running {};
	std::atomic<size_t>                                        m_underrun_next {0};
	std::atomic<uint64_t>                                      m_underrun_overflow {0};
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
inline std::atomic<int32_t>  g_warm_front_running {0};
inline std::atomic<int32_t>  g_warm_back_running {0};
inline std::atomic<bool>     g_child_alive {false};
inline std::atomic<int32_t>  g_prescan_running {0};
inline std::atomic<uint32_t> g_warm_workers {0};

/// A counter never reads below zero. Scopes only add and then take back what they added, so a
/// racing reader cannot see a negative value. Only a test setter that overwrites a counter while
/// scopes are live can drive it below zero, and the clamp keeps that from showing as a huge count.
[[nodiscard]] inline uint32_t ReadCount(const std::atomic<int32_t>& counter) noexcept {
	const auto running = counter.load(std::memory_order_relaxed);
	return running > 0 ? static_cast<uint32_t>(running) : 0;
}
} // namespace Detail

/// The queue lane a warm build runs on. This is the report's own type, since common cannot include
/// the graphics headers. The pipeline cache maps its PipelineBuildQueue::Lane onto it. Other is for
/// a build that is not on the warm queue, and counts toward the total only.
enum class WarmLane : uint8_t { Front, Back, Other };

/// How many background warm-up builds are running now, in every lane.
[[nodiscard]] inline uint32_t WarmBuildsRunning() noexcept {
	return Detail::ReadCount(Detail::g_warm_running);
}

/// How many warm-up builds from the front lane are running now.
[[nodiscard]] inline uint32_t WarmFrontBuildsRunning() noexcept {
	return Detail::ReadCount(Detail::g_warm_front_running);
}

/// How many warm-up builds from the back lane are running now.
[[nodiscard]] inline uint32_t WarmBackBuildsRunning() noexcept {
	return Detail::ReadCount(Detail::g_warm_back_running);
}

/// Counts one warm-up build as running for as long as it lives, when the report is on. A worker
/// makes it around the build, and the rows of worker compiles read the count when they begin. It
/// raises the total and its lane's count, and lowers both on exit.
class WarmBuildScope {
public:
	explicit WarmBuildScope(WarmLane lane = WarmLane::Other) noexcept
	    : m_active(Enabled()), m_lane(lane) {
		if (m_active) {
			Detail::g_warm_running.fetch_add(1, std::memory_order_relaxed);
			if (auto* lane_count = LaneCount()) {
				lane_count->fetch_add(1, std::memory_order_relaxed);
			}
		}
	}
	~WarmBuildScope() {
		if (m_active) {
			if (auto* lane_count = LaneCount()) {
				lane_count->fetch_sub(1, std::memory_order_relaxed);
			}
			Detail::g_warm_running.fetch_sub(1, std::memory_order_relaxed);
		}
	}
	WarmBuildScope(const WarmBuildScope&)            = delete;
	WarmBuildScope& operator=(const WarmBuildScope&) = delete;

private:
	/// The counter of this scope's lane, or null for a build outside both lanes.
	[[nodiscard]] std::atomic<int32_t>* LaneCount() const noexcept {
		switch (m_lane) {
			case WarmLane::Front: return &Detail::g_warm_front_running;
			case WarmLane::Back: return &Detail::g_warm_back_running;
			case WarmLane::Other: break;
		}
		return nullptr;
	}

	bool     m_active;
	WarmLane m_lane;
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

/// The figure behind the prescan_running underrun tag, which is the pre-scan work in progress now.
/// It counts the pre-scan's pipeline builds running on a worker, the Plan and Compile requests sent
/// to the helper whose outcome the preparer has not handled yet, and 1 while a preparer pass runs.
/// It measures work and not whether the helper process is alive, so an idle helper counts 0, and it
/// is deliberately not folded into GeneratorChildAlive. The scanner is deliberately left out. Its
/// walk runs on every submission while the pre-scan is on and is bounded in time, so counting it
/// would tag nearly every underrun and say nothing about the helper or the builds. Each unit is a
/// PrescanRunningScope, and only scopes made while the report is on count.
[[nodiscard]] inline uint32_t PrescanRunning() noexcept {
	return Detail::ReadCount(Detail::g_prescan_running);
}

/// Overwrites the prescan_running figure, for a test that needs a value no real work would give.
/// Live scopes still add and subtract on top of it, so a test calls it with no pre-scan running.
inline void SetPrescanRunningForTest(uint32_t in_progress) noexcept {
	const auto clamped = std::min<uint32_t>(in_progress, std::numeric_limits<int32_t>::max());
	Detail::g_prescan_running.store(static_cast<int32_t>(clamped), std::memory_order_relaxed);
}

/// Counts one unit of pre-scan work toward PrescanRunning for as long as it lives, when the report
/// is on. It is move-only, so a pending request can carry its unit through a map, and a moved-from
/// scope gives nothing back. Whether it counts is fixed when it is made, so it always gives back
/// exactly what it took even if the report is turned off meanwhile.
class PrescanRunningScope {
public:
	PrescanRunningScope() noexcept
	    : m_active(Enabled()) {
		if (m_active) {
			Detail::g_prescan_running.fetch_add(1, std::memory_order_relaxed);
		}
	}
	~PrescanRunningScope() { Release(); }
	PrescanRunningScope(PrescanRunningScope&& other) noexcept
	    : m_active(std::exchange(other.m_active, false)) {}
	PrescanRunningScope& operator=(PrescanRunningScope&& other) noexcept {
		if (this != &other) {
			Release();
			m_active = std::exchange(other.m_active, false);
		}
		return *this;
	}
	PrescanRunningScope(const PrescanRunningScope&)            = delete;
	PrescanRunningScope& operator=(const PrescanRunningScope&) = delete;

private:
	/// Gives the unit back once.
	void Release() noexcept {
		if (m_active) {
			Detail::g_prescan_running.fetch_sub(1, std::memory_order_relaxed);
			m_active = false;
		}
	}

	bool m_active;
};

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

/// Records one graphics prescan miss row.
inline void AddPrescanMissRow(const PrescanMissRow& row) {
	if (Enabled()) {
		Global().AddPrescanMissRow(row);
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

/// Records a deferred draw that S2 refused, by the refusing rule, and how long it waited.
inline void AddDeferredRefusedSkip(std::string_view reason, double waited_ms) {
	if (Enabled()) {
		Global().AddDeferredRefusedSkip(reason, waited_ms);
	}
}

/// Records skipped draws whose target was then read as a texture, a sign that a skip left damage.
inline void AddSampledAfterSkip(std::string_view target_class, uint32_t draws) {
	if (Enabled()) {
		Global().AddSampledAfterSkip(target_class, draws);
	}
}

/// Adds one to a view-history count when the report is on.
inline void AddViewHistoryCount(ViewHistoryCount which) {
	if (Enabled()) {
		Global().AddViewHistoryCount(which);
	}
}

/// Counts colour targets that passed only on their completed frames when the report is on. A draw
/// with none takes no lock.
inline void AddColorCompletedFrames(uint32_t targets) {
	if (targets != 0 && Enabled()) {
		Global().AddColorCompletedFrames(targets);
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
