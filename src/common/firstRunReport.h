#ifndef KYTY_COMMON_FIRST_RUN_REPORT_H_
#define KYTY_COMMON_FIRST_RUN_REPORT_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
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
	/// Records a miss-path translation. Only GPU-thread time during play counts as blocked.
	void AddTranslate(Kind kind, std::string_view stage, uint64_t hash, double ms, Thread thread);
	/// Records a driver compile. Only GPU-thread time during play counts as blocked. start_ns is
	/// when the compile began, on the NowNs clock, and links GPU-thread rows to the submission.
	/// Without it the row is left unlinked.
	void AddCompile(Kind kind, uint64_t id, uint64_t id_ps, double create_ms, Thread thread,
	                int64_t start_ns = NO_TIME_NS);
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
	/// The GPU-thread time blocked so far in play for the kind.
	[[nodiscard]] double BlockedMs(Kind kind) const;
	/// Records a module that --cold-pipelines could not stamp, so it compiled unstamped.
	void AddStampFailure();
	/// Forgets everything recorded. For tests, which share the process-wide report.
	void Reset();
	/// What the boot warm-up stored.
	[[nodiscard]] BootInfo Boot() const;

	/// Works out the totals, histogram and top lists from what was recorded so far.
	[[nodiscard]] Summary Summarize() const;
	/// Renders the whole report as JSON text.
	[[nodiscard]] std::string ToJson(const Metadata& metadata, uint64_t audio_holds,
	                                 double run_seconds) const;
	/// Copy of the compile rows, for tests.
	[[nodiscard]] std::vector<CompileEntry> Compiles() const;
	/// Copy of the ready rows, for tests.
	[[nodiscard]] std::vector<ReadyEntry> Readies() const;

private:
	// Moves the pause being merged to the finished ones when it belongs to another flip. Caller
	// holds the lock.
	void FlushPendingPauseLocked(int64_t current_flip);
	// Adds GPU-thread blocking to the pause of the current flip. Caller holds the lock.
	void AddBlockedLocked(Kind kind, double ms);
	// Pauses including the one still being merged. Caller holds the lock.
	[[nodiscard]] std::vector<Pause> AllPausesLocked() const;

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
/// construction to destruction and records it.
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
