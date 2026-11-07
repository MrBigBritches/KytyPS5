#include "common/firstRunReport.h"

#include "common/file.h"
#include "common/gpuStall.h"
#include "common/logging/log.h"
#include "common/stringUtils.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <exception>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <random>

namespace Common::FirstRunReport {

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

double Round3(double value) {
	return std::round(value * ROUNDING_SCALE) / ROUNDING_SCALE;
}

double ElapsedMs(std::chrono::steady_clock::time_point start) {
	return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
	    .count();
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

void Report::AddCompile(Kind kind, uint64_t id, uint64_t id_ps, double create_ms, Thread thread) {
	std::lock_guard lock(m_mutex);
	m_compiles.push_back({.kind      = kind,
	                      .id        = id,
	                      .id_ps     = id_ps,
	                      .create_ms = create_ms,
	                      .phase     = m_phase,
	                      .thread    = thread,
	                      .flip      = m_flip});
	if (thread == Thread::Gpu && m_phase == Phase::Play) {
		AddBlockedLocked(kind, create_ms);
	}
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

double Report::BlockedMs(Kind kind) const {
	std::lock_guard lock(m_mutex);
	return m_blocked_ms[static_cast<size_t>(kind)];
}

void Report::AddStampFailure() {
	std::lock_guard lock(m_mutex);
	m_stamp_failures++;
}

void Report::Reset() {
	std::lock_guard lock(m_mutex);
	m_phase             = Phase::Play;
	m_flip              = 0;
	m_flips             = 0;
	m_stamp_failures    = 0;
	m_boot              = {};
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
	m_refusals.clear();
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

} // namespace

Summary Report::Summarize() const {
	std::lock_guard lock(m_mutex);
	Summary         summary;
	summary.blocked_graphics_ms = m_blocked_ms[static_cast<size_t>(Kind::Graphics)];
	summary.blocked_compute_ms  = m_blocked_ms[static_cast<size_t>(Kind::Compute)];
	summary.translate_ms        = m_translate_ms;
	summary.translate_gpu_graphics_ms = m_translate_gpu_graphics_ms;
	summary.flips               = m_flips;
	summary.stamp_failures      = m_stamp_failures;
	summary.skips               = m_skips;
	summary.skips_by_target     = m_skips_by_target;
	summary.sampled_after_skip_by_target = m_sampled_after_skip;
	summary.refusals                     = m_refusals;

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

nlohmann::json CompileToJson(const CompileEntry& compile) {
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
	        {"refused", refused}};
}

nlohmann::json BootToJson(const BootInfo& boot) {
	return {{"warm_up_ms", Round3(boot.warm_up_ms)},
	        {"programs", boot.programs},
	        {"modules", boot.modules},
	        {"pipelines", boot.pipelines}};
}

} // namespace

std::string Report::ToJson(const Metadata& metadata, uint64_t audio_holds,
                           double run_seconds) const {
	const auto summary  = Summarize();
	const auto compiles = Compiles();
	const auto boot     = Boot();

	nlohmann::json flags = nlohmann::json::object();
	for (const auto& [name, on]: metadata.flags) {
		flags[name] = on;
	}
	nlohmann::json compile_rows = nlohmann::json::array();
	for (const auto& compile: compiles) {
		compile_rows.push_back(CompileToJson(compile));
	}
	nlohmann::json ready_rows = nlohmann::json::array();
	for (const auto& ready: Readies()) {
		ready_rows.push_back(ReadyToJson(ready));
	}
	const nlohmann::json report = {
	    {"title_id", metadata.title_id},
	    {"build", metadata.build},
	    {"flags", flags},
	    {"cold", metadata.cold},
	    {"cold_stamp_failures", summary.stamp_failures},
	    {"partial", metadata.partial},
	    {"boot", BootToJson(boot)},
	    {"play", PlayToJson(summary)},
	    {"pauses", PausesToJson(summary)},
	    {"compiles", compile_rows},
	    {"ready", ready_rows},
	    {"skips", SkipsToJson(summary)},
	    {"audio_holds", audio_holds},
	    {"flips", summary.flips},
	    {"run_seconds", Round3(run_seconds)},
	};
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
	const auto text = Global().ToJson(metadata, GpuStall::HoldCount(), run_seconds);
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
		m_start = std::chrono::steady_clock::now();
	}
}

CompileScope::~CompileScope() {
	if (m_active) {
		Global().AddCompile(m_kind, m_id, m_id_ps, ElapsedMs(m_start), m_thread);
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
