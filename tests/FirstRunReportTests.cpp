#include "common/firstRunReport.h"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>
#include <string>
#include <vector>

namespace {

using Common::FirstRunReport::Kind;
using Common::FirstRunReport::Phase;
using Common::FirstRunReport::Report;
using Common::FirstRunReport::SkipKind;
using Common::FirstRunReport::Thread;

// Every duration here is a made-up number handed to the report, which never reads a clock, so no
// case depends on how fast the machine runs.
constexpr double kEpsilon = 1e-9;

void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "FirstRunReportTests: failed: %s\n", message);
    std::abort();
  }
}

bool Near(double left, double right) { return std::abs(left - right) < kEpsilon; }

// Consecutive blocking within one flip is one pause, and the next flip starts another. A baseline
// of the same blocking spread over distinct flips shows that only the flip number merges them.
void TestPausesMergeByFlip() {
  Report merged;
  merged.OnFlip(5);
  merged.AddTranslate(Kind::Graphics, "ps", 1, 300.0, Thread::Gpu);
  merged.AddCompile(Kind::Graphics, 1, 2, 200.0, Thread::Gpu);
  merged.AddCompile(Kind::Compute, 3, 0, 100.0, Thread::Gpu);
  merged.OnFlip(6);
  merged.AddCompile(Kind::Graphics, 4, 5, 700.0, Thread::Gpu);
  const auto summary = merged.Summarize();
  Check(summary.pause_count == 2, "same-flip blocking was not merged into one pause");
  Check(summary.longest_pauses.size() == 2 && Near(summary.longest_pauses[0].ms, 700.0) &&
            summary.longest_pauses[0].flip == 6 && summary.longest_pauses[0].events == 1,
        "the longest pause lost its flip or contents");
  Check(Near(summary.longest_pauses[1].ms, 600.0) && summary.longest_pauses[1].events == 3 &&
            summary.longest_pauses[1].flip == 5,
        "the merged pause did not sum its three blocking events");

  Report spread;
  for (int64_t flip = 1; flip <= 3; ++flip) {
    spread.OnFlip(flip);
    spread.AddCompile(Kind::Graphics, 1, 2, 200.0, Thread::Gpu);
  }
  Check(spread.Summarize().pause_count == 3, "blocking in different flips merged");
}

// A pause still being merged when the report is read still counts.
void TestPendingPauseCounts() {
  Report report;
  report.OnFlip(9);
  report.AddCompile(Kind::Compute, 1, 0, 120.0, Thread::Gpu);
  const auto summary = report.Summarize();
  Check(summary.pause_count == 1 && Near(summary.longest_pause_ms, 120.0),
        "the unfinished pause was left out");
}

// Only GPU-thread waiting during play is blocked. The same compile on a worker, or during boot,
// still shows up as a row and as translate time but never as blocked time or a pause.
void TestOnlyGpuThreadBlocks() {
  Report gpu;
  gpu.AddTranslate(Kind::Graphics, "vs", 7, 400.0, Thread::Gpu);
  gpu.AddCompile(Kind::Compute, 8, 0, 250.0, Thread::Gpu);
  const auto on_gpu = gpu.Summarize();
  Check(Near(on_gpu.blocked_graphics_ms, 400.0) && Near(on_gpu.blocked_compute_ms, 250.0),
        "GPU thread waiting was not split by kind");

  Report worker;
  worker.AddTranslate(Kind::Graphics, "vs", 7, 400.0, Thread::Worker);
  worker.AddCompile(Kind::Compute, 8, 0, 250.0, Thread::Worker);
  const auto off_gpu = worker.Summarize();
  Check(Near(off_gpu.blocked_graphics_ms, 0.0) && Near(off_gpu.blocked_compute_ms, 0.0) &&
            off_gpu.pause_count == 0,
        "worker thread time counted as blocked");
  Check(Near(off_gpu.translate_ms, 400.0) && worker.Compiles().size() == 1 &&
            worker.Compiles()[0].thread == Thread::Worker,
        "worker work was not recorded as worker work");

  Report boot;
  boot.SetPhase(Phase::Boot);
  boot.AddCompile(Kind::Graphics, 1, 2, 500.0, Thread::Gpu);
  boot.SetPhase(Phase::Play);
  boot.AddCompile(Kind::Graphics, 3, 4, 50.0, Thread::Gpu);
  const auto rows = boot.Compiles();
  Check(rows.size() == 2 && rows[0].phase == Phase::Boot && rows[1].phase == Phase::Play,
        "compile rows were not tagged with their phase");
  Check(Near(boot.Summarize().blocked_graphics_ms, 50.0), "boot compile counted as blocked");
}

// Waiting for a compile that a worker is running blocks the GPU thread, though no compile ran on
// it. Like any blocking it counts toward the kind's total and the flip's pause during play only,
// and it adds no compile row, because the worker recorded its own. The baseline is the same figure
// recorded as a worker compile counts for nothing.
void TestWaitingBlocksTheGpuThread() {
  Report waited;
  waited.OnFlip(3);
  waited.AddCompile(Kind::Graphics, 1, 2, 900.0, Thread::Worker);
  Check(Near(waited.Summarize().blocked_graphics_ms, 0.0), "the worker compile alone blocked");
  waited.AddWait(Kind::Graphics, 120.0);
  const auto summary = waited.Summarize();
  Check(Near(summary.blocked_graphics_ms, 120.0) && Near(summary.blocked_compute_ms, 0.0),
        "waiting was not counted as blocked graphics time");
  Check(summary.pause_count == 1 && Near(summary.longest_pause_ms, 120.0) &&
            summary.longest_pauses[0].flip == 3 && summary.longest_pauses[0].events == 1,
        "waiting was not merged into the pause of its flip");
  Check(waited.Compiles().size() == 1, "waiting added a compile row");

  Report boot;
  boot.SetPhase(Phase::Boot);
  boot.AddWait(Kind::Graphics, 120.0);
  Check(Near(boot.Summarize().blocked_graphics_ms, 0.0), "waiting during boot counted as blocked");
}

// A ready row keeps what the GPU thread learned when it took a background build, and is written
// to the file.
void TestReadyRows() {
  Report report;
  Check(report.Readies().empty(), "a new report had ready rows");
  report.AddReady(Kind::Graphics, 0x11, 0x22, 42.5, 3);
  const auto rows = report.Readies();
  Check(rows.size() == 1 && rows[0].id == 0x11 && rows[0].id_ps == 0x22 &&
            Near(rows[0].ready_ms, 42.5) && rows[0].skipped_draws == 3,
        "the ready row lost a field");
  const auto json = report.ToJson({}, 0, 1.0);
  for (const char *needle : {"\"ready\"", "\"ready_ms\": 42.5", "\"skipped_draws\": 3",
                             "\"vs\": \"0000000000000011\""}) {
    Check(json.find(needle) != std::string::npos, needle);
  }
  report.Reset();
  Check(report.Readies().empty(), "Reset kept the ready rows");
}

// Skipped draws that were sampled afterwards are summed per target class, written to the file, and
// forgotten by Reset. The baseline is a report with skips and no samples, which writes none.
void TestSampledAfterSkip() {
  Report quiet;
  quiet.AddSkip(SkipKind::Draw, "depth_only");
  Check(quiet.Summarize().sampled_after_skip_by_target.empty(), "skips alone were counted as sampled");
  Report report;
  report.AddSampledAfterSkip("depth", 2);
  report.AddSampledAfterSkip("color", 1);
  report.AddSampledAfterSkip("depth", 3);
  const auto summary = report.Summarize();
  Check(summary.sampled_after_skip_by_target.size() == 2 &&
            summary.sampled_after_skip_by_target[0].first == "depth" &&
            summary.sampled_after_skip_by_target[0].second == 5 &&
            summary.sampled_after_skip_by_target[1].second == 1,
        "sampled skips were not summed per target class");
  const auto json = report.ToJson({}, 0, 1.0);
  Check(json.find("\"sampled_after_skip_by_target_class\"") != std::string::npos &&
            json.find("\"depth\": 5") != std::string::npos,
        "the file lacks the sampled skip counts");
  report.Reset();
  Check(report.Summarize().sampled_after_skip_by_target.empty(), "Reset kept sampled skips");
}

// Refused draws are summed per rule with their blocked time, written to the file, and forgotten by
// Reset. The blocked total is readable while it grows, and the graphics translate that blocked the
// GPU thread is kept apart. A worker's translate, or one in boot, is in neither.
void TestRefusalsAndBlockedTotals() {
  Report report;
  report.AddRefusal("too_few_draws", 10.0);
  report.AddRefusal("writer", 1.5);
  report.AddRefusal("too_few_draws", 5.0);
  const auto summary = report.Summarize();
  Check(summary.refusals.size() == 2 && summary.refusals[0].reason == "too_few_draws" &&
            summary.refusals[0].draws == 2 && Near(summary.refusals[0].blocked_ms, 15.0) &&
            summary.refusals[1].draws == 1 && Near(summary.refusals[1].blocked_ms, 1.5),
        "refusals were not summed per rule");
  const auto json = report.ToJson({}, 0, 1.0);
  for (const char *needle : {"\"refused\"", "\"too_few_draws\"", "\"draws\": 2",
                             "\"blocked_ms\": 15.0"}) {
    Check(json.find(needle) != std::string::npos, needle);
  }
  report.Reset();
  Check(report.Summarize().refusals.empty(), "Reset kept refusals");

  Report totals;
  totals.AddTranslate(Kind::Graphics, "vs", 1, 40.0, Thread::Gpu);
  totals.AddTranslate(Kind::Graphics, "vs", 2, 7.0, Thread::Worker);
  totals.AddTranslate(Kind::Compute, "cs", 3, 9.0, Thread::Gpu);
  totals.AddCompile(Kind::Graphics, 1, 2, 25.0, Thread::Gpu);
  Check(Near(totals.BlockedMs(Kind::Graphics), 65.0) && Near(totals.BlockedMs(Kind::Compute), 9.0),
        "the blocked total was not readable per kind");
  Check(Near(totals.Summarize().translate_gpu_graphics_ms, 40.0),
        "the GPU-thread graphics translate was not kept apart from the others");
}

// Pauses fall in the highest bucket whose bound they reach, so each edge belongs to the upper
// bucket and anything below the first bound is counted apart.
void TestHistogramEdges() {
  struct Case {
    double   ms;
    int      bucket; // -1 means under the first bound.
  };
  constexpr std::array<Case, 14> cases {{{99.999, -1},  {100.0, 0},    {249.999, 0},  {250.0, 1},
                                         {499.0, 1},    {500.0, 2},    {999.0, 2},    {1000.0, 3},
                                         {2499.0, 3},   {2500.0, 4},   {4999.0, 4},   {5000.0, 5},
                                         {10000.0, 6},  {250000.0, 6}}};
  Report report;
  int64_t flip = 0;
  std::array<uint32_t, 7> expected {};
  uint32_t expected_under = 0;
  for (const auto &test : cases) {
    report.OnFlip(++flip);
    report.AddCompile(Kind::Graphics, 1, 2, test.ms, Thread::Gpu);
    if (test.bucket < 0) {
      ++expected_under;
    } else {
      ++expected[static_cast<size_t>(test.bucket)];
    }
  }
  const auto summary = report.Summarize();
  Check(summary.pause_histogram == expected, "a pause landed in the wrong histogram bucket");
  Check(summary.pauses_under_first_bound == expected_under, "short pauses were not counted apart");
  Check(Near(summary.longest_pause_ms, 250000.0), "longest pause is wrong");
}

// Only the worst ten translations and the longest twenty pauses are kept, longest first.
void TestTopListsAreCapped() {
  Report report;
  for (int index = 1; index <= 15; ++index) {
    report.AddTranslate(Kind::Graphics, "ps", static_cast<uint64_t>(index), index * 10.0,
                        Thread::Worker);
  }
  for (int64_t flip = 1; flip <= 25; ++flip) {
    report.OnFlip(flip);
    report.AddCompile(Kind::Graphics, 1, 2, static_cast<double>(flip), Thread::Gpu);
  }
  const auto summary = report.Summarize();
  Check(summary.worst_translate.size() == Common::FirstRunReport::WORST_TRANSLATE_LIMIT &&
            summary.worst_translate.front().hash == 15 && summary.worst_translate.back().hash == 6,
        "the worst translations were not the ten slowest in order");
  Check(summary.longest_pauses.size() == Common::FirstRunReport::LONGEST_PAUSE_LIMIT &&
            summary.longest_pauses.front().flip == 25 && summary.longest_pauses.back().flip == 6,
        "the longest pauses were not the twenty longest in order");
  Check(summary.pause_count == 25, "the pause count lost pauses that the list dropped");
}

void TestSkipsAndFlips() {
  Report report;
  report.AddSkip(SkipKind::Draw, "color");
  report.AddSkip(SkipKind::Draw, "color");
  report.AddSkip(SkipKind::Dispatch, "depth");
  report.OnFlip(1);
  report.OnFlip(2);
  const auto summary = report.Summarize();
  Check(summary.skips[0] == 2 && summary.skips[1] == 1, "skips were not counted by kind");
  Check(summary.skips_by_target.size() == 2 && summary.skips_by_target[0].second == 2,
        "skips were not counted by target class");
  Check(summary.flips == 2, "flips were not counted");
  Check(Report().Summarize().skips == std::array<uint32_t, 2> {}, "a new report had skips");
}

// The text carries the fields a reader of the file needs. This only checks they are present and
// carry the recorded numbers, since the figures themselves are checked on the summary above.
void TestJsonContents() {
  Report report;
  report.SetBoot({.warm_up_ms = 1234.5, .programs = 3, .modules = 4, .pipelines = 5});
  report.OnFlip(11);
  report.AddCompile(Kind::Graphics, 0xab, 0xcd, 321.0, Thread::Gpu);
  report.AddCompile(Kind::Compute, 0xef, 0, 5.0, Thread::Worker);
  const Common::FirstRunReport::Metadata metadata {
      .title_id = "PPSA00000", .build = "test-build", .flags = {{"--a", true}, {"--b", false}},
      .cold = true, .partial = true};
  report.AddStampFailure();
  const auto json = report.ToJson(metadata, 7, 12.5);
  for (const char *needle :
       {"\"title_id\": \"PPSA00000\"", "\"build\": \"test-build\"", "\"--a\": true",
        "\"--b\": false", "\"cold\": true", "\"warm_up_ms\": 1234.5", "\"blocked_ms\"",
        "\"worst_translate\"", "\"pause_histogram\"", "\"longest_pause_ms\": 321.0",
        "\"vs\": \"00000000000000ab\"", "\"ps\": \"00000000000000cd\"",
        "\"id\": \"00000000000000ef\"", "\"thread\": \"worker\"", "\"audio_holds\": 7",
        "\"run_seconds\": 12.5", "\"flips\": 1,", "\"partial\": true",
        "\"cold_stamp_failures\": 1"}) {
    if (json.find(needle) == std::string::npos) {
      std::fprintf(stderr, "FirstRunReportTests: JSON lacks %s\n", needle);
      std::abort();
    }
  }
}

// The disabled path records nothing, and the enabled path times the scope it wraps.
void TestScopesRespectTheFlag() {
  Common::FirstRunReport::Configure(false, false);
  const auto before = Common::FirstRunReport::Global().Compiles().size();
  { Common::FirstRunReport::CompileScope scope(Kind::Compute, 1, 0, true); }
  Check(Common::FirstRunReport::Global().Compiles().size() == before,
        "a scope recorded while the report was off");
  Common::FirstRunReport::Configure(true, false);
  constexpr auto kSleep = std::chrono::milliseconds(20);
  constexpr double kLowerBoundMs = 15.0; // Below the sleep, so a loaded machine still passes.
  {
    Common::FirstRunReport::CompileScope scope(Kind::Compute, 1, 0, false);
    std::this_thread::sleep_for(kSleep);
  }
  const auto rows = Common::FirstRunReport::Global().Compiles();
  Check(rows.size() == before + 1 && rows.back().thread == Thread::Worker &&
            rows.back().create_ms >= kLowerBoundMs,
        "a scope did not time the work it wrapped");
  Common::FirstRunReport::Configure(false, false);
}

constexpr int64_t kMs = 1000000; // Nanoseconds per millisecond.
constexpr double kMaxPlausibleMs = 60.0 * 1000.0; // No test here runs for a minute.

using Common::FirstRunReport::NO_SUBMISSION;
using Common::FirstRunReport::NO_TIME_NS;
using Common::FirstRunReport::SubmissionScope;
using Common::FirstRunReport::SubmissionStamp;

// Strips spaces and newlines so a JSON fragment can be compared whatever the indent.
std::string Compact(const std::string &text) {
  std::string out;
  for (const char letter : text)
    if (letter != ' ' && letter != '\n') out.push_back(letter);
  return out;
}

// A GPU-thread compile inside a submission keeps the submission id, queue and both times. Each
// baseline row differs from it in one way only and has none of the fields, so the fields cannot
// come from anywhere but the submission context.
void TestCompileRowsKeepSubmission() {
  Report report;
  report.SetEpoch(250 * kMs);
  report.OnFlip(1);
  report.BeginSubmission({.id = 7, .enqueued_ns = 1000 * kMs, .picked_ns = 1100 * kMs, .queue = 1}, true);
  report.AddCompile(Kind::Compute, 1, 0, 5.0, Thread::Gpu, 1250 * kMs);
  report.AddCompile(Kind::Compute, 2, 0, 5.0, Thread::Worker, 1250 * kMs); // Baseline, a worker.
  report.AddCompile(Kind::Compute, 3, 0, 5.0, Thread::Gpu);                // Baseline, no start.
  report.SetPhase(Phase::Boot);
  report.AddCompile(Kind::Compute, 4, 0, 5.0, Thread::Gpu, 1250 * kMs); // Baseline, warm-up.
  report.SetPhase(Phase::Play);
  report.EndSubmission();
  report.AddCompile(Kind::Compute, 5, 0, 5.0, Thread::Gpu, 1250 * kMs); // Baseline, between.
  const auto rows = report.Compiles();
  Check(rows.size() == 5, "a compile row went missing");
  Check(rows[0].submission == 7 && rows[0].queue == 1 && Near(rows[0].visible_ms, 250.0) &&
            Near(rows[0].picked_ms, 150.0) && rows[0].start_ns == 1250 * kMs,
        "the row lost its submission, queue or timing");
  for (size_t index = 1; index < rows.size(); ++index) {
    Check(rows[index].submission == NO_SUBMISSION && rows[index].start_ns == NO_TIME_NS,
          "a worker, boot, untimed or between-submission row took a submission");
  }
  const auto json = report.ToJson({}, 0, 1.0);
  for (const char *needle : {"\"submission\": 7", "\"queue\": 1", "\"visible_ms\": 250.0",
                             "\"picked_ms\": 150.0", "\"start_ms\": 1000.0"}) {
    Check(json.find(needle) != std::string::npos, needle);
  }
  // Four unset rows and one set row, so each key appears exactly once.
  for (const char *key : {"\"visible_ms\"", "\"picked_ms\"", "\"start_ms\"", "\"submission\":"}) {
    Check(json.find(key, json.find(key) + 1) == std::string::npos,
          "an unset row wrote a submission field");
  }
}

// A submission that blocks, lets another run, and resumes in a later flip keeps the enqueue and
// first-pickup times of its stamp and is counted once. This drives the real scope and stamp logic.
// Submission 8 never resumes and is the baseline for the counts and the row times.
void TestResumeKeepsStampAndCountsOnce() {
  Common::FirstRunReport::Configure(true, false);
  auto &global = Common::FirstRunReport::Global();
  global.Reset();
  const int64_t base = Common::FirstRunReport::NowNs() - 1000 * kMs;
  SubmissionStamp seven {.id = 7, .enqueued_ns = base, .queue = 1};
  SubmissionStamp eight {.id = 8, .enqueued_ns = base + 500 * kMs};
  global.OnFlip(1);
  int64_t first_pickup = NO_TIME_NS;
  {
    SubmissionScope scope(seven, true);
    first_pickup = seven.picked_ns;
    global.AddCompile(Kind::Compute, 1, 0, 1.0, Thread::Gpu, first_pickup + 100 * kMs);
  }
  {
    SubmissionScope scope(eight, true);
    global.AddCompile(Kind::Compute, 2, 0, 1.0, Thread::Gpu, eight.picked_ns + 10 * kMs);
  }
  global.OnFlip(2);
  {
    SubmissionScope scope(seven, true); // The resume.
    global.AddCompile(Kind::Compute, 3, 0, 1.0, Thread::Gpu, first_pickup + 900 * kMs);
  }
  global.OnFlip(3);
  const auto rows = global.Compiles();
  const auto summary = global.Summarize();
  Common::FirstRunReport::Configure(false, false);
  global.Reset();
  Check(first_pickup != NO_TIME_NS && seven.picked_ns == first_pickup,
        "the resume moved the first pickup time");
  Check(rows.size() == 3 && rows[0].submission == 7 && rows[2].submission == 7 && rows[2].flip == 2,
        "the resumed submission lost its rows");
  Check(Near(rows[0].picked_ms, 100.0) && Near(rows[2].picked_ms, 900.0),
        "picked_ms did not count from the first pickup");
  // Visible minus picked is the constant gap between the enqueue and the first pickup.
  const double gap = static_cast<double>(first_pickup - base) / static_cast<double>(kMs);
  Check(Near(rows[0].visible_ms - rows[0].picked_ms, gap) &&
            Near(rows[2].visible_ms - rows[2].picked_ms, gap),
        "visible_ms did not count from the first enqueue");
  Check(Near(rows[1].visible_ms - rows[1].picked_ms,
             static_cast<double>(eight.picked_ns - eight.enqueued_ns) / static_cast<double>(kMs)),
        "the baseline submission's times are off");
  Check(summary.submissions == 2, "a resumed submission was counted twice");
  Check(summary.submissions_per_flip == std::vector<uint32_t> {0, 2, 0},
        "submissions were not counted by the flip counter they began under");
}

// The fragment of the file with the counts, with the values asserted, not only the key.
void TestSubmissionCountsInJson() {
  Report report;
  report.OnFlip(1);
  report.BeginSubmission({.id = 1}, true);
  report.BeginSubmission({.id = 2}, true);
  report.OnFlip(2);
  report.BeginSubmission({.id = 3}, false); // A resume is not counted.
  report.OnFlip(3);
  const auto json = Compact(report.ToJson({}, 0, 1.0));
  Check(json.find("\"submissions\":{\"per_flip\":[0,2,0],\"total\":2}") != std::string::npos,
        "the file lacks the submission counts");
  report.Reset();
  Check(report.Summarize().submissions == 0 && report.Summarize().submissions_per_flip.empty(),
        "Reset kept the submission counts");
}

// The scopes the emulator really uses carry the context to the row, and the enqueue time comes
// from the stamp factory.
void TestScopesCarrySubmission() {
  Common::FirstRunReport::Configure(true, false);
  Common::FirstRunReport::Global().Reset();
  constexpr auto kSleep = std::chrono::milliseconds(20);
  constexpr double kLowerBoundMs = 15.0; // Below the sleep, so a loaded machine still passes.
  auto stamp = SubmissionStamp::New(2);
  std::this_thread::sleep_for(kSleep);
  {
    SubmissionScope submission(stamp, true);
    Common::FirstRunReport::CompileScope scope(Kind::Compute, 1, 0, true);
  }
  { Common::FirstRunReport::CompileScope scope(Kind::Compute, 2, 0, true); }
  const auto rows = Common::FirstRunReport::Global().Compiles();
  Check(rows.size() == 2 && rows[0].submission == stamp.id && rows[0].queue == 2 &&
            rows[1].submission == NO_SUBMISSION,
        "the scopes did not link a compile to its submission");
  // The upper bound catches an enqueue time of zero, which would read as the host's uptime.
  Check(rows[0].visible_ms >= kLowerBoundMs && rows[0].visible_ms < kMaxPlausibleMs,
        "visible_ms is not the time since the stamp was made");
  Check(rows[0].picked_ms >= 0.0 && rows[0].picked_ms <= rows[0].visible_ms,
        "the first pickup is not between the enqueue and the compile");
  Check(Common::FirstRunReport::Global().Summarize().submissions == 1,
        "the submission scope did not count");
  Common::FirstRunReport::Global().Reset();
}

// A scope does nothing for a real id with the report off, and for NO_SUBMISSION with it on. Each
// is the baseline of the other, since only one of the two conditions differs, and a real id with
// the report on shows the scope can act at all.
void TestScopeIsInert() {
  Common::FirstRunReport::Configure(false, false);
  Common::FirstRunReport::Global().Reset();
  Check(SubmissionStamp::New(1).id == NO_SUBMISSION, "a stamp was made with the report off");
  SubmissionStamp real {.id = 5, .enqueued_ns = 1};
  { SubmissionScope scope(real, true); }
  Check(real.picked_ns == NO_TIME_NS && !real.counted &&
            Common::FirstRunReport::Global().Summarize().submissions == 0,
        "a scope acted with the report off");

  Common::FirstRunReport::Configure(true, false);
  SubmissionStamp none;
  { SubmissionScope scope(none, true); }
  Check(none.picked_ns == NO_TIME_NS && !none.counted &&
            Common::FirstRunReport::Global().Summarize().submissions == 0,
        "a scope acted for NO_SUBMISSION");
  SubmissionStamp live {.id = 5, .enqueued_ns = 1};
  { SubmissionScope scope(live, true); }
  Check(live.picked_ns != NO_TIME_NS && live.counted &&
            Common::FirstRunReport::Global().Summarize().submissions == 1,
        "a scope did nothing for a real id with the report on");
  Common::FirstRunReport::Configure(false, false);
  Common::FirstRunReport::Global().Reset();
}

// Builds a module with the given instructions after the five header words.
std::vector<uint32_t> Module(const std::vector<std::vector<uint32_t>> &instructions) {
  std::vector<uint32_t> code {0x07230203, 0x00010600, 0, 100, 0};
  for (const auto &instruction : instructions) {
    code.insert(code.end(), instruction.begin(), instruction.end());
  }
  return code;
}

constexpr uint32_t Op(uint32_t opcode, uint32_t words) { return (words << 16u) | opcode; }

const std::vector<uint32_t> kCapability {Op(17, 2), 1};
const std::vector<uint32_t> kExtension {Op(10, 3), 0x656d6f73, 0};
const std::vector<uint32_t> kImport {Op(11, 3), 1, 0x4c534c47};
const std::vector<uint32_t> kMemoryModel {Op(14, 3), 0, 1};
const std::vector<uint32_t> kEntryPoint {Op(15, 4), 5, 1, 0x6e69616d};
const std::vector<uint32_t> kExecutionMode {Op(16, 3), 1, 17};
const std::vector<uint32_t> kExecutionModeId {Op(331, 3), 1, 2};
const std::vector<uint32_t> kSource {Op(3, 3), 2, 450};
const std::vector<uint32_t> kName {Op(5, 3), 1, 0x6e69616d};

// The stamp must sit directly before the first instruction that is not part of the module header,
// and removing it must give back the original module.
void TestStampPlacement() {
  const auto base = Module({kCapability, kExtension, kImport, kMemoryModel, kEntryPoint,
                            kExecutionMode, kExecutionModeId, kName, kSource});
  const auto stamped = Common::FirstRunReport::StampSpirv(base, 0x123456789abcdefull);
  Check(stamped.size() == base.size() + 4, "the stamp is not four words");
  const size_t at = 5 + 2 + 3 + 3 + 3 + 4 + 3 + 3; // The words before OpName.
  Check(stamped[at] == Op(4, 4), "the stamp is not an OpSourceExtension of four words");
  std::vector<uint32_t> without(stamped.begin(), stamped.begin() + at);
  without.insert(without.end(), stamped.begin() + at + 4, stamped.end());
  Check(without == base, "removing the stamp did not give back the original module");
  // The instruction after the stamp is the one that was first after the header.
  Check(stamped[at + 4] == Op(5, 3), "the stamp is not directly before the first debug word");

  // A module whose first non-header instruction is OpSource gets it there too.
  const auto source_first = Module({kCapability, kMemoryModel, kEntryPoint, kSource});
  const auto source_stamped = Common::FirstRunReport::StampSpirv(source_first, 1);
  Check(source_stamped[5 + 2 + 3 + 4] == Op(4, 4), "the stamp was not placed before OpSource");

  // A module with nothing after its header takes the stamp at the end.
  const auto header_only = Module({kCapability, kMemoryModel});
  const auto appended = Common::FirstRunReport::StampSpirv(header_only, 1);
  Check(appended.size() == header_only.size() + 4 && appended[header_only.size()] == Op(4, 4),
        "a header-only module did not take the stamp at its end");
}

// The string must run to its terminator, and two nonces must give two different modules.
void TestStampText() {
  const auto base = Module({kCapability, kMemoryModel, kName});
  for (const uint64_t nonce : {uint64_t {0}, ~uint64_t {0}, uint64_t {0x0123456789abcdef}}) {
    const auto stamped = Common::FirstRunReport::StampSpirv(base, nonce);
    const size_t at = 5 + 2 + 3;
    std::array<unsigned char, 12> bytes {};
    for (size_t word = 0; word < 3; ++word) {
      for (size_t byte = 0; byte < 4; ++byte) {
        bytes[word * 4 + byte] = static_cast<unsigned char>(stamped[at + 1 + word] >> (byte * 8));
      }
    }
    for (size_t index = 0; index < 11; ++index) {
      Check(bytes[index] >= 0x20 && bytes[index] < 0x7f, "the stamp has a non-text byte early");
    }
    Check(bytes[11] == 0, "the stamp has no terminating zero");
  }
  Check(Common::FirstRunReport::StampSpirv(base, 1) != Common::FirstRunReport::StampSpirv(base, 2),
        "two nonces gave the same module");
  // One nonce serves the whole run, and configuring another run picks a new one.
  Common::FirstRunReport::Configure(false, true);
  const auto first = Common::FirstRunReport::StampNonce();
  Check(first == Common::FirstRunReport::StampNonce(), "the nonce changed within a run");
  Common::FirstRunReport::Configure(false, true);
  Check(first != Common::FirstRunReport::StampNonce(), "a second run reused the nonce");
  Common::FirstRunReport::Configure(false, false);
}

void TestStampRejectsBadModules() {
  Check(Common::FirstRunReport::StampSpirv({}, 1).empty(), "an empty module was stamped");
  auto bad_magic = Module({kCapability});
  bad_magic[0] = 0;
  Check(Common::FirstRunReport::StampSpirv(bad_magic, 1).empty(), "a bad magic was stamped");
  auto zero_length = Module({kCapability});
  zero_length[5] = 0;
  Check(Common::FirstRunReport::StampSpirv(zero_length, 1).empty(),
        "a zero-length instruction was stamped");
  auto truncated = Module({kCapability, kMemoryModel});
  truncated.pop_back();
  Check(Common::FirstRunReport::StampSpirv(truncated, 1).empty(),
        "a truncated instruction was stamped");
}

std::string ReadAll(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

// Lists the files of a folder by name, so a leftover .tmp would show.
std::vector<std::string> Names(const std::filesystem::path &folder) {
  std::vector<std::string> names;
  for (const auto &entry : std::filesystem::directory_iterator(folder))
    names.push_back(entry.path().filename().string());
  return names;
}

// The file is created, holds the recorded figures and is complete, and a second call writes
// nothing more. The title id is not valid UTF-8 text, so the writer must not throw on it.
void TestWriteReportOnce() {
  const auto folder = std::filesystem::temp_directory_path() /
                      ("kyty_first_run_report_" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directories(folder);
  const auto previous = std::filesystem::current_path();
  std::filesystem::current_path(folder);

  Common::FirstRunReport::Configure(true, false);
  Common::FirstRunReport::Global().Reset();
  Common::FirstRunReport::Begin({.title_id = std::string("PPSA") + static_cast<char>(0xFF) + "TEST", .build = "b", .flags = {{"--x", true}}});
  Common::FirstRunReport::MarkPartial();
  {
    // Begin sets the point start_ms counts from, so a row stamped just after it starts near zero.
    auto stamp = SubmissionStamp::New(0);
    SubmissionScope submission(stamp, true);
    Common::FirstRunReport::CompileScope scope(Kind::Compute, 0xaaa, 0, true);
  }
  Common::FirstRunReport::Global().AddCompile(Kind::Compute, 0xabc, 0, 12.0, Thread::Gpu);
  Common::FirstRunReport::WriteReportOnce();

  const auto reports = folder / "_FirstRunReports";
  Check(std::filesystem::exists(reports), "no report folder was made");
  auto names = Names(reports);
  Check(names.size() == 1 && names[0].starts_with("PPSA_TEST-") && names[0].ends_with(".json"),
        "exactly one finished report file was expected, with no .tmp left");
  const auto text = ReadAll(reports / names[0]);
  for (const char *needle : {"\"id\": \"0000000000000abc\"", "\"partial\": true", "\"build\": \"b\"",
                             "\"--x\": true"})
    Check(text.find(needle) != std::string::npos, "the written report lacks a recorded value");
  Check(!text.empty() && text.front() == '{' && text.back() == '}',
        "the written report is not a whole JSON object");
  const auto start_key = text.find("\"start_ms\": ");
  Check(start_key != std::string::npos, "the written report lacks start_ms");
  const double start_ms = std::stod(text.substr(start_key + std::string("\"start_ms\": ").size()));
  Check(start_ms >= 0.0 && start_ms < kMaxPlausibleMs, "start_ms is not counted from the start of the run");

  // A second call, after more was recorded, must change nothing.
  Common::FirstRunReport::Global().AddCompile(Kind::Compute, 0xdef, 0, 1.0, Thread::Gpu);
  Common::FirstRunReport::WriteReportOnce();
  Check(Names(reports) == names && ReadAll(reports / names[0]) == text,
        "a second call wrote again");

  Common::FirstRunReport::Configure(false, false);
  Common::FirstRunReport::Global().Reset();
  std::filesystem::current_path(previous);
  std::filesystem::remove_all(folder);
}

void TestReportPath() {
  using namespace std::chrono;
  const auto when = sys_days {year {2026} / November / 7} + hours {13} + minutes {45} + seconds {9};
  Check(Common::FirstRunReport::ReportPath("PPSA01746", when).generic_string() ==
            "_FirstRunReports/PPSA01746-20261107-134509.json",
        "the report path is not <title>-<UTC time>.json");
  const auto leap = sys_days {year {2024} / February / 29} + hours {0} + minutes {0} + seconds {1};
  Check(Common::FirstRunReport::ReportPath("", leap).generic_string() ==
            "_FirstRunReports/UNKNOWN-20240229-000001.json",
        "a leap day or a missing title id is wrong in the report path");
}

} // namespace

int main() {
  TestPausesMergeByFlip();
  TestPendingPauseCounts();
  TestOnlyGpuThreadBlocks();
  TestWaitingBlocksTheGpuThread();
  TestReadyRows();
  TestSampledAfterSkip();
  TestRefusalsAndBlockedTotals();
  TestHistogramEdges();
  TestTopListsAreCapped();
  TestSkipsAndFlips();
  TestJsonContents();
  TestScopesRespectTheFlag();
  TestCompileRowsKeepSubmission();
  TestResumeKeepsStampAndCountsOnce();
  TestSubmissionCountsInJson();
  TestScopesCarrySubmission();
  TestScopeIsInert();
  TestStampPlacement();
  TestStampText();
  TestStampRejectsBadModules();
  TestReportPath();
  TestWriteReportOnce();
  std::puts("FirstRunReportTests: all cases passed");
  return 0;
}
