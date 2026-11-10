#include "common/firstRunReport.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
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
  const auto json = report.ToJson({}, 0, 1.0, 0.0);
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
  const auto json = report.ToJson({}, 0, 1.0, 0.0);
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
  const auto json = report.ToJson({}, 0, 1.0, 0.0);
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
  const auto json = report.ToJson(metadata, 7, 12.5, 2500.5);
  for (const char *needle :
       {"\"title_id\": \"PPSA00000\"", "\"build\": \"test-build\"", "\"--a\": true",
        "\"--b\": false", "\"cold\": true", "\"warm_up_ms\": 1234.5", "\"blocked_ms\"",
        "\"worst_translate\"", "\"pause_histogram\"", "\"longest_pause_ms\": 321.0",
        "\"vs\": \"00000000000000ab\"", "\"ps\": \"00000000000000cd\"",
        "\"id\": \"00000000000000ef\"", "\"thread\": \"worker\"", "\"audio_holds\": 7", "\"audio_held_ms\": 2500.5",
        "\"run_seconds\": 12.5", "\"flips\": 1,", "\"partial\": true",
        "\"cold_stamp_failures\": 1"}) {
    if (json.find(needle) == std::string::npos) {
      std::fprintf(stderr, "FirstRunReportTests: JSON lacks %s\n", needle);
      std::abort();
    }
  }
}

// The deferred-draw section is written with every figure empty or zero when nothing was recorded,
// which is the baseline. Recorded resolves, ineligible reasons and would-skip draws then appear
// with their own numbers, so only the recording can have put them there, and Reset forgets them.
void TestDeferredDrawsSection() {
  Report report;
  const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["deferred_draws"];
  Check(empty["resolves"].empty() && empty["resolves_per_flip"].empty() &&
            empty["ineligible"].empty() && empty["s2_would_skip"]["draws"] == 0 &&
            empty["s2_would_skip"]["waited_ms"] == 0.0,
        "an empty report had deferred-draw figures");
  Check(empty["warm_inline"]["during_pause"] == 0 && empty["warm_inline"]["over_hold_threshold"] == 0 &&
            empty["warm_inline"]["total"] == 0 && empty["warm_inline"]["total_ms"] == 0.0 &&
            empty["warm_inline"]["total_over_hold_threshold"] == 0 &&
            empty["warm_waited"]["count"] == 0 && empty["warm_waited"]["ms"] == 0.0 &&
            empty["warm_waited"]["over_hold_threshold"] == 0,
        "an empty report had warm-up builds");

  // The first resolve runs in submission 9 of flip 0, so it carries both.
  Common::FirstRunReport::SubmissionStamp stamp;
  stamp.id = 9;
  report.BeginSubmission(stamp, true);
  report.AddDeferredResolve({.trigger = "fault", .holes = 3, .distinct_keys = 2,
                             .resolve_wait_ms = 12.5, .record_ms = 3.5, .pre_resolve_ms = 7.5,
                             .save_ms = 0.5, .hold_count_change = 4,
                             .window_hold_count_change = 6, .held_ms = 2.5,
                             .window_held_ms = 40.5});
  report.EndSubmission();
  report.AddDeferredResolve({.trigger = "submit", .holes = 1});
  report.OnFlip(1);
  report.OnFlip(2);
  // Flip 2 is still running, and its resolve has to be reported although no flip follows it.
  report.AddDeferredResolve({.trigger = "shutdown", .holes = 1});
  report.AddDeferredIneligible("image_heap");
  report.AddDeferredIneligible("open_query");
  report.AddDeferredIneligible("image_heap");
  report.AddDeferredWouldSkip(4.0);
  report.AddDeferredWouldSkip(2.5);
  report.AddDeferredWarmInline(false);
  report.AddDeferredWarmInline(true);
  report.AddDeferredWarmInline(true);
  report.AddWarmCompute(false, 10.0, false);
  report.AddWarmCompute(false, 300.0, true);
  report.AddWarmCompute(true, 5.5, false);
  report.AddWarmCompute(true, 260.0, true);
  report.AddWarmCompute(true, 255.0, true);

  const auto summary = report.Summarize();
  Check(summary.deferred_resolves.size() == 3 && summary.deferred_resolves[0].holes == 3,
        "deferred resolves were not kept in order");
  Check(summary.deferred_resolves_per_flip == std::vector<uint32_t> {2, 0, 1},
        "deferred resolves were not counted per flip, the running one included");
  Check(summary.deferred_ineligible.size() == 2 && summary.deferred_ineligible[0].second == 2 &&
            summary.deferred_ineligible[1].second == 1,
        "ineligible draws were not counted per reason");
  Check(summary.deferred_would_skip_draws == 2 && Near(summary.deferred_would_skip_waited_ms, 6.5),
        "would-skip draws were not summed");

  const auto section = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["deferred_draws"];
  const auto &rows   = section["resolves"];
  Check(rows.size() == 3 && rows[0]["holes"] == 3 && rows[0]["distinct_keys"] == 2 &&
            rows[0]["resolve_wait_ms"] == 12.5 && rows[0]["record_ms"] == 3.5 &&
            rows[0]["pre_resolve_ms"] == 7.5 && rows[0]["save_ms"] == 0.5 &&
            rows[0]["hold_count_change"] == 4 && rows[0]["window_hold_count_change"] == 6 &&
            rows[0]["held_ms"] == 2.5 && rows[0]["window_held_ms"] == 40.5,
        "a resolve row lacks its figures");
  Check(rows[0]["flip"] == 0 && rows[1]["flip"] == 0 && rows[2]["flip"] == 2,
        "resolve rows lack the flip they ran in");
  Check(rows[0]["submission"] == 9 && !rows[1].contains("submission"),
        "only the resolve inside a submission should name it");
  Check(rows[0]["trigger"] == "fault" && rows[1]["trigger"] == "submit" &&
            rows[2]["trigger"] == "shutdown",
        "resolve rows lack their trigger");
  Check(section["resolves_per_flip"] == nlohmann::json::array({2, 0, 1}),
        "the JSON lacks the resolves of the running flip");
  Check(section["ineligible"]["image_heap"] == 2 && section["ineligible"]["open_query"] == 1,
        "ineligible reasons were not written");
  Check(section["s2_would_skip"]["draws"] == 2 && section["s2_would_skip"]["waited_ms"] == 6.5,
        "would-skip figures were not written");
  Check(summary.deferred_warm_inline == 3 && summary.deferred_warm_inline_over_hold_threshold == 2 &&
            section["warm_inline"]["during_pause"] == 3 &&
            section["warm_inline"]["over_hold_threshold"] == 2,
        "inline warm-up builds were not counted, or not split by the hold threshold");
  Check(summary.warm_inline_total == 2 && Near(summary.warm_inline_total_ms, 310.0) &&
            summary.warm_inline_total_over_hold_threshold == 1 && summary.warm_waited_count == 3 &&
            Near(summary.warm_waited_ms, 520.5) && summary.warm_waited_over_hold_threshold == 2,
        "warm-up builds taken by a dispatch were not summed");
  Check(section["warm_inline"]["total"] == 2 && section["warm_inline"]["total_ms"] == 310.0 &&
            section["warm_inline"]["total_over_hold_threshold"] == 1 &&
            section["warm_waited"]["count"] == 3 && section["warm_waited"]["ms"] == 520.5 &&
            section["warm_waited"]["over_hold_threshold"] == 2,
        "warm-up builds taken by a dispatch were not written");

  report.Reset();
  const auto cleared = report.Summarize();
  Check(cleared.deferred_resolves.empty() && cleared.deferred_resolves_per_flip.empty() &&
            cleared.deferred_ineligible.empty() && cleared.deferred_would_skip_draws == 0 &&
            Near(cleared.deferred_would_skip_waited_ms, 0.0),
        "Reset kept deferred-draw figures");
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

// A report whose device was never asked says pipeline libraries are not active, with no reason,
// which is the baseline. An inactive device's reason and an active device are each written as
// stored, so a report cannot claim libraries a device refused. Reset forgets the answer.
void TestPipelineLibraryState() {
  Report report;
  const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["graphics_pipeline_library"];
  Check(empty["active"] == false && empty["inactive_reason"] == "",
        "a report whose device was never asked claimed pipeline libraries");
  report.SetPipelineLibrary({.active = false, .inactive_reason = "fast linking is missing"});
  const auto refused = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["graphics_pipeline_library"];
  Check(refused["active"] == false && refused["inactive_reason"] == "fast linking is missing",
        "the reason a device refused pipeline libraries was not written");
  report.SetPipelineLibrary({.active = true});
  const auto active = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["graphics_pipeline_library"];
  Check(active["active"] == true && active["inactive_reason"] == "",
        "an active device was not written as active");
  report.Reset();
  Check(!report.PipelineLibrary().active, "Reset kept the pipeline library state");
}

// What a report that recorded no libraries says.
void CheckLibraryCountersAreZero(const nlohmann::json &zero) {
  for (const char *name : {"vertex_input", "pre_raster", "fragment", "output"}) {
    const auto &kind = zero["libraries"][name];
    Check(kind["built"] == 0 && kind["hits"] == 0 && kind["failed"] == 0 && kind["refused_hits"] == 0 && kind["ms"] == 0.0,
          "a report with no libraries made claimed some");
  }
  Check(zero["links"]["count"] == 0 && zero["links"]["ms_total"] == 0.0 && zero["links"]["ms_max"] == 0.0,
        "a report with no links claimed some");
  for (const char *name : {"rect_list", "tess", "mesh", "create_failed"}) {
    Check(zero["fallback_full"][name] == 0, "a report with no fallbacks claimed some");
  }
}

// Records a different number of events of each kind, so a counter filed under another kind shows.
void AddLibraryExample(Report &report) {
  using Common::FirstRunReport::FallbackKind;
  using Common::FirstRunReport::LibraryKind;
  report.AddLibraryBuilt(LibraryKind::VertexInput, 1.0);
  report.AddLibraryBuilt(LibraryKind::PreRaster, 2.5);
  report.AddLibraryBuilt(LibraryKind::PreRaster, 0.5);
  report.AddLibraryHit(LibraryKind::Fragment);
  report.AddLibraryHit(LibraryKind::Fragment);
  report.AddLibraryHit(LibraryKind::Fragment);
  report.AddLibraryFailed(LibraryKind::Output);
  report.AddLibraryRefusedHit(LibraryKind::Output);
  report.AddLibraryRefusedHit(LibraryKind::Output);
  report.AddLibraryLink(0.25);
  report.AddLibraryLink(1.75);
  report.AddLibraryLink(0.5);
  report.AddLibraryFallback(FallbackKind::RectList);
  report.AddLibraryFallback(FallbackKind::Tess);
  report.AddLibraryFallback(FallbackKind::Tess);
  report.AddLibraryFallback(FallbackKind::Mesh);
  report.AddLibraryFallback(FallbackKind::Mesh);
  report.AddLibraryFallback(FallbackKind::Mesh);
  report.AddLibraryFallback(FallbackKind::CreateFailed);
  report.AddLibraryFallback(FallbackKind::CreateFailed);
  report.AddLibraryFallback(FallbackKind::CreateFailed);
  report.AddLibraryFallback(FallbackKind::CreateFailed);
}

// What AddLibraryExample's events must read as.
void CheckLibraryExample(const nlohmann::json &json) {
  const auto &libraries = json["libraries"];
  Check(libraries["vertex_input"]["built"] == 1 && libraries["vertex_input"]["ms"] == 1.0 &&
            libraries["vertex_input"]["hits"] == 0,
        "the vertex input library was not counted as built alone");
  Check(libraries["pre_raster"]["built"] == 2 && libraries["pre_raster"]["ms"] == 3.0,
        "the pre-rasterization builds were not summed");
  Check(libraries["fragment"]["hits"] == 3 && libraries["fragment"]["built"] == 0,
        "the fragment hits were not counted apart from builds");
  Check(libraries["output"]["refused_hits"] == 2 && libraries["output"]["hits"] == 0,
        "requests answered by a refused library were counted as hits");
  Check(libraries["output"]["failed"] == 1 && libraries["output"]["built"] == 0,
        "the failed output library was counted as built");
  Check(json["links"]["count"] == 3 && json["links"]["ms_total"] == 2.5 && json["links"]["ms_max"] == 1.75,
        "the links were not counted, summed and maximised");
  Check(json["fallback_full"]["rect_list"] == 1 && json["fallback_full"]["tess"] == 2 &&
            json["fallback_full"]["mesh"] == 3 && json["fallback_full"]["create_failed"] == 4,
        "a fallback was counted under another reason");
}

// The library counters start at zero, which is the baseline every figure below is compared with. Each
// kind of library and each fallback reason is written under its own name, with the figures it was
// given and none of the others', so a counter filed under the wrong kind fails. The link maximum is the
// largest link and not the last, and Reset forgets all of it.
void TestPipelineLibraryCounters() {
  Report report;
  CheckLibraryCountersAreZero(
      nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["graphics_pipeline_library"]);
  AddLibraryExample(report);
  CheckLibraryExample(
      nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["graphics_pipeline_library"]);
  report.Reset();
  Check(report.LibraryStats().links == 0 && report.LibraryStats().fallback_full[3] == 0 &&
            report.LibraryStats().libraries[1].built == 0,
        "Reset kept the library counters");
}

// The rows of pipelines linked from libraries. An empty report has none and a zero summary, which is the
// baseline. Each row keeps its own figures and the flip it was added in whatever the caller passed,
// the summary adds them and takes the largest, and only a pipeline whose kinds include both stage
// libraries counts as having built them in sequence. Reset forgets the rows.
void TestLibraryPipelineRows() {
  using Common::FirstRunReport::LibraryKind;
  using Common::FirstRunReport::LibraryPipelineRow;
  constexpr uint32_t kPre = 1u << static_cast<uint32_t>(LibraryKind::PreRaster);
  constexpr uint32_t kFragment = 1u << static_cast<uint32_t>(LibraryKind::Fragment);
  constexpr uint32_t kVertexInput = 1u << static_cast<uint32_t>(LibraryKind::VertexInput);
  Report report;
  const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["graphics_pipeline_library"]["pipelines"];
  Check(empty["rows"].empty() && empty["summary"]["all"]["count"] == 0 &&
            empty["summary"]["all"]["built_both_stage_libraries"] == 0 &&
            empty["summary"]["gpu"]["total_ms"]["max"] == 0.0,
        "an empty report claimed library pipelines");
  report.OnFlip(7);
  report.AddLibraryPipeline({.vs = 0xa, .ps = 0xb, .flip = 99, .thread = Thread::Gpu, .built_ms = 4.0,
                             .waited_ms = 1.0, .link_ms = 0.25, .total_ms = 6.0, .built_count = 2,
                             .built_kinds = kPre | kFragment});
  report.AddLibraryPipeline({.vs = 0xa, .ps = 0xc, .thread = Thread::Worker, .built_ms = 1.5, .waited_ms = 3.0,
                             .link_ms = 0.5, .total_ms = 5.5, .built_count = 2,
                             .built_kinds = kFragment | kVertexInput});
  report.AddLibraryPipeline({.vs = 0xd, .ps = 0xe, .thread = Thread::Worker, .link_ms = 0.25, .total_ms = 0.5});
  const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["graphics_pipeline_library"]["pipelines"];
  const auto &rows = json["rows"];
  Check(rows.size() == 3 && rows[0]["vs"] == "000000000000000a" && rows[0]["flip"] == 7 && rows[0]["thread"] == "gpu" &&
            rows[0]["built_ms"] == 4.0 && rows[0]["waited_ms"] == 1.0 && rows[0]["link_ms"] == 0.25 &&
            rows[0]["total_ms"] == 6.0 && rows[0]["built_count"] == 2,
        "a library pipeline row did not keep its figures");
  Check(rows[0]["built_kinds"] == nlohmann::json::array({"pre_raster", "fragment"}) &&
            rows[1]["built_kinds"] == nlohmann::json::array({"vertex_input", "fragment"}) &&
            rows[2]["built_kinds"].empty() && rows[1]["thread"] == "worker",
        "the kinds a row built were not named");
  const auto &summary = json["summary"]["all"];
  Check(summary["count"] == 3 && summary["built_ms"]["sum"] == 5.5 && summary["built_ms"]["max"] == 4.0 &&
            summary["waited_ms"]["sum"] == 4.0 && summary["waited_ms"]["max"] == 3.0 &&
            summary["link_ms"]["sum"] == 1.0 && summary["total_ms"]["sum"] == 12.0 &&
            summary["total_ms"]["max"] == 6.0,
        "the summary did not add and maximise the rows");
  Check(summary["built_both_stage_libraries"] == 1 && summary["built_any_library"] == 2,
        "the pipelines that built both stage libraries were not counted apart from those that built one");
  // The first row is the GPU thread's and the other two are workers', and each side has its own figures.
  const auto &gpu = json["summary"]["gpu"], &worker = json["summary"]["worker"];
  Check(gpu["count"] == 1 && gpu["built_ms"]["sum"] == 4.0 && gpu["waited_ms"]["sum"] == 1.0 &&
            gpu["total_ms"]["sum"] == 6.0 && gpu["built_both_stage_libraries"] == 1,
        "the GPU thread's summary took in a worker's row, or missed its own");
  Check(worker["count"] == 2 && worker["built_ms"]["sum"] == 1.5 && worker["waited_ms"]["sum"] == 3.0 &&
            worker["total_ms"]["sum"] == 6.0 && worker["built_both_stage_libraries"] == 0 &&
            worker["built_any_library"] == 1,
        "the workers' summary took in the GPU thread's row, or missed its own");
  report.Reset();
  Check(report.LibraryPipelines().empty(), "Reset kept the library pipeline rows");
}

// The scope adds the libraries a thread makes, the waits and the link to one row, and records it only
// when committed. An uncommitted scope, which is a pipeline that fell back, leaves no row. Notes from
// outside any scope go nowhere, and with the report off nothing is recorded at all.
void TestLibraryPipelineScope() {
  using Common::FirstRunReport::LibraryKind;
  using Common::FirstRunReport::LibraryPipelineScope;
  Common::FirstRunReport::Configure(true, false);
  auto &report = Common::FirstRunReport::Global();
  report.Reset();
  LibraryPipelineScope::NoteBuilt(LibraryKind::Fragment, 9.0);
  {
    LibraryPipelineScope scope(1, 2, true);
    LibraryPipelineScope::NoteBuilt(LibraryKind::PreRaster, 2.0);
    LibraryPipelineScope::NoteBuilt(LibraryKind::Fragment, 3.0);
    LibraryPipelineScope::NoteWaited(4.0);
    LibraryPipelineScope::NoteLink(0.5);
    scope.Commit();
  }
  { LibraryPipelineScope fell_back(3, 4, false); LibraryPipelineScope::NoteBuilt(LibraryKind::Output, 1.0); }
  const auto rows = report.LibraryPipelines();
  Check(rows.size() == 1 && rows[0].vs == 1 && rows[0].ps == 2 && rows[0].thread == Thread::Gpu &&
            rows[0].built_ms == 5.0 && rows[0].built_count == 2 && rows[0].waited_ms == 4.0 &&
            rows[0].link_ms == 0.5 && rows[0].total_ms >= 0.0,
        "the scope did not collect one row of what the thread did, or recorded outside its scope");
  report.Reset();
  Common::FirstRunReport::Configure(false, false);
  { LibraryPipelineScope off(5, 6, true); LibraryPipelineScope::NoteBuilt(LibraryKind::PreRaster, 1.0); off.Commit(); }
  Check(report.LibraryPipelines().empty(), "a scope recorded while the report was off");
}

// An empty report writes no waits of draws that could not be deferred, which is the baseline, so
// each figure below can only come from AddDeferredIneligibleWait. Each row keeps its own reason,
// stages and wait, the flip is the one running when it was added whatever the caller passed, and
// the totals sum the waits per reason. Two reasons with different sums show a wait was not added to
// the wrong reason or counted once per row. Reset forgets both.
void TestIneligibleWaits() {
  Report report;
  const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["deferred_draws"];
  Check(empty["ineligible_waits"].empty() && empty["ineligible_waited_ms"].empty(),
        "an empty report had waits of draws that could not be deferred");
  report.AddDeferredIneligibleWait({.reason = "non_push", .vs = 0x11, .ps = 0x22, .waited_ms = 686.5, .flip = 99});
  report.OnFlip(3);
  report.AddDeferredIneligibleWait(
      {.reason = "indirect", .vs = 0x33, .ps = 0, .waited_ms = 40.25, .gpu_thread = false});
  report.AddDeferredIneligibleWait({.reason = "non_push", .vs = 0x44, .ps = 0x55, .waited_ms = 1081.0});
  const auto section = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["deferred_draws"];
  const auto &rows   = section["ineligible_waits"];
  Check(rows.size() == 3, "waits of draws that could not be deferred were lost or added");
  Check(rows[0]["reason"] == "non_push" && rows[0]["flip"] == 0 && rows[0]["vs"] == "0000000000000011" &&
            rows[0]["ps"] == "0000000000000022" && rows[0]["waited_ms"] == 686.5 &&
            rows[0]["gpu_thread"] == true,
        "the first wait row lacks its fields, or kept the caller's flip");
  Check(rows[1]["reason"] == "indirect" && rows[1]["flip"] == 3 && rows[1]["vs"] == "0000000000000033" &&
            rows[1]["ps"] == "0000000000000000" && rows[1]["waited_ms"] == 40.25 &&
            rows[1]["gpu_thread"] == false,
        "the second wait row lacks its fields");
  Check(rows[2]["reason"] == "non_push" && rows[2]["vs"] == "0000000000000044" && rows[2]["waited_ms"] == 1081.0,
        "the third wait row lacks its fields");
  Check(section["ineligible_waited_ms"].size() == 2 && section["ineligible_waited_ms"]["non_push"] == 1767.5 &&
            section["ineligible_waited_ms"]["indirect"] == 40.25,
        "the waits were not summed per reason");
  report.Reset();
  const auto cleared = report.Summarize();
  Check(cleared.deferred_ineligible_waits.empty() && cleared.deferred_ineligible_waited_ms.empty(),
        "Reset kept the waits of draws that could not be deferred");
}

// An empty report writes no warm-up take rows, which is the baseline, so each row below can only
// come from AddWarmTake. Each row keeps its own id, how and stopped time, the flip is the one
// running when it was added, whatever the caller passed, at_ms counts from the epoch, and a row
// with no time has no at_ms. The three rows differ in every field, so a row written from another
// row's data, or rows out of order, fail. Reset forgets them.
void TestWarmTakeRows() {
  Report report;
  Check(nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["warm_takes"].empty(),
        "an empty report had warm-up take rows");
  report.SetEpoch(250 * kMs);
  report.AddWarmTake({.id = 0xab, .how = Common::FirstRunReport::WarmTakeHow::Done,
                      .stopped_ms = 0.0, .at_ns = 1250 * kMs, .flip = 99});
  report.OnFlip(4);
  report.AddWarmTake({.id = 0xcd, .how = Common::FirstRunReport::WarmTakeHow::Waited,
                      .stopped_ms = 12.5, .at_ns = 3250 * kMs});
  report.AddWarmTake({.id = 0xef, .how = Common::FirstRunReport::WarmTakeHow::Inline,
                      .stopped_ms = 7000.5});
  const auto rows = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["warm_takes"];
  Check(rows.size() == 3, "warm-up take rows were lost or added");
  Check(rows[0]["id"] == "00000000000000ab" && rows[0]["how"] == "done" && rows[0]["flip"] == 0 &&
            rows[0]["stopped_ms"] == 0.0 && rows[0]["at_ms"] == 1000.0,
        "the first take row lacks its fields, or kept the caller's flip");
  Check(rows[1]["id"] == "00000000000000cd" && rows[1]["how"] == "waited" && rows[1]["flip"] == 4 &&
            rows[1]["stopped_ms"] == 12.5 && rows[1]["at_ms"] == 3000.0,
        "the waited take row lacks its fields");
  Check(rows[2]["id"] == "00000000000000ef" && rows[2]["how"] == "inline" && rows[2]["flip"] == 4 &&
            rows[2]["stopped_ms"] == 7000.5 && !rows[2].contains("at_ms"),
        "the inline take row lacks its fields, or has a time it was not given");
  report.Reset();
  Check(report.WarmTakes().empty(), "Reset kept warm-up take rows");
}

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
  const auto json = report.ToJson({}, 0, 1.0, 0.0);
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
  const auto json = Compact(report.ToJson({}, 0, 1.0, 0.0));
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

// ---- The generation block of --generate-shader-store. ----

using Common::FirstRunReport::ShaderStoreGenerationReport;

constexpr const char* GENERATION_KEY = "shader_store_generation";

// The generation block of a report's JSON, or null when it has none.
nlohmann::json GenerationBlock(const Report& report) {
	const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0));
	return json.contains(GENERATION_KEY) ? json[GENERATION_KEY] : nlohmann::json(nullptr);
}

// A report that had no generation asked of it has no block, and one that was refused has the
// reason and nothing else. Every other key of the two reports is the same, which is what a run
// without the flag must keep.
void TestGenerationBlockFlagOffAndInactive() {
	Report     off;
	const auto off_json = nlohmann::json::parse(off.ToJson({}, 0, 1.0, 0.0));
	Check(!off_json.contains(GENERATION_KEY), "a report without the flag had a generation block");

	Report refused;
	refused.SetGenerationInactive("no_eboot");
	const auto refused_json = nlohmann::json::parse(refused.ToJson({}, 0, 1.0, 0.0));
	Check(refused_json.contains(GENERATION_KEY), "a refused generation left no block");
	const nlohmann::json expected = {{"active", false}, {"inactive_reason", "no_eboot"}};
	Check(refused_json[GENERATION_KEY] == expected, "the inactive block is not just the reason");
	auto without_block = refused_json;
	without_block.erase(GENERATION_KEY);
	// The process peaks are read when each report is written, so two reports differ in them
	// whatever the flag is. They are left out of the comparison, which is of everything else.
	auto off_without_peaks = off_json;
	off_without_peaks.erase("process");
	without_block.erase("process");
	Check(without_block == off_without_peaks, "the block changed another part of the report");

	// Reset, which tests use between cases, takes the generation state away again.
	refused.Reset();
	Check(!nlohmann::json::parse(refused.ToJson({}, 0, 1.0, 0.0)).contains(GENERATION_KEY),
	      "Reset kept the generation block");
}

// One change of one figure, and where it shows in the block.
struct FieldCase {
	const char*                                       name;
	std::function<void(ShaderStoreGenerationReport&)> change;
	const char*                                       pointer;
	nlohmann::json                                    expected;
};

// Every figure of the block has its own place. Each case changes one figure of an empty block and
// the block then differs from the empty one in that place only, which is how the baseline is
// the same block without the change.
void TestGenerationBlockFields() {
	const auto block_of = [](const ShaderStoreGenerationReport& figures) {
		Report report;
		report.SetGenerationFinal(figures);
		return GenerationBlock(report);
	};
	const auto base = block_of({});
	Check(base["source"] == "generated" && base["file_bytes"] == 0 && !base.contains("verify_ms") &&
	          base["loader_abandoned"] == false && base["regenerated_after_damage"] == false,
	      "the empty block does not say it generated");
	Check(base["active"] == true && base["persisted"] == "not_finished" &&
	          !base.contains("persist_error") && base["supervisor"]["outcome"] == "" &&
	          !base["supervisor"].contains("finished_ms") &&
	          !base["supervisor"].contains("ended_ms") &&
	          !base["supervisor"].contains("child_open_ms") &&
	          base["supervisor"]["failures"].empty() && base["worker"].empty() &&
	          base["drain"]["calls"] == 0,
	      "the empty block is not empty");
	const std::vector<FieldCase> cases = {
	    {"active", [](auto& b) { b.active = false; }, "/active", false},
	    {"inactive_reason", [](auto& b) { b.inactive_reason = "restarts_capped"; },
	     "/inactive_reason", "restarts_capped"},
	    {"persisted", [](auto& b) { b.persisted = "renamed"; }, "/persisted", "renamed"},
	    {"persist_error", [](auto& b) { b.persist_error = "rename of a to b failed"; },
	     "/persist_error", "rename of a to b failed"},
	    {"source", [](auto& b) { b.source = "loaded_regenerated"; }, "/source",
	     "loaded_regenerated"},
	    {"file_bytes", [](auto& b) { b.file_bytes = 65000000; }, "/file_bytes", 65000000},
	    {"verify_ms", [](auto& b) { b.verify_ms = 312.5; }, "/verify_ms", 312.5},
	    {"loader_abandoned", [](auto& b) { b.loader_abandoned = true; }, "/loader_abandoned", true},
	    {"regenerated_after_damage", [](auto& b) { b.regenerated_after_damage = true; },
	     "/regenerated_after_damage", true},
	    {"outcome", [](auto& b) { b.outcome = "finished"; }, "/supervisor/outcome", "finished"},
	    {"spawns", [](auto& b) { b.spawns = 4; }, "/supervisor/spawns", 4},
	    {"restarts", [](auto& b) { b.restarts = 3; }, "/supervisor/restarts", 3},
	    {"spawn_ms", [](auto& b) { b.spawn_ms = 12.5; }, "/supervisor/spawn_ms", 12.5},
	    {"child_open_ms", [](auto& b) { b.child_open_ms = 700.25; }, "/supervisor/child_open_ms",
	     700.25},
	    {"first_record_ms", [](auto& b) { b.first_record_ms = 800.0; },
	     "/supervisor/first_record_ms", 800.0},
	    {"last_record_ms", [](auto& b) { b.last_record_ms = 9000.5; }, "/supervisor/last_record_ms",
	     9000.5},
	    {"finished_ms", [](auto& b) { b.finished_ms = 70000.0; }, "/supervisor/finished_ms",
	     70000.0},
	    {"ended_ms", [](auto& b) { b.ended_ms = 71000.0; }, "/supervisor/ended_ms", 71000.0},
	    {"decode_ms_max", [](auto& b) { b.decode_ms_max = 6.5; }, "/supervisor/decode_ms_max", 6.5},
	    {"terminate_wait_ms", [](auto& b) { b.terminate_wait_ms = 3.25; },
	     "/supervisor/terminate_wait_ms", 3.25},
	    {"frames_delivered", [](auto& b) { b.frames_delivered = 203; },
	     "/supervisor/frames_delivered", 203},
	    {"bad_records", [](auto& b) { b.bad_records = 2; }, "/supervisor/bad_records", 2},
	    {"duplicates", [](auto& b) { b.duplicates = 5; }, "/supervisor/duplicates", 5},
	    {"posted", [](auto& b) { b.posted = 201; }, "/supervisor/posted", 201},
	    {"predicted_programs", [](auto& b) { b.predicted_programs = 119; },
	     "/supervisor/predicted_programs", 119},
	    {"failed_hashes", [](auto& b) { b.failed_hashes = {0xabcull, 0x1ull}; },
	     "/supervisor/failed_hashes",
	     nlohmann::json::array({"0000000000000abc", "0000000000000001"})},
	    {"skip_hashes", [](auto& b) { b.skip_hashes = {0x933533ed9726680dull}; },
	     "/supervisor/skip_hashes", nlohmann::json::array({"933533ed9726680d"})},
	    {"dropped",
	     [](auto& b) { b.dropped = {{.hash = 0x77, .rank = 5, .reason = "push_start"}}; },
	     "/supervisor/dropped",
	     nlohmann::json::array(
	         {{{"hash", "0000000000000077"}, {"rank", 5}, {"reason", "push_start"}}})},
	    {"worker_number", [](auto& b) { b.worker = {{"exit", "0"}}; }, "/worker/exit", 0},
	    {"worker_text", [](auto& b) { b.worker = {{"worst", "abc"}}; }, "/worker/worst", "abc"},
	    // Hashes made only of decimal digits stay text, with their leading zeros.
	    {"worker_hash_digits", [](auto& b) { b.worker = {{"worst_hash", "0012345678901234"}}; },
	     "/worker/worst_hash", "0012345678901234"},
	    {"worker_order_digits", [](auto& b) { b.worker = {{"order", "0000000000000042"}}; },
	     "/worker/order", "0000000000000042"},
	    {"drain_calls", [](auto& b) { b.drain.calls = 30; }, "/drain/calls", 30},
	    {"drain_over_1ms", [](auto& b) { b.drain.over_1ms = 4; }, "/drain/over_1ms", 4},
	    {"drain_over_4ms", [](auto& b) { b.drain.over_4ms = 1; }, "/drain/over_4ms", 1},
	    {"drain_max_ms", [](auto& b) { b.drain.max_ms = 1.75; }, "/drain/max_ms", 1.75},
	    {"drain_total_ms", [](auto& b) { b.drain.total_ms = 22.5; }, "/drain/total_ms", 22.5},
	    {"drain_lag_ms_max", [](auto& b) { b.drain.lag_ms_max = 55.5; }, "/drain/lag_ms_max", 55.5},
	    {"drain_item_max_ms", [](auto& b) { b.drain.item_max_ms = 0.5; }, "/drain/item_max_ms",
	     0.5},
	    {"drain_adopted", [](auto& b) { b.drain.adopted = 200; }, "/drain/adopted", 200},
	    {"drain_duplicates", [](auto& b) { b.drain.duplicates = 6; }, "/drain/duplicates", 6},
	    {"drain_plan_mismatch", [](auto& b) { b.drain.plan_mismatch = 7; }, "/drain/plan_mismatch",
	     7},
	    {"drain_plan_mismatch_reverse", [](auto& b) { b.drain.plan_mismatch_reverse = 8; },
	     "/drain/plan_mismatch_reverse", 8},
	    {"drain_failed_installs", [](auto& b) { b.drain.failed_installs = 9; },
	     "/drain/failed_installs", 9},
	    {"drain_dropped_stopped", [](auto& b) { b.drain.dropped_stopped = 10; },
	     "/drain/dropped_stopped", 10},
	    {"drain_posted", [](auto& b) { b.drain.posted = 11; }, "/drain/posted", 11},
	    {"drain_handles_created", [](auto& b) { b.drain.handles_created = 12; },
	     "/drain/handles_created", 12},
	    {"drain_handles_destroyed", [](auto& b) { b.drain.handles_destroyed = 13; },
	     "/drain/handles_destroyed", 13},
	    {"drain_bad_plans", [](auto& b) { b.drain.bad_plans = 14; }, "/drain/bad_plans", 14},
	    {"drain_undrained_at_stop", [](auto& b) { b.drain.undrained_at_stop = 15; },
	     "/drain/undrained_at_stop", 15},
	    {"drain_pending", [](auto& b) { b.drain.pending = 16; }, "/drain/pending", 16},
	};
	for (const auto& field: cases) {
		ShaderStoreGenerationReport figures;
		field.change(figures);
		auto expected_block                                         = base;
		expected_block[nlohmann::json::json_pointer(field.pointer)] = field.expected;
		const auto actual                                           = block_of(figures);
		if (actual != expected_block || actual == base) {
			std::fprintf(stderr, "FirstRunReportTests: the figure %s is not where it belongs\n",
			             field.name);
			std::abort();
		}
	}

	// The cap is read from the outcome, so it is true for that outcome and false for another.
	ShaderStoreGenerationReport capped;
	capped.outcome = "restarts_capped";
	Check(block_of(capped)["supervisor"]["restarts_capped"] == true &&
	          base["supervisor"]["restarts_capped"] == false,
	      "restarts_capped does not follow the outcome");

	// The peak comes with its source, and a block without one has neither.
	ShaderStoreGenerationReport peak;
	peak.peak_bytes       = 123456789ull;
	peak.peak_source      = "commit";
	const auto peak_block = block_of(peak)["supervisor"];
	Check(peak_block["peak_bytes"] == 123456789ull && peak_block["peak_source"] == "commit" &&
	          !base["supervisor"].contains("peak_bytes") &&
	          !base["supervisor"].contains("peak_source"),
	      "the peak and its source are not shown together");

	// Rows of the lists carry every field of the row, the optional ones only when present.
	ShaderStoreGenerationReport rows;
	rows.failures    = {{.hash      = 0xf1,
	                     .rank      = 9,
	                     .exit_code = 321,
	                     .kind      = "hang",
	                     .last_line = "start",
	                     .spawn     = 2,
	                     .cpu_ms    = 250.5,
	                     .out_tail  = "EXIT text"},
	                    {.hash = 0xf2, .kind = "crash", .spawn = 3}};
	rows.spawn_usage = {
	    {.spawn = 1, .cpu_ms = 10.5, .peak_bytes = 4096, .peak_source = "working_set"},
	    {.spawn = 2, .cpu_ms = 20.0}};
	const auto  row_block = block_of(rows)["supervisor"];
	const auto& failures  = row_block["failures"];
	Check(failures.size() == 2 && failures[0]["hash"] == "00000000000000f1" &&
	          failures[0]["rank"] == 9 && failures[0]["exit_code"] == 321 &&
	          failures[0]["kind"] == "hang" && failures[0]["last_line"] == "start" &&
	          failures[0]["spawn"] == 2 && failures[0]["cpu_ms"] == 250.5 &&
	          failures[0]["out_tail"] == "EXIT text",
	      "a failure row lacks a figure");
	Check(failures[1]["spawn"] == 3 && !failures[1].contains("cpu_ms"),
	      "a failure without a processor time has one");
	const auto& usage = row_block["spawn_usage"];
	Check(usage.size() == 2 && usage[0]["spawn"] == 1 && usage[0]["cpu_ms"] == 10.5 &&
	          usage[0]["peak_bytes"] == 4096 && usage[0]["peak_source"] == "working_set" &&
	          usage[1]["spawn"] == 2 && !usage[1].contains("peak_bytes") &&
	          !usage[1].contains("peak_source"),
	      "a usage row lacks a figure, or a missing peak is shown as zero");
}

// finished_at_ms is the finish counted from the report's epoch, and finished_ms stays a duration
// from the supervisor thread's start. The thread starts 700 ms after the epoch here, so the two
// differ by that offset, and a finished_at_ms copied from finished_ms or left on the raw clock
// would not show it. The baseline is the same block with no finish, which has no finished_at_ms.
void TestGenerationFinishedAtMs() {
	constexpr int64_t MS               = 1'000'000;
	constexpr int64_t EPOCH_NS         = 250 * MS;
	constexpr double  THREAD_OFFSET_MS = 700.0;
	constexpr double  FINISHED_MS      = 9000.0;
	const auto        block_of         = [&](const ShaderStoreGenerationReport& figures) {
        Report report;
        report.SetEpoch(EPOCH_NS);
        report.SetGenerationFinal(figures);
        return GenerationBlock(report)["supervisor"];
	};
	ShaderStoreGenerationReport unfinished;
	unfinished.outcome = "stopped";
	unfinished.ended_ms = FINISHED_MS;
	const auto base     = block_of(unfinished);
	Check(!base.contains("finished_at_ms") && !base.contains("finished_ms"),
	      "a run that did not finish has a finish time");

	ShaderStoreGenerationReport finished = unfinished;
	finished.outcome                     = "finished";
	finished.finished_ms                 = FINISHED_MS;
	// Nine seconds from a thread that started 700 ms after the epoch is 9700 ms from the epoch.
	finished.finished_ns = EPOCH_NS + static_cast<int64_t>((THREAD_OFFSET_MS + FINISHED_MS) * MS);
	const auto actual    = block_of(finished);
	Check(actual["finished_ms"] == FINISHED_MS, "finished_ms is no longer the duration");
	Check(actual["finished_at_ms"] == THREAD_OFFSET_MS + FINISHED_MS,
	      "finished_at_ms is not counted from the epoch");
	Check(actual["finished_at_ms"].get<double>() - actual["finished_ms"].get<double>() ==
	          THREAD_OFFSET_MS,
	      "finished_at_ms and finished_ms do not differ by the thread's offset");

	// A finished_ms with no finish moment, as an older source gives, writes no finished_at_ms.
	ShaderStoreGenerationReport duration_only = finished;
	duration_only.finished_ns.reset();
	Check(!block_of(duration_only).contains("finished_at_ms") &&
	          block_of(duration_only).contains("finished_ms"),
	      "finished_at_ms was written from finished_ms");
}

// The rows of the generation block put each time on the run's clock and leave out a time that did
// not happen. The empty block has the shape of the full one, so a reader finds every key. The
// baseline is the empty block, and a row with a missing time is the baseline of the row with it.
void TestGenerationRowsJson() {
	constexpr int64_t MS       = 1'000'000;
	const auto        block_of = [](const ShaderStoreGenerationReport& figures) {
        Report report;
        report.SetEpoch(250 * MS);
        report.SetGenerationFinal(figures);
        return GenerationBlock(report);
	};
	const auto empty = block_of({});
	Check(empty["predictions"].is_array() && empty["predictions"].empty() &&
	          empty["generated_modules"].is_array() && empty["generated_modules"].empty() &&
	          empty["slow_takes"].is_array() && empty["slow_takes"].empty() &&
	          empty["unused_build_ms"] == 0.0 && empty["runtime_misses"]["total"] == 0 &&
	          empty["runtime_misses"]["classes"].is_object(),
	      "the empty block lacks the keys of the rows");

	ShaderStoreGenerationReport figures;
	figures.predictions       = {{.hash = 0xab, .outcome = "predicted", .candidates = 3},
	                             {.hash = 0xcd, .outcome = "calls", .candidates = 0}};
	figures.generated_modules = {
	    {.id           = 0x11,
	     .code_hash    = 0xab,
	     .posted_ns    = 1250 * MS,
	     .adopted_ns   = 1500 * MS,
	     .ready_ns     = 2250 * MS,
	     .first_use_ns = 3250 * MS,
	     .build_ms     = 12.5,
	     .lead_ms      = -4.5,
	     .how          = "done"},
	    {.id = 0x12, .posted_ns = 1250 * MS, .adopted_ns = 1500 * MS, .how = "unused"}};
	figures.unused_build_ms                = 7.5;
	figures.misses.total                   = 3;
	figures.misses.create_ms               = 4.25;
	figures.misses.classes                 = {{"wrong_spec", 2, 3.0}, {"unscanned", 1, 1.25}};
	figures.misses.refused                 = {{"duplicate", 1}};
	figures.misses.wrong_spec_programs     = 1;
	figures.misses.wrong_spec_max_variants = 2;
	figures.slow_takes                     = {{.id         = 0x11,
	                                           .generated  = true,
	                                           .how        = "waited",
	                                           .stopped_ms = 300.5,
	                                           .at_ns      = 3250 * MS,
	                                           .paused     = true,
	                                           .posted_ns  = 1250 * MS,
	                                           .ready_ns   = 2250 * MS},
	                                          {.id = 0x99, .how = "inline", .stopped_ms = 260.0}};
	const auto  block                      = block_of(figures);
	const auto& predictions                = block["predictions"];
	Check(predictions.size() == 2 && predictions[0]["hash"] == "00000000000000ab" &&
	          predictions[0]["outcome"] == "predicted" && predictions[0]["candidates"] == 3 &&
	          predictions[1]["outcome"] == "calls" && predictions[1]["candidates"] == 0,
	      "a prediction row lacks a figure");
	const auto& modules = block["generated_modules"];
	Check(modules.size() == 2 && modules[0]["id"] == "0000000000000011" &&
	          modules[0]["code_hash"] == "00000000000000ab" && modules[0]["posted_ms"] == 1000.0 &&
	          modules[0]["adopted_ms"] == 1250.0 && modules[0]["ready_ms"] == 2000.0 &&
	          modules[0]["first_use_ms"] == 3000.0 && modules[0]["build_ms"] == 12.5 &&
	          modules[0]["lead_ms"] == -4.5 && modules[0]["how"] == "done",
	      "a module row lacks a figure, or its times are not from the run start");
	Check(modules[1]["how"] == "unused" && !modules[1].contains("ready_ms") &&
	          !modules[1].contains("first_use_ms") && !modules[1].contains("build_ms") &&
	          !modules[1].contains("lead_ms"),
	      "a module that was never built shows a time it did not have");
	Check(block["unused_build_ms"] == 7.5, "unused_build_ms is not the figure");
	const auto& misses = block["runtime_misses"];
	Check(misses["total"] == 3 && misses["create_ms"] == 4.25 &&
	          misses["classes"]["wrong_spec"]["count"] == 2 &&
	          misses["classes"]["wrong_spec"]["create_ms"] == 3.0 &&
	          misses["classes"]["unscanned"]["count"] == 1 && misses["refused"]["duplicate"] == 1 &&
	          misses["wrong_spec_programs"] == 1 && misses["wrong_spec_max_variants"] == 2,
	      "the misses lack a figure");
	const auto& slow = block["slow_takes"];
	Check(slow.size() == 2 && slow[0]["id"] == "0000000000000011" && slow[0]["generated"] == true &&
	          slow[0]["how"] == "waited" && slow[0]["stopped_ms"] == 300.5 &&
	          slow[0]["paused"] == true && slow[0]["at_ms"] == 3000.0 &&
	          slow[0]["posted_ms"] == 1000.0 && slow[0]["ready_ms"] == 2000.0 &&
	          !slow[0].contains("adopted_ms"),
	      "a slow take row lacks a figure");
	Check(slow[1]["generated"] == false && slow[1]["paused"] == false &&
	          !slow[1].contains("at_ms") && !slow[1].contains("posted_ms") &&
	          !slow[1].contains("ready_ms"),
	      "a take of a module generation did not make shows a generated time");
}

// ---- Guest audio underruns, the machine, and what a worker compile met. ----

// The underruns of a report: each one is stored with its time on the NowNs clock, between the clock
// before the call and after it. An empty report has the key with nothing in it, so every report
// shows it. The baseline of the count is the empty report.
void TestAudioUnderrunsAreStamped() {
	Report     report;
	const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["audio_underruns"];
	Check(empty["count"] == 0 && empty["overflow"] == 0 && empty["at_ms"].is_array() &&
	          empty["at_ms"].empty(),
	      "an empty report lacks the underruns key or has an underrun");

	const int64_t before = Common::FirstRunReport::NowNs();
	report.AddAudioUnderrun();
	report.AddAudioUnderrun();
	const int64_t after = Common::FirstRunReport::NowNs();
	const auto    taken = report.AudioUnderrunsSoFar();
	Check(taken.at_ns.size() == 2 && taken.overflow == 0, "the underruns were not both stored");
	for (const auto at_ns: taken.at_ns) {
		Check(at_ns >= before && at_ns <= after, "an underrun is not stamped with the NowNs clock");
	}
	Check(taken.at_ns.size() == 2 && taken.at_ns[0] <= taken.at_ns[1],
	      "the underruns are not in the order they happened");
	// The same clock as the epoch, so the time in the file is counted from the run start. The epoch
	// is 250 ms before the first underrun, so the first time is exactly that, which a time on the
	// raw clock could not be.
	constexpr int64_t epoch_lead_ns = 250 * 1000000;
	report.SetEpoch(taken.at_ns[0] - epoch_lead_ns);
	const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["audio_underruns"];
	Check(json["count"] == 2 && json["at_ms"].size() == 2 && json["at_ms"][0] == 250.0 &&
	          json["at_ms"][1] >= 250.0,
	      "the underruns are not counted from the run start in the file");
	// Reset empties the report and starts its slots again, so the whole capacity is free. A report
	// that kept its index would count the last of them as overflow.
	report.Reset();
	Check(report.AudioUnderrunsSoFar().at_ns.empty(), "Reset kept the underruns");
	for (size_t index = 0; index < Common::FirstRunReport::AUDIO_UNDERRUN_CAPACITY; index++) {
		report.AddAudioUnderrun();
	}
	const auto refilled = report.AudioUnderrunsSoFar();
	Check(refilled.at_ns.size() == Common::FirstRunReport::AUDIO_UNDERRUN_CAPACITY &&
	          refilled.overflow == 0,
	      "Reset did not free the slots of the underruns");
}

// The report is the only gate: with it off the free function stores nothing, and with it on the
// same call does. The baseline is the call with the report on.
void TestAudioUnderrunsRespectTheFlag() {
	auto& global = Common::FirstRunReport::Global();
	global.Reset();
	Common::FirstRunReport::Configure(false, false);
	Common::FirstRunReport::AddAudioUnderrun();
	Check(global.AudioUnderrunsSoFar().at_ns.empty() && global.AudioUnderrunsSoFar().overflow == 0,
	      "an underrun was stored while the report was off");
	Common::FirstRunReport::Configure(true, false);
	Common::FirstRunReport::AddAudioUnderrun();
	Check(global.AudioUnderrunsSoFar().at_ns.size() == 1,
	      "an underrun was not stored with the report on");
	Common::FirstRunReport::Configure(false, false);
	global.Reset();
}

// Past the capacity an underrun is only counted, and the stored ones are not overwritten. The
// baseline is exactly the capacity, which counts no overflow. Threads share the last slots, so the
// total must still be exact.
void TestAudioUnderrunsOverflow() {
	constexpr size_t capacity = Common::FirstRunReport::AUDIO_UNDERRUN_CAPACITY;
	Report           exact;
	for (size_t index = 0; index < capacity; index++) {
		exact.AddAudioUnderrun();
	}
	Check(exact.AudioUnderrunsSoFar().at_ns.size() == capacity &&
	          exact.AudioUnderrunsSoFar().overflow == 0,
	      "the capacity itself was counted as an overflow");

	Report over;
	for (size_t index = 0; index < capacity + 5; index++) {
		over.AddAudioUnderrun();
	}
	const auto taken = over.AudioUnderrunsSoFar();
	Check(taken.at_ns.size() == capacity && taken.overflow == 5,
	      "underruns past the capacity were lost or wrapped");
	const auto json = nlohmann::json::parse(over.ToJson({}, 0, 1.0, 0.0))["audio_underruns"];
	Check(json["count"] == capacity + 5 && json["overflow"] == 5 &&
	          json["at_ms"].size() == capacity,
	      "the count in the file does not include the overflow");

	constexpr size_t         threads    = 4;
	constexpr size_t         per_thread = 3000;
	Report                   racing;
	std::vector<std::thread> callers;
	for (size_t caller = 0; caller < threads; caller++) {
		callers.emplace_back([&] {
			for (size_t index = 0; index < per_thread; index++) {
				racing.AddAudioUnderrun();
			}
		});
	}
	for (auto& caller: callers) {
		caller.join();
	}
	const auto raced = racing.AudioUnderrunsSoFar();
	Check(raced.at_ns.size() + raced.overflow == threads * per_thread &&
	          raced.at_ns.size() == capacity,
	      "racing underruns were lost or stored past the capacity");
}

// The machine and process blocks are in every report. The processor is named, there is at least one
// core, the memory is the installed memory, and the warm worker count is what was set. The process
// peaks are there on the platforms that give them, and the working set is no more than the memory.
void TestMachineBlock() {
	Common::FirstRunReport::SetWarmWorkerCount(7);
	Report     report;
	const auto json    = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0));
	const auto machine = json["machine"];
	Check(machine["cpu_model"].is_string() && !machine["cpu_model"].get<std::string>().empty(),
	      "the machine has no processor name");
	// The platforms that can name their processor do, so a name that is still "unknown" is a reader
	// that failed. Others may have none to give.
#if defined(_M_X64) || defined(__x86_64__) || defined(__APPLE__)
	Check(machine["cpu_model"] != "unknown", "the processor was not read");
#endif
	Check(machine["logical_cores"].get<uint32_t>() >= 1, "the machine has no cores");
	constexpr uint64_t smallest_memory = 64ull * 1024 * 1024;
	Check(machine["ram_bytes"].get<uint64_t>() > smallest_memory, "the machine has no memory");
	Check(machine["warm_workers"] == 7, "the warm worker count is not the one set");
	Common::FirstRunReport::SetWarmWorkerCount(0);
	Check(nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0))["machine"]["warm_workers"] == 0,
	      "the warm worker count did not follow the setting");
	const auto         process       = json["process"];
	constexpr uint64_t smallest_peak = 1024 * 1024;
	Check(process.contains("peak_working_set_bytes") &&
	          process["peak_working_set_bytes"].get<uint64_t>() > smallest_peak &&
	          process["peak_working_set_bytes"].get<uint64_t>() <=
	              machine["ram_bytes"].get<uint64_t>(),
	      "the process has no sane peak working set");
#ifdef _WIN32
	Check(process.contains("peak_commit_bytes") &&
	          process["peak_commit_bytes"].get<uint64_t>() > smallest_peak,
	      "the process has no peak commit");
#else
	// Only Windows has a peak commit. Linux's VmPeak is the address space, which is not commit.
	Check(!process.contains("peak_commit_bytes"), "a platform with no peak commit gave one");
#endif
}

// The row of a worker compile shows when it began and what it met: warm builds running, and whether
// the generator's worker was alive. The baseline is the same compile with none running. A
// GPU-thread row has none of the three keys, so it is as it was. With the report off a warm build
// is not counted.
void TestWorkerCompileContext() {
	using Common::FirstRunReport::CompileScope;
	using Common::FirstRunReport::WarmBuildScope;
	auto& global = Common::FirstRunReport::Global();
	global.Reset();
	Common::FirstRunReport::Configure(false, false);
	{
		const WarmBuildScope off;
		Check(Common::FirstRunReport::WarmBuildsRunning() == 0,
		      "a warm build was counted while the report was off");
	}
	Common::FirstRunReport::Configure(true, false);
	{
		CompileScope baseline(Kind::Graphics, 1, 2, false);
	}
	{
		const WarmBuildScope first;
		const WarmBuildScope second;
		Common::FirstRunReport::SetGeneratorChildAlive(true);
		Check(Common::FirstRunReport::WarmBuildsRunning() == 2, "the warm builds were not counted");
		{
			CompileScope busy(Kind::Graphics, 3, 4, false);
		}
		{
			CompileScope on_gpu(Kind::Compute, 5, 0, true);
		}
	}
	Common::FirstRunReport::SetGeneratorChildAlive(false);
	Check(Common::FirstRunReport::WarmBuildsRunning() == 0,
	      "a finished warm build is still counted");
	const auto rows = nlohmann::json::parse(global.ToJson({}, 0, 1.0, 0.0))["compiles"];
	Check(rows.size() == 3, "the compile rows were lost");
	if (rows.size() == 3) {
		Check(rows[0]["thread"] == "worker" && rows[0].contains("at_ms") &&
		          rows[0]["at_ms"].is_number() && rows[0]["warm_running"] == 0 &&
		          rows[0]["child_alive"] == false,
		      "the baseline worker row lacks its context or has one it did not meet");
		Check(rows[1]["thread"] == "worker" && rows[1]["warm_running"] == 2 &&
		          rows[1]["child_alive"] == true,
		      "the worker row did not carry the warm builds and the generator it met");
		Check(rows[2]["thread"] == "gpu" && !rows[2].contains("at_ms") &&
		          !rows[2].contains("warm_running") && !rows[2].contains("child_alive"),
		      "a GPU-thread row got the context of a worker row");
	}
	Common::FirstRunReport::Configure(false, false);
	global.Reset();
}

// With a source, the writer asks it each time. A final figure replaces it at once and the source is
// never called again, so the owner may destroy what the source reads. The baseline is the same
// report before the final figure.
void TestGenerationSourceThenFinal() {
	Report           report;
	std::atomic<int> calls {0};
	report.SetGenerationSource([&calls] {
		ShaderStoreGenerationReport figures;
		figures.spawns = 3;
		calls++;
		return figures;
	});
	Check(GenerationBlock(report)["supervisor"]["spawns"] == 3 && calls == 1,
	      "the writer did not use the source");
	Check(GenerationBlock(report)["supervisor"]["spawns"] == 3 && calls == 2,
	      "the source was not asked again");
	ShaderStoreGenerationReport final_figures;
	final_figures.spawns = 9;
	report.SetGenerationFinal(final_figures);
	Check(GenerationBlock(report)["supervisor"]["spawns"] == 9, "the final figures were not used");
	Check(calls == 2, "the source was called after the final figures replaced it");
}

// The owner destroys what the source reads right after SetGenerationFinal returns. A writer that
// called the source after that would read a destroyed object, which this test models as a flag that
// the source checks.
void TestGenerationWriterNeverTouchesADestroyedSource() {
	Report            report;
	std::atomic<bool> alive {true};
	std::atomic<int>  violations {0};
	report.SetGenerationSource([&] {
		if (!alive) violations++;
		return ShaderStoreGenerationReport {};
	});
	Check(GenerationBlock(report).contains("supervisor") && violations == 0,
	      "the live source failed");
	report.SetGenerationFinal({});
	alive            = false;
	const auto after = GenerationBlock(report);
	Check(violations == 0, "the writer called a source that was destroyed");
	Check(after["active"] == true, "the final figures were not written after the source went");
}

// The writer is inside the source when the owner hands over the final figures. The hand-over must
// wait for the writer, because the owner destroys the state right after it, and the writer must be
// out of the source by then. The source here is slow on purpose.
void TestGenerationFinalWaitsForAWriterInTheSource() {
	Report            report;
	std::atomic<bool> inside {false};
	std::atomic<bool> release {false};
	std::atomic<bool> left_source {false};
	report.SetGenerationSource([&] {
		inside = true;
		while (!release) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		left_source = true;
		return ShaderStoreGenerationReport {};
	});
	std::thread writer([&] { (void)GenerationBlock(report); });
	while (!inside) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	std::atomic<bool> handed_over {false};
	std::thread       owner([&] {
        report.SetGenerationFinal({});
        handed_over = true;
    });
	// A hand-over that did not wait would be done by now.
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	Check(!handed_over, "the final figures were handed over while a writer was in the source");
	release = true;
	writer.join();
	owner.join();
	Check(handed_over && left_source, "the hand-over or the writer never finished");
}

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
  TestDeferredDrawsSection();
  TestWarmTakeRows();
  TestIneligibleWaits();
  TestPipelineLibraryState();
  TestPipelineLibraryCounters();
  TestLibraryPipelineRows();
  TestLibraryPipelineScope();
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
  TestGenerationBlockFlagOffAndInactive();
  TestGenerationBlockFields();
  TestGenerationRowsJson();
  TestGenerationFinishedAtMs();
  TestAudioUnderrunsAreStamped();
  TestAudioUnderrunsRespectTheFlag();
  TestAudioUnderrunsOverflow();
  TestMachineBlock();
  TestWorkerCompileContext();
  TestGenerationSourceThenFinal();
  TestGenerationWriterNeverTouchesADestroyedSource();
  TestGenerationFinalWaitsForAWriterInTheSource();
  std::puts("FirstRunReportTests: all cases passed");
  return 0;
}
