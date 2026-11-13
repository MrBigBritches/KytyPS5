#include "common/firstRunReport.h"
#include "common/gpuStall.h"

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
  const auto json = report.ToJson({}, 0, 1.0, 0.0, {});
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
  const auto json = report.ToJson({}, 0, 1.0, 0.0, {});
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
  const auto json = report.ToJson({}, 0, 1.0, 0.0, {});
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
  const auto json = report.ToJson(metadata, 7, 12.5, 2500.5, {});
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
  const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["deferred_draws"];
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

  const auto section = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["deferred_draws"];
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
  const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["graphics_pipeline_library"];
  Check(empty["active"] == false && empty["inactive_reason"] == "",
        "a report whose device was never asked claimed pipeline libraries");
  report.SetPipelineLibrary({.active = false, .inactive_reason = "fast linking is missing"});
  const auto refused = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["graphics_pipeline_library"];
  Check(refused["active"] == false && refused["inactive_reason"] == "fast linking is missing",
        "the reason a device refused pipeline libraries was not written");
  report.SetPipelineLibrary({.active = true});
  const auto active = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["graphics_pipeline_library"];
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
      nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["graphics_pipeline_library"]);
  AddLibraryExample(report);
  CheckLibraryExample(
      nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["graphics_pipeline_library"]);
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
  const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["graphics_pipeline_library"]["pipelines"];
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
  const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["graphics_pipeline_library"]["pipelines"];
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
  const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["deferred_draws"];
  Check(empty["ineligible_waits"].empty() && empty["ineligible_waited_ms"].empty(),
        "an empty report had waits of draws that could not be deferred");
  report.AddDeferredIneligibleWait({.reason = "non_push", .vs = 0x11, .ps = 0x22, .waited_ms = 686.5, .flip = 99});
  report.OnFlip(3);
  report.AddDeferredIneligibleWait(
      {.reason = "indirect", .vs = 0x33, .ps = 0, .waited_ms = 40.25, .gpu_thread = false});
  report.AddDeferredIneligibleWait({.reason = "non_push", .vs = 0x44, .ps = 0x55, .waited_ms = 1081.0});
  const auto section = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["deferred_draws"];
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
  Check(nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["warm_takes"].empty(),
        "an empty report had warm-up take rows");
  report.SetEpoch(250 * kMs);
  report.AddWarmTake({.id = 0xab, .how = Common::FirstRunReport::WarmTakeHow::Done,
                      .stopped_ms = 0.0, .at_ns = 1250 * kMs, .flip = 99});
  report.OnFlip(4);
  report.AddWarmTake({.id = 0xcd, .how = Common::FirstRunReport::WarmTakeHow::Waited,
                      .stopped_ms = 12.5, .at_ns = 3250 * kMs});
  report.AddWarmTake({.id = 0xef, .how = Common::FirstRunReport::WarmTakeHow::Inline,
                      .stopped_ms = 7000.5});
  const auto rows = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["warm_takes"];
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
  const auto json = report.ToJson({}, 0, 1.0, 0.0, {});
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

// The lead of a compile comes from the submission that holds it, not from the newest one. Submission
// 2 is enqueued later than 1, began while 1 was blocked, and 1 then resumes and compiles. The newest
// stamp would give a negative lead here, and the current stamp gives the first submission's lead.
// The same compile in a run with only submission 1 is the baseline for the number.
void TestSubmitLeadUsesContainingSubmission() {
  Report baseline;
  baseline.BeginSubmission({.id = 1, .enqueued_ns = 1000 * kMs, .picked_ns = 1100 * kMs}, true);
  baseline.AddCompile(Kind::Compute, 9, 0, 5.0, Thread::Gpu, 1300 * kMs);

  Report report;
  report.BeginSubmission({.id = 1, .enqueued_ns = 1000 * kMs, .picked_ns = 1100 * kMs}, true);
  report.BeginSubmission({.id = 2, .enqueued_ns = 5000 * kMs, .picked_ns = 5100 * kMs}, true);
  report.BeginSubmission({.id = 1, .enqueued_ns = 1000 * kMs, .picked_ns = 1100 * kMs}, false);
  report.AddCompile(Kind::Compute, 9, 0, 5.0, Thread::Gpu, 1300 * kMs);

  const auto leads = report.Summarize().submit_leads;
  Check(leads.n == 1 && Near(leads.p50_ms, 300.0) && Near(leads.max_ms, 300.0),
        "the lead was not taken from the containing submission");
  Check(leads.n == baseline.Summarize().submit_leads.n &&
            Near(leads.p50_ms, baseline.Summarize().submit_leads.p50_ms),
        "a second submission changed the lead");
  const auto json = Compact(report.ToJson({}, 0, 1.0, 0.0, {}));
  Check(json.find("\"submit_lead_summary\":{\"max_ms\":300.0,\"n\":1,\"p10_ms\":300.0,\"p50_ms\":300.0,"
                  "\"p90_ms\":300.0}") != std::string::npos,
        "the file lacks the submit lead summary");
}

// The summary counts only inline compute compiles of guest shaders in a submission in play, and its
// percentiles are nearest rank. Twelve leads 10 to 120 ms are added out of order, so only a sort
// puts them right, and p10 and p90 land on fractional ranks (1.2 and 10.8), which rounding down
// would move. The excluded rows carry a lead of 9999 ms, so any that leaked in would show in max
// and n. The graphics, tiler and PSSR rows each test one filter. The worker, no-start, warm-up
// and between rows are regression cover, since the submission test alone already leaves them out.
void TestSubmitLeadSummary() {
  constexpr int64_t kHugeLeadMs = 9999;
  Report            report;
  const auto        empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}));
  Check(report.Summarize().submit_leads.n == 0 && empty.contains("submit_lead_summary") &&
            empty["submit_lead_summary"] == nlohmann::json({{"n", 0}}),
        "an empty report did not write exactly n 0");
  report.BeginSubmission({.id = 1, .enqueued_ns = 0}, true);
  for (const int64_t lead_ms: {70, 20, 120, 10, 90, 50, 110, 30, 100, 60, 40, 80}) {
    report.AddCompile(Kind::Compute, 1, 0, 1.0, Thread::Gpu, lead_ms * kMs);
  }
  // A guest id that shares the helpers' tag bits must still count. Its 60 ms lead keeps the
  // nearest-rank percentiles of the 13 leads at 20, 60 and 110.
  constexpr uint64_t kTagLookalikeId = 0xFFFF0000DEADBEEFull;
  report.AddCompile(Kind::Compute, kTagLookalikeId, 0, 1.0, Thread::Gpu, 60 * kMs);
  report.AddCompile(Kind::Graphics, 2, 3, 1.0, Thread::Gpu, kHugeLeadMs * kMs);  // Not compute.
  report.AddCompile(Kind::Compute, Common::FirstRunReport::HELPER_ID_TILER, 0, 1.0, Thread::Gpu,
                    kHugeLeadMs * kMs); // A helper pipeline.
  report.AddCompile(Kind::Compute, Common::FirstRunReport::HELPER_ID_PSSR, 0, 1.0, Thread::Gpu,
                    kHugeLeadMs * kMs); // Another helper.
  report.AddCompile(Kind::Compute, 4, 0, 1.0, Thread::Worker, kHugeLeadMs * kMs); // A worker.
  report.AddCompile(Kind::Compute, 5, 0, 1.0, Thread::Gpu);                       // No start.
  report.SetPhase(Phase::Boot);
  report.AddCompile(Kind::Compute, 6, 0, 1.0, Thread::Gpu, kHugeLeadMs * kMs); // Warm-up.
  report.SetPhase(Phase::Play);
  report.EndSubmission();
  report.AddCompile(Kind::Compute, 7, 0, 1.0, Thread::Gpu, kHugeLeadMs * kMs); // Between.
  const auto leads = report.Summarize().submit_leads;
  Check(leads.n == 13, "the lead count includes a row it should not or misses one");
  Check(Near(leads.p10_ms, 20.0) && Near(leads.p50_ms, 60.0) && Near(leads.p90_ms, 110.0) &&
            Near(leads.max_ms, 120.0),
        "a lead percentile is not the nearest rank");
  report.Reset();
  Check(report.Summarize().submit_leads.n == 0, "Reset kept the leads");
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
  const auto json = Compact(report.ToJson({}, 0, 1.0, 0.0, {}));
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
	const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}));
	return json.contains(GENERATION_KEY) ? json[GENERATION_KEY] : nlohmann::json(nullptr);
}

// A report that had no generation asked of it has no block, and one that was refused has the
// reason and nothing else. Every other key of the two reports is the same, which is what a run
// without the flag must keep.
void TestGenerationBlockFlagOffAndInactive() {
	Report     off;
	const auto off_json = nlohmann::json::parse(off.ToJson({}, 0, 1.0, 0.0, {}));
	Check(!off_json.contains(GENERATION_KEY), "a report without the flag had a generation block");

	Report refused;
	refused.SetGenerationInactive("no_eboot");
	const auto refused_json = nlohmann::json::parse(refused.ToJson({}, 0, 1.0, 0.0, {}));
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
	Check(!nlohmann::json::parse(refused.ToJson({}, 0, 1.0, 0.0, {})).contains(GENERATION_KEY),
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
	    {"files_removed", [](auto& b) { b.files_removed = 3; }, "/files_removed", 3},
	    {"remove_failures", [](auto& b) { b.remove_failures = 2; }, "/remove_failures", 2},
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
	    // The same row with a module id gains exactly that key, and the row above has none.
	    {"dropped_head",
	     [](auto& b) {
		     b.dropped = {{.hash      = 0x77,
		                   .rank      = 5,
		                   .reason    = "push_start_head",
		                   .module_id = 0xabcdef0123456789ull}};
	     },
	     "/supervisor/dropped",
	     nlohmann::json::array({{{"hash", "0000000000000077"},
	                             {"rank", 5},
	                             {"reason", "push_start_head"},
	                             {"module_id", "abcdef0123456789"}}})},
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
	     .how          = "done",
	     .tier         = 0,
	     .wide_tail    = false,
	     .candidates   = 3},
	    {.id         = 0x12,
	     .posted_ns  = 1250 * MS,
	     .adopted_ns = 1500 * MS,
	     .how        = "unused",
	     .tier       = 2,
	     .wide_tail  = true,
	     .candidates = 0},
	    {.id = 0x13, .posted_ns = 1250 * MS, .adopted_ns = 1500 * MS, .how = "unused", .tier = 1}};
	// Two late modules with different values, and hashes with leading zeros.
	figures.misses.late_modules  = {{.id           = 0x0a,
	                                 .code_hash    = 0x00ab,
	                                 .miss_ns      = 1750 * MS,
	                                 .delivered_ns = 2250 * MS,
	                                 .via          = "adopted",
	                                 .create_ms    = 6.5,
	                                 .misses       = 3,
	                                 .adopted_ns   = 2500 * MS},
	                                {.id           = 0x0b,
	                                 .code_hash    = 0x00cd,
	                                 .miss_ns      = 1500 * MS,
	                                 .delivered_ns = 1625 * MS,
	                                 .via          = "refused",
	                                 .misses       = 1}};
	figures.misses.late_overflow = 5;

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
	Check(modules.size() == 3 && modules[0]["id"] == "0000000000000011" &&
	          modules[0]["code_hash"] == "00000000000000ab" && modules[0]["posted_ms"] == 1000.0 &&
	          modules[0]["adopted_ms"] == 1250.0 && modules[0]["ready_ms"] == 2000.0 &&
	          modules[0]["first_use_ms"] == 3000.0 && modules[0]["build_ms"] == 12.5 &&
	          modules[0]["lead_ms"] == -4.5 && modules[0]["how"] == "done",
	      "a module row lacks a figure, or its times are not from the run start");
	Check(modules[1]["how"] == "unused" && !modules[1].contains("ready_ms") &&
	          !modules[1].contains("first_use_ms") && !modules[1].contains("build_ms") &&
	          !modules[1].contains("lead_ms"),
	      "a module that was never built shows a time it did not have");
	// Two rows with different values, so a writer of constants fails. A known count of 0 is a
	// number and an unknown count is absent, so the two cannot be mistaken for each other.
	Check(modules[0]["tier"] == 0 && modules[0]["wide_tail"] == false &&
	          modules[0]["candidates"] == 3 && modules[1]["tier"] == 2 &&
	          modules[1]["wide_tail"] == true && modules[1]["candidates"] == 0,
	      "a module row lacks its tier, wide flag or candidates, or they are not the row's own");
	Check(modules[2]["tier"] == 1 && modules[2]["wide_tail"] == false &&
	          !modules[2].contains("candidates"),
	      "a module with an unknown candidate count shows one");
	Check(block["unused_build_ms"] == 7.5, "unused_build_ms is not the figure");
	const auto& misses = block["runtime_misses"];
	Check(misses["total"] == 3 && misses["create_ms"] == 4.25 &&
	          misses["classes"]["wrong_spec"]["count"] == 2 &&
	          misses["classes"]["wrong_spec"]["create_ms"] == 3.0 &&
	          misses["classes"]["unscanned"]["count"] == 1 && misses["refused"]["duplicate"] == 1 &&
	          misses["wrong_spec_programs"] == 1 && misses["wrong_spec_max_variants"] == 2,
	      "the misses lack a figure");
	const auto& late = misses["late_modules"];
	Check(late.size() == 2 && late[0]["id"] == "000000000000000a" &&
	          late[0]["code_hash"] == "00000000000000ab" && late[0]["miss_ms"] == 1500.0 &&
	          late[0]["delivered_ms"] == 2000.0 && late[0]["via"] == "adopted" &&
	          late[0]["create_ms"] == 6.5 && late[0]["misses"] == 3 &&
	          late[0]["adopted_ms"] == 2250.0 && misses["late_overflow"] == 5,
	      "a late module row lacks a figure, or its times are not from the run start");
	Check(late[1]["id"] == "000000000000000b" && late[1]["code_hash"] == "00000000000000cd" &&
	          late[1]["miss_ms"] == 1250.0 && late[1]["delivered_ms"] == 1375.0 &&
	          late[1]["via"] == "refused" && late[1]["create_ms"] == 0.0 &&
	          late[1]["misses"] == 1 && !late[1].contains("adopted_ms"),
	      "the second late module row is not its own");
	Check(empty["runtime_misses"]["late_modules"].is_array() &&
	          empty["runtime_misses"]["late_modules"].empty() &&
	          empty["runtime_misses"]["late_overflow"] == 0,
	      "the empty block lacks the late module keys or lists one");
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
	const auto empty = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["audio_underruns"];
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
	const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["audio_underruns"];
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
	const auto json = nlohmann::json::parse(over.ToJson({}, 0, 1.0, 0.0, {}))["audio_underruns"];
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
	const auto json    = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}));
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
	Check(nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["machine"]["warm_workers"] == 0,
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
	const auto rows = nlohmann::json::parse(global.ToJson({}, 0, 1.0, 0.0, {}))["compiles"];
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

// A warm build counts toward its own lane and the total, and not the other lane. A Front scope is
// the baseline for a Back scope and the reverse, and a scope with no lane counts toward the total
// only. Every count drops back to zero when the scopes end.
void TestWarmBuildScopeLanes() {
	using Common::FirstRunReport::WarmBackBuildsRunning;
	using Common::FirstRunReport::WarmBuildScope;
	using Common::FirstRunReport::WarmBuildsRunning;
	using Common::FirstRunReport::WarmFrontBuildsRunning;
	using Common::FirstRunReport::WarmLane;
	Common::FirstRunReport::Configure(true, false);
	{
		const WarmBuildScope front(WarmLane::Front);
		Check(WarmFrontBuildsRunning() == 1 && WarmBackBuildsRunning() == 0 &&
		          WarmBuildsRunning() == 1,
		      "a front scope did not count as front and total only");
	}
	{
		const WarmBuildScope back(WarmLane::Back);
		Check(WarmFrontBuildsRunning() == 0 && WarmBackBuildsRunning() == 1 &&
		          WarmBuildsRunning() == 1,
		      "a back scope did not count as back and total only");
	}
	{
		const WarmBuildScope other;
		Check(WarmFrontBuildsRunning() == 0 && WarmBackBuildsRunning() == 0 &&
		          WarmBuildsRunning() == 1,
		      "a scope with no lane counted toward a lane");
	}
	Check(WarmFrontBuildsRunning() == 0 && WarmBackBuildsRunning() == 0 && WarmBuildsRunning() == 0,
	      "a finished scope is still counted");
	Common::FirstRunReport::Configure(false, false);
	{
		const WarmBuildScope off(WarmLane::Back);
		Check(WarmBackBuildsRunning() == 0, "a lane was counted while the report was off");
	}
}

// A worker row carries the Front and Back builds it met. The baseline is a row begun with none
// running, and the row begun with 1 Front and 2 Back must carry 1 and 2, not 2 and 1.
void TestWorkerCompileLaneCounts() {
	using Common::FirstRunReport::CompileScope;
	using Common::FirstRunReport::WarmBuildScope;
	using Common::FirstRunReport::WarmLane;
	auto& global = Common::FirstRunReport::Global();
	global.Reset();
	Common::FirstRunReport::Configure(true, false);
	{
		CompileScope baseline(Kind::Graphics, 1, 2, false);
	}
	{
		const WarmBuildScope front(WarmLane::Front);
		const WarmBuildScope back_one(WarmLane::Back);
		const WarmBuildScope back_two(WarmLane::Back);
		CompileScope         busy(Kind::Graphics, 3, 4, false);
	}
	const auto rows = nlohmann::json::parse(global.ToJson({}, 0, 1.0, 0.0, {}))["compiles"];
	Check(rows.size() == 2, "the compile rows were lost");
	if (rows.size() == 2) {
		Check(rows[0].contains("warm_front") && rows[0].contains("warm_back") &&
		          rows[0]["warm_front"] == 0 && rows[0]["warm_back"] == 0,
		      "the baseline row lacks the lane keys or has lanes it did not meet");
		Check(rows[1]["warm_front"] == 1 && rows[1]["warm_back"] == 2 &&
		          rows[1]["warm_running"] == 3,
		      "the row did not carry the lanes it met");
	}
	Common::FirstRunReport::Configure(false, false);
	global.Reset();
}

// Each underrun carries what ran when it happened, in the order of its time. The first has 0 Front
// and 3 Back with the generator alive and the second 2 Front and 0 Back with it not alive, so a
// value that was not stored, or that was taken from the other slot, differs. Totals are not
// compared with the lanes, since they are separate loads.
void TestAudioUnderrunsCarryLanes() {
	using Common::FirstRunReport::WarmBuildScope;
	using Common::FirstRunReport::WarmLane;
	Report report;
	Common::FirstRunReport::Configure(true, false);
	{
		const WarmBuildScope back_one(WarmLane::Back);
		const WarmBuildScope back_two(WarmLane::Back);
		const WarmBuildScope back_three(WarmLane::Back);
		Common::FirstRunReport::SetGeneratorChildAlive(true);
		report.AddAudioUnderrun();
	}
	{
		const WarmBuildScope front_one(WarmLane::Front);
		const WarmBuildScope front_two(WarmLane::Front);
		Common::FirstRunReport::SetGeneratorChildAlive(false);
		report.AddAudioUnderrun();
	}
	Common::FirstRunReport::Configure(false, false);
	const auto taken = report.AudioUnderrunsSoFar();
	Check(taken.at_ns.size() == 2 && taken.warm_front.size() == 2 && taken.warm_back.size() == 2 &&
	          taken.child_alive.size() == 2,
	      "the underrun fields are not parallel to the times");
	if (taken.at_ns.size() == 2 && taken.warm_front.size() == 2 && taken.warm_back.size() == 2 &&
	    taken.child_alive.size() == 2) {
		Check(taken.warm_front[0] == 0 && taken.warm_back[0] == 3 && taken.child_alive[0] == 1,
		      "the first underrun lost its lanes or its generator");
		Check(taken.warm_front[1] == 2 && taken.warm_back[1] == 0 && taken.child_alive[1] == 0,
		      "the second underrun lost its lanes or its generator");
	}
	const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["audio_underruns"];
	Check(json["warm_front"] == nlohmann::json::array({0, 2}) &&
	          json["warm_back"] == nlohmann::json::array({3, 0}) &&
	          json["child_alive"] == nlohmann::json::array({1, 0}),
	      "the underrun arrays in the file are not parallel to at_ms");
	// Reset clears the counts with the times, so a refill starts from nothing.
	report.Reset();
	Check(report.AudioUnderrunsSoFar().warm_back.empty(), "Reset kept the underrun counts");
	report.AddAudioUnderrun();
	const auto again = report.AudioUnderrunsSoFar();
	Check(again.warm_front.size() == 1 && again.warm_front[0] == 0 && again.warm_back[0] == 0 &&
	          again.child_alive[0] == 0,
	      "Reset left old counts in a reused slot");
	const auto empty = nlohmann::json::parse(Report().ToJson({}, 0, 1.0, 0.0, {}))["audio_underruns"];
	Check(empty["warm_front"].is_array() && empty["warm_front"].empty() &&
	          empty["child_alive"].is_array(),
	      "an empty report lacks the underrun arrays");
}

// The flag-off free function stores no lanes, and an overflowed underrun adds no entry to them. The
// baseline is the same call with the report on, which stores one.
void TestAudioUnderrunLanesFlagAndOverflow() {
	auto& global = Common::FirstRunReport::Global();
	global.Reset();
	Common::FirstRunReport::Configure(false, false);
	Common::FirstRunReport::AddAudioUnderrun();
	Check(global.AudioUnderrunsSoFar().warm_front.empty() &&
	          global.AudioUnderrunsSoFar().child_alive.empty(),
	      "lanes were stored while the report was off");
	Common::FirstRunReport::Configure(true, false);
	Common::FirstRunReport::AddAudioUnderrun();
	Check(global.AudioUnderrunsSoFar().warm_front.size() == 1,
	      "lanes were not stored with the report on");
	Common::FirstRunReport::Configure(false, false);
	global.Reset();

	constexpr size_t capacity = Common::FirstRunReport::AUDIO_UNDERRUN_CAPACITY;
	Report           over;
	for (size_t index = 0; index < capacity + 3; index++) {
		over.AddAudioUnderrun();
	}
	const auto taken = over.AudioUnderrunsSoFar();
	Check(taken.overflow == 3 && taken.at_ns.size() == capacity &&
	          taken.warm_front.size() == capacity && taken.warm_back.size() == capacity &&
	          taken.child_alive.size() == capacity,
	      "an overflow changed the stored lanes or was not counted");
}

// The held time by cause is written for every cause, prescan ones at 0 included, and adds up to the
// held total within 1 ms. The total is a number the test chose and the causes are written from a
// breakdown of 1 ms for the first cause, 2 ms for the second and so on, so a cause dropped from the
// file leaves the sum short of the total and one fewer key.
void TestHeldByCauseInJson() {
  namespace GS = Common::GpuStall;
  GS::HoldBreakdown holds;
  double            total_ms = 0.0;
  for (size_t cause = 0; cause < GS::HOLD_CAUSE_COUNT; cause++) {
    holds.cause_ns[cause]    = (cause + 1) * 1'000'000;
    holds.cause_count[cause] = cause + 1;
    total_ms += static_cast<double>(cause + 1);
  }
  holds.histogram.Add(30'000'000);
  holds.exact = {.count = 7, .overflow = 2, .p50_ms = 11.5, .p95_ms = 12.5, .max_ms = 13.5};
  const auto json = nlohmann::json::parse(Report().ToJson({}, 3, 1.0, total_ms, holds));
  const auto held = json["audio_held_by_cause"]["held_ms"];
  Check(held.size() == GS::HOLD_CAUSE_COUNT, "a cause is missing from the held-by-cause object");
  Check(json["audio_held_by_cause"]["holds"].size() == GS::HOLD_CAUSE_COUNT,
        "a cause is missing from the hold counts");
  double summed = 0.0;
  for (const auto &entry : held.items()) summed += entry.value().get<double>();
  Check(std::abs(summed - json["audio_held_ms"].get<double>()) < 1.0,
        "the causes in the file do not add up to the held total");
  for (const char *name :
       {"required_graphics", "required_compute", "warm_inline", "warm_inline_paused",
        "generated_inline", "generated_inline_paused", "warm_waited", "generated_waited", "prescan_waited", "prescan_in_flight",
        "prescan_never_posted_async", "prescan_never_posted_indirect", "prescan_never_posted_deep",
        "prescan_refused", "deferred_draw_resolve", "ineligible_draw", "library_once", "prescan_queued",
        "other"})
    Check(held.contains(name), name);
  Check(held["required_graphics"] == 1.0 &&
            held["other"] == static_cast<double>(GS::HOLD_CAUSE_COUNT),
        "a cause was written under another cause's name");
  const auto histogram = json["audio_hold_histogram"];
  Check(histogram["count"] == 1 && histogram["max_ms"] == 30.0 && histogram["p50_bound_ms"] == 30.0 &&
            histogram["p95_bound_ms"] == 30.0,
        "the hold histogram is not in the file");
  const auto exact = json["audio_holds_exact"];
  Check(exact["count"] == 7 && exact["overflow"] == 2 && exact["p50_ms"] == 11.5 &&
            exact["p95_ms"] == 12.5 && exact["max_ms"] == 13.5,
        "the exact hold figures are not in the file as given");
  const auto empty = nlohmann::json::parse(Report().ToJson({}, 0, 1.0, 0.0, {}));
  Check(empty["audio_held_by_cause"]["held_ms"].size() == GS::HOLD_CAUSE_COUNT &&
            empty["audio_held_by_cause"]["held_ms"]["other"] == 0.0,
        "a report with no breakdown lacks the causes");
}

// The prescan tag is its own count. The first underrun has the generator alive and no pre-scan work,
// the second three units of pre-scan work and no generator, and the third more than a uint16 holds,
// so a tag that was copied from the generator's flag, never stored, or not saturated, differs.
void TestUnderrunsCarryThePrescanTag() {
  Report report;
  Common::FirstRunReport::Configure(true, false);
  Check(Common::FirstRunReport::PrescanRunning() == 0, "the pre-scan tag is not 0 by default");
  Common::FirstRunReport::SetGeneratorChildAlive(true);
  report.AddAudioUnderrun();
  Common::FirstRunReport::SetGeneratorChildAlive(false);
  Common::FirstRunReport::SetPrescanRunningForTest(3);
  report.AddAudioUnderrun();
  Common::FirstRunReport::SetPrescanRunningForTest(70000);
  report.AddAudioUnderrun();
  Common::FirstRunReport::SetPrescanRunningForTest(0);
  Common::FirstRunReport::Configure(false, false);
  const auto taken = report.AudioUnderrunsSoFar();
  Check(taken.prescan_running == std::vector<uint16_t> {0, 3, 65535} &&
            taken.child_alive == std::vector<uint8_t> {1, 0, 0},
        "the pre-scan tag is not kept apart from the generator's");
  const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}))["audio_underruns"];
  Check(json["prescan_running"] == nlohmann::json::array({0, 3, 65535}),
        "the pre-scan tag is not in the file");
  report.Reset();
  report.AddAudioUnderrun();
  Check(report.AudioUnderrunsSoFar().prescan_running == std::vector<uint16_t> {0},
        "Reset left an old pre-scan tag in a reused slot");
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

// ---- The scanner's part of the prescan block. ----

using Common::FirstRunReport::PrescanScannerReport;

constexpr const char* PRESCAN_KEY = "prescan";

// The prescan block of a report's JSON, or null when it has none.
nlohmann::json PrescanBlock(const Report& report) {
	const auto json = nlohmann::json::parse(report.ToJson({}, 0, 1.0, 0.0, {}));
	return json.contains(PRESCAN_KEY) ? json[PRESCAN_KEY] : nlohmann::json(nullptr);
}

// A report with no scanner figures has no prescan block, which is how a run with the prescan off keeps
// its report as it was. The baseline is the same report with a source, which has the block and differs
// in nothing else. A final figure alone makes the block too, and Reset takes it away. A writer that
// always wrote the block would show it in the first report.
void TestPrescanBlockAbsentWithoutSource() {
	Report     off;
	const auto off_json = nlohmann::json::parse(off.ToJson({}, 0, 1.0, 0.0, {}));
	Check(!off_json.contains(PRESCAN_KEY), "a report without a scanner had a prescan block");

	Report on;
	on.SetPrescanScanSource([] { return PrescanScannerReport {}; });
	auto on_json = nlohmann::json::parse(on.ToJson({}, 0, 1.0, 0.0, {}));
	Check(on_json.contains(PRESCAN_KEY), "a report with a scanner source had no prescan block");
	on_json.erase(PRESCAN_KEY);
	// The process peaks are read when each report is written, so they are left out of the comparison.
	auto off_without_peaks = off_json;
	off_without_peaks.erase("process");
	on_json.erase("process");
	Check(on_json == off_without_peaks, "the prescan block changed another part of the report");

	Report final_only;
	final_only.SetPrescanScanFinal({});
	Check(!PrescanBlock(final_only).is_null(), "final scanner figures alone left no prescan block");
	// Reset takes away a source and final figures alike.
	on.Reset();
	Check(PrescanBlock(on).is_null(), "Reset kept a prescan block from a source");
	final_only.Reset();
	Check(PrescanBlock(final_only).is_null(), "Reset kept a prescan block from final figures");
}

// One figure of the scanner's report, and where it shows in the block.
struct PrescanFieldCase {
	const char*                                             pointer;
	std::function<void(PrescanScannerReport&, uint64_t)> set;
};

// Each case sets one plain figure of the report to the value it is given.
#define PRESCAN_FIELD(pointer, member)                                                    \
	PrescanFieldCase {                                                                    \
		pointer, [](PrescanScannerReport& figures, uint64_t value) { figures.member = value; } \
	}

// The walks' counts and the walk time.
std::vector<PrescanFieldCase> PrescanScanFieldCases() {
	return {PRESCAN_FIELD("/scan/submissions", scan.submissions),
	        PRESCAN_FIELD("/scan/dispatches", scan.dispatches),
	        PRESCAN_FIELD("/scan/no_shader_state", scan.no_shader_state),
	        PRESCAN_FIELD("/scan/program_inherited", scan.program_inherited),
	        PRESCAN_FIELD("/scan/uncertain_program", scan.uncertain_program),
	        PRESCAN_FIELD("/scan/ib_followed", scan.ib_followed),
	        PRESCAN_FIELD("/scan/ib_unreadable", scan.ib_unreadable),
	        PRESCAN_FIELD("/scan/ib_depth_limited", scan.ib_depth_limited),
	        PRESCAN_FIELD("/scan/ib_revisited", scan.ib_revisited),
	        PRESCAN_FIELD("/scan/budget_exhausted", scan.budget_exhausted),
	        PRESCAN_FIELD("/scan/bad_packets", scan.bad_packets),
	        PRESCAN_FIELD("/scan/resets", scan.resets),
	        PRESCAN_FIELD("/scan/dwords_read", scan.dwords_read),
	        PRESCAN_FIELD("/scan/dwords_max", scan.dwords_max),
	        PRESCAN_FIELD("/scan/dispatches_cached", scan.dispatches_cached),
	        PRESCAN_FIELD("/scan/time_capped", scan.time_capped),
	        PRESCAN_FIELD("/scan/scan_allocations", scan.scan_allocations),
	        PRESCAN_FIELD("/scan/cache_remembered_busy", scan.cache_remembered_busy),
	        PRESCAN_FIELD("/scan/depth_max", scan.depth_max),
	        PRESCAN_FIELD("/scan/register_list_unreadable", scan.register_list_unreadable),
	        PRESCAN_FIELD("/scan/branch_mode1", scan.branch_mode1),
	        PRESCAN_FIELD("/scan/branch_mode2", scan.branch_mode2),
	        PRESCAN_FIELD("/scan/branch_mode2_empty_else", scan.branch_mode2_empty_else),
	        PRESCAN_FIELD("/scan/ib_revisit_doubts", scan.ib_revisit_doubts),
	        PRESCAN_FIELD("/scan/scan_us/p50", scan.scan_us.p50),
	        PRESCAN_FIELD("/scan/scan_us/p90", scan.scan_us.p90),
	        PRESCAN_FIELD("/scan/scan_us/p99", scan.scan_us.p99),
	        PRESCAN_FIELD("/scan/scan_us/max", scan.scan_us.max),
	        PRESCAN_FIELD("/scan/scan_at_least_1024_us", scan.scan_at_least_1024_us),
	        {"/scan/uncertain_cleared_by/branch",
	         [](PrescanScannerReport& figures, uint64_t value) {
		         figures.scan.uncertain_cleared_by = {{"branch", value}};
	         }}};
}

// The queue's counts and its three times.
std::vector<PrescanFieldCase> PrescanQueueFieldCases() {
	return {PRESCAN_FIELD("/queue/pushed", queue.pushed),
	        PRESCAN_FIELD("/queue/dropped", queue.dropped),
	        PRESCAN_FIELD("/queue/dropped_no_buffer", queue.dropped_no_buffer),
	        PRESCAN_FIELD("/queue/late_skipped", queue.late_skipped),
	        PRESCAN_FIELD("/queue/markers_lost", queue.markers_lost),
	        PRESCAN_FIELD("/queue/truncated", queue.truncated),
	        PRESCAN_FIELD("/queue/copied_dwords", queue.copied_dwords),
	        PRESCAN_FIELD("/queue/peak", queue.peak),
	        PRESCAN_FIELD("/queue/copy_us/p50", queue.copy_us.p50),
	        PRESCAN_FIELD("/queue/copy_us/p90", queue.copy_us.p90),
	        PRESCAN_FIELD("/queue/copy_us/p99", queue.copy_us.p99),
	        PRESCAN_FIELD("/queue/copy_us/max", queue.copy_us.max),
	        PRESCAN_FIELD("/queue/push_us/p50", queue.push_us.p50),
	        PRESCAN_FIELD("/queue/push_us/p90", queue.push_us.p90),
	        PRESCAN_FIELD("/queue/push_us/p99", queue.push_us.p99),
	        PRESCAN_FIELD("/queue/push_us/max", queue.push_us.max),
	        PRESCAN_FIELD("/queue/lag_ms/p50", queue.lag_ms.p50),
	        PRESCAN_FIELD("/queue/lag_ms/p90", queue.lag_ms.p90),
	        PRESCAN_FIELD("/queue/lag_ms/p99", queue.lag_ms.p99),
	        PRESCAN_FIELD("/queue/lag_ms/max", queue.lag_ms.max)};
}

// The thread's counts, the reads and the offers.
std::vector<PrescanFieldCase> PrescanThreadFieldCases() {
	return {PRESCAN_FIELD("/thread/jobs", jobs),
	        PRESCAN_FIELD("/thread/lag_skipped", lag_skipped),
	        PRESCAN_FIELD("/thread/stopped_by_preparer", stopped_by_preparer),
	        PRESCAN_FIELD("/thread/wakes", wakes),
	        PRESCAN_FIELD("/reads/calls", read_calls),
	        PRESCAN_FIELD("/reads/max_us", read_max_us),
	        PRESCAN_FIELD("/reads/over_1ms", read_over_1ms),
	        PRESCAN_FIELD("/offer/posted", offers_posted),
	        PRESCAN_FIELD("/offer/skipped", offers_skipped),
	        PRESCAN_FIELD("/offer/dropped", offers_dropped)};
}

#undef PRESCAN_FIELD

// Every figure of the scanner's report has its own place in the block. Each case sets one figure of an
// empty report to a value no other case uses, and the block then differs from the empty one in that
// place only, so the baseline is the same block without the change. Two figures written under each
// other's keys, or one not written, would put the value in the wrong place or nowhere.
void TestPrescanScanFieldsRoundTrip() {
	const auto block_of = [](const PrescanScannerReport& figures) {
		Report report;
		report.SetPrescanScanFinal(figures);
		return PrescanBlock(report);
	};
	const auto base = block_of({});
	Check(base.is_object() && base["scan"]["submissions"] == 0 && base["scan"]["uncertain_cleared_by"].empty() &&
	          base["offer"]["dropped"] == 0,
	      "the empty prescan block is not empty");
	auto cases = PrescanScanFieldCases();
	for (auto& more: {PrescanQueueFieldCases(), PrescanThreadFieldCases()}) {
		cases.insert(cases.end(), more.begin(), more.end());
	}
	// Distinct from 0 and from one another, so a figure that took another's value shows.
	constexpr uint64_t FIRST_VALUE = 1001;
	for (size_t index = 0; index < cases.size(); ++index) {
		const uint64_t       value = FIRST_VALUE + index;
		PrescanScannerReport figures;
		cases[index].set(figures, value);
		auto expected                                                = base;
		expected[nlohmann::json::json_pointer(cases[index].pointer)] = value;
		const auto actual                                            = block_of(figures);
		if (actual != expected || actual == base) {
			std::fprintf(stderr, "FirstRunReportTests: the prescan figure %s is not where it belongs\n",
			             cases[index].pointer);
			std::abort();
		}
	}
}

// With a source, the writer asks it each time. A final figure replaces it at once and the source is
// never called again, so the GPU may destroy the scanner the source reads. The baseline is the same
// report before the final figure, which shows the source's figure.
void TestPrescanScanFinalReplacesSource() {
	Report           report;
	std::atomic<int> calls {0};
	report.SetPrescanScanSource([&calls] {
		PrescanScannerReport figures;
		figures.scan.submissions = 3;
		calls++;
		return figures;
	});
	Check(PrescanBlock(report)["scan"]["submissions"] == 3 && calls == 1,
	      "the writer did not use the scanner source");
	PrescanScannerReport final_figures;
	final_figures.scan.submissions = 9;
	report.SetPrescanScanFinal(final_figures);
	Check(PrescanBlock(report)["scan"]["submissions"] == 9, "the final scanner figures were not used");
	Check(calls == 1, "the scanner source was called after the final figures replaced it");
}

// The GPU destroys the scanner right after SetPrescanScanFinal returns. A writer that called the source
// after that would read a destroyed object, which this test models as a flag that the source checks.
// The baseline is the live source before the hand-over, which is called without a violation.
void TestPrescanWriterNeverTouchesADestroyedSource() {
	Report            report;
	std::atomic<bool> alive {true};
	std::atomic<int>  violations {0};
	report.SetPrescanScanSource([&] {
		if (!alive) violations++;
		return PrescanScannerReport {};
	});
	Check(PrescanBlock(report).contains("scan") && violations == 0, "the live scanner source failed");
	report.SetPrescanScanFinal({});
	alive            = false;
	const auto after = PrescanBlock(report);
	Check(violations == 0, "the writer called a scanner source that was destroyed");
	Check(after.contains("scan"), "the final scanner figures were not written after the source went");
}

// The writer is inside the source when the GPU hands over the final figures. The hand-over must wait
// for the writer, because the GPU destroys the scanner right after it, and the writer must be out of
// the source by then. The source here is slow on purpose. A writer that copied the source and called
// it after unlocking would let the hand-over finish while it was still inside.
void TestPrescanFinalWaitsForAWriterInTheSource() {
	Report            report;
	std::atomic<bool> inside {false};
	std::atomic<bool> release {false};
	std::atomic<bool> left_source {false};
	report.SetPrescanScanSource([&] {
		inside = true;
		while (!release) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		left_source = true;
		return PrescanScannerReport {};
	});
	std::thread writer([&] { (void)PrescanBlock(report); });
	while (!inside) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	std::atomic<bool> handed_over {false};
	std::thread       owner([&] {
        report.SetPrescanScanFinal({});
        handed_over = true;
    });
	// A hand-over that did not wait would be done by now.
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	const bool waited = !handed_over;
	release           = true;
	writer.join();
	owner.join();
	Check(waited, "the final scanner figures were handed over while a writer was in the source");
	Check(handed_over && left_source, "the hand-over or the writer never finished");
}

// The generation block's drain says it takes the prescan's modules too only when it does, so a
// run with both producers does not credit the prescan's work to the generator. The baseline is the
// same figures without the mark, whose drain has no such key, which is how a run without the
// prescan keeps its report as it was. A writer that always wrote the key fails the baseline, and
// one that never did fails the marked block. The mark changes nothing else in the block.
void TestGenerationDrainMarksPrescan() {
	const auto block_of = [](const ShaderStoreGenerationReport& figures) {
		Report report;
		report.SetGenerationFinal(figures);
		return GenerationBlock(report);
	};
	ShaderStoreGenerationReport figures;
	const auto                  unmarked = block_of(figures);
	figures.drain.includes_prescan       = true;
	const auto marked                    = block_of(figures);
	Check(unmarked.is_object() && !unmarked["drain"].contains("includes_prescan"),
	      "a generation without the prescan marked its drain");
	Check(marked["drain"].contains("includes_prescan") &&
	          marked["drain"]["includes_prescan"] == true,
	      "a generation beside the prescan did not mark its drain");
	auto without_mark = marked;
	without_mark["drain"].erase("includes_prescan");
	Check(without_mark == unmarked, "the mark changed another part of the generation block");
}

// ---- The pipeline cache's part of the prescan block. ----

using Common::FirstRunReport::PrescanCacheReport;

// The keys each part of the prescan block writes at its top level.
const std::vector<std::string> PRESCAN_SCANNER_KEYS = {"scan", "queue", "thread", "reads", "offer"};
const std::vector<std::string> PRESCAN_CACHE_KEYS   = {"state", "preparer", "drain",  "builds",
                                                       "holds", "misses",   "records"};

// The top-level keys of a JSON object, in the object's own order.
std::vector<std::string> KeysOf(const nlohmann::json& object) {
	std::vector<std::string> keys;
	for (const auto& item: object.items()) {
		keys.push_back(item.key());
	}
	return keys;
}

// Whether the block holds every key of the list.
bool HasAllKeys(const nlohmann::json& block, const std::vector<std::string>& keys) {
	for (const auto& key: keys) {
		if (!block.contains(key)) return false;
	}
	return true;
}

// Whether the block holds no key of the list.
bool HasNoKey(const nlohmann::json& block, const std::vector<std::string>& keys) {
	for (const auto& key: keys) {
		if (block.contains(key)) return false;
	}
	return true;
}

// Cache figures whose every part differs from an empty one, so a part written from the wrong
// figures shows.
PrescanCacheReport SomeCacheFigures() {
	PrescanCacheReport figures;
	figures.state.posted                 = 2;
	figures.preparer.items               = 3;
	figures.drain.adopted                = 4;
	figures.builds.builds                = 5;
	figures.records.unused_at_exit       = 6;
	figures.holds.in_flight_at_miss_only = 7;
	figures.misses.inline_compiles       = 8;
	return figures;
}

// The prescan block of a report given the final figures of either part, or both. A report is
// large, so each is made in a call of its own and gone before the next.
nlohmann::json BlockOfParts(const std::optional<PrescanScannerReport>& scanner,
                            const std::optional<PrescanCacheReport>&   cache) {
	Report report;
	if (scanner) report.SetPrescanScanFinal(*scanner);
	if (cache) report.SetPrescanCacheFinal(*cache);
	return PrescanBlock(report);
}

// Either part alone makes the prescan block with its own keys only, and with both the block holds
// the two parts side by side, each as it is alone. The baseline of each part is the other part
// alone, and the block with neither is absent, as the scanner checks show. A writer that dropped
// the cache's figures would leave the cache alone with no block. One that let one part replace the
// other would lose the scanner's keys or the cache's when both are there. A child block is written
// only when there is one.
void TestPrescanCacheBlockMerges() {
	const auto                     cache_block = BlockOfParts(std::nullopt, SomeCacheFigures());
	const std::vector<std::string> cache_keys  = {"builds", "drain",   "holds", "misses",
	                                              "preparer", "records", "state"};
	Check(cache_block.is_object() && KeysOf(cache_block) == cache_keys,
	      "the cache alone did not make a block of exactly its own keys");
	Check(cache_block["state"]["posted"] == 2 && cache_block["records"]["unused_at_exit"] == 6 &&
	          cache_block["holds"]["in_flight_at_miss_only"] == 7,
	      "the cache alone wrote other figures than its own");
	Check(cache_block["misses"]["inline_compiles"] == 8,
	      "the cache alone wrote other miss figures than its own");
	const auto scan_block = BlockOfParts(PrescanScannerReport {}, std::nullopt);
	Check(HasAllKeys(scan_block, PRESCAN_SCANNER_KEYS) &&
	          HasNoKey(scan_block, PRESCAN_CACHE_KEYS) && !scan_block.contains("child"),
	      "the scanner alone wrote keys of the cache");
	const auto both_block = BlockOfParts(PrescanScannerReport {}, SomeCacheFigures());
	Check(HasAllKeys(both_block, PRESCAN_SCANNER_KEYS) &&
	          HasAllKeys(both_block, PRESCAN_CACHE_KEYS),
	      "with both parts the block lost the keys of one");
	for (const auto& key: PRESCAN_SCANNER_KEYS) {
		Check(both_block[key] == scan_block[key], "a scanner key changed");
	}
	for (const auto& key: PRESCAN_CACHE_KEYS) {
		Check(both_block[key] == cache_block[key], "a cache key changed");
	}
	auto with_child = SomeCacheFigures();
	with_child.child.emplace();
	Check(!cache_block.contains("child") &&
	          BlockOfParts(std::nullopt, with_child).contains("child"),
	      "the child block did not follow whether there were child figures");
}

// The cache's part changes nothing else in the report. The baseline is the same report without
// it, which differs only in the prescan block. Reset takes the cache's source away, and its final
// figures.
void TestPrescanCacheBlockLeavesTheRest() {
	Report cache_only;
	cache_only.SetPrescanCacheSource([] { return SomeCacheFigures(); });
	auto cache_json = nlohmann::json::parse(cache_only.ToJson({}, 0, 1.0, 0.0, {}));
	Check(cache_json.contains(PRESCAN_KEY), "a report with a cache source had no prescan block");
	nlohmann::json off_json;
	{
		Report off;
		off_json = nlohmann::json::parse(off.ToJson({}, 0, 1.0, 0.0, {}));
	}
	cache_json.erase(PRESCAN_KEY);
	// The process peaks are read when each report is written, so they are left out of the
	// comparison.
	off_json.erase("process");
	cache_json.erase("process");
	Check(cache_json == off_json, "the cache's prescan block changed another part of the report");
	cache_only.Reset();
	Check(PrescanBlock(cache_only).is_null(), "Reset kept a prescan block from a cache source");
	cache_only.SetPrescanCacheFinal(SomeCacheFigures());
	cache_only.Reset();
	Check(PrescanBlock(cache_only).is_null(),
	      "Reset kept a prescan block from cache final figures");
}

// One figure of the cache's report, and where it shows in the block. The setter puts the value it
// is given into the figures, in the figure's own type, and returns what the block must hold for it.
struct PrescanCacheFieldCase {
	const char*                                                  pointer;
	std::function<nlohmann::json(PrescanCacheReport&, uint64_t)> set;
};

// A fraction a time keeps through the writer's rounding to thousandths, so a time differs from a
// count.
constexpr double PRESCAN_TIME_FRACTION = 0.25;

// A count, set to the value itself.
#define CACHE_COUNT(pointer, member)                                                  \
	PrescanCacheFieldCase {                                                           \
		pointer, [](PrescanCacheReport& figures, uint64_t value) -> nlohmann::json {  \
			figures.member = value;                                                   \
			return value;                                                             \
		}                                                                             \
	}
// A time in milliseconds, set to the value and a fraction.
#define CACHE_TIME(pointer, member)                                                   \
	PrescanCacheFieldCase {                                                           \
		pointer, [](PrescanCacheReport& figures, uint64_t value) -> nlohmann::json {  \
			figures.member = static_cast<double>(value) + PRESCAN_TIME_FRACTION;      \
			return static_cast<double>(value) + PRESCAN_TIME_FRACTION;                \
		}                                                                             \
	}
// A flag, set to true.
#define CACHE_FLAG(pointer, member)                                                   \
	PrescanCacheFieldCase {                                                           \
		pointer, [](PrescanCacheReport& figures, uint64_t) -> nlohmann::json {        \
			figures.member = true;                                                    \
			return true;                                                              \
		}                                                                             \
	}

// The state's counts and the preparer's counts before it sends anything.
std::vector<PrescanCacheFieldCase> PrescanStateIntakeCases() {
	return {CACHE_COUNT("/state/posted", state.posted),
	        CACHE_COUNT("/state/busy", state.busy),
	        CACHE_COUNT("/state/capped", state.capped),
	        CACHE_COUNT("/state/settled", state.settled),
	        CACHE_COUNT("/state/uncertain", state.uncertain),
	        CACHE_COUNT("/state/dropped_full", state.dropped_full),
	        CACHE_COUNT("/preparer/items", preparer.items),
	        CACHE_COUNT("/preparer/no_map_entry", preparer.no_map_entry),
	        CACHE_COUNT("/preparer/zero_size", preparer.zero_size),
	        CACHE_COUNT("/preparer/indirect_thread_dims", preparer.indirect_thread_dims),
	        CACHE_COUNT("/preparer/bad_item", preparer.bad_item),
	        CACHE_COUNT("/preparer/backed_off", preparer.backed_off),
	        CACHE_COUNT("/preparer/blocked", preparer.blocked),
	        CACHE_COUNT("/preparer/plan_requests", preparer.plan_requests),
	        CACHE_COUNT("/preparer/compile_requests", preparer.compile_requests),
	        CACHE_COUNT("/preparer/code_unreadable", preparer.code_unreadable),
	        CACHE_COUNT("/preparer/code_changed", preparer.code_changed),
	        CACHE_COUNT("/preparer/plan_failed", preparer.plan_failed),
	        CACHE_COUNT("/preparer/has_calls", preparer.has_calls),
	        CACHE_COUNT("/preparer/srt_faults", preparer.srt_faults),
	        CACHE_COUNT("/preparer/materialize_failed", preparer.materialize_failed),
	        CACHE_COUNT("/preparer/indirect_args_unknown", preparer.indirect_args_unknown),
	        {"/preparer/invalid/image_mips",
	         [](PrescanCacheReport& figures, uint64_t value) -> nlohmann::json {
		         figures.preparer.invalid = {{"image_mips", value}};
		         return value;
	         }}};
}

// The preparer's variants, outcomes and modules.
std::vector<PrescanCacheFieldCase> PrescanPreparerOutcomeCases() {
	return {CACHE_COUNT("/preparer/variant_modules", preparer.variant_modules),
	        CACHE_COUNT("/preparer/known_variants", preparer.known_variants),
	        CACHE_COUNT("/preparer/capped_variants", preparer.capped_variants),
	        CACHE_COUNT("/preparer/known_skipped", preparer.known_skipped),
	        CACHE_COUNT("/preparer/heap_image_variants", preparer.heap_image_variants),
	        CACHE_COUNT("/preparer/covered_variants", preparer.covered_variants),
	        CACHE_COUNT("/preparer/known_modules", preparer.known_modules),
	        CACHE_COUNT("/preparer/known_dropped", preparer.known_dropped),
	        CACHE_COUNT("/preparer/refused_sends", preparer.refused_sends),
	        CACHE_COUNT("/preparer/never_retry", preparer.never_retry),
	        CACHE_COUNT("/preparer/blamed", preparer.blamed),
	        CACHE_COUNT("/preparer/dropped", preparer.dropped),
	        CACHE_COUNT("/preparer/compile_failed", preparer.compile_failed),
	        CACHE_COUNT("/preparer/key_mismatch", preparer.key_mismatch),
	        CACHE_COUNT("/preparer/decode_failed", preparer.decode_failed),
	        CACHE_COUNT("/preparer/adopted", preparer.adopted),
	        CACHE_COUNT("/preparer/taken", preparer.taken),
	        CACHE_COUNT("/preparer/late_duplicates", preparer.late_duplicates),
	        CACHE_COUNT("/preparer/taken_dropped", preparer.taken_dropped),
	        CACHE_COUNT("/preparer/plans_dropped", preparer.plans_dropped),
	        CACHE_COUNT("/preparer/forgotten", preparer.forgotten),
	        CACHE_COUNT("/preparer/verdicts_evicted", preparer.verdicts_evicted),
	        CACHE_FLAG("/preparer/failed", preparer.failed),
	        CACHE_FLAG("/preparer/lane_off", preparer.lane_off)};
}

// The child's counts, its two words and its round trips.
std::vector<PrescanCacheFieldCase> PrescanChildCases() {
	return {CACHE_COUNT("/child/spawns", child->spawns),
	        CACHE_COUNT("/child/failed_spawns", child->failed_spawns),
	        CACHE_COUNT("/child/crashes", child->crashes),
	        CACHE_COUNT("/child/hangs", child->hangs),
	        CACHE_COUNT("/child/protocol_errors", child->protocol_errors),
	        CACHE_COUNT("/child/failures", child->failures),
	        CACHE_COUNT("/child/never_retry", child->never_retry),
	        CACHE_COUNT("/child/never_retry_programs", child->never_retry_programs),
	        CACHE_FLAG("/child/lane_off", child->lane_off),
	        CACHE_COUNT("/child/submitted", child->submitted),
	        CACHE_COUNT("/child/answered", child->answered),
	        CACHE_COUNT("/child/blamed", child->blamed),
	        CACHE_COUNT("/child/dropped", child->dropped),
	        CACHE_COUNT("/child/resent", child->resent),
	        CACHE_COUNT("/child/refused", child->refused),
	        CACHE_COUNT("/child/late_replies", child->late_replies),
	        CACHE_COUNT("/child/unexpected_replies", child->unexpected_replies),
	        CACHE_COUNT("/child/callback_failures", child->callback_failures),
	        CACHE_COUNT("/child/peak_outstanding", child->peak_outstanding),
	        CACHE_COUNT("/child/plan_rtt_ms/count", child->plan_rtt.count),
	        CACHE_TIME("/child/plan_rtt_ms/p50", child->plan_rtt.p50_ms),
	        CACHE_TIME("/child/plan_rtt_ms/p90", child->plan_rtt.p90_ms),
	        CACHE_TIME("/child/plan_rtt_ms/max", child->plan_rtt.max_ms),
	        CACHE_COUNT("/child/compile_rtt_ms/count", child->compile_rtt.count),
	        CACHE_TIME("/child/compile_rtt_ms/p50", child->compile_rtt.p50_ms),
	        CACHE_TIME("/child/compile_rtt_ms/p90", child->compile_rtt.p90_ms),
	        CACHE_TIME("/child/compile_rtt_ms/max", child->compile_rtt.max_ms),
	        {"/child/lane_off_reason",
	         [](PrescanCacheReport& figures, uint64_t value) -> nlohmann::json {
		         figures.child->lane_off_reason = "reason_" + std::to_string(value);
		         return figures.child->lane_off_reason;
	         }},
	        {"/child/priority", [](PrescanCacheReport& figures, uint64_t value) -> nlohmann::json {
		         figures.child->priority = "priority_" + std::to_string(value);
		         return figures.child->priority;
	         }}};
}

// The drain, the builds, the GPU thread's holds and the records.
std::vector<PrescanCacheFieldCase> PrescanDrainBuildsRecordsCases() {
	return {CACHE_COUNT("/drain/adopted", drain.adopted),
	        CACHE_COUNT("/drain/duplicates", drain.duplicates),
	        CACHE_COUNT("/drain/plan_mismatch", drain.plan_mismatch),
	        CACHE_COUNT("/drain/failed_installs", drain.failed_installs),
	        CACHE_COUNT("/drain/dropped_stopped", drain.dropped_stopped),
	        CACHE_TIME("/drain/item_max_ms", drain.item_max_ms),
	        CACHE_COUNT("/drain/both_producers/calls", drain.both_calls),
	        CACHE_COUNT("/drain/both_producers/over_1ms", drain.both_over_1ms),
	        CACHE_COUNT("/drain/both_producers/over_4ms", drain.both_over_4ms),
	        CACHE_TIME("/drain/both_producers/max_ms", drain.both_max_ms),
	        CACHE_TIME("/drain/both_producers/total_ms", drain.both_total_ms),
	        CACHE_TIME("/drain/both_producers/lag_ms_max", drain.both_lag_ms_max),
	        CACHE_COUNT("/builds/builds", builds.builds),
	        CACHE_COUNT("/builds/worker_builds", builds.worker_builds),
	        CACHE_TIME("/builds/total_ms", builds.total_ms),
	        CACHE_COUNT("/holds/in_flight_at_miss_only", holds.in_flight_at_miss_only),
	        CACHE_COUNT("/builds/upper_bounds/in_scope", builds.in_scope),
	        CACHE_TIME("/builds/upper_bounds/in_scope_ms", builds.in_scope_ms),
	        CACHE_COUNT("/builds/upper_bounds/in_hold", builds.in_hold),
	        CACHE_TIME("/builds/upper_bounds/in_hold_ms", builds.in_hold_ms),
	        CACHE_COUNT("/records/stored_first_use", records.stored_first_use),
	        CACHE_COUNT("/records/retained_peak_bytes", records.retained_peak_bytes),
	        CACHE_COUNT("/records/dropped_over_cap", records.dropped_over_cap),
	        CACHE_COUNT("/records/unused_at_exit", records.unused_at_exit)};
}

// A list of named counts holding one name, set to the value, as the miss figures are written.
#define CACHE_NAMED(pointer, member, name)                                            \
	PrescanCacheFieldCase {                                                           \
		pointer, [](PrescanCacheReport& figures, uint64_t value) -> nlohmann::json {  \
			figures.member = {{name, value}};                                         \
			return value;                                                             \
		}                                                                             \
	}

// The GPU thread's own compiles by reason and by verdict, and the one upper bound, which must sit
// under upper_bounds.
std::vector<PrescanCacheFieldCase> PrescanMissesCases() {
	return {CACHE_COUNT("/misses/inline_compiles", misses.inline_compiles),
	        CACHE_NAMED("/misses/by_reason/never_posted", misses.by_reason, "never_posted"),
	        CACHE_NAMED("/misses/refused_by/built", misses.refused_by, "built"),
	        CACHE_COUNT("/misses/upper_bounds/built_undrained", misses.built_undrained)};
}

#undef CACHE_NAMED
#undef CACHE_COUNT
#undef CACHE_TIME
#undef CACHE_FLAG

// Every figure of the cache's report has its own place in the block. Each case sets one figure of
// figures that are empty but for a child, to a value no other case uses, and the block then
// differs from the empty one in that place only, so the baseline is the same block without the
// change. Two figures written under each other's keys, or one not written, would put the value in
// the wrong place or nowhere. The overlap figures must sit under upper_bounds, which labels them as
// bounds, and the GPU thread's holds under holds, apart from the builds.
void TestPrescanCacheFieldsRoundTrip() {
	const auto empty = [] {
		PrescanCacheReport figures;
		figures.child.emplace();
		return figures;
	};
	const auto block_of = [](const PrescanCacheReport& figures) {
		Report report;
		report.SetPrescanCacheFinal(figures);
		return PrescanBlock(report);
	};
	const auto base = block_of(empty());
	Check(base.is_object() && base["preparer"]["items"] == 0 &&
	          base["preparer"]["invalid"].empty() && base["child"]["priority"] == "" &&
	          base["builds"]["upper_bounds"]["in_hold_ms"] == 0.0,
	      "the empty cache block is not empty");
	const auto empty_at = [&base](const char* pointer, const nlohmann::json& value) {
		const nlohmann::json::json_pointer at(pointer);
		return base.contains(at) && base.at(at) == value;
	};
	Check(empty_at("/misses/inline_compiles", 0) &&
	          empty_at("/misses/by_reason", nlohmann::json::object()) &&
	          empty_at("/misses/refused_by", nlohmann::json::object()) &&
	          empty_at("/misses/upper_bounds/built_undrained", 0),
	      "the empty cache block's misses are not empty");
	auto cases = PrescanStateIntakeCases();
	for (auto& more: {PrescanPreparerOutcomeCases(), PrescanChildCases(),
	                  PrescanDrainBuildsRecordsCases(), PrescanMissesCases()}) {
		cases.insert(cases.end(), more.begin(), more.end());
	}
	// Distinct from 0 and from one another, so a figure that took another's value shows.
	constexpr uint64_t FIRST_VALUE = 2001;
	for (size_t index = 0; index < cases.size(); ++index) {
		auto       figures = empty();
		const auto value   = cases[index].set(figures, FIRST_VALUE + index);
		auto       expected = base;
		expected[nlohmann::json::json_pointer(cases[index].pointer)] = value;
		const auto actual = block_of(figures);
		if (actual != expected || actual == base) {
			std::fprintf(stderr,
			             "FirstRunReportTests: the prescan cache figure %s is not where it "
			             "belongs\n",
			             cases[index].pointer);
			std::abort();
		}
	}
}

// With a source, the writer asks it each time. A final figure replaces it at once and the source
// is never called again, so the cache may destroy what the source reads. The baseline is the same
// report before the final figure, which shows the source's figure and holds the source alone.
void TestPrescanCacheFinalReplacesSource() {
	Report           report;
	std::atomic<int> calls {0};
	report.SetPrescanCacheSource([&calls] {
		PrescanCacheReport figures;
		figures.preparer.items = 3;
		calls++;
		return figures;
	});
	Check(PrescanBlock(report)["preparer"]["items"] == 3 && calls == 1,
	      "the writer did not use the cache source");
	Check(report.HoldsPrescanCacheSource() && !report.HoldsPrescanCacheFinal(),
	      "a registered cache source was not held alone");
	PrescanCacheReport final_figures;
	final_figures.preparer.items = 9;
	report.SetPrescanCacheFinal(final_figures);
	Check(!report.HoldsPrescanCacheSource() && report.HoldsPrescanCacheFinal(),
	      "the final cache figures did not take the source's place");
	Check(PrescanBlock(report)["preparer"]["items"] == 9, "the final cache figures were not used");
	Check(calls == 1, "the cache source was called after the final figures replaced it");
}

// The cache is torn down right after SetPrescanCacheFinal returns. A writer that called the source
// after that would read a destroyed object, which this test models as a flag that the source
// checks. The baseline is the live source before the hand-over, which is called without a
// violation.
void TestPrescanCacheWriterNeverTouchesADestroyedSource() {
	Report            report;
	std::atomic<bool> alive {true};
	std::atomic<int>  violations {0};
	report.SetPrescanCacheSource([&] {
		if (!alive) violations++;
		return PrescanCacheReport {};
	});
	Check(PrescanBlock(report).contains("preparer") && violations == 0,
	      "the live cache source failed");
	report.SetPrescanCacheFinal({});
	alive            = false;
	const auto after = PrescanBlock(report);
	Check(violations == 0, "the writer called a cache source that was destroyed");
	Check(after.contains("preparer"),
	      "the final cache figures were not written after the source went");
}

// The writer is inside the source when the cache hands over the final figures. The hand-over must
// wait for the writer, because the cache's state goes right after it, and the writer must be out
// of the source by then. The source here is slow on purpose. A writer that copied the source and
// called it after unlocking would let the hand-over finish while it was still inside.
void TestPrescanCacheFinalWaitsForAWriterInTheSource() {
	Report            report;
	std::atomic<bool> inside {false};
	std::atomic<bool> release {false};
	std::atomic<bool> left_source {false};
	report.SetPrescanCacheSource([&] {
		inside = true;
		while (!release) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		left_source = true;
		return PrescanCacheReport {};
	});
	std::thread writer([&] { (void)PrescanBlock(report); });
	while (!inside) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	std::atomic<bool> handed_over {false};
	std::thread       owner([&] {
        report.SetPrescanCacheFinal({});
        handed_over = true;
    });
	// A hand-over that did not wait would be done by now.
	std::this_thread::sleep_for(std::chrono::milliseconds(200));
	const bool waited = !handed_over;
	release           = true;
	writer.join();
	owner.join();
	Check(waited, "the final cache figures were handed over while a writer was in the source");
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
  TestSubmitLeadUsesContainingSubmission();
  TestSubmitLeadSummary();
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
  TestWarmBuildScopeLanes();
  TestWorkerCompileLaneCounts();
  TestAudioUnderrunsCarryLanes();
  TestAudioUnderrunLanesFlagAndOverflow();
  TestHeldByCauseInJson();
  TestUnderrunsCarryThePrescanTag();
  TestGenerationSourceThenFinal();
  TestGenerationWriterNeverTouchesADestroyedSource();
  TestGenerationFinalWaitsForAWriterInTheSource();
  TestPrescanBlockAbsentWithoutSource();
  TestPrescanScanFieldsRoundTrip();
  TestPrescanScanFinalReplacesSource();
  TestPrescanWriterNeverTouchesADestroyedSource();
  TestPrescanFinalWaitsForAWriterInTheSource();
  TestGenerationDrainMarksPrescan();
  TestPrescanCacheBlockMerges();
  TestPrescanCacheBlockLeavesTheRest();
  TestPrescanCacheFieldsRoundTrip();
  TestPrescanCacheFinalReplacesSource();
  TestPrescanCacheWriterNeverTouchesADestroyedSource();
  TestPrescanCacheFinalWaitsForAWriterInTheSource();
  std::puts("FirstRunReportTests: all cases passed");
  return 0;
}
