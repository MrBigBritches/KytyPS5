#include "graphics/presentation/system/core/frameStats.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

namespace {

using Libs::Graphics::SystemUi::FrameStats;

constexpr double TOLERANCE = 1e-9;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiStatsTests: failed: %s\n", message);
    std::abort();
  }
}

// True when two values agree to within rounding noise.
bool Near(double actual, double expected) { return std::fabs(actual - expected) <= TOLERANCE; }

// Adds each of `values` to `stats`.
void AddAll(FrameStats &stats, std::initializer_list<double> values) {
  for (const double value : values) {
    stats.Add(value);
  }
}

// With nothing added, every statistic is zero.
void TestEmpty() {
  const FrameStats stats(8);
  Check(stats.Count() == 0 && stats.Mean() == 0.0 && stats.Max() == 0.0 && stats.P95() == 0.0,
        "empty statistics are all zero");
}

// Samples 1 to 19 and one outlier of 100 give a different mean (14.5), median (10.5), 95th
// percentile (19, the 19th of 20) and maximum (100), so a percentile that returned the maximum,
// the median or the mean would fail. Adding the outlier first proves the percentile sorts.
void TestSkewed() {
  FrameStats stats(32);
  stats.Add(100);
  for (int value = 19; value >= 1; value--) {
    stats.Add(value);
  }
  Check(stats.Count() == 20, "every sample is counted");
  Check(Near(stats.Mean(), 14.5), "the mean includes the outlier");
  Check(Near(stats.Max(), 100), "the maximum is the outlier");
  Check(Near(stats.P95(), 19), "the 95th percentile is the 19th of 20 by nearest rank");
}

// Past its capacity, samples still count towards the mean and maximum, but the percentile uses
// only the samples kept.
void TestCapacity() {
  FrameStats stats(4);
  AddAll(stats, {1, 2, 3, 4, 1000});
  Check(stats.Count() == 5, "a sample past the capacity is counted");
  Check(Near(stats.Mean(), 202) && Near(stats.Max(), 1000), "it counts towards the mean and maximum");
  Check(Near(stats.P95(), 4), "but the percentile uses only the four kept");
}

// The maximum of negative samples is the largest of them, not zero.
void TestNegative() {
  FrameStats stats(4);
  AddAll(stats, {-5, -3, -9});
  Check(Near(stats.Max(), -3), "the maximum of negative samples is the largest");
}

// After a reset, new samples are measured on their own.
void TestReset() {
  FrameStats stats(4);
  AddAll(stats, {50, 60, 70, 80});
  stats.Reset();
  Check(stats.Count() == 0 && stats.Mean() == 0.0 && stats.P95() == 0.0, "a reset forgets every sample");
  stats.Add(7);
  Check(Near(stats.Mean(), 7) && Near(stats.Max(), 7) && Near(stats.P95(), 7),
        "samples after a reset are measured on their own");
}

} // namespace

int main() {
  TestEmpty();
  TestSkewed();
  TestCapacity();
  TestNegative();
  TestReset();
  std::printf("System UI stats tests passed\n");
  return 0;
}
