#include "graphics/presentation/system/binding/eventQueue.h"
#include "graphics/presentation/system/binding/published.h"
#include "graphics/presentation/system/binding/uiActivity.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

namespace {

using Libs::Graphics::SystemUi::Binding;
using Libs::Graphics::SystemUi::EventQueue;
using Libs::Graphics::SystemUi::Published;
using Libs::Graphics::SystemUi::UiActivity;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiBindingTests: failed: %s\n", message);
    std::abort();
  }
}

// A value that counts how often it is copied, so a test can tell a refresh that copies from one
// that doesn't.
struct CopyCounter {
  static inline int copies = 0;
  int value = 0;

  CopyCounter() = default;
  explicit CopyCounter(int initial) : value(initial) {}
  CopyCounter(const CopyCounter &other) : value(other.value) { copies++; }
  CopyCounter(CopyCounter &&) = default;
  CopyCounter &operator=(const CopyCounter &other) {
    value = other.value;
    copies++;
    return *this;
  }
  CopyCounter &operator=(CopyCounter &&) = default;
};

// A refresh picks up each publish once, and skips when nothing changed. Several publishes
// between refreshes give the latest value.
void TestBindingRefresh() {
  Published<int> source;
  Binding<int> binding(source);
  Check(!binding.Refresh() && binding.Value() == 0, "nothing to refresh before the first publish");
  source.Publish(5);
  Check(binding.Refresh() && binding.Value() == 5 && binding.Revision() == 1, "a publish is picked up");
  Check(!binding.Refresh(), "a second refresh without a publish does nothing");
  source.Publish(6);
  source.Publish(7);
  Check(binding.Refresh() && binding.Value() == 7 && binding.Revision() == 3, "the latest of several publishes");
}

// The value is copied only when it changed. The baseline is the copy count after the first
// refresh, and a refresh with nothing new must leave it alone.
void TestBindingCopiesOnlyOnChange() {
  Published<CopyCounter> source;
  Binding<CopyCounter> binding(source);
  source.Publish(CopyCounter(1));
  binding.Refresh();
  const int after_first = CopyCounter::copies;
  for (int i = 0; i < 10; i++) {
    binding.Refresh();
  }
  Check(CopyCounter::copies == after_first, "refreshing an unchanged value copies nothing");
  source.Publish(CopyCounter(2));
  binding.Refresh();
  Check(CopyCounter::copies > after_first && binding.Value().value == 2, "a changed value is copied");
}

// Publishing and pushing wake the activity, so the UI draws even while the game presents no
// frames. The baseline is the same calls without an activity, which must not crash.
void TestWake() {
  UiActivity activity;
  Published<int> published(&activity);
  EventQueue<int> queue(4, &activity);
  const auto before = activity.Revision();
  published.Publish(1);
  Check(activity.Revision() == before + 1, "a publish wakes the UI");
  queue.Push(1);
  Check(activity.Revision() == before + 2, "a push wakes the UI");
  Published<int> quiet;
  quiet.Publish(1);
  EventQueue<int> quiet_queue(4);
  quiet_queue.Push(1);
  Check(activity.Revision() == before + 2, "sources without an activity wake nothing");
}

// A block of words that are all equal when written whole. A reader that sees them differ has
// read half of one publish and half of another.
struct Block {
  std::array<uint64_t, 32> words{};
};

// True once `deadline` has passed, so a broken race fails with a message instead of hanging.
bool Expired(std::chrono::steady_clock::time_point deadline) { return std::chrono::steady_clock::now() > deadline; }

// A publisher thread and a reader thread race. The publisher keeps going until the reader has
// refreshed many times while it was still publishing, so a pass shows the race really happened.
// Every block the reader sees must be whole, and its words must equal the revision it was read
// with, because publish i writes value i at revision i. Revisions never go backwards.
void TestPublishedAcrossThreads() {
  constexpr int OVERLAPPING_REFRESHES = 1000;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  Published<Block> source;
  std::atomic<bool> stop = false;
  uint64_t published = 0;
  std::thread publisher([&] {
    while (!stop) {
      Block block;
      block.words.fill(++published);
      source.Publish(block);
    }
  });
  Binding<Block> binding(source);
  int overlapping = 0;
  uint64_t last = 0;
  bool torn = false, mismatched = false, backwards = false;
  while (overlapping < OVERLAPPING_REFRESHES && !Expired(deadline)) {
    if (!binding.Refresh()) {
      continue;
    }
    overlapping++;
    const auto &words = binding.Value().words;
    torn |= std::any_of(words.begin(), words.end(), [&](uint64_t word) { return word != words[0]; });
    mismatched |= words[0] != binding.Revision();
    backwards |= binding.Revision() < last;
    last = binding.Revision();
  }
  stop = true;
  publisher.join();
  Check(overlapping == OVERLAPPING_REFRESHES, "the reader raced the publisher before the deadline");
  Check(!torn, "a reader never sees half of a publish");
  Check(!mismatched, "a value is read with the revision it was published at");
  Check(!backwards, "revisions never go backwards");
  binding.Refresh();
  Check(binding.Revision() == published && binding.Value().words[0] == published, "the last publish is picked up");
}

// Events come out oldest first, a drain empties the queue, and an empty queue drains to nothing.
void TestQueueOrder() {
  EventQueue<int> queue(8);
  Check(!queue.HasPending() && queue.Drain().empty(), "an empty queue drains to nothing");
  for (int i = 1; i <= 3; i++) {
    queue.Push(i);
  }
  Check(queue.HasPending(), "pushed events are pending");
  Check(queue.Drain() == std::vector<int>{1, 2, 3}, "events come out oldest first");
  Check(!queue.HasPending() && queue.Drain().empty(), "a drain empties the queue");
}

// A full queue drops its oldest event and counts it, keeping the newest. A capacity of zero
// holds one event rather than none.
void TestQueueOverflow() {
  EventQueue<int> queue(3);
  for (int i = 1; i <= 5; i++) {
    queue.Push(i);
  }
  Check(queue.Drain() == std::vector<int>{3, 4, 5}, "the newest events are kept");
  Check(queue.Dropped() == 2, "dropped events are counted");
  EventQueue<int> tiny(0);
  tiny.Push(1);
  tiny.Push(2);
  Check(tiny.Drain() == std::vector<int>{2}, "a zero capacity holds one event");
}

// Producers on several threads push while the present thread drains. Nothing is lost or
// duplicated, and each producer's events stay in its own order.
void TestQueueAcrossThreads() {
  constexpr int PRODUCERS = 4, EVENTS = 20000;
  EventQueue<int> queue(PRODUCERS * EVENTS);
  std::vector<std::thread> producers;
  for (int producer = 0; producer < PRODUCERS; producer++) {
    producers.emplace_back([&queue, producer] {
      for (int i = 0; i < EVENTS; i++) {
        queue.Push(producer * EVENTS + i);
      }
    });
  }
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  std::vector<int> next(PRODUCERS, 0);
  int received = 0;
  bool in_order = true;
  while (received < PRODUCERS * EVENTS && !Expired(deadline)) {
    for (const int event : queue.Drain()) {
      const int producer = event / EVENTS;
      in_order &= event % EVENTS == next[producer]++;
      received++;
    }
  }
  for (auto &producer : producers) {
    producer.join();
  }
  Check(received == PRODUCERS * EVENTS, "every event arrives before the deadline");
  Check(in_order && queue.Dropped() == 0, "every event arrives once, in its producer's order");
  Check(queue.Drain().empty(), "nothing extra arrives");
}

// A frame that requests another leaves the UI active, and one that doesn't makes it idle. The
// state changes only at the end of a frame, so other threads never see a half-drawn frame's
// answer.
void TestActivityFrames() {
  UiActivity activity;
  Check(!activity.Active(), "idle before any frame");
  activity.BeginFrame();
  activity.RequestFrame();
  activity.EndFrame();
  Check(activity.Active(), "a frame that requests another leaves the UI active");
  activity.BeginFrame();
  Check(activity.Active(), "the answer only changes at the end of a frame");
  activity.EndFrame();
  Check(!activity.Active(), "a frame without a request leaves the UI idle");
}

} // namespace

int main() {
  TestBindingRefresh();
  TestBindingCopiesOnlyOnChange();
  TestWake();
  TestPublishedAcrossThreads();
  TestQueueOrder();
  TestQueueOverflow();
  TestQueueAcrossThreads();
  TestActivityFrames();
  std::puts("System UI binding tests passed");
  return 0;
}
