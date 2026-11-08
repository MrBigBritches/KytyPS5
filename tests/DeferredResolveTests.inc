  // The wait, the report and the validation of a resolve of deferred draws. Included inside
  // VulkanHarness, after DrawRecordTests.inc.

  // The fake builds of a resolve, counting how many run at once.
  struct FakeBuildCounts {
    std::atomic<uint32_t> running{0};
    std::atomic<uint32_t> peak{0};
    std::atomic<uint32_t> arrived{0};
    std::atomic<bool> give_up{false};
  };

  // Keys for fake builds, numbered from one.
  static std::vector<GraphicsPipelineKey> FakeKeys(uint32_t count) {
    std::vector<GraphicsPipelineKey> keys;
    for (uint32_t number = 1; number <= count; ++number) {
      keys.push_back(PipelineCacheTestAccess::FakeKey(number));
    }
    return keys;
  }

  // Queues one fake build per key. Each notes how many are running, then waits until all of them are,
  // or until the test gives up, which only the serial case needs.
  static void PostLatchBuilds(PipelineCache &cache, const std::vector<GraphicsPipelineKey> &keys,
                              FakeBuildCounts &counts) {
    const auto builds = static_cast<uint32_t>(keys.size());
    for (const auto &key : keys) {
      PipelineCacheTestAccess::PostFakeBuild(cache, key, [&counts, builds] {
        const uint32_t now = ++counts.running;
        uint32_t seen = counts.peak.load();
        while (seen < now && !counts.peak.compare_exchange_weak(seen, now)) {
        }
        ++counts.arrived;
        // Bounded, so a resolve that serialises the builds fails the test instead of hanging it.
        const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (counts.arrived.load() < builds && !counts.give_up.load() &&
               std::chrono::steady_clock::now() < limit) {
          std::this_thread::yield();
        }
        --counts.running;
      });
    }
  }

  // The resolve waits for every build at once. Each fake build waits until all of them run together,
  // so the resolve can only finish if the workers ran them side by side and the resolve only waited.
  // A queue of one worker is the baseline. Its first build cannot finish before it is told to give
  // up, so the others cannot start, and no more than one runs at once.
  void CheckParallelResolveWait() {
    constexpr const char *name = "ResolveParallelWait";
    constexpr uint32_t builds = 4;
    for (const bool serial : {false, true}) {
      AsyncDrawRig rig(*this, true);
      auto &cache = rig.Pipelines();
      PipelineCacheTestAccess::ReplaceBuildQueue(cache, serial ? 1 : builds);
      const auto keys = FakeKeys(builds);
      FakeBuildCounts counts;
      PostLatchBuilds(cache, keys, counts);
      PipelineCache::DeferredAcquireStats stats;
      std::vector<PipelineCache::Pipeline> built;
      if (!serial) {
        built = cache.AcquireDeferredPipelines(keys, false, stats);
        Require(name, "parallel", counts.peak.load() == builds && counts.arrived.load() == builds,
                "the builds did not all run at once");
      } else {
        // The first build runs on the only worker before the resolve starts, so the resolve has to
        // wait for it and cannot reach the others.
        WaitUntilTrue(name, "first build", [&] { return counts.arrived.load() == 1; });
        std::thread resolver([&] { built = cache.AcquireDeferredPipelines(keys, false, stats); });
        Require(name, "serial blocked", counts.arrived.load() == 1 && counts.running.load() == 1,
                "a second build started while the only worker was busy");
        counts.give_up = true;
        resolver.join();
        Require(name, "serial", counts.peak.load() == 1,
                "the single worker ran more than one build at once");
      }
      Require(name, "result", built.size() == builds && stats.distinct_keys == builds &&
                  PipelineCacheTestAccess::PendingBuilds(cache) == 0,
              "the resolve did not take every build");
      for (const auto &key : keys) PipelineCacheTestAccess::ForgetFakePipeline(cache, key);
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Keys that repeat are counted once and waited for once, and a build is taken once.
  void CheckResolveDistinctKeys() {
    constexpr const char *name = "ResolveDistinctKeys";
    AsyncDrawRig rig(*this, true);
    auto &cache = rig.Pipelines();
    const auto keys = FakeKeys(3);
    for (const auto &key : keys) PipelineCacheTestAccess::PostFakeBuild(cache, key, [] {});
    const std::vector<GraphicsPipelineKey> holes{keys[0], keys[1], keys[0], keys[2], keys[0], keys[1]};
    PipelineCache::DeferredAcquireStats stats;
    const auto built = cache.AcquireDeferredPipelines(holes, false, stats);
    Require(name, "one pipeline per hole", built.size() == holes.size(),
            "the resolve did not return a pipeline for each hole");
    Require(name, "distinct", stats.distinct_keys == 3, "the distinct keys were not counted once");
    for (const auto &key : keys) PipelineCacheTestAccess::ForgetFakePipeline(cache, key);
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // The scope is held while the builds are waited for and never again. Fake builds first wait for a
  // scope to exist, so the state they see is the resolve's. One scope for several keys counts as one
  // hold of a thread that was parked in it. Scopes are counted, since a hold is only counted when a
  // thread was parked, and a scope per key would not show in it. ScopeCount is process wide, so the
  // test assumes no other thread opens a scope while it runs.
  void CheckResolveStallScope() {
    constexpr const char *name = "ResolveStallScope";
    constexpr uint32_t distinct = 3;
    namespace Stall = Common::GpuStall;
    AsyncDrawRig rig(*this, true);
    // After the rig, whose config load sets the stall hold from the emulator's settings.
    Stall::Configure(true, std::chrono::milliseconds(0));
    auto &cache = rig.Pipelines();
    const auto keys = FakeKeys(distinct);
    const uint64_t scopes_before = Stall::ScopeCount();
    const uint64_t holds_before = Stall::HoldCount();
    std::atomic<uint32_t> stalled_seen{0};
    std::atomic<bool> release{false};
    for (const auto &key : keys) {
      PipelineCacheTestAccess::PostFakeBuild(cache, key, [&] {
        while (Stall::ScopeCount() == scopes_before) std::this_thread::yield();
        if (Stall::IsStalled()) ++stalled_seen;
        while (!release.load()) std::this_thread::yield();
      });
    }
    const std::vector<GraphicsPipelineKey> holes{keys[0], keys[1], keys[2], keys[1]};
    PipelineCache::DeferredAcquireStats stats;
    std::thread resolver([&] { (void)cache.AcquireDeferredPipelines(holes, true, stats); });
    WaitUntilTrue(name, "scope", [&] { return Stall::IsStalled(); });
    std::thread audio([] { (void)Stall::WaitWhileStalled(); });
    WaitUntilTrue(name, "parked", [&] { return Stall::WaiterCount() == 1; });
    release = true;
    resolver.join();
    audio.join();
    Require(name, "builds ran inside the scope", stalled_seen.load() == distinct,
            "a build ran while no stall was in progress");
    Require(name, "one scope", Stall::ScopeCount() - scopes_before == 1,
            "the resolve opened more or fewer than one scope");
    Require(name, "one hold", Stall::HoldCount() - holds_before == 1,
            "the resolve was not counted as one hold of the parked thread");
    Require(name, "released", !Stall::IsStalled(), "the stall went on after the resolve");
    for (const auto &key : keys) PipelineCacheTestAccess::ForgetFakePipeline(cache, key);
    Stall::Configure(false);
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Splits `count` holes of the prepared draw, with the build of its key left to the resolve.
  void SplitHolesBehindBuild(AsyncDrawRig &rig, const HoleSource &source, uint32_t count,
                             PipelineCacheTestAccess::BuildHold &hold) {
    PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    PipelineCacheTestAccess::RetireToBackground(rig.Pipelines(), source.key);
    for (uint32_t index = 0; index < count; ++index) SplitHoleOf(rig, source);
  }

  // Recording the holes takes no part in the stall. The resolve of two holes of one key opens one
  // scope, and nothing the hole records is recorded while a stall is in progress. A recording that
  // ran with no scope would pass the second check alone, so the probe must have seen rows, and the
  // scope count shows that a scope was there at all.
  void CheckResolveRecordsOutsideStall() {
    constexpr const char *name = "ResolveRecordsOutsideStall";
    namespace Stall = Common::GpuStall;
    CommandStream::Scope hooks;
    AsyncDrawRig rig(*this, true);
    Stall::Configure(true, std::chrono::milliseconds(0));
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    CommandSchedulerTestAccess::HoldAudioFromThisThread(scheduler);
    rig.PaintSentinels();
    const auto source = PrepareHoleSource(rig);
    PipelineCacheTestAccess::BuildHold hold;
    SplitHolesBehindBuild(rig, source, 2, hold);
    uint32_t rows_in_stall = 0;
    uint32_t rows_seen = 0;
    CommandStream::Recorder::Reset(true);
    CommandStream::Recorder::probe = [&] {
      ++rows_seen;
      if (Stall::IsStalled()) ++rows_in_stall;
    };
    const uint64_t scopes_before = Stall::ScopeCount();
    scheduler.Flush();
    CommandStream::Recorder::probe = nullptr;
    CommandStream::Recorder::active = false;
    hold.release = true;
    Require(name, "recorded", rows_seen > 0 && scheduler.PendingHoleCount() == 0,
            "the holes were not recorded, so the probe saw nothing");
    Require(name, "one scope", Stall::ScopeCount() - scopes_before == 1,
            "the resolve did not open exactly one scope for its two holes");
    Require(name, "none while recording", rows_in_stall == 0,
            "a hole was recorded while the stall was in progress");
    Stall::Configure(false);
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A resolve adds one row to the report with its holes, distinct keys, flip and trigger. Two holes
  // of one key tell holes from keys, and the second resolve, under a trigger scope, tells the
  // trigger from the default.
  void CheckResolveReport() {
    constexpr const char *name = "ResolveReport";
    constexpr int64_t flip = 5;
    CommandStream::Scope hooks;
    AsyncDrawRig rig(*this, true);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    rig.PaintSentinels();
    const auto source = PrepareHoleSource(rig);
    Common::FirstRunReport::Global().OnFlip(flip);
    const uint64_t holds_before = Common::GpuStall::HoldCount();
    {
      PipelineCacheTestAccess::BuildHold hold;
      SplitHolesBehindBuild(rig, source, 2, hold);
      scheduler.Flush();
      hold.release = true;
    }
    {
      const CommandScheduler::TriggerScope fault(scheduler, CommandScheduler::ResolveTrigger::Fault);
      SplitHoleOf(rig, source);
      scheduler.Flush();
    }
    const uint64_t holds_after = Common::GpuStall::HoldCount();
    const auto rows = Common::FirstRunReport::Global().Summarize().deferred_resolves;
    Require(name, "two rows", rows.size() == 2, "each resolve did not add exactly one row");
    if (rows.size() != 2) return;
    Require(name, "holes and keys",
            rows[0].holes == 2 && rows[0].distinct_keys == 1 && rows[1].holes == 1 &&
                rows[1].distinct_keys == 1,
            "a row does not carry its holes and distinct keys");
    Require(name, "flip", rows[0].flip == flip && rows[1].flip == flip,
            "a row is not stamped with the flip it ended in");
    Require(name, "trigger", rows[0].trigger == "submit" && rows[1].trigger == "fault",
            "a row does not name what forced the resolve");
    Require(name, "times",
            rows[0].resolve_wait_ms > 0.0 && rows[0].record_ms > 0.0 && rows[0].pre_resolve_ms > 0.0 &&
                rows[0].save_ms >= 0.0 && rows[1].record_ms > 0.0,
            "a time in a row was not measured");
    Require(name, "holds", rows[0].hold_count_change + rows[1].hold_count_change ==
                holds_after - holds_before,
            "the rows do not add up to the holds counted meanwhile");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // The names the report uses for each trigger.
  void CheckResolveTriggerNames() {
    constexpr const char *name = "ResolveTriggerNames";
    using Trigger = CommandScheduler::ResolveTrigger;
    const std::vector<std::pair<Trigger, std::string_view>> names{
        {Trigger::Submit, "submit"},     {Trigger::Fault, "fault"},
        {Trigger::StreamWrap, "stream_wrap"}, {Trigger::Cap, "cap"},
        {Trigger::Shutdown, "shutdown"}, {Trigger::Readback, "readback"},
        {Trigger::Label, "label"},       {Trigger::Flip, "flip"},
        {Trigger::Unmap, "unmap"},       {Trigger::SliceEnd, "slice_end"}};
    for (const auto &[trigger, expected] : names) {
      Require(name, expected.data(), CommandScheduler::ResolveTriggerName(trigger) == expected,
              "a trigger has another name than the report documents");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A positive control for the check below. Two writes to one buffer with nothing between them are a
  // hazard, which the layer must report, or a result of zero hazards would prove nothing.
  void CheckSyncValidationSeesHazard() {
    constexpr const char *name = "SyncValidationControl";
    namespace Validation = Libs::Graphics::VulkanValidation;
    auto buffer = CreateHostBuffer(name, 256, vk::BufferUsageFlagBits::eTransferDst, {});
    Validation::ResetMessages();
    {
      SideBuffer side(RuntimeContext());
      side.buffer.fillBuffer(buffer.buffer, 0, 256, 1);
      side.buffer.fillBuffer(buffer.buffer, 0, 256, 2);
    }
    const auto messages = Validation::Messages();
    const auto hazards = std::ranges::count_if(messages, [](const auto &message) {
      return message.find(" hazard ") != std::string::npos;
    });
    DestroyBuffer(&buffer);
    Require(name, "hazard seen", hazards > 0,
            "the layer reported no hazard for two unsynchronized writes, so zero proves nothing");
    Validation::ResetMessages();
  }

  // Runs the hole scenarios under the validation layer with synchronization validation, and reads
  // the messages it reported. It skips, and says why, when the device was not made with the layer.
  // A hazard between recorded commands fails the test. Other messages are printed and not counted,
  // since this device lacks features the emulator's own device has, such as depth clamp, and the
  // blocking draw the scenarios compare with reports the same ones.
  void CheckHoleSyncValidation() {
    constexpr const char *name = "HoleSyncValidation";
    if (!SyncValidationActive()) {
      std::printf("[gpu]     %-32s skipped: %s\n", name, SyncValidationReason().c_str());
      return;
    }
    CheckSyncValidationSeesHazard();
    namespace Validation = Libs::Graphics::VulkanValidation;
    const auto run = [&](const char *scenario, auto &&check) {
      Validation::ResetMessages();
      check();
      const auto messages = Validation::Messages();
      const auto hazards = std::ranges::count_if(messages, [](const auto &message) {
        return message.find(" hazard ") != std::string::npos;
      });
      std::printf("[gpu]     %-32s %s: %llu messages, %lld hazards\n", name, scenario,
                  static_cast<unsigned long long>(Validation::MessageCount()),
                  static_cast<long long>(hazards));
      for (const auto &message : messages) std::fprintf(stderr, "%s: %s\n", scenario, message.c_str());
      Require(name, scenario, hazards == 0, "the validation layer reported a synchronization hazard");
    };
    run("HoleRecordedStream", [&] { CheckHoleRecordedStream(); });
    run("HoleBookkeeping", [&] { CheckHoleBookkeeping(); });
    run("HoleEarlyAdoption", [&] { CheckHoleEarlyAdoption(); });
    std::printf("[gpu]     %-32s ok\n", name);
  }

  void CheckDeferredResolve() {
    CheckResolveTriggerNames();
    CheckParallelResolveWait();
    CheckResolveDistinctKeys();
    CheckResolveStallScope();
    CheckResolveRecordsOutsideStall();
    CheckResolveReport();
    CheckHoleSyncValidation();
  }
