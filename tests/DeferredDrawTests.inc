  // Draws that become holes for real under --deferred-pipeline-draws. Included inside
  // VulkanHarness, after DeferredResolveTests.inc.

  static constexpr uint32_t DeferredHoleScriptDraws = 12;

  enum class DeferredBlend { Overwrite, Alpha, Add };

  // One step of the order script. A draw sets a pixel colour, a blend and a depth state, then draws
  // the rig's full-screen triangle. A snapshot copies the colour target to a buffer without
  // submitting. A warm draw is made, and its pipeline built, before the script starts.
  struct DeferredStep {
    std::array<float, 4> color;
    DeferredBlend blend;
    vk::CompareOp depth_op;
    float depth;
    bool depth_write;
    bool warm = false;
    bool snapshot = false;
    // A compute dispatch that loads a texel of the colour target as a texture and stores it to
    // guest memory, in the stream between the draws.
    bool compute_read = false;
  };

  // The twelve draws of the order script, with two snapshots between them. Blends and depth tests make
  // the result depend on the order the draws run in, and some draws share a pipeline.
  static std::vector<DeferredStep> DeferredOrderScript() {
    using Op = vk::CompareOp;
    constexpr auto Over = DeferredBlend::Overwrite;
    constexpr auto Alpha = DeferredBlend::Alpha;
    constexpr auto Add = DeferredBlend::Add;
    return {
        {{0.1f, 0.2f, 0.3f, 1.0f}, Over, Op::eAlways, 0.50f, true, true},
        {{0.9f, 0.8f, 0.7f, 0.5f}, Alpha, Op::eAlways, 0.40f, true},
        {{0.2f, 0.6f, 0.2f, 0.25f}, Alpha, Op::eLess, 0.45f, true},
        {.snapshot = true},
        {.compute_read = true},
        {{0.3f, 0.3f, 0.9f, 1.0f}, Over, Op::eLess, 0.30f, true},
        {{0.05f, 0.05f, 0.05f, 0.0f}, Add, Op::eAlways, 0.30f, false},
        {{0.7f, 0.1f, 0.1f, 1.0f}, Over, Op::eGreater, 0.35f, true, true},
        {{0.4f, 0.4f, 0.9f, 0.75f}, Alpha, Op::eAlways, 0.25f, true},
        {.snapshot = true},
        {.compute_read = true},
        {{0.6f, 0.2f, 0.8f, 0.4f}, Alpha, Op::eLess, 0.26f, true},
        {{0.1f, 0.0f, 0.0f, 0.0f}, Add, Op::eAlways, 0.20f, true},
        {{0.0f, 0.5f, 0.0f, 1.0f}, Over, Op::eLess, 0.10f, true},
        {{0.9f, 0.8f, 0.7f, 0.5f}, Alpha, Op::eAlways, 0.15f, true},
        {{0.1f, 0.2f, 0.3f, 1.0f}, Over, Op::eAlways, 0.12f, true},
    };
  }

  // Makes the rig's pixel shader one that writes this colour. Another colour is another shader and
  // so another pipeline.
  static void UsePixelColor(AsyncDrawRig &rig, const std::array<float, 4> &color) {
    static std::map<std::array<u32, 4>, std::vector<u32>> shaders;
    std::array<u32, 4> bits{};
    for (size_t channel = 0; channel < bits.size(); ++channel) bits[channel] = std::bit_cast<u32>(color[channel]);
    auto &code = shaders[bits];
    if (code.empty()) {
      for (u32 channel = 0; channel < 4; ++channel) AppendVMovLiteral(&code, 1 + channel, bits[channel]);
      code.push_back(EncodeExp0(0x00, 0xf));
      code.push_back(EncodeExp1(1, 2, 3, 4));
      AppendEnd(&code);
    }
    const auto address = reinterpret_cast<uint64_t>(code.data());
    ShaderMapUserData(address, {.type = Prospero::ShaderBinaryType::kPs,
                                .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
    rig.shaders.SetPsShaderBase(address);
  }

  static void SetDeferredBlend(AsyncDrawRig &rig, DeferredBlend mode) {
    using Factor = Prospero::BlendFactor;
    auto blend = rig.registers.GetBlendControl(0);
    blend.enable = mode != DeferredBlend::Overwrite;
    const auto source = mode == DeferredBlend::Alpha ? Factor::kSrcAlpha : Factor::kOne;
    const auto destination = mode == DeferredBlend::Alpha ? Factor::kOneMinusSrcAlpha
                             : mode == DeferredBlend::Add ? Factor::kOne : Factor::kZero;
    blend.color_srcblend = blend.alpha_srcblend = static_cast<uint8_t>(source);
    blend.color_destblend = blend.alpha_destblend = static_cast<uint8_t>(destination);
    blend.color_comb_fcn = blend.alpha_comb_fcn = 0;
    rig.registers.SetBlendControl(0, blend);
  }

  // The depth test, and the depth the draw lands on. The vertex shader puts every vertex at zero, so
  // the viewport's offset is the depth of the whole triangle.
  static void SetDeferredDepth(AsyncDrawRig &rig, vk::CompareOp op, float depth, bool write) {
    rig.registers.SetDepthControl(
        {.z_enable = true, .z_write_enable = write, .zfunc = static_cast<uint8_t>(op)});
    constexpr float half = AsyncExtent / 2;
    rig.registers.SetViewportScaleOffset(0, half, half, half, half, 1, depth);
  }

  static void ApplyDeferredStep(AsyncDrawRig &rig, const DeferredStep &step) {
    UsePixelColor(rig, step.color);
    SetDeferredBlend(rig, step.blend);
    SetDeferredDepth(rig, step.depth_op, step.depth, step.depth_write);
  }

  // Gives the rig's context the guest GPU that a compute dispatch and a readback ask for, and takes
  // it away again when it goes out of scope. The draws still run on the calling thread.
  struct GpuScope {
    AsyncDrawRig &rig;
    explicit GpuScope(AsyncDrawRig &owner) : rig(owner) {
      rig.context->InitializeGpu(nullptr);
      LibKernel::Memory::InstallGpuResources(&*rig.context);
    }
    ~GpuScope() {
      LibKernel::Memory::InstallGpuResources(nullptr);
      rig.context->ShutdownGpu();
    }
    GpuScope(const GpuScope &) = delete;
    GpuScope &operator=(const GpuScope &) = delete;
  };

  static constexpr uint64_t DeferredComputeOutputOffset = 0x200000;
  static constexpr uint32_t DeferredProbeTexel = AsyncExtent / 2;

  // Dispatches a compute shader that loads one texel of the colour target as a texture and stores
  // its four floats to guest memory at the slot's address. The load reads what the draws before it
  // wrote, so the dispatch sits in the stream between holes.
  void RecordComputeRead(AsyncDrawRig &rig, uint32_t slot) {
    static const auto code = [] {
      std::vector<u32> shader;
      AppendVMovU32(&shader, 21, DeferredProbeTexel);
      AppendVMovU32(&shader, 22, DeferredProbeTexel);
      AppendVMovU32(&shader, 20, 0);
      shader.insert(shader.end(), {EncodeMimg0(0x00, 0xf), EncodeMimg1(1, 21), 0xbf8c3f70u,
                                   EncodeMubuf0(0x1e), EncodeMubuf1(1, 2, 20)});
      AppendEnd(&shader);
      return shader;
    }();
    const auto address = reinterpret_cast<uint64_t>(code.data());
    ShaderMapUserData(address, {.type = Prospero::ShaderBinaryType::kCs,
                                .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
    const uint64_t color_address = AsyncBase + AsyncColorOffset;
    const ShaderTextureResource sharp{{static_cast<u32>(color_address >> 8u),
        (static_cast<u32>(Prospero::BufferFormat::k32_32_32_32Float) << 20u) |
            (((AsyncExtent - 1u) & 3u) << 30u),
        ((AsyncExtent - 1u) >> 2u) | ((AsyncExtent - 1u) << 14u),
        DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::TileMode::kLinear) << 20u) |
            (static_cast<u32>(Prospero::ImageType::kColor2D) << 28u),
        0, 0x00700000u, 0, 0}};
    ShaderBufferResource output{};
    output.UpdateAddress48(AsyncBase + DeferredComputeOutputOffset + slot * 0x100);
    output.fields[2] = 4 * sizeof(u32);
    output.fields[3] = DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::BufferFormat::k32UInt) << 12u);
    rig.shaders.SetCsShader({.data_addr = address, .num_thread_x = 1, .num_thread_y = 1,
                             .num_thread_z = 1, .wave_size = 64, .user_sgpr = 12});
    for (u32 field = 0; field < 8; ++field) rig.shaders.SetCsUserSgpr(field, sharp.fields[field]);
    for (u32 field = 0; field < 4; ++field) rig.shaders.SetCsUserSgpr(8 + field, output.fields[field]);
    rig.context->GetRenderExecutor().DispatchDirect(rig.Scheduler().Current(), 1, 1, 1, 0x41u, false);
  }

  // The four floats a compute read stored, as words. The read waits for the GPU.
  std::vector<u32> ReadComputeOutput(AsyncDrawRig &rig, uint32_t slot) {
    const uint64_t address = AsyncBase + DeferredComputeOutputOffset + slot * 0x100;
    rig.context->GetBufferCache().ReadMemory(address, 4 * sizeof(u32));
    std::vector<u32> words(4);
    Require("DeferredCompute", "readback",
            LibKernel::Memory::TryReadBacking(address, words.data(), words.size() * sizeof(u32)),
            "the compute output could not be read back");
    return words;
  }

  // Records a copy of the colour target to a new buffer, in the scheduler's current commands and
  // without submitting, so it reads what the commands before it wrote. The caller reads and
  // destroys the buffer after a submit.
  Buffer RecordColorSnapshot(AsyncDrawRig &rig) {
    auto &scheduler = rig.Scheduler();
    auto &image = rig.Textures().GetImage(rig.color.image_id);
    const auto bytes = image.info.bytes_per_block * AsyncExtent * AsyncExtent;
    auto probe = CreateHostBuffer("DeferredSnapshot", bytes, vk::BufferUsageFlagBits::eTransferDst, {});
    scheduler.Current().EndRendering();
    image.Transit(vk::ImageLayout::eTransferSrcOptimal, vk::AccessFlagBits2::eTransferRead, {},
                  scheduler.Current().Handle());
    vk::BufferImageCopy copy{};
    copy.imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1};
    copy.imageExtent = vk::Extent3D{AsyncExtent, AsyncExtent, 1};
    scheduler.Current().Handle().copyImageToBuffer(image.backing.image,
                                                   vk::ImageLayout::eTransferSrcOptimal, probe.buffer, 1, &copy);
    vk::BufferMemoryBarrier barrier{};
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = probe.buffer;
    barrier.size = probe.size;
    scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                 vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
                                                 &barrier, 0, nullptr);
    return probe;
  }

  struct DeferredRun {
    std::vector<u32> color;
    std::vector<u32> depth;
    std::vector<std::vector<u32>> snapshots;
    std::vector<std::vector<u32>> compute_reads;
    size_t holes = 0;
    size_t pipelines_made = 0;
  };

  using DeferredIdentity = std::tuple<std::array<float, 4>, DeferredBlend, vk::CompareOp, bool>;

  // The pipelines the script's cold draws want, told apart by what makes them differ: the pixel
  // colour, the blend and the depth state. A warm draw has built its own before the script starts.
  static std::set<DeferredIdentity> ColdDeferredIdentities(const std::vector<DeferredStep> &steps) {
    std::set<DeferredIdentity> warm;
    std::set<DeferredIdentity> cold;
    for (const auto &step : steps) {
      if (step.warm) warm.insert({step.color, step.blend, step.depth_op, step.depth_write});
    }
    for (const auto &step : steps) {
      const DeferredIdentity identity{step.color, step.blend, step.depth_op, step.depth_write};
      if (!step.snapshot && !step.compute_read && !step.warm && !warm.contains(identity)) cold.insert(identity);
    }
    return cold;
  }

  // The draws that a deferred run makes holes of. A draw is a hole when no warm draw built its
  // pipeline, and a hole stays pending until the resolve, so a repeat of its pipeline is a hole too.
  static size_t ExpectedDeferredHoles(const std::vector<DeferredStep> &steps) {
    const auto cold = ColdDeferredIdentities(steps);
    return static_cast<size_t>(std::ranges::count_if(steps, [&](const DeferredStep &step) {
      return !step.snapshot && !step.compute_read && !step.warm &&
             cold.contains({step.color, step.blend, step.depth_op, step.depth_write});
    }));
  }

  // Runs the script on a fresh rig. Warm draws go first, with the build workers free, so their
  // pipelines exist. Then both targets are painted and the rest of the script runs. A deferred run
  // holds the workers while it draws, so no build finishes before its draw, and releases them when
  // the script ends, so the resolve is what waits for the builds.
  DeferredRun RunDeferredScript(const char *name, bool deferred, const std::vector<DeferredStep> &steps) {
    DeferredRun run;
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, false, false, false, false, deferred);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    // After the rig, so it goes first at the end of the run.
    std::optional<GpuScope> gpu;
    if (std::ranges::any_of(steps, [](const auto &step) { return step.compute_read; })) gpu.emplace(rig);
    for (const auto &step : steps) {
      if (!step.warm) continue;
      ApplyDeferredStep(rig, step);
      Require(name, "warm draw", rig.Draw(), "a warm draw was refused");
    }
    scheduler.Flush();
    rig.PaintSentinels();
    if (deferred) PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    Common::FirstRunReport::Global().Reset();
    const auto before = PipelineCacheTestAccess::Pipelines(rig.Pipelines())[0];
    std::vector<Buffer> probes;
    uint32_t compute_reads = 0;
    for (const auto &step : steps) {
      if (step.snapshot) {
        probes.push_back(RecordColorSnapshot(rig));
      } else if (step.compute_read) {
        RecordComputeRead(rig, compute_reads++);
      } else if (!step.warm) {
        ApplyDeferredStep(rig, step);
        Require(name, "draw", rig.Draw(), "a draw reported failure");
      }
    }
    run.holes = scheduler.PendingHoleCount();
    run.pipelines_made = PipelineCacheTestAccess::Pipelines(rig.Pipelines())[0] - before;
    hold.release = true;
    scheduler.Finish();
    Require(name, "no hole left", scheduler.PendingHoleCount() == 0, "a finish left a hole pending");
    for (auto &probe : probes) {
      run.snapshots.push_back(ReadBuffer(name, probe, probe.size / sizeof(u32)));
      DestroyBuffer(&probe);
    }
    for (uint32_t slot = 0; slot < compute_reads; ++slot) {
      run.compute_reads.push_back(ReadComputeOutput(rig, slot));
    }
    run.color = rig.ReadColor();
    run.depth = rig.ReadDepth();
    return run;
  }

  // Draws made while their pipelines build give the same pictures as draws that wait for them. The
  // blocking run is the baseline. The script's blends and depth tests make its result depend on
  // the order of the draws, a snapshot reads the target between draws, and some draws share a
  // pipeline, so a hole in the wrong place, a missing hole or a missing barrier changes a byte.
  void CheckDeferredOrderEquivalence() {
    constexpr const char *name = "DeferredOrder";
    const auto steps = DeferredOrderScript();
    const auto baseline = RunDeferredScript(name, false, steps);
    const auto expected_holes = ExpectedDeferredHoles(steps);
    Require(name, "script size",
            std::ranges::count_if(steps, [](const auto &step) { return !step.snapshot && !step.compute_read; }) ==
                DeferredHoleScriptDraws,
            "the script does not have the draws it claims");
    // The baseline builds a pipeline for each distinct cold state, which checks the model of
    // which draws share a pipeline that the hole count below relies on.
    const auto cold = ColdDeferredIdentities(steps);
    Require(name, "baseline pipelines", baseline.pipelines_made == cold.size() && baseline.holes == 0,
            "the blocking run did not build one pipeline per distinct unbuilt state");
    const auto deferred = RunDeferredScript(name, true, steps);
    Require(name, "hole count", deferred.holes == expected_holes,
            "the deferred run did not turn every draw with a missing pipeline into a hole");
    const auto summary = Common::FirstRunReport::Global().Summarize();
    uint32_t resolved = 0;
    for (const auto &row : summary.deferred_resolves) resolved += row.holes;
    Require(name, "resolved", resolved == expected_holes && summary.deferred_ineligible.empty(),
            "the holes were not all resolved, or a draw was left to block");
    Require(name, "snapshots", deferred.snapshots == baseline.snapshots,
            "a read between holes saw another picture than in the blocking run");
    Require(name, "compute reads", deferred.compute_reads == baseline.compute_reads,
            "a compute read between holes saw another texel than in the blocking run");
    Require(name, "color", deferred.color == baseline.color,
            "the colour target differs from the blocking run");
    Require(name, "depth", deferred.depth == baseline.depth,
            "the depth target differs from the blocking run");
    // The baseline must have drawn something an earlier or later draw would change.
    // The compute read loads the very texel the snapshot before it copied, so it read the draws.
    Require(name, "compute reads the draws",
            baseline.compute_reads.size() == 2 && baseline.snapshots.size() == 2 &&
                std::vector<u32>(baseline.snapshots[0].begin() +
                                     (DeferredProbeTexel * AsyncExtent + DeferredProbeTexel) * 4,
                                 baseline.snapshots[0].begin() +
                                     (DeferredProbeTexel * AsyncExtent + DeferredProbeTexel) * 4 + 4) ==
                    baseline.compute_reads[0] &&
                baseline.compute_reads[0] != baseline.compute_reads[1],
            "the compute read did not load the texel the snapshot copied");
    Require(name, "script is order dependent",
            !AllEqual(baseline.color, AsyncSentinelColor) && baseline.snapshots.size() == 2 &&
                baseline.snapshots[0] != baseline.snapshots[1] && baseline.snapshots[1] != baseline.color,
            "the script's pictures do not change between its snapshots");
    std::printf("[gpu]     %-32s ok (%zu holes)\n", name, expected_holes);
  }

  // One scene for the clear test. The first draw consumes a depth clear and writes no depth, so
  // the clear value is what it leaves. The second, when asked for, writes a depth of its own.
  struct DeferredClearRun {
    std::vector<u32> depth;
    size_t holes = 0;
  };

  DeferredClearRun RunDeferredClearScene(const char *name, bool deferred, bool second_draw) {
    DeferredClearRun run;
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, false, false, false, false, deferred);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    rig.PaintSentinels();
    if (deferred) PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    SetDeferredDepth(rig, vk::CompareOp::eAlways, 0.0f, false);
    rig.SetDepthClear(true);
    Require(name, "clearing draw", rig.Draw(), "the draw that consumes the clear failed");
    rig.SetDepthClear(false);
    if (second_draw) {
      rig.SetSecondPipeline(true);
      SetDeferredDepth(rig, vk::CompareOp::eAlways, 0.6f, true);
      Require(name, "writing draw", rig.Draw(), "the draw after the clear failed");
    }
    run.holes = scheduler.PendingHoleCount();
    hold.release = true;
    scheduler.Finish();
    run.depth = rig.ReadDepth();
    return run;
  }

  // A hole after a draw that consumed a depth clear. The first draw is itself a hole that takes the
  // clear. The depth target must end as it does in the blocking run, with the clear value where
  // nothing wrote and the later draw's depth where it did. A clear that is lost leaves the
  // sentinel, and one that is applied again after the later draw's pass replaces its depth. The
  // blocking run is the baseline, and its absolute values show what the test claims.
  void CheckDeferredDepthClear() {
    constexpr const char *name = "DeferredDepthClear";
    for (const bool second_draw : {false, true}) {
      const auto baseline = RunDeferredClearScene(name, false, second_draw);
      const auto deferred = RunDeferredClearScene(name, true, second_draw);
      const float expected = second_draw ? 0.6f : AsyncClearDepth;
      Require(name, "baseline value", AllEqual(baseline.depth, std::bit_cast<u32>(expected)),
              "the blocking run does not end with the depth the scene is built to leave");
      Require(name, "holes", deferred.holes == (second_draw ? 2u : 1u),
              "the draws were not turned into holes");
      Require(name, "depth", deferred.depth == baseline.depth,
              second_draw ? "the clear was applied again, or lost, around the later draw"
                          : "the clear a hole consumed was not applied");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  static std::vector<std::pair<std::string, uint32_t>> DeferredIneligible() {
    return Common::FirstRunReport::Global().Summarize().deferred_ineligible;
  }

  using IneligibleCounts = std::vector<std::pair<std::string, uint32_t>>;

  // A draw that cannot be recorded later blocks as before, and is counted under its reason. Each
  // case has a baseline that differs only in the behaviour under test, and defers.
  void CheckDeferredEligibility() {
    constexpr const char *name = "DeferredEligibility";
    // An open guest occlusion query counts the samples of the draws inside it.
    for (const bool open_query : {false, true}) {
      PipelineCacheTestAccess::BuildHold hold;
      AsyncDrawRig rig(*this, false, false, true, false, true);
      auto &scheduler = rig.Scheduler();
      CommandSchedulerTestAccess::AllowThisThread(scheduler);
      rig.PaintSentinels();
      PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
      alignas(16) std::array<uint64_t, OCCLUSION_QUERY_DBS * 2> results{};
      auto &queries = rig.context->GetOcclusionQueries();
      const auto address = reinterpret_cast<uint64_t>(results.data());
      if (open_query) queries.Dump(address);
      Common::FirstRunReport::Global().Reset();
      Require(name, "query draw", rig.Draw(), "the draw reported failure");
      if (open_query) {
        Require(name, "open query blocks",
                scheduler.PendingHoleCount() == 0 && AllDrawn(rig.ReadColor()) &&
                    DeferredIneligible() == IneligibleCounts{{"open_query", 1}},
                "a draw inside an open query was deferred, or not counted under its reason");
        queries.Dump(address + sizeof(uint64_t));
        queries.Publish();
        Require(name, "query counted the draw", PassedSamples(results) > 0,
                "the blocked draw was not counted by the query");
      } else {
        Require(name, "no query defers",
                scheduler.PendingHoleCount() == 1 && DeferredIneligible().empty(),
                "the draw with no open query was not deferred");
      }
      hold.release = true;
      scheduler.Finish();
    }
    // A set of descriptors that is not pushed is allocated in the buffer the draw was met in.
    for (const bool push : {true, false}) {
      PipelineCacheTestAccess::BuildHold hold;
      AsyncDrawRig rig(*this, false, false, false, false, true);
      auto &scheduler = rig.Scheduler();
      CommandSchedulerTestAccess::AllowThisThread(scheduler);
      auto &graphics = rig.context->GetGraphics();
      const auto device_limit = graphics.max_push_descriptors;
      graphics.max_push_descriptors = push ? DrawStreamPushDescriptorLimit : 0;
      rig.PaintSentinels();
      PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
      const auto rig_vertex = UseAddressVertex(rig, DrawStreamReadOffset);
      Common::FirstRunReport::Global().Reset();
      Require(name, "descriptor draw", DrawStreamAuto(rig, 3, 1), "the draw reported failure");
      rig.shaders.SetEsShaderBase(rig_vertex);
      auto &pipelines = rig.Pipelines();
      if (push) {
        Require(name, "push defers", scheduler.PendingHoleCount() == 1 && DeferredIneligible().empty(),
                "a draw that pushes its descriptors was not deferred");
        const auto key = CommandSchedulerTestAccess::HeldHoleKey(scheduler, 1);
        Require(name, "predicted push", PipelineCacheTestAccess::PredictedPush(pipelines, key),
                "the layout code did not predict push descriptors for the deferred draw");
        hold.release = true;
        scheduler.Finish();
        Require(name, "built push", PipelineCacheTestAccess::BuiltPush(pipelines, key),
                "the pipeline built for a predicted push draw does not push");
      } else {
        const auto counts = PipelineCacheTestAccess::PushPipelineCounts(pipelines);
        Require(name, "non-push blocks",
                scheduler.PendingHoleCount() == 0 && DeferredIneligible() == IneligibleCounts{{"non_push", 1}} &&
                    counts[1] == 1 && AllDrawn(rig.ReadColor()),
                "a draw with a descriptor set was deferred, or not counted under its reason");
        hold.release = true;
        scheduler.Finish();
      }
      graphics.max_push_descriptors = device_limit;
    }
    // A shader that samples through the shared image heap needs its set bound in the draw's buffer.
    for (const bool heap : {false, true}) {
      PipelineCacheTestAccess::BuildHold hold;
      AsyncDrawRig rig(*this, false, false, false, false, true);
      auto &scheduler = rig.Scheduler();
      CommandSchedulerTestAccess::AllowThisThread(scheduler);
      rig.PaintSentinels();
      PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
      // The modules exist once a draw has asked for them, and a deferred draw does not need one yet.
      Require(name, "heap warm up", rig.Draw() && scheduler.PendingHoleCount() == 1,
              "the first draw of the heap case was not deferred");
      scheduler.Flush();
      rig.SetSecondPipeline(true);
      if (heap) PipelineCacheTestAccess::MarkModulesUseImageHeap(rig.Pipelines());
      Common::FirstRunReport::Global().Reset();
      Require(name, "heap draw", rig.Draw(), "the draw reported failure");
      if (heap) {
        Require(name, "image heap blocks",
                scheduler.PendingHoleCount() == 0 && DeferredIneligible() == IneligibleCounts{{"image_heap", 1}},
                "a draw using the image heap was deferred, or not counted under its reason");
      } else {
        Require(name, "no heap defers",
                scheduler.PendingHoleCount() == 1 && DeferredIneligible().empty(),
                "the draw without the image heap was not deferred");
      }
      hold.release = true;
      scheduler.Finish();
    }
    // A draw that cannot be deferred is counted under its reason even when an S2 skip then takes it.
    // The same draw is skipped with both flags on, and with an open query it cannot be deferred.
    const auto skipped = RunFlagScene(name, true, true, true);
    Require(name, "skip fallback is counted",
            skipped.holes == 0 && skipped.skipped && skipped.summary.skips[0] == 1 &&
                skipped.summary.deferred_ineligible == IneligibleCounts{{"open_query", 1}},
            "a draw skipped because it could not be deferred was not counted as ineligible");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Changes the registers a draw takes its primitive type, object id and clip planes from.
  static void ChangeDrawRegisters(AsyncDrawRig &rig) {
    rig.user_config.SetPrimitiveType(Prospero::PrimitiveType::kQuadListLegacy);
    rig.user_config.SetObjectId(0x1234);
    auto control = rig.registers.GetClipControl();
    control.user_clip_planes = 3;
    rig.registers.SetClipControl(control);
    for (uint32_t i = 0; i < 8; ++i) rig.registers.SetUserClipPlane(i, 0x40000000u + i);
  }

  // The commands the draw of eight vertices records. A deferred run records them when it resolves,
  // and `change_registers` says when the registers change: never, after the hole is made, or (for
  // the blocking run) before the draw is met.
  enum class RegisterChange { Never, AfterHole, BeforeDraw };

  std::vector<std::string> RecordDeferredRegisterRows(bool deferred, RegisterChange change) {
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, false, false, false, false, deferred);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    rig.PaintSentinels();
    if (deferred) PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    if (change == RegisterChange::BeforeDraw) ChangeDrawRegisters(rig);
    // A blocking draw records its commands as it is met, and a deferred one when it resolves.
    CommandStream::Recorder::Reset(!deferred);
    Require("DeferredRegisters", "draw", DrawStreamAuto(rig, 8, 1), "the draw reported failure");
    Require("DeferredRegisters", "hole", scheduler.PendingHoleCount() == (deferred ? 1u : 0u),
            "the draw was not a hole, or the blocking run made one");
    if (change == RegisterChange::AfterHole) ChangeDrawRegisters(rig);
    hold.release = true;
    if (deferred) CommandStream::Recorder::Reset(true);
    scheduler.Flush();
    auto rows = CommandStream::Recorder::rows;
    CommandStream::Recorder::active = false;
    return rows;
  }

  // The primitive type, object id and clip planes a hole's draw was met with are the ones it
  // records with, though the registers change before it is recorded. Its commands match those of
  // a run where they never change. The blocking run with the changed registers is the baseline that
  // shows they would have changed the commands, by splitting the eight vertices into two quads and
  // pushing the clip planes.
  void CheckDeferredRegistersAfterHole() {
    constexpr const char *name = "DeferredRegisters";
    CommandStream::Scope hooks;
    const auto steady = RecordDeferredRegisterRows(true, RegisterChange::Never);
    const auto changed = RecordDeferredRegisterRows(true, RegisterChange::AfterHole);
    const auto sensitive = RecordDeferredRegisterRows(false, RegisterChange::BeforeDraw);
    Require(name, "hole records the draw", HasRow(steady, "vkCmdDraw(8, 1, 0, 0)"),
            "the resolved hole did not record the draw it was met with");
    Require(name, "registers do not reach the hole",
            RenderExecutorTestAccess::RowMatches(name, changed, steady),
            "registers changed after the hole was made reached the commands it recorded");
    Require(name, "registers would change the commands",
            HasRow(sensitive, "vkCmdDraw(4, ") && !HasRow(sensitive, "vkCmdDraw(8, ") &&
                HasRow(sensitive, "vkCmdPushConstants("),
            "the changed registers do not show in a draw met after them, so the test proves nothing");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A guest query opened after a hole counts nothing of the hole, even when the hole is recorded
  // while the query is open. The hole is met before the query opens, so its samples are not the
  // query's, and a hole that asked the query state at record time would count a whole target.
  void CheckDeferredQueryOpenedAfterHole() {
    constexpr const char *name = "DeferredQueryAfterHole";
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, false, false, true, false, true);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    rig.PaintSentinels();
    PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    alignas(16) std::array<uint64_t, OCCLUSION_QUERY_DBS * 2> results{};
    auto &queries = rig.context->GetOcclusionQueries();
    const auto address = reinterpret_cast<uint64_t>(results.data());
    Require(name, "draw", rig.Draw() && scheduler.PendingHoleCount() == 1,
            "the draw before the query was not deferred");
    queries.Dump(address);
    // The query is open while the hole is recorded.
    hold.release = true;
    scheduler.Flush();
    queries.Dump(address + sizeof(uint64_t));
    queries.Publish();
    Require(name, "query counts nothing", PassedSamples(results) == 0,
            "a query opened after the hole counted samples of the hole");
    Require(name, "hole drew", AllDrawn(rig.ReadColor()), "the hole did not draw");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // The state of a run for the flag tests. The history draws make the S2 rules willing to skip a
  // draw of the second pipeline. They go first with the build workers free.
  struct FlagRun {
    Common::FirstRunReport::Summary summary;
    size_t holes = 0;
    bool skipped = false;
  };

  FlagRun RunFlagScene(const char *name, bool async, bool deferred, bool open_query) {
    FlagRun run;
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, async, false, true, false, deferred);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    rig.BuildFrameHistory();
    scheduler.Flush();
    rig.SetSecondPipeline(true);
    rig.SetDepthClear(false);
    rig.PaintSentinels();
    PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    alignas(16) std::array<uint64_t, OCCLUSION_QUERY_DBS * 2> results{};
    auto &queries = rig.context->GetOcclusionQueries();
    const auto address = reinterpret_cast<uint64_t>(results.data());
    if (open_query) queries.Dump(address);
    Common::FirstRunReport::Global().Reset();
    Require(name, "draw", rig.Draw(), "the draw reported failure");
    run.holes = scheduler.PendingHoleCount();
    run.skipped = AllSentinelColor(rig.ReadColor());
    if (open_query) {
      queries.Dump(address + sizeof(uint64_t));
      queries.Publish();
    }
    // Both ReadColor above and the summary below are after the draw and before the build is released.
    run.summary = Common::FirstRunReport::Global().Summarize();
    hold.release = true;
    scheduler.Finish();
    return run;
  }

  // The flag stands alone. With only --deferred-pipeline-draws a draw the S2 rules would skip is
  // deferred, nothing is skipped, no skip rule is asked and a draw that cannot be deferred blocks.
  // With only --async-pipelines the same draw is skipped, which is the baseline. With both a draw
  // S2 would skip is deferred and the report says what that cost it.
  void CheckDeferredFlagStandsAlone() {
    constexpr const char *name = "DeferredFlagStandsAlone";
    const auto skip_only = RunFlagScene(name, true, false, false);
    Require(name, "async alone skips",
            skip_only.skipped && skip_only.holes == 0 && skip_only.summary.skips[0] == 1,
            "with --async-pipelines alone the draw was not skipped, so it is no baseline");

    const auto flag_only = RunFlagScene(name, false, true, false);
    Require(name, "flag alone defers", flag_only.holes == 1 && !flag_only.skipped,
            "with the flag alone the draw was not deferred");
    Require(name, "flag alone skips nothing",
            flag_only.summary.skips[0] == 0 && flag_only.summary.refusals.empty() &&
                flag_only.summary.deferred_would_skip_draws == 0,
            "with the flag alone a skip rule was asked, or something was skipped");

    const auto flag_blocked = RunFlagScene(name, false, true, true);
    Require(name, "flag alone blocks what it cannot defer",
            flag_blocked.holes == 0 && !flag_blocked.skipped && flag_blocked.summary.skips[0] == 0 &&
                flag_blocked.summary.deferred_ineligible == IneligibleCounts{{"open_query", 1}},
            "with the flag alone a draw that cannot be deferred was skipped");

    const auto both = RunFlagScene(name, true, true, false);
    Require(name, "both defer first", both.holes == 1 && !both.skipped && both.summary.skips[0] == 0,
            "with both flags a draw the skip rules allow was skipped, not deferred");
    Require(name, "would skip is reported",
            both.summary.deferred_would_skip_draws == 1 && both.summary.refusals.empty(),
            "the deferred draw was not reported as one S2 would have skipped");

    const auto both_blocked = RunFlagScene(name, true, true, true);
    Require(name, "skip is the fallback", both_blocked.holes == 0 && both_blocked.skipped &&
                both_blocked.summary.skips[0] == 1 && both_blocked.summary.deferred_would_skip_draws == 0 &&
                both_blocked.summary.deferred_ineligible == IneligibleCounts{{"open_query", 1}},
            "with both flags a draw that cannot be deferred was not skipped by the S2 rules");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // How long a deferred draw that S2 would have skipped waited is the time from its hole until its
  // build is ready. The workers are held for a measured while, so the wait is at least that.
  void CheckDeferredWouldSkipWait() {
    constexpr const char *name = "DeferredWouldSkipWait";
    constexpr auto hold_for = std::chrono::milliseconds(40);
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, true, false, false, false, true);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    rig.BuildFrameHistory();
    scheduler.Flush();
    rig.SetSecondPipeline(true);
    rig.SetDepthClear(false);
    PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    Common::FirstRunReport::Global().Reset();
    Require(name, "draw", rig.Draw() && scheduler.PendingHoleCount() == 1, "the draw was not deferred");
    std::this_thread::sleep_for(hold_for);
    hold.release = true;
    scheduler.Finish();
    const auto summary = Common::FirstRunReport::Global().Summarize();
    Require(name, "waited", summary.deferred_would_skip_draws == 1 &&
                summary.deferred_would_skip_waited_ms >= hold_for.count() * 0.9,
            "the report did not count the time the deferred draw waited for its build");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // The build queue starts with six workers while --deferred-pipeline-draws is on, with or without
  // --async-pipelines, and four with --async-pipelines alone, each under the cap of half the
  // hardware threads. The flag alone still has a queue.
  void CheckDeferredWorkerCount() {
    constexpr const char *name = "DeferredWorkerCount";
    struct Case {
      bool async, deferred;
      uint32_t workers;
    };
    for (const auto test : {Case{true, false, PipelineBuildWorkerCount(PIPELINE_BUILD_WORKERS)},
                            Case{false, true, PipelineBuildWorkerCount(DEFERRED_PIPELINE_BUILD_WORKERS)},
                            Case{true, true, PipelineBuildWorkerCount(DEFERRED_PIPELINE_BUILD_WORKERS)}}) {
      AsyncDrawRig rig(*this, test.async, false, false, false, test.deferred);
      Require(name, "queue", PipelineCacheTestAccess::HasBuildQueue(rig.Pipelines()),
              "the flags did not give the cache a build queue");
      Require(name, "workers", PipelineCacheTestAccess::BuildWorkers(rig.Pipelines()) == test.workers,
              "the queue started with another number of workers");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A draw that becomes a hole counts as work in the current command buffer, where it was met, so
  // the submit cadence is due after it. Without the draw nothing is due.
  void CheckDeferredNotesWork() {
    constexpr const char *name = "DeferredNotesWork";
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, false, false, false, false, true);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    rig.PaintSentinels();
    PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    CommandSchedulerTestAccess::SetCadence(
        scheduler, MakeTestCadence(true, SubmitCadence::Clock::duration::zero()));
    Require(name, "nothing due", !CommandSchedulerTestAccess::CadenceDue(scheduler),
            "the cadence was due before any work was recorded");
    Require(name, "draw", rig.Draw() && scheduler.PendingHoleCount() == 1,
            "the draw was not deferred");
    Require(name, "due", CommandSchedulerTestAccess::CadenceDue(scheduler),
            "a hole did not count as work for the submit cadence");
    hold.release = true;
    scheduler.Finish();
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Only the GPU thread may split the command stream. A draw met on another thread blocks as before,
  // counted under its reason, and the same draw on a thread the scheduler allows is deferred.
  void CheckDeferredNeedsGpuThread() {
    constexpr const char *name = "DeferredGpuThread";
    for (const bool allowed : {true, false}) {
      PipelineCacheTestAccess::BuildHold hold;
      AsyncDrawRig rig(*this, false, false, false, false, true);
      auto &scheduler = rig.Scheduler();
      if (allowed) CommandSchedulerTestAccess::AllowThisThread(scheduler);
      rig.PaintSentinels();
      PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
      Common::FirstRunReport::Global().Reset();
      Require(name, "draw", rig.Draw(), "the draw reported failure");
      if (allowed) {
        Require(name, "defers", scheduler.PendingHoleCount() == 1 && DeferredIneligible().empty(),
                "the draw on an allowed thread was not deferred");
      } else {
        Require(name, "blocks",
                scheduler.PendingHoleCount() == 0 && AllDrawn(rig.ReadColor()) &&
                    DeferredIneligible() == IneligibleCounts{{"thread", 1}},
                "a draw on another thread was deferred, or not counted under its reason");
      }
      hold.release = true;
      scheduler.Finish();
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Runs one scenario and counts the hazards the validation layer reports for it. A hazard fails
  // the test, and the number of messages is printed.
  void RunUnderValidation(const char *scenario, const std::function<void()> &check) {
    constexpr const char *name = "DeferredValidation";
    namespace Validation = Libs::Graphics::VulkanValidation;
    Validation::ResetMessages();
    check();
    const auto messages = Validation::Messages();
    const auto hazards = static_cast<int64_t>(Validation::HazardCount());
    std::printf("[gpu]     %-32s %s: %llu messages, %lld hazards\n", name, scenario,
                static_cast<unsigned long long>(Validation::MessageCount()),
                static_cast<long long>(hazards));
    for (const auto &message : messages) std::fprintf(stderr, "%s: %s\n", scenario, message.c_str());
    Require(name, scenario, hazards == 0, "the validation layer reported a synchronization hazard");
  }

  // The deferred draw scenarios under synchronization validation, which skips and says why when the
  // device was not made with the layer. The control proves the layer reports a hazard at all.
  void CheckDeferredDrawsUnderValidation() {
    constexpr const char *name = "DeferredValidation";
    if (!SyncValidationActive()) {
      std::printf("[gpu]     %-32s skipped: %s\n", name, SyncValidationReason().c_str());
      return;
    }
    CheckSyncValidationSeesHazard();
    RunUnderValidation("DeferredOrder", [&] { CheckDeferredOrderEquivalence(); });
    RunUnderValidation("DeferredDepthClear", [&] { CheckDeferredDepthClear(); });
    RunUnderValidation("DeferredStencilClear", [&] { CheckDeferredStencilClear(); });
    RunUnderValidation("DeferredHtileClear", [&] { CheckDeferredHtileClear(); });
    RunUnderValidation("DeferredRegisters", [&] { CheckDeferredRegistersAfterHole(); });
    RunUnderValidation("DeferredQueryAfterHole", [&] { CheckDeferredQueryOpenedAfterHole(); });
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Records one rendering instance that clears a colour or a depth image and ends, in a command buffer.
  void RecordClearInstance(vk::CommandBuffer command, vk::ImageView view, bool depth,
                           vk::ImageLayout layout) {
    vk::RenderingAttachmentInfo attachment{};
    attachment.imageView = view;
    attachment.imageLayout = layout;
    attachment.loadOp = vk::AttachmentLoadOp::eClear;
    attachment.storeOp = vk::AttachmentStoreOp::eStore;
    vk::RenderingInfo info{};
    info.renderArea.extent = vk::Extent2D{AsyncExtent, AsyncExtent};
    info.layerCount = 1;
    if (depth) {
      info.pDepthAttachment = &attachment;
    } else {
      info.colorAttachmentCount = 1;
      info.pColorAttachments = &attachment;
    }
    command.beginRendering(info);
    command.endRendering();
  }

  // How the two instances of a control are placed and ordered.
  enum class ControlLayout { OneBuffer, TwoBuffers, TwoBuffersBarrier };

  // Submits two rendering instances that write one image, in one buffer or in two buffers of one
  // submit, with a barrier between them or none. Returns the hazard messages the layer reported.
  size_t CountAttachmentHazards(AsyncDrawRig &rig, bool depth, ControlLayout layout,
                                bool transfer_second) {
    auto &graphics = rig.context->GetGraphics();
    auto &textures = rig.Textures();
    const auto id = depth ? rig.depth.image_id : rig.color.image_id;
    const auto desc = depth ? rig.depth.desc : rig.color.desc;
    const auto view = depth ? textures.FindDepthTarget(id, desc) : textures.FindRenderTarget(id, desc);
    auto &image = textures.GetImage(id);
    // A transfer write needs a layout both kinds of write accept.
    const auto attachment_layout = transfer_second ? vk::ImageLayout::eGeneral
                                   : depth ? vk::ImageLayout::eDepthStencilAttachmentOptimal
                                           : vk::ImageLayout::eColorAttachmentOptimal;
    const auto access = depth ? vk::AccessFlagBits2::eDepthStencilAttachmentRead |
                                    vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                              : vk::AccessFlagBits2::eColorAttachmentRead |
                                    vk::AccessFlagBits2::eColorAttachmentWrite;
    image.Transit(attachment_layout, access, {}, rig.Scheduler().Current().Handle());
    rig.Scheduler().Finish();

    vk::CommandPoolCreateInfo pool_info{};
    pool_info.queueFamilyIndex = graphics.queue_family;
    vk::CommandPool pool;
    Require("AttachmentControl", "pool", graphics.device.createCommandPool(&pool_info, nullptr, &pool) == vk::Result::eSuccess,
            "no command pool");
    vk::CommandBufferAllocateInfo allocate{};
    allocate.commandPool = pool;
    allocate.commandBufferCount = 2;
    std::array<vk::CommandBuffer, 2> commands{};
    Require("AttachmentControl", "buffers", graphics.device.allocateCommandBuffers(&allocate, commands.data()) == vk::Result::eSuccess,
            "no command buffers");
    vk::CommandBufferBeginInfo begin{};
    for (auto command : commands) (void)command.begin(&begin);
    const auto barrier = [&](vk::CommandBuffer command) {
      vk::MemoryBarrier2 memory{};
      memory.srcStageMask = depth ? vk::PipelineStageFlagBits2::eLateFragmentTests
                                  : vk::PipelineStageFlagBits2::eColorAttachmentOutput;
      memory.dstStageMask = transfer_second ? vk::PipelineStageFlagBits2::eAllTransfer
                            : memory.srcStageMask | vk::PipelineStageFlagBits2::eEarlyFragmentTests;
      memory.srcAccessMask = depth ? vk::AccessFlagBits2::eDepthStencilAttachmentWrite
                                   : vk::AccessFlagBits2::eColorAttachmentWrite;
      memory.dstAccessMask = transfer_second ? vk::AccessFlagBits2::eTransferWrite : access;
      vk::DependencyInfo dependency{};
      dependency.memoryBarrierCount = 1;
      dependency.pMemoryBarriers = &memory;
      command.pipelineBarrier2(dependency);
    };
    RecordClearInstance(commands[0], view, depth, attachment_layout);
    const auto second = layout == ControlLayout::OneBuffer ? commands[0] : commands[1];
    if (layout == ControlLayout::TwoBuffersBarrier) barrier(second);
    if (transfer_second) {
      const vk::ImageSubresourceRange range{depth ? vk::ImageAspectFlagBits::eDepth
                                                  : vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
      if (depth) {
        const vk::ClearDepthStencilValue value{0.25f, 0};
        second.clearDepthStencilImage(image.backing.image, vk::ImageLayout::eGeneral, &value, 1, &range);
      } else {
        const vk::ClearColorValue value{std::array{0.25f, 0.25f, 0.25f, 0.25f}};
        second.clearColorImage(image.backing.image, vk::ImageLayout::eGeneral, &value, 1, &range);
      }
    } else {
      RecordClearInstance(second, view, depth, attachment_layout);
    }
    for (auto command : commands) (void)command.end();
    vk::SubmitInfo submit{};
    submit.commandBufferCount = layout == ControlLayout::OneBuffer ? 1u : 2u;
    submit.pCommandBuffers = commands.data();
    namespace Validation = Libs::Graphics::VulkanValidation;
    Validation::ResetMessages();
    Require("AttachmentControl", "submit", graphics.queue.submit(1, &submit, nullptr) == vk::Result::eSuccess, "submit failed");
    (void)graphics.queue.waitIdle();
    const auto messages = Validation::Messages();
    for (const auto &message : messages) std::fprintf(stderr, "control: %s%s", message.c_str(), "\n");
    const auto hazards = static_cast<size_t>(Validation::HazardCount());
    graphics.device.destroyCommandPool(pool, nullptr);
    return hazards;
  }

  // A positive control for hazards between rendering instances that write one attachment. The hole
  // layout is two buffers in one submit with no barrier. If the layer reports it there, a missing
  // hole barrier is visible to validation. The barrier variant is the baseline.
  void CheckAttachmentHazardControl() {
    constexpr const char *name = "AttachmentHazardControl";
    if (!SyncValidationActive()) {
      std::printf("[gpu]     %-32s skipped: %s\n", name, SyncValidationReason().c_str());
      return;
    }
    for (const bool transfer : {false, true}) {
    for (const bool depth : {false, true}) {
      for (const auto layout : {ControlLayout::OneBuffer, ControlLayout::TwoBuffers,
                                ControlLayout::TwoBuffersBarrier}) {
        AsyncDrawRig rig(*this, false);
        rig.PaintSentinels();
        const auto hazards = CountAttachmentHazards(rig, depth, layout, transfer);
        // The layer tracks an attachment's store as a write, so a transfer write after it with no
        // barrier is a hazard and the same pair with a barrier is not.
        if (transfer && layout == ControlLayout::TwoBuffers) {
          Require(name, "transfer after attachment", hazards > 0,
                  "the layer did not report a transfer write after an attachment write");
        }
        if (transfer && layout == ControlLayout::TwoBuffersBarrier) {
          Require(name, "ordered by a barrier", hazards == 0,
                  "the layer reported a hazard where a barrier orders the writes");
        }
        // The one-buffer pair has no barrier either, which the spec calls a hazard. The layer does not
        // report it today, and a newer one may, so it is printed and never asserted.
        std::printf("[gpu]     %-32s %s then %s, %s: %zu hazards\n", name, depth ? "depth" : "colour",
                    transfer ? "transfer write" : "attachment write",
                    layout == ControlLayout::OneBuffer ? "one buffer"
                    : layout == ControlLayout::TwoBuffers ? "two buffers, no barrier"
                                                          : "two buffers, barrier",
                    hazards);
      }
    }
    }
  }

  // One hole that consumes a pending HTILE clear of the depth target, which no register asks for.
  DeferredClearRun RunDeferredHtileScene(const char *name, bool deferred) {
    DeferredClearRun run;
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, false, true, false, false, deferred);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    rig.PaintSentinels();
    rig.MarkHtileClear();
    if (deferred) PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    SetDeferredDepth(rig, vk::CompareOp::eAlways, 0.0f, false);
    rig.SetDepthClear(false);
    Require(name, "htile clearing draw", rig.Draw(), "the draw that consumes the clear failed");
    run.holes = scheduler.PendingHoleCount();
    hold.release = true;
    scheduler.Finish();
    run.depth = rig.ReadDepth();
    return run;
  }

  // A hole that consumes a pending HTILE clear applies it once, as the blocking run does. The clear
  // value is the register's, and no register clear is enabled, so only the HTILE path can supply it.
  void CheckDeferredHtileClear() {
    constexpr const char *name = "DeferredHtileClear";
    const auto baseline = RunDeferredHtileScene(name, false);
    const auto deferred = RunDeferredHtileScene(name, true);
    Require(name, "baseline value", AllEqual(baseline.depth, std::bit_cast<u32>(AsyncClearDepth)),
            "the blocking run does not end with the cleared depth, so the scene proves nothing");
    Require(name, "hole", deferred.holes == 1, "the draw was not turned into a hole");
    Require(name, "depth", deferred.depth == baseline.depth,
            "the HTILE clear a hole consumed was not applied");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  static constexpr uint64_t DeferredStencilOffset = 0x60000;
  static constexpr uint8_t DeferredStencilSentinel = 0x11;
  static constexpr uint8_t DeferredStencilClear = 0x5a;

  // The depth target of a rig with a stencil plane of its own, as the guest names one.
  static void GiveRigStencil(AsyncDrawRig &rig) {
    auto target = rig.registers.GetDepthRenderTarget();
    target.stencil_info.format = Prospero::StencilFormat::k8UInt;
    target.stencil_read_base_addr = target.stencil_write_base_addr = AsyncBase + DeferredStencilOffset;
    rig.registers.SetDepthRenderTarget(target);
  }

  struct DeferredStencilRun {
    std::vector<u32> depth;
    std::vector<uint8_t> stencil;
    size_t holes = 0;
  };

  // One hole that consumes a stencil clear and no depth clear, over a depth and stencil target
  // painted with sentinels. Nothing writes either plane, so what is left is the clear alone.
  DeferredStencilRun RunDeferredStencilScene(const char *name, bool deferred) {
    DeferredStencilRun run;
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, false, false, false, false, deferred);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    GiveRigStencil(rig);
    rig.PaintSentinels();
    vk::ClearValue sentinel{};
    sentinel.depthStencil = vk::ClearDepthStencilValue{0.0f, DeferredStencilSentinel};
    TextureCacheTestAccess::ClearImage(rig.Textures(), scheduler.Current(), rig.depth.image_id,
                                       {vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1}, sentinel);
    if (deferred) PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    SetDeferredDepth(rig, vk::CompareOp::eAlways, 0.0f, false);
    rig.registers.SetStencilClearValue(DeferredStencilClear);
    rig.registers.SetRenderControl({.depth_clear_enable = false, .stencil_clear_enable = true});
    Require(name, "stencil clearing draw", rig.Draw(), "the draw that consumes the clear failed");
    rig.registers.SetRenderControl({});
    run.holes = scheduler.PendingHoleCount();
    hold.release = true;
    scheduler.Finish();
    auto &image = rig.Textures().GetImage(rig.depth.image_id);
    auto probe = CreateHostBuffer(name, AsyncExtent * AsyncExtent, vk::BufferUsageFlagBits::eTransferDst, {});
    vk::BufferImageCopy copy{};
    copy.imageSubresource = {vk::ImageAspectFlagBits::eStencil, 0, 0, 1};
    copy.imageExtent = vk::Extent3D{AsyncExtent, AsyncExtent, 1};
    image.Download(std::span{&copy, 1}, probe.buffer, 0, probe.size);
    vk::BufferMemoryBarrier barrier{};
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = probe.buffer;
    barrier.size = probe.size;
    scheduler.Current().Handle().pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                                 vk::PipelineStageFlagBits::eHost, {}, 0, nullptr, 1,
                                                 &barrier, 0, nullptr);
    scheduler.Finish();
    const auto words = ReadBuffer(name, probe, probe.size / sizeof(u32));
    run.stencil.resize(probe.size);
    std::memcpy(run.stencil.data(), words.data(), probe.size);
    DestroyBuffer(&probe);
    run.depth = rig.ReadDepth();
    return run;
  }

  // A hole that consumes a stencil clear leaves the stencil plane as the blocking run does, with the
  // clear value everywhere and the depth plane untouched. The sentinels tell a lost clear from one
  // that was applied. A pending HTILE clear has its own test below.
  void CheckDeferredStencilClear() {
    constexpr const char *name = "DeferredStencilClear";
    const auto baseline = RunDeferredStencilScene(name, false);
    const auto deferred = RunDeferredStencilScene(name, true);
    Require(name, "baseline value",
            std::ranges::all_of(baseline.stencil, [](uint8_t value) { return value == DeferredStencilClear; }) &&
                AllEqual(baseline.depth, std::bit_cast<u32>(AsyncSentinelDepth)),
            "the blocking run does not end with the stencil clear value and the untouched depth");
    Require(name, "hole", deferred.holes == 1, "the draw was not turned into a hole");
    Require(name, "stencil", deferred.stencil == baseline.stencil,
            "the stencil clear a hole consumed was not applied");
    Require(name, "depth", deferred.depth == baseline.depth, "the depth plane changed");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  void CheckDeferredDraws() {
    CheckDeferredOrderEquivalence();
    CheckDeferredDepthClear();
    CheckDeferredStencilClear();
    CheckDeferredHtileClear();
    CheckDeferredEligibility();
    CheckDeferredRegistersAfterHole();
    CheckDeferredQueryOpenedAfterHole();
    CheckDeferredFlagStandsAlone();
    CheckDeferredWouldSkipWait();
    CheckDeferredWorkerCount();
    CheckDeferredNotesWork();
    CheckDeferredNeedsGpuThread();
  }
