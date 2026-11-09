  // Tests of the split descriptor layout of --graphics-pipeline-library, in which the VS bindings
  // are set 0, the image heap stays set 1 and the PS bindings are set 2. Included inside
  // VulkanHarness, after GraphicsCreateInfoTests.inc.

  // Turns the library mode on in the runtime context for the life of the scope, as a device that
  // has the library and --graphics-pipeline-library would. The context alone decides how pipelines
  // and modules are built, so a test outside a scope builds exactly what it always did.
  struct SplitLayoutScope {
    GraphicContext &graphics;
    bool previous;
    SplitLayoutScope(VulkanHarness &harness, bool on)
        : graphics(harness.m_runtime_context),
          previous(harness.m_runtime_context.graphics_pipeline_library_enabled) {
      graphics.graphics_pipeline_library_enabled = on;
    }
    ~SplitLayoutScope() { graphics.graphics_pipeline_library_enabled = previous; }
    SplitLayoutScope(const SplitLayoutScope &) = delete;
    SplitLayoutScope &operator=(const SplitLayoutScope &) = delete;
  };

  // Where the shaders of these tests read, at addresses no other test's shader reads.
  static constexpr uint64_t SplitVertexReadOffset = DrawStreamReadOffset + 0x500;
  static constexpr uint64_t SplitPixelReadOffset = DrawStreamReadOffset + 0x600;
  // The prediction, deferral and recording checks each read at an address of their own, so each is a
  // shader of its own.
  static constexpr uint64_t SplitPredictionReadOffset = SplitVertexReadOffset + 0x100;
  static constexpr uint64_t SplitDeferralReadOffset = SplitVertexReadOffset + 0x200;
  static constexpr uint64_t SplitRecordReadOffset = SplitVertexReadOffset + 0x300;
  // The vertex shader scales its position by the first value, so a draw covers the target only if
  // the shader read it. The pixel shader adds the second to the rig's gray of 0.25, which is exact.
  static constexpr float SplitVertexScale = 1.0f;
  static constexpr float SplitPixelAddend = 0.125f;
  static constexpr u32 SplitPixelBits = std::bit_cast<u32>(0.25f + SplitPixelAddend);
  // The first descriptor limit lets a pixel set be pushed and the second lets none be.
  static constexpr uint32_t SplitPushLimit = DrawStreamPushDescriptorLimit;
  static constexpr uint32_t SplitNoPushLimit = 0;

  // Writes a word the shaders read. A global load reads a page that has no buffer as zero, so the
  // page is backed first, and the cache is told of the write so the GPU sees it at the draw.
  static void WriteGuestWord(AsyncDrawRig &rig, uint64_t offset, u32 value) {
    auto &cache = rig.context->GetBufferCache();
    (void)cache.FindBufferForShaderRead(AsyncBase + offset, sizeof(u32));
    *reinterpret_cast<u32 *>(AsyncBase + offset) = value;
    cache.InvalidateMemory(AsyncBase + offset, sizeof(u32));
    cache.PromoteBdaSync();
  }

  // What the rig's picture is when both stages read what they were given. Every texel is the
  // gray plus the addend with alpha one.
  static std::vector<u32> SplitExpectedPicture() {
    std::vector<u32> picture(AsyncExtent * AsyncExtent * 4u, SplitPixelBits);
    for (size_t index = 3; index < picture.size(); index += 4) picture[index] = AsyncDrawnAlpha;
    return picture;
  }

  // One draw of the shaders that read by address in both stages.
  struct SplitDraw {
    std::vector<u32> color;
    std::vector<std::string> rows;
    // What the pipeline the draw built says about its layout.
    bool split_layout = false;
    bool push_only = false;
    bool pixel_set_push = false;
    bool vertex_layout = false;
  };

  // Draws once with the library mode `split` and the given limit on push descriptors, and keeps the
  // picture, the recorded commands and what the pipeline says about its layout.
  SplitDraw RunSplitPixelDraw(bool split, uint32_t push_limit) {
    SplitDraw run;
    CommandStream::Scope hooks;
    SplitLayoutScope layout(*this, split);
    AsyncDrawRig rig(*this, false);
    auto &graphics = rig.context->GetGraphics();
    const auto device_limit = graphics.max_push_descriptors;
    graphics.max_push_descriptors = push_limit;
    WriteGuestWord(rig, SplitVertexReadOffset, std::bit_cast<u32>(SplitVertexScale));
    WriteGuestWord(rig, SplitPixelReadOffset, std::bit_cast<u32>(SplitPixelAddend));
    UseAddressVertex(rig, SplitVertexReadOffset, true);
    UseAddressPixel(rig, SplitPixelReadOffset);
    rig.PaintSentinels();
    CommandStream::Recorder::Reset(true);
    Require("SplitLayoutPixels", "draw", rig.Draw(), "the draw reported failure");
    run.rows = CommandStream::Recorder::rows;
    CommandStream::Recorder::active = false;
    run.color = rig.ReadColor();
    const auto key = PipelineCacheTestAccess::OnlyKey(rig.Pipelines());
    Require("SplitLayoutPixels", "one pipeline", key.has_value(),
            "the draw did not build exactly one pipeline");
    const auto &pipeline = PipelineCacheTestAccess::PipelineOf(rig.Pipelines(), *key);
    run.split_layout = pipeline.uses_split_sets;
    run.push_only = pipeline.uses_push_descriptors;
    run.pixel_set_push = pipeline.pixel_set_is_push;
    run.vertex_layout = pipeline.vertex_set_layout != nullptr;
    graphics.max_push_descriptors = device_limit;
    return run;
  }

  // The native binding numbers the pixel and vertex address buffers get, which tell the stages
  // apart in a recorded descriptor write. The vertex ones are below the pixel ones. The text ends in
  // the space that follows the number in a row, so it does not match a longer number.
  static std::string SplitBindingText(ShaderType stage) {
    using Kind = ShaderRecompiler::IR::DescriptorBindingKind;
    return "bind=" + std::to_string(ShaderRecompiler::IR::NativeBinding(stage, Kind::BdaPagetable)) + " ";
  }

  // What the recorded commands of a draw say about where its descriptors went. In the split layout
  // the vertex set is bound at set 0 and the pixel set is pushed at set 2 or bound at set 2.
  void CheckSplitRows(const char *name, const SplitDraw &on, bool pixel_push) {
    const auto vertex_bind = SplitBindingText(ShaderType::Vertex);
    const auto pixel_bind = SplitBindingText(ShaderType::Pixel);
    Require(name, "vertex set at 0", HasRow(on.rows, "first 0 set#"),
            "the vertex bindings were not bound as a descriptor set at set 0");
    if (pixel_push) {
      Require(name, "pixel set pushed at 2",
              HasRow(on.rows, "set 2; " + pixel_bind) && !HasRow(on.rows, "set 0; " + pixel_bind),
              "the pixel bindings were not pushed at set 2");
    } else {
      Require(name, "pixel set bound at 2",
              HasRow(on.rows, "first 2 set#") && !HasRow(on.rows, "vkCmdPushDescriptorSet("),
              "the pixel bindings were not bound as a descriptor set at set 2");
    }
    Require(name, "vertex bindings not pushed", !HasRow(on.rows, vertex_bind),
            "the vertex bindings were written into a push set");
  }

  // One pass of the pixel check in one layout. The picture is the absolute expected one, the pipeline
  // has the layout under test, and the recorded commands put each stage's descriptors where that
  // layout says. Returns the picture, for the comparison between layouts.
  std::vector<u32> CheckSplitPixelPass(bool split, uint32_t limit) {
    constexpr const char *name = "SplitLayoutPixels";
    const bool pixel_push = limit != SplitNoPushLimit;
    const auto run = RunSplitPixelDraw(split, limit);
    // A lost vertex read leaves nothing drawn and a lost pixel read leaves the gray without the
    // addend, so the absolute picture sees either one, which two layouts that both lose a read
    // would not show by agreeing.
    Require(name, split ? "split picture" : "baseline picture", run.color == SplitExpectedPicture(),
            "the draw left another picture than both stages reading their values");
    Require(name, "layout", run.split_layout == split,
            "the pipeline did not get the layout under test");
    if (split) {
      Require(name, "pixel set", run.vertex_layout && run.pixel_set_push == pixel_push,
              "the pipeline did not get the pixel set the limit allows");
      // The vertex set is never pushed, so only the single set can be push only.
      Require(name, "push only", !run.push_only,
              "a pipeline whose VS has bindings claims to need no descriptor set");
      CheckSplitRows(name, run, pixel_push);
    } else {
      Require(name, "push only", run.push_only == pixel_push,
              "the pipeline's push flag does not say whether it allocates a descriptor set");
    }
    return run.color;
  }

  // The split layout draws the picture the single set draws, with the pixel set pushed and with it
  // allocated.
  void CheckSplitLayoutPixels() {
    constexpr const char *name = "SplitLayoutPixels";
    for (const uint32_t limit : {SplitPushLimit, SplitNoPushLimit}) {
      const auto off = CheckSplitPixelPass(false, limit);
      const auto on = CheckSplitPixelPass(true, limit);
      Require(name, "same picture", on == off,
              "the split layout drew another picture than the single set");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Counts the modules a cache compiled for one draw, told apart by stage and by whether they have
  // a library handle.
  struct SplitModuleCounts {
    size_t pixel = 0;
    size_t pixel_with_library = 0;
    size_t other = 0;
    size_t other_with_library = 0;
    // A library handle that is the plain handle would not be a patched copy.
    size_t library_is_plain = 0;
  };

  // Draws once in the given library mode and counts the modules the cache compiled for it.
  SplitModuleCounts CountSplitModules(bool library_mode) {
    SplitModuleCounts counts;
    SplitLayoutScope layout(*this, library_mode);
    AsyncDrawRig rig(*this, false);
    rig.PaintSentinels();
    Require("SplitLayoutModules", "draw", rig.Draw(), "the draw reported failure");
    for (const auto &module : PipelineCacheTestAccess::Modules(rig.Pipelines())) {
      const bool pixel = module.stage == ShaderType::Pixel;
      (pixel ? counts.pixel : counts.other)++;
      if (module.gpl_module != nullptr) {
        (pixel ? counts.pixel_with_library : counts.other_with_library)++;
        counts.library_is_plain += module.gpl_module == module.module ? 1u : 0u;
      }
    }
    return counts;
  }

  // Without the library mode no module has a library handle, and with it every pixel module has
  // one and no other module does. A device that cannot have the library only runs the first part,
  // since the second needs the mode on. The first part has a pixel module to count, so a harness
  // that built no module would not pass it. The second part is its baseline.
  void CheckSplitLayoutModules() {
    constexpr const char *name = "SplitLayoutModules";
    const auto inactive = CountSplitModules(false);
    Require(name, "inactive has modules", inactive.pixel != 0 && inactive.other != 0,
            "the draw compiled no pixel module and no other module, so nothing was counted");
    Require(name, "inactive makes none",
            inactive.pixel_with_library == 0 && inactive.other_with_library == 0,
            "a module got a library handle while the library mode was off");
    if (m_library_available) {
      const auto active = CountSplitModules(true);
      Require(name, "active pixel modules",
              active.pixel == inactive.pixel && active.pixel_with_library == active.pixel &&
                  active.library_is_plain == 0,
              "a pixel module did not get a patched copy while the library mode was on");
      Require(name, "active other modules", active.other_with_library == 0,
              "a module that is not a pixel shader got a library handle");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // How many DescriptorSet decorations in a module have the given set. This walks the module itself,
  // apart from the code under test, and counts OpDecorate instructions only.
  static size_t CountSetDecorations(const std::vector<u32> &spirv, u32 set) {
    constexpr size_t header_words = 5;
    constexpr u32 op_decorate = 71;
    constexpr u32 decoration_descriptor_set = 34;
    constexpr u32 length_shift = 16, opcode_mask = 0xffffu;
    size_t count = 0;
    for (size_t at = header_words; at < spirv.size(); at += spirv[at] >> length_shift) {
      if ((spirv[at] & opcode_mask) == op_decorate && spirv[at + 2] == decoration_descriptor_set &&
          spirv[at + 3] == set) {
        ++count;
      }
    }
    return count;
  }

  // The words in which two modules of the same length differ.
  static size_t CountChangedWords(const std::vector<u32> &before, const std::vector<u32> &after) {
    size_t changed = 0;
    for (size_t word = 0; word < before.size(); ++word) changed += before[word] != after[word];
    return changed;
  }

  // Appends the instructions of a heap image sample to the first block of the program.
  static void AppendHeapSample(ShaderRecompiler::IR::Program &program) {
    using namespace ShaderRecompiler::IR;
    auto *block = program.blocks.front();
    auto &key = block->AppendNewInst(ValueOpcode::LaneId);
    auto &image = block->AppendNewInst(ValueOpcode::GetImageResource,
                                       {Value(&key), Value(0u), Value(0u), Value(0u), Value(0u),
                                        Value(0u), Value(0u), Value(0u)});
    image.SetFlags<uint32_t>(0u);
    auto &sampler = block->AppendNewInst(ValueOpcode::GetSamplerResource,
                                         {Value(0u), Value(0u), Value(0u), Value(0u)});
    sampler.SetFlags<uint32_t>(0u);
    auto &address = block->AppendNewInst(
        ValueOpcode::MakeImageAddress,
        {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
         Value(0u), Value(0u), Value(0u), Value(0u), Value(0u)});
    MemoryInfo memory{};
    memory.kind = ResourceKind::Image;
    memory.dmask = 0xf;
    memory.image_sample_flags = ShaderRecompiler::Decoder::ImageSampleFlagLod;
    memory.image_dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
    memory.image_address_components = 3;
    program.memory_info.push_back(memory);
    const MemoryFlags memory_flags{0u, 0x10f0u};
    uint64_t memory_flag_bits = 0;
    std::memcpy(&memory_flag_bits, &memory_flags, sizeof(memory_flags));
    auto &sample = block->AppendNewInst(
        ValueOpcode::ImageSampleRaw,
        {Value(&image), Value(&sampler), Value(&address), Value(true)}, memory_flag_bits);
    auto &sample_x = block->AppendNewInst(ValueOpcode::CompositeExtractU32x4,
                                          {Value(&sample), Value(0u)});
    block->AppendNewInst(ValueOpcode::ReferenceU32, {Value(&sample_x)});
  }

  // Describes the resources of the heap sample, which are a heap image root and one candidate, a
  // sampler, and the descriptor sources that select them.
  static void DescribeHeapSample(ShaderRecompiler::IR::Program &program) {
    using namespace ShaderRecompiler::IR;
    program.descriptor_sources.resize(2);
    program.descriptor_sources[0].dword_count = 8;
    program.descriptor_sources[0].indirect_descriptor = DescriptorSource::IndirectDescriptor{
        .selector = DescriptorSource::IndirectDescriptor::SelectorRead{
            .source = 0, .stride = 224, .offset = 12},
        .table_source = 0, .table_stride = 32};
    program.descriptor_sources[1].dword_count = 4;
    ImageResource root{};
    root.source = 0;
    root.first_use_pc = 0x10f0u;
    root.resource_class = ImageResourceClass::Sampled;
    root.numeric_class = Prospero::TextureNumericClass::Float;
    root.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2D;
    root.read = true;
    root.indirect_root = 0;
    root.heap = true;
    root.indirect_resources = {0u, 1u};
    auto candidate = root;
    candidate.indirect_resources.clear();
    program.info.images = {root, candidate};
    program.info.samplers.push_back({1u, 0x10f0u});
    program.info.sampled_pairs.push_back({0u, 0u, 0x10f0u});
  }

  // A pixel shader that samples through the resident image heap. The heap array is in set 1 and
  // the sampler is in set 0, so it has decorations of both sets.
  static std::vector<u32> EmitHeapPixelSpirv() {
    ShaderRecompiler::IR::Program program{};
    program.stage = ShaderType::Pixel;
    program.wave_size = 32;
    program.srt_plan_complete = true;
    program.resource_tracking_complete = true;
    program.shader_info_complete = true;
    program.block_storage.push_back(std::make_unique<ShaderRecompiler::IR::Block>());
    program.blocks.push_back(program.block_storage.back().get());
    AppendHeapSample(program);
    DescribeHeapSample(program);
    ShaderPixelInputInfo pixel{};
    program.shader_info_complete = false;
    CollectShaderInfo(program, {.pixel = &pixel});
    return ShaderRecompiler::Spirv::EmitProgram(program, {.pixel = &pixel});
  }

  // Moving the PS bindings to set 2 changes every set 0 decoration and nothing else. The image heap
  // is in set 1, hard-coded by the emitter, and must stay there, and no other word may change. The
  // emitted shaders are a PS that reads by address, which has set 0 decorations only, and a PS that
  // samples through the heap, which has both. A patch that also moved set 1 fails the second.
  void CheckSplitLayoutPatch() {
    constexpr const char *name = "SplitLayoutPatch";
    constexpr u32 emitted_set = 0, heap_set = 1, pixel_set = 2;
    GraphicsCase address_case;
    address_case.name = name;
    address_case.fragment_code = AddressPixelCode(AsyncBase + SplitPixelReadOffset);
    const auto address_spirv = CompileFragmentCase(address_case).spirv;
    const auto heap_spirv = EmitHeapPixelSpirv();
    ValidateSpirv(name, heap_spirv);
    for (const auto *spirv : {&address_spirv, &heap_spirv}) {
      const bool has_heap = spirv == &heap_spirv;
      const auto moved = MoveDescriptorSet(*spirv, emitted_set, pixel_set);
      const size_t emitted = CountSetDecorations(*spirv, emitted_set);
      const size_t heap = CountSetDecorations(*spirv, heap_set);
      Require(name, "shader has set 0 resources", emitted != 0 && (heap != 0) == has_heap,
              "the shader does not have the decorations this case is about");
      Require(name, "no set 0 left", CountSetDecorations(moved, emitted_set) == 0,
              "a descriptor set 0 decoration was left in the patched shader");
      Require(name, "all moved to the pixel set",
              CountSetDecorations(moved, pixel_set) ==
                  emitted + CountSetDecorations(*spirv, pixel_set),
              "the set 0 decorations did not all become the pixel set");
      Require(name, "heap untouched", CountSetDecorations(moved, heap_set) == heap,
              "the image heap decorations changed");
      Require(name, "only decorations changed",
              moved.size() == spirv->size() && CountChangedWords(*spirv, moved) == emitted,
              "a word that is not a set 0 decoration changed, or the module changed size");
      ValidateSpirv(name, moved);
    }
    std::printf("[host]    %-32s ok\n", name);
  }

  // One check of the production device query against the device itself.
  void CheckSplitLayoutDevice() {
    constexpr const char *name = "SplitLayoutDevice";
    const auto listed = m_physical_device.enumerateDeviceExtensionProperties();
    Require(name, "extension list", listed.result == vk::Result::eSuccess,
            "could not list the device extensions");
    const auto has = [&](const char *extension) {
      return std::ranges::any_of(listed.value, [&](const auto &found) {
        return std::strcmp(found.extensionName.data(), extension) == 0;
      });
    };
    // The device is asked with the flag off first. Nothing is added and the answer is no.
    Config::Load({.printf_direction = Config::LogDirection::Silent});
    std::vector<const char *> extensions;
    Require(name, "flag off",
            !EnablePipelineLibrary(m_physical_device, listed.value, extensions) &&
                extensions.empty() && !QueryPipelineLibrarySupport(m_physical_device, listed.value).flag,
            "the library was enabled, or the query read the flag, with the flag off");
    // With the flag on, the answer is what the device says on its own.
    Config::Load({.printf_direction = Config::LogDirection::Silent,
                  .graphics_pipeline_library_enabled = true});
    const auto support = QueryPipelineLibrarySupport(m_physical_device, listed.value);
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT> features;
    m_physical_device.getFeatures2(&features.get<vk::PhysicalDeviceFeatures2>());
    vk::StructureChain<vk::PhysicalDeviceProperties2,
                       vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT> properties;
    m_physical_device.getProperties2(&properties.get<vk::PhysicalDeviceProperties2>());
    const auto &library = properties.get<vk::PhysicalDeviceGraphicsPipelineLibraryPropertiesEXT>();
    const bool extensions_listed = has(VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME) &&
                                   has(VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME);
    const bool meets = extensions_listed &&
        features.get<vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT>().graphicsPipelineLibrary &&
        library.graphicsPipelineLibraryFastLinking &&
        library.graphicsPipelineLibraryIndependentInterpolationDecoration;
    std::vector<const char *> enabled;
    const bool on = EnablePipelineLibrary(m_physical_device, listed.value, enabled);
    Config::Load({.printf_direction = Config::LogDirection::Silent});
    Require(name, "flag read", support.flag, "the query did not read the flag");
    Require(name, "extensions", support.extensions == extensions_listed,
            "the query disagrees with the extension list");
    Require(name, "answer", on == meets && (!m_library_requested || on == m_library_available),
            "the production enable code disagrees with the device, or with how the harness was made");
    Require(name, "extensions added", enabled.size() == (on ? 2u : 0u),
            "the enable code added other extensions than the two the library needs");
    std::printf("[gpu]     %-32s ok (library %s)\n", name, on ? "available" : "not available");
  }

  // The pipeline the one draw of the rig built, with what the layout code predicted for its key.
  struct SplitPrediction {
    bool predicted_push = false;
    bool built_push = false;
    bool split_layout = false;
  };

  // Draws once with the given stage reading by address and the push limit, and asks the layout code
  // for its prediction for the key the draw made. `rect_list` draws a rectangle list, which libraries
  // do not build, and so the split layout is not used for it.
  SplitPrediction PredictSplitDraw(bool split, AddressStage stage, uint32_t push_limit,
                                   bool rect_list = false) {
    SplitPrediction result;
    SplitLayoutScope layout(*this, split);
    AsyncDrawRig rig(*this, false);
    auto &graphics = rig.context->GetGraphics();
    const auto device_limit = graphics.max_push_descriptors;
    graphics.max_push_descriptors = push_limit;
    if (rect_list) rig.user_config.SetPrimitiveType(Prospero::PrimitiveType::kRectList);
    UseAddressStage(rig, stage, SplitPredictionReadOffset);
    rig.PaintSentinels();
    Require("SplitLayoutPrediction", "draw", rig.Draw(), "the draw reported failure");
    const auto key = PipelineCacheTestAccess::OnlyKey(rig.Pipelines());
    Require("SplitLayoutPrediction", "one pipeline", key.has_value(),
            "the draw did not build exactly one pipeline");
    result.predicted_push = PipelineCacheTestAccess::PredictedPush(rig.Pipelines(), *key);
    result.built_push = PipelineCacheTestAccess::BuiltPush(rig.Pipelines(), *key);
    result.split_layout = PipelineCacheTestAccess::PipelineOf(rig.Pipelines(), *key).uses_split_sets;
    graphics.max_push_descriptors = device_limit;
    return result;
  }

  // The cases of the prediction check. The single set cases are the baselines of the split ones.
  struct SplitPredictionCase {
    const char *label;
    bool split;
    AddressStage stage;
    uint32_t limit;
    bool rect_list;
    bool expect_split_layout;
    bool expect_push;
  };

  // The cases of the prediction check. Each has the layout the draw should get and whether the
  // layout code should say push, for a stage that reads by address, a push limit and a draw kind.
  static std::vector<SplitPredictionCase> SplitPredictionCases() {
    return {
        {"single set, vertex bindings", false, AddressStage::Vertex, SplitPushLimit, false, false, true},
        {"split, vertex bindings", true, AddressStage::Vertex, SplitPushLimit, false, true, false},
        {"split, pixel bindings pushed", true, AddressStage::Pixel, SplitPushLimit, false, true, true},
        {"split, pixel bindings in a set", true, AddressStage::Pixel, SplitNoPushLimit, false, true, false},
        {"single set, pixel bindings in a set", false, AddressStage::Pixel, SplitNoPushLimit, false, false, false},
        {"split, rect list keeps the single set", true, AddressStage::Vertex, SplitPushLimit, true, false, true},
        {"single set, rect list", false, AddressStage::Vertex, SplitPushLimit, true, false, true},
    };
  }

  // Runs the prediction cases of one layout. With `split` unset these are the baselines.
  void CheckSplitPredictionPass(bool split) {
    constexpr const char *name = "SplitLayoutPrediction";
    for (const auto &item : SplitPredictionCases()) {
      if (item.split != split) continue;
      const auto result = PredictSplitDraw(item.split, item.stage, item.limit, item.rect_list);
      Require(name, item.label,
              result.split_layout == item.expect_split_layout &&
                  result.predicted_push == item.expect_push && result.built_push == item.expect_push,
              "the prediction, the built pipeline or the layout kind is not the expected one");
    }
  }

  // The prediction for a draw that has not been built agrees with the pipeline that is built, in
  // both layouts. A vertex shader with bindings makes the draw non-push under the split layout and
  // not under the single set, which is the case the single-set rule gets wrong. A pixel shader with
  // bindings is pushed in both. A draw the libraries do not build keeps the single set, so it is
  // judged by it, which needs the key and not just the shaders.
  void CheckSplitLayoutPrediction() {
    CheckSplitPredictionPass(false);
    CheckSplitPredictionPass(true);
    std::printf("[gpu]     %-32s ok\n", "SplitLayoutPrediction");
  }

  struct SplitDeferral {
    size_t holes = 0;
    IneligibleCounts ineligible;
    bool predicted_push = true;
    bool built_push = true;
    std::vector<u32> color;
  };

  // One draw of a stage that reads by address with --deferred-pipeline-draws on and the build held,
  // as a deferred draw meets it, then the picture after the build is let go.
  SplitDeferral RunSplitDeferral(bool split, AddressStage stage, uint32_t push_limit) {
    SplitDeferral run;
    SplitLayoutScope layout(*this, split);
    PipelineCacheTestAccess::BuildHold hold;
    AsyncDrawRig rig(*this, false, false, false, false, true);
    auto &scheduler = rig.Scheduler();
    CommandSchedulerTestAccess::AllowThisThread(scheduler);
    auto &graphics = rig.context->GetGraphics();
    const auto device_limit = graphics.max_push_descriptors;
    graphics.max_push_descriptors = push_limit;
    rig.PaintSentinels();
    PipelineCacheTestAccess::HoldBuildQueue(rig.Pipelines(), hold);
    const auto replaced = UseAddressStage(rig, stage, SplitDeferralReadOffset);
    Common::FirstRunReport::Global().Reset();
    Require("SplitLayoutDeferral", "draw", DrawStreamAuto(rig, 3, 1), "the draw reported failure");
    RestoreAddressStage(rig, stage, replaced);
    run.holes = scheduler.PendingHoleCount();
    run.ineligible = DeferredIneligible();
    std::optional<GraphicsPipelineKey> key;
    if (run.holes == 1) key = CommandSchedulerTestAccess::HeldHoleKey(scheduler, 1);
    if (key) run.predicted_push = PipelineCacheTestAccess::PredictedPush(rig.Pipelines(), *key);
    hold.release = true;
    scheduler.Finish();
    if (key) run.built_push = PipelineCacheTestAccess::BuiltPush(rig.Pipelines(), *key);
    run.color = rig.ReadColor();
    graphics.max_push_descriptors = device_limit;
    return run;
  }

  // Without --deferred-non-push-draws a draw is deferred only if it allocates no descriptor set, and
  // the layout decides that. Under the split layout a draw with VS bindings is blocked and counted as
  // non_push, where the single set defers the same draw, which is the baseline. A pixel set that
  // fits is pushed and defers under both layouts, and one that does not is blocked under both.
  void CheckSplitDeferralPass(bool split) {
    constexpr const char *name = "SplitLayoutDeferral";
    const auto vertex = RunSplitDeferral(split, AddressStage::Vertex, SplitPushLimit);
    if (split) {
      Require(name, "split blocks vertex bindings",
              vertex.holes == 0 && vertex.ineligible == IneligibleCounts{{"non_push", 1}} &&
                  AllDrawn(vertex.color),
              "a draw whose VS needs a descriptor set was deferred, or not counted as non_push");
    } else {
      Require(name, "single set defers vertex bindings",
              vertex.holes == 1 && vertex.ineligible.empty() && vertex.predicted_push &&
                  vertex.built_push && AllDrawn(vertex.color),
              "the baseline draw with vertex bindings was not deferred as a push draw");
    }
    const auto pixel_push = RunSplitDeferral(split, AddressStage::Pixel, SplitPushLimit);
    Require(name, split ? "split defers pushed pixel set" : "single set defers pushed pixel set",
            pixel_push.holes == 1 && pixel_push.ineligible.empty() && pixel_push.predicted_push &&
                pixel_push.built_push && AllDrawn(pixel_push.color),
            "a draw whose PS set is pushed was not deferred as a push draw");
    const auto pixel_set = RunSplitDeferral(split, AddressStage::Pixel, SplitNoPushLimit);
    Require(name, split ? "split blocks pixel set" : "single set blocks pixel set",
            pixel_set.holes == 0 && pixel_set.ineligible == IneligibleCounts{{"non_push", 1}} &&
                AllDrawn(pixel_set.color),
            "a draw whose PS set is allocated was deferred, or not counted as non_push");
  }

  // Runs the deferral check in the single set and in the split layout.
  void CheckSplitLayoutDeferral() {
    CheckSplitDeferralPass(false);
    CheckSplitDeferralPass(true);
    std::printf("[gpu]     %-32s ok\n", "SplitLayoutDeferral");
  }

  // The recording half of a draw into a side buffer, as CheckDrawRecordIntoSideBuffer does it, for
  // the split layout. The stage that reads by address decides which set the writes go to, and the
  // limit decides whether a pixel set is pushed. The vertex set is never pushed.
  void CheckSplitDrawRecordIntoSideBuffer(AddressStage stage, bool push_descriptors) {
    constexpr const char *name = "SplitDrawRecord";
    const bool vertex = stage == AddressStage::Vertex;
    CommandStream::Scope hooks;
    SplitLayoutScope layout(*this, true);
    AsyncDrawRig rig(*this, false);
    auto &graphics = rig.context->GetGraphics();
    const auto device_limit = graphics.max_push_descriptors;
    graphics.max_push_descriptors = push_descriptors ? SplitPushLimit : SplitNoPushLimit;
    UseAddressStage(rig, stage, SplitRecordReadOffset);
    auto &executor = rig.context->GetRenderExecutor();
    DrawView view;
    const auto *pipeline = RenderExecutorTestAccess::PrepareAutoDraw(
        executor, rig.Scheduler().Current(), {.vertex_count = 3, .instance_count = 1},
        vk::PrimitiveTopology::eTriangleList, view);
    Require(name, "prepared",
            pipeline != nullptr && pipeline->uses_split_sets &&
                pipeline->uses_push_descriptors == (!vertex && push_descriptors),
            "the draw was not prepared with the split layout and the descriptor path under test");
    rig.ResolveTargets();
    for (uint32_t index = 0; index < view.stage_count; ++index) {
      view.bindings[index]->runtime = nullptr;
    }
    SideBuffer side(graphics);
    PlainDrawEnv env;
    CommandStream::Recorder::Reset(true);
    RenderExecutorTestAccess::RecordDraw(executor, side.buffer, view, *pipeline, env);
    CommandStream::Recorder::active = false;
    const auto rows = CommandStream::Recorder::rows;
    const auto &written = vertex ? env.scratch.writes : env.scratch.pixel_writes;
    const auto &other = vertex ? env.scratch.pixel_writes : env.scratch.writes;
    Require(name, "writes of the stage's own set", written.size() == 2 && other.empty(),
            "the descriptors were not written for the set of the stage that reads");
    const bool pushed = !vertex && push_descriptors;
    Require(name, "descriptor path",
            HasRow(rows, pushed ? "vkCmdPushDescriptorSet(" : "vkCmdBindDescriptorSets(") &&
                !HasRow(rows, pushed ? "vkCmdBindDescriptorSets(" : "vkCmdPushDescriptorSet("),
            "the descriptors took the other path");
    Require(name, "descriptor set number",
            HasRow(rows, pushed ? "set 2; " : vertex ? "first 0 set#" : "first 2 set#"),
            "the descriptors went to another set than the stage's own");
    RenderExecutorTestAccess::ResetBindings(executor);
    graphics.max_push_descriptors = device_limit;
    std::printf("[gpu]     %-32s ok (%s, %s)\n", name, vertex ? "vertex" : "pixel",
                push_descriptors ? "push allowed" : "no push");
  }

  // The tests of draws that already exist for the single set, run with the library mode as given.
  // The address vertex shader gives the VS set and the address pixel shader the PS set. With the
  // mode off these are the single set baselines, and with it on they use the split layout.
  void CheckSplitDrawsPass(bool split) {
    SplitLayoutScope layout(*this, split);
    for (const auto stage : {AddressStage::Vertex, AddressStage::Pixel}) {
      CheckHoleRecordedStreamNonPush(stage);
      CheckDeferredNonPush(stage);
      CheckDeferredNonPushPoolRetire(stage);
    }
    if (!split) return;
    CheckSplitDrawRecordIntoSideBuffer(AddressStage::Vertex, true);
    CheckSplitDrawRecordIntoSideBuffer(AddressStage::Vertex, false);
    CheckSplitDrawRecordIntoSideBuffer(AddressStage::Pixel, true);
    CheckSplitDrawRecordIntoSideBuffer(AddressStage::Pixel, false);
  }

  // Runs the tests of existing draws and holes with the split layout on.
  void CheckSplitLayoutDraws() {
    CheckSplitDrawsPass(true);
  }

  // What the validation layer said during one run. That is how many messages, how many hazards, and
  // the kinds of message, by the VUID each one names.
  struct ValidationOutcome {
    uint64_t count = 0;
    uint64_t hazards = 0;
    std::set<std::string> kinds;
  };

  // The kind of a message is the VUID it ends with, or its start if it names none.
  static std::string ValidationKind(const std::string &message) {
    constexpr size_t kind_prefix_length = 80;
    const auto at = message.find("#VUID-");
    if (at == std::string::npos) return message.substr(0, kind_prefix_length);
    return message.substr(at + 1, message.find(')', at) - at - 1);
  }

  // Runs `check` with the message count reset and returns what the layer said meanwhile.
  ValidationOutcome MeasureValidation(const std::function<void()> &check) {
    namespace Validation = Libs::Graphics::VulkanValidation;
    Validation::ResetMessages();
    check();
    ValidationOutcome outcome;
    outcome.count = Validation::MessageCount();
    outcome.hazards = Validation::HazardCount();
    for (const auto &message : Validation::Messages()) outcome.kinds.insert(ValidationKind(message));
    return outcome;
  }

  // Runs one scenario in both layouts under validation. This device has features off that the
  // shaders of the rig use, so the single set run is not clean, and its messages are the baseline.
  // The split run may not report a kind of message the single set run does not, nor a hazard. The
  // counts are printed, but not compared, since the two runs do not draw the same number of times.
  void CheckSplitScenarioUnderValidation(const char *scenario,
                                         const std::function<void(bool)> &run) {
    constexpr const char *name = "SplitLayoutValidation";
    const auto off = MeasureValidation([&] { run(false); });
    const auto on = MeasureValidation([&] { run(true); });
    std::printf("[gpu]     %-32s %s: single set %llu messages, split %llu messages, %llu hazards\n",
                name, scenario, static_cast<unsigned long long>(off.count),
                static_cast<unsigned long long>(on.count),
                static_cast<unsigned long long>(on.hazards));
    for (const auto &kind : on.kinds) {
      std::printf("[gpu]     %-32s %s: split run message kind %s\n", name, scenario, kind.c_str());
    }
    for (const auto &kind : on.kinds) {
      Require(name, scenario, off.kinds.contains(kind),
              ("the split layout reported a message the single set does not: " + kind).c_str());
    }
    Require(name, scenario, on.hazards == 0 && off.hazards == 0,
            "the layer reported a synchronization hazard");
  }

  // A pipeline destroyed and then the cache that owns its set layouts must make the layer say
  // nothing. Unlike the scenarios, this code makes no shader module or draw, so there is no noise
  // for a baseline to excuse, and a double destroy or a destroy of a layout in use is a message.
  void CheckCacheOwnedLayoutsUnderValidation() {
    constexpr const char *name = "SplitLayoutValidation";
    const auto owned = MeasureValidation([&] { CheckPipelineKeepsSharedLayouts(); });
    std::printf("[gpu]     %-32s cache-owned layouts: %llu messages\n", name,
                static_cast<unsigned long long>(owned.count));
    Require(name, "cache-owned layouts", owned.count == 0,
            "destroying a pipeline and then its cache made the validation layer report a message");
  }

  // The split layout scenarios under validation, skipped when the layer is not on or the device
  // cannot have the library. The control proves the layer reports a hazard at all.
  void CheckSplitLayoutUnderValidation() {
    constexpr const char *name = "SplitLayoutValidation";
    if (!SyncValidationActive() || !m_library_available) {
      std::printf("[gpu]     %-32s skipped: %s\n", name,
                  !m_library_available ? "the device has no graphics pipeline library"
                                       : SyncValidationReason().c_str());
      return;
    }
    CheckSyncValidationSeesHazard();
    CheckSplitScenarioUnderValidation("pixels", [&](bool split) {
      for (const uint32_t limit : {SplitPushLimit, SplitNoPushLimit}) {
        (void)CheckSplitPixelPass(split, limit);
      }
    });
    CheckSplitScenarioUnderValidation("modules", [&](bool split) { (void)CountSplitModules(split); });
    CheckSplitScenarioUnderValidation("prediction", [&](bool split) { CheckSplitPredictionPass(split); });
    CheckSplitScenarioUnderValidation("deferral", [&](bool split) { CheckSplitDeferralPass(split); });
    CheckSplitScenarioUnderValidation("draws", [&](bool split) { CheckSplitDrawsPass(split); });
    CheckSplitScenarioUnderValidation("heap", [&](bool split) { CheckSplitHeapPass(split); });
    CheckCacheOwnedLayoutsUnderValidation();
    CheckSplitScenarioUnderValidation("sharing", [&](bool split) { CheckSharedLayouts(split); });
    CheckSplitScenarioUnderValidation("teardown", [&](bool split) { CheckSharedLayoutTeardown(split); });
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // The limits on the contents of a set count the whole pipeline layout, so the VS set and the PS set
  // are judged together. Here each set is under the limit and both together are over it. The two
  // sets that fit exactly are the baseline, which the check must accept.
  void CheckSplitLayoutLimits() {
    constexpr const char *name = "SplitLayoutLimits";
    constexpr uint32_t set_limit = 10, other_limit = 100, half_over = 6, half_fits = 5;
    vk::PhysicalDeviceLimits limits{};
    limits.maxDescriptorSetStorageBuffers = set_limit;
    limits.maxDescriptorSetSamplers = other_limit;
    limits.maxPerStageDescriptorStorageBuffers = other_limit;
    limits.maxPerStageDescriptorSamplers = other_limit;
    limits.maxPerStageResources = other_limit;
    const auto bindings = [](vk::ShaderStageFlagBits stage, uint32_t count) {
      return std::vector<vk::DescriptorSetLayoutBinding>{
          {0, vk::DescriptorType::eStorageBuffer, count, stage, nullptr}};
    };
    uint64_t count = 0;
    const auto vertex = bindings(vk::ShaderStageFlagBits::eVertex, half_over);
    const auto pixel = bindings(vk::ShaderStageFlagBits::eFragment, half_over);
    Require(name, "each set fits",
            DescriptorLimitError(limits, vertex, count).empty() &&
                DescriptorLimitError(limits, pixel, count).empty(),
            "a set under the limit was refused");
    Require(name, "both sets together are over",
            SplitLayoutLimitError(limits, vertex, pixel).find("storage-buffer descriptors exceed device limit: 12/10") !=
                std::string::npos,
            "the VS set and the PS set together over the limit were accepted");
    Require(name, "baseline fits",
            SplitLayoutLimitError(limits, bindings(vk::ShaderStageFlagBits::eVertex, half_fits),
                                  bindings(vk::ShaderStageFlagBits::eFragment, half_fits))
                .empty(),
            "two sets that exactly fit the limit were refused");
    std::printf("[host]    %-32s ok\n", name);
  }

  // Where the shaders of the sharing checks read. The vertex shader has bindings, and the two pixel
  // shaders are different programs with the same bindings.
  static constexpr uint64_t SplitSharedVertexOffset = SplitVertexReadOffset + 0x400;
  static constexpr uint64_t SplitSharedPixelOffsetA = SplitVertexReadOffset + 0x500;
  static constexpr uint64_t SplitSharedPixelOffsetB = SplitVertexReadOffset + 0x600;
  // The layouts of the three pipelines of the sharing check. Two share the vertex shader, and two
  // share the pixel shader. A cache that shared everything would fail the third, whose vertex shader
  // has no bindings.
  static constexpr size_t SplitSharedDistinctLayouts = 3;

  // What a pipeline of the sharing checks says about its set layouts.
  struct SharedLayoutPipeline {
    GraphicsPipelineKey key{};
    bool split = false;
    vk::DescriptorSetLayout vertex = nullptr;
    vk::DescriptorSetLayout pixel = nullptr;
    vk::DescriptorSetLayout single = nullptr;
  };

  static SharedLayoutPipeline LayoutsOfKey(const PipelineCache &cache, const GraphicsPipelineKey &key) {
    const auto &pipeline = PipelineCacheTestAccess::PipelineOf(cache, key);
    return {key, pipeline.uses_split_sets, pipeline.vertex_set_layout, pipeline.pixel_set_layout,
            pipeline.descriptor_set_layout};
  }

  // The layouts of the one pipeline that is not in `seen`, which is added to it.
  static SharedLayoutPipeline NewestLayouts(const PipelineCache &cache,
                                            std::vector<GraphicsPipelineKey> &seen) {
    for (const auto &key : PipelineCacheTestAccess::GraphicsKeys(cache)) {
      if (std::ranges::find(seen, key) == seen.end()) {
        seen.push_back(key);
        return LayoutsOfKey(cache, key);
      }
    }
    Require("SplitLayoutSharing", "new pipeline", false, "the draw did not build a pipeline");
    return {};
  }

  // Draws once with the given vertex shader and a pixel shader that reads at `pixel_offset`.
  // `plain_vertex` is the rig's own vertex shader, which has no bindings.
  SharedLayoutPipeline DrawSharedLayoutPipeline(AsyncDrawRig &rig, uint64_t plain_vertex,
                                                bool vertex_bindings, uint64_t pixel_offset,
                                                std::vector<GraphicsPipelineKey> &seen) {
    if (vertex_bindings) {
      UseAddressVertex(rig, SplitSharedVertexOffset);
    } else {
      rig.shaders.SetEsShaderBase(plain_vertex);
    }
    UseAddressPixel(rig, pixel_offset);
    Require("SplitLayoutSharing", "draw", rig.Draw(), "the draw reported failure");
    return NewestLayouts(rig.Pipelines(), seen);
  }

  // What the three draws of the sharing check left behind.
  struct SharedLayoutRun {
    SharedLayoutPipeline first, same_vertex, plain_vertex;
    size_t batches_start = 0, batches_first = 0, batches_second = 0;
    size_t cached = 0;
  };

  // Draws three pipelines. The first has a vertex shader with bindings, the second has the same
  // vertex shader and another pixel shader, and the third has a vertex shader without bindings and
  // the first one's pixel shader. The heap is read after the first and the second.
  SharedLayoutRun RunSharedLayoutDraws(bool split, uint32_t push_limit) {
    SharedLayoutRun run;
    SplitLayoutScope layout(*this, split);
    AsyncDrawRig rig(*this, false);
    auto &graphics = rig.context->GetGraphics();
    const auto device_limit = graphics.max_push_descriptors;
    graphics.max_push_descriptors = push_limit;
    const auto plain = rig.shaders.GetVs().es_regs.data_addr;
    auto &heap = rig.context->GetDescriptorHeap();
    rig.PaintSentinels();
    std::vector<GraphicsPipelineKey> seen;
    run.batches_start = DescriptorHeapTestAccess::Batches(heap);
    run.first = DrawSharedLayoutPipeline(rig, plain, true, SplitSharedPixelOffsetA, seen);
    run.batches_first = DescriptorHeapTestAccess::Batches(heap);
    run.same_vertex = DrawSharedLayoutPipeline(rig, plain, true, SplitSharedPixelOffsetB, seen);
    run.batches_second = DescriptorHeapTestAccess::Batches(heap);
    run.plain_vertex = DrawSharedLayoutPipeline(rig, plain, false, SplitSharedPixelOffsetA, seen);
    auto &sets = PipelineCacheTestAccess::SetLayouts(rig.Pipelines());
    run.cached = sets.Size();
    Require("SplitLayoutSharing", "picture", AllDrawn(rig.ReadColor()),
            "the draws left another picture");
    graphics.max_push_descriptors = device_limit;
    return run;
  }

  // The layouts of three split pipelines are shared where their stages agree. The pipeline whose
  // vertex shader has no bindings is the baseline, since a cache that handed out one layout, or a
  // pipeline that kept its own, would not show the difference between it and the other two.
  void CheckSharedSplitLayouts(const SharedLayoutRun &run) {
    constexpr const char *name = "SplitLayoutSharing";
    Require(name, "split layouts",
            run.first.split && run.same_vertex.split && run.plain_vertex.split &&
                run.first.vertex != nullptr && run.first.pixel != nullptr &&
                run.plain_vertex.vertex != nullptr && run.first.single == nullptr,
            "the pipelines did not get the split layout");
    Require(name, "same vertex shader, same set 0 layout",
            run.first.vertex == run.same_vertex.vertex, "two pipelines with one vertex shader did not share set 0");
    Require(name, "same bindings, same set 2 layout",
            run.first.pixel == run.same_vertex.pixel && run.first.pixel == run.plain_vertex.pixel,
            "pipelines with pixel shaders of the same bindings did not share set 2");
    Require(name, "other vertex bindings, other set 0 layout",
            run.first.vertex != run.plain_vertex.vertex,
            "a vertex shader without bindings got the layout of one with bindings");
    Require(name, "one create per distinct layout",
            run.cached == SplitSharedDistinctLayouts,
            "the cache made a layout more than once");
  }

  // Without the library mode the pipelines keep the layout of their own, and the cache is not used.
  // That is the baseline of the sharing check, and the status quo that the off path must keep.
  void CheckUnsharedSingleLayouts(const SharedLayoutRun &run) {
    constexpr const char *name = "SplitLayoutSharing";
    Require(name, "single set layouts",
            !run.first.split && !run.same_vertex.split && !run.plain_vertex.split &&
                run.first.single != nullptr && run.same_vertex.single != nullptr &&
                run.first.vertex == nullptr && run.first.pixel == nullptr,
            "a pipeline got the split layout with the mode off");
    Require(name, "each pipeline owns its layout",
            run.first.single != run.same_vertex.single && run.first.single != run.plain_vertex.single,
            "two single set pipelines shared a layout");
    Require(name, "cache unused", run.cached == 0,
            "the shared layout cache was used with the mode off");
  }

  // The heap holds a batch per layout handle. Shared layouts give the second pipeline nothing new,
  // and the baseline of separate single set layouts gives it one more. A pushed set takes no set from
  // the heap, so this needs a limit that pushes nothing.
  void CheckSharedLayoutBatches(bool split, const SharedLayoutRun &run) {
    Require("SplitLayoutSharing", "first pipeline took batches", run.batches_first > run.batches_start,
            "the first draw took no descriptor set from the heap");
    if (split) {
      Require("SplitLayoutSharing", "second pipeline shares the batches",
              run.batches_second == run.batches_first,
              "a pipeline with shared layouts took batches of its own");
    } else {
      Require("SplitLayoutSharing", "baseline: separate layouts take more batches",
              run.batches_second > run.batches_first,
              "single set pipelines did not take a batch each");
    }
  }

  // One pass of the sharing check in one layout mode and push limit.
  void CheckSharedLayoutsPass(bool split, uint32_t push_limit) {
    const auto run = RunSharedLayoutDraws(split, push_limit);
    if (split) {
      CheckSharedSplitLayouts(run);
    } else {
      CheckUnsharedSingleLayouts(run);
    }
    if (push_limit == SplitNoPushLimit) CheckSharedLayoutBatches(split, run);
  }

  // Runs the sharing check with the layouts pushed and not.
  void CheckSharedLayouts(bool split) {
    for (const uint32_t limit : {SplitPushLimit, SplitNoPushLimit}) CheckSharedLayoutsPass(split, limit);
  }

  // Builds the pipeline of the pixel shader that reads at `pixel_offset`, with the vertex shader that
  // has bindings. No draw is recorded, so the pipeline may be destroyed at once.
  HoleSource PrepareSharedLayoutSource(AsyncDrawRig &rig, uint64_t pixel_offset) {
    UseAddressVertex(rig, SplitSharedVertexOffset);
    UseAddressPixel(rig, pixel_offset);
    return PrepareHoleSource(rig);
  }

  // A pipeline destroyed before the cache leaves the layouts it shared alive for the pipelines that
  // still use them and for the one built next. The set layouts a pipeline destroys are counted, and a
  // split pipeline must destroy none while a single set pipeline destroys its one, which is the
  // baseline. A layout destroyed with the pipeline anyway is also a dead handle in the second
  // pipeline and in the layout made from it, which the validation layer reports. The cache
  // destroys each layout once, when the rig destroys the context after this.
  void CheckSharedLayoutTeardown(bool split) {
    constexpr const char *name = "SplitLayoutTeardown";
    SplitLayoutScope layout(*this, split);
    AsyncDrawRig rig(*this, false);
    auto &cache = rig.Pipelines();
    auto &sets = PipelineCacheTestAccess::SetLayouts(cache);
    const auto first = LayoutsOfKey(cache, PrepareSharedLayoutSource(rig, SplitSharedPixelOffsetA).key);
    const auto second_key = PrepareSharedLayoutSource(rig, SplitSharedPixelOffsetB).key;
    const auto cached = sets.Size();
    const auto destroys = PipelineSetLayoutDestroyCount();
    PipelineCacheTestAccess::DestroyGraphicsPipeline(cache, first.key);
    Require(name, "set layouts destroyed by the pipeline", PipelineSetLayoutDestroyCount() - destroys == (split ? 0u : 1u),
            "destroying a pipeline destroyed another number of set layouts than the ones it owns");
    Require(name, "layouts kept", sets.Size() == cached && cached == (split ? 2u : 0u),
            "destroying a pipeline changed the layouts of the cache");
    const auto second = LayoutsOfKey(cache, second_key);
    if (split) {
      const auto set_layouts = SplitSetLayouts(second.vertex, rig.context->GetGraphics().image_heap_layout,
                                               second.pixel);
      vk::PipelineLayoutCreateInfo info{};
      info.flags = vk::PipelineLayoutCreateFlagBits::eIndependentSetsEXT;
      info.setLayoutCount = SPLIT_SET_COUNT;
      info.pSetLayouts = set_layouts.data();
      vk::PipelineLayout probe = nullptr;
      auto &device = rig.context->GetGraphics().device;
      Require(name, "layout from the shared sets", device.createPipelineLayout(&info, nullptr, &probe) == vk::Result::eSuccess,
              "a pipeline layout could not be made from the shared set layouts");
      device.destroyPipelineLayout(probe, nullptr);
    }
    const auto rebuilt = LayoutsOfKey(cache, PrepareSharedLayoutSource(rig, SplitSharedPixelOffsetA).key);
    Require(name, "rebuilt pipeline gets the same layouts",
            !split || (rebuilt.vertex == first.vertex && rebuilt.pixel == first.pixel &&
                       sets.Size() == cached),
            "the pipeline built after the first was destroyed did not get the shared layouts back");
    std::printf("[gpu]     %-32s ok (%s)\n", name, split ? "split" : "single set");
  }

  // Destroying a split pipeline destroys its pipeline layout and not the set layouts, which belong
  // to the cache. The device is called directly by DestroyPipelineObjects and by the cache, so a set
  // layout destroyed by both is a double destroy that only the validation layer can see. This check
  // is therefore run under the layer by CheckSplitLayoutUnderValidation, which requires no messages.
  // Without the layer the destroy count and the cache destroying each layout once still show it.
  void CheckPipelineKeepsSharedLayouts() {
    constexpr const char *name = "SplitLayoutTeardown";
    std::vector<vk::DescriptorSetLayout> created, destroyed;
    {
      DescriptorSetLayoutCache cache(
          [&](const vk::DescriptorSetLayoutCreateInfo &info) {
            vk::DescriptorSetLayout made = nullptr;
            Require(name, "make layout", m_device.createDescriptorSetLayout(&info, nullptr, &made) == vk::Result::eSuccess,
                    "a set layout could not be made");
            created.push_back(made);
            return made;
          },
          [&](vk::DescriptorSetLayout layout) {
            destroyed.push_back(layout);
            m_device.destroyDescriptorSetLayout(layout, nullptr);
          });
      const std::array bindings{
          vk::DescriptorSetLayoutBinding{0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr}};
      PipelineCache::Pipeline pipeline;
      pipeline.uses_split_sets = true;
      pipeline.vertex_set_layout = cache.GetOrCreate(bindings, {});
      pipeline.pixel_set_layout = cache.GetOrCreate({}, {});
      const std::array set_layouts{pipeline.vertex_set_layout, pipeline.pixel_set_layout};
      vk::PipelineLayoutCreateInfo info{};
      info.setLayoutCount = static_cast<uint32_t>(set_layouts.size());
      info.pSetLayouts = set_layouts.data();
      Require(name, "make pipeline layout",
              m_device.createPipelineLayout(&info, nullptr, &pipeline.pipeline_layout) == vk::Result::eSuccess,
              "a pipeline layout could not be made");
      // The count is the tripwire that needs no validation layer. The baseline is a single set
      // pipeline, whose one layout is its own and is destroyed with it.
      const auto destroys_before = PipelineSetLayoutDestroyCount();
      DestroyPipelineObjects(m_device, pipeline);
      Require(name, "split pipeline destroyed no set layout",
              PipelineSetLayoutDestroyCount() == destroys_before,
              "destroying a split pipeline destroyed a set layout");
      PipelineCache::Pipeline single;
      const vk::DescriptorSetLayoutCreateInfo single_info{};
      Require(name, "make own layout",
              m_device.createDescriptorSetLayout(&single_info, nullptr, &single.descriptor_set_layout) ==
                  vk::Result::eSuccess,
              "a set layout could not be made");
      DestroyPipelineObjects(m_device, single);
      Require(name, "single set pipeline destroyed its layout",
              PipelineSetLayoutDestroyCount() == destroys_before + 1,
              "the destroy count did not count the layout of a single set pipeline");
      Require(name, "cache not asked to destroy early", created.size() == 2 && destroyed.empty(),
              "the cache destroyed a set layout while it was alive");
    }
    Require(name, "cache destroyed each once",
            destroyed.size() == created.size() &&
                std::set<vk::DescriptorSetLayout>(destroyed.begin(), destroyed.end()) ==
                    std::set<vk::DescriptorSetLayout>(created.begin(), created.end()),
            "the cache did not destroy every layout it made exactly once");
    std::printf("[gpu]     %-32s ok (cache owns the layouts)\n", name);
  }

  // All the tests of the shared set layouts, which need a device with the library for the split half.
  void CheckSplitLayoutSharing() {
    CheckPipelineKeepsSharedLayouts();
    for (const bool split : {false, true}) {
      CheckSharedLayouts(split);
      CheckSharedLayoutTeardown(split);
    }
    std::printf("[gpu]     %-32s ok\n", "SplitLayoutSharing");
  }

  // All the tests of the split layout. The device part is skipped, with the reason, when the device
  // has no graphics pipeline library.
  void CheckSplitLayout() {
    constexpr const char *name = "SplitLayout";
    CheckSplitLayoutPatch();
    CheckSplitLayoutLimits();
    CheckSplitLayoutModules();
    CheckSplitLayoutDevice();
    if (!m_library_available) {
      std::printf("[gpu]     %-32s skipped: the device has no graphics pipeline library\n", name);
      return;
    }
    CheckSplitLayoutPixels();
    CheckSplitLayoutPrediction();
    CheckSplitLayoutDeferral();
    CheckSplitLayoutDraws();
    CheckSplitLayoutStore();
    CheckSplitLayoutHeap();
    CheckSplitLayoutSharing();
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // What a session over a shader store sees of the split layout.
  struct SplitStoreSession {
    size_t pixel_modules = 0;
    size_t pixel_with_library = 0;
    size_t other_with_library = 0;
    std::array<size_t, 2> pipelines{};
  };

  // Opens a render context over the store, with the library mode as given. With `draw` it compiles
  // the rig-like vertex and pixel program and builds the pipeline, which writes the store. Without,
  // it only looks at what loading the store made, which are the modules and the warmed pipelines.
  SplitStoreSession RunSplitStoreSession(const std::filesystem::path &store, bool library_mode,
                                         bool draw, const std::vector<u32> &vertex,
                                         const std::vector<u32> &pixel) {
    SplitStoreSession session;
    SplitLayoutScope layout(*this, library_mode);
    RenderContext context(RuntimeContext(), store);
    auto &pipelines = context.GetPipelineCache();
    if (draw) {
      context.InitializeGpu(nullptr);
      LibKernel::Memory::InstallGpuResources(&context);
      context.GetGpu().SendCommandSync([&] {
        CommandProcessor processor(context, 0);
        processor.BufferInit();
        auto &registers = processor.GetCtx();
        auto &shaders = processor.GetShCtx();
        registers.SetRenderTargetMask(0xf);
        registers.SetShaderMask(0xf);
        registers.SetTargetOutputMode(0, 4);
        registers.SetViewportTransformControl(0x300);
        processor.GetUcfg().SetPrimitiveType(Prospero::PrimitiveType::kTriList);
        shaders.SetEsShaderBase(reinterpret_cast<uint64_t>(vertex.data()));
        shaders.SetPsShaderBase(reinterpret_cast<uint64_t>(pixel.data()));
        auto &scheduler = context.GetCommandScheduler();
        DrawRenderState state{.topology = vk::PrimitiveTopology::eTriangleList};
        pipelines.GetGraphicsPrograms(scheduler.Current(), state);
        (void)pipelines.GetGraphicsPipeline(scheduler.Current(), state);
        scheduler.Finish();
      });
      context.ShutdownGpu();
      LibKernel::Memory::InstallGpuResources(nullptr);
    }
    for (const auto &module : PipelineCacheTestAccess::Modules(pipelines)) {
      const bool is_pixel = module.stage == ShaderType::Pixel;
      session.pixel_modules += is_pixel ? 1u : 0u;
      session.pixel_with_library += is_pixel && module.gpl_module != nullptr ? 1u : 0u;
      session.other_with_library += !is_pixel && module.gpl_module != nullptr ? 1u : 0u;
    }
    session.pipelines = PipelineCacheTestAccess::SplitPipelineCounts(pipelines);
    return session;
  }

  // The modules that loading a shader store makes get their library handle too, and the pipelines
  // warmed from the store get the split layout. The store is written by a session with the library
  // mode on and read back three ways. The sessions with the mode on see handles and split pipelines,
  // and a session with it off, over the same store, sees neither, which is the baseline. The store
  // itself is the one a session without the mode writes, since neither ids nor the format change.
  void CheckSplitLayoutStore() {
    constexpr const char *name = "SplitLayoutStore";
    // Static, since the shader map keeps the address of the code and sessions read it later.
    static const std::vector<u32> vertex = [] {
      std::vector<u32> code;
      AppendVMovU32(&code, 0, 0);
      AppendVMovLiteral(&code, 1, 0x3f800000u);
      code.insert(code.end(), {EncodeExp0(0x0c, 0xf), EncodeExp1(0, 0, 0, 1)});
      AppendEnd(&code);
      return code;
    }();
    static const std::vector<u32> pixel = [] {
      std::vector<u32> code;
      AppendVMovLiteral(&code, 0, 0x3f800000u);
      code.insert(code.end(), {EncodeExp0(0x00, 0xf), EncodeExp1(0, 0, 0, 0)});
      AppendEnd(&code);
      return code;
    }();
    static ShaderUserData vertex_user_data{};
    ShaderMapUserData(reinterpret_cast<uint64_t>(vertex.data()),
                      {.type = Prospero::ShaderBinaryType::kGs, .user_data = &vertex_user_data,
                       .code_size_bytes = static_cast<u32>(vertex.size() * sizeof(u32))});
    ShaderMapUserData(reinterpret_cast<uint64_t>(pixel.data()),
                      {.type = Prospero::ShaderBinaryType::kPs,
                       .code_size_bytes = static_cast<u32>(pixel.size() * sizeof(u32))});
    const auto store = std::filesystem::temp_directory_path() /
                       fmt::format("kyty_split_store_{}",
                                   std::chrono::steady_clock::now().time_since_epoch().count()) /
                       "TEST00001.shaders";
    const auto written = RunSplitStoreSession(store, true, true, vertex, pixel);
    Require(name, "written", written.pixel_modules == 1 && written.pixel_with_library == 1 &&
                written.other_with_library == 0 && written.pipelines[0] == 1,
            "the session that compiled the program did not make the library handle and the split layout");
    const auto loaded = RunSplitStoreSession(store, true, false, vertex, pixel);
    Require(name, "loaded with the mode on",
            loaded.pixel_modules == 1 && loaded.pixel_with_library == 1 &&
                loaded.other_with_library == 0 && loaded.pipelines[0] == 1 && loaded.pipelines[1] == 0,
            "loading the store did not give the pixel module its library handle or the pipeline the split layout");
    const auto baseline = RunSplitStoreSession(store, false, false, vertex, pixel);
    Require(name, "loaded with the mode off",
            baseline.pixel_modules == 1 && baseline.pixel_with_library == 0 &&
                baseline.pipelines[0] == 0 && baseline.pipelines[1] == 1,
            "loading the store with the mode off made a library handle or a split pipeline");
    std::error_code ignored;
    std::filesystem::remove_all(store.parent_path(), ignored);
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A draw whose pipeline uses the image heap, in the library mode as given. The modules are marked
  // as heap users after a first draw has made them, so the next pipeline is built as a heap user, the
  // way DeferredEligibility does it. The heap set is allocated from the layout the context holds and
  // is bound at set 1 with the pipeline's own layout, so the layer checks that the layout has that
  // same set layout at set 1. The picture is the plain gray in both modes.
  void CheckSplitHeapPass(bool split) {
    constexpr const char *name = "SplitLayoutHeap";
    CommandStream::Scope hooks;
    SplitLayoutScope layout(*this, split);
    AsyncDrawRig rig(*this, false);
    rig.PaintSentinels();
    Require(name, "first draw", rig.Draw(), "the draw reported failure");
    rig.SetSecondPipeline(true);
    PipelineCacheTestAccess::MarkModulesUseImageHeap(rig.Pipelines());
    CommandStream::Recorder::Reset(true);
    Require(name, "heap draw", rig.Draw(), "the draw reported failure");
    const auto rows = CommandStream::Recorder::rows;
    CommandStream::Recorder::active = false;
    Require(name, "picture", AllDrawn(rig.ReadColor()), "the heap draw left another picture");
    Require(name, "heap pipeline has the layout under test",
            PipelineCacheTestAccess::HeapPipelines(rig.Pipelines(), split) == 1,
            "the pipeline that uses the image heap does not have the layout under test");
    Require(name, "heap set at 1", HasRow(rows, "first 1 set#"),
            "the image heap set was not bound at set 1");
  }

  // Runs the heap check in the single set and in the split layout.
  void CheckSplitLayoutHeap() {
    CheckSplitHeapPass(false);
    CheckSplitHeapPass(true);
    std::printf("[gpu]     %-32s ok\n", "SplitLayoutHeap");
  }
