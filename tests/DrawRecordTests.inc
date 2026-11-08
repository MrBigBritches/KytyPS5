  // The commands a draw records, pinned so that splitting a draw into its eager and recording
  // halves cannot change them. Included inside VulkanHarness, after AsyncDrawRig.

  static constexpr uint64_t DrawStreamIndexOffset = 0x60000;
  static constexpr uint32_t DrawStreamIndexCount = 8;
  static constexpr uint64_t DrawStreamVertexOffset = 0x70000;
  static constexpr uint64_t DrawStreamReadOffset = 0x80000;
  static constexpr uint32_t DrawStreamPushDescriptorLimit = 32;

  // Index values for the indexed scenes, as 16 and as 32 bits, one after the other.
  void WriteDrawStreamIndices() {
    auto *indices16 = reinterpret_cast<uint16_t *>(AsyncBase + DrawStreamIndexOffset);
    auto *indices32 = reinterpret_cast<uint32_t *>(AsyncBase + DrawStreamIndexOffset + 0x100);
    for (uint32_t i = 0; i < DrawStreamIndexCount; ++i) {
      indices16[i] = static_cast<uint16_t>(i);
      indices32[i] = i;
    }
  }

  // One recorded scene. Each draws once or twice through the real draw entry points.
  struct DrawStreamScene {
    const char *label;
    std::function<void(AsyncDrawRig &)> run;
  };

  static bool DrawStreamAuto(AsyncDrawRig &rig, uint32_t count, uint32_t instances,
                             uint32_t first_vertex = 0) {
    return rig.Draw({.vertex_count = count,
                     .instance_count = instances,
                     .first_vertex = first_vertex});
  }

  static bool DrawStreamIndexed(AsyncDrawRig &rig, uint32_t count, bool wide,
                                int32_t base_vertex) {
    auto &executor = rig.context->GetRenderExecutor();
    return RenderExecutorTestAccess::DrawIndex(
        executor, rig.Scheduler().Current(),
        {.index_count = count,
         .index_addr = reinterpret_cast<const void *>(
             AsyncBase + DrawStreamIndexOffset + (wide ? 0x100 : 0)),
         .instance_count = 1,
         .index_type_and_size = wide ? 1u : 0u,
         .base_vertex = base_vertex});
  }

  // The scenes in the order they are drawn. Later scenes see what earlier ones left behind, so a
  // draw that repeats the last one shows what is not recorded twice.
  std::vector<DrawStreamScene> DrawStreamScenes() {
    using Prim = Prospero::PrimitiveType;
    const auto primitive = [](AsyncDrawRig &rig, Prim type) {
      rig.user_config.SetPrimitiveType(type);
    };
    return {
        {"tri list with depth clear",
         [](AsyncDrawRig &rig) {
           rig.SetDepthClear(true);
           DrawStreamAuto(rig, 3, 1);
           rig.SetDepthClear(false);
         }},
        {"tri list repeated", [](AsyncDrawRig &rig) { DrawStreamAuto(rig, 3, 1); }},
        {"instanced", [](AsyncDrawRig &rig) { DrawStreamAuto(rig, 3, 3); }},
        {"first vertex", [](AsyncDrawRig &rig) { DrawStreamAuto(rig, 3, 1, 6); }},
        {"indexed 16 bit", [](AsyncDrawRig &rig) { DrawStreamIndexed(rig, 3, false, 0); }},
        {"indexed base vertex", [](AsyncDrawRig &rig) { DrawStreamIndexed(rig, 3, false, 4); }},
        {"indexed 32 bit", [](AsyncDrawRig &rig) { DrawStreamIndexed(rig, 3, true, 0); }},
        {"tri strip",
         [=](AsyncDrawRig &rig) {
           primitive(rig, Prim::kTriStrip);
           DrawStreamAuto(rig, 4, 1);
         }},
        {"quad list",
         [=](AsyncDrawRig &rig) {
           primitive(rig, Prim::kQuadListLegacy);
           DrawStreamAuto(rig, 8, 1);
         }},
        {"quad list first vertex",
         [](AsyncDrawRig &rig) { DrawStreamAuto(rig, 8, 1, 4); }},
        {"quad list indexed", [](AsyncDrawRig &rig) { DrawStreamIndexed(rig, 8, false, 0); }},
        {"quad list indexed base vertex",
         [](AsyncDrawRig &rig) { DrawStreamIndexed(rig, 4, true, 2); }},
        {"user clip planes",
         [=](AsyncDrawRig &rig) {
           primitive(rig, Prim::kTriList);
           auto control = rig.registers.GetClipControl();
           control.user_clip_planes = 3;
           rig.registers.SetClipControl(control);
           for (uint32_t i = 0; i < 8; ++i) {
             rig.registers.SetUserClipPlane(i, 0x3f800000u + i);
           }
           DrawStreamAuto(rig, 3, 1);
           control.user_clip_planes = 0;
           rig.registers.SetClipControl(control);
         }},
        {"cull mode changes",
         [](AsyncDrawRig &rig) {
           rig.registers.SetModeControl({.cull_back = true, .face = true});
           DrawStreamAuto(rig, 3, 1);
           rig.registers.SetModeControl({});
         }},
        {"address buffers in a descriptor set",
         [this](AsyncDrawRig &rig) { DrawStreamAddressBuffers(rig, DrawStreamReadOffset, false); }},
        {"address buffers pushed",
         [this](AsyncDrawRig &rig) {
           DrawStreamAddressBuffers(rig, DrawStreamReadOffset + 0x100, true);
         }},
        {"depth clear after other draws",
         [](AsyncDrawRig &rig) {
           rig.SetDepthClear(true);
           DrawStreamAuto(rig, 3, 1);
           rig.SetDepthClear(false);
         }},
        {"vertex buffer", [this](AsyncDrawRig &rig) { DrawStreamVertexBuffer(rig); }},
        {"vertex buffer and a GDS barrier",
         [this](AsyncDrawRig &rig) { DrawStreamVertexBuffer(rig, true); }},
    };
  }

  // Makes the rig's vertex shader one that reads guest memory by address, which binds the BDA page
  // table and the fault buffer. What it reads is zero, added to a position, so the picture is the
  // same. Another read address makes another shader. Returns the shader it replaced.
  uint64_t UseAddressVertex(AsyncDrawRig &rig, uint64_t read_offset) {
    const uint64_t read_address = AsyncBase + read_offset;
    static std::map<uint64_t, std::vector<u32>> shaders;
    auto &code = shaders[read_address];
    if (code.empty()) {
      AppendVMovLiteral(&code, 1, 0xbf800000u);
      AppendVMovLiteral(&code, 2, 0x40400000u);
      code.push_back(EncodeVopc(0xc2, InlineU32(1), 5));
      code.push_back(EncodeVop2(0x01, 3, Vgpr(1), 2));
      code.push_back(EncodeVopc(0xc2, InlineU32(2), 5));
      code.push_back(EncodeVop2(0x01, 4, Vgpr(1), 2));
      AppendVMovLiteral(&code, 8, static_cast<u32>(read_address));
      AppendVMovLiteral(&code, 9, static_cast<u32>(read_address >> 32u));
      code.push_back(EncodeFlat0(0x0c, 0));
      code.push_back(EncodeFlat1(5, 0x7d, 0, 8));
      code.push_back(EncodeVop2(0x03, 3, Vgpr(5), 3));
      AppendVMovU32(&code, 0, 0);
      AppendVMovLiteral(&code, 6, 0x3f800000u);
      code.push_back(EncodeExp0(0x0c, 0xf));
      code.push_back(EncodeExp1(3, 4, 0, 6));
      for (u32 parameter = 0; parameter < 8; parameter++) {
        code.push_back(EncodeExp0(0x20 + parameter, 0xf));
        code.push_back(EncodeExp1(0, 0, 0, 0));
      }
      AppendEnd(&code);
    }
    static ShaderUserData user_data{};
    const auto address = reinterpret_cast<uint64_t>(code.data());
    ShaderMapUserData(address,
                      {.type = Prospero::ShaderBinaryType::kGs,
                       .user_data = &user_data,
                       .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
    const auto rig_vertex = rig.shaders.GetVs().es_regs.data_addr;
    rig.shaders.SetEsShaderBase(address);
    return rig_vertex;
  }

  // Draws with the address vertex shader. The device's limit on push descriptors is set as given
  // while its pipeline is made, which decides whether the draw pushes its descriptors.
  void DrawStreamAddressBuffers(AsyncDrawRig &rig, uint64_t read_offset, bool push_descriptors) {
    auto &graphics = rig.context->GetGraphics();
    const auto device_limit = graphics.max_push_descriptors;
    graphics.max_push_descriptors = push_descriptors ? DrawStreamPushDescriptorLimit : 0;
    const auto rig_vertex = UseAddressVertex(rig, read_offset);
    DrawStreamAuto(rig, 3, 1);
    rig.shaders.SetEsShaderBase(rig_vertex);
    graphics.max_push_descriptors = device_limit;
  }

  // Makes the rig's vertex shader one that fetches its positions from a vertex buffer. With `gds`
  // it also adds to a GDS counter, so the draw needs the barrier that comes with the image
  // transitions. Returns the shader it replaced.
  uint64_t UseFetchVertex(AsyncDrawRig &rig, bool gds = false) {
    constexpr uint64_t vertices_address = AsyncBase + DrawStreamVertexOffset;
    constexpr std::array<float, 12> triangle{-1, -1, 0, 1, 3, -1, 0, 1, -1, 3, 0, 1};
    std::memcpy(reinterpret_cast<void *>(vertices_address), triangle.data(), sizeof(triangle));
    static std::array<std::vector<u32>, 2> fetch_vertices;
    auto &fetch_vertex = fetch_vertices[gds];
    if (fetch_vertex.empty()) {
      auto &code = fetch_vertex;
      code = {EncodeSMovB32(0, InlineU32(0)), EncodeSmem0(0x02u, 20, 4),
              EncodeSmem1(0),                 EncodeVop2(0x01u, 0, Vgpr(8), 5),
              EncodeMubuf0(0x03u, 0, true),   EncodeMubuf1(9, 5, 0)};
      if (gds) {
        AppendVMovU32(&code, 20, 0);
        AppendVMovU32(&code, 21, 1);
        code.push_back(EncodeDs0(0x00, 0, true));
        code.push_back(EncodeDs1(22, 21, 20));
      }
      code.push_back(EncodeExp0(0x0c, 0xf));
      code.push_back(EncodeExp1(9, 10, 11, 12));
      for (u32 parameter = 0; parameter < 8; parameter++) {
        code.push_back(EncodeExp0(0x20 + parameter, 0xf));
        code.push_back(EncodeExp1(0, 0, 0, 0));
      }
      AppendEnd(&code);
    }
    static std::array<uint16_t, static_cast<size_t>(DirectResourceType::Last) + 1> fetch_offsets;
    fetch_offsets.fill(ILLEGAL_DIRECT_OFFSET);
    fetch_offsets[static_cast<size_t>(DirectResourceType::PtrVertexBufferTable)] = 0;
    fetch_offsets[static_cast<size_t>(DirectResourceType::PtrVertexAttribDescTable)] = 2;
    static ShaderUserData fetch_user_data{
        .direct_resource_offset = fetch_offsets.data(),
        .direct_resource_count = static_cast<uint16_t>(fetch_offsets.size())};
    static ShaderSemantic semantic{.hardware_mapping = 9, .size_in_elements = 4};
    static const uint32_t vertex_buffer[]{
        static_cast<u32>(vertices_address),
        static_cast<u32>(vertices_address >> 32u) | (16u << 16u), 10,
        DstSel(4, 5, 6, 7) | (static_cast<u32>(Prospero::BufferFormat::k32_32_32_32Float) << 12u)};
    static const uint32_t attribute =
        static_cast<u32>(Prospero::VertexAttribFormat::k32_32_32_32Float) << 5u;
    const auto fetch_vs = reinterpret_cast<uint64_t>(fetch_vertex.data());
    ShaderMapUserData(fetch_vs,
                      {.type = Prospero::ShaderBinaryType::kGs,
                       .user_data = &fetch_user_data,
                       .input_semantics = &semantic,
                       .num_input_semantics = 1,
                       .code_size_bytes = static_cast<uint32_t>(fetch_vertex.size() * sizeof(u32))});
    const uint64_t tables[]{reinterpret_cast<uint64_t>(vertex_buffer),
                            reinterpret_cast<uint64_t>(&attribute)};
    const auto rig_vertex = rig.shaders.GetVs().es_regs.data_addr;
    rig.shaders.SetEsShaderBase(fetch_vs);
    rig.shaders.SetGsShaderResource2({.user_sgpr = 4});
    for (u32 i = 0; i < 4; i++) {
      rig.shaders.SetGsUserSgpr(i, static_cast<u32>(tables[i / 2] >> (i % 2 * 32)));
    }
    return rig_vertex;
  }

  // Draws with the fetch vertex shader, once without and once with an index buffer, then puts the
  // rig's own vertex shader back.
  void DrawStreamVertexBuffer(AsyncDrawRig &rig, bool gds = false) {
    const auto rig_vertex = UseFetchVertex(rig, gds);
    DrawStreamAuto(rig, 3, 1);
    DrawStreamIndexed(rig, 3, false, 0);
    rig.shaders.SetEsShaderBase(rig_vertex);
    rig.shaders.SetGsShaderResource2({});
  }

  // Runs every scene on a rig that builds its pipelines on the calling thread, and returns the
  // commands recorded, with a row before each scene naming it.
  std::vector<std::string> RecordDrawStream() {
    CommandStream::Scope hooks;
    AsyncDrawRig rig(*this, false);
    WriteDrawStreamIndices();
    rig.PaintSentinels();
    CommandStream::Recorder::Reset(true);
    for (const auto &scene : DrawStreamScenes()) {
      CommandStream::Add(std::string("== ") + scene.label);
      scene.run(rig);
    }
    auto rows = CommandStream::Recorder::rows;
    CommandStream::Recorder::active = false;
    return rows;
  }

  // Rows captured by running the scenes above through ExecutePreparedDraw as it was at 70eeb2c9,
  // before it was split into PrepareDraw and RecordDraw. Handles are numbered in the order they
  // first appear. The scenes sample no images, have no open guest query and no shader that writes
  // a buffer. They pin the draw calls, the index and vertex buffer binds, the push constants, the
  // pushed and the set descriptors with the address buffers, the dynamic state, the pass begin and
  // end, and the attachment barriers.
  static const std::vector<std::string> &DrawStreamRows() {
    static const std::vector<std::string> rows = {
      "== tri list with depth clear",
      "vkCmdPipelineBarrier2( img{image#0 TransferDstOptimal -> ColorAttachmentOptimal { AllTransfer }/{ TransferWrite } -> { ComputeShader | AllGraphics }/{ ColorAttachmentRead | ColorAttachmentWrite } { Color } mip 0+4294967295 layer 0+4294967295} )",
      "vkCmdPipelineBarrier2( img{image#1 TransferDstOptimal -> DepthAttachmentOptimal { AllTransfer }/{ TransferWrite } -> { ComputeShader | AllGraphics }/{ DepthStencilAttachmentRead | DepthStencilAttachmentWrite } { Depth } mip 0+4294967295 layer 0+4294967295} )",
      "vkCmdSetViewportWithCount(0.000000 0.000000 32.000000 32.000000 -1.000000 1.000000)",
      "vkCmdSetScissorWithCount(0,0 32x32)",
      "vkCmdSetDepthTestEnable(1)",
      "vkCmdSetDepthWriteEnable(0)",
      "vkCmdSetDepthCompareOp(7)",
      "vkCmdSetDepthBoundsTestEnable(0)",
      "vkCmdSetDepthBiasEnable(0)",
      "vkCmdSetStencilTestEnable(0)",
      "vkCmdSetCullMode(0)",
      "vkCmdSetFrontFace(0)",
      "vkCmdSetBlendConstants(0.000000 0.000000 0.000000 0.000000)",
      "vkCmdSetColorWriteEnableEXT(1)",
      "vkCmdSetLineWidth(1.000000)",
      "vkCmdSetAttachmentFeedbackLoopEnableEXT(0)",
      "vkCmdBeginRendering(32x32 layers 1 color0{view=view#0 layout=ColorAttachmentOptimal load=Load store=Store depth=0.000000 stencil=0} depth{view=view#1 layout=DepthAttachmentOptimal load=Clear store=Store depth=0.750000 stencil=0})",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDraw(3, 1, 0, 0)",
      "== tri list repeated",
      "vkCmdSetDepthWriteEnable(1)",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDraw(3, 1, 0, 0)",
      "== instanced",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDraw(3, 3, 0, 0)",
      "== first vertex",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDraw(3, 1, 6, 0)",
      "== indexed 16 bit",
      "vkCmdBindIndexBuffer(buffer#0, 0, 0)",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDrawIndexed(3, 1, 0, 0, 0)",
      "== indexed base vertex",
      "vkCmdBindIndexBuffer(buffer#0, 64, 0)",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDrawIndexed(3, 1, 0, 4, 0)",
      "== indexed 32 bit",
      "vkCmdBindIndexBuffer(buffer#0, 128, 1)",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDrawIndexed(3, 1, 0, 0, 0)",
      "== tri strip",
      "vkCmdBindPipeline(0, pipeline#1)",
      "vkCmdDraw(4, 1, 0, 0)",
      "== quad list",
      "vkCmdBindPipeline(0, pipeline#2)",
      "vkCmdDraw(4, 1, 0, 0)",
      "vkCmdDraw(4, 1, 4, 0)",
      "== quad list first vertex",
      "vkCmdBindPipeline(0, pipeline#2)",
      "vkCmdDraw(4, 1, 4, 0)",
      "vkCmdDraw(4, 1, 8, 0)",
      "== quad list indexed",
      "vkCmdBindIndexBuffer(buffer#0, 192, 0)",
      "vkCmdBindPipeline(0, pipeline#2)",
      "vkCmdDrawIndexed(4, 1, 0, 0, 0)",
      "vkCmdDrawIndexed(4, 1, 4, 0, 0)",
      "== quad list indexed base vertex",
      "vkCmdBindIndexBuffer(buffer#0, 256, 1)",
      "vkCmdBindPipeline(0, pipeline#2)",
      "vkCmdDrawIndexed(4, 1, 0, 2, 0)",
      "== user clip planes",
      "vkCmdPushConstants(layout#0, { Vertex | Fragment }, +20/32, 0000803f0100803f0200803f0300803f0400803f0500803f0600803f0700803f)",
      "vkCmdBindPipeline(0, pipeline#3)",
      "vkCmdDraw(3, 1, 0, 0)",
      "== cull mode changes",
      "vkCmdSetCullMode(2)",
      "vkCmdSetFrontFace(1)",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDraw(3, 1, 0, 0)",
      "== address buffers in a descriptor set",
      "vkCmdBindDescriptorSets(Graphics, layout#1, first 0 set#0, dynamic 0)",
      "vkCmdSetCullMode(0)",
      "vkCmdSetFrontFace(0)",
      "vkCmdBindPipeline(0, pipeline#4)",
      "vkCmdDraw(3, 1, 0, 0)",
      "== address buffers pushed",
      "vkCmdPushDescriptorSet(Graphics, layout#2, set 0; bind=53 StorageBuffer n=1 [buffer=buffer#1 +0/805306368]; bind=54 StorageBuffer n=1 [buffer=buffer#2 +0/12582912])",
      "vkCmdBindPipeline(0, pipeline#5)",
      "vkCmdDraw(3, 1, 0, 0)",
      "== depth clear after other draws",
      "vkCmdSetDepthWriteEnable(0)",
      "vkCmdEndRendering()",
      "vkCmdBeginRendering(32x32 layers 1 color0{view=view#0 layout=ColorAttachmentOptimal load=Load store=Store depth=0.000000 stencil=0} depth{view=view#1 layout=DepthAttachmentOptimal load=Clear store=Store depth=0.750000 stencil=0})",
      "vkCmdBindPipeline(0, pipeline#0)",
      "vkCmdDraw(3, 1, 0, 0)",
      "== vertex buffer",
      "vkCmdBindVertexBuffers2(first 0; buffer#0 +320/160)",
      "vkCmdSetDepthWriteEnable(1)",
      "vkCmdBindPipeline(0, pipeline#6)",
      "vkCmdDraw(3, 1, 0, 0)",
      "vkCmdBindVertexBuffers2(first 0; buffer#0 +512/160)",
      "vkCmdBindIndexBuffer(buffer#0, 704, 0)",
      "vkCmdBindPipeline(0, pipeline#6)",
      "vkCmdDrawIndexed(3, 1, 0, 0, 0)",
      "== vertex buffer and a GDS barrier",
      "vkCmdBindVertexBuffers2(first 0; buffer#0 +768/160)",
      "vkCmdEndRendering()",
      "vkCmdPipelineBarrier({ ComputeShader | Transfer | Host | AllGraphics } -> { VertexShader } memory 0 buffers 1 images 0)",
      "vkCmdBindDescriptorSets(Graphics, layout#3, first 0 set#1, dynamic 0)",
      "vkCmdBeginRendering(32x32 layers 1 color0{view=view#0 layout=ColorAttachmentOptimal load=Load store=Store depth=0.000000 stencil=0} depth{view=view#1 layout=DepthAttachmentOptimal load=Load store=Store depth=0.750000 stencil=0})",
      "vkCmdBindPipeline(0, pipeline#7)",
      "vkCmdDraw(3, 1, 0, 0)",
      "vkCmdBindVertexBuffers2(first 0; buffer#0 +960/160)",
      "vkCmdEndRendering()",
      "vkCmdPipelineBarrier({ ComputeShader | Transfer | Host | AllGraphics } -> { VertexShader } memory 0 buffers 1 images 0)",
      "vkCmdBindDescriptorSets(Graphics, layout#3, first 0 set#2, dynamic 0)",
      "vkCmdBindIndexBuffer(buffer#0, 1152, 0)",
      "vkCmdBeginRendering(32x32 layers 1 color0{view=view#0 layout=ColorAttachmentOptimal load=Load store=Store depth=0.000000 stencil=0} depth{view=view#1 layout=DepthAttachmentOptimal load=Load store=Store depth=0.750000 stencil=0})",
      "vkCmdBindPipeline(0, pipeline#7)",
      "vkCmdDrawIndexed(3, 1, 0, 0, 0)",
    };
    return rows;
  }

  // The commands of every scene are the ones the draws recorded before they were split.
  void CheckDrawStream() {
    constexpr const char *name = "DrawStream";
    Require(name, "commands unchanged",
            RenderExecutorTestAccess::RowMatches(name, RecordDrawStream(), DrawStreamRows()),
            "a draw recorded other commands than it did before it was split into its eager and "
            "recording halves");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A scene drawn through the two halves of a draw, with something changed in between.
  struct SplitScene {
    const char *label;
    Prospero::PrimitiveType primitive;
    vk::PrimitiveTopology topology;
    uint32_t vertex_count;
    // The user clip planes enabled, as a mask.
    uint32_t clip_planes;
    // The draw pushes its object id, which the rig's shaders never ask for.
    bool object_id;
  };
  enum class SplitChange { None, Registers, View };
  static constexpr uint32_t SplitObjectId = 0x1234;
  static constexpr uint32_t SplitOtherObjectId = 0xdead;
  static constexpr uint32_t SplitClipDwords = 8;

  static std::vector<SplitScene> SplitScenes() {
    using Prim = Prospero::PrimitiveType;
    using Topology = vk::PrimitiveTopology;
    return {{"quad list", Prim::kQuadListLegacy, Topology::eTriangleFan, 8, 0, false},
            {"clip planes", Prim::kTriList, Topology::eTriangleList, 3, 3, false},
            {"object id", Prim::kTriList, Topology::eTriangleList, 3, 0, true}};
  }

  // Sets the registers a scene draws with.
  static void SetSplitRegisters(AsyncDrawRig &rig, const SplitScene &scene) {
    rig.user_config.SetPrimitiveType(scene.primitive);
    rig.user_config.SetObjectId(SplitObjectId);
    auto control = rig.registers.GetClipControl();
    control.user_clip_planes = static_cast<uint8_t>(scene.clip_planes);
    rig.registers.SetClipControl(control);
    for (uint32_t i = 0; i < SplitClipDwords; ++i) {
      rig.registers.SetUserClipPlane(i, 0x3f800000u + i);
    }
  }

  // Moves every register the draw captured to another value, as the next draw would.
  static void ChangeSplitRegisters(AsyncDrawRig &rig) {
    rig.user_config.SetPrimitiveType(Prospero::PrimitiveType::kPointList);
    rig.user_config.SetObjectId(SplitOtherObjectId);
    for (uint32_t i = 0; i < SplitClipDwords; ++i) {
      rig.registers.SetUserClipPlane(i, 0x40000000u + i);
    }
  }

  // Moves the same values in the view, to show the recording would follow them.
  static void ChangeSplitView(DrawView &view) {
    view.primitive_type = Prospero::PrimitiveType::kPointList;
    view.object_id = SplitOtherObjectId;
    for (uint32_t i = 0; i < SplitClipDwords; ++i) {
      view.user_clip_plane_dwords[i] = 0x40000000u + i;
    }
  }

  // The commands the recording half of a scene records into the scheduler's buffer, on a fresh rig
  // so that no dynamic state or open pass is left over from another run.
  std::vector<std::string> RecordSplitScene(const SplitScene &scene, SplitChange change) {
    CommandStream::Scope hooks;
    AsyncDrawRig rig(*this, false);
    rig.PaintSentinels();
    SetSplitRegisters(rig, scene);
    CommandStream::Recorder::Reset(true);
    RenderExecutorTestAccess::DrawAutoSplit(
        rig.context->GetRenderExecutor(), rig.Scheduler().Current(),
        {.vertex_count = scene.vertex_count, .instance_count = 1}, scene.topology,
        [&](DrawView &view) {
          if (scene.object_id) view.draw_dwords = 1;
          if (change == SplitChange::Registers) ChangeSplitRegisters(rig);
          if (change == SplitChange::View) ChangeSplitView(view);
        });
    CommandStream::Recorder::active = false;
    return CommandStream::Recorder::rows;
  }

  // The same scene through the ordinary entry point, which has no gap between the halves.
  std::vector<std::string> RecordPlainScene(const SplitScene &scene) {
    CommandStream::Scope hooks;
    AsyncDrawRig rig(*this, false);
    rig.PaintSentinels();
    SetSplitRegisters(rig, scene);
    CommandStream::Recorder::Reset(true);
    rig.Draw({.vertex_count = scene.vertex_count, .instance_count = 1});
    CommandStream::Recorder::active = false;
    return CommandStream::Recorder::rows;
  }

  static bool HasRow(const std::vector<std::string> &rows, const std::string &part) {
    return std::ranges::any_of(
        rows, [&](const auto &row) { return row.find(part) != std::string::npos; });
  }

  // Recording a draw uses what the draw captured and nothing the registers say afterwards. For
  // each scene, changing the registers between the halves changes nothing, and changing the same
  // values in the view does, so the scene can tell a recording that follows the registers.
  void CheckDrawRecordIgnoresRegisters() {
    constexpr const char *name = "DrawRecordRegisters";
    for (const auto &scene : SplitScenes()) {
      const auto steady = RecordSplitScene(scene, SplitChange::None);
      Require(name, scene.label, !steady.empty() && HasRow(steady, "vkCmdDraw("),
              "the scene recorded no draw");
      if (!scene.object_id) {
        Require(name, scene.label, steady == RecordPlainScene(scene),
                "drawing through the two halves recorded other commands than DrawAuto");
      }
      Require(name, scene.label,
              RenderExecutorTestAccess::RowMatches(scene.label,
                                                   RecordSplitScene(scene, SplitChange::Registers),
                                                   steady),
              "registers changed after the eager half reached the recorded commands");
      Require(name, scene.label, RecordSplitScene(scene, SplitChange::View) != steady,
              "the captured values do not show in the recorded commands, so the scene proves "
              "nothing about them");
    }
    // The object id is the register's own value when the draw was met.
    Require(name, "object id captured",
            HasRow(RecordSplitScene(SplitScenes()[2], SplitChange::None), "+0/4, 34120000"),
            "the draw did not push the object id its registers held when it was met");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Records a draw into a command buffer of its own, with no state shared with the scheduler's.
  struct PlainDrawEnv final : DrawRecordEnv {
    DescriptorScratch scratch;
    DescriptorScratch &Scratch() override { return scratch; }
    void BeginRendering(vk::CommandBuffer command, const RenderState &rendering) override {
      RecordBeginRendering(command, rendering);
    }
    int32_t BeginDraw(vk::CommandBuffer) override { return -1; }
    void EndDraw(vk::CommandBuffer command, int32_t, const DrawCallInfo &) override {
      RecordEndRendering(command);
    }
  };

  // A command buffer that is never submitted, for a draw recorded away from the scheduler's.
  struct SideBuffer {
    GraphicContext &graphics;
    vk::CommandPool pool;
    vk::CommandBuffer buffer;

    explicit SideBuffer(GraphicContext &owner) : graphics(owner) {
      vk::CommandPoolCreateInfo pool_info{};
      pool_info.queueFamilyIndex = graphics.queue_family;
      Require("SideBuffer", "pool",
              graphics.device.createCommandPool(&pool_info, nullptr, &pool) == vk::Result::eSuccess,
              "no side command pool");
      vk::CommandBufferAllocateInfo allocate{};
      allocate.commandPool = pool;
      allocate.commandBufferCount = 1;
      Require("SideBuffer", "buffer",
              graphics.device.allocateCommandBuffers(&allocate, &buffer) == vk::Result::eSuccess,
              "no side command buffer");
      vk::CommandBufferBeginInfo begin{};
      Require("SideBuffer", "begin", buffer.begin(&begin) == vk::Result::eSuccess,
              "the side buffer did not begin");
    }
    ~SideBuffer() {
      (void)buffer.end();
      graphics.device.destroyCommandPool(pool, nullptr);
    }
    SideBuffer(const SideBuffer &) = delete;
    SideBuffer &operator=(const SideBuffer &) = delete;
  };

  // The binding in a set of writes that has the given descriptor type and nth place among them.
  static const vk::DescriptorBufferInfo *WrittenBuffer(const DescriptorScratch &scratch,
                                                       size_t index) {
    return index < scratch.writes.size() ? scratch.writes[index].pBufferInfo : nullptr;
  }

  // The recording half runs later into another buffer. It writes its descriptors into the scratch
  // the caller gives it, takes the address buffers from the view, transits nothing and leaves every
  // image as the eager half set it. With push descriptors and with a descriptor set.
  void CheckDrawRecordIntoSideBuffer(bool push_descriptors) {
    const char *name = push_descriptors ? "DrawRecordPushed" : "DrawRecordDescriptorSet";
    CommandStream::Scope hooks;
    AsyncDrawRig rig(*this, false);
    auto &graphics = rig.context->GetGraphics();
    graphics.max_push_descriptors = push_descriptors ? DrawStreamPushDescriptorLimit : 0;
    UseAddressVertex(rig, DrawStreamReadOffset + 0x200);
    auto &executor = rig.context->GetRenderExecutor();
    DrawView view;
    const auto *pipeline = RenderExecutorTestAccess::PrepareAutoDraw(
        executor, rig.Scheduler().Current(), {.vertex_count = 3, .instance_count = 1},
        vk::PrimitiveTopology::eTriangleList, view);
    Require(name, "prepared", pipeline != nullptr && pipeline->uses_push_descriptors == push_descriptors,
            "the draw was not prepared with the descriptor path under test");
    rig.ResolveTargets();
    const auto &image = rig.Textures().GetImage(rig.color.image_id);
    const auto layout = image.backing.state.layout;
    const auto access = image.backing.state.access_mask;
    // The page table view is replaced, which only a recording that reads the view can show.
    view.address_buffers.views[0].buffer = view.address_buffers.views[1].buffer;

    // The runtimes live in the draw's stack frame, which is gone. Recording must not read them.
    for (uint32_t stage = 0; stage < view.stage_count; ++stage) {
      view.bindings[stage]->runtime = nullptr;
    }
    SideBuffer side(graphics);
    PlainDrawEnv env;
    CommandStream::Recorder::Reset(true);
    RenderExecutorTestAccess::RecordDraw(executor, side.buffer, view, *pipeline, env);
    CommandStream::Recorder::active = false;
    const auto rows = CommandStream::Recorder::rows;

    Require(name, "own scratch", env.scratch.writes.size() == 2 &&
                RenderExecutorTestAccess::Scratch(executor).writes.empty(),
            "the descriptors were not written into the scratch the caller gave");
    Require(name, "address buffers from the view",
            WrittenBuffer(env.scratch, 0) != nullptr && WrittenBuffer(env.scratch, 1) != nullptr &&
                WrittenBuffer(env.scratch, 0)->buffer == view.address_buffers.views[1].buffer &&
                WrittenBuffer(env.scratch, 1)->buffer == view.address_buffers.views[1].buffer,
            "the descriptors did not use the address buffers the view carries");
    Require(name, "descriptor path",
            HasRow(rows, push_descriptors ? "vkCmdPushDescriptorSet(" : "vkCmdBindDescriptorSets(") &&
                !HasRow(rows, push_descriptors ? "vkCmdBindDescriptorSets(" : "vkCmdPushDescriptorSet("),
            "the descriptors took the other path");
    Require(name, "plain rendering and one draw",
            HasRow(rows, "vkCmdBeginRendering(") && HasRow(rows, "vkCmdEndRendering(") &&
                HasRow(rows, "vkCmdDraw(3, 1, 0, 0)"),
            "the draw was not recorded in a rendering of its own");
    Require(name, "no barrier", !HasRow(rows, "Barrier"), "recording a draw recorded a barrier");
    Require(name, "layouts left alone",
            image.backing.state.layout == layout && image.backing.state.access_mask == access,
            "recording a draw changed an image's tracked layout");
    RenderExecutorTestAccess::ResetBindings(executor);
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // The vertex buffers are bound by the eager half when the caller asks, where the draw has always
  // bound them, and by the recording half when it does not.
  void CheckDrawRecordVertexBinds(bool bind_early) {
    const char *name = bind_early ? "DrawRecordVertexEarly" : "DrawRecordVertexLate";
    constexpr const char *bind = "vkCmdBindVertexBuffers2(";
    CommandStream::Scope hooks;
    AsyncDrawRig rig(*this, false);
    WriteDrawStreamIndices();
    UseFetchVertex(rig);
    auto &executor = rig.context->GetRenderExecutor();
    DrawView view;
    CommandStream::Recorder::Reset(true);
    const auto *pipeline = RenderExecutorTestAccess::PrepareAutoDraw(
        executor, rig.Scheduler().Current(), {.vertex_count = 3, .instance_count = 1},
        vk::PrimitiveTopology::eTriangleList, view, bind_early);
    const auto eager_rows = CommandStream::Recorder::rows;
    CommandStream::Recorder::Reset(true);
    SideBuffer side(rig.context->GetGraphics());
    PlainDrawEnv env;
    Require(name, "prepared", pipeline != nullptr, "the draw was not prepared");
    RenderExecutorTestAccess::RecordDraw(executor, side.buffer, view, *pipeline, env);
    CommandStream::Recorder::active = false;
    Require(name, "bound where asked",
            HasRow(eager_rows, bind) == bind_early &&
                HasRow(CommandStream::Recorder::rows, bind) == !bind_early,
            "the vertex buffers were not bound by the half the caller chose");
    Require(name, "bound once",
            std::ranges::count_if(CommandStream::Recorder::rows, [](const auto &row) {
              return row.find("vkCmdBindVertexBuffers2(") != std::string::npos;
            }) == (bind_early ? 0 : 1),
            "the vertex buffers were bound twice or not at all");
    RenderExecutorTestAccess::ResetBindings(executor);
    std::printf("[gpu]     %-32s ok\n", name);
  }

  void CheckDrawRecord() {
    CheckDrawStream();
    CheckDrawRecordIgnoresRegisters();
    CheckDrawRecordIntoSideBuffer(true);
    CheckDrawRecordIntoSideBuffer(false);
    CheckDrawRecordVertexBinds(true);
    CheckDrawRecordVertexBinds(false);
  }
