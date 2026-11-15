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
  // same. Another read address makes another shader. Returns the shader it replaced. With `scale`
  // the read value multiplies the position instead, so the draw covers the target only if the
  // shader read what the test wrote there.
  uint64_t UseAddressVertex(AsyncDrawRig &rig, uint64_t read_offset, bool scale = false) {
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
      code.push_back(EncodeVop2(scale ? 0x08 : 0x03, 3, Vgpr(5), 3));
      if (scale) code.push_back(EncodeVop2(0x08, 4, Vgpr(5), 4));
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

  // The code of a pixel shader that writes the rig's gray plus the word at `read_address`.
  static std::vector<u32> AddressPixelCode(uint64_t read_address) {
    std::vector<u32> code;
    AppendVMovLiteral(&code, 1, 0x3e800000u);
    AppendVMovLiteral(&code, 2, 0x3f800000u);
    AppendVMovLiteral(&code, 8, static_cast<u32>(read_address));
    AppendVMovLiteral(&code, 9, static_cast<u32>(read_address >> 32u));
    code.push_back(EncodeFlat0(0x0c, 0));
    code.push_back(EncodeFlat1(5, 0x7d, 0, 8));
    code.push_back(EncodeVop2(0x03, 1, Vgpr(5), 1));
    code.push_back(EncodeExp0(0x00, 0xf));
    code.push_back(EncodeExp1(1, 1, 1, 2));
    AppendEnd(&code);
    return code;
  }

  // Makes the rig's pixel shader one that reads guest memory by address, which binds the BDA page
  // table and the fault buffer in the pixel stage. It writes the rig's gray plus what it reads, and
  // what it reads is zero unless a test wrote there, so the picture is the same. Another read
  // address makes another shader. Returns the shader it replaced.
  uint64_t UseAddressPixel(AsyncDrawRig &rig, uint64_t read_offset) {
    const uint64_t read_address = AsyncBase + read_offset;
    static std::map<uint64_t, std::vector<u32>> shaders;
    auto &code = shaders[read_address];
    if (code.empty()) code = AddressPixelCode(read_address);
    const auto address = reinterpret_cast<uint64_t>(code.data());
    ShaderMapUserData(address, {.type = Prospero::ShaderBinaryType::kPs,
                                .code_size_bytes = static_cast<u32>(code.size() * sizeof(u32))});
    const auto rig_pixel = rig.shaders.GetPs().ps_regs.data_addr;
    rig.shaders.SetPsShaderBase(address);
    return rig_pixel;
  }

  // The stage of the rig that reads guest memory by address, which gives that stage descriptors.
  enum class AddressStage { Vertex, Pixel };

  // Makes the given stage read guest memory by address, and returns the shader it replaced.
  uint64_t UseAddressStage(AsyncDrawRig &rig, AddressStage stage, uint64_t read_offset) {
    return stage == AddressStage::Vertex ? UseAddressVertex(rig, read_offset)
                                         : UseAddressPixel(rig, read_offset);
  }

  // Puts back the shader UseAddressStage replaced.
  static void RestoreAddressStage(AsyncDrawRig &rig, AddressStage stage, uint64_t replaced) {
    if (stage == AddressStage::Vertex) rig.shaders.SetEsShaderBase(replaced);
    else rig.shaders.SetPsShaderBase(replaced);
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

  // A draw prepared as the first half of a blocking draw, with what a hole needs from it. The pipeline
  // is the one the draw built, and its key is the one a hole waits on.
  struct HoleSource {
    DrawView view;
    const PipelineCache::Pipeline *pipeline = nullptr;
    GraphicsPipelineKey key{};
  };

  // The vertex buffers are left to the hole's own buffer, as a hole requires.
  HoleSource PrepareHoleSource(AsyncDrawRig &rig, bool strip_clears = true) {
    HoleSource source;
    source.pipeline = RenderExecutorTestAccess::PrepareAutoDraw(
        rig.context->GetRenderExecutor(), rig.Scheduler().Current(),
        {.vertex_count = 3, .instance_count = 1}, vk::PrimitiveTopology::eTriangleList,
        source.view, false);
    Require("DeferredHoles", "prepared", source.pipeline != nullptr, "the draw was not prepared");
    const auto key = PipelineCacheTestAccess::KeyOf(rig.Pipelines(), source.pipeline);
    Require("DeferredHoles", "key", key.has_value(), "the draw's pipeline has no key");
    source.key = *key;
    // The eager pass of a real hole consumes the draw's clears, so the hole's view carries none.
    if (strip_clears) {
      source.view.rendering.depth_stencil_attachment.depth_clear = false;
      source.view.rendering.depth_stencil_attachment.stencil_clear = false;
    }
    return source;
  }

  // Splits the scheduler's buffer for a hole of the prepared draw, then drops the executor's
  // bindings the way a finished draw does. The hole keeps its own copies.
  void SplitHoleOf(AsyncDrawRig &rig, const HoleSource &source) {
    rig.Scheduler().SplitForHole(std::make_unique<DrawHole>(source.view, source.key));
    RenderExecutorTestAccess::ResetBindings(rig.context->GetRenderExecutor());
  }

  // seg0, hole0, seg1 go out in that order, and the draw after the split sets all its state. The
  // draw before the hole is blocking, so the state was committed and only an invalidation at the
  // split makes the draw after it set the viewport again. Without the split nothing is set.
  void CheckHoleBookkeeping() {
    constexpr const char *name = "HoleBookkeeping";
    CommandStream::Scope hooks;
    const auto third_draw_rows = [&](bool split) {
      AsyncDrawRig rig(*this, false);
      auto &scheduler = rig.Scheduler();
      CommandSchedulerTestAccess::AllowThisThread(scheduler);
      rig.PaintSentinels();
      Require(name, "first draw", rig.Draw(), "a draw with a ready pipeline was refused");
      const auto tick = scheduler.CurrentTick();
      if (split) {
        CommandSchedulerTestAccess::SetCadence(
            scheduler, MakeTestCadence(true, SubmitCadence::Clock::duration::zero()));
        scheduler.NoteRecordedWork();
        const auto source = PrepareHoleSource(rig);
        SplitHoleOf(rig, source);
        Require(name, "tick and cadence kept",
                scheduler.CurrentTick() == tick && CommandSchedulerTestAccess::CadenceDue(scheduler),
                "the split changed the tick or reset the submit cadence");
        Require(name, "one pending hole",
                scheduler.PendingHoleCount() == 1 &&
                    CommandSchedulerTestAccess::HeldCount(scheduler) == 2 &&
                    !CommandSchedulerTestAccess::HeldIsHole(scheduler, 0) &&
                    CommandSchedulerTestAccess::HeldIsHole(scheduler, 1),
                "the split did not leave one segment and one reserved hole");
      } else {
        Require(name, "second draw", rig.Draw(), "a draw with a ready pipeline was refused");
      }
      CommandStream::Recorder::Reset(true);
      Require(name, "third draw", rig.Draw(), "a draw with a ready pipeline was refused");
      auto rows = CommandStream::Recorder::rows;
      CommandStream::Recorder::active = false;
      if (split) {
        const auto first = CommandSchedulerTestAccess::HeldBuffer(scheduler, 0);
        const auto hole = CommandSchedulerTestAccess::HeldBuffer(scheduler, 1);
        const auto last = scheduler.Current().Handle();
        CommandStream::Recorder::Reset(true);
        scheduler.Flush();
        const auto resolved = CommandStream::Recorder::rows;
        CommandStream::Recorder::active = false;
        const auto sent = CommandSchedulerTestAccess::SubmitBuffers(scheduler);
        Require(name, "submit order", sent == std::vector<vk::CommandBuffer>{first, hole, last},
                "the submit did not send the segment, the hole and the next segment in that order");
        Require(name, "resolved",
                scheduler.PendingHoleCount() == 0 &&
                    CommandSchedulerTestAccess::HeldCount(scheduler) == 0 &&
                    scheduler.CurrentTick() == tick + 1,
                "the submit left a hole pending or did not take one tick");
        Require(name, "hole sets all state", HasRow(resolved, "vkCmdSetViewportWithCount("),
                "the hole did not set the dynamic state, which its buffer starts without");
      }
      return rows;
    };
    Require(name, "baseline sets no state",
            !HasRow(third_draw_rows(false), "vkCmdSetViewportWithCount("),
            "a draw in the same buffer set state that did not change, so the split proves nothing");
    Require(name, "state invalidated at the split",
            HasRow(third_draw_rows(true), "vkCmdSetViewportWithCount("),
            "the draw after the split did not set its state in the new buffer");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A segment nothing was recorded into is left out of the submit. Two holes in a row leave one
  // between them, and the last segment is empty too. With no hole the one buffer is still sent.
  void CheckHoleEmptySegments() {
    constexpr const char *name = "HoleEmptySegments";
    CommandStream::Scope hooks;
    {
      AsyncDrawRig rig(*this, false);
      auto &scheduler = rig.Scheduler();
      CommandSchedulerTestAccess::AllowThisThread(scheduler);
      rig.PaintSentinels();
      const auto source = PrepareHoleSource(rig);
      SplitHoleOf(rig, source);
      scheduler.SplitForHole(std::make_unique<DrawHole>(source.view, source.key));
      Require(name, "layout",
              CommandSchedulerTestAccess::HeldCount(scheduler) == 4 &&
                  CommandSchedulerTestAccess::HeldHasCommands(scheduler, 0) &&
                  !CommandSchedulerTestAccess::HeldHasCommands(scheduler, 2),
              "the segments between the splits were not told apart by what was recorded");
      const auto first = CommandSchedulerTestAccess::HeldBuffer(scheduler, 0);
      const auto hole_one = CommandSchedulerTestAccess::HeldBuffer(scheduler, 1);
      const auto hole_two = CommandSchedulerTestAccess::HeldBuffer(scheduler, 3);
      scheduler.Flush();
      Require(name, "empty segments skipped",
              CommandSchedulerTestAccess::SubmitBuffers(scheduler) ==
                  std::vector<vk::CommandBuffer>{first, hole_one, hole_two},
              "the submit sent an empty segment or lost a buffer");
    }
    AsyncDrawRig rig(*this, false);
    rig.PaintSentinels();
    Require(name, "plain draw", rig.Draw(), "a draw with a ready pipeline was refused");
    rig.Scheduler().Flush();
    Require(name, "one buffer without holes",
            CommandSchedulerTestAccess::SubmitBuffers(rig.Scheduler()).size() == 1,
            "a submit with no hole did not send exactly its one buffer");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Flush, Finish and Wait each resolve every hole, so none is pending when they return.
  void CheckHoleSyncPoints() {
    constexpr const char *name = "HoleSyncPoints";
    CommandStream::Scope hooks;
    for (int sync = 0; sync < 3; ++sync) {
      AsyncDrawRig rig(*this, false);
      auto &scheduler = rig.Scheduler();
      CommandSchedulerTestAccess::AllowThisThread(scheduler);
      rig.PaintSentinels();
      SplitHoleOf(rig, PrepareHoleSource(rig));
      Require(name, "hole pending", scheduler.PendingHoleCount() == 1, "the split left no hole");
      if (sync == 0) scheduler.Flush();
      if (sync == 1) scheduler.Finish();
      if (sync == 2) scheduler.Wait(scheduler.CurrentTick());
      Require(name, sync == 0 ? "flush" : sync == 1 ? "finish" : "wait",
              scheduler.PendingHoleCount() == 0 &&
                  CommandSchedulerTestAccess::HeldCount(scheduler) == 0,
              "the call returned with a hole not recorded");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A hole's build may be taken into the cache before the hole is resolved, by a drain of the
  // published builds or by a blocking draw of the same key. Taking it again is fatal, so the
  // resolve must find it built. The last case has neither, and the resolve takes the build itself.
  void CheckHoleEarlyAdoption() {
    constexpr const char *name = "HoleEarlyAdoption";
    CommandStream::Scope hooks;
    for (int early = 0; early < 3; ++early) {
      PipelineCacheTestAccess::BuildHold hold;
      AsyncDrawRig rig(*this, true);
      auto &scheduler = rig.Scheduler();
      auto &cache = rig.Pipelines();
      CommandSchedulerTestAccess::AllowThisThread(scheduler);
      rig.PaintSentinels();
      const auto source = PrepareHoleSource(rig);
      PipelineCacheTestAccess::HoldBuildQueue(cache, hold);
      PipelineCacheTestAccess::RetireToBackground(cache, source.key);
      SplitHoleOf(rig, source);
      Require(name, "pending",
              PipelineCacheTestAccess::PendingBuilds(cache) == 1 &&
                  !PipelineCacheTestAccess::IsBuilt(cache, source.key),
              "the hole's build was not left pending");
      if (early == 0) {
        hold.release = true;
        WaitUntilTrue(name, "build",
                      [&] { return PipelineCacheTestAccess::FinishedBuilds(cache) == 1; });
        WaitUntilTrue(name, "published",
                      [&] { return PipelineCacheTestAccess::HasPublished(cache); });
        PipelineCacheTestAccess::DrainPublished(cache);
      } else if (early == 1) {
        PipelineCacheTestAccess::FinishPending(cache, source.key);
      }
      Require(name, "taken early", PipelineCacheTestAccess::IsBuilt(cache, source.key) == (early != 2),
              "the build was taken before the resolve when it should not be, or not when it should");
      scheduler.Flush();
      Require(name, "resolved",
              scheduler.PendingHoleCount() == 0 && PipelineCacheTestAccess::PendingBuilds(cache) == 0 &&
                  PipelineCacheTestAccess::IsBuilt(cache, source.key) &&
                  CommandSchedulerTestAccess::SubmitBuffers(scheduler).size() == 2,
              "the resolve did not record the hole against its built pipeline");
      hold.release = true;
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  static bool IsStateBarrier(const std::string &row) {
    return row.rfind("vkCmdPipelineBarrier2( mem{", 0) == 0;
  }

  // The stream a hole records equals the stream of the same draw met in order, with two additions.
  // The additions are the memory barriers at the start and the end of the hole's buffer, and the
  // split itself, which adds no commands. The hole's copy of the bindings has no runtime, so a
  // recording that read one would have stopped.
  void CheckHoleRecordedStream() {
    constexpr const char *name = "HoleRecordedStream";
    CommandStream::Scope hooks;
    std::vector<std::string> blocking;
    {
      AsyncDrawRig rig(*this, false);
      rig.PaintSentinels();
      CommandStream::Recorder::Reset(true);
      Require(name, "blocking draw", rig.Draw(), "a draw with a ready pipeline was refused");
      rig.Scheduler().Flush();
      blocking = CommandStream::Recorder::rows;
      CommandStream::Recorder::active = false;
    }
    AsyncDrawRig rig(*this, false);
    CommandSchedulerTestAccess::AllowThisThread(rig.Scheduler());
    rig.PaintSentinels();
    CommandStream::Recorder::Reset(true);
    const auto source = PrepareHoleSource(rig);
    Require(name, "source keeps its runtime", source.view.bindings[0]->runtime != nullptr,
            "the test draw had no runtime to drop, so the copy proves nothing");
    SplitHoleOf(rig, source);
    rig.Scheduler().Flush();
    const auto rows = CommandStream::Recorder::rows;
    CommandStream::Recorder::active = false;

    Require(name, "blocking has no barrier", std::ranges::none_of(blocking, IsStateBarrier),
            "the blocking stream has a memory barrier, so the hole's cannot be told apart");
    std::vector<std::string> without_barriers;
    std::vector<size_t> barrier_places;
    for (size_t index = 0; index < rows.size(); ++index) {
      if (IsStateBarrier(rows[index])) barrier_places.push_back(index);
      else without_barriers.push_back(rows[index]);
    }
    Require(name, "two barriers", barrier_places.size() == 2,
            "the hole's buffer did not start and end with one memory barrier");
    Require(name, "same stream",
            RenderExecutorTestAccess::RowMatches(name, without_barriers, blocking),
            "the hole recorded other commands than the same draw met in order");
    const auto first_state = static_cast<size_t>(
        std::ranges::find_if(rows,
                             [](const auto &row) {
                               return row.find("vkCmdSetViewportWithCount(") != std::string::npos;
                             }) -
        rows.begin());
    Require(name, "barriers at the ends",
            barrier_places.size() == 2 && barrier_places[0] < first_state &&
                barrier_places[1] + 1 == rows.size(),
            "the barriers are not at the start and the end of the hole's buffer");
    Require(name, "barrier stages",
            barrier_places.size() == 2 && rows[barrier_places[0]] == rows[barrier_places[1]] &&
                HasRow({rows[barrier_places[0]]}, "ColorAttachmentOutput") &&
                HasRow({rows[barrier_places[0]]}, "LateFragmentTests") &&
                HasRow({rows[barrier_places[0]]}, "DepthStencilAttachmentRead"),
            "the barrier does not order attachment writes against attachment reads and writes");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Where the address vertex shader of the non-push stream reads. Each read address makes another
  // shader, so it differs from the +0x100, +0x200 and +0x300 that other tests read.
  static constexpr uint64_t NonPushStreamReadOffset = DrawStreamReadOffset + 0x400;

  // The commands of the address vertex draw with no push descriptors allowed, so its descriptors go
  // in a set. With `hole` the draw is split into a hole and recorded at the flush.
  // `stage` is the stage that reads by address, so under the split layout the set can be the VS
  // set or the PS set.
  std::vector<std::string> RecordNonPushDrawStream(const char *name, bool hole,
                                                   AddressStage stage = AddressStage::Vertex) {
    AsyncDrawRig rig(*this, false);
    if (hole) CommandSchedulerTestAccess::AllowThisThread(rig.Scheduler());
    auto &graphics = rig.context->GetGraphics();
    const auto device_limit = graphics.max_push_descriptors;
    graphics.max_push_descriptors = 0;
    const auto replaced = UseAddressStage(rig, stage, NonPushStreamReadOffset);
    rig.PaintSentinels();
    CommandStream::Recorder::Reset(true);
    if (hole) {
      const auto source = PrepareHoleSource(rig);
      Require(name, "set draw", source.pipeline != nullptr && !source.pipeline->uses_push_descriptors,
              "the hole's draw does not use a descriptor set, so the test proves nothing");
      SplitHoleOf(rig, source);
    } else {
      Require(name, "blocking draw", rig.Draw(), "a draw with a ready pipeline was refused");
    }
    rig.Scheduler().Flush();
    auto rows = CommandStream::Recorder::rows;
    CommandStream::Recorder::active = false;
    RestoreAddressStage(rig, stage, replaced);
    graphics.max_push_descriptors = device_limit;
    return rows;
  }

  // A hole whose descriptors go in a set records the stream of the same draw met in order, with the
  // two barriers of a hole added, as HoleRecordedStream shows for pushed descriptors. The set is
  // bound inside the hole's buffer, between its barriers, and nothing is pushed. A hole that bound
  // its set in the segment, or not at all, fails.
  void CheckHoleRecordedStreamNonPush(AddressStage stage = AddressStage::Vertex) {
    constexpr const char *name = "HoleRecordedStreamNonPush";
    CommandStream::Scope hooks;
    const auto blocking = RecordNonPushDrawStream(name, false, stage);
    const auto rows = RecordNonPushDrawStream(name, true, stage);
    std::vector<std::string> without_barriers;
    std::vector<size_t> barrier_places;
    for (size_t index = 0; index < rows.size(); ++index) {
      if (IsStateBarrier(rows[index])) barrier_places.push_back(index);
      else without_barriers.push_back(rows[index]);
    }
    Require(name, "blocking binds a set",
            HasRow(blocking, "vkCmdBindDescriptorSets(") && !HasRow(blocking, "vkCmdPushDescriptorSet(") &&
                std::ranges::none_of(blocking, IsStateBarrier),
            "the blocking draw did not bind a descriptor set, so the hole cannot be compared");
    Require(name, "two barriers", barrier_places.size() == 2,
            "the hole's buffer did not start and end with one memory barrier");
    Require(name, "same stream", RenderExecutorTestAccess::RowMatches(name, without_barriers, blocking),
            "the hole recorded other commands than the same draw met in order");
    const auto bind = static_cast<size_t>(
        std::ranges::find_if(rows, [](const auto &row) { return row.rfind("vkCmdBindDescriptorSets(", 0) == 0; }) -
        rows.begin());
    Require(name, "set bound in the hole",
            barrier_places.size() == 2 && barrier_places[0] < bind && bind < barrier_places[1] &&
                !HasRow(rows, "vkCmdPushDescriptorSet("),
            "the hole's descriptor set was not bound between the hole's barriers");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A hole never clears. A draw that clears is told apart from one that does not by the check the
  // hole's rendering makes, which stops the process on a clear, so the test calls the check itself.
  // The hole of a clearing draw, with the clear stripped as the eager pass would, loads.
  void CheckHoleClearsStripped() {
    constexpr const char *name = "HoleClearsStripped";
    CommandStream::Scope hooks;
    AsyncDrawRig rig(*this, false);
    CommandSchedulerTestAccess::AllowThisThread(rig.Scheduler());
    rig.PaintSentinels();
    rig.SetDepthClear(true);
    auto source = PrepareHoleSource(rig, false);
    Require(name, "draw clears", RenderStateClears(source.view.rendering),
            "the clearing draw's view has no clear, so the test proves nothing");
    source.view.rendering.depth_stencil_attachment.depth_clear = false;
    Require(name, "check passes a stripped view", !RenderStateClears(source.view.rendering),
            "a view with no clear failed the check");
    source.view.rendering.depth_stencil_attachment.depth_clear = true;
    Require(name, "check sees a depth clear", RenderStateClears(source.view.rendering),
            "a view that clears the depth passed the check");
    source.view.rendering.depth_stencil_attachment.depth_clear = false;
    source.view.rendering.depth_stencil_attachment.stencil_clear = true;
    Require(name, "check sees a stencil clear", RenderStateClears(source.view.rendering),
            "a view that clears the stencil passed the check");
    source.view.rendering.depth_stencil_attachment.stencil_clear = false;
    SplitHoleOf(rig, source);
    CommandStream::Recorder::Reset(true);
    rig.Scheduler().Flush();
    const auto rows = CommandStream::Recorder::rows;
    CommandStream::Recorder::active = false;
    Require(name, "hole loads", HasRow(rows, "depth{") && HasRow(rows, "load=Load") &&
                !HasRow(rows, "load=Clear"),
            "the hole's rendering cleared an attachment");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A hole counts for the report with its verdict and the refusing targets it was made with, as one,
  // so a resolve row reads both from the same hole. A hole made without targets, the baseline, gives
  // none, so the targets come from the hole and not from the verdict or the critical flag.
  void CheckHoleVerdictTargets() {
    constexpr const char *name = "HoleVerdictTargets";
    AsyncDrawRig rig(*this, false);
    CommandSchedulerTestAccess::AllowThisThread(rig.Scheduler());
    rig.PaintSentinels();
    const auto source = PrepareHoleSource(rig);
    const auto targets = std::make_shared<Common::FirstRunReport::SkipTargets>();
    const DrawHole kept(source.view, source.key, "color_history", targets);
    const DrawHole bare(source.view, source.key, "color_history");
    const auto with = kept.Verdict(true);
    const auto without = bare.Verdict(true);
    Require(name, "kept", with.targets == targets.get() && with.critical && with.verdict != nullptr &&
                              std::strcmp(with.verdict, "color_history") == 0,
            "the hole's verdict lost its targets, its critical flag or its rule");
    Require(name, "bare", without.targets == nullptr && without.critical,
            "a hole made without targets counted with some");
    RenderExecutorTestAccess::ResetBindings(rig.context->GetRenderExecutor());
    rig.Scheduler().Flush();
    std::printf("[gpu]     %-32s ok\n", name);
  }

  void CheckDeferredHoles() {
    CheckHoleVerdictTargets();
    CheckHoleClearsStripped();
    CheckHoleBookkeeping();
    CheckHoleEmptySegments();
    CheckHoleSyncPoints();
    CheckHoleEarlyAdoption();
    CheckHoleRecordedStream();
    CheckHoleRecordedStreamNonPush();
  }

  void CheckDrawRecord() {
    CheckDeferredHoles();
    CheckDeferredResolve();
    CheckDeferredDraws();
    CheckDrawStream();
    CheckDrawRecordIgnoresRegisters();
    CheckDrawRecordIntoSideBuffer(true);
    CheckDrawRecordIntoSideBuffer(false);
    CheckDrawRecordVertexBinds(true);
    CheckDrawRecordVertexBinds(false);
  }
