  // Pins the Vulkan state that CreatePipelineInternal gives the driver for a graphics pipeline, so
  // that building it from per-part builders cannot change it. Included inside VulkanHarness, after
  // DeferredDrawTests.inc.

  // The text of each create info seen by the observer, in call order.
  static inline std::string CreateInfoCapture;

  // Appends one formatted line to a dump.
  static void DumpLine(std::string &dump, const char *format, ...) {
    char text[512];
    va_list arguments;
    va_start(arguments, format);
    std::vsnprintf(text, sizeof(text), format, arguments);
    va_end(arguments);
    dump += text;
    dump += '\n';
  }

  // Dumps the structs that hang off a pNext, in chain order. The order is part of the state, since
  // the driver is given the chain as it stands.
  static void DumpChain(std::string &dump, const char *owner, const void *next) {
    for (auto node = static_cast<const VkBaseInStructure *>(next); node != nullptr;
         node = node->pNext) {
      switch (node->sType) {
        case VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT:
          DumpLine(dump, "%s chain depth_clip_control negative_one_to_one=%u", owner,
                   reinterpret_cast<const VkPipelineViewportDepthClipControlCreateInfoEXT *>(node)
                       ->negativeOneToOne);
          break;
        case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT:
          DumpLine(dump, "%s chain depth_clip enable=%u", owner,
                   reinterpret_cast<const VkPipelineRasterizationDepthClipStateCreateInfoEXT *>(node)
                       ->depthClipEnable);
          break;
        case VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT:
          DumpLine(dump, "%s chain provoking_vertex mode=%d", owner,
                   static_cast<int>(
                       reinterpret_cast<const VkPipelineRasterizationProvokingVertexStateCreateInfoEXT *>(
                           node)->provokingVertexMode));
          break;
        case VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO: {
          const auto &rendering = *reinterpret_cast<const VkPipelineRenderingCreateInfo *>(node);
          std::string formats;
          for (uint32_t index = 0; index < rendering.colorAttachmentCount; ++index) {
            formats += std::to_string(rendering.pColorAttachmentFormats[index]) + ",";
          }
          DumpLine(dump, "%s chain rendering view_mask=%u colors=%u formats=%s depth=%d stencil=%d",
                   owner, rendering.viewMask, rendering.colorAttachmentCount, formats.c_str(),
                   static_cast<int>(rendering.depthAttachmentFormat),
                   static_cast<int>(rendering.stencilAttachmentFormat));
          break;
        }
        default: DumpLine(dump, "%s chain unknown sType=%d", owner, static_cast<int>(node->sType));
      }
    }
  }

  // Dumps the sections of a create info that do not hang off a pointer the dump can name.
  static void DumpStages(std::string &dump, const std::array<const ShaderModule *, 4> &stages,
                         const vk::GraphicsPipelineCreateInfo &info) {
    DumpLine(dump, "flags=%u stage_count=%u", static_cast<uint32_t>(info.flags), info.stageCount);
    for (uint32_t index = 0; index < info.stageCount; ++index) {
      const auto &stage = info.pStages[index];
      const char *label = "helper";
      const char *names[] = {"vs", "hs", "tes", "ps"};
      for (size_t slot = 0; slot < stages.size(); ++slot) {
        if (stages[slot] != nullptr && stages[slot]->module == stage.module) label = names[slot];
      }
      DumpLine(dump, "stage[%u] flags=%u stage=%u module=%s name=%s required_subgroup_size_pNext=%d",
               index, static_cast<uint32_t>(stage.flags), static_cast<uint32_t>(stage.stage), label,
               stage.pName, stage.pNext != nullptr);
    }
  }

  // Dumps vertex input, input assembly and tessellation state, which a mesh pipeline leaves null.
  static void DumpVertexState(std::string &dump, const vk::GraphicsPipelineCreateInfo &info) {
    if (info.pVertexInputState == nullptr) {
      DumpLine(dump, "vertex_input null");
    } else {
      const auto &input = *info.pVertexInputState;
      DumpLine(dump, "vertex_input flags=%u bindings=%u attributes=%u", static_cast<uint32_t>(input.flags),
               input.vertexBindingDescriptionCount, input.vertexAttributeDescriptionCount);
      for (uint32_t index = 0; index < input.vertexBindingDescriptionCount; ++index) {
        const auto &binding = input.pVertexBindingDescriptions[index];
        DumpLine(dump, "  binding %u stride=%u rate=%d", binding.binding, binding.stride,
                 static_cast<int>(binding.inputRate));
      }
      for (uint32_t index = 0; index < input.vertexAttributeDescriptionCount; ++index) {
        const auto &attribute = input.pVertexAttributeDescriptions[index];
        DumpLine(dump, "  attribute location=%u binding=%u format=%d offset=%u", attribute.location,
                 attribute.binding, static_cast<int>(attribute.format), attribute.offset);
      }
    }
    if (info.pInputAssemblyState == nullptr) {
      DumpLine(dump, "input_assembly null");
    } else {
      DumpLine(dump, "input_assembly topology=%d restart=%u",
               static_cast<int>(info.pInputAssemblyState->topology),
               info.pInputAssemblyState->primitiveRestartEnable);
    }
    if (info.pTessellationState == nullptr) {
      DumpLine(dump, "tessellation null");
    } else {
      DumpLine(dump, "tessellation flags=%u patch_control_points=%u",
               static_cast<uint32_t>(info.pTessellationState->flags),
               info.pTessellationState->patchControlPoints);
    }
  }

  // Dumps viewport, rasterization, multisample and depth-stencil state.
  static void DumpRasterState(std::string &dump, const vk::GraphicsPipelineCreateInfo &info) {
    const auto &viewport = *info.pViewportState;
    DumpLine(dump, "viewport viewports=%u scissors=%u pViewports=%d pScissors=%d",
             viewport.viewportCount, viewport.scissorCount, viewport.pViewports != nullptr,
             viewport.pScissors != nullptr);
    DumpChain(dump, "viewport", viewport.pNext);
    const auto &raster = *info.pRasterizationState;
    DumpLine(dump, "raster clamp=%u discard=%u polygon=%d cull=%u front=%d bias=%u line_width=%.1f",
             raster.depthClampEnable, raster.rasterizerDiscardEnable,
             static_cast<int>(raster.polygonMode), static_cast<uint32_t>(raster.cullMode),
             static_cast<int>(raster.frontFace), raster.depthBiasEnable, raster.lineWidth);
    DumpChain(dump, "raster", raster.pNext);
    const auto &multisample = *info.pMultisampleState;
    DumpLine(dump, "multisample samples=%u sample_shading=%u min_sample_shading=%.1f mask=%d a2c=%u a2one=%u",
             static_cast<uint32_t>(multisample.rasterizationSamples), multisample.sampleShadingEnable,
             multisample.minSampleShading, multisample.pSampleMask != nullptr,
             multisample.alphaToCoverageEnable, multisample.alphaToOneEnable);
    if (info.pDepthStencilState == nullptr) {
      DumpLine(dump, "depth_stencil null");
    } else {
      const auto &depth = *info.pDepthStencilState;
      DumpLine(dump, "depth_stencil test=%u write=%u compare=%d bounds=%u stencil=%u min=%.1f max=%.1f",
               depth.depthTestEnable, depth.depthWriteEnable, static_cast<int>(depth.depthCompareOp),
               depth.depthBoundsTestEnable, depth.stencilTestEnable, depth.minDepthBounds,
               depth.maxDepthBounds);
    }
  }

  // Dumps blend state and the dynamic states, which include every blend attachment.
  static void DumpOutputState(std::string &dump, const vk::GraphicsPipelineCreateInfo &info) {
    const auto &blend = *info.pColorBlendState;
    DumpLine(dump, "blend logic_op_enable=%u attachments=%u constants=%.1f,%.1f,%.1f,%.1f",
             blend.logicOpEnable, blend.attachmentCount, blend.blendConstants[0],
             blend.blendConstants[1], blend.blendConstants[2], blend.blendConstants[3]);
    for (uint32_t index = 0; index < blend.attachmentCount; ++index) {
      const auto &attachment = blend.pAttachments[index];
      DumpLine(dump, "  attachment mask=%u enable=%u color=%d,%d,%d alpha=%d,%d,%d",
               static_cast<uint32_t>(attachment.colorWriteMask), attachment.blendEnable,
               static_cast<int>(attachment.srcColorBlendFactor),
               static_cast<int>(attachment.dstColorBlendFactor),
               static_cast<int>(attachment.colorBlendOp),
               static_cast<int>(attachment.srcAlphaBlendFactor),
               static_cast<int>(attachment.dstAlphaBlendFactor),
               static_cast<int>(attachment.alphaBlendOp));
    }
    std::string states;
    for (uint32_t index = 0; index < info.pDynamicState->dynamicStateCount; ++index) {
      states += std::to_string(static_cast<int>(info.pDynamicState->pDynamicStates[index])) + ",";
    }
    DumpLine(dump, "dynamic count=%u states=%s", info.pDynamicState->dynamicStateCount, states.c_str());
  }

  // The observer. It cannot capture, so it appends to CreateInfoCapture.
  static void CaptureGraphicsCreateInfo(const GraphicsPipelineKey &,
                                        const std::array<const ShaderModule *, 4> &stages,
                                        const vk::GraphicsPipelineCreateInfo &info) {
    std::string &dump = CreateInfoCapture;
    DumpLine(dump, "--- pipeline");
    DumpChain(dump, "info", info.pNext);
    DumpStages(dump, stages, info);
    DumpVertexState(dump, info);
    DumpRasterState(dump, info);
    DumpOutputState(dump, info);
  }

  // A key for a plain pipeline, which each case changes. One color target and no depth.
  static GraphicsPipelineKey CreateInfoBaselineKey() {
    GraphicsPipelineKey key {};
    key.stage_ids = {1, 0, 0, 2};
    key.rendering.color_formats[0] = vk::Format::eR8G8B8A8Unorm;
    key.rendering.color_count = 1;
    key.static_params.topology = vk::PrimitiveTopology::eTriangleList;
    key.static_params.color_mask[0] = 0xf;
    for (auto &factor : key.static_params.color_srcblend) factor = static_cast<uint8_t>(Prospero::BlendFactor::kOne);
    for (auto &factor : key.static_params.color_destblend) factor = static_cast<uint8_t>(Prospero::BlendFactor::kZero);
    for (auto &factor : key.static_params.alpha_srcblend) factor = static_cast<uint8_t>(Prospero::BlendFactor::kOne);
    for (auto &factor : key.static_params.alpha_destblend) factor = static_cast<uint8_t>(Prospero::BlendFactor::kZero);
    return key;
  }

  // Everything the multisample, blend, depth, vertex input and raster fields of a key can do at
  // once, with three color targets whose blends all differ.
  static GraphicsPipelineKey CreateInfoBusyKey(bool provoking_last) {
    auto key = CreateInfoBaselineKey();
    key.rendering.color_formats = {vk::Format::eR8G8B8A8Unorm, vk::Format::eR16G16B16A16Sfloat,
                                   vk::Format::eB8G8R8A8Srgb};
    key.rendering.color_count = 3;
    key.rendering.depth_format = vk::Format::eD32SfloatS8Uint;
    key.rendering.stencil_format = vk::Format::eD32SfloatS8Uint;
    key.vertex_input.binding_count = 2;
    key.vertex_input.attribute_count = 3;
    key.vertex_input.bindings[0] = {24, 0};
    key.vertex_input.bindings[1] = {8, 1};
    key.vertex_input.attributes[0] = {0, 0};
    key.vertex_input.attributes[1] = {12, 0};
    key.vertex_input.attributes[2] = {4, 1};
    auto &params = key.static_params;
    params.negative_one_to_one = true;
    params.depth_clip_enable = false;
    params.topology = vk::PrimitiveTopology::eTriangleStrip;
    params.primitive_restart_enable = true;
    params.samples = 4;
    params.sample_shading_enable = true;
    params.alpha_to_coverage_enable = true;
    params.provoking_vtx_last = provoking_last;
    params.polygon_mode = vk::PolygonMode::eLine;
    using Factor = Prospero::BlendFactor;
    using Op = Prospero::BlendOp;
    const auto factor = [](Factor value) { return static_cast<uint8_t>(value); };
    const auto op = [](Op value) { return static_cast<uint8_t>(value); };
    params.color_mask[0] = 0x5;
    params.color_mask[1] = 0xf;
    params.color_mask[2] = 0x0;
    params.blend_enable[0] = true;
    params.blend_enable[1] = true;
    params.separate_alpha_blend[1] = true;
    params.color_srcblend[0] = factor(Factor::kSrcAlpha);
    params.color_destblend[0] = factor(Factor::kOneMinusSrcAlpha);
    params.color_comb_fcn[0] = op(Op::kAdd);
    params.color_srcblend[1] = factor(Factor::kDstColor);
    params.color_destblend[1] = factor(Factor::kConstantColor);
    params.color_comb_fcn[1] = op(Op::kReverseSubtract);
    params.alpha_srcblend[1] = factor(Factor::kOne);
    params.alpha_destblend[1] = factor(Factor::kSrc1Alpha);
    params.alpha_comb_fcn[1] = op(Op::kMax);
    // Target 2 has blending off and arbitrary leftover factors, which must still reach the state.
    params.color_srcblend[2] = factor(Factor::kOneMinusDstAlpha);
    params.color_destblend[2] = factor(Factor::kSrcAlphaSaturate);
    params.color_comb_fcn[2] = op(Op::kMin);
    return key;
  }

  // Builds the pipelines of the golden cases with the observer on, and returns what it saw. The
  // pipelines are real, so the driver accepts every state in the dump.
  std::string CaptureCreateInfoCases() {
    auto &graphics = RuntimeContext();
    const bool feedback_loop = graphics.attachment_feedback_loop_enabled;
    // The dynamic state list differs with this device feature, and the golden has it off.
    graphics.attachment_feedback_loop_enabled = false;
    ShaderModule vertex;
    vertex.id = 1;
    vertex.module = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, m_device);
    vertex.info.stage = ShaderType::Vertex;
    vertex.vertex_formats = {vk::Format::eR32G32B32Sfloat, vk::Format::eR16G16Unorm,
                             vk::Format::eR8G8B8A8Unorm};
    ShaderModule pixel;
    pixel.id = 2;
    pixel.module = CompileSPV(GPU_VIDEO_OUT_OVERLAY_SPV, m_device);
    pixel.info.stage = ShaderType::Pixel;

    CreateInfoCapture.clear();
    SetGraphicsCreateInfoObserver(&CaptureGraphicsCreateInfo);
    struct Case {
      GraphicsPipelineKey key;
      bool with_pixel;
    };
    auto depth_only = CreateInfoBaselineKey();
    depth_only.stage_ids[3] = 0;
    depth_only.rendering.color_count = 0;
    depth_only.rendering.depth_format = vk::Format::eD32Sfloat;
    auto rect_list = CreateInfoBaselineKey();
    rect_list.static_params.patch_control_points = 3;
    rect_list.static_params.topology = vk::PrimitiveTopology::ePatchList;
    for (const auto &item : {Case {CreateInfoBaselineKey(), true},
                             Case {CreateInfoBusyKey(graphics.provoking_vertex_last_enabled), true},
                             Case {depth_only, false}, Case {rect_list, true}}) {
      PipelineCache::Pipeline pipeline;
      const std::array<const ShaderModule *, 4> stages {&vertex, nullptr, nullptr,
                                                        item.with_pixel ? &pixel : nullptr};
      CreatePipelineInternal(graphics, pipeline, item.key, stages, nullptr);
      m_device.destroyPipeline(pipeline.pipeline, nullptr);
      m_device.destroyPipelineLayout(pipeline.pipeline_layout, nullptr);
      m_device.destroyDescriptorSetLayout(pipeline.descriptor_set_layout, nullptr);
    }
    SetGraphicsCreateInfoObserver(nullptr);
    graphics.attachment_feedback_loop_enabled = feedback_loop;
    m_device.destroyShaderModule(vertex.module, nullptr);
    m_device.destroyShaderModule(pixel.module, nullptr);
    return CreateInfoCapture;
  }

  // The golden text is the dump the code before the split gave, captured from it with the same
  // observer. Anything the split moved, dropped or reordered in the state shows as a difference.
  // What else could make this pass is a change that the dump does not print. It prints every field
  // the builders set, and the pNext chains in order, but not the pipeline layout.
  void CheckGraphicsCreateInfo() {
    constexpr const char *name = "GraphicsCreateInfo";
    if (!m_rasterization_supported) {
      std::printf("[gpu]     %-32s skipped (no rasterization features)\n", name);
      return;
    }
    const auto actual = CaptureCreateInfoCases();
    if (actual != GraphicsCreateInfoGolden()) {
      std::ofstream("graphics_create_info_actual.txt") << actual;
    }
    Require(name, "monolithic create info", actual == GraphicsCreateInfoGolden(),
            "the create info of a graphics pipeline differs from the one before the split, "
            "see graphics_create_info_actual.txt");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // The create info dump of the code before the split, for the four cases of CaptureCreateInfoCases.
  static std::string GraphicsCreateInfoGolden() {
    return R"GOLDEN(--- pipeline
info chain rendering view_mask=0 colors=1 formats=37, depth=0 stencil=0
flags=0 stage_count=2
stage[0] flags=0 stage=1 module=vs name=main required_subgroup_size_pNext=0
stage[1] flags=0 stage=16 module=ps name=main required_subgroup_size_pNext=0
vertex_input flags=0 bindings=0 attributes=0
input_assembly topology=3 restart=0
tessellation null
viewport viewports=0 scissors=0 pViewports=0 pScissors=0
viewport chain depth_clip_control negative_one_to_one=0
raster clamp=1 discard=0 polygon=0 cull=0 front=0 bias=0 line_width=0.0
raster chain provoking_vertex mode=0
raster chain depth_clip enable=1
multisample samples=1 sample_shading=0 min_sample_shading=1.0 mask=0 a2c=0 a2one=0
depth_stencil null
blend logic_op_enable=0 attachments=1 constants=0.0,0.0,0.0,0.0
  attachment mask=15 enable=0 color=1,0,0 alpha=1,0,0
dynamic count=19 states=1000267003,1000267004,2,1000267000,1000267001,1000267006,1000267007,1000267008,1000377002,3,1000267010,1000267011,6,8,7,4,1000267009,5,1000381000,
--- pipeline
info chain rendering view_mask=0 colors=3 formats=37,97,50, depth=130 stencil=130
flags=0 stage_count=2
stage[0] flags=0 stage=1 module=vs name=main required_subgroup_size_pNext=0
stage[1] flags=0 stage=16 module=ps name=main required_subgroup_size_pNext=0
vertex_input flags=0 bindings=2 attributes=3
  binding 0 stride=24 rate=0
  binding 1 stride=8 rate=1
  attribute location=0 binding=0 format=106 offset=0
  attribute location=1 binding=0 format=77 offset=12
  attribute location=2 binding=1 format=37 offset=4
input_assembly topology=4 restart=1
tessellation null
viewport viewports=0 scissors=0 pViewports=0 pScissors=0
viewport chain depth_clip_control negative_one_to_one=1
raster clamp=1 discard=0 polygon=1 cull=0 front=0 bias=0 line_width=0.0
raster chain provoking_vertex mode=1
raster chain depth_clip enable=0
multisample samples=4 sample_shading=1 min_sample_shading=1.0 mask=0 a2c=1 a2one=0
depth_stencil test=0 write=0 compare=0 bounds=0 stencil=0 min=0.0 max=0.0
blend logic_op_enable=0 attachments=3 constants=0.0,0.0,0.0,0.0
  attachment mask=5 enable=1 color=6,7,0 alpha=6,7,0
  attachment mask=15 enable=1 color=4,10,2 alpha=1,17,4
  attachment mask=0 enable=0 color=9,14,3 alpha=9,14,3
dynamic count=19 states=1000267003,1000267004,2,1000267000,1000267001,1000267006,1000267007,1000267008,1000377002,3,1000267010,1000267011,6,8,7,4,1000267009,5,1000381000,
--- pipeline
info chain rendering view_mask=0 colors=0 formats= depth=126 stencil=0
flags=0 stage_count=1
stage[0] flags=0 stage=1 module=vs name=main required_subgroup_size_pNext=0
vertex_input flags=0 bindings=0 attributes=0
input_assembly topology=3 restart=0
tessellation null
viewport viewports=0 scissors=0 pViewports=0 pScissors=0
viewport chain depth_clip_control negative_one_to_one=0
raster clamp=1 discard=0 polygon=0 cull=0 front=0 bias=0 line_width=0.0
raster chain provoking_vertex mode=0
raster chain depth_clip enable=1
multisample samples=1 sample_shading=0 min_sample_shading=1.0 mask=0 a2c=0 a2one=0
depth_stencil test=0 write=0 compare=0 bounds=0 stencil=0 min=0.0 max=0.0
blend logic_op_enable=0 attachments=0 constants=0.0,0.0,0.0,0.0
dynamic count=18 states=1000267003,1000267004,2,1000267000,1000267001,1000267006,1000267007,1000267008,1000377002,3,1000267010,1000267011,6,8,7,4,1000267009,5,
--- pipeline
info chain rendering view_mask=0 colors=1 formats=37, depth=0 stencil=0
flags=0 stage_count=4
stage[0] flags=0 stage=1 module=vs name=main required_subgroup_size_pNext=0
stage[1] flags=0 stage=16 module=ps name=main required_subgroup_size_pNext=0
stage[2] flags=0 stage=2 module=helper name=main required_subgroup_size_pNext=0
stage[3] flags=0 stage=4 module=helper name=main required_subgroup_size_pNext=0
vertex_input flags=0 bindings=0 attributes=0
input_assembly topology=10 restart=0
tessellation flags=0 patch_control_points=3
viewport viewports=0 scissors=0 pViewports=0 pScissors=0
viewport chain depth_clip_control negative_one_to_one=0
raster clamp=1 discard=0 polygon=0 cull=0 front=0 bias=0 line_width=0.0
raster chain provoking_vertex mode=0
raster chain depth_clip enable=1
multisample samples=1 sample_shading=0 min_sample_shading=1.0 mask=0 a2c=0 a2one=0
depth_stencil null
blend logic_op_enable=0 attachments=1 constants=0.0,0.0,0.0,0.0
  attachment mask=15 enable=0 color=1,0,0 alpha=1,0,0
dynamic count=19 states=1000267003,1000267004,2,1000267000,1000267001,1000267006,1000267007,1000267008,1000377002,3,1000267010,1000267011,6,8,7,4,1000267009,5,1000381000,
)GOLDEN";
  }
