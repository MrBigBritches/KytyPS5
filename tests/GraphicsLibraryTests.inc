  // Tests of the graphics pipeline libraries that --graphics-pipeline-library links into pipelines.
  // Included inside VulkanHarness, after SplitLayoutTests.inc. They build pipelines with
  // CreatePipelineInternal from keys and hand written shaders, once whole and once from libraries,
  // draw both, and compare what they draw.

  // How the pixel shader of a case interpolates its one input.
  enum class LibraryInterp { Smooth, Flat, NoPerspective };

  // The size of the targets, and the colors the targets are cleared to.
  static constexpr uint32_t LibraryExtent = 32;
  static constexpr std::array<float, 4> LibraryClear {0.25f, 0.5f, 0.75f, 0.5f};
  // Vertex data. A vertex is a position and a color of four floats each, and a case draws up to two
  // triangles of three vertices.
  static constexpr uint32_t LibraryVertexFloats = 8;
  static constexpr uint32_t LibraryTriangleVertices = 3;
  static constexpr uint32_t LibraryVertexStride = LibraryVertexFloats * sizeof(float);
  static constexpr uint32_t LibraryColorOffset = 4 * sizeof(float);
  // The most colors a case writes, and how many samples a multisampled case has.
  static constexpr uint32_t LibraryMaxOutputs = 3;
  static constexpr uint32_t LibraryMsaaSamples = 4;
  // The blend constant that the constant color factors read.
  static constexpr std::array<float, 4> LibraryBlendConstants {0.5f, 0.25f, 0.125f, 0.75f};

  // Everything a case varies. The key starts from LibraryBaseKey, and `state` changes it.
  struct LibraryCase {
    std::string name;
    LibraryInterp interp = LibraryInterp::Smooth;
    // Color targets written, from one to LibraryMaxOutputs.
    uint32_t outputs = 1;
    bool msaa = false;
    // Two overlapping triangles at different depths and a depth target.
    bool depth = false;
    // The dynamic depth test of the draws.
    bool depth_test = true;
    // No pixel shader and no color target, like a shadow map. The picture is the depth target.
    bool depth_only = false;
    // The alpha of the vertex colors.
    float alpha = 1.0f;
    std::function<void(GraphicsPipelineKey &)> state;
  };

  // The shaders and modules of the rig, which a case picks from.
  struct LibraryShaders {
    ShaderModule vertex;
    ShaderModule other_vertex;
    // By interpolation, with one output, and a smooth one with LibraryMaxOutputs.
    std::array<ShaderModule, 3> pixel;
    ShaderModule pixel_mrt;
  };

  static constexpr uint64_t LibraryVertexId = 0x71;
  static constexpr uint64_t LibraryOtherVertexId = 0x72;
  static constexpr uint64_t LibraryPixelIdBase = 0x80;

  // Assembles SPIR-V text and stops when it does not assemble.
  std::vector<u32> AssembleLibrarySpirv(const char *name, const std::string &text) {
    spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_2);
    std::vector<u32> binary;
    Require(name, "assemble", tools.Assemble(text, &binary), "the test shader did not assemble");
    return binary;
  }

  // A vertex shader that passes its position and its color on.
  std::string LibraryVertexText() {
    return R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint Vertex %main "main" %in_pos %in_color %per_vertex %out_color
      OpDecorate %in_pos Location 0
      OpDecorate %in_color Location 1
      OpDecorate %out_color Location 0
      OpDecorate %per_vertex_type Block
      OpMemberDecorate %per_vertex_type 0 BuiltIn Position
      %void = OpTypeVoid
      %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
      %vec4 = OpTypeVector %float 4
      %per_vertex_type = OpTypeStruct %vec4
      %ptr_in = OpTypePointer Input %vec4
      %ptr_out = OpTypePointer Output %vec4
      %ptr_per_vertex = OpTypePointer Output %per_vertex_type
      %int = OpTypeInt 32 1
      %int0 = OpConstant %int 0
      %in_pos = OpVariable %ptr_in Input
      %in_color = OpVariable %ptr_in Input
      %per_vertex = OpVariable %ptr_per_vertex Output
      %out_color = OpVariable %ptr_out Output
      %main = OpFunction %void None %fn
      %entry = OpLabel
      %pos = OpLoad %vec4 %in_pos
      %color = OpLoad %vec4 %in_color
      %pos_ptr = OpAccessChain %ptr_out %per_vertex %int0
      OpStore %pos_ptr %pos
      OpStore %out_color %color
      OpReturn
      OpFunctionEnd)";
  }

  // A pixel shader with one input that has the given decoration and `outputs` color outputs. The
  // first output is the color, the second swaps red and blue and the third halves it, so that
  // the targets of a case with several differ.
  std::string LibraryPixelText(const char *decoration, uint32_t outputs) {
    std::string interface_list, decorations, variables, stores;
    for (uint32_t index = 0; index < outputs; index++) {
      const auto name = "%out" + std::to_string(index);
      interface_list += " " + name;
      decorations += "OpDecorate " + name + " Location " + std::to_string(index) + "\n";
      variables += name + " = OpVariable %ptr_out Output\n";
    }
    stores += "OpStore %out0 %color\n";
    if (outputs > 1) {
      stores += "%swapped = OpVectorShuffle %vec4 %color %color 2 1 0 3\nOpStore %out1 %swapped\n";
    }
    if (outputs > 2) {
      stores += "%halved = OpVectorTimesScalar %vec4 %color %half\nOpStore %out2 %halved\n";
    }
    return std::string(R"(
      OpCapability Shader
      OpMemoryModel Logical GLSL450
      OpEntryPoint Fragment %main "main" %in_color)") + interface_list + R"(
      OpExecutionMode %main OriginUpperLeft
      OpDecorate %in_color Location 0
      )" + decoration + "\n" + decorations + R"(
      %void = OpTypeVoid
      %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
      %vec4 = OpTypeVector %float 4
      %half = OpConstant %float 0.5
      %ptr_in = OpTypePointer Input %vec4
      %ptr_out = OpTypePointer Output %vec4
      %in_color = OpVariable %ptr_in Input
      )" + variables + R"(
      %main = OpFunction %void None %fn
      %entry = OpLabel
      %color = OpLoad %vec4 %in_color
      )" + stores + R"(
      OpReturn
      OpFunctionEnd)";
  }

  // Makes the module of a stage. A pixel shader also gets the copy the split layout uses.
  ShaderModule MakeLibraryModule(const char *name, uint64_t id, ShaderType stage,
                                 const std::vector<u32> &spirv) {
    ShaderModule module;
    module.id = id;
    module.module = CreateShaderModule(name, spirv);
    module.info.stage = stage;
    if (stage == ShaderType::Pixel) {
      module.gpl_module = CreateShaderModule(name, MoveDescriptorSet(spirv, 0, SPLIT_PIXEL_SET));
    } else {
      module.vertex_formats = {vk::Format::eR32G32B32A32Sfloat, vk::Format::eR32G32B32A32Sfloat};
    }
    return module;
  }

  // Makes every module the library cases use.
  LibraryShaders MakeLibraryShaders(const char *name) {
    LibraryShaders shaders;
    const auto vertex = AssembleLibrarySpirv(name, LibraryVertexText());
    shaders.vertex = MakeLibraryModule(name, LibraryVertexId, ShaderType::Vertex, vertex);
    // The same code under another id, which is how two vertex shaders can have equal state.
    shaders.other_vertex = MakeLibraryModule(name, LibraryOtherVertexId, ShaderType::Vertex, vertex);
    const char *decorations[] = {"", "OpDecorate %in_color Flat", "OpDecorate %in_color NoPerspective"};
    for (size_t index = 0; index < shaders.pixel.size(); index++) {
      shaders.pixel[index] =
          MakeLibraryModule(name, LibraryPixelIdBase + index, ShaderType::Pixel,
                            AssembleLibrarySpirv(name, LibraryPixelText(decorations[index], 1)));
    }
    shaders.pixel_mrt = MakeLibraryModule(
        name, LibraryPixelIdBase + shaders.pixel.size(), ShaderType::Pixel,
        AssembleLibrarySpirv(name, LibraryPixelText("", LibraryMaxOutputs)));
    return shaders;
  }

  void DestroyLibraryShaders(LibraryShaders &shaders) {
    for (auto *module : {&shaders.vertex, &shaders.other_vertex, &shaders.pixel[0], &shaders.pixel[1],
                         &shaders.pixel[2], &shaders.pixel_mrt}) {
      m_device.destroyShaderModule(module->module, nullptr);
      m_device.destroyShaderModule(module->gpl_module, nullptr);
    }
  }

  // The key every case starts from. One vertex buffer of positions and colors, the color formats of
  // the mixed targets, and a mask that writes every channel.
  static GraphicsPipelineKey LibraryBaseKey(const LibraryCase &item) {
    GraphicsPipelineKey key;
    key.stage_ids[VERTEX_STAGE_INDEX] = LibraryVertexId;
    key.vertex_input.binding_count = 1;
    key.vertex_input.bindings[0] = {LibraryVertexStride, 0};
    key.vertex_input.attribute_count = 2;
    key.vertex_input.attributes[0] = {0, 0};
    key.vertex_input.attributes[1] = {LibraryColorOffset, 0};
    key.rendering.color_formats = {vk::Format::eR8G8B8A8Unorm, vk::Format::eR16G16B16A16Sfloat,
                                   vk::Format::eB8G8R8A8Srgb};
    key.rendering.color_count = item.outputs;
    if (item.depth) key.rendering.depth_format = vk::Format::eD32Sfloat;
    auto &params = key.static_params;
    params.topology = vk::PrimitiveTopology::eTriangleList;
    for (uint32_t index = 0; index < RENDER_COLOR_ATTACHMENTS_MAX; index++) {
      params.color_mask[index] = 0xf;
    }
    if (item.msaa) params.samples = LibraryMsaaSamples;
    return key;
  }

  // The pixel shader a case uses. Its id is part of the key.
  // A depth only case has none.
  ShaderModule *LibraryPixelOf(LibraryShaders &shaders, const LibraryCase &item) {
    if (item.depth_only) return nullptr;
    return item.outputs > 1 ? &shaders.pixel_mrt : &shaders.pixel[static_cast<size_t>(item.interp)];
  }

  // The four values of one vertex, in clip space. `ndc` is the position after the divide, and the
  // clip position is that times w, so that w changes how a smooth color is interpolated.
  static void AddLibraryVertex(std::vector<float> &data, std::array<float, 3> ndc, float w,
                               std::array<float, 3> rgb, float alpha) {
    data.insert(data.end(), {ndc[0] * w, ndc[1] * w, ndc[2] * w, w, rgb[0], rgb[1], rgb[2], alpha});
  }

  // The vertices of a case. The plain case has one triangle whose corners have other w and other
  // depths, one of them past the far plane, so that interpolation, clip space and depth clip all
  // show in the picture. The depth case has two overlapping triangles at constant depths.
  static std::vector<float> LibraryVertices(const LibraryCase &item) {
    std::vector<float> data;
    if (item.depth) {
      constexpr float near_depth = 0.3f, far_depth = 0.7f;
      AddLibraryVertex(data, {-0.8f, -0.8f, near_depth}, 1.0f, {1, 0, 0}, item.alpha);
      AddLibraryVertex(data, {0.8f, -0.8f, near_depth}, 1.0f, {0, 1, 0}, item.alpha);
      AddLibraryVertex(data, {0.0f, 0.8f, near_depth}, 1.0f, {0, 0, 1}, item.alpha);
      AddLibraryVertex(data, {-0.4f, -0.9f, far_depth}, 1.0f, {1, 1, 0}, item.alpha);
      AddLibraryVertex(data, {0.9f, -0.9f, far_depth}, 1.0f, {0, 1, 1}, item.alpha);
      AddLibraryVertex(data, {0.4f, 0.9f, far_depth}, 1.0f, {1, 0, 1}, item.alpha);
      return data;
    }
    AddLibraryVertex(data, {-0.8f, -0.8f, -0.25f}, 1.0f, {1, 0, 0}, item.alpha);
    AddLibraryVertex(data, {0.8f, -0.8f, 0.25f}, 3.0f, {0, 1, 0}, item.alpha);
    AddLibraryVertex(data, {0.0f, 0.8f, 1.25f}, 1.0f, {0, 0, 1}, item.alpha);
    return data;
  }

  // Everything made for one rendering of a case, destroyed together.
  struct LibraryTargets {
    std::array<Image, LibraryMaxOutputs> color;
    // The single sample images the multisampled targets resolve into, when the case is multisampled.
    std::array<Image, LibraryMaxOutputs> resolved;
    vk::Image depth_image = nullptr;
    vk::DeviceMemory depth_memory = nullptr;
    vk::ImageView depth_view = nullptr;
  };

  // Makes the color targets of a case, and its depth target.
  LibraryTargets MakeLibraryTargets(const char *name, const LibraryCase &item,
                                    const GraphicsPipelineKey &key) {
    LibraryTargets targets;
    const auto samples = item.msaa ? vulkan_sample_count(LibraryMsaaSamples) : vk::SampleCountFlagBits::e1;
    for (uint32_t index = 0; index < item.outputs; index++) {
      const auto format = key.rendering.color_formats[index];
      const auto words = (vk::blockSize(format) + 3u) / 4u;
      const auto make = [&](vk::SampleCountFlagBits count) {
        return CreateImageMips(name, LibraryExtent, LibraryExtent, format,
                               vk::ImageUsageFlagBits::eColorAttachment, {}, words,
                               vk::ImageLayout::eGeneral, vk::ImageType::e2D,
                               vk::ImageViewType::e2D, 1, 0, 0, count);
      };
      targets.color[index] = make(samples);
      if (item.msaa) targets.resolved[index] = make(vk::SampleCountFlagBits::e1);
    }
    if (item.depth) {
      vk::ImageCreateInfo create{};
      create.imageType = vk::ImageType::e2D;
      create.format = vk::Format::eD32Sfloat;
      create.extent = vk::Extent3D{LibraryExtent, LibraryExtent, 1};
      create.mipLevels = create.arrayLayers = 1;
      create.samples = vk::SampleCountFlagBits::e1;
      create.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eTransferSrc;
      RequireVk(name, "depth", m_device.createImage(&create, nullptr, &targets.depth_image), "createImage");
      vk::MemoryRequirements requirements = m_device.getImageMemoryRequirements(targets.depth_image);
      u32 type = 0;
      Require(name, "depth", FindMemoryType(requirements.memoryTypeBits, {}, &type), "no memory");
      vk::MemoryAllocateInfo allocate{.allocationSize = requirements.size, .memoryTypeIndex = type};
      RequireVk(name, "depth", m_device.allocateMemory(&allocate, nullptr, &targets.depth_memory), "allocate");
      RequireVk(name, "depth", m_device.bindImageMemory(targets.depth_image, targets.depth_memory, 0), "bind");
      vk::ImageViewCreateInfo view{};
      view.image = targets.depth_image;
      view.viewType = vk::ImageViewType::e2D;
      view.format = vk::Format::eD32Sfloat;
      view.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
      RequireVk(name, "depth", m_device.createImageView(&view, nullptr, &targets.depth_view), "view");
    }
    return targets;
  }

  void DestroyLibraryTargets(LibraryTargets &targets) {
    for (auto &image : targets.color) DestroyImage(&image);
    for (auto &image : targets.resolved) DestroyImage(&image);
    m_device.destroyImageView(targets.depth_view, nullptr);
    m_device.destroyImage(targets.depth_image, nullptr);
    m_device.freeMemory(targets.depth_memory, nullptr);
  }

  // Sets every dynamic state a pipeline of this device takes, with the depth test of the case.
  void SetLibraryDynamicState(vk::CommandBuffer cmd, const LibraryCase &item) {
    const vk::Viewport viewport{0, 0, static_cast<float>(LibraryExtent), static_cast<float>(LibraryExtent), 0, 1};
    const vk::Rect2D scissor{{0, 0}, {LibraryExtent, LibraryExtent}};
    cmd.setViewportWithCount(1, &viewport);
    cmd.setScissorWithCount(1, &scissor);
    cmd.setLineWidth(1);
    cmd.setCullMode(vk::CullModeFlagBits::eNone);
    cmd.setFrontFace(vk::FrontFace::eCounterClockwise);
    cmd.setDepthTestEnable(item.depth && item.depth_test);
    cmd.setDepthWriteEnable(item.depth && item.depth_test);
    cmd.setDepthCompareOp(vk::CompareOp::eLess);
    cmd.setDepthBoundsTestEnable(false);
    cmd.setDepthBiasEnable(false);
    cmd.setStencilTestEnable(false);
    cmd.setBlendConstants(LibraryBlendConstants.data());
    const std::array<vk::Bool32, LibraryMaxOutputs> write{true, true, true};
    // A pipeline with no color targets has no color write enable state.
    if (item.outputs != 0) cmd.setColorWriteEnableEXT(item.outputs, write.data());
    if (RuntimeContext().attachment_feedback_loop_enabled) {
      cmd.setAttachmentFeedbackLoopEnableEXT({});
    }
  }

  // The color attachments of a case. A multisampled target is cleared, resolved and then dropped.
  std::array<vk::RenderingAttachmentInfo, LibraryMaxOutputs>
  LibraryColorAttachments(const LibraryCase &item, const LibraryTargets &targets) {
    std::array<vk::RenderingAttachmentInfo, LibraryMaxOutputs> colors{};
    for (uint32_t index = 0; index < item.outputs; index++) {
      colors[index].imageView = targets.color[index].view;
      colors[index].imageLayout = vk::ImageLayout::eGeneral;
      colors[index].loadOp = vk::AttachmentLoadOp::eClear;
      colors[index].storeOp = item.msaa ? vk::AttachmentStoreOp::eDontCare : vk::AttachmentStoreOp::eStore;
      colors[index].clearValue.color.setFloat32(LibraryClear);
      if (item.msaa) {
        colors[index].resolveMode = vk::ResolveModeFlagBits::eAverage;
        colors[index].resolveImageView = targets.resolved[index].view;
        colors[index].resolveImageLayout = vk::ImageLayout::eGeneral;
      }
    }
    return colors;
  }

  // Moves the depth target into the layout the draws write it in.
  void BarrierLibraryDepth(vk::CommandBuffer cmd, const LibraryTargets &targets) {
    vk::ImageMemoryBarrier to_depth{};
    to_depth.newLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    to_depth.srcQueueFamilyIndex = to_depth.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_depth.image = targets.depth_image;
    to_depth.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
    to_depth.dstAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eEarlyFragmentTests,
                        {}, 0, nullptr, 0, nullptr, 1, &to_depth);
  }

  // Records the draws of a case, one per triangle, in a render pass over the targets.
  void RecordLibraryDraws(vk::CommandBuffer cmd, const LibraryCase &item, const LibraryTargets &targets,
                          vk::Pipeline pipeline, vk::Buffer vertices, uint32_t vertex_count) {
    auto colors = LibraryColorAttachments(item, targets);
    vk::RenderingAttachmentInfo depth{};
    depth.imageView = targets.depth_view;
    depth.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    depth.loadOp = vk::AttachmentLoadOp::eClear;
    // Only a depth only case reads the depth target back.
    depth.storeOp = item.depth_only ? vk::AttachmentStoreOp::eStore : vk::AttachmentStoreOp::eDontCare;
    depth.clearValue.depthStencil = vk::ClearDepthStencilValue{1.0f, 0};
    vk::RenderingInfo rendering{};
    rendering.renderArea.extent = vk::Extent2D{LibraryExtent, LibraryExtent};
    rendering.layerCount = 1;
    rendering.colorAttachmentCount = item.outputs;
    rendering.pColorAttachments = colors.data();
    rendering.pDepthAttachment = item.depth ? &depth : nullptr;
    if (item.depth) BarrierLibraryDepth(cmd, targets);
    cmd.beginRendering(rendering);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    SetLibraryDynamicState(cmd, item);
    const vk::DeviceSize offset = 0;
    cmd.bindVertexBuffers(0, 1, &vertices, &offset);
    for (uint32_t first = 0; first < vertex_count; first += LibraryTriangleVertices) {
      cmd.draw(LibraryTriangleVertices, 1, first, 0);
    }
    cmd.endRendering();
  }

  // Reads the depth target of a depth only case, one float per texel as a word.
  std::vector<u32> ReadLibraryDepth(const char *name, const LibraryTargets &targets) {
    constexpr size_t words = size_t{LibraryExtent} * LibraryExtent;
    auto staging = CreateHostBuffer(name, words * sizeof(u32), vk::BufferUsageFlagBits::eTransferDst, {});
    vk::CommandBuffer cmd = BeginCommands(name, "library depth");
    vk::ImageMemoryBarrier to_copy{};
    to_copy.oldLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    to_copy.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    to_copy.srcQueueFamilyIndex = to_copy.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    to_copy.image = targets.depth_image;
    to_copy.subresourceRange = {vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1};
    to_copy.srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
    to_copy.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eLateFragmentTests, vk::PipelineStageFlagBits::eTransfer, {},
                        0, nullptr, 0, nullptr, 1, &to_copy);
    vk::BufferImageCopy copy{};
    copy.imageSubresource = {vk::ImageAspectFlagBits::eDepth, 0, 0, 1};
    copy.imageExtent = vk::Extent3D{LibraryExtent, LibraryExtent, 1};
    cmd.copyImageToBuffer(targets.depth_image, vk::ImageLayout::eTransferSrcOptimal, staging.buffer, 1, &copy);
    EndSubmitAndFree(name, "library depth", cmd);
    auto read = ReadBuffer(name, staging, words);
    DestroyBuffer(&staging);
    return read;
  }

  // Draws a case with a pipeline and returns the words of every target, resolved if multisampled.
  std::vector<u32> RenderLibraryCase(const char *name, const LibraryCase &item,
                                     const GraphicsPipelineKey &key,
                                     const PipelineCache::Pipeline &pipeline) {
    auto targets = MakeLibraryTargets(name, item, key);
    const auto vertices = LibraryVertices(item);
    std::vector<u32> words(vertices.size());
    std::memcpy(words.data(), vertices.data(), vertices.size() * sizeof(float));
    auto buffer = CreateHostBuffer(name, words.size() * sizeof(u32),
                                   vk::BufferUsageFlagBits::eVertexBuffer, words);
    vk::CommandBuffer cmd = BeginCommands(name, "library draw");
    RecordLibraryDraws(cmd, item, targets, pipeline.pipeline, buffer.buffer,
                       static_cast<uint32_t>(vertices.size() / LibraryVertexFloats));
    EndSubmitAndFree(name, "library draw", cmd);
    std::vector<u32> picture;
    if (item.depth_only) picture = ReadLibraryDepth(name, targets);
    for (uint32_t index = 0; index < item.outputs; index++) {
      auto read = ReadImage(name, item.msaa ? &targets.resolved[index] : &targets.color[index]);
      picture.insert(picture.end(), read.begin(), read.end());
    }
    DestroyBuffer(&buffer);
    DestroyLibraryTargets(targets);
    return picture;
  }

  // Builds pipelines from keys, whole or from libraries, and owns what they need. The libraries are
  // the production type, and are destroyed after every pipeline linked from them.
  struct LibraryRig {
    VulkanHarness &harness;
    GraphicContext &graphics;
    DescriptorSetLayoutCache set_layouts;
    GraphicsLibraryCache libraries;
    LibraryShaders shaders;
    std::mutex mutex;
    std::vector<PipelineCache::Pipeline> pipelines;
    vk::DescriptorSetLayout previous_heap;

    explicit LibraryRig(VulkanHarness &owner)
        : harness(owner), graphics(owner.RuntimeContext()), set_layouts(owner.m_device),
          libraries(owner.m_device), shaders(owner.MakeLibraryShaders("GraphicsLibraries")),
          previous_heap(graphics.image_heap_layout) {
      // The harness context has no image heap, and libraries need a heap set. An empty layout stands
      // in for it, and the cache owns it.
      if (graphics.image_heap_layout == nullptr) graphics.image_heap_layout = set_layouts.GetOrCreate({}, {});
    }
    ~LibraryRig() {
      graphics.image_heap_layout = previous_heap;
      for (const auto &pipeline : pipelines) DestroyPipelineObjects(harness.m_device, pipeline);
      harness.DestroyLibraryShaders(shaders);
    }
    LibraryRig(const LibraryRig &) = delete;
    LibraryRig &operator=(const LibraryRig &) = delete;

    // Builds one pipeline. The caller has set the library mode of the context, and `use_libraries`
    // says whether the cache is given. Safe on several threads once the mode is set.
    PipelineCache::Pipeline Build(GraphicsPipelineKey key, ShaderModule &vertex, ShaderModule *pixel,
                                  bool use_libraries) {
      key.stage_ids[VERTEX_STAGE_INDEX] = vertex.id;
      key.stage_ids[PIXEL_STAGE_INDEX] = pixel != nullptr ? pixel->id : 0;
      PipelineCache::Pipeline pipeline;
      const std::array<const ShaderModule *, 4> stages{&vertex, nullptr, nullptr, pixel};
      CreatePipelineInternal(graphics, set_layouts, pipeline, key, stages, nullptr,
                             use_libraries ? &libraries : nullptr);
      std::scoped_lock lock(mutex);
      pipelines.push_back(pipeline);
      return pipeline;
    }

    // Builds the pipeline of a case, whole with the mode off or from libraries with it on.
    PipelineCache::Pipeline BuildCase(const LibraryCase &item, bool use_libraries,
                                      GraphicsPipelineKey *key_out = nullptr) {
      SplitLayoutScope mode(harness, use_libraries);
      auto key = LibraryBaseKey(item);
      if (item.state) item.state(key);
      if (key_out != nullptr) *key_out = key;
      return Build(key, shaders.vertex, harness.LibraryPixelOf(shaders, item), use_libraries);
    }
  };

  // Blend state for the first three targets. The first blends by source alpha, the second blends with
  // a constant factor and has separate alpha blending, and the third does not blend and masks a channel.
  static void LibraryMixedBlends(GraphicsPipelineKey &key) {
    using Factor = Prospero::BlendFactor;
    using Op = Prospero::BlendOp;
    const auto factor = [](Factor value) { return static_cast<uint8_t>(value); };
    const auto op = [](Op value) { return static_cast<uint8_t>(value); };
    auto &params = key.static_params;
    params.blend_enable[0] = true;
    params.color_srcblend[0] = factor(Factor::kSrcAlpha);
    params.color_destblend[0] = factor(Factor::kOneMinusSrcAlpha);
    params.color_comb_fcn[0] = op(Op::kAdd);
    params.blend_enable[1] = true;
    params.separate_alpha_blend[1] = true;
    params.color_srcblend[1] = factor(Factor::kDstColor);
    params.color_destblend[1] = factor(Factor::kConstantColor);
    params.color_comb_fcn[1] = op(Op::kReverseSubtract);
    params.alpha_srcblend[1] = factor(Factor::kOne);
    params.alpha_destblend[1] = factor(Factor::kOneMinusSrcAlpha);
    params.alpha_comb_fcn[1] = op(Op::kMax);
    params.color_mask[2] = 0x5;
  }

  // The cases of the equivalence check. The pairs that LibrarySensitivePairs lists differ in one thing
  // the pipeline state holds, so a library that lost that thing draws the other half of the pair.
  std::vector<LibraryCase> LibraryCases() {
    std::vector<LibraryCase> cases;
    const auto add = [&](LibraryCase item) { cases.push_back(std::move(item)); };
    add({.name = "smooth"});
    add({.name = "noperspective", .interp = LibraryInterp::NoPerspective});
    add({.name = "flat first", .interp = LibraryInterp::Flat});
    if (RuntimeContext().provoking_vertex_last_enabled) {
      add({.name = "flat last", .interp = LibraryInterp::Flat,
           .state = [](GraphicsPipelineKey &key) { key.static_params.provoking_vtx_last = true; }});
    }
    add({.name = "depth clip on",
         .state = [](GraphicsPipelineKey &key) { key.static_params.depth_clip_enable = true; }});
    add({.name = "depth clip off",
         .state = [](GraphicsPipelineKey &key) { key.static_params.depth_clip_enable = false; }});
    add({.name = "clip space 0 to 1",
         .state = [](GraphicsPipelineKey &key) { key.static_params.negative_one_to_one = false; }});
    add({.name = "clip space -1 to 1",
         .state = [](GraphicsPipelineKey &key) { key.static_params.negative_one_to_one = true; }});
    add({.name = "msaa and sample shading", .msaa = true,
         .state = [](GraphicsPipelineKey &key) { key.static_params.sample_shading_enable = true; }});
    add({.name = "msaa", .msaa = true});
    add({.name = "alpha to coverage on", .msaa = true, .alpha = 0.5f,
         .state = [](GraphicsPipelineKey &key) { key.static_params.alpha_to_coverage_enable = true; }});
    add({.name = "alpha to coverage off", .msaa = true, .alpha = 0.5f});
    add({.name = "mixed targets and blends", .outputs = LibraryMaxOutputs, .alpha = 0.5f,
         .state = [](GraphicsPipelineKey &key) { LibraryMixedBlends(key); }});
    add({.name = "mixed targets, one alpha blend field", .outputs = LibraryMaxOutputs, .alpha = 0.5f,
         .state = [](GraphicsPipelineKey &key) {
           LibraryMixedBlends(key);
           key.static_params.separate_alpha_blend[1] = false;
         }});
    add({.name = "depth test on", .depth = true, .depth_test = true});
    add({.name = "depth test off", .depth = true, .depth_test = false});
    // Like a shadow map. There is no pixel shader and no color target, so the pipeline has an empty PS
    // set, no color write enable state and nothing but the depth target to draw on.
    add({.name = "depth only", .outputs = 0, .depth = true, .depth_only = true});
    return cases;
  }

  // The pairs of cases that must draw different pictures, so each half is evidence the state counts.
  static std::vector<std::pair<const char *, const char *>> LibrarySensitivePairs() {
    return {{"smooth", "noperspective"}, {"flat first", "flat last"},
            {"depth clip on", "depth clip off"}, {"clip space 0 to 1", "clip space -1 to 1"},
            {"alpha to coverage on", "alpha to coverage off"},
            {"mixed targets and blends", "mixed targets, one alpha blend field"},
            {"depth test on", "depth test off"}};
  }

  // A picture with at least two different words was drawn on, and is not only the cleared target.
  static bool LibraryPictureHasContent(const std::vector<u32> &picture) {
    return std::ranges::any_of(picture, [&](u32 word) { return word != picture.front(); });
  }

  // Draws every case with a whole pipeline and, when asked, with a pipeline linked from libraries,
  // and compares. All the library pipelines share one cache, as in the emulator, so a library that is
  // reused for a case it does not fit draws the wrong picture. With `use_libraries` false only the
  // whole pipelines are drawn, which is the baseline of the validation run.
  void RunLibraryEquivalence(bool use_libraries) {
    constexpr const char *name = "GraphicsLibraryPixels";
    LibraryRig rig(*this);
    std::map<std::string, std::vector<u32>> whole_pictures;
    for (const auto &item : LibraryCases()) {
      GraphicsPipelineKey key;
      const auto whole = rig.BuildCase(item, false, &key);
      Require(name, item.name.c_str(), !whole.from_libraries && whole.pipeline != nullptr,
              "the whole pipeline was not built whole");
      const auto picture = RenderLibraryCase(name, item, key, whole);
      Require(name, item.name.c_str(), LibraryPictureHasContent(picture), "the case drew nothing");
      whole_pictures[item.name] = picture;
      if (!use_libraries) continue;
      const auto linked = rig.BuildCase(item, true);
      Require(name, item.name.c_str(), linked.from_libraries && linked.pipeline != nullptr,
              "the pipeline was not linked from libraries");
      Require(name, item.name.c_str(), RenderLibraryCase(name, item, key, linked) == picture,
              "a pipeline linked from libraries drew another picture than the whole pipeline");
    }
    for (const auto &[first, second] : LibrarySensitivePairs()) {
      if (!whole_pictures.contains(first) || !whole_pictures.contains(second)) continue;
      Require(name, first, whole_pictures[first] != whole_pictures[second],
              "the other half of the pair draws the same picture, so the pair proves nothing");
    }
    if (use_libraries) {
      Require(name, "sharing", rig.libraries.Counts(LibraryPart::PreRaster).hits != 0 &&
                                   rig.libraries.Counts(LibraryPart::VertexInput).hits != 0,
              "no library was reused, so the check did not exercise the shared cache");
    }
  }

  void CheckLibraryEquivalence() {
    RunLibraryEquivalence(true);
    std::printf("[gpu]     %-32s ok\n", "GraphicsLibraryPixels");
  }

  // Turns the first-run report on for the life of the scope, with nothing recorded yet, so a test can
  // compare what the report counted with what the cache counted. The configuration is put back after.
  struct LibraryReportScope {
    LibraryReportScope() {
      Config::Load({.printf_direction = Config::LogDirection::Silent, .first_run_report_enabled = true});
      Common::FirstRunReport::Global().Reset();
    }
    ~LibraryReportScope() {
      Common::FirstRunReport::Global().Reset();
      Config::Load({.printf_direction = Config::LogDirection::Silent});
    }
    LibraryReportScope(const LibraryReportScope &) = delete;
    LibraryReportScope &operator=(const LibraryReportScope &) = delete;
  };

  // Two pipelines with the same vertex shader share the pre-rasterization library, and the baseline of
  // two vertex shaders makes two. The other parts follow their own keys: the pixel shader splits the
  // fragment shader library and nothing else, and the equal vertex input and output state is made
  // once. The report counts the same builds as the cache does.
  void CheckLibraryDedup() {
    constexpr const char *name = "GraphicsLibraryDedup";
    LibraryReportScope report;
    LibraryRig rig(*this);
    SplitLayoutScope mode(*this, true);
    const auto key = LibraryBaseKey(LibraryCase{});
    auto &libraries = rig.libraries;
    const auto first = rig.Build(key, rig.shaders.vertex, &rig.shaders.pixel[0], true);
    const auto second = rig.Build(key, rig.shaders.vertex, &rig.shaders.pixel[1], true);
    Require(name, "linked", first.from_libraries && second.from_libraries,
            "a pipeline was not linked from libraries");
    Require(name, "one pre-rasterization library",
            libraries.Size(LibraryPart::PreRaster) == 1 && libraries.Counts(LibraryPart::PreRaster).built == 1,
            "two pipelines with the same vertex shader did not share a pre-rasterization library");
    Require(name, "two fragment libraries", libraries.Size(LibraryPart::Fragment) == 2,
            "two pixel shaders shared a fragment shader library");
    const auto third = rig.Build(key, rig.shaders.other_vertex, &rig.shaders.pixel[0], true);
    Require(name, "linked other", third.from_libraries, "a pipeline was not linked from libraries");
    Require(name, "two pre-rasterization libraries", libraries.Size(LibraryPart::PreRaster) == 2,
            "two vertex shaders shared a pre-rasterization library");
    Require(name, "the rest is shared",
            libraries.Size(LibraryPart::Fragment) == 2 && libraries.Size(LibraryPart::VertexInput) == 1 &&
                libraries.Size(LibraryPart::Output) == 1,
            "a vertex shader changed a library that does not read it");
    const auto stats = Common::FirstRunReport::Global().LibraryStats();
    constexpr std::array<LibraryPart, LIBRARY_PART_COUNT> parts{
        LibraryPart::VertexInput, LibraryPart::PreRaster, LibraryPart::Fragment, LibraryPart::Output};
    for (const auto part : parts) {
      const auto counted = libraries.Counts(part);
      const auto &reported = stats.libraries[static_cast<size_t>(part)];
      Require(name, "report", reported.built == counted.built && reported.hits == counted.hits,
              "the report counted other libraries than the cache");
    }
    Require(name, "report links", stats.links == libraries.Links() && stats.links == 3,
            "the report did not count the three links");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // The variants of the concurrency check: a vertex shader, a pixel shader and a blend state.
  struct LibraryVariant {
    ShaderModule *vertex;
    ShaderModule *pixel;
    bool blend;
  };

  // Every combination of the two vertex shaders, three pixel shaders and two blend states.
  static std::vector<LibraryVariant> LibraryVariants(LibraryShaders &shaders) {
    std::vector<LibraryVariant> variants;
    for (auto *vertex : {&shaders.vertex, &shaders.other_vertex}) {
      for (auto &pixel : shaders.pixel) {
        for (const bool blend : {false, true}) variants.push_back({vertex, &pixel, blend});
      }
    }
    return variants;
  }

  // The key of a variant. Blending changes only the fragment output.
  static GraphicsPipelineKey LibraryVariantKey(const LibraryVariant &variant) {
    auto key = LibraryBaseKey(LibraryCase{.alpha = 0.5f});
    if (variant.blend) {
      key.static_params.blend_enable[0] = true;
      key.static_params.color_srcblend[0] = static_cast<uint8_t>(Prospero::BlendFactor::kSrcAlpha);
      key.static_params.color_destblend[0] = static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrcAlpha);
    }
    return key;
  }

  // Threads build pipelines that share libraries, each walking the variants from another start. Every
  // library is built once and counted once, every thread gets a pipeline linked from libraries, and
  // the pictures of those pipelines equal the whole pipelines'. What else could pass the counts is
  // threads that built once each by luck, which is why the variants are many and the threads start
  // together. With `use_libraries` false the same threads build whole pipelines, which is the baseline.
  void RunLibraryConcurrency(bool use_libraries) {
    constexpr const char *name = "GraphicsLibraryThreads";
    constexpr size_t thread_count = 8;
    LibraryRig rig(*this);
    SplitLayoutScope mode(*this, use_libraries);
    const auto variants = LibraryVariants(rig.shaders);
    std::vector<std::vector<PipelineCache::Pipeline>> built(
        thread_count, std::vector<PipelineCache::Pipeline>(variants.size()));
    std::atomic<bool> go{false};
    std::vector<std::thread> threads;
    for (size_t thread = 0; thread < thread_count; thread++) {
      threads.emplace_back([&, thread] {
        while (!go.load()) std::this_thread::yield();
        for (size_t step = 0; step < variants.size(); step++) {
          const auto index = (step + thread) % variants.size();
          const auto &variant = variants[index];
          built[thread][index] = rig.Build(LibraryVariantKey(variant), *variant.vertex, variant.pixel, use_libraries);
        }
      });
    }
    go = true;
    for (auto &thread : threads) thread.join();
    for (const auto &row : built) {
      for (const auto &pipeline : row) {
        Require(name, "valid", pipeline.pipeline != nullptr && pipeline.from_libraries == use_libraries,
                "a pipeline was not made, or not in the way it was asked for");
      }
    }
    if (!use_libraries) return;
    CheckLibraryThreadCounts(rig, thread_count * variants.size());
    CheckLibraryThreadPictures(rig, variants, built[0]);
  }

  // Each distinct library was built once, which is one vertex input, two pre-rasterization (the vertex shaders),
  // three fragment (the pixel shaders) and two output (the blends). The rest of the requests were hits.
  void CheckLibraryThreadCounts(LibraryRig &rig, size_t requests) {
    constexpr const char *name = "GraphicsLibraryThreads";
    const std::array<std::pair<LibraryPart, size_t>, LIBRARY_PART_COUNT> expected{{
        {LibraryPart::VertexInput, 1}, {LibraryPart::PreRaster, 2}, {LibraryPart::Fragment, 3},
        {LibraryPart::Output, 2}}};
    for (const auto &[part, distinct] : expected) {
      const auto counts = rig.libraries.Counts(part);
      Require(name, "built once", counts.built == distinct && rig.libraries.Size(part) == distinct &&
                                      counts.failed == 0,
              "a library was built more or fewer times than it has distinct keys");
      Require(name, "hits", counts.built + counts.hits == requests,
              "the requests are not the builds and the hits");
    }
    Require(name, "links", rig.libraries.Links() == requests, "a pipeline was not linked once");
  }

  // The first thread's pipeline of each variant draws the picture of a whole pipeline.
  void CheckLibraryThreadPictures(LibraryRig &rig, const std::vector<LibraryVariant> &variants,
                                  const std::vector<PipelineCache::Pipeline> &pipelines) {
    constexpr const char *name = "GraphicsLibraryThreads";
    const LibraryCase item{.name = "threads", .alpha = 0.5f};
    for (size_t index = 0; index < variants.size(); index++) {
      const auto key = LibraryVariantKey(variants[index]);
      const auto actual = RenderLibraryCase(name, item, key, pipelines[index]);
      SplitLayoutScope off(*this, false);
      const auto whole = rig.Build(key, *variants[index].vertex, variants[index].pixel, false);
      Require(name, "picture", actual == RenderLibraryCase(name, item, key, whole),
              "a pipeline built by threads drew another picture than the whole pipeline");
    }
  }

  void CheckLibraryConcurrency() {
    RunLibraryConcurrency(true);
    std::printf("[gpu]     %-32s ok\n", "GraphicsLibraryThreads");
  }

  // The step the fault injector fails, as a LibraryStep number, or LibraryNoFailure for none.
  static constexpr int LibraryNoFailure = -1;
  static inline std::atomic<int> LibraryFailingStep{LibraryNoFailure};
  static bool LibraryFailsConfiguredStep(LibraryStep step) {
    return static_cast<int>(step) == LibraryFailingStep.load();
  }

  // When the driver refuses a library or the link, the pipeline is built whole instead, draws what
  // the whole pipeline draws, and is counted. The refusal is kept, so a second pipeline that needs the
  // same library does not try again and is counted too. The baseline is no refusal, which links and
  // counts nothing, so the counts can only come from the refusal.
  void CheckLibraryFallback() {
    constexpr const char *name = "GraphicsLibraryFallback";
    using Common::FirstRunReport::FallbackKind;
    const LibraryCase item{.name = "fallback"};
    for (const int step : {LibraryNoFailure, static_cast<int>(LibraryStep::VertexInput),
                           static_cast<int>(LibraryStep::PreRaster), static_cast<int>(LibraryStep::Fragment),
                           static_cast<int>(LibraryStep::Output), static_cast<int>(LibraryStep::Link)}) {
      LibraryRig rig(*this);
      GraphicsPipelineKey key;
      const auto whole = rig.BuildCase(item, false, &key);
      const auto reference = RenderLibraryCase(name, item, key, whole);
      LibraryFailingStep = step;
      SetLibraryFaultInjector(&LibraryFailsConfiguredStep);
      const auto first = rig.BuildCase(item, true);
      const auto second = rig.BuildCase(item, true);
      SetLibraryFaultInjector(nullptr);
      LibraryFailingStep = LibraryNoFailure;
      const bool refused = step != LibraryNoFailure;
      Require(name, "valid", first.pipeline != nullptr && second.pipeline != nullptr,
              "a refused pipeline was not built whole");
      Require(name, "how built", first.from_libraries != refused && second.from_libraries != refused,
              "the pipelines were not built the way the refusal says");
      Require(name, "counted", rig.libraries.Fallbacks(FallbackKind::CreateFailed) == (refused ? 2u : 0u),
              "the fallbacks were not counted, or were counted without a refusal");
      if (refused && step < static_cast<int>(LibraryStep::Link)) {
        const auto counts = rig.libraries.Counts(static_cast<LibraryPart>(step));
        Require(name, "kept", counts.failed == 1 && counts.built == 0,
                "a refused library was tried again, or counted as built");
      }
      Require(name, "picture", RenderLibraryCase(name, item, key, first) == reference,
              "a pipeline built after a refusal drew another picture");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // A pipeline that libraries do not cover is built whole and counted against its reason. The rectangle
  // list is built for real. Tessellation and mesh pipelines need shaders this check does not have, so
  // the decision that builds or skips the libraries is asked directly, with stages that name them. The
  // baseline is an eligible pipeline, and the same pipelines with the mode off, which count nothing.
  void CheckLibraryIneligible() {
    constexpr const char *name = "GraphicsLibraryIneligible";
    using Common::FirstRunReport::FallbackKind;
    LibraryReportScope report;
    LibraryRig rig(*this);
    CheckLibraryRectList(rig);
    SplitLayoutScope mode(*this, true);
    ShaderModule hull, evaluation;
    hull.id = 0x91;
    evaluation.id = 0x92;
    auto key = LibraryBaseKey(LibraryCase{});
    const auto decide = [&](GraphicsPipelineKey pipeline_key, ShaderModule &vertex, bool with_tess) {
      pipeline_key.stage_ids[VERTEX_STAGE_INDEX] = vertex.id;
      const std::array<const ShaderModule *, 4> stages{&vertex, with_tess ? &hull : nullptr,
                                                       with_tess ? &evaluation : nullptr, &rig.shaders.pixel[0]};
      return ShouldUseLibraries(rig.graphics, &rig.libraries, pipeline_key, stages);
    };
    Require(name, "eligible", decide(key, rig.shaders.vertex, false) &&
                                  rig.libraries.Fallbacks(FallbackKind::Tess) == 0 &&
                                  rig.libraries.Fallbacks(FallbackKind::Mesh) == 0,
            "an ordinary pipeline was refused libraries, or counted as a fallback");
    auto tess_key = key;
    tess_key.static_params.patch_control_points = 3;
    Require(name, "tessellation", !decide(tess_key, rig.shaders.vertex, true) &&
                                      rig.libraries.Fallbacks(FallbackKind::Tess) == 1,
            "a tessellation pipeline was not counted as tessellation");
    ShaderModule mesh = rig.shaders.vertex;
    mesh.info.stage = ShaderType::Mesh;
    Require(name, "mesh", !decide(key, mesh, false) && rig.libraries.Fallbacks(FallbackKind::Mesh) == 1,
            "a mesh pipeline was not counted as a mesh");
    Require(name, "no other reason",
            rig.libraries.Fallbacks(FallbackKind::RectList) == 1 && rig.libraries.Fallbacks(FallbackKind::CreateFailed) == 0 &&
                rig.libraries.Fallbacks(FallbackKind::Tess) == 1,
            "a fallback was counted under another reason");
    const auto stats = Common::FirstRunReport::Global().LibraryStats();
    Require(name, "report", stats.fallback_full[static_cast<size_t>(FallbackKind::RectList)] == 1 &&
                                stats.fallback_full[static_cast<size_t>(FallbackKind::Tess)] == 1 &&
                                stats.fallback_full[static_cast<size_t>(FallbackKind::Mesh)] == 1,
            "the report did not count the fallbacks the cache did");
    SplitLayoutScope off(*this, false);
    Require(name, "mode off", !decide(tess_key, rig.shaders.vertex, true) &&
                                  rig.libraries.Fallbacks(FallbackKind::Tess) == 1,
            "a fallback was counted with the library mode off");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Builds a rectangle list under the library mode, which is built whole and counted, makes no
  // library, and has the single set layout it has without the mode.
  void CheckLibraryRectList(LibraryRig &rig) {
    constexpr const char *name = "GraphicsLibraryIneligible";
    using Common::FirstRunReport::FallbackKind;
    ShaderModule vertex;
    vertex.id = 0x93;
    vertex.module = CompileSPV(GPU_BLIT_FS_TRIANGLE_SPV, m_device);
    vertex.info.stage = ShaderType::Vertex;
    ShaderModule pixel;
    pixel.id = 0x94;
    pixel.module = CompileSPV(GPU_VIDEO_OUT_OVERLAY_SPV, m_device);
    pixel.info.stage = ShaderType::Pixel;
    auto key = LibraryBaseKey(LibraryCase{});
    key.vertex_input = {};
    key.static_params.patch_control_points = 3;
    key.static_params.topology = vk::PrimitiveTopology::ePatchList;
    SplitLayoutScope mode(*this, true);
    const auto pipeline = rig.Build(key, vertex, &pixel, true);
    Require(name, "rectangle list built", pipeline.pipeline != nullptr && !pipeline.from_libraries &&
                                              !pipeline.uses_split_sets,
            "a rectangle list was not built whole with the single set layout");
    Require(name, "rectangle list counted", rig.libraries.Fallbacks(FallbackKind::RectList) == 1,
            "a rectangle list was not counted");
    Require(name, "no libraries", rig.libraries.Size(LibraryPart::PreRaster) == 0 &&
                                      rig.libraries.Size(LibraryPart::VertexInput) == 0,
            "a rectangle list made libraries");
    m_device.destroyShaderModule(vertex.module, nullptr);
    m_device.destroyShaderModule(pixel.module, nullptr);
  }

  // The bytes a part builder wrote, for each part, read through the pointers so a difference in
  // anything the builder produced shows.
  struct LibraryPartSnapshots {
    std::string vertex_input, pre_raster, fragment, output;
  };

  template <class Value>
  static void SnapshotAppend(std::string &out, const Value &value) {
    out.append(reinterpret_cast<const char *>(&value), sizeof(Value));
  }

  static void SnapshotMultisample(std::string &out, const vk::PipelineMultisampleStateCreateInfo &state) {
    SnapshotAppend(out, state.rasterizationSamples);
    SnapshotAppend(out, state.sampleShadingEnable);
    SnapshotAppend(out, state.minSampleShading);
    SnapshotAppend(out, state.alphaToCoverageEnable);
  }

  // Runs the four builders over a pipeline and keeps what they wrote.
  LibraryPartSnapshots SnapshotLibraryParts(GraphicContext &graphics, PartFixtures::Pipeline &pipeline) {
    LibraryPartSnapshots out;
    const auto stages = pipeline.Stages();
    const auto &key = pipeline.key;
    VertexInputParts vertex_input;
    FillVertexInputParts(vertex_input, key, pipeline.vertex);
    SnapshotAppend(out.vertex_input, vertex_input.bindings);
    SnapshotAppend(out.vertex_input, vertex_input.attributes);
    SnapshotAppend(out.vertex_input, vertex_input.vertex_input.vertexBindingDescriptionCount);
    SnapshotAppend(out.vertex_input, vertex_input.vertex_input.vertexAttributeDescriptionCount);
    SnapshotAppend(out.vertex_input, vertex_input.input_assembly.topology);
    SnapshotAppend(out.vertex_input, vertex_input.input_assembly.primitiveRestartEnable);
    PreRasterParts pre_raster;
    FillPreRasterParts(pre_raster, graphics, key, stages, HelperModules{});
    SnapshotAppend(out.pre_raster, pre_raster.stage_count);
    for (uint32_t index = 0; index < pre_raster.stage_count; index++) {
      SnapshotAppend(out.pre_raster, pre_raster.stages[index].stage);
      SnapshotAppend(out.pre_raster, pre_raster.stages[index].module);
    }
    SnapshotAppend(out.pre_raster, pre_raster.depth_clip_control.negativeOneToOne);
    SnapshotAppend(out.pre_raster, pre_raster.depth_clip.depthClipEnable);
    SnapshotAppend(out.pre_raster, pre_raster.provoking_vertex.provokingVertexMode);
    SnapshotAppend(out.pre_raster, pre_raster.rasterizer.polygonMode);
    SnapshotAppend(out.pre_raster, pre_raster.tessellation_state != nullptr);
    SnapshotAppend(out.pre_raster, pre_raster.tessellation.patchControlPoints);
    FragmentShaderParts fragment;
    FillFragmentShaderParts(fragment, key, stages[PIXEL_STAGE_INDEX]);
    SnapshotAppend(out.fragment, fragment.has_stage);
    SnapshotAppend(out.fragment, fragment.stage.module);
    SnapshotMultisample(out.fragment, fragment.multisampling);
    FragmentOutputParts output;
    FillFragmentOutputParts(output, key);
    SnapshotAppend(out.output, output.attachments);
    SnapshotAppend(out.output, output.blending.attachmentCount);
    SnapshotAppend(out.output, output.rendering.colorAttachmentCount);
    for (uint32_t index = 0; index < output.rendering.colorAttachmentCount; index++) {
      SnapshotAppend(out.output, output.rendering.pColorAttachmentFormats[index]);
    }
    SnapshotAppend(out.output, output.rendering.depthAttachmentFormat);
    SnapshotAppend(out.output, output.rendering.stencilAttachmentFormat);
    SnapshotMultisample(out.output, output.multisampling);
    return out;
  }

  // The part bits that differ between two snapshots.
  static unsigned LibraryChangedSnapshots(const LibraryPartSnapshots &before, const LibraryPartSnapshots &after) {
    return (before.vertex_input != after.vertex_input ? PartFixtures::VertexInputBit : 0u) |
           (before.pre_raster != after.pre_raster ? PartFixtures::PreRasterBit : 0u) |
           (before.fragment != after.fragment ? PartFixtures::FragmentBit : 0u) |
           (before.output != after.output ? PartFixtures::FragmentOutputBit : 0u);
  }

  // The part bits whose key differs between two pipelines.
  static unsigned LibraryChangedKeys(PartFixtures::Pipeline &before, PartFixtures::Pipeline &after) {
    const auto left = PartFixtures::MakePartKeys(before);
    const auto right = PartFixtures::MakePartKeys(after);
    return (!(left.vertex_input == right.vertex_input) ? PartFixtures::VertexInputBit : 0u) |
           (!(left.pre_raster == right.pre_raster) ? PartFixtures::PreRasterBit : 0u) |
           (!(left.fragment == right.fragment) ? PartFixtures::FragmentBit : 0u) |
           (!(left.output == right.output) ? PartFixtures::FragmentOutputBit : 0u);
  }

  // A part builder may only read what its part key names. Each field is flipped alone, and the parts
  // whose builder output changes must be among the parts whose key changes, because a builder that read
  // a field its key leaves out would let two pipelines share a library that is wrong for one. What
  // else could pass is builders that read nothing, so the fields below must also change the output of
  // exactly the parts that use them, and every part's output must change for some field.
  void CheckPartBuildersReadOnlyTheirKeys() {
    constexpr const char *name = "PartBuilders";
    auto &graphics = RuntimeContext();
    const std::map<std::string, unsigned> exact{
        {"topology", PartFixtures::VertexInputBit},
        {"vertex.vertex_formats[0]", PartFixtures::VertexInputBit},
        {"vertex_input.bindings.stride[0]", PartFixtures::VertexInputBit},
        {"depth_clip_enable", PartFixtures::PreRasterBit},
        {"negative_one_to_one", PartFixtures::PreRasterBit},
        {"polygon_mode", PartFixtures::PreRasterBit},
        {"samples", PartFixtures::FragmentBit | PartFixtures::FragmentOutputBit},
        {"alpha_to_coverage_enable", PartFixtures::FragmentBit | PartFixtures::FragmentOutputBit},
        {"blend_enable[0]", PartFixtures::FragmentOutputBit},
        {"color_srcblend[1]", PartFixtures::FragmentOutputBit},
        {"rendering.color_formats[0]", PartFixtures::FragmentOutputBit},
        {"rendering.depth_format", PartFixtures::FragmentOutputBit}};
    unsigned seen = 0, matched = 0;
    for (const auto &field : PartFixtures::EveryField()) {
      if (field.name == "provoking_vtx_last" && !graphics.provoking_vertex_last_enabled) continue;
      PartFixtures::Pipeline base, flipped;
      PartFixtures::FillBaseline(base);
      PartFixtures::FillBaseline(flipped);
      // The builders read a format for every attribute the key names, so the module has them all.
      base.vertex.vertex_formats.resize(ShaderVertexInputInfo::RES_MAX, vk::Format::eR32Sfloat);
      flipped.vertex.vertex_formats = base.vertex.vertex_formats;
      field.flip(flipped);
      const auto built = LibraryChangedSnapshots(SnapshotLibraryParts(graphics, base),
                                                 SnapshotLibraryParts(graphics, flipped));
      const auto keyed = LibraryChangedKeys(base, flipped);
      Require(name, field.name.c_str(), (built & ~keyed) == 0,
              "a part builder read a field that its part key leaves out");
      seen |= built;
      if (const auto found = exact.find(field.name); found != exact.end()) {
        Require(name, field.name.c_str(), built == found->second,
                "a field did not change the builders of exactly the parts that use it");
        matched++;
      }
    }
    Require(name, "every part", seen == (PartFixtures::VertexInputBit | PartFixtures::PreRasterBit |
                                         PartFixtures::FragmentBit | PartFixtures::FragmentOutputBit),
            "some part's builder never changed, so the check saw nothing of it");
    Require(name, "named fields", matched >= exact.size() - 1, "the named fields were not all found");
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // Which libraries read a dynamic state, as expected by the tests. This is a literal table and is not
  // derived from the code under test, so moving a state to another library fails here.
  struct ExpectedOwner {
    vk::DynamicState state;
    bool pre_raster, fragment, output;
  };

  // The expected owner of every dynamic state, and the states that depend on the device or the
  // targets, which are checked apart.
  static std::vector<ExpectedOwner> LibraryExpectedOwners() {
    using State = vk::DynamicState;
    return {
        {State::eViewportWithCount, true, false, false},
        {State::eScissorWithCount, true, false, false},
        {State::eLineWidth, true, false, false},
        {State::eCullMode, true, false, false},
        {State::eFrontFace, true, false, false},
        {State::eDepthBiasEnable, true, false, false},
        {State::eDepthBias, true, false, false},
        {State::eDepthTestEnable, false, true, false},
        {State::eDepthWriteEnable, false, true, false},
        {State::eDepthCompareOp, false, true, false},
        {State::eStencilTestEnable, false, true, false},
        {State::eStencilOp, false, true, false},
        {State::eStencilCompareMask, false, true, false},
        {State::eStencilReference, false, true, false},
        {State::eStencilWriteMask, false, true, false},
#if !defined(__APPLE__)
        // MoltenVK has no dynamic depth bounds, so the pipeline does not list them there.
        {State::eDepthBoundsTestEnable, false, true, false},
        {State::eDepthBounds, false, true, false},
#endif
        {State::eBlendConstants, false, false, true},
    };
  }

  // Every state of the table is in exactly the lists it should be in, and the lists hold nothing else
  // than the table and the two device dependent states.
  void CheckLibraryStateOwners(const DynamicStateParts &parts, uint32_t colors) {
    constexpr const char *name = "LibraryDynamicStates";
    const auto has = [](const std::vector<vk::DynamicState> &list, vk::DynamicState state) {
      return std::ranges::find(list, state) != list.end();
    };
    size_t expected_entries = 0;
    for (const auto &owner : LibraryExpectedOwners()) {
      Require(name, "owner",
              has(parts.pre_raster_states, owner.state) == owner.pre_raster &&
                  has(parts.fragment_states, owner.state) == owner.fragment &&
                  has(parts.output_states, owner.state) == owner.output,
              (std::string("a state is in the wrong libraries: ") + vk::to_string(owner.state)).c_str());
      expected_entries += (owner.pre_raster ? 1 : 0) + (owner.fragment ? 1 : 0) + (owner.output ? 1 : 0);
    }
    const bool color_write = has(parts.states, vk::DynamicState::eColorWriteEnableEXT);
    const bool feedback = RuntimeContext().attachment_feedback_loop_enabled;
    Require(name, "color write enable",
            has(parts.output_states, vk::DynamicState::eColorWriteEnableEXT) == (colors != 0 && color_write) &&
                !has(parts.pre_raster_states, vk::DynamicState::eColorWriteEnableEXT) &&
                !has(parts.fragment_states, vk::DynamicState::eColorWriteEnableEXT),
            "the color write enable state is in the wrong libraries or does not follow the targets");
    const size_t extra = (colors != 0 && color_write ? 1 : 0) + (feedback ? 2 : 0);
    Require(name, "no other states",
            parts.pre_raster_states.size() + parts.fragment_states.size() + parts.output_states.size() ==
                expected_entries + extra,
            "a library names a state that the table does not");
  }

  // The dynamic states of the libraries are cut from the whole list: together they name exactly the
  // states the whole pipeline has, a state only two parts read is in both, and no library names a state
  // outside its part. The whole list is the baseline, and a color write enable only appears with
  // targets.
  void CheckLibraryDynamicStates() {
    constexpr const char *name = "LibraryDynamicStates";
    auto &graphics = RuntimeContext();
    for (const uint32_t colors : {0u, 2u}) {
      GraphicsPipelineKey key;
      key.rendering.color_count = colors;
      DynamicStateParts parts;
      FillDynamicStateParts(parts, graphics, key);
      std::multiset<vk::DynamicState> whole(parts.states.begin(), parts.states.end());
      std::multiset<vk::DynamicState> cut;
      for (const auto *list : {&parts.pre_raster_states, &parts.fragment_states, &parts.output_states}) {
        cut.insert(list->begin(), list->end());
      }
      const bool feedback = graphics.attachment_feedback_loop_enabled;
      // The feedback loop state is the one state two parts read, and it is listed in each of them.
      if (feedback) cut.erase(cut.find(vk::DynamicState::eAttachmentFeedbackLoopEnableEXT));
      Require(name, "union", whole == cut, "the libraries do not name the states the whole pipeline does");
      const auto has = [](const std::vector<vk::DynamicState> &list, vk::DynamicState state) {
        return std::ranges::find(list, state) != list.end();
      };
      CheckLibraryStateOwners(parts, colors);
      Require(name, "feedback loop",
              !feedback || (has(parts.fragment_states, vk::DynamicState::eAttachmentFeedbackLoopEnableEXT) &&
                            has(parts.output_states, vk::DynamicState::eAttachmentFeedbackLoopEnableEXT) &&
                            !has(parts.pre_raster_states, vk::DynamicState::eAttachmentFeedbackLoopEnableEXT)),
              "the feedback loop state is not in the libraries that read it");
    }
    std::printf("[gpu]     %-32s ok\n", name);
  }

  // One library create info as the observer saw it: which part it is, and the dynamic states it names.
  struct ObservedLibrary {
    vk::GraphicsPipelineLibraryFlagsEXT part;
    bool has_dynamic_state = false;
    std::vector<vk::DynamicState> dynamic;
  };
  static inline std::mutex ObservedLibrariesMutex;
  static inline std::vector<ObservedLibrary> ObservedLibraries;

  // Keeps the part and dynamic states of a library create info. A whole pipeline has no library part
  // in its chain and is not kept.
  static void CaptureLibraryCreateInfo(const GraphicsPipelineKey &,
                                       const std::array<const ShaderModule *, 4> &,
                                       const vk::GraphicsPipelineCreateInfo &info) {
    ObservedLibrary seen;
    bool is_library = false;
    for (auto node = static_cast<const VkBaseInStructure *>(info.pNext); node != nullptr; node = node->pNext) {
      if (node->sType == VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT) {
        seen.part = vk::GraphicsPipelineLibraryFlagsEXT(
            reinterpret_cast<const VkGraphicsPipelineLibraryCreateInfoEXT *>(node)->flags);
        is_library = true;
      }
    }
    if (!is_library) return;
    if (info.pDynamicState != nullptr) {
      seen.has_dynamic_state = true;
      seen.dynamic.assign(info.pDynamicState->pDynamicStates,
                          info.pDynamicState->pDynamicStates + info.pDynamicState->dynamicStateCount);
    }
    std::scoped_lock lock(ObservedLibrariesMutex);
    ObservedLibraries.push_back(std::move(seen));
  }

  // The dynamic states that reach the driver with each library are the lists of FillDynamicStateParts for
  // the part, and the vertex input library names none. This is read from the create infos themselves,
  // because a device may treat these states as dynamic whichever library names them, so a drawing
  // check alone cannot tell a library that left its states out.
  void CheckLibraryCreateInfoDynamicStates() {
    constexpr const char *name = "LibraryDynamicStates";
    LibraryRig rig(*this);
    SplitLayoutScope mode(*this, true);
    auto key = LibraryBaseKey(LibraryCase{.outputs = 2});
    {
      std::scoped_lock lock(ObservedLibrariesMutex);
      ObservedLibraries.clear();
    }
    SetGraphicsCreateInfoObserver(&CaptureLibraryCreateInfo);
    const auto pipeline = rig.Build(key, rig.shaders.vertex, &rig.shaders.pixel[0], true);
    SetGraphicsCreateInfoObserver(nullptr);
    key.stage_ids[VERTEX_STAGE_INDEX] = rig.shaders.vertex.id;
    DynamicStateParts expected;
    FillDynamicStateParts(expected, rig.graphics, key);
    Require(name, "linked", pipeline.from_libraries, "the pipeline was not linked from libraries");
    std::scoped_lock lock(ObservedLibrariesMutex);
    Require(name, "four libraries", ObservedLibraries.size() == LIBRARY_PART_COUNT,
            "the observer did not see exactly one create info for each library");
    using Part = vk::GraphicsPipelineLibraryFlagBitsEXT;
    for (const auto &seen : ObservedLibraries) {
      const std::vector<vk::DynamicState> *wanted = nullptr;
      if (seen.part == Part::ePreRasterizationShaders) wanted = &expected.pre_raster_states;
      if (seen.part == Part::eFragmentShader) wanted = &expected.fragment_states;
      if (seen.part == Part::eFragmentOutputInterface) wanted = &expected.output_states;
      if (wanted == nullptr) {
        Require(name, "vertex input", seen.part == Part::eVertexInputInterface && !seen.has_dynamic_state,
                "the vertex input library names dynamic states");
        continue;
      }
      Require(name, "library states", !wanted->empty() && seen.dynamic == *wanted,
              "a library was created with other dynamic states than the ones of its part");
    }
    std::printf("[gpu]     %-32s ok (create infos)\n", name);
  }

  // All the tests of the libraries. The device part needs rasterization features and the library.
  void CheckGraphicsLibraries() {
    constexpr const char *name = "GraphicsLibraries";
    if (!m_rasterization_supported) {
      std::printf("[gpu]     %-32s skipped (no rasterization features)\n", name);
      return;
    }
    CheckPartBuildersReadOnlyTheirKeys();
    CheckLibraryDynamicStates();
    CheckLibraryCreateInfoDynamicStates();
    CheckLibraryDedup();
    CheckLibraryEquivalence();
    CheckLibraryConcurrency();
    CheckLibraryFallback();
    CheckLibraryIneligible();
    std::printf("[gpu]     %-32s ok\n", name);
  }
