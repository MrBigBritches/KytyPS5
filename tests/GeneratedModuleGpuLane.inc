  // Gives a test the GPU thread of a fresh GPU, for the checks of what the pipeline cache does on
  // that thread. The body runs on the calling thread and sends work to the GPU thread through
  // gpu.SendCommandSync, so the thread that makes a cache and the thread that looks shaders up in it
  // can differ, as they do in the emulator. Included inside VulkanHarness, after
  // GraphicsLibraryTests.inc.
  void WithGpuLane(const std::function<void(GuestGpu &)> &body) {
    EnsureRuntimeContext();
    auto &context = Renderer();
    context.InitializeGpu(nullptr);
    body(context.GetGpu());
    context.ShutdownGpu();
  }
