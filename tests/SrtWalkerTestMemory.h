#ifndef KYTY_TESTS_SRTWALKERTESTMEMORY_H_
#define KYTY_TESTS_SRTWALKERTESTMEMORY_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

#include <xbyak/xbyak.h>

#include <csignal>
#include <cstddef>
#include <cstdlib>
#include <iterator>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <ucontext.h>
#elif defined(__APPLE__)
#include <sys/ucontext.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

// Serves SRT walker loads from fake guest addresses: a walker load that faults completes with
// the test reader's DWORD, or with zero like the emulator when the reader fails or is null.
//
// It routes a fault in the same order as the emulator's exception handler. A non-faulting walk
// skips the load and never reaches the optional GPU fault stand-in. Any other walk offers the
// fault to the stand-in first, and a true answer retries the load, like HandleGpuFault does
// after resolving a watched page. The stand-in is process wide so it also serves walks run on
// other threads.
class FakeGuestMemory {
public:
  using GpuFaultStandIn = bool (*)(void *userdata, uint64_t fault_address);

  explicit FakeGuestMemory(Libs::Graphics::ShaderRecompiler::IR::SrtMemoryReader reader = nullptr,
                           void *userdata = nullptr, GpuFaultStandIn gpu_fault = nullptr,
                           void *gpu_userdata = nullptr) {
    Current() = {reader, userdata};
    GpuHook() = {gpu_fault, gpu_userdata};
#if defined(__linux__) || defined(__APPLE__)
    struct sigaction action {};
    action.sa_sigaction = Handle;
    action.sa_flags = SA_SIGINFO;
    for (size_t index = 0; index < std::size(kSignals); ++index)
      sigaction(kSignals[index], &action, &m_previous[index]);
#elif defined(_WIN32)
    m_handler = AddVectoredExceptionHandler(1, Handle);
#endif
  }
  ~FakeGuestMemory() {
#if defined(__linux__) || defined(__APPLE__)
    for (size_t index = 0; index < std::size(kSignals); ++index)
      sigaction(kSignals[index], &m_previous[index], nullptr);
#elif defined(_WIN32)
    RemoveVectoredExceptionHandler(m_handler);
#endif
    Current() = {};
    GpuHook() = {};
  }
  FakeGuestMemory(const FakeGuestMemory &) = delete;
  FakeGuestMemory &operator=(const FakeGuestMemory &) = delete;

private:
  using Reader = std::pair<Libs::Graphics::ShaderRecompiler::IR::SrtMemoryReader, void *>;
  static Reader &Current() {
    static thread_local Reader reader;
    return reader;
  }
  static uint32_t Read(uint64_t address) {
    uint32_t word = 0;
    const auto [reader, userdata] = Current();
    return reader != nullptr && reader(userdata, address, {&word, 1}) ? word : 0u;
  }
  using Hook = std::pair<GpuFaultStandIn, void *>;
  static Hook &GpuHook() {
    static Hook hook;
    return hook;
  }
  struct Resolution {
    uint32_t length = 0; // Bytes to step over, zero when the load retries.
    uint32_t value = 0;  // What the skipped load reads.
  };
  // Decides one faulting walker load the way the emulator's handler does.
  static Resolution Resolve(uint64_t rip, uint64_t fault_address, uint64_t load_address) {
    using namespace Libs::Graphics::ShaderRecompiler::IR;
    // The same routing decision the emulator's handler calls. Only the NotWalker case differs,
    // where the stand-in plays HandleGpuFault.
    uint32_t routed_length = 0;
    const auto route = SrtWalkerRouteFault(rip, routed_length);
    if (route == SrtFaultRoute::Fatal) std::abort();
    if (route == SrtFaultRoute::SkipLoad) return {routed_length, 0u};
    const auto [gpu_fault, gpu_userdata] = GpuHook();
    if (gpu_fault != nullptr && gpu_fault(gpu_userdata, fault_address)) return {};
    const auto length = SrtWalkerLoadLength(rip);
    if (length == 0) std::abort();
    return {length, Read(load_address)};
  }
#if defined(__linux__) || defined(__APPLE__)
#if defined(__APPLE__)
  // macOS raises SIGBUS for some protection faults and SIGSEGV for others.
  static constexpr int kSignals[] = {SIGSEGV, SIGBUS};
#else
  static constexpr int kSignals[] = {SIGSEGV};
#endif
  static void Handle(int, siginfo_t *signal_info, void *context) {
    auto *native = static_cast<ucontext_t *>(context);
#if defined(__APPLE__)
    auto &rip = native->uc_mcontext->__ss.__rip;
    auto &rax = native->uc_mcontext->__ss.__rax;
    const auto r10 = native->uc_mcontext->__ss.__r10;
#else
    auto &rip = native->uc_mcontext.gregs[REG_RIP];
    auto &rax = native->uc_mcontext.gregs[REG_RAX];
    const auto r10 = native->uc_mcontext.gregs[REG_R10];
#endif
    const auto resolution =
        Resolve(rip, reinterpret_cast<uint64_t>(signal_info->si_addr), r10);
    if (resolution.length == 0) return;
    rax = resolution.value;
    rip += resolution.length;
  }
  struct sigaction m_previous[std::size(kSignals)] {};
#elif defined(_WIN32)
  static LONG CALLBACK Handle(EXCEPTION_POINTERS *info) {
    auto &context = *info->ContextRecord;
    // Checked first because routing a fault has side effects, it counts a skipped load.
    if (info->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION ||
        Libs::Graphics::ShaderRecompiler::IR::SrtWalkerLoadLength(context.Rip) == 0)
      return EXCEPTION_CONTINUE_SEARCH;
    const auto resolution =
        Resolve(context.Rip, info->ExceptionRecord->ExceptionInformation[1], context.R10);
    if (resolution.length != 0) {
      context.Rax = resolution.value;
      context.Rip += resolution.length;
    }
    return EXCEPTION_CONTINUE_EXECUTION;
  }
  void *m_handler = nullptr;
#endif
};

// Evaluates one value through a walker compiled for it, keeping the plan's resource walker.
inline bool EvaluateSrtValue(Libs::Graphics::ShaderRecompiler::IR::ResourcePlan &plan,
                             const Libs::Graphics::ShaderRecompiler::IR::SrtRuntime &runtime,
                             Libs::Graphics::ShaderRecompiler::IR::Value value, uint32_t &result) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  const auto *inst = value.Resolve().TryInstruction();
  const auto output = inst != nullptr ? inst->WalkOutput() : UINT32_MAX;
  auto code = std::move(plan.srt_code);
  auto scratch = std::move(plan.srt_scratch);
  const auto user_data = plan.srt_user_data;
  const auto captures = plan.srt_captures;
  CompileSrtWalker(plan, {&value, 1});
  std::vector<uint32_t> flat;
  const bool known = WalkSrt(plan, runtime, flat, nullptr) && WalkedValue(plan, value, result);
  plan.srt_code = std::move(code);
  plan.srt_scratch = std::move(scratch);
  plan.srt_user_data = user_data;
  plan.srt_captures = captures;
  if (inst != nullptr) inst->SetWalkOutput(output);
  return known;
}

// Walks the plan's resources and reads back one descriptor.
inline bool EvaluateSrtDescriptor(const Libs::Graphics::ShaderRecompiler::IR::ResourcePlan &plan,
                                  const Libs::Graphics::ShaderRecompiler::IR::SrtRuntime &runtime,
                                  uint32_t source,
                                  Libs::Graphics::ShaderRecompiler::IR::DescriptorValue &result) {
  using namespace Libs::Graphics::ShaderRecompiler::IR;
  std::vector<uint32_t> flat;
  return WalkSrt(plan, runtime, flat, nullptr) && WalkedDescriptor(plan, source, result);
}

#endif // KYTY_TESTS_SRTWALKERTESTMEMORY_H_
