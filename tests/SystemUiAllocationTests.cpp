// Holds the system UI to its budget of heap allocations on a steady frame. The UI draws on the render
// thread every frame, and a frame's work settles once its storage has grown, so a frame that keeps
// allocating is a cost worth stopping. The count is exact on a given standard library, so unlike a time
// it can't flake. This has its own executable because replacing operator new applies to all of it. It
// counts operator new and ImGui's allocator, which is what the framework can reach. It is meant for
// Release builds, as debug standard library containers allocate a proxy of their own.
#include "graphics/presentation/system/components/button.h"
#include "graphics/presentation/system/components/card.h"
#include "graphics/presentation/system/components/list.h"
#include "graphics/presentation/system/components/modal.h"
#include "graphics/presentation/system/components/slider.h"
#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/components/toggle.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/fonts.h"
#include "graphics/presentation/system/imgui/imageStore.h"
#include "graphics/presentation/system/imgui/interaction.h"
#include "graphics/presentation/system/views/trophyToast.h"

#include "imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <span>
#include <string>
#include <vector>

namespace {

// Allocations made while counting is on. It is on only around the work being measured, so what the
// test itself allocates doesn't count.
bool g_counting = false;
uint64_t g_allocations = 0;

void *Allocate(std::size_t size) {
  if (g_counting) {
    ++g_allocations;
  }
  return std::malloc(size != 0 ? size : 1);
}

void *AllocateAligned(std::size_t size, std::align_val_t alignment) {
  if (g_counting) {
    ++g_allocations;
  }
  const auto align = static_cast<std::size_t>(alignment);
#ifdef _WIN32
  return _aligned_malloc(size != 0 ? size : 1, align);
#else
  void *memory = nullptr;
  return posix_memalign(&memory, align, size != 0 ? size : 1) == 0 ? memory : nullptr;
#endif
}

void FreeAligned(void *memory) {
#ifdef _WIN32
  _aligned_free(memory);
#else
  std::free(memory);
#endif
}

// ImGui allocates through functions of its own and never reaches operator new, so a frame could
// otherwise look clean by moving its allocations there.
void *ImGuiAllocate(std::size_t size, void *) { return Allocate(size); }
void ImGuiFree(void *memory, void *) { std::free(memory); }

void *AllocateOrAbort(std::size_t size) {
  void *memory = Allocate(size);
  if (memory == nullptr) {
    std::abort();
  }
  return memory;
}

void *AllocateAlignedOrAbort(std::size_t size, std::align_val_t alignment) {
  void *memory = AllocateAligned(size, alignment);
  if (memory == nullptr) {
    std::abort();
  }
  return memory;
}

} // namespace

// Every form is replaced, as a form left out would go uncounted without any sign of it.
void *operator new(std::size_t size) { return AllocateOrAbort(size); }
void *operator new[](std::size_t size) { return AllocateOrAbort(size); }
void *operator new(std::size_t size, const std::nothrow_t &) noexcept { return Allocate(size); }
void *operator new[](std::size_t size, const std::nothrow_t &) noexcept { return Allocate(size); }
void *operator new(std::size_t size, std::align_val_t alignment) { return AllocateAlignedOrAbort(size, alignment); }
void *operator new[](std::size_t size, std::align_val_t alignment) { return AllocateAlignedOrAbort(size, alignment); }
void *operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t &) noexcept { return AllocateAligned(size, alignment); }
void *operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t &) noexcept { return AllocateAligned(size, alignment); }
void operator delete(void *memory) noexcept { std::free(memory); }
void operator delete[](void *memory) noexcept { std::free(memory); }
void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void *memory, std::size_t) noexcept { std::free(memory); }
void operator delete(void *memory, const std::nothrow_t &) noexcept { std::free(memory); }
void operator delete[](void *memory, const std::nothrow_t &) noexcept { std::free(memory); }
void operator delete(void *memory, std::align_val_t) noexcept { FreeAligned(memory); }
void operator delete[](void *memory, std::align_val_t) noexcept { FreeAligned(memory); }
void operator delete(void *memory, std::size_t, std::align_val_t) noexcept { FreeAligned(memory); }
void operator delete[](void *memory, std::size_t, std::align_val_t) noexcept { FreeAligned(memory); }
void operator delete(void *memory, std::align_val_t, const std::nothrow_t &) noexcept { FreeAligned(memory); }
void operator delete[](void *memory, std::align_val_t, const std::nothrow_t &) noexcept { FreeAligned(memory); }

namespace {

using namespace Libs::Graphics::SystemUi;

constexpr const char *REGULAR_FONT = KYTY_TEST_FONT_REGULAR;
constexpr const char *BOLD_FONT = KYTY_TEST_FONT_BOLD;
constexpr int FRAMES_PER_SECOND = 60;
constexpr float TOAST_SECONDS = 6.0f;
constexpr int FRAMES_PER_LIFE = static_cast<int>((TOAST_SECONDS + 1.0f) * FRAMES_PER_SECOND);
constexpr int ICON_SIDE = 64;
constexpr int RGBA_CHANNELS = 4;
constexpr unsigned char ICON_GREY = 200;
constexpr int GOLD = 2;
constexpr size_t STORE_CAPACITY = 16;

// The most heap allocations a steady frame of the trophy toast may make.
constexpr uint64_t TOAST_FRAME_CEILING = 0;

// Stops the test run with `message` when `value` is false.
void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiAllocationTests: failed: %s\n", message);
    std::abort();
  }
}

// Runs `work` and returns how many allocations it made.
template <class Work> uint64_t AllocationsIn(Work &&work) {
  g_allocations = 0;
  g_counting = true;
  work();
  g_counting = false;
  return g_allocations;
}

// What a replaced operator is called with has to reach it, and the compiler is free to drop a new and
// delete pair it can see, so the pointers go through here to keep them.
void Keep(void *memory) {
  static void *volatile kept = nullptr;
  kept = memory;
}

// The counter sees every form of allocation the framework can reach, operator new and ImGui's, and it
// sees nothing while it is off. An over-long string and a vector stand for how the framework allocates.
void TestCounterSeesEveryForm() {
  Check(AllocationsIn([] {
          auto *memory = new int;
          Keep(memory);
          delete memory;
        }) == 1, "new is counted");
  Check(AllocationsIn([] {
          auto *memory = new int[4];
          Keep(memory);
          delete[] memory;
        }) == 1, "new[] is counted");
  Check(AllocationsIn([] {
          auto *memory = new (std::nothrow) int;
          Keep(memory);
          delete memory;
        }) == 1, "nothrow new is counted");
  Check(AllocationsIn([] {
          auto *memory = ::operator new(16, std::align_val_t{64});
          Keep(memory);
          ::operator delete(memory, std::align_val_t{64});
        }) == 1, "aligned new is counted");
  Check(AllocationsIn([] {
          std::string text(100, 'x');
          Keep(text.data());
        }) >= 1, "a long string is counted");
  Check(AllocationsIn([] {
          std::vector<int> numbers(8);
          Keep(numbers.data());
        }) == 1, "a vector is counted");
  Check(AllocationsIn([] {
          void *memory = ImGui::MemAlloc(16);
          Keep(memory);
          ImGui::MemFree(memory);
        }) == 1, "ImGui's allocator is counted");
  Check(AllocationsIn([] {}) == 0, "nothing is counted when nothing allocates");
  const uint64_t before = g_allocations;
  auto *uncounted = new int;
  Keep(uncounted);
  delete uncounted;
  Check(g_allocations == before, "or while counting is off");
}

// A `side` pixel square PNG, so the store decodes a real icon.
std::vector<std::byte> Png(int side) {
  std::vector<unsigned char> pixels(static_cast<size_t>(side * side * RGBA_CHANNELS), ICON_GREY);
  int length = 0;
  unsigned char *encoded = stbi_write_png_to_mem(pixels.data(), side * RGBA_CHANNELS, side, side, RGBA_CHANNELS, &length);
  Check(encoded != nullptr, "the icon encodes");
  std::vector<std::byte> bytes(reinterpret_cast<std::byte *>(encoded), reinterpret_cast<std::byte *>(encoded) + length);
  STBIW_FREE(encoded);
  return bytes;
}

// The moment `frame` frames after an arbitrary origin.
UiClock::time_point FrameTime(int frame) {
  return UiClock::time_point{} + std::chrono::duration_cast<UiClock::duration>(
                                     std::chrono::duration<double>(static_cast<double>(frame) / FRAMES_PER_SECOND));
}

// What the frames of a toast's life did.
struct ToastLife {
  // The most allocations in one frame that drew, leaving out the frame it was shown in, which adds it
  // to the store and is not steady.
  uint64_t most = 0;
  int drawn_frames = 0;
};

// What stays the same from toast to toast, as it does in the overlay, whose clock keeps going.
struct ToastScene {
  const UiContext &context;
  const UiFonts &fonts;
  ImageStore &images;
  InteractionStore &store;
  UiFrame &frame;
  TrophyToastView &view;
  int clock = 0;
};

// Plays one toast's whole life, entrance, shine, hold and exit, as the overlay does.
ToastLife PlayToast(ToastScene &scene) {
  const auto &[context, fonts, images, store, frame, view, clock] = scene;
  const Rect area = context.scale.SafeArea();
  const int start = clock;
  view.Show({"Master Alchemist", GOLD, Png(ICON_SIDE)}, "Trophy earned!", DEFAULT_THEME, {TOAST_SECONDS}, FrameTime(start));
  ToastLife life;
  for (int at = start; at < start + FRAMES_PER_LIFE && !view.Idle(); at++) {
    scene.clock = at + 1;
    ImGui::NewFrame();
    ImDrawList &list = *ImGui::GetForegroundDrawList();
    const uint64_t made = AllocationsIn([&] {
      frame.Begin(DEFAULT_THEME, context.scale, fonts, store, FrameTime(at));
      (void)view.Draw(frame, list, area);
      store.Collect();
    });
    const bool drew = list.VtxBuffer.Size > 0;
    ImGui::EndFrame();
    // The frame that retires the toast frees its icon, which is not a steady frame either.
    if (at > start && drew && !view.Idle()) {
      Check(images.LiveCount() == 1, "the toast has its icon, as a toast without one would allocate less");
      life.most = std::max(life.most, made);
      ++life.drawn_frames;
    }
  }
  return life;
}

// Once a first toast has grown the framework's storage, every frame of the next one stays within its
// budget, at every point in its life. It has to really draw, or a frame that draws nothing would pass
// with no allocations at all.
void TestToastFrames() {
  auto &io = ImGui::GetIO();
  io.DeltaTime = 1.0f / FRAMES_PER_SECOND;
  io.DisplaySize = {1920, 1080};
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
  io.IniFilename = nullptr;
  const UiFonts fonts = LoadUiFonts(*io.Fonts, REGULAR_FONT, BOLD_FONT);
  Check(fonts.regular != nullptr && fonts.bold != nullptr, "the fonts load, as ImGui's default would measure differently");
  const UiContext context{DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, 1920, 1080), fonts};
  {
    ImageStore images(STORE_CAPACITY);
    InteractionStore store;
    UiFrame frame;
    TrophyToastView view(images);
    ToastScene scene{context, fonts, images, store, frame, view};
    (void)PlayToast(scene);
    const ToastLife steady = PlayToast(scene);
    std::printf("trophy toast: at most %llu allocations in a steady frame, over %d frames, ceiling %llu\n",
                static_cast<unsigned long long>(steady.most), steady.drawn_frames, static_cast<unsigned long long>(TOAST_FRAME_CEILING));
    constexpr int MOST_OF_A_LIFE = 300;
    Check(steady.drawn_frames >= MOST_OF_A_LIFE, "the toast draws through its life");
    Check(steady.most <= TOAST_FRAME_CEILING, "a steady frame of the toast stays within its budget");
  }
}

// Long enough that each is on the heap wherever a standard library keeps short strings inline.
constexpr const char *LONG_ROW = "A row with more text than any small string holds";
constexpr const char *LONG_DIALOG_ID = "a dialog named with more than any small string holds";

// How a component's scene went once it had settled.
struct SceneResult {
  uint64_t most = 0;
  int drawing_frames = 0;
};

// Draws `draw(frame, list)` for a while in a window of its own, to let storage grow, and then counts
// what steady frames allocate. `draw` must report whether it drew through `drew`.
template <class Draw> SceneResult RunScene(const UiContext &context, const UiFonts &fonts, InteractionStore &store, UiFrame &frame, Draw &&draw) {
  constexpr int SETTLING_FRAMES = 40;
  constexpr int MEASURED_FRAMES = 40;
  SceneResult result;
  for (int at = 0; at < SETTLING_FRAMES + MEASURED_FRAMES; at++) {
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("scene", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground | ImGuiWindowFlags_NoSavedSettings);
    bool drew = false;
    const uint64_t made = AllocationsIn([&] {
      frame.Begin(DEFAULT_THEME, context.scale, fonts, store, FrameTime(at));
      drew = draw(frame, *ImGui::GetWindowDrawList());
      store.Collect();
    });
    ImGui::End();
    ImGui::EndFrame();
    if (at >= SETTLING_FRAMES && drew) {
      result.most = std::max(result.most, made);
      ++result.drawing_frames;
    }
  }
  return result;
}

// Measures, arranges and renders `root` in the top left of a list, and says whether it drew.
bool Show(UiFrame &frame, Element root, ImDrawList &list) {
  const float unlimited = std::numeric_limits<float>::infinity();
  const Vec2 size = Measure(frame, root, {unlimited, unlimited});
  if (!Arrange(frame, root, {16, 16, size.x, size.y})) {
    return false;
  }
  const int before = list.VtxBuffer.Size;
  return !Render(frame, root, list).error.has_value() && list.VtxBuffer.Size > before;
}

// A list with long rows and a heading, some rows with two lines, and more rows than fit, so some are
// out of view.
bool DrawListScene(UiFrame &frame, ImDrawList &list) {
  std::array<Element, 13> rows;
  rows[0] = ListHeading(frame, "heading", {.text = "A heading longer than any small string holds"});
  static constexpr std::array<std::string_view, 12> KEYS{"r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "r11"};
  for (size_t index = 0; index < KEYS.size(); index++) {
    rows[index + 1] = ListRow(frame, KEYS[index], {LONG_ROW, index % 2 == 0 ? std::string_view{} : std::string_view{LONG_ROW}, index == 3});
  }
  return Show(frame, List(frame, "list", rows, {.width = Dp(400.0f), .height = Dp(300.0f)}), list);
}

// A slider with a long label and a reserve, in room to show it.
bool DrawSliderScene(UiFrame &frame, ImDrawList &list, float &value) {
  SliderProps props;
  props.value_label = "a value too long for a small string";
  props.label_reserve = "the widest value too long for a small string";
  return Show(frame, Slider(frame, "slider", &value, props, {.width = Dp(900.0f)}).root, list);
}

// Two tabs and a button in a row.
bool DrawTabsScene(UiFrame &frame, ImDrawList &list, int &selected) {
  static constexpr std::array<std::string_view, 2> LABELS{"A tab label too long for a small string", "Another"};
  const TabBarElement bar = TabBar(frame, "tabs", LABELS, &selected);
  const float unlimited = std::numeric_limits<float>::infinity();
  const Vec2 size = Measure(frame, bar.element, {unlimited, unlimited});
  if (!Arrange(frame, bar.element, {16, 16, size.x, size.y})) {
    return false;
  }
  const int before = list.VtxBuffer.Size;
  return !Render(frame, bar.element, list).error.has_value() && list.VtxBuffer.Size > before;
}

// An open dialog with a long id, a body that wraps, and a button.
bool DrawModalScene(UiFrame &frame) {
  const std::array<Element, 1> buttons{Button(frame, "ok", {.label = "A button label too long for a small string"})};
  const DialogParts parts = DialogContent(frame, "dialog", {"A title too long for a small string", "A body long enough to need wrapping across the width of the dialog's card, and more than one line of it."}, buttons);
  const Rect card = FitDialogCard(frame, parts);
  ModalScope modal(frame, LONG_DIALOG_ID, card, true);
  if (!modal.Visible() || !modal.Place(frame, parts)) {
    return false;
  }
  const int before = modal.List().VtxBuffer.Size;
  return !Render(frame, parts.content, modal.List()).error.has_value() && modal.List().VtxBuffer.Size > before;
}

// A button with focus, which draws its ring, and the state behind it, each frame. Focus is the usual
// state of something on a menu, and it counts only while the control really has it.
bool DrawFocusedScene(UiFrame &frame, ImDrawList &list) {
  const Element button = Button(frame, "focused", {.label = "A button label too long for a small string"});
  (void)AssignIds(frame, button, CurrentIdSeed());
  FocusControl(IdOf(frame, button));
  if (!Show(frame, button, list)) {
    return false;
  }
  const auto input = InputOf(frame, button);
  return input.has_value() && input->interaction.state.focused && input->interaction.visuals.focus > 0.0f;
}

// Every kind of component holds to the budget of no allocations once it has settled. Each scene has to
// really draw, as one that drew nothing would pass, and uses text long enough to be on the heap.
void TestComponentFrames() {
  // TestToastFrames has set up the display and the rest of the context's input.
  auto &io = ImGui::GetIO();
  const UiFonts fonts = LoadUiFonts(*io.Fonts, REGULAR_FONT, BOLD_FONT);
  Check(fonts.regular != nullptr && fonts.bold != nullptr, "the fonts load");
  const UiContext context{DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, 1920, 1080), fonts};
  InteractionStore store;
  UiFrame frame;
  float value = 0.5f;
  bool on = true;
  bool off = false;
  int selected = 0;
  const auto check = [&](const char *name, auto &&draw) {
    const SceneResult result = RunScene(context, fonts, store, frame, draw);
    std::printf("%s: at most %llu allocations in a steady frame, over %d frames\n", name, static_cast<unsigned long long>(result.most), result.drawing_frames);
    constexpr int ALL_MEASURED = 40;
    if (result.drawing_frames != ALL_MEASURED || result.most != 0) {
      std::fprintf(stderr, "SystemUiAllocationTests: failed: %s drew %d frames with at most %llu allocations\n", name, result.drawing_frames, static_cast<unsigned long long>(result.most));
      std::abort();
    }
  };
  check("button", [](UiFrame &f, ImDrawList &list) { return Show(f, Button(f, "button", {.label = "A button label too long for a small string"}), list); });
  check("toggle", [&](UiFrame &f, ImDrawList &list) { return Show(f, Toggle(f, "toggle", &on), list); });
  check("toggle off, with its outline", [&](UiFrame &f, ImDrawList &list) { return Show(f, Toggle(f, "toggle off", &off), list); });
  check("slider", [&](UiFrame &f, ImDrawList &list) { return DrawSliderScene(f, list, value); });
  check("tabs", [&](UiFrame &f, ImDrawList &list) { return DrawTabsScene(f, list, selected); });
  check("card and progress", [](UiFrame &f, ImDrawList &list) {
    const std::array<Element, 1> inside{ProgressBar(f, {.value = 0.5f, .fill = Rgb(0x2391FF)}, {.width = Dp(200.0f)})};
    return Show(f, Card(f, {}, {}, inside), list);
  });
  check("list", DrawListScene);
  check("focused button", DrawFocusedScene);
  // Last, as nothing closes its popup, which would sit in front of whatever came after.
  check("modal", [](UiFrame &f, ImDrawList &) { return DrawModalScene(f); });
}

} // namespace

int main() {
  // Before the context, which allocates with whatever is set when it is made.
  ImGui::SetAllocatorFunctions(ImGuiAllocate, ImGuiFree);
  ImGui::CreateContext();
  TestCounterSeesEveryForm();
  TestToastFrames();
  TestComponentFrames();
  ImGui::DestroyContext();
  std::printf("System UI allocation tests passed\n");
  return 0;
}
