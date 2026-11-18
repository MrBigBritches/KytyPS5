// A headless benchmark of the system UI's trophy toast. For each output size it plays whole toast
// lives at 60 frames a second and reports how long TrophyToastView::Draw takes and how much it
// draws. It isn't a test and isn't run by ctest, and its timings are for reading, as they depend on
// the machine. It exits with an error when the toast draws nothing during its hold or still draws
// after leaving, so its numbers always measure real drawing, and it never fails on time.
// SystemUiAllocationTests is what holds the toast to a budget.
#include "SystemUiElementTestHooks.h"
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/core/frameStats.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/imgui/fonts.h"
#include "graphics/presentation/system/imgui/imageStore.h"
#include "graphics/presentation/system/views/trophyToast.h"

#include "imgui.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <vector>

namespace {

using namespace Libs::Graphics::SystemUi;
using BenchClock = std::chrono::steady_clock;
using Micros = std::chrono::duration<double, std::micro>;

constexpr const char *REGULAR_FONT = KYTY_TEST_FONT_REGULAR;
constexpr const char *BOLD_FONT = KYTY_TEST_FONT_BOLD;
constexpr int FRAMES_PER_SECOND = 60;
constexpr float TOAST_SECONDS = 6.0f;
// One more second than the toast, so the frames after it has left are measured too.
constexpr int FRAMES_PER_RUN = static_cast<int>((TOAST_SECONDS + 1.0f) * FRAMES_PER_SECOND);
constexpr int WARM_UP_RUNS = 2;
constexpr int RUNS = 20;
constexpr int ICON_SIDE = 256;
constexpr int RGBA_CHANNELS = 4;
constexpr unsigned char ICON_GREY = 200;
constexpr int GOLD = 2;
// A frame in the middle of the hold, which must draw, and one after the toast, which must not.
constexpr int HOLD_FRAME = FRAMES_PER_SECOND * 3;
constexpr int GONE_FRAME = FRAMES_PER_RUN - 1;

// An output size to measure at.
struct Output {
  const char *name;
  float width;
  float height;
};

constexpr Output OUTPUTS[] = {{"1080p", 1920, 1080}, {"4K", 3840, 2160}};
// Every run loads an icon that is never destroyed without a renderer, so the store needs a slot
// for each. A full store would silently measure toasts without an icon.
constexpr size_t STORE_CAPACITY = (WARM_UP_RUNS + RUNS) * std::size(OUTPUTS);

// What one output's runs measured.
struct Results {
  FrameStats first_us{RUNS};
  FrameStats steady_us{static_cast<size_t>(RUNS * FRAMES_PER_RUN)};
  FrameStats vertices{static_cast<size_t>(RUNS * FRAMES_PER_RUN)};
  FrameStats indices{static_cast<size_t>(RUNS * FRAMES_PER_RUN)};
  FrameStats commands{static_cast<size_t>(RUNS * FRAMES_PER_RUN)};
  bool hold_drew = true;
  bool gone_was_empty = true;
};

// A `side` pixel square PNG, so the store decodes a real icon.
std::vector<std::byte> Png(int side) {
  std::vector<unsigned char> pixels(static_cast<size_t>(side * side * RGBA_CHANNELS), ICON_GREY);
  int length = 0;
  unsigned char *encoded =
      stbi_write_png_to_mem(pixels.data(), side * RGBA_CHANNELS, side, side, RGBA_CHANNELS, &length);
  if (encoded == nullptr) {
    std::fprintf(stderr, "SystemUiBench: could not encode the icon\n");
    std::exit(1);
  }
  std::vector<std::byte> bytes(reinterpret_cast<std::byte *>(encoded), reinterpret_cast<std::byte *>(encoded) + length);
  STBIW_FREE(encoded);
  return bytes;
}

// The moment `frame` frames after an arbitrary origin.
UiClock::time_point FrameTime(int frame) {
  return UiClock::time_point{} + std::chrono::duration_cast<UiClock::duration>(
                                     std::chrono::duration<double>(static_cast<double>(frame) / FRAMES_PER_SECOND));
}

// The draw commands in `list` that draw something, which are the ones ImGui::Render keeps. A new
// frame starts the list with one empty command, and a clip change can leave another at the end.
int DrawingCommands(const ImDrawList &list) {
  int count = 0;
  for (const auto &command : list.CmdBuffer) {
    count += command.ElemCount > 0 || command.UserCallback != nullptr ? 1 : 0;
  }
  return count;
}

// Plays one toast life, adding each frame to `results` when `record` is true. `draw(view, list,
// frame)` draws one frame, and is what is timed. Only frames that drew count, since the toast leaves
// before the run ends, so the times match the in-game statistics, which only count frames with a
// toast on screen.
template <class Draw>
void RunToast(ImageStore &store, Results &results, bool record, Draw &&draw) {
  TrophyToastView view(store);
  view.Show({"Master Alchemist", GOLD, Png(ICON_SIDE)}, "Trophy earned!", DEFAULT_THEME, TrophyToastOptions{TOAST_SECONDS}, FrameTime(0));
  for (int frame = 0; frame < FRAMES_PER_RUN; frame++) {
    ImGui::NewFrame();
    ImDrawList &list = *ImGui::GetForegroundDrawList();
    // The list is empty after a new frame, so everything in it is the toast's.
    const auto start = BenchClock::now();
    (void)draw(view, list, frame);
    const double drawn_us = Micros(BenchClock::now() - start).count();
    const int vertices = list.VtxBuffer.Size;
    const int indices = list.IdxBuffer.Size;
    const int commands = DrawingCommands(list);
    ImGui::EndFrame();
    if (!record) {
      continue;
    }
    if (frame == 0) {
      results.first_us.Add(drawn_us);
    } else if (vertices > 0) {
      results.steady_us.Add(drawn_us);
      results.vertices.Add(vertices);
      results.indices.Add(indices);
      results.commands.Add(commands);
    }
    results.hold_drew = results.hold_drew && (frame != HOLD_FRAME || vertices > 0);
    results.gone_was_empty = results.gone_was_empty && (frame != GONE_FRAME || vertices == 0);
  }
}

// Prints one line of results at `output`.
void Print(const Output &output, const Results &results) {
  std::printf("%-6s first frame %7.1f us mean %7.1f max | later frames that drew %5.2f us mean %5.2f p95 %7.1f max | "
              "vertices %6.0f mean %6.0f max | indices %6.0f mean %6.0f max | commands %4.1f mean %3.0f max\n",
              output.name, results.first_us.Mean(), results.first_us.Max(), results.steady_us.Mean(),
              results.steady_us.P95(), results.steady_us.Max(), results.vertices.Mean(), results.vertices.Max(),
              results.indices.Mean(), results.indices.Max(), results.commands.Mean(), results.commands.Max());
}

// True when the toast in `results` drew during its hold and nothing after it left.
bool Drew(const Output &output, const Results &results) {
  if (!results.hold_drew || !results.gone_was_empty) {
    std::fprintf(stderr, "SystemUiBench: %s: the toast %s\n", output.name,
                 results.hold_drew ? "still drew after it had left" : "drew nothing during its hold");
    return false;
  }
  return true;
}

// Measures the view at `output` after warming up, prints its results, and fails when it draws
// wrongly.
bool Measure(const Output &output, const UiFonts &fonts, ImageStore &store) {
  ImGui::GetIO().DisplaySize = {output.width, output.height};
  const UiContext context{DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, output.width, output.height), fonts};
  const Rect area = context.scale.SafeArea();
  InteractionStore interactions;
  UiFrame ui;
  Results results;
  // How many text nodes were laid out a second time while arranging, over the frames that drew.
  uint64_t relayouts = 0;
  uint64_t drawn_frames = 0;
  for (int run = 0; run < WARM_UP_RUNS + RUNS; run++) {
    RunToast(store, results, run >= WARM_UP_RUNS, [&](auto &view, ImDrawList &list, int frame) {
      ui.Begin(DEFAULT_THEME, context.scale, fonts, interactions, FrameTime(frame));
      const bool more = view.Draw(ui, list, area);
      // Collected each frame as the overlay does, so the toast's animation, kept in the store under
      // its key, is forgotten once it has left and the next run starts its own.
      interactions.Collect();
      if (list.VtxBuffer.Size > 0) {
        relayouts += FrameTestHooks::TextRelayouts(ui);
        ++drawn_frames;
      }
      return more;
    });
  }
  Print(output, results);
  std::printf("%-6s text relayouts per drawn frame %.2f\n", output.name,
              static_cast<double>(relayouts) / static_cast<double>(drawn_frames));
  return Drew(output, results);
}

} // namespace

int main() {
  ImGui::CreateContext();
  auto &io = ImGui::GetIO();
  io.DeltaTime = 1.0f / FRAMES_PER_SECOND;
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
  io.IniFilename = nullptr;
  const UiFonts fonts = LoadUiFonts(*io.Fonts, REGULAR_FONT, BOLD_FONT);
  bool passed = true;
  {
    ImageStore store(STORE_CAPACITY);
    for (const auto &output : OUTPUTS) {
      passed = Measure(output, fonts, store) && passed;
    }
    // The store must go before the context it registered its images with.
  }
  ImGui::DestroyContext();
  return passed ? 0 : 1;
}
