#ifndef EMULATOR_TESTS_SYSTEMUINAVHARNESS_H_
#define EMULATOR_TESTS_SYSTEMUINAVHARNESS_H_

// A headless ImGui context for testing controls with gamepad navigation and the mouse. Each frame
// runs the test's drawing inside one full-screen window, on a clock that advances 60 frames a
// second, and collects the interaction store afterwards, as a view would.
#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/imgui/fonts.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/imgui/interactContext.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include "imgui.h"
#include "imgui_internal.h"

#include <chrono>
#include <functional>
#include <type_traits>

namespace SystemUiTest {

using namespace Libs::Graphics::SystemUi;

// What a harness changes from its defaults, which match a 1080p output with ImGui's default font
// and ImGui's own navigation settings.
struct HarnessOptions {
  float width = 1920;
  float height = 1080;
  // Loads the fonts the UI draws with into the harness's atlas. Without it the UI has no fonts and
  // text uses ImGui's default.
  std::function<UiFonts(ImFontAtlas &)> load_fonts;
  // Applies the system UI's navigation settings, as --system-ui does.
  bool configure_navigation = false;
};

#ifdef KYTY_TEST_FONT_REGULAR
// Loads the real fonts into `atlas`, then bakes every glyph the system UI uses at every size it
// uses, before any frame runs. Glyphs are otherwise baked in the order they are first measured, and
// that order decides the atlas packing and so the pixels at each glyph's edges. Two renderings that
// ask for text in a different order, such as a layout that measures off-view rows, then differ in
// UVs without drawing anything differently. Pre-baking makes the comparison independent of order.
inline UiFonts LoadPrebakedFonts(ImFontAtlas &atlas, float width, float height) {
  const UiFonts fonts = LoadUiFonts(atlas, KYTY_TEST_FONT_REGULAR, KYTY_TEST_FONT_BOLD);
  PrebakeGlyphs(fonts, DEFAULT_THEME.type, ScaleContext(DEFAULT_THEME.scale, width, height));
  return fonts;
}
#endif

class NavHarness {
public:
  static constexpr double FRAME_MS = 1000.0 / 60.0;

  explicit NavHarness(const HarnessOptions &options = {}) : m_width(options.width), m_height(options.height) {
    // CreateContext only makes the new context current when none is, so a second harness alive
    // at once would otherwise drive the first one's context.
    m_context = ImGui::CreateContext();
    ImGui::SetCurrentContext(m_context);
    auto &io = ImGui::GetIO();
    io.DisplaySize = {options.width, options.height};
    io.DeltaTime = static_cast<float>(FRAME_MS / 1000.0);
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures | ImGuiBackendFlags_HasGamepad;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
    // As the overlay sets it. ImGui only activates from the gamepad while its nav cursor shows.
    io.ConfigNavCursorVisibleAlways = true;
    io.IniFilename = nullptr;
    io.Fonts->AddFontDefault();
    if (options.load_fonts) {
      m_fonts = options.load_fonts(*io.Fonts);
    }
    if (options.configure_navigation) {
      ConfigureNavigation(DEFAULT_THEME);
    }
    // Without anti-aliasing, shapes keep exact edges, so tests can measure them.
    ImGui::GetStyle().AntiAliasedFill = false;
    ImGui::GetStyle().AntiAliasedLines = false;
  }
  ~NavHarness() { ImGui::DestroyContext(m_context); }
  NavHarness(const NavHarness &) = delete;
  NavHarness &operator=(const NavHarness &) = delete;

  // The current frame's time.
  [[nodiscard]] UiClock::time_point Now() const {
    return UiClock::time_point{} + std::chrono::duration_cast<UiClock::duration>(std::chrono::duration<double, std::milli>(m_ms));
  }

  // Runs one frame inside a full-screen window, calling `draw` with a UiFrame that has been begun for
  // the frame, as a caller of the system UI has. A `draw` that takes an InteractContext is given the
  // internal type instead, for the tests of the element layer that register hand-made controls. A
  // generic lambda, with an `auto &` parameter, is invocable with a UiFrame, so it gets the frame.
  template <class Draw> void Frame(Draw &&draw) {
    ImGui::SetCurrentContext(m_context);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);
    ImGui::Begin("view", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoBackground |
                                      ImGuiWindowFlags_NoSavedSettings);
    const UiContext ui{DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, m_width, m_height), m_fonts};
    if constexpr (std::is_invocable_v<Draw &, UiFrame &>) {
      m_frame.Begin(DEFAULT_THEME, ScaleContext(DEFAULT_THEME.scale, m_width, m_height), m_fonts, store, Now());
      draw(m_frame);
    } else {
      InteractContext context{ui, store, *ImGui::GetWindowDrawList(), Now()};
      draw(context);
    }
    ImGui::End();
    ImGui::EndFrame();
    store.Collect();
    m_ms += FRAME_MS;
  }

  // Runs `count` frames.
  template <class Draw> void Frames(int count, Draw &&draw) {
    for (int frame = 0; frame < count; frame++) {
      Frame(draw);
    }
  }

  // Presses and releases `key` over two frames, then runs one more so a navigation move, which
  // ImGui resolves on the frame after the press, has landed.
  template <class Draw> void Press(ImGuiKey key, Draw &&draw) {
    ImGui::SetCurrentContext(m_context);
    ImGui::GetIO().AddKeyEvent(key, true);
    Frame(draw);
    ImGui::GetIO().AddKeyEvent(key, false);
    Frame(draw);
    Frame(draw);
  }

  // Presses or releases `key`, taking effect on the next frame.
  void Key(ImGuiKey key, bool down) {
    ImGui::SetCurrentContext(m_context);
    ImGui::GetIO().AddKeyEvent(key, down);
  }

  // Presses or releases the left mouse button, taking effect on the next frame.
  void MouseButton(bool down) {
    ImGui::SetCurrentContext(m_context);
    ImGui::GetIO().AddMouseButtonEvent(0, down);
  }

  // Moves the mouse to `x`, `y` and runs a frame.
  template <class Draw> void MoveMouse(float x, float y, Draw &&draw) {
    ImGui::SetCurrentContext(m_context);
    ImGui::GetIO().AddMousePosEvent(x, y);
    Frame(draw);
  }

  InteractionStore store;

private:
  float m_width;
  float m_height;
  UiFonts m_fonts;
  UiFrame m_frame;
  ImGuiContext *m_context = nullptr;
  double m_ms = 0.0;
};

} // namespace SystemUiTest

#endif // EMULATOR_TESTS_SYSTEMUINAVHARNESS_H_
