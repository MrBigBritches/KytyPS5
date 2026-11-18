#ifndef EMULATOR_TESTS_SYSTEMUIELEMENTTESTKIT_H_
#define EMULATOR_TESTS_SYSTEMUIELEMENTTESTKIT_H_

// The small pieces the element tests share, so a second test target does not copy them. The older
// SystemUiElementTests.cpp keeps its own.
#include "SystemUiNavHarness.h"
#include "graphics/presentation/system/core/scale.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/imgui/fonts.h"

#include <cstdio>
#include <cstdlib>

namespace SystemUiTest {

using namespace Libs::Graphics::SystemUi;

// Stops the test run with `message` when `value` is false.
inline void Check(bool value, const char *message) {
  if (!value) {
    std::fprintf(stderr, "SystemUiElementInteractionTests: failed: %s\n", message);
    std::abort();
  }
}

// A frame with the objects it borrows, at 720p unless told otherwise.
struct FrameRig {
  Theme theme = DEFAULT_THEME;
  ScaleContext scale;
  UiFonts fonts;
  InteractionStore store;
  UiFrame frame;

  explicit FrameRig(float width = 1280, float height = 720) : scale(DEFAULT_THEME.scale, width, height) {}

  void Begin() { frame.Begin(theme, scale, fonts, store, UiClock::now()); }
};

} // namespace SystemUiTest

#endif // EMULATOR_TESTS_SYSTEMUIELEMENTTESTKIT_H_
