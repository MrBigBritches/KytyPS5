#ifndef EMULATOR_TESTS_SYSTEMUITABSKIT_H_
#define EMULATOR_TESTS_SYSTEMUITABSKIT_H_

// How the tests draw a tab bar element: DrawElementBar builds one, lays it out, renders it, and
// reports what it found.
#include "SystemUiElementTestHooks.h"
#include "SystemUiNavHarness.h"
#include "graphics/presentation/system/components/tabs.h"
#include "graphics/presentation/system/components/text.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"

#include "imgui.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <string_view>

namespace SystemUiTest {

// How many texts the frame laid out again at a new width since its last Begin.
inline unsigned TextRelayoutsOf(const Libs::Graphics::SystemUi::UiFrame &frame) { return Libs::Graphics::SystemUi::FrameTestHooks::TextRelayouts(frame); }

// What drawing one bar element found.
struct BarOutcome {
  // The bar's changed flag, set by Render.
  bool changed = false;
  // The tab that has focus, or -1.
  int focusedTab = -1;
};

// Builds a bar element in `frame`, which the caller has begun, lays it out with its left edge and
// top at `rect`'s and its width asked for as `rect.width`, and renders it, faded by `opacity`.
inline BarOutcome DrawElementBar(Libs::Graphics::SystemUi::UiFrame &frame, std::string_view key, const Rect &rect,
                                 std::span<const std::string_view> labels, int &selected, bool shoulders = false, float opacity = 1.0f) {
  using namespace Libs::Graphics::SystemUi;
  BarOutcome outcome;
  TabBarProps props;
  props.shoulders = shoulders;
  props.changed = &outcome.changed;
  const ItemLayout item{.width = FromPixels(frame.Scale(), rect.width)};
  const auto bar = TabBar(frame, key, labels, &selected, props, item);
  const float unlimited = std::numeric_limits<float>::infinity();
  const auto size = Measure(frame, bar.element, {unlimited, unlimited});
  (void)Arrange(frame, bar.element, {rect.x, rect.y, size.x, size.y});
  (void)Render(frame, bar.element, CurrentDrawList(), PaintState{.opacity = opacity});
  for (int index = 0; index < static_cast<int>(labels.size()); ++index) {
    const auto input = InputOf(frame, TabOf(bar, index));
    if (input && input->interaction.state.focused) {
      outcome.focusedTab = index;
    }
  }
  return outcome;
}

} // namespace SystemUiTest

#endif // EMULATOR_TESTS_SYSTEMUITABSKIT_H_
