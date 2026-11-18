#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_EFFECTSPAGE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_EFFECTSPAGE_H_

#include "graphics/presentation/system/animation/timeline.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/element/frame.h"

#include <array>
#include <string_view>

namespace Libs::Graphics::SystemUi::Gallery {

struct GalleryState;
struct PageEntry;

// The effects page's views, in tab order.
inline constexpr std::array<std::string_view, 3> EFFECTS_VIEWS {"Depth", "Shine", "Motion"};

/// How far through a loop of `period_ms` the clock `now` is, in milliseconds. The clock counts from
/// boot, too many milliseconds for a float to hold finely after a day or two, so this works in
/// double and only its small result is a float. The page's samples run from the frame's clock,
/// which a screenshot run fixes, so a capture always shows the same moment.
[[nodiscard]] float LoopMs(UiClock::time_point now, float period_ms);

/// Draws the effects page into `pane`. Depth shows the elevation levels and the glows. Shine sweeps
/// a card over and over, above frames held at points along the sweep, so a still image says as
/// much. Motion plots each easing curve and shows each duration, with a marker running along each.
/// A toggle on those two applies the reduced-motion rule, which on Shine means no shine at all.
void DrawEffectsPage(UiFrame& frame, GalleryState& state, const PageEntry& page, const Rect& pane);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_EFFECTSPAGE_H_
