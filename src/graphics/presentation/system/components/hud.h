#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_HUD_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_HUD_H_

#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"

namespace Libs::Graphics::SystemUi {

// The padding and corners of a heads-up panel, such as the hint bar or the performance counters,
// in reference pixels. No component draws one yet, only the gallery reads these. They
// are here, with a header of their own, until the HUD component is built, which will add its element
// function to this file.
struct HudTokens {
	float padding_x = 12.0f;
	float padding_y = 8.0f;
	float radius    = RadiusScale {}.sm;

	/// True when no length is negative.
	[[nodiscard]] constexpr bool IsValid() const {
		return padding_x >= 0.0f && padding_y >= 0.0f && radius >= 0.0f;
	}
};

/// The HUD's tokens for `theme`: its corners come from the radius scale.
[[nodiscard]] constexpr HudTokens HudDefaults(const Theme& theme) {
	HudTokens tokens;
	tokens.radius = theme.radius.sm;
	return tokens;
}

static_assert(HudDefaults(DEFAULT_THEME).IsValid());

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_HUD_H_
