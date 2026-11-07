#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_BUTTON_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_BUTTON_H_

#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

#include <cstdint>
#include <string_view>

namespace Libs::Graphics::SystemUi {

// What a button is for. Primary is the one main action of a screen, Secondary an ordinary one,
// Ghost a quiet one such as Cancel next to a primary, and Danger an action that destroys data.
enum class ButtonVariant : uint8_t { Primary, Secondary, Ghost, Danger };

enum class ButtonSize : uint8_t { Small, Medium, Large };

// A button's sizes, in reference pixels.
struct ButtonTokens {
	float height_small    = 48.0f;
	float height_medium   = 56.0f;
	float height_large    = 64.0f;
	float padding_x       = SpaceScale {}.s5;
	float padding_x_small = SpaceScale {}.s4;
	// Short labels such as "OK" still get a comfortable target.
	float min_width = 120.0f;
	float radius    = RadiusScale {}.md;

	/// True when the heights rise from small to large and every length is usable.
	[[nodiscard]] constexpr bool IsValid() const {
		return height_small > 0.0f && height_small <= height_medium &&
		       height_medium <= height_large && padding_x >= 0.0f && padding_x_small >= 0.0f &&
		       min_width >= 0.0f && radius >= 0.0f;
	}
};

/// The button's tokens for `theme`: its padding and radius come from the space and radius scales.
[[nodiscard]] constexpr ButtonTokens ButtonDefaults(const Theme& theme) {
	ButtonTokens tokens;
	tokens.padding_x       = theme.space.s5;
	tokens.padding_x_small = theme.space.s4;
	tokens.radius          = theme.radius.md;
	return tokens;
}

static_assert(ButtonDefaults(DEFAULT_THEME).IsValid());

// A button's content and look.
struct ButtonProps {
	std::string_view label;
	ButtonVariant    variant  = ButtonVariant::Secondary;
	ButtonSize       size     = ButtonSize::Medium;
	bool             disabled = false;
	// Focused first when its window appears, such as Cancel in a dialog that deletes something.
	bool default_focus = false;
	// Fields of ButtonDefaults to change for this button, such as its radius. Every other token
	// stays what the theme gives.
	TokenPatch<ButtonTokens> tokens;
	// A Ghost button has no surface of its own, so it needs telling when what's behind it is light.
	// Its label is then dark and its overlays shade. Other variants bring their own surface.
	bool on_light = false;
};

/// A button as an element: a box in the variant's colours holding its label, made interactive. The
/// label is centred and copied into the frame, so it need only last until this call returns.
/// Measured with no width, it is the label plus the size's padding on each side, never narrower
/// than the theme's minimum, at the size's height. `item` replaces either length and adds the
/// button's share of its container. `key` names it among its siblings, so give each button its own,
/// as its focus and highlights are kept under it.
///
/// Render takes its input. It reports an activation through RenderResult::activated and InputOf on
/// the frame the gamepad's confirm button is pressed or a mouse click over it is released, and runs
/// the callback given to OnActivate, if any, on that frame. A disabled button is skipped by
/// navigation and never activates, and it fades with the theme's disabled opacity and keeps any
/// focus ring it had, faded with it. A press shrinks the button about its centre and the label
/// with it, but the label keeps its size, so a press never cuts it short. Fade or move the whole
/// button with WithPose or the opacity of Paint. Returns a none handle when a part is refused, with
/// `reason` set if given.
[[nodiscard]] Element Button(UiFrame& frame, std::string_view key, const ButtonProps& props,
                             const ItemLayout& item = {}, ElementError* reason = nullptr);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_BUTTON_H_
