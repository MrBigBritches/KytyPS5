#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_MODAL_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_MODAL_H_

#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/core/geometry.h"
#include "graphics/presentation/system/core/theme.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/core/units.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>

namespace Libs::Graphics::SystemUi {

// A dialog card's widths and corners, in reference pixels, and its shadow.
struct DialogTokens {
	float     width_small  = 560.0f;
	float     width_medium = 720.0f;
	float     width_large  = 960.0f;
	float     radius       = RadiusScale {}.xl;
	Elevation elevation    = ElevationScale {}.level4;

	/// True when the widths rise from small to large and the radius is not negative.
	[[nodiscard]] constexpr bool IsValid() const {
		return width_small > 0.0f && width_small <= width_medium && width_medium <= width_large &&
		       radius >= 0.0f;
	}
};

/// The dialog's tokens for `theme`: its corners and shadow come from the radius and elevation
/// scales.
[[nodiscard]] constexpr DialogTokens DialogDefaults(const Theme& theme) {
	DialogTokens tokens;
	tokens.radius    = theme.radius.xl;
	tokens.elevation = theme.elevation.level4;
	return tokens;
}

static_assert(DialogDefaults(DEFAULT_THEME).IsValid());

/// The dialog card's rectangle, in output pixels, for a dialog `height` tall, a length in design
/// units like every other size, so a fixed height grows with the output. It is at the theme's
/// medium width, centred on the output and never wider or taller than its content frame less the
/// dialog padding. `tokens` changes fields of DialogDefaults, such as the width.
[[nodiscard]] Rect DialogCard(const UiFrame& frame, Dp height,
                              const TokenPatch<DialogTokens>& tokens = {});

// What a dialog says.
struct DialogProps {
	std::string_view title;
	// The body, which wraps to the card's width.
	std::string_view body;
};

// The parts of a dialog's content, for laying them out and for reading where they went.
struct DialogParts {
	// The whole content, which is what to Measure, Arrange and Render. A none handle when refused.
	Element content;
	// The title's text, always on one line, cut short with an ellipsis when it is too wide.
	Element title;
	// The body's text, wrapped to the width. It sits in a clipping box that takes whatever room
	// the title and the buttons leave.
	Element body;
};

/// The content of a dialog as elements: the title, the body wrapped to the width, and a row of
/// `buttons` resting on the bottom edge at the right, with the theme's gap between them and
/// between the parts. Pass the buttons as the elements Button makes, keyed by the caller, so each
/// reports its own activation. Lay it out with ModalScope::Place, which measures it at the content
/// rectangle's width, so the body knows how many lines it has, and arranges it there, or size a
/// card to it first with FitDialogCard, then Render it into the scope's List(). Text is copied into
/// the frame. `key` names it among its siblings.
///
/// The body is text only, so nothing in it can take focus. Anything the player must reach goes in
/// the buttons.
///
/// A dialog with more to say than fits is answerable. The title and the buttons keep their size and
/// the body gives way, scrolling in the room that is left, down to nothing, so the buttons are
/// always on the card. The D-pad's up and down scroll it, as they aren't used by a row of buttons,
/// as do the wheel and a scroll bar. A title wider than the card is cut short with an ellipsis.
/// Buttons that together are wider than the card shrink toward their minimum widths, with their
/// labels cut short with an ellipsis, and stop at the theme's minimum button width, so two buttons
/// can still overflow the card on a very small output. Returns none handles when a part is refused,
/// with `reason` set if given.
[[nodiscard]] DialogParts DialogContent(UiFrame& frame, std::string_view key,
                                        const DialogProps& props, std::span<const Element> buttons,
                                        const ItemLayout& item   = {},
                                        ElementError*     reason = nullptr);

/// Measures `parts.content` for a content rectangle `width` wide, with no limit on its height, and
/// returns the size it wants. Measuring at the width the dialog will have is what lets the body
/// know how many lines it has. ModalScope::Place does it for a scope's own content rectangle.
Vec2 MeasureDialog(UiFrame& frame, const DialogParts& parts, float width);

/// The card for the dialog `parts`, as tall as its content needs and the theme's padding, at the
/// width and in the place DialogCard gives, and never taller than the output allows. A body of
/// several lines gives a taller card than one of a single line. Measures `parts`, so call it before
/// ModalScope::Place, which keeps that measurement when the card it was given has the content width
/// that was measured.
[[nodiscard]] Rect FitDialogCard(UiFrame& frame, const DialogParts& parts,
                                 const TokenPatch<DialogTokens>& tokens = {});

// A modal dialog over the whole screen: a dimming scrim, a raised card, and focus kept inside
// the card while it is open. Build the dialog's content as elements in the same frame while
// Visible, and Render it into List(). When it closes, focus returns to the control that
// had it before it opened. A click on the scrim doesn't close it, since blocking system dialogs
// must be answered. A dialog opened from another is drawn inside the first one's Visible block,
// since two open side by side would each close the other.
class ModalScope {
public:
	/// Opens or keeps open the modal `id` with its card at `card` while `open`, and closes it
	/// when `open` turns false, at once, with focus going back where it was. `frame` is the
	/// caller's, already begun, which carries everything else the dialog needs, and in which the
	/// scrim and the card are built as elements and painted into the modal's window, under a clip
	/// widened to the whole output so the scrim and the card's shadow aren't cut to the card.
	/// `opacity` fades the scrim and card together. For an exit fade, keep `open` true while it
	/// fades and stop drawing the dialog's controls. `card` is in output pixels, a place on the
	/// output as DialogCard and FitDialogCard return, and not a size to be scaled, so size a fixed
	/// card with DialogCard's height in design units. `tokens` changes fields of DialogDefaults for
	/// this dialog, such as its corners. `area` is what the scrim covers and the card's shadow is
	/// held to, the whole output unless given, for a dialog shown over part of a screen such as a
	/// preview. It changes what's drawn only, as the dialog keeps focus from everything else
	/// whichever it is.
	ModalScope(UiFrame& frame, std::string_view id, const Rect& card, bool open,
	           float opacity = 1.0f, const TokenPatch<DialogTokens>& tokens = {},
	           const std::optional<Rect>& area = std::nullopt);
	/// Ends the modal.
	~ModalScope();
	ModalScope(const ModalScope&)            = delete;
	ModalScope& operator=(const ModalScope&) = delete;

	/// True while the modal is showing, so its controls should be drawn.
	[[nodiscard]] bool Visible() const { return m_visible; }
	/// True on the frame the player pressed back, the gamepad's cancel button or Escape, which
	/// the caller usually answers by closing the modal. A dialog that must be answered can ignore
	/// it, and with ConfigureNavigation focus stays where it was.
	[[nodiscard]] bool BackPressed() const { return m_back; }
	/// Calls `callback()` right now, before returning, when back was pressed this frame, and does
	/// nothing otherwise, so the close is reported by the callback on the frame it happens as well
	/// as by BackPressed. Nothing is stored, so the callback need not be trivially copyable and may
	/// capture anything, unlike OnActivate's. It may change state the caller owns, such as closing
	/// the modal.
	template <class F>
	void OnBack(F callback) const {
		static_assert(std::is_invocable_v<F>, "The callback is called with no arguments.");
		if (m_back) {
			callback();
		}
	}
	/// The draw list of the modal's window, to Render the dialog's elements into, which has meaning
	/// only while Visible. When the modal is not showing it is the list of the caller's window,
	/// which nothing of the dialog should be drawn into. A hand-made control is registered with
	/// Interact(frame, ...) while the scope is alive.
	[[nodiscard]] ImDrawList& List() { return *m_list; }
	/// The space inside the card for the dialog's content, inset by the dialog padding.
	[[nodiscard]] Rect Content() const;
	/// Lays out the dialog `parts` in Content(), measured at its width, and returns whether it was
	/// arranged. This is the way to lay a dialog out, since measuring at another width leaves the
	/// body not knowing how many lines it has.
	bool Place(UiFrame& frame, const DialogParts& parts) const;

private:
	// In this order, since the list belongs to the window that opening the modal begins.
	Rect        m_card;
	bool        m_visible = false;
	bool        m_back    = false;
	ImDrawList* m_list    = nullptr;
	// The dialog padding in output pixels, which Content() insets the card by.
	float m_padding = 0.0f;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_MODAL_H_
