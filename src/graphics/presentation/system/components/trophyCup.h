#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TROPHYCUP_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TROPHYCUP_H_

#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/core/color.h"
#include "graphics/presentation/system/core/tokenPatch.h"
#include "graphics/presentation/system/element/frame.h"
#include "graphics/presentation/system/element/props.h"

namespace Libs::Graphics::SystemUi {

// A trophy cup's colour and size.
struct TrophyCupProps {
	// The cup's colour, such as a trophy's grade. Transparent draws nothing.
	Color color;
	// Fields of ToastDefaults to change for this cup, such as its width.
	TokenPatch<ToastTokens> tokens;
};

/// A trophy cup as an element, drawn in the rectangle it is laid out in with the opacity and pose
/// it inherits. An axis that `item` leaves without a size takes the cup's natural size from the
/// toast tokens, 22 by 21 design units, so a bare cup is as big as the one before a toast's title.
/// Returns a none handle when the node is refused, with `reason` set if given.
[[nodiscard]] Element TrophyCup(UiFrame& frame, const TrophyCupProps& props,
                                const ItemLayout& item = {}, ElementError* reason = nullptr);

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_COMPONENTS_TROPHYCUP_H_
