#include "graphics/presentation/system/components/trophyCup.h"

#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/imgui/convert.h"

namespace Libs::Graphics::SystemUi {

namespace {

// The cup is drawn on a 22 by 21 grid, the size the overlay drew it at before, centred on the
// bowl's top-middle point (11, 11). Each part is a rectangle on that grid.
constexpr float GRID_WIDTH    = 22.0f;
constexpr float GRID_HEIGHT   = 21.0f;
constexpr float CENTRE_X      = 11.0f;
constexpr float CENTRE_Y      = 11.0f;
constexpr Rect  BOWL          = {-8.0f, -11.0f, 16.0f, 13.0f};
constexpr Rect  LEFT_HANDLE   = {-11.0f, -9.0f, 5.0f, 6.0f};
constexpr Rect  RIGHT_HANDLE  = {6.0f, -9.0f, 5.0f, 6.0f};
constexpr Rect  STEM          = {-1.5f, 2.0f, 3.0f, 5.0f};
constexpr Rect  BASE          = {-5.0f, 7.0f, 10.0f, 3.0f};
constexpr float CORNER        = 2.0f;
constexpr float HANDLE_STROKE = 1.8f;
constexpr float BASE_CORNER   = 1.5f;

// Draws a trophy cup filling `rect`, in `color` faded by `opacity`. Its handles reach the
// rectangle's sides and its base sits on the rectangle's bottom.
void DrawTrophyCup(ImDrawList& list, const Rect& rect, Color color, float opacity) {
	const Color tint = color.MulAlpha(opacity);
	if (tint.a == 0 || rect.width <= 0.0f || rect.height <= 0.0f) {
		return;
	}
	const float unit_x = rect.width / GRID_WIDTH;
	const float unit_y = rect.height / GRID_HEIGHT;
	const float unit   = unit_x < unit_y ? unit_x : unit_y;
	const ImU32 packed = ToImU32(tint);
	// A part of the grid, placed in `rect`.
	const auto part = [&](const Rect& grid) {
		return Rect {rect.x + (CENTRE_X + grid.x) * unit_x, rect.y + (CENTRE_Y + grid.y) * unit_y,
		             grid.width * unit_x, grid.height * unit_y};
	};
	const auto bowl  = part(BOWL);
	const auto left  = part(LEFT_HANDLE);
	const auto right = part(RIGHT_HANDLE);
	const auto stem  = part(STEM);
	const auto base  = part(BASE);
	list.AddRectFilled(TopLeft(bowl), BottomRight(bowl), packed, CORNER * unit,
	                   ImDrawFlags_RoundCornersBottom);
	list.AddRect(TopLeft(left), BottomRight(left), packed, CORNER * unit, HANDLE_STROKE * unit,
	             ImDrawFlags_RoundCornersLeft);
	list.AddRect(TopLeft(right), BottomRight(right), packed, CORNER * unit, HANDLE_STROKE * unit,
	             ImDrawFlags_RoundCornersRight);
	list.AddRectFilled(TopLeft(stem), BottomRight(stem), packed);
	list.AddRectFilled(TopLeft(base), BottomRight(base), packed, BASE_CORNER * unit);
}

// Draws the cup in the node's painted rectangle. Trivially copyable so Custom can store it.
struct CupPaint {
	Color color;

	void operator()(const CustomPaint& painting) const {
		DrawTrophyCup(painting.list, painting.painted, color, painting.opacity);
	}
};

} // namespace

Element TrophyCup(UiFrame& frame, const TrophyCupProps& props, const ItemLayout& item,
                  ElementError* reason) {
	const auto toast = props.tokens.Apply(ToastDefaults(frame.ActiveTheme()));
	ItemLayout sized = item;
	if (!sized.width) {
		sized.width = Dp(toast.cup_width);
	}
	if (!sized.height) {
		sized.height = Dp(toast.cup_height);
	}
	return Custom(frame, sized, CupPaint {props.color}, reason);
}

} // namespace Libs::Graphics::SystemUi
