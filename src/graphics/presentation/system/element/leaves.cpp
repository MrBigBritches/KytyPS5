#include "graphics/presentation/system/element/leaves.h"

#include "graphics/presentation/system/element/frameInternals.h"
#include "imgui.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <utility>

namespace Libs::Graphics::SystemUi {

namespace {

// A text laid out for a width this close to the one it is given needs no new layout. Arranging
// snaps widths to whole pixels, so anything under half a pixel is rounding and not a new width.
constexpr float RELAYOUT_EPSILON = 0.5f;

// How wide the widest unbreakable word of `layout`'s text is. Words end at spaces and line breaks,
// which is where wrapped text can break, so this is the least width the text can take without a
// word being cut.
float LongestWord(const TextLayout& layout, std::string_view text) {
	if (layout.font == nullptr) {
		return 0.0f;
	}
	float widest = 0.0f;
	for (size_t start = 0; start < text.size();) {
		const size_t end = std::min(text.find_first_of(" \n", start), text.size());
		if (end > start) {
			const char* begin = text.data() + start;
			const float width =
			    layout.font
			        ->CalcTextSizeA(layout.font_size, FLT_MAX, 0.0f, begin, begin + (end - start))
			        .x;
			widest = std::max(widest, width);
		}
		start = end + 1;
	}
	return widest;
}

// The width of the ellipsis `layout`'s font would draw, the least a truncated line can show. It
// is the font's own character when it has one, or three dots.
float EllipsisWidth(const TextLayout& layout) {
	if (layout.font == nullptr) {
		return 0.0f;
	}
	constexpr ImWchar          ELLIPSIS      = 0x2026;
	constexpr std::string_view ELLIPSIS_UTF8 = "\xE2\x80\xA6";
	const std::string_view     mark = layout.font->IsGlyphInFont(ELLIPSIS) ? ELLIPSIS_UTF8 : "...";
	return layout.font
	    ->CalcTextSizeA(layout.font_size, FLT_MAX, 0.0f, mark.data(), mark.data() + mark.size())
	    .x;
}

// The least width a text can take, which depends on how it gives way. Wrapped text can't go
// below its longest word. Truncated text can go down to its ellipsis, and clipped text to nothing,
// as CSS lets text that truncates or clips shrink below its content.
float MinimumWidth(const TextLayout& layout, const TextProps& props) {
	switch (props.overflow) {
		case TextOverflow::Wrap: return LongestWord(layout, props.text);
		case TextOverflow::Ellipsis: return EllipsisWidth(layout);
		case TextOverflow::Clip: break;
	}
	return 0.0f;
}

// The size an image takes from its explicit width and height. One axis alone is completed from
// the texture's shape and snapped to a whole pixel, and neither gives nothing to measure.
Vec2 ImageSize(const ScaleContext& scale, const ItemLayout& item, Vec2 texture) {
	const bool  shaped = texture.x > 0.0f && texture.y > 0.0f;
	const float width  = item.width ? ToPixels(scale, *item.width) : 0.0f;
	const float height = item.height ? ToPixels(scale, *item.height) : 0.0f;
	if (item.width && item.height) {
		return {width, height};
	}
	if (item.width) {
		return {width, shaped ? SnapPixel(width * texture.y / texture.x) : 0.0f};
	}
	if (item.height) {
		return {shaped ? SnapPixel(height * texture.x / texture.y) : 0.0f, height};
	}
	return {};
}

// The corner of a layout rectangle under `transform`. At a scale of 1 only the translation is
// rounded, which for the whole-pixel corners layout produces is rounding the corner, and it leaves
// text's own half-pixel placement alone.
Vec2 MapCorner(Vec2 corner, const PaintTransform& transform) {
	// Poses that cancel, like 0.8 then 1.25, multiply to a hair off 1 and still rest on the pixels.
	constexpr float IDENTITY_SCALE  = 1.0f;
	constexpr float SCALE_TOLERANCE = 1e-5f;
	if (std::fabs(transform.scale - IDENTITY_SCALE) <= SCALE_TOLERANCE) {
		return {corner.x + SnapPixel(transform.translate.x),
		        corner.y + SnapPixel(transform.translate.y)};
	}
	return {corner.x * transform.scale + transform.translate.x,
	        corner.y * transform.scale + transform.translate.y};
}

// Makes the layout of `data`'s text in `limit` the node's own, reusing one its slot kept when the
// text and everything it depends on are the same, and returns it. Making one is the only step that
// allocates, and measures every glyph, so a steady frame does neither.
const CachedTextLayout& LayOut(UiFrame& frame, const Node& node, TextData& data, float limit) {
	const TextLayoutKey key  = MakeTextLayoutKey(frame.Ui(), data.props, limit);
	TextSlot&           slot = FrameInternals::TextCache(frame)[node.payload];
	data.laidOut             = true;
	for (uint8_t index = 0; index < slot.entries.size(); index++) {
		const CachedTextLayout& entry = slot.entries[index];
		if (entry.used && entry.key == key && entry.text == data.props.text) {
			data.entry = index;
			return entry;
		}
	}
	FrameInternals::CountTextLayoutBuilt(frame);
	data.entry             = slot.next;
	slot.next              = static_cast<uint8_t>((slot.next + 1) % slot.entries.size());
	CachedTextLayout& made = slot.entries[data.entry];
	made.used              = true;
	made.key               = key;
	made.text.assign(data.props.text);
	made.layout    = LayoutText(frame.Ui(), data.props, limit);
	made.min_width = MinimumWidth(made.layout, data.props);
	return made;
}

// The layout `data` last asked for, or an empty one for a text never measured.
const TextLayout& ActiveLayout(const UiFrame& frame, const Node& node, const TextData& data) {
	static const TextLayout NOT_LAID_OUT;
	return data.laidOut ? FrameInternals::TextCache(frame)[node.payload].entries[data.entry].layout
	                    : NOT_LAID_OUT;
}

} // namespace

Rect MapRect(const Rect& rect, const PaintTransform& transform) {
	const Vec2 corner = MapCorner(rect.Position(), transform);
	return {corner.x, corner.y, rect.width * transform.scale, rect.height * transform.scale};
}

Rect MapCentred(const Rect& rect, const PaintTransform& transform) {
	// The centre is mapped like any point and the size is kept, so text keeps its glyph sizes.
	const Vec2 centre = rect.Center();
	const Vec2 mapped = MapCorner(centre, transform);
	return {mapped.x - rect.width / 2.0f, mapped.y - rect.height / 2.0f, rect.width, rect.height};
}

Element Text(UiFrame& frame, const TextProps& props, const ItemLayout& item, ElementError* reason) {
	const Element text = FrameInternals::AddNode(frame, NodeKind::Text, {}, item, reason);
	if (text.IsNone()) {
		return text;
	}
	auto& texts                                = FrameInternals::Texts(frame);
	FrameInternals::Find(frame, text)->payload = static_cast<uint32_t>(texts.size());
	// The caller's view may not outlive the call, so the frame keeps its own copy.
	TextData data;
	data.props      = props;
	data.props.text = frame.Intern(props.text);
	texts.push_back(std::move(data));
	auto& slots = FrameInternals::TextCache(frame);
	if (slots.size() < texts.size()) {
		slots.resize(texts.size());
	}
	return text;
}

Element Image(UiFrame& frame, ImTextureData* texture, Vec2 texture_size, ImageFit fit, Dp radius,
              const ItemLayout& item, ElementError* reason) {
	const Element image = FrameInternals::AddNode(frame, NodeKind::Image, {}, item, reason);
	if (image.IsNone()) {
		return image;
	}
	auto& images                                = FrameInternals::Images(frame);
	FrameInternals::Find(frame, image)->payload = static_cast<uint32_t>(images.size());
	images.push_back({texture, texture_size, fit, radius});
	return image;
}

NodeSize MeasureText(UiFrame& frame, const Node& node, Vec2 max) {
	TextData&         data  = FrameInternals::Texts(frame)[node.payload];
	const ItemLayout& item  = FrameInternals::Item(frame, node);
	const auto&       scale = frame.Scale();
	// An explicit width is the width the text lays out in, wherever the space on offer ends.
	const float limit = item.width ? ToPixels(scale, *item.width) : max.x;
	const CachedTextLayout& laid_out = LayOut(frame, node, data, limit);
	// Text narrower than its limit is laid out the same at its own width, so recording that width
	// spares Arrange a new layout when it gives the text exactly the width it took.
	// A text with an explicit width was laid out in exactly that width, which Arrange gives it
	// again, so the narrower text it took is no reason to lay it out a second time.
	data.laidOutWidth = item.width ? limit
	                               : (std::isfinite(limit) ? std::min(limit, laid_out.layout.size.x)
	                                                       : laid_out.layout.size.x);

	Vec2 size = laid_out.layout.size;
	// The minimum height is the laid-out height, since text can't shrink vertically.
	Vec2 min {laid_out.min_width, size.y};
	if (item.width) {
		size.x = min.x = limit;
	}
	if (item.height) {
		size.y = min.y = ToPixels(scale, *item.height);
	}
	return {size, min};
}

Element AddCustom(UiFrame& frame, const ItemLayout& item, CustomInvoke invoke, const void* bytes,
                  size_t size, size_t alignment, ElementError* reason) {
	const Element custom = FrameInternals::AddNode(frame, NodeKind::Custom, {}, item, reason);
	if (custom.IsNone()) {
		return custom;
	}
	const uint32_t offset = StoreCallable(frame, bytes, size, alignment);
	// AddNode refuses in the hook and in callbacks first, so this is a second line of defence. It
	// keeps a record holding NO_CALLABLE from ever reaching Paint. The node is left without a payload and the
	// caller gets no handle to it.
	assert(offset != NO_CALLABLE && "a custom node must not be built where nothing can be stored");
	if (offset == NO_CALLABLE) {
		RefuseBuild(FrameInternals::BuildRefusal(frame).value_or(ElementError::BuildingInHook),
		            reason);
		return {};
	}
	auto& customs                                = FrameInternals::Customs(frame);
	FrameInternals::Find(frame, custom)->payload = static_cast<uint32_t>(customs.size());
	customs.push_back({invoke, offset});
	return custom;
}

uint32_t StoreCallable(UiFrame& frame, const void* bytes, size_t size, size_t alignment) {
	FrameInternals::AssertNotInCallbackOrHook(frame);
	if (FrameInternals::BuildRefusal(frame).has_value()) {
		return NO_CALLABLE;
	}
	// The storage starts aligned to the default new alignment, which the templates checked the
	// callable does not exceed, so rounding the offset up aligns the callable itself.
	auto&        storage = FrameInternals::CustomBytes(frame);
	const size_t offset  = (storage.size() + alignment - 1) / alignment * alignment;
	// The offset is stored in 32 bits, which 4 GiB of callables would overflow.
	assert(offset + size <= UINT32_MAX && "callable storage is over 4 GiB");
	storage.resize(offset + size);
	std::memcpy(storage.data() + offset, bytes, size);
	return static_cast<uint32_t>(offset);
}

const std::byte* StoredBytes(const UiFrame& frame, uint32_t offset) {
	return FrameInternals::CustomBytes(frame).data() + offset;
}

NodeSize MeasureCustom(const UiFrame& frame, const Node& node) {
	const ItemLayout& item  = FrameInternals::Item(frame, node);
	const auto&       scale = frame.Scale();
	// An axis with no explicit length has nothing to measure, so it takes none.
	const Vec2 size {item.width ? ToPixels(scale, *item.width) : 0.0f,
	                 item.height ? ToPixels(scale, *item.height) : 0.0f};
	return {size, size};
}

void PaintCustom(const UiFrame& frame, const Node& node, ImDrawList& list, const PaintState& state,
                 const Rect& painted) {
	const CustomData& data  = FrameInternals::Customs(frame)[node.payload];
	const ItemInput   input = node.decor == Node::NO_DECOR
	                              ? ItemInput {}
	                              : FrameInternals::Decors(frame)[node.decor].input;
	const CustomPaint paint {list, painted, state.transform, state.opacity, frame, node.id, input};
	data.invoke(StoredBytes(frame, data.offset), paint);
}

NodeSize MeasureImage(const UiFrame& frame, const Node& node) {
	const ImageData& data = FrameInternals::Images(frame)[node.payload];
	const Vec2 size = ImageSize(frame.Scale(), FrameInternals::Item(frame, node), data.textureSize);
	return {size, size};
}

void ArrangeText(UiFrame& frame, const Node& node, const Rect& rect) {
	TextData& data = FrameInternals::Texts(frame)[node.payload];
	if (std::fabs(rect.width - data.laidOutWidth) <= RELAYOUT_EPSILON) {
		return;
	}
	FrameInternals::CountTextRelayout(frame);
	(void)LayOut(frame, node, data, rect.width);
	data.laidOutWidth = rect.width;
}

void PaintText(const UiFrame& frame, const Node& node, ImDrawList& list, const PaintState& state) {
	const TextData&   data   = FrameInternals::Texts(frame)[node.payload];
	const TextLayout& layout = ActiveLayout(frame, node, data);
	// Text taller than its rectangle starts at the top and spills below, so its first line is
	// never lost above the rectangle.
	const float spare = std::max(0.0f, node.arranged.height - layout.size.y);
	const Rect  drawn {node.arranged.x, node.arranged.y + spare / 2.0f, node.arranged.width,
                      layout.size.y};
	DrawText(list, layout, data.props, MapCentred(drawn, state.transform), state.opacity);
}

void PaintImage(const UiFrame& frame, const Node& node, ImDrawList& list, const PaintState& state) {
	const ImageData& data = FrameInternals::Images(frame)[node.payload];
	DrawImage(list, frame.Ui(), data.texture, data.textureSize,
	          MapRect(node.arranged, state.transform), data.radius.value, data.fit, state.opacity);
}

} // namespace Libs::Graphics::SystemUi
