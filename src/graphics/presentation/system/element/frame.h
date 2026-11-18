#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_FRAME_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_FRAME_H_

#include "graphics/presentation/system/core/context.h"
#include "graphics/presentation/system/element/decor.h"
#include "graphics/presentation/system/element/element.h"
#include "graphics/presentation/system/element/node.h"
#include "graphics/presentation/system/element/nodeData.h"
#include "graphics/presentation/system/element/props.h"
#include "graphics/presentation/system/imgui/interaction.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics::SystemUi {

// Why a handle was refused.
enum class ElementError : uint8_t {
	None,               // the handle refers to nothing
	Stale,              // made in an earlier frame
	OutOfRange,         // names a node the frame doesn't have
	AlreadyParented,    // a child that already has a parent, or is listed twice
	NotBegun,           // the frame has not had Begin called yet
	InvalidPose,        // a pose scale that is not positive and finite, or an alpha that is NaN
	InvalidDecor,       // a shine or focus ring strength that is NaN
	DuplicateKey,       // two children of one node share a key
	IdCollision,        // two siblings, or two roots in one frame, would get the same id
	InvalidKey,         // an empty key
	NestedInteractive,  // an interactive node inside another, or holding one
	AlreadyRendered,    // a root that Render has already drawn this frame
	NotInteractive,     // a callback asked of a node that MakeInteractive has not made interactive
	NotClippingBox,     // a list asked of a node that is not a box with Overflow::Clip
	NotText,            // a text colour asked of a node that is not a text node
	BuildingInHook,     // a node built while Render's afterInteract hook was running
	BuildingInCallback, // the tree built or changed while one of Render's callbacks was running
	AlreadyInteractive, // a second MakeInteractive on a node, which would drop what the first set
	NoLabels,           // a tab bar given no labels
	NoValue,            // a control with no value, kept last as frame.cpp checks each has a bit
};

/// Logs that the element layer refused something for the reason `why`, so a tree that quietly drew
/// nothing can be traced. Most callers drop the refusal, and a refused tree simply doesn't show. Each
/// reason is logged once per run, as a refusal usually repeats every frame. None isn't logged, since
/// a none handle is usually what an earlier refusal handed on, or an optional part left unset.
/// Returns whether this was the first report of the reason.
bool ReportRefusal(ElementError why);

/// Reports `why` and sets `reason` to it if given, for a component refusing before it builds a
/// node.
void RefuseBuild(ElementError why, ElementError* reason);

// A tree whose ids AssignIds gave this frame, and its root's id.
struct RootRecord {
	Element  element;
	uint32_t id = 0;
};

// Everything one frame of system UI needs, passed once instead of a context, an interaction
// context and a store. It owns the element tree's storage and keeps its capacity from frame to
// frame, so a steady-state frame allocates nothing. A frame belongs to the render or ImGui thread
// and is not thread safe.
// Recorded rows of a list, internal to the element layer and complete only where it is defined.
struct ListRows;

class UiFrame {
public:
	/// Defined in frame.cpp, where the list pool's element type is complete.
	UiFrame();
	~UiFrame();
	// A frame hands out handles into its own storage, so copying one would share nothing useful.
	UiFrame(const UiFrame&)            = delete;
	UiFrame& operator=(const UiFrame&) = delete;

	/// Starts a new frame. Clears the nodes, children and interned text but keeps their capacity,
	/// and bumps the serial so every handle from an earlier frame is stale. `theme` and `store`
	/// must last until the next Begin, so pass a named theme, never a temporary. Called from one of
	/// Render's callbacks or its afterInteract hook it does nothing, and debug builds assert.
	void Begin(const Theme& theme, const ScaleContext& scale, const UiFonts& fonts,
	           InteractionStore& store, UiClock::time_point now);
	/// Refuses a temporary theme, which would dangle as soon as the call returned.
	void Begin(const Theme&& theme, const ScaleContext& scale, const UiFonts& fonts,
	           InteractionStore& store, UiClock::time_point now) = delete;

	/// The context the older DrawBox and LayoutText functions take. Valid after Begin, which only
	/// debug builds check.
	[[nodiscard]] const UiContext& Ui() const {
		assert(context_.has_value() && "Begin has not been called");
		return *context_;
	}
	/// The theme of this frame. Named for what it is because Theme is also the type.
	[[nodiscard]] const Theme& ActiveTheme() const { return Ui().theme; }
	/// The scale of this frame.
	[[nodiscard]] const ScaleContext& Scale() const { return Ui().scale; }
	/// The store that animates this frame's controls. Valid after Begin, which only debug builds
	/// check.
	[[nodiscard]] InteractionStore& Store() const {
		assert(store_ != nullptr && "Begin has not been called");
		return *store_;
	}
	/// How many times Begin has restarted the frame, which changes with every Begin. A caller that
	/// wants to know the frame was begun again since it last looked compares it with what it saw,
	/// as the gallery does to catch a frame that was not begun for the ImGui frame it draws in.
	[[nodiscard]] uint32_t BeginCount() const { return serial_; }
	/// The time this frame was begun at.
	[[nodiscard]] UiClock::time_point Now() const { return now_; }

	/// Copies `text` into frame storage and returns a view of the copy. The copy stays at the same
	/// address until the next Begin, so a pointer taken from the view may be kept that long, and
	/// views from earlier calls stay valid however much is interned after them.
	[[nodiscard]] std::string_view Intern(std::string_view text);

	/// Copies `size` bytes at `bytes` into frame storage and returns the copy, or null for none.
	/// Like text from Intern the copy stays where it is until the next Begin, however much is
	/// stored after it, and the caller may write to it. It is not aligned for any type, so objects
	/// are read and written with memcpy. A component uses it to give a trivially copyable callable
	/// something variable-sized to point at, such as a table of handles or a callback's bytes.
	[[nodiscard]] std::byte* StoreBytes(const void* bytes, size_t size);

	/// A stack of elements a component can build a run of children on, so it doesn't allocate a
	/// vector for them each frame. It keeps its capacity from frame to frame, and Begin empties it.
	/// A component notes its size, pushes its children, uses them, and resizes it back to the noted
	/// size, so a child that builds a run of its own in between does so above the first's. Use
	/// indices and not pointers into it, as pushing may move it.
	[[nodiscard]] std::vector<Element>& ElementStack() { return elementStack_; }

	/// `props` laid out in `max_width` pixels, as LayoutText lays it out, for text drawn by a Custom
	/// node or measured while building, which has no text node to keep its layout. A layout kept from
	/// an earlier frame is reused when the text and what it depends on are the same, so a steady frame
	/// allocates nothing and measures no glyphs. The result lasts until the next Begin. It is const so
	/// a Custom node's paint, which gets a const frame, can ask, and belongs to the render thread like
	/// the rest of the frame. One not asked for in a while is forgotten. It suits text that repeats,
	/// as text that changes every frame would pile up until it is forgotten. Such text belongs in a
	/// Text node, which keeps a few layouts and no more.
	[[nodiscard]] const TextLayout& LaidOutText(const TextProps& props, float max_width) const;

	/// Copies `text` into frame storage with a terminating NUL, for an ImGui call that wants a C
	/// string. Like Intern, the copy stays where it is until the next Begin.
	[[nodiscard]] const char* InternCString(std::string_view text);

	/// The total capacity of the frame's storage in bytes, for tests and benchmarks.
	[[nodiscard]] size_t ReservedBytes() const;

private:
	// The primitives reach the node storage through FrameInternals, which keeps AddNode, Find and
	// Children out of the component API without making them public here. A friend suits better
	// than a detail namespace because the access has to reach private members.
	friend struct FrameInternals;
	// Lets tests forge handles and move the serial, which the primitives must not be able to do.
	friend struct FrameTestHooks;

	// Room for `size` bytes of interned text, which stay put until the next Begin.
	char* ReserveText(size_t size);

	// A run of interned text. Chunks are never resized, so earlier views stay put.
	struct TextChunk {
		std::unique_ptr<char[]> data;
		size_t                  capacity = 0;
		size_t                  used     = 0;
	};

	// These are documented on their forwarders in FrameInternals.
	Element AddNode(NodeKind kind, std::span<const Element> children, const ItemLayout& item,
	                ElementError* reason);
	bool    HasDuplicateKey(std::span<const Element> children);
	std::optional<ElementError> ClaimChildren(std::span<const Element> children, uint32_t parent);
	Node*                       Find(Element element);
	[[nodiscard]] const Node*   Find(Element element) const;
	[[nodiscard]] std::optional<ElementError> Check(Element element) const;
	[[nodiscard]] std::span<const Element>    Children(const Node& node) const;

	std::optional<UiContext> context_;
	InteractionStore*        store_ = nullptr;
	UiClock::time_point      now_ {};
	// Starts at 1 so a handle made before Begin never matches the none serial.
	uint32_t serial_ = 1;

	std::vector<Node>    nodes_;
	std::vector<Element> children_;
	// The keyed children of the node being added, sorted to find a repeated key. Kept at capacity.
	std::vector<const Node*> keyScratch_;
	// The ids of one node's children while AssignIds checks them, and the ids of the roots it has
	// given this frame. Kept at capacity, and the roots are cleared by Begin.
	std::vector<uint32_t>   idScratch_;
	std::vector<RootRecord> rootIds_;
	// The roots Render has drawn this frame, so a second Render of one is refused.
	std::vector<Element> renderedRoots_;
	// The scratch stack of ElementStack, emptied by Begin and kept at capacity.
	std::vector<Element> elementStack_;
	// Set while Render runs a node's callbacks, which must not build or change elements, since that
	// could move the storage the callbacks run from, or the children the walk is going through. Debug
	// builds assert on it, and every build refuses.
	bool inCallback_ = false;
	// Set while Render runs its afterInteract hook, which may restyle nodes but must not add
	// any. Debug builds assert on it, and AddNode refuses in every build.
	bool inHook_ = false;
	// The rows of the lists Render is walking, one entry for each depth of nesting, kept at
	// capacity between frames.
	std::vector<ListRows>  listPool_;
	std::vector<TextChunk> chunks_;
	size_t                 currentChunk_ = 0;

	// Per-node data and layout scratch, cleared with the nodes and kept at capacity.
	std::vector<ItemLayout>   items_;
	std::vector<BoxData>      boxes_;
	std::vector<TextData>     texts_;
	// The text layouts kept across frames, one slot per text position. Begin leaves them alone.
	std::vector<TextSlot> textCache_;
	// The layouts LaidOutText keeps, by a hash of what they show. Mutable as it is a cache, which a
	// const frame's paint fills. Begin leaves it alone, apart from forgetting what is long unused.
	mutable std::unordered_map<uint64_t, PaintedText> paintedTexts_;
	std::vector<ImageData>    images_;
	std::vector<DecorData>    decor_;
	std::vector<InteractData> interacts_;
	std::vector<CustomData>   customs_;
	// The bytes of the custom nodes' callables, each at an aligned offset. Offsets and not
	// pointers, since growing the vector moves the bytes.
	std::vector<std::byte> customBytes_;
	std::vector<FlexItem>  flexItems_;
	std::vector<float>     flexLengths_;
	std::vector<Rect>      flexRects_;
	// One flag per child of each box being solved, a stack like the lengths and rectangles, since a
	// box reads its flags while its children solve theirs.
	std::vector<uint8_t> flexCollapsed_;
	// The solver's working space. A solve never starts another, so one is enough at any depth.
	FlexScratch flexScratch_;

	// Which measure pass the frame is in. Each top-level Measure starts a new one, and a node's
	// natural size is only reused within the pass that made it, since a later pass may offer
	// different space and leave the subtree's state out of date.
	uint32_t measureGeneration_ = 0;
	// Counts of work done this frame, for tests of the memo and of text layout.
	uint32_t naturalMeasures_ = 0;
	uint32_t measureCalls_    = 0;
	uint32_t textRelayouts_   = 0;
	mutable uint32_t textLayoutsBuilt_ = 0;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_ELEMENT_FRAME_H_
