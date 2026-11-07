#include "graphics/presentation/system/element/frame.h"

#include "common/logging/log.h"
#include "graphics/presentation/system/element/frameInternals.h"
#include "graphics/presentation/system/imgui/interactContext.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <utility>

namespace Libs::Graphics::SystemUi {

// Big enough that typical labels share a chunk, small enough to waste little on a quiet frame.
static constexpr size_t TEXT_CHUNK_BYTES = 4096;

namespace {

// A kept painted text is looked at for forgetting this often, and forgotten when it has gone this
// many frames without being asked for, about ten seconds at 60 Hz, so a list scrolled back shortly
// after doesn't lay its rows out again.
constexpr uint32_t PAINTED_TEXT_SWEEP_BEGINS  = 64;
constexpr uint32_t PAINTED_TEXT_UNUSED_BEGINS = 600;

// Mixes `value`'s bytes into `seed`, in the way FNV-1a does.
template <class T> uint64_t MixHash(uint64_t seed, const T& value) {
	constexpr uint64_t PRIME = 0x100000001b3ULL;
	const auto*        bytes = reinterpret_cast<const unsigned char*>(&value);
	for (size_t index = 0; index < sizeof(T); index++) {
		seed = (seed ^ bytes[index]) * PRIME;
	}
	return seed;
}

// A hash of what a layout is made from. A match is still checked against the key and the bytes, so
// a collision costs a new layout and never a wrong one, unless two texts collide within a frame,
// when the first's layout is replaced under it. That takes a full 64 bit collision between them.
uint64_t HashOf(const TextLayoutKey& key, std::string_view text) {
	constexpr uint64_t OFFSET = 0xcbf29ce484222325ULL;
	uint64_t           hash   = OFFSET;
	for (const char byte: text) {
		hash = MixHash(hash, byte);
	}
	hash = MixHash(hash, key.font);
	hash = MixHash(hash, key.font_size);
	hash = MixHash(hash, key.line_height);
	hash = MixHash(hash, key.width);
	hash = MixHash(hash, key.density);
	hash = MixHash(hash, key.max_lines);
	hash = MixHash(hash, key.overflow);
	return MixHash(hash, key.shrink_to_fit);
}

// The reasons logged so far, one bit each. It is atomic only so that two reports can't both log.
std::atomic<uint32_t> g_reported_refusals {0};

static_assert(static_cast<uint32_t>(ElementError::NoValue) < 32,
              "every reason needs a bit of its own");

// The name of `why` for the log.
const char* RefusalName(ElementError why) {
	switch (why) {
		case ElementError::None: return "None";
		case ElementError::Stale: return "Stale";
		case ElementError::OutOfRange: return "OutOfRange";
		case ElementError::AlreadyParented: return "AlreadyParented";
		case ElementError::NotBegun: return "NotBegun";
		case ElementError::InvalidPose: return "InvalidPose";
		case ElementError::InvalidDecor: return "InvalidDecor";
		case ElementError::DuplicateKey: return "DuplicateKey";
		case ElementError::IdCollision: return "IdCollision";
		case ElementError::InvalidKey: return "InvalidKey";
		case ElementError::NestedInteractive: return "NestedInteractive";
		case ElementError::AlreadyRendered: return "AlreadyRendered";
		case ElementError::NotInteractive: return "NotInteractive";
		case ElementError::NotClippingBox: return "NotClippingBox";
		case ElementError::NotText: return "NotText";
		case ElementError::BuildingInHook: return "BuildingInHook";
		case ElementError::BuildingInCallback: return "BuildingInCallback";
		case ElementError::AlreadyInteractive: return "AlreadyInteractive";
		case ElementError::NoLabels: return "NoLabels";
		case ElementError::NoValue: return "NoValue";
	}
	return "unknown";
}

} // namespace

bool ReportRefusal(ElementError why) {
	if (why == ElementError::None) {
		return false;
	}
	const uint32_t bit = 1u << static_cast<uint32_t>(why);
	if ((g_reported_refusals.fetch_or(bit) & bit) != 0) {
		return false;
	}
	LOGF("[SystemUi] an element was refused: %s. Later refusals for this reason are not logged.\n",
	     RefusalName(why));
	return true;
}

void RefuseBuild(ElementError why, ElementError* reason) {
	(void)ReportRefusal(why);
	if (reason != nullptr) {
		*reason = why;
	}
}

void ForgetReportedRefusals() {
	g_reported_refusals = 0;
}

UiFrame::UiFrame()  = default;
UiFrame::~UiFrame() = default;

void UiFrame::Begin(const Theme& theme, const ScaleContext& scale, const UiFonts& fonts,
                    InteractionStore& store, UiClock::time_point now) {
	// Starting a frame from Render's hook would clear the tree Paint is about to draw, and from a
	// callback the tree the walk is in the middle of.
	assert(!inHook_ && !inCallback_ && "Begin must not be called from inside Render");
	if (inHook_ || inCallback_) {
		return;
	}
	context_.emplace(UiContext {theme, scale, fonts});
	store_ = &store;
	now_   = now;
	// The serial wraps only after UINT32_MAX frames, and zero belongs to the none handle.
	inCallback_ = false;
	inHook_     = false;
	++serial_;
	if (serial_ == Element::NONE_SERIAL) {
		++serial_;
	}

	nodes_.clear();
	children_.clear();
	items_.clear();
	boxes_.clear();
	texts_.clear();
	images_.clear();
	decor_.clear();
	interacts_.clear();
	customs_.clear();
	rootIds_.clear();
	renderedRoots_.clear();
	elementStack_.clear();
	customBytes_.clear();
	naturalMeasures_ = 0;
	measureCalls_    = 0;
	textRelayouts_   = 0;
	textLayoutsBuilt_ = 0;
	flexItems_.clear();
	flexLengths_.clear();
	flexRects_.clear();
	flexCollapsed_.clear();
	for (TextChunk& chunk: chunks_) {
		chunk.used = 0;
	}
	currentChunk_ = 0;
	if (serial_ % PAINTED_TEXT_SWEEP_BEGINS == 0) {
		std::erase_if(paintedTexts_, [serial = serial_](const auto& entry) {
			return serial - entry.second.last_used > PAINTED_TEXT_UNUSED_BEGINS;
		});
	}
}

const TextLayout& UiFrame::LaidOutText(const TextProps& props, float max_width) const {
	const TextLayoutKey key    = MakeTextLayoutKey(Ui(), props, max_width);
	PaintedText&        entry  = paintedTexts_[HashOf(key, props.text)];
	const bool          usable = entry.last_used != 0 && entry.key == key && entry.text == props.text;
	entry.last_used            = serial_;
	if (!usable) {
		++textLayoutsBuilt_;
		entry.key = key;
		entry.text.assign(props.text);
		entry.layout = LayoutText(Ui(), props, max_width);
	}
	return entry.layout;
}

std::string_view UiFrame::Intern(std::string_view text) {
	if (text.empty()) {
		return {};
	}
	char* dest = ReserveText(text.size());
	std::memcpy(dest, text.data(), text.size());
	return {dest, text.size()};
}

const char* UiFrame::InternCString(std::string_view text) {
	char* dest = ReserveText(text.size() + 1);
	// An empty view may have no data, which memcpy may not be given even to copy nothing.
	if (!text.empty()) {
		std::memcpy(dest, text.data(), text.size());
	}
	dest[text.size()] = '\0';
	return dest;
}

char* UiFrame::ReserveText(size_t size) {
	// Skip chunks that can't take the text. Their tail stays unused until the next Begin, which is
	// cheaper than searching every chunk for a fit.
	while (currentChunk_ < chunks_.size() &&
	       chunks_[currentChunk_].capacity - chunks_[currentChunk_].used < size) {
		++currentChunk_;
	}
	if (currentChunk_ == chunks_.size()) {
		TextChunk chunk;
		// Text longer than a chunk gets a chunk of its own.
		chunk.capacity = std::max(TEXT_CHUNK_BYTES, size);
		chunk.data     = std::make_unique<char[]>(chunk.capacity);
		chunks_.push_back(std::move(chunk));
	}
	TextChunk& chunk = chunks_[currentChunk_];
	char*      dest  = chunk.data.get() + chunk.used;
	chunk.used += size;
	return dest;
}

std::byte* UiFrame::StoreBytes(const void* bytes, size_t size) {
	const std::string_view copy = Intern({static_cast<const char*>(bytes), size});
	// The chunk is the frame's own writable storage, so dropping the view's const is sound.
	return const_cast<std::byte*>(reinterpret_cast<const std::byte*>(copy.data()));
}

size_t UiFrame::ReservedBytes() const {
	size_t total =
	    nodes_.capacity() * sizeof(Node) + children_.capacity() * sizeof(Element) +
	    items_.capacity() * sizeof(ItemLayout) + boxes_.capacity() * sizeof(BoxData) +
	    texts_.capacity() * sizeof(TextData) + images_.capacity() * sizeof(ImageData) +
	    decor_.capacity() * sizeof(DecorData) + interacts_.capacity() * sizeof(InteractData) +
	    customs_.capacity() * sizeof(CustomData) + customBytes_.capacity() +
	    keyScratch_.capacity() * sizeof(const Node*) + idScratch_.capacity() * sizeof(uint32_t) +
	    rootIds_.capacity() * sizeof(RootRecord) + renderedRoots_.capacity() * sizeof(Element) +
	    elementStack_.capacity() * sizeof(Element) + flexItems_.capacity() * sizeof(FlexItem) +
	    flexLengths_.capacity() * sizeof(float) + flexRects_.capacity() * sizeof(Rect) +
	    flexCollapsed_.capacity() + flexScratch_.safe.capacity() * sizeof(FlexItem) +
	    flexScratch_.lengths.capacity() * sizeof(float) + flexScratch_.frozen.capacity() +
	    textCache_.capacity() * sizeof(TextSlot) + listPool_.capacity() * sizeof(ListRows);
	total += paintedTexts_.size() * sizeof(PaintedText);
	for (const TextSlot& slot: textCache_) {
		for (const CachedTextLayout& entry: slot.entries) {
			total += entry.text.capacity();
		}
	}
	for (const TextChunk& chunk: chunks_) {
		total += chunk.capacity;
	}
	for (const ListRows& rows: listPool_) {
		total += rows.rows.capacity() * sizeof(RecordedRow);
	}
	return total;
}

namespace {

// Orders keys so equal ones sit together. Number keys come before text keys.
bool KeyLess(const Node* left, const Node* right) {
	if (left->keyIsText != right->keyIsText) {
		return !left->keyIsText;
	}
	return left->keyIsText ? left->key < right->key : left->keyNumber < right->keyNumber;
}

// Whether two keyed nodes have the same key.
bool KeyEqual(const Node* left, const Node* right) {
	if (left->keyIsText != right->keyIsText) {
		return false;
	}
	return left->keyIsText ? left->key == right->key : left->keyNumber == right->keyNumber;
}

} // namespace

bool UiFrame::HasDuplicateKey(std::span<const Element> children) {
	keyScratch_.clear();
	for (const Element child: children) {
		const Node* node = Find(child);
		if (node != nullptr && node->hasKey) {
			keyScratch_.push_back(node);
		}
	}
	// Sorting finds a repeat in n log n, which keeps a long list of keyed rows cheap.
	std::sort(keyScratch_.begin(), keyScratch_.end(), KeyLess);
	return std::adjacent_find(keyScratch_.begin(), keyScratch_.end(), KeyEqual) !=
	       keyScratch_.end();
}

std::optional<ElementError> UiFrame::ClaimChildren(std::span<const Element> children,
                                                   uint32_t                 parent) {
	// Claim as we go and undo on failure, so a refused node leaves nothing half built. Claiming as
	// we go also catches a child listed twice, whose second look finds it taken.
	size_t                      claimed = 0;
	std::optional<ElementError> failure;
	for (const Element child: children) {
		Node* childNode = Find(child);
		if (childNode == nullptr) {
			failure = Check(child);
			break;
		}
		if (childNode->parent != Node::NO_PARENT) {
			failure = ElementError::AlreadyParented;
			break;
		}
		childNode->parent = parent;
		++claimed;
	}
	if (!failure.has_value() && HasDuplicateKey(children)) {
		failure = ElementError::DuplicateKey;
	}
	if (failure.has_value()) {
		for (size_t i = 0; i < claimed; ++i) {
			Find(children[i])->parent = Node::NO_PARENT;
		}
	}
	return failure;
}

Element UiFrame::AddNode(NodeKind kind, std::span<const Element> children, const ItemLayout& item,
                         ElementError* reason) {
	FrameInternals::AssertNotInCallbackOrHook(*this);
	const auto refuse = [reason](ElementError why) {
		RefuseBuild(why, reason);
		return Element();
	};
	if (const auto why = FrameInternals::BuildRefusal(*this); why.has_value()) {
		return refuse(*why);
	}
	if (!context_.has_value()) {
		return refuse(ElementError::NotBegun);
	}
	const auto index = static_cast<uint32_t>(nodes_.size());
	if (const auto failure = ClaimChildren(children, index); failure.has_value()) {
		return refuse(*failure);
	}

	Node node;
	node.kind       = kind;
	node.firstChild = static_cast<uint32_t>(children_.size());
	node.childCount = static_cast<uint32_t>(children.size());
	node.item       = static_cast<uint32_t>(items_.size());
	items_.push_back(item);
	// Inserting from the frame's own child storage would be undefined once it grows, and only the
	// one-parent check above rules that out, since every stored child already has a parent.
	children_.insert(children_.end(), children.begin(), children.end());
	nodes_.push_back(node);
	return Element(index, serial_);
}

std::optional<ElementError> UiFrame::Check(Element element) const {
	if (element.IsNone()) {
		return ElementError::None;
	}
	if (element.serial_ != serial_) {
		return ElementError::Stale;
	}
	if (element.index_ >= nodes_.size()) {
		return ElementError::OutOfRange;
	}
	return std::nullopt;
}

const Node* UiFrame::Find(Element element) const {
	return Check(element).has_value() ? nullptr : &nodes_[element.index_];
}

Node* UiFrame::Find(Element element) {
	// The const lookup does the checking, and nodes_ is not const here, so the cast is safe.
	return const_cast<Node*>(std::as_const(*this).Find(element));
}

std::span<const Element> UiFrame::Children(const Node& node) const {
	return std::span<const Element>(children_).subspan(node.firstChild, node.childCount);
}

} // namespace Libs::Graphics::SystemUi
