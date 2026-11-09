#include "graphics/shader/storegen/computePredictor.h"

#include "graphics/shader/moduleId.h"

#include <algorithm>
#include <array>
#include <numeric>

namespace Libs::Graphics::StoreGen {

namespace {

namespace IR = ShaderRecompiler::IR;

static_assert(IDENTITY_BUFFER_SWIZZLE == DstSel(4, 5, 6, 7),
              "the identity swizzle is the specialization default");

// The sl00 element kinds of a formatted buffer or an image.
constexpr uint64_t KIND_FLOAT_ZERO = 0;
constexpr uint64_t KIND_FLOAT      = 1;
constexpr uint64_t KIND_INT        = 5;
constexpr uint64_t KIND_UINT       = 9;

// A formatted buffer's element, by sl00 kind and component count.
// The x1 float, int, uint and x4 float rows were seen on run A. The others assume the game picks
// the natural 32-bit format, with missing components reading 0 and W 1. Formats are
// Prospero::BufferFormat values, and the swizzles are DstSel words.
struct TypedRow {
	uint64_t     kind;
	uint64_t     count;
	BufferFields fields;
};
constexpr std::array<TypedRow, 12> TYPED = {{
    {KIND_FLOAT, 1, {4, 22, 516}},
    {KIND_INT, 1, {4, 21, 516}},
    {KIND_UINT, 1, {4, 20, 516}},
    {KIND_FLOAT, 4, {16, 77, 4012}},
    {KIND_FLOAT, 2, {8, 64, 556}},
    {KIND_FLOAT, 3, {12, 74, 940}},
    {KIND_UINT, 2, {8, 62, 556}},
    {KIND_UINT, 3, {12, 72, 940}},
    {KIND_UINT, 4, {16, 75, 4012}},
    {KIND_INT, 2, {8, 63, 556}},
    {KIND_INT, 3, {12, 73, 940}},
    {KIND_INT, 4, {16, 76, 4012}},
}};
static_assert(static_cast<uint32_t>(Prospero::BufferFormat::k32Float) == 22);
static_assert(static_cast<uint32_t>(Prospero::BufferFormat::k32_32Float) == 64);
static_assert(static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32Float) == 74);
static_assert(static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) == 77);

constexpr uint64_t THREE_COMPONENTS = 3;
constexpr uint64_t FOUR_COMPONENTS  = 4;

// The formatted buffer row of an sl00 element kind and component count.
// @return the row, or null when the table has none.
const BufferFields* FindTyped(uint64_t kind, uint64_t count) {
	for (const auto& row: TYPED) {
		if (row.kind == kind && row.count == count) return &row.fields;
	}
	return nullptr;
}

// The TextureNumericClass of an image's element kind. Kind 0 appears on float4 textures and is
// taken as float.
std::optional<Prospero::TextureNumericClass> NumericClassOfKind(uint64_t kind) {
	switch (kind) {
		case KIND_FLOAT_ZERO:
		case KIND_FLOAT: return Prospero::TextureNumericClass::Float;
		case KIND_UINT: return Prospero::TextureNumericClass::Uint;
		case KIND_INT: return Prospero::TextureNumericClass::Sint;
		default: return std::nullopt;
	}
}

// The slot a user or EUD descriptor occupies in the header tables.
// @return nothing for a literal, a key with no slot, or an offset that two slots share.
std::optional<SlotRef> SlotOfKey(const DescriptorKey& key, const HeaderSlots& slots) {
	const auto offset = SlotOffsetOfKey(key, slots);
	return offset ? SlotAtOffset(slots, *offset) : std::nullopt;
}

} // namespace

std::string DescriptorKeyText(const DescriptorKey& key) {
	switch (key.kind) {
		case DescriptorKeyKind::User: return "('user', " + std::to_string(key.base) + ")";
		case DescriptorKeyKind::Eud:
			return "('eud', " + std::to_string(key.base) + ", " + std::to_string(key.offset) + ")";
		case DescriptorKeyKind::Literal:
			return "('literal', " + std::to_string(key.base) + ", " + std::to_string(key.offset) +
			       ", " + std::to_string(key.literal_word2) + ", " +
			       std::to_string(key.literal_word3) + ")";
	}
	return {};
}

std::optional<uint32_t> SlotOffsetOfKey(const DescriptorKey& key, const HeaderSlots& slots) {
	switch (key.kind) {
		case DescriptorKeyKind::User: return key.base;
		case DescriptorKeyKind::Eud:
			// Only the table the header names is a slot table, and only a dword-aligned offset
			// names a slot.
			if (!slots.eud_pointer || *slots.eud_pointer != key.base ||
			    key.offset % sizeof(uint32_t) != 0) {
				return std::nullopt;
			}
			return EUD_FIRST_DWORD + key.offset / static_cast<uint32_t>(sizeof(uint32_t));
		case DescriptorKeyKind::Literal: return std::nullopt;
	}
	return std::nullopt;
}

const Sl00Resource* ResourceAtSlot(const SlotRef& slot, std::span<const Sl00Resource> records) {
	const Sl00Resource* found = nullptr;
	for (const auto& record: records) {
		// System values such as threadID share the record layout and a register number, so records
		// of the Other and Unreadable kinds are skipped.
		if (record.register_index != slot.index || record.resource_class != slot.sharp_class ||
		    record.shape.kind == Sl00ShapeKind::Other ||
		    record.shape.kind == Sl00ShapeKind::Unreadable) {
			continue;
		}
		if (found != nullptr) return nullptr;
		found = &record;
	}
	return found;
}

std::optional<BufferFields> BoundBufferFields(const BufferFacts&  buffer,
                                              const Sl00Resource* record) {
	if (buffer.key.kind == DescriptorKeyKind::Literal) {
		// A literal pool table has no stride, and no rule covers a formatted access to one.
		return buffer.formatted ? std::nullopt
		                        : std::optional<BufferFields> {{0, 0, IDENTITY_BUFFER_SWIZZLE}};
	}
	if (record == nullptr) return std::nullopt;
	const auto& shape = record->shape;
	switch (shape.kind) {
		case Sl00ShapeKind::ConstantBuffer:
			return BufferFields {CONSTANT_BUFFER_STRIDE, 0, IDENTITY_BUFFER_SWIZZLE};
		case Sl00ShapeKind::Raw: return BufferFields {BYTE_VIEW_STRIDE, 0, IDENTITY_BUFFER_SWIZZLE};
		case Sl00ShapeKind::Structured:
			// The stride is the element size. Unverified, as no used program has one.
			if (shape.stride != 0 && !buffer.formatted) {
				return BufferFields {static_cast<uint32_t>(shape.stride), 0,
				                     IDENTITY_BUFFER_SWIZZLE};
			}
			return std::nullopt;
		case Sl00ShapeKind::Typed: {
			if (!shape.element_kind || !shape.element_count ||
			    FindTyped(*shape.element_kind, *shape.element_count) == nullptr) {
				return std::nullopt;
			}
			auto count = *shape.element_count;
			// 96-bit formats have no typed UAV store, so a written three-component buffer is bound
			// with the four-component format.
			if (buffer.written && count == THREE_COMPONENTS &&
			    FindTyped(*shape.element_kind, FOUR_COMPONENTS) != nullptr) {
				count = FOUR_COMPONENTS;
			}
			// Without a formatted access the game binds a byte view. Seen once on run A, a guess.
			return buffer.formatted ? *FindTyped(*shape.element_kind, count)
			                        : BufferFields {BYTE_VIEW_STRIDE, 0, IDENTITY_BUFFER_SWIZZLE};
		}
		default: return std::nullopt;
	}
}

IR::ResourceSpecialization::Buffer BoundBufferSpecialization(const BufferFields& fields) {
	IR::ResourceSpecialization::Buffer buffer;
	buffer.packed_stride      = fields.stride;
	buffer.descriptor_format  = static_cast<Prospero::BufferFormat>(fields.format);
	buffer.descriptor_swizzle = fields.swizzle;
	return buffer;
}

IR::ResourceSpecialization::Buffer NullBufferSpecialization(const BufferFacts& buffer) {
	IR::ResourceSpecialization::Buffer result;
	result.packed_stride      = 0;
	result.descriptor_format  = Prospero::BufferFormat::kInvalid;
	result.descriptor_swizzle = buffer.formatted ? 0u : IDENTITY_BUFFER_SWIZZLE;
	result.zero_stride_oob    = true;
	result.scalar_empty       = buffer.scalar;
	return result;
}

std::optional<IR::ResourceSpecialization::Image>
BoundImageSpecialization(const ImageFacts& image, const Sl00Resource* record) {
	if (record == nullptr || record->shape.kind != Sl00ShapeKind::Image ||
	    !record->shape.element_kind) {
		return std::nullopt;
	}
	const auto numeric = NumericClassOfKind(*record->shape.element_kind);
	if (!numeric) return std::nullopt;
	IR::ResourceSpecialization::Image result;
	result.numeric_class = *numeric;
	result.dimension     = static_cast<ShaderRecompiler::Decoder::ImageDimension>(image.dimension);
	result.cube          = image.cube;
	return result;
}

bool IsNullableBuffer(const std::optional<SlotRef>& slot, const std::string& name) {
	if (!slot) return false;
	return (slot->sharp_class == Sl00Class::ReadOnly &&
	        slot->index >= GLOBAL_TABLE_FIRST_REGISTER) ||
	       name == NULLABLE_NAME;
}

namespace {

// Appends every combination of `count` of the indices, in lexicographic order.
void AppendCombinations(std::span<const uint32_t> indices, size_t count,
                        std::vector<std::vector<uint32_t>>& out) {
	std::vector<size_t> chosen(count);
	std::iota(chosen.begin(), chosen.end(), size_t {0});
	while (true) {
		auto& pattern = out.emplace_back();
		for (const auto at: chosen)
			pattern.push_back(indices[at]);
		// Advance to the next combination, which moves the rightmost index that has room.
		size_t position = count;
		while (position > 0 && chosen[position - 1] == indices.size() - count + position - 1)
			position--;
		if (position == 0) return;
		chosen[position - 1]++;
		for (size_t next = position; next < count; next++)
			chosen[next] = chosen[next - 1] + 1;
	}
}

} // namespace

std::vector<std::vector<uint32_t>> NullPatterns(std::span<const uint32_t> nullable,
                                                NullPolicy                policy) {
	// A program with too many nullable buffers would make 2^n candidates, so only all-bound and
	// all-null are tried.
	if (policy == NullPolicy::Subsets && nullable.size() > MAX_SUBSET_NULLABLE)
		policy = NullPolicy::Coupled;
	if (policy == NullPolicy::None || nullable.empty())
		return std::vector<std::vector<uint32_t>>(1);
	if (policy == NullPolicy::Coupled) {
		return {std::vector<uint32_t> {}, std::vector<uint32_t>(nullable.begin(), nullable.end())};
	}
	std::vector<std::vector<uint32_t>> patterns;
	for (size_t count = 0; count <= nullable.size(); count++)
		AppendCombinations(nullable, count, patterns);
	return patterns;
}

namespace {

// The buffers' fields, and the indices of those the game may leave unbound.
struct ClassifiedBuffers {
	std::vector<BufferFields> fields;
	std::vector<uint32_t>     nullable;
};

// Finds each buffer's slot, record and fields. A buffer the rules do not cover gets a problem line.
ClassifiedBuffers ClassifyBuffers(const PredictInputs& inputs, std::vector<std::string>& problems) {
	ClassifiedBuffers result;
	for (uint32_t index = 0; index < inputs.facts.buffers.size(); index++) {
		const auto& buffer = inputs.facts.buffers[index];
		const auto  slot   = SlotOfKey(buffer.key, inputs.slots);
		const auto* record = slot ? ResourceAtSlot(*slot, inputs.records) : nullptr;
		const auto  bound  = BoundBufferFields(buffer, record);
		if (!bound) {
			problems.push_back("buffer " + std::to_string(index) + " " +
			                   DescriptorKeyText(buffer.key));
		}
		result.fields.push_back(bound.value_or(BufferFields {}));
		if (IsNullableBuffer(slot, record != nullptr ? record->name : std::string {}))
			result.nullable.push_back(index);
	}
	return result;
}

// Finds each image's specialization. An image the rules do not cover gets a problem line.
std::vector<IR::ResourceSpecialization::Image> ClassifyImages(const PredictInputs&      inputs,
                                                              std::vector<std::string>& problems) {
	std::vector<IR::ResourceSpecialization::Image> images;
	for (uint32_t index = 0; index < inputs.facts.images.size(); index++) {
		const auto& image  = inputs.facts.images[index];
		const auto  slot   = SlotOfKey(image.key, inputs.slots);
		const auto* record = slot ? ResourceAtSlot(*slot, inputs.records) : nullptr;
		const auto  bound  = BoundImageSpecialization(image, record);
		if (!bound) {
			problems.push_back("image " + std::to_string(index) + " " +
			                   DescriptorKeyText(image.key));
		}
		images.push_back(bound.value_or(IR::ResourceSpecialization::Image {}));
	}
	return images;
}

} // namespace

Prediction PredictCandidates(const PredictInputs& inputs) {
	Prediction  prediction;
	const auto& facts   = inputs.facts;
	const auto  buffers = ClassifyBuffers(inputs, prediction.problems);
	const auto  images  = ClassifyImages(inputs, prediction.problems);
	if (!prediction.problems.empty() || inputs.static_state.empty()) return prediction;

	const uint32_t push_start = facts.buffers.empty() ? IR::PushData::NoStart : 0u;
	for (auto& nulls: NullPatterns(buffers.nullable, inputs.policy)) {
		Candidate candidate;
		candidate.push_start = push_start;
		for (uint32_t index = 0; index < facts.buffers.size(); index++) {
			const bool null = std::ranges::find(nulls, index) != nulls.end();
			candidate.specialization.buffers.push_back(
			    null ? NullBufferSpecialization(facts.buffers[index])
			         : BoundBufferSpecialization(buffers.fields[index]));
		}
		candidate.specialization.images = images;
		candidate.specialization.samplers.assign(facts.samplers.size(),
		                                         IR::ResourceSpecialization::Sampler {});
		candidate.nulls = std::move(nulls);
		candidate.id =
		    ShaderModuleId(ShaderType::Compute, inputs.code_hash, inputs.user_data_count,
		                   inputs.static_state, push_start, candidate.specialization, {});
		prediction.candidates.push_back(std::move(candidate));
	}
	return prediction;
}

} // namespace Libs::Graphics::StoreGen
