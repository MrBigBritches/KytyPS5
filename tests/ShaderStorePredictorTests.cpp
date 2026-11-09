// Tests the compute module predictor, the header slot decoder and the shared module id.
// Everything here uses synthetic values, so no game file and no device are needed. Each rule is
// checked against a baseline that differs only in the behaviour under test.
#include "graphics/shader/moduleId.h"
#include "graphics/shader/storegen/computePredictor.h"
#include "graphics/shader/storegen/headerSlots.h"
#include "graphics/shader/storegen/shaderHeaderLayout.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;
namespace IR     = Libs::Graphics::ShaderRecompiler::IR;
namespace Layout = Libs::Graphics::StoreGen::HeaderLayout;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Bytes = std::vector<uint8_t>;

// The address of a temporary that lives to the end of the full expression.
template <typename T>
const T* Ptr(const T& value) {
	return &value;
}

// ---- Header blob layout used by the slot tests. ----
constexpr uint64_t HEADER_BYTES     = 0x200;
constexpr uint64_t USER_AT          = 0x100; // Where the user data table sits in the header.
constexpr uint64_t DIRECT_AT        = 0x140;
constexpr uint64_t SHARP_AT         = 0x160; // Each class table is 0x20 bytes from here.
constexpr uint64_t SHARP_STRIDE     = 0x20;
constexpr uint16_t EUD_POINTER_SGPR = 12;
// One table entry is 16 bits.
constexpr uint64_t BYTES_PER_ENTRY = sizeof(uint16_t);

// Writes a value at a byte offset of a blob.
template <typename T>
void Put(Bytes& bytes, uint64_t at, T value) {
	std::memcpy(bytes.data() + at, &value, sizeof(T));
}

// One class table, T U S B, as the raw 16-bit entries.
using SharpTables = std::array<std::vector<uint16_t>, 4>;

// Builds a header whose user data table holds the direct entries and the sharp tables.
Bytes MakeHeader(const std::vector<uint16_t>& direct, const SharpTables& tables) {
	Bytes bytes(HEADER_BYTES, 0);
	Put<uint64_t>(bytes, Layout::OFFSET_USER_DATA, USER_AT - Layout::OFFSET_USER_DATA);
	Put<uint64_t>(bytes, USER_AT + Layout::USER_DIRECT_POINTER, DIRECT_AT - USER_AT);
	Put<uint16_t>(bytes, USER_AT + Layout::USER_DIRECT_COUNT, static_cast<uint16_t>(direct.size()));
	for (size_t index = 0; index < direct.size(); index++)
		Put<uint16_t>(bytes, DIRECT_AT + BYTES_PER_ENTRY * index, direct[index]);
	for (uint64_t table = 0; table < 4; table++) {
		const uint64_t field =
		    USER_AT + Layout::USER_SHARP_POINTERS + Layout::USER_POINTER_SIZE * table;
		const uint64_t at = SHARP_AT + SHARP_STRIDE * table;
		Put<uint64_t>(bytes, field, at - field);
		Put<uint16_t>(bytes, USER_AT + Layout::USER_SHARP_COUNTS + BYTES_PER_ENTRY * table,
		              static_cast<uint16_t>(tables[table].size()));
		for (size_t slot = 0; slot < tables[table].size(); slot++)
			Put<uint16_t>(bytes, at + BYTES_PER_ENTRY * slot, tables[table][slot]);
	}
	return bytes;
}

constexpr uint16_t UNUSED = 0xFFFF;
// Direct entries 0 to 4 are other pointers, and entry 5 is the EUD pointer.
std::vector<uint16_t> DirectWithEud(uint16_t eud) {
	return {0, 2, 4, 6, 8, eud};
}

// The slot that holds a user data offset, as the decoder names it.
std::optional<SlotRef> SlotOf(const HeaderSlots& slots, uint32_t offset) {
	return SlotAtOffset(slots, offset);
}

constexpr SlotRef T0 {Sl00Class::ReadOnly, 0};
constexpr SlotRef T2 {Sl00Class::ReadOnly, 2};
constexpr SlotRef U1 {Sl00Class::ReadWrite, 1};
constexpr SlotRef B0 {Sl00Class::ConstantBuffer, 0};

void TestHeaderSlotTables() {
	// T has two slots, with the second unused. U has one slot holding the masked offset 8. B has
	// one at 40.
	SharpTables tables;
	tables[0]        = {0, UNUSED, 36};
	tables[1]        = {0x8004, 8};
	tables[3]        = {40};
	const auto slots = DecodeHeaderSlots(MakeHeader(DirectWithEud(EUD_POINTER_SGPR), tables));
	CHECK(slots.has_value());
	CHECK(slots->entries.size() == 5);
	CHECK(slots->eud_pointer == EUD_POINTER_SGPR);
	CHECK(SlotOf(*slots, 0) == T0);
	CHECK(SlotOf(*slots, 36) == T2); // An unused slot still counts as a table position.
	CHECK(SlotOf(*slots, 8) == U1);
	CHECK(SlotOf(*slots, 40) == B0);
	// The top bit of an entry is not part of the offset, so 0x8004 names offset 4. A mask that kept
	// it would put the slot at 32772.
	CHECK(SlotOf(*slots, 4) == (SlotRef {Sl00Class::ReadWrite, 0}));
	CHECK(!SlotOf(*slots, 0x8004).has_value());
	CHECK(!SlotOf(*slots, 12).has_value());
}

void TestHeaderEudPointerIsDirectEntryFive() {
	SharpTables tables;
	// The baseline has the pointer in entry 5.
	CHECK(DecodeHeaderSlots(MakeHeader(DirectWithEud(EUD_POINTER_SGPR), tables))->eud_pointer ==
	      EUD_POINTER_SGPR);
	// The same value in entry 4 is not the pointer.
	CHECK(!DecodeHeaderSlots(MakeHeader({0, 2, 4, 6, EUD_POINTER_SGPR}, tables))
	           ->eud_pointer.has_value());
	// An unused entry 5 is no pointer.
	CHECK(!DecodeHeaderSlots(MakeHeader(DirectWithEud(UNUSED), tables))->eud_pointer.has_value());
}

void TestHeaderSlotUniqueness() {
	SharpTables tables;
	tables[0]           = {16};
	tables[1]           = {20};
	const auto baseline = DecodeHeaderSlots(MakeHeader(DirectWithEud(EUD_POINTER_SGPR), tables));
	CHECK(SlotOf(*baseline, 16) == T0);
	// Two slots at one offset name neither.
	tables[1]         = {16};
	const auto shared = DecodeHeaderSlots(MakeHeader(DirectWithEud(EUD_POINTER_SGPR), tables));
	CHECK(!SlotOf(*shared, 16).has_value());
}

void TestHeaderFailures() {
	SharpTables tables;
	tables[0]   = {0};
	auto header = MakeHeader(DirectWithEud(EUD_POINTER_SGPR), tables);
	CHECK(DecodeHeaderSlots(header).has_value());
	// A header cut short of the table is refused, and one too short for the pointer too.
	CHECK(!DecodeHeaderSlots(std::span<const uint8_t>(header).first(SHARP_AT + 1)).has_value());
	CHECK(!DecodeHeaderSlots(std::span<const uint8_t>(header).first(4)).has_value());
	// A wild table pointer is refused.
	auto wild = header;
	Put<uint64_t>(wild, USER_AT + Layout::USER_SHARP_POINTERS, 0xFFFFFFFFFFFF0000ull);
	CHECK(!DecodeHeaderSlots(wild).has_value());
	// No user data table gives no slots and no pointer.
	auto none = header;
	Put<uint64_t>(none, Layout::OFFSET_USER_DATA, 0);
	const auto empty = DecodeHeaderSlots(none);
	CHECK(empty.has_value() && empty->entries.empty() && !empty->eud_pointer.has_value());
}

// ---- Keys and slots ----
// A key for descriptors in consecutive user SGPRs from `sgpr`.
DescriptorKey UserKey(uint32_t sgpr) {
	return {DescriptorKeyKind::User, sgpr, 0, 0, 0};
}
// A key for a descriptor in the EUD table that `pointer` holds, at a byte offset.
DescriptorKey EudKey(uint32_t pointer, uint32_t byte_offset) {
	return {DescriptorKeyKind::Eud, pointer, byte_offset, 0, 0};
}
// A literal pool key with the values Yumia's first literal buffer has.
DescriptorKey LiteralKey() {
	return {DescriptorKeyKind::Literal, 5900, 12452, 368, 268455940};
}

void TestSlotOffsetOfKey() {
	HeaderSlots slots;
	slots.eud_pointer = EUD_POINTER_SGPR;
	CHECK(SlotOffsetOfKey(UserKey(8), slots) == 8u);
	// An EUD dword is the offset after the 32 user SGPRs. Byte 96 is dword 24, so offset 56.
	CHECK(SlotOffsetOfKey(EudKey(EUD_POINTER_SGPR, 96), slots) == 56u);
	// One change at a time from the baseline above.
	CHECK(!SlotOffsetOfKey(EudKey(EUD_POINTER_SGPR + 1, 96), slots).has_value()); // another pointer
	CHECK(!SlotOffsetOfKey(EudKey(EUD_POINTER_SGPR, 98), slots).has_value());     // unaligned
	HeaderSlots no_pointer;
	CHECK(!SlotOffsetOfKey(EudKey(EUD_POINTER_SGPR, 96), no_pointer)
	           .has_value()); // no pointer in the header
	CHECK(!SlotOffsetOfKey(LiteralKey(), slots).has_value());
	CHECK(DescriptorKeyText(EudKey(12, 96)) == "('eud', 12, 96)");
	CHECK(DescriptorKeyText(UserKey(4)) == "('user', 4)");
	CHECK(DescriptorKeyText(LiteralKey()) == "('literal', 5900, 12452, 368, 268455940)");
}

// ---- Records ----
// An sl00 record at a register and class with the given shape.
Sl00Resource Record(const char* name, uint8_t reg, Sl00Class sharp_class, Sl00Shape shape) {
	Sl00Resource record;
	record.name           = name;
	record.register_index = reg;
	record.resource_class = sharp_class;
	record.shape          = shape;
	return record;
}
// A shape of a kind with no other fields.
Sl00Shape ShapeOf(Sl00ShapeKind kind) {
	Sl00Shape shape;
	shape.kind = kind;
	return shape;
}
// A formatted buffer's shape with an element kind and component count.
Sl00Shape TypedShape(uint64_t kind, uint64_t count) {
	auto shape          = ShapeOf(Sl00ShapeKind::Typed);
	shape.element_kind  = kind;
	shape.element_count = count;
	return shape;
}
// A structured buffer's shape with an element stride.
Sl00Shape StructuredShape(uint64_t stride) {
	auto shape   = ShapeOf(Sl00ShapeKind::Structured);
	shape.stride = stride;
	return shape;
}
// An image's shape with an element kind.
Sl00Shape ImageShape(uint64_t kind) {
	auto shape         = ShapeOf(Sl00ShapeKind::Image);
	shape.element_kind = kind;
	return shape;
}

void TestResourceAtSlot() {
	const std::vector<Sl00Resource> baseline = {
	    Record("a", 0, Sl00Class::ReadOnly, ShapeOf(Sl00ShapeKind::Raw))};
	CHECK(ResourceAtSlot(T0, baseline) != nullptr && ResourceAtSlot(T0, baseline)->name == "a");
	CHECK(ResourceAtSlot(T2, baseline) == nullptr);                        // another register
	CHECK(ResourceAtSlot({Sl00Class::ReadWrite, 0}, baseline) == nullptr); // another class
	// A system value shares the record layout and the register, and must not make the slot
	// ambiguous.
	auto with_other = baseline;
	with_other.push_back(Record("threadID", 0, Sl00Class::ReadOnly, ShapeOf(Sl00ShapeKind::Other)));
	with_other.push_back(Record("bad", 0, Sl00Class::ReadOnly, ShapeOf(Sl00ShapeKind::Unreadable)));
	CHECK(ResourceAtSlot(T0, with_other) != nullptr && ResourceAtSlot(T0, with_other)->name == "a");
	// A second readable record at the slot makes it ambiguous.
	auto two = baseline;
	two.push_back(Record("b", 0, Sl00Class::ReadOnly, ShapeOf(Sl00ShapeKind::Raw)));
	CHECK(ResourceAtSlot(T0, two) == nullptr);
}

// The translation facts of a buffer.
BufferFacts Facts(DescriptorKey key, bool formatted = false, bool scalar = false,
                  bool written = false) {
	return {key, formatted, scalar, written};
}
using Fields                  = BufferFields;
constexpr uint32_t ID_SWIZZLE = IDENTITY_BUFFER_SWIZZLE;

void TestFieldRules() {
	const auto cb =
	    Record("c", 0, Sl00Class::ConstantBuffer, ShapeOf(Sl00ShapeKind::ConstantBuffer));
	const auto raw = Record("r", 0, Sl00Class::ReadWrite, ShapeOf(Sl00ShapeKind::Raw));
	CHECK(BoundBufferFields(Facts(UserKey(0)), &cb) == (Fields {16, 0, ID_SWIZZLE}));
	CHECK(BoundBufferFields(Facts(UserKey(0)), &raw) == (Fields {1, 0, ID_SWIZZLE}));
	// A structured buffer takes its stride from the record. The baseline is the same record read
	// through a formatted access, which no rule covers, and a zero stride, which none covers
	// either.
	const auto structured = Record("s", 0, Sl00Class::ReadOnly, StructuredShape(164));
	CHECK(BoundBufferFields(Facts(UserKey(0)), &structured) == (Fields {164, 0, ID_SWIZZLE}));
	CHECK(!BoundBufferFields(Facts(UserKey(0), true), &structured).has_value());
	const auto zero_stride = Record("s", 0, Sl00Class::ReadOnly, StructuredShape(0));
	CHECK(!BoundBufferFields(Facts(UserKey(0)), &zero_stride).has_value());
	// No record, and a record of an unknown shape, are outside the rules.
	CHECK(!BoundBufferFields(Facts(UserKey(0)), nullptr).has_value());
	const auto image = Record("i", 0, Sl00Class::ReadOnly, ImageShape(1));
	CHECK(!BoundBufferFields(Facts(UserKey(0)), &image).has_value());
}

void TestTypedRules() {
	const auto float4 = Record("t", 0, Sl00Class::ReadOnly, TypedShape(1, 4));
	// A formatted access gets the table row, and the baseline without it gets a byte view.
	CHECK(BoundBufferFields(Facts(UserKey(0), true), &float4) == (Fields {16, 77, 4012}));
	CHECK(BoundBufferFields(Facts(UserKey(0), false), &float4) == (Fields {1, 0, ID_SWIZZLE}));
	// Each table row, by kind and count.
	CHECK(BoundBufferFields(Facts(UserKey(0), true),
	                        Ptr(Record("t", 0, Sl00Class::ReadOnly, TypedShape(1, 1)))) ==
	      (Fields {4, 22, 516}));
	const auto int2 = Record("t", 0, Sl00Class::ReadOnly, TypedShape(5, 2));
	CHECK(BoundBufferFields(Facts(UserKey(0), true), &int2) == (Fields {8, 63, 556}));
	const auto uint1 = Record("t", 0, Sl00Class::ReadOnly, TypedShape(9, 1));
	CHECK(BoundBufferFields(Facts(UserKey(0), true), &uint1) == (Fields {4, 20, 516}));
	// A written three-component buffer is bound as four components. The baseline is the same
	// record without the write, which keeps three.
	const auto uint3 = Record("t", 0, Sl00Class::ReadOnly, TypedShape(9, 3));
	CHECK(BoundBufferFields(Facts(UserKey(0), true, false, false), &uint3) ==
	      (Fields {12, 72, 940}));
	CHECK(BoundBufferFields(Facts(UserKey(0), true, false, true), &uint3) ==
	      (Fields {16, 75, 4012}));
	// Kinds and counts outside the table are not covered, even unformatted.
	const auto odd_kind = Record("t", 0, Sl00Class::ReadOnly, TypedShape(0, 4));
	CHECK(!BoundBufferFields(Facts(UserKey(0), true), &odd_kind).has_value());
	CHECK(!BoundBufferFields(Facts(UserKey(0), false), &odd_kind).has_value());
	const auto five = Record("t", 0, Sl00Class::ReadOnly, TypedShape(1, 5));
	CHECK(!BoundBufferFields(Facts(UserKey(0), true), &five).has_value());
	auto no_count = TypedShape(1, 4);
	no_count.element_count.reset();
	CHECK(!BoundBufferFields(Facts(UserKey(0), true),
	                         Ptr(Record("t", 0, Sl00Class::ReadOnly, no_count)))
	           .has_value());
}

void TestLiteralRule() {
	// A literal table has no record and no stride. Formatted access is not covered.
	CHECK(BoundBufferFields(Facts(LiteralKey()), nullptr) == (Fields {0, 0, ID_SWIZZLE}));
	CHECK(!BoundBufferFields(Facts(LiteralKey(), true), nullptr).has_value());
	// The baseline is a user key with no record, which has no rule.
	CHECK(!BoundBufferFields(Facts(UserKey(0)), nullptr).has_value());
}

void TestBufferEncodings() {
	const auto bound = BoundBufferSpecialization({16, 77, 4012});
	CHECK(bound.packed_stride == 16);
	CHECK(bound.descriptor_format == static_cast<Prospero::BufferFormat>(77));
	CHECK(bound.descriptor_swizzle == 4012);
	CHECK(!bound.zero_stride_oob && !bound.scalar_empty);
	CHECK(bound.indirect_root == IR::BufferResource::NoIndirectBuffer);
	// A null buffer reads as a zeroed descriptor, with the empty-scalar flag taken from the facts.
	const auto null_plain = NullBufferSpecialization(Facts(UserKey(0)));
	CHECK(null_plain.packed_stride == 0 &&
	      null_plain.descriptor_format == Prospero::BufferFormat::kInvalid);
	CHECK(null_plain.zero_stride_oob);
	CHECK(!null_plain.scalar_empty);
	CHECK(null_plain.descriptor_swizzle == ID_SWIZZLE);
	CHECK(null_plain.indirect_root == IR::BufferResource::NoIndirectBuffer);
	CHECK(NullBufferSpecialization(Facts(UserKey(0), false, true)).scalar_empty);
	// A formatted access has a zero swizzle when null. Only that changes against the baseline.
	const auto null_formatted = NullBufferSpecialization(Facts(UserKey(0), true));
	CHECK(null_formatted.descriptor_swizzle == 0);
	CHECK(null_formatted.zero_stride_oob && !null_formatted.scalar_empty);
}

void TestImageEncoding() {
	ImageFacts facts {UserKey(0), 3, false};
	const auto baseline =
	    BoundImageSpecialization(facts, Ptr(Record("i", 0, Sl00Class::ReadOnly, ImageShape(1))));
	CHECK(baseline.has_value());
	CHECK(baseline->numeric_class == Prospero::TextureNumericClass::Float);
	CHECK(baseline->dimension == static_cast<ShaderRecompiler::Decoder::ImageDimension>(3));
	CHECK(baseline->mip_count == 1);
	CHECK(baseline->shader_swizzle == IR::ShaderImageIdentitySwizzle);
	CHECK(baseline->conversion_format == Prospero::BufferFormat::kInvalid);
	CHECK(baseline->indirect_root == IR::ImageResource::NoIndirectImage);
	CHECK(!baseline->cube && !baseline->skip_degamma && !baseline->fmask && !baseline->heap);
	const auto numeric = [&](uint64_t kind) {
		return BoundImageSpecialization(facts,
		                                Ptr(Record("i", 0, Sl00Class::ReadOnly, ImageShape(kind))));
	};
	CHECK(numeric(0)->numeric_class == Prospero::TextureNumericClass::Float);
	CHECK(numeric(9)->numeric_class == Prospero::TextureNumericClass::Uint);
	CHECK(numeric(5)->numeric_class == Prospero::TextureNumericClass::Sint);
	CHECK(!numeric(2).has_value());
	// Only the cube flag differs here.
	facts.cube = true;
	const auto cube =
	    BoundImageSpecialization(facts, Ptr(Record("i", 0, Sl00Class::ReadOnly, ImageShape(1))));
	CHECK(cube->cube);
	auto cube_off = *cube;
	cube_off.cube = false;
	CHECK(cube_off == *baseline);
	// Not an image, or no record, is outside the rules.
	CHECK(!BoundImageSpecialization(
	           facts, Ptr(Record("i", 0, Sl00Class::ReadOnly, ShapeOf(Sl00ShapeKind::Raw))))
	           .has_value());
	CHECK(!BoundImageSpecialization(facts, nullptr).has_value());
}

void TestNullableRule() {
	CHECK(IsNullableBuffer(T0, "") == false);
	CHECK(IsNullableBuffer(SlotRef {Sl00Class::ReadOnly, GLOBAL_TABLE_FIRST_REGISTER}, ""));
	CHECK(IsNullableBuffer(SlotRef {Sl00Class::ReadOnly, GLOBAL_TABLE_FIRST_REGISTER + 3}, ""));
	// One below the threshold, and the same register in another class, are not nullable.
	CHECK(!IsNullableBuffer(SlotRef {Sl00Class::ReadOnly, GLOBAL_TABLE_FIRST_REGISTER - 1}, ""));
	CHECK(!IsNullableBuffer(SlotRef {Sl00Class::ReadWrite, GLOBAL_TABLE_FIRST_REGISTER}, ""));
	CHECK(!IsNullableBuffer(SlotRef {Sl00Class::ConstantBuffer, GLOBAL_TABLE_FIRST_REGISTER}, ""));
	// The camera constants are nullable wherever they sit, but not without a slot.
	CHECK(IsNullableBuffer(B0, "CameraConstBuf"));
	CHECK(!IsNullableBuffer(B0, "OtherBuf"));
	CHECK(!IsNullableBuffer(std::nullopt, "CameraConstBuf"));
	CHECK(GLOBAL_TABLE_FIRST_REGISTER == 29);
}

using Patterns = std::vector<std::vector<uint32_t>>;

void TestNullPatternCap() {
	std::vector<uint32_t> nullable(MAX_SUBSET_NULLABLE);
	std::iota(nullable.begin(), nullable.end(), 0u);
	// At the cap every subset is tried.
	CHECK(NullPatterns(nullable, NullPolicy::Subsets).size() ==
	      (size_t {1} << MAX_SUBSET_NULLABLE));
	// One more buffer would double it, so only all-bound and all-null are tried.
	nullable.push_back(MAX_SUBSET_NULLABLE);
	const auto capped = NullPatterns(nullable, NullPolicy::Subsets);
	CHECK(capped.size() == 2);
	CHECK(capped.size() == 2 && capped[0].empty() && capped[1] == nullable);
	CHECK(MAX_SUBSET_NULLABLE == 6);
}

void TestNullPatterns() {
	const std::vector<uint32_t> three    = {2, 5, 7};
	const Patterns              expected = {{}, {2}, {5}, {7}, {2, 5}, {2, 7}, {5, 7}, {2, 5, 7}};
	CHECK(NullPatterns(three, NullPolicy::Subsets) == expected);
	const std::vector<uint32_t> four          = {0, 1, 2, 3};
	const auto                  four_patterns = NullPatterns(four, NullPolicy::Subsets);
	CHECK(four_patterns.size() == 16);
	CHECK((four_patterns[5] == std::vector<uint32_t> {0, 1}));
	CHECK((four_patterns[10] == std::vector<uint32_t> {2, 3}));
	CHECK((four_patterns[11] == std::vector<uint32_t> {0, 1, 2}));
	// The other policies, and the baseline of no nullable buffers.
	CHECK(NullPatterns(three, NullPolicy::None) == Patterns {{}});
	CHECK(NullPatterns(three, NullPolicy::Coupled) == (Patterns {{}, {2, 5, 7}}));
	CHECK(NullPatterns({}, NullPolicy::Subsets) == Patterns {{}});
}

// ---- Whole-program prediction ----
constexpr uint64_t PROGRAM_HASH = 0x1234abcd5678ef01ull;
constexpr uint32_t USER_DATA    = 16;

struct Program {
	std::vector<uint32_t>     state {16, 49216, 32, 1};
	ResourceFacts             facts;
	HeaderSlots               slots;
	std::vector<Sl00Resource> records;

	// Adds a buffer in a user SGPR slot with a matching raw record.
	void AddBuffer(Sl00Class sharp_class, uint32_t index, const char* name, bool scalar = false) {
		const uint32_t offset = static_cast<uint32_t>(slots.entries.size()) * 4;
		slots.entries.push_back({{sharp_class, index}, offset});
		records.push_back(
		    Record(name, static_cast<uint8_t>(index), sharp_class, ShapeOf(Sl00ShapeKind::Raw)));
		facts.buffers.push_back(Facts(UserKey(offset), false, scalar));
	}
	Prediction Predict(NullPolicy policy = NullPolicy::Subsets) const {
		return PredictCandidates({PROGRAM_HASH, USER_DATA, state, facts, slots, records, policy});
	}
};

void TestThreeNullableGiveEightCandidates() {
	Program program;
	program.AddBuffer(Sl00Class::ReadOnly, 29, "g0");
	program.AddBuffer(Sl00Class::ReadOnly, 30, "g1");
	program.AddBuffer(Sl00Class::ReadOnly, 31, "g2");
	const auto prediction = program.Predict();
	CHECK(prediction.problems.empty());
	CHECK(prediction.candidates.size() == 8);
	if (prediction.candidates.size() != 8) return;
	const Patterns     expected = {{}, {0}, {1}, {2}, {0, 1}, {0, 2}, {1, 2}, {0, 1, 2}};
	std::set<uint64_t> ids;
	for (size_t index = 0; index < 8; index++) {
		const auto& candidate = prediction.candidates[index];
		CHECK(candidate.nulls == expected[index]);
		ids.insert(candidate.id);
		// Each candidate nulls exactly its buffers. The unbound ones are marked, and the rest have
		// the raw view.
		for (uint32_t buffer = 0; buffer < 3; buffer++) {
			const bool null = std::ranges::find(candidate.nulls, buffer) != candidate.nulls.end();
			CHECK(candidate.specialization.buffers[buffer].zero_stride_oob == null);
			CHECK(candidate.specialization.buffers[buffer].packed_stride == (null ? 0u : 1u));
		}
	}
	CHECK(ids.size() == 8);
	// The base candidate leaves everything bound.
	CHECK(prediction.candidates[0].nulls.empty());
	// The baseline is the same program with no nullable slots.
	Program plain;
	plain.AddBuffer(Sl00Class::ReadOnly, 1, "a");
	plain.AddBuffer(Sl00Class::ReadOnly, 2, "b");
	plain.AddBuffer(Sl00Class::ReadOnly, 3, "c");
	CHECK(plain.Predict().candidates.size() == 1);
}

void TestCandidateIdAndPushStart() {
	Program program;
	program.AddBuffer(Sl00Class::ReadOnly, 1, "a");
	const auto prediction = program.Predict();
	CHECK(prediction.candidates.size() == 1);
	const auto& candidate = prediction.candidates[0];
	CHECK(candidate.push_start == 0);
	CHECK(candidate.id == ShaderModuleId(ShaderType::Compute, PROGRAM_HASH, USER_DATA,
	                                     program.state, 0, candidate.specialization, {}));
	// A different push start gives a different id, so the id check above can tell a wrong start.
	CHECK(candidate.id != ShaderModuleId(ShaderType::Compute, PROGRAM_HASH, USER_DATA,
	                                     program.state, 1, candidate.specialization, {}));
	// A program with no buffers has no push data.
	Program images_only;
	images_only.facts.images.push_back({UserKey(0), 3, false});
	images_only.slots.entries.push_back({T0, 0});
	images_only.records.push_back(Record("img", 0, Sl00Class::ReadOnly, ImageShape(1)));
	const auto images = images_only.Predict();
	CHECK(images.candidates.size() == 1);
	CHECK(images.candidates[0].push_start == IR::PushData::NoStart);
	CHECK(images.candidates[0].specialization.images.size() == 1);
	CHECK(images.candidates[0].id == ShaderModuleId(ShaderType::Compute, PROGRAM_HASH, USER_DATA,
	                                                images_only.state, IR::PushData::NoStart,
	                                                images.candidates[0].specialization, {}));
}

void TestScalarComesFromTranslationFacts() {
	// The record is a plain raw buffer, so nothing in the sl00 data says scalar. The flag is the
	// translation's, and it decides the null candidate's scalar_empty.
	Program scalar;
	scalar.AddBuffer(Sl00Class::ReadOnly, 29, "g", true);
	Program not_scalar;
	not_scalar.AddBuffer(Sl00Class::ReadOnly, 29, "g", false);
	const auto with_scalar    = scalar.Predict();
	const auto without_scalar = not_scalar.Predict();
	CHECK(with_scalar.candidates.size() == 2 && without_scalar.candidates.size() == 2);
	CHECK(with_scalar.candidates[1].specialization.buffers[0].scalar_empty);
	CHECK(!without_scalar.candidates[1].specialization.buffers[0].scalar_empty);
	CHECK(with_scalar.candidates[1].id != without_scalar.candidates[1].id);
	// A constant buffer record does not make a buffer scalar either.
	Program cb;
	cb.AddBuffer(Sl00Class::ReadOnly, 29, "cb", false);
	cb.records[0].shape = ShapeOf(Sl00ShapeKind::ConstantBuffer);
	CHECK(!cb.Predict().candidates[1].specialization.buffers[0].scalar_empty);
}

void TestFormattedAndWrittenComeFromFacts() {
	Program program;
	program.AddBuffer(Sl00Class::ReadOnly, 1, "t");
	program.records[0].shape           = TypedShape(9, 3);
	program.facts.buffers[0].formatted = true;
	// Formatted and not written keeps three components. Written changes only that, to four.
	const auto read_only             = program.Predict();
	program.facts.buffers[0].written = true;
	const auto written               = program.Predict();
	CHECK(read_only.candidates.size() == 1 && written.candidates.size() == 1);
	CHECK(read_only.candidates[0].specialization.buffers[0].packed_stride == 12);
	CHECK(written.candidates[0].specialization.buffers[0].packed_stride == 16);
	// Unformatted gives the byte view, whatever the record says.
	program.facts.buffers[0].formatted = false;
	CHECK(program.Predict().candidates[0].specialization.buffers[0].packed_stride == 1);
}

void TestProblemsLeaveNoCandidates() {
	Program program;
	program.AddBuffer(Sl00Class::ReadOnly, 1, "a");
	CHECK(program.Predict().candidates.size() == 1);
	// The same program with a record that no rule covers has none, and says which buffer.
	program.records[0].shape = ShapeOf(Sl00ShapeKind::Other);
	const auto none          = program.Predict();
	CHECK(none.candidates.empty());
	CHECK(none.problems.size() == 1);
	// An image with no record is outside the rules too.
	Program image;
	image.facts.images.push_back({UserKey(0), 3, false});
	CHECK(image.Predict().candidates.empty());
	// An empty static state gives no candidate.
	Program no_state;
	no_state.AddBuffer(Sl00Class::ReadOnly, 1, "a");
	no_state.state.clear();
	CHECK(no_state.Predict().candidates.empty());
}

void TestSamplersAndLiterals() {
	Program program;
	program.AddBuffer(Sl00Class::ReadOnly, 1, "a");
	const auto none = program.Predict();
	program.facts.samplers.push_back(UserKey(20));
	program.facts.samplers.push_back(EudKey(12, 0));
	// A sampler without a key, such as a static one, still takes an entry, as python counts it.
	program.facts.samplers.push_back(std::nullopt);
	const auto samplers = program.Predict();
	CHECK(samplers.candidates[0].specialization.samplers.size() == 3);
	CHECK(none.candidates[0].specialization.samplers.empty());
	CHECK(samplers.candidates[0].id != none.candidates[0].id);
	// A literal buffer is never nullable and takes the literal rule.
	Program literal;
	literal.facts.buffers.push_back(Facts(LiteralKey()));
	const auto predicted = literal.Predict();
	CHECK(predicted.candidates.size() == 1);
	CHECK(predicted.candidates[0].specialization.buffers[0].packed_stride == 0);
	CHECK(predicted.candidates[0].specialization.buffers[0].descriptor_swizzle == ID_SWIZZLE);
}

// ---- The shared module id ----
// The inputs of PipelineCacheTestAccess::FixedModuleId, so the golden is the same value. Each input
// is non-default, and the specialization holds one buffer, image and sampler with every field set.
uint64_t FixedModuleId(uint32_t push_start, bool with_specialization = true) {
	using Spec = IR::ResourceSpecialization;
	const std::vector<uint32_t>   state {0x11u, 0x22u, 0x33u};
	const std::array<uint32_t, 3> function_code {0xaau, 0xbbu, 0xccu};
	Spec                          specialization;
	if (with_specialization) {
		specialization.buffers.push_back(
		    {.packed_stride              = 16,
		     .descriptor_format          = static_cast<Prospero::BufferFormat>(3),
		     .descriptor_swizzle         = 0x0123,
		     .zero_stride_oob            = true,
		     .scalar_empty               = true,
		     .indirect_root              = 5,
		     .indirect_mapping_offset    = 6,
		     .indirect_search_iterations = 7});
		specialization.images.push_back(
		    {.numeric_class              = static_cast<Prospero::TextureNumericClass>(2),
		     .dimension                  = static_cast<decltype(Spec::Image::dimension)>(1),
		     .mip_count                  = 4,
		     .conversion_format          = static_cast<Prospero::BufferFormat>(9),
		     .shader_swizzle             = 0x0456,
		     .srgb_decode                = static_cast<Prospero::BufferFormat>(10),
		     .indirect_root              = 8,
		     .indirect_mapping_offset    = 9,
		     .indirect_search_iterations = 10,
		     .cube                       = true,
		     .skip_degamma               = true,
		     .fmask                      = true,
		     .heap                       = true});
		specialization.samplers.push_back(
		    {.indirect_root = 11, .indirect_mapping_offset = 12, .indirect_search_iterations = 13});
	}
	return ShaderModuleId(ShaderType::Compute, 0x0123456789abcdefull, 12, state, push_start,
	                      specialization, function_code);
}

void TestModuleIdGolden() {
	// The same golden as ModuleIdGolden in the compute tests, which reaches the id through
	// PipelineCache::ModuleId. The two must stay equal, or the generator and the cache disagree.
	constexpr uint64_t GOLDEN = 0xe7e70feb629b182full;
	constexpr uint32_t PUSH   = 7;
	CHECK(FixedModuleId(PUSH) == GOLDEN);
	CHECK(FixedModuleId(PUSH + 1) != GOLDEN);
	CHECK(FixedModuleId(PUSH, false) != GOLDEN);
}

} // namespace

int main() {
	TestHeaderSlotTables();
	TestHeaderEudPointerIsDirectEntryFive();
	TestHeaderSlotUniqueness();
	TestHeaderFailures();
	TestSlotOffsetOfKey();
	TestResourceAtSlot();
	TestFieldRules();
	TestTypedRules();
	TestLiteralRule();
	TestBufferEncodings();
	TestImageEncoding();
	TestNullableRule();
	TestNullPatterns();
	TestNullPatternCap();
	TestThreeNullableGiveEightCandidates();
	TestCandidateIdAndPushStart();
	TestScalarComesFromTranslationFacts();
	TestFormattedAndWrittenComeFromFacts();
	TestProblemsLeaveNoCandidates();
	TestSamplersAndLiterals();
	TestModuleIdGolden();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("shader store predictor tests passed\n");
	return 0;
}
