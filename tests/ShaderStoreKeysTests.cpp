// Tests the route K classifier, which names the descriptors of a translated compute program as user
// SGPRs, extended user data dwords or literal pool tables. The plans are built by hand out of typed
// IR values, so no shader is translated and no device is needed.
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/storegen/resourceKeys.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <string>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;
using namespace Libs::Graphics::ShaderRecompiler::IR;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

constexpr uint32_t EUD_POINTER   = 12;
constexpr uint32_t TABLE_OFFSET  = 96;
constexpr uint32_t LITERAL_PC    = 5900;
constexpr uint32_t LITERAL_TABLE = 12452;
constexpr uint32_t LITERAL_WORD2 = 368;
constexpr uint32_t LITERAL_WORD3 = 268455940;

// Builds the parts of a ResourcePlan that the classifier reads.
struct PlanBuilder {
	ResourcePlan plan;

	Value Make(ValueOpcode opcode, std::initializer_list<Value> args = {}) {
		auto&  inst  = plan.value_storage.emplace_back(opcode);
		size_t index = 0;
		for (const auto& arg: args)
			inst.SetArg(index++, arg);
		return Value(&inst);
	}

	Value User(uint32_t reg) {
		return Make(ValueOpcode::GetUserData, {Value(static_cast<ScalarReg>(reg))});
	}

	// One dword read from the extended user data table, the way translation lists it: a flat SRT
	// read of a scalar load. The load's immediate offset is in the memory metadata and its own
	// dynamic argument is zero.
	Value TableDword(uint32_t pointer, uint32_t byte_offset, uint32_t dynamic = 0) {
		MemoryInfo memory;
		memory.kind   = ResourceKind::ScalarAddress;
		memory.offset = byte_offset;
		plan.memory_info.push_back(memory);
		const auto handle =
		    Make(ValueOpcode::GetAddressResource, {User(pointer), User(pointer + 1)});
		const auto read = Make(ValueOpcode::LoadAddressU32,
		                       {handle, Value(dynamic), Value(uint32_t {0}), Value(false)});
		read.Instruction()->SetFlags(
		    MemoryFlags {.index = static_cast<uint32_t>(plan.memory_info.size() - 1), .pc = 0});
		const auto slot = static_cast<uint32_t>(plan.srt_reads.size());
		plan.srt_reads.push_back({read, slot});
		return Make(ValueOpcode::ReadConst, {Make(ValueOpcode::GetSrtResource), Value(slot)});
	}

	// The low dword of a descriptor built from s_getpc, with the add's operands in either order.
	// `half` is the extract index, which is 0 for a real low dword.
	Value LiteralLow(uint32_t pc_after, uint32_t table_offset, bool swapped = false,
	                 uint32_t half = 0) {
		const auto base = Make(ValueOpcode::GetShaderBase);
		const auto sum  = Make(ValueOpcode::IAdd64, {base, Value(uint64_t {pc_after})});
		const auto low  = Make(ValueOpcode::CompositeExtractU64, {sum, Value(half)});
		return swapped ? Make(ValueOpcode::IAdd32, {low, Value(table_offset)})
		               : Make(ValueOpcode::IAdd32, {Value(table_offset), low});
	}

	// The high dword of a descriptor built from s_getpc, the way translation lists the add with
	// carry: the high half of the pc sum plus the carry of the low add.
	Value LiteralHigh(uint32_t pc_after, uint32_t table_offset) {
		const auto base  = Make(ValueOpcode::GetShaderBase);
		const auto sum   = Make(ValueOpcode::IAdd64, {base, Value(uint64_t {pc_after})});
		const auto high  = Make(ValueOpcode::CompositeExtractU64, {sum, Value(uint32_t {1})});
		const auto carry = Make(ValueOpcode::SelectU32,
		                        {Make(ValueOpcode::ULessThan32,
		                              {LiteralLow(pc_after, table_offset), Value(table_offset)}),
		                         Value(uint32_t {1}), Value(uint32_t {0})});
		return Make(ValueOpcode::IAdd32, {high, carry});
	}

	// A descriptor whose dwords are the given values.
	uint32_t Source(std::initializer_list<Value> dwords, bool indirect = false) {
		DescriptorSource source;
		for (const auto& dword: dwords)
			source.dwords[source.dword_count++] = dword;
		if (indirect) source.indirect_descriptor.emplace();
		plan.descriptor_sources.push_back(std::move(source));
		return static_cast<uint32_t>(plan.descriptor_sources.size() - 1);
	}

	uint32_t UserSource(uint32_t first, uint32_t count) {
		DescriptorSource source;
		for (uint32_t word = 0; word < count; word++)
			source.dwords[source.dword_count++] = User(first + word);
		plan.descriptor_sources.push_back(std::move(source));
		return static_cast<uint32_t>(plan.descriptor_sources.size() - 1);
	}

	uint32_t TableSource(uint32_t pointer, uint32_t first_offset, uint32_t count) {
		DescriptorSource source;
		for (uint32_t word = 0; word < count; word++) {
			source.dwords[source.dword_count++] = TableDword(pointer, first_offset + 4 * word);
		}
		plan.descriptor_sources.push_back(std::move(source));
		return static_cast<uint32_t>(plan.descriptor_sources.size() - 1);
	}

	// A literal pool descriptor. The high dword's pc may differ from the low dword's.
	uint32_t LiteralSource(bool swapped = false, uint32_t high_pc = LITERAL_PC) {
		return Source({LiteralLow(LITERAL_PC, LITERAL_TABLE, swapped),
		               LiteralHigh(high_pc, LITERAL_TABLE), Value(LITERAL_WORD2),
		               Value(LITERAL_WORD3)});
	}

	void AddBuffer(uint32_t source, bool formatted = false, bool scalar = false,
	               bool written = false) {
		BufferResource buffer;
		buffer.source    = source;
		buffer.formatted = formatted;
		buffer.scalar    = scalar;
		buffer.written   = written;
		plan.info.buffers.push_back(buffer);
	}

	void AddImage(uint32_t source, bool r128 = false, bool cube = false) {
		ImageResource image;
		image.source    = source;
		image.r128      = r128;
		image.cube      = cube;
		image.dimension = ShaderRecompiler::Decoder::ImageDimension::Dim2DArray;
		plan.info.images.push_back(image);
	}

	void AddSampler(uint32_t source) {
		SamplerResource sampler;
		sampler.source = source;
		plan.info.samplers.push_back(sampler);
	}
};

// A key for descriptors in consecutive user SGPRs from `sgpr`.
DescriptorKey UserKey(uint32_t sgpr) {
	return {DescriptorKeyKind::User, sgpr, 0, 0, 0};
}
// A key for a descriptor in the EUD table that `pointer` holds, at a byte offset.
DescriptorKey EudKey(uint32_t pointer, uint32_t offset) {
	return {DescriptorKeyKind::Eud, pointer, offset, 0, 0};
}

void TestUserDescriptor() {
	PlanBuilder builder;
	builder.AddBuffer(builder.UserSource(4, 4), true, false, false);
	builder.AddBuffer(builder.UserSource(8, 4), false, true, false);
	builder.AddBuffer(builder.UserSource(12, 4), false, false, true);
	builder.AddBuffer(builder.UserSource(16, 4));
	const auto facts = ExtractResourceFacts(builder.plan);
	CHECK(facts.has_value());
	CHECK(facts->buffers.size() == 4);
	CHECK(facts->buffers[0].key == UserKey(4));
	// Each flag is Kyty's own, and each buffer sets one flag, so a swapped flag shows. The last
	// buffer is the baseline with none set.
	const auto flags = [&](size_t index) {
		const auto& buffer = facts->buffers[index];
		return std::array<bool, 3> {buffer.formatted, buffer.scalar, buffer.written};
	};
	CHECK((flags(0) == std::array<bool, 3> {true, false, false}));
	CHECK((flags(1) == std::array<bool, 3> {false, true, false}));
	CHECK((flags(2) == std::array<bool, 3> {false, false, true}));
	CHECK((flags(3) == std::array<bool, 3> {false, false, false}));
	// Consecutive SGPRs are required. Registers 4, 5, 7, 6 are not a descriptor.
	PlanBuilder broken;
	broken.AddBuffer(
	    broken.Source({broken.User(4), broken.User(5), broken.User(7), broken.User(6)}));
	CHECK(!ExtractResourceFacts(broken.plan).has_value());
}

void TestEudDescriptorTakesOffsetFromMemoryInfo() {
	PlanBuilder builder;
	builder.AddBuffer(builder.TableSource(EUD_POINTER, TABLE_OFFSET, 4));
	const auto facts = ExtractResourceFacts(builder.plan);
	CHECK(facts.has_value());
	// The byte offset is the memory metadata's, whereas the load's own argument is zero. Taking the
	// argument would give offset 0.
	CHECK(facts.has_value() && facts->buffers[0].key == EudKey(EUD_POINTER, TABLE_OFFSET));
	// The dynamic part adds to it.
	PlanBuilder      dynamic;
	DescriptorSource source;
	for (uint32_t word = 0; word < 4; word++)
		source.dwords[source.dword_count++] =
		    dynamic.TableDword(EUD_POINTER, TABLE_OFFSET - 8 + 4 * word, 8);
	dynamic.plan.descriptor_sources.push_back(source);
	dynamic.AddBuffer(0);
	CHECK(ExtractResourceFacts(dynamic.plan)->buffers[0].key == EudKey(EUD_POINTER, TABLE_OFFSET));
	// Offsets that skip a dword are not one descriptor. This is the baseline's single change.
	PlanBuilder gap;
	gap.AddBuffer(gap.Source({gap.TableDword(EUD_POINTER, 96), gap.TableDword(EUD_POINTER, 100),
	                          gap.TableDword(EUD_POINTER, 108), gap.TableDword(EUD_POINTER, 112)}));
	CHECK(!ExtractResourceFacts(gap.plan).has_value());
	// A pointer that is not a register pair is not an EUD table.
	PlanBuilder      split;
	DescriptorSource bad;
	for (uint32_t word = 0; word < 4; word++) {
		MemoryInfo memory;
		memory.offset = word * 4;
		split.plan.memory_info.push_back(memory);
		const auto handle =
		    split.Make(ValueOpcode::GetAddressResource, {split.User(12), split.User(14)});
		const auto read =
		    split.Make(ValueOpcode::LoadAddressU32,
		               {handle, Value(uint32_t {0}), Value(uint32_t {0}), Value(false)});
		read.Instruction()->SetFlags(MemoryFlags {
		    .index = static_cast<uint32_t>(split.plan.memory_info.size() - 1), .pc = 0});
		split.plan.srt_reads.push_back({read, static_cast<uint32_t>(split.plan.srt_reads.size())});
		bad.dwords[bad.dword_count++] =
		    split.Make(ValueOpcode::ReadConst,
		               {split.Make(ValueOpcode::GetSrtResource),
		                Value(static_cast<uint32_t>(split.plan.srt_reads.size() - 1))});
	}
	split.plan.descriptor_sources.push_back(bad);
	split.AddBuffer(0);
	CHECK(!ExtractResourceFacts(split.plan).has_value());
}

void TestLiteralDescriptor() {
	for (const bool swapped: {false, true}) {
		PlanBuilder builder;
		builder.AddBuffer(builder.LiteralSource(swapped));
		const auto facts = ExtractResourceFacts(builder.plan);
		CHECK(facts.has_value());
		if (!facts) continue;
		const auto& key = facts->buffers[0].key;
		CHECK(key.kind == DescriptorKeyKind::Literal && key.base == LITERAL_PC &&
		      key.offset == LITERAL_TABLE);
		CHECK(key.literal_word2 == LITERAL_WORD2 && key.literal_word3 == LITERAL_WORD3);
	}
	// With a word that is not a constant, the same shape is an unknown descriptor.
	PlanBuilder unknown;
	unknown.AddBuffer(unknown.Source({unknown.LiteralLow(LITERAL_PC, LITERAL_TABLE),
	                                  unknown.LiteralHigh(LITERAL_PC, LITERAL_TABLE),
	                                  unknown.User(3), Value(LITERAL_WORD3)}));
	CHECK(!ExtractResourceFacts(unknown.plan).has_value());
	// The high dword must be the high half of the same pc sum. Another pc, a dword that is not an
	// add of a pc half, and a low dword that extracts the high half are not a literal table.
	PlanBuilder other_pc;
	other_pc.AddBuffer(other_pc.LiteralSource(false, LITERAL_PC + 4));
	CHECK(!ExtractResourceFacts(other_pc.plan).has_value());
	PlanBuilder plain_high;
	plain_high.AddBuffer(
	    plain_high.Source({plain_high.LiteralLow(LITERAL_PC, LITERAL_TABLE), Value(uint32_t {1}),
	                       Value(LITERAL_WORD2), Value(LITERAL_WORD3)}));
	CHECK(!ExtractResourceFacts(plain_high.plan).has_value());
	PlanBuilder wrong_half;
	wrong_half.AddBuffer(
	    wrong_half.Source({wrong_half.LiteralLow(LITERAL_PC, LITERAL_TABLE, false, 1),
	                       wrong_half.LiteralHigh(LITERAL_PC, LITERAL_TABLE), Value(LITERAL_WORD2),
	                       Value(LITERAL_WORD3)}));
	CHECK(!ExtractResourceFacts(wrong_half.plan).has_value());
}

void TestSkips() {
	// The baseline, a plain user descriptor, is accepted.
	PlanBuilder baseline;
	baseline.AddBuffer(baseline.UserSource(0, 4));
	CHECK(ExtractResourceFacts(baseline.plan).has_value());
	// An indirect descriptor skips the program, with a reason.
	PlanBuilder indirect;
	indirect.AddBuffer(indirect.Source(
	    {indirect.User(0), indirect.User(1), indirect.User(2), indirect.User(3)}, true));
	std::string why;
	CHECK(!ExtractResourceFacts(indirect.plan, &why).has_value());
	CHECK(why.find("buffer 0") == 0);
	// A dword that is arithmetic on a user register is not classified.
	PlanBuilder unknown;
	unknown.AddBuffer(
	    unknown.Source({unknown.Make(ValueOpcode::IAdd32, {unknown.User(0), Value(uint32_t {4})}),
	                    unknown.User(1), unknown.User(2), unknown.User(3)}));
	CHECK(!ExtractResourceFacts(unknown.plan).has_value());
	// A source index outside the plan, and a descriptor with too few dwords, are skipped.
	PlanBuilder range;
	range.AddBuffer(7);
	CHECK(!ExtractResourceFacts(range.plan).has_value());
	PlanBuilder narrow;
	narrow.AddBuffer(narrow.UserSource(0, 3));
	CHECK(!ExtractResourceFacts(narrow.plan).has_value());
	// An indirect buffer root skips the program even when its descriptor is plain.
	PlanBuilder root;
	root.AddBuffer(root.UserSource(0, 4));
	root.plan.info.buffers[0].indirect_root = 1;
	CHECK(!ExtractResourceFacts(root.plan).has_value());
}

void TestStaticSamplerIsKeptWithoutKey() {
	// The baseline sampler is four user SGPRs and has a key.
	PlanBuilder baseline;
	baseline.AddBuffer(baseline.UserSource(0, 4));
	baseline.AddSampler(baseline.UserSource(8, 4));
	const auto plain = ExtractResourceFacts(baseline.plan);
	CHECK(plain.has_value() && plain->samplers.size() == 1 && plain->samplers[0] == UserKey(8));
	// A static sampler made of four immediate dwords is still counted, with no key.
	PlanBuilder immediate;
	immediate.AddBuffer(immediate.UserSource(0, 4));
	immediate.AddSampler(immediate.Source(
	    {Value(uint32_t {1}), Value(uint32_t {2}), Value(uint32_t {3}), Value(uint32_t {4})}));
	const auto facts = ExtractResourceFacts(immediate.plan);
	CHECK(facts.has_value());
	CHECK(facts.has_value() && facts->samplers.size() == 1 && !facts->samplers[0].has_value());
	CHECK(facts.has_value() && facts->buffers.size() == 1);
	// An indirect sampler descriptor still skips the program.
	PlanBuilder indirect;
	indirect.AddSampler(indirect.Source(
	    {indirect.User(0), indirect.User(1), indirect.User(2), indirect.User(3)}, true));
	CHECK(!ExtractResourceFacts(indirect.plan).has_value());
}

void TestMalformedUserDataInstruction() {
	// A user data read with no register argument is not a descriptor dword. The baseline has the
	// register.
	PlanBuilder baseline;
	baseline.AddBuffer(baseline.UserSource(0, 4));
	CHECK(ExtractResourceFacts(baseline.plan).has_value());
	PlanBuilder bare;
	bare.AddBuffer(bare.Source(
	    {bare.Make(ValueOpcode::GetUserData), bare.User(1), bare.User(2), bare.User(3)}));
	CHECK(!ExtractResourceFacts(bare.plan).has_value());
}

void TestOrderAndImagesAndSamplers() {
	PlanBuilder builder;
	builder.AddBuffer(builder.TableSource(EUD_POINTER, TABLE_OFFSET, 4));
	builder.AddBuffer(builder.UserSource(8, 4));
	builder.AddBuffer(builder.LiteralSource());
	builder.AddImage(builder.UserSource(0, 8), false, true);
	builder.AddImage(builder.UserSource(16, 4), true);
	builder.AddSampler(builder.TableSource(EUD_POINTER, 0, 4));
	const auto facts = ExtractResourceFacts(builder.plan);
	CHECK(facts.has_value());
	if (!facts) return;
	// Kyty's order is kept.
	CHECK(facts->buffers.size() == 3);
	CHECK(facts->buffers[0].key == EudKey(EUD_POINTER, TABLE_OFFSET));
	CHECK(facts->buffers[1].key == UserKey(8));
	CHECK(facts->buffers[2].key.kind == DescriptorKeyKind::Literal);
	CHECK(facts->images.size() == 2);
	CHECK(facts->images[0].key == UserKey(0) && facts->images[0].cube);
	CHECK(facts->images[0].dimension ==
	      static_cast<uint32_t>(ShaderRecompiler::Decoder::ImageDimension::Dim2DArray));
	// A 128-bit image is four dwords wide, and the next image is not cube.
	CHECK(facts->images[1].key == UserKey(16) && !facts->images[1].cube);
	CHECK(facts->samplers.size() == 1 && facts->samplers[0] == EudKey(EUD_POINTER, 0));
	// An eight-dword image needs eight dwords, whereas a four-dword source that is not r128 is too
	// short.
	PlanBuilder short_image;
	short_image.AddImage(short_image.UserSource(0, 4), false);
	CHECK(!ExtractResourceFacts(short_image.plan).has_value());
	// An indirect sampler skips the program, against the plain one above.
	PlanBuilder indirect_sampler;
	indirect_sampler.AddSampler(indirect_sampler.UserSource(0, 4));
	CHECK(ExtractResourceFacts(indirect_sampler.plan).has_value());
	indirect_sampler.plan.info.samplers[0].indirect_root = 0;
	CHECK(!ExtractResourceFacts(indirect_sampler.plan).has_value());
}

} // namespace

int main() {
	TestUserDescriptor();
	TestEudDescriptorTakesOffsetFromMemoryInfo();
	TestLiteralDescriptor();
	TestSkips();
	TestStaticSamplerIsKeptWithoutKey();
	TestMalformedUserDataInstruction();
	TestOrderAndImagesAndSamplers();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("shader store keys tests passed\n");
	return 0;
}

// The typed IR's implementation, amalgamated to keep this target small like
// scalar_provenance_tests.
#include "../src/graphics/shader/recompiler/ir/Block.cpp"
#include "../src/graphics/shader/recompiler/ir/Program.cpp"
#include "../src/graphics/shader/recompiler/ir/Type.cpp"
#include "../src/graphics/shader/recompiler/ir/Value.cpp"
#include "../src/graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
