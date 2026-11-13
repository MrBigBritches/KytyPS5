// Tests the shader store record format that the pipeline cache and the store generator share. The
// modules are synthetic, so no game file and no device are needed. The expected bytes are a golden
// written by the pre-change pipeline cache encoding, before EncodeModuleRecord existed, so a change
// in the format cannot pass by agreeing with itself.
#include "ShaderStoreRecordFixture.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderStoreRecord.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>
#include <xxhash.h>

namespace {

using namespace Libs::Graphics;
namespace Fixture = StoreRecordFixture;
namespace IR      = Libs::Graphics::ShaderRecompiler::IR;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Bytes = std::vector<uint8_t>;

#include "ShaderStoreRecordGolden.inc"

Bytes GoldenOf(const uint8_t* begin, size_t size) {
	return {begin, begin + size};
}

Bytes GoldenCompute() {
	return GoldenOf(GOLDEN_RECORD_COMPUTE, sizeof(GOLDEN_RECORD_COMPUTE));
}

Bytes GoldenVertex() {
	return GoldenOf(GOLDEN_RECORD_VERTEX, sizeof(GOLDEN_RECORD_VERTEX));
}

// A frame is a StoreFrame and then the compressed bytes.
StoreFrame FrameOf(const Bytes& frame) {
	StoreFrame header;
	std::memcpy(&header, frame.data(), sizeof(header));
	return header;
}

// ---- The record bytes. ----

// Only an encoder that writes the same bytes as the old code for these inputs satisfies this. The
// two modules differ in every field, so an encoder that ignores its input fails one of them.
void TestRecordEqualsGolden() {
	const auto compute = Fixture::Make(0);
	const auto vertex  = Fixture::Make(1);
	CHECK(EncodeModuleRecord(compute.Inputs()) == GoldenCompute());
	CHECK(EncodeModuleRecord(vertex.Inputs()) == GoldenVertex());
	// The baseline that keeps the two checks apart.
	CHECK(GoldenCompute() != GoldenVertex());
}

// The first byte is what LoadRecord switches on, and the enum values are the store format.
void TestRecordTypeByte() {
	CHECK(static_cast<uint8_t>(StoreRecord::Module) == 0);
	CHECK(static_cast<uint8_t>(StoreRecord::GraphicsPipeline) == 1);
	CHECK(static_cast<uint8_t>(StoreRecord::TessellationLayout) == 2);
	const auto compute = Fixture::Make(0);
	CHECK(EncodeModuleRecord(compute.Inputs()).front() == 0);
}

enum class Field {
	None,
	Stage,
	Hash,
	UserDataCount,
	StaticState,
	Plan,
	Id,
	Info,
	Specialization,
	VertexFormats,
	RectParameters,
	MeshTask,
	Spirv,
};

// Encodes `base` with one field taken from `donor`. Every field of the two fixture modules differs.
Bytes EncodeWithDonor(const Fixture::Module& base, const Fixture::Module& donor, Field field) {
	const auto& of = [&](Field name) -> const Fixture::Module& {
		return field == name ? donor : base;
	};
	return EncodeModuleRecord({.stage           = of(Field::Stage).stage,
	                           .hash            = of(Field::Hash).hash,
	                           .user_data_count = of(Field::UserDataCount).user_data_count,
	                           .static_state    = of(Field::StaticState).static_state,
	                           .plan            = of(Field::Plan).plan,
	                           .id              = of(Field::Id).id,
	                           .info            = of(Field::Info).info,
	                           .specialization  = of(Field::Specialization).specialization,
	                           .vertex_formats  = of(Field::VertexFormats).vertex_formats,
	                           .rect_parameters = of(Field::RectParameters).rect_parameters,
	                           .mesh_task       = of(Field::MeshTask).mesh_task,
	                           .spirv           = of(Field::Spirv).spirv});
}

// Each input must reach the bytes. The golden already pins the values, and this names the field
// that is lost when one is not, against a baseline that differs in that field alone.
void TestEveryInputReachesTheRecord() {
	const auto base  = Fixture::Make(0);
	const auto donor = Fixture::Make(1);
	const auto plain = EncodeWithDonor(base, donor, Field::None);
	CHECK(plain == GoldenCompute());
	for (int field = static_cast<int>(Field::Stage); field <= static_cast<int>(Field::Spirv);
	     field++) {
		const auto mixed = EncodeWithDonor(base, donor, static_cast<Field>(field));
		if (mixed == plain) {
			std::fprintf(stderr, "field %d does not reach the record\n", field);
		}
		CHECK(mixed != plain);
	}
}

// ---- Reading a record back. ----

struct Loaded {
	DecodedModuleRecord record;
	IR::ResourcePlan    plan;
	bool                ok = false;
};

// Decodes a raw record the way LoadRecord does, and loads its plan.
Loaded Decode(const Bytes& raw) {
	Loaded     loaded;
	IR::Reader reader {raw};
	auto       type = StoreRecord::GraphicsPipeline;
	IR::IoAll(reader, type);
	DecodeModuleRecord(reader, loaded.record);
	loaded.ok = type == StoreRecord::Module && reader.ok && reader.bytes.empty() &&
	            IR::LoadResourcePlan(loaded.record.plan, loaded.plan);
	return loaded;
}

// The record of a decoded module, to compare bytes where the types have no equality.
Bytes Reencode(const Loaded& loaded) {
	const auto& record = loaded.record;
	return EncodeModuleRecord({.stage           = record.stage,
	                           .hash            = record.hash,
	                           .user_data_count = record.user_data_count,
	                           .static_state    = record.static_state,
	                           .plan            = loaded.plan,
	                           .id              = record.id,
	                           .info            = record.info,
	                           .specialization  = record.specialization,
	                           .vertex_formats  = record.vertex_formats,
	                           .rect_parameters = record.rect_parameters,
	                           .mesh_task       = record.mesh_task,
	                           .spirv           = record.spirv});
}

// The id and the plan that went in come out. Variant 1 is the baseline, since a decoder that
// returned fixed values or the other module's would match one of the two but not both.
void TestRoundTrip() {
	for (const uint32_t variant: {0u, 1u}) {
		const auto module   = Fixture::Make(variant);
		const auto framed   = FrameStoreRecord(EncodeModuleRecord(module.Inputs()));
		const auto unframed = UnframeStoreRecord(framed);
		CHECK(unframed.has_value());
		if (!unframed) continue;
		CHECK(unframed->frame_bytes == framed.size());
		const auto loaded = Decode(unframed->raw);
		CHECK(loaded.ok);
		const auto& record = loaded.record;
		CHECK(record.id == module.id);
		CHECK(record.stage == module.stage);
		CHECK(record.hash == module.hash);
		CHECK(record.user_data_count == module.user_data_count);
		CHECK(record.static_state == module.static_state);
		CHECK(record.spirv == module.spirv);
		CHECK(record.vertex_formats == module.vertex_formats);
		CHECK(record.mesh_task == module.mesh_task);
		CHECK(record.specialization == module.specialization);
		CHECK(record.info.bindings.push_data_start_dword ==
		      module.info.bindings.push_data_start_dword);
		// The plan that was stored is the plan that loaded.
		CHECK(record.plan == IR::SaveResourcePlan(module.plan));
		CHECK(IR::SaveResourcePlan(loaded.plan) == record.plan);
		CHECK(loaded.plan.shader_hash == module.hash);
		// Every field, including the ones with no equality, comes back as it went in.
		CHECK(Reencode(loaded) == unframed->raw);
	}
	// As a baseline, a different module's plan and id do not read back as this one's.
	const auto first  = Decode(GoldenCompute());
	const auto second = Decode(GoldenVertex());
	CHECK(first.ok && second.ok);
	CHECK(first.record.id != second.record.id);
	CHECK(first.record.plan != second.record.plan);
}

// A record cut short or with bytes left over does not decode cleanly, which LoadRecord relies on.
void TestDecodeRejectsDamage() {
	auto truncated = GoldenCompute();
	truncated.pop_back();
	CHECK(!Decode(truncated).ok);
	auto extended = GoldenCompute();
	extended.push_back(0);
	CHECK(!Decode(extended).ok);
	CHECK(Decode(GoldenCompute()).ok);
	auto wrong_type = GoldenCompute();
	wrong_type[0]   = static_cast<uint8_t>(StoreRecord::TessellationLayout);
	CHECK(!Decode(wrong_type).ok);
}

// ---- Framing. ----

// The compressed bytes depend on the zstd version, which is vendored, and on the level. This
// golden is what the old Append wrote, so a change of level or of the frame fields fails it. A
// later zstd upgrade can legitimately break it while old stores still load, which
// TestOldStoreStillLoads checks without depending on zstd's output.
void TestFrameEqualsGolden() {
	const auto framed = FrameStoreRecord(GoldenCompute());
	CHECK(framed == GoldenOf(GOLDEN_FRAME_COMPUTE, sizeof(GOLDEN_FRAME_COMPUTE)));
	CHECK(framed != FrameStoreRecord(GoldenVertex()));
	CHECK(sizeof(StoreFrame) == 16);
	CHECK(STORE_COMPRESSION_LEVEL == 3);
}

// The frame fields, from the bytes and not from the code under test.
void TestFrameFields() {
	const auto raw    = GoldenCompute();
	const auto framed = FrameStoreRecord(raw);
	const auto header = FrameOf(framed);
	CHECK(header.raw_size == raw.size());
	CHECK(header.compressed_size == framed.size() - sizeof(StoreFrame));
	// The checksum covers the compressed bytes, not the raw record.
	CHECK(header.checksum ==
	      XXH3_64bits(framed.data() + sizeof(StoreFrame), header.compressed_size));
	CHECK(header.checksum != XXH3_64bits(raw.data(), raw.size()));
}

// Frames follow one another, and the reader steps by each one's own size.
void TestFramesInSequence() {
	const auto first  = FrameStoreRecord(GoldenCompute());
	const auto second = FrameStoreRecord(GoldenVertex());
	Bytes      store  = first;
	store.insert(store.end(), second.begin(), second.end());
	const auto one = UnframeStoreRecord(store);
	CHECK(one.has_value() && one->raw == GoldenCompute() && one->frame_bytes == first.size());
	if (!one) return;
	const auto two = UnframeStoreRecord(std::span<const uint8_t>(store).subspan(one->frame_bytes));
	CHECK(two.has_value() && two->raw == GoldenVertex() && two->frame_bytes == second.size());
	// Nothing after the last frame.
	CHECK(!UnframeStoreRecord(std::span<const uint8_t>(store).subspan(store.size())).has_value());
}

// Every way a frame can be damaged ends the store there. Each is checked against the intact frame,
// so a reader that rejects every frame would not pass the first check.
void TestDamagedFramesAreRefused() {
	const auto framed = FrameStoreRecord(GoldenCompute());
	CHECK(UnframeStoreRecord(framed).has_value());
	// Cut short, in the header and in the compressed bytes.
	CHECK(!UnframeStoreRecord(std::span<const uint8_t>(framed).first(sizeof(StoreFrame) - 1)));
	CHECK(!UnframeStoreRecord(std::span<const uint8_t>(framed).first(framed.size() - 1)));
	// A flipped byte inside the compressed payload, which is past zstd's frame header, so only the
	// checksum can tell.
	auto flipped = framed;
	flipped[framed.size() - 8] ^= 0x01;
	CHECK(!UnframeStoreRecord(flipped));
	// A checksum that is wrong while the compressed bytes are intact.
	auto checksum = framed;
	checksum[offsetof(StoreFrame, checksum)] ^= 0x01;
	CHECK(!UnframeStoreRecord(checksum));
	// A raw size that disagrees with the zstd frame.
	auto sized = framed;
	sized[offsetof(StoreFrame, raw_size)] ^= 0x01;
	CHECK(!UnframeStoreRecord(sized));
	// A compressed size past the end of the store.
	auto length = framed;
	length[offsetof(StoreFrame, compressed_size) + 1] ^= 0x40;
	CHECK(!UnframeStoreRecord(length));
}

// Counts allocations of 64 MiB or more, which no test here makes on purpose, and refuses them.
constexpr size_t  HUGE_ALLOCATION_BYTES = size_t {64} << 20;
std::atomic<bool> g_huge_allocation {false};

// The checksum does not cover the frame header, so raw_size is untrusted. A frame that claims the
// largest size must be refused before anything is allocated for it.
void TestHugeRawSizeAllocatesNothing() {
	const auto     framed  = FrameStoreRecord(GoldenCompute());
	auto           huge    = framed;
	const uint32_t claimed = UINT32_MAX;
	std::memcpy(huge.data() + offsetof(StoreFrame, raw_size), &claimed, sizeof(claimed));
	g_huge_allocation = false;
	bool refused      = false;
	try {
		refused = !UnframeStoreRecord(huge).has_value();
	} catch (const std::bad_alloc&) {
	}
	CHECK(refused);
	CHECK(!g_huge_allocation);
	// The intact frame still reads, so a reader that refuses everything does not pass.
	CHECK(UnframeStoreRecord(framed).has_value());
}

// Stores written before this code, with the frame the old Append wrote, must still load. This does
// not depend on what zstd outputs today.
void TestOldStoreStillLoads() {
	const auto golden   = GoldenOf(GOLDEN_FRAME_COMPUTE, sizeof(GOLDEN_FRAME_COMPUTE));
	const auto unframed = UnframeStoreRecord(golden);
	CHECK(unframed.has_value());
	if (!unframed) return;
	CHECK(unframed->raw == GoldenCompute());
	CHECK(unframed->frame_bytes == golden.size());
	const auto loaded = Decode(unframed->raw);
	CHECK(loaded.ok);
	CHECK(loaded.record.id == Fixture::Make(0).id);
}

// ---- The module and key bindings. ----

// The shape of the pipeline cache's ShaderModule and ProgramKey, with an int enum for the formats.
enum class MockFormat : int32_t {};

struct MockModule {
	uint64_t                       id         = 0;
	int                            handle     = 77;
	int                            gpl_handle = 78;
	IR::CompiledShaderInfo         info;
	IR::ResourceSpecialization     specialization;
	std::vector<uint32_t>          function_code {5, 6};
	std::vector<MockFormat>        vertex_formats;
	std::vector<RectListParameter> rect_parameters;
	bool                           mesh_task = false;
	// Stands for the retained record, which neither a fill nor an encode touches.
	int retained = 79;
};

struct MockKey {
	ShaderType            stage           = ShaderType::Unknown;
	uint64_t              hash            = 0;
	uint32_t              user_data_count = 0;
	std::vector<uint32_t> static_state;
};

MockModule MockModuleOf(const Fixture::Module& module) {
	MockModule mock;
	mock.id              = module.id;
	mock.info            = module.info;
	mock.specialization  = module.specialization;
	mock.rect_parameters = module.rect_parameters;
	mock.mesh_task       = module.mesh_task;
	for (const auto raw: module.vertex_formats) {
		mock.vertex_formats.push_back(static_cast<MockFormat>(raw));
	}
	return mock;
}

MockKey MockKeyOf(const Fixture::Module& module) {
	return {module.stage, module.hash, module.user_data_count, module.static_state};
}

bool SameRect(const std::vector<RectListParameter>& a, const std::vector<RectListParameter>& b) {
	return std::ranges::equal(a, b, [](const auto& x, const auto& y) {
		return x.input_location == y.input_location && x.output_location == y.output_location &&
		       x.flat == y.flat;
	});
}

// A decoded record fills the module field by field. The vertex module fills every field of the tail
// and the compute module is the baseline that differs in exactly those, so a fill that drops one
// leaves the mock as the baseline has it.
void TestFillModuleFromRecord() {
	const auto vertex  = Fixture::Make(1);
	const auto compute = Fixture::Make(0);
	MockModule filled;
	FillModuleFromRecord(std::move(Decode(GoldenVertex()).record), filled);
	CHECK(filled.id == vertex.id);
	CHECK(filled.mesh_task);
	CHECK(filled.vertex_formats.size() == vertex.vertex_formats.size());
	for (size_t i = 0; i < filled.vertex_formats.size() && i < vertex.vertex_formats.size(); i++) {
		CHECK(static_cast<int32_t>(filled.vertex_formats[i]) == vertex.vertex_formats[i]);
	}
	CHECK(SameRect(filled.rect_parameters, vertex.rect_parameters));
	CHECK(!filled.rect_parameters.empty());
	CHECK(filled.specialization == vertex.specialization);
	CHECK(filled.info.bindings.push_data_start_dword == vertex.info.bindings.push_data_start_dword);
	CHECK(filled.info.wave_size == vertex.info.wave_size);
	// What a stored module does not hold stays as it was.
	CHECK(filled.handle == 77 && filled.gpl_handle == 78);
	CHECK((filled.function_code == std::vector<uint32_t> {5, 6}));
	CHECK(filled.retained == 79);
	// The baseline fills the compute module over stale values, so the tail must be reset.
	MockModule baseline;
	baseline.mesh_task       = true;
	baseline.vertex_formats  = {MockFormat {9}};
	baseline.rect_parameters = vertex.rect_parameters;
	FillModuleFromRecord(std::move(Decode(GoldenCompute()).record), baseline);
	CHECK(baseline.id == compute.id);
	CHECK(!baseline.mesh_task);
	CHECK(baseline.vertex_formats.empty());
	CHECK(baseline.rect_parameters.empty());
}

void TestFillKeyFromRecord() {
	const auto vertex = Fixture::Make(1);
	auto       record = Decode(GoldenVertex()).record;
	MockKey    key;
	FillKeyFromRecord(record, key);
	CHECK(key.stage == vertex.stage);
	CHECK(key.hash == vertex.hash);
	CHECK(key.user_data_count == vertex.user_data_count);
	CHECK(key.static_state == vertex.static_state);
	// The baseline key is another program's.
	auto    other = Decode(GoldenCompute()).record;
	MockKey other_key;
	FillKeyFromRecord(other, other_key);
	CHECK(other_key.hash != key.hash && other_key.static_state != key.static_state);
}

// Inputs built from a module and a key of the cache's shape encode to the golden. Each one-change
// copy of the mock must then differ, so every field of both types reaches the record.
void TestRecordInputsOf() {
	const auto           vertex = Fixture::Make(1);
	const auto           key    = MockKeyOf(vertex);
	const auto           mock   = MockModuleOf(vertex);
	std::vector<int32_t> formats;
	const auto           encode = [&](const MockKey& k, const MockModule& m) {
        return EncodeModuleRecord(RecordInputsOf(k, vertex.plan, m, formats, vertex.spirv));
	};
	CHECK(encode(key, mock) == GoldenVertex());
	const auto changed_module = [&](auto&& change) {
		auto copy = mock;
		change(copy);
		return encode(key, copy) != GoldenVertex();
	};
	CHECK(changed_module([](MockModule& m) { m.id ^= 1; }));
	CHECK(changed_module([](MockModule& m) { m.info.wave_size += 1; }));
	CHECK(changed_module([](MockModule& m) { m.specialization.buffers.clear(); }));
	CHECK(changed_module([](MockModule& m) { m.vertex_formats.clear(); }));
	CHECK(changed_module([](MockModule& m) { m.vertex_formats[0] = MockFormat {38}; }));
	CHECK(changed_module([](MockModule& m) { m.rect_parameters.clear(); }));
	CHECK(changed_module([](MockModule& m) { m.mesh_task = false; }));
	// The handles, the function code and the retained record are not stored.
	CHECK(!changed_module([](MockModule& m) {
		m.handle     = 1;
		m.gpl_handle = 2;
		m.function_code.clear();
		m.retained = 3;
	}));
	const auto changed_key = [&](auto&& change) {
		auto copy = key;
		change(copy);
		return encode(copy, mock) != GoldenVertex();
	};
	CHECK(changed_key([](MockKey& k) { k.stage = ShaderType::Pixel; }));
	CHECK(changed_key([](MockKey& k) { k.hash += 1; }));
	CHECK(changed_key([](MockKey& k) { k.user_data_count += 1; }));
	CHECK(changed_key([](MockKey& k) { k.static_state.push_back(1); }));
	// The formats are copied out as raw ints.
	CHECK((formats == std::vector<int32_t> {37, 109, 50}));
}

} // namespace

// Replaces global new in this binary so that TestHugeRawSizeAllocatesNothing can see an allocation
// the frame reader should never make.
void* operator new(size_t size) {
	if (size >= HUGE_ALLOCATION_BYTES) {
		g_huge_allocation = true;
		throw std::bad_alloc();
	}
	if (void* memory = std::malloc(size != 0 ? size : 1)) return memory;
	throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
	std::free(memory);
}

void operator delete(void* memory, size_t) noexcept {
	std::free(memory);
}

int main() {
	TestRecordEqualsGolden();
	TestRecordTypeByte();
	TestEveryInputReachesTheRecord();
	TestRoundTrip();
	TestDecodeRejectsDamage();
	TestFrameEqualsGolden();
	TestFrameFields();
	TestFramesInSequence();
	TestDamagedFramesAreRefused();
	TestHugeRawSizeAllocatesNothing();
	TestOldStoreStillLoads();
	TestFillModuleFromRecord();
	TestFillKeyFromRecord();
	TestRecordInputsOf();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("shader store record tests passed\n");
	return 0;
}

// The typed IR's implementation, amalgamated to keep this target small like
// shader_store_keys_tests.
#include "../src/graphics/shader/recompiler/ir/Block.cpp"
#include "../src/graphics/shader/recompiler/ir/Program.cpp"
#include "../src/graphics/shader/recompiler/ir/Type.cpp"
#include "../src/graphics/shader/recompiler/ir/Value.cpp"
#include "../src/graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
