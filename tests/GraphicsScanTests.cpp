// Tests the graphics half of the submit-time walk, with the command processor's own register tables
// and the GPU thread's pixel input code, which is why it links the whole emulator. Each check that
// proves a behaviour compares against a baseline stream that differs only in that behaviour.
#include "common/assert.h"
#include "common/emulatorConfig.h"
#include "common/gpuStall.h"
#include "graphics/guest_gpu/command_processor/graphicsScanShadow.h"
#include "graphics/guest_gpu/command_processor/graphicsScanSightings.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/prescanGraphics.h"
#include "graphics/shader/recompiler/frontend/translate/Translate.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/shaderCompiler.h"

#include <atomic>
#include <bit>
#include <chrono>
#include <map>
#include <thread>
#include <optional>
#include <span>
#include <string>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Stream = std::vector<uint32_t>;

constexpr uint64_t PS_ADDRESS = 0x0000004567890100ull;
// The guest address registers hold bits 8 to 39 in LO and 40 to 47 in HI.
constexpr uint32_t PGM_LO_SHIFT = 8;
constexpr uint32_t PGM_HI_SHIFT = 40;
// DB_SHADER_CONTROL with KILL_ENABLE, which makes the pixel shader run with no color target, so a
// draw needs no render target format to be offered.
constexpr uint32_t DB_SHADER_CONTROL_KILL = 1u << Pm4::DB_SHADER_CONTROL_KILL_ENABLE_SHIFT;
// One interpolator, and the perspective center input in both SPI_PS_INPUT_ENA and SPI_PS_INPUT_ADDR.
constexpr uint32_t ONE_INTERPOLATOR   = 1;
constexpr uint32_t PERSP_CENTER_INPUT = 0x2u;
// An interpolator setting a test can recognise.
constexpr uint32_t INTERPOLATOR_SETTING = 0x00000425u;
// CB_COLORn_ATTRIB3 with CMASK_PIPE_ALIGNED and not DCC_PIPE_ALIGNED, which its decoder EXITs on.
constexpr uint32_t ATTRIB3_MISALIGNED = 1u << Pm4::CB_COLOR0_ATTRIB3_CMASK_PIPE_ALIGNED_SHIFT;
// CONTEXT_STATE operations.
constexpr uint32_t CONTEXT_STATE_PUSH = 1;
constexpr uint32_t CONTEXT_STATE_POP  = 2;

void Append(Stream& stream, const Stream& more) {
	stream.insert(stream.end(), more.begin(), more.end());
}

Stream SetSh(uint32_t reg, const Stream& values) {
	Stream packet {KYTY_PM4(2 + values.size(), Pm4::IT_SET_SH_REG, 0), reg};
	Append(packet, values);
	return packet;
}

Stream SetContext(uint32_t reg, const Stream& values) {
	Stream packet {KYTY_PM4(2 + values.size(), Pm4::IT_SET_CONTEXT_REG, 0), reg};
	Append(packet, values);
	return packet;
}

Stream ContextState(uint32_t operation) {
	return {KYTY_PM4(3, Pm4::IT_NOP, Pm4::R_CONTEXT_STATE), operation, 0u};
}

Stream DrawAuto() {
	return {KYTY_PM4(3, Pm4::IT_DRAW_INDEX_AUTO, 0), 3u, 0u};
}

// The writes that make a pixel shader current and running, with one interpolated input, but without
// the interpolator's own setting.
Stream PixelSetup() {
	Stream stream;
	Append(stream, SetSh(Pm4::SPI_SHADER_PGM_LO_PS, {static_cast<uint32_t>(PS_ADDRESS >> PGM_LO_SHIFT),
	                                                 static_cast<uint32_t>(PS_ADDRESS >> PGM_HI_SHIFT)}));
	Append(stream, SetContext(Pm4::DB_SHADER_CONTROL, {DB_SHADER_CONTROL_KILL}));
	Append(stream, SetContext(Pm4::SPI_PS_INPUT_ENA, {PERSP_CENTER_INPUT, PERSP_CENTER_INPUT}));
	Append(stream, SetContext(Pm4::SPI_PS_IN_CONTROL, {ONE_INTERPOLATOR}));
	return stream;
}

// A reader for walks that follow no buffer.
ScanMemoryReader NoMemory() {
	return [](uint64_t, void*, uint64_t) { return false; };
}

struct GraphicsResult {
	std::vector<ScannedDraw> draws;
	GraphicsScanStats        stats;
	ScanStats                scan;
};

// Walks the stream on queue 0 with the given graphics shadow, the way the scanner walks the graphics
// queue, and takes what the shadow found.
GraphicsResult WalkWith(GraphicsScanShadow& graphics, const Stream& commands) {
	CsShadow                     compute;
	std::vector<ScannedDispatch> dispatches;
	ScanScratch                  scratch;
	GraphicsResult               result;
	SubmitScanner(NoMemory()).Scan(commands, 0, false, compute, dispatches, result.scan, scratch, nullptr,
	                               &graphics);
	result.draws = std::exchange(graphics.draws, {});
	result.stats = std::exchange(graphics.stats, {});
	return result;
}

// Walks the stream with a fresh graphics shadow.
GraphicsResult Walk(const Stream& commands) {
	GraphicsScanShadow graphics;
	return WalkWith(graphics, commands);
}

// The interpolator setting written by SET_CONTEXT_REG reaches the draw's input info. The baseline is
// the same stream without the write, whose draw has the reset value, so nothing but the write puts
// the setting there.
void TestInterpolatorWriteReachesDraw() {
	Stream with_write = PixelSetup();
	Append(with_write, SetContext(Pm4::SPI_PS_INPUT_CNTL_0, {INTERPOLATOR_SETTING}));
	Append(with_write, DrawAuto());
	Stream baseline = PixelSetup();
	Append(baseline, DrawAuto());

	const auto written = Walk(with_write);
	const auto plain   = Walk(baseline);
	CHECK(written.draws.size() == 1);
	CHECK(plain.draws.size() == 1);
	if (written.draws.size() == 1 && plain.draws.size() == 1) {
		CHECK(written.draws[0].ps_program_address == PS_ADDRESS);
		CHECK(written.draws[0].ps_info.input_num == ONE_INTERPOLATOR);
		CHECK(written.draws[0].ps_info.interpolator_settings[0] == INTERPOLATOR_SETTING);
		CHECK(plain.draws[0].ps_info.interpolator_settings[0] == 0);
		// The setting is part of the static key, so the two draws are told apart.
		CHECK(written.draws[0].key_hash != plain.draws[0].key_hash);
	}
}

// A draw whose pixel shader address was written before the last doubt is counted and not offered,
// since its program may be stale. The baseline walks the same two lists with no doubt between them and
// offers the draw, so only the doubt holds it back.
void TestDrawNeedsProgramWriteSinceDoubt() {
	Stream draw_only;
	Append(draw_only, DrawAuto());

	GraphicsScanShadow doubted;
	(void)WalkWith(doubted, PixelSetup());
	doubted.MarkUncertain();
	const auto held = WalkWith(doubted, draw_only);
	CHECK(held.draws.empty());
	CHECK(held.stats.ps_unwritten == 1);

	GraphicsScanShadow trusted;
	(void)WalkWith(trusted, PixelSetup());
	CHECK(WalkWith(trusted, draw_only).draws.size() == 1);
}

// A CONTEXT_STATE pop with no push, which the processor EXITs on, is counted and puts the shadow in
// doubt, so the draw after it is held back. The baseline pushes first and its draw is offered.
void TestPopWithoutPushDoesNotExit() {
	Stream unbalanced = PixelSetup();
	Append(unbalanced, ContextState(CONTEXT_STATE_POP));
	Append(unbalanced, DrawAuto());
	const auto popped = Walk(unbalanced);
	CHECK(popped.stats.context_state_imbalance == 1);
	CHECK(popped.draws.empty());
	CHECK(popped.stats.ps_unwritten == 1);

	Stream balanced = PixelSetup();
	Append(balanced, ContextState(CONTEXT_STATE_PUSH));
	Append(balanced, ContextState(CONTEXT_STATE_POP));
	Append(balanced, DrawAuto());
	const auto pushed = Walk(balanced);
	CHECK(pushed.stats.context_state_imbalance == 0);
	CHECK(pushed.draws.size() == 1);
}

// A CB_COLOR0_ATTRIB3 value whose decoder EXITs is left out and counted, and the walk goes on to the
// draw. Without the deny list this test does not return.
void TestDeniedAttrib3DoesNotExit() {
	Stream stream = PixelSetup();
	Append(stream, SetContext(Pm4::CB_COLOR0_ATTRIB3, {ATTRIB3_MISALIGNED}));
	Append(stream, DrawAuto());
	const auto result = Walk(stream);
	CHECK(result.stats.registers_denied == 1);
	CHECK(result.draws.size() == 1);
}

// SPI_PS_INPUT_ENA differing from SPI_PS_INPUT_ADDR, which ShaderGetStaticInputInfoPS EXITs on, is
// counted and not offered. The baseline, PixelSetup's equal pair, is offered.
void TestUnsupportedInputsAreCounted() {
	Stream stream = PixelSetup();
	Append(stream, SetContext(Pm4::SPI_PS_INPUT_ADDR, {PERSP_CENTER_INPUT | 1u}));
	Append(stream, DrawAuto());
	const auto result = Walk(stream);
	CHECK(result.stats.unsupported_inputs == 1);
	CHECK(result.draws.empty());
}

// The same draw twice in one list is offered once, and the cache counts the repeat.
void TestRepeatedDrawIsCached() {
	Stream stream = PixelSetup();
	Append(stream, DrawAuto());
	Append(stream, DrawAuto());
	const auto result = Walk(stream);
	CHECK(result.draws.size() == 1);
	CHECK(result.stats.cached == 1);
}

// User clip plane 0, and CLIP_DISABLE, in PA_CL_CLIP_CNTL.
constexpr uint32_t CLIP_PLANE_0 = 1u << Pm4::PA_CL_CLIP_CNTL_UCP_ENA_SHIFT;
constexpr uint32_t CLIP_DISABLE = 1u << Pm4::PA_CL_CLIP_CNTL_CLIP_DISABLE_SHIFT;
// The push cursors GetGraphicsPrograms gives a plain vertex stage with no planes and with plane 0.
constexpr uint32_t CURSOR_NO_PLANES = 0;
constexpr uint32_t CURSOR_PLANE_0   = 9;

// The same pixel shader drawn with user clip plane 0 and without it gives two draws, whose push
// cursors are those the GPU thread computes, 9 and 0. The baseline is one draw without planes, which
// gives one draw at cursor 0, so the second draw and its cursor come from the plane.
void TestUserClipPlanesMoveThePushCursor() {
	Stream both = PixelSetup();
	Append(both, SetContext(Pm4::PA_CL_CLIP_CNTL, {CLIP_PLANE_0}));
	Append(both, DrawAuto());
	Append(both, SetContext(Pm4::PA_CL_CLIP_CNTL, {0u}));
	Append(both, DrawAuto());
	const auto two = Walk(both);
	CHECK(two.draws.size() == 2);
	if (two.draws.size() == 2) {
		CHECK(ScannedDrawPushCursor(two.draws[0], false) == CURSOR_PLANE_0);
		CHECK(ScannedDrawPushCursor(two.draws[1], false) == CURSOR_NO_PLANES);
	}
	Stream plain = PixelSetup();
	Append(plain, DrawAuto());
	const auto one = Walk(plain);
	CHECK(one.draws.size() == 1);
	if (one.draws.size() == 1) {
		CHECK(ScannedDrawPushCursor(one.draws[0], false) == CURSOR_NO_PLANES);
	}
}

// CLIP_DISABLE turns the planes off, as on the GPU thread, and planes with a vertex stage that writes
// clip distances, which GetGraphicsPrograms EXITs on, are skipped and counted.
void TestClipPlaneSkipsOnlyTheExitCase() {
	Stream disabled = PixelSetup();
	Append(disabled, SetContext(Pm4::PA_CL_CLIP_CNTL, {CLIP_PLANE_0 | CLIP_DISABLE}));
	Append(disabled, DrawAuto());
	const auto off = Walk(disabled);
	CHECK(off.draws.size() == 1 && off.draws[0].user_clip_planes == 0);

	constexpr uint32_t clip_distance_0 = 1;
	Stream exiting = PixelSetup();
	Append(exiting, SetContext(Pm4::PA_CL_VS_OUT_CNTL, {clip_distance_0}));
	Append(exiting, SetContext(Pm4::PA_CL_CLIP_CNTL, {CLIP_PLANE_0}));
	Append(exiting, DrawAuto());
	const auto skipped = Walk(exiting);
	CHECK(skipped.draws.empty() && skipped.stats.skipped_clip_planes == 1);
}

// Guest memory for the vertex key test, a few regions read by address.
struct VertexMemory {
	std::map<uint64_t, std::vector<uint32_t>> regions;

	static bool Read(void* userdata, uint64_t address, std::span<uint32_t> words) {
		const auto& memory = *static_cast<VertexMemory*>(userdata);
		for (const auto& [base, region]: memory.regions) {
			const uint64_t size = region.size() * sizeof(uint32_t);
			if (address >= base && address + words.size_bytes() <= base + size) {
				std::memcpy(words.data(), reinterpret_cast<const uint8_t*>(region.data()) + (address - base),
				            words.size_bytes());
				return true;
			}
		}
		return false;
	}
};

constexpr uint64_t SEMANTICS_ADDRESS = 0x10000;
constexpr uint64_t ATTRIB_A_ADDRESS  = 0x20000;
constexpr uint64_t ATTRIB_B_ADDRESS  = 0x30000;
constexpr uint64_t BUFFER_ADDRESS    = 0x40000;
// The user registers the attribute and buffer table pointers sit in.
constexpr int ATTRIB_REG = 0;
constexpr int BUFFER_REG = 2;
// An attribute of four channels at VGPR 4 from semantic 0, and two formats that differ.
constexpr uint32_t FORMAT_SHIFT = 5;
constexpr uint32_t FORMAT_A     = static_cast<uint32_t>(Prospero::VertexAttribFormat::k8UNorm);
constexpr uint32_t FORMAT_B     = static_cast<uint32_t>(Prospero::VertexAttribFormat::k8SNorm);

// The vertex key a draw with its attribute table at `attrib_address` gets, through the worker's own
// builder with the memory above.
std::optional<GraphicsPrescanSkip> VertexInfoOf(VertexMemory& memory, uint64_t attrib_address,
                                                ShaderVertexInputInfo& info) {
	ScannedDraw draw;
	draw.vs_regs.gs_user_sgpr.value[ATTRIB_REG]     = static_cast<uint32_t>(attrib_address);
	draw.vs_regs.gs_user_sgpr.value[ATTRIB_REG + 1] = static_cast<uint32_t>(attrib_address >> 32u);
	draw.vs_regs.gs_user_sgpr.value[BUFFER_REG]     = static_cast<uint32_t>(BUFFER_ADDRESS);
	draw.vs_regs.gs_user_sgpr.value[BUFFER_REG + 1] = static_cast<uint32_t>(BUFFER_ADDRESS >> 32u);
	ShaderMapSnapshot vertex;
	vertex.input_semantics_address = SEMANTICS_ADDRESS;
	vertex.num_input_semantics     = 1;
	vertex.vertex_attrib_reg       = ATTRIB_REG;
	vertex.vertex_buffer_reg       = BUFFER_REG;
	return ScannedVertexInputInfo(draw, vertex, {16384, 16384}, &VertexMemory::Read, &memory, info);
}

std::vector<uint32_t> VertexKeyOf(VertexMemory& memory, uint64_t attrib_address) {
	ShaderVertexInputInfo info;
	std::vector<uint32_t> key;
	if (!VertexInfoOf(memory, attrib_address, info)) {
		BuildStageStaticKey(info, key);
	}
	return key;
}

// The memory of the key test with one attribute table of each format.
VertexMemory TwoFormatMemory() {
	ShaderSemantic semantic {};
	semantic.semantic         = 0;
	semantic.hardware_mapping = 4;
	semantic.size_in_elements = 4;
	VertexMemory memory;
	memory.regions[SEMANTICS_ADDRESS] = {std::bit_cast<uint32_t>(semantic)};
	memory.regions[ATTRIB_A_ADDRESS]  = {FORMAT_A << FORMAT_SHIFT};
	memory.regions[ATTRIB_B_ADDRESS]  = {FORMAT_B << FORMAT_SHIFT};
	memory.regions[BUFFER_ADDRESS]    = {0x1000u, 0x00100000u, 64u, 0u};
	return memory;
}

// Crash 2 of iteration 3. An attribute whose format has no host format, here format 0 with a buffer
// descriptor of format 0, is rejected by the worker. With the guard gone the info reaches
// VertexInputFormats, which this test then calls, and it EXITs, so the ctest of --guards-only fails.
// The baseline is the same table with a known format, which passes the guard and the formats.
void TestVertexFormatGuard() {
	auto memory = TwoFormatMemory();
	constexpr uint64_t format_zero_address = 0x50000;
	memory.regions[format_zero_address]   = {0u};
	ShaderVertexInputInfo info;
	const auto            skip = VertexInfoOf(memory, format_zero_address, info);
	if (!skip) {
		(void)PrescanVertexInputFormats(info, ShaderRecompiler::IR::CompiledShaderInfo {});
	}
	CHECK(skip == GraphicsPrescanSkip::InvalidVertexFormat);
	ShaderVertexInputInfo known;
	CHECK(!VertexInfoOf(memory, ATTRIB_A_ADDRESS, known));
	CHECK(PrescanVertexInputFormats(known, ShaderRecompiler::IR::CompiledShaderInfo {}).size() == 1);
}

// The worker refuses the vertex data the GPU thread EXITs on, each case against the known format draw
// above, which builds. Metadata refused at map time gives vertex_metadata before any table is read, and
// so do more input semantics than the tables hold. RES_MAX semantics, one fewer, pass that bound and
// reach the read, which fails on the one-semantic memory, so the bound sits exactly past RES_MAX. A
// buffer descriptor asking for ADD_TID fails the vertex table check, which the GPU thread EXITs on, so
// with the worker's failure flag gone this test ends the process and the guards ctest fails.
void TestVertexInfoRefusals() {
	// ADD_TID_ENABLE is bit 23 of a buffer descriptor's fourth word.
	constexpr uint32_t add_tid_enable = 1u << 23u;
	constexpr auto     res_max        = static_cast<uint32_t>(ShaderVertexInputInfo::RES_MAX);
	// The device's largest viewport, as VertexInfoOf passes it.
	constexpr std::array<uint32_t, 2> max_viewport {16384, 16384};
	const auto info_of = [&](VertexMemory& memory, uint32_t semantics, bool vertex_error) {
		ScannedDraw draw;
		draw.vs_regs.gs_user_sgpr.value[ATTRIB_REG]     = static_cast<uint32_t>(ATTRIB_A_ADDRESS);
		draw.vs_regs.gs_user_sgpr.value[ATTRIB_REG + 1] = static_cast<uint32_t>(ATTRIB_A_ADDRESS >> 32u);
		draw.vs_regs.gs_user_sgpr.value[BUFFER_REG]     = static_cast<uint32_t>(BUFFER_ADDRESS);
		draw.vs_regs.gs_user_sgpr.value[BUFFER_REG + 1] = static_cast<uint32_t>(BUFFER_ADDRESS >> 32u);
		ShaderMapSnapshot vertex;
		vertex.input_semantics_address = SEMANTICS_ADDRESS;
		vertex.num_input_semantics     = semantics;
		vertex.vertex_attrib_reg       = ATTRIB_REG;
		vertex.vertex_buffer_reg       = BUFFER_REG;
		vertex.vertex_error            = vertex_error;
		ShaderVertexInputInfo info;
		return ScannedVertexInputInfo(draw, vertex, max_viewport, &VertexMemory::Read, &memory, info);
	};
	auto memory = TwoFormatMemory();
	CHECK(!info_of(memory, 1, false));
	CHECK(info_of(memory, 1, true) == GraphicsPrescanSkip::VertexMetadata);
	CHECK(info_of(memory, res_max + 1, false) == GraphicsPrescanSkip::VertexMetadata);
	CHECK(info_of(memory, res_max, false) == GraphicsPrescanSkip::VertexTables);
	auto add_tid                       = TwoFormatMemory();
	add_tid.regions[BUFFER_ADDRESS][3] = add_tid_enable;
	CHECK(info_of(add_tid, 1, false) == GraphicsPrescanSkip::VertexTables);
}

// Crash 1 of iteration 3. A pixel shader's buffer whose specialized stride carries ADD_TID is rejected.
// With the guard gone the test calls the check info collection makes, which EXITs, so the ctest fails.
// The baseline is the same specialization for compute, which ADD_TID is valid in.
void TestAddTidGuard() {
	namespace IR = ShaderRecompiler::IR;
	IR::ResourcePlan plan;
	plan.info.buffers.emplace_back();
	IR::ResourceSpecialization specialization;
	specialization.buffers.push_back({.packed_stride = IR::BUFFER_ADD_TID_STRIDE_BIT | 16u});
	const auto pixel = ValidateGraphicsSpecialization(ShaderType::Pixel, plan, specialization);
	if (!pixel) {
		IR::RequireBufferAddTidSupported(ShaderType::Pixel, specialization.buffers[0].packed_stride);
	}
	CHECK(pixel == GraphicsPrescanSkip::InvalidBufferDescriptor);
	CHECK(!ValidateGraphicsSpecialization(ShaderType::Compute, plan, specialization));
}

// The memory of a draw whose attribute names no format, so the buffer descriptor's own format and
// selects reach the fetch, with `x_select` as the first component's select. Select 2 is reserved.
constexpr uint64_t RAW_ATTRIB_ADDRESS = 0x60000;
constexpr uint32_t SELECT_RESERVED    = 2;
constexpr uint32_t SELECT_X           = 4;
VertexMemory RawFetchMemory(uint32_t x_select) {
	constexpr uint32_t buffer_format_shift = 12;
	auto memory = TwoFormatMemory();
	memory.regions[RAW_ATTRIB_ADDRESS] = {0u};
	memory.regions[BUFFER_ADDRESS]     = {0x1000u, 0x00100000u, 64u,
	                                      (static_cast<uint32_t>(Prospero::BufferFormat::k32_32_32_32Float) << buffer_format_shift) |
	                                          DstSel(x_select, 5, 6, 7)};
	return memory;
}

// Translates one BUFFER_LOAD_FORMAT_XYZW into VGPR 4 as an embedded fetch of attribute 0, the path
// through TranslateEmbeddedFetch.
void TranslateFetch(const ShaderVertexInputInfo& info) {
	namespace Frontend = ShaderRecompiler::Frontend;
	constexpr uint32_t load_format_xyzw = 3;
	constexpr uint32_t mubuf_encoding   = 0x38u << 26u;
	constexpr uint32_t mubuf_idxen      = 1u << 13u;
	constexpr uint32_t mubuf_opcode_at  = 18;
	constexpr uint32_t vdata_at         = 8;
	constexpr uint32_t soffset_none     = 128u << 24u;
	constexpr uint32_t s_endpgm         = 0x80000000u | (0x7fu << 23u) | (0x01u << 16u);
	constexpr uint32_t fetch_components = 4;
	const uint32_t code[] = {mubuf_encoding | mubuf_idxen | (load_format_xyzw << mubuf_opcode_at),
	                         soffset_none | (4u << vdata_at), s_endpgm};
	ShaderRecompiler::Decoder::Program decoded;
	ShaderRecompiler::Decoder::DecodeProgram(code, decoded);
	Frontend::EmbeddedFetchPlan fetch;
	fetch.loads.push_back({.pc = 0, .attrib_id = 0, .components = fetch_components});
	Frontend::TranslateOptions options {};
	options.stage             = ShaderType::Vertex;
	options.wave_size         = 32;
	options.user_data_count   = 0;
	options.input_info.vertex = &info;
	options.embedded_fetch    = &fetch;
	(void)Frontend::TranslateProgram(decoded, ShaderRecompiler::CFG::BuildGraph(decoded), options);
}

// The fetch translated under the worker's own RunRejectable. False where it was rejected.
bool TranslateFetchRejectable(const ShaderVertexInputInfo& info, GraphicsRejectTally& tally) {
	return RunRejectable(tally, [&] { TranslateFetch(info); });
}

// What the worker does with a draw's vertex fetch: build its info, and translate it rejectably.
std::optional<GraphicsPrescanSkip> WorkerVertexFetch(VertexMemory& memory, GraphicsRejectTally& tally) {
	ShaderVertexInputInfo info;
	if (const auto skip = VertexInfoOf(memory, RAW_ATTRIB_ADDRESS, info)) {
		return skip;
	}
	if (!TranslateFetchRejectable(info, tally)) {
		return GraphicsPrescanSkip::RecompilerRejected;
	}
	return std::nullopt;
}

// Crash of iteration 3b. A buffer descriptor read stale at worker time selects reserved channel 2,
// which TranslateEmbeddedFetch EXITs on. The guard skips the job as invalid_vertex_fetch before it
// translates. With the guard gone the scope still catches the EXIT and the job comes back
// recompiler_rejected, so this check fails without the process ending. With the scope gone too the
// process EXITs. The baseline differs only in the select and translates.
void TestVertexFetchGuard() {
	GraphicsRejectTally tally;
	auto                reserved = RawFetchMemory(SELECT_RESERVED);
	auto                valid    = RawFetchMemory(SELECT_X);
	CHECK(WorkerVertexFetch(reserved, tally) == GraphicsPrescanSkip::InvalidVertexFetch);
	CHECK(!WorkerVertexFetch(valid, tally));
	CHECK(tally.Sites().empty());
}

// A pixel shader's formatted buffer whose specialized descriptor selects reserved channel 2 is rejected
// as invalid_descriptor, the descriptor check the emitter EXITs on. The baseline differs only in the
// select and passes, so the refusal comes from the descriptor and not from the buffer.
void TestInvalidDescriptorGuard() {
	namespace IR = ShaderRecompiler::IR;
	IR::ResourcePlan plan;
	plan.info.buffers.emplace_back().formatted = true;
	IR::ResourceSpecialization reserved;
	reserved.buffers.push_back({.descriptor_format  = Prospero::BufferFormat::k32_32_32_32Float,
	                            .descriptor_swizzle = DstSel(SELECT_RESERVED, 5, 6, 7)});
	IR::ResourceSpecialization valid    = reserved;
	valid.buffers[0].descriptor_swizzle = DstSel(SELECT_X, 5, 6, 7);
	CHECK(ValidateGraphicsSpecialization(ShaderType::Pixel, plan, reserved) == GraphicsPrescanSkip::InvalidDescriptor);
	CHECK(!ValidateGraphicsSpecialization(ShaderType::Pixel, plan, valid));
}

// The scope catches the translator's own EXIT: the same reserved select, translated without the guard
// under RunRejectable, comes back rejected and tallied once at its line in Attribute.cpp, and a second
// rejection at that line adds to its count instead of a new line. The baseline differs only in the
// select and completes. With the scope gone the process EXITs.
void TestRejectScopeCatchesFetchExit() {
	GraphicsRejectTally   tally;
	ShaderVertexInputInfo reserved_info;
	ShaderVertexInputInfo valid_info;
	auto                  reserved = RawFetchMemory(SELECT_RESERVED);
	auto                  valid    = RawFetchMemory(SELECT_X);
	(void)VertexInfoOf(reserved, RAW_ATTRIB_ADDRESS, reserved_info);
	CHECK(!VertexInfoOf(valid, RAW_ATTRIB_ADDRESS, valid_info));
	CHECK(TranslateFetchRejectable(valid_info, tally));
	CHECK(!TranslateFetchRejectable(reserved_info, tally));
	CHECK(!TranslateFetchRejectable(reserved_info, tally));
	const auto sites = tally.Sites();
	CHECK(sites.size() == 1);
	CHECK(!sites.empty() && sites[0].where.find("Attribute.cpp") != std::string::npos && sites[0].count == 2);
}

// The tally keeps each source line apart, in the order first caught, with its first message. A second
// line gets its own site rather than adding to the first, the baseline being a repeat of the first
// line, which adds to its count and keeps its first message.
void TestRejectTallyKeepsLines() {
	constexpr int       first_line  = 10;
	constexpr int       second_line = 20;
	GraphicsRejectTally tally;
	tally.Note("First.cpp", first_line, "first");
	tally.Note("Second.cpp", second_line, "second");
	tally.Note("First.cpp", first_line, "first again");
	const auto sites = tally.Sites();
	CHECK(sites.size() == 2);
	CHECK(sites.size() == 2 && sites[0].where == "First.cpp:10" && sites[0].count == 2 && sites[0].text == "first");
	CHECK(sites.size() == 2 && sites[1].where == "Second.cpp:20" && sites[1].count == 1 && sites[1].text == "second");
}

// The death test. A scope that has ended leaves EXIT as it was, so the reserved select translated
// outside any scope ends the process with EXIT's code. Its ctest is marked WILL_FAIL. A throw that
// escapes, or a translation that returns, ends with 0 instead, which that ctest reports as a failure.
int ExitOutsideScope() {
	{ const Common::ExitRejectScope ended; }
	auto                  reserved = RawFetchMemory(SELECT_RESERVED);
	ShaderVertexInputInfo info;
	(void)VertexInfoOf(reserved, RAW_ATTRIB_ADDRESS, info);
	try {
		TranslateFetch(info);
	} catch (const Common::ExitRejected&) {
		std::puts("EXIT threw outside any scope");
		return 0;
	}
	std::puts("EXIT did not happen");
	return 0;
}

// The most of three jobs on three workers that ran at once, each through a GraphicsJobSlot when
// `slotted`. A job holds on until all three are inside or a deadline passes, so a third job let in
// beside the first two is seen inside with them.
uint32_t PeakOfThreeJobs(bool slotted, GraphicsPrescanCounters& counters) {
	constexpr uint32_t job_count = 3;
	constexpr auto     hold_for  = std::chrono::milliseconds(300);
	std::atomic<uint32_t> running {0};
	std::atomic<uint32_t> inside {0};
	std::atomic<uint32_t> entered {0};
	std::atomic<uint32_t> peak {0};
	PipelineBuildQueue    queue(job_count);
	const auto            job = [&] {
		std::optional<GraphicsJobSlot> slot;
		if (slotted) {
			slot.emplace(running, counters, queue);
		}
		const uint32_t now = inside.fetch_add(1) + 1;
		uint32_t       seen = peak.load();
		while (seen < now && !peak.compare_exchange_weak(seen, now)) {
		}
		entered.fetch_add(1);
		const auto deadline = std::chrono::steady_clock::now() + hold_for;
		while (entered.load() < job_count && std::chrono::steady_clock::now() < deadline) {
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		inside.fetch_sub(1);
	};
	std::vector<PipelineBuildQueue::JobPtr> jobs;
	for (uint32_t index = 0; index < job_count; index++) {
		jobs.push_back(queue.Post(job));
	}
	for (const auto& posted: jobs) {
		(void)queue.Acquire(*posted);
	}
	CHECK(entered.load() == job_count);
	return peak.load();
}

// Three graphics jobs queued at once run at most two at a time, and the third waits for a slot. The
// baseline runs the same three jobs without slots and sees all three at once, so the harness can see
// a third job and the limit comes from the slots.
void TestRunningJobsCapped() {
	GraphicsPrescanCounters capped;
	GraphicsPrescanCounters free;
	CHECK(PeakOfThreeJobs(true, capped) == 2);
	CHECK(capped.capped_waits.load() >= 1);
	CHECK(PeakOfThreeJobs(false, free) == 3);
	// The slots were held for at least the 300 ms the first two jobs wait, all of it counted as busy
	// time, and none is still open. The jobs without slots count none.
	constexpr uint64_t held_at_least_ns = 250'000'000;
	CHECK(capped.busy_ns >= held_at_least_ns && capped.busy_jobs == 0);
	CHECK(GraphicsBusyFraction(capped) > 0.0 && GraphicsBusyFraction(capped) <= 1.0);
	CHECK(free.busy_ns == 0 && GraphicsBusyFraction(free) == 0.0);
}

// Two draws whose attribute tables differ only in one attribute's format give two vertex keys, and so
// two module ids. The baseline reads the same table twice and gives one key, so the difference comes
// from the format. The scanner does not offer the second draw, since its key leaves out the table
// pointers (iteration 3e), so this split is made only by the worker, which reads the tables.
void TestAttributeFormatSplitsVertexKeys() {
	ShaderSemantic semantic {};
	semantic.semantic         = 0;
	semantic.hardware_mapping = 4;
	semantic.size_in_elements = 4;
	VertexMemory memory;
	memory.regions[SEMANTICS_ADDRESS] = {std::bit_cast<uint32_t>(semantic)};
	memory.regions[ATTRIB_A_ADDRESS]  = {FORMAT_A << FORMAT_SHIFT};
	memory.regions[ATTRIB_B_ADDRESS]  = {FORMAT_B << FORMAT_SHIFT};
	memory.regions[BUFFER_ADDRESS]    = {0x1000u, 0x00100000u, 64u, 0u};
	const auto key_a  = VertexKeyOf(memory, ATTRIB_A_ADDRESS);
	const auto key_a2 = VertexKeyOf(memory, ATTRIB_A_ADDRESS);
	const auto key_b  = VertexKeyOf(memory, ATTRIB_B_ADDRESS);
	CHECK(!key_a.empty() && !key_b.empty());
	CHECK(key_a == key_a2);
	CHECK(key_a != key_b);
}

// The vertex info takes the rest of its key from the draw, as the GPU thread does: wave32 from
// VGT_SHADER_STAGES_EN bit 22, LDS_SIZE in granules of 128 dwords, and with CLIP_DISABLE the clip space
// from the first viewport, its half extent half the device's largest viewport capped at the guest's
// 16384. User clip planes pass through only without CLIP_DISABLE. The baseline draw sets none of these
// and gets wave 64, no LDS, no clip space and no planes, so each field comes from its register.
void TestVertexInfoCarriesDrawState() {
	constexpr uint32_t                wave32_stage_bit = 0x00400000u;
	constexpr uint32_t                lds_granules     = 3;
	constexpr uint32_t                lds_dwords       = lds_granules * 128;
	constexpr uint32_t                clip_planes      = 0x5u;
	constexpr std::array<uint32_t, 2> device_viewport {32768, 8192};
	const auto                        info_of = [&](const ScannedDraw& draw) {
		VertexMemory          memory;
		ShaderVertexInputInfo info;
		CHECK(!ScannedVertexInputInfo(draw, ShaderMapSnapshot {}, device_viewport, &VertexMemory::Read, &memory, info));
		return info;
	};
	const auto plain = info_of(ScannedDraw {});
	CHECK(plain.wave_size == 64 && plain.lds_size_dwords == 0 && !plain.clip_space.enabled &&
	      plain.user_clip_planes == 0);

	ScannedDraw clipped;
	clipped.shader_stages             = wave32_stage_bit;
	clipped.gs_lds_size               = lds_granules;
	clipped.clip_control.clip_disable = true;
	clipped.viewport_scale            = {2.0f, 3.0f};
	clipped.viewport_offset           = {4.0f, 5.0f};
	clipped.user_clip_planes          = clip_planes;
	const auto  state                 = info_of(clipped);
	const auto& clip                  = state.clip_space;
	CHECK(state.wave_size == 32 && state.lds_size_dwords == lds_dwords && state.user_clip_planes == 0);
	CHECK(clip.enabled && clip.scale[0] == 2.0f && clip.scale[1] == 3.0f && clip.offset[0] == 4.0f &&
	      clip.offset[1] == 5.0f && clip.half_extent[0] == 8192.0f && clip.half_extent[1] == 4096.0f);

	ScannedDraw planes;
	planes.user_clip_planes = clip_planes;
	const auto planed       = info_of(planes);
	CHECK(planed.user_clip_planes == clip_planes && !planed.clip_space.enabled);
}

// The pixel and vertex halves offered by a walk, each counted over its draws.
struct HalfOffers {
	size_t ps = 0;
	size_t vs = 0;
};

HalfOffers OffersOf(const GraphicsResult& result) {
	HalfOffers offers;
	for (const auto& draw: result.draws) {
		offers.ps += draw.offer_ps ? 1 : 0;
		offers.vs += draw.offer_vs ? 1 : 0;
	}
	return offers;
}

// Two draws of one pixel shader and one vertex program that differ in the vertex user data, here the
// attribute table pointer, give one pixel offer and one vertex offer. The baseline differs only in
// the second draw's vertex program address and gives one pixel offer and two vertex offers, so the
// walk does offer a second vertex half, and it is the user data that the vertex key leaves out.
void TestVertexHalfIgnoresUserData() {
	constexpr uint64_t vs_address   = 0x0000004500001000ull;
	constexpr uint64_t vs_address_2 = 0x0000004500002000ull;
	const auto         set_vs       = [](uint64_t address) {
		return SetSh(Pm4::SPI_SHADER_PGM_LO_ES,
		             {static_cast<uint32_t>(address >> PGM_LO_SHIFT), static_cast<uint32_t>(address >> PGM_HI_SHIFT)});
	};
	const auto set_table = [](uint64_t address) {
		return SetSh(Pm4::SPI_SHADER_USER_DATA_GS_0, {static_cast<uint32_t>(address), static_cast<uint32_t>(address >> 32u)});
	};
	const auto two_draws = [&](uint64_t second_vs) {
		// The vertex stage reads four user registers, the two table pointers.
		constexpr uint32_t vertex_user_registers = 4;
		Stream stream = PixelSetup();
		Append(stream, SetSh(Pm4::SPI_SHADER_PGM_RSRC2_GS, {vertex_user_registers << Pm4::SPI_SHADER_PGM_RSRC2_GS_USER_SGPR_SHIFT}));
		Append(stream, set_vs(vs_address));
		Append(stream, set_table(ATTRIB_A_ADDRESS));
		Append(stream, DrawAuto());
		Append(stream, set_vs(second_vs));
		Append(stream, set_table(ATTRIB_B_ADDRESS));
		Append(stream, DrawAuto());
		return OffersOf(Walk(stream));
	};
	const auto same_vs = two_draws(vs_address);
	CHECK(same_vs.ps == 1);
	CHECK(same_vs.vs == 1);
	const auto other_vs = two_draws(vs_address_2);
	CHECK(other_vs.ps == 1);
	CHECK(other_vs.vs == 2);
}

// One vertex program drawn after one pixel shader and then after another gives two vertex offers,
// one per pairing, since the vertex shader's push start follows the pixel shader's. The baseline
// draws the same vertex program twice after one pixel shader and gives one vertex offer, so the
// second offer comes from the pixel shader alone. Each pixel shader is offered once either way.
void TestVertexHalfPairsWithPixel() {
	constexpr uint64_t ps_address_2 = PS_ADDRESS + 0x1000;
	const auto         set_ps       = [](uint64_t address) {
		return SetSh(Pm4::SPI_SHADER_PGM_LO_PS,
		             {static_cast<uint32_t>(address >> PGM_LO_SHIFT), static_cast<uint32_t>(address >> PGM_HI_SHIFT)});
	};
	const auto two_draws = [&](uint64_t second_ps) {
		Stream stream = PixelSetup();
		Append(stream, DrawAuto());
		Append(stream, set_ps(second_ps));
		Append(stream, DrawAuto());
		return OffersOf(Walk(stream));
	};
	const auto paired = two_draws(ps_address_2);
	CHECK(paired.ps == 2);
	CHECK(paired.vs == 2);
	const auto same = two_draws(PS_ADDRESS);
	CHECK(same.ps == 1);
	CHECK(same.vs == 1);
}

// A busy interval from 0.5 s to 2.25 s after the epoch is split at the second boundaries into 500,
// 1000 and 250 ms, each second marked as having a slot holder, and the timeline group ends at the
// second now. The baseline interval lies inside second 0 and touches nothing else, so the split
// comes from the boundaries.
void TestBusyTimelineSplitsSeconds() {
	constexpr int64_t epoch_ns = 1'000'000'000;
	constexpr int64_t ms_ns    = 1'000'000;
	auto              split    = std::make_unique<GraphicsPrescanCounters>();
	AddBusyToTimeline(*split, epoch_ns, epoch_ns + 500 * ms_ns, epoch_ns + 2250 * ms_ns);
	const auto group = GraphicsBusyTimeline(*split, epoch_ns, epoch_ns + 2900 * ms_ns);
	CHECK(group.name == "busy_timeline" && group.series.size() == 3);
	if (group.series.size() == 3) {
		CHECK((group.series[0].second == std::vector<double> {500.0, 1000.0, 250.0}));
		CHECK((group.series[1].second == std::vector<double> {1.0, 1.0, 1.0}));
	}
	auto inside = std::make_unique<GraphicsPrescanCounters>();
	AddBusyToTimeline(*inside, epoch_ns, epoch_ns + 100 * ms_ns, epoch_ns + 400 * ms_ns);
	CHECK(inside->timeline_busy_us[0] == 300'000 && inside->timeline_busy_us[1] == 0 && inside->timeline_peak[1] == 0);
}

// A slot still held when the timeline is read counts up to now. An interval open from 0.5 s and read at
// 1.5 s gives 500 ms in each of seconds 0 and 1, and reading it leaves the counters as they were. The
// baseline is the same start with no holder, which counts nothing, so the time comes from the holder.
void TestOpenIntervalCountsToNow() {
	constexpr int64_t epoch_ns = 1'000'000'000;
	constexpr int64_t ms_ns    = 1'000'000;
	auto              open     = std::make_unique<GraphicsPrescanCounters>();
	open->busy_jobs            = 1;
	open->busy_since_ns        = epoch_ns + 500 * ms_ns;
	const auto group           = GraphicsBusyTimeline(*open, epoch_ns, epoch_ns + 1500 * ms_ns);
	CHECK(group.series.size() == 3 && (group.series[0].second == std::vector<double> {500.0, 500.0}));
	CHECK(open->timeline_busy_us[0] == 0 && open->timeline_busy_us[1] == 0);
	auto closed           = std::make_unique<GraphicsPrescanCounters>();
	closed->busy_since_ns = epoch_ns + 500 * ms_ns;
	const auto none       = GraphicsBusyTimeline(*closed, epoch_ns, epoch_ns + 1500 * ms_ns);
	CHECK(none.series.size() == 3 && (none.series[0].second == std::vector<double> {0.0, 0.0}));
}

// Two slots held together with the report's epoch set mark their second with two holders, and add
// their time to it when let go. The baseline holds the same two slots with no epoch, which leaves the
// timeline empty while the busy time still counts, so the timeline comes from the slots and the epoch.
// Two holders, since a release alone marks a second with one. The epoch is put back after.
void TestSlotsFillTheTimeline() {
	constexpr auto     hold_for = std::chrono::milliseconds(5);
	constexpr uint32_t held_us  = 4'000;
	auto&              report   = Common::FirstRunReport::Global();
	const auto         saved    = report.EpochNs();
	PipelineBuildQueue queue(1);
	const auto         hold_two = [&](GraphicsPrescanCounters& counters) {
		std::atomic<uint32_t> running {0};
		const GraphicsJobSlot first(running, counters, queue);
		const GraphicsJobSlot second(running, counters, queue);
		std::this_thread::sleep_for(hold_for);
	};
	auto timed = std::make_unique<GraphicsPrescanCounters>();
	report.SetEpoch(Common::FirstRunReport::NowNs());
	hold_two(*timed);
	report.SetEpoch(saved);
	CHECK(timed->timeline_peak[0] == 2 && timed->timeline_busy_us[0] >= held_us);
	auto untimed = std::make_unique<GraphicsPrescanCounters>();
	hold_two(*untimed);
	CHECK(untimed->timeline_peak[0] == 0 && untimed->timeline_busy_us[0] == 0 && untimed->busy_ns != 0);
}

// The adaptive cap on an injected clock. A signal at 10 s drops the cap to 1 until 13 s. A second
// signal at 12 s re-arms the whole window, so the cap is still 1 at 14.9 s and back to 2 at 15 s, with
// 5 s of reduced time in two triggers, split across the timeline's seconds 10 to 14. The baseline has
// no signal and stays at 2, so the drops come from the signals.
void TestAdaptiveCapRearms() {
	constexpr int64_t s_ns  = 1'000'000'000;
	constexpr int64_t ms_ns = 1'000'000;
	GraphicsCapGovernor once;
	CHECK(once.CapAt(10 * s_ns, 10 * s_ns, 0) == 1);
	CHECK(once.CapAt(12900 * ms_ns, 10 * s_ns, 0) == 1);
	CHECK(once.CapAt(13 * s_ns, 10 * s_ns, 0) == 2);
	CHECK(once.Triggers() == 1 && once.ReducedMs(20 * s_ns) == 3000.0);

	GraphicsCapGovernor twice;
	CHECK(twice.CapAt(10 * s_ns, 10 * s_ns, 0) == 1);
	CHECK(twice.CapAt(12 * s_ns, 12 * s_ns, 0) == 1);
	CHECK(twice.CapAt(14900 * ms_ns, 12 * s_ns, 0) == 1);
	CHECK(twice.CapAt(15 * s_ns, 12 * s_ns, 0) == 2);
	CHECK(twice.Triggers() == 2 && twice.ReducedMs(20 * s_ns) == 5000.0);
	// Half way through the second window, the part still ahead is not counted yet.
	CHECK(twice.ReducedMs(13 * s_ns) == 3000.0);
	const auto timeline = twice.ReducedTimelineUs();
	CHECK(timeline[9] == 0 && timeline[10] == 1'000'000 && timeline[14] == 1'000'000 && timeline[15] == 0);

	GraphicsCapGovernor quiet;
	CHECK(quiet.CapAt(10 * s_ns, 0, 0) == 2 && quiet.CapAt(14 * s_ns, 0, 0) == 2);
	CHECK(quiet.Triggers() == 0 && quiet.ReducedMs(20 * s_ns) == 0.0);
}

// A throttle signal while a job holds a slot leaves that job running, and a second job asking for a
// slot then waits, since the window allows one running job, and is counted as a paused start. It is
// let go by stopping the queue rather than by waiting out the 3 s. The baseline holds one slot with no
// signal and gets the second at once. Runs after every other slot test, since the signal stays for
// 3 s and a fresh governor would see it.
void TestWindowPausesNewStarts() {
	constexpr auto     settle = std::chrono::milliseconds(200);
	PipelineBuildQueue queue(1);
	{
		auto                  quiet = std::make_unique<GraphicsPrescanCounters>();
		std::atomic<uint32_t> running {0};
		const GraphicsJobSlot first(running, *quiet, queue);
		const GraphicsJobSlot second(running, *quiet, queue);
		CHECK(running.load() == 2 && quiet->paused_starts.load() == 0);
	}
	auto                  counters = std::make_unique<GraphicsPrescanCounters>();
	std::atomic<uint32_t> running {0};
	const GraphicsJobSlot held(running, *counters, queue);
	Common::GpuStall::NoteAudioLowWater();
	std::atomic<bool> started {false};
	std::thread       second([&] {
		const GraphicsJobSlot slot(running, *counters, queue);
		started = running.load() == 2;
	});
	std::this_thread::sleep_for(settle);
	CHECK(running.load() == 1 && !started.load());
	(void)queue.Stop();
	second.join();
	CHECK(!started.load() && counters->paused_starts.load() == 1 && running.load() == 1);
}

// 600 jobs posted to a paused queue: the first GRAPHICS_PRESCAN_QUEUE_CAP are queued and the other
// 88 dropped as full. The baseline posts 512 to another paused queue and drops none, so the drops
// begin at the cap and not before.
void TestQueueCapDropsPastLimit() {
	constexpr uint32_t posts     = 600;
	const auto         post_many = [](uint32_t count, GraphicsQueueBound& bound) {
		PipelineBuildQueue queue(1);
		queue.Pause(PipelineBuildQueue::Clock::now() + PipelineBuildQueue::MAX_PAUSE);
		uint32_t queued = 0;
		for (uint32_t index = 0; index < count; index++) {
			const auto lane = index % 2 == 0 ? PipelineBuildQueue::Lane::Front : PipelineBuildQueue::Lane::Back;
			queued += PostBounded(queue, bound, lane, [] {}) ? 1 : 0;
		}
		(void)queue.Stop();
		return queued;
	};
	GraphicsQueueBound over;
	CHECK(post_many(posts, over) == 512);
	CHECK(over.dropped_full.load() == 88);
	CHECK(over.depth_max[0].load() + over.depth_max[1].load() == 512);
	GraphicsQueueBound at_cap;
	CHECK(post_many(512, at_cap) == 512);
	CHECK(at_cap.dropped_full.load() == 0);
}

// Whether a sighting's outcome is the named one. A null outcome is none.
bool OutcomeIs(const char* outcome, std::string_view expected) {
	return outcome != nullptr && std::string_view(outcome) == expected;
}

// The bit of each verdict, shorter to read in the checks.
constexpr uint32_t RECORDED_BIT     = ScanVerdictBit(ScanVerdict::Recorded);
constexpr uint32_t CACHED_BIT       = ScanVerdictBit(ScanVerdict::Cached);
constexpr uint32_t NO_PIXEL_BIT     = ScanVerdictBit(ScanVerdict::NoPixelStage);
constexpr uint32_t PS_UNWRITTEN_BIT = ScanVerdictBit(ScanVerdict::PsUnwritten);
constexpr uint32_t UNSUPPORTED_BIT  = ScanVerdictBit(ScanVerdict::UnsupportedInputs);

// The sightings keep each stage of an address apart. Scan verdicts add up, so a shader that met
// several keeps every one, where a single verdict alone, the baseline, is one bit. Names come out in
// verdict order with recorded first. A job verdict replaces the job and a repeat does not, the verdict
// being the baseline that differs only in the kind, while a repeat still fills an empty job. Queued
// stamps the first offer and Posted the last post. Up to the cap every address is kept, the baseline,
// and one more new address is dropped and counted while a kept one still updates.
void TestSightingsTable() {
	constexpr uint64_t      address = 0x1000;
	GraphicsShaderSightings sightings;
	sightings.NoteScan(SightingStage::Pixel, address, ScanVerdict::Recorded, 10);
	auto seen = sightings.Find(SightingStage::Pixel, address);
	CHECK(seen && seen->scan_verdicts == RECORDED_BIT && seen->first_seen_ns == 10 && seen->job == nullptr);
	sightings.NoteScan(SightingStage::Pixel, address, ScanVerdict::PsUnwritten, 30);
	seen = sightings.Find(SightingStage::Pixel, address);
	CHECK(seen && seen->scan_verdicts == (RECORDED_BIT | PS_UNWRITTEN_BIT) && seen->first_seen_ns == 10);
	CHECK(ScanVerdictNames(UNSUPPORTED_BIT | RECORDED_BIT | NO_PIXEL_BIT) ==
	      std::vector<std::string>({"recorded", "no_pixel_stage", "unsupported_inputs"}));
	sightings.NoteJob(SightingStage::Pixel, address, "deduped", SightingKind::Repeat, 40);
	seen = sightings.Find(SightingStage::Pixel, address);
	CHECK(seen && OutcomeIs(seen->job, "deduped"));
	for (const int64_t at: {50, 60}) {
		sightings.NoteJob(SightingStage::Pixel, address, "queued", SightingKind::Queued, at);
	}
	sightings.NoteJob(SightingStage::Pixel, address, "deduped", SightingKind::Repeat, 65);
	seen = sightings.Find(SightingStage::Pixel, address);
	CHECK(seen && OutcomeIs(seen->job, "queued") && seen->offered_ns == 50);
	for (const int64_t at: {70, 80}) {
		sightings.NoteJob(SightingStage::Pixel, address, "posted", SightingKind::Posted, at);
	}
	seen = sightings.Find(SightingStage::Pixel, address);
	CHECK(seen && OutcomeIs(seen->job, "posted") && seen->offered_ns == 50 && seen->posted_ns == 80);
	CHECK(!sightings.Find(SightingStage::Vertex, address));
	sightings.NoteScan(SightingStage::Vertex, address, ScanVerdict::NoPixelStage, 90);
	const auto vertex = sightings.Find(SightingStage::Vertex, address);
	seen              = sightings.Find(SightingStage::Pixel, address);
	CHECK(vertex && vertex->scan_verdicts == NO_PIXEL_BIT && seen &&
	      seen->scan_verdicts == (RECORDED_BIT | PS_UNWRITTEN_BIT));
	sightings.NoteScan(SightingStage::Pixel, 0, ScanVerdict::Recorded, 0);
	CHECK(!sightings.Find(SightingStage::Pixel, 0) && sightings.Kept() == 2);

	GraphicsShaderSightings full;
	for (uint64_t index = 1; index <= GRAPHICS_SIGHTING_CAP; index++) {
		full.NoteScan(SightingStage::Pixel, index, ScanVerdict::Recorded, 0);
	}
	CHECK(full.Kept() == GRAPHICS_SIGHTING_CAP && full.Dropped() == 0);
	full.NoteJob(SightingStage::Pixel, GRAPHICS_SIGHTING_CAP + 1, "queued", SightingKind::Queued, 0);
	CHECK(full.Kept() == GRAPHICS_SIGHTING_CAP && full.Dropped() == 1 &&
	      !full.Find(SightingStage::Pixel, GRAPHICS_SIGHTING_CAP + 1));
	full.NoteScan(SightingStage::Pixel, 1, ScanVerdict::UnsupportedInputs, 0);
	const auto kept = full.Find(SightingStage::Pixel, 1);
	CHECK(kept && kept->scan_verdicts == (RECORDED_BIT | UNSUPPORTED_BIT) && full.Dropped() == 1);
}

// The vertex shader address of the walk's sighting tests, apart from PS_ADDRESS.
constexpr uint64_t VS_ADDRESS = 0x0000004567890200ull;
// A second pair for the draw with no pixel stage, so its sightings are never another draw's.
constexpr uint64_t QUIET_PS_ADDRESS = 0x0000004567890300ull;
constexpr uint64_t QUIET_VS_ADDRESS = 0x0000004567890400ull;

// The writes that make `address` the vertex shader's program.
Stream VertexAddress(uint64_t address) {
	return SetSh(Pm4::SPI_SHADER_PGM_LO_ES,
	             {static_cast<uint32_t>(address >> PGM_LO_SHIFT), static_cast<uint32_t>(address >> PGM_HI_SHIFT)});
}

// Walks the stream with a fresh shadow that notes into `sightings`.
GraphicsResult WalkNoting(GraphicsShaderSightings& sightings, const Stream& commands) {
	GraphicsScanShadow shadow;
	shadow.SetSightings(&sightings);
	return WalkWith(shadow, commands);
}

// The walk notes its verdict on each draw for the draw's shader addresses. A draw it records is noted
// as recorded for both stages, and the same draw again, which its cache skips, adds cached. The same
// setup refused for its inputs, the baseline that differs only in the refusing register, is noted as
// unsupported_inputs alone, so the verdict comes from the draw and not from the address writes. A draw
// whose pixel shader does not run notes only its vertex shader, though a pixel shader address was
// written, while the recorded draw shows a written address is noted when its shader runs.
void TestWalkNotesSightings() {
	Stream recorded = VertexAddress(VS_ADDRESS);
	Append(recorded, PixelSetup());
	Append(recorded, DrawAuto());
	Append(recorded, DrawAuto());
	GraphicsShaderSightings noted;
	// The second draw was met and skipped by the cache, so its verdict was really offered.
	CHECK(WalkNoting(noted, recorded).stats.cached == 1);
	const auto ps = noted.Find(SightingStage::Pixel, PS_ADDRESS);
	const auto vs = noted.Find(SightingStage::Vertex, VS_ADDRESS);
	CHECK(ps && ps->scan_verdicts == (RECORDED_BIT | CACHED_BIT) && ps->job == nullptr);
	CHECK(vs && vs->scan_verdicts == (RECORDED_BIT | CACHED_BIT));

	Stream refused = VertexAddress(VS_ADDRESS);
	Append(refused, PixelSetup());
	Append(refused, SetContext(Pm4::SPI_PS_INPUT_ADDR, {PERSP_CENTER_INPUT | 1u}));
	Append(refused, DrawAuto());
	GraphicsShaderSightings refusals;
	(void)WalkNoting(refusals, refused);
	const auto refused_ps = refusals.Find(SightingStage::Pixel, PS_ADDRESS);
	CHECK(refused_ps && refused_ps->scan_verdicts == UNSUPPORTED_BIT);

	Stream quiet = VertexAddress(QUIET_VS_ADDRESS);
	Append(quiet, SetSh(Pm4::SPI_SHADER_PGM_LO_PS, {static_cast<uint32_t>(QUIET_PS_ADDRESS >> PGM_LO_SHIFT),
	                                                static_cast<uint32_t>(QUIET_PS_ADDRESS >> PGM_HI_SHIFT)}));
	Append(quiet, DrawAuto());
	GraphicsShaderSightings no_pixel;
	CHECK(WalkNoting(no_pixel, quiet).stats.no_pixel_stage == 1);
	const auto quiet_vs = no_pixel.Find(SightingStage::Vertex, QUIET_VS_ADDRESS);
	CHECK(quiet_vs && quiet_vs->scan_verdicts == NO_PIXEL_BIT);
	CHECK(!no_pixel.Find(SightingStage::Pixel, QUIET_PS_ADDRESS));
}

// One walk draws the same shaders three ways, recorded, refused for their inputs and with no pixel
// stage, as a vertex shader shared by colour and depth-only passes is. Every verdict stays on its
// address, so a recorded vertex shader never reads as no_pixel_stage alone, and the pixel shader gets
// no no_pixel_stage. The walk's own cache of noted verdicts lets each new verdict through, which the
// separate walks above, one verdict each, are the baseline for.
void TestWalkKeepsEveryVerdict() {
	Stream mixed = VertexAddress(VS_ADDRESS);
	Append(mixed, PixelSetup());
	Append(mixed, DrawAuto());
	Append(mixed, SetContext(Pm4::SPI_PS_INPUT_ADDR, {PERSP_CENTER_INPUT | 1u}));
	Append(mixed, DrawAuto());
	Append(mixed, SetContext(Pm4::DB_SHADER_CONTROL, {0u}));
	Append(mixed, DrawAuto());
	GraphicsShaderSightings sightings;
	const auto              walked = WalkNoting(sightings, mixed);
	CHECK(walked.stats.recorded == 1 && walked.stats.unsupported_inputs == 1 && walked.stats.no_pixel_stage == 1);
	const auto vs = sightings.Find(SightingStage::Vertex, VS_ADDRESS);
	const auto ps = sightings.Find(SightingStage::Pixel, PS_ADDRESS);
	CHECK(vs && vs->scan_verdicts == (RECORDED_BIT | UNSUPPORTED_BIT | NO_PIXEL_BIT));
	CHECK(ps && ps->scan_verdicts == (RECORDED_BIT | UNSUPPORTED_BIT));
}

// A walk without a graphics shadow reports what it did before there was one. The same stream walked
// with one finds the draw, so the stream does reach the graphics half.
void TestNoGraphicsLeavesWalkAlone() {
	Stream stream = PixelSetup();
	Append(stream, DrawAuto());
	CsShadow                     compute;
	std::vector<ScannedDispatch> dispatches;
	ScanStats                    stats;
	ScanScratch                  scratch;
	SubmitScanner(NoMemory()).Scan(stream, 0, false, compute, dispatches, stats, scratch, nullptr, nullptr);
	CHECK(dispatches.empty());
	CHECK(stats.submissions == 1);
	CHECK(Walk(stream).draws.size() == 1);
}

// The NOP the processor's reset arrives as, which submitScan.cpp names DISPATCH_RESET_PACKET. Its count
// field makes it two dwords long.
constexpr uint32_t RESET_PACKET = 0xC0001024u;
// The type 2 filler word, which the processor steps over as one dword.
constexpr uint32_t FILLER_WORD = 0x80000000u;
// Bit 0 of a type 3 header, which marks a packet the processor may skip under a predicate.
constexpr uint32_t PREDICATE_BIT = 1;
// BRANCH mode 1 runs the then buffer or nothing, and mode 2 the then buffer or the else buffer.
constexpr uint32_t BRANCH_THEN_ONLY    = 1;
constexpr uint32_t BRANCH_THEN_OR_ELSE = 2;
// Packet lengths in dwords, the header included.
constexpr uint32_t INDIRECT_BUFFER_LEN  = 4;
constexpr uint32_t BRANCH_LEN           = 14;
constexpr uint32_t COND_EXEC_LEN        = 5;
constexpr uint32_t SET_REG_INDIRECT_LEN = 5;
constexpr uint32_t UCONFIG_REG_LEN      = 3;
constexpr uint32_t CLEAR_STATE_LEN      = 2;
constexpr uint32_t DRAW_PACKET_LEN      = 5;
// The guest address of the value a COND_EXEC tests, which the walk never reads.
constexpr uint32_t COND_EXEC_TESTED_ADDRESS = 0x1000u;
// The INDEX field of SET_UCONFIG_REG_INDEX, bits 28 to 31, here with bit 31 set. The indexed form's
// 28-bit offset mask drops it, where the plain form's selector mask, bits 28 to 30, would keep it.
constexpr uint32_t UCONFIG_INDEX_FIELD = 0x8u << 28u;
// A compute program address, for writes of compute state only.
constexpr uint64_t CS_ADDRESS = 0x0000004567890500ull;
// Guest addresses of the buffers the memory walks read, and one that nothing backs.
constexpr uint64_t ARM_ADDRESS        = 0x70000;
constexpr uint64_t COPY_ADDRESS       = 0x71000;
constexpr uint64_t OTHER_COPY_ADDRESS = 0x72000;
constexpr uint64_t LIST_ADDRESS       = 0x73000;
constexpr uint64_t ELSE_ARM_ADDRESS   = 0x74000;
constexpr uint64_t UNREADABLE_ADDRESS = 0xdead0000ull;

// Guest memory as regions of dwords by address.
using Memory = std::map<uint64_t, Stream>;

Stream IndirectBuffer(uint64_t address, uint32_t dwords) {
	return {KYTY_PM4(INDIRECT_BUFFER_LEN, Pm4::IT_INDIRECT_BUFFER, 0), static_cast<uint32_t>(address),
	        static_cast<uint32_t>(address >> 32u), dwords};
}

// A BRANCH of `mode` with its then and else buffers. The six compare words are ignored by the walk.
Stream Branch(uint32_t mode, uint64_t then_address, uint32_t then_dwords, uint64_t else_address,
              uint32_t else_dwords) {
	return {KYTY_PM4(BRANCH_LEN, Pm4::IT_INDIRECT_BUFFER, 0), mode, 0u, 0u, 0u, 0u, 0u, 0u,
	        static_cast<uint32_t>(then_address), static_cast<uint32_t>(then_address >> 32u), then_dwords,
	        static_cast<uint32_t>(else_address), static_cast<uint32_t>(else_address >> 32u), else_dwords};
}

// A COND_EXEC that may skip the `skipped` dwords after it.
Stream CondExec(uint32_t skipped) {
	return {KYTY_PM4(COND_EXEC_LEN, Pm4::IT_COND_EXEC, 0), COND_EXEC_TESTED_ADDRESS, 0u, 0u, skipped};
}

// An indirect register list packet of `opcode` whose `pairs` offset and value pairs are at `address`.
Stream IndirectList(uint32_t opcode, uint64_t address, uint32_t pairs) {
	return {KYTY_PM4(SET_REG_INDIRECT_LEN, opcode, 0), static_cast<uint32_t>(address),
	        static_cast<uint32_t>(address >> 32u), 0u, pairs};
}

// The packet with its predicate bit set, so the processor may skip it.
Stream Predicated(Stream packet) {
	packet[0] |= PREDICATE_BIT;
	return packet;
}

// The write of the recognisable interpolator setting.
Stream SettingWrite() {
	return SetContext(Pm4::SPI_PS_INPUT_CNTL_0, {INTERPOLATOR_SETTING});
}

// A write of compute state only, the compute program's address. The compute walk takes it as a key
// write, and the graphics shadow leaves it out as a denied register.
Stream ComputeProgramWrite() {
	return SetSh(Pm4::COMPUTE_PGM_LO, {static_cast<uint32_t>(CS_ADDRESS >> PGM_LO_SHIFT),
	                                   static_cast<uint32_t>(CS_ADDRESS >> PGM_HI_SHIFT)});
}

// A uconfig write of the primitive type, as SET_UCONFIG_REG or, when `indexed`, as SET_UCONFIG_REG_INDEX
// with an INDEX field the plain form's mask would leave in the offset.
Stream PrimitiveTypeWrite(Prospero::PrimitiveType type, bool indexed) {
	const uint32_t opcode = indexed ? Pm4::IT_SET_UCONFIG_REG_INDEX : Pm4::IT_SET_UCONFIG_REG;
	const uint32_t offset = indexed ? UCONFIG_INDEX_FIELD | Pm4::VGT_PRIMITIVE_TYPE : Pm4::VGT_PRIMITIVE_TYPE;
	return {KYTY_PM4(UCONFIG_REG_LEN, opcode, 0), offset, static_cast<uint32_t>(type)};
}

// The writes that make PS_ADDRESS the pixel shader's program again, after a doubt.
Stream PixelAddress() {
	return SetSh(Pm4::SPI_SHADER_PGM_LO_PS, {static_cast<uint32_t>(PS_ADDRESS >> PGM_LO_SHIFT),
	                                         static_cast<uint32_t>(PS_ADDRESS >> PGM_HI_SHIFT)});
}

// Walks the stream as Walk does, with a reader of `memory` in place of NoMemory.
GraphicsResult WalkReading(const Stream& commands, const Memory& memory) {
	const ScanMemoryReader reader = [&memory](uint64_t address, void* destination, uint64_t bytes) {
		for (const auto& [base, words]: memory) {
			const uint64_t size = words.size() * sizeof(uint32_t);
			if (address >= base && address + bytes <= base + size) {
				std::memcpy(destination, reinterpret_cast<const uint8_t*>(words.data()) + (address - base), bytes);
				return true;
			}
		}
		return false;
	};
	GraphicsScanShadow           graphics;
	CsShadow                     compute;
	std::vector<ScannedDispatch> dispatches;
	ScanScratch                  scratch;
	GraphicsResult               result;
	SubmitScanner(reader).Scan(commands, 0, false, compute, dispatches, result.scan, scratch, nullptr, &graphics);
	result.draws = std::exchange(graphics.draws, {});
	result.stats = std::exchange(graphics.stats, {});
	return result;
}

// The interpolator setting of a walk's only draw, or nothing when it did not offer exactly one.
std::optional<uint32_t> OnlyDrawSetting(const GraphicsResult& result) {
	if (result.draws.size() != 1) {
		return std::nullopt;
	}
	return result.draws[0].ps_info.interpolator_settings[0];
}

// How many times the compute walk doubted its own state for `cause`.
uint64_t CauseCount(const GraphicsResult& result, UncertainCause cause) {
	return result.scan.uncertain_cleared_by[static_cast<size_t>(cause)];
}

// The walk of the pixel setup, the setting, `between`, the pixel setup again and a draw. Writing the
// setup again lets the draw through after anything `between` forgets or doubts.
GraphicsResult SettingAfter(const Stream& between) {
	Stream stream = PixelSetup();
	Append(stream, SettingWrite());
	Append(stream, between);
	Append(stream, PixelSetup());
	Append(stream, DrawAuto());
	return Walk(stream);
}

// Every draw opcode of the processor's dispatch table is met as a draw, and only the indirect ones are
// marked, since their vertex shader key carries the packet's offsets. The baseline is a packet of the
// same length that is not a draw, a compute dispatch, which the walk meets and records no draw for.
void TestEveryDrawOpcodeIsADraw() {
	const auto walk_packet = [](uint32_t opcode) {
		Stream stream = PixelSetup();
		Append(stream, {KYTY_PM4(DRAW_PACKET_LEN, opcode, 0), 0u, 0u, 0u, 0u});
		return Walk(stream);
	};
	const std::pair<uint32_t, bool> opcodes[] = {
	    {Pm4::IT_DRAW_INDIRECT, true},        {Pm4::IT_DRAW_INDEX_INDIRECT, true},
	    {Pm4::IT_DRAW_INDIRECT_MULTI, true},  {Pm4::IT_DRAW_INDEX_INDIRECT_MULTI, true},
	    {Pm4::IT_DRAW_INDEX_2, false},        {Pm4::IT_DRAW_INDEX_OFFSET_2, false},
	    {Pm4::IT_DISPATCH_DRAW_PREAMBLE, false}, {Pm4::IT_DRAW_INDEX_AUTO, false}};
	for (const auto& [opcode, indirect]: opcodes) {
		const auto result = walk_packet(opcode);
		CHECK(result.draws.size() == 1 && result.draws[0].indirect == indirect);
	}
	const auto dispatch = walk_packet(Pm4::IT_DISPATCH_DIRECT);
	CHECK(dispatch.scan.dispatches == 1 && dispatch.stats.draws == 0 && dispatch.draws.empty());
}

// A graphics write the processor may skip, under a predicate or inside a COND_EXEC region, puts the
// shadow in doubt, so the draw after it is held back. The same condition around a write of compute
// state only leaves the graphics shadow trusted and its draw offered, though the compute walk doubts
// its own state there, which its cause count shows. Under each condition the two cases differ only in
// what the packet writes, and the unconditional graphics write, the last baseline, is offered.
void TestConditionalWriteDoubts() {
	const auto walk_after = [](const Stream& before, const Stream& write) {
		Stream stream = PixelSetup();
		Append(stream, before);
		Append(stream, write);
		Append(stream, DrawAuto());
		return Walk(stream);
	};
	const auto in_region = [&](const Stream& write) {
		return walk_after(CondExec(static_cast<uint32_t>(write.size())), write);
	};
	const auto predicated_graphics = walk_after({}, Predicated(SettingWrite()));
	const auto region_graphics     = in_region(SettingWrite());
	const auto predicated_compute  = walk_after({}, Predicated(ComputeProgramWrite()));
	const auto region_compute      = in_region(ComputeProgramWrite());
	const auto plain               = walk_after({}, SettingWrite());
	CHECK(predicated_graphics.draws.empty() && predicated_graphics.stats.ps_unwritten == 1);
	CHECK(region_graphics.draws.empty() && region_graphics.stats.ps_unwritten == 1);
	CHECK(predicated_compute.draws.size() == 1 && predicated_compute.stats.doubts == 0 &&
	      CauseCount(predicated_compute, UncertainCause::Predicated) == 1);
	CHECK(region_compute.draws.size() == 1 && region_compute.stats.doubts == 0 &&
	      CauseCount(region_compute, UncertainCause::CondExecRegion) == 1);
	CHECK(plain.draws.size() == 1 && plain.stats.doubts == 0);
}

// A BRANCH then arm, which the processor may not run, is walked from the state the branch met, and
// that state is put back after it. The arm writes the setting and pops a context state with none
// pushed, whose count shows the arm was walked. The pixel shader's address is written again after the
// branch, which leaves the shadow in doubt. The baseline makes the same writes outside any branch and
// its draw carries the setting, so the reset value comes from the restore.
void TestBranchArmStateIsRestored() {
	Stream arm = SettingWrite();
	Append(arm, ContextState(CONTEXT_STATE_POP));
	Stream branched = PixelSetup();
	Append(branched, Branch(BRANCH_THEN_ONLY, ARM_ADDRESS, static_cast<uint32_t>(arm.size()), 0, 0));
	Append(branched, PixelAddress());
	Append(branched, DrawAuto());
	Stream direct = PixelSetup();
	Append(direct, arm);
	Append(direct, PixelAddress());
	Append(direct, DrawAuto());
	const auto restored = WalkReading(branched, {{ARM_ADDRESS, arm}});
	CHECK(restored.stats.context_state_imbalance == 1 && OnlyDrawSetting(restored) == 0u);
	CHECK(OnlyDrawSetting(Walk(direct)) == INTERPOLATOR_SETTING);
}

// A mode 2 BRANCH with an else buffer walks the else arm and then the then arm, each from the state the
// branch met, and chains, so the draw sits in the then arm. The else arm writes the setting and pops a
// context state with none pushed, whose count shows it was walked, and the then arm writes the pixel
// shader's address and draws. The baseline moves the else arm's writes to the start of the then arm,
// leaving a filler word as the else arm, and its draw carries the setting, so the then arm's draw sees
// writes made before it, and the reset value comes from the restore after the else arm.
void TestElseArmStateIsRestored() {
	const auto walk_arms = [](const Stream& else_arm, const Stream& then_arm) {
		Stream stream = PixelSetup();
		Append(stream, Branch(BRANCH_THEN_OR_ELSE, ARM_ADDRESS, static_cast<uint32_t>(then_arm.size()),
		                      ELSE_ARM_ADDRESS, static_cast<uint32_t>(else_arm.size())));
		return WalkReading(stream, {{ARM_ADDRESS, then_arm}, {ELSE_ARM_ADDRESS, else_arm}});
	};
	Stream writes = SettingWrite();
	Append(writes, ContextState(CONTEXT_STATE_POP));
	Stream draw = PixelAddress();
	Append(draw, DrawAuto());
	Stream writes_then_draw = writes;
	Append(writes_then_draw, draw);
	const auto restored = walk_arms(writes, draw);
	CHECK(restored.scan.branch_mode2 == 1 && restored.stats.context_state_imbalance == 1 &&
	      OnlyDrawSetting(restored) == 0u);
	CHECK(OnlyDrawSetting(walk_arms({FILLER_WORD}, writes_then_draw)) == INTERPOLATOR_SETTING);
}

// An indirect buffer that cannot be read may have written any state, so it puts the shadow in doubt
// and the draw after it is held back. The baseline is the same stream without the buffer.
void TestUnreadableBufferDoubts() {
	constexpr uint32_t unread_dwords = 4;
	Stream             unread        = PixelSetup();
	Append(unread, IndirectBuffer(UNREADABLE_ADDRESS, unread_dwords));
	Append(unread, DrawAuto());
	Stream plain = PixelSetup();
	Append(plain, DrawAuto());
	const auto doubted = Walk(unread);
	CHECK(doubted.draws.empty() && doubted.stats.ps_unwritten == 1);
	CHECK(Walk(plain).draws.size() == 1);
}

// A buffer met a second time is not walked again. Its compute writes would be the same, but the walk's
// clean mark does not cover graphics writes, so the skip puts the shadow in doubt. The baseline calls
// two copies of the buffer at two addresses, both walked, and its draw is offered with the setting.
void TestRevisitedBufferDoubts() {
	const Stream   buffer = SettingWrite();
	const auto     size   = static_cast<uint32_t>(buffer.size());
	const Memory   memory {{COPY_ADDRESS, buffer}, {OTHER_COPY_ADDRESS, buffer}};
	Stream         twice  = PixelSetup();
	Append(twice, IndirectBuffer(COPY_ADDRESS, size));
	Append(twice, IndirectBuffer(COPY_ADDRESS, size));
	Append(twice, DrawAuto());
	Stream two_copies = PixelSetup();
	Append(two_copies, IndirectBuffer(COPY_ADDRESS, size));
	Append(two_copies, IndirectBuffer(OTHER_COPY_ADDRESS, size));
	Append(two_copies, DrawAuto());
	const auto revisited = WalkReading(twice, memory);
	CHECK(revisited.scan.ib_revisited == 1 && revisited.draws.empty() && revisited.stats.ps_unwritten == 1);
	CHECK(OnlyDrawSetting(WalkReading(two_copies, memory)) == INTERPOLATOR_SETTING);
}

// A SET_CONTEXT_REG_INDIRECT list in guest memory writes its pairs to the graphics shadow. The
// baseline is the same stream without the list, whose draw has the reset value.
void TestIndirectContextListApplies() {
	constexpr uint32_t one_pair = 1;
	Stream             listed   = PixelSetup();
	Append(listed, IndirectList(Pm4::IT_SET_CONTEXT_REG_INDIRECT, LIST_ADDRESS, one_pair));
	Append(listed, DrawAuto());
	Stream plain = PixelSetup();
	Append(plain, DrawAuto());
	const Memory memory {{LIST_ADDRESS, {Pm4::SPI_PS_INPUT_CNTL_0, INTERPOLATOR_SETTING}}};
	CHECK(OnlyDrawSetting(WalkReading(listed, memory)) == INTERPOLATOR_SETTING);
	CHECK(OnlyDrawSetting(Walk(plain)) == 0u);
}

// The processor's reset forgets every register, not only whether the pixel shader's address was
// written, which a doubt also forgets. The setting written before the reset is gone from the draw after
// it, though the pixel setup is written again, and the compute walk counts the reset, which shows the
// packet was taken as one. The baseline is the same stream without the reset, whose draw carries the
// setting.
void TestResetForgetsState() {
	const auto forgotten = SettingAfter({RESET_PACKET, 0u});
	CHECK(forgotten.scan.resets == 1 && OnlyDrawSetting(forgotten) == 0u);
	CHECK(OnlyDrawSetting(SettingAfter({})) == INTERPOLATOR_SETTING);
}

// CLEAR_STATE resets the context registers, so the setting written before it is gone from the draw
// after it, though the pixel setup is written again. The baseline is SettingAfter's stream without it.
void TestClearStateResetsContext() {
	const auto cleared = SettingAfter({KYTY_PM4(CLEAR_STATE_LEN, Pm4::IT_CLEAR_STATE, 0), 0u});
	CHECK(cleared.stats.draws == 1 && OnlyDrawSetting(cleared) == 0u);
	CHECK(OnlyDrawSetting(SettingAfter({})) == INTERPOLATOR_SETTING);
}

// A uconfig write of the primitive type reaches the draw's class checks, so a rect list draw is skipped
// and counted. The baseline is the same write of a triangle list, whose draw is offered. The indexed
// form reads a 28-bit offset, so its INDEX field, with bit 31 set, still lands on the primitive type.
void TestPrimitiveTypeReachesClassChecks() {
	const auto walk_with = [](Prospero::PrimitiveType type, bool indexed) {
		Stream stream = PixelSetup();
		Append(stream, PrimitiveTypeWrite(type, indexed));
		Append(stream, DrawAuto());
		return Walk(stream);
	};
	for (const bool indexed: {false, true}) {
		const auto rect      = walk_with(Prospero::PrimitiveType::kRectList, indexed);
		const auto triangles = walk_with(Prospero::PrimitiveType::kTriList, indexed);
		CHECK(rect.stats.draws == 1 && rect.draws.empty() && rect.stats.skipped_rect_list == 1);
		CHECK(triangles.draws.size() == 1 && triangles.stats.skipped_rect_list == 0);
	}
}

// A SET_UCONFIG_REG_INDIRECT list in guest memory writes its pairs to the uconfig registers, so a rect
// list from the list skips the draw. The baseline is the same list holding a triangle list.
void TestIndirectUconfigListApplies() {
	constexpr uint32_t one_pair  = 1;
	const auto         walk_with = [](Prospero::PrimitiveType type) {
		Stream stream = PixelSetup();
		Append(stream, IndirectList(Pm4::IT_SET_UCONFIG_REG_INDIRECT, LIST_ADDRESS, one_pair));
		Append(stream, DrawAuto());
		return WalkReading(stream, {{LIST_ADDRESS, {Pm4::VGT_PRIMITIVE_TYPE, static_cast<uint32_t>(type)}}});
	};
	const auto rect = walk_with(Prospero::PrimitiveType::kRectList);
	CHECK(rect.stats.draws == 1 && rect.draws.empty() && rect.stats.skipped_rect_list == 1);
	CHECK(walk_with(Prospero::PrimitiveType::kTriList).draws.size() == 1);
}

// Target 0 as an 11_11_10 UNorm target, the packed UNorm format, with every channel on in both masks
// and written in the packed export mode. CB_COLOR0_BASE holds the address from bit 8.
constexpr uint64_t PACKED_TARGET_ADDRESS = 0x0000001000000000ull;
constexpr uint32_t COLOR_BASE_SHIFT      = 8;
constexpr uint32_t PACKED_TARGET_INFO =
    static_cast<uint32_t>(Prospero::ChannelLayout::k11_11_10) << Pm4::CB_COLOR0_INFO_FORMAT_SHIFT;
constexpr uint32_t TARGET0_CHANNELS    = 0xfu;
constexpr uint32_t PACKED_UNORM_EXPORT = 9;
constexpr uint32_t BLEND_ENABLE        = 1u << Pm4::CB_BLEND0_CONTROL_ENABLE_SHIFT;

// A blended packed UNorm target, which GetGraphicsPrograms EXITs on, is counted and not offered. The
// baseline is the same target unblended, which is offered with its packed mask set, so the walk reaches
// the packed check and the refusal comes from the blend alone.
void TestPackedUnormIsNotOffered() {
	const auto walk_with = [](bool blended) {
		Stream stream = PixelSetup();
		Append(stream, SetContext(Pm4::CB_COLOR0_BASE, {static_cast<uint32_t>(PACKED_TARGET_ADDRESS >> COLOR_BASE_SHIFT)}));
		Append(stream, SetContext(Pm4::CB_COLOR0_INFO, {PACKED_TARGET_INFO}));
		Append(stream, SetContext(Pm4::CB_TARGET_MASK, {TARGET0_CHANNELS, TARGET0_CHANNELS}));
		Append(stream, SetContext(Pm4::SPI_SHADER_COL_FORMAT, {PACKED_UNORM_EXPORT}));
		Append(stream, SetContext(Pm4::CB_BLEND0_CONTROL, {blended ? BLEND_ENABLE : 0u}));
		Append(stream, DrawAuto());
		return Walk(stream);
	};
	const auto blended = walk_with(true);
	CHECK(blended.draws.empty() && blended.stats.unsupported_packed_unorm == 1);
	const auto plain = walk_with(false);
	CHECK(plain.draws.size() == 1 && plain.stats.unsupported_packed_unorm == 0);
	CHECK(!plain.draws.empty() && plain.draws[0].ps_info.target_packed_unorm_mask == 1u);
}

} // namespace

int main(int argc, char** argv) {
	// The vertex table code reads the Config, which the emulator always has and this test makes.
	Config::Initialize();
	if (argc == 2 && std::strcmp(argv[1], "--exit-outside-scope") == 0) {
		return ExitOutsideScope();
	}
	TestVertexFormatGuard();
	TestVertexInfoRefusals();
	TestAddTidGuard();
	TestVertexFetchGuard();
	TestInvalidDescriptorGuard();
	TestRejectScopeCatchesFetchExit();
	TestRejectTallyKeepsLines();
	// The guard tests run alone as their own ctest, so a guard removed shows as that test failing.
	if (argc == 2 && std::strcmp(argv[1], "--guards-only") == 0) {
		if (g_failures != 0) {
			return 1;
		}
		std::puts("GraphicsScanTests guards passed");
		return 0;
	}
	TestInterpolatorWriteReachesDraw();
	TestDrawNeedsProgramWriteSinceDoubt();
	TestPopWithoutPushDoesNotExit();
	TestDeniedAttrib3DoesNotExit();
	TestUnsupportedInputsAreCounted();
	TestRepeatedDrawIsCached();
	TestNoGraphicsLeavesWalkAlone();
	TestSightingsTable();
	TestWalkNotesSightings();
	TestWalkKeepsEveryVerdict();
	TestUserClipPlanesMoveThePushCursor();
	TestClipPlaneSkipsOnlyTheExitCase();
	TestAttributeFormatSplitsVertexKeys();
	TestVertexInfoCarriesDrawState();
	TestRunningJobsCapped();
	TestVertexHalfIgnoresUserData();
	TestVertexHalfPairsWithPixel();
	TestEveryDrawOpcodeIsADraw();
	TestConditionalWriteDoubts();
	TestBranchArmStateIsRestored();
	TestElseArmStateIsRestored();
	TestUnreadableBufferDoubts();
	TestRevisitedBufferDoubts();
	TestIndirectContextListApplies();
	TestResetForgetsState();
	TestClearStateResetsContext();
	TestPrimitiveTypeReachesClassChecks();
	TestIndirectUconfigListApplies();
	TestPackedUnormIsNotOffered();
	TestQueueCapDropsPastLimit();
	TestBusyTimelineSplitsSeconds();
	TestOpenIntervalCountsToNow();
	TestSlotsFillTheTimeline();
	TestAdaptiveCapRearms();
	// Last, since its throttle signal holds every fresh governor at no starts for 3 s.
	TestWindowPausesNewStarts();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("GraphicsScanTests passed");
	return 0;
}
