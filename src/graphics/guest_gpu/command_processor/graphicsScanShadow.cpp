#include "graphics/guest_gpu/command_processor/graphicsScanShadow.h"

#include "common/firstRunReport.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/pm4Dispatch.h"
#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/pixelInputs.h"
#include "graphics/host_gpu/renderer/pipeline/prescanVariants.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderCompiler.h"
#include "libs/agc.h"

#include <utility>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

// The registers of one color target's block repeat this far apart.
constexpr uint32_t CB_COLOR_STRIDE = 15;
// The compute shader registers, from the first COMPUTE_ register to the end of its user data. The
// compute shadow keeps them, and the decoder of one of them EXITs on a register it does not know.
constexpr uint32_t COMPUTE_USER_DATA_SLOTS = 16;
constexpr uint32_t COMPUTE_SH_FIRST        = Pm4::COMPUTE_START_X;
constexpr uint32_t COMPUTE_SH_END          = Pm4::COMPUTE_USER_DATA_0 + COMPUTE_USER_DATA_SLOTS;
// Each user accumulator block has four registers, whose decoder EXITs on a value above 7 bits.
constexpr uint32_t USER_ACCUM_SLOTS = 4;
// How many branch saves are reserved, so that common nesting allocates nothing.
constexpr size_t RESERVED_SAVES = 4;
// The clip and cull distance enables of PA_CL_VS_OUT_CNTL, which GetGraphicsPrograms refuses with
// user clip planes.
constexpr uint32_t VS_OUT_CLIP_CULL_DISTANCE_MASK = 0xffffu;

// Context registers whose decoder EXITs on some values. None of them decides a pixel shader's key, so
// leaving them out costs nothing. They are DB_SHADING_RATE_ENCODING, CB_COLORn_DCC_CONTROL and
// CB_COLORn_ATTRIB3.
bool IsDeniedContextRegister(uint32_t offset) {
	if (offset == Pm4::DB_SHADING_RATE_ENCODING) {
		return true;
	}
	if (offset >= Pm4::CB_COLOR0_DCC_CONTROL && offset <= Pm4::CB_COLOR7_DCC_CONTROL &&
	    (offset - Pm4::CB_COLOR0_DCC_CONTROL) % CB_COLOR_STRIDE == 0) {
		return true;
	}
	return offset >= Pm4::CB_COLOR0_ATTRIB3 && offset <= Pm4::CB_COLOR7_ATTRIB3;
}

// Shader registers the graphics shadow leaves out, which are the compute ones and the user
// accumulators, whose decoders may EXIT.
bool IsDeniedShaderRegister(uint32_t offset) {
	if (offset >= COMPUTE_SH_FIRST && offset < COMPUTE_SH_END) {
		return true;
	}
	for (const uint32_t base: {Pm4::SPI_SHADER_USER_ACCUM_PS_0, Pm4::SPI_SHADER_USER_ACCUM_ESGS_0,
	                           Pm4::SPI_SHADER_USER_ACCUM_LSHS_0}) {
		if (offset >= base && offset - base < USER_ACCUM_SLOTS) {
			return true;
		}
	}
	return false;
}

} // namespace

uint32_t ScannedDrawPushCursor(const ScannedDraw& draw, bool mesh_active) {
	return ShaderRecompiler::IR::PushData::GraphicsDrawDwords(mesh_active, false, draw.user_clip_planes);
}

GraphicsScanShadow::GraphicsScanShadow(): m_cache(PRESCAN_PROBE_INTERVAL), m_vs_cache(PRESCAN_PROBE_INTERVAL) {
	m_saves.reserve(RESERVED_SAVES);
	draws.reserve(SCAN_DISPATCH_CAPACITY);
}

GuestRegisterFile GraphicsScanShadow::Registers() {
	return {m_state.ctx, m_state.ucfg, m_state.sh, m_state.index_type_and_size};
}

void GraphicsScanShadow::ApplyRegister(GraphicsRegisterSpace space, uint32_t offset, uint32_t value) {
	switch (space) {
		case GraphicsRegisterSpace::Context: ApplyContextRegister(offset, value); break;
		case GraphicsRegisterSpace::Shader: ApplyShaderRegister(offset, value); break;
		case GraphicsRegisterSpace::Uconfig: ApplyUconfigRegister(offset, value); break;
	}
}

// The processor EXITs on an offset past its table and on a null entry, so both are counted and left
// out, as is the deny list.
void GraphicsScanShadow::ApplyContextRegister(uint32_t offset, uint32_t value) {
	if (offset == Pm4::CX_NOP) {
		return;
	}
	if (offset < Pm4::CX_NUM && IsDeniedContextRegister(offset)) {
		stats.registers_denied++;
		return;
	}
	if (offset >= Pm4::CX_NUM || g_hw_ctx_indirect_func[offset] == nullptr) {
		stats.registers_unknown++;
		return;
	}
	g_hw_ctx_indirect_func[offset](Registers(), offset, value);
	state_dirty = true;
}

// Also notes a write of either half of the pixel shader's address, which a draw needs since the last
// doubt.
void GraphicsScanShadow::ApplyShaderRegister(uint32_t offset, uint32_t value) {
	if (offset == Pm4::SH_NOP) {
		return;
	}
	if (offset < Pm4::SH_NUM && IsDeniedShaderRegister(offset)) {
		stats.registers_denied++;
		return;
	}
	if (offset >= Pm4::SH_NUM || g_hw_sh_indirect_func[offset] == nullptr) {
		stats.registers_unknown++;
		return;
	}
	g_hw_sh_indirect_func[offset](Registers(), offset, value);
	state_dirty = true;
	if (offset == Pm4::SPI_SHADER_PGM_LO_PS) {
		m_state.ps_lo_written = true;
	} else if (offset == Pm4::SPI_SHADER_PGM_HI_PS) {
		m_state.ps_hi_written = true;
	}
}

void GraphicsScanShadow::ApplyUconfigRegister(uint32_t offset, uint32_t value) {
	if (offset == Pm4::UC_NOP) {
		return;
	}
	if (offset >= Pm4::UC_NUM || g_hw_uc_indirect_func[offset] == nullptr) {
		stats.registers_unknown++;
		return;
	}
	g_hw_uc_indirect_func[offset](Registers(), offset, value);
	state_dirty = true;
}

bool GraphicsScanShadow::IsInternalDataPacket(uint32_t header, const uint32_t* payload) const {
	return Gen5::AgcIsInternalDataPacket(header, payload);
}

// As CommandProcessor::ApplyContextStateOperation, except that a push over a saved context, a pop with
// none and an unknown operation, which the processor EXITs on, put the shadow in doubt instead.
void GraphicsScanShadow::ApplyContextState(uint32_t operation) {
	state_dirty     = true;
	auto&      ctx  = m_state.ctx;
	auto&      save = m_state.saved_ctx;
	const auto note_imbalance = [this] {
		stats.context_state_imbalance++;
		MarkUncertain();
	};
	switch (static_cast<ContextStateOperation>(operation)) {
		case ContextStateOperation::Clear: ctx.Reset(); break;
		case ContextStateOperation::Push:
		case ContextStateOperation::PushClear:
			if (save.has_value()) {
				note_imbalance();
			}
			save.emplace(ctx);
			if (static_cast<ContextStateOperation>(operation) == ContextStateOperation::PushClear) {
				ctx.Reset();
			}
			break;
		case ContextStateOperation::Pop:
			if (!save.has_value()) {
				note_imbalance();
				break;
			}
			ctx = *save;
			save.reset();
			break;
		default: note_imbalance(); break;
	}
}

// The reset leaves every register known, so it needs no doubt, but a draw still waits for the pixel
// shader's address to be written. Saves for branch arms around the reset are kept.
void GraphicsScanShadow::Reset() {
	m_state     = State {};
	state_dirty = true;
}

void GraphicsScanShadow::MarkUncertain() {
	m_state.ps_lo_written = false;
	m_state.ps_hi_written = false;
	stats.doubts++;
}

void GraphicsScanShadow::SaveState() {
	m_saves.push_back(m_state);
}

void GraphicsScanShadow::RestoreState() {
	if (!m_saves.empty()) {
		m_state = m_saves.back();
	}
}

void GraphicsScanShadow::DropSavedState() {
	if (!m_saves.empty()) {
		m_saves.pop_back();
	}
}

// The user clip planes GetGraphicsPrograms applies, which CLIP_DISABLE turns off.
uint32_t GraphicsScanShadow::ActiveUserClipPlanes() const {
	const auto& clip = m_state.ctx.GetClipControl();
	return clip.clip_disable ? 0u : clip.user_clip_planes;
}

// The classes of draw whose key the prescan does not predict. Tessellation and rect lists change the
// stages around the pixel shader, and the object primitive id changes where its push data starts.
// Active user clip planes are predicted, except where GetGraphicsPrograms EXITs on them, which is a
// vertex stage that writes clip or cull distances (the low half of PA_CL_VS_OUT_CNTL). Tessellation
// and rect lists, its other two EXITs there, were skipped already.
std::optional<ScanVerdict> GraphicsScanShadow::SkippedByClass() {
	const auto  primitive = m_state.ucfg.GetPrimType();
	const auto& ctx       = m_state.ctx;
	if (primitive == Prospero::PrimitiveType::kPatch) {
		stats.skipped_tessellation++;
		return ScanVerdict::SkippedTessellation;
	}
	if (Prospero::IsRectList(primitive)) {
		stats.skipped_rect_list++;
		return ScanVerdict::SkippedRectList;
	}
	if (ctx.HasObjectPrimitiveIdOutput()) {
		stats.skipped_object_id++;
		return ScanVerdict::SkippedObjectId;
	}
	if (ActiveUserClipPlanes() != 0 &&
	    (ctx.GetShaderRegisters().m_paClVsOutCntl & VS_OUT_CLIP_CULL_DISTANCE_MASK) != 0) {
		stats.skipped_clip_planes++;
		return ScanVerdict::SkippedClipPlanes;
	}
	return std::nullopt;
}

// Without sightings this returns at once, so a run without the report pays one test per draw. With
// them a verdict the key has already seen is a lookup in the walk's own map, and only a new verdict
// bit takes the sightings' lock and the clock, at most once per verdict per key. A key past the cap
// is not noted, since the sightings would turn it away too.
void GraphicsScanShadow::NoteSighting(ScanVerdict verdict, bool pixel) {
	if (m_sightings == nullptr) {
		return;
	}
	const uint32_t bit  = ScanVerdictBit(verdict);
	const auto     note = [&](SightingStage stage, uint64_t address) {
		const auto key   = SightingKey(stage, address);
		auto       found = m_noted.find(key);
		if (found == m_noted.end()) {
			if (address == 0 || m_noted.size() >= GRAPHICS_SIGHTING_CAP) {
				return;
			}
			found = m_noted.emplace(key, 0u).first;
		}
		if ((found->second & bit) != 0) {
			return;
		}
		found->second |= bit;
		m_sightings->NoteScan(stage, address, verdict, Common::FirstRunReport::NowNs());
	};
	note(SightingStage::Vertex, m_state.sh.GetVs().es_regs.data_addr);
	if (pixel) {
		note(SightingStage::Pixel, m_state.sh.GetPs().ps_regs.data_addr);
	}
}

// Works out the draw's pixel input info as GetGraphicsPrograms does, after every check that stands in
// for one of its EXITs. The shadow reads no shader map, so the mapped data is empty and the worker adds
// the scratch size and the custom interpolation mask. Each way out notes its verdict as a sighting.
void GraphicsScanShadow::RecordDraw(uint32_t queue, bool indirect) {
	stats.draws++;
	const auto&    ctx               = m_state.ctx;
	const auto&    ps                = m_state.sh.GetPs();
	const uint32_t color_output_mask = PixelColorOutputMask(ctx);
	if (!PixelStageActive(ctx, ps, color_output_mask)) {
		stats.no_pixel_stage++;
		// The pixel shader address left in the registers is not this draw's, so only the vertex
		// shader's sighting is noted.
		return NoteSighting(ScanVerdict::NoPixelStage, false);
	}
	if (!m_state.PsProgramWritten()) {
		stats.ps_unwritten++;
		return NoteSighting(ScanVerdict::PsUnwritten);
	}
	if (const auto skipped = SkippedByClass()) {
		return NoteSighting(*skipped);
	}
	const auto& sh = ctx.GetShaderRegisters();
	if (!ShaderPsInputsSupported(sh)) {
		stats.unsupported_inputs++;
		return NoteSighting(ScanVerdict::UnsupportedInputs);
	}
	PixelTargetExportMapping mapping {};
	if (!TryPixelTargetExportMapping(ctx, color_output_mask, mapping)) {
		stats.unsupported_target_format++;
		return NoteSighting(ScanVerdict::UnsupportedTargetFormat);
	}
	ScannedDraw draw;
	ShaderGetStaticInputInfoPS(ps, sh, mapping, ShaderMappedData {}, draw.ps_info);
	// Rect lists, tessellation and the object primitive id were skipped above, so the vertex stage only
	// matters to the worker's push data start.
	const PixelDrawShape shape {.primitive_type = m_state.ucfg.GetPrimType()};
	if (FinishPixelInputInfo(ctx, shape, draw.ps_info) != PixelInputStatus::Ok) {
		stats.unsupported_packed_unorm++;
		return NoteSighting(ScanVerdict::UnsupportedPackedUnorm);
	}
	AppendDraw(queue, indirect, std::move(draw));
}

// The vertex half's key words: what the vertex shader's key and the pre-rasterization library depend
// on that sits in registers. User data, and with it the attribute and buffer table pointers, and the
// viewport are left out, since they change from draw to draw and keying on them offered almost every
// draw again. The worker reads the tables, so a draw whose formats differ only there is not offered
// again, and neither is a CLIP_DISABLE draw that differs only in its viewport.
void GraphicsScanShadow::AddVertexKeyWords(const ScannedDraw& draw) {
	const auto& es        = draw.vs_regs.es_regs;
	const auto& gs        = draw.vs_regs.gs_regs;
	const auto  low_word  = [](uint64_t value) { return static_cast<uint32_t>(value); };
	const auto  high_word = [](uint64_t value) { return static_cast<uint32_t>(value >> 32u); };
	m_vs_key_words.clear();
	// The pixel half's key too, since the vertex shader's push data starts where the pixel shader's
	// ends, so each pairing is offered once and its job uses that pixel shader's bindings. The key_hash
	// must be set before this is called.
	m_vs_key_words.insert(m_vs_key_words.end(),
	                      {low_word(es.data_addr), high_word(es.data_addr), draw.shader_stages, draw.gs_lds_size,
	                       gs.rsrc2.user_sgpr, draw.pa_cl_vs_out_cntl, static_cast<uint32_t>(draw.indirect),
	                       low_word(draw.key_hash), high_word(draw.key_hash)});
	const auto& clip = draw.clip_control;
	const auto& mode = draw.mode_control;
	m_vs_key_words.push_back(static_cast<uint32_t>(clip.dx_clip_space) | (static_cast<uint32_t>(clip.clip_disable) << 1u) |
	                      (static_cast<uint32_t>(clip.min_z_clip_disable) << 2u) |
	                      (static_cast<uint32_t>(clip.max_z_clip_disable) << 3u) |
	                      (static_cast<uint32_t>(mode.provoking_vtx_last) << 4u) |
	                      (static_cast<uint32_t>(mode.cull_front) << 5u) | (static_cast<uint32_t>(mode.cull_back) << 6u) |
	                      (mode.poly_mode << 8u) | (mode.polymode_front_ptype << 12u) |
	                      (mode.polymode_back_ptype << 16u));
}

// Keys the draw by its program and static key words, skips it when the cache has it, and otherwise
// fills in what the worker needs and appends it.
void GraphicsScanShadow::AppendDraw(uint32_t queue, bool indirect, ScannedDraw&& draw) {
	const auto& ps = m_state.sh.GetPs();
	BuildStageStaticKey(draw.ps_info, m_key_words);
	// The planes change the module through its push start, so the same program with and without them
	// is two draws to offer.
	draw.user_clip_planes = ActiveUserClipPlanes();
	m_key_words.push_back(draw.user_clip_planes);
	const auto& ctx         = m_state.ctx;
	const auto& vs          = m_state.sh.GetVs();
	draw.vs_regs            = vs;
	draw.vs_program_address = vs.es_regs.data_addr;
	draw.shader_stages      = ctx.GetShaderStages();
	draw.gs_lds_size        = vs.gs_regs.rsrc2.lds_size;
	draw.pa_cl_vs_out_cntl  = ctx.GetShaderRegisters().m_paClVsOutCntl;
	draw.clip_control       = ctx.GetClipControl();
	draw.mode_control       = ctx.GetModeControl();
	const auto& viewport    = ctx.GetScreenViewport().viewports[0];
	draw.viewport_scale     = {viewport.xscale, viewport.yscale};
	draw.viewport_offset    = {viewport.xoffset, viewport.yoffset};
	draw.indirect           = indirect;
	// The program address seeds each hash, so one key at two addresses gives two hashes.
	draw.key_hash = XXH3_64bits_withSeed(m_key_words.data(), m_key_words.size() * sizeof(uint32_t),
	                                     ps.ps_regs.data_addr);
	AddVertexKeyWords(draw);
	draw.vs_key_hash = XXH3_64bits_withSeed(m_vs_key_words.data(), m_vs_key_words.size() * sizeof(uint32_t),
	                                        draw.vs_program_address);
	// Each half has its own cache, and the draw is skipped only when both have seen it.
	const auto ps_verdict = m_cache.Check(draw.key_hash);
	const auto vs_verdict = draw.vs_key_hash != 0 ? m_vs_cache.Check(draw.vs_key_hash) : DispatchCache::Verdict::Skip;
	draw.offer_ps         = ps_verdict != DispatchCache::Verdict::Skip;
	draw.offer_vs         = vs_verdict != DispatchCache::Verdict::Skip;
	if (!draw.offer_ps && !draw.offer_vs) {
		stats.cached++;
		return NoteSighting(ScanVerdict::Cached);
	}
	NoteSighting(ScanVerdict::Recorded);
	draw.force_probe        = ps_verdict == DispatchCache::Verdict::Probe || vs_verdict == DispatchCache::Verdict::Probe;
	draw.ps_program_address = ps.ps_regs.data_addr;
	draw.ps_regs            = ps;
	draw.alpha_to_coverage  = ctx.GetShaderRegisters().AlphaToMask();
	draw.queue              = queue;
	m_cache.Remember(draw.key_hash);
	if (draw.vs_key_hash != 0) {
		m_vs_cache.Remember(draw.vs_key_hash);
	}
	draws.push_back(std::move(draw));
	stats.recorded++;
}

} // namespace Libs::Graphics
