// Tests the draw-time pixel input helpers that GetGraphicsPrograms and the graphics prescan share.
// Each check that proves a behaviour compares against a baseline that differs from it in one register
// only, so nothing else in the draw could make it pass.
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/pipeline/pixelInputs.h"
#include "graphics/shader/shader.h"

#include <cstdint>
#include <cstdio>

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

constexpr uint64_t TARGET_ADDRESS  = 0x0000001000000000ull;
constexpr uint64_t PROGRAM_ADDRESS = 0x0000004567890100ull;
// Every channel of target 0 on, in CB_TARGET_MASK and CB_SHADER_MASK.
constexpr uint32_t TARGET0_CHANNELS = 0xfu;
// An export mode that writes four channels and is neither none, alpha-less nor packed UNorm.
constexpr uint8_t FOUR_CHANNEL_EXPORT = 4;

// A draw that writes target 0, an RGBA8 UNorm target in the reversed channel order, so its export
// mapping is not the identity and a copy of it can be told from a cleared one. Blending is on with
// factors that read no second source.
HW::Context BaselineContext() {
	HW::Context context;
	context.SetColorBase(0, HW::ColorBase {.addr = TARGET_ADDRESS});
	HW::ColorInfo info;
	info.format        = Prospero::ChannelLayout::k8_8_8_8;
	info.channel_type  = Prospero::ChannelType::kUNorm;
	info.channel_order = Prospero::ChannelOrder::kReversed;
	context.SetColorInfo(0, info);
	context.SetRenderTargetMask(TARGET0_CHANNELS);
	context.SetShaderMask(TARGET0_CHANNELS);
	context.SetTargetOutputMode(0, FOUR_CHANNEL_EXPORT);
	HW::BlendControl blend;
	blend.enable          = true;
	blend.color_srcblend  = static_cast<uint8_t>(Prospero::BlendFactor::kOne);
	blend.color_destblend = static_cast<uint8_t>(Prospero::BlendFactor::kZero);
	blend.alpha_srcblend  = static_cast<uint8_t>(Prospero::BlendFactor::kOne);
	blend.alpha_destblend = static_cast<uint8_t>(Prospero::BlendFactor::kZero);
	context.SetBlendControl(0, blend);
	return context;
}

HW::PixelShaderInfo Program() {
	HW::PixelShaderInfo regs;
	regs.ps_regs.data_addr = PROGRAM_ADDRESS;
	return regs;
}

// What one draw's helpers gave.
struct PixelOutcome {
	uint32_t             color_output_mask = 0;
	bool                 active            = false;
	bool                 mapped            = false;
	PixelInputStatus     status            = PixelInputStatus::Ok;
	ShaderPixelInputInfo info;
};

// Runs the helpers in the order GetGraphicsPrograms does. The part of the info ShaderGetStaticInputInfoPS
// fills from the registers is copied in by hand, since that function lives with the shader map.
PixelOutcome Run(const HW::Context& context, const PixelDrawShape& shape = {}) {
	PixelOutcome outcome;
	outcome.color_output_mask = PixelColorOutputMask(context);
	outcome.active            = PixelStageActive(context, Program(), outcome.color_output_mask);
	PixelTargetExportMapping mapping {};
	outcome.mapped = TryPixelTargetExportMapping(context, outcome.color_output_mask, mapping);
	const auto& sh = context.GetShaderRegisters();
	outcome.info.target_shader_mask = sh.m_cbShaderMask;
	for (uint32_t index = 0; index < PIXEL_COLOR_TARGETS; index++) {
		const auto slot                         = ShaderPixelExportTarget(sh.m_cbShaderMask, index);
		outcome.info.target_output_mode[index]    = sh.target_output_mode[index];
		outcome.info.target_export_mapping[index] = sh.target_output_mode[index] != 0 && slot < 8
		                                                ? mapping[slot]
		                                                : Prospero::ColorComponentMapping {};
	}
	outcome.status = FinishPixelInputInfo(context, shape, outcome.info);
	return outcome;
}

// The baseline writes target 0 with a real mapping and reads no second source. The variants below each
// change one thing from it.
void TestBaseline() {
	const auto outcome = Run(BaselineContext());
	CHECK(outcome.color_output_mask == 1u);
	CHECK(outcome.active);
	CHECK(outcome.mapped);
	CHECK(outcome.status == PixelInputStatus::Ok);
	CHECK(!outcome.info.target_export_mapping[0].IsIdentity());
	CHECK(outcome.info.target_uint_mask == 0 && outcome.info.target_sint_mask == 0);
	CHECK(outcome.info.parameter_mode == ShaderPixelParameterMode::FirstVertex);
	CHECK(!outcome.info.dual_source_blending);
	CHECK(outcome.info.alpha_blend_source == ShaderAlphaBlendSource::None);
	CHECK(outcome.info.target_output_mode[1] == 0);
}

// A signed integer target 0 sets its export's bit in the signed mask. Only the channel type differs from
// the baseline, whose signed mask is empty, and no other path sets that mask.
void TestSignedTargetSetsSintMask() {
	auto          context = BaselineContext();
	HW::ColorInfo info    = context.GetRenderTarget(0).info;
	info.channel_type     = Prospero::ChannelType::kSInt;
	context.SetColorInfo(0, info);
	const auto outcome = Run(context);
	CHECK(outcome.status == PixelInputStatus::Ok);
	CHECK(outcome.info.target_sint_mask == 1u);
	CHECK(outcome.info.target_uint_mask == 0);
}

// A blend factor that reads the second source makes MRT1 a copy of MRT0, mapping included. The alpha
// blend source path also sets dual_source_blending, but it clears MRT1's mapping and leaves the alpha
// blend source set, so the copied mapping and the empty alpha source say the dual-source path ran.
void TestDualSourceFactorCopiesMrt0() {
	auto             context = BaselineContext();
	HW::BlendControl blend   = context.GetBlendControl(0);
	blend.color_srcblend     = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
	context.SetBlendControl(0, blend);
	const auto outcome = Run(context);
	CHECK(outcome.status == PixelInputStatus::Ok);
	CHECK(outcome.info.dual_source_blending);
	CHECK(outcome.info.alpha_blend_source == ShaderAlphaBlendSource::None);
	CHECK(outcome.info.target_output_mode[1] == FOUR_CHANNEL_EXPORT);
	CHECK(outcome.info.target_export_mapping[1].packed == outcome.info.target_export_mapping[0].packed);
	CHECK(!outcome.info.target_export_mapping[1].IsIdentity());
}

// PROVOKING_VTX_LAST makes flat inputs come from the last vertex. The baseline has the same primitive
// and takes them from the first.
void TestProvokingLastVertex() {
	auto            context = BaselineContext();
	HW::ModeControl mode    = context.GetModeControl();
	mode.provoking_vtx_last = true;
	context.SetModeControl(mode);
	CHECK(Run(context).info.parameter_mode == ShaderPixelParameterMode::LastVertex);
}

// A rect list interpolates as a rectangle, except on a mesh draw, which differs from it only in that.
void TestRectListUnlessMesh() {
	const PixelDrawShape rect {.primitive_type = Prospero::PrimitiveType::kRectList};
	PixelDrawShape       mesh_rect = rect;
	mesh_rect.mesh_active          = true;
	CHECK(Run(BaselineContext(), rect).info.parameter_mode == ShaderPixelParameterMode::Rectangle);
	CHECK(Run(BaselineContext(), mesh_rect).info.parameter_mode == ShaderPixelParameterMode::FirstVertex);
}

// A written target whose format the host has no render target for is refused, where the GPU thread's
// TextureGetRenderTargetFormat would EXIT. The baseline differs only in the format and is mapped.
void TestUnsupportedTargetFormatIsRefused() {
	auto          context = BaselineContext();
	HW::ColorInfo info    = context.GetRenderTarget(0).info;
	info.format           = Prospero::ChannelLayout::kInvalid;
	context.SetColorInfo(0, info);
	PixelTargetExportMapping mapping {};
	CHECK(!TryPixelTargetExportMapping(context, PixelColorOutputMask(context), mapping));
	CHECK(TryPixelTargetExportMapping(BaselineContext(), 1u, mapping));
}

// With no channel of target 0 left on in CB_TARGET_MASK the shader writes nothing and has no side
// effect, so it does not run. The baseline differs only in the mask.
void TestMaskedTargetLeavesStageInactive() {
	auto context = BaselineContext();
	context.SetRenderTargetMask(0);
	const auto outcome = Run(context);
	CHECK(outcome.color_output_mask == 0);
	CHECK(!outcome.active);
}

// The export mode a packed UNorm target must be written in.
constexpr uint8_t PACKED_UNORM_EXPORT = 9;

// The baseline with target 0 made an 11_11_10 UNorm target, the packed UNorm format, written whole in
// the packed export mode and unblended, which the host can write.
HW::Context PackedUnormContext() {
	auto          context = BaselineContext();
	HW::ColorInfo info    = context.GetRenderTarget(0).info;
	info.format           = Prospero::ChannelLayout::k11_11_10;
	info.channel_order    = Prospero::ChannelOrder::kStandard;
	context.SetColorInfo(0, info);
	context.SetTargetOutputMode(0, PACKED_UNORM_EXPORT);
	HW::BlendControl blend = context.GetBlendControl(0);
	blend.enable           = false;
	context.SetBlendControl(0, blend);
	return context;
}

// A packed UNorm target that is blended, or written in another export mode, is refused where the GPU
// thread EXITs. The baseline is the same target unblended in the packed mode, which is accepted and
// sets its packed mask, so the packed path is reached and each refusal comes from its one register.
void TestPackedUnormIsRefused() {
	const auto accepted = Run(PackedUnormContext());
	CHECK(accepted.status == PixelInputStatus::Ok);
	CHECK(accepted.info.target_packed_unorm_mask == 1u);

	auto             blended = PackedUnormContext();
	HW::BlendControl blend   = blended.GetBlendControl(0);
	blend.enable             = true;
	blended.SetBlendControl(0, blend);
	CHECK(Run(blended).status == PixelInputStatus::UnsupportedPackedUnorm);

	auto other_mode = PackedUnormContext();
	other_mode.SetTargetOutputMode(0, FOUR_CHANNEL_EXPORT);
	CHECK(Run(other_mode).status == PixelInputStatus::UnsupportedPackedUnorm);
}

// The EXITing mapping gives what the Try form gives for a target both accept. The baseline's reversed
// channel order makes its mapping other than the identity, so a mapping left default does not match.
void TestExitingMappingMatchesTry() {
	const auto               context = BaselineContext();
	const auto               written = PixelColorOutputMask(context);
	PixelTargetExportMapping tried {};
	CHECK(TryPixelTargetExportMapping(context, written, tried));
	const auto exiting = PixelTargetExportMappingOf(context, written);
	CHECK(!exiting[0].IsIdentity());
	for (uint32_t slot = 0; slot < PIXEL_COLOR_TARGETS; slot++) {
		CHECK(exiting[slot].packed == tried[slot].packed);
	}
}

} // namespace

int main() {
	TestBaseline();
	TestSignedTargetSetsSintMask();
	TestDualSourceFactorCopiesMrt0();
	TestProvokingLastVertex();
	TestRectListUnlessMesh();
	TestUnsupportedTargetFormatIsRefused();
	TestMaskedTargetLeavesStageInactive();
	TestPackedUnormIsRefused();
	TestExitingMappingMatchesTry();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("PixelInputsTests passed");
	return 0;
}
