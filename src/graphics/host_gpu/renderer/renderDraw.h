#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace Common::FirstRunReport {
struct SkipTargets;
} // namespace Common::FirstRunReport

namespace Libs::Graphics {

struct ShaderVertexInputInfo;
struct DrawRenderState;
class TextureCache;

[[nodiscard]] std::pair<int32_t, uint32_t>
ResolveDrawOffsets(uint32_t index_offset, const ShaderVertexInputInfo& vs_input_info);

/// The fewest draws a target must have had in each of two frames in a row before a draw into it may
/// go without its pipeline. A single-draw bake or a portrait should not pass. It is a guess and
/// has not been measured on any game.
inline constexpr uint32_t MIN_DRAWS_FOR_SKIP = 8;

/// Records the targets of a draw that ran in their frame histories. A draw that was skipped is not
/// recorded, and neither is one still waiting to be asked about, so the history is only the
/// earlier draws that ran.
void NoteDrawTargets(TextureCache& cache, const DrawRenderState& state);

/// Whether a draw with no pipeline yet may go without it. All of these must hold, and anything else
/// waits for the pipeline.
///  - No stage writes anything besides the render targets, since a skipped write is never replayed.
///  - The draw has a depth target, drawn into at least MIN_DRAWS_FOR_SKIP times in this frame and
///    in the one before, with a clear of the whole depth target in both. Only the clear register
///    counts, and only when the render area is the whole target. An HTILE clear does not count.
///  - Each color target was drawn into that many times in both frames. This does NOT show that the
///    color targets are rebuilt, because a color clear is invisible to the CPU. A color target that
///    is written every frame but never cleared, such as an accumulation or trail buffer drawn
///    beside a depth buffer that is cleared, passes, and a skipped draw into it is lost for good.
///    That risk is accepted, and the report counts skipped draws that are sampled afterwards.
///  - Each target is viewed whole, as one level with all its layers, because the history belongs
///    to the image. With --async-pipelines-prefer-skip a partial view, such as one layer or mip, is
///    judged instead by the history of its exact range.
/// With `depth_only` a draw with any color target waits too.
///
/// The rule that refused a draw. None means the draw may skip. The order is the order of the checks,
/// so a draw that breaks several rules is blamed on the first.
enum class SkipRefusal : uint8_t {
	None,
	/// No depth target. A color-only pass shows no sign of being rebuilt.
	NoDepth,
	/// A stage writes beyond the render targets.
	Writer,
	/// Depth-only mode, and the draw has a color target.
	DepthOnlyModeColor,
	/// A target is viewed as part of its image, as a layer or a mip.
	PartialView,
	/// No register clear in both frames, but an HTILE clear waited for the draws of both. An HTILE
	/// clear is not taken as a clear, so this is the cost of that rule.
	HtileOnlyClear,
	/// No register clear of the whole depth target in both frames.
	DepthNotCleared,
	/// Fewer than MIN_DRAWS_FOR_SKIP draws into the depth target in both frames, though the frame
	/// before had a register clear, so a lower floor might have been enough.
	TooFewDraws,
	/// Fewer draws too, and no register clear in the frame before either, so a lower floor would
	/// not have helped. A draw is blamed on one of the two, never both.
	TooFewDrawsAlsoUncleared,
	/// A color target without enough draws in both frames.
	ColorHistory,
};

/// How many refusal reasons there are.
inline constexpr size_t SKIP_REFUSAL_COUNT = static_cast<size_t>(SkipRefusal::ColorHistory) + 1;

/// The name a report gives a refusal.
[[nodiscard]] const char* SkipRefusalName(SkipRefusal refusal);

/// Applies the rule above and says which part refused the draw. For a ColorHistory refusal the index
/// of the refusing colour target in the draw's colour targets goes to `refusing_color` when given.
[[nodiscard]] SkipRefusal ExplainSkipRefusal(TextureCache& cache, const DrawRenderState& state,
                                             bool depth_only, uint32_t* refusing_color = nullptr);

/// For the first-run report, the facts of the colour target that refused a draw for its colour
/// history and of the draw's depth target, see TextureCache::DescribeSkipTarget. Null for any other
/// refusal and with the report off, so a run without the report pays nothing.
/// @param refusing_color the index ExplainSkipRefusal gave for the refusal.
[[nodiscard]] std::shared_ptr<const Common::FirstRunReport::SkipTargets>
DescribeSkipRefusal(TextureCache& cache, const DrawRenderState& state, SkipRefusal refusal, uint32_t refusing_color);

/// Whether ExplainSkipRefusal finds no reason to refuse.
[[nodiscard]] bool DrawMaySkipPipeline(TextureCache& cache, const DrawRenderState& state,
                                       bool depth_only);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_RENDERDRAW_H_
