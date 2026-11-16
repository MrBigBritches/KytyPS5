#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GRAPHICS_SCAN_SHADOW_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GRAPHICS_SCAN_SHADOW_H

#include "graphics/guest_gpu/command_processor/graphicsScanSightings.h"
#include "graphics/guest_gpu/command_processor/guestRegisterFile.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/guest_gpu/hardwareContext.h"

#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

// The graphics register state the prescan's scanner keeps for the graphics queue, and how it turns a
// draw into what building the draw's pixel shader ahead needs. Registers are decoded by the command
// processor's own tables through a GuestRegisterFile, and the key by the GPU thread's own pixel input
// code, so the two cannot drift apart. Nothing here may EXIT on guest state, so every write and draw
// that the processor's code would EXIT on is left out and counted instead.
namespace Libs::Graphics {

/// Where a scanned draw's pixel shader push data starts, as GetGraphicsPrograms works it out with
/// GraphicsDrawDwords. The object primitive id is never set on a draw the scanner offers.
/// @param mesh_active whether the vertex stage runs as a mesh shader, see VertexStageIsMesh.
[[nodiscard]] uint32_t ScannedDrawPushCursor(const ScannedDraw& draw, bool mesh_active);

/// The scanner's twin of the command processor's graphics state, for one queue and one thread.
class GraphicsScanShadow final : public GraphicsScan {
public:
	/// Starts with every register at its reset value and the shadow in doubt, since a scanner that
	/// starts while the game runs has not seen the writes that set the queue's state. Each draw's
	/// vertex half is keyed apart from its pixel half, so a draw whose vertex half is new is offered
	/// for it alone.
	GraphicsScanShadow();

	void ApplyRegister(GraphicsRegisterSpace space, uint32_t offset, uint32_t value) override;
	[[nodiscard]] bool IsInternalDataPacket(uint32_t header, const uint32_t* payload) const override;
	void ApplyContextState(uint32_t operation) override;
	void Reset() override;
	void MarkUncertain() override;
	void RecordDraw(uint32_t queue, bool indirect) override;
	void SaveState() override;
	void RestoreState() override;
	void DropSavedState() override;

	/// Whether both halves of the pixel shader's address were written since the last doubt, which a
	/// draw needs before it is offered.
	[[nodiscard]] bool PsProgramWrittenSinceDoubt() const { return m_state.PsProgramWritten(); }
	/// The context registers, for tests.
	[[nodiscard]] const HW::Context& Context() const { return m_state.ctx; }
	/// Gives the walk the sightings it notes its verdict on each draw's shader addresses in, or null
	/// for none, which is how a run without the first-run report goes. Called before the first walk.
	void SetSightings(GraphicsShaderSightings* sightings) { m_sightings = sightings; }

private:
	// Everything a branch arm may change, so a save and a restore are one copy.
	struct State {
		HW::Context                ctx;
		HW::UserConfig             ucfg;
		HW::Shader                 sh;
		uint32_t                   index_type_and_size = 0;
		std::optional<HW::Context> saved_ctx;
		bool                       ps_lo_written = false;
		bool                       ps_hi_written = false;

		[[nodiscard]] bool PsProgramWritten() const { return ps_lo_written && ps_hi_written; }
	};

	[[nodiscard]] GuestRegisterFile Registers();
	void ApplyContextRegister(uint32_t offset, uint32_t value);
	void ApplyShaderRegister(uint32_t offset, uint32_t value);
	void ApplyUconfigRegister(uint32_t offset, uint32_t value);
	[[nodiscard]] uint32_t ActiveUserClipPlanes() const;
	// The verdict of the class that skips the draw, or nothing when none does.
	[[nodiscard]] std::optional<ScanVerdict> SkippedByClass();
	void AppendDraw(uint32_t queue, bool indirect, ScannedDraw&& draw);
	void AddVertexKeyWords(const ScannedDraw& draw);
	// Notes the walk's verdict on the current draw for its vertex shader and, when `pixel`, its pixel
	// shader.
	void NoteSighting(ScanVerdict verdict, bool pixel = true);

	State              m_state;
	std::vector<State> m_saves;
	// Pixel halves already offered, by key hash, and vertex halves, by vertex key hash, so repeats are
	// not copied again.
	DispatchCache m_cache;
	DispatchCache m_vs_cache;
	// The static key words of the last draw's pixel and vertex halves, kept so that hashing one
	// allocates nothing.
	std::vector<uint32_t> m_key_words;
	std::vector<uint32_t> m_vs_key_words;
	// The sightings, or null, and the verdict bits already noted for each sighting key, so a draw that
	// meets a verdict its key has seen costs one lookup here and takes no lock. Bounded like the
	// sightings.
	GraphicsShaderSightings*               m_sightings = nullptr;
	std::unordered_map<uint64_t, uint32_t> m_noted;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GRAPHICS_SCAN_SHADOW_H
