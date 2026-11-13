#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANHEAPPROBE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANHEAPPROBE_H_

#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"

// The image heap stand-in the prescan's preparer walks with. It tells a snapshot whose images the
// GPU thread would keep resident from one the GPU thread would specialize as the preparer does.
namespace Libs::Graphics {

/// An image heap that keeps nothing resident, which the preparer walks with in place of the GPU
/// thread's heap, since only the GPU thread may use that one. It declines every table, so the walk
/// reads the table entry by entry exactly as it does with no heap at all, and it remembers whether a
/// table was offered. The GPU thread's heap may keep an offered table resident, and the runtime's
/// module is then specialized with heap kinds that the preparer's never has, so the two modules
/// rarely share an id.
class PrescanHeapProbe final: public ShaderRecompiler::IR::ImageHeapSource {
public:
	/// Notes that the walk offered a table and declines it.
	bool Acquire(const ShaderRecompiler::IR::ImageHeapRequest& /*request*/,
	             ShaderRecompiler::IR::ImageHeapView& /*view*/) override {
		m_offered = true;
		return false;
	}
	/// Whether any walk offered a table since the probe was made.
	[[nodiscard]] bool Offered() const { return m_offered; }

private:
	bool m_offered = false;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANHEAPPROBE_H_
