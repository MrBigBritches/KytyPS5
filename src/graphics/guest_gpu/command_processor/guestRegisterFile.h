#ifndef GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GUEST_REGISTER_FILE_H
#define GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GUEST_REGISTER_FILE_H

#include "graphics/guest_gpu/hardwareContext.h"

#include <cstdint>

namespace Libs::Graphics {

class CommandProcessor;

/// The guest register state that the register tables in pm4Dispatch.h write. It only refers to the
/// state, which lives in the command processor on the GPU thread and in the graphics scan shadow on
/// the scanner thread, so both decode registers with the same tables. It is cheap to copy and is
/// passed by value.
class GuestRegisterFile {
public:
	/// Refers to state that must outlive the view.
	GuestRegisterFile(HW::Context& ctx, HW::UserConfig& ucfg, HW::Shader& sh,
	                  uint32_t& index_type_and_size)
	    : m_ctx(&ctx), m_ucfg(&ucfg), m_sh(&sh), m_index_type_and_size(&index_type_and_size) {}

	/// Refers to a command processor's state. It is implicit so that callers that hand a command
	/// processor to a register table, as tests and the indirect draw patching do, need no change.
	GuestRegisterFile(CommandProcessor& cp); // NOLINT(google-explicit-constructor)

	/// The context registers.
	[[nodiscard]] HW::Context& GetCtx() const { return *m_ctx; }
	/// The user config registers.
	[[nodiscard]] HW::UserConfig& GetUcfg() const { return *m_ucfg; }
	/// The shader registers.
	[[nodiscard]] HW::Shader& GetShCtx() const { return *m_sh; }
	/// Sets the index type from VGT_INDEX_TYPE, keeping its two type bits as the command processor
	/// does.
	void SetIndexType(uint32_t index_type_and_size) const {
		*m_index_type_and_size = index_type_and_size & INDEX_TYPE_MASK;
	}

private:
	static constexpr uint32_t INDEX_TYPE_MASK = 0x3u;

	HW::Context*    m_ctx;
	HW::UserConfig* m_ucfg;
	HW::Shader*     m_sh;
	uint32_t*       m_index_type_and_size;
};

} // namespace Libs::Graphics

#endif // GRAPHICS_GUEST_GPU_COMMAND_PROCESSOR_GUEST_REGISTER_FILE_H
