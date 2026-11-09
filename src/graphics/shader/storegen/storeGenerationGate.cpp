#include "graphics/shader/storegen/storeGenerationGate.h"

#include "common/file.h"

namespace Libs::Graphics::StoreGen {

std::optional<std::string> StoreGenerationAvailability(const StoreGenerationInputs& inputs) {
	if (!inputs.background_warm_up_enabled) {
		return "needs_background_warm_up";
	}
	if (inputs.shader_store_path_empty) {
		return "no_shader_store";
	}
	if (inputs.store_header_empty) {
		return "no_build_stamp";
	}
	// IsFileExisting also answers for a member of an archive, as in `<host.zar>!/eboot.bin`.
	if (inputs.eboot_path.empty() || !Common::File::IsFileExisting(inputs.eboot_path)) {
		return "no_eboot";
	}
	if (inputs.executable_path.empty() || !Common::File::IsFileExisting(inputs.executable_path)) {
		return "no_executable";
	}
	return std::nullopt;
}

} // namespace Libs::Graphics::StoreGen
