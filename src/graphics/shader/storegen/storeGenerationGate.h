#ifndef GRAPHICS_SHADER_STOREGEN_STORE_GENERATION_GATE_H
#define GRAPHICS_SHADER_STOREGEN_STORE_GENERATION_GATE_H

#include <filesystem>
#include <optional>
#include <string>

namespace Libs::Graphics::StoreGen {

/// What decides whether --generate-shader-store can run. Plain values, so a test needs no Config
/// and no device.
struct StoreGenerationInputs {
	/// Whether --background-warm-up is on, which the generated modules need to be built early.
	bool background_warm_up_enabled = false;
	/// Whether the shader store path that WarmUp was given is empty. It is empty outside a Release
	/// build, without --shader-cache, with the debug dump on, and without a title id.
	bool shader_store_path_empty = true;
	/// Whether the StoreHeader is empty, which it is when the build stamp is unknown.
	bool store_header_empty = true;
	/// The eboot's path as the worker reads it, which may name a member of an archive.
	std::filesystem::path eboot_path;
	/// The executable that will be spawned.
	std::filesystem::path executable_path;
};

/// The reason generation cannot run, or nothing when it can. The reasons are the names the report
/// and the log use: needs_background_warm_up, no_shader_store, no_build_stamp, no_eboot and
/// no_executable. They are checked in that order.
[[nodiscard]] std::optional<std::string>
StoreGenerationAvailability(const StoreGenerationInputs& inputs);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_STORE_GENERATION_GATE_H
