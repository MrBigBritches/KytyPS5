#ifndef GRAPHICS_SHADER_STOREGEN_WORKER_PROCESS_H
#define GRAPHICS_SHADER_STOREGEN_WORKER_PROCESS_H

#include <filesystem>

namespace Libs::Graphics::StoreGen {

/// What the emulator binary does when it is started as `kyty_emulator --shader-store-worker
/// <request file>`. In order it reads the request, turns the crash reporting off, appends the
/// process's stdout and stderr to `<progress log>.out` (so the text of a recompiler EXIT survives
/// for the supervisor, across restarts), arms the child's guards against the parent going away, and
/// runs the worker. The request comes first because it holds the parent's process id, and nothing
/// before the guards starts a thread. It never starts the emulator's subsystems or Config.
/// @return the process's exit code. WORKER_EXIT_BAD_REQUEST when the request is unreadable.
int RunShaderStoreWorkerProcess(const std::filesystem::path& request_file);

/// The command line word that selects the worker mode, read by main before anything else.
inline constexpr const char* WORKER_COMMAND_LINE_FLAG = "--shader-store-worker";

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_WORKER_PROCESS_H
