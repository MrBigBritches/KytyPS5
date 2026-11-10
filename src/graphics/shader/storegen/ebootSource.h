#ifndef GRAPHICS_SHADER_STOREGEN_EBOOT_SOURCE_H
#define GRAPHICS_SHADER_STOREGEN_EBOOT_SOURCE_H

#include <filesystem>

namespace Libs::Graphics::StoreGen {

/// The path a shader store worker reads the title's eboot from, which is what the kernel's /app0
/// mount would resolve the guest path to, which is the mount's directory joined with what follows
/// `/app0/`. On Linux the kernel also matches a folder game's file names without regard to case,
/// and this does not, so a dump that names it EBOOT.BIN only leaves generation off.
/// For a game in a .zar the directory is `<host.zar>!`, so the result is `<host.zar>!/eboot.bin`,
/// which Common::File reads without a mount. A guest path that is not under /app0 is a host path
/// and is returned as it is. It needs no mount, so it can be asked before the kernel starts.
/// @param app0_dir the directory, or the archive root, that /app0 is mounted to.
/// @param elf      the guest path of the program, such as `/app0/eboot.bin`.
[[nodiscard]] std::filesystem::path EbootPathFor(const std::filesystem::path& app0_dir,
                                                 const std::filesystem::path& elf);

/// Records where the running title's eboot is, as a path the shader store worker reads. The
/// emulator sets it once, from Run before Init, since Init makes the render context and its
/// pipeline cache, which read the path in WarmUp. It is only set when --generate-shader-store is
/// on. Later calls are ignored, so the first value stays.
/// @return false when the path was already read while it was not set, which means the setter came
///         too late for the pipeline cache and generation was off for this run. The path is stored
///         anyway.
bool SetEbootPath(std::filesystem::path path);

/// The path that was set, or an empty path when none was.
[[nodiscard]] std::filesystem::path EbootPath();

/// Forgets the path, for a test that runs several titles in one process. The emulator never calls
/// it.
void ClearEbootPathForTest();

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_EBOOT_SOURCE_H
