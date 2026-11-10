#ifndef GRAPHICS_SHADER_STOREGEN_EBOOT_SOURCE_H
#define GRAPHICS_SHADER_STOREGEN_EBOOT_SOURCE_H

#include <filesystem>

namespace Libs::Graphics::StoreGen {

/// Records where the running title's eboot is, as a path the shader store worker reads. The
/// emulator sets it once, after /app0 is mounted and before the render context makes its pipeline
/// cache, and only when --generate-shader-store is on. For a game in a .zar the path is
/// `<host.zar>!/eboot.bin`, which Common::File reads without a mount. Later calls are ignored, so
/// the first value stays.
void SetEbootPath(std::filesystem::path path);

/// The path that was set, or an empty path when none was.
[[nodiscard]] std::filesystem::path EbootPath();

/// Forgets the path, for a test that runs several titles in one process. The emulator never calls
/// it.
void ClearEbootPathForTest();

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_EBOOT_SOURCE_H
