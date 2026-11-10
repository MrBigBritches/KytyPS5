#include "graphics/shader/storegen/ebootSource.h"

#include <utility>

namespace Libs::Graphics::StoreGen {

namespace {

// Written before any thread reads it: the emulator sets it during Run, before the render context
// exists, and the pipeline cache reads it in WarmUp on the main thread.
std::filesystem::path& StoredPath() {
	static std::filesystem::path path;
	return path;
}

} // namespace

void SetEbootPath(std::filesystem::path path) {
	if (StoredPath().empty()) {
		StoredPath() = std::move(path);
	}
}

std::filesystem::path EbootPath() {
	return StoredPath();
}

void ClearEbootPathForTest() {
	StoredPath().clear();
}

} // namespace Libs::Graphics::StoreGen
