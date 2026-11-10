#include "graphics/shader/storegen/ebootSource.h"

#include "common/stringUtils.h"

#include <string>
#include <string_view>
#include <utility>

namespace Libs::Graphics::StoreGen {

namespace {

// Written before any thread reads it. The emulator sets it at the start of Run, and the pipeline
// cache reads it in WarmUp on the main thread.
struct Stored {
	std::filesystem::path path;
	// Set when the path was read while it was empty, which makes a later setter too late.
	bool read_unset = false;
};

Stored& State() {
	static Stored stored;
	return stored;
}

constexpr std::string_view APP0_PREFIX = "/app0/";

} // namespace

std::filesystem::path EbootPathFor(const std::filesystem::path& app0_dir,
                                   const std::filesystem::path& elf) {
	const auto text = Common::PathToGenericString(elf);
	if (text.starts_with(APP0_PREFIX)) {
		return app0_dir / Common::PathFromUtf8(text.substr(APP0_PREFIX.size()));
	}
	return elf;
}

bool SetEbootPath(std::filesystem::path path) {
	auto& state = State();
	if (state.path.empty()) {
		state.path = std::move(path);
	}
	return !state.read_unset;
}

std::filesystem::path EbootPath() {
	auto& state = State();
	if (state.path.empty()) {
		state.read_unset = true;
	}
	return state.path;
}

void ClearEbootPathForTest() {
	State() = {};
}

} // namespace Libs::Graphics::StoreGen
