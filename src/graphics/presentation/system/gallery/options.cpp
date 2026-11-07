#include "graphics/presentation/system/gallery/options.h"

#include <charconv>
#include <cstddef>
#include <string_view>
#include <system_error>

namespace Libs::Graphics::SystemUi::Gallery {
namespace {

constexpr std::string_view FLAG_SCREENSHOT = "--screenshot";
constexpr std::string_view FLAG_WIDTH      = "--width";
constexpr std::string_view FLAG_HEIGHT     = "--height";
constexpr std::string_view FLAG_HELP       = "--help";

// Reads `text` as a whole number above zero, and nothing else. from_chars is used over atoi
// because it reports trailing junk and overflow instead of guessing.
bool ParseSize(std::string_view text, int& out) {
	int         value  = 0;
	const char* end    = text.data() + text.size();
	const auto  parsed = std::from_chars(text.data(), end, value);
	if (text.empty() || parsed.ec != std::errc() || parsed.ptr != end || value <= 0) {
		return false;
	}
	out = value;
	return true;
}

// Reads the value of a size flag at `args[index]`, advancing `index` past it. Returns the error
// message, or an empty string on success.
std::string ReadSize(std::span<const std::string_view> args, size_t& index, int& out) {
	const std::string_view flag = args[index];
	if (index + 1 >= args.size() || args[index + 1].starts_with("--")) {
		return std::string(flag) + " needs a value";
	}
	const std::string_view value = args[++index];
	if (!ParseSize(value, out)) {
		return std::string(flag) + " needs a whole number above zero, got '" + std::string(value) +
		       "'";
	}
	return {};
}

} // namespace

ParseResult ParseGalleryOptions(std::span<const std::string_view> args) {
	ParseResult result;
	for (size_t index = 0; index < args.size(); index++) {
		const std::string_view flag = args[index];
		if (flag == FLAG_HELP) {
			result.options.help = true;
		} else if (flag == FLAG_SCREENSHOT) {
			if (index + 1 >= args.size() || args[index + 1].starts_with("--")) {
				result.error = std::string(flag) + " needs a value";
			} else {
				result.options.screenshot_dir = std::string(args[++index]);
			}
		} else if (flag == FLAG_WIDTH) {
			result.error = ReadSize(args, index, result.options.width);
		} else if (flag == FLAG_HEIGHT) {
			result.error = ReadSize(args, index, result.options.height);
		} else {
			result.error = "unknown option '" + std::string(flag) + "'";
		}
		if (!result.Ok()) {
			break;
		}
	}
	return result;
}

std::string_view GalleryUsage() {
	return "Usage: system_ui_gallery [options]\n"
	       "  --screenshot <dir>  Write a screenshot of each page to <dir>\n"
	       "  --width <n>         Window width in pixels (default 1920)\n"
	       "  --height <n>        Window height in pixels (default 1080)\n"
	       "  --help              Show this text\n";
}

} // namespace Libs::Graphics::SystemUi::Gallery
