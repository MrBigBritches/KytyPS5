#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_OPTIONS_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_OPTIONS_H_

#include <span>
#include <string>
#include <string_view>

namespace Libs::Graphics::SystemUi::Gallery {

// The window size the gallery opens at when none is given.
inline constexpr int DEFAULT_WIDTH  = 1920;
inline constexpr int DEFAULT_HEIGHT = 1080;

// What the command line asks of the gallery.
struct GalleryOptions {
	// The folder to write screenshots to. Empty when none was asked for.
	std::string screenshot_dir;
	int         width  = DEFAULT_WIDTH;
	int         height = DEFAULT_HEIGHT;
	// Set by --help, so the caller prints the usage and exits.
	bool help = false;
};

// The outcome of reading the command line. Errors are values because exceptions are off.
struct ParseResult {
	GalleryOptions options;
	// Empty when the arguments were valid, else what was wrong with them.
	std::string error;

	[[nodiscard]] bool Ok() const { return error.empty(); }
};

/// Reads the gallery's command line, `args` being the arguments after the program name. A flag
/// that needs a value but has none, a size that is not a whole number above zero, and a flag
/// the gallery doesn't know all give an error message, with the options read so far.
[[nodiscard]] ParseResult ParseGalleryOptions(std::span<const std::string_view> args);

/// The usage text printed by --help and after an error.
[[nodiscard]] std::string_view GalleryUsage();

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_OPTIONS_H_
