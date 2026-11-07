#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_CAPTURE_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_CAPTURE_H_

#include "graphics/presentation/system/gallery/page.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace Libs::Graphics::SystemUi::Gallery {

// A rendered frame as read back from the GPU.
struct CapturedImage {
	std::vector<uint8_t> pixels;
	int                  width  = 0;
	int                  height = 0;
	// Bytes from the start of one row to the start of the next, which may exceed width * 4.
	size_t row_pitch = 0;
	// True when the pixels are in blue, green, red, alpha order.
	bool bgra = false;
};

// One screenshot to take.
struct ScreenshotJob {
	// The page's index in the page list.
	int         page = 0;
	// The view to show, for a page with views.
	int         view = 0;
	std::string file_name;
};

/// Packs `pixels`, `height` rows of `width` 4-byte pixels each `row_pitch` bytes apart, into
/// tightly packed RGBA. With `bgra`, the red and blue channels are swapped. Returns an empty
/// vector when the size is not positive, the pitch is less than a row, or `pixels` is too short.
[[nodiscard]] std::vector<uint8_t> ToRgba(std::span<const uint8_t> pixels, int width, int height,
                                          size_t row_pitch, bool bgra);

/// The file name for a screenshot of `page`, its group and id in lower case, as in
/// "components-button.png". A view after the first adds its name, as in "styles-colour-roles.png".
[[nodiscard]] std::string ScreenshotName(const PageEntry& page, int view = 0);

/// A job for every view of every page in `pages`, in order, and one for a page with no views.
[[nodiscard]] std::vector<ScreenshotJob> ScreenshotJobs(std::span<const PageEntry> pages);

/// `rgba`, `width` by `height` tightly packed, as the bytes of a PNG, or nothing when the size
/// doesn't match the data.
[[nodiscard]] std::vector<std::byte> EncodePng(std::span<const uint8_t> rgba, int width, int height);

/// Writes `rgba`, `width` by `height` tightly packed, to `path` as a PNG. Returns false when the
/// size doesn't match the data or the file can't be written. The folder must exist.
[[nodiscard]] bool WritePng(const std::filesystem::path& path, std::span<const uint8_t> rgba,
                            int width, int height);

} // namespace Libs::Graphics::SystemUi::Gallery

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_GALLERY_CAPTURE_H_
