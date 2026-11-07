#include "graphics/presentation/system/gallery/capture.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <string_view>
#include <system_error>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#define STBIWDEF static inline
#define STBI_WRITE_NO_STDIO
#include "stb_image_write.h"

namespace Libs::Graphics::SystemUi::Gallery {
namespace {

constexpr size_t BYTES_PER_PIXEL = 4;
constexpr size_t RED_BYTE        = 0;
constexpr size_t BLUE_BYTE       = 2;
// stb's stride argument for PNG is in bytes, and RGBA has four components.
constexpr int RGBA_COMPONENTS = 4;

// Appends `text` in lower case to `out`.
void AppendLower(std::string& out, std::string_view text) {
	for (const char letter: text) {
		out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(letter))));
	}
}

// Receives stb's output in pieces and appends it to the bytes in `context`.
void AppendChunk(void* context, void* data, int size) {
	auto*       bytes = static_cast<std::vector<std::byte>*>(context);
	const auto* first = static_cast<const std::byte*>(data);
	bytes->insert(bytes->end(), first, first + size);
}

// Receives stb's output in pieces and appends it to the stream in `context`.
void WriteChunk(void* context, void* data, int size) {
	static_cast<std::ofstream*>(context)->write(static_cast<const char*>(data), size);
}

} // namespace

std::vector<uint8_t> ToRgba(std::span<const uint8_t> pixels, int width, int height,
                            size_t row_pitch, bool bgra) {
	if (width <= 0 || height <= 0) {
		return {};
	}
	const size_t row_bytes = static_cast<size_t>(width) * BYTES_PER_PIXEL;
	const size_t rows      = static_cast<size_t>(height);
	// The last row needs only its own bytes, not the padding after it.
	if (row_pitch < row_bytes || pixels.size() < row_pitch * (rows - 1) + row_bytes) {
		return {};
	}
	std::vector<uint8_t> rgba(row_bytes * rows);
	for (size_t row = 0; row < rows; row++) {
		const uint8_t* source = pixels.data() + row * row_pitch;
		uint8_t*       target = rgba.data() + row * row_bytes;
		for (size_t offset = 0; offset < row_bytes; offset += BYTES_PER_PIXEL) {
			for (size_t channel = 0; channel < BYTES_PER_PIXEL; channel++) {
				target[offset + channel] = source[offset + channel];
			}
			if (bgra) {
				target[offset + RED_BYTE]  = source[offset + BLUE_BYTE];
				target[offset + BLUE_BYTE] = source[offset + RED_BYTE];
			}
		}
	}
	return rgba;
}

std::string ScreenshotName(const PageEntry& page, int view) {
	std::string name;
	AppendLower(name, GroupTitle(page.group));
	name.push_back('-');
	AppendLower(name, page.id);
	if (view > 0 && view < static_cast<int>(page.view_names.size())) {
		name.push_back('-');
		AppendLower(name, page.view_names[static_cast<size_t>(view)]);
	}
	name += ".png";
	return name;
}

std::vector<ScreenshotJob> ScreenshotJobs(std::span<const PageEntry> pages) {
	std::vector<ScreenshotJob> jobs;
	jobs.reserve(pages.size());
	for (size_t index = 0; index < pages.size(); index++) {
		const int views = std::max(1, static_cast<int>(pages[index].view_names.size()));
		for (int view = 0; view < views; view++) {
			jobs.push_back({static_cast<int>(index), view, ScreenshotName(pages[index], view)});
		}
	}
	return jobs;
}

std::vector<std::byte> EncodePng(std::span<const uint8_t> rgba, int width, int height) {
	std::vector<std::byte> bytes;
	if (width <= 0 || height <= 0 ||
	    rgba.size() != static_cast<size_t>(width) * static_cast<size_t>(height) * BYTES_PER_PIXEL) {
		return bytes;
	}
	(void)stbi_write_png_to_func(AppendChunk, &bytes, width, height, RGBA_COMPONENTS, rgba.data(),
	                             width * RGBA_COMPONENTS);
	return bytes;
}

bool WritePng(const std::filesystem::path& path, std::span<const uint8_t> rgba, int width,
              int height) {
	if (width <= 0 || height <= 0 ||
	    rgba.size() != static_cast<size_t>(width) * static_cast<size_t>(height) * BYTES_PER_PIXEL) {
		return false;
	}
	// Writing through a stream, not stb's own fopen, keeps non-ASCII folder names working on
	// Windows.
	std::ofstream file(path, std::ios::binary | std::ios::trunc);
	if (!file) {
		return false;
	}
	const int  row_bytes = width * RGBA_COMPONENTS;
	const bool encoded   = stbi_write_png_to_func(WriteChunk, &file, width, height, RGBA_COMPONENTS,
	                                              rgba.data(), row_bytes) != 0;
	file.close();
	if (!encoded || file.fail()) {
		// A half-written PNG would pass for a screenshot, so it is removed.
		std::error_code ignored;
		std::filesystem::remove(path, ignored);
		return false;
	}
	return true;
}

} // namespace Libs::Graphics::SystemUi::Gallery
