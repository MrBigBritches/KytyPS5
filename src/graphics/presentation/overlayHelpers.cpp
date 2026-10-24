#include "graphics/presentation/overlayHelpers.h"

#include "common/logging/log.h"
#include "imgui_internal.h"
#include "stb_image.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace Libs::Graphics {

std::vector<std::string_view> WrapText(std::string_view text, float font_size, float width) {
	std::vector<std::string_view> lines;
	auto* const                  font = ImGui::GetFont();
	for (;;) {
		const auto newline   = text.find('\n');
		const auto paragraph = text.substr(0, newline);
		auto       line      = paragraph.substr(0, 0);
		for (auto start = paragraph.find_first_not_of(" \t\r"); start != std::string_view::npos;) {
			const auto end  = std::min(paragraph.find_first_of(" \t\r", start), paragraph.size());
			const auto word = paragraph.substr(start, end - start);
			const std::string_view candidate(line.empty() ? word.begin() : line.begin(),
			                                 word.end());
			// Keep an oversized word intact; the caller clips or lets it overflow.
			if (!line.empty() && font->CalcTextSizeA(font_size, std::numeric_limits<float>::max(),
			                                       0.0f, candidate.data(),
			                                       candidate.data() + candidate.size()).x > width) {
				lines.push_back(line);
				line = word;
			} else {
				line = candidate;
			}
			start = paragraph.find_first_not_of(" \t\r", end);
		}
		lines.push_back(line);
		if (newline == std::string_view::npos) {
			return lines;
		}
		text.remove_prefix(newline + 1);
	}
}

std::unique_ptr<ImTextureData> LoadPngTexture(std::span<const std::byte> png, int max_size,
                                              const char* what) {
	if (png.empty() || png.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
		return nullptr;
	}
	const auto* data   = reinterpret_cast<const stbi_uc*>(png.data());
	const int   size   = static_cast<int>(png.size());
	int         width  = 0;
	int         height = 0;
	if (!stbi_info_from_memory(data, size, &width, &height, nullptr) || width <= 0 ||
	    height <= 0 || width > max_size || height > max_size) {
		LOGF("%s: not an image, or larger than %dx%d\n", what, max_size, max_size);
		return nullptr;
	}
	auto* pixels = stbi_load_from_memory(data, size, &width, &height, nullptr, 4);
	if (pixels == nullptr) {
		LOGF("%s: could not decode: %s\n", what, stbi_failure_reason());
		return nullptr;
	}
	auto texture = std::make_unique<ImTextureData>();
	texture->Create(ImTextureFormat_RGBA32, width, height);
	std::memcpy(texture->Pixels, pixels,
	            static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
	texture->UseColors = true;
	stbi_image_free(pixels);
	ImGui::RegisterUserTexture(texture.get());
	return texture;
}

} // namespace Libs::Graphics
