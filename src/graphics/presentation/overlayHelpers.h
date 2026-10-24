#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_OVERLAYHELPERS_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_OVERLAYHELPERS_H_

#include "imgui.h"

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace Libs::Graphics {

// Word-wraps text in the current ImGui font at `font_size` to lines no wider than `width`,
// breaking at spaces and at newlines. A word wider than the line keeps a line of its own.
std::vector<std::string_view> WrapText(std::string_view text, float font_size, float width);

// Decodes a PNG into an RGBA texture registered with ImGui, or null (logged with `what`) if it
// is not a PNG or is larger than `max_size` pixels on a side.
std::unique_ptr<ImTextureData> LoadPngTexture(std::span<const std::byte> png, int max_size,
                                              const char* what);

} // namespace Libs::Graphics

#endif /* EMULATOR_SRC_GRAPHICS_PRESENTATION_OVERLAYHELPERS_H_ */
