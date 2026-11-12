#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_IMAGECONVERSION_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_IMAGECONVERSION_H_

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"

#include <cstdint>

// Which image conversion formats the SPIR-V emitter can load and store through. The emitter and the
// prescan's descriptor validation both ask here, so a snapshot the validation passes is one the
// emitter accepts. It is inline rather than constexpr because RemapTextureFormat is defined out of
// line in gpu_format.cpp.
namespace Libs::Graphics::ShaderRecompiler::IR {

/// The most components a converted image texel can have.
inline constexpr uint32_t MAX_IMAGE_CONVERSION_COMPONENTS = 4;

/// Whether an image access reads texels or writes them, since stores accept fewer formats.
enum class ImageConversionUse : uint8_t { Load, Store };

/// Gives the record the emitter converts an image's texels through, or an empty record when the image
/// has no conversion format.
[[nodiscard]] constexpr Format::BufferFormatInfo ImageConversionInfo(const ImageResource& image) {
	return image.conversion_format == Prospero::BufferFormat::kInvalid
	           ? Format::BufferFormatInfo {}
	           : Format::GetFormatInfo(image.conversion_format);
}

/// Decides whether the emitter accepts an image's conversion format for `use`. An image with no
/// conversion format is always accepted.
/// - A load needs a format the host remaps, unless the image is a raw signed storage image or reads
///   depth bits, with one to four components. It must also be Uscaled or a packed 32 bit Uint or
///   Unorm record, again unless it is raw signed or reads depth bits.
/// - A store also needs a Sint format or a packed Uint or Unorm record.
[[nodiscard]] inline bool ImageConversionAccepted(const ImageResource& image, ImageConversionUse use) {
	using Format::ComponentType;
	const auto format = image.conversion_format;
	if (format == Prospero::BufferFormat::kInvalid) {
		return true;
	}
	const auto info = Format::GetFormatInfo(format);
	if (!ReadsDepthBits(image)) {
		const bool raw_signed = RawSignedStorageFormat(image, format) != Prospero::BufferFormat::kInvalid;
		const bool remapped   = Prospero::RemapTextureFormat(format) != format;
		if ((!remapped && !raw_signed) || info.component_count == 0u ||
		    info.component_count > MAX_IMAGE_CONVERSION_COMPONENTS) {
			return false;
		}
		const bool packed_word = (info.type == ComponentType::Uint || info.type == ComponentType::Unorm) &&
		                         info.packed_bitfield && info.byte_size == sizeof(uint32_t);
		if (!raw_signed && info.type != ComponentType::Uscaled && !packed_word) {
			return false;
		}
	}
	if (use == ImageConversionUse::Store) {
		return info.type == ComponentType::Sint ||
		       (info.packed_bitfield && (info.type == ComponentType::Uint || info.type == ComponentType::Unorm));
	}
	return true;
}

} // namespace Libs::Graphics::ShaderRecompiler::IR

#endif // EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_IR_IMAGECONVERSION_H_
