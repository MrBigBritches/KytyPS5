#ifndef GRAPHICS_SHADER_STOREGEN_SHADER_HEADER_LAYOUT_H
#define GRAPHICS_SHADER_STOREGEN_SHADER_HEADER_LAYOUT_H

#include <cstdint>
#include <cstring>
#include <span>
#include <type_traits>

// Where the fields the store generator reads sit in the AGC shader header, which is the
// Libs::Graphics::Shader struct with its pointers stored as offsets relative to their own field.
namespace Libs::Graphics::StoreGen::HeaderLayout {

constexpr uint32_t FILE_MAGIC              = 0x34333231;
constexpr uint32_t FILE_VERSION            = 0x00000018;
constexpr uint64_t OFFSET_MAGIC            = 0x00;
constexpr uint64_t OFFSET_VERSION          = 0x04;
constexpr uint64_t OFFSET_SH_REGISTERS     = 0x20;
constexpr uint64_t OFFSET_SHADER_SIZE      = 0x44;
constexpr uint64_t OFFSET_SCRATCH_DWORDS   = 0x54;
constexpr uint64_t OFFSET_TYPE             = 0x5A;
constexpr uint64_t OFFSET_NUM_SH_REGISTERS = 0x5C;
// The user data table pointer, stored relative to its own field.
constexpr uint64_t OFFSET_USER_DATA = 0x08;
// Inside the user data table: five relative pointers (the direct table, then one per sharp class),
// the direct entry count, and four sharp slot counts.
constexpr uint64_t USER_DIRECT_POINTER = 0x00;
constexpr uint64_t USER_SHARP_POINTERS = 0x08;
constexpr uint64_t USER_POINTER_SIZE   = 8;
constexpr uint64_t USER_DIRECT_COUNT   = 0x2C;
constexpr uint64_t USER_SHARP_COUNTS   = 0x2E;
constexpr uint64_t SHARP_CLASS_COUNT   = 4;
// A table entry is 16 bits. This value means the slot is unused, and the low 15 bits are the
// offset.
constexpr uint16_t SLOT_UNUSED      = 0xFFFF;
constexpr uint16_t SLOT_OFFSET_MASK = 0x7FFF;
// The direct entry that holds the extended user data pointer.
constexpr uint64_t DIRECT_EUD_ENTRY = 5;
// The smallest header that holds every field above.
constexpr uint64_t MIN_HEADER_SIZE = 0x5D;
// Shader binary type 0 is a compute program. Every other value is a graphics stage.
constexpr uint8_t TYPE_COMPUTE = 0;
// One SH register entry is an offset and a value.
constexpr uint64_t SH_REGISTER_SIZE = 8;

/// Reads a trivially copyable value at a byte offset, or fails when it would run past the end.
/// @return false when offset + sizeof(T) exceeds the span.
template <typename T>
[[nodiscard]] bool Read(std::span<const uint8_t> bytes, uint64_t offset, T& out) {
	static_assert(std::is_trivially_copyable_v<T>);
	if (offset > bytes.size() || sizeof(T) > bytes.size() - offset) {
		return false;
	}
	std::memcpy(&out, bytes.data() + offset, sizeof(T));
	return true;
}

} // namespace Libs::Graphics::StoreGen::HeaderLayout

#endif // GRAPHICS_SHADER_STOREGEN_SHADER_HEADER_LAYOUT_H
