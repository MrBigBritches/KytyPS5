#ifndef GRAPHICS_SHADER_STOREGEN_HEADER_SLOTS_H
#define GRAPHICS_SHADER_STOREGEN_HEADER_SLOTS_H

#include "graphics/shader/storegen/sl00Metadata.h"

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// Decodes the sharp slot tables of an AGC shader header without a graphics device and without
// reading Config. It is a port of header_tables and slot_of in the python predictor.
namespace Libs::Graphics::StoreGen {

/// Offsets below this are user SGPRs. From here up they are dwords of the extended user data
/// table, so the offset minus this is the EUD dword.
constexpr uint32_t EUD_FIRST_DWORD = 32;

/// A slot of a sharp table, such as T3 or B0.
struct SlotRef {
	/// The class of the table. Its values are in the order of the header's four tables.
	Sl00Class sharp_class = Sl00Class::ReadOnly;
	/// The slot's index in its table. The sl00 record of a resource names the same number.
	uint32_t index = 0;

	bool operator==(const SlotRef&) const = default;
};

/// Where each used slot lives, and where the extended user data pointer is.
struct HeaderSlots {
	/// One used slot and the user data offset that holds its descriptor.
	struct Entry {
		SlotRef  slot;
		uint32_t offset = 0; ///< Below EUD_FIRST_DWORD a user SGPR, otherwise EUD dword + 32.
	};
	std::vector<Entry> entries;
	/// The user SGPR holding the extended user data pointer, from direct entry 5.
	std::optional<uint32_t> eud_pointer;
};

/// Decodes the slot tables of a shader header.
/// Every offset and count read from the header is checked against it.
/// @param header the .shader_header section.
/// @return nothing when the table runs outside the header. A header with no user data table gives
///         no slots and no EUD pointer.
std::optional<HeaderSlots> DecodeHeaderSlots(std::span<const uint8_t> header);

/// The slot that holds the descriptor at a user data offset.
/// @return nothing when no slot or more than one slot holds that offset.
std::optional<SlotRef> SlotAtOffset(const HeaderSlots& slots, uint32_t offset);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_HEADER_SLOTS_H
