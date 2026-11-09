#include "graphics/shader/storegen/headerSlots.h"

#include "graphics/shader/storegen/shaderHeaderLayout.h"

namespace Libs::Graphics::StoreGen {

namespace {

namespace Layout = HeaderLayout;

constexpr uint64_t BYTES_PER_ENTRY = sizeof(uint16_t);

// Reads the direct entry that holds the EUD pointer. It is absent when the table is shorter or the
// entry is unused.
std::optional<uint32_t> ReadEudPointer(std::span<const uint8_t> header, uint64_t user) {
	uint64_t direct_relative = 0;
	uint16_t direct_count    = 0;
	if (!Layout::Read(header, user + Layout::USER_DIRECT_POINTER, direct_relative) ||
	    !Layout::Read(header, user + Layout::USER_DIRECT_COUNT, direct_count)) {
		return std::nullopt;
	}
	if (direct_relative == 0 || direct_count <= Layout::DIRECT_EUD_ENTRY) {
		return std::nullopt;
	}
	uint16_t entry = 0;
	if (!Layout::Read(header, user + direct_relative + Layout::DIRECT_EUD_ENTRY * BYTES_PER_ENTRY,
	                  entry) ||
	    entry == Layout::SLOT_UNUSED) {
		return std::nullopt;
	}
	return entry;
}

} // namespace

std::optional<HeaderSlots> DecodeHeaderSlots(std::span<const uint8_t> header) {
	HeaderSlots result;
	uint64_t    user_relative = 0;
	if (!Layout::Read(header, Layout::OFFSET_USER_DATA, user_relative)) {
		return std::nullopt;
	}
	if (user_relative == 0) {
		return result;
	}
	// Unsigned wrap-around lets a table that precedes its field work, and a wild value fails the
	// range checks below.
	const uint64_t user = Layout::OFFSET_USER_DATA + user_relative;
	result.eud_pointer  = ReadEudPointer(header, user);
	for (uint64_t table = 0; table < Layout::SHARP_CLASS_COUNT; table++) {
		uint16_t count    = 0;
		uint64_t relative = 0;
		if (!Layout::Read(header, user + Layout::USER_SHARP_COUNTS + table * BYTES_PER_ENTRY,
		                  count) ||
		    !Layout::Read(header,
		                  user + Layout::USER_SHARP_POINTERS + table * Layout::USER_POINTER_SIZE,
		                  relative)) {
			return std::nullopt;
		}
		// The table's pointer is relative to its own field.
		const uint64_t at =
		    user + Layout::USER_SHARP_POINTERS + table * Layout::USER_POINTER_SIZE + relative;
		for (uint32_t slot = 0; slot < count; slot++) {
			uint16_t raw = 0;
			if (!Layout::Read(header, at + slot * BYTES_PER_ENTRY, raw)) {
				return std::nullopt;
			}
			if (raw != Layout::SLOT_UNUSED) {
				result.entries.push_back({{static_cast<Sl00Class>(table), slot},
				                          static_cast<uint32_t>(raw & Layout::SLOT_OFFSET_MASK)});
			}
		}
	}
	return result;
}

std::optional<SlotRef> SlotAtOffset(const HeaderSlots& slots, uint32_t offset) {
	std::optional<SlotRef> match;
	for (const auto& entry: slots.entries) {
		if (entry.offset != offset) {
			continue;
		}
		if (match) {
			return std::nullopt; // Two slots share the offset, so it names neither.
		}
		match = entry.slot;
	}
	return match;
}

} // namespace Libs::Graphics::StoreGen
