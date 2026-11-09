#ifndef GRAPHICS_SHADER_STOREGEN_SL00_METADATA_H
#define GRAPHICS_SHADER_STOREGEN_SL00_METADATA_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Decodes the resource records of an sl00 `.metadata` section without a graphics device and without
// reading Config, so the shader store worker process can run it. It is a port of
// metadata_resources in the python predictor, with the same results on every blob that python
// decodes.
namespace Libs::Graphics::StoreGen {

/// The metadata sizes this decoder accepts. Real sections are a few kilobytes.
constexpr size_t SL00_MAX_METADATA_BYTES = 1u << 20;

/// The most resource records one section may yield. Real compute programs have tens.
constexpr size_t SL00_MAX_RESOURCES = 1024;

/// The work a decode may do is a budget of this many bytes read per byte of metadata, plus a base
/// for tiny sections. Every byte the decoder reads and every attribute it steps over is charged, so
/// the budget bounds the time a hostile section can cost. A section that exhausts it gets no
/// value. Real sections use a small fraction of it.
constexpr uint64_t SL00_WORK_PER_BYTE = 64;
constexpr uint64_t SL00_WORK_BASE     = 4096;

/// What a resource's type record says it is. The predictor must leave out both Other and
/// Unreadable, as python's own list of buffer kinds does.
enum class Sl00ShapeKind {
	/// The type record could not be read, because it runs off the section's end, or because its
	/// abbreviation lies outside the section.
	Unreadable,
	/// A type record that is none of the kinds below. `flag` tells which flag it carried.
	Other,
	/// A constant buffer, marked by attribute 30.
	ConstantBuffer,
	/// A byte address buffer, with flag 2.
	Raw,
	/// A formatted buffer, with flag 0. `element_kind` and `element_count` describe one element.
	Typed,
	/// A structured buffer, with flag 1. `stride` is the element size in bytes.
	Structured,
	/// An image, with flag 7, 8 or 13.
	Image,
};

/// The shape of a resource, which decides how the predictor describes it. Only the fields python's
/// shape tuple carries for the kind are set. Typed has the element kind and count, Structured has
/// the stride, and Image and Other have the flag, element kind and element count.
struct Sl00Shape {
	Sl00ShapeKind kind = Sl00ShapeKind::Unreadable;
	/// The type record's attribute 41. It is 1 when that attribute is a form 80 flag, as in python.
	std::optional<uint64_t> flag;
	/// The element record's attribute 41, the scalar kind. 0 or 1 is float, 5 is int, 9 is uint.
	std::optional<uint64_t> element_kind;
	/// The element record's attribute 43, the component count.
	std::optional<uint64_t> element_count;
	/// The element size in bytes, set for Structured only.
	uint64_t stride = 0;
};

/// The sharp table class of a resource. The first four values are in the order of python's
/// CLASSES = "TUSB", so a value is also an index into that string.
enum class Sl00Class : uint8_t {
	ReadOnly       = 0, ///< T, a read-only texture or buffer.
	ReadWrite      = 1, ///< U, a read-write resource.
	Sampler        = 2, ///< S. The python decoder never gives it.
	ConstantBuffer = 3, ///< B.
	Unknown        = 4, ///< The type record or its abbreviation could not be read.
};

/// The letter python uses for a class, or 0 for Unknown.
constexpr char Sl00ClassLetter(Sl00Class resource_class) {
	constexpr char LETTERS[] = "TUSB";
	return resource_class == Sl00Class::Unknown ? '\0'
	                                            : LETTERS[static_cast<size_t>(resource_class)];
}

/// One resource of a program.
struct Sl00Resource {
	std::string name;
	/// The byte just before the record's name reference, which is the slot's register.
	uint8_t register_index = 0;
	/// Where the type record's abbreviation is. It is empty when reading the type record raised. It
	/// may lie outside the section when the record's abbreviation does. Python calls this the tag.
	std::optional<uint64_t> abbreviation_offset;
	Sl00Shape               shape;
	/// Where the type record starts in the section.
	uint64_t type_offset = 0;
	/// B for a constant buffer, U for a read-write resource and T for the rest.
	Sl00Class resource_class = Sl00Class::Unknown;
};

/// How much of its budget a decode used.
struct Sl00DecodeStats {
	uint64_t work_used   = 0;
	uint64_t work_budget = 0;
};

/// Decodes the resource records of an sl00 `.metadata` section.
/// Every offset and length read from the blob is checked against it, so a truncated or hostile
/// blob never reads out of range. The records come last first, as the python decoder lists them.
/// @param metadata the whole `.metadata` section.
/// @param stats if given, receives the work used and the budget. Both are 0 when the section is
///              refused for its size before any work starts.
/// @return the records, which are empty when the section has no sl00 string table. It has no
/// value when the section is larger than SL00_MAX_METADATA_BYTES, yields more than
/// SL00_MAX_RESOURCES records or exhausts its work budget.
std::optional<std::vector<Sl00Resource>> DecodeSl00Resources(std::span<const uint8_t> metadata,
                                                             Sl00DecodeStats* stats = nullptr);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_SL00_METADATA_H
