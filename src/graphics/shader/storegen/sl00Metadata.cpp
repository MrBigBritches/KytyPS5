#include "graphics/shader/storegen/sl00Metadata.h"

#include <algorithm>
#include <array>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace Libs::Graphics::StoreGen {

namespace {

// ---- Attribute and form numbers of the sl00 abbreviations. ----
constexpr uint64_t                FORM_FLAG                 = 80;
constexpr uint64_t                ATTRIBUTE_ELEMENT         = 36;
constexpr uint64_t                ATTRIBUTE_KIND            = 41;
constexpr uint64_t                ATTRIBUTE_COUNT           = 43;
constexpr uint64_t                ATTRIBUTE_MEMBER_NAME     = 30;
constexpr uint64_t                ATTRIBUTE_MEMBER_OFFSET   = 33;
constexpr uint64_t                ATTRIBUTE_MEMBER_SIZE     = 37;
constexpr uint64_t                ATTRIBUTE_CONSTANT_BUFFER = 30;
constexpr uint64_t                ATTRIBUTE_READ_WRITE      = 40;
constexpr uint64_t                FLAG_TYPED                = 0;
constexpr uint64_t                FLAG_STRUCTURED           = 1;
constexpr uint64_t                FLAG_RAW                  = 2;
constexpr std::array<uint64_t, 3> FLAG_IMAGES {7, 8, 13};
// A form 80 attribute carries no data. Python stores it as True, which compares equal to 1.
constexpr uint64_t FLAG_FORM_VALUE = 1;
constexpr uint64_t DWORD_BYTES     = 4;

// ---- Resource record scan. ----
// Record rows closer together than this belong to the same run at the end of the stream.
constexpr size_t  MAX_ROW_GAP     = 16;
constexpr size_t  SCAN_END_MARGIN = 4;
constexpr uint8_t REGISTER_LIMIT  = 0x80;
constexpr uint8_t PRINTABLE_FIRST = 32;
constexpr uint8_t PRINTABLE_LAST  = 126;

// ---- Variable length integers. ----
constexpr uint8_t  ULEB_CONTINUE = 0x80;
constexpr uint8_t  ULEB_PAYLOAD  = 0x7F;
constexpr unsigned ULEB_BITS     = 7;
constexpr unsigned VALUE_BITS    = 64;
// A value of 64 bits fits in this many bytes. Later bytes may only be padding.
constexpr size_t ULEB_VALUE_BYTES = 10;

constexpr std::string_view VERSION_SUFFIX = " build ";
constexpr size_t           VERSION_GROUPS = 4;

/// One decoded variable length integer and where the next field starts.
struct Uleb {
	uint64_t value;
	size_t   next;
};

/// Reads variable length integers from a blob in constant time per call, however long the run of
/// continuation bytes is. The python decoder tries one at every byte, so a blob of nothing but
/// continuation bytes would cost quadratic time without the two tables built here.
class Reader {
public:
	/// Builds the tables for `data`, which must outlive the reader, with a work budget that is
	/// proportional to its size.
	explicit Reader(std::span<const uint8_t> data)
	    : m_data(data), m_budget(SL00_WORK_BASE + SL00_WORK_PER_BYTE * data.size()) {
		const size_t size = data.size();
		m_run_end.assign(size + 1, static_cast<uint32_t>(size));
		m_nonzero_prefix.assign(size + 1, 0);
		for (size_t index = size; index-- > 0;) {
			m_run_end[index] = (data[index] & ULEB_CONTINUE) == 0 ? static_cast<uint32_t>(index)
			                                                      : m_run_end[index + 1];
		}
		for (size_t index = 0; index < size; index++) {
			m_nonzero_prefix[index + 1] =
			    m_nonzero_prefix[index] + ((data[index] & ULEB_PAYLOAD) != 0);
		}
	}

	[[nodiscard]] size_t                   Size() const { return m_data.size(); }
	[[nodiscard]] std::span<const uint8_t> Data() const { return m_data; }
	[[nodiscard]] uint64_t                 WorkUsed() const { return m_used; }
	[[nodiscard]] uint64_t                 WorkBudget() const { return m_budget; }
	/// Whether the work budget ran out, after which every read fails.
	[[nodiscard]] bool Exhausted() const { return m_exhausted; }

	/// Charges `units` of work. It fails, and keeps failing, once the budget would be passed.
	bool Charge(uint64_t units) const {
		if (m_exhausted || units > m_budget - m_used) {
			m_exhausted = true;
			return false;
		}
		m_used += units;
		return true;
	}

	/// Reads an unsigned LEB128 value at `at`. It fails where python raises IndexError, which is
	/// when `at` is outside the blob or the value runs off its end. A value that does not fit 64
	/// bits is clamped to the maximum, which no offset or table key can equal. Each byte of the
	/// value that is looked at is charged to the work budget, and a read fails when the budget is
	/// gone.
	[[nodiscard]] std::optional<Uleb> Read(uint64_t at) const {
		if (at >= m_data.size()) return std::nullopt;
		const size_t end = m_run_end[at];
		if (end >= m_data.size()) return std::nullopt;
		// Only the first ten bytes can set bits of a 64-bit value. Any later byte with a payload
		// overflows, and the prefix count finds that without visiting the byte.
		const size_t value_end = std::min<size_t>(end, at + ULEB_VALUE_BYTES - 1);
		if (!Charge(value_end - at + 1)) return std::nullopt;
		uint64_t value    = 0;
		bool     overflow = false;
		unsigned shift    = 0;
		// At most ten bytes are visited, so the shift stays below 64.
		for (size_t index = at; index <= value_end; index++, shift += ULEB_BITS) {
			const uint64_t payload = m_data[index] & ULEB_PAYLOAD;
			value |= payload << shift;
			overflow = overflow || (shift > 0 && (payload >> (VALUE_BITS - shift)) != 0);
		}
		overflow = overflow || m_nonzero_prefix[end + 1] != m_nonzero_prefix[value_end + 1];
		return Uleb {overflow ? std::numeric_limits<uint64_t>::max() : value, end + 1};
	}

private:
	std::span<const uint8_t> m_data;
	uint64_t                 m_budget;
	mutable uint64_t         m_used      = 0;
	mutable bool             m_exhausted = false;
	// For each byte, the index of the first byte at or after it that ends a variable length
	// integer.
	std::vector<uint32_t> m_run_end;
	// How many bytes before each index carry payload bits.
	std::vector<uint32_t> m_nonzero_prefix;
};

// ---- String table. ----

/// The sl00 strings by their offset, and where the record stream starts after them.
struct StringTable {
	std::unordered_map<uint64_t, std::string> strings;
	size_t                                    records_start = 0;
};

/// Whether a byte is an ASCII digit, which is all that python's `\d` matches on bytes.
bool IsDigit(uint8_t byte) {
	return byte >= '0' && byte <= '9';
}

/// Whether `data` has `d+.d+.d+.d+ build ` at `at`. A digit run is always taken whole, since the
/// byte that ends it is never a digit, so no backtracking is needed.
bool MatchesVersionAt(std::span<const uint8_t> data, size_t at) {
	for (size_t group = 0; group < VERSION_GROUPS; group++) {
		const size_t digits_start = at;
		while (at < data.size() && IsDigit(data[at]))
			at++;
		if (at == digits_start) return false;
		if (group + 1 < VERSION_GROUPS) {
			if (at >= data.size() || data[at] != '.') return false;
			at++;
		}
	}
	if (data.size() - at < VERSION_SUFFIX.size()) return false;
	return std::equal(VERSION_SUFFIX.begin(), VERSION_SUFFIX.end(), data.begin() + at);
}

/// The leftmost place the version text starts. A start in the middle of a digit run matches only
/// if the start of the run does, so those starts are skipped and the search stays linear.
std::optional<size_t> FindVersionText(std::span<const uint8_t> data) {
	for (size_t at = 0; at < data.size(); at++) {
		if (!IsDigit(data[at]) || (at > 0 && IsDigit(data[at - 1]))) continue;
		if (MatchesVersionAt(data, at)) return at;
	}
	return std::nullopt;
}

/// Reads the NUL terminated printable strings that start at the version text. The table ends at
/// the first empty or unprintable string, which is where the record stream starts. With no
/// version text there are no strings and the stream starts at the end.
StringTable ReadStringTable(std::span<const uint8_t> data) {
	StringTable table;
	table.records_start = data.size();
	const auto version  = FindVersionText(data);
	if (!version) return table;
	size_t at = *version;
	while (true) {
		const auto end =
		    std::find(data.begin() + std::min(at, data.size()), data.end(), uint8_t {0});
		const size_t end_index = static_cast<size_t>(end - data.begin());
		const bool   printable =
		    std::all_of(data.begin() + std::min(at, data.size()), end, [](uint8_t byte) {
			    return byte >= PRINTABLE_FIRST && byte <= PRINTABLE_LAST;
		    });
		if (end == data.end() || end_index == at || !printable) {
			table.records_start = at;
			return table;
		}
		table.strings[at].assign(data.begin() + at, end);
		at = end_index + 1;
	}
}

// ---- Abbreviations and records. ----

using AttributeForm = std::pair<uint64_t, uint64_t>;

/// An abbreviation, which is a tag and the attribute and form of each field a record has.
struct Abbreviation {
	uint64_t                   tag     = 0;
	bool                       tag_odd = false;
	std::vector<AttributeForm> pairs;
};

/// Reads the abbreviation at `offset`, which is the tag and then attribute and form pairs up to a
/// 0, 0 pair. Python stops early when fewer than two bytes remain, and so does this. It fails where
/// python raises IndexError.
std::optional<Abbreviation> ReadAbbreviation(const Reader& reader, uint64_t offset) {
	const auto tag = reader.Read(offset);
	if (!tag) return std::nullopt;
	Abbreviation abbreviation;
	abbreviation.tag = tag->value;
	// The low bit is taken from the byte, because a clamped value no longer has it.
	abbreviation.tag_odd = (reader.Data()[offset] & 1) != 0;
	size_t at            = tag->next;
	while (at + 1 < reader.Size()) {
		const auto attribute = reader.Read(at);
		if (!attribute) return std::nullopt;
		const auto form = reader.Read(attribute->next);
		if (!form) return std::nullopt;
		at = form->next;
		if (attribute->value == 0 && form->value == 0) break;
		abbreviation.pairs.emplace_back(attribute->value, form->value);
	}
	return abbreviation;
}

/// A decoded record, which is an abbreviation's attributes with their values.
struct Die {
	std::vector<uint64_t>                      attributes;
	std::vector<std::pair<uint64_t, uint64_t>> values; // In order. A repeated attribute wins last.
	std::optional<uint64_t>                    link;   // Set for an odd tag.

	[[nodiscard]] bool HasAttribute(uint64_t attribute) const {
		return std::find(attributes.begin(), attributes.end(), attribute) != attributes.end();
	}

	/// The value of an attribute, where the last one listed wins as in a python dict.
	[[nodiscard]] std::optional<uint64_t> Value(uint64_t attribute) const {
		for (auto it = values.rbegin(); it != values.rend(); ++it) {
			if (it->first == attribute) return it->second;
		}
		return std::nullopt;
	}
};

/// The outcome of reading a record. Failed is where python raises, Absent is where it returns None.
enum class DieStatus { Failed, Absent, Present };

/// A record and how reading it went. The record is only meaningful when Present.
struct DieRead {
	DieStatus status = DieStatus::Failed;
	Die       die;
};

/// Decodes a record. Failed is where python raises IndexError. Absent is where python returns None,
/// which is when the abbreviation offset is outside the blob. Each value is one variable length
/// integer, except that a form 80 attribute has no data and reads as 1.
DieRead ReadDie(const Reader& reader, uint64_t at) {
	const auto code = reader.Read(at);
	if (!code) return {};
	if (code->value >= reader.Size()) return {DieStatus::Absent, {}};
	const auto abbreviation = ReadAbbreviation(reader, code->value);
	if (!abbreviation) return {};
	DieRead result;
	size_t  next = code->next;
	// A record with an odd tag carries a link to a sibling before its attributes.
	if (abbreviation->tag_odd) {
		const auto link = reader.Read(next);
		if (!link) return {};
		result.die.link = link->value;
		next            = link->next;
	}
	for (const auto& [attribute, form]: abbreviation->pairs) {
		// A form 80 attribute reads no bytes, so each pair is charged on its own.
		if (!reader.Charge(1)) return {};
		result.die.attributes.push_back(attribute);
		if (form == FORM_FLAG) {
			result.die.values.emplace_back(attribute, FLAG_FORM_VALUE);
			continue;
		}
		const auto value = reader.Read(next);
		if (!value) return {};
		result.die.values.emplace_back(attribute, value->value);
		next = value->next;
	}
	result.status = DieStatus::Present;
	return result;
}

/// The sum of two values, or the maximum when it would not fit.
uint64_t SaturatingAdd(uint64_t left, uint64_t right) {
	return right > std::numeric_limits<uint64_t>::max() - left
	           ? std::numeric_limits<uint64_t>::max()
	           : left + right;
}

/// The byte size of a struct element, which is the largest member offset plus size. The members
/// are walked from the struct's attribute 30 through each member's sibling link, and the walk stops
/// at a member with no size, with no link, outside the blob or already seen. A member whose
/// abbreviation is outside the blob has no size either. It fails where python
/// raises IndexError, which includes an element offset outside the blob.
std::optional<uint64_t> StructSize(const Reader& reader, uint64_t element_at) {
	const auto element = ReadDie(reader, element_at);
	if (element.status == DieStatus::Failed) return std::nullopt;
	std::optional<uint64_t>      at   = element.status == DieStatus::Present
	                                        ? element.die.Value(ATTRIBUTE_MEMBER_NAME)
	                                        : std::nullopt;
	uint64_t                     size = 0;
	std::unordered_set<uint64_t> seen;
	while (at && *at < reader.Size() && !seen.contains(*at)) {
		seen.insert(*at);
		const auto member = ReadDie(reader, *at);
		if (member.status == DieStatus::Failed) return std::nullopt;
		const auto member_size = member.die.Value(ATTRIBUTE_MEMBER_SIZE);
		if (!member_size) break;
		const uint64_t member_end =
		    SaturatingAdd(member.die.Value(ATTRIBUTE_MEMBER_OFFSET).value_or(0), *member_size);
		size = std::max(size, member_end);
		at   = member.die.link;
	}
	return size;
}

/// The size of one structured buffer element, from the scalar element when it has a kind and a
/// count and from the struct's members when it does not. A count of zero counts as none.
std::optional<uint64_t> StructuredStride(const Reader& reader, const Sl00Shape& shape,
                                         std::optional<uint64_t> element_at) {
	if (shape.element_kind && shape.element_count && *shape.element_count != 0) {
		const uint64_t count = *shape.element_count;
		return count > std::numeric_limits<uint64_t>::max() / DWORD_BYTES
		           ? std::numeric_limits<uint64_t>::max()
		           : DWORD_BYTES * count;
	}
	return element_at ? StructSize(reader, *element_at) : std::optional<uint64_t> {0};
}

/// Reads the element record that a type record points at. The element is empty when the offset is
/// missing or outside the blob. It returns false where python raises IndexError.
bool ReadElement(const Reader& reader, std::optional<uint64_t> element_at, Die& element) {
	if (!element_at || *element_at >= reader.Size()) return true;
	const auto read = ReadDie(reader, *element_at);
	if (read.status == DieStatus::Failed) return false;
	if (read.status == DieStatus::Present) element = read.die;
	return true;
}

/// Empties the fields that python's tuple for this kind does not carry, so that a shape holds only
/// what the predictor can use.
void KeepKindFields(Sl00Shape& shape) {
	switch (shape.kind) {
		case Sl00ShapeKind::Raw:
		case Sl00ShapeKind::Structured:
			shape.flag.reset();
			shape.element_kind.reset();
			shape.element_count.reset();
			break;
		case Sl00ShapeKind::Typed: shape.flag.reset(); break;
		default: break;
	}
}

/// The class that a record's attributes give, which is B for attribute 30, U for attribute 40 and
/// T otherwise.
Sl00Class ClassOf(const Die& die) {
	if (die.HasAttribute(ATTRIBUTE_CONSTANT_BUFFER)) return Sl00Class::ConstantBuffer;
	return die.HasAttribute(ATTRIBUTE_READ_WRITE) ? Sl00Class::ReadWrite : Sl00Class::ReadOnly;
}

/// Classifies a resource's type record and sets its class from the same record's attributes, which
/// are the abbreviation's. The class is left Unknown when the record is absent. It fails where
/// python raises IndexError, and then the caller drops the class too.
std::optional<Sl00Shape> TypeShape(const Reader& reader, uint64_t reference,
                                   Sl00Class& resource_class) {
	Sl00Shape  shape;
	const auto resource = ReadDie(reader, reference);
	if (resource.status == DieStatus::Failed) return std::nullopt;
	if (resource.status == DieStatus::Absent) return shape;
	resource_class = ClassOf(resource.die);
	if (resource.die.HasAttribute(ATTRIBUTE_CONSTANT_BUFFER)) {
		shape.kind = Sl00ShapeKind::ConstantBuffer;
		return shape;
	}
	shape.flag            = resource.die.Value(ATTRIBUTE_KIND);
	const auto element_at = resource.die.Value(ATTRIBUTE_ELEMENT);
	Die        element;
	if (!ReadElement(reader, element_at, element)) return std::nullopt;
	shape.element_kind  = element.Value(ATTRIBUTE_KIND);
	shape.element_count = element.Value(ATTRIBUTE_COUNT);
	shape.kind          = Sl00ShapeKind::Other;
	if (shape.flag == FLAG_RAW) {
		shape.kind = Sl00ShapeKind::Raw;
	} else if (shape.flag == FLAG_TYPED) {
		shape.kind = Sl00ShapeKind::Typed;
	} else if (shape.flag == FLAG_STRUCTURED) {
		shape.kind        = Sl00ShapeKind::Structured;
		const auto stride = StructuredStride(reader, shape, element_at);
		if (!stride) return std::nullopt;
		shape.stride = *stride;
	} else if (shape.flag && std::find(FLAG_IMAGES.begin(), FLAG_IMAGES.end(), *shape.flag) !=
	                             FLAG_IMAGES.end()) {
		shape.kind = Sl00ShapeKind::Image;
	}
	KeepKindFields(shape);
	return shape;
}

// ---- Resource records. ----

/// A place in the stream where a name and a type reference follow a register byte.
struct Row {
	size_t             at;
	const std::string* name;
	uint64_t           reference;
	uint8_t            register_index;
};

/// Finds the rows of the last run, which are the positions that read as resource records with no
/// two further apart than a record is long. Records near the stream's end are self offset,
/// register, name, type, then flags, so a row is a name that is a string table key, then a type
/// reference inside the stream, behind a byte below 0x80. Python stops the whole search where a
/// read runs off the blob, and so does this. The last four bytes are never positions. Only the last
/// run survives, so the rows are dropped whenever a gap ends one. The result is in blob order.
std::vector<Row> FindRows(const Reader& reader, const StringTable& table) {
	std::vector<Row> rows;
	const size_t     size = reader.Size();
	if (size <= SCAN_END_MARGIN) return rows;
	for (size_t at = table.records_start; at < size - SCAN_END_MARGIN; at++) {
		const auto name = reader.Read(at);
		if (!name) break;
		const auto found = table.strings.find(name->value);
		if (found == table.strings.end()) continue;
		const auto reference = reader.Read(name->next);
		if (!reference) break;
		const bool in_stream = reference->value >= table.records_start && reference->value < size;
		if (in_stream && at >= 1 && reader.Data()[at - 1] < REGISTER_LIMIT) {
			if (!rows.empty() && at - rows.back().at > MAX_ROW_GAP) rows.clear();
			rows.push_back({at, &found->second, reference->value, reader.Data()[at - 1]});
		}
	}
	return rows;
}

/// Builds one resource from a row. A type record whose read raises leaves the shape Unreadable,
/// the abbreviation offset empty and the class Unknown, as python does when it catches IndexError.
/// A record whose abbreviation lies outside the section is also Unreadable with class Unknown, but
/// keeps that out-of-range abbreviation offset, as python keeps its tag.
Sl00Resource BuildResource(const Reader& reader, const Row& row) {
	Sl00Resource resource;
	resource.name             = *row.name;
	resource.register_index   = row.register_index;
	resource.type_offset      = row.reference;
	Sl00Class  resource_class = Sl00Class::Unknown;
	const auto shape          = TypeShape(reader, row.reference, resource_class);
	const auto code           = reader.Read(row.reference);
	if (shape && code) {
		resource.shape               = *shape;
		resource.abbreviation_offset = code->value;
		resource.resource_class      = resource_class;
	}
	return resource;
}

} // namespace

std::optional<std::vector<Sl00Resource>> DecodeSl00Resources(std::span<const uint8_t> metadata,
                                                             Sl00DecodeStats*         stats) {
	if (stats) *stats = {};
	if (metadata.size() > SL00_MAX_METADATA_BYTES) return std::nullopt;
	const Reader      reader(metadata);
	const StringTable table = ReadStringTable(metadata);
	const auto        rows  = FindRows(reader, table);
	if (rows.size() > SL00_MAX_RESOURCES) {
		if (stats) *stats = {reader.WorkUsed(), reader.WorkBudget()};
		return std::nullopt;
	}
	std::vector<Sl00Resource> resources;
	resources.reserve(rows.size());
	// The rows are in blob order and python lists the last one first.
	for (auto it = rows.rbegin(); it != rows.rend(); ++it) {
		resources.push_back(BuildResource(reader, *it));
	}
	if (stats) *stats = {reader.WorkUsed(), reader.WorkBudget()};
	if (reader.Exhausted()) return std::nullopt;
	return resources;
}

} // namespace Libs::Graphics::StoreGen
