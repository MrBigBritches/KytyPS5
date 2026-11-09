#include "graphics/shader/storegen/ebootShaderScan.h"

#include "graphics/shader/shader.h"
#include "graphics/shader/storegen/shaderHeaderLayout.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <xxhash.h>

namespace Libs::Graphics::StoreGen {

namespace {

namespace Layout = HeaderLayout;

constexpr std::array<uint8_t, 4> ELF_MAGIC {0x7f, 'E', 'L', 'F'};
constexpr uint8_t                ELF_CLASS_64       = 2;
constexpr uint8_t                ELF_DATA_LITTLE    = 1;
constexpr uint64_t               ELF_HEADER_SIZE    = 64;
constexpr uint64_t               ELF_OFFSET_CLASS   = 4;
constexpr uint64_t               ELF_OFFSET_DATA    = 5;
constexpr uint64_t               ELF_OFFSET_SHOFF   = 0x28;
constexpr uint64_t               ELF_OFFSET_SHENT   = 0x3A;
constexpr uint64_t               ELF_OFFSET_SHNUM   = 0x3C;
constexpr uint64_t               ELF_OFFSET_SHSTR   = 0x3E;
constexpr uint16_t               SECTION_ENTRY_SIZE = 64;
// Real shader ELFs have a handful of sections. The cap bounds the work a hostile table can cost.
constexpr uint16_t MAX_SECTIONS = 64;
// A section header entry holds the name offset at 0, the file offset at 0x18 and the size at 0x20.
constexpr uint64_t SECTION_OFFSET_NAME = 0x00;
constexpr uint64_t SECTION_OFFSET_FILE = 0x18;
constexpr uint64_t SECTION_OFFSET_SIZE = 0x20;
constexpr uint32_t FOOTER_SIZE_BYTES   = 0x30;
constexpr uint32_t FOOTER_CODE_WORD    = 5;
constexpr uint32_t FOOTER_LITERAL_WORD = 6;

constexpr std::string_view NAME_HEADER   = ".shader_header";
constexpr std::string_view NAME_TEXT     = ".shader_text";
constexpr std::string_view NAME_METADATA = ".metadata";

// A byte range inside the eboot. `present` is false until a section of that name is seen.
struct Range {
	uint64_t offset  = 0;
	uint64_t size    = 0;
	bool     present = false;

	[[nodiscard]] uint64_t End() const { return offset + size; }
};

struct ElfSections {
	Range    header;
	Range    text;
	Range    metadata;
	uint64_t end    = 0; // Where the section table ends, which is where the next ELF can start.
	uint64_t extent = 0; // The furthest byte the ELF's own sections and table reach.
};

// Whether [offset, offset + size) lies inside a buffer of `limit` bytes, without overflowing.
bool InBounds(uint64_t offset, uint64_t size, uint64_t limit) {
	return offset <= limit && size <= limit - offset;
}

// Whether the fixed ELF header describes a 64-bit little-endian file with a usable section table.
bool ReadElfTableShape(std::span<const uint8_t> eboot, uint64_t start, uint64_t& table_offset,
                       uint16_t& count, uint16_t& names_index) {
	uint16_t entry_size = 0;
	uint64_t shoff      = 0;
	if (!InBounds(start, ELF_HEADER_SIZE, eboot.size()) ||
	    eboot[start + ELF_OFFSET_CLASS] != ELF_CLASS_64 ||
	    eboot[start + ELF_OFFSET_DATA] != ELF_DATA_LITTLE ||
	    !Layout::Read(eboot, start + ELF_OFFSET_SHOFF, shoff) ||
	    !Layout::Read(eboot, start + ELF_OFFSET_SHENT, entry_size) ||
	    !Layout::Read(eboot, start + ELF_OFFSET_SHNUM, count) ||
	    !Layout::Read(eboot, start + ELF_OFFSET_SHSTR, names_index)) {
		return false;
	}
	if (entry_size != SECTION_ENTRY_SIZE || count == 0 || count > MAX_SECTIONS ||
	    names_index >= count || shoff > eboot.size() - start ||
	    !InBounds(start + shoff, uint64_t {count} * SECTION_ENTRY_SIZE, eboot.size())) {
		return false;
	}
	table_offset = start + shoff;
	return true;
}

// Reads one section header entry as an absolute, bounds-checked file range plus its name offset.
bool ReadSectionEntry(std::span<const uint8_t> eboot, uint64_t start, uint64_t entry_at,
                      uint32_t& name_offset, Range& range) {
	uint64_t offset = 0;
	uint64_t size   = 0;
	if (!Layout::Read(eboot, entry_at + SECTION_OFFSET_NAME, name_offset) ||
	    !Layout::Read(eboot, entry_at + SECTION_OFFSET_FILE, offset) ||
	    !Layout::Read(eboot, entry_at + SECTION_OFFSET_SIZE, size) ||
	    offset > eboot.size() - start || !InBounds(start + offset, size, eboot.size())) {
		return false;
	}
	range = {start + offset, size, true};
	return true;
}

// The NUL-terminated name at `name_offset` in the name table, or empty when it does not terminate.
std::string_view SectionName(std::span<const uint8_t> names, uint32_t name_offset) {
	if (name_offset >= names.size()) {
		return {};
	}
	const auto* text = reinterpret_cast<const char*>(names.data()) + name_offset;
	const auto* nul  = static_cast<const char*>(std::memchr(text, 0, names.size() - name_offset));
	return nul == nullptr ? std::string_view {}
	                      : std::string_view {text, static_cast<size_t>(nul - text)};
}

// Which of the three wanted sections a name stands for, or null for any other section.
Range* WantedSlot(ElfSections& sections, std::string_view name) {
	if (name == NAME_HEADER) return &sections.header;
	if (name == NAME_TEXT) return &sections.text;
	if (name == NAME_METADATA) return &sections.metadata;
	return nullptr;
}

// Parses the section table of the ELF at `start` and keeps the three sections the generator needs.
// Returns nothing when the ELF is malformed or has no .shader_text.
std::optional<ElfSections> ParseShaderElf(std::span<const uint8_t> eboot, uint64_t start) {
	uint64_t table_offset = 0;
	uint16_t count        = 0;
	uint16_t names_index  = 0;
	if (!ReadElfTableShape(eboot, start, table_offset, count, names_index)) {
		return std::nullopt;
	}
	uint32_t unused_name = 0;
	Range    names_range;
	if (!ReadSectionEntry(eboot, start, table_offset + uint64_t {names_index} * SECTION_ENTRY_SIZE,
	                      unused_name, names_range)) {
		return std::nullopt;
	}
	const auto names = eboot.subspan(names_range.offset, names_range.size);

	ElfSections sections;
	sections.end    = table_offset + uint64_t {count} * SECTION_ENTRY_SIZE;
	sections.extent = sections.end;
	for (uint16_t index = 0; index < count; index++) {
		uint32_t name_offset = 0;
		Range    range;
		// A section the generator does not need may be anything, so only its name decides whether
		// its entry has to be valid.
		const bool entry_ok = ReadSectionEntry(
		    eboot, start, table_offset + uint64_t {index} * SECTION_ENTRY_SIZE, name_offset, range);
		Range* slot = WantedSlot(sections, SectionName(names, name_offset));
		if (slot == nullptr) {
			continue;
		}
		if (!entry_ok) {
			return std::nullopt;
		}
		*slot           = range;
		sections.extent = std::max(sections.extent, range.End());
	}
	if (!sections.text.present) {
		return std::nullopt;
	}
	return sections;
}

// Copies a section out of the eboot, or gives an empty vector when the ELF has no such section.
std::vector<uint8_t> CopyRange(std::span<const uint8_t> eboot, const Range& range) {
	if (!range.present) {
		return {};
	}
	const auto bytes = eboot.subspan(range.offset, range.size);
	return {bytes.begin(), bytes.end()};
}

// Whether the header is a compute program's. Sets `shader_size` to the binary length it declares.
// Returns false and leaves `is_compute` unset when the header is too short or has the wrong magic.
bool ReadHeaderKind(const std::vector<uint8_t>& header, bool& is_compute, uint32_t& shader_size) {
	uint32_t magic   = 0;
	uint32_t version = 0;
	uint8_t  type    = 0;
	if (header.size() < Layout::MIN_HEADER_SIZE ||
	    !Layout::Read<uint32_t>(header, Layout::OFFSET_MAGIC, magic) ||
	    !Layout::Read<uint32_t>(header, Layout::OFFSET_VERSION, version) ||
	    !Layout::Read<uint32_t>(header, Layout::OFFSET_SHADER_SIZE, shader_size) ||
	    !Layout::Read<uint8_t>(header, Layout::OFFSET_TYPE, type) || magic != Layout::FILE_MAGIC ||
	    version != Layout::FILE_VERSION) {
		return false;
	}
	is_compute = type == Layout::TYPE_COMPUTE;
	return true;
}

// What became of a shader binary whose length the header declares.
enum class BinaryOutcome { Footer, NoFooter, Malformed, Oversize };

// Copies the shader binary that follows .shader_text and hashes the part the runtime hashes. A
// binary with no valid footer is hashed whole, as the runtime does, and has code and literal sizes
// of 0.
BinaryOutcome BuildProgramCode(std::span<const uint8_t> eboot, const ElfSections& sections,
                               uint32_t shader_size, ScannedProgram& program) {
	if (shader_size == 0 || shader_size % sizeof(uint32_t) != 0) {
		return BinaryOutcome::Malformed;
	}
	// The binary must lie inside the ELF's own extent, which also bounds the copy a hostile header
	// can ask for.
	if (!InBounds(sections.text.offset, shader_size, eboot.size()) ||
	    sections.text.offset + shader_size > sections.extent) {
		return BinaryOutcome::Oversize;
	}
	// A copy, because the file bytes are not aligned for the uint32_t reads the runtime's
	// ShaderProgramSize does.
	std::vector<uint32_t> binary(shader_size / sizeof(uint32_t));
	std::memcpy(binary.data(), eboot.data() + sections.text.offset, shader_size);
	const uint32_t program_size = ShaderProgramSize(binary.data(), shader_size);
	const bool     has_footer   = program_size != shader_size;
	if (has_footer) {
		const auto* footer = binary.data() + (shader_size - FOOTER_SIZE_BYTES) / sizeof(uint32_t);
		program.code_size_bytes    = footer[FOOTER_CODE_WORD];
		program.literal_size_bytes = footer[FOOTER_LITERAL_WORD];
	}
	binary.resize(program_size / sizeof(uint32_t));
	program.code_hash = XXH3_64bits(binary.data(), program_size);
	program.code      = std::move(binary);
	return has_footer ? BinaryOutcome::Footer : BinaryOutcome::NoFooter;
}

// Applies a finished compute program to the result, merging it with an earlier one of the same
// hash.
void AddProgram(ScannedProgram&& program, std::unordered_map<uint64_t, size_t>& index_by_hash,
                ScanResult& result) {
	const auto [entry, inserted] =
	    index_by_hash.try_emplace(program.code_hash, result.programs.size());
	if (inserted) {
		result.programs.push_back(std::move(program));
		return;
	}
	result.counts.duplicates++;
	if (result.programs[entry->second].header != program.header) {
		result.counts.duplicate_header_differs++;
	}
}

// Turns the ELF at `start` into a program when it is a compute shader, and counts what became of
// it. Returns where the next ELF can start.
uint64_t HandleElf(std::span<const uint8_t> eboot, uint64_t start,
                   std::unordered_map<uint64_t, size_t>& index_by_hash, ScanResult& result) {
	const uint64_t next = start + 1;
	result.counts.elf_candidates++;
	const auto sections = ParseShaderElf(eboot, start);
	if (!sections || !sections->header.present) {
		result.counts.rejected++;
		return next;
	}
	ScannedProgram program;
	bool           is_compute  = false;
	uint32_t       shader_size = 0;
	program.header             = CopyRange(eboot, sections->header);
	if (!ReadHeaderKind(program.header, is_compute, shader_size)) {
		result.counts.rejected++;
		return next;
	}
	// The section table ends the ELF, so the next one cannot start inside it.
	const uint64_t after = std::max(next, sections->end);
	if (!is_compute) {
		result.counts.graphics_skipped++;
		return after;
	}
	switch (BuildProgramCode(eboot, *sections, shader_size, program)) {
		case BinaryOutcome::Malformed: result.counts.rejected++; return after;
		case BinaryOutcome::Oversize: result.counts.oversize++; return after;
		case BinaryOutcome::NoFooter: result.counts.no_footer++; break;
		case BinaryOutcome::Footer: break;
	}
	program.metadata   = CopyRange(eboot, sections->metadata);
	program.elf_offset = start;
	AddProgram(std::move(program), index_by_hash, result);
	return after;
}

} // namespace

ScanResult ScanEbootShaders(std::span<const uint8_t> eboot) {
	ScanResult                           result;
	std::unordered_map<uint64_t, size_t> index_by_hash;
	uint64_t                             position = 0;
	while (position + ELF_MAGIC.size() <= eboot.size()) {
		const auto* base = eboot.data();
		const auto* hit  = static_cast<const uint8_t*>(
            std::memchr(base + position, ELF_MAGIC[0], eboot.size() - position));
		if (hit == nullptr) {
			break;
		}
		const uint64_t start = static_cast<uint64_t>(hit - base);
		if (!InBounds(start, ELF_MAGIC.size(), eboot.size()) ||
		    std::memcmp(hit, ELF_MAGIC.data(), ELF_MAGIC.size()) != 0) {
			position = start + 1;
			continue;
		}
		position = HandleElf(eboot, start, index_by_hash, result);
	}
	return result;
}

} // namespace Libs::Graphics::StoreGen
