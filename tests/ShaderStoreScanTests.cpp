// Tests the eboot compute shader scan, the static key built from a shader header and the sl00
// metadata decoder.
// Everything here uses synthetic blobs, so no game file is needed.
#include "graphics/guest_gpu/csRegisters.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/ebootShaderScan.h"
#include "graphics/shader/storegen/shaderHeaderLayout.h"
#include "graphics/shader/storegen/sl00Metadata.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;
namespace Layout = Libs::Graphics::StoreGen::HeaderLayout;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Bytes = std::vector<uint8_t>;

// ---- Shader binary and header values used by the blobs. ----
constexpr uint32_t CODE_SIZE             = 40; // Rounds up to 48.
constexpr uint32_t ALIGNED_CODE          = 48;
constexpr uint32_t LITERAL_SIZE          = 20;
constexpr uint32_t PADDING_SIZE          = 12; // Junk between the literals and the footer.
constexpr uint32_t FOOTER_SIZE           = 0x30;
constexpr uint32_t FOOTER_VERSION        = 1;
constexpr uint32_t FOOTER_VERSION_OFFSET = 0x10;
constexpr uint32_t FOOTER_CODE_OFFSET    = 0x14;
constexpr uint32_t FOOTER_LITERAL_OFFSET = 0x18;
constexpr uint32_t BAD_FOOTER_VERSION    = 2;
constexpr uint32_t HUGE_LITERAL_SIZE     = 0x10000000;
constexpr uint32_t HEADER_SIZE           = 0x60;
constexpr uint64_t HEADER_SH_OFFSET      = 0x60;
constexpr uint32_t SCRATCH_DWORDS        = 7;
constexpr uint8_t  TYPE_GRAPHICS         = 1;
constexpr uint32_t THREADS_X             = 64;
constexpr uint32_t THREADS_Y             = 2;
constexpr uint32_t THREADS_Z             = 4;
constexpr uint32_t OTHER_THREADS_X       = 128;
constexpr uint32_t BYTE_FILL_MULTIPLIER  = 7;
constexpr uint32_t BYTE_FILL_SEED_STEP   = 31;
constexpr uint32_t BYTE_FILL_OFFSET      = 3;
constexpr uint32_t METADATA_BYTES        = 24;
constexpr uint8_t  METADATA_FILL         = 0xA0;
constexpr uint32_t UNKNOWN_SH_REGISTER   = 0x2FE;
constexpr uint32_t UNKNOWN_SH_VALUE      = 0x12345678;
constexpr uint32_t UNKNOWN_START_Y_VALUE = 99;

// The device values the keys are built for. The python reference uses subgroup 32.
constexpr uint32_t HOST_SUBGROUP     = 32;
constexpr uint32_t OTHER_SUBGROUP    = 64;
constexpr uint32_t WIDE_SHARED_BYTES = 65536;
// Below the 1536 bytes that the test shader's LDS needs.
constexpr uint32_t NARROW_SHARED_BYTES = 1024;
constexpr ComputeKeyDevice WIDE_DEVICE {HOST_SUBGROUP, WIDE_SHARED_BYTES};
constexpr ComputeKeyDevice NARROW_DEVICE {OTHER_SUBGROUP, NARROW_SHARED_BYTES};
constexpr uint32_t         OTHER_WAVE_SIZE = 32;

// Compute register values written with the bit layout of the hardware, spelled out here with plain
// shifts so that the golden does not depend on the Pm4 constants the decoder uses.
constexpr uint32_t FLOAT_MODE     = 0xC0;
constexpr uint32_t RSRC1_VGPRS    = 5;
constexpr uint32_t RSRC1          = RSRC1_VGPRS | (FLOAT_MODE << 12u);
constexpr uint32_t USER_SGPR      = 16;
constexpr uint32_t TIDIG_COMP_CNT = 2;
constexpr uint32_t LDS_GRANULES   = 3;
constexpr uint32_t RSRC2          = 1u | (USER_SGPR << 1u) | (1u << 7u) | (0u << 8u) | (1u << 9u) |
                           (1u << 10u) | (TIDIG_COMP_CNT << 11u) | (LDS_GRANULES << 15u);
constexpr uint32_t LDS_DWORDS_PER_GRANULE = 128;

// ---- Little-endian writers. ----

// Writes a value at a byte offset, growing the buffer with zeros when it is too short.
template <typename T>
void Put(Bytes& bytes, uint64_t at, T value) {
	if (bytes.size() < at + sizeof(T)) {
		bytes.resize(at + sizeof(T));
	}
	std::memcpy(bytes.data() + at, &value, sizeof(T));
}

// Writes a value at the end of the buffer.
template <typename T>
void Append(Bytes& bytes, T value) {
	Put(bytes, bytes.size(), value);
}

// Writes the bytes of one buffer at the end of another.
void AppendBytes(Bytes& bytes, const Bytes& extra) {
	bytes.insert(bytes.end(), extra.begin(), extra.end());
}

// A shader binary, and the range of it the runtime hashes.
struct Binary {
	Bytes bytes;
	Bytes hashed;
};

// Builds code, literals, junk padding and the 0x30-byte footer. `seed` changes the code bytes.
Binary BuildBinary(uint32_t seed, uint32_t code_size = CODE_SIZE,
                   uint32_t literal_size = LITERAL_SIZE, uint32_t version = FOOTER_VERSION) {
	Binary         binary;
	const uint32_t aligned = (code_size + 15u) & ~15u;
	for (uint32_t i = 0; i < aligned + literal_size + PADDING_SIZE; i++) {
		binary.bytes.push_back(static_cast<uint8_t>(
		    (i * BYTE_FILL_MULTIPLIER + seed * BYTE_FILL_SEED_STEP + BYTE_FILL_OFFSET) & 0xffu));
	}
	binary.hashed.assign(binary.bytes.begin(), binary.bytes.begin() + aligned + literal_size);
	Bytes footer(FOOTER_SIZE, 0);
	std::memcpy(footer.data(), "barefoot", 8);
	Put<uint32_t>(footer, FOOTER_VERSION_OFFSET, version);
	Put<uint32_t>(footer, FOOTER_CODE_OFFSET, code_size);
	Put<uint32_t>(footer, FOOTER_LITERAL_OFFSET, literal_size);
	AppendBytes(binary.bytes, footer);
	return binary;
}

struct Reg {
	uint32_t offset;
	uint32_t value;
};

// The registers of a typical compute program.
std::vector<Reg> StandardRegisters() {
	return {{Pm4::COMPUTE_PGM_LO, 0},
	        {Pm4::COMPUTE_PGM_HI, 0},
	        {Pm4::COMPUTE_PGM_RSRC1, RSRC1},
	        {Pm4::COMPUTE_PGM_RSRC2, RSRC2},
	        {Pm4::COMPUTE_NUM_THREAD_X, THREADS_X},
	        {Pm4::COMPUTE_NUM_THREAD_Y, THREADS_Y},
	        {Pm4::COMPUTE_NUM_THREAD_Z, THREADS_Z}};
}

// Builds a .shader_header with the SH register table right after the fixed part.
Bytes BuildHeader(uint8_t type, uint32_t shader_size, const std::vector<Reg>& registers) {
	Bytes header(HEADER_SIZE, 0);
	Put<uint32_t>(header, Layout::OFFSET_MAGIC, Layout::FILE_MAGIC);
	Put<uint32_t>(header, Layout::OFFSET_VERSION, Layout::FILE_VERSION);
	// Stored relative to the pointer's own field, as the AGC header does.
	Put<uint64_t>(header, Layout::OFFSET_SH_REGISTERS,
	              HEADER_SH_OFFSET - Layout::OFFSET_SH_REGISTERS);
	Put<uint32_t>(header, Layout::OFFSET_SHADER_SIZE, shader_size);
	Put<uint16_t>(header, Layout::OFFSET_SCRATCH_DWORDS, static_cast<uint16_t>(SCRATCH_DWORDS));
	Put<uint8_t>(header, Layout::OFFSET_TYPE, type);
	Put<uint8_t>(header, Layout::OFFSET_NUM_SH_REGISTERS, static_cast<uint8_t>(registers.size()));
	for (const auto& reg: registers) {
		Append<uint32_t>(header, reg.offset);
		Append<uint32_t>(header, reg.value);
	}
	return header;
}

// An ELF under construction, with the places a hostile variant patches.
struct Elf {
	Bytes    bytes;
	uint64_t table_offset  = 0;
	uint64_t header_offset = 0; // Where the .shader_header bytes start in the ELF.
	uint64_t text_offset   = 0; // Where the .shader_text bytes start in the ELF.
};

constexpr uint64_t ELF_HEADER_BYTES    = 64;
constexpr uint64_t SECTION_ENTRY_BYTES = 64;
constexpr uint16_t SECTION_COUNT       = 5; // Null, names, header, text, metadata.
constexpr uint64_t ENTRY_FILE_OFFSET   = 0x18;
constexpr uint64_t ENTRY_SIZE_FIELD    = 0x20;
constexpr uint64_t ENTRY_NAME_FIELD    = 0x00;
constexpr uint16_t NAMES_INDEX         = 1;
constexpr uint16_t HEADER_INDEX        = 2;
constexpr uint16_t TEXT_INDEX          = 3;
constexpr uint16_t METADATA_INDEX      = 4;
constexpr uint64_t ELF_DATA_FIELD      = 0x05;
constexpr uint64_t ELF_CLASS_FIELD     = 0x04;
constexpr uint64_t ELF_SHOFF_FIELD     = 0x28;
constexpr uint64_t ELF_SHENT_FIELD     = 0x3A;
constexpr uint64_t ELF_SHNUM_FIELD     = 0x3C;
constexpr uint64_t ELF_SHSTR_FIELD     = 0x3E;
constexpr uint16_t OVER_LIMIT_SECTIONS = 65;
constexpr uint32_t TABLE_PADDING_BYTES =
    (OVER_LIMIT_SECTIONS - SECTION_COUNT) * SECTION_ENTRY_BYTES;
constexpr uint8_t  ELF_BIG_ENDIAN  = 2;
constexpr uint8_t  ELF_CLASS_32    = 1;
constexpr uint32_t TAIL_JUNK_BYTES = 7;
constexpr uint32_t BIG_TAIL_BYTES  = 4096;

// The file offset of a section table entry.
uint64_t EntryAt(const Elf& elf, uint16_t index) {
	return elf.table_offset + index * SECTION_ENTRY_BYTES;
}

// Lays out an ELF64 with .shader_header, .shader_text and .metadata. The section table is last, so
// every proper prefix of the file is missing part of it.
Elf BuildElf(const Bytes& header, const Bytes& text, const Bytes& metadata) {
	// Names are offsets into this table, so each one is found rather than counted by hand.
	const std::string names = std::string(1, '\0') + ".shstrtab" + '\0' + ".shader_header" + '\0' +
	                          ".shader_text" + '\0' + ".metadata" + '\0';
	const auto name_at = [&](const char* name) { return static_cast<uint32_t>(names.find(name)); };
	Elf        elf;
	Bytes&     file = elf.bytes;
	file.assign(ELF_HEADER_BYTES, 0);
	file[0]               = 0x7f;
	file[1]               = 'E';
	file[2]               = 'L';
	file[3]               = 'F';
	file[ELF_CLASS_FIELD] = 2; // 64-bit.
	file[ELF_DATA_FIELD]  = 1; // Little-endian.
	struct Placed {
		uint32_t name;
		uint64_t offset;
		uint64_t size;
	};
	std::array<Placed, SECTION_COUNT> placed {};
	const auto place = [&](uint16_t index, uint32_t name, const Bytes& data) {
		while (file.size() % 16 != 0)
			file.push_back(0);
		placed[index] = {name, file.size(), data.size()};
		AppendBytes(file, data);
	};
	place(NAMES_INDEX, name_at(".shstrtab"), Bytes(names.begin(), names.end()));
	place(HEADER_INDEX, name_at(".shader_header"), header);
	place(TEXT_INDEX, name_at(".shader_text"), text);
	place(METADATA_INDEX, name_at(".metadata"), metadata);
	elf.header_offset = placed[HEADER_INDEX].offset;
	elf.text_offset   = placed[TEXT_INDEX].offset;
	while (file.size() % 16 != 0)
		file.push_back(0);
	elf.table_offset = file.size();
	for (uint16_t index = 0; index < SECTION_COUNT; index++) {
		Bytes entry(SECTION_ENTRY_BYTES, 0);
		Put<uint32_t>(entry, ENTRY_NAME_FIELD, placed[index].name);
		Put<uint64_t>(entry, ENTRY_FILE_OFFSET, placed[index].offset);
		Put<uint64_t>(entry, ENTRY_SIZE_FIELD, placed[index].size);
		AppendBytes(file, entry);
	}
	Put<uint64_t>(file, ELF_SHOFF_FIELD, elf.table_offset);
	Put<uint16_t>(file, ELF_SHENT_FIELD, SECTION_ENTRY_BYTES);
	Put<uint16_t>(file, ELF_SHNUM_FIELD, SECTION_COUNT);
	Put<uint16_t>(file, ELF_SHSTR_FIELD, NAMES_INDEX);
	return elf;
}

// Metadata bytes that differ with the seed, so a test can tell which ELF a program came from.
Bytes MetadataBytes(uint8_t seed) {
	Bytes bytes;
	for (uint32_t i = 0; i < METADATA_BYTES; i++) {
		bytes.push_back(static_cast<uint8_t>(METADATA_FILL + seed + i));
	}
	return bytes;
}

// One complete compute or graphics ELF.
Elf BuildProgramElf(uint8_t type, uint32_t code_seed, uint8_t metadata_seed = 1,
                    const std::vector<Reg>& registers = StandardRegisters()) {
	const auto binary = BuildBinary(code_seed);
	return BuildElf(BuildHeader(type, static_cast<uint32_t>(binary.bytes.size()), registers),
	                binary.bytes, MetadataBytes(metadata_seed));
}

// Puts ELFs into an eboot behind a few bytes that are not aligned, a false magic and an ELF-like
// stub that is not a shader. `tail_bytes` of junk follow the last ELF.
Bytes BuildEboot(const std::vector<const Elf*>& elfs, uint32_t tail_bytes = TAIL_JUNK_BYTES) {
	Bytes eboot {1, 2, 3, 0x7f, 'E', 'L', 'X', 9, 9};
	// A real magic whose table fields are whatever bytes follow, so it must be rejected.
	const Bytes stub {0x7f, 'E', 'L', 'F', 2, 1, 0, 0, 0, 0};
	AppendBytes(eboot, stub);
	for (const auto* elf: elfs) {
		while (eboot.size() % 16 != 5)
			eboot.push_back(0xEE);
		AppendBytes(eboot, elf->bytes);
	}
	for (uint32_t i = 0; i < tail_bytes; i++)
		eboot.push_back(0xCC);
	return eboot;
}

// The XXH3 hash the runtime gives a byte range.
uint64_t Xxh3(const Bytes& bytes) {
	return XXH3_64bits(bytes.data(), bytes.size());
}

// The bytes of a word vector.
Bytes AsBytes(const std::vector<uint32_t>& words) {
	Bytes bytes(words.size() * sizeof(uint32_t));
	if (!words.empty()) std::memcpy(bytes.data(), words.data(), bytes.size());
	return bytes;
}

// ---- Scan tests. ----

// One compute ELF gives one program whose hash is the XXH3 of code rounded to 16 plus literals.
void TestComputeProgramHash() {
	const auto binary = BuildBinary(1);
	const auto elf =
	    BuildElf(BuildHeader(0, static_cast<uint32_t>(binary.bytes.size()), StandardRegisters()),
	             binary.bytes, MetadataBytes(1));
	const auto eboot  = BuildEboot({&elf});
	const auto result = ScanEbootShaders(eboot);

	// The ranges this hash could be mistaken for. The test must not pass for any of them.
	const Bytes code_only(binary.bytes.begin(), binary.bytes.begin() + ALIGNED_CODE);
	const Bytes unaligned(binary.bytes.begin(), binary.bytes.begin() + CODE_SIZE);
	const auto  expected = Xxh3(binary.hashed);
	CHECK(expected != Xxh3(code_only));
	CHECK(expected != Xxh3(unaligned));
	CHECK(expected != Xxh3(binary.bytes));
	CHECK(binary.hashed.size() == ALIGNED_CODE + LITERAL_SIZE);

	CHECK(result.programs.size() == 1);
	CHECK(result.counts.elf_candidates ==
	      2); // The stub and the real ELF. The false magic is not one.
	CHECK(result.counts.rejected == 1);
	CHECK(result.counts.no_footer == 0 && result.counts.oversize == 0);
	if (result.programs.size() != 1) return;
	const auto& program = result.programs[0];
	CHECK(program.code_hash == expected);
	CHECK(program.code_size_bytes == CODE_SIZE);
	CHECK(program.literal_size_bytes == LITERAL_SIZE);
	CHECK(AsBytes(program.code) == binary.hashed);
	CHECK(program.header ==
	      BuildHeader(0, static_cast<uint32_t>(binary.bytes.size()), StandardRegisters()));
	CHECK(program.metadata == MetadataBytes(1));
	CHECK(program.elf_offset < eboot.size() && std::memcmp(eboot.data() + program.elf_offset,
	                                                       "\x7f"
	                                                       "ELF",
	                                                       4) == 0);
}

// A graphics ELF is skipped, and the same blob as compute is taken, so the type byte is what
// decided.
void TestGraphicsElfSkipped() {
	const auto graphics = BuildProgramElf(TYPE_GRAPHICS, 1);
	const auto compute  = BuildProgramElf(0, 1);
	const auto skipped  = ScanEbootShaders(BuildEboot({&graphics}));
	CHECK(skipped.programs.empty());
	CHECK(skipped.counts.graphics_skipped == 1);
	CHECK(skipped.counts.rejected == 1); // Only the stub.
	CHECK(ScanEbootShaders(BuildEboot({&compute})).programs.size() == 1);

	// With a graphics ELF first, the count is one and it is the compute program that is returned.
	const auto other = BuildProgramElf(0, 2);
	const auto mixed = ScanEbootShaders(BuildEboot({&graphics, &other}));
	CHECK(mixed.programs.size() == 1);
	CHECK(mixed.counts.graphics_skipped == 1);
	if (mixed.programs.size() == 1) {
		CHECK(mixed.programs[0].code_hash == Xxh3(BuildBinary(2).hashed));
	}
}

// A field patch. `width` is the size in bytes of the value written at `offset` in the ELF.
struct Patch {
	uint64_t offset;
	uint64_t value;
	uint8_t  width;
};

// Applies patches to a copy of an ELF, adds `padding` zero bytes after it, and embeds it in an
// eboot.
Bytes EbootWithPatchedElf(const Elf& elf, const std::vector<Patch>& patches, uint32_t padding = 0,
                          uint32_t tail_bytes = TAIL_JUNK_BYTES) {
	Elf copy = elf;
	copy.bytes.resize(copy.bytes.size() + padding, 0);
	for (const auto& patch: patches) {
		switch (patch.width) {
			case 1:
				Put<uint8_t>(copy.bytes, patch.offset, static_cast<uint8_t>(patch.value));
				break;
			case 2:
				Put<uint16_t>(copy.bytes, patch.offset, static_cast<uint16_t>(patch.value));
				break;
			case 4:
				Put<uint32_t>(copy.bytes, patch.offset, static_cast<uint32_t>(patch.value));
				break;
			default: Put<uint64_t>(copy.bytes, patch.offset, patch.value); break;
		}
	}
	return BuildEboot({&copy}, tail_bytes);
}

constexpr uint64_t NEAR_MAX      = ~uint64_t {0} - 3;
constexpr uint64_t PAST_THE_FILE = 1u << 30;

// Truncation at every length, and each way the ELF or header can be wrong, is rejected.
void TestHostileElfs() {
	const auto elf  = BuildProgramElf(0, 1);
	const auto good = BuildEboot({&elf});
	CHECK(ScanEbootShaders(good).programs.size() == 1); // The baseline each variant differs from.

	// Every proper prefix of the ELF is missing part of the section table.
	const uint64_t elf_start = good.size() - elf.bytes.size() - TAIL_JUNK_BYTES;
	for (uint64_t length = 0; length < elf_start + elf.bytes.size(); length++) {
		const Bytes prefix(good.begin(), good.begin() + length);
		CHECK(ScanEbootShaders(prefix).programs.empty());
	}

	// Padding alone must not change the verdict, or the section-count variant below proves nothing.
	CHECK(ScanEbootShaders(EbootWithPatchedElf(elf, {}, TABLE_PADDING_BYTES)).programs.size() == 1);

	struct Variant {
		const char*        name;
		std::vector<Patch> patches;
		uint32_t           padding = 0;
	};
	const auto                 text_entry   = EntryAt(elf, TEXT_INDEX);
	const auto                 header_entry = EntryAt(elf, HEADER_INDEX);
	const std::vector<Variant> variants {
	    {"text offset past the file", {{text_entry + ENTRY_FILE_OFFSET, PAST_THE_FILE, 8}}},
	    {"text offset wraps", {{text_entry + ENTRY_FILE_OFFSET, NEAR_MAX, 8}}},
	    {"text size wraps", {{text_entry + ENTRY_SIZE_FIELD, NEAR_MAX, 8}}},
	    {"header offset past the file", {{header_entry + ENTRY_FILE_OFFSET, PAST_THE_FILE, 8}}},
	    {"header size too big", {{header_entry + ENTRY_SIZE_FIELD, PAST_THE_FILE, 8}}},
	    {"metadata size too big",
	     {{EntryAt(elf, METADATA_INDEX) + ENTRY_SIZE_FIELD, PAST_THE_FILE, 8}}},
	    {"names offset past the file",
	     {{EntryAt(elf, NAMES_INDEX) + ENTRY_FILE_OFFSET, PAST_THE_FILE, 8}}},
	    {"table offset past the file", {{ELF_SHOFF_FIELD, PAST_THE_FILE, 8}}},
	    {"table offset wraps", {{ELF_SHOFF_FIELD, NEAR_MAX, 8}}},
	    {"too many sections", {{ELF_SHNUM_FIELD, 0xFFFF, 2}}},
	    {"65 sections with room for the table",
	     {{ELF_SHNUM_FIELD, OVER_LIMIT_SECTIONS, 2}},
	     TABLE_PADDING_BYTES},
	    {"no sections", {{ELF_SHNUM_FIELD, 0, 2}}},
	    {"wrong entry size", {{ELF_SHENT_FIELD, 56, 2}}},
	    {"names index out of range", {{ELF_SHSTR_FIELD, 99, 2}}},
	    {"32-bit class", {{ELF_CLASS_FIELD, ELF_CLASS_32, 1}}},
	    {"big-endian data", {{ELF_DATA_FIELD, ELF_BIG_ENDIAN, 1}}},
	    {"header magic + 1",
	     {{elf.header_offset + Layout::OFFSET_MAGIC, Layout::FILE_MAGIC + 1, 4}}},
	    {"header version + 1",
	     {{elf.header_offset + Layout::OFFSET_VERSION, Layout::FILE_VERSION + 1, 4}}},
	};
	for (const auto& variant: variants) {
		const auto eboot  = EbootWithPatchedElf(elf, variant.patches, variant.padding);
		const auto result = ScanEbootShaders(eboot);
		if (!result.programs.empty()) {
			std::fprintf(stderr, "variant accepted: %s\n", variant.name);
		}
		CHECK(result.programs.empty());
	}
}

// Declared sizes and footers. A binary the runtime would hash whole is hashed whole, and one that
// claims bytes the ELF does not have is refused.
void TestDeclaredSizeAndFooter() {
	const auto binary = BuildBinary(1);
	const auto size   = static_cast<uint32_t>(binary.bytes.size());
	// Builds an eboot around a binary whose header declares `declared_size` bytes. The text section
	// holds the binary, followed by the metadata section.
	struct Built {
		Bytes    eboot;
		uint64_t text_at; // Where the text starts in the eboot.
	};
	const auto make = [&](uint32_t declared_size, const Binary& blob) {
		const auto elf   = BuildElf(BuildHeader(0, declared_size, StandardRegisters()), blob.bytes,
		                            MetadataBytes(1));
		auto       eboot = BuildEboot({&elf});
		const auto at    = eboot.size() - TAIL_JUNK_BYTES - elf.bytes.size() + elf.text_offset;
		return Built {eboot, at};
	};
	// The whole of the declared range, as the runtime hashes a binary with no footer.
	const auto whole_hash = [](const Built& built, uint32_t length) {
		return Xxh3(Bytes(built.eboot.begin() + built.text_at,
		                  built.eboot.begin() + built.text_at + length));
	};

	const auto baseline = make(size, binary);
	const auto good     = ScanEbootShaders(baseline.eboot);
	CHECK(good.programs.size() == 1 && good.counts.no_footer == 0);

	// Declared past the binary but inside the ELF, so the footer is not at the end.
	const auto longer = ScanEbootShaders(make(size + 4, binary).eboot);
	CHECK(longer.programs.size() == 1 && longer.counts.no_footer == 1);
	if (longer.programs.size() == 1) {
		CHECK(longer.programs[0].code_hash == whole_hash(make(size + 4, binary), size + 4));
		CHECK(longer.programs[0].code_size_bytes == 0 &&
		      longer.programs[0].literal_size_bytes == 0);
		CHECK(longer.programs[0].code.size() * sizeof(uint32_t) == size + 4);
	}
	const auto shorter = ScanEbootShaders(make(size - 4, binary).eboot);
	CHECK(shorter.programs.size() == 1 && shorter.counts.no_footer == 1);
	if (shorter.programs.size() == 1) {
		CHECK(shorter.programs[0].code_hash == whole_hash(make(size - 4, binary), size - 4));
	}
	// Smaller than a footer.
	const auto tiny = ScanEbootShaders(make(FOOTER_SIZE - 4, binary).eboot);
	CHECK(tiny.programs.size() == 1 && tiny.counts.no_footer == 1);

	// Footers the runtime does not accept. The binary is hashed whole.
	const auto wrong_version = BuildBinary(1, CODE_SIZE, LITERAL_SIZE, BAD_FOOTER_VERSION);
	const auto no_code       = BuildBinary(1, 0);
	auto       huge_literals = BuildBinary(1);
	Put<uint32_t>(huge_literals.bytes, size - FOOTER_SIZE + FOOTER_LITERAL_OFFSET,
	              HUGE_LITERAL_SIZE);
	for (const Binary* blob:
	     {&wrong_version, &no_code, static_cast<const Binary*>(&huge_literals)}) {
		const auto built  = make(static_cast<uint32_t>(blob->bytes.size()), *blob);
		const auto result = ScanEbootShaders(built.eboot);
		CHECK(result.programs.size() == 1 && result.counts.no_footer == 1);
		if (result.programs.size() == 1) {
			CHECK(result.programs[0].code_hash ==
			      whole_hash(built, static_cast<uint32_t>(blob->bytes.size())));
		}
	}

	// Sizes that cannot be a binary at all.
	CHECK(ScanEbootShaders(make(0, binary).eboot).programs.empty());
	CHECK(ScanEbootShaders(make(size + 2, binary).eboot).programs.empty());
	// Claims more than the file holds, and more than the ELF holds while the file has the bytes.
	const auto past_file = ScanEbootShaders(make(0xFFFFFFFCu, binary).eboot);
	CHECK(past_file.programs.empty() && past_file.counts.oversize == 1);
	const auto elf        = BuildElf(BuildHeader(0, size + BIG_TAIL_BYTES / 2, StandardRegisters()),
	                                 binary.bytes, MetadataBytes(1));
	const auto beyond_elf = ScanEbootShaders(BuildEboot({&elf}, BIG_TAIL_BYTES));
	CHECK(beyond_elf.programs.empty() && beyond_elf.counts.oversize == 1);
	// The same declared size is accepted once the ELF itself holds that many bytes, so the ELF's
	// own extent is what decided and not the size of the file.
	Bytes padded_text = binary.bytes;
	padded_text.resize(padded_text.size() + BIG_TAIL_BYTES, 0);
	const auto elf_with_room =
	    BuildElf(BuildHeader(0, size + BIG_TAIL_BYTES / 2, StandardRegisters()), padded_text,
	             MetadataBytes(1));
	const auto with_room = ScanEbootShaders(BuildEboot({&elf_with_room}, BIG_TAIL_BYTES));
	CHECK(with_room.programs.size() == 1 && with_room.counts.oversize == 0);
	// A text section that ends before the footer is still a valid program, as the runtime maps by
	// the header's size and not by the section's.
	const auto elf_short_text = BuildProgramElf(0, 1);
	for (const uint64_t text_size: {uint64_t {size - FOOTER_SIZE}, uint64_t {0}}) {
		const auto eboot = EbootWithPatchedElf(
		    elf_short_text,
		    {{EntryAt(elf_short_text, TEXT_INDEX) + ENTRY_SIZE_FIELD, text_size, 8}});
		const auto result = ScanEbootShaders(eboot);
		CHECK(result.programs.size() == 1 && result.counts.no_footer == 0);
		if (result.programs.size() == 1) {
			CHECK(result.programs[0].code_hash == Xxh3(binary.hashed));
		}
	}
}

// Two ELFs with the same code give one program, taken from the first.
void TestDeduplicateFirstWins() {
	const auto first  = BuildProgramElf(0, 1, 1);
	const auto second = BuildProgramElf(0, 1, 2); // Same code and header, different metadata.
	auto       other_registers   = StandardRegisters();
	other_registers.back().value = OTHER_THREADS_X;
	const auto changed_header    = BuildProgramElf(0, 1, 3, other_registers);
	const auto third             = BuildProgramElf(0, 2, 3);
	const auto result            = ScanEbootShaders(BuildEboot({&first, &second}));
	CHECK(result.programs.size() == 1);
	CHECK(result.counts.duplicates == 1);
	CHECK(result.counts.duplicate_header_differs == 0);
	if (result.programs.size() == 1) {
		// The metadata tells which ELF the program came from. The code cannot.
		CHECK(result.programs[0].metadata == MetadataBytes(1));
	}
	// A duplicate with another header is counted as such, and the first header is the one kept.
	const auto differs = ScanEbootShaders(BuildEboot({&first, &changed_header}));
	CHECK(differs.programs.size() == 1);
	CHECK(differs.counts.duplicates == 1 && differs.counts.duplicate_header_differs == 1);
	if (differs.programs.size() == 1) {
		const auto first_binary = BuildBinary(1);
		CHECK(
		    differs.programs[0].header ==
		    BuildHeader(0, static_cast<uint32_t>(first_binary.bytes.size()), StandardRegisters()));
	}
	// Different code is not merged.
	const auto distinct = ScanEbootShaders(BuildEboot({&first, &third}));
	CHECK(distinct.programs.size() == 2);
	CHECK(distinct.counts.duplicates == 0);
	// First in file order, even when the later ELF would sort first.
	const auto reversed = ScanEbootShaders(BuildEboot({&second, &first}));
	CHECK(reversed.programs.size() == 1);
	if (reversed.programs.size() == 1) CHECK(reversed.programs[0].metadata == MetadataBytes(2));
}

// ---- Decoder and static key tests. ----

// Every field of a decoded compute stage, so that a comparison cannot skip one.
std::vector<uint64_t> Fields(const HW::CsStageRegisters& r) {
	return {r.data_addr,
	        r.num_thread_x,
	        r.num_thread_y,
	        r.num_thread_z,
	        r.vgprs,
	        r.priority,
	        r.float_mode,
	        r.dx10_clamp,
	        r.debug_mode,
	        r.ieee_mode,
	        r.require_forward_progress,
	        r.fp16_overflow,
	        r.threadgroup_configuration,
	        r.wave_size,
	        r.scratch_en,
	        r.user_sgpr,
	        r.tgid_x_en,
	        r.tgid_y_en,
	        r.tgid_z_en,
	        r.tg_size_en,
	        r.tidig_comp_cnt,
	        r.lds_size,
	        r.shared_vgprs};
}

// A verbatim copy of the compute register switch of HwShSetCsRegister as it was before the decoder
// was pulled out. It is the oracle the shared decoder is compared with, and it is over 50 lines on
// purpose, because any edit to it would weaken the comparison.
bool OldDecode(HW::CsStageRegisters& cs_regs, uint32_t cmd_offset, uint32_t value) {
	switch (cmd_offset) {
		case Pm4::COMPUTE_PGM_LO:
			cs_regs.data_addr &= 0xFFFFFF00000000FFull;
			cs_regs.data_addr |= static_cast<uint64_t>(value) << 8u;
			break;
		case Pm4::COMPUTE_PGM_HI:
			cs_regs.data_addr &= 0xFFFF00FFFFFFFFFFull;
			cs_regs.data_addr |= (static_cast<uint64_t>(value) & 0xffu) << 40u;
			break;
		case Pm4::COMPUTE_PGM_RSRC1:
			cs_regs.vgprs =
			    (value >> Pm4::COMPUTE_PGM_RSRC1_VGPRS_SHIFT) & Pm4::COMPUTE_PGM_RSRC1_VGPRS_MASK;
			cs_regs.priority = (value >> Pm4::COMPUTE_PGM_RSRC1_PRIORITY_SHIFT) &
			                   Pm4::COMPUTE_PGM_RSRC1_PRIORITY_MASK;
			cs_regs.float_mode = (value >> Pm4::COMPUTE_PGM_RSRC1_FLOAT_MODE_SHIFT) &
			                     Pm4::COMPUTE_PGM_RSRC1_FLOAT_MODE_MASK;
			cs_regs.dx10_clamp    = ((value >> Pm4::COMPUTE_PGM_RSRC1_DX10_CLAMP_SHIFT) &
                                  Pm4::COMPUTE_PGM_RSRC1_DX10_CLAMP_MASK) != 0u;
			cs_regs.debug_mode    = ((value >> Pm4::COMPUTE_PGM_RSRC1_DEBUG_MODE_SHIFT) &
                                  Pm4::COMPUTE_PGM_RSRC1_DEBUG_MODE_MASK) != 0u;
			cs_regs.ieee_mode     = ((value >> Pm4::COMPUTE_PGM_RSRC1_IEEE_MODE_SHIFT) &
                                 Pm4::COMPUTE_PGM_RSRC1_IEEE_MODE_MASK) != 0u;
			cs_regs.fp16_overflow = ((value >> Pm4::COMPUTE_PGM_RSRC1_FP16_OVFL_SHIFT) &
			                         Pm4::COMPUTE_PGM_RSRC1_FP16_OVFL_MASK) != 0u;
			cs_regs.threadgroup_configuration = ((value >> Pm4::COMPUTE_PGM_RSRC1_WGP_MODE_SHIFT) &
			                                     Pm4::COMPUTE_PGM_RSRC1_WGP_MODE_MASK) != 0u;
			cs_regs.require_forward_progress =
			    ((value >> Pm4::COMPUTE_PGM_RSRC1_FWD_PROGRESS_SHIFT) &
			     Pm4::COMPUTE_PGM_RSRC1_FWD_PROGRESS_MASK) != 0u;
			break;
		case Pm4::COMPUTE_PGM_RSRC2:
			cs_regs.scratch_en = ((value >> Pm4::COMPUTE_PGM_RSRC2_SCRATCH_EN_SHIFT) &
			                      Pm4::COMPUTE_PGM_RSRC2_SCRATCH_EN_MASK) != 0u;
			cs_regs.user_sgpr  = (value >> Pm4::COMPUTE_PGM_RSRC2_USER_SGPR_SHIFT) &
			                    Pm4::COMPUTE_PGM_RSRC2_USER_SGPR_MASK;
			cs_regs.tgid_x_en      = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_X_EN_SHIFT) &
                                 Pm4::COMPUTE_PGM_RSRC2_TGID_X_EN_MASK) != 0u;
			cs_regs.tgid_y_en      = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_Y_EN_SHIFT) &
                                 Pm4::COMPUTE_PGM_RSRC2_TGID_Y_EN_MASK) != 0u;
			cs_regs.tgid_z_en      = ((value >> Pm4::COMPUTE_PGM_RSRC2_TGID_Z_EN_SHIFT) &
                                 Pm4::COMPUTE_PGM_RSRC2_TGID_Z_EN_MASK) != 0u;
			cs_regs.tg_size_en     = ((value >> Pm4::COMPUTE_PGM_RSRC2_TG_SIZE_EN_SHIFT) &
                                  Pm4::COMPUTE_PGM_RSRC2_TG_SIZE_EN_MASK) != 0u;
			cs_regs.tidig_comp_cnt = (value >> Pm4::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_SHIFT) &
			                         Pm4::COMPUTE_PGM_RSRC2_TIDIG_COMP_CNT_MASK;
			cs_regs.lds_size = (value >> Pm4::COMPUTE_PGM_RSRC2_LDS_SIZE_SHIFT) &
			                   Pm4::COMPUTE_PGM_RSRC2_LDS_SIZE_MASK;
			break;
		case Pm4::COMPUTE_PGM_RSRC3:
			cs_regs.shared_vgprs = (value >> Pm4::COMPUTE_PGM_RSRC3_SHARED_VGPRS_SHIFT) &
			                       Pm4::COMPUTE_PGM_RSRC3_SHARED_VGPRS_MASK;
			break;
		case Pm4::COMPUTE_NUM_THREAD_X: cs_regs.num_thread_x = value; break;
		case Pm4::COMPUTE_NUM_THREAD_Y: cs_regs.num_thread_y = value; break;
		case Pm4::COMPUTE_NUM_THREAD_Z: cs_regs.num_thread_z = value; break;
		case Pm4::COMPUTE_START_X:
		case Pm4::COMPUTE_START_Y:
		case Pm4::COMPUTE_START_Z:
		case Pm4::COMPUTE_RESOURCE_LIMITS:
		case Pm4::COMPUTE_TMPRING_SIZE:
		case Pm4::COMPUTE_PACE_ID: break;
		default: return false;
	}
	return true;
}

constexpr std::array<uint32_t, 14> COMPUTE_REGISTERS {
    Pm4::COMPUTE_PGM_LO,       Pm4::COMPUTE_PGM_HI,       Pm4::COMPUTE_PGM_RSRC1,
    Pm4::COMPUTE_PGM_RSRC2,    Pm4::COMPUTE_PGM_RSRC3,    Pm4::COMPUTE_NUM_THREAD_X,
    Pm4::COMPUTE_NUM_THREAD_Y, Pm4::COMPUTE_NUM_THREAD_Z, Pm4::COMPUTE_START_X,
    Pm4::COMPUTE_START_Y,      Pm4::COMPUTE_START_Z,      Pm4::COMPUTE_RESOURCE_LIMITS,
    Pm4::COMPUTE_TMPRING_SIZE, Pm4::COMPUTE_PACE_ID};
constexpr uint32_t BATTERY_WRITES = 600;
constexpr uint32_t RANDOM_SEED    = 2463534242u;
constexpr uint32_t XORSHIFT_LEFT  = 13;
constexpr uint32_t XORSHIFT_RIGHT = 17;
constexpr uint32_t XORSHIFT_TAIL  = 5;

// The shared decoder agrees with the code it replaced on every field after the same writes. Random
// values reach every bit of every field, so a field the decoder breaks is seen. The PM4 handler is
// the decoder's only other caller and is a three-line wrapper around it.
void TestDecoderMatchesOldCode() {
	// xorshift32, because the low bits of a linear congruential generator repeat with a short
	// period and would leave some registers never chosen.
	uint32_t   state = RANDOM_SEED;
	const auto next  = [&] {
        state ^= state << XORSHIFT_LEFT;
        state ^= state >> XORSHIFT_RIGHT;
        state ^= state << XORSHIFT_TAIL;
        return state;
	};
	std::vector<Reg>     writes;
	HW::CsStageRegisters oracle;
	HW::CsStageRegisters shared;
	bool                 all_equal = true;
	for (uint32_t i = 0; i < BATTERY_WRITES; i++) {
		const uint32_t reg   = COMPUTE_REGISTERS[next() % COMPUTE_REGISTERS.size()];
		const uint32_t value = next();
		writes.push_back({reg, value});
		CHECK(OldDecode(oracle, reg, value));
		CHECK(DecodeCsRegister(shared, reg, value));
		all_equal = all_equal && Fields(oracle) == Fields(shared);
	}
	CHECK(all_equal);
	// Every register must have been written, or a register left out of the battery could break
	// unseen.
	for (const auto reg: COMPUTE_REGISTERS) {
		CHECK(std::any_of(writes.begin(), writes.end(),
		                  [&](const Reg& write) { return write.offset == reg; }));
	}
	// A baseline that is not trivially equal, because the writes did change many fields.
	CHECK(Fields(oracle) != Fields(HW::CsStageRegisters {}));

	const HW::CsStageRegisters before = shared;
	CHECK(!DecodeCsRegister(shared, Pm4::COMPUTE_USER_DATA_0, UNKNOWN_SH_VALUE));
	CHECK(Fields(before) == Fields(shared));
}

// The key built from a header is the words the python reference gives for the same registers.
void TestStaticKeyGolden() {
	const auto header = BuildHeader(0, 128, StandardRegisters());
	const auto key    = BuildComputeStaticKey(header, WIDE_DEVICE, MOTOR_DISPATCH);
	CHECK(key.has_value());
	if (!key) return;
	const std::vector<uint32_t> expected {USER_SGPR,
	                                      MOTOR_DISPATCH.wave_size | (FLOAT_MODE << 8u),
	                                      HOST_SUBGROUP,
	                                      TIDIG_COMP_CNT + 1,
	                                      LDS_GRANULES * LDS_DWORDS_PER_GRANULE,
	                                      SCRATCH_DWORDS,
	                                      0,
	                                      THREADS_X,
	                                      1, // The enable bits are x and z only.
	                                      THREADS_Y,
	                                      0,
	                                      THREADS_Z,
	                                      1,
	                                      1};
	CHECK(key->state == expected);
	CHECK(key->user_data_count == USER_SGPR);
	CHECK(key->input_info.host_subgroup_size == HOST_SUBGROUP);

	// The key the runtime's own two functions make from registers decoded by the old code.
	HW::CsStageRegisters cs_regs;
	cs_regs.wave_size = static_cast<uint8_t>(MOTOR_DISPATCH.wave_size);
	for (const auto& reg: StandardRegisters())
		CHECK(OldDecode(cs_regs, reg.offset, reg.value));
	ShaderComputeInputInfo runtime {};
	ShaderGetStaticInputInfoCS(cs_regs, SCRATCH_DWORDS, runtime);
	runtime.host_subgroup_size = HOST_SUBGROUP;
	std::vector<uint32_t> runtime_state;
	BuildStageStaticKey(runtime, runtime_state);
	CHECK(runtime_state == key->state);
}

// The device and dispatch inputs reach the key, and lds_storage follows the runtime's comparison.
void TestDeviceAndDispatchInputs() {
	const auto header = BuildHeader(0, 128, StandardRegisters());
	const auto wide   = BuildComputeStaticKey(header, WIDE_DEVICE, MOTOR_DISPATCH);
	const auto narrow = BuildComputeStaticKey(header, NARROW_DEVICE, MOTOR_DISPATCH);
	CHECK(wide && narrow);
	if (!wide || !narrow) return;
	CHECK(wide->state[2] == HOST_SUBGROUP && narrow->state[2] == OTHER_SUBGROUP);
	CHECK(!wide->input_info.lds_storage && narrow->input_info.lds_storage);
	// Only the subgroup word differs, so the rest of the key does not depend on the device.
	auto wide_state = wide->state;
	wide_state[2]   = narrow->state[2];
	CHECK(wide_state == narrow->state);

	// Each dispatch input changes its own word and no other.
	const auto wave32 = BuildComputeStaticKey(header, WIDE_DEVICE, {OTHER_WAVE_SIZE, false, false});
	const auto threads =
	    BuildComputeStaticKey(header, WIDE_DEVICE, {MOTOR_DISPATCH.wave_size, true, false});
	const auto async =
	    BuildComputeStaticKey(header, WIDE_DEVICE, {MOTOR_DISPATCH.wave_size, false, true});
	CHECK(wave32 && threads && async);
	if (!wave32 || !threads || !async) return;
	const auto differing_words = [&](const ComputeStaticKey& other) {
		std::vector<size_t> words;
		for (size_t i = 0; i < other.state.size(); i++) {
			if (other.state[i] != wide->state[i]) words.push_back(i);
		}
		return words;
	};
	CHECK(differing_words(*wave32) == std::vector<size_t> {1});
	CHECK(differing_words(*threads) == std::vector<size_t> {6});
	CHECK(differing_words(*async) == std::vector<size_t> {6});
	CHECK(threads->state[6] != async->state[6]);
}

// Headers the key cannot be built from give nothing, and a register the decoder does not know is
// ignored where it would not change the key.
void TestHeaderFailuresAndUnknownRegisters() {
	const auto header = BuildHeader(0, 128, StandardRegisters());
	const auto base   = BuildComputeStaticKey(header, WIDE_DEVICE, MOTOR_DISPATCH);
	CHECK(base.has_value());

	auto with_unknown = StandardRegisters();
	with_unknown.insert(with_unknown.begin() + 2, {UNKNOWN_SH_REGISTER, UNKNOWN_SH_VALUE});
	with_unknown.push_back({Pm4::COMPUTE_START_Y, UNKNOWN_START_Y_VALUE});
	const auto unknown_key =
	    BuildComputeStaticKey(BuildHeader(0, 128, with_unknown), WIDE_DEVICE, MOTOR_DISPATCH);
	CHECK(unknown_key.has_value());
	if (base && unknown_key) CHECK(unknown_key->state == base->state);

	const auto no_rsrc2 = BuildHeader(0, 128, {{Pm4::COMPUTE_PGM_RSRC1, RSRC1}});
	CHECK(!BuildComputeStaticKey(no_rsrc2, WIDE_DEVICE, MOTOR_DISPATCH));
	const auto no_rsrc1 = BuildHeader(0, 128, {{Pm4::COMPUTE_PGM_RSRC2, RSRC2}});
	CHECK(!BuildComputeStaticKey(no_rsrc1, WIDE_DEVICE, MOTOR_DISPATCH));
	// A register table that runs past the section is refused.
	Bytes cut = header;
	cut.resize(cut.size() - 4);
	CHECK(!BuildComputeStaticKey(cut, WIDE_DEVICE, MOTOR_DISPATCH));
	CHECK(!BuildComputeStaticKey(Bytes(8, 0), WIDE_DEVICE, MOTOR_DISPATCH));
	// Registers are listed but the table was never placed. A null pointer read as an offset of zero
	// would find a valid table right after the pointer field, so the header holds one there, and
	// only the rule that refuses a null table makes it fail.
	Bytes unplaced = header;
	Put<uint64_t>(unplaced, Layout::OFFSET_SH_REGISTERS, 0);
	constexpr uint64_t SECOND_ENTRY = Layout::OFFSET_SH_REGISTERS + Layout::SH_REGISTER_SIZE;
	Put<uint32_t>(unplaced, SECOND_ENTRY, Pm4::COMPUTE_PGM_RSRC1);
	Put<uint32_t>(unplaced, SECOND_ENTRY + sizeof(uint32_t), RSRC1);
	Put<uint32_t>(unplaced, SECOND_ENTRY + Layout::SH_REGISTER_SIZE, Pm4::COMPUTE_PGM_RSRC2);
	Put<uint32_t>(unplaced, SECOND_ENTRY + Layout::SH_REGISTER_SIZE + sizeof(uint32_t), RSRC2);
	CHECK(!BuildComputeStaticKey(unplaced, WIDE_DEVICE, MOTOR_DISPATCH));
}

// ---- sl00 metadata tests. ----

// Attribute and form numbers of the format, spelled out here so the tests do not depend on the
// decoder's constants.
constexpr uint32_t SL_FORM_ULEB    = 1; // Every form but 80 is one variable length integer.
constexpr uint32_t SL_FORM_FLAG    = 80;
constexpr uint32_t SL_ATTR_HEAD    = 30; // A constant buffer's mark, and a struct's first member.
constexpr uint32_t SL_ATTR_OFFSET  = 33;
constexpr uint32_t SL_ATTR_ELEMENT = 36;
constexpr uint32_t SL_ATTR_SIZE    = 37;
constexpr uint32_t SL_ATTR_READ_WRITE = 40;
constexpr uint32_t SL_ATTR_KIND       = 41;
constexpr uint32_t SL_ATTR_COUNT      = 43;
constexpr uint32_t SL_ATTR_OTHER      = 99;
// Abbreviation tags. An odd tag gives the record a sibling link.
constexpr uint32_t SL_TAG_CONSTANT = 2;
constexpr uint32_t SL_TAG_RESOURCE = 4;
constexpr uint32_t SL_TAG_ELEMENT  = 6;
constexpr uint32_t SL_TAG_STRUCT   = 8;
constexpr uint32_t SL_TAG_LINKED   = 9;
constexpr uint32_t SL_TAG_LAST     = 10;
// Resource flags and element kinds.
constexpr uint32_t SL_FLAG_TYPED      = 0;
constexpr uint32_t SL_FLAG_STRUCTURED = 1;
constexpr uint32_t SL_FLAG_RAW        = 2;
constexpr uint32_t SL_KIND_FLOAT      = 1;
constexpr uint32_t SL_KIND_UINT       = 9;
constexpr uint32_t SL_KIND_IMAGE      = 0;
// The flags that make an image, and flags that make nothing the decoder knows.
constexpr std::array<uint64_t, 3> SL_IMAGE_FLAGS {7, 8, 13};
constexpr std::array<uint64_t, 8> SL_OTHER_FLAGS {3, 4, 5, 6, 9, 12, 14, 255};
// The decoder keeps only rows that are at most this far apart. A gap of zeros that long ends the
// run.
constexpr uint32_t SL_ROW_GAP           = 16;
constexpr uint32_t SL_STREAM_GAP        = SL_ROW_GAP + 4;
constexpr uint32_t SL_TRAILING_BYTES    = 4;
constexpr uint8_t  SL_FIRST_REGISTER    = 3;
constexpr uint8_t  SL_SECOND_REGISTER   = SL_FIRST_REGISTER + 1;
constexpr uint8_t  SL_THIRD_REGISTER    = SL_FIRST_REGISTER + 2;
constexpr uint8_t  SL_REGISTER_LIMIT    = 0x80;
constexpr uint32_t SL_WIDE_ULEB_BYTES   = 4;        // How far the game pads some values.
constexpr uint32_t SL_FILLER_NAME_BYTES = 130;      // Pushes later names past offset 127.
constexpr uint32_t SL_MAX_METADATA      = 1u << 20; // The decoder's size cap.
constexpr uint32_t SL_MAX_RESOURCES     = 1024;     // The decoder's record cap.
constexpr uint32_t SL_HUGE_RUN_BYTES    = 128 * 1024;
constexpr uint32_t SL_LONG_ULEB_BYTES   = 10;
// The form 80 probe: each member is a three byte code and a three byte link.
constexpr uint32_t SL_PROBE_PAD          = 3;
constexpr uint32_t SL_PROBE_MEMBER_BYTES = 2 * SL_PROBE_PAD;
constexpr uint32_t SL_PROBE_SMALL        = 20;
constexpr uint32_t SL_PROBE_UNDER        = 150;
constexpr uint32_t SL_PROBE_OVER         = 200;
constexpr uint32_t SL_PROBE_HUGE         = 3000;

// A variable length integer of at least `min_bytes` bytes, padded with continuation bytes.
Bytes Uleb(uint64_t value, uint32_t min_bytes = 1) {
	Bytes bytes;
	do {
		auto byte = static_cast<uint8_t>(value & 0x7Fu);
		value >>= 7u;
		if (value != 0 || bytes.size() + 1 < min_bytes) byte |= 0x80u;
		bytes.push_back(byte);
	} while (value != 0 || bytes.size() < min_bytes);
	return bytes;
}

using AttributeForms = std::vector<std::pair<uint32_t, uint32_t>>;

// The attributes with the ordinary form, which is one variable length integer each.
AttributeForms Plain(std::initializer_list<uint32_t> attributes) {
	AttributeForms pairs;
	for (const auto attribute: attributes)
		pairs.emplace_back(attribute, SL_FORM_ULEB);
	return pairs;
}

// Writes an sl00 section. The strings come first and are the names the resource rows refer to, an
// empty string ends them, and then abbreviations, records and finally the resource rows follow.
class Sl00Writer {
public:
	explicit Sl00Writer(const std::vector<std::string>& names,
	                    const std::string&              version = "1.2.3.4 build 7") {
		// An abbreviation at offset 0 that records with code 0 can use, then the version text.
		m_bytes = {3, 5, 6, 0, 0};
		AddText(version);
		for (const auto& name: names) {
			m_offsets[name] = Here();
			AddText(name);
		}
		m_start = Here();
		m_bytes.push_back(0);
	}

	uint32_t Here() const { return static_cast<uint32_t>(m_bytes.size()); }
	// Where the record stream starts, which is the empty string that ends the table.
	uint32_t Start() const { return m_start; }
	uint32_t NameOffset(const std::string& name) const { return m_offsets.at(name); }

	// An abbreviation, which is a tag and the (attribute, form) pairs and then a 0, 0 pair.
	uint32_t Abbrev(uint32_t tag, const AttributeForms& pairs, bool terminate = true) {
		const uint32_t at = Here();
		AppendBytes(m_bytes, Uleb(tag));
		for (const auto& [attribute, form]: pairs) {
			AppendBytes(m_bytes, Uleb(attribute));
			AppendBytes(m_bytes, Uleb(form));
		}
		if (terminate) {
			m_bytes.push_back(0);
			m_bytes.push_back(0);
		}
		return at;
	}

	// A record, which is its abbreviation's offset, the sibling link of an odd tag and the values.
	// A form 80 attribute has no value, so the caller leaves it out.
	uint32_t Die(uint32_t abbreviation, const std::vector<uint64_t>& values,
	             std::optional<uint64_t> link = std::nullopt, uint32_t pad = 1) {
		const uint32_t at = Here();
		AppendBytes(m_bytes, Uleb(abbreviation, pad));
		if (link) AppendBytes(m_bytes, Uleb(*link, pad));
		for (const auto value: values)
			AppendBytes(m_bytes, Uleb(value, pad));
		return at;
	}

	void Zeros(uint32_t count) { m_bytes.insert(m_bytes.end(), count, 0); }
	void Raw(const Bytes& bytes) { AppendBytes(m_bytes, bytes); }

	// A resource row, which is the register, the name, the type reference and a flags byte. It
	// returns where the name starts, which is the row's position.
	uint32_t Resource(uint8_t register_index, const std::string& name, uint64_t type_at,
	                  uint32_t pad = 1) {
		return ResourceRaw(register_index, Uleb(NameOffset(name), pad), Uleb(type_at, pad));
	}

	uint32_t ResourceRaw(uint8_t register_index, const Bytes& name_bytes, const Bytes& type_bytes) {
		m_bytes.push_back(register_index);
		const uint32_t at = Here();
		AppendBytes(m_bytes, name_bytes);
		AppendBytes(m_bytes, type_bytes);
		m_bytes.push_back(1);
		return at;
	}

	// The section with zero bytes after the last row.
	Bytes Finish(uint32_t trailing = SL_TRAILING_BYTES) const {
		Bytes bytes = m_bytes;
		bytes.insert(bytes.end(), trailing, 0);
		return bytes;
	}

private:
	void AddText(const std::string& text) {
		m_bytes.insert(m_bytes.end(), text.begin(), text.end());
		m_bytes.push_back(0);
	}

	Bytes                                     m_bytes;
	std::unordered_map<std::string, uint32_t> m_offsets;
	uint32_t                                  m_start = 0;
};

// One record in words, so that a test can name the whole expectation in one string.
std::string ShapeText(const Sl00Shape& shape) {
	const auto number = [](const std::optional<uint64_t>& value) {
		return value ? std::to_string(*value) : std::string("-");
	};
	const auto element = [&] {
		return " " + number(shape.element_kind) + " " + number(shape.element_count);
	};
	switch (shape.kind) {
		case Sl00ShapeKind::Unreadable: return std::string("unreadable");
		case Sl00ShapeKind::Other: return "other " + number(shape.flag) + element();
		case Sl00ShapeKind::ConstantBuffer: return std::string("cb");
		case Sl00ShapeKind::Raw: return std::string("raw");
		case Sl00ShapeKind::Typed: return "typed" + element();
		case Sl00ShapeKind::Structured: return "structured " + std::to_string(shape.stride);
		case Sl00ShapeKind::Image: return "image " + number(shape.flag) + element();
	}
	return "?";
}

std::string ResourceText(const Sl00Resource& resource) {
	return resource.name + " r" + std::to_string(resource.register_index) + " " +
	       (resource.resource_class != Sl00Class::Unknown
	            ? std::string(1, Sl00ClassLetter(resource.resource_class))
	            : std::string("-")) +
	       " " + ShapeText(resource.shape);
}

using Texts = std::vector<std::string>;

// The decoded records in words, or a single marker when the decoder gave no value.
Texts DecodeTexts(const Bytes& metadata) {
	const auto records = DecodeSl00Resources(metadata);
	if (!records) return {"<no value>"};
	Texts texts;
	for (const auto& record: *records)
		texts.push_back(ResourceText(record));
	return texts;
}

// Reports a mismatch with both sides, so that a failure says what differed.
void CheckTexts(const Texts& actual, const Texts& expected, const char* what) {
	if (actual != expected) {
		std::fprintf(stderr, "%s differs\n", what);
		for (const auto& text: actual)
			std::fprintf(stderr, "  got  %s\n", text.c_str());
		for (const auto& text: expected)
			std::fprintf(stderr, "  want %s\n", text.c_str());
	}
	CHECK(actual == expected);
}

// A section with one resource named gOne whose type record is written by `make_type`.
Bytes BuildSingle(const std::function<uint32_t(Sl00Writer&)>& make_type,
                  uint8_t                                     register_index = SL_FIRST_REGISTER,
                  uint32_t                                    trailing       = SL_TRAILING_BYTES) {
	Sl00Writer writer({"gOne"});
	const auto type_at = make_type(writer);
	writer.Zeros(SL_STREAM_GAP);
	writer.Resource(register_index, "gOne", type_at);
	return writer.Finish(trailing);
}

// The one record's description, or the marker when the count is not one.
std::string SingleText(const Bytes& metadata) {
	const auto texts = DecodeTexts(metadata);
	return texts.size() == 1 ? texts[0] : "<" + std::to_string(texts.size()) + " records>";
}

// A typed-looking resource record with an element offset, a flag and an optional read-write mark.
uint32_t TypedRecord(Sl00Writer& writer, std::optional<uint64_t> element_at, uint64_t flag,
                     bool read_write = false) {
	AttributeForms        pairs = Plain({SL_ATTR_ELEMENT, SL_ATTR_KIND});
	std::vector<uint64_t> values {element_at.value_or(0), flag};
	if (!element_at) {
		pairs  = Plain({SL_ATTR_KIND});
		values = {flag};
	}
	if (read_write) {
		pairs.emplace_back(SL_ATTR_READ_WRITE, SL_FORM_ULEB);
		values.push_back(0);
	}
	return writer.Die(writer.Abbrev(SL_TAG_RESOURCE, pairs), values);
}

// An element record with the scalar kind and the component count.
uint32_t ElementRecord(Sl00Writer& writer, uint64_t kind, uint64_t count) {
	return writer.Die(writer.Abbrev(SL_TAG_ELEMENT, Plain({SL_ATTR_KIND, SL_ATTR_COUNT})),
	                  {kind, count});
}

// A resource of one flag over an element of kind float and count four.
std::string FlagText(uint64_t flag) {
	return SingleText(BuildSingle([&](Sl00Writer& writer) {
		return TypedRecord(writer, ElementRecord(writer, SL_KIND_FLOAT, 4), flag);
	}));
}

// ---- The kinds of resource. ----

// The members of a struct are written last first, so that each link points back to a record that
// exists. The largest end is in the middle, so the first, the last and the largest all differ.
constexpr uint64_t CHAIN_SIZE      = 40;
constexpr uint64_t CHAIN_FIRST_END = 8;
constexpr uint64_t CHAIN_LAST_END  = 12;
constexpr uint64_t CHAIN_BIG_SIZE  = 32;

// Abbreviations of struct members, one with a sibling link and one without.
struct MemberAbbrevs {
	uint32_t linked;
	uint32_t last;
	uint32_t no_size;
};

MemberAbbrevs WriteMemberAbbrevs(Sl00Writer& writer) {
	return {writer.Abbrev(SL_TAG_LINKED, Plain({SL_ATTR_OFFSET, SL_ATTR_SIZE})),
	        writer.Abbrev(SL_TAG_LAST, Plain({SL_ATTR_OFFSET, SL_ATTR_SIZE})),
	        writer.Abbrev(SL_TAG_LINKED, Plain({SL_ATTR_OFFSET}))};
}

// A struct of three members that ends at 40, which is first (0, 8), then (8, 32), then (8, 4).
uint32_t WriteChain(Sl00Writer& writer, const MemberAbbrevs& abbrevs) {
	const auto third  = writer.Die(abbrevs.last, {8, 4});
	const auto second = writer.Die(abbrevs.linked, {8, CHAIN_BIG_SIZE}, third);
	return writer.Die(abbrevs.linked, {0, 8}, second);
}

// A structured resource over a struct whose first member is written by `members`. The text is that
// of the one resource.
std::string StructText(const std::function<uint32_t(Sl00Writer&, const MemberAbbrevs&)>& members) {
	return SingleText(BuildSingle([&](Sl00Writer& writer) {
		const auto abbrevs = WriteMemberAbbrevs(writer);
		const auto head    = members(writer, abbrevs);
		const auto record = writer.Die(writer.Abbrev(SL_TAG_STRUCT, Plain({SL_ATTR_HEAD})), {head});
		return TypedRecord(writer, record, SL_FLAG_STRUCTURED);
	}));
}

// The section with a resource of every kind. The names start behind a filler so that name offsets
// need two bytes, as do the references.
struct KindsBlob {
	Bytes    bytes;
	uint32_t typed_at;
	uint32_t typed_abbreviation;
};

KindsBlob BuildKinds() {
	const std::string filler(SL_FILLER_NAME_BYTES, 'F');
	Sl00Writer        writer({filler, "gConstants", "gRaw", "gTyped", "gTypedRw", "gStructured",
	                          "gStructStruct", "gImage", "gOther"});
	const auto        a_constant = writer.Abbrev(SL_TAG_CONSTANT, Plain({SL_ATTR_HEAD}));
	const auto        a_raw      = writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND}));
	const auto a_typed = writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_ELEMENT, SL_ATTR_KIND}));
	const auto a_rw =
	    writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_ELEMENT, SL_ATTR_KIND, SL_ATTR_READ_WRITE}));
	const auto abbrevs   = WriteMemberAbbrevs(writer);
	const auto float4    = ElementRecord(writer, SL_KIND_FLOAT, 4);
	const auto uint2     = ElementRecord(writer, SL_KIND_UINT, 2);
	const auto image     = ElementRecord(writer, SL_KIND_IMAGE, 4);
	const auto chain     = WriteChain(writer, abbrevs);
	const auto struct_at = writer.Die(writer.Abbrev(SL_TAG_STRUCT, Plain({SL_ATTR_HEAD})), {chain});
	const auto constant  = writer.Die(a_constant, {0});
	const auto raw       = writer.Die(a_raw, {SL_FLAG_RAW});
	const auto typed     = writer.Die(a_typed, {float4, SL_FLAG_TYPED});
	const auto typed_rw  = writer.Die(a_rw, {float4, SL_FLAG_TYPED, 0});
	const auto structured = writer.Die(a_typed, {uint2, SL_FLAG_STRUCTURED});
	const auto in_struct  = writer.Die(a_typed, {struct_at, SL_FLAG_STRUCTURED});
	const auto as_image   = writer.Die(a_typed, {image, 8});
	const auto other      = writer.Die(a_typed, {float4, 5});
	writer.Zeros(SL_STREAM_GAP);
	uint8_t reg = SL_FIRST_REGISTER;
	writer.Resource(reg++, "gConstants", constant);
	writer.Resource(reg++, "gRaw", raw);
	writer.Resource(reg++, "gTyped", typed);
	writer.Resource(reg++, "gTypedRw", typed_rw);
	writer.Resource(reg++, "gStructured", structured);
	writer.Resource(reg++, "gStructStruct", in_struct);
	writer.Resource(reg++, "gImage", as_image);
	writer.Resource(reg++, "gOther", other);
	return {writer.Finish(), typed, a_typed};
}

// Every kind of shape and class comes out, last record first. Each expected string is distinct from
// the others, so a decoder that gave one shape or class to all would fail.
void TestSl00ResourceKinds() {
	const auto  kinds = BuildKinds();
	const Texts expected {"gOther r10 T other 5 1 4",
	                      "gImage r9 T image 8 0 4",
	                      "gStructStruct r8 T structured 40",
	                      "gStructured r7 T structured 8",
	                      "gTypedRw r6 U typed 1 4",
	                      "gTyped r5 T typed 1 4",
	                      "gRaw r4 T raw",
	                      "gConstants r3 B cb"};
	CheckTexts(DecodeTexts(kinds.bytes), expected, "kinds");

	// The record that a row refers to and its abbreviation are reported as the writer placed them.
	const auto records = DecodeSl00Resources(kinds.bytes);
	CHECK(records && records->size() == expected.size());
	if (!records || records->size() != expected.size()) return;
	const auto& typed = (*records)[expected.size() - 1 - 2];
	CHECK(typed.name == "gTyped");
	CHECK(typed.type_offset == kinds.typed_at);
	CHECK(typed.abbreviation_offset == kinds.typed_abbreviation);
	CHECK(typed.register_index == SL_FIRST_REGISTER + 2);

	// A shape holds only what python's tuple has for its kind, so the fields of another kind are
	// empty.
	const auto& structured = (*records)[expected.size() - 1 - 4];
	const auto& raw        = (*records)[expected.size() - 1 - 1];
	const auto& constant   = (*records)[expected.size() - 1 - 0];
	CHECK(!typed.shape.flag && typed.shape.element_kind && typed.shape.element_count);
	CHECK(typed.shape.stride == 0);
	CHECK(!structured.shape.flag && !structured.shape.element_kind &&
	      !structured.shape.element_count);
	CHECK(!raw.shape.flag && !raw.shape.element_kind && !raw.shape.element_count);
	CHECK(!constant.shape.flag && !constant.shape.element_kind && !constant.shape.element_count);
}

// A tag wider than 64 bits keeps its low bit, since python's integer does. The record with an even
// wide tag has no sibling link, and the baseline with an odd one has it.
void TestSl00WideTag() {
	const auto make = [](uint8_t first_byte) {
		return SingleText(BuildSingle([&](Sl00Writer& writer) {
			const auto abbreviation = writer.Here();
			// Ten continuation bytes then a last byte, which is a tag of 2^70 plus the low bit.
			Bytes tag(SL_LONG_ULEB_BYTES, 0x80);
			tag[0] = first_byte;
			tag.push_back(0x01);
			writer.Raw(tag);
			writer.Raw(Uleb(SL_ATTR_KIND));
			writer.Raw(Uleb(SL_FORM_ULEB));
			writer.Raw({0, 0});
			const bool odd = (first_byte & 1) != 0;
			return writer.Die(abbreviation, {SL_FLAG_RAW},
			                  odd ? std::optional<uint64_t> {0} : std::nullopt);
		}));
	};
	CHECK(make(0x80) == "gOne r3 T raw");
	CHECK(make(0x81) == "gOne r3 T raw");
}

// Attribute 40 is the only difference between a read-write resource and a read-only one.
void TestSl00ReadWriteAttribute() {
	const auto make = [](bool read_write) {
		return SingleText(BuildSingle([&](Sl00Writer& writer) {
			return TypedRecord(writer, ElementRecord(writer, SL_KIND_FLOAT, 4), SL_FLAG_TYPED,
			                   read_write);
		}));
	};
	CHECK(make(false) == "gOne r3 T typed 1 4");
	CHECK(make(true) == "gOne r3 U typed 1 4");
}

// A constant buffer is marked by attribute 30 alone, and the same record without it is a raw buffer
// with no element, so the mark is what decided.
void TestSl00ConstantBufferMark() {
	const auto make = [](uint32_t attribute) {
		return SingleText(BuildSingle([&](Sl00Writer& writer) {
			return writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({attribute})), {SL_FLAG_RAW});
		}));
	};
	CHECK(make(SL_ATTR_HEAD) == "gOne r3 B cb");
	CHECK(make(SL_ATTR_KIND) == "gOne r3 T raw");
}

// The flag decides the shape and nothing else does, so one record is built with each flag value.
void TestSl00Flags() {
	CHECK(FlagText(SL_FLAG_TYPED) == "gOne r3 T typed 1 4");
	CHECK(FlagText(SL_FLAG_RAW) == "gOne r3 T raw");
	CHECK(FlagText(SL_FLAG_STRUCTURED) == "gOne r3 T structured 16");
	for (const uint64_t image_flag: SL_IMAGE_FLAGS) {
		CHECK(FlagText(image_flag) == "gOne r3 T image " + std::to_string(image_flag) + " 1 4");
	}
	for (const uint64_t other_flag: SL_OTHER_FLAGS) {
		CHECK(FlagText(other_flag) == "gOne r3 T other " + std::to_string(other_flag) + " 1 4");
	}
}

// ---- Struct size and the sibling link. ----

// The odd-tag link is what connects members, so the size is the largest end of all three. Each
// way the chain can be cut gives the size of the members read before the cut.
void TestSl00StructChain() {
	const auto whole = StructText([](Sl00Writer& writer, const MemberAbbrevs& abbrevs) {
		return WriteChain(writer, abbrevs);
	});
	// The largest is in the middle, so the first member (8), the last (12), the biggest size alone
	// (32) and the largest end (40) all differ.
	CHECK(whole == "gOne r3 T structured " + std::to_string(CHAIN_SIZE));
	CHECK(CHAIN_SIZE != CHAIN_FIRST_END && CHAIN_SIZE != CHAIN_LAST_END &&
	      CHAIN_SIZE != CHAIN_BIG_SIZE);

	// A link that points back at its own record ends the walk after that member.
	const auto cycle = StructText([](Sl00Writer& writer, const MemberAbbrevs& abbrevs) {
		const auto self = writer.Here();
		return writer.Die(abbrevs.linked, {0, 8}, self);
	});
	CHECK(cycle == "gOne r3 T structured " + std::to_string(CHAIN_FIRST_END));

	// A link past the end of the blob ends the walk, it is not an error.
	const auto beyond = StructText([](Sl00Writer& writer, const MemberAbbrevs& abbrevs) {
		return writer.Die(abbrevs.linked, {0, 8}, SL_MAX_METADATA);
	});
	CHECK(beyond == "gOne r3 T structured " + std::to_string(CHAIN_FIRST_END));

	// A member with no size ends the walk, and the members behind it are not counted.
	const auto no_size = StructText([](Sl00Writer& writer, const MemberAbbrevs& abbrevs) {
		const auto tail   = WriteChain(writer, abbrevs);
		const auto middle = writer.Die(abbrevs.no_size, {0}, tail);
		return writer.Die(abbrevs.linked, {0, 8}, middle);
	});
	CHECK(no_size == "gOne r3 T structured " + std::to_string(CHAIN_FIRST_END));

	// A chain of one member without a link, as an even tag has none.
	const auto single = StructText([](Sl00Writer& writer, const MemberAbbrevs& abbrevs) {
		return writer.Die(abbrevs.last, {2, 6});
	});
	CHECK(single == "gOne r3 T structured 8");

	// A struct with no first member has size 0.
	const auto empty = SingleText(BuildSingle([](Sl00Writer& writer) {
		const auto record = writer.Die(writer.Abbrev(SL_TAG_STRUCT, Plain({SL_ATTR_OFFSET})), {0});
		return TypedRecord(writer, record, SL_FLAG_STRUCTURED);
	}));
	CHECK(empty == "gOne r3 T structured 0");
}

// A structured resource takes four bytes per component when its element has a kind and a count,
// and the size of its struct otherwise. The same members and flag are used in every variant.
void TestSl00StrideRules() {
	const auto stride = [](std::optional<uint64_t> kind, std::optional<uint64_t> count) {
		return SingleText(BuildSingle([&](Sl00Writer& writer) {
			const auto            abbrevs = WriteMemberAbbrevs(writer);
			const auto            member  = writer.Die(abbrevs.last, {0, 24});
			AttributeForms        pairs;
			std::vector<uint64_t> values;
			if (kind) {
				pairs.emplace_back(SL_ATTR_KIND, SL_FORM_ULEB);
				values.push_back(*kind);
			}
			if (count) {
				pairs.emplace_back(SL_ATTR_COUNT, SL_FORM_ULEB);
				values.push_back(*count);
			}
			pairs.emplace_back(SL_ATTR_HEAD, SL_FORM_ULEB);
			values.push_back(member);
			const auto element = writer.Die(writer.Abbrev(SL_TAG_ELEMENT, pairs), values);
			return TypedRecord(writer, element, SL_FLAG_STRUCTURED);
		}));
	};
	// The struct is 24 bytes, so 4 * 4 = 16 can only come from the count.
	CHECK(stride(SL_KIND_FLOAT, 4) == "gOne r3 T structured 16");
	CHECK(stride(SL_KIND_FLOAT, 0) == "gOne r3 T structured 24"); // A count of zero counts as none.
	CHECK(stride(std::nullopt, 4) == "gOne r3 T structured 24");  // A count needs a kind.
	CHECK(stride(SL_KIND_FLOAT, std::nullopt) == "gOne r3 T structured 24");
	CHECK(stride(SL_KIND_UINT, 2) == "gOne r3 T structured 8");

	// A structured resource with no element has size 0, and with an element that is not in the blob
	// the whole record is unreadable, which python gets from an IndexError.
	const auto no_element = SingleText(BuildSingle(
	    [](Sl00Writer& writer) { return TypedRecord(writer, std::nullopt, SL_FLAG_STRUCTURED); }));
	CHECK(no_element == "gOne r3 T structured 0");
	const auto outside = [](uint64_t flag) {
		return SingleText(BuildSingle(
		    [&](Sl00Writer& writer) { return TypedRecord(writer, SL_MAX_METADATA, flag); }));
	};
	CHECK(outside(SL_FLAG_STRUCTURED) == "gOne r3 - unreadable");
	// A typed resource with the same element offset only has no element, so the guard on the offset
	// is what separates the two.
	CHECK(outside(SL_FLAG_TYPED) == "gOne r3 T typed - -");
}

// ---- Form 80 and value encodings. ----

// A form 80 attribute has no data and reads as 1. The records below differ from their baselines
// only in the form of one attribute.
void TestSl00FormFlag() {
	const auto text = [](const AttributeForms&                          pairs,
	                     std::function<std::vector<uint64_t>(uint32_t)> values) {
		return SingleText(BuildSingle([&](Sl00Writer& writer) {
			const auto element = ElementRecord(writer, SL_KIND_UINT, 2);
			return writer.Die(writer.Abbrev(SL_TAG_RESOURCE, pairs), values(element));
		}));
	};
	// The flag attribute as form 80 reads as 1, which is structured, and has no value in the
	// record.
	const auto flag_form = text({{SL_ATTR_ELEMENT, SL_FORM_ULEB}, {SL_ATTR_KIND, SL_FORM_FLAG}},
	                            [](uint32_t element) { return std::vector<uint64_t> {element}; });
	const auto flag_data = text(Plain({SL_ATTR_ELEMENT, SL_ATTR_KIND}), [](uint32_t element) {
		return std::vector<uint64_t> {element, SL_FLAG_STRUCTURED};
	});
	CHECK(flag_form == "gOne r3 T structured 8");
	CHECK(flag_form == flag_data);
	// A form 80 attribute of no interest takes no bytes, so what follows it is not shifted. The
	// baseline has the same attributes without it.
	const auto with_flag =
	    text({{SL_ATTR_OTHER, SL_FORM_FLAG},
	          {SL_ATTR_ELEMENT, SL_FORM_ULEB},
	          {SL_ATTR_KIND, SL_FORM_ULEB}},
	         [](uint32_t element) { return std::vector<uint64_t> {element, SL_FLAG_TYPED}; });
	const auto without = text(Plain({SL_ATTR_ELEMENT, SL_ATTR_KIND}), [](uint32_t element) {
		return std::vector<uint64_t> {element, SL_FLAG_TYPED};
	});
	CHECK(with_flag == "gOne r3 T typed 9 2");
	CHECK(with_flag == without);
	// The same attribute as an ordinary form does take a value, so the form is what is read.
	const auto as_value =
	    text(Plain({SL_ATTR_OTHER, SL_ATTR_ELEMENT, SL_ATTR_KIND}),
	         [](uint32_t element) { return std::vector<uint64_t> {0, element, SL_FLAG_TYPED}; });
	CHECK(as_value == with_flag);
}

// A repeated attribute keeps its last value, as a python dict does.
void TestSl00RepeatedAttribute() {
	const auto text = [](uint64_t first, uint64_t second) {
		return SingleText(BuildSingle([&](Sl00Writer& writer) {
			const auto element = ElementRecord(writer, SL_KIND_FLOAT, 4);
			const auto abbrev  = writer.Abbrev(SL_TAG_RESOURCE,
			                                   Plain({SL_ATTR_ELEMENT, SL_ATTR_KIND, SL_ATTR_KIND}));
			return writer.Die(abbrev, {element, first, second});
		}));
	};
	CHECK(text(SL_FLAG_TYPED, SL_FLAG_RAW) == "gOne r3 T raw");
	CHECK(text(SL_FLAG_RAW, SL_FLAG_TYPED) == "gOne r3 T typed 1 4");
}

// Values and references padded to four bytes read as the same numbers. The same section is built
// with and without the padding.
void TestSl00PaddedValues() {
	const auto make = [](uint32_t pad) {
		Sl00Writer writer({"gA", "gB"});
		const auto abbrev = writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_ELEMENT, SL_ATTR_KIND}));
		const auto element_abbrev =
		    writer.Abbrev(SL_TAG_ELEMENT, Plain({SL_ATTR_KIND, SL_ATTR_COUNT}));
		const auto element = writer.Die(element_abbrev, {SL_KIND_UINT, 3}, std::nullopt, pad);
		const auto typed   = writer.Die(abbrev, {element, SL_FLAG_TYPED}, std::nullopt, pad);
		writer.Zeros(SL_STREAM_GAP);
		writer.Resource(SL_FIRST_REGISTER, "gA", typed, pad);
		writer.Resource(SL_FIRST_REGISTER + 1, "gB", typed, pad);
		return writer.Finish();
	};
	const Texts expected {"gB r4 T typed 9 3", "gA r3 T typed 9 3"};
	CheckTexts(DecodeTexts(make(1)), expected, "unpadded");
	CheckTexts(DecodeTexts(make(SL_WIDE_ULEB_BYTES)), expected, "padded");
	CHECK(make(1).size() < make(SL_WIDE_ULEB_BYTES).size());
}

// ---- Which positions are resource rows. ----

// A section with three rows gA, gB and gC over one raw type record, with `gap` zero bytes before gB
// that make gA and gB `distance` bytes apart.
Texts RowsWithDistance(uint32_t distance) {
	Sl00Writer writer({"gA", "gB", "gC"});
	const auto record =
	    writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})), {SL_FLAG_RAW});
	writer.Zeros(SL_STREAM_GAP);
	const auto first = writer.Resource(SL_FIRST_REGISTER, "gA", record);
	const auto used  = writer.Here() - first;
	// The next row's name is one byte after the register that follows the gap.
	writer.Zeros(distance - used - 1);
	writer.Resource(SL_SECOND_REGISTER, "gB", record);
	writer.Resource(SL_THIRD_REGISTER, "gC", record);
	return DecodeTexts(writer.Finish());
}

// Rows at most 16 bytes apart are one run. A row further back than that is not a record.
void TestSl00RowSpacing() {
	CheckTexts(RowsWithDistance(SL_ROW_GAP), {"gC r5 T raw", "gB r4 T raw", "gA r3 T raw"},
	           "distance 16");
	CheckTexts(RowsWithDistance(SL_ROW_GAP + 1), {"gC r5 T raw", "gB r4 T raw"}, "distance 17");
}

// A section where the one row has a register byte of `register_index`.
void TestSl00RegisterByte() {
	const auto make = [](uint8_t register_index) {
		return DecodeTexts(BuildSingle(
		    [](Sl00Writer& writer) {
			    return writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})),
			                      {SL_FLAG_RAW});
		    },
		    register_index));
	};
	CheckTexts(make(SL_REGISTER_LIMIT - 1), {"gOne r127 T raw"}, "register 0x7F");
	CheckTexts(make(SL_REGISTER_LIMIT), {}, "register 0x80");
	CheckTexts(make(0xFF), {}, "register 0xFF");
	CheckTexts(make(0), {"gOne r0 T raw"}, "register 0");
}

// A section with one row whose reference is `reference`, and where its stream and record are.
struct OneRow {
	Bytes    bytes;
	uint32_t start;
	uint32_t record;
};

OneRow BuildOneRow(uint64_t reference) {
	Sl00Writer writer({"gOne"});
	const auto record =
	    writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})), {SL_FLAG_RAW});
	writer.Zeros(SL_STREAM_GAP);
	// The reference is two bytes wide, so the section's size does not depend on its value.
	writer.ResourceRaw(SL_FIRST_REGISTER, Uleb(writer.NameOffset("gOne")), Uleb(reference, 2));
	return {writer.Finish(), writer.Start(), record};
}

// The records decoded from a one row section whose reference is chosen from the stream's start, the
// section's size and the record's real place.
Texts RowWithReference(const std::function<uint64_t(uint32_t, uint32_t, uint32_t)>& pick) {
	const auto layout = BuildOneRow(0);
	return DecodeTexts(
	    BuildOneRow(pick(layout.start, static_cast<uint32_t>(layout.bytes.size()), layout.record))
	        .bytes);
}

// A row's reference must lie inside the record stream, which starts at the empty string that ends
// the table and runs to the end of the section.
void TestSl00ReferenceBounds() {
	const Texts included {"gOne r3 T raw"};
	// The reference is the record's real place, so this is the baseline each case differs from.
	CheckTexts(RowWithReference([](uint32_t, uint32_t, uint32_t record) { return record; }),
	           included, "real record");
	// A reference to the very first stream byte is in range, and one byte before it is not.
	CHECK(RowWithReference([](uint32_t start, uint32_t, uint32_t) { return start; }).size() == 1);
	CHECK(RowWithReference([](uint32_t start, uint32_t, uint32_t) { return start - 1; }).empty());
	// The last byte of the section is in range, and one past it is not.
	CHECK(RowWithReference([](uint32_t, uint32_t size, uint32_t) { return size - 1; }).size() == 1);
	CHECK(RowWithReference([](uint32_t, uint32_t size, uint32_t) { return size; }).empty());
	CHECK(RowWithReference([](uint32_t, uint32_t size, uint32_t) { return size + 1000; }).empty());
}

// The last four bytes of the section are never positions, so a row whose name starts there is lost.
// With the margin one byte wider or narrower the answer changes.
void TestSl00EndMargin() {
	const auto make = [](uint32_t trailing) {
		return DecodeTexts(BuildSingle(
		    [](Sl00Writer& writer) {
			    return writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})),
			                      {SL_FLAG_RAW});
		    },
		    SL_FIRST_REGISTER, trailing));
	};
	// A row is the name, the reference and a flags byte, so with one trailing byte the name starts
	// exactly four bytes from the end.
	CheckTexts(make(1), {}, "name four bytes from the end");
	CheckTexts(make(2), {"gOne r3 T raw"}, "name five bytes from the end");
	CheckTexts(make(SL_TRAILING_BYTES), {"gOne r3 T raw"}, "name seven bytes from the end");
}

// A name that is not at the start of a string is not a key.
void TestSl00NameKeys() {
	const auto make = [](uint32_t shift) {
		Sl00Writer writer({"gOne"});
		const auto record =
		    writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})), {SL_FLAG_RAW});
		writer.Zeros(SL_STREAM_GAP);
		writer.ResourceRaw(SL_FIRST_REGISTER, Uleb(writer.NameOffset("gOne") + shift),
		                   Uleb(record));
		return DecodeTexts(writer.Finish());
	};
	CheckTexts(make(0), {"gOne r3 T raw"}, "key");
	CheckTexts(make(1), {}, "key plus one");
}

// The strings end at the first empty or unprintable one. Names behind it are not keys.
void TestSl00StringTableEnd() {
	const auto make = [](const std::string& middle) {
		Sl00Writer writer({"gFirst", middle, "gLast"});
		const auto record =
		    writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})), {SL_FLAG_RAW});
		writer.Zeros(SL_STREAM_GAP);
		writer.Resource(SL_FIRST_REGISTER, "gFirst", record);
		writer.Resource(SL_SECOND_REGISTER, "gLast", record);
		return DecodeTexts(writer.Finish());
	};
	CheckTexts(make("gMiddle"), {"gLast r4 T raw", "gFirst r3 T raw"}, "printable");
	// The same string with one control byte ends the table, so gLast is no key.
	CheckTexts(make("gMid\x01le"), {"gFirst r3 T raw"}, "unprintable");
	CheckTexts(make("gMid\x7Fle"), {"gFirst r3 T raw"}, "delete");
	CheckTexts(make("gMid le"), {"gLast r4 T raw", "gFirst r3 T raw"}, "space is printable");
	CheckTexts(make("gMid~le"), {"gLast r4 T raw", "gFirst r3 T raw"}, "tilde is printable");
	// An empty string ends the table, and the writer's own terminator is the next empty string.
	CheckTexts(make(""), {"gFirst r3 T raw"}, "empty");
}

// The table starts at the first `d.d.d.d build `. Anything else is no table and no records.
void TestSl00VersionText() {
	const auto make = [](const std::string& version) {
		Sl00Writer writer({"gOne"}, version);
		const auto record =
		    writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})), {SL_FLAG_RAW});
		writer.Zeros(SL_STREAM_GAP);
		writer.Resource(SL_FIRST_REGISTER, "gOne", record);
		return DecodeTexts(writer.Finish());
	};
	const Texts one {"gOne r3 T raw"};
	CheckTexts(make("1.2.3.4 build 7"), one, "plain");
	CheckTexts(make("10.20.300.4000 build x"), one, "long groups");
	// The leftmost match counts, so digits in front of it do not matter.
	CheckTexts(make("99.1.2.3.4 build 7"), one, "digits before");
	CheckTexts(make("1.2.3 build 7"), {}, "three groups");
	CheckTexts(make("1.2.3.4build 7"), {}, "no space");
	CheckTexts(make("1.2.3.4 Build 7"), {}, "capital");
	CheckTexts(make("1.2.3.4 build"), {}, "no trailing space");
	CheckTexts(make("1..3.4 build 7"), {}, "empty group");
	CheckTexts(make("a.b.c.d build 7"), {}, "letters");
	// A section that lacks the text is empty and not a failure.
	const auto none = DecodeSl00Resources(Bytes {1, 2, 3});
	CHECK(none && none->empty());
}

// ---- Type records that cannot be read. ----

// A section with one row, and a type record that `tail` writes after the row and whose offset it
// returns. The row is written before the record's place is known, so the section is laid out twice.
Bytes BuildWithTail(const std::function<uint32_t(Sl00Writer&)>& tail) {
	uint32_t   record_at = 0;
	const auto build     = [&](uint64_t reference) {
        Sl00Writer writer({"gOne"});
        writer.Zeros(SL_STREAM_GAP);
        writer.ResourceRaw(SL_FIRST_REGISTER, Uleb(writer.NameOffset("gOne")), Uleb(reference, 2));
        record_at = tail(writer);
        return writer.Finish(0);
	};
	(void)build(0);
	return build(record_at);
}

// The one record decoded from a section built by BuildWithTail. `has_abbreviation` tells a record
// whose abbreviation offset was read from one that was lost before that.
struct Lost {
	std::string text;
	bool        has_abbreviation = false;
};

Lost LostRecord(const std::function<uint32_t(Sl00Writer&)>& tail) {
	const auto records = DecodeSl00Resources(BuildWithTail(tail));
	if (!records || records->size() != 1) return {"<not one record>", false};
	return {ResourceText((*records)[0]), (*records)[0].abbreviation_offset.has_value()};
}

constexpr uint32_t SL_CODE_BYTES = 2;

// A record whose abbreviation lies outside the section has a code and no shape, and one whose
// values run off the end has neither. Python gives the first None and the second an IndexError.
void TestSl00UnreadableTypeRecords() {
	// Far outside the section.
	const auto far = LostRecord([](Sl00Writer& writer) {
		const auto record = writer.Here();
		writer.Raw(Uleb(SL_MAX_METADATA, SL_CODE_BYTES + 1));
		return record;
	});
	CHECK(far.text == "gOne r3 - unreadable");
	CHECK(far.has_abbreviation);

	// A code equal to the section's size is the smallest one outside it. The record is the last
	// bytes, so the size is where the writer will be after the code.
	const auto at_end = LostRecord([](Sl00Writer& writer) {
		const auto record = writer.Here();
		writer.Raw(Uleb(writer.Here() + SL_CODE_BYTES, SL_CODE_BYTES));
		return record;
	});
	CHECK(at_end.text == "gOne r3 - unreadable");
	CHECK(at_end.has_abbreviation);
	// One lower is the code's own last byte, which is inside, so it is read as an abbreviation of
	// tag 0 with no attributes. That is a record with no flag, which is not unreadable.
	const auto inside = LostRecord([](Sl00Writer& writer) {
		const auto record = writer.Here();
		writer.Raw(Uleb(writer.Here() + SL_CODE_BYTES - 1, SL_CODE_BYTES));
		return record;
	});
	CHECK(inside.text == "gOne r3 T other - - -");
	CHECK(inside.has_abbreviation);

	// The record's one value is a continuation byte at the end of the section, so it runs off.
	const auto truncated = LostRecord([](Sl00Writer& writer) {
		const auto abbreviation = writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND}));
		const auto record       = writer.Here();
		writer.Raw(Uleb(abbreviation));
		writer.Raw({0x80, 0x80});
		return record;
	});
	CHECK(truncated.text == "gOne r3 - unreadable");
	CHECK(!truncated.has_abbreviation);
}

// An abbreviation at the very end with no 0, 0 pair is read up to the end, as python does when
// fewer than two bytes remain. The baseline has the pair, and both read the same.
void TestSl00AbbreviationAtEnd() {
	const auto make = [](bool terminate) {
		return DecodeTexts(BuildWithTail([&](Sl00Writer& writer) {
			// The record is a code and one value, for attribute 30, and its abbreviation follows.
			const auto record       = writer.Here();
			const auto abbreviation = writer.Here() + SL_CODE_BYTES + 1;
			writer.Raw(Uleb(abbreviation, SL_CODE_BYTES));
			writer.Raw(Uleb(0));
			writer.Abbrev(SL_TAG_CONSTANT, Plain({SL_ATTR_HEAD}), terminate);
			return record;
		}));
	};
	CheckTexts(make(true), {"gOne r3 B cb"}, "terminated");
	CheckTexts(make(false), {"gOne r3 B cb"}, "ends at the end");
}

// ---- Hostile sections. ----

// The sizes that bound the work and the memory.
void TestSl00Caps() {
	const Bytes empty;
	const auto  none = DecodeSl00Resources(empty);
	CHECK(none && none->empty());

	Bytes at_limit(SL_MAX_METADATA, 0);
	CHECK(DecodeSl00Resources(at_limit).has_value());
	at_limit.push_back(0);
	CHECK(!DecodeSl00Resources(at_limit).has_value());

	// Exactly the cap of rows is decoded, and one more than the cap is refused.
	const auto rows = [](uint32_t count) {
		Sl00Writer writer({"gOne"});
		const auto record =
		    writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})), {SL_FLAG_RAW});
		writer.Zeros(SL_STREAM_GAP);
		for (uint32_t i = 0; i < count; i++)
			writer.Resource(SL_FIRST_REGISTER, "gOne", record);
		return DecodeSl00Resources(writer.Finish());
	};
	const auto full = rows(SL_MAX_RESOURCES);
	CHECK(full && full->size() == SL_MAX_RESOURCES);
	CHECK(!rows(SL_MAX_RESOURCES + 1).has_value());
}

// A section that is nothing but continuation bytes behind a valid table is decoded within its work
// budget, because a run is measured once and not followed from every position. A reader that
// visited each byte of a run would spend the budget and the result would have no value. The 0xFF
// run has payload in every byte, so it also has a value that overflows at every position.
void TestSl00ContinuationRun() {
	for (const uint8_t byte: {0x80, 0xFF}) {
		Sl00Writer writer({"gOne"});
		writer.Raw(Bytes(SL_HUGE_RUN_BYTES, byte));
		Sl00DecodeStats stats;
		const auto      records = DecodeSl00Resources(writer.Finish(), &stats);
		CHECK(records && records->empty());
		CHECK(stats.work_used > 0 && stats.work_used < stats.work_budget);
	}
}

// A struct whose members have `pairs` form 80 attributes each costs no data bytes per attribute,
// but every record still steps over all of them. With `members` members chained by their links the
// work is pairs times members, which the budget must refuse.
Bytes BuildFormFlagProbe(uint32_t pairs, uint32_t members) {
	return BuildSingle([&](Sl00Writer& writer) {
		const AttributeForms flags(pairs, {SL_ATTR_SIZE, SL_FORM_FLAG});
		const auto           a_member = writer.Abbrev(SL_TAG_LINKED, flags);
		const auto           a_struct = writer.Abbrev(SL_TAG_STRUCT, Plain({SL_ATTR_HEAD}));
		const auto           first    = writer.Here();
		for (uint32_t index = 0; index < members; index++) {
			// Each member is six bytes and links to the next, so the last links to the struct.
			writer.Die(a_member, {}, writer.Here() + SL_PROBE_MEMBER_BYTES, SL_PROBE_PAD);
		}
		const auto record = writer.Die(a_struct, {first}, std::nullopt, SL_PROBE_PAD);
		return TypedRecord(writer, record, SL_FLAG_STRUCTURED);
	});
}

// The work budget stops a section whose cost is far above its size, and lets one just under it
// through. The cases are counted work and not time, so they do not depend on the machine.
void TestSl00WorkBudget() {
	// A control well under the budget decodes, and says how much of it was used.
	Sl00DecodeStats small;
	const auto      control =
	    DecodeSl00Resources(BuildFormFlagProbe(SL_PROBE_SMALL, SL_PROBE_SMALL), &small);
	CHECK(control && control->size() == 1);
	CHECK(small.work_used > small.work_budget / 100 && small.work_used < small.work_budget);

	// Close to the limit but under it, so the work counted is nearly all of the budget.
	Sl00DecodeStats close_to_limit;
	const auto      under =
	    DecodeSl00Resources(BuildFormFlagProbe(SL_PROBE_UNDER, SL_PROBE_UNDER), &close_to_limit);
	CHECK(under && under->size() == 1);
	CHECK(close_to_limit.work_used > close_to_limit.work_budget / 2 &&
	      close_to_limit.work_used <= close_to_limit.work_budget);

	// Past the limit by a little, and by a lot. Each gets no value, and says it ran out.
	for (const uint32_t count: {SL_PROBE_OVER, SL_PROBE_HUGE}) {
		Sl00DecodeStats stats;
		CHECK(!DecodeSl00Resources(BuildFormFlagProbe(count, count), &stats).has_value());
		CHECK(stats.work_used <= stats.work_budget);
	}
}

// A value that does not fit in 64 bits is not cut down to its low bits. The name's low 64 bits are
// the real key in each variant, and only the bits above them differ.
void TestSl00WideNames() {
	const auto make = [](uint8_t tenth_byte, bool eleven_bytes) {
		Sl00Writer writer({"gOne"});
		const auto record =
		    writer.Die(writer.Abbrev(SL_TAG_RESOURCE, Plain({SL_ATTR_KIND})), {SL_FLAG_RAW});
		writer.Zeros(SL_STREAM_GAP);
		// The key then eight more continuation bytes. The tenth byte holds bits 63 to 69, so 0x02
		// is bit 64. An eleventh byte holds bits 70 and up.
		Bytes name(SL_LONG_ULEB_BYTES - 1, 0x80);
		name[0] = static_cast<uint8_t>(writer.NameOffset("gOne") | 0x80u);
		if (eleven_bytes) {
			name.push_back(0x80);
			name.push_back(0x01);
		} else {
			name.push_back(tenth_byte);
		}
		writer.ResourceRaw(SL_FIRST_REGISTER, name, Uleb(record));
		return DecodeTexts(writer.Finish());
	};
	CheckTexts(make(0x00, false), {"gOne r3 T raw"}, "ten bytes, value is the key");
	CheckTexts(make(0x02, false), {}, "bit 64 set in the tenth byte");
	CheckTexts(make(0x00, true), {}, "bit 70 set in an eleventh byte");
}

// The names in the section.
const std::vector<std::string> KIND_NAMES {"gConstants",  "gRaw",          "gTyped", "gTypedRw",
                                           "gStructured", "gStructStruct", "gImage", "gOther"};

// Crash and hang test, not a memory-safety proof and not a mutation killer. The ASan runtime here
// is 32-bit only, so nothing proves that no read goes out of range. Each prefix is copied to a
// vector of exactly its size, every result must have a value, and every record it returns must name
// a real resource.
void TestSl00TruncatedAndFlipped() {
	const auto  kinds = BuildKinds();
	const auto& bytes = kinds.bytes;
	CHECK(DecodeSl00Resources(bytes)->size() == KIND_NAMES.size());
	size_t short_of_all = 0;
	for (size_t length = 0; length < bytes.size(); length++) {
		const Bytes prefix(bytes.begin(), bytes.begin() + length);
		const auto  records = DecodeSl00Resources(prefix);
		CHECK(records.has_value());
		if (!records) continue;
		CHECK(records->size() <= KIND_NAMES.size());
		for (const auto& record: *records) {
			const bool is_filler = record.name == std::string(SL_FILLER_NAME_BYTES, 'F');
			CHECK(is_filler ||
			      std::find(KIND_NAMES.begin(), KIND_NAMES.end(), record.name) != KIND_NAMES.end());
		}
		if (records->size() < KIND_NAMES.size()) short_of_all++;
	}
	// Most prefixes lose some records, so the loop did cut into them.
	CHECK(short_of_all > bytes.size() / 2);
	CHECK(DecodeSl00Resources(Bytes(bytes.begin(), bytes.begin() + 12))->empty());

	// Every byte replaced by values that make long runs, big numbers and short links.
	for (size_t at = 0; at < bytes.size(); at++) {
		for (const uint8_t value: {0x00, 0x01, 0x7F, 0x80, 0xFF}) {
			Bytes flipped      = bytes;
			flipped[at]        = value;
			const auto records = DecodeSl00Resources(flipped);
			CHECK(records.has_value());
			if (records) CHECK(records->size() <= SL_MAX_RESOURCES);
		}
	}
}

} // namespace

int main() {
	TestComputeProgramHash();
	TestGraphicsElfSkipped();
	TestHostileElfs();
	TestDeclaredSizeAndFooter();
	TestDeduplicateFirstWins();
	TestDecoderMatchesOldCode();
	TestStaticKeyGolden();
	TestDeviceAndDispatchInputs();
	TestHeaderFailuresAndUnknownRegisters();
	TestSl00ResourceKinds();
	TestSl00WideTag();
	TestSl00ReadWriteAttribute();
	TestSl00ConstantBufferMark();
	TestSl00Flags();
	TestSl00StructChain();
	TestSl00StrideRules();
	TestSl00FormFlag();
	TestSl00RepeatedAttribute();
	TestSl00PaddedValues();
	TestSl00RowSpacing();
	TestSl00RegisterByte();
	TestSl00ReferenceBounds();
	TestSl00EndMargin();
	TestSl00NameKeys();
	TestSl00StringTableEnd();
	TestSl00VersionText();
	TestSl00UnreadableTypeRecords();
	TestSl00AbbreviationAtEnd();
	TestSl00Caps();
	TestSl00ContinuationRun();
	TestSl00WorkBudget();
	TestSl00WideNames();
	TestSl00TruncatedAndFlipped();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("shader store scan tests passed\n");
	return 0;
}
