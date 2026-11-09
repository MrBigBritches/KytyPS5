// Tests the eboot compute shader scan and the static key built from a shader header.
// Everything here uses synthetic blobs, so no game file is needed.
#include "graphics/guest_gpu/csRegisters.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/ebootShaderScan.h"
#include "graphics/shader/storegen/shaderHeaderLayout.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("shader store scan tests passed\n");
	return 0;
}
