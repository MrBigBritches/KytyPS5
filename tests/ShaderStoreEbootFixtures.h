#ifndef TESTS_SHADER_STORE_EBOOT_FIXTURES_H
#define TESTS_SHADER_STORE_EBOOT_FIXTURES_H

// The synthetic eboot of the shader store tests: a few tiny compute programs, the binary, header,
// metadata and ELF layout around them, and the constants of the host they are built for. Tests that
// need a game's eboot without a game file include this.
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/moduleId.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/computePredictor.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/ebootShaderScan.h"
#include "graphics/shader/storegen/headerSlots.h"
#include "graphics/shader/storegen/shaderHeaderLayout.h"
#include "graphics/shader/storegen/sl00Metadata.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace StoreGenFixtures {

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;
namespace IR     = Libs::Graphics::ShaderRecompiler::IR;
namespace Layout = Libs::Graphics::StoreGen::HeaderLayout;
namespace fs     = std::filesystem;

using Bytes = std::vector<uint8_t>;

// ---- Values of the host and the programs. ----
constexpr uint32_t HOST_SUBGROUP_SIZE = 32;
constexpr uint32_t MAX_SHARED_BYTES   = 32768;
constexpr uint64_t BUILD_STAMP        = 0x1122334455667788ull;
constexpr uint64_t OTHER_BUILD_STAMP  = 0x1122334455667789ull;
constexpr uint32_t PARENT_PID         = 4242;

// The nullable buffers of Motor start at this register, as the predictor's rule has it.
constexpr uint32_t NULLABLE_FIRST  = 29;
constexpr uint32_t NULLABLE_SECOND = 30;
constexpr uint32_t PLAIN_REGISTER  = 1;
constexpr uint32_t OUTPUT_REGISTER = 0;
constexpr uint32_t UNUSED_REGISTER = 2;
// User SGPRs of the descriptors. Each descriptor is four SGPRs.
constexpr uint32_t SGPR_FIRST           = 0;
constexpr uint32_t SGPR_SECOND          = 4;
constexpr uint32_t SGPR_OUTPUT          = 8;
constexpr uint32_t USER_SGPR_COUNT      = 12;
constexpr uint32_t SGPRS_PER_DESCRIPTOR = 4;

// ---- Shader encodings, spelled out here with plain shifts. ----
constexpr uint32_t MUBUF_PREFIX       = 0x38;
constexpr uint32_t MUBUF_LOAD_DWORD   = 0x0c;
constexpr uint32_t MUBUF_STORE_DWORD  = 0x1c;
constexpr uint32_t MUBUF_OFFEN_BIT    = 12;
constexpr uint32_t MUBUF_INLINE_ZERO  = 128;
constexpr uint32_t VOP2_ADD_NC_U32    = 0x25;
constexpr uint32_t VGPR_BASE          = 256;
constexpr uint32_t SOP2_ADD_U32       = 0x00;
constexpr uint32_t VOP1_PREFIX        = 0x3f;
constexpr uint32_t VOP1_MOV_B32       = 0x01;
constexpr uint32_t DS_PREFIX          = 0x36;
constexpr uint32_t DS_WRITE_B32       = 0x0d;
constexpr uint32_t LDS_GRANULE_SHIFT  = 15;
constexpr uint32_t SOP1_SWAPPC        = 0x21;
constexpr uint32_t INLINE_CONSTANT_0  = 128;
constexpr uint32_t S_WAITCNT_ALL      = 0xbf8c3f70u;
constexpr uint32_t S_NOP              = 0xbf800000u;
constexpr uint32_t S_ENDPGM           = 0xbf810000u;
constexpr uint32_t S_CODE_END         = 0xbf9f0000u;
constexpr uint32_t WORDS_PER_16_BYTES = 4;
// A scalar ALU word starts with these bits. SOP1 and SOP2 differ in the next field.
constexpr uint32_t SOP_PREFIX       = 0x80000000u;
constexpr uint32_t SOP1_FIELD       = 0x7du;
constexpr uint32_t SOP1_FIELD_SHIFT = 23;
constexpr uint32_t SOP_DST_SHIFT    = 16;
constexpr uint32_t SOP_SRC1_SHIFT   = 8;
constexpr uint32_t SOP1_OP_SHIFT    = 8;
constexpr uint32_t SWAPPC_SDST      = 0;
constexpr uint32_t SWAPPC_SSRC      = 0;
constexpr uint32_t ADD_CONSTANT     = 16;
// The eboot's layout. ELFs start at this alignment plus this remainder, behind a few junk bytes.
constexpr uint64_t ELF_ALIGNMENT       = 16;
constexpr uint64_t ELF_ALIGN_REMAINDER = 5;
constexpr uint8_t  ELF_PADDING_BYTE    = 0xEE;
constexpr uint8_t  EBOOT_TAIL_BYTE     = 0xCC;
constexpr uint8_t  ELF_MAGIC_FIRST     = 0x7f;
constexpr uint8_t  EBOOT_LEAD_FILL     = 9;
// The start of an sl00 section, which the decoder expects before the version text.
constexpr uint8_t SL_PREFIX_BYTES[] = {3, 5, 6, 0, 0};
// Bytes of a record left behind as a crash would leave a torn write.
constexpr size_t TORN_BYTES = 5;
// More than the file holds.
constexpr size_t PAST_THE_FILE = 1000;
// The rank after the last candidate of the main programs.
constexpr uint64_t MAIN_CANDIDATES = 7;

// ---- The binary, header and ELF layout, as ShaderStoreScanTests builds them. ----
constexpr uint32_t FOOTER_SIZE           = 0x30;
constexpr uint32_t FOOTER_VERSION        = 1;
constexpr uint32_t FOOTER_VERSION_OFFSET = 0x10;
constexpr uint32_t FOOTER_CODE_OFFSET    = 0x14;
constexpr uint32_t FOOTER_LITERAL_OFFSET = 0x18;
constexpr uint32_t HEADER_SIZE           = 0x400;
constexpr uint64_t HEADER_SH_OFFSET      = 0x60;
constexpr uint64_t USER_AT               = 0x100;
constexpr uint64_t SHARP_AT              = 0x160;
// Room for the 64 slots of a class, so that the slot of register 29 or 30 does not run into the
// next table.
constexpr uint64_t SHARP_STRIDE      = 0x80;
constexpr uint16_t SLOT_UNUSED_ENTRY = 0xFFFF;
constexpr uint32_t SCRATCH_DWORDS    = 7;
constexpr uint32_t THREADS_X         = 64;
constexpr uint32_t FLOAT_MODE        = 0xC0;
constexpr uint32_t RSRC1_VGPRS       = 5;
constexpr uint32_t TIDIG_COMP_CNT    = 0;
constexpr uint32_t TAIL_JUNK_BYTES   = 7;

constexpr uint64_t ELF_HEADER_BYTES    = 64;
constexpr uint64_t SECTION_ENTRY_BYTES = 64;
constexpr uint16_t SECTION_COUNT       = 5;
constexpr uint64_t ENTRY_FILE_OFFSET   = 0x18;
constexpr uint64_t ENTRY_SIZE_FIELD    = 0x20;
constexpr uint16_t NAMES_INDEX         = 1;
constexpr uint16_t HEADER_INDEX        = 2;
constexpr uint16_t TEXT_INDEX          = 3;
constexpr uint16_t METADATA_INDEX      = 4;
constexpr uint64_t ELF_CLASS_FIELD     = 0x04;
constexpr uint64_t ELF_DATA_FIELD      = 0x05;
constexpr uint64_t ELF_SHOFF_FIELD     = 0x28;
constexpr uint64_t ELF_SHENT_FIELD     = 0x3A;
constexpr uint64_t ELF_SHNUM_FIELD     = 0x3C;
constexpr uint64_t ELF_SHSTR_FIELD     = 0x3E;

// ---- sl00 attribute numbers, spelled out so the tests do not depend on the decoder's constants.
// ----
constexpr uint32_t SL_FORM_ULEB       = 1;
constexpr uint32_t SL_ATTR_KIND       = 41;
constexpr uint32_t SL_ATTR_READ_WRITE = 40;
constexpr uint32_t SL_TAG_RESOURCE    = 4;
constexpr uint32_t SL_FLAG_RAW        = 2;
// A type flag that no shape rule covers, so a buffer with it is outside the predictor's rules.
constexpr uint32_t SL_FLAG_NO_RULE   = 5;
constexpr uint32_t SL_STREAM_GAP     = 20;
constexpr uint32_t SL_TRAILING_BYTES = 4;

// ---- Little-endian writers. ----

// Writes a value at a byte offset, growing the buffer with zeros when it is too short.
template <typename T>
void Put(Bytes& bytes, uint64_t at, T value) {
	if (bytes.size() < at + sizeof(T)) {
		bytes.resize(at + sizeof(T));
	}
	std::memcpy(bytes.data() + at, &value, sizeof(T));
}

// Writes the bytes of one buffer at the end of another.
void AppendBytes(Bytes& bytes, const Bytes& extra) {
	bytes.insert(bytes.end(), extra.begin(), extra.end());
}

// ---- Programs. ----

// A buffer a program uses, and where its descriptor and its sl00 record are.
struct BufferUse {
	Sl00Class   sharp_class = Sl00Class::ReadOnly;
	uint32_t    index       = 0; // The slot's number, which the sl00 record names too.
	uint32_t    sgpr        = 0; // The first user SGPR of the descriptor.
	std::string name;
	// The record's type is one no predictor rule covers.
	bool no_rule = false;
};

// What a test program does. It loads dwords from each buffer, adds them and stores the sum.
struct ProgramSpec {
	std::vector<BufferUse> loads;
	bool                   store = true;
	// Extra no-ops, which make the program bigger so the programs have different sizes.
	uint32_t padding_nops = 0;
	// Calls through a scalar target, which the store does not hold.
	bool calls = false;
	// Moves the first load's descriptor through an add, which the classifier cannot name.
	bool derived_descriptor = false;
	// Leaves out the compute resource registers, so no static key can be made.
	bool without_registers = false;
	// Records in the metadata that the code never uses. The decoder wants more than one record.
	std::vector<BufferUse> unused_records;
	// The name of the output's sl00 record, which the predictor reads to find buffers that may be
	// unbound.
	std::string output_name = "gOut";
	// The first SGPR of the output descriptor, and how many user SGPRs the program has.
	uint32_t output_sgpr = SGPR_OUTPUT;
	uint32_t user_sgprs  = USER_SGPR_COUNT;
	// Reads user SGPR 0 into a value that goes to LDS, and uses no buffer.
	bool user_data_only = false;
	// Words put in the code just before its end, as they are. A test uses them to put in an
	// encoding that the recompiler does not take.
	std::vector<uint32_t> extra_words;
};

BufferUse OutputBuffer(const ProgramSpec& spec) {
	return {Sl00Class::ReadWrite, OUTPUT_REGISTER, spec.output_sgpr, spec.output_name, false};
}

// The two nullable buffers of Motor, the plain one, and the output.
BufferUse FirstGlobal() {
	return {Sl00Class::ReadOnly, NULLABLE_FIRST, SGPR_FIRST, "gGlobalFirst", false};
}
BufferUse SecondGlobal() {
	return {Sl00Class::ReadOnly, NULLABLE_SECOND, SGPR_SECOND, "gGlobalSecond", false};
}
BufferUse PlainBuffer() {
	return {Sl00Class::ReadOnly, PLAIN_REGISTER, SGPR_FIRST, "gPlain", false};
}

// The word that loads or stores a dword through the descriptor that starts at `sgpr`.
void AppendMubuf(std::vector<uint32_t>& code, uint32_t opcode, uint32_t vdata, uint32_t sgpr) {
	code.push_back((MUBUF_PREFIX << 26) | (opcode << 18) | (1u << MUBUF_OFFEN_BIT));
	code.push_back((MUBUF_INLINE_ZERO << 24) | ((sgpr / SGPRS_PER_DESCRIPTOR) << 16) |
	               (vdata << 8));
}

// The code of a program. Every load goes to its own VGPR, the loads are added into v1 and stored.
std::vector<uint32_t> BuildCode(const ProgramSpec& spec) {
	std::vector<uint32_t> code;
	if (spec.user_data_only) {
		code.push_back((VOP1_PREFIX << 25) | (1u << 17) | (VOP1_MOV_B32 << 9) | SGPR_FIRST);
		code.push_back((DS_PREFIX << 26) | (DS_WRITE_B32 << 18));
		code.push_back(1u << 8);
		if (spec.store) {
			AppendMubuf(code, MUBUF_STORE_DWORD, 1, spec.output_sgpr);
		}
		code.push_back(S_ENDPGM);
		return code;
	}
	if (spec.calls) {
		code.push_back(SOP_PREFIX | (SOP1_FIELD << SOP1_FIELD_SHIFT) |
		               (SWAPPC_SDST << SOP_DST_SHIFT) | (SOP1_SWAPPC << SOP1_OP_SHIFT) |
		               SWAPPC_SSRC);
	}
	if (spec.derived_descriptor) {
		// s4 = s0 + 16 makes the descriptor at s[4:7] depend on arithmetic, not on user data.
		code.push_back(SOP_PREFIX | (SOP2_ADD_U32 << SOP1_FIELD_SHIFT) |
		               (SGPR_SECOND << SOP_DST_SHIFT) |
		               ((INLINE_CONSTANT_0 + ADD_CONSTANT) << SOP_SRC1_SHIFT) | SGPR_FIRST);
	}
	uint32_t vgpr = 1;
	for (const auto& load: spec.loads) {
		const uint32_t sgpr = spec.derived_descriptor ? SGPR_SECOND : load.sgpr;
		AppendMubuf(code, MUBUF_LOAD_DWORD, vgpr++, sgpr);
	}
	code.push_back(S_WAITCNT_ALL);
	for (uint32_t extra = 2; extra < vgpr; extra++) {
		code.push_back((VOP2_ADD_NC_U32 << 25) | (1u << 17) | (extra << 9) | (VGPR_BASE + 1));
	}
	if (spec.store) {
		AppendMubuf(code, MUBUF_STORE_DWORD, 1, spec.output_sgpr);
	}
	code.insert(code.end(), spec.padding_nops, S_NOP);
	code.insert(code.end(), spec.extra_words.begin(), spec.extra_words.end());
	code.push_back(S_ENDPGM);
	return code;
}

// A shader binary, which is the code padded to 16 bytes and the footer the scan reads.
Bytes BuildBinary(const ProgramSpec& spec) {
	auto       code       = BuildCode(spec);
	const auto code_bytes = static_cast<uint32_t>(code.size() * sizeof(uint32_t));
	while (code.size() % WORDS_PER_16_BYTES != 0) {
		code.push_back(S_CODE_END);
	}
	Bytes binary(code.size() * sizeof(uint32_t));
	std::memcpy(binary.data(), code.data(), binary.size());
	Bytes footer(FOOTER_SIZE, 0);
	std::memcpy(footer.data(), "barefoot", 8);
	Put<uint32_t>(footer, FOOTER_VERSION_OFFSET, FOOTER_VERSION);
	Put<uint32_t>(footer, FOOTER_CODE_OFFSET, code_bytes);
	Put<uint32_t>(footer, FOOTER_LITERAL_OFFSET, 0);
	AppendBytes(binary, footer);
	return binary;
}

// Builds the sharp slot tables of the user data table, T U S B, from the buffers' slots.
std::array<std::vector<uint16_t>, 4> SlotTables(const ProgramSpec& spec) {
	std::array<std::vector<uint16_t>, 4> tables;
	auto                                 use = [&](const BufferUse& buffer) {
        auto& table = tables[static_cast<size_t>(buffer.sharp_class)];
        if (table.size() <= buffer.index) {
            table.resize(buffer.index + 1, SLOT_UNUSED_ENTRY);
        }
        table[buffer.index] = static_cast<uint16_t>(buffer.sgpr);
	};
	for (const auto& load: spec.loads) {
		use(load);
	}
	if (spec.store) {
		use(OutputBuffer(spec));
	}
	return tables;
}

// The .shader_header with the SH registers and the user data tables.
Bytes BuildHeader(const ProgramSpec& spec, uint32_t shader_size) {
	const uint32_t rsrc1 = RSRC1_VGPRS | (FLOAT_MODE << 12u);
	const uint32_t rsrc2 = 1u | (spec.user_sgprs << 1u) | (1u << 7u) | (1u << 9u) | (1u << 10u) |
	                       (TIDIG_COMP_CNT << 11u) |
	                       ((spec.user_data_only ? 1u : 0u) << LDS_GRANULE_SHIFT);
	std::vector<std::pair<uint32_t, uint32_t>> registers {{Pm4::COMPUTE_PGM_LO, 0},
	                                                      {Pm4::COMPUTE_PGM_HI, 0},
	                                                      {Pm4::COMPUTE_NUM_THREAD_X, THREADS_X},
	                                                      {Pm4::COMPUTE_NUM_THREAD_Y, 1},
	                                                      {Pm4::COMPUTE_NUM_THREAD_Z, 1}};
	if (!spec.without_registers) {
		registers.emplace_back(Pm4::COMPUTE_PGM_RSRC1, rsrc1);
		registers.emplace_back(Pm4::COMPUTE_PGM_RSRC2, rsrc2);
	}
	Bytes header(HEADER_SIZE, 0);
	Put<uint32_t>(header, Layout::OFFSET_MAGIC, Layout::FILE_MAGIC);
	Put<uint32_t>(header, Layout::OFFSET_VERSION, Layout::FILE_VERSION);
	Put<uint64_t>(header, Layout::OFFSET_SH_REGISTERS,
	              HEADER_SH_OFFSET - Layout::OFFSET_SH_REGISTERS);
	Put<uint32_t>(header, Layout::OFFSET_SHADER_SIZE, shader_size);
	Put<uint16_t>(header, Layout::OFFSET_SCRATCH_DWORDS, static_cast<uint16_t>(SCRATCH_DWORDS));
	Put<uint8_t>(header, Layout::OFFSET_TYPE, Layout::TYPE_COMPUTE);
	Put<uint8_t>(header, Layout::OFFSET_NUM_SH_REGISTERS, static_cast<uint8_t>(registers.size()));
	for (size_t index = 0; index < registers.size(); index++) {
		Put<uint32_t>(header, HEADER_SH_OFFSET + index * Layout::SH_REGISTER_SIZE,
		              registers[index].first);
		Put<uint32_t>(header,
		              HEADER_SH_OFFSET + index * Layout::SH_REGISTER_SIZE + sizeof(uint32_t),
		              registers[index].second);
	}
	// The user data table, with no direct entries and one slot table per class.
	Put<uint64_t>(header, Layout::OFFSET_USER_DATA, USER_AT - Layout::OFFSET_USER_DATA);
	Put<uint64_t>(header, USER_AT + Layout::USER_DIRECT_POINTER, SHARP_AT - USER_AT);
	const auto tables = SlotTables(spec);
	for (uint64_t table = 0; table < tables.size(); table++) {
		const uint64_t field =
		    USER_AT + Layout::USER_SHARP_POINTERS + Layout::USER_POINTER_SIZE * table;
		const uint64_t at = SHARP_AT + SHARP_STRIDE * table;
		Put<uint64_t>(header, field, at - field);
		Put<uint16_t>(header, USER_AT + Layout::USER_SHARP_COUNTS + sizeof(uint16_t) * table,
		              static_cast<uint16_t>(tables[table].size()));
		for (size_t slot = 0; slot < tables[table].size(); slot++) {
			Put<uint16_t>(header, at + sizeof(uint16_t) * slot, tables[table][slot]);
		}
	}
	return header;
}

// ---- sl00 metadata. This is a reduced form of the writer in ShaderStoreScanTests, which writes
// only what raw buffers need. ----

// A variable length integer.
Bytes Uleb(uint64_t value) {
	Bytes bytes;
	do {
		auto byte = static_cast<uint8_t>(value & 0x7Fu);
		value >>= 7u;
		if (value != 0) byte |= 0x80u;
		bytes.push_back(byte);
	} while (value != 0);
	return bytes;
}

// Writes an sl00 section for raw buffers. The names come first, then two abbreviations, then the
// type records, and then one resource row per buffer.
class Sl00Builder {
public:
	explicit Sl00Builder(const std::vector<std::string>& names) {
		// A prefix the decoder expects, then the version text.
		m_bytes.assign(std::begin(SL_PREFIX_BYTES), std::end(SL_PREFIX_BYTES));
		AddText("1.2.3.4 build 7");
		for (const auto& name: names) {
			m_offsets[name] = static_cast<uint32_t>(m_bytes.size());
			AddText(name);
		}
		m_bytes.push_back(0);
	}

	// An abbreviation of a resource type with the given attributes.
	uint32_t Abbrev(const std::vector<uint32_t>& attributes) {
		const auto at = static_cast<uint32_t>(m_bytes.size());
		AppendBytes(m_bytes, Uleb(SL_TAG_RESOURCE));
		for (const auto attribute: attributes) {
			AppendBytes(m_bytes, Uleb(attribute));
			AppendBytes(m_bytes, Uleb(SL_FORM_ULEB));
		}
		m_bytes.push_back(0);
		m_bytes.push_back(0);
		return at;
	}

	// A type record, which is its abbreviation and the values.
	uint32_t Type(uint32_t abbreviation, const std::vector<uint64_t>& values) {
		const auto at = static_cast<uint32_t>(m_bytes.size());
		AppendBytes(m_bytes, Uleb(abbreviation));
		for (const auto value: values) {
			AppendBytes(m_bytes, Uleb(value));
		}
		return at;
	}

	// Ends the type records.
	void StartRows() { m_bytes.insert(m_bytes.end(), SL_STREAM_GAP, 0); }

	// A resource row, which is the register, the name, the type and a flags byte.
	void Row(uint8_t register_index, const std::string& name, uint32_t type_at) {
		m_bytes.push_back(register_index);
		AppendBytes(m_bytes, Uleb(m_offsets.at(name)));
		AppendBytes(m_bytes, Uleb(type_at));
		m_bytes.push_back(1);
	}

	// The section's bytes, with the zero bytes the decoder wants after the last row.
	Bytes Finish() const {
		Bytes bytes = m_bytes;
		bytes.insert(bytes.end(), SL_TRAILING_BYTES, 0);
		return bytes;
	}

private:
	void AddText(const std::string& text) {
		m_bytes.insert(m_bytes.end(), text.begin(), text.end());
		m_bytes.push_back(0);
	}

	Bytes                                     m_bytes;
	std::unordered_map<std::string, uint32_t> m_offsets;
};

// The metadata of a program, with a raw record for every buffer it uses.
Bytes BuildMetadata(const ProgramSpec& spec) {
	std::vector<BufferUse> uses = spec.loads;
	uses.insert(uses.end(), spec.unused_records.begin(), spec.unused_records.end());
	if (spec.store) {
		uses.push_back(OutputBuffer(spec));
	}
	std::vector<std::string> names;
	for (const auto& use: uses) {
		names.push_back(use.name);
	}
	Sl00Builder writer(names);
	const auto  read_abbreviation  = writer.Abbrev({SL_ATTR_KIND});
	const auto  write_abbreviation = writer.Abbrev({SL_ATTR_KIND, SL_ATTR_READ_WRITE});
	const auto  raw_read           = writer.Type(read_abbreviation, {SL_FLAG_RAW});
	const auto  raw_write          = writer.Type(write_abbreviation, {SL_FLAG_RAW, 0});
	const auto  no_rule            = writer.Type(read_abbreviation, {SL_FLAG_NO_RULE});
	writer.StartRows();
	for (const auto& use: uses) {
		const auto type = use.sharp_class == Sl00Class::ReadWrite ? raw_write
		                  : use.no_rule                           ? no_rule
		                                                          : raw_read;
		writer.Row(static_cast<uint8_t>(use.index), use.name, type);
	}
	return writer.Finish();
}

// ---- The ELF and the eboot, as ShaderStoreScanTests builds them. ----

// Lays out an ELF64 with .shader_header, .shader_text and .metadata. The section table is last.
Bytes BuildElf(const Bytes& header, const Bytes& text, const Bytes& metadata) {
	const std::string names = std::string(1, '\0') + ".shstrtab" + '\0' + ".shader_header" + '\0' +
	                          ".shader_text" + '\0' + ".metadata" + '\0';
	const auto name_at = [&](const char* name) { return static_cast<uint32_t>(names.find(name)); };
	Bytes      file(ELF_HEADER_BYTES, 0);
	file[0]               = 0x7f;
	file[1]               = 'E';
	file[2]               = 'L';
	file[3]               = 'F';
	file[ELF_CLASS_FIELD] = 2;
	file[ELF_DATA_FIELD]  = 1;
	struct Placed {
		uint32_t name;
		uint64_t offset;
		uint64_t size;
	};
	std::array<Placed, SECTION_COUNT> placed {};
	const auto place = [&](uint16_t index, uint32_t name, const Bytes& data) {
		while (file.size() % 16 != 0) {
			file.push_back(0);
		}
		placed[index] = {name, file.size(), data.size()};
		AppendBytes(file, data);
	};
	place(NAMES_INDEX, name_at(".shstrtab"), Bytes(names.begin(), names.end()));
	place(HEADER_INDEX, name_at(".shader_header"), header);
	place(TEXT_INDEX, name_at(".shader_text"), text);
	place(METADATA_INDEX, name_at(".metadata"), metadata);
	while (file.size() % 16 != 0) {
		file.push_back(0);
	}
	const uint64_t table_offset = file.size();
	for (uint16_t index = 0; index < SECTION_COUNT; index++) {
		Bytes entry(SECTION_ENTRY_BYTES, 0);
		Put<uint32_t>(entry, 0, placed[index].name);
		Put<uint64_t>(entry, ENTRY_FILE_OFFSET, placed[index].offset);
		Put<uint64_t>(entry, ENTRY_SIZE_FIELD, placed[index].size);
		AppendBytes(file, entry);
	}
	Put<uint64_t>(file, ELF_SHOFF_FIELD, table_offset);
	Put<uint16_t>(file, ELF_SHENT_FIELD, SECTION_ENTRY_BYTES);
	Put<uint16_t>(file, ELF_SHNUM_FIELD, SECTION_COUNT);
	Put<uint16_t>(file, ELF_SHSTR_FIELD, NAMES_INDEX);
	return file;
}

// An eboot that holds one ELF for each program, in the order given, behind a few junk bytes.
Bytes BuildEboot(const std::vector<ProgramSpec>& programs) {
	Bytes eboot {1, 2, 3, ELF_MAGIC_FIRST, 'E', 'L', 'X', EBOOT_LEAD_FILL, EBOOT_LEAD_FILL};
	for (const auto& spec: programs) {
		const auto binary = BuildBinary(spec);
		const auto elf = BuildElf(BuildHeader(spec, static_cast<uint32_t>(binary.size())), binary,
		                          BuildMetadata(spec));
		while (eboot.size() % ELF_ALIGNMENT != ELF_ALIGN_REMAINDER) {
			eboot.push_back(ELF_PADDING_BYTE);
		}
		AppendBytes(eboot, elf);
	}
	eboot.insert(eboot.end(), TAIL_JUNK_BYTES, EBOOT_TAIL_BYTE);
	return eboot;
}

// ---- The programs the tests use. ----

// Two nullable buffers, so it has four candidates. It is the biggest of the three.
ProgramSpec ProgramBig() {
	return {{FirstGlobal(), SecondGlobal()}, true, 8, false, false, false};
}
// One nullable buffer, so it has two candidates.
ProgramSpec ProgramMiddle() {
	return {{FirstGlobal()}, true, 4, false, false, false};
}
// No nullable buffer, so it has one candidate. It is the smallest.
ProgramSpec ProgramSmall() {
	return {{PlainBuffer()}, true, 0, false, false, false};
}
// The eboot lists the biggest program first, so an eboot-order walk differs from the rule.
std::vector<ProgramSpec> MainPrograms() {
	return {ProgramBig(), ProgramMiddle(), ProgramSmall()};
}

// ---- What the predictor says, from facts written by hand. ----

// The buffers of a program in the order the translator lists them, which is the order the code
// uses them in, with the flags the code gives them.
ResourceFacts FactsOf(const ProgramSpec& spec) {
	ResourceFacts facts;
	for (const auto& load: spec.loads) {
		facts.buffers.push_back(
		    {{DescriptorKeyKind::User, load.sgpr, 0, 0, 0}, false, false, false});
	}
	if (spec.store) {
		facts.buffers.push_back(
		    {{DescriptorKeyKind::User, spec.output_sgpr, 0, 0, 0}, false, false, true});
	}
	return facts;
}

// A candidate of a program as the test expects the worker to name it.
struct Expected {
	uint64_t              hash = 0;
	std::vector<uint32_t> nulls;
	uint64_t              id = 0;
};

// The candidates of the program in `scanned`, from the predictor and the facts above.
std::vector<Expected> PredictedFor(const ProgramSpec& spec, const ScannedProgram& scanned) {
	const ComputeKeyDevice device {HOST_SUBGROUP_SIZE, MAX_SHARED_BYTES};
	const auto             key     = BuildComputeStaticKey(scanned.header, device, MOTOR_DISPATCH);
	const auto             slots   = DecodeHeaderSlots(scanned.header);
	const auto             records = DecodeSl00Resources(scanned.metadata);
	const auto             facts   = FactsOf(spec);
	std::vector<Expected>  expected;
	if (!key || !slots || !records) {
		return expected;
	}
	const auto prediction = PredictCandidates({scanned.code_hash, key->user_data_count, key->state,
	                                           facts, *slots, *records, NullPolicy::Subsets});
	for (const auto& candidate: prediction.candidates) {
		expected.push_back({scanned.code_hash, candidate.nulls, candidate.id});
	}
	return expected;
}

// The candidate with these nulls.
const Expected& Pick(const std::vector<Expected>& candidates, const std::vector<uint32_t>& nulls) {
	static const Expected none;
	for (const auto& candidate: candidates) {
		if (candidate.nulls == nulls) return candidate;
	}
	return none;
}

} // namespace

} // namespace StoreGenFixtures

#endif // TESTS_SHADER_STORE_EBOOT_FIXTURES_H
