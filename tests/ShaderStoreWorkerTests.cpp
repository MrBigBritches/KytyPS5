// Tests the shader store worker, which turns a game's eboot into store records without a graphics
// device. The eboot is built here from a few tiny compute programs, which the worker translates for
// real, so no game file is needed. Each check names what else could satisfy it and compares
// against a baseline that differs only in the behaviour under test.
#include "common/childProcess.h"
#include "common/file.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/shader/moduleId.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/computePredictor.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/ebootShaderScan.h"
#include "graphics/shader/storegen/headerSlots.h"
#include "graphics/shader/storegen/shaderHeaderLayout.h"
#include "graphics/shader/storegen/shaderStoreWorker.h"
#include "graphics/shader/storegen/sl00Metadata.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;
namespace IR     = Libs::Graphics::ShaderRecompiler::IR;
namespace Layout = Libs::Graphics::StoreGen::HeaderLayout;
namespace fs     = std::filesystem;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

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

// ---- Files. ----

// A directory for the files of one test, which is removed with it.
class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		m_path             = fs::temp_directory_path() /
		         ("kyty_store_worker_tests_" + std::to_string(Common::CurrentProcessId()) + "_" +
		          std::to_string(counter++));
		fs::remove_all(m_path);
		fs::create_directories(m_path);
	}
	~TempDirectory() {
		std::error_code ignored;
		fs::remove_all(m_path, ignored);
	}
	TempDirectory(const TempDirectory&)            = delete;
	TempDirectory& operator=(const TempDirectory&) = delete;

	fs::path operator/(const std::string& name) const { return m_path / name; }

private:
	fs::path m_path;
};

// Reads a whole file through Common::File, which opens it so that the worker can still write to it.
// A std::ifstream does not, on Windows, while the worker holds the file.
Bytes ReadFile(const fs::path& path) {
	Common::File file;
	if (!file.Open(path, Common::File::Mode::Read)) {
		return {};
	}
	const auto bytes = file.ReadWholeBuffer();
	return {reinterpret_cast<const uint8_t*>(bytes.data()),
	        reinterpret_cast<const uint8_t*>(bytes.data()) + bytes.size()};
}

// Writes a whole file, replacing it.
void WriteFile(const fs::path& path, const Bytes& bytes) {
	std::ofstream out(path, std::ios::binary | std::ios::trunc);
	out.write(reinterpret_cast<const char*>(bytes.data()),
	          static_cast<std::streamsize>(bytes.size()));
}

// The lines of a text file.
std::vector<std::string> ReadLines(const fs::path& path) {
	const auto               bytes = ReadFile(path);
	std::istringstream       in(std::string(bytes.begin(), bytes.end()));
	std::vector<std::string> lines;
	for (std::string line; std::getline(in, line);) {
		lines.push_back(line);
	}
	return lines;
}

// The words of a line.
std::vector<std::string> Words(const std::string& line) {
	std::istringstream       stream(line);
	std::vector<std::string> words;
	for (std::string word; stream >> word;) {
		words.push_back(word);
	}
	return words;
}

// A hash as the worker's log writes it, which is 16 hex digits.
std::string HashText(uint64_t value) {
	char text[20];
	std::snprintf(text, sizeof text, "%016llx", static_cast<unsigned long long>(value));
	return text;
}

// ---- Running a worker. ----

// The StoreHeader bytes, written as PipelineCache writes them.
Bytes MakeStoreHeader(uint64_t stamp = BUILD_STAMP) {
	IR::Writer                       writer;
	auto                             jump_tables = false;
	ShaderRecompiler::Spirv::Profile profile;
	uint32_t                         shared = MAX_SHARED_BYTES;
	IR::IoAll(writer, stamp, jump_tables, profile, shared);
	return std::move(writer.bytes);
}

// A request that writes into the given directory. The eboot goes to a file there.
WorkerRequest MakeRequest(const TempDirectory& directory, const Bytes& eboot,
                          const std::string& tag = "run") {
	WorkerRequest request;
	request.store_header       = MakeStoreHeader();
	request.host_subgroup_size = HOST_SUBGROUP_SIZE;
	WriteFile(directory / (tag + ".eboot"), eboot);
	request.eboot_path     = Utf8FromPath(directory / (tag + ".eboot"));
	request.output_path    = Utf8FromPath(directory / (tag + ".out"));
	request.progress_path  = Utf8FromPath(directory / (tag + ".progress"));
	request.generated_path = Utf8FromPath(directory / (tag + ".generated"));
	request.parent_pid     = PARENT_PID;
	return request;
}

// What a run left behind.
struct RunOutput {
	WorkerResult             result;
	Bytes                    file;
	std::vector<std::string> log;
};

// Runs the worker in this process and collects what it left in its files.
RunOutput Run(const WorkerRequest& request, const WorkerHooks& hooks = {}) {
	RunOutput run;
	run.result = RunShaderStoreWorker(request, hooks);
	run.file   = ReadFile(PathFromUtf8(request.output_path));
	run.log    = ReadLines(PathFromUtf8(request.progress_path));
	return run;
}

// A record of the output, with the bytes it was framed from.
struct StoredRecord {
	DecodedModuleRecord record;
	Bytes               raw;
};

// Reads the frames after the header until the file ends. `exact` tells that every byte was used.
std::vector<StoredRecord> ReadRecords(const Bytes& file, size_t header_bytes, bool& exact) {
	std::vector<StoredRecord> records;
	size_t                    at = header_bytes;
	exact                        = file.size() >= header_bytes;
	while (exact && at < file.size()) {
		const auto frame = UnframeStoreRecord(std::span<const uint8_t>(file).subspan(at));
		if (!frame) {
			exact = false;
			break;
		}
		IR::Reader reader {frame->raw};
		auto       type = StoreRecord::GraphicsPipeline;
		IR::IoAll(reader, type);
		StoredRecord stored;
		DecodeModuleRecord(reader, stored.record);
		exact      = exact && type == StoreRecord::Module && reader.ok && reader.bytes.empty();
		stored.raw = frame->raw;
		records.push_back(std::move(stored));
		at += frame->frame_bytes;
	}
	return records;
}

// The size of the generated header for the requests above.
size_t HeaderBytes(const WorkerRequest& request) {
	return request.store_header.size() + GENERATED_HEADER_TAIL_BYTES;
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

// The expected records of the main programs, in the order the rule gives. Every base candidate
// comes first, smallest program first, and then the variants, in the same program order.
std::vector<Expected> ExpectedOrder(const Bytes& eboot) {
	const auto scan   = ScanEbootShaders(eboot);
	const auto big    = PredictedFor(ProgramBig(), scan.programs.at(0));
	const auto middle = PredictedFor(ProgramMiddle(), scan.programs.at(1));
	const auto small  = PredictedFor(ProgramSmall(), scan.programs.at(2));
	return {Pick(small, {}), Pick(middle, {}), Pick(big, {}),    Pick(middle, {0}),
	        Pick(big, {0}),  Pick(big, {1}),   Pick(big, {0, 1})};
}

// The ids of the records, in file order.
std::vector<uint64_t> IdsOf(const std::vector<StoredRecord>& records) {
	std::vector<uint64_t> ids;
	for (const auto& stored: records) {
		ids.push_back(stored.record.id);
	}
	return ids;
}

// The code hashes of the records, in file order.
std::vector<uint64_t> IdsOfHashes(const std::vector<StoredRecord>& records) {
	std::vector<uint64_t> hashes;
	for (const auto& stored: records) {
		hashes.push_back(stored.record.hash);
	}
	return hashes;
}

// The ids the predictor expects, in the order given.
std::vector<uint64_t> IdsOf(const std::vector<Expected>& expected) {
	std::vector<uint64_t> ids;
	for (const auto& candidate: expected) {
		ids.push_back(candidate.id);
	}
	return ids;
}

// The `start` lines of a log, as "rank hash".
std::vector<std::string> StartLines(const std::vector<std::string>& log) {
	std::vector<std::string> starts;
	for (const auto& line: log) {
		const auto words = Words(line);
		if (words.size() == 3 && words[0] == "start") {
			starts.push_back(words[1] + " " + words[2]);
		}
	}
	return starts;
}

// ---- Request and header formats. ----

// A request with every field set to a value of its own, for the format tests.
WorkerRequest FullRequest() {
	WorkerRequest request;
	request.store_header       = MakeStoreHeader();
	request.host_subgroup_size = HOST_SUBGROUP_SIZE;
	request.eboot_path         = "D:/Games/PPSA00000.zar!/eboot.bin";
	request.output_path        = "cache/title.generated.123.tmp";
	request.progress_path      = "cache/title.generated.123.tmp.progress";
	request.generated_path     = "cache/title.generated";
	request.kept               = 0x123456789ull;
	request.start_rank         = 77;
	request.skip_hashes        = {0x1111222233334444ull, 5};
	request.failed_hashes      = {0xfedcba9876543210ull};
	request.parent_pid         = PARENT_PID;
	request.mode               = WorkerMode::Verify;
	request.generator_version  = 9;
	request.expected_order     = 0x0123456789abcdefull;
	return request;
}

// Whether every field of two requests is equal.
bool SameRequest(const WorkerRequest& a, const WorkerRequest& b) {
	return a.store_header == b.store_header && a.host_subgroup_size == b.host_subgroup_size &&
	       a.eboot_path == b.eboot_path && a.output_path == b.output_path &&
	       a.progress_path == b.progress_path && a.generated_path == b.generated_path &&
	       a.kept == b.kept && a.start_rank == b.start_rank && a.skip_hashes == b.skip_hashes &&
	       a.failed_hashes == b.failed_hashes && a.parent_pid == b.parent_pid && a.mode == b.mode &&
	       a.generator_version == b.generator_version && a.expected_order == b.expected_order;
}

// Every field of a request comes back, and each way a request can be wrong is refused.
void TestRequestFormat() {
	const auto request = FullRequest();
	const auto bytes   = EncodeWorkerRequest(request);
	const auto decoded = DecodeWorkerRequest(bytes);
	CHECK(decoded && SameRequest(*decoded, request));
	// The baseline differs in one field each time, so a decoder that lost a field is caught.
	WorkerRequest other = request;
	other.start_rank    = 78;
	CHECK(!SameRequest(other, request));
	other      = request;
	other.mode = WorkerMode::Generate;
	CHECK(!SameRequest(other, request));

	// Every proper prefix is refused, and so is a trailing byte.
	for (size_t length = 0; length < bytes.size(); length++) {
		CHECK(!DecodeWorkerRequest(std::span<const uint8_t>(bytes).first(length)));
	}
	auto longer = bytes;
	longer.push_back(0);
	CHECK(!DecodeWorkerRequest(longer));
	// A wrong magic or version, and a mode that does not exist.
	auto bad_magic = bytes;
	bad_magic[0] ^= 0xFF;
	CHECK(!DecodeWorkerRequest(bad_magic));
	auto bad_version = bytes;
	bad_version[sizeof(uint32_t)] ^= 0xFF;
	CHECK(!DecodeWorkerRequest(bad_version));
	WorkerRequest bad_mode = request;
	bad_mode.mode          = static_cast<WorkerMode>(7);
	CHECK(!DecodeWorkerRequest(EncodeWorkerRequest(bad_mode)));
	// A count that claims more elements than there are bytes cannot allocate them.
	auto hostile = bytes;
	Put<uint32_t>(hostile, 2 * sizeof(uint32_t), 0xFFFFFFFFu);
	CHECK(!DecodeWorkerRequest(hostile));
}

// A request survives a file, and the worker entry point tells a missing file, a damaged one and a
// good one with a missing eboot apart.
void TestRequestFile() {
	TempDirectory directory;
	const auto    request = FullRequest();
	CHECK(WriteWorkerRequestFile(directory / "request.bin", request));
	const auto read = ReadWorkerRequestFile(directory / "request.bin");
	CHECK(read && SameRequest(*read, request));
	CHECK(!ReadWorkerRequestFile(directory / "missing.bin"));

	CHECK(RunShaderStoreWorkerFromFile(directory / "missing.bin") == WORKER_EXIT_BAD_REQUEST);
	WriteFile(directory / "damaged.bin", Bytes {1, 2, 3});
	CHECK(RunShaderStoreWorkerFromFile(directory / "damaged.bin") == WORKER_EXIT_BAD_REQUEST);
	// This request is well formed, so it reaches the eboot, which does not exist.
	CHECK(RunShaderStoreWorkerFromFile(directory / "request.bin") == WORKER_EXIT_EBOOT_UNREADABLE);
	// A store header that cannot be decoded is a bad request, not an eboot problem.
	auto broken_header         = request;
	broken_header.store_header = {1, 2, 3};
	CHECK(WriteWorkerRequestFile(directory / "broken.bin", broken_header));
	CHECK(RunShaderStoreWorkerFromFile(directory / "broken.bin") == WORKER_EXIT_BAD_REQUEST);
	// A flag byte that is neither 0 nor 1 is refused, though the header is the right size. The
	// baseline is the good header, which gets as far as the missing eboot.
	auto flag_header                           = request;
	flag_header.store_header[sizeof(uint64_t)] = 2;
	CHECK(RunShaderStoreWorker(flag_header).exit_code == WORKER_EXIT_BAD_REQUEST);
	CHECK(RunShaderStoreWorker(request).exit_code == WORKER_EXIT_EBOOT_UNREADABLE);
}

// The generated header is the store header, the version and the identity, and the identity does not
// depend on the order of the programs.
void TestGeneratedHeader() {
	const auto                  store_header = MakeStoreHeader();
	std::vector<ScannedProgram> programs(3);
	programs[0].code_hash                  = 0x30;
	programs[1].code_hash                  = 0x10;
	programs[2].code_hash                  = 0x20;
	const auto                    identity = ComputeEbootIdentity(1000, programs);
	const std::array<uint64_t, 3> sorted {0x10, 0x20, 0x30};
	CHECK(identity.size == 1000);
	CHECK(identity.program_hash == XXH3_64bits(sorted.data(), sizeof sorted));
	// Programs in another order, a changed hash and a changed size each give what they should.
	std::vector<ScannedProgram> shuffled {programs[2], programs[0], programs[1]};
	CHECK(ComputeEbootIdentity(1000, shuffled) == identity);
	auto changed         = programs;
	changed[0].code_hash = 0x31;
	CHECK(!(ComputeEbootIdentity(1000, changed) == identity));
	CHECK(!(ComputeEbootIdentity(1001, programs) == identity));

	const auto header = EncodeGeneratedHeader(store_header, 5, identity);
	CHECK(header.size() == store_header.size() + GENERATED_HEADER_TAIL_BYTES);
	CHECK(std::equal(store_header.begin(), store_header.end(), header.begin()));
	auto parsed = ParseGeneratedHeader(header, store_header.size());
	CHECK(parsed && parsed->store_header == store_header && parsed->generator_version == 5 &&
	      parsed->identity == identity && parsed->bytes == header.size());
	CHECK(!ParseGeneratedHeader(std::span<const uint8_t>(header).first(header.size() - 1),
	                            store_header.size()));
	// Another version or identity gives other bytes, so a byte compare tells them apart.
	CHECK(EncodeGeneratedHeader(store_header, 6, identity) != header);
	CHECK(EncodeGeneratedHeader(store_header, 5, ComputeEbootIdentity(1001, programs)) != header);
}

// ---- Worker tests. ----

// The records are the predicted candidates in the rule's order, each one decodes, and its id is the
// predicted id and also the id the runtime would compute from the record's own module.
void TestRecordsInRankOrder() {
	TempDirectory directory;
	const auto    eboot    = BuildEboot(MainPrograms());
	const auto    request  = MakeRequest(directory, eboot);
	const auto    run      = Run(request);
	const auto    expected = ExpectedOrder(eboot);
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(expected.size() == 7);

	// The header is the request's store header, the version and the identity of this eboot.
	const auto scan   = ScanEbootShaders(eboot);
	const auto header = ParseGeneratedHeader(run.file, request.store_header.size());
	CHECK(header && header->store_header == request.store_header);
	CHECK(header && header->generator_version == SHADER_STORE_GENERATOR_VERSION);
	CHECK(header && header->identity == ComputeEbootIdentity(eboot.size(), scan.programs));

	bool       exact   = false;
	const auto records = ReadRecords(run.file, HeaderBytes(request), exact);
	CHECK(exact);
	CHECK(records.size() == expected.size());
	// The order of the ids is the order of the rule. An eboot order walk would list the big program
	// first, and a size only sort would put the middle program's variant before the big base.
	CHECK(IdsOf(records) == IdsOf(expected));
	std::vector<uint64_t> distinct = IdsOf(records);
	std::ranges::sort(distinct);
	CHECK(std::ranges::adjacent_find(distinct) == distinct.end());
	for (size_t index = 0; index < std::min(records.size(), expected.size()); index++) {
		const auto& record = records[index].record;
		CHECK(record.stage == ShaderType::Compute && record.hash == expected[index].hash);
		// This uses the worker's own formula, so it only shows the record agrees with itself.
		// The comparison with the predicted ids above is the real check.
		CHECK(record.id == ShaderModuleId(record.stage, record.hash, record.user_data_count,
		                                  record.static_state,
		                                  record.info.bindings.push_data_start_dword,
		                                  record.specialization, {}));
		CHECK(!record.spirv.empty());
	}
	CHECK(run.result.records_written == 7 && run.result.candidates == 7);
	CHECK(run.result.push_start_mismatches == 0 && run.result.drops.empty());
	CHECK(run.result.programs_scanned == 3 && run.result.programs_predicted == 3);
}

// The log has a start line for every rank, in order, each before its record, and the summary last.
void TestProgressLog() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot);
	const auto    run     = Run(request);
	const auto    scan    = ScanEbootShaders(eboot);
	// Programs by rank, which are small, middle, big, middle, big, big, big.
	const std::array<size_t, 7> program_of_rank {2, 1, 0, 1, 0, 0, 0};
	std::vector<std::string>    expected_starts;
	for (size_t rank = 0; rank < program_of_rank.size(); rank++) {
		expected_starts.push_back(std::to_string(rank) + " " +
		                          HashText(scan.programs[program_of_rank[rank]].code_hash));
	}
	CHECK(StartLines(run.log) == expected_starts);
	// The programs are logged before they are predicted, and the summary closes the log.
	size_t predicts = 0;
	for (const auto& line: run.log) {
		predicts += line.starts_with("predict ") ? 1 : 0;
	}
	CHECK(predicts == 3);
	CHECK(!run.log.empty() && run.log.back().starts_with("summary "));

	// While a candidate is translated its start line is the log's last line, and the output holds
	// only the records before it. A line written after the translation would not be there yet.
	TempDirectory second;
	const auto    hooked_request = MakeRequest(second, eboot);
	size_t        checked        = 0;
	bool          ordered        = true;
	WorkerHooks   hooks;
	hooks.before_candidate = [&](uint64_t rank) {
		const auto log     = ReadLines(PathFromUtf8(hooked_request.progress_path));
		const auto output  = ReadFile(PathFromUtf8(hooked_request.output_path));
		bool       exact   = false;
		const auto records = ReadRecords(output, HeaderBytes(hooked_request), exact);
		ordered            = ordered && !log.empty() && StartLines(log).size() == rank + 1 &&
		          Words(log.back()).size() == 3 && Words(log.back())[0] == "start" &&
		          Words(log.back())[1] == std::to_string(rank) && exact && records.size() == rank;
		checked++;
	};
	const auto hooked = Run(hooked_request, hooks);
	CHECK(hooked.result.exit_code == WORKER_EXIT_DONE);
	CHECK(checked == 7 && ordered);
}

// A worker started at a rank writes the records from that rank on, the same bytes a full run wrote.
void TestStartRank() {
	TempDirectory directory;
	const auto    eboot = BuildEboot(MainPrograms());
	const auto    full  = Run(MakeRequest(directory, eboot, "full"));
	bool          exact = false;
	const auto all = ReadRecords(full.file, HeaderBytes(MakeRequest(directory, eboot, "x")), exact);
	constexpr uint64_t START = 3;
	auto               from  = MakeRequest(directory, eboot, "from");
	from.start_rank          = START;
	const auto partial       = Run(from);
	const auto tail          = ReadRecords(partial.file, HeaderBytes(from), exact);
	CHECK(exact && all.size() == 7 && tail.size() == 7 - START);
	for (size_t index = 0; index < tail.size() && index + START < all.size(); index++) {
		CHECK(tail[index].raw == all[index + START].raw);
	}
	CHECK(partial.result.candidates_before_start == START && partial.result.candidates == 7);
	// The log starts at the start rank and counts up from there.
	const auto starts = StartLines(partial.log);
	CHECK(starts.size() == 7 - START && !starts.empty() && starts.front().starts_with("3 "));
	CHECK(starts.back().starts_with("6 "));
}

// A full run of the main programs, which the skip tests compare against.
struct FullRun {
	Bytes                     eboot;
	uint64_t                  big = 0;
	WorkerRequest             request;
	RunOutput                 run;
	std::vector<StoredRecord> records;
};

FullRun MakeFullRun(const TempDirectory& directory) {
	FullRun full;
	full.eboot   = BuildEboot(MainPrograms());
	full.big     = ScanEbootShaders(full.eboot).programs[0].code_hash;
	full.request = MakeRequest(directory, full.eboot, "full");
	full.run     = Run(full.request);
	bool exact   = false;
	full.records = ReadRecords(full.run.file, HeaderBytes(full.request), exact);
	return full;
}

// How many lines of a log start with the text.
size_t CountLines(const std::vector<std::string>& log, const std::string& prefix) {
	return static_cast<size_t>(std::ranges::count_if(
	    log, [&](const std::string& line) { return line.starts_with(prefix); }));
}

// A program in the request's skip hashes is never predicted, so it has no ranks at all. The other
// programs come out as the bytes they have in a full run, in the same order among themselves.
void TestSkipHashes() {
	TempDirectory directory;
	const auto    full = MakeFullRun(directory);
	// The big program's records are in the baseline, so it is the skip that removes them.
	CHECK(full.records.size() == 7 &&
	      CountLines(full.run.log, "predict " + HashText(full.big)) == 1);
	CHECK(std::ranges::count(IdsOfHashes(full.records), full.big) == 4);

	auto skip        = MakeRequest(directory, full.eboot, "skip");
	skip.skip_hashes = {full.big};
	const auto run   = Run(skip);
	bool       exact = false;
	const auto kept  = ReadRecords(run.file, HeaderBytes(skip), exact);
	// Four of the seven candidates are the big program's, so three records are left.
	CHECK(exact && kept.size() == 3 && run.result.programs_skipped_by_request == 1);
	CHECK(run.result.candidates == 3 && run.result.records_written == 3);
	CHECK(std::ranges::count(IdsOfHashes(kept), full.big) == 0);
	if (kept.size() == 3 && full.records.size() == 7) {
		CHECK(kept[0].raw == full.records[0].raw && kept[1].raw == full.records[1].raw &&
		      kept[2].raw == full.records[3].raw);
	}
	// Left out before prediction, so it is not even translated for the prediction.
	CHECK(CountLines(run.log, "predict " + HashText(full.big)) == 0);
}

// A program in failed_hashes keeps its ranks and is not built, so the ranks of the other programs
// do not move and a restart can go on at the next rank.
void TestFailedHashes() {
	TempDirectory directory;
	const auto    full = MakeFullRun(directory);

	auto failed          = MakeRequest(directory, full.eboot, "failed");
	failed.failed_hashes = {full.big};
	const auto run       = Run(failed);
	bool       exact     = false;
	const auto rest      = ReadRecords(run.file, HeaderBytes(failed), exact);
	CHECK(exact && rest.size() == 3 && run.result.candidates_failed_earlier == 4);
	CHECK(run.result.candidates == 7 && run.result.records_written == 3);
	// The ranks of the others are what they were, which are 0, 1 and 3.
	const auto full_starts = StartLines(full.run.log);
	const auto starts      = StartLines(run.log);
	CHECK(starts.size() == 3 && full_starts.size() == 7);
	if (starts.size() == 3 && full_starts.size() == 7) {
		CHECK(starts[0] == full_starts[0] && starts[1] == full_starts[1] &&
		      starts[2] == full_starts[3]);
	}
	if (rest.size() == 3 && full.records.size() == 7) {
		CHECK(rest[0].raw == full.records[0].raw && rest[1].raw == full.records[1].raw &&
		      rest[2].raw == full.records[3].raw);
	}
	// A failed program is still predicted, which is what keeps its ranks.
	CHECK(CountLines(run.log, "predict " + HashText(full.big)) == 1);
}

// A restarted worker keeps the good bytes of the file and carries on, so the file ends up the one
// an uninterrupted run makes. A kept file that does not start with this header is refused.
void TestKeptAppends() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot, "full");
	const auto    full    = Run(request);
	bool          exact   = false;
	const auto    records = ReadRecords(full.file, HeaderBytes(request), exact);
	// Cut the file after three records, and leave half of the fourth behind as a crash would.
	size_t cut = HeaderBytes(request);
	for (size_t index = 0; index < 3; index++) {
		cut += FrameStoreRecord(records[index].raw).size();
	}
	Bytes torn(full.file.begin(), full.file.begin() + static_cast<ptrdiff_t>(cut + TORN_BYTES));
	auto  resume = MakeRequest(directory, eboot, "resume");
	WriteFile(PathFromUtf8(resume.output_path), torn);
	resume.kept           = cut;
	resume.start_rank     = 3;
	resume.expected_order = full.result.order_hash;
	const auto resumed    = Run(resume);
	CHECK(resumed.result.exit_code == WORKER_EXIT_DONE);
	CHECK(resumed.file == full.file);

	// The same bytes with another stamp in front do not match this request's header.
	auto wrong = MakeRequest(directory, eboot, "wrong");
	WriteFile(PathFromUtf8(wrong.output_path), torn);
	wrong.store_header = MakeStoreHeader(OTHER_BUILD_STAMP);
	wrong.kept         = cut;
	wrong.start_rank   = 3;
	CHECK(RunShaderStoreWorker(wrong).exit_code == WORKER_EXIT_OUTPUT_FAILED);
	// A file shorter than what is said to be kept is refused too.
	auto short_file = MakeRequest(directory, eboot, "short");
	WriteFile(PathFromUtf8(short_file.output_path), torn);
	short_file.kept = cut + PAST_THE_FILE;
	CHECK(RunShaderStoreWorker(short_file).exit_code == WORKER_EXIT_OUTPUT_FAILED);
}

// A request in Verify mode against the finished file of the main programs.
struct VerifySetup {
	Bytes    eboot;
	fs::path finished;
};

VerifySetup MakeFinishedFile(const TempDirectory& directory) {
	VerifySetup setup;
	setup.eboot = BuildEboot(MainPrograms());
	auto make   = MakeRequest(directory, setup.eboot, "make");
	CHECK(Run(make).result.exit_code == WORKER_EXIT_DONE);
	// The finished file is what the supervisor would have renamed the output to.
	setup.finished = directory / "title.generated";
	WriteFile(setup.finished, ReadFile(PathFromUtf8(make.output_path)));
	return setup;
}

WorkerRequest VerifyRequest(const TempDirectory& directory, const VerifySetup& setup,
                            const Bytes& eboot, const std::string& tag) {
	auto request           = MakeRequest(directory, eboot, tag);
	request.mode           = WorkerMode::Verify;
	request.generated_path = Utf8FromPath(setup.finished);
	return request;
}

// Verify mode leaves a current file alone, and writes nothing.
void TestVerifyUpToDate() {
	TempDirectory directory;
	const auto    setup   = MakeFinishedFile(directory);
	const auto    request = VerifyRequest(directory, setup, setup.eboot, "same");
	const auto    run     = Run(request);
	CHECK(run.result.exit_code == WORKER_EXIT_UP_TO_DATE);
	CHECK(!fs::exists(PathFromUtf8(request.output_path)) && run.result.records_written == 0);
	CHECK(run.log.empty());

	// The same programs in another order are the same eboot size and the same identity, so only the
	// identity decides, and it does not depend on the order.
	auto swapped = MainPrograms();
	std::swap(swapped[1], swapped[2]);
	const auto reordered = VerifyRequest(directory, setup, BuildEboot(swapped), "reordered");
	CHECK(Run(reordered).result.exit_code == WORKER_EXIT_UP_TO_DATE);

	// Generate mode ignores a current file, and so does Verify mode when there is no file.
	auto generate           = MakeRequest(directory, setup.eboot, "generate");
	generate.generated_path = request.generated_path;
	CHECK(Run(generate).result.exit_code == WORKER_EXIT_DONE);
	auto none = MakeRequest(directory, setup.eboot, "none");
	none.mode = WorkerMode::Verify;
	CHECK(Run(none).result.exit_code == WORKER_EXIT_DONE);
}

// Verify mode generates again when the eboot, the host values or the generator version changed. The
// baseline for each is the unchanged request above, which is up to date.
void TestVerifyRegenerates() {
	TempDirectory directory;
	const auto    setup = MakeFinishedFile(directory);
	CHECK(Run(VerifyRequest(directory, setup, setup.eboot, "same")).result.exit_code ==
	      WORKER_EXIT_UP_TO_DATE);

	// Another eboot gives another identity. One program changed is enough.
	auto changed_programs            = MainPrograms();
	changed_programs[2].padding_nops = 1;
	const auto changed = VerifyRequest(directory, setup, BuildEboot(changed_programs), "changed");
	const auto regenerated = Run(changed);
	CHECK(regenerated.result.exit_code == WORKER_EXIT_DONE);
	CHECK(regenerated.result.records_written == 7 && fs::exists(PathFromUtf8(changed.output_path)));
	const auto header = ParseGeneratedHeader(regenerated.file, changed.store_header.size());
	CHECK(header && header->identity == regenerated.result.identity);

	auto stamp         = VerifyRequest(directory, setup, setup.eboot, "stamp");
	stamp.store_header = MakeStoreHeader(OTHER_BUILD_STAMP);
	CHECK(Run(stamp).result.exit_code == WORKER_EXIT_DONE);
	auto version              = VerifyRequest(directory, setup, setup.eboot, "version");
	version.generator_version = SHADER_STORE_GENERATOR_VERSION + 1;
	CHECK(Run(version).result.exit_code == WORKER_EXIT_DONE);
}

// Programs the store cannot hold are counted by what kept them out, and never reach the output.
void TestProgramsLeftOut() {
	TempDirectory directory;
	ProgramSpec   calls          = ProgramSmall();
	calls.calls                  = true;
	ProgramSpec derived          = ProgramSmall();
	derived.derived_descriptor   = true;
	derived.padding_nops         = 1;
	ProgramSpec odd              = ProgramMiddle();
	odd.loads[0].no_rule         = true;
	ProgramSpec unreadable       = ProgramSmall();
	unreadable.without_registers = true;
	unreadable.padding_nops      = 2;
	auto good                    = ProgramSmall();
	good.padding_nops            = 3;
	const auto  eboot            = BuildEboot({calls, derived, odd, unreadable, good});
	const auto  request          = MakeRequest(directory, eboot);
	const auto  run              = Run(request);
	const auto& result           = run.result;
	CHECK(result.exit_code == WORKER_EXIT_DONE);
	CHECK(result.programs_scanned == 5 && result.programs_predicted == 1);
	CHECK(result.programs_with_calls == 1);
	CHECK(result.programs_unclassified == 1);
	CHECK(result.programs_with_problems == 1);
	CHECK(result.programs_unreadable == 1);
	CHECK(result.candidates == 1 && result.records_written == 1);
	bool exact = false;
	CHECK(ReadRecords(run.file, HeaderBytes(request), exact).size() == 1 && exact);
}

// A program with no buffer that reads user data. The rule says a program with no buffers has no
// push data, but the module keeps the user data it reads there, so it starts at 0.
ProgramSpec ProgramUserDataOnly() {
	ProgramSpec spec;
	spec.store          = false;
	spec.user_data_only = true;
	return spec;
}

// The same program with an output buffer. The rule says a program with buffers starts at 0, which
// is right, so nothing differs from the program above except the buffer.
ProgramSpec ProgramUserDataWithBuffer() {
	ProgramSpec spec = ProgramUserDataOnly();
	spec.store       = true;
	spec.unused_records.push_back(
	    {Sl00Class::ReadOnly, UNUSED_REGISTER, SGPR_SECOND, "gUnused", false});
	return spec;
}

// The check on the push data start drops a program whose module starts elsewhere than the rule
// said. Here the rule is wrong for real, since the program has no buffer and still has push data.
void TestPushStartTripwireOnRealMiss() {
	TempDirectory directory;
	const auto    right_eboot = BuildEboot({ProgramUserDataWithBuffer()});
	const auto    right       = Run(MakeRequest(directory, right_eboot, "right"));
	CHECK(right.result.exit_code == WORKER_EXIT_DONE);
	CHECK(right.result.push_start_mismatches == 0 && right.result.drops.empty());
	CHECK(right.result.records_written == 1);

	const auto wrong_eboot   = BuildEboot({ProgramUserDataOnly()});
	const auto wrong_request = MakeRequest(directory, wrong_eboot, "wrong");
	const auto wrong         = Run(wrong_request);
	const auto scan          = ScanEbootShaders(wrong_eboot);
	CHECK(wrong.result.exit_code == WORKER_EXIT_DONE);
	CHECK(wrong.result.candidates == 1 && wrong.result.programs_predicted == 1);
	CHECK(wrong.result.push_start_mismatches == 1 && wrong.result.drops.size() == 1);
	if (wrong.result.drops.size() == 1) {
		const auto& drop = wrong.result.drops[0];
		CHECK(drop.reason == DropReason::PushStartMismatch && drop.rank == 0);
		CHECK(drop.hash == scan.programs[0].code_hash);
		// The rule said none, and the module starts at the first dword.
		CHECK(drop.predicted_push == IR::PushData::NoStart && drop.actual_push == 0);
	}
	// A record filed under the id the runtime would not compute is never found, so none is kept.
	bool       exact   = false;
	const auto records = ReadRecords(wrong.file, HeaderBytes(wrong_request), exact);
	CHECK(exact && records.empty() && wrong.result.records_written == 0);
	CHECK(wrong.result.candidates_dropped == 1);
	bool logged = false;
	for (const auto& line: wrong.log) {
		logged =
		    logged || line.find("push_start predicted=4294967295 actual=0") != std::string::npos;
	}
	CHECK(logged);
}

// A prediction made wrong on purpose drops the program, and only that program. Its later candidates
// are not built, though they would be right, and the other programs come out as in a full run.
void TestPushStartTripwireDropsTheRest() {
	TempDirectory directory;
	const auto    eboot            = BuildEboot(MainPrograms());
	const auto    scan             = ScanEbootShaders(eboot);
	const auto    big              = scan.programs[0].code_hash;
	const auto    baseline_request = MakeRequest(directory, eboot, "baseline");
	const auto    baseline         = Run(baseline_request);
	bool          exact            = false;
	const auto    all = ReadRecords(baseline.file, HeaderBytes(baseline_request), exact);
	CHECK(baseline.result.push_start_mismatches == 0 && all.size() == 7);

	constexpr uint32_t WRONG_START = 7;
	WorkerHooks        hooks;
	hooks.after_prediction = [&](uint64_t hash, std::vector<Candidate>& candidates) {
		if (hash == big) candidates.front().push_start = WRONG_START;
	};
	const auto request = MakeRequest(directory, eboot, "wrong");
	const auto run     = Run(request, hooks);
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(run.result.push_start_mismatches == 1 && run.result.drops.size() == 1);
	if (run.result.drops.size() == 1) {
		const auto& drop = run.result.drops[0];
		CHECK(drop.hash == big && drop.rank == 2);
		CHECK(drop.predicted_push == WRONG_START && drop.actual_push == 0);
	}
	// Seven candidates, and the four of the big program are gone. Its base is rank 2, and the
	// others are ranks 4, 5 and 6, which would not trip on their own.
	CHECK(run.result.candidates == 7 && run.result.records_written == 3);
	CHECK(run.result.candidates_dropped == 4);
	const auto records = ReadRecords(run.file, HeaderBytes(request), exact);
	CHECK(exact && records.size() == 3 && all.size() == 7);
	if (records.size() == 3 && all.size() == 7) {
		CHECK(records[0].raw == all[0].raw && records[1].raw == all[1].raw &&
		      records[2].raw == all[3].raw);
	}
	for (const auto& stored: records) {
		CHECK(stored.record.hash != big);
	}
	// The ranks 4 to 6 have no start line, so nothing was translated for them.
	CHECK(StartLines(run.log).size() == 3 + 1);
}

// A restart with nothing left to write still cuts the file back to the kept bytes, so a torn record
// does not stay at its end. The baseline is the torn file itself, which is longer.
void TestKeptTruncatesTornBytes() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot, "full");
	const auto    full    = Run(request);
	bool          exact   = false;
	const auto    records = ReadRecords(full.file, HeaderBytes(request), exact);
	size_t        cut     = HeaderBytes(request);
	for (size_t index = 0; index < 3; index++) {
		cut += FrameStoreRecord(records[index].raw).size();
	}
	Bytes torn(full.file.begin(), full.file.begin() + static_cast<ptrdiff_t>(cut + TORN_BYTES));
	auto  resume = MakeRequest(directory, eboot, "resume");
	WriteFile(PathFromUtf8(resume.output_path), torn);
	resume.kept           = cut;
	resume.start_rank     = MAIN_CANDIDATES;
	resume.expected_order = full.result.order_hash;
	const auto resumed    = Run(resume);
	CHECK(resumed.result.exit_code == WORKER_EXIT_DONE && resumed.result.records_written == 0);
	CHECK(torn.size() == cut + TORN_BYTES);
	CHECK(resumed.file.size() == cut);
	CHECK(std::equal(resumed.file.begin(), resumed.file.end(), full.file.begin()));
}

// A restart that computes another ordering than the run it continues exits without touching the
// kept bytes. The baseline is the same restart with the right ordering, which goes on.
void TestOrderGuard() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot, "full");
	const auto    full    = Run(request);
	CHECK(full.result.order_hash != 0);
	bool       exact   = false;
	const auto records = ReadRecords(full.file, HeaderBytes(request), exact);
	size_t     cut     = HeaderBytes(request);
	for (size_t index = 0; index < 3; index++) {
		cut += FrameStoreRecord(records[index].raw).size();
	}
	const Bytes kept(full.file.begin(), full.file.begin() + static_cast<ptrdiff_t>(cut));

	auto wrong = MakeRequest(directory, eboot, "wrong");
	WriteFile(PathFromUtf8(wrong.output_path), kept);
	wrong.kept           = cut;
	wrong.start_rank     = 3;
	wrong.expected_order = full.result.order_hash ^ 1;
	const auto refused   = Run(wrong);
	CHECK(refused.result.exit_code == WORKER_EXIT_ORDER_MISMATCH);
	CHECK(refused.file == kept && refused.result.records_written == 0);

	auto right = MakeRequest(directory, eboot, "right");
	WriteFile(PathFromUtf8(right.output_path), kept);
	right.kept           = cut;
	right.start_rank     = 3;
	right.expected_order = full.result.order_hash;
	CHECK(Run(right).result.exit_code == WORKER_EXIT_DONE);
	// A restart that keeps no bytes but starts past rank 0 is guarded too.
	auto ranked_only           = MakeRequest(directory, eboot, "ranked_only");
	ranked_only.start_rank     = 3;
	ranked_only.expected_order = full.result.order_hash ^ 1;
	CHECK(Run(ranked_only).result.exit_code == WORKER_EXIT_ORDER_MISMATCH);
	// A new file has nothing to disagree with, and an expected order of zero means no expectation.
	auto fresh           = MakeRequest(directory, eboot, "fresh");
	fresh.expected_order = full.result.order_hash ^ 1;
	CHECK(Run(fresh).result.exit_code == WORKER_EXIT_DONE);
}

// Each program's `predict` line is the log's last line when its prediction starts.
void TestPredictLineComesFirst() {
	TempDirectory directory;
	const auto    eboot   = BuildEboot(MainPrograms());
	const auto    request = MakeRequest(directory, eboot);
	size_t        checked = 0;
	bool          ordered = true;
	WorkerHooks   hooks;
	hooks.before_prediction = [&](uint64_t hash) {
		const auto log = ReadLines(PathFromUtf8(request.progress_path));
		ordered        = ordered && !log.empty() && log.back() == "predict " + HashText(hash);
		checked++;
	};
	CHECK(Run(request, hooks).result.exit_code == WORKER_EXIT_DONE);
	CHECK(checked == 3 && ordered);
}

// The worker goes on past a drop. This covers a crash later in a run that has already dropped a
// program. The supervisor restarts with the dropped program in failed_hashes and the rank after
// the drop, and the file ends up as the run that went on. The baseline is the same restart without
// failed_hashes, which builds the program again.
void TestRestartAfterDrop() {
	TempDirectory directory;
	const auto    eboot = BuildEboot(MainPrograms());
	const auto    big   = ScanEbootShaders(eboot).programs[0].code_hash;
	WorkerHooks   hooks;
	hooks.after_prediction = [&](uint64_t hash, std::vector<Candidate>& candidates) {
		if (hash == big) candidates.front().push_start = 7;
	};
	const auto request = MakeRequest(directory, eboot, "first");
	const auto first   = Run(request, hooks);
	CHECK(first.result.drops.size() == 1 && first.result.records_written == 3);
	if (first.result.drops.size() != 1) return;
	const auto& drop    = first.result.drops[0];
	bool        exact   = false;
	const auto  records = ReadRecords(first.file, HeaderBytes(request), exact);
	// The bytes the supervisor kept when it saw the drop line, which are the records before it.
	size_t cut = HeaderBytes(request);
	for (size_t index = 0; index < 2; index++) {
		cut += FrameStoreRecord(records[index].raw).size();
	}
	const Bytes kept(first.file.begin(), first.file.begin() + static_cast<ptrdiff_t>(cut));

	auto restart = MakeRequest(directory, eboot, "restart");
	WriteFile(PathFromUtf8(restart.output_path), kept);
	restart.kept           = cut;
	restart.start_rank     = drop.rank + 1;
	restart.expected_order = first.result.order_hash;
	restart.failed_hashes  = {drop.hash};
	const auto restarted   = Run(restart);
	CHECK(restarted.result.exit_code == WORKER_EXIT_DONE);
	CHECK(restarted.file == first.file);

	auto again = MakeRequest(directory, eboot, "again");
	WriteFile(PathFromUtf8(again.output_path), kept);
	again.kept           = cut;
	again.start_rank     = drop.rank + 1;
	again.expected_order = first.result.order_hash;
	const auto rebuilt   = Run(again);
	CHECK(rebuilt.result.exit_code == WORKER_EXIT_DONE && rebuilt.file != first.file);
}

// A specialization with an extra buffer is not dropped, since indirect children are appended to it.
// Only a program with more buffers than the specialization is. The fixture has no program with an
// image or a sampler, so those two comparisons are not exercised here.
void TestExtraSpecializationBufferIsKept() {
	TempDirectory directory;
	const auto    eboot = BuildEboot(MainPrograms());
	WorkerHooks   hooks;
	hooks.after_prediction = [&](uint64_t, std::vector<Candidate>& candidates) {
		auto& buffers = candidates.front().specialization.buffers;
		auto  extra   = buffers.back();
		// A child that points at the first buffer, as an indirect child does.
		extra.indirect_root = 0;
		buffers.push_back(extra);
	};
	const auto run = Run(MakeRequest(directory, eboot), hooks);
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(run.result.resource_count_mismatches == 0);
	CHECK(std::ranges::none_of(run.result.drops, [](const WorkerDrop& drop) {
		return drop.reason == DropReason::ResourceCountMismatch;
	}));
	// Every candidate is written, so none was dropped for another reason either.
	CHECK(run.result.records_written == MAIN_CANDIDATES && run.result.drops.empty());
}

// The comparison that guards a specialization, with hand built specializations. Each kind has a
// program that lists more than the specialization, which is refused, and the baseline with equal
// counts, which is not. A specialization with an extra entry of every kind is accepted.
void TestProgramExceedsSpecialization() {
	const auto make = [](size_t buffers, size_t images, size_t samplers) {
		IR::ResourceSpecialization specialization;
		specialization.buffers.resize(buffers);
		specialization.images.resize(images);
		specialization.samplers.resize(samplers);
		return specialization;
	};
	constexpr size_t COUNT = 2;
	const auto       equal = make(COUNT, COUNT, COUNT);
	CHECK(!ProgramExceedsSpecialization(COUNT, COUNT, COUNT, equal));
	CHECK(ProgramExceedsSpecialization(COUNT + 1, COUNT, COUNT, equal));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT + 1, COUNT, equal));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT, COUNT + 1, equal));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT, COUNT, make(COUNT - 1, COUNT, COUNT)));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT, COUNT, make(COUNT, COUNT - 1, COUNT)));
	CHECK(ProgramExceedsSpecialization(COUNT, COUNT, COUNT, make(COUNT, COUNT, COUNT - 1)));
	CHECK(
	    !ProgramExceedsSpecialization(COUNT, COUNT, COUNT, make(COUNT + 1, COUNT + 1, COUNT + 1)));
	CHECK(!ProgramExceedsSpecialization(0, 0, 0, make(0, 0, 0)));
}

// A translation that lists another number of buffers than the prediction drops the program, as a
// wrong push start does. The prediction is made wrong by taking a buffer off the base candidate.
void TestResourceCountMismatch() {
	TempDirectory directory;
	const auto    eboot = BuildEboot(MainPrograms());
	const auto    big   = ScanEbootShaders(eboot).programs[0].code_hash;
	WorkerHooks   hooks;
	hooks.after_prediction = [&](uint64_t hash, std::vector<Candidate>& candidates) {
		auto& buffers = candidates.front().specialization.buffers;
		if (hash == big) buffers.erase(buffers.begin());
	};
	const auto request = MakeRequest(directory, eboot);
	const auto run     = Run(request, hooks);
	CHECK(run.result.exit_code == WORKER_EXIT_DONE);
	CHECK(run.result.resource_count_mismatches == 1 && run.result.push_start_mismatches == 0);
	CHECK(run.result.drops.size() == 1);
	if (run.result.drops.size() == 1) {
		CHECK(run.result.drops[0].reason == DropReason::ResourceCountMismatch);
		CHECK(run.result.drops[0].hash == big && run.result.drops[0].rank == 2);
	}
	// The big program's four candidates are gone and the other three are written.
	CHECK(run.result.records_written == 3 && run.result.candidates_dropped == 4);
	CHECK(StartLines(run.log).size() == 4);
	bool logged = false;
	for (const auto& line: run.log) {
		logged = logged || line.ends_with("resource_count");
	}
	CHECK(logged);
}

} // namespace

int main() {
	TestRequestFormat();
	TestRequestFile();
	TestGeneratedHeader();
	TestRecordsInRankOrder();
	TestProgressLog();
	TestStartRank();
	TestSkipHashes();
	TestFailedHashes();
	TestKeptAppends();
	TestVerifyUpToDate();
	TestVerifyRegenerates();
	TestProgramsLeftOut();
	TestPushStartTripwireOnRealMiss();
	TestPushStartTripwireDropsTheRest();
	TestKeptTruncatesTornBytes();
	TestOrderGuard();
	TestPredictLineComesFirst();
	TestRestartAfterDrop();
	TestResourceCountMismatch();
	TestProgramExceedsSpecialization();
	TestExtraSpecializationBufferIsKept();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("shader store worker tests passed");
	return 0;
}
