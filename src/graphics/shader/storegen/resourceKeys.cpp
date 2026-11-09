#include "graphics/shader/storegen/resourceKeys.h"

namespace Libs::Graphics::StoreGen {

namespace {

namespace IR = ShaderRecompiler::IR;

// The descriptor widths in dwords.
constexpr uint32_t BUFFER_DWORDS  = 4;
constexpr uint32_t IMAGE_DWORDS   = 8;
constexpr uint32_t R128_DWORDS    = 4;
constexpr uint32_t SAMPLER_DWORDS = 4;
// A literal pool descriptor has its table address in dwords 0 and 1 and constants in 2 and 3.
constexpr uint32_t LITERAL_WORD2 = 2;
constexpr uint32_t LITERAL_WORD3 = 3;

// What one descriptor dword is rooted in.
struct DwordRoot {
	enum class Kind { Unknown, User, Eud } kind = Kind::Unknown;
	uint32_t reg    = 0; // The user SGPR, or the EUD pointer's low SGPR.
	uint32_t offset = 0; // The byte offset in the EUD table.
};

// Classifies a dword as a user SGPR, or one dword of an extended user data table.
// A table dword is a scalar load, whose address is a pair of user SGPRs. The load's immediate
// offset is in the memory metadata and not in the instruction's arguments, which only hold the
// dynamic part.
DwordRoot ClassifyDword(const IR::ResourcePlan& plan, IR::Value value) {
	value            = value.Resolve();
	const auto* inst = value.TryInstruction();
	if (inst == nullptr) return {};
	if (inst->GetOpcode() == IR::ValueOpcode::GetUserData) {
		if (inst->NumArgs() != 1 || inst->Arg(0).GetType() != IR::Type::ScalarReg) return {};
		return {DwordRoot::Kind::User, IR::RegIndex(inst->Arg(0).ScalarRegister()), 0};
	}
	if (inst->GetOpcode() != IR::ValueOpcode::ReadConst || inst->NumArgs() != 2) return {};
	const auto slot = inst->Arg(1).Resolve();
	if (!slot.IsImmediate() || slot.GetType() != IR::Type::U32 ||
	    slot.U32() >= plan.srt_reads.size())
		return {};
	const auto* read = plan.srt_reads[slot.U32()].value.Resolve().TryInstruction();
	if (read == nullptr || read->GetOpcode() != IR::ValueOpcode::LoadAddressU32 ||
	    read->NumArgs() < 2)
		return {};
	const auto* handle  = read->Arg(0).Resolve().TryInstruction();
	const auto  dynamic = read->Arg(1).Resolve();
	if (handle == nullptr || handle->GetOpcode() != IR::ValueOpcode::GetAddressResource ||
	    handle->NumArgs() != 2 || !dynamic.IsImmediate() || dynamic.GetType() != IR::Type::U32) {
		return {};
	}
	const auto low  = ClassifyDword(plan, handle->Arg(0));
	const auto high = ClassifyDword(plan, handle->Arg(1));
	if (low.kind != DwordRoot::Kind::User || high.kind != DwordRoot::Kind::User ||
	    high.reg != low.reg + 1)
		return {};
	const auto memory = read->Flags<IR::MemoryFlags>().index;
	if (memory >= plan.memory_info.size()) return {};
	return {DwordRoot::Kind::Eud, low.reg, plan.memory_info[memory].offset + dynamic.U32()};
}

// The pc-relative sum of a literal pool descriptor, IAdd64(GetShaderBase, imm P), that one dword
// of the descriptor extracts a half of.
// @param half   0 for the low dword, 1 for the high dword.
// @param pc_after receives P, the pc after the s_getpc.
// @return whether `extract` is CompositeExtractU64(sum, half) of such a sum.
bool ReadPcHalf(IR::Value extract, uint32_t half, uint32_t& pc_after) {
	const auto* inst = extract.Resolve().TryInstruction();
	if (inst == nullptr || inst->GetOpcode() != IR::ValueOpcode::CompositeExtractU64 ||
	    inst->NumArgs() != 2)
		return false;
	const auto index = inst->Arg(1).Resolve();
	// IsImmediate must come before GetType, which is only defined for an immediate.
	if (!index.IsImmediate() || index.GetType() != IR::Type::U32 || index.U32() != half)
		return false;
	const auto* sum = inst->Arg(0).Resolve().TryInstruction();
	if (sum == nullptr || sum->GetOpcode() != IR::ValueOpcode::IAdd64 || sum->NumArgs() != 2)
		return false;
	const auto* base  = sum->Arg(0).Resolve().TryInstruction();
	const auto  delta = sum->Arg(1).Resolve();
	if (base == nullptr || base->GetOpcode() != IR::ValueOpcode::GetShaderBase ||
	    !delta.IsImmediate() || delta.GetType() != IR::Type::U64) {
		return false;
	}
	pc_after = static_cast<uint32_t>(delta.U64());
	return true;
}

// Finds the pc-relative half among the two operands of a 32-bit add, which may come in either
// order. The other operand is returned in `other`.
bool ReadAddOfPcHalf(IR::Value value, uint32_t half, uint32_t& pc_after, IR::Value& other) {
	const auto* add = value.Resolve().TryInstruction();
	if (add == nullptr || add->GetOpcode() != IR::ValueOpcode::IAdd32 || add->NumArgs() != 2)
		return false;
	for (size_t side = 0; side < 2; side++) {
		if (ReadPcHalf(add->Arg(side), half, pc_after)) {
			other = add->Arg(1 - side).Resolve();
			return true;
		}
	}
	return false;
}

// Recognises the first two dwords of a descriptor built from s_getpc. Dword 0 is
// IAdd32(imm A, extract(IAdd64(GetShaderBase, imm P), 0)) and dword 1 is the matching carry add of
// the high half, IAdd32(extract(IAdd64(GetShaderBase, imm P), 1), carry), with the same P, as the
// python tracker requires of a pc and a pchi word. P is the pc after the s_getpc and A is the
// table's offset from it.
bool ReadLiteralBase(IR::Value low, IR::Value high, uint32_t& pc_after, uint32_t& table_offset) {
	IR::Value offset;
	IR::Value carry;
	uint32_t  high_pc = 0;
	if (!ReadAddOfPcHalf(low, 0, pc_after, offset) || !offset.IsImmediate() ||
	    offset.GetType() != IR::Type::U32 || !ReadAddOfPcHalf(high, 1, high_pc, carry) ||
	    high_pc != pc_after) {
		return false;
	}
	table_offset = offset.U32();
	return true;
}

// Names a descriptor of `width` dwords, or nothing when it is not four user SGPRs, a run of EUD
// dwords or a literal pool table.
std::optional<DescriptorKey> KeyOfSource(const IR::ResourcePlan& plan, uint32_t source_index,
                                         uint32_t width) {
	if (source_index >= plan.descriptor_sources.size()) return std::nullopt;
	const auto& source = plan.descriptor_sources[source_index];
	if (source.indirect_descriptor || source.dword_count < width) return std::nullopt;

	std::vector<DwordRoot> roots;
	for (uint32_t word = 0; word < width; word++)
		roots.push_back(ClassifyDword(plan, source.dwords[word]));
	bool all_user = true;
	bool all_eud  = true;
	for (uint32_t word = 0; word < width; word++) {
		all_user = all_user && roots[word].kind == DwordRoot::Kind::User &&
		           roots[word].reg == roots[0].reg + word;
		all_eud = all_eud && roots[word].kind == DwordRoot::Kind::Eud &&
		          roots[word].reg == roots[0].reg &&
		          roots[word].offset == roots[0].offset + word * sizeof(uint32_t);
	}
	if (all_user) return DescriptorKey {DescriptorKeyKind::User, roots[0].reg, 0, 0, 0};
	if (all_eud) return DescriptorKey {DescriptorKeyKind::Eud, roots[0].reg, roots[0].offset, 0, 0};

	uint32_t pc_after     = 0;
	uint32_t table_offset = 0;
	// IsImmediate must come before GetType, which is only defined for an immediate.
	const auto word2 =
	    width > LITERAL_WORD3 ? source.dwords[LITERAL_WORD2].Resolve() : IR::Value {};
	const auto word3 =
	    width > LITERAL_WORD3 ? source.dwords[LITERAL_WORD3].Resolve() : IR::Value {};
	if (width == BUFFER_DWORDS && word2.IsImmediate() && word3.IsImmediate() &&
	    word2.GetType() == IR::Type::U32 && word3.GetType() == IR::Type::U32 &&
	    ReadLiteralBase(source.dwords[0], source.dwords[1], pc_after, table_offset)) {
		return DescriptorKey {DescriptorKeyKind::Literal, pc_after, table_offset, word2.U32(),
		                      word3.U32()};
	}
	return std::nullopt;
}

} // namespace

std::optional<ResourceFacts> ExtractResourceFacts(const IR::ResourcePlan& plan, std::string* why) {
	const auto fail = [&](const char* kind, size_t index,
	                      const char* reason) -> std::optional<ResourceFacts> {
		if (why != nullptr) *why = std::string(kind) + " " + std::to_string(index) + " " + reason;
		return std::nullopt;
	};
	ResourceFacts facts;
	const auto&   info = plan.info;
	for (size_t index = 0; index < info.buffers.size(); index++) {
		const auto& buffer = info.buffers[index];
		if (buffer.indirect_root != IR::BufferResource::NoIndirectBuffer)
			return fail("buffer", index, "is indirect");
		const auto key = KeyOfSource(plan, buffer.source, BUFFER_DWORDS);
		if (!key) return fail("buffer", index, "has a descriptor that is indirect or unclassified");
		facts.buffers.push_back({*key, buffer.formatted, buffer.scalar, buffer.written});
	}
	for (size_t index = 0; index < info.images.size(); index++) {
		const auto& image = info.images[index];
		if (image.indirect_root != IR::ImageResource::NoIndirectImage || image.heap)
			return fail("image", index, "is indirect");
		const auto key = KeyOfSource(plan, image.source, image.r128 ? R128_DWORDS : IMAGE_DWORDS);
		if (!key) return fail("image", index, "has a descriptor that is indirect or unclassified");
		facts.images.push_back({*key, static_cast<uint32_t>(image.dimension), image.cube});
	}
	// Python counts every sampler and the predictor only needs the count, so a sampler that does
	// not classify, such as a static one made of immediate dwords, is kept with no key.
	for (size_t index = 0; index < info.samplers.size(); index++) {
		const auto& sampler = info.samplers[index];
		if (sampler.indirect_root != IR::SamplerResource::NoIndirectSampler ||
		    (sampler.source < plan.descriptor_sources.size() &&
		     plan.descriptor_sources[sampler.source].indirect_descriptor)) {
			return fail("sampler", index, "is indirect");
		}
		facts.samplers.push_back(KeyOfSource(plan, sampler.source, SAMPLER_DWORDS));
	}
	return facts;
}

} // namespace Libs::Graphics::StoreGen
