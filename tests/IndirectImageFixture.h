#ifndef KYTY_TESTS_INDIRECTIMAGEFIXTURE_H_
#define KYTY_TESTS_INDIRECTIMAGEFIXTURE_H_

#include "graphics/guest_gpu/gpu_defs.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceTracking.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// An IR program built by hand, and the indirect image table program the resource tracking tests
// and the prescan preparer tests share. The table program reads a 2D image descriptor out of a
// table in guest memory, and HeapTableSetup gives the user data and memory such a walk reads. Kept
// in its own namespace, since a test that includes it may have its own Program or Fixture.
namespace IndirectImageTest {

using namespace Libs::Graphics::ShaderRecompiler::IR;
using Libs::Graphics::ShaderComputeInputInfo;
using Libs::Graphics::ShaderType;
namespace Decoder = Libs::Graphics::ShaderRecompiler::Decoder;

// Throws when a fixture is built wrongly, which the including test reports as a failure.
inline void FixtureCheck(bool condition, const char *message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

struct Fixture {
  Program program;
  Block *block = nullptr;
  ShaderComputeInputInfo compute_info{};

  explicit Fixture(ShaderType stage = ShaderType::Compute) {
    program.stage = stage;
    program.user_data_count = 64;
    block = AddBlock();
  }

  Block *AddBlock() {
    auto storage = std::make_unique<Block>();
    auto *result = storage.get();
    program.block_storage.push_back(std::move(storage));
    program.blocks.push_back(result);
    program.blocks.back()->id =
        static_cast<uint32_t>(program.blocks.size() - 1u);
    return result;
  }

  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args = {},
             uint64_t flags = 0, Block *destination = nullptr) {
    if (NumArgsOf(opcode) != std::numeric_limits<size_t>::max() &&
        NumArgsOf(opcode) != args.size()) {
      throw std::runtime_error(std::string(ValueOpcodeName(opcode)) +
                               " argument count");
    }
    auto &inst = (destination != nullptr ? destination : block)
                     ->AppendNewInst(opcode, args, flags);
    return Value(&inst);
  }

  template <typename T>
  Value Emit(ValueOpcode opcode, std::initializer_list<Value> args, T flags,
             Block *destination = nullptr) {
    uint64_t bits = 0;
    std::memcpy(&bits, &flags, sizeof(flags));
    return Emit(opcode, args, bits, destination);
  }

  Value UserData(uint32_t index) {
    return Emit(ValueOpcode::GetUserData,
                {Value(static_cast<ScalarReg>(index))});
  }

  MemoryFlags AddMemory(MemoryInfo memory, uint32_t pc) {
    const auto index = static_cast<uint32_t>(program.memory_info.size());
    program.memory_info.push_back(memory);
    return {index, pc};
  }

  Value Buffer(std::array<Value, 4> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetBufferResource,
                {dwords[0], dwords[1], dwords[2], dwords[3]},
                MemoryFlags{0, pc});
  }

  Value Address(Value low, Value high, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetAddressResource, {low, high},
                MemoryFlags{0, pc});
  }

  Value Image(std::array<Value, 8> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetImageResource,
                {dwords[0], dwords[1], dwords[2], dwords[3], dwords[4],
                 dwords[5], dwords[6], dwords[7]},
                MemoryFlags{0, pc});
  }

  Value Sampler(std::array<Value, 4> dwords, uint32_t pc = 0) {
    return Emit(ValueOpcode::GetSamplerResource,
                {dwords[0], dwords[1], dwords[2], dwords[3]},
                MemoryFlags{0, pc});
  }

  Value ImageAddress() {
    return Emit(ValueOpcode::MakeImageAddress,
                {Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                 Value(0u), Value(0u), Value(0u), Value(0u), Value(0u),
                 Value(0u), Value(0u), Value(0u)});
  }

  void PlanAndTrack() { PlanAndTrack({.compute = &compute_info}); }

  void PlanAndTrack(Libs::Graphics::ShaderStageInputInfo input) {
    for (size_t index = 0; index < program.blocks.size(); ++index) {
      const auto condition = program.blocks[index]->condition;
      if (!condition.IsEmpty())
        Emit(ValueOpcode::Reference, {condition}, 0, program.blocks[index]);
    }
    for (auto *target : program.blocks) {
      for (auto &inst : *target) {
        if (inst.HasUses() || inst.MayHaveSideEffects() ||
            (BufferAccessOf(inst.GetOpcode()) == BufferAccess::None &&
             AddressOpcodeInfoOf(inst.GetOpcode()).access == AddressAccess::None &&
             ImageOpcodeInfoOf(inst.GetOpcode()).access == ImageAccess::None))
          continue;
        auto value = Value(&inst);
        if (value.GetType() == Type::U32x4)
          value = Emit(ValueOpcode::CompositeExtractU32x4, {value, Value(0u)}, 0, target);
        if (value.GetType() == Type::U8)
          value = Emit(ValueOpcode::ConvertU32U8, {value}, 0, target);
        FixtureCheck(value.GetType() == Type::U32, "unhandled memory result type in fixture");
        Emit(ValueOpcode::ReferenceU32, {value}, 0, target);
      }
    }
    TrackResources(program, {}, {}, input);
  }
};

struct LinearTestMemory {
  uint64_t base = 0x1000;
  std::vector<uint32_t> words = std::vector<uint32_t>(0x2200 / 4);
  uint64_t fail_address = UINT64_MAX;
  uint64_t watched_address = UINT64_MAX;
  uint32_t watched_reads = 0;
  size_t watched_dwords = 0;
  uint32_t reads = 0;
  uint32_t descriptor_reads = 0;
};

inline bool ReadLinearTestMemory(void *userdata, uint64_t address, std::span<uint32_t> values) {
  auto *memory = static_cast<LinearTestMemory *>(userdata);
  if (memory == nullptr || address < memory->base ||
      values.size_bytes() > memory->words.size() * sizeof(uint32_t) ||
      address - memory->base > memory->words.size() * sizeof(uint32_t) - values.size_bytes() ||
      (memory->fail_address >= address && memory->fail_address - address < values.size_bytes())) {
    return false;
  }
  std::memcpy(values.data(), reinterpret_cast<const uint8_t *>(memory->words.data()) +
                                (address - memory->base), values.size_bytes());
  ++memory->reads;
  if (values.size() == 8u) ++memory->descriptor_reads;
  if (memory->watched_address >= address &&
      memory->watched_address - address < values.size_bytes()) {
    ++memory->watched_reads;
    memory->watched_dwords = values.size();
  }
  return true;
}

inline std::unique_ptr<Fixture>
MakeIndirectImageFixture(bool malformed, uint32_t table_stride = 32, uint32_t table_offset = 0,
                         uint32_t table_shader_offset = 0, bool nested = false,
                         ValueOpcode access = ValueOpcode::ImageSampleRaw, bool finite_key = false) {
  auto fixture = std::make_unique<Fixture>();
  std::array<Value, 4> material_words;
  std::array<Value, 4> heap_words;
  for (uint32_t dword = 0; dword < 4; dword++) {
    material_words[dword] = fixture->UserData(dword);
    heap_words[dword] = fixture->UserData(dword + 4u);
  }
  const auto material = fixture->Buffer(material_words, 0x10d8);
  const auto heap = finite_key ? fixture->Address(heap_words[0], heap_words[1], 0x10d8)
                               : fixture->Buffer(heap_words, 0x10d8);
  const auto invocation = fixture->Emit(
      ValueOpcode::GetBuiltin,
      {Value(static_cast<uint32_t>(finite_key ? StageInputKind::WorkgroupId
                                              : StageInputKind::GlobalInvocationId)), Value(0u)});
  const auto selector = finite_key
      ? fixture->Emit(ValueOpcode::BitwiseXor32, {invocation, Value(1u)})
      : fixture->Emit(ValueOpcode::ReadFirstLane, {invocation, Value(true)});
  Value member;
  if (nested) {
    std::array<Value, 4> root_words;
    for (uint32_t word = 0; word < 4u; ++word) root_words[word] = fixture->UserData(12u + word);
    const auto root = fixture->Buffer(root_words);
    MemoryInfo root_memory;
    root_memory.kind = ResourceKind::ScalarBuffer;
    const auto root_offset = fixture->Emit(ValueOpcode::IMul32, {selector, Value(32u)});
    const auto pointer = fixture->Emit(ValueOpcode::ReadConstBuffer, {root, root_offset},
                                       fixture->AddMemory(root_memory, 0x1000));
    const auto masked = fixture->Emit(ValueOpcode::BitwiseAnd32, {pointer, Value(0xfffffff1u)});
    member = fixture->Emit(ValueOpcode::IAdd32, {masked, Value(16u)});
  } else {
    member = fixture->Emit(ValueOpcode::IMul32, {selector, Value(224u)});
  }
  fixture->Emit(ValueOpcode::ReferenceU32, {member});
  MemoryInfo material_scalar;
  material_scalar.kind = ResourceKind::ScalarBuffer;
  material_scalar.offset = 4u;
  auto key = fixture->Emit(ValueOpcode::ReadConstBuffer, {material, member},
                           fixture->AddMemory(material_scalar, 0x10d8));
  if (finite_key) key = fixture->Emit(ValueOpcode::BitFieldUExtract,
                                     {key, Value(16u), Value(5u)});
  auto heap_offset = fixture->Emit(ValueOpcode::IMul32, {key, Value(table_stride)});
  if (table_shader_offset != 0u)
    heap_offset = fixture->Emit(ValueOpcode::IAdd32, {heap_offset, Value(table_shader_offset)});
  std::array<Value, 8> image_words;
  MemoryInfo heap_scalar;
  heap_scalar.kind = finite_key ? ResourceKind::ScalarAddress : ResourceKind::ScalarBuffer;
  for (uint32_t dword = 0; dword < image_words.size(); dword++) {
    auto component = heap_scalar;
    component.offset = table_offset + dword * sizeof(uint32_t);
    if (malformed && dword == image_words.size() - 1u) {
      component.offset += sizeof(uint32_t);
    }
    const auto flags = fixture->AddMemory(component, 0x10d8);
    image_words[dword] = finite_key
        ? fixture->Emit(ValueOpcode::LoadAddressU32,
                        {heap, heap_offset, Value(0u), Value(true)}, flags)
        : fixture->Emit(ValueOpcode::ReadConstBuffer, {heap, heap_offset}, flags);
  }
  if (nested) {
    const auto field = fixture->Emit(ValueOpcode::BitwiseAnd32, {image_words[4], Value(7u)});
    fixture->Emit(ValueOpcode::ReferenceU32, {field});
  }
  const auto image = fixture->Image(image_words, 0x10f0);
  const auto sampler =
      fixture->Sampler({Value(0u), Value(0u), Value(0u), Value(0u)}, 0x10f0);
  MemoryInfo sample;
  sample.kind = ResourceKind::Image;
  sample.image_dimension = Decoder::ImageDimension::Dim2D;
  const auto address = fixture->ImageAddress();
  const auto flags = fixture->AddMemory(sample, 0x10f0);
  Value result;
  if (access == ValueOpcode::ImageWrite) {
    const auto texel = fixture->Emit(ValueOpcode::CompositeConstructU32x4,
                                    {Value(1u), Value(2u), Value(3u), Value(4u)});
    fixture->Emit(access, {image, address, texel, Value(true)}, flags);
  } else if (access == ValueOpcode::ImageRead) {
    result = fixture->Emit(access, {image, address, Value(true)}, flags);
  } else if (access == ValueOpcode::ImageQueryDimensions) {
    result = fixture->Emit(access, {image, address}, flags);
  } else if (access == ValueOpcode::ImageAtomicIAdd32) {
    result = fixture->Emit(access, {image, address, Value(1u), Value(true)}, flags);
  } else {
    result = fixture->Emit(access, {image, sampler, address, Value(true)}, flags);
  }
  if (!result.IsEmpty()) {
    if (result.GetType() == Type::U32x4)
      result = fixture->Emit(ValueOpcode::CompositeExtractU32x4, {result, Value(0u)});
    fixture->Emit(ValueOpcode::ReferenceU32, {result});
  }
  return fixture;
}

// The 2D image descriptor, user data and memory of the image heap checks over the indirect image
// fixture. The table at 0x2000 has 48-byte records and holds the descriptor at its byte 16, and the
// selector payload at 0x3000 stays on the GPU. It is built in place and never copied, since a
// runtime keeps pointers into it.
struct HeapTableSetup {
  std::array<uint32_t, 8> descriptor{};
  std::array<uint32_t, 16> data{0x3000u, 4u << 16u, 0x100000u, 0u,
                                0x2000u, 48u << 16u, 3u, 0u};
  LinearTestMemory memory;

  HeapTableSetup() {
    descriptor[0] = 0x20u;
    descriptor[1] = static_cast<uint32_t>(Libs::Graphics::Prospero::BufferFormat::k32_32_32_32Float) << 20u;
    descriptor[2] = 3u | (3u << 14u);
    descriptor[3] = Libs::Graphics::DstSel(4, 5, 6, 7) |
        (static_cast<uint32_t>(Libs::Graphics::Prospero::ImageType::kColor2D) << 28u);
    std::copy_n(data.begin(), 4u, data.begin() + 12u);
    memory.fail_address = 0x3000u;
    std::copy(descriptor.begin(), descriptor.end(), memory.words.begin() + 0x1010u / 4u);
  }
  HeapTableSetup(const HeapTableSetup &) = delete;
  HeapTableSetup &operator=(const HeapTableSetup &) = delete;
};

} // namespace IndirectImageTest

#endif // KYTY_TESTS_INDIRECTIMAGEFIXTURE_H_
