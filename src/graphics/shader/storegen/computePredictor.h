#ifndef GRAPHICS_SHADER_STOREGEN_COMPUTE_PREDICTOR_H
#define GRAPHICS_SHADER_STOREGEN_COMPUTE_PREDICTOR_H

#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/storegen/headerSlots.h"
#include "graphics/shader/storegen/sl00Metadata.h"

#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

// Predicts the module specializations a Motor game will use for a compute program, from the
// program's resource lists, its header slots and its sl00 metadata, with no graphics device and no
// Config. The rules are those of predict_keys.py, which were fitted to one game, so a wrong guess
// only costs a module that is never used. Nothing here can make the runtime take a wrong module,
// because the runtime compares the full specialization before it takes one.
namespace Libs::Graphics::StoreGen {

/// How a descriptor is reached from the shader.
enum class DescriptorKeyKind {
	User,    ///< Consecutive user SGPRs.
	Eud,     ///< Dwords of the extended user data table.
	Literal, ///< A table in the shader's own literal pool, addressed from s_getpc.
};

/// Names a descriptor the way python's descriptor_key does.
struct DescriptorKey {
	DescriptorKeyKind kind = DescriptorKeyKind::User;
	/// User, the first SGPR. Eud, the SGPR that holds the table pointer. Literal, the pc after the
	/// s_getpc.
	uint32_t base = 0;
	/// Eud, the byte offset in the table. Literal, the table's offset from that pc.
	uint32_t offset = 0;
	/// Literal only, the third and fourth descriptor dwords, which are constants.
	uint32_t literal_word2 = 0;
	uint32_t literal_word3 = 0;

	bool operator==(const DescriptorKey&) const = default;
};

/// The key as python prints it, such as `('eud', 12, 96)`, for diagnostics and diffs.
std::string DescriptorKeyText(const DescriptorKey& key);

/// The user data offset a user or EUD key lives at, the slot convention of the header tables.
/// @return nothing for a literal, which has no slot.
std::optional<uint32_t> SlotOffsetOfKey(const DescriptorKey& key, const HeaderSlots& slots);

/// A buffer the translated program uses, with the facts Kyty's own translation gives.
struct BufferFacts {
	DescriptorKey key;
	bool          formatted = false; ///< Accessed through a formatted (typed) instruction.
	bool          scalar    = false; ///< Read through a scalar buffer load.
	bool          written   = false; ///< Written by the program.
};

/// An image the translated program uses.
struct ImageFacts {
	DescriptorKey key;
	uint32_t      dimension = 0; ///< The recompiler's Decoder::ImageDimension value.
	bool          cube      = false;
};

/// The resource lists of a translated program, each in Kyty's own order.
struct ResourceFacts {
	std::vector<BufferFacts> buffers;
	std::vector<ImageFacts>  images;
	/// A sampler that is not a plain user or EUD descriptor, such as a static one made of immediate
	/// dwords, has no key. Only the count matters to the predictor.
	std::vector<std::optional<DescriptorKey>> samplers;
};

/// Which null patterns are tried for the nullable buffers.
enum class NullPolicy {
	None,    ///< Only the all-bound candidate.
	Coupled, ///< All bound, and all null.
	Subsets, ///< Every subset, ordered by size and then by combination.
};

/// The stride, format and swizzle of a bound buffer.
struct BufferFields {
	uint32_t stride  = 0;
	uint32_t format  = 0; ///< A Prospero::BufferFormat value.
	uint32_t swizzle = 0;

	bool operator==(const BufferFields&) const = default;
};

/// The values the rules use, named once.
constexpr uint32_t CONSTANT_BUFFER_STRIDE  = 16;
constexpr uint32_t BYTE_VIEW_STRIDE        = 1;
constexpr uint32_t IDENTITY_BUFFER_SWIZZLE = 4012;
/// Motor's global tables start at this T register, and the buffers there may be left unbound.
constexpr uint32_t GLOBAL_TABLE_FIRST_REGISTER = 29;
/// The name of a buffer that may also be left unbound.
constexpr const char* NULLABLE_NAME = "CameraConstBuf";

/// The sl00 record declared at a slot.
/// Only records with the slot's register and sharp class count, and records whose type is Other or
/// Unreadable are left out, as python's resource_of does.
/// @return nothing unless exactly one record qualifies.
const Sl00Resource* ResourceAtSlot(const SlotRef& slot, std::span<const Sl00Resource> records);

/// The stride, format and swizzle a bound buffer gets, or nothing when the rules do not cover it.
/// @param record the buffer's sl00 record, or null when it has none.
std::optional<BufferFields> BoundBufferFields(const BufferFacts&  buffer,
                                              const Sl00Resource* record);

/// The specialization of a buffer the game binds, with the fields from BoundBufferFields.
ShaderRecompiler::IR::ResourceSpecialization::Buffer
BoundBufferSpecialization(const BufferFields& fields);

/// The specialization of a buffer the game leaves unbound, which reads as a zeroed descriptor.
ShaderRecompiler::IR::ResourceSpecialization::Buffer
NullBufferSpecialization(const BufferFacts& buffer);

/// The specialization of a plain bound image, or nothing when the record is not an image of a known
/// element kind.
std::optional<ShaderRecompiler::IR::ResourceSpecialization::Image>
BoundImageSpecialization(const ImageFacts& image, const Sl00Resource* record);

/// Whether the game may leave a buffer unbound, which is Motor's global tables and the camera
/// constants.
/// @param slot the buffer's slot, or nothing for a buffer with none.
/// @param name the buffer's sl00 name, empty when it has no record.
bool IsNullableBuffer(const std::optional<SlotRef>& slot, const std::string& name);

/// The most nullable buffers whose subsets are all tried. Each extra buffer doubles the candidates,
/// and every candidate is a full translation, so above this the Subsets policy falls back to
/// Coupled. Real programs have at most 3.
constexpr size_t MAX_SUBSET_NULLABLE = 6;

/// The null sets to try for the nullable buffers.
/// @param nullable the indices of the nullable buffers, in buffer order.
/// @return the sets, all-bound first. With the Subsets policy they go by size, then by combination
///         in lexicographic order. More than MAX_SUBSET_NULLABLE nullable buffers give the Coupled
///         sets.
std::vector<std::vector<uint32_t>> NullPatterns(std::span<const uint32_t> nullable,
                                                NullPolicy                policy);

/// One module the program may be compiled to.
struct Candidate {
	ShaderRecompiler::IR::ResourceSpecialization specialization;
	/// The indices of the buffers left unbound, empty for the all-bound candidate.
	std::vector<uint32_t> nulls;
	/// The module's push data start, by the rule that a program with buffers starts at 0 and one
	/// without has none. This is a rule and not a measurement, so compare it with the compiled
	/// module's `bindings.push_data_start_dword`. It is wrong for a program with no buffers that
	/// reads user data or dispatch threads through push data, for a module whose ShaderDataDwords
	/// run past the 32 dwords of push data, and when specialization makes buffers non-live.
	uint32_t push_start = 0;
	/// The id the pipeline cache gives the module, from ShaderModuleId.
	uint64_t id = 0;
};

/// The inputs of one program's prediction. The referenced values must outlive the call.
struct PredictInputs {
	uint64_t code_hash = 0;
	/// The program's user data dword count. It must equal the first static state word, which is the
	/// user SGPR count, because the module id hashes both.
	uint32_t                      user_data_count = 0;
	const std::vector<uint32_t>&  static_state;
	const ResourceFacts&          facts;
	const HeaderSlots&            slots;
	std::span<const Sl00Resource> records;
	NullPolicy                    policy = NullPolicy::Subsets;
};

/// What a prediction made.
struct Prediction {
	/// Empty when any buffer or image is outside the rules. Base first.
	std::vector<Candidate> candidates;
	/// One line per buffer or image the rules do not cover.
	std::vector<std::string> problems;
};

/// Predicts the candidates of a compute program with no GPU-selected calls.
/// The caller skips a program with GPU calls and one whose resources could not be classified.
Prediction PredictCandidates(const PredictInputs& inputs);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_COMPUTE_PREDICTOR_H
