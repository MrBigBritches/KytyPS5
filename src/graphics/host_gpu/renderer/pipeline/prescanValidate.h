#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANVALIDATE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANVALIDATE_H_

#include "graphics/shader/recompiler/BufferFormat.h"
#include "graphics/shader/recompiler/ir/ImageConversion.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

// Checks the specialization that a submit-time snapshot gave before the preparer sends it to the
// shader helper process. The descriptors were read when the game submitted, and one that the game had
// not written yet, or had already recycled, can hold values that make the helper's SPIR-V emitter exit.
// The helper would then crash and be restarted over a snapshot the runtime never meets, because the
// runtime reads the descriptors when it dispatches. Refusing them here avoids that churn.
//
// The fatal checks covered, each through the emitter's own predicate where there is one, are these.
// - ApplyResourceSpecialization exits when a specialization has fewer buffers, images or samplers than
//   its plan, when an indirect child names a root outside the plan, or when any buffer, image or
//   sampler entry names a root outside the specialization.
// - DefineDescriptors exits when DescriptorBindingForImage has no binding for an image, when an image
//   has no mips, or when an image without dynamic mips has more than one. DescriptorBindingForImage
//   also covers ImageType's exits for a sampled atomic image, a signed or unsupported storage image
//   and an atomic image that is not Uint.
// - ImageConversionFormat and StoreTexel exit on a conversion format ImageConversionAccepted refuses.
// - EmitImage fails a depth comparison on a converted image (spirvEmitterImage.cpp around 1029-1033).
//   An image resource is tracked per compare flag, so depth_compare says exactly that.
// - EmitImage fails sampling a k11_11_10UNorm image that is not a direct 2D one (around 922-932).
//   This is conservative, since a sampled image may also be read without a sampler.
// - ResolveFormattedSource exits on a reserved destination select of a formatted buffer. This rule is
//   conservative. BufferResource does not record which components are read, whether the buffer is
//   stored or whether the access is a typed TBUFFER one, so all four selectors are checked even where
//   the emitter would skip some. Refusing too much only loses a prescan build, never a frame.
//
// Known uncovered. These depend on the instructions or on runtime keys, which the plan does not
// record per image, so the helper process and its never-retry list contain them instead.
// - spirvEmitterImage.cpp around 922-932, the rest of the packed UNorm rule. A direct 2D
//   k11_11_10UNorm image must be sampled by a 32 bit sample_lz with a sampler that has no indirect
//   candidates.
// - spirvEmitterImage.cpp around 1034-1053, gather on a Dim1DArray image, and a Dim1D gather through an
//   indirect root or with compare, offset, horizontal or non-zero-level flags. Which images are
//   gathered is not known from the plan.
// - spirvEmitterImage.cpp around 720-727, an indirect image key with no provenance or fewer than two
//   runtime candidates, and around 879 and 884-885, a missing typed sampler variant or a finite
//   sampler key with no provenance.
// - spirvEmitterAnalysis.cpp line 227, a heap image reached with array index 0.
namespace Libs::Graphics {

/// Why a specialization was refused.
enum class DescriptorInvalid : uint8_t {
	/// A formatted buffer with a known format has a reserved destination select.
	ReservedDstSel,
	/// An image has no descriptor binding the emitter can give it, from its class, numeric class,
	/// dimension or atomic use.
	ImageBinding,
	/// An image has no mips, or more than one without dynamic mips.
	ImageMips,
	/// An image's conversion format is one the emitter cannot load or store through.
	ImageFormat,
	/// A converted image is sampled in a way the emitter cannot, with depth comparison, or as packed
	/// UNorm when it is not a direct 2D image.
	ConvertedSampling,
	/// An indirect entry names a root that does not exist.
	IndirectRoot,
	/// The specialization has fewer entries of some kind than its plan.
	EntryCount,
	Count,
};

/// Names a refusal reason for the report, or gives an empty string for Count.
[[nodiscard]] inline const char* DescriptorInvalidName(DescriptorInvalid reason) {
	switch (reason) {
		case DescriptorInvalid::ReservedDstSel: return "reserved_dst_sel";
		case DescriptorInvalid::ImageBinding: return "image_binding";
		case DescriptorInvalid::ImageMips: return "image_mips";
		case DescriptorInvalid::ImageFormat: return "image_format";
		case DescriptorInvalid::ConvertedSampling: return "converted_sampling";
		case DescriptorInvalid::IndirectRoot: return "indirect_root";
		case DescriptorInvalid::EntryCount: return "entry_count";
		case DescriptorInvalid::Count: break;
	}
	return "";
}

namespace PrescanValidateDetail {

using namespace ShaderRecompiler;

/// How many destination selectors a buffer descriptor's swizzle holds, one per output component.
inline constexpr uint32_t DST_SEL_COMPONENTS = 4;

/// Finds the plan entry that specialization entry `index` is built from, as ApplyResourceSpecialization
/// does. An entry past the plan is an indirect child copied from its root, which must be in the plan.
/// An entry inside the plan is its own, and a root it names must exist in the specialization.
/// @return the plan index, or nullopt when a root is out of range.
template <class Spec>
[[nodiscard]] std::optional<size_t> PlanIndex(size_t plan_size, std::span<const Spec> spec, size_t index,
                                              uint32_t no_root) {
	const auto root = spec[index].indirect_root;
	if (index < plan_size) {
		if (root != no_root && root >= spec.size()) {
			return std::nullopt;
		}
		return index;
	}
	if (root >= plan_size) {
		return std::nullopt;
	}
	return root;
}

/// Checks every specialization entry of one kind. Each must resolve to a plan entry, and `check`
/// then judges the pair.
/// @return the first refusal, or nullopt.
template <class Plan, class Spec, class Check>
[[nodiscard]] std::optional<DescriptorInvalid> FindInvalidEntry(std::span<const Plan> plan,
                                                                std::span<const Spec> spec, uint32_t no_root,
                                                                Check&& check) {
	for (size_t index = 0; index < spec.size(); ++index) {
		const auto plan_index = PlanIndex(plan.size(), spec, index, no_root);
		if (!plan_index.has_value()) {
			return DescriptorInvalid::IndirectRoot;
		}
		if (const auto bad = check(plan[*plan_index], spec[index])) {
			return bad;
		}
	}
	return std::nullopt;
}

/// Finds a reserved destination select in a formatted buffer, which is the check behind "buffer
/// descriptor has reserved dst_sel". A buffer not read through a format is never judged by its format
/// fields, and neither is one whose format has an unknown component type, since the emitter reads and
/// writes that one raw.
[[nodiscard]] inline std::optional<DescriptorInvalid>
CheckBuffer(const IR::BufferResource& plan, const IR::ResourceSpecialization::Buffer& spec) {
	if (!plan.formatted) {
		return std::nullopt;
	}
	const auto info = Format::GetFormatInfo(spec.descriptor_format);
	if (info.type == Format::ComponentType::Unknown) {
		return std::nullopt;
	}
	for (uint32_t component = 0; component < DST_SEL_COMPONENTS; ++component) {
		const auto selector = GetDstSel(spec.descriptor_swizzle, component);
		if (Format::ResolveFormattedSource(info, selector).kind == Format::FormattedSourceKind::Invalid) {
			return DescriptorInvalid::ReservedDstSel;
		}
	}
	return std::nullopt;
}

/// Builds the image the emitter would see, the plan's image with the specialization's values laid
/// over it. Only the fields the checks below read are set, so the plan's candidate list is never
/// copied. DescriptorBindingForImage reads the class, numeric class, dimension, depth compare and
/// atomic flags, and ImageConversionAccepted reads the conversion format with the class, read,
/// written and atomic flags. A field either predicate starts to read must be added here.
[[nodiscard]] inline IR::ImageResource ImageForChecks(const IR::ImageResource&                plan,
                                                      const IR::ResourceSpecialization::Image& spec) {
	IR::ImageResource image;
	image.resource_class    = plan.resource_class;
	image.mip_mode          = plan.mip_mode;
	image.read              = plan.read;
	image.written           = plan.written;
	image.atomic            = plan.atomic;
	image.atomic64          = plan.atomic64;
	image.depth_compare     = plan.depth_compare;
	image.numeric_class     = spec.numeric_class;
	image.dimension         = spec.dimension;
	image.mip_count         = spec.mip_count;
	image.conversion_format = spec.conversion_format;
	image.indirect_root     = spec.indirect_root;
	return image;
}

/// Finds a converted image sampled in a way the emitter fails. Depth comparison never works on a
/// converted image, and packed UNorm sampling needs a direct 2D image.
[[nodiscard]] inline std::optional<DescriptorInvalid> CheckConvertedSampling(const IR::ImageResource& image) {
	if (image.depth_compare && image.conversion_format != Prospero::BufferFormat::kInvalid) {
		return DescriptorInvalid::ConvertedSampling;
	}
	if (image.resource_class == IR::ImageResourceClass::Sampled &&
	    image.conversion_format == Prospero::BufferFormat::k11_11_10UNorm &&
	    (image.dimension != Decoder::ImageDimension::Dim2D ||
	     image.indirect_root != IR::ImageResource::NoIndirectImage)) {
		return DescriptorInvalid::ConvertedSampling;
	}
	return std::nullopt;
}

/// Judges the image the emitter would see through the emitter's own predicates.
[[nodiscard]] inline std::optional<DescriptorInvalid>
CheckImage(const IR::ImageResource& plan, const IR::ResourceSpecialization::Image& spec) {
	const auto image = ImageForChecks(plan, spec);
	if (image.mip_count == 0u || (image.mip_mode != IR::ImageMipMode::Dynamic && image.mip_count != 1u)) {
		return DescriptorInvalid::ImageMips;
	}
	if (!IR::DescriptorBindingForImage(image).has_value()) {
		return DescriptorInvalid::ImageBinding;
	}
	// A written image may be stored through, which accepts fewer formats than a load.
	const auto use = image.written ? IR::ImageConversionUse::Store : IR::ImageConversionUse::Load;
	if (!IR::ImageConversionAccepted(image, use)) {
		return DescriptorInvalid::ImageFormat;
	}
	return CheckConvertedSampling(image);
}

/// Samplers carry no descriptor values the emitter refuses, so only their roots are checked.
[[nodiscard]] inline std::optional<DescriptorInvalid> CheckSampler(const IR::SamplerResource& /*plan*/,
                                                                   const IR::ResourceSpecialization::Sampler& /*spec*/) {
	return std::nullopt;
}

} // namespace PrescanValidateDetail

/// Finds the first entry of the specialization that the emitter would refuse, buffers first, then
/// images, then samplers, or nothing when every one passes. Indirect children past the end of the
/// plan are checked against their root's plan entry.
/// @param buffers         the buffers of the plan the specialization was materialized from.
/// @param images          the images of the same plan.
/// @param samplers        the samplers of the same plan.
/// @param specialization  the specialization the snapshot gave.
/// @return the reason the first refused entry fails, or nullopt.
[[nodiscard]] inline std::optional<DescriptorInvalid>
FindInvalidDescriptor(std::span<const ShaderRecompiler::IR::BufferResource>  buffers,
                      std::span<const ShaderRecompiler::IR::ImageResource>   images,
                      std::span<const ShaderRecompiler::IR::SamplerResource> samplers,
                      const ShaderRecompiler::IR::ResourceSpecialization&    specialization) {
	using namespace ShaderRecompiler;
	using Spec = IR::ResourceSpecialization;
	namespace Detail = PrescanValidateDetail;
	if (specialization.buffers.size() < buffers.size() || specialization.images.size() < images.size() ||
	    specialization.samplers.size() < samplers.size()) {
		return DescriptorInvalid::EntryCount;
	}
	if (const auto bad = Detail::FindInvalidEntry(buffers, std::span<const Spec::Buffer>(specialization.buffers),
	                                              IR::BufferResource::NoIndirectBuffer, Detail::CheckBuffer)) {
		return bad;
	}
	if (const auto bad = Detail::FindInvalidEntry(images, std::span<const Spec::Image>(specialization.images),
	                                              IR::ImageResource::NoIndirectImage, Detail::CheckImage)) {
		return bad;
	}
	return Detail::FindInvalidEntry(samplers, std::span<const Spec::Sampler>(specialization.samplers),
	                                IR::SamplerResource::NoIndirectSampler, Detail::CheckSampler);
}

/// Finds the first refused entry of the specialization, taking the buffers, images and samplers from
/// the plan it was materialized from.
[[nodiscard]] inline std::optional<DescriptorInvalid>
FindInvalidDescriptor(const ShaderRecompiler::IR::ResourcePlan&           plan,
                      const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	return FindInvalidDescriptor(plan.info.buffers, plan.info.images, plan.info.samplers, specialization);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANVALIDATE_H_
