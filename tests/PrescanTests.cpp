// Tests the prescan's variant tracker, its descriptor validation and the list of retained store
// records. All are headers over plain types, so no guest memory, translation or device is needed.
// Each check that proves a behaviour has a baseline that differs only in that behaviour, as the
// comments say.
#include "graphics/host_gpu/renderer/pipeline/prescanValidate.h"
#include "graphics/host_gpu/renderer/pipeline/prescanVariants.h"
#include "graphics/host_gpu/renderer/pipeline/retainedRecords.h"

#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using IntTracker = VariantTracker<int>;

// How far a probe search runs before giving up, so a tracker that never probes fails the check
// instead of looping for ever.
constexpr uint32_t PROBE_SEARCH_LIMIT = 2 * PRESCAN_PROBE_INTERVAL;

// The swizzle that reads x, y, z and w from components 0 to 3.
constexpr uint32_t IDENTITY_DST_SEL = DstSel(4, 5, 6, 7);

// Two snapshots of one program with different specializations make two modules, and two with the
// same specialization make one. The second case is the baseline of the first, since the program, the
// gate and the number of snapshots are the same, and only the specialization differs.
void TestVariantsFollowSpecialization() {
	IntTracker different;
	CHECK(different.TryPost(7) == IntTracker::Gate::Post);
	CHECK(different.NoteSnapshot(7, 100) == IntTracker::Outcome::New);
	CHECK(different.TryPost(7) == IntTracker::Gate::Post);
	CHECK(different.NoteSnapshot(7, 200) == IntTracker::Outcome::New);

	IntTracker same;
	CHECK(same.TryPost(7) == IntTracker::Gate::Post);
	CHECK(same.NoteSnapshot(7, 100) == IntTracker::Outcome::New);
	CHECK(same.TryPost(7) == IntTracker::Gate::Post);
	CHECK(same.NoteSnapshot(7, 100) == IntTracker::Outcome::Known);

	// A program with a snapshot in flight is not posted again, and another program is unaffected.
	IntTracker busy;
	CHECK(busy.TryPost(1) == IntTracker::Gate::Post);
	CHECK(busy.TryPost(1) == IntTracker::Gate::Busy);
	CHECK(busy.TryPost(2) == IntTracker::Gate::Post);
	busy.NoteDropped(1);
	CHECK(busy.TryPost(1) == IntTracker::Gate::Post);
}

// A program stops at the cap and is counted once, and a program that keeps giving nothing new
// settles, is not posted, and is probed once in a while.
void TestVariantCapAndSettling() {
	IntTracker capped;
	for (uint32_t index = 0; index < PRESCAN_VARIANT_CAP; ++index) {
		CHECK(capped.TryPost(1) == IntTracker::Gate::Post);
		CHECK(capped.NoteSnapshot(1, static_cast<int>(index)) == IntTracker::Outcome::New);
	}
	CHECK(capped.CappedUnits() == 1);
	CHECK(capped.TryPost(1) == IntTracker::Gate::Capped);

	IntTracker settled;
	CHECK(settled.TryPost(1) == IntTracker::Gate::Post);
	CHECK(settled.NoteSnapshot(1, 5) == IntTracker::Outcome::New);
	for (uint32_t index = 0; index < PRESCAN_STALE_LIMIT; ++index) {
		CHECK(settled.TryPost(1) == IntTracker::Gate::Post);
		CHECK(settled.NoteSnapshot(1, 5) == IntTracker::Outcome::Known);
	}
	CHECK(settled.SettledUnits() == 1);
	uint32_t skipped = 0;
	while (skipped < PROBE_SEARCH_LIMIT && settled.TryPost(1) == IntTracker::Gate::Settled) {
		++skipped;
	}
	CHECK(skipped == PRESCAN_PROBE_INTERVAL - 1);
	// The probe found a new specialization, which unsettles the program.
	CHECK(settled.NoteSnapshot(1, 6) == IntTracker::Outcome::New);
	CHECK(settled.SettledUnits() == 0 && settled.TryPost(1) == IntTracker::Gate::Post);
}

// Posts snapshots of `unit` that all give the same specialization until the program settles.
void SettleUnit(IntTracker& tracker, uint64_t unit) {
	for (uint32_t index = 0; index <= PRESCAN_STALE_LIMIT; ++index) {
		CHECK(tracker.TryPost(unit) == IntTracker::Gate::Post);
		(void)tracker.NoteSnapshot(unit, 1);
	}
}

// Counts how many dispatches of a settled `unit` are held back before one is posted as a probe. The
// probe is then in flight. Gives PROBE_SEARCH_LIMIT when no probe comes.
uint32_t CountUntilProbe(IntTracker& tracker, uint64_t unit) {
	uint32_t skipped = 0;
	while (skipped < PROBE_SEARCH_LIMIT && tracker.TryPost(unit) == IntTracker::Gate::Settled) {
		++skipped;
	}
	return skipped;
}

// A dispatch that the scanner's cache already let through as its probe posts a settled program at
// once. A tracker that never settles would also post it, so the baseline is the same settled program
// asked without the probe flag, which is held back.
void TestCacheProbePostsSettledProgram() {
	IntTracker tracker;
	SettleUnit(tracker, 3);
	CHECK(tracker.SettledUnits() == 1);
	CHECK(tracker.TryPost(3, false) == IntTracker::Gate::Settled);
	CHECK(tracker.TryPost(3, true) == IntTracker::Gate::Post);
}

// A probe that finds nothing new starts a fresh interval, so the next probe comes after as many held
// dispatches again. A tracker that never probed would fail the first count, which is the baseline, and
// one that kept counting from the old interval would post at once the second time.
void TestProbeResetsInterval() {
	IntTracker tracker;
	SettleUnit(tracker, 5);
	CHECK(CountUntilProbe(tracker, 5) == PRESCAN_PROBE_INTERVAL - 1);
	CHECK(tracker.NoteSnapshot(5, 1) == IntTracker::Outcome::Known);
	CHECK(CountUntilProbe(tracker, 5) == PRESCAN_PROBE_INTERVAL - 1);
}

// Snapshots whose specialization could not be built count towards settling, so a program that keeps
// giving bad descriptors stops being posted. The baseline is the same number of snapshots dropped
// without a specialization, which leaves the program postable, so the settling is the reject and not
// the count of posts.
void TestRejectsSettle() {
	IntTracker rejected;
	IntTracker dropped;
	for (uint32_t index = 0; index < PRESCAN_STALE_LIMIT; ++index) {
		CHECK(rejected.TryPost(4) == IntTracker::Gate::Post);
		rejected.NoteRejected(4);
		CHECK(dropped.TryPost(4) == IntTracker::Gate::Post);
		dropped.NoteDropped(4);
	}
	CHECK(rejected.TryPost(4) == IntTracker::Gate::Settled);
	CHECK(dropped.TryPost(4) == IntTracker::Gate::Post);
}

// A retired program is never posted again but is not counted as capped. The baseline is a program
// that reached the cap, which is counted, so the count tells the two apart.
void TestRetireCapsWithoutCounting() {
	IntTracker retired;
	CHECK(retired.TryPost(9) == IntTracker::Gate::Post);
	retired.Retire(9);
	CHECK(retired.TryPost(9) == IntTracker::Gate::Capped);
	CHECK(retired.CappedUnits() == 0);

	IntTracker full;
	for (uint32_t index = 0; index < PRESCAN_VARIANT_CAP; ++index) {
		CHECK(full.TryPost(9) == IntTracker::Gate::Post);
		(void)full.NoteSnapshot(9, static_cast<int>(index));
	}
	CHECK(full.TryPost(9) == IntTracker::Gate::Capped);
	CHECK(full.CappedUnits() == 1);
}

// One program dispatched with two user-data counts gets two trackers, so neither is held back by the
// other's snapshot in flight. A key that changed on every call would pass that too, so the baseline
// is the same program and count twice, which shares one tracker and is held back. The code hash and
// static key each still separate units, so a key of the count alone fails as well.
void TestUnitKeySeparatesUserDataCount() {
	constexpr uint64_t          CODE_HASH  = 0x5EEDC0DE12345678ull;
	constexpr uint64_t          OTHER_CODE = CODE_HASH + 1u;
	const std::vector<uint32_t> static_state {64u, 1u, 1u, 0x20u};
	const std::vector<uint32_t> other_state {32u, 1u, 1u, 0x20u};
	const uint64_t              two_sgprs  = PrescanUnitKey(CODE_HASH, 2, static_state);
	const uint64_t              four_sgprs = PrescanUnitKey(CODE_HASH, 4, static_state);

	IntTracker separate;
	CHECK(separate.TryPost(two_sgprs) == IntTracker::Gate::Post);
	CHECK(separate.TryPost(four_sgprs) == IntTracker::Gate::Post);
	CHECK(separate.Units() == 2);

	IntTracker shared;
	CHECK(shared.TryPost(two_sgprs) == IntTracker::Gate::Post);
	CHECK(shared.TryPost(PrescanUnitKey(CODE_HASH, 2, static_state)) == IntTracker::Gate::Busy);
	CHECK(shared.Units() == 1);

	CHECK(PrescanUnitKey(OTHER_CODE, 2, static_state) != two_sgprs);
	CHECK(PrescanUnitKey(CODE_HASH, 2, other_state) != two_sgprs);
}

// A specialization with a formatted buffer that holds a reserved destination select is refused and
// named, and the same buffer with a valid descriptor passes. The valid case differs from the refused
// ones only in the swizzle or the format, so a validator that refused everything, or nothing, fails
// one of them.
void TestInvalidDescriptorsRefused() {
	using namespace Libs::Graphics::ShaderRecompiler;
	std::vector<IR::BufferResource> buffers(1);
	buffers[0].formatted = true;
	std::vector<IR::ImageResource> images;
	IR::ResourceSpecialization spec;
	spec.buffers.resize(1);
	spec.buffers[0].descriptor_format  = Prospero::BufferFormat::k32_32_32_32Float;
	spec.buffers[0].descriptor_swizzle = IDENTITY_DST_SEL;
	CHECK(!FindInvalidDescriptor(buffers, images, {},spec).has_value());

	// Selector 2 is reserved.
	auto reserved                          = spec;
	reserved.buffers[0].descriptor_swizzle = DstSel(2, 5, 6, 7);
	CHECK(FindInvalidDescriptor(buffers, images, {},reserved) == DescriptorInvalid::ReservedDstSel);
	// The reserved selector in the last component is found too, not only the first.
	auto reserved_last                          = spec;
	reserved_last.buffers[0].descriptor_swizzle = DstSel(4, 5, 6, 3);
	CHECK(FindInvalidDescriptor(buffers, images, {},reserved_last) == DescriptorInvalid::ReservedDstSel);
	// Zero and one selectors are valid, so a swizzle that uses them passes.
	auto constants                          = spec;
	constants.buffers[0].descriptor_swizzle = DstSel(4, 0, 1, 7);
	CHECK(!FindInvalidDescriptor(buffers, images, {},constants).has_value());
	// A descriptor that was never written has no known format, which the emitter reads raw.
	auto unwritten                         = spec;
	unwritten.buffers[0].descriptor_format = Prospero::BufferFormat::kInvalid;
	CHECK(!FindInvalidDescriptor(buffers, images, {},unwritten).has_value());
	// A buffer that is not read through a format is not judged by its format fields.
	buffers[0].formatted = false;
	CHECK(!FindInvalidDescriptor(buffers, images, {},reserved).has_value());
	buffers[0].formatted = true;

	// An image with no mips is refused, and with its mips it passes.
	images.resize(1);
	images[0].resource_class = IR::ImageResourceClass::Sampled;
	images[0].numeric_class  = Prospero::TextureNumericClass::Float;
	images[0].dimension      = Decoder::ImageDimension::Dim2D;
	spec.images.resize(1);
	spec.images[0].numeric_class = Prospero::TextureNumericClass::Float;
	spec.images[0].dimension     = Decoder::ImageDimension::Dim2D;
	spec.images[0].mip_count     = 1;
	CHECK(!FindInvalidDescriptor(buffers, images, {},spec).has_value());
	auto no_mips                = spec;
	no_mips.images[0].mip_count = 0;
	CHECK(FindInvalidDescriptor(buffers, images, {},no_mips) == DescriptorInvalid::ImageMips);
	// A numeric class the descriptor turned unsupported is refused. One the plan already had is
	// refused too, since the emitter judges the whole image.
	auto unsupported                    = spec;
	unsupported.images[0].numeric_class = Prospero::TextureNumericClass::Unsupported;
	CHECK(FindInvalidDescriptor(buffers, images, {},unsupported) == DescriptorInvalid::ImageBinding);
	images[0].numeric_class = Prospero::TextureNumericClass::Unsupported;
	CHECK(FindInvalidDescriptor(buffers, images, {},unsupported) == DescriptorInvalid::ImageBinding);
}

namespace IR = Libs::Graphics::ShaderRecompiler::IR;
using ImageDimension = Libs::Graphics::ShaderRecompiler::Decoder::ImageDimension;

// One formatted buffer, a plan and a specialization, that every validation rule passes.
struct BufferCase {
	std::vector<IR::BufferResource> plan = std::vector<IR::BufferResource>(1);
	IR::ResourceSpecialization      spec;

	BufferCase() {
		plan[0].formatted = true;
		spec.buffers.resize(1);
		spec.buffers[0].descriptor_format  = Prospero::BufferFormat::k32_32_32_32Float;
		spec.buffers[0].descriptor_swizzle = IDENTITY_DST_SEL;
	}

	/// The validator's answer for the case, with no images.
	[[nodiscard]] std::optional<DescriptorInvalid> Validate() const {
		return FindInvalidDescriptor(plan, std::vector<IR::ImageResource> {}, {}, spec);
	}
};

// One sampled 2D float image with no conversion format, a plan and a specialization equal to it, that
// every validation rule passes. Materialization leaves the conversion format invalid for any format
// the host does not remap, so that is the common case.
struct ImageCase {
	std::vector<IR::ImageResource> plan = std::vector<IR::ImageResource>(1);
	IR::ResourceSpecialization     spec;

	ImageCase() {
		plan[0].resource_class = IR::ImageResourceClass::Sampled;
		plan[0].numeric_class  = Prospero::TextureNumericClass::Float;
		plan[0].dimension      = ImageDimension::Dim2D;
		spec.images.resize(1);
		spec.images[0].numeric_class = plan[0].numeric_class;
		spec.images[0].dimension     = plan[0].dimension;
		spec.images[0].mip_count     = 1;
	}

	/// Makes the image a storage image of `numeric_class`, atomic when `atomic` is set.
	void MakeStorage(Prospero::TextureNumericClass numeric_class, bool atomic = false) {
		plan[0].resource_class       = IR::ImageResourceClass::Storage;
		plan[0].atomic               = atomic;
		plan[0].numeric_class        = numeric_class;
		spec.images[0].numeric_class = numeric_class;
	}

	/// The validator's answer for the case, with no buffers.
	[[nodiscard]] std::optional<DescriptorInvalid> Validate() const {
		return FindInvalidDescriptor(std::vector<IR::BufferResource> {}, plan, {}, spec);
	}
};

// A reserved destination select in any component is refused for that reason. A validator that
// refused every buffer would pass, so the baseline is the same buffer with the identity swizzle. One
// that refused every selector below 4 would pass too, so the zero and one selectors must pass.
void TestReservedDstSelRefused() {
	const BufferCase valid;
	CHECK(!valid.Validate().has_value());

	BufferCase first;
	first.spec.buffers[0].descriptor_swizzle = DstSel(2, 5, 6, 7);
	CHECK(first.Validate() == DescriptorInvalid::ReservedDstSel);
	BufferCase last;
	last.spec.buffers[0].descriptor_swizzle = DstSel(4, 5, 6, 3);
	CHECK(last.Validate() == DescriptorInvalid::ReservedDstSel);

	BufferCase constants;
	constants.spec.buffers[0].descriptor_swizzle = DstSel(0, 1, 0, 1);
	CHECK(!constants.Validate().has_value());
}

// A formatted buffer whose format has an unknown component type passes even with reserved selectors,
// since the emitter reads and writes it raw and never resolves them. A validator that passed every
// buffer would pass that too, so the baseline is the same reserved selectors over a known format,
// which is refused.
void TestUnknownBufferFormatTakesRawPath() {
	BufferCase known;
	known.spec.buffers[0].descriptor_swizzle = DstSel(2, 3, 2, 3);
	CHECK(known.Validate() == DescriptorInvalid::ReservedDstSel);

	BufferCase unknown                        = known;
	unknown.spec.buffers[0].descriptor_format = Prospero::BufferFormat::kInvalid;
	CHECK(!unknown.Validate().has_value());
}

// An image with no mips is refused. The baseline is the same image with one mip.
void TestImageMipsRefused() {
	const ImageCase valid;
	CHECK(!valid.Validate().has_value());

	ImageCase no_mips;
	no_mips.spec.images[0].mip_count = 0;
	CHECK(no_mips.Validate() == DescriptorInvalid::ImageMips);
}

// An image whose shader does not pick its mip may have only one, so two are refused. A validator that
// refused every count above one would pass that too, so the baseline is the same two mips on an image
// with dynamic mips, which passes.
void TestFixedMipModeNeedsOneMip() {
	ImageCase fixed;
	fixed.spec.images[0].mip_count = 2;
	CHECK(fixed.Validate() == DescriptorInvalid::ImageMips);

	ImageCase dynamic        = fixed;
	dynamic.plan[0].mip_mode = IR::ImageMipMode::Dynamic;
	CHECK(!dynamic.Validate().has_value());
}

// An unsupported numeric class is refused, whether the descriptor turned it so or the plan already had
// it, and a storage image turned signed is refused. A sampled image turned signed passes, since only
// storage images lack a signed binding, which is the baseline for the storage case.
void TestImageNumericClassRefused() {
	const ImageCase valid;
	CHECK(!valid.Validate().has_value());

	ImageCase unsupported;
	unsupported.spec.images[0].numeric_class = Prospero::TextureNumericClass::Unsupported;
	CHECK(unsupported.Validate() == DescriptorInvalid::ImageBinding);
	ImageCase already             = unsupported;
	already.plan[0].numeric_class = Prospero::TextureNumericClass::Unsupported;
	CHECK(already.Validate() == DescriptorInvalid::ImageBinding);

	ImageCase storage_sint;
	storage_sint.MakeStorage(Prospero::TextureNumericClass::Float);
	storage_sint.spec.images[0].numeric_class = Prospero::TextureNumericClass::Sint;
	CHECK(storage_sint.Validate() == DescriptorInvalid::ImageBinding);
	ImageCase sampled_sint;
	sampled_sint.spec.images[0].numeric_class = Prospero::TextureNumericClass::Sint;
	CHECK(!sampled_sint.Validate().has_value());
}

// An unknown dimension is refused. A dimension that changed to another known one passes, so the rule
// is the unknown value and not any change.
void TestImageDimensionRefused() {
	const ImageCase valid;
	CHECK(!valid.Validate().has_value());

	ImageCase unknown;
	unknown.spec.images[0].dimension = ImageDimension::Unknown;
	CHECK(unknown.Validate() == DescriptorInvalid::ImageBinding);
	ImageCase changed;
	changed.spec.images[0].dimension = ImageDimension::Dim2DArray;
	CHECK(!changed.Validate().has_value());
}

// A storage image whose descriptor turned it into a multisampled array has no binding. The baseline is
// the same dimension on a sampled image, which has one, so the rule is the storage use.
void TestStorageMsaaArrayRefused() {
	ImageCase storage;
	storage.MakeStorage(Prospero::TextureNumericClass::Float);
	storage.spec.images[0].dimension = ImageDimension::Dim2DMsaaArray;
	CHECK(storage.Validate() == DescriptorInvalid::ImageBinding);

	ImageCase sampled;
	sampled.spec.images[0].dimension = ImageDimension::Dim2DMsaaArray;
	CHECK(!sampled.Validate().has_value());
}

// An atomic image whose descriptor turned it multisampled has no binding. The baseline is the same
// atomic image in 2D, which has one, so the rule is the dimension.
void TestAtomicMsaaRefused() {
	ImageCase msaa;
	msaa.MakeStorage(Prospero::TextureNumericClass::Uint, true);
	msaa.spec.images[0].dimension = ImageDimension::Dim2DMsaa;
	CHECK(msaa.Validate() == DescriptorInvalid::ImageBinding);

	ImageCase flat;
	flat.MakeStorage(Prospero::TextureNumericClass::Uint, true);
	CHECK(!flat.Validate().has_value());
}

// An atomic image whose descriptor gives it a float class is refused, since image atomics need Uint.
// The baseline is the same atomic image with Uint, and a non-atomic float storage image passes too,
// so the rule is the atomic use.
void TestAtomicFloatRefused() {
	ImageCase atomic_float;
	atomic_float.MakeStorage(Prospero::TextureNumericClass::Float, true);
	CHECK(atomic_float.Validate() == DescriptorInvalid::ImageBinding);

	ImageCase atomic_uint;
	atomic_uint.MakeStorage(Prospero::TextureNumericClass::Uint, true);
	CHECK(!atomic_uint.Validate().has_value());
	ImageCase plain_float;
	plain_float.MakeStorage(Prospero::TextureNumericClass::Float);
	CHECK(!plain_float.Validate().has_value());
}

// The conversion formats materialization really produces are the ones the host remaps, such as
// k8_8UScaled, which a load accepts. A format the host does not remap is refused, both k32Float,
// which also has the wrong layout, and k2_10_10_10UNorm, which has the right layout and fails only for
// not being remapped. A written image may be stored through, and stores refuse Uscaled, so the same
// k8_8UScaled is refused on a written storage image while a packed Uint format passes there.
void TestImageFormatRefused() {
	constexpr auto REMAP_ONLY = Prospero::BufferFormat::k2_10_10_10UNorm;
	constexpr auto REMAP_INFO = Libs::Graphics::ShaderRecompiler::Format::GetFormatInfo(REMAP_ONLY);
	static_assert(REMAP_INFO.packed_bitfield && REMAP_INFO.byte_size == sizeof(uint32_t),
	              "the remap-only case needs a layout the load rule accepts");
	const ImageCase valid;
	CHECK(!valid.Validate().has_value());

	ImageCase uscaled;
	uscaled.spec.images[0].conversion_format = Prospero::BufferFormat::k8_8UScaled;
	CHECK(!uscaled.Validate().has_value());
	ImageCase float_format;
	float_format.spec.images[0].conversion_format = Prospero::BufferFormat::k32Float;
	CHECK(float_format.Validate() == DescriptorInvalid::ImageFormat);
	ImageCase not_remapped;
	not_remapped.spec.images[0].conversion_format = REMAP_ONLY;
	CHECK(not_remapped.Validate() == DescriptorInvalid::ImageFormat);

	ImageCase stored = uscaled;
	stored.MakeStorage(Prospero::TextureNumericClass::Uint);
	CHECK(!stored.Validate().has_value());
	stored.plan[0].written = true;
	CHECK(stored.Validate() == DescriptorInvalid::ImageFormat);
	stored.spec.images[0].conversion_format = Prospero::BufferFormat::k11_11_10UInt;
	CHECK(!stored.Validate().has_value());
}

// Indirect children past the end of the plan are judged against their root's plan entry, and a child
// whose root is outside the plan is refused. The baseline is the same child with valid values, which
// passes, so a refusal is the child's own descriptor and not its presence.
void TestIndirectChildrenChecked() {
	ImageCase child;
	child.spec.images.push_back(child.spec.images[0]);
	child.spec.images[1].indirect_root = 0;
	CHECK(!child.Validate().has_value());
	ImageCase no_mips                = child;
	no_mips.spec.images[1].mip_count = 0;
	CHECK(no_mips.Validate() == DescriptorInvalid::ImageMips);
	ImageCase orphan                    = child;
	orphan.spec.images[1].indirect_root = 1;
	CHECK(orphan.Validate() == DescriptorInvalid::IndirectRoot);
	// A plan entry may name a root, but it must exist in the specialization.
	ImageCase stray                    = child;
	stray.spec.images[0].indirect_root = 2;
	CHECK(stray.Validate() == DescriptorInvalid::IndirectRoot);

	BufferCase buffer_child;
	buffer_child.spec.buffers.push_back(buffer_child.spec.buffers[0]);
	buffer_child.spec.buffers[1].indirect_root = 0;
	CHECK(!buffer_child.Validate().has_value());
	buffer_child.spec.buffers[1].descriptor_swizzle = DstSel(2, 5, 6, 7);
	CHECK(buffer_child.Validate() == DescriptorInvalid::ReservedDstSel);
}

// When a buffer and an image are both bad the buffer's reason is given, and the image alone gives its
// own, so the order is buffers first and the image is still checked.
void TestBuffersCheckedBeforeImages() {
	BufferCase buffer;
	buffer.spec.buffers[0].descriptor_swizzle = DstSel(2, 5, 6, 7);
	ImageCase image;
	image.spec.images[0].mip_count = 0;

	IR::ResourceSpecialization both;
	both.buffers = buffer.spec.buffers;
	both.images  = image.spec.images;
	CHECK(FindInvalidDescriptor(buffer.plan, image.plan, {}, both) == DescriptorInvalid::ReservedDstSel);

	const BufferCase valid_buffer;
	both.buffers = valid_buffer.spec.buffers;
	CHECK(FindInvalidDescriptor(valid_buffer.plan, image.plan, {}, both) == DescriptorInvalid::ImageMips);
}

// An image sampled with depth comparison that the descriptor gave a conversion format is refused. The
// baseline is the same compared image with no conversion, which passes, and the same conversion on an
// image without comparison passes too, so the rule is the pair and neither half alone.
void TestDepthCompareConvertedRefused() {
	ImageCase compared;
	compared.plan[0].depth_compare = true;
	CHECK(!compared.Validate().has_value());

	ImageCase converted                        = compared;
	converted.spec.images[0].conversion_format = Prospero::BufferFormat::k8_8UScaled;
	CHECK(converted.Validate() == DescriptorInvalid::ConvertedSampling);

	ImageCase plain             = converted;
	plain.plan[0].depth_compare = false;
	CHECK(!plain.Validate().has_value());
}

// A sampled image converted from k11_11_10UNorm passes only as a direct 2D image. The baseline is that
// direct 2D image, which passes. A 2D array and an indirect image are refused, and a storage image with
// the same format and array dimension passes, so the rule is the sampling.
void TestPackedUnormSamplingNeedsDirect2D() {
	ImageCase direct;
	direct.spec.images[0].conversion_format = Prospero::BufferFormat::k11_11_10UNorm;
	CHECK(!direct.Validate().has_value());

	ImageCase array                = direct;
	array.spec.images[0].dimension = ImageDimension::Dim2DArray;
	CHECK(array.Validate() == DescriptorInvalid::ConvertedSampling);
	ImageCase indirect                    = direct;
	indirect.spec.images[0].indirect_root = 0;
	CHECK(indirect.Validate() == DescriptorInvalid::ConvertedSampling);

	ImageCase storage = array;
	storage.MakeStorage(Prospero::TextureNumericClass::Float);
	CHECK(!storage.Validate().has_value());
}

// Sampler entries follow the same root rules as buffers and images. The baseline is a child sampler
// whose root is in the plan, which passes, so a refusal is the root and not the child.
void TestSamplerRootsChecked() {
	const std::vector<IR::SamplerResource> plan(1);
	IR::ResourceSpecialization             spec;
	spec.samplers.resize(2);
	spec.samplers[1].indirect_root = 0;
	CHECK(!FindInvalidDescriptor({}, {}, plan, spec).has_value());

	auto orphan                      = spec;
	orphan.samplers[1].indirect_root = 1;
	CHECK(FindInvalidDescriptor({}, {}, plan, orphan) == DescriptorInvalid::IndirectRoot);
	auto stray                      = spec;
	stray.samplers[0].indirect_root = 2;
	CHECK(FindInvalidDescriptor({}, {}, plan, stray) == DescriptorInvalid::IndirectRoot);
}

// A specialization with fewer buffers, images or samplers than its plan is refused. The baseline for
// each is the same plan with a specialization of matching length, which passes.
void TestShortSpecializationRefused() {
	const BufferCase buffers;
	CHECK(!buffers.Validate().has_value());
	BufferCase no_buffers;
	no_buffers.spec.buffers.clear();
	CHECK(no_buffers.Validate() == DescriptorInvalid::EntryCount);

	const ImageCase images;
	CHECK(!images.Validate().has_value());
	ImageCase no_images;
	no_images.spec.images.clear();
	CHECK(no_images.Validate() == DescriptorInvalid::EntryCount);

	const std::vector<IR::SamplerResource> samplers(1);
	IR::ResourceSpecialization             spec;
	spec.samplers.resize(1);
	CHECK(!FindInvalidDescriptor({}, {}, samplers, spec).has_value());
	spec.samplers.clear();
	CHECK(FindInvalidDescriptor({}, {}, samplers, spec) == DescriptorInvalid::EntryCount);
}

// Every refusal reason has its own non-empty report name, and Count has none.
void TestRefusalNamesDistinct() {
	std::set<std::string> names;
	for (uint32_t reason = 0; reason < static_cast<uint32_t>(DescriptorInvalid::Count); ++reason) {
		const std::string name = DescriptorInvalidName(static_cast<DescriptorInvalid>(reason));
		CHECK(!name.empty());
		names.insert(name);
	}
	CHECK(names.size() == static_cast<size_t>(DescriptorInvalid::Count));
	CHECK(std::string(DescriptorInvalidName(DescriptorInvalid::Count)).empty());
}

// ---- The list of retained store records. ----

// The bytes of one test record, and a cap that holds many of them.
constexpr uint64_t RECORD_BYTES   = 100;
constexpr uint64_t ROOMY_CAP      = 10 * RECORD_BYTES;
constexpr uint8_t  RECORD_FILLING = 0xAB;

std::unique_ptr<RetainedRecord> MakeRecord(uint64_t size = RECORD_BYTES) {
	return std::make_unique<RetainedRecord>(std::vector<uint8_t>(size, RECORD_FILLING));
}

// When a check fails under a mutation the list may hold links to freed records, or records may
// point at a freed list. Leaking all of them keeps the failing run from walking freed memory, so it
// reports the failure instead of crashing. For the same reason the tests that do not destroy a
// waiting record on purpose declare their records before the list, so the list releases them first.
void LeakOnFailure(std::unique_ptr<RetainedRecordList>& list,
                   std::initializer_list<std::unique_ptr<RetainedRecord>*> records) {
	(void)list.release();
	for (auto* record: records) {
		(void)record->release();
	}
}

// A record destroyed while it waits leaves the list, and is counted neither as dropped nor as
// unused. The baseline keeps the record, which the release then counts as unused. The mutation is
// to take the unlink out of the record's destructor, and the list still counts the record.
void TestDestroyedRecordUnlinks() {
	for (const bool destroy: {false, true}) {
		auto list   = std::make_unique<RetainedRecordList>(ROOMY_CAP);
		auto first  = MakeRecord();
		auto second = MakeRecord();
		list->Add(*first);
		list->Add(*second);
		if (destroy) {
			first.reset();
		}
		const uint64_t waiting  = destroy ? 1 : 2;
		const auto     counters = list->Counters();
		const bool     unlinked =
		    counters.retained == waiting && counters.retained_bytes == waiting * RECORD_BYTES;
		CHECK(unlinked);
		if (!unlinked) {
			LeakOnFailure(list, {&first, &second});
			continue;
		}
		list->Release();
		const auto released = list->Counters();
		CHECK(released.unused_at_exit == waiting && released.dropped_over_cap == 0 &&
		      released.retained == 0 && released.retained_bytes == 0);
	}
}

// Adding a record that already waits does nothing. The baseline adds a second record, which counts.
// The mutation is to take the Linked check out of Add, and the one record is counted twice.
void TestAddTwiceDoesNothing() {
	for (const bool same: {false, true}) {
		auto first  = MakeRecord();
		auto second = MakeRecord();
		auto list   = std::make_unique<RetainedRecordList>(ROOMY_CAP);
		list->Add(*first);
		list->Add(same ? *first : *second);
		const uint64_t waiting  = same ? 1 : 2;
		const auto     counters = list->Counters();
		const bool     counted_once =
		    counters.retained == waiting && counters.retained_bytes == waiting * RECORD_BYTES;
		CHECK(counted_once);
		if (!counted_once) {
			LeakOnFailure(list, {&first, &second});
		}
	}
}

// A record the cap dropped gives nothing to Take and is not counted as stored. The cap holds one
// record, so the second one added drops the first. The baseline takes the second, which waits and
// gives its bytes. The mutation is to take the list check out of Take, and the dropped record is
// counted as stored.
void TestTakeAfterDropGivesNothing() {
	for (const bool take_dropped: {false, true}) {
		auto               first  = MakeRecord();
		auto               second = MakeRecord();
		RetainedRecordList list(RECORD_BYTES);
		list.Add(*first);
		list.Add(*second);
		CHECK(!first->Linked() && second->Linked() && list.Counters().dropped_over_cap == 1);
		const auto bytes = list.Take(take_dropped ? *first : *second);
		CHECK(bytes.size() == (take_dropped ? 0 : RECORD_BYTES));
		CHECK(list.Counters().stored_first_use == (take_dropped ? 0u : 1u));
	}
}

// Take from a list the record does not wait in gives nothing and leaves the record where it was.
// The baseline takes it from its own list. The mutation is to take the list check out of Take, and
// the other list hands out the record's bytes.
void TestTakeFromOtherListGivesNothing() {
	for (const bool other: {false, true}) {
		auto               record = MakeRecord();
		RetainedRecordList own(ROOMY_CAP);
		RetainedRecordList stranger(ROOMY_CAP);
		own.Add(*record);
		const auto bytes = (other ? stranger : own).Take(*record);
		CHECK(bytes.size() == (other ? 0 : RECORD_BYTES));
		CHECK(record->Linked() == other && own.Counters().retained == (other ? 1u : 0u));
		CHECK(stranger.Counters().stored_first_use == 0);
	}
}

// A record larger than the whole cap is dropped alone, and the records that wait stay. The cap holds
// two and a half records. The baseline adds a third record of the usual size, which goes over the
// cap and drops the oldest. The mutation is to take the early check out of Add, and the oversized
// record drops both waiting records before it drops itself.
void TestOversizedRecordDropsOnlyItself() {
	constexpr uint64_t cap = 2 * RECORD_BYTES + RECORD_BYTES / 2;
	for (const bool oversized: {false, true}) {
		auto               first  = MakeRecord();
		auto               second = MakeRecord();
		auto               third  = MakeRecord(oversized ? 2 * cap : RECORD_BYTES);
		RetainedRecordList list(cap);
		list.Add(*first);
		list.Add(*second);
		list.Add(*third);
		const auto counters = list.Counters();
		CHECK(counters.dropped_over_cap == 1 && counters.retained == 2 &&
		      counters.retained_bytes == 2 * RECORD_BYTES);
		CHECK(first->Linked() == oversized && second->Linked() && third->Linked() == !oversized);
		CHECK(third->bytes.empty() == oversized && first->bytes.empty() == !oversized);
	}
}

// A second release adds nothing to the unused count. The first release of two waiting records is the
// baseline, which counts both. The mutation is to count the waiting records in Release without
// taking them out, and the second release counts them again.
void TestReleaseTwiceAddsNothing() {
	auto               first  = MakeRecord();
	auto               second = MakeRecord();
	RetainedRecordList list(ROOMY_CAP);
	list.Add(*first);
	list.Add(*second);
	list.Release();
	CHECK(list.Counters().unused_at_exit == 2 && list.Counters().retained == 0);
	list.Release();
	CHECK(list.Counters().unused_at_exit == 2);
	CHECK(!first->Linked() && first->bytes.empty() && !second->Linked() && second->bytes.empty());
}

// A list destroyed before its records detaches them, so destroying a record later touches no list.
// The baseline destroys the record first, which leaves the list empty. The mutation is to give the
// list a destructor that does not release, and the record still points at the destroyed list.
void TestListDestroyedBeforeRecord() {
	for (const bool list_first: {false, true}) {
		auto list   = std::make_unique<RetainedRecordList>(ROOMY_CAP);
		auto record = MakeRecord();
		list->Add(*record);
		if (!list_first) {
			record.reset();
			const bool empty = list->Counters().retained == 0 && list->Counters().retained_bytes == 0;
			CHECK(empty);
			if (!empty) {
				LeakOnFailure(list, {});
			}
			continue;
		}
		list.reset();
		const bool detached = !record->Linked();
		CHECK(detached);
		if (!detached) {
			(void)record.release();
		}
	}
}

} // namespace

int main() {
	TestVariantsFollowSpecialization();
	TestVariantCapAndSettling();
	TestCacheProbePostsSettledProgram();
	TestProbeResetsInterval();
	TestRejectsSettle();
	TestRetireCapsWithoutCounting();
	TestUnitKeySeparatesUserDataCount();
	TestInvalidDescriptorsRefused();
	TestReservedDstSelRefused();
	TestUnknownBufferFormatTakesRawPath();
	TestImageMipsRefused();
	TestFixedMipModeNeedsOneMip();
	TestImageNumericClassRefused();
	TestImageDimensionRefused();
	TestStorageMsaaArrayRefused();
	TestAtomicMsaaRefused();
	TestAtomicFloatRefused();
	TestImageFormatRefused();
	TestIndirectChildrenChecked();
	TestBuffersCheckedBeforeImages();
	TestDepthCompareConvertedRefused();
	TestPackedUnormSamplingNeedsDirect2D();
	TestSamplerRootsChecked();
	TestShortSpecializationRefused();
	TestRefusalNamesDistinct();
	TestDestroyedRecordUnlinks();
	TestAddTwiceDoesNothing();
	TestTakeAfterDropGivesNothing();
	TestTakeFromOtherListGivesNothing();
	TestOversizedRecordDropsOnlyItself();
	TestReleaseTwiceAddsNothing();
	TestListDestroyedBeforeRecord();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("PrescanTests passed");
	return 0;
}
