#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANPREPARER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANPREPARER_H_

#include "graphics/host_gpu/renderer/pipeline/prescanHandoff.h"
#include "graphics/host_gpu/renderer/pipeline/prescanState.h"
#include "graphics/host_gpu/renderer/pipeline/prescanValidate.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/recompiler/ir/passes/SrtWalker.h"
#include "graphics/shader/shader.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/computeStaticKey.h"
#include "graphics/shader/storegen/computeTranslation.h"
#include "graphics/shader/storegen/prescanTransport.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

// The pre-scan's preparer. It takes the dispatches the scanner found, decides with PrescanState
// which snapshots to prepare, asks the helper for each program's resource plan once, specializes
// every posted snapshot against it in this process, refuses descriptors that would crash the
// helper, and asks the helper to compile each new specialization. A compiled module goes to the
// pipeline cache's intake through a sink. It never waits for the helper and the GPU thread never
// waits for it, so a request still in flight when the game needs its module is simply compiled by
// the GPU thread as before, and the late answer is dropped by the intake as a duplicate.
namespace Libs::Graphics {

/// After this many modules of one program were built and none of them used yet, the program is
/// prepared on probes only, since its later variants are likely to go unused too. A module counts
/// as used when the GPU thread took its pipeline, or when the drain found the runtime had already
/// compiled it, which is a late answer and not a wasted guess.
inline constexpr uint32_t PRESCAN_UNTAKEN_LIMIT = 3;
/// How many module reports the GPU thread can make before the preparer reads them. A full ring
/// drops the report, which only keeps a program backed off a little longer.
inline constexpr size_t PRESCAN_TAKEN_CAPACITY = 1024;
/// The most programs whose plans the preparer keeps. Past it the oldest idle plans are dropped,
/// which only costs a Plan request again.
inline constexpr size_t PRESCAN_PLAN_CACHE_CAP = 4096;
/// How many idle plans one drop removes, so a full cache is not emptied at once and the plans in
/// use stay.
inline constexpr size_t PRESCAN_PLAN_DROP_BATCH = PRESCAN_PLAN_CACHE_CAP / 8;

/// The specialization a snapshot gives, which is what the variant tracker compares.
using PrescanSpec = ShaderRecompiler::IR::ResourceSpecialization;

/// What the preparer counted, for the tests and the report. Only the preparer writes it, and others
/// read the copy it publishes after each pass.
struct PrescanPreparerCounters {
	/// Items taken from the ring.
	uint64_t items = 0;
	/// Items skipped before the state saw them, whose program is not mapped, whose direct dispatch
	/// has a zero size, whose indirect dispatch is in thread dimensions, or whose user data count is
	/// past the registers.
	uint64_t no_map_entry         = 0;
	uint64_t zero_size            = 0;
	uint64_t indirect_thread_dims = 0;
	uint64_t bad_item             = 0;
	/// Items of a program backed off after PRESCAN_UNTAKEN_LIMIT unused modules, which were not
	/// probes.
	uint64_t backed_off = 0;
	/// Posted snapshots of a program whose Compile was blamed or refused for good, which were
	/// retired at once.
	uint64_t blocked = 0;
	/// Requests sent, by kind.
	uint64_t plan_requests    = 0;
	uint64_t compile_requests = 0;
	/// Snapshots dropped because the program's code could not be read or no longer had its hash.
	uint64_t code_unreadable = 0;
	uint64_t code_changed    = 0;
	/// Plans the helper could not give, or that did not load.
	uint64_t plan_failed = 0;
	/// Programs the helper found to have GPU-selected calls.
	uint64_t has_calls = 0;
	/// Snapshots whose non-faulting walk skipped a load. They give no variant and are never sent.
	uint64_t srt_faults = 0;
	/// Snapshots whose materialization failed for another reason.
	uint64_t materialize_failed = 0;
	/// Indirect dispatches whose plan reads the group counts, which are not known at scan time.
	uint64_t indirect_args_unknown = 0;
	/// Snapshots refused before they were sent, by DescriptorInvalid reason.
	std::array<uint64_t, static_cast<size_t>(DescriptorInvalid::Count)> invalid {};
	/// What the variant tracker found in the snapshots that were specialized.
	uint64_t variant_modules = 0;
	uint64_t known_variants  = 0;
	uint64_t capped_variants = 0;
	/// Requests the transport refused for now (lane off, queue full, stopped), and for good.
	uint64_t refused_sends = 0;
	uint64_t never_retry   = 0;
	/// Outcomes. Blamed requests are never sent again and never fall back to in-process work.
	uint64_t blamed         = 0;
	uint64_t dropped        = 0;
	uint64_t compile_failed = 0;
	/// Answers whose static key words were not the ones the preparer built, which means the helper
	/// and the runtime disagree. The answer is not used.
	uint64_t key_mismatch = 0;
	/// Compile answers whose record did not decode or was not for the program asked about.
	uint64_t decode_failed = 0;
	/// Modules handed to the sink.
	uint64_t adopted = 0;
	/// Module ids the GPU thread reported as taken, ids the drain refused as late duplicates, and
	/// reports the full ring dropped.
	uint64_t taken           = 0;
	uint64_t late_duplicates = 0;
	uint64_t taken_dropped   = 0;
	/// Plans dropped from the cache to keep it under PRESCAN_PLAN_CACHE_CAP.
	uint64_t plans_dropped = 0;
	/// Specializations forgotten after a send refused for now or a dropped Compile, so they can be
	/// sent again.
	uint64_t forgotten = 0;
	/// Whether the preparer stopped on an error.
	bool failed = false;
	/// Whether the helper's lane turned off for the session, which turned the prescan off with it.
	bool lane_off = false;
	/// The state's own counts.
	PrescanCounters state;
};

/// Looks a program up in the shader map, by its guest address.
using PrescanMapLookup = std::function<std::optional<ShaderMapSnapshot>(uint64_t address)>;
/// Takes a compiled module on the preparer's thread, decoded and as the helper's EncodeModuleRecord
/// bytes, which the shader store can take as they are. It may create Vulkan handles and must not
/// wait for the GPU thread.
using PrescanSink =
    std::function<void(DecodedModuleRecord&& record, std::vector<uint8_t>&& record_bytes)>;

/// What a preparer is made with.
struct PrescanPreparerConfig {
	/// The shader map. Production passes ShaderTryGetMapped.
	PrescanMapLookup lookup;
	/// Reads guest memory for the program's code and for the walk's table reads. It must never
	/// fault or wait for a readback. Production passes a TryReadBacking wrapper.
	ShaderRecompiler::IR::SrtMemoryReader read_memory = nullptr;
	/// The host values the helper translates with, from the running device and Config.
	StoreGen::HostValues host;
	/// The host subgroup size and the shared memory limit, as GetComputeProgram takes them.
	StoreGen::ComputeKeyDevice device {};
	/// Makes the transport, which sends its outcomes to the preparer.
	StoreGen::PrescanTransportFactory make_transport;
	/// Takes each compiled module.
	PrescanSink sink;
};

/// Prepares the scanner's dispatches on its own thread, or on the caller's when a test drives it
/// with RunOnce.
///
/// One pass drains the ring and asks the state about every item before it prepares any snapshot,
/// so the scanner gets its answers at once and repeats of a busy program are not left in the ring.
/// Then it takes the helper's answers, then prepares the snapshots that were posted. A program's
/// first snapshot copies its code, checks the hash and sends a Plan. The plan is the preparer's own
/// copy, since the walker writes scratch through a const plan. Each snapshot is specialized with a
/// non-faulting walk and no image heap. A walk that skipped a load gives no variant and nothing is
/// sent. A specialization the emitter would refuse is not sent. A new one is sent as a Compile, and
/// its answer is checked against the preparer's own static key before the sink gets it.
///
/// Blamed and NeverRetry answers retire the program and are never retried here or in this process.
/// QueueFull, Stopped and Dropped only skip the snapshot, and the GPU thread compiles the module
/// when the game needs it. A Compile refused for now or dropped is forgotten by the variant
/// tracker, so the same specialization is sent again by a later snapshot. LaneOff skips the
/// snapshot too, and since the lane stays off for the session it also turns the prescan off. The
/// preparer sets the hand-off's preparer_stopped and admits nothing more. Once the pass is out of
/// the snapshot's preparation it stops the transport, which delivers every outcome still on its
/// way, takes those outcomes, and gives back every snapshot it still holds, so each admission gets
/// exactly one outcome. Then its thread ends, so no dispatch is admitted only to be refused again.
///
/// Every call into the transport is made by the preparer's thread, or by the caller of RunOnce,
/// and Stop stops the transport only after that thread is joined, so no two transport calls ever
/// overlap.
class PrescanPreparer {
public:
	/// Makes the preparer and its transport. The hand-off must outlive it. No thread runs until
	/// Start.
	PrescanPreparer(PrescanHandoff& handoff, PrescanPreparerConfig config);
	/// Stops the preparer.
	~PrescanPreparer();
	PrescanPreparer(const PrescanPreparer&)            = delete;
	PrescanPreparer& operator=(const PrescanPreparer&) = delete;

	/// Starts the preparer's thread, below normal priority. Call it at most once, and never with
	/// RunOnce.
	void Start();
	/// Stops the thread, joins it, and only then stops the transport, so the transport is never
	/// called from two threads at once. Outstanding requests complete as Dropped and are discarded.
	/// It sets the hand-off's preparer_stopped first. Any thread but the transport's may call it,
	/// any number of times.
	void Stop();
	/// Does one pass on the calling thread, for a test that drives the preparer without its thread.
	/// An error thrown by a pass propagates. The snapshots the pass was preparing are given back to
	/// the state first, while snapshots already waiting for a plan keep their admissions until the
	/// plan's outcome comes. Once the helper's lane turned the prescan off, a pass does nothing.
	/// @return whether the pass found any work.
	bool RunOnce();

	/// Reports that the GPU thread took the pipeline of a module the preparer built, for the
	/// back-off. Only the GPU thread may call it. It never waits.
	void NoteTaken(uint64_t module_id) noexcept;
	/// Reports that the drain refused a module the preparer built because the runtime had already
	/// compiled it. The answer came late but the guess was right, so it does not count towards the
	/// back-off. Only the GPU thread may call it. It never waits.
	void NoteDuplicate(uint64_t module_id) noexcept;

	/// The counters as the last pass published them. Any thread may call it.
	[[nodiscard]] PrescanPreparerCounters Counters() const;

	/// The state of a preparer. Defined in the source file.
	struct Impl;

private:
	std::unique_ptr<Impl> m_impl;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANPREPARER_H_
