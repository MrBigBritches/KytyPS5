#include "graphics/host_gpu/renderer/pipeline/prescanPreparer.h"

#include "common/firstRunReport.h"
#include "common/hostException.h"
#include "common/logging/log.h"
#include "common/threadPriority.h"
#include "graphics/host_gpu/renderer/pipeline/prescanHeapProbe.h"
#include "graphics/shader/moduleId.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/shaderCompiler.h"
#include "graphics/shader/storegen/streamWorker.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <deque>
#include <exception>
#include <fmt/format.h>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

namespace IR = ShaderRecompiler::IR;
using StoreGen::StreamCompletion;
using StoreGen::StreamOutcome;
using StoreGen::StreamRequest;
using StoreGen::StreamRequestKind;
using StoreGen::StreamStatus;
using StoreGen::StreamSubmitStatus;

using State     = PrescanState<PrescanSpec>;
using Admission = State::Admission;
using Gate      = State::Gate;
using Outcome   = State::Outcome;
using SharedCode = std::shared_ptr<const std::vector<uint8_t>>;

/// How many more shared code entries than plans may pile up before the expired ones are swept.
constexpr size_t SHARED_CODE_SLACK = 64;

// One snapshot the state posted, waiting to be prepared or for its program's plan.
struct Snapshot {
	PrescanItem item;
	Admission   admission;
};

// What the preparer keeps of one program for its later snapshots, which is the code it sends and
// the plan every snapshot is specialized against.
struct PlanEntry {
	enum class Plan : uint8_t { None, Requested, Ready };

	uint64_t              code_hash           = 0;
	uint32_t              user_data_count     = 0;
	uint32_t              code_size_bytes     = 0;
	uint32_t              scratch_size_dwords = 0;
	std::vector<uint32_t> static_state;
	// Shared with the transport, which may still hold it for a request after the entry is gone, and
	// with every unit of the same code.
	SharedCode code;
	Plan       plan_state = Plan::None;
	// The preparer's own copy. The walker writes scratch through a const plan, so no other thread
	// may ever walk it.
	IR::ResourcePlan plan;
	// Snapshots posted while the plan was requested, still holding their admissions.
	std::vector<Snapshot> waiting;
	// Snapshots of the program in the ready queue, which keep the entry from being dropped.
	uint32_t queued = 0;
	// The pass that last used the entry, so the oldest idle plans are dropped first.
	uint64_t last_used = 0;
};

// How many requests each unit has outstanding, kept so the hand-off's in-flight table marks a unit
// while any of its requests is out and clears it only when the last one is done. A unit can have a
// Plan and Compiles of several specializations out at once. Every new request marks its unit again,
// which costs one relaxed store and puts back a unit that another unit sharing its slot replaced in
// the meantime. Only the preparer thread, or the thread that joined it, may call it, since the
// counts are plain.
class InFlightUnits {
public:
	// Writes the marks into `table`, which the hand-off owns and which outlives the preparer.
	explicit InFlightUnits(PrescanInFlightTable& table): m_table(table) {}

	// Counts one more request of `unit` and marks the unit.
	void Add(uint64_t unit) {
		m_counts[unit]++;
		m_table.Mark(unit);
	}
	// Counts one request of `unit` as done, and clears the unit when it was the last. Every verdict
	// the request's outcome gave is published before this, and the fence keeps those stores ahead
	// of the clear. A GPU thread that reads the unit as clear and then fences with acquire is sure
	// to see them, and never takes a unit the helper just finished for one it never saw.
	void Release(uint64_t unit) noexcept {
		const auto found = m_counts.find(unit);
		if (found == m_counts.end()) {
			return;
		}
		if (--found->second == 0) {
			m_counts.erase(found);
			std::atomic_thread_fence(std::memory_order_release);
			m_table.Clear(unit);
		}
	}

private:
	PrescanInFlightTable&                  m_table;
	std::unordered_map<uint64_t, uint32_t> m_counts;
};

// One request's share of the work the prescan_running underrun tag counts, and of its unit's
// in-flight count. It lives in the request's pending entry, from just before the send until the
// outcome is handled, the send is refused or throws, or the entries are cleared by a stop, a failure
// or the end of the thread. A request the supervisor sends again to a respawned helper keeps its
// entry, so it stays counted. It is move-only so both shares follow the entry and a moved-from token
// gives nothing back. The underrun share counts only while the report is on, as for every
// PrescanRunningScope. The in-flight share always counts, since hold causes are always recorded.
class PendingWork {
public:
	// Counts toward the underrun tag only, with no unit in flight.
	PendingWork() = default;
	// Counts toward the underrun tag and counts one request of `unit` in `units`, which marks it.
	PendingWork(InFlightUnits& units, uint64_t unit): m_units(&units), m_unit(unit) {
		units.Add(unit);
	}
	// Gives both shares back, the unit's only if this token still holds it.
	~PendingWork() { ReleaseUnit(); }
	// Takes both shares from `other`, which then gives nothing back.
	PendingWork(PendingWork&& other) noexcept
	    : m_running(std::move(other.m_running)), m_units(std::exchange(other.m_units, nullptr)),
	      m_unit(other.m_unit) {}
	// Gives this token's shares back, then takes both of `other`'s, which then gives nothing back.
	PendingWork& operator=(PendingWork&& other) noexcept {
		if (this != &other) {
			ReleaseUnit();
			m_running = std::move(other.m_running);
			m_units   = std::exchange(other.m_units, nullptr);
			m_unit    = other.m_unit;
		}
		return *this;
	}
	PendingWork(const PendingWork&)            = delete;
	PendingWork& operator=(const PendingWork&) = delete;

private:
	// Gives the unit's share back once. A moved-from token holds no count, so it gives nothing.
	void ReleaseUnit() noexcept {
		if (m_units != nullptr) {
			std::exchange(m_units, nullptr)->Release(m_unit);
		}
	}

	Common::FirstRunReport::PrescanRunningScope m_running;
	InFlightUnits*                              m_units = nullptr;
	uint64_t                                    m_unit  = 0;
};

// A request whose outcome has not come yet, with what checking that outcome needs. A Compile keeps
// its specialization, so a refused or dropped one can be forgotten and sent again.
struct PendingRequest {
	StreamRequestKind          kind            = StreamRequestKind::Plan;
	uint64_t                   unit            = 0;
	uint64_t                   code_hash       = 0;
	uint32_t                   user_data_count = 0;
	std::vector<uint32_t>      static_state;
	std::optional<PrescanSpec> spec;
	PendingWork                work;
};

// A report from the GPU thread about a module the preparer built. A duplicate is one the drain
// refused because the runtime had compiled it first.
struct ModuleNote {
	uint64_t id        = 0;
	bool     duplicate = false;
};

// What becomes of the snapshots that waited for a plan.
enum class PlanVerdict : uint8_t { Ready, Retry, Retire };

// Gives a posted snapshot back to the state with exactly one outcome. If preparation throws before
// one is given, the destructor gives NoteDropped, so the program is not left in flight for good.
class AdmissionGuard {
public:
	AdmissionGuard(State& state, const Admission& admission)
	    : m_state(state), m_admission(admission) {}
	~AdmissionGuard() {
		if (!m_open) {
			return;
		}
		// A destructor must not throw, and the only throw here is the state refusing an admission
		// that is no longer open, which leaves nothing to give back.
		try {
			m_state.NoteDropped(m_admission);
		} catch (const std::exception&) {
		}
	}
	AdmissionGuard(const AdmissionGuard&)            = delete;
	AdmissionGuard& operator=(const AdmissionGuard&) = delete;

	// The snapshot gave a specialization that is not built.
	void Reject() {
		m_state.NoteRejected(m_admission);
		m_open = false;
	}
	// The program can never be prepared.
	void Retire() {
		m_state.Retire(m_admission);
		m_open = false;
	}
	// The snapshot gave this specialization.
	[[nodiscard]] Outcome Snapshot(const PrescanSpec& spec) {
		const auto outcome = m_state.NoteSnapshot(m_admission, spec);
		m_open             = false;
		return outcome;
	}
	// The snapshot waits for its plan, which gives it an outcome later.
	void Park() { m_open = false; }

private:
	State&    m_state;
	Admission m_admission;
	bool      m_open = true;
};

// Whether a direct dispatch has no groups on some axis, which the runtime skips.
bool IsZeroSized(const StoreGen::PrescanDispatchInputs& inputs) {
	return !inputs.indirect && std::ranges::any_of(inputs.workgroup_counts,
	                                               [](uint32_t count) { return count == 0; });
}

// Whether a plan reads the group counts, which an indirect dispatch keeps in guest memory that the
// scan cannot vouch for.
bool ReadsGroupCounts(const IR::ResourcePlan& plan) {
	return std::ranges::any_of(plan.descriptor_sources, [](const auto& source) {
		return source.indirect_descriptor && source.indirect_descriptor->workgroup_axis != UINT32_MAX;
	});
}

// Reads a Compile answer's record, which must be a compute module of the program asked about.
std::optional<DecodedModuleRecord> DecodeCompiled(std::span<const uint8_t> bytes,
                                                  const PendingRequest&    request) {
	IR::Reader reader {bytes};
	auto       type = StoreRecord::GraphicsPipeline;
	IR::IoAll(reader, type);
	if (!reader.ok || type != StoreRecord::Module) {
		return std::nullopt;
	}
	DecodedModuleRecord record;
	DecodeModuleRecord(reader, record);
	if (!reader.ok || !reader.bytes.empty() || record.stage != ShaderType::Compute ||
	    record.hash != request.code_hash || record.user_data_count != request.user_data_count ||
	    record.static_state != request.static_state) {
		return std::nullopt;
	}
	return record;
}

// Whether a plan entry has nothing in hand, so it may be dropped.
bool IsIdle(const PlanEntry& entry) {
	return entry.plan_state != PlanEntry::Plan::Requested && entry.waiting.empty() &&
	       entry.queued == 0;
}

// The verdict a gate's answer gives its unit. Busy gives none, since the unit's snapshot in hand
// will decide. The gate answers a retired unit Capped too, and that unit reads Retired, which puts
// its verdict back after another unit pushed it out. A plan that failed or a Plan refused for good
// retires a unit without blocking it, so the blocked paths would never put that one back.
PrescanUnitVerdict VerdictOfGate(Gate gate, bool retired) {
	switch (gate) {
		case Gate::Post: return PrescanUnitVerdict::Admitted;
		case Gate::Capped:
			return retired ? PrescanUnitVerdict::Retired : PrescanUnitVerdict::Capped;
		case Gate::Settled: return PrescanUnitVerdict::Settled;
		case Gate::Busy: break;
	}
	return PrescanUnitVerdict::None;
}

} // namespace

struct PrescanPreparer::Impl {
	Impl(PrescanHandoff& shared, PrescanPreparerConfig supplied);

	bool RunOnce();
	void ThreadMain();
	void Loop();
	void Fail(const char* what) noexcept;
	void TurnOff();
	void Stop();
	void OnComplete(StreamCompletion&& completion) noexcept;
	void Publish();
	void PublishVerdict(uint64_t unit, PrescanUnitVerdict verdict) noexcept;

	bool DrainNotes();
	void NoteModule(const ModuleNote& note);
	bool DrainKnown();
	bool IsKnown(const PlanEntry& entry, const PrescanSpec& spec) const;
	bool AdmitRing();
	void AdmitItem(PrescanItem item);
	bool Skipped(const PrescanItem& item, const std::optional<ShaderMapSnapshot>& mapped);
	PlanEntry& EntryFor(uint64_t unit, const PrescanItem& item, const ShaderMapSnapshot& mapped,
	                    std::vector<uint32_t>&& static_state);
	void       DropIdlePlans();
	void       ReleaseUnit(uint64_t unit);
	void       Block(uint64_t unit);

	bool DrainCompletions();
	void HandleCompletion(StreamCompletion&& completion);
	void HandlePlanOutcome(const PendingRequest& request, const StreamCompletion& completion);
	PlanVerdict JudgePlan(const PendingRequest& request, const StreamCompletion& completion,
	                      PlanEntry& entry);
	void        FinishWaiting(PlanEntry& entry, PlanVerdict verdict);
	void HandleCompileOutcome(const PendingRequest& request, StreamCompletion& completion);
	void ForgetSpec(const PendingRequest& request);
	void Adopt(const PendingRequest& request, std::vector<uint8_t>&& payload);

	bool PrepareReady();
	void SettleAfterLaneOff();
	void DropReady();
	void PrepareSnapshot(const Snapshot& snapshot);
	void RequestPlan(PlanEntry& entry, const Snapshot& snapshot, AdmissionGuard& guard);
	bool CopyCode(PlanEntry& entry, const PrescanItem& item, uint64_t unit);
	void Specialize(PlanEntry& entry, const Snapshot& snapshot, AdmissionGuard& guard);
	bool Materialize(PlanEntry& entry, const PrescanItem& item, PrescanSpec& spec, bool& heap_image);
	void CountOutcome(Outcome outcome);
	void SendCompile(PlanEntry& entry, const Snapshot& snapshot, const PrescanSpec& spec);
	StreamRequest      MakeRequest(const PlanEntry& entry, const PrescanItem& item,
	                               StreamRequestKind kind) const;
	StreamSubmitStatus Send(StreamRequest request, const PlanEntry& entry, uint64_t unit,
	                        const PrescanSpec* spec = nullptr);

	PrescanHandoff&       handoff;
	PrescanPreparerConfig config;
	State                 state;
	// The transport's callback queues its outcomes here. Only this mutex is shared with it, and it
	// is never held across a call into the transport, so Stop can never wait on itself.
	std::mutex                   completion_mutex;
	std::deque<StreamCompletion> completions;
	// Set when the callback could not queue an outcome. The preparer then stops, since it can no
	// longer tell which of its requests were answered.
	std::atomic<bool> completion_lost {false};
	// Filled by the GPU thread, read by the preparer.
	Common::SpscRing<ModuleNote, PRESCAN_TAKEN_CAPACITY> notes;
	// Skip keys of the modules the cache took in, filled by the thread that owns the program cache
	// and read by the preparer into known_modules. A ring and not a shared set, so the GPU thread
	// never takes a lock the lower-priority preparer could be holding.
	Common::SpscRing<uint64_t, PRESCAN_KNOWN_CAPACITY> known_notes;

	// The preparer's own.
	std::unordered_map<uint64_t, PlanEntry>                      plans;
	std::unordered_map<uint64_t, std::weak_ptr<const std::vector<uint8_t>>> shared_code;
	// Declared before the pending entries, whose tokens give their counts back to it when they go.
	InFlightUnits                                                in_flight_units;
	std::unordered_map<uint64_t, PendingRequest>                 pending;
	std::unordered_set<uint64_t>                                 blocked_units;
	std::unordered_map<uint64_t, uint32_t>                       untaken_by_unit;
	std::unordered_map<uint64_t, uint64_t>                       unit_of_module;
	std::unordered_set<uint64_t>                                 known_modules;
	std::deque<Snapshot>                                         ready;
	uint64_t                                                     next_tag = 1;
	uint64_t                                                     pass     = 0;
	PrescanPreparerCounters                                      counters;

	mutable std::mutex      published_mutex;
	PrescanPreparerCounters published;

	std::atomic<bool> stopping {false};
	std::mutex        stop_mutex;
	// Made last, since the factory hands it a callback into this object.
	std::unique_ptr<StoreGen::PrescanTransport> transport;
	std::thread                                 thread;
};

PrescanPreparer::Impl::Impl(PrescanHandoff& shared, PrescanPreparerConfig supplied)
    : handoff(shared), config(std::move(supplied)), state(shared), in_flight_units(shared.in_flight) {
	if (!config.make_transport || !config.lookup || !config.sink) {
		throw std::invalid_argument("PrescanPreparer needs a transport factory, a lookup and a sink");
	}
	// The seed lives on in the set only, so the config's copy is freed.
	known_modules.insert(config.known_modules.begin(), config.known_modules.end());
	std::vector<uint64_t>().swap(config.known_modules);
	counters.known_modules = known_modules.size();
	transport = config.make_transport(config.host, config.device, [this](StreamCompletion&& done) {
		OnComplete(std::move(done));
	});
	if (transport == nullptr) {
		throw std::invalid_argument("PrescanPreparer's transport factory made no transport");
	}
}

// One pass. The ring goes first and whole, so every item is answered before any is prepared. The
// answers go before the preparation, so snapshots whose plan just came are prepared in this pass.
// The cache's module reports go after the ring, so a key whose report happens before an admitted
// item's push is known when that item is prepared.
// Once the helper's lane is off nothing could be sent, so a pass does nothing. A pass counts as one
// unit of pre-scan work for the underrun tag while it runs, even with no request outstanding.
bool PrescanPreparer::Impl::RunOnce() {
	if (counters.lane_off) {
		return false;
	}
	if (completion_lost.load(std::memory_order_acquire)) {
		throw std::runtime_error("an answer from the helper could not be queued");
	}
	const Common::FirstRunReport::PrescanRunningScope running;
	pass++;
	bool worked = DrainNotes();
	worked      = AdmitRing() || worked;
	worked      = DrainCompletions() || worked;
	worked      = DrainKnown() || worked;
	worked      = PrepareReady() || worked;
	Publish();
	return worked;
}

// The preparer thread's body. It lowers its priority, runs passes until Stop, and turns any error
// into Fail, since an uncaught throw would end the process while the game runs on fine without the
// prescan. PrescanState throws on misuse, and anything else may throw too.
void PrescanPreparer::Impl::ThreadMain() {
#if KYTY_PLATFORM == KYTY_PLATFORM_LINUX
	// The walker's faults are handled on this thread, which needs its own signal stack.
	if (!Common::HostException::InitializeThreadSignalStack()) {
		Log::WriteToConsoleAndLog("Prescan: the preparer thread could not get its own signal stack\n");
	}
#endif
	// Below normal, like the warm workers, so the guest's threads win any contention.
	(void)Common::LowerCurrentThreadPriority();
	try {
		Loop();
	} catch (const std::exception& error) {
		Fail(error.what());
	} catch (...) {
		Fail("an unknown error");
	}
	// No pass reads the entries once the thread ends. A lane that turned off leaves the entries of
	// requests the stopped transport never answered, and they would hold up the underrun tag until
	// Stop, which may never come while the game runs. If the helper was in fact still working on one
	// of them, that work stops counting here. That is accepted, since the lane is off for the session
	// and nothing the helper finishes is used, while a count stuck above 0 would tag every later
	// underrun.
	pending.clear();
}

// The epoch is read before the pass, so a bump that lands while the pass drains makes the wait
// return at once, and no work is left behind. A pass that turned the prescan off ends the thread,
// since no later pass would do anything.
void PrescanPreparer::Impl::Loop() {
	while (!stopping.load(std::memory_order_acquire)) {
		const uint32_t seen = handoff.wake_epoch.load(std::memory_order_acquire);
		(void)RunOnce();
		if (stopping.load(std::memory_order_acquire) || counters.lane_off) {
			return;
		}
		handoff.wake_epoch.wait(seen, std::memory_order_acquire);
	}
}

// Stops the prescan for the rest of the run. The failure is marked before anything that could
// throw, and each step is tried alone, so the thread always ends cleanly. The transport is stopped
// from the preparer's own thread, which is the only one that calls it.
void PrescanPreparer::Impl::Fail(const char* what) noexcept {
	counters.failed = true;
	handoff.preparer_stopped.store(true, std::memory_order_release);
	try {
		Log::WriteToConsoleAndLog(fmt::format(
		    "Prescan: the preparer stopped on an error, so the prescan is off: {}\n", what));
	} catch (...) {
	}
	try {
		Publish();
	} catch (...) {
	}
	try {
		transport->Stop();
	} catch (...) {
	}
	// No pass handles another outcome, so the entries go now and the underrun tag drops with them.
	// ThreadMain clears again when the thread ends, which then finds nothing.
	pending.clear();
}

// The helper's lane is off for the session, so every later send would be refused. Without this
// each dispatch would be admitted, refused and given back with NoteDropped, over and over, while
// the scanner kept filling the ring. The scanner is told through the hand-off. It runs inside a
// send, so it only marks the prescan off, and PrepareReady settles what is held once the send's
// caller is done with its plan entry.
void PrescanPreparer::Impl::TurnOff() {
	if (counters.lane_off) {
		return;
	}
	counters.lane_off = true;
	handoff.preparer_stopped.store(true, std::memory_order_release);
	Log::WriteToConsoleAndLog(
	    "Prescan: the helper is off for this session, so the prescan is off\n");
}

// The thread is joined before the transport is stopped, so the transport never runs Stop beside a
// Submit from the preparer. Stopping it completes what it holds as Dropped, which nothing reads. On
// a normal Stop the thread's end has already cleared the pending entries before the transport
// stops. The clear here is for a preparer whose thread never ran, as when a test drives it with
// RunOnce. Its entries would otherwise hold up the underrun tag, since the preparer outlives
// StopPrescan. The hand-off's stop flag goes first. The thread that sees `stopping` then sees it
// too, so every in-flight mark the thread clears on its way out comes after the flag, and a GPU
// thread that finds a unit clear finds the prescan stopped.
void PrescanPreparer::Impl::Stop() {
	std::lock_guard lock(stop_mutex);
	handoff.preparer_stopped.store(true, std::memory_order_release);
	stopping.store(true, std::memory_order_release);
	WakePreparer(handoff);
	if (thread.joinable()) {
		thread.join();
	}
	transport->Stop();
	pending.clear();
}

// The transport's callback. It only queues the outcome and wakes the preparer, since the
// supervisor's deadlines include the time spent here. It never throws. An outcome it cannot queue
// is marked lost, and the next pass stops the prescan.
void PrescanPreparer::Impl::OnComplete(StreamCompletion&& completion) noexcept {
	try {
		std::lock_guard lock(completion_mutex);
		completions.push_back(std::move(completion));
	} catch (...) {
		completion_lost.store(true, std::memory_order_release);
	}
	WakePreparer(handoff);
}

// Copies the counters, with the state's own, where Counters() reads them from another thread.
void PrescanPreparer::Impl::Publish() {
	auto copy  = counters;
	copy.state = state.Counters();
	std::lock_guard lock(published_mutex);
	published = std::move(copy);
}

// Records the preparer's latest decision about a unit for the GPU thread, and counts a verdict that
// pushed another unit's out. It is written whether or not the report is on, like the in-flight
// table. Some paths write nothing on purpose and leave the unit's previous verdict. A Busy gate
// means the unit already has a snapshot in hand. A Known outcome adds nothing to what the unit's
// earlier snapshots decided. A snapshot parked for its plan, or one whose plan came and is queued
// again, is still Admitted. Snapshots given back by DropReady, Fail or Stop were never decided.
void PrescanPreparer::Impl::PublishVerdict(uint64_t unit, PrescanUnitVerdict verdict) noexcept {
	counters.verdicts_evicted += handoff.verdicts.Publish(unit, verdict) ? 1 : 0;
}

// Takes the GPU thread's reports on the modules it used.
bool PrescanPreparer::Impl::DrainNotes() {
	ModuleNote note;
	bool       any = false;
	while (notes.TryPop(note)) {
		any = true;
		NoteModule(note);
	}
	counters.taken_dropped = notes.Dropped();
	return any;
}

// Counts a module the GPU thread took or the drain found compiled already as used, so its program
// moves away from the back-off. A module refused for any other reason stays unused.
void PrescanPreparer::Impl::NoteModule(const ModuleNote& note) {
	(note.duplicate ? counters.late_duplicates : counters.taken)++;
	const auto found = unit_of_module.find(note.id);
	if (found == unit_of_module.end()) {
		return;
	}
	const auto untaken = untaken_by_unit.find(found->second);
	if (untaken != untaken_by_unit.end() && --untaken->second == 0) {
		untaken_by_unit.erase(untaken);
	}
	unit_of_module.erase(found);
}

// Takes the keys of the modules the cache took in since the last pass. It runs after the ring was
// admitted and just before the snapshots are prepared. When the key's report happens before the
// dispatch's push, on one thread or ordered by other synchronization, the preparer's acquire of
// the item pairs with that push's release and this drain's acquire of the key ring sees the key.
// Reports and pushes from threads that nothing orders can still race, and a key that loses only
// costs a duplicate Compile, which the drain refuses.
bool PrescanPreparer::Impl::DrainKnown() {
	uint64_t key = 0;
	bool     any = false;
	while (known_notes.TryPop(key)) {
		any = true;
		known_modules.insert(key);
	}
	counters.known_modules = known_modules.size();
	counters.known_dropped = known_notes.Dropped();
	return any;
}

// Whether the cache already has the module of a new specialization. The key is made by the one
// function the cache makes it with, from the unit's fields the Compile would echo.
bool PrescanPreparer::Impl::IsKnown(const PlanEntry& entry, const PrescanSpec& spec) const {
	return !known_modules.empty() &&
	       known_modules.contains(ShaderModuleSkipKey(ShaderType::Compute, entry.code_hash,
	                                                  entry.user_data_count, entry.static_state, spec));
}

// Takes every item the scanner pushed and asks the state about each.
bool PrescanPreparer::Impl::AdmitRing() {
	PrescanItem item;
	bool        any = false;
	while (handoff.ring.TryPop(item)) {
		any = true;
		counters.items++;
		AdmitItem(item);
	}
	return any;
}

// Looks the item up, builds the static key the runtime would, and asks the state. A posted
// snapshot joins the ready queue. The item is a copy, so its code hash can be filled in.
void PrescanPreparer::Impl::AdmitItem(PrescanItem item) {
	const auto mapped = config.lookup(item.program_address);
	if (Skipped(item, mapped)) {
		return;
	}
	const auto info =
	    StoreGen::MakePrescanInputInfo(item.inputs, mapped->scratch_size_dwords, config.device);
	if (!info) {
		counters.indirect_thread_dims++;
		return;
	}
	item.code_hash = mapped->hash;
	std::vector<uint32_t> static_state;
	BuildStageStaticKey(*info, static_state);
	const uint64_t unit    = PrescanUnitKey(item.code_hash, item.user_data_count, static_state);
	const auto     untaken = untaken_by_unit.find(unit);
	const bool backoff = untaken != untaken_by_unit.end() && untaken->second >= PRESCAN_UNTAKEN_LIMIT;
	if (backoff && !item.force_probe) {
		// Settled makes the scanner send only the probes of this key.
		counters.backed_off++;
		handoff.feedback.Publish(item.key_hash, PrescanFeedback::Answer::Settled);
		PublishVerdict(unit, PrescanUnitVerdict::BackedOff);
		return;
	}
	const auto admission = state.Admit(item, static_state);
	PublishVerdict(unit, VerdictOfGate(admission.gate, state.Variants().IsRetired(unit)));
	if (admission.gate != Gate::Post) {
		return;
	}
	AdmissionGuard guard(state, admission);
	if (blocked_units.contains(unit)) {
		// Block wrote Retired already. Writing it again puts it back if another unit pushed it out.
		counters.blocked++;
		PublishVerdict(unit, PrescanUnitVerdict::Retired);
		guard.Retire();
		ReleaseUnit(unit);
		return;
	}
	auto& entry = EntryFor(unit, item, *mapped, std::move(static_state));
	ready.push_back({item, admission});
	entry.queued++;
	guard.Park();
}

// Whether an item is skipped before the state sees it, counting why.
bool PrescanPreparer::Impl::Skipped(const PrescanItem&                      item,
                                    const std::optional<ShaderMapSnapshot>& mapped) {
	if (!mapped || mapped->hash == 0) {
		// The state never takes a code hash of 0.
		counters.no_map_entry++;
		return true;
	}
	if (item.user_data_count > static_cast<uint32_t>(HW::UserSgprInfo::SGPRS_MAX) ||
	    mapped->code_size_bytes == 0 || mapped->code_size_bytes % sizeof(uint32_t) != 0) {
		counters.bad_item++;
		return true;
	}
	if (IsZeroSized(item.inputs)) {
		counters.zero_size++;
		return true;
	}
	return false;
}

// The plan entry of a unit, made from the item and its shader map entry when the unit is new. A
// full cache drops its oldest idle plans first.
PlanEntry& PrescanPreparer::Impl::EntryFor(uint64_t unit, const PrescanItem& item,
                                           const ShaderMapSnapshot& mapped,
                                           std::vector<uint32_t>&&  static_state) {
	if (const auto found = plans.find(unit); found != plans.end()) {
		found->second.last_used = pass;
		return found->second;
	}
	if (plans.size() >= PRESCAN_PLAN_CACHE_CAP) {
		DropIdlePlans();
	}
	auto& entry               = plans[unit];
	entry.code_hash           = item.code_hash;
	entry.user_data_count     = item.user_data_count;
	entry.code_size_bytes     = mapped.code_size_bytes;
	entry.scratch_size_dwords = mapped.scratch_size_dwords;
	entry.static_state        = std::move(static_state);
	entry.last_used           = pass;
	return entry;
}

// Drops up to one batch of the least recently used plans of programs with nothing in hand. A
// program whose plan is on its way, or that has snapshots queued or waiting, keeps its entry.
void PrescanPreparer::Impl::DropIdlePlans() {
	std::vector<std::pair<uint64_t, uint64_t>> idle;
	for (const auto& [unit, entry]: plans) {
		if (IsIdle(entry)) {
			idle.emplace_back(entry.last_used, unit);
		}
	}
	const size_t drop = std::min(idle.size(), PRESCAN_PLAN_DROP_BATCH);
	std::nth_element(idle.begin(), idle.begin() + static_cast<std::ptrdiff_t>(drop), idle.end());
	for (size_t index = 0; index < drop; ++index) {
		plans.erase(idle[index].second);
	}
	counters.plans_dropped += drop;
}

// Frees what a unit that will never be prepared again holds. An idle entry goes whole, and one with
// snapshots in hand gives up its code, which those snapshots never send. The entry may be erased
// here, so a caller must not use its own reference to the entry after this returns.
void PrescanPreparer::Impl::ReleaseUnit(uint64_t unit) {
	const auto found = plans.find(unit);
	if (found == plans.end()) {
		return;
	}
	if (IsIdle(found->second)) {
		plans.erase(found);
		return;
	}
	found->second.code.reset();
}

// Marks a unit whose Compile was blamed, had calls or was refused for good, so its later snapshots
// are retired before they send anything.
void PrescanPreparer::Impl::Block(uint64_t unit) {
	blocked_units.insert(unit);
	PublishVerdict(unit, PrescanUnitVerdict::Retired);
	ReleaseUnit(unit);
}

// Takes every outcome the transport queued and handles each.
bool PrescanPreparer::Impl::DrainCompletions() {
	std::deque<StreamCompletion> arrived;
	{
		std::lock_guard lock(completion_mutex);
		arrived.swap(completions);
	}
	for (auto& completion: arrived) {
		HandleCompletion(std::move(completion));
	}
	return !arrived.empty();
}

// Matches an outcome to its request by tag, since it may have come before Submit returned, counts
// how it ended and hands it on by kind. The request's work token moves into the local copy, so the
// request stays counted, and its unit in flight, until its outcome is handled, from the helper's
// answer to the end of this. When the sink throws, the token clears the unit while the error
// unwinds, before Fail sets the hand-off's stop flag. A GPU thread can then find the unit clear
// and the prescan still running for the few microseconds between, which is accepted, since it
// only gives one miss the cause of a running prescan.
void PrescanPreparer::Impl::HandleCompletion(StreamCompletion&& completion) {
	const auto found = pending.find(completion.tag);
	if (found == pending.end()) {
		return;
	}
	const PendingRequest request = std::move(found->second);
	pending.erase(found);
	counters.blamed += completion.outcome == StreamOutcome::Blamed ? 1 : 0;
	counters.dropped += completion.outcome == StreamOutcome::Dropped ? 1 : 0;
	if (request.kind == StreamRequestKind::Plan) {
		HandlePlanOutcome(request, completion);
	} else {
		HandleCompileOutcome(request, completion);
	}
}

// Settles a Plan's outcome. The snapshots that waited go on to be prepared, back to the state, or
// are retired, and a retired program's entry is freed. A plan that came writes no verdict, since
// the snapshots it lets on decide. An entry with its Plan out is never idle, so it is never dropped
// and the early return is not reached.
void PrescanPreparer::Impl::HandlePlanOutcome(const PendingRequest&   request,
                                              const StreamCompletion& completion) {
	const auto found = plans.find(request.unit);
	if (found == plans.end()) {
		return;
	}
	auto&      entry   = found->second;
	const auto verdict = JudgePlan(request, completion, entry);
	if (verdict != PlanVerdict::Ready) {
		PublishVerdict(request.unit, verdict == PlanVerdict::Retry ? PrescanUnitVerdict::Deferred
		                                                           : PrescanUnitVerdict::Retired);
	}
	entry.plan_state =
	    verdict == PlanVerdict::Ready ? PlanEntry::Plan::Ready : PlanEntry::Plan::None;
	FinishWaiting(entry, verdict);
	if (verdict == PlanVerdict::Retire) {
		ReleaseUnit(request.unit);
	}
}

// What a Plan's outcome means for the program. A blamed or failed plan retires it, since the child
// would fail on it again, while a dropped one may be asked for again.
PlanVerdict PrescanPreparer::Impl::JudgePlan(const PendingRequest&   request,
                                             const StreamCompletion& completion, PlanEntry& entry) {
	if (completion.outcome == StreamOutcome::Blamed) {
		return PlanVerdict::Retire;
	}
	const auto& response = completion.response;
	if (completion.outcome == StreamOutcome::Dropped || response.status == StreamStatus::Dropped) {
		counters.dropped += completion.outcome == StreamOutcome::Answered ? 1 : 0;
		return PlanVerdict::Retry;
	}
	if (response.status == StreamStatus::HasCalls) {
		counters.has_calls++;
		return PlanVerdict::Retire;
	}
	if (response.status == StreamStatus::Ok && response.static_state != request.static_state) {
		counters.key_mismatch++;
		return PlanVerdict::Retire;
	}
	IR::ResourcePlan plan;
	if (response.status != StreamStatus::Ok || !IR::LoadResourcePlan(response.payload, plan)) {
		counters.plan_failed++;
		return PlanVerdict::Retire;
	}
	entry.plan = std::move(plan);
	return PlanVerdict::Ready;
}

// Moves the snapshots that waited for the plan on. They go to the ready queue when it came, and
// back to the state otherwise.
void PrescanPreparer::Impl::FinishWaiting(PlanEntry& entry, PlanVerdict verdict) {
	auto waiting = std::move(entry.waiting);
	entry.waiting.clear();
	for (const auto& snapshot: waiting) {
		if (verdict == PlanVerdict::Ready) {
			ready.push_back(snapshot);
			entry.queued++;
			continue;
		}
		AdmissionGuard guard(state, snapshot.admission);
		if (verdict == PlanVerdict::Retire) {
			guard.Retire();
		}
	}
}

// A Compile's specialization is already recorded, so only the answer's fate is decided here. A
// blamed module or a program with calls blocks the program. A dropped one is forgotten so a later
// snapshot sends it again, while a failed one stays recorded, since the child would fail again. An
// adopted answer's payload is moved on, since the store takes those bytes as they are.
void PrescanPreparer::Impl::HandleCompileOutcome(const PendingRequest& request,
                                                 StreamCompletion&     completion) {
	if (completion.outcome == StreamOutcome::Blamed) {
		Block(request.unit);
		return;
	}
	if (completion.outcome == StreamOutcome::Dropped) {
		PublishVerdict(request.unit, PrescanUnitVerdict::Deferred);
		ForgetSpec(request);
		return;
	}
	auto& response = completion.response;
	switch (response.status) {
		case StreamStatus::Ok: break;
		case StreamStatus::HasCalls:
			counters.has_calls++;
			Block(request.unit);
			return;
		case StreamStatus::Dropped:
			counters.dropped++;
			PublishVerdict(request.unit, PrescanUnitVerdict::Deferred);
			ForgetSpec(request);
			return;
		case StreamStatus::Failed:
			counters.compile_failed++;
			PublishVerdict(request.unit, PrescanUnitVerdict::CompileFailed);
			return;
	}
	if (response.static_state != request.static_state) {
		counters.key_mismatch++;
		PublishVerdict(request.unit, PrescanUnitVerdict::CompileFailed);
		return;
	}
	Adopt(request, std::move(response.payload));
}

// Lets the variant tracker forget a Compile's specialization that was never built.
void PrescanPreparer::Impl::ForgetSpec(const PendingRequest& request) {
	if (request.spec) {
		state.Forget(request.unit, *request.spec);
		counters.forgotten++;
	}
}

// Decodes the record and hands it to the sink with its bytes, which makes its shader module on this
// thread. The module counts as unused until the GPU thread reports it. The unit reads Built only
// once the sink took the module, since a sink that throws installed nothing.
void PrescanPreparer::Impl::Adopt(const PendingRequest& request, std::vector<uint8_t>&& payload) {
	auto record = DecodeCompiled(payload, request);
	if (!record) {
		counters.decode_failed++;
		PublishVerdict(request.unit, PrescanUnitVerdict::CompileFailed);
		return;
	}
	untaken_by_unit[request.unit]++;
	unit_of_module[record->id] = request.unit;
	counters.adopted++;
	config.sink(std::move(*record), std::move(payload));
	PublishVerdict(request.unit, PrescanUnitVerdict::Built);
}

// Prepares the posted snapshots. When one throws, the rest still hold open admissions, so they are
// given back before the error goes on. A send that found the lane off stops the loop, and the
// prescan is settled once the loop is out of every snapshot's preparation. The settle is inside
// the try, so a drain that throws still gives back what it had not reached.
bool PrescanPreparer::Impl::PrepareReady() {
	if (ready.empty()) {
		return false;
	}
	try {
		while (!ready.empty() && !counters.lane_off) {
			const Snapshot snapshot = ready.front();
			ready.pop_front();
			PrepareSnapshot(snapshot);
		}
		if (counters.lane_off) {
			SettleAfterLaneOff();
		}
	} catch (...) {
		DropReady();
		throw;
	}
	return true;
}

// Gives every snapshot the prescan still holds its one outcome, once the lane is off. The
// supervisor marks its lane off before it hands the Dropped outcomes over, so a send can see
// LaneOff while outcomes of requests already sent are still on their way. Stopping the transport
// joins whatever delivers them, so every outcome is queued by the time it returns, and the drain
// then gives each snapshot that waited for a plan its outcome. Snapshots the drain moves to the
// ready queue, and the ones the loop never reached, are given back last. It runs between
// snapshots, never inside one's preparation, since the drain may free a plan entry that a
// preparation still holds. The preparer's Stop stops the transport again later, which then does
// nothing.
void PrescanPreparer::Impl::SettleAfterLaneOff() {
	transport->Stop();
	(void)DrainCompletions();
	DropReady();
}

// Gives every snapshot still in the ready queue back to the state, after an error stopped the pass
// or the lane turned off.
void PrescanPreparer::Impl::DropReady() {
	for (const auto& snapshot: ready) {
		if (const auto found = plans.find(snapshot.admission.unit); found != plans.end()) {
			found->second.queued -= found->second.queued > 0 ? 1 : 0;
		}
		const AdmissionGuard guard(state, snapshot.admission);
	}
	ready.clear();
}

// Prepares one posted snapshot. A blocked program is retired first, which catches a snapshot that
// was admitted before the outcome that blocked it was read. Otherwise the snapshot waits for its
// program's plan, asks for it, or is specialized against it. The snapshot's own queued count keeps
// its entry from being dropped, so the return for a missing entry is not reached and writes no
// verdict.
void PrescanPreparer::Impl::PrepareSnapshot(const Snapshot& snapshot) {
	AdmissionGuard guard(state, snapshot.admission);
	const uint64_t unit  = snapshot.admission.unit;
	const auto     found = plans.find(unit);
	if (found != plans.end()) {
		found->second.queued -= found->second.queued > 0 ? 1 : 0;
		found->second.last_used = pass;
	}
	if (blocked_units.contains(unit)) {
		// Block wrote Retired already. Writing it again puts it back if another unit pushed it out.
		counters.blocked++;
		PublishVerdict(unit, PrescanUnitVerdict::Retired);
		guard.Retire();
		ReleaseUnit(unit);
		return;
	}
	if (found == plans.end()) {
		return;
	}
	auto& entry = found->second;
	switch (entry.plan_state) {
		case PlanEntry::Plan::Requested:
			entry.waiting.push_back(snapshot);
			guard.Park();
			return;
		case PlanEntry::Plan::None: RequestPlan(entry, snapshot, guard); return;
		case PlanEntry::Plan::Ready: Specialize(entry, snapshot, guard); return;
	}
}

// Sends the program's Plan, after copying its code the first time. The snapshot waits for the plan
// only once the send was queued. A refusal for now gives it back to the state through the guard,
// and a refusal for good retires the program. Send writes the verdict of either refusal.
void PrescanPreparer::Impl::RequestPlan(PlanEntry& entry, const Snapshot& snapshot,
                                        AdmissionGuard& guard) {
	const uint64_t unit = snapshot.admission.unit;
	if (entry.code == nullptr && !CopyCode(entry, snapshot.item, unit)) {
		return;
	}
	const auto status = Send(MakeRequest(entry, snapshot.item, StreamRequestKind::Plan), entry, unit);
	if (status == StreamSubmitStatus::Queued) {
		entry.plan_state = PlanEntry::Plan::Requested;
		entry.waiting.push_back(snapshot);
		guard.Park();
		return;
	}
	if (status == StreamSubmitStatus::NeverRetry) {
		guard.Retire();
		ReleaseUnit(unit);
	}
}

// Copies the program's code through the never-faulting reader, and checks it still hashes as the
// shader map says, since the game may have put another program at the address since the scan. Code
// another unit of the same program still holds is shared instead of copied again. A failure is the
// verdict of `unit`, the unit the entry belongs to.
bool PrescanPreparer::Impl::CopyCode(PlanEntry& entry, const PrescanItem& item, uint64_t unit) {
	if (const auto found = shared_code.find(entry.code_hash); found != shared_code.end()) {
		if (auto code = found->second.lock(); code != nullptr && code->size() == entry.code_size_bytes) {
			entry.code = std::move(code);
			return true;
		}
	}
	std::vector<uint32_t> words(entry.code_size_bytes / sizeof(uint32_t));
	if (config.read_memory == nullptr ||
	    !config.read_memory(nullptr, item.program_address, std::span<uint32_t>(words))) {
		counters.code_unreadable++;
		PublishVerdict(unit, PrescanUnitVerdict::CodeUnreadable);
		return false;
	}
	if (XXH3_64bits(words.data(), entry.code_size_bytes) != entry.code_hash) {
		counters.code_changed++;
		PublishVerdict(unit, PrescanUnitVerdict::CodeChanged);
		return false;
	}
	auto bytes = std::make_shared<std::vector<uint8_t>>(entry.code_size_bytes);
	std::memcpy(bytes->data(), words.data(), entry.code_size_bytes);
	entry.code = std::move(bytes);
	if (shared_code.size() > plans.size() + SHARED_CODE_SLACK) {
		std::erase_if(shared_code, [](const auto& pair) { return pair.second.expired(); });
	}
	shared_code[entry.code_hash] = entry.code;
	return true;
}

// Specializes a snapshot against the program's plan, refuses what would crash the helper, records
// the specialization and sends a new one as a Compile. A new one whose module the cache already has
// is recorded as built, so later snapshots that give it count towards settling, and it is not sent,
// since the drain would refuse the answer as a duplicate. The admission's outcome is the snapshot
// either way, so the skip neither leaves the unit in flight nor counts as a refusal. A Known
// outcome writes no verdict, since it adds nothing to what the unit's earlier snapshots decided.
void PrescanPreparer::Impl::Specialize(PlanEntry& entry, const Snapshot& snapshot,
                                       AdmissionGuard& guard) {
	const uint64_t unit = snapshot.admission.unit;
	if (snapshot.item.inputs.indirect && ReadsGroupCounts(entry.plan)) {
		counters.indirect_args_unknown++;
		PublishVerdict(unit, PrescanUnitVerdict::IndirectArgs);
		guard.Reject();
		return;
	}
	PrescanSpec spec;
	bool        heap_image = false;
	if (!Materialize(entry, snapshot.item, spec, heap_image)) {
		PublishVerdict(unit, PrescanUnitVerdict::Rejected);
		guard.Reject();
		return;
	}
	if (const auto invalid = FindInvalidDescriptor(entry.plan, spec)) {
		counters.invalid[static_cast<size_t>(*invalid)]++;
		PublishVerdict(unit, PrescanUnitVerdict::Rejected);
		guard.Reject();
		return;
	}
	const auto outcome = guard.Snapshot(spec);
	CountOutcome(outcome);
	if (outcome != Outcome::New) {
		return;
	}
	counters.heap_image_variants += heap_image ? 1 : 0;
	if (IsKnown(entry, spec)) {
		counters.known_skipped++;
		PublishVerdict(unit, PrescanUnitVerdict::Built);
		return;
	}
	SendCompile(entry, snapshot, spec);
}

// Materializes with a walk that never faults and never waits. The image heap belongs to the GPU
// thread, so the walk gets a probe that declines every table, which specializes exactly as no heap
// would, and says whether the GPU thread's heap would have been offered one. A walk that skipped a
// load read zeros, so its specialization is a guess and is neither recorded nor sent. It is
// rejected, which counts towards settling the program.
bool PrescanPreparer::Impl::Materialize(PlanEntry& entry, const PrescanItem& item,
                                        PrescanSpec& spec, bool& heap_image) {
	PrescanHeapProbe     heap;
	const IR::SrtRuntime runtime {
	    .user_data             = std::span<const uint32_t>(item.user_data.value, item.user_data_count),
	    .shader_base           = item.program_address,
	    .read_specialization_memory = config.read_memory,
	    .workgroup_counts      = item.inputs.workgroup_counts,
	    .indirect_args_address = item.inputs.indirect_args_address,
	    .image_heap            = &heap,
	};
	IR::ResourceSnapshot resources;
	uint32_t             skipped = 0;
	const bool           built   = IR::MaterializeResources(
        entry.plan, runtime, resources, spec, {.non_faulting = true, .skipped_loads = &skipped});
	heap_image = heap.Offered();
	if (skipped != 0) {
		counters.srt_faults++;
		return false;
	}
	if (!built) {
		counters.materialize_failed++;
		return false;
	}
	return true;
}

// Adds a variant tracker outcome to its counter.
void PrescanPreparer::Impl::CountOutcome(Outcome outcome) {
	switch (outcome) {
		case Outcome::New: counters.variant_modules++; break;
		case Outcome::Known: counters.known_variants++; break;
		case Outcome::Capped: counters.capped_variants++; break;
	}
}

// Sends a new specialization as a Compile. A refusal for good blocks the program, and a refusal for
// now is forgotten in Send. Code a released unit gave up is copied again first. That copy is only a
// guard. Only ReleaseUnit drops an entry's code, and every unit it releases is blocked or retired,
// so none of its snapshots reaches a Compile, and the copy here never runs today.
void PrescanPreparer::Impl::SendCompile(PlanEntry& entry, const Snapshot& snapshot,
                                        const PrescanSpec& spec) {
	const uint64_t unit = snapshot.admission.unit;
	if (entry.code == nullptr && !CopyCode(entry, snapshot.item, unit)) {
		state.Forget(unit, spec);
		counters.forgotten++;
		return;
	}
	auto request           = MakeRequest(entry, snapshot.item, StreamRequestKind::Compile);
	request.specialization = StoreGen::EncodeSpecialization(spec);
	if (Send(std::move(request), entry, unit, &spec) == StreamSubmitStatus::NeverRetry) {
		Block(unit);
	}
}

// A request for the program of a plan entry, with the item's registers and dispatch flags.
StreamRequest PrescanPreparer::Impl::MakeRequest(const PlanEntry& entry, const PrescanItem& item,
                                                 StreamRequestKind kind) const {
	StreamRequest request;
	request.kind                = kind;
	request.code_hash           = entry.code_hash;
	request.dispatch            = item.inputs;
	request.user_data_count     = entry.user_data_count;
	request.scratch_size_dwords = entry.scratch_size_dwords;
	return request;
}

// The pending entry is made before the send, since the outcome may arrive before Submit returns,
// and it is taken back when Submit throws. Its work token counts the request for the underrun tag
// and marks its unit in flight from that moment, for a Plan and a Compile alike, and erasing the
// entry gives both back. A Compile refused for now is forgotten, so a later snapshot sends it
// again. A lane that is off turns the prescan off, and the caller still treats that refusal as one
// for now, so the snapshot in hand is given back. A refusal's verdict, and the stop flag of a lane
// that is off, are written before the entry is erased, since erasing it may clear the unit's
// in-flight mark, and a GPU thread that sees the unit clear must find both. When Submit throws,
// the entry goes before Fail sets the stop flag, and a GPU thread can find the unit clear and the
// prescan still running for the few microseconds between. That is accepted, since it only gives
// one miss the cause of a running prescan.
StreamSubmitStatus PrescanPreparer::Impl::Send(StreamRequest request, const PlanEntry& entry,
                                               uint64_t unit, const PrescanSpec* spec) {
	const uint64_t tag  = next_tag++;
	const auto     kind = request.kind;
	pending.emplace(tag, PendingRequest {kind, unit, entry.code_hash, entry.user_data_count,
	                                     entry.static_state,
	                                     spec != nullptr ? std::optional(*spec) : std::nullopt,
	                                     PendingWork {in_flight_units, unit}});
	StoreGen::StreamSubmitResult result;
	try {
		result = transport->Submit(std::move(request), entry.code, tag);
	} catch (...) {
		pending.erase(tag);
		throw;
	}
	if (result.status == StreamSubmitStatus::Queued) {
		(kind == StreamRequestKind::Plan ? counters.plan_requests : counters.compile_requests)++;
		return result.status;
	}
	const bool never = result.status == StreamSubmitStatus::NeverRetry;
	PublishVerdict(unit, never ? PrescanUnitVerdict::Retired : PrescanUnitVerdict::Deferred);
	if (result.status == StreamSubmitStatus::LaneOff) {
		TurnOff();
	}
	pending.erase(tag);
	(never ? counters.never_retry : counters.refused_sends)++;
	if (spec != nullptr && !never) {
		state.Forget(unit, *spec);
		counters.forgotten++;
	}
	return result.status;
}

PrescanPreparer::PrescanPreparer(PrescanHandoff& handoff, PrescanPreparerConfig config)
    : m_impl(std::make_unique<Impl>(handoff, std::move(config))) {}

PrescanPreparer::~PrescanPreparer() {
	Stop();
}

void PrescanPreparer::Start() {
	m_impl->thread = std::thread([impl = m_impl.get()] { impl->ThreadMain(); });
}

void PrescanPreparer::Stop() {
	m_impl->Stop();
}

bool PrescanPreparer::RunOnce() {
	return m_impl->RunOnce();
}

void PrescanPreparer::NoteTaken(uint64_t module_id) noexcept {
	(void)m_impl->notes.TryPush({.id = module_id, .duplicate = false});
}

void PrescanPreparer::NoteDuplicate(uint64_t module_id) noexcept {
	(void)m_impl->notes.TryPush({.id = module_id, .duplicate = true});
}

void PrescanPreparer::NoteKnownModule(uint64_t skip_key) noexcept {
	if (Stopped()) {
		return;
	}
	(void)m_impl->known_notes.TryPush(skip_key);
}

// The acquire pairs with the release in TurnOff, Fail and Stop.
bool PrescanPreparer::Stopped() const noexcept {
	return m_impl->handoff.preparer_stopped.load(std::memory_order_acquire);
}

PrescanPreparerCounters PrescanPreparer::Counters() const {
	std::lock_guard lock(m_impl->published_mutex);
	return m_impl->published;
}

Common::FirstRunReport::PrescanStateReport PrescanStateReportOf(const PrescanCounters& counters) {
	return {.posted       = counters.posted,
	        .busy         = counters.busy,
	        .capped       = counters.capped,
	        .settled      = counters.settled,
	        .uncertain    = counters.uncertain,
	        .dropped_full = counters.dropped_full};
}

namespace {

// The counts of what the preparer met before it sent anything, with each refusal by its name.
void CopyIntakeCounts(const PrescanPreparerCounters&                counters,
                      Common::FirstRunReport::PrescanPreparerReport& report) {
	report.items                 = counters.items;
	report.no_map_entry          = counters.no_map_entry;
	report.zero_size             = counters.zero_size;
	report.indirect_thread_dims  = counters.indirect_thread_dims;
	report.bad_item              = counters.bad_item;
	report.backed_off            = counters.backed_off;
	report.blocked               = counters.blocked;
	report.plan_requests         = counters.plan_requests;
	report.compile_requests      = counters.compile_requests;
	report.code_unreadable       = counters.code_unreadable;
	report.code_changed          = counters.code_changed;
	report.plan_failed           = counters.plan_failed;
	report.has_calls             = counters.has_calls;
	report.srt_faults            = counters.srt_faults;
	report.materialize_failed    = counters.materialize_failed;
	report.indirect_args_unknown = counters.indirect_args_unknown;
	for (size_t reason = 0; reason < counters.invalid.size(); ++reason) {
		report.invalid.emplace_back(DescriptorInvalidName(static_cast<DescriptorInvalid>(reason)),
		                            counters.invalid[reason]);
	}
}

// The counts of the variants, the requests' outcomes and the modules.
void CopyOutcomeCounts(const PrescanPreparerCounters&                counters,
                       Common::FirstRunReport::PrescanPreparerReport& report) {
	report.variant_modules     = counters.variant_modules;
	report.known_variants      = counters.known_variants;
	report.capped_variants     = counters.capped_variants;
	report.known_skipped       = counters.known_skipped;
	report.heap_image_variants = counters.heap_image_variants;
	// A heap variant is counted only with a new variant, so it never exceeds them. The minimum only
	// keeps a report from wrapping round if that ever changed.
	report.covered_variants =
	    counters.variant_modules - std::min(counters.heap_image_variants, counters.variant_modules);
	report.known_modules    = counters.known_modules;
	report.known_dropped    = counters.known_dropped;
	report.refused_sends    = counters.refused_sends;
	report.never_retry      = counters.never_retry;
	report.blamed           = counters.blamed;
	report.dropped          = counters.dropped;
	report.compile_failed   = counters.compile_failed;
	report.key_mismatch     = counters.key_mismatch;
	report.decode_failed    = counters.decode_failed;
	report.adopted          = counters.adopted;
	report.taken            = counters.taken;
	report.late_duplicates  = counters.late_duplicates;
	report.taken_dropped    = counters.taken_dropped;
	report.plans_dropped    = counters.plans_dropped;
	report.forgotten        = counters.forgotten;
	report.verdicts_evicted = counters.verdicts_evicted;
	report.failed           = counters.failed;
	report.lane_off         = counters.lane_off;
}

} // namespace

Common::FirstRunReport::PrescanPreparerReport
PrescanPreparerReportOf(const PrescanPreparerCounters& counters) {
	Common::FirstRunReport::PrescanPreparerReport report;
	CopyIntakeCounts(counters, report);
	CopyOutcomeCounts(counters, report);
	return report;
}

namespace {

// One kind of round trip, as the supervisor's histogram summed it up.
Common::FirstRunReport::PrescanRttFigures RttOf(const StoreGen::StreamRttSummary& rtt) {
	return {.count = rtt.count, .p50_ms = rtt.p50_ms, .p90_ms = rtt.p90_ms, .max_ms = rtt.max_ms};
}

} // namespace

Common::FirstRunReport::PrescanChildReport
PrescanChildReportOf(const StoreGen::StreamSupervisorStats& stats) {
	Common::FirstRunReport::PrescanChildReport report;
	report.spawns               = stats.spawns;
	report.failed_spawns        = stats.failed_spawns;
	report.crashes              = stats.crashes;
	report.hangs                = stats.hangs;
	report.protocol_errors      = stats.protocol_errors;
	report.failures             = stats.failures;
	report.never_retry          = stats.never_retry;
	report.never_retry_programs = stats.never_retry_programs;
	report.lane_off             = stats.lane_off;
	report.lane_off_reason      = stats.lane_off_reason;
	report.submitted            = stats.submitted;
	report.answered             = stats.answered;
	report.blamed               = stats.blamed;
	report.dropped              = stats.dropped;
	report.resent               = stats.resent;
	report.refused              = stats.refused;
	report.late_replies         = stats.late_replies;
	report.unexpected_replies   = stats.unexpected_replies;
	report.callback_failures    = stats.callback_failures;
	report.peak_outstanding     = stats.peak_outstanding;
	report.priority             = stats.priority;
	report.plan_rtt             = RttOf(stats.plan_rtt);
	report.compile_rtt          = RttOf(stats.compile_rtt);
	return report;
}

} // namespace Libs::Graphics
