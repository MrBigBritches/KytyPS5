// Tests the prescan's hand-off from the scanner thread to the preparer thread. That is the
// lock-free ring, the table of answers the scanner reads, and the preparer's state over the variant
// tracker. The tracker's own rules are tested in PrescanTests, so the state's tests check only that
// it feeds the tracker the right unit and probe flag, counts its answers and publishes them. Each
// check that proves a behaviour has a baseline that differs only in that behaviour, as the comments
// say.
#include "common/spscRing.h"
#include "graphics/host_gpu/renderer/pipeline/prescanHandoff.h"
#include "graphics/host_gpu/renderer/pipeline/prescanMiss.h"
#include "graphics/host_gpu/renderer/pipeline/prescanState.h"
#include "graphics/host_gpu/renderer/pipeline/prescanVariants.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <typeinfo>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;

int g_failures = 0;

// Counts the allocations made while it is switched on, from every thread, by the operator new
// below.
std::atomic<bool>     g_count_allocations {false};
std::atomic<uint64_t> g_allocation_count {0};
// Where a test puts a pointer it allocated, so the allocation is visible outside the function.
std::atomic<void*> g_allocation_sink {nullptr};

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using IntState = PrescanState<int>;

// A small ring, so filling it takes a handful of pushes.
constexpr size_t SMALL_CAPACITY = 8;
using SmallRing                 = Common::SpscRing<uint64_t, SMALL_CAPACITY>;

constexpr uint64_t CODE_HASH  = 0x5EEDC0DE12345678ull;
constexpr uint64_t OTHER_CODE = 0x0BADF00D87654321ull;
constexpr uint64_t KEY        = 0x1234ABCD5678EF00ull;
constexpr uint64_t OTHER_KEY  = 0x7777AAAA5555CC00ull;
constexpr uint32_t USER_SGPRS = 2;

// How far a probe search runs before giving up, so a state that never probes fails the check
// instead of looping for ever.
constexpr uint32_t PROBE_SEARCH_LIMIT = 2 * PRESCAN_PROBE_INTERVAL;

// The static key words every item of these tests shares. Units differ by code hash and count.
const std::vector<uint32_t> STATIC_STATE {64u, 1u, 1u, 0x20u};

// An item of the program `code_hash` with dispatch key `key_hash` that loads `user_data_count`
// SGPRs.
PrescanItem ItemOf(uint64_t key_hash, uint64_t code_hash = CODE_HASH,
                   uint32_t user_data_count = USER_SGPRS) {
	PrescanItem item;
	item.key_hash        = key_hash;
	item.code_hash       = code_hash;
	item.user_data_count = user_data_count;
	return item;
}

// Asks the state about `item` and, when it is posted, notes a snapshot that gave `spec`. Returns
// the gate's answer.
IntState::Gate AdmitAndNote(IntState& state, const PrescanItem& item, int spec) {
	const auto admission = state.Admit(item, STATIC_STATE);
	if (admission.gate == IntState::Gate::Post) {
		(void)state.NoteSnapshot(admission, spec);
	}
	return admission.gate;
}

// Posts snapshots of `item` that all give the same specialization until its unit settles.
void SettleItem(IntState& state, const PrescanItem& item) {
	for (uint32_t index = 0; index <= PRESCAN_STALE_LIMIT; ++index) {
		CHECK(AdmitAndNote(state, item, 1) == IntState::Gate::Post);
	}
}

// Items come out in the order they went in, also once the indices have wrapped past the capacity.
// The baseline is the empty ring, which gives nothing, so a ring that made items up would fail.
void TestRingIsFirstInFirstOut() {
	SmallRing ring;
	uint64_t  out = 0;
	CHECK(!ring.TryPop(out));
	uint64_t next_in  = 1;
	uint64_t next_out = 1;
	// Three items in and two out each round walks the indices round the ring several times.
	for (uint32_t round = 0; round < 3 * SMALL_CAPACITY; ++round) {
		for (uint32_t push = 0; push < 3 && next_in - next_out < SMALL_CAPACITY; ++push) {
			CHECK(ring.TryPush(next_in++));
		}
		for (uint32_t pop = 0; pop < 2; ++pop) {
			CHECK(ring.TryPop(out) && out == next_out++);
		}
	}
	while (ring.TryPop(out)) {
		CHECK(out == next_out++);
	}
	CHECK(next_out == next_in);
	CHECK(ring.Dropped() == 0);
}

// A full ring refuses the next item and counts it, and keeps what it holds. The baseline is the
// pushes up to the capacity, which all succeed with nothing counted, so a ring that refused
// everything fails. A ring one slot too large takes the extra item and fails the count.
void TestFullRingDropsAndCounts() {
	SmallRing ring;
	for (uint64_t value = 1; value <= SMALL_CAPACITY; ++value) {
		CHECK(ring.TryPush(value));
	}
	CHECK(ring.Dropped() == 0);
	CHECK(!ring.TryPush(100));
	CHECK(!ring.TryPush(101));
	CHECK(ring.Dropped() == 2);
	// Taking one frees exactly one slot.
	uint64_t out = 0;
	CHECK(ring.TryPop(out) && out == 1);
	CHECK(ring.TryPush(SMALL_CAPACITY + 1));
	CHECK(!ring.TryPush(102));
	CHECK(ring.Dropped() == 3);
	for (uint64_t value = 2; value <= SMALL_CAPACITY + 1; ++value) {
		CHECK(ring.TryPop(out) && out == value);
	}
	CHECK(!ring.TryPop(out));
}

// An item wide enough that a torn read or an overwrite shows as words that disagree with its
// number.
struct StressItem {
	uint64_t                seq = 0;
	std::array<uint64_t, 7> words {};
};

constexpr uint64_t STRESS_ITEMS    = 2'000'000;
constexpr size_t   STRESS_CAPACITY = 64;
constexpr uint64_t STRESS_MIX      = 0x9E3779B97F4A7C15ull;
// The consumer stops for a moment once in this many items, so the producer fills the ring and the
// full path runs as well as the empty one.
constexpr uint64_t STRESS_PAUSE_EVERY = 4096;
constexpr auto     STRESS_PAUSE       = std::chrono::microseconds(50);

using StressRing = Common::SpscRing<StressItem, STRESS_CAPACITY>;

// The words an item with number `seq` carries.
StressItem MakeStressItem(uint64_t seq) {
	StressItem item;
	item.seq = seq;
	for (size_t index = 0; index < item.words.size(); ++index) {
		item.words[index] = seq * STRESS_MIX + index;
	}
	return item;
}

// What the consumer thread saw. Kept apart from CHECK, which only the main thread may use.
struct StressSeen {
	uint64_t received     = 0;
	uint64_t out_of_order = 0;
	uint64_t torn         = 0;
};

// Checks one received item against the last number seen.
void CheckStressItem(const StressItem& item, uint64_t& last_seq, StressSeen& seen) {
	seen.received++;
	seen.out_of_order += item.seq <= last_seq ? 1 : 0;
	seen.torn += item.words != MakeStressItem(item.seq).words ? 1 : 0;
	last_seq = item.seq;
}

// Pops until the producer is done and the ring is empty, pausing now and then.
void ConsumeStress(StressRing& ring, const std::atomic<bool>& producer_done, StressSeen& seen) {
	StressItem item;
	uint64_t   last_seq = 0;
	for (;;) {
		if (ring.TryPop(item)) {
			CheckStressItem(item, last_seq, seen);
			if (seen.received % STRESS_PAUSE_EVERY == 0) {
				const auto until = std::chrono::steady_clock::now() + STRESS_PAUSE;
				while (std::chrono::steady_clock::now() < until) {
				}
			}
			continue;
		}
		// The acquire pairs with the producer's release after its last push, so a drain after it
		// sees every item.
		if (producer_done.load(std::memory_order_acquire)) {
			while (ring.TryPop(item)) {
				CheckStressItem(item, last_seq, seen);
			}
			return;
		}
	}
}

// With one producer and one consumer running at once, every item is received once and in order or
// was refused and counted, and nothing else is lost. A received item is never torn. The baseline is
// that both paths ran, items received and items dropped, so a run that never filled the ring proves
// nothing about the full path and fails.
void TestRingStressTwoThreads() {
	auto              ring = std::make_unique<StressRing>();
	std::atomic<bool> producer_done {false};
	StressSeen        seen;
	std::thread       consumer([&] { ConsumeStress(*ring, producer_done, seen); });
	uint64_t          refused = 0;
	for (uint64_t seq = 1; seq <= STRESS_ITEMS; ++seq) {
		refused += ring->TryPush(MakeStressItem(seq)) ? 0 : 1;
	}
	producer_done.store(true, std::memory_order_release);
	consumer.join();
	CHECK(seen.out_of_order == 0);
	CHECK(seen.torn == 0);
	CHECK(refused == ring->Dropped());
	CHECK(seen.received + ring->Dropped() == STRESS_ITEMS);
	CHECK(seen.received > 0 && ring->Dropped() > 0);
}

// The scanner's whole path allocates nothing once the hand-off is built. That covers a ring that
// fills and drops, the drain and a skip that writes the scanner's cache. The counter is not stuck
// at zero, since a vector that grows in the same section is seen, and the drop and the skip did
// happen. The ring's slots are over-aligned, so an aligned allocation on its own is also checked
// to be counted, or a ring that allocated its slots late would go unseen.
void TestNoAllocationAfterConstruction() {
	auto          handoff = std::make_unique<PrescanHandoff>();
	DispatchCache cache(PRESCAN_PROBE_INTERVAL);
	PrescanItem   item = ItemOf(KEY);
	PrescanItem   out;
	g_allocation_count  = 0;
	g_count_allocations = true;
	for (uint64_t seq = 0; seq <= PRESCAN_RING_CAPACITY; ++seq) {
		item.seq = seq;
		(void)OfferToPreparer(*handoff, cache, item);
	}
	while (handoff->ring.TryPop(out)) {
	}
	handoff->feedback.Publish(KEY, PrescanFeedback::Answer::Settled);
	const auto skipped  = OfferToPreparer(*handoff, cache, item);
	g_count_allocations = false;
	CHECK(g_allocation_count == 0);
	CHECK(handoff->ring.Dropped() == 1);
	CHECK(skipped == PrescanOffer::Skipped);

	g_allocation_count  = 0;
	g_count_allocations = true;
	std::vector<uint32_t> grown;
	for (uint32_t index = 0; index < 100; ++index) {
		grown.push_back(index);
	}
	g_count_allocations = false;
	CHECK(g_allocation_count >= 1);

	g_allocation_count  = 0;
	g_count_allocations = true;
	auto aligned        = std::make_unique<PrescanItem>();
	// Publishing the pointer makes the allocation observable, so it cannot be optimised away.
	g_allocation_sink.store(aligned.get(), std::memory_order_relaxed);
	g_count_allocations = false;
	CHECK(g_allocation_count >= 1);
	g_allocation_sink.store(nullptr, std::memory_order_relaxed);
}

// The ring item carries what the preparer needs from a scanned dispatch, with the user data count
// taken from the registers the runtime reads it from. The baseline is a second dispatch that
// differs in every carried field, so a field the copy left at its default fails one of the two.
void TestItemCarriesTheDispatch() {
	for (const uint32_t variant: {0u, 1u}) {
		ScannedDispatch dispatch;
		dispatch.program_address            = 0x1000u + variant;
		dispatch.regs.cs_regs.user_sgpr     = static_cast<uint8_t>(4u + variant);
		dispatch.regs.cs_regs.wave_size     = variant == 0 ? 32 : 64;
		dispatch.regs.cs_user_sgpr.value[3] = 0xABCD0000u + variant;
		dispatch.key_hash                   = KEY + variant;
		dispatch.queue                      = 2u + variant;
		dispatch.uncertain_program          = variant == 1;
		dispatch.force_probe                = variant == 0;
		dispatch.workgroup_counts           = {3u + variant, 1u, 1u};
		dispatch.async_queue                = variant == 1;
		dispatch.indirect                   = variant == 0;
		dispatch.indirect_args_address      = 0x7000u + variant;
		const PrescanItem item              = MakePrescanItem(dispatch, 40u + variant);
		CHECK(item.inputs.async_compute == (variant == 1) &&
		      item.inputs.indirect == (variant == 0));
		CHECK(item.inputs.indirect_args_address == 0x7000u + variant);
		CHECK(item.program_address == 0x1000u + variant);
		CHECK(item.user_data_count == 4u + variant);
		CHECK(item.inputs.cs_regs.wave_size == (variant == 0 ? 32 : 64));
		CHECK(item.user_data.value[3] == 0xABCD0000u + variant);
		CHECK(item.key_hash == KEY + variant && item.seq == 40u + variant &&
		      item.queue == 2u + variant);
		CHECK(item.uncertain_program == (variant == 1) && item.force_probe == (variant == 0));
		CHECK(item.inputs.workgroup_counts[0] == 3u + variant);
		CHECK(item.code_hash == 0);
	}
}

// Two programs seen for the first time are both posted, so units follow the program. The baseline
// is the same program asked twice, which is held back as busy and published so, which rules out a
// gate that posts everything. A unit that ignored the code hash would hold the second program back.
void TestNewUnitPosts() {
	auto     handoff = std::make_unique<PrescanHandoff>();
	IntState state(*handoff);
	CHECK(state.Admit(ItemOf(KEY, CODE_HASH), STATIC_STATE).gate == IntState::Gate::Post);
	CHECK(state.Admit(ItemOf(OTHER_KEY, OTHER_CODE), STATIC_STATE).gate == IntState::Gate::Post);
	CHECK(state.Counters().posted == 2);
	CHECK(handoff->feedback.Read(KEY) == PrescanFeedback::Answer::None);

	CHECK(state.Admit(ItemOf(KEY, CODE_HASH), STATIC_STATE).gate == IntState::Gate::Busy);
	CHECK(state.Counters().busy == 1);
	CHECK(handoff->feedback.Read(KEY) == PrescanFeedback::Answer::Busy);
}

// Snapshots that keep giving the same specialization settle the unit, the gate holds the next item
// back and the scanner is told. The baseline is the same number of snapshots that ended with no
// specialization, which leaves the unit postable, so the settling comes from the noted snapshots
// and not from the number of posts.
void TestRepeatedSnapshotsSettle() {
	auto     handoff = std::make_unique<PrescanHandoff>();
	IntState state(*handoff);
	SettleItem(state, ItemOf(KEY));
	CHECK(state.Admit(ItemOf(KEY), STATIC_STATE).gate == IntState::Gate::Settled);
	CHECK(state.Counters().settled == 1);
	CHECK(state.Variants().SettledUnits() == 1);
	CHECK(handoff->feedback.Read(KEY) == PrescanFeedback::Answer::Settled);

	auto     baseline_handoff = std::make_unique<PrescanHandoff>();
	IntState baseline(*baseline_handoff);
	for (uint32_t index = 0; index <= PRESCAN_STALE_LIMIT; ++index) {
		const auto admission = baseline.Admit(ItemOf(KEY), STATIC_STATE);
		CHECK(admission.gate == IntState::Gate::Post);
		if (admission.gate == IntState::Gate::Post) {
			baseline.NoteDropped(admission);
		}
	}
	CHECK(baseline.Admit(ItemOf(KEY), STATIC_STATE).gate == IntState::Gate::Post);
	CHECK(baseline.Counters().settled == 0);
}

// A snapshot with a specialization the unit has not had is new, and the unit is posted again after
// it. The baseline is the same snapshot repeated, which is known, so a state that lost the
// specialization on the way to the tracker, and saw every snapshot as the same, fails the second
// note.
void TestNewSpecializationPostsAgain() {
	auto       handoff = std::make_unique<PrescanHandoff>();
	IntState   state(*handoff);
	const auto first = state.Admit(ItemOf(KEY), STATIC_STATE);
	CHECK(first.gate == IntState::Gate::Post);
	CHECK(state.NoteSnapshot(first, 100) == IntState::Outcome::New);
	const auto second = state.Admit(ItemOf(KEY), STATIC_STATE);
	CHECK(second.gate == IntState::Gate::Post &&
	      state.NoteSnapshot(second, 200) == IntState::Outcome::New);
	const auto third = state.Admit(ItemOf(KEY), STATIC_STATE);
	CHECK(third.gate == IntState::Gate::Post &&
	      state.NoteSnapshot(third, 200) == IntState::Outcome::Known);
}

// A unit at its cap is held back even when the item is the scanner's probe, each refusal is
// counted, and the scanner is told the key is settled. The baseline is the unit one specialization
// short of the cap, which is still posted, so the hold comes from the cap.
void TestCapHolds() {
	auto     handoff = std::make_unique<PrescanHandoff>();
	IntState state(*handoff);
	for (int spec = 0; spec + 1 < static_cast<int>(PRESCAN_VARIANT_CAP); ++spec) {
		CHECK(AdmitAndNote(state, ItemOf(KEY), spec) == IntState::Gate::Post);
	}
	CHECK(state.Variants().CappedUnits() == 0);
	CHECK(AdmitAndNote(state, ItemOf(KEY), static_cast<int>(PRESCAN_VARIANT_CAP)) ==
	      IntState::Gate::Post);
	CHECK(state.Variants().CappedUnits() == 1);

	CHECK(state.Admit(ItemOf(KEY), STATIC_STATE).gate == IntState::Gate::Capped);
	PrescanItem probe = ItemOf(KEY);
	probe.force_probe = true;
	CHECK(state.Admit(probe, STATIC_STATE).gate == IntState::Gate::Capped);
	CHECK(state.Counters().capped == 2);
	CHECK(handoff->feedback.Read(KEY) == PrescanFeedback::Answer::Settled);
}

// A settled unit lets one item in PRESCAN_PROBE_INTERVAL through as its own probe, and an item the
// scanner's cache sent as a probe through at once. The baseline is the plain item asked right after
// settling, which is held back, so the passes are the probes and not a unit that never settled.
void TestProbePassesAtInterval() {
	auto     handoff = std::make_unique<PrescanHandoff>();
	IntState state(*handoff);
	SettleItem(state, ItemOf(KEY));
	uint32_t held = 0;
	while (held < PROBE_SEARCH_LIMIT &&
	       state.Admit(ItemOf(KEY), STATIC_STATE).gate == IntState::Gate::Settled) {
		++held;
	}
	CHECK(held == PRESCAN_PROBE_INTERVAL - 1);

	auto     cache_handoff = std::make_unique<PrescanHandoff>();
	IntState cached(*cache_handoff);
	SettleItem(cached, ItemOf(KEY));
	CHECK(cached.Admit(ItemOf(KEY), STATIC_STATE).gate == IntState::Gate::Settled);
	PrescanItem probe = ItemOf(KEY);
	probe.force_probe = true;
	CHECK(cached.Admit(probe, STATIC_STATE).gate == IntState::Gate::Post);
}

// One program dispatched with two user data counts is two units, so neither is held back by the
// other's snapshot in flight. The baseline is the same program and count asked twice, which is held
// back, so a state that dropped the count from the unit fails.
void TestUserDataCountSeparatesUnits() {
	auto     handoff = std::make_unique<PrescanHandoff>();
	IntState state(*handoff);
	CHECK(state.Admit(ItemOf(KEY, CODE_HASH, 2), STATIC_STATE).gate == IntState::Gate::Post);
	CHECK(state.Admit(ItemOf(OTHER_KEY, CODE_HASH, 4), STATIC_STATE).gate == IntState::Gate::Post);
	CHECK(state.Variants().Units() == 2);
	CHECK(state.Admit(ItemOf(KEY, CODE_HASH, 2), STATIC_STATE).gate == IntState::Gate::Busy);
}

// An item whose program the walk could not vouch for is still asked about and posted, and is
// counted as uncertain. The baseline is a certain item, which leaves the count alone.
void TestUncertainItemsCounted() {
	auto     handoff = std::make_unique<PrescanHandoff>();
	IntState state(*handoff);
	CHECK(state.Admit(ItemOf(KEY), STATIC_STATE).gate == IntState::Gate::Post);
	CHECK(state.Counters().uncertain == 0);
	PrescanItem uncertain       = ItemOf(OTHER_KEY, OTHER_CODE);
	uncertain.uncertain_program = true;
	CHECK(state.Admit(uncertain, STATIC_STATE).gate == IntState::Gate::Post);
	CHECK(state.Counters().uncertain == 1);
	CHECK(state.Counters().posted == 2);
}

// Returns whether `call` threw an exception of exactly the type `Error`. A derived type does not
// count, so std::invalid_argument is not taken for std::logic_error.
template <class Error, class Call>
bool Throws(Call&& call) {
	try {
		call();
	} catch (const std::exception& error) {
		return typeid(error) == typeid(Error);
	}
	return false;
}

// Each outcome call uses up a posted admission, so a second outcome for it is refused, and so is an
// outcome for an admission of an earlier post of the same unit. The baseline is the first outcome
// for each, which passes, so the refusals are the reuse and not the call. A check of the gate
// alone, or of the unit's in-flight mark, would let the reuse through.
void TestAdmissionsAreSingleUse() {
	auto       handoff = std::make_unique<PrescanHandoff>();
	IntState   state(*handoff);
	const auto admit = [&] { return state.Admit(ItemOf(KEY), STATIC_STATE); };

	const auto snapshot = admit();
	CHECK(!Throws<std::logic_error>([&] { (void)state.NoteSnapshot(snapshot, 1); }));
	CHECK(Throws<std::logic_error>([&] { (void)state.NoteSnapshot(snapshot, 2); }));
	const auto dropped = admit();
	CHECK(!Throws<std::logic_error>([&] { state.NoteDropped(dropped); }));
	CHECK(Throws<std::logic_error>([&] { state.NoteDropped(dropped); }));
	const auto rejected = admit();
	CHECK(!Throws<std::logic_error>([&] { state.NoteRejected(rejected); }));
	CHECK(Throws<std::logic_error>([&] { state.NoteRejected(rejected); }));

	// An old admission is refused while a newer post of the unit is open, and the newer one passes.
	const auto newer = admit();
	CHECK(newer.gate == IntState::Gate::Post && newer.generation != dropped.generation);
	CHECK(Throws<std::logic_error>([&] { state.NoteDropped(dropped); }));
	CHECK(!Throws<std::logic_error>([&] { state.Retire(newer); }));
	CHECK(Throws<std::logic_error>([&] { state.Retire(newer); }));
}

// An outcome for an item that was not posted is refused, since it would clear another snapshot's
// in-flight mark. An item with no code hash is refused too. The baselines are the same calls for a
// posted item and an item with a hash, which pass.
void TestMisuseRefused() {
	auto       handoff = std::make_unique<PrescanHandoff>();
	IntState   state(*handoff);
	const auto posted = state.Admit(ItemOf(KEY), STATIC_STATE);
	const auto busy   = state.Admit(ItemOf(KEY), STATIC_STATE);
	CHECK(busy.gate == IntState::Gate::Busy);
	CHECK(Throws<std::logic_error>([&] { (void)state.NoteSnapshot(busy, 1); }));
	CHECK(Throws<std::logic_error>([&] { state.NoteDropped(busy); }));
	CHECK(!Throws<std::logic_error>([&] { (void)state.NoteSnapshot(posted, 1); }));
	CHECK(Throws<std::invalid_argument>([&] { (void)state.Admit(ItemOf(KEY, 0), STATIC_STATE); }));
	CHECK(!Throws<std::invalid_argument>(
	    [&] { (void)state.Admit(ItemOf(KEY, CODE_HASH), STATIC_STATE); }));
}

// A key the preparer answered settled is not sent again and goes into the scanner's cache, so the
// next dispatch of it is skipped during the walk. A busy answer does the same. The baseline is a
// key the preparer never answered, which is posted. Each check uses a fresh cache, so a key the
// cache remembers from its own post is not taken for one it remembers from an answer.
void TestScannerSkipsWhatThePreparerAnswered() {
	auto          handoff = std::make_unique<PrescanHandoff>();
	IntState      state(*handoff);
	DispatchCache posting_cache(PRESCAN_PROBE_INTERVAL);
	CHECK(OfferToPreparer(*handoff, posting_cache, ItemOf(OTHER_KEY, OTHER_CODE)) ==
	      PrescanOffer::Posted);

	SettleItem(state, ItemOf(KEY));
	CHECK(state.Admit(ItemOf(KEY), STATIC_STATE).gate == IntState::Gate::Settled);
	DispatchCache settled_cache(PRESCAN_PROBE_INTERVAL);
	CHECK(OfferToPreparer(*handoff, settled_cache, ItemOf(KEY)) == PrescanOffer::Skipped);
	CHECK(settled_cache.Check(KEY) == DispatchCache::Verdict::Skip);

	// The other program has its first snapshot in flight, so a second dispatch of it is answered
	// busy.
	PrescanItem taken;
	CHECK(handoff->ring.TryPop(taken) && taken.key_hash == OTHER_KEY);
	CHECK(state.Admit(taken, STATIC_STATE).gate == IntState::Gate::Post);
	CHECK(state.Admit(taken, STATIC_STATE).gate == IntState::Gate::Busy);
	DispatchCache busy_cache(PRESCAN_PROBE_INTERVAL);
	CHECK(OfferToPreparer(*handoff, busy_cache, ItemOf(OTHER_KEY, OTHER_CODE)) ==
	      PrescanOffer::Skipped);
	CHECK(busy_cache.Check(OTHER_KEY) == DispatchCache::Verdict::Skip);
}

// A probe is sent even for a settled key. A dispatch whose program the walk could not vouch for
// follows the answer like a certain one, but the cache never remembers it, whether it was skipped
// or posted. The baseline is the same key certain and not a probe, which is skipped and remembered,
// so the send comes from the probe flag and the forgetting from the uncertain flag.
void TestProbesSentAndUncertainNeverRemembered() {
	auto          handoff = std::make_unique<PrescanHandoff>();
	DispatchCache cache(PRESCAN_PROBE_INTERVAL);
	handoff->feedback.Publish(KEY, PrescanFeedback::Answer::Settled);

	PrescanItem uncertain       = ItemOf(KEY);
	uncertain.uncertain_program = true;
	CHECK(OfferToPreparer(*handoff, cache, uncertain) == PrescanOffer::Skipped);
	CHECK(cache.Check(KEY) == DispatchCache::Verdict::Pass);
	PrescanItem unanswered       = ItemOf(OTHER_KEY, OTHER_CODE);
	unanswered.uncertain_program = true;
	CHECK(OfferToPreparer(*handoff, cache, unanswered) == PrescanOffer::Posted);
	CHECK(cache.Check(OTHER_KEY) == DispatchCache::Verdict::Pass);

	PrescanItem probe = ItemOf(KEY);
	probe.force_probe = true;
	CHECK(OfferToPreparer(*handoff, cache, probe) == PrescanOffer::Posted);

	DispatchCache certain_cache(PRESCAN_PROBE_INTERVAL);
	CHECK(OfferToPreparer(*handoff, certain_cache, ItemOf(KEY)) == PrescanOffer::Skipped);
	CHECK(certain_cache.Check(KEY) == DispatchCache::Verdict::Skip);
}

// Feeds `repeats` dispatches of one item through the scanner's path as the walk would, which asks
// the cache before it reports a dispatch. Returns how many reached the ring.
uint32_t ScanRepeats(PrescanHandoff& handoff, DispatchCache& cache, const PrescanItem& item,
                     uint32_t repeats) {
	uint32_t posted = 0;
	for (uint32_t index = 0; index < repeats; ++index) {
		const auto verdict = cache.Check(item.key_hash);
		if (verdict == DispatchCache::Verdict::Skip) {
			continue;
		}
		PrescanItem offered = item;
		offered.force_probe = verdict == DispatchCache::Verdict::Probe;
		posted += OfferToPreparer(handoff, cache, offered) == PrescanOffer::Posted ? 1 : 0;
	}
	return posted;
}

// Many dispatches of one key before the preparer has answered it put one item on the ring, since a
// posted key is remembered as busy. The baseline is the same repeats of an uncertain dispatch,
// which the cache may not remember, and every one of them is posted, so the one post comes from the
// cache and not from the count of repeats.
void TestRepeatsBeforeAnswerPostOnce() {
	constexpr uint32_t REPEATS = 100;
	static_assert(REPEATS < PRESCAN_PROBE_INTERVAL, "the repeats must end before a probe is due");
	auto          handoff = std::make_unique<PrescanHandoff>();
	DispatchCache cache(PRESCAN_PROBE_INTERVAL);
	CHECK(ScanRepeats(*handoff, cache, ItemOf(KEY), REPEATS) == 1);

	PrescanItem uncertain       = ItemOf(OTHER_KEY, OTHER_CODE);
	uncertain.uncertain_program = true;
	CHECK(ScanRepeats(*handoff, cache, uncertain, REPEATS) == REPEATS);
}

// Takes the next item from the ring and asks the state about it, as the preparer does.
IntState::Admission TakeAndAdmit(PrescanHandoff& handoff, IntState& state) {
	PrescanItem taken;
	CHECK(handoff.ring.TryPop(taken));
	return state.Admit(taken, STATIC_STATE);
}

// Runs one key seen only while its program is uncertain through a post, a busy answer and the
// outcome `close`, and says whether a dispatch after the outcome reaches the ring. The same offer
// before the outcome is skipped, which is checked here as the baseline.
template <class Close>
bool UncertainKeyPostsAfterOutcome(Close&& close) {
	auto          handoff = std::make_unique<PrescanHandoff>();
	IntState      state(*handoff);
	DispatchCache cache(PRESCAN_PROBE_INTERVAL);
	PrescanItem   uncertain     = ItemOf(KEY);
	uncertain.uncertain_program = true;
	CHECK(OfferToPreparer(*handoff, cache, uncertain) == PrescanOffer::Posted);
	const auto posted = TakeAndAdmit(*handoff, state);
	CHECK(posted.gate == IntState::Gate::Post);
	CHECK(OfferToPreparer(*handoff, cache, uncertain) == PrescanOffer::Posted);
	CHECK(TakeAndAdmit(*handoff, state).gate == IntState::Gate::Busy);
	CHECK(OfferToPreparer(*handoff, cache, uncertain) == PrescanOffer::Skipped);
	close(state, posted);
	return OfferToPreparer(*handoff, cache, uncertain) == PrescanOffer::Posted;
}

// A busy answer is taken back when the post it waited on closes, whether the snapshot was noted or
// dropped. The scanner never caches an uncertain dispatch, so it never probes one, and a busy
// answer left behind would skip that key for good. The baseline is the offer before the outcome,
// which is skipped, so the post afterwards comes from the outcome.
void TestBusyAnswerClearsWhenPostCloses() {
	CHECK(UncertainKeyPostsAfterOutcome([](IntState& state, const IntState::Admission& admission) {
		(void)state.NoteSnapshot(admission, 1);
	}));
	CHECK(UncertainKeyPostsAfterOutcome([](IntState& state, const IntState::Admission& admission) {
		state.NoteDropped(admission);
	}));
}

// A dispatch dropped by a full ring is not remembered, so a later dispatch of its key may still be
// sent. The baseline is the same key offered to a ring with room, which is posted and remembered
// as busy, so the forgetting comes from the drop.
void TestFullRingDropNotRemembered() {
	auto          full = std::make_unique<PrescanHandoff>();
	DispatchCache filling_cache(PRESCAN_PROBE_INTERVAL);
	for (size_t index = 0; index < PRESCAN_RING_CAPACITY; ++index) {
		(void)OfferToPreparer(*full, filling_cache, ItemOf(KEY));
	}
	DispatchCache dropped_cache(PRESCAN_PROBE_INTERVAL);
	CHECK(OfferToPreparer(*full, dropped_cache, ItemOf(OTHER_KEY, OTHER_CODE)) ==
	      PrescanOffer::Dropped);
	CHECK(dropped_cache.Check(OTHER_KEY) == DispatchCache::Verdict::Pass);

	auto          roomy = std::make_unique<PrescanHandoff>();
	DispatchCache posted_cache(PRESCAN_PROBE_INTERVAL);
	CHECK(OfferToPreparer(*roomy, posted_cache, ItemOf(OTHER_KEY, OTHER_CODE)) ==
	      PrescanOffer::Posted);
	CHECK(posted_cache.Check(OTHER_KEY) == DispatchCache::Verdict::Skip);
}

// A dispatch offered to a full ring is dropped, and the preparer's counters show it. The baseline
// is the offers that filled the ring, which were all posted with nothing counted.
void TestFullRingCountedByState() {
	auto          handoff = std::make_unique<PrescanHandoff>();
	IntState      state(*handoff);
	DispatchCache cache(PRESCAN_PROBE_INTERVAL);
	uint64_t      posted = 0;
	for (size_t index = 0; index < PRESCAN_RING_CAPACITY; ++index) {
		posted += OfferToPreparer(*handoff, cache, ItemOf(KEY)) == PrescanOffer::Posted ? 1 : 0;
	}
	CHECK(posted == PRESCAN_RING_CAPACITY);
	CHECK(state.Counters().dropped_full == 0);
	CHECK(OfferToPreparer(*handoff, cache, ItemOf(KEY)) == PrescanOffer::Dropped);
	CHECK(state.Counters().dropped_full == 1);
}

// Two keys that share a slot in the answer table never read each other's answer, and clearing one
// leaves the other alone. The baseline is the key that was published, which reads its own answer.
void TestFeedbackKeysDoNotMix() {
	auto           handoff  = std::make_unique<PrescanHandoff>();
	auto&          feedback = handoff->feedback;
	const uint64_t sharer   = KEY + (uint64_t {1} << 40u);
	feedback.Publish(KEY, PrescanFeedback::Answer::Busy);
	CHECK(feedback.Read(KEY) == PrescanFeedback::Answer::Busy);
	CHECK(feedback.Read(sharer) == PrescanFeedback::Answer::None);
	feedback.Publish(sharer, PrescanFeedback::Answer::None);
	CHECK(feedback.Read(KEY) == PrescanFeedback::Answer::Busy);
	feedback.Publish(KEY, PrescanFeedback::Answer::None);
	CHECK(feedback.Read(KEY) == PrescanFeedback::Answer::None);
}

// Every answer reads back for keys whose two lowest bits, where the answer is packed, are each of
// 00, 01, 10 and 11, and clearing reads back as none. KEY ends in 00 and is the baseline, so a
// table that stored the key's own low bits beside the answer, or compared against the unmasked
// key, fails on the other three.
void TestFeedbackPackingKeepsEveryLowBitPattern() {
	auto  handoff  = std::make_unique<PrescanHandoff>();
	auto& feedback = handoff->feedback;
	static_assert((KEY & 3u) == 0, "the baseline key must end in 00");
	for (const uint64_t low_bits: {0u, 1u, 2u, 3u}) {
		const uint64_t key = KEY | low_bits;
		feedback.Publish(key, PrescanFeedback::Answer::Busy);
		CHECK(feedback.Read(key) == PrescanFeedback::Answer::Busy);
		feedback.Publish(key, PrescanFeedback::Answer::Settled);
		CHECK(feedback.Read(key) == PrescanFeedback::Answer::Settled);
		feedback.Publish(key, PrescanFeedback::Answer::None);
		CHECK(feedback.Read(key) == PrescanFeedback::Answer::None);
	}
}

// Two units that share a slot of the in-flight table. A is marked, B is marked over it, and clearing
// A then leaves B marked, since a unit clears only its own mark. A is no longer marked once B took
// the slot, which is the false negative the table allows and never a false positive. The baselines
// are each unit reading its own mark right after it was marked, and B reading clear after its own
// Clear, so a table that marked nothing or never cleared fails. Unit 0 means no unit, so marking it
// leaves the unit in slot 0 alone and reads as not marked. The table is a separate instance from the
// scanner's answer table, so an answer published for the same key does not read as in flight, and a
// unit marked does not read as an answer. The mutations are to clear without the key check, and
// clearing A wipes B, and to let unit 0 through, which replaces the unit in slot 0 and is caught.
void TestInFlightTableCollision() {
	auto           handoff = std::make_unique<PrescanHandoff>();
	auto&          table   = handoff->in_flight;
	const uint64_t unit_a  = KEY;
	const uint64_t unit_b  = KEY + (uint64_t {1} << 40u);
	CHECK(!table.Contains(unit_a) && !table.Contains(unit_b));
	table.Mark(unit_a);
	CHECK(table.Contains(unit_a) && !table.Contains(unit_b));
	table.Mark(unit_b);
	CHECK(table.Contains(unit_b) && !table.Contains(unit_a));
	table.Clear(unit_a);
	CHECK(table.Contains(unit_b));
	table.Clear(unit_b);
	CHECK(!table.Contains(unit_b));

	const uint64_t slot_zero_unit = uint64_t {1} << 40u;
	table.Mark(slot_zero_unit);
	table.Mark(0);
	CHECK(!table.Contains(0) && table.Contains(slot_zero_unit));
	table.Clear(0);
	CHECK(table.Contains(slot_zero_unit));

	handoff->feedback.Publish(OTHER_KEY, PrescanFeedback::Answer::Settled);
	CHECK(!table.Contains(OTHER_KEY));
	CHECK(handoff->feedback.Read(slot_zero_unit) == PrescanFeedback::Answer::None);
}

// The ring's producer slot goes to the first claim only, and stays taken. The first claim is the
// baseline, so a claim that always fails is caught there, and the third shows the second's refusal is
// not a one-off. The mutations are to return true always, and to test without storing, and both let
// the second claim through. That the test and the store are one atomic step is checked by review only.
void TestProducerClaimedOnce() {
	auto handoff = std::make_unique<PrescanHandoff>();
	CHECK(handoff->TryClaimProducer());
	CHECK(!handoff->TryClaimProducer());
	CHECK(!handoff->TryClaimProducer());
}

// What the GPU thread read for one inline compile, and the reason it must give.
struct MissRow {
	bool               noted;
	bool               in_flight;
	PrescanUnitVerdict verdict;
	bool               stopped;
	bool               deep;
	bool               indirect;
	bool               async;
	PrescanMiss        expected;
};

// The reason of one row, through the two steps the GPU thread takes.
PrescanMiss MissOf(const MissRow& row) {
	const auto unposted =
	    ClassifyUnposted(row.verdict, row.stopped, row.deep, row.indirect, row.async);
	return ClassifyMiss(row.noted, row.in_flight, unposted);
}

// Each reason wins over every reason below it. Row 0 has every fact set, with the miss not noted,
// and gives ModulePresent. Rows 1, 2, 4, 5, 6, 7 and 8 form a chain in which each row clears only
// the fact of the reason the row before it gave, so it gives the next reason down while every
// lower reason's fact is still set. Row 2 is Deferred with the prescan stopped and gives Stopped,
// since a stopped prescan's verdicts are history. Row 4 differs from it only in the stop and gives
// Refused, so it is row 2's baseline. Row 3 is the stop alone. Every verdict alone refuses. The
// mutation is the order with Refused above Stopped, which makes row 2 read Refused.
void TestMissPrecedence() {
	using enum PrescanUnitVerdict;
	using enum PrescanMiss;
	const std::array<MissRow, 9> rows {{
	    {false, true, Deferred, true, true, true, true, ModulePresent},
	    {true, true, Deferred, true, true, true, true, InFlight},
	    {true, false, Deferred, true, true, true, true, Stopped},
	    {true, false, None, true, false, false, false, Stopped},
	    {true, false, Deferred, false, true, true, true, Refused},
	    {true, false, None, false, true, true, true, NeverPostedDeep},
	    {true, false, None, false, false, true, true, NeverPostedIndirect},
	    {true, false, None, false, false, false, true, NeverPostedAsync},
	    {true, false, None, false, false, false, false, NeverPosted},
	}};
	for (const auto& row: rows) {
		CHECK(MissOf(row) == row.expected);
	}
	const auto last = static_cast<uint8_t>(PRESCAN_UNIT_VERDICT_LAST);
	for (auto code = static_cast<uint8_t>(Admitted); code <= last; ++code) {
		const auto verdict = static_cast<PrescanUnitVerdict>(code);
		CHECK(MissOf({true, false, verdict, false, false, false, false, Refused}) == Refused);
	}
}

// Every reason charges the cause the table gives, none charges Other, and each cause reserved for
// a miss is charged by exactly one reason, so its hold time means one thing. The explicit rows are
// the baseline the counts are checked beside, so a table that swapped two reserved causes keeps
// the counts at one each and still fails a row. The mutation is to map NeverPostedAsync to
// RequiredCompute, which fails its row and leaves PrescanNeverPostedAsync charged by none.
void TestMissCauseTable() {
	using Common::GpuStall::HoldCause;
	using enum PrescanMiss;
	const std::array<std::pair<PrescanMiss, HoldCause>, PRESCAN_MISS_COUNT> table {{
	    {ModulePresent, HoldCause::RequiredCompute},
	    {InFlight, HoldCause::PrescanInFlight},
	    {Stopped, HoldCause::RequiredCompute},
	    {Refused, HoldCause::PrescanRefused},
	    {NeverPostedDeep, HoldCause::PrescanNeverPostedDeep},
	    {NeverPostedIndirect, HoldCause::PrescanNeverPostedIndirect},
	    {NeverPostedAsync, HoldCause::PrescanNeverPostedAsync},
	    {NeverPosted, HoldCause::RequiredCompute},
	}};
	std::array<uint32_t, Common::GpuStall::HOLD_CAUSE_COUNT> charged_by {};
	for (const auto& [miss, cause]: table) {
		CHECK(HoldCauseOf(miss) == cause);
		CHECK(HoldCauseOf(miss) != HoldCause::Other);
		charged_by[static_cast<size_t>(HoldCauseOf(miss))]++;
	}
	for (const auto reserved:
	     {HoldCause::PrescanNeverPostedAsync, HoldCause::PrescanNeverPostedIndirect,
	      HoldCause::PrescanNeverPostedDeep, HoldCause::PrescanRefused,
	      HoldCause::PrescanInFlight}) {
		CHECK(charged_by[static_cast<size_t>(reserved)] == 1);
	}
}

// Each reason and each refusing verdict has the name the report writes, in its own slot. The names
// are written out here, so a table shifted by one, or two names swapped, fails. The verdicts start
// at Admitted, so None has no name and the slot of each verdict is its value less one. The
// mutation is to shift the reason names by one.
void TestMissNames() {
	using enum PrescanMiss;
	const std::array<std::pair<PrescanMiss, std::string_view>, PRESCAN_MISS_COUNT> reasons {{
	    {ModulePresent, "module_present"},
	    {InFlight, "in_flight"},
	    {Stopped, "stopped"},
	    {Refused, "refused"},
	    {NeverPostedDeep, "never_posted_deep"},
	    {NeverPostedIndirect, "never_posted_indirect"},
	    {NeverPostedAsync, "never_posted_async"},
	    {NeverPosted, "never_posted"},
	}};
	for (const auto& [miss, name]: reasons) {
		CHECK(PRESCAN_MISS_NAMES[static_cast<size_t>(miss)] == name);
	}
	using enum PrescanUnitVerdict;
	const std::array<std::pair<PrescanUnitVerdict, std::string_view>, PRESCAN_REFUSED_VERDICT_COUNT>
	    verdicts {{
	        {Admitted, "admitted"},
	        {Deferred, "deferred"},
	        {Built, "built"},
	        {Rejected, "rejected"},
	        {IndirectArgs, "indirect_args"},
	        {Retired, "retired"},
	        {CompileFailed, "compile_failed"},
	        {Capped, "capped"},
	        {Settled, "settled"},
	        {BackedOff, "backed_off"},
	        {CodeUnreadable, "code_unreadable"},
	        {CodeChanged, "code_changed"},
	    }};
	for (const auto& [verdict, name]: verdicts) {
		CHECK(PRESCAN_REFUSED_VERDICT_NAMES[RefusedVerdictIndex(verdict)] == name);
	}
}

} // namespace

// Counts the allocations of the tests, for TestNoAllocationAfterConstruction.
void* operator new(std::size_t size) {
	if (g_count_allocations.load(std::memory_order_relaxed)) {
		g_allocation_count.fetch_add(1, std::memory_order_relaxed);
	}
	if (void* memory = std::malloc(size == 0 ? 1 : size)) {
		return memory;
	}
	throw std::bad_alloc();
}

void operator delete(void* memory) noexcept {
	std::free(memory);
}

void operator delete(void* memory, std::size_t) noexcept {
	std::free(memory);
}

// The aligned forms are counted too, since a ring of PrescanItem slots is over-aligned and its
// storage comes from them. The block is cut from a larger malloc, and the pointer malloc gave sits
// just before the aligned address so that delete can find it.
void* operator new(std::size_t size, std::align_val_t alignment) {
	if (g_count_allocations.load(std::memory_order_relaxed)) {
		g_allocation_count.fetch_add(1, std::memory_order_relaxed);
	}
	const auto align = static_cast<std::size_t>(alignment);
	void*      raw   = std::malloc(size + align + sizeof(void*));
	if (raw == nullptr) {
		throw std::bad_alloc();
	}
	const auto address = (reinterpret_cast<std::uintptr_t>(raw) + sizeof(void*) + align - 1) &
	                     ~static_cast<std::uintptr_t>(align - 1);
	reinterpret_cast<void**>(address)[-1] = raw;
	return reinterpret_cast<void*>(address);
}

void operator delete(void* memory, std::align_val_t) noexcept {
	if (memory != nullptr) {
		std::free(static_cast<void**>(memory)[-1]);
	}
}

void operator delete(void* memory, std::size_t, std::align_val_t alignment) noexcept {
	operator delete(memory, alignment);
}

int main() {
	TestRingIsFirstInFirstOut();
	TestFullRingDropsAndCounts();
	TestRingStressTwoThreads();
	TestNoAllocationAfterConstruction();
	TestItemCarriesTheDispatch();
	TestNewUnitPosts();
	TestRepeatedSnapshotsSettle();
	TestNewSpecializationPostsAgain();
	TestCapHolds();
	TestProbePassesAtInterval();
	TestUserDataCountSeparatesUnits();
	TestUncertainItemsCounted();
	TestMisuseRefused();
	TestAdmissionsAreSingleUse();
	TestScannerSkipsWhatThePreparerAnswered();
	TestProbesSentAndUncertainNeverRemembered();
	TestRepeatsBeforeAnswerPostOnce();
	TestBusyAnswerClearsWhenPostCloses();
	TestFullRingDropNotRemembered();
	TestFullRingCountedByState();
	TestFeedbackKeysDoNotMix();
	TestFeedbackPackingKeepsEveryLowBitPattern();
	TestInFlightTableCollision();
	TestProducerClaimedOnce();
	TestMissPrecedence();
	TestMissCauseTable();
	TestMissNames();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("PrescanHandoffTests passed");
	return 0;
}
