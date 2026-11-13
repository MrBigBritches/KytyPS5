// Tests the prescan's scanner thread. The scanner reads indirect buffers through a reader the test
// supplies, and posts to a hand-off the test owns and drains, so no guest memory, preparer or device is
// needed. Each check that proves a behaviour has a baseline that differs only in that behaviour, as the
// comments say. Queue overflow eviction marking the victim's queue in doubt is covered for one queue by
// TestDropMarksShadowUncertain in SubmitScanTests, and here across two queues.
#include "graphics/guest_gpu/command_processor/prescanScanner.h"
#include "graphics/guest_gpu/command_processor/scanJobQueue.h"
#include "graphics/guest_gpu/command_processor/submitScan.h"
#include "graphics/guest_gpu/pm4.h"
#include "graphics/host_gpu/renderer/pipeline/prescanHandoff.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace std::chrono_literals;
using Clock = std::chrono::steady_clock;

int g_failures = 0;

// The bytes the calling thread has allocated through the operator new below. Each thread has its own
// count, so the scanner thread's figure is not muddied by what the test thread builds.
thread_local uint64_t t_allocated_bytes = 0;
thread_local uint64_t t_allocations     = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Stream = std::vector<uint32_t>;

// How the program address splits over COMPUTE_PGM_LO and COMPUTE_PGM_HI, and the bits PGM_HI holds.
constexpr uint32_t ADDRESS_LOW_SHIFT  = 8;
constexpr uint32_t ADDRESS_HIGH_SHIFT = 40;
constexpr uint64_t ADDRESS_HIGH_MASK  = ~((uint64_t {1} << ADDRESS_HIGH_SHIFT) - 1);
// Program addresses with a high byte, so that a later write of the low half alone shows whether the
// high half was reset.
constexpr uint64_t PROGRAM_A = 0x00001A0000010000ull;
constexpr uint64_t PROGRAM_B = 0x00002B0000020000ull;
// How far apart programs that must have different keys are placed.
constexpr uint64_t PROGRAM_STRIDE = 0x100;
// A low half written on its own, and the address it gives once the high half is zero.
constexpr uint32_t LOW_ONLY_VALUE   = 0x00000300u;
constexpr uint64_t LOW_ONLY_ADDRESS = uint64_t {LOW_ONLY_VALUE} << ADDRESS_LOW_SHIFT;
// The thread group size every test program declares.
constexpr uint32_t GROUP_SIZE_X       = 64;
constexpr uint32_t GROUP_SIZE_Y       = 1;
constexpr uint32_t GROUP_SIZE_Z       = 1;
constexpr uint32_t USER_SGPRS         = 2;
constexpr uint32_t RSRC2_USER_SHIFT   = 1;
constexpr uint32_t RSRC2_TGID_X_EN    = 1u << 7u;
constexpr uint32_t FILLER_WORD        = 0x80000000u;
constexpr uint32_t GRAPHICS_QUEUE     = 0;
constexpr uint32_t ASYNC_QUEUE        = 1;
constexpr uint32_t OTHER_ASYNC_QUEUE  = 2;
// How long a test waits for the scanner before it calls the scanner stuck.
constexpr auto WAIT_LIMIT = 10s;
// The lag bound of every test that is not about the bound. The scanner runs below normal priority, and
// a loaded machine can starve it for seconds, so the real bound would lag skip fresh jobs and fail
// checks that have nothing to do with it.
constexpr auto UNBOUNDED_LAG = std::chrono::hours {1};
// The lag bound of the tests about it. Only a job's arrival is moved back, so a long bound costs no
// time, and the fresh case at three quarters of it leaves five seconds for the scanner to reach it.
constexpr auto TEST_LAG_BOUND = 20s;

void Append(Stream& stream, const Stream& more) {
	stream.insert(stream.end(), more.begin(), more.end());
}

// A SET_SH_REG packet writing consecutive registers from `reg`.
Stream SetSh(uint32_t reg, const Stream& values) {
	Stream packet {KYTY_PM4(2 + values.size(), Pm4::IT_SET_SH_REG, 0), reg};
	Append(packet, values);
	return packet;
}

// The packets that make a compute program current, both halves of its address included.
Stream SetProgram(uint64_t address) {
	Stream stream;
	Append(stream, SetSh(Pm4::COMPUTE_PGM_LO, {static_cast<uint32_t>(address >> ADDRESS_LOW_SHIFT),
	                                           static_cast<uint32_t>(address >> ADDRESS_HIGH_SHIFT)}));
	Append(stream, SetSh(Pm4::COMPUTE_PGM_RSRC1, {0u, (USER_SGPRS << RSRC2_USER_SHIFT) | RSRC2_TGID_X_EN}));
	Append(stream, SetSh(Pm4::COMPUTE_NUM_THREAD_X, {GROUP_SIZE_X, GROUP_SIZE_Y, GROUP_SIZE_Z}));
	return stream;
}

// A write of the low half of the program address alone.
Stream SetLowHalf(uint32_t value) {
	return SetSh(Pm4::COMPUTE_PGM_LO, {value});
}

Stream Dispatch() {
	return {KYTY_PM4(5, Pm4::IT_DISPATCH_DIRECT, 0), 1u, 1u, 1u, 0u};
}

Stream IndirectBuffer(uint64_t address, uint32_t dwords) {
	return {KYTY_PM4(4, Pm4::IT_INDIRECT_BUFFER, 0), static_cast<uint32_t>(address),
	        static_cast<uint32_t>(address >> 32u), dwords};
}

// Waits by spinning, never by sleeping, so a short delay is not stretched by the timer resolution.
void SpinFor(Clock::duration delay) {
	const auto until = Clock::now() + delay;
	while (Clock::now() < until) {
	}
}

// Spins until `done` holds, yielding between looks, and false when the limit passed first.
template <typename Done>
bool SpinUntil(Done done, Clock::duration limit = WAIT_LIMIT) {
	const auto deadline = Clock::now() + limit;
	while (!done()) {
		if (Clock::now() > deadline) {
			return false;
		}
		std::this_thread::yield();
	}
	return true;
}

// A reader for lists that follow no buffer, which fails every read.
ScanMemoryReader NoMemory() {
	return [](uint64_t, void*, uint64_t) { return false; };
}

// Guest memory made of a few regions, which fails every read that is not inside one. An optional spin
// before each read makes it slow.
struct FakeMemory {
	std::map<uint64_t, Stream> regions;
	Clock::duration            delay {0};

	ScanMemoryReader Reader() const {
		return [this](uint64_t address, void* destination, uint64_t bytes) {
			SpinFor(delay);
			for (const auto& [base, words]: regions) {
				const uint64_t size = words.size() * sizeof(uint32_t);
				if (address >= base && address + bytes <= base + size) {
					std::memcpy(destination, reinterpret_cast<const uint8_t*>(words.data()) + (address - base),
					            bytes);
					return true;
				}
			}
			return false;
		};
	}
};

// One scanner over a hand-off the test owns. Both are on the heap, since the ring is a megabyte.
struct Harness {
	std::unique_ptr<PrescanHandoff> handoff = std::make_unique<PrescanHandoff>();
	std::unique_ptr<PrescanScanner> scanner;

	explicit Harness(ScanMemoryReader reader, std::chrono::microseconds lag_bound = UNBOUNDED_LAG)
	    : scanner(std::make_unique<PrescanScanner>(*handoff, std::move(reader), lag_bound)) {}

	// Copies and pushes a list as a submitting thread does, made to look `age` old.
	uint64_t Submit(uint32_t queue, const Stream& list, Clock::duration age = Clock::duration::zero()) {
		auto prepared = scanner->PrepareCommands(queue, list);
		prepared.job.arrival -= age;
		return scanner->Push(std::move(prepared.job), prepared.copy_us);
	}

	// Pushes a reset marker as a flip suspend point does, made to look `age` old.
	uint64_t Reset(uint32_t queue, Clock::duration age = Clock::duration::zero()) {
		auto job = PrescanScanner::PrepareReset(queue);
		job.arrival -= age;
		return scanner->Push(std::move(job), 0.0);
	}

	// Waits until the thread has finished `jobs` jobs.
	bool WaitForJobs(uint64_t jobs) const {
		return SpinUntil([&] { return scanner->Stats().jobs >= jobs; });
	}

	// Takes everything the scanner posted.
	std::vector<PrescanItem> Drain() const {
		std::vector<PrescanItem> items;
		PrescanItem              item;
		while (handoff->ring.TryPop(item)) {
			items.push_back(item);
		}
		return items;
	}
};

// A dispatch with its program set in the same list is posted once, with the program, queue and job
// number the walk saw. The job is the queue's second, so a number stuck at 1 shows. The baseline is the
// same program with no dispatch, which is walked and posts nothing, so the item comes from the dispatch
// and not from the program write. Nothing else in the test writes to the ring.
void TestScannerPostsDispatch() {
	Harness with(NoMemory());
	with.scanner->Start();
	(void)with.Submit(OTHER_ASYNC_QUEUE, SetProgram(PROGRAM_B));
	Stream list = SetProgram(PROGRAM_A);
	Append(list, Dispatch());
	const uint64_t seq = with.Submit(OTHER_ASYNC_QUEUE, list);
	CHECK(seq == 2);
	CHECK(with.WaitForJobs(2));
	const auto items = with.Drain();
	CHECK(items.size() == 1);
	if (!items.empty()) {
		CHECK(items[0].program_address == PROGRAM_A);
		CHECK(items[0].queue == OTHER_ASYNC_QUEUE && items[0].inputs.async_compute);
		CHECK(items[0].seq == seq && !items[0].uncertain_program);
	}
	CHECK(with.scanner->Stats().offers_posted == 1);

	Harness without(NoMemory());
	without.scanner->Start();
	(void)without.Submit(OTHER_ASYNC_QUEUE, SetProgram(PROGRAM_A));
	CHECK(without.WaitForJobs(1));
	CHECK(without.scanner->Stats().scan.submissions == 1);
	CHECK(without.Drain().empty() && without.scanner->Stats().offers_posted == 0);
}

// The sum of a histogram's buckets.
uint64_t Total(const PrescanHistogram& histogram) {
	return std::accumulate(histogram.begin(), histogram.end(), uint64_t {0});
}

// Every push lands in the push histogram, every walked list in the scan and lag histograms, and a
// reset marker only in the push histogram, since it is neither walked nor late. Nothing else adds to
// them, so each total is the count of its own events, and a histogram left unfed shows as zero.
void TestStatsHistogramsCount() {
	Harness harness(NoMemory());
	for (uint32_t list = 0; list < 3; ++list) {
		(void)harness.Submit(ASYNC_QUEUE, SetProgram(PROGRAM_A + list * PROGRAM_STRIDE));
	}
	(void)harness.Reset(ASYNC_QUEUE);
	harness.scanner->Start();
	CHECK(harness.WaitForJobs(4));
	const auto stats = harness.scanner->Stats();
	CHECK(Total(stats.push_us_histogram) == 4);
	CHECK(Total(stats.scan_us_histogram) == 3 && Total(stats.lag_ms_histogram) == 3);
	CHECK(std::accumulate(stats.queue.copy_us_histogram.begin(), stats.queue.copy_us_histogram.end(),
	                      uint64_t {0}) == 3);
}

// Three dispatches of three programs in one list wake the preparer once. The baseline is the same
// programs with no dispatch, which wakes it not at all, so the one wake belongs to the posts and the
// scanner does not wake for every walk. A wake per item would show as three.
void TestWakeOncePerSubmission() {
	const auto wakes_for = [](bool dispatch) {
		Harness harness(NoMemory());
		Stream  list;
		for (const uint64_t program: {PROGRAM_A, PROGRAM_B, PROGRAM_A + PROGRAM_STRIDE}) {
			Append(list, SetProgram(program));
			if (dispatch) {
				Append(list, Dispatch());
			}
		}
		const uint32_t before = harness.handoff->wake_epoch.load();
		harness.scanner->Start();
		(void)harness.Submit(ASYNC_QUEUE, list);
		CHECK(harness.WaitForJobs(1));
		CHECK(harness.Drain().size() == (dispatch ? 3u : 0u));
		return harness.handoff->wake_epoch.load() - before;
	};
	CHECK(wakes_for(true) == 1);
	CHECK(wakes_for(false) == 0);
}

// Once the preparer has stopped, the next job ends the thread without a walk, counts it, and stops the
// queue, so a later copy is refused. The baseline is the same list with the flag clear, which posts its
// dispatch and leaves the thread running and the queue copying.
void TestPreparerStoppedStopsScanner() {
	Stream list = SetProgram(PROGRAM_A);
	Append(list, Dispatch());

	Harness stopped(NoMemory());
	stopped.handoff->preparer_stopped.store(true);
	stopped.scanner->Start();
	(void)stopped.Submit(ASYNC_QUEUE, list);
	CHECK(SpinUntil([&] { return stopped.scanner->Finished(); }));
	CHECK(stopped.Drain().empty());
	CHECK(stopped.scanner->Stats().stopped_by_preparer == 1);
	auto late = stopped.scanner->PrepareCommands(ASYNC_QUEUE, list);
	CHECK(!late.job.copied);
	CHECK(stopped.scanner->Push(std::move(late.job), late.copy_us) == 0);

	Harness running(NoMemory());
	running.scanner->Start();
	(void)running.Submit(ASYNC_QUEUE, list);
	CHECK(running.WaitForJobs(1));
	CHECK(running.Drain().size() == 1);
	CHECK(!running.scanner->Finished() && running.scanner->Stats().stopped_by_preparer == 0);
	auto later = running.scanner->PrepareCommands(ASYNC_QUEUE, list);
	CHECK(later.job.copied);
	(void)running.scanner->Push(std::move(later.job), later.copy_us);
}

// Whether the dispatch of a first list that writes only the low half of the program is uncertain, with
// or without a reset marker before it.
bool FirstLowHalfDispatchIsUncertain(bool reset_first) {
	Harness harness(NoMemory());
	if (reset_first) {
		(void)harness.Reset(ASYNC_QUEUE);
	}
	Stream list = SetLowHalf(LOW_ONLY_VALUE);
	Append(list, Dispatch());
	(void)harness.Submit(ASYNC_QUEUE, list);
	harness.scanner->Start();
	CHECK(harness.WaitForJobs(reset_first ? 2 : 1));
	const auto items = harness.Drain();
	CHECK(items.size() == 1);
	CHECK(items.empty() || items[0].program_address == LOW_ONLY_ADDRESS);
	return items.empty() || items[0].uncertain_program;
}

// A scanner that starts mid-run has not seen the high half of the program, so a first list that writes
// only the low half dispatches an uncertain program. The baseline puts a reset marker first, which
// makes the shadow known, and the same list is then certain, so the doubt comes from starting and not
// from the list.
void TestAttachMarksShadowUncertain() {
	CHECK(FirstLowHalfDispatchIsUncertain(false));
	CHECK(!FirstLowHalfDispatchIsUncertain(true));
}

// What the third list's dispatch looked like after a second list of a given age set program B.
struct InheritOutcome {
	uint64_t program     = 0;
	bool     uncertain   = false;
	uint64_t lag_skipped = 0;
	uint64_t walked      = 0;
};

// Sets program A, then program B in a list `age` old, then dispatches with whatever program is current.
InheritOutcome DispatchAfterAgedProgram(Clock::duration age) {
	Harness harness(NoMemory(), TEST_LAG_BOUND);
	(void)harness.Submit(ASYNC_QUEUE, SetProgram(PROGRAM_A));
	(void)harness.Submit(ASYNC_QUEUE, SetProgram(PROGRAM_B), age);
	(void)harness.Submit(ASYNC_QUEUE, Dispatch());
	harness.scanner->Start();
	CHECK(harness.WaitForJobs(3));
	const auto items = harness.Drain();
	const auto stats = harness.scanner->Stats();
	CHECK(items.size() == 1);
	InheritOutcome outcome {.lag_skipped = stats.lag_skipped, .walked = stats.scan.submissions};
	if (!items.empty()) {
		outcome.program   = items[0].program_address;
		outcome.uncertain = items[0].uncertain_program;
	}
	return outcome;
}

// A list twice the lag bound old is not walked, is counted, and leaves the queue in doubt, so the next
// list's dispatch, which inherits its program, is uncertain and still names program A. The baseline is
// the same list three quarters of the bound old, which leaves a quarter of the bound, five seconds, as
// margin for the scanner to reach it. It is walked, so the dispatch names program B and is certain. A
// skip that forgot the doubt would leave program A certain.
void TestLagBoundSkipsStaleJob() {
	const auto stale = DispatchAfterAgedProgram(2 * TEST_LAG_BOUND);
	CHECK(stale.lag_skipped == 1 && stale.walked == 2);
	CHECK(stale.program == PROGRAM_A && stale.uncertain);

	const auto fresh = DispatchAfterAgedProgram(TEST_LAG_BOUND * 3 / 4);
	CHECK(fresh.lag_skipped == 0 && fresh.walked == 3);
	CHECK(fresh.program == PROGRAM_B && !fresh.uncertain);
}

// Sets program A, then sends a reset marker `age` old, or none, then a list that writes only the low half
// of the program and dispatches. Returns that dispatch, and whether anything was lag skipped.
std::pair<PrescanItem, uint64_t> DispatchAfterReset(std::optional<Clock::duration> age) {
	Harness harness(NoMemory(), TEST_LAG_BOUND);
	(void)harness.Submit(ASYNC_QUEUE, SetProgram(PROGRAM_A));
	if (age) {
		(void)harness.Reset(ASYNC_QUEUE, *age);
	}
	Stream list = SetLowHalf(LOW_ONLY_VALUE);
	Append(list, Dispatch());
	(void)harness.Submit(ASYNC_QUEUE, list);
	harness.scanner->Start();
	CHECK(harness.WaitForJobs(age ? 3 : 2));
	const auto items = harness.Drain();
	CHECK(items.size() == 1);
	return {items.empty() ? PrescanItem {} : items[0], harness.scanner->Stats().lag_skipped};
}

// A reset marker twice the lag bound old still resets the shadow. The dispatch after it has the low half
// alone and a zero high half, and is certain, exactly as after a fresh marker, which is the baseline.
// Without any marker the same list keeps program A's high half, which shows the check tells a reset from
// none. A marker skipped like a stale list would leave the high half and the doubt.
void TestStaleResetStaysCertain() {
	const auto [stale, stale_skipped] = DispatchAfterReset(2 * TEST_LAG_BOUND);
	CHECK(stale.program_address == LOW_ONLY_ADDRESS && !stale.uncertain_program && stale_skipped == 0);

	const auto [fresh, fresh_skipped] = DispatchAfterReset(Clock::duration::zero());
	CHECK(fresh.program_address == LOW_ONLY_ADDRESS && !fresh.uncertain_program && fresh_skipped == 0);

	const auto [none, none_skipped] = DispatchAfterReset(std::nullopt);
	CHECK(none.program_address == ((PROGRAM_A & ADDRESS_HIGH_MASK) | LOW_ONLY_ADDRESS));
	CHECK(!none.uncertain_program && none_skipped == 0);
}

// Four buffers that each set a program and dispatch, behind a reader that spins half a millisecond per
// read, so the whole walk takes about two milliseconds.
constexpr uint32_t SLOW_BUFFERS       = 4;
constexpr auto     SLOW_READ          = 500us;
constexpr uint64_t SLOW_BUFFER_BASE   = 0xA00000;
constexpr uint64_t SLOW_BUFFER_STRIDE = 0x1000;

Stream SlowList(FakeMemory& memory) {
	Stream list;
	for (uint32_t index = 0; index < SLOW_BUFFERS; ++index) {
		const uint64_t address = SLOW_BUFFER_BASE + index * SLOW_BUFFER_STRIDE;
		Stream         buffer  = SetProgram(PROGRAM_A + index * PROGRAM_STRIDE);
		Append(buffer, Dispatch());
		memory.regions[address] = buffer;
		Append(list, IndirectBuffer(address, static_cast<uint32_t>(buffer.size())));
	}
	return list;
}

// A fresh list gets what is left of the lag bound, not the old 1 ms cap, so a 2 ms walk finds all four
// dispatches and is not capped. Each read is timed, one call per buffer and none shorter than its spin.
// The baseline is the same list and reader in a scanner whose cap is set to 1 ms, which is capped
// after two reads, so the slow reader really does outlast the old cap.
void TestFreshJobWalksPastOneMillisecond() {
	FakeMemory memory;
	memory.delay      = SLOW_READ;
	const Stream list = SlowList(memory);

	Harness harness(memory.Reader());
	harness.scanner->Start();
	(void)harness.Submit(ASYNC_QUEUE, list);
	CHECK(harness.WaitForJobs(1));
	CHECK(harness.Drain().size() == SLOW_BUFFERS);
	const auto walked = harness.scanner->Stats();
	CHECK(walked.scan.time_capped == 0);
	CHECK(walked.reads.calls == SLOW_BUFFERS);
	CHECK(walked.reads.max_us >= static_cast<uint64_t>(SLOW_READ.count()));

	SubmitScanner old_cap(memory.Reader());
	old_cap.SetTimeCap(1ms);
	CsShadow                     shadow;
	std::vector<ScannedDispatch> found;
	ScanStats                    stats;
	old_cap.Scan(list, ASYNC_QUEUE, true, shadow, found, stats);
	CHECK(stats.time_capped == 1 && found.size() < SLOW_BUFFERS);
}

// A read that spins twice the slow threshold is counted as slow. The fast reads of the other tests are
// never checked for zero, since a preempted thread can make any read slow.
constexpr auto VERY_SLOW_READ = 2 * SCAN_SLOW_READ;

void TestSlowReadCounted() {
	FakeMemory memory;
	memory.delay      = VERY_SLOW_READ;
	const Stream list = SlowList(memory);
	Harness      harness(memory.Reader());
	harness.scanner->Start();
	(void)harness.Submit(ASYNC_QUEUE, list);
	CHECK(harness.WaitForJobs(1));
	const auto reads = harness.scanner->Stats().reads;
	CHECK(reads.calls == SLOW_BUFFERS && reads.slow >= 1);
	CHECK(reads.max_us >= static_cast<uint64_t>(std::chrono::microseconds(VERY_SLOW_READ).count()));
}

// The calls a reader saw, by address and size.
struct ReadLog {
	std::vector<std::pair<uint64_t, uint64_t>> calls;
	// The call, counted from 1, that fails, or 0 for none.
	size_t fail_at = 0;

	ScanMemoryReader Reader(const std::vector<uint8_t>& source, uint64_t base) {
		return [this, &source, base](uint64_t address, void* destination, uint64_t bytes) {
			calls.emplace_back(address, bytes);
			if (calls.size() == fail_at || address < base || address + bytes > base + source.size()) {
				return false;
			}
			std::memcpy(destination, source.data() + (address - base), bytes);
			return true;
		};
	}
};

// A read of 256 KiB and 4 bytes is five calls of at most 64 KiB that follow each other and fill the
// destination exactly. The baseline is a read of 64 KiB, which is one call, so the split comes from
// the size. A failing third chunk ends the read there with false, and a set stop flag ends it before
// any call, where the same read with the flag clear succeeds.
void TestChunkedReadSplits() {
	constexpr uint64_t BASE  = 0x40000000ull;
	constexpr uint64_t BYTES = 256 * 1024 + 4;
	// A byte pattern that does not repeat at the chunk size, so a chunk copied to the wrong place shows.
	constexpr uint32_t PATTERN_STEP   = 7;
	constexpr uint32_t PATTERN_OFFSET = 3;
	std::vector<uint8_t> source(BYTES);
	for (size_t index = 0; index < source.size(); ++index) {
		source[index] = static_cast<uint8_t>(index * PATTERN_STEP + PATTERN_OFFSET);
	}
	std::vector<uint8_t> destination(BYTES, 0);
	ReadLog              split;
	ScanReadTiming       timing;
	CHECK(ChunkedScanRead(split.Reader(source, BASE), BASE, destination.data(), BYTES, nullptr, &timing));
	CHECK(split.calls.size() == 5 && timing.calls == 5);
	uint64_t next = BASE;
	for (const auto& [address, bytes]: split.calls) {
		CHECK(address == next && bytes <= SCAN_READ_CHUNK_BYTES);
		next += bytes;
	}
	CHECK(next == BASE + BYTES && destination == source);

	ReadLog one;
	CHECK(ChunkedScanRead(one.Reader(source, BASE), BASE, destination.data(), SCAN_READ_CHUNK_BYTES));
	CHECK(one.calls.size() == 1);

	ReadLog failing;
	failing.fail_at = 3;
	CHECK(!ChunkedScanRead(failing.Reader(source, BASE), BASE, destination.data(), BYTES));
	CHECK(failing.calls.size() == 3);

	std::atomic<bool> stop {true};
	ReadLog           stopped;
	CHECK(!ChunkedScanRead(stopped.Reader(source, BASE), BASE, destination.data(), BYTES, &stop));
	CHECK(stopped.calls.empty());
	stop = false;
	CHECK(ChunkedScanRead(stopped.Reader(source, BASE), BASE, destination.data(), BYTES, &stop));
}

// The scanner wraps the reader it is given, so a 256 KiB indirect buffer reaches that reader as four
// calls of at most 64 KiB, and the reads are counted in the stats. A scanner that used the reader as it
// was given would show one call of 256 KiB.
void TestScannerChunksBufferReads() {
	constexpr uint64_t BASE    = 0x50000000ull;
	constexpr uint32_t DWORDS  = 64 * 1024;
	std::vector<uint8_t> source(size_t {DWORDS} * sizeof(uint32_t));
	for (size_t word = 0; word < DWORDS; ++word) {
		std::memcpy(source.data() + word * sizeof(uint32_t), &FILLER_WORD, sizeof(uint32_t));
	}
	ReadLog log;
	Harness harness(log.Reader(source, BASE));
	harness.scanner->Start();
	(void)harness.Submit(ASYNC_QUEUE, IndirectBuffer(BASE, DWORDS));
	CHECK(harness.WaitForJobs(1));
	CHECK(harness.scanner->Stats().scan.ib_followed == 1);
	CHECK(log.calls.size() == 4 && harness.scanner->Stats().reads.calls == 4);
	CHECK(std::ranges::all_of(log.calls, [](const auto& call) { return call.second <= SCAN_READ_CHUNK_BYTES; }));
}

// Stop returns while the thread waits for a job, and a second Stop does nothing. The thread is checked
// to be still running first, since Stop on a thread that had already ended would prove nothing. Stop is
// run on a thread of its own under a watchdog, since a Stop that only joined would never return and the
// test could not go on. A scanner that never started stops as well, twice.
void TestStopIsPromptAndIdempotent() {
	Harness harness(NoMemory());
	harness.scanner->Start();
	(void)harness.Submit(ASYNC_QUEUE, SetProgram(PROGRAM_A));
	CHECK(harness.WaitForJobs(1));
	CHECK(!harness.scanner->Finished());
	std::atomic<bool> stopped {false};
	std::thread       stopper([&] {
        harness.scanner->Stop();
        stopped = true;
    });
	if (!SpinUntil([&] { return stopped.load(); })) {
		std::fprintf(stderr, "%s:%d: Stop did not return within the watchdog\n", __FILE__, __LINE__);
		std::_Exit(1);
	}
	stopper.join();
	harness.scanner->Stop();
	CHECK(harness.scanner->Finished());

	Harness never(NoMemory());
	never.scanner->Stop();
	never.scanner->Stop();
	CHECK(!never.scanner->Finished());
}

// Whether Start throws std::logic_error.
bool StartRefused(PrescanScanner& scanner) {
	try {
		scanner.Start();
	} catch (const std::logic_error&) {
		return true;
	}
	return false;
}

// Start after Stop is refused, since the stopped queue would end the new thread at once and it would
// look like a running scanner. The baseline is a fresh scanner, whose Start is accepted, and a second
// Start on it is refused as well.
void TestStartAfterStopRefused() {
	Harness stopped(NoMemory());
	stopped.scanner->Stop();
	CHECK(StartRefused(*stopped.scanner));
	CHECK(!stopped.scanner->Finished());

	Harness fresh(NoMemory());
	CHECK(!StartRefused(*fresh.scanner));
	CHECK(StartRefused(*fresh.scanner));
}

// Stop ends a walk that is in the middle of a long buffer read at the next chunk, so it waits for one
// chunk and not the whole buffer. The reader holds its first call until the queue is seen stopped,
// which Stop does after it sets the flag, so the order is fixed and no timing is involved. The baseline
// is TestScannerChunksBufferReads, where the same 256 KiB buffer takes four calls when nothing stops it.
void TestStopEndsLongRead() {
	constexpr uint64_t BASE   = 0x70000000ull;
	constexpr uint32_t DWORDS = 64 * 1024;
	std::atomic<bool>     in_read {false};
	std::atomic<bool>     release {false};
	std::atomic<uint32_t> calls {0};
	Harness               harness([&](uint64_t, void* destination, uint64_t bytes) {
        calls++;
        in_read = true;
        while (!release.load()) {
            std::this_thread::yield();
        }
        auto* words = static_cast<uint32_t*>(destination);
        std::fill(words, words + bytes / sizeof(uint32_t), FILLER_WORD);
        return true;
    });
	harness.scanner->Start();
	(void)harness.Submit(ASYNC_QUEUE, IndirectBuffer(BASE, DWORDS));
	CHECK(SpinUntil([&] { return in_read.load(); }));
	std::thread stopper([&] { harness.scanner->Stop(); });
	const bool queue_stopped = SpinUntil([&] {
		auto probe = harness.scanner->PrepareCommands(ASYNC_QUEUE, Stream {1u});
		const bool refused = !probe.job.copied;
		(void)harness.scanner->Push(std::move(probe.job), probe.copy_us);
		return refused;
	});
	CHECK(queue_stopped);
	release = true;
	stopper.join();
	CHECK(calls == 1 && harness.scanner->Finished());
	CHECK(harness.scanner->Stats().scan.ib_unreadable == 1);
}

// What the scanner thread had allocated when it last called the reader, and how many times it called
// it. The reader runs on the scanner thread, so it can read that thread's own count.
std::atomic<uint64_t> g_scanner_bytes {0};
std::atomic<uint64_t> g_scanner_allocations {0};
std::atomic<uint64_t> g_snapshots {0};
constexpr uint64_t    WARM_BUFFER = 0x60000000ull;
constexpr uint32_t    WARM_DWORDS = 4;
// The low half of the first steady program. Each dispatch takes the next one, so their keys differ.
constexpr uint32_t FIRST_LOW_HALF = 0x1000;
// How many dispatches each steady job has. More than the first job's one, so dispatch storage that was
// not reserved grows during them, and few enough that every post fits in the ring.
constexpr uint32_t STEADY_DISPATCHES = 40;
constexpr uint32_t STEADY_JOBS       = 100;

// Notes the scanner thread's allocations and fills the buffer with filler packets. It allocates nothing.
bool SnapshotReader(uint64_t, void* destination, uint64_t bytes) {
	g_snapshots++;
	g_scanner_bytes       = t_allocated_bytes;
	g_scanner_allocations = t_allocations;
	auto* words           = static_cast<uint32_t*>(destination);
	std::fill(words, words + bytes / sizeof(uint32_t), FILLER_WORD);
	return true;
}

// A list that calls the snapshot buffer and then dispatches `dispatches` programs. Only the low half of
// each program is written, so every dispatch stays uncertain and the settled cache never skips one.
Stream SnapshotList(uint32_t dispatches) {
	Stream list = IndirectBuffer(WARM_BUFFER, WARM_DWORDS);
	for (uint32_t index = 0; index < dispatches; ++index) {
		Append(list, SetLowHalf(FIRST_LOW_HALF + index));
		Append(list, Dispatch());
	}
	return list;
}

// After its first job the scanner thread allocates nothing, over a hundred jobs that walk, post and
// wake. Its count is read on its own thread, through the reader, and a job with only the buffer call
// comes last so the count covers the whole of the job before it. The baseline is the first job, which
// allocates the walk's 4 MB arena, so the count does see the scanner thread. Dispatch storage that was
// not reserved would grow in the first job with more dispatches than the first. The count only moves
// when the reader runs, so every job after warm-up must have read its buffer, or an unchanged count
// could just mean nothing was read.
void TestScannerNoAllocationAfterWarmup() {
	Harness harness(SnapshotReader);
	harness.scanner->Start();
	(void)harness.Submit(ASYNC_QUEUE, SnapshotList(1));
	(void)harness.Submit(ASYNC_QUEUE, SnapshotList(0));
	CHECK(harness.WaitForJobs(2));
	const uint64_t warm_bytes       = g_scanner_bytes;
	const uint64_t warm_allocations = g_scanner_allocations;
	const uint64_t warm_snapshots   = g_snapshots;
	CHECK(warm_bytes >= uint64_t {SCAN_DWORD_BUDGET} * sizeof(uint32_t));

	const Stream steady = SnapshotList(STEADY_DISPATCHES);
	for (uint32_t job = 0; job < STEADY_JOBS; ++job) {
		(void)harness.Submit(ASYNC_QUEUE, steady);
	}
	(void)harness.Submit(ASYNC_QUEUE, SnapshotList(0));
	CHECK(harness.WaitForJobs(STEADY_JOBS + 3));
	CHECK(g_snapshots - warm_snapshots == STEADY_JOBS + 1);
	CHECK(g_scanner_allocations == warm_allocations && g_scanner_bytes == warm_bytes);
	const auto stats = harness.scanner->Stats();
	CHECK(stats.scan.ib_followed == STEADY_JOBS + 3);
	CHECK(stats.offers_posted == 1 + uint64_t {STEADY_JOBS} * STEADY_DISPATCHES);
	CHECK(stats.wakes == 1 + STEADY_JOBS && stats.queue.dropped == 0);
}

// NoteBegan reaches the queue, so a list the GPU thread began before the scanner got to it is skipped.
// That is the baseline. After Stop the GPU thread may still be finishing submissions, and its NoteBegan
// calls must neither throw nor change anything. That part can only fail by throwing, which ends the
// test, or by crashing. A queue id past the last slot is included, which the queue clamps, and a
// mutation that removed the clamp would be undefined behaviour that this test cannot be relied on to
// catch.
void TestNoteBeganAfterStop() {
	constexpr uint64_t LATER_SEQ   = 10;
	constexpr uint32_t PAST_QUEUES = SCAN_MAX_QUEUES + 5;
	Harness            harness(NoMemory());
	const uint64_t     first = harness.Submit(ASYNC_QUEUE, SetProgram(PROGRAM_A));
	(void)harness.Submit(ASYNC_QUEUE, SetProgram(PROGRAM_B));
	harness.scanner->NoteBegan(ASYNC_QUEUE, first);
	harness.scanner->Start();
	CHECK(harness.WaitForJobs(1));
	CHECK(harness.scanner->Stats().queue.late_skipped == 1);
	harness.scanner->Stop();
	harness.scanner->NoteBegan(ASYNC_QUEUE, first + LATER_SEQ);
	harness.scanner->NoteBegan(PAST_QUEUES, LATER_SEQ);
	const auto stats = harness.scanner->Stats();
	CHECK(stats.queue.late_skipped == 1 && stats.jobs == 1);
}

// Whether the dispatch on queue 1 is uncertain after `fills` empty graphics lists were queued behind it.
// The queue holds SCAN_JOB_QUEUE_CAP jobs, so enough of them evict the oldest list, which is queue 1's
// program write that sits between its reset and its dispatch.
constexpr uint32_t QUEUE_ONE_JOBS = 3;
constexpr uint32_t BASELINE_FILLS = 10;

bool AsyncDispatchUncertainAfterFills(uint32_t fills) {
	Harness harness(NoMemory());
	(void)harness.Reset(ASYNC_QUEUE);
	(void)harness.Submit(ASYNC_QUEUE, SetProgram(PROGRAM_B));
	Stream list = SetLowHalf(LOW_ONLY_VALUE);
	Append(list, Dispatch());
	(void)harness.Submit(ASYNC_QUEUE, list);
	for (uint32_t fill = 0; fill < fills; ++fill) {
		(void)harness.Submit(GRAPHICS_QUEUE, Stream {});
	}
	const auto queued = harness.scanner->Stats().queue;
	CHECK(queued.dropped == (fills + QUEUE_ONE_JOBS > SCAN_JOB_QUEUE_CAP ? 1u : 0u));
	harness.scanner->Start();
	CHECK(harness.WaitForJobs(queued.pushed - queued.dropped));
	const auto items = harness.Drain();
	CHECK(items.size() == 1);
	return items.empty() || items[0].uncertain_program;
}

// A graphics list that overflows the queue evicts queue 1's oldest list, and it is queue 1's shadow that
// is put in doubt, so its dispatch is uncertain. The baseline queues ten graphics lists, which evicts
// nothing, and the same dispatch is certain. Exactly one list is evicted, so the doubt cannot come from
// a second eviction charged to the queue that pushed.
void TestCrossQueueEvictionMarksUncertain() {
	CHECK(!AsyncDispatchUncertainAfterFills(BASELINE_FILLS));
	// One fill more than the queue holds beside queue 1's three jobs, so exactly one list is evicted.
	CHECK(AsyncDispatchUncertainAfterFills(SCAN_JOB_QUEUE_CAP - QUEUE_ONE_JOBS + 1));
}

} // namespace

// Counts every allocation on the thread that makes it, for TestScannerNoAllocationAfterWarmup.
void* operator new(std::size_t size) {
	t_allocations++;
	t_allocated_bytes += size;
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

int main() {
	// First, so that a Stop that hangs is reported by its watchdog and not by an earlier test's destructor.
	TestStopIsPromptAndIdempotent();
	TestStartAfterStopRefused();
	TestScannerPostsDispatch();
	TestStatsHistogramsCount();
	TestWakeOncePerSubmission();
	TestPreparerStoppedStopsScanner();
	TestAttachMarksShadowUncertain();
	TestLagBoundSkipsStaleJob();
	TestStaleResetStaysCertain();
	TestFreshJobWalksPastOneMillisecond();
	TestSlowReadCounted();
	TestChunkedReadSplits();
	TestScannerChunksBufferReads();
	TestStopEndsLongRead();
	TestScannerNoAllocationAfterWarmup();
	TestNoteBeganAfterStop();
	TestCrossQueueEvictionMarksUncertain();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::puts("PrescanScannerTests passed");
	return 0;
}
