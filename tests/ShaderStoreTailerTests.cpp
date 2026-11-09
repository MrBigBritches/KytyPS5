// Tests the tailer that follows a store worker's output file. The records are synthetic bytes
// framed by the store's own FrameStoreRecord, so no game file and no device are needed. Every test
// names what else could pass it and compares against a baseline that differs only in the behaviour
// under test.
#include "common/config.h"
#include "common/file.h"
#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/frameTailer.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Bytes = std::vector<uint8_t>;

constexpr uint32_t  GENERATOR_VERSION = 7;
constexpr size_t    STORE_HEADER_SIZE = 16;
const EbootIdentity IDENTITY {.size = 123456, .program_hash = 0xfeedfacecafebeefull};

Bytes StoreHeaderBytes() {
	Bytes header(STORE_HEADER_SIZE);
	for (size_t index = 0; index < header.size(); index++) {
		header[index] = static_cast<uint8_t>(0xA0 + index);
	}
	return header;
}

// Bytes that zstd cannot shrink, so a frame's size follows the record's and every record differs.
Bytes RandomRecord(uint32_t seed, size_t size) {
	Bytes    record(size);
	uint64_t state = 0x9E3779B97F4A7C15ull * (seed + 1);
	for (auto& byte: record) {
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		byte = static_cast<uint8_t>(state >> 24);
	}
	return record;
}

// The header of a generated file that the tests' tailer expects.
Bytes Header() {
	return EncodeGeneratedHeader(StoreHeaderBytes(), GENERATOR_VERSION, IDENTITY);
}

// A tailer that expects the test header and the test identity.
FrameTailer MakeTailer() {
	return {StoreHeaderBytes(), GENERATOR_VERSION, IDENTITY};
}

// A generated file made from records, with the offset just after each frame.
struct Generated {
	Bytes                 file;
	std::vector<Bytes>    records;
	std::vector<uint64_t> ends;
	std::vector<Bytes>    frames;
};

Generated MakeGenerated(const std::vector<size_t>& sizes) {
	Generated generated;
	generated.file = Header();
	for (size_t index = 0; index < sizes.size(); index++) {
		generated.records.push_back(RandomRecord(static_cast<uint32_t>(index), sizes[index]));
		generated.frames.push_back(FrameStoreRecord(generated.records.back()));
		generated.file.insert(generated.file.end(), generated.frames.back().begin(),
		                      generated.frames.back().end());
		generated.ends.push_back(generated.file.size());
	}
	return generated;
}

// The sizes include a one byte record and a bigger one, so the frames have different lengths.
const std::vector<size_t> RECORD_SIZES = {40, 1, 700, 250, 1300};

// Moves the frames of one poll onto the end of the frames handed over so far.
void Append(std::vector<TailedFrame>& all, std::vector<TailedFrame>&& more) {
	for (auto& frame: more) {
		all.push_back(std::move(frame));
	}
}

// How many frames end at or before a position, which is how many a tailer may have handed over once
// it has been fed that many bytes.
size_t FramesCompleteBy(const Generated& generated, size_t position) {
	return static_cast<size_t>(
	    std::ranges::count_if(generated.ends, [&](uint64_t end) { return end <= position; }));
}

// The first `count` bytes of a file.
Bytes Prefix(const Bytes& file, size_t count) {
	return Bytes(file.begin(), file.begin() + static_cast<ptrdiff_t>(count));
}

// ---- Chunked feeding. ----

// The baseline is the whole file fed at once. Every chunk size must hand over the same frames with
// the same offsets, and never a frame before its last byte was fed. A tailer that handed over a
// short frame, or that did not advance its offsets, passes neither check.
void TestChunkedFeeding() {
	const auto generated = MakeGenerated(RECORD_SIZES);
	auto       whole     = MakeTailer();
	whole.Feed(generated.file);
	const auto baseline = whole.Poll(false);
	CHECK(baseline.size() == RECORD_SIZES.size());
	for (size_t index = 0; index < baseline.size() && index < RECORD_SIZES.size(); index++) {
		CHECK(baseline[index].raw == generated.records[index]);
		CHECK(baseline[index].end_offset == generated.ends[index]);
	}
	CHECK(whole.GoodOffset() == generated.file.size());
	// The offsets are all different, so a tailer that reports one fixed offset cannot pass.
	CHECK(std::ranges::adjacent_find(generated.ends) == generated.ends.end());

	for (size_t chunk = 1; chunk <= generated.file.size(); chunk++) {
		auto                     tailer = MakeTailer();
		std::vector<TailedFrame> got;
		for (size_t at = 0; at < generated.file.size(); at += chunk) {
			const auto count = std::min(chunk, generated.file.size() - at);
			tailer.Feed(std::span<const uint8_t>(generated.file).subspan(at, count));
			Append(got, tailer.Poll(false));
			CHECK(got.size() == FramesCompleteBy(generated, at + count));
		}
		CHECK(got.size() == baseline.size());
		for (size_t index = 0; index < got.size() && index < baseline.size(); index++) {
			CHECK(got[index].raw == baseline[index].raw);
			CHECK(got[index].end_offset == baseline[index].end_offset);
		}
		CHECK(tailer.Status() == TailStatus::Waiting);
		(void)tailer.Poll(true);
		CHECK(tailer.Status() == TailStatus::Complete);
		CHECK(tailer.BufferedBytes() == 0);
	}
}

// A second poll with no new bytes hands over nothing, so an offset that did not advance, which
// would hand the same frames over again, fails here.
void TestFramesAreHandedOverOnce() {
	const auto generated = MakeGenerated(RECORD_SIZES);
	auto       tailer    = MakeTailer();
	tailer.Feed(generated.file);
	CHECK(tailer.Poll(false).size() == RECORD_SIZES.size());
	CHECK(tailer.Poll(false).empty());
	CHECK(tailer.Poll(true).empty());
	CHECK(tailer.GoodOffset() == generated.file.size());
}

// The tailer keeps the undelivered tail and no more.
void TestBufferHoldsOnlyTheTail() {
	const auto generated = MakeGenerated(RECORD_SIZES);
	auto       tailer    = MakeTailer();
	const auto cut       = generated.ends[3] + 5;
	tailer.Feed(std::span<const uint8_t>(generated.file).first(cut));
	CHECK(tailer.Poll(false).size() == 4);
	CHECK(tailer.BufferedBytes() == 5);
	// The delivered frames are not kept. A tailer that kept them would hold the whole file here.
	CHECK(tailer.RetainedBytes() == 5);
}

// ---- A frame that is not complete or not good, while the child lives. ----

// A frame cut at every possible point is held while the child lives, and what was handed over
// before it stays the same. The rest arriving gives the frame. Waiting for the whole file would
// hand the first frame over late, so the first check pins that frame being delivered at once.
void TestTornFrameWhileAlive() {
	const auto generated = MakeGenerated(RECORD_SIZES);
	const auto begin     = generated.ends[0];
	const auto end       = generated.ends[1];
	for (size_t cut = begin; cut < end; cut++) {
		auto tailer = MakeTailer();
		tailer.Feed(std::span<const uint8_t>(generated.file).first(cut));
		const auto first = tailer.Poll(false);
		CHECK(first.size() == 1);
		CHECK(tailer.Status() == TailStatus::Waiting);
		CHECK(tailer.GoodOffset() == begin);
		// Polling again with the child alive still holds it and gives nothing.
		CHECK(tailer.Poll(false).empty());
		CHECK(tailer.Status() == TailStatus::Waiting);
		tailer.Feed(std::span<const uint8_t>(generated.file).subspan(cut));
		const auto rest = tailer.Poll(false);
		CHECK(rest.size() == RECORD_SIZES.size() - 1);
		CHECK(!rest.empty() && rest.front().raw == generated.records[1]);
		CHECK(!rest.empty() && rest.front().end_offset == end);
	}
}

// Does what a poller does for a held item, which is to drop what the tailer holds and feed the file
// again from GoodOffset.
void Rewind(FrameTailer& tailer, const Bytes& file) {
	tailer.DropUndelivered();
	tailer.Feed(std::span<const uint8_t>(file).subspan(tailer.GoodOffset()));
}

// A frame with all its bytes and a bad checksum at the end of the bytes is held while the child
// lives. Once it exited it is held once more, and is final when it is judged after the rewind, with
// the file good up to the frame before. The same bytes with the checksum intact are the baseline,
// which hands over all three.
void TestCorruptTrailingFrame() {
	const auto generated = MakeGenerated({300, 400, 500});
	auto       baseline  = MakeTailer();
	baseline.Feed(generated.file);
	CHECK(baseline.Poll(false).size() == 3);

	const size_t frame_begin = generated.ends[0];
	// A flipped checksum, and a flipped payload byte that only the checksum can tell.
	const size_t flips[] = {frame_begin + offsetof(StoreFrame, checksum), generated.ends[1] - 8};
	for (const auto flip: flips) {
		auto damaged = Prefix(generated.file, generated.ends[1]);
		damaged[flip] ^= 0x01;
		auto tailer = MakeTailer();
		tailer.Feed(damaged);
		CHECK(tailer.Poll(false).size() == 1);
		CHECK(tailer.Status() == TailStatus::Waiting);
		CHECK(tailer.WantsReread());
		CHECK(tailer.GoodOffset() == generated.ends[0]);
		CHECK(tailer.BufferedBytes() == damaged.size() - generated.ends[0]);

		CHECK(tailer.Poll(true).empty());
		CHECK(tailer.Status() == TailStatus::Waiting && tailer.WantsReread());
		Rewind(tailer, damaged);
		CHECK(tailer.Poll(true).empty());
		CHECK(tailer.Status() == TailStatus::DamagedTail);
		CHECK(!tailer.WantsReread());
		CHECK(tailer.GoodOffset() == generated.ends[0]);
	}
}

// A bad frame with a good one after it is not a write caught half way, since the writer finished it
// before it began the next. Its bytes may be older than the ones after it, so it is held first and
// is final when it is judged again after a rewind, and the frame after it is not handed over. The
// trailing flip above is the baseline that is not final.
void TestCorruptFrameBeforeMoreBytes() {
	const auto generated = MakeGenerated({300, 400, 500});
	auto       damaged   = generated.file;
	damaged[generated.ends[0] + offsetof(StoreFrame, checksum)] ^= 0x01;
	auto tailer = MakeTailer();
	tailer.Feed(damaged);
	CHECK(tailer.Poll(false).size() == 1);
	CHECK(tailer.Status() == TailStatus::Waiting && tailer.WantsReread());
	Rewind(tailer, damaged);
	CHECK(tailer.Poll(false).empty());
	CHECK(tailer.Status() == TailStatus::DamagedFrame);
	CHECK(tailer.IsFinal() && !tailer.WantsReread());
	CHECK(tailer.GoodOffset() == generated.ends[0]);
	// Final, so the exit and more bytes change nothing.
	tailer.Feed(RandomRecord(3, 50));
	CHECK(tailer.Poll(false).empty());
	CHECK(tailer.Poll(true).empty());
	CHECK(tailer.Status() == TailStatus::DamagedFrame);
	CHECK(tailer.GoodOffset() == generated.ends[0]);
}

// The poll after a rewind judges everything it reads, also an item behind a frame that it hands
// over, since all those bytes were read together. A good frame is handed over and then a bad frame
// with a good one after it is final in that poll. A tailer that took the state to end with the
// first delivered frame would hold the item again.
void TestFreshPollJudgesItemBehindDeliveredFrame() {
	const auto generated = MakeGenerated({300, 400, 500});
	auto       damaged   = generated.file;
	damaged[generated.ends[0] + offsetof(StoreFrame, checksum)] ^= 0x01;
	auto tailer = MakeTailer();
	tailer.Feed(Header());
	CHECK(tailer.Poll(false).empty());
	Rewind(tailer, damaged);
	CHECK(tailer.Poll(false).size() == 1);
	CHECK(tailer.Status() == TailStatus::DamagedFrame);
	CHECK(tailer.GoodOffset() == generated.ends[0]);
}

// A short tail is final once the child has exited, and is not while it lives.
void TestShortTailAfterExit() {
	const auto generated = MakeGenerated(RECORD_SIZES);
	auto       tailer    = MakeTailer();
	tailer.Feed(std::span<const uint8_t>(generated.file).first(generated.ends[2] + 3));
	CHECK(tailer.Poll(false).size() == 3);
	CHECK(tailer.Status() == TailStatus::Waiting);
	CHECK(tailer.Poll(true).empty());
	CHECK(tailer.Status() == TailStatus::DamagedTail);
	CHECK(tailer.GoodOffset() == generated.ends[2]);
}

// ---- The cap on a frame. ----

// A frame header and nothing else, with the given compressed size.
Bytes FrameHeaderOnly(uint32_t compressed_size) {
	const StoreFrame frame {.compressed_size = compressed_size, .raw_size = 1, .checksum = 0};
	Bytes            bytes(sizeof(frame));
	std::memcpy(bytes.data(), &frame, sizeof(frame));
	return bytes;
}

// A frame size over the cap is judged like any bad item. It is held at first, since a header caught
// half way can hold any size. It becomes final only in the poll right after a rewind, once the
// child exited when nothing follows it, or with the child alive when bytes follow it. A size
// exactly at the cap is the baseline, which is a frame that is only incomplete, so a good frame
// before it is handed over and the tailer waits.
void TestOversizeFrame() {
	const auto generated = MakeGenerated({100, 200});
	const auto good      = std::span<const uint8_t>(generated.file).first(generated.ends[0]);
	for (const uint32_t size: {MAX_STORE_FRAME_COMPRESSED_BYTES + 1, 0xFFFFFFFFu}) {
		auto held = MakeTailer();
		held.Feed(good);
		held.Feed(FrameHeaderOnly(size));
		CHECK(held.Poll(false).size() == 1);
		CHECK(held.Status() == TailStatus::Waiting && held.WantsReread());
		CHECK(held.GoodOffset() == generated.ends[0]);
		CHECK(held.Poll(true).empty());
		CHECK(held.Status() == TailStatus::Waiting && held.WantsReread());
		auto       held_file = Bytes(good.begin(), good.end());
		const auto only      = FrameHeaderOnly(size);
		held_file.insert(held_file.end(), only.begin(), only.end());
		Rewind(held, held_file);
		CHECK(held.Poll(true).empty());
		CHECK(held.Status() == TailStatus::FrameTooLarge);
		CHECK(held.GoodOffset() == generated.ends[0]);

		auto followed = MakeTailer();
		followed.Feed(good);
		followed.Feed(FrameHeaderOnly(size));
		followed.Feed(RandomRecord(1, 64));
		CHECK(followed.Poll(false).size() == 1);
		CHECK(followed.Status() == TailStatus::Waiting && followed.WantsReread());
		auto       followed_file = Bytes(good.begin(), good.end());
		const auto huge          = FrameHeaderOnly(size);
		followed_file.insert(followed_file.end(), huge.begin(), huge.end());
		const auto after = RandomRecord(1, 64);
		followed_file.insert(followed_file.end(), after.begin(), after.end());
		Rewind(followed, followed_file);
		CHECK(followed.Poll(false).empty());
		CHECK(followed.Status() == TailStatus::FrameTooLarge);
		CHECK(!followed.WantsReread());
		CHECK(followed.GoodOffset() == generated.ends[0]);
		// Final, so later bytes and the exit change nothing.
		CHECK(followed.Poll(false).empty());
		CHECK(followed.Poll(true).empty());
		CHECK(followed.Status() == TailStatus::FrameTooLarge);
	}
	auto at_cap = MakeTailer();
	at_cap.Feed(good);
	at_cap.Feed(FrameHeaderOnly(MAX_STORE_FRAME_COMPRESSED_BYTES));
	CHECK(at_cap.Poll(false).size() == 1);
	CHECK(at_cap.Status() == TailStatus::Waiting && !at_cap.WantsReread());
	CHECK(at_cap.GoodOffset() == generated.ends[0]);
}

// ---- The header. ----

// A different StoreHeader, version or identity is an error. It is held at first and becomes final
// only in the poll right after a rewind, with good frames after it, or once the child exited when
// nothing follows it. The matching header is the baseline, and a tailer that did not check the
// header hands the frames over.
void TestHeaderMismatch() {
	const auto generated   = MakeGenerated({100, 200});
	const auto frames      = Bytes(generated.file.begin() + static_cast<ptrdiff_t>(Header().size()),
	                               generated.file.end());
	auto       store_other = StoreHeaderBytes();
	store_other[3] ^= 1;
	const Bytes headers[] = {
	    EncodeGeneratedHeader(store_other, GENERATOR_VERSION, IDENTITY),
	    EncodeGeneratedHeader(StoreHeaderBytes(), GENERATOR_VERSION + 1, IDENTITY),
	    EncodeGeneratedHeader(StoreHeaderBytes(), GENERATOR_VERSION,
	                          {IDENTITY.size + 1, IDENTITY.program_hash}),
	    EncodeGeneratedHeader(StoreHeaderBytes(), GENERATOR_VERSION,
	                          {IDENTITY.size, IDENTITY.program_hash ^ 1}),
	};
	for (const auto& header: headers) {
		auto file = header;
		file.insert(file.end(), frames.begin(), frames.end());
		auto followed = MakeTailer();
		followed.Feed(file);
		CHECK(followed.Poll(false).empty());
		CHECK(followed.Status() == TailStatus::Waiting && followed.WantsReread());
		Rewind(followed, file);
		CHECK(followed.Poll(false).empty());
		CHECK(followed.Status() == TailStatus::HeaderError);
		CHECK(followed.Poll(true).empty());
		CHECK(followed.Status() == TailStatus::HeaderError);

		auto held = MakeTailer();
		held.Feed(header);
		CHECK(held.Poll(false).empty());
		CHECK(held.Status() == TailStatus::Waiting && held.WantsReread());
		CHECK(held.GoodOffset() == 0);
		CHECK(held.Poll(true).empty());
		CHECK(held.Status() == TailStatus::Waiting && held.WantsReread());
		Rewind(held, header);
		CHECK(held.Poll(true).empty());
		CHECK(held.Status() == TailStatus::HeaderError);
	}
	auto baseline = MakeTailer();
	baseline.Feed(generated.file);
	CHECK(baseline.Poll(false).size() == 2);
}

// An incomplete header waits, at every length, and hands over nothing. After the exit it is a
// damaged tail that is good up to zero. The complete header is the baseline.
void TestIncompleteHeaderWaits() {
	const auto header = Header();
	for (size_t length = 0; length < header.size(); length++) {
		auto tailer = MakeTailer();
		tailer.Feed(std::span<const uint8_t>(header).first(length));
		CHECK(tailer.Poll(false).empty());
		CHECK(tailer.Status() == TailStatus::Waiting);
		CHECK(tailer.GoodOffset() == 0);
		CHECK(tailer.Poll(true).empty());
		CHECK(tailer.Status() == TailStatus::DamagedTail);
		CHECK(tailer.GoodOffset() == 0);
	}
	auto complete = MakeTailer();
	complete.Feed(header);
	CHECK(complete.Poll(true).empty());
	CHECK(complete.Status() == TailStatus::Complete);
	CHECK(complete.GoodOffset() == header.size());
}

// Without an expected identity the tailer takes the file's own, reports it, and hands the frames
// over. The store header and version are still checked.
void TestIdentityFromFile() {
	const auto  generated = MakeGenerated({100, 200});
	FrameTailer tailer {StoreHeaderBytes(), GENERATOR_VERSION, std::nullopt};
	CHECK(!tailer.Identity().has_value());
	tailer.Feed(Header());
	CHECK(tailer.Poll(false).empty());
	CHECK(tailer.Status() == TailStatus::Waiting);
	CHECK(tailer.Identity() == std::optional<EbootIdentity>(IDENTITY));
	tailer.Feed(std::span<const uint8_t>(generated.file).subspan(Header().size()));
	const auto frames = tailer.Poll(false);
	CHECK(frames.size() == 2);
	CHECK(frames.size() == 2 && frames[1].raw == generated.records[1]);

	FrameTailer other {StoreHeaderBytes(), GENERATOR_VERSION + 1, std::nullopt};
	other.Feed(generated.file);
	CHECK(other.Poll(false).empty());
	Rewind(other, generated.file);
	CHECK(other.Poll(false).empty());
	CHECK(other.Status() == TailStatus::HeaderError);
}

// ---- The poller. ----

// A path in the temporary directory that no other run of the tests uses, made from the clock.
std::filesystem::path TempPath(const char* name) {
	const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
	return std::filesystem::temp_directory_path() /
	       (std::string("kyty_tailer_") + name + "_" + std::to_string(stamp) + ".tmp");
}

// A temporary file with a tailer and a poller on it, so the scenarios below are short. The file is
// removed with the object.
struct PolledFile {
	std::filesystem::path path;
	FrameTailer           tailer = MakeTailer();
	FileTailPoller        poller;

	explicit PolledFile(const char* name, uint64_t max_poll_bytes = MAX_POLL_READ_BYTES)
	    : path(TempPath(name)), poller(path, tailer, max_poll_bytes) {}
	~PolledFile() { std::filesystem::remove(path); }
	PolledFile(const PolledFile&)            = delete;
	PolledFile& operator=(const PolledFile&) = delete;
};

// Writes `bytes` into a file at an offset through a writer that is open for the call.
void WriteAt(const std::filesystem::path& path, uint64_t offset, std::span<const uint8_t> bytes,
             bool create) {
	Common::File writer;
	CHECK(create ? writer.Create(path) : writer.Open(path, Common::File::Mode::ReadWrite));
	CHECK(writer.Seek(offset));
	writer.Write(bytes.data(), static_cast<uint32_t>(bytes.size()));
	CHECK(writer.Flush());
}

// Cuts a file back to a size.
void TruncateTo(const std::filesystem::path& path, uint64_t size) {
	Common::File file;
	CHECK(file.Open(path, Common::File::Mode::ReadWrite));
	CHECK(file.Truncate(size));
}

// A file that is not there is nothing yet, and an exit without a file is a damaged tail at zero.
// The file appearing afterwards is read from its start.
void TestPollerMissingFile() {
	const auto     path   = TempPath("missing");
	auto           tailer = MakeTailer();
	FileTailPoller poller(path, tailer);
	CHECK(poller.Poll(false).empty());
	CHECK(tailer.Status() == TailStatus::Waiting);
	CHECK(poller.BytesRead() == 0);

	const auto generated = MakeGenerated({100});
	{
		Common::File writer;
		CHECK(writer.Create(path));
		writer.Write(generated.file.data(), static_cast<uint32_t>(generated.file.size()));
		CHECK(writer.Flush());
		const auto frames = poller.Poll(false);
		CHECK(frames.size() == 1);
		CHECK(poller.BytesRead() == generated.file.size());
	}
	std::filesystem::remove(path);

	auto           gone = MakeTailer();
	FileTailPoller gone_poller(TempPath("never"), gone);
	CHECK(gone_poller.Poll(true).empty());
	CHECK(gone.Status() == TailStatus::DamagedTail);
	CHECK(gone.GoodOffset() == 0);
}

// The writer keeps the file open and appends in pieces that cut frames, and the poller reads while
// it is open. The frames must come from the file, so each poll's count is compared with the number
// of frames complete in the bytes written so far, and BytesRead with that size.
void TestPollerFollowsOpenWriter() {
	const auto               path      = TempPath("follow");
	const auto               generated = MakeGenerated(RECORD_SIZES);
	auto                     tailer    = MakeTailer();
	FileTailPoller           poller(path, tailer);
	std::vector<TailedFrame> got;
	{
		Common::File writer;
		CHECK(writer.Create(path));
		constexpr size_t PIECE = 97;
		for (size_t at = 0; at < generated.file.size(); at += PIECE) {
			const auto count = std::min(PIECE, generated.file.size() - at);
			writer.Write(generated.file.data() + at, static_cast<uint32_t>(count));
			CHECK(writer.Flush());
			Append(got, poller.Poll(false));
			CHECK(poller.BytesRead() == at + count);
			CHECK(got.size() == FramesCompleteBy(generated, at + count));
		}
		// Polling again with nothing new gives nothing, so the poller did not read from the start.
		CHECK(poller.Poll(false).empty());
		CHECK(poller.BytesRead() == generated.file.size());
	}
	CHECK(got.size() == RECORD_SIZES.size());
	for (size_t index = 0; index < got.size() && index < RECORD_SIZES.size(); index++) {
		CHECK(got[index].raw == generated.records[index]);
		CHECK(got[index].end_offset == generated.ends[index]);
	}
	CHECK(poller.Poll(true).empty());
	CHECK(tailer.Status() == TailStatus::Complete);
	std::filesystem::remove(path);
}

// A frame with all its bytes but garbage in them, which is what a reader sees if it catches the
// writer mid-write. The garbage is the last thing in the file, so it may be a write in flight. It
// is overwritten with the real bytes afterwards, and the frames after it are written then too. The
// baseline is a file whose frame is never fixed, which is held and final once the child exits. Only
// a poller that reads the held frame again, from the last good frame, delivers it and the two
// frames after it.
void TestPollerRecoversFrameCaughtMidWrite() {
	const auto generated = MakeGenerated({300, 400, 500, 200});
	auto       garbage   = Prefix(generated.file, generated.ends[1]);
	std::fill(garbage.begin() + static_cast<ptrdiff_t>(generated.ends[0] + sizeof(StoreFrame)),
	          garbage.end(), uint8_t {0xEE});
	CHECK(Prefix(generated.file, garbage.size()) != garbage);

	PolledFile fixed("midwrite");
	WriteAt(fixed.path, 0, garbage, true);
	CHECK(fixed.poller.Poll(false).size() == 1);
	CHECK(fixed.tailer.Status() == TailStatus::Waiting);
	CHECK(fixed.tailer.GoodOffset() == generated.ends[0]);
	// The poller went back to the held frame's start, and not to the end of the file.
	CHECK(fixed.poller.BytesRead() == generated.ends[0]);
	// Still garbage, so it is held again and nothing past it is handed over.
	CHECK(fixed.poller.Poll(false).empty());
	CHECK(fixed.tailer.Status() == TailStatus::Waiting);
	WriteAt(fixed.path, generated.ends[0],
	        std::span<const uint8_t>(generated.file).subspan(generated.ends[0]), false);
	const auto rest = fixed.poller.Poll(false);
	CHECK(rest.size() == 3);
	for (size_t index = 0; index < rest.size() && index < 3; index++) {
		CHECK(rest[index].raw == generated.records[index + 1]);
		CHECK(rest[index].end_offset == generated.ends[index + 1]);
	}
	CHECK(fixed.poller.Poll(true).empty());
	CHECK(fixed.tailer.Status() == TailStatus::Complete);

	// The baseline, whose frame is never fixed. It is final once the child exits.
	PolledFile stuck("stuck");
	WriteAt(stuck.path, 0, garbage, true);
	CHECK(stuck.poller.Poll(false).size() == 1);
	CHECK(stuck.poller.Poll(false).empty());
	CHECK(stuck.tailer.Status() == TailStatus::Waiting);
	// The poll before this one went back to the frame, so this one is the first after a rewind and
	// judges the item for good, as the exit says nothing more will be written.
	CHECK(stuck.poller.Poll(true).empty());
	CHECK(stuck.tailer.Status() == TailStatus::DamagedTail);
	CHECK(stuck.tailer.GoodOffset() == generated.ends[0]);
	// The polls while the child lived went back to the frame's start each time, and the poll after
	// the exit read on to the end of the file and stayed there, since nothing changes after the
	// exit. A poller that went back after the exit too would end at the frame's start.
	CHECK(stuck.poller.BytesRead() == garbage.size());
}

// A garbage frame with a good frame after it, all in the file when the poller first reads it, is
// held once and read again, and the second read judges it for good. That is one rewind and then the
// final status DamagedFrame, and a final tailer reads no more even when the file grows. The test
// above is the baseline where the same garbage has nothing after it, which never becomes final
// while the child lives.
void TestPollerBadFrameBeforeMoreBytes() {
	const auto generated = MakeGenerated({300, 400, 500, 200});
	auto       garbage   = generated.file;
	std::fill(garbage.begin() + static_cast<ptrdiff_t>(generated.ends[0] + sizeof(StoreFrame)),
	          garbage.begin() + static_cast<ptrdiff_t>(generated.ends[1]), uint8_t {0xEE});
	PolledFile file("followed");
	WriteAt(file.path, 0, garbage, true);
	CHECK(file.poller.Poll(false).size() == 1);
	CHECK(file.tailer.Status() == TailStatus::Waiting);
	CHECK(file.poller.BytesRead() == generated.ends[0]);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.tailer.Status() == TailStatus::DamagedFrame);
	CHECK(file.poller.BytesRead() == garbage.size());
	WriteAt(file.path, garbage.size(), RandomRecord(9, 100), false);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.poller.BytesRead() == garbage.size());
	CHECK(file.tailer.RetainedBytes() == garbage.size() - generated.ends[0]);
}

// The first half of a frame is read and polled. Then the file holds garbage in that frame, and the
// rest and a good frame arrive. The tailer holds the old first half and the new rest, which do not
// unframe, and a good frame follows them. But the old half is not from the read that gave the rest,
// so the item must not be final yet. It is read again, which is one rewind to the frame's start,
// and is then final. The baseline is the same bad bytes in one piece, which also takes one rewind.
// A tailer that took the stale bytes for proof would be final after the first poll in the first
// case.
void TestPollerStaleBytesAreNotFinal() {
	const auto generated = MakeGenerated({300, 400, 500});
	auto       garbage   = generated.file;
	std::fill(garbage.begin() + static_cast<ptrdiff_t>(generated.ends[0] + sizeof(StoreFrame)),
	          garbage.begin() + static_cast<ptrdiff_t>(generated.ends[1]), uint8_t {0xEE});
	const size_t half = (generated.ends[0] + generated.ends[1]) / 2;

	PolledFile stale("stale");
	WriteAt(stale.path, 0, Prefix(generated.file, half), true);
	CHECK(stale.poller.Poll(false).size() == 1);
	CHECK(stale.poller.BytesRead() == half);
	WriteAt(stale.path, 0, garbage, false);
	CHECK(stale.poller.Poll(false).empty());
	CHECK(stale.tailer.Status() == TailStatus::Waiting);
	CHECK(stale.poller.BytesRead() == generated.ends[0]);
	CHECK(stale.poller.Poll(false).empty());
	CHECK(stale.tailer.Status() == TailStatus::DamagedFrame);
	CHECK(stale.tailer.GoodOffset() == generated.ends[0]);

	PolledFile whole("not_stale");
	WriteAt(whole.path, 0, garbage, true);
	CHECK(whole.poller.Poll(false).size() == 1);
	CHECK(whole.tailer.Status() == TailStatus::Waiting);
	CHECK(whole.poller.BytesRead() == generated.ends[0]);
	CHECK(whole.poller.Poll(false).empty());
	CHECK(whole.tailer.Status() == TailStatus::DamagedFrame);
}

// The state after a rewind covers one poll only. After RestartFrom a poll reads half of a frame,
// and the next poll reads garbage over it and a good frame after. The old half and the new rest are
// not from one read, so the item is not final yet. It takes one more rewind and is final after
// that. A tailer that kept the state until a frame was handed over would be final in the second
// poll. The baseline is TestPollerStaleBytesAreNotFinal, the same steps without RestartFrom.
void TestPollerRestartStateCoversOnePoll() {
	const auto generated = MakeGenerated({300, 400, 500});
	auto       garbage   = generated.file;
	std::fill(garbage.begin() + static_cast<ptrdiff_t>(generated.ends[0] + sizeof(StoreFrame)),
	          garbage.begin() + static_cast<ptrdiff_t>(generated.ends[1]), uint8_t {0xEE});
	const size_t half = (generated.ends[0] + generated.ends[1]) / 2;

	PolledFile file("restart_state");
	WriteAt(file.path, 0, Prefix(generated.file, generated.ends[0]), true);
	CHECK(file.poller.Poll(false).size() == 1);
	file.poller.RestartFrom();
	WriteAt(file.path, generated.ends[0],
	        std::span<const uint8_t>(generated.file)
	            .subspan(generated.ends[0], half - generated.ends[0]),
	        false);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.poller.BytesRead() == half);
	WriteAt(file.path, 0, garbage, false);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.tailer.Status() == TailStatus::Waiting);
	CHECK(file.poller.BytesRead() == generated.ends[0]);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.tailer.Status() == TailStatus::DamagedFrame);
}

// A bad item at the end of the bytes after the exit is held once and is final in the poll after the
// rewind, and a short tail is final at once. The first is the stuck case of the tests above, and
// the second is the torn tail of TestPollerRestart. Here the item is a garbage frame at the end,
// and the tailer is told of the exit before any rewind has happened.
void TestExitDoesNotJudgeStaleTrailingItem() {
	const auto generated = MakeGenerated({300, 400});
	auto       garbage   = Prefix(generated.file, generated.ends[1]);
	std::fill(garbage.begin() + static_cast<ptrdiff_t>(generated.ends[0] + sizeof(StoreFrame)),
	          garbage.end(), uint8_t {0xEE});
	auto tailer = MakeTailer();
	tailer.Feed(garbage);
	CHECK(tailer.Poll(true).size() == 1);
	CHECK(tailer.Status() == TailStatus::Waiting && tailer.WantsReread());
	Rewind(tailer, garbage);
	CHECK(tailer.Poll(true).empty());
	CHECK(tailer.Status() == TailStatus::DamagedTail);
	CHECK(tailer.GoodOffset() == generated.ends[0]);
}

// A file that is gone after frames were handed over is Truncated, and not nothing yet. The baseline
// is the missing file before any frame, which waits, as TestPollerMissingFile checks.
void TestPollerFileGoneAfterFrames() {
	const auto generated = MakeGenerated({300});
	PolledFile file("gone");
	WriteAt(file.path, 0, generated.file, true);
	CHECK(file.poller.Poll(false).size() == 1);
	std::filesystem::remove(file.path);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.tailer.Status() == TailStatus::Truncated);
}

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
// A file that is there but cannot be opened for a moment, as when a scanner holds it without
// sharing, is nothing new and not Truncated. TestPollerFileGoneAfterFrames is the baseline, where
// the same failed open comes from a file that is really gone.
void TestPollerBusyFileIsNotTruncated() {
	const auto generated = MakeGenerated({300});
	PolledFile file("busy");
	WriteAt(file.path, 0, generated.file, true);
	CHECK(file.poller.Poll(false).size() == 1);
	// No sharing at all, so the poller's open fails while the file still exists.
	HANDLE holder = CreateFileW(file.path.c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING,
	                            FILE_ATTRIBUTE_NORMAL, nullptr);
	CHECK(holder != INVALID_HANDLE_VALUE);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.tailer.Status() != TailStatus::Truncated);
	CloseHandle(holder);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.tailer.Status() != TailStatus::Truncated);
}
#endif

// A file cut back below GoodOffset has lost frames that were handed over, so the status is the
// final Truncated and the poller is not restarted silently. The baseline is a cut to GoodOffset
// exactly, which is the restart that TestPollerFileShrinks covers.
void TestPollerShrinkBelowGoodOffset() {
	const auto generated = MakeGenerated({300, 400, 500, 200});
	PolledFile file("below");
	WriteAt(file.path, 0, generated.file, true);
	CHECK(file.poller.Poll(false).size() == 4);
	TruncateTo(file.path, generated.ends[1]);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.tailer.Status() == TailStatus::Truncated);
	CHECK(file.tailer.GoodOffset() == generated.ends[3]);
	CHECK(file.poller.BytesRead() == generated.file.size());
	// The file growing again does not undo it, and neither does RestartFrom, since the file cannot
	// be resumed from GoodOffset. Only a new tailer clears it.
	WriteAt(file.path, generated.ends[1],
	        std::span<const uint8_t>(generated.file).subspan(generated.ends[1]), false);
	CHECK(file.poller.Poll(true).empty());
	CHECK(file.tailer.Status() == TailStatus::Truncated);
	file.poller.RestartFrom();
	CHECK(file.tailer.Status() == TailStatus::Truncated);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.tailer.Status() == TailStatus::Truncated);

	// Three frames and the start of the fourth are read, and the file is cut to the third's end.
	PolledFile exact("exact");
	WriteAt(exact.path, 0, Prefix(generated.file, generated.ends[2] + 20), true);
	CHECK(exact.poller.Poll(false).size() == 3);
	CHECK(exact.poller.BytesRead() == generated.ends[2] + 20);
	TruncateTo(exact.path, generated.ends[2]);
	CHECK(exact.poller.Poll(false).empty());
	CHECK(exact.tailer.Status() == TailStatus::Waiting);
	CHECK(exact.poller.BytesRead() == generated.ends[2]);
}

// A header that does not match, and an oversize frame header, are held when they are the last bytes
// and read again from where the bad item begins, which is zero for the header. Each is then fixed
// in place and the frames arrive. The baseline of each is the same file that is never fixed, which
// is a final error once the child exited.
void TestPollerRecoversHeaderAndSizeCaughtMidWrite() {
	const auto generated = MakeGenerated({300, 400});
	auto       other     = StoreHeaderBytes();
	other[0] ^= 1;
	const auto bad_header = EncodeGeneratedHeader(other, GENERATOR_VERSION, IDENTITY);

	PolledFile header_file("badheader");
	WriteAt(header_file.path, 0, bad_header, true);
	CHECK(header_file.poller.Poll(false).empty());
	CHECK(header_file.tailer.Status() == TailStatus::Waiting);
	CHECK(header_file.poller.BytesRead() == 0);
	WriteAt(header_file.path, 0, generated.file, false);
	CHECK(header_file.poller.Poll(false).size() == 2);
	CHECK(header_file.poller.BytesRead() == generated.file.size());

	PolledFile header_stuck("badheader_stuck");
	WriteAt(header_stuck.path, 0, bad_header, true);
	CHECK(header_stuck.poller.Poll(false).empty());
	CHECK(header_stuck.poller.Poll(true).empty());
	CHECK(header_stuck.tailer.Status() == TailStatus::HeaderError);

	auto       oversize = Prefix(generated.file, generated.ends[0]);
	const auto huge     = FrameHeaderOnly(MAX_STORE_FRAME_COMPRESSED_BYTES + 1);
	oversize.insert(oversize.end(), huge.begin(), huge.end());
	PolledFile size_file("oversize");
	WriteAt(size_file.path, 0, oversize, true);
	CHECK(size_file.poller.Poll(false).size() == 1);
	CHECK(size_file.tailer.Status() == TailStatus::Waiting);
	CHECK(size_file.poller.BytesRead() == generated.ends[0]);
	WriteAt(size_file.path, generated.ends[0],
	        std::span<const uint8_t>(generated.file).subspan(generated.ends[0]), false);
	const auto fixed = size_file.poller.Poll(false);
	CHECK(fixed.size() == 1 && !fixed.empty() && fixed[0].raw == generated.records[1]);

	PolledFile size_stuck("oversize_stuck");
	WriteAt(size_stuck.path, 0, oversize, true);
	CHECK(size_stuck.poller.Poll(false).size() == 1);
	CHECK(size_stuck.poller.Poll(true).empty());
	CHECK(size_stuck.tailer.Status() == TailStatus::FrameTooLarge);
	CHECK(size_stuck.tailer.GoodOffset() == generated.ends[0]);
}

// The file of a child that died with a torn frame at the end, whose bytes are not the start of the
// real frame, so reading on from them cannot work.
Bytes TornFile(const Generated& generated) {
	auto torn = Prefix(generated.file, generated.ends[0] + 20);
	std::fill(torn.begin() + static_cast<ptrdiff_t>(generated.ends[0] + sizeof(StoreFrame)),
	          torn.end(), uint8_t {0xEE});
	return torn;
}

// A restart of the child. The first child dies with a torn frame at the end, so the tailer is
// DamagedTail with the offset before it. The supervisor cuts the file back to that offset, and the
// restarted child appends the real frames. RestartFrom makes the poller read on from the offset, so
// each frame arrives once. The baseline is the same steps without RestartFrom, where the tailer
// stays final and delivers nothing.
void TestPollerRestart() {
	const auto generated = MakeGenerated({300, 400, 500});
	const auto torn      = TornFile(generated);
	const auto real_rest = std::span<const uint8_t>(generated.file).subspan(generated.ends[0]);

	for (const bool restart: {true, false}) {
		PolledFile file(restart ? "restart" : "no_restart");
		WriteAt(file.path, 0, torn, true);
		CHECK(file.poller.Poll(false).size() == 1);
		CHECK(file.poller.Poll(true).empty());
		CHECK(file.tailer.Status() == TailStatus::DamagedTail);
		const auto kept = file.tailer.GoodOffset();
		CHECK(kept == generated.ends[0]);
		TruncateTo(file.path, kept);
		WriteAt(file.path, kept, real_rest, false);
		if (restart) {
			file.poller.RestartFrom();
			CHECK(file.poller.BytesRead() == kept);
			CHECK(file.tailer.Status() == TailStatus::Waiting);
		}
		const auto frames = file.poller.Poll(false);
		if (!restart) {
			CHECK(frames.empty());
			// The poller did not read the file again, so the real frames were never fed.
			CHECK(file.poller.BytesRead() == torn.size());
			continue;
		}
		CHECK(frames.size() == 2);
		for (size_t index = 0; index < frames.size() && index < 2; index++) {
			CHECK(frames[index].raw == generated.records[index + 1]);
			CHECK(frames[index].end_offset == generated.ends[index + 1]);
		}
		CHECK(file.poller.Poll(true).empty());
		CHECK(file.tailer.Status() == TailStatus::Complete);
		CHECK(file.tailer.GoodOffset() == generated.file.size());
	}
}

// A file cut back below what the poller has read is read again from GoodOffset, and is not taken
// for a file with nothing new. After the exit the file is cut to GoodOffset with nothing appended,
// and one poll goes back to GoodOffset. The real frames written afterwards arrive once. A poller
// that ignored the shrink would stay at the torn offset.
void TestPollerFileShrinks() {
	const auto generated = MakeGenerated({300, 400});
	const auto torn      = TornFile(generated);
	PolledFile file("shrink");
	WriteAt(file.path, 0, torn, true);
	CHECK(file.poller.Poll(false).size() == 1);
	CHECK(file.poller.Poll(true).empty());
	CHECK(file.tailer.Status() == TailStatus::DamagedTail);
	CHECK(file.poller.BytesRead() == torn.size());
	TruncateTo(file.path, generated.ends[0]);
	CHECK(file.poller.Poll(false).empty());
	CHECK(file.poller.BytesRead() == generated.ends[0]);
	CHECK(file.tailer.Status() == TailStatus::Waiting);
	WriteAt(file.path, generated.ends[0],
	        std::span<const uint8_t>(generated.file).subspan(generated.ends[0]), false);
	const auto frames = file.poller.Poll(false);
	CHECK(frames.size() == 1 && !frames.empty() && frames[0].raw == generated.records[1]);
	CHECK(file.poller.Poll(true).empty());
	CHECK(file.tailer.Status() == TailStatus::Complete);
}

// A backlog over the cap of one poll is read over several polls, and every frame arrives once and
// in order. Each poll reads at most the cap. The cap is lowered so that the backlog is small. With
// the child exited the status stays Waiting until the last bytes were read, so a poll that stopped
// early is not taken for the end of the file.
void TestPollerCapsBytesPerPoll() {
	const auto         generated = MakeGenerated(RECORD_SIZES);
	constexpr uint64_t CAP       = 300;
	CHECK(generated.file.size() > 5 * CAP);
	for (const bool exited: {false, true}) {
		PolledFile file("cap", CAP);
		WriteAt(file.path, 0, generated.file, true);
		std::vector<TailedFrame> got;
		uint64_t                 before = 0;
		size_t                   polls  = 0;
		while (file.poller.BytesRead() < generated.file.size() && polls < 100) {
			Append(got, file.poller.Poll(exited));
			CHECK(file.poller.BytesRead() - before <= CAP);
			CHECK(!file.tailer.IsFinal() || file.poller.BytesRead() == generated.file.size());
			before = file.poller.BytesRead();
			polls++;
		}
		CHECK(polls >= 5);
		Append(got, file.poller.Poll(exited));
		CHECK(got.size() == RECORD_SIZES.size());
		for (size_t index = 0; index < got.size() && index < RECORD_SIZES.size(); index++) {
			CHECK(got[index].raw == generated.records[index]);
			CHECK(got[index].end_offset == generated.ends[index]);
		}
		CHECK(file.tailer.Status() == (exited ? TailStatus::Complete : TailStatus::Waiting));
	}
}

} // namespace

int main() {
	TestChunkedFeeding();
	TestFramesAreHandedOverOnce();
	TestBufferHoldsOnlyTheTail();
	TestTornFrameWhileAlive();
	TestCorruptTrailingFrame();
	TestCorruptFrameBeforeMoreBytes();
	TestFreshPollJudgesItemBehindDeliveredFrame();
	TestShortTailAfterExit();
	TestOversizeFrame();
	TestHeaderMismatch();
	TestIncompleteHeaderWaits();
	TestIdentityFromFile();
	TestPollerMissingFile();
	TestPollerFollowsOpenWriter();
	TestPollerRecoversFrameCaughtMidWrite();
	TestPollerBadFrameBeforeMoreBytes();
	TestPollerStaleBytesAreNotFinal();
	TestPollerRestartStateCoversOnePoll();
	TestExitDoesNotJudgeStaleTrailingItem();
	TestPollerFileGoneAfterFrames();
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	TestPollerBusyFileIsNotTruncated();
#endif
	TestPollerShrinkBelowGoodOffset();
	TestPollerRecoversHeaderAndSizeCaughtMidWrite();
	TestPollerRestart();
	TestPollerFileShrinks();
	TestPollerCapsBytesPerPoll();
	if (g_failures != 0) {
		std::fprintf(stderr, "%d check(s) failed\n", g_failures);
		return 1;
	}
	std::printf("shader store tailer tests passed\n");
	return 0;
}

// The typed IR's implementation, amalgamated to keep this target small like
// shader_store_record_tests. The record framing links the plan code, which needs it.
#include "../src/graphics/shader/recompiler/ir/Block.cpp"
#include "../src/graphics/shader/recompiler/ir/Program.cpp"
#include "../src/graphics/shader/recompiler/ir/Type.cpp"
#include "../src/graphics/shader/recompiler/ir/Value.cpp"
#include "../src/graphics/shader/recompiler/ir/opcodes/ValueOpcodes.cpp"
