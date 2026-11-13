// Tests the frames a Stream worker and its supervisor exchange and the pipe they travel through.
// The frames are synthetic, and the pipes are made inside this process, so no worker, no game file
// and no device are needed. Every test names what else could pass it and compares against a
// baseline that differs only in the behaviour under test.
#include "common/childProcess.h"
#include "common/config.h"
#include "graphics/shader/recompiler/ir/Serialize.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"
#include "graphics/shader/storegen/streamChannel.h"
#include "graphics/shader/storegen/streamProtocol.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <new>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Libs::Graphics;
using namespace Libs::Graphics::StoreGen;

int g_failures = 0;

// Counts the allocations made while it is switched on, from every thread, by the operator new below.
std::atomic<bool>     g_count_allocations {false};
std::atomic<uint64_t> g_allocation_count {0};

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

using Bytes = std::vector<uint8_t>;

/// How long the whole run may take before the watchdog fails it, so a read that never returns
/// fails the run rather than hanging it.
constexpr std::chrono::seconds TEST_TIMEOUT {120};

/// Where the header's fields sit, as streamProtocol.h lays them out.
constexpr size_t HEADER_VERSION_OFFSET  = 4;
constexpr size_t HEADER_KIND_OFFSET     = 6;
constexpr size_t HEADER_RESERVED_OFFSET = 7;
constexpr size_t HEADER_LENGTH_OFFSET   = 8;
/// Where fields sit in the payloads. A request starts with seq (8 bytes) and the kind, and a
/// response with seq, the kind and the status. The request's first flag, dx10_clamp, follows the
/// code hash, data_addr, three thread counts and three byte registers.
constexpr size_t REQUEST_KIND_OFFSET  = STREAM_FRAME_HEADER_BYTES + 8;
constexpr size_t RESPONSE_STATUS_OFFSET = STREAM_FRAME_HEADER_BYTES + 9;
constexpr size_t REQUEST_FIRST_FLAG_OFFSET = STREAM_FRAME_HEADER_BYTES + 8 + 1 + 8 + 8 + 12 + 3;
/// The two length words after each payload's fixed fields. The request's fixed fields take 88
/// bytes (seq 8, kind 1, hash 8, dispatch 63, user data count 4, scratch 4), and the response's 10.
constexpr size_t REQUEST_FIRST_LENGTH_OFFSET   = STREAM_FRAME_HEADER_BYTES + 88;
constexpr size_t REQUEST_SECOND_LENGTH_OFFSET  = REQUEST_FIRST_LENGTH_OFFSET + 4;
constexpr size_t RESPONSE_FIRST_LENGTH_OFFSET  = STREAM_FRAME_HEADER_BYTES + 10;
constexpr size_t RESPONSE_SECOND_LENGTH_OFFSET = RESPONSE_FIRST_LENGTH_OFFSET + 4;
/// Added to a length so that the true sum or product only differs from the frame's size above 32
/// bits. Twice 2^31 is 2^32, and four times 2^30 is 2^32.
constexpr uint32_t HALF_WRAP    = 0x80000000u;
constexpr uint32_t QUARTER_WRAP = 0x40000000u;

/// A payload size for a frame far larger than the pipe's 64 KiB buffer.
constexpr size_t LARGE_PAYLOAD_BYTES = size_t {1} << 20u;
/// How many large frames each side sends in the full duplex test.
constexpr int DUPLEX_FRAMES = 3;
/// How long the accept test waits for a client that never comes, and the most it may take.
constexpr std::chrono::milliseconds SHORT_ACCEPT {50};
constexpr std::chrono::milliseconds ACCEPT_TIMEOUT_LIMIT {5000};
constexpr std::chrono::milliseconds CONNECTED_ACCEPT {1000};
/// How long a blocked call has to wake after Interrupt, and how long the test lets it block
/// first, so that the call is really waiting inside the system when it is interrupted.
constexpr std::chrono::milliseconds WAKE_LIMIT {1000};
constexpr std::chrono::milliseconds BLOCK_FIRST {100};
/// How long a call made after Interrupt may take, which is far less than any wait it could do.
constexpr std::chrono::milliseconds AT_ONCE_LIMIT {200};
/// How many equal frames the allocation tests read after the warm-up frame.
constexpr int FRAMES_AFTER_WARMUP = 7;
/// How long the connect retry test lets a connect keep trying, far longer than the test needs.
constexpr std::chrono::milliseconds CONNECT_RETRY_LIMIT {5000};

// A request with every field set to a value of its own, and every flag true, since each defaults to
// false and a flag the frame lost would come back false.
StreamRequest FullRequest(StreamRequestKind kind) {
	StreamRequest request;
	request.seq                       = 0x1122334455667788ull;
	request.kind                      = kind;
	request.code_hash                 = 0xA1B2C3D4E5F60718ull;
	request.code                      = {0xBE, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06};
	auto& regs                        = request.dispatch.cs_regs;
	regs.data_addr                    = 0x0000004567890100ull;
	regs.num_thread_x                 = 64;
	regs.num_thread_y                 = 2;
	regs.num_thread_z                 = 3;
	regs.vgprs                        = 11;
	regs.priority                     = 2;
	regs.float_mode                   = 0xC0;
	regs.dx10_clamp                   = true;
	regs.debug_mode                   = true;
	regs.ieee_mode                    = true;
	regs.require_forward_progress     = true;
	regs.fp16_overflow                = true;
	regs.threadgroup_configuration    = true;
	regs.wave_size                    = 32;
	regs.scratch_en                   = true;
	regs.user_sgpr                    = 12;
	regs.tgid_x_en                    = true;
	regs.tgid_y_en                    = true;
	regs.tgid_z_en                    = true;
	regs.tg_size_en                   = true;
	regs.tidig_comp_cnt               = 2;
	regs.lds_size                     = 0x1234;
	regs.shared_vgprs                 = 7;
	request.dispatch.async_compute    = true;
	request.dispatch.thread_dimensions = true;
	request.dispatch.indirect         = true;
	request.dispatch.workgroup_counts = {5, 6, 7};
	request.dispatch.indirect_args_address = 0x0000001234567890ull;
	request.user_data_count           = 12;
	request.scratch_size_dwords       = 0x400;
	request.specialization            = {9, 8, 7, 6, 5};
	return request;
}

// Whether two register snapshots are equal, field by field, written out here rather than through
// the frame's own field list, so a field that list misses is still compared.
bool SameRegisters(const HW::CsStageRegisters& a, const HW::CsStageRegisters& b) {
	return a.data_addr == b.data_addr && a.num_thread_x == b.num_thread_x &&
	       a.num_thread_y == b.num_thread_y && a.num_thread_z == b.num_thread_z &&
	       a.vgprs == b.vgprs && a.priority == b.priority && a.float_mode == b.float_mode &&
	       a.dx10_clamp == b.dx10_clamp && a.debug_mode == b.debug_mode &&
	       a.ieee_mode == b.ieee_mode && a.require_forward_progress == b.require_forward_progress &&
	       a.fp16_overflow == b.fp16_overflow &&
	       a.threadgroup_configuration == b.threadgroup_configuration &&
	       a.wave_size == b.wave_size && a.scratch_en == b.scratch_en &&
	       a.user_sgpr == b.user_sgpr && a.tgid_x_en == b.tgid_x_en &&
	       a.tgid_y_en == b.tgid_y_en && a.tgid_z_en == b.tgid_z_en &&
	       a.tg_size_en == b.tg_size_en && a.tidig_comp_cnt == b.tidig_comp_cnt &&
	       a.lds_size == b.lds_size && a.shared_vgprs == b.shared_vgprs;
}

// Whether every field of two requests is equal.
bool SameRequest(const StreamRequest& a, const StreamRequest& b) {
	const auto& da = a.dispatch;
	const auto& db = b.dispatch;
	return a.seq == b.seq && a.kind == b.kind && a.code_hash == b.code_hash && a.code == b.code &&
	       SameRegisters(da.cs_regs, db.cs_regs) && da.async_compute == db.async_compute &&
	       da.thread_dimensions == db.thread_dimensions && da.indirect == db.indirect &&
	       da.workgroup_counts == db.workgroup_counts &&
	       da.indirect_args_address == db.indirect_args_address &&
	       a.user_data_count == b.user_data_count &&
	       a.scratch_size_dwords == b.scratch_size_dwords && a.specialization == b.specialization;
}

// A response with every field set to a value of its own. Only an Ok response carries a payload,
// as the format requires.
StreamResponse FullResponse(StreamRequestKind kind, StreamStatus status) {
	StreamResponse response;
	response.seq    = 0x8877665544332211ull;
	response.kind   = kind;
	response.status = status;
	if (status == StreamStatus::Ok) {
		response.payload = {0x10, 0x20, 0x30, 0x40, 0x50};
	}
	response.static_state = {0xDEADBEEFu, 1u, 0x80000000u};
	return response;
}

// Whether every field of two responses is equal.
bool SameResponse(const StreamResponse& a, const StreamResponse& b) {
	return a.seq == b.seq && a.kind == b.kind && a.status == b.status && a.payload == b.payload &&
	       a.static_state == b.static_state;
}

Bytes RequestFrame(const StreamRequest& request) {
	Bytes frame;
	CHECK(EncodeStreamRequest(request, frame));
	return frame;
}

Bytes ResponseFrame(const StreamResponse& response) {
	Bytes frame;
	CHECK(EncodeStreamResponse(response, frame));
	return frame;
}

// A byte source over fixed bytes that stops each read at the next cut, the way a pipe hands over
// what has arrived so far. Reading allocates nothing, so it can stand in for the pipe in the
// allocation test.
class CutSource final: public StreamByteSource {
public:
	CutSource(std::span<const uint8_t> bytes, std::span<const size_t> cuts)
	    : m_bytes(bytes), m_cuts(cuts) {}

	size_t ReadSome(std::span<uint8_t> into) override {
		m_reads++;
		while (!m_cuts.empty() && m_cuts.front() <= m_offset) {
			m_cuts = m_cuts.subspan(1);
		}
		const size_t limit = m_cuts.empty() ? m_bytes.size() : std::min(m_cuts.front(), m_bytes.size());
		const size_t count = std::min(into.size(), limit - m_offset);
		std::copy_n(m_bytes.begin() + static_cast<ptrdiff_t>(m_offset), count, into.begin());
		m_offset += count;
		return count;
	}

	[[nodiscard]] uint32_t Reads() const { return m_reads; }

private:
	std::span<const uint8_t> m_bytes;
	std::span<const size_t>  m_cuts;
	size_t                   m_offset = 0;
	uint32_t                 m_reads  = 0;
};

// A copy of a frame with one 32-bit word replaced.
Bytes WithWord(const Bytes& frame, size_t offset, uint32_t value) {
	auto changed = frame;
	std::memcpy(changed.data() + offset, &value, sizeof(value));
	return changed;
}

// The 32-bit word at an offset of a frame.
uint32_t WordAt(const Bytes& frame, size_t offset) {
	uint32_t value = 0;
	std::memcpy(&value, frame.data() + offset, sizeof(value));
	return value;
}

// Whether a reader hands over exactly these bytes as a frame.
bool ReadsFrame(StreamFrameReader& reader, StreamByteSource& source, std::span<const uint8_t> frame) {
	const auto read = reader.ReadFrame(source);
	return read.status == StreamReadStatus::Frame && std::ranges::equal(read.frame, frame);
}

// ---- Envelopes. ----

// Every field of a request of each kind comes back. A decoder that dropped a field would leave it
// at its default, and every value here differs from its default. The comparison could itself miss a
// field, so the baseline changes one field at a time and must compare unequal.
void TestRequestRoundTrip() {
	for (const auto kind: {StreamRequestKind::Plan, StreamRequestKind::Compile}) {
		const auto request = FullRequest(kind);
		const auto decoded = DecodeStreamRequest(RequestFrame(request));
		CHECK(decoded && SameRequest(*decoded, request));
	}
	const auto request = FullRequest(StreamRequestKind::Compile);
	auto       other   = request;
	other.dispatch.cs_regs.shared_vgprs = 8;
	CHECK(!SameRequest(other, request));
	other                          = request;
	other.dispatch.cs_regs.tg_size_en = false;
	CHECK(!SameRequest(other, request));
	other                                  = request;
	other.dispatch.indirect_args_address = 1;
	CHECK(!SameRequest(other, request));
	other                = request;
	other.specialization = {};
	CHECK(!SameRequest(other, request));
	// Empty byte strings are a payload of their own length, and still come back.
	StreamRequest empty;
	const auto    decoded_empty = DecodeStreamRequest(RequestFrame(empty));
	CHECK(decoded_empty && SameRequest(*decoded_empty, empty));
}

// Every kind and status of a response comes back. A decoder that gave every response one status
// would pass for that status alone, so all four are checked against both kinds.
void TestResponseRoundTrip() {
	for (const auto kind: {StreamRequestKind::Plan, StreamRequestKind::Compile}) {
		for (const auto status: {StreamStatus::Ok, StreamStatus::HasCalls, StreamStatus::Dropped,
		                         StreamStatus::Failed}) {
			const auto response = FullResponse(kind, status);
			const auto decoded  = DecodeStreamResponse(ResponseFrame(response));
			CHECK(decoded && SameResponse(*decoded, response));
		}
	}
	const auto response = FullResponse(StreamRequestKind::Plan, StreamStatus::Ok);
	auto       other    = response;
	other.static_state.back() ^= 1u;
	CHECK(!SameResponse(other, response));
	other        = response;
	other.status = StreamStatus::Failed;
	CHECK(!SameResponse(other, response));
}

// Every proper prefix of a frame is refused, and so is a frame with a byte after it. Each prefix is
// a view of the whole frame, so a decoder that ignored the view's length would read the real bytes
// past its end and decode them. Only the length check refuses them, which makes the mutation that
// skips it fail here every time rather than by chance. The baseline is the whole frame, which
// decodes.
void TestTruncatedFrameRefused() {
	const auto request = RequestFrame(FullRequest(StreamRequestKind::Compile));
	CHECK(DecodeStreamRequest(request));
	for (size_t length = 0; length < request.size(); length++) {
		CHECK(!DecodeStreamRequest(std::span<const uint8_t>(request).first(length)));
	}
	auto longer = request;
	longer.push_back(0);
	CHECK(!DecodeStreamRequest(longer));

	const auto response = ResponseFrame(FullResponse(StreamRequestKind::Plan, StreamStatus::Ok));
	CHECK(DecodeStreamResponse(response));
	for (size_t length = 0; length < response.size(); length++) {
		CHECK(!DecodeStreamResponse(std::span<const uint8_t>(response).first(length)));
	}
	CHECK(ParseStreamFrameHeader(std::span<const uint8_t>(request).first(STREAM_FRAME_HEADER_BYTES - 1))
	          .error == StreamHeaderError::Short);
}

// A frame with another magic is refused by the header parse, by the decoders and by the reader,
// which stops for good. The baseline is the same frame with its magic intact. A reader that went on
// after a bad header would read the second frame here, so it must not read again.
void TestWrongMagicRefused() {
	const auto good = RequestFrame(FullRequest(StreamRequestKind::Plan));
	auto       bad  = good;
	bad[0] ^= 0xFF;
	CHECK(ParseStreamFrameHeader(good).error == StreamHeaderError::None);
	CHECK(ParseStreamFrameHeader(bad).error == StreamHeaderError::BadMagic);
	CHECK(DecodeStreamRequest(good) && !DecodeStreamRequest(bad));

	Bytes stream = bad;
	stream.insert(stream.end(), good.begin(), good.end());
	CutSource         source(stream, {});
	StreamFrameReader reader;
	const auto        first = reader.ReadFrame(source);
	CHECK(first.status == StreamReadStatus::BadHeader &&
	      first.header_error == StreamHeaderError::BadMagic);
	const uint32_t reads = source.Reads();
	CHECK(reader.ReadFrame(source).status == StreamReadStatus::BadHeader);
	CHECK(source.Reads() == reads);
}

// Another protocol version is refused. The baseline is the same frame at the current version.
void TestVersionMismatchRefused() {
	const auto good = ResponseFrame(FullResponse(StreamRequestKind::Compile, StreamStatus::Ok));
	auto       newer = good;
	newer[HEADER_VERSION_OFFSET]++;
	CHECK(DecodeStreamResponse(good));
	CHECK(ParseStreamFrameHeader(newer).error == StreamHeaderError::BadVersion);
	CHECK(!DecodeStreamResponse(newer));
	CutSource         source(newer, {});
	StreamFrameReader reader;
	CHECK(reader.ReadFrame(source).header_error == StreamHeaderError::BadVersion);
}

// Sets one byte of a copy of a frame.
Bytes WithByte(const Bytes& frame, size_t offset, uint8_t value) {
	auto changed    = frame;
	changed[offset] = value;
	return changed;
}

// Kinds, statuses and flags that do not exist are refused, and so is a frame of the other kind. The
// baselines are the unchanged frames, which decode, so each refusal comes from the byte changed.
void TestUnknownKindRefused() {
	const auto request  = RequestFrame(FullRequest(StreamRequestKind::Plan));
	const auto response = ResponseFrame(FullResponse(StreamRequestKind::Plan, StreamStatus::Ok));
	CHECK(DecodeStreamRequest(request) && DecodeStreamResponse(response));
	for (const uint8_t kind: {uint8_t {0}, uint8_t {3}, uint8_t {0xFF}}) {
		const auto changed = WithByte(request, HEADER_KIND_OFFSET, kind);
		CHECK(ParseStreamFrameHeader(changed).error == StreamHeaderError::UnknownKind);
		CHECK(!DecodeStreamRequest(changed));
	}
	CHECK(!DecodeStreamRequest(WithByte(request, HEADER_RESERVED_OFFSET, 1)));
	// A request kind and a response status that do not exist.
	CHECK(!DecodeStreamRequest(WithByte(request, REQUEST_KIND_OFFSET, 3)));
	CHECK(!DecodeStreamRequest(WithByte(request, REQUEST_KIND_OFFSET, 0)));
	CHECK(!DecodeStreamResponse(WithByte(response, RESPONSE_STATUS_OFFSET, 9)));
	CHECK(!DecodeStreamResponse(WithByte(response, RESPONSE_STATUS_OFFSET, 0)));
	// A flag byte that is neither 0 nor 1. The baseline sets it to 0, which decodes as false.
	const auto cleared = DecodeStreamRequest(WithByte(request, REQUEST_FIRST_FLAG_OFFSET, 0));
	CHECK(cleared && !cleared->dispatch.cs_regs.dx10_clamp);
	CHECK(!DecodeStreamRequest(WithByte(request, REQUEST_FIRST_FLAG_OFFSET, 2)));
	// Each decoder takes only its own frame kind.
	CHECK(!DecodeStreamResponse(request) && !DecodeStreamRequest(response));
}

// A length over the cap is refused from the header alone, before the reader grows its buffer for
// it. The baseline is the same header at the cap, which the reader waits for and then finds cut.
void TestOversizeLengthRefused() {
	auto       frame = RequestFrame(FullRequest(StreamRequestKind::Plan));
	const auto over  = MAX_STREAM_PAYLOAD_BYTES + 1;
	std::memcpy(frame.data() + HEADER_LENGTH_OFFSET, &over, sizeof(over));
	CHECK(ParseStreamFrameHeader(frame).error == StreamHeaderError::TooLarge);
	CutSource         source(frame, {});
	StreamFrameReader reader;
	CHECK(reader.ReadFrame(source).header_error == StreamHeaderError::TooLarge);
	CHECK(reader.BufferBytes() == STREAM_FRAME_HEADER_BYTES);

	const auto at_cap = MAX_STREAM_PAYLOAD_BYTES;
	std::memcpy(frame.data() + HEADER_LENGTH_OFFSET, &at_cap, sizeof(at_cap));
	CHECK(ParseStreamFrameHeader(frame).error == StreamHeaderError::None);
	CutSource         at_cap_source(frame, {});
	StreamFrameReader at_cap_reader;
	CHECK(at_cap_reader.ReadFrame(at_cap_source).status == StreamReadStatus::Truncated);
	CHECK(at_cap_reader.BufferBytes() == STREAM_FRAME_HEADER_BYTES + MAX_STREAM_PAYLOAD_BYTES);
}

// The length words inside a payload must add up to it exactly, even with the header's length left
// alone. Only these words differ from the baseline frames, which decode. The baseline also reads the
// words back, so the patches are known to hit them. The wrapped cases differ from a fitting sum or
// product only above 32 bits, so a check done in 32 bits would let them through.
void TestInnerLengthsMustFillPayload() {
	const auto request  = RequestFrame(FullRequest(StreamRequestKind::Compile));
	const auto response = ResponseFrame(FullResponse(StreamRequestKind::Plan, StreamStatus::Ok));
	CHECK(DecodeStreamRequest(request) && DecodeStreamResponse(response));
	CHECK(WordAt(request, REQUEST_FIRST_LENGTH_OFFSET) == 8 &&
	      WordAt(request, REQUEST_SECOND_LENGTH_OFFSET) == 5);
	CHECK(WordAt(response, RESPONSE_FIRST_LENGTH_OFFSET) == 5 &&
	      WordAt(response, RESPONSE_SECOND_LENGTH_OFFSET) == 3);
	// One byte too many, and one too few.
	CHECK(!DecodeStreamRequest(WithWord(request, REQUEST_FIRST_LENGTH_OFFSET, 9)));
	CHECK(!DecodeStreamResponse(WithWord(response, RESPONSE_FIRST_LENGTH_OFFSET, 4)));
	// Four bytes per word wraps a 32-bit product back to the true size.
	CHECK(!DecodeStreamResponse(WithWord(response, RESPONSE_SECOND_LENGTH_OFFSET, 3 + QUARTER_WRAP)));
	// Two lengths whose 32-bit sum wraps back to the true size.
	const auto wrapped = WithWord(WithWord(request, REQUEST_FIRST_LENGTH_OFFSET, 8 + HALF_WRAP),
	                              REQUEST_SECOND_LENGTH_OFFSET, 5 + HALF_WRAP);
	CHECK(!DecodeStreamRequest(wrapped));
}

// A payload travels only with the Ok status. A response that says Failed and still carries bytes
// neither encodes nor decodes. The Failed frame is the Ok frame with its status byte changed, so the
// refusal comes from that byte. The baselines are the Ok frame and a Failed frame with no payload,
// which both decode.
void TestPayloadOnlyWithOk() {
	auto  response = FullResponse(StreamRequestKind::Plan, StreamStatus::Ok);
	Bytes frame;
	CHECK(EncodeStreamResponse(response, frame) && DecodeStreamResponse(frame));
	response.status = StreamStatus::Failed;
	CHECK(!EncodeStreamResponse(response, frame) && frame.empty());
	const auto ok_frame = ResponseFrame(FullResponse(StreamRequestKind::Plan, StreamStatus::Ok));
	CHECK(!DecodeStreamResponse(WithByte(ok_frame, RESPONSE_STATUS_OFFSET,
	                                     static_cast<uint8_t>(StreamStatus::Failed))));
	CHECK(DecodeStreamResponse(ResponseFrame(FullResponse(StreamRequestKind::Plan, StreamStatus::Failed))));
}

// ---- Frame reader. ----

// A frame cut at any one point arrives whole, and so do two frames in a row with cuts inside the
// header and the payload, followed by a clean close. A reader that took one read for a frame would
// hand over a frame with stale bytes. The cut sources prove each frame took more than one read, so
// the test cannot pass on frames that arrived in one piece.
void TestSplitFrameReassembles() {
	const auto first  = RequestFrame(FullRequest(StreamRequestKind::Plan));
	const auto second = ResponseFrame(FullResponse(StreamRequestKind::Compile, StreamStatus::Ok));
	for (size_t cut = 1; cut < first.size(); cut++) {
		const size_t      cuts[] = {cut};
		CutSource         source(first, cuts);
		StreamFrameReader reader;
		CHECK(ReadsFrame(reader, source, first));
		CHECK(source.Reads() >= 2);
	}
	Bytes stream = first;
	stream.insert(stream.end(), second.begin(), second.end());
	const size_t      cuts[] = {5, STREAM_FRAME_HEADER_BYTES + 10, first.size() + 3,
	                            first.size() + STREAM_FRAME_HEADER_BYTES + 2};
	CutSource         source(stream, cuts);
	StreamFrameReader reader;
	CHECK(ReadsFrame(reader, source, first));
	CHECK(ReadsFrame(reader, source, second));
	CHECK(source.Reads() >= 6);
	CHECK(reader.ReadFrame(source).status == StreamReadStatus::Closed);
}

// A stream that ends inside a frame is Truncated, whether it ends in the header or the payload, and
// never hands over the partial frame. The baseline is a stream that ends right after a frame, which
// hands the frame over and is then Closed.
void TestStreamEndingInsideFrame() {
	const auto frame = RequestFrame(FullRequest(StreamRequestKind::Compile));
	for (const size_t keep: {size_t {1}, STREAM_FRAME_HEADER_BYTES - 1, STREAM_FRAME_HEADER_BYTES,
	                         frame.size() - 1}) {
		CutSource         source(std::span<const uint8_t>(frame).first(keep), {});
		StreamFrameReader reader;
		CHECK(reader.ReadFrame(source).status == StreamReadStatus::Truncated);
	}
	CutSource         whole(frame, {});
	StreamFrameReader reader;
	CHECK(ReadsFrame(reader, whole, frame));
	CHECK(reader.ReadFrame(whole).status == StreamReadStatus::Closed);
	CutSource         nothing(std::span<const uint8_t> {}, {});
	StreamFrameReader empty_reader;
	CHECK(empty_reader.ReadFrame(nothing).status == StreamReadStatus::Closed);
}

// Reads frames with the allocation counter on, after one warm-up frame read with it on too.
// Returns the allocations of the warm-up read and of the reads after it.
template <class ReadOne> std::pair<uint64_t, uint64_t> CountReadAllocations(ReadOne read_one) {
	g_allocation_count = 0;
	g_count_allocations = true;
	const bool warm     = read_one();
	g_count_allocations = false;
	const uint64_t warmup = g_allocation_count.exchange(0);
	bool all_read         = warm;
	g_count_allocations   = true;
	for (int frame = 0; frame < FRAMES_AFTER_WARMUP; frame++) {
		all_read = read_one() && all_read;
	}
	g_count_allocations = false;
	CHECK(all_read);
	return {warmup, g_allocation_count.exchange(0)};
}

// Once the reader has seen a frame of a size, reading more frames of that size allocates nothing,
// from a cut source and from a real pipe. The counter could miss the reader's allocations, so the
// baseline is the warm-up read, where the reader grows its buffer and the counter must see it.
void TestReadPathAllocatesNothingAfterWarmup() {
	const auto frame = RequestFrame(FullRequest(StreamRequestKind::Compile));
	Bytes      stream;
	for (int copy = 0; copy <= FRAMES_AFTER_WARMUP; copy++) {
		stream.insert(stream.end(), frame.begin(), frame.end());
	}
	const size_t      cuts[] = {3, frame.size() + 20, 2 * frame.size() + 1};
	CutSource         source(stream, cuts);
	StreamFrameReader reader;
	const auto [warmup, after] = CountReadAllocations(
	    [&] { return reader.ReadFrame(source).status == StreamReadStatus::Frame; });
	CHECK(warmup > 0);
	CHECK(after == 0);

	auto pair = CreateStreamPipePair();
	CHECK(pair.has_value());
	if (!pair) {
		return;
	}
	StreamChannel sender(std::move(pair->first));
	StreamChannel receiver(std::move(pair->second));
	// All the frames fit in the pipe's buffer, so they are written before any read.
	for (int copy = 0; copy <= FRAMES_AFTER_WARMUP; copy++) {
		CHECK(sender.Send(FullRequest(StreamRequestKind::Compile)));
	}
	const auto [pipe_warmup, pipe_after] = CountReadAllocations(
	    [&] { return receiver.ReadFrame().status == StreamReadStatus::Frame; });
	CHECK(pipe_warmup > 0);
	CHECK(pipe_after == 0);
}

// ---- Pipe. ----

// A request goes one way and a response the other through a channel pair, and each decodes to what
// was sent. A pair wired to itself would hand a channel its own request, so each side checks it got
// the other kind. Closing one end is read as Closed by the other, and a send to a closed end fails.
void TestPipeRoundTrip() {
	auto pair = CreateStreamPipePair();
	CHECK(pair.has_value());
	if (!pair) {
		return;
	}
	StreamChannel supervisor(std::move(pair->first));
	StreamChannel worker(std::move(pair->second));
	for (const auto kind: {StreamRequestKind::Plan, StreamRequestKind::Compile}) {
		const auto request = FullRequest(kind);
		CHECK(supervisor.Send(request));
		const auto got_request = worker.ReadFrame();
		CHECK(got_request.status == StreamReadStatus::Frame);
		const auto decoded_request = DecodeStreamRequest(got_request.frame);
		CHECK(decoded_request && SameRequest(*decoded_request, request));

		const auto response = FullResponse(kind, StreamStatus::HasCalls);
		CHECK(worker.Send(response));
		const auto got_response = supervisor.ReadFrame();
		CHECK(got_response.status == StreamReadStatus::Frame);
		const auto decoded_response = DecodeStreamResponse(got_response.frame);
		CHECK(decoded_response && SameResponse(*decoded_response, response));
	}
	worker.Close();
	CHECK(supervisor.ReadFrame().status == StreamReadStatus::Closed);
	CHECK(!supervisor.Send(FullRequest(StreamRequestKind::Plan)));
}

// A payload of a megabyte whose bytes follow no short pattern, so a piece put in the wrong place or
// lost would change it.
std::vector<uint8_t> LargePayload(uint32_t seed) {
	std::vector<uint8_t> payload(LARGE_PAYLOAD_BYTES);
	for (size_t index = 0; index < payload.size(); index++) {
		payload[index] = static_cast<uint8_t>(index * 31u + (index >> 11u) + seed);
	}
	return payload;
}

// A frame much larger than the pipe's buffer crosses while the other side is still writing it, so
// the reader must wait for it piece by piece. The decoded payload equals what was sent, which
// proves the pieces arrived whole and in order.
void TestLargeFrameOverPipe() {
	auto pair = CreateStreamPipePair();
	CHECK(pair.has_value());
	if (!pair) {
		return;
	}
	auto response    = FullResponse(StreamRequestKind::Compile, StreamStatus::Ok);
	response.payload = LargePayload(0);
	const auto  frame = ResponseFrame(response);
	StreamPipe  writer_end(std::move(pair->second));
	bool        written = false;
	std::thread writer([&] { written = writer_end.WriteAll(frame); });
	StreamFrameReader reader;
	const auto        read = reader.ReadFrame(pair->first);
	writer.join();
	CHECK(written);
	CHECK(read.status == StreamReadStatus::Frame);
	const auto decoded = DecodeStreamResponse(read.frame);
	CHECK(decoded && SameResponse(*decoded, response));
}

// Both ends send several large frames at once while each drains the other on its own thread. Each
// frame fills the pipe many times over, so a pipe that could not read and write at the same time
// on one end would leave both writers waiting for ever, which the watchdog fails. Every frame must
// arrive and decode to what was sent.
void TestFullDuplex() {
	auto pair = CreateStreamPipePair();
	CHECK(pair.has_value());
	if (!pair) {
		return;
	}
	StreamChannel left(std::move(pair->first));
	StreamChannel right(std::move(pair->second));
	auto          request = FullRequest(StreamRequestKind::Compile);
	request.code          = LargePayload(1);
	auto response         = FullResponse(StreamRequestKind::Compile, StreamStatus::Ok);
	response.payload      = LargePayload(2);
	std::atomic<int> good_requests {0};
	std::atomic<int> good_responses {0};
	std::thread      right_reader([&] {
        for (int frame = 0; frame < DUPLEX_FRAMES; frame++) {
            const auto read    = right.ReadFrame();
            const auto decoded = DecodeStreamRequest(read.frame);
            good_requests += decoded && SameRequest(*decoded, request) ? 1 : 0;
        }
    });
	std::thread left_reader([&] {
		for (int frame = 0; frame < DUPLEX_FRAMES; frame++) {
			const auto read    = left.ReadFrame();
			const auto decoded = DecodeStreamResponse(read.frame);
			good_responses += decoded && SameResponse(*decoded, response) ? 1 : 0;
		}
	});
	std::atomic<bool> left_sent {true};
	std::thread       left_writer([&] {
        for (int frame = 0; frame < DUPLEX_FRAMES; frame++) {
            left_sent = left.Send(request) && left_sent;
        }
    });
	bool right_sent = true;
	for (int frame = 0; frame < DUPLEX_FRAMES; frame++) {
		right_sent = right.Send(response) && right_sent;
	}
	left_writer.join();
	right_reader.join();
	left_reader.join();
	CHECK(left_sent && right_sent);
	CHECK(good_requests == DUPLEX_FRAMES && good_responses == DUPLEX_FRAMES);
}

// Sends one byte from one end and reads it at the other, which proves the two are connected.
bool ByteCrosses(StreamPipe& from, StreamPipe& to) {
	constexpr uint8_t MARK   = 0x5A;
	const uint8_t     sent[] = {MARK};
	uint8_t           got[]  = {0};
	return from.WriteAll(sent) && to.ReadSome(got) == 1 && got[0] == MARK;
}

// A listener with no client gives up after about its timeout and can still take a client
// afterwards. One that ignored the timeout would hang, which the watchdog fails, one that did not
// wait at all fails the lower bound, and one that returned a pipe without a client fails the first
// check. The byte that crosses proves the later accept joined the client and the server.
void TestAcceptTimesOut() {
	const uint32_t own      = Common::CurrentProcessId();
	auto           listener = StreamPipeListener::Create("accept-timeout");
	CHECK(listener.has_value());
	if (!listener) {
		return;
	}
	const auto start   = std::chrono::steady_clock::now();
	const auto nobody  = listener->Accept(SHORT_ACCEPT, own);
	const auto elapsed = std::chrono::steady_clock::now() - start;
	CHECK(!nobody.has_value());
	CHECK(elapsed >= SHORT_ACCEPT / 2);
	CHECK(elapsed < ACCEPT_TIMEOUT_LIMIT);
	auto client = ConnectStreamPipe(listener->Name());
	auto server = listener->Accept(CONNECTED_ACCEPT, own);
	CHECK(client.has_value() && server.has_value());
	if (client && server) {
		CHECK(ByteCrosses(*client, *server));
	}
	CHECK(listener->Name().empty());
}

// A client must be the process the listener expects. The client here is always this process, so
// an expected id of anything else must be refused, and the refused client is dropped. The baseline
// is the same connection with this process's id, which is accepted on the same listener.
void TestPeerProcessChecked() {
	const uint32_t own      = Common::CurrentProcessId();
	auto           listener = StreamPipeListener::Create("peer-check");
	CHECK(listener.has_value());
	if (!listener) {
		return;
	}
	auto stranger = ConnectStreamPipe(listener->Name());
	CHECK(stranger.has_value());
	// Windows process ids are multiples of four, so this one is never a real process.
	CHECK(!listener->Accept(CONNECTED_ACCEPT, own + 1).has_value());
	uint8_t byte[] = {0};
	CHECK(!stranger || stranger->ReadSome(byte) == 0);
	auto client = ConnectStreamPipe(listener->Name());
	auto server = listener->Accept(CONNECTED_ACCEPT, own);
	CHECK(client.has_value() && server.has_value());
	if (client && server) {
		CHECK(ByteCrosses(*client, *server));
	}
}

// A client that connects and leaves before Accept is not handed over, and the listener can still
// take the next client. On Windows the pipe must be disconnected for that, or the next client
// finds it busy. POSIX queues the connection, so Accept hands it over and its first read is a
// close, which is checked instead.
void TestClientLeftBeforeAccept() {
	const uint32_t own      = Common::CurrentProcessId();
	auto           listener = StreamPipeListener::Create("client-left");
	CHECK(listener.has_value());
	if (!listener) {
		return;
	}
	auto leaver = ConnectStreamPipe(listener->Name());
	CHECK(leaver.has_value());
	leaver.reset();
	auto left = listener->Accept(SHORT_ACCEPT, own);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	CHECK(!left.has_value());
	auto client = ConnectStreamPipe(listener->Name());
	auto server = listener->Accept(CONNECTED_ACCEPT, own);
	CHECK(client.has_value() && server.has_value());
	if (client && server) {
		CHECK(ByteCrosses(*client, *server));
	}
#else
	uint8_t byte[] = {0};
	CHECK(!left || left->ReadSome(byte) == 0);
#endif
}

/// What InterruptBlockedCall saw.
struct InterruptOutcome {
	bool blocked = false; ///< The call was still waiting when Interrupt came.
	bool woke    = false; ///< The call returned within WAKE_LIMIT of Interrupt.
	bool result  = false; ///< What the call returned.
};

// Runs a call that should block on a thread, interrupts the pipe once it has waited a while, and
// reports whether it woke. A call that does not wake is released by unblock, so the thread can still
// be joined and the run goes on to report the failure.
template <class Call, class Unblock>
InterruptOutcome InterruptBlockedCall(StreamPipe& pipe, Call call, Unblock unblock) {
	std::promise<bool> done;
	auto               finished = done.get_future();
	std::thread        worker([&] { done.set_value(call()); });
	std::this_thread::sleep_for(BLOCK_FIRST);
	InterruptOutcome outcome;
	outcome.blocked = finished.wait_for(std::chrono::milliseconds {0}) != std::future_status::ready;
	pipe.Interrupt();
	outcome.woke = finished.wait_for(WAKE_LIMIT) == std::future_status::ready;
	if (!outcome.woke) {
		unblock(finished);
	}
	worker.join();
	outcome.result = finished.get();
	return outcome;
}

// Whether a call returns within AT_ONCE_LIMIT, and what it returned.
template <class Call> std::pair<bool, bool> ReturnsAtOnce(Call call) {
	const auto start  = std::chrono::steady_clock::now();
	const bool result = call();
	return {std::chrono::steady_clock::now() - start < AT_ONCE_LIMIT, result};
}

// A read blocked on a pipe that nothing writes to wakes soon after Interrupt and fails. Later calls
// fail at once, even a read with a byte waiting and a write with room in the pipe, so the stop
// lasts. A pipe that only woke the blocked call would hand over that byte or send that write.
void TestStopWakesBlockedRead() {
	auto pair = CreateStreamPipePair();
	CHECK(pair.has_value());
	if (!pair) {
		return;
	}
	auto&   stopped = pair->first;
	auto&   peer    = pair->second;
	uint8_t byte[]  = {0};
	const auto outcome = InterruptBlockedCall(
	    stopped, [&] { return stopped.ReadSome(byte) != 0; },
	    [&](std::future<bool>&) { (void)peer.WriteAll(byte); });
	CHECK(outcome.blocked && outcome.woke && !outcome.result);
	// The peer's write is not checked, since on POSIX Interrupt has shut the socket down.
	(void)peer.WriteAll(byte);
	const auto [read_prompt, read_got] = ReturnsAtOnce([&] { return stopped.ReadSome(byte) != 0; });
	CHECK(read_prompt && !read_got);
	const auto [write_prompt, written] = ReturnsAtOnce([&] { return stopped.WriteAll(byte); });
	CHECK(write_prompt && !written);
}

// A write blocked on a pipe whose buffer is full, since nothing reads the other end, wakes soon
// after Interrupt and fails, and a later write fails at once.
void TestStopWakesBlockedWrite() {
	auto pair = CreateStreamPipePair();
	CHECK(pair.has_value());
	if (!pair) {
		return;
	}
	auto&      stopped = pair->first;
	auto&      peer    = pair->second;
	const auto big     = LargePayload(3);
	const auto outcome = InterruptBlockedCall(
	    stopped, [&] { return stopped.WriteAll(big); },
	    [&](std::future<bool>&) {
		    // A write that ignored the stop writes every byte, so exactly that many are drained.
		    std::vector<uint8_t> sink(LARGE_PAYLOAD_BYTES);
		    size_t               drained = 0;
		    while (drained < big.size()) {
			    const size_t got = peer.ReadSome(sink);
			    if (got == 0) {
				    break;
			    }
			    drained += got;
		    }
	    });
	CHECK(outcome.blocked && outcome.woke && !outcome.result);
	const uint8_t byte[]             = {0};
	const auto [write_prompt, wrote] = ReturnsAtOnce([&] { return stopped.WriteAll(byte); });
	CHECK(write_prompt && !wrote);
}

// Each end of a pair sees this process at the other end, the client asking about the server and the
// server about the client. The baseline is another id, which neither end takes, and an end that is
// not open takes none. Both ends are this process, so this cannot tell which end's id the system
// gave, and the real child smoke test in ShaderStoreDispatchTests covers that.
void TestPeerProcessOfEachEnd() {
	const uint32_t own  = Common::CurrentProcessId();
	auto           pair = CreateStreamPipePair();
	CHECK(pair.has_value());
	if (!pair) {
		return;
	}
	CHECK(pair->first.IsPeerProcess(own) && pair->second.IsPeerProcess(own));
	CHECK(!pair->first.IsPeerProcess(own + 1) && !pair->second.IsPeerProcess(own + 1));
	const StreamPipe closed;
	CHECK(!closed.IsPeerProcess(own));
}

// A connect that finds the pipe busy tries again until the listener lets go of the client it has
// and takes this one. A listener serves one client, so a second client finds it busy until Accept
// refuses the first. A connect that gave up on a busy pipe fails here. The baseline is a name that
// no listener has, which fails at once rather than after the timeout, so the retry is kept to the
// busy case. POSIX queues the second client instead of refusing it, so there only the outcome is
// checked, not that a retry happened.
void TestConnectRetriesWhileBusy() {
	const uint32_t own      = Common::CurrentProcessId();
	auto           listener = StreamPipeListener::Create("connect-retry");
	CHECK(listener.has_value());
	if (!listener) {
		return;
	}
	// A copy, since a successful Accept empties the listener's name while the connector reads it.
	const std::string name  = listener->Name();
	auto              first = ConnectStreamPipe(name);
	CHECK(first.has_value());
	std::optional<StreamPipe> second;
	std::thread connector([&] { second = ConnectStreamPipeWithin(name, CONNECT_RETRY_LIMIT); });
	// Long enough that the connector meets the busy pipe before the first client is refused.
	std::this_thread::sleep_for(BLOCK_FIRST);
	CHECK(!listener->Accept(CONNECTED_ACCEPT, own + 1).has_value());
	auto server = listener->Accept(CONNECTED_ACCEPT, own);
	connector.join();
	CHECK(second.has_value() && server.has_value());
	if (second && server) {
		CHECK(ByteCrosses(*second, *server));
	}
	auto       nobody = StreamPipeListener::Create("connect-nobody");
	const auto gone   = nobody ? nobody->Name() : std::string("missing");
	nobody.reset();
	const auto [prompt, connected] = ReturnsAtOnce(
	    [&] { return ConnectStreamPipeWithin(gone, CONNECT_RETRY_LIMIT).has_value(); });
	CHECK(prompt && !connected);
}

// ---- Worker request. ----

// A request file in Stream mode decodes once it names its pipe, and the name comes back. A Stream
// request with no pipe could only fail to connect, so it is refused as a bad request. The baseline
// is the same request in Generate mode, which decodes without a pipe.
void TestStreamModeNeedsPipeName() {
	WorkerRequest request;
	request.store_header       = {1, 2, 3, 4};
	request.host_subgroup_size = 64;
	request.mode               = WorkerMode::Generate;
	CHECK(DecodeWorkerRequest(EncodeWorkerRequest(request)));
	request.mode = WorkerMode::Stream;
	CHECK(!DecodeWorkerRequest(EncodeWorkerRequest(request)));
	request.stream_pipe = R"(\\.\pipe\kyty-stream-1-test-0123456789abcdef)";
	const auto decoded  = DecodeWorkerRequest(EncodeWorkerRequest(request));
	CHECK(decoded && decoded->mode == WorkerMode::Stream &&
	      decoded->stream_pipe == request.stream_pipe);
}

// A request's fields written by hand in the order of the file, at a given version and with or
// without the stream pipe name that version 3 added.
Bytes HandEncodedRequest(const WorkerRequest& request, uint32_t version, bool with_pipe) {
	namespace IR = ShaderRecompiler::IR;
	IR::Writer writer;
	auto       fields = request;
	auto       magic  = WORKER_REQUEST_MAGIC;
	IR::IoAll(writer, magic, version, fields.store_header, fields.host_subgroup_size,
	          fields.eboot_path, fields.output_path, fields.progress_path, fields.generated_path,
	          fields.kept, fields.start_rank, fields.skip_hashes, fields.failed_hashes,
	          fields.parent_pid, fields.mode, fields.generator_version, fields.expected_order);
	if (with_pipe) {
		IR::IoAll(writer, fields.stream_pipe);
	}
	return std::move(writer.bytes);
}

// A request in the layout before the pipe name, at version 2, is refused, and so is that layout
// labelled version 3, which is cut short. The baseline is the same fields at version 3 with the
// name, which decode, and which are byte for byte what EncodeWorkerRequest writes, so the hand
// layout is the real one.
void TestOldRequestLayoutRefused() {
	WorkerRequest request;
	request.store_header       = {1, 2, 3, 4};
	request.host_subgroup_size = 64;
	request.parent_pid         = 4242;
	request.mode               = WorkerMode::Generate;
	request.stream_pipe        = "pipe";
	const auto current         = HandEncodedRequest(request, WORKER_REQUEST_VERSION, true);
	CHECK(current == EncodeWorkerRequest(request));
	CHECK(DecodeWorkerRequest(current));
	CHECK(WORKER_REQUEST_VERSION == 3);
	CHECK(!DecodeWorkerRequest(HandEncodedRequest(request, 2, false)));
	CHECK(!DecodeWorkerRequest(HandEncodedRequest(request, WORKER_REQUEST_VERSION, false)));
}

// Fails the run when the tests take far longer than they should, since a read that never returns
// would otherwise hang it.
void StartWatchdog() {
	std::thread([] {
		std::this_thread::sleep_for(TEST_TIMEOUT);
		std::fprintf(stderr, "shader_store_stream_tests: timed out\n");
		std::_Exit(1);
	}).detach();
}

} // namespace

// Counts the allocations of the tests, for TestReadPathAllocatesNothingAfterWarmup.
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

int main() {
	StartWatchdog();
	TestRequestRoundTrip();
	TestResponseRoundTrip();
	TestTruncatedFrameRefused();
	TestWrongMagicRefused();
	TestVersionMismatchRefused();
	TestUnknownKindRefused();
	TestOversizeLengthRefused();
	TestInnerLengthsMustFillPayload();
	TestPayloadOnlyWithOk();
	TestSplitFrameReassembles();
	TestStreamEndingInsideFrame();
	TestReadPathAllocatesNothingAfterWarmup();
	TestPipeRoundTrip();
	TestLargeFrameOverPipe();
	TestFullDuplex();
	TestAcceptTimesOut();
	TestPeerProcessChecked();
	TestClientLeftBeforeAccept();
	TestStopWakesBlockedRead();
	TestStopWakesBlockedWrite();
	TestPeerProcessOfEachEnd();
	TestConnectRetriesWhileBusy();
	TestStreamModeNeedsPipeName();
	TestOldRequestLayoutRefused();
	if (g_failures != 0) {
		std::fprintf(stderr, "shader_store_stream_tests: %d failures\n", g_failures);
		return 1;
	}
	std::printf("shader_store_stream_tests: all passed\n");
	return 0;
}
