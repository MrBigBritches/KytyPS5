#include "graphics/shader/storegen/streamProtocol.h"

#include <array>
#include <cstring>
#include <type_traits>

namespace Libs::Graphics::StoreGen {

namespace {

/// The two length words after a payload's fixed fields, which give the sizes of what follows.
using BlobLengths = std::array<uint32_t, 2>;

// Appends each field's bytes. A flag takes one byte holding 0 or 1, so a reader can refuse others.
class FieldWriter {
public:
	explicit FieldWriter(std::vector<uint8_t>& out): m_out(out) {}

	template <class T> void operator()(const T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		if constexpr (std::is_same_v<T, bool>) {
			m_out.push_back(value ? 1 : 0);
		} else {
			const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
			m_out.insert(m_out.end(), bytes, bytes + sizeof(T));
		}
	}

private:
	std::vector<uint8_t>& m_out;
};

// Adds up how many bytes the fields take, so the fixed part's size follows the field list.
class FieldSizer {
public:
	template <class T> void operator()(const T& /*value*/) {
		m_bytes += std::is_same_v<T, bool> ? 1 : sizeof(T);
	}

	[[nodiscard]] size_t Bytes() const { return m_bytes; }

private:
	size_t m_bytes = 0;
};

// Reads fields from bytes the caller has already checked are long enough, so it checks no bounds
// itself. A flag byte other than 0 or 1 marks the read bad.
class FieldReader {
public:
	explicit FieldReader(const uint8_t* at): m_at(at) {}

	template <class T> void operator()(T& value) {
		static_assert(std::is_trivially_copyable_v<T>);
		if constexpr (std::is_same_v<T, bool>) {
			const uint8_t byte = *m_at++;
			m_ok               = m_ok && byte <= 1;
			value              = byte != 0;
		} else {
			std::memcpy(&value, m_at, sizeof(T));
			m_at += sizeof(T);
		}
	}

	[[nodiscard]] bool           Ok() const { return m_ok; }
	[[nodiscard]] const uint8_t* At() const { return m_at; }

private:
	const uint8_t* m_at = nullptr;
	bool           m_ok = true;
};

// Hands each field to the visitor in the order given, which is the order on the wire.
template <class Visitor, class... Fields> void VisitAll(Visitor& visit, Fields&... fields) {
	(visit(fields), ...);
}

// Every register of the snapshot in declaration order. The structured binding stops compiling when
// a register is added, so the frame cannot silently leave one out.
template <class Visitor, class Registers> void VisitCsRegisters(Visitor& visit, Registers& regs) {
	auto& [data_addr, num_thread_x, num_thread_y, num_thread_z, vgprs, priority, float_mode,
	       dx10_clamp, debug_mode, ieee_mode, require_forward_progress, fp16_overflow,
	       threadgroup_configuration, wave_size, scratch_en, user_sgpr, tgid_x_en, tgid_y_en,
	       tgid_z_en, tg_size_en, tidig_comp_cnt, lds_size, shared_vgprs] = regs;
	VisitAll(visit, data_addr, num_thread_x, num_thread_y, num_thread_z, vgprs, priority,
	         float_mode, dx10_clamp, debug_mode, ieee_mode, require_forward_progress, fp16_overflow,
	         threadgroup_configuration, wave_size, scratch_en, user_sgpr, tgid_x_en, tgid_y_en,
	         tgid_z_en, tg_size_en, tidig_comp_cnt, lds_size, shared_vgprs);
}

// The dispatch inputs, bound the same way as the registers.
template <class Visitor, class Dispatch> void VisitDispatch(Visitor& visit, Dispatch& dispatch) {
	auto& [cs_regs, async_compute, thread_dimensions, indirect, workgroup_counts,
	       indirect_args_address] = dispatch;
	VisitCsRegisters(visit, cs_regs);
	VisitAll(visit, async_compute, thread_dimensions, indirect, workgroup_counts[0],
	         workgroup_counts[1], workgroup_counts[2], indirect_args_address);
}

// The fixed part of a request payload, which is every field but the two byte strings, then their
// lengths. The byte strings follow in the order of the lengths.
template <class Visitor, class Request>
void VisitRequestFixed(Visitor& visit, Request& request, BlobLengths& lengths) {
	auto& [seq, kind, code_hash, code, dispatch, user_data_count, scratch_size_dwords,
	       specialization] = request;
	VisitAll(visit, seq, kind, code_hash);
	VisitDispatch(visit, dispatch);
	VisitAll(visit, user_data_count, scratch_size_dwords, lengths[0], lengths[1]);
}

// The fixed part of a response payload, then the payload's byte count and the static state's word
// count.
template <class Visitor, class Response>
void VisitResponseFixed(Visitor& visit, Response& response, BlobLengths& lengths) {
	auto& [seq, kind, status, payload, static_state] = response;
	VisitAll(visit, seq, kind, status, lengths[0], lengths[1]);
}

// The size of a request payload's fixed part, worked out once from the field list.
size_t RequestFixedBytes() {
	static const size_t bytes = [] {
		FieldSizer          sizer;
		const StreamRequest request;
		BlobLengths         lengths {};
		VisitRequestFixed(sizer, request, lengths);
		return sizer.Bytes();
	}();
	return bytes;
}

// The size of a response payload's fixed part, worked out once from the field list.
size_t ResponseFixedBytes() {
	static const size_t bytes = [] {
		FieldSizer           sizer;
		const StreamResponse response;
		BlobLengths          lengths {};
		VisitResponseFixed(sizer, response, lengths);
		return sizer.Bytes();
	}();
	return bytes;
}

// The payload size of a response's frame, taken in 64 bits so a huge payload cannot wrap.
uint64_t ResponsePayloadBytes(const StreamResponse& response) {
	return static_cast<uint64_t>(ResponseFixedBytes()) + response.payload.size() +
	       response.static_state.size() * sizeof(uint32_t);
}

// Whether a request kind read off the wire is one that exists.
bool KnownRequestKind(StreamRequestKind kind) {
	return kind == StreamRequestKind::Plan || kind == StreamRequestKind::Compile;
}

// Whether a response status read off the wire is one that exists.
bool KnownStatus(StreamStatus status) {
	return status == StreamStatus::Ok || status == StreamStatus::HasCalls ||
	       status == StreamStatus::Dropped || status == StreamStatus::Failed;
}

// Whether a response carries a payload only with the Ok status, since any other status means there
// is no answer to carry.
bool PayloadFitsStatus(StreamStatus status, uint64_t payload_bytes) {
	return status == StreamStatus::Ok || payload_bytes == 0;
}

// Whether a header's kind byte names a frame kind that exists.
bool KnownFrameKind(uint8_t kind) {
	return kind == static_cast<uint8_t>(StreamFrameKind::Request) ||
	       kind == static_cast<uint8_t>(StreamFrameKind::Response);
}

// Starts a frame with its header. The payload length is known up front, so the frame never moves.
void BeginFrame(std::vector<uint8_t>& frame, StreamFrameKind kind, uint32_t payload_bytes) {
	frame.clear();
	frame.reserve(STREAM_FRAME_HEADER_BYTES + payload_bytes);
	FieldWriter write {frame};
	uint32_t    magic    = STREAM_FRAME_MAGIC;
	uint16_t    version  = STREAM_PROTOCOL_VERSION;
	uint8_t     reserved = 0;
	VisitAll(write, magic, version, kind, reserved, payload_bytes);
}

// Appends raw bytes to a frame, for the byte strings after a payload's fixed part.
void AppendBytes(std::vector<uint8_t>& frame, const void* data, size_t size) {
	const auto* bytes = static_cast<const uint8_t*>(data);
	frame.insert(frame.end(), bytes, bytes + size);
}

// The payload of a frame of the expected kind, or nothing when the header is refused, is of another
// kind, or the frame is not exactly as long as the header says. Everything after this trusts the
// declared length, so this is the one length check between a frame's bytes and its decoding.
std::optional<std::span<const uint8_t>> FramePayload(std::span<const uint8_t> frame,
                                                     StreamFrameKind          expected) {
	const auto parsed = ParseStreamFrameHeader(frame);
	if (parsed.error != StreamHeaderError::None || parsed.header.kind != expected) {
		return std::nullopt;
	}
	if (frame.size() - STREAM_FRAME_HEADER_BYTES != parsed.header.payload_bytes) {
		return std::nullopt;
	}
	return std::span<const uint8_t>(frame.data() + STREAM_FRAME_HEADER_BYTES,
	                                parsed.header.payload_bytes);
}

// Whether the fixed part and the byte strings it announces fill the payload exactly. The sum is
// taken in 64 bits so lengths near the 32-bit limit cannot wrap.
bool FillsPayload(size_t payload_bytes, size_t fixed_bytes, uint64_t first_bytes,
                  uint64_t second_bytes) {
	return static_cast<uint64_t>(fixed_bytes) + first_bytes + second_bytes == payload_bytes;
}

} // namespace

StreamHeaderParse ParseStreamFrameHeader(std::span<const uint8_t> bytes) {
	StreamHeaderParse parse;
	if (bytes.size() < STREAM_FRAME_HEADER_BYTES) {
		return parse;
	}
	FieldReader read {bytes.data()};
	uint8_t     kind     = 0;
	uint8_t     reserved = 0;
	VisitAll(read, parse.header.magic, parse.header.version, kind, reserved,
	         parse.header.payload_bytes);
	if (parse.header.magic != STREAM_FRAME_MAGIC) {
		parse.error = StreamHeaderError::BadMagic;
	} else if (parse.header.version != STREAM_PROTOCOL_VERSION) {
		parse.error = StreamHeaderError::BadVersion;
	} else if (!KnownFrameKind(kind) || reserved != 0) {
		parse.error = StreamHeaderError::UnknownKind;
	} else if (parse.header.payload_bytes > MAX_STREAM_PAYLOAD_BYTES) {
		parse.error = StreamHeaderError::TooLarge;
	} else {
		parse.error       = StreamHeaderError::None;
		parse.header.kind = static_cast<StreamFrameKind>(kind);
	}
	return parse;
}

bool EncodeStreamRequest(const StreamRequest& request, std::vector<uint8_t>& frame) {
	frame.clear();
	const uint64_t payload_bytes = static_cast<uint64_t>(RequestFixedBytes()) + request.code.size() +
	                               request.specialization.size();
	if (payload_bytes > MAX_STREAM_PAYLOAD_BYTES) {
		return false;
	}
	BeginFrame(frame, StreamFrameKind::Request, static_cast<uint32_t>(payload_bytes));
	FieldWriter write {frame};
	BlobLengths lengths {static_cast<uint32_t>(request.code.size()),
	                     static_cast<uint32_t>(request.specialization.size())};
	VisitRequestFixed(write, request, lengths);
	AppendBytes(frame, request.code.data(), request.code.size());
	AppendBytes(frame, request.specialization.data(), request.specialization.size());
	return true;
}

bool StreamResponseEncodes(const StreamResponse& response) {
	return ResponsePayloadBytes(response) <= MAX_STREAM_PAYLOAD_BYTES &&
	       PayloadFitsStatus(response.status, response.payload.size());
}

bool EncodeStreamResponse(const StreamResponse& response, std::vector<uint8_t>& frame) {
	frame.clear();
	if (!StreamResponseEncodes(response)) {
		return false;
	}
	const uint64_t state_bytes   = response.static_state.size() * sizeof(uint32_t);
	const uint64_t payload_bytes = ResponsePayloadBytes(response);
	BeginFrame(frame, StreamFrameKind::Response, static_cast<uint32_t>(payload_bytes));
	FieldWriter write {frame};
	BlobLengths lengths {static_cast<uint32_t>(response.payload.size()),
	                     static_cast<uint32_t>(response.static_state.size())};
	VisitResponseFixed(write, response, lengths);
	AppendBytes(frame, response.payload.data(), response.payload.size());
	AppendBytes(frame, response.static_state.data(), state_bytes);
	return true;
}

std::optional<StreamRequest> DecodeStreamRequest(std::span<const uint8_t> frame) {
	const auto   payload     = FramePayload(frame, StreamFrameKind::Request);
	const size_t fixed_bytes = RequestFixedBytes();
	if (!payload || payload->size() < fixed_bytes) {
		return std::nullopt;
	}
	StreamRequest request;
	BlobLengths   lengths {};
	FieldReader   read {payload->data()};
	VisitRequestFixed(read, request, lengths);
	// The kind byte is checked after the read, since a read into an enum takes any value.
	if (!read.Ok() || !KnownRequestKind(request.kind) ||
	    !FillsPayload(payload->size(), fixed_bytes, lengths[0], lengths[1])) {
		return std::nullopt;
	}
	const uint8_t* code = read.At();
	request.code.assign(code, code + lengths[0]);
	request.specialization.assign(code + lengths[0], code + lengths[0] + lengths[1]);
	return request;
}

std::optional<StreamResponse> DecodeStreamResponse(std::span<const uint8_t> frame) {
	const auto   payload     = FramePayload(frame, StreamFrameKind::Response);
	const size_t fixed_bytes = ResponseFixedBytes();
	if (!payload || payload->size() < fixed_bytes) {
		return std::nullopt;
	}
	StreamResponse response;
	BlobLengths    lengths {};
	FieldReader    read {payload->data()};
	VisitResponseFixed(read, response, lengths);
	const uint64_t state_bytes = static_cast<uint64_t>(lengths[1]) * sizeof(uint32_t);
	if (!read.Ok() || !KnownRequestKind(response.kind) || !KnownStatus(response.status) ||
	    !PayloadFitsStatus(response.status, lengths[0]) ||
	    !FillsPayload(payload->size(), fixed_bytes, lengths[0], state_bytes)) {
		return std::nullopt;
	}
	const uint8_t* bytes = read.At();
	response.payload.assign(bytes, bytes + lengths[0]);
	// The words are copied as bytes, since the payload gives them no alignment.
	response.static_state.resize(lengths[1]);
	if (state_bytes > 0) {
		std::memcpy(response.static_state.data(), bytes + lengths[0], state_bytes);
	}
	return response;
}

} // namespace Libs::Graphics::StoreGen
