#ifndef GRAPHICS_SHADER_STOREGEN_STREAM_PROTOCOL_H
#define GRAPHICS_SHADER_STOREGEN_STREAM_PROTOCOL_H

#include "graphics/shader/storegen/prescanInputs.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

// The frames a Stream worker and its supervisor exchange through a pipe. Each frame is a fixed
// header with a magic, a version, a frame kind and the payload's length, followed by that many
// payload bytes. Both ends run on the same machine from the same build, so values travel in host
// byte order. Nothing here needs a graphics device or Config.
namespace Libs::Graphics::StoreGen {

/// Starts every stream frame, and holds "STRM".
constexpr uint32_t STREAM_FRAME_MAGIC = 0x4D525453;
/// The layout of the frame header and of both payloads. A reader refuses any other value.
constexpr uint16_t STREAM_PROTOCOL_VERSION = 1;
/// The size of a frame header, which is the magic, the version, the kind, a zero byte and the
/// payload length.
constexpr size_t STREAM_FRAME_HEADER_BYTES =
    sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint32_t);
/// The largest payload a writer sends and a reader waits for. The largest module record seen is
/// about 5.7 MB before compression, so 32 MiB leaves room while a corrupt length still cannot make
/// a reader allocate gigabytes.
constexpr uint32_t MAX_STREAM_PAYLOAD_BYTES = 32u * 1024u * 1024u;

/// What a frame carries, which is its header's kind byte.
enum class StreamFrameKind : uint8_t {
	Request  = 1, ///< A StreamRequest, from the supervisor to the worker.
	Response = 2, ///< A StreamResponse, from the worker to the supervisor.
};

/// What a request asks the worker for.
enum class StreamRequestKind : uint8_t {
	Plan    = 1, ///< The program's resource plan bytes.
	Compile = 2, ///< The program compiled with the given specialization, as EncodeModuleRecord bytes.
};

/// How the worker answered a request.
enum class StreamStatus : uint8_t {
	Ok       = 1, ///< The payload holds the answer.
	HasCalls = 2, ///< The program makes calls, which the worker does not translate.
	Dropped  = 3, ///< The worker gave the request up, for example when it was stopped.
	Failed   = 4, ///< Translation or compilation failed.
};

/// The fixed header in front of every frame's payload.
struct StreamFrameHeader {
	uint32_t        magic         = 0;
	uint16_t        version       = 0;
	StreamFrameKind kind          = StreamFrameKind::Request;
	uint32_t        payload_bytes = 0;
};

/// Why a frame header was refused.
enum class StreamHeaderError : uint8_t {
	None,        ///< The header is good.
	Short,       ///< Fewer bytes than a header.
	BadMagic,    ///< The magic is not STREAM_FRAME_MAGIC.
	BadVersion,  ///< The version is not STREAM_PROTOCOL_VERSION.
	UnknownKind, ///< The kind byte is no StreamFrameKind, or the zero byte is not zero.
	TooLarge,    ///< The payload length is over MAX_STREAM_PAYLOAD_BYTES.
};

/// A parsed frame header, and why it was refused when it was.
struct StreamHeaderParse {
	StreamHeaderError error = StreamHeaderError::Short;
	StreamFrameHeader header;
};

/// Reads the frame header at the start of bytes. The bytes after the header are not looked at.
[[nodiscard]] StreamHeaderParse ParseStreamFrameHeader(std::span<const uint8_t> bytes);

/// One job for a Stream worker. The worker reads no device and no Config for it, since the host
/// values come from the WorkerRequest that started it.
struct StreamRequest {
	/// The supervisor's number for the request, echoed in the response.
	uint64_t          seq  = 0;
	StreamRequestKind kind = StreamRequestKind::Plan;
	/// The hash of the program's code, as the shader map computes it.
	uint64_t code_hash = 0;
	/// The program's code.
	std::vector<uint8_t> code;
	/// The compute registers and dispatch flags the submit scan saw, for MakePrescanInputInfo.
	PrescanDispatchInputs dispatch {};
	/// How many user data registers the program reads, rsrc2.user_sgpr.
	uint32_t user_data_count = 0;
	/// The scratch size the shader map holds for the program, which no register carries.
	uint32_t scratch_size_dwords = 0;
	/// The encoded resource specialization of a Compile request, empty for a Plan request.
	std::vector<uint8_t> specialization;
};

/// A Stream worker's answer to one request.
struct StreamResponse {
	/// The seq of the request this answers.
	uint64_t          seq    = 0;
	StreamRequestKind kind   = StreamRequestKind::Plan;
	StreamStatus      status = StreamStatus::Ok;
	/// The plan bytes of a Plan answer or the EncodeModuleRecord bytes of a Compile answer. It must
	/// be empty unless the status is Ok, and a frame that breaks this neither encodes nor decodes.
	std::vector<uint8_t> payload;
	/// The static key words the worker built the answer for, so the supervisor can check them.
	std::vector<uint32_t> static_state;
};

/// Writes a request as one frame, replacing what frame held. The vector's storage is reused, so a
/// caller that keeps it allocates only when a frame is larger than any before.
/// @return false when the payload would be over MAX_STREAM_PAYLOAD_BYTES, and frame is then empty.
[[nodiscard]] bool EncodeStreamRequest(const StreamRequest& request, std::vector<uint8_t>& frame);

/// Writes a response as one frame, replacing what frame held, the way EncodeStreamRequest does.
/// @return false when the payload would be over MAX_STREAM_PAYLOAD_BYTES or the response has a
///         payload with a status other than Ok, and frame is then empty.
[[nodiscard]] bool EncodeStreamResponse(const StreamResponse& response, std::vector<uint8_t>& frame);

/// Reads a request frame.
/// @param frame exactly one frame, the header and the payload it declares.
/// @return nothing when the header is refused or is not a Request, when frame is not exactly as
///         long as its header says, when the payload's own lengths do not fill it exactly, or when
///         a kind or flag byte holds a value that does not exist.
[[nodiscard]] std::optional<StreamRequest> DecodeStreamRequest(std::span<const uint8_t> frame);

/// Reads a response frame, with the same refusals as DecodeStreamRequest, an unknown status, and a
/// payload with a status other than Ok.
/// @param frame exactly one frame, the header and the payload it declares.
[[nodiscard]] std::optional<StreamResponse> DecodeStreamResponse(std::span<const uint8_t> frame);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_STREAM_PROTOCOL_H
