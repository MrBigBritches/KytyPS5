#ifndef GRAPHICS_SHADER_STOREGEN_STREAM_CHANNEL_H
#define GRAPHICS_SHADER_STOREGEN_STREAM_CHANNEL_H

#include "graphics/shader/storegen/streamProtocol.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// The pipe a Stream worker and its supervisor talk through, and the reader that takes whole frames
// out of it. A read blocks until the bytes are there, so nothing polls, and a frame that arrives in
// pieces is waited for rather than read again. The worker's progress log stays a file, since a
// crash must leave it behind for blame.
//
// Windows uses a named pipe opened for overlapped I/O, and each call waits for its own I/O, so the
// calls still block. POSIX uses a Unix domain socket. ChildProcess gives a child no inherited
// handles, so the child connects by name.
namespace Libs::Graphics::StoreGen {

/// Where a frame reader takes its bytes from.
class StreamByteSource {
public:
	virtual ~StreamByteSource() = default;

	/// Blocks until at least one byte is there and copies up to into.size() bytes.
	/// @return how many bytes were copied, or 0 once the other end has closed or the read failed.
	virtual size_t ReadSome(std::span<uint8_t> into) = 0;

protected:
	StreamByteSource()                                   = default;
	StreamByteSource(const StreamByteSource&)            = default;
	StreamByteSource& operator=(const StreamByteSource&) = default;
	StreamByteSource(StreamByteSource&&)                 = default;
	StreamByteSource& operator=(StreamByteSource&&)      = default;
};

/// What a frame read gave.
enum class StreamReadStatus : uint8_t {
	Frame,     ///< A whole frame with a good header. Its payload is not decoded yet.
	Closed,    ///< The other end closed between two frames.
	Truncated, ///< The other end closed inside a frame.
	BadHeader, ///< A header was refused, so the stream is out of step for good.
};

/// The result of one frame read.
struct StreamReadResult {
	StreamReadStatus status = StreamReadStatus::Closed;
	/// The whole frame, header and payload, when the status is Frame. It points into the reader's
	/// buffer and stays valid until the next read.
	std::span<const uint8_t> frame;
	/// Why the header was refused, when the status is BadHeader.
	StreamHeaderError header_error = StreamHeaderError::None;
};

/// Takes whole frames out of a byte source. It reads exactly the header and then exactly the
/// payload, so it never holds bytes of the next frame, and it keeps one buffer that grows to the
/// largest frame seen, so a read allocates nothing once it has seen a frame that large.
class StreamFrameReader {
public:
	/// Reads the next frame, waiting for as many reads of the source as it takes. Any status but
	/// Frame is final, and later calls return it again without reading.
	[[nodiscard]] StreamReadResult ReadFrame(StreamByteSource& source);

	/// The size of the buffer, which is the largest frame read so far.
	[[nodiscard]] size_t BufferBytes() const { return m_buffer.size(); }

private:
	bool             Fill(StreamByteSource& source, size_t from, size_t to);
	StreamReadResult Finish(StreamReadStatus status, StreamHeaderError header_error);

	std::vector<uint8_t>            m_buffer;
	std::optional<StreamReadResult> m_final;
};

/// One end of a connected stream pipe. Moving is allowed and copying is not. One thread may read
/// while another writes, but two threads must not read, or write, at the same time.
///
/// To stop a pipe that other threads are reading or writing, call Interrupt, then join those
/// threads, then Close. Interrupt is the one call that is safe while another thread is inside a read
/// or a write. Closing or moving the pipe at such a time is not allowed.
class StreamPipe final: public StreamByteSource {
public:
	StreamPipe();
	~StreamPipe() override;
	StreamPipe(StreamPipe&& other) noexcept;
	StreamPipe& operator=(StreamPipe&& other) noexcept;
	StreamPipe(const StreamPipe&)            = delete;
	StreamPipe& operator=(const StreamPipe&) = delete;

	/// Blocks until at least one byte arrives. Returns 0 when the other end has closed, the read
	/// failed, this end was interrupted, or this end is not open.
	size_t ReadSome(std::span<uint8_t> into) override;

	/// Writes every byte, blocking while the pipe is full.
	/// @return false when the other end has closed, a write failed, this end was interrupted, or
	///         this end is not open. Some of the bytes may have been written by then.
	[[nodiscard]] bool WriteAll(std::span<const uint8_t> bytes);

	/// Wakes any read or write blocked on this end, and makes every later one fail at once. It
	/// cannot be undone, and it may be called from any thread, any number of times. A read or write
	/// that completed just as it was called may still report its bytes. On POSIX it also shuts the
	/// socket down, so the other end reads a close, while on Windows the other end sees nothing
	/// until Close.
	void Interrupt();

	/// Whether this end holds an open connection.
	[[nodiscard]] bool IsOpen() const;

	/// Closes this end, which makes the other end's reads return 0.
	void Close();

	/// The system's state of one end. Defined in the source file, per platform.
	struct Impl;

private:
	friend class StreamPipeListener;
	friend std::optional<StreamPipe> ConnectStreamPipe(const std::string& name);

	explicit StreamPipe(std::unique_ptr<Impl> impl);

	std::unique_ptr<Impl> m_impl;
};

/// The server end of a named stream pipe, which accepts one client. Only the current user can open
/// it. On Windows the pipe's DACL grants access to the current user alone, and on POSIX the socket
/// sits in a private directory that only the current user can enter. One thread uses a listener,
/// and it is destroyed on that thread.
class StreamPipeListener {
public:
	StreamPipeListener();
	~StreamPipeListener();
	StreamPipeListener(StreamPipeListener&& other) noexcept;
	StreamPipeListener& operator=(StreamPipeListener&& other) noexcept;
	StreamPipeListener(const StreamPipeListener&)            = delete;
	StreamPipeListener& operator=(const StreamPipeListener&) = delete;

	/// Creates the server end under a new name made from this process, the tag and a random part.
	/// A name that is already taken is never joined, and a few new names are tried instead.
	/// @param tag a short ASCII word that tells the pipes of one process apart in a listing.
	/// @return nothing when no name could be created.
	[[nodiscard]] static std::optional<StreamPipeListener> Create(std::string_view tag);

	/// The name a client passes to ConnectStreamPipe. Empty once the listener is spent.
	[[nodiscard]] const std::string& Name() const;

	/// Waits up to timeout for a client and hands over the connected pipe. The listener is spent
	/// after a success. A timeout, a client that left before it was accepted, or a client of
	/// another process refuses the connection and leaves the listener able to wait again.
	/// @param timeout           how long to wait, clamped to what the system can wait for.
	/// @param expected_peer_pid the process id the client must have. On macOS without
	///                          LOCAL_PEERPID only the client's user is checked.
	[[nodiscard]] std::optional<StreamPipe> Accept(std::chrono::milliseconds timeout,
	                                               uint32_t                  expected_peer_pid);

	/// The system's state of a listener. Defined in the source file, per platform.
	struct Impl;

private:
	std::unique_ptr<Impl> m_impl;
};

/// Connects to a listener by name, as a child does with the name its parent gave it.
/// @return nothing when no listener has that name or it already has its client.
[[nodiscard]] std::optional<StreamPipe> ConnectStreamPipe(const std::string& name);

/// Two connected ends in this process, made through a listener and a connect with a name of their
/// own, so the same code is used as between two processes.
/// @return nothing when the pipe could not be made.
[[nodiscard]] std::optional<std::pair<StreamPipe, StreamPipe>> CreateStreamPipePair();

/// A stream pipe that sends and reads whole frames. Reading reuses the frame reader's buffer and
/// sending reuses one encode buffer, so neither allocates once the largest frame was seen. The
/// threading rules are StreamPipe's.
class StreamChannel {
public:
	explicit StreamChannel(StreamPipe pipe);

	/// Reads the next frame, as StreamFrameReader::ReadFrame does.
	[[nodiscard]] StreamReadResult ReadFrame() { return m_reader.ReadFrame(m_pipe); }

	/// Encodes and writes a request.
	/// @return false when it does not encode or the write failed.
	[[nodiscard]] bool Send(const StreamRequest& request);

	/// Encodes and writes a response.
	/// @return false when it does not encode or the write failed.
	[[nodiscard]] bool Send(const StreamResponse& response);

	/// Wakes a blocked ReadFrame or Send for good, as StreamPipe::Interrupt does.
	void Interrupt() { m_pipe.Interrupt(); }

	/// Closes the pipe, which the other end reads as Closed. Interrupt and join first when other
	/// threads use the channel.
	void Close() { m_pipe.Close(); }

private:
	StreamPipe           m_pipe;
	StreamFrameReader    m_reader;
	std::vector<uint8_t> m_send_buffer;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_STREAM_CHANNEL_H
