#include "graphics/shader/storegen/streamChannel.h"

#include "common/childProcess.h"
#include "common/config.h"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <random>
#include <string>
#include <system_error>

#include <fmt/format.h>

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <sddl.h>
#else
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace Libs::Graphics::StoreGen {

namespace {

/// The most one system read or write moves, which keeps every count inside 32 bits.
constexpr size_t MAX_IO_BYTES = size_t {1} << 20u;

/// How long CreateStreamPipePair waits for its own client, which has already connected by then.
constexpr std::chrono::milliseconds PAIR_ACCEPT_TIMEOUT {1000};

/// What every listener's pipe name or socket directory starts with.
constexpr std::string_view NAME_PREFIX = "kyty-stream-";

} // namespace

// ---- Frame reader. ----

// Reads until bytes [from, to) of the buffer are filled. A source that ends first fails the fill.
// A short read is the normal case for a pipe, so it only means that more reads are needed.
bool StreamFrameReader::Fill(StreamByteSource& source, size_t from, size_t to) {
	while (from < to) {
		const size_t got = source.ReadSome(std::span<uint8_t>(m_buffer).subspan(from, to - from));
		if (got == 0) {
			return false;
		}
		from += got;
	}
	return true;
}

// Makes a status final, so a stream that ended or lost its step is never read again.
StreamReadResult StreamFrameReader::Finish(StreamReadStatus status, StreamHeaderError header_error) {
	m_final = StreamReadResult {.status = status, .frame = {}, .header_error = header_error};
	return *m_final;
}

StreamReadResult StreamFrameReader::ReadFrame(StreamByteSource& source) {
	if (m_final) {
		return *m_final;
	}
	if (m_buffer.size() < STREAM_FRAME_HEADER_BYTES) {
		m_buffer.resize(STREAM_FRAME_HEADER_BYTES);
	}
	// The first byte tells a close between frames from one inside a frame.
	const size_t first = source.ReadSome(std::span<uint8_t>(m_buffer).first(STREAM_FRAME_HEADER_BYTES));
	if (first == 0) {
		return Finish(StreamReadStatus::Closed, StreamHeaderError::None);
	}
	if (!Fill(source, first, STREAM_FRAME_HEADER_BYTES)) {
		return Finish(StreamReadStatus::Truncated, StreamHeaderError::None);
	}
	const auto parsed = ParseStreamFrameHeader(std::span<const uint8_t>(m_buffer));
	if (parsed.error != StreamHeaderError::None) {
		return Finish(StreamReadStatus::BadHeader, parsed.error);
	}
	const size_t frame_bytes = STREAM_FRAME_HEADER_BYTES + parsed.header.payload_bytes;
	// Grows only, so a run of frames no larger than the largest so far allocates nothing.
	if (m_buffer.size() < frame_bytes) {
		m_buffer.resize(frame_bytes);
	}
	if (!Fill(source, STREAM_FRAME_HEADER_BYTES, frame_bytes)) {
		return Finish(StreamReadStatus::Truncated, StreamHeaderError::None);
	}
	return {.status       = StreamReadStatus::Frame,
	        .frame        = std::span<const uint8_t>(m_buffer).first(frame_bytes),
	        .header_error = StreamHeaderError::None};
}

// ---- Channel. ----

StreamChannel::StreamChannel(StreamPipe pipe): m_pipe(std::move(pipe)) {}

bool StreamChannel::Send(const StreamRequest& request) {
	return EncodeStreamRequest(request, m_send_buffer) && m_pipe.WriteAll(m_send_buffer);
}

bool StreamChannel::Send(const StreamResponse& response) {
	return EncodeStreamResponse(response, m_send_buffer) && m_pipe.WriteAll(m_send_buffer);
}

std::optional<std::pair<StreamPipe, StreamPipe>> CreateStreamPipePair() {
	auto listener = StreamPipeListener::Create("pair");
	if (!listener) {
		return std::nullopt;
	}
	auto client = ConnectStreamPipe(listener->Name());
	if (!client) {
		return std::nullopt;
	}
	auto server = listener->Accept(PAIR_ACCEPT_TIMEOUT, Common::CurrentProcessId());
	if (!server) {
		return std::nullopt;
	}
	return std::pair<StreamPipe, StreamPipe> {std::move(*server), std::move(*client)};
}

// ---- Pipe and listener members that do not depend on the platform. ----

StreamPipe::StreamPipe() = default;
StreamPipe::StreamPipe(std::unique_ptr<Impl> impl): m_impl(std::move(impl)) {}
StreamPipe::StreamPipe(StreamPipe&& other) noexcept            = default;
StreamPipe& StreamPipe::operator=(StreamPipe&& other) noexcept = default;
StreamPipe::~StreamPipe()                                      = default;

bool StreamPipe::IsOpen() const {
	return m_impl != nullptr;
}

void StreamPipe::Close() {
	m_impl.reset();
}

StreamPipeListener::StreamPipeListener() = default;
StreamPipeListener::StreamPipeListener(StreamPipeListener&& other) noexcept            = default;
StreamPipeListener& StreamPipeListener::operator=(StreamPipeListener&& other) noexcept = default;
StreamPipeListener::~StreamPipeListener()                                              = default;

#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS

// ---- Windows, a named pipe opened for overlapped I/O. ----

namespace {

/// How many names a listener tries before it gives up. Each has a fresh random part, so a taken
/// name is a collision or a squatter, and a few more tries are plenty for the first.
constexpr int CREATE_ATTEMPTS = 4;

/// The size of each direction's buffer in the pipe.
constexpr DWORD PIPE_BUFFER_BYTES = 64u * 1024u;
/// A listener serves exactly one client, so the pipe has one instance.
constexpr DWORD PIPE_MAX_INSTANCES = 1;

/// Owns a handle and closes it. Both a null handle and INVALID_HANDLE_VALUE mean none.
struct OwnedHandle {
	HANDLE value = nullptr;

	OwnedHandle() = default;
	explicit OwnedHandle(HANDLE handle) { Reset(handle); }
	~OwnedHandle() { Reset(nullptr); }
	OwnedHandle(const OwnedHandle&)            = delete;
	OwnedHandle& operator=(const OwnedHandle&) = delete;

	/// Closes the handle held, and holds the new one.
	void Reset(HANDLE handle) {
		if (value != nullptr) {
			CloseHandle(value);
		}
		value = handle == INVALID_HANDLE_VALUE ? nullptr : handle;
	}
};

/// Owns memory the system allocated with LocalAlloc and frees it.
struct OwnedLocal {
	void* value = nullptr;

	OwnedLocal() = default;
	~OwnedLocal() {
		if (value != nullptr) {
			LocalFree(value);
		}
	}
	OwnedLocal(const OwnedLocal&)            = delete;
	OwnedLocal& operator=(const OwnedLocal&) = delete;
};

// A manual reset event, which is what an OVERLAPPED wait and a lasting stop signal need.
HANDLE MakeEvent() {
	return CreateEventW(nullptr, TRUE, FALSE, nullptr);
}

} // namespace

struct StreamPipe::Impl {
	OwnedHandle pipe;
	OwnedHandle read_event {MakeEvent()};
	OwnedHandle write_event {MakeEvent()};
	/// Set by Interrupt and never reset, so every later read and write sees it.
	OwnedHandle stop_event {MakeEvent()};

	[[nodiscard]] bool Valid() const {
		return pipe.value != nullptr && read_event.value != nullptr &&
		       write_event.value != nullptr && stop_event.value != nullptr;
	}
};

struct StreamPipeListener::Impl {
	std::unique_ptr<StreamPipe::Impl> pipe = std::make_unique<StreamPipe::Impl>();
	OwnedHandle                       connect_event {MakeEvent()};
	/// The connect that waits for a client. It stays pending from one Accept to the next, so a
	/// client can connect while no Accept is running. The Impl lives on the heap, so its address
	/// stays put while the system holds it.
	OVERLAPPED  connect {};
	bool        pending = false;
	std::string name;

	Impl() = default;
	~Impl() {
		// The system writes to the OVERLAPPED until the connect ends, so it is ended first.
		if (pending && pipe) {
			DWORD moved = 0;
			CancelIoEx(pipe->pipe.value, &connect);
			GetOverlappedResult(pipe->pipe.value, &connect, &moved, TRUE);
		}
	}
	Impl(const Impl&)            = delete;
	Impl& operator=(const Impl&) = delete;
};

namespace {

// The name as UTF-16, as the wide system calls take it.
std::wstring WideName(const std::string& name) {
	const int length = MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.size()),
	                                       nullptr, 0);
	std::wstring wide(static_cast<size_t>(std::max(length, 0)), L'\0');
	if (length > 0) {
		MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.size()), wide.data(),
		                    length);
	}
	return wide;
}

// A pipe name for this process, the tag and 64 random bits, so another process cannot guess it and
// create it first.
std::string MakeStreamPipeName(std::string_view tag) {
	std::random_device random;
	const uint64_t     part = (static_cast<uint64_t>(random()) << 32u) | random();
	return fmt::format(R"(\\.\pipe\{}{}-{}-{:016x})", NAME_PREFIX, Common::CurrentProcessId(), tag,
	                   part);
}

// The current user's SID as text, or nothing when the token cannot be read.
std::optional<std::wstring> CurrentUserSid() {
	HANDLE token_handle = nullptr;
	if (OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token_handle) == FALSE) {
		return std::nullopt;
	}
	const OwnedHandle token(token_handle);
	DWORD             size = 0;
	(void)GetTokenInformation(token.value, TokenUser, nullptr, 0, &size);
	std::vector<uint8_t> buffer(size);
	if (size == 0 ||
	    GetTokenInformation(token.value, TokenUser, buffer.data(), size, &size) == FALSE) {
		return std::nullopt;
	}
	const auto* user = reinterpret_cast<const TOKEN_USER*>(buffer.data());
	OwnedLocal  text;
	if (ConvertSidToStringSidW(user->User.Sid, reinterpret_cast<LPWSTR*>(&text.value)) == FALSE) {
		return std::nullopt;
	}
	return std::wstring(static_cast<const wchar_t*>(text.value));
}

// A security descriptor whose protected DACL grants full access to the current user and to no one
// else, so no other account on the machine can open the pipe.
bool MakeOwnerOnlyDescriptor(OwnedLocal& descriptor) {
	const auto sid = CurrentUserSid();
	if (!sid) {
		return false;
	}
	const std::wstring sddl = L"D:P(A;;GA;;;" + *sid + L")";
	return ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl.c_str(), SDDL_REVISION_1,
	                                                            &descriptor.value, nullptr) != FALSE;
}

// Creates the server end under one name. The first instance flag fails when the name exists, so
// the pipe is never one another process made first.
std::unique_ptr<StreamPipeListener::Impl> CreatePipeAt(const std::string& name,
                                                       OwnedLocal&        descriptor) {
	auto                impl = std::make_unique<StreamPipeListener::Impl>();
	SECURITY_ATTRIBUTES attributes {.nLength              = sizeof(SECURITY_ATTRIBUTES),
	                                .lpSecurityDescriptor = descriptor.value,
	                                .bInheritHandle       = FALSE};
	impl->pipe->pipe.Reset(CreateNamedPipeW(
	    WideName(name).c_str(),
	    PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED | FILE_FLAG_FIRST_PIPE_INSTANCE,
	    PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
	    PIPE_MAX_INSTANCES, PIPE_BUFFER_BYTES, PIPE_BUFFER_BYTES, 0, &attributes));
	if (!impl->pipe->Valid() || impl->connect_event.value == nullptr) {
		return nullptr;
	}
	impl->name = name;
	return impl;
}

// Starts an overlapped read or write with start, then waits for it or for the stop event. The
// stop event is checked before the I/O is issued too, so a stopped pipe stays stopped even for an
// I/O that would complete at once. A stop cancels the I/O, and the cancelled I/O is still waited
// for, since the system writes to the OVERLAPPED until it ends. An I/O that completed before the
// cancel keeps its bytes. Returns how many bytes moved, or nothing on a failure, a close of the
// other end, or a stop.
template <class Start>
std::optional<DWORD> RunIo(HANDLE pipe, HANDLE io_event, HANDLE stop_event, Start start) {
	if (WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
		return std::nullopt;
	}
	OVERLAPPED overlapped {};
	overlapped.hEvent = io_event;
	if (start(overlapped) == FALSE && GetLastError() != ERROR_IO_PENDING) {
		return std::nullopt;
	}
	const HANDLE events[] = {io_event, stop_event};
	// The I/O event comes first, so an I/O that completed together with a stop is taken as done.
	if (WaitForMultipleObjects(2, events, FALSE, INFINITE) != WAIT_OBJECT_0) {
		CancelIoEx(pipe, &overlapped);
	}
	DWORD moved = 0;
	if (GetOverlappedResult(pipe, &overlapped, &moved, TRUE) == FALSE) {
		return std::nullopt;
	}
	return moved;
}

// A timeout as WaitForSingleObject takes it. A negative one does not wait, and one too long for a
// DWORD waits as long as a DWORD allows short of INFINITE.
DWORD WaitMilliseconds(std::chrono::milliseconds timeout) {
	constexpr auto longest = static_cast<long long>(INFINITE - 1);
	return static_cast<DWORD>(std::clamp<long long>(timeout.count(), 0, longest));
}

/// How a wait for a client ended.
enum class ClientWait : uint8_t {
	Connected, ///< A client is connected.
	None,      ///< No client came in time, and the pipe is still listening.
	Left,      ///< A client connected and closed before it was accepted.
};

// Puts the pipe in the listening state, which a pipe needs after a disconnect before any client
// can connect to it. A connect that has to wait is left pending in the listener.
ClientWait StartListening(StreamPipeListener::Impl& impl) {
	impl.connect        = {};
	impl.connect.hEvent = impl.connect_event.value;
	ResetEvent(impl.connect_event.value);
	if (ConnectNamedPipe(impl.pipe->pipe.value, &impl.connect) != FALSE) {
		return ClientWait::Connected;
	}
	switch (GetLastError()) {
		// The client connected before this call.
		case ERROR_PIPE_CONNECTED: return ClientWait::Connected;
		case ERROR_NO_DATA: return ClientWait::Left;
		case ERROR_IO_PENDING: impl.pending = true; return ClientWait::None;
		// Any other error leaves nothing pending, so Accept returns at once as if it timed out.
		// A caller that loops on short Accepts bounds this with its own deadline.
		default: return ClientWait::None;
	}
}

// Waits up to timeout for a client. A timeout leaves the connect pending, so a client that
// connects before the next Accept is still taken.
ClientWait WaitForClient(StreamPipeListener::Impl& impl, std::chrono::milliseconds timeout) {
	if (!impl.pending) {
		const auto started = StartListening(impl);
		if (!impl.pending) {
			return started;
		}
	}
	if (WaitForSingleObject(impl.connect_event.value, WaitMilliseconds(timeout)) != WAIT_OBJECT_0) {
		return ClientWait::None;
	}
	impl.pending = false;
	DWORD moved  = 0;
	// A connect that ended in an error had a client that went away.
	return GetOverlappedResult(impl.pipe->pipe.value, &impl.connect, &moved, FALSE) != FALSE
	           ? ClientWait::Connected
	           : ClientWait::Left;
}

// Whether the connected client is the expected process.
bool ClientIsProcess(HANDLE pipe, uint32_t expected_pid) {
	ULONG pid = 0;
	return GetNamedPipeClientProcessId(pipe, &pid) != FALSE && pid == expected_pid;
}

} // namespace

size_t StreamPipe::ReadSome(std::span<uint8_t> into) {
	if (!m_impl || into.empty()) {
		return 0;
	}
	const auto want = static_cast<DWORD>(std::min(into.size(), MAX_IO_BYTES));
	// A read that succeeds with no bytes is a zero length write by the other end, which this
	// protocol never makes, so it is read past rather than taken for a close.
	for (;;) {
		const auto got = RunIo(m_impl->pipe.value, m_impl->read_event.value,
		                       m_impl->stop_event.value, [&](OVERLAPPED& overlapped) {
			                       return ReadFile(m_impl->pipe.value, into.data(), want, nullptr,
			                                       &overlapped);
		                       });
		if (!got) {
			return 0;
		}
		if (*got > 0) {
			return *got;
		}
	}
}

bool StreamPipe::WriteAll(std::span<const uint8_t> bytes) {
	if (!m_impl) {
		return false;
	}
	while (!bytes.empty()) {
		const auto want = static_cast<DWORD>(std::min(bytes.size(), MAX_IO_BYTES));
		const auto put  = RunIo(m_impl->pipe.value, m_impl->write_event.value,
		                        m_impl->stop_event.value, [&](OVERLAPPED& overlapped) {
                                   return WriteFile(m_impl->pipe.value, bytes.data(), want,
                                                    nullptr, &overlapped);
                               });
		if (!put || *put == 0) {
			return false;
		}
		bytes = bytes.subspan(*put);
	}
	return true;
}

void StreamPipe::Interrupt() {
	if (m_impl) {
		SetEvent(m_impl->stop_event.value);
	}
}

std::optional<StreamPipeListener> StreamPipeListener::Create(std::string_view tag) {
	OwnedLocal descriptor;
	if (!MakeOwnerOnlyDescriptor(descriptor)) {
		return std::nullopt;
	}
	for (int attempt = 0; attempt < CREATE_ATTEMPTS; attempt++) {
		if (auto impl = CreatePipeAt(MakeStreamPipeName(tag), descriptor)) {
			StreamPipeListener listener;
			listener.m_impl = std::move(impl);
			return listener;
		}
	}
	return std::nullopt;
}

const std::string& StreamPipeListener::Name() const {
	static const std::string none;
	return m_impl ? m_impl->name : none;
}

std::optional<StreamPipe> StreamPipeListener::Accept(std::chrono::milliseconds timeout,
                                                     uint32_t                  expected_peer_pid) {
	if (!m_impl) {
		return std::nullopt;
	}
	const HANDLE pipe = m_impl->pipe->pipe.value;
	const auto   wait = WaitForClient(*m_impl, timeout);
	if (wait == ClientWait::None) {
		return std::nullopt;
	}
	// A client that left, or one of another process, is dropped, and the pipe listens again at
	// once, so the next client can connect before the next Accept.
	if (wait == ClientWait::Left || !ClientIsProcess(pipe, expected_peer_pid)) {
		DisconnectNamedPipe(pipe);
		(void)StartListening(*m_impl);
		return std::nullopt;
	}
	auto accepted = StreamPipe(std::move(m_impl->pipe));
	m_impl.reset();
	return accepted;
}

std::optional<StreamPipe> ConnectStreamPipe(const std::string& name) {
	auto impl = std::make_unique<StreamPipe::Impl>();
	// Identification only, so the server can never act as this process.
	impl->pipe.Reset(CreateFileW(WideName(name).c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
	                             OPEN_EXISTING,
	                             FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT |
	                                 SECURITY_IDENTIFICATION,
	                             nullptr));
	if (!impl->Valid()) {
		return std::nullopt;
	}
	return StreamPipe(std::move(impl));
}

#else

// ---- POSIX, a Unix domain socket. Not compiled on the Windows build machine. ----

namespace {

#if defined(MSG_NOSIGNAL)
/// A write to a closed socket returns an error instead of raising SIGPIPE.
constexpr int SEND_FLAGS = MSG_NOSIGNAL;
#else
constexpr int SEND_FLAGS = 0;
#endif

/// A listener serves exactly one client.
constexpr int LISTEN_BACKLOG = 1;

/// The socket's file name inside the listener's private directory. It is short, so the whole
/// path fits in sun_path.
constexpr std::string_view SOCKET_FILE_NAME = "s";

/// Owns a descriptor and closes it. A negative descriptor means none.
struct OwnedFd {
	int value = -1;

	OwnedFd() = default;
	~OwnedFd() { Reset(-1); }
	OwnedFd(const OwnedFd&)            = delete;
	OwnedFd& operator=(const OwnedFd&) = delete;

	/// Closes the descriptor held, and holds the new one.
	void Reset(int fd) {
		if (value >= 0) {
			::close(value);
		}
		value = fd;
	}
};

// On macOS, which has no MSG_NOSIGNAL, turns SIGPIPE off for the socket.
void DisableSigpipe([[maybe_unused]] int fd) {
#if defined(SO_NOSIGPIPE)
	int on = 1;
	(void)::setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof(on));
#endif
}

// A new stream socket kept out of children. SOCK_CLOEXEC sets that atomically where it exists, and
// macOS, which lacks it, sets it right after.
int MakeSocket() {
#if defined(SOCK_CLOEXEC)
	const int fd = ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
#else
	const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd >= 0) {
		(void)::fcntl(fd, F_SETFD, FD_CLOEXEC);
	}
#endif
	if (fd >= 0) {
		DisableSigpipe(fd);
	}
	return fd;
}

// Accepts a connection kept out of children, the way MakeSocket makes one.
int AcceptSocket(int listener) {
#if defined(__linux__)
	const int fd = ::accept4(listener, nullptr, nullptr, SOCK_CLOEXEC);
#else
	const int fd = ::accept(listener, nullptr, nullptr);
	if (fd >= 0) {
		(void)::fcntl(fd, F_SETFD, FD_CLOEXEC);
	}
#endif
	if (fd >= 0) {
		DisableSigpipe(fd);
	}
	return fd;
}

// The socket address of a path, or nothing when the path does not fit, which is logged since it
// means the temporary directory's path is unusually long.
std::optional<sockaddr_un> SocketAddress(const std::string& path) {
	sockaddr_un address {};
	address.sun_family = AF_UNIX;
	if (path.size() >= sizeof(address.sun_path)) {
		std::fprintf(stderr, "Stream pipe: the socket path %s is too long for a Unix socket\n",
		             path.c_str());
		return std::nullopt;
	}
	std::copy(path.begin(), path.end(), address.sun_path);
	return address;
}

// Whether the connected peer is the expected process. Linux and macOS give the peer's process id.
// Elsewhere only the peer's user can be checked.
bool PeerIsProcess(int fd, [[maybe_unused]] uint32_t expected_pid) {
#if defined(SO_PEERCRED)
	ucred     credentials {};
	socklen_t length = sizeof(credentials);
	return ::getsockopt(fd, SOL_SOCKET, SO_PEERCRED, &credentials, &length) == 0 &&
	       credentials.pid == static_cast<pid_t>(expected_pid);
#elif defined(LOCAL_PEERPID)
	pid_t     pid    = 0;
	socklen_t length = sizeof(pid);
	return ::getsockopt(fd, SOL_LOCAL, LOCAL_PEERPID, &pid, &length) == 0 &&
	       pid == static_cast<pid_t>(expected_pid);
#else
	uid_t user  = 0;
	gid_t group = 0;
	return ::getpeereid(fd, &user, &group) == 0 && user == ::geteuid();
#endif
}

// A poll timeout from a duration, from 0 to INT_MAX milliseconds.
int PollMilliseconds(std::chrono::milliseconds timeout) {
	return static_cast<int>(std::clamp<long long>(timeout.count(), 0, INT_MAX));
}

} // namespace

struct StreamPipe::Impl {
	OwnedFd fd;
	/// Set by Interrupt, and checked before each read and write so the pipe stays stopped.
	std::atomic<bool> interrupted {false};
};

struct StreamPipeListener::Impl {
	OwnedFd     fd;
	std::string directory;
	std::string path;

	Impl() = default;
	~Impl() {
		if (!path.empty()) {
			(void)::unlink(path.c_str());
		}
		if (!directory.empty()) {
			(void)::rmdir(directory.c_str());
		}
	}
	Impl(const Impl&)            = delete;
	Impl& operator=(const Impl&) = delete;
};

size_t StreamPipe::ReadSome(std::span<uint8_t> into) {
	if (!m_impl || into.empty()) {
		return 0;
	}
	while (!m_impl->interrupted.load(std::memory_order_acquire)) {
		const ssize_t got =
		    ::recv(m_impl->fd.value, into.data(), std::min(into.size(), MAX_IO_BYTES), 0);
		if (got > 0) {
			return static_cast<size_t>(got);
		}
		if (got == 0 || errno != EINTR) {
			return 0;
		}
	}
	return 0;
}

bool StreamPipe::WriteAll(std::span<const uint8_t> bytes) {
	if (!m_impl) {
		return false;
	}
	while (!bytes.empty()) {
		if (m_impl->interrupted.load(std::memory_order_acquire)) {
			return false;
		}
		const ssize_t put = ::send(m_impl->fd.value, bytes.data(),
		                           std::min(bytes.size(), MAX_IO_BYTES), SEND_FLAGS);
		if (put < 0 && errno == EINTR) {
			continue;
		}
		if (put <= 0) {
			return false;
		}
		bytes = bytes.subspan(static_cast<size_t>(put));
	}
	return true;
}

void StreamPipe::Interrupt() {
	if (m_impl) {
		m_impl->interrupted.store(true, std::memory_order_release);
		// Wakes a recv or send blocked in another thread, which then returns 0 or fails.
		(void)::shutdown(m_impl->fd.value, SHUT_RDWR);
	}
}

std::optional<StreamPipeListener> StreamPipeListener::Create(std::string_view tag) {
	std::error_code error;
	auto            temp = std::filesystem::temp_directory_path(error);
	if (error) {
		temp = "/tmp";
	}
	// mkdtemp makes a new directory only this user can enter, so the socket cannot be reached by
	// another user and never meets a stale socket of an earlier process with the same id.
	auto pattern = (temp / fmt::format("{}{}-{}-XXXXXX", NAME_PREFIX, Common::CurrentProcessId(),
	                                   tag))
	                   .string();
	if (::mkdtemp(pattern.data()) == nullptr) {
		return std::nullopt;
	}
	auto impl       = std::make_unique<Impl>();
	impl->directory = pattern;
	const auto path = pattern + "/" + std::string(SOCKET_FILE_NAME);
	const auto address = SocketAddress(path);
	impl->fd.Reset(MakeSocket());
	if (!address || impl->fd.value < 0 ||
	    ::bind(impl->fd.value, reinterpret_cast<const sockaddr*>(&*address), sizeof(*address)) != 0) {
		return std::nullopt;
	}
	impl->path = path;
	if (::listen(impl->fd.value, LISTEN_BACKLOG) != 0) {
		return std::nullopt;
	}
	StreamPipeListener listener;
	listener.m_impl = std::move(impl);
	return listener;
}

const std::string& StreamPipeListener::Name() const {
	static const std::string none;
	return m_impl ? m_impl->path : none;
}

std::optional<StreamPipe> StreamPipeListener::Accept(std::chrono::milliseconds timeout,
                                                     uint32_t                  expected_peer_pid) {
	if (!m_impl) {
		return std::nullopt;
	}
	// A signal that interrupts the poll ends it like a timeout, and the caller may call again.
	pollfd waiting {.fd = m_impl->fd.value, .events = POLLIN, .revents = 0};
	if (::poll(&waiting, 1, PollMilliseconds(timeout)) <= 0) {
		return std::nullopt;
	}
	auto impl = std::make_unique<StreamPipe::Impl>();
	impl->fd.Reset(AcceptSocket(m_impl->fd.value));
	// A peer of another process is closed here, and the listener goes on listening.
	if (impl->fd.value < 0 || !PeerIsProcess(impl->fd.value, expected_peer_pid)) {
		return std::nullopt;
	}
	m_impl.reset();
	return StreamPipe(std::move(impl));
}

std::optional<StreamPipe> ConnectStreamPipe(const std::string& name) {
	const auto address = SocketAddress(name);
	if (!address) {
		return std::nullopt;
	}
	auto impl = std::make_unique<StreamPipe::Impl>();
	impl->fd.Reset(MakeSocket());
	if (impl->fd.value < 0 ||
	    ::connect(impl->fd.value, reinterpret_cast<const sockaddr*>(&*address), sizeof(*address)) !=
	        0) {
		return std::nullopt;
	}
	return StreamPipe(std::move(impl));
}

#endif

} // namespace Libs::Graphics::StoreGen
