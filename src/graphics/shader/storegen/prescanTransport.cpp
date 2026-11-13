#include "graphics/shader/storegen/prescanTransport.h"

#include "common/config.h"
#include "common/logging/log.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"
#include "graphics/shader/storegen/streamWorker.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
#include <cerrno>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace Libs::Graphics::StoreGen {

InProcessPrescanTransport::InProcessPrescanTransport(const HostValues&        host,
                                                     const ComputeKeyDevice&  device,
                                                     PrescanCompletionHandler handler)
    : m_host(host), m_device(device), m_handler(std::move(handler)) {}

// The code is copied into the request, since the worker's answer reads it from there. The seq is
// given under the lock, and the answer runs outside it, so a handler that submits again cannot
// deadlock on it.
StreamSubmitResult InProcessPrescanTransport::Submit(StreamRequest                               request,
                                                     std::shared_ptr<const std::vector<uint8_t>> code,
                                                     uint64_t                                    tag) {
	if (code != nullptr) {
		request.code = *code;
	}
	{
		std::lock_guard lock(m_mutex);
		if (m_stopped) {
			return {StreamSubmitStatus::Stopped, 0};
		}
		(request.kind == StreamRequestKind::Plan ? m_plans : m_compiles)
		    .fetch_add(1, std::memory_order_relaxed);
		request.seq = m_next_seq++;
		if (m_hold && request.kind == StreamRequestKind::Compile) {
			const uint64_t seq = request.seq;
			m_held.push_back({std::move(request), tag});
			return {StreamSubmitStatus::Queued, seq};
		}
	}
	const uint64_t seq = request.seq;
	Answer(request, tag);
	return {StreamSubmitStatus::Queued, seq};
}

// Held requests complete as Dropped, as the supervisor's outstanding ones do when it stops.
void InProcessPrescanTransport::Stop() {
	std::deque<HeldRequest> held;
	{
		std::lock_guard lock(m_mutex);
		m_stopped = true;
		held.swap(m_held);
	}
	for (auto& entry: held) {
		StreamCompletion completion;
		completion.seq     = entry.request.seq;
		completion.kind    = entry.request.kind;
		completion.outcome = StreamOutcome::Dropped;
		completion.tag     = entry.tag;
		m_handler(std::move(completion));
	}
}

void InProcessPrescanTransport::SetHoldCompiles(bool hold) {
	std::lock_guard lock(m_mutex);
	m_hold = hold;
}

size_t InProcessPrescanTransport::Held() const {
	std::lock_guard lock(m_mutex);
	return m_held.size();
}

void InProcessPrescanTransport::ReleaseHeld() {
	std::deque<HeldRequest> held;
	{
		std::lock_guard lock(m_mutex);
		held.swap(m_held);
	}
	for (const auto& entry: held) {
		Answer(entry.request, entry.tag);
	}
}

void InProcessPrescanTransport::Answer(const StreamRequest& request, uint64_t tag) {
	StreamCompletion completion;
	completion.seq      = request.seq;
	completion.kind     = request.kind;
	completion.outcome  = StreamOutcome::Answered;
	completion.tag      = tag;
	completion.response = AnswerStreamRequest(request, m_host, m_device);
	m_handler(std::move(completion));
}

PrescanTransportFactory InProcessPrescanTransportFactory(InProcessPrescanTransport** made) {
	return [made](const HostValues& host, const ComputeKeyDevice& device,
	              PrescanCompletionHandler handler) -> std::unique_ptr<PrescanTransport> {
		auto transport = std::make_unique<InProcessPrescanTransport>(host, device, std::move(handler));
		if (made != nullptr) {
			*made = transport.get();
		}
		return transport;
	};
}

namespace {

/// The build stamp in the helper's header. The helper skips the stamp, and the pre-scan writes no
/// store, so any value would do.
constexpr uint64_t HELPER_HEADER_STAMP = 0;

// The supervisor's settings with what the child needs from this device. The child decodes the host
// values from the store header and takes its device from them and the host subgroup size, so a
// child given these builds the modules the preparer's own static key expects. The handler becomes
// the supervisor's callback as it is, so the callback does no more than the handler.
StreamSupervisorConfig WithHost(StreamSupervisorConfig config, const HostValues& host,
                                const ComputeKeyDevice& device, PrescanCompletionHandler handler) {
	if (device.max_compute_shared_memory_bytes != host.max_shared_bytes) {
		throw std::invalid_argument(
		    "the prescan helper takes its shared memory limit from the host values, which differ "
		    "from the device's");
	}
	config.request.store_header       = EncodeHostHeader(host, HELPER_HEADER_STAMP);
	config.request.host_subgroup_size = device.host_subgroup_size;
	config.on_complete                = std::move(handler);
	return config;
}

} // namespace

// The files are named before the config moves into the supervisor, which is why m_files is
// declared first.
SupervisorPrescanTransport::SupervisorPrescanTransport(StreamSupervisorConfig   config,
                                                       const HostValues&        host,
                                                       const ComputeKeyDevice&  device,
                                                       PrescanCompletionHandler handler)
    : m_files(StreamSupervisorFilesOf(config.files_base)),
      m_supervisor(WithHost(std::move(config), host, device, std::move(handler))) {}

SupervisorPrescanTransport::~SupervisorPrescanTransport() {
	Stop();
}

// The shared-code overload, so the caller keeps its own copy of the code for a later request. The
// supervisor only queues, so this never waits for the child.
StreamSubmitResult SupervisorPrescanTransport::Submit(StreamRequest                               request,
                                                      std::shared_ptr<const std::vector<uint8_t>> code,
                                                      uint64_t                                    tag) {
	return m_supervisor.Submit(std::move(request), std::move(code), tag);
}

// Kills the child and completes what is left as Dropped before it returns, then settles the files
// once. The supervisor's Stop may be called again, and then does nothing.
void SupervisorPrescanTransport::Stop() {
	m_supervisor.Stop();
	std::call_once(m_files_settled, [this] { SettleFiles(); });
}

// The child was ended by Stop, so nothing holds the files any more. Files kept after a failure, a
// blame or a lane that turned off explain what went wrong, and the next run's start-up cleanup
// removes them, since their name carries this process's id. It runs from the destructor too, so
// nothing may escape it. A copy or a log line that fails only leaves the files for that cleanup.
void SupervisorPrescanTransport::SettleFiles() noexcept {
	try {
		const auto stats = m_supervisor.Stats();
		if (stats.lane_off || stats.blamed > 0 || stats.failures > 0) {
			LOGF("Prescan helper: its files are kept for diagnosis at %s, and the next run's "
			     "start-up cleanup removes them\n",
			     Utf8FromPath(m_files.progress).c_str());
			return;
		}
		for (const auto& path: {m_files.request, m_files.progress, m_files.output}) {
			std::error_code ignored;
			std::filesystem::remove(path, ignored);
		}
	} catch (...) {
	}
}

namespace {

/// The stem of the helper's file names, which the process id follows.
constexpr std::string_view HELPER_FILE_STEM = "kyty_prescan_";
/// What follows the process id in each of the helper's file names.
constexpr std::array<std::string_view, 3> HELPER_FILE_SUFFIXES = {".request", ".progress",
                                                                  ".progress.out"};

/// The directory the helper's files go in, inside the user's temporary or runtime directory, so
/// the start-up cleanup only ever looks at the helper's own files.
constexpr std::string_view HELPER_DIRECTORY_NAME = "kyty_prescan";

#if KYTY_PLATFORM != KYTY_PLATFORM_WINDOWS
/// The mode of the directory made for the helper's files, which only its owner may enter.
constexpr mode_t PRIVATE_DIRECTORY_MODE = S_IRWXU;

// Whether a path is a directory this user owns and nobody else may enter. lstat looks at the path
// itself, so a link another user planted there is refused rather than followed.
bool IsPrivateDirectory(const std::filesystem::path& path) {
	struct stat info {};
	return ::lstat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode) &&
	       info.st_uid == ::getuid() && (info.st_mode & (S_IRWXG | S_IRWXO)) == 0;
}

// Makes the directory with mode 0700 when it is not there, and gives it only when it is private.
// A directory another user made first under this name fails the check, so EEXIST is fine here.
std::optional<std::filesystem::path> MakePrivateDirectory(const std::filesystem::path& path) {
	if (::mkdir(path.c_str(), PRIVATE_DIRECTORY_MODE) != 0 && errno != EEXIST) {
		return std::nullopt;
	}
	return IsPrivateDirectory(path) ? std::optional(path) : std::nullopt;
}
#endif

} // namespace

std::optional<std::filesystem::path> PrescanHelperDirectory() {
	std::error_code error;
	auto            temp = std::filesystem::temp_directory_path(error);
#if KYTY_PLATFORM == KYTY_PLATFORM_WINDOWS
	// The temporary directory is the user's own, so no other user can reach it.
	if (error) {
		return std::nullopt;
	}
	const auto directory = temp / HELPER_DIRECTORY_NAME;
	std::filesystem::create_directories(directory, error);
	return !error && std::filesystem::is_directory(directory, error) ? std::optional(directory)
	                                                                 : std::nullopt;
#else
	if (const char* runtime = std::getenv("XDG_RUNTIME_DIR");
	    runtime != nullptr && runtime[0] == '/' && IsPrivateDirectory(runtime)) {
		const auto inside = std::filesystem::path(runtime) / HELPER_DIRECTORY_NAME;
		if (auto made = MakePrivateDirectory(inside)) {
			return made;
		}
	}
	if (error) {
		temp = "/tmp";
	}
	// The per-user parent keeps other users out, and the subdirectory keeps the cleanup to the
	// helper's own files, as on the other paths.
	const auto parent = MakePrivateDirectory(temp / ("kyty-" + std::to_string(::getuid())));
	return parent ? MakePrivateDirectory(*parent / HELPER_DIRECTORY_NAME) : std::nullopt;
#endif
}

std::filesystem::path PrescanHelperFilesBase(const std::filesystem::path& directory, uint32_t pid) {
	return directory / (std::string(HELPER_FILE_STEM) + std::to_string(pid));
}

// The stem and the suffix are matched as plain text, and the id must be digits only, so a name that
// merely starts like one of ours is never taken.
std::optional<uint32_t> PrescanHelperFilePid(std::string_view name) {
	if (!name.starts_with(HELPER_FILE_STEM)) {
		return std::nullopt;
	}
	name.remove_prefix(HELPER_FILE_STEM.size());
	uint32_t   pid    = 0;
	const auto digits = std::from_chars(name.data(), name.data() + name.size(), pid);
	if (digits.ec != std::errc() || digits.ptr == name.data()) {
		return std::nullopt;
	}
	const std::string_view rest(digits.ptr,
	                            static_cast<size_t>(name.data() + name.size() - digits.ptr));
	if (std::ranges::find(HELPER_FILE_SUFFIXES, rest) == HELPER_FILE_SUFFIXES.end()) {
		return std::nullopt;
	}
	return pid;
}

StaleCleanup RemoveStalePrescanFiles(const std::filesystem::path& directory, uint32_t current_pid,
                                     const std::function<bool(uint32_t)>&         is_running,
                                     const std::function<void(std::string_view)>& log) {
	return RemoveDeadProcessFiles(directory, current_pid, PrescanHelperFilePid, is_running, log,
	                              "Prescan helper");
}

PrescanTransportFactory SupervisorPrescanTransportFactory(StreamSupervisorConfig       config,
                                                          SupervisorPrescanTransport** made) {
	return [config = std::move(config), made](const HostValues& host, const ComputeKeyDevice& device,
	                                          PrescanCompletionHandler handler)
	           -> std::unique_ptr<PrescanTransport> {
		auto transport =
		    std::make_unique<SupervisorPrescanTransport>(config, host, device, std::move(handler));
		if (made != nullptr) {
			*made = transport.get();
		}
		return transport;
	};
}

} // namespace Libs::Graphics::StoreGen
