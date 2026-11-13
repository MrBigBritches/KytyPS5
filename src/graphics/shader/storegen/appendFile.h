#ifndef GRAPHICS_SHADER_STOREGEN_APPEND_FILE_H
#define GRAPHICS_SHADER_STOREGEN_APPEND_FILE_H

#include "common/file.h"

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>

// The file writer of the shader store worker's output and progress log, shared by its Generate and
// Stream modes.
namespace Libs::Graphics::StoreGen {

/// A file that is appended to. Common::File writes straight to the operating system on every
/// platform, so a write is visible to another process as soon as it returns and a crash of this
/// process loses nothing written before it. A flush would only add FlushFileBuffers or fsync, which
/// protects against power loss, and this file can be generated again. The start line test reads
/// the log through another handle, so it would catch a worker that held its lines back. One thread
/// uses it at a time.
class AppendFile {
public:
	/// Opens a log, which is appended to when it exists.
	bool OpenAppend(const std::filesystem::path& path);

	/// Starts the output. With nothing kept it is a new file that begins with the header. Otherwise
	/// the first `kept` bytes stay, after a check that they begin with the same header, and the
	/// writes go after them.
	bool OpenOutput(const std::filesystem::path& path, std::span<const uint8_t> header, uint64_t kept);

	/// Writes bytes. Returns false when they were not all written.
	bool Append(std::span<const uint8_t> bytes);

	/// Writes one text line in one write, so a reader never sees half of it.
	bool Line(const std::string& text);

private:
	Common::File m_file;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_APPEND_FILE_H
