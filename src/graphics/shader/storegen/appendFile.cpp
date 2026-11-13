#include "graphics/shader/storegen/appendFile.h"

#include <algorithm>
#include <vector>

namespace Libs::Graphics::StoreGen {

bool AppendFile::OpenAppend(const std::filesystem::path& path) {
	Common::File::CreateDirectories(path.parent_path());
	if (m_file.Open(path, Common::File::Mode::ReadWrite)) {
		return m_file.Seek(m_file.Size());
	}
	return m_file.Create(path);
}

bool AppendFile::OpenOutput(const std::filesystem::path& path, std::span<const uint8_t> header,
                            uint64_t kept) {
	Common::File::CreateDirectories(path.parent_path());
	if (kept == 0) {
		return m_file.Create(path) && Append(header);
	}
	if (!m_file.Open(path, Common::File::Mode::ReadWrite) || kept < header.size() ||
	    m_file.Size() < kept) {
		return false;
	}
	std::vector<uint8_t> existing(header.size());
	m_file.Read(existing.data(), static_cast<uint32_t>(existing.size()));
	return std::ranges::equal(existing, header) && m_file.Truncate(kept) && m_file.Seek(kept);
}

bool AppendFile::Append(std::span<const uint8_t> bytes) {
	uint32_t written = 0;
	m_file.Write(bytes.data(), static_cast<uint32_t>(bytes.size()), &written);
	return written == bytes.size();
}

bool AppendFile::Line(const std::string& text) {
	const auto line = text + "\n";
	return Append({reinterpret_cast<const uint8_t*>(line.data()), line.size()});
}

} // namespace Libs::Graphics::StoreGen
