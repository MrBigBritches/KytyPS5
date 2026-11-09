#ifndef GRAPHICS_SHADER_STOREGEN_EBOOT_SHADER_SCAN_H
#define GRAPHICS_SHADER_STOREGEN_EBOOT_SHADER_SCAN_H

#include <cstdint>
#include <span>
#include <vector>

// Finds the compute shaders embedded in a game's eboot without a graphics device and without
// reading Config, so the shader store worker process can run it.
namespace Libs::Graphics::StoreGen {

/// One compute program found in an eboot.
struct ScannedProgram {
	/// XXH3 of `code`, the same hash the runtime gives the program when the game maps it.
	uint64_t code_hash = 0;
	/// The code size from the binary's footer, without the literal buffer. It is 0 when the binary
	/// has no valid footer.
	uint32_t code_size_bytes = 0;
	/// The literal buffer size from the footer.
	uint32_t literal_size_bytes = 0;
	/// The bytes that are hashed and translated, which are the code rounded up to 16 plus the
	/// literals. A binary with no valid footer is hashed and kept whole.
	std::vector<uint32_t> code;
	/// The .shader_header section, with the user data tables and SH registers the key is built
	/// from.
	std::vector<uint8_t> header;
	/// The .metadata section, which is empty when the ELF has none.
	std::vector<uint8_t> metadata;
	/// Where the ELF starts in the eboot, for diagnostics.
	uint64_t elf_offset = 0;
};

/// How many candidates the scan met and what became of them.
struct ScanCounts {
	uint32_t elf_candidates = 0; ///< Places where the bytes 7f 45 4c 46 appear.
	uint32_t rejected = 0; ///< Candidates that are malformed, or whose declared size is not usable.
	uint32_t graphics_skipped = 0; ///< Well-formed shader ELFs that are not compute programs.
	uint32_t duplicates       = 0; ///< Compute programs whose code hash was already seen.
	/// Duplicates whose header bytes differ from those of the program that was kept.
	uint32_t duplicate_header_differs = 0;
	/// Compute programs kept with no valid footer. The runtime hashes these whole, and so does the
	/// scan.
	uint32_t no_footer = 0;
	/// Compute ELFs whose declared binary size runs past the ELF's own sections and table.
	uint32_t oversize = 0;
};

/// What a scan found.
struct ScanResult {
	std::vector<ScannedProgram> programs; ///< In eboot order.
	ScanCounts                  counts;
};

/// Scans an eboot for compute shader programs.
/// Every offset and size read from the file is checked against the file, so a truncated or hostile
/// blob gives fewer programs and never reads out of bounds. Programs are deduplicated by code hash
/// and the first ELF in the file wins.
/// @param eboot the whole eboot file.
ScanResult ScanEbootShaders(std::span<const uint8_t> eboot);

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_EBOOT_SHADER_SCAN_H
