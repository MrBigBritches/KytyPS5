// Tests the path that the shader store generator reads the title's eboot from. The emulator builds
// it before the kernel's /app0 mount exists, because the graphics subsystem reads it in WarmUp
// while the kernel starts, so it has to be what the mount would resolve it to. The archive is a
// real .zar made by the writer of the archive tests, with the awkward characters of a real game's
// file name. Each check names what else could pass it and compares against a baseline that differs
// in one change.
#include "ArchiveTestFixture.h"
#include "common/archive.h"
#include "common/childProcess.h"
#include "common/file.h"
#include "graphics/shader/storegen/ebootSource.h"
#include "graphics/shader/storegen/storeGenerationGate.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace Libs::Graphics::StoreGen;
namespace fs = std::filesystem;

int g_failures = 0;

#define CHECK(condition)                                                                           \
	do {                                                                                           \
		if (!(condition)) {                                                                        \
			std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #condition);     \
			++g_failures;                                                                          \
		}                                                                                          \
	} while (false)

// A directory for one test, removed with it.
class TempDirectory {
public:
	TempDirectory() {
		static int counter = 0;
		m_path             = fs::temp_directory_path() /
		         ("kyty_eboot_source_tests_" + std::to_string(Common::CurrentProcessId()) + "_" +
		          std::to_string(counter++));
		fs::remove_all(m_path);
		fs::create_directories(m_path);
	}
	~TempDirectory() {
		std::error_code ignored;
		fs::remove_all(m_path, ignored);
	}
	TempDirectory(const TempDirectory&)            = delete;
	TempDirectory& operator=(const TempDirectory&) = delete;
	fs::path       operator/(const std::string& name) const { return m_path / name; }

private:
	fs::path m_path;
};

// What else the gate needs to pass, so that the eboot is what it decides on.
StoreGenerationInputs InputsFor(const fs::path& eboot) {
	StoreGenerationInputs inputs;
	inputs.background_warm_up_enabled = true;
	inputs.shader_store_path_empty    = false;
	inputs.store_header_empty         = false;
	inputs.eboot_path                 = eboot;
	inputs.executable_path            = Common::ExecutablePath();
	return inputs;
}

// The archive of a game, named as a real one is, with spaces and an ampersand.
fs::path MakeGameArchive(const TempDirectory& directory) {
	const auto                 archive = directory / "PPSA00000 - Game & Memories - 01.007.000.zar";
	const std::vector<uint8_t> payload {1, 2, 3};
	CHECK(ArchiveTests::CreateArchive(archive, payload));
	return archive;
}

// A game started from an archive: Main gives /app0 the root of the archive and the program
// /app0/eboot.bin. The path is the archive's member, it exists, the member is the eboot, and the
// gate accepts it. The baselines change one thing each: no path, which is what the generator got
// when the path was asked for before it was set, and a member that is not there.
void TestArchiveGame() {
	TempDirectory directory;
	const auto    archive = MakeGameArchive(directory);
	const auto    app0    = Common::MakeArchivePath(archive);

	const auto eboot = EbootPathFor(app0, "/app0/eboot.bin");
	CHECK(eboot == Common::MakeArchivePath(archive, "eboot.bin"));
	CHECK(Common::IsArchivePath(eboot));
	CHECK(Common::File::IsFileExisting(eboot));
	Common::File member;
	CHECK(member.Open(eboot, Common::File::Mode::Read) &&
	      member.Size() == ArchiveTests::Eboot.size());
	CHECK(!StoreGenerationAvailability(InputsFor(eboot)).has_value());

	CHECK(StoreGenerationAvailability(InputsFor({})) == "no_eboot");
	const auto missing = EbootPathFor(app0, "/app0/absent.bin");
	CHECK(!Common::File::IsFileExisting(missing));
	CHECK(StoreGenerationAvailability(InputsFor(missing)) == "no_eboot");
}

// A game in a folder, and a game started from one ELF file, whose /app0 is the file's folder. The
// result is the file itself.
void TestFolderGames() {
	TempDirectory directory;
	{
		std::FILE* file = std::fopen((directory / "eboot.bin").string().c_str(), "wb");
		CHECK(file != nullptr);
		if (file) std::fclose(file);
		file = std::fopen((directory / "game.elf").string().c_str(), "wb");
		if (file) std::fclose(file);
	}
	const auto folder = directory / "";
	const auto app0   = fs::path(folder);
	CHECK(EbootPathFor(app0, "/app0/eboot.bin") == app0 / "eboot.bin");
	CHECK(Common::File::IsFileExisting(EbootPathFor(app0, "/app0/eboot.bin")));
	CHECK(EbootPathFor(app0, "/app0/game.elf") == app0 / "game.elf");
	CHECK(
	    !StoreGenerationAvailability(InputsFor(EbootPathFor(app0, "/app0/game.elf"))).has_value());
	// A program that is not under /app0 is a host path, which stays as it is.
	const auto host = directory / "elsewhere.elf";
	CHECK(EbootPathFor(app0, host) == host);
	// The prefix is a directory: a guest path that only begins with the letters is not under /app0.
	CHECK(EbootPathFor(app0, "/app0x/eboot.bin") == fs::path("/app0x/eboot.bin"));
}

// The path is asked for by the pipeline cache while the kernel starts, so a setter that comes after
// the first read is too late, and says so. The path is still kept, and the first one stays. The
// baseline sets it first, which is how Run does it.
void TestSetterMustComeFirst() {
	ClearEbootPathForTest();
	CHECK(SetEbootPath("first.eboot"));
	CHECK(EbootPath() == fs::path("first.eboot"));
	CHECK(SetEbootPath("second.eboot"));
	CHECK(EbootPath() == fs::path("first.eboot"));

	ClearEbootPathForTest();
	CHECK(EbootPath().empty());
	CHECK(!SetEbootPath("late.eboot"));
	CHECK(EbootPath() == fs::path("late.eboot"));
	ClearEbootPathForTest();
}

} // namespace

int main() {
	TestArchiveGame();
	TestFolderGames();
	TestSetterMustComeFirst();
	if (g_failures == 0) {
		std::puts("Shader store eboot source tests passed");
	}
	return g_failures == 0 ? 0 : 1;
}
