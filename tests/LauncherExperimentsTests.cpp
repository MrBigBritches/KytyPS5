#include "configuration.h"
#include "experimentSettings.h"
#include "experimentsTab.h"

#include <QApplication>
#include <QCheckBox>
#include <QJsonObject>
#include <QProcess>
#include <QSettings>
#include <QSignalBlocker>
#include <QTemporaryDir>
#include <QVariantMap>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <utility>

namespace {

// Aborts with a message, so a failure stops the run where it happened.
void Check(bool value, const char* message) {
	if (!value) {
		std::fprintf(stderr, "LauncherExperimentsTests: failed: %s\n", message);
		std::abort();
	}
}

// The flags the emulator accepts, written out here rather than taken from the table under test,
// so a typo in the table fails.
const QStringList ALL_FLAGS {"--audio-compile-hold", "--structured-jump-tables",
                             "--select-guarded-loads"};

// Builds settings with each experiment set explicitly.
ExperimentSettings Make(bool audio, bool structured, bool selected) {
	ExperimentSettings settings;
	settings.audio_compile_hold     = audio;
	settings.structured_jump_tables = structured;
	settings.select_guarded_loads   = selected;
	return settings;
}

// The emulator flags the settings would pass.
QStringList Flags(const ExperimentSettings& settings) {
	QStringList args;
	settings.AppendFlags(&args);
	return args;
}

// Every experiment starts on, so a default config passes all three flags in table order.
void TestDefaultsPassEveryFlag() {
	Check(Flags(ExperimentSettings {}) == ALL_FLAGS, "default experiments did not pass every flag");
}

// Each experiment on alone passes exactly its own flag. Comparing whole lists catches a field
// paired with another experiment's flag, which a check for any flag would miss.
void TestEachExperimentPassesOnlyItsFlag() {
	Check(Flags(Make(true, false, false)) == QStringList {"--audio-compile-hold"},
	      "audio compile hold alone did not pass exactly its flag");
	Check(Flags(Make(false, true, false)) == QStringList {"--structured-jump-tables"},
	      "structured jump tables alone did not pass exactly its flag");
	Check(Flags(Make(false, false, true)) == QStringList {"--select-guarded-loads"},
	      "selected guarded loads alone did not pass exactly its flag");
	Check(Flags(Make(false, false, false)).isEmpty(), "experiments all off still passed a flag");
}

// The launcher builds one argument list, so the flags must be appended, not replace it.
void TestFlagsAppendToExistingArguments() {
	QStringList args {"--sentinel"};
	Make(true, false, false).AppendFlags(&args);
	Check(args == QStringList({"--sentinel", "--audio-compile-hold"}),
	      "experiment flags replaced the existing arguments");
}

// A config saved before the tab existed has none of the keys and must read as the defaults.
// Starting from all off means a ReadFrom that changes nothing cannot pass.
void TestMissingKeysReadAsDefaults() {
	auto from_map = Make(false, false, false);
	from_map.ReadFrom(QVariantMap {});
	Check(from_map.CombinedState() == Qt::Checked, "missing map keys did not read as the defaults");

	QTemporaryDir dir;
	Check(dir.isValid(), "temporary directory was not created");
	QSettings ini(dir.filePath("empty.ini"), QSettings::IniFormat);
	auto      from_ini = Make(false, false, false);
	from_ini.ReadFrom(ini);
	Check(from_ini.CombinedState() == Qt::Checked, "missing INI keys did not read as the defaults");
}

// Uneven values written to an INI read back unchanged, so each key reaches its own field.
void TestValuesRoundTripThroughIni() {
	QTemporaryDir dir;
	Check(dir.isValid(), "temporary directory was not created");
	for (const auto& written: {Make(true, false, true), Make(false, true, false)}) {
		const auto path = dir.filePath("round_trip.ini");
		{
			QSettings ini(path, QSettings::IniFormat);
			QVariantMap values;
			written.InsertInto(&values);
			for (auto it = values.cbegin(); it != values.cend(); ++it) ini.setValue(it.key(), it.value());
		}
		QSettings ini(path, QSettings::IniFormat);
		auto      read = Make(!written.audio_compile_hold, !written.structured_jump_tables,
		                      !written.select_guarded_loads);
		read.ReadFrom(ini);
		Check(Flags(read) == Flags(written), "experiments did not round-trip through an INI file");
	}
}

// Like every other launcher setting in the config file, each value is stored as "true" or "false".
void TestValuesAreStoredAsStrings() {
	QVariantMap values;
	Make(true, false, true).InsertInto(&values);
	Check(values.size() == ALL_FLAGS.size(), "experiments did not add exactly one key each");
	// QVariant equality converts between types, so a bool would pass the value checks below.
	for (auto it = values.cbegin(); it != values.cend(); ++it) {
		Check(it.value().typeId() == QMetaType::QString, "an experiment was not stored as a string");
	}
	Check(values.value("experiment_audio_compile_hold") == QVariant(QStringLiteral("true")) &&
	          values.value("experiment_structured_jump_tables") == QVariant(QStringLiteral("false")) &&
	          values.value("experiment_select_guarded_loads") == QVariant(QStringLiteral("true")),
	      "experiments were not stored as true or false strings under their keys");
}

// Each saved key, written out here, turns off only its own experiment. Game-settings import is
// keyed by name, and a round trip through the same table cannot catch two keys swapped in it.
void TestEachKeyReachesItsOwnExperiment() {
	const std::pair<const char*, QStringList> cases[] = {
	    {"experiment_audio_compile_hold", {"--structured-jump-tables", "--select-guarded-loads"}},
	    {"experiment_structured_jump_tables", {"--audio-compile-hold", "--select-guarded-loads"}},
	    {"experiment_select_guarded_loads", {"--audio-compile-hold", "--structured-jump-tables"}},
	};
	for (const auto& [key, remaining]: cases) {
		ExperimentSettings settings;
		settings.ReadFrom(QVariantMap {{QString::fromLatin1(key), QStringLiteral("false")}});
		Check(Flags(settings) == remaining, "a saved key did not turn off only its own experiment");
	}
}

// The Enable all box shows all, none or some, and turns every experiment on or off together.
void TestCombinedStateAndSetAll() {
	Check(Make(true, true, true).CombinedState() == Qt::Checked, "all on was not Checked");
	Check(Make(false, false, false).CombinedState() == Qt::Unchecked, "all off was not Unchecked");
	Check(Make(true, false, true).CombinedState() == Qt::PartiallyChecked,
	      "some on was not PartiallyChecked");
	auto settings = Make(true, false, true);
	settings.SetAll(false);
	Check(Flags(settings).isEmpty(), "SetAll(false) left an experiment on");
	settings.SetAll(true);
	Check(Flags(settings) == ALL_FLAGS, "SetAll(true) left an experiment off");
}

// The labels, written out here, sit on the boxes of the experiments they name.
void TestBoxesNameTheirExperiments() {
	ExperimentsTab tab;
	const QStringList expected {"Audio compile hold", "Structured jump tables",
	                            "Selected guarded loads"};
	for (std::size_t index = 0; index < tab.Boxes().size(); index++) {
		Check(tab.Boxes()[index]->text() == expected[static_cast<int>(index)],
		      "an experiment box shows another experiment's label");
	}
	tab.Load(Make(false, true, false));
	Check(!tab.Boxes()[0]->isChecked() && tab.Boxes()[1]->isChecked() && !tab.Boxes()[2]->isChecked(),
	      "loading checked the wrong box");
	Check(Flags(tab.Values()) == QStringList {"--structured-jump-tables"},
	      "the ticked boxes did not read back as the same experiments");
}

// Enable all shows whether all, none or some of the experiments are on.
void TestEnableAllShowsTheExperiments() {
	ExperimentsTab tab;
	Check(tab.EnableAllBox()->checkState() == Qt::Checked, "a new tab did not show all on");
	tab.Load(Make(false, false, false));
	Check(tab.EnableAllBox()->checkState() == Qt::Unchecked, "all off did not show Unchecked");
	tab.Load(Make(true, false, true));
	Check(tab.EnableAllBox()->checkState() == Qt::PartiallyChecked,
	      "some on did not show the mixed state");
}

// Clicking Enable all turns everything on from mixed or all off, and off from all on. Clicking it
// from all off must not land on the mixed state, which Qt's own tri-state cycle would do.
void TestEnableAllClicks() {
	ExperimentsTab tab;
	tab.Load(Make(true, false, true));
	tab.EnableAllBox()->click();
	Check(Flags(tab.Values()) == ALL_FLAGS && tab.EnableAllBox()->checkState() == Qt::Checked,
	      "clicking Enable all from mixed did not turn everything on");
	tab.EnableAllBox()->click();
	Check(Flags(tab.Values()).isEmpty() && tab.EnableAllBox()->checkState() == Qt::Unchecked,
	      "clicking Enable all from all on did not turn everything off");
	tab.EnableAllBox()->click();
	Check(Flags(tab.Values()) == ALL_FLAGS && tab.EnableAllBox()->checkState() == Qt::Checked,
	      "clicking Enable all from all off did not turn everything on");
}

// Changing one experiment updates Enable all, and changing it back restores it.
void TestEnableAllFollowsEachBox() {
	ExperimentsTab tab;
	tab.Boxes()[1]->click();
	Check(tab.EnableAllBox()->checkState() == Qt::PartiallyChecked,
	      "turning one experiment off did not show the mixed state");
	tab.Boxes()[1]->click();
	Check(tab.EnableAllBox()->checkState() == Qt::Checked,
	      "turning it back on did not show all on");
}

// Loading values the boxes already hold emits no toggles, so Enable all must still be refreshed,
// as a Reset to the same values relies on.
void TestLoadRefreshesAStaleEnableAll() {
	ExperimentsTab tab;
	{
		const QSignalBlocker blocker(tab.EnableAllBox());
		tab.EnableAllBox()->setCheckState(Qt::Unchecked);
	}
	tab.Load(ExperimentSettings {});
	Check(tab.EnableAllBox()->checkState() == Qt::Checked,
	      "loading unchanged values left Enable all stale");
}

// A game's experiments are saved with its config file and read back unchanged. Starting the
// loaded config from the opposite values means a missing save or load cannot pass.
void TestConfigurationSavesExperiments() {
	QTemporaryDir dir;
	Check(dir.isValid(), "temporary directory was not created");
	for (const auto& written: {Make(true, false, true), Make(false, true, false)}) {
		const auto path = dir.filePath("config.ini");
		{
			Configuration saved;
			saved.experiments = written;
			QSettings ini(path, QSettings::IniFormat);
			saved.WriteSettings(&ini);
		}
		Configuration loaded;
		loaded.experiments = Make(!written.audio_compile_hold, !written.structured_jump_tables,
		                          !written.select_guarded_loads);
		QSettings ini(path, QSettings::IniFormat);
		loaded.ReadSettings(&ini);
		Check(Flags(loaded.experiments) == Flags(written),
		      "experiments did not survive saving and loading a game config");
	}
}

// Global settings reach games through CopyEmulatorSettingsFrom, so it must carry the experiments.
void TestGlobalSettingsCopyExperiments() {
	Configuration global;
	global.experiments = Make(true, false, true);
	Configuration game;
	game.experiments = Make(false, true, false);
	game.CopyEmulatorSettingsFrom(global);
	Check(Flags(game.experiments) == Flags(global.experiments),
	      "copying the global settings did not carry the experiments");
}

// Exported game settings leave the experiments out, so launchers without them can import the
// file. Importing keeps the game's own experiments, and an experiment key is not accepted.
void TestExportLeavesExperimentsOut() {
	Configuration config;
	config.experiments = Make(true, false, true);
	const auto exported = config.GameSettings();
	for (auto it = exported.cbegin(); it != exported.cend(); ++it) {
		Check(!it.key().startsWith(QStringLiteral("experiment_")),
		      "exported game settings included an experiment");
	}
	QString error;
	Check(config.SetGameSettings(QJsonObject::fromVariantMap(exported), error),
	      "re-importing exported game settings failed");
	Check(Flags(config.experiments) == QStringList({"--audio-compile-hold", "--select-guarded-loads"}),
	      "importing game settings changed the game's experiments");
	Check(!config.SetGameSettings(QJsonObject {{"experiment_audio_compile_hold", "false"}}, error),
	      "importing an experiment key was accepted");
}

// Runs the emulator's argument check with the given flags and returns its exit code.
int EmulatorAccepts(const QString& emulator, const QStringList& flags) {
	constexpr int timeout_ms = 30000;
	QProcess process;
	process.start(emulator, QStringList {"--help"} + flags);
	Check(process.waitForFinished(timeout_ms), "the emulator did not finish its argument check");
	return process.exitCode();
}

// The emulator rejects any option it does not know, so a flag it lacks would stop every launch.
// An unknown flag must be rejected too, which shows the check can fail.
void TestEmulatorAcceptsEveryFlag(const QString& emulator) {
	Check(EmulatorAccepts(emulator, Flags(ExperimentSettings {})) == 0,
	      "the emulator rejected an experiment flag the launcher passes");
	// The unknown flag goes first, as a missing flag among the real ones would.
	Check(EmulatorAccepts(emulator, QStringList {"--experiments-not-a-flag"} +
	                                    Flags(ExperimentSettings {})) != 0,
	      "the emulator accepted an unknown flag, so this check proves nothing");
}

} // namespace

int main(int argc, char* argv[]) {
	// The tab is a widget, so the tests need an application, drawn nowhere.
	qputenv("QT_QPA_PLATFORM", "offscreen");
	QApplication app(argc, argv);
	TestDefaultsPassEveryFlag();
	TestEachExperimentPassesOnlyItsFlag();
	TestFlagsAppendToExistingArguments();
	TestMissingKeysReadAsDefaults();
	TestValuesRoundTripThroughIni();
	TestValuesAreStoredAsStrings();
	TestEachKeyReachesItsOwnExperiment();
	TestCombinedStateAndSetAll();
	TestBoxesNameTheirExperiments();
	TestEnableAllShowsTheExperiments();
	TestEnableAllClicks();
	TestEnableAllFollowsEachBox();
	TestLoadRefreshesAStaleEnableAll();
	TestConfigurationSavesExperiments();
	TestGlobalSettingsCopyExperiments();
	TestExportLeavesExperimentsOut();
	// ctest passes the emulator, so its flags are checked against the launcher's.
	if (argc > 1) {
		TestEmulatorAcceptsEveryFlag(QString::fromLocal8Bit(argv[1]));
	}
	std::puts("LauncherExperimentsTests: all cases passed");
	return 0;
}
