#ifndef LAUNCHER_INCLUDE_EXPERIMENTSETTINGS_H_
#define LAUNCHER_INCLUDE_EXPERIMENTSETTINGS_H_

#include <QString>
#include <QStringList>
#include <QVariantMap>
#include <Qt>

#include <array>
#include <cstddef>

class QSettings;

// Fork-only experiments from unmerged changes, each turned on by an opt-in emulator flag. They are
// kept apart from the upstream settings so a rebuilt release branch merges them cleanly.
struct ExperimentSettings {
	// This fork exists to try the experiments, so they start on.
	static constexpr bool DEFAULT_ENABLED = true;

	bool example_experiment_name     = DEFAULT_ENABLED;

	/// Reads the experiments, using the default for any key an older config does not have.
	template <class Settings>
	void ReadFrom(const Settings& settings);
	/// Adds the experiments to a settings map, stored as strings like every other launcher setting.
	void InsertInto(QVariantMap* values) const;
	/// Saves the experiments to a config file, in the same string form as InsertInto.
	void WriteTo(QSettings* settings) const;
	/// Appends the emulator flag of each experiment that is on.
	void AppendFlags(QStringList* args) const;
	/// Whether all, none or only some of the experiments are on.
	[[nodiscard]] Qt::CheckState CombinedState() const;
	/// Turns every experiment on or off.
	void SetAll(bool enabled);
};

// One experiment's field, settings key and emulator flag, listed together so a field can never be
// paired with another experiment's key or flag.
struct ExperimentEntry {
	bool ExperimentSettings::* field;
	const char*                key;
	const char*                flag;
};

inline constexpr std::array EXPERIMENT_ENTRIES {
    ExperimentEntry {&ExperimentSettings::example_experiment_name, 
										"experiment_example_experiment_name","--example_experiment_flag"},
};

// True when no two entries edit the same field, so with the size check below every field has
// exactly one entry.
constexpr bool ExperimentFieldsAreDistinct() {
	for (std::size_t first = 0; first < EXPERIMENT_ENTRIES.size(); first++) {
		for (std::size_t second = first + 1; second < EXPERIMENT_ENTRIES.size(); second++) {
			if (EXPERIMENT_ENTRIES[first].field == EXPERIMENT_ENTRIES[second].field) {
				return false;
			}
		}
	}
	return true;
}

// A field without a table entry would never be saved or passed, so every bool must have exactly
// one entry of its own.
static_assert(sizeof(ExperimentSettings) == EXPERIMENT_ENTRIES.size() * sizeof(bool),
              "every experiment needs a table entry");
static_assert(ExperimentFieldsAreDistinct(), "two experiment entries edit the same field");

template <class Settings>
void ExperimentSettings::ReadFrom(const Settings& settings) {
	for (const auto& entry: EXPERIMENT_ENTRIES) {
		this->*entry.field =
		    settings.value(QString::fromLatin1(entry.key), DEFAULT_ENABLED).toBool();
	}
}

#endif /* LAUNCHER_INCLUDE_EXPERIMENTSETTINGS_H_ */
