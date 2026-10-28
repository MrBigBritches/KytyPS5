#include "experimentSettings.h"

#include <QSettings>
#include <QVariant>

#include <cstddef>

void ExperimentSettings::InsertInto(QVariantMap* values) const {
	for (const auto& entry: EXPERIMENT_ENTRIES) {
		// Stored as strings like every other launcher setting in the config file.
		values->insert(QString::fromLatin1(entry.key), QVariant(this->*entry.field).toString());
	}
}

void ExperimentSettings::WriteTo(QSettings* settings) const {
	QVariantMap values;
	InsertInto(&values);
	for (auto it = values.cbegin(); it != values.cend(); ++it) {
		settings->setValue(it.key(), it.value());
	}
}

void ExperimentSettings::AppendFlags(QStringList* args) const {
	for (const auto& entry: EXPERIMENT_ENTRIES) {
		if (this->*entry.field) {
			args->append(QString::fromLatin1(entry.flag));
		}
	}
}

Qt::CheckState ExperimentSettings::CombinedState() const {
	std::size_t enabled = 0;
	for (const auto& entry: EXPERIMENT_ENTRIES) {
		enabled += this->*entry.field ? 1u : 0u;
	}
	if (enabled == 0) {
		return Qt::Unchecked;
	}
	return enabled == EXPERIMENT_ENTRIES.size() ? Qt::Checked : Qt::PartiallyChecked;
}

void ExperimentSettings::SetAll(bool enabled) {
	for (const auto& entry: EXPERIMENT_ENTRIES) {
		this->*entry.field = enabled;
	}
}
