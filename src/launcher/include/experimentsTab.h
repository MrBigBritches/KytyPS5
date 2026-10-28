#ifndef LAUNCHER_INCLUDE_EXPERIMENTSTAB_H_
#define LAUNCHER_INCLUDE_EXPERIMENTSTAB_H_

#include "experimentSettings.h"

#include <QWidget>

#include <array>

class QCheckBox;

// The fork-only Experiments tab of the configuration dialog. It lives in its own widget so the
// dialog needs only a few added lines, which keeps rebuilt release branches merging cleanly.
class ExperimentsTab: public QWidget {
	Q_OBJECT

public:
	/// Builds the tab showing the default experiments.
	explicit ExperimentsTab(QWidget* parent = nullptr);

	/// Shows the given experiments and updates Enable all to match.
	void Load(const ExperimentSettings& settings);
	/// The experiments as they are currently ticked.
	[[nodiscard]] ExperimentSettings Values() const;
	/// The Enable all box, exposed so tests can click it as a user would.
	[[nodiscard]] QCheckBox* EnableAllBox() const { return m_enable_all; }
	/// The box of each experiment, in the order of EXPERIMENT_ENTRIES.
	[[nodiscard]] const std::array<QCheckBox*, EXPERIMENT_ENTRIES.size()>& Boxes() const {
		return m_boxes;
	}

private:
	// Sets Enable all to show whether all, none or some of the experiments are on.
	void RefreshEnableAll();
	// Turns every experiment on or off when Enable all is clicked.
	void ApplyEnableAll();

	QCheckBox*                                       m_enable_all = nullptr;
	std::array<QCheckBox*, EXPERIMENT_ENTRIES.size()> m_boxes {};
};

#endif /* LAUNCHER_INCLUDE_EXPERIMENTSTAB_H_ */
