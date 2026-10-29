#include "experimentsTab.h"

#include <QCheckBox>
#include <QFrame>
#include <QGroupBox>
#include <QLabel>
#include <QSignalBlocker>
#include <QVBoxLayout>

#include <cstddef>

namespace {

// The text of one experiment's box, keyed by its field so a label cannot land on the wrong box.
struct ExperimentText {
	bool ExperimentSettings::* field;
	const char*                label;
	const char*                tooltip;
};

constexpr std::array<ExperimentText, EXPERIMENT_ENTRIES.size()> EXPERIMENT_TEXTS {{
    {&ExperimentSettings::audio_compile_hold, QT_TRANSLATE_NOOP("ExperimentsTab", "Audio compile hold"),
     QT_TRANSLATE_NOOP("ExperimentsTab",
                       "Pause game audio while a long shader pipeline compile freezes the game, so "
                       "audio-paced cutscenes stay in sync.")},
    {&ExperimentSettings::structured_jump_tables,
     QT_TRANSLATE_NOOP("ExperimentsTab", "Structured jump tables"),
     QT_TRANSLATE_NOOP("ExperimentsTab",
                       "Compile shader jump tables as structured control flow instead of a "
                       "dispatcher loop, which makes shader compiles much faster. Changing this "
                       "recompiles the game's shaders.")},
    {&ExperimentSettings::select_guarded_loads,
     QT_TRANSLATE_NOOP("ExperimentsTab", "Selected guarded loads"),
     QT_TRANSLATE_NOOP("ExperimentsTab",
                       "Emit bounds-checked shader loads as selects instead of branches, which "
                       "speeds up pipeline compiles. Changing this recompiles the game's shaders.")},
}};

// True when the text table lists the experiments in the same order as the settings table, so a
// box built from one index always shows the text of the field it edits.
constexpr bool TextsMatchEntries() {
	for (std::size_t index = 0; index < EXPERIMENT_ENTRIES.size(); index++) {
		if (EXPERIMENT_TEXTS[index].field != EXPERIMENT_ENTRIES[index].field) {
			return false;
		}
	}
	return true;
}
static_assert(TextsMatchEntries(), "experiment texts must follow EXPERIMENT_ENTRIES order");

} // namespace

ExperimentsTab::ExperimentsTab(QWidget* parent): QWidget(parent) {
	auto* group  = new QGroupBox(tr("Experimental features"), this);
	auto* layout = new QVBoxLayout(group);
	auto* note   = new QLabel(tr("Experimental features from unmerged changes. They may cause "
	                             "problems in some games."),
	                          group);
	note->setWordWrap(true);
	layout->addWidget(note);
	m_enable_all = new QCheckBox(tr("Enable all"), group);
	m_enable_all->setObjectName("checkBox_experiments_enable_all");
	m_enable_all->setToolTip(tr("Turn all experimental features on or off at once."));
	layout->addWidget(m_enable_all);
	auto* separator = new QFrame(group);
	separator->setFrameShape(QFrame::HLine);
	separator->setFrameShadow(QFrame::Sunken);
	layout->addWidget(separator);
	for (std::size_t index = 0; index < EXPERIMENT_ENTRIES.size(); index++) {
		const auto& text = EXPERIMENT_TEXTS[index];
		m_boxes[index]   = new QCheckBox(tr(text.label), group);
		m_boxes[index]->setToolTip(tr(text.tooltip));
		layout->addWidget(m_boxes[index]);
		connect(m_boxes[index], &QCheckBox::toggled, this, &ExperimentsTab::RefreshEnableAll);
	}
	connect(m_enable_all, &QCheckBox::clicked, this, &ExperimentsTab::ApplyEnableAll);
	auto* page_layout = new QVBoxLayout(this);
	page_layout->addWidget(group);
	page_layout->addStretch();
	Load(ExperimentSettings {});
}

void ExperimentsTab::Load(const ExperimentSettings& settings) {
	for (std::size_t index = 0; index < EXPERIMENT_ENTRIES.size(); index++) {
		m_boxes[index]->setChecked(settings.*EXPERIMENT_ENTRIES[index].field);
	}
	// Ticking a box to the value it already has emits nothing, so refresh here as well.
	RefreshEnableAll();
}

ExperimentSettings ExperimentsTab::Values() const {
	ExperimentSettings settings;
	for (std::size_t index = 0; index < EXPERIMENT_ENTRIES.size(); index++) {
		settings.*EXPERIMENT_ENTRIES[index].field = m_boxes[index]->isChecked();
	}
	return settings;
}

void ExperimentsTab::RefreshEnableAll() {
	const auto        state = Values().CombinedState();
	const QSignalBlocker blocker(m_enable_all);
	// The mixed state is shown only when the experiments differ, never reached by clicking.
	m_enable_all->setTristate(state == Qt::PartiallyChecked);
	m_enable_all->setCheckState(state);
}

void ExperimentsTab::ApplyEnableAll() {
	// Qt has already moved the box to its next state, so decide from the experiments instead.
	// From all on a click turns them off, and from mixed or all off it turns them on.
	auto settings = Values();
	settings.SetAll(settings.CombinedState() != Qt::Checked);
	Load(settings);
}
