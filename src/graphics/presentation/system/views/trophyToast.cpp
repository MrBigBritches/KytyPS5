#include "graphics/presentation/system/views/trophyToast.h"

#include "graphics/presentation/system/animation/easing.h"
#include "graphics/presentation/system/components/toast.h"
#include "graphics/presentation/system/element/interact.h"
#include "graphics/presentation/system/element/primitives.h"
#include "graphics/presentation/system/layout/anchor.h"

#include <algorithm>
#include <limits>
#include <utility>

namespace Libs::Graphics::SystemUi {

namespace {

using Milliseconds = std::chrono::duration<float, std::milli>;

constexpr float MS_PER_SECOND = 1000.0f;
// Trophy icons are 256 or 512 pixels square. Anything larger isn't an icon.
constexpr int MAX_ICON_SIDE = 512;
// The trophy grades, as libTrophy numbers them.
constexpr int32_t PLATINUM = 1;
constexpr int32_t GOLD     = 2;
constexpr int32_t SILVER   = 3;
constexpr int32_t BRONZE   = 4;

} // namespace

Timeline ToastTimeline(const ToastTokens& toast, float seconds) {
	const Vec2       above {0.0f, -toast.slide};
	const Transition enter {
	    {toast.enter_fade_ms, Ease::Enter}, {toast.enter_move_ms, Ease::Enter}, above, 1.0f};
	const Transition exit {
	    {toast.exit_fade_ms, Ease::Exit}, {toast.exit_move_ms, Ease::Exit}, above, 1.0f};
	const float moving = std::max(toast.enter_fade_ms, toast.enter_move_ms) +
	                     std::max(toast.exit_fade_ms, toast.exit_move_ms);
	return {0.0f, enter, std::max(0.0f, seconds * MS_PER_SECOND - moving), exit};
}

Color TrophyGradeColor(const Theme& theme, int32_t grade) {
	switch (grade) {
		case PLATINUM: return theme.colors.trophy.platinum;
		case GOLD: return theme.colors.trophy.gold;
		case SILVER: return theme.colors.trophy.silver;
		case BRONZE: return theme.colors.trophy.bronze;
		default: return {};
	}
}

TrophyToastView::TrophyToastView(ImageStore& images): m_images(images) {}

TrophyToastView::~TrophyToastView() {
	Retire();
}

void TrophyToastView::Show(TrophyUnlock unlock, std::string subtitle, const Theme& theme,
                           const TrophyToastOptions& options, UiClock::time_point now) {
	Retire();
	// A trophy without an icon is normal, so it isn't sent to the store to be reported as bad.
	const auto icon =
	    unlock.icon_png.empty() ? std::nullopt : m_images.Add(unlock.icon_png, MAX_ICON_SIDE);
	const auto     toast    = options.tokens.Apply(ToastDefaults(theme));
	const Timeline timeline = ToastTimeline(toast, options.seconds);
	m_shown                 = Shown {std::move(unlock),
                     std::move(subtitle),
                     icon,
                     options.tokens,
                     options.reduced_motion ? ReducedMotion(timeline, theme.motion.reduced_fade_max)
	                                                        : timeline,
                     m_next_serial++,
                     now,
                     toast.enter_fade_ms + toast.shine_delay_ms,
                     theme.motion.shine,
                     !options.reduced_motion};
}

void TrophyToastView::Retire() {
	if (m_shown.has_value() && m_shown->icon.has_value()) {
		m_images.Release(*m_shown->icon);
	}
	m_shown.reset();
}

ToastProps TrophyToastView::Props(const UiContext& context, UiClock::time_point now) const {
	const Color grade = TrophyGradeColor(context.theme, m_shown->unlock.grade);
	ToastProps  props {m_shown->unlock.name, m_shown->subtitle};
	if (m_shown->icon.has_value()) {
		props.icon      = m_images.Texture(*m_shown->icon);
		props.icon_size = m_images.Size(*m_shown->icon);
	}
	props.icon_glow    = grade;
	props.title_cup    = grade;
	props.tokens       = m_shown->tokens;
	const int32_t rank = m_shown->unlock.grade;
	if (m_shown->shines && (rank == PLATINUM || rank == GOLD)) {
		// The shine sweeps once, a moment after the toast has faded in.
		const float since = Milliseconds(now - m_shown->start).count() - m_shown->shine_start_ms;
		const float sweep = since / m_shown->shine_ms;
		if (sweep >= 0.0f && sweep <= 1.0f) {
			props.shine = ApplyEase(Ease::Standard, sweep);
		}
	}
	return props;
}

bool TrophyToastView::Draw(UiFrame& frame, ImDrawList& list, const Rect& area) {
	if (!m_shown.has_value()) {
		return false;
	}
	const auto props = Props(frame.Ui(), frame.Now());
	ToastParts toast = Toast(frame, props, frame.Scale().ContentFrame().width);
	// Keyed by the toast's own number, so its start is kept in the store under that key, and a
	// toast that replaces another starts its own entrance.
	if (Keyed(frame, toast.root, m_shown->serial).IsNone()) {
		Retire();
		return false;
	}
	const Vec2 size =
	    Measure(frame, toast.root,
	            {frame.Scale().ContentFrame().width, std::numeric_limits<float>::infinity()});
	const Rect placed = Place(area, size, Anchor::TopRight);
	// The entrance and exit slide and fade the whole card, as the keyed animation says. It is never
	// dismissed, so it leaves when its hold is over.
	if (!Arrange(frame, toast.root, placed) ||
	    WithAnimation(frame, toast.root, Animate {m_shown->timeline}).IsNone() ||
	    Render(frame, toast.root, list).error.has_value()) {
		Retire();
		return false;
	}
	const auto phase = PhaseOf(frame, toast.root);
	if (!phase.has_value() || *phase == Phase::Gone) {
		Retire();
		return false;
	}
	// It needs frames while it moves, and while it holds unless the hold has no end.
	const bool resting = *phase == Phase::Shown && HoldsUntilDismissed(m_shown->timeline);
	return !resting || props.shine.has_value();
}

} // namespace Libs::Graphics::SystemUi
