#ifndef EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_FRAMESTATS_H_
#define EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_FRAMESTATS_H_

#include <cstddef>
#include <vector>

namespace Libs::Graphics::SystemUi {

// Samples of one measurement over a span of frames, such as a toast's draw time while it is on
// screen. Every sample counts towards the mean and the maximum. Only the first `capacity` are
// kept for the 95th percentile, so adding never allocates after construction.
class FrameStats {
public:
	/// Statistics that keep up to `capacity` samples for the percentile.
	explicit FrameStats(size_t capacity);

	/// Adds one sample.
	void Add(double value);
	/// Forgets every sample, keeping the capacity.
	void Reset();

	/// How many samples were added, kept or not.
	[[nodiscard]] size_t Count() const { return m_count; }
	/// The mean of every sample, or 0 with none.
	[[nodiscard]] double Mean() const;
	/// The largest sample, or 0 with none.
	[[nodiscard]] double Max() const;
	/// The 95th percentile of the kept samples by nearest rank, or 0 with none. It sorts a copy, so
	/// call it when reporting, not every frame.
	[[nodiscard]] double P95() const;

private:
	std::vector<double> m_kept;
	size_t              m_capacity;
	size_t              m_count = 0;
	double              m_sum   = 0.0;
	double              m_max   = 0.0;
};

} // namespace Libs::Graphics::SystemUi

#endif // EMULATOR_SRC_GRAPHICS_PRESENTATION_SYSTEM_CORE_FRAMESTATS_H_
