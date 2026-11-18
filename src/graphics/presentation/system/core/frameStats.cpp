#include "graphics/presentation/system/core/frameStats.h"

#include <algorithm>
#include <cmath>

namespace Libs::Graphics::SystemUi {

namespace {

constexpr double PERCENTILE = 0.95;

} // namespace

FrameStats::FrameStats(size_t capacity): m_capacity(capacity) {
	m_kept.reserve(capacity);
}

void FrameStats::Add(double value) {
	if (m_kept.size() < m_capacity) {
		m_kept.push_back(value);
	}
	m_max = m_count == 0 ? value : std::max(m_max, value);
	m_sum += value;
	m_count++;
}

void FrameStats::Reset() {
	m_kept.clear();
	m_count = 0;
	m_sum   = 0.0;
	m_max   = 0.0;
}

double FrameStats::Mean() const {
	return m_count == 0 ? 0.0 : m_sum / static_cast<double>(m_count);
}

double FrameStats::Max() const {
	return m_max;
}

double FrameStats::P95() const {
	if (m_kept.empty()) {
		return 0.0;
	}
	// The nearest rank is the smallest sample with at least 95% of the samples at or below it.
	auto         sorted = m_kept;
	const size_t rank =
	    static_cast<size_t>(std::ceil(PERCENTILE * static_cast<double>(sorted.size())));
	const auto at = sorted.begin() + static_cast<std::ptrdiff_t>(rank - 1);
	std::nth_element(sorted.begin(), at, sorted.end());
	return *at;
}

} // namespace Libs::Graphics::SystemUi
