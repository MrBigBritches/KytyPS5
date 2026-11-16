#include "graphics/host_gpu/renderer/pipeline/prescanGraphics.h"

#include "common/assert.h"
#include "common/firstRunReport.h"
#include "common/gpuStall.h"
#include "common/logging/log.h"
#include "graphics/host_gpu/renderer/pipeline/prescanValidate.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/recompiler/ir/passes/ShaderInfoCollection.h"
#include "graphics/shader/shaderCompiler.h"

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <optional>
#include <string>
#include <thread>
#include <utility>

namespace Libs::Graphics {

namespace {

namespace IR = ShaderRecompiler::IR;
using Clock  = std::chrono::steady_clock;

// Microseconds in a millisecond, for the counters, which are kept in whole microseconds.
constexpr double US_PER_MS = 1000.0;
// Nanoseconds in a microsecond and in a second, for the busy time and its timeline.
constexpr uint64_t NS_PER_US     = 1000;
constexpr int64_t  NS_PER_SECOND = 1'000'000'000;

// The timeline second of a time on the NowNs clock, counted from the epoch, or nothing before the
// epoch, without one, or past the timeline's end.
std::optional<size_t> TimelineSecond(int64_t epoch_ns, int64_t at_ns) {
	if (epoch_ns == Common::FirstRunReport::NO_TIME_NS || at_ns < epoch_ns) {
		return std::nullopt;
	}
	const auto second = static_cast<uint64_t>((at_ns - epoch_ns) / NS_PER_SECOND);
	return second < GRAPHICS_PRESCAN_TIMELINE_SECONDS ? std::optional(static_cast<size_t>(second)) : std::nullopt;
}

// Raises an atomic maximum. Any thread may race on it, and a lost race retries.
void RaiseMax(std::atomic<uint64_t>& maximum, uint64_t value) {
	uint64_t seen = maximum.load(std::memory_order_relaxed);
	while (seen < value && !maximum.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
	}
}

uint64_t WholeMicros(Clock::duration duration) {
	return static_cast<uint64_t>(
	    std::max<int64_t>(0, std::chrono::duration_cast<std::chrono::microseconds>(duration).count()));
}

} // namespace

// The count is taken with a compare and swap, so two jobs never both take the last slot. A job that
// finds the cap full sleeps a step and looks again rather than waiting on a condition, which keeps
// the release a plain decrement. The cap is the governor's at each look, so a throttle signal stops
// new starts at once and jobs already running finish.
GraphicsJobSlot::GraphicsJobSlot(std::atomic<uint32_t>& running, GraphicsPrescanCounters& counters,
                                 const PipelineBuildQueue& queue)
    : m_running(running), m_counters(counters) {
	// The epoch is read before any mutex, so the report's lock is never taken inside one.
	const int64_t epoch_ns = Common::FirstRunReport::Global().EpochNs();
	bool          capped   = false;
	bool          paused   = false;
	while (!queue.Stopping()) {
		const uint32_t cap  = counters.cap_governor.CapAt(Common::FirstRunReport::NowNs(),
		                                                  Common::GpuStall::LastThrottleSignalNs(), epoch_ns);
		uint32_t       seen = m_running.load(std::memory_order_relaxed);
		if (seen < cap && m_running.compare_exchange_weak(seen, seen + 1, std::memory_order_acq_rel)) {
			m_held = true;
			break;
		}
		if (seen >= cap) {
			capped = true;
			paused = paused || cap < GRAPHICS_PRESCAN_MAX_RUNNING_JOBS;
			std::this_thread::sleep_for(GRAPHICS_PRESCAN_ADMIT_STEP);
		}
	}
	counters.capped_waits.fetch_add(capped ? 1 : 0, std::memory_order_relaxed);
	counters.paused_starts.fetch_add(paused ? 1 : 0, std::memory_order_relaxed);
	if (!m_held) {
		return;
	}
	const int64_t    now_ns   = Common::FirstRunReport::NowNs();
	std::scoped_lock lock(counters.busy_mutex);
	if (counters.busy_jobs++ == 0) {
		counters.busy_since    = Clock::now();
		counters.busy_since_ns = now_ns;
	}
	if (const auto second = TimelineSecond(epoch_ns, now_ns)) {
		auto& peak = counters.timeline_peak[*second];
		peak       = static_cast<uint8_t>(std::max<uint32_t>(peak, std::min<uint32_t>(counters.busy_jobs, UINT8_MAX)));
	}
}

// A slot that was never held, because the queue stopped, gives nothing back.
GraphicsJobSlot::~GraphicsJobSlot() {
	if (!m_held) {
		return;
	}
	const int64_t epoch_ns = Common::FirstRunReport::Global().EpochNs();
	const int64_t now_ns   = Common::FirstRunReport::NowNs();
	{
		std::scoped_lock lock(m_counters.busy_mutex);
		if (--m_counters.busy_jobs == 0) {
			m_counters.busy_ns += WholeMicros(Clock::now() - m_counters.busy_since) * NS_PER_US;
			AddBusyToTimeline(m_counters, epoch_ns, m_counters.busy_since_ns, now_ns);
		}
	}
	m_running.fetch_sub(1, std::memory_order_acq_rel);
}

namespace {
// Adds an interval on the NowNs clock to the timeline seconds it covers from the epoch, split at each
// second boundary, and marks those seconds in `peak` when given. An interval that began before the
// epoch counts from it, and the part past the timeline's end is dropped.
void AddIntervalToSeconds(std::array<uint32_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS>& seconds_us,
                          std::array<uint8_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS>* peak, int64_t epoch_ns,
                          int64_t begin_ns, int64_t end_ns) {
	if (epoch_ns == Common::FirstRunReport::NO_TIME_NS || end_ns <= begin_ns) {
		return;
	}
	int64_t at_ns = std::max(begin_ns, epoch_ns);
	while (at_ns < end_ns) {
		const auto second = TimelineSecond(epoch_ns, at_ns);
		if (!second) {
			return;
		}
		const int64_t second_end_ns = epoch_ns + static_cast<int64_t>(*second + 1) * NS_PER_SECOND;
		const int64_t piece_end_ns  = std::min(end_ns, second_end_ns);
		seconds_us[*second] += static_cast<uint32_t>((piece_end_ns - at_ns) / static_cast<int64_t>(NS_PER_US));
		if (peak != nullptr) {
			(*peak)[*second] = std::max<uint8_t>((*peak)[*second], 1);
		}
		at_ns = piece_end_ns;
	}
}
} // namespace

void AddBusyToTimeline(GraphicsPrescanCounters& counters, int64_t epoch_ns, int64_t begin_ns, int64_t end_ns) {
	AddIntervalToSeconds(counters.timeline_busy_us, &counters.timeline_peak, epoch_ns, begin_ns, end_ns);
}

// A new signal adds the part of its window that the window in force did not cover yet, so the reduced
// time is the union of the windows, and so is its timeline.
uint32_t GraphicsCapGovernor::CapAt(int64_t now_ns, int64_t signal_ns, int64_t epoch_ns) {
	std::scoped_lock lock(m_mutex);
	if (signal_ns != 0 && signal_ns != m_seen_signal_ns) {
		m_seen_signal_ns         = signal_ns;
		const int64_t new_until  = signal_ns + GRAPHICS_PRESCAN_REDUCED_CAP_NS;
		const int64_t added_from = std::max(signal_ns, m_reduced_until);
		if (new_until > added_from) {
			m_scheduled_ns += new_until - added_from;
			AddIntervalToSeconds(m_timeline_us, nullptr, epoch_ns, added_from, new_until);
		}
		m_reduced_until = std::max(m_reduced_until, new_until);
		m_triggers++;
	}
	return now_ns < m_reduced_until ? GRAPHICS_PRESCAN_REDUCED_RUNNING_JOBS : GRAPHICS_PRESCAN_MAX_RUNNING_JOBS;
}

uint64_t GraphicsCapGovernor::Triggers() const {
	std::scoped_lock lock(m_mutex);
	return m_triggers;
}

// The part of the window still ahead of now is not reduced time yet.
double GraphicsCapGovernor::ReducedMs(int64_t now_ns) const {
	constexpr double ns_per_ms = 1e6;
	std::scoped_lock lock(m_mutex);
	const int64_t    ahead_ns = std::clamp<int64_t>(m_reduced_until - now_ns, 0, m_scheduled_ns);
	return static_cast<double>(m_scheduled_ns - ahead_ns) / ns_per_ms;
}

std::array<uint32_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS> GraphicsCapGovernor::ReducedTimelineUs() const {
	std::scoped_lock lock(m_mutex);
	return m_timeline_us;
}

// The open interval is added to a copy, so reading the timeline changes nothing.
Common::FirstRunReport::PrescanGraphicsReport::Group GraphicsBusyTimeline(const GraphicsPrescanCounters& counters,
                                                                          int64_t epoch_ns, int64_t now_ns) {
	Common::FirstRunReport::PrescanGraphicsReport::Group group {"busy_timeline", {}};
	const auto last = TimelineSecond(epoch_ns, now_ns);
	if (epoch_ns == Common::FirstRunReport::NO_TIME_NS || now_ns < epoch_ns) {
		return group;
	}
	const size_t seconds = last ? *last + 1 : GRAPHICS_PRESCAN_TIMELINE_SECONDS;
	GraphicsPrescanCounters copy;
	{
		std::scoped_lock lock(counters.busy_mutex);
		copy.timeline_busy_us = counters.timeline_busy_us;
		copy.timeline_peak    = counters.timeline_peak;
		if (counters.busy_jobs != 0) {
			AddBusyToTimeline(copy, epoch_ns, counters.busy_since_ns, now_ns);
		}
	}
	const auto          reduced_us = counters.cap_governor.ReducedTimelineUs();
	std::vector<double> busy_ms(seconds);
	std::vector<double> peak(seconds);
	std::vector<double> reduced_ms(seconds);
	for (size_t second = 0; second < seconds; second++) {
		busy_ms[second]    = static_cast<double>(copy.timeline_busy_us[second]) / US_PER_MS;
		peak[second]       = static_cast<double>(copy.timeline_peak[second]);
		reduced_ms[second] = static_cast<double>(reduced_us[second]) / US_PER_MS;
	}
	group.series.emplace_back("busy_ms", std::move(busy_ms));
	group.series.emplace_back("peak_running", std::move(peak));
	group.series.emplace_back("reduced_ms", std::move(reduced_ms));
	return group;
}

// The interval still open when it is read counts up to now.
double GraphicsBusyFraction(const GraphicsPrescanCounters& counters) {
	const auto       now = Clock::now();
	std::scoped_lock lock(counters.busy_mutex);
	uint64_t         busy_ns = counters.busy_ns;
	if (counters.busy_jobs != 0) {
		busy_ns += WholeMicros(now - counters.busy_since) * NS_PER_US;
	}
	const uint64_t total_ns = WholeMicros(now - counters.started) * NS_PER_US;
	return total_ns == 0 ? 0.0 : static_cast<double>(busy_ns) / static_cast<double>(total_ns);
}

// The specialization is the one the walk read at worker time, which may be staler than the draw's, so
// each check is the predicate the compile itself EXITs on. ADD_TID is judged over every buffer, which
// is wider than info collection, which judges only the buffers an operation reaches.
std::optional<GraphicsPrescanSkip>
ValidateGraphicsSpecialization(ShaderType stage, const ShaderRecompiler::IR::ResourcePlan& plan,
                               const ShaderRecompiler::IR::ResourceSpecialization& specialization) {
	for (const auto& buffer: specialization.buffers) {
		if (ShaderRecompiler::IR::BufferAddTidUnsupported(stage, buffer.packed_stride)) {
			return GraphicsPrescanSkip::InvalidBufferDescriptor;
		}
	}
	if (FindInvalidDescriptor(plan, specialization)) {
		return GraphicsPrescanSkip::InvalidDescriptor;
	}
	return std::nullopt;
}

void GraphicsRejectTally::Note(const char* file, int line, const std::string& text) {
	auto where = fmt::format("{}:{}", file != nullptr ? file : "?", line);
	std::scoped_lock lock(m_mutex);
	const auto site = std::ranges::find(m_sites, where, &Site::where);
	if (site != m_sites.end()) {
		site->count++;
		return;
	}
	Log::Write({}, fmt::format("Graphics prescan recompiler_rejected at {}: {}\n", where, text));
	m_sites.push_back({std::move(where), text, 1});
}

std::vector<GraphicsRejectTally::Site> GraphicsRejectTally::Sites() const {
	std::scoped_lock lock(m_mutex);
	return m_sites;
}

// The counts go up before the post, so a worker that starts the job at once never takes them below
// zero, and the total is reserved with one increment, so two posters never both take the last place.
bool PostBounded(PipelineBuildQueue& queue, GraphicsQueueBound& bound, PipelineBuildQueue::Lane lane,
                 Common::UniqueFunction<void> work) {
	// A Head job waits at the head of the front lane, so it is counted with the front lane.
	const auto index = lane == PipelineBuildQueue::Lane::Back ? 1u : 0u;
	if (bound.pending.fetch_add(1, std::memory_order_relaxed) >= GRAPHICS_PRESCAN_QUEUE_CAP) {
		bound.pending.fetch_sub(1, std::memory_order_relaxed);
		bound.dropped_full.fetch_add(1, std::memory_order_relaxed);
		return false;
	}
	const uint32_t depth = bound.depth[index].fetch_add(1, std::memory_order_relaxed) + 1;
	RaiseMax(bound.depth_max[index], depth);
	auto job = queue.Post(
	    [&bound, index, work = std::move(work)]() mutable {
		    bound.depth[index].fetch_sub(1, std::memory_order_relaxed);
		    bound.pending.fetch_sub(1, std::memory_order_relaxed);
		    work();
	    },
	    lane);
	if (job == nullptr) {
		bound.depth[index].fetch_sub(1, std::memory_order_relaxed);
		bound.pending.fetch_sub(1, std::memory_order_relaxed);
		return false;
	}
	return true;
}

// Only ExitRejected is caught, so any other exception still reaches the queue as before.
bool RunRejectable(GraphicsRejectTally& tally, const std::function<void()>& step) {
	try {
		const Common::ExitRejectScope scope;
		step();
		return true;
	} catch (const Common::ExitRejected& rejected) {
		tally.Note(rejected.file, rejected.line, rejected.text);
		return false;
	}
}

namespace {

// The vertex tables one draw's vertex shader reads, copied through the reader. The attribute table is
// read up to the largest semantic and the buffer table up to the largest buffer an attribute names.
struct VertexTables {
	std::array<ShaderSemantic, ShaderVertexInputInfo::RES_MAX> semantics {};
	uint32_t                                                    semantic_count = 0;
	// A semantic is 8 bits, and a buffer index 5 bits with four words per buffer.
	std::array<uint32_t, 256>                                  attrib {};
	std::array<uint32_t, ShaderVertexInputInfo::RES_MAX * 4>   buffer {};
};

// A 64-bit table pointer from two user registers, as ShaderGetStaticVertexInputInfo builds it.
uint64_t UserPointer(const HW::UserSgprInfo& user, int reg) {
	return static_cast<uint64_t>(user.value[reg]) | (static_cast<uint64_t>(user.value[reg + 1]) << 32u);
}

// Copies the semantics and both tables. False where a read fails or a pointer is null, which the GPU
// thread refuses too.
bool ReadVertexTables(const ScannedDraw& draw, const ShaderMapSnapshot& vertex, GraphicsPrescanReader read,
                      void* userdata, VertexTables& tables) {
	constexpr uint32_t index_mask    = 0x1fu;
	constexpr uint32_t buffer_words  = 4;
	tables.semantic_count            = vertex.num_input_semantics;
	const auto& user                 = draw.vs_regs.gs_user_sgpr;
	const uint64_t attrib_address    = UserPointer(user, vertex.vertex_attrib_reg);
	const uint64_t buffer_address    = UserPointer(user, vertex.vertex_buffer_reg);
	auto semantic_words = std::span<uint32_t>(reinterpret_cast<uint32_t*>(tables.semantics.data()), tables.semantic_count);
	if (attrib_address == 0 || buffer_address == 0 || vertex.input_semantics_address == 0 ||
	    !read(userdata, vertex.input_semantics_address, semantic_words)) {
		return false;
	}
	uint32_t max_semantic = 0;
	for (uint32_t i = 0; i < tables.semantic_count; i++) {
		max_semantic = std::max<uint32_t>(max_semantic, tables.semantics[i].semantic);
	}
	if (!read(userdata, attrib_address, std::span<uint32_t>(tables.attrib.data(), max_semantic + 1u))) {
		return false;
	}
	uint32_t max_index = 0;
	for (uint32_t i = 0; i < tables.semantic_count; i++) {
		max_index = std::max(max_index, tables.attrib[tables.semantics[i].semantic] & index_mask);
	}
	return read(userdata, buffer_address, std::span<uint32_t>(tables.buffer.data(), (max_index + 1u) * buffer_words));
}

// The clip space and user clip planes GetGraphicsPrograms puts on the last pre-rasterization stage.
void AddClipState(const ScannedDraw& draw, std::array<uint32_t, 2> max_viewport, ShaderVertexInputInfo& info) {
	// GetGraphicsPrograms caps the clip extent at the guest's largest viewport.
	constexpr uint32_t max_guest_viewport = 16384;
	constexpr float    half               = 0.5f;
	if (draw.clip_control.clip_disable) {
		auto& clip = info.clip_space;
		for (size_t axis = 0; axis < 2; axis++) {
			clip.scale[axis]       = draw.viewport_scale[axis];
			clip.offset[axis]      = draw.viewport_offset[axis];
			clip.half_extent[axis] = static_cast<float>(std::min(max_viewport[axis], max_guest_viewport)) * half;
		}
		clip.enabled = true;
	} else if (draw.user_clip_planes != 0) {
		info.user_clip_planes = draw.user_clip_planes;
	}
}

// Whether every attribute's buffer can be fetched by TranslateEmbeddedFetch without EXIT. A fetch reads
// at most four components and which ones is only known after translation, so all four selectors are
// judged. That rejects only a buffer with a reserved selector (2 or 3) on a component no fetch reads.
bool EmbeddedFetchesValid(const ShaderVertexInputInfo& info) {
	constexpr uint32_t max_fetch_components = 4;
	for (int index = 0; index < info.resources_num; index++) {
		if (!ShaderRecompiler::EmbeddedFetchFormatValid(info.resources[index], max_fetch_components)) {
			return false;
		}
	}
	return true;
}

} // namespace

std::optional<GraphicsPrescanSkip> ScannedVertexInputInfo(const ScannedDraw& draw, const ShaderMapSnapshot& vertex,
                                                          std::array<uint32_t, 2> max_viewport,
                                                          GraphicsPrescanReader read, void* userdata,
                                                          ShaderVertexInputInfo& info) {
	// VGT_SHADER_STAGES_EN bit 22 runs the vertex stage in wave32, and LDS_SIZE counts 128 dwords.
	constexpr uint32_t wave32_stage_bit = 0x00400000u;
	constexpr uint32_t lds_granule      = 128;
	if (vertex.vertex_error) {
		return GraphicsPrescanSkip::VertexMetadata;
	}
	auto tables = std::make_unique<VertexTables>();
	if (vertex.vertex_attrib_reg >= 0) {
		if (vertex.num_input_semantics == 0 || vertex.num_input_semantics > ShaderVertexInputInfo::RES_MAX) {
			return GraphicsPrescanSkip::VertexMetadata;
		}
		if (!ReadVertexTables(draw, vertex, read, userdata, *tables)) {
			return GraphicsPrescanSkip::VertexTables;
		}
	}
	HW::ShaderRegisters sh {};
	sh.m_paClVsOutCntl = draw.pa_cl_vs_out_cntl;
	if (!ShaderTryStaticVertexInputInfo(sh, vertex.scratch_size_dwords, vertex.vertex_attrib_reg,
	                                    vertex.vertex_buffer_reg,
	                                    std::span<const ShaderSemantic>(tables->semantics.data(), tables->semantic_count),
	                                    tables->attrib.data(), tables->buffer.data(), info)) {
		return GraphicsPrescanSkip::VertexTables;
	}
	// The attribute widths and formats come from the tables, so one VertexInputFormats would EXIT on
	// is rejected before anything is translated.
	if (!PrescanVertexInputsSupported(info)) {
		return GraphicsPrescanSkip::InvalidVertexFormat;
	}
	if (!EmbeddedFetchesValid(info)) {
		return GraphicsPrescanSkip::InvalidVertexFetch;
	}
	info.wave_size      = (draw.shader_stages & wave32_stage_bit) != 0 ? 32u : 64u;
	info.lds_size_dwords = draw.gs_lds_size * lds_granule;
	AddClipState(draw, max_viewport, info);
	return std::nullopt;
}

} // namespace Libs::Graphics
