#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANGRAPHICS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANGRAPHICS_H_

#include "common/firstRunReport.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineBuildQueue.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

// The pipeline cache's side of the graphics prescan (experimental). The scanner offers the draws it
// found, a low priority worker translates each new draw's pixel shader the way GetProgram would and
// posts the module into the generated intake, and two more build its fragment library ahead. The
// GPU thread still makes its own key and only takes a prescan module that FindModule matches. Nothing
// a worker does may EXIT on guest data, so every way a job can stop short is a counted skip.
namespace Libs::Graphics {

/// Why a translate job stopped short.
enum class GraphicsPrescanSkip : uint8_t {
	CodeUnmapped,          ///< No shader is mapped at the pixel shader's address any more.
	TooManyUserData,       ///< More user data registers than GetShaderParams accepts.
	CodeUnreadable,        ///< The program could not be read without faulting.
	CodeChanged,           ///< The program read does not hash as the shader map says.
	SemanticsUnreadable,   ///< The input semantics could not be read without faulting.
	VertexUnmapped,        ///< No shader is mapped at the vertex stage's address, so its kind is unknown.
	HasCalls,              ///< The program has GPU-selected calls, which the store does not hold.
	MaterializeFaulted,    ///< The resource walk met memory it could not read.
	MaterializeFailed,     ///< The resource walk failed.
	InvalidDescriptor,     ///< A descriptor the emitter would refuse, see FindInvalidDescriptor.
	ExceedsSpecialization, ///< More resources than the specialization, which the compile EXITs on.
	VertexIndirect,        ///< An indirect draw, whose vertex shader key carries the packet's offsets.
	VertexMesh,            ///< The vertex stage runs as a mesh shader, which the prescan leaves out.
	VertexMetadata,        ///< The vertex metadata was refused at map time, which the GPU thread EXITs on.
	VertexTables,          ///< The attribute or buffer tables were unreadable or ones the GPU thread EXITs on.
	InvalidBufferDescriptor, ///< A buffer asks for ADD_TID outside compute, which info collection EXITs on.
	InvalidVertexFormat,     ///< An attribute VertexInputFormats EXITs on, by width or format.
	InvalidVertexFetch,      ///< A buffer whose format or selects the embedded fetch translation EXITs on.
	RecompilerRejected,      ///< Translation or compilation EXITed on the job's data, caught by ExitRejectScope.
	DedupedLate,             ///< The same program key, push start and specialization was built or claimed.
	VertexNoPixel,           ///< A vertex job whose draw's pixel shader was not built, so its push start is unknown.
	Count,
};

/// How many whole seconds of the run the busy timeline covers, 15 minutes.
inline constexpr size_t GRAPHICS_PRESCAN_TIMELINE_SECONDS = 900;

/// The GPU thread's VertexInputFormats and ResolvePolygonMode, for the worker's vertex records and
/// pre-rasterization keys.
[[nodiscard]] std::vector<vk::Format> PrescanVertexInputFormats(const ShaderVertexInputInfo& input_info,
                                                                const ShaderRecompiler::IR::CompiledShaderInfo& program);
[[nodiscard]] vk::PolygonMode PrescanPolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back);
/// Whether VertexInputFormats can run on the info without EXIT, with the predicate it EXITs on.
[[nodiscard]] bool PrescanVertexInputsSupported(const ShaderVertexInputInfo& input_info);

/// Rejects a specialization the compile would EXIT on, after Materialize and right before the
/// compile: a descriptor the emitter refuses (FindInvalidDescriptor), or a buffer whose packed stride
/// asks for ADD_TID in a stage other than compute, which info collection refuses.
[[nodiscard]] std::optional<GraphicsPrescanSkip>
ValidateGraphicsSpecialization(ShaderType stage, const ShaderRecompiler::IR::ResourcePlan& plan,
                               const ShaderRecompiler::IR::ResourceSpecialization& specialization);

/// Reads guest memory without faulting, as the worker's reader does. Tests pass their own.
using GraphicsPrescanReader = bool (*)(void* userdata, uint64_t address, std::span<uint32_t> words);

/// Builds a scanned draw's vertex shader input info as PrepareProgram and GetGraphicsPrograms would for
/// a plain vertex stage, reading the semantics and the attribute and buffer tables through `read`.
/// @param vertex       the shader map's entry of the vertex program.
/// @param max_viewport the device's maxViewportDimensions, which CLIP_DISABLE's clip space reads.
/// @return the skip class where the GPU thread would EXIT or a read failed, otherwise nothing.
[[nodiscard]] std::optional<GraphicsPrescanSkip>
ScannedVertexInputInfo(const ScannedDraw& draw, const ShaderMapSnapshot& vertex, std::array<uint32_t, 2> max_viewport,
                       GraphicsPrescanReader read, void* userdata, ShaderVertexInputInfo& info);

/// The recompiler EXITs the worker's ExitRejectScope caught, by source line, for the report (experimental).
/// Each distinct line is logged once, the first time it is caught.
class GraphicsRejectTally {
public:
	/// One source line that rejected, with the first message and how many jobs it rejected.
	struct Site {
		std::string where;
		std::string text;
		uint64_t    count = 0;
	};
	/// Counts a caught EXIT and logs its line the first time.
	void Note(const char* file, int line, const std::string& text);
	/// Every line so far, in the order first caught.
	[[nodiscard]] std::vector<Site> Sites() const;

private:
	mutable std::mutex m_mutex;
	std::vector<Site>  m_sites;
};

/// Runs `step` under a Common::ExitRejectScope, so an EXIT in it throws instead of ending the process.
/// @return false where it threw, after noting the EXIT in `tally`. Only translation and compilation may
///         run here. MaterializeResources must not, since its generated walker code cannot unwind.
[[nodiscard]] bool RunRejectable(GraphicsRejectTally& tally, const std::function<void()>& step);

/// How long a throttle signal holds the graphics prescan at GRAPHICS_PRESCAN_REDUCED_RUNNING_JOBS, and
/// that cap. Each further signal re-arms the whole window. Jobs already running finish.
inline constexpr int64_t  GRAPHICS_PRESCAN_REDUCED_CAP_NS       = 3'000'000'000;
inline constexpr uint32_t GRAPHICS_PRESCAN_REDUCED_RUNNING_JOBS = 1;

/// The adaptive running-job cap (experimental). It is GRAPHICS_PRESCAN_MAX_RUNNING_JOBS, and drops to
/// GRAPHICS_PRESCAN_REDUCED_RUNNING_JOBS for GRAPHICS_PRESCAN_REDUCED_CAP_NS after each throttle signal
/// (Common::GpuStall::LastThrottleSignalNs). The clock is passed in, so a test can drive it. Thread safe.
class GraphicsCapGovernor {
public:
	/// The cap at `now_ns` given the last signal's time, 0 for none. A signal not seen before re-arms
	/// the window to end GRAPHICS_PRESCAN_REDUCED_CAP_NS after it and counts one trigger. With an epoch
	/// on the same clock, the reduced time is also kept per second from it for the busy timeline.
	[[nodiscard]] uint32_t CapAt(int64_t now_ns, int64_t signal_ns,
	                             int64_t epoch_ns = Common::FirstRunReport::NO_TIME_NS);
	/// The triggers seen, and the reduced time up to `now_ns` in milliseconds.
	[[nodiscard]] uint64_t Triggers() const;
	[[nodiscard]] double   ReducedMs(int64_t now_ns) const;
	/// The reduced time per timeline second, in microseconds, including a window that runs past now.
	[[nodiscard]] std::array<uint32_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS> ReducedTimelineUs() const;

private:
	mutable std::mutex                                       m_mutex;
	int64_t                                                  m_seen_signal_ns  = 0;
	int64_t                                                  m_reduced_until   = 0;
	int64_t                                                  m_scheduled_ns    = 0;
	uint64_t                                                 m_triggers        = 0;
	std::array<uint32_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS> m_timeline_us {};
};

/// The counts the workers and the GPU thread add to, and the report reads from any thread.
struct GraphicsPrescanCounters {
	/// The adaptive cap the running slots are taken under.
	GraphicsCapGovernor cap_governor;
	/// Graphics jobs that waited for a running slot under GRAPHICS_PRESCAN_MAX_RUNNING_JOBS, and of
	/// those the ones that waited while a reduced window lowered the cap.
	std::atomic<uint64_t> capped_waits {0};
	std::atomic<uint64_t> paused_starts {0};
	/// The time at least one graphics job held a running slot, for graphics_busy_fraction, kept on
	/// the 0 to 1 and 1 to 0 changes of the slot holders under the mutex, and when counting began.
	mutable std::mutex                    busy_mutex;
	uint32_t                              busy_jobs = 0;
	std::chrono::steady_clock::time_point busy_since {};
	uint64_t                              busy_ns = 0;
	std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
	/// The busy time and the most slot holders in each whole second from the report's epoch, under the
	/// same mutex, with the open interval's start on the NowNs clock. Seconds past the end are dropped.
	std::array<uint32_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS> timeline_busy_us {};
	std::array<uint8_t, GRAPHICS_PRESCAN_TIMELINE_SECONDS>  timeline_peak {};
	int64_t                                                 busy_since_ns = 0;
};

/// The most graphics jobs running at once across both queues, translate and library, so the graphics
/// prescan never takes more than two cores from the game however many workers wait.
inline constexpr uint32_t GRAPHICS_PRESCAN_MAX_RUNNING_JOBS = 2;
/// How long a graphics job sleeps between looks while the cap is full.
inline constexpr std::chrono::milliseconds GRAPHICS_PRESCAN_ADMIT_STEP {2};

/// The most jobs one graphics queue holds waiting, both lanes together. A job posted past it is dropped.
inline constexpr uint32_t GRAPHICS_PRESCAN_QUEUE_CAP = 512;

/// The waiting jobs of one graphics queue by lane, the deepest each lane got, and the jobs dropped
/// because the queue was full. Lane 0 is Front and lane 1 Back.
struct GraphicsQueueBound {
	std::atomic<uint32_t>                pending {0};
	std::array<std::atomic<uint32_t>, 2> depth {};
	std::array<std::atomic<uint64_t>, 2> depth_max {};
	std::atomic<uint64_t>                dropped_full {0};
};

/// Posts `work` to `lane` of `queue` unless the queue already holds GRAPHICS_PRESCAN_QUEUE_CAP waiting
/// jobs, which counts the job in dropped_full. The job leaves the counts when it starts.
/// @return whether the job was queued. False also after the queue stopped, which drops nothing.
[[nodiscard]] bool PostBounded(PipelineBuildQueue& queue, GraphicsQueueBound& bound, PipelineBuildQueue::Lane lane,
                               Common::UniqueFunction<void> work);

/// Holds one of the GRAPHICS_PRESCAN_MAX_RUNNING_JOBS running slots for a graphics job while it lives.
/// The compute prescan goes first by priority instead of a wait, since the graphics workers run a step
/// below its workers. The wait for a slot ends when the queue stops, and the job then holds none.
class GraphicsJobSlot {
public:
	GraphicsJobSlot(std::atomic<uint32_t>& running, GraphicsPrescanCounters& counters, const PipelineBuildQueue& queue);
	~GraphicsJobSlot();
	GraphicsJobSlot(const GraphicsJobSlot&)            = delete;
	GraphicsJobSlot& operator=(const GraphicsJobSlot&) = delete;

private:
	std::atomic<uint32_t>&   m_running;
	GraphicsPrescanCounters& m_counters;
	bool                     m_held = false;
};

/// The share of the time since the counters were made during which at least one graphics job held a
/// running slot, the base rate the prescan_graphics_running underrun tag is compared with.
[[nodiscard]] double GraphicsBusyFraction(const GraphicsPrescanCounters& counters);

/// Adds a busy interval, on the NowNs clock, to the timeline seconds it covers from `epoch_ns`, split at
/// each second boundary, and marks each of those seconds as having had at least one slot holder. The
/// caller holds busy_mutex.
void AddBusyToTimeline(GraphicsPrescanCounters& counters, int64_t epoch_ns, int64_t begin_ns, int64_t end_ns);

/// The busy_timeline group: busy_ms and peak_running per whole second from the report's epoch, up to
/// the second now, at most GRAPHICS_PRESCAN_TIMELINE_SECONDS. The interval still open counts up to now.
[[nodiscard]] Common::FirstRunReport::PrescanGraphicsReport::Group
GraphicsBusyTimeline(const GraphicsPrescanCounters& counters, int64_t epoch_ns, int64_t now_ns);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_PRESCANGRAPHICS_H_
