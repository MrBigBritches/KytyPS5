#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/firstRunReport.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"
#include "graphics/guest_gpu/command_processor/scanOrderGate.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;
class ScanDifferential;
class PrescanScanner;
struct PreparedScanJob;
struct GuestGpuOptions;
struct GuestGpuTestAccess;

class GuestGpu final {
public:
	/// Makes the GPU thread with default-made options, which ignore the Config, so a GPU made this
	/// way never runs the prescan even with --prescan-compute-builds. A real run goes through
	/// RenderContext::InitializeGpu, which passes GuestGpuOptions::FromConfig().
	explicit GuestGpu(RenderContext& renderer);
	/// Makes the GPU thread with the given options. The submit-time prescan, when the options turn it
	/// on, is started here on the main thread before the GPU thread, after the shader store's warm-up.
	GuestGpu(RenderContext& renderer, const GuestGpuOptions& options);
	~GuestGpu();
	KYTY_CLASS_NO_COPY(GuestGpu);

	void               Shutdown();
	[[nodiscard]] bool IsStopping();
	void               SendCommand(Common::UniqueFunction<void>&& command);
	void               SendCommandSync(Common::UniqueFunction<void>&& command);

	// Submitted command memory is borrowed and must remain valid until GPU execution completes.
	void Submit(std::span<const uint32_t> commands);
	void SubmitCompute(uint32_t queue, std::span<const uint32_t> commands);
	void SubmitFlipPreparation(uint64_t request_id);
	// Insert an ordered graphics drain; only a previous suspend point can block the caller.
	void SuspendPoint();
	// Wait for guest command processing, including all compute queues (not native GPU completion).
	void              WaitForIdle();
	[[nodiscard]] int GetFrameNum() const;

	[[nodiscard]] static bool IsGpuThread() noexcept;
	/// How many buffers the command processor of this thread has on its stack, the top-level list
	/// included, so a dispatch in the top-level list reads 1 and one behind two calls reads 3. It
	/// is 0 outside a CommandProcessor::Process, as in a command sent with SendCommandSync or a
	/// test. A command the GPU thread runs from inside ProcessPm4 reads the running submission's
	/// depth.
	[[nodiscard]] static uint32_t CurrentDispatchDepth() noexcept;

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;
	static_assert(QueueCount <= SCAN_MAX_QUEUES, "every queue needs its own scan order lock and sequence");

	enum class SubmissionType { Commands, FlipPreparation, SuspendPoint };

	struct Submission {
		SubmissionType            type     = SubmissionType::Commands;
		uint32_t                  queue_id = 0;
		std::span<const uint32_t> commands;
		Pm4Execution              execution;
		bool                      blocked         = false;
		uint64_t                  flip_request_id = 0;
		// First-run report context. Set in Enqueue and kept when a blocked submission goes back
		// to the queue, so its times stay those of the first enqueue. Its id is NO_SUBMISSION
		// with the report off.
		Common::FirstRunReport::SubmissionStamp report;
		// The sequence number --scan-differential gave the submission's scan, or 0 when it was not
		// scanned. Kept when a blocked submission goes back to the queue.
		uint64_t scan_seq = 0;
		// The number the prescan's job queue gave the submission's job, or 0 when none was pushed. It is
		// apart from scan_seq, which --scan-differential numbers on its own. Kept when a blocked
		// submission goes back to the queue.
		uint64_t prescan_seq = 0;
	};

	// Starts the prescan when the options turn it on. Main thread, before the GPU thread starts.
	void              StartPrescan(const GuestGpuOptions& options);
	// The prescan's job for a submission, copied outside the order gate, or nothing when there is no
	// scanner or the submission has no job.
	[[nodiscard]] std::optional<PreparedScanJob> PreparePrescanJob(const Submission& submission);
	// The order gate's scan step. Pushes the prescan's job first, then runs --scan-differential's scan
	// or reset, and returns the differential's sequence number.
	uint64_t          ScanInGate(Submission& submission, std::optional<PreparedScanJob>& prescan_job);
	void              Enqueue(Submission submission);
	// Puts a submission on its queue and wakes the GPU thread.
	void              PushSubmission(Submission submission);
	void              ProcessCommands();
	bool              Process(Submission& submission);
	static void       ThreadRun(void* data);
	CommandProcessor& GetProcessor(uint32_t queue_id);
	// Reports a dispatch the command processor is about to run to --scan-differential. GPU thread only.
	void NoteCpDispatch(const HW::CsStageRegisters& cs, uint32_t initiator, bool async_queue);

	RenderContext&                                 m_renderer;
	Common::Mutex                                  m_queue_mutex;
	std::mutex                                     m_shutdown_mutex;
	Common::CondVar                                m_work_available;
	Common::CondVar                                m_idle;
	std::array<std::deque<Submission>, QueueCount> m_queues;
	std::deque<Common::UniqueFunction<void>>       m_commands;
	std::atomic_uint32_t                           m_pending_commands {0};
	uint32_t                                       m_next_queue        = 0;
	uint32_t                                       m_submission_count  = 0;
	bool                                           m_processing        = false;
	bool                                           m_accepting         = true;
	bool                                           m_stopping          = false;
	bool                                           m_shutdown_complete = false;
	// Completion callbacks can outlive GuestGpu during renderer shutdown.
	std::shared_ptr<std::binary_semaphore> m_suspend_point_ready =
	    std::make_shared<std::binary_semaphore>(1);

	std::unique_ptr<CommandProcessor>                                m_gfx_cp;
	std::array<std::unique_ptr<CommandProcessor>, ComputeQueueCount> m_compute_cp;

	std::atomic_int m_done_num = 0;
	// Suspend points the GPU thread has processed, for the first-run report. GPU thread only.
	int64_t m_gpu_flip = 0;
	// Null unless --scan-differential is on, and then every use is behind a null check, so a run without
	// the flag does nothing more for it.
	std::unique_ptr<ScanDifferential> m_scan_differential;
	// With --scan-differential or the prescan on, Enqueue runs the scan or shadow reset and the push to
	// the queue through this, so the scans run in the order the submissions queue. A SuspendPoint and
	// the command lists around it can come from different game threads, and a reset scanned out of order
	// would leave the shadow with the wrong registers. The lock order is the gate's queue lock, then
	// m_queue_mutex.
	ScanOrderGate m_scan_order;
	// Null unless the options turn the prescan on and this GPU attached the cache's only scanner, and
	// then every use is behind a null check. Made before the GPU thread starts and destroyed with this
	// object, so the GPU thread's NoteBegan never outlives it.
	std::unique_ptr<PrescanScanner> m_prescan_scanner;
	// Whether this GPU attached the scanner, so its Shutdown stops the cache's preparer after the
	// scanner and the GPU thread are joined.
	bool m_prescan_enabled_here = false;
	// The queue and scan sequence of the submission the GPU thread is processing, which the command
	// processor reports its dispatches under. GPU thread only. A sequence of 0 means none.
	uint32_t        m_scan_queue = 0;
	uint64_t        m_scan_seq   = 0;
	std::jthread    m_thread;

	friend class CommandProcessor;
	friend struct GuestGpuTestAccess;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
