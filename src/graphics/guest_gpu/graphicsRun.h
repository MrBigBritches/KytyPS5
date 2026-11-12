#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_

#include "common/abi.h"
#include "common/common.h"
#include "common/firstRunReport.h"
#include "common/threads.h"
#include "common/uniqueFunction.h"
#include "graphics/guest_gpu/command_processor/commandProcessor.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <semaphore>
#include <span>
#include <thread>

namespace Libs::Graphics {

class RenderContext;
class ScanDifferential;

class GuestGpu final {
public:
	explicit GuestGpu(RenderContext& renderer);
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

private:
	static constexpr uint32_t ComputePipeCount     = 7;
	static constexpr uint32_t QueuesPerComputePipe = 8;
	static constexpr uint32_t ComputeQueueCount    = ComputePipeCount * QueuesPerComputePipe;
	static constexpr uint32_t ComputeQueueBase     = 0x20;
	static constexpr uint32_t QueueCount           = 1 + ComputeQueueCount;

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
	};

	void              Enqueue(Submission submission);
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
	// With the flag on, held in Enqueue from before the scan or shadow reset until after the push to the
	// queue, so the scans run in the order the submissions queue. A SuspendPoint and the command lists
	// around it can come from different game threads, and a reset scanned out of order would leave the
	// shadow with the wrong registers. The lock order is this mutex, then m_queue_mutex.
	std::mutex m_scan_order_mutex;
	// The queue and scan sequence of the submission the GPU thread is processing, which the command
	// processor reports its dispatches under. GPU thread only. A sequence of 0 means none.
	uint32_t        m_scan_queue = 0;
	uint64_t        m_scan_seq   = 0;
	std::jthread    m_thread;

	friend class CommandProcessor;
};
} // namespace Libs::Graphics

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_GRAPHICSRUN_H_ */
