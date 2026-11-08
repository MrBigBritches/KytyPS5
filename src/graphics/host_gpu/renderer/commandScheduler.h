#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_

#include "common/common.h"
#include "common/firstRunReport.h"
#include "common/uniqueFunction.h"
#include "graphics/host_gpu/renderer/drawRecord.h"
#include "graphics/host_gpu/renderer/dynamicState.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"
#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/submitCadence.h"

#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <thread>
#include <vector>

namespace Libs::Graphics {

class CommandScheduler {
public:
	/// Creates a scheduler. Only the render scheduler asks for the submit cadence. The presenter's
	/// scheduler submits at its own frame boundaries and must never split its command buffer, so
	/// every caller states the choice.
	CommandScheduler(RenderContext& context, GraphicContext& graphics, bool submit_cadence);
	~CommandScheduler();
	KYTY_CLASS_NO_COPY(CommandScheduler);

	void           Begin(HW::Context& registers, HW::UserConfig& user_config, HW::Shader& shaders);
	void           EndRendering() { m_command.EndRendering(); }
	void           Flush();
	void           Flush(SubmitInfo& submit);
	void           Finish();
	CommandBuffer& BeginCommand();
	uint64_t       Submit(SubmitInfo submit = {});
	// Deferred callbacks can observe an externally owned drain, but cannot initiate shutdown:
	// the priority runner cannot join itself.
	void                      Shutdown();
	void                      Wait(uint64_t tick);
	void                      PopPendingOperations();
	void                      DrainPriorityOperations();
	void                      WaitPriorityOperations(uint64_t tick);
	// Guest-memory completions use the priority queue; normal callbacks maintain GPU resources.
	void                      DeferOperation(Common::UniqueFunction<void>&& operation);
	void                      DeferPriorityOperation(Common::UniqueFunction<void>&& operation);
	[[nodiscard]] bool        HasPendingPriorityOperations();
	/// Ends the current command buffer, reserves an empty buffer for `hole` after it and starts a new
	/// current buffer, all in one call so a split can never be half done. Nothing is submitted, the
	/// tick does not change and the submit cadence keeps counting. The new buffer starts with none of
	/// the dynamic state set, so the scheduler's copy is invalidated. The hole is recorded into its
	/// buffer by the next Submit, which then sends the segments and holes in order. GPU thread or
	/// shutdown only, and never from inside Submit.
	void SplitForHole(std::unique_ptr<DrawHole> hole);
	/// What forced a resolve of deferred draws, for the first-run report. Submit is the default, for
	/// every caller that does not say more.
	enum class ResolveTrigger : uint8_t {
		Submit,
		Fault,
		StreamWrap,
		Cap,
		Shutdown,
		Readback,
		Label,
		Flip,
		Unmap,
		SliceEnd
	};
	/// The name the report gives a trigger.
	[[nodiscard]] static const char* ResolveTriggerName(ResolveTrigger trigger) noexcept;
	/// Names the trigger of any resolve that happens while it lives. Put one around a Wait or Flush
	/// whose caller knows why it submits.
	class TriggerScope {
	public:
		TriggerScope(CommandScheduler& scheduler, ResolveTrigger trigger)
		    : m_scheduler(scheduler), m_previous(scheduler.m_trigger) {
			scheduler.m_trigger = trigger;
		}
		~TriggerScope() { m_scheduler.m_trigger = m_previous; }
		KYTY_CLASS_NO_COPY(TriggerScope);

	private:
		CommandScheduler& m_scheduler;
		ResolveTrigger    m_previous;
	};
	/// Whether the calling thread may split now. That is the GPU thread, any thread once shutdown
	/// began, and never from inside a submit or a resolve. A draw on another thread blocks instead.
	[[nodiscard]] bool CanSplitHere() const { return OnHoleThread() && !m_in_submit && !m_in_resolve; }
	/// The most holes that may wait for one submission. The command buffer pool never shrinks, and a
	/// buffer is reserved for each hole, so a long run of misses must not grow it without bound.
	static constexpr size_t MAX_PENDING_HOLES = 256;
	/// Whether so many holes wait that another draw must block instead. Only a question.
	[[nodiscard]] bool HoleCapReached() const noexcept { return m_pending_holes >= MAX_PENDING_HOLES; }
	/// Asks for a flush at the next packet boundary, since a handler must not submit. A draw that
	/// found the cap reached calls this. The request stays until a resolve has run.
	void RequestHoleFlush() noexcept { m_hole_flush_requested = true; }
	/// Whether a draw found the cap reached and the holes have not been resolved since.
	[[nodiscard]] bool HoleFlushRequested() const noexcept {
		return m_hole_flush_requested && m_pending_holes != 0;
	}
	/// Submits, and so resolves, if the cap asked for it. Called at a packet boundary, where a flush
	/// is allowed. The report names the cap as the trigger. Returns whether it submitted.
	bool FlushIfHoleCapReached();
	/// Resolves the holes that are still pending when the context is destroyed, on the destroying
	/// thread and before the builds they wait for are stopped, and waits for the GPU to finish them.
	/// Does nothing when no hole waits.
	void ResolveHolesForShutdown();
	/// Holes split off and not yet recorded. Every Flush, Finish and Wait leaves none.
	[[nodiscard]] size_t PendingHoleCount() const noexcept { return m_pending_holes; }
	/// Tells the submit cadence that a draw or dispatch went into the current command buffer.
	void NoteRecordedWork() { m_cadence.NoteWork(); }
	/// Returns whether the submit cadence is on, which makes scheduler ticks much faster than
	/// frames.
	[[nodiscard]] bool SubmitCadenceEnabled() const noexcept { return m_cadence.Enabled(); }
	/// Returns whether the PM4 loop should submit the current command buffer now. While nothing is
	/// due the inline gate reads at most the clock, once per boundary stride, so the GPU progress
	/// is read only after the time gate passed.
	[[nodiscard]] bool TakeCadenceSubmit() {
		return m_cadence.DueAtBoundary() && TakeDueCadenceSubmit();
	}
	[[nodiscard]] static bool InDeferredOperation() noexcept;

	[[nodiscard]] bool Active() const noexcept { return m_command.m_registers != nullptr; }
	void                           CheckActive() const;
	CommandBuffer&                 Current();
	[[nodiscard]] uint64_t         CurrentTick() const noexcept { return m_master.CurrentTick(); }
	[[nodiscard]] bool             IsFree(uint64_t tick);
	[[nodiscard]] MasterSemaphore& GetMasterSemaphore() noexcept { return m_master; }
	[[nodiscard]] DynamicState&    GetDynamicState() noexcept { return m_dynamic_state; }
	[[nodiscard]] RenderContext&   Context() const noexcept { return m_context; }
	[[nodiscard]] GraphicContext&  Graphics() const noexcept { return m_graphics; }

private:
	class CommandPool {
	public:
		CommandPool(GraphicContext& graphics, MasterSemaphore& master);
		~CommandPool();
		KYTY_CLASS_NO_COPY(CommandPool);

		vk::CommandBuffer Commit();

	private:
		static constexpr size_t GrowStep = 4;

		size_t Grow();

		GraphicContext&                m_graphics;
		MasterSemaphore&               m_master;
		vk::CommandPool                m_pool = nullptr;
		std::vector<vk::CommandBuffer> m_buffers;
		std::vector<uint64_t>          m_ticks;
		size_t                         m_hint = 0;
	};

	enum class OperationState { Open, Draining, Closed };

	struct PendingOperation {
		Common::UniqueFunction<void> callback;
		uint64_t                     tick = 0;
	};

	friend struct CommandSchedulerTestAccess;

	/// Refreshes the GPU progress and asks the policy whether a due submit may go ahead, which it
	/// refuses while the GPU lags. Also feeds the held back counter.
	[[nodiscard]] bool TakeDueCadenceSubmit();

	/// Returns how many submitted command buffers the GPU has not finished. It saturates at zero,
	/// since the known GPU tick can pass the last submitted tick between two reads.
	[[nodiscard]] static uint64_t GpuLag(uint64_t current_tick, uint64_t known_gpu_tick) noexcept;

	/// One piece of the next submission, in submit order. A segment holds the commands recorded
	/// between two splits. A hole holds a reserved buffer and the draw to record into it.
	struct HeldBuffer {
		vk::CommandBuffer         buffer       = nullptr;
		bool                      has_commands = false;
		std::unique_ptr<DrawHole> hole;
		bool                      recorded = false;
	};

	/// Takes a pool buffer, begins it as the current one and marks the dynamic state as unset in
	/// it. It leaves the submit cadence alone.
	void BeginBuffer();
	/// Builds every pending hole's buffer, waiting for the pipelines it needs, and adds a row to the
	/// first-run report. It touches only the scheduler and the pipeline cache, because callers reach
	/// Submit from inside the caches.
	void ResolveHoles();
	/// Waits for the pipelines of every unrecorded hole, in hole order, filling the wait columns of
	/// the report row. This is the only part of a resolve that holds guest audio.
	[[nodiscard]] std::vector<PipelineCache::Pipeline>
	AcquireHolePipelines(Common::FirstRunReport::DeferredResolve& row);
	/// Records every unrecorded hole into its buffer against the pipeline at the same place in `built`.
	void RecordHoles(std::span<const PipelineCache::Pipeline> built);
	/// Whether the calling thread may split or resolve. That is the GPU thread, any thread once
	/// shutdown began, and a thread a test named.
	[[nodiscard]] bool OnHoleThread() const;
	/// Sets a flag for its lifetime and stops if it was already set, which means Submit or
	/// ResolveHoles was entered again from inside itself.
	class ReentryGuard {
	public:
		explicit ReentryGuard(bool& flag): m_flag(flag) {
			EXIT_IF(m_flag);
			m_flag = true;
		}
		~ReentryGuard() { m_flag = false; }
		KYTY_CLASS_NO_COPY(ReentryGuard);

	private:
		bool& m_flag;
	};

	void BeginNext();
	void PriorityOperationsThread(std::stop_token stop);
	void QueueOperation(Common::UniqueFunction<void>&& operation, bool priority);
	void RunOperation(Common::UniqueFunction<void>&& operation);

	MasterSemaphore              m_master;
	RenderContext&               m_context;
	GraphicContext&              m_graphics;
	CommandPool                  m_command_pool;
	CommandBuffer                m_command;
	DynamicState                 m_dynamic_state;
	std::queue<PendingOperation> m_pending_operations;
	std::queue<PendingOperation> m_priority_operations;
	mutable std::mutex           m_operation_mutex;
	std::condition_variable      m_operation_available;
	std::jthread                 m_priority_thread;
	uint64_t                     m_priority_active_tick = 0; // 0 while none runs
	OperationState               m_operation_state      = OperationState::Open;
	SubmitCadence                m_cadence;
	std::vector<HeldBuffer>        m_held;
	std::vector<vk::CommandBuffer> m_submit_buffers;
	DescriptorScratch              m_hole_scratch;
	size_t                         m_pending_holes = 0;
	bool                           m_in_submit     = false;
	bool                           m_in_resolve    = false;
	std::thread::id                m_extra_hole_thread;
	// Set by tests so a resolve on their thread holds guest audio as the GPU thread's would.
	bool                           m_hold_audio_extra_thread = false;
	// Set when a draw found the cap reached, so the next packet boundary flushes. Cleared by the resolve
	// that follows, which is also what makes a flush with no request do nothing.
	bool                           m_hole_flush_requested    = false;
	// When the first pending hole was split in, for the report's time before the resolve.
	std::optional<std::chrono::steady_clock::time_point> m_first_hole_time;
	ResolveTrigger                 m_trigger = ResolveTrigger::Submit;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_COMMANDSCHEDULER_H_
