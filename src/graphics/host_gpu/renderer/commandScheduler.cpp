#include "graphics/host_gpu/renderer/commandScheduler.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/firstRunReport.h"
#include "common/gpuStall.h"
#include "common/logging/log.h"
#include "graphics/guest_gpu/graphicsRun.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/renderContext.h"

#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <optional>

namespace Libs::Graphics {

static thread_local CommandScheduler* g_deferred_callback_scheduler = nullptr;

namespace {

constexpr double NS_PER_MS = 1.0e6;

double ElapsedMs(std::chrono::steady_clock::time_point from,
                 std::chrono::steady_clock::time_point to) {
	return std::chrono::duration<double, std::milli>(to - from).count();
}

// Orders the attachment writes of the buffer before it against the attachment reads and writes of
// the buffer after it. Draws on one attachment in separate render passes get no other barrier,
// because the layout transitions the CPU tracks add none between two uses of the same layout.
void RecordHoleBarrier(vk::CommandBuffer command) {
	constexpr auto attachment_stages = vk::PipelineStageFlagBits2::eColorAttachmentOutput |
	                                   vk::PipelineStageFlagBits2::eEarlyFragmentTests |
	                                   vk::PipelineStageFlagBits2::eLateFragmentTests;
	vk::MemoryBarrier2 barrier {};
	barrier.srcStageMask  = attachment_stages;
	barrier.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite |
	                        vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
	barrier.dstStageMask  = attachment_stages;
	barrier.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentRead |
	                        vk::AccessFlagBits2::eColorAttachmentWrite |
	                        vk::AccessFlagBits2::eDepthStencilAttachmentRead |
	                        vk::AccessFlagBits2::eDepthStencilAttachmentWrite;
	vk::DependencyInfo dependency {};
	dependency.memoryBarrierCount = 1;
	dependency.pMemoryBarriers    = &barrier;
	command.pipelineBarrier2(dependency);
}

// Records a hole's draw into a buffer of its own. The rendering is a pass of its own that loads the
// attachments, since any clear was applied where the draw was met and its view says so. Holes never touch the occlusion
// queries, whose open state is the live one and not the draw's.
class HoleDrawEnv final: public DrawRecordEnv {
public:
	HoleDrawEnv(DescriptorScratch& scratch, vk::PipelineStageFlags shader_write_stages)
	    : m_scratch(scratch), m_shader_write_stages(shader_write_stages) {}

	DescriptorScratch& Scratch() override { return m_scratch; }

	void BeginRendering(vk::CommandBuffer command, const RenderState& rendering) override {
		if (m_shader_write_stages) {
			ShaderWriteHazardBarrier(command, m_shader_write_stages);
		}
		// A clear belongs to the pass recorded where the draw was met. Dropping it here would lose it.
		EXIT_IF(RenderStateClears(rendering));
		RecordBeginRendering(command, rendering);
	}

	int32_t BeginDraw(vk::CommandBuffer) override { return -1; }

	void EndDraw(vk::CommandBuffer command, int32_t, const DrawCallInfo& draw) override {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::Draws);
		Common::DebugCounters::Add(Common::DebugCounters::Counter::DrawInstances,
		                           draw.instance_count);
		RecordEndRendering(command);
		if (m_shader_write_stages) {
			ShaderWriteBarrier(command, m_shader_write_stages);
		}
	}

private:
	DescriptorScratch&     m_scratch;
	vk::PipelineStageFlags m_shader_write_stages;
};

} // namespace

CommandScheduler::CommandPool::CommandPool(GraphicContext& graphics, MasterSemaphore& master)
    : m_graphics(graphics), m_master(master) {
	EXIT_IF(graphics.queue_family == static_cast<uint32_t>(-1));
	vk::CommandPoolCreateInfo create {};
	create.queueFamilyIndex = graphics.queue_family;
	create.flags            = vk::CommandPoolCreateFlagBits::eTransient |
	                          vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
	const auto result       = graphics.device.createCommandPool(&create, nullptr, &m_pool);
	EXIT_NOT_IMPLEMENTED(result != vk::Result::eSuccess || m_pool == nullptr);
}

CommandScheduler::CommandPool::~CommandPool() {
	m_graphics.device.destroyCommandPool(m_pool, nullptr);
}

size_t CommandScheduler::CommandPool::Grow() {
	const auto first = m_ticks.size();
	m_ticks.resize(first + GrowStep);
	m_buffers.resize(first + GrowStep);

	vk::CommandBufferAllocateInfo allocate {};
	allocate.commandPool        = m_pool;
	allocate.level              = vk::CommandBufferLevel::ePrimary;
	allocate.commandBufferCount = static_cast<uint32_t>(GrowStep);
	EXIT_IF(m_graphics.device.allocateCommandBuffers(&allocate, m_buffers.data() + first) !=
	        vk::Result::eSuccess);
	return first;
}

vk::CommandBuffer CommandScheduler::CommandPool::Commit() {
	auto       gpu_tick = m_master.KnownGpuTick();
	const auto search   = [this, &gpu_tick](size_t begin, size_t end) -> std::optional<size_t> {
		for (size_t index = begin; index < end; ++index) {
			if (gpu_tick >= m_ticks[index]) {
				m_ticks[index] = m_master.CurrentTick();
				return index;
			}
		}
		return std::nullopt;
	};

	auto found = search(m_hint, m_ticks.size());
	if (!found) {
		m_master.Refresh();
		gpu_tick = m_master.KnownGpuTick();
		found    = search(m_hint, m_ticks.size());
	}
	if (!found) {
		found = search(0, m_hint);
	}
	if (!found) {
		found           = Grow();
		m_ticks[*found] = m_master.CurrentTick();
	}

	m_hint = (*found + 1) % m_ticks.size();
	return m_buffers[*found];
}

bool CommandScheduler::InDeferredOperation() noexcept {
	return g_deferred_callback_scheduler != nullptr;
}

CommandScheduler::CommandScheduler(RenderContext& context, GraphicContext& graphics,
                                   bool submit_cadence)
    : m_master(graphics), m_context(context), m_graphics(graphics),
      m_command_pool(graphics, m_master), m_command(*this),
      m_priority_thread([this](std::stop_token stop) { PriorityOperationsThread(stop); }),
      m_cadence(submit_cadence) {}

CommandScheduler::~CommandScheduler() {
	Shutdown();
}

void CommandScheduler::Shutdown() {
	{
		std::unique_lock lock(m_operation_mutex);
		if (m_operation_state == OperationState::Closed) {
			return;
		}
		if (g_deferred_callback_scheduler == this) {
			EXIT_IF(m_operation_state == OperationState::Open);
			// A priority callback cannot join its own runner, while a normal callback can be
			// executing inside the shutdown owner's final PopPendingOperations. The owning
			// thread will finish shutdown after this callback returns.
			return;
		}
		if (m_operation_state == OperationState::Draining) {
			m_operation_available.wait(
			    lock, [this] { return m_operation_state == OperationState::Closed; });
			return;
		}
		m_operation_state = OperationState::Draining;
	}
	if (!m_command.IsInvalid()) {
		const TriggerScope trigger(*this, ResolveTrigger::Shutdown);
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	PopPendingOperations();
	DrainPriorityOperations();
	m_priority_thread.request_stop();
	m_operation_available.notify_all();
	if (m_priority_thread.joinable()) {
		m_priority_thread.join();
	}
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(!m_pending_operations.empty() || !m_priority_operations.empty() ||
		        m_priority_active_tick != 0);
		m_operation_state = OperationState::Closed;
	}
	m_operation_available.notify_all();
}

void CommandScheduler::Begin(HW::Context& registers, HW::UserConfig& user_config,
                             HW::Shader& shaders) {
	{
		std::lock_guard lock(m_operation_mutex);
		EXIT_IF(m_operation_state != OperationState::Open);
	}
	m_command.Bind(registers, user_config, shaders);

	if (m_command.IsInvalid()) {
		BeginNext();
	}
}

void CommandScheduler::Flush() {
	SubmitInfo submit;
	Flush(submit);
}

void CommandScheduler::Flush(SubmitInfo& submit) {
	Submit(submit);
	BeginNext();
}

void CommandScheduler::Finish() {
	CheckActive();
	if (!m_command.IsInvalid()) {
		Submit();
	}
	m_master.Wait(CurrentTick() - 1);
	BeginNext();
	PopPendingOperations();
}

void CommandScheduler::Wait(uint64_t tick) {
	EXIT_IF(tick > CurrentTick());
	if (tick == CurrentTick()) {
		CheckActive();
		// A stream-buffer wrap can wait while a draw is being prepared through a reference to
		// Current(). The wrapper stays stable while its pooled Vulkan buffer is retired. Deferred
		// resources are released only at the next GPU operation boundary.
		const auto submitted_tick = Submit();
		EXIT_IF(submitted_tick != tick);
		m_master.Wait(tick);
		BeginNext();
	} else {
		m_master.Wait(tick);
	}
}

void CommandScheduler::PopPendingOperations() {
	m_master.Refresh();
	for (;;) {
		PendingOperation operation;
		{
			std::lock_guard lock(m_operation_mutex);
			if (m_pending_operations.empty() ||
			    !m_master.IsFree(m_pending_operations.front().tick)) {
				return;
			}
			operation = std::move(m_pending_operations.front());
			m_pending_operations.pop();
		}
		WaitPriorityOperations(operation.tick);
		RunOperation(std::move(operation.callback));
	}
}

void CommandScheduler::DeferOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), false);
}

void CommandScheduler::DeferPriorityOperation(Common::UniqueFunction<void>&& operation) {
	QueueOperation(std::move(operation), true);
}

void CommandScheduler::QueueOperation(Common::UniqueFunction<void>&& operation, bool priority) {
	CheckActive();
	EXIT_IF(!operation);
	std::unique_lock lock(m_operation_mutex);
	if (m_operation_state == OperationState::Open) {
		auto& queue = priority ? m_priority_operations : m_pending_operations;
		queue.push({std::move(operation), CurrentTick()});
		lock.unlock();
		if (priority) {
			m_operation_available.notify_one();
		}
		return;
	}
	if (g_deferred_callback_scheduler != this) {
		m_operation_available.wait(lock,
		                           [this] { return m_operation_state == OperationState::Closed; });
	}
	lock.unlock();
	operation();
}

bool CommandScheduler::HasPendingPriorityOperations() {
	std::lock_guard lock(m_operation_mutex);
	return !m_priority_operations.empty() || m_priority_active_tick != 0;
}

void CommandScheduler::PriorityOperationsThread(std::stop_token stop) {
	while (!stop.stop_requested()) {
		PendingOperation operation;
		{
			std::unique_lock lock(m_operation_mutex);
			m_operation_available.wait(lock, [this, &stop] {
				return stop.stop_requested() || !m_priority_operations.empty();
			});
			if (stop.stop_requested()) {
				return;
			}
			operation = std::move(m_priority_operations.front());
			m_priority_operations.pop();
			m_priority_active_tick = operation.tick;
		}
		m_master.Wait(operation.tick);
		if (!stop.stop_requested()) {
			RunOperation(std::move(operation.callback));
		}
		{
			std::lock_guard lock(m_operation_mutex);
			m_priority_active_tick = 0;
		}
		m_operation_available.notify_all();
	}
}

void CommandScheduler::DrainPriorityOperations() {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(
	    lock, [this] { return m_priority_operations.empty() && m_priority_active_tick == 0; });
}

void CommandScheduler::WaitPriorityOperations(uint64_t tick) {
	EXIT_IF(g_deferred_callback_scheduler == this);
	std::unique_lock lock(m_operation_mutex);
	m_operation_available.wait(lock, [this, tick] {
		const bool active_before_or_at =
		    m_priority_active_tick != 0 && m_priority_active_tick <= tick;
		const bool queued_before_or_at =
		    !m_priority_operations.empty() && m_priority_operations.front().tick <= tick;
		return !active_before_or_at && !queued_before_or_at;
	});
}

void CommandScheduler::RunOperation(Common::UniqueFunction<void>&& operation) {
	auto* previous                = g_deferred_callback_scheduler;
	g_deferred_callback_scheduler = this;
	operation();
	g_deferred_callback_scheduler = previous;
}

bool CommandScheduler::IsFree(uint64_t tick) {
	if (m_master.IsFree(tick)) {
		return true;
	}
	m_master.Refresh();
	return m_master.IsFree(tick);
}

void CommandScheduler::CheckActive() const {
	EXIT_IF(!Active());
}

CommandBuffer& CommandScheduler::Current() {
	EXIT_IF(m_command.IsInvalid());
	return m_command;
}

void CommandScheduler::BeginBuffer() {
	m_command.m_buffer = m_command_pool.Commit();
	m_command.Begin();
	m_command.m_used = false;
	m_dynamic_state.Invalidate();
}

CommandBuffer& CommandScheduler::BeginCommand() {
	EXIT_IF(!m_command.IsInvalid());
	EXIT_IF(!m_held.empty());
	BeginBuffer();
	// Any submit starts a new command buffer, so the cadence counts again from its first draw.
	m_cadence.Reset();
	return m_command;
}

bool CommandScheduler::OnHoleThread() const {
	if (GuestGpu::IsGpuThread() || std::this_thread::get_id() == m_extra_hole_thread) {
		return true;
	}
	std::lock_guard lock(m_operation_mutex);
	return m_operation_state != OperationState::Open;
}

void CommandScheduler::SplitForHole(std::unique_ptr<DrawHole> hole) {
	EXIT_IF(hole == nullptr || m_command.IsInvalid());
	EXIT_IF(m_in_submit || m_in_resolve);
	EXIT_IF(!OnHoleThread());
	// End closes an open rendering, which a segment must not leave open. It marks the buffer as used,
	// so the answer is read first.
	const bool segment_used = m_command.m_used;
	m_command.End();
	m_held.push_back({m_command.m_buffer, segment_used, nullptr, false});
	m_held.push_back({m_command_pool.Commit(), true, std::move(hole), false});
	if (m_pending_holes == 0) {
		m_first_hole_time = std::chrono::steady_clock::now();
		m_first_hole_holds = Common::GpuStall::HoldCount();
		m_first_hole_held_ns = Common::GpuStall::HeldNs();
		// The resolve waits for compiles, which the warm-up would compete with for cores.
		m_context.GetPipelineCache().PauseWarmBuilds();
	}
	m_pending_holes++;
	// The cadence is not reset here. A split is not a submit, and the draws on both sides of it
	// belong to the same tick.
	BeginBuffer();
}

const char* CommandScheduler::ResolveTriggerName(ResolveTrigger trigger) noexcept {
	switch (trigger) {
		case ResolveTrigger::Submit: return "submit";
		case ResolveTrigger::Fault: return "fault";
		case ResolveTrigger::StreamWrap: return "stream_wrap";
		case ResolveTrigger::Cap: return "cap";
		case ResolveTrigger::Shutdown: return "shutdown";
		case ResolveTrigger::Readback: return "readback";
		case ResolveTrigger::Label: return "label";
		case ResolveTrigger::Flip: return "flip";
		case ResolveTrigger::Unmap: return "unmap";
		case ResolveTrigger::SliceEnd: return "slice_end";
	}
	return "submit";
}

std::vector<PipelineCache::Pipeline>
CommandScheduler::AcquireHolePipelines(Common::FirstRunReport::DeferredResolve& row) {
	std::vector<GraphicsPipelineKey> keys;
	keys.reserve(m_pending_holes);
	for (const auto& held: m_held) {
		if (held.hole != nullptr && !held.recorded) {
			keys.push_back(held.hole->Key());
		}
	}
	// A resolve from the shutdown path or another thread must never hold guest audio, since only the
	// GPU thread's waits stop the frames that audio is paced against.
	PipelineCache::DeferredAcquireStats stats;
	auto built = m_context.GetPipelineCache().AcquireDeferredPipelines(
	    keys, GuestGpu::IsGpuThread() || m_hold_audio_extra_thread, stats);
	row.distinct_keys   = stats.distinct_keys;
	row.resolve_wait_ms = stats.wait_ms;
	row.save_ms         = stats.save_ms;
	// The image heap's set is bound in the segment's buffer, which the hole's buffer lacks.
	EXIT_IF(std::ranges::any_of(built, [](const auto& pipeline) { return pipeline.uses_image_heap; }));
	return built;
}

void CommandScheduler::RecordHoles(std::span<const PipelineCache::Pipeline> built) {
	auto&  executor = m_context.GetRenderExecutor();
	size_t next     = 0;
	for (auto& held: m_held) {
		if (held.hole == nullptr || held.recorded) {
			continue;
		}
		vk::CommandBufferBeginInfo begin {};
		begin.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
		EXIT_NOT_IMPLEMENTED(held.buffer.begin(&begin) != vk::Result::eSuccess);
		RecordHoleBarrier(held.buffer);
		HoleDrawEnv env(m_hole_scratch, held.hole->View().shader_write_stages);
		executor.RecordDraw(held.buffer, held.hole->View(), built[next++], env);
		// Orders the hole's writes before the segment that follows it.
		RecordHoleBarrier(held.buffer);
		EXIT_NOT_IMPLEMENTED(held.buffer.end() != vk::Result::eSuccess);
		held.recorded = true;
	}
}

bool CommandScheduler::FlushIfHoleCapReached() {
	if (!HoleFlushRequested()) {
		return false;
	}
	const TriggerScope trigger(*this, ResolveTrigger::Cap);
	Flush();
	return true;
}

void CommandScheduler::ResolveHolesForShutdown() {
	if (m_pending_holes == 0) {
		return;
	}
	// A hole needs a scheduler that is running to be recorded into.
	EXIT_IF(!Active() || m_command.IsInvalid());
	// The destroying thread is not the GPU thread, which has stopped, and must be allowed to resolve.
	const auto previous = m_extra_hole_thread;
	m_extra_hole_thread = std::this_thread::get_id();
	{
		const TriggerScope trigger(*this, ResolveTrigger::Shutdown);
		Finish();
	}
	m_extra_hole_thread = previous;
	EXIT_IF(m_pending_holes != 0);
}

void CommandScheduler::ResolveHoles() {
	if (m_pending_holes == 0) {
		return;
	}
	EXIT_IF(!OnHoleThread() || !m_first_hole_time.has_value());
	const ReentryGuard guard(m_in_resolve);
	using Clock             = std::chrono::steady_clock;
	const auto started      = Clock::now();
	const auto holds_before = Common::GpuStall::HoldCount();
	const auto held_before  = Common::GpuStall::HeldNs();

	Common::FirstRunReport::DeferredResolve row;
	row.trigger        = ResolveTriggerName(m_trigger);
	row.holes          = static_cast<uint32_t>(m_pending_holes);
	row.pre_resolve_ms = ElapsedMs(*m_first_hole_time, started);
	// The wait is the only part under the audio hold. Recording below runs after the hold ends.
	const auto built       = AcquireHolePipelines(row);
	const auto record_from = Clock::now();
	RecordHoles(built);
	row.record_ms         = ElapsedMs(record_from, Clock::now());
	row.hold_count_change = Common::GpuStall::HoldCount() - holds_before;
	row.held_ms           = static_cast<double>(Common::GpuStall::HeldNs() - held_before) / NS_PER_MS;

	m_pending_holes        = 0;
	m_hole_flush_requested = false;
	m_context.GetPipelineCache().ResumeWarmBuilds();
	// From the first hole to here, so a hold that opened while the draws were still being met counts.
	row.window_hold_count_change = Common::GpuStall::HoldCount() - m_first_hole_holds;
	row.window_held_ms =
	    static_cast<double>(Common::GpuStall::HeldNs() - m_first_hole_held_ns) / NS_PER_MS;
	m_first_hole_time.reset();
	Common::FirstRunReport::AddDeferredResolve(row);
	// Log file only, since a cold burst resolves often and would flood the console.
	Log::Write(Log::Color::Default, fmt::format(
	    "Deferred draws resolved ({}): {} holes, {} keys, wait {:.1f} ms, record {:.1f} ms, "
	    "before resolve {:.1f} ms, save {:.1f} ms, holds {} in the resolve and {} from the first hole, "
	    "audio held {:.1f} ms in the resolve and {:.1f} ms from the first hole\n",
	    row.trigger, row.holes, row.distinct_keys, row.resolve_wait_ms, row.record_ms,
	    row.pre_resolve_ms, row.save_ms, row.hold_count_change, row.window_hold_count_change,
	    row.held_ms, row.window_held_ms));
}

uint64_t CommandScheduler::GpuLag(uint64_t current_tick, uint64_t known_gpu_tick) noexcept {
	// The next tick to be issued is current_tick, so the last submitted one is current_tick - 1.
	return current_tick > known_gpu_tick + 1 ? current_tick - 1 - known_gpu_tick : 0;
}

bool CommandScheduler::TakeDueCadenceSubmit() {
	// The known GPU tick only advances on a refresh, so a stale one would report a lag that is gone.
	m_master.Refresh();
	const bool submit = m_cadence.ShouldSubmit(GpuLag(CurrentTick(), m_master.KnownGpuTick()));
	// ShouldSubmit is the only place a hold is counted, so the count is complete here.
	if (const auto held_back = m_cadence.TakeHeldBack(); held_back != 0) {
		Common::DebugCounters::Add(Common::DebugCounters::Counter::CadenceHeldBack, held_back);
	}
	return submit;
}

uint64_t CommandScheduler::Submit(SubmitInfo submit) {
	EXIT_IF(m_command.IsInvalid());
	const ReentryGuard guard(m_in_submit);

	ResolveHoles();
	const bool last_used = m_command.m_used;
	m_command.End();
	// Segments and holes go in order. A segment nothing was recorded into is left out, unless it is
	// the only buffer, which a submit has always sent.
	m_submit_buffers.clear();
	for (const auto& held: m_held) {
		if (held.hole != nullptr || held.has_commands) {
			m_submit_buffers.push_back(held.buffer);
		}
	}
	if (last_used || m_held.empty()) {
		m_submit_buffers.push_back(m_command.m_buffer);
	}
	auto& graphics = m_graphics;
	EXIT_IF(graphics.queue == nullptr);

	vk::Result result;
	uint64_t   tick;
	{
		Common::LockGuard lock(graphics.queue_mutex);
		tick = m_master.NextTick();
		submit.AddSignal(m_master.Handle(), tick);

		vk::TimelineSemaphoreSubmitInfo timeline_info {};
		timeline_info.waitSemaphoreValueCount   = submit.num_wait_semaphores;
		timeline_info.pWaitSemaphoreValues      = submit.wait_ticks.data();
		timeline_info.signalSemaphoreValueCount = submit.num_signal_semaphores;
		timeline_info.pSignalSemaphoreValues    = submit.signal_ticks.data();

		vk::SubmitInfo submit_info {};
		submit_info.pNext                = &timeline_info;
		submit_info.waitSemaphoreCount   = submit.num_wait_semaphores;
		submit_info.pWaitSemaphores      = submit.wait_semaphores.data();
		submit_info.pWaitDstStageMask    = submit.wait_stages.data();
		submit_info.commandBufferCount   = static_cast<uint32_t>(m_submit_buffers.size());
		submit_info.pCommandBuffers      = m_submit_buffers.data();
		submit_info.signalSemaphoreCount = submit.num_signal_semaphores;
		submit_info.pSignalSemaphores    = submit.signal_semaphores.data();

		result = graphics.queue.submit(1, &submit_info, nullptr);
	}
	Common::DebugCounters::Add(Common::DebugCounters::Counter::QueueSubmits);

	if (result != vk::Result::eSuccess) {
		EXIT("vkQueueSubmit failed: %s, tick=%" PRIu64 "\n", vk::to_string(result).c_str(), tick);
	}

	m_command.m_buffer = nullptr;
	m_held.clear();
	return tick;
}

void CommandScheduler::BeginNext() {
	CheckActive();
	BeginCommand();
}

} // namespace Libs::Graphics
