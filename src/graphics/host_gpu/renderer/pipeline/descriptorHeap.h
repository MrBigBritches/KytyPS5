#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_

#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <array>
#include <deque>
#include <unordered_map>

namespace Libs::Graphics {

struct GraphicContext;
class MasterSemaphore;

class DescriptorHeap {
public:
	DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore);
	~DescriptorHeap();
	KYTY_CLASS_NO_COPY(DescriptorHeap);

	/// Hands out a set of the layout from the current pool. A full pool is retired, stamped with the
	/// current tick, and reset for reuse only once the GPU has finished that tick.
	[[nodiscard]] vk::DescriptorSet Commit(vk::DescriptorSetLayout layout);

	/// The sets one pool holds. Tests fill a pool with it, to make a chosen Commit retire it.
	static constexpr uint32_t POOL_SETS = 1024;

private:
	friend struct DescriptorHeapTestAccess;

	static constexpr uint32_t DescriptorSetBatch = 32;

	struct Batch {
		std::array<vk::DescriptorSet, DescriptorSetBatch> sets {};
		uint32_t                                          size       = 0;
		uint32_t                                          allocation = DescriptorSetBatch;
	};

	[[nodiscard]] bool Allocate(vk::DescriptorSetLayout layout, Batch& batch);
	void               CreateDescriptorPool();

	GraphicContext&                                     m_graphics;
	MasterSemaphore&                                    m_master_semaphore;
	vk::DescriptorPool                                  m_current_pool = nullptr;
	std::deque<std::pair<vk::DescriptorPool, uint64_t>> m_pending_pools;
	// A layout key must outlive every later Commit. That only matters if layouts are ever evicted.
	std::unordered_map<vk::DescriptorSetLayout, Batch>  m_sets;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORHEAP_H_
