#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCRIPTORSETLAYOUTCACHE_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCRIPTORSETLAYOUTCACHE_H_

// Descriptor set layouts that are shared and live as long as the cache. Two layouts with the same
// bindings and create flags are one handle, so pipelines, and later pipeline libraries, that agree
// on a set agree on its VkDescriptorSetLayout too. The descriptor heap keeps one batch of sets per
// layout handle, so one handle per distinct layout also keeps its batches few.

#include "common/assert.h"
#include "common/common.h"
#include "graphics/host_gpu/vulkanCommon.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <span>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

// What names a descriptor set layout. A binding is four words, and the key ends with the create
// flags. Bindings are sorted by number, so the order they were listed in does not split a key.
// Only the bindings and the create flags are keyed. A pNext chain on the create info, such as
// binding flags, is not supported, and neither are immutable samplers, which stop the process.
class DescriptorSetLayoutKey {
public:
	DescriptorSetLayoutKey(std::span<const vk::DescriptorSetLayoutBinding> bindings,
	                       vk::DescriptorSetLayoutCreateFlags              flags);

	bool                      operator==(const DescriptorSetLayoutKey& other) const = default;
	[[nodiscard]] std::size_t Hash() const;
	// The bindings in key order, as Vulkan takes them.
	[[nodiscard]] std::vector<vk::DescriptorSetLayoutBinding> Bindings() const;
	[[nodiscard]] vk::DescriptorSetLayoutCreateFlags          Flags() const;

private:
	// The words of one binding, in the order they are stored.
	enum BindingWord : std::size_t {
		WORD_BINDING,
		WORD_TYPE,
		WORD_COUNT,
		WORD_STAGES,
		WORDS_PER_BINDING
	};

	std::vector<uint32_t> m_words;
};

// Finds or makes descriptor set layouts under a mutex, since pipelines are built on worker threads.
// A layout is made while the lock is held. That is cheap, and it means two threads that ask for the
// same layout never both make it. The layouts are destroyed together when the cache is. Nothing is
// ever evicted, so the number of layouts the cache holds is also the number it has made.
class DescriptorSetLayoutCache {
public:
	// The create callback runs under the lock, so it must not call back into the cache.
	using CreateFunction =
	    std::function<vk::DescriptorSetLayout(const vk::DescriptorSetLayoutCreateInfo&)>;
	using DestroyFunction = std::function<void(vk::DescriptorSetLayout)>;

	// A cache that makes its layouts on `device`. The device must outlive the cache. It is defined
	// here so that the unit test, which has no Vulkan loader, can link the rest of the cache.
	explicit DescriptorSetLayoutCache(vk::Device device)
	    : DescriptorSetLayoutCache(
	          [device](const vk::DescriptorSetLayoutCreateInfo& info) {
		          vk::DescriptorSetLayout layout = nullptr;
		          EXIT_IF(device.createDescriptorSetLayout(&info, nullptr, &layout) !=
		                  vk::Result::eSuccess);
		          return layout;
	          },
	          [device](vk::DescriptorSetLayout layout) {
		          device.destroyDescriptorSetLayout(layout, nullptr);
	          }) {}
	// A cache that makes layouts through the given functions, which tests use to count them.
	DescriptorSetLayoutCache(CreateFunction create, DestroyFunction destroy);
	// Destroys every layout. Everything that uses a layout must be gone by then, which are the
	// pipeline layouts and the pipelines made with them.
	~DescriptorSetLayoutCache();
	KYTY_CLASS_NO_COPY(DescriptorSetLayoutCache);

	// Returns the layout for these bindings, which may be none, making it on the first request.
	// The handle stays valid until the cache is destroyed.
	[[nodiscard]] vk::DescriptorSetLayout
	GetOrCreate(std::span<const vk::DescriptorSetLayoutBinding> bindings,
	            vk::DescriptorSetLayoutCreateFlags              flags);

	// How many layouts the cache holds, which is also how many it has made.
	[[nodiscard]] std::size_t Size() const;

private:
	struct KeyHash {
		std::size_t operator()(const DescriptorSetLayoutKey& key) const { return key.Hash(); }
	};

	CreateFunction                                                               m_create;
	DestroyFunction                                                              m_destroy;
	mutable std::mutex                                                           m_mutex;
	std::unordered_map<DescriptorSetLayoutKey, vk::DescriptorSetLayout, KeyHash> m_layouts;
};

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_DESCRIPTORSETLAYOUTCACHE_H_
