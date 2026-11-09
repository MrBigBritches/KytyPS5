#include "graphics/host_gpu/renderer/pipeline/descriptorSetLayoutCache.h"

#include "common/assert.h"

#include <algorithm>
#include <ranges>
#include <xxhash.h>

namespace Libs::Graphics {

// Packs the bindings, sorted by number, and then the create flags.
DescriptorSetLayoutKey::DescriptorSetLayoutKey(
    std::span<const vk::DescriptorSetLayoutBinding> bindings,
    vk::DescriptorSetLayoutCreateFlags              flags) {
	std::vector<vk::DescriptorSetLayoutBinding> sorted(bindings.begin(), bindings.end());
	std::ranges::sort(sorted, {}, &vk::DescriptorSetLayoutBinding::binding);
	m_words.reserve(sorted.size() * WORDS_PER_BINDING + 1);
	for (std::size_t index = 0; index < sorted.size(); index++) {
		const auto& binding = sorted[index];
		// Immutable samplers are not in the key, so a layout with them could be shared wrongly.
		EXIT_IF(binding.pImmutableSamplers != nullptr);
		// A repeated number is an invalid layout, and sorting would hide it.
		EXIT_IF(index > 0 && sorted[index - 1].binding == binding.binding);
		m_words.push_back(binding.binding);
		m_words.push_back(static_cast<uint32_t>(binding.descriptorType));
		m_words.push_back(binding.descriptorCount);
		m_words.push_back(static_cast<uint32_t>(binding.stageFlags));
	}
	m_words.push_back(static_cast<uint32_t>(static_cast<VkDescriptorSetLayoutCreateFlags>(flags)));
}

// Hashes the key words with XXH3, as the part keys of the pipeline libraries are.
std::size_t DescriptorSetLayoutKey::Hash() const {
	return XXH3_64bits(m_words.data(), m_words.size() * sizeof(uint32_t));
}

std::vector<vk::DescriptorSetLayoutBinding> DescriptorSetLayoutKey::Bindings() const {
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	// The last word is the create flags.
	for (std::size_t word = 0; word + 1 < m_words.size(); word += WORDS_PER_BINDING) {
		bindings.push_back(
		    {m_words[word + WORD_BINDING],
		     static_cast<vk::DescriptorType>(m_words[word + WORD_TYPE]), m_words[word + WORD_COUNT],
		     static_cast<vk::ShaderStageFlags>(m_words[word + WORD_STAGES]), nullptr});
	}
	return bindings;
}

vk::DescriptorSetLayoutCreateFlags DescriptorSetLayoutKey::Flags() const {
	return static_cast<vk::DescriptorSetLayoutCreateFlags>(m_words.back());
}

DescriptorSetLayoutCache::DescriptorSetLayoutCache(CreateFunction create, DestroyFunction destroy)
    : m_create(std::move(create)), m_destroy(std::move(destroy)) {}

DescriptorSetLayoutCache::~DescriptorSetLayoutCache() {
	for (const auto& layout: m_layouts | std::views::values) {
		m_destroy(layout);
	}
}

vk::DescriptorSetLayout
DescriptorSetLayoutCache::GetOrCreate(std::span<const vk::DescriptorSetLayoutBinding> bindings,
                                      vk::DescriptorSetLayoutCreateFlags              flags) {
	DescriptorSetLayoutKey key(bindings, flags);
	std::scoped_lock       lock(m_mutex);
	if (const auto found = m_layouts.find(key); found != m_layouts.end()) {
		return found->second;
	}
	// Made from the key's own bindings, so the layout is exactly what the key names.
	const auto                        sorted = key.Bindings();
	vk::DescriptorSetLayoutCreateInfo info {};
	info.flags        = key.Flags();
	info.bindingCount = static_cast<uint32_t>(sorted.size());
	info.pBindings    = sorted.data();
	const auto layout = m_create(info);
	m_layouts.emplace(std::move(key), layout);
	return layout;
}

std::size_t DescriptorSetLayoutCache::Size() const {
	std::scoped_lock lock(m_mutex);
	return m_layouts.size();
}

} // namespace Libs::Graphics
