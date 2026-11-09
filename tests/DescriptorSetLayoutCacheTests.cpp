// Tests the cache that shares descriptor set layouts. No device is needed, since the cache makes
// its layouts through functions the tests give it, which also count the layouts made and destroyed.

#include "graphics/host_gpu/renderer/pipeline/descriptorSetLayoutCache.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace Libs::Graphics;

namespace {

// How long a create takes in the thread test. A race between two creates of the same key needs a
// window to show up in.
constexpr std::chrono::microseconds kSlowCreate(2000);
constexpr unsigned kThreadCount = 16;
constexpr unsigned kKeyCount    = 3;
// The layouts the destroy test makes, which differ in the count of one binding.
constexpr unsigned kDestroyLayouts = 4;

constexpr vk::DescriptorSetLayoutCreateFlags kPush =
    vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR;
constexpr vk::DescriptorSetLayoutCreateFlags kNoFlags {};

// Stops the test with a message when a check fails.
void Check(bool passed, const std::string& what) {
	if (!passed) {
		std::fprintf(stderr, "DescriptorSetLayoutCacheTests: failed: %s\n", what.c_str());
		std::abort();
	}
}

// The number a fake handle stands for.
uint64_t Number(vk::DescriptorSetLayout layout) {
	return reinterpret_cast<uint64_t>(static_cast<VkDescriptorSetLayout>(layout));
}

// A cache over fake layouts. Each created layout is a new number, so two requests that got the same
// handle really got the same layout, and a destroyed handle can be matched to a created one.
struct CountingBackend {
	std::atomic<uint64_t>    next {1};
	std::atomic<uint64_t>    creates {0};
	std::chrono::microseconds create_delay {0};
	std::vector<uint64_t>    destroyed;

	vk::DescriptorSetLayout Create(const vk::DescriptorSetLayoutCreateInfo&) {
		creates++;
		if (create_delay.count() != 0) {
			std::this_thread::sleep_for(create_delay);
		}
		return vk::DescriptorSetLayout(reinterpret_cast<VkDescriptorSetLayout>(next++));
	}
	void Destroy(vk::DescriptorSetLayout layout) { destroyed.push_back(Number(layout)); }
};

DescriptorSetLayoutCache MakeCache(CountingBackend& backend) {
	return DescriptorSetLayoutCache([&](const auto& info) { return backend.Create(info); },
	                                [&](auto layout) { backend.Destroy(layout); });
}

// Two bindings that differ in every field from each other, so a flip of one field of the second
// is a flip of a field that really has a value.
std::vector<vk::DescriptorSetLayoutBinding> BaselineBindings() {
	return {{0, vk::DescriptorType::eStorageBuffer, 2, vk::ShaderStageFlagBits::eVertex, nullptr},
	        {5, vk::DescriptorType::eSampledImage, 3, vk::ShaderStageFlagBits::eFragment, nullptr}};
}

// Equal bindings and flags give one handle. Each single change gives another. The baseline of the
// comparison is the unchanged list, so a change that is ignored gives the same handle and fails.
void TestIdentity() {
	CountingBackend backend;
	auto            cache    = MakeCache(backend);
	const auto      baseline = BaselineBindings();
	const auto      handle   = cache.GetOrCreate(baseline, kNoFlags);
	Check(handle != nullptr, "a layout was null");
	Check(cache.GetOrCreate(baseline, kNoFlags) == handle, "equal bindings gave another handle");
	Check(cache.GetOrCreate(BaselineBindings(), kNoFlags) == handle,
	      "an equal copy gave another handle");

	// The same bindings listed in another order are the same layout.
	auto reversed = baseline;
	std::swap(reversed[0], reversed[1]);
	Check(cache.GetOrCreate(reversed, kNoFlags) == handle, "the order of the bindings split the key");

	Check(cache.GetOrCreate(baseline, kPush) != handle, "the create flags did not split the key");

	using Edit                                 = void (*)(vk::DescriptorSetLayoutBinding&);
	const std::pair<const char*, Edit> edits[] = {
	    {"binding number", [](auto& binding) { binding.binding += 1; }},
	    {"type", [](auto& binding) { binding.descriptorType = vk::DescriptorType::eStorageImage; }},
	    {"count", [](auto& binding) { binding.descriptorCount += 1; }},
	    {"stage flags", [](auto& binding) { binding.stageFlags = vk::ShaderStageFlagBits::eCompute; }},
	};
	std::set<uint64_t> handles {Number(handle)};
	for (const auto& [name, edit]: edits) {
		auto changed = baseline;
		edit(changed[1]);
		const auto other = cache.GetOrCreate(changed, kNoFlags);
		Check(other != handle, std::string("a change of the ") + name + " gave the same handle");
		handles.insert(Number(other));
	}
	Check(handles.size() == std::size(edits) + 1,
	      "the changed lists did not each get a layout of their own");

	// A shorter list is not equal to a longer one that starts the same.
	Check(cache.GetOrCreate(std::span(baseline).first(1), kNoFlags) != handle,
	      "a prefix shared the layout");
	Check(cache.GetOrCreate({}, kNoFlags) != cache.GetOrCreate({}, kPush),
	      "empty pushed and plain layouts are one");
	Check(cache.GetOrCreate({}, kNoFlags) == cache.GetOrCreate({}, kNoFlags),
	      "empty layouts are not shared");
	Check(backend.creates == cache.Size(), "the cache holds another number of layouts than it made");
	std::printf("DescriptorSetLayoutCacheTests: identity ok\n");
}

// The layout that is made has the bindings and the flags that the key names, in every field, and
// a key without flags makes a layout without flags.
void TestCreateInfo() {
	vk::DescriptorSetLayoutCreateInfo           made {};
	std::vector<vk::DescriptorSetLayoutBinding> made_bindings;
	DescriptorSetLayoutCache                    cache(
        [&](const vk::DescriptorSetLayoutCreateInfo& info) {
	        made          = info;
	        made_bindings = {info.pBindings, info.pBindings + info.bindingCount};
	        return vk::DescriptorSetLayout(reinterpret_cast<VkDescriptorSetLayout>(uintptr_t {1}));
        },
        [](auto) {});
	auto bindings = BaselineBindings();
	std::swap(bindings[0], bindings[1]);
	(void)cache.GetOrCreate(bindings, kPush);
	Check(made.bindingCount == 2, "the layout has another number of bindings");
	const auto expected = BaselineBindings();
	for (size_t index = 0; index < expected.size(); index++) {
		Check(made_bindings[index].binding == expected[index].binding &&
		          made_bindings[index].descriptorType == expected[index].descriptorType &&
		          made_bindings[index].descriptorCount == expected[index].descriptorCount &&
		          made_bindings[index].stageFlags == expected[index].stageFlags &&
		          made_bindings[index].pImmutableSamplers == nullptr,
		      "a binding was not made as the key names it");
	}
	Check(made.flags == kPush, "a pushed layout was made without the push flag");

	(void)cache.GetOrCreate(bindings, kNoFlags);
	Check(made.flags == kNoFlags, "a plain layout was made with flags");
	std::printf("DescriptorSetLayoutCacheTests: create info ok\n");
}

// Many threads asking for one key at once get one handle and make one layout. The create takes a
// moment, so a cache that looked up outside its lock would let several threads reach it.
void TestThreads() {
	CountingBackend backend;
	backend.create_delay = kSlowCreate;
	auto                               cache = MakeCache(backend);
	std::atomic<bool>                  go {false};
	std::vector<std::thread>           threads;
	std::vector<std::vector<uint64_t>> results(kThreadCount);
	for (unsigned index = 0; index < kThreadCount; ++index) {
		threads.emplace_back([&, index] {
			while (!go.load()) {
				std::this_thread::yield();
			}
			// Every thread asks for the same keys, and the keys differ in the count of the binding.
			for (unsigned key = 0; key < kKeyCount; ++key) {
				auto bindings = BaselineBindings();
				bindings[0].descriptorCount += key;
				results[index].push_back(Number(cache.GetOrCreate(bindings, kNoFlags)));
			}
		});
	}
	go = true;
	for (auto& thread: threads) {
		thread.join();
	}
	for (unsigned key = 0; key < kKeyCount; ++key) {
		std::set<uint64_t> handles;
		for (const auto& result: results) {
			handles.insert(result[key]);
		}
		Check(handles.size() == 1, "threads asking for one key got more than one handle");
	}
	Check(backend.creates == kKeyCount && cache.Size() == kKeyCount, "a key was created more than once");
	std::printf("DescriptorSetLayoutCacheTests: threads ok\n");
}

// The cache destroys each layout it made once, and only when it is destroyed itself.
void TestDestroy() {
	CountingBackend    backend;
	std::set<uint64_t> made;
	{
		auto cache = MakeCache(backend);
		for (unsigned count = 1; count <= kDestroyLayouts; ++count) {
			auto bindings               = BaselineBindings();
			bindings[0].descriptorCount = count;
			made.insert(Number(cache.GetOrCreate(bindings, kNoFlags)));
			// Asking again must not make or destroy anything.
			(void)cache.GetOrCreate(bindings, kNoFlags);
		}
		Check(backend.destroyed.empty(), "a layout was destroyed while the cache was alive");
	}
	const std::set<uint64_t> destroyed(backend.destroyed.begin(), backend.destroyed.end());
	Check(backend.destroyed.size() == made.size() && destroyed == made,
	      "the cache did not destroy each layout it made exactly once");
	std::printf("DescriptorSetLayoutCacheTests: destroy ok\n");
}

} // namespace

int main() {
	TestIdentity();
	TestCreateInfo();
	TestThreads();
	TestDestroy();
	std::printf("DescriptorSetLayoutCacheTests: ok\n");
	return 0;
}
