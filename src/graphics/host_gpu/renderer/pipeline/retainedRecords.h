#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RETAINEDRECORDS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RETAINEDRECORDS_H_

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// The store records that pre-scanned modules keep until their first use. A prescan module is a guess
// about what the game will run, so its record goes to the shader store only once a lookup finds the
// module, and a guess the game never runs leaves nothing in the store. The records wait in one list,
// oldest first, under a cap on their bytes. Header only, since every file that destroys a
// ShaderModule destroys its record too.
namespace Libs::Graphics {

/// The most bytes of store records the prescan's modules keep while they wait for their first use.
/// Past it the oldest are dropped, and their modules are then never stored.
inline constexpr uint64_t PRESCAN_RETAIN_CAP_BYTES = 64ull * 1024ull * 1024ull;

class RetainedRecordList;

/// The store record of one prescan module, as the helper encoded it, and its place in the list. The
/// links live in the record, so the list never allocates a node of its own. A record the list
/// dropped stays with its module, empty and unlinked, until the module's first use or its end.
struct RetainedRecord {
	RetainedRecord() = default;
	/// Takes the bytes of a module record, which are appended to the store as they are.
	explicit RetainedRecord(std::vector<uint8_t>&& record_bytes): bytes(std::move(record_bytes)) {}
	/// Leaves the list first, so a module destroyed while its record waits leaves no dangling link.
	~RetainedRecord();
	RetainedRecord(const RetainedRecord&)            = delete;
	RetainedRecord& operator=(const RetainedRecord&) = delete;

	/// Whether the record waits in a list.
	[[nodiscard]] bool Linked() const noexcept { return list != nullptr; }

	/// The record's bytes, empty once the list dropped it or once they were taken for the store.
	std::vector<uint8_t> bytes;
	/// The neighbours in the list, towards its oldest and its newest end. Null at either end and
	/// while the record is not in a list.
	RetainedRecord* older = nullptr;
	RetainedRecord* newer = nullptr;
	/// The list the record waits in, or null.
	RetainedRecordList* list = nullptr;
};

/// Owns the retained record of a module. The record belongs to the module the program cache holds
/// and not to its value, so a copy of a module has no record. A move takes the record along, and the
/// list is unaffected, since it links the records and not the modules.
class RetainedRecordSlot {
public:
	RetainedRecordSlot() = default;
	/// A copy starts without a record.
	RetainedRecordSlot(const RetainedRecordSlot& /*other*/) noexcept {}
	/// Assigning a copy leaves this slot without a record, as a copy has none.
	RetainedRecordSlot& operator=(const RetainedRecordSlot& other) noexcept {
		if (this != &other) {
			record.reset();
		}
		return *this;
	}
	RetainedRecordSlot(RetainedRecordSlot&&) noexcept            = default;
	RetainedRecordSlot& operator=(RetainedRecordSlot&&) noexcept = default;
	~RetainedRecordSlot()                                        = default;

	/// Whether the module has a record, waiting or dropped.
	explicit operator bool() const noexcept { return record != nullptr; }

	std::unique_ptr<RetainedRecord> record;
};

/// What the list of retained records did. The pipeline cache passes it on through GenerationStats,
/// and the first-run report does not read it yet.
struct RetainedRecordCounters {
	/// Records taken out for the store at their module's first use.
	uint64_t stored_first_use = 0;
	/// The most bytes the list held at once, measured after each new record was capped.
	uint64_t retained_peak_bytes = 0;
	/// Records dropped to stay under the cap, the oldest first, or alone when one is larger than
	/// the whole cap. Their modules are never stored.
	uint64_t dropped_over_cap = 0;
	/// Records that still waited when the list was released at the stop.
	uint64_t unused_at_exit = 0;
	/// Records and bytes waiting now.
	uint64_t retained       = 0;
	uint64_t retained_bytes = 0;
};

/// The records that wait for their module's first use, oldest first, under a cap on their bytes.
/// Every operation is constant time, apart from the drops a new record causes. The thread that owns
/// the program cache owns it, which is the GPU thread while the game runs.
class RetainedRecordList {
public:
	/// @param cap_bytes the most bytes the list holds, which a test may set small.
	explicit RetainedRecordList(uint64_t cap_bytes = PRESCAN_RETAIN_CAP_BYTES) noexcept
	    : m_cap_bytes(cap_bytes) {}
	/// Detaches every record still waiting, so a module destroyed later never touches the list.
	~RetainedRecordList() { Release(); }
	RetainedRecordList(const RetainedRecordList&)            = delete;
	RetainedRecordList& operator=(const RetainedRecordList&) = delete;

	/// Puts a record in as the newest, then drops the oldest records until the bytes fit the cap. A
	/// record larger than the whole cap is dropped alone and never enters, so it costs the waiting
	/// records nothing. A record already in a list is left alone.
	void Add(RetainedRecord& record) noexcept;
	/// Takes a waiting record's bytes out for the store and counts its first use. A record that was
	/// dropped, or never added, gives nothing.
	[[nodiscard]] std::vector<uint8_t> Take(RetainedRecord& record) noexcept;
	/// Drops every waiting record and counts it as unused at exit. Calling it again adds nothing.
	void Release() noexcept;
	/// Changes the cap, which applies from the next Add.
	void SetCap(uint64_t cap_bytes) noexcept { m_cap_bytes = cap_bytes; }
	/// What the list did so far.
	[[nodiscard]] RetainedRecordCounters Counters() const noexcept { return m_counters; }

private:
	// A record leaves its list from its own destructor.
	friend struct RetainedRecord;

	// Removes a record from the list and leaves its bytes alone.
	void Unlink(RetainedRecord& record) noexcept;
	// Frees the bytes of a record and takes it out of the list.
	void Drop(RetainedRecord& record) noexcept;

	RetainedRecord*        m_oldest = nullptr;
	RetainedRecord*        m_newest = nullptr;
	uint64_t               m_cap_bytes;
	RetainedRecordCounters m_counters;
};

inline RetainedRecord::~RetainedRecord() {
	if (list != nullptr) {
		list->Unlink(*this);
	}
}

inline void RetainedRecordList::Add(RetainedRecord& record) noexcept {
	if (record.Linked()) {
		return;
	}
	// The cap counts size() and not capacity(). The bytes come from the helper's answer at the size
	// it framed them, so the two are close, and the cap is a bound on memory and not an exact budget.
	if (record.bytes.size() > m_cap_bytes) {
		std::vector<uint8_t>().swap(record.bytes);
		m_counters.dropped_over_cap++;
		return;
	}
	record.list  = this;
	record.older = m_newest;
	record.newer = nullptr;
	(m_newest != nullptr ? m_newest->newer : m_oldest) = &record;
	m_newest = &record;
	m_counters.retained++;
	m_counters.retained_bytes += record.bytes.size();
	// The oldest go first. The new record fits the cap alone, so the loop stops before reaching it.
	while (m_counters.retained_bytes > m_cap_bytes && m_oldest != nullptr) {
		Drop(*m_oldest);
		m_counters.dropped_over_cap++;
	}
	m_counters.retained_peak_bytes =
	    std::max(m_counters.retained_peak_bytes, m_counters.retained_bytes);
}

inline std::vector<uint8_t> RetainedRecordList::Take(RetainedRecord& record) noexcept {
	if (record.list != this) {
		return {};
	}
	Unlink(record);
	m_counters.stored_first_use++;
	return std::move(record.bytes);
}

inline void RetainedRecordList::Release() noexcept {
	while (m_oldest != nullptr) {
		Drop(*m_oldest);
		m_counters.unused_at_exit++;
	}
}

inline void RetainedRecordList::Unlink(RetainedRecord& record) noexcept {
	if (record.list != this) {
		return;
	}
	(record.older != nullptr ? record.older->newer : m_oldest) = record.newer;
	(record.newer != nullptr ? record.newer->older : m_newest) = record.older;
	record.older = nullptr;
	record.newer = nullptr;
	record.list  = nullptr;
	m_counters.retained--;
	m_counters.retained_bytes -= record.bytes.size();
}

// The bytes are counted out by Unlink before they are freed, so the totals stay exact.
inline void RetainedRecordList::Drop(RetainedRecord& record) noexcept {
	Unlink(record);
	std::vector<uint8_t>().swap(record.bytes);
}

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_PIPELINE_RETAINEDRECORDS_H_
