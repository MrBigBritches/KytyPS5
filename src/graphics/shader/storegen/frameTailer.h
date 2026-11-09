#ifndef GRAPHICS_SHADER_STOREGEN_FRAME_TAILER_H
#define GRAPHICS_SHADER_STOREGEN_FRAME_TAILER_H

#include "graphics/shader/shaderStoreRecord.h"
#include "graphics/shader/storegen/storeWorkerFormat.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <vector>

// Reads the file a store worker is still writing. The tailer is pure and is fed bytes, and the
// poller is the only part that touches a file. Neither needs a graphics device.
//
// The writer is one thread that appends each frame with a single write, one after the other. So
// bytes that follow a bad item mean the item was finished and is really bad, while a bad item at
// the very end of the bytes read so far may be a write that was caught half way. Such an item is
// held while the child lives and read again from the last good frame. The bytes of an item may
// come from older polls than the bytes after it, so a bad item is only final when it is judged on
// bytes that were all read together, which is right after such a re-read.
namespace Libs::Graphics::StoreGen {

/// The most a poller reads from the file in one poll, which leaves the rest for the next one. It is
/// two frames of the largest size, so a poll always has room for the frame it waits for and the
/// start of the next, and what a tailer buffers is bounded.
constexpr uint64_t MAX_POLL_READ_BYTES = 2ull * MAX_STORE_FRAME_COMPRESSED_BYTES;

// A window that starts at GoodOffset must reach past the largest frame, so that a bad item cannot
// end exactly at the window's end and be held again for ever.
static_assert(MAX_POLL_READ_BYTES > sizeof(StoreFrame) + MAX_STORE_FRAME_COMPRESSED_BYTES);

/// How a tailer stands.
enum class TailStatus : uint8_t {
	/// More bytes may still come, or the child may still finish a frame.
	Waiting,
	/// The child exited and the file ended exactly after a frame.
	Complete,
	/// The header is complete and is not the one expected. Final for this child.
	HeaderError,
	/// A frame announces more compressed bytes than MAX_STORE_FRAME_COMPRESSED_BYTES allows. Final
	/// for this child.
	FrameTooLarge,
	/// A frame is complete and bad, and more bytes follow it, all read together, so it is not a
	/// write caught half way. Final for this child, and the file is good up to GoodOffset.
	DamagedFrame,
	/// The child exited and the tail is short or bad. Final for this child, and a restart polls on
	/// from GoodOffset.
	DamagedTail,
	/// The file was cut back to less than GoodOffset, or is gone after frames were handed over.
	/// Only the poller sets it. The file cannot be resumed from GoodOffset, since a worker refuses
	/// a `kept` beyond the file's size. The supervisor starts again with kept 0, a new tailer and a
	/// new poller, and deduplicates the records it already holds by module id. Nothing clears it
	/// but a new tailer.
	Truncated,
};

/// A record handed over by a tailer.
struct TailedFrame {
	/// The record, as before it was framed.
	std::vector<uint8_t> raw;
	/// The offset in the file just after this frame, which is the `kept` of a restart.
	uint64_t end_offset = 0;
};

/// Hands over the complete frames of a generated file as its bytes arrive.
class FrameTailer {
public:
	/// @param store_header      the StoreHeader bytes the file must start with.
	/// @param generator_version the version the file's header must name.
	/// @param identity          the eboot identity the header must hold, or nothing to accept the
	///                          file's own, which Identity then returns.
	FrameTailer(std::vector<uint8_t> store_header, uint32_t generator_version,
	            std::optional<EbootIdentity> identity);

	/// Appends bytes that came after all the bytes fed so far. Any chunk size is fine.
	void Feed(std::span<const uint8_t> bytes);

	/// Hands over every complete good frame that follows the ones already handed over, in order.
	/// It stops at the first frame that is not complete and good.
	///
	/// A bad item is a header that is not the expected one, a frame size over the cap, or a
	/// complete frame that does not unframe. It is final only when it is judged in the first Poll
	/// after DropUndelivered, since its bytes may be older than the bytes after it, and when it has
	/// bytes after it or the child has exited. Otherwise it is held with WantsReread true. A short
	/// tail is no bad item, so it only waits while the child lives and is final at once after the
	/// exit.
	/// @param child_exited whether the child has exited and written all it will write.
	[[nodiscard]] std::vector<TailedFrame> Poll(bool child_exited);

	/// How the tailer stands after the last Poll.
	[[nodiscard]] TailStatus Status() const { return m_status; }

	/// Whether the status is anything but Waiting.
	[[nodiscard]] bool IsFinal() const { return m_status != TailStatus::Waiting; }

	/// The offset just after the last good frame, or just after the header when there is none, or
	/// zero while the header is not complete. After a final status this is where to truncate the
	/// file.
	[[nodiscard]] uint64_t GoodOffset() const { return m_good_offset; }

	/// The identity from the file's header once the header is complete.
	[[nodiscard]] std::optional<EbootIdentity> Identity() const { return m_identity; }

	/// How many bytes are buffered and not yet handed over.
	[[nodiscard]] size_t BufferedBytes() const { return m_buffer.size() - m_consumed; }

	/// How many bytes the tailer holds in memory, which after a Poll is the same as BufferedBytes
	/// because the handed over bytes are dropped.
	[[nodiscard]] size_t RetainedBytes() const { return m_buffer.size(); }

	/// Whether the last Poll held a bad item. The caller drops the buffer and feeds the file again
	/// from GoodOffset, since the bytes it has fed may be a write that was not finished and are not
	/// read again otherwise.
	[[nodiscard]] bool WantsReread() const { return m_wants_reread; }

	/// Drops every byte that was not handed over and clears a final status, except Truncated, so
	/// the caller can feed the file again from GoodOffset. It is for a held item, and for a restart
	/// of the child. A bad item in the next Poll, and only in that one, is judged on bytes that
	/// were all read after this call.
	void DropUndelivered() {
		m_buffer.clear();
		m_consumed     = 0;
		m_wants_reread = false;
		m_after_rewind = true;
		if (m_status != TailStatus::Truncated) {
			m_status = TailStatus::Waiting;
		}
	}

	/// Makes the status Truncated, for a poller that found the file cut back below GoodOffset or
	/// gone. Only a new tailer clears it.
	void MarkTruncated() { m_status = TailStatus::Truncated; }

private:
	bool                       CheckHeader(bool child_exited);
	std::optional<TailedFrame> TakeFrame(bool child_exited);
	void JudgeBadItem(bool trailing, bool child_exited, TailStatus final_status);

	std::vector<uint8_t>         m_store_header;
	uint32_t                     m_generator_version = 0;
	std::optional<EbootIdentity> m_identity;
	std::vector<uint8_t>         m_buffer;
	/// How much of m_buffer is already handed over, so the buffer is not shifted for every frame.
	size_t m_consumed     = 0;
	bool   m_header_done  = false;
	bool   m_wants_reread = false;
	/// Set by a rewind, and taken into m_fresh by the next Poll.
	bool m_after_rewind = false;
	/// Whether the current Poll is the first after a rewind, so that the bytes it judges were all
	/// read after it. A bad item is only final then.
	bool       m_fresh       = false;
	uint64_t   m_good_offset = 0;
	TailStatus m_status      = TailStatus::Waiting;
};

/// Follows a file a worker appends to, from the last offset it read. The file is opened for each
/// poll and closed again, so the worker's writes and a restart's truncation are never blocked.
class FileTailPoller {
public:
	/// @param path          the worker's output file, which need not exist yet.
	/// @param tailer        receives the bytes. It must outlive the poller.
	/// @param max_poll_bytes the most one poll reads. A value below one maximum frame, which is
	///                       MAX_STORE_FRAME_COMPRESSED_BYTES plus a frame header, is for tests
	///                       only. With the default, a bad item that ends at the window's end
	///                       cannot be held again for ever, since the window that starts at
	///                       GoodOffset always reaches past the item.
	FileTailPoller(std::filesystem::path path, FrameTailer& tailer,
	               uint64_t max_poll_bytes = MAX_POLL_READ_BYTES);

	/// Reads the bytes that were appended since the last poll, up to the cap, feeds them and
	/// returns the frames that became complete. A missing file means nothing yet while no frame was
	/// handed over, since the worker writes none while it predicts. A file cut back to GoodOffset
	/// or more is read again from GoodOffset. A file cut back to less than GoodOffset, or gone
	/// after frames were handed over, gives the final status Truncated.
	///
	/// A held bad item is read again from GoodOffset on the next poll, which is also what judges it
	/// for good. Once the child exited the caller keeps polling until the status is not Waiting,
	/// since a poll that left bytes unread does not yet tell the tailer that the child exited, and
	/// since a bad item at the end is held once more before it is final.
	///
	/// What the supervisor does with the status:
	/// - On DamagedFrame or FrameTooLarge while the child lives, it terminates the child, cuts
	///   the output file, and restarts fresh: kept 0, a new tailer and poller, since a rank cannot
	///   be derived from a file offset. HeaderError ends generation, since no restart would write
	///   another header. On Truncated it restarts fresh too. After a restart with kept = 0, as
	///   after exit code 14, the fresh run delivers records the inbox already holds, so the
	///   supervisor must deduplicate them by module id.
	/// - It treats Complete as done only together with the child's exit code 0. Exit code 10 is
	///   Verify mode's answer, which nothing asks for yet, so it ends generation.
	/// - After a crash it calls RestartFrom itself, and it does not poll with child_exited true
	///   between that and the respawn. A fresh restart makes a new tailer and poller instead.
	/// @param child_exited passed to FrameTailer::Poll. The caller reads whether the child exited
	///                     before calling this, since bytes written just before the exit must still
	///                     be read after it is seen.
	[[nodiscard]] std::vector<TailedFrame> Poll(bool child_exited);

	/// Prepares for a restarted child, which the caller starts with `kept` set to GoodOffset. The
	/// bytes not handed over are dropped, the status is cleared and the next poll reads from
	/// GoodOffset. Call it after the child exited and before it is spawned again. A restart that
	/// keeps nothing, as after the order mismatch exit, makes a new tailer and a new poller. It
	/// does nothing after Truncated, which only a new tailer clears.
	void RestartFrom();

	/// The number of file bytes read so far.
	[[nodiscard]] uint64_t BytesRead() const { return m_offset; }

private:
	bool ReadNewBytes();

	std::filesystem::path m_path;
	FrameTailer&          m_tailer;
	uint64_t              m_max_poll_bytes = MAX_POLL_READ_BYTES;
	uint64_t              m_offset         = 0;
};

} // namespace Libs::Graphics::StoreGen

#endif // GRAPHICS_SHADER_STOREGEN_FRAME_TAILER_H
