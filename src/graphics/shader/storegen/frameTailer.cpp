#include "graphics/shader/storegen/frameTailer.h"

#include "common/file.h"
#include "graphics/shader/shaderStoreRecord.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace Libs::Graphics::StoreGen {

namespace {

/// The most the poller reads from the file into its buffer at a time, so a poll reads a big
/// backlog in pieces and never holds more than this beside the tailer's own buffer.
constexpr uint32_t POLL_READ_CHUNK_BYTES = 1u << 20;

} // namespace

FrameTailer::FrameTailer(std::vector<uint8_t> store_header, uint32_t generator_version,
                         std::optional<EbootIdentity> identity)
    : m_store_header(std::move(store_header)), m_generator_version(generator_version),
      m_identity(identity) {}

void FrameTailer::Feed(std::span<const uint8_t> bytes) {
	m_buffer.insert(m_buffer.end(), bytes.begin(), bytes.end());
}

// Decides what a bad item means. It is final with the given status only when all its bytes were
// read after the last rewind, which is the first poll after it, and when it is not a write that may
// still be in flight, which is the case when bytes follow it or the child exited. Otherwise it is
// held and the caller is asked to read it again, since the item's own bytes may be older than the
// ones after it, or may have been caught half way.
void FrameTailer::JudgeBadItem(bool trailing, bool child_exited, TailStatus final_status) {
	if (m_fresh && (!trailing || child_exited)) {
		m_status = final_status;
	} else {
		m_wants_reread = true;
	}
}

// Checks the generated file's header once all of it is buffered, and takes it off the buffer when
// it is the expected one. Returns false while the header is incomplete or when it is wrong, and a
// wrong one is judged by JudgeBadItem.
bool FrameTailer::CheckHeader(bool child_exited) {
	const size_t header_bytes = m_store_header.size() + GENERATED_HEADER_TAIL_BYTES;
	if (BufferedBytes() < header_bytes) {
		return false;
	}
	const auto header =
	    ParseGeneratedHeader(std::span<const uint8_t>(m_buffer).subspan(m_consumed, header_bytes),
	                         m_store_header.size());
	const bool matches = header && header->store_header == m_store_header &&
	                     header->generator_version == m_generator_version &&
	                     (!m_identity || *m_identity == header->identity);
	if (!matches) {
		JudgeBadItem(BufferedBytes() == header_bytes, child_exited, TailStatus::HeaderError);
		return false;
	}
	m_identity = header->identity;
	m_consumed += header_bytes;
	m_good_offset = header_bytes;
	m_header_done = true;
	return true;
}

// Takes the next frame off the buffer when it is complete and good. Returns nothing when more bytes
// are needed, or when the frame is bad, which JudgeBadItem then rules on.
std::optional<TailedFrame> FrameTailer::TakeFrame(bool child_exited) {
	const auto pending = std::span<const uint8_t>(m_buffer).subspan(m_consumed);
	if (pending.size() < sizeof(StoreFrame)) {
		return std::nullopt;
	}
	StoreFrame frame;
	std::memcpy(&frame, pending.data(), sizeof(frame));
	// Checked before waiting for the bytes, so a corrupt size does not make the tailer wait for
	// them.
	if (frame.compressed_size > MAX_STORE_FRAME_COMPRESSED_BYTES) {
		JudgeBadItem(pending.size() == sizeof(frame), child_exited, TailStatus::FrameTooLarge);
		return std::nullopt;
	}
	auto unframed = UnframeStoreRecord(pending);
	if (!unframed) {
		const size_t frame_bytes = sizeof(frame) + frame.compressed_size;
		// A frame that is not all there yet only waits.
		if (pending.size() >= frame_bytes) {
			const bool trailing = pending.size() == frame_bytes;
			JudgeBadItem(trailing, child_exited,
			             trailing ? TailStatus::DamagedTail : TailStatus::DamagedFrame);
		}
		return std::nullopt;
	}
	m_consumed += unframed->frame_bytes;
	m_good_offset += unframed->frame_bytes;
	return TailedFrame {.raw = std::move(unframed->raw), .end_offset = m_good_offset};
}

std::vector<TailedFrame> FrameTailer::Poll(bool child_exited) {
	std::vector<TailedFrame> frames;
	if (m_status == TailStatus::HeaderError || m_status == TailStatus::FrameTooLarge ||
	    m_status == TailStatus::DamagedFrame || m_status == TailStatus::Truncated) {
		return frames;
	}
	m_status       = TailStatus::Waiting;
	m_wants_reread = false;
	// The flag of a rewind covers this poll only, since the next poll may add bytes to a half
	// frame.
	m_fresh        = m_after_rewind;
	m_after_rewind = false;
	if (m_header_done || CheckHeader(child_exited)) {
		while (auto frame = TakeFrame(child_exited)) {
			frames.push_back(std::move(*frame));
		}
	}
	// Drops the handed over bytes once per poll, so a poll costs the undelivered tail and no more.
	m_buffer.erase(m_buffer.begin(), m_buffer.begin() + static_cast<ptrdiff_t>(m_consumed));
	m_consumed = 0;
	if (m_status == TailStatus::Waiting && child_exited && !m_wants_reread) {
		m_status =
		    BufferedBytes() == 0 && m_header_done ? TailStatus::Complete : TailStatus::DamagedTail;
	}
	return frames;
}

FileTailPoller::FileTailPoller(std::filesystem::path path, FrameTailer& tailer,
                               uint64_t max_poll_bytes)
    : m_path(std::move(path)), m_tailer(tailer), m_max_poll_bytes(max_poll_bytes) {}

// Reads what was appended since the last read, up to the cap of one poll, and feeds it to the
// tailer. A file that is shorter than what was read was cut back, so the poll starts again from
// GoodOffset, or the tailer is Truncated when the file is shorter than GoodOffset itself. A final
// tailer is not fed, since it would only buffer bytes it will not use. Returns whether the read
// reached the end of the file, which is true when there was nothing to read.
bool FileTailPoller::ReadNewBytes() {
	Common::File file;
	// The worker writes no file while it predicts, so a file that is not there is nothing yet.
	// After frames were handed over it is gone for good. An open can also fail for a moment on
	// Windows, for a scanner or a pending delete, so only a file that is really absent is
	// Truncated.
	if (!file.Open(m_path, Common::File::Mode::Read)) {
		std::error_code missing_error;
		const bool      absent = !std::filesystem::exists(m_path, missing_error) && !missing_error;
		if (absent && m_tailer.GoodOffset() > 0) {
			m_tailer.MarkTruncated();
		}
		return true;
	}
	const uint64_t size = file.Size();
	if (size < m_tailer.GoodOffset()) {
		// Frames that were handed over are no longer in the file, which a restart cannot repair.
		m_tailer.MarkTruncated();
		return true;
	}
	if (size < m_offset) {
		RestartFrom();
	}
	if (m_tailer.IsFinal() || size <= m_offset || !file.Seek(m_offset)) {
		return true;
	}
	const uint64_t stop = m_offset + std::min<uint64_t>(size - m_offset, m_max_poll_bytes);
	// Sized on the first poll that has bytes to read and kept, so later polls do not allocate and
	// zero a megabyte each. Only the bytes a read returned are ever fed.
	if (m_chunk.empty()) {
		m_chunk.resize(POLL_READ_CHUNK_BYTES);
	}
	while (m_offset < stop) {
		const auto want = static_cast<uint32_t>(std::min<uint64_t>(stop - m_offset, m_chunk.size()));
		uint32_t   got  = 0;
		file.Read(m_chunk.data(), want, &got);
		// A read that gives nothing is the end of the file, like a file that is missing. It must
		// not say that more is coming, or a poller after the child's exit would never finish.
		if (got == 0) {
			return true;
		}
		m_tailer.Feed(std::span<const uint8_t>(m_chunk).first(got));
		m_offset += got;
	}
	return m_offset >= size;
}

std::vector<TailedFrame> FileTailPoller::Poll(bool child_exited) {
	const bool at_end = ReadNewBytes();
	// A poll that stopped before the end of the file has not seen what the child wrote last, so the
	// tailer is not told yet that the child exited.
	auto frames = m_tailer.Poll(child_exited && at_end);
	// A bad item may have been read before the writer's bytes landed in it, and bytes already fed
	// are never read again. So a held item is dropped and read afresh from the last good frame, and
	// the next poll judges it for good, on bytes that were all read together.
	if (m_tailer.WantsReread()) {
		RestartFrom();
		m_rewinds++;
	}
	return frames;
}

void FileTailPoller::RestartFrom() {
	// A file that lost handed over frames cannot be resumed from GoodOffset, so only a new tailer
	// clears this status.
	if (m_tailer.Status() == TailStatus::Truncated) {
		return;
	}
	m_tailer.DropUndelivered();
	m_offset = m_tailer.GoodOffset();
}

} // namespace Libs::Graphics::StoreGen
