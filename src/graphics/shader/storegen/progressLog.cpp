#include "graphics/shader/storegen/progressLog.h"

#include "common/file.h"

#include <algorithm>
#include <charconv>

namespace Libs::Graphics::StoreGen {

namespace {

// The digits of a hash, which the worker writes as {:016x}.
constexpr size_t HASH_DIGITS      = 16;
constexpr int    HEX_BASE         = 16;
constexpr size_t READ_CHUNK_BYTES = 64 * 1024;

// The words of a line, split at single spaces. An empty word, from two spaces in a row or a space
// at either end, is kept so the caller sees that the line is not the worker's.
std::vector<std::string_view> SplitWords(std::string_view line) {
	std::vector<std::string_view> words;
	size_t                        begin = 0;
	while (true) {
		const auto end = line.find(' ', begin);
		if (end == std::string_view::npos) {
			words.push_back(line.substr(begin));
			return words;
		}
		words.push_back(line.substr(begin, end - begin));
		begin = end + 1;
	}
}

// A decimal number that fills the whole word.
template <typename T>
bool ParseNumber(std::string_view word, T& value) {
	const auto [end, error] = std::from_chars(word.data(), word.data() + word.size(), value);
	return error == std::errc() && end == word.data() + word.size() && !word.empty();
}

// A hash of exactly 16 hex digits, so a line cut short is never taken for a whole one.
bool ParseHash(std::string_view word, uint64_t& value) {
	if (word.size() != HASH_DIGITS) {
		return false;
	}
	const auto [end, error] =
	    std::from_chars(word.data(), word.data() + word.size(), value, HEX_BASE);
	return error == std::errc() && end == word.data() + word.size();
}

// The value of `name=value`, or nothing when the word is not that.
std::optional<std::string_view> ValueOf(std::string_view word, std::string_view name) {
	if (word.size() <= name.size() || !word.starts_with(name) || word[name.size()] != '=') {
		return std::nullopt;
	}
	return word.substr(name.size() + 1);
}

std::optional<ProgressLine> ParseDrop(const std::vector<std::string_view>& words) {
	// drop <rank> <hash> <reason> [predicted=<n> actual=<n>]
	constexpr size_t MIN_WORDS  = 4;
	constexpr size_t PUSH_WORDS = 6;
	if (words.size() < MIN_WORDS) {
		return std::nullopt;
	}
	ProgressLine line;
	line.kind = ProgressKind::Drop;
	if (!ParseNumber(words[1], line.rank_or_count) || !ParseHash(words[2], line.hash)) {
		return std::nullopt;
	}
	const auto reason = words[3];
	if (reason == "resource_count" && words.size() == MIN_WORDS) {
		line.reason = ProgressDropReason::ResourceCount;
		return line;
	}
	if (reason == "frame_too_large" && words.size() == MIN_WORDS) {
		line.reason = ProgressDropReason::FrameTooLarge;
		return line;
	}
	if (reason == "push_start" && words.size() == PUSH_WORDS) {
		const auto predicted = ValueOf(words[4], "predicted");
		const auto actual    = ValueOf(words[5], "actual");
		if (predicted && actual && ParseNumber(*predicted, line.predicted_push) &&
		    ParseNumber(*actual, line.actual_push)) {
			line.reason = ProgressDropReason::PushStart;
			return line;
		}
	}
	return std::nullopt;
}

std::optional<ProgressLine> ParseSummary(const std::vector<std::string_view>& words) {
	// summary exit=<n> <name>=<value> ...
	ProgressLine line;
	line.kind     = ProgressKind::Summary;
	bool has_exit = false;
	for (size_t i = 1; i < words.size(); i++) {
		const auto split = words[i].find('=');
		if (split == std::string_view::npos || split == 0) {
			return std::nullopt;
		}
		const auto name  = words[i].substr(0, split);
		const auto value = words[i].substr(split + 1);
		if (name == "exit") {
			if (!ParseNumber(value, line.exit_code)) {
				return std::nullopt;
			}
			has_exit = true;
		}
		line.fields.emplace_back(std::string(name), std::string(value));
	}
	return has_exit ? std::optional<ProgressLine>(line) : std::nullopt;
}

} // namespace

std::optional<ProgressLine> ParseProgressLine(std::string_view line) {
	const auto   words = SplitWords(line);
	const auto   kind  = words[0];
	ProgressLine parsed;
	if (kind == "predict" && words.size() == 2) {
		parsed.kind = ProgressKind::Predict;
		return ParseHash(words[1], parsed.hash) ? std::optional<ProgressLine>(parsed)
		                                        : std::nullopt;
	}
	if ((kind == "start" || kind == "order") && words.size() == 3) {
		parsed.kind = kind == "start" ? ProgressKind::Start : ProgressKind::Order;
		return ParseNumber(words[1], parsed.rank_or_count) && ParseHash(words[2], parsed.hash)
		           ? std::optional<ProgressLine>(parsed)
		           : std::nullopt;
	}
	if (kind == "drop") {
		return ParseDrop(words);
	}
	if (kind == "summary") {
		return ParseSummary(words);
	}
	return std::nullopt;
}

ProgressLogReader::ProgressLogReader(std::filesystem::path path): m_path(std::move(path)) {}

void ProgressLogReader::BeginSpawn(ProgressClock::time_point now) {
	uint64_t size = 0;
	if (Common::File file; file.Open(m_path, Common::File::Mode::Read)) {
		size = file.Size();
	}
	BeginSpawnAt(size, now);
}

void ProgressLogReader::BeginSpawnAt(uint64_t offset, ProgressClock::time_point now) {
	m_offset = offset;
	// A last line without its newline belongs to the spawn that was cut off, not to this one.
	m_partial.clear();
	m_skipping          = false;
	m_lines_since_spawn = 0;
	m_last_line.reset();
	m_last_line_time    = now;
	m_start_since_spawn = false;
	m_drops.clear();
	m_summary.reset();
	std::ranges::fill(m_kind_counts, size_t {0});
}

void ProgressLogReader::BeginFreshRun(ProgressClock::time_point now) {
	m_last_start.reset();
	m_order.reset();
	BeginSpawn(now);
}

void ProgressLogReader::TakeLine(std::string_view line, ProgressClock::time_point now) {
	if (!line.empty() && line.back() == '\r') {
		line.remove_suffix(1);
	}
	if (line.empty()) {
		return;
	}
	auto parsed = ParseProgressLine(line);
	if (!parsed) {
		m_malformed++;
		return;
	}
	m_lines_since_spawn++;
	m_kind_counts[static_cast<size_t>(parsed->kind)]++;
	m_last_line_time = now;
	switch (parsed->kind) {
		case ProgressKind::Start:
			m_start_since_spawn = true;
			m_last_start        = parsed;
			break;
		case ProgressKind::Drop: m_drops.push_back(*parsed); break;
		case ProgressKind::Order: m_order = parsed; break;
		case ProgressKind::Summary: m_summary = parsed; break;
		case ProgressKind::Predict:
			if (m_predicted_seen.insert(parsed->hash).second) {
				m_predicted.push_back(parsed->hash);
			}
			if (!m_first_predict_time) {
				m_first_predict_time = now;
			}
			break;
	}
	m_last_line = std::move(parsed);
}

size_t ProgressLogReader::Poll(ProgressClock::time_point now) {
	Common::File file;
	// The worker opens its log after it has read the eboot, so a log that is not there is nothing
	// yet.
	if (!file.Open(m_path, Common::File::Mode::Read)) {
		m_caught_up = true;
		return 0;
	}
	const uint64_t size = file.Size();
	if (size < m_offset) {
		// Only the worker appends to the log, so a shorter one is another file in its place. It is
		// read from its start, since its lines are all new, and a held partial line is of the old
		// one.
		m_offset = 0;
		m_partial.clear();
		m_skipping = false;
		m_resets++;
	}
	if (size == m_offset || !file.Seek(m_offset)) {
		m_caught_up = size == m_offset;
		return 0;
	}
	const uint64_t    stop = m_offset + std::min(size - m_offset, MAX_PROGRESS_POLL_BYTES);
	std::vector<char> chunk(READ_CHUNK_BYTES);
	const size_t      before = m_lines_since_spawn;
	while (m_offset < stop) {
		const auto want = static_cast<uint32_t>(std::min<uint64_t>(stop - m_offset, chunk.size()));
		uint32_t   got  = 0;
		file.Read(chunk.data(), want, &got);
		if (got == 0) {
			break;
		}
		m_offset += got;
		std::string_view fresh(chunk.data(), got);
		// The rest of a line that was too long is skipped up to its newline.
		if (m_skipping) {
			const auto newline = fresh.find('\n');
			if (newline == std::string_view::npos) {
				continue;
			}
			fresh.remove_prefix(newline + 1);
			m_skipping = false;
		}
		m_partial.append(fresh);
		size_t begin = 0;
		for (auto end = m_partial.find('\n', begin); end != std::string::npos;
		     end      = m_partial.find('\n', begin)) {
			TakeLine(std::string_view(m_partial).substr(begin, end - begin), now);
			begin = end + 1;
		}
		m_partial.erase(0, begin);
		// A line this long is not the worker's. Dropping it keeps the held bytes bounded, and its
		// rest is skipped up to the newline, so no fragment of it is read as a line.
		if (m_partial.size() > MAX_PROGRESS_LINE_BYTES) {
			m_malformed++;
			m_partial.clear();
			m_skipping = true;
		}
	}
	m_caught_up = m_offset >= size;
	return m_lines_since_spawn - before;
}

size_t PollWhileProgressing(const std::function<size_t()>&       poll,
                            const std::function<bool()>&         caught_up,
                            const std::function<ProgressMark()>& mark) {
	size_t found = 0;
	while (true) {
		const auto before = mark();
		found += poll();
		if (caught_up()) {
			return found;
		}
		// A poll that is not caught up and moved nothing, as with a failed seek or a read that gave
		// no bytes, would give the same again, so the loop ends instead of spinning.
		if (mark() == before) {
			return found;
		}
	}
}

size_t ProgressLogReader::PollToEnd(ProgressClock::time_point now) {
	return PollWhileProgressing([&] { return Poll(now); }, [&] { return m_caught_up; },
	                            [&] { return ProgressMark {m_offset, m_resets}; });
}

} // namespace Libs::Graphics::StoreGen
