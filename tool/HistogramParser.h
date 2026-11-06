#ifndef HISTOGRAM_PARSER_HEADER
#define HISTOGRAM_PARSER_HEADER

#include "AioGroupReader.h"
#include "../src/MsdPrefixCodec.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Incremental MSD group stream parser feeding a single histogram.
//
// width is (k + 3) / 4 key bytes. count_bytes is the high-file count width,
// or 0 for low.bin. High records contribute one bucket per stored count; low
// groups contribute their header N directly to bucket 1 without visiting any
// record payload.
class GroupCounter
{
public:
    GroupCounter(size_t width, uint32_t count_bytes, uint32_t min_freq,
                 uint32_t max_freq, std::vector<int64_t>& histogram)
        : width_(width), count_bytes_(count_bytes), min_freq_(min_freq),
          max_freq_(max_freq), histogram_(histogram)
    {
        if (!width || width > 32 || count_bytes > 4 || min_freq > max_freq)
            throw std::runtime_error("invalid MSD histogram layout");
        if (histogram_.size() != static_cast<size_t>(max_freq) - min_freq + 1)
            throw std::runtime_error("histogram bucket count mismatch");
    }

    // Consumes an arbitrary byte window; group headers and records may span
    // window boundaries.
    void consume(const uint8_t* data, size_t size)
    {
        size_t pos = 0;
        while (pos < size)
        {
            switch (phase_)
            {
            case Phase::HEADER:
            {
                const size_t take = std::min(msd_prefix::GROUP_HEADER_BYTES - header_have_,
                                             size - pos);
                std::memcpy(header_.data() + header_have_, data + pos, take);
                header_have_ += take;
                pos += take;
                if (header_have_ == msd_prefix::GROUP_HEADER_BYTES) begin_group();
                break;
            }
            case Phase::PREFIX:
            {
                const size_t take = std::min(prefix_remaining_, size - pos);
                prefix_remaining_ -= take;
                pos += take;
                if (!prefix_remaining_) begin_body();
                break;
            }
            case Phase::PAYLOAD:
            {
                const size_t take = static_cast<size_t>(
                    std::min<uint64_t>(payload_remaining_, size - pos));
                payload_remaining_ -= take;
                pos += take;
                if (!payload_remaining_) finish_low_group();
                break;
            }
            case Phase::SUFFIX:
            {
                const size_t take = std::min(suffix_remaining_, size - pos);
                suffix_remaining_ -= take;
                pos += take;
                if (!suffix_remaining_)
                {
                    count_have_ = 0;
                    phase_ = Phase::COUNT;
                }
                break;
            }
            case Phase::COUNT:
            {
                const size_t take = std::min<size_t>(count_bytes_ - count_have_, size - pos);
                std::memcpy(count_.data() + count_have_, data + pos, take);
                count_have_ += take;
                pos += take;
                if (count_have_ == count_bytes_) finish_count();
                break;
            }
            }
        }
    }

    // The stream must end exactly on a group boundary.
    void finish() const
    {
        if (phase_ != Phase::HEADER || header_have_ != 0)
            throw std::runtime_error("truncated MSD group stream");
    }

    uint64_t records() const noexcept
    {
        return records_;
    }

private:
    enum class Phase : uint8_t
    {
        HEADER,
        PREFIX,
        PAYLOAD,
        SUFFIX,
        COUNT
    };

    void begin_group()
    {
        const auto header = msd_prefix::parse_group_header(
            std::span<const uint8_t>(header_), width_);
        prefix_bytes_ = header.prefix_bytes;
        group_records_ = header.records;
        remaining_records_ = header.records;
        suffix_bytes_ = width_ - prefix_bytes_;
        records_ += group_records_;
        header_have_ = 0;

        prefix_remaining_ = prefix_bytes_;
        if (prefix_remaining_) phase_ = Phase::PREFIX;
        else begin_body();
    }

    void begin_body()
    {
        if (count_bytes_ == 0)
        {
            payload_remaining_ = uint64_t(group_records_) * suffix_bytes_;
            phase_ = payload_remaining_ ? Phase::PAYLOAD : Phase::HEADER;
        }
        else
        {
            suffix_remaining_ = suffix_bytes_;
            phase_ = Phase::SUFFIX;
        }
    }

    void finish_low_group()
    {
        if (min_freq_ <= 1 && max_freq_ >= 1)
            histogram_[static_cast<size_t>(1 - min_freq_)] += group_records_;
        phase_ = Phase::HEADER;
    }

    void finish_count()
    {
        uint32_t value = 0;
        std::memcpy(&value, count_.data(), count_bytes_);
        if (value >= min_freq_ && value <= max_freq_)
            histogram_[static_cast<size_t>(value - min_freq_)] += 1;

        count_have_ = 0;
        if (--remaining_records_)
        {
            suffix_remaining_ = suffix_bytes_;
            phase_ = Phase::SUFFIX;
        }
        else
        {
            phase_ = Phase::HEADER;
        }
    }

    const size_t width_;
    const uint32_t count_bytes_;
    const uint32_t min_freq_;
    const uint32_t max_freq_;
    std::vector<int64_t>& histogram_;

    Phase phase_ = Phase::HEADER;
    std::array<uint8_t, msd_prefix::GROUP_HEADER_BYTES> header_{};
    size_t header_have_ = 0;
    size_t prefix_bytes_ = 0;
    size_t prefix_remaining_ = 0;
    size_t suffix_bytes_ = 0;
    size_t suffix_remaining_ = 0;
    uint64_t payload_remaining_ = 0;
    uint32_t group_records_ = 0;
    uint32_t remaining_records_ = 0;
    std::array<uint8_t, sizeof(uint32_t)> count_{};
    uint32_t count_have_ = 0;
    uint64_t records_ = 0;
};

// Reads one high.bin or low.bin through AioGroupReader and accumulates into
// histogram. count_bytes == 0 selects the low path. on_bytes receives each
// consumed window size and may be omitted. Returns the file's record count.
template <typename OnBytes>
inline uint64_t parse_group_file(const std::string& path, size_t width,
                                 uint32_t count_bytes, uint32_t min_freq,
                                 uint32_t max_freq, std::vector<int64_t>& histogram,
                                 OnBytes&& on_bytes)
{
    AioGroupReader reader;
    reader.open(path);
    GroupCounter counter(width, count_bytes, min_freq, max_freq, histogram);

    const uint8_t* data = nullptr;
    size_t size = 0;
    while (reader.next(data, size))
    {
        counter.consume(data, size);
        on_bytes(static_cast<uint64_t>(size));
        reader.release();
    }
    counter.finish();
    reader.join();
    return counter.records();
}

inline uint64_t parse_group_file(const std::string& path, size_t width,
                                 uint32_t count_bytes, uint32_t min_freq,
                                 uint32_t max_freq, std::vector<int64_t>& histogram)
{
    return parse_group_file(path, width, count_bytes, min_freq, max_freq, histogram,
                            [](uint64_t) {});
}

#endif
