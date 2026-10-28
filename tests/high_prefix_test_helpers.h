#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

// Independent, test-only decoder. Do not call the production serializer here.
namespace high_prefix_test {
struct Record {
    std::vector<unsigned char> key;
    uint32_t count;
    uint8_t prefix_bytes;
};

inline std::vector<Record> decode(std::span<const uint8_t> bytes, size_t width, size_t count_bytes)
{
    if (!width || width > 32 || !count_bytes || count_bytes > 4)
        throw std::runtime_error("invalid high dimensions");
    std::vector<Record> output;
    while (!bytes.empty()) {
        if (bytes.size() < 5) throw std::runtime_error("truncated high header");
        const size_t prefix = bytes[0];
        uint32_t records;
        std::memcpy(&records, bytes.data() + 1, 4);
        if (prefix >= width) throw std::runtime_error("invalid high prefix");
        const size_t stride = width - prefix + count_bytes;
        if (!records || records > ((128 << 10) - 5 - prefix) / stride)
            throw std::runtime_error("invalid high count");
        const size_t length = 5 + prefix + size_t(records) * stride;
        if (bytes.size() < length) throw std::runtime_error("truncated high payload");
        const uint8_t* common = bytes.data() + 5;
        const uint8_t* source = common + prefix;
        for (uint32_t i = 0; i < records; ++i, source += stride) {
            Record record{std::vector<unsigned char>(width), 0, static_cast<uint8_t>(prefix)};
            std::memcpy(record.key.data(), common, prefix);
            std::memcpy(record.key.data() + prefix, source, width - prefix);
            std::memcpy(&record.count, source + width - prefix, count_bytes);
            output.push_back(std::move(record));
        }
        bytes = bytes.subspan(length);
    }
    return output;
}
} // namespace high_prefix_test
