#ifndef MSD_PREFIX_CODEC_HEADER
#define MSD_PREFIX_CODEC_HEADER

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>

// A file is a sequence of groups, with no file/batch headers or alignment:
// uint8_t P, native uint32_t N, P prefix bytes, N * (W-P) suffix bytes.
// W comes from infos.bin. Integers are copied, never cast from unaligned data.
namespace msd_prefix {
inline constexpr size_t GROUP_HEADER_BYTES = 5;
inline constexpr size_t BUFFER_BYTES = 2ULL << 20;
inline constexpr size_t FINAL_LOW_BUFFER_BYTES = 4 * BUFFER_BYTES;

inline constexpr size_t batch_records(size_t width) noexcept
{
    return (BUFFER_BYTES - GROUP_HEADER_BYTES) / width;
}

namespace detail {
inline uint8_t* encode_group(uint8_t* input, uint8_t* scratch, uint32_t count,
    size_t width, size_t prefix, uint8_t* output)
{
    while (count > 1 && prefix + 1 < width) {
        std::array<uint32_t, 256> counts{};
        std::array<uint32_t, 256> counts1{};
        size_t i = 0;
        for (; i + 1 < count; i += 2) {
            const uint8_t b0 = input[i * width + prefix];
            const uint8_t b1 = input[(i + 1) * width + prefix];
            ++counts[b0];
            ++counts1[b1];
        }
        for (; i < count; ++i) ++counts[input[i * width + prefix]];
        size_t branches = 0;
        for (size_t b = 0; b < counts.size(); ++b) {
            counts[b] += counts1[b];
            branches += counts[b] != 0;
        }
        // split - current = B*(6+P) - (5+P) - N.
        if (branches * (6 + prefix) >= GROUP_HEADER_BYTES + prefix + count) break;
        if (branches == 1) { ++prefix; continue; }

        std::array<uint32_t, 257> offsets{};
        for (size_t b = 0; b < counts.size(); ++b) offsets[b + 1] = offsets[b] + counts[b];
        auto next = offsets;
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* key = input + i * width;
            std::memcpy(scratch + size_t(next[key[prefix]]++) * width, key, width);
        }
        for (size_t b = 0; b < counts.size(); ++b) {
            if (!counts[b]) continue;
            const size_t offset = size_t(offsets[b]) * width;
            output = encode_group(scratch + offset, input + offset, counts[b], width, prefix + 1, output);
        }
        return output;
    }
    *output++ = static_cast<uint8_t>(prefix);
    std::memcpy(output, &count, sizeof(count));
    output += sizeof(count);
    std::memcpy(output, input, prefix);
    output += prefix;
    if (!prefix) {
        std::memcpy(output, input, size_t(count) * width);
        return output + size_t(count) * width;
    }
    const size_t suffix = width - prefix;
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(output, input + i * width + prefix, suffix);
        output += suffix;
    }
    return output;
}
} // namespace detail

// Internal writer contract: width in [1,32], count <= batch_records(width),
// disjoint input/scratch/output, each with BUFFER_BYTES available. Records may
// be reordered. Every split lowers cost, so output <= count*width + 5 <= C.
inline size_t encode(uint8_t* input, uint8_t* scratch, uint32_t count,
    size_t width, uint8_t* output)
{
    if (!count) return 0;
    return detail::encode_group(input, scratch, count, width, 0, output) - output;
}

struct DecodedGroup {
    size_t encoded_bytes;
    uint32_t records;
};

struct GroupHeader {
    size_t prefix_bytes;
    uint32_t records;
    size_t encoded_bytes;
};

// Only the five-byte header must be present; streaming readers fetch the
// prefix and payload separately from their already buffered input.
inline GroupHeader parse_group_header(std::span<const uint8_t> input, size_t width)
{
    if (!width || width > 32) throw std::runtime_error("invalid MSD key width");
    if (input.size() < GROUP_HEADER_BYTES) throw std::runtime_error("truncated MSD group header");
    const size_t prefix = input[0];
    uint32_t count;
    std::memcpy(&count, input.data() + 1, sizeof(count));
    if (prefix >= width) throw std::runtime_error("invalid MSD prefix length");
    if (!count || count > batch_records(width)) throw std::runtime_error("invalid MSD record count");
    return {prefix, count, GROUP_HEADER_BYTES + prefix + size_t(count) * (width - prefix)};
}

// Decode one complete in-memory group, allowing subsequent groups in input.
// The destination holds complete keys, including repeated keys in the stream.
inline DecodedGroup decode_group(std::span<const uint8_t> input, size_t width,
    std::span<uint8_t> output)
{
    const auto header = parse_group_header(input, width);
    const size_t prefix = header.prefix_bytes;
    const uint32_t count = header.records;
    const size_t suffix = width - prefix;
    const size_t bytes = header.encoded_bytes;
    if (input.size() < bytes) throw std::runtime_error("truncated MSD group payload");
    if (output.size() < size_t(count) * width) throw std::runtime_error("MSD decode buffer too small");
    const uint8_t* common = input.data() + GROUP_HEADER_BYTES;
    const uint8_t* source = common + prefix;
    if (!prefix) std::memcpy(output.data(), source, size_t(count) * width);
    else {
        for (size_t i = 0; i < count; ++i) {
            std::memcpy(output.data() + i * width, common, prefix);
            std::memcpy(output.data() + i * width + prefix, source + i * suffix, suffix);
        }
    }
    return {bytes, count};
}
} // namespace msd_prefix
#endif
