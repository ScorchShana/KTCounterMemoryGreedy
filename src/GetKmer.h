#ifndef GET_KMER_HEADER
#define GET_KMER_HEADER

#include "kmer.h"
#include <algorithm>

class GetKmer
{
    const uint32_t k = k_length;
    const uint32_t words = (k + 31) / 32;
    const uint32_t insert_shift = (32 - (k % 32)) % 32 * 2;
    const uint64_t back_mask = ~uint64_t{0} << insert_shift;
    const size_t bytes = kmer_bytes();
    const size_t last_bytes = bytes - (words - 1) * 8;
    std::array<uint64_t, 4> seq{};
    std::array<uint64_t, 4> rev{};
    std::array<uint8_t, 256> char_to_2bit{};
    // Skip k-1 valid bases before emitting the first window.
    uint32_t warmup_remaining = k - 1;

    template <uint32_t Words>
    void roll(uint64_t code) noexcept
    {
        for (uint32_t i = Words - 1; i > 0; --i)
            seq[i] = (seq[i] >> 2) | (seq[i - 1] << 62);
        seq[0] = (seq[0] >> 2) | (code << 62);
        seq[Words - 1] &= back_mask;
        for (uint32_t i = 0; i + 1 < Words; ++i)
            rev[i] = (rev[i] << 2) | (rev[i + 1] >> 62);
        rev[Words - 1] = (rev[Words - 1] << 2) | ((code ^ 3) << insert_shift);
    }

    template <uint32_t Words>
    uint64_t write_prefix(unsigned char* output) const noexcept
    {
        uint64_t mask = 0;
        for (uint32_t i = 0; i < Words; ++i) {
            if (seq[i] != rev[i]) {
                mask = -static_cast<uint64_t>(seq[i] < rev[i]);
                break;
            }
        }
        // memcpy permits unaligned compact destinations; words are serialized MSB first.
        for (uint32_t i = 0; i + 1 < Words; ++i) {
            const uint64_t word = __builtin_bswap64((seq[i] & mask) | (rev[i] & ~mask));
            std::memcpy(output + i * 8, &word, 8);
        }
        // Return the serialized last word; the batch selects its store width.
        return __builtin_bswap64((seq[Words - 1] & mask) | (rev[Words - 1] & ~mask));
    }

public:
    GetKmer() noexcept
    {
        char_to_2bit.fill(255);
        for (auto c : {'A', 'a'}) char_to_2bit[c] = 0;
        for (auto c : {'C', 'c'}) char_to_2bit[c] = 1;
        for (auto c : {'G', 'g'}) char_to_2bit[c] = 2;
        for (auto c : {'T', 't', 'U', 'u'}) char_to_2bit[c] = 3;
    }
    void clear() noexcept { seq.fill(0); rev.fill(0); warmup_remaining = k - 1; }
    bool get_next_one(unsigned char base, unsigned char* output) noexcept
    {
        const uint8_t code = char_to_2bit[base];
        if (code > 3) { clear(); return false; }
        return batch_insert(&code, 1, output) != 0;
    }
    // codes contains validated 0..3 bases; output has room for count compact records.
    uint32_t batch_insert(const unsigned char* codes, uint32_t count, unsigned char* output) noexcept
    {
        // Dispatch once per run of bases; word loops can then be unrolled.
        switch (words) {
        case 1: return batch_insert_words<1>(codes, count, output);
        case 2: return batch_insert_words<2>(codes, count, output);
        case 3: return batch_insert_words<3>(codes, count, output);
        default: return batch_insert_words<4>(codes, count, output);
        }
    }

private:
    template <uint32_t Words>
    uint32_t batch_insert_words(const unsigned char* codes, uint32_t count, unsigned char* output) noexcept
    {
        const uint32_t warmup = std::min(count, warmup_remaining);
        for (uint32_t i = 0; i < warmup; ++i) roll<Words>(codes[i]);
        warmup_remaining -= warmup;

        if (warmup == count) return 0;
        const size_t tail_offset = (Words - 1) * 8;
        // Select the tail stores once per batch, outside the output loops.
        switch (last_bytes) {
        case 1:
            for (uint32_t i = warmup; i < count; ++i) {
                roll<Words>(codes[i]);
                const uint64_t word = write_prefix<Words>(output);
                std::memcpy(output + tail_offset, &word, 1);
                output += bytes;
            }
            break;
        case 2:
            for (uint32_t i = warmup; i < count; ++i) {
                roll<Words>(codes[i]);
                const uint64_t word = write_prefix<Words>(output);
                std::memcpy(output + tail_offset, &word, 2);
                output += bytes;
            }
            break;
        case 3:
            for (uint32_t i = warmup; i < count; ++i) {
                roll<Words>(codes[i]);
                const uint64_t word = write_prefix<Words>(output);
                const auto* tail = reinterpret_cast<const unsigned char*>(&word);
                std::memcpy(output + tail_offset, tail, 2);
                std::memcpy(output + tail_offset + 2, tail + 2, 1);
                output += bytes;
            }
            break;
        case 4:
            for (uint32_t i = warmup; i < count; ++i) {
                roll<Words>(codes[i]);
                const uint64_t word = write_prefix<Words>(output);
                std::memcpy(output + tail_offset, &word, 4);
                output += bytes;
            }
            break;
        case 5:
            for (uint32_t i = warmup; i < count; ++i) {
                roll<Words>(codes[i]);
                const uint64_t word = write_prefix<Words>(output);
                const auto* tail = reinterpret_cast<const unsigned char*>(&word);
                std::memcpy(output + tail_offset, tail, 4);
                std::memcpy(output + tail_offset + 4, tail + 4, 1);
                output += bytes;
            }
            break;
        case 6:
            for (uint32_t i = warmup; i < count; ++i) {
                roll<Words>(codes[i]);
                const uint64_t word = write_prefix<Words>(output);
                const auto* tail = reinterpret_cast<const unsigned char*>(&word);
                std::memcpy(output + tail_offset, tail, 4);
                std::memcpy(output + tail_offset + 4, tail + 4, 2);
                output += bytes;
            }
            break;
        case 7:
            for (uint32_t i = warmup; i < count; ++i) {
                roll<Words>(codes[i]);
                const uint64_t word = write_prefix<Words>(output);
                const auto* tail = reinterpret_cast<const unsigned char*>(&word);
                std::memcpy(output + tail_offset, tail, 4);
                std::memcpy(output + tail_offset + 4, tail + 4, 2);
                std::memcpy(output + tail_offset + 6, tail + 6, 1);
                output += bytes;
            }
            break;
        default: // 8 bytes
            for (uint32_t i = warmup; i < count; ++i) {
                roll<Words>(codes[i]);
                const uint64_t word = write_prefix<Words>(output);
                std::memcpy(output + tail_offset, &word, 8);
                output += bytes;
            }
            break;
        }
        return count - warmup;
    }
};
#endif
