#pragma once

#include "../src/kmer.h"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

struct Options
{
    std::string tmp_dir;
    bool is_precise = false;
    uint32_t k_len = 0;
    uint32_t count_max = 0;
    uint32_t count_bytes = 4;
    uint32_t max_threads = 0;
    uint64_t max_memory_bytes = 0;
    std::string output_file;
    uint32_t min_freq = 1;
    uint32_t max_freq = 10000;
};

using AtomicHistogram = std::vector<std::atomic<int64_t>>;

struct RecordLayout
{
    uint32_t full_words;
    uint32_t tail_bytes;
    uint32_t count_bytes;
    uint64_t key_bytes;
    uint64_t record_bytes;

    RecordLayout(uint32_t k_len, uint32_t bytes)
        : full_words(k_len / 32), tail_bytes((2 * (k_len % 32) + 7) / 8),
          count_bytes(bytes), key_bytes(full_words * sizeof(uint64_t) + tail_bytes),
          record_bytes(key_bytes + bytes)
    {
        if (k_len == 0 || k_len > 128 || bytes > 4)
            throw std::runtime_error("invalid compact record layout");
    }
};

template <uint32_t N>
inline void unpack_key(const char* record, const RecordLayout& layout, kmer<N>& key) noexcept
{
    const size_t full_bytes = layout.full_words * sizeof(uint64_t);
    std::memcpy(key.data.data(), record, full_bytes);
    if (layout.tail_bytes)
    {
        uint64_t tail = 0;
        std::memcpy(reinterpret_cast<char*>(&tail) + 8 - layout.tail_bytes,
                    record + full_bytes, layout.tail_bytes);
        key.data[layout.full_words] = tail;
    }
    // The caller zero-initializes each local key once; every payload word is
    // overwritten on each use, while words beyond the layout stay zero.
}

inline uint32_t unpack_count(const char* record, const RecordLayout& layout) noexcept
{
    uint32_t count = 0;
    std::memcpy(&count, record + layout.key_bytes, layout.count_bytes);
    return count;
}

inline bool in_range(uint64_t count, uint32_t min, uint32_t max) noexcept
{
    return count >= min && count <= max;
}

inline size_t histogram_index(uint64_t count, uint32_t min) noexcept
{
    return static_cast<size_t>(count - min);
}

inline void add_frequency(std::vector<int64_t>& histogram, uint64_t count,
                          uint32_t min, uint32_t max, int64_t delta) noexcept
{
    if (in_range(count, min, max)) histogram[histogram_index(count, min)] += delta;
}

inline void correct_histogram(std::vector<int64_t>& histogram, uint32_t count,
                              uint32_t count_max, uint32_t min, uint32_t max) noexcept
{
    add_frequency(histogram, count, min, max, -1);
    add_frequency(histogram, std::min<uint64_t>(static_cast<uint64_t>(count) + 1, count_max),
                  min, max, 1);
}

inline void merge_histogram(const std::vector<int64_t>& local, AtomicHistogram& global) noexcept
{
    for (size_t i = 0; i < local.size(); ++i)
        if (local[i]) global[i].fetch_add(local[i], std::memory_order_relaxed);
}
