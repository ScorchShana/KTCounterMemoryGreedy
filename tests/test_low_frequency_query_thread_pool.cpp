#include "../tool/FlatConcurrentHashMap.h"
#include "../tool/LowFrequencyQueryThreadPool.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace
{
    constexpr uint64_t kRingCapacity = 4;
    constexpr uint64_t kBlockBytes = 512;
    constexpr uint32_t kWords = 2;

    using Kmer = kmer<kWords>;

    Kmer make_kmer(uint64_t first, uint64_t second)
    {
        Kmer value{};
        value.reset();
        value.data[0] = first;
        value.data[1] = second;
        return value;
    }

    uint64_t packed_kmer_bytes_for_k(uint32_t k)
    {
        const uint64_t full_data_count = k / BASES_PER_U64T;
        const uint64_t tail_bits = 2 * (k % BASES_PER_U64T);
        const uint64_t tail_bytes = (tail_bits + 7) / 8;
        return full_data_count * sizeof(uint64_t) + tail_bytes;
    }

    void pack_kmer(uint32_t k, const Kmer& value, char* output)
    {
        const uint64_t full_data_count = k / BASES_PER_U64T;
        const uint64_t tail_bits = 2 * (k % BASES_PER_U64T);
        const uint64_t tail_bytes = (tail_bits + 7) / 8;
        uint64_t offset = 0;

        const uint64_t full_bytes = full_data_count * sizeof(uint64_t);
        if (full_bytes > 0)
        {
            std::memcpy(output, value.data.data(), static_cast<size_t>(full_bytes));
            offset += full_bytes;
        }

        if (tail_bytes > 0)
        {
            const uint64_t mask = (~uint64_t{0}) << (64 - tail_bits);
            uint64_t tail_data = value.data[full_data_count] & mask;
            std::memcpy(
                output + offset,
                reinterpret_cast<const char*>(&tail_data) + (sizeof(uint64_t) - tail_bytes),
                static_cast<size_t>(tail_bytes));
        }
    }

    void init_histogram(std::vector<std::atomic<int64_t>>& histogram)
    {
        for (auto& value : histogram)
        {
            value.store(0, std::memory_order_relaxed);
        }
    }

    bool expect_histogram(
        const std::vector<std::atomic<int64_t>>& histogram,
        const std::vector<int64_t>& expected,
        const char* label)
    {
        for (size_t i = 0; i < expected.size(); ++i)
        {
            const int64_t actual = histogram[i].load(std::memory_order_relaxed);
            if (actual != expected[i])
            {
                std::cerr << label << " histogram mismatch at " << i
                    << ": expected " << expected[i]
                    << ", got " << actual << '\n';
                return false;
            }
        }
        return true;
    }

    bool test_packed_low_frequency_query()
    {
        constexpr uint32_t k = 33;
        constexpr uint32_t min_freq = 1;
        constexpr uint32_t max_freq = 4;
        constexpr size_t hist_size = max_freq - min_freq + 1;

        const Kmer count_two = make_kmer(0x0123456789abcdefULL, 0xc000000000000000ULL);
        const Kmer count_four = make_kmer(0x1111222233334444ULL, 0x8000000000000000ULL);
        const Kmer missing = make_kmer(0x9999aaaabbbbccccULL, 0x4000000000000000ULL);

        FlatConcurrentHashMap<kWords> hash_map(2);
        if (!hash_map.insert_unique(count_two, 2))
        {
            std::cerr << "failed to insert count_two\n";
            return false;
        }
        if (!hash_map.insert_unique(count_four, 4))
        {
            std::cerr << "failed to insert count_four\n";
            return false;
        }
        hash_map.seal();

        std::vector<std::atomic<int64_t>> histogram(hist_size);
        init_histogram(histogram);

        SPMCRingMemoryPool<kRingCapacity> pool(kBlockBytes, 1);
        LowFrequencyQueryThreadPool<kWords, kRingCapacity> threads(
            &pool,
            &hash_map,
            &histogram,
            2,
            k,
            min_freq,
            max_freq,
            hist_size);

        char* block_ptr = nullptr;
        pool.producer_dequeue(block_ptr);
        const uint64_t packed_kmer_bytes = packed_kmer_bytes_for_k(k);
        pack_kmer(k, count_two, block_ptr);
        pack_kmer(k, count_four, block_ptr + packed_kmer_bytes);
        pack_kmer(k, missing, block_ptr + 2 * packed_kmer_bytes);

        pool.producer_enqueue(content_type{block_ptr, 3});
        pool.producer_set_finished();

        threads.start();
        threads.join();

        return expect_histogram(histogram, {1, -1, 1, -1}, "packed low-frequency query");
    }
}

int main()
{
    if (!test_packed_low_frequency_query())
    {
        return 1;
    }

    std::cout << "LowFrequencyQueryThreadPool tests passed\n";
    return 0;
}
