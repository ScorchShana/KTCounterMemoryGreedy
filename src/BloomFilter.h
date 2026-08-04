#ifndef BLOOM_FILTER_HEADER
#define BLOOM_FILTER_HEADER

#include "kmer.h"
#include "HashFunction.h"
#include "../include/xxh3.h"
#include "ConcurrentMemoryPool.h"

#include <array>
#include <cstdint>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>

template <uint32_t N>
class BloomFilter
{

private:
    uint64_t* filter_bins = nullptr;
    uint64_t capacity_;
    uint64_t mod;

public:
    static constexpr uint64_t NUM_HASHES = 3;                          // 使用8个哈希函数
    static constexpr uint64_t BITS_PER_ELEMENT = sizeof(uint64_t) * 8; // 每个元素分配的比特数
    static constexpr uint64_t BITS_MOD = BITS_PER_ELEMENT - 1;         // 用于计算位偏移的掩码

    static constexpr uint64_t SEED_A = 0x9e3779b97f4a7c15ULL; // 黄金比例
    static constexpr uint64_t SEED_B = 0x6a09e667f3bcc909ULL; // √2

    struct InsertProbe
    {
        uint64_t block_idx;
        uint64_t insert_num;
    };

    explicit BloomFilter(size_t in_capacity)
        : capacity_(in_capacity), mod(capacity_ - 1)
    {
        // capacity_ must be power of 2
        void* p = nullptr;
        posix_memalign(&p, 64, capacity_ * sizeof(uint64_t));
        if (p == nullptr) [[unlikely]]
        {
            throw std::bad_alloc();
        }
        filter_bins = static_cast<uint64_t*>(p);
        std::memset(filter_bins, 0, capacity_ * sizeof(uint64_t));
        // filter_bins = new uint64_t[2 * capacity_]();
    }

    BloomFilter(const BloomFilter&) = delete;
    BloomFilter& operator=(const BloomFilter&) = delete;

    BloomFilter(BloomFilter&& other) noexcept
        : filter_bins(other.filter_bins),
        capacity_(other.capacity_),
        mod(other.mod)
    {
        other.filter_bins = nullptr;
        other.capacity_ = 0;
        other.mod = 0;
    }

    BloomFilter& operator=(BloomFilter&& other) noexcept
    {
        if (this != &other)
        {
            std::free(filter_bins);
            filter_bins = other.filter_bins;
            capacity_ = other.capacity_;
            mod = other.mod;

            other.filter_bins = nullptr;
            other.capacity_ = 0;
            other.mod = 0;
        }
        return *this;
    }

    ~BloomFilter()
    {
        std::free(filter_bins);
        filter_bins = nullptr;
    }

    InsertProbe prepare_insert(const kmer<N>& k_mer) const noexcept
    {
        const auto hash_res = XXH3_128bits(&k_mer, sizeof(k_mer));
        const uint64_t h1 = hash_res.low64;
        const uint64_t h2 = (hash_res.high64 | 1ULL);

        return { h1 & mod, calculate_insert_num(h1, h2) };
    }

    void prefetch_insert(const InsertProbe& probe) const noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(filter_bins + probe.block_idx, 1, 1);
#else
        (void)probe;
#endif
    }

    bool insert_prepared(const InsertProbe& probe) noexcept
    {
        uint64_t* __restrict bins = filter_bins;
        const uint64_t old_val = bins[probe.block_idx];
        bins[probe.block_idx] = old_val | probe.insert_num;
        return (old_val & probe.insert_num) != probe.insert_num; // kmer not exists in first filters
    }

    bool insert(const kmer<N>& k_mer) noexcept
    {
        const InsertProbe probe = prepare_insert(k_mer);
        return insert_prepared(probe);
    }

    uint64_t* get_filter_bins()
    {
        return filter_bins;
    }

private:
    uint64_t calculate_insert_num(const uint64_t h1, const uint64_t h2)const noexcept
    {
        uint64_t insert_num = 0;

#pragma unroll
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            insert_num |= (1ULL << ((h1 + i * h2) & BITS_MOD));
        }
        return insert_num;
    }
};


template <uint32_t N>
class ConcurrentBloomFilter
{

private:
    std::atomic<uint64_t>* filter_bins = nullptr;
    uint64_t capacity_;
    uint64_t mod;

public:
    static constexpr uint64_t NUM_HASHES = 3;                          // 使用8个哈希函数
    static constexpr uint64_t BITS_PER_ELEMENT = sizeof(uint64_t) * 8; // 每个元素分配的比特数
    static constexpr uint64_t BITS_MOD = BITS_PER_ELEMENT - 1;         // 用于计算位偏移的掩码

    static constexpr uint64_t SEED_A = 0x9e3779b97f4a7c15ULL; // 黄金比例
    static constexpr uint64_t SEED_B = 0x6a09e667f3bcc909ULL; // √2

    struct InsertProbe
    {
        uint64_t block_idx;
        uint64_t insert_num;
    };

    explicit ConcurrentBloomFilter(size_t in_capacity, ConcurrentMemoryPool* memory_pool)
        : capacity_(in_capacity), mod(capacity_ - 1)
    {
        // capacity_ must be power of 2
        filter_bins = static_cast<std::atomic<uint64_t>*>(memory_pool->allocate_large(capacity_ * sizeof(std::atomic<uint64_t>)));
        for (size_t i = 0; i < capacity_; ++i)
        {
            new (&filter_bins[i]) std::atomic<uint64_t>(0); // 值初始化
        }
    }

    ConcurrentBloomFilter(const ConcurrentBloomFilter&) = delete;
    ConcurrentBloomFilter& operator=(const ConcurrentBloomFilter&) = delete;

    ConcurrentBloomFilter(ConcurrentBloomFilter&& other) noexcept
        : filter_bins(other.filter_bins),
        capacity_(other.capacity_),
        mod(other.mod)
    {
        other.filter_bins = nullptr;
        other.capacity_ = 0;
        other.mod = 0;
    }

    ConcurrentBloomFilter& operator=(ConcurrentBloomFilter&& other) noexcept
    {
        if (this != &other)
        {
            filter_bins = other.filter_bins;
            capacity_ = other.capacity_;
            mod = other.mod;

            other.filter_bins = nullptr;
            other.capacity_ = 0;
            other.mod = 0;
        }
        return *this;
    }

    ~ConcurrentBloomFilter()
    {
    }

    InsertProbe prepare_insert(const kmer<N>& k_mer) const noexcept
    {
        const auto hash_res = XXH3_128bits(&k_mer, sizeof(k_mer));
        const uint64_t h1 = hash_res.low64;
        const uint64_t h2 = (hash_res.high64 | 1ULL);

        return { h1 & mod, calculate_insert_num(h1, h2) };
    }

    void prefetch_insert(const InsertProbe& probe) const noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(filter_bins + probe.block_idx, 1, 0);
#else
        (void)probe;
#endif
    }

    bool insert_prepared(const InsertProbe& probe) noexcept
    {
        const uint64_t first_snapshot = filter_bins[probe.block_idx].load(std::memory_order_relaxed);
        if ((first_snapshot & probe.insert_num) == probe.insert_num)
        {
            return false;// kmer exists in first filters
        }
        const uint64_t old_val = filter_bins[probe.block_idx].fetch_or(probe.insert_num, std::memory_order_relaxed);
        return (old_val & probe.insert_num) != probe.insert_num; // kmer not exists in first filters
    }

    bool insert(const kmer<N>& k_mer) noexcept
    {
        const InsertProbe probe = prepare_insert(k_mer);
        return insert_prepared(probe); 
    }

    std::atomic<uint64_t>* get_filter_bins()
    {
        return filter_bins;
    }

private:
    uint64_t calculate_insert_num(const uint64_t h1, const uint64_t h2)const noexcept
    {
        uint64_t insert_num = 0;

#pragma unroll
        for (uint64_t i = 0; i < NUM_HASHES; i++)
        {
            insert_num |= (1ULL << ((h1 + i * h2) & BITS_MOD));
        }
        return insert_num;
    }
};

#endif
