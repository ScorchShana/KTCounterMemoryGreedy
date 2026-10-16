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

/**
 * @brief 一个 k-mer 在两趟遍历之间的出现次数分类
 *
 *   ONCE  —— 只出现过一次（BF_once 命中、BF_multi 未命中）→ 写盘
 *   MULTI —— 出现过两次及以上（BF_multi 命中）             → 进树
 */
enum class Occurrence : uint8_t
{
    ONCE = 0,
    MULTI = 1,
};

/**
 * @brief 双 Bloom Filter：BF_once(“至少出现过一次”) + BF_multi(“至少出现过两次”)
 *
 * 第一趟遍历调用 insert_prepared()，只更新过滤器、不做任何路由：
 *
 *     BF_multi 命中      → 多次出现，返回 MULTI
 *     否则 BF_once 命中  → 第二次出现，插入 BF_multi，返回 MULTI
 *     否则               → 首次出现，插入 BF_once，返回 ONCE
 *
 * 第二趟遍历调用 query_prepared()，只查询、不再写入：
 *
 *     BF_multi 命中               → 多次出现，进树
 *     BF_once 命中 且 BF_multi 未命中 → 只出现一次，写盘
 *
 * 查询是只读的，所以第二趟的并发/乱序都不会影响判定结果；同时由于第二趟
 * 不再写入，也就不需要任何 per-thread 缓冲或 HashSet 去重。
 *
 * 内存布局：bins[0, capacity_) = BF_once，bins[capacity_, 2*capacity_) = BF_multi。
 */
template <uint32_t N>
class ConcurrentBloomFilter
{

private:
    std::atomic<uint64_t>* bins = nullptr;
    std::atomic<uint64_t>* once_bins = nullptr;  // bins[0, capacity_)
    std::atomic<uint64_t>* multi_bins = nullptr; // bins[capacity_, 2*capacity_)
    uint64_t capacity_;
    uint64_t mod;

public:
    static constexpr uint64_t NUM_HASHES = 3;                          // 使用8个哈希函数
    static constexpr uint64_t BITS_PER_ELEMENT = sizeof(uint64_t) * 8; // 每个元素分配的比特数
    static constexpr uint64_t BITS_MOD = BITS_PER_ELEMENT - 1;         // 用于计算位偏移的掩码

    static constexpr uint64_t SEED_A = 0x9e3779b97f4a7c15ULL; // 黄金比例
    static constexpr uint64_t SEED_B = 0x6a09e667f3bcc909ULL; // √2

    // 一次哈希计算的结果：插入与查询共用，避免第二趟重复算 hash
    struct Probe
    {
        uint64_t block_idx;
        uint64_t bits;
    };

    explicit ConcurrentBloomFilter(size_t in_capacity, ConcurrentMemoryPool* memory_pool)
        : capacity_(in_capacity), mod(capacity_ - 1)
    {
        // capacity_ must be power of 2
        bins = static_cast<std::atomic<uint64_t>*>(
            memory_pool->allocate_large(2 * capacity_ * sizeof(std::atomic<uint64_t>)));
        once_bins = bins;
        multi_bins = bins + capacity_;
        for (size_t i = 0; i < 2 * capacity_; ++i)
        {
            new (&bins[i]) std::atomic<uint64_t>(0); // BF_once 与 BF_multi 全置 0
        }
    }

    ConcurrentBloomFilter(const ConcurrentBloomFilter&) = delete;
    ConcurrentBloomFilter& operator=(const ConcurrentBloomFilter&) = delete;

    ConcurrentBloomFilter(ConcurrentBloomFilter&& other) noexcept
        : bins(other.bins),
        once_bins(other.once_bins),
        multi_bins(other.multi_bins),
        capacity_(other.capacity_),
        mod(other.mod)
    {
        other.bins = nullptr;
        other.once_bins = nullptr;
        other.multi_bins = nullptr;
        other.capacity_ = 0;
        other.mod = 0;
    }

    ConcurrentBloomFilter& operator=(ConcurrentBloomFilter&& other) noexcept
    {
        if (this != &other)
        {
            bins = other.bins;
            once_bins = other.once_bins;
            multi_bins = other.multi_bins;
            capacity_ = other.capacity_;
            mod = other.mod;

            other.bins = nullptr;
            other.once_bins = nullptr;
            other.multi_bins = nullptr;
            other.capacity_ = 0;
            other.mod = 0;
        }
        return *this;
    }

    ~ConcurrentBloomFilter()
    {
    }

    uint64_t capacity() const noexcept { return capacity_; }

    Probe prepare(const kmer<N>& k_mer) const noexcept
    {
        const auto hash_res = XXH3_128bits(&k_mer, sizeof(k_mer));
        return { hash_res.low64 & mod, calculate_bits(hash_res.high64) };
    }

    void prefetch(const Probe& probe) const noexcept
    {
#if defined(__GNUC__) || defined(__clang__)
        __builtin_prefetch(bins + probe.block_idx, 1, 0);
#else
        (void)probe;
#endif
    }

    // ------------------------------------------------------------------
    // 第一趟：更新 BF_once / BF_multi
    //
    // 三个分派都用 fetch_or 的返回值做判定，而不是“先 load 再 fetch_or”，
    // 这样即使两个线程同时插入同一个 k-mer（非 owner 线程会写别人的
    // filter），也恰好只有一个线程拿到 ONCE，另一个拿到 MULTI。
    // ------------------------------------------------------------------
    Occurrence insert_prepared(const Probe& probe) noexcept
    {
        std::atomic<uint64_t>& multi_cell = multi_bins[probe.block_idx];
        std::atomic<uint64_t>& once_cell = once_bins[probe.block_idx];

        // BF_multi 命中：已出现 2 次，本次为第 3 次及以上
        if ((multi_cell.load(std::memory_order_relaxed) & probe.bits) == probe.bits)
        {
            return Occurrence::MULTI;
        }

        // BF_multi 未命中、BF_once 命中：已出现 1 次，本次为第 2 次，插入 BF_multi
        if ((once_cell.load(std::memory_order_relaxed) & probe.bits) == probe.bits)
        {
            multi_cell.fetch_or(probe.bits, std::memory_order_relaxed);
            return Occurrence::MULTI;
        }

        // 都未命中：首次出现，插入 BF_once
        const uint64_t once_before = once_cell.fetch_or(probe.bits, std::memory_order_relaxed);
        if ((once_before & probe.bits) == probe.bits)
        {
            // 竞争：另一个线程已经记录了首次出现，本次其实是第 2 次
            multi_cell.fetch_or(probe.bits, std::memory_order_relaxed);
            return Occurrence::MULTI;
        }
        return Occurrence::ONCE;
    }

    Occurrence insert(const kmer<N>& k_mer) noexcept
    {
        return insert_prepared(prepare(k_mer));
    }

    // ------------------------------------------------------------------
    // 第二趟：只查询，不写入
    // ------------------------------------------------------------------
    bool contains_multi_prepared(const Probe& probe) const noexcept
    {
        return (multi_bins[probe.block_idx].load(std::memory_order_relaxed) & probe.bits) == probe.bits;
    }

    bool contains_once_prepared(const Probe& probe) const noexcept
    {
        return (once_bins[probe.block_idx].load(std::memory_order_relaxed) & probe.bits) == probe.bits;
    }

    // 第二趟路由：MULTI → 进树；ONCE → 写盘
    Occurrence query_prepared(const Probe& probe) const noexcept
    {
        if (contains_multi_prepared(probe))
        {
            return Occurrence::MULTI;
        }
        return Occurrence::ONCE;
    }

    Occurrence query(const kmer<N>& k_mer) const noexcept
    {
        return query_prepared(prepare(k_mer));
    }

private:
    uint64_t calculate_bits(const uint64_t h) const noexcept
    {
        uint64_t bits = 0;
        const uint64_t p0 = h & BITS_MOD;
        bits |= (1ULL << p0);
        const uint64_t p1 = (h >> 21) & BITS_MOD;
        bits |= (1ULL << p1);
        const uint64_t p2 = (h >> 42) & BITS_MOD;
        bits |= (1ULL << p2);

        return bits;
    }
};

#endif
