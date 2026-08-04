#ifndef CONCURRENT_COUNTING_HASH_MAP_HEADER
#define CONCURRENT_COUNTING_HASH_MAP_HEADER

#include "kmer.h"
#include "definition.h"
#include "SpinLock.h"
#include "ConcurrentMemoryPool.h"
#include "HashFunction.h"

#include <cstdint>
#include <atomic>
#include <vector>
#include <algorithm>
#include <new>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <immintrin.h>
#endif



static constexpr uint8_t get_ctrl_value(const uint64_t hash_value)
{
    return static_cast<uint8_t>((hash_value >> 57) | 0x80u); // 使用哈希值的高8位，并确保最高位为1
}

template <uint32_t N>
struct alignas(PAGE_SIZE) NodeBlock
{
    static_assert(N == 1 || N == 2 || N == 4, "N must be 1, 2, or 4");

#if defined(__AVX2__)
    static constexpr size_t GROUP_SIZE = 32;
#elif defined(__SSE4_2__)
    static constexpr size_t GROUP_SIZE = 16;
#else
    static constexpr size_t GROUP_SIZE = 8;
#endif

    static constexpr size_t kMaxSlots = []()
    {
        if constexpr (N == 1)
            return 312;
        if constexpr (N == 2)
            return 192;
        if constexpr (N == 4)
            return 109;
    }();

    static constexpr size_t kCtrlSize = ((kMaxSlots + 31) / 32) * 32;
    static constexpr size_t kCtrlUint64Size = kCtrlSize / 8;

    std::atomic<NodeBlock<N> *> next = nullptr;
    std::atomic<int> slot_count = 0;
    char padding1[32 - sizeof(std::atomic<NodeBlock<N> *>) - sizeof(std::atomic<int>)]; // 填充至 32 字节

    // 控制字数组（uint8_t，但使用 atomic_ref 进行同步操作）
    alignas(32) std::array<std::atomic<uint64_t>, kCtrlUint64Size> ctrl{};

    // 键数组（定长数组，每个键是 uint64_t[N]）
    std::array<kmer<N>, kMaxSlots> keys{};

    // 值数组（原子计数器）
    std::array<std::atomic<uint32_t>, kMaxSlots> values{};

    char padding2[PAGE_SIZE - sizeof(std::atomic<NodeBlock<N> *>) - sizeof(padding1) - sizeof(std::atomic<int>) - sizeof(ctrl) - sizeof(keys) - sizeof(values)]; // 填充至 4096 字节

public:
#ifdef CONCURRENT_COUNTING_HASH_MAP_TESTING
    template <typename Fn>
    void debug_visit_slots(Fn &&fn) const
    {
        int count = std::min(slot_count.load(std::memory_order_relaxed), static_cast<int>(kMaxSlots));
        for (int i = 0; i < count; ++i)
        {
            uint64_t word = ctrl[i / 8].load(std::memory_order_acquire);
            uint8_t c = static_cast<uint8_t>((word >> ((i % 8) * 8)) & 0xFFu);
            if ((c & 0x80u) != 0)
            {
                fn(keys[i], values[i].load(std::memory_order_relaxed));
            }
        }
    }
#endif

    NodeBlock()
    {
        for (int i = 0; i < values.size(); ++i)
        {
            values[i].store(0, std::memory_order_relaxed);
        }

        for (int i = 0; i < ctrl.size(); ++i)
        {
            ctrl[i].store(0, std::memory_order_relaxed);
        }
        for (int i = kMaxSlots; i < kCtrlSize; ++i)
        {
            ctrl[i / 8].fetch_or(0x7FULL << ((i % 8) * 8), std::memory_order_relaxed); // 超出实际槽位范围的控制字设置为 0x7F，确保它们永远不会被误识为匹配或空槽
        }
    }

    NodeBlock(const NodeBlock &) = delete;
    NodeBlock &operator=(const NodeBlock &) = delete;

    bool try_increment(const kmer<N> &key, uint8_t ctrl_value, const uint32_t increment_value)
    {
        std::array<uint64_t, GROUP_SIZE / 8> ctrl_cache; // 256/128 位控制字的缓存（每 64 位一个 uint64_t）
        int count = std::min(slot_count.load(std::memory_order_relaxed), static_cast<int>(kMaxSlots));
        int cycle_count = (count + GROUP_SIZE - 1) / GROUP_SIZE;
        for (int i = 0; i < cycle_count; ++i)
        {
            int base = i * GROUP_SIZE / 8;
#if defined(__AVX2__)
            for (int j = 0; j < GROUP_SIZE / 8; ++j)
            {
                ctrl_cache[j] = ctrl[base + j].load(std::memory_order_relaxed);
            }
            __m256i mm_ctrl = _mm256_loadu_si256(
                reinterpret_cast<const __m256i *>(ctrl_cache.data()));
            __m256i fp_vec = _mm256_set1_epi8(static_cast<char>(ctrl_value));
            uint32_t match_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(mm_ctrl, fp_vec));

#elif defined(__SSE4_2__)
            for (int j = 0; j < GROUP_SIZE / 8; ++j)
            {
                ctrl_cache[j] = ctrl[base + j].load(std::memory_order_relaxed);
            }
            __m128i mm_ctrl = _mm_loadu_si128(
                reinterpret_cast<const __m128i *>(ctrl_cache.data()));
            __m128i fp_vec = _mm_set1_epi8(static_cast<char>(ctrl_value));
            uint32_t match_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(mm_ctrl, fp_vec));

#else
            for (int j = 0; j < GROUP_SIZE / 8; ++j)
            {
                ctrl_cache[j] = ctrl[base + j].load(std::memory_order_relaxed);
            }
            uint32_t match_mask = 0;
            const uint64_t ctrl_value_64 = ctrl_cache[0];
            for (size_t j = 0; j < GROUP_SIZE; ++j)
            {
                uint8_t c = (ctrl_value_64 >> (j * 8)) & 0xFF;
                if (c == ctrl_value)
                    match_mask |= (1u << j);
            }
#endif
            while (match_mask)
            {
                int bit_pos = __builtin_ctz(match_mask);
                int idx = i * GROUP_SIZE + bit_pos;

                if (((ctrl[base + bit_pos / 8].load(std::memory_order_acquire) >> (8 * (bit_pos % 8))) & 0xFF) == ctrl_value)
                {
                    if (keys[idx] == key) [[likely]]
                    {
                        values[idx].fetch_add(increment_value, std::memory_order_relaxed);
                        return true;
                    }
                }
                match_mask &= (match_mask - 1);
            }
        }
        return false;
    }

    // 在锁内完成插入
    bool insert(const kmer<N> &key, uint8_t ctrl_value, const uint32_t increment_value)
    {
        // 在锁内，可以保证slot_count正确读取
        int idx = slot_count.load(std::memory_order_relaxed);
        if (idx >= kMaxSlots)
        {
            return false; // 当前块已满，插入失败
        }
        slot_count.store(idx + 1, std::memory_order_relaxed); // 增加槽位计数
        keys[idx] = key;
        values[idx].store(increment_value, std::memory_order_relaxed);
        uint64_t cur_ctrl_value = ctrl[idx / 8].load(std::memory_order_relaxed);
        cur_ctrl_value |= static_cast<uint64_t>(ctrl_value) << ((idx % 8) * 8);
        ctrl[idx / 8].store(cur_ctrl_value, std::memory_order_release);
        return true;
    }
};

const int node_block_size = sizeof(NodeBlock<2>); // 4096 字节

template <uint32_t N>
class ConcurrentCountingHashMap
{

    struct Bucket
    {
        SpinLock bucket_lock;
        std::atomic<NodeBlock<N> *> head{nullptr};
    };

    const uint64_t capacity_;
    const uint64_t mod_;
    ConcurrentMemoryPool *memory_pool_;
    Bucket *buckets_;
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> size_{0};

    static_assert(sizeof(NodeBlock<N>) == 4096, "NodeBlock size must be 4KB");

public:
    static constexpr uint64_t BUCKET_SIZE = sizeof(Bucket);

    ConcurrentCountingHashMap(const uint64_t in_capacity, char *bucket_memory, ConcurrentMemoryPool *memory_pool)
        : capacity_(in_capacity), mod_(in_capacity - 1), memory_pool_(memory_pool), buckets_(reinterpret_cast<Bucket *>(bucket_memory))
    {
        // 初始化桶
        for (uint64_t i = 0; i < capacity_; ++i)
        {
            new (&buckets_[i]) Bucket();
        }
    }

#ifdef CONCURRENT_COUNTING_HASH_MAP_TESTING
    template <typename Fn>
    void debug_visit(Fn &&fn) const
    {
        for (uint64_t i = 0; i < capacity_; ++i)
        {
            const Bucket &bucket = buckets_[i];
            NodeBlock<N> *node = bucket.head.load(std::memory_order_acquire);
            while (node)
            {
                node->debug_visit_slots(fn);
                node = node->next.load(std::memory_order_acquire);
            }
        }
    }
#endif

    void increment(const kmer<N> &key, const uint32_t increment_value = 1)
    {
        uint64_t h = hash_func(key);
        uint8_t ctrl_value = get_ctrl_value(h);
        uint64_t bucket_idx = h & mod_;
        Bucket &bucket = buckets_[bucket_idx];
        NodeBlock<N> *head_node = bucket.head.load(std::memory_order_acquire);

        // 1. 尝试无锁增量
        NodeBlock<N> *node = head_node;
        NodeBlock<N> *next_node = nullptr;

        while (node)
        {
            if (node->try_increment(key, ctrl_value, increment_value)) [[likely]]
            {
                return; // 成功增量
            }
            next_node = node->next.load(std::memory_order_acquire);
            if (!next_node)
            {
                break; // 到达链表末尾
            }
            else
            {
                node = next_node;
            }
        }

        // 2. 获取锁并插入
        std::lock_guard<SpinLock> lock(bucket.bucket_lock);

        // 再次检查是否有其他线程已经插入了这个键
        head_node = bucket.head.load(std::memory_order_acquire);

        if (head_node == nullptr)
        {
            // 桶为空，直接插入新节点
            NodeBlock<N> *new_node = reinterpret_cast<NodeBlock<N> *>(memory_pool_->allocate());
            new (new_node) NodeBlock<N>();

            new_node->insert(key, ctrl_value, increment_value);
            bucket.head.store(new_node, std::memory_order_release);
            size_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        node = (node == nullptr) ? head_node : node; // 如果之前没有遍历过链表，先从头节点开始

        next_node = nullptr;
        while (node)
        {
            if (node->try_increment(key, ctrl_value, increment_value))
            {
                return; // 成功增量
            }
            next_node = node->next.load(std::memory_order_acquire);
            if (!next_node)
            {
                break; // 到达链表末尾
            }
            else
            {
                node = next_node;
            }
        }

        if (!node->insert(key, ctrl_value, increment_value))
        {
            NodeBlock<N> *new_node = reinterpret_cast<NodeBlock<N> *>(memory_pool_->allocate());
            new (new_node) NodeBlock<N>();
            new_node->insert(key, ctrl_value, increment_value);
            node->next.store(new_node, std::memory_order_release);
        }
        size_.fetch_add(1, std::memory_order_relaxed);
    }
};

#endif