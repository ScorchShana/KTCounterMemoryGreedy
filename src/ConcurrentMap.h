#ifndef CONCURRENT_MAP_HEADER
#define CONCURRENT_MAP_HEADER

#include "definition.h"
#include "kmer.h"
#include "ConcurrentMemoryPool.h"
#include "../include/komihash.h"
#include "../include/xxh3.h"
#include "HashFunction.h"
#include "../include/rapidhash.h"
#include "FinalDrainWriter.h"

#include <array>
#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <vector>
#include <utility>
#include <mutex>

template <uint32_t N>
struct bucket
{
    std::atomic<concurrent_node<N>*> head{ nullptr };
};

inline bool check_is_prime(uint64_t num)
{
    for (uint64_t i = 2; i * i <= num; i++)
    {
        if (num % i == 0)
            return false;
    }
    return true;
}

inline uint64_t get_min_prime(uint64_t num)
{
    if (num % 2 == 0)
        num++;
    while (!check_is_prime(num))
        num += 2;
    return num;
}

template <uint32_t N>
class ConcurrentMap
{

    static_assert(KMER_BLOCK_SIZE >= sizeof(concurrent_node<N>), "KMER_BLOCK_SIZE too small for concurrent_node");

    static constexpr size_t align_up(const size_t value, const size_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static constexpr size_t NODE_SLOT_STRIDE = align_up(sizeof(concurrent_node<N>), alignof(concurrent_node<N>));

    const uint64_t capacity;
    const uint64_t mod;
    ConcurrentMemoryPool* memory_pool = nullptr;
    bucket<N>* buckets;

    // 新的 thread-local 状态（顺序分配 + 单槽回收）
    inline static thread_local int thread_id = -1;
    inline static thread_local std::byte* cur_block_ = nullptr;           // 当前 4KB block
    inline static thread_local uint32_t cur_slot_ = 0;                    // 当前 block 已用槽位
    inline static thread_local concurrent_node<N>* waste_slot_ = nullptr; // CAS 失败回收槽

#ifdef TEST_MODE
    inline static std::mutex mtx;
#endif

public:

    static constexpr size_t SLOT_BLOCK_POINTER_NUM = 4;
    static constexpr size_t BUCKET_SIZE = sizeof(bucket<N>);
    static constexpr size_t SLOT_HEADER_SIZE = sizeof(void*) * SLOT_BLOCK_POINTER_NUM;
    static constexpr size_t SLOTS_PER_BLOCK = (KMER_BLOCK_SIZE - SLOT_HEADER_SIZE) / NODE_SLOT_STRIDE;

    static_assert(SLOTS_PER_BLOCK > 0, "KMER_BLOCK_SIZE too small for aligned concurrent_node slots");

    struct slot_block
    {
        std::array<slot_block*, SLOT_BLOCK_POINTER_NUM> last_blocks{};
        std::array<concurrent_node<N>, SLOTS_PER_BLOCK> slots{};
    };

    using block_pointers_tuple = std::tuple<slot_block*, uint64_t, uint64_t>;
    inline static std::vector<block_pointers_tuple> block_pointers_vector;
    inline static uint32_t k_length;

    // 用于回溯填充 last_blocks 的历史缓冲区（最近 4 个已完成块）
    inline static thread_local std::array<slot_block*, SLOT_BLOCK_POINTER_NUM> tls_recents = { nullptr, nullptr, nullptr, nullptr };


    // 按元素数量分配桶，并记录线程数
    explicit ConcurrentMap(const uint64_t in_capacity, char* bucket_memory, ConcurrentMemoryPool* in_memory_pool)
        : capacity(in_capacity), mod(capacity - 1), memory_pool(in_memory_pool), buckets(reinterpret_cast<bucket<N>*>(bucket_memory))
    {
        for (uint64_t i = 0; i < capacity; ++i)
        {
            new (buckets + i) bucket<N>(); // placement new 构造桶头
        }
    }

    // 析构：释放桶数组
    ~ConcurrentMap() = default;

    static void set_thread_num(uint32_t num_threads)
    {
        block_pointers_vector.resize(num_threads);
        for (uint32_t i = 0;i < num_threads;i++)
        {
            std::get<0>(block_pointers_vector[i]) = nullptr;
            std::get<1>(block_pointers_vector[i]) = 0;
            std::get<2>(block_pointers_vector[i]) = 0;
        }
    }

    static void set_thread_id(uint32_t id)
    {
        thread_id = id;
    }

    static void set_k_length(uint32_t k_len)
    {
        k_length = k_len;
    }

    static void add_thread_node_count(uint64_t count)
    {
        std::get<1>(block_pointers_vector[thread_id]) += count;
    }

    // 哈希：将 k-mer 映射到桶索引
    static uint64_t hash_func(const kmer<N>& k_mer)
    {
        const uint64_t h1 = komihash(&k_mer, sizeof(kmer<N>), 0);
        const uint64_t h2 = rapidhash(&k_mer, sizeof(kmer<N>));
        const uint64_t res = mix_hash(h1, h2);
        return res;
    }

    static void export_thread_node_count(FinalDrainWriter<N>& writer, const int goal_thread_id)
    {
        const uint64_t full_words = k_length / BASES_PER_U64T;
        const uint64_t tail_bits = 2 * (k_length % BASES_PER_U64T);
        uint64_t remaining = std::get<1>(block_pointers_vector[goal_thread_id]);
        slot_block* block_ptr = std::get<0>(block_pointers_vector[goal_thread_id]);
        uint64_t block_count = std::get<2>(block_pointers_vector[goal_thread_id]);

#ifdef TEST_MODE
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "Thread " << goal_thread_id << " has " << remaining << " nodes in " << block_count << " blocks." << std::endl;
        }
#endif

        if (remaining == 0 || block_ptr == nullptr) return;

        // for (int i = 0;i < SLOT_BLOCK_POINTER_NUM;i++)
        // {
        //     if (block_ptr->last_blocks[i] == nullptr) break;
        //     __builtin_prefetch(block_ptr->last_blocks[i], 0, 0);
        // }
        for (int i = 0;i < 2;i++)
        {
            if (block_ptr->last_blocks[i] == nullptr) break;
            __builtin_prefetch(block_ptr->last_blocks[i], 0, 2);
        }

        uint64_t first_block_count = 0;

        if (block_count > 1)
        {
            if ((block_count - 1) * SLOTS_PER_BLOCK < remaining)
            {
                first_block_count = remaining - (block_count - 1) * SLOTS_PER_BLOCK;
            }
            else
            {
                first_block_count = 0;
            }
        }
        else
        {
            first_block_count = remaining;
        }

        if (first_block_count > 0) [[likely]]
        {
            writer.write_map_record(block_ptr->slots.data(), first_block_count);

            remaining -= first_block_count;
        }

        block_ptr = block_ptr->last_blocks[0];
        // if (block_ptr->last_blocks[SLOT_BLOCK_POINTER_NUM - 1] != nullptr)
        // {
        //     __builtin_prefetch(block_ptr->last_blocks[SLOT_BLOCK_POINTER_NUM - 1], 0, 0);
        // }
        if (block_ptr->last_blocks[1] != nullptr)
        {
            __builtin_prefetch(block_ptr->last_blocks[1], 0, 2);
        }


        while (remaining > 0)
        {
            // if (block_ptr->last_blocks[SLOT_BLOCK_POINTER_NUM - 1] != nullptr) [[likely]]
            // {
            //     __builtin_prefetch(block_ptr->last_blocks[SLOT_BLOCK_POINTER_NUM - 1], 0, 0);
            // }
            if (block_ptr->last_blocks[1] != nullptr) [[likely]]
            {
                {
                    __builtin_prefetch(block_ptr->last_blocks[1], 0, 2);
                }
            }

            writer.write_map_record(block_ptr->slots.data(), SLOTS_PER_BLOCK);

            remaining -= SLOTS_PER_BLOCK;
            block_ptr = block_ptr->last_blocks[0];
        }

    }

    void increment(const kmer<N>& k_mer, uint64_t& local_size_count, const uint32_t& count = 1)
    {

        const uint64_t index = hash_func(k_mer) & mod;
        std::atomic<concurrent_node<N>*>& bucket = bucket_head(index);

        concurrent_node<N>* old_chain_first_node = bucket.load(std::memory_order_acquire);
        concurrent_node<N>* last_find_first_node = nullptr;

        for (concurrent_node<N>* cur_node = old_chain_first_node; cur_node != last_find_first_node; cur_node = cur_node->next)
        {
            if (k_mer == cur_node->k_mer)
            {
                {
                    uint32_t cur = cur_node->count.load(std::memory_order_relaxed);
                    if (cur < count_max)
                    {
                        uint32_t prev = cur_node->count.fetch_add(count, std::memory_order_relaxed);
                        if (prev + count < prev || prev + count > count_max) [[unlikely]]
                        {
                            cur_node->count.store(count_max, std::memory_order_relaxed);
                        }
                    }
                }
                return;
            }
        }
        last_find_first_node = old_chain_first_node;

        concurrent_node<N>* new_chain_first_node = nullptr;

        new_chain_first_node = allocate_node();

        new_chain_first_node->k_mer = k_mer;
        new_chain_first_node->count.store(std::min<uint32_t>(count_max, count), std::memory_order_relaxed);

        while (true)
        {
            new_chain_first_node->next = old_chain_first_node;
            if (bucket.compare_exchange_strong(old_chain_first_node, new_chain_first_node,
                std::memory_order_release, std::memory_order_acquire))
            {
                local_size_count++;
                return;
            }

            for (concurrent_node<N>* cur_node = old_chain_first_node; cur_node != last_find_first_node; cur_node = cur_node->next)
            {
                if (k_mer == cur_node->k_mer)
                {
                    {
                        uint32_t cur = cur_node->count.load(std::memory_order_relaxed);
                        if (cur < count_max)
                        {
                            uint32_t prev = cur_node->count.fetch_add(count, std::memory_order_relaxed);
                            if (prev + count < prev || prev + count > count_max) [[unlikely]]
                            {
                                cur_node->count.store(count_max, std::memory_order_relaxed);
                            }
                        }
                    }
                    release_node(new_chain_first_node);
                    return;
                }
            }
            last_find_first_node = old_chain_first_node;

            cpu_relax(); // 短暂暂停，避免过高的缓存行刷新
        }
    }

    void increment(const kmer<N>& k_mer, const uint32_t& count = 1)
    {

        const uint64_t index = hash_func(k_mer) & mod;
        std::atomic<concurrent_node<N>*>& bucket = bucket_head(index);

        concurrent_node<N>* old_chain_first_node = bucket.load(std::memory_order_acquire);
        concurrent_node<N>* last_find_first_node = nullptr;

        for (concurrent_node<N>* cur_node = old_chain_first_node; cur_node != last_find_first_node; cur_node = cur_node->next)
        {
            if (k_mer == cur_node->k_mer)
            {
                {
                    uint32_t cur = cur_node->count.load(std::memory_order_relaxed);
                    if (cur < count_max)
                    {
                        uint32_t prev = cur_node->count.fetch_add(count, std::memory_order_relaxed);
                        if (prev + count < prev || prev + count > count_max) [[unlikely]]
                        {
                            cur_node->count.store(count_max, std::memory_order_relaxed);
                        }
                    }
                }
                return;
            }
        }
        last_find_first_node = old_chain_first_node;

        concurrent_node<N>* new_chain_first_node = nullptr;

        new_chain_first_node = allocate_node();

        new_chain_first_node->k_mer = k_mer;
        new_chain_first_node->count.store(std::min<uint32_t>(count_max, count), std::memory_order_relaxed);

        while (true)
        {
            new_chain_first_node->next = old_chain_first_node;
            if (bucket.compare_exchange_strong(old_chain_first_node, new_chain_first_node,
                std::memory_order_release, std::memory_order_acquire))
            {
                return;
            }

            for (concurrent_node<N>* cur_node = old_chain_first_node; cur_node != last_find_first_node; cur_node = cur_node->next)
            {
                if (k_mer == cur_node->k_mer)
                {
                    {
                        uint32_t cur = cur_node->count.load(std::memory_order_relaxed);
                        if (cur < count_max)
                        {
                            uint32_t prev = cur_node->count.fetch_add(count, std::memory_order_relaxed);
                            if (prev + count < prev || prev + count > count_max) [[unlikely]]
                            {
                                cur_node->count.store(count_max, std::memory_order_relaxed);
                            }
                        }
                    }
                    release_node(new_chain_first_node);
                    return;
                }
            }
            last_find_first_node = old_chain_first_node;

            cpu_relax(); // 短暂暂停，避免过高的缓存行刷新
        }
    }

    template <typename Visitor>
    void debug_visit(Visitor&& visitor) const
    {
        for (uint64_t i = 0; i < capacity; ++i)
        {
            concurrent_node<N>* node = buckets[i].head.load(std::memory_order_acquire);
            while (node != nullptr)
            {
                visitor(node->k_mer, node->count.load(std::memory_order_relaxed));
                node = node->next;
            }
        }
    }

    // 获取桶头指针
    inline std::atomic<concurrent_node<N>*>& bucket_head(const uint64_t index) const noexcept
    {
        return buckets[index].head;
    };

private:
    // 从线程本地 handle 分配节点
    [[nodiscard]] inline concurrent_node<N>* allocate_node()
    {
        // 1. 优先复用 CAS 失败回收的节点
        if (waste_slot_ != nullptr)
        {
            concurrent_node<N>* node = waste_slot_;
            waste_slot_ = nullptr;
            return node;
        }

        // 2. 从 cur_block 顺序分配
        if (cur_block_ == nullptr || cur_slot_ >= SLOTS_PER_BLOCK)
        {
            slot_block* new_block = reinterpret_cast<slot_block*>(memory_pool->allocate());
            cur_slot_ = 0;

            std::get<0>(block_pointers_vector[thread_id]) = new_block;
            std::get<2>(block_pointers_vector[thread_id])++;

            new_block->last_blocks = tls_recents;

            std::memmove(tls_recents.data() + 1, tls_recents.data(), sizeof(slot_block*) * (SLOT_BLOCK_POINTER_NUM - 1));
            tls_recents[0] = new_block;

            cur_block_ = reinterpret_cast<std::byte*>(new_block) + SLOT_HEADER_SIZE; // 跳过头部指针区域
        }

        std::byte* slot = cur_block_ + (cur_slot_ * NODE_SLOT_STRIDE);
        cur_slot_++;
        return reinterpret_cast<concurrent_node<N> *>(slot);
    };

    // 归还节点到线程本地 handle
    inline void release_node(concurrent_node<N>* node) noexcept
    {
        // 单槽位回收：CAS 竞争失败时暂存，供下次分配复用
        if (waste_slot_ == nullptr)
        {
            waste_slot_ = node;
        }
        // 如果槽位已被占用，丢弃节点（理论上不会发生，因为 CAS 失败是串行的）
    }
};
#endif