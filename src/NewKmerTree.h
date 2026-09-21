#ifndef KMER_TREE_HEADER
#define KMER_TREE_HEADER

#include "definition.h"
#include "kmer.h"
#include "ConcurrentMemoryPool.h"
#include "LayerQueues.h"
#include "MPMCRingQueue.h"
#include "FixedStack.h"
#include "FinalDrainWriter.h"
#include "SpinLock.h"
#include "BloomFilter.h"
#include "RingMemoryPool.h"
#include "CountingHashMap.h"
#include "ConcurrentOpenAddressHashMap.h"
#include "ConcurrentMapWriter.h"
#include "SplitMix.h"

#include <atomic>
#include <vector>
#include <memory>
#include <new>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <fstream>
#include <string>
#include <algorithm>
#include <array>
#include <mutex>
#include <map>

template <uint32_t N>
struct node
{
    // cache line 0-2 for write
    // SpinLock buffer_lock;
    SpinLock buffer_lock;
    std::atomic<int> writer_count{ 0 }; // 当前正在写入的线程数
    std::atomic<ConcurrentOpenAddressHashMap<N>*> hash_map{ nullptr };
    kmer_block<N>* active_block = nullptr;
    uint64_t count = 0; // counter for used block
    std::array<kmer_block<N>*, MAX_KMER_BLOCK_NUM> kmer_blocks{};
    // char padding[3 * CACHE_LINE_SIZE - 3 * sizeof(uint64_t) - sizeof(std::array<kmer_block<N> *, MAX_KMER_BLOCK_NUM>)];

    // cache line 3 for read
    alignas(CACHE_LINE_SIZE) std::atomic<node<N>*> children_ptr = nullptr;
};

// const int node_size = sizeof(node<1>);

template <uint32_t N>
class KmerTree
{

    static constexpr size_t align_up(const size_t value, const size_t alignment)
    {
        return (value + alignment - 1) & ~(alignment - 1);
    }

    static constexpr uint64_t COUNTING_HASH_TABLE_SIZE = 256 * 1024;
    static constexpr uint64_t KMER_BUFFER_CAPACITY = KMER_BLOCK_SIZE / sizeof(kmer<N>);

    static constexpr int WRITER_WAITING_MAX_BACKOFF = 64;
    static constexpr int WRITER_WAITING_SPIN_TIME = 256;

    static constexpr uint32_t TASK_ENQUEUE_RETRY_LIMIT = 2;
    static constexpr uint64_t SCATTER_BLOCK_BATCH_SIZE = 4;

    struct DrainFrame
    {
        node<N>* node_ptr;
        uint32_t depth;
    };

    struct MergeCursor
    {
        kmer_block<N>* block;
        uint64_t index;
    };

    uintptr_t MAGIC_POINTER = 0x1; // 用于标记特殊指针（如正在构建的 Bloom Filter）

    // k-mer 序列实际长度
    uint32_t k_length;
    // 跨线程、跨深度的层级队列
    LayerQueues<N>* layer_queue_;
    // 给节点分配空间使用的全局并发内存池
    ConcurrentMemoryPool* memory_pool;
    // 用于给负责导出的流提供临时数据块的内存池
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* export_ring_pool;

    // 以下是利用 thread_local 防止多线程互斥开销的临时统计数组
    static inline thread_local std::array<uint32_t, 1ULL << (2 * NODE_BASES)> thread_local_block_prefix_counts{};
    static inline thread_local std::array<uint32_t, 1ULL << (2 * NODE_BASES)> thread_local_block_prefix_sums{};
    static inline thread_local std::array<kmer<N>, SCATTER_BLOCK_BATCH_SIZE* KMER_BLOCK_SIZE / sizeof(kmer<N>)> thread_local_block_for_copy{};
    // 本地任务缓存栈，避免频繁向全局队列 push/pop
    static inline thread_local std::vector<Task<N>> thread_local_task_stack;
    static inline thread_local std::vector<ExportRecord<N>> thread_local_export_buffer;
    static inline thread_local CountingHashMap<N> thread_local_counting_hash_map;
    // 提前分配的spare block
    static inline thread_local char* thread_local_spare_block = nullptr;

    static inline thread_local SpinBackoff<> thread_local_spin_backoff;

    static inline thread_local SplitMix64 rng;

    static inline thread_local uint64_t thread_local_kmers_in_map = 0;

    static inline thread_local SpinBackoff<32, 64, 64 + 8> classifier_enqueue_spin_backoff;

public:
    // 根节点数组，2^(2 * ROOT_BASES) 个，每个对应一种短前缀
    node<N>* root_nodes;

#ifdef TEST_MODE
    alignas(CACHE_LINE_SIZE) std::atomic<long long> total_kmers_added{ 0 };
    static inline thread_local uint64_t classifier_wait_cycles = 0;
    static inline thread_local uint32_t dealing_root_index = 0;
#endif

    // 构造函数：初始化字典树相关组件
    explicit KmerTree(uint32_t k_len, ConcurrentMemoryPool* in_memory_pool, LayerQueues<N>* in_layer_queue, RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* in_export_ring_pool)
        : k_length(k_len), layer_queue_(in_layer_queue), memory_pool(in_memory_pool), export_ring_pool(in_export_ring_pool)
    {
        root_nodes = new node<N>[1ULL << (2 * ROOT_BASES)]();

        constexpr uint64_t node_num = 1ULL << (2 * NODE_BASES);
        for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); ++i)
        {
            // 分配 (1 << (2 * NODE_BASES))个连续节点
            void* raw_mem = memory_pool->allocate();
            new (raw_mem) node<N>[node_num]();
            root_nodes[i].children_ptr.store(reinterpret_cast<node<N> *>(raw_mem), std::memory_order_relaxed);
        }
    }

    ~KmerTree()
    {
        delete[] root_nodes;
    }

    inline RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* get_export_ring_pool() const noexcept
    {
        return export_ring_pool;
    }

    inline uint32_t get_k_length() const noexcept
    {
        return k_length;
    }

    // 主线程添加函数
    void main_add_kmer(const kmer<N>& kmer_val)
    {
        // 1. 计算路由索引 (前 ROOT_BASES 个碱基)
        // 假设 kmer 数据高位在 data[0]
        const uint64_t root_idx = get_root_prefix(kmer_val);

        node<N>& target_node = root_nodes[root_idx];

        bool has_deferred_task = false;
        Task<N> deferred_task;

        {
            std::lock_guard lock(target_node.buffer_lock);

            if (target_node.active_block == nullptr)
            {
                // 未初始化
                target_node.active_block = reinterpret_cast<kmer_block<N> *>(memory_pool->allocate());
                target_node.active_block->count = 0;
                target_node.kmer_blocks[target_node.count++] = target_node.active_block;
            }

            else if (target_node.active_block->count >= get_block_capacity())
            {

                if (target_node.count >= MAX_KMER_BLOCK_NUM)
                {
                    // Capture task metadata under lock, then enqueue after lock release.
                    deferred_task.current_node = &target_node;
                    deferred_task.depth = 0;
                    deferred_task.kmer_blocks = target_node.kmer_blocks;
                    deferred_task.count = target_node.count;
                    has_deferred_task = true;

                    target_node.count = 1;
                    target_node.active_block = reinterpret_cast<kmer_block<N> *>(memory_pool->allocate());
                    target_node.kmer_blocks[0] = target_node.active_block;
                    target_node.active_block->count = 0;
                }
                else
                {
                    // 申请一个新块写入
                    target_node.active_block = reinterpret_cast<kmer_block<N> *>(memory_pool->allocate());
                    target_node.kmer_blocks[target_node.count++] = target_node.active_block;
                    target_node.active_block->count = 0;
                }
            }

            // 5. 未满直接写入
            target_node.active_block->k_mers[target_node.active_block->count++] = kmer_val;
        }

        if (has_deferred_task)
        {
            auto queue_ptr = layer_queue_->get_queue(0);
            layer_queue_->increase_size(0);
            queue_ptr->enqueue(deferred_task);
        }
    }

    void flush_local_root_nodes(node<N>* local_root_nodes, uint64_t start_root_node_index)
    {

        Task<N> task{};
        constexpr size_t capacity = get_block_capacity();
        constexpr uint64_t mod = (1ULL << (2 * ROOT_BASES)) - 1;

        auto queue_ptr = layer_queue_->get_queue(0);

        for (uint64_t k = 0; k < (1ULL << (2 * ROOT_BASES)); k++)
        {

            uint64_t i = (start_root_node_index + k) & mod;

            node<N>* target_root = &root_nodes[i];
            node<N>* local_root = &local_root_nodes[i];

            task.current_node = target_root;
            task.depth = 0;

            for (uint64_t block_index = 0; block_index < local_root->count; ++block_index)
            {
                bool enqueue_required = false;

                uint64_t remaining = local_root->kmer_blocks[block_index]->count;
                uint64_t block_for_copy_offset = 0;
                uint64_t block_original_count;
                uint64_t space_left;
                uint64_t this_copy;
                kmer_block<N>* active_block;

                while (remaining > 0)
                {
                    ensure_spare_block();

                    {
                        std::lock_guard lock(target_root->buffer_lock);
                        target_root->writer_count.fetch_add(1, std::memory_order_relaxed);
                        active_block = target_root->active_block;

                        if (active_block == nullptr) [[unlikely]]
                        {
                            active_block = reinterpret_cast<kmer_block<N> *>(get_spare_block());
                            active_block->count = 0;
                            target_root->active_block = active_block;
                            target_root->kmer_blocks[target_root->count++] = target_root->active_block;
                        }
                        else if (active_block->count >= capacity) [[unlikely]]
                        {
                            if (target_root->count >= MAX_KMER_BLOCK_NUM) [[unlikely]]
                            {
                                // node已满
                                task.kmer_blocks = target_root->kmer_blocks;
                                task.count = target_root->count;

                                enqueue_required = true;

                                target_root->count = 0;

                                active_block = reinterpret_cast<kmer_block<N> *>(get_spare_block());
                                active_block->count = 0;
                                target_root->active_block = active_block;
                                target_root->kmer_blocks[target_root->count++] = target_root->active_block;

                                thread_local_spin_backoff.reset();

                                while (target_root->writer_count.load(std::memory_order_acquire) > 1)
                                {
                                    thread_local_spin_backoff.backoff();
                                }
                            }
                            else
                            {
                                active_block = reinterpret_cast<kmer_block<N> *>(get_spare_block());
                                active_block->count = 0;
                                target_root->active_block = active_block;
                                target_root->kmer_blocks[target_root->count++] = target_root->active_block;
                            }
                        }

                        block_original_count = active_block->count;
                        space_left = capacity - block_original_count;
                        this_copy = static_cast<uint64_t>((space_left < remaining) ? space_left : remaining);
                        active_block->count += this_copy;
                    }

                    std::memcpy(active_block->k_mers.data() + block_original_count,
                        local_root->kmer_blocks[block_index]->k_mers.data() + block_for_copy_offset,
                        static_cast<size_t>(this_copy) * sizeof(kmer<N>));

                    block_for_copy_offset += this_copy;
                    remaining -= this_copy;
                    target_root->writer_count.fetch_sub(1, std::memory_order_release);

                    if (enqueue_required) [[unlikely]]
                    {
                        enqueue_required = false;
                        // 必须正常入队到0层队列
#ifdef TEST_MODE
                        uint64_t start_cycles = __rdtsc();
#endif

                        layer_queue_->increase_size(0);
                        queue_ptr->enqueue(task);

#ifdef TEST_MODE
                        uint64_t end_cycles = __rdtsc();
                        classifier_wait_cycles += end_cycles - start_cycles;
#endif
                    }
                }
                memory_pool->deallocate(local_root->kmer_blocks[block_index]);
            }
        }
        release_spare_block();
    }

    void main_add_kmer_block_with_local_root_nodes(std::array<kmer<N>,
        PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>)>& kmer_block_for_copy,
        std::array<uint32_t, 1ULL << (2 * ROOT_BASES)>& kmer_prefix_counts,
        const std::array<uint32_t, 1ULL << (2 * ROOT_BASES)>& prefix_ordered_by_owner,
        const uint32_t cnt,
        node<N>* local_root_nodes)
    {
        Task<N> task{};
        uint32_t local_increase_count = 0;

        constexpr uint32_t capacity = get_block_capacity();
        uint64_t read_offset = 0;
        auto queue_ptr = layer_queue_->get_queue(0);

        for (uint64_t index = 0; index < cnt; index++)
        {
            uint64_t i = prefix_ordered_by_owner[index];

            node<N>* target_root = &local_root_nodes[i];

            uint64_t remaining = kmer_prefix_counts[i];

            kmer_block<N>* active_block = target_root->active_block;

            if (active_block == nullptr) [[unlikely]]
            {
                active_block = reinterpret_cast<kmer_block<N> *>(memory_pool->allocate());
                active_block->count = 0;
                target_root->active_block = active_block;
                target_root->kmer_blocks[target_root->count++] = target_root->active_block;
            }

            const uint64_t first_block_original_count = active_block->count;
            const uint64_t first_space_left = capacity - first_block_original_count;
            const uint64_t first_copy_this_time = (first_space_left < remaining) ? first_space_left : remaining;
            remaining -= first_copy_this_time;
            std::memcpy(active_block->k_mers.data() + first_block_original_count,
                kmer_block_for_copy.data() + read_offset,
                static_cast<size_t>(first_copy_this_time) * sizeof(kmer<N>));
            read_offset += first_copy_this_time;
            active_block->count += first_copy_this_time;

#ifdef TEST_MODE
            total_kmers_added.fetch_add(first_copy_this_time, std::memory_order_relaxed);
#endif

            while (remaining > 0)
            {
                const uint64_t copy_this_time = (capacity < remaining) ? capacity : remaining;
                remaining -= copy_this_time;

                if (target_root->count >= MAX_KMER_BLOCK_NUM) [[unlikely]]
                {
                    // node已满
                    task.current_node = root_nodes + i;
                    task.depth = 0;
                    task.kmer_blocks = target_root->kmer_blocks;
                    task.count = target_root->count;

#ifdef TEST_MODE
                    uint64_t start_cycles = __rdtsc();
#endif

                    ++local_increase_count;
                    if (queue_ptr->try_enqueue(task))
                    {
                        classifier_enqueue_spin_backoff.reset();
                    }
                    else
                    {
                        classifier_enqueue_spin_backoff.backoff();
                        while (!queue_ptr->try_enqueue(task))
                        {
                            classifier_enqueue_spin_backoff.backoff();
                        }
                        classifier_enqueue_spin_backoff.decay();
                    }

#ifdef TEST_MODE
                    uint64_t end_cycles = __rdtsc();
                    classifier_wait_cycles += end_cycles - start_cycles;
#endif

                    target_root->count = 0;
                }

                active_block = reinterpret_cast<kmer_block<N> *>(memory_pool->allocate());
                active_block->count = 0;
                target_root->active_block = active_block;
                target_root->kmer_blocks[target_root->count++] = target_root->active_block;

                std::memcpy(active_block->k_mers.data(),
                    kmer_block_for_copy.data() + read_offset,
                    static_cast<size_t>(copy_this_time) * sizeof(kmer<N>));

                active_block->count = copy_this_time;

                read_offset += copy_this_time;

#ifdef TEST_MODE
                total_kmers_added.fetch_add(copy_this_time, std::memory_order_relaxed);
#endif
            }
        }
        if (local_increase_count > 0)
        {
            layer_queue_->increase_size(0, local_increase_count);
        }
    }

    // 线程池工作线程函数
    void thread_add_kmer(const Task<N>& current_task)
    {
        node<N>* current_node = current_task.current_node;

        if (current_task.depth >= MAX_DEPTH - 1)
        {
            insert_kmer_in_task_to_node_hash_map_with_local_hash_map(current_task);
            return;
        }

        // 继续下放
        scatter_kmer_blocks_to_children(current_task);
        // for (int block_index = 0; block_index < current_task.count; ++block_index)
        // {
        //     calculate_block_prefix_counts(current_task.kmer_blocks[block_index], current_task.depth);
        //     push_kmers_into_thread_local_block_for_copy(current_task.kmer_blocks[block_index], current_task.depth);
        //     flush_block_to_children(current_node, current_task.depth);
        // }

        for (int block_index = 0; block_index < current_task.count; ++block_index)
        {
            memory_pool->deallocate(current_task.kmer_blocks[block_index]);
        }
    }

    size_t get_local_stack_size() const
    {
        return thread_local_task_stack.size();
    }

    std::size_t deal_with_local_stack(std::size_t max_task = 64)
    {
        std::size_t processed_tasks = 0;
        while (!thread_local_task_stack.empty() && processed_tasks < max_task)
        {
            Task<N> task = thread_local_task_stack.back();
            thread_local_task_stack.pop_back();
            thread_add_kmer(task);
            processed_tasks++;
        }
        return processed_tasks;
    }

    bool check_and_deal_with_local_stack()
    {
        bool res = thread_local_task_stack.size() > 0;
        while (!thread_local_task_stack.empty())
        {
            Task<N> task = thread_local_task_stack.back();
            thread_local_task_stack.pop_back();
            thread_add_kmer(task);
        }
        return res;
    }

    void mark_finish_export()
    {
        export_ring_pool->producer_set_finished();
    }

    ConcurrentMemoryPool* get_memory_pool() const noexcept
    {
        return memory_pool;
    }

private:
    void insert_kmer_in_task_to_node_hash_map_with_local_hash_map(const Task<N>& current_task)
    {
        node<N>* parent = current_task.current_node;


        uint64_t local_size_count = 0;

        thread_local_counting_hash_map.clear();
        for (uint64_t block_index = 0; block_index < current_task.count; ++block_index)
        {
            kmer_block<N>* input_kmer_block = current_task.kmer_blocks[block_index];

            for (uint64_t i = 0; i < input_kmer_block->count; ++i)
            {
                thread_local_counting_hash_map.increment(input_kmer_block->k_mers[i]);
            }
        }

        uint64_t hash_map_capacity = std::bit_ceil(thread_local_counting_hash_map.size() * 5 / 4);
        hash_map_capacity = std::max<uint64_t>(hash_map_capacity, concurrent_hash_map_min_capacity);
        hash_map_capacity = std::min<uint64_t>(hash_map_capacity, concurrent_hash_map_max_capacity);
        ConcurrentOpenAddressHashMap<N>* hash_map = ensure_hash_map(parent, hash_map_capacity);

        if (hash_map == nullptr) [[unlikely]]
        {
            thread_local_task_stack.push_back(current_task);
            return;
        }

        flush_local_counting_hash_map_to_hash_map(hash_map, local_size_count);

        for (uint64_t block_index = 0; block_index < current_task.count; ++block_index) {
            kmer_block<N>* input_kmer_block = current_task.kmer_blocks[block_index];
            memory_pool->deallocate(input_kmer_block);
        }

        thread_local_kmers_in_map += local_size_count;
    }

    void insert_kmer_in_task_to_node_hash_map_without_local_hash_map(const Task<N>& current_task)
    {
        node<N>* parent = current_task.current_node;
        // const uint64_t root_prefix = get_root_prefix(current_task.kmer_blocks[0]->k_mers[0]);
        ConcurrentOpenAddressHashMap<N>* hash_map = ensure_hash_map(parent, concurrent_hash_map_min_capacity);

        if (hash_map == nullptr) [[unlikely]]
        {
            thread_local_task_stack.push_back(current_task);
            return;
        }

        uint64_t local_size_count = 0;

        for (uint64_t block_index = 0; block_index < current_task.count; ++block_index)
        {
            kmer_block<N>* input_kmer_block = current_task.kmer_blocks[block_index];

            for (uint64_t i = 0; i < input_kmer_block->count; ++i)
            {
                hash_map->increment(input_kmer_block->k_mers[i], 1, local_size_count);
            }
            memory_pool->deallocate(input_kmer_block);
        }
        thread_local_kmers_in_map += local_size_count;
    }

    void scatter_kmer_blocks_to_children(const Task<N>& current_task)
    {
        uint64_t cur_block_index = 0;

        while (cur_block_index < current_task.count)
        {
            const uint32_t depth = current_task.depth;

            thread_local_block_prefix_counts.fill(0);

            const uint64_t block_index_boundary = std::min<uint64_t>(
                current_task.count, cur_block_index + SCATTER_BLOCK_BATCH_SIZE);
            for (uint64_t block_index = cur_block_index; block_index < block_index_boundary; block_index++)
            {
                kmer_block<N>* block_ptr = current_task.kmer_blocks[block_index];
                for (uint64_t index = 0; index < block_ptr->count; index++)
                {
                    const uint64_t prefix = get_node_prefix(block_ptr->k_mers[index], depth);
                    thread_local_block_prefix_counts[prefix]++;
                }
            }

            thread_local_block_prefix_sums[0] = 0;
            for (uint64_t i = 1; i < thread_local_block_prefix_counts.size(); ++i)
            {
                thread_local_block_prefix_sums[i] = thread_local_block_prefix_sums[i - 1] + thread_local_block_prefix_counts[i - 1];
            }

            for (uint64_t block_index = cur_block_index; block_index < block_index_boundary; block_index++)
            {
                kmer_block<N>* block_ptr = current_task.kmer_blocks[block_index];
                for (uint64_t index = 0; index < block_ptr->count; index++)
                {
                    const uint64_t prefix = get_node_prefix(block_ptr->k_mers[index], depth);
                    const uint64_t pos = thread_local_block_prefix_sums[prefix];
                    thread_local_block_for_copy[pos] = block_ptr->k_mers[index];
                    thread_local_block_prefix_sums[prefix]++;
                }
            }
            cur_block_index = block_index_boundary;

            flush_block_to_children(current_task.current_node, current_task.depth);
        }

    }

    void calculate_block_prefix_counts(kmer_block<N>* block_ptr, const uint32_t depth)
    {
        thread_local_block_prefix_counts.fill(0);
        for (uint64_t index = 0; index < block_ptr->count; index++)
        {
            const uint64_t prefix = get_node_prefix(block_ptr->k_mers[index], depth);
            thread_local_block_prefix_counts[prefix]++;
        }

        thread_local_block_prefix_sums[0] = 0;
        for (uint64_t i = 1; i < thread_local_block_prefix_counts.size(); ++i)
        {
            thread_local_block_prefix_sums[i] = thread_local_block_prefix_sums[i - 1] + thread_local_block_prefix_counts[i - 1];
        }
    }

    void drain_part_of_batch_to_child(node<N>* child_node, const uint64_t prefix, const uint64_t in_block_for_copy_offset, const uint32_t current_depth, std::vector<Task<N>>& drain_stack)
    {
        uint64_t remaining = thread_local_block_prefix_counts[prefix];
        uint64_t block_for_copy_offset = in_block_for_copy_offset;
        constexpr uint32_t capacity = get_block_capacity();

        while (remaining > 0)
        {
            kmer_block<N>* active_block = child_node->active_block;
            if (active_block == nullptr || active_block->count >= capacity) [[unlikely]]
            {
                if (child_node->count >= MAX_KMER_BLOCK_NUM) [[unlikely]]
                {
                    Task<N> new_task{};
                    new_task.current_node = child_node;
                    new_task.depth = current_depth + 1;
                    new_task.count = child_node->count;
                    new_task.kmer_blocks = child_node->kmer_blocks;
                    drain_stack.push_back(new_task);

                    child_node->count = 0;
                    child_node->active_block = nullptr;
                }

                active_block = reinterpret_cast<kmer_block<N> *>(memory_pool->allocate());
                active_block->count = 0;
                child_node->active_block = active_block;
                child_node->kmer_blocks[child_node->count++] = active_block;
            }

            const uint64_t block_original_count = active_block->count;
            const uint64_t this_copy = std::min<uint64_t>(capacity - block_original_count, remaining);

            std::memcpy(active_block->k_mers.data() + block_original_count,
                thread_local_block_for_copy.data() + block_for_copy_offset,
                static_cast<size_t>(this_copy) * sizeof(kmer<N>));

            active_block->count += this_copy;
            block_for_copy_offset += this_copy;
            remaining -= this_copy;
        }
    }

    void scatter_drain_kmer_blocks_to_children(const Task<N>& task, node<N>* child_node_base, std::vector<Task<N>>& drain_stack)
    {
        uint64_t cur_block_index = 0;
        const uint32_t depth = static_cast<uint32_t>(task.depth);

        while (cur_block_index < task.count)
        {
            thread_local_block_prefix_counts.fill(0);

            const uint64_t block_index_boundary = std::min<uint64_t>(
                task.count, cur_block_index + SCATTER_BLOCK_BATCH_SIZE);
            for (uint64_t block_index = cur_block_index; block_index < block_index_boundary; ++block_index)
            {
                kmer_block<N>* block_ptr = task.kmer_blocks[block_index];
                for (uint64_t index = 0; index < block_ptr->count; ++index)
                {
                    const uint64_t prefix = get_node_prefix(block_ptr->k_mers[index], depth);
                    thread_local_block_prefix_counts[prefix]++;
                }
            }

            thread_local_block_prefix_sums[0] = 0;
            for (uint64_t prefix = 1; prefix < thread_local_block_prefix_counts.size(); ++prefix)
            {
                thread_local_block_prefix_sums[prefix] =
                    thread_local_block_prefix_sums[prefix - 1] + thread_local_block_prefix_counts[prefix - 1];
            }

            for (uint64_t block_index = cur_block_index; block_index < block_index_boundary; ++block_index)
            {
                kmer_block<N>* block_ptr = task.kmer_blocks[block_index];
                for (uint64_t index = 0; index < block_ptr->count; ++index)
                {
                    const uint64_t prefix = get_node_prefix(block_ptr->k_mers[index], depth);
                    const uint64_t pos = thread_local_block_prefix_sums[prefix]++;
                    thread_local_block_for_copy[pos] = block_ptr->k_mers[index];
                }
            }

            for (uint64_t prefix = 0; prefix < thread_local_block_prefix_counts.size(); ++prefix)
            {
                if (thread_local_block_prefix_counts[prefix] == 0)
                {
                    continue;
                }

                const uint64_t block_for_copy_offset =
                    thread_local_block_prefix_sums[prefix] - thread_local_block_prefix_counts[prefix];
                drain_part_of_batch_to_child(
                    child_node_base + prefix, prefix, block_for_copy_offset, depth, drain_stack);
            }

            for (uint64_t block_index = cur_block_index; block_index < block_index_boundary; ++block_index)
            {
                memory_pool->deallocate(task.kmer_blocks[block_index]);
            }

            cur_block_index = block_index_boundary;
        }
    }

    void drain_task(Task<N>& t, std::vector<Task<N>>& drain_stack)
    {
        node<N>* current_node = t.current_node;

        if (t.depth >= MAX_DEPTH - 1) [[unlikely]]
        {
            insert_kmer_in_task_to_node_hash_map_with_local_hash_map(t);
            return;
        }

        node<N>* existing_child_slab = load_child_slab_existing(current_node);

        if (existing_child_slab == nullptr)
        {
            constexpr uint64_t capacity = get_block_capacity();

            uint64_t incoming_kmers = 0;
            for (uint64_t block_index = 0; block_index < t.count; ++block_index)
            {
                incoming_kmers += t.kmer_blocks[block_index]->count;
            }

            bool create_child = false;

            uint64_t available_slots = 0;
            if (current_node->count < MAX_KMER_BLOCK_NUM)
            {
                if (current_node->active_block == nullptr)
                {
                    available_slots = (MAX_KMER_BLOCK_NUM - current_node->count) * capacity;
                }
                else
                {
                    available_slots = (capacity - current_node->active_block->count) +
                        (MAX_KMER_BLOCK_NUM - current_node->count) * capacity;
                }
            }

            create_child = (incoming_kmers > available_slots);

            if (!create_child)
            {
                if (current_node->active_block == nullptr)
                {
                    current_node->active_block = reinterpret_cast<kmer_block<N> *>(memory_pool->allocate());
                    current_node->active_block->count = 0;
                    current_node->kmer_blocks[current_node->count++] = current_node->active_block;
                }

                for (uint64_t block_index = 0; block_index < t.count; ++block_index)
                {
                    kmer_block<N>* input_kmer_block = t.kmer_blocks[block_index];
                    const uint64_t kmer_count = input_kmer_block->count;

                    const uint64_t first_copy = (capacity - current_node->active_block->count) > kmer_count ? kmer_count : (capacity - current_node->active_block->count);
                    const uint64_t remaining = kmer_count - first_copy;

                    if (first_copy > 0) [[likely]]
                    {
                        std::memcpy(current_node->active_block->k_mers.data() + current_node->active_block->count,
                            input_kmer_block->k_mers.data(),
                            first_copy * sizeof(kmer<N>));
                        current_node->active_block->count += first_copy;
                    }

                    if (remaining > 0)
                    {
                        kmer_block<N>* new_block = reinterpret_cast<kmer_block<N> *>(memory_pool->allocate());
                        new_block->count = 0;
                        current_node->active_block = new_block;
                        current_node->kmer_blocks[current_node->count++] = new_block;

                        std::memcpy(new_block->k_mers.data(),
                            input_kmer_block->k_mers.data() + first_copy,
                            remaining * sizeof(kmer<N>));
                        new_block->count += remaining;
                    }

                    memory_pool->deallocate(input_kmer_block);
                }

                return;
            }
            else
            {
                if (t.depth >= MAX_DEPTH - 1)
                {
                    // Can't create children at this depth
                    insert_kmer_in_task_to_node_hash_map_with_local_hash_map(t);
                    return;
                }
                ensure_child_slab(current_node);
                if (current_node->count > 0)
                {
                    Task<N> carry_task;
                    carry_task.current_node = current_node;
                    carry_task.depth = t.depth;
                    carry_task.count = current_node->count;
                    carry_task.kmer_blocks = current_node->kmer_blocks;

                    current_node->count = 0;
                    current_node->active_block = nullptr;

                    drain_stack.push_back(carry_task);
                }

                drain_stack.push_back(t);
                return;
            }
        }
        if (t.depth >= MAX_DEPTH - 1)
        {
            std::cout << "Warning: Reached max depth but child slab already exists. Inserting into hash map." << std::endl;
            // Has existing child slab but can't descend at this depth
            insert_kmer_in_task_to_node_hash_map_with_local_hash_map(t);
            return;
        }
        scatter_drain_kmer_blocks_to_children(t, existing_child_slab, drain_stack);
    }

    /*void final_drain_range(uint64_t begin, uint64_t end,  &writer)
    {
        std::vector<DrainFrame> node_stack;
        std::vector<Task<N>> drain_stack;

        node_stack.reserve(1024);
        drain_stack.reserve(1024);

        for (uint64_t i = begin; i < end; ++i)
        {
            node_stack.push_back(DrainFrame{&root_nodes[i], 0});
        }

        while (!node_stack.empty())
        {
            DrainFrame frame = node_stack.back();
            node_stack.pop_back();

            node<N> *current = frame.node_ptr;

            if (frame.depth >= MAX_DEPTH)
            {
                if (current->count > 0)
                {
                    Task<N> task;
                    task.current_node = current;
                    task.depth = frame.depth;
                    task.count = current->count;
                    task.kmer_blocks = current->kmer_blocks;
                    insert_kmer_in_task_to_node_hash_map_without_local_hash_map(task);
                    current->count = 0;
                    current->active_block = nullptr;
                }
                continue;
            }

            if (current->count > 0)
            {
                Task<N> task;
                task.current_node = current;
                task.depth = frame.depth;
                task.count = current->count;
                task.kmer_blocks = current->kmer_blocks;
                current->count = 0;
                current->active_block = nullptr;

                drain_stack.push_back(task);
                while (!drain_stack.empty())
                {
                    Task<N> t = drain_stack.back();
                    drain_stack.pop_back();

                    drain_task(t, drain_stack);
                }
            }

            node<N> *child_slab = load_child_slab_existing(current);
            if (child_slab != nullptr)
            {
                const uint32_t child_depth = frame.depth + 1;
                constexpr uint64_t node_num = 1ULL << (2 * NODE_BASES);
                for (uint64_t c = node_num; c-- > 0;)
                {
                    node_stack.push_back(DrainFrame{&child_slab[c], child_depth});
                }
            }
            else
            {
                sort_and_export_leaf(current, writer);
            }
        }
    }*/

public:
    void final_drain_root(node<N>* root_node, FinalDrainWriter<N>& writer)
    {

#ifdef TEST_MODE
        dealing_root_index = root_node - root_nodes;
#endif

        std::vector<DrainFrame> node_stack;
        std::vector<Task<N>> drain_stack;

        node_stack.reserve(1024);
        drain_stack.reserve(1024);

        node_stack.push_back(DrainFrame{ root_node, 0 });

        while (!node_stack.empty())
        {
            DrainFrame frame = node_stack.back();
            node_stack.pop_back();

            node<N>* current = frame.node_ptr;

            if (frame.depth >= MAX_DEPTH - 1)
            {
                ConcurrentOpenAddressHashMap<N>* hash_map = current->hash_map.load(std::memory_order_acquire);
                if (current->count > 0)
                {
                    if (hash_map != nullptr)
                    {
                        // Full node: already has frequency-counting infrastructure
                        Task<N> task;
                        task.current_node = current;
                        task.depth = frame.depth;
                        task.count = current->count;
                        task.kmer_blocks = current->kmer_blocks;
                        insert_kmer_in_task_to_node_hash_map_with_local_hash_map(task);
                        current->count = 0;
                        current->active_block = nullptr;
                        export_hash_map(writer, hash_map);
                    }
                    else
                    {
                        // Non-full node: export as regular leaf
                        sort_and_export_leaf(current, writer);
                    }
                }
                else {
                    if (hash_map != nullptr)
                    {
                        // Edge case: has hash map but no pending k-mers in blocks, still need to export hash map contents
                        export_hash_map(writer, hash_map);
                    }
                }
                continue;
            }

            if (current->count > 0)
            {
                Task<N> task;
                task.current_node = current;
                task.depth = frame.depth;
                task.count = current->count;
                task.kmer_blocks = current->kmer_blocks;
                current->count = 0;
                current->active_block = nullptr;

                drain_stack.push_back(task);
                while (!drain_stack.empty())
                {
                    Task<N> t = drain_stack.back();
                    drain_stack.pop_back();

                    drain_task(t, drain_stack);
                }
            }

            node<N>* child_slab = load_child_slab_existing(current);
            if (child_slab != nullptr)
            {
                const uint32_t child_depth = frame.depth + 1;
                constexpr uint64_t node_num = 1ULL << (2 * NODE_BASES);
                for (uint64_t c = node_num; c-- > 0;)
                {
                    node_stack.push_back(DrainFrame{ &child_slab[c], child_depth });
                }
            }
            else
            {
                sort_and_export_leaf(current, writer);
            }
        }
    }

private:
    void push_kmers_into_thread_local_block_for_copy(kmer_block<N>* block_ptr, const uint32_t depth)
    {
        for (uint64_t index = 0; index < block_ptr->count; index++)
        {
            const uint64_t prefix = get_node_prefix(block_ptr->k_mers[index], depth);
            const uint64_t pos = thread_local_block_prefix_sums[prefix];
            thread_local_block_for_copy[pos] = block_ptr->k_mers[index];
            thread_local_block_prefix_sums[prefix]++;
        }
    }

    void flush_part_of_block_to_child(node<N>* child_node, const uint64_t prefix, const uint64_t in_block_for_copy_offset, const uint32_t current_depth)
    {
        Task<N> task{};

        uint64_t block_for_copy_offset = in_block_for_copy_offset;
        uint64_t remaining = thread_local_block_prefix_counts[prefix];
        constexpr uint32_t capacity = get_block_capacity();
        bool enqueue_required = false;

        uint64_t block_original_count;
        uint64_t space_left;
        uint64_t this_copy;
        kmer_block<N>* active_block;

        task.current_node = child_node;
        task.depth = current_depth + 1;

        ensure_spare_block();

        while (remaining > 0)
        {
            bool need_spare = false;

            {
                std::lock_guard lock(child_node->buffer_lock);
                child_node->writer_count.fetch_add(1, std::memory_order_relaxed);

                active_block = child_node->active_block;

                bool need_new_block =
                    (active_block == nullptr) ||
                    (active_block->count >= capacity);

                if (active_block == nullptr) [[unlikely]]
                {
                    active_block = reinterpret_cast<kmer_block<N> *>(get_spare_block());
                    active_block->count = 0;
                    child_node->active_block = active_block;
                    child_node->kmer_blocks[child_node->count++] = child_node->active_block;

                    need_spare = true;
                }
                else if (active_block->count >= capacity) [[unlikely]]
                {
                    if (child_node->count >= MAX_KMER_BLOCK_NUM) [[unlikely]]
                    {
                        // node已满
                        task.kmer_blocks = child_node->kmer_blocks;
                        task.count = child_node->count;

                        enqueue_required = true;

                        child_node->count = 0;

                        active_block = reinterpret_cast<kmer_block<N> *>(get_spare_block());
                        active_block->count = 0;
                        child_node->active_block = active_block;
                        child_node->kmer_blocks[child_node->count++] = child_node->active_block;

                        need_spare = true;

                        thread_local_spin_backoff.reset();

                        while (child_node->writer_count.load(std::memory_order_acquire) > 1)
                        {
                            thread_local_spin_backoff.backoff();
                        }
                    }
                    else
                    {
                        active_block = reinterpret_cast<kmer_block<N> *>(get_spare_block());
                        active_block->count = 0;
                        child_node->active_block = active_block;
                        child_node->kmer_blocks[child_node->count++] = active_block;

                        need_spare = true;
                    }
                }

                block_original_count = active_block->count;
                space_left = capacity - block_original_count;
                this_copy = static_cast<uint64_t>((space_left < remaining) ? space_left : remaining);
                active_block->count += this_copy;
            }

            std::memcpy(active_block->k_mers.data() + block_original_count,
                thread_local_block_for_copy.data() + block_for_copy_offset,
                static_cast<size_t>(this_copy) * sizeof(kmer<N>));

            block_for_copy_offset += this_copy;
            remaining -= this_copy;
            child_node->writer_count.fetch_sub(1, std::memory_order_release);

            if (enqueue_required) [[unlikely]]
            {
                enqueue_required = false;
                // 正常入队到下一层队列
                auto queue_ptr = layer_queue_->get_queue(current_depth + 1);
                uint32_t retry_count = 0;
                thread_local_spin_backoff.reset();
                layer_queue_->increase_size(current_depth + 1);
                while (!queue_ptr->try_enqueue(task))
                {
                    retry_count++;
                    if (retry_count >= TASK_ENQUEUE_RETRY_LIMIT)
                    {
                        // 入队失败过多次，直接放到本地栈，后续由工作线程自己处理
                        thread_local_task_stack.push_back(task);
                        layer_queue_->decrease_size(current_depth + 1);
                        break;
                    }
                    thread_local_spin_backoff.backoff();
                }
            }

            if (need_spare) [[unlikely]]
            {
                ensure_spare_block();
            }
        }
    }

    // 将一个kmer bin写入到字节点中,current_depth是当前层级深度（不是儿子的深度）
    void flush_block_to_children(node<N>* current_node, const uint32_t current_depth)
    {
        uint64_t current_offset = 0;
        node<N>* child_node_base = ensure_child_slab(current_node);
        const uint64_t start_index = rng();
        constexpr uint64_t prefix_mod = (1ULL << (2 * NODE_BASES)) - 1;
        for (uint64_t delta = 0; delta < (1ULL << (2 * NODE_BASES)); delta++)
        {
            const uint64_t prefix = (start_index + delta) & prefix_mod;
            if (thread_local_block_prefix_counts[prefix] == 0)
            {
                continue;
            }

            current_offset = thread_local_block_prefix_sums[prefix] - thread_local_block_prefix_counts[prefix];

            node<N>* child_node = child_node_base + prefix;
            flush_part_of_block_to_child(child_node, prefix, current_offset, current_depth);


        }
    }

    void flush_local_counting_hash_map_to_hash_map(ConcurrentOpenAddressHashMap<N>* hash_map, uint64_t& local_size_count)
    {
        thread_local_counting_hash_map.for_each([&](const kmer<N>& kmer_key, const uint32_t count)
            { hash_map->increment(kmer_key, count, local_size_count); });
    }

    node<N>* ensure_child_slab(node<N>* parent)
    {
        node<N>* CONSTRUCTING = reinterpret_cast<node<N> *>(MAGIC_POINTER);
        node<N>* child_slab = parent->children_ptr.load(std::memory_order_acquire);

        if (child_slab != nullptr && child_slab != CONSTRUCTING) [[likely]]
        {
            return child_slab;
        }

        if (child_slab == nullptr)
        {
            node<N>* expected = nullptr;

            if (parent->children_ptr.compare_exchange_strong(expected, CONSTRUCTING,
                std::memory_order_relaxed, std::memory_order_relaxed))
            {
                constexpr uint64_t node_num = 1ULL << (2 * NODE_BASES);
                // 分配 (1 << (2 * NODE_BASES))个连续节点
                void* raw_mem = memory_pool->allocate();
                new (raw_mem) node<N>[node_num]();
                child_slab = static_cast<node<N> *>(raw_mem);

                // 预取整个子块（后续分桶会访问其中的节点）
                __builtin_prefetch(&child_slab[0], 0, 0);

                parent->children_ptr.store(child_slab, std::memory_order_release);
            }
            else
            {
                // CAS 失败，说明别人正在创建，自旋等待
                thread_local_spin_backoff.reset();
                while (parent->children_ptr.load(std::memory_order_relaxed) == CONSTRUCTING)
                {
                    thread_local_spin_backoff.backoff();
                }
                child_slab = parent->children_ptr.load(std::memory_order_acquire);
            }
        }
        else if (child_slab == CONSTRUCTING)
        {
            // 正在创建中，自旋等待
            thread_local_spin_backoff.reset();
            while (parent->children_ptr.load(std::memory_order_relaxed) == CONSTRUCTING)
            {
                thread_local_spin_backoff.backoff();
            }
            child_slab = parent->children_ptr.load(std::memory_order_acquire);
        }

        return child_slab;
    }

    node<N>* load_child_slab_existing(node<N>* parent)
    {
        node<N>* child_slab = parent->children_ptr.load(std::memory_order_acquire);
        node<N>* CONSTRUCTING = reinterpret_cast<node<N> *>(MAGIC_POINTER);

        if (child_slab == CONSTRUCTING)
        {
            while (parent->children_ptr.load(std::memory_order_relaxed) == CONSTRUCTING)
            {
                cpu_relax();
            }
            child_slab = parent->children_ptr.load(std::memory_order_acquire);
        }

        return child_slab;
    }

    // 等待 hash map 构造完成，超时后返回 nullptr
    [[nodiscard]] ConcurrentOpenAddressHashMap<N>* wait_for_hash_map_construction(node<N>* parent)
    {
        ConcurrentOpenAddressHashMap<N>* CONSTRUCTING = reinterpret_cast<ConcurrentOpenAddressHashMap<N>*>(MAGIC_POINTER);

        static constexpr int BACKOFF_LIMIT = 32;
        static constexpr int RETRY_LIMIT = 32;

        int backoff_count = 1;
        int retry_count = 0;

        for (int retry_count = 0; retry_count < RETRY_LIMIT; retry_count++)
        {
            ConcurrentOpenAddressHashMap<N>* current = parent->hash_map.load(std::memory_order_acquire);
            if (current != CONSTRUCTING)
            {
                return current;  // 构造完成，返回有效指针（正常情况）
            }
            else
            {
                // 构造中，进行自旋等待
                for (int i = 0; i < backoff_count; i++)
                {
                    cpu_relax();
                }
                backoff_count = std::min(backoff_count * 2, BACKOFF_LIMIT);
            }
        }

        return nullptr;  // 超时，返回 nullptr
    }

    [[nodiscard]] ConcurrentOpenAddressHashMap<N>* ensure_hash_map(node<N>* parent, const uint64_t capacity)
    {
        ConcurrentOpenAddressHashMap<N>* hash_map = parent->hash_map.load(std::memory_order_acquire);
        ConcurrentOpenAddressHashMap<N>* CONSTRUCTING = reinterpret_cast<ConcurrentOpenAddressHashMap<N> *>(MAGIC_POINTER);

        if (hash_map != nullptr && hash_map != CONSTRUCTING) [[likely]]
        {
            return hash_map;
        }

        if (hash_map == nullptr)
        {
            ConcurrentOpenAddressHashMap<N>* expected = nullptr;

            if (parent->hash_map.compare_exchange_strong(expected, CONSTRUCTING,
                std::memory_order_relaxed, std::memory_order_relaxed))
            {
                ConcurrentOpenAddressHashMap<N>* hash_map_mem = nullptr;
                hash_map_mem = reinterpret_cast<ConcurrentOpenAddressHashMap<N>*>(ConcurrentOpenAddressHashMap<N>::get_map_metadata_mem());

                new (hash_map_mem) ConcurrentOpenAddressHashMap<N>(capacity);
                hash_map = hash_map_mem;
                parent->hash_map.store(hash_map, std::memory_order_release);
            }
            else
            {
                // CAS 失败，说明别人正在创建，自旋等待
                hash_map = wait_for_hash_map_construction(parent);
            }
        }
        else if (hash_map == CONSTRUCTING)
        {
            // 正在创建中，自旋等待
            hash_map = wait_for_hash_map_construction(parent);
        }

        return hash_map;
    }

    void append_export_record(FinalDrainWriter<N>& writer, const kmer<N>& key, const uint32_t count)
    {
        if (count + 1 < filter_min || count > filter_max)
        {
            return;
        }

        writer.write_kmer_record(key.data.data(), count);
    }

    void sort_and_export_leaf(node<N>* leaf, FinalDrainWriter<N>& writer)
    {
        if (leaf->count == 0)
        {
            return;
        }

        const uint64_t block_count = leaf->count;

        for (uint64_t block_index = 0; block_index < block_count; ++block_index)
        {
            kmer_block<N>* block = leaf->kmer_blocks[block_index];
            std::sort(block->k_mers.begin(), block->k_mers.begin() + static_cast<std::ptrdiff_t>(block->count));
        }

        std::array<MergeCursor, MAX_KMER_BLOCK_NUM> cursors{};
        uint64_t active_cursor_count = 0;
        for (uint64_t i = 0; i < block_count; ++i)
        {
            if (leaf->kmer_blocks[i]->count > 0)
            {
                cursors[active_cursor_count++] = MergeCursor{ leaf->kmer_blocks[i], 0 };
            }
        }

        bool has_prev = false;
        kmer<N> prev_key;
        uint32_t prev_count = 0;

        while (active_cursor_count > 0)
        {
            uint64_t min_pos = 0;
            for (uint64_t i = 1; i < active_cursor_count; ++i)
            {
                const kmer<N>& lhs = cursors[i].block->k_mers[cursors[i].index];
                const kmer<N>& rhs = cursors[min_pos].block->k_mers[cursors[min_pos].index];
                if (lhs < rhs)
                {
                    min_pos = i;
                }
            }

            const kmer<N>& current_key = cursors[min_pos].block->k_mers[cursors[min_pos].index];
            if (!has_prev) [[unlikely]]
            {
                prev_key = current_key;
                prev_count = 1;
                has_prev = true;
            }
            else if (current_key == prev_key)
            {
                ++prev_count;
            }
            else
            {
                prev_count = std::min(prev_count, static_cast<uint32_t>(count_max));
                append_export_record(writer, prev_key, prev_count);
                prev_key = current_key;
                prev_count = 1;
            }

            ++cursors[min_pos].index;
            if (cursors[min_pos].index >= cursors[min_pos].block->count)
            {
                cursors[min_pos] = cursors[active_cursor_count - 1];
                --active_cursor_count;
            }
        }

        if (has_prev) [[likely]]
        {
            prev_count = std::min(prev_count, static_cast<uint32_t>(count_max));
            append_export_record(writer, prev_key, prev_count);
        }

        for (uint64_t i = 0; i < block_count; ++i)
        {
            memory_pool->deallocate(leaf->kmer_blocks[i]);
        }

        leaf->count = 0;
        leaf->active_block = nullptr;
    }

    void export_hash_map(FinalDrainWriter<N>& writer, ConcurrentOpenAddressHashMap<N>* hash_map)
    {
        hash_map->for_each_entry(
            [&](const kmer<N>& key, const uint32_t count)
            {
                append_export_record(writer, key, count);
            });
#ifdef TEST_MODE
        hash_map->count_to_histogram();
        uint32_t segment_count = hash_map->get_segment_count();
        if (segment_count > 7)
        {
            std::pair<uint64_t, uint64_t> kmer_infos = hash_map->get_kmer_infos();
            std::cout << "Root : " << dealing_root_index << ", segment count: " << segment_count << ", singleton k-mers: " << kmer_infos.first << ", unique k-mers: " << kmer_infos.second << std::endl;
        }
#endif
    }

    void ensure_spare_block()
    {
        if (thread_local_spare_block == nullptr)
        {
            thread_local_spare_block = reinterpret_cast<char*>(memory_pool->allocate());
        }
    }

    // must use after ensure_spare_block()
    char* get_spare_block()
    {
        char* block = thread_local_spare_block;
        thread_local_spare_block = nullptr;
        return block;
    }

    void release_spare_block()
    {
        memory_pool->deallocate(thread_local_spare_block);
        thread_local_spare_block = nullptr;
    }

    // ==========================================
    // 辅助函数
    // ==========================================

    static constexpr size_t get_block_capacity()
    {
        return (KMER_BLOCK_SIZE - sizeof(uint64_t)) / sizeof(kmer<N>);
    }

    // 提取 Root 层的路由索引 (前 ROOT_BASES 个碱基)
    // 假设 kmer.data[0] 存储高位
    constexpr uint64_t get_root_prefix(const kmer<N>& val) const
    {
        // 假设每个 uint64_t 存 32 个碱基 (2 bits/base)
        // 需要提取前 ROOT_BASES 个碱基 -> 2*ROOT_BASES bits
        // 数据靠左对齐 (MSB)
        constexpr uint32_t shift = 2 * (BASES_PER_U64T - ROOT_BASES);
        return val.data[0] >> shift;
    }

    // 提取中间层的路由索引 (前 NODE_BASES 个碱基)
    uint64_t get_node_prefix(const kmer<N>& val, uint32_t depth) const
    {
        // 计算当前深度对应的 bit 偏移
        // Root层消耗 ROOT_BASES
        // 之后的每一层消耗 NODE_BASES (4)

        if constexpr (NODE_BASES != 2 && NODE_BASES != 4 && NODE_BASES != 8)
        {

            // 计算跳过的碱基数与对应的 bit 位偏移（从 data[0] 的 MSB 开始）
            uint32_t total_bases_skipped = ROOT_BASES + depth * NODE_BASES;
            uint32_t bit_pos = total_bases_skipped * 2; // bits from MSB of data[0]
            uint32_t u64_idx = bit_pos / 64;
            uint32_t bit_offset = bit_pos % 64;            // 0..63
            constexpr uint32_t need_bits = NODE_BASES * 2; // 例如 8
            if (u64_idx >= N)
                return 0; // 越界保护

            uint64_t part = 0;
            uint64_t high = val.data[u64_idx];

            // 先把当前块的有效位左移到高位
            part = (bit_offset == 0) ? high : (high << bit_offset);

            // 如果跨越边界，需要把下一个 uint64 的高位拼上来
            if (bit_offset + need_bits > 64)
            {
                uint64_t low = (u64_idx + 1 < N) ? val.data[u64_idx + 1] : 0;
                part |= (low >> (64 - bit_offset));
            }

            // 将目标段移动到最低位并掩码
            part >>= (64 - need_bits);
            uint64_t mask = (need_bits >= 64) ? ~0ULL : ((1ULL << need_bits) - 1);
            return part & mask;
        }
        else
        {
            const uint32_t total_bases_skipped = ROOT_BASES + depth * NODE_BASES;

            // 定位到具体的 data 数组索引和位偏移
            const uint32_t u64_idx = total_bases_skipped / BASES_PER_U64T;
            const uint32_t bit_offset_in_u64 = (total_bases_skipped % BASES_PER_U64T) * 2;

            uint64_t chunk = val.data[u64_idx];

            // 逻辑左移去掉前面的碱基
            chunk <<= bit_offset_in_u64;

            // 逻辑右移将目标推到最低位
            // 我们需要 4 个碱基 (8 bits)
            chunk >>= (64 - (NODE_BASES * 2));

            // 如果跨越了 uint64_t 边界 (很少见，因为 NODE_BASES=4, 8 bits 容易对齐)
            // 如果 bit_offset_in_u64 + 8 > 64，则需要读取 data[u64_idx+1]
            // 这里简化假设 NODE_BASES 使得边界对齐较好 (32能被4整除)

            return chunk & ((1ULL << (NODE_BASES * 2)) - 1);
        }
    }
};

#endif
