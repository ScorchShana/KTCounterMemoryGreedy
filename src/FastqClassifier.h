#ifndef FASTQCLASSIFIER_HEADER
#define FASTQCLASSIFIER_HEADER

#include "definition.h"
#include "kmer.h"
#include "RingMemoryPool.h"
#include "NewKmerTree.h"
#include "BloomFilter.h"
#include "SplitMix.h"
#include "MPSCRingQueue.h"
#include "ConcurrentMemoryPool.h"
#include "SpinBackoff.h"

#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

/**
 * @brief 两趟遍历中分类器所处的阶段
 *
 *   BuildFilters —— 第一趟：只用 BF_once / BF_multi 统计每个 k-mer 的出现次数，
 *                   不做导出、不入树、不需要 export block。
 *   QueryRoute   —— 第二趟：只查询两个过滤器，
 *                   BF_multi 命中 → 多次出现，进树；
 *                   否则          → 只出现一次，写盘。
 *
 * 过滤器由 main 在两趟之前一次性创建并发布到 global_bloom_filter，
 * 因此两个阶段的 FastqClassifier 都只是引用它们，不拥有它们。
 */
enum class ClassifyPass : uint8_t
{
    BuildFilters = 0,
    QueryRoute = 1,
};

template <uint32_t N>
class FastqClassifier
{

    // 自旋参数
    static constexpr int SLEEP_THRESHOLD = 128 + 16;
    static constexpr int YIELD_THRESHOLD = 128;
    static constexpr int MAX_BACKOFF = 32;

    static constexpr uint64_t EXPORT_KMER_BLOCK_CAPACITY = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
    static constexpr uint32_t BLOOM_PREFETCH_DISTANCE = 16; // 预取 Bloom Filter 的距离（单位：k-mer数量）

    static constexpr uint32_t MAX_CHECK_LOCAL_ROUND = 128;
    static constexpr uint32_t MIN_CHECK_LOCAL_ROUND = 16;

    static constexpr uint64_t PREFIX_COUNT = 1ULL << (2 * ROOT_BASES);

    int k_len;
    uint32_t classifier_index;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_classifier_ring_pool;
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>* classify_task_queue;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    KmerTree<N>* tree;
    ClassifyPass pass;

    std::array<node<N>, PREFIX_COUNT> local_root_nodes{};
    std::array<uint8_t, PREFIX_COUNT> local_prefix_owners{};

    std::array<uint32_t, PREFIX_COUNT> local_block_prefix_counts{};
    std::array<uint32_t, PREFIX_COUNT> local_block_prefix_sums{};
    std::array<ConcurrentBloomFilter<N>*, PREFIX_COUNT> local_global_bloom_filter{};

    std::array<uint32_t, PREFIX_COUNT> prefix_ordered_by_owner{};

    std::array<kmer<N>, PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>)> local_block_for_copy{};

    ExportBlock<N>* export_block_ptr = nullptr;
    uint64_t export_kmer_block_count = 0;

    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> enqueue_to_export_writer_backoff;
    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_from_export_writer_backoff;

    std::vector<uint32_t> local_owned_preifx{};

    uint32_t check_local_first_round = 16;
    uint32_t local_first_round = 0;

    SplitMix64 rng;

public:
    uint64_t total_deal_kmer_count = 0;
#ifdef TEST_MODE
    uint64_t producer_enqueue_spin_time{ 0 };
    uint64_t producer_dequeue_spin_time{ 0 };
    uint64_t consumer_enqueue_spin_time{ 0 };
    uint64_t consumer_dequeue_spin_time{ 0 };
    bool not_first_flag = false;
    uint64_t total_kmers_exported = 0;
    uint64_t total_kmers_send_to_tree = 0;
    uint64_t classifier_wait_cycles = 0;
    uint64_t global_tasks = 0;
    uint64_t local_tasks = 0;
    uint64_t owner_tasks = 0;
#endif

    explicit FastqClassifier(uint32_t in_k_len,
        uint32_t in_classifier_index,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_ring_pool,
        MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>* in_classify_task_queue,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* in_global_classifier_task_queue,
        KmerTree<N>* in_tree,
        ClassifyPass in_pass)
        : k_len(in_k_len),
        classifier_index(in_classifier_index),
        parser_classifier_ring_pool(in_ring_pool),
        classify_task_queue(in_classify_task_queue),
        global_classifier_task_queue(in_global_classifier_task_queue),
        tree(in_tree),
        pass(in_pass),
        export_block_ptr(nullptr),
        export_kmer_block_count(0)
    {
        local_prefix_owners = prefix_owners;

        // 双 Bloom 过滤器由 main 在第一趟之前创建并发布，两个阶段共用同一批过滤器
        for (uint64_t i = 0; i < PREFIX_COUNT; i++)
        {
            local_global_bloom_filter[i] = static_cast<ConcurrentBloomFilter<N>*>(global_bloom_filter[i]);

            if (local_prefix_owners[i] == classifier_index)
            {
                local_owned_preifx.push_back(i);
            }
        }

        // 只有第二趟才需要导出缓冲
        if (pass == ClassifyPass::QueryRoute)
        {
            char* raw_block_ptr = nullptr;
            dequeue_data_to_export_writer(raw_block_ptr);
            export_block_ptr = reinterpret_cast<ExportBlock<N>*>(raw_block_ptr);
        }
    }

    ~FastqClassifier()
    {
    }

    void classify_and_push()
    {
        content_type content;
        //bool not_empty = true;
        local_first_round = 0;
        check_local_first_round = MIN_CHECK_LOCAL_ROUND * 2;

        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> enqueue_backoff;
        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_backoff;

        while (true)
        {
            if (!parser_classifier_ring_pool->producer_finished()) [[likely]]
            {
                bool not_empty = false;
                if (local_first_round < check_local_first_round) [[likely]]
                {
                    local_first_round++;
                    not_empty = classify_task_queue->try_dequeue(content);
#ifdef TEST_MODE
                    if (not_empty)
                        local_tasks++;
#endif
                    if (!not_empty)
                    {
                        not_empty = global_classifier_task_queue->try_dequeue(content);

#ifdef TEST_MODE
                        if (not_empty)
                            global_tasks++;
#endif
                    }
                }
                else
                {
                    const uint64_t global_queue_size = global_classifier_task_queue->size();
                    if (global_queue_size > PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY / 2) {
                        check_local_first_round = std::max(check_local_first_round / 2, MIN_CHECK_LOCAL_ROUND);
                    }
                    else if (global_queue_size < PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY / 4) {
                        check_local_first_round = std::min(check_local_first_round * 2, MAX_CHECK_LOCAL_ROUND);
                    }

                    local_first_round = 0;
                    not_empty = global_classifier_task_queue->try_dequeue(content);
#ifdef TEST_MODE
                    if (not_empty)
                        global_tasks++;
#endif
                    if (!not_empty)
                    {
                        not_empty = classify_task_queue->try_dequeue(content);

#ifdef TEST_MODE
                        if (not_empty)
                            local_tasks++;
#endif
                    }
                }


                if (not_empty)
                {

#ifdef TEST_MODE
                    not_first_flag = true;
#endif

                    dequeue_backoff.reset();

                    dispatch_block(reinterpret_cast<kmer<N> *>(content.data), content.length);

                    if (parser_classifier_ring_pool->consumer_try_enqueue(content.data))
                    {
                        // 无等待
                        enqueue_backoff.double_decay();
                    }
                    else
                    {
                        // 自旋等待
                        enqueue_backoff.backoff();

                        while (!parser_classifier_ring_pool->consumer_try_enqueue(content.data))
                        {
#ifdef TEST_MODE
                            consumer_enqueue_spin_time++;
#endif
                            enqueue_backoff.backoff();
                        }

                        enqueue_backoff.decay();
                    }
                }
                else
                {

#ifdef TEST_MODE
                    if (not_first_flag)
                        consumer_dequeue_spin_time++;
#endif

                    dequeue_backoff.backoff();
                }
            }
            else
            {
                while (classify_task_queue->try_dequeue(content))
                {
                    dispatch_block(reinterpret_cast<kmer<N> *>(content.data), content.length);
                    parser_classifier_ring_pool->consumer_enqueue(content.data);
                }

                while (global_classifier_task_queue->try_dequeue(content))
                {
                    dispatch_block(reinterpret_cast<kmer<N> *>(content.data), content.length);
                    parser_classifier_ring_pool->consumer_enqueue(content.data);
                }

                break;
            }
        }

        // 第一趟只统计出现次数，既不落盘也不入树
        if (pass == ClassifyPass::QueryRoute)
        {
            enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
            export_kmer_block_count = 0;

            tree->flush_local_root_nodes(local_root_nodes.data(), rng());
        }

#ifdef TEST_MODE
        // 第一趟没有树
        if (tree != nullptr)
        {
            classifier_wait_cycles = tree->classifier_wait_cycles;
        }
#endif
    }


private:
    void dispatch_block(kmer<N>* kmer_data, const uint64_t kmer_count)
    {
        if (pass == ClassifyPass::BuildFilters)
        {
            // 第一趟：只按 prefix 分组更新双 Bloom，不需要区分 owner
            build_filters_block(kmer_data, kmer_count);
        }
        else if (local_prefix_owners[get_root_prefix(kmer_data[0])] == classifier_index) [[likely]]
        {
            process_owned_block(kmer_data, kmer_count);

#ifdef TEST_MODE
            owner_tasks++;
#endif
        }
        else
        {
            process_other_block(kmer_data, kmer_count);
        }
    }

    // ================= 第一趟：只建过滤器 =================
    void build_filters_block(kmer<N>* kmer_data, const uint64_t kmer_count)
    {
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        build_filters_local_block();
        total_deal_kmer_count += kmer_count;
    }

    void build_filters_local_block() noexcept
    {
        uint64_t read_offset = 0;
        for (uint64_t prefix = 0; prefix < PREFIX_COUNT; prefix++)
        {
            const uint32_t prefix_count = local_block_prefix_counts[prefix];
            if (prefix_count == 0)
            {
                continue;
            }

            ConcurrentBloomFilter<N>& bloom_filter = *local_global_bloom_filter[prefix];

            if (prefix_count < BLOOM_PREFETCH_DISTANCE / 2) [[unlikely]]
            {
                for (uint32_t i = 0; i < prefix_count; ++i)
                {
                    bloom_filter.insert(local_block_for_copy[read_offset++]);
                }
                continue;
            }

            std::array<typename ConcurrentBloomFilter<N>::Probe, BLOOM_PREFETCH_DISTANCE> probes;
            const uint64_t prefix_begin = read_offset;
            const uint32_t warmup_count = (prefix_count < BLOOM_PREFETCH_DISTANCE) ? prefix_count : BLOOM_PREFETCH_DISTANCE;

            for (uint32_t i = 0; i < warmup_count; ++i)
            {
                probes[i] = bloom_filter.prepare(local_block_for_copy[prefix_begin + i]);
                bloom_filter.prefetch(probes[i]);
            }

            for (uint32_t i = 0; i < prefix_count; ++i)
            {
                const uint32_t slot = i % BLOOM_PREFETCH_DISTANCE;

                bloom_filter.insert_prepared(probes[slot]);

                const uint32_t next_i = i + BLOOM_PREFETCH_DISTANCE;
                if (next_i < prefix_count)
                {
                    probes[slot] = bloom_filter.prepare(local_block_for_copy[prefix_begin + next_i]);
                    bloom_filter.prefetch(probes[slot]);
                }
            }

            read_offset += prefix_count;
        }
    }

    // ================= 第二趟：查询并路由 =================
    void process_owned_block(kmer<N>* kmer_data, const uint64_t kmer_count)
    {
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        classify_owned_local_block();
        total_deal_kmer_count += kmer_count;
    }

    void process_other_block(kmer<N>* kmer_data, const uint64_t kmer_count)
    {
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        classify_other_local_block();
        total_deal_kmer_count += kmer_count;
    }

    void calculate_block_prefix_counts(kmer<N>* kmer_data, const uint64_t kmer_count)
    {

        memset(local_block_prefix_counts.data(), 0, local_block_prefix_counts.size() * sizeof(uint32_t));

        // 可以考虑simd加速
        for (uint64_t i = 0; i < kmer_count; ++i)
        {
            const uint64_t prefix = get_root_prefix(kmer_data[i]);
            local_block_prefix_counts[prefix]++;
        }

        local_block_prefix_sums[0] = 0;
        for (uint64_t i = 1; i < local_block_prefix_counts.size(); ++i)
        {
            local_block_prefix_sums[i] = local_block_prefix_sums[i - 1] + local_block_prefix_counts[i - 1];
        }
    }

    void push_kmers_into_local_block_for_copy(kmer<N>* kmer_data, const uint64_t kmer_count)
    {
        for (uint64_t index = 0; index < kmer_count; index++)
        {
            const uint64_t prefix = get_root_prefix(kmer_data[index]);
            const uint64_t pos = local_block_prefix_sums[prefix];
            local_block_for_copy[pos] = kmer_data[index];
            local_block_prefix_sums[prefix]++;
        }
    }

    void export_one_kmer(const kmer<N>& val) noexcept
    {
        export_block_ptr->k_mers[export_kmer_block_count++] = val;
        if (export_kmer_block_count == EXPORT_KMER_BLOCK_CAPACITY) [[unlikely]]
        {
            enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
            export_kmer_block_count = 0;
            char* raw_block_ptr = nullptr;
            dequeue_data_to_export_writer(raw_block_ptr);
            export_block_ptr = reinterpret_cast<ExportBlock<N>*>(raw_block_ptr);
        }
#ifdef TEST_MODE
        total_kmers_exported++;
#endif
    }

    void classify_owned_local_block() noexcept
    {

        uint64_t local_block_count = 0;

        uint64_t read_offset = 0;
        for (uint64_t prefix = 0; prefix < local_block_prefix_counts.size(); prefix++)
        {
            const uint32_t prefix_count = local_block_prefix_counts[prefix];
            if (prefix_count == 0)
            {
                continue;
            }

            uint32_t to_tree_count = 0;

            ConcurrentBloomFilter<N>& bloom_filter = *local_global_bloom_filter[prefix];

            if (prefix_count < BLOOM_PREFETCH_DISTANCE / 2) [[unlikely]]
            {
                for (uint32_t i = 0; i < prefix_count; ++i)
                {
                    const kmer<N>& val = local_block_for_copy[read_offset];

                    if (bloom_filter.query(val) == Occurrence::MULTI)
                    {
                        local_block_for_copy[local_block_count++] = val;
                        to_tree_count++;
                    }
                    else
                    {
                        export_one_kmer(val);
                    }

                    read_offset++;
                }

                local_block_prefix_counts[prefix] = to_tree_count;

#ifdef TEST_MODE
                total_kmers_send_to_tree += to_tree_count;
#endif
                continue;
            }

            std::array<typename ConcurrentBloomFilter<N>::Probe, BLOOM_PREFETCH_DISTANCE> probes;
            const uint64_t prefix_begin = read_offset;
            const uint32_t warmup_count = (prefix_count < BLOOM_PREFETCH_DISTANCE) ? prefix_count : BLOOM_PREFETCH_DISTANCE;

            for (uint32_t i = 0; i < warmup_count; ++i)
            {
                const uint64_t idx = prefix_begin + i;
                probes[i] = bloom_filter.prepare(local_block_for_copy[idx]);
                bloom_filter.prefetch(probes[i]);
            }

            for (uint32_t i = 0; i < prefix_count; ++i)
            {
                const uint32_t slot = i % BLOOM_PREFETCH_DISTANCE;
                const uint64_t idx = prefix_begin + i;
                const kmer<N>& val = local_block_for_copy[idx];

                const Occurrence occ = bloom_filter.query_prepared(probes[slot]);

                const uint32_t next_i = i + BLOOM_PREFETCH_DISTANCE;
                if (next_i < prefix_count)
                {
                    const uint64_t next_idx = prefix_begin + next_i;
                    probes[slot] = bloom_filter.prepare(local_block_for_copy[next_idx]);
                    bloom_filter.prefetch(probes[slot]);
                }

                if (occ == Occurrence::MULTI)
                {
                    local_block_for_copy[local_block_count++] = val;
                    to_tree_count++;
                }
                else
                {
                    export_one_kmer(val);
                }
            }

            read_offset += prefix_count;
            local_block_prefix_counts[prefix] = to_tree_count;
#ifdef TEST_MODE
            total_kmers_send_to_tree += to_tree_count;
#endif
        }

        if (local_block_count > 0) [[likely]]
        {
            uint32_t cnt = 0;
            for (uint64_t i = 0; i < local_owned_preifx.size(); i++)
            {
                prefix_ordered_by_owner[cnt] = local_owned_preifx[i];
                cnt = (local_block_prefix_counts[local_owned_preifx[i]] > 0) ? cnt + 1 : cnt;
            }
            tree->main_add_kmer_block_with_local_root_nodes(local_block_for_copy, local_block_prefix_counts, prefix_ordered_by_owner, cnt, local_root_nodes.data());
        }
    }

    void classify_other_local_block() noexcept
    {
        uint64_t local_block_count = 0;

        uint64_t read_offset = 0;
        for (uint64_t prefix = 0; prefix < local_block_prefix_counts.size(); prefix++)
        {
            const uint32_t prefix_count = local_block_prefix_counts[prefix];
            if (prefix_count == 0)
            {
                continue;
            }

            uint32_t prefix_export_count = 0;

            // 非 owner 前缀：共享全局 filter，只查询、逐个判定
            ConcurrentBloomFilter<N>& bloom_filter = *local_global_bloom_filter[prefix];

            for (uint32_t i = 0; i < prefix_count; i++)
            {
                const kmer<N>& val = local_block_for_copy[read_offset];

                if (bloom_filter.query(val) == Occurrence::MULTI)
                {
                    // 多次出现：立刻进树缓冲
                    local_block_for_copy[local_block_count++] = val;
                }
                else
                {
                    // 只出现一次：写盘
                    export_one_kmer(val);
                    prefix_export_count++;
                }

                read_offset++;
            }

            local_block_prefix_counts[prefix] -= prefix_export_count;
#ifdef TEST_MODE
            // export 已在 export_one_kmer 中计数；此处计进树的数量
            total_kmers_send_to_tree += local_block_prefix_counts[prefix];
#endif
        }

        if (local_block_count > 0) [[likely]]
        {
            uint32_t cnt = 0;
            for (uint64_t i = 0; i < PREFIX_COUNT; i++)
            {
                prefix_ordered_by_owner[cnt] = i;
                cnt = (local_block_prefix_counts[i] > 0) ? cnt + 1 : cnt;
            }
            tree->main_add_kmer_block_with_local_root_nodes(local_block_for_copy, local_block_prefix_counts, prefix_ordered_by_owner, cnt, local_root_nodes.data());
        }
    }

    uint64_t get_root_prefix(const kmer<N>& k_mer) const
    {
        constexpr uint32_t shift_bits = 64 - (ROOT_BASES * 2);
        return k_mer.data[0] >> shift_bits;
    }

    void enqueue_content_to_export_writer(const content_type& content)
    {
        RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* export_pool = tree->get_export_ring_pool();

        if (export_pool->producer_try_enqueue(content))
        {
            enqueue_to_export_writer_backoff.double_decay();
            return;
        }

        while (!export_pool->producer_try_enqueue(content))
        {

#ifdef TEST_MODE
            producer_enqueue_spin_time++;
#endif
            enqueue_to_export_writer_backoff.backoff();
        }

        enqueue_to_export_writer_backoff.decay();
    }

    void dequeue_data_to_export_writer(char*& data)
    {
        RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* export_pool = tree->get_export_ring_pool();

        if (export_pool->producer_try_dequeue(data))
        {
            dequeue_from_export_writer_backoff.double_decay();
            return;
        }

        while (!export_pool->producer_try_dequeue(data))
        {

#ifdef TEST_MODE
            producer_dequeue_spin_time++;
#endif
            dequeue_from_export_writer_backoff.backoff();
        }

        dequeue_from_export_writer_backoff.decay();
    }
};

#endif // FASTQCLASSIFIER_HEADER
