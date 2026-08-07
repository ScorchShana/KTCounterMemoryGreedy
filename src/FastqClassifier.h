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
#include <cstdint>
#include <bitset>
#include <cstdlib>
#include <vector>
#include <barrier>

template <uint32_t N>
class FastqClassifier
{

    // 自旋参数
    static constexpr int SLEEP_THRESHOLD = 128;
    static constexpr int YIELD_THRESHOLD = 64;
    static constexpr int MAX_BACKOFF = 64;

    static constexpr uint64_t EXPORT_KMER_BLOCK_CAPACITY = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
    static constexpr uint32_t BLOOM_PREFETCH_DISTANCE = 4; // 预取 Bloom Filter 的距离（单位：k-mer数量）

    int k_len;
    uint32_t classifier_index;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_classifier_ring_pool;
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>* classify_task_queue;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    KmerTree<N>* tree;

    std::array<node<N>, 1ULL << (2 * ROOT_BASES)> local_root_nodes{};
    std::array<uint8_t, 1ULL << (2 * ROOT_BASES)> local_prefix_owners{};
    std::array<uint8_t, 1ULL << (2 * ROOT_BASES)> prefix_to_bloom_filter_index{};

    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> local_block_prefix_counts{};
    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> local_block_prefix_sums{};
    std::array<ConcurrentBloomFilter<N>*, 1ULL << (2 * ROOT_BASES)> local_global_bloom_filter{};

    std::array<kmer<N>, PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>)> local_block_for_copy{};

    ExportBlock<N>* export_block_ptr = nullptr;
    uint64_t export_kmer_block_count = 0;

    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> enqueue_to_export_writer_backoff;
    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_from_export_writer_backoff;

    std::vector<ConcurrentBloomFilter<N>> local_bloom_filters;



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
#endif

    explicit FastqClassifier(uint32_t in_k_len,
        uint32_t in_classifier_index,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_ring_pool,
        MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>* in_classify_task_queue,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* in_global_classifier_task_queue,
        KmerTree<N>* in_tree,
        ConcurrentMemoryPool* in_memory_pool,
        std::barrier<>& in_barrier)
        : k_len(in_k_len), classifier_index(in_classifier_index), global_classifier_task_queue(in_global_classifier_task_queue), parser_classifier_ring_pool(in_ring_pool), classify_task_queue(in_classify_task_queue), tree(in_tree),
        export_block_ptr(nullptr), export_kmer_block_count(0)
    {
        char* raw_block_ptr = nullptr;
        dequeue_data_to_export_writer(raw_block_ptr);
        export_block_ptr = reinterpret_cast<ExportBlock<N> *>(raw_block_ptr);

        local_prefix_owners = prefix_owners;

        for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); i++)
        {
            if (local_prefix_owners[i] == classifier_index)
            {
                local_bloom_filters.emplace_back(bloom_filter_capacity[i], in_memory_pool);
                prefix_to_bloom_filter_index[i] = local_bloom_filters.size() - 1;
            }
        }

        uint32_t local_bloom_filters_index = 0;
        for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); i++)
        {
            if (local_prefix_owners[i] == classifier_index) {
                global_bloom_filter[i] = &local_bloom_filters[local_bloom_filters_index++];
            }

        }

        in_barrier.arrive_and_wait();

        for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); i++)
        {
            local_global_bloom_filter[i] = static_cast<ConcurrentBloomFilter<N>*>(global_bloom_filter[i]);
        }
    }

    ~FastqClassifier()
    {
    }

    void classify_and_push()
    {
        content_type content;
        //bool not_empty = true;

        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> enqueue_backoff;
        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_backoff;

        while (true)
        {
            if (!parser_classifier_ring_pool->producer_finished()) [[likely]]
            {
                bool not_empty = classify_task_queue->try_dequeue(content);
                if (!not_empty)
                {
                    not_empty = global_classifier_task_queue->try_dequeue(content);
                }

                if (not_empty)
                {

#ifdef TEST_MODE
                    not_first_flag = true;
#endif

                    dequeue_backoff.double_decay();


                    kmer<N>* kmer_data = reinterpret_cast<kmer<N> *>(content.data);
                    const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                    if (local_prefix_owners[get_root_prefix(kmer_data[0])] == classifier_index) [[likelyF]]
                    {
                        process_owned_block(kmer_data, kmer_count);
                    }
                    else
                    {
                        process_other_block(kmer_data, kmer_count);
                    }

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
                    kmer<N>* kmer_data = reinterpret_cast<kmer<N> *>(content.data);
                    const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                    process_owned_block(kmer_data, kmer_count);
                    parser_classifier_ring_pool->consumer_enqueue(content.data);
                }

                while (global_classifier_task_queue->try_dequeue(content))
                {
                    kmer<N>* kmer_data = reinterpret_cast<kmer<N> *>(content.data);
                    const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                    process_other_block(kmer_data, kmer_count);
                    parser_classifier_ring_pool->consumer_enqueue(content.data);
                }

                break;
            }
        }

        //above is new

        /*while (not_empty || !parser_classifier_ring_pool->producer_finished())
        {

            not_empty = classify_task_queue->try_dequeue(content);
            if (!not_empty)
            {
                not_empty = global_classifier_task_queue->try_dequeue(content);
            }
            if (not_empty)
            {

#ifdef TEST_MODE
                not_first_flag = true;
#endif

                dequeue_backoff.decay();


                kmer<N>* kmer_data = reinterpret_cast<kmer<N> *>(content.data);
                const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                if (local_prefix_owners[get_root_prefix(kmer_data[0])] == classifier_index) [[likelyF]]
                {
                    process_owned_block(kmer_data, kmer_count);
                }
                else {
                    process_other_block(kmer_data, kmer_count);
                }



                if (parser_classifier_ring_pool->consumer_try_enqueue(content.data))
                {
                    // 无等待
                    enqueue_backoff.decay();
                }
                else
                {
                    // 自旋等待
                    enqueue_backoff.decay();

                    while (!parser_classifier_ring_pool->consumer_try_enqueue(content.data))
                    {
#ifdef TEST_MODE
                        consumer_enqueue_spin_time++;
#endif
                        enqueue_backoff.backoff();
                    }
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
        }*/

        enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
        export_kmer_block_count = 0;

        tree->flush_local_root_nodes(local_root_nodes.data(), rng());
    }

private:
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

            uint32_t prefix_export_count = 0;

            const uint32_t bloom_filter_index = prefix_to_bloom_filter_index[prefix];

            ConcurrentBloomFilter<N>& bloom_filter = local_bloom_filters[bloom_filter_index];

            if (prefix_count < BLOOM_PREFETCH_DISTANCE / 2) [[unlikely]]
            {
                for (uint32_t i = 0; i < prefix_count; ++i)
                {
                    const kmer<N>& val = local_block_for_copy[read_offset];

                    if (bloom_filter.insert(val) == Occurrence::FIRST)
                    {
                        export_block_ptr->k_mers[export_kmer_block_count++] = val;
                        prefix_export_count++;

                        if (export_kmer_block_count == EXPORT_KMER_BLOCK_CAPACITY) [[unlikely]]
                        {
                            enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
                            export_kmer_block_count = 0;
                            char* raw_block_ptr = nullptr;
                            dequeue_data_to_export_writer(raw_block_ptr);
                            export_block_ptr = reinterpret_cast<ExportBlock<N> *>(raw_block_ptr);
                        }
                    }
                    else
                    {
                        local_block_for_copy[local_block_count++] = val;
                    }

                    read_offset++;
                }

                local_block_prefix_counts[prefix] -= prefix_export_count;

#ifdef TEST_MODE
                total_kmers_exported += prefix_export_count;
                total_kmers_send_to_tree += local_block_prefix_counts[prefix];
#endif
                continue;
            }

            std::array<typename ConcurrentBloomFilter<N>::InsertProbe, BLOOM_PREFETCH_DISTANCE> probes;
            const uint64_t prefix_begin = read_offset;
            const uint32_t warmup_count = (prefix_count < BLOOM_PREFETCH_DISTANCE) ? prefix_count : BLOOM_PREFETCH_DISTANCE;

            for (uint32_t i = 0; i < warmup_count; ++i)
            {
                const uint64_t idx = prefix_begin + i;
                probes[i] = bloom_filter.prepare_insert(local_block_for_copy[idx]);
                bloom_filter.prefetch_insert(probes[i]);
            }

            for (uint32_t i = 0; i < prefix_count; ++i)
            {
                const uint32_t slot = i % BLOOM_PREFETCH_DISTANCE;
                const uint64_t idx = prefix_begin + i;
                const kmer<N>& val = local_block_for_copy[idx];

                const Occurrence occ = bloom_filter.insert_prepared(probes[slot]);

                const uint32_t next_i = i + BLOOM_PREFETCH_DISTANCE;
                if (next_i < prefix_count)
                {
                    const uint64_t next_idx = prefix_begin + next_i;
                    probes[slot] = bloom_filter.prepare_insert(local_block_for_copy[next_idx]);
                    bloom_filter.prefetch_insert(probes[slot]);
                }

                if (occ == Occurrence::FIRST)
                {
                    export_block_ptr->k_mers[export_kmer_block_count++] = val;
                    prefix_export_count++;

                    if (export_kmer_block_count == EXPORT_KMER_BLOCK_CAPACITY) [[unlikely]]
                    {
                        enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
                        export_kmer_block_count = 0;
                        char* raw_block_ptr = nullptr;
                        dequeue_data_to_export_writer(raw_block_ptr);
                        export_block_ptr = reinterpret_cast<ExportBlock<N> *>(raw_block_ptr);
                    }
                }
                else
                {
                    local_block_for_copy[local_block_count++] = val;
                }
            }

            read_offset += prefix_count;
            local_block_prefix_counts[prefix] -= prefix_export_count;
#ifdef TEST_MODE
            total_kmers_exported += prefix_export_count;
            total_kmers_send_to_tree += local_block_prefix_counts[prefix];
#endif
        }

        if (local_block_count > 0) [[likely]]
        {
            tree->main_add_kmer_block_with_local_root_nodes(local_block_for_copy, local_block_prefix_counts, local_root_nodes.data());
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

            ConcurrentBloomFilter<N>& bloom_filter = *local_global_bloom_filter[prefix];

            // if (prefix_count < BLOOM_PREFETCH_DISTANCE / 2) [[unlikely]]
            // {
            //     for (uint32_t i = 0; i < prefix_count; ++i)
            //     {
            //         const kmer<N>& val = local_block_for_copy[read_offset];

            //         if (bloom_filter.insert(val))
            //         {
            //             export_block_ptr->k_mers[export_kmer_block_count++] = val;
            //             prefix_export_count++;

            //             if (export_kmer_block_count == EXPORT_KMER_BLOCK_CAPACITY) [[unlikely]]
            //             {
            //                 enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
            //                 export_kmer_block_count = 0;
            //                 char* raw_block_ptr = nullptr;
            //                 dequeue_data_to_export_writer(raw_block_ptr);
            //                 export_block_ptr = reinterpret_cast<ExportBlock<N> *>(raw_block_ptr);
            //             }
            //         }
            //         else
            //         {
            //             local_block_for_copy[local_block_count++] = val;
            //         }

            //         read_offset++;
            //     }

            //     local_block_prefix_counts[prefix] -= prefix_export_count;
            //     continue;
            // }

            // std::array<typename ConcurrentBloomFilter<N>::InsertProbe, BLOOM_PREFETCH_DISTANCE> probes;
            // const uint64_t prefix_begin = read_offset;
            // const uint32_t warmup_count = (prefix_count < BLOOM_PREFETCH_DISTANCE) ? prefix_count : BLOOM_PREFETCH_DISTANCE;

            // for (uint32_t i = 0; i < warmup_count; ++i)
            // {
            //     const uint64_t idx = prefix_begin + i;
            //     probes[i] = bloom_filter.prepare_insert(local_block_for_copy[idx]);
            //     bloom_filter.prefetch_insert(probes[i]);
            // }

            // for (uint32_t i = 0; i < prefix_count; ++i)
            // {
            //     const uint32_t slot = i % BLOOM_PREFETCH_DISTANCE;
            //     const uint64_t idx = prefix_begin + i;
            //     const kmer<N>& val = local_block_for_copy[idx];

            //     const bool is_first = bloom_filter.insert_prepared(probes[slot]);

            //     const uint32_t next_i = i + BLOOM_PREFETCH_DISTANCE;
            //     if (next_i < prefix_count)
            //     {
            //         const uint64_t next_idx = prefix_begin + next_i;
            //         probes[slot] = bloom_filter.prepare_insert(local_block_for_copy[next_idx]);
            //         bloom_filter.prefetch_insert(probes[slot]);
            //     }

            //     if (is_first)
            //     {
            //         export_block_ptr->k_mers[export_kmer_block_count++] = val;
            //         prefix_export_count++;

            //         if (export_kmer_block_count == EXPORT_KMER_BLOCK_CAPACITY) [[unlikely]]
            //         {
            //             enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
            //             export_kmer_block_count = 0;
            //             char* raw_block_ptr = nullptr;
            //             dequeue_data_to_export_writer(raw_block_ptr);
            //             export_block_ptr = reinterpret_cast<ExportBlock<N> *>(raw_block_ptr);
            //         }
            //     }
            //     else
            //     {
            //         local_block_for_copy[local_block_count++] = val;
            //     }
            // }

            for (uint32_t i = 0; i < prefix_count; i++)
            {
                const kmer<N>& val = local_block_for_copy[read_offset];

                if (bloom_filter.insert(val) == Occurrence::FIRST)
                {
                    export_block_ptr->k_mers[export_kmer_block_count++] = val;
                    prefix_export_count++;

                    if (export_kmer_block_count == EXPORT_KMER_BLOCK_CAPACITY) [[unlikely]]
                    {
                        enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
                        export_kmer_block_count = 0;
                        char* raw_block_ptr = nullptr;
                        dequeue_data_to_export_writer(raw_block_ptr);
                        export_block_ptr = reinterpret_cast<ExportBlock<N> *>(raw_block_ptr);
                    }
                }
                else
                {
                    local_block_for_copy[local_block_count++] = val;
                }

                read_offset++;
            }

            //read_offset += prefix_count;
            local_block_prefix_counts[prefix] -= prefix_export_count;
#ifdef TEST_MODE
            total_kmers_exported += prefix_export_count;
            total_kmers_send_to_tree += local_block_prefix_counts[prefix];
#endif
        }

        if (local_block_count > 0) [[likely]]
        {

            tree->main_add_kmer_block_with_local_root_nodes(local_block_for_copy, local_block_prefix_counts, local_root_nodes.data());
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
