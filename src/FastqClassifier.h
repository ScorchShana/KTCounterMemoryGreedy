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
#include "kmer.h"
#include "RunState.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <new>
#include <vector>
#include <barrier>

template<typename CountType = uint32_t>
class FastqClassifier
{

    friend struct FastqClassifierTestAccess;

    // 自旋参数
    static constexpr int SLEEP_THRESHOLD = 128 + 16;
    static constexpr int YIELD_THRESHOLD = 128;
    static constexpr int MAX_BACKOFF = 32;

    const uint64_t EXPORT_KMER_BLOCK_CAPACITY = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
    static constexpr uint32_t BLOOM_PREFETCH_DISTANCE = 16; // 预取 Bloom Filter 的距离（单位：k-mer数量）

    static constexpr uint32_t MAX_CHECK_LOCAL_ROUND = 128;
    static constexpr uint32_t MIN_CHECK_LOCAL_ROUND = 16;

    uint32_t classifier_index;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_classifier_ring_pool;
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>* classify_task_queue;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    KmerTree<CountType>* tree;

    std::array<node<CountType>, 1ULL << (2 * ROOT_BASES)> local_root_nodes{};
    std::array<uint8_t, 1ULL << (2 * ROOT_BASES)> local_prefix_owners{};
    std::array<uint8_t, 1ULL << (2 * ROOT_BASES)> local_prefix_to_bloom_filter_index{};
    std::array<uint8_t, 1ULL << (2 * ROOT_BASES)> local_bloom_filter_index_to_prefix{};

    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> local_block_prefix_counts{};
    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> local_block_prefix_sums{};
    std::array<ConcurrentBloomFilter*, 1ULL << (2 * ROOT_BASES)> local_global_bloom_filter{};

    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> prefix_ordered_by_owner{};

    // Compact REPEAT records in place; FIRST records go directly to Export.
    std::array<unsigned char, PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE> local_block_for_copy{};

    ExportBlock* export_block_ptr = nullptr;
    uint64_t export_kmer_block_count = 0;

    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> enqueue_to_export_writer_backoff;
    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_from_export_writer_backoff;

    std::vector<ConcurrentBloomFilter> local_bloom_filters;
    std::vector<uint32_t> local_owned_preifx{};

    uint32_t check_local_first_round = 16;
    uint32_t local_first_round = 0;


    SplitMix64 rng;

    ConcurrentMemoryPool* memory_pool_;
    size_t checkpoint_bytes_ = 0;

public:
    uint64_t total_deal_kmer_count = 0;
    uint64_t first_occurrence_count = 0;
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

    explicit FastqClassifier(uint32_t in_classifier_index,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_ring_pool,
        MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>* in_classify_task_queue,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* in_global_classifier_task_queue,
        KmerTree<CountType>* in_tree,
        ConcurrentMemoryPool* in_memory_pool,
        const std::array<void*, 256>* bloom_storage = nullptr)
        : classifier_index(in_classifier_index), parser_classifier_ring_pool(in_ring_pool),
        classify_task_queue(in_classify_task_queue), global_classifier_task_queue(in_global_classifier_task_queue),
        tree(in_tree),
        memory_pool_(in_memory_pool)
    {
        KmerTree<CountType>::initialize_roots(local_root_nodes.data());
        char* raw_block_ptr = nullptr;
        dequeue_data_to_export_writer(raw_block_ptr);
        export_block_ptr = reinterpret_cast<ExportBlock*>(raw_block_ptr);

        local_prefix_owners = prefix_owners;

        for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); i++)
        {
            if (local_prefix_owners[i] == classifier_index)
            {
                local_bloom_filters.emplace_back(bloom_filter_capacity[i], in_memory_pool, bloom_storage ? (*bloom_storage)[i] : nullptr);
                local_prefix_to_bloom_filter_index[i] = local_bloom_filters.size() - 1;
                local_bloom_filter_index_to_prefix[local_bloom_filters.size() - 1] = i;
            }
        }

        uint32_t local_bloom_filters_index = 0;
        for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); i++)
        {
            if (local_prefix_owners[i] == classifier_index) {
                global_bloom_filter[i] = &local_bloom_filters[local_bloom_filters_index++];
                local_owned_preifx.push_back(i);
            }

        }

    }

    // Called after every owner has constructed and retained its filters.
    void initialize_shared_bloom() noexcept
    {
        for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); i++)
        {
            local_global_bloom_filter[i] = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[i]);
        }
    }

    void classify_first_pass()
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


                    unsigned char* kmer_data = reinterpret_cast<unsigned char*>(content.data);
                    const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                    if (local_prefix_owners[get_root_prefix((kmer_data + (0) * kmer_bytes()))] == classifier_index) [[likely]]
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

                    if (parser_classifier_ring_pool->consumer_try_enqueue(content.data))
                    {
                        // 无等待
                        enqueue_backoff.double_decay();
                    }
                    else
                    {
                        do
                        {
#ifdef TEST_MODE
                            consumer_enqueue_spin_time++;
#endif
                            enqueue_backoff.backoff();
                        }
                        while (!parser_classifier_ring_pool->consumer_try_enqueue(content.data));

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
                    unsigned char* kmer_data = reinterpret_cast<unsigned char*>(content.data);
                    const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                    process_owned_block(kmer_data, kmer_count);
                    if (!parser_classifier_ring_pool->consumer_try_enqueue(content.data)) {
                        parser_classifier_ring_pool->consumer_enqueue(content.data);
                    }
                }

                while (global_classifier_task_queue->try_dequeue(content))
                {
                    unsigned char* kmer_data = reinterpret_cast<unsigned char*>(content.data);
                    const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                    process_other_block(kmer_data, kmer_count);
                    if (!parser_classifier_ring_pool->consumer_try_enqueue(content.data)) {
                        parser_classifier_ring_pool->consumer_enqueue(content.data);
                    }
                }

                break;
            }
        }
        enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
        export_kmer_block_count = 0;
        export_block_ptr = nullptr;
    }

    void finish_tree_input()
    {
        tree->flush_local_root_nodes(local_root_nodes.data(), rng());

#ifdef TEST_MODE
        classifier_wait_cycles = tree->classifier_wait_cycles;
#endif
    }


private:
    void record_checkpoint()
    {
        checkpoint_bytes_ += kmer_bytes();
        if (checkpoint_bytes_ >= BLOCK_SIZE) {
            checkpoint_bytes_ = 0;
            memory_pool_->run_state().check();
        }
    }
    void process_owned_block(const unsigned char* kmer_data, const uint64_t kmer_count)
    {
        memory_pool_->run_state().check();
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        classify_owned_local_block();
        total_deal_kmer_count += kmer_count;
    }

    void process_other_block(const unsigned char* kmer_data, const uint64_t kmer_count)
    {
        memory_pool_->run_state().check();
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        classify_other_local_block();
        total_deal_kmer_count += kmer_count;
    }

    void calculate_block_prefix_counts(const unsigned char* kmer_data, const uint64_t kmer_count)
    {

        memset(local_block_prefix_counts.data(), 0, local_block_prefix_counts.size() * sizeof(uint32_t));

        // 可以考虑simd加速
        for (uint64_t i = 0; i < kmer_count; ++i)
        {
            const uint64_t prefix = get_root_prefix((kmer_data + (i)*kmer_bytes()));
            local_block_prefix_counts[prefix]++;
        }

        local_block_prefix_sums[0] = 0;
        for (uint64_t i = 1; i < local_block_prefix_counts.size(); ++i)
        {
            local_block_prefix_sums[i] = local_block_prefix_sums[i - 1] + local_block_prefix_counts[i - 1];
        }
    }

    void push_kmers_into_local_block_for_copy(const unsigned char* kmer_data, const uint64_t kmer_count)
    {
        for (uint64_t index = 0; index < kmer_count; index++)
        {
            const uint64_t prefix = get_root_prefix((kmer_data + (index)*kmer_bytes()));
            const uint64_t pos = local_block_prefix_sums[prefix]++;
            std::memcpy((local_block_for_copy.data() + (pos)*kmer_bytes()), (kmer_data + (index)*kmer_bytes()), kmer_bytes());
        }
    }

    void export_kmer_batch(const unsigned char* records, size_t count)
    {
        const size_t width = kmer_bytes();
        while (count != 0) {
            const size_t take = std::min<size_t>(count, EXPORT_KMER_BLOCK_CAPACITY - export_kmer_block_count);
            std::memcpy(export_block_ptr->k_mers.data() + export_kmer_block_count * width, records, take * width);
            export_kmer_block_count += take;
            records += take * width;
            count -= take;
#ifdef TEST_MODE
            total_kmers_exported += take;
#endif
            if (export_kmer_block_count == EXPORT_KMER_BLOCK_CAPACITY) {
                enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
                export_kmer_block_count = 0;
                char* raw_block_ptr = nullptr;
                dequeue_data_to_export_writer(raw_block_ptr);
                export_block_ptr = reinterpret_cast<ExportBlock*>(raw_block_ptr);
            }
        }
    }

    void export_one_kmer(const unsigned char* val)
    {
        std::memcpy((export_block_ptr->k_mers.data() + (export_kmer_block_count++) * kmer_bytes()), val, kmer_bytes());
        if (export_kmer_block_count == EXPORT_KMER_BLOCK_CAPACITY) [[unlikely]]
        {
            enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
            export_kmer_block_count = 0;
            char* raw_block_ptr = nullptr;
            dequeue_data_to_export_writer(raw_block_ptr);
            export_block_ptr = reinterpret_cast<ExportBlock*>(raw_block_ptr);
        }
#ifdef TEST_MODE
        total_kmers_exported++;
#endif
    }

    void push_first_occurrence(const unsigned char* val)
    {
        ++first_occurrence_count;
        export_one_kmer(val);
    }

    void classify_owned_local_block()
    {

        uint64_t local_block_count = 0;

        uint64_t read_offset = 0;
        for (uint64_t bloom_filter_index = 0; bloom_filter_index < local_bloom_filters.size(); bloom_filter_index++)
        {
            const uint32_t prefix = local_bloom_filter_index_to_prefix[bloom_filter_index];
            const uint32_t prefix_count = local_block_prefix_counts[prefix];
            if (prefix_count == 0)
            {
                continue;
            }

            uint32_t to_tree_count = 0;
            uint64_t run_count = 0;

            ConcurrentBloomFilter& bloom_filter = local_bloom_filters[bloom_filter_index];

            if (prefix_count < BLOOM_PREFETCH_DISTANCE / 2) [[unlikely]]
            {
                for (uint32_t i = 0; i < prefix_count; ++i)
                {
                    const unsigned char* val = (local_block_for_copy.data() + (read_offset)*kmer_bytes());

                    record_checkpoint();
                    const Occurrence occ = bloom_filter.insert(val);
                    if (occ == Occurrence::FIRST)
                    {
                        if (run_count != 0 && local_block_count != read_offset - run_count)
                            std::memmove(local_block_for_copy.data() + local_block_count * kmer_bytes(),
                                local_block_for_copy.data() + (read_offset - run_count) * kmer_bytes(), run_count * kmer_bytes());
                        local_block_count += run_count;
                        run_count = 0;
                        push_first_occurrence(val);
                    }
                    else
                    {
                        ++run_count;
                        ++to_tree_count;
                    }

                    read_offset++;
                }

                if (run_count != 0 && local_block_count != read_offset - run_count)
                    std::memmove(local_block_for_copy.data() + local_block_count * kmer_bytes(),
                        local_block_for_copy.data() + (read_offset - run_count) * kmer_bytes(), run_count * kmer_bytes());
                local_block_count += run_count;

                local_block_prefix_counts[prefix] = to_tree_count;

#ifdef TEST_MODE
                total_kmers_send_to_tree += to_tree_count;
#endif
                continue;
            }

            std::array<typename ConcurrentBloomFilter::InsertProbe, BLOOM_PREFETCH_DISTANCE> probes;
            const uint64_t prefix_begin = read_offset;
            const uint32_t warmup_count = (prefix_count < BLOOM_PREFETCH_DISTANCE) ? prefix_count : BLOOM_PREFETCH_DISTANCE;

            for (uint32_t i = 0; i < warmup_count; ++i)
            {
                const uint64_t idx = prefix_begin + i;
                probes[i] = bloom_filter.prepare_insert((local_block_for_copy.data() + (idx)*kmer_bytes()));
                bloom_filter.prefetch_insert(probes[i]);
            }

            for (uint32_t i = 0; i < prefix_count; ++i)
            {
                const uint32_t slot = i % BLOOM_PREFETCH_DISTANCE;
                const uint64_t idx = prefix_begin + i;
                const unsigned char* val = (local_block_for_copy.data() + (idx)*kmer_bytes());

                record_checkpoint();
                const Occurrence occ = bloom_filter.insert_prepared(probes[slot]);

                const uint32_t next_i = i + BLOOM_PREFETCH_DISTANCE;
                if (next_i < prefix_count)
                {
                    const uint64_t next_idx = prefix_begin + next_i;
                    probes[slot] = bloom_filter.prepare_insert((local_block_for_copy.data() + (next_idx)*kmer_bytes()));
                    bloom_filter.prefetch_insert(probes[slot]);
                }

                if (occ == Occurrence::FIRST)
                {
                    if (run_count != 0 && local_block_count != idx - run_count)
                        std::memmove(local_block_for_copy.data() + local_block_count * kmer_bytes(),
                            local_block_for_copy.data() + (idx - run_count) * kmer_bytes(), run_count * kmer_bytes());
                    local_block_count += run_count;
                    run_count = 0;
                    push_first_occurrence(val);
                }
                else
                {
                    ++run_count;
                    ++to_tree_count;
                }
            }

            read_offset += prefix_count;
            if (run_count != 0 && local_block_count != read_offset - run_count)
                std::memmove(local_block_for_copy.data() + local_block_count * kmer_bytes(),
                    local_block_for_copy.data() + (read_offset - run_count) * kmer_bytes(), run_count * kmer_bytes());
            local_block_count += run_count;

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

    void classify_other_local_block()
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
            uint64_t run_count = 0;

            // Shared single-layer Bloom filter.
            ConcurrentBloomFilter& bloom_filter = *local_global_bloom_filter[prefix];

            for (uint32_t i = 0; i < prefix_count; i++)
            {
                const unsigned char* val = (local_block_for_copy.data() + (read_offset)*kmer_bytes());

                record_checkpoint();
                const Occurrence occ = bloom_filter.insert(val);
                if (occ == Occurrence::FIRST)
                {
                    if (run_count != 0 && local_block_count != read_offset - run_count)
                        std::memmove(local_block_for_copy.data() + local_block_count * kmer_bytes(),
                            local_block_for_copy.data() + (read_offset - run_count) * kmer_bytes(), run_count * kmer_bytes());
                    local_block_count += run_count;
                    run_count = 0;
                    push_first_occurrence(val);
                }
                else
                {
                    ++run_count;
                    ++to_tree_count;
                }

                read_offset++;
            }

            if (run_count != 0 && local_block_count != read_offset - run_count)
                std::memmove(local_block_for_copy.data() + local_block_count * kmer_bytes(),
                    local_block_for_copy.data() + (read_offset - run_count) * kmer_bytes(), run_count * kmer_bytes());
            local_block_count += run_count;

            local_block_prefix_counts[prefix] = to_tree_count;
#ifdef TEST_MODE
            total_kmers_send_to_tree += to_tree_count;
#endif
        }

        if (local_block_count > 0) [[likely]]
        {
            uint32_t cnt = 0;
            for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); i++)
            {
                prefix_ordered_by_owner[cnt] = i;
                cnt = (local_block_prefix_counts[i] > 0) ? cnt + 1 : cnt;
            }
            tree->main_add_kmer_block_with_local_root_nodes(local_block_for_copy, local_block_prefix_counts, prefix_ordered_by_owner, cnt, local_root_nodes.data());
        }
    }

    uint64_t get_root_prefix(const unsigned char* k_mer) const
    {
        return k_mer[0];
    }

    void enqueue_content_to_export_writer(const content_type& content)
    {
        RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* export_pool = tree->get_export_ring_pool();

        if (export_pool->producer_try_enqueue(content))
        {
            enqueue_to_export_writer_backoff.double_decay();
            return;
        }
        do
        {

#ifdef TEST_MODE
            producer_enqueue_spin_time++;
#endif
            enqueue_to_export_writer_backoff.backoff();
        }
        while (!export_pool->producer_try_enqueue(content));

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
        do
        {

#ifdef TEST_MODE
            producer_dequeue_spin_time++;
#endif
            dequeue_from_export_writer_backoff.backoff();
        }
        while (!export_pool->producer_try_dequeue(data));

        dequeue_from_export_writer_backoff.decay();
    }
};

#endif // FASTQCLASSIFIER_HEADER
