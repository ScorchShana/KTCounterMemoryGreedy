#ifndef FASTQCLASSIFIER_HEADER
#define FASTQCLASSIFIER_HEADER

#include "definition.h"
#include "kmer.h"
#include "RingMemoryPool.h"
#include "NewKmerTree.h"
#include "SplitMix.h"
#include "MPSCRingQueue.h"
#include "ConcurrentMemoryPool.h"
#include "SpinBackoff.h"
#include "RunState.h"

#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

template<typename CountType = uint32_t>
class FastqClassifier
{
    friend struct FastqClassifierTestAccess;

    static constexpr int SLEEP_THRESHOLD = 128 + 16;
    static constexpr int YIELD_THRESHOLD = 128;
    static constexpr int MAX_BACKOFF = 32;
    static constexpr uint32_t MAX_CHECK_LOCAL_ROUND = 128;
    static constexpr uint32_t MIN_CHECK_LOCAL_ROUND = 16;

    uint32_t classifier_index;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_classifier_ring_pool;
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>* classify_task_queue;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    KmerTree<CountType>* tree;

    std::array<node<CountType>, 1ULL << (2 * ROOT_BASES)> local_root_nodes{};
    std::array<uint8_t, 1ULL << (2 * ROOT_BASES)> local_prefix_owners{};
    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> local_block_prefix_counts{};
    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> local_block_prefix_sums{};
    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> prefix_ordered_by_owner{};
    std::array<unsigned char, PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE> local_block_for_copy{};
    std::vector<uint32_t> local_owned_preifx{};

    uint32_t check_local_first_round = 16;
    uint32_t local_first_round = 0;
    SplitMix64 rng;
    ConcurrentMemoryPool* memory_pool_;

public:
    uint64_t total_deal_kmer_count = 0;
#ifdef TEST_MODE
    uint64_t consumer_enqueue_spin_time{0};
    uint64_t consumer_dequeue_spin_time{0};
    bool not_first_flag = false;
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
        PhaseBarrier& in_barrier)
        : classifier_index(in_classifier_index), parser_classifier_ring_pool(in_ring_pool),
        classify_task_queue(in_classify_task_queue), global_classifier_task_queue(in_global_classifier_task_queue),
        tree(in_tree), memory_pool_(in_memory_pool)
    {
        KmerTree<CountType>::initialize_roots(local_root_nodes.data());
        local_prefix_owners = prefix_owners;
        for (uint32_t prefix = 0; prefix < local_prefix_owners.size(); ++prefix) {
            if (local_prefix_owners[prefix] == classifier_index) local_owned_preifx.push_back(prefix);
        }
        in_barrier.arrive_and_wait();
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
    }

    void finish_tree_input()
    {
        tree->flush_local_root_nodes(local_root_nodes.data(), rng());

#ifdef TEST_MODE
        classifier_wait_cycles = tree->classifier_wait_cycles;
#endif
    }


private:
    void process_owned_block(const unsigned char* kmer_data, const uint64_t kmer_count)
    {
        memory_pool_->run_state().check();
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        classify_owned_local_block();
        total_deal_kmer_count += kmer_count;
#ifdef TEST_MODE
        total_kmers_send_to_tree += kmer_count;
#endif
    }

    void process_other_block(const unsigned char* kmer_data, const uint64_t kmer_count)
    {
        memory_pool_->run_state().check();
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        classify_other_local_block();
        total_deal_kmer_count += kmer_count;
#ifdef TEST_MODE
        total_kmers_send_to_tree += kmer_count;
#endif
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

    void classify_owned_local_block()
    {
        uint32_t count = 0;
        for (uint32_t prefix : local_owned_preifx) {
            if (local_block_prefix_counts[prefix] != 0) prefix_ordered_by_owner[count++] = prefix;
        }
        if (count != 0) {
            tree->main_add_kmer_block_with_local_root_nodes(local_block_for_copy, local_block_prefix_counts,
                prefix_ordered_by_owner, count, local_root_nodes.data());
        }
    }

    void classify_other_local_block()
    {
        uint32_t count = 0;
        for (uint32_t prefix = 0; prefix < local_block_prefix_counts.size(); ++prefix) {
            if (local_block_prefix_counts[prefix] != 0) prefix_ordered_by_owner[count++] = prefix;
        }
        if (count != 0) {
            tree->main_add_kmer_block_with_local_root_nodes(local_block_for_copy, local_block_prefix_counts,
                prefix_ordered_by_owner, count, local_root_nodes.data());
        }
    }

    uint64_t get_root_prefix(const unsigned char* k_mer) const
    {
        return k_mer[0];
    }
};

#endif // FASTQCLASSIFIER_HEADER
