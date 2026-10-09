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
#include "ReplacingHashMap.h"

#include <array>
#include <algorithm>
#include <bit>
#include <cstdint>
#include <bitset>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>
#include <barrier>

template <uint32_t N>
class FastqClassifier
{

    // 每线程 FIRST 缓存：key 数组不超过 512 MiB，控制字节和位图另计。
    static constexpr size_t FIRST_MAP_KEY_BYTES = 512ULL * 1024 * 1024;
    static constexpr size_t FIRST_MAP_CAPACITY =
        std::bit_floor(FIRST_MAP_KEY_BYTES / sizeof(kmer<N>));
    using FirstOccurrenceMap = ReplacingHashMap<N, FIRST_MAP_CAPACITY>;

    static_assert(FIRST_MAP_CAPACITY >= 16,
        "FIRST map must hold at least one probe group");
    static_assert(FirstOccurrenceMap::capacity() == FIRST_MAP_CAPACITY,
        "FIRST map slot count must match the key memory budget");
    static_assert(FirstOccurrenceMap::capacity() * sizeof(kmer<N>) <= FIRST_MAP_KEY_BYTES,
        "FIRST map keys must fit within 512 MiB");

    // 自旋参数
    static constexpr int SLEEP_THRESHOLD = 128 + 16;
    static constexpr int YIELD_THRESHOLD = 128;
    static constexpr int MAX_BACKOFF = 32;

    static constexpr uint64_t EXPORT_KMER_BLOCK_CAPACITY = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
    static constexpr size_t INPUT_BLOCK_CAPACITY = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
    static constexpr uint32_t BLOOM_PREFETCH_DISTANCE = 16; // 预取 Bloom Filter 的距离（单位：k-mer数量）

    static constexpr uint32_t MAX_CHECK_LOCAL_ROUND = 128;
    static constexpr uint32_t MIN_CHECK_LOCAL_ROUND = 16;

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

    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> prefix_ordered_by_owner{};

    // Input starts at B; output starts at 0 and may expand to 2B entries.
    // At most two outputs per input keep writes behind unread input.
    std::array<kmer<N>, 2 * INPUT_BLOCK_CAPACITY> local_block_for_copy{};

    ExportBlock<N>* export_block_ptr = nullptr;
    uint64_t export_kmer_block_count = 0;

    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> enqueue_to_export_writer_backoff;
    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_from_export_writer_backoff;

    std::vector<ConcurrentBloomFilter<N>> local_bloom_filters;
    std::vector<uint32_t> local_owned_preifx{};

    uint32_t check_local_first_round = 16;
    uint32_t local_first_round = 0;


    SplitMix64 rng;

    // owned/other 路径共用，只有本 classifier 线程访问。
    FirstOccurrenceMap* first_occurrence_map_ = nullptr;

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
        ConcurrentMemoryPool* in_memory_pool,
        std::barrier<>& in_barrier)
        : k_len(in_k_len), classifier_index(in_classifier_index), global_classifier_task_queue(in_global_classifier_task_queue), parser_classifier_ring_pool(in_ring_pool), classify_task_queue(in_classify_task_queue), tree(in_tree),
        export_block_ptr(nullptr), export_kmer_block_count(0)
    {
        char* raw_block_ptr = nullptr;
        dequeue_data_to_export_writer(raw_block_ptr);
        export_block_ptr = reinterpret_cast<ExportBlock<N> *>(raw_block_ptr);

        // pool large 分配（须 init_arenas 之后）；空槽的 key 无须初始化。
        void* map_mem = in_memory_pool->allocate_large(sizeof(FirstOccurrenceMap));
        first_occurrence_map_ = new (map_mem) FirstOccurrenceMap;

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
                local_owned_preifx.push_back(i);
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


                    kmer<N>* kmer_data = reinterpret_cast<kmer<N> *>(content.data);
                    const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                    if (local_prefix_owners[get_root_prefix(kmer_data[0])] == classifier_index) [[likelyF]]
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
        export_first_occurrences_on_finish();
        enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
        export_kmer_block_count = 0;

        tree->flush_local_root_nodes(local_root_nodes.data(), rng());

#ifdef TEST_MODE
        classifier_wait_cycles = tree->classifier_wait_cycles;
#endif
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
            local_block_for_copy[INPUT_BLOCK_CAPACITY + pos] = kmer_data[index];
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

    void push_occurrence_first(const kmer<N>& val) noexcept
    {
        kmer<N> evicted_key;
        if (first_occurrence_map_->insert(val, evicted_key)) [[unlikely]]
        {
            export_one_kmer(evicted_key);
        }
    }

    uint32_t route_occurrence(const kmer<N>& val, Occurrence occ,
        uint64_t& output_count) noexcept
    {
        if (occ == Occurrence::FIRST)
        {
            push_occurrence_first(val);
            return 0;
        }

        const bool recovered_first = occ == Occurrence::SECOND
            && first_occurrence_map_->erase(val);
        local_block_for_copy[output_count++] = val;
        if (recovered_first)
        {
            local_block_for_copy[output_count++] = val;
            return 2;
        }
        return 1;
    }

    // One-shot terminal export; the FIRST map is not accessed afterward.
    void export_first_occurrences_on_finish() noexcept
    {
        first_occurrence_map_->for_each(
            [&](const kmer<N>& val) noexcept
            {
                export_one_kmer(val);
            });
    }

    void classify_owned_local_block() noexcept
    {

        uint64_t local_block_count = 0;

        uint64_t read_offset = INPUT_BLOCK_CAPACITY;
        for (uint64_t prefix = 0; prefix < local_block_prefix_counts.size(); prefix++)
        {
            const uint32_t prefix_count = local_block_prefix_counts[prefix];
            if (prefix_count == 0)
            {
                continue;
            }

            uint32_t to_tree_count = 0;

            const uint32_t bloom_filter_index = prefix_to_bloom_filter_index[prefix];

            ConcurrentBloomFilter<N>& bloom_filter = local_bloom_filters[bloom_filter_index];

            if (prefix_count < BLOOM_PREFETCH_DISTANCE / 2) [[unlikely]]
            {
                for (uint32_t i = 0; i < prefix_count; ++i)
                {
                    const kmer<N> val = local_block_for_copy[read_offset];

                    const Occurrence occ = bloom_filter.insert(val);
                    to_tree_count += route_occurrence(val, occ, local_block_count);

                    read_offset++;
                }

                local_block_prefix_counts[prefix] = to_tree_count;

#ifdef TEST_MODE
                total_kmers_send_to_tree += to_tree_count;
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
                const kmer<N> val = local_block_for_copy[idx];

                const Occurrence occ = bloom_filter.insert_prepared(probes[slot]);

                const uint32_t next_i = i + BLOOM_PREFETCH_DISTANCE;
                if (next_i < prefix_count)
                {
                    const uint64_t next_idx = prefix_begin + next_i;
                    probes[slot] = bloom_filter.prepare_insert(local_block_for_copy[next_idx]);
                    bloom_filter.prefetch_insert(probes[slot]);
                }

                to_tree_count += route_occurrence(val, occ, local_block_count);
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

        uint64_t read_offset = INPUT_BLOCK_CAPACITY;
        for (uint64_t prefix = 0; prefix < local_block_prefix_counts.size(); prefix++)
        {
            const uint32_t prefix_count = local_block_prefix_counts[prefix];
            if (prefix_count == 0)
            {
                continue;
            }

            uint32_t to_tree_count = 0;

            // 共享 Bloom filter；FIRST 缓存仍属于当前 classifier。
            ConcurrentBloomFilter<N>& bloom_filter = *local_global_bloom_filter[prefix];

            for (uint32_t i = 0; i < prefix_count; i++)
            {
                const kmer<N> val = local_block_for_copy[read_offset];

                const Occurrence occ = bloom_filter.insert(val);
                to_tree_count += route_occurrence(val, occ, local_block_count);

                read_offset++;
            }

            local_block_prefix_counts[prefix] = to_tree_count;
#ifdef TEST_MODE
            // Includes recovered FIRST occurrences; exports are counted separately.
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
