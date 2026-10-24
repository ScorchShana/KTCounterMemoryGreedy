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
#include "ConcurrentFirstBlockRing.h"
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
    const size_t INPUT_BLOCK_CAPACITY = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
    static constexpr uint32_t BLOOM_PREFETCH_DISTANCE = 16; // 预取 Bloom Filter 的距离（单位：k-mer数量）
    static constexpr size_t EVICTED_BUFFER_BYTES = 32ULL * 1024;

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

    // Each input emits at most one ordinary output. Compact in place;
    // delayed FIRST recovery has independent storage and prefix counts.
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
    const bool first_cache_enabled_;
    const int first_arena_;
    size_t checkpoint_bytes_ = 0;
    ConcurrentFirstBlockRing* first_occurrence_ring_;
    uint8_t* first_block_;
    size_t first_block_count_ = 0;
    const size_t first_block_capacity_;
    std::array<uint8_t, EVICTED_BUFFER_BYTES> evicted_buffer_;
    size_t evicted_count_ = 0;
    size_t evicted_capacity_ = 0;
    unsigned char evicted_grouped_[PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE];
    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> evicted_prefix_counts_{};
    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> evicted_prefix_offsets_{};

    struct RepeatLookup
    {
        ConcurrentBloomFilter* filter;
        typename ConcurrentBloomFilter::InsertProbe probe;
        uint32_t prefix;
    };

public:
    uint64_t total_deal_kmer_count = 0;
    uint64_t low_records_scanned = 0;
    uint64_t low_records_recovered = 0;
    uint64_t first_drain_singletons = 0;
#ifdef TEST_MODE
    uint64_t first_direct_pending = 0;
    uint64_t first_memory_to_tree = 0, first_memory_discarded = 0, first_memory_kept = 0;
    uint64_t first_memory_tree_batches = 0;
    uint64_t low_tree_batches = 0;
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
    uint64_t first_occurrence_count = 0;
    uint64_t repeat_query_count = 0;
    uint64_t recovered_first_count = 0;
#endif

    explicit FastqClassifier(uint32_t in_classifier_index,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_ring_pool,
        MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>* in_classify_task_queue,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* in_global_classifier_task_queue,
        KmerTree<CountType>* in_tree,
        ConcurrentMemoryPool* in_memory_pool,
        PhaseBarrier& in_barrier,
        ConcurrentFirstBlockRing* in_first_ring,
        uint32_t in_producer_rank,
        bool first_cache_enabled = false,
        const std::array<void*, 256>* bloom_storage = nullptr)
        : classifier_index(in_classifier_index), parser_classifier_ring_pool(in_ring_pool),
        classify_task_queue(in_classify_task_queue), global_classifier_task_queue(in_global_classifier_task_queue),
        tree(in_tree),
        memory_pool_(in_memory_pool), first_cache_enabled_(first_cache_enabled), first_arena_(in_classifier_index% in_memory_pool->arena_count()),
        first_occurrence_ring_(in_first_ring), first_block_(in_first_ring ? in_first_ring->initial_block(in_producer_rank) : nullptr),
        first_block_capacity_(ConcurrentFirstBlockRing::BLOCK_BYTES / kmer_bytes())
    {
        KmerTree<CountType>::initialize_roots(local_root_nodes.data());
        char* raw_block_ptr = nullptr;
        dequeue_data_to_export_writer(raw_block_ptr);
        export_block_ptr = reinterpret_cast<ExportBlock*>(raw_block_ptr);

        const size_t packed_bytes = kmer_bytes();
        evicted_capacity_ = EVICTED_BUFFER_BYTES / packed_bytes;

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

        in_barrier.arrive_and_wait();

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
                            memory_pool_->seal_thread_first_if_reclaiming();
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

                    memory_pool_->seal_thread_first_if_reclaiming();
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
                        memory_pool_->seal_thread_first();
                        parser_classifier_ring_pool->consumer_enqueue(content.data);
                    }
                }

                while (global_classifier_task_queue->try_dequeue(content))
                {
                    unsigned char* kmer_data = reinterpret_cast<unsigned char*>(content.data);
                    const uint64_t kmer_count = content.length; // length 就是 k-mer数量
                    process_other_block(kmer_data, kmer_count);
                    if (!parser_classifier_ring_pool->consumer_try_enqueue(content.data)) {
                        memory_pool_->seal_thread_first();
                        parser_classifier_ring_pool->consumer_enqueue(content.data);
                    }
                }

                break;
            }
        }
        memory_pool_->seal_thread_first();
        enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
        export_kmer_block_count = 0;
        export_block_ptr = nullptr;
    }

    template <bool WRITE_LOW>
    void reconcile_low()
    {
        if constexpr (WRITE_LOW) {
            char* raw_block_ptr = nullptr;
            dequeue_data_to_export_writer(raw_block_ptr);
            export_block_ptr = reinterpret_cast<ExportBlock*>(raw_block_ptr);
        }
        if (first_occurrence_ring_) first_drain_singletons += drain_first_occurrences<WRITE_LOW>();
        size_t hit_count = 0;
        local_first_round = 0;
        check_local_first_round = MIN_CHECK_LOCAL_ROUND * 2;
        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> backoff;
        content_type content;
        while (true)
        {
            if (parser_classifier_ring_pool->producer_finished()) {
                while (classify_task_queue->try_dequeue(content))
                    reconcile_low_block<WRITE_LOW>(content, hit_count);
                while (global_classifier_task_queue->try_dequeue(content))
                    reconcile_low_block<WRITE_LOW>(content, hit_count);
                break;
            }
            if (try_dequeue_low_task(content)) {
                backoff.decay();
                reconcile_low_block<WRITE_LOW>(content, hit_count);
            }
            else {
                backoff.backoff();
            }
        }
        flush_recovered_tree_batch(hit_count);
        if constexpr (WRITE_LOW) {
            enqueue_content_to_export_writer({ reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count });
            export_kmer_block_count = 0;
            export_block_ptr = nullptr;
        }
    }

    size_t first_tail_records() const noexcept { return memory_pool_->thread_first_records(); }

    template<bool WRITE_LOW>
    void process_memory_first()
    {
        if (!first_cache_enabled_) return;
        const size_t width = kmer_bytes();
        typename KmerTree<CountType>::FirstScanBatch batch;
        if constexpr (WRITE_LOW) {
            char* raw = nullptr;
            dequeue_data_to_export_writer(raw);
            export_block_ptr = reinterpret_cast<ExportBlock*>(raw);
        }
        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> backoff;
        for (;;) {
            memory_pool_->run_state().check();
            for (int i = 0; i < memory_pool_->arena_count(); ++i) memory_pool_->poll_first_progress(i);
            if (memory_pool_->first_empty()) break;
            if (tree->try_prepare_first_scan(batch, ConcurrentMemoryPool::first_segment_record_capacity())
                != ConcurrentMemoryPool::RamReserveStatus::Ready) { backoff.backoff(); continue; }
            ConcurrentMemoryPool::FirstSegmentView view;
            const auto status = memory_pool_->try_claim_first(first_arena_, ConcurrentMemoryPool::FirstConsumer::Scan, view);
            if (status == ConcurrentMemoryPool::FirstClaimStatus::Done) break;
            if (status == ConcurrentMemoryPool::FirstClaimStatus::Busy) { backoff.backoff(); continue; }
            size_t segment_hits = 0, segment_misses = 0;
            try {
                for (size_t offset = 0; offset < view.record_count;) {
                    memory_pool_->run_state().check();
                    const size_t take = std::min(evicted_capacity_, view.record_count - offset);
                    const auto* records = view.records + offset * width;
                    std::array<RepeatLookup, BLOOM_PREFETCH_DISTANCE> lookups;
                    for (size_t i = 0; i < std::min<size_t>(take, BLOOM_PREFETCH_DISTANCE); ++i)
                        prepare_repeat_lookup(lookups[i], records + i * width);
                    size_t hit_run = 0, miss_run = 0;
                    for (size_t i = 0; i < take; ++i) {
                        const auto lookup = lookups[i % BLOOM_PREFETCH_DISTANCE];
                        const bool repeated = lookup.filter->contains_repeat_prepared(lookup.probe);
                        if (i + BLOOM_PREFETCH_DISTANCE < take)
                            prepare_repeat_lookup(lookups[i % BLOOM_PREFETCH_DISTANCE], records + (i + BLOOM_PREFETCH_DISTANCE) * width);
                        if (repeated) {
                            if constexpr (WRITE_LOW) {
                                if (miss_run) { export_kmer_batch(records + (i - miss_run) * width, miss_run); miss_run = 0; }
                            }
                            ++hit_run;
                            ++segment_hits;
                        } else {
                            if (hit_run) { tree->append_first_scan_hits(batch, records + (i - hit_run) * width, hit_run, local_root_nodes.data()); hit_run = 0; }
                            if constexpr (WRITE_LOW) ++miss_run;
                            ++segment_misses;
                        }
                    }
                    if (hit_run) tree->append_first_scan_hits(batch, records + (take - hit_run) * width, hit_run, local_root_nodes.data());
                    if constexpr (WRITE_LOW) if (miss_run) export_kmer_batch(records + (take - miss_run) * width, miss_run);
                    offset += take;
                }
            } catch (...) { memory_pool_->complete_first(view); throw; }
            memory_pool_->complete_first(view);
            tree->submit_first_scan(batch);
            first_drain_singletons += segment_misses;
#ifdef TEST_MODE
            repeat_query_count += segment_hits + segment_misses;
            first_memory_to_tree += segment_hits;
            recovered_first_count += segment_hits;
            total_kmers_send_to_tree += segment_hits;
            if (segment_hits) ++first_memory_tree_batches; // Nonempty source batches, no 64 KiB input staging.
            if constexpr (WRITE_LOW) first_memory_kept += segment_misses;
            else first_memory_discarded += segment_misses;
#endif
            backoff.reset();
        }
        tree->release_unused_first_scan_pages(batch);
        if constexpr (WRITE_LOW) {
            enqueue_content_to_export_writer({reinterpret_cast<char*>(export_block_ptr), export_kmer_block_count});
            export_kmer_block_count = 0;
            export_block_ptr = nullptr;
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
    void append_recovered_tree_hits(const unsigned char* records, size_t count, size_t& hit_count)
    {
        const size_t width = kmer_bytes();
        while (count != 0) {
            const size_t take = std::min(count, INPUT_BLOCK_CAPACITY - hit_count);
            std::memcpy(evicted_grouped_ + hit_count * width, records, take * width);
            hit_count += take;
            records += take * width;
            count -= take;
            if (hit_count == INPUT_BLOCK_CAPACITY) flush_recovered_tree_batch(hit_count);
        }
    }

    void flush_recovered_tree_batch(size_t& hit_count)
    {
        if (hit_count == 0) return;
        calculate_block_prefix_counts(evicted_grouped_, hit_count);
        push_kmers_into_local_block_for_copy(evicted_grouped_, hit_count);
        uint32_t root_count = 0;
        for (uint32_t prefix = 0; prefix < local_block_prefix_counts.size(); ++prefix)
            if (local_block_prefix_counts[prefix] != 0) prefix_ordered_by_owner[root_count++] = prefix;
        tree->main_add_kmer_block_with_local_root_nodes(local_block_for_copy, local_block_prefix_counts,
            prefix_ordered_by_owner, root_count, local_root_nodes.data());
        low_records_recovered += hit_count;
#ifdef TEST_MODE
        ++low_tree_batches;
#endif
#ifdef TEST_MODE
        total_kmers_send_to_tree += hit_count;
#endif
        hit_count = 0;
    }

    void record_checkpoint()
    {
        checkpoint_bytes_ += kmer_bytes();
        if (checkpoint_bytes_ >= BLOCK_SIZE) {
            checkpoint_bytes_ = 0;
            memory_pool_->run_state().check();
            if (first_cache_enabled_) memory_pool_->seal_thread_first_if_reclaiming();
        }
    }
    bool try_dequeue_low_task(content_type& content)
    {
        if (local_first_round < check_local_first_round) {
            ++local_first_round;
            return classify_task_queue->try_dequeue(content) || global_classifier_task_queue->try_dequeue(content);
        }
        const uint64_t global_queue_size = global_classifier_task_queue->size();
        if (global_queue_size > PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY / 2)
            check_local_first_round = std::max(check_local_first_round / 2, MIN_CHECK_LOCAL_ROUND);
        else if (global_queue_size < PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY / 4)
            check_local_first_round = std::min(check_local_first_round * 2, MAX_CHECK_LOCAL_ROUND);
        local_first_round = 0;
        return global_classifier_task_queue->try_dequeue(content) || classify_task_queue->try_dequeue(content);
    }

    template <bool WRITE_LOW>
    void reconcile_low_block(const content_type& content, size_t& hit_count)
    {
        memory_pool_->run_state().check();
        const size_t width = kmer_bytes();
        std::array<RepeatLookup, BLOOM_PREFETCH_DISTANCE> lookups;
        const auto* keys = reinterpret_cast<const unsigned char*>(content.data);
        for (size_t i = 0; i < std::min<size_t>(content.length, BLOOM_PREFETCH_DISTANCE); ++i)
            prepare_repeat_lookup(lookups[i], keys + i * width);

        size_t run_count = 0;
        [[maybe_unused]] size_t miss_run_count = 0;
        for (size_t i = 0; i < content.length; ++i)
        {
            const auto lookup = lookups[i % BLOOM_PREFETCH_DISTANCE];
            const bool repeated = lookup.filter->contains_repeat_prepared(lookup.probe);
            if (i + BLOOM_PREFETCH_DISTANCE < content.length)
                prepare_repeat_lookup(lookups[i % BLOOM_PREFETCH_DISTANCE],
                    keys + (i + BLOOM_PREFETCH_DISTANCE) * width);
            if (repeated) {
                if constexpr (WRITE_LOW) {
                    if (miss_run_count != 0) {
                        export_kmer_batch(keys + (i - miss_run_count) * width, miss_run_count);
                        miss_run_count = 0;
                    }
                }
                ++run_count;
            } else {
                if (run_count != 0) {
                    append_recovered_tree_hits(keys + (i - run_count) * width, run_count, hit_count);
                    run_count = 0;
                }
                if constexpr (WRITE_LOW) ++miss_run_count;
            }
        }
        if (run_count != 0)
            append_recovered_tree_hits(keys + (content.length - run_count) * width, run_count, hit_count);
        if constexpr (WRITE_LOW) {
            if (miss_run_count != 0)
                export_kmer_batch(keys + (content.length - miss_run_count) * width, miss_run_count);
        }
        low_records_scanned += content.length;
        parser_classifier_ring_pool->consumer_enqueue(content.data);
    }

    void process_owned_block(const unsigned char* kmer_data, const uint64_t kmer_count)
    {
        memory_pool_->run_state().check();
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        classify_owned_local_block();
        if (first_cache_enabled_) memory_pool_->seal_thread_first_if_reclaiming();
        total_deal_kmer_count += kmer_count;
    }

    void process_other_block(const unsigned char* kmer_data, const uint64_t kmer_count)
    {
        memory_pool_->run_state().check();
        calculate_block_prefix_counts(kmer_data, kmer_count);
        push_kmers_into_local_block_for_copy(kmer_data, kmer_count);
        classify_other_local_block();
        if (first_cache_enabled_) memory_pool_->seal_thread_first_if_reclaiming();
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
#ifdef TEST_MODE
        first_occurrence_count++;
#endif
        if (!first_occurrence_ring_) {
            if (!first_cache_enabled_ || !memory_pool_->cache_first_record(first_arena_, val)) {
                export_one_kmer(val);
#ifdef TEST_MODE
                ++first_direct_pending;
#endif
            }
            return;
        }
        const size_t width = kmer_bytes();
        std::memcpy(first_block_ + first_block_count_ * width, val, width);
        if (++first_block_count_ == first_block_capacity_)
        {
            if (first_occurrence_ring_->publish_full_block(first_block_))
                append_evicted_records(first_block_, first_block_capacity_);
            first_block_count_ = 0;
        }
    }

    template<bool WRITE_MISSES = true>
    uint64_t append_evicted_records(const uint8_t* records, size_t count)
    {
        const size_t width = kmer_bytes();
        uint64_t misses = 0;
        while (count != 0)
        {
            const size_t take = std::min(count, evicted_capacity_ - evicted_count_);
            std::memcpy(evicted_buffer_.data() + evicted_count_ * width, records, take * width);
            evicted_count_ += take;
            records += take * width;
            count -= take;
            if (evicted_count_ == evicted_capacity_) misses += flush_evicted_batch<WRITE_MISSES>();
        }
        return misses;
    }

    void group_evicted()
    {
        const size_t width = kmer_bytes();
        evicted_prefix_counts_.fill(0);
        for (size_t i = 0; i < evicted_count_; ++i)
        {
            ++evicted_prefix_counts_[kmer::root_prefix(evicted_buffer_.data() + i * width)];
        }

        uint32_t offset = 0;
        for (size_t prefix = 0; prefix < evicted_prefix_counts_.size(); ++prefix)
        {
            evicted_prefix_offsets_[prefix] = offset;
            offset += evicted_prefix_counts_[prefix];
        }

        // Scatter each compact record into its root's contiguous range.
        for (size_t i = 0; i < evicted_count_; ++i)
        {
            const uint8_t* record = evicted_buffer_.data() + i * width;
            const uint32_t prefix = kmer::root_prefix(record);
            std::memcpy(evicted_grouped_ + evicted_prefix_offsets_[prefix]++ * width, record, width);
        }
    }

    void prepare_repeat_lookup(RepeatLookup& lookup, const unsigned char* key) const
    {
        const uint32_t prefix = get_root_prefix(key);
        auto* filter = local_global_bloom_filter[prefix];
        lookup = { filter, filter->prepare_insert(key), prefix };
        filter->prefetch_contains(lookup.probe);
    }

    template<bool WRITE_MISSES = true>
    uint64_t flush_evicted_batch()
    {
        if (evicted_count_ == 0)
        {
            return 0;
        }
        group_evicted();

        // 记录已按 root 连续排列，命中项可以直接向前收集。
        evicted_prefix_counts_.fill(0);
        std::array<RepeatLookup, BLOOM_PREFETCH_DISTANCE> lookups;
        for (size_t i = 0; i < std::min<size_t>(evicted_count_, BLOOM_PREFETCH_DISTANCE); ++i)
        {
            prepare_repeat_lookup(lookups[i], (evicted_grouped_ + (i)*kmer_bytes()));
        }

        size_t output_count = 0;
        size_t run_count = 0;
        for (size_t i = 0; i < evicted_count_; ++i)
        {
            const auto lookup = lookups[i % BLOOM_PREFETCH_DISTANCE];
            const unsigned char* val = (evicted_grouped_ + (i)*kmer_bytes());
            const bool repeated = lookup.filter->contains_repeat_prepared(lookup.probe);
            if (i + BLOOM_PREFETCH_DISTANCE < evicted_count_)
            {
                prepare_repeat_lookup(lookups[i % BLOOM_PREFETCH_DISTANCE], (evicted_grouped_ + (i + BLOOM_PREFETCH_DISTANCE) * kmer_bytes()));
            }
            if (repeated)
            {
                ++run_count;
                ++evicted_prefix_counts_[lookup.prefix];
            }
            else
            {
                if (run_count != 0 && output_count != i - run_count)
                    std::memmove(evicted_grouped_ + output_count * kmer_bytes(),
                        evicted_grouped_ + (i - run_count) * kmer_bytes(), run_count * kmer_bytes());
                output_count += run_count;
                run_count = 0;
                if constexpr (WRITE_MISSES) export_one_kmer(val);
            }
        }
        if (run_count != 0 && output_count != evicted_count_ - run_count)
            std::memmove(evicted_grouped_ + output_count * kmer_bytes(),
                evicted_grouped_ + (evicted_count_ - run_count) * kmer_bytes(), run_count * kmer_bytes());
        output_count += run_count;

        size_t read_offset = 0;
        for (uint32_t prefix = 0; prefix < evicted_prefix_counts_.size(); ++prefix)
        {
            const uint32_t count = evicted_prefix_counts_[prefix];
            if (count > 0)
            {
                tree->main_add_kmers_to_local_root(evicted_grouped_ + read_offset * kmer_bytes(), count, prefix, local_root_nodes.data());
                read_offset += count;
            }
        }
#ifdef TEST_MODE
        repeat_query_count += evicted_count_;
        recovered_first_count += output_count;
        total_kmers_send_to_tree += output_count;
#endif
        const uint64_t misses = evicted_count_ - output_count;
        evicted_count_ = 0;
        return misses;
    }

    template<bool WRITE_MISSES = true>
    uint64_t drain_first_occurrences()
    {
        uint64_t misses = append_evicted_records<WRITE_MISSES>(first_block_, first_block_count_);
        first_block_count_ = 0;
        uint64_t begin, end;
        while (first_occurrence_ring_->claim_drain_range(begin, end))
        {
            for (uint64_t slot = begin; slot < end; ++slot)
                misses += append_evicted_records<WRITE_MISSES>(
                    first_occurrence_ring_->drain_block(slot), first_block_capacity_);
        }
        return misses + flush_evicted_batch<WRITE_MISSES>();
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

            // Shared Bloom filter and node-local FIRST ring.
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
            // Delayed FIRST recovery is counted by flush_evicted_batch().
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
            memory_pool_->seal_thread_first_if_reclaiming();
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
            memory_pool_->seal_thread_first_if_reclaiming();
            dequeue_from_export_writer_backoff.backoff();
        }
        while (!export_pool->producer_try_dequeue(data));

        dequeue_from_export_writer_backoff.decay();
    }
};

#endif // FASTQCLASSIFIER_HEADER
