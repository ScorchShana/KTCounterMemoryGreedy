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
#include "HashSet.h"

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

    // 自旋参数
    static constexpr int SLEEP_THRESHOLD = 128;
    static constexpr int YIELD_THRESHOLD = 64;
    static constexpr int MAX_BACKOFF = 64;

    static constexpr uint64_t EXPORT_KMER_BLOCK_CAPACITY = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
    static constexpr uint32_t BLOOM_PREFETCH_DISTANCE = 4; // 预取 Bloom Filter 的距离（单位：k-mer数量）

    // Owned 双缓冲 + HashSet：编译期常量；Buf2 = Buf1/4；CAP = bit_ceil(Buf2/0.875)
    // per_thread ≈ (B1+B2)*sizeof(kmer) + sizeof(HashSet) + 2*TREE_CHUNK*sizeof(kmer)
    // static constexpr size_t OCC_BUF1_CAPACITY =
    //     std::max<size_t>(8 * EXPORT_KMER_BLOCK_CAPACITY,
    //         std::bit_ceil(static_cast<size_t>(
    //             4 * PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>) * 1.15)) * 4);
    static constexpr size_t OCC_BUF1_CAPACITY = 512 * 1024 * 1024 / sizeof(kmer<N>); // 512 M
    static constexpr size_t OCC_HASHSET_CAPACITY = std::bit_ceil(std::max<size_t>(size_t{ 32 }, OCC_BUF1_CAPACITY / 4));
    static constexpr size_t OCC_BUF2_CAPACITY = OCC_HASHSET_CAPACITY * 7 / 8;
    static constexpr size_t TREE_CHUNK_KMER_CAP =
        PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);

    static_assert(OCC_HASHSET_CAPACITY >= 32 && (OCC_HASHSET_CAPACITY & (OCC_HASHSET_CAPACITY - 1)) == 0,
        "OCC_HASHSET_CAPACITY must be a power of two and >= 32");
    static_assert(OCC_BUF2_CAPACITY <= OCC_HASHSET_CAPACITY * 7 / 8,
        "OCC_BUF2_CAPACITY must fit HashSet load factor");
    static_assert(OCC_BUF1_CAPACITY >= OCC_BUF2_CAPACITY && OCC_BUF2_CAPACITY >= 1,
        "OCC_BUF capacities invalid");

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

    // Owned FIRST/SECOND 缓冲与 HashSet：allocate_large（仅 owned 路径写入）
    kmer<N>* occ_buf1_ = nullptr;
    kmer<N>* occ_buf2_ = nullptr;
    size_t occ_buf1_size_ = 0;
    size_t occ_buf2_size_ = 0;
    HashSet<N, OCC_HASHSET_CAPACITY>* occ_hash_set_ = nullptr;

    // F-batch flush：无序累积 → counting sort → main_add（不得复用 local_block_for_copy）
    std::array<kmer<N>, TREE_CHUNK_KMER_CAP> occ_tree_chunk_{};
    std::array<kmer<N>, TREE_CHUNK_KMER_CAP> occ_tree_chunk_sorted_{};
    std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> occ_tree_chunk_prefix_counts_{};

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

        // Buf1/Buf2/HashSet：pool large 分配（须 init_arenas 之后；本线程 Arena）
        occ_buf1_ = static_cast<kmer<N>*>(
            in_memory_pool->allocate_large(OCC_BUF1_CAPACITY * sizeof(kmer<N>)));
        occ_buf2_ = static_cast<kmer<N>*>(
            in_memory_pool->allocate_large(OCC_BUF2_CAPACITY * sizeof(kmer<N>)));
        void* hash_set_mem = in_memory_pool->allocate_large(sizeof(HashSet<N, OCC_HASHSET_CAPACITY>));
        occ_hash_set_ = new (hash_set_mem) HashSet<N, OCC_HASHSET_CAPACITY>();
        occ_buf1_size_ = 0;
        occ_buf2_size_ = 0;

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
        if (occ_hash_set_ != nullptr)
        {
            occ_hash_set_->~HashSet();
            occ_hash_set_ = nullptr;
        }
        occ_buf1_ = nullptr;
        occ_buf2_ = nullptr;
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
                    if (local_prefix_owners[get_root_prefix(kmer_data[0])] == classifier_index) [[likely]]
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

        flush_occurrence_buffers();

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
        occ_buf1_[occ_buf1_size_++] = val;
        if (occ_buf1_size_ == OCC_BUF1_CAPACITY)
        {
            flush_occurrence_buffers();
        }
    }

    void push_occurrence_second(const kmer<N>& val) noexcept
    {
        occ_buf2_[occ_buf2_size_++] = val;
        if (occ_buf2_size_ == OCC_BUF2_CAPACITY)
        {
            flush_occurrence_buffers();
        }
    }

    // F-batch：对 occ_tree_chunk_[0..chunk_n) counting sort → occ_tree_chunk_sorted_ → main_add
    void commit_tree_chunk(const size_t chunk_n) noexcept
    {
        if (chunk_n == 0)
        {
            return;
        }

        std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> cnt{};
        for (size_t i = 0; i < chunk_n; ++i)
        {
            cnt[get_root_prefix(occ_tree_chunk_[i])]++;
        }

        std::array<uint32_t, 1ULL << (2 * ROOT_BASES)> scatter_pos{};
        scatter_pos[0] = 0;
        for (uint64_t p = 1; p < scatter_pos.size(); ++p)
        {
            scatter_pos[p] = scatter_pos[p - 1] + cnt[p - 1];
        }

        for (size_t i = 0; i < chunk_n; ++i)
        {
            const uint64_t p = get_root_prefix(occ_tree_chunk_[i]);
            occ_tree_chunk_sorted_[scatter_pos[p]++] = occ_tree_chunk_[i];
        }

        std::memset(occ_tree_chunk_prefix_counts_.data(), 0,
            occ_tree_chunk_prefix_counts_.size() * sizeof(uint32_t));
        for (uint64_t p = 0; p < cnt.size(); ++p)
        {
            occ_tree_chunk_prefix_counts_[p] = cnt[p];
        }

        tree->main_add_kmer_block_with_local_root_nodes(
            occ_tree_chunk_sorted_, occ_tree_chunk_prefix_counts_, local_root_nodes.data());

#ifdef TEST_MODE
        total_kmers_send_to_tree += chunk_n;
#endif
    }

    void flush_occurrence_buffers() noexcept
    {
        if (occ_buf1_size_ == 0 && occ_buf2_size_ == 0)
        {
            return;
        }

        occ_hash_set_->clear();

        size_t chunk_n = 0;

        auto append_tree_kmer = [&](const kmer<N>& k) noexcept
            {
                occ_tree_chunk_[chunk_n++] = k;
                if (chunk_n == TREE_CHUNK_KMER_CAP)
                {
                    commit_tree_chunk(chunk_n);
                    chunk_n = 0;
                }
            };

        for (size_t i = 0; i < occ_buf2_size_; ++i)
        {
            occ_hash_set_->insert(occ_buf2_[i]);
            append_tree_kmer(occ_buf2_[i]);
        }

        for (size_t i = 0; i < occ_buf1_size_; ++i)
        {
            const kmer<N>& x = occ_buf1_[i];
            if (occ_hash_set_->contains(x))
            {
                append_tree_kmer(x);
            }
            else
            {
                export_one_kmer(x);
            }
        }

        if (chunk_n > 0)
        {
            commit_tree_chunk(chunk_n);
        }

        occ_buf1_size_ = 0;
        occ_buf2_size_ = 0;
        occ_hash_set_->clear();
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

            uint32_t third_count = 0;

            const uint32_t bloom_filter_index = prefix_to_bloom_filter_index[prefix];

            ConcurrentBloomFilter<N>& bloom_filter = local_bloom_filters[bloom_filter_index];

            if (prefix_count < BLOOM_PREFETCH_DISTANCE / 2) [[unlikely]]
            {
                for (uint32_t i = 0; i < prefix_count; ++i)
                {
                    const kmer<N>& val = local_block_for_copy[read_offset];

                    const Occurrence occ = bloom_filter.insert(val);
                    if (occ == Occurrence::FIRST)
                    {
                        push_occurrence_first(val);
                    }
                    else if (occ == Occurrence::SECOND)
                    {
                        push_occurrence_second(val);
                    }
                    else
                    {
                        local_block_for_copy[local_block_count++] = val;
                        third_count++;
                    }

                    read_offset++;
                }

                local_block_prefix_counts[prefix] = third_count;

#ifdef TEST_MODE
                total_kmers_send_to_tree += third_count;
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
                    push_occurrence_first(val);
                }
                else if (occ == Occurrence::SECOND)
                {
                    push_occurrence_second(val);
                }
                else
                {
                    local_block_for_copy[local_block_count++] = val;
                    third_count++;
                }
            }

            read_offset += prefix_count;
            local_block_prefix_counts[prefix] = third_count;
#ifdef TEST_MODE
            total_kmers_send_to_tree += third_count;
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

            // Bloom 保持原样：共享 filter + 逐个 insert，无 prefetch；不走本线程 Buf 去重
            ConcurrentBloomFilter<N>& bloom_filter = *local_global_bloom_filter[prefix];

            for (uint32_t i = 0; i < prefix_count; i++)
            {
                const kmer<N>& val = local_block_for_copy[read_offset];

                if (bloom_filter.insert(val) == Occurrence::FIRST)
                {
                    export_one_kmer(val);
                    prefix_export_count++;
                }
                else
                {
                    // SECOND 与 THIRD_PLUS：立刻进树缓冲
                    local_block_for_copy[local_block_count++] = val;
                }

                read_offset++;
            }

            // uint32_t third_count = 0;
            // for (uint32_t i = 0; i < prefix_count; i++)
            // {
            //     const kmer<N>& val = local_block_for_copy[read_offset];

            //     const Occurrence occ = bloom_filter.insert(val);

            //     if (occ == Occurrence::FIRST)
            //     {
            //         push_occurrence_first(val);
            //     }
            //     else if (occ == Occurrence::SECOND)
            //     {
            //         push_occurrence_second(val);
            //     }
            //     else
            //     {
            //         local_block_for_copy[local_block_count++] = val;
            //         third_count++;
            //     }

            //     read_offset++;
            // }

            local_block_prefix_counts[prefix] -= prefix_export_count;
#ifdef TEST_MODE
            // export 已在 export_one_kmer 中计数；此处计非 FIRST 进树
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
