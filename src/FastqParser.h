#ifndef FASTQ_PARSER_HEADER
#define FASTQ_PARSER_HEADER

#include "definition.h"
#include "kmer.h"
#include "GetKmer.h"
#include "NewKmerTree.h"
#include "RingMemoryPool.h"
#include "ConcurrentMemoryPool.h"
#include "MPSCRingQueue.h"
#include "SplitMix.h"
#include "SpinBackoff.h"

#include <cstring>
#include <vector>
#include <array>
#include <memory>
#include <thread>
#include <random>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <nmmintrin.h>
#else
#endif

#ifdef TEST_MODE
#include <x86intrin.h>
#endif

template <uint32_t N>
class FastqParser
{
    // 自旋参数
    static constexpr int SLEEP_THRESHOLD = 128 + 32;
    static constexpr int YIELD_THRESHOLD = 128;
    static constexpr int MAX_BACKOFF = 64;

    int k_len;

    uint32_t owner_start = 0;
    uint64_t total_read_kmer = 0;
    uint32_t classifier_num;
    uint32_t kmer_buffer_count = 0;

    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> classifier_task_queues;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>* reader_parser_ring_pool;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_classifier_ring_pool;

    alignas(CACHE_LINE_SIZE) std::array<uint8_t, 1ULL << (2 * ROOT_BASES)> local_prefix_owners{};

    alignas(CACHE_LINE_SIZE) std::vector<uint32_t> local_block_owner_counts{};
    alignas(CACHE_LINE_SIZE) std::vector<uint32_t> local_block_owner_sums{};
    std::vector<content_type> owner_contents{};

    std::array<kmer<N>, PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>)> local_block_for_copy{};
    std::array<kmer<N>, PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>)> kmer_buffer{};

    GetKmer<N> get_kmer;

    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> enqueue_to_classifier_backoff;
    SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_from_classifier_backoff;

    SplitMix64 rng;

    alignas(CACHE_LINE_SIZE) static std::atomic<uint64_t> rng_seed;

public:
#ifdef TEST_MODE
    uint64_t producer_enqueue_spin_time{ 0 };
    uint64_t producer_dequeue_spin_time{ 0 };
    uint64_t consumer_enqueue_spin_time{ 0 };
    uint64_t consumer_dequeue_spin_time{ 0 };
    bool not_first_flag = false;
    uint64_t flush_cycles = 0;
    uint64_t parse_total_cycles = 0;
    uint64_t queue_wait_cycles = 0;
#endif

    explicit FastqParser(uint32_t in_k_len,
        std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>>& in_classifier_task_queues,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* in_global_classifier_task_queue,
        RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>* in_reader_parser_ring_pool,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_parser_classifier_ring_pool)
        : k_len(in_k_len), classifier_task_queues(in_classifier_task_queues),
        global_classifier_task_queue(in_global_classifier_task_queue), reader_parser_ring_pool(in_reader_parser_ring_pool), parser_classifier_ring_pool(in_parser_classifier_ring_pool), get_kmer(in_k_len)
    {
        local_prefix_owners = prefix_owners;
        kmer_buffer_count = 0;
        classifier_num = static_cast<uint32_t>(classifier_task_queues.size());
        local_block_owner_counts.resize(classifier_num, 0);
        local_block_owner_sums.resize(classifier_num, 0);
        owner_contents.resize(classifier_num);
        SplitMix64::seed(rng_seed.fetch_add(1));
    }

    void parse_and_push()
    {
        content_type reader_parser_content;
        // bool not_empty = true;

        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> enqueue_backoff;
        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_backoff;

        for (int i = 0; i < classifier_num; i++)
        {
            dequeue_data_from_classifier(owner_contents[i].data);
            owner_contents[i].length = 0;
        }


        while (true)
        {
            if (!reader_parser_ring_pool->producer_finished()) [[likely]]
            {
                if (reader_parser_ring_pool->consumer_try_dequeue(reader_parser_content))
                {

#ifdef TEST_MODE
                    not_first_flag = true;
#endif

                    dequeue_backoff.double_decay();

                    parse(reader_parser_content.data, reader_parser_content.length);

                    if (reader_parser_ring_pool->consumer_try_enqueue(reader_parser_content.data))
                    {
                        // enqueue 无等待
                        enqueue_backoff.double_decay();
                    }
                    else
                    {
                        // enqueue 等待
                        enqueue_backoff.backoff();

                        while (!reader_parser_ring_pool->consumer_try_enqueue(reader_parser_content.data))
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
                while (reader_parser_ring_pool->consumer_try_dequeue(reader_parser_content))
                {
                    parse(reader_parser_content.data, reader_parser_content.length);
                    reader_parser_ring_pool->consumer_enqueue(reader_parser_content.data);
                }
                break;
            }
        }

        // above is new

        // 当队列确实为空且生产者已结束时才退出循环
        /*while (not_empty || !reader_parser_ring_pool->producer_finished())
        {
            not_empty = reader_parser_ring_pool->consumer_try_dequeue(reader_parser_content);
            if (not_empty)
            {
                // 无等待

#ifdef TEST_MODE
                not_first_flag = true;
#endif

                dequeue_backoff.decay();

                parse(reader_parser_content.data, reader_parser_content.length);

                if (reader_parser_ring_pool->consumer_try_enqueue(reader_parser_content.data))
                {
                    // enqueue 无等待
                    enqueue_backoff.decay();
                }
                else
                {
                    // enqueue 等待
                    enqueue_backoff.backoff();

                    while (!reader_parser_ring_pool->consumer_try_enqueue(reader_parser_content.data))
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
                //等待

#ifdef TEST_MODE
                if (not_first_flag)
                    consumer_dequeue_spin_time++;
#endif
                dequeue_backoff.backoff();

            }
        }*/

        flush_kmer_buffer();

        for (uint32_t owner = 0; owner < owner_contents.size(); owner++)
        {
            if (owner_contents[owner].length == 0)
            {
                continue;
            }
            enqueue_content_to_classifier(owner);
        }
    }

    uint64_t get_total_read_kmer() const noexcept
    {
        return total_read_kmer;
    }

private:

    void parse(char const* data_ptr, const uint64_t length)
    {
        // 获取 k-mer，把 k-mer写入到parser_classifier_content的data里面，length是k-mer的数量，data最大是64KB
        // 解析出 k-mer 后，total_read_kmer += k-mer数量;
        // 解析出 k-mer 后，写入 parser_classifier_content.data，并设置 parser_classifier_content.length
#ifdef TEST_MODE
        uint64_t parse_start = __rdtsc();
#endif

        get_kmer.clear();

#if defined(__AVX2__)
        const __m256i MASK_UPPER = _mm256_set1_epi8(0xDF);
        const __m256i A = _mm256_set1_epi8('A');
        const __m256i C = _mm256_set1_epi8('C');
        const __m256i G = _mm256_set1_epi8('G');
        const __m256i T = _mm256_set1_epi8('T');
        const __m256i U = _mm256_set1_epi8('U');

#elif defined(__SSE4_2__)
        const __m128i MASK_UPPER = _mm_set1_epi8(0xDF);
        const __m128i A = _mm_set1_epi8('A');
        const __m128i C = _mm_set1_epi8('C');
        const __m128i G = _mm_set1_epi8('G');
        const __m128i T = _mm_set1_epi8('T');
        const __m128i U = _mm_set1_epi8('U');
#else
#endif

#if defined(__AVX2__)
        // 一次处理32字节
        const char* ptr = data_ptr;
        const char* end = data_ptr + length;
        alignas(32) uint8_t codes[32];

        while (static_cast<size_t>(end - ptr) >= 32)
        {
            size_t chunk = 32;
            __m256i input = _mm256_loadu_si256((__m256i*)ptr);

            __m256i upper = _mm256_and_si256(input, MASK_UPPER);
            __m256i uA = _mm256_cmpeq_epi8(upper, A);
            __m256i uC = _mm256_cmpeq_epi8(upper, C);
            __m256i uG = _mm256_cmpeq_epi8(upper, G);
            __m256i uTU = _mm256_or_si256(_mm256_cmpeq_epi8(upper, T), _mm256_cmpeq_epi8(upper, U));

            __m256i bit0_mask = _mm256_or_si256(uC, uTU);
            __m256i bit1_mask = _mm256_or_si256(uG, uTU);

            __m256i bit0 = _mm256_and_si256(bit0_mask, _mm256_set1_epi8(1));
            __m256i bit1 = _mm256_and_si256(bit1_mask, _mm256_set1_epi8(2));
            __m256i encode = _mm256_or_si256(bit1, bit0);

            _mm256_store_si256((__m256i*)codes, encode);

            __m256i base_mask = _mm256_or_si256(
                _mm256_or_si256(uA, uC),
                bit1_mask);

            uint32_t base_bits = _mm256_movemask_epi8(base_mask) & ((1ULL << chunk) - 1);

            uint32_t all_valid_bits = base_bits;
            uint32_t invalid_bits = (~all_valid_bits) & ((1ULL << chunk) - 1);

            // 合并所有位（有效位 + 无效位），遍历处理
            uint32_t bits = all_valid_bits | invalid_bits;
            while (bits)
            {
                int idx = __builtin_ctz(bits);
                uint32_t bit = 1u << idx;

                if (base_bits & bit)
                {
                    // 找到一段连续碱基运行；计算其长度
                    uint32_t run_bits = base_bits >> idx;
                    int run_len = 1;
                    // 限定最长 32 个碱基
                    while ((idx + run_len < 32) && (run_len < 32) && (run_bits & (1u << run_len)))
                    {
                        ++run_len;
                    }

                    // 将此运行中的所有碱基编码打包为一个 uint64_t
                    uint64_t packed = 0;
                    for (int i = 0; i < run_len; ++i)
                    {
                        packed = (packed << 2) | codes[idx + i];
                    }

                    // 批次插入前确保缓冲区有足够空间（一次最多写入 run_len 个 k-mer）
                    constexpr uint64_t KMER_BUF_CAP = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
                    if (kmer_buffer_count + static_cast<uint64_t>(run_len) > KMER_BUF_CAP) [[unlikely]]
                    {
#ifdef TEST_MODE
                        uint64_t flush_start = __rdtsc();
#endif
                        flush_kmer_buffer();
#ifdef TEST_MODE
                        uint64_t flush_end = __rdtsc();
                        flush_cycles += (flush_end - flush_start);
#endif
                    }

                    // 批次插入：一次更新 k-mer 状态，同时拿到新完成的 k-mer
                    uint32_t new_kmers = get_kmer.batch_insert(packed, run_len,
                        &kmer_buffer[kmer_buffer_count]);
                    kmer_buffer_count += new_kmers;

                    // 清除已处理的位
                    uint32_t clear_mask = static_cast<uint32_t>(((1ULL << run_len) - 1ULL) << idx);
                    bits &= ~clear_mask;
                }
                else
                {
                    // 遇到无效字符或换行符，清空 k-mer 状态
                    get_kmer.clear();
                    bits &= bits - 1;
                }
            }
            ptr += chunk;
        }

        constexpr uint64_t KMER_BUF_CAP = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
        while (ptr < end)
        {
            const char c = *ptr;
            ptr++;
            if (get_kmer.get_next_one(c))
            {
                if (kmer_buffer_count + 1 > KMER_BUF_CAP) [[unlikely]]
                {
#ifdef TEST_MODE
                    uint64_t flush_start = __rdtsc();
#endif
                    flush_kmer_buffer();
#ifdef TEST_MODE
                    uint64_t flush_end = __rdtsc();
                    flush_cycles += (flush_end - flush_start);
#endif
                }
                kmer_buffer[kmer_buffer_count++] = get_kmer.canonical_kmer;
            }
        }

#elif defined(__SSE4_2__)
        // 一次处理16字节
        const char* ptr = data_ptr;
        const char* end = data_ptr + length;
        alignas(16) uint8_t codes[16];

        while (static_cast<size_t>(end - ptr) >= 16)
        {
            size_t chunk = 16;
            __m128i input = _mm_loadu_si128((__m128i*)ptr);

            __m128i upper = _mm_and_si128(input, MASK_UPPER);
            __m128i uA = _mm_cmpeq_epi8(upper, A);
            __m128i uC = _mm_cmpeq_epi8(upper, C);
            __m128i uG = _mm_cmpeq_epi8(upper, G);
            __m128i uTU = _mm_or_si128(_mm_cmpeq_epi8(upper, T), _mm_cmpeq_epi8(upper, U));

            __m128i bit0_mask = _mm_or_si128(uC, uTU);
            __m128i bit1_mask = _mm_or_si128(uG, uTU);
            __m128i bit0 = _mm_and_si128(bit0_mask, _mm_set1_epi8(1));
            __m128i bit1 = _mm_and_si128(bit1_mask, _mm_set1_epi8(2));
            __m128i encode = _mm_or_si128(bit1, bit0);

            __m128i base_mask = _mm_or_si128(_mm_or_si128(uA, uC), bit1_mask);
            uint16_t base_bits = static_cast<uint16_t>(_mm_movemask_epi8(base_mask) & ((1u << chunk) - 1));

            _mm_store_si128((__m128i*)codes, encode);

            // 合并有效位与无效位
            uint16_t all_valid = base_bits;
            uint16_t invalid = static_cast<uint16_t>((~all_valid) & ((1u << chunk) - 1));
            uint16_t bits = all_valid | invalid;
            while (bits)
            {
                int idx = __builtin_ctz(bits);
                uint16_t bit = 1u << idx;

                if (base_bits & bit)
                {
                    uint16_t run_bits = base_bits >> idx;
                    int run_len = 1;
                    while ((idx + run_len < 16) && (run_len < 16) && (run_bits & (1u << run_len)))
                    {
                        ++run_len;
                    }

                    uint32_t packed = 0;
                    for (int i = 0; i < run_len; ++i)
                    {
                        packed = (packed << 2) | codes[idx + i];
                    }

                    constexpr uint64_t KMER_BUF_CAP = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
                    if (kmer_buffer_count + static_cast<uint64_t>(run_len) > KMER_BUF_CAP) [[unlikely]]
                    {
#ifdef TEST_MODE
                        uint64_t flush_start = __rdtsc();
#endif
                        flush_kmer_buffer();
#ifdef TEST_MODE
                        uint64_t flush_end = __rdtsc();
                        flush_cycles += (flush_end - flush_start);
#endif
                    }

                    uint32_t new_kmers = get_kmer.batch_insert(packed, run_len,
                        &kmer_buffer[kmer_buffer_count]);
                    kmer_buffer_count += new_kmers;

                    uint16_t clear_mask = static_cast<uint16_t>(((1U << run_len) - 1U) << idx);
                    bits &= ~clear_mask;
                }
                else
                {
                    get_kmer.clear();
                    bits &= bits - 1;
                }
            }
            ptr += chunk;
        }

        constexpr uint64_t KMER_BUF_CAP = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
        while (ptr < end)
        {
            const char c = *ptr;
            ptr++;
            if (get_kmer.get_next_one(c))
            {
                if (kmer_buffer_count + 1 > KMER_BUF_CAP) [[unlikely]]
                {
#ifdef TEST_MODE
                    uint64_t flush_start = __rdtsc();
#endif
                    flush_kmer_buffer();
#ifdef TEST_MODE
                    uint64_t flush_end = __rdtsc();
                    flush_cycles += (flush_end - flush_start);
#endif
                }
                kmer_buffer[kmer_buffer_count++] = get_kmer.canonical_kmer;
            }
        }
#else
        for (int i = 0; i < length; ++i)
        {
            const char c = data_ptr[i];
            if (get_kmer.get_next_one(c))
            {
                kmer_buffer[kmer_buffer_count++] = get_kmer.canonical_kmer;
                if (kmer_buffer_count >= (PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>))) [[unlikely]]
                {
#ifdef TEST_MODE
                    uint64_t flush_start = __rdtsc();
#endif
                    flush_kmer_buffer();
#ifdef TEST_MODE
                    uint64_t flush_end = __rdtsc();
                    flush_cycles += (flush_end - flush_start);
#endif
                }
            }
        }
#endif

#ifdef TEST_MODE
        uint64_t parse_end = __rdtsc();
        parse_total_cycles += (parse_end - parse_start);
#endif

    }

private:
    uint64_t get_classifier_owner(const kmer<N>& k_mer) const
    {
        constexpr uint32_t shift_bits = 64 - (ROOT_BASES * 2);
        return local_prefix_owners[k_mer.data[0] >> shift_bits];
    }

    void calculate_block_owner_counts(kmer<N>* kmer_data, const uint64_t kmer_count)
    {

        memset(local_block_owner_counts.data(), 0, local_block_owner_counts.size() * sizeof(uint32_t));

        // 可以考虑simd加速
        for (uint64_t i = 0; i < kmer_count; ++i)
        {
            const uint64_t owner = get_classifier_owner(kmer_data[i]);
            local_block_owner_counts[owner]++;
        }

        local_block_owner_sums[0] = 0;
        for (uint64_t i = 1; i < local_block_owner_counts.size(); ++i)
        {
            local_block_owner_sums[i] = local_block_owner_sums[i - 1] + local_block_owner_counts[i - 1];
        }
    }

    void push_kmers_into_local_block_for_copy(kmer<N>* kmer_data, const uint64_t kmer_count)
    {
        for (uint64_t index = 0; index < kmer_count; index++)
        {
            const uint64_t owner = get_classifier_owner(kmer_data[index]);
            const uint64_t pos = local_block_owner_sums[owner];
            local_block_for_copy[pos] = kmer_data[index];
            local_block_owner_sums[owner]++;
        }
    }

    void divide_kmers_into_owner_contents()
    {
        constexpr uint32_t max_kmers_per_block = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);

        owner_start = rng() % owner_contents.size();

        for (uint32_t i = 0; i < owner_contents.size(); i++)
        {
            uint32_t owner = (i + owner_start) % owner_contents.size();
            uint32_t remaining_count = local_block_owner_counts[owner];
            uint64_t read_offset = local_block_owner_sums[owner] - local_block_owner_counts[owner];

            if (owner_contents[owner].length + remaining_count > max_kmers_per_block)
            {
                const uint32_t copy_count = (max_kmers_per_block - owner_contents[owner].length);
                std::memcpy(owner_contents[owner].data + owner_contents[owner].length * sizeof(kmer<N>), local_block_for_copy.data() + read_offset, copy_count * sizeof(kmer<N>));
                owner_contents[owner].length += copy_count;

#ifdef TEST_MODE
                uint64_t queue_wait_start = __rdtsc();
#endif
                enqueue_content_to_classifier(owner);
#ifdef TEST_MODE
                uint64_t queue_wait_end = __rdtsc();
                queue_wait_cycles += queue_wait_end - queue_wait_start;
#endif
                read_offset += copy_count;

                owner_contents[owner].length = 0;

#ifdef TEST_MODE
                queue_wait_start = __rdtsc();
#endif
                dequeue_data_from_classifier(owner_contents[owner].data);
#ifdef TEST_MODE
                queue_wait_end = __rdtsc();
                queue_wait_cycles += queue_wait_end - queue_wait_start;
#endif

                remaining_count -= copy_count;
            }

            std::memcpy(owner_contents[owner].data + owner_contents[owner].length * sizeof(kmer<N>), local_block_for_copy.data() + read_offset, remaining_count * sizeof(kmer<N>));
            owner_contents[owner].length += remaining_count;
            read_offset += remaining_count;
        }

    }

    void flush_kmer_buffer()
    {
        calculate_block_owner_counts(kmer_buffer.data(), kmer_buffer_count);
        push_kmers_into_local_block_for_copy(kmer_buffer.data(), kmer_buffer_count);
        divide_kmers_into_owner_contents();
        total_read_kmer += kmer_buffer_count;
        kmer_buffer_count = 0;
    }

    void enqueue_content_to_classifier(const uint32_t owner_id)
    {

        if (classifier_task_queues[owner_id]->try_enqueue(owner_contents[owner_id]))
        {
            enqueue_to_classifier_backoff.double_decay();
            return;
        }

        if (global_classifier_task_queue->try_enqueue(owner_contents[owner_id]))
        {
            enqueue_to_classifier_backoff.double_decay();
            return;
        }

        enqueue_to_classifier_backoff.backoff();

        while (true)
        {

#ifdef TEST_MODE
            producer_enqueue_spin_time++;
#endif

            if (classifier_task_queues[owner_id]->try_enqueue(owner_contents[owner_id]))
            {
                break;
            }
            if (global_classifier_task_queue->try_enqueue(owner_contents[owner_id]))
            {
                break;
            }
            enqueue_to_classifier_backoff.backoff();

        }

        enqueue_to_classifier_backoff.decay();
    }

    void dequeue_data_from_classifier(char*& data)
    {

        if (parser_classifier_ring_pool->producer_try_dequeue(data))
        {
            dequeue_from_classifier_backoff.double_decay();
            return;
        }

        dequeue_from_classifier_backoff.backoff();

        while (!parser_classifier_ring_pool->producer_try_dequeue(data))
        {

#ifdef TEST_MODE
            producer_dequeue_spin_time++;
#endif

            dequeue_from_classifier_backoff.backoff();

        }

        dequeue_from_classifier_backoff.decay();
    }
};

template<uint32_t N>
alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> FastqParser<N>::rng_seed{ 0 };

#endif