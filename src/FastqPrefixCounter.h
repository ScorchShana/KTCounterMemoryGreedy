#ifndef FASTQ_PREFIX_COUNTER_HEADER
#define FASTQ_PREFIX_COUNTER_HEADER

#include "definition.h"
#include "kmer.h"
#include "RingMemoryPool.h"
#include "SpinBackoff.h"

#include <cstdint>
#include <array>
#include <vector>

template <uint32_t N>
class FastqPrefixCounter
{

    int k_len;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_counter_ring_pool;

public:
    std::vector<uint32_t> prefix_counts = std::vector<uint32_t>(1ULL << (2 * ROOT_BASES)); // 256 个前缀的计数器

    FastqPrefixCounter(int in_k_len, RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_ring_pool)
        : k_len(in_k_len), parser_counter_ring_pool(in_ring_pool)
    {
    }

    void count_prefixes()
    {
        content_type content;

        SpinBackoff<> backoff;

        while(true)
        {
            if (!parser_counter_ring_pool->producer_finished()) [[likely]]
            {
                if (parser_counter_ring_pool->consumer_try_dequeue(content))
                {
                    backoff.decay();
                    kmer<N>* kmer_data = reinterpret_cast<kmer<N> *>(content.data);
                    uint64_t kmer_count = content.length;
                    process_block(kmer_data, kmer_count);
                    parser_counter_ring_pool->consumer_enqueue(content.data); // Return block to pool
                }
                else
                {
                    backoff.backoff();
                }
            }
            else
            {
                while(parser_counter_ring_pool->consumer_try_dequeue(content))
                {
                    kmer<N>* kmer_data = reinterpret_cast<kmer<N> *>(content.data);
                    uint64_t kmer_count = content.length;
                    process_block(kmer_data, kmer_count);
                    parser_counter_ring_pool->consumer_enqueue(content.data); // Return block to pool
                }
                break;
            }
        }

        // above is new

        /*
        while (true)
        {
            if (!parser_counter_ring_pool->consumer_try_dequeue(content))
            {
                if (parser_counter_ring_pool->producer_finished())
                {
                    break; // No more data will come, exit loop
                }
                else
                {
                    if (spin_count < YIELD_THRESHOLD)
                    {
                        for (int i = 0; i < backoff_iterations; ++i)
                        {
                            cpu_relax();
                        }
                        backoff_iterations = std::min(backoff_iterations * 2, MAX_BACKOFF);
                        spin_count++;
                    }
                    else
                    {
                        backoff_iterations = 1;
                        spin_count = 0;
                        std::this_thread::yield();
                    }
                    continue;
                }
            }

            char* data_ptr = content.data;
            uint64_t kmer_count = content.length;

            kmer<N>* kmer_array = reinterpret_cast<kmer<N> *>(data_ptr);
            for (uint64_t i = 0; i < kmer_count; ++i)
            {
                const kmer<N>& k_mer = kmer_array[i];
                uint64_t prefix = get_root_prefix(k_mer);
                prefix_counts[prefix]++;
            }

            parser_counter_ring_pool->consumer_enqueue(data_ptr); // Return block to pool
        }*/
    }

private:
    uint64_t get_root_prefix(const kmer<N>& k_mer) const
    {
        constexpr uint32_t shift_bits = 2 * (BASES_PER_U64T - ROOT_BASES);
        return k_mer.data[0] >> shift_bits;
    }

    uint64_t get_prefix(const kmer<N>& k_mer) const
    {
        constexpr uint32_t shift_bits = 2 * (BASES_PER_U64T - 10);
        return k_mer.data[0] >> shift_bits;
    }

    void process_block(kmer<N>* kmer_data, const uint64_t kmer_count)
    {
        for (uint64_t i = 0; i < kmer_count; ++i)
        {
            const kmer<N>& k_mer = kmer_data[i];
            uint64_t prefix = get_root_prefix(k_mer);
            prefix_counts[prefix]++;
        }
    }
};

#endif // FASTQ_PREFIX_COUNTER_HEADER