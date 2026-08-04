#ifndef LOW_FREQUENCY_QUERY_THREAD_POOL_HEADER
#define LOW_FREQUENCY_QUERY_THREAD_POOL_HEADER

#include "FlatConcurrentHashMap.h"
#include "../src/SpinBackoff.h"
#include "../src/RingMemoryPool.h"

#include <algorithm>
#include <atomic>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>


template <uint32_t N, uint64_t RING_CAPACITY>
class LowFrequencyQueryThreadPool
{

    static constexpr int SLEEP_THRESHOLD = 128;
    static constexpr int YIELD_THRESHOLD = 64;
    static constexpr int MAX_BACKOFF = 256;
    static constexpr uint64_t QUERY_PREFETCH_DISTANCE = 8;

public:
    LowFrequencyQueryThreadPool(
        SPMCRingMemoryPool<RING_CAPACITY>* pool,
        const FlatConcurrentHashMap<N>* hash_map,
        std::vector<std::atomic<int64_t>>* global_histogram,
        const uint32_t worker_count,
        const uint32_t k_len,
        const uint32_t min_freq,
        const uint32_t max_freq,
        const size_t hist_size,
        const uint32_t count_max = std::numeric_limits<uint32_t>::max())
        : pool_(pool),
        hash_map_(hash_map),
        global_histogram_(global_histogram),
        worker_count_(worker_count),
        k_len_(k_len),
        min_freq_(min_freq),
        max_freq_(max_freq),
        hist_size_(hist_size),
        count_max_(count_max),
        full_data_count_(k_len / BASES_PER_U64T),
        tail_bits_(2ULL * (k_len % BASES_PER_U64T)),
        tail_bytes_((tail_bits_ + 7ULL) / 8ULL),
        packed_kmer_bytes_(full_data_count_ * sizeof(uint64_t) + tail_bytes_)
    {
    }

    void start()
    {
        workers_.reserve(worker_count_);
        for (uint32_t i = 0; i < worker_count_; ++i)
        {
            workers_.emplace_back(&LowFrequencyQueryThreadPool::worker_loop, this);
        }
    }

    void join()
    {
        for (auto& worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

private:
    static bool in_range(const uint64_t freq, const uint32_t min_freq, const uint32_t max_freq) noexcept
    {
        return freq >= min_freq && freq <= max_freq;
    }

    static size_t histogram_index(const uint64_t freq, const uint32_t min_freq) noexcept
    {
        return static_cast<size_t>(freq - min_freq);
    }

    static void merge_histogram(
        const std::vector<int64_t>& local_histogram,
        std::vector<std::atomic<int64_t>>& global_histogram)
    {
        for (size_t i = 0; i < local_histogram.size(); ++i)
        {
            const int64_t value = local_histogram[i];
            if (value != 0)
            {
                global_histogram[i].fetch_add(value, std::memory_order_relaxed);
            }
        }
    }

    static void normalize_kmer(kmer<N>& key, const uint32_t k_len)
    {
        const uint32_t full_words = k_len / BASES_PER_U64T;
        const uint32_t tail_bases = k_len % BASES_PER_U64T;

        if (tail_bases == 0)
        {
            for (uint32_t i = full_words; i < N; ++i)
            {
                key.data[i] = 0;
            }
            return;
        }

        const uint32_t tail_bits = tail_bases * 2;
        const uint64_t mask = (~uint64_t{ 0 }) << (64 - tail_bits);
        key.data[full_words] &= mask;
        for (uint32_t i = full_words + 1; i < N; ++i)
        {
            key.data[i] = 0;
        }
    }

    void unpack_kmer(const char* record, kmer<N>& output) const noexcept
    {
        const uint64_t full_bytes = full_data_count_ * sizeof(uint64_t);
        if (full_bytes > 0)
        {
            std::memcpy(output.data.data(), record, static_cast<size_t>(full_bytes));
        }

        if (tail_bytes_ > 0)
        {
            uint64_t tail_data = 0;
            std::memcpy(
                reinterpret_cast<char*>(&tail_data) + (sizeof(uint64_t) - tail_bytes_),
                record + full_bytes,
                static_cast<size_t>(tail_bytes_));

            output.data[full_data_count_] = tail_data;
            for (uint32_t j = full_data_count_ + 1; j < N; j++)
            {
                output.data[j] = 0;
            }
        }
        else
        {
            for (uint32_t j = full_data_count_; j < N; j++)
            {
                output.data[j] = 0;
            }
        }
    }

    void worker_loop()
    {
        std::vector<int64_t> local_histogram(hist_size_, 0);
        content_type content{};

        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_backoff;

        while (true)
        {
            if (pool_->consumer_try_dequeue(content))
            {
                dequeue_backoff.decay();
                process_block(content, local_histogram);
                pool_->consumer_enqueue(content.data);
            }
            else if (pool_->producer_finished()) [[unlikely]]
            {
                while (pool_->consumer_try_dequeue(content))
                {
                    process_block(content, local_histogram);
                    pool_->consumer_enqueue(content.data);
                }
                break;
            }
            else
            {
                dequeue_backoff.backoff();
            }
        }

        merge_histogram(local_histogram, *global_histogram_);
    }

    void process_block(const content_type& content, std::vector<int64_t>& local_histogram)
    {
        if (content.length == 0) [[unlikely]]
        {
            return;
        }

        const char* packed_kmers = content.data;
        using Lookup = typename FlatConcurrentHashMap<N>::PreparedLookup;

        std::array<Lookup, QUERY_PREFETCH_DISTANCE> lookups{};
        std::array<kmer<N>, QUERY_PREFETCH_DISTANCE> unpacked_kmers{};
        const uint64_t warmup_count = std::min<uint64_t>(content.length, QUERY_PREFETCH_DISTANCE);
        for (uint64_t i = 0; i < warmup_count; ++i)
        {
            unpack_kmer(packed_kmers + i * packed_kmer_bytes_, unpacked_kmers[static_cast<size_t>(i)]);
            lookups[static_cast<size_t>(i)] = hash_map_->prepare_lookup(unpacked_kmers[static_cast<size_t>(i)]);
            hash_map_->prefetch(lookups[static_cast<size_t>(i)]);
        }

        for (uint64_t i = 0; i < content.length; ++i)
        {
            const uint64_t lookup_slot = i % QUERY_PREFETCH_DISTANCE;
            const Lookup lookup = lookups[static_cast<size_t>(lookup_slot)];
            const kmer<N>& key = unpacked_kmers[static_cast<size_t>(lookup_slot)];

            uint32_t count = 0;
            if (hash_map_->find_prepared(key, lookup, count)) [[unlikely]]
            {
                if (in_range(count, min_freq_, max_freq_)) [[likely]]
                {
                    local_histogram[histogram_index(count, min_freq_)] -= 1;
                }

                uint64_t merged_count = static_cast<uint64_t>(count) + 1ULL;
                // Clamp at count_max (max storable count for the file format)
                if (merged_count > count_max_)
                    merged_count = count_max_;
                if (in_range(merged_count, min_freq_, max_freq_)) [[likely]]
                {
                    local_histogram[histogram_index(merged_count, min_freq_)] += 1;
                }
            }
            else if (in_range(1, min_freq_, max_freq_))
            {
                local_histogram[histogram_index(1, min_freq_)] += 1;
            }

            const uint64_t next_index = i + QUERY_PREFETCH_DISTANCE;
            if (next_index < content.length)
            {
                unpack_kmer(
                    packed_kmers + next_index * packed_kmer_bytes_,
                    unpacked_kmers[static_cast<size_t>(lookup_slot)]);
                lookups[static_cast<size_t>(lookup_slot)] =
                    hash_map_->prepare_lookup(unpacked_kmers[static_cast<size_t>(lookup_slot)]);
                hash_map_->prefetch(lookups[static_cast<size_t>(lookup_slot)]);
            }
        }
    }

    SPMCRingMemoryPool<RING_CAPACITY>* pool_ = nullptr;
    const FlatConcurrentHashMap<N>* hash_map_ = nullptr;
    std::vector<std::atomic<int64_t>>* global_histogram_ = nullptr;
    uint32_t worker_count_ = 0;
    uint32_t k_len_ = 0;
    uint32_t min_freq_ = 0;
    uint32_t max_freq_ = 0;
    size_t hist_size_ = 0;
    uint32_t count_max_ = std::numeric_limits<uint32_t>::max();
    uint64_t full_data_count_ = 0;
    uint64_t tail_bits_ = 0;
    uint64_t tail_bytes_ = 0;
    uint64_t packed_kmer_bytes_ = 0;
    std::vector<std::thread> workers_;
};

#endif
