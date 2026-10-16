#ifndef LOW_FREQUENCY_THREAD_POOL_HEADER
#define LOW_FREQUENCY_THREAD_POOL_HEADER

#include "FlatConcurrentHashMap.h"
#include "HistogramCommon.h"
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
class LowFrequencyThreadPool
{

    static constexpr int SLEEP_THRESHOLD = 128;
    static constexpr int YIELD_THRESHOLD = 64;
    static constexpr int MAX_BACKOFF = 256;
    static constexpr uint64_t QUERY_PREFETCH_DISTANCE = 8;

public:
    LowFrequencyThreadPool(
        SPMCRingMemoryPool<RING_CAPACITY>* pool,
        const FlatConcurrentHashMap<N>* hash_map,
        std::vector<std::atomic<int64_t>>* global_histogram,
        const uint32_t worker_count,
        const uint32_t k_len,
        const uint32_t min_freq,
        const uint32_t max_freq,
        const size_t hist_size,
        const uint32_t count_max = std::numeric_limits<uint32_t>::max(),
        std::atomic<uint64_t>* matched = nullptr)
        : pool_(pool),
        hash_map_(hash_map),
        global_histogram_(global_histogram),
        worker_count_(worker_count),
        layout_(k_len, 0),
        matched_(matched),
        min_freq_(min_freq),
        max_freq_(max_freq),
        hist_size_(hist_size),
        count_max_(count_max)
    {
    }

    ~LowFrequencyThreadPool()
    {
        if (!pool_->producer_finished()) pool_->producer_set_finished();
        join();
    }

    void start()
    {
        local_histograms_.reserve(worker_count_);
        workers_.reserve(worker_count_);
        for (uint32_t i = 0; i < worker_count_; ++i)
            local_histograms_.emplace_back(hist_size_, 0);
        for (uint32_t i = 0; i < worker_count_; ++i)
            workers_.emplace_back(&LowFrequencyThreadPool::worker_loop, this, i);
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
    void worker_loop(uint32_t index)
    {
        auto& local_histogram = local_histograms_[index];
        uint64_t local_matched = 0;
        content_type content{};

        SpinBackoff<MAX_BACKOFF, YIELD_THRESHOLD, SLEEP_THRESHOLD> dequeue_backoff;

        while (true)
        {
            if (pool_->consumer_try_dequeue(content))
            {
                dequeue_backoff.decay();
                process_block(content, local_histogram, local_matched);
                pool_->consumer_enqueue(content.data);
            }
            else if (pool_->producer_finished()) [[unlikely]]
            {
                while (pool_->consumer_try_dequeue(content))
                {
                    process_block(content, local_histogram, local_matched);
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
        if (matched_) matched_->fetch_add(local_matched, std::memory_order_relaxed);
    }

    void process_block(const content_type& content, std::vector<int64_t>& local_histogram, uint64_t& local_matched)
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
            unpack_key(packed_kmers + i * layout_.key_bytes, layout_, unpacked_kmers[static_cast<size_t>(i)]);
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
                correct_histogram(local_histogram, count, count_max_, min_freq_, max_freq_);
                ++local_matched;
            }
            else if (!matched_)
            {
                add_frequency(local_histogram, 1, min_freq_, max_freq_, 1);
            }

            const uint64_t next_index = i + QUERY_PREFETCH_DISTANCE;
            if (next_index < content.length)
            {
                unpack_key(
                    packed_kmers + next_index * layout_.key_bytes, layout_,
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
    RecordLayout layout_;
    std::atomic<uint64_t>* matched_ = nullptr;
    uint32_t min_freq_ = 0;
    uint32_t max_freq_ = 0;
    size_t hist_size_ = 0;
    uint32_t count_max_ = std::numeric_limits<uint32_t>::max();
    std::vector<std::thread> workers_;
    std::vector<std::vector<int64_t>> local_histograms_;
};

#endif
