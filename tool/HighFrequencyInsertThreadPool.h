#ifndef HIGH_FREQUENCY_INSERT_THREAD_POOL_HEADER
#define HIGH_FREQUENCY_INSERT_THREAD_POOL_HEADER

#include "FlatConcurrentHashMap.h"
#include "../src/SpinBackoff.h"
#include "../src/RingMemoryPool.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

template <uint32_t N, uint64_t RING_CAPACITY>
class HighFrequencyInsertThreadPool
{
    using Record = ExportRecord<N>;

    static constexpr int SLEEP_THRESHOLD = 128;
    static constexpr int YIELD_THRESHOLD = 64;
    static constexpr int MAX_BACKOFF = 256;

public:
    HighFrequencyInsertThreadPool(
        SPMCRingMemoryPool<RING_CAPACITY>* pool,
        FlatConcurrentHashMap<N>* hash_map,
        std::vector<std::atomic<int64_t>>* global_histogram,
        const uint32_t worker_count,
        const uint32_t k_len,
        const uint32_t min_freq,
        const uint32_t max_freq,
        const size_t hist_size)
        : pool_(pool),
        hash_map_(hash_map),
        global_histogram_(global_histogram),
        worker_count_(worker_count),
        k_len_(k_len),
        min_freq_(min_freq),
        max_freq_(max_freq),
        hist_size_(hist_size)
    {
    }

    void start()
    {
        workers_.reserve(worker_count_);
        for (uint32_t i = 0; i < worker_count_; ++i)
        {
            workers_.emplace_back(&HighFrequencyInsertThreadPool::worker_loop, this);
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

    [[nodiscard]] bool insert_failed() const noexcept
    {
        return insert_failed_.load(std::memory_order_acquire);
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
        if (content.length == 0)
        {
            return;
        }

        auto* records = reinterpret_cast<Record*>(content.data);
        for (uint64_t i = 0; i < content.length; ++i)
        {
            Record record = records[i];
            const uint64_t merged_count = static_cast<uint64_t>(record.count) + 1ULL;
            if (merged_count < min_freq_)
            {
                continue;
            }

            if (in_range(record.count, min_freq_, max_freq_)) [[likely]]
            {
                local_histogram[histogram_index(record.count, min_freq_)] += 1;
            }

            if (!hash_map_->insert_unique(record.key, record.count)) [[unlikely]]
            {
                insert_failed_.store(true, std::memory_order_release);
            }
        }
    }

    SPMCRingMemoryPool<RING_CAPACITY>* pool_ = nullptr;
    FlatConcurrentHashMap<N>* hash_map_ = nullptr;
    std::vector<std::atomic<int64_t>>* global_histogram_ = nullptr;
    uint32_t worker_count_ = 0;
    uint32_t k_len_ = 0;
    uint32_t min_freq_ = 0;
    uint32_t max_freq_ = 0;
    size_t hist_size_ = 0;
    std::vector<std::thread> workers_;
    std::atomic<bool> insert_failed_{ false };
};

#endif
