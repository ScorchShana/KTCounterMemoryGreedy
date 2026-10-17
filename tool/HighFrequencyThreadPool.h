#pragma once

#include "HistogramCommon.h"
#include "FlatConcurrentHashMap.h"
#include "../src/RingMemoryPool.h"
#include "../src/SpinBackoff.h"
#include <thread>

// Workers insert compact records directly from ring blocks. Only the count
// is decoded here; the table restores bytes locally for the legacy hash.
template <uint32_t N, uint64_t RING_CAPACITY>
class HighFrequencyThreadPool
{
    static constexpr int SLEEP_THRESHOLD = 64 + 16;
    static constexpr int YIELD_THRESHOLD = 64;
    static constexpr int MAX_BACKOFF = 256;

public:
    HighFrequencyThreadPool(
        SPMCRingMemoryPool<RING_CAPACITY>* pool,
        FlatConcurrentHashMap<N>* hash_map,
        std::vector<std::atomic<int64_t>>* global_histogram,
        const uint32_t worker_count,
        const uint32_t k_len,
        const uint32_t count_bytes,
        const uint32_t min_freq,
        const uint32_t max_freq,
        const size_t hist_size)
        : pool_(pool),
        hash_map_(hash_map),
        global_histogram_(global_histogram),
        worker_count_(worker_count),
        layout_(k_len, count_bytes),
        min_freq_(min_freq),
        max_freq_(max_freq),
        hist_size_(hist_size)
    {
    }

    ~HighFrequencyThreadPool()
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
            workers_.emplace_back(&HighFrequencyThreadPool::worker_loop, this, i);
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
    void worker_loop(uint32_t index)
    {
        auto& local_histogram = local_histograms_[index];
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

        uint32_t count = 0;

        const char* record = content.data;
        const char* const record_end = record + content.length * layout_.record_bytes;
        for (; record < record_end; record += layout_.record_bytes)
        {
            count = unpack_count(record, layout_);

            if (in_range(count, min_freq_, max_freq_)) [[likely]]
            {
                local_histogram[histogram_index(count, min_freq_)] += 1;
            }

            if (!hash_map_->insert_unique(record, count)) [[unlikely]]
            {
                insert_failed_.store(true, std::memory_order_release);
            }
        }
    }

    SPMCRingMemoryPool<RING_CAPACITY>* pool_ = nullptr;
    FlatConcurrentHashMap<N>* hash_map_ = nullptr;
    std::vector<std::atomic<int64_t>>* global_histogram_ = nullptr;
    uint32_t worker_count_ = 0;
    RecordLayout layout_;
    uint32_t min_freq_ = 0;
    uint32_t max_freq_ = 0;
    size_t hist_size_ = 0;
    std::vector<std::thread> workers_;
    std::vector<std::vector<int64_t>> local_histograms_;
    std::atomic<bool> insert_failed_{ false };
};

// Approximate phase: same compact blocks, but only the count field is read.
template <uint32_t N, uint64_t RING_CAPACITY>
class ApproximateHighFrequencyThreadPool
{
    static constexpr int MAX_SPIN_TIME = 128;
    static constexpr int MAX_BACKOFF = 256;

public:
    ApproximateHighFrequencyThreadPool(
        SPMCRingMemoryPool<RING_CAPACITY>* pool,
        std::vector<std::atomic<int64_t>>* global_histogram,
        const uint32_t worker_count,
        const uint32_t k_len,
        const uint32_t count_bytes,
        const uint32_t min_freq,
        const uint32_t max_freq,
        const size_t hist_size)
        : pool_(pool),
        global_histogram_(global_histogram),
        worker_count_(worker_count),
        layout_(k_len, count_bytes),
        min_freq_(min_freq),
        max_freq_(max_freq),
        hist_size_(hist_size)
    {
    }

    ~ApproximateHighFrequencyThreadPool()
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
            workers_.emplace_back(&ApproximateHighFrequencyThreadPool::worker_loop, this, i);
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
        content_type content{};

        int spin_time = 0;
        int backoff = 2;

        while (true)
        {
            if (pool_->consumer_try_dequeue(content))
            {
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
                spin_time++;
                if (spin_time > MAX_SPIN_TIME)
                {
                    spin_time = 0;
                    backoff = 2;
                    std::this_thread::yield();
                }
                else
                {
                    for (int i = 0; i < backoff; i++)
                    {
                        cpu_relax();
                    }
                    backoff = std::min(backoff * 2, MAX_BACKOFF);
                }
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

        const char* record = content.data;
        const char* const record_end = record + content.length * layout_.record_bytes;
        for (; record < record_end; record += layout_.record_bytes)
        {
            const uint32_t count = unpack_count(record, layout_);
            if (in_range(count, min_freq_, max_freq_)) [[likely]]
            {
                local_histogram[histogram_index(count, min_freq_)] += 1;
            }
        }
    }

    SPMCRingMemoryPool<RING_CAPACITY>* pool_ = nullptr;
    std::vector<std::atomic<int64_t>>* global_histogram_ = nullptr;
    uint32_t worker_count_ = 0;
    RecordLayout layout_;
    uint32_t min_freq_ = 0;
    uint32_t max_freq_ = 0;
    size_t hist_size_ = 0;
    std::vector<std::thread> workers_;
    std::vector<std::vector<int64_t>> local_histograms_;
};

