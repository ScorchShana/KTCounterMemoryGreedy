#ifndef APPROXIMATE_HIGH_FREQUENCY_HEADER
#define APPROXIMATE_HIGH_FREQUENCY_HEADER

#include "../src/RingMemoryPool.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>


template<uint32_t N, uint64_t RING_CAPACITY>
class ApproximateHighFrequencyThreadPool {

    using Record = ExportRecord<N>;

    static constexpr int MAX_SPIN_TIME = 128;
    static constexpr int MAX_BACKOFF = 256;

public:
    ApproximateHighFrequencyThreadPool(
        SPMCRingMemoryPool<RING_CAPACITY>* pool,
        std::vector<std::atomic<int64_t>>* global_histogram,
        const uint32_t worker_count,
        const uint32_t k_len,
        const uint32_t min_freq,
        const uint32_t max_freq,
        const size_t hist_size)
        : pool_(pool),
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
            workers_.emplace_back(&ApproximateHighFrequencyThreadPool::worker_loop, this);
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

    void worker_loop()
    {
        std::vector<int64_t> local_histogram(hist_size_, 0);
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
                else {
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
        if (content.length == 0)
        {
            return;
        }

        auto* records = reinterpret_cast<Record*>(content.data);
        for (uint64_t i = 0; i < content.length; ++i)
        {
            Record record = records[i];
            if (in_range(record.count, min_freq_, max_freq_)) [[likely]]
            {
                local_histogram[histogram_index(record.count, min_freq_)] += 1;
            }
        }
    }

    SPMCRingMemoryPool<RING_CAPACITY>* pool_ = nullptr;
    std::vector<std::atomic<int64_t>>* global_histogram_ = nullptr;
    uint32_t worker_count_ = 0;
    uint32_t k_len_ = 0;
    uint32_t min_freq_ = 0;
    uint32_t max_freq_ = 0;
    size_t hist_size_ = 0;
    std::vector<std::thread> workers_;
};



#endif