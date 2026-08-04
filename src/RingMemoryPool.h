#ifndef RING_MEMORY_POOL_HEADER
#define RING_MEMORY_POOL_HEADER

#include "definition.h"
#include "MPMCRingQueue.h"
#include "SPMCRingQueue.h"
#include "MPSCRingQueue.h"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <cstdlib>
#include <iostream>

template <uint64_t CAPACITY = 4096>
class RingMemoryPool
{
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "RingMemoryPool capacity must be a power of 2");

public:
    explicit RingMemoryPool(uint64_t block_size_, uint32_t producer_num)
        : block_size_(block_size_),
          memory_pool_(check_pool_size(CAPACITY, block_size_)),
          producer_to_consumer_(),
          consumer_to_producer_(),
          active_producers_(producer_num)
    {
        if (CAPACITY == 0)
        {
            std::cerr << "RingMemoryPool capacity must be > 0" << std::endl;
            std::exit(-1);
        }
        if (block_size_ == 0)
        {
            std::cerr << "RingMemoryPool block_size_ must be > 0" << std::endl;
            std::exit(-1);
        }

        char *base = memory_pool_.data();
        for (uint64_t i = 0; i < CAPACITY; ++i)
        {
            char *block_ptr = base + static_cast<std::ptrdiff_t>(i * block_size_);
            // Preload all blocks into returned queue so producers can reuse them.
            consumer_to_producer_.enqueue(block_ptr);
        }
    }

    RingMemoryPool(const RingMemoryPool &) = delete;
    RingMemoryPool &operator=(const RingMemoryPool &) = delete;

    inline void producer_set_finished()
    {
        active_producers_.fetch_sub(1, std::memory_order_release);
    }

    inline bool producer_finished() const noexcept
    {
        return active_producers_.load(std::memory_order_acquire) <= 0;
    }

    void reset_producers(uint32_t producer_num)
    {
        active_producers_.store(producer_num, std::memory_order_release);
    }

    // Producer pushes processed data for consumer.
    void producer_enqueue(const content_type &content)
    {
        producer_to_consumer_.enqueue(content);
    }

    bool producer_try_enqueue(const content_type &content)
    {
        return producer_to_consumer_.try_enqueue(content);
    }

    // Producer gets a reusable block returned by consumer.
    bool producer_try_dequeue(char *&block_ptr)
    {
        return consumer_to_producer_.try_dequeue(block_ptr);
    }

    void producer_dequeue(char *&block_ptr)
    {
        consumer_to_producer_.dequeue(block_ptr);
    }

    // Consumer pops processed data from producer.
    bool consumer_try_dequeue(content_type &content)
    {
        return producer_to_consumer_.try_dequeue(content);
    }

    void consumer_dequeue(content_type &content)
    {
        return producer_to_consumer_.dequeue(content);
    }

    // Consumer returns used block for producer reuse.
    bool consumer_try_enqueue(char *block_ptr)
    {
        return consumer_to_producer_.try_enqueue(block_ptr);
    }

    void consumer_enqueue(char *block_ptr)
    {
        consumer_to_producer_.enqueue(block_ptr);
    }

    uint64_t capacity() const noexcept { return CAPACITY; }
    uint64_t blockSize() const noexcept { return block_size_; }

private:
    static size_t check_pool_size(uint64_t capacity, uint64_t block_size)
    {
        if (capacity == 0 || block_size == 0)
        {
            return 0;
        }
        if (capacity > (std::numeric_limits<uint64_t>::max() / block_size))
        {
            std::cerr << "RingMemoryPool capacity * block_size_ overflow" << std::endl;
            std::exit(-1);
        }
        const uint64_t bytes = capacity * block_size;
        if (bytes > std::numeric_limits<size_t>::max())
        {
            std::cerr << "RingMemoryPool pool size exceeds size_t" << std::endl;
            std::exit(-1);
        }
        return bytes;
    }

    uint64_t block_size_ = 0;
    std::vector<char> memory_pool_;
    alignas(CACHE_LINE_SIZE) MPMCRingQueue<content_type, CAPACITY> producer_to_consumer_;
    alignas(CACHE_LINE_SIZE) MPMCRingQueue<char *, CAPACITY> consumer_to_producer_;
    alignas(CACHE_LINE_SIZE) std::atomic<int> active_producers_{0};
};

template <uint64_t CAPACITY = 4096>
class SPMCRingMemoryPool
{
    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "SPMCRingMemoryPool capacity must be a power of 2");

public:
    explicit SPMCRingMemoryPool(uint64_t block_size_, uint32_t producer_num)
        : block_size_(block_size_),
          memory_pool_(check_pool_size(CAPACITY, block_size_)),
          producer_to_consumer_(),
          consumer_to_producer_(),
          active_producers_(producer_num)
    {
        if (CAPACITY == 0)
        {
            std::cerr << "SPMCRingMemoryPool capacity must be > 0" << std::endl;
            std::exit(-1);
        }
        if (block_size_ == 0)
        {
            std::cerr << "SPMCRingMemoryPool block_size_ must be > 0" << std::endl;
            std::exit(-1);
        }

        char *base = memory_pool_.data();
        for (uint64_t i = 0; i < CAPACITY; ++i)
        {
            char *block_ptr = base + static_cast<std::ptrdiff_t>(i * block_size_);
            // Preload all blocks into returned queue so producers can reuse them.
            consumer_to_producer_.enqueue(block_ptr);
        }
    }

    SPMCRingMemoryPool(const SPMCRingMemoryPool &) = delete;
    SPMCRingMemoryPool &operator=(const SPMCRingMemoryPool &) = delete;

    inline void producer_set_finished()
    {
        active_producers_.fetch_sub(1, std::memory_order_release);
    }

    inline bool producer_finished() const noexcept
    {
        return active_producers_.load(std::memory_order_acquire) <= 0;
    }

    void reset_producers(uint32_t producer_num)
    {
        active_producers_.store(producer_num, std::memory_order_release);
    }

    // Producer pushes processed data for consumer.
    void producer_enqueue(const content_type &content)
    {
        producer_to_consumer_.enqueue(content);
    }

    bool producer_try_enqueue(const content_type &content)
    {
        return producer_to_consumer_.try_enqueue(content);
    }

    // Producer gets a reusable block returned by consumer.
    bool producer_try_dequeue(char *&block_ptr)
    {
        return consumer_to_producer_.try_dequeue(block_ptr);
    }

    void producer_dequeue(char *&block_ptr)
    {
        consumer_to_producer_.dequeue(block_ptr);
    }

    // Consumer pops processed data from producer.
    bool consumer_try_dequeue(content_type &content)
    {
        return producer_to_consumer_.try_dequeue(content);
    }

    void consumer_dequeue(content_type &content)
    {
        return producer_to_consumer_.dequeue(content);
    }

    // Consumer returns used block for producer reuse.
    bool consumer_try_enqueue(char *block_ptr)
    {
        return consumer_to_producer_.try_enqueue(block_ptr);
    }

    void consumer_enqueue(char *block_ptr)
    {
        consumer_to_producer_.enqueue(block_ptr);
    }

    uint64_t capacity() const noexcept { return CAPACITY; }
    uint64_t blockSize() const noexcept { return block_size_; }

private:
    static size_t check_pool_size(uint64_t capacity, uint64_t block_size)
    {
        if (capacity == 0 || block_size == 0)
        {
            return 0;
        }
        if (capacity > (std::numeric_limits<uint64_t>::max() / block_size))
        {
            std::cerr << "RingMemoryPool capacity * block_size_ overflow" << std::endl;
            std::exit(-1);
        }
        const uint64_t bytes = capacity * block_size;
        if (bytes > std::numeric_limits<size_t>::max())
        {
            std::cerr << "RingMemoryPool pool size exceeds size_t" << std::endl;
            std::exit(-1);
        }
        return bytes;
    }

    uint64_t block_size_ = 0;
    std::vector<char> memory_pool_;
    alignas(CACHE_LINE_SIZE) SPMCRingQueue<content_type, CAPACITY> producer_to_consumer_;
    alignas(CACHE_LINE_SIZE) MPSCRingQueue<char *, CAPACITY> consumer_to_producer_;
    alignas(CACHE_LINE_SIZE) std::atomic<int> active_producers_{0};
};

#endif // RING_MEMORY_POOL_HEADER
