#ifndef LOCK_QUEUE_HEADER
#define LOCK_QUEUE_HEADER

#include <cstdint>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <cassert>
#include <cstdlib>
#include <iostream>

template <typename T, uint64_t CAPACITY = 4096>
class RingLockQueue
{

    static_assert((CAPACITY & (CAPACITY - 1)) == 0, "RingLockQueue capacity must be a power of 2");

public:
    explicit RingLockQueue(uint32_t active_producers)
        : active_producers_(active_producers), buffer_(CAPACITY)
    {
        if (active_producers == 0)
        {
            std::cerr << "RingLockQueue active_producers must be > 0" << std::endl;
            std::exit(-1);
        }
    }

    bool enqueue_no_lock(const T &item)
    {
        if (full())
        {
            return false;
        }
        buffer_[tail_] = item;
        ++size_;
        tail_ = (tail_ + 1) & (CAPACITY - 1);
        return true;
    }

    void enqueue(const T &item)
    {
        std::unique_lock lock(mtx_);

        not_full_cv_.wait(lock, [this]()
                          { return (!full()); });

        const bool was_empty = (size_ == 0);
        buffer_[tail_] = item;
        ++size_;
        tail_ = (tail_ + 1) & (CAPACITY - 1);

        if (was_empty)
        {
            not_empty_cv_.notify_one();
        }
    }

    bool dequeue(T &item)
    {
        std::unique_lock lock(mtx_);
        not_empty_cv_.wait(lock, [this]()
                           { return (!empty()) || active_producers_ == 0; });

        if (empty())
        {
            return false;
        }

        const bool was_full = full();
        item = buffer_[head_];
        head_ = (head_ + 1) & (CAPACITY - 1);
        --size_;

        if (was_full)
        {
            not_full_cv_.notify_one();
        }
        return true;
    }

    inline uint64_t size() const
    {
        return size_;
    }

    inline bool empty() const
    {
        return size_ == 0;
    }

    inline bool full() const
    {
        return size_ >= CAPACITY;
    }

    bool finished()
    {
        std::lock_guard lock(mtx_);
        return active_producers_ == 0 && empty();
    }

    void set_finished()
    {
        std::lock_guard lock(mtx_);
        --active_producers_;
        if (active_producers_ == 0)
        {
            not_empty_cv_.notify_all();
        }
    }

private:
    std::vector<T> buffer_;
    std::condition_variable not_empty_cv_;
    std::condition_variable not_full_cv_;
    uint64_t head_ = 0;
    uint64_t tail_ = 0;
    uint64_t size_ = 0;
    uint32_t active_producers_ = 0;
    alignas(64) std::mutex mtx_;
};

#endif