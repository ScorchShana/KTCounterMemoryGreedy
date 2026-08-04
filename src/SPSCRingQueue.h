#ifndef SPSC_RING_QUEUE_HEADER
#define SPSC_RING_QUEUE_HEADER


#include "definition.h"
#include "SpinBackoff.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <type_traits>
#include <utility>

template <typename T, std::size_t Capacity>
class SPSCRingQueue
{

    static_assert(Capacity > 1, "Capacity must be greater than 1");
    static_assert((Capacity& (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    struct alignas(CACHE_LINE_SIZE) Cell
    {
        std::atomic<std::size_t> sequence;
        T data;
    };

    constexpr static std::size_t mask = Capacity - 1;

    constexpr static int SPIN_LIMIT = 1024;
    constexpr static int BACKOFF_LIMIT = 64;

    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> enqueue_pos;
    alignas(CACHE_LINE_SIZE) std::atomic<std::size_t> dequeue_pos;
    std::array<Cell, Capacity> buffer;

public:
    SPSCRingQueue(const SPSCRingQueue&) = delete;
    SPSCRingQueue& operator=(const SPSCRingQueue&) = delete;

    SPSCRingQueue() : enqueue_pos(0), dequeue_pos(0)
    {
        for (size_t i = 0; i < Capacity; ++i)
        {
            buffer[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    bool try_enqueue(const T& item)
    {
        Cell* cell;
        std::size_t pos = enqueue_pos.load(std::memory_order_relaxed);

        for (;;)
        {
            cell = &buffer[pos & mask];
            const std::size_t seq = cell->sequence.load(std::memory_order_acquire);

            if (seq == pos)
            {
                enqueue_pos.store(pos + 1, std::memory_order_relaxed);
                cell->data = item;
                // 发布数据
                cell->sequence.store(pos + 1, std::memory_order_release);
                return true;
            }
            else if (seq < pos)
            {
                return false; // genuinely full
            }

            cpu_relax();
            pos = enqueue_pos.load(std::memory_order_relaxed);
        }
    }

    void enqueue(const T& item)
    {
        SpinBackoff backoff;
        while (!try_enqueue(item))
        {
            backoff.backoff();
        }
    }

    bool try_dequeue(T& item)
    {
        Cell* cell;
        std::size_t pos = dequeue_pos.load(std::memory_order_relaxed);

        for (;;)
        {
            cell = &buffer[pos & mask];
            const std::size_t seq = cell->sequence.load(std::memory_order_acquire);

            if (seq == pos + 1)
            {
                dequeue_pos.store(pos + 1, std::memory_order_relaxed);
                item = cell->data;
                cell->sequence.store(pos + Capacity, std::memory_order_release);
                return true;
            }
            else if (seq < pos + 1)
            {
                return false; // genuinely empty
            }

            cpu_relax();
            pos = dequeue_pos.load(std::memory_order_relaxed);
        }
    }


    void dequeue(T& item)
    {
        SpinBackoff backoff;
        while (!try_dequeue(item))
        {
            backoff.backoff();
        }
    }

    std::uint64_t size() const
    {
        std::size_t cur_enqueue_pos = enqueue_pos.load(std::memory_order_relaxed);
        std::size_t cur_dequeue_pos = dequeue_pos.load(std::memory_order_relaxed);

        return static_cast<std::uint64_t>(cur_enqueue_pos - cur_dequeue_pos);
    }

    bool empty() const
    {
        std::size_t cur_enqueue_pos = enqueue_pos.load(std::memory_order_relaxed);
        std::size_t cur_dequeue_pos = dequeue_pos.load(std::memory_order_relaxed);

        return cur_enqueue_pos == cur_dequeue_pos;
    }
};

#endif