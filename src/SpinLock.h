#ifndef SPIN_LOCK_HEADER
#define SPIN_LOCK_HEADER

#include "SpinBackoff.h"

#include <atomic>
#include <cstdint>
#include <thread>

class SpinLock
{
private:
    std::atomic_flag flag = ATOMIC_FLAG_INIT;

#ifdef TEST_MODE
    static inline std::atomic<uint64_t> global_spin_loops_{0};
    static inline thread_local uint64_t tls_spin_loops_{0};
    static constexpr uint64_t SPIN_LOOP_FLUSH_THRESHOLD = 1ULL << 14;

    static inline void add_spin_loops(const uint64_t loops) noexcept
    {
        if (loops == 0)
        {
            return;
        }

        tls_spin_loops_ += loops;
        if (tls_spin_loops_ >= SPIN_LOOP_FLUSH_THRESHOLD)
        {
            global_spin_loops_.fetch_add(tls_spin_loops_, std::memory_order_relaxed);
            tls_spin_loops_ = 0;
        }
    }
#endif

public:
#ifdef TEST_MODE
    static inline void flush_spin_loops_for_current_thread() noexcept
    {
        if (tls_spin_loops_ != 0)
        {
            global_spin_loops_.fetch_add(tls_spin_loops_, std::memory_order_relaxed);
            tls_spin_loops_ = 0;
        }
    }

    static inline uint64_t spin_loops() noexcept
    {
        return global_spin_loops_.load(std::memory_order_relaxed) + tls_spin_loops_;
    }

    static inline void reset_spin_loops() noexcept
    {
        global_spin_loops_.store(0, std::memory_order_relaxed);
        tls_spin_loops_ = 0;
    }
#else
    static inline void flush_spin_loops_for_current_thread() noexcept {}
    static inline uint64_t spin_loops() noexcept { return 0; }
    static inline void reset_spin_loops() noexcept {}
#endif

    void lock()
    {
        SpinBackoff<> backoff;

#ifdef TEST_MODE
        uint64_t local_spin_loops = 0;
#endif

        while (true)
        {
            // 1. 测试（只读）：当锁被持有时自旋
            while (flag.test(std::memory_order_relaxed))
            {
#ifdef TEST_MODE
                ++local_spin_loops;
#endif
                backoff.backoff();
            }

            // 2. 测试并设置（原子写）：尝试获取锁
            if (!flag.test_and_set(std::memory_order_acquire))
            {
#ifdef TEST_MODE
                add_spin_loops(local_spin_loops);
#endif
                return; // 获取成功
            }

        }
    }

    bool try_lock()
    {
        return !flag.test_and_set(std::memory_order_acquire);
    }

    void unlock()
    {
        // memory_order_release 保证释放锁之前的读写不会重排到释放锁之后
        flag.clear(std::memory_order_release);
    }
};

//const int a=sizeof(SpinLock);


#endif