#ifndef SPIN_LOCK_HEADER
#define SPIN_LOCK_HEADER

#include <atomic>
#include <cstdint>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <emmintrin.h>
static inline void cpu_relax() noexcept
{
    _mm_pause();
}
#elif defined(__arm__) || defined(__aarch64__) || defined(_M_ARM64)
static inline void cpu_relax() noexcept
{
#if (defined(__ARM_ARCH_6K__) ||  \
     defined(__ARM_ARCH_6Z__) ||  \
     defined(__ARM_ARCH_6ZK__) || \
     defined(__ARM_ARCH_6T2__) || \
     defined(__ARM_ARCH_7__) ||   \
     defined(__ARM_ARCH_7A__) ||  \
     defined(__ARM_ARCH_7R__) ||  \
     defined(__ARM_ARCH_7M__) ||  \
     defined(__ARM_ARCH_7S__) ||  \
     defined(__ARM_ARCH_8A__) ||  \
     defined(__aarch64__))
    asm volatile("yield" ::: "memory");

#elif defined(_M_ARM64)
    __yield();

#else
    asm volatile("nop" ::: "memory");
#endif
}
#endif

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
        int backoff_iterations = 2;
        constexpr int MAX_BACKOFF = 64;
        int spin_count = 0;
        constexpr int YIELD_THRESHOLD = 256;

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
                if (spin_count < YIELD_THRESHOLD)
                {
                    // 执行 'backoff_iterations' 次暂停指令
                    for (int i = 0; i < backoff_iterations; ++i)
                    {
                        cpu_relax();
                    }
                    // 指数增加退避时间，直到上限
                    if (backoff_iterations < MAX_BACKOFF)
                    {
                        backoff_iterations *= 2;
                    }
                    spin_count++;
                }
                else
                {
                    // 3. 向操作系统让出：如果我们自旋太久，持锁线程
                    // 可能已被抢占。让操作系统调度其他线程。
                    std::this_thread::yield();
                }
            }

            // 2. 测试并设置（原子写）：尝试获取锁
            if (!flag.test_and_set(std::memory_order_acquire))
            {
#ifdef TEST_MODE
                add_spin_loops(local_spin_loops);
#endif
                return; // 获取成功
            }

            // 如果执行到这里，说明另一个线程刚好在我们之前抢到了锁。
            // 这意味着竞争激烈。增加退避时间。
            if (backoff_iterations < MAX_BACKOFF)
            {
                backoff_iterations *= 2;
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