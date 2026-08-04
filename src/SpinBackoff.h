#ifndef SPIN_BACKOFF_HEADER
#define SPIN_BACKOFF_HEADER

#include "definition.h"

#include <chrono>


template <int MAX_BACKOFF = 64, int YIELD_THRESHOLD = 64, int SLEEP_THRESHOLD = 64 + 16>
class SpinBackoff {
    static constexpr int BACKOFF_START = 1; // 初始 backoff 次数为 1
    static constexpr int MAX_SLEEP_TIME_US = 500; // 最大睡眠时间为 500 微秒
public:
    void backoff()
    {
        if (count_ < YIELD_THRESHOLD)
        {
            // 动态自旋：执行 backoff_ 次 cpu_relax
            for (int i = 0; i < backoff_; i++)
            {
                cpu_relax();
            }
            // 指数增长，但限制上限
            backoff_ = std::min(backoff_ * 2, MAX_BACKOFF);
        }
        else if (count_ < SLEEP_THRESHOLD)
        {
            std::this_thread::yield();
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::microseconds(sleep_time_us_));
            sleep_time_us_ = std::min(sleep_time_us_ * 2, MAX_SLEEP_TIME_US);
        }
        ++count_;
    }

    void decay()
    {
        count_ >>= 1;          // 阶段计数器指数衰减
        backoff_ = (backoff_ > 1) ? backoff_ / 2 : 1; // backoff 也指数衰减
        sleep_time_us_ = (sleep_time_us_ > 1) ? sleep_time_us_ / 2 : 1; // sleep_time 也指数衰减
    }

    void double_decay()
    {
        count_ >>= 2;          // 阶段计数器双倍指数衰减
        backoff_ = (backoff_ > 3) ? backoff_ / 4 : 1; // backoff 也双倍指数衰减
        sleep_time_us_ = (sleep_time_us_ > 3) ? sleep_time_us_ / 4 : 1; // sleep_time 也双倍指数衰减
    }

    void reset()
    {
        count_ = 0;
        backoff_ = BACKOFF_START; // 重置 backoff 次数为初始值
        sleep_time_us_ = 1;
    }

private:
    int count_ = 0;
    int backoff_ = BACKOFF_START;          // pause 初始值
    int sleep_time_us_ = 1;    // 初始睡眠时间为 1 微秒
};

#endif