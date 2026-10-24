#ifndef FINAL_DRAIN_WRITER_THREAD_HEADER
#define FINAL_DRAIN_WRITER_THREAD_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "SpinBackoff.h"
#include "KtpFormat.h"

#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

// high.bin 的写出线程：把 final drain 的 ring 块解码成 (key,count) 后写入 KTP1。
class FinalDrainWriterThread
{
    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> pool_;
    std::thread thread_;
    std::unique_ptr<ktp::Writer> ktp_;
    bool ktp_written_ = false;

    SpinBackoff<> consumer_enqueue_backoff;

public:

#ifdef TEST_MODE
    bool first_flag = false;
    uint64_t consumer_enqueue_spin_time{ 0 };
    uint64_t consumer_dequeue_spin_time{ 0 };
#endif

    FinalDrainWriterThread(uint32_t block_size, uint32_t producer_count)
        : pool_(block_size, producer_count),
          ktp_(std::make_unique<ktp::Writer>(k_length, /*has_counts=*/true)) {
    }

    auto* pool() { return &pool_; }

    void start()
    {
        thread_ = std::thread(&FinalDrainWriterThread::writer_loop, this);
    }

    void join()
    {
        if (thread_.joinable()) thread_.join();
        if (ktp_ && !ktp_written_)
        {
            ktp_written_ = true;
            try {
                ktp_->write(temp_dir + "high.final.tmp");
            } catch (...) {
                if (RunState::active) RunState::active->fail(std::current_exception(), "final writer");
                throw;
            }
        }
    }

    FinalDrainWriterThread(const FinalDrainWriterThread&) = delete;
    FinalDrainWriterThread& operator=(const FinalDrainWriterThread&) = delete;

    ~FinalDrainWriterThread()
    {
        RunState::fail_on_unwind();
        try { join(); } catch (...) {}
#ifdef TEST_MODE
        std::cout << "Final Drain Writer Thread consumer enqueue spin time: " << consumer_enqueue_spin_time << std::endl;
        std::cout << "Final Drain Writer Thread consumer dequeue spin time: " << consumer_dequeue_spin_time << std::endl;
#endif
    }

private:
    void writer_loop()
    {
        run_worker("final writer", [&] {
            SpinBackoff<> backoff;
            content_type content;
            while (true)
            {
                if (pool_.consumer_try_dequeue(content))
                {
#ifdef TEST_MODE
                    first_flag = true;
#endif
                    backoff.double_decay();
                    process_block(content);
                }
                else if (pool_.producer_finished())
                {
                    while (pool_.consumer_try_dequeue(content)) process_block(content);
                    break;
                }
                else
                {
#ifdef TEST_MODE
                    if (first_flag) consumer_dequeue_spin_time++;
#endif
                    backoff.backoff();
                }
            }
        });
    }

    void process_block(const content_type& content)
    {
        if (content.length == 0) [[unlikely]]
        {
            pool_.consumer_enqueue(content.data);
            return;
        }

        const uint64_t record_bytes = ::kmer_bytes() + count_max_bytes;
        const uint64_t records = content.length / record_bytes;
        const char* record = content.data;
        for (uint64_t i = 0; i < records; ++i, record += record_bytes)
        {
            uint32_t count = 0;
            std::memcpy(&count, record + ::kmer_bytes(), count_max_bytes);
            ktp_->add(record, count);
        }

        if (pool_.consumer_try_enqueue(content.data))
        {
            consumer_enqueue_backoff.double_decay();
        }
        else
        {
#ifdef TEST_MODE
            consumer_enqueue_spin_time++;
#endif
            consumer_enqueue_backoff.backoff();
            while (!pool_.consumer_try_enqueue(content.data))
            {
#ifdef TEST_MODE
                consumer_enqueue_spin_time++;
#endif
                consumer_enqueue_backoff.backoff();
            }
            consumer_enqueue_backoff.decay();
        }
    }
};

#endif // FINAL_DRAIN_WRITER_THREAD_HEADER
