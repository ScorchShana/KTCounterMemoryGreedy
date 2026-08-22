#ifndef FINAL_DRAIN_WRITER_THREAD_HEADER
#define FINAL_DRAIN_WRITER_THREAD_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "SpinBackoff.h"

#include <aio.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <array>
#include <cstring>
#include <iostream>

class FinalDrainWriterThread
{
    static constexpr uint64_t AIO_BUFFER_SIZE = 512 * 1024;
    static constexpr uint32_t NUM_AIO_BUFFERS = 4;
    // static constexpr uint64_t AIO_BUFFER_SIZE = 1ULL * 1024 * 1024;

    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> pool_;
    int fd_;
    std::thread thread_;

    struct aiocb cbs_[NUM_AIO_BUFFERS];
    bool cbs_active_[NUM_AIO_BUFFERS];
    std::array<char*, NUM_AIO_BUFFERS> buffer_;
    std::array<uint64_t, NUM_AIO_BUFFERS> buffer_count_;
    uint32_t current_buffer_index_ = 0;
    uint64_t file_offset_ = 0;

    SpinBackoff<> consumer_enqueue_backoff;

public:

#ifdef TEST_MODE
    bool first_flag = false;
    uint64_t consumer_enqueue_spin_time{ 0 };
    uint64_t consumer_dequeue_spin_time{ 0 };
#endif
    FinalDrainWriterThread(uint32_t block_size, uint32_t producer_count)
        : pool_(block_size, producer_count), fd_(-1) {
    }

    auto* pool() { return &pool_; }

    void start()
    {
        fd_ = ::open((temp_dir + "high.bin").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) [[unlikely]]
        {
            std::cerr << "Failed to open high.bin\n";
            std::exit(-1);
        }
        for (uint32_t i = 0; i < NUM_AIO_BUFFERS; ++i)
        {
            cbs_active_[i] = false;
            buffer_[i] = nullptr;
            buffer_count_[i] = 0;
        }
        thread_ = std::thread(&FinalDrainWriterThread::writer_loop, this);
    }

    void join()
    {
        if (thread_.joinable()) thread_.join();
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
        for (int i = 0; i < NUM_AIO_BUFFERS; i++)
        {
            if (buffer_[i] != nullptr)
            {
                ::free(buffer_[i]);
                buffer_[i] = nullptr;
            }
        }
    }

    FinalDrainWriterThread(const FinalDrainWriterThread&) = delete;
    FinalDrainWriterThread& operator=(const FinalDrainWriterThread&) = delete;

    ~FinalDrainWriterThread()
    {
        join();
#ifdef TEST_MODE
        std::cout << "Final Drain Writer Thread consumer enqueue spin time: " << consumer_enqueue_spin_time << std::endl;
        std::cout << "Final Drain Writer Thread consumer dequeue spin time: " << consumer_dequeue_spin_time << std::endl;
#endif
    }

private:
    void writer_loop()
    {
        for (int i = 0; i < NUM_AIO_BUFFERS; i++)
        {
            void* buffer_ptr = nullptr;
            int ret = ::posix_memalign(&buffer_ptr, 4096, AIO_BUFFER_SIZE);
            if (ret != 0) {
                std::cerr << "posix_memalign failed for buffer: " << strerror(ret) << std::endl;
                std::exit(-1);
            }
            buffer_[i] = static_cast<char*>(buffer_ptr);
        }

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
                while (pool_.consumer_try_dequeue(content))
                {
                    process_block(content);
                    cpu_relax();
                }
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

        for (uint32_t i = 0; i < NUM_AIO_BUFFERS; ++i)
        {
            if (cbs_active_[i])
            {
                wait_write_finish(i);
            }
        }
        if (buffer_count_[current_buffer_index_] > 0)
        {
            async_write(current_buffer_index_);
            wait_write_finish(current_buffer_index_);
        }

        for (uint32_t i = 0; i < NUM_AIO_BUFFERS; ++i)
        {
            if (buffer_[i]) ::free(buffer_[i]);
            buffer_[i] = nullptr;
        }
    }

    void process_block(const content_type& content)
    {
        if (content.length == 0) [[unlikely]]
        {
            pool_.consumer_enqueue(content.data);
            return;
        }

        if (buffer_count_[current_buffer_index_] + content.length >= AIO_BUFFER_SIZE)
        {
            uint64_t to_copy = AIO_BUFFER_SIZE - buffer_count_[current_buffer_index_];
            std::memcpy(buffer_[current_buffer_index_] + buffer_count_[current_buffer_index_],
                content.data, to_copy);
            uint64_t remaining = content.length - to_copy;
            buffer_count_[current_buffer_index_] += to_copy;

            async_write(current_buffer_index_);
            current_buffer_index_ = (current_buffer_index_ + 1) % NUM_AIO_BUFFERS;
            wait_write_finish(current_buffer_index_);

            std::memcpy(buffer_[current_buffer_index_], content.data + to_copy, remaining);
            buffer_count_[current_buffer_index_] = remaining;
        }
        else
        {
            std::memcpy(buffer_[current_buffer_index_] + buffer_count_[current_buffer_index_],
                content.data, content.length);
            buffer_count_[current_buffer_index_] += content.length;
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

    void async_write(uint32_t idx)
    {
        std::memset(&cbs_[idx], 0, sizeof(struct aiocb));
        cbs_[idx].aio_fildes = fd_;
        cbs_[idx].aio_buf = buffer_[idx];
        cbs_[idx].aio_nbytes = buffer_count_[idx];
        cbs_[idx].aio_offset = file_offset_;

        if (::aio_write(&cbs_[idx]) != 0) [[unlikely]]
        {
            std::cerr << "aio_write failed\n";
            std::exit(-1);
        }

        file_offset_ += buffer_count_[idx];
        buffer_count_[idx] = 0;
        cbs_active_[idx] = true;
    }

    void wait_write_finish(uint32_t idx)
    {
        if (!cbs_active_[idx]) [[unlikely]]
        {
            return;
        }

        const aiocb* list[1] = { &cbs_[idx] };
        int err;
        while ((err = ::aio_error(&cbs_[idx])) == EINPROGRESS)
        {
            if (::aio_suspend(list, 1, nullptr) != 0 && errno != EINTR) [[unlikely]]
            {
                std::cerr << "aio_suspend failed\n";
                std::exit(-1);
            }
        }

        err = ::aio_error(&cbs_[idx]);
        if (err != 0) [[unlikely]]
        {
            std::cerr << "aio_write error\n";
            std::exit(-1);
        }

        ssize_t n = ::aio_return(&cbs_[idx]);
        if (n != static_cast<ssize_t>(cbs_[idx].aio_nbytes)) [[unlikely]]
        {
            std::cerr << "partial aio_write\n";
            std::exit(-1);
        }

        cbs_active_[idx] = false;
    }
};

/*
class FinalDrainWriterThread
{
    static constexpr uint64_t RAW_BUFFER_SIZE = 4ULL * 1024 * 1024;
    static_assert(RAW_BUFFER_SIZE >= FINAL_DRAIN_RING_POOL_BLOCK_SIZE,
        "RAW_BUFFER_SIZE must be greater than or equal to FINAL_DRAIN_RING_POOL_BLOCK_SIZE");

    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> pool_;
    int fd_;
    std::thread thread_;

    char* buffer_ = nullptr;
    uint64_t buffer_count_ = 0;

    SpinBackoff<> consumer_enqueue_backoff;

public:

#ifdef TEST_MODE
    bool first_flag = false;
    uint64_t consumer_enqueue_spin_time{ 0 };
    uint64_t consumer_dequeue_spin_time{ 0 };
#endif
    FinalDrainWriterThread(uint32_t block_size, uint32_t producer_count)
        : pool_(block_size, producer_count), fd_(-1) {
    }

    auto* pool() { return &pool_; }

    void start()
    {
        fd_ = ::open((temp_dir + "high.bin").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) [[unlikely]]
        {
            std::cerr << "Failed to open high.bin\n";
            std::exit(-1);
        }
        thread_ = std::thread(&FinalDrainWriterThread::writer_loop, this);
    }

    void join()
    {
        if (thread_.joinable()) thread_.join();
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
        if (buffer_ != nullptr)
        {
            ::free(buffer_);
            buffer_ = nullptr;
        }
    }

    FinalDrainWriterThread(const FinalDrainWriterThread&) = delete;
    FinalDrainWriterThread& operator=(const FinalDrainWriterThread&) = delete;

    ~FinalDrainWriterThread()
    {
        join();
#ifdef TEST_MODE
        std::cout << "Final Drain Writer Thread consumer enqueue spin time: " << consumer_enqueue_spin_time << std::endl;
        std::cout << "Final Drain Writer Thread consumer dequeue spin time: " << consumer_dequeue_spin_time << std::endl;
#endif
    }

private:
    void writer_loop()
    {
        void* buffer_ptr = nullptr;
        int ret = ::posix_memalign(&buffer_ptr, 4096, RAW_BUFFER_SIZE);
        if (ret != 0) {
            std::cerr << "posix_memalign failed for buffer: " << strerror(ret) << std::endl;
            std::exit(-1);
        }
        buffer_ = static_cast<char*>(buffer_ptr);
        buffer_count_ = 0;

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
                while (pool_.consumer_try_dequeue(content))
                {
                    process_block(content);
                    cpu_relax();
                }
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

        if (buffer_count_ > 0)
        {
            sync_write();
        }

        if (buffer_)
        {
            ::free(buffer_);
            buffer_ = nullptr;
        }
    }

    void process_block(const content_type& content)
    {
        if (content.length == 0) [[unlikely]]
        {
            pool_.consumer_enqueue(content.data);
            return;
        }

        if (buffer_count_ + content.length >= RAW_BUFFER_SIZE)
        {
            uint64_t to_copy = RAW_BUFFER_SIZE - buffer_count_;
            std::memcpy(buffer_ + buffer_count_, content.data, to_copy);
            buffer_count_ += to_copy;
            uint64_t remaining = content.length - to_copy;

            sync_write();

            std::memcpy(buffer_, content.data + to_copy, remaining);
            buffer_count_ = remaining;
        }
        else
        {
            std::memcpy(buffer_ + buffer_count_, content.data, content.length);
            buffer_count_ += content.length;
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

    void sync_write()
    {
        uint64_t written = 0;
        while (written < buffer_count_)
        {
            const ssize_t n = ::write(fd_, buffer_ + written, buffer_count_ - written);
            if (n < 0)
            {
                if (errno == EINTR) continue;
                std::cerr << "write failed\n";
                std::exit(-1);
            }
            if (n == 0)
            {
                std::cerr << "write returned 0\n";
                std::exit(-1);
            }
            written += static_cast<uint64_t>(n);
        }
        buffer_count_ = 0;
    }
};
*/

#endif // FINAL_DRAIN_WRITER_THREAD_HEADER
