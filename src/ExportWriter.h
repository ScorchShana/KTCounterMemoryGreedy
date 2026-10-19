#ifndef EXPORT_WRITER_HEADER
#define EXPORT_WRITER_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "kmer.h"
#include "SpinBackoff.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <array>
#include <thread>
#include <atomic>
#include <unordered_map>
#include <algorithm>

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <aio.h>






class ExportWriter
{

    constexpr static uint64_t RAW_BUFFER_SIZE = 2ULL * 1024 * 1024; // 1MB 原始数据块大小
    // constexpr static uint64_t RAW_BUFFER_SIZE = 512 * 1024;
    constexpr static uint32_t NUM_AIO_BUFFERS = 2;
    static_assert(RAW_BUFFER_SIZE >= EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, "RAW_BUFFER_SIZE must be greater than or equal to EXPORT_RING_MEMORY_POOL_BLOCK_SIZE");

private:
    // 连接到 FastqParser，用于获取含有低频 k-mer 的数据块
    const size_t record_bytes = kmer_bytes();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* ring_memory_pool = nullptr;

    struct aiocb cbs[NUM_AIO_BUFFERS];
    int fd;
    bool cbs_active[NUM_AIO_BUFFERS];
    std::array<char*, NUM_AIO_BUFFERS> buffer{};
    std::array<uint64_t, NUM_AIO_BUFFERS> buffer_count; // 当前缓冲区中 k-mer 的数量
    uint32_t current_buffer_index = 0;
    uint64_t file_offset = 0; // 已经写入文件的字节数

    // 专门执行落盘操作的单线程
    std::thread worker_thread;
    std::atomic<bool> is_running{ false };

public:
    ExportWriter() = delete;
    ExportWriter(const ExportWriter&) = delete;
    ExportWriter& operator=(const ExportWriter&) = delete;
    ExportWriter(ExportWriter&&) = delete;
    ExportWriter& operator=(ExportWriter&&) = delete;

    explicit ExportWriter(RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* pool,
        const std::string& filename = temp_dir + "low.bin")
        : ring_memory_pool(pool), fd(-1)
    {
        fd = ::open(filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) [[unlikely]]
        {
            std::cerr << "Failed to open export file: " << filename << std::endl;
            std::exit(-1);
        }

        for (int i = 0; i < NUM_AIO_BUFFERS; ++i)
        {
            std::memset(&cbs[i], 0, sizeof(struct aiocb));
            buffer[i] = nullptr;
            buffer_count[i] = 0;
            cbs_active[i] = false;
        }
    }

    void join()
    {
        if (is_running && worker_thread.joinable())
        {
            worker_thread.join();
            is_running = false;
        }

        const std::string filename = temp_dir + "low.bin";

        if (fd >= 0)
        {
            ::close(fd);
            fd = -1;
        }
    }

    ~ExportWriter()
    {
        if (fd >= 0)
        {
            ::close(fd);
            fd = -1;
        }
        for (auto& buf : buffer) {
            if (buf != nullptr)
            {
                ::free(buf);
                buf = nullptr;
            }
        }
    }

    void start()
    {
        is_running = true;
        worker_thread = std::thread(&ExportWriter::writer_loop, this);
    }

private:
    void writer_loop()
    {

        std::memset(cbs, 0, sizeof(cbs));

        for (uint32_t i = 0; i < NUM_AIO_BUFFERS; ++i) {
            void* buffer_ptr = nullptr;
            int ret = ::posix_memalign(&buffer_ptr, 4096, RAW_BUFFER_SIZE);
            if (ret != 0) {
                std::cerr << "posix_memalign failed for buffer: " << strerror(ret) << std::endl;
                std::exit(-1);
            }
            buffer[i] = static_cast<char*>(buffer_ptr);
            buffer_count[i] = 0;
        }

        SpinBackoff backoff;

        char* block_ptr = nullptr;


        content_type content;

        while (true)
        {
            if (ring_memory_pool->consumer_try_dequeue(content))
            {
                backoff.decay();
                block_ptr = content.data;
                ExportBlock* export_kmer_block = reinterpret_cast<ExportBlock *>(block_ptr);
                if (content.length > 0) [[likely]]
                {
                    process_block(export_kmer_block, content.length);
                }
                else {
                    ring_memory_pool->consumer_enqueue(block_ptr);
                }
                // ring_memory_pool->consumer_enqueue(block_ptr);

            }
            else if (ring_memory_pool->producer_finished())
            {
                while (ring_memory_pool->consumer_try_dequeue(content))
                {
                    block_ptr = content.data;
                    ExportBlock* export_kmer_block = reinterpret_cast<ExportBlock *>(block_ptr);
                    if (content.length > 0) [[likely]]
                    {
                        process_block(export_kmer_block, content.length);
                    }
                    else {
                        ring_memory_pool->consumer_enqueue(block_ptr);
                    }
                    cpu_relax();
                }
                break;
            }
            else
            {
                backoff.backoff();
            }
        }


        for (uint32_t i = 0; i < NUM_AIO_BUFFERS; ++i)
        {
            if (cbs_active[i])
            {
                wait_write_finish(i);
            }
        }

        if (buffer_count[current_buffer_index] > 0)
        {
            async_write(current_buffer_index);
            wait_write_finish(current_buffer_index);
        }

        for (auto& buf : buffer) {
            if (buf != nullptr)
            {
                ::free(buf);
                buf = nullptr;
            }
        }
    }

    void process_block(ExportBlock* block, uint64_t count)
    {
        const unsigned char* input = block->k_mers.data();
        size_t remaining = count * record_bytes;
        while (remaining != 0) {
            const size_t copied = std::min<size_t>(remaining,
                RAW_BUFFER_SIZE - buffer_count[current_buffer_index]);
            std::memcpy(buffer[current_buffer_index] + buffer_count[current_buffer_index], input, copied);
            buffer_count[current_buffer_index] += copied;
            input += copied;
            remaining -= copied;
            if (buffer_count[current_buffer_index] == RAW_BUFFER_SIZE) {
                async_write(current_buffer_index);
                current_buffer_index = (current_buffer_index + 1) % NUM_AIO_BUFFERS;
                wait_write_finish(current_buffer_index);
            }
        }
        ring_memory_pool->consumer_enqueue(reinterpret_cast<char*>(block));
    }

    void wait_write_finish(const uint32_t buffer_index)
    {
        if (!cbs_active[buffer_index]) [[unlikely]]
        {
            return;
        }

        const aiocb* list[1] = { &cbs[buffer_index] };
        int err;
        while ((err = ::aio_error(&cbs[buffer_index])) == EINPROGRESS)
        {
            if (::aio_suspend(list, 1, nullptr) != 0 && errno != EINTR) [[unlikely]]
            {
                std::cerr << "aio_suspend failed\n";
                std::exit(-1);
            }
        }

        err = ::aio_error(&cbs[buffer_index]);
        if (err != 0) [[unlikely]]
        {
            std::cerr << "aio_write error\n";
            std::exit(-1);
        }

        ssize_t n = ::aio_return(&cbs[buffer_index]);
        if (n != static_cast<ssize_t>(cbs[buffer_index].aio_nbytes)) [[unlikely]]
        {
            std::cerr << "partial aio_write\n";
            std::exit(-1);
        }

        cbs_active[buffer_index] = false;
    }

    void async_write(const uint32_t buffer_index)
    {
        std::memset(&cbs[buffer_index], 0, sizeof(struct aiocb));
        cbs[buffer_index].aio_fildes = fd;
        cbs[buffer_index].aio_buf = buffer[buffer_index];
        cbs[buffer_index].aio_nbytes = buffer_count[buffer_index];
        cbs[buffer_index].aio_offset = file_offset;

        if (::aio_write(&cbs[buffer_index]) != 0) [[unlikely]]
        {
            std::cerr << "aio_write failed\n";
            std::exit(-1);
        }

        file_offset += buffer_count[buffer_index];
        buffer_count[buffer_index] = 0;
        cbs_active[buffer_index] = true;
    }

    constexpr uint64_t get_root_prefix(const unsigned char* val) const
    {
        return val[0];
    }

};



#endif // EXPORT_WRITER_HEADER
