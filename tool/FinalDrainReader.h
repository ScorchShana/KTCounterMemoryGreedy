#ifndef FINAL_DRAIN_READER_HEADER
#define FINAL_DRAIN_READER_HEADER

#include "../src/definition.h"
#include "../src/kmer.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <aio.h>

template <uint32_t N>
class FinalDrainReader
{
    constexpr static uint64_t RAW_BUFFER_SIZE = 512 * 1024;

    int fd = -1;
    uint32_t k_length;
    uint64_t kmer_bytes;           // compact k-mer bytes in file
    uint32_t count_bytes_;        // count width (1-4 bytes)
    uint64_t compact_record_size; // kmer_bytes + count_bytes_
    uint64_t full_words;
    uint64_t tail_bits;
    uint64_t tail_bytes;

    uint64_t record_amount = 0;
    uint64_t scheduled_record_count = 0;
    uint64_t delivered_record_count = 0;
    uint32_t current_buffer_index = 0;
    std::array<struct aiocb, 2> cbs{};
    std::array<char*, 2> buffer{};
    std::array<bool, 2> cbs_active{};
    std::array<uint64_t, 2> buffer_record_count{};
    std::array<uint64_t, 2> buffer_cursor{};

public:
    explicit FinalDrainReader(uint32_t k, uint32_t count_bytes = sizeof(uint32_t))
        : k_length(k), count_bytes_(count_bytes)
    {
        full_words = k_length / BASES_PER_U64T;
        tail_bits = 2 * (k_length % BASES_PER_U64T);
        tail_bytes = (tail_bits + 7) / 8;
        kmer_bytes = full_words * sizeof(uint64_t) + tail_bytes;
        compact_record_size = kmer_bytes + count_bytes_;

        for (uint32_t i = 0; i < 2; ++i)
        {
            std::memset(&cbs[i], 0, sizeof(struct aiocb));
        }
    }

    FinalDrainReader(const FinalDrainReader&) = delete;
    FinalDrainReader& operator=(const FinalDrainReader&) = delete;
    FinalDrainReader(FinalDrainReader&&) = delete;
    FinalDrainReader& operator=(FinalDrainReader&&) = delete;

    ~FinalDrainReader()
    {
        close();
    }

    void open(const std::string& filename)
    {
        if (fd >= 0)
        {
            close();
        }

        fd = ::open(filename.c_str(), O_RDONLY);
        if (fd < 0) [[unlikely]]
        {
            std::cerr << "Failed to open file: " << filename << std::endl;
            std::exit(-1);
        }

        struct stat st;
        if (::fstat(fd, &st) != 0) [[unlikely]]
        {
            std::cerr << "Failed to stat file: " << filename << std::endl;
            std::exit(-1);
        }

        const uint64_t file_size = static_cast<uint64_t>(st.st_size);
        if (file_size % compact_record_size != 0) [[unlikely]]
        {
            std::cerr << "Invalid final drain file size: " << file_size
                << " is not divisible by compact record size "
                << compact_record_size << std::endl;
            std::exit(-1);
        }

        ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

        record_amount = file_size / compact_record_size;
        scheduled_record_count = 0;
        delivered_record_count = 0;
        current_buffer_index = 0;

        for (uint32_t i = 0; i < 2; ++i)
        {
            std::memset(&cbs[i], 0, sizeof(struct aiocb));
            if (buffer[i] == nullptr)
            {
                buffer[i] = new char[RAW_BUFFER_SIZE];
            }
            cbs_active[i] = false;
            buffer_record_count[i] = 0;
            buffer_cursor[i] = 0;
        }

        async_read(0);
    }

    void close()
    {
        for (uint32_t i = 0; i < 2; ++i)
        {
            wait_read_finish(i);
        }

        if (fd >= 0)
        {
            ::close(fd);
            fd = -1;
        }

        for (uint32_t i = 0; i < 2; ++i)
        {
            delete[] buffer[i];
            buffer[i] = nullptr;
            cbs_active[i] = false;
            buffer_record_count[i] = 0;
            buffer_cursor[i] = 0;
            std::memset(&cbs[i], 0, sizeof(struct aiocb));
        }

        record_amount = 0;
        scheduled_record_count = 0;
        delivered_record_count = 0;
        current_buffer_index = 0;
    }

    uint64_t get_record_amount() const
    {
        return record_amount;
    }

    bool finished() const noexcept
    {
        return delivered_record_count >= record_amount;
    }

    // Read compact records from file, expand to full ExportRecord<N> in memory
    uint64_t read_records(ExportRecord<N>* out_buffer, uint64_t max_records_to_read)
    {
        if (finished()) [[unlikely]]
        {
            return 0;
        }

        uint64_t total_read = 0;
        while (total_read < max_records_to_read && !finished())
        {
            uint32_t idx = current_buffer_index;

            if (cbs_active[idx])
            {
                wait_read_finish(idx);
                async_read(1 - idx);
            }

            if (buffer_cursor[idx] >= buffer_record_count[idx])
            {
                current_buffer_index = 1 - idx;
                continue;
            }

            const uint64_t available = buffer_record_count[idx] - buffer_cursor[idx];
            if (available == 0)
            {
                current_buffer_index = 1 - idx;
                continue;
            }

            const uint64_t to_copy = std::min(max_records_to_read - total_read, available);

            // Expand compact records to full ExportRecord<N>
            for (uint64_t i = 0; i < to_copy; ++i)
            {
                const char* src = buffer[idx] + buffer_cursor[idx] * compact_record_size + i * compact_record_size;
                ExportRecord<N>& dst = out_buffer[total_read + i];

                // Zero out the full k-mer first
                dst.key.reset();

                // Copy full uint64_t words
                std::memcpy(dst.key.data.data(), src, full_words * sizeof(uint64_t));

                // Copy tail bytes and expand to uint64_t (MSB-aligned)
                if (tail_bytes > 0)
                {
                    uint64_t tail_data = 0;
                    std::memcpy(reinterpret_cast<char*>(&tail_data) + (8 - tail_bytes),
                                src + full_words * sizeof(uint64_t), tail_bytes);
                    dst.key.data[full_words] = tail_data;
                }

                // Copy count (count_bytes_ bytes, expanded to uint32_t)
                dst.count = 0;
                std::memcpy(&dst.count, src + kmer_bytes, count_bytes_);
            }

            buffer_cursor[idx] += to_copy;
            delivered_record_count += to_copy;
            total_read += to_copy;

            if (buffer_cursor[idx] >= buffer_record_count[idx])
            {
                buffer_record_count[idx] = 0;
                buffer_cursor[idx] = 0;
                current_buffer_index = 1 - idx;
            }
        }

        return total_read;
    }

private:
    void async_read(uint32_t buffer_index)
    {
        if (scheduled_record_count >= record_amount || cbs_active[buffer_index]) [[unlikely]]
        {
            return;
        }

        const uint64_t remaining = record_amount - scheduled_record_count;
        const uint64_t buffer_capacity = RAW_BUFFER_SIZE / compact_record_size;
        const uint64_t read_count = std::min(buffer_capacity, remaining);
        if (read_count == 0)
        {
            return;
        }

        const uint64_t read_bytes = read_count * compact_record_size;
        const uint64_t read_offset = scheduled_record_count * compact_record_size;

        buffer_record_count[buffer_index] = read_count;
        buffer_cursor[buffer_index] = 0;

        std::memset(&cbs[buffer_index], 0, sizeof(struct aiocb));
        cbs[buffer_index].aio_fildes = fd;
        cbs[buffer_index].aio_buf = buffer[buffer_index];
        cbs[buffer_index].aio_nbytes = read_bytes;
        cbs[buffer_index].aio_offset = static_cast<off_t>(read_offset);

        if (::aio_read(&cbs[buffer_index]) != 0) [[unlikely]]
        {
            std::cerr << "aio_read failed" << std::endl;
            std::exit(-1);
        }

        cbs_active[buffer_index] = true;
        scheduled_record_count += read_count;
    }

    void wait_read_finish(const uint32_t buffer_index)
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
                std::cerr << "aio_suspend failed" << std::endl;
                std::exit(-1);
            }
        }

        err = ::aio_error(&cbs[buffer_index]);
        if (err != 0) [[unlikely]]
        {
            std::cerr << "aio_read error: " << err << std::endl;
            std::exit(-1);
        }

        ssize_t n = ::aio_return(&cbs[buffer_index]);
        if (n != static_cast<ssize_t>(cbs[buffer_index].aio_nbytes)) [[unlikely]]
        {
            std::cerr << "partial aio_read" << std::endl;
            std::exit(-1);
        }

        cbs_active[buffer_index] = false;
    }
};

#endif