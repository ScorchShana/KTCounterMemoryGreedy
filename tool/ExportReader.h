#ifndef EXPORT_READER_HEADER
#define EXPORT_READER_HEADER

#include "../src/definition.h"
#include "../src/kmer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <aio.h>
#include <iostream>

template <uint32_t N>
class ExportReader
{
    constexpr static uint64_t RAW_BUFFER_SIZE = 512 * 1024;

    int fd;
    uint32_t k;
    uint64_t full_data_count;
    uint64_t tail_bits;
    uint64_t tail_bytes;
    uint64_t packed_kmer_bytes;
    const uint64_t buffer_kmer_capacity;
    const uint64_t buffer_bytes;
    uint64_t kmer_amount;
    uint64_t scheduled_kmer_count;
    uint64_t delivered_kmer_count;
    uint32_t current_buffer_index;

    std::array<struct aiocb, 2> cbs{};
    std::array<char*, 2> buffer{};
    std::array<bool, 2> cbs_active{};
    std::array<uint64_t, 2> buffer_kmer_count{};
    std::array<uint64_t, 2> buffer_cursor{};

public:
    ExportReader() = delete;
    ExportReader(const ExportReader&) = delete;
    ExportReader& operator=(const ExportReader&) = delete;
    ExportReader(ExportReader&&) = delete;
    ExportReader& operator=(ExportReader&&) = delete;

    explicit ExportReader(uint32_t in_k)
        : fd(-1),
        k(in_k),
        full_data_count(k / BASES_PER_U64T),
        tail_bits(2 * (k % BASES_PER_U64T)),
        tail_bytes((tail_bits + 7) / 8),
        packed_kmer_bytes(full_data_count * sizeof(uint64_t) + tail_bytes),
        buffer_kmer_capacity(packed_kmer_bytes == 0 ? 0 : RAW_BUFFER_SIZE / packed_kmer_bytes),
        buffer_bytes(buffer_kmer_capacity* packed_kmer_bytes),
        kmer_amount(0),
        scheduled_kmer_count(0),
        delivered_kmer_count(0),
        current_buffer_index(0)
    {
        if (k == 0 || k > N * BASES_PER_U64T ||
            packed_kmer_bytes == 0 || buffer_kmer_capacity == 0) [[unlikely]]
        {
            std::cerr << "Invalid k-mer length for ExportReader: " << k << std::endl;
            std::exit(-1);
        }

        for (uint32_t i = 0; i < 2; ++i)
        {
            std::memset(&cbs[i], 0, sizeof(struct aiocb));
        }
    }

    ~ExportReader()
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
        if (file_size % packed_kmer_bytes != 0) [[unlikely]]
        {
            std::cerr << "Invalid low.bin size: " << file_size
                << " is not divisible by packed k-mer size "
                << packed_kmer_bytes << std::endl;
            std::exit(-1);
        }

        ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

        kmer_amount = file_size / packed_kmer_bytes;
        scheduled_kmer_count = 0;
        delivered_kmer_count = 0;
        current_buffer_index = 0;

        for (uint32_t i = 0; i < 2; ++i)
        {
            std::memset(&cbs[i], 0, sizeof(struct aiocb));
            if (buffer[i] == nullptr)
            {
                buffer[i] = new char[buffer_bytes];
            }
            cbs_active[i] = false;
            buffer_kmer_count[i] = 0;
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
            buffer_kmer_count[i] = 0;
            buffer_cursor[i] = 0;
            std::memset(&cbs[i], 0, sizeof(struct aiocb));
        }

        kmer_amount = 0;
        scheduled_kmer_count = 0;
        delivered_kmer_count = 0;
        current_buffer_index = 0;
    }

    uint64_t get_kmer_amount() const
    {
        return kmer_amount;
    }

    bool finished() const noexcept
    {
        return delivered_kmer_count >= kmer_amount;
    }

    uint64_t read_packed_kmers(char* out_buffer, uint64_t max_kmers_to_read)
    {
        {
            if (finished()) [[unlikely]]
            {
                return 0;
            }

            uint64_t total_read = 0;
            while (total_read < max_kmers_to_read && !finished())
            {
                uint32_t idx = current_buffer_index;

                if (cbs_active[idx]) [[likely]]
                {
                    wait_read_finish(idx);
                    async_read(1 - idx);
                }

                if (buffer_cursor[idx] >= buffer_kmer_count[idx])
                {
                    current_buffer_index = 1 - idx;
                    continue;
                }

                const uint64_t available = buffer_kmer_count[idx] - buffer_cursor[idx];
                if (available == 0) [[unlikely]]
                {
                    current_buffer_index = 1 - idx;
                    continue;
                }

                const uint64_t to_copy = std::min(max_kmers_to_read - total_read, available);
                const char* src = buffer[idx] + buffer_cursor[idx] * packed_kmer_bytes;
                std::memcpy(
                    out_buffer + total_read * packed_kmer_bytes,
                    src,
                    static_cast<size_t>(to_copy * packed_kmer_bytes));

                buffer_cursor[idx] += to_copy;
                delivered_kmer_count += to_copy;
                total_read += to_copy;

                if (buffer_cursor[idx] >= buffer_kmer_count[idx])
                {
                    buffer_kmer_count[idx] = 0;
                    buffer_cursor[idx] = 0;
                    current_buffer_index = 1 - idx;
                }
            }

            return total_read;
        }
    }


    uint64_t read_kmers(kmer<N>* out_buffer, uint64_t max_kmers_to_read)
    {
        if (finished()) [[unlikely]]
        {
            return 0;
        }

        uint64_t total_read = 0;
        while (total_read < max_kmers_to_read && !finished())
        {
            uint32_t idx = current_buffer_index;

            if (cbs_active[idx]) [[likely]]
            {
                wait_read_finish(idx);
                async_read(1 - idx);
            }

            if (buffer_cursor[idx] >= buffer_kmer_count[idx])
            {
                current_buffer_index = 1 - idx;
                continue;
            }

            const uint64_t available = buffer_kmer_count[idx] - buffer_cursor[idx];
            if (available == 0) [[unlikely]]
            {
                current_buffer_index = 1 - idx;
                continue;
            }

            const uint64_t to_copy = std::min(max_kmers_to_read - total_read, available);
            const char* src = buffer[idx] + buffer_cursor[idx] * packed_kmer_bytes;
            unpack_kmers_from_buffer(src, out_buffer + total_read, to_copy);

            buffer_cursor[idx] += to_copy;
            delivered_kmer_count += to_copy;
            total_read += to_copy;

            if (buffer_cursor[idx] >= buffer_kmer_count[idx])
            {
                buffer_kmer_count[idx] = 0;
                buffer_cursor[idx] = 0;
                current_buffer_index = 1 - idx;
            }
        }

        return total_read;
    }

private:

    void async_read(uint32_t buffer_index)
    {
        if (scheduled_kmer_count >= kmer_amount || cbs_active[buffer_index]) [[unlikely]]
        {
            return;
        }

        const uint64_t remaining = kmer_amount - scheduled_kmer_count;
        const uint64_t read_count = std::min(buffer_kmer_capacity, remaining);
        if (read_count == 0)
        {
            return;
        }

        const uint64_t read_bytes = read_count * packed_kmer_bytes;
        const uint64_t read_offset = scheduled_kmer_count * packed_kmer_bytes;

        std::memset(&cbs[buffer_index], 0, sizeof(struct aiocb));
        cbs[buffer_index].aio_fildes = fd;
        cbs[buffer_index].aio_buf = buffer[buffer_index];
        cbs[buffer_index].aio_nbytes = static_cast<size_t>(read_bytes);
        cbs[buffer_index].aio_offset = static_cast<off_t>(read_offset);

        buffer_kmer_count[buffer_index] = read_count;
        buffer_cursor[buffer_index] = 0;

        if (::aio_read(&cbs[buffer_index]) != 0) [[unlikely]]
        {
            std::cerr << "aio_read failed" << std::endl;
            std::exit(-1);
        }

        scheduled_kmer_count += read_count;
        cbs_active[buffer_index] = true;
    }

    void wait_read_finish(uint32_t buffer_index)
    {
        if (!cbs_active[buffer_index])
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
            std::cerr << "aio_read error" << std::endl;
            std::exit(-1);
        }

        const uint64_t expected_bytes = buffer_kmer_count[buffer_index] * packed_kmer_bytes;
        const ssize_t n = ::aio_return(&cbs[buffer_index]);
        if (n != static_cast<ssize_t>(expected_bytes)) [[unlikely]]
        {
            std::cerr << "partial aio_read" << std::endl;
            std::exit(-1);
        }

        cbs_active[buffer_index] = false;
    }

    void unpack_kmers_from_buffer(const char* src, kmer<N>* output, uint64_t count) const
    {
        // std::memset(output, 0, sizeof(kmer<N>) * count);

        for (uint64_t i = 0; i < count; ++i)
        {
            const char* record = src + i * packed_kmer_bytes;
            const uint64_t full_bytes = full_data_count * sizeof(uint64_t);
            if (full_bytes > 0)
            {
                std::memcpy(output[i].data.data(), record, static_cast<size_t>(full_bytes));
                record += full_bytes;
            }

            if (tail_bytes > 0)
            {
                uint64_t tail_data = 0;

                std::memcpy(
                    reinterpret_cast<char*>(&tail_data) + (sizeof(uint64_t) - tail_bytes),
                    record,
                    static_cast<size_t>(tail_bytes));

                output[i].data[full_data_count] = tail_data;
                for (uint32_t j = full_data_count + 1; j < N; j++)
                {
                    output[i].data[j] = 0;
                }
            }
            else {
                for (uint32_t j = full_data_count; j < N; j++)
                {
                    output[i].data[j] = 0;
                }
            }
        }
    }
};

#endif
