#ifndef CONCURRENT_MAP_WRITER_HEADER
#define CONCURRENT_MAP_WRITER_HEADER

#include "definition.h"

#include <string>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

class ConcurrentMapWriter
{
private:
    int out_file = -1;
    std::vector<char> buffer;
    int buffer_offset = 0;
    uint64_t local_sorted_kmer_count = 0;

public:
    ConcurrentMapWriter(const ConcurrentMapWriter&) = delete;
    ConcurrentMapWriter& operator=(const ConcurrentMapWriter&) = delete;
    ConcurrentMapWriter(ConcurrentMapWriter&&) = delete;
    ConcurrentMapWriter& operator=(ConcurrentMapWriter&&) = delete;

    explicit ConcurrentMapWriter() : out_file(-1), buffer(DRAIN_EXPORT_BUFFER_SIZE), buffer_offset(0), local_sorted_kmer_count(0) {}

    void open(const uint32_t thread_id)
    {
        buffer_offset = 0;
        std::string file_name = temp_dir + "thread_" + std::to_string(thread_id) + ".bin";
        out_file = ::open(file_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (out_file < 0)
        {
            std::cerr << "failed to open concurrent map output file" << std::endl;
            std::exit(-1);
        }
    }

    ~ConcurrentMapWriter()
    {
        if (out_file != -1)
        {
            ::close(out_file);
            out_file = -1;
        }
        if (local_sorted_kmer_count > 0)
        {
            sorted_kmer_count.fetch_add(local_sorted_kmer_count, std::memory_order_relaxed);
            local_sorted_kmer_count = 0;
        }
    }

    void close()
    {
        if (out_file != -1)
        {
            if (buffer_offset > 0)
            {
                ::write(out_file, buffer.data(), buffer_offset);
                buffer_offset = 0;
            }
            ::close(out_file);
            out_file = -1;
        }
        sorted_kmer_count.fetch_add(local_sorted_kmer_count, std::memory_order_relaxed);
        local_sorted_kmer_count = 0;
    }

    void write_kmer_record(const uint64_t* kmer_data, uint64_t full_words,
        uint64_t tail_bits, uint32_t count)
    {
        const uint64_t tail_bytes = (tail_bits + 7) / 8;
        const uint64_t kmer_bytes = full_words * sizeof(uint64_t) + tail_bytes;
        const uint32_t total = static_cast<uint32_t>(kmer_bytes + sizeof(uint32_t));

        local_sorted_kmer_count++;
        if (total + buffer_offset > DRAIN_EXPORT_BUFFER_SIZE) [[unlikely]]
        {
            if (!write_all(buffer.data(), buffer_offset)) [[unlikely]]
            {
                std::cerr << "Failed to write k-mer data" << std::endl;
                std::exit(-1);
            }
            buffer_offset = 0;
        }

        // Write full uint64_t words
        std::memcpy(buffer.data() + buffer_offset, kmer_data,
            full_words * sizeof(uint64_t));
        buffer_offset += static_cast<int>(full_words * sizeof(uint64_t));

        // Write tail bytes 
        if (tail_bytes > 0)
        {
            uint64_t mask = (~uint64_t{ 0 }) << (64 - tail_bits);
            uint64_t tail_data = kmer_data[full_words] & mask;
            std::memcpy(buffer.data() + buffer_offset,
                reinterpret_cast<const char*>(&tail_data) + (8 - tail_bytes),
                tail_bytes);
            buffer_offset += static_cast<int>(tail_bytes);
        }

        // Write 32-bit count
        std::memcpy(buffer.data() + buffer_offset, &count, sizeof(uint32_t));
        buffer_offset += static_cast<int>(sizeof(uint32_t));
    }


private:
    bool write_all(void* data, size_t count) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        size_t left = count;

        while (left > 0) {
            ssize_t n = ::write(out_file, p, left);

            if (n > 0) [[unlikely]]
            {
                p += n;
                left -= static_cast<size_t>(n);
                continue;
            }
            else if (n < 0 && errno == EINTR) [[unlikely]]
            {
                continue;
            }

            return false;
        }

        return true;
    }
};

#endif