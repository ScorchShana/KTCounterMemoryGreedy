#ifndef PARTITION_HEADER
#define PARTITION_HEADER

#include "../src/definition.h"
#include "../src/kmer.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace partition
{

    constexpr uint64_t PARTITION_WRITER_BUFFER_SIZE = 128ULL * 1024;

    struct PartitionWriter
    {
        int fd = -1;
        std::vector<char> buffer;
        size_t cursor = 0;

        PartitionWriter()
        {
            buffer.resize(PARTITION_WRITER_BUFFER_SIZE);
        }

        ~PartitionWriter()
        {
            flush();
            if (fd >= 0)
            {
                ::close(fd);
                fd = -1;
            }
        }

        PartitionWriter(const PartitionWriter&) = delete;
        PartitionWriter& operator=(const PartitionWriter&) = delete;
        PartitionWriter(PartitionWriter&& other) noexcept
            : fd(other.fd), buffer(std::move(other.buffer)), cursor(other.cursor)
        {
            other.fd = -1;
            other.cursor = 0;
        }
        PartitionWriter& operator=(PartitionWriter&& other) noexcept
        {
            if (this != &other)
            {
                flush();
                if (fd >= 0)
                    ::close(fd);
                fd = other.fd;
                buffer = std::move(other.buffer);
                cursor = other.cursor;
                other.fd = -1;
                other.cursor = 0;
            }
            return *this;
        }

        bool open(const std::string& path)
        {
            fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0)
            {
                std::cerr << "PartitionWriter: failed to open " << path
                          << ": " << std::strerror(errno) << std::endl;
                return false;
            }
            return true;
        }

        void write(const char* data, size_t len)
        {
            if (cursor + len > PARTITION_WRITER_BUFFER_SIZE)
            {
                flush();
            }
            if (len <= PARTITION_WRITER_BUFFER_SIZE - cursor)
            {
                std::memcpy(buffer.data() + cursor, data, len);
                cursor += len;
            }
            else
            {
                // Large write: flush first, then write directly
                flush();
                size_t written = 0;
                while (written < len)
                {
                    ssize_t n = ::write(fd, data + written, len - written);
                    if (n < 0)
                    {
                        std::cerr << "PartitionWriter: write error: "
                                  << std::strerror(errno) << std::endl;
                        std::exit(-1);
                    }
                    written += static_cast<size_t>(n);
                }
            }
        }

        void flush()
        {
            if (cursor > 0)
            {
                size_t written = 0;
                while (written < cursor)
                {
                    ssize_t n = ::write(fd, buffer.data() + written, cursor - written);
                    if (n < 0)
                    {
                        std::cerr << "PartitionWriter: flush error: "
                                  << std::strerror(errno) << std::endl;
                        std::exit(-1);
                    }
                    written += static_cast<size_t>(n);
                }
                cursor = 0;
            }
        }
    };

    // Extract the high P-base prefix from a packed k-mer byte sequence.
    // Returns a value in [0, 2^P - 1].
    // The packed format stores data[0] in little-endian in the first few bytes,
    // with bases left-aligned (first base = highest 2 bits of data[0]).
    inline uint64_t extract_prefix_from_packed(const char* packed_kmer,
                                               uint32_t P,
                                               uint32_t k_len)
    {
        const uint32_t full_words = k_len / BASES_PER_U64T;
        const uint32_t tail_bits = 2 * (k_len % BASES_PER_U64T);
        const uint32_t tail_bytes = (tail_bits + 7) / 8;

        uint64_t data0;
        if (full_words >= 1)
        {
            // data[0] is fully stored as the first 8 bytes (little-endian)
            std::memcpy(&data0, packed_kmer, sizeof(uint64_t));
        }
        else
        {
            // k_len < 32: data stored MSB-aligned in tail_bytes
            data0 = 0;
            std::memcpy(reinterpret_cast<char*>(&data0) + (8 - tail_bytes),
                        packed_kmer, tail_bytes);
        }

        return data0 >> (64 - P);
    }

    // Partition high.bin into 2^P sub-files based on k-mer prefix.
    // Returns a vector of record counts per partition.
    template <uint32_t N>
    std::vector<uint64_t> partition_high_file(const std::string& tmp_dir,
                                              uint32_t P,
                                              uint32_t k_len)
    {
        const uint64_t num_partitions = 1ULL << P;
        const uint32_t full_words = k_len / BASES_PER_U64T;
        const uint32_t tail_bits = 2 * (k_len % BASES_PER_U64T);
        const uint32_t tail_bytes = (tail_bits + 7) / 8;
        const uint64_t kmer_bytes = full_words * sizeof(uint64_t) + tail_bytes;
        const uint64_t compact_rec_size = kmer_bytes + sizeof(uint32_t);

        const std::string high_path = tmp_dir + "high.bin";

        int fd = ::open(high_path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "partition_high_file: failed to open " << high_path
                      << ": " << std::strerror(errno) << std::endl;
            std::exit(-1);
        }

        struct stat st {};
        if (::fstat(fd, &st) != 0)
        {
            std::cerr << "partition_high_file: failed to stat " << high_path << std::endl;
            ::close(fd);
            std::exit(-1);
        }
        const uint64_t file_size = static_cast<uint64_t>(st.st_size);
        const uint64_t record_count = file_size / compact_rec_size;

        // mmap the input file
        const char* mapped = static_cast<const char*>(
            ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
        ::close(fd);
        if (mapped == MAP_FAILED)
        {
            std::cerr << "partition_high_file: mmap failed" << std::endl;
            std::exit(-1);
        }

        // Create partition directories and writers
        std::vector<PartitionWriter> writers(num_partitions);
        std::vector<uint64_t> record_counts(num_partitions, 0);

        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            const std::string part_dir = tmp_dir + "partition_" + std::to_string(i);
            // Create directory (ignore error if it already exists)
            ::mkdir(part_dir.c_str(), 0755);
            const std::string part_high = part_dir + "/high.bin";
            if (!writers[i].open(part_high))
            {
                ::munmap(const_cast<char*>(mapped), file_size);
                std::exit(-1);
            }
        }

        std::cout << "partitioning high.bin into " << num_partitions
                  << " buckets (P=" << P << "), " << record_count << " records..."
                  << std::endl;

        for (uint64_t i = 0; i < record_count; ++i)
        {
            const char* record = mapped + i * compact_rec_size;
            const uint64_t prefix = extract_prefix_from_packed(record, P, k_len);
            writers[prefix].write(record, compact_rec_size);
            record_counts[prefix]++;

            if ((i & 0xFFFFF) == 0xFFFFF)
            {
                std::cout << "\rpartitioning high.bin: "
                          << (i * 100 / record_count) << "%" << std::flush;
            }
        }
        std::cout << "\rpartitioning high.bin: 100%" << std::endl;

        // Flush and close all writers
        for (auto& w : writers)
        {
            w.flush();
            if (w.fd >= 0)
            {
                ::close(w.fd);
                w.fd = -1;
            }
        }

        ::munmap(const_cast<char*>(mapped), file_size);
        return record_counts;
    }

    // Partition low.bin into 2^P sub-files based on k-mer prefix.
    // Returns a vector of k-mer counts per partition.
    template <uint32_t N>
    std::vector<uint64_t> partition_low_file(const std::string& tmp_dir,
                                             uint32_t P,
                                             uint32_t k_len)
    {
        const uint64_t num_partitions = 1ULL << P;
        const uint32_t full_words = k_len / BASES_PER_U64T;
        const uint32_t tail_bits = 2 * (k_len % BASES_PER_U64T);
        const uint32_t tail_bytes = (tail_bits + 7) / 8;
        const uint64_t kmer_bytes = full_words * sizeof(uint64_t) + tail_bytes;

        const std::string low_path = tmp_dir + "low.bin";

        int fd = ::open(low_path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "partition_low_file: failed to open " << low_path
                      << ": " << std::strerror(errno) << std::endl;
            std::exit(-1);
        }

        struct stat st {};
        if (::fstat(fd, &st) != 0)
        {
            std::cerr << "partition_low_file: failed to stat " << low_path << std::endl;
            ::close(fd);
            std::exit(-1);
        }
        const uint64_t file_size = static_cast<uint64_t>(st.st_size);
        const uint64_t kmer_count = file_size / kmer_bytes;

        // mmap the input file
        const char* mapped = static_cast<const char*>(
            ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
        ::close(fd);
        if (mapped == MAP_FAILED)
        {
            std::cerr << "partition_low_file: mmap failed" << std::endl;
            std::exit(-1);
        }

        // Create partition directories and writers
        std::vector<PartitionWriter> writers(num_partitions);
        std::vector<uint64_t> kmer_counts(num_partitions, 0);

        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            const std::string part_dir = tmp_dir + "partition_" + std::to_string(i);
            ::mkdir(part_dir.c_str(), 0755);
            const std::string part_low = part_dir + "/low.bin";
            if (!writers[i].open(part_low))
            {
                ::munmap(const_cast<char*>(mapped), file_size);
                std::exit(-1);
            }
        }

        std::cout << "partitioning low.bin into " << num_partitions
                  << " buckets (P=" << P << "), " << kmer_count << " k-mers..."
                  << std::endl;

        for (uint64_t i = 0; i < kmer_count; ++i)
        {
            const char* kmer = mapped + i * kmer_bytes;
            const uint64_t prefix = extract_prefix_from_packed(kmer, P, k_len);
            writers[prefix].write(kmer, kmer_bytes);
            kmer_counts[prefix]++;

            if ((i & 0xFFFFF) == 0xFFFFF)
            {
                std::cout << "\rpartitioning low.bin: "
                          << (i * 100 / kmer_count) << "%" << std::flush;
            }
        }
        std::cout << "\rpartitioning low.bin: 100%" << std::endl;

        // Flush and close all writers
        for (auto& w : writers)
        {
            w.flush();
            if (w.fd >= 0)
            {
                ::close(w.fd);
                w.fd = -1;
            }
        }

        ::munmap(const_cast<char*>(mapped), file_size);
        return kmer_counts;
    }

} // namespace partition

#endif
