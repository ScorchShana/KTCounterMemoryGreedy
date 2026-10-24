#ifndef PARTITION_HEADER
#define PARTITION_HEADER

#include "../src/definition.h"
#include "../tool/LegacyKmer.h"
#include "KtpInput.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
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

    // 按 key 的前 P 位取分区号（key 按字节大端解释，与 dump 的排序一致）
    inline uint64_t extract_prefix(const uint8_t* key, uint32_t bits) noexcept
    {
        uint64_t value = 0;
        for (uint32_t i = 0; i < bits; ++i)
            value = (value << 1) | ((key[i >> 3] >> (7 - (i & 7))) & 1u);
        return value;
    }

    // 顺序遍历输入记录；KTP1 走解码器，旧格式走 mmap。
    template <typename Fn>
    inline uint64_t for_each_record(const std::string& path, uint32_t k_len,
                                    uint32_t count_bytes, Fn&& fn)
    {
        const uint32_t key_bytes = ktp::key_bytes_for(k_len);
        if (ktp::is_ktp_file(path))
        {
            ktp::Reader reader;
            reader.open(path);
            if (reader.k() != k_len)
                throw std::runtime_error("partition: KTP1 k mismatch " + path);
            uint64_t count = 0;
            reader.decode_all([&](const uint8_t* key, uint32_t value) {
                fn(key, value);
                ++count;
            });
            return count;
        }

        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("partition: failed to open " + path);
        struct stat st{};
        if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("partition: stat failed " + path); }
        const uint64_t file_size = static_cast<uint64_t>(st.st_size);
        const uint64_t record_bytes = key_bytes + count_bytes;
        if (record_bytes == 0 || file_size % record_bytes)
        {
            ::close(fd);
            throw std::runtime_error("partition: misaligned input " + path);
        }
        const char* mapped = static_cast<const char*>(
            ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
        ::close(fd);
        if (mapped == MAP_FAILED)
            throw std::runtime_error("partition: mmap failed " + path);

        const uint64_t records = file_size / record_bytes;
        for (uint64_t i = 0; i < records; ++i)
        {
            const uint8_t* record = reinterpret_cast<const uint8_t*>(mapped) + i * record_bytes;
            uint32_t value = 1;
            if (count_bytes > 0)
            {
                value = 0;
                for (uint32_t b = 0; b < count_bytes; ++b)
                    value |= uint32_t{record[key_bytes + b]} << (8 * b);
            }
            fn(record, value);
        }
        ::munmap(const_cast<char*>(mapped), file_size);
        return records;
    }

    // Partition high.bin into 2^P sub-files based on k-mer prefix.
    // Returns a vector of record counts per partition.
    inline std::vector<uint64_t> partition_high_file(const std::string& tmp_dir,
                                                     uint32_t P,
                                                     uint32_t k_len,
                                                     uint32_t count_bytes)
    {
        const uint64_t num_partitions = 1ULL << P;
        const uint32_t key_bytes = ktp::key_bytes_for(k_len);
        const uint32_t prefix_bits = std::min<uint32_t>(P, 8u * key_bytes);

        const std::string high_path = tmp_dir + "high.bin";
        std::vector<PartitionWriter> writers(num_partitions);
        std::vector<uint64_t> record_counts(num_partitions, 0);

        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            const std::string part_dir = tmp_dir + "partition_" + std::to_string(i);
            ::mkdir(part_dir.c_str(), 0755);
            const std::string part_high = part_dir + "/high.bin";
            if (!writers[i].open(part_high))
                throw std::runtime_error("partition: failed to open " + part_high);
        }

        std::cout << "partitioning high.bin into " << num_partitions
                  << " buckets (P=" << P << ", prefix_bits=" << prefix_bits << ")..."
                  << std::endl;

        const uint64_t total = for_each_record(high_path, k_len, count_bytes,
            [&](const uint8_t* key, uint32_t count) {
                const uint64_t prefix = extract_prefix(key, prefix_bits) << (P - prefix_bits);
                char record[64];
                std::memcpy(record, key, key_bytes);
                for (uint32_t b = 0; b < count_bytes; ++b)
                    record[key_bytes + b] = static_cast<char>((count >> (8 * b)) & 0xffu);
                writers[prefix].write(record, key_bytes + count_bytes);
                ++record_counts[prefix];
            });

        for (auto& w : writers) w.flush();
        std::cout << "partitioned high.bin: " << total << " records" << std::endl;
        return record_counts;
    }

    // Partition low.bin into 2^P sub-files based on k-mer prefix.
    // Returns a vector of record counts per partition.
    inline std::vector<uint64_t> partition_low_file(const std::string& tmp_dir,
                                                    uint32_t P,
                                                    uint32_t k_len)
    {
        const uint64_t num_partitions = 1ULL << P;
        const uint32_t key_bytes = ktp::key_bytes_for(k_len);
        const uint32_t prefix_bits = std::min<uint32_t>(P, 8u * key_bytes);

        const std::string low_path = tmp_dir + "low.bin";
        std::vector<PartitionWriter> writers(num_partitions);
        std::vector<uint64_t> record_counts(num_partitions, 0);

        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            const std::string part_dir = tmp_dir + "partition_" + std::to_string(i);
            ::mkdir(part_dir.c_str(), 0755);
            const std::string part_low = part_dir + "/low.bin";
            if (!writers[i].open(part_low))
                throw std::runtime_error("partition: failed to open " + part_low);
        }

        std::cout << "partitioning low.bin into " << num_partitions
                  << " buckets (P=" << P << ", prefix_bits=" << prefix_bits << ")..."
                  << std::endl;

        const uint64_t total = for_each_record(low_path, k_len, 0,
            [&](const uint8_t* key, uint32_t) {
                const uint64_t prefix = extract_prefix(key, prefix_bits) << (P - prefix_bits);
                writers[prefix].write(reinterpret_cast<const char*>(key), key_bytes);
                ++record_counts[prefix];
            });

        for (auto& w : writers) w.flush();
        std::cout << "partitioned low.bin: " << total << " records" << std::endl;
        return record_counts;
    }

} // namespace partition

#endif
