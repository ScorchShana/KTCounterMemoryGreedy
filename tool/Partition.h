#ifndef PARTITION_HEADER
#define PARTITION_HEADER

// Splits high.bin / low.bin into 2^P bucket directories so that every bucket
// fits the memory budget of dump_tool.
//
// Input files use the group layout described in MsdGroupReader.h. Buckets are
// routed by the top P bits of the packed key; the rewritten files use the same
// group layout, with the bucket's whole byte prefix stored once per group.

#include "../src/definition.h"
#include "../tool/LegacyKmer.h"
#include "MsdGroupReader.h"

#include <algorithm>
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
    constexpr uint64_t PARTITION_GROUP_PAYLOAD_BYTES = 64ULL * 1024;

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

    // Top `bits` bits of a packed key (first base in the MSB of byte 0).
    // Bits past the meaningful key bytes must be zero padded by the caller.
    inline uint64_t extract_prefix_bits(const uint8_t* packed_key, uint32_t bits) noexcept
    {
        uint64_t value = 0;
        for (uint32_t i = 0; i < bits; ++i)
            value = (value << 1) | ((packed_key[i >> 3] >> (7 - (i & 7))) & 1u);
        return value;
    }

    // Accumulates whole groups of one bucket. Every record in a bucket shares
    // the bucket prefix, so the prefix is stored once per group.
    struct GroupWriter
    {
        PartitionWriter out;
        std::vector<uint8_t> payload;
        std::vector<uint8_t> prefix;
        uint32_t records = 0;
        uint32_t count_bytes = 0;
        uint8_t prefix_bytes = 0;

        bool open(const std::string& path, uint32_t in_count_bytes, uint8_t in_prefix_bytes)
        {
            count_bytes = in_count_bytes;
            prefix_bytes = in_prefix_bytes;
            return out.open(path);
        }

        void add(const uint8_t* key, uint32_t key_bytes, uint32_t count)
        {
            const size_t record_bytes = size_t(key_bytes - prefix_bytes) + count_bytes;
            if (records != 0 && payload.size() + record_bytes > PARTITION_GROUP_PAYLOAD_BYTES)
                flush();
            if (records == 0)
            {
                prefix.assign(key, key + prefix_bytes);
                payload.reserve(PARTITION_GROUP_PAYLOAD_BYTES);
            }
            else if (prefix_bytes != 0 &&
                     std::memcmp(prefix.data(), key, prefix_bytes) != 0)
            {
                std::cerr << "partition: bucket prefix mismatch" << std::endl;
                std::exit(-1);
            }
            payload.insert(payload.end(), key + prefix_bytes, key + key_bytes);
            for (uint32_t i = 0; i < count_bytes; ++i)
                payload.push_back(static_cast<uint8_t>(count >> (8 * i)));
            ++records;
        }

        void flush()
        {
            if (records == 0)
                return;
            uint8_t header[msd_group::GROUP_HEADER_BYTES];
            header[0] = prefix_bytes;
            std::memcpy(header + 1, &records, sizeof(records));
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            if (!prefix.empty())
                out.write(reinterpret_cast<const char*>(prefix.data()), prefix.size());
            out.write(reinterpret_cast<const char*>(payload.data()), payload.size());
            payload.clear();
            prefix.clear();
            records = 0;
        }
    };

    inline void create_partition_dirs(const std::string& tmp_dir, uint64_t num_partitions)
    {
        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            const std::string part_dir = tmp_dir + "partition_" + std::to_string(i);
            // Create directory (ignore error if it already exists)
            ::mkdir(part_dir.c_str(), 0755);
        }
    }

    // Partition high.bin into 2^P sub-files based on the k-mer prefix bits.
    // Returns a vector of record counts per partition.
    template <uint32_t N>
    std::vector<uint64_t> partition_high_file(const std::string& tmp_dir,
                                              uint32_t P,
                                              uint32_t k_len,
                                              uint32_t count_bytes = sizeof(uint32_t))
    {
        static_assert(N == 1 || N == 2 || N == 4, "unsupported k-mer width");
        const uint64_t num_partitions = 1ULL << P;
        msd_group::File source = msd_group::open_file(tmp_dir + "high.bin", k_len, count_bytes);
        const uint32_t key_bytes = source.key_bytes;
        const uint8_t group_prefix =
            static_cast<uint8_t>(std::min<uint64_t>(P / 8, key_bytes - 1));

        create_partition_dirs(tmp_dir, num_partitions);
        std::vector<GroupWriter> writers(num_partitions);
        std::vector<uint64_t> record_counts(num_partitions, 0);
        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            const std::string part_high =
                tmp_dir + "partition_" + std::to_string(i) + "/high.bin";
            if (!writers[i].open(part_high, count_bytes, group_prefix))
                std::exit(-1);
        }

        std::cout << "partitioning high.bin into " << num_partitions
                  << " buckets (P=" << P << "), " << source.records << " records..."
                  << std::endl;

        std::vector<char> group;
        uint8_t key[msd_group::MAX_KEY_BYTES];
        for (size_t index = 0; index < source.groups.size(); ++index)
        {
            msd_group::read_group(source, index, group);
            const char* data = group.data();
            const uint32_t records = msd_group::group_records(data);
            for (uint32_t i = 0; i < records; ++i)
            {
                std::memset(key, 0, sizeof(key));
                msd_group::expand_key(source, data, i, key);
                const uint64_t bucket = extract_prefix_bits(key, P);
                writers[bucket].add(key, key_bytes, msd_group::group_count_value(source, data, i));
                ++record_counts[bucket];
            }
        }
        for (auto& writer : writers)
            writer.flush();

        std::cout << "partitioned high.bin: " << source.records << " records" << std::endl;
        return record_counts;
    }

    // Partition low.bin into 2^P sub-files based on the k-mer prefix bits.
    // Returns a vector of k-mer counts per partition.
    template <uint32_t N>
    std::vector<uint64_t> partition_low_file(const std::string& tmp_dir,
                                             uint32_t P,
                                             uint32_t k_len)
    {
        static_assert(N == 1 || N == 2 || N == 4, "unsupported k-mer width");
        const uint64_t num_partitions = 1ULL << P;
        msd_group::File source = msd_group::open_file(tmp_dir + "low.bin", k_len, 0);
        const uint32_t key_bytes = source.key_bytes;
        const uint8_t group_prefix =
            static_cast<uint8_t>(std::min<uint64_t>(P / 8, key_bytes - 1));

        create_partition_dirs(tmp_dir, num_partitions);
        std::vector<GroupWriter> writers(num_partitions);
        std::vector<uint64_t> kmer_counts(num_partitions, 0);
        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            const std::string part_low =
                tmp_dir + "partition_" + std::to_string(i) + "/low.bin";
            if (!writers[i].open(part_low, 0, group_prefix))
                std::exit(-1);
        }

        std::cout << "partitioning low.bin into " << num_partitions
                  << " buckets (P=" << P << "), " << source.records << " k-mers..."
                  << std::endl;

        std::vector<char> group;
        uint8_t key[msd_group::MAX_KEY_BYTES];
        for (size_t index = 0; index < source.groups.size(); ++index)
        {
            msd_group::read_group(source, index, group);
            const char* data = group.data();
            const uint32_t records = msd_group::group_records(data);
            for (uint32_t i = 0; i < records; ++i)
            {
                std::memset(key, 0, sizeof(key));
                msd_group::expand_key(source, data, i, key);
                const uint64_t bucket = extract_prefix_bits(key, P);
                writers[bucket].add(key, key_bytes, 1);
                ++kmer_counts[bucket];
            }
        }
        for (auto& writer : writers)
            writer.flush();

        std::cout << "partitioned low.bin: " << source.records << " k-mers" << std::endl;
        return kmer_counts;
    }

} // namespace partition

#endif
