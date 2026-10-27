#ifndef MSD_GROUP_READER_HEADER
#define MSD_GROUP_READER_HEADER

// Reader for the intermediate k-mer files produced by KTCounterMemoryGreedy.
//
// Both files are plain concatenations of groups: no file header, version,
// alignment or checksum. Integers are native (little endian) copies.
//
//   high.bin: u8 P | u32 N | u8 prefix[P] | N * { u8 suffix[W-P]; u8 count[count_bytes] }
//   low.bin : u8 P | u32 N | u8 prefix[P] | N * { u8 suffix[W-P] }
//
// W = (k + 3) / 4 key bytes, two-bit packed: the first base lives in the most
// significant bits of byte 0, the last byte is zero padded. Records inside a
// group may have been reordered by the writer, and every group holds at least
// one record (writers never emit empty groups).
//
// dump_tool and the hash map keep using legacy_kmer<N>, so records are expanded
// into that word layout while decoding.

#include "../src/definition.h"
#include "LegacyKmer.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace msd_group
{
    inline constexpr size_t GROUP_HEADER_BYTES = 5;
    inline constexpr uint32_t MAX_KEY_BYTES = (MAX_K + 3) / 4;

    [[noreturn]] inline void fail(const std::string& message)
    {
        std::fprintf(stderr, "%s\n", message.c_str());
        std::exit(-1);
    }

    inline void read_exact_at(int fd, void* buffer, size_t bytes, uint64_t offset,
                              const std::string& path)
    {
        auto* out = static_cast<uint8_t*>(buffer);
        while (bytes != 0)
        {
            const ssize_t n = ::pread(fd, out, bytes, static_cast<off_t>(offset));
            if (n < 0)
            {
                if (errno == EINTR) continue;
                fail("failed to read " + path + ": " + std::strerror(errno));
            }
            if (n == 0) fail("truncated intermediate file: " + path);
            out += n;
            bytes -= static_cast<size_t>(n);
            offset += static_cast<uint64_t>(n);
        }
    }

    struct Group
    {
        uint64_t offset = 0;        // byte offset of the group header
        uint64_t first_record = 0;  // index of this group's first record
        uint32_t records = 0;
        uint8_t prefix_bytes = 0;
    };

    struct File
    {
        std::string path;
        int fd = -1;
        uint32_t k_len = 0;
        uint32_t key_bytes = 0;
        uint32_t count_bytes = 0; // 0 for low.bin
        uint64_t size = 0;
        uint64_t records = 0;
        std::vector<Group> groups;

        File() = default;
        File(const File&) = delete;
        File& operator=(const File&) = delete;

        File(File&& other) noexcept { move_from(other); }
        File& operator=(File&& other) noexcept
        {
            if (this != &other)
            {
                if (fd >= 0) ::close(fd);
                move_from(other);
            }
            return *this;
        }

        ~File()
        {
            if (fd >= 0) ::close(fd);
        }

        template <uint32_t N>
        uint64_t read_records(uint64_t first, uint64_t count, ExportRecord<N>* out,
                              std::vector<char>& scratch, uint64_t* bytes_read = nullptr) const;

        // Same, but writes flat records: key bytes (prefix + suffix) followed by
        // the count bytes. That is the layout the histogram worker pools expect.
        uint64_t read_flat_records(uint64_t first, uint64_t count, char* out,
                                   std::vector<char>& scratch, uint64_t* bytes_read = nullptr) const;

    private:
        void move_from(File& other) noexcept
        {
            path = std::move(other.path);
            fd = other.fd;
            k_len = other.k_len;
            key_bytes = other.key_bytes;
            count_bytes = other.count_bytes;
            size = other.size;
            records = other.records;
            groups = std::move(other.groups);
            other.fd = -1;
            other.size = 0;
            other.records = 0;
        }
    };

    inline uint64_t group_bytes(const File& file, const Group& group)
    {
        return GROUP_HEADER_BYTES + group.prefix_bytes +
            uint64_t(group.records) * (uint64_t(file.key_bytes - group.prefix_bytes) + file.count_bytes);
    }

    inline uint8_t group_prefix_bytes(const char* group) noexcept
    {
        return static_cast<uint8_t>(group[0]);
    }

    inline uint32_t group_records(const char* group) noexcept
    {
        uint32_t records = 0;
        std::memcpy(&records, group + 1, sizeof(records));
        return records;
    }

    inline const char* group_prefix(const char* group) noexcept
    {
        return group + GROUP_HEADER_BYTES;
    }

    inline const char* group_payload(const char* group) noexcept
    {
        return group + GROUP_HEADER_BYTES + group_prefix_bytes(group);
    }

    inline size_t group_record_bytes(const File& file, uint8_t prefix_bytes) noexcept
    {
        return size_t(file.key_bytes - prefix_bytes) + file.count_bytes;
    }

    inline uint32_t group_count_value(const File& file, const char* group, uint32_t index) noexcept
    {
        if (file.count_bytes == 0) return 1;
        const uint8_t prefix = group_prefix_bytes(group);
        const size_t record_bytes = group_record_bytes(file, prefix);
        uint32_t count = 0;
        std::memcpy(&count, group_payload(group) + size_t(index) * record_bytes + (file.key_bytes - prefix),
                    file.count_bytes);
        return count;
    }

    inline void expand_key(const File& file, const char* group, uint32_t index, uint8_t* key) noexcept
    {
        const uint8_t prefix = group_prefix_bytes(group);
        const size_t record_bytes = group_record_bytes(file, prefix);
        std::memcpy(key, group_prefix(group), prefix);
        std::memcpy(key + prefix, group_payload(group) + size_t(index) * record_bytes,
                    file.key_bytes - prefix);
    }

    // Rebuild the legacy_kmer<N> that dump_tool and the hash map work with.
    //
    // The writer stores each key in the packed layout produced by GetKmer (the
    // window's last base in the most significant bits of byte 0), so the stored
    // bytes are the reverse of the k-mer. kmer_to_buf prints legacy_kmer words
    // in forward order, and the two layouts collapse to a single cheap rule:
    // legacy word w is the big-endian 64-bit load of stored bytes [8w, 8w + 8).
    template <uint32_t N>
    inline void packed_to_legacy(const uint8_t* packed, uint32_t k_len, legacy_kmer<N>& out) noexcept
    {
        static_assert(N == 1 || N == 2 || N == 4, "legacy_kmer width");
        out.reset();
        const uint32_t width = (k_len + 3) / 4;
        const uint32_t words = (k_len + BASES_PER_U64T - 1) / BASES_PER_U64T;
        for (uint32_t w = 0; w < words; ++w)
        {
            const uint32_t offset = w * sizeof(uint64_t);
            const uint32_t available = offset < width
                ? std::min<uint32_t>(sizeof(uint64_t), width - offset) : 0;
            uint64_t value = 0;
            if (available == sizeof(uint64_t))
            {
                std::memcpy(&value, packed + offset, sizeof(value));
            }
            else if (available != 0)
            {
                uint8_t tail[sizeof(uint64_t)]{};
                std::memcpy(tail, packed + offset, available);
                std::memcpy(&value, tail, sizeof(value));
            }
            out.data[w] = __builtin_bswap64(value);
        }
    }

    // Scan only group headers, build the record index and validate the file.
    inline File open_file(const std::string& path, uint32_t k_len, uint32_t count_bytes)
    {
        File file;
        file.path = path;
        file.k_len = k_len;
        file.key_bytes = (k_len + 3) / 4;
        file.count_bytes = count_bytes;
        if (file.key_bytes == 0 || file.key_bytes > MAX_KEY_BYTES)
            fail("invalid k-mer length for intermediate file: " + path);

        file.fd = ::open(path.c_str(), O_RDONLY);
        if (file.fd < 0) fail("failed to open " + path + ": " + std::strerror(errno));

        struct stat st{};
        if (::fstat(file.fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
            fail("failed to stat " + path);
        file.size = static_cast<uint64_t>(st.st_size);
        ::posix_fadvise(file.fd, 0, 0, POSIX_FADV_SEQUENTIAL);

        uint64_t offset = 0;
        constexpr size_t INDEX_WINDOW_BYTES = 1ULL << 20;
        std::vector<char> window;
        uint64_t window_start = 0;
        size_t window_size = 0;
        while (offset < file.size)
        {
            if (offset < window_start || offset + GROUP_HEADER_BYTES > window_start + window_size)
            {
                window_start = offset;
                window_size = static_cast<size_t>(
                    std::min<uint64_t>(INDEX_WINDOW_BYTES, file.size - offset));
                window.resize(window_size);
                read_exact_at(file.fd, window.data(), window_size, offset, path);
            }
            const size_t rel = static_cast<size_t>(offset - window_start);
            if (rel + GROUP_HEADER_BYTES > window_size)
                fail("truncated group header in " + path);
            const uint8_t prefix = static_cast<uint8_t>(window[rel]);
            uint32_t records = 0;
            std::memcpy(&records, window.data() + rel + 1, sizeof(records));
            if (prefix >= file.key_bytes)
                fail("invalid group prefix length in " + path + " at offset " + std::to_string(offset));
            if (records == 0)
                fail("empty group in " + path + " at offset " + std::to_string(offset));
            const uint64_t bytes = GROUP_HEADER_BYTES + prefix +
                uint64_t(records) * (uint64_t(file.key_bytes - prefix) + count_bytes);
            if (bytes > file.size - offset)
                fail("truncated group in " + path + " at offset " + std::to_string(offset));
            file.groups.push_back(Group{offset, file.records, records, prefix});
            file.records += records;
            offset += bytes;
        }
        return file;
    }

    inline void read_group(const File& file, size_t index, std::vector<char>& scratch)
    {
        const Group& group = file.groups[index];
        scratch.resize(static_cast<size_t>(group_bytes(file, group)));
        read_exact_at(file.fd, scratch.data(), scratch.size(), group.offset, file.path);
    }

    // Read one window covering whole groups, starting at group `index`, and
    // return the first group after the window. Windows keep the number of
    // pread calls low when a file holds many small groups.
    inline size_t read_group_window(const File& file, size_t index, std::vector<char>& scratch)
    {
        constexpr uint64_t WINDOW_BYTES = 1ULL << 20;
        const uint64_t start = file.groups[index].offset;
        uint64_t span = 0;
        size_t end = index;
        while (end < file.groups.size())
        {
            const uint64_t bytes = group_bytes(file, file.groups[end]);
            if (span != 0 && span + bytes > WINDOW_BYTES)
                break;
            span += bytes;
            ++end;
            if (span >= WINDOW_BYTES)
                break;
        }
        scratch.resize(static_cast<size_t>(span));
        read_exact_at(file.fd, scratch.data(), scratch.size(), start, file.path);
        return end;
    }

    template <uint32_t N>
    uint64_t File::read_records(uint64_t first, uint64_t count, ExportRecord<N>* out,
                                std::vector<char>& scratch, uint64_t* bytes_read) const
    {
        if (count == 0 || first >= records) return 0;
        count = std::min(count, records - first);

        const auto it = std::upper_bound(groups.begin(), groups.end(), first,
            [](uint64_t value, const Group& group) { return value < group.first_record; });
        size_t index = static_cast<size_t>(it - groups.begin());
        if (index != 0) --index;

        uint64_t emitted = 0;
        while (emitted < count)
        {
            if (index >= groups.size()) fail("record index outside group table of " + path);
            const size_t window_end = read_group_window(*this, index, scratch);
            size_t pos = 0;
            uint8_t key[MAX_KEY_BYTES];
            while (index < window_end && emitted < count)
            {
                const Group& group = groups[index];
                const char* data = scratch.data() + pos;
                const uint8_t prefix = group_prefix_bytes(data);
                const size_t record_bytes = group_record_bytes(*this, prefix);
                const uint64_t wanted = first + emitted;
                const uint64_t skip = wanted > group.first_record ? wanted - group.first_record : 0;
                if (skip >= group.records)
                {
                    pos += static_cast<size_t>(group_bytes(*this, group));
                    ++index;
                    continue;
                }

                const uint64_t take = std::min<uint64_t>(count - emitted, group.records - skip);
                for (uint64_t i = 0; i < take; ++i)
                {
                    const uint32_t record = static_cast<uint32_t>(skip + i);
                    std::memcpy(key, group_prefix(data), prefix);
                    std::memcpy(key + prefix, group_payload(data) + size_t(record) * record_bytes,
                                key_bytes - prefix);
                    packed_to_legacy<N>(key, k_len, out[emitted + i].key);
                    out[emitted + i].count = group_count_value(*this, data, record);
                }
                // Count every group exactly once even though a batch may begin
                // in the middle of a group owned by the previous batch/worker.
                if (bytes_read != nullptr && group.first_record >= first)
                    *bytes_read += group_bytes(*this, group);
                emitted += take;
                pos += static_cast<size_t>(group_bytes(*this, group));
                ++index;
            }
        }
        return emitted;
    }

    inline uint64_t File::read_flat_records(uint64_t first, uint64_t count, char* out,
                                            std::vector<char>& scratch, uint64_t* bytes_read) const
    {
        if (count == 0 || first >= records) return 0;
        count = std::min(count, records - first);
        const uint64_t record_bytes = uint64_t(key_bytes) + count_bytes;

        const auto it = std::upper_bound(groups.begin(), groups.end(), first,
            [](uint64_t value, const Group& group) { return value < group.first_record; });
        size_t index = static_cast<size_t>(it - groups.begin());
        if (index != 0) --index;

        uint64_t emitted = 0;
        while (emitted < count)
        {
            if (index >= groups.size()) fail("record index outside group table of " + path);
            const size_t window_end = read_group_window(*this, index, scratch);
            size_t pos = 0;
            while (index < window_end && emitted < count)
            {
                const Group& group = groups[index];
                const char* data = scratch.data() + pos;
                const uint8_t prefix = group_prefix_bytes(data);
                const size_t stored_record = group_record_bytes(*this, prefix);
                const uint64_t wanted = first + emitted;
                const uint64_t skip = wanted > group.first_record ? wanted - group.first_record : 0;
                if (skip >= group.records)
                {
                    pos += static_cast<size_t>(group_bytes(*this, group));
                    ++index;
                    continue;
                }
                const uint64_t take = std::min<uint64_t>(count - emitted, group.records - skip);
                for (uint64_t i = 0; i < take; ++i)
                {
                    const uint32_t record = static_cast<uint32_t>(skip + i);
                    char* dst = out + static_cast<size_t>(emitted + i) * record_bytes;
                    std::memcpy(dst, group_prefix(data), prefix);
                    std::memcpy(dst + prefix, group_payload(data) + size_t(record) * stored_record,
                                key_bytes - prefix);
                    if (count_bytes != 0)
                        std::memcpy(dst + key_bytes,
                                    group_payload(data) + size_t(record) * stored_record + (key_bytes - prefix),
                                    count_bytes);
                }
                if (bytes_read != nullptr && group.first_record >= first)
                    *bytes_read += group_bytes(*this, group);
                emitted += take;
                pos += static_cast<size_t>(group_bytes(*this, group));
                ++index;
            }
        }
        return emitted;
    }
} // namespace msd_group

#endif // MSD_GROUP_READER_HEADER
