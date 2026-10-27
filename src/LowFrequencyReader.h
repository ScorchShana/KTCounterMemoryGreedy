#ifndef LOW_FREQUENCY_READER_HEADER
#define LOW_FREQUENCY_READER_HEADER

#include "RingMemoryPool.h"
#include "MsdPrefixCodec.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <memory>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

class LowFrequencyReader
{
    friend struct LowFrequencyReaderTestAccess;
public:
    static constexpr size_t READ_CACHE_BYTES = 256ULL << 10;
private:
    const size_t packed_bytes;
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>& blocks;
    std::thread worker;
    int fd = -1;
    uint64_t read_records = 0, read_bytes = 0, read_calls_ = 0;
    std::unique_ptr<uint8_t[]> cache;
    size_t cache_pos = 0, cache_size = 0;
    off_t file_size = 0, file_pos = 0;
    ssize_t (*pread_)(int, void*, size_t, off_t) = ::pread;

    void copy_exact(void* destination, size_t bytes)
    {
        auto* out = static_cast<uint8_t*>(destination);
        while (bytes) {
            if (cache_pos == cache_size) {
                RunState::check_active();
                if (file_pos == file_size) throw std::runtime_error("truncated pending MSD group");
                const size_t request = std::min<uint64_t>(READ_CACHE_BYTES, file_size - file_pos);
                ssize_t n;
                do {
                    ++read_calls_;
                    n = pread_(fd, cache.get(), request, file_pos);
                    if (n < 0 && errno == EINTR) RunState::check_active();
                } while (n < 0 && errno == EINTR);
                if (n < 0) throw std::system_error(errno, std::generic_category(), "read pending shard");
                if (!n) throw std::runtime_error("truncated pending MSD shard");
                cache_pos = 0;
                cache_size = static_cast<size_t>(n);
                file_pos += n;
                read_bytes += n;
            }
            const size_t take = std::min(bytes, cache_size - cache_pos);
            std::memcpy(out, cache.get() + cache_pos, take);
            cache_pos += take;
            out += take;
            bytes -= take;
        }
    }

public:
    explicit LowFrequencyReader(SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>& in_blocks)
        : packed_bytes(kmer_bytes()), blocks(in_blocks) {}

    void start(const std::string& filename, uint64_t shard_count)
    {
        if (blocks.blockSize() < msd_prefix::GROUP_HEADER_BYTES + packed_bytes)
            throw std::invalid_argument("pending read block cannot hold one MSD record");
        cache = std::make_unique<uint8_t[]>(READ_CACHE_BYTES);
        worker = std::thread([this, filename, shard_count] {
            run_worker("pending reader", [&] {
                char* block = nullptr;
                size_t used = 0, decoded = 0;
                try {
                    for (uint64_t shard = 0; shard < shard_count; ++shard) {
                        RunState::check_active();
                        const std::string path = filename + "." + std::to_string(shard);
                        fd = ::open(path.c_str(), O_RDONLY);
                        if (fd < 0) throw std::system_error(errno, std::generic_category(), "open pending shard " + path);
                        struct stat st{};
                        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
                            throw std::runtime_error("Invalid pending low shard: " + path);
                        file_size = st.st_size;
                        file_pos = 0;
                        cache_pos = cache_size = 0;
                        ::posix_fadvise(fd, 0, file_size, POSIX_FADV_SEQUENTIAL);
                        while (cache_pos < cache_size || file_pos < file_size) {
                            std::array<uint8_t, msd_prefix::GROUP_HEADER_BYTES> bytes;
                            copy_exact(bytes.data(), bytes.size());
                            const auto group = msd_prefix::parse_group_header(bytes, packed_bytes);
                            std::array<uint8_t, 32> prefix;
                            copy_exact(prefix.data(), group.prefix_bytes);
                            const size_t header = msd_prefix::GROUP_HEADER_BYTES + group.prefix_bytes;
                            const size_t suffix = packed_bytes - group.prefix_bytes;
                            uint32_t remaining = group.records;
                            while (remaining) {
                                if (!block) {
                                    blocks.producer_dequeue(block);
                                    used = decoded = 0;
                                }
                                const size_t room = blocks.blockSize() - used;
                                const size_t take = room >= header + suffix
                                    ? std::min({size_t(remaining), (room - header) / suffix,
                                        (msd_prefix::BUFFER_BYTES - decoded) / packed_bytes}) : 0;
                                if (!take) {
                                    blocks.producer_enqueue({block, used});
                                    block = nullptr;
                                    continue;
                                }
                                auto* target = reinterpret_cast<uint8_t*>(block) + used;
                                target[0] = static_cast<uint8_t>(group.prefix_bytes);
                                const uint32_t count = static_cast<uint32_t>(take);
                                std::memcpy(target + 1, &count, sizeof(count));
                                std::memcpy(target + msd_prefix::GROUP_HEADER_BYTES, prefix.data(), group.prefix_bytes);
                                copy_exact(target + header, take * suffix);
                                used += header + take * suffix;
                                decoded += take * packed_bytes;
                                remaining -= count;
                                read_records += count;
                            }
                        }
                        if (block) {
                            blocks.producer_enqueue({block, used});
                            block = nullptr;
                        }
                        RunState::check_active();
                        const int closing = fd;
                        fd = -1;
                        if (::close(closing) != 0)
                            throw std::system_error(errno, std::generic_category(), "close pending shard " + path);
                        if (::unlink(path.c_str()) != 0)
                            throw std::system_error(errno, std::generic_category(), "remove pending shard " + path);
                    }
                    blocks.producer_set_finished();
                } catch (...) {
                    if (block) while (!blocks.consumer_try_enqueue(block)) std::this_thread::yield();
                    throw;
                }
            });
        });
    }

    ~LowFrequencyReader() { RunState::fail_on_unwind(); join(); }
    void join()
    {
        if (worker.joinable()) worker.join();
        if (fd >= 0) { ::close(fd); fd = -1; }
    }
    uint64_t records_read() const noexcept { return read_records; }
    uint64_t encoded_bytes_read() const noexcept { return read_bytes; }
    uint64_t read_calls() const noexcept { return read_calls_; }
};
#endif
