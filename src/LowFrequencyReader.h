#ifndef LOW_FREQUENCY_READER_HEADER
#define LOW_FREQUENCY_READER_HEADER

#include <stdexcept>
#include "RingMemoryPool.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

class LowFrequencyReader
{
    const size_t packed_bytes;
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>& blocks;
    std::thread worker;
    int fd = -1;
    uint64_t read_records = 0;

public:
    LowFrequencyReader( SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>& in_blocks)
        : packed_bytes(kmer_bytes()), blocks(in_blocks) {}

    void start(const std::string& filename, uint64_t shard_count)
    {
        worker = std::thread([this, filename, shard_count]
            {
                run_worker("pending reader", [&] {
                const uint64_t block_capacity = blocks.blockSize() / packed_bytes;
                for (uint64_t shard = 0; shard < shard_count; ++shard)
                {
                    RunState::check_active();
                    const std::string path = filename + "." + std::to_string(shard);
                    fd = ::open(path.c_str(), O_RDONLY);
                    if (fd < 0)
                        throw std::system_error(errno, std::generic_category(), "open pending shard " + path);
                    struct stat st{};
                    if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size % packed_bytes != 0)
                        throw std::runtime_error("Invalid pending low shard: " + path);
                    const uint64_t total_records = st.st_size / packed_bytes;
                    ::posix_fadvise(fd, 0, st.st_size, POSIX_FADV_SEQUENTIAL);
                    uint64_t shard_records = 0;
                    while (shard_records < total_records)
                    {
                        char* block;
                        blocks.producer_dequeue(block);
                        const uint64_t count = std::min(block_capacity, total_records - shard_records);
                        const size_t bytes = count * packed_bytes;
                        size_t completed = 0;
                        while (completed < bytes)
                        {
                            const ssize_t n = ::pread(fd, block + completed, bytes - completed,
                                static_cast<off_t>(shard_records * packed_bytes + completed));
                            if (n < 0 && errno == EINTR) continue;
                            if (n <= 0) {
                                std::cerr << "Failed to read complete pending low block\n";
                                throw std::runtime_error("input I/O failed");
                            }
                            completed += static_cast<size_t>(n);
                        }
                        blocks.producer_enqueue({block, count});
                        shard_records += count;
                        read_records += count;
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
                });
            });
    }

    ~LowFrequencyReader() { RunState::fail_on_unwind(); join(); }

    void join()
    {
        if (worker.joinable()) worker.join();
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
    }

    uint64_t records_read() const noexcept { return read_records; }
};

#endif
