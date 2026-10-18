#ifndef LOW_FREQUENCY_READER_HEADER
#define LOW_FREQUENCY_READER_HEADER

#include "RingMemoryPool.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

class LowFrequencyReader
{
    const size_t packed_bytes;
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>& blocks;
    std::thread worker;
    int fd = -1;
    uint64_t total_records = 0;
    uint64_t read_records = 0;

public:
    LowFrequencyReader(uint32_t k, SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>& in_blocks)
        : packed_bytes((k + 3) / 4), blocks(in_blocks) {}

    void start(const std::string& filename)
    {
        fd = ::open(filename.c_str(), O_RDONLY);
        if (fd < 0) {
            std::perror("Failed to open pending low file");
            std::exit(-1);
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size % packed_bytes != 0) {
            std::cerr << "Invalid pending low file: " << filename << '\n';
            std::exit(-1);
        }
        total_records = st.st_size / packed_bytes;
        ::posix_fadvise(fd, 0, st.st_size, POSIX_FADV_SEQUENTIAL);

        worker = std::thread([this]
            {
                const uint64_t block_capacity = blocks.blockSize() / packed_bytes;
                while (read_records < total_records)
                {
                    char* block;
                    blocks.producer_dequeue(block);
                    const uint64_t count = std::min(block_capacity, total_records - read_records);
                    const size_t bytes = count * packed_bytes;
                    size_t completed = 0;
                    while (completed < bytes)
                    {
                        const ssize_t n = ::pread(fd, block + completed, bytes - completed,
                            static_cast<off_t>(read_records * packed_bytes + completed));
                        if (n < 0 && errno == EINTR) continue;
                        if (n <= 0) {
                            std::cerr << "Failed to read complete pending low block\n";
                            std::exit(-1);
                        }
                        completed += static_cast<size_t>(n);
                    }
                    blocks.producer_enqueue({block, count});
                    read_records += count;
                }
                blocks.producer_set_finished();
            });
    }

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
