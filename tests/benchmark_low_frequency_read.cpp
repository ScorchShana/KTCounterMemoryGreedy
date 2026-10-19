// Run via benchmark_low_frequency_read.py; diagnostic headers are generated in a private temporary directory.
#include "BenchMetrics.h"
#include "CompactRecordReader.h"
#include "LowFrequencyThreadPool.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <signal.h>
#include <sys/resource.h>
#include <time.h>

namespace
{
    constexpr uint64_t ring_capacity = 256;
    constexpr uint64_t block_bytes = 512 * 1024;
#ifndef BENCH_LOW_RECORDS
#define BENCH_LOW_RECORDS 32000000ULL
#endif
    constexpr uint64_t low_records = BENCH_LOW_RECORDS;
    static_assert(low_records > 0 && low_records % 4 == 0);
    constexpr uint64_t high_records = low_records / 4;
    constexpr uint64_t kmer_bytes = 11;
    constexpr uint32_t hist_size = 10000;
    constexpr uint32_t bench_count_max = 255;
    using Pool = SPMCRingMemoryPool<ring_capacity>;

    legacy_kmer<2> key_for(uint64_t id)
    {
        legacy_kmer<2> key{};
        key.data[0] = id * 0x9e3779b97f4a7c15ULL;
        key.data[1] = ((id ^ (id >> 16) ^ 0xabcde) & 0x3ffff) << 46;
        return key;
    }

    uint32_t count_for(uint64_t id) { return 1 + (id / 4) % bench_count_max; }

    void generate(const std::filesystem::path& path)
    {
        std::ofstream out(path, std::ios::binary);
        std::vector<char> batch(65536 * kmer_bytes);
        for (uint64_t first = 0; first < low_records; first += 65536)
        {
            uint64_t amount = std::min<uint64_t>(65536, low_records - first);
            for (uint64_t j = 0; j < amount; ++j)
            {
                auto key = key_for(first + j);
                std::memcpy(batch.data() + j * kmer_bytes, key.data.data(), 8);
                std::memcpy(batch.data() + j * kmer_bytes + 8,
                            reinterpret_cast<const char*>(&key.data[1]) + 5, 3);
            }
            out.write(batch.data(), amount * kmer_bytes);
        }
        out.close();
        if (!out) throw std::runtime_error("fixture write failed");
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0 || ::fsync(fd) != 0) throw std::runtime_error("fixture fsync failed");
        ::close(fd);
    }

    void condition_cache(const std::filesystem::path& path, bool cached)
    {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cache preparation open failed");
        if (cached)
        {
            std::vector<char> buffer(4 * 1024 * 1024);
            while (true)
            {
                auto n = ::read(fd, buffer.data(), buffer.size());
                if (n == 0) break;
                if (n < 0 && errno == EINTR) continue;
                if (n < 0) throw std::runtime_error("cache preparation read failed");
            }
        }
        else if (::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED) != 0)
            throw std::runtime_error("file cache eviction advice failed");
        ::close(fd);
    }

    double cpu_seconds()
    {
        rusage usage{};
        if (::getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
        return usage.ru_utime.tv_sec + usage.ru_utime.tv_usec * 1e-6 +
               usage.ru_stime.tv_sec + usage.ru_stime.tv_usec * 1e-6;
    }

    struct Result
    {
        double total = 0, producer = 0, read = 0, producer_wait = 0, worker_wait = 0;
        double worker_processing = 0, cpu = 0, aio_wait = 0, copy = 0;
        uint64_t requests = 0, blocks = 0;
    };

    struct PendingRead
    {
        aiocb cb{};
        char* block = nullptr;
        uint64_t offset = 0, bytes = 0, delivered = 0;
        bool active = false;
    };

    void submit_read(PendingRead& pending, int fd, Result& result)
    {
        std::memset(&pending.cb, 0, sizeof(pending.cb));
        pending.cb.aio_fildes = fd;
        pending.cb.aio_buf = pending.block + pending.delivered;
        pending.cb.aio_nbytes = pending.bytes - pending.delivered;
        pending.cb.aio_offset = pending.offset + pending.delivered;
        pending.cb.aio_sigevent.sigev_notify = SIGEV_NONE;
        ++result.requests;
        if (::aio_read(&pending.cb) != 0)
            throw std::runtime_error("direct aio_read submission failed");
        pending.active = true;
    }

    void wait_for_read(PendingRead& pending)
    {
        const aiocb* list[] = {&pending.cb};
        while (::aio_error(&pending.cb) == EINPROGRESS)
            if (::aio_suspend(list, 1, nullptr) != 0 && errno != EINTR)
                throw std::runtime_error("direct aio_suspend failed");
    }

    void produce_direct_aio(const std::filesystem::path& path, uint32_t depth,
                            Pool& pool, Result& result)
    {
        std::vector<PendingRead> pending(depth);
        const uint64_t total_bytes = low_records * kmer_bytes;
        const uint64_t bytes_per_block = block_bytes / kmer_bytes * kmer_bytes;
        uint64_t next_offset = 0, outstanding = 0;
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("direct AIO open failed");
        try
        {
            if (::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL) != 0)
                throw std::runtime_error("direct AIO fadvise failed");
            auto schedule = [&](PendingRead& slot)
            {
                double t = bench_now();
                pool.producer_dequeue(slot.block);
                result.producer_wait += bench_now() - t;
                slot.offset = next_offset;
                slot.bytes = std::min(bytes_per_block, total_bytes - next_offset);
                slot.delivered = 0;
                t = bench_now();
                submit_read(slot, fd, result);
                result.read += bench_now() - t;
                next_offset += slot.bytes;
                ++outstanding;
            };
            for (auto& slot : pending)
                if (next_offset < total_bytes) schedule(slot);
            size_t index = 0;
            while (outstanding)
            {
                auto& slot = pending[index];
                if (slot.block)
                {
                    double t = bench_now();
                    while (slot.delivered < slot.bytes)
                    {
                        double wait_begin = bench_now();
                        wait_for_read(slot);
                        result.aio_wait += bench_now() - wait_begin;
                        int error = ::aio_error(&slot.cb);
                        ssize_t bytes = ::aio_return(&slot.cb);
                        slot.active = false;
                        if (error || bytes <= 0)
                            throw std::runtime_error("direct AIO read failed or early EOF");
                        slot.delivered += bytes;
                        if (slot.delivered < slot.bytes) submit_read(slot, fd, result);
                    }
                    result.read += bench_now() - t;
                    // Publish only whole, fully read records. The ring block
                    // then belongs to the query worker until it returns it.
                    pool.producer_enqueue(content_type{slot.block, slot.bytes / kmer_bytes});
                    slot.block = nullptr;
                    --outstanding;
                    if (next_offset < total_bytes) schedule(slot);
                }
                index = (index + 1) % pending.size();
            }
            pool.producer_set_finished();
        }
        catch (...)
        {
            // In-flight AIO must finish before the pool storage can be freed.
            for (auto& slot : pending)
                if (slot.active)
                {
                    const aiocb* list[] = {&slot.cb};
                    while (::aio_error(&slot.cb) == EINPROGRESS) ::aio_suspend(list, 1, nullptr);
                    ::aio_return(&slot.cb);
                    slot.active = false;
                }
            ::close(fd);
            throw;
        }
        ::close(fd);
    }

    struct ReadCase
    {
        bool direct;
        uint32_t aio_depth;
        const char* name;
    };

    Result run(const std::filesystem::path& path, bool direct, uint32_t workers,
               FlatConcurrentHashMap<2>& map, const std::vector<int64_t>& high_hist,
               const std::vector<int64_t>& expected, uint32_t aio_depth = 0)
    {
        Pool pool(block_bytes, 1);
        AtomicHistogram histogram(hist_size);
        for (size_t i = 0; i < hist_size; ++i) histogram[i].store(high_hist[i]);
        for (auto& stats : bench_worker_metrics) stats = {};
        LowFrequencyThreadPool<2, ring_capacity> threads(
            &pool, &map, &histogram, workers, 41, 1, hist_size, hist_size, bench_count_max);
        Result result;
        const uint64_t records_per_block = block_bytes / kmer_bytes;
        double cpu_begin = cpu_seconds(), start = bench_now();
        threads.start();
        double producer_begin = bench_now();
        try
        {
            if (!direct)
            {
                CompactRecordReader reader(41, 0);
                reader.open(path.string());
                while (!reader.finished())
                {
                    char* block = nullptr;
                    double t = bench_now();
                    pool.producer_dequeue(block);
                    result.producer_wait += bench_now() - t;
                    t = bench_now();
                    uint64_t count = reader.read_compact_records(block, records_per_block);
                    result.read += bench_now() - t;
                    pool.producer_enqueue(content_type{block, count});
                }
                pool.producer_set_finished();
                reader.close();
                result.requests = reader.bench_requests;
                result.aio_wait = reader.bench_aio_wait;
                result.copy = reader.bench_copy;
            }
            else if (aio_depth)
            {
                produce_direct_aio(path, aio_depth, pool, result);
            }
            else
            {
                int fd = ::open(path.c_str(), O_RDONLY);
                if (fd < 0) throw std::runtime_error("pread open failed");
                try
                {
                    if (::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL) != 0)
                        throw std::runtime_error("pread fadvise failed");
                    uint64_t offset = 0, total_bytes = low_records * kmer_bytes;
                    while (offset < total_bytes)
                    {
                        char* block = nullptr;
                        double t = bench_now();
                        pool.producer_dequeue(block);
                        result.producer_wait += bench_now() - t;
                        uint64_t bytes = std::min(records_per_block * kmer_bytes, total_bytes - offset);
                        uint64_t delivered = 0;
                        t = bench_now();
                        while (delivered < bytes)
                        {
                            ++result.requests;
                            ssize_t n = ::pread(fd, block + delivered, bytes - delivered, offset + delivered);
                            if (n < 0 && errno == EINTR) continue;
                            if (n <= 0) throw std::runtime_error("pread failed or early EOF");
                            delivered += n;
                        }
                        result.read += bench_now() - t;
                        pool.producer_enqueue(content_type{block, bytes / kmer_bytes});
                        offset += bytes;
                    }
                    pool.producer_set_finished();
                }
                catch (...) { ::close(fd); throw; }
                ::close(fd);
            }
        }
        catch (...)
        {
            if (!pool.producer_finished()) pool.producer_set_finished();
            threads.join();
            throw;
        }
        result.producer = bench_now() - producer_begin;
        threads.join();
        result.total = bench_now() - start;
        result.cpu = cpu_seconds() - cpu_begin;
        for (uint32_t i = 0; i < workers; ++i)
        {
            result.worker_wait += bench_worker_metrics[i].queue_wait;
            result.worker_processing += bench_worker_metrics[i].processing;
            result.blocks += bench_worker_metrics[i].blocks;
        }
        if (result.blocks != (low_records + records_per_block - 1) / records_per_block)
            throw std::runtime_error("processed block count mismatch");
        for (size_t i = 0; i < expected.size(); ++i)
            if (histogram[i].load() != expected[i])
                throw std::runtime_error("serial histogram mismatch at frequency " + std::to_string(i + 1));
        return result;
    }
}

int main(int argc, char** argv)
{
    if (argc != 3 && argc != 4) return 1;
    try
    {
        std::filesystem::path root(argv[1]);
        unsigned repetitions = std::stoul(argv[2]);
        std::vector<ReadCase> cases{{false, 0, "aio2m"}, {true, 0, "pread"}};
        if (argc == 4)
        {
            if (std::string(argv[3]) != "direct-aio") throw std::runtime_error("unknown comparison mode");
            cases = {{true, 0, "pread"}, {true, 1, "aio_direct_q1"}, {true, 4, "aio_direct_q4"},
                     {true, 8, "aio_direct_q8"}, {true, 16, "aio_direct_q16"}};
        }
        auto path = root / "low.bin";
        generate(path);
        FlatConcurrentHashMap<2> map(high_records, 16);
        std::vector<int64_t> high_hist(hist_size, 0), expected(hist_size, 0);
        for (uint64_t id = 0; id < low_records; ++id)
        {
            uint32_t final_count = 1;
            if (id % 4 == 0)
            {
                uint32_t count = count_for(id);
                if (!map.insert_unique(key_for(id), count)) throw std::runtime_error("insert failure");
                ++high_hist[count - 1];
                final_count = std::min(count + 1, bench_count_max);
            }
            ++expected[final_count - 1];
        }
        map.seal();
        std::ofstream csv(root / "raw.csv");
        csv << "cache,workers,repeat,method,bytes,total_s,producer_s,read_call_s,read_call_gib_s,phase_gib_s,producer_queue_wait_s,worker_queue_wait_sum_s,worker_queue_wait_avg_s,worker_processing_sum_s,cpu_s,aio_wait_s,copy_s,read_requests,blocks,verified\n";
        csv << std::fixed << std::setprecision(9);
        for (uint32_t workers : {16U, 64U})
        {
            condition_cache(path, true);
            for (const auto& test : cases)
                run(path, test.direct, workers, map, high_hist, expected, test.aio_depth);
            for (bool cached : {true, false})
                for (unsigned repeat = 0; repeat < repetitions; ++repeat)
                    for (size_t order = 0; order < cases.size(); ++order)
                    {
                        const auto& test = cases[(repeat + order) % cases.size()];
                        condition_cache(path, cached);
                        auto r = run(path, test.direct, workers, map, high_hist, expected, test.aio_depth);
                        double gib = low_records * kmer_bytes / (1024. * 1024 * 1024);
                        csv << (cached ? "warm" : "evict_advised") << ',' << workers << ',' << repeat << ','
                            << test.name << ',' << low_records * kmer_bytes << ','
                            << r.total << ',' << r.producer << ',' << r.read << ',' << gib / r.read << ',' << gib / r.total << ','
                            << r.producer_wait << ',' << r.worker_wait << ',' << r.worker_wait / workers << ','
                            << r.worker_processing << ',' << r.cpu << ',' << r.aio_wait << ',' << r.copy << ','
                            << r.requests << ',' << r.blocks << ",true\n";
                        csv.flush();
                        std::cout << (cached ? "warm" : "evict_advised") << " workers=" << workers
                                  << " repeat=" << repeat << " method=" << test.name
                                  << " total=" << r.total << " read=" << r.read
                                  << " producer_wait=" << r.producer_wait << " worker_wait_avg=" << r.worker_wait / workers
                                  << " cpu=" << r.cpu << std::endl;
                    }
        }
        if (!csv) throw std::runtime_error("CSV output failed");
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
