#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ExportReader.h"
#include "FinalDrainReader.h"
#include "FlatConcurrentHashMap.h"
#include "HighFrequencyInsertThreadPool.h"
#include "LowFrequencyQueryThreadPool.h"
#include "ApproximateHighFrequency.h"

#include "../src/RingMemoryPool.h"

namespace
{
    constexpr uint64_t HISTOGRAM_RING_CAPACITY = 1ULL << 7;
    constexpr uint64_t HISTOGRAM_BLOCK_BYTES = 128ULL * 1024ULL;
    constexpr uint64_t ROOT_BUCKET_COUNT = 1ULL << (2 * ROOT_BASES);
    constexpr uint64_t BYTES_PER_GIB = 1024ULL * 1024ULL * 1024ULL;

    struct Infos
    {
        uint32_t k_len = 0;
        uint32_t count_max = 0;
        uint32_t count_bytes = 4;
    };

    struct Options
    {
        std::string tmp_dir;
        bool is_precise = false;
        uint32_t k_len = 0;
        uint32_t count_max = 0;
        uint32_t count_bytes = 4;
        uint32_t max_threads = 0;
        uint64_t max_memory_bytes = 0;
        std::string output_file;
        uint32_t min_freq = 1;
        uint32_t max_freq = 10000;
    };

    struct RootFileInfo
    {
        std::string filename;
        uint64_t record_count = 0;
        uint64_t file_size = 0;
    };

    using AtomicHistogram = std::vector<std::atomic<int64_t>>;

    class ProgressPrinter
    {
    public:
        explicit ProgressPrinter(const uint64_t total_bytes)
            : total_bytes_(total_bytes)
        {
        }

        void start()
        {
            std::cout << "progress 0%";
            std::flush(std::cout);
            if (total_bytes_ == 0)
            {
                last_percent_ = 100;
            }
        }

        void add(const uint64_t bytes)
        {
            if (total_bytes_ == 0 || bytes == 0)
            {
                return;
            }

            completed_bytes_ += bytes;
            uint64_t percent = (completed_bytes_ >= total_bytes_)
                ? 100
                : (completed_bytes_ * 100) / total_bytes_;
            if (percent > 99)
            {
                percent = 99;
            }

            if (percent > last_percent_)
            {
                std::cout << "\rprogress " << percent << "%";
                std::flush(std::cout);
            }
            last_percent_ = percent;
        }

        void finish()
        {
            if (!finished_)
            {
                std::cout << "\rprogress 100%" << std::endl;
                finished_ = true;
            }
        }

    private:
        uint64_t total_bytes_ = 0;
        uint64_t completed_bytes_ = 0;
        uint64_t last_percent_ = 0;
        bool finished_ = false;
    };

    void init_histogram(AtomicHistogram& histogram)
    {
        for (auto& value : histogram)
        {
            value.store(0, std::memory_order_relaxed);
        }
    }

    [[nodiscard]] std::string with_trailing_slash(std::string path)
    {
        if (path.empty())
        {
            return "./";
        }
        const char tail = path.back();
        if (tail != '/' && tail != '\\')
        {
            path.push_back('/');
        }
        return path;
    }

    [[nodiscard]] uint32_t parse_u32(const char* text, const char* name)
    {
        size_t parsed = 0;
        const unsigned long long value = std::stoull(text, &parsed);
        if (parsed != std::strlen(text) || value > std::numeric_limits<uint32_t>::max()) [[unlikely]]
        {
            std::cerr << "invalid " << name << ": " << text << std::endl;
            exit(-1);
        }
        return static_cast<uint32_t>(value);
    }

    [[nodiscard]] uint64_t parse_memory_gib(const char* text)
    {
        size_t parsed = 0;
        const long double value = std::stold(text, &parsed);
        if (parsed != std::strlen(text) || value <= 0.0L) [[unlikely]]
        {
            std::cerr << "invalid max_memory_gb: " << text << std::endl;
            exit(-1);
        }

        const long double bytes = value * static_cast<long double>(BYTES_PER_GIB);
        if (bytes > static_cast<long double>(std::numeric_limits<uint64_t>::max())) [[unlikely]]
        {
            std::cerr << "max_memory_gb is too large" << std::endl;
            exit(-1);
        }
        return static_cast<uint64_t>(bytes);
    }

    [[nodiscard]] Options parse_options(int argc, char* argv[])
    {
        if (argc < 6 || argc > 8)
        {
            std::cerr << "Usage: histogram_tool <precise/approximate> <tmp_dir> <max_threads> <max_memory_gb> <output_file> [min_freq=1] [max_freq=10000]"
                << std::endl;
            exit(-1);
        }

        Options options;
        options.tmp_dir = with_trailing_slash(argv[2]);
        options.max_threads = parse_u32(argv[3], "max_threads");
        options.max_memory_bytes = parse_memory_gib(argv[4]);
        options.output_file = argv[5];

        if (std::strcmp(argv[1], "precise") == 0)
         {
            options.is_precise = true;
        }
        else if (std::strcmp(argv[1], "approximate") == 0)
        {
            options.is_precise = false;
        }
        else
        {
            std::cerr << "invalid mode: must be 'precise' or 'approximate'" << std::endl;
            exit(-1);
        }
        if (argc >= 7)
        {
            options.min_freq = parse_u32(argv[6], "min_freq");
        }
        if (argc >= 8)
        {
            options.max_freq = parse_u32(argv[7], "max_freq");
        }

        if (options.max_threads == 0)
        {
            std::cerr << "invalid max_threads: must be greater than 0" << std::endl;
            exit(-1);
        }
        if (options.max_freq < options.min_freq)
        {
            std::cerr << "invalid max_freq: must be greater than or equal to min_freq" << std::endl;
            exit(-1);
        }
        return options;
    }

    [[nodiscard]] std::string high_filename(const std::string& tmp_dir)
    {
        return tmp_dir + "high.bin";
    }

    void close_fd(const int fd) noexcept
    {
        if (fd >= 0)
        {
            ::close(fd);
        }
    }

    [[nodiscard]] Infos read_infos(const std::string& tmp_dir)
    {
        Infos info;
        const std::string path = tmp_dir + "infos.bin";
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "failed to open " << path << ": " << std::strerror(errno) << std::endl;
            exit(-1);
        }
        uint32_t buf[2];
        if (::read(fd, buf, sizeof(buf)) != static_cast<ssize_t>(sizeof(buf)))
        {
            std::cerr << "failed to read " << path << std::endl;
            ::close(fd);
            exit(-1);
        }
        ::close(fd);
        info.k_len = buf[0];
        info.count_max = buf[1];
        if (info.count_max <= 0xFF)
            info.count_bytes = 1;
        else if (info.count_max <= 0xFFFF)
            info.count_bytes = 2;
        else if (info.count_max <= 0xFFFFFF)
            info.count_bytes = 3;
        else
            info.count_bytes = 4;
        if (info.k_len == 0 || info.k_len > MAX_K)
        {
            std::cerr << "invalid k_len from infos.bin: " << info.k_len << std::endl;
            exit(-1);
        }
        
        return info;
    }

    template <uint32_t N>
    [[nodiscard]] uint64_t packed_kmer_bytes_for_k(const uint32_t k_len)
    {
        const uint64_t full_data_count = k_len / BASES_PER_U64T;
        const uint64_t tail_bits = 2ULL * (k_len % BASES_PER_U64T);
        const uint64_t tail_bytes = (tail_bits + 7ULL) / 8ULL;
        const uint64_t packed_kmer_bytes = full_data_count * sizeof(uint64_t) + tail_bytes;
        if (packed_kmer_bytes == 0 || k_len > N * BASES_PER_U64T) [[unlikely]]
        {
            std::cerr << "invalid k-mer length for packed byte calculation: " << k_len << std::endl;
            exit(-1);
        }
        return packed_kmer_bytes;
    }
    
    template <uint32_t N>
    [[nodiscard]] std::vector<RootFileInfo> collect_high_files(
        const std::string& tmp_dir, uint64_t& expected_unique_insert, uint32_t k_len, uint32_t count_bytes)
    {
        const uint64_t compact_record_size = packed_kmer_bytes_for_k<N>(k_len) + count_bytes;
        std::vector<RootFileInfo> high_files;

        const std::string filename = high_filename(tmp_dir);
        const int fd = ::open(filename.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "failed to open " << filename << ": " << std::strerror(errno) << std::endl;
            exit(-1);
        }

        struct stat st{};
        if (::fstat(fd, &st) != 0)
        {
            close_fd(fd);
            std::cerr << "failed to stat " << filename << ": " << std::strerror(errno) << std::endl;
            exit(-1);
        }
        close_fd(fd);

        const uint64_t file_size = static_cast<uint64_t>(st.st_size);
        if (file_size == 0)
        {
            std::cerr << "empty high.bin file" << std::endl;
            exit(-1);
        }
        if (file_size % compact_record_size != 0)
        {
            std::cerr << "invalid high.bin file size: " << file_size
                      << " is not divisible by compact record size " << compact_record_size << std::endl;
            exit(-1);
        }
        const uint64_t record_count = file_size / compact_record_size;
        expected_unique_insert += record_count;
        high_files.push_back(RootFileInfo{ filename, record_count, file_size });

        return high_files;
    }

    template <uint32_t N>
    [[nodiscard]] uint64_t low_file_size_bytes(const std::string& tmp_dir, const uint32_t k_len)
    {
        const std::string filename = tmp_dir + "low.bin";
        const int fd = ::open(filename.c_str(), O_RDONLY);
        if (fd < 0) [[unlikely]]
        {
            std::cerr << "failed to open " << filename << ": " << std::strerror(errno) << std::endl;
            exit(-1);
        }

        struct stat st
        {
        };
        if (::fstat(fd, &st) != 0) [[unlikely]]
        {
            close_fd(fd);
            std::cerr << "failed to stat " << filename << ": " << std::strerror(errno) << std::endl;
            exit(-1);
        }
        close_fd(fd);

        const uint64_t file_size = static_cast<uint64_t>(st.st_size);
        const uint64_t packed_kmer_bytes = packed_kmer_bytes_for_k<N>(k_len);
        if (file_size % packed_kmer_bytes != 0) [[unlikely]]
        {
            std::cerr << "invalid low.bin size: " << file_size
                << " is not divisible by packed k-mer size "
                << packed_kmer_bytes << std::endl;
            exit(-1);
        }
        return file_size;
    }

    template <uint32_t N>
    [[nodiscard]] uint64_t estimate_peak_memory_bytes(
        const uint64_t expected_unique_insert,
        const uint32_t max_threads,
        const size_t hist_size)
    {
        uint64_t total = FlatConcurrentHashMap<N>::required_mmap_bytes(expected_unique_insert);

        total += HISTOGRAM_RING_CAPACITY * HISTOGRAM_BLOCK_BYTES;
        total += HISTOGRAM_RING_CAPACITY * HISTOGRAM_BLOCK_BYTES;

        total += static_cast<uint64_t>(hist_size) * sizeof(std::atomic<int64_t>);
        total += static_cast<uint64_t>(hist_size) * sizeof(int64_t) * max_threads;

        constexpr uint64_t io_slack_bytes = 8ULL * 1024ULL * 1024ULL;
        total += io_slack_bytes;

        return total;
    }

    template <uint32_t N>
    void enqueue_high_records(
        const std::vector<RootFileInfo>& high_files,
        SPMCRingMemoryPool<HISTOGRAM_RING_CAPACITY>& pool,
        uint32_t k_len,
        uint32_t count_bytes,
        ProgressPrinter* progress)
    {
        using Record = ExportRecord<N>;
        constexpr uint64_t RECORDS_PER_BLOCK = HISTOGRAM_BLOCK_BYTES / sizeof(Record);
        static_assert(RECORDS_PER_BLOCK > 0, "histogram high-frequency block is too small");
        const uint64_t compact_rec_size = packed_kmer_bytes_for_k<N>(k_len) + count_bytes;

        for (const RootFileInfo& high_file : high_files)
        {
            FinalDrainReader<N> reader(k_len, count_bytes);
            reader.open(high_file.filename);

            while (!reader.finished())
            {
                char* block_ptr = nullptr;
                pool.producer_dequeue(block_ptr);
                auto* records = reinterpret_cast<Record*>(block_ptr);
                const uint64_t count = reader.read_records(records, RECORDS_PER_BLOCK);
                if (count == 0) [[unlikely]]
                {
                    pool.consumer_enqueue(block_ptr);
                    break;
                }
                progress->add(count * compact_rec_size);

                pool.producer_enqueue(content_type{ block_ptr, count });
            }
        }

        pool.producer_set_finished();
    }

    template <uint32_t N>
    void enqueue_low_kmers(
        const uint32_t k_len,
        SPMCRingMemoryPool<HISTOGRAM_RING_CAPACITY>& pool,
        const uint64_t packed_kmer_bytes,
        ProgressPrinter* progress)
    {
        const uint64_t kmers_per_block = HISTOGRAM_BLOCK_BYTES / packed_kmer_bytes;
        if (kmers_per_block == 0) [[unlikely]]
        {
            std::cerr << "histogram low-frequency block is too small for packed k-mer size "
                << packed_kmer_bytes << std::endl;
            exit(-1);
        }

        ExportReader<N> reader(k_len);
        reader.open(temp_dir + "low.bin");

        while (!reader.finished())
        {
            char* block_ptr = nullptr;
            pool.producer_dequeue(block_ptr);
            const uint64_t count = reader.read_packed_kmers(block_ptr, kmers_per_block);
            if (count == 0) [[unlikely]]
            {
                pool.consumer_enqueue(block_ptr);
                break;
            }
            progress->add(count * packed_kmer_bytes);

            pool.producer_enqueue(content_type{ block_ptr, count });
        }

        pool.producer_set_finished();
    }

    void write_histogram(
        const std::string& output_filename,
        const AtomicHistogram& histogram,
        const uint32_t min_freq)
    {
        std::ofstream output(output_filename, std::ios::out | std::ios::trunc);
        if (!output)
        {
            std::cerr << "failed to open output file: " << output_filename << std::endl;
            exit(1);
        }

        for (size_t i = 0; i < histogram.size(); ++i)
        {
            output << (static_cast<uint64_t>(min_freq) + i) << '\t'
                << histogram[i].load(std::memory_order_relaxed) << '\n';
        }
    }

    // Expand compact k-mer records from raw bytes (used by streaming path).
    template <uint32_t N>
    static void expand_compact_records(
        const char* src, uint64_t count, ExportRecord<N>* out,
        uint32_t full_words, uint32_t tail_bytes, uint64_t kmer_bytes, uint32_t count_bytes)
    {
        const uint64_t rec_size = kmer_bytes + count_bytes;
        for (uint64_t i = 0; i < count; ++i)
        {
            const char* record = src + i * rec_size;
            ExportRecord<N>& dst = out[i];
            dst.key.reset();
            std::memcpy(dst.key.data.data(), record, full_words * sizeof(uint64_t));
            if (tail_bytes > 0)
            {
                uint64_t tail = 0;
                std::memcpy(reinterpret_cast<char*>(&tail) + (8 - tail_bytes),
                            record + full_words * sizeof(uint64_t), tail_bytes);
                dst.key.data[full_words] = tail;
            }
            dst.count = 0;
            std::memcpy(&dst.count, record + kmer_bytes, count_bytes);
        }
    }

    // Streaming-segment fallback for insufficient memory.
    // Splits high.bin into segments that each fit in the hash map.
    // For each segment, streams ALL of low.bin to merge counts.
    template <uint32_t N>
    int run_streaming_precise_histogram_tool(const Options& options)
    {
        const size_t hist_size = static_cast<size_t>(
            static_cast<uint64_t>(options.max_freq) - static_cast<uint64_t>(options.min_freq) + 1ULL);
        const uint32_t worker_count = std::max<uint32_t>(1, options.max_threads);
        const uint32_t k_len = options.k_len;
        const uint32_t count_bytes = options.count_bytes;

        const uint32_t full_words = k_len / BASES_PER_U64T;
        const uint32_t tail_bits = 2 * (k_len % BASES_PER_U64T);
        const uint32_t tail_bytes = (tail_bits + 7) / 8;
        const uint64_t kmer_bytes = full_words * sizeof(uint64_t) + tail_bytes;
        const uint64_t compact_rec_size = kmer_bytes + count_bytes;

        // Open and mmap high.bin
        const std::string high_path = options.tmp_dir + "high.bin";
        int high_fd = ::open(high_path.c_str(), O_RDONLY);
        if (high_fd < 0)
        {
            std::cerr << "streaming: failed to open " << high_path << '\n';
            exit(-1);
        }
        struct stat high_st {};
        ::fstat(high_fd, &high_st);
        const uint64_t high_file_size = static_cast<uint64_t>(high_st.st_size);
        const uint64_t total_high_records = high_file_size / compact_rec_size;

        const char* high_mapped = static_cast<const char*>(
            ::mmap(nullptr, high_file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, high_fd, 0));
        ::close(high_fd);
        if (high_mapped == MAP_FAILED)
        {
            std::cerr << "streaming: mmap high.bin failed\n";
            exit(-1);
        }

        // Open and mmap low.bin
        const std::string low_path = options.tmp_dir + "low.bin";
        int low_fd = ::open(low_path.c_str(), O_RDONLY);
        if (low_fd < 0)
        {
            std::cerr << "streaming: failed to open " << low_path << '\n';
            ::munmap(const_cast<char*>(high_mapped), high_file_size);
            exit(-1);
        }
        struct stat low_st {};
        ::fstat(low_fd, &low_st);
        const uint64_t low_file_size = static_cast<uint64_t>(low_st.st_size);
        const uint64_t total_low_kmers = low_file_size / kmer_bytes;

        const char* low_mapped = static_cast<const char*>(
            ::mmap(nullptr, low_file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, low_fd, 0));
        ::close(low_fd);
        if (low_mapped == MAP_FAILED)
        {
            std::cerr << "streaming: mmap low.bin failed\n";
            ::munmap(const_cast<char*>(high_mapped), high_file_size);
            exit(-1);
        }

        // Compute segment count: reduce records_per_segment until hash map fits
        const uint64_t histogram_overhead =
            static_cast<uint64_t>(hist_size) * sizeof(std::atomic<int64_t>)
            + static_cast<uint64_t>(hist_size) * sizeof(int64_t) * worker_count
            + 4ULL * 1024 * 1024;  // 4MB I/O slack

        uint32_t num_segments = 1;
        while (num_segments < 1024)
        {
            const uint64_t records_per_seg =
                (total_high_records + num_segments - 1) / num_segments;
            const uint64_t hash_mem =
                FlatConcurrentHashMap<N>::required_mmap_bytes(records_per_seg);
            if (hash_mem + histogram_overhead <= options.max_memory_bytes)
                break;
            num_segments *= 2;
        }
        if (num_segments >= 1024)
        {
            std::cerr << "streaming: even with 1024 segments, hash map won't fit in "
                      << options.max_memory_bytes << " bytes\n";
            ::munmap(const_cast<char*>(high_mapped), high_file_size);
            ::munmap(const_cast<char*>(low_mapped), low_file_size);
            return 2;
        }

        std::cout << "Streaming mode: " << total_high_records << " high records in "
                  << num_segments << " segment(s)" << std::endl;

        AtomicHistogram global_histogram(hist_size);
        init_histogram(global_histogram);
        std::atomic<uint64_t> total_matched{0};

        const uint64_t records_per_segment =
            (total_high_records + num_segments - 1) / num_segments;

        for (uint32_t seg = 0; seg < num_segments; ++seg)
        {
            const uint64_t seg_start = seg * records_per_segment;
            const uint64_t seg_end = std::min(seg_start + records_per_segment, total_high_records);
            const uint64_t seg_count = seg_end - seg_start;
            if (seg_count == 0) continue;

            std::cout << "Segment " << (seg + 1) << "/" << num_segments
                      << " (" << seg_count << " records)" << std::endl;

            FlatConcurrentHashMap<N> hash_map(seg_count, worker_count);
            std::atomic<uint64_t> next_chunk{0};
            constexpr uint64_t CHUNK_SIZE = 65536;

            // Phase A: Insert this segment's high records into hash map
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (uint32_t t = 0; t < worker_count; ++t)
            {
                workers.emplace_back([&]()
                {
                    std::vector<ExportRecord<N>> buffer(CHUNK_SIZE);
                    std::vector<int64_t> local_hist(hist_size, 0);
                    for (;;)
                    {
                        uint64_t chunk_idx = next_chunk.fetch_add(1, std::memory_order_relaxed);
                        const uint64_t chunk_start = seg_start + chunk_idx * CHUNK_SIZE;
                        if (chunk_start >= seg_end) break;
                        const uint64_t chunk_end = std::min(chunk_start + CHUNK_SIZE, seg_end);
                        const uint64_t n = chunk_end - chunk_start;

                        const char* src = high_mapped + chunk_start * compact_rec_size;
                        expand_compact_records<N>(src, n, buffer.data(),
                                                   full_words, tail_bytes, kmer_bytes, count_bytes);
                        for (uint64_t i = 0; i < n; ++i)
                        {
                            const uint64_t c = buffer[i].count;
                            if (c >= options.min_freq && c <= options.max_freq)
                                local_hist[static_cast<size_t>(c - options.min_freq)] += 1;
                            hash_map.insert_unique(buffer[i].key, buffer[i].count);
                        }
                    }
                    for (size_t j = 0; j < hist_size; ++j)
                        if (local_hist[j] != 0)
                            global_histogram[j].fetch_add(local_hist[j], std::memory_order_relaxed);
                });
            }
            for (auto& w : workers) w.join();

            hash_map.seal();

            // Phase B: Stream ALL of low.bin, query hash map
            next_chunk.store(0, std::memory_order_relaxed);
            workers.clear();
            for (uint32_t t = 0; t < worker_count; ++t)
            {
                workers.emplace_back([&]()
                {
                    std::vector<int64_t> local_hist(hist_size, 0);
                    uint64_t local_matched = 0;
                    kmer<N> key{};

                    for (;;)
                    {
                        uint64_t chunk_idx = next_chunk.fetch_add(1, std::memory_order_relaxed);
                        const uint64_t chunk_start = chunk_idx * CHUNK_SIZE;
                        if (chunk_start >= total_low_kmers) break;
                        const uint64_t chunk_end = std::min(chunk_start + CHUNK_SIZE, total_low_kmers);

                        for (uint64_t i = chunk_start; i < chunk_end; ++i)
                        {
                            const char* record = low_mapped + i * kmer_bytes;
                            key.reset();
                            const uint64_t full_bytes = full_words * sizeof(uint64_t);
                            if (full_bytes > 0)
                                std::memcpy(key.data.data(), record, full_bytes);
                            if (tail_bytes > 0)
                            {
                                uint64_t tail = 0;
                                std::memcpy(reinterpret_cast<char*>(&tail) + (8 - tail_bytes),
                                            record + full_bytes, tail_bytes);
                                key.data[full_words] = tail;
                            }

                            auto lookup = hash_map.prepare_lookup(key);
                            hash_map.prefetch(lookup);
                            uint32_t stored_count = 0;
                            if (hash_map.find_prepared(key, lookup, stored_count))
                            {
                                if (stored_count >= options.min_freq && stored_count <= options.max_freq)
                                    local_hist[static_cast<size_t>(stored_count - options.min_freq)] -= 1;
                                uint64_t merged = static_cast<uint64_t>(stored_count) + 1;
                                if (merged > options.count_max)
                                    merged = options.count_max;
                                if (merged >= options.min_freq && merged <= options.max_freq)
                                    local_hist[static_cast<size_t>(merged - options.min_freq)] += 1;
                                local_matched++;
                            }
                        }
                    }
                    for (size_t j = 0; j < hist_size; ++j)
                        if (local_hist[j] != 0)
                            global_histogram[j].fetch_add(local_hist[j], std::memory_order_relaxed);
                    total_matched.fetch_add(local_matched, std::memory_order_relaxed);
                });
            }
            for (auto& w : workers) w.join();
        }

        // Adjust cnt[1] for low kmers that never matched any high record
        const uint64_t unmatched = total_low_kmers - total_matched.load(std::memory_order_relaxed);
        if (options.min_freq <= 1 && 1 <= options.max_freq)
        {
            const size_t idx1 = static_cast<size_t>(1 - options.min_freq);
            global_histogram[idx1].fetch_add(static_cast<int64_t>(unmatched), std::memory_order_relaxed);
        }

        std::cout << "Matched " << total_matched << " low kmers, "
                  << unmatched << " unmatched (freq=1)" << std::endl;

        write_histogram(options.output_file, global_histogram, options.min_freq);

        ::munmap(const_cast<char*>(high_mapped), high_file_size);
        ::munmap(const_cast<char*>(low_mapped), low_file_size);
        return 0;
    }

    // Core precise processing without memory check (ring-pool-based).
    template <uint32_t N>
    int run_precise_histogram_tool_impl(const Options& options)
    {
        temp_dir = options.tmp_dir;

        const size_t hist_size = static_cast<size_t>(
            static_cast<uint64_t>(options.max_freq) - static_cast<uint64_t>(options.min_freq) + 1ULL);
        const uint32_t worker_count = std::max<uint32_t>(1, options.max_threads);

        uint64_t expected_unique_insert = 0;
        auto high_files = collect_high_files<N>(options.tmp_dir, expected_unique_insert, options.k_len, options.count_bytes);
        uint64_t high_file_bytes = 0;
        for (const RootFileInfo& high_file : high_files)
            high_file_bytes += high_file.file_size;

        const uint64_t low_file_bytes = low_file_size_bytes<N>(options.tmp_dir, options.k_len);
        const uint64_t packed_low_kmer_bytes = packed_kmer_bytes_for_k<N>(options.k_len);
        ProgressPrinter progress(high_file_bytes + low_file_bytes);

        AtomicHistogram global_histogram(hist_size);
        init_histogram(global_histogram);

        FlatConcurrentHashMap<N> hash_map(expected_unique_insert, worker_count);
        progress.start();

        {
            SPMCRingMemoryPool<HISTOGRAM_RING_CAPACITY> high_pool(HISTOGRAM_BLOCK_BYTES, 1);
            HighFrequencyInsertThreadPool<N, HISTOGRAM_RING_CAPACITY> high_threads(
                &high_pool, &hash_map, &global_histogram,
                worker_count, options.k_len, options.min_freq, options.max_freq, hist_size);
            high_threads.start();
            enqueue_high_records<N>(high_files, high_pool, options.k_len, options.count_bytes, &progress);
            high_threads.join();
            if (high_threads.insert_failed())
            {
                std::cerr << "failed to insert one or more high-frequency records" << std::endl;
                exit(-1);
            }
        }

        hash_map.seal();

        {
            SPMCRingMemoryPool<HISTOGRAM_RING_CAPACITY> low_pool(HISTOGRAM_BLOCK_BYTES, 1);
            LowFrequencyQueryThreadPool<N, HISTOGRAM_RING_CAPACITY> low_threads(
                &low_pool, &hash_map, &global_histogram,
                worker_count, options.k_len, options.min_freq, options.max_freq, hist_size,
                options.count_max);
            low_threads.start();
            enqueue_low_kmers<N>(options.k_len, low_pool, packed_low_kmer_bytes, &progress);
            low_threads.join();
        }

        write_histogram(options.output_file, global_histogram, options.min_freq);
        progress.finish();
        return 0;
    }

    // Public entry point: checks memory, falls back to streaming if needed.
    template <uint32_t N>
    int run_precise_histogram_tool(const Options& options)
    {
        temp_dir = options.tmp_dir;

        const size_t hist_size = static_cast<size_t>(
            static_cast<uint64_t>(options.max_freq) - static_cast<uint64_t>(options.min_freq) + 1ULL);
        const uint32_t worker_count = std::max<uint32_t>(1, options.max_threads);

        uint64_t expected_unique_insert = 0;
        (void)collect_high_files<N>(options.tmp_dir, expected_unique_insert, options.k_len, options.count_bytes);

        const uint64_t estimated_peak_memory =
            estimate_peak_memory_bytes<N>(expected_unique_insert, worker_count, hist_size);
        if (estimated_peak_memory > options.max_memory_bytes)
        {
            std::cout << "estimated_peak_memory_bytes=" << estimated_peak_memory
                << " exceeds max_memory_bytes=" << options.max_memory_bytes
                << ", falling back to streaming mode" << std::endl;
            return run_streaming_precise_histogram_tool<N>(options);
        }

        return run_precise_histogram_tool_impl<N>(options);
    }

    int precise_dispatch_by_k_len(const Options& options)
    {
        if (options.k_len <= 32)
        {
            return run_precise_histogram_tool<1>(options);
        }
        else if (options.k_len <= 64)
        {
            return run_precise_histogram_tool<2>(options);
        }
        else if (options.k_len <= MAX_K)
        {
            return run_precise_histogram_tool<4>(options);
        }
        else
        {
            std::cerr << "unsupported k_len: " << options.k_len << '\n';
            exit(-1);
        }
    }

    template <uint32_t N>
    int run_approximate_histogram_tool(const Options& options){
        temp_dir = options.tmp_dir;

        const size_t hist_size = static_cast<size_t>(
            static_cast<uint64_t>(options.max_freq) - static_cast<uint64_t>(options.min_freq) + 1ULL);
        const uint32_t worker_count = std::max<uint32_t>(1, options.max_threads);

        uint64_t expected_unique_insert = 0;
        auto high_files = collect_high_files<N>(options.tmp_dir, expected_unique_insert, options.k_len, options.count_bytes);
        uint64_t high_file_bytes = 0;
        for (const RootFileInfo& high_file : high_files)
        {
            high_file_bytes += high_file.file_size;
        }

        ProgressPrinter progress(high_file_bytes);

        AtomicHistogram global_histogram(hist_size);
        init_histogram(global_histogram);

        progress.start();

        {
            SPMCRingMemoryPool<HISTOGRAM_RING_CAPACITY> approximate_pool(HISTOGRAM_BLOCK_BYTES, 1);
            ApproximateHighFrequencyThreadPool<N, HISTOGRAM_RING_CAPACITY> approximate_threads(
                &approximate_pool,
                &global_histogram,
                worker_count,
                options.k_len,
                options.min_freq,
                options.max_freq,
                hist_size);

            approximate_threads.start();
            enqueue_high_records<N>(high_files, approximate_pool, options.k_len, options.count_bytes, &progress);
            approximate_threads.join();
        }

        write_histogram(options.output_file, global_histogram, options.min_freq);
        progress.finish();
        return 0;
    }

    int approximate_dispatch_by_k_len(const Options& options)
    {
        if (options.k_len <= 32)
        {
            return run_approximate_histogram_tool<1>(options);
        }
        else if (options.k_len <= 64)
        {
            return run_approximate_histogram_tool<2>(options);
        }
        else if (options.k_len <= MAX_K)
        {
            return run_approximate_histogram_tool<4>(options);
        }
        else
        {
            std::cerr << "unsupported k_len: " << options.k_len << '\n';
            exit(-1);
        }
    }
}

int main(int argc, char* argv[])
{
    Options options = parse_options(argc, argv);
    const Infos info = read_infos(options.tmp_dir);
    options.k_len = info.k_len;
    options.count_max = info.count_max;
    options.count_bytes = info.count_bytes;
    return options.is_precise ? precise_dispatch_by_k_len(options) : approximate_dispatch_by_k_len(options);
}
