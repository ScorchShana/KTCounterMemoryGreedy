#include "HistogramRunner.h"
#include "CompactRecordReader.h"
#include "HighFrequencyThreadPool.h"
#include "LowFrequencyThreadPool.h"

#include <fstream>
#include <iostream>
#include <pthread.h>
#include <sys/stat.h>

namespace
{
    constexpr uint64_t RING_CAPACITY = 1ULL << 8;
    constexpr uint64_t BLOCK_BYTES = 512ULL * 1024;

    class ProgressPrinter
    {
    public:
        explicit ProgressPrinter(long double total) : total_(total)
        {
            std::cout << "progress 0%" << std::flush;
        }
        void add(uint64_t bytes)
        {
            completed_ += bytes;
            unsigned percent = total_ ? static_cast<unsigned>(std::min(99.0L, completed_ * 100 / total_)) : 99;
            if (percent > last_)
            {
                std::cout << "\rprogress " << percent << '%' << std::flush;
                last_ = percent;
            }
        }
        void finish() { std::cout << "\rprogress 100%" << std::endl; }
    private:
        long double total_ = 0, completed_ = 0;
        unsigned last_ = 0;
    };

    uint64_t file_size(const std::string& path, uint64_t record_bytes)
    {
        struct stat st{};
        if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
            throw std::runtime_error("failed to stat regular input " + path);
        uint64_t bytes = static_cast<uint64_t>(st.st_size);
        if (bytes % record_bytes) throw std::runtime_error("misaligned compact input " + path);
        return bytes;
    }

    void read_infos(Options& options)
    {
        const std::string path = options.tmp_dir + "infos.bin";
        if (file_size(path, 1) != 8) throw std::runtime_error("invalid infos.bin size");
        uint32_t fields[2]{};
        std::ifstream input(path, std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(fields), sizeof(fields)))
            throw std::runtime_error("failed to read " + path);
        options.k_len = fields[0];
        options.count_max = fields[1];
        if (!options.k_len || options.k_len > MAX_K || !options.count_max)
            throw std::runtime_error("invalid infos.bin k_len or count_max");
        options.count_bytes = options.count_max <= 255 ? 1 : options.count_max <= 65535 ? 2 :
            options.count_max <= 16777215 ? 3 : 4;
    }

    uint64_t checked_add(uint64_t a, uint64_t b)
    {
        if (b > std::numeric_limits<uint64_t>::max() - a)
            throw std::runtime_error("histogram memory estimate overflow");
        return a + b;
    }

    uint64_t checked_multiply(uint64_t a, uint64_t b)
    {
        if (b && a > std::numeric_limits<uint64_t>::max() / b)
            throw std::runtime_error("histogram memory estimate overflow");
        return a * b;
    }

    // High and low phases are sequential: exactly one ring and reader are live.
    // Include all local histograms, default worker stacks, and allocation/I/O
    // slack, in addition to the mmap table.
    uint64_t fixed_memory_bytes(uint32_t workers, size_t buckets)
    {
        pthread_attr_t attr;
        if (pthread_attr_init(&attr) != 0) throw std::runtime_error("failed to query thread attributes");
        size_t stack_bytes = 0;
        int error = pthread_attr_getstacksize(&attr, &stack_bytes);
        pthread_attr_destroy(&attr);
        if (error) throw std::runtime_error("failed to query thread stack size");
        uint64_t bytes = RING_CAPACITY * BLOCK_BYTES + CompactRecordReader::BUFFER_BYTES;
        bytes = checked_add(bytes, checked_multiply(buckets, sizeof(std::atomic<int64_t>)));
        bytes = checked_add(bytes, checked_multiply(checked_multiply(buckets, sizeof(int64_t)), workers));
        bytes = checked_add(bytes, checked_multiply(workers, checked_add(stack_bytes, 4096)));
        return checked_add(bytes, 2ULL * 1024 * 1024);
    }

    void enqueue_records(const std::string& path, uint32_t k, uint32_t count_bytes,
                         SPMCRingMemoryPool<RING_CAPACITY>& pool, ProgressPrinter& progress,
                         uint64_t first = 0, uint64_t count = std::numeric_limits<uint64_t>::max())
    {
        RecordLayout layout(k, count_bytes);
        CompactRecordReader reader(k, count_bytes);
        reader.open(path, first, count);
        while (!reader.finished())
        {
            char* block = nullptr;
            pool.producer_dequeue(block);
            uint64_t records = reader.read_compact_records(block, BLOCK_BYTES / layout.record_bytes);
            pool.producer_enqueue(content_type{block, records});
            progress.add(records * layout.record_bytes);
        }
        pool.producer_set_finished();
    }

    void write_histogram(const Options& options, const AtomicHistogram& histogram)
    {
        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) throw std::runtime_error("failed to open output file: " + options.output_file);
        for (size_t i = 0; i < histogram.size(); ++i)
            output << static_cast<uint64_t>(options.min_freq) + i << '\t'
                   << histogram[i].load(std::memory_order_relaxed) << '\n';
        output.close();
        if (!output) throw std::runtime_error("failed to write output file: " + options.output_file);
    }

    template <uint32_t N>
    void process_exact_segment(const Options& options, uint64_t first, uint64_t records,
                               AtomicHistogram& histogram, ProgressPrinter& progress,
                               SPMCRingMemoryPool<RING_CAPACITY>& pool,
                               std::atomic<uint64_t>* matched)
    {
        const uint32_t table_k_len = options.k_len;
        FlatConcurrentHashMap<N> hash_map(records, options.max_threads, table_k_len);
        pool.reset_producers(1);
        {
            HighFrequencyThreadPool<N, RING_CAPACITY> threads(
                &pool, &hash_map, &histogram, options.max_threads, options.k_len,
                options.count_bytes, options.min_freq, options.max_freq, histogram.size());
            threads.start();
            enqueue_records(options.tmp_dir + "high.bin", options.k_len, options.count_bytes,
                            pool, progress, first, records);
            threads.join();
            if (threads.insert_failed()) throw std::runtime_error("failed to insert high-frequency records");
        }
        hash_map.seal();
        pool.reset_producers(1);
        {
            LowFrequencyThreadPool<N, RING_CAPACITY> threads(
                &pool, &hash_map, &histogram, options.max_threads, options.k_len,
                options.min_freq, options.max_freq, histogram.size(), options.count_max, matched);
            threads.start();
            enqueue_records(options.tmp_dir + "low.bin", options.k_len, 0, pool, progress);
            threads.join();
        }
    }

    template <uint32_t N>
    int run_precise(const Options& options, uint64_t records, uint64_t high_bytes,
                    uint64_t low_bytes, size_t buckets, uint64_t fixed)
    {
        uint64_t available = options.max_memory_bytes - fixed;
        const uint32_t table_k_len = options.k_len;
        if (FlatConcurrentHashMap<N>::required_mmap_bytes(records ? 1 : 0, table_k_len) > available)
            throw std::runtime_error("insufficient memory budget for histogram hash table");
        uint64_t per_segment = records;
        bool segmented = FlatConcurrentHashMap<N>::required_mmap_bytes(records, table_k_len) > available;
        if (segmented)
        {
            // Find the largest contiguous range whose actual rounded mmap fits.
            uint64_t lo = 1, hi = records;
            while (lo < hi)
            {
                uint64_t mid = lo + (hi - lo + 1) / 2;
                if (FlatConcurrentHashMap<N>::required_mmap_bytes(mid, table_k_len) <= available) lo = mid;
                else hi = mid - 1;
            }
            per_segment = lo;
        }
        uint64_t segments = records ? 1 + (records - 1) / per_segment : 1;
        if (segmented)
            std::cout << "Streaming mode: " << records << " high records in " << segments << " segment(s)\n";
        ProgressPrinter progress(static_cast<long double>(high_bytes) +
                                 static_cast<long double>(low_bytes) * segments);
        AtomicHistogram histogram(buckets);
        for (auto& value : histogram) value.store(0, std::memory_order_relaxed);
        // Reuse the raw block storage after every joined phase and segment.
        SPMCRingMemoryPool<RING_CAPACITY> pool(BLOCK_BYTES, 1);
        std::atomic<uint64_t> matched{0};
        uint64_t first = 0;
        for (uint64_t segment = 0; segment < segments; ++segment)
        {
            uint64_t count = std::min(per_segment, records - first);
            process_exact_segment<N>(options, first, count, histogram, progress, pool, segmented ? &matched : nullptr);
            first += count;
        }
        if (segmented)
        {
            uint64_t low_count = low_bytes / RecordLayout(options.k_len, 0).key_bytes;
            uint64_t matched_count = matched.load(std::memory_order_relaxed);
            if (matched_count > low_count) throw std::runtime_error("invalid duplicate high-frequency keys across segments");
            if (in_range(1, options.min_freq, options.max_freq))
                histogram[histogram_index(1, options.min_freq)].fetch_add(
                    static_cast<int64_t>(low_count - matched_count), std::memory_order_relaxed);
        }
        write_histogram(options, histogram);
        progress.finish();
        return 0;
    }

    template <uint32_t N>
    int run(const Options& options)
    {
        RecordLayout layout(options.k_len, options.count_bytes);
        uint64_t high_bytes = file_size(options.tmp_dir + "high.bin", layout.record_bytes);
        uint64_t low_bytes = options.is_precise ? file_size(options.tmp_dir + "low.bin", layout.key_bytes) : 0;
        uint64_t records = high_bytes / layout.record_bytes;
        size_t buckets = static_cast<size_t>(static_cast<uint64_t>(options.max_freq) - options.min_freq + 1);
        uint64_t fixed = fixed_memory_bytes(options.max_threads, buckets);
        if (fixed > options.max_memory_bytes)
            throw std::runtime_error("insufficient memory budget for histogram buffers and workers");
        if (options.is_precise)
            return run_precise<N>(options, records, high_bytes, low_bytes, buckets, fixed);

        ProgressPrinter progress(high_bytes);
        AtomicHistogram histogram(buckets);
        for (auto& value : histogram) value.store(0, std::memory_order_relaxed);
        {
            SPMCRingMemoryPool<RING_CAPACITY> pool(BLOCK_BYTES, 1);
            ApproximateHighFrequencyThreadPool<N, RING_CAPACITY> threads(
                &pool, &histogram, options.max_threads, options.k_len, options.count_bytes,
                options.min_freq, options.max_freq, buckets);
            threads.start();
            enqueue_records(options.tmp_dir + "high.bin", options.k_len, options.count_bytes, pool, progress);
            threads.join();
        }
        write_histogram(options, histogram);
        progress.finish();
        return 0;
    }
}

int run_histogram_tool(Options options)
{
    if (options.tmp_dir.empty()) options.tmp_dir = "./";
    if (options.tmp_dir.back() != '/') options.tmp_dir += '/';
    if (!options.max_threads || options.max_freq < options.min_freq)
        throw std::runtime_error("invalid histogram options");
    read_infos(options);
    if (options.k_len <= 32) return run<1>(options);
    if (options.k_len <= 64) return run<2>(options);
    return run<4>(options);
}
