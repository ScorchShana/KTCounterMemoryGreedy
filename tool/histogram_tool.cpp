#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <aio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "ExportReader.h"

#include "../src/RingMemoryPool.h"
#include "../src/SpinBackoff.h"

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

    // Main-thread AIO double-buffered reader. It only copies compact records out
    // of the file, byte for byte, into a ring-pool block; unpacking happens in
    // the workers, so no whole block is ever expanded into ExportRecord<N>.
    class CompactRecordReader
    {
        static constexpr uint64_t RAW_BUFFER_SIZE = 512 * 1024;

    public:
        explicit CompactRecordReader(const uint32_t k_len, const uint32_t count_bytes)
        {
            const uint64_t full_words = k_len / BASES_PER_U64T;
            const uint64_t tail_bits = 2 * (k_len % BASES_PER_U64T);
            const uint64_t tail_bytes = (tail_bits + 7) / 8;
            compact_record_size_ = full_words * sizeof(uint64_t) + tail_bytes + count_bytes;

            for (struct aiocb& cb : cbs_)
            {
                std::memset(&cb, 0, sizeof(struct aiocb));
            }
        }

        CompactRecordReader(const CompactRecordReader&) = delete;
        CompactRecordReader& operator=(const CompactRecordReader&) = delete;
        CompactRecordReader(CompactRecordReader&&) = delete;
        CompactRecordReader& operator=(CompactRecordReader&&) = delete;

        ~CompactRecordReader()
        {
            close();
        }

        void open(const std::string& filename)
        {
            if (fd_ >= 0)
            {
                close();
            }

            fd_ = ::open(filename.c_str(), O_RDONLY);
            if (fd_ < 0) [[unlikely]]
            {
                std::cerr << "failed to open " << filename << ": " << std::strerror(errno) << std::endl;
                exit(-1);
            }

            struct stat st{};
            if (::fstat(fd_, &st) != 0) [[unlikely]]
            {
                std::cerr << "failed to stat " << filename << ": " << std::strerror(errno) << std::endl;
                exit(-1);
            }

            const uint64_t file_size = static_cast<uint64_t>(st.st_size);
            if (file_size % compact_record_size_ != 0) [[unlikely]]
            {
                std::cerr << "invalid high.bin size: " << file_size
                    << " is not divisible by compact record size "
                    << compact_record_size_ << std::endl;
                exit(-1);
            }

            ::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);

            record_amount_ = file_size / compact_record_size_;
            scheduled_record_count_ = 0;
            delivered_record_count_ = 0;
            current_buffer_index_ = 0;

            for (uint32_t i = 0; i < 2; ++i)
            {
                std::memset(&cbs_[i], 0, sizeof(struct aiocb));
                if (buffers_[i] == nullptr)
                {
                    buffers_[i] = new char[RAW_BUFFER_SIZE];
                }
                cbs_active_[i] = false;
                buffer_record_count_[i] = 0;
                buffer_cursor_[i] = 0;
            }

            async_read(0);
        }

        void close()
        {
            for (uint32_t i = 0; i < 2; ++i)
            {
                wait_read_finish(i);
            }

            if (fd_ >= 0)
            {
                ::close(fd_);
                fd_ = -1;
            }

            for (uint32_t i = 0; i < 2; ++i)
            {
                delete[] buffers_[i];
                buffers_[i] = nullptr;
                cbs_active_[i] = false;
                buffer_record_count_[i] = 0;
                buffer_cursor_[i] = 0;
                std::memset(&cbs_[i], 0, sizeof(struct aiocb));
            }

            record_amount_ = 0;
            scheduled_record_count_ = 0;
            delivered_record_count_ = 0;
            current_buffer_index_ = 0;
        }

        [[nodiscard]] bool finished() const noexcept
        {
            return delivered_record_count_ >= record_amount_;
        }

        // Copies whole records verbatim; the bytes handed to the pool are the
        // bytes on disk. Small requests, requests that straddle the two buffers
        // and reads past EOF are all handled here.
        uint64_t read_compact_records(char* out_buffer, const uint64_t max_records_to_read)
        {
            if (finished()) [[unlikely]]
            {
                return 0;
            }

            uint64_t total_read = 0;
            while (total_read < max_records_to_read && !finished())
            {
                const uint32_t idx = current_buffer_index_;

                if (cbs_active_[idx])
                {
                    wait_read_finish(idx);
                    async_read(1 - idx);
                }

                const uint64_t available = buffer_record_count_[idx] - buffer_cursor_[idx];
                if (available == 0) [[unlikely]]
                {
                    current_buffer_index_ = 1 - idx;
                    continue;
                }

                const uint64_t to_copy = std::min(max_records_to_read - total_read, available);
                std::memcpy(
                    out_buffer + total_read * compact_record_size_,
                    buffers_[idx] + buffer_cursor_[idx] * compact_record_size_,
                    static_cast<size_t>(to_copy * compact_record_size_));

                buffer_cursor_[idx] += to_copy;
                delivered_record_count_ += to_copy;
                total_read += to_copy;

                if (buffer_cursor_[idx] >= buffer_record_count_[idx])
                {
                    buffer_record_count_[idx] = 0;
                    buffer_cursor_[idx] = 0;
                    current_buffer_index_ = 1 - idx;
                }
            }

            return total_read;
        }

    private:
        void async_read(const uint32_t buffer_index)
        {
            if (scheduled_record_count_ >= record_amount_ || cbs_active_[buffer_index]) [[unlikely]]
            {
                return;
            }

            const uint64_t remaining = record_amount_ - scheduled_record_count_;
            const uint64_t buffer_capacity = RAW_BUFFER_SIZE / compact_record_size_;
            const uint64_t read_count = std::min(buffer_capacity, remaining);
            if (read_count == 0)
            {
                return;
            }

            const uint64_t read_bytes = read_count * compact_record_size_;
            const uint64_t read_offset = scheduled_record_count_ * compact_record_size_;

            buffer_record_count_[buffer_index] = read_count;
            buffer_cursor_[buffer_index] = 0;

            std::memset(&cbs_[buffer_index], 0, sizeof(struct aiocb));
            cbs_[buffer_index].aio_fildes = fd_;
            cbs_[buffer_index].aio_buf = buffers_[buffer_index];
            cbs_[buffer_index].aio_nbytes = static_cast<size_t>(read_bytes);
            cbs_[buffer_index].aio_offset = static_cast<off_t>(read_offset);

            if (::aio_read(&cbs_[buffer_index]) != 0) [[unlikely]]
            {
                std::cerr << "aio_read failed: " << std::strerror(errno) << std::endl;
                exit(-1);
            }

            cbs_active_[buffer_index] = true;
            scheduled_record_count_ += read_count;
        }

        void wait_read_finish(const uint32_t buffer_index)
        {
            if (!cbs_active_[buffer_index]) [[unlikely]]
            {
                return;
            }

            const aiocb* list[1] = { &cbs_[buffer_index] };
            while (::aio_error(&cbs_[buffer_index]) == EINPROGRESS)
            {
                if (::aio_suspend(list, 1, nullptr) != 0 && errno != EINTR) [[unlikely]]
                {
                    std::cerr << "aio_suspend failed" << std::endl;
                    exit(-1);
                }
            }

            const int err = ::aio_error(&cbs_[buffer_index]);
            if (err != 0) [[unlikely]]
            {
                std::cerr << "aio_read error: " << err << std::endl;
                exit(-1);
            }

            const ssize_t bytes_read = ::aio_return(&cbs_[buffer_index]);
            if (bytes_read != static_cast<ssize_t>(cbs_[buffer_index].aio_nbytes)) [[unlikely]]
            {
                std::cerr << "partial aio_read" << std::endl;
                exit(-1);
            }

            cbs_active_[buffer_index] = false;
        }

        int fd_ = -1;
        uint64_t compact_record_size_ = 0;

        uint64_t record_amount_ = 0;
        uint64_t scheduled_record_count_ = 0;
        uint64_t delivered_record_count_ = 0;
        uint32_t current_buffer_index_ = 0;
        std::array<struct aiocb, 2> cbs_{};
        std::array<char*, 2> buffers_{};
        std::array<bool, 2> cbs_active_{};
        std::array<uint64_t, 2> buffer_record_count_{};
        std::array<uint64_t, 2> buffer_cursor_{};
    };

    // Shared by both worker pools below.
    [[nodiscard]] bool in_range(const uint64_t freq, const uint32_t min_freq, const uint32_t max_freq) noexcept
    {
        return freq >= min_freq && freq <= max_freq;
    }

    [[nodiscard]] size_t histogram_index(const uint64_t freq, const uint32_t min_freq) noexcept
    {
        return static_cast<size_t>(freq - min_freq);
    }

    void merge_histogram(
        const std::vector<int64_t>& local_histogram,
        std::vector<std::atomic<int64_t>>& global_histogram)
    {
        for (size_t i = 0; i < local_histogram.size(); ++i)
        {
            const int64_t value = local_histogram[i];
            if (value != 0)
            {
                global_histogram[i].fetch_add(value, std::memory_order_relaxed);
            }
        }
    }

    // Approximate phase: same compact blocks, but only the count field is read.
    template <uint32_t N, uint64_t RING_CAPACITY>
    class CompactApproximateThreadPool
    {
        static constexpr int MAX_SPIN_TIME = 128;
        static constexpr int MAX_BACKOFF = 256;

    public:
        CompactApproximateThreadPool(
            SPMCRingMemoryPool<RING_CAPACITY>* pool,
            std::vector<std::atomic<int64_t>>* global_histogram,
            const uint32_t worker_count,
            const uint32_t k_len,
            const uint32_t count_bytes,
            const uint32_t min_freq,
            const uint32_t max_freq,
            const size_t hist_size)
            : pool_(pool),
            global_histogram_(global_histogram),
            worker_count_(worker_count),
            count_bytes_(count_bytes),
            kmer_bytes_(packed_kmer_bytes_for_k<N>(k_len)),
            compact_record_size_(kmer_bytes_ + count_bytes),
            min_freq_(min_freq),
            max_freq_(max_freq),
            hist_size_(hist_size)
        {
        }

        void start()
        {
            workers_.reserve(worker_count_);
            for (uint32_t i = 0; i < worker_count_; ++i)
            {
                workers_.emplace_back(&CompactApproximateThreadPool::worker_loop, this);
            }
        }

        void join()
        {
            for (auto& worker : workers_)
            {
                if (worker.joinable())
                {
                    worker.join();
                }
            }
        }

    private:
        void worker_loop()
        {
            std::vector<int64_t> local_histogram(hist_size_, 0);
            content_type content{};

            int spin_time = 0;
            int backoff = 2;

            while (true)
            {
                if (pool_->consumer_try_dequeue(content))
                {
                    process_block(content, local_histogram);
                    pool_->consumer_enqueue(content.data);
                }
                else if (pool_->producer_finished()) [[unlikely]]
                {
                    while (pool_->consumer_try_dequeue(content))
                    {
                        process_block(content, local_histogram);
                        pool_->consumer_enqueue(content.data);
                    }
                    break;
                }
                else
                {
                    spin_time++;
                    if (spin_time > MAX_SPIN_TIME)
                    {
                        spin_time = 0;
                        backoff = 2;
                        std::this_thread::yield();
                    }
                    else
                    {
                        for (int i = 0; i < backoff; i++)
                        {
                            cpu_relax();
                        }
                        backoff = std::min(backoff * 2, MAX_BACKOFF);
                    }
                }
            }

            merge_histogram(local_histogram, *global_histogram_);
        }

        void process_block(const content_type& content, std::vector<int64_t>& local_histogram)
        {
            if (content.length == 0) [[unlikely]]
            {
                return;
            }

            const char* record = content.data;
            const char* const record_end = record + content.length * compact_record_size_;
            for (; record < record_end; record += compact_record_size_)
            {
                uint32_t count = 0;
                std::memcpy(&count, record + kmer_bytes_, static_cast<size_t>(count_bytes_));
                if (in_range(count, min_freq_, max_freq_)) [[likely]]
                {
                    local_histogram[histogram_index(count, min_freq_)] += 1;
                }
            }
        }

        SPMCRingMemoryPool<RING_CAPACITY>* pool_ = nullptr;
        std::vector<std::atomic<int64_t>>* global_histogram_ = nullptr;
        uint32_t worker_count_ = 0;
        uint32_t count_bytes_ = sizeof(uint32_t);
        uint64_t kmer_bytes_ = 0;
        uint64_t compact_record_size_ = 0;
        uint32_t min_freq_ = 0;
        uint32_t max_freq_ = 0;
        size_t hist_size_ = 0;
        std::vector<std::thread> workers_;
    };

    template <uint32_t N>
    void enqueue_high_records(
        const std::vector<RootFileInfo>& high_files,
        SPMCRingMemoryPool<HISTOGRAM_RING_CAPACITY>& pool,
        uint32_t k_len,
        uint32_t count_bytes,
        ProgressPrinter* progress)
    {
        const uint64_t compact_rec_size = packed_kmer_bytes_for_k<N>(k_len) + count_bytes;
        const uint64_t records_per_block = HISTOGRAM_BLOCK_BYTES / compact_rec_size;
        if (records_per_block == 0) [[unlikely]]
        {
            std::cerr << "histogram high-frequency block is too small for compact record size "
                << compact_rec_size << std::endl;
            exit(-1);
        }

        for (const RootFileInfo& high_file : high_files)
        {
            CompactRecordReader reader(k_len, count_bytes);
            reader.open(high_file.filename);

            while (!reader.finished())
            {
                char* block_ptr = nullptr;
                pool.producer_dequeue(block_ptr);
                // Compact records go through the pool untouched.
                const uint64_t count = reader.read_compact_records(block_ptr, records_per_block);
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

    // ==================================================================
    // 两趟遍历专用快路径
    //
    // 第二趟的过滤器查询是只读的：同一个 k-mer 的每一次出现都会得到完全
    // 相同的判定，所以它必然整条走向同一个去处 —— low.bin 与 high.bin 的
    // key 集合永不重叠。于是：
    //
    //   high.bin —— 记录里的 count 就是最终计数，直接入直方图；
    //   low.bin  —— 每条都是“只出现一次”的 k-mer，频次恒为 1，
    //               只需数条目数，不必逐条查询。
    //
    // 因此既不需要构建高频哈希表，也不需要为每个 low 条目做一次查找：
    // 临时内存只剩直方图本身（O(max_freq - min_freq)），与数据规模无关。
    // ==================================================================
    template <uint32_t N>
    int run_two_pass_histogram_tool(const Options& options)
    {
        const size_t hist_size = static_cast<size_t>(
            static_cast<uint64_t>(options.max_freq) - static_cast<uint64_t>(options.min_freq) + 1ULL);
        const uint32_t worker_count = std::max<uint32_t>(1, options.max_threads);
        const uint32_t count_bytes = options.count_bytes;
        const uint64_t kmer_bytes = packed_kmer_bytes_for_k<N>(options.k_len);
        const uint64_t compact_rec_size = kmer_bytes + count_bytes;

        uint64_t expected_unique_insert = 0;
        const auto high_files = collect_high_files<N>(options.tmp_dir, expected_unique_insert, options.k_len, count_bytes);
        uint64_t high_file_bytes = 0;
        for (const RootFileInfo& high_file : high_files)
        {
            high_file_bytes += high_file.file_size;
        }

        const uint64_t low_file_bytes = low_file_size_bytes<N>(options.tmp_dir, options.k_len);
        const uint64_t low_kmer_count = low_file_bytes / kmer_bytes;

        ProgressPrinter progress(high_file_bytes + low_file_bytes);
        AtomicHistogram global_histogram(hist_size);
        init_histogram(global_histogram);
        progress.start();

        // ---- 第一段：high.bin 的 count 就是最终计数 ----
        constexpr uint64_t CHUNK_RECORDS = 1ULL << 18;
        for (const RootFileInfo& high_file : high_files)
        {
            const int fd = ::open(high_file.filename.c_str(), O_RDONLY);
            if (fd < 0) [[unlikely]]
            {
                std::cerr << "failed to open " << high_file.filename << ": " << std::strerror(errno) << std::endl;
                exit(-1);
            }
            const char* mapped = static_cast<const char*>(
                ::mmap(nullptr, high_file.file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
            ::close(fd);
            if (mapped == MAP_FAILED) [[unlikely]]
            {
                std::cerr << "mmap high.bin failed" << std::endl;
                exit(-1);
            }

            const uint64_t record_count = high_file.file_size / compact_rec_size;
            std::atomic<uint64_t> next_chunk{ 0 };
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (uint32_t t = 0; t < worker_count; ++t)
            {
                workers.emplace_back([&]()
                    {
                        std::vector<int64_t> local_hist(hist_size, 0);
                        for (;;)
                        {
                            const uint64_t chunk = next_chunk.fetch_add(1, std::memory_order_relaxed);
                            const uint64_t begin = chunk * CHUNK_RECORDS;
                            if (begin >= record_count)
                            {
                                break;
                            }
                            const uint64_t end = std::min(begin + CHUNK_RECORDS, record_count);
                            const char* cursor = mapped + begin * compact_rec_size + kmer_bytes;
                            for (uint64_t i = begin; i < end; ++i, cursor += compact_rec_size)
                            {
                                uint32_t count = 0;
                                std::memcpy(&count, cursor, count_bytes);
                                if (count >= options.min_freq && count <= options.max_freq) [[likely]]
                                {
                                    local_hist[static_cast<size_t>(count - options.min_freq)] += 1;
                                }
                            }
                        }
                        for (size_t j = 0; j < hist_size; ++j)
                        {
                            if (local_hist[j] != 0)
                            {
                                global_histogram[j].fetch_add(local_hist[j], std::memory_order_relaxed);
                            }
                        }
                    });
            }
            for (auto& worker : workers)
            {
                worker.join();
            }

            ::munmap(const_cast<char*>(mapped), high_file.file_size);
            progress.add(high_file.file_size);
        }

        // ---- 第二段：low.bin 全部是频次 1，只累加条目数 ----
        if (options.min_freq <= 1 && 1 <= options.max_freq)
        {
            global_histogram[static_cast<size_t>(1 - options.min_freq)]
                .fetch_add(static_cast<int64_t>(low_kmer_count), std::memory_order_relaxed);
        }
        progress.add(low_file_bytes);
        progress.finish();

        write_histogram(options.output_file, global_histogram, options.min_freq);
        return 0;
    }

    int precise_dispatch_by_k_len(const Options& options)
    {
        if (options.k_len <= 32)
        {
            return run_two_pass_histogram_tool<1>(options);
        }
        else if (options.k_len <= 64)
        {
            return run_two_pass_histogram_tool<2>(options);
        }
        else if (options.k_len <= MAX_K)
        {
            return run_two_pass_histogram_tool<4>(options);
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
            CompactApproximateThreadPool<N, HISTOGRAM_RING_CAPACITY> approximate_threads(
                &approximate_pool,
                &global_histogram,
                worker_count,
                options.k_len,
                options.count_bytes,
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
    const auto tool_start = std::chrono::steady_clock::now();

    Options options = parse_options(argc, argv);
    const Infos info = read_infos(options.tmp_dir);
    options.k_len = info.k_len;
    options.count_max = info.count_max;
    options.count_bytes = info.count_bytes;
    const int exit_code = options.is_precise
        ? precise_dispatch_by_k_len(options)
        : approximate_dispatch_by_k_len(options);

    const auto tool_end = std::chrono::steady_clock::now();
    std::cout << "Total elapsed us: "
        << std::chrono::duration_cast<std::chrono::microseconds>(tool_end - tool_start).count()
        << std::endl;

    return exit_code;
}
