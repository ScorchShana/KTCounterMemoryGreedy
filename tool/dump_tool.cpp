#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "FlatConcurrentHashMap.h"
#include "Partition.h"

namespace
{
    constexpr uint64_t ROOT_BUCKET_COUNT = 1ULL << (2 * ROOT_BASES);
    constexpr uint64_t BYTES_PER_GIB = 1024ULL * 1024ULL * 1024ULL;
    constexpr char BASE_CHARS[] = {'A', 'C', 'G', 'T'};

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
        bool sort_output = false;
        uint32_t k_len = 0;
        uint32_t count_max = 0;
        uint32_t count_bytes = 4;
        uint32_t max_threads = 0;
        uint64_t max_memory_bytes = 0;
        std::string output_file;
        uint32_t min_freq = 1;
        uint32_t max_freq = std::numeric_limits<uint32_t>::max();
    };

    struct RootFileInfo
    {
        std::string filename;
        uint64_t record_count = 0;
        uint64_t file_size = 0;
    };

    class ProgressPrinter
    {
    public:
        explicit ProgressPrinter(uint64_t total_bytes) : total_(total_bytes) {}
        void start() { std::cout << "progress 0%        " << std::flush; }
        void add(uint64_t bytes)
        {
            if (total_ == 0 || bytes == 0)
                return;
            uint64_t done = done_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
            uint64_t pct = (done >= total_) ? 100 : (done * 100) / total_;
            if (pct > 99)
                pct = 99;
            uint64_t prev = last_pct_.load(std::memory_order_relaxed);
            while (pct > prev)
            {
                if (last_pct_.compare_exchange_weak(prev, pct,
                                                    std::memory_order_relaxed, std::memory_order_relaxed))
                {
                    std::lock_guard<std::mutex> lock(cout_mutex_);
                    std::cout << "\rprogress " << std::setw(3) << pct << "%     " << std::flush;
                    break;
                }
            }
        }
        void finish()
        {
            bool expected = false;
            if (finished_.compare_exchange_strong(expected, true))
                std::cout << "\rprogress 100%     " << std::endl;
        }

    private:
        uint64_t total_{0};
        std::atomic<uint64_t> done_{0};
        std::atomic<uint64_t> last_pct_{0};
        std::atomic<bool> finished_{false};
        std::mutex cout_mutex_;
    };

    template <uint32_t N>
    static char *kmer_to_buf(const kmer<N> &key, uint32_t k_len, char *buf)
    {
        const uint32_t full_words = k_len / BASES_PER_U64T;
        const uint32_t tail_bases = k_len % BASES_PER_U64T;
        for (uint32_t i = 0; i < k_len; ++i)
        {
            uint32_t word_idx, bit_shift;
            if (tail_bases > 0 && i < tail_bases)
            {
                word_idx = full_words;
                bit_shift = 64 - 2 * tail_bases + 2 * i;
            }
            else
            {
                uint32_t offset = i - (tail_bases > 0 ? tail_bases : 0);
                word_idx = full_words - 1 - offset / BASES_PER_U64T;
                bit_shift = 2 * (offset % BASES_PER_U64T);
            }
            *buf++ = BASE_CHARS[(key.data[word_idx] >> bit_shift) & 0x3];
        }
        return buf;
    }

    inline char *format_uint32(uint32_t n, char *buf_end)
    {
        do
        {
            *--buf_end = '0' + (n % 10);
            n /= 10;
        } while (n);
        return buf_end;
    }

    static bool in_range(uint64_t value, uint32_t lo, uint32_t hi) noexcept
    {
        return value >= lo && value <= hi;
    }

    // thread write output blocks to disk
    class AsyncWriter
    {
        static constexpr size_t BLOCK_SIZE = 2ULL * 1024 * 1024;
        static constexpr int NUM_BLOCKS = 8;
        int fd_;
        std::thread thread_;
        std::mutex mutex_;
        std::condition_variable cv_;
        std::queue<std::pair<char *, size_t>> ready_queue_;
        std::vector<char *> free_list_;
        bool done_ = false;

        void writer_loop()
        {
            while (true)
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this]
                         { return !ready_queue_.empty() || done_; });
                if (ready_queue_.empty())
                    break;
                auto [buf, bytes] = ready_queue_.front();
                ready_queue_.pop();
                lock.unlock();
                for (size_t offset = 0; offset < bytes;)
                {
                    ssize_t n = ::write(fd_, buf + offset, bytes - offset);
                    if (n < 0)
                    {
                        std::cerr << "write error\n";
                        exit(1);
                    }
                    offset += static_cast<size_t>(n);
                }
                lock.lock();
                free_list_.push_back(buf);
                cv_.notify_all();
            }
        }

    public:
        void open(const std::string &path)
        {
            fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd_ < 0)
            {
                std::cerr << "open failed: " << path << '\n';
                exit(1);
            }
            for (int i = 0; i < NUM_BLOCKS; ++i)
                free_list_.push_back(new char[BLOCK_SIZE]);
            thread_ = std::thread(&AsyncWriter::writer_loop, this);
        }
        char *acquire_buffer()
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [this]
                     { return !free_list_.empty(); });
            char *buf = free_list_.back();
            free_list_.pop_back();
            return buf;
        }
        void release_buffer(char *buf, size_t bytes)
        {
            std::lock_guard lock(mutex_);
            ready_queue_.emplace(buf, bytes);
            cv_.notify_one();
        }
        void finish()
        {
            {
                std::lock_guard lock(mutex_);
                done_ = true;
            }
            cv_.notify_all();
            thread_.join();
            ::close(fd_);
            for (char *buf : free_list_)
                delete[] buf;
            while (!ready_queue_.empty())
            {
                delete[] ready_queue_.front().first;
                ready_queue_.pop();
            }
        }
        static constexpr size_t block_bytes() { return BLOCK_SIZE; }
    };

    //  Writes kmer and lines directly into AsyncWriter blocks.
    template <uint32_t N>
    class LineWriter
    {
        AsyncWriter *writer_;
        char *block_;
        char *cursor_;
        char *block_end_;
        uint32_t k_len_;
        char count_buf_[16];

    public:
        LineWriter(AsyncWriter *aw, uint32_t k_len)
            : writer_(aw), k_len_(k_len)
        {
            block_ = writer_->acquire_buffer();
            cursor_ = block_;
            block_end_ = block_ + AsyncWriter::block_bytes();
        }
        ~LineWriter() { writer_->release_buffer(block_, static_cast<size_t>(cursor_ - block_)); }

        void write_line(const kmer<N> &key, uint32_t count)
        {
            // Reserve space: k_len bases + tab + count + newline + margin
            if (static_cast<size_t>(block_end_ - cursor_) < static_cast<size_t>(k_len_) + 22)
            {
                writer_->release_buffer(block_, static_cast<size_t>(cursor_ - block_));
                block_ = writer_->acquire_buffer();
                cursor_ = block_;
                block_end_ = block_ + AsyncWriter::block_bytes();
            }
            cursor_ = kmer_to_buf<N>(key, k_len_, cursor_);
            *cursor_++ = '\t';
            char *digits = format_uint32(count, count_buf_ + 16);
            size_t digit_len = static_cast<size_t>((count_buf_ + 16) - digits);
            std::memcpy(cursor_, digits, digit_len);
            cursor_ += digit_len;
            *cursor_++ = '\n';
        }
    };

    using uint128_t = unsigned __int128;

    struct Key128
    {
        uint64_t hi, lo;
        Key128() : hi(0), lo(0) {}
        Key128(uint64_t v) : hi(0), lo(v) {}
        Key128(uint64_t h, uint64_t l) : hi(h), lo(l) {}
        Key128 operator<<(int shift) const
        {
            if (shift == 0)
                return *this;
            if (shift < 64)
                return {(hi << shift) | (lo >> (64 - shift)), lo << shift};
            return {lo << (shift - 64), 0};
        }
        Key128 operator>>(int shift) const
        {
            if (shift == 0)
                return *this;
            if (shift < 64)
                return {hi >> shift, (lo >> shift) | (hi << (64 - shift))};
            return {0, hi >> (shift - 64)};
        }
        Key128 operator|(uint64_t v) const { return {hi, lo | v}; }
        uint64_t operator&(uint64_t m) const { return lo & m; }
        Key128 &operator>>=(int shift)
        {
            *this = *this >> shift;
            return *this;
        }
        auto operator<=>(const Key128 &) const = default;
    };

    template <uint32_t N>
    struct SortKeyType;
    template <>
    struct SortKeyType<1>
    {
        using type = uint64_t;
        static constexpr uint32_t bits = 64;
    };
    template <>
    struct SortKeyType<2>
    {
        using type = uint128_t;
        static constexpr uint32_t bits = 128;
    };
    template <>
    struct SortKeyType<4>
    {
        using type = Key128;
        static constexpr uint32_t bits = 256;
    };

    template <uint32_t N>
    static inline uint8_t extract_base_at(const kmer<N> &key, uint32_t pos,
                                          uint32_t full_words, uint32_t tail_bases)
    {
        uint32_t word_idx, bit_shift;
        if (tail_bases > 0 && pos < tail_bases)
        {
            word_idx = full_words;
            bit_shift = 64 - 2 * tail_bases + 2 * pos;
        }
        else
        {
            uint32_t offset = pos - (tail_bases > 0 ? tail_bases : 0);
            word_idx = full_words - 1 - offset / BASES_PER_U64T;
            bit_shift = 2 * (offset % BASES_PER_U64T);
        }
        return (key.data[word_idx] >> bit_shift) & 0x3;
    }

    template <typename KeyT, uint32_t N>
    static KeyT make_sort_key(const kmer<N> &kmer, uint32_t k_len)
    {
        const uint32_t full_words = k_len / BASES_PER_U64T;
        const uint32_t tail_bases = k_len % BASES_PER_U64T;
        KeyT key = 0;
        for (uint32_t i = 0; i < k_len; ++i)
            key = (key << 2) | extract_base_at<N>(kmer, i, full_words, tail_bases);
        return key;
    }

    // 11-bit LSD radix sort
    constexpr uint32_t RADIX_BITS = 11;
    constexpr uint32_t RADIX_BUCKETS = 1U << RADIX_BITS;
    constexpr uint32_t RADIX_MASK = RADIX_BUCKETS - 1;

    template <typename KeyT>
    struct SortEntry
    {
        KeyT key;
        uint32_t count;
    };

    static inline uint32_t radix_digit(uint64_t key, uint32_t shift) { return static_cast<uint32_t>((key >> shift) & RADIX_MASK); }
    static inline uint32_t radix_digit(uint128_t key, uint32_t shift) { return static_cast<uint32_t>((key >> shift) & RADIX_MASK); }
    static inline uint32_t radix_digit(Key128 key, uint32_t shift) { return static_cast<uint32_t>(((key >> shift) & RADIX_MASK)); }

    template <typename KeyT>
    static void parallel_radix_sort(std::vector<SortEntry<KeyT>> &entries,
                                    uint32_t total_bits, uint32_t worker_count)
    {
        uint64_t count = entries.size();
        if (count <= 1)
            return;

        std::vector<SortEntry<KeyT>> temp(count);
        uint32_t passes = (total_bits + RADIX_BITS - 1) / RADIX_BITS;
        SortEntry<KeyT> *src = entries.data();
        SortEntry<KeyT> *dst = temp.data();

        for (uint32_t pass = 0; pass < passes; ++pass)
        {
            uint32_t shift = pass * RADIX_BITS;
            std::vector<std::vector<uint64_t>> local_hists(worker_count,
                                                           std::vector<uint64_t>(RADIX_BUCKETS, 0));
            uint64_t per_thread = (count + worker_count - 1) / worker_count;

            // parallel histogram
            std::vector<std::thread> workers;
            workers.reserve(worker_count);
            for (uint32_t tid = 0; tid < worker_count; ++tid)
            {
                workers.emplace_back([&, tid]()
                                     {
                uint64_t start = tid * per_thread;
                uint64_t end   = std::min(start + per_thread, count);
                if (start >= end) return;
                auto& hist = local_hists[tid];
                for (uint64_t i = start; i < end; ++i)
                    hist[radix_digit(src[i].key, shift)]++; });
            }
            for (auto &w : workers)
                w.join();

            // global prefix sum
            std::vector<uint64_t> global_offsets(RADIX_BUCKETS);
            uint64_t sum = 0;
            for (uint32_t b = 0; b < RADIX_BUCKETS; ++b)
            {
                uint64_t bucket_count = 0;
                for (uint32_t t = 0; t < worker_count; ++t)
                    bucket_count += local_hists[t][b];
                global_offsets[b] = sum;
                sum += bucket_count;
            }

            // convert per-thread histograms to per-thread scatter cursors
            for (uint32_t b = 0; b < RADIX_BUCKETS; ++b)
            {
                uint64_t cursor = global_offsets[b];
                for (uint32_t t = 0; t < worker_count; ++t)
                {
                    uint64_t cnt = local_hists[t][b];
                    local_hists[t][b] = cursor;
                    cursor += cnt;
                }
            }

            // parallel scatter
            workers.clear();
            for (uint32_t tid = 0; tid < worker_count; ++tid)
            {
                workers.emplace_back([&, tid]()
                                     {
                uint64_t start = tid * per_thread;
                uint64_t end   = std::min(start + per_thread, count);
                if (start >= end) return;
                auto& cur = local_hists[tid];
                for (uint64_t i = start; i < end; ++i) {
                    uint32_t bucket = radix_digit(src[i].key, shift);
                    dst[cur[bucket]++] = src[i];
                } });
            }
            for (auto &w : workers)
                w.join();
            std::swap(src, dst);
        }
        if (src == temp.data())
            entries.swap(temp);
    }

    static std::string with_trailing_slash(std::string path)
    {
        if (path.empty())
            return "./";
        char tail = path.back();
        if (tail != '/' && tail != '\\')
            path.push_back('/');
        return path;
    }
    static uint32_t parse_u32(const char *str, const char *name)
    {
        size_t len = 0;
        auto value = std::stoull(str, &len);
        if (len != std::strlen(str) || value > UINT32_MAX)
        {
            std::cerr << "invalid " << name << ": " << str << '\n';
            exit(-1);
        }
        return static_cast<uint32_t>(value);
    }
    static uint64_t parse_memory_gib(const char *str)
    {
        size_t len = 0;
        auto value = std::stold(str, &len);
        if (len != std::strlen(str) || value <= 0.0L)
        {
            std::cerr << "invalid max_memory_gb: " << str << '\n';
            exit(-1);
        }
        auto bytes = value * static_cast<long double>(BYTES_PER_GIB);
        if (bytes > static_cast<long double>(UINT64_MAX))
        {
            std::cerr << "max_memory_gb too large\n";
            exit(-1);
        }
        return static_cast<uint64_t>(bytes);
    }
    static Options parse_options(int argc, char *argv[])
    {
        if (argc < 6 || argc > 9)
        {
            std::cerr << "Usage: dump_tool <precise/approximate> <tmp_dir>"
                      << " <max_threads> <max_memory_gb> <output_file>"
                      << " [min_freq] [max_freq] [--sort]\n";
            exit(-1);
        }
        Options opts;
        opts.tmp_dir = with_trailing_slash(argv[2]);
        opts.max_threads = parse_u32(argv[3], "max_threads");
        opts.max_memory_bytes = parse_memory_gib(argv[4]);
        opts.output_file = argv[5];
        if (std::strcmp(argv[1], "precise") == 0)
            opts.is_precise = true;
        else if (std::strcmp(argv[1], "approximate") == 0)
            opts.is_precise = false;
        else
        {
            std::cerr << "mode must be precise or approximate\n";
            exit(-1);
        }

        int next_arg = 6;
        if (argc > next_arg && argv[next_arg][0] != '-')
            opts.min_freq = parse_u32(argv[next_arg++], "min_freq");
        if (argc > next_arg && argv[next_arg][0] != '-')
            opts.max_freq = parse_u32(argv[next_arg++], "max_freq");
        if (argc > next_arg && std::strcmp(argv[next_arg], "--sort") == 0)
            opts.sort_output = true;

        // k_len is read from infos.bin later
        if (opts.max_threads == 0)
        {
            std::cerr << "invalid max_threads\n";
            exit(-1);
        }
        if (opts.max_freq < opts.min_freq)
        {
            std::cerr << "max_freq < min_freq\n";
            exit(-1);
        }
        return opts;
    }

    static Infos read_infos(const std::string& tmp_dir)
    {
        Infos info;
        const std::string path = tmp_dir + "infos.bin";
        const int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "failed to open " << path << ": " << std::strerror(errno) << '\n';
            exit(-1);
        }
        uint32_t buf[2];
        if (::read(fd, buf, sizeof(buf)) != static_cast<ssize_t>(sizeof(buf)))
        {
            std::cerr << "failed to read " << path << '\n';
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
            std::cerr << "invalid k_len from infos.bin: " << info.k_len << '\n';
            exit(-1);
        }

        return info;
    }

    static std::string high_filename(const std::string &dir)
    {
        return dir + "high" + ".bin";
    }

    template <uint32_t N>
    static uint64_t packed_kmer_bytes(uint32_t k_len)
    {
        uint64_t full_words = k_len / BASES_PER_U64T;
        uint64_t tail_bits = 2ULL * (k_len % BASES_PER_U64T);
        uint64_t tail_bytes = (tail_bits + 7) / 8;
        uint64_t bytes_per_kmer = full_words * sizeof(uint64_t) + tail_bytes;
        if (bytes_per_kmer == 0 || k_len > N * BASES_PER_U64T)
        {
            std::cerr << "invalid k_len for packed bytes: " << k_len << '\n';
            exit(-1);
        }
        return bytes_per_kmer;
    }

    // Read compact k-mer records from file and expand to full ExportRecord<N>
    template <uint32_t N>
    static uint64_t read_compact_high_records(
        int fd, uint64_t file_offset, uint64_t count,
        ExportRecord<N>* out, uint32_t k_len, uint32_t count_bytes)
    {
        const uint64_t full_words = k_len / BASES_PER_U64T;
        const uint64_t tail_bits = 2ULL * (k_len % BASES_PER_U64T);
        const uint64_t tail_bytes = (tail_bits + 7) / 8;
        const uint64_t kmer_bytes = full_words * sizeof(uint64_t) + tail_bytes;
        const uint64_t compact_rec = kmer_bytes + count_bytes;

        const size_t total_bytes = static_cast<size_t>(count * compact_rec);
        std::vector<char> buf(total_bytes);
        ssize_t n = ::pread(fd, buf.data(), total_bytes,
                            static_cast<off_t>(file_offset * compact_rec));
        if (n != static_cast<ssize_t>(total_bytes))
        {
            std::cerr << "short read reading compact high records\n";
            exit(-1);
        }

        for (uint64_t i = 0; i < count; ++i)
        {
            const char* src = buf.data() + i * compact_rec;
            ExportRecord<N>& dst = out[i];
            dst.key.reset();
            std::memcpy(dst.key.data.data(), src, full_words * sizeof(uint64_t));
            if (tail_bytes > 0)
            {
                uint64_t tail = 0;
                std::memcpy(reinterpret_cast<char*>(&tail) + (8 - tail_bytes),
                            src + full_words * sizeof(uint64_t), tail_bytes);
                dst.key.data[full_words] = tail;
            }
            dst.count = 0;
            std::memcpy(&dst.count, src + kmer_bytes, count_bytes);
        }
        return total_bytes;
    }

    // collect high file data
    template <uint32_t N>
    static std::vector<RootFileInfo> collect_high_files(const std::string &tmp_dir,
                                                          uint64_t &total_records,
                                                          uint32_t k_len,
                                                          uint32_t count_bytes)
    {
        const uint64_t compact_rec = packed_kmer_bytes<N>(k_len) + count_bytes;
        std::vector<RootFileInfo> files;

        std::string path = high_filename(tmp_dir);
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "failed to open " << path << ": " << std::strerror(errno) << '\n';
            exit(-1);
        }
        struct stat st{};
        if (::fstat(fd, &st) != 0)
        {
            std::cerr << "failed to stat " << path << ": " << std::strerror(errno) << '\n';
            ::close(fd);
            exit(-1);
        }
        ::close(fd);
        uint64_t size = static_cast<uint64_t>(st.st_size);
        if (size == 0)
        {
            std::cerr << "empty high.bin file\n";
            exit(-1);
        }
        if (size % compact_rec != 0)
        {
            std::cerr << "bad high file size: " << path << '\n';
            exit(-1);
        }
        uint64_t records = size / compact_rec;
        total_records += records;
        files.push_back({std::move(path), records, size});

        return files;
    }

    // build hash map from high files
    template <uint32_t N>
    static void build_hash_map(const std::vector<RootFileInfo> &files,
                               FlatConcurrentHashMap<N> &hash_map,
                               uint32_t worker_count,
                               uint32_t min_freq,
                               uint32_t max_freq,
                               uint32_t k_len,
                               uint32_t count_bytes,
                               ProgressPrinter *progress)
    {
        std::atomic<uint64_t> next_file{0};

        auto worker = [&]()
        {
            std::vector<ExportRecord<N>> buffer(65536);
            for (;;)
            {
                uint64_t index = next_file.fetch_add(1, std::memory_order_relaxed);
                if (index >= files.size())
                    break;
                const RootFileInfo &info = files[index];
                int fd = ::open(info.filename.c_str(), O_RDONLY);
                if (fd < 0)
                {
                    std::cerr << "open: " << info.filename << '\n';
                    exit(-1);
                }

                uint64_t remaining = info.record_count;
                uint64_t offset = 0;
                while (remaining > 0)
                {
                    uint64_t batch = std::min<uint64_t>(remaining, 65536);
                    uint64_t bytes = read_compact_high_records<N>(fd, offset, batch,
                                                                  buffer.data(), k_len, count_bytes);
                    for (uint64_t i = 0; i < batch; ++i)
                        if (in_range(buffer[i].count, min_freq, max_freq))
                            hash_map.insert_unique(buffer[i].key, buffer[i].count);
                    remaining -= batch;
                    offset += batch;
                    progress->add(bytes);
                }
                ::close(fd);
            }
        };

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (uint32_t i = 0; i < worker_count; ++i)
            workers.emplace_back(worker);
        for (auto &w : workers)
            w.join();
    }

    // process low.bin
    template <uint32_t N>
    static void process_low_freq(const std::string &low_path,
                                 FlatConcurrentHashMap<N> &hash_map,
                                 uint32_t k_len,
                                 uint32_t min_freq,
                                 uint32_t max_freq,
                                 uint32_t worker_count,
                                 uint64_t packed_bytes,
                                 uint32_t count_max,
                                 ProgressPrinter *progress,
                                 AsyncWriter *writer)
    {
        int fd = ::open(low_path.c_str(), O_RDONLY);
        if (fd < 0)
        {
            std::cerr << "open low.bin failed\n";
            exit(-1);
        }
        struct stat st{};
        ::fstat(fd, &st);
        uint64_t file_size = static_cast<uint64_t>(st.st_size);
        uint64_t total_kmers = file_size / packed_bytes;

        const char *mapped = static_cast<const char *>(
            ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
        ::close(fd);
        if (mapped == MAP_FAILED)
        {
            std::cerr << "mmap low.bin failed\n";
            exit(-1);
        }

        const uint64_t full_data_words = k_len / BASES_PER_U64T;
        const uint64_t tail_bits = 2ULL * (k_len % BASES_PER_U64T);
        const uint64_t tail_bytes = (tail_bits + 7) / 8;
        uint64_t kmers_per_worker = (total_kmers + worker_count - 1) / worker_count;

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (uint32_t tid = 0; tid < worker_count; ++tid)
        {
            workers.emplace_back([&, tid]()
                                 {
            uint64_t start = tid * kmers_per_worker;
            uint64_t end   = std::min(start + kmers_per_worker, total_kmers);
            if (start >= end) return;

            LineWriter<N> line_writer(writer, k_len);
            kmer<N> key{};

            for (uint64_t i = start; i < end; ++i) {
                const char* record = mapped + i * packed_bytes;
                uint64_t full_bytes = full_data_words * sizeof(uint64_t);
                if (full_bytes > 0)
                    std::memcpy(key.data.data(), record, full_bytes);
                if (tail_bytes > 0) {
                    uint64_t tail = 0;
                    std::memcpy(reinterpret_cast<char*>(&tail) + (sizeof(uint64_t) - tail_bytes),
                                record + full_bytes, tail_bytes);
                    key.data[full_data_words] = tail;
                    for (uint64_t j = full_data_words + 1; j < N; ++j) key.data[j] = 0;
                } else {
                    for (uint64_t j = full_data_words; j < N; ++j) key.data[j] = 0;
                }

                auto lookup = hash_map.prepare_lookup(key);
                hash_map.prefetch(lookup);
                uint32_t count = 0;
                uint64_t slot  = UINT64_MAX;
                if (hash_map.find_prepared_slot(key, lookup, count, slot)) {
                    uint64_t merged = static_cast<uint64_t>(count) + 1;
                    if (merged > count_max) merged = count_max;
                    if (in_range(merged, min_freq, max_freq))
                        line_writer.write_line(key, static_cast<uint32_t>(merged));
                    *hash_map.mutable_count_at(slot) = 0;
                } else if (in_range(1, min_freq, max_freq)) {
                    line_writer.write_line(key, 1);
                }
            }
            progress->add((end - start) * packed_bytes); });
        }
        for (auto &w : workers)
            w.join();
        ::munmap(const_cast<char *>(mapped), file_size);
    }

    // process low.bin
    template <uint32_t N>
    static void process_low_freq_collect(const std::string &low_path,
                                         FlatConcurrentHashMap<N> &hash_map,
                                         std::vector<ExportRecord<N>> &results,
                                         uint32_t k_len,
                                         uint32_t min_freq,
                                         uint32_t max_freq,
                                         uint32_t worker_count,
                                         uint64_t packed_bytes,
                                         uint32_t count_max,
                                         ProgressPrinter *progress)
    {
        int fd = ::open(low_path.c_str(), O_RDONLY);
        struct stat st{};
        ::fstat(fd, &st);
        uint64_t file_size = static_cast<uint64_t>(st.st_size);
        uint64_t total_kmers = file_size / packed_bytes;

        const char *mapped = static_cast<const char *>(
            ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
        ::close(fd);

        const uint64_t full_data_words = k_len / BASES_PER_U64T;
        const uint64_t tail_bytes = (2ULL * (k_len % BASES_PER_U64T) + 7) / 8;
        uint64_t kmers_per_worker = (total_kmers + worker_count - 1) / worker_count;

        std::vector<std::vector<ExportRecord<N>>> high_results(worker_count);
        for (auto &v : high_results)
            v.reserve(total_kmers / worker_count + 1024);

        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (uint32_t tid = 0; tid < worker_count; ++tid)
        {
            workers.emplace_back([&, tid]()
                                 {
            uint64_t start = tid * kmers_per_worker;
            uint64_t end   = std::min(start + kmers_per_worker, total_kmers);
            if (start >= end) return;
            auto& local = high_results[tid];
            kmer<N> key{};

            for (uint64_t i = start; i < end; ++i) {
                const char* record = mapped + i * packed_bytes;
                uint64_t full_bytes = full_data_words * sizeof(uint64_t);
                if (full_bytes > 0)
                    std::memcpy(key.data.data(), record, full_bytes);
                if (tail_bytes > 0) {
                    uint64_t tail = 0;
                    std::memcpy(reinterpret_cast<char*>(&tail) + (sizeof(uint64_t) - tail_bytes),
                                record + full_bytes, tail_bytes);
                    key.data[full_data_words] = tail;
                    for (uint64_t j = full_data_words + 1; j < N; ++j) key.data[j] = 0;
                } else {
                    for (uint64_t j = full_data_words; j < N; ++j) key.data[j] = 0;
                }

                auto lookup = hash_map.prepare_lookup(key);
                hash_map.prefetch(lookup);
                uint32_t count = 0;
                uint64_t slot  = UINT64_MAX;
                if (hash_map.find_prepared_slot(key, lookup, count, slot)) {
                    uint64_t merged = static_cast<uint64_t>(count) + 1;
                    if (merged > count_max) merged = count_max;
                    if (in_range(merged, min_freq, max_freq))
                        local.push_back({key, static_cast<uint32_t>(merged)});
                    *hash_map.mutable_count_at(slot) = 0;
                } else if (in_range(1, min_freq, max_freq)) {
                    local.push_back({key, 1});
                }
            }
            progress->add((end - start) * packed_bytes); });
        }
        for (auto &w : workers)
            w.join();
        ::munmap(const_cast<char *>(mapped), file_size);

        uint64_t total = 0;
        for (auto &v : high_results)
            total += v.size();
        results.reserve(results.size() + total);
        for (auto &v : high_results)
        {
            results.insert(results.end(), v.begin(), v.end());
            v.clear();
        }
    }

    // collect remaining hash entries
    template <uint32_t N>
    static void collect_remaining_stream(FlatConcurrentHashMap<N> &hash_map,
                                         uint32_t min_freq,
                                         uint32_t max_freq,
                                         AsyncWriter *writer,
                                         uint32_t k_len)
    {
        LineWriter<N> line_writer(writer, k_len);
        hash_map.for_each_entry([&](const kmer<N> &key, uint32_t count)
                                {
        if (count > 0 && in_range(count, min_freq, max_freq))
            line_writer.write_line(key, count); });
    }

    //  Phase 3 collect remaining
    template <uint32_t N>
    static void collect_remaining_into_vector(FlatConcurrentHashMap<N> &hash_map,
                                              std::vector<ExportRecord<N>> &results,
                                              uint32_t min_freq, uint32_t max_freq)
    {
        hash_map.for_each_entry([&](const kmer<N> &key, uint32_t count)
                                {
        if (count > 0 && in_range(count, min_freq, max_freq))
            results.push_back({key, count}); });
    }

    // ── Partition-based fallback for insufficient memory ──

    template <uint32_t N>
    static uint32_t compute_partition_bits(uint64_t total_records,
                                           uint64_t max_memory_bytes)
    {
        for (uint32_t P = 1; P <= 10; ++P)
        {
            const uint64_t records_per_bucket =
                (total_records + (1ULL << P) - 1) / (1ULL << P);
            const uint64_t estimated_mem =
                FlatConcurrentHashMap<N>::required_mmap_bytes(records_per_bucket);

            // Add conservative overhead for mmap/low-file mmap/thread buffers
            constexpr uint64_t overhead = 64ULL * 1024 * 1024;

            if (estimated_mem + overhead <= max_memory_bytes)
            {
                return P;
            }
        }
        return 0;
    }

    // Multi-way merge of sorted partition outputs
    static void merge_sorted_outputs(const std::vector<std::string>& input_files,
                                     const std::string& output_file)
    {
        struct HeapEntry
        {
            std::string line;
            size_t file_idx;
            bool operator<(const HeapEntry& other) const
            {
                // Min-heap: smaller k-mer string should come first
                return line > other.line;
            }
        };

        // Use raw pointers since we manage lifetime locally
        std::vector<std::ifstream*> streams;
        std::priority_queue<HeapEntry> heap;

        for (size_t i = 0; i < input_files.size(); ++i)
        {
            auto* s = new std::ifstream(input_files[i]);
            streams.push_back(s);
            std::string line;
            if (std::getline(*s, line))
            {
                heap.push({std::move(line), i});
            }
        }

        std::ofstream out(output_file, std::ios::out | std::ios::trunc);
        if (!out)
        {
            std::cerr << "merge_sorted_outputs: failed to open " << output_file << '\n';
            for (auto* s : streams) { s->close(); delete s; }
            std::exit(1);
        }

        while (!heap.empty())
        {
            auto entry = heap.top();
            heap.pop();
            out << entry.line << '\n';

            std::string next_line;
            if (std::getline(*streams[entry.file_idx], next_line))
            {
                heap.push({std::move(next_line), entry.file_idx});
            }
        }

        for (auto* s : streams)
        {
            s->close();
            delete s;
        }
    }

    // Concatenate partition outputs (already prefix-ordered)
    static void concat_outputs(const std::vector<std::string>& input_files,
                               const std::string& output_file)
    {
        std::ofstream out(output_file, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!out)
        {
            std::cerr << "concat_outputs: failed to open " << output_file << '\n';
            std::exit(1);
        }

        std::vector<char> buf(4ULL * 1024 * 1024);
        for (const auto& path : input_files)
        {
            std::ifstream in(path, std::ios::binary);
            while (in)
            {
                in.read(buf.data(), buf.size());
                out.write(buf.data(), in.gcount());
            }
        }
    }

    template <uint32_t N>
    static int run_partitioned_precise(const Options& opts)
    {
        // Count total high records
        uint64_t total_high_records = 0;
        (void)collect_high_files<N>(opts.tmp_dir, total_high_records, opts.k_len, opts.count_bytes);

        const uint32_t P = compute_partition_bits<N>(total_high_records, opts.max_memory_bytes);
        if (P == 0)
        {
            std::cerr << "insufficient memory: even at P=10 (1024 partitions), "
                << "estimated memory still exceeds " << opts.max_memory_bytes << " bytes\n";
            return 2;
        }

        const uint64_t num_partitions = 1ULL << P;
        std::cout << "Using P=" << P << " (" << num_partitions
                  << " partitions) to fit in " << opts.max_memory_bytes << " bytes"
                  << std::endl;

        // Phase 1: Partition high.bin and low.bin
        const auto high_counts = partition::partition_high_file<N>(
            opts.tmp_dir, P, opts.k_len);
        const auto low_counts = partition::partition_low_file<N>(
            opts.tmp_dir, P, opts.k_len);

        // Phase 2: Process each partition independently
        std::vector<std::string> part_outputs;
        part_outputs.reserve(num_partitions);

        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            if (high_counts[i] == 0 && low_counts[i] == 0)
            {
                const std::string empty_marker = opts.tmp_dir + "partition_" + std::to_string(i) + "/empty";
                part_outputs.push_back(empty_marker);
                continue;
            }

            const std::string part_dir = opts.tmp_dir + "partition_" + std::to_string(i) + "/";
            const std::string part_output = part_dir + "output.txt";

            Options part_opts = opts;
            part_opts.tmp_dir = part_dir;
            part_opts.output_file = part_output;

            std::cout << "Processing partition " << i << "/" << num_partitions
                      << " (high=" << high_counts[i] << ", low=" << low_counts[i] << ")"
                      << std::endl;

            run_precise_impl<N>(part_opts);
            part_outputs.push_back(part_output);
        }

        // Phase 3: Merge partition outputs
        // Filter out empty partitions
        std::vector<std::string> real_outputs;
        for (const auto& p : part_outputs)
        {
            struct stat st {};
            if (::stat(p.c_str(), &st) == 0 && st.st_size > 0)
            {
                real_outputs.push_back(p);
            }
        }

        if (opts.sort_output)
        {
            std::cout << "Merging " << real_outputs.size()
                      << " sorted partition outputs..." << std::endl;
            merge_sorted_outputs(real_outputs, opts.output_file);
        }
        else
        {
            std::cout << "Concatenating " << real_outputs.size()
                      << " partition outputs..." << std::endl;
            concat_outputs(real_outputs, opts.output_file);
        }

        // Phase 4: Clean up partition files
        for (uint64_t i = 0; i < num_partitions; ++i)
        {
            const std::string part_dir = opts.tmp_dir + "partition_" + std::to_string(i) + "/";
            std::remove((part_dir + "high.bin").c_str());
            std::remove((part_dir + "low.bin").c_str());
            std::remove((part_dir + "output.txt").c_str());
            ::rmdir(part_dir.c_str());
        }

        std::cout << "Partitioned dump complete." << std::endl;
        return 0;
    }

    // ── Phase 4  sort records then write through AsyncWriter
    template <uint32_t N>
    static void sort_and_write(std::vector<ExportRecord<N>> &results,
                               const std::string &output_file,
                               uint32_t k_len,
                               uint32_t worker_count)
    {
        using KeyT = typename SortKeyType<N>::type;
        uint32_t sort_bits = std::min<uint32_t>(SortKeyType<N>::bits, k_len * 2);
        uint64_t count = results.size();

        // Build sort entries
        std::vector<SortEntry<KeyT>> entries(count);
        const uint32_t full_words = k_len / BASES_PER_U64T;
        const uint32_t tail_bases = k_len % BASES_PER_U64T;
        for (uint64_t i = 0; i < count; ++i)
        {
            entries[i].key = make_sort_key<KeyT, N>(results[i].key, k_len);
            entries[i].count = results[i].count;
        }
        // Free the large results vector early
        results.clear();
        results.shrink_to_fit();

        // Parallel LSD radix sort
        parallel_radix_sort<KeyT>(entries, sort_bits, worker_count);

        // Write output through AsyncWriter
        AsyncWriter writer;
        writer.open(output_file);
        {
            char *block = writer.acquire_buffer();
            char *cursor = block;
            char *block_end = block + AsyncWriter::block_bytes();
            char count_buf[16];

            for (uint64_t i = 0; i < count; ++i)
            {
                if (static_cast<size_t>(block_end - cursor) < static_cast<size_t>(k_len) + 22)
                {
                    writer.release_buffer(block, static_cast<size_t>(cursor - block));
                    block = writer.acquire_buffer();
                    cursor = block;
                    block_end = block + AsyncWriter::block_bytes();
                }
                // Write bases from sort key (oldest at MSB)
                for (uint32_t j = 0; j < k_len; ++j)
                {
                    uint32_t shift = 2 * (k_len - 1 - j);
                    *cursor++ = BASE_CHARS[static_cast<uint32_t>((entries[i].key >> shift) & 0x3)];
                }
                *cursor++ = '\t';
                char *digits = format_uint32(entries[i].count, count_buf + 16);
                size_t digit_len = static_cast<size_t>((count_buf + 16) - digits);
                std::memcpy(cursor, digits, digit_len);
                cursor += digit_len;
                *cursor++ = '\n';
            }
            if (cursor > block)
                writer.release_buffer(block, static_cast<size_t>(cursor - block));
        }
        writer.finish();
    }

    // Precise mode — core processing (no memory check).
    template <uint32_t N>
    static int run_precise_impl(const Options &opts)
    {
        temp_dir = opts.tmp_dir;
        uint32_t wc = std::max<uint32_t>(1, opts.max_threads);
        uint64_t pkb = packed_kmer_bytes<N>(opts.k_len);

        // Collect high file metadata
        uint64_t total_high_records = 0;
        auto high_files = collect_high_files<N>(opts.tmp_dir, total_high_records, opts.k_len, opts.count_bytes);
        uint64_t high_bytes = 0;
        for (auto &rf : high_files)
            high_bytes += rf.file_size;

        // Check low.bin
        uint64_t low_bytes = 0;
        {
            std::string low_path = opts.tmp_dir + "low.bin";
            int fd = ::open(low_path.c_str(), O_RDONLY);
            if (fd < 0)
            {
                std::cerr << "open low.bin failed\n";
                exit(-1);
            }
            struct stat st{};
            ::fstat(fd, &st);
            ::close(fd);
            low_bytes = static_cast<uint64_t>(st.st_size);
            if (low_bytes % pkb != 0)
            {
                std::cerr << "low.bin size error\n";
                exit(-1);
            }
        }

        ProgressPrinter progress(high_bytes + low_bytes);
        progress.start();

        // build hash map
        FlatConcurrentHashMap<N> hash_map(total_high_records, wc);
        build_hash_map<N>(high_files, hash_map, wc,
                          opts.min_freq, opts.max_freq, opts.k_len, opts.count_bytes, &progress);
        hash_map.seal();

        if (opts.sort_output)
        {
            // Sort path
            std::vector<ExportRecord<N>> results;
            process_low_freq_collect<N>(opts.tmp_dir + "low.bin", hash_map, results,
                                        opts.k_len, opts.min_freq, opts.max_freq,
                                        wc, pkb, opts.count_max, &progress);
            collect_remaining_into_vector<N>(hash_map, results,
                                             opts.min_freq, opts.max_freq);
            progress.finish();
            sort_and_write<N>(results, opts.output_file, opts.k_len, wc);
        }
        else
        {
            AsyncWriter writer;
            writer.open(opts.output_file);
            process_low_freq<N>(opts.tmp_dir + "low.bin", hash_map,
                                opts.k_len, opts.min_freq, opts.max_freq,
                                wc, pkb, opts.count_max, &progress, &writer);
            collect_remaining_stream<N>(hash_map, opts.min_freq, opts.max_freq,
                                        &writer, opts.k_len);
            writer.finish();
            progress.finish();
        }
        return 0;
    }

    // Precise mode — public entry point with memory check.
    template <uint32_t N>
    static int run_precise(const Options &opts)
    {
        temp_dir = opts.tmp_dir;

        uint64_t total_high_records = 0;
        (void)collect_high_files<N>(opts.tmp_dir, total_high_records, opts.k_len, opts.count_bytes);

        // Memory check
        uint64_t estimated_mem = FlatConcurrentHashMap<N>::required_mmap_bytes(total_high_records);
        if (estimated_mem > opts.max_memory_bytes)
        {
            std::cout << "estimated memory " << estimated_mem
                      << " exceeds max " << opts.max_memory_bytes
                      << ", falling back to partitioned mode" << std::endl;
            return run_partitioned_precise<N>(opts);
        }

        return run_precise_impl<N>(opts);
    }

    // Approximate mode
    template <uint32_t N>
    static int run_approximate(const Options &opts)
    {
        temp_dir = opts.tmp_dir;
        uint32_t wc = std::max<uint32_t>(1, opts.max_threads);

        uint64_t total_records = 0;
        auto high_files = collect_high_files<N>(opts.tmp_dir, total_records, opts.k_len, opts.count_bytes);
        uint64_t total_bytes = 0;
        for (auto &rf : high_files)
            total_bytes += rf.file_size;

        ProgressPrinter progress(total_bytes);
        progress.start();

        if (opts.sort_output)
        {
            // Collect all matching records, compute sort keys, global sort, write
            using KeyT = typename SortKeyType<N>::type;
            std::vector<std::vector<ExportRecord<N>>> high_results(wc);
            for (auto &v : high_results)
                v.reserve(total_records / wc + 1024);
            std::atomic<uint64_t> next_file{0};

            auto worker = [&](uint32_t tid)
            {
                std::vector<ExportRecord<N>> buffer(65536);
                auto &local = high_results[tid];
                for (;;)
                {
                    uint64_t index = next_file.fetch_add(1, std::memory_order_relaxed);
                    if (index >= high_files.size())
                        break;
                    const RootFileInfo &info = high_files[index];
                    int fd = ::open(info.filename.c_str(), O_RDONLY);
                    if (fd < 0)
                    {
                        std::cerr << "open: " << info.filename << '\n';
                        exit(-1);
                    }
                    uint64_t remaining = info.record_count;
                    uint64_t offset = 0;
                    while (remaining > 0)
                    {
                        uint64_t batch = std::min<uint64_t>(remaining, 65536);
                        uint64_t bytes = read_compact_high_records<N>(
                            fd, offset, batch, buffer.data(), opts.k_len, opts.count_bytes);
                        for (uint64_t i = 0; i < batch; ++i)
                            if (in_range(buffer[i].count, opts.min_freq, opts.max_freq))
                                local.push_back(buffer[i]);
                        remaining -= batch;
                        offset += batch;
                        progress.add(bytes);
                    }
                    ::close(fd);
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(wc);
            for (uint32_t i = 0; i < wc; ++i)
                workers.emplace_back(worker, i);
            for (auto &w : workers)
                w.join();
            progress.finish();

            // sort and write with parallel radix sort
            uint64_t total = 0;
            for (auto &v : high_results)
                total += v.size();
            std::vector<ExportRecord<N>> results;
            results.reserve(total);
            for (auto &v : high_results)
            {
                results.insert(results.end(), v.begin(), v.end());
                v.clear();
            }
            sort_and_write<N>(results, opts.output_file, opts.k_len, wc);
        }
        else
        {
            // Streaming path: each worker formats output directly
            AsyncWriter writer;
            writer.open(opts.output_file);
            std::atomic<uint64_t> next_file{0};

            auto worker = [&]()
            {
                LineWriter<N> line_writer(&writer, opts.k_len);
                std::vector<ExportRecord<N>> buffer(65536);
                for (;;)
                {
                    uint64_t index = next_file.fetch_add(1, std::memory_order_relaxed);
                    if (index >= high_files.size())
                        break;
                    const RootFileInfo &info = high_files[index];
                    int fd = ::open(info.filename.c_str(), O_RDONLY);
                    if (fd < 0)
                    {
                        std::cerr << "open: " << info.filename << '\n';
                        exit(-1);
                    }
                    uint64_t remaining = info.record_count;
                    uint64_t offset = 0;
                    while (remaining > 0)
                    {
                        uint64_t batch = std::min<uint64_t>(remaining, 65536);
                        uint64_t bytes = read_compact_high_records<N>(
                            fd, offset, batch, buffer.data(), opts.k_len, opts.count_bytes);
                        for (uint64_t i = 0; i < batch; ++i)
                            if (in_range(buffer[i].count, opts.min_freq, opts.max_freq))
                                line_writer.write_line(buffer[i].key, buffer[i].count);
                        remaining -= batch;
                        offset += batch;
                        progress.add(bytes);
                    }
                    ::close(fd);
                }
            };
            std::vector<std::thread> workers;
            workers.reserve(wc);
            for (uint32_t i = 0; i < wc; ++i)
                workers.emplace_back(worker);
            for (auto &w : workers)
                w.join();
            writer.finish();
            progress.finish();
        }
        return 0;
    }

    static int dispatch_precise(const Options &opts)
    {
        if (opts.k_len <= 32)
            return run_precise<1>(opts);
        else if (opts.k_len <= 64)
            return run_precise<2>(opts);
        else if (opts.k_len <= MAX_K)
            return run_precise<4>(opts);
        exit(-1);
    }
    static int dispatch_approximate(const Options &opts)
    {
        if (opts.k_len <= 32)
            return run_approximate<1>(opts);
        else if (opts.k_len <= 64)
            return run_approximate<2>(opts);
        else if (opts.k_len <= MAX_K)
            return run_approximate<4>(opts);
        exit(-1);
    }
}

int main(int argc, char *argv[])
{
    Options opts = parse_options(argc, argv);
    const Infos info = read_infos(opts.tmp_dir);
    opts.k_len = info.k_len;
    opts.count_max = info.count_max;
    opts.count_bytes = info.count_bytes;
    return opts.is_precise ? dispatch_precise(opts) : dispatch_approximate(opts);
}