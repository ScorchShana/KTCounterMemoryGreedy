#ifndef FASTQ_READER_HEADER
#define FASTQ_READER_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "SPSCRingQueue.h"
#include "SpinBackoff.h"
#include "GzipStreamer.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <utility>
#include <iostream>
#include <cstdlib>
#include <zlib.h>
#include <thread>
#include <atomic>
#include <algorithm>

template <uint32_t N>
class FastqReader
{

    using content_type = std::pair<char*, uint64_t>;

    enum class State
    {
        ReadHeader,
        ReadSequence,
        ReadPlus,
        ReadQuality
    };

    static constexpr uint64_t kNoNewlineInBlock = static_cast<uint64_t>(-1);
    static constexpr uint64_t PIPELINE_QUEUE_CAPACITY = 16; // must be power of 2

    int acbs_index = 0;

    State state_ = State::ReadHeader;
    int fd_ = -1;
    off_t file_size_ = 0;
    std::atomic<off_t> have_read_{ 0 };
    uint64_t chunk_size_;
    std::vector<std::string> filenames_;
    size_t file_index_ = 0;
    SPMCRingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>* ring_memory_pool_ptr_;
    char* file_buffer;
    char left_buffer_[128];
    size_t left_buffer_size_ = 0;
    bool is_gz_file = false;
    gzFile gzfile_ = nullptr;

    std::vector<char*> pipeline_buffers_;
    SPSCRingQueue<::content_type, PIPELINE_QUEUE_CAPACITY> pipeline_data_queue_;
    SPSCRingQueue<char*, PIPELINE_QUEUE_CAPACITY> pipeline_free_queue_;
    std::thread io_thread_;

public:
#ifdef TEST_MODE
    uint64_t total_dequeue_spin_time = 0;
    uint64_t total_enqueue_spin_time = 0;
    uint64_t aio_wait_spin_time = 0;
#endif

    int k;

    explicit FastqReader(const std::vector<std::string>& filenames, const int in_k, uint64_t chunk_size, SPMCRingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>* in_ring_memory_pool_ptr)
        : acbs_index(0), filenames_(filenames), ring_memory_pool_ptr_(in_ring_memory_pool_ptr),
        k(in_k), chunk_size_(chunk_size)
    {
        assert(ring_memory_pool_ptr_ != nullptr);
        assert(k > 0 && k < 128);
        assert(ring_memory_pool_ptr_->blockSize() > 1024);
        if (filenames_.empty()) { std::cerr << "No input files" << std::endl; std::exit(-1); }

        // 始终预分配 pipeline buffers
        pipeline_buffers_.resize(PIPELINE_QUEUE_CAPACITY);
        for (uint64_t i = 0; i < PIPELINE_QUEUE_CAPACITY; ++i)
        {
            pipeline_buffers_[i] = static_cast<char*>(std::aligned_alloc(4096, chunk_size_));
            if (!pipeline_buffers_[i]) { std::cerr << "Failed to allocate pipeline buffer" << std::endl; std::exit(-1); }
            pipeline_free_queue_.enqueue(pipeline_buffers_[i]);
        }

        file_buffer = static_cast<char*>(std::aligned_alloc(4096, chunk_size_));
        if (!file_buffer) { std::cerr << "Failed to allocate file buffer" << std::endl; std::exit(-1); }
    }

    ~FastqReader()
    {
        if (io_thread_.joinable()) io_thread_.join();
        if (gzfile_ != nullptr) gzclose(gzfile_);
        if (fd_ != -1) ::close(fd_);
        if (file_buffer) std::free(file_buffer);
        for (char* buf : pipeline_buffers_) if (buf) std::free(buf);
    }

    void read()
    {
        for (file_index_ = 0; file_index_ < filenames_.size(); ++file_index_)
        {
            open_current_file();

            // 共同: 释放构造分配的 file_buffer(由I/O线程填入pipeline buffer)
            if (file_buffer) { std::free(file_buffer); file_buffer = nullptr; }

            // 启动 I/O 线程 (gzread 或 ::read)
            io_thread_ = std::thread(&FastqReader::io_thread_func, this);

            // 解析线程: 始终从 pipeline queue 取数据
            ::content_type current_input{ nullptr, 0 };
            bool have_input = false;
            parse_loop(current_input, have_input);

            io_thread_.join();
            if (have_input && current_input.data != nullptr)
                pipeline_free_queue_.enqueue(current_input.data);
            file_buffer = nullptr;

            close_current_file();
            state_ = State::ReadHeader;
            left_buffer_size_ = 0;
            have_read_.store(0, std::memory_order_relaxed);
        }
        ring_memory_pool_ptr_->producer_set_finished();
    }

private:
    void open_current_file()
    {
        const std::string& fname = filenames_[file_index_];
        std::cout << "FastqReader: " << (file_index_ + 1) << "/"
            << filenames_.size() << ": " << fname << std::endl;

        fd_ = ::open(fname.data(), O_RDONLY);
        if (fd_ == -1) { std::cerr << "Failed to open: " << fname << std::endl; std::exit(-1); }
        struct stat st;
        if (fstat(fd_, &st) == -1) { std::cerr << "Failed to stat" << std::endl; std::exit(-1); }
        file_size_ = st.st_size;

        unsigned char buf[2];
        ssize_t n = ::read(fd_, buf, 2);
        is_gz_file = (n == 2 && buf[0] == 0x1F && buf[1] == 0x8B);
        if (lseek(fd_, 0, SEEK_SET) == -1) { std::cerr << "Failed to seek" << std::endl; std::exit(-1); }

        if (is_gz_file)
        {
            gzfile_ = gzopen(fname.c_str(), "rb");
            if (gzfile_ == nullptr) { std::cerr << "Failed to open gzip: " << fname << std::endl; std::exit(-1); }
        }
        else
        {
            posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL);
        }
    }

    void close_current_file()
    {
        if (gzfile_ != nullptr) { gzclose(gzfile_); gzfile_ = nullptr; }
        if (fd_ != -1) { ::close(fd_); fd_ = -1; }
        file_size_ = 0;
    }

    // 统一 I/O 线程: gzip → gzread, 非 gzip → ::read
    void io_thread_func()
    {
        while (true)
        {
            char* buf = nullptr;
            pipeline_free_queue_.dequeue(buf);

            ssize_t bytes_read;
            if (is_gz_file)
            {
                bytes_read = gzread(gzfile_, buf, static_cast<unsigned int>(chunk_size_));
                if (bytes_read > 0)
                    have_read_.store(gzoffset(gzfile_), std::memory_order_relaxed);
            }
            else
            {
                bytes_read = ::read(fd_, buf, chunk_size_);
                if (bytes_read > 0)
                    have_read_.fetch_add(static_cast<off_t>(bytes_read), std::memory_order_relaxed);
            }

            if (bytes_read < 0) [[unlikely]]
            {
                std::cerr << "Failed to read fastq data in I/O thread" << std::endl;
                std::exit(-1);
            }
            if (bytes_read == 0)
            {
                pipeline_free_queue_.enqueue(buf);
                pipeline_data_queue_.enqueue({ nullptr, 0 });
                break;
            }
            pipeline_data_queue_.enqueue({ buf, static_cast<uint64_t>(bytes_read) });
        }
    }

    // 统一解析循环: 始终从 pipeline queue dequeue
    void parse_loop(::content_type& current_input, bool& have_input)
    {
        assert(ring_memory_pool_ptr_ != nullptr);
        char* block_ptr = nullptr;
        const uint64_t block_size = ring_memory_pool_ptr_->blockSize();

        const uint64_t overlap = (k > 1) ? static_cast<uint64_t>(k - 1) : 0;
        uint64_t write_size = 0;
        uint64_t last_newline_pos = kNoNewlineInBlock;
        bool has_block = false;
        uint64_t input_pos = 0, input_size = 0;
        bool eof = false, stop = false;
        double cur_percent = 0.0;
        int last_reported_percent = 0;

        while (!stop)
        {
            if (!has_block)
                acquire_block(block_ptr, write_size, last_newline_pos, has_block);

            if (input_pos >= input_size && !eof)
            {
                // 归还旧 buffer，从 I/O 线程取新数据
                if (have_input && current_input.data != nullptr)
                {
                    pipeline_free_queue_.enqueue(current_input.data);
                    have_input = false;
                }
                pipeline_data_queue_.dequeue(current_input);
                if (current_input.data == nullptr) eof = true;
                else
                {
                    have_input = true;
                    file_buffer = current_input.data;
                    input_pos = 0;
                    input_size = current_input.length;
                    off_t off = have_read_.load(std::memory_order_relaxed);
                    if (file_size_ > 0)
                        cur_percent = static_cast<double>(off) / static_cast<double>(file_size_);
                }

                if (!eof)
                {
                    int pct = static_cast<int>(cur_percent * 100);
                    if (pct - last_reported_percent >= 1 || pct == 100)
                        std::cout << "\rFastqReader progress: " << pct << "%";
                    if (pct >= 100) [[unlikely]] std::cout << std::endl;
                    last_reported_percent = pct;
                }
            }

            if (input_pos >= input_size) { if (eof) { publish_current_block(block_ptr, write_size, last_newline_pos, has_block); break; } continue; }

            const char* input_begin = file_buffer;
            if (state_ != State::ReadSequence)
            {
                const char* cur = input_begin + input_pos;
                const uint64_t remain = input_size - input_pos;
                const void* nl = std::memchr(cur, '\n', remain);
                if (nl == nullptr) { input_pos = input_size; continue; }
                input_pos = static_cast<uint64_t>(static_cast<const char*>(nl) - input_begin) + 1;
                state_ = advance_state_on_newline(state_);
                continue;
            }

            const char* seq_cur = input_begin + input_pos;
            const uint64_t seq_remain = input_size - input_pos;
            const void* nl = std::memchr(seq_cur, '\n', seq_remain);
            const uint64_t seq_len = (nl == nullptr) ? seq_remain
                : static_cast<uint64_t>(static_cast<const char*>(nl) - seq_cur);

            uint64_t copied = 0;
            while (copied < seq_len)
            {
                if (write_size == block_size) { store_overlap_from_block_end(block_ptr, write_size, last_newline_pos, overlap); publish_current_block(block_ptr, write_size, last_newline_pos, has_block); acquire_block(block_ptr, write_size, last_newline_pos, has_block); }
                const uint64_t rem = block_size - write_size;
                const uint64_t tc = (seq_len - copied < rem) ? (seq_len - copied) : rem;
                std::memcpy(block_ptr + write_size, seq_cur + copied, tc);
                write_size += tc; copied += tc; input_pos += tc;
            }
            if (stop) break;
            if (nl == nullptr) continue;
            if (write_size == block_size) { left_buffer_size_ = 0; publish_current_block(block_ptr, write_size, last_newline_pos, has_block); acquire_block(block_ptr, write_size, last_newline_pos, has_block); }
            last_newline_pos = write_size;
            block_ptr[write_size++] = '\n'; ++input_pos;
            state_ = advance_state_on_newline(state_);
        }
    }

    State advance_state_on_newline(const State current) const
    {
        switch (current) {
        case State::ReadHeader: return State::ReadSequence;
        case State::ReadSequence: return State::ReadPlus;
        case State::ReadPlus: return State::ReadQuality;
        case State::ReadQuality: return State::ReadHeader;
        default: return State::ReadHeader;
        }
    }

    void acquire_block(char*& block_ptr, uint64_t& write_size, uint64_t& last_newline_pos, bool& has_block)
    {
        ring_memory_pool_ptr_->producer_dequeue(block_ptr);
        has_block = true; write_size = 0; last_newline_pos = kNoNewlineInBlock;
        if (left_buffer_size_ > 0) { std::memcpy(block_ptr, left_buffer_, left_buffer_size_); write_size = left_buffer_size_; left_buffer_size_ = 0; }
    }

    inline void store_overlap_from_block_end(const char* block_ptr, uint64_t write_size, uint64_t last_newline_pos, uint64_t overlap)
    {
        left_buffer_size_ = 0;
        if (overlap == 0 || write_size == 0) return;
        const uint64_t keep = (write_size < overlap) ? write_size : overlap;
        const uint64_t start = write_size - keep;
        if (last_newline_pos != kNoNewlineInBlock && last_newline_pos >= start) {
            const uint64_t kp = write_size - (last_newline_pos + 1);
            if (kp > 0) { std::memcpy(left_buffer_, block_ptr + last_newline_pos + 1, kp); left_buffer_size_ = static_cast<size_t>(kp); }
            return;
        }
        std::memcpy(left_buffer_, block_ptr + start, keep);
        left_buffer_size_ = static_cast<size_t>(keep);
    }

    inline void publish_current_block(char*& block_ptr, uint64_t& write_size, uint64_t& last_newline_pos, bool& has_block)
    {
        if (!has_block) return;
        if (write_size > 0) ring_memory_pool_ptr_->producer_enqueue({ block_ptr, write_size });
        else ring_memory_pool_ptr_->consumer_enqueue(block_ptr);
        has_block = false; block_ptr = nullptr; write_size = 0; last_newline_pos = kNoNewlineInBlock;
    }
};

template <uint32_t N>
class ReaderThreadPool
{
    using content_type = std::pair<char*, uint64_t>;

    enum class State
    {
        ReadHeader,
        ReadSequence,
        ReadPlus,
        ReadQuality
    };

    static constexpr uint64_t kNoNewlineInBlock = static_cast<uint64_t>(-1);
    static constexpr uint64_t GZ_CHUNK_SIZE = 512 * 1024; // 512 KB

    int k_;
    uint64_t base_chunk_size_;
    RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>* ring_pool_ptr_;

    std::vector<std::string> gz_files_;
    std::vector<std::string> plain_files_;
    uint32_t reader_count_;
    std::vector<std::unique_ptr<std::thread>> threads_;
    std::vector<std::string> assignments;

    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> file_index_{ 0 };

    inline static thread_local SpinBackoff<> enqueue_backoff;
    inline static thread_local SpinBackoff<> dequeue_backoff;

public:
    explicit ReaderThreadPool(
        const std::vector<std::string>& filenames,
        int in_k,
        uint64_t chunk_size,
        RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>* pool_ptr)
        : k_(in_k), base_chunk_size_(chunk_size), ring_pool_ptr_(pool_ptr)
    {
        assert(ring_pool_ptr_ != nullptr);
        assert(k_ > 0 && k_ < 128);
        assert(ring_pool_ptr_->blockSize() > 1024);
        if (filenames.empty()) { std::cerr << "No input files" << std::endl; std::exit(-1); }

        classify_files(filenames);
        sort_files_by_size();

        reader_count_ = (gz_files_.size() >= 2) ? 2 : 1;
        threads_.reserve(reader_count_);
    }

    void start()
    {
        file_index_.store(0, std::memory_order_relaxed);
        assignments.clear();
        for (size_t i = 0; i < gz_files_.size(); ++i)
            assignments.push_back(gz_files_[i]);
        for (size_t i = 0; i < plain_files_.size(); ++i)
            assignments.push_back(plain_files_[i]);

        for (uint32_t i = 0; i < reader_count_; ++i)
        {
            threads_.push_back(std::make_unique<std::thread>([this]() {
                reader_worker();
                }));
        }
    }

    void join()
    {
        for (auto& t : threads_)
        {
            if (t->joinable()) t->join();
        }
    }

private:
    void classify_files(const std::vector<std::string>& filenames)
    {
        for (const auto& f : filenames)
        {
            if (f.size() >= 3 && f.compare(f.size() - 3, 3, ".gz") == 0)
                gz_files_.push_back(f);
            else
                plain_files_.push_back(f);
        }
    }

    void sort_files_by_size()
    {
        struct file_info {
            std::string filename;
            uint64_t size;
        };
        std::vector<file_info> infos;

        for (const auto& gz_filename : gz_files_)
        {
            struct stat st;
            uint64_t file_size = 0;
            if (stat(gz_filename.c_str(), &st) == -1) file_size = 0;
            else file_size = static_cast<uint64_t>(st.st_size);
            infos.push_back({ gz_filename, file_size });
        }

        std::sort(infos.begin(), infos.end(), [](const file_info& a, const file_info& b) {
            return a.size > b.size;
            });
        gz_files_.clear();
        for (const auto& info : infos)
        {
            gz_files_.push_back(info.filename);
        }

        infos.clear();
        for (const auto& plain_filename : plain_files_)
        {
            struct stat st;
            uint64_t file_size = 0;
            if (stat(plain_filename.c_str(), &st) == -1) file_size = 0;
            else file_size = static_cast<uint64_t>(st.st_size);
            infos.push_back({ plain_filename, file_size });
        }

        std::sort(infos.begin(), infos.end(), [](const file_info& a, const file_info& b) {
            return a.size > b.size;
            });
        plain_files_.clear();
        for (const auto& info : infos)
        {
            plain_files_.push_back(info.filename);
        }


    }

    static State advance_state(const State current)
    {
        switch (current) {
        case State::ReadHeader: return State::ReadSequence;
        case State::ReadSequence: return State::ReadPlus;
        case State::ReadPlus: return State::ReadQuality;
        case State::ReadQuality: return State::ReadHeader;
        default: return State::ReadHeader;
        }
    }

    inline void acquire_block(char*& block_ptr, uint64_t& write_size, uint64_t& last_newline_pos,
        bool& has_block, size_t& left_buffer_size_, char* left_buffer_)
    {
        if (ring_pool_ptr_->producer_try_dequeue(block_ptr))
        {
            dequeue_backoff.double_decay();
        }
        else
        {
            while (!ring_pool_ptr_->producer_try_dequeue(block_ptr))
            {
                dequeue_backoff.backoff();
            }
            dequeue_backoff.decay();
        }


        has_block = true; write_size = 0; last_newline_pos = kNoNewlineInBlock;
        if (left_buffer_size_ > 0)
        {
            std::memcpy(block_ptr, left_buffer_, left_buffer_size_);
            write_size = left_buffer_size_;
            left_buffer_size_ = 0;
        }
    }

    inline void publish_current_block(char*& block_ptr, uint64_t& write_size,
        uint64_t& last_newline_pos, bool& has_block)
    {
        if (!has_block) return;

        if (write_size > 0) {
            if (ring_pool_ptr_->producer_try_enqueue({ block_ptr, write_size }))
            {
                enqueue_backoff.double_decay();
            }
            else
            {
                while (!ring_pool_ptr_->producer_try_enqueue({ block_ptr, write_size }))
                {
                    enqueue_backoff.backoff();
                }
                enqueue_backoff.decay();
            }
        }
        else ring_pool_ptr_->consumer_enqueue(block_ptr);

        has_block = false;
        block_ptr = nullptr;
        write_size = 0;
        last_newline_pos = kNoNewlineInBlock;
    }

    inline void store_overlap_from_block_end(const char* block_ptr, uint64_t write_size,
        uint64_t last_newline_pos, uint64_t overlap,
        size_t& left_buffer_size_, char* left_buffer_)
    {
        left_buffer_size_ = 0;
        if (overlap == 0 || write_size == 0) return;
        const uint64_t keep = (write_size < overlap) ? write_size : overlap;
        const uint64_t start = write_size - keep;
        if (last_newline_pos != kNoNewlineInBlock && last_newline_pos >= start) {
            const uint64_t kp = write_size - (last_newline_pos + 1);
            if (kp > 0) { std::memcpy(left_buffer_, block_ptr + last_newline_pos + 1, kp); left_buffer_size_ = static_cast<size_t>(kp); }
            return;
        }
        std::memcpy(left_buffer_, block_ptr + start, keep);
        left_buffer_size_ = static_cast<size_t>(keep);
    }

    void reader_worker()
    {
        assert(ring_pool_ptr_ != nullptr);
        const uint64_t block_size = ring_pool_ptr_->blockSize();

        State state_;
        char left_buffer_[128];
        size_t left_buffer_size_ = 0;
        const uint64_t overlap = (k_ > 1) ? static_cast<uint64_t>(k_ - 1) : 0;

        GzipStreamer gzip_streamer;

        uint64_t cur_file_index = file_index_.fetch_add(1, std::memory_order_relaxed);
        for (; cur_file_index < assignments.size(); cur_file_index = file_index_.fetch_add(1, std::memory_order_relaxed))
        {
            std::string file = assignments[cur_file_index];
            const bool is_gz = (file.size() >= 3 && file.compare(file.size() - 3, 3, ".gz") == 0);
            const uint64_t effective_chunk_size = is_gz ? GZ_CHUNK_SIZE : base_chunk_size_;

            int fd = -1;
            gzFile gzfile = nullptr;

            if (is_gz)
            {
                // gzip_streamer.open(file);
                gzfile = gzopen(file.c_str(), "rb");
                if (gzfile == nullptr) { std::cerr << "Failed to open gzip: " << file << std::endl; std::exit(-1); }
                gzbuffer(gzfile, GZ_CHUNK_SIZE / 2);
            }
            else
            {
                fd = ::open(file.c_str(), O_RDONLY);
                if (fd == -1) { std::cerr << "Failed to open: " << file << std::endl; std::exit(-1); }
                posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
            }

            std::vector<char> read_buf(effective_chunk_size);

            state_ = State::ReadHeader;
            left_buffer_size_ = 0;
            char* block_ptr = nullptr;
            uint64_t write_size = 0;
            bool has_block = false;
            uint64_t last_newline_pos = kNoNewlineInBlock;

            uint64_t input_pos = 0, input_size = 0;
            bool eof = false;
            char* input_begin = read_buf.data();

            while (true)
            {
                // char* input_begin = read_buf.data();
                if (input_pos >= input_size && !eof)
                {
                    ssize_t bytes_read;
                    if (is_gz)
                    {
                        bytes_read = gzread(gzfile, read_buf.data(), static_cast<unsigned int>(effective_chunk_size));
                        input_begin = read_buf.data();
                        // uint8_t* gzip_input_data = nullptr;
                        // size_t gizp_bytes_read = 0;
                        // gzip_streamer.next(gzip_input_data, gizp_bytes_read);
                        // input_begin = reinterpret_cast<char*>(gzip_input_data);
                        // bytes_read = static_cast<ssize_t>(gizp_bytes_read);
                    }
                    else
                    {
                        bytes_read = ::read(fd, read_buf.data(), effective_chunk_size);
                        input_begin = read_buf.data();
                    }


                    if (bytes_read < 0) [[unlikely]]
                    {
                        std::cerr << "Failed to read fastq data: " << file << std::endl;
                        std::exit(-1);
                    }
                    if (bytes_read <= 0)
                    {
                        eof = true;
                    }
                    else
                    {
                        input_pos = 0;
                        input_size = static_cast<uint64_t>(bytes_read);
                    }
                }

                if (!has_block)
                    acquire_block(block_ptr, write_size, last_newline_pos, has_block, left_buffer_size_, left_buffer_);

                if (input_pos >= input_size)
                {
                    if (eof) { publish_current_block(block_ptr, write_size, last_newline_pos, has_block); break; }
                    continue;
                }

                // const char* input_begin = read_buf.data();
                if (state_ != State::ReadSequence)
                {
                    const char* cur = input_begin + input_pos;
                    const uint64_t remain = input_size - input_pos;
                    const void* nl = std::memchr(cur, '\n', remain);
                    if (nl == nullptr) { input_pos = input_size; continue; }
                    input_pos = static_cast<uint64_t>(static_cast<const char*>(nl) - input_begin) + 1;
                    state_ = advance_state(state_);
                    continue;
                }

                const char* seq_cur = input_begin + input_pos;
                const uint64_t seq_remain = input_size - input_pos;
                const void* nl = std::memchr(seq_cur, '\n', seq_remain);
                const uint64_t seq_len = (nl == nullptr) ? seq_remain
                    : static_cast<uint64_t>(static_cast<const char*>(nl) - seq_cur);

                uint64_t copied = 0;
                while (copied < seq_len)
                {
                    if (write_size == block_size) { store_overlap_from_block_end(block_ptr, write_size, last_newline_pos, overlap, left_buffer_size_, left_buffer_); publish_current_block(block_ptr, write_size, last_newline_pos, has_block); acquire_block(block_ptr, write_size, last_newline_pos, has_block, left_buffer_size_, left_buffer_); }
                    const uint64_t rem = block_size - write_size;
                    const uint64_t tc = (seq_len - copied < rem) ? (seq_len - copied) : rem;
                    std::memcpy(block_ptr + write_size, seq_cur + copied, tc);
                    write_size += tc; copied += tc; input_pos += tc;
                }
                if (nl == nullptr) continue;
                if (write_size == block_size) { left_buffer_size_ = 0; publish_current_block(block_ptr, write_size, last_newline_pos, has_block); acquire_block(block_ptr, write_size, last_newline_pos, has_block, left_buffer_size_, left_buffer_); }
                last_newline_pos = write_size;
                block_ptr[write_size++] = '\n'; ++input_pos;
                state_ = advance_state(state_);
            }

            // if (is_gz) gzip_streamer.close();
            if(is_gz) gzclose(gzfile);
            else ::close(fd);

            std::cout << "FastqReader: completed " << file << std::endl;
        }

        ring_pool_ptr_->producer_set_finished();
    }
};

#endif