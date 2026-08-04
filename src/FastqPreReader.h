#ifndef FASTQ_PRE_READER_HEADER
#define FASTQ_PRE_READER_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "SPSCRingQueue.h"

#include <string>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <utility>
#include <iostream>
#include <algorithm>
#include <zlib.h>
#include <thread>
#include <atomic>

template <uint32_t N>
class FastqPreReader
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
    static constexpr uint64_t PIPELINE_QUEUE_CAPACITY = 16;

    State state_ = State::ReadHeader;
    int fd_ = -1;
    ssize_t file_size_ = 0;
    std::atomic<ssize_t> have_read_{0};
    ssize_t need_read_ = 0;
    uint64_t chunk_size_;
    std::vector<std::string> filenames_;
    size_t file_index_ = 0;
    RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>* ring_memory_pool_ptr_;
    char* file_buffer;
    char left_buffer_[128];
    size_t left_buffer_size_ = 0;
    bool is_gz_file = false;
    gzFile gzfile_ = nullptr;
    uint64_t quality_sum_   = 0;
    uint64_t quality_count_ = 0;

    std::vector<char*> pipeline_buffers_;
    SPSCRingQueue<::content_type, PIPELINE_QUEUE_CAPACITY> pipeline_data_queue_;
    SPSCRingQueue<char*, PIPELINE_QUEUE_CAPACITY> pipeline_free_queue_;
    std::thread io_thread_;

public:
    int k;

    explicit FastqPreReader(const std::vector<std::string>& filenames, const int in_k, uint64_t chunk_size, RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>* in_ring_memory_pool_ptr)
        : filenames_(filenames), ring_memory_pool_ptr_(in_ring_memory_pool_ptr),
        k(in_k), chunk_size_(chunk_size)
    {
        assert(ring_memory_pool_ptr_ != nullptr);
        assert(k > 0 && k < 128);
        assert(ring_memory_pool_ptr_->blockSize() > 1024);
        if (filenames_.empty()) { std::cerr << "No input files" << std::endl; std::exit(-1); }

        pipeline_buffers_.resize(PIPELINE_QUEUE_CAPACITY);
        for (uint64_t i = 0; i < PIPELINE_QUEUE_CAPACITY; ++i)
        {
            pipeline_buffers_[i] = static_cast<char*>(std::aligned_alloc(4096, chunk_size_));
            if (!pipeline_buffers_[i]) { std::cerr << "Failed to allocate pipeline buffer" << std::endl; std::exit(-1); }
            pipeline_free_queue_.enqueue(pipeline_buffers_[i]);
        }

        // need_read_ 在 pre_read() 中按文件数动态设置
        file_buffer = new char[chunk_size_];
    }

    ~FastqPreReader()
    {
        if (io_thread_.joinable()) io_thread_.join();
        if (gzfile_ != nullptr) gzclose(gzfile_);
        if (fd_ != -1) ::close(fd_);
        delete[] file_buffer;
        for (char* buf : pipeline_buffers_) if (buf) std::free(buf);
    }

    void pre_read()
    {
        const ssize_t per_file_limit = (filenames_.size() == 1)
            ? static_cast<ssize_t>(256U * 1024 * 1024)
            : static_cast<ssize_t>(128U * 1024 * 1024);

        for (file_index_ = 0; file_index_ < filenames_.size(); ++file_index_)
        {
            open_current_file();
            need_read_ = std::min(file_size_, per_file_limit);
            have_read_.store(0, std::memory_order_relaxed);

            if (file_buffer) { delete[] file_buffer; file_buffer = nullptr; }
            io_thread_ = std::thread(&FastqPreReader::io_thread_func, this);

            ::content_type current_input{nullptr, 0};
            bool have_input = false;
            parse_loop(current_input, have_input);

            io_thread_.join();
            if (have_input && current_input.data != nullptr)
                pipeline_free_queue_.enqueue(current_input.data);
            file_buffer = nullptr;

            close_current_file();
            state_ = State::ReadHeader;
            left_buffer_size_ = 0;
        }

        if (quality_count_ > 0) avgQuality = static_cast<uint8_t>(quality_sum_ / quality_count_);
        ring_memory_pool_ptr_->producer_set_finished();
    }

    uint64_t get_estimated_raw_fastq_file_size() const noexcept
    {
        uint64_t total = 0;
        for (const auto& f : filenames_)
        {
            int fd = ::open(f.data(), O_RDONLY);
            if (fd < 0) continue;
            struct stat st;
            if (::fstat(fd, &st) == 0)
            {
                unsigned char buf[2];
                ssize_t n = ::read(fd, buf, 2);
                total += (n == 2 && buf[0] == 0x1F && buf[1] == 0x8B)
                    ? static_cast<uint64_t>(st.st_size) * 4 : static_cast<uint64_t>(st.st_size);
            }
            ::close(fd);
        }
        return (total > 0) ? total : 1;
    }

private:
    void open_current_file()
    {
        const std::string& fname = filenames_[file_index_];
        std::cout << "FastqPreReader: " << (file_index_ + 1) << "/"
                  << filenames_.size() << ": " << fname << std::endl;
        fd_ = ::open(fname.data(), O_RDONLY);
        if (fd_ == -1) { std::cerr << "Failed to open" << std::endl; std::exit(-1); }
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
            if (gzfile_ == nullptr) { std::cerr << "Failed to open gzip" << std::endl; std::exit(-1); }
        }
        else
        {
            posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
        }
    }

    void close_current_file()
    {
        if (gzfile_ != nullptr) { gzclose(gzfile_); gzfile_ = nullptr; }
        if (fd_ != -1) { ::close(fd_); fd_ = -1; }
        file_size_ = 0;
    }

    void io_thread_func()
    {
        while (true)
        {
            char* buf = nullptr;
            pipeline_free_queue_.dequeue(buf);

            ssize_t bytes_read;
            if (is_gz_file)
                bytes_read = gzread(gzfile_, buf, static_cast<unsigned int>(chunk_size_));
            else
                bytes_read = ::read(fd_, buf, chunk_size_);

            if (bytes_read < 0) [[unlikely]] { std::cerr << "Failed to read" << std::endl; std::exit(-1); }
            if (bytes_read == 0)
            {
                pipeline_free_queue_.enqueue(buf);
                pipeline_data_queue_.enqueue({nullptr, 0});
                break;
            }

            have_read_.fetch_add(static_cast<ssize_t>(bytes_read), std::memory_order_relaxed);
            pipeline_data_queue_.enqueue({buf, static_cast<uint64_t>(bytes_read)});

            if (have_read_.load(std::memory_order_relaxed) >= need_read_)
            {
                pipeline_data_queue_.enqueue({nullptr, 0});
                break;
            }
        }
    }

    void parse_loop(::content_type& current_input, bool& have_input)
    {
        assert(ring_memory_pool_ptr_ != nullptr);
        char* block_ptr = nullptr;
        const uint64_t block_size = ring_memory_pool_ptr_->blockSize();

        const uint64_t overlap = (k > 1) ? static_cast<uint64_t>(k - 1) : 0;
        uint64_t write_size = 0, last_newline_pos = kNoNewlineInBlock;
        bool has_block = false;
        uint64_t input_pos = 0, input_size = 0;
        bool eof = false, stop = false;

        while (!stop)
        {
            if (!has_block) acquire_block(block_ptr, write_size, last_newline_pos, has_block);

            if (input_pos >= input_size && !eof)
            {
                if (have_input && current_input.data != nullptr)
                {
                    pipeline_free_queue_.enqueue(current_input.data);
                    have_input = false;
                }
                pipeline_data_queue_.dequeue(current_input);
                if (current_input.data == nullptr) eof = true;
                else { have_input = true; file_buffer = current_input.data; input_pos = 0; input_size = current_input.length; }
            }

            if (input_pos >= input_size) { if (eof) { publish_current_block(block_ptr, write_size, last_newline_pos, has_block); break; } continue; }

            const char* input_begin = file_buffer;
            if (state_ != State::ReadSequence)
            {
                const char* cur = input_begin + input_pos;
                const uint64_t remain = input_size - input_pos;
                const void* nl = std::memchr(cur, '\n', remain);
                if (nl == nullptr) { input_pos = input_size; continue; }
                const char* newline = static_cast<const char*>(nl);
                if (state_ == State::ReadQuality)
                {
                    for (const char* p = cur; p < newline; ++p) quality_sum_ += static_cast<uint64_t>(*p);
                    quality_count_ += static_cast<uint64_t>(newline - cur);
                }
                input_pos = static_cast<uint64_t>(newline - input_begin) + 1;
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

    off_t find_next_record_start(off_t start_pos)
    {
        if (lseek(fd_, start_pos, SEEK_SET) == -1) return start_pos;
        char buf[65536];
        ssize_t n = ::read(fd_, buf, sizeof(buf));
        if (n <= 1) { lseek(fd_, start_pos, SEEK_SET); return start_pos; }
        for (ssize_t i = 1; i < n; ++i)
            if (buf[i - 1] == '\n' && buf[i] == '@') { off_t a = start_pos + static_cast<off_t>(i); lseek(fd_, a, SEEK_SET); return a; }
        lseek(fd_, start_pos, SEEK_SET);
        return start_pos;
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

#endif