#pragma once

#include "HistogramCommon.h"
#include <aio.h>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <memory>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

// Double-buffered raw input. Only complete, unchanged on-disk records are
// copied to the caller; decoding belongs to the consumer threads.
class CompactRecordReader
{
public:
    static constexpr uint64_t RAW_BUFFER_SIZE = 512 * 1024;
    static constexpr uint64_t BUFFER_BYTES = 2 * RAW_BUFFER_SIZE;

    CompactRecordReader(uint32_t k_len, uint32_t count_bytes)
        : record_bytes_(RecordLayout(k_len, count_bytes).record_bytes) {}
    CompactRecordReader(const CompactRecordReader&) = delete;
    CompactRecordReader& operator=(const CompactRecordReader&) = delete;
    ~CompactRecordReader() { close(); }

    void open(const std::string& filename, uint64_t first_record = 0,
              uint64_t record_count = std::numeric_limits<uint64_t>::max())
    {
        close();
        fd_ = ::open(filename.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("failed to open " + filename + ": " + std::strerror(errno));
        try
        {
            struct stat st{};
            if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
                throw std::runtime_error("failed to stat regular input " + filename);
            uint64_t bytes = static_cast<uint64_t>(st.st_size);
            if (bytes % record_bytes_)
                throw std::runtime_error("misaligned compact input " + filename);
            uint64_t records = bytes / record_bytes_;
            if (first_record > records || (record_count != std::numeric_limits<uint64_t>::max() &&
                                          record_count > records - first_record))
                throw std::runtime_error("invalid compact record range");
            first_record_ = first_record;
            record_count_ = record_count == std::numeric_limits<uint64_t>::max()
                ? records - first_record : record_count;
            ::posix_fadvise(fd_, static_cast<off_t>(first_record * record_bytes_),
                            static_cast<off_t>(record_count_ * record_bytes_), POSIX_FADV_SEQUENTIAL);
            if (record_count_)
            {
                for (auto& buffer : buffers_) buffer = std::make_unique<char[]>(RAW_BUFFER_SIZE);
                schedule(0);
            }
        }
        catch (...)
        {
            close();
            throw;
        }
    }

    void close() noexcept
    {
        for (uint32_t i = 0; i < 2; ++i)
        {
            if (active_[i])
            {
                wait(i);
                ::aio_return(&cbs_[i]);
                active_[i] = false;
            }
            buffers_[i].reset();
            available_[i] = cursor_[i] = 0;
        }
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        first_record_ = record_count_ = scheduled_ = delivered_ = 0;
        current_ = 0;
    }

    [[nodiscard]] bool finished() const noexcept { return delivered_ >= record_count_; }

    uint64_t read_compact_records(char* output, uint64_t max_records)
    {
        uint64_t total = 0;
        while (total < max_records && !finished())
        {
            uint32_t i = current_;
            if (active_[i])
            {
                wait(i);
                int error = ::aio_error(&cbs_[i]);
                ssize_t bytes = ::aio_return(&cbs_[i]);
                active_[i] = false;
                if (error || bytes != static_cast<ssize_t>(cbs_[i].aio_nbytes))
                    throw std::runtime_error("failed or partial compact input read");
                schedule(1 - i);
            }
            uint64_t count = std::min(max_records - total, available_[i] - cursor_[i]);
            std::memcpy(output + total * record_bytes_, buffers_[i].get() + cursor_[i] * record_bytes_,
                        static_cast<size_t>(count * record_bytes_));
            total += count;
            delivered_ += count;
            cursor_[i] += count;
            if (cursor_[i] == available_[i])
            {
                available_[i] = cursor_[i] = 0;
                current_ = 1 - i;
            }
        }
        return total;
    }

private:
    void wait(uint32_t i) noexcept
    {
        const aiocb* list[] = {&cbs_[i]};
        while (::aio_error(&cbs_[i]) == EINPROGRESS)
            if (::aio_suspend(list, 1, nullptr) != 0 && errno != EINTR) std::this_thread::yield();
    }

    void schedule(uint32_t i)
    {
        if (scheduled_ == record_count_ || active_[i]) return;
        uint64_t count = std::min(RAW_BUFFER_SIZE / record_bytes_, record_count_ - scheduled_);
        std::memset(&cbs_[i], 0, sizeof(aiocb));
        cbs_[i].aio_fildes = fd_;
        cbs_[i].aio_buf = buffers_[i].get();
        cbs_[i].aio_nbytes = count * record_bytes_;
        cbs_[i].aio_offset = static_cast<off_t>((first_record_ + scheduled_) * record_bytes_);
        if (::aio_read(&cbs_[i]) != 0)
            throw std::runtime_error(std::string("aio_read failed: ") + std::strerror(errno));
        active_[i] = true;
        available_[i] = count;
        cursor_[i] = 0;
        scheduled_ += count;
    }

    int fd_ = -1;
    uint64_t record_bytes_;
    uint64_t first_record_ = 0, record_count_ = 0, scheduled_ = 0, delivered_ = 0;
    uint32_t current_ = 0;
    std::array<aiocb, 2> cbs_{};
    std::array<std::unique_ptr<char[]>, 2> buffers_{};
    std::array<bool, 2> active_{};
    std::array<uint64_t, 2> available_{}, cursor_{};
};
