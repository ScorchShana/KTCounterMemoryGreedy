#pragma once

#include "HistogramCommon.h"
#include "../src/KtpFormat.h"

#include <memory>
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Read complete raw records directly into the caller's buffer;
// decoding belongs to the consumer threads.
class CompactRecordReader
{
public:
    static constexpr uint64_t RAW_BUFFER_SIZE = 2 * 1024 * 1024; // Maximum pread request.
    static constexpr uint64_t BUFFER_BYTES = 0; // No internal read buffers.

    CompactRecordReader(uint32_t k_len, uint32_t count_bytes)
        : record_bytes_(RecordLayout(k_len, count_bytes).record_bytes), count_bytes_(count_bytes), k_len_(k_len) {}
    CompactRecordReader(const CompactRecordReader&) = delete;
    CompactRecordReader& operator=(const CompactRecordReader&) = delete;
    ~CompactRecordReader() { close(); }

    void open(const std::string& filename, uint64_t first_record = 0,
              uint64_t record_count = std::numeric_limits<uint64_t>::max())
    {
        close();
        if (ktp::is_ktp_file(filename)) {
            ktp_ = std::make_unique<ktp::Reader>();
            ktp_->open(filename);
            if (ktp_->k() != k_len_) throw std::runtime_error("KTP1 k mismatch");
            const uint64_t records = ktp_->records();
            if (first_record > records || (record_count != std::numeric_limits<uint64_t>::max() &&
                                          record_count > records - first_record))
                throw std::runtime_error("invalid compact record range");
            first_record_ = first_record;
            record_count_ = record_count == std::numeric_limits<uint64_t>::max()
                ? records - first_record : record_count;
            return;
        }
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
        }
        catch (...)
        {
            close();
            throw;
        }
    }

    void close() noexcept
    {
        ktp_.reset();
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
        first_record_ = record_count_ = delivered_ = 0;
    }

    [[nodiscard]] bool finished() const noexcept { return delivered_ >= record_count_; }

    uint64_t read_compact_records(char* output, uint64_t max_records)
    {
        uint64_t count = std::min(max_records, record_count_ - delivered_);
        if (ktp_) {
            char* cursor = output;
            const uint32_t key_bytes = ktp_->key_bytes();
            ktp_->decode_range(first_record_ + delivered_, count, [&](const uint8_t* key, uint32_t value) {
                std::memcpy(cursor, key, key_bytes);
                cursor += key_bytes;
                for (uint32_t b = 0; b < count_bytes_; ++b) *cursor++ = static_cast<char>((value >> (8 * b)) & 0xffu);
            });
            delivered_ += count;
            return count;
        }
        uint64_t bytes = count * record_bytes_;
        uint64_t offset = (first_record_ + delivered_) * record_bytes_;
        uint64_t completed = 0;
        while (completed < bytes)
        {
            ssize_t result = ::pread(fd_, output + completed,
                                     static_cast<size_t>(std::min(bytes - completed, RAW_BUFFER_SIZE)),
                                     static_cast<off_t>(offset + completed));
            if (result < 0)
            {
                if (errno == EINTR) continue;
                throw std::runtime_error(std::string("pread failed: ") + std::strerror(errno));
            }
            if (result == 0) throw std::runtime_error("unexpected end of compact input");
            completed += static_cast<uint64_t>(result);
        }
        delivered_ += count;
        return count;
    }

private:
    int fd_ = -1;
    uint64_t record_bytes_;
    uint32_t count_bytes_;
    uint32_t k_len_ = 0;
    std::unique_ptr<ktp::Reader> ktp_;
    uint64_t first_record_ = 0, record_count_ = 0, delivered_ = 0;
};
