#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <zlib.h>

class GzipStreamer {
public:
    static constexpr size_t COMPRESSED_CHUNK_SIZE = 1ULL * 1024 * 1024;
    static constexpr size_t DECOMPRESSED_BUF_SIZE = 256ULL * 1024;
    static constexpr size_t PAGE_ALIGN = 4096;

    GzipStreamer()
    {
        allocate_buffers();
    }

    GzipStreamer(const GzipStreamer&) = delete;
    GzipStreamer& operator=(const GzipStreamer&) = delete;

    ~GzipStreamer()
    {
        close();
        free_buffers();
    }

    void open(const char* filename)
    {
        close();

        fd_ = ::open(filename, O_RDONLY | O_CLOEXEC);
        if (fd_ < 0)
        {
            std::cerr << "GzipStreamer: open failed: " << ::strerror(errno) << std::endl;
            std::exit(-1);
        }

        if (::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL) != 0)
        {
            std::cerr << "GzipStreamer: posix_fadvise(POSIX_FADV_SEQUENTIAL) failed: "
                << ::strerror(errno) << std::endl;
        }

        std::memset(&stream_, 0, sizeof(stream_));
        if (inflateInit2(&stream_, 31) != Z_OK)
        {
            std::cerr << "GzipStreamer: inflateInit2 failed" << std::endl;
            ::close(fd_);
            fd_ = -1;
            std::exit(-1);
        }

        stream_initialized_ = true;
        eof_ = false;
        file_off_ = 0;
        got_output_ = false;
        ::readahead(fd_, 0, COMPRESSED_CHUNK_SIZE);
    }

    void open(const std::string& filename)
    {
        open(filename.c_str());
    }

    void close()
    {
        if (stream_initialized_)
        {
            inflateEnd(&stream_);
            stream_initialized_ = false;
        }
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
        stream_.next_in = nullptr;
        stream_.avail_in = 0;
        eof_ = false;
        file_off_ = 0;
        got_output_ = false;
    }

    /**
     * 获取下一块解压数据
     * @param out_data  输出数据指针（指向内部缓冲，下次调用会失效）
     * @param out_size  输出数据长度
     * @return true 还有数据，false 文件结束
     */
    bool next(uint8_t*& out_data, size_t& out_size)
    {
        out_data = nullptr;
        out_size = 0;

        if (fd_ < 0 || !stream_initialized_)
        {
            return false;
        }

        stream_.next_out = decompressed_buf_;
        stream_.avail_out = static_cast<uInt>(DECOMPRESSED_BUF_SIZE);

        while (true)
        {
            if (stream_.avail_in == 0 && !eof_)
            {
                feed_input();
            }

            if (stream_.avail_in == 0 && eof_)
            {
                break;
            }

            int ret = inflate(&stream_, Z_NO_FLUSH);

            if (ret == Z_STREAM_END)
            {
                if (stream_.avail_out == 0)
                {
                    break;
                }
                if (stream_.avail_in == 0 && !eof_)
                {
                    feed_input();
                }
                if (stream_.avail_in > 0)
                {
                    if (inflateReset(&stream_) != Z_OK)
                    {
                        std::cerr << "GzipStreamer: inflateReset failed" << std::endl;
                        std::exit(-1);
                    }
                    continue;
                }
                break;
            }

            if (ret == Z_DATA_ERROR && got_output_)
            {
                break;
            }

            if (ret != Z_OK && ret != Z_BUF_ERROR)
            {
                std::cerr << "GzipStreamer: inflate error code: " << ret << std::endl;
                std::exit(-1);
            }

            if (stream_.avail_out == 0)
            {
                break;
            }

            if (eof_ && stream_.avail_in == 0)
            {
                break;
            }
        }

        out_size = DECOMPRESSED_BUF_SIZE - stream_.avail_out;
        if (out_size == 0)
        {
            return false;
        }

        got_output_ = true;
        out_data = decompressed_buf_;
        return true;
    }

    bool is_open() const { return fd_ >= 0; }

private:
    bool feed_input()
    {
        ssize_t n;
        do
        {
            n = ::read(fd_, compressed_buf_, COMPRESSED_CHUNK_SIZE);
        } while (n < 0 && errno == EINTR);

        if (n < 0)
        {
            std::cerr << "GzipStreamer: read failed: " << ::strerror(errno) << std::endl;
            std::exit(-1);
        }
        if (n == 0)
        {
            eof_ = true;
            return false;
        }

        file_off_ += n;
        // ::readahead(fd_, file_off_, COMPRESSED_CHUNK_SIZE);
        stream_.next_in = compressed_buf_;
        stream_.avail_in = static_cast<uInt>(n);
        return true;
    }

    void allocate_buffers()
    {
        if (posix_memalign(reinterpret_cast<void**>(&compressed_buf_),
            PAGE_ALIGN, COMPRESSED_CHUNK_SIZE) != 0)
        {
            std::cerr << "GzipStreamer: posix_memalign(compressed_buf) failed" << std::endl;
            std::exit(-1);
        }
        if (posix_memalign(reinterpret_cast<void**>(&decompressed_buf_),
            PAGE_ALIGN, DECOMPRESSED_BUF_SIZE) != 0)
        {
            free(compressed_buf_);
            compressed_buf_ = nullptr;
            std::cerr << "GzipStreamer: posix_memalign(decompressed_buf) failed" << std::endl;
            std::exit(-1);
        }
    }

    void free_buffers()
    {
        if (compressed_buf_)
        {
            free(compressed_buf_);
            compressed_buf_ = nullptr;
        }
        if (decompressed_buf_)
        {
            free(decompressed_buf_);
            decompressed_buf_ = nullptr;
        }
    }

    int          fd_ = -1;
    z_stream     stream_{};
    bool         stream_initialized_ = false;
    bool         eof_ = false;
    bool         got_output_ = false;
    off_t        file_off_ = 0;

    uint8_t* compressed_buf_ = nullptr;
    uint8_t* decompressed_buf_ = nullptr;
};
