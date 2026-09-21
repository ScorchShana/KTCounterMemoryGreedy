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
    // 压缩数据每次从磁盘读取的块大小（建议 16~64 MiB）
    static constexpr size_t COMPRESSED_CHUNK_SIZE = 4ULL * 1024 * 1024;  // 4 MB

    // 解压后输出缓冲区大小
    static constexpr size_t DECOMPRESSED_BUF_SIZE = 256ULL * 1024;   // 256 KB

    // 缓存行大小（现代 x86_64 / ARM64 几乎都是 64）
    static constexpr size_t CACHE_LINE_SIZE = 64;

    GzipStreamer()
    {
        allocate_buffers();
    }

    // 禁止拷贝
    GzipStreamer(const GzipStreamer&) = delete;
    GzipStreamer& operator=(const GzipStreamer&) = delete;

    ~GzipStreamer()
    {
        close();
        free_buffers();
    }

    /**
     * 打开文件并做顺序读取优化
     */
    void open(const char* filename) {
        close();  // 先关闭已打开的

        fd_ = ::open(filename, O_RDONLY | O_CLOEXEC);
        if (fd_ < 0)
        {
            std::cerr << "GzipStreamer: open failed: " << ::strerror(errno) << std::endl;
            std::exit(-1);
        }

        // 告诉内核我们会顺序访问整个文件（增大 readahead 窗口）
        if (::posix_fadvise(fd_, 0, 0, POSIX_FADV_SEQUENTIAL) != 0)
        {
            // 非致命，仅警告
            std::cerr << "GzipStreamer: posix_fadvise(POSIX_FADV_SEQUENTIAL) failed: "
                << ::strerror(errno) << std::endl;
        }

        // 初始化 z_stream
        std::memset(&stream_, 0, sizeof(stream_));
        if (inflateInit2(&stream_, 31) != Z_OK)
        {  // 31 = 支持 gzip header
            std::cerr << "GzipStreamer: inflateInit2 failed" << std::endl;
            ::close(fd_);
            fd_ = -1;
            std::exit(-1);
        }

        stream_initialized_ = true;
        eof_ = false;
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
        eof_ = false;
    }

    /**
     * 获取下一块解压数据
     * @param out_data  输出数据指针（指向内部缓冲，下次调用会失效）
     * @param out_size  输出数据长度
     * @return true 还有数据，false 文件结束
     */
    bool next(uint8_t*& out_data, size_t& out_size) {
        out_data = nullptr;
        out_size = 0;

        if (fd_ < 0 || !stream_initialized_ || eof_)
        {
            return false;
        }

        stream_.next_out = decompressed_buf_;
        stream_.avail_out = static_cast<uInt>(DECOMPRESSED_BUF_SIZE);

        while (true)
        {
            // 输入耗尽时，从磁盘再读一大块压缩数据
            if (stream_.avail_in == 0 && !eof_)
            {
                ssize_t n = ::read(fd_, compressed_buf_, COMPRESSED_CHUNK_SIZE);
                if (n < 0)
                {
                    if (errno == EINTR) continue;  // 被信号打断，重试
                    std::cerr << "GzipStreamer: read failed: " << strerror(errno) << std::endl;
                    std::exit(-1);
                }
                if (n == 0)
                {
                    eof_ = true;
                }
                else
                {
                    stream_.next_in = compressed_buf_;
                    stream_.avail_in = static_cast<uInt>(n);
                }
            }

            int ret = inflate(&stream_, Z_NO_FLUSH);

            if (ret == Z_STREAM_END)
            {
                // 一个 gzip member 结束，尝试支持多流 gzip
                if (stream_.avail_in > 0 || !eof_)
                {
                    if (inflateReset(&stream_) != Z_OK)
                    {
                        std::cerr << "GzipStreamer: inflateReset failed" << std::endl;
                        std::exit(-1);
                    }
                    continue;
                }
                // 真正结束
                break;
            }

            if (ret != Z_OK && ret != Z_BUF_ERROR)
            {
                std::cerr << "GzipStreamer: inflate error code: " << ret << std::endl;
                std::exit(-1);
            }

            // 输出缓冲区满了，返回给调用者
            if (stream_.avail_out == 0)
            {
                break;
            }

            // 既没有输入也没有输出 → 结束
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

        out_data = decompressed_buf_;
        return true;
    }

    bool is_open() const { return fd_ >= 0; }

private:
    void allocate_buffers() {
        if (posix_memalign(reinterpret_cast<void**>(&compressed_buf_),
            CACHE_LINE_SIZE, COMPRESSED_CHUNK_SIZE) != 0) {
            std::cerr << "GzipStreamer: posix_memalign(compressed_buf) failed" << std::endl;
            std::exit(-1);
        }
        if (posix_memalign(reinterpret_cast<void**>(&decompressed_buf_),
            CACHE_LINE_SIZE, DECOMPRESSED_BUF_SIZE) != 0) {
            free(compressed_buf_);
            compressed_buf_ = nullptr;
            std::cerr << "GzipStreamer: posix_memalign(decompressed_buf) failed" << std::endl;
            std::exit(-1);
        }
    }

    void free_buffers() {
        if (compressed_buf_) {
            free(compressed_buf_);
            compressed_buf_ = nullptr;
        }
        if (decompressed_buf_) {
            free(decompressed_buf_);
            decompressed_buf_ = nullptr;
        }
    }

    int          fd_ = -1;
    z_stream     stream_{};
    bool         stream_initialized_ = false;
    bool         eof_ = false;

    uint8_t* compressed_buf_ = nullptr;
    uint8_t* decompressed_buf_ = nullptr;
};