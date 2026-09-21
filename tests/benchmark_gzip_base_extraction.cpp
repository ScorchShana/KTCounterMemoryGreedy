// 单线程基准测试: 对比 GzipStreamer 与 gzread 读取 gzip 压缩 FASTQ 的速度。
// 两种方法解压出的数据都按 FastqReader.h 的方式做"提取碱基"处理
// (跳过 header/+/quality 行, 把 sequence 行拷贝进 32KB 块, 含 k-1 重叠保留),
// 处理结果(碱基总数 + CRC32)必须一致, 以此验证两种读取方式等价。
//
// 用法: benchmark_gzip_base_extraction_test <file1,file2,...> <k>
//   文件列表用逗号分隔; k 为 k-mer 长度 (决定块间重叠长度), 必填

#include "../src/GzipStreamer.h"

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <zlib.h>

namespace
{
    // 与 FastqReader/ReaderThreadPool 生产配置保持一致
    constexpr uint64_t kExtractBlockSize = 32ULL * 1024; // = READER_PARSER_RING_MEMORY_POOL_BLOCK_SIZE
    constexpr uint64_t kGzChunkSize = 512ULL * 1024;     // = ReaderThreadPool::GZ_CHUNK_SIZE
    constexpr uint64_t kGzInternalBufferSize = kGzChunkSize / 2;
    constexpr uint64_t kNoNewlineInBlock = static_cast<uint64_t>(-1);

    enum class State
    {
        ReadHeader,
        ReadSequence,
        ReadPlus,
        ReadQuality
    };

    State advance_state_on_newline(const State current)
    {
        switch (current)
        {
        case State::ReadHeader: return State::ReadSequence;
        case State::ReadSequence: return State::ReadPlus;
        case State::ReadPlus: return State::ReadQuality;
        case State::ReadQuality: return State::ReadHeader;
        default: return State::ReadHeader;
        }
    }

    // 复刻 FastqReader 的解析逻辑: 提取每条 record 的 sequence 行写入块,
    // 每写满一个块就统计一次 (块内容含 '\n' 分隔符与跨块重叠碱基)
    class BaseExtractor
    {
    public:
        explicit BaseExtractor(const uint32_t k)
            : overlap_(k > 1 ? static_cast<uint64_t>(k - 1) : 0)
        {
            block_ = static_cast<char*>(std::aligned_alloc(4096, kExtractBlockSize));
            if (block_ == nullptr)
            {
                std::cerr << "failed to allocate extract block\n";
                std::exit(1);
            }
        }

        ~BaseExtractor()
        {
            std::free(block_);
        }

        BaseExtractor(const BaseExtractor&) = delete;
        BaseExtractor& operator=(const BaseExtractor&) = delete;

        // 测试开始时清零统计并重置解析状态
        void reset()
        {
            state_ = State::ReadHeader;
            write_size_ = 0;
            last_newline_pos_ = kNoNewlineInBlock;
            left_buffer_size_ = 0;
            total_bases_ = 0;
            decompressed_bytes_ = 0;
            crc_ = crc32(0L, Z_NULL, 0);
        }

        // 每个文件开始时重置解析状态 (与 FastqReader::read 一致)
        void start_file()
        {
            state_ = State::ReadHeader;
            write_size_ = 0;
            last_newline_pos_ = kNoNewlineInBlock;
            left_buffer_size_ = 0;
        }

        // 文件结束时把最后未满的块计入统计
        void finish_file()
        {
            publish_current_block();
        }

        void feed(const char* input_begin, const uint64_t input_size)
        {
            decompressed_bytes_ += input_size;

            uint64_t input_pos = 0;
            while (input_pos < input_size)
            {
                if (state_ != State::ReadSequence)
                {
                    const char* cur = input_begin + input_pos;
                    const uint64_t remain = input_size - input_pos;
                    const void* nl = std::memchr(cur, '\n', remain);
                    if (nl == nullptr) { input_pos = input_size; break; }
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
                    if (write_size_ == kExtractBlockSize)
                    {
                        store_overlap_from_block_end();
                        publish_current_block();
                        acquire_block();
                    }
                    const uint64_t rem = kExtractBlockSize - write_size_;
                    const uint64_t tc = (seq_len - copied < rem) ? (seq_len - copied) : rem;
                    std::memcpy(block_ + write_size_, seq_cur + copied, tc);
                    write_size_ += tc; copied += tc; input_pos += tc;
                    total_bases_ += tc;
                }
                if (nl == nullptr) break;
                if (write_size_ == kExtractBlockSize)
                {
                    left_buffer_size_ = 0;
                    publish_current_block();
                    acquire_block();
                }
                last_newline_pos_ = write_size_;
                block_[write_size_++] = '\n'; ++input_pos;
                state_ = advance_state_on_newline(state_);
            }
        }

        uint64_t total_bases() const { return total_bases_; }
        uint64_t decompressed_bytes() const { return decompressed_bytes_; }
        uint64_t crc() const { return static_cast<uint64_t>(crc_); }

    private:
        void acquire_block()
        {
            write_size_ = 0;
            last_newline_pos_ = kNoNewlineInBlock;
            if (left_buffer_size_ > 0)
            {
                std::memcpy(block_, left_buffer_, left_buffer_size_);
                write_size_ = left_buffer_size_;
                left_buffer_size_ = 0;
            }
        }

        void publish_current_block()
        {
            if (write_size_ > 0)
            {
                crc_ = crc32(crc_, reinterpret_cast<Bytef*>(block_), static_cast<uInt>(write_size_));
            }
            write_size_ = 0;
            last_newline_pos_ = kNoNewlineInBlock;
        }

        void store_overlap_from_block_end()
        {
            left_buffer_size_ = 0;
            if (overlap_ == 0 || write_size_ == 0) return;
            const uint64_t keep = (write_size_ < overlap_) ? write_size_ : overlap_;
            const uint64_t start = write_size_ - keep;
            if (last_newline_pos_ != kNoNewlineInBlock && last_newline_pos_ >= start)
            {
                const uint64_t kp = write_size_ - (last_newline_pos_ + 1);
                if (kp > 0)
                {
                    std::memcpy(left_buffer_, block_ + last_newline_pos_ + 1, kp);
                    left_buffer_size_ = static_cast<size_t>(kp);
                }
                return;
            }
            std::memcpy(left_buffer_, block_ + start, keep);
            left_buffer_size_ = static_cast<size_t>(keep);
        }

        State state_ = State::ReadHeader;
        char* block_ = nullptr;
        uint64_t write_size_ = 0;
        uint64_t last_newline_pos_ = kNoNewlineInBlock;
        char left_buffer_[128];
        size_t left_buffer_size_ = 0;
        const uint64_t overlap_;

        uint64_t total_bases_ = 0;
        uint64_t decompressed_bytes_ = 0;
        uLong crc_ = 0; // 提取出的块流的 CRC32
    };

    struct RunResult
    {
        double seconds = 0.0;
        uint64_t bases = 0;
        uint64_t crc = 0;
        uint64_t decompressed_bytes = 0;
    };

    std::vector<std::string> parse_file_list(const std::string& arg)
    {
        std::vector<std::string> files;
        size_t start = 0;
        while (start <= arg.size())
        {
            const size_t comma = arg.find(',', start);
            std::string token = arg.substr(start,
                comma == std::string::npos ? std::string::npos : comma - start);
            const size_t first = token.find_first_not_of(" \t");
            if (first != std::string::npos)
            {
                const size_t last = token.find_last_not_of(" \t");
                files.push_back(token.substr(first, last - first + 1));
            }
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        return files;
    }

    // 与 FastqReader::open_current_file 一样用魔数判断 gzip
    bool is_gzip_file(const std::string& path)
    {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) return false;
        unsigned char buf[2];
        const ssize_t n = ::read(fd, buf, 2);
        ::close(fd);
        return n == 2 && buf[0] == 0x1F && buf[1] == 0x8B;
    }

    RunResult run_gzip_streamer(const std::vector<std::string>& files, BaseExtractor& extractor)
    {
        RunResult result;
        GzipStreamer streamer;
        uint8_t* chunk_data = nullptr;
        size_t chunk_size = 0;

        extractor.reset();
        // steady_clock 单调递增, 不受系统时间跳变影响
        const auto start = std::chrono::steady_clock::now();
        for (const std::string& file : files)
        {
            streamer.open(file);
            extractor.start_file();
            while (streamer.next(chunk_data, chunk_size))
            {
                extractor.feed(reinterpret_cast<const char*>(chunk_data),
                    static_cast<uint64_t>(chunk_size));
            }
            extractor.finish_file();
            streamer.close();
        }
        const auto end = std::chrono::steady_clock::now();

        result.seconds = std::chrono::duration<double>(end - start).count();
        result.bases = extractor.total_bases();
        result.crc = extractor.crc();
        result.decompressed_bytes = extractor.decompressed_bytes();
        return result;
    }

    RunResult run_gzread(const std::vector<std::string>& files, BaseExtractor& extractor)
    {
        RunResult result;
        std::vector<char> buffer(kGzChunkSize);

        extractor.reset();
        const auto start = std::chrono::steady_clock::now();
        for (const std::string& file : files)
        {
            gzFile gz = gzopen(file.c_str(), "rb");
            if (gz == nullptr)
            {
                std::cerr << "gzopen failed: " << file << '\n';
                std::exit(1);
            }
            gzbuffer(gz, static_cast<unsigned>(kGzInternalBufferSize));

            extractor.start_file();
            while (true)
            {
                const int n = gzread(gz, buffer.data(), static_cast<unsigned>(buffer.size()));
                if (n < 0)
                {
                    std::cerr << "gzread failed: " << file << '\n';
                    gzclose(gz);
                    std::exit(1);
                }
                if (n == 0) break;
                extractor.feed(buffer.data(), static_cast<uint64_t>(n));
            }
            extractor.finish_file();
            gzclose(gz);
        }
        const auto end = std::chrono::steady_clock::now();

        result.seconds = std::chrono::duration<double>(end - start).count();
        result.bases = extractor.total_bases();
        result.crc = extractor.crc();
        result.decompressed_bytes = extractor.decompressed_bytes();
        return result;
    }

    bool same_result(const RunResult& a, const RunResult& b)
    {
        return a.bases == b.bases && a.crc == b.crc
            && a.decompressed_bytes == b.decompressed_bytes;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <gz_file1,gz_file2,...> <k>\n";
        return 1;
    }

    try
    {
        const std::vector<std::string> all_files = parse_file_list(argv[1]);
        if (all_files.empty())
        {
            std::cerr << "no input files\n";
            return 1;
        }

        const uint32_t k = static_cast<uint32_t>(std::stoul(argv[2]));
        if (k < 1 || k >= 128)
        {
            std::cerr << "k must be in [1, 127]\n";
            return 1;
        }

        std::vector<std::string> gz_files;
        uint64_t compressed_bytes = 0;
        for (const std::string& file : all_files)
        {
            if (!std::filesystem::exists(file))
            {
                std::cerr << "file not found: " << file << '\n';
                return 1;
            }
            if (!is_gzip_file(file))
            {
                std::cout << "skipping non-gzip file: " << file << '\n';
                continue;
            }
            const uint64_t size = std::filesystem::file_size(file);
            compressed_bytes += size;
            gz_files.push_back(file);
            std::cout << "input gzip file: " << file << " (" << size << " bytes compressed)\n";
        }
        if (gz_files.empty())
        {
            std::cerr << "no gzip input files\n";
            return 1;
        }

        std::cout << "compressed total: " << compressed_bytes << " bytes, files: "
                  << gz_files.size() << ", k: " << k << ", single thread\n\n";

        BaseExtractor extractor(k);
        const RunResult streamer = run_gzip_streamer(gz_files, extractor);
        const RunResult gzread = run_gzread(gz_files, extractor);

        if (!same_result(streamer, gzread))
        {
            std::cerr << "extracted bases mismatch between GzipStreamer and gzread\n";
            return 1;
        }

        const double mib = static_cast<double>(streamer.decompressed_bytes) / (1024.0 * 1024.0);
        std::cout << std::fixed;
        std::cout << "GzipStreamer: " << std::setprecision(3) << streamer.seconds << " s ("
                  << std::setprecision(1) << (mib / streamer.seconds) << " MiB/s decompressed)\n";
        std::cout << "gzread:       " << std::setprecision(3) << gzread.seconds << " s ("
                  << std::setprecision(1) << (mib / gzread.seconds) << " MiB/s decompressed)\n";

        std::cout << "\ndecompressed " << streamer.decompressed_bytes
                  << " bytes, extracted " << streamer.bases
                  << " bases (crc32 0x" << std::hex << streamer.crc << std::dec << ")\n";

        const double ratio = gzread.seconds / streamer.seconds;
        std::cout << std::fixed << std::setprecision(2);
        if (ratio >= 1.0)
        {
            std::cout << "GzipStreamer is " << ratio << "x faster than gzread\n";
        }
        else
        {
            std::cout << "gzread is " << (1.0 / ratio) << "x faster than GzipStreamer\n";
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
