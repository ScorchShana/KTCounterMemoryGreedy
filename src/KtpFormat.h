#ifndef KTP_FORMAT_HEADER
#define KTP_FORMAT_HEADER

// KTP1：中间 k-mer 记录文件，只按块排序 + 前缀编码。
//
// 写端按到达顺序攒满 block_records 条记录后：块内排序 → 首条写满 key 位，
// 其余只写与上一条的差异后缀 → 追加到文件。写端内存只有一块。
//
// 布局（小端）：
//   [0, 32)         header
//   [32, table)     blocks
//   [table, EOF)    u64 block_offset[n_blocks + 1]
//
// header（32 B）：
//   char[4] magic = "KTP1"
//   u32 k
//   u32 block_records
//   u32 has_counts
//   u64 n_records
//   u32 reserved[2]
//
// block：
//   u32 key_stream_bytes; u8 key_bits[]; [counts varint...]
//
// 块内前缀编码：
//   extra = 与上一条 key 的前导相同位数（2 bit 粒度向下取整）
//   首条：写满 key_bits 位
//   其余：写 sh_bits 位的 extra/2，再写 key_bits-extra 位后缀
//   sh_bits 由 key_bits 推导，不落盘。

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace ktp {

constexpr uint32_t kHeaderBytes = 32;
constexpr uint32_t kMaxK = 128;
constexpr uint32_t kMaxKeyBytes = (kMaxK + 3) / 4;   // 32
constexpr uint32_t kDefaultBlockRecords = 16384;

inline uint32_t key_bytes_for(uint32_t k) noexcept { return (k + 3) / 4; }
inline bool supported_k(uint32_t k) noexcept { return k >= 1 && k <= kMaxK; }

// 从 data 的 bit 位取一位（MSB-first）
inline uint8_t get_bit(const uint8_t* data, unsigned bit) noexcept
{
    return static_cast<uint8_t>((data[bit >> 3] >> (7 - (bit & 7))) & 1u);
}

inline void set_bit(uint8_t* data, unsigned bit, uint8_t value) noexcept
{
    const uint8_t mask = static_cast<uint8_t>(1u << (7 - (bit & 7)));
    if (value) data[bit >> 3] = static_cast<uint8_t>(data[bit >> 3] | mask);
    else data[bit >> 3] = static_cast<uint8_t>(data[bit >> 3] & ~mask);
}

inline unsigned shared_bits(const uint8_t* a, const uint8_t* b, unsigned bits) noexcept
{
    const unsigned full = bits / 8;
    for (unsigned i = 0; i < full; ++i) {
        const uint8_t x = static_cast<uint8_t>(a[i] ^ b[i]);
        if (x) return i * 8 + static_cast<unsigned>(std::countl_zero(x));
    }
    const unsigned rem = bits % 8;
    if (rem) {
        const uint8_t mask = static_cast<uint8_t>(0xffu << (8 - rem));
        const uint8_t x = static_cast<uint8_t>((a[full] ^ b[full]) & mask);
        if (x) return full * 8 + static_cast<unsigned>(std::countl_zero(x));
    }
    return bits;
}

inline unsigned bits_for_value(unsigned max_value) noexcept
{
    unsigned bits = 1;
    while (bits < 32 && (1ull << bits) <= max_value) ++bits;
    return bits;
}

class BitWriter
{
public:
    explicit BitWriter(std::vector<uint8_t>& out) : out_(out) {}

    void put(uint64_t value, unsigned bits)
    {
        while (bits > 0) {
            const unsigned room = 64 - used_;
            const unsigned take = std::min(bits, room);
            const uint64_t chunk = (value >> (bits - take)) & mask(take);
            acc_ = take == 64 ? chunk : ((acc_ << take) | chunk);
            used_ += take;
            bits -= take;
            while (used_ >= 8) {
                used_ -= 8;
                out_.push_back(static_cast<uint8_t>((acc_ >> used_) & 0xffu));
            }
        }
    }

    // 从 data 的 start 位开始写 count 位；源对齐的部分按整字节批量写
    void put_bits(const uint8_t* data, unsigned start, unsigned count)
    {
        unsigned i = 0;
        while (i < count && ((start + i) & 7) != 0) { put(get_bit(data, start + i), 1); ++i; }
        while (i + 8 <= count) { put(data[(start + i) >> 3], 8); i += 8; }
        while (i < count) { put(get_bit(data, start + i), 1); ++i; }
    }

    void flush()
    {
        if (used_ > 0) {
            out_.push_back(static_cast<uint8_t>((acc_ << (8 - used_)) & 0xffu));
            used_ = 0;
        }
    }

private:
    static uint64_t mask(unsigned bits) noexcept
    {
        return bits >= 64 ? ~uint64_t{0} : ((uint64_t{1} << bits) - 1);
    }

    std::vector<uint8_t>& out_;
    uint64_t acc_ = 0;
    unsigned used_ = 0;
};

class BitReader
{
public:
    BitReader(const uint8_t* data, size_t bytes) : data_(data), bits_(bytes * 8) {}

    uint64_t get(unsigned bits)
    {
        if (bit_ + bits > bits_) throw std::runtime_error("KTP1: truncated key stream");
        uint64_t value = 0;
        while (bits > 0) {
            const size_t byte = bit_ >> 3;
            const unsigned offset = static_cast<unsigned>(bit_ & 7);
            const unsigned take = std::min(bits, 8 - offset);
            value = (value << take) | ((data_[byte] >> (8 - offset - take)) & ((1u << take) - 1u));
            bit_ += take;
            bits -= take;
        }
        return value;
    }

    void get_bits_into(uint8_t* out, unsigned start, unsigned count)
    {
        for (unsigned i = 0; i < count; ++i) set_bit(out, start + i, static_cast<uint8_t>(get(1)));
    }

private:
    const uint8_t* data_;
    size_t bits_;
    size_t bit_ = 0;
};

// 流式写端：只保留一个块的内存。
class Writer
{
public:
    Writer(const std::string& path, uint32_t k, bool has_counts,
           uint32_t block_records = kDefaultBlockRecords)
        : path_(path), k_(k), key_bytes_(key_bytes_for(k)), key_bits_(8 * key_bytes_),
          block_records_(block_records), sh_bits_(bits_for_value(key_bits_ / 2)),
          has_counts_(has_counts)
    {
        if (!supported_k(k) || block_records_ == 0)
            throw std::runtime_error("KTP1 writer: invalid k or block_records");
        out_.open(path, std::ios::binary | std::ios::trunc);
        if (!out_) throw std::runtime_error("KTP1 writer: cannot open " + path);
        const uint8_t zeros[kHeaderBytes]{};
        write_raw(zeros, kHeaderBytes);
        block_.reserve(block_records_);
    }

    Writer(const Writer&) = delete;
    Writer& operator=(const Writer&) = delete;

    void add(const void* raw_key, uint32_t count = 1)
    {
        if (finished_) throw std::runtime_error("KTP1 writer: add() after finish()");
        Rec rec{};
        std::memcpy(rec.key.data(), raw_key, key_bytes_);
        rec.count = count;
        block_.push_back(rec);
        if (block_.size() == block_records_) flush_block();
    }

    void finish()
    {
        if (finished_) throw std::runtime_error("KTP1 writer: finish() called twice");
        flush_block();
        finished_ = true;

        const uint64_t table_offset = static_cast<uint64_t>(out_.tellp());
        offsets_.push_back(table_offset);
        for (const uint64_t offset : offsets_) write_u64(offset);
        write_header(table_offset);
        out_.flush();
        if (!out_) throw std::runtime_error("KTP1 writer: write failed " + path_);
        out_.close();
    }

private:
    struct Rec
    {
        std::array<uint8_t, kMaxKeyBytes> key{};
        uint32_t count = 0;
    };

    void flush_block()
    {
        if (block_.empty()) return;
        const size_t count = block_.size();
        std::sort(block_.begin(), block_.end(), [this](const Rec& a, const Rec& b) {
            const int cmp = std::memcmp(a.key.data(), b.key.data(), key_bytes_);
            return cmp != 0 ? cmp < 0 : a.count < b.count;
        });

        std::vector<uint8_t> keys;
        BitWriter writer(keys);
        std::array<uint8_t, kMaxKeyBytes> prev{};
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* key = block_[i].key.data();
            if (i == 0) {
                writer.put_bits(key, 0, key_bits_);
            } else {
                const unsigned shared = std::min(shared_bits(prev.data(), key, key_bits_) & ~1u, key_bits_);
                writer.put(shared / 2, sh_bits_);
                writer.put_bits(key, shared, key_bits_ - shared);
            }
            std::memcpy(prev.data(), key, key_bytes_);
        }
        writer.flush();

        offsets_.push_back(static_cast<uint64_t>(out_.tellp()));
        write_u32(static_cast<uint32_t>(keys.size()));
        write_raw(keys.data(), keys.size());
        if (has_counts_) {
            for (const Rec& rec : block_) write_varint(rec.count);
        }
        records_ += count;
        block_.clear();
    }

    void write_header(uint64_t table_offset)
    {
        uint8_t header[kHeaderBytes]{};
        std::memcpy(header, "KTP1", 4);
        put_u32(header + 4, k_);
        put_u32(header + 8, block_records_);
        put_u32(header + 12, has_counts_ ? 1u : 0u);
        put_u64(header + 16, records_);
        out_.seekp(0);
        write_raw(header, kHeaderBytes);
        out_.seekp(table_offset);
        if (!out_) throw std::runtime_error("KTP1 writer: header failed " + path_);
    }

    void write_u32(uint32_t v)
    {
        uint8_t bytes[4];
        put_u32(bytes, v);
        write_raw(bytes, 4);
    }

    void write_u64(uint64_t v)
    {
        uint8_t bytes[8];
        put_u64(bytes, v);
        write_raw(bytes, 8);
    }

    void write_varint(uint32_t value)
    {
        uint8_t bytes[5];
        unsigned n = 0;
        while (value >= 0x80) {
            bytes[n++] = static_cast<uint8_t>(value | 0x80);
            value >>= 7;
        }
        bytes[n++] = static_cast<uint8_t>(value);
        write_raw(bytes, n);
    }

    void write_raw(const void* data, size_t bytes)
    {
        out_.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
        if (!out_) throw std::runtime_error("KTP1 writer: write failed " + path_);
    }

    static void put_u32(uint8_t* p, uint32_t v) noexcept
    {
        for (unsigned i = 0; i < 4; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
    }

    static void put_u64(uint8_t* p, uint64_t v) noexcept
    {
        for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
    }

    std::string path_;
    std::ofstream out_;
    uint32_t k_ = 0;
    uint32_t key_bytes_ = 0;
    uint32_t key_bits_ = 0;
    uint32_t block_records_ = 0;
    uint32_t sh_bits_ = 0;
    bool has_counts_ = false;
    uint64_t records_ = 0;
    std::vector<Rec> block_;
    std::vector<uint64_t> offsets_;
    bool finished_ = false;
};

class Reader
{
public:
    Reader() = default;
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    ~Reader() { release(); }

    void open(const std::string& path)
    {
        release();
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("KTP1 reader: cannot open " + path);
        struct stat st{};
        if (::fstat(fd_, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < static_cast<off_t>(kHeaderBytes))
            throw std::runtime_error("KTP1 reader: bad file " + path);
        size_ = static_cast<size_t>(st.st_size);
        void* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("KTP1 reader: mmap failed " + path);
        data_ = static_cast<const uint8_t*>(mapping);
        try { parse(); } catch (...) { release(); throw; }
    }

    [[nodiscard]] uint64_t records() const noexcept { return records_; }
    [[nodiscard]] uint32_t k() const noexcept { return k_; }
    [[nodiscard]] uint32_t key_bytes() const noexcept { return key_bytes_; }
    [[nodiscard]] bool has_counts() const noexcept { return has_counts_; }

    void decode_all(const std::function<void(const uint8_t*, uint32_t)>& callback) const
    {
        for (uint64_t b = 0; b < n_blocks_; ++b) decode_block(b, 0, block_size(b), callback);
    }

    void decode_range(uint64_t first, uint64_t count,
                      const std::function<void(const uint8_t*, uint32_t)>& callback) const
    {
        if (first > records_ || count > records_ - first) throw std::out_of_range("KTP1: range");
        uint64_t remaining = count;
        uint64_t index = first;
        while (remaining > 0) {
            const uint64_t block = index / block_records_;
            const uint64_t local = index % block_records_;
            const uint64_t take = std::min(remaining, block_size(block) - local);
            decode_block(block, local, take, callback);
            index += take;
            remaining -= take;
        }
    }

    // 还原成旧格式字节流（key_bytes 字节 + count 小端 count_bytes 字节）
    std::vector<char> to_legacy(uint32_t count_bytes) const
    {
        std::vector<char> out(static_cast<size_t>(records_) * (key_bytes_ + count_bytes));
        char* cursor = out.data();
        decode_all([&](const uint8_t* key, uint32_t count) {
            std::memcpy(cursor, key, key_bytes_);
            cursor += key_bytes_;
            for (uint32_t i = 0; i < count_bytes; ++i) *cursor++ = static_cast<char>(count >> (8 * i));
        });
        return out;
    }

private:
    [[nodiscard]] uint64_t block_size(uint64_t block) const noexcept
    {
        return block + 1 < n_blocks_ ? block_records_ : records_ - block * block_records_;
    }

    void release() noexcept
    {
        if (data_) ::munmap(const_cast<uint8_t*>(data_), size_);
        if (fd_ >= 0) ::close(fd_);
        data_ = nullptr;
        fd_ = -1;
        size_ = 0;
    }

    void parse()
    {
        if (std::memcmp(data_, "KTP1", 4) != 0) throw std::runtime_error("KTP1: bad magic");
        k_ = get_u32(data_ + 4);
        block_records_ = get_u32(data_ + 8);
        has_counts_ = get_u32(data_ + 12) != 0;
        records_ = get_u64(data_ + 16);
        key_bytes_ = key_bytes_for(k_);
        sh_bits_ = bits_for_value(8 * key_bytes_ / 2);
        if (!supported_k(k_) || block_records_ == 0) throw std::runtime_error("KTP1: bad header");

        n_blocks_ = records_ == 0 ? 0 : (records_ + block_records_ - 1) / block_records_;
        const uint64_t table_bytes = 8ull * (n_blocks_ + 1);
        if (size_ < kHeaderBytes + table_bytes) throw std::runtime_error("KTP1: truncated");
        table_offset_ = size_ - table_bytes;

        offsets_.resize(n_blocks_ + 1);
        for (uint64_t b = 0; b <= n_blocks_; ++b) offsets_[b] = get_u64(data_ + table_offset_ + 8 * b);
        if (offsets_.front() != kHeaderBytes || offsets_.back() != table_offset_)
            throw std::runtime_error("KTP1: bad block table");
    }

    void decode_block(uint64_t block, uint64_t skip, uint64_t limit,
                      const std::function<void(const uint8_t*, uint32_t)>& callback) const
    {
        const uint8_t* section = data_ + offsets_[block];
        const uint32_t stream_bytes = get_u32(section);
        const uint8_t* key_stream = section + 4;
        if (offsets_[block] + 4ull + stream_bytes > table_offset_) throw std::runtime_error("KTP1: bad block");
        BitReader reader(key_stream, stream_bytes);
        const uint8_t* counts = key_stream + stream_bytes;
        const uint64_t key_bits = 8 * key_bytes_;
        const unsigned sh_bits = sh_bits_;

        std::array<uint8_t, kMaxKeyBytes> key{};
        size_t counts_pos = 0;
        for (uint64_t i = 0; i < skip + limit; ++i) {
            if (i == 0) {
                reader.get_bits_into(key.data(), 0, key_bits);
            } else {
                const unsigned extra = 2 * static_cast<unsigned>(reader.get(sh_bits));
                if (extra > key_bits) throw std::runtime_error("KTP1: bad prefix");
                reader.get_bits_into(key.data(), extra, key_bits - extra);
            }
            uint32_t count = 1;
            if (has_counts_) {
                uint64_t value = 0;
                for (unsigned shift = 0;; shift += 7) {
                    const uint8_t byte = counts[counts_pos++];
                    value |= static_cast<uint64_t>(byte & 0x7f) << shift;
                    if ((byte & 0x80) == 0) break;
                    if (shift > 28) throw std::runtime_error("KTP1: bad varint");
                }
                count = static_cast<uint32_t>(value);
            }
            if (i >= skip) callback(key.data(), count);
        }
    }

    static uint32_t get_u32(const uint8_t* p) noexcept
    {
        return uint32_t{p[0]} | (uint32_t{p[1]} << 8) | (uint32_t{p[2]} << 16) | (uint32_t{p[3]} << 24);
    }

    static uint64_t get_u64(const uint8_t* p) noexcept
    {
        uint64_t v = 0;
        for (unsigned i = 0; i < 8; ++i) v |= uint64_t{p[i]} << (8 * i);
        return v;
    }

    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    uint32_t k_ = 0;
    uint32_t key_bytes_ = 0;
    uint32_t block_records_ = 0;
    uint32_t sh_bits_ = 0;
    bool has_counts_ = false;
    uint64_t records_ = 0;
    uint64_t n_blocks_ = 0;
    uint64_t table_offset_ = 0;
    std::vector<uint64_t> offsets_;
};

inline bool is_ktp_file(const std::string& path)
{
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    char magic[4]{};
    const ssize_t n = ::read(fd, magic, 4);
    ::close(fd);
    return n == 4 && std::memcmp(magic, "KTP1", 4) == 0;
}

} // namespace ktp

#endif // KTP_FORMAT_HEADER
