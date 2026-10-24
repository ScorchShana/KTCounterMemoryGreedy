#ifndef KTP_FORMAT_HEADER
#define KTP_FORMAT_HEADER

// KTP1（v2）：中间 k-mer 记录格式，块内排序 + 前缀编码，支持任意 k（1..128）。
//
// 背景：build/tmp/{low,high}.bin 原来是“定长 key + count”的顺序文件，相邻记录
// 基本不共享前缀。KTP1 写出时按 key 高 B 位分桶，桶内每 block_records 条排序后
// 做前缀编码，读端按同一规则解码还原。
//
// 文件布局（全部小端）：
//   [0, 64)              header
//   [64, 64+8*(NB+1))    u64 bucket_offset[]
//   [.., +4*NB)          u32 bucket_count[]
//   data                 bucket sections
//
// header（64 B）：
//   char[4] magic = "KTP1"
//   u32 version   = 2
//   u32 k
//   u32 key_bytes            ceil(k/4)，即每条 key 的字节数
//   u32 bucket_bits  B（2..min(12, 8*key_bytes)）
//   u32 sh_bits              “额外共享前缀位数/2”字段的位宽
//   u32 block_records
//   u32 has_counts
//   u32 mode       = 1（block-sorted）
//   u32 reserved
//   u64 n_records
//   u64 n_buckets
//
// bucket section：
//   u32 count; u32 key_stream_bytes; u8 key_bits[]; [counts varint...]
//
// key 编码（key 当作不透明的 key_bytes*8 位、按字节大端解释）：
//   bucket = key 的前 B 位
//   每组第一条：写 bucket 之后的 (key_bits-B) 位
//   其余：写 sh_bits 位的 (额外共享位数/2)，再写剩余后缀位
//   共享位数按 2 bit 粒度向下取整
//
// key 的字节布局由写端决定，本格式只保证逐字节还原。

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <fstream>

namespace ktp {

constexpr uint32_t kFormatVersion = 2;
constexpr uint32_t kHeaderBytes = 64;
constexpr uint32_t kMinBucketBits = 2;
constexpr uint32_t kMaxBucketBits = 12;
constexpr uint32_t kMaxK = 128;
constexpr uint32_t kMaxKeyBytes = (kMaxK + 3) / 4;   // 32
constexpr uint32_t kDefaultBucketBits = 6;
constexpr uint32_t kDefaultBlockRecords = 16384;

using KeyBuf = std::array<uint8_t, kMaxKeyBytes>;

inline uint32_t key_bytes_for(uint32_t k) noexcept { return (k + 3) / 4; }
inline bool supported_k(uint32_t k) noexcept { return k >= 1 && k <= kMaxK; }

// ── 位操作 ────────────────────────────────────────────────────────────────
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

inline uint32_t get_bits_uint(const uint8_t* data, unsigned start, unsigned count) noexcept
{
    uint32_t value = 0;
    for (unsigned i = 0; i < count; ++i) value = (value << 1) | get_bit(data, start + i);
    return value;
}

inline void put_bits_uint(uint8_t* data, unsigned start, unsigned count, uint32_t value) noexcept
{
    for (unsigned i = 0; i < count; ++i)
        set_bit(data, start + i, static_cast<uint8_t>((value >> (count - 1 - i)) & 1u));
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

inline unsigned sh_field_bits(unsigned key_bits, unsigned bucket_bits) noexcept
{
    const unsigned max_extra = key_bits - bucket_bits;
    return bits_for_value(max_extra / 2);
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
            const unsigned shift = bits - take;
            const uint64_t chunk = (value >> shift) & mask64(take);
            acc_ = take == 64 ? chunk : ((acc_ << take) | chunk);
            used_ += take;
            bits -= take;
            while (used_ >= 8) {
                used_ -= 8;
                out_.push_back(static_cast<uint8_t>((acc_ >> used_) & 0xffu));
            }
        }
    }

    // 从 data 的 start 位开始写 count 位（MSB-first）
    void put_bits(const uint8_t* data, unsigned start, unsigned count)
    {
        unsigned i = 0;
        while (i < count && ((start + i) & 7) != 0) { put(get_bit(data, start + i), 1); ++i; }
        while (i + 8 <= count) {
            put(data[(start + i) >> 3], 8);
            i += 8;
        }
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
    static uint64_t mask64(unsigned bits) noexcept
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
            const unsigned available = 8 - offset;
            const unsigned take = std::min(bits, available);
            const unsigned shift = available - take;
            const unsigned chunk = (data_[byte] >> shift) & ((1u << take) - 1u);
            value = (value << take) | chunk;
            bit_ += take;
            bits -= take;
        }
        return value;
    }

    void get_bits_into(uint8_t* out, unsigned start, unsigned count)
    {
        unsigned i = 0;
        while (i < count && ((start + i) & 7) != 0) { set_bit(out, start + i, static_cast<uint8_t>(get(1))); ++i; }
        while (i + 8 <= count) {
            const uint8_t byte = static_cast<uint8_t>(get(8));
            if ((start + i) & 7) {
                for (unsigned b = 0; b < 8; ++b) set_bit(out, start + i + b, static_cast<uint8_t>((byte >> (7 - b)) & 1u));
            } else {
                out[(start + i) >> 3] = byte;
            }
            i += 8;
        }
        while (i < count) { set_bit(out, start + i, static_cast<uint8_t>(get(1))); ++i; }
    }

private:
    const uint8_t* data_;
    size_t bits_;
    size_t bit_ = 0;
};

struct Header
{
    uint32_t k = 0;
    uint32_t key_bytes = 0;
    uint32_t bucket_bits = 0;
    uint32_t sh_bits = 0;
    uint32_t block_records = 0;
    uint32_t has_counts = 0;
    uint64_t records = 0;
    uint64_t buckets = 0;
};

class Writer
{
public:
    Writer(uint32_t k, bool has_counts, uint32_t bucket_bits = kDefaultBucketBits,
           uint32_t block_records = kDefaultBlockRecords)
    {
        if (!supported_k(k)) throw std::runtime_error("KTP1 writer: k out of range");
        header_.k = k;
        header_.key_bytes = key_bytes_for(k);
        header_.bucket_bits = std::min(bucket_bits, 8u * header_.key_bytes);
        if (header_.bucket_bits < kMinBucketBits || header_.bucket_bits > kMaxBucketBits)
            throw std::runtime_error("KTP1 writer: bucket_bits out of range");
        header_.block_records = block_records;
        if (block_records == 0) throw std::runtime_error("KTP1 writer: block_records must be > 0");
        header_.has_counts = has_counts ? 1u : 0u;
        header_.buckets = 1ull << header_.bucket_bits;
        header_.sh_bits = sh_field_bits(8u * header_.key_bytes, header_.bucket_bits);
        buckets_.resize(static_cast<size_t>(header_.buckets));
    }

    void add(const void* raw_key, uint32_t count = 1)
    {
        if (finished_) throw std::runtime_error("KTP1 writer: add() after write()");
        Rec rec{};
        std::memcpy(rec.key.data(), raw_key, header_.key_bytes);
        rec.count = count;
        const uint32_t bucket = bucket_of(rec.key.data());
        buckets_[bucket].push_back(rec);
        ++header_.records;
    }

    // 便捷重载：仅 k<=32（key_bytes<=8）时可用，按大端 8 字节写入
    void add_u64(uint64_t key, uint32_t count = 1)
    {
        if (header_.key_bytes != 8) throw std::runtime_error("KTP1 writer: add_u64 requires k<=32");
        uint8_t bytes[8];
        for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<uint8_t>(key >> (8 * (7 - i)));
        add(bytes, count);
    }

    [[nodiscard]] uint64_t records() const noexcept { return header_.records; }
    [[nodiscard]] uint32_t key_bytes() const noexcept { return header_.key_bytes; }

    uint64_t write(const std::string& path)
    {
        if (finished_) throw std::runtime_error("KTP1 writer: write() called twice");
        finished_ = true;

        const uint32_t nb = static_cast<uint32_t>(header_.buckets);
        std::vector<uint8_t> file;
        file.resize(kHeaderBytes + 8ull * (nb + 1) + 4ull * nb, 0);

        std::vector<uint64_t> offsets(nb + 1, 0);
        std::vector<uint32_t> counts(nb, 0);
        std::vector<uint8_t> section;

        for (uint32_t b = 0; b < nb; ++b) {
            offsets[b] = file.size();
            counts[b] = static_cast<uint32_t>(buckets_[b].size());
            if (buckets_[b].empty()) continue;
            section.clear();
            encode_bucket(b, section);
            file.insert(file.end(), section.begin(), section.end());
            std::vector<Rec>().swap(buckets_[b]);
        }
        offsets[nb] = file.size();

        write_header(file.data());
        for (uint32_t b = 0; b <= nb; ++b) put_u64(file.data() + kHeaderBytes + 8ull * b, offsets[b]);
        const size_t count_table = kHeaderBytes + 8ull * (nb + 1);
        for (uint32_t b = 0; b < nb; ++b) put_u32(file.data() + count_table + 4ull * b, counts[b]);

        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("KTP1 writer: cannot open " + path);
        out.write(reinterpret_cast<const char*>(file.data()), static_cast<std::streamsize>(file.size()));
        out.flush();   // 缓冲写要到 flush 才能发现 /dev/full 之类的 I/O 错误
        if (!out) throw std::runtime_error("KTP1 writer: write failed " + path);
        out.close();
        if (!out) throw std::runtime_error("KTP1 writer: close failed " + path);
        return file.size();
    }

private:
    struct Rec
    {
        KeyBuf key{};
        uint32_t count = 0;
        uint32_t pad = 0;
    };

    [[nodiscard]] uint32_t bucket_of(const uint8_t* key) const noexcept
    {
        return get_bits_uint(key, 0, header_.bucket_bits);
    }

    void encode_bucket(uint32_t b, std::vector<uint8_t>& out)
    {
        auto& recs = buckets_[b];
        const unsigned key_bits = 8u * header_.key_bytes;
        const unsigned base_width = key_bits - header_.bucket_bits;

        const auto less = [this](const Rec& a, const Rec& c) {
            const int cmp = std::memcmp(a.key.data(), c.key.data(), header_.key_bytes);
            if (cmp != 0) return cmp < 0;
            return a.count < c.count;
        };
        for (size_t base = 0; base < recs.size(); base += header_.block_records) {
            const size_t end = std::min(recs.size(), base + header_.block_records);
            std::sort(recs.begin() + base, recs.begin() + end, less);
        }

        std::vector<uint8_t> keys;
        keys.reserve(recs.size() * ((base_width + 7) / 8) + 64);
        BitWriter writer(keys);
        const unsigned sh_bits = header_.sh_bits;

        KeyBuf prev{};
        for (size_t base = 0; base < recs.size(); base += header_.block_records) {
            const size_t end = std::min(recs.size(), base + header_.block_records);
            for (size_t i = base; i < end; ++i) {
                const uint8_t* v = recs[i].key.data();
                if (i == base) {
                    writer.put_bits(v, header_.bucket_bits, base_width);
                } else {
                    unsigned shared = shared_bits(prev.data(), v, key_bits);
                    shared &= ~1u;   // 2 bit 粒度
                    unsigned extra = shared > header_.bucket_bits ? shared - header_.bucket_bits : 0u;
                    extra = std::min(extra, base_width);
                    extra &= ~1u;
                    const unsigned width = base_width - extra;
                    writer.put(extra / 2, sh_bits);
                    writer.put_bits(v, key_bits - width, width);
                }
                std::memcpy(prev.data(), v, header_.key_bytes);
            }
        }
        writer.flush();

        append_u32(out, static_cast<uint32_t>(recs.size()));
        append_u32(out, static_cast<uint32_t>(keys.size()));
        out.insert(out.end(), keys.begin(), keys.end());

        if (header_.has_counts) {
            for (const Rec& rec : recs) {
                uint32_t value = rec.count;
                while (value >= 0x80) {
                    out.push_back(static_cast<uint8_t>(value | 0x80));
                    value >>= 7;
                }
                out.push_back(static_cast<uint8_t>(value));
            }
        }
    }

    void write_header(uint8_t* p) const
    {
        std::memset(p, 0, kHeaderBytes);
        std::memcpy(p, "KTP1", 4);
        put_u32(p + 4, kFormatVersion);
        put_u32(p + 8, header_.k);
        put_u32(p + 12, header_.key_bytes);
        put_u32(p + 16, header_.bucket_bits);
        put_u32(p + 20, header_.sh_bits);
        put_u32(p + 24, header_.block_records);
        put_u32(p + 28, header_.has_counts);
        put_u32(p + 32, 1);   // mode = block-sorted
        put_u64(p + 40, header_.records);
        put_u64(p + 48, header_.buckets);
    }

    static void append_u32(std::vector<uint8_t>& out, uint32_t v)
    {
        out.push_back(static_cast<uint8_t>(v));
        out.push_back(static_cast<uint8_t>(v >> 8));
        out.push_back(static_cast<uint8_t>(v >> 16));
        out.push_back(static_cast<uint8_t>(v >> 24));
    }

    static void put_u32(uint8_t* p, uint32_t v) noexcept
    {
        p[0] = static_cast<uint8_t>(v);
        p[1] = static_cast<uint8_t>(v >> 8);
        p[2] = static_cast<uint8_t>(v >> 16);
        p[3] = static_cast<uint8_t>(v >> 24);
    }

    static void put_u64(uint8_t* p, uint64_t v) noexcept
    {
        for (unsigned i = 0; i < 8; ++i) p[i] = static_cast<uint8_t>(v >> (8 * i));
    }

    Header header_{};
    std::vector<std::vector<Rec>> buckets_;
    bool finished_ = false;
};

class Reader
{
public:
    Reader() = default;
    Reader(const Reader&) = delete;
    Reader& operator=(const Reader&) = delete;
    Reader(Reader&& other) noexcept { move_from(other); }
    Reader& operator=(Reader&& other) noexcept
    {
        if (this != &other) { release(); move_from(other); }
        return *this;
    }
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
        try {
            parse();
        } catch (...) {
            release();
            throw;
        }
    }

    [[nodiscard]] uint64_t records() const noexcept { return header_.records; }
    [[nodiscard]] uint32_t k() const noexcept { return header_.k; }
    [[nodiscard]] uint32_t key_bytes() const noexcept { return header_.key_bytes; }
    [[nodiscard]] bool has_counts() const noexcept { return header_.has_counts != 0; }

    // 回调里的 key 指针只在回调期间有效
    void decode_all(const std::function<void(const uint8_t*, uint32_t)>& callback) const
    {
        for (uint32_t b = 0; b < header_.buckets; ++b) {
            if (bucket_counts_[b] == 0) continue;
            decode_bucket(b, 0, std::numeric_limits<uint64_t>::max(), callback);
        }
    }

    void decode_range(uint64_t first, uint64_t count,
                      const std::function<void(const uint8_t*, uint32_t)>& callback) const
    {
        if (first > header_.records || count > header_.records - first) throw std::out_of_range("KTP1: range");
        if (count == 0) return;
        uint64_t index = first;
        uint64_t remaining = count;
        uint32_t b = bucket_of_record(index);
        while (remaining > 0) {
            const uint64_t in_bucket = std::min<uint64_t>(remaining, prefix_[b + 1] - index);
            decode_bucket(b, index - prefix_[b], in_bucket, callback);
            index += in_bucket;
            remaining -= in_bucket;
            ++b;
        }
    }

    // 还原成旧格式字节流（key_bytes 字节 + count 小端 count_bytes 字节）
    std::vector<char> to_legacy(uint32_t count_bytes) const
    {
        std::vector<char> out(static_cast<size_t>(header_.records) * (header_.key_bytes + count_bytes));
        char* cursor = out.data();
        const uint32_t kb = header_.key_bytes;
        const bool counts = header_.has_counts;
        decode_all([&](const uint8_t* key, uint32_t count) {
            std::memcpy(cursor, key, kb);
            cursor += kb;
            if (counts && count_bytes) {
                for (uint32_t i = 0; i < count_bytes; ++i)
                    *cursor++ = static_cast<char>((count >> (8 * i)) & 0xffu);
            }
        });
        return out;
    }

private:
    [[nodiscard]] uint32_t bucket_of_record(uint64_t index) const
    {
        const auto it = std::upper_bound(prefix_.begin(), prefix_.end(), index);
        return static_cast<uint32_t>(it - prefix_.begin() - 1);
    }

    void move_from(Reader& other) noexcept
    {
        fd_ = other.fd_;
        data_ = other.data_;
        size_ = other.size_;
        header_ = other.header_;
        bucket_offsets_ = std::move(other.bucket_offsets_);
        bucket_counts_ = std::move(other.bucket_counts_);
        prefix_ = std::move(other.prefix_);
        other.fd_ = -1;
        other.data_ = nullptr;
        other.size_ = 0;
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
        const uint8_t* p = data_;
        if (std::memcmp(p, "KTP1", 4) != 0) throw std::runtime_error("KTP1: bad magic");
        if (get_u32(p + 4) != kFormatVersion) throw std::runtime_error("KTP1: unsupported version");
        header_.k = get_u32(p + 8);
        header_.key_bytes = get_u32(p + 12);
        header_.bucket_bits = get_u32(p + 16);
        header_.sh_bits = get_u32(p + 20);
        header_.block_records = get_u32(p + 24);
        header_.has_counts = get_u32(p + 28);
        header_.records = get_u64(p + 40);
        header_.buckets = get_u64(p + 48);
        if (!supported_k(header_.k)) throw std::runtime_error("KTP1: bad k");
        if (header_.key_bytes != key_bytes_for(header_.k)) throw std::runtime_error("KTP1: bad key_bytes");
        if (header_.bucket_bits < kMinBucketBits || header_.bucket_bits > kMaxBucketBits ||
            header_.bucket_bits > 8 * header_.key_bytes)
            throw std::runtime_error("KTP1: bad bucket_bits");
        if (header_.block_records == 0) throw std::runtime_error("KTP1: bad block_records");
        if (header_.buckets != (1ull << header_.bucket_bits)) throw std::runtime_error("KTP1: bucket mismatch");

        const uint32_t nb = static_cast<uint32_t>(header_.buckets);
        const size_t count_table = kHeaderBytes + 8ull * (nb + 1);
        if (size_ < count_table + 4ull * nb) throw std::runtime_error("KTP1: truncated table");

        bucket_offsets_.resize(nb + 1);
        for (uint32_t b = 0; b <= nb; ++b) bucket_offsets_[b] = get_u64(p + kHeaderBytes + 8ull * b);
        bucket_counts_.resize(nb);
        prefix_.assign(nb + 1, 0);
        uint64_t sum = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            bucket_counts_[b] = get_u32(p + count_table + 4ull * b);
            sum += bucket_counts_[b];
            prefix_[b + 1] = sum;
        }
        if (sum != header_.records) throw std::runtime_error("KTP1: record count mismatch");
        const uint64_t data_start = count_table + 4ull * nb;
        if (bucket_offsets_.front() != data_start) throw std::runtime_error("KTP1: bad data offset");
        if (bucket_offsets_.back() != size_) throw std::runtime_error("KTP1: bad length");
        for (uint32_t b = 0; b < nb; ++b)
            if (bucket_offsets_[b] > bucket_offsets_[b + 1]) throw std::runtime_error("KTP1: bad offsets");
    }

    void decode_bucket(uint32_t b, uint64_t skip, uint64_t limit,
                       const std::function<void(const uint8_t*, uint32_t)>& callback) const
    {
        const uint8_t* section = data_ + bucket_offsets_[b];
        const uint32_t count = get_u32(section);
        const uint32_t key_stream_bytes = get_u32(section + 4);
        const uint8_t* key_stream = section + 8;
        if (bucket_offsets_[b] + 8ull + key_stream_bytes > size_) throw std::runtime_error("KTP1: truncated section");
        BitReader reader(key_stream, key_stream_bytes);

        const unsigned key_bits = 8u * header_.key_bytes;
        const unsigned base_width = key_bits - header_.bucket_bits;
        const unsigned sh_bits = header_.sh_bits;
        KeyBuf key{};
        put_bits_uint(key.data(), 0, header_.bucket_bits, b);
        const uint8_t* counts = section + 8 + key_stream_bytes;
        size_t counts_pos = 0;

        uint64_t emitted = 0;
        for (uint32_t i = 0; i < count && emitted < limit; ++i) {
            if ((i % header_.block_records) == 0) {
                KeyBuf fresh{};
                put_bits_uint(fresh.data(), 0, header_.bucket_bits, b);
                reader.get_bits_into(fresh.data(), header_.bucket_bits, base_width);
                key = fresh;
            } else {
                const unsigned extra = 2 * static_cast<unsigned>(reader.get(sh_bits));
                if (extra > base_width) throw std::runtime_error("KTP1: bad prefix length");
                const unsigned width = base_width - extra;
                reader.get_bits_into(key.data(), header_.bucket_bits + extra, width);
            }
            uint32_t value = 1;
            if (header_.has_counts) {
                uint64_t v = 0;
                unsigned shift = 0;
                for (;;) {
                    const uint8_t byte = counts[counts_pos++];
                    v |= static_cast<uint64_t>(byte & 0x7f) << shift;
                    if ((byte & 0x80) == 0) break;
                    shift += 7;
                    if (shift > 28) throw std::runtime_error("KTP1: varint overflow");
                }
                value = static_cast<uint32_t>(v);
            }
            if (i >= skip) {
                callback(key.data(), value);
                ++emitted;
            }
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

    Header header_{};
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    size_t size_ = 0;
    std::vector<uint64_t> bucket_offsets_;
    std::vector<uint32_t> bucket_counts_;
    std::vector<uint64_t> prefix_;
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
