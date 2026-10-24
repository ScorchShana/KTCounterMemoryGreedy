#ifndef KTP_INPUT_HEADER
#define KTP_INPUT_HEADER

// 旧布局：每条记录 key_bytes（k<=32 时 8 字节）+ count_bytes（0/1/2/4 小端）。
// KTP1 文件在这里被解码成同样的字节流，工具其余逻辑无需感知格式。
// 旧格式仍然 mmap，避免把大文件整体读进内存；KTP1 当前整体解码到内存。

#include "../src/KtpFormat.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace ktpinput {

inline uint64_t stat_bytes(const std::string& path)
{
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
        throw std::runtime_error("failed to stat regular input " + path);
    return static_cast<uint64_t>(st.st_size);
}

inline bool is_ktp(const std::string& path) { return ktp::is_ktp_file(path); }

// 记录条数：KTP1 读 header，旧格式用文件大小 / 记录大小。
inline uint64_t records_of(const std::string& path, uint64_t record_bytes)
{
    if (is_ktp(path)) {
        ktp::Reader reader;
        reader.open(path);
        return reader.records();
    }
    const uint64_t bytes = stat_bytes(path);
    if (record_bytes == 0 || bytes % record_bytes)
        throw std::runtime_error("misaligned compact input " + path);
    return bytes / record_bytes;
}

// 旧布局等价字节数，用于进度统计/内存估算。
inline uint64_t legacy_bytes(const std::string& path, uint64_t record_bytes)
{
    return records_of(path, record_bytes) * record_bytes;
}

struct View
{
    std::vector<char> owned;   // KTP1 解码结果
    void* mapping = nullptr;   // 旧格式 mmap
    size_t mapping_bytes = 0;
    const char* data = nullptr;
    uint64_t bytes = 0;
    uint64_t record_bytes = 0;

    View() = default;
    View(const View&) = delete;
    View& operator=(const View&) = delete;
    View(View&& other) noexcept { move_from(other); }
    View& operator=(View&& other) noexcept
    {
        if (this != &other) {
            release();
            move_from(other);
        }
        return *this;
    }
    ~View() { release(); }

    [[nodiscard]] uint64_t records() const { return record_bytes ? bytes / record_bytes : 0; }

private:
    void release() noexcept
    {
        if (mapping) ::munmap(mapping, mapping_bytes);
        mapping = nullptr;
        mapping_bytes = 0;
    }
    void move_from(View& other) noexcept
    {
        owned = std::move(other.owned);
        mapping = other.mapping;
        mapping_bytes = other.mapping_bytes;
        bytes = other.bytes;
        record_bytes = other.record_bytes;
        data = other.owned.empty() ? static_cast<const char*>(other.mapping) : other.owned.data();
        other.mapping = nullptr;
        other.mapping_bytes = 0;
    }
};

// 读取文件并返回旧布局字节流；KTP1 会被解码，旧格式仍走 mmap。
inline View load(const std::string& path, uint32_t k_len, uint32_t count_bytes, uint64_t record_bytes)
{
    View view;
    view.record_bytes = record_bytes;
    if (is_ktp(path)) {
        ktp::Reader reader;
        reader.open(path);
        if (reader.k() != k_len)
            throw std::runtime_error("KTP1 k mismatch for " + path);
        view.owned = reader.to_legacy(count_bytes);
        view.bytes = view.owned.size();
        if (record_bytes && view.bytes % record_bytes)
            throw std::runtime_error("KTP1 legacy size mismatch for " + path);
        view.data = view.owned.data();
        return view;
    }

    const uint64_t bytes = stat_bytes(path);
    if (record_bytes == 0 || bytes % record_bytes)
        throw std::runtime_error("misaligned compact input " + path);
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path);
    void* mapping = ::mmap(nullptr, static_cast<size_t>(bytes), PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    ::close(fd);
    if (mapping == MAP_FAILED) throw std::runtime_error("mmap failed " + path);
    view.mapping = mapping;
    view.mapping_bytes = static_cast<size_t>(bytes);
    view.data = static_cast<const char*>(mapping);
    view.bytes = bytes;
    return view;
}

} // namespace ktpinput

#endif // KTP_INPUT_HEADER
