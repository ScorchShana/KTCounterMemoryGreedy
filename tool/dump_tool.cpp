#include "AioGroupReader.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <charconv>
#include <cmath>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>
#include <fcntl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
constexpr size_t MiB = 1ULL << 20;
constexpr size_t BlockBytes = 2 * MiB;
constexpr size_t BlockCount = 8;
constexpr size_t RadixBuckets = 2048;
constexpr size_t MaxRanges = 64;
constexpr size_t TaskTextBytes = 256 * 1024;

[[noreturn]] void io_error(const std::string& operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}
class File {
    int fd_ = -1;
public:
    explicit File(int fd = -1) : fd_(fd) {}
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File() { if (fd_ >= 0) ::close(fd_); }
    int get() const { return fd_; }
    void close() {
        int fd = fd_; fd_ = -1;
        if (fd >= 0 && ::close(fd)) io_error("close");
    }
};
void write_all(int fd, const void* data, size_t bytes) {
    auto p = static_cast<const uint8_t*>(data);
    while (bytes) {
        ssize_t n = ::write(fd, p, bytes);
        if (n < 0) { if (errno == EINTR) continue; io_error("write"); }
        if (!n) throw std::runtime_error("write made no progress");
        p += n; bytes -= size_t(n);
    }
}
struct Options {
    std::string directory, output;
    bool precise = false, sorted = false;
    unsigned workers = 0, k = 0, key_bytes = 0, count_bytes = 0;
    uint32_t minimum = 1, maximum = 10000;
    uint64_t memory = 0;
    bool low() const { return precise && minimum <= 1 && maximum >= 1; }
    bool accepts(uint32_t n) const { return n >= minimum && n <= maximum; }
};
uint32_t parse_u32(const char* str) {
    uint32_t value;
    const char* end = str + std::strlen(str);
    auto result = std::from_chars(str, end, value);
    if (result.ec != std::errc() || result.ptr != end) throw std::runtime_error("invalid integer argument");
    return value;
}
Options options(int argc, char** argv) {
    if (argc < 6 || argc > 9) throw std::runtime_error(
        "Usage: dump_tool <precise/approximate> <tmp_dir> <max_threads> <max_memory_gb> <output_file> [min_freq] [max_freq] [--sort]");
    Options o;
    if (std::strcmp(argv[1], "precise") == 0) o.precise = true;
    else if (std::strcmp(argv[1], "approximate")) throw std::runtime_error("invalid mode");
    o.directory = argv[2];
    if (o.directory.empty() || o.directory.back() != '/') o.directory += '/';
    o.output = argv[5]; o.workers = parse_u32(argv[3]);
    size_t used = 0;
    long double gib = std::stold(argv[4], &used);
    long double bytes = gib * (1ULL << 30);
    if (used != std::strlen(argv[4]) || !std::isfinite(bytes) || bytes < 1 ||
        bytes >= static_cast<long double>(UINT64_MAX)) throw std::runtime_error("invalid max_memory_gb");
    o.memory = static_cast<uint64_t>(bytes);
    int next = 6;
    if (next < argc && std::strcmp(argv[next], "--sort")) o.minimum = parse_u32(argv[next++]);
    if (next < argc && std::strcmp(argv[next], "--sort")) o.maximum = parse_u32(argv[next++]);
    if (next < argc && std::strcmp(argv[next], "--sort") == 0) { o.sorted = true; ++next; }
    if (next != argc || !o.workers || o.minimum > o.maximum) throw std::runtime_error("invalid dump arguments");
    File info(::open((o.directory + "infos.bin").c_str(), O_RDONLY | O_NONBLOCK | O_CLOEXEC));
    if (info.get() < 0) io_error("open infos.bin");
    struct stat info_stat{};
    if (::fstat(info.get(), &info_stat)) io_error("stat infos.bin");
    if (!S_ISREG(info_stat.st_mode)) throw std::runtime_error("infos.bin is not a regular file");
    uint32_t values[2];
    size_t done = 0;
    while (done < sizeof(values)) {
        ssize_t n = ::read(info.get(), reinterpret_cast<char*>(values) + done, sizeof(values) - done);
        if (n < 0) { if (errno == EINTR) continue; io_error("read infos.bin"); }
        if (!n) throw std::runtime_error("truncated infos.bin");
        done += size_t(n);
    }
    info.close();
    o.k = values[0]; uint32_t maximum = values[1];
    if (!o.k || o.k > 128 || !maximum) throw std::runtime_error("invalid infos.bin");
    o.key_bytes = (o.k + 3) / 4;
    o.count_bytes = maximum <= 255 ? 1 : maximum <= 65535 ? 2 : maximum <= 16777215 ? 3 : 4;
    return o;
}

// Output is opened without truncation until all selected input identities have
// been checked, including symlinks and hard links. The caller owns the fd.
int open_output(const Options& o) {
    std::array<struct stat, 3> inputs{};
    const std::array<std::string, 3> names = {"infos.bin", "high.bin", "low.bin"};
    size_t count = o.low() ? 3 : 2;
    for (size_t i = 0; i < count; ++i) {
        if (::stat((o.directory + names[i]).c_str(), &inputs[i])) io_error("stat " + names[i]);
        if (!S_ISREG(inputs[i].st_mode)) throw std::runtime_error("input is not a regular file: " + names[i]);
    }
    int fd = ::open(o.output.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) io_error("open output");
    try {
        struct stat out{};
        if (::fstat(fd, &out)) io_error("stat output");
        for (size_t i = 0; i < count; ++i)
            if (out.st_dev == inputs[i].st_dev && out.st_ino == inputs[i].st_ino)
                throw std::runtime_error("output aliases input " + names[i]);
        if (S_ISREG(out.st_mode) && ::ftruncate(fd, 0)) io_error("truncate output");
    } catch (...) { ::close(fd); throw; }
    return fd;
}

// A persistent pool. Both the queue and the running task descriptors are
// bounded; all failures are delivered on the parser thread after tasks drain.
class Workers {
    std::vector<std::thread> threads_;
    std::deque<std::function<void()>> queue_;
    std::mutex mutex_;
    std::condition_variable ready_, space_, idle_;
    size_t pending_ = 0, limit_;
    bool stop_ = false;
    std::exception_ptr error_;
    void run() noexcept {
        for (;;) {
            std::function<void()> task;
            bool execute;
            {
                std::unique_lock lock(mutex_);
                ready_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) return;
                task = std::move(queue_.front()); queue_.pop_front();
                execute = !error_;
                space_.notify_one();
            }
            if (execute) try { task(); }
            catch (...) { std::lock_guard lock(mutex_); if (!error_) error_ = std::current_exception(); }
            // Destroy captures before publishing completion (input window lifetime).
            task = {};
            {
                std::lock_guard lock(mutex_);
                if (--pending_ == 0) idle_.notify_all();
                if (error_) space_.notify_all();
            }
        }
    }
public:
    explicit Workers(unsigned count) : limit_(size_t(count) * 2) {
        threads_.reserve(count);
        try { for (unsigned i = 0; i < count; ++i) threads_.emplace_back(&Workers::run, this); }
        catch (...) { shutdown(); throw; }
    }
    ~Workers() { shutdown(); }
    void shutdown() noexcept {
        { std::lock_guard lock(mutex_); stop_ = true; }
        ready_.notify_all();
        for (auto& thread : threads_) if (thread.joinable()) thread.join();
    }
    template<class F> void submit(F&& task) {
        std::unique_lock lock(mutex_);
        space_.wait(lock, [&] { return queue_.size() < limit_ || error_; });
        if (error_) std::rethrow_exception(error_);
        queue_.emplace_back(std::forward<F>(task)); ++pending_;
        ready_.notify_one();
    }
    void wait() {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [&] { return !pending_; });
        if (error_) std::rethrow_exception(error_);
    }
};

class AsyncWriter {
    struct Block { std::unique_ptr<char[]> data; size_t bytes = 0; };
    File file_;
    std::array<Block, BlockCount> blocks_;
    std::array<size_t, BlockCount> free_{}, ready_{};
    size_t free_count_ = BlockCount, head_ = 0, queued_ = 0;
    std::mutex mutex_;
    std::condition_variable has_free_, has_ready_;
    std::thread thread_;
    bool done_ = false;
    std::exception_ptr error_;
    void run() noexcept {
        try {
            for (;;) {
                size_t id;
                {
                    std::unique_lock lock(mutex_);
                    has_ready_.wait(lock, [&] { return done_ || queued_; });
                    if (!queued_) return;
                    id = ready_[head_]; head_ = (head_ + 1) % BlockCount; --queued_;
                }
                write_all(file_.get(), blocks_[id].data.get(), blocks_[id].bytes);
                discard(id);
            }
        } catch (...) {
            { std::lock_guard lock(mutex_); error_ = std::current_exception(); }
            has_free_.notify_all(); has_ready_.notify_all();
        }
    }
public:
    explicit AsyncWriter(int fd) : file_(fd) {
        for (size_t i = 0; i < BlockCount; ++i) {
            blocks_[i].data.reset(new char[BlockBytes]); free_[i] = i;
        }
        thread_ = std::thread(&AsyncWriter::run, this);
    }
    ~AsyncWriter() { stop(); }
    void stop() noexcept {
        { std::lock_guard lock(mutex_); done_ = true; }
        has_ready_.notify_all();
        if (thread_.joinable()) thread_.join();
    }
    size_t acquire() {
        std::unique_lock lock(mutex_);
        has_free_.wait(lock, [&] { return free_count_ || error_; });
        if (error_) std::rethrow_exception(error_);
        return free_[--free_count_];
    }
    char* data(size_t id) { return blocks_[id].data.get(); }
    void discard(size_t id) noexcept {
        { std::lock_guard lock(mutex_); free_[free_count_++] = id; }
        has_free_.notify_one();
    }
    void publish(size_t id, size_t bytes) {
        std::lock_guard lock(mutex_);
        if (error_) std::rethrow_exception(error_);
        blocks_[id].bytes = bytes;
        ready_[(head_ + queued_) % BlockCount] = id; ++queued_;
        has_ready_.notify_one();
    }
    void finish() {
        stop();
        if (error_) std::rethrow_exception(error_);
        file_.close();
    }
};
class TextBlock {
    AsyncWriter& writer_;
    size_t id_ = BlockCount, bytes_ = 0;
public:
    explicit TextBlock(AsyncWriter& writer) : writer_(writer) {}
    ~TextBlock() { if (id_ != BlockCount) writer_.discard(id_); }
    char* reserve(size_t n) {
        if (id_ != BlockCount && bytes_ + n > BlockBytes) flush();
        if (id_ == BlockCount) { id_ = writer_.acquire(); bytes_ = 0; }
        return writer_.data(id_) + bytes_;
    }
    void advance(size_t n) { bytes_ += n; }
    void flush() {
        if (id_ == BlockCount) return;
        writer_.publish(id_, bytes_); id_ = BlockCount; bytes_ = 0;
    }
};

struct DecodeTables {
    std::array<std::array<char, 4>, 256> forward{}, reverse{};
    std::array<uint8_t, 256> reversed{};
    constexpr DecodeTables() {
        for (unsigned b = 0; b < 256; ++b) for (unsigned j = 0; j < 4; ++j) {
            forward[b][j] = "ACGT"[(b >> (6 - 2*j)) & 3];
            reverse[b][j] = "ACGT"[(b >> (2*j)) & 3];
            reversed[b] |= uint8_t(((b >> (2*j)) & 3) << (6 - 2*j));
        }
    }
};
constexpr DecodeTables tables;
struct Range {
    const uint8_t* data = nullptr;
    std::array<uint8_t, 32> prefix{};
    std::array<uint8_t, 36> single{};
    uint32_t records = 0;
    uint8_t prefix_bytes = 0;
    bool copied = false;
    const uint8_t* payload() const { return copied ? single.data() : data; }
};
struct Task {
    std::array<Range, MaxRanges> ranges;
    size_t size = 0, records = 0;
};
// Conservative bounded descriptor/queue accounting, plus allocator, run I/O,
// stream/heap state and 8 MiB management allowance. Stacks and OS cache are not
// part of this managed-memory budget. Histogram storage is reused per pass.
uint64_t fixed_memory(const Options& o) {
    return 8 * MiB + BlockCount * BlockBytes + 8 * MiB +
        uint64_t(o.workers) * (5 * sizeof(Task) + 2 * RadixBuckets * sizeof(size_t) + 1024);
}
uint32_t count_of(const uint8_t* p, unsigned bytes) {
    uint32_t count = 0;
    std::memcpy(&count, p, bytes);
    return count;
}
uint8_t key_byte(const Range& range, const uint8_t* suffix, size_t i) {
    return i < range.prefix_bytes ? range.prefix[i] : suffix[i - range.prefix_bytes];
}
void write_count(char* start, char* cursor, uint32_t count, TextBlock& block) {
    *cursor++ = '\t';
    cursor = std::to_chars(cursor, cursor + 10, count).ptr;
    *cursor++ = '\n'; block.advance(size_t(cursor - start));
}
void write_stored(const Range& range, const uint8_t* suffix, uint32_t count,
                  const Options& o, TextBlock& block) {
    char* start = block.reserve(o.k + 12), *cursor = start;
    size_t byte = o.key_bytes - 1;
    unsigned tail = o.k % 4;
    if (tail) {
        std::memcpy(cursor, tables.reverse[key_byte(range,suffix,byte)].data() + 4-tail, tail);
        cursor += tail;
    } else {
        std::memcpy(cursor, tables.reverse[key_byte(range,suffix,byte)].data(), 4); cursor += 4;
    }
    while (byte) {
        --byte;
        std::memcpy(cursor, tables.reverse[key_byte(range,suffix,byte)].data(), 4); cursor += 4;
    }
    write_count(start,cursor,count,block);
}
void stream_task(const Task& task, unsigned cb, const Options& o, AsyncWriter& writer) {
    TextBlock block(writer);
    for (size_t j = 0; j < task.size; ++j) {
        const auto& range = task.ranges[j];
        size_t suffix = o.key_bytes - range.prefix_bytes, stride = suffix + cb;
        const uint8_t* p = range.payload();
        for (size_t i = 0; i < range.records; ++i, p += stride) {
            uint32_t count = cb ? count_of(p + suffix,cb) : 1;
            if (o.accepts(count)) write_stored(range,p,count,o,block);
        }
    }
    block.flush();
}
class Streaming {
    Workers& pool_;
    AsyncWriter& writer_;
    const Options& options_;
public:
    Streaming(Workers& pool, AsyncWriter& writer, const Options& o) : pool_(pool),writer_(writer),options_(o) {}
    size_t task_limit() const { return TaskTextBytes / (options_.k + 12); }
    void push(Task&& task, unsigned cb) {
        pool_.submit([this,task=std::move(task),cb] { stream_task(task,cb,options_,writer_); });
    }
    void flush(unsigned) { pool_.wait(); }
};

// Only the parser touches group state. Tasks carry immutable prefixes and own
// the one-record boundary copies; all other ranges refer directly to this
// window. Even on parser/submission failure, workers drain before AIO teardown.
template<class Consumer>
void read_groups(const std::string& path, unsigned cb, const Options& o,
                 Workers& pool, Consumer& consumer) {
    AioGroupReader reader;
    reader.open(path);
    std::array<uint8_t,37> header{};
    std::array<uint8_t,36> carry{};
    size_t header_used = 0, header_need = 5, carry_used = 0;
    size_t prefix = 0, stride = 0;
    uint32_t remaining = 0;
    Task task;
    auto submit = [&] {
        if (task.size) { consumer.push(std::move(task),cb); task = Task{}; }
    };
    auto add = [&](Range&& range) {
        task.records += range.records; task.ranges[task.size++] = std::move(range);
        if (task.size == MaxRanges || task.records >= consumer.task_limit()) submit();
    };
    try {
        const uint8_t* data; size_t bytes;
        while (reader.next(data,bytes)) {
            size_t pos = 0;
            while (pos < bytes) {
                if (!remaining) {
                    size_t take = std::min(header_need - header_used, bytes - pos);
                    std::memcpy(header.data()+header_used,data+pos,take); header_used += take; pos += take;
                    if (header_used != header_need) continue;
                    if (header_need == 5) {
                        prefix = header[0]; remaining = count_of(header.data()+1,4);
                        if (prefix >= o.key_bytes || !remaining) throw std::runtime_error("invalid group header in " + path);
                        stride = o.key_bytes - prefix + cb;
                        header_need = 5 + prefix;
                        // Prefix may still need bytes before payload processing.
                    }
                }
                if (header_used < header_need) {
                    size_t take = std::min(header_need - header_used, bytes - pos);
                    std::memcpy(header.data()+header_used,data+pos,take); header_used += take; pos += take;
                    if (header_used != header_need) continue;
                }
                if (carry_used) {
                    size_t take = std::min(stride - carry_used, bytes - pos);
                    std::memcpy(carry.data()+carry_used,data+pos,take); carry_used += take; pos += take;
                    if (carry_used != stride) continue;
                    Range range; range.prefix_bytes = prefix; range.records = 1; range.copied = true;
                    std::memcpy(range.prefix.data(),header.data()+5,prefix);
                    std::memcpy(range.single.data(),carry.data(),stride);
                    add(std::move(range)); carry_used = 0; --remaining;
                } else {
                    size_t count = std::min<size_t>(remaining, (bytes-pos)/stride);
                    if (count) {
                        count = std::min(count, consumer.task_limit() - task.records);
                        Range range; range.prefix_bytes = prefix; range.records = count; range.data = data + pos;
                        std::memcpy(range.prefix.data(),header.data()+5,prefix);
                        add(std::move(range)); pos += count * stride; remaining -= count;
                    } else if (pos < bytes) {
                        carry_used = bytes - pos;
                        std::memcpy(carry.data(),data+pos,carry_used); pos = bytes;
                    }
                }
                if (!remaining) { header_used = 0; header_need = 5; }
            }
            submit(); consumer.flush(cb); reader.release();
        }
        if (remaining || header_used || carry_used) throw std::runtime_error("truncated group in " + path);
        reader.join();
    } catch (...) {
        reader.stop();
        try { pool.wait(); } catch (...) {}
        reader.release(); reader.join(); throw;
    }
}

template<size_t N> struct Entry { std::array<uint64_t,N> key; uint32_t count; };
template<size_t N>
Entry<N> make_entry(const Range& range, const uint8_t* suffix, uint32_t count, const Options& o) {
    Entry<N> entry{}; entry.count = count;
    const unsigned padding = (4 - o.k % 4) % 4 * 2;
    for (size_t j = 0; j < o.key_bytes; ++j) {
        const size_t stored = o.key_bytes - 1 - j;
        unsigned byte = unsigned(tables.reversed[key_byte(range,suffix,stored)]) << padding;
        if (padding && stored) byte |= tables.reversed[key_byte(range,suffix,stored-1)] >> (8-padding);
        entry.key[j/8] |= uint64_t(byte & 255) << (56 - 8*(j%8));
    }
    return entry;
}
template<size_t N> unsigned digit(const Entry<N>& entry, unsigned shift) {
    size_t word = N - 1 - shift/64;
    unsigned bits = shift % 64;
    uint64_t value = entry.key[word] >> bits;
    if (bits > 53 && word) value |= entry.key[word-1] << (64-bits);
    return unsigned(value & (RadixBuckets-1));
}
template<size_t N> void write_entry(const Entry<N>& entry, const Options& o, TextBlock& block) {
    char* start = block.reserve(o.k + 12), *cursor = start;
    for (size_t j = 0; j < o.key_bytes; ++j) {
        uint8_t byte = uint8_t(entry.key[j/8] >> (56-8*(j%8)));
        size_t take = std::min<size_t>(4,o.k-4*j);
        std::memcpy(cursor,tables.forward[byte].data(),take); cursor += take;
    }
    write_count(start,cursor,entry.count,block);
}

// Internal runs are text-direction bytes followed by a native uint32 count.
// Only this directory's own numbered files are ever read or removed.
class Runs {
    std::string directory_;
    const Options& options_;
public:
    uint64_t bytes_written = 0;
    explicit Runs(const Options& o) : options_(o) {}
    ~Runs() {
        if (!directory_.empty()) { std::error_code ec; std::filesystem::remove_all(directory_,ec); }
    }
    std::string path(size_t round, size_t index) {
        if (directory_.empty()) {
            std::string pattern = options_.directory + "dump-sort-XXXXXX";
            std::vector<char> name(pattern.begin(),pattern.end()); name.push_back(0);
            if (!::mkdtemp(name.data())) io_error("mkdtemp sort directory");
            directory_ = name.data();
        }
        return directory_ + '/' + std::to_string(round) + '-' + std::to_string(index);
    }
    void remove(size_t round, size_t index) {
        if (::unlink(path(round,index).c_str())) io_error("unlink sort run");
    }
};
class RunWriter {
    File file_;
    std::unique_ptr<uint8_t[]> buffer_{new uint8_t[MiB]};
    size_t used_ = 0;
    unsigned width_;
    Runs& runs_;
public:
    RunWriter(const std::string& path, unsigned width, Runs& runs)
        : file_(::open(path.c_str(), O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600)), width_(width),runs_(runs) {
        if (file_.get() < 0) io_error("create sort run");
    }
    template<size_t N> void put(const Entry<N>& entry) {
        if (used_ + width_ + 4 > MiB) flush();
        for (size_t i = 0; i < width_; ++i)
            buffer_[used_++] = uint8_t(entry.key[i/8] >> (56-8*(i%8)));
        std::memcpy(buffer_.get()+used_,&entry.count,4); used_ += 4;
    }
    void flush() { write_all(file_.get(),buffer_.get(),used_); runs_.bytes_written += used_; used_ = 0; }
    void finish() { flush(); file_.close(); }
};
template<size_t N> class RunReader {
    File file_;
    std::unique_ptr<uint8_t[]> buffer_{new uint8_t[MiB]};
    size_t pos_ = 0, bytes_ = 0;
    unsigned width_;
    bool byte(uint8_t& value) {
        if (pos_ == bytes_) {
            ssize_t n;
            do { n = ::read(file_.get(),buffer_.get(),MiB); } while (n < 0 && errno == EINTR);
            if (n < 0) io_error("read sort run");
            bytes_ = size_t(n); pos_ = 0;
            if (!n) return false;
        }
        value = buffer_[pos_++]; return true;
    }
public:
    RunReader(const std::string& path, unsigned width) : file_(::open(path.c_str(),O_RDONLY|O_CLOEXEC)),width_(width) {
        if (file_.get() < 0) io_error("open sort run");
    }
    bool next(Entry<N>& entry) {
        uint8_t value;
        if (!byte(value)) return false;
        entry = {}; entry.key[0] = uint64_t(value) << 56;
        for (size_t i = 1; i < width_+4; ++i) {
            if (!byte(value)) throw std::runtime_error("truncated sort run");
            if (i < width_) entry.key[i/8] |= uint64_t(value) << (56-8*(i%8));
            else entry.count |= uint32_t(value) << (8*(i-width_));
        }
        return true;
    }
};

template<size_t N> class Sorting {
    using Record = Entry<N>;
    const Options& o_;
    Workers& pool_;
    AsyncWriter& writer_;
    Runs runs_;
    size_t capacity_, size_ = 0, run_count_ = 0;
    std::allocator<Record> allocator_;
    Record *first_ = nullptr, *second_ = nullptr;
    std::vector<std::array<size_t,RadixBuckets>> histograms_;
    struct Batch { Task task; size_t selected = 0; };
    std::vector<Batch> batch_;
    void sort() {
        Record *source = first_, *destination = second_;
        size_t workers = std::min<size_t>(o_.workers,size_);
        if (size_ <= 1) return;
        try {
            for (unsigned pass = 0; pass < 2*o_.k; pass += 11) {
                unsigned shift = N*64 - 2*o_.k + pass;
                for (size_t t = 0; t < workers; ++t) pool_.submit([&,t,shift] {
                    auto& hist = histograms_[t]; hist.fill(0);
                    size_t begin = size_*t/workers, end = size_*(t+1)/workers;
                    for (size_t i = begin; i < end; ++i) ++hist[digit(source[i],shift)];
                });
                pool_.wait();
                size_t offset = 0;
                for (size_t b = 0; b < RadixBuckets; ++b) for (size_t t = 0; t < workers; ++t) {
                    size_t count = histograms_[t][b]; histograms_[t][b] = offset; offset += count;
                }
                for (size_t t = 0; t < workers; ++t) pool_.submit([&,t,shift] {
                    auto& cursor = histograms_[t];
                    size_t begin = size_*t/workers, end = size_*(t+1)/workers;
                    for (size_t i = begin; i < end; ++i) destination[cursor[digit(source[i],shift)]++] = source[i];
                });
                pool_.wait(); std::swap(source,destination);
            }
        } catch (...) {
            try { pool_.wait(); } catch (...) {}
            throw;
        }
        first_ = source; second_ = destination;
    }
    void spill() {
        sort();
        RunWriter run(runs_.path(0,run_count_),o_.key_bytes,runs_);
        for (size_t i = 0; i < size_; ++i) run.put(first_[i]);
        run.finish(); ++run_count_; size_ = 0;
    }
    void release_arrays() noexcept {
        if (first_) allocator_.deallocate(first_,capacity_);
        if (second_) allocator_.deallocate(second_,capacity_);
        first_ = second_ = nullptr;
    }
    size_t fan_in() const {
        struct rlimit limit{};
        if (::getrlimit(RLIMIT_NOFILE,&limit)) io_error("getrlimit");
        uint64_t descriptors = limit.rlim_cur > 16 ? limit.rlim_cur - 16 : 0;
        // The arrays and AIO reader have gone away. Keep all fixed overhead
        // reserved, except for the 8 MiB AIO buffers, and reserve run output.
        uint64_t available = o_.memory - (fixed_memory(o_) - 8*MiB) - MiB;
        return size_t(std::min<uint64_t>({64,descriptors,available/MiB}));
    }
    template<class Sink> void merge(size_t round, size_t begin, size_t end, Sink& sink) {
        struct Node { Record record; size_t reader; };
        struct Greater {
            bool operator()(const Node& a,const Node& b) const { return a.record.key > b.record.key; }
        };
        std::vector<std::unique_ptr<RunReader<N>>> readers;
        readers.reserve(end-begin);
        std::priority_queue<Node,std::vector<Node>,Greater> heap;
        for (size_t i = begin; i < end; ++i) {
            readers.push_back(std::make_unique<RunReader<N>>(runs_.path(round,i),o_.key_bytes));
            Record entry;
            if (readers.back()->next(entry)) heap.push({entry,i-begin});
        }
        while (!heap.empty()) {
            auto node = heap.top(); heap.pop(); sink(node.record);
            if (readers[node.reader]->next(node.record)) heap.push(node);
        }
    }
public:
    Sorting(const Options& o, Workers& pool, AsyncWriter& writer)
        : o_(o),pool_(pool),writer_(writer),runs_(o),
          capacity_(size_t(std::min<uint64_t>((o.memory-fixed_memory(o))/(2*sizeof(Record)),
                                            std::numeric_limits<size_t>::max()/sizeof(Record)))) {
        if (!capacity_) throw std::runtime_error("memory budget cannot hold sorting records");
        histograms_.resize(o.workers); batch_.reserve(size_t(o.workers)*2);
        try { first_ = allocator_.allocate(capacity_); second_ = allocator_.allocate(capacity_); }
        catch (...) { release_arrays(); throw; }
        std::cout << "sort capacity " << capacity_ << '\n';
    }
    ~Sorting() { release_arrays(); }
    size_t task_limit() const { return std::min(capacity_,TaskTextBytes/(o_.k+12)); }
    void push(Task&& task, unsigned cb) {
        batch_.push_back({std::move(task),0});
        if (batch_.size() == size_t(o_.workers)*2) process(cb);
    }
    void process(unsigned cb) {
        try {
            for (auto& b : batch_) pool_.submit([&,ptr=&b] {
                size_t selected = 0;
                for (size_t j = 0; j < ptr->task.size; ++j) {
                    const auto& r = ptr->task.ranges[j];
                    if (!cb) { selected += r.records; continue; }
                    size_t suffix = o_.key_bytes-r.prefix_bytes, stride = suffix+cb;
                    const uint8_t* p = r.payload()+suffix;
                    for (size_t i = 0; i < r.records; ++i,p+=stride) selected += o_.accepts(count_of(p,cb));
                }
                ptr->selected = selected;
            });
            pool_.wait();
            for (auto& b : batch_) {
                if (!b.selected) continue;
                if (b.selected > capacity_-size_) { pool_.wait(); spill(); }
                size_t start = size_; size_ += b.selected;
                pool_.submit([&,ptr=&b,start] {
                    size_t position = start;
                    for (size_t j = 0; j < ptr->task.size; ++j) {
                        const auto& r = ptr->task.ranges[j];
                        size_t suffix = o_.key_bytes-r.prefix_bytes, stride = suffix+cb;
                        const uint8_t* p = r.payload();
                        for (size_t i = 0; i < r.records; ++i,p+=stride) {
                            uint32_t count = cb ? count_of(p+suffix,cb) : 1;
                            if (o_.accepts(count)) first_[position++] = make_entry<N>(r,p,count,o_);
                        }
                    }
                });
            }
            pool_.wait(); batch_.clear();
        } catch (...) { try { pool_.wait(); } catch (...) {} throw; }
    }
    void flush(unsigned cb) { if (!batch_.empty()) process(cb); }
    void finish() {
        if (!run_count_) {
            sort(); TextBlock block(writer_);
            for (size_t i = 0; i < size_; ++i) write_entry(first_[i],o_,block);
            block.flush();
            std::cout << "sort temporary bytes 0\nsort merge passes 0\n"; return;
        }
        if (size_) spill();
        release_arrays();
        size_t fan = fan_in(), round = 0, passes = 0;
        if (run_count_ > 1 && fan < 2) throw std::runtime_error("insufficient memory or file descriptors for two-way merge");
        while (run_count_ > fan) {
            size_t next = 0;
            for (size_t begin = 0; begin < run_count_; begin += fan) {
                size_t end = std::min(run_count_,begin+fan);
                RunWriter output(runs_.path(round+1,next++),o_.key_bytes,runs_);
                auto sink = [&](const Record& entry) { output.put(entry); };
                merge(round,begin,end,sink); output.finish();
                for (size_t i = begin; i < end; ++i) runs_.remove(round,i);
            }
            run_count_ = next; ++round; ++passes;
        }
        TextBlock block(writer_);
        auto sink = [&](const Record& entry) { write_entry(entry,o_,block); };
        merge(round,0,run_count_,sink); block.flush(); ++passes;
        std::cout << "sort temporary bytes " << runs_.bytes_written << "\nsort merge passes " << passes << '\n';
    }
};

template<size_t N> void sorted_dump(const Options& o, Workers& pool, AsyncWriter& writer) {
    Sorting<N> sorting(o,pool,writer);
    read_groups(o.directory+"high.bin",o.count_bytes,o,pool,sorting);
    if (o.low()) {
        read_groups(o.directory+"low.bin",0,o,pool,sorting);
    }
    sorting.finish();
}
}
int main(int argc,char** argv) {
    try {
        Options o = options(argc,argv);
        if (fixed_memory(o) >= o.memory) throw std::runtime_error("max_memory_gb cannot hold fixed dump buffers");
        AsyncWriter writer(open_output(o));
        Workers pool(o.workers);
        std::cout << "progress 0%\n";
        if (o.sorted) {
            if (o.k <= 32) sorted_dump<1>(o,pool,writer);
            else if (o.k <= 64) sorted_dump<2>(o,pool,writer);
            else sorted_dump<4>(o,pool,writer);
        } else {
            Streaming streaming(pool,writer,o);
            read_groups(o.directory+"high.bin",o.count_bytes,o,pool,streaming);
            if (o.low()) read_groups(o.directory+"low.bin",0,o,pool,streaming);
        }
        writer.finish(); std::cout << "progress 100%\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "dump_tool: " << error.what() << '\n'; return 1;
    }
}
