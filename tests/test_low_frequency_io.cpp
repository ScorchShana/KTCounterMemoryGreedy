#include "../src/LowFrequencyReader.h"
#include "../src/LowFrequencyParser.h"
#include "../src/ExportWriter.h"
#include "../src/ParserThreadPool.h"
#include "low_msd_test_helpers.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <condition_variable>
#include <mutex>
#include <chrono>
#include <stdexcept>
#include <sys/wait.h>
#include <vector>

struct LowFrequencyReaderTestAccess {
    static void read_with(LowFrequencyReader& reader, ssize_t (*fn)(int, void*, size_t, off_t)) { reader.pread_ = fn; }
};

struct ExportWriterTestAccess {
    static void limit(ExportWriter& writer, uint64_t bytes) { writer.shard_limit_bytes_ = bytes; }
};

namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

template <uint32_t N>
void check_io(const std::filesystem::path& directory, uint32_t k, size_t count,
    size_t block_size = LOW_FREQUENCY_RING_MEMORY_POOL_BLOCK_SIZE,
    size_t available_output_blocks = PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY)
{
    k_length = k;
    const size_t width = kmer_bytes();
    const auto input_path = directory / "pending.bin";
    const auto output_path = directory / "low.bin";
    std::vector<uint8_t> input(count * width);
    std::vector<std::vector<uint8_t>> expected;
    for (size_t i = 0; i < count; ++i) {
        auto* key = input.data() + i * width;
        key[0] = i % 256;
        for (size_t j = 0; j < 4; ++j) key[j + 1] = (i >> (8 * j)) & 255;
        if (i % 3 != 0)
            expected.emplace_back(input.begin() + i * width, input.begin() + (i + 1) * width);
    }
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(block_size, 1);
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> parsed(PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, 2);
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    ExportWriter pending(&exports, input_path.string(), ExportWriter::OutputKind::Pending);
    pending.start();
    const size_t export_records = exports.blockSize() / width;
    for (size_t pos = 0; pos < count;) {
        char* block;
        exports.producer_dequeue(block);
        const size_t n = std::min(export_records, count - pos);
        std::memcpy(block, input.data() + pos * width, n * width);
        exports.producer_enqueue({block, n});
        pos += n;
    }
    exports.producer_set_finished();
    pending.join();
    require(!std::filesystem::exists(input_path), "pending used the unnumbered filename");
    if (input.size() <= LOW_PENDING_SHARD_BYTES)
        require(pending.shard_count() == 1, "empty or complete input created an extra shard");
    std::vector<std::vector<uint8_t>> all_pending;
    for (uint64_t i = 0; i < pending.shard_count(); ++i) {
        const auto path = input_path.string() + "." + std::to_string(i);
        auto keys = low_msd_test::read(path, width, 37);
        all_pending.insert(all_pending.end(), keys.begin(), keys.end());
        require(std::filesystem::file_size(path) < LOW_PENDING_SHARD_BYTES + msd_prefix::BUFFER_BYTES,
            "pending shard exceeded encoded batch allowance");
    }
    std::vector<std::vector<uint8_t>> all_input;
    for (size_t i = 0; i < count; ++i) all_input.emplace_back(input.begin() + i * width, input.begin() + (i + 1) * width);
    std::sort(all_input.begin(), all_input.end());
    std::sort(all_pending.begin(), all_pending.end());
    require(all_input == all_pending, "pending encoded multiset differs");
    require(!std::filesystem::exists(input_path.string() + "." + std::to_string(pending.shard_count())),
        "pending created an empty trailing shard");
    exports.reset_producers(1);
    // Decoding must clear unused bytes even when a block contains first-pass data.
    for (size_t i = 0; i < parsed.capacity(); ++i) {
        char* block;
        parsed.producer_dequeue(block);
        std::memset(block, 0xff, parsed.blockSize());
        parsed.consumer_enqueue(block);
    }
    std::vector<char*> reserved;
    for (size_t i = available_output_blocks; i < parsed.capacity(); ++i) {
        char* block;
        parsed.producer_dequeue(block);
        reserved.push_back(block);
    }
    LowFrequencyReader reader(reads);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners;
    for (int i = 0; i < 2; ++i)
        owners.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
    for (size_t i = 0; i < prefix_owners.size(); ++i) prefix_owners[i] = i % owners.size();
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    // Low-pass thread count is independent of the first-pass parser count.
    ParserThreadPool parsers(owners, &global, nullptr, &parsed, 1);
    ExportWriter writer(&exports, output_path.string(), ExportWriter::OutputKind::FinalLow);
    writer.start();
    parsers.start_low_pass(reads, 2);
    reader.start(input_path.string(), pending.shard_count());
    std::vector<std::thread> consumers;
    for (int thread = 0; thread < 2; ++thread) {
        consumers.emplace_back([&, thread] {
            char* output;
            exports.producer_dequeue(output);
            size_t kept = 0;
            const size_t capacity = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / width;
            SpinBackoff<> backoff;
            content_type content;
            bool local = false;
            auto dequeue = [&] {
                local = owners[thread]->try_dequeue(content);
                return local || global.try_dequeue(content);
            };
            while (true) {
                bool available = dequeue();
                if (!available && parsed.producer_finished()) {
                    available = dequeue();
                    if (!available) break;
                }
                if (!available) {
                    backoff.backoff();
                    continue;
                }
                backoff.decay();
                const auto* keys = reinterpret_cast<const unsigned char*>(content.data);
                require(content.length > 0, "parser published an empty block");
                const auto owner = prefix_owners[keys[0]];
                require(!local || owner == thread, "block reached the wrong owner queue");
                for (size_t j = 0; j < content.length; ++j) {
                    require(prefix_owners[keys[j * width]] == owner, "block mixes classifier owners");
                    size_t id = 0;
                    for (size_t x = 0; x < 4; ++x) id |= size_t(keys[j * width + x + 1]) << (8 * x);
                    require(id < count && std::memcmp(keys + j * width, input.data() + id * width, width) == 0,
                        "parser changed a compact record");
                    if (id % 3 != 0) {
                        std::memcpy(output + kept++ * width, keys + j * width, width);
                        if (kept == capacity) {
                            exports.producer_enqueue({output, kept});
                            exports.producer_dequeue(output);
                            kept = 0;
                        }
                    }
                }
                parsed.consumer_enqueue(content.data);
            }
            exports.producer_enqueue({output, kept});
        });
    }
    reader.join();
    parsers.join();
    require(parsed.producer_finished(), "low-pass producers did not all finish");
    for (auto& consumer : consumers) consumer.join();
    exports.producer_set_finished();
    writer.join();
    require(reader.records_read() == count, "reader lost records");
    for (uint64_t i = 0; i < pending.shard_count(); ++i)
        require(!std::filesystem::exists(input_path.string() + "." + std::to_string(i)), "reader retained a consumed shard");
    require(parsers.get_total_read_kmer() == 0, "second pass changed the FASTQ input count");
    auto actual = low_msd_test::read(output_path, width);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    require(actual == expected, "output multiset differs");
    content_type leftover;
    require(!parsed.consumer_try_dequeue(leftover), "parser used the pool's public task queue");
    for (auto* block : reserved) parsed.consumer_enqueue(block);

    auto check_returned = [](auto& pool) {
        char* block;
        std::vector<char*> returned;
        while (pool.producer_try_dequeue(block)) returned.push_back(block);
        require(returned.size() == pool.capacity(), "some blocks were not returned");
        std::sort(returned.begin(), returned.end());
        require(std::adjacent_find(returned.begin(), returned.end()) == returned.end(), "block returned twice");
        for (auto* data : returned) pool.consumer_enqueue(data);
    };
    check_returned(reads);
    check_returned(parsed);
    check_returned(exports);
}

// Independent wire fixtures include raw groups larger than one input block.
void check_encoded_reader(const std::filesystem::path& directory)
{
    for (const uint32_t k : {1U, 28U, 33U, 55U, 128U}) {
        k_length = k;
        const size_t width = kmer_bytes();
        RunState state;
        ActiveRunScope scope(state);
        const auto base = (directory / ("encoded-" + std::to_string(k))).string();
        std::vector<std::vector<uint8_t>> expected;
        {
            std::ofstream stream(base + ".0", std::ios::binary);
            std::vector<size_t> prefixes{0};
            if (width > 1) prefixes.push_back(1);
            if (width > 2) prefixes.push_back(width - 1);
            for (size_t prefix : prefixes) {
                const uint32_t count = uint32_t(std::min<size_t>(90003, msd_prefix::batch_records(width)));
                stream.put(char(prefix));
                stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
                std::vector<uint8_t> common(prefix, 0x51);
                stream.write(reinterpret_cast<const char*>(common.data()), common.size());
                for (uint32_t i = 0; i < count; ++i) {
                    std::vector<uint8_t> key(width);
                    std::copy(common.begin(), common.end(), key.begin());
                    for (size_t j = prefix; j < width; ++j) key[j] = uint8_t((i * 17 + j * 31) >> ((j % 3) * 3));
                    stream.write(reinterpret_cast<const char*>(key.data() + prefix), width - prefix);
                    expected.push_back(std::move(key));
                }
            }
        }
        const uint64_t encoded_bytes = std::filesystem::file_size(base + ".0");
        // Empty shards are valid and must still be closed and removed.
        std::ofstream(base + ".1", std::ios::binary);
        SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(4096, 1);
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> parsed(65536, 4);
        std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners;
        for (int i = 0; i < 4; ++i) owners.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
        for (size_t i = 0; i < prefix_owners.size(); ++i) prefix_owners[i] = i % owners.size();
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
        ParserThreadPool parsers(owners, &global, nullptr, &parsed, 1);
        LowFrequencyReader reader(reads);
        parsers.start_low_pass(reads, 4);
        reader.start(base, 2);
        std::vector<std::vector<uint8_t>> actual;
        while (!state.failed()) {
            content_type content;
            bool found = global.try_dequeue(content);
            size_t queue = owners.size();
            for (size_t i = 0; !found && i < owners.size(); ++i)
                if ((found = owners[i]->try_dequeue(content))) queue = i;
            if (!found) {
                if (parsed.producer_finished()) {
                    bool empty = global.size() == 0;
                    for (const auto& owner : owners) empty = empty && owner->size() == 0;
                    if (empty) break;
                }
                std::this_thread::yield();
                continue;
            }
            for (size_t i = 0; i < content.length; ++i) {
                const auto* key = reinterpret_cast<const uint8_t*>(content.data) + i * width;
                if (queue < owners.size() && prefix_owners[key[0]] != queue)
                    state.fail(std::make_exception_ptr(std::runtime_error("encoded group reached wrong owner")), "test");
                actual.emplace_back(key, key + width);
            }
            parsed.consumer_enqueue(content.data);
        }
        reader.join(); parsers.join(); state.check();
        std::sort(actual.begin(), actual.end()); std::sort(expected.begin(), expected.end());
        require(actual == expected && reader.records_read() == expected.size(), "encoded reader/parser multiset differs");
        require(reader.encoded_bytes_read() == encoded_bytes, "reader counted virtual headers as file bytes");
        require(reader.read_calls() <= (encoded_bytes + LowFrequencyReader::READ_CACHE_BYTES - 1) / LowFrequencyReader::READ_CACHE_BYTES,
            "reader issued per-group reads");
        require(!std::filesystem::exists(base + ".0") && !std::filesystem::exists(base + ".1"), "encoded shards retained");
        char* block; size_t returned = 0;
        while (reads.producer_try_dequeue(block)) ++returned;
        require(returned == reads.capacity(), "encoded reader leaked input blocks");
    }
}

ssize_t fragmented_read(int fd, void* data, size_t bytes, off_t offset)
{
    static unsigned calls = 0; // Only the single Reader thread uses this hook.
    if (++calls % 7 == 0) { errno = EINTR; return -1; }
    return ::pread(fd, data, std::min<size_t>(bytes, 7), offset);
}

void check_reader_boundaries(const std::filesystem::path& directory, bool fragmented)
{
    k_length = 128;
    RunState state; ActiveRunScope scope(state);
    const auto base = (directory / (fragmented ? "short-cache" : "full-cache")).string();
    const uint32_t count = uint32_t(msd_prefix::batch_records(32));
    {
        std::ofstream stream(base + ".0", std::ios::binary);
        for (int g = 0; g < 4; ++g) {
            stream.put(31); stream.write(reinterpret_cast<const char*>(&count), sizeof(count));
            std::array<char, 31> prefix; prefix.fill(char(0x40 + g));
            stream.write(prefix.data(), prefix.size());
            for (uint32_t i = 0; i < count; ++i) stream.put(char(i));
        }
    }
    const auto encoded = std::filesystem::file_size(base + ".0");
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(LOW_FREQUENCY_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    LowFrequencyReader reader(reads);
    if (fragmented) LowFrequencyReaderTestAccess::read_with(reader, fragmented_read);
    reader.start(base, 1);
    size_t records = 0, tasks = 0;
    while (!state.failed()) {
        content_type content;
        bool available = reads.consumer_try_dequeue(content);
        if (!available && reads.producer_finished()) {
            available = reads.consumer_try_dequeue(content);
            if (!available) break;
        }
        if (!available) { std::this_thread::yield(); continue; }
        size_t pos = 0, decoded = 0;
        while (pos < content.length) {
            const auto input = std::span(reinterpret_cast<const uint8_t*>(content.data) + pos, content.length - pos);
            const auto header = msd_prefix::parse_group_header(input, 32);
            require(header.prefix_bytes == 31 && header.encoded_bytes <= input.size(), "cache split changed a header");
            const auto* prefix = input.data() + 5;
            const auto* suffix = prefix + 31;
            for (size_t i = 0; i < header.records; ++i) {
                const uint8_t expected = uint8_t(0x40 + records / count);
                require(std::all_of(prefix, prefix + 31, [expected](uint8_t b) { return b == expected; })
                    && suffix[i] == uint8_t(records % count), "cache boundary changed a key");
                ++records;
            }
            decoded += header.records * 32;
            pos += header.encoded_bytes;
        }
        require(decoded <= msd_prefix::BUFFER_BYTES && content.length <= reads.blockSize(), "reader exceeded task work bound");
        ++tasks;
        reads.consumer_enqueue(content.data);
    }
    reader.join(); state.check();
    require(records == 4 * size_t(count) && reader.records_read() == records && tasks >= 4, "reader lost a fragmented group");
    require(reader.encoded_bytes_read() == encoded, "reader counted synthesized headers");
    require(fragmented || reader.read_calls() == (encoded + LowFrequencyReader::READ_CACHE_BYTES - 1) / LowFrequencyReader::READ_CACHE_BYTES,
        "small groups triggered additional syscalls");
}

ssize_t cancel_interrupted_read(int, void*, size_t, off_t)
{
    static bool first = true;
    if (first) {
        first = false;
        RunState::active->fail(std::make_exception_ptr(std::runtime_error("cancel read")), "cancel test");
        errno = EINTR;
        return -1;
    }
    return 0; // Old retry loops terminate so a failing regression can still join.
}

void check_reader_failures(const std::filesystem::path& directory)
{
    k_length = 33;
    for (int mode = 0; mode < 6; ++mode) {
        RunState state; ActiveRunScope scope(state);
        const auto base = (directory / ("reader-bad-" + std::to_string(mode))).string();
        std::vector<uint8_t> bytes(5, 0);
        uint32_t count = mode == 0 ? 0 : mode == 1 ? uint32_t(msd_prefix::batch_records(9) + 1) : 1;
        bytes[0] = mode == 2 ? 9 : mode == 3 ? 8 : 0;
        std::memcpy(bytes.data() + 1, &count, 4);
        if (mode == 4) bytes.resize(13); // One missing payload byte after acquiring an output block.
        {
            std::ofstream file(base + ".0", std::ios::binary);
            file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(64, 1);
        LowFrequencyReader reader(reads);
        if (mode == 5) LowFrequencyReaderTestAccess::read_with(reader, cancel_interrupted_read);
        reader.start(base, 1); reader.join();
        require(state.failed() && !reads.producer_finished() && std::filesystem::exists(base + ".0"),
            "bad/cancelled shard was reported consumed");
        char* block; size_t returned = 0;
        while (reads.producer_try_dequeue(block)) ++returned;
        require(returned == reads.capacity(), "reader failure leaked its private block");
        if (mode == 5) require(reader.read_calls() == 1, "EINTR retry ignored cancellation");
    }
}

void check_parser_failure_cleanup()
{
    k_length = 33; prefix_owners.fill(0);
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(64, 1);
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> parsed(64, 1);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners{
        std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>()};
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    char* block; reads.producer_dequeue(block);
    const uint32_t one = 1;
    block[0] = 1; std::memcpy(block + 1, &one, 4); std::memset(block + 5, 0x51, 9);
    block[14] = 9; std::memcpy(block + 15, &one, 4); // Invalid P, after an owner tail is acquired.
    reads.producer_enqueue({block, 19}); reads.producer_set_finished();
    LowFrequencyParser parser(owners, &global);
    bool rejected = false;
    try { parser.parse_and_push(reads, parsed); } catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "invalid parser group accepted");
    size_t returned = 0;
    while (reads.producer_try_dequeue(block)) ++returned;
    require(returned == reads.capacity(), "parser exception leaked its input block");
    returned = 0;
    while (parsed.producer_try_dequeue(block)) ++returned;
    require(returned == parsed.capacity(), "parser exception leaked owner tail blocks");
}

void check_queue_threshold(size_t queued)
{
    constexpr uint32_t k = 33;
    k_length = k;
    prefix_owners.fill(0);
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(64, 1);
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> parsed(64, 1);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners{
        std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>()};
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    for (size_t i = 0; i < queued; ++i) {
        char* block;
        parsed.producer_dequeue(block);
        std::memset(block, 0, kmer_bytes());
        owners[0]->enqueue({block, 1});
    }
    std::vector<unsigned char> key(kmer_bytes());
    key[4] = 1;
    char* block;
    reads.producer_dequeue(block);
    block[0] = 0;
    const uint32_t one = 1;
    std::memcpy(block + 1, &one, sizeof(one));
    std::memcpy(block + 5, key.data(), key.size());
    reads.producer_enqueue({block, 5 + key.size()});
    reads.producer_set_finished();
    LowFrequencyParser parser( owners, &global);
    parser.parse_and_push(reads, parsed);
    parsed.producer_set_finished();
    const bool spill = queued >= CLASSIFIER_TASK_QUEUE_HALF_WATERMARK;
    require(owners[0]->size() == queued + !spill, "wrong owner queue selection at watermark");
    require(global.size() == size_t(spill), "wrong global queue selection at watermark");
    size_t found = 0;
    content_type content;
    auto collect = [&] {
        require(content.length == 1, "wrong tail block length");
        found += std::memcmp(content.data, key.data(), key.size()) == 0;
        parsed.consumer_enqueue(content.data);
    };
    while (owners[0]->try_dequeue(content)) collect();
    while (global.try_dequeue(content)) collect();
    require(found == 1, "tail was lost or duplicated during queue selection");
    size_t returned = 0;
    while (parsed.producer_try_dequeue(block)) ++returned;
    require(returned == parsed.capacity(), "queue routing leaked output blocks");
}

// Hold the ring full so shard deletion and failures are tested before replay ends.
void check_reader_lifecycle(const std::filesystem::path& directory, int failure)
{
    k_length = 41;
    RunState state;
    ActiveRunScope active(state);
    const auto base = (directory / ("lifecycle-" + std::to_string(failure))).string();
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(5 + kmer_bytes(), 1);
    // Truncation must affect bytes not already buffered by the new Reader.
    const size_t count = reads.capacity() + 4 + LowFrequencyReader::READ_CACHE_BYTES / (5 + kmer_bytes());
    for (int shard = 0; shard < 3; ++shard) {
        std::ofstream out(base + "." + std::to_string(shard), std::ios::binary);
        const size_t begin = shard == 0 ? 0 : 1;
        const size_t end = shard == 0 ? 1 : count;
        for (uint64_t i = begin; i < end; ++i) {
            std::array<char, 11> record{};
            std::memcpy(record.data(), &i, sizeof(i));
            low_msd_test::write_raw(out, std::span(reinterpret_cast<const uint8_t*>(record.data()), record.size()), kmer_bytes());
        }
    }
    LowFrequencyReader reader(reads);
    reader.start(base, 2);
    std::array<content_type, 2> held;
    for (auto& content : held) reads.consumer_dequeue(content);
    // The second record proves that shard 1 has been opened and sized.
    require(!std::filesystem::exists(base + ".0"), "reader waited for consumers before deleting shard 0");
    require(std::filesystem::exists(base + ".1") && !reads.producer_finished(), "reader prematurely completed shard 1");
    if (failure == 1) {
        require(::truncate((base + ".1").c_str(), 0) == 0, "failed to truncate reader fixture");
    } else if (failure == 2) {
        std::filesystem::rename(base + ".1", base + ".saved");
        std::filesystem::create_directory(base + ".1");
    }
    uint64_t next = 0;
    for (auto& content : held) {
        uint64_t id;
        std::memcpy(&id, content.data + 5, sizeof(id));
        require(content.length == 5 + kmer_bytes() && id == next++, "reader changed cross-shard order");
        reads.consumer_enqueue(content.data);
    }
    while (!state.failed()) {
        content_type content;
        bool available = reads.consumer_try_dequeue(content);
        if (!available && reads.producer_finished()) {
            available = reads.consumer_try_dequeue(content);
            if (!available) break;
        }
        if (!available) {
            // Injected failures are asserted below, rather than thrown by backoff.
            std::this_thread::yield();
            continue;
        }
        uint64_t id;
        std::memcpy(&id, content.data + 5, sizeof(id));
        require(content.length == 5 + kmer_bytes() && id == next++, "reader changed cross-shard bytes");
        reads.consumer_enqueue(content.data);
    }
    reader.join();
    if (failure) {
        require(state.failed() && !reads.producer_finished(), "reader failure reported successful EOF");
        require(std::filesystem::exists(base + ".1"), "reader removed a failed shard");
    } else {
        state.check();
        require(next == count && reader.records_read() == count, "reader did not accumulate shard counts");
        require(!std::filesystem::exists(base + ".1"), "reader retained its final shard");
    }
    require(std::filesystem::exists(base + ".2"), "reader read beyond the supplied shard count");
}

void check_invalid_middle_shard(const std::filesystem::path& directory, bool missing)
{
    k_length = 41;
    RunState state;
    ActiveRunScope active(state);
    const auto base = (directory / (missing ? "missing" : "partial-middle")).string();
    std::ofstream(base + ".0", std::ios::binary);
    if (!missing) std::ofstream(base + ".1", std::ios::binary).put('x');
    std::ofstream(base + ".2", std::ios::binary).put('x');
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(5 + kmer_bytes(), 1);
    LowFrequencyReader reader(reads);
    reader.start(base, 3);
    reader.join();
    require(state.failed() && !reads.producer_finished(), "invalid middle shard was treated as EOF");
    require(!std::filesystem::exists(base + ".0"), "reader retained a completed empty shard");
    require(std::filesystem::exists(base + ".2"), "reader continued past a bad shard");
    if (!missing) require(std::filesystem::exists(base + ".1"), "reader deleted a partial record");
}

int fail_close(int fd)
{
    const int result = ::close(fd);
    if (result != 0) return result;
    errno = EIO;
    return -1;
}

// A deterministic AIO double keeps both output slots in flight. The snapshot
// is compared at aio_return, after more input has been collected and encoded.
struct SlowAio {
    struct Write { aiocb* cb; std::vector<uint8_t> bytes; };
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<Write> writes;
    size_t submissions = 0, returns = 0, peak = 0;
    bool release = false, overwritten = false;
    int failure = 0;
};
SlowAio* slow_aio;
int slow_submit(aiocb* cb)
{
    auto& io = *slow_aio;
    std::lock_guard lock(io.mutex);
    ++io.submissions;
    if (io.failure == 1 && io.submissions == 2) { errno = EIO; return -1; }
    const auto* data = static_cast<const uint8_t*>(const_cast<const void*>(cb->aio_buf));
    io.writes.push_back({cb, std::vector<uint8_t>(data, data + cb->aio_nbytes)});
    io.peak = std::max(io.peak, io.writes.size());
    io.changed.notify_all();
    return 0;
}
int slow_error(const aiocb*)
{
    std::lock_guard lock(slow_aio->mutex);
    if ((!slow_aio->release && !slow_aio->failure)
        || (slow_aio->failure == 1 && slow_aio->submissions < 2)) return EINPROGRESS;
    return slow_aio->failure == 3 ? EIO : 0;
}
ssize_t slow_result(aiocb* cb)
{
    auto& io = *slow_aio;
    std::lock_guard lock(io.mutex);
    auto found = std::find_if(io.writes.begin(), io.writes.end(),
        [cb](const auto& write) { return write.cb == cb; });
    if (found == io.writes.end()) { io.overwritten = true; return -1; }
    const void* data = const_cast<const void*>(cb->aio_buf);
    if (found->bytes.size() != cb->aio_nbytes || std::memcmp(data, found->bytes.data(), found->bytes.size()))
        io.overwritten = true;
    const ssize_t bytes = ::pwrite(cb->aio_fildes, data, cb->aio_nbytes, cb->aio_offset);
    io.writes.erase(found);
    ++io.returns;
    return io.failure == 2 && bytes > 0 ? bytes - 1 : bytes;
}
int slow_suspend(const aiocb* const[], int, const timespec*) { return 0; }

void check_final_aio(const std::filesystem::path& directory, int failure)
{
    k_length = 55;
    const size_t width = kmer_bytes();
    const size_t batch = msd_prefix::batch_records(width);
    const size_t count = 3 * batch + 17;
    RunState state;
    ActiveRunScope scope(state);
    SlowAio io;
    io.failure = failure;
    slow_aio = &io;
    ExportWriter::IoOps ops;
    ops.submit = slow_submit; ops.error = slow_error; ops.result = slow_result; ops.suspend = slow_suspend;
    if (failure == 4) ops.close = fail_close;
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(65536, 1);
    const auto path = directory / ("final-aio-" + std::to_string(failure));
    ExportWriter writer(&exports, path.string(), ExportWriter::OutputKind::FinalLow, nullptr, &ops);
    std::vector<std::vector<uint8_t>> expected;
    expected.reserve(count);
    for (size_t pos = 0; pos < count;) {
        char* block;
        exports.producer_dequeue(block);
        const size_t take = std::min(count - pos, exports.blockSize() / width);
        for (size_t i = 0; i < take; ++i) {
            std::vector<uint8_t> key(width, 0x51);
            uint64_t id = pos + i;
            std::memcpy(key.data() + 3, &id, sizeof(id));
            std::memcpy(block + i * width, key.data(), width);
            expected.push_back(std::move(key));
        }
        exports.producer_enqueue({block, take});
        pos += take;
    }
    exports.producer_set_finished();
    writer.start();
    if (!failure) {
        bool occupied;
        {
            std::unique_lock lock(io.mutex);
            occupied = io.changed.wait_for(lock, std::chrono::seconds(5), [&] { return io.submissions >= 2; });
        }
        // Let the writer reach its wait with a third full input batch. Slots
        // must remain untouched until we explicitly permit completion.
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        bool waited;
        {
            std::lock_guard lock(io.mutex);
            waited = occupied && io.submissions == 2 && io.returns == 0;
            io.release = true;
        }
        require(waited, "FinalLow did not wait for both busy slots");
    }
    bool rejected = false;
    try { writer.join(); } catch (const std::exception&) { rejected = true; }
    require(rejected == bool(failure) && state.failed() == bool(failure), "FinalLow failure propagation");
    require(io.writes.empty() && !io.overwritten && io.peak <= 2, "AIO buffer reused before collection or leaked");
    if (!failure) {
        require(io.submissions == 4 && io.returns == 4 && io.peak == 2, "full/tail batches or slot reuse differ");
        auto actual = low_msd_test::read(path, width, 37);
        std::sort(actual.begin(), actual.end());
        std::sort(expected.begin(), expected.end());
        require(actual == expected, "slow AIO changed full or tail batch multiset");
        require(std::filesystem::file_size(path) < count * width, "compressible output did not shrink");
        char* block;
        size_t returned = 0;
        while (exports.producer_try_dequeue(block)) ++returned;
        require(returned == exports.capacity(), "FinalLow did not return input blocks");
    }
}

void check_rotation_error(const std::filesystem::path& directory, bool close_error)
{
    k_length = 41;
    RunState state;
    ActiveRunScope active(state);
    const auto base = (directory / (close_error ? "close-error" : "open-error")).string();
    if (!close_error) std::filesystem::create_directory(base + ".1");
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    ExportWriter::IoOps ops;
    if (close_error) ops.close = fail_close;
    ExportWriter writer(&exports, base, ExportWriter::OutputKind::Pending, nullptr, &ops);
    const size_t capacity = exports.blockSize() / kmer_bytes();
    const size_t records = 3 * msd_prefix::batch_records(kmer_bytes()) + 2 * capacity;
    ExportWriterTestAccess::limit(writer, 1);
    for (size_t pos = 0; pos < records;) {
        char* block;
        exports.producer_dequeue(block);
        const size_t count = std::min(capacity, records - pos);
        std::memset(block, 0, count * kmer_bytes());
        exports.producer_enqueue({block, count});
        pos += count;
    }
    exports.producer_set_finished();
    writer.start();
    bool rejected = false;
    try { writer.join(); } catch (const std::exception&) { rejected = true; }
    require(rejected && state.failed() && writer.shard_count() == 1, "rotation failure was ignored");
}

void check_errors(const std::filesystem::path& directory)
{
    k_length = 81;
    const auto bad_path = directory / "partial.bin";
    std::ofstream(bad_path.string() + ".0", std::ios::binary).put('x');
    for (int mode = 0; mode < 2; ++mode) {
        const pid_t child = ::fork();
        require(child >= 0, "fork failed");
        if (child == 0) {
            RunState state;
            ActiveRunScope active(state);
            if (mode == 0) {
                SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(LOW_FREQUENCY_RING_MEMORY_POOL_BLOCK_SIZE, 1);
                LowFrequencyReader reader(reads);
                reader.start(bad_path.string(), 1);
                reader.join();
                state.check();
            } else {
                RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
                ExportWriter writer(&exports, "/dev/full", ExportWriter::OutputKind::FinalLow);
                writer.start();
                char* block;
                exports.producer_dequeue(block);
                std::memset(block, 0, kmer_bytes());
                exports.producer_enqueue({block, 1});
                exports.producer_set_finished();
                writer.join();
            }
            ::_exit(0);
        }
        int status;
        ::waitpid(child, &status, 0);
        require(WIFEXITED(status) && WEXITSTATUS(status) != 0, "I/O error was ignored");
    }
}
}

int main(int argc, char** argv)
{
    char name[] = "/tmp/tree-low-io-XXXXXX";
    const std::filesystem::path directory = ::mkdtemp(name);
    try {
        if (argc > 1 && std::string(argv[1]) == "--reader-failures-only") {
            check_reader_failures(directory);
            std::filesystem::remove_all(directory);
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--parser-failure-only") {
            check_parser_failure_cleanup();
            std::filesystem::remove_all(directory);
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--encoded-only") {
            check_reader_boundaries(directory, false);
            check_reader_boundaries(directory, true);
            check_encoded_reader(directory);
            std::filesystem::remove_all(directory);
            std::cout << "Encoded reader/parser tests passed\n";
            return 0;
        }
        if (argc > 1 && std::string(argv[1]) == "--pending-only") {
            check_io<2>(directory, 33, 61003);
            std::filesystem::remove_all(directory);
            return 0;
        }
        check_encoded_reader(directory);
        check_reader_boundaries(directory, false);
        check_reader_boundaries(directory, true);
        check_parser_failure_cleanup();
        check_reader_failures(directory);
        // Fork error cases before POSIX AIO starts helper threads in this process.
        if (!(argc > 1 && std::string(argv[1]) == "--no-death")) check_errors(directory);
        for (int failure = 0; failure <= 4; ++failure) check_final_aio(directory, failure);
        for (int failure = 0; failure < 3; ++failure) check_reader_lifecycle(directory, failure);
        check_invalid_middle_shard(directory, true);
        check_invalid_middle_shard(directory, false);
        check_queue_threshold(CLASSIFIER_TASK_QUEUE_HALF_WATERMARK - 1);
        check_queue_threshold(CLASSIFIER_TASK_QUEUE_HALF_WATERMARK);
        check_io<1>(directory, 31, 0);
        check_io<1>(directory, 32, 1); // A zero-kept block must be returned.
        check_io<2>(directory, 33, 61003);
        check_io<2>(directory, 41, 61003);
        check_rotation_error(directory, false);
        check_rotation_error(directory, true);
        if constexpr (LOW_PENDING_SHARD_BYTES <= (8ULL << 20)) {
            check_io<1>(directory, 32, (LOW_PENDING_SHARD_BYTES + 7) / 8);
            check_io<1>(directory, 32, (LOW_PENDING_SHARD_BYTES + 7) / 8 + 1);
        }
        // Exercise ring wraparound and an unaligned tail with two parsers.
        check_io<2>(directory, 33, (LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY + 3) * (4096 / 9) + 3, 4096);
        // More active owners than free blocks must publish partial blocks to make progress.
        check_io<2>(directory, 33, 61003, 4096, 1);
        check_io<2>(directory, 64, 33003);
        check_io<4>(directory, 65, 32003);
        check_io<4>(directory, 81, 66 * (512 * 1024 / 21) + 3);
        check_io<4>(directory, 128, 17003);
        std::filesystem::remove_all(directory);
        std::cout << "LowFrequencyIO tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
