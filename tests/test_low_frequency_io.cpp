#include "../src/LowFrequencyReader.h"
#include "../src/LowFrequencyParser.h"
#include "../src/ExportWriter.h"
#include "../src/ParserThreadPool.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <sys/wait.h>
#include <vector>

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
    size_t offset = 0;
    for (uint64_t i = 0; i < pending.shard_count(); ++i) {
        std::ifstream shard(input_path.string() + "." + std::to_string(i), std::ios::binary);
        require(shard.good(), "pending shard is missing");
        std::vector<uint8_t> actual((std::istreambuf_iterator<char>(shard)), {});
        require(actual.size() % width == 0, "pending shard split a packed record");
        require(actual.size() < LOW_PENDING_SHARD_BYTES + exports.blockSize(), "pending shard exceeded block allowance");
        if (i + 1 < pending.shard_count())
            require(actual.size() >= LOW_PENDING_SHARD_BYTES, "pending shard rotated prematurely");
        require(actual.size() <= input.size() - offset, "pending grew extra bytes");
        require(std::equal(actual.begin(), actual.end(), input.begin() + offset), "pending bytes changed across shards");
        offset += actual.size();
    }
    require(offset == input.size(), "pending lost bytes");
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
    std::ifstream out(output_path, std::ios::binary);
    std::vector<uint8_t> output((std::istreambuf_iterator<char>(out)), {});
    require(output.size() == expected.size() * width, "partial output record");
    std::vector<std::vector<uint8_t>> actual;
    for (size_t pos = 0; pos < output.size(); pos += width)
        actual.emplace_back(output.begin() + pos, output.begin() + pos + width);
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
    std::memcpy(block, key.data(), key.size());
    reads.producer_enqueue({block, 1});
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
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(kmer_bytes(), 1);
    const size_t count = reads.capacity() + 4;
    for (int shard = 0; shard < 3; ++shard) {
        std::ofstream out(base + "." + std::to_string(shard), std::ios::binary);
        const size_t begin = shard == 0 ? 0 : 1;
        const size_t end = shard == 0 ? 1 : count;
        for (uint64_t i = begin; i < end; ++i) {
            std::array<char, 11> record{};
            std::memcpy(record.data(), &i, sizeof(i));
            out.write(record.data(), record.size());
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
        std::memcpy(&id, content.data, sizeof(id));
        require(content.length == 1 && id == next++, "reader changed cross-shard order");
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
        std::memcpy(&id, content.data, sizeof(id));
        require(content.length == 1 && id == next++, "reader changed cross-shard bytes");
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
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(kmer_bytes(), 1);
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
    const size_t records = LOW_PENDING_SHARD_BYTES / kmer_bytes() + 2 * capacity;
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

int main()
{
    char name[] = "/tmp/tree-low-io-XXXXXX";
    const std::filesystem::path directory = ::mkdtemp(name);
    try {
        // Fork error cases before POSIX AIO starts helper threads in this process.
        check_errors(directory);
        for (int failure = 0; failure < 3; ++failure) check_reader_lifecycle(directory, failure);
        check_invalid_middle_shard(directory, true);
        check_invalid_middle_shard(directory, false);
        check_queue_threshold(CLASSIFIER_TASK_QUEUE_HALF_WATERMARK - 1);
        check_queue_threshold(CLASSIFIER_TASK_QUEUE_HALF_WATERMARK);
        check_io<1>(directory, 31, 0);
        check_io<1>(directory, 32, 1); // A zero-kept block must be returned.
        check_io<2>(directory, 33, 61003);
        check_io<2>(directory, 41, 61003);
        if constexpr (LOW_PENDING_SHARD_BYTES <= (8ULL << 20)) {
            check_rotation_error(directory, false);
            check_rotation_error(directory, true);
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
