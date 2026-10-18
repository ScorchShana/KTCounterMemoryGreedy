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
    PackedKmerCodec<N> codec(k);
    const size_t width = codec.packed_bytes();
    const auto input_path = directory / "pending.bin";
    const auto output_path = directory / "low.bin";
    std::vector<uint8_t> input(count * width);
    std::vector<std::vector<uint8_t>> expected;
    for (size_t i = 0; i < count; ++i) {
        kmer<N> key{};
        key.data[0] = (uint64_t(i % 256) << 56) | (uint64_t(i) << 16);
        codec.pack(key, input.data() + i * width);
        if (i % 3 != 0)
            expected.emplace_back(input.begin() + i * width, input.begin() + (i + 1) * width);
    }
    {
        std::ofstream out(input_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(input.data()), input.size());
    }
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(block_size, 1);
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> parsed(PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, 2);
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
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
    LowFrequencyReader reader(k, reads);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners;
    for (int i = 0; i < 2; ++i)
        owners.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
    for (size_t i = 0; i < prefix_owners.size(); ++i) prefix_owners[i] = i % owners.size();
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    // Low-pass thread count is independent of the first-pass parser count.
    ParserThreadPool<N> parsers(k, owners, &global, nullptr, &parsed, 1);
    ExportWriter<N> writer(k, &exports, output_path.string());
    writer.start();
    parsers.start_low_pass(reads, 2);
    reader.start(input_path.string());
    std::vector<std::thread> consumers;
    for (int thread = 0; thread < 2; ++thread) {
        consumers.emplace_back([&, thread] {
            char* output;
            exports.producer_dequeue(output);
            size_t kept = 0;
            constexpr size_t capacity = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
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
                const auto* keys = reinterpret_cast<const kmer<N>*>(content.data);
                require(content.length > 0, "parser published an empty block");
                const auto owner = prefix_owners[keys[0].data[0] >> 56];
                require(!local || owner == thread, "block reached the wrong owner queue");
                for (size_t j = 0; j < content.length; ++j) {
                    require(prefix_owners[keys[j].data[0] >> 56] == owner, "block mixes classifier owners");
                    const size_t id = (keys[j].data[0] >> 16) & 0xffffffffULL;
                    kmer<N> expected_key{};
                    expected_key.data[0] = (uint64_t(id % 256) << 56) | (uint64_t(id) << 16);
                    require(keys[j] == expected_key, "decoder left non-payload bytes in key");
                    if (id % 3 != 0) {
                        reinterpret_cast<ExportBlock<N>*>(output)->k_mers[kept++] = keys[j];
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
    prefix_owners.fill(0);
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(64, 1);
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> parsed(64, 1);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners{
        std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>()};
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    for (size_t i = 0; i < queued; ++i) {
        char* block;
        parsed.producer_dequeue(block);
        reinterpret_cast<kmer<2>*>(block)->reset();
        owners[0]->enqueue({block, 1});
    }
    kmer<2> key{};
    key.data[0] = 1ULL << 32;
    char* block;
    reads.producer_dequeue(block);
    PackedKmerCodec<2>(k).pack(key, reinterpret_cast<uint8_t*>(block));
    reads.producer_enqueue({block, 1});
    reads.producer_set_finished();
    LowFrequencyParser<2> parser(k, owners, &global);
    parser.parse_and_push(reads, parsed);
    parsed.producer_set_finished();
    const bool spill = queued >= CLASSIFIER_TASK_QUEUE_HALF_WATERMARK;
    require(owners[0]->size() == queued + !spill, "wrong owner queue selection at watermark");
    require(global.size() == size_t(spill), "wrong global queue selection at watermark");
    size_t found = 0;
    content_type content;
    auto collect = [&] {
        require(content.length == 1, "wrong tail block length");
        found += *reinterpret_cast<const kmer<2>*>(content.data) == key;
        parsed.consumer_enqueue(content.data);
    };
    while (owners[0]->try_dequeue(content)) collect();
    while (global.try_dequeue(content)) collect();
    require(found == 1, "tail was lost or duplicated during queue selection");
    size_t returned = 0;
    while (parsed.producer_try_dequeue(block)) ++returned;
    require(returned == parsed.capacity(), "queue routing leaked output blocks");
}

void check_errors(const std::filesystem::path& directory)
{
    const auto bad_path = directory / "partial.bin";
    std::ofstream(bad_path, std::ios::binary).put('x');
    for (int mode = 0; mode < 2; ++mode) {
        const pid_t child = ::fork();
        require(child >= 0, "fork failed");
        if (child == 0) {
            if (mode == 0) {
                SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(LOW_FREQUENCY_RING_MEMORY_POOL_BLOCK_SIZE, 1);
                LowFrequencyReader reader(81, reads);
                reader.start(bad_path.string());
                reader.join();
            } else {
                RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
                ExportWriter<4> writer(81, &exports, "/dev/full");
                writer.start();
                char* block;
                exports.producer_dequeue(block);
                std::memset(block, 0, sizeof(kmer<4>));
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
        check_queue_threshold(CLASSIFIER_TASK_QUEUE_HALF_WATERMARK - 1);
        check_queue_threshold(CLASSIFIER_TASK_QUEUE_HALF_WATERMARK);
        check_io<1>(directory, 31, 0);
        check_io<1>(directory, 32, 1); // A zero-kept block must be returned.
        check_io<2>(directory, 33, 61003);
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
