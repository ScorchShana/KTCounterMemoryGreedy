// Standalone transport measurement; compile unchanged against the saved raw
// baseline with -DPENDING_BENCH_RAW. No production format switch is introduced.
#include "../src/ExportWriter.h"
#include "../src/LowFrequencyReader.h"
#include "../src/LowFrequencyParser.h"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <sys/resource.h>

using Clock = std::chrono::steady_clock;
static double seconds(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }
static double cpu_seconds()
{
    timespec ts{};
    if (::clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts)) throw std::runtime_error("process clock failed");
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char** argv)
{
    try {
        if (argc != 7 && argc != 8) throw std::invalid_argument("usage: benchmark_pending_msd DIR RAW_BYTES K PARSERS OWNERS uniform|hot|long-prefix [warm|cold]");
        const std::filesystem::path directory = argv[1];
        const uint64_t bytes = std::stoull(argv[2]);
        k_length = std::stoul(argv[3]);
        const size_t parsers = std::stoul(argv[4]), owners = std::stoul(argv[5]);
        const std::string pattern = argv[6];
        const bool cold = argc == 8 && std::string(argv[7]) == "cold";
        if (argc == 8 && std::string(argv[7]) != "cold" && std::string(argv[7]) != "warm")
            throw std::invalid_argument("invalid cache mode");
        if (!k_length || k_length > 128 || !parsers || !owners || owners > 256
            || (pattern != "uniform" && pattern != "hot" && pattern != "long-prefix"))
            throw std::invalid_argument("invalid benchmark dimensions");
        const size_t width = kmer_bytes();
        const uint64_t count = bytes / width;
        const size_t dictionary_size = width == 1 ? 256 : width == 2 ? 65536 : (1ULL << 20);
        std::vector<uint8_t> dictionary(dictionary_size * width);
        // Precompute all keys outside timing; trailing bytes give each a unique ID.
        uint64_t rng = 0x6a09e667f3bcc909ULL;
        for (size_t id = 0; id < dictionary_size; ++id) {
            auto* key = dictionary.data() + id * width;
            for (size_t b = 0; b < width; ++b) {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                key[b] = uint8_t(rng);
            }
            if (pattern == "hot") key[0] = id % 10 ? 0 : uint8_t(id);
            if (pattern == "long-prefix") std::fill(key, key + width, 0x51);
            for (size_t b = 0; b < std::min<size_t>(width, 3); ++b) key[width - 1 - b] = uint8_t(id >> (b * 8));
        }
        std::filesystem::create_directories(directory);
        const auto path = directory / "pending";
        RunState state; ActiveRunScope active(state);
        auto exports = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
        ExportWriter writer(exports.get(), path.string(), ExportWriter::OutputKind::Pending);
        const double write_cpu_start = cpu_seconds();
        const auto write_start = Clock::now();
        writer.start();
        const size_t block_capacity = exports->blockSize() / width;
        for (uint64_t pos = 0; pos < count;) {
            char* block; exports->producer_dequeue(block);
            const size_t take = std::min<uint64_t>(block_capacity, count - pos);
            size_t copied = 0;
            while (copied < take) {
                const size_t id = (pos + copied) % dictionary_size;
                const size_t n = std::min(take - copied, dictionary_size - id);
                std::memcpy(block + copied * width, dictionary.data() + id * width, n * width);
                copied += n;
            }
            exports->producer_enqueue({block, take});
            pos += take;
        }
        exports->producer_set_finished(); writer.join(); state.check();
        uint64_t encoded = 0;
        for (uint64_t i = 0; i < writer.shard_count(); ++i) {
            const auto shard = path.string() + "." + std::to_string(i);
            encoded += std::filesystem::file_size(shard);
            if (cold) {
                const int fd = ::open(shard.c_str(), O_RDWR);
                if (fd < 0) throw std::system_error(errno, std::generic_category(), "open benchmark shard");
                int result;
                do { result = ::fsync(fd); } while (result && errno == EINTR);
                const int error = result ? errno : ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                ::close(fd);
                if (error) throw std::system_error(error, std::generic_category(), "flush/evict private benchmark shard");
            }
        }
        const double write_seconds = seconds(write_start), write_cpu = cpu_seconds() - write_cpu_start;
#ifndef PENDING_BENCH_RAW
        if (writer.encoded_bytes_written() != encoded) throw std::runtime_error("writer byte count differs from shards");
#endif
        exports.reset();
        auto reads = std::make_unique<SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>>(LOW_FREQUENCY_RING_MEMORY_POOL_BLOCK_SIZE, 1);
        auto parsed = std::make_unique<RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>>(PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, parsers);
        std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> queues;
        for (size_t i = 0; i < owners; ++i) queues.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
        for (size_t i = 0; i < prefix_owners.size(); ++i) prefix_owners[i] = i % owners;
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
        std::vector<std::vector<uint64_t>> frequencies(owners, std::vector<uint64_t>(dictionary_size));
        std::vector<std::thread> parser_threads, consumer_threads;
        LowFrequencyReader reader(*reads);
        const double read_cpu_start = cpu_seconds();
        const auto read_start = Clock::now();
        try {
            for (size_t t = 0; t < owners; ++t) consumer_threads.emplace_back([&, t] {
                run_worker("benchmark owner", [&] {
                    SpinBackoff<> backoff;
                    for (;;) {
                        content_type content;
                        bool local = queues[t]->try_dequeue(content);
                        bool found = local || global.try_dequeue(content);
                        if (!found && parsed->producer_finished()) {
                            local = queues[t]->try_dequeue(content);
                            found = local || global.try_dequeue(content);
                            if (!found) break;
                        }
                        if (!found) { backoff.backoff(); continue; }
                        backoff.decay();
                        const auto* keys = reinterpret_cast<const uint8_t*>(content.data);
                        const size_t owner = prefix_owners[keys[0]];
                        if (local && owner != t) throw std::runtime_error("wrong owner queue");
                        for (size_t j = 0; j < content.length; ++j) {
                            const auto* key = keys + j * width;
                            size_t id = 0;
                            for (size_t b = 0; b < std::min<size_t>(width, 3); ++b) id |= size_t(key[width - 1 - b]) << (b * 8);
                            if (id >= dictionary_size || prefix_owners[key[0]] != owner || std::memcmp(key, dictionary.data() + id * width, width))
                                throw std::runtime_error("transport changed a key");
                            ++frequencies[t][id];
                        }
                        parsed->consumer_enqueue(content.data);
                    }
                });
            });
            for (size_t t = 0; t < parsers; ++t) parser_threads.emplace_back([&] {
                run_worker("benchmark parser", [&] {
                    LowFrequencyParser parser(queues, &global);
                    parser.parse_and_push(*reads, *parsed);
                    parsed->producer_set_finished();
                });
            });
            reader.start(path.string(), writer.shard_count());
            reader.join();
        } catch (...) { state.fail(std::current_exception(), "benchmark replay"); }
        reader.join();
        for (auto& thread : parser_threads) thread.join();
        for (auto& thread : consumer_threads) thread.join();
        state.check();
        const double read_seconds = seconds(read_start), read_cpu = cpu_seconds() - read_cpu_start;
        if (reader.records_read() != count) throw std::runtime_error("reader record count differs");
        for (size_t id = 0; id < dictionary_size; ++id) {
            uint64_t actual = 0;
            for (const auto& frequency : frequencies) actual += frequency[id];
            if (actual != count / dictionary_size + (id < count % dictionary_size))
                throw std::runtime_error("exact per-key frequency differs");
        }
        uint64_t read_bytes = encoded;
        int64_t calls = -1;
#ifndef PENDING_BENCH_RAW
        read_bytes = reader.encoded_bytes_read(); calls = reader.read_calls();
        if (read_bytes != encoded) throw std::runtime_error("encoded byte conservation failed");
#endif
        rusage usage{}; ::getrusage(RUSAGE_SELF, &usage);
        std::cout << "{\"raw_bytes\":" << count * width << ",\"records\":" << count
            << ",\"encoded_bytes\":" << encoded << ",\"read_bytes\":" << read_bytes
            << ",\"read_calls\":" << calls << ",\"write_seconds\":" << write_seconds
            << ",\"read_seconds\":" << read_seconds << ",\"transport_seconds\":" << write_seconds + read_seconds
            << ",\"write_cpu_seconds\":" << write_cpu << ",\"read_cpu_seconds\":" << read_cpu
            << ",\"peak_rss_kib\":" << usage.ru_maxrss << ",\"verified\":true}\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
