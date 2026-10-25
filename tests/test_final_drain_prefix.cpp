#include "../src/FinalDrainWriter.h"
#include "../src/FinalDrainWriterThread.h"
#include "../src/MsdPrefixCodec.h"
#include "high_prefix_test_helpers.h"

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>

namespace {
using Pool = RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY>;
using Key = std::vector<unsigned char>;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

std::array<unsigned char, 32> make_key(size_t width, size_t prefix, uint32_t id, uint32_t k)
{
    std::array<unsigned char, 32> key{};
    for (size_t i = 0; i < width; ++i) key[i] = (id >> (8 * (i % 4))) + i * 37;
    for (size_t i = 0; i < prefix; ++i) key[i] = i ? 0x34 : 0x12;
    if (k % 4) key[width - 1] &= static_cast<unsigned char>(0xff << (8 - 2 * (k % 4)));
    return key;
}

void boundary_case(Pool& pool, uint32_t k, uint32_t c, uint8_t prefix, uint32_t records)
{
    k_length = k; count_max_bytes = c;
    const size_t width = kmer_bytes();
    const uint32_t mask = static_cast<uint32_t>((uint64_t{1} << (8 * c)) - 1);
    const uint64_t before = sorted_kmer_count.load();
    {
        FinalDrainWriter writer(&pool);
        writer.begin_node(prefix);
        for (uint32_t i = 0; i < records; ++i) {
            const auto key = make_key(width, prefix, i, k);
            writer.write_kmer_record(key.data(), (i * 997u + 7) & mask);
        }
        writer.close();
        writer.close();
    }
    require(sorted_kmer_count.load() - before == records, "close changed output statistics");
    uint32_t seen = 0;
    content_type content;
    while (pool.consumer_try_dequeue(content)) {
        require(content.length <= FINAL_DRAIN_RING_POOL_BLOCK_SIZE, "oversized producer block");
        const auto bytes = std::span(reinterpret_cast<const uint8_t*>(content.data), content.length);
        if (!bytes.empty()) {
            require(bytes[0] == prefix, "wrong node prefix length");
            uint32_t count; std::memcpy(&count, bytes.data() + 1, 4);
            require(bytes.size() == 5 + prefix + size_t(count) * (width - prefix + c),
                "unexpected padding or incomplete group");
        }
        for (const auto& record : high_prefix_test::decode(bytes, width, c)) {
            const auto expected = make_key(width, prefix, seen, k);
            require(std::memcmp(record.key.data(), expected.data(), width) == 0, "key changed at block boundary");
            require(record.count == ((seen * 997u + 7) & mask), "count changed at block boundary");
            ++seen;
        }
        pool.consumer_enqueue(content.data);
    }
    require(seen == records, "records lost at group boundary");
}

void node_changes(Pool& pool)
{
    k_length = 12; count_max_bytes = 1;
    std::array<unsigned char, 6> keys{0x12, 0x34, 0xa5, 0x12, 0x34, 0xb6};
    {
        FinalDrainWriter writer(&pool);
        writer.begin_node(1); // A completely filtered/empty node emits no header.
        writer.begin_node(2);
        writer.write_kmer_record(keys.data(), 7);
        writer.write_kmer_record(keys.data() + 3, 9);
        writer.close();
    }
    content_type content;
    require(pool.consumer_try_dequeue(content) && content.length == 11, "unexpected group size");
    std::vector<uint8_t> input(msd_prefix::BUFFER_BYTES), scratch(input.size()), low(input.size());
    std::memcpy(input.data(), keys.data(), keys.size());
    const size_t low_bytes = msd_prefix::encode(input.data(), scratch.data(), 2, 3, low.data());
    require(low_bytes == 9 && low[0] == 2, "low fixture failed to form the intended prefix");
    require(std::memcmp(content.data, low.data(), 7) == 0, "high header/prefix differs from low.bin");
    require(static_cast<uint8_t>(content.data[5]) == 0x12 && static_cast<uint8_t>(content.data[6]) == 0x34,
        "prefix uses the integer memory order");
    auto decoded = high_prefix_test::decode({reinterpret_cast<const uint8_t*>(content.data), content.length}, 3, 1);
    require(decoded.size() == 2 && decoded[0].count == 7 && decoded[1].count == 9, "header fixture counts differ");
    pool.consumer_enqueue(content.data);
    {
        FinalDrainWriter writer(&pool);
        for (uint8_t p : {1, 2, 1}) {
            writer.begin_node(p);
            writer.write_kmer_record(keys.data(), p);
        }
        writer.begin_node(2); // Empty final node.
        writer.close();
    }
    require(pool.consumer_try_dequeue(content) && content.length == 27, "node change flushed a partial block");
    decoded = high_prefix_test::decode({reinterpret_cast<const uint8_t*>(content.data), content.length}, 3, 1);
    require(decoded.size() == 3 && decoded[0].count == 1 && decoded[1].count == 2 && decoded[2].count == 1,
        "mixed prefix groups changed records");
    pool.consumer_enqueue(content.data);
    require(!pool.consumer_try_dequeue(content), "small nodes submitted separate blocks");
}

std::vector<uint8_t> read_file(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "missing final file");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

void opaque_aggregation(const std::filesystem::path& dir)
{
    temp_dir = dir.string() + "/";
    std::vector<uint8_t> expected(4 * 1024 * 1024 + 37);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = (i * 17) ^ (i >> 13);
    FinalDrainWriterThread sink(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
    sink.start();
    for (size_t offset = 0; offset < expected.size();) {
        char* block; sink.pool()->producer_dequeue(block);
        const size_t bytes = std::min<size_t>(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, expected.size() - offset);
        std::memcpy(block, expected.data() + offset, bytes);
        sink.pool()->producer_enqueue({block, bytes});
        offset += bytes;
    }
    sink.pool()->producer_set_finished(); sink.join();
    RunState::check_active();
    require(read_file(dir / "high.final.tmp") == expected, "Writer changed opaque byte stream");
}

void multi_producer(const std::filesystem::path& dir)
{
    temp_dir = dir.string() + "/";
    k_length = 36; count_max_bytes = 2;
    constexpr uint32_t producers = 4, records = 100000;
    FinalDrainWriterThread sink(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, producers);
    sink.start();
    std::vector<std::thread> threads;
    for (uint32_t t = 0; t < producers; ++t) threads.emplace_back([&, t] {
        run_worker("prefix test producer", [&] {
            FinalDrainWriter writer(sink.pool());
            writer.begin_node(1);
            const uint32_t local_records = records - t * 1237;
            std::array<unsigned char, 9> key{static_cast<unsigned char>(t), 0};
            for (uint32_t i = 0; i < local_records; ++i) {
                if (i == local_records / 3) {
                    writer.begin_node(2);
                    key[1] = 0xff;
                }
                for (uint32_t b = 0; b < 4; ++b) key[2 + b] = i >> (8 * b);
                writer.write_kmer_record(key.data(), i % 65535 + 1);
            }
            writer.close(); sink.pool()->producer_set_finished();
        });
    });
    for (auto& thread : threads) thread.join();
    sink.join(); RunState::check_active();
    const auto bytes = read_file(dir / "high.final.tmp");
    std::vector<bool> seen(producers * records);
    size_t total = 0;
    for (const auto& record : high_prefix_test::decode(bytes, 9, 2)) {
        const uint32_t t = record.key[0];
        uint32_t id = 0;
        for (uint32_t b = 0; b < 4; ++b) id |= uint32_t(record.key[2 + b]) << (8 * b);
        require(t < producers, "interleaved producer identity");
        const uint32_t local_records = records - t * 1237;
        require(id < local_records, "interleaved record identity");
        require(record.key[1] == (id < local_records / 3 ? 0 : 0xff) &&
            record.prefix_bytes == (id < local_records / 3 ? 1 : 2), "interleaved node prefix");
        require(!seen[t * records + id] && record.count == id % 65535 + 1, "duplicate/lost concurrent count");
        require(record.key[6] == 0 && record.key[7] == 0 && record.key[8] == 0, "concurrent suffix corrupted");
        seen[t * records + id] = true; ++total;
    }
    require(total == producers * records - 1237 * producers * (producers - 1) / 2, "lost concurrent records");
}
} // namespace

int main()
{
    RunState state; ActiveRunScope scope(state);
    const auto dir = std::filesystem::temp_directory_path() / ("tree-final-prefix-" + std::to_string(::getpid()));
    try {
        {
            Pool pool(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
            for (uint32_t k = 1; k <= 128; ++k) {
                const size_t width = (k + 3) / 4;
                for (uint32_t c = 1; c <= 4; ++c)
                    for (uint8_t p = 0; p <= std::min<size_t>(2, width - 1); ++p)
                        for (uint32_t n : {0, 1, 2, 5, 6, 7}) boundary_case(pool, k, c, p, n);
            }
            for (uint32_t k = 1; k <= 128; ++k) {
                const uint32_t width = (k + 3) / 4;
                for (uint32_t c = 1; c <= 4; ++c)
                    for (uint8_t p = 0; p <= std::min<uint32_t>(2, width - 1); ++p) {
                        const uint32_t m = (FINAL_DRAIN_RING_POOL_BLOCK_SIZE - 5 - p) / (width - p + c);
                        for (uint32_t n : {m - 1, m, m + 1, 2 * m + 1}) boundary_case(pool, k, c, p, n);
                    }
            }
            node_changes(pool);
        }
        std::filesystem::create_directory(dir);
        opaque_aggregation(dir);
        multi_producer(dir);
        std::filesystem::remove_all(dir);
        std::cout << "Final drain prefix tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(dir);
        return 1;
    }
}
