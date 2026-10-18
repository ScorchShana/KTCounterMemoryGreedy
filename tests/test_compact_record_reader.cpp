#include "../tool/CompactRecordReader.h"
#include "../tool/HighFrequencyThreadPool.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace
{
    void require(bool condition, const char* message)
    {
        if (!condition) throw std::runtime_error(message);
    }

    void write_file(const std::filesystem::path& file, const std::vector<char>& bytes)
    {
        std::ofstream output(file, std::ios::binary | std::ios::trunc);
        output.write(bytes.data(), bytes.size());
        require(static_cast<bool>(output), "fixture write failed");
    }

    void test_ranges(const std::filesystem::path& file)
    {
        for (uint32_t count_bytes : {0U, 1U, 2U, 3U, 4U})
        {
            uint64_t record_bytes = 32 + count_bytes;
            uint64_t records = 2 * CompactRecordReader::RAW_BUFFER_SIZE / record_bytes + 37;
            std::vector<char> original(records * record_bytes);
            for (size_t i = 0; i < original.size(); ++i) original[i] = static_cast<char>((i * 17 + i / 31) % 251);
            write_file(file, original);
            CompactRecordReader reader(128, count_bytes);
            reader.open(file.string(), 17, records - 34);
            require(reader.read_compact_records(nullptr, 0) == 0, "zero request");
            std::vector<char> result;
            std::vector<char> block(record_bytes * 22003);
            const std::array<uint64_t, 3> requests{1, 22003, 3};
            for (size_t i = 0; !reader.finished(); ++i)
            {
                uint64_t request = requests[i % requests.size()];
                uint64_t count = reader.read_compact_records(block.data(), request);
                result.insert(result.end(), block.begin(), block.begin() + count * record_bytes);
            }
            require(reader.finished(), "range not finished");
            require(reader.read_compact_records(block.data(), 1) == 0, "read past range");
            require(result == std::vector<char>(original.begin() + 17 * record_bytes,
                                               original.end() - 17 * record_bytes), "raw range bytes changed");
            reader.open(file.string(), records, 0);
            require(reader.finished(), "empty range not finished");
            bool rejected = false;
            try { reader.open(file.string(), records - 1, 2); }
            catch (const std::exception&) { rejected = true; }
            require(rejected, "out-of-bounds range accepted");
            reader.open(file.string(), 0, 1);
            require(reader.read_compact_records(block.data(), 2) == 1, "reopen failed");
            reader.close();
            reader.close();
        }
    }

    void test_short_read_cleanup(const std::filesystem::path& file)
    {
        // The first AIO may already be complete at truncation. The next buffer
        // is submitted only by read_compact_records, so its short read is certain.
        std::vector<char> bytes(2 * CompactRecordReader::RAW_BUFFER_SIZE + 32, 0);
        write_file(file, bytes);
        FlatConcurrentHashMap<1> map(1);
        AtomicHistogram histogram(4);
        for (auto& bucket : histogram) bucket.store(0);
        SPMCRingMemoryPool<4> pool(1024, 1);
        bool failed = false;
        {
            HighFrequencyThreadPool<1, 4> workers(&pool, &map, &histogram, 2, 32, 0, 1, 4, 4);
            workers.start();
            CompactRecordReader reader(32, 0);
            reader.open(file.string());
            std::filesystem::resize_file(file, 0);
            std::vector<char> block(bytes.size());
            try { reader.read_compact_records(block.data(), bytes.size() / 8); }
            catch (const std::exception&) { failed = true; }
            // Destruction must finish and join waiting consumers after failure.
        }
        require(failed, "short input read ignored");
        require(pool.producer_finished(), "failed producer not stopped");
    }

    void test_insert_failure()
    {
        FlatConcurrentHashMap<1> map(1);
        map.seal();
        AtomicHistogram histogram(4);
        for (auto& bucket : histogram) bucket.store(0);
        SPMCRingMemoryPool<4> pool(1024, 1);
        HighFrequencyThreadPool<1, 4> workers(&pool, &map, &histogram, 2, 32, 1, 1, 4, 4);
        workers.start();
        char* block = nullptr;
        pool.producer_dequeue(block);
        std::memset(block, 0, 9);
        block[8] = 2;
        pool.producer_enqueue(content_type{block, 1});
        pool.producer_set_finished();
        workers.join();
        require(workers.insert_failed(), "insert failure was not reported");
    }
}

int main()
{
    std::string pattern = (std::filesystem::temp_directory_path() / "compact-reader-test-XXXXXX").string();
    char* directory = ::mkdtemp(pattern.data());
    if (!directory) return 1;
    std::filesystem::path root(directory);
    try
    {
        test_ranges(root / "raw.bin");
        test_short_read_cleanup(root / "short.bin");
        test_insert_failure();
        std::filesystem::remove_all(root);
        std::cout << "Compact reader range, read failure, thread cleanup and insert failure tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(root);
        return 1;
    }
}
