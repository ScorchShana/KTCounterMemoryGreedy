#include "../src/FastqPreReader.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <stdexcept>
#include <string_view>

namespace
{
constexpr uint32_t kLength = 5;
constexpr uint64_t kChunkSize = 4096;
using ReaderPool = RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>;
const std::string sequences[] = {"ACGTACGTACGTACGTACGTACGTACGT", "TGCATGCATGCATGCATGCATGCATGCA"};

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

class TestDirectory
{
public:
    std::filesystem::path path;

    TestDirectory()
    {
        char name[] = "/tmp/tree-pre-reader-test-XXXXXX";
        require(::mkdtemp(name) != nullptr, "failed to create test directory");
        path = name;
    }

    ~TestDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void write_fastq(const std::filesystem::path& path, size_t records, bool compressed)
{
    std::string input;
    for (size_t i = 0; i < records; ++i)
        input += "@id\n" + sequences[i % 2] + "\n+\n"
            + std::string(sequences[i % 2].size(), static_cast<char>('I' + i % 2)) + "\n";
    // Each 64-byte record fits the input chunk exactly, including quality lines.
    require(input.size() == records * 64, "unexpected fixture record length");
    if (compressed) {
        gzFile file = gzopen(path.c_str(), "wb");
        require(file != nullptr, "failed to create gzip fixture");
        const int written = input.empty() ? 0 : gzwrite(file, input.data(), input.size());
        const int status = gzclose(file);
        require(written == static_cast<int>(input.size()) && status == Z_OK, "failed to write gzip fixture");
    } else {
        std::ofstream file(path, std::ios::binary);
        file.write(input.data(), input.size());
        require(file.good(), "failed to write plain fixture");
    }
}

void count_kmers(std::string_view data, std::map<std::string, uint64_t>& counts)
{
    size_t start = 0;
    while (start < data.size()) {
        const size_t newline = data.find('\n', start);
        const size_t end = newline == std::string_view::npos ? data.size() : newline;
        for (size_t pos = start; pos + kLength <= end; ++pos)
            ++counts[std::string(data.substr(pos, kLength))];
        start = end + 1;
    }
}

void check_inputs(const std::vector<std::pair<std::string, size_t>>& inputs)
{
    std::vector<std::string> filenames;
    std::map<std::string, uint64_t> expected, actual;
    uint64_t quality_count = 0, quality_sum = 0;
    for (const auto& [filename, records] : inputs) {
        filenames.push_back(filename);
        for (size_t i = 0; i < records; ++i) {
            count_kmers(sequences[i % 2], expected);
            quality_count += sequences[i % 2].size();
            quality_sum += sequences[i % 2].size() * ('I' + i % 2);
        }
    }

    RunState state;
    ActiveRunScope active(state);
    ReaderPool pool(2048, 1);
    FastqPreReader reader(filenames, kChunkSize, &pool);
    auto worker = std::async(std::launch::async, [&] {
        run_worker("pre-reader test", [&] { reader.pre_read(); });
    });
    try {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (worker.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
            require(std::chrono::steady_clock::now() < deadline, "pre-reader did not finish");
            content_type content;
            if (pool.consumer_try_dequeue(content)) {
                count_kmers({content.data, content.length}, actual);
                pool.consumer_enqueue(content.data);
            } else {
                std::this_thread::yield();
            }
        }
    } catch (...) {
        state.fail(std::current_exception(), "pre-reader test");
    }
    worker.get();
    state.check();
    content_type content;
    while (pool.consumer_try_dequeue(content)) {
        count_kmers({content.data, content.length}, actual);
        pool.consumer_enqueue(content.data);
    }
    require(pool.producer_finished(), "pre-reader did not publish completion");
    require(actual == expected, "pre-reader lost, duplicated, or changed sequence data");
    require(reader.get_quality_count() == quality_count, "pre-reader changed quality count");
    require(reader.get_quality_sum() == quality_sum, "pre-reader changed quality sum");
}

void check_failure_cleanup(const std::string& filename)
{
    RunState state;
    ActiveRunScope active(state);
    ReaderPool pool(2048, 1);
    // Leave only one output block available. After publishing it, parsing must
    // wait for another block while file_buffer still borrows a pipeline buffer.
    std::vector<char*> held;
    for (size_t i = 1; i < pool.capacity(); ++i) {
        char* block;
        pool.producer_dequeue(block);
        held.push_back(block);
    }
    FastqPreReader reader({filename}, kChunkSize, &pool);
    auto worker = std::async(std::launch::async, [&] {
        run_worker("pre-reader test", [&] { reader.pre_read(); });
    });
    bool borrowed_input = false;
    try {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        content_type first;
        while (!pool.consumer_try_dequeue(first)) {
            state.check();
            require(std::chrono::steady_clock::now() < deadline, "pre-reader did not publish the first block");
            std::this_thread::yield();
        }
        borrowed_input = true;
        // Keep this block too, preventing normal completion before cancellation.
        state.fail(std::make_exception_ptr(std::runtime_error("injected downstream failure")), "test consumer");
        require(worker.wait_until(deadline) == std::future_status::ready, "pre-reader did not stop on failure");
    } catch (...) {
        state.fail(std::current_exception(), "pre-reader test");
    }
    worker.get();
    require(borrowed_input, "failed to reach borrowed-buffer cleanup");
    bool propagated = false;
    try { state.check(); }
    catch (const std::runtime_error& error) {
        propagated = std::string_view(error.what()) == "injected downstream failure";
    }
    require(propagated, "pre-reader did not preserve the original failure");
    require(!pool.producer_finished(), "failed pre-reader announced successful completion");
    // Destruction joins I/O and releases every pipeline allocation exactly once.
}
}

int main(int argc, char* argv[])
{
    try {
        k_length = kLength;
        TestDirectory directory;
        const auto empty_plain = directory.path / "empty.fastq";
        const auto empty_gzip = directory.path / "empty.fastq.gz";
        const auto small_plain = directory.path / "small.fastq";
        const auto small_gzip = directory.path / "small.fastq.gz";
        const auto large_plain = directory.path / "large.fastq";
        const auto large_gzip = directory.path / "large.fastq.gz";
        write_fastq(empty_plain, 0, false);
        write_fastq(empty_gzip, 0, true);
        write_fastq(small_plain, 1, false);
        write_fastq(small_gzip, 1, true);
        write_fastq(large_plain, 2048, false);
        write_fastq(large_gzip, 2048, true);
        if (argc == 1 || std::string_view(argv[1]) == "--normal") {
            check_inputs({{empty_plain.string(), 0}});
            check_inputs({{empty_gzip.string(), 0}});
            check_inputs({{small_plain.string(), 1}});
            check_inputs({{small_gzip.string(), 1}});
            check_inputs({{large_plain.string(), 2048}});
            check_inputs({{large_gzip.string(), 2048}});
            std::vector<std::pair<std::string, size_t>> inputs;
            for (size_t i = 0; i < 32; ++i) {
                inputs.emplace_back(empty_plain.string(), 0);
                inputs.emplace_back(small_gzip.string(), 1);
                inputs.emplace_back(empty_gzip.string(), 0);
                inputs.emplace_back(small_plain.string(), 1);
            }
            inputs.emplace_back(large_gzip.string(), 2048);
            check_inputs(inputs);
        }
        if (argc == 1 || std::string_view(argv[1]) == "--failure") {
            check_failure_cleanup(large_plain.string());
            check_failure_cleanup(large_gzip.string());
        }
        std::cout << "FastqPreReader tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
