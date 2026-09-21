#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <vector>

#include <unistd.h>

#include "../src/FastqReader.h"
#include "../src/RingMemoryPool.h"

namespace
{
    constexpr uint32_t kWords = 1;
    constexpr int kLength = 5;
    using ReaderPool = RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>;

    class TestDirectory
    {
    public:
        TestDirectory()
            : path_(std::filesystem::temp_directory_path()
                / ("tree_v4_reader_thread_pool_test_" + std::to_string(::getpid())))
        {
            std::filesystem::create_directories(path_);
        }

        ~TestDirectory()
        {
            std::error_code error;
            std::filesystem::remove_all(path_, error);
        }

        const std::filesystem::path& path() const noexcept
        {
            return path_;
        }

    private:
        std::filesystem::path path_;
    };

    bool write_fastq(const std::filesystem::path& path, const std::string& input)
    {
        std::ofstream output(path, std::ios::binary);
        output.write(input.data(), static_cast<std::streamsize>(input.size()));
        if (!output)
        {
            std::cerr << "Failed to write test FASTQ: " << path << '\n';
            return false;
        }
        return true;
    }

    std::vector<std::string> read_sequence_blocks(
        const std::vector<std::filesystem::path>& paths,
        uint64_t input_chunk_size,
        uint64_t output_block_size)
    {
        std::vector<std::string> filenames;
        filenames.reserve(paths.size());
        for (const std::filesystem::path& path : paths)
        {
            filenames.push_back(path.string());
        }

        ReaderPool pool(output_block_size, 1);
        ReaderThreadPool<kWords> reader(filenames, kLength, input_chunk_size, 1, &pool);

        reader.start();
        reader.join();

        std::vector<std::string> blocks;
        content_type content{};
        while (pool.consumer_try_dequeue(content))
        {
            blocks.emplace_back(content.data, content.length);
            pool.consumer_enqueue(content.data);
        }

        if (!pool.producer_finished())
        {
            std::cerr << "Reader pool did not report completion\n";
            return {};
        }
        return blocks;
    }

    std::vector<std::string> read_sequence_blocks(
        const std::filesystem::path& path,
        uint64_t input_chunk_size,
        uint64_t output_block_size)
    {
        return read_sequence_blocks(
            std::vector<std::filesystem::path>{path},
            input_chunk_size,
            output_block_size);
    }

    bool expect_blocks_equal(
        const std::vector<std::string>& actual,
        const std::vector<std::string>& expected,
        const char* label)
    {
        if (actual == expected)
        {
            return true;
        }

        std::cerr << label << " output mismatch\n"
            << "  actual block count: " << actual.size() << '\n'
            << "  expected block count: " << expected.size() << '\n';
        for (size_t i = 0; i < actual.size(); ++i)
        {
            std::cerr << "  actual[" << i << "] size: " << actual[i].size() << '\n';
        }
        for (size_t i = 0; i < expected.size(); ++i)
        {
            std::cerr << "  expected[" << i << "] size: " << expected[i].size() << '\n';
        }
        return false;
    }

    bool test_four_states_across_input_chunks(const TestDirectory& directory)
    {
        const std::filesystem::path path = directory.path() / "four_states.fastq";
        const std::string input =
            "@read-1 annotation\n"
            "ACGTACGTAC\n"
            "+read-1 annotation\n"
            "@+ACGT!?#$\n"
            "@read-2\n"
            "NNNNACGTA\n"
            "+\n"
            "#########\n";

        if (!write_fastq(path, input))
        {
            return false;
        }

        const std::vector<std::string> expected{ "ACGTACGTAC\nNNNNACGTA\n" };
        const std::vector<std::string> single_byte_chunks =
            read_sequence_blocks(path, 1, 2048);
        if (!expect_blocks_equal(
            single_byte_chunks,
            expected,
            "four-state single-byte-input-chunk test"))
        {
            return false;
        }

        const std::vector<std::string> single_chunk =
            read_sequence_blocks(path, 4096, 2048);
        return expect_blocks_equal(single_chunk, expected, "four-state single-input-chunk test");
    }

    bool test_crlf_output_is_preserved(const TestDirectory& directory)
    {
        const std::filesystem::path path = directory.path() / "crlf.fastq";
        const std::string input =
            "@read-1\r\n"
            "ACGTAC\r\n"
            "+\r\n"
            "IIIIII\r\n";

        if (!write_fastq(path, input))
        {
            return false;
        }

        const std::vector<std::string> actual = read_sequence_blocks(path, 5, 2048);
        return expect_blocks_equal(actual, { "ACGTAC\r\n" }, "CRLF preservation test");
    }

    bool test_sequence_across_output_blocks(const TestDirectory& directory)
    {
        constexpr uint64_t output_block_size = 2048;
        const std::filesystem::path path = directory.path() / "long_sequence.fastq";
        const std::string sequence(2500, 'A');
        const std::string quality(2500, 'I');
        const std::string input = "@long-read\n" + sequence + "\n+\n" + quality + "\n";

        if (!write_fastq(path, input))
        {
            return false;
        }

        const std::vector<std::string> actual =
            read_sequence_blocks(path, 113, output_block_size);
        const std::vector<std::string> expected{
            std::string(output_block_size, 'A'),
            std::string((kLength - 1) + (sequence.size() - output_block_size), 'A') + "\n",
        };
        return expect_blocks_equal(actual, expected, "cross-output-block test");
    }

    bool test_read_buffer_is_reused_across_files(const TestDirectory& directory)
    {
        const std::filesystem::path longer_path = directory.path() / "longer.fastq";
        const std::filesystem::path shorter_path = directory.path() / "shorter.fastq";
        const std::string longer_input =
            "@longer\n"
            "ACGTACGT\n"
            "+\n"
            "IIIIIIII\n";
        const std::string shorter_input =
            "@short\n"
            "TT\n"
            "+\n"
            "II\n";

        if (!write_fastq(longer_path, longer_input) || !write_fastq(shorter_path, shorter_input))
        {
            return false;
        }

        const std::vector<std::string> actual = read_sequence_blocks(
            std::vector<std::filesystem::path>{shorter_path, longer_path},
            7,
            2048);
        return expect_blocks_equal(
            actual,
            { "ACGTACGT\n", "TT\n" },
            "read-buffer reuse test");
    }
}

int main()
{
    const TestDirectory directory;

    if (!test_four_states_across_input_chunks(directory))
    {
        return 1;
    }
    if (!test_crlf_output_is_preserved(directory))
    {
        return 1;
    }
    if (!test_sequence_across_output_blocks(directory))
    {
        return 1;
    }
    if (!test_read_buffer_is_reused_across_files(directory))
    {
        return 1;
    }

    std::cout << "ReaderThreadPool tests passed\n";
    return 0;
}
