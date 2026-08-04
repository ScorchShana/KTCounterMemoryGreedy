#include "../src/ExportWriter.h"
#include "../src/RingMemoryPool.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{
    template <uint32_t N>
    using Kmer = kmer<N>;

    template <uint32_t N>
    using Record = ExportRecord<N>;

    std::string shell_quote(const std::string& value)
    {
        std::string quoted = "'";
        for (const char ch : value)
        {
            if (ch == '\'')
            {
                quoted += "'\\''";
            }
            else
            {
                quoted.push_back(ch);
            }
        }
        quoted.push_back('\'');
        return quoted;
    }

    std::filesystem::path make_test_dir(const std::string& name)
    {
        const std::filesystem::path dir = std::filesystem::current_path() / name;
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        return dir;
    }

    template <uint32_t N>
    Kmer<N> make_kmer(const std::vector<uint64_t>& words)
    {
        Kmer<N> value{};
        value.reset();
        for (uint32_t i = 0; i < N && i < words.size(); ++i)
        {
            value.data[i] = words[i];
        }
        return value;
    }

    template <uint32_t N>
    void write_root_file(
        const std::filesystem::path& dir,
        const uint32_t root_id,
        const std::vector<Record<N>>& records)
    {
        const std::filesystem::path filename = dir / ("root_" + std::to_string(root_id) + ".bin");
        std::ofstream output(filename, std::ios::binary | std::ios::trunc);
        if (!output)
        {
            throw std::runtime_error("failed to open root file for writing");
        }
        if (!records.empty())
        {
            output.write(
                reinterpret_cast<const char*>(records.data()),
                static_cast<std::streamsize>(records.size() * sizeof(Record<N>)));
        }
    }

    template <uint32_t N>
    void write_low_bin(
        const std::filesystem::path& dir,
        const uint32_t k_len,
        const std::vector<Kmer<N>>& records)
    {
        temp_dir = dir.generic_string() + "/";

        RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> pool(
            EXPORT_RING_MEMORY_POOL_BLOCK_SIZE,
            1);
        ExportWriter<N> writer(k_len, &pool);

        size_t offset = 0;
        while (offset < records.size())
        {
            char* block_ptr = nullptr;
            pool.producer_dequeue(block_ptr);
            auto* export_block = reinterpret_cast<ExportBlock<N>*>(block_ptr);
            const size_t count = std::min(export_block->k_mers.size(), records.size() - offset);
            for (size_t i = 0; i < count; ++i)
            {
                export_block->k_mers[i] = records[offset + i];
            }
            pool.producer_enqueue(content_type{block_ptr, count});
            offset += count;
        }

        pool.producer_set_finished();
        writer.start();
        writer.join();
    }

    std::vector<int64_t> read_histogram(const std::filesystem::path& filename)
    {
        std::ifstream input(filename);
        if (!input)
        {
            throw std::runtime_error("failed to open histogram output");
        }

        std::string line;
        std::getline(input, line);

        std::vector<int64_t> values;
        while (std::getline(input, line))
        {
            std::istringstream iss(line);
            uint64_t freq = 0;
            int64_t count = 0;
            if (!(iss >> freq >> count))
            {
                throw std::runtime_error("invalid histogram line: " + line);
            }
            values.push_back(count);
        }
        return values;
    }

    int run_tool(
        const std::string& tool_path,
        const std::filesystem::path& dir,
        const uint32_t k_len,
        const uint32_t min_freq,
        const uint32_t max_freq,
        const std::string& output_name,
        const std::string& memory_gb = "1")
    {
        const std::filesystem::path output_path = dir / output_name;
        const std::filesystem::path stdout_path = dir / "tool_stdout.txt";
        const std::string command =
            shell_quote(tool_path) + " " +
            shell_quote(dir.generic_string()) + " " +
            std::to_string(k_len) + " 4 " +
            memory_gb + " " +
            shell_quote(output_path.generic_string()) + " " +
            std::to_string(min_freq) + " " +
            std::to_string(max_freq) +
            " > " + shell_quote(stdout_path.generic_string());
        return std::system(command.c_str());
    }

    bool expect_progress_output(const std::filesystem::path& dir, const char* label)
    {
        std::ifstream input(dir / "tool_stdout.txt");
        if (!input)
        {
            std::cerr << label << " missing stdout capture\n";
            return false;
        }

        const std::string content(
            (std::istreambuf_iterator<char>(input)),
            std::istreambuf_iterator<char>());
        if (content.find("progress 0%") == std::string::npos ||
            content.find("progress 100%") == std::string::npos)
        {
            std::cerr << label << " progress output missing 0% or 100%\n";
            return false;
        }
        return true;
    }

    bool expect_histogram(
        const std::filesystem::path& filename,
        const std::vector<int64_t>& expected,
        const char* label)
    {
        const std::vector<int64_t> actual = read_histogram(filename);
        if (actual == expected)
        {
            return true;
        }

        std::cerr << label << " histogram mismatch\nactual:";
        for (const int64_t value : actual)
        {
            std::cerr << ' ' << value;
        }
        std::cerr << "\nexpected:";
        for (const int64_t value : expected)
        {
            std::cerr << ' ' << value;
        }
        std::cerr << '\n';
        return false;
    }

    bool test_k33_two_phase(const std::string& tool_path)
    {
        const std::filesystem::path dir = make_test_dir("tmp_histogram_k33");

        const Kmer<2> a = make_kmer<2>({0x0123456789abcdefULL, 0xc000000000000000ULL});
        const Kmer<2> b = make_kmer<2>({0x1111111111111111ULL, 0x8000000000000000ULL});
        const Kmer<2> c = make_kmer<2>({0x2222222222222222ULL, 0x4000000000000000ULL});
        const Kmer<2> e = make_kmer<2>({0x3333333333333333ULL, 0x0000000000000000ULL});

        write_root_file<2>(dir, 0, {
            Record<2>{a, 2},
            Record<2>{c, 1},
            Record<2>{e, 4},
        });
        write_low_bin<2>(dir, 33, {a, b, c, e});

        if (run_tool(tool_path, dir, 33, 1, 4, "hist.tsv") != 0)
        {
            std::cerr << "k33 tool run failed\n";
            return false;
        }

        return expect_progress_output(dir, "k33") &&
            expect_histogram(dir / "hist.tsv", {1, 1, 1, 0}, "k33");
    }

    bool test_min_filter(const std::string& tool_path)
    {
        const std::filesystem::path dir = make_test_dir("tmp_histogram_min_filter");

        const Kmer<2> g = make_kmer<2>({0x4444444444444444ULL, 0xc000000000000000ULL});
        const Kmer<2> h = make_kmer<2>({0x5555555555555555ULL, 0x8000000000000000ULL});
        const Kmer<2> i = make_kmer<2>({0x6666666666666666ULL, 0x4000000000000000ULL});
        const Kmer<2> j = make_kmer<2>({0x7777777777777777ULL, 0x0000000000000000ULL});

        write_root_file<2>(dir, 1, {
            Record<2>{g, 1},
            Record<2>{h, 2},
            Record<2>{i, 5},
        });
        write_low_bin<2>(dir, 33, {g, h, i, j});

        if (run_tool(tool_path, dir, 33, 3, 5, "hist.tsv") != 0)
        {
            std::cerr << "min-filter tool run failed\n";
            return false;
        }

        return expect_histogram(dir / "hist.tsv", {1, 0, 0}, "min-filter");
    }

    bool test_n1_and_n4_dispatch(const std::string& tool_path)
    {
        {
            const std::filesystem::path dir = make_test_dir("tmp_histogram_n1");
            const Kmer<1> key = make_kmer<1>({0xaaaaaaaa55555555ULL});
            write_root_file<1>(dir, 2, {Record<1>{key, 1}});
            write_low_bin<1>(dir, 32, {key});

            if (run_tool(tool_path, dir, 32, 1, 3, "hist.tsv") != 0)
            {
                std::cerr << "n1 tool run failed\n";
                return false;
            }
            if (!expect_histogram(dir / "hist.tsv", {0, 1, 0}, "n1"))
            {
                return false;
            }
        }

        {
            const std::filesystem::path dir = make_test_dir("tmp_histogram_n4");
            const Kmer<4> key = make_kmer<4>({
                0xaaaaaaaa55555555ULL,
                0x0123456789abcdefULL,
                0xc000000000000000ULL,
                0x0000000000000000ULL});
            write_root_file<4>(dir, 3, {Record<4>{key, 1}});
            write_low_bin<4>(dir, 65, {key});

            if (run_tool(tool_path, dir, 65, 1, 3, "hist.tsv") != 0)
            {
                std::cerr << "n4 tool run failed\n";
                return false;
            }
            if (!expect_histogram(dir / "hist.tsv", {0, 1, 0}, "n4"))
            {
                return false;
            }
        }

        return true;
    }

    bool test_invalid_root_size(const std::string& tool_path)
    {
        const std::filesystem::path dir = make_test_dir("tmp_histogram_invalid_root");
        write_low_bin<2>(dir, 33, {});

        std::ofstream invalid(dir / "root_0.bin", std::ios::binary | std::ios::trunc);
        invalid.put('\0');
        invalid.close();

        if (run_tool(tool_path, dir, 33, 1, 3, "hist.tsv") == 0)
        {
            std::cerr << "invalid root size should fail\n";
            return false;
        }
        return true;
    }

    bool test_memory_stub(const std::string& tool_path)
    {
        const std::filesystem::path dir = make_test_dir("tmp_histogram_memory_stub");
        write_low_bin<1>(dir, 32, {});

        if (run_tool(tool_path, dir, 32, 1, 3, "hist.tsv", "0.001") == 0)
        {
            std::cerr << "memory stub should fail\n";
            return false;
        }
        return true;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <histogram_tool_path>\n";
        return 1;
    }

    const std::string tool_path = argv[1];
    try
    {
        if (!test_k33_two_phase(tool_path))
        {
            return 1;
        }
        if (!test_min_filter(tool_path))
        {
            return 1;
        }
        if (!test_n1_and_n4_dispatch(tool_path))
        {
            return 1;
        }
        if (!test_invalid_root_size(tool_path))
        {
            return 1;
        }
        if (!test_memory_stub(tool_path))
        {
            return 1;
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    std::cout << "Histogram tool tests passed\n";
    return 0;
}
