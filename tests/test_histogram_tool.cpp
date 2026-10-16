#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace
{
    using Records = std::vector<std::pair<uint64_t, uint32_t>>;
    std::filesystem::path test_dir;
    std::string tool;

    void require(bool ok, const std::string& message)
    {
        if (!ok) throw std::runtime_error(message);
    }

    std::string quote(const std::string& value)
    {
        std::string result = "'";
        for (char ch : value) result += ch == '\'' ? "'\\''" : std::string(1, ch);
        return result + "'";
    }

    // Independent encoder: base-4 digits occupy the most significant bits of
    // each word; complete words and the high bytes of a partial word go to disk.
    void write_key(std::ofstream& out, uint32_t k, uint64_t id)
    {
        for (uint32_t base = 0; base < k; base += 32)
        {
            uint64_t word = 0;
            uint32_t bases = std::min(32U, k - base);
            uint64_t digits = id ^ ((base / 32) * 0x9e3779b97f4a7c15ULL);
            for (uint32_t j = 0; j < bases; ++j)
            {
                word |= (digits & 3) << (62 - 2 * j);
                digits >>= 2;
            }
            uint32_t bytes = (bases + 3) / 4;
            const char* data = reinterpret_cast<const char*>(&word);
            out.write(data + (bases == 32 ? 0 : 8 - bytes), bytes);
        }
    }

    void fixture(uint32_t k, uint32_t cap, const Records& high, const std::vector<uint64_t>& low)
    {
        std::ofstream info(test_dir / "infos.bin", std::ios::binary);
        info.write(reinterpret_cast<const char*>(&k), 4);
        info.write(reinterpret_cast<const char*>(&cap), 4);
        uint32_t bytes = cap <= 255 ? 1 : cap <= 65535 ? 2 : cap <= 16777215 ? 3 : 4;
        std::ofstream high_out(test_dir / "high.bin", std::ios::binary);
        for (auto [key, count] : high)
        {
            write_key(high_out, k, key);
            high_out.write(reinterpret_cast<const char*>(&count), bytes);
        }
        std::ofstream low_out(test_dir / "low.bin", std::ios::binary);
        for (uint64_t key : low) write_key(low_out, k, key);
    }

    int run(const std::string& mode, uint32_t threads, const std::string& memory,
            uint32_t min, uint32_t max, const std::string& output = "hist.tsv")
    {
        std::string command = quote(tool) + " " + mode + " " + quote(test_dir.string()) + " " +
            std::to_string(threads) + " " + memory + " " + quote((test_dir / output).string()) +
            " " + std::to_string(min) + " " + std::to_string(max) + " > " +
            quote((test_dir / "stdout").string()) + " 2> " + quote((test_dir / "stderr").string());
        return std::system(command.c_str());
    }

    std::string read(const char* name)
    {
        std::ifstream input(test_dir / name);
        return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    }

    void check(const Records& high, const std::vector<uint64_t>& low, uint32_t cap,
               uint32_t min, uint32_t max, bool precise, uint32_t threads = 4,
               const std::string& memory = "1", bool segmented = false)
    {
        // Serial reference joins unique keys, then builds the histogram of final
        // counts. It does not repeat the production histogram correction rules.
        std::map<uint64_t, uint64_t> counts;
        for (auto [key, count] : high) counts.emplace(key, count);
        if (precise)
            for (uint64_t key : low)
            {
                auto found = counts.find(key);
                if (found == counts.end()) counts[key] = 1;
                else found->second = std::min<uint64_t>(cap, found->second + 1);
            }
        std::vector<int64_t> expected(static_cast<size_t>(max) - min + 1, 0);
        for (auto [key, count] : counts)
            if (count >= min && count <= max) ++expected[count - min];
        int status = run(precise ? "precise" : "approximate", threads, memory, min, max);
        require(status == 0, "tool failed: " + read("stderr"));
        std::ifstream input(test_dir / "hist.tsv");
        uint64_t freq;
        int64_t count;
        // The TSV has no header: verify its first bucket as well as every other.
        for (size_t i = 0; i < expected.size(); ++i)
        {
            require(static_cast<bool>(input >> freq >> count), "missing TSV row");
            require(freq == min + i && count == expected[i], "histogram mismatch at " + std::to_string(min + i));
        }
        require(!(input >> freq), "extra TSV rows");
        auto output = read("stdout");
        require(output.find("progress 0%") != std::string::npos &&
                output.find("progress 100%") != std::string::npos, "missing progress");
        if (segmented) require(output.find("Streaming mode:") != std::string::npos, "segmented path not exercised");
    }

    void test_boundaries()
    {
        for (uint32_t k : {1U, 4U, 31U, 32U, 33U, 63U, 64U, 65U, 95U, 96U, 97U, 127U, 128U})
            for (uint32_t cap : {1U, 255U, 65535U, 16777215U, 4294967295U})
            {
                // Two keys also fit k=1. A saturated high record must stay in
                // its original bucket when it is present in low.bin.
                Records high{{0, 1}, {1, cap}};
                std::vector<uint64_t> low{0, 1, 2};
                fixture(k, cap, high, low);
                check(high, low, cap, 1, 4, true, 1);
                check(high, low, cap, 1, 4, true, 4);
                check(high, low, cap, cap - 1, cap, true);
                check(high, low, cap, 1, 4, false);
                check(high, low, cap, cap - 1, cap, false);
            }
        Records high{{0, 1}, {1, 2}, {2, 5}};
        std::vector<uint64_t> low{0, 1, 2, 3};
        fixture(33, 255, high, low);
        check(high, low, 255, 3, 5, true);
        check(high, low, 255, 0, 5, true);
    }

    void test_empty_and_errors()
    {
        for (Records high : {Records{}, Records{{0, 2}}})
            for (std::vector<uint64_t> low : {std::vector<uint64_t>{}, std::vector<uint64_t>{0, 1}})
            {
                fixture(33, 255, high, low);
                check(high, low, 255, 1, 4, true);
                check(high, low, 255, 1, 4, false);
            }
        for (const std::string mode : {"precise", "approximate"})
        {
            fixture(33, 255, {{0, 2}}, {});
            require(run(mode, 1, "0.001", 1, 4) != 0, "insufficient budget accepted");
            require(run(mode, 1, "nan", 1, 4) != 0, "NaN budget accepted");
            require(run(mode, 1, "inf", 1, 4) != 0, "infinite budget accepted");
            require(run(mode, 1, "1", 1, 4, "missing/out.tsv") != 0, "bad output path accepted");
            // A writable file that fails on flush catches buffered output errors.
            if (std::filesystem::exists("/dev/full"))
            {
                std::filesystem::create_symlink("/dev/full", test_dir / "full");
                require(run(mode, 1, "1", 1, 4, "full") != 0, "output write failure ignored");
                std::filesystem::remove(test_dir / "full");
            }
            for (const char* name : {"infos.bin", "high.bin", "low.bin"})
            {
                fixture(33, 255, {{0, 2}}, {});
                std::ofstream corrupt(test_dir / name, std::ios::binary | std::ios::app);
                corrupt.put('x');
                corrupt.close();
                // approximate mode consumes high only, as before.
                if (mode == "approximate" && std::string(name) == "low.bin") continue;
                require(run(mode, 1, "1", 1, 4) != 0, "misaligned input accepted: " + std::string(name));
            }
            for (auto [k, cap] : Records{{0, 255}, {129, 255}, {33, 0}})
            {
                fixture(static_cast<uint32_t>(k), cap, {}, {});
                require(run(mode, 1, "1", 1, 4) != 0, "bad metadata accepted");
            }
            for (const char* name : {"infos.bin", "high.bin", "low.bin"})
            {
                fixture(33, 255, {{0, 2}}, {});
                std::filesystem::remove(test_dir / name);
                if (mode == "approximate" && std::string(name) == "low.bin") continue;
                require(run(mode, 1, "1", 1, 4) != 0, "missing input accepted");
            }
        }
    }

    void test_large_stream()
    {
        Records high;
        std::vector<uint64_t> low;
        for (uint64_t i = 0; i < 300000; ++i)
        {
            high.emplace_back(i, static_cast<uint32_t>(1 + i % 10));
            if (i % 2 == 0) low.push_back(i);
        }
        low.push_back(400000);
        fixture(128, 255, high, low);
        check(high, low, 255, 1, 11, false, 1);
        check(high, low, 255, 1, 11, false, 4);
        check(high, low, 255, 1, 11, true, 1);
        check(high, low, 255, 1, 11, true, 4);
        check(high, low, 255, 1, 11, true, 1, "0.035", true);
        check(high, low, 255, 1, 11, true, 4, "0.060", true);
        check(high, low, 255, 3, 9, true, 1, "0.035", true);
        fixture(128, 255, high, {});
        check(high, {}, 255, 1, 11, true, 1, "0.035", true);
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2) return 1;
    tool = std::filesystem::absolute(argv[1]).string();
    std::string pattern = (std::filesystem::temp_directory_path() / "histogram-test-XXXXXX").string();
    char* dir = ::mkdtemp(pattern.data());
    if (!dir) return 1;
    test_dir = dir;
    try
    {
        test_boundaries();
        test_empty_and_errors();
        test_large_stream();
        std::filesystem::remove_all(test_dir);
        std::cout << "Histogram serial-reference, boundary, error and segmented tests passed\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        std::filesystem::remove_all(test_dir);
        return 1;
    }
}
