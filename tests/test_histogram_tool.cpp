#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

namespace
{
    struct HighGroup
    {
        uint8_t prefix;
        std::vector<uint32_t> counts;
    };

    struct LowGroup
    {
        uint8_t prefix;
        uint32_t records;
    };

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

    std::filesystem::path input(const char* name)
    {
        return test_dir / name;
    }

    uint32_t count_width(uint32_t count_max)
    {
        return count_max <= 255 ? 1 : count_max <= 65535 ? 2 :
            count_max <= 16777215 ? 3 : 4;
    }

    size_t batch_limit(size_t width)
    {
        return (2ULL * 1024 * 1024 - 5) / width;
    }

    void write_infos(uint32_t k, uint32_t count_max)
    {
        std::ofstream out(input("infos.bin"), std::ios::binary | std::ios::trunc);
        require(static_cast<bool>(out), "failed to create infos.bin");
        out.write(reinterpret_cast<const char*>(&k), sizeof(k));
        out.write(reinterpret_cast<const char*>(&count_max), sizeof(count_max));
        out.close();
        require(static_cast<bool>(out), "failed to write infos.bin");
    }

    // Independent encoder: u8 P | u32 N | prefix[P] |
    // N * { suffix[W-P]; count[count_bytes] }, with counts omitted when null.
    void write_group(std::ofstream& out, uint8_t prefix, size_t width, uint32_t records,
                     uint32_t count_bytes, const uint32_t* counts, uint32_t seed)
    {
        require(prefix < width && records > 0 && records <= batch_limit(width),
                "invalid fixture group");
        std::vector<uint8_t> buffer;
        buffer.reserve(5 + prefix + size_t(records) * (width - prefix + count_bytes));
        buffer.push_back(prefix);
        for (uint32_t shift = 0; shift < 32; shift += 8)
            buffer.push_back(static_cast<uint8_t>(records >> shift));
        for (size_t i = 0; i < prefix; ++i)
            buffer.push_back(static_cast<uint8_t>(seed + i));
        for (uint32_t record = 0; record < records; ++record)
        {
            for (size_t i = 0; i < width - prefix; ++i)
                buffer.push_back(static_cast<uint8_t>(seed + record * 13 + i));
            if (counts)
                for (uint32_t byte = 0; byte < count_bytes; ++byte)
                    buffer.push_back(static_cast<uint8_t>(counts[record] >> (8 * byte)));
        }
        out.write(reinterpret_cast<const char*>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size()));
        require(static_cast<bool>(out), "failed to write fixture group");
    }

    void fixture(uint32_t k, uint32_t count_max, const std::vector<HighGroup>& high,
                 const std::vector<LowGroup>& low)
    {
        const size_t width = (k + 3) / 4;
        const uint32_t count_bytes = count_width(count_max);
        write_infos(k, count_max);
        {
            std::ofstream out(input("high.bin"), std::ios::binary | std::ios::trunc);
            require(static_cast<bool>(out), "failed to create high.bin");
            for (size_t i = 0; i < high.size(); ++i)
                write_group(out, high[i].prefix, width,
                            static_cast<uint32_t>(high[i].counts.size()), count_bytes,
                            high[i].counts.data(), static_cast<uint32_t>(i * 31 + 7));
            out.close();
            require(static_cast<bool>(out), "failed to write high.bin");
        }
        {
            std::ofstream out(input("low.bin"), std::ios::binary | std::ios::trunc);
            require(static_cast<bool>(out), "failed to create low.bin");
            for (size_t i = 0; i < low.size(); ++i)
                write_group(out, low[i].prefix, width, low[i].records, 0, nullptr,
                            static_cast<uint32_t>(i * 31 + 19));
            out.close();
            require(static_cast<bool>(out), "failed to write low.bin");
        }
    }

    int invoke(const std::vector<std::string>& args)
    {
        std::string command = quote(tool);
        for (const auto& arg : args) command += " " + quote(arg);
        command += " > " + quote((test_dir / "stdout").string()) +
            " 2> " + quote((test_dir / "stderr").string());
        return std::system(command.c_str());
    }

    int run(uint32_t min, uint32_t max, const std::string& output = "hist.tsv")
    {
        return invoke({test_dir.string(), (test_dir / output).string(),
                       std::to_string(min), std::to_string(max)});
    }

    std::string read(const char* name)
    {
        std::ifstream input_file(test_dir / name);
        return {std::istreambuf_iterator<char>(input_file), std::istreambuf_iterator<char>()};
    }

    void patch_byte(const char* name, std::streamoff offset, uint8_t value)
    {
        std::fstream file(input(name), std::ios::in | std::ios::out | std::ios::binary);
        require(static_cast<bool>(file), std::string("failed to open ") + name);
        file.seekp(offset);
        file.put(static_cast<char>(value));
        file.close();
        require(static_cast<bool>(file), std::string("failed to patch ") + name);
    }

    void patch_u32(const char* name, std::streamoff offset, uint32_t value)
    {
        std::fstream file(input(name), std::ios::in | std::ios::out | std::ios::binary);
        require(static_cast<bool>(file), std::string("failed to open ") + name);
        file.seekp(offset);
        file.write(reinterpret_cast<const char*>(&value), sizeof(value));
        file.close();
        require(static_cast<bool>(file), std::string("failed to patch ") + name);
    }

    void append(const char* name, const std::string& bytes)
    {
        std::ofstream file(input(name), std::ios::binary | std::ios::app);
        require(static_cast<bool>(file), std::string("failed to open ") + name);
        file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        file.close();
        require(static_cast<bool>(file), std::string("failed to append ") + name);
    }

    void verify(uint32_t min, uint32_t max, const std::vector<HighGroup>& high,
                const std::vector<LowGroup>& low)
    {
        std::vector<int64_t> expected(static_cast<size_t>(max) - min + 1, 0);
        for (const auto& group : high)
            for (uint32_t count : group.counts)
                if (count >= min && count <= max) ++expected[count - min];
        if (min <= 1 && max >= 1)
        {
            uint64_t low_records = 0;
            for (const auto& group : low) low_records += group.records;
            expected[1 - min] += static_cast<int64_t>(low_records);
        }

        const int status = run(min, max);
        require(status == 0, "tool failed: " + read("stderr"));
        std::ifstream tsv(input("hist.tsv"));
        uint64_t freq;
        int64_t count;
        for (size_t i = 0; i < expected.size(); ++i)
        {
            require(static_cast<bool>(tsv >> freq >> count), "missing TSV row");
            require(freq == static_cast<uint64_t>(min) + i, "wrong TSV frequency");
            require(count == expected[i], "histogram mismatch at " + std::to_string(freq));
        }
        require(!(tsv >> freq), "extra TSV rows");
        const std::string output = read("stdout");
        require(output.find("progress 0%") != std::string::npos &&
                output.find("progress 100%") != std::string::npos, "missing progress");
        require(output.find("Streaming mode:") == std::string::npos,
                "unexpected segmented histogram output");
    }

    void check(uint32_t k, uint32_t count_max, const std::vector<HighGroup>& high,
               const std::vector<LowGroup>& low, uint32_t min, uint32_t max)
    {
        fixture(k, count_max, high, low);
        verify(min, max, high, low);
    }

    void reset_fixture()
    {
        fixture(33, 255, {{2, {1, 2}}, {0, {5}}}, {{1, 3}});
    }

    void require_failure(uint32_t min, uint32_t max, const std::string& message)
    {
        require(run(min, max) != 0, message + ": " + read("stderr"));
    }

    void test_boundaries()
    {
        for (uint32_t k : {1U, 4U, 18U, 31U, 32U, 33U, 41U, 63U, 64U, 65U, 81U, 95U,
                           96U, 97U, 127U, 128U})
        {
            const size_t width = (k + 3) / 4;
            const auto second_prefix = static_cast<uint8_t>(std::min<size_t>(2, width - 1));
            for (uint32_t count_max : {1U, 255U, 65535U, 16777215U, 4294967295U})
            {
                const auto clamp = [count_max](uint32_t value)
                {
                    return std::min(value, count_max);
                };
                const std::vector<HighGroup> high{
                    {0, {1, 1, clamp(2), clamp(3), clamp(4), count_max}},
                    {second_prefix, {count_max, count_max, 1, clamp(2)}}};
                const std::vector<LowGroup> low{{0, 3}, {second_prefix, 2}};
                check(k, count_max, high, low, 1, 4);
                check(k, count_max, high, low, count_max - 1, count_max);
                check(k, count_max, high, low, 2, 4);
            }
        }
    }

    void test_empty_and_errors()
    {
        const std::vector<HighGroup> full_high{{2, {1, 2}}, {0, {5}}};
        const std::vector<LowGroup> full_low{{1, 3}};
        for (const auto& high : {std::vector<HighGroup>{}, full_high})
            for (const auto& low : {std::vector<LowGroup>{}, full_low})
            {
                check(33, 255, high, low, 1, 4);
                check(33, 255, high, low, 2, 4);
            }

        require(invoke({}) != 0, "missing arguments accepted");
        require(invoke({test_dir.string()}) != 0, "missing output accepted");
        reset_fixture();
        require(invoke({test_dir.string(), (test_dir / "hist.tsv").string(), "1", "4", "5"}) != 0,
                "extra argument accepted");
        require(run(4, 1) != 0, "reversed range accepted");
        require(invoke({test_dir.string(), (test_dir / "hist.tsv").string(), "-1", "4"}) != 0,
                "negative min accepted");
        require(invoke({test_dir.string(), (test_dir / "hist.tsv").string(), "abc", "4"}) != 0,
                "non-numeric min accepted");
        require(invoke({test_dir.string(), (test_dir / "hist.tsv").string(), "", "4"}) != 0,
                "empty min accepted");
        require(run(1, 4, "missing/out.tsv") != 0, "invalid output path accepted");
        if (std::filesystem::exists("/dev/full"))
        {
            std::filesystem::create_symlink("/dev/full", test_dir / "full");
            require(run(1, 4, "full") != 0, "output write failure ignored");
            std::filesystem::remove(test_dir / "full");
        }

        for (const char* name : {"infos.bin", "high.bin", "low.bin"})
        {
            reset_fixture();
            std::filesystem::remove(input(name));
            require_failure(1, 4, std::string("missing input accepted: ") + name);
        }
        // A missing or corrupt low.bin is irrelevant unless frequency 1 is requested.
        reset_fixture();
        std::filesystem::remove(input("low.bin"));
        require(run(2, 4) == 0, "unused low.bin was required");
        reset_fixture();
        std::filesystem::resize_file(input("low.bin"),
                                     std::filesystem::file_size(input("low.bin")) - 1);
        require(run(2, 4) == 0, "unused low.bin was parsed");
        require_failure(1, 4, "truncated low.bin accepted");

        reset_fixture();
        std::filesystem::resize_file(input("infos.bin"), 7);
        require_failure(1, 4, "short infos.bin accepted");
        reset_fixture();
        append("infos.bin", "x");
        require_failure(1, 4, "long infos.bin accepted");
        for (auto [k, count_max] : {std::pair<uint32_t, uint32_t>{0, 255}, {129, 255}, {33, 0}})
        {
            reset_fixture();
            write_infos(k, count_max);
            require_failure(1, 4, "invalid infos.bin metadata accepted");
        }

        reset_fixture();
        std::filesystem::resize_file(input("high.bin"), 3);
        require_failure(1, 4, "partial group header accepted");
        reset_fixture();
        std::filesystem::resize_file(input("high.bin"), 6);
        require_failure(1, 4, "partial group prefix accepted");
        reset_fixture();
        std::filesystem::resize_file(input("high.bin"),
                                     std::filesystem::file_size(input("high.bin")) - 1);
        require_failure(1, 4, "truncated high count accepted");
        reset_fixture();
        append("high.bin", std::string(1, '\0'));
        require_failure(1, 4, "trailing partial header accepted");
        reset_fixture();
        append("high.bin", std::string(5, '\0'));
        require_failure(1, 4, "trailing empty group accepted");
        reset_fixture();
        patch_byte("high.bin", 0, 9);
        require_failure(1, 4, "invalid prefix length accepted");
        reset_fixture();
        patch_u32("high.bin", 1, 0);
        require_failure(1, 4, "empty group accepted");
        reset_fixture();
        patch_u32("high.bin", 1, static_cast<uint32_t>(batch_limit(9) + 1));
        require_failure(1, 4, "oversized group accepted");
    }

    void test_large_stream()
    {
        constexpr uint32_t k = 33;
        constexpr uint32_t count_max = 255;
        constexpr uint32_t group_records = 200000;
        std::vector<HighGroup> high;
        std::vector<LowGroup> low;
        for (uint32_t group = 0; group < 12; ++group)
        {
            HighGroup spec{static_cast<uint8_t>(group % 4), {}};
            spec.counts.resize(group_records);
            for (uint32_t i = 0; i < group_records; ++i)
                spec.counts[i] = i % 997 == 0 ? count_max : 1 + (group * 7 + i) % 9;
            high.push_back(std::move(spec));
        }
        for (uint32_t group = 0; group < 8; ++group)
            low.push_back({static_cast<uint8_t>(group % 3), group_records});

        fixture(k, count_max, high, low);
        require(std::filesystem::file_size(input("high.bin")) > 8ULL * 1024 * 1024,
                "high stream does not span the AIO window depth");
        require(std::filesystem::file_size(input("low.bin")) > 8ULL * 1024 * 1024,
                "low stream does not span the AIO window depth");
        verify(1, 10, high, low);
        verify(3, 7, high, low);
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
        std::cout << "Histogram group-format, boundary, error and AIO-window tests passed\n";
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        std::filesystem::remove_all(test_dir);
        return 1;
    }
}
