#include "../src/GzipStreamer.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <zlib.h>

namespace
{
    constexpr int kRepetitions = 3;
    constexpr uint64_t kDefaultSyntheticSizeMiB = 64;
    constexpr size_t kGzReadChunkSize = 256ULL * 1024;
    constexpr size_t kGzInternalBufferSize = 1ULL * 1024 * 1024;
    constexpr size_t kWriteChunkSize = 1ULL * 1024 * 1024;

    struct DecompressResult
    {
        double seconds = 0.0;
        uint64_t crc = 0;
        uint64_t bytes = 0;
    };

    uint64_t crc_of(const uint8_t* data, size_t size)
    {
        uLong crc = crc32(0L, Z_NULL, 0);
        while (size > 0)
        {
            const uInt step = static_cast<uInt>(std::min<uint64_t>(size, 1ULL << 30));
            crc = crc32(crc, data, step);
            data += step;
            size -= step;
        }
        return static_cast<uint64_t>(crc);
    }

    std::vector<uint8_t> generate_fastq_like_data(uint64_t target_bytes)
    {
        std::vector<uint8_t> data;
        data.reserve(static_cast<size_t>(target_bytes) + 512);

        std::mt19937_64 rng(20260907u);
        static const char bases[] = {'A', 'C', 'G', 'T'};
        uint64_t record_id = 0;

        while (data.size() < target_bytes)
        {
            const std::string id = "@read_" + std::to_string(record_id++) + "\n";
            const size_t seq_len = 100 + rng() % 51;

            data.insert(data.end(), id.begin(), id.end());
            for (size_t i = 0; i < seq_len; ++i)
            {
                data.push_back(static_cast<uint8_t>(bases[rng() % 4]));
            }
            data.push_back('\n');
            data.push_back('+');
            data.push_back('\n');
            for (size_t i = 0; i < seq_len; ++i)
            {
                data.push_back(static_cast<uint8_t>('!' + rng() % 94));
            }
            data.push_back('\n');
        }
        data.resize(static_cast<size_t>(target_bytes));
        return data;
    }

    bool write_gzip_file(const std::filesystem::path& gz_path, const std::vector<uint8_t>& data)
    {
        gzFile gz = gzopen(gz_path.string().c_str(), "wb");
        if (gz == nullptr)
        {
            std::cerr << "failed to create gzip test file: " << gz_path << '\n';
            return false;
        }
        size_t offset = 0;
        while (offset < data.size())
        {
            const size_t chunk = std::min(kWriteChunkSize, data.size() - offset);
            if (gzwrite(gz, data.data() + offset, static_cast<unsigned>(chunk)) !=
                static_cast<int>(chunk))
            {
                std::cerr << "gzwrite failed\n";
                gzclose(gz);
                return false;
            }
            offset += chunk;
        }
        gzclose(gz);
        return true;
    }

    std::filesystem::path prepare_gzip_file(
        const std::string& file_arg,
        bool use_file_arg,
        uint64_t size_mib,
        uint64_t& expected_crc)
    {
        if (use_file_arg)
        {
            const std::filesystem::path gz_path(file_arg);
            if (!std::filesystem::exists(gz_path))
            {
                std::cerr << "gzip file not found: " << gz_path << '\n';
                return {};
            }
            std::cout << "using gzip file: " << gz_path << " ("
                      << std::filesystem::file_size(gz_path) << " bytes)\n";
            return gz_path;
        }

        const std::filesystem::path dir = std::filesystem::current_path() / "tmp_gzip_benchmark";
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);

        std::cout << "generating " << size_mib << " MiB FASTQ-like test data...\n";
        std::vector<uint8_t> data = generate_fastq_like_data(size_mib * 1024 * 1024);
        expected_crc = crc_of(data.data(), data.size());

        const std::filesystem::path gz_path = dir / "benchmark_input.fastq.gz";
        if (!write_gzip_file(gz_path, data))
        {
            return {};
        }
        std::cout << "wrote " << gz_path << " ("
                  << std::filesystem::file_size(gz_path) << " bytes compressed)\n";
        return gz_path;
    }

    DecompressResult run_gzip_streamer(const std::filesystem::path& gz_path)
    {
        DecompressResult result;
        GzipStreamer streamer;
        streamer.open(gz_path.string());

        uint8_t* chunk_data = nullptr;
        size_t chunk_size = 0;
        const auto start = std::chrono::high_resolution_clock::now();
        while (streamer.next(chunk_data, chunk_size))
        {
            result.crc = static_cast<uint64_t>(crc32(
                static_cast<uLong>(result.crc), chunk_data, static_cast<uInt>(chunk_size)));
            result.bytes += chunk_size;
        }
        const auto end = std::chrono::high_resolution_clock::now();
        result.seconds = std::chrono::duration<double>(end - start).count();
        return result;
    }

    DecompressResult run_gzread(const std::filesystem::path& gz_path)
    {
        DecompressResult result;
        gzFile gz = gzopen(gz_path.string().c_str(), "rb");
        if (gz == nullptr)
        {
            std::cerr << "gzopen failed: " << gz_path << '\n';
            std::exit(1);
        }
        gzbuffer(gz, static_cast<unsigned>(kGzInternalBufferSize));

        std::vector<uint8_t> buffer(kGzReadChunkSize);
        const auto start = std::chrono::high_resolution_clock::now();
        while (true)
        {
            const int n = gzread(gz, buffer.data(), static_cast<unsigned>(buffer.size()));
            if (n < 0)
            {
                std::cerr << "gzread failed\n";
                gzclose(gz);
                std::exit(1);
            }
            if (n == 0)
            {
                break;
            }
            result.crc = static_cast<uint64_t>(crc32(
                static_cast<uLong>(result.crc), buffer.data(), static_cast<uInt>(n)));
            result.bytes += static_cast<uint64_t>(n);
        }
        const auto end = std::chrono::high_resolution_clock::now();
        result.seconds = std::chrono::duration<double>(end - start).count();
        gzclose(gz);
        return result;
    }

    double best_seconds(const std::vector<DecompressResult>& runs)
    {
        double best = runs.front().seconds;
        for (const auto& run : runs)
        {
            best = std::min(best, run.seconds);
        }
        return best;
    }

    void print_method_summary(const char* label, const std::vector<DecompressResult>& runs,
        uint64_t bytes)
    {
        const double best = best_seconds(runs);
        const double mib = static_cast<double>(bytes) / (1024.0 * 1024.0);
        std::cout << label << " best: " << std::fixed << std::setprecision(3) << best
                  << " s (" << std::setprecision(1) << (mib / best) << " MiB/s)\n";
    }
}

int main(int argc, char* argv[])
{
    if (argc > 2)
    {
        std::cerr << "Usage: " << argv[0] << " [gzip_file | size_mib]\n";
        return 1;
    }

    try
    {
        const std::string arg = argc == 2 ? argv[1] : "";
        const bool numeric_arg =
            !arg.empty() && arg.find_first_not_of("0123456789") == std::string::npos;
        const uint64_t size_mib = numeric_arg ? std::stoull(arg) : kDefaultSyntheticSizeMiB;
        const bool use_file_arg = argc == 2 && !numeric_arg;

        uint64_t expected_crc = 0;
        const std::filesystem::path gz_path =
            prepare_gzip_file(arg, use_file_arg, size_mib, expected_crc);
        if (gz_path.empty())
        {
            return 1;
        }

        std::vector<DecompressResult> streamer_runs;
        std::vector<DecompressResult> gzread_runs;
        for (int run = 0; run < kRepetitions; ++run)
        {
            if (run % 2 == 0)
            {
                streamer_runs.push_back(run_gzip_streamer(gz_path));
                gzread_runs.push_back(run_gzread(gz_path));
            }
            else
            {
                gzread_runs.push_back(run_gzread(gz_path));
                streamer_runs.push_back(run_gzip_streamer(gz_path));
            }

            const DecompressResult& streamer = streamer_runs.back();
            const DecompressResult& gzread = gzread_runs.back();
            std::cout << "run " << run + 1 << "/" << kRepetitions << ": GzipStreamer "
                      << std::fixed << std::setprecision(3) << streamer.seconds << " s, gzread "
                      << gzread.seconds << " s\n";

            if (streamer.crc != gzread.crc || streamer.bytes != gzread.bytes)
            {
                std::cerr << "decompressed data mismatch between GzipStreamer and gzread\n";
                return 1;
            }
            if (expected_crc != 0 && streamer.crc != expected_crc)
            {
                std::cerr << "decompressed data mismatch against original data\n";
                return 1;
            }
        }

        const uint64_t bytes = streamer_runs.front().bytes;
        std::cout << "decompressed " << bytes << " bytes\n";
        print_method_summary("GzipStreamer", streamer_runs, bytes);
        print_method_summary("gzread", gzread_runs, bytes);

        const double ratio = best_seconds(gzread_runs) / best_seconds(streamer_runs);
        std::cout << std::fixed << std::setprecision(2);
        if (ratio >= 1.0)
        {
            std::cout << "GzipStreamer is " << ratio << "x faster than gzread\n";
        }
        else
        {
            std::cout << "gzread is " << (1.0 / ratio) << "x faster than GzipStreamer\n";
        }
    }
    catch (const std::exception& ex)
    {
        std::cerr << ex.what() << '\n';
        return 1;
    }

    return 0;
}
