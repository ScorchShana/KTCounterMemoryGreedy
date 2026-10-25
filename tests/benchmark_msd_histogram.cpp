// Standalone production-codec benchmark. Build with the same Release flags
// for each source snapshot; pin the process with taskset in the paired runner.
#include "../src/MsdPrefixCodec.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using Bytes = std::vector<uint8_t>;
static constexpr std::array<const char*, 5> DISTRIBUTIONS{
    "uniform", "two_hot", "hot90", "long_prefix", "multilevel"};

static Bytes fixture(size_t width, size_t count, const std::string& distribution)
{
    Bytes raw(count * width);
    std::mt19937 random(1729);
    for (auto& byte : raw) byte = uint8_t(random());
    for (size_t i = 0; i < count; ++i) {
        auto* key = raw.data() + i * width;
        if (distribution == "two_hot") key[0] = i % 2 ? 0xff : 0;
        else if (distribution == "hot90") key[0] = i % 10 ? 0 : uint8_t(1 + random() % 255);
        else if (distribution == "long_prefix" || distribution == "multilevel") {
            for (size_t p = 0; p < width - 3; ++p)
                key[p] = distribution == "long_prefix" ? 0x53 : uint8_t((i >> (p % 8)) & 1);
            // Odd multiplication permutes 24-bit suffixes, guaranteeing unique
            // keys even for Q while keeping the suffix order unsorted.
            const uint32_t suffix = uint32_t(i * 0x9e3779U) & 0xffffffU;
            key[width - 3] = uint8_t(suffix >> 16);
            key[width - 2] = uint8_t(suffix >> 8);
            key[width - 1] = uint8_t(suffix);
        }
    }
    return raw;
}

int main(int argc, char** argv)
{
    try {
        size_t selected_width = 0, selected_count = 0;
        bool full_batch = false;
        double min_ms = 100;
        std::string selected_distribution;
        std::filesystem::path dump_dir;
        for (int i = 1; i < argc; ++i) {
            const std::string option(argv[i]);
            if (i + 1 == argc) throw std::runtime_error("missing option value");
            const std::string value(argv[++i]);
            if (option == "--width") selected_width = std::stoul(value);
            else if (option == "--records") {
                full_batch = value == "Q";
                if (!full_batch) selected_count = std::stoul(value);
            } else if (option == "--distribution") selected_distribution = value;
            else if (option == "--min-ms") min_ms = std::stod(value);
            else if (option == "--dump-dir") dump_dir = value;
            else throw std::runtime_error("unknown option: " + option);
        }
        const std::array<size_t, 4> widths{7, 10, 14, 32};
        if ((selected_width && std::find(widths.begin(), widths.end(), selected_width) == widths.end())
            || !(min_ms > 0) || !std::isfinite(min_ms)
            || (!selected_distribution.empty() && std::find(DISTRIBUTIONS.begin(), DISTRIBUTIONS.end(),
                selected_distribution) == DISTRIBUTIONS.end()))
            throw std::runtime_error("invalid width, distribution, or minimum duration");
        if (!dump_dir.empty()) std::filesystem::create_directories(dump_dir);
        // These buffers are reused for all cases; input restoration is outside
        // every timed interval. The codec has no allocation in its timed path.
        Bytes input(msd_prefix::BUFFER_BYTES), scratch(msd_prefix::BUFFER_BYTES), output(msd_prefix::BUFFER_BYTES);
        std::cout << "width,records,distribution,iterations,encode_seconds,ns_per_encode,mib_s,encoded_bytes,checksum\n"
                  << std::setprecision(12);
        for (size_t width : widths) {
            if (selected_width && width != selected_width) continue;
            const size_t q = msd_prefix::batch_records(width);
            const std::vector<size_t> counts = full_batch ? std::vector<size_t>{q}
                : selected_count ? std::vector<size_t>{selected_count}
                : std::vector<size_t>{16, 64, 256, 1024, 4096, q};
            for (size_t count : counts) {
                if (!count || count > q) throw std::runtime_error("record count exceeds batch capacity");
                for (const char* distribution : DISTRIBUTIONS) {
                    if (!selected_distribution.empty() && distribution != selected_distribution) continue;
                    const Bytes raw = fixture(width, count, distribution);
                    size_t encoded = 0;
                    for (unsigned warmup = 0; warmup < 8; ++warmup) {
                        std::memcpy(input.data(), raw.data(), raw.size());
                        encoded = msd_prefix::encode(input.data(), scratch.data(), uint32_t(count), width, output.data());
                    }
                    double seconds = 0;
                    uint64_t iterations = 0;
                    do {
                        std::memcpy(input.data(), raw.data(), raw.size());
                        const auto start = std::chrono::steady_clock::now();
                        encoded = msd_prefix::encode(input.data(), scratch.data(), uint32_t(count), width, output.data());
                        const auto stop = std::chrono::steady_clock::now();
                        seconds += std::chrono::duration<double>(stop - start).count();
                        ++iterations;
                    } while (seconds * 1000 < min_ms);
                    if (encoded > raw.size() + msd_prefix::GROUP_HEADER_BYTES || encoded > output.size())
                        throw std::runtime_error("encoded capacity bound");
                    uint64_t checksum = 14695981039346656037ULL;
                    for (size_t i = 0; i < encoded; ++i) checksum = (checksum ^ output[i]) * 1099511628211ULL;
                    if (!dump_dir.empty()) {
                        const auto stem = dump_dir / (std::to_string(width) + "-" + std::to_string(count) + "-" + distribution);
                        for (const auto& [suffix, data, bytes] : {
                            std::tuple{".raw", raw.data(), raw.size()},
                            std::tuple{".msd", static_cast<const uint8_t*>(output.data()), encoded}}) {
                            std::ofstream stream(stem.string() + suffix, std::ios::binary);
                            stream.write(reinterpret_cast<const char*>(data), bytes);
                            if (!stream) throw std::runtime_error("cannot write fixture dump");
                        }
                    }
                    const double per_encode = seconds / iterations;
                    std::cout << width << ',' << count << ',' << distribution << ',' << iterations << ','
                              << seconds << ',' << per_encode * 1e9 << ','
                              << raw.size() / (1024.0 * 1024.0) / per_encode << ',' << encoded << ',' << checksum << '\n';
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
