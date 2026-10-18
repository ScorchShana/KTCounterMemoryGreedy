#include "HistogramRunner.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
    constexpr uint64_t BYTES_PER_GIB = 1024ULL * 1024ULL * 1024ULL;
    [[nodiscard]] std::string with_trailing_slash(std::string path)
    {
        if (path.empty())
        {
            return "./";
        }
        const char tail = path.back();
        if (tail != '/' && tail != '\\')
        {
            path.push_back('/');
        }
        return path;
    }

    [[nodiscard]] uint32_t parse_u32(const char* text, const char* name)
    {
        if (!text[0] || text[0] == '-') throw std::runtime_error(std::string("invalid ") + name);
        size_t parsed = 0;
        const unsigned long long value = std::stoull(text, &parsed);
        if (parsed != std::strlen(text) || value > std::numeric_limits<uint32_t>::max()) [[unlikely]]
        {
            std::cerr << "invalid " << name << ": " << text << std::endl;
            exit(-1);
        }
        return static_cast<uint32_t>(value);
    }

    [[nodiscard]] uint64_t parse_memory_gib(const char* text)
    {
        size_t parsed = 0;
        const long double value = std::stold(text, &parsed);
        if (parsed != std::strlen(text) || !std::isfinite(value) || value <= 0.0L) [[unlikely]]
        {
            std::cerr << "invalid max_memory_gb: " << text << std::endl;
            exit(-1);
        }

        const long double bytes = value * static_cast<long double>(BYTES_PER_GIB);
        if (bytes > static_cast<long double>(std::numeric_limits<uint64_t>::max())) [[unlikely]]
        {
            std::cerr << "max_memory_gb is too large" << std::endl;
            exit(-1);
        }
        if (bytes < 1) throw std::runtime_error("max_memory_gb is too small");
        return static_cast<uint64_t>(bytes);
    }

    [[nodiscard]] Options parse_options(int argc, char* argv[])
    {
        if (argc < 6 || argc > 8)
        {
            std::cerr << "Usage: histogram_tool <precise/approximate> <tmp_dir> <max_threads> <max_memory_gb> <output_file> [min_freq=1] [max_freq=10000]"
                << std::endl;
            exit(-1);
        }

        Options options;
        options.tmp_dir = with_trailing_slash(argv[2]);
        options.max_threads = parse_u32(argv[3], "max_threads");
        options.max_memory_bytes = parse_memory_gib(argv[4]);
        options.output_file = argv[5];

        if (std::strcmp(argv[1], "precise") == 0)
         {
            options.is_precise = true;
        }
        else if (std::strcmp(argv[1], "approximate") == 0)
        {
            options.is_precise = false;
        }
        else
        {
            std::cerr << "invalid mode: must be 'precise' or 'approximate'" << std::endl;
            exit(-1);
        }
        if (argc >= 7)
        {
            options.min_freq = parse_u32(argv[6], "min_freq");
        }
        if (argc >= 8)
        {
            options.max_freq = parse_u32(argv[7], "max_freq");
        }

        if (options.max_threads == 0)
        {
            std::cerr << "invalid max_threads: must be greater than 0" << std::endl;
            exit(-1);
        }
        if (options.max_freq < options.min_freq)
        {
            std::cerr << "invalid max_freq: must be greater than or equal to min_freq" << std::endl;
            exit(-1);
        }
        return options;
    }

}

int main(int argc, char* argv[])
{
    try
    {
        return run_histogram_tool(parse_options(argc, argv));
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
