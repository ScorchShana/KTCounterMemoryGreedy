#include "HistogramRunner.h"
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
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

    [[nodiscard]] Options parse_options(int argc, char* argv[])
    {
        if (argc < 3 || argc > 5)
        {
            std::cerr << "Usage: histogram_tool <tmp_dir> <output_file> [min_freq=1] [max_freq=10000]"
                << std::endl;
            exit(-1);
        }

        Options options;
        options.tmp_dir = with_trailing_slash(argv[1]);
        options.output_file = argv[2];
        if (argc >= 4)
        {
            options.min_freq = parse_u32(argv[3], "min_freq");
        }
        if (argc >= 5)
        {
            options.max_freq = parse_u32(argv[4], "max_freq");
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
