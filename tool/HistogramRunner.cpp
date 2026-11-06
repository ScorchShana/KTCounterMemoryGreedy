#include "HistogramRunner.h"
#include "HistogramParser.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

namespace
{
    class ProgressPrinter
    {
    public:
        explicit ProgressPrinter(long double total) : total_(total)
        {
            std::cout << "progress 0%" << std::flush;
        }
        void add(uint64_t bytes)
        {
            completed_ += bytes;
            unsigned percent = total_ ? static_cast<unsigned>(std::min(99.0L, completed_ * 100 / total_)) : 99;
            if (percent > last_)
            {
                std::cout << "\rprogress " << percent << '%' << std::flush;
                last_ = percent;
            }
        }
        void finish() { std::cout << "\rprogress 100%" << std::endl; }
    private:
        long double total_ = 0, completed_ = 0;
        unsigned last_ = 0;
    };

    uint64_t input_size(const std::string& path)
    {
        struct stat st{};
        if (::stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0)
            throw std::runtime_error("failed to stat regular input " + path);
        return static_cast<uint64_t>(st.st_size);
    }

    void read_infos(Options& options)
    {
        const std::string path = options.tmp_dir + "infos.bin";
        if (input_size(path) != 8) throw std::runtime_error("invalid infos.bin size");
        uint32_t fields[2]{};
        std::ifstream input(path, std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(fields), sizeof(fields)))
            throw std::runtime_error("failed to read " + path);
        options.k_len = fields[0];
        options.count_max = fields[1];
        if (!options.k_len || options.k_len > MAX_K || !options.count_max)
            throw std::runtime_error("invalid infos.bin k_len or count_max");
        options.count_bytes = options.count_max <= 255 ? 1 : options.count_max <= 65535 ? 2 :
            options.count_max <= 16777215 ? 3 : 4;
    }

    void write_histogram(const Options& options, const std::vector<int64_t>& histogram)
    {
        std::ofstream output(options.output_file, std::ios::trunc);
        if (!output) throw std::runtime_error("failed to open output file: " + options.output_file);
        for (size_t i = 0; i < histogram.size(); ++i)
            output << static_cast<uint64_t>(options.min_freq) + i << '\t' << histogram[i] << '\n';
        output.close();
        if (!output) throw std::runtime_error("failed to write output file: " + options.output_file);
    }
}

int run_histogram_tool(Options options)
{
    if (options.tmp_dir.empty()) options.tmp_dir = "./";
    if (options.tmp_dir.back() != '/') options.tmp_dir += '/';
    if (options.max_freq < options.min_freq)
        throw std::runtime_error("invalid histogram frequency range");
    read_infos(options);

    const std::string high_path = options.tmp_dir + "high.bin";
    const std::string low_path = options.tmp_dir + "low.bin";
    const bool include_singletons = options.min_freq <= 1 && options.max_freq >= 1;
    uint64_t total_bytes = input_size(high_path);
    if (include_singletons) total_bytes += input_size(low_path);

    const size_t buckets = static_cast<size_t>(
        static_cast<uint64_t>(options.max_freq) - options.min_freq + 1);
    std::vector<int64_t> histogram(buckets, 0);
    const size_t width = (options.k_len + 3) / 4;

    ProgressPrinter progress(static_cast<long double>(total_bytes));
    const auto on_bytes = [&progress](uint64_t bytes) { progress.add(bytes); };
    parse_group_file(high_path, width, options.count_bytes, options.min_freq,
                     options.max_freq, histogram, on_bytes);
    if (include_singletons)
        parse_group_file(low_path, width, 0, options.min_freq, options.max_freq,
                         histogram, on_bytes);

    write_histogram(options, histogram);
    progress.finish();
    return 0;
}
