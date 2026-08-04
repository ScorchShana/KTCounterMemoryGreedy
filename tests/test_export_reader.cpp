#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "../src/ExportWriter.h"
#include "../src/RingMemoryPool.h"
#include "../tool/ExportReader.h"

namespace
{
    constexpr uint32_t kWords = 2;
    using Kmer = kmer<kWords>;

    Kmer make_kmer(uint64_t first, uint64_t second)
    {
        Kmer value{};
        value.reset();
        value.data[0] = first;
        value.data[1] = second;
        return value;
    }

    Kmer expected_after_writer_pack(const Kmer& input, uint32_t k)
    {
        Kmer expected{};
        expected.reset();

        const uint64_t full_data_count = k / BASES_PER_U64T;
        const uint64_t tail_bits = 2 * (k % BASES_PER_U64T);

        for (uint64_t i = 0; i < full_data_count; ++i)
        {
            expected.data[i] = input.data[i];
        }

        if (tail_bits > 0)
        {
            const uint64_t mask = (~uint64_t{0}) << (64 - tail_bits);
            expected.data[full_data_count] = input.data[full_data_count] & mask;
        }

        return expected;
    }

    uint64_t packed_kmer_bytes_for_k(uint32_t k)
    {
        const uint64_t full_data_count = k / BASES_PER_U64T;
        const uint64_t tail_bits = 2 * (k % BASES_PER_U64T);
        const uint64_t tail_bytes = (tail_bits + 7) / 8;
        return full_data_count * sizeof(uint64_t) + tail_bytes;
    }

    void set_test_temp_dir(const std::string& name)
    {
        const std::filesystem::path dir = std::filesystem::current_path() / name;
        std::filesystem::create_directories(dir);
        temp_dir = dir.generic_string() + "/";
    }

    void write_low_bin(uint32_t k, const std::vector<Kmer>& records, const std::string& name)
    {
        set_test_temp_dir(name);

        RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> pool(
            EXPORT_RING_MEMORY_POOL_BLOCK_SIZE,
            1);
        ExportWriter<kWords> writer(k, &pool);

        size_t offset = 0;
        while (offset < records.size())
        {
            char* block_ptr = nullptr;
            pool.producer_dequeue(block_ptr);

            auto* export_block = reinterpret_cast<ExportBlock<kWords>*>(block_ptr);
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

    std::vector<char> read_low_bin_bytes()
    {
        std::ifstream input(temp_dir + "low.bin", std::ios::binary);
        return std::vector<char>(
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>());
    }

    bool expect_equal(const Kmer& actual, const Kmer& expected, const char* label)
    {
        if (actual == expected)
        {
            return true;
        }

        std::cerr << label << " mismatch\n"
                  << "  actual:   " << std::hex << actual.data[0] << " "
                  << actual.data[1] << "\n"
                  << "  expected: " << expected.data[0] << " "
                  << expected.data[1] << std::dec << "\n";
        return false;
    }

    bool expect_packed_reader_matches_file(
        uint32_t k,
        const std::vector<size_t>& request_sizes,
        const char* label)
    {
        const std::vector<char> expected = read_low_bin_bytes();
        const uint64_t packed_kmer_bytes = packed_kmer_bytes_for_k(k);
        if (expected.size() % packed_kmer_bytes != 0)
        {
            std::cerr << label << " expected byte size is not k-mer aligned\n";
            return false;
        }

        ExportReader<kWords> reader(k);
        reader.open(temp_dir + "low.bin");

        std::vector<char> actual(expected.size());
        uint64_t total_read = 0;
        size_t request_index = 0;
        while (!reader.finished())
        {
            const size_t request_size = request_sizes[request_index++ % request_sizes.size()];
            const uint64_t got = reader.read_packed_kmers(
                actual.data() + total_read * packed_kmer_bytes,
                request_size);
            if (got == 0)
            {
                std::cerr << label << " packed read returned 0 before finished\n";
                return false;
            }
            total_read += got;
        }

        if (total_read * packed_kmer_bytes != expected.size())
        {
            std::cerr << label << " packed delivered byte count mismatch\n";
            return false;
        }

        char eof_check = 0;
        if (reader.read_packed_kmers(&eof_check, 1) != 0)
        {
            std::cerr << label << " packed EOF read should return 0\n";
            return false;
        }

        if (actual != expected)
        {
            std::cerr << label << " packed bytes mismatch\n";
            return false;
        }
        return true;
    }

    bool test_k33()
    {
        constexpr uint32_t k = 33;
        const std::vector<Kmer> input{
            make_kmer(0x0123456789abcdefULL, 0xc0000000000000ffULL),
            make_kmer(0xfedcba9876543210ULL, 0x8000000000001234ULL),
            make_kmer(0x1111222233334444ULL, 0x40000000ffffffffULL),
        };

        write_low_bin(k, input, "tmp_export_reader_k33");

        ExportReader<kWords> reader(k);
        reader.open(temp_dir + "low.bin");
        if (reader.finished())
        {
            std::cerr << "k=33 should not be finished after opening non-empty low.bin\n";
            return false;
        }
        if (reader.get_kmer_amount() != input.size())
        {
            std::cerr << "k=33 amount mismatch\n";
            return false;
        }

        std::vector<Kmer> actual(input.size());
        if (reader.read_kmers(actual.data(), 1) != 1)
        {
            std::cerr << "k=33 first read returned wrong count\n";
            return false;
        }
        if (reader.finished())
        {
            std::cerr << "k=33 should not be finished after partial read\n";
            return false;
        }
        if (reader.read_kmers(actual.data() + 1, input.size() - 1) != input.size() - 1)
        {
            std::cerr << "k=33 second read returned wrong count\n";
            return false;
        }
        if (!reader.finished())
        {
            std::cerr << "k=33 should be finished after all k-mers are delivered\n";
            return false;
        }
        if (reader.read_kmers(actual.data(), 1) != 0)
        {
            std::cerr << "k=33 EOF read should return 0\n";
            return false;
        }
        if (!reader.finished())
        {
            std::cerr << "k=33 should remain finished after EOF read\n";
            return false;
        }

        for (size_t i = 0; i < input.size(); ++i)
        {
            if (!expect_equal(actual[i], expected_after_writer_pack(input[i], k), "k=33"))
            {
                return false;
            }
        }
        return expect_packed_reader_matches_file(k, {1, 2}, "k=33");
    }

    bool test_k64()
    {
        constexpr uint32_t k = 64;
        const std::vector<Kmer> input{
            make_kmer(0x0123456789abcdefULL, 0xfedcba9876543210ULL),
            make_kmer(0xaaaaaaaa55555555ULL, 0x123456789abcdef0ULL),
        };

        write_low_bin(k, input, "tmp_export_reader_k64");

        ExportReader<kWords> reader(k);
        reader.open(temp_dir + "low.bin");
        if (reader.finished())
        {
            std::cerr << "k=64 should not be finished after opening non-empty low.bin\n";
            return false;
        }
        if (reader.get_kmer_amount() != input.size())
        {
            std::cerr << "k=64 amount mismatch\n";
            return false;
        }

        std::vector<Kmer> actual(input.size());
        if (reader.read_kmers(actual.data(), input.size()) != input.size())
        {
            std::cerr << "k=64 read returned wrong count\n";
            return false;
        }
        if (!reader.finished())
        {
            std::cerr << "k=64 should be finished after all k-mers are delivered\n";
            return false;
        }
        if (reader.read_kmers(actual.data(), 1) != 0)
        {
            std::cerr << "k=64 EOF read should return 0\n";
            return false;
        }

        for (size_t i = 0; i < input.size(); ++i)
        {
            if (!expect_equal(actual[i], expected_after_writer_pack(input[i], k), "k=64"))
            {
                return false;
            }
        }
        return expect_packed_reader_matches_file(k, {1, 3}, "k=64");
    }

    bool test_empty_low_bin()
    {
        constexpr uint32_t k = 33;
        write_low_bin(k, {}, "tmp_export_reader_empty");

        ExportReader<kWords> reader(k);
        reader.open(temp_dir + "low.bin");
        if (!reader.finished())
        {
            std::cerr << "empty low.bin should be finished after open\n";
            return false;
        }
        if (reader.get_kmer_amount() != 0)
        {
            std::cerr << "empty low.bin amount should be 0\n";
            return false;
        }

        Kmer actual{};
        if (reader.read_kmers(&actual, 1) != 0)
        {
            std::cerr << "empty low.bin read should return 0\n";
            return false;
        }
        return expect_packed_reader_matches_file(k, {1}, "empty low.bin");
    }

    bool test_cross_buffer_reads()
    {
        constexpr uint32_t k = 33;
        constexpr size_t record_count = 60000;
        std::vector<Kmer> input;
        input.reserve(record_count);
        for (size_t i = 0; i < record_count; ++i)
        {
            input.push_back(make_kmer(
                0x0123456789abcdefULL ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL),
                0xc000000000000000ULL | (static_cast<uint64_t>(i) & 0x00ffffffffffffffULL)));
        }

        write_low_bin(k, input, "tmp_export_reader_cross_buffer");

        ExportReader<kWords> reader(k);
        reader.open(temp_dir + "low.bin");
        if (reader.finished())
        {
            std::cerr << "cross-buffer reader should not be finished after opening non-empty low.bin\n";
            return false;
        }
        if (reader.get_kmer_amount() != input.size())
        {
            std::cerr << "cross-buffer amount mismatch\n";
            return false;
        }

        const std::vector<size_t> request_sizes{1, 7, 4096, 12345, 2, 60000};
        size_t read_pos = 0;
        size_t request_index = 0;
        while (!reader.finished())
        {
            const size_t request_size = request_sizes[request_index++ % request_sizes.size()];
            std::vector<Kmer> batch(request_size);
            const uint64_t got = reader.read_kmers(batch.data(), request_size);
            if (got == 0)
            {
                std::cerr << "cross-buffer read returned 0 before finished\n";
                return false;
            }

            for (uint64_t i = 0; i < got; ++i)
            {
                if (!expect_equal(batch[i], expected_after_writer_pack(input[read_pos + i], k), "cross-buffer"))
                {
                    return false;
                }
            }
            read_pos += static_cast<size_t>(got);

            if (read_pos < input.size() && reader.finished())
            {
                std::cerr << "cross-buffer reader finished before all records were delivered\n";
                return false;
            }
        }

        if (read_pos != input.size())
        {
            std::cerr << "cross-buffer delivered count mismatch\n";
            return false;
        }

        Kmer eof_check{};
        if (reader.read_kmers(&eof_check, 1) != 0)
        {
            std::cerr << "cross-buffer EOF read should return 0\n";
            return false;
        }

        return expect_packed_reader_matches_file(k, request_sizes, "cross-buffer");
    }
}

int main()
{
    if (!test_k33())
    {
        return 1;
    }
    if (!test_k64())
    {
        return 1;
    }
    if (!test_empty_low_bin())
    {
        return 1;
    }
    if (!test_cross_buffer_reads())
    {
        return 1;
    }

    std::cout << "ExportReader tests passed\n";
    return 0;
}
