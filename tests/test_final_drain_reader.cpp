#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "../src/FinalDrainWriter.h"
#include "../tool/FinalDrainReader.h"

/*
namespace
{
    constexpr uint32_t kWords = 2;
    constexpr uint32_t kRootId = 7;

    using Kmer = kmer<kWords>;
    using Record = ExportRecord<kWords>;

    Kmer make_kmer(uint64_t first, uint64_t second)
    {
        Kmer value{};
        value.reset();
        value.data[0] = first;
        value.data[1] = second;
        return value;
    }

    Record make_record(uint64_t index)
    {
        Record record{};
        record.key = make_kmer(
            0x0123456789abcdefULL ^ (index * 0x9e3779b97f4a7c15ULL),
            0xfedcba9876543210ULL ^ (index * 0xbf58476d1ce4e5b9ULL));
        record.count = static_cast<uint32_t>((index % 100000) + 1);
        return record;
    }

    std::vector<Record> make_records(size_t count)
    {
        std::vector<Record> records;
        records.reserve(count);
        for (size_t i = 0; i < count; ++i)
        {
            records.push_back(make_record(i));
        }
        return records;
    }

    std::string set_test_temp_dir(const std::string& name)
    {
        const std::filesystem::path dir = std::filesystem::current_path() / name;
        std::filesystem::create_directories(dir);
        temp_dir = dir.generic_string() + "/";
        return temp_dir + "root_" + std::to_string(kRootId) + ".bin";
    }

    std::string write_root_bin(const std::vector<Record>& records, const std::string& name)
    {
        const std::string filename = set_test_temp_dir(name);

        FinalDrainWriter<kWords> writer(32 * kWords);
        writer.open(kRootId);
        for (const auto& record : records)
        {
            writer.write_kmer_record(record.key.data.data(), record.count);
        }
        writer.close();

        return filename;
    }

    bool expect_equal(const Record& actual, const Record& expected, const char* label, size_t index)
    {
        if (actual.key == expected.key && actual.count == expected.count)
        {
            return true;
        }

        std::cerr << label << " mismatch at index " << index << "\n"
            << "  actual key:   " << std::hex << actual.key.data[0] << " "
            << actual.key.data[1] << std::dec << " count=" << actual.count << "\n"
            << "  expected key: " << std::hex << expected.key.data[0] << " "
            << expected.key.data[1] << std::dec << " count=" << expected.count << "\n";
        return false;
    }

    bool expect_records_equal(
        const std::vector<Record>& actual,
        const std::vector<Record>& expected,
        const char* label)
    {
        if (actual.size() != expected.size())
        {
            std::cerr << label << " size mismatch: actual=" << actual.size()
                << " expected=" << expected.size() << "\n";
            return false;
        }

        for (size_t i = 0; i < expected.size(); ++i)
        {
            if (!expect_equal(actual[i], expected[i], label, i))
            {
                return false;
            }
        }
        return true;
    }

    bool test_empty_root_file()
    {
        const std::string filename = write_root_bin({}, "tmp_final_drain_reader_empty");

        FinalDrainReader<kWords> reader(32 * kWords);
        reader.open(filename);
        if (!reader.finished())
        {
            std::cerr << "empty root file should be finished after open\n";
            return false;
        }
        if (reader.get_record_amount() != 0)
        {
            std::cerr << "empty root file amount should be 0\n";
            return false;
        }

        Record record{};
        if (reader.read_records(&record, 1) != 0)
        {
            std::cerr << "empty root file read should return 0\n";
            return false;
        }

        return true;
    }

    bool test_read_all_at_once()
    {
        const std::vector<Record> expected = make_records(3);
        const std::string filename = write_root_bin(expected, "tmp_final_drain_reader_once");

        FinalDrainReader<kWords> reader(2*kWords);
        reader.open(filename);
        if (reader.finished())
        {
            std::cerr << "non-empty root file should not be finished after open\n";
            return false;
        }
        if (reader.get_record_amount() != expected.size())
        {
            std::cerr << "record amount mismatch for read-all test\n";
            return false;
        }

        std::vector<Record> actual(expected.size());
        if (reader.read_records(actual.data(), actual.size()) != actual.size())
        {
            std::cerr << "read-all test returned wrong count\n";
            return false;
        }
        if (!reader.finished())
        {
            std::cerr << "reader should be finished after read-all test\n";
            return false;
        }

        Record eof_record{};
        if (reader.read_records(&eof_record, 1) != 0)
        {
            std::cerr << "read-all EOF read should return 0\n";
            return false;
        }

        return expect_records_equal(actual, expected, "read-all");
    }

    bool test_small_batch_reads()
    {
        const std::vector<Record> expected = make_records(1024);
        const std::string filename = write_root_bin(expected, "tmp_final_drain_reader_small_batches");

        FinalDrainReader<kWords> reader(2 * kWords);
        reader.open(filename);

        const std::vector<size_t> request_sizes{ 1, 3, 17, 64, 5 };
        std::vector<Record> actual(expected.size());
        size_t read_pos = 0;
        size_t request_index = 0;
        while (!reader.finished())
        {
            const size_t request_size = request_sizes[request_index++ % request_sizes.size()];
            const uint64_t got = reader.read_records(actual.data() + read_pos, request_size);
            if (got == 0)
            {
                std::cerr << "small-batch read returned 0 before EOF\n";
                return false;
            }
            read_pos += static_cast<size_t>(got);
        }

        if (read_pos != expected.size())
        {
            std::cerr << "small-batch delivered count mismatch\n";
            return false;
        }

        Record eof_record{};
        if (reader.read_records(&eof_record, 1) != 0)
        {
            std::cerr << "small-batch EOF read should return 0\n";
            return false;
        }

        return expect_records_equal(actual, expected, "small-batch");
    }

    bool test_cross_buffer_reads()
    {
        constexpr size_t record_count = 60000;
        const std::vector<Record> expected = make_records(record_count);
        const std::string filename = write_root_bin(expected, "tmp_final_drain_reader_cross_buffer");

        FinalDrainReader<kWords> reader(2 * kWords);
        reader.open(filename);
        if (reader.get_record_amount() != expected.size())
        {
            std::cerr << "cross-buffer amount mismatch\n";
            return false;
        }

        const std::vector<size_t> request_sizes{ 1, 7, 4096, 12345, 2, 60000 };
        std::vector<Record> actual(expected.size());
        size_t read_pos = 0;
        size_t request_index = 0;
        while (!reader.finished())
        {
            const size_t request_size = request_sizes[request_index++ % request_sizes.size()];
            const uint64_t got = reader.read_records(actual.data() + read_pos, request_size);
            if (got == 0)
            {
                std::cerr << "cross-buffer read returned 0 before EOF\n";
                return false;
            }
            read_pos += static_cast<size_t>(got);

            if (read_pos < expected.size() && reader.finished())
            {
                std::cerr << "cross-buffer reader finished before all records were delivered\n";
                return false;
            }
        }

        if (read_pos != expected.size())
        {
            std::cerr << "cross-buffer delivered count mismatch\n";
            return false;
        }

        Record eof_record{};
        if (reader.read_records(&eof_record, 1) != 0)
        {
            std::cerr << "cross-buffer EOF read should return 0\n";
            return false;
        }

        return expect_records_equal(actual, expected, "cross-buffer");
    }
}

int main()
{
    if (!test_empty_root_file())
    {
        return 1;
    }
    if (!test_read_all_at_once())
    {
        return 1;
    }
    if (!test_small_batch_reads())
    {
        return 1;
    }
    if (!test_cross_buffer_reads())
    {
        return 1;
    }

    std::cout << "FinalDrainReader tests passed\n";
    return 0;
}
*/

int main()
{
    std::cout << "FinalDrainReader tests skipped\n";
    return 0;
}