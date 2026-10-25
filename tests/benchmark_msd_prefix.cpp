// Standalone experiment; intentionally not registered with the project build.
// Build with C++20, -O3 -DNDEBUG -march=native, and -pthread -lrt -latomic -lnuma.
#include "../src/GetKmer.h"
#include "../src/BloomFilter.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <sys/resource.h>
#include <unordered_map>

namespace {
using Byte = unsigned char;
using Bytes = std::vector<Byte>;
using View = std::span<const Byte>;
using Clock = std::chrono::steady_clock;
constexpr size_t BATCH_BYTES = 2ULL << 20;
constexpr size_t BLOOM_BLOCKS = 1ULL << 24;
constexpr std::array<uint32_t, 3> K_VALUES{28, 40, 55};
constexpr unsigned FIXED_DEPTH_LIMIT = 4;

unsigned first_depth_limit(uint32_t k)
{
    // Distinct FIRST keys must retain at least one suffix byte.
    return (k - 1) / 4;
}

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

struct Histogram {
    std::array<uint64_t, 33> groups{};
    std::array<uint64_t, 33> records{};
};

void write_u32(Byte* dest, uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i) dest[i] = static_cast<Byte>(value >> (i * 8));
}

uint32_t read_u32(const Byte* source)
{
    uint32_t value = 0;
    for (unsigned i = 0; i < 4; ++i) value |= uint32_t{source[i]} << (i * 8);
    return value;
}

class MsdCodec {
    uint32_t k_;
    size_t width_;
    size_t batch_bytes_;
    Bytes work_;
    Bytes scratch_;
    Bytes encoded_;
    size_t used_ = 0;

    void emit(const Byte* source, uint32_t count, unsigned prefix, Histogram& histogram)
    {
        Byte* dest = encoded_.data() + used_;
        dest[0] = static_cast<Byte>(prefix);
        write_u32(dest + 1, count);
        std::memcpy(dest + 5, source, prefix);
        dest += 5 + prefix;
        const size_t suffix = width_ - prefix;
        if (prefix == 0) {
            std::memcpy(dest, source, size_t{count} * width_);
        } else if (suffix != 0) {
            for (size_t i = 0; i < count; ++i)
                std::memcpy(dest + i * suffix, source + i * width_ + prefix, suffix);
        }
        used_ += 5 + prefix + size_t{count} * suffix;
        ++histogram.groups[prefix];
        histogram.records[prefix] += count;
    }

    void partition(Byte* source, Byte* scratch, uint32_t count, unsigned prefix,
                   unsigned depth, Histogram& histogram)
    {
        if (count <= 1 || prefix == width_ || prefix >= depth) {
            emit(source, count, prefix, histogram);
            return;
        }
        std::array<uint32_t, 256> counts{};
        for (size_t i = 0; i < count; ++i) ++counts[source[i * width_ + prefix]];
        uint32_t branches = 0;
        for (uint32_t n : counts) branches += n != 0;
        const uint64_t parent_bytes = 5 + prefix + uint64_t{count} * (width_ - prefix);
        const uint64_t child_bytes = uint64_t{branches} * (6 + prefix)
                                   + uint64_t{count} * (width_ - prefix - 1);
        if (child_bytes >= parent_bytes) {
            emit(source, count, prefix, histogram);
            return;
        }
        if (branches == 1) {
            partition(source, scratch, count, prefix + 1, depth, histogram);
            return;
        }
        std::array<uint32_t, 256> positions{};
        uint32_t offset = 0;
        for (size_t digit = 0; digit < counts.size(); ++digit) {
            positions[digit] = offset;
            offset += counts[digit];
        }
        for (size_t i = 0; i < count; ++i) {
            const Byte* key = source + i * width_;
            const uint32_t slot = positions[key[prefix]]++;
            std::memcpy(scratch + size_t{slot} * width_, key, width_);
        }
        offset = 0;
        for (uint32_t n : counts) {
            if (n) {
                // Alternate the two arrays; child ranges never overlap siblings.
                partition(scratch + size_t{offset} * width_, source + size_t{offset} * width_,
                          n, prefix + 1, depth, histogram);
            }
            offset += n;
        }
    }

public:
    explicit MsdCodec(uint32_t k, size_t batch_bytes = BATCH_BYTES)
        : k_(k), width_((k + 3) / 4), batch_bytes_(batch_bytes),
          work_(batch_bytes), scratch_(batch_bytes), encoded_(batch_bytes + 13)
    {
        require(k >= 1 && k <= MAX_K, "invalid codec k");
        require(batch_bytes >= width_, "batch cannot hold one key");
    }
    View encode(View input, unsigned depth, bool stream_header, Histogram& histogram)
    {
        require(input.size() <= batch_bytes_ && input.size() % width_ == 0, "invalid input batch");
        used_ = 0;
        if (stream_header) {
            write_u32(encoded_.data(), 1);
            write_u32(encoded_.data() + 4, k_);
            used_ = 8;
        }
        if (!input.empty()) {
            std::memcpy(work_.data(), input.data(), input.size());
            partition(work_.data(), scratch_.data(), static_cast<uint32_t>(input.size() / width_),
                      0, depth, histogram);
        }
        return {encoded_.data(), used_};
    }
    View decode(View input, bool stream_header, Bytes& output) const
    {
        require(output.size() >= batch_bytes_, "decoder output buffer too small");
        size_t pos = 0, output_bytes = 0;
        if (stream_header) {
            require(input.size() >= 8, "truncated stream header");
            require(read_u32(input.data()) == 1, "unsupported stream version");
            require(read_u32(input.data() + 4) == k_, "stream k mismatch");
            pos = 8;
        }
        while (pos < input.size()) {
            require(input.size() - pos >= 5, "truncated group header");
            const size_t prefix = input[pos];
            const uint32_t count = read_u32(input.data() + pos + 1);
            pos += 5;
            require(prefix <= width_, "prefix longer than key");
            require(count != 0, "zero record count");
            require(count <= (batch_bytes_ - output_bytes) / width_, "decoded batch too large");
            const size_t suffix = width_ - prefix;
            const size_t suffix_bytes = size_t{count} * suffix;
            require(input.size() - pos >= prefix, "truncated prefix");
            require(input.size() - pos - prefix >= suffix_bytes, "truncated suffix");
            const Byte* shared = input.data() + pos;
            const Byte* suffixes = shared + prefix;
            Byte* dest = output.data() + output_bytes;
            if (prefix == 0) {
                std::memcpy(dest, suffixes, suffix_bytes);
            } else {
                for (size_t i = 0; i < count; ++i) {
                    std::memcpy(dest + i * width_, shared, prefix);
                    std::memcpy(dest + i * width_ + prefix, suffixes + i * suffix, suffix);
                }
            }
            pos += prefix + suffix_bytes;
            output_bytes += size_t{count} * width_;
        }
        return {output.data(), output_bytes};
    }
};

class FastqRecords {
    std::istream& input_;
    std::string header_, plus_, quality_;
public:
    explicit FastqRecords(std::istream& input) : input_(input) {}
    bool next(std::string& sequence)
    {
        if (!std::getline(input_, header_)) {
            require(input_.eof() && !input_.bad(), "FASTQ read failed");
            return false;
        }
        require(static_cast<bool>(std::getline(input_, sequence))
                && static_cast<bool>(std::getline(input_, plus_))
                && static_cast<bool>(std::getline(input_, quality_)), "truncated FASTQ record");
        for (std::string* line : {&header_, &sequence, &plus_, &quality_})
            if (!line->empty() && line->back() == '\r') line->pop_back();
        require(!header_.empty() && header_[0] == '@', "invalid FASTQ header");
        require(!plus_.empty() && plus_[0] == '+', "invalid FASTQ separator");
        require(sequence.size() == quality_.size(), "FASTQ sequence/quality length mismatch");
        return true;
    }
};

struct Result {
    uint32_t k = 0;
    unsigned depth = 0;
    std::string_view mode;
    size_t batch_bytes = 0;
    uint64_t reads = 0, valid_records = 0, first_records = 0, batches = 0;
    uint64_t full_batches = 0, tail_records = 0;
    uint64_t bloom_blocks = 0, encoded_bytes = 0, peak_rss_kib = 0;
    double encode_seconds = 0, decode_seconds = 0;
    Histogram histogram;
};

std::array<Result, 3> run_k(std::istream&, uint32_t, size_t, size_t = BATCH_BYTES);

// Comparison uses full records and keeps duplicates; sorting is verification only.
void sort_records(View input, size_t width, Bytes& sorted, std::vector<uint32_t>& order)
{
    order.resize(input.size() / width);
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [input, width](uint32_t a, uint32_t b) {
        return std::memcmp(input.data() + size_t{a} * width,
                           input.data() + size_t{b} * width, width) < 0;
    });
    sorted.resize(input.size());
    for (size_t i = 0; i < order.size(); ++i)
        std::memcpy(sorted.data() + i * width, input.data() + size_t{order[i]} * width, width);
}

std::array<Result, 3> run_k(std::istream& input, uint32_t k, size_t bloom_blocks, size_t batch_bytes)
{
    k_length = k;
    const size_t width = kmer_bytes();
    const size_t batch_capacity = batch_bytes / width;
    void* allocation = nullptr;
    if (::posix_memalign(&allocation, 64, bloom_bytes(bloom_blocks)) || !allocation) throw std::bad_alloc();
    std::unique_ptr<void, decltype(&std::free)> storage(allocation, &std::free);
    ConcurrentBloomFilter bloom(bloom_blocks, nullptr, storage.get());
    GetKmer extractor;
    MsdCodec codec(k, batch_bytes);
    Bytes batch(batch_bytes), decoded(batch_bytes), sorted_input, sorted_output;
    sorted_input.reserve(batch_bytes);
    sorted_output.reserve(batch_bytes);
    std::vector<uint32_t> order;
    order.reserve(batch_capacity);
    std::array<Result, 3> rows{};
    rows[0].mode = "raw";
    rows[1].mode = "fixed4";
    rows[2].mode = "auto";
    rows[1].depth = std::min(FIXED_DEPTH_LIMIT, first_depth_limit(k));
    rows[2].depth = first_depth_limit(k);
    uint64_t reads = 0, valid_records = 0, first_records = 0, batches = 0;
    uint64_t full_batches = 0, tail_records = 0;
    size_t batch_count = 0;
    bool stream_header = true;

    const auto flush_batch = [&] {
        const View raw(batch.data(), batch_count * width);
        sort_records(raw, width, sorted_input, order);
        for (Result& row : rows) {
            const auto encode_start = Clock::now();
            const View encoded = codec.encode(raw, row.depth, stream_header, row.histogram);
            row.encode_seconds += std::chrono::duration<double>(Clock::now() - encode_start).count();
            row.encoded_bytes += encoded.size();
            const auto decode_start = Clock::now();
            const View output = codec.decode(encoded, stream_header, decoded);
            row.decode_seconds += std::chrono::duration<double>(Clock::now() - decode_start).count();
            require(output.size() == raw.size(), "FASTQ batch decoded count mismatch");
            sort_records(output, width, sorted_output, order);
            require(sorted_input == sorted_output, "FASTQ batch decoded multiset mismatch");
        }
        stream_header = false;
        if (batch_count) {
            ++batches;
            if (batch_count == batch_capacity) ++full_batches;
            else tail_records = batch_count;
        }
        batch_count = 0;
        if (batches && batches % 16 == 0)
            std::cerr << "k=" << k << " batches=" << batches << " FIRST=" << first_records << '\n';
    };

    FastqRecords reader(input);
    std::string sequence;
    std::array<Byte, (MAX_K + 3) / 4> key{};
    while (reader.next(sequence)) {
        ++reads;
        extractor.clear();
        for (unsigned char base : sequence) {
            if (!extractor.get_next_one(base, key.data())) continue;
            ++valid_records;
            if (bloom.insert(key.data()) != Occurrence::FIRST) continue;
            ++first_records;
            std::memcpy(batch.data() + batch_count * width, key.data(), width);
            if (++batch_count == batch_capacity) flush_batch();
        }
    }
    if (batch_count || stream_header) flush_batch();
    rusage usage{};
    require(::getrusage(RUSAGE_SELF, &usage) == 0, "getrusage failed");
    for (Result& row : rows) {
        row.k = k;
        row.batch_bytes = batch_bytes;
        row.reads = reads;
        row.valid_records = valid_records;
        row.first_records = first_records;
        row.batches = batches;
        row.full_batches = full_batches;
        row.tail_records = tail_records;
        row.bloom_blocks = bloom_blocks;
        row.peak_rss_kib = static_cast<uint64_t>(usage.ru_maxrss);
    }
    return rows;
}

void check_roundtrip(MsdCodec& codec, View input, uint32_t k, unsigned depth)
{
    Histogram histogram;
    const View encoded = codec.encode(input, depth, true, histogram);
    Bytes output(BATCH_BYTES), expected, actual;
    std::vector<uint32_t> order;
    const View decoded = codec.decode(encoded, true, output);
    require(decoded.size() == input.size(), "roundtrip record count changed");
    const size_t width = (k + 3) / 4;
    sort_records(input, width, expected, order);
    sort_records(decoded, width, actual, order);
    require(actual == expected, "roundtrip multiset changed");
    uint64_t measured_bytes = 8, records = 0;
    for (size_t p = 0; p < histogram.groups.size(); ++p) {
        require(!histogram.groups[p] || p <= std::min<size_t>(depth, width), "depth limit exceeded");
        measured_bytes += histogram.groups[p] * (5 + p) + histogram.records[p] * (width - std::min(p, width));
        records += histogram.records[p];
    }
    require(records == input.size() / width, "histogram lost records");
    require(measured_bytes == encoded.size(), "serialized size disagrees with histogram");
    require(encoded.size() <= 8 + (input.empty() ? 0 : input.size() + 5), "adaptive split enlarged raw group");
}

void codec_self_tests()
{
    // Hand-derived wire bytes catch missing headers and struct padding.
    MsdCodec small(5);
    Histogram histogram;
    const Bytes one{0x12, 0x40};
    const Bytes expected{1, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0x12, 0x40};
    const View encoded_one = small.encode(one, 0, true, histogram);
    require(Bytes(encoded_one.begin(), encoded_one.end()) == expected, "raw group wire format changed");
    check_roundtrip(small, {}, 5, 4);
    check_roundtrip(small, one, 5, 4);

    Bytes identical(20 * 2);
    for (size_t i = 0; i < 20; ++i) {
        identical[i * 2] = 0x12;
        identical[i * 2 + 1] = 0x40;
    }
    Histogram full_prefix;
    require(small.encode(identical, 4, true, full_prefix).size() == 15, "zero-length suffix not encoded");
    require(full_prefix.groups[2] == 1 && full_prefix.records[2] == 20, "identical group not preserved");
    check_roundtrip(small, identical, 5, 4);
    Histogram limited;
    require(small.encode(identical, 1, true, limited).size() == 34, "depth cap did not retain suffixes");

    Bytes fragmented(256 * 2);
    for (size_t i = 0; i < 256; ++i) {
        fragmented[i * 2] = static_cast<Byte>(i);
        fragmented[i * 2 + 1] = 0;
    }
    Histogram fallback;
    require(small.encode(fragmented, 4, true, fallback).size() == 525, "unprofitable split was accepted");
    require(fallback.groups[0] == 1, "raw fallback missing");

    const Bytes tied{0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
    Histogram equal_cost;
    small.encode(tied, 4, true, equal_cost);
    require(equal_cost.groups[0] == 1, "equal-cost split was accepted");

    // A profitable two-way split exercises scatter and both child ranges.
    Bytes skewed(400 * 2);
    for (size_t i = 0; i < 400; ++i) {
        skewed[i * 2] = i % 4 ? 0x12 : 0x34;
        skewed[i * 2 + 1] = static_cast<Byte>((i % 4) << 6);
    }
    Histogram branches;
    require(small.encode(skewed, 1, true, branches).size() == 420, "two-way split size is wrong");
    require(branches.groups[1] == 2, "scatter lost a branch");
    check_roundtrip(small, skewed, 5, 4);

    std::mt19937 random(1);
    for (uint32_t k : {1U, 5U, 28U, 40U, 55U, 128U}) {
        MsdCodec codec(k);
        const size_t width = (k + 3) / 4;
        Bytes records(2048 * width);
        for (Byte& byte : records) byte = static_cast<Byte>(random());
        if (k % 4) {
            const Byte mask = static_cast<Byte>(0xffU << (8 - 2 * (k % 4)));
            for (size_t i = width - 1; i < records.size(); i += width) records[i] &= mask;
        }
        for (unsigned depth : {0U, FIXED_DEPTH_LIMIT, first_depth_limit(k)})
            check_roundtrip(codec, records, k, depth);
    }

    // Headerless continuation batches use the same k as the initial stream.
    Histogram continuation;
    const View continuation_bytes = small.encode(one, 0, false, continuation);
    Bytes decoded(BATCH_BYTES);
    const View restored = small.decode(continuation_bytes, false, decoded);
    require(restored.size() == one.size() && std::equal(restored.begin(), restored.end(), one.begin()),
            "continuation batch changed records");

    const std::vector<Bytes> malformed{
        {1, 0, 0},                                      // Truncated stream header.
        {2, 0, 0, 0, 5, 0, 0, 0},                       // Unsupported version.
        {1, 0, 0, 0, 0, 0, 0, 0},                       // Invalid k.
        {1, 0, 0, 0, 4, 0, 0, 0},                       // Wrong stream k.
        {1, 0, 0, 0, 5, 0, 0, 0, 0, 1},                 // Truncated group header.
        {1, 0, 0, 0, 5, 0, 0, 0, 3, 1, 0, 0, 0},        // Prefix longer than key.
        {1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0},        // Zero record count.
        {1, 0, 0, 0, 5, 0, 0, 0, 0, 1, 0, 0, 0, 0x12},  // Truncated suffix.
        {1, 0, 0, 0, 5, 0, 0, 0, 2, 1, 0, 0, 0, 0x12},  // Truncated prefix.
        {1, 0, 0, 0, 5, 0, 0, 0, 2, 255, 255, 255, 255, 0x12, 0x40}
    };
    for (const Bytes& bad : malformed) {
        bool rejected = false;
        try { small.decode(bad, true, decoded); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "malformed stream accepted");
    }
    std::cout << "Codec self-tests passed\n";
}

void fastq_self_tests()
{
    std::istringstream valid("@one\r\nACGTA\r\n+\r\n!!!!!\r\n@two\nacNta\n+two\n#####");
    FastqRecords records(valid);
    std::string sequence;
    require(records.next(sequence) && sequence == "ACGTA", "CRLF FASTQ was changed");
    require(records.next(sequence) && sequence == "acNta", "FASTQ sequence was changed");
    require(!records.next(sequence), "FASTQ EOF not detected");
    for (const char* text : {"@x\nAC\n+\n", "x\nAC\n+\n!!\n",
                             "@x\nAC\nx\n!!\n", "@x\nAC\n+\n!\n"}) {
        std::istringstream bad(text);
        FastqRecords reader(bad);
        bool rejected = false;
        try { reader.next(sequence); }
        catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "malformed FASTQ accepted");
    }

    // Homopolymers have two distinct canonical keys. Short reads and the N
    // must not connect windows across read or invalid-base boundaries.
    std::ostringstream fixture;
    for (const std::string& read : {std::string(64, 'A'), std::string(64, 'A'), std::string(64, 'C'),
                                   std::string(20, 'A'), std::string(20, 'A'),
                                   std::string(20, 'A') + 'N' + std::string(20, 'C')})
        fixture << "@read\n" << read << "\n+\n" << std::string(read.size(), '!') << '\n';
    for (uint32_t k : K_VALUES) {
        std::istringstream input(fixture.str());
        const auto rows = run_k(input, k, 64);
        for (const Result& row : rows) {
            require(row.reads == 6, "FASTQ read count changed");
            require(row.valid_records == 3 * (65 - k), "read or N boundary leaked k-mers");
            require(row.first_records == 2 && row.batches == 1, "Bloom FIRST extraction changed");
            require(row.encoded_bytes == 13 + 2 * ((k + 3) / 4), "FIRST encoded size changed");
        }
    }
    std::istringstream empty;
    const auto empty_rows = run_k(empty, 28, 64);
    for (const Result& row : empty_rows)
        require(row.first_records == 0 && row.batches == 0 && row.encoded_bytes == 8,
                "empty FASTQ stream was not represented by its header");
    std::cout << "FASTQ and FIRST self-tests passed\n";
}

void automatic_depth_self_tests()
{
    const std::array<std::pair<uint32_t, unsigned>, 8> limits{{
        {1, 0}, {4, 0}, {5, 1}, {8, 1}, {28, 6}, {40, 9}, {55, 13}, {128, 31}
    }};
    for (const auto& [k, expected] : limits)
        require(first_depth_limit(k) == expected, "automatic FIRST depth limit is wrong");

    // Each of the first d bytes branches two ways. Sixteen distinct final
    // bytes amortize every split, including scatters beyond four bytes.
    const std::array<std::pair<uint32_t, size_t>, 3> cases{{{28, 1736}, {40, 15368}, {55, 278536}}};
    for (const auto& [k, expected_bytes] : cases) {
        const size_t width = (k + 3) / 4;
        const unsigned depth = first_depth_limit(k);
        const uint32_t leaves = 1U << depth;
        Bytes records(size_t{16} * leaves * width);
        for (unsigned suffix = 0; suffix < 16; ++suffix) {
            for (uint32_t route = 0; route < leaves; ++route) {
                Byte* key = records.data() + (size_t{suffix} * leaves + route) * width;
                for (unsigned p = 0; p < depth; ++p)
                    key[p] = static_cast<Byte>((route >> (depth - 1 - p)) & 1U);
                key[width - 1] = static_cast<Byte>(suffix << (k % 4 ? 8 - 2 * (k % 4) : 0));
            }
        }
        MsdCodec codec(k);
        Histogram histogram;
        require(codec.encode(records, depth, true, histogram).size() == expected_bytes,
                "deep branching encoded size is wrong");
        require(histogram.groups[depth] == leaves && histogram.records[depth] == size_t{16} * leaves,
                "deep branching did not reach the automatic depth limit");
        check_roundtrip(codec, records, k, depth);
        if (k == 28) {
            Histogram third, fourth;
            require(codec.encode(records, 3, true, third).size() == 4168 && third.groups[3] == 8,
                    "third-level branch handling is wrong");
            require(codec.encode(records, 4, true, fourth).size() == 3224 && fourth.groups[4] == 16,
                    "fourth-level branch handling is wrong");
        }
    }

    const size_t larger_batch = 4ULL << 20;
    MsdCodec larger(28, larger_batch);
    Bytes raw(larger_batch / 7 * 7, 0x40), decoded(larger_batch);
    Histogram histogram;
    const View encoded = larger.encode(raw, 0, true, histogram);
    require(encoded.size() == raw.size() + 13, "4 MiB raw group overhead is wrong");
    const View restored = larger.decode(encoded, true, decoded);
    require(restored.size() == raw.size() && std::equal(restored.begin(), restored.end(), raw.begin()),
            "4 MiB batch roundtrip changed records");
    MsdCodec smaller(28);
    Bytes too_small(BATCH_BYTES);
    bool rejected = false;
    try { smaller.decode(encoded, true, too_small); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "decoder accepted more records than its batch capacity");
    std::cout << "Automatic depth and variable-capacity self-tests passed\n";
}

void batch_boundary_self_tests()
{
    // Thirty bytes hold four seven-byte records, leaving two unusable bytes.
    for (unsigned count : {0U, 1U, 3U, 4U, 5U, 8U, 9U}) {
        std::ostringstream fixture;
        for (unsigned i = 0; i < count; ++i) {
            std::string read(28, 'A');
            read[26] = "ACGT"[i / 4];
            read[27] = "ACGT"[i % 4];
            fixture << "@unique\n" << read << "\n+\n" << std::string(28, '!') << '\n';
        }
        if (count)
            fixture << "@duplicate\n" << std::string(28, 'A') << "\n+\n" << std::string(28, '!') << '\n';
        std::istringstream input(fixture.str());
        const auto rows = run_k(input, 28, 1024, 30);
        require(rows.size() == 3, "streaming experiment must compare exactly three configurations");
        const std::array<std::string_view, 3> names{"raw", "fixed4", "auto"};
        const std::array<unsigned, 3> depths{0, 4, 6};
        for (size_t i = 0; i < rows.size(); ++i) {
            const Result& row = rows[i];
            require(row.mode == names[i] && row.depth == depths[i], "configuration depth or label is wrong");
            require(row.batch_bytes == 30, "configured streaming capacity was ignored");
            require(row.first_records == count, "Bloom state did not persist across batches");
            require(row.batches == (count + 3) / 4 && row.full_batches == count / 4
                    && row.tail_records == count % 4, "streaming batch boundary is wrong");
        }
        require(rows[0].encoded_bytes == 8 + count * 7 + rows[0].batches * 5,
                "batch flushing duplicated headers or emitted an extra EOF group");
        require(rows[2].encoded_bytes <= rows[1].encoded_bytes
                && rows[1].encoded_bytes <= rows[0].encoded_bytes, "adaptive configuration enlarged output");
    }
    std::cout << "Streaming batch-boundary self-tests passed\n";
}

void write_results(std::ostream& csv, const std::vector<Result>& rows, std::ostream& display = std::cout)
{
    csv << "k,batch_mib,batch_bytes,mode,depth_limit,max_prefix,reads,valid_kmers,first_kmers,"
           "batches,full_batches,tail_records,record_bytes,bloom_blocks,bloom_bytes,"
           "raw_first_bytes,encoded_bytes,saved_bytes,saved_pct,encoded_over_raw_pct,metadata_bytes,"
           "encode_seconds,decode_seconds,encode_mib_s,decode_mib_s,process_peak_rss_kib,verified";
    for (unsigned p = 0; p <= MAX_K / 4; ++p)
        csv << ",groups_p" << p << ",records_p" << p << ",metadata_bytes_p" << p;
    csv << '\n' << std::fixed << std::setprecision(9);
    display << "\nk  MiB mode   limit maxp FIRST       raw_bytes     encoded_bytes saved_bytes   saved%    enc_MiB/s dec_MiB/s\n";
    for (const Result& row : rows) {
        const size_t width = (row.k + 3) / 4;
        const uint64_t raw = row.first_records * width;
        const int64_t saved = static_cast<int64_t>(raw) - static_cast<int64_t>(row.encoded_bytes);
        const double raw_mib = static_cast<double>(raw) / (1ULL << 20);
        const double encode_speed = row.encode_seconds > 0 ? raw_mib / row.encode_seconds : 0;
        const double decode_speed = row.decode_seconds > 0 ? raw_mib / row.decode_seconds : 0;
        uint64_t metadata = 8;
        unsigned max_prefix = 0;
        for (size_t p = 0; p < row.histogram.groups.size(); ++p) {
            metadata += row.histogram.groups[p] * (5 + p);
            if (row.histogram.groups[p]) max_prefix = static_cast<unsigned>(p);
        }
        csv << row.k << ',' << row.batch_bytes / (1ULL << 20) << ',' << row.batch_bytes << ','
            << row.mode << ',' << row.depth << ',' << max_prefix << ',' << row.reads << ',' << row.valid_records << ','
            << row.first_records << ',' << row.batches << ',' << row.full_batches << ',' << row.tail_records << ','
            << width << ',' << row.bloom_blocks << ','
            << bloom_bytes(row.bloom_blocks) << ',' << raw << ',' << row.encoded_bytes << ',' << saved << ',';
        if (raw) csv << 100.0 * static_cast<double>(saved) / raw;
        else csv << "N/A";
        csv << ',';
        if (raw) csv << 100.0 * static_cast<double>(row.encoded_bytes) / raw;
        else csv << "N/A";
        csv << ',' << metadata << ',' << row.encode_seconds << ',' << row.decode_seconds << ','
            << encode_speed << ',' << decode_speed << ',' << row.peak_rss_kib << ",true";
        for (unsigned p = 0; p <= MAX_K / 4; ++p)
            csv << ',' << row.histogram.groups[p] << ',' << row.histogram.records[p] << ','
                << row.histogram.groups[p] * (5 + p);
        csv << '\n';
        display << std::setw(2) << row.k << ' ' << std::setw(3) << row.batch_bytes / (1ULL << 20) << ' '
                  << std::setw(6) << row.mode << ' ' << std::setw(4) << row.depth << ' '
                  << std::setw(4) << max_prefix << ' '
                  << std::setw(10) << row.first_records << ' ' << std::setw(13) << raw << ' '
                  << std::setw(13) << row.encoded_bytes << ' ' << std::setw(11) << saved << ' '
                  << std::fixed << std::setprecision(3);
        if (raw) display << std::setw(8) << 100.0 * static_cast<double>(saved) / raw;
        else display << std::setw(8) << "N/A";
        display << ' ' << std::setw(9) << encode_speed << ' ' << std::setw(9) << decode_speed << '\n';
    }
}

void report_self_tests()
{
    Result row;
    row.k = 28;
    row.depth = 6;
    row.mode = "auto";
    row.batch_bytes = 4ULL << 20;
    row.first_records = row.tail_records = 1024;
    row.batches = 1;
    row.encoded_bytes = 1736;
    row.histogram.groups[6] = 64;
    row.histogram.records[6] = 1024;
    std::ostringstream csv, display;
    write_results(csv, {row}, display);
    std::istringstream lines(csv.str());
    std::string header, data, name, value;
    std::getline(lines, header);
    std::getline(lines, data);
    std::istringstream names(header), values(data);
    std::unordered_map<std::string, std::string> fields;
    while (std::getline(names, name, ',')) {
        require(static_cast<bool>(std::getline(values, value, ',')), "CSV row has too few columns");
        require(fields.emplace(name, value).second, "duplicate CSV column");
    }
    require(!std::getline(values, value, ','), "CSV row has too many columns");
    for (const char* required : {"batch_mib", "batch_bytes", "mode", "depth_limit", "max_prefix",
                                 "full_batches", "tail_records", "groups_p6", "records_p6", "groups_p32"})
        require(fields.contains(required), "CSV does not expose batch or deep-prefix results");
    require(fields.at("batch_mib") == "4" && fields.at("batch_bytes") == "4194304"
            && fields.at("mode") == "auto" && fields.at("depth_limit") == "6"
            && fields.at("max_prefix") == "6", "CSV configuration metadata is wrong");
    require(fields.at("full_batches") == "0" && fields.at("tail_records") == "1024",
            "CSV batch accounting is wrong");
    require(fields.at("groups_p6") == "64" && fields.at("records_p6") == "1024"
            && fields.at("metadata_bytes") == "712" && fields.at("saved_bytes") == "5432",
            "CSV deep-prefix byte accounting is wrong");
    require(display.str().find("auto") != std::string::npos, "terminal output lost the configuration label");
    std::cout << "Report self-tests passed\n";
}
} // namespace

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--self-test") {
            automatic_depth_self_tests();
            batch_boundary_self_tests();
            codec_self_tests();
            fastq_self_tests();
            report_self_tests();
            return 0;
        }
        std::filesystem::path fastq, output_dir;
        size_t batch_bytes = BATCH_BYTES;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option(argv[i]);
            require((option == "--fastq" || option == "--output-dir" || option == "--batch-mib") && i + 1 < argc,
                    "Usage: benchmark_msd_prefix --self-test | --fastq PATH --output-dir DIR [--batch-mib 2|4]");
            if (option == "--fastq") fastq = argv[++i];
            else if (option == "--output-dir") output_dir = argv[++i];
            else {
                const std::string_view value(argv[++i]);
                require(value == "2" || value == "4", "--batch-mib must be 2 or 4");
                batch_bytes = (value == "2" ? 2ULL : 4ULL) << 20;
            }
        }
        require(!fastq.empty() && !output_dir.empty(),
                "Usage: benchmark_msd_prefix --self-test | --fastq PATH --output-dir DIR [--batch-mib 2|4]");
        std::filesystem::create_directories(output_dir);
        std::vector<Result> rows;
        for (uint32_t k : K_VALUES) {
            std::ifstream input(fastq, std::ios::binary);
            require(input.is_open(), "cannot open FASTQ input");
            std::cerr << "Starting k=" << k << ", Bloom=" << bloom_bytes(BLOOM_BLOCKS)
                      << " bytes, batch=" << batch_bytes << " bytes, auto_limit=" << first_depth_limit(k) << '\n';
            const auto result = run_k(input, k, BLOOM_BLOCKS, batch_bytes);
            rows.insert(rows.end(), result.begin(), result.end());
            std::cerr << "Finished k=" << k << ": reads=" << result[0].reads
                      << " valid=" << result[0].valid_records << " FIRST=" << result[0].first_records
                      << " batches=" << result[0].batches << " full=" << result[0].full_batches
                      << " tail_records=" << result[0].tail_records << "; all configurations verified\n";
        }
        std::ofstream csv(output_dir / "summary.csv");
        require(csv.is_open(), "cannot create summary.csv");
        write_results(csv, rows);
        csv.close();
        require(!csv.fail(), "cannot write summary.csv");
        std::cout << "\nAll configurations passed exact roundtrip verification.\n"
                  << "CSV: " << output_dir / "summary.csv" << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
