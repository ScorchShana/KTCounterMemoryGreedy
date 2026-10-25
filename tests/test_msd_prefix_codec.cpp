#include "../src/MsdPrefixCodec.h"
#include "low_msd_test_helpers.h"
#include <sstream>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

using Bytes = std::vector<uint8_t>;
static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}
static std::vector<Bytes> keys(const Bytes& data, size_t width)
{
    std::vector<Bytes> result;
    for (size_t i = 0; i < data.size(); i += width)
        result.emplace_back(data.begin() + i, data.begin() + i + width);
    std::sort(result.begin(), result.end());
    return result;
}
static Bytes roundtrip(Bytes input, size_t width)
{
    const auto expected = keys(input, width);
    Bytes scratch(msd_prefix::BUFFER_BYTES), output(msd_prefix::BUFFER_BYTES, 0xee);
    const size_t size = msd_prefix::encode(input.data(), scratch.data(), input.size() / width, width, output.data());
    require(size <= input.size() + 5 && size <= output.size(), "encoded capacity bound");
    output.resize(size);
    Bytes decoded(input.size());
    size_t pos = 0, records = 0;
    while (pos < size) {
        auto group = msd_prefix::decode_group(std::span(output).subspan(pos), width, std::span(decoded).subspan(records * width));
        pos += group.encoded_bytes;
        records += group.records;
    }
    require(records * width == input.size() && keys(decoded, width) == expected, "roundtrip multiset");
    return output;
}
static void malformed(Bytes input, size_t width, size_t output_bytes = msd_prefix::BUFFER_BYTES)
{
    Bytes output(output_bytes);
    bool rejected = false;
    try { (void)msd_prefix::decode_group(input, width, output); }
    catch (const std::exception&) { rejected = true; }
    require(rejected, "malformed group accepted");
}
static void histogram_boundaries()
{
    // Known single-group layouts catch missing lanes, a dropped tail, and
    // counts merged into the wrong bucket without relying on the decoder.
    for (size_t width : {2U, 7U, 10U, 14U, 32U}) {
        std::vector<size_t> counts;
        for (size_t n = 0; n <= 17; ++n) counts.push_back(n);
        for (size_t boundary : {32U, 64U, 256U, 1024U, 4096U})
            for (size_t n = boundary - 1; n <= boundary + 1; ++n) counts.push_back(n);
        const size_t q = msd_prefix::batch_records(width);
        if (width > 2)
            for (size_t n = q - 3; n <= q; ++n) counts.push_back(n);
        for (size_t n : counts) {
            Bytes data(n * width, 0xff);
            for (size_t i = 0; i < n; ++i) data[i * width + width - 1] = uint8_t(i);
            const auto encoded = roundtrip(data, width);
            Bytes expected;
            if (n) {
                const size_t prefix = n == 1 ? 0 : width - 1;
                expected.push_back(uint8_t(prefix));
                const uint32_t count = uint32_t(n);
                const auto* native = reinterpret_cast<const uint8_t*>(&count);
                expected.insert(expected.end(), native, native + sizeof(count));
                expected.insert(expected.end(), prefix, 0xff);
                for (size_t i = 0; i < n; ++i)
                    expected.insert(expected.end(), data.begin() + i * width + prefix,
                                    data.begin() + (i + 1) * width);
            }
            require(encoded == expected, "same-bucket lane/tail wire layout");
            // Each lane alternately sees the extremes, all 256 buckets,
            // a 90% hot bucket, or branches after a long shared prefix.
            // Also scatter Q-3..Q, so a missing lane/tail changes child counts.
            for (unsigned mode = 0; mode < (n > 4097 ? 1U : 4U); ++mode) {
                std::fill(data.begin(), data.end(), 0xff);
                for (size_t i = 0; i < n; ++i) {
                    data[i * width + width - 1] = uint8_t(i);
                    const size_t p = mode == 3 ? width - 2 : 0;
                    data[i * width + p] = mode == 0 ? (i % 2 ? 0xff : 0x00)
                        : mode == 1 ? uint8_t(i) : mode == 2 ? (i % 10 ? 0 : uint8_t(i))
                        : uint8_t(i % 4);
                }
                roundtrip(data, width);
            }
        }
    }
}
int main()
{
    try {
        static_assert(msd_prefix::BUFFER_BYTES == 2ULL << 20);
        static_assert(msd_prefix::FINAL_LOW_BUFFER_BYTES == 8ULL << 20);
        histogram_boundaries();
        std::mt19937 rng(4381);
        for (unsigned k = 1; k <= 128; ++k) {
            const size_t w = (k + 3) / 4;
            require(msd_prefix::batch_records(w) == (msd_prefix::BUFFER_BYTES - 5) / w, "batch capacity");
            roundtrip({}, w);
            Bytes single(w, 0x39);
            const auto encoded = roundtrip(single, w);
            require(encoded.size() == w + 5 && encoded[0] == 0, "single record must stop");
            uint32_t n = 0;
            std::memcpy(&n, encoded.data() + 1, 4);
            require(n == 1 && std::equal(single.begin(), single.end(), encoded.begin() + 5), "native packed layout");
            for (unsigned mode = 0; mode < 4; ++mode) {
                Bytes data(1301 * w);
                for (size_t i = 0; i < data.size(); ++i)
                    data[i] = mode == 0 ? 0x61 : mode == 1 && i % w < w / 2 ? 0x21 : uint8_t(rng() % (mode == 2 ? 2 : 256));
                roundtrip(data, w);
            }
        }
        // Strict cost boundary: B=2, P=0 => split only when N > 7.
        Bytes tied{0,1,0,2,0,3,0,4,1,5,1,6,1,7};
        const auto raw_group = roundtrip(tied, 2);
        Bytes raw_layout{0};
        uint32_t raw_count = 7;
        const auto* native_count = reinterpret_cast<const uint8_t*>(&raw_count);
        raw_layout.insert(raw_layout.end(), native_count, native_count + 4);
        raw_layout.insert(raw_layout.end(), tied.begin(), tied.end());
        require(raw_group == raw_layout, "equal-cost raw group or five-byte layout changed");
        tied.insert(tied.end(), {1,8});
        const auto split = roundtrip(tied, 2);
        require(split[0] == 1 && split.size() == 20, "profitable two-way split");
        for (size_t slice : {1U, 2U, 3U, 4U, 5U, 7U, 11U, 17U, 65536U}) {
            std::string stream_bytes(reinterpret_cast<const char*>(split.data()), split.size());
            stream_bytes += stream_bytes;
            std::istringstream stream(stream_bytes);
            auto actual = low_msd_test::read(stream, 2, slice);
            std::sort(actual.begin(), actual.end());
            Bytes twice = tied; twice.insert(twice.end(), tied.begin(), tied.end());
            require(actual == keys(twice, 2), "sliced concatenated groups");
        }
        Bytes identical(20 * 32, 0x42);
        const auto deep = roundtrip(identical, 32);
        require(deep[0] == 31 && deep.size() == 56, "single branch must reach W-1");
        uint32_t deep_count;
        std::memcpy(&deep_count, deep.data() + 1, 4);
        require(deep_count == 20 && std::all_of(deep.begin() + 5, deep.end(),
            [](uint8_t byte) { return byte == 0x42; }), "packed prefix/suffix layout changed");
        // Ten binary branching levels exercise recursive ping-pong scatter well
        // beyond the first few bytes. Every leaf retains repeated full keys.
        Bytes branches(32768 * 32, 0x23);
        for (size_t i = 0; i < 32768; ++i)
            for (size_t p = 0; p < 10; ++p) branches[i * 32 + p] = (i >> p) & 1;
        const auto branched = roundtrip(branches, 32);
        require(branched[0] == 31, "profitable deep multi-branch split stopped early");
        // Full and tail batches at widths including non-divisors of the slot.
        for (size_t w : {1U, 7U, 10U, 14U, 32U}) {
            Bytes data(msd_prefix::batch_records(w) * w);
            for (auto& b : data) b = uint8_t(rng());
            roundtrip(data, w);
            std::fill(data.begin(), data.end(), 0x51);
            roundtrip(data, w);
            data.resize(data.size() - w);
            roundtrip(data, w);
        }
        const auto valid = roundtrip(Bytes(40, 0x51), 4);
        for (size_t n = 0; n < valid.size(); ++n)
            malformed(Bytes(valid.begin(), valid.begin() + n), 4);
        auto bad = valid;
        bad[0] = 4; malformed(bad, 4);
        bad = valid;
        uint32_t n = 0; std::memcpy(bad.data() + 1, &n, 4); malformed(bad, 4);
        n = uint32_t(msd_prefix::batch_records(4) + 1);
        std::memcpy(bad.data() + 1, &n, 4); malformed(bad, 4);
        malformed(valid, 0); malformed(valid, 33); malformed(valid, 4, 39);
        std::cout << "MsdPrefixCodec tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
