#include "../src/CountingHashMap.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

using Key = std::vector<unsigned char>;
Key make_key(uint64_t value)
{
    Key key(kmer_bytes());
    for (size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<unsigned char>(value);
        value = (value >> 8) ^ (value * 0x9e3779b97f4a7c15ULL);
    }
    return key;
}
using Expected = std::map<Key, uint32_t>;

// Catches missing/high-bit lanes, sign-bit-only occupancy checks, duplicate
// mirror-tail visits, wrong callback order/counts, and changed reference types.
void check(const CountingHashMap& map, const Expected& expected, size_t width = kmer_bytes())
{
    Expected actual;
    size_t visits = 0, previous_slot = 0;
    map.for_each([&](const unsigned char* key, const auto& count) {
        static_assert(std::is_const_v<std::remove_reference_t<decltype(count)>>);
        const size_t slot = (key - map.keys_.data()) / width;
        require(slot < map.capacity(), "visited mirrored controls or invalid key");
        require(visits == 0 || slot > previous_slot, "callback order changed");
        previous_slot = slot;
        ++visits;
        require(actual.emplace(Key(key, key + width), count).second, "duplicate key callback");
    });
    require(actual == expected, "keys or counts differ");
    require(visits == expected.size() && map.size() == expected.size(),
        "visit count differs");
}

void test_boundaries(uint32_t k)
{
    k_length = k;
    CountingHashMap map;
    Expected expected;
    check(map, expected);
    const size_t cap = map.capacity();
    const std::vector<size_t> slots{
        0, 1, 14, 15, 16, 30, 31, 32, 33, 62, 63, 64,
        cap - 33, cap - 32, cap - 17, cap - 16, cap - 2, cap - 1};
    uint64_t next_value = 1;
    bool has_low_fp = false, has_high_fp = false;
    for (size_t slot : slots) {
        Key key;
        for (;;) {
            key = make_key(next_value++);
            const uint64_t hash = rapidhash(key.data(), key.size());
            if ((hash & (cap - 1)) == slot) {
                const uint8_t fp = uint8_t(hash >> 56);
                has_low_fp |= fp != 0 && fp < 128;
                has_high_fp |= fp >= 128 || fp == 0;
                break;
            }
        }
        const unsigned repeats = unsigned(slot % 5 + 1);
        for (unsigned i = 0; i < repeats; ++i) map.increment(key.data());
        expected[key] = repeats;
        require(std::memcmp(map.keys_.data() + slot * kmer_bytes(), key.data(), kmer_bytes()) == 0, "boundary fixture landed in wrong slot");
        check(map, expected);
    }
    require(has_low_fp && has_high_fp, "fixture needs both fingerprint signs");
    map.for_each([&](const auto& key, auto& count) {
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(count)>>);
        count += 7;
        expected[Key(key, key + kmer_bytes())] += 7;
    });
    check(map, expected);

    // Both overloads must propagate callback exceptions.
    struct Stop {};
    unsigned called = 0;
    try {
        map.for_each([&](const auto&, auto&) {
            if (++called == 3) throw Stop{};
        });
        require(false, "mutable callback exception swallowed");
    } catch (const Stop&) { require(called == 3, "exception did not stop traversal"); }
    called = 0;
    try {
        std::as_const(map).for_each([&](const auto&, const auto&) {
            if (++called == 3) throw Stop{};
        });
        require(false, "const callback exception swallowed");
    } catch (const Stop&) { require(called == 3, "exception did not stop traversal"); }

    map.clear();
    expected.clear();
    check(map, expected);
    for (uint64_t i = 1; i <= 70; ++i) {
        auto key = make_key(i);
        map.increment(key.data());
        ++expected[key];
    }
    check(map, expected);
}

void test_density_and_reuse(uint32_t k, size_t width = 0)
{
    k_length = k;
    if (!width) width = kmer_bytes();
    CountingHashMap map(width);
    Expected expected;
    const size_t limit = MAX_KMER_BLOCK_NUM * (4092 / width);
    for (size_t size : {size_t{0}, size_t{1}, size_t{8}, size_t{32},
                       size_t{64}, size_t{257}, limit}) {
        map.clear();
        expected.clear();
        size_t inserted = 0;
        for (size_t i = 1; inserted < size; ++i) {
            auto key = make_key(i);
            key.resize(width);
            map.increment(key.data());
            ++expected[key];
            ++inserted;
            if (i % 3 == 0 && inserted < size) {
                map.increment(key.data());
                ++expected[key];
                ++inserted;
            }
        }
        check(map, expected, width);
    }
    // A valid Task can consist entirely of distinct records. Keep maximum-load
    // coverage separate from batches where duplicates consume the record budget.
    if (width >= 4) {
        map.clear(); expected.clear();
        for (size_t i = 0; i < limit; ++i) {
            Key key(width, 0);
            for (size_t byte = 0; byte < std::min(width, sizeof(i)); ++byte)
                key[byte] = i >> (8 * byte);
            map.increment(key.data()); expected.emplace(key, 1);
        }
        require(map.size() == limit, "maximum distinct Task lost entries");
        check(map, expected, width);
    }
}
}

void counting_hash_map_link_probe();

int main()
{
    try {
        counting_hash_map_link_probe();
        k_length = 1;
        CountingHashMap batch;
        static_assert(std::is_same_v<decltype(batch.counts_)::value_type, uint16_t>);
        const unsigned char key = 0;
        for (unsigned i = 0; i < 65472; ++i) batch.increment(&key);
        check(batch, {{Key{key}, 65472}});
        batch.clear();
        batch.increment(&key);
        check(batch, {{Key{key}, 1}});
        for (uint32_t k : {31, 32, 33, 63, 64, 65, 81, 127, 128}) {
            test_boundaries(k);
            test_density_and_reuse(k);
        }
        for (size_t width : {size_t{1}, size_t{2}, size_t{7}, size_t{19}}) test_density_and_reuse(81, width);
        std::cout << "CountingHashMap tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
