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

template<uint32_t N>
kmer<N> make_key(uint64_t value)
{
    kmer<N> result;
    for (uint32_t i = 0; i < N; ++i)
        result.data[i] = (value + i) * 0x9e3779b97f4a7c15ULL;
    return result;
}

template<uint32_t N>
using Expected = std::map<std::array<uint64_t, N>, uint32_t>;

// Catches missing/high-bit lanes, sign-bit-only occupancy checks, duplicate
// mirror-tail visits, wrong callback order/counts, and changed reference types.
template<uint32_t N>
void check(const CountingHashMap<N>& map, const Expected<N>& expected)
{
    Expected<N> actual;
    size_t visits = 0, previous_slot = 0;
    map.for_each([&](const kmer<N>& key, const auto& count) {
        static_assert(std::is_const_v<std::remove_reference_t<decltype(count)>>);
        const size_t slot = &key - map.keys_;
        require(slot < map.capacity(), "visited mirrored controls or invalid key");
        require(visits == 0 || slot > previous_slot, "callback order changed");
        previous_slot = slot;
        ++visits;
        require(actual.emplace(key.data, count).second, "duplicate key callback");
    });
    require(actual == expected, "keys or counts differ");
    require(visits == expected.size() && map.size() == expected.size(),
        "visit count differs");
}

template<uint32_t N>
void test_boundaries()
{
    CountingHashMap<N> map;
    Expected<N> expected;
    check<N>(map, expected);
    const size_t cap = map.capacity();
    const std::vector<size_t> slots{
        0, 1, 14, 15, 16, 30, 31, 32, 33, 62, 63, 64,
        cap - 33, cap - 32, cap - 17, cap - 16, cap - 2, cap - 1};
    uint64_t next_value = 1;
    bool has_low_fp = false, has_high_fp = false;
    for (size_t slot : slots) {
        kmer<N> key;
        for (;;) {
            key = make_key<N>(next_value++);
            const uint64_t hash = rapidhash(&key, sizeof(key));
            if ((hash & (cap - 1)) == slot) {
                const uint8_t fp = uint8_t(hash >> 56);
                has_low_fp |= fp != 0 && fp < 128;
                has_high_fp |= fp >= 128 || fp == 0;
                break;
            }
        }
        const unsigned repeats = unsigned(slot % 5 + 1);
        for (unsigned i = 0; i < repeats; ++i) map.increment(key);
        expected[key.data] = repeats;
        require(map.keys_[slot] == key, "boundary fixture landed in wrong slot");
        check<N>(map, expected);
    }
    require(has_low_fp && has_high_fp, "fixture needs both fingerprint signs");
    map.for_each([&](const auto& key, auto& count) {
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(count)>>);
        count += 7;
        expected[key.data] += 7;
    });
    check<N>(map, expected);

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
    check<N>(map, expected);
    for (uint64_t i = 1; i <= 70; ++i) {
        auto key = make_key<N>(i);
        map.increment(key);
        ++expected[key.data];
    }
    check<N>(map, expected);
}

template<uint32_t N>
void test_density_and_reuse()
{
    CountingHashMap<N> map;
    Expected<N> expected;
    const size_t limit = MAX_KMER_BLOCK_NUM * KMER_BLOCK_SIZE / sizeof(kmer<N>);
    for (size_t size : {size_t{0}, size_t{1}, size_t{8}, size_t{32},
                       size_t{64}, size_t{257}, limit}) {
        map.clear();
        expected.clear();
        for (size_t i = 1; i <= size; ++i) {
            auto key = make_key<N>(i);
            map.increment(key);
            ++expected[key.data];
            if (i % 3 == 0) {
                map.increment(key);
                ++expected[key.data];
            }
        }
        check<N>(map, expected);
    }
}
}

int main()
{
    try {
        test_boundaries<1>(); test_boundaries<2>();
        test_boundaries<3>(); test_boundaries<4>();
        test_density_and_reuse<1>(); test_density_and_reuse<2>();
        test_density_and_reuse<3>(); test_density_and_reuse<4>();
        std::cout << "CountingHashMap tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
