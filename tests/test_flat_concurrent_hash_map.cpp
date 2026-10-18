#include "../tool/FlatConcurrentHashMap.h"
#include "../include/xxh3.h"

#include <algorithm>
#include <barrier>
#include <bit>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <limits>
#include <stdexcept>
#include <array>
#include <unistd.h>

namespace
{
    constexpr uint32_t kWords = 2;
    using Kmer = kmer<kWords>;

    Kmer make_kmer(uint64_t value)
    {
        Kmer key{};
        key.reset();
        key.data[0] = value * 0x9e3779b97f4a7c15ULL + 0x85ebca77c2b2ae63ULL;
        key.data[1] = (value ^ 0xd1b54a32d192ed03ULL) * 0xbf58476d1ce4e5b9ULL;
        return key;
    }

    uint64_t test_hash(const Kmer& key)
    {
        return XXH3_64bits(&key, sizeof(key));
    }

    uint8_t test_fingerprint(const Kmer& key)
    {
        return static_cast<uint8_t>((test_hash(key) & 0x7FULL) | 0x80U);
    }

    uint64_t test_initial_slot(const Kmer& key, uint64_t capacity)
    {
        return (test_hash(key) >> 7) % capacity;
    }

    uint64_t test_round_up(uint64_t value, uint64_t alignment)
    {
        const uint64_t remainder = value % alignment;
        return remainder == 0 ? value : value + alignment - remainder;
    }

    bool find_key_with_fingerprint(uint8_t fingerprint, Kmer& out_key)
    {
        for (uint64_t i = 1; i < 1'000'000; ++i)
        {
            Kmer key = make_kmer(i);
            if (test_fingerprint(key) == fingerprint)
            {
                out_key = key;
                return true;
            }
        }
        return false;
    }

    bool find_keys_with_start_slot(uint64_t capacity, uint64_t start_slot, size_t count, std::vector<Kmer>& out_keys)
    {
        out_keys.clear();
        for (uint64_t i = 1; i < 5'000'000 && out_keys.size() < count; ++i)
        {
            Kmer key = make_kmer(i + 0x10000000ULL);
            if (test_initial_slot(key, capacity) == start_slot)
            {
                out_keys.push_back(key);
            }
        }
        return out_keys.size() == count;
    }

    bool expect_found(const FlatConcurrentHashMap<kWords>& map, const Kmer& key, uint32_t expected, const char* label)
    {
        uint32_t actual = 0;
        if (!map.find(key, actual))
        {
            std::cerr << label << " should be found\n";
            return false;
        }
        if (actual != expected)
        {
            std::cerr << label << " count mismatch: expected " << expected
                << ", got " << actual << "\n";
            return false;
        }
        if (!map.contains(key))
        {
            std::cerr << label << " contains() should be true\n";
            return false;
        }

        const auto lookup = map.prepare_lookup(key);
        map.prefetch(lookup);
        uint32_t prepared_actual = 0;
        if (!map.find_prepared(key, lookup, prepared_actual))
        {
            std::cerr << label << " should be found with prepared lookup\n";
            return false;
        }
        if (prepared_actual != expected)
        {
            std::cerr << label << " prepared count mismatch: expected " << expected
                << ", got " << prepared_actual << "\n";
            return false;
        }
        return true;
    }

    bool expect_not_found(const FlatConcurrentHashMap<kWords>& map, const Kmer& key, const char* label)
    {
        uint32_t ignored = 0;
        if (map.find(key, ignored))
        {
            std::cerr << label << " should not be found\n";
            return false;
        }

        const auto lookup = map.prepare_lookup(key);
        map.prefetch(lookup);
        if (map.find_prepared(key, lookup, ignored))
        {
            std::cerr << label << " should not be found with prepared lookup\n";
            return false;
        }
        return true;
    }

    bool test_single_thread_insert_find()
    {
        constexpr uint64_t record_count = 256;
        FlatConcurrentHashMap<kWords> map(record_count);
        std::vector<Kmer> keys;
        keys.reserve(record_count);

        for (uint64_t i = 0; i < record_count; ++i)
        {
            Kmer key = make_kmer(i + 1);
            keys.push_back(key);
            if (!map.insert_unique(key, static_cast<uint32_t>(i + 10)))
            {
                std::cerr << "single-thread insert failed at " << i << "\n";
                return false;
            }
        }

        map.seal();

        for (uint64_t i = 0; i < record_count; ++i)
        {
            if (!expect_found(map, keys[static_cast<size_t>(i)], static_cast<uint32_t>(i + 10), "single-thread key"))
            {
                return false;
            }
        }

        return expect_not_found(map, make_kmer(0xfffffff0ULL), "missing key");
    }

    bool test_concurrent_insert()
    {
        constexpr uint64_t thread_count = 8;
        constexpr uint64_t keys_per_thread = 256;
        constexpr uint64_t total_keys = thread_count * keys_per_thread;

        FlatConcurrentHashMap<kWords> map(total_keys, thread_count);
        std::vector<Kmer> keys;
        keys.reserve(total_keys);
        for (uint64_t i = 0; i < total_keys; ++i)
        {
            keys.push_back(make_kmer(i + 10'000));
        }

        std::barrier sync_point(static_cast<std::ptrdiff_t>(thread_count));
        std::vector<std::thread> threads;
        threads.reserve(thread_count);
        std::atomic<bool> ok{ true };

        for (uint64_t thread_index = 0; thread_index < thread_count; ++thread_index)
        {
            threads.emplace_back([&, thread_index]()
                {
                    sync_point.arrive_and_wait();
                    const uint64_t begin = thread_index * keys_per_thread;
                    const uint64_t end = begin + keys_per_thread;
                    for (uint64_t i = begin; i < end; ++i)
                    {
                        if (!map.insert_unique(keys[static_cast<size_t>(i)], static_cast<uint32_t>(i + 1)))
                        {
                            ok.store(false, std::memory_order_relaxed);
                        }
                    }
                });
        }

        for (auto& thread : threads)
        {
            thread.join();
        }

        if (!ok.load(std::memory_order_relaxed))
        {
            std::cerr << "concurrent insert reported failure\n";
            return false;
        }
        map.seal();

        for (uint64_t i = 0; i < total_keys; i += 97)
        {
            if (!expect_found(map, keys[static_cast<size_t>(i)], static_cast<uint32_t>(i + 1), "concurrent key"))
            {
                return false;
            }
        }
        return true;
    }

    bool test_fingerprint_0x80()
    {
        Kmer key{};
        if (!find_key_with_fingerprint(0x80U, key))
        {
            std::cerr << "failed to find key with fingerprint 0x80\n";
            return false;
        }

        FlatConcurrentHashMap<kWords> map(1);
        if (!map.insert_unique(key, 12345))
        {
            std::cerr << "fingerprint 0x80 insert failed\n";
            return false;
        }
        map.seal();
        return expect_found(map, key, 12345, "fingerprint 0x80 key");
    }

    bool test_tail_mirror_wraparound()
    {
        FlatConcurrentHashMap<kWords> map(1);
        const uint64_t capacity = map.capacity();
        const uint64_t start_slot = capacity - 2;

        std::vector<Kmer> keys;
        if (!find_keys_with_start_slot(capacity, start_slot, 4, keys))
        {
            std::cerr << "failed to find tail-start keys\n";
            return false;
        }

        for (uint32_t i = 0; i < keys.size(); ++i)
        {
            if (!map.insert_unique(keys[i], 700U + i))
            {
                std::cerr << "tail mirror insert failed at " << i << "\n";
                return false;
            }
        }

        map.seal();

        if (!expect_found(map, keys[2], 702, "wrapped slot key 0"))
        {
            return false;
        }
        if (!expect_found(map, keys[3], 703, "wrapped slot key 1"))
        {
            return false;
        }
        return true;
    }

    bool test_non_power_of_two_capacity()
    {
        FlatConcurrentHashMap<kWords> map(60);
        if (std::has_single_bit(map.capacity()))
        {
            std::cerr << "capacity should not be forced to a power of two, got "
                << map.capacity() << "\n";
            return false;
        }

        std::vector<Kmer> keys;
        keys.reserve(60);
        for (uint64_t i = 0; i < 60; ++i)
        {
            Kmer key = make_kmer(i + 100'000);
            keys.push_back(key);
            if (!map.insert_unique(key, static_cast<uint32_t>(i + 5000)))
            {
                std::cerr << "non-power capacity insert failed\n";
                return false;
            }
        }

        map.seal();

        for (uint64_t i = 0; i < keys.size(); ++i)
        {
            if (!expect_found(map, keys[static_cast<size_t>(i)], static_cast<uint32_t>(i + 5000), "non-power key"))
            {
                return false;
            }
        }
        return true;
    }

    bool test_full_table_failure()
    {
        FlatConcurrentHashMap<kWords> map(1);
        const uint64_t capacity = map.capacity();

        for (uint64_t i = 0; i < capacity; ++i)
        {
            if (!map.insert_unique(make_kmer(i + 700'000), static_cast<uint32_t>(i)))
            {
                std::cerr << "full-table setup insert failed at " << i << "\n";
                return false;
            }
        }

        if (map.insert_unique(make_kmer(900'000), 1))
        {
            std::cerr << "insert should fail when table is full\n";
            return false;
        }

        map.seal();
        return true;
    }

    template <uint32_t N>
    kmer<N> make_compact_key(uint32_t k, uint64_t id)
    {
        kmer<N> key{};
        for (uint32_t base = 0; base < k; ++base)
        {
            uint64_t digits = id ^ ((base / 32) * 0x9e3779b97f4a7c15ULL);
            key.data[base / 32] |= ((digits >> (2 * (base % 32))) & 3) << (62 - 2 * (base % 32));
        }
        return key;
    }

    // Catches lost tail bytes, incorrect slot strides, and broken reconstruction.
    template <uint32_t N>
    bool test_compact_keys(uint32_t k)
    {
        using Map = FlatConcurrentHashMap<N>;
        if constexpr (!requires { Map(1, 1, k); Map::required_mmap_bytes(1, k); })
        {
            std::cerr << "compact key layout interface missing\n";
            return false;
        }
        else
        {
            const uint32_t records = k == 1 ? 2 : 128;
            Map map(records, 4, k);
            std::vector<kmer<N>> keys;
            for (uint32_t i = 0; i < records; ++i) keys.push_back(make_compact_key<N>(k, i));
            std::atomic<bool> ok{true};
            std::vector<std::thread> threads;
            for (uint32_t t = 0; t < 4; ++t)
                threads.emplace_back([&, t] {
                    for (uint32_t i = t; i < records; i += 4)
                        if (!map.insert_unique(keys[i], i + 1)) ok.store(false);
                });
            for (auto& thread : threads) thread.join();
            map.seal();
            threads.clear();
            for (uint32_t t = 0; t < 4; ++t)
                threads.emplace_back([&] {
                    for (uint32_t i = 0; i < records; ++i)
                    {
                        uint32_t count = 0;
                        uint64_t slot = 0;
                        auto lookup = map.prepare_lookup(keys[i]);
                        map.prefetch(lookup);
                        if (!map.find_prepared(keys[i], lookup, count) || count != i + 1 ||
                            !map.find_prepared_slot(keys[i], lookup, count, slot) || count != i + 1 ||
                            *map.count_at(slot) != count) ok.store(false);
                    }
                });
            for (auto& thread : threads) thread.join();
            std::vector<bool> seen(records, false);
            map.for_each_entry([&](const kmer<N>& key, uint32_t count) {
                if (!count || count > records || seen[count - 1] || !(key == keys[count - 1]))
                    ok.store(false);
                else seen[count - 1] = true;
            });
            uint32_t ignored = 0;
            if (map.find(make_compact_key<N>(k, records), ignored) ||
                map.insert_unique(keys[0], 1)) ok.store(false);
            if (!ok.load() || std::find(seen.begin(), seen.end(), false) != seen.end())
            {
                std::cerr << "compact key lookup/iteration failed at k=" << k << '\n';
                return false;
            }

            // Enough entries to expose savings despite the 2 MiB mapping granularity.
            constexpr uint64_t expected_unique = 200000;
            uint64_t capacity = test_round_up(307693, Map::GROUP_SIZE);
            uint64_t key_bytes = (k + 3) / 4;
            uint64_t expected = test_round_up(capacity + Map::GROUP_SIZE - 1, 2097152) +
                test_round_up(capacity * key_bytes, 2097152) + test_round_up(capacity * 4, 2097152);
            if (Map::required_mmap_bytes(expected_unique, k) != expected)
            {
                std::cerr << "compact map budget mismatch at k=" << k << '\n';
                return false;
            }
            return true;
        }
    }

    // Packed and legacy interfaces must hash exactly the effective packed bytes.
    template <uint32_t N>
    bool test_packed_interfaces(uint32_t k)
    {
        using Map = FlatConcurrentHashMap<N>;
        const uint32_t bytes = (k + 3) / 4;
        const uint32_t full = bytes & ~7U;
        const uint32_t tail = bytes & 7U;
        Map map(24, 2, k);
        std::array<kmer<N>, 24> keys;
        std::array<std::array<char, sizeof(kmer<N>)>, 24> packed{};
        const uint32_t records = k == 1 ? 2 : k == 2 ? 12 : keys.size();
        for (uint32_t i = 0; i < records; ++i)
        {
            keys[i] = make_compact_key<N>(k, i);
            const char* raw = reinterpret_cast<const char*>(&keys[i]);
            std::memcpy(packed[i].data(), raw, full);
            if (tail) std::memcpy(packed[i].data() + full, raw + full + 8 - tail, tail);
            auto old_lookup = map.prepare_lookup(keys[i]);
            auto new_lookup = map.prepare_lookup(packed[i].data());
            const uint64_t hash = XXH3_64bits(packed[i].data(), bytes);
            if (new_lookup.base != (hash >> 7) % map.capacity() ||
                new_lookup.fp != static_cast<uint8_t>((hash & 0x7f) | 0x80) ||
                old_lookup.base != new_lookup.base || old_lookup.fp != new_lookup.fp)
            {
                std::cerr << "packed-byte hash mismatch at k=" << k << '\n';
                return false;
            }
            if (!(i % 2 ? map.insert_unique(keys[i], i + 10) :
                          map.insert_unique(packed[i].data(), i + 10))) return false;
        }
        map.seal();
        for (uint32_t i = 0; i < records; ++i)
        {
            uint32_t count = 0;
            auto lookup = map.prepare_lookup(packed[i].data());
            uint64_t slot = 0, legacy_slot = 0;
            if (!map.find_prepared(packed[i].data(), lookup, count) || count != i + 10 ||
                !map.find_prepared_slot(packed[i].data(), lookup, count, slot) || count != i + 10 ||
                !map.find_prepared_slot(keys[i], lookup, count, legacy_slot) || legacy_slot != slot ||
                *map.count_at(slot) != i + 10 || !map.find(keys[i], count) || count != i + 10) return false;
        }
        // A packed record ends immediately before an inaccessible page. Neither
        // hashing nor memcmp may load the following count/record/page.
        const size_t page = static_cast<size_t>(::sysconf(_SC_PAGESIZE));
        void* memory = ::mmap(nullptr, page * 2, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (memory == MAP_FAILED) return false;
        if (::mprotect(static_cast<char*>(memory) + page, page, PROT_NONE) != 0)
        {
            ::munmap(memory, page * 2);
            return false;
        }
        char* end_key = static_cast<char*>(memory) + page - bytes;
        std::memcpy(end_key, packed[0].data(), bytes);
        uint32_t count = 0;
        auto lookup = map.prepare_lookup(end_key);
        bool ok = map.find_prepared(end_key, lookup, count) && count == 10 &&
                  !map.insert_unique(end_key, 1);
        auto missing = make_compact_key<N>(k, records);
        if (k > 1)
        {
            std::memcpy(end_key, &missing, full);
            if (tail) std::memcpy(end_key + full,
                reinterpret_cast<const char*>(&missing) + full + 8 - tail, tail);
            ok = ok && !map.find_prepared(end_key, map.prepare_lookup(end_key), count);
        }
        ::munmap(memory, page * 2);
        if (!ok) std::cerr << "packed interface failure at k=" << k << '\n';
        return ok;
    }

    template <typename Map = FlatConcurrentHashMap<2>>
    bool test_compact_tail_collision()
    {
        if constexpr (!requires { Map(1, 1, 41); }) return false;
        else
        {
            Map map(1, 1, 41);
            std::vector<Kmer> keys;
            for (uint64_t tail = 1; tail < (1U << 18) && keys.size() < 4; ++tail)
            {
                Kmer key{};
                key.data[0] = 0x123456789abcdef0ULL;
                key.data[1] = tail << 46;
                const auto lookup = map.prepare_lookup(key);
                if (lookup.fp == 0x80 && lookup.base == map.capacity() - 2) keys.push_back(key);
            }
            if (keys.size() != 4) return false;
            std::array<std::array<char, 11>, 4> packed{};
            for (uint32_t i = 0; i < keys.size(); ++i)
            {
                std::memcpy(packed[i].data(), &keys[i], 8);
                std::memcpy(packed[i].data() + 8,
                    reinterpret_cast<const char*>(&keys[i]) + 13, 3);
                if (i < 3 && !map.insert_unique(packed[i].data(), 20 + i)) return false;
            }
            map.seal();
            for (uint32_t i = 0; i < 3; ++i)
            {
                uint32_t count = 0;
                uint64_t slot = 0;
                if (!map.find_prepared_slot(keys[i], map.prepare_lookup(keys[i]), count, slot) ||
                    count != 20 + i || slot != (map.capacity() - 2 + i) % map.capacity()) return false;
                *map.mutable_count_at(slot) = 50 + i;
                if (!map.find(keys[i], count) || count != 50 + i ||
                    !map.find_prepared(packed[i].data(), map.prepare_lookup(packed[i].data()), count) ||
                    count != 50 + i) return false;
            }
            uint32_t count = 0;
            if (map.find(keys[3], count) ||
                map.find_prepared(packed[3].data(), map.prepare_lookup(packed[3].data()), count))
            {
                std::cerr << "different compact tail falsely matched\n";
                return false;
            }
            return true;
        }
    }

    template <typename Map = FlatConcurrentHashMap<2>>
    bool test_compact_layout_errors()
    {
        if constexpr (!requires { Map(1, 1, 41); Map::required_mmap_bytes(1, 41); }) return false;
        else
        {
            for (uint32_t k : {0U, 65U, 129U})
            {
                bool rejected = false;
                try { Map map(1, 1, k); } catch (const std::exception&) { rejected = true; }
                if (!rejected) return false;
                rejected = false;
                try { (void)Map::required_mmap_bytes(1, k); } catch (const std::exception&) { rejected = true; }
                if (!rejected) return false;
            }
            for (auto [records, k] : {std::pair{std::numeric_limits<uint64_t>::max(), 41U},
                                     std::pair{std::numeric_limits<uint64_t>::max() / 4, 41U},
                                     std::pair{std::numeric_limits<uint64_t>::max() / 8, 1U}})
            {
                bool rejected = false;
                try { (void)Map::required_mmap_bytes(records, k); }
                catch (const std::exception&) { rejected = true; }
                if (!rejected) { std::cerr << "oversized compact map accepted\n"; return false; }
            }
            return true;
        }
    }

    bool test_required_mmap_bytes()
    {
        using Map = FlatConcurrentHashMap<kWords>;
        constexpr uint64_t huge_page_bytes = 2ULL * 1024ULL * 1024ULL;
        constexpr uint64_t samples[] = { 0, 1, 60, 256, 2048 };

        for (const uint64_t expected_unique : samples)
        {
            const uint64_t mmap_bytes = Map::required_mmap_bytes(expected_unique);
            if (mmap_bytes == 0)
            {
                std::cerr << "required_mmap_bytes returned zero for "
                    << expected_unique << "\n";
                return false;
            }
            if (mmap_bytes % huge_page_bytes != 0)
            {
                std::cerr << "required_mmap_bytes is not 2MB aligned for "
                    << expected_unique << ": " << mmap_bytes << "\n";
                return false;
            }

            const Map map(expected_unique);
            const uint64_t capacity = map.capacity();
            const uint64_t ctrl_payload_bytes = capacity + Map::GROUP_SIZE - 1;
            const uint64_t kmer_payload_bytes = capacity * sizeof(kmer<kWords>);
            const uint64_t count_payload_bytes = capacity * sizeof(uint32_t);
            const uint64_t payload_bytes = ctrl_payload_bytes + kmer_payload_bytes + count_payload_bytes;
            const uint64_t expected_mmap_bytes =
                test_round_up(ctrl_payload_bytes, huge_page_bytes) +
                test_round_up(kmer_payload_bytes, huge_page_bytes) +
                test_round_up(count_payload_bytes, huge_page_bytes);

            if (mmap_bytes < payload_bytes)
            {
                std::cerr << "required_mmap_bytes does not cover payload for "
                    << expected_unique << "\n";
                return false;
            }
            if (mmap_bytes != expected_mmap_bytes)
            {
                std::cerr << "required_mmap_bytes mismatch for " << expected_unique
                    << ": expected " << expected_mmap_bytes
                    << ", got " << mmap_bytes << "\n";
                return false;
            }
        }

        return true;
    }
}

int main()
{
    for (uint32_t k = 1; k <= 128; ++k)
    {
        bool ok = k <= 32 ? test_packed_interfaces<1>(k) :
                  k <= 64 ? test_packed_interfaces<2>(k) :
                            test_packed_interfaces<4>(k);
        if (!ok) return 1;
    }
    for (uint32_t k : {1U, 4U, 5U, 9U, 13U, 17U, 21U, 25U, 29U, 31U, 32U})
        if (!test_compact_keys<1>(k)) return 1;
    for (uint32_t k : {33U, 37U, 41U, 45U, 49U, 53U, 57U, 63U, 64U})
        if (!test_compact_keys<2>(k)) return 1;
    for (uint32_t k : {65U, 81U, 95U, 96U, 97U, 127U, 128U})
        if (!test_compact_keys<4>(k)) return 1;
    if (!test_compact_tail_collision() || !test_compact_layout_errors()) return 1;
    if (!test_required_mmap_bytes())
    {
        return 1;
    }
    if (!test_single_thread_insert_find())
    {
        return 1;
    }
    if (!test_concurrent_insert())
    {
        return 1;
    }
    if (!test_fingerprint_0x80())
    {
        return 1;
    }
    if (!test_tail_mirror_wraparound())
    {
        return 1;
    }
    if (!test_non_power_of_two_capacity())
    {
        return 1;
    }
    if (!test_full_table_failure())
    {
        return 1;
    }

    std::cout << "FlatConcurrentHashMap tests passed\n";
    return 0;
}
