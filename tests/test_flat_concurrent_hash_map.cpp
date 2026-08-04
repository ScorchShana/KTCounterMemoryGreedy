#include "../tool/FlatConcurrentHashMap.h"
#include "../include/xxh3.h"

#include <barrier>
#include <bit>
#include <cstdint>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

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
