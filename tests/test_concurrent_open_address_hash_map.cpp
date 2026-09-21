#define CONCURRENT_OPEN_ADDRESS_HASH_MAP_TESTING

#include "../src/ConcurrentOpenAddressHashMap.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace
{
constexpr uint32_t kWords = 2;
using Map = ConcurrentOpenAddressHashMap<kWords>;
using Kmer = kmer<kWords>;

struct KmerLess
{
    bool operator()(const Kmer& lhs, const Kmer& rhs) const noexcept
    {
        return lhs.data < rhs.data;
    }
};

using Counts = std::map<Kmer, uint64_t, KmerLess>;

struct Operation
{
    Kmer key{};
    uint32_t value = 1;
};

struct SegmentSnapshot
{
    int64_t size = 0;
    bool sealed = false;
    uint64_t capacity = 0;
    uint64_t entries = 0;
};

struct Snapshot
{
    Counts counts;
    std::vector<SegmentSnapshot> segments;
    bool duplicate_key = false;
};

class ScopedMapMaxCapacity
{
public:
    explicit ScopedMapMaxCapacity(uint64_t max_capacity)
        : previous_max_capacity_(concurrent_hash_map_max_capacity)
    {
        concurrent_hash_map_max_capacity = max_capacity;
    }

    ~ScopedMapMaxCapacity()
    {
        concurrent_hash_map_max_capacity = previous_max_capacity_;
    }

    ScopedMapMaxCapacity(const ScopedMapMaxCapacity&) = delete;
    ScopedMapMaxCapacity& operator=(const ScopedMapMaxCapacity&) = delete;

private:
    uint64_t previous_max_capacity_;
};

Kmer make_kmer(uint64_t value)
{
    Kmer key{};
    key.data[0] = value * 0x9e3779b97f4a7c15ULL + 0x85ebca77c2b2ae63ULL;
    key.data[1] = (value ^ 0xd1b54a32d192ed03ULL) * 0xbf58476d1ce4e5b9ULL;
    return key;
}

uint64_t segment_limit(uint64_t capacity)
{
    return static_cast<uint64_t>(static_cast<double>(capacity) * 0.8);
}

uint64_t next_segment_capacity(uint64_t capacity, uint64_t max_capacity)
{
    return std::min<uint64_t>(capacity * 2, max_capacity);
}

std::vector<uint64_t> expected_segment_capacities(uint64_t entry_count,
    uint64_t initial_capacity,
    uint64_t max_capacity)
{
    std::vector<uint64_t> capacities;
    uint64_t capacity = initial_capacity;
    capacities.push_back(capacity);

    while (entry_count > segment_limit(capacity))
    {
        entry_count -= segment_limit(capacity);
        capacity = next_segment_capacity(capacity, max_capacity);
        capacities.push_back(capacity);
    }
    return capacities;
}

void add_expected(Counts& expected, const Operation& op)
{
    uint64_t& count = expected[op.key];
    count = std::min<uint64_t>(
        std::numeric_limits<uint32_t>::max(), count + op.value);
}

Snapshot take_snapshot(const Map& map)
{
    Snapshot snapshot;
    map.debug_visit_segments(
        [&](uint64_t segment_index, int64_t size, bool sealed, uint64_t capacity)
        {
            if (snapshot.segments.size() <= segment_index)
            {
                snapshot.segments.resize(static_cast<size_t>(segment_index + 1));
            }
            snapshot.segments[static_cast<size_t>(segment_index)] = { size, sealed, capacity, 0 };
        });

    map.debug_visit_entries(
        [&](uint64_t segment_index, const Kmer& key, uint32_t count)
        {
            if (snapshot.segments.size() <= segment_index)
            {
                snapshot.segments.resize(static_cast<size_t>(segment_index + 1));
            }
            ++snapshot.segments[static_cast<size_t>(segment_index)].entries;
            const auto [it, inserted] = snapshot.counts.emplace(key, count);
            if (!inserted)
            {
                snapshot.duplicate_key = true;
                it->second += count;
            }
        });
    return snapshot;
}

bool validate_map(const Map& map,
    const Counts& expected,
    uint64_t initial_capacity,
    uint64_t max_capacity,
    const char* label)
{
    const Snapshot snapshot = take_snapshot(map);
    if (snapshot.segments.empty())
    {
        std::cerr << label << ": no root segment\n";
        return false;
    }
    if (snapshot.duplicate_key)
    {
        std::cerr << label << ": duplicate key found across slots or segments\n";
        return false;
    }
    if (map.debug_inserting_slot_count() != 0)
    {
        std::cerr << label << ": INSERTING slot remains after all threads joined\n";
        return false;
    }
    if (snapshot.counts != expected)
    {
        std::cerr << label << ": key/count snapshot differs from the reference map"
                  << " (expected keys=" << expected.size()
                  << ", observed keys=" << snapshot.counts.size() << ")\n";
        return false;
    }

    const std::vector<uint64_t> expected_capacities = expected_segment_capacities(
        static_cast<uint64_t>(expected.size()), initial_capacity, max_capacity);
    if (snapshot.segments.size() != expected_capacities.size())
    {
        std::cerr << label << ": segment count mismatch: expected "
                  << expected_capacities.size() << ", got " << snapshot.segments.size() << "\n";
        return false;
    }

    for (size_t i = 0; i < snapshot.segments.size(); ++i)
    {
        const SegmentSnapshot& segment = snapshot.segments[i];
        const uint64_t expected_capacity = expected_capacities[i];
        const uint64_t limit = segment_limit(expected_capacity);
        if (segment.capacity != expected_capacity)
        {
            std::cerr << label << ": segment " << i
                      << " capacity mismatch: expected " << expected_capacity
                      << ", got " << segment.capacity << "\n";
            return false;
        }
        if (segment.size < 0 || static_cast<uint64_t>(segment.size) != segment.entries)
        {
            std::cerr << label << ": segment " << i
                      << " size mismatch: size=" << segment.size
                      << ", entries=" << segment.entries << "\n";
            return false;
        }
        if (segment.entries > limit)
        {
            std::cerr << label << ": segment " << i << " exceeds load limit\n";
            return false;
        }
        const bool is_tail = i + 1 == snapshot.segments.size();
        if (!is_tail && !segment.sealed)
        {
            std::cerr << label << ": non-tail segment " << i << " is not sealed\n";
            return false;
        }
        if (!is_tail && segment.entries != limit)
        {
            std::cerr << label << ": non-tail segment " << i
                      << " entry count mismatch: expected " << limit
                      << ", got " << segment.entries << "\n";
            return false;
        }
        if (is_tail && segment.sealed)
        {
            std::cerr << label << ": tail segment is unexpectedly sealed\n";
            return false;
        }
    }
    return true;
}

std::vector<Kmer> find_keys_for_slot(uint64_t capacity,
    uint64_t slot,
    size_t count,
    uint64_t& cursor)
{
    std::vector<Kmer> keys;
    keys.reserve(count);
    while (keys.size() < count && cursor < 100'000'000ULL)
    {
        Kmer key = make_kmer(cursor++);
        if ((Map::hash_key(key) & (capacity - 1)) == slot)
        {
            keys.push_back(key);
        }
    }
    return keys;
}

bool insert_one(Map& map,
    const Operation& op,
    uint64_t& local_count,
    Counts& expected)
{
    map.increment(op.key, op.value, local_count);
    add_expected(expected, op);
    return true;
}

long long run_parallel(Map& map,
    ConcurrentMemoryPool& pool,
    const std::vector<std::vector<Operation>>& operations,
    std::vector<uint64_t>& local_counts)
{
    const size_t thread_count = operations.size();
    local_counts.assign(thread_count, 0);
    std::barrier start_barrier(static_cast<std::ptrdiff_t>(thread_count));
    std::atomic<long long> max_elapsed_ns{ 0 };
    std::vector<std::thread> threads;
    threads.reserve(thread_count);

    for (size_t thread_index = 0; thread_index < thread_count; ++thread_index)
    {
        threads.emplace_back([&, thread_index]
            {
                Map::set_memory_pool(&pool);
                start_barrier.arrive_and_wait();
                const auto start = std::chrono::steady_clock::now();
                for (const Operation& op : operations[thread_index])
                {
                    map.increment(op.key, op.value, local_counts[thread_index]);
                }
                const auto end = std::chrono::steady_clock::now();
                const long long elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
                long long observed = max_elapsed_ns.load(std::memory_order_relaxed);
                while (elapsed > observed &&
                    !max_elapsed_ns.compare_exchange_weak(observed,
                        elapsed,
                        std::memory_order_release,
                        std::memory_order_relaxed))
                {
                }
            });
    }
    for (std::thread& thread : threads)
    {
        thread.join();
    }
    return max_elapsed_ns.load(std::memory_order_acquire);
}

bool test_growth_boundaries_and_old_key_updates()
{
    constexpr uint64_t initial_capacity = 8;
    constexpr uint64_t max_capacity = 32;
    ScopedMapMaxCapacity scoped_max_capacity(max_capacity);
    Map map(initial_capacity);
    Counts expected;
    uint64_t local_count = 0;
    if (!validate_map(map, expected, initial_capacity, max_capacity, "empty"))
    {
        return false;
    }

    std::vector<Kmer> keys;
    keys.reserve(44);
    for (uint64_t i = 0; i < 44; ++i)
    {
        const Kmer key = make_kmer(100 + i);
        keys.push_back(key);
        insert_one(map, { key, 1 }, local_count, expected);

        const uint64_t inserted = i + 1;
        const char* label = nullptr;
        switch (inserted)
        {
        case 6: label = "growth boundary 6"; break;
        case 7: label = "growth boundary 7"; break;
        case 18: label = "growth boundary 18"; break;
        case 19: label = "growth boundary 19"; break;
        case 43: label = "growth boundary 43"; break;
        case 44: label = "growth boundary 44"; break;
        default: break;
        }

        if (label != nullptr &&
            !validate_map(map, expected, initial_capacity, max_capacity, label))
        {
            return false;
        }
    }
    if (local_count != expected.size())
    {
        std::cerr << "boundary: local_count mismatch\n";
        return false;
    }

    const uint64_t local_count_before_updates = local_count;
    const size_t segment_key_indices[] = { 0, 6, 18 };
    for (size_t key_index : segment_key_indices)
    {
        insert_one(map, { keys[key_index], 7 }, local_count, expected);
    }
    if (local_count != local_count_before_updates)
    {
        std::cerr << "old-key updates: local_count unexpectedly increased\n";
        return false;
    }
    return validate_map(map,
        expected,
        initial_capacity,
        max_capacity,
        "old-key updates across segments");
}

bool test_hot_key(ConcurrentMemoryPool& pool)
{
    constexpr uint64_t capacity = 8;
    ScopedMapMaxCapacity scoped_max_capacity(capacity);
    constexpr size_t thread_count = 8;
    constexpr size_t operations_per_thread = 4'000;
    const Kmer hot_key = make_kmer(10'000);
    std::vector<std::vector<Operation>> operations(thread_count);
    Counts expected;
    for (size_t t = 0; t < thread_count; ++t)
    {
        operations[t].assign(operations_per_thread, { hot_key, 1 });
        for (const Operation& op : operations[t])
        {
            add_expected(expected, op);
        }
    }

    Map map(capacity);
    std::vector<uint64_t> local_counts;
    run_parallel(map, pool, operations, local_counts);
    uint64_t unique_count = 0;
    for (uint64_t count : local_counts)
    {
        unique_count += count;
    }
    if (unique_count != 1)
    {
        std::cerr << "hot key: expected one unique insertion, got " << unique_count << "\n";
        return false;
    }
    return validate_map(map, expected, capacity, capacity, "hot key");
}

bool test_mixed_workload(ConcurrentMemoryPool& pool,
    uint64_t seed,
    size_t thread_count,
    size_t operations_per_thread,
    const char* label)
{
    constexpr uint64_t initial_capacity = 32;
    constexpr uint64_t max_capacity = 128;
    ScopedMapMaxCapacity scoped_max_capacity(max_capacity);
    std::vector<Kmer> hot_keys;
    for (uint64_t i = 0; i < 8; ++i)
    {
        hot_keys.push_back(make_kmer(50'000 + i));
    }

    std::vector<std::vector<Operation>> operations(thread_count);
    Counts expected;
    for (size_t t = 0; t < thread_count; ++t)
    {
        std::mt19937_64 rng(seed + t * 0x9e3779b97f4a7c15ULL);
        operations[t].reserve(operations_per_thread);
        for (size_t i = 0; i < operations_per_thread; ++i)
        {
            Operation op;
            if ((rng() % 100) < 45)
            {
                op.key = hot_keys[static_cast<size_t>(rng() % hot_keys.size())];
            }
            else
            {
                op.key = make_kmer(seed * 1'000'000ULL + t * operations_per_thread + i + 100'000ULL);
            }
            op.value = static_cast<uint32_t>(1 + rng() % 3);
            operations[t].push_back(op);
            add_expected(expected, op);
        }
    }

    Map map(initial_capacity);
    std::vector<uint64_t> local_counts;
    run_parallel(map, pool, operations, local_counts);
    uint64_t unique_count = 0;
    for (uint64_t count : local_counts)
    {
        unique_count += count;
    }
    if (unique_count != expected.size())
    {
        std::cerr << label << ": local_count mismatch: expected " << expected.size()
                  << ", got " << unique_count << "\n";
        return false;
    }
    return validate_map(map, expected, initial_capacity, max_capacity, label);
}

bool test_collision_and_fingerprint_boundaries()
{
    constexpr uint64_t initial_capacity = 16;
    constexpr uint64_t max_capacity = 64;
    ScopedMapMaxCapacity scoped_max_capacity(max_capacity);
    uint64_t cursor = 1;
    std::vector<Kmer> keys = find_keys_for_slot(
        initial_capacity, initial_capacity - 1, 20, cursor);
    if (keys.size() != 20)
    {
        std::cerr << "collision: could not generate enough colliding keys\n";
        return false;
    }

    Kmer fingerprint_80{};
    bool found_fingerprint = false;
    while (cursor < 100'000'000ULL)
    {
        Kmer candidate = make_kmer(cursor++);
        if (Map::fingerprint(Map::hash_key(candidate)) == 0x80U)
        {
            fingerprint_80 = candidate;
            found_fingerprint = true;
            break;
        }
    }
    if (!found_fingerprint)
    {
        std::cerr << "collision: could not generate fingerprint 0x80\n";
        return false;
    }

    const uint8_t collision_fingerprint = Map::fingerprint(Map::hash_key(keys.front()));
    Kmer same_fingerprint{};
    bool found_same_fingerprint = false;
    while (cursor < 100'000'000ULL)
    {
        Kmer candidate = make_kmer(cursor++);
        if (Map::fingerprint(Map::hash_key(candidate)) == collision_fingerprint &&
            !(candidate == keys.front()))
        {
            same_fingerprint = candidate;
            found_same_fingerprint = true;
            break;
        }
    }
    if (!found_same_fingerprint)
    {
        std::cerr << "collision: could not generate equal fingerprints\n";
        return false;
    }

    Map map(initial_capacity);
    Counts expected;
    uint64_t local_count = 0;
    for (const Kmer& key : keys)
    {
        insert_one(map, { key, 1 }, local_count, expected);
    }
    insert_one(map, { fingerprint_80, 7 }, local_count, expected);
    insert_one(map, { same_fingerprint, 11 }, local_count, expected);
    return validate_map(map,
        expected,
        initial_capacity,
        max_capacity,
        "collision and fingerprint");
}

enum class HookMode
{
    NONE,
    SEALED_FAST_PATH,
    POST_CAS_SEALED_CHECK
};

enum HookRole
{
    ROLE_NONE = 0,
    ROLE_READER = 1,
    ROLE_PUBLISHER = 2,
    ROLE_CANDIDATE = 3
};

thread_local int hook_role = ROLE_NONE;

struct RaceController
{
    HookMode mode = HookMode::NONE;
    Kmer target{};
    std::mutex mutex;
    std::condition_variable cv;
    bool reader_before_seen = false;
    bool release_reader_before = false;
    bool publisher_reserved_seen = false;
    bool release_publisher = false;
    bool reader_after_sealed_seen = false;
    bool release_reader_after_sealed = false;
    bool candidate_before_cas_seen = false;
    bool release_candidate = false;

    void on_event(Map::DebugEvent event, const Kmer& key)
    {
        if (!(key == target))
        {
            return;
        }
        std::unique_lock<std::mutex> lock(mutex);
        if (mode == HookMode::SEALED_FAST_PATH)
        {
            if (hook_role == ROLE_READER &&
                event == Map::DebugEvent::BEFORE_SEALED_CHECK &&
                !reader_before_seen)
            {
                reader_before_seen = true;
                cv.notify_all();
                cv.wait(lock, [&] { return release_reader_before; });
            }
            else if (hook_role == ROLE_PUBLISHER &&
                event == Map::DebugEvent::AFTER_SIZE_RESERVED_BEFORE_PUBLISH &&
                !publisher_reserved_seen)
            {
                publisher_reserved_seen = true;
                cv.notify_all();
                cv.wait(lock, [&] { return release_publisher; });
            }
            else if (hook_role == ROLE_READER &&
                event == Map::DebugEvent::AFTER_SEALED_OBSERVED &&
                !reader_after_sealed_seen)
            {
                reader_after_sealed_seen = true;
                cv.notify_all();
                cv.wait(lock, [&] { return release_reader_after_sealed; });
            }
        }
        else if (mode == HookMode::POST_CAS_SEALED_CHECK &&
            hook_role == ROLE_CANDIDATE &&
            event == Map::DebugEvent::AFTER_OPEN_CHECK_BEFORE_CTRL_CAS &&
            !candidate_before_cas_seen)
        {
            candidate_before_cas_seen = true;
            cv.notify_all();
            cv.wait(lock, [&] { return release_candidate; });
        }
    }

    bool wait_for(bool RaceController::*member)
    {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(5), [&] { return this->*member; });
    }

    void release_all()
    {
        std::lock_guard<std::mutex> lock(mutex);
        release_reader_before = true;
        release_publisher = true;
        release_reader_after_sealed = true;
        release_candidate = true;
        cv.notify_all();
    }
};

RaceController* active_race_controller = nullptr;

void race_hook(Map::DebugEvent event, const Map*, const Kmer& key)
{
    if (active_race_controller != nullptr)
    {
        active_race_controller->on_event(event, key);
    }
}

bool test_sealed_fast_path_race(ConcurrentMemoryPool& pool)
{
    constexpr uint64_t capacity = 8;
    ScopedMapMaxCapacity scoped_max_capacity(capacity);
    uint64_t cursor = 2'000'000;
    std::vector<Kmer> keys;
    for (uint64_t slot = 0; slot <= 6; ++slot)
    {
        std::vector<Kmer> generated = find_keys_for_slot(capacity, slot, 1, cursor);
        if (generated.empty())
        {
            std::cerr << "sealed race: key generation failed\n";
            return false;
        }
        keys.push_back(generated.front());
    }

    Map map(capacity);
    Counts expected;
    uint64_t setup_local_count = 0;
    for (size_t i = 0; i < 5; ++i)
    {
        insert_one(map, { keys[i], 1 }, setup_local_count, expected);
    }

    RaceController controller;
    controller.mode = HookMode::SEALED_FAST_PATH;
    controller.target = keys[5];
    active_race_controller = &controller;
    Map::set_debug_hook(&race_hook);

    uint64_t reader_local_count = 0;
    uint64_t publisher_local_count = 0;
    uint64_t overflow_local_count = 0;
    std::thread reader([&]
        {
            Map::set_memory_pool(&pool);
            hook_role = ROLE_READER;
            map.increment(keys[5], 1, reader_local_count);
        });

    if (!controller.wait_for(&RaceController::reader_before_seen))
    {
        controller.release_all();
        reader.join();
        Map::set_debug_hook(nullptr);
        active_race_controller = nullptr;
        std::cerr << "sealed race: reader did not reach pre-sealed hook\n";
        return false;
    }

    std::thread publisher([&]
        {
            Map::set_memory_pool(&pool);
            hook_role = ROLE_PUBLISHER;
            map.increment(keys[5], 1, publisher_local_count);
        });
    if (!controller.wait_for(&RaceController::publisher_reserved_seen))
    {
        controller.release_all();
        reader.join();
        publisher.join();
        Map::set_debug_hook(nullptr);
        active_race_controller = nullptr;
        std::cerr << "sealed race: publisher did not reserve the last slot\n";
        return false;
    }

    std::thread overflow([&]
        {
            Map::set_memory_pool(&pool);
            hook_role = ROLE_NONE;
            map.increment(keys[6], 1, overflow_local_count);
        });
    overflow.join();

    {
        std::lock_guard<std::mutex> lock(controller.mutex);
        controller.release_reader_before = true;
        controller.cv.notify_all();
    }
    if (!controller.wait_for(&RaceController::reader_after_sealed_seen))
    {
        controller.release_all();
        reader.join();
        publisher.join();
        Map::set_debug_hook(nullptr);
        active_race_controller = nullptr;
        std::cerr << "sealed race: reader did not observe sealed\n";
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(controller.mutex);
        controller.release_publisher = true;
        controller.cv.notify_all();
    }
    publisher.join();
    {
        std::lock_guard<std::mutex> lock(controller.mutex);
        controller.release_reader_after_sealed = true;
        controller.cv.notify_all();
    }
    reader.join();
    Map::set_debug_hook(nullptr);
    active_race_controller = nullptr;

    add_expected(expected, { keys[5], 1 });
    add_expected(expected, { keys[5], 1 });
    add_expected(expected, { keys[6], 1 });

    const uint64_t unique_count = setup_local_count + reader_local_count +
        publisher_local_count + overflow_local_count;
    if (unique_count != expected.size())
    {
        std::cerr << "sealed race: local_count mismatch\n";
        return false;
    }
    return validate_map(map, expected, capacity, capacity, "sealed fast-path race");
}

bool test_post_cas_sealed_race(ConcurrentMemoryPool& pool)
{
    constexpr uint64_t capacity = 8;
    ScopedMapMaxCapacity scoped_max_capacity(capacity);
    uint64_t cursor = 4'000'000;
    std::vector<Kmer> keys;
    for (uint64_t slot = 0; slot < capacity; ++slot)
    {
        std::vector<Kmer> generated = find_keys_for_slot(capacity, slot, 1, cursor);
        if (generated.empty())
        {
            std::cerr << "post-CAS race: key generation failed\n";
            return false;
        }
        keys.push_back(generated.front());
    }

    Map map(capacity);
    Counts expected;
    uint64_t setup_local_count = 0;
    for (size_t i = 0; i < 6; ++i)
    {
        insert_one(map, { keys[i], 1 }, setup_local_count, expected);
    }

    RaceController controller;
    controller.mode = HookMode::POST_CAS_SEALED_CHECK;
    controller.target = keys[6];
    active_race_controller = &controller;
    Map::set_debug_hook(&race_hook);

    uint64_t candidate_local_count = 0;
    std::thread candidate([&]
        {
            Map::set_memory_pool(&pool);
            hook_role = ROLE_CANDIDATE;
            map.increment(keys[6], 1, candidate_local_count);
        });
    if (!controller.wait_for(&RaceController::candidate_before_cas_seen))
    {
        controller.release_all();
        candidate.join();
        Map::set_debug_hook(nullptr);
        active_race_controller = nullptr;
        std::cerr << "post-CAS race: candidate did not reach hook\n";
        return false;
    }

    hook_role = ROLE_NONE;
    uint64_t overflow_local_count = 0;
    map.increment(keys[7], 1, overflow_local_count);
    {
        std::lock_guard<std::mutex> lock(controller.mutex);
        controller.release_candidate = true;
        controller.cv.notify_all();
    }
    candidate.join();
    Map::set_debug_hook(nullptr);
    active_race_controller = nullptr;

    add_expected(expected, { keys[6], 1 });
    add_expected(expected, { keys[7], 1 });
    const uint64_t unique_count = setup_local_count + candidate_local_count + overflow_local_count;
    if (unique_count != expected.size())
    {
        std::cerr << "post-CAS race: local_count mismatch\n";
        return false;
    }
    return validate_map(map, expected, capacity, capacity, "post-CAS sealed race");
}

bool run_correctness_suite(ConcurrentMemoryPool& pool)
{
    if (!test_growth_boundaries_and_old_key_updates()) return false;
    if (!test_hot_key(pool)) return false;
    if (!test_mixed_workload(pool, 0x1234ULL, 8, 2'000, "mixed workload")) return false;
    if (!test_collision_and_fingerprint_boundaries()) return false;
    if (!test_sealed_fast_path_race(pool)) return false;
    if (!test_post_cas_sealed_race(pool)) return false;

    constexpr uint64_t seeds[] = { 0x101ULL, 0x202ULL, 0x303ULL, 0x404ULL };
    constexpr size_t thread_counts[] = { 1, 2, 4, 8 };
    for (size_t threads : thread_counts)
    {
        for (uint64_t seed : seeds)
        {
            const std::string label = "stress threads=" + std::to_string(threads) +
                " seed=" + std::to_string(seed);
            if (!test_mixed_workload(pool, seed, threads, 1'000, label.c_str()))
            {
                return false;
            }
        }
    }
    return true;
}

struct BenchmarkConfig
{
    size_t threads = std::max(1U, std::thread::hardware_concurrency());
    size_t operations_per_thread = 100'000;
    uint64_t initial_capacity = 1024;
    uint64_t max_capacity = 1024;
    double duplicate_ratio = 0.75;
    size_t repetitions = 3;
};

bool parse_u64(const char* text, uint64_t& value)
{
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = static_cast<uint64_t>(parsed);
    return true;
}

bool parse_double(const char* text, double& value)
{
    if (text == nullptr || *text == '\0') return false;
    char* end = nullptr;
    value = std::strtod(text, &end);
    return end != text && *end == '\0';
}

bool parse_benchmark_config(int argc, char** argv, BenchmarkConfig& cfg)
{
    bool max_capacity_provided = false;
    for (int i = 2; i < argc; i += 2)
    {
        if (i + 1 >= argc)
        {
            return false;
        }
        uint64_t integer_value = 0;
        const std::string option = argv[i];
        if (option == "--threads" && parse_u64(argv[i + 1], integer_value))
        {
            cfg.threads = static_cast<size_t>(integer_value);
        }
        else if (option == "--ops" && parse_u64(argv[i + 1], integer_value))
        {
            cfg.operations_per_thread = static_cast<size_t>(integer_value);
        }
        else if (option == "--capacity" && parse_u64(argv[i + 1], integer_value))
        {
            cfg.initial_capacity = integer_value;
        }
        else if (option == "--max-capacity" && parse_u64(argv[i + 1], integer_value))
        {
            cfg.max_capacity = integer_value;
            max_capacity_provided = true;
        }
        else if (option == "--duplicate-ratio" && parse_double(argv[i + 1], cfg.duplicate_ratio))
        {
        }
        else if (option == "--repetitions" && parse_u64(argv[i + 1], integer_value))
        {
            cfg.repetitions = static_cast<size_t>(integer_value);
        }
        else
        {
            return false;
        }
    }
    if (!max_capacity_provided)
    {
        cfg.max_capacity = cfg.initial_capacity;
    }
    return cfg.threads > 0 && cfg.operations_per_thread > 0 && cfg.repetitions > 0 &&
        cfg.initial_capacity >= 8 &&
        (cfg.initial_capacity & (cfg.initial_capacity - 1)) == 0 &&
        cfg.max_capacity >= 8 &&
        (cfg.max_capacity & (cfg.max_capacity - 1)) == 0 &&
        cfg.initial_capacity <= cfg.max_capacity &&
        cfg.duplicate_ratio >= 0.0 && cfg.duplicate_ratio <= 1.0;
}

bool run_benchmark(ConcurrentMemoryPool& pool, const BenchmarkConfig& cfg)
{
    ScopedMapMaxCapacity scoped_max_capacity(cfg.max_capacity);
    std::vector<Kmer> hot_keys;
    for (uint64_t i = 0; i < 64; ++i)
    {
        hot_keys.push_back(make_kmer(80'000'000ULL + i));
    }

    std::vector<std::vector<Operation>> operations(cfg.threads);
    Counts expected;
    for (size_t t = 0; t < cfg.threads; ++t)
    {
        std::mt19937_64 rng(0xBADC0FFEEULL + t);
        operations[t].reserve(cfg.operations_per_thread);
        for (size_t i = 0; i < cfg.operations_per_thread; ++i)
        {
            Operation op;
            const double roll = static_cast<double>(rng()) /
                static_cast<double>(std::numeric_limits<uint64_t>::max());
            if (roll < cfg.duplicate_ratio)
            {
                op.key = hot_keys[static_cast<size_t>(rng() % hot_keys.size())];
            }
            else
            {
                op.key = make_kmer(100'000'000ULL + t * cfg.operations_per_thread + i);
            }
            operations[t].push_back(op);
            add_expected(expected, op);
        }
    }

    std::vector<double> throughputs;
    throughputs.reserve(cfg.repetitions);
    uint64_t final_segment_count = 0;

    std::vector<std::vector<Operation>> warmup_operations(cfg.threads);
    for (size_t t = 0; t < cfg.threads; ++t)
    {
        const size_t warmup_count = std::min<size_t>(1'000, operations[t].size());
        warmup_operations[t].assign(operations[t].begin(), operations[t].begin() + warmup_count);
    }
    {
        Map warmup_map(cfg.initial_capacity);
        std::vector<uint64_t> warmup_local_counts;
        run_parallel(warmup_map, pool, warmup_operations, warmup_local_counts);
    }

    for (size_t repetition = 0; repetition < cfg.repetitions; ++repetition)
    {
        Map map(cfg.initial_capacity);
        std::vector<uint64_t> local_counts;
        const long long elapsed_ns = run_parallel(map, pool, operations, local_counts);
        if (elapsed_ns <= 0 ||
            !validate_map(map,
                expected,
                cfg.initial_capacity,
                cfg.max_capacity,
                "benchmark validation"))
        {
            return false;
        }
        const double seconds = static_cast<double>(elapsed_ns) * 1e-9;
        const double total_operations = static_cast<double>(cfg.threads) *
            static_cast<double>(cfg.operations_per_thread);
        throughputs.push_back(total_operations / seconds);
        final_segment_count = take_snapshot(map).segments.size();
    }

    std::sort(throughputs.begin(), throughputs.end());
    const double median = throughputs[throughputs.size() / 2];
    const double best = throughputs.back();
    const double total_operations = static_cast<double>(cfg.threads) *
        static_cast<double>(cfg.operations_per_thread);
    std::cout << "threads=" << cfg.threads
              << " ops_per_thread=" << cfg.operations_per_thread
              << " total_ops=" << static_cast<uint64_t>(total_operations)
              << " duplicate_ratio=" << cfg.duplicate_ratio
              << " initial_capacity=" << cfg.initial_capacity
              << " max_capacity=" << cfg.max_capacity
              << " unique_keys=" << expected.size()
              << " segments=" << final_segment_count << "\n"
              << "median_mops=" << median / 1'000'000.0
              << " best_mops=" << best / 1'000'000.0
              << " median_ns_per_op=" << 1e9 / median << "\n";
    return true;
}
} // namespace

int main(int argc, char** argv)
{
    count_max = std::numeric_limits<uint32_t>::max();
    ConcurrentMemoryPool pool(512ULL * 1024ULL * 1024ULL);
    pool.init_arenas();
    Map::set_memory_pool(&pool);

    if (argc > 1 && std::string(argv[1]) == "--benchmark")
    {
        BenchmarkConfig cfg;
        if (!parse_benchmark_config(argc, argv, cfg))
        {
            std::cerr << "Usage: " << argv[0]
                      << " --benchmark [--threads N] [--ops N] [--capacity N]"
                      << " [--max-capacity N]"
                      << " [--duplicate-ratio R] [--repetitions N]\n";
            return 2;
        }
        return run_benchmark(pool, cfg) ? 0 : 1;
    }

    if (!run_correctness_suite(pool))
    {
        return 1;
    }
    std::cout << "ConcurrentOpenAddressHashMap tests passed\n";
    return 0;
}
