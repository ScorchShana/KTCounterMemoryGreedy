#include "../src/ConcurrentFirstBlockRing.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

struct ConcurrentMemoryPoolTestAccess
{
    static int nodes(const ConcurrentMemoryPool& pool) { return pool.num_arenas_; }
    static void page_size(ConcurrentMemoryPool& pool, size_t bytes) { pool.page_size_ = bytes; }
    static Arena& arena(ConcurrentMemoryPool& pool, int node) { return pool.arenas_[node]; }
};

struct ConcurrentFirstBlockRingTestAccess
{
    static size_t capacity(const ConcurrentFirstBlockRing& ring) { return ring.slot_count_; }
    static size_t bytes(const ConcurrentFirstBlockRing& ring) { return ring.allocated_bytes_; }
    static uint8_t* data(const ConcurrentFirstBlockRing& ring) { return ring.data_; }
    static uint64_t ticket(ConcurrentFirstBlockRing& ring) { return ring.head_.fetch_add(1, std::memory_order_relaxed); }
    static bool exchange(ConcurrentFirstBlockRing& ring, uint8_t*& block, uint64_t ticket)
    { return ring.exchange_block(block, ticket); }
};

namespace {
using Access = ConcurrentFirstBlockRingTestAccess;
size_t ring_bytes(size_t slots, uint32_t producers)
{
    const size_t metadata = align_up(sizeof(ConcurrentFirstBlockRing), CACHE_LINE_SIZE)
        + slots * sizeof(std::atomic<uint8_t*>);
    return align_up(metadata, PAGE_SIZE)
        + (slots + producers) * ConcurrentFirstBlockRing::BLOCK_BYTES;
}

void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

void expect_exit(const std::function<void()>& action)
{
    const pid_t child = fork();
    require(child >= 0, "fork failed");
    if (child == 0) {
        try { action(); } catch (const std::bad_alloc&) { _exit(255); }
        _exit(0);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child, "waitpid failed");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 255, "invalid request did not exit(-1)");
}

void check_failures()
{
    expect_exit([] { ConcurrentMemoryPool pool(2ULL << 20); pool.allocate_large_from_numa(1, 0); });
    for (const size_t bytes : {size_t{0}, std::numeric_limits<size_t>::max(), size_t{8ULL << 20}})
        expect_exit([=] { ConcurrentMemoryPool pool(2ULL << 20); pool.init_arenas(); pool.allocate_large_from_numa(bytes, 0); });
    for (int node : {-1, MAX_NUMA_NODES})
        expect_exit([=] { ConcurrentMemoryPool pool(2ULL << 20); pool.init_arenas(); pool.allocate_large_from_numa(1, node); });
    expect_exit([] {
        ConcurrentMemoryPool pool(2ULL << 20);
        pool.init_arenas();
        pool.allocate_large_from_numa(1, ConcurrentMemoryPoolTestAccess::nodes(pool));
    });
    for (auto [producers, budget] : std::vector<std::pair<uint32_t, size_t>>{
            {0, ring_bytes(1, 1)}, {1, 0}, {1, ring_bytes(1, 1) - 1}, {2, ring_bytes(1, 2) - 1},
            {UINT32_MAX, ring_bytes(1, 2)}, {1, std::numeric_limits<size_t>::max()}})
        expect_exit([=] { ConcurrentMemoryPool pool(2ULL << 20); pool.init_arenas();
            ConcurrentFirstBlockRing::create(&pool, 0, producers, budget); });
    // Exhaust the selected arena's bump region: the new allocator never reuses a free-list block.
    expect_exit([] {
        ConcurrentMemoryPool pool(2ULL << 20);
        pool.init_arenas();
        auto& arena = ConcurrentMemoryPoolTestAccess::arena(pool, 0);
        const size_t bytes = static_cast<char*>(arena.end_addr) - static_cast<char*>(arena.start_addr);
        pool.allocate_large_from_numa(bytes, 0);
        pool.allocate_large_from_numa(1, 0);
    });
}

void check_allocator()
{
    ConcurrentMemoryPool pool(32ULL << 20);
    pool.init_arenas();
    // Allocation uses fixed 4 KiB pages even if the mapping uses huge pages.
    ConcurrentMemoryPoolTestAccess::page_size(pool, HUGE_PAGE_SIZE);
    auto& tls = ConcurrentMemoryPool::get_thread_local_cache();
    require(tls.pool == nullptr, "test unexpectedly has TLS state");
    for (int node = 0; node < ConcurrentMemoryPoolTestAccess::nodes(pool); ++node) {
        auto& arena = ConcurrentMemoryPoolTestAccess::arena(pool, node);
        if (arena.block_count() < 100) continue;
        auto* a = static_cast<char*>(pool.allocate_large_from_numa(1, node));
        auto* b = static_cast<char*>(pool.allocate_large_from_numa(PAGE_SIZE + 1, node));
        auto* c = static_cast<char*>(pool.allocate_large_from_numa(PAGE_SIZE, node));
        require(reinterpret_cast<uintptr_t>(a) % PAGE_SIZE == 0 && b == a + PAGE_SIZE && c == b + 2 * PAGE_SIZE,
            "allocation did not round consumption to pages");
        require(arena.contains(a) && arena.contains(c), "allocation came from a different arena");
        constexpr size_t threads = 8, allocations = 8;
        std::vector<void*> results(threads * allocations);
        std::vector<std::thread> workers;
        for (size_t t = 0; t < threads; ++t)
            workers.emplace_back([&, t] {
                for (size_t i = 0; i < allocations; ++i)
                    results[t * allocations + i] = pool.allocate_large_from_numa(PAGE_SIZE + 1, node);
            });
        for (auto& worker : workers) worker.join();
        std::sort(results.begin(), results.end(), std::less<void*>{});
        for (size_t i = 0; i < results.size(); ++i) {
            require(arena.contains(results[i]) && reinterpret_cast<uintptr_t>(results[i]) % PAGE_SIZE == 0,
                "concurrent allocation has wrong arena/alignment");
            if (i) require(static_cast<char*>(results[i]) >= static_cast<char*>(results[i - 1]) + 2 * PAGE_SIZE,
                "concurrent allocations overlap");
        }
    }
    require(tls.pool == nullptr, "specified arena allocation touched TLS");
}

void fill(uint8_t* block, uint64_t id)
{
    for (size_t i = 0; i < ConcurrentFirstBlockRing::BLOCK_BYTES / sizeof(uint64_t); ++i) {
        const uint64_t value = id ^ (0x9e3779b97f4a7c15ULL * i);
        std::memcpy(block + i * sizeof(value), &value, sizeof(value));
    }
}

void record(const uint8_t* block, std::vector<std::atomic<uint32_t>>& seen)
{
    uint64_t id;
    std::memcpy(&id, block, sizeof(id));
    require(id < seen.size(), "published block contains invalid record");
    for (size_t i = 0; i < ConcurrentFirstBlockRing::BLOCK_BYTES / sizeof(uint64_t); ++i) {
        uint64_t value;
        std::memcpy(&value, block + i * sizeof(value), sizeof(value));
        require(value == (id ^ (0x9e3779b97f4a7c15ULL * i)), "block was torn or reused before consumption");
    }
    require(seen[id].fetch_add(1, std::memory_order_relaxed) == 0, "block was consumed twice");
}

void check_layout_and_serial()
{
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    // Includes the slot-metadata page boundary and budgets with sub-page remainders.
    const size_t metadata_boundary = (PAGE_SIZE - align_up(sizeof(ConcurrentFirstBlockRing), CACHE_LINE_SIZE))
        / sizeof(std::atomic<uint8_t*>);
    for (size_t slots : {size_t{1}, size_t{2}, size_t{3}, size_t{7},
            metadata_boundary - 1, metadata_boundary, metadata_boundary + 1}) {
        for (size_t extra : {size_t{0}, size_t{PAGE_SIZE - 1}}) {
            const size_t budget = ring_bytes(slots, 1) + extra;
            auto* ring = ConcurrentFirstBlockRing::create(&pool, 0, 1, budget);
            size_t expected = 0;
            for (size_t m = 1; m < budget / ConcurrentFirstBlockRing::BLOCK_BYTES; ++m) {
                const size_t metadata = align_up(sizeof(ConcurrentFirstBlockRing), CACHE_LINE_SIZE) + m * sizeof(std::atomic<uint8_t*>);
                const size_t total = align_up(metadata, PAGE_SIZE) + (m + 1) * ConcurrentFirstBlockRing::BLOCK_BYTES;
                if (total <= budget) expected = m;
            }
            require(Access::capacity(*ring) == expected && Access::bytes(*ring) <= budget,
                "ring did not select maximum capacity inside full allocation budget");
            require(reinterpret_cast<uintptr_t>(ring) % PAGE_SIZE == 0 &&
                reinterpret_cast<uintptr_t>(Access::data(*ring)) % PAGE_SIZE == 0, "ring data is not page aligned");
            uint64_t begin, end;
            require(!ring->claim_drain_range(begin, end), "empty ring drained a block");
        }
    }
    for (size_t published : {1, 2, 6}) {
        auto* ring = ConcurrentFirstBlockRing::create(&pool, 0, 3, ring_bytes(7, 3));
        require(ring->initial_block(1) == ring->initial_block(0) + ConcurrentFirstBlockRing::BLOCK_BYTES &&
            ring->initial_block(2) == ring->initial_block(1) + ConcurrentFirstBlockRing::BLOCK_BYTES,
            "producer blocks have the wrong stride");
        auto* block = ring->initial_block(2);
        std::vector<std::atomic<uint32_t>> seen(published);
        for (size_t i = 0; i < published; ++i) {
            fill(block, i);
            require(!ring->publish_full_block(block), "partially filled ring evicted a full block");
        }
        uint64_t begin, end;
        size_t drained = 0;
        while (ring->claim_drain_range(begin, end))
            for (auto slot = begin; slot < end; ++slot) { record(ring->drain_block(slot), seen); ++drained; }
        require(drained == published, "partial drain visited an empty slot or missed a full slot");
        for (auto& count : seen) require(count.load() == 1, "partial drain lost a block");
    }
    for (size_t slots : {1, 2, 3, 7}) {
        auto* ring = ConcurrentFirstBlockRing::create(&pool, 0, 1, ring_bytes(slots, 1));
        auto* block = ring->initial_block(0);
        std::set<uint8_t*> reserved;
        std::vector<std::atomic<uint32_t>> seen(slots * 5);
        for (size_t i = 0; i < seen.size(); ++i) {
            fill(block, i);
            const bool old = ring->publish_full_block(block);
            require(old == (i >= slots), "first fill incorrectly returned an old FIRST block");
            if (old) record(block, seen);
            else require(reserved.insert(block).second, "reserved empty block acquired twice");
        }
        uint64_t begin, end;
        size_t drained = 0;
        while (ring->claim_drain_range(begin, end))
            for (auto slot = begin; slot < end; ++slot) { record(ring->drain_block(slot), seen); ++drained; }
        require(drained == slots && reserved.size() == slots, "drain or initial fill omitted a slot");
        for (auto& count : seen) require(count.load() == 1, "serial ring lost a block");
    }
}

void check_delayed_ticket()
{
    ConcurrentMemoryPool pool(2ULL << 20);
    pool.init_arenas();
    auto* ring = ConcurrentFirstBlockRing::create(&pool, 0, 2, ring_bytes(1, 2));
    auto* delayed = ring->initial_block(0);
    auto* fast = ring->initial_block(1);
    fill(delayed, 0);
    const auto ticket = Access::ticket(*ring); // producer pauses before exchange
    std::vector<std::atomic<uint32_t>> seen(4);
    fill(fast, 1);
    require(!ring->publish_full_block(fast), "later ticket did not acquire the reserved empty block");
    auto* reserved = fast;
    fill(fast, 2);
    require(ring->publish_full_block(fast), "wrapping producer lost the previous full block");
    record(fast, seen);
    require(Access::exchange(*ring, delayed, ticket), "old ticket incorrectly assumed its slot was empty");
    require(delayed == reserved, "delayed exchange did not acquire the actual current block");
    record(delayed, seen);
    fill(fast, 3);
    require(ring->publish_full_block(fast), "slot became empty after first use");
    record(fast, seen);
    uint64_t begin, end;
    require(ring->claim_drain_range(begin, end) && begin == 0 && end == 1, "one-slot drain failed");
    record(ring->drain_block(0), seen);
    for (auto& count : seen) require(count.load() == 1, "out-of-order tickets lost a block");
}

void check_concurrent(size_t slots, uint32_t producers)
{
    ConcurrentMemoryPool pool(8ULL << 20);
    pool.init_arenas();
    auto* ring = ConcurrentFirstBlockRing::create(&pool, 0, producers, ring_bytes(slots, producers));
    constexpr size_t per_thread = 800;
    std::vector<std::atomic<uint32_t>> seen(producers * per_thread);
    std::vector<std::atomic<uint32_t>> drained(slots);
    std::vector<std::thread> workers;
    std::barrier finish(producers);
    for (uint32_t t = 0; t < producers; ++t)
        workers.emplace_back([&, t] {
            auto* block = ring->initial_block(t);
            for (size_t i = 0; i < per_thread; ++i) {
                fill(block, t * per_thread + i);
                if (i % 17 == 0) std::this_thread::yield();
                if (ring->publish_full_block(block)) record(block, seen);
            }
            finish.arrive_and_wait();
            uint64_t begin, end;
            while (ring->claim_drain_range(begin, end)) {
                require(end > begin && end - begin <= 4 && end <= slots, "invalid drain range");
                for (auto slot = begin; slot < end; ++slot) {
                    require(drained[slot].fetch_add(1) == 0, "overlapping drain claims");
                    record(ring->drain_block(slot), seen);
                }
            }
        });
    for (auto& worker : workers) worker.join();
    for (auto& count : seen) require(count.load() == 1, "concurrent ring lost a block");
    for (auto& count : drained) require(count.load() == 1, "drain missed a slot");
}
}

int main(int argc, char**)
{
    try {
        if (argc == 1) check_failures(); // TSan runs the threaded checks without fork/exit cases.
        check_allocator();
        check_layout_and_serial();
        check_delayed_ticket();
        for (size_t slots : {1, 2, 3, 7, 31}) check_concurrent(slots, 8);
        std::cout << "ConcurrentFirstBlockRing tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
