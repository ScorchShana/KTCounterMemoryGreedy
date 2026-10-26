#include "first_pool_fixture.h"
#include "../src/ConcurrentMemoryPool.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

struct ConcurrentMemoryPoolTestAccess
{
    static void mapped_arena(ConcurrentMemoryPool& pool)
    {
        pool.num_arenas_ = 1;
        Arena& arena = pool.arenas_[0];
        arena.start_addr = pool.memory_start_;
        arena.initialize(pool.memory_size_);
        arena.end_addr = static_cast<char*>(arena.start_addr) + arena.capacity;
        arena.high = HUGE_PAGE_SIZE;
        arena.low = BLOCK_SIZE;
        arena.segment_capacity = arena.capacity / FIRST_SEGMENT_SIZE;
        // Address-indexed descriptors use 64 bytes per 2 MiB (512 MiB here).
        arena.segments = std::make_unique<FirstSegmentDescriptor[]>(arena.segment_capacity);
        for (size_t i = 0; i < arena.segment_capacity; ++i)
            arena.segments[i].data = static_cast<char*>(arena.start_addr) + i * FIRST_SEGMENT_SIZE;
        pool.arenas_initialized_.store(true, std::memory_order_release);
        madvise(pool.memory_start_, pool.memory_size_, MADV_NOHUGEPAGE);
    }
};

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

int main()
{
    if (!std::getenv("TREE_TEST_LARGE_MAPPING")) {
        std::cout << "Skipped: set TREE_TEST_LARGE_MAPPING=1 for a 16 TiB virtual mapping\n";
        return 77;
    }
    try {
        k_length = 33;
        ConcurrentMemoryPool pool(16ULL << 40);
        ConcurrentMemoryPoolTestAccess::mapped_arena(pool);
        auto* base = static_cast<char*>(pool.allocate_large_from_numa((8ULL << 40) - BLOCK_SIZE, 0));
        auto* below = static_cast<char*>(pool.allocate_large_from_numa(BLOCK_SIZE, 0));
        auto* above = static_cast<char*>(pool.allocate_large_from_numa(BLOCK_SIZE, 0));
        require(below == base + (8ULL << 40) - BLOCK_SIZE && above == base + (8ULL << 40),
            "mapped allocation truncated at 8 TiB");
        *static_cast<volatile char*>(below) = 17;
        *static_cast<volatile char*>(above) = 29;
        ConcurrentMemoryPool::FirstSegmentView view;
        const size_t gap = pool.gap_bytes(0);
        std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 41);
        require(cache_full_first(pool, 0, records.data()), "mapped FIRST publication failed");
        pool.set_spill_service(true);
        pool.request_first_drain();
        const auto status = pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Spill, view);
        require(status == ConcurrentMemoryPool::FirstClaimStatus::Data && pool.gap_bytes(0) == gap - FIRST_SEGMENT_SIZE, "mapped FIRST allocation failed");
        require(view.records[0] == 41, "mapped FIRST payload changed");
        require(pool.complete_first(view), "mapped FIRST completion failed");
        require(pool.gap_bytes(0) == gap, "mapped FIRST reclamation truncated");
        require(*below == 17 && *above == 29, "mapped boundary pages overlap");
        std::cout << "16 TiB virtual mapping passed; boundary pages, FIRST payload, and 512 MiB metadata touched\n";
    } catch (const std::bad_alloc&) {
        std::cout << "Skipped: host cannot reserve the 16 TiB virtual mapping\n";
        return 77;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
