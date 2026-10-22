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
        arena.capacity = arena.first_offset = pool.memory_size_;
        arena.end_addr = static_cast<char*>(arena.start_addr) + arena.capacity;
        arena.high = HUGE_PAGE_SIZE;
        arena.low = BLOCK_SIZE;
        arena.segment_capacity = arena.capacity / FIRST_SEGMENT_SIZE;
        // Only slot zero is exercised. Do not fault millions of descriptors
        // merely to validate the mapped production address calculations.
        arena.segments = std::make_unique<FirstSegmentDescriptor[]>(1);
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
        ConcurrentMemoryPool pool(16ULL << 40);
        ConcurrentMemoryPoolTestAccess::mapped_arena(pool);
        auto* base = static_cast<char*>(pool.allocate_large_from_numa((8ULL << 40) - BLOCK_SIZE, 0));
        auto* below = static_cast<char*>(pool.allocate_large_from_numa(BLOCK_SIZE, 0));
        auto* above = static_cast<char*>(pool.allocate_large_from_numa(BLOCK_SIZE, 0));
        require(below == base + (8ULL << 40) - BLOCK_SIZE && above == base + (8ULL << 40),
            "mapped allocation truncated at 8 TiB");
        *static_cast<volatile char*>(below) = 17;
        *static_cast<volatile char*>(above) = 29;
        FirstSegmentRef ref;
        const size_t gap = pool.gap_bytes(0);
        std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 41);
        require(pool.cache_first_segment(0, records.data()), "mapped FIRST publication failed");
        pool.request_first_drain();
        auto* first = pool.claim_spill(0, ref);
        require(first && pool.gap_bytes(0) == gap - FIRST_SEGMENT_SIZE, "mapped FIRST allocation failed");
        *static_cast<volatile char*>(first->data) = 41;
        pool.empty_first(ref);
        require(pool.gap_bytes(0) == gap, "mapped FIRST reclamation truncated");
        require(*below == 17 && *above == 29, "mapped boundary pages overlap");
        std::cout << "16 TiB virtual mapping passed; boundary pages and one full FIRST segment touched, no physical-memory claim\n";
    } catch (const std::bad_alloc&) {
        std::cout << "Skipped: host cannot reserve the 16 TiB virtual mapping\n";
        return 77;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
