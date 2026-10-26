#include "../src/ConcurrentMemoryPool.h"
#include "../src/MemoryBudget.h"
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

int main()
{
    try {
        // Exercise the production boundary arithmetic without constructing
        // pointers or descriptor arrays for these capacities.
        for (uint64_t tib : {8ULL, 16ULL, 64ULL}) {
            Arena arena;
            arena.initialize(tib << 40);
            uint64_t offset;
            arena.high = HUGE_PAGE_SIZE;
            require(arena.try_reserve_tree((8ULL << 40) - BLOCK_SIZE, offset), "Tree truncated below 8 TiB");
            require(arena.tree_end == (8ULL << 40) - BLOCK_SIZE, "Tree offset truncated");
            require(arena.try_reserve_tree(BLOCK_SIZE, offset), "Tree cannot reach 8 TiB");
            if (tib == 8) {
                require(!arena.try_reserve_tree(BLOCK_SIZE, offset), "Tree crossed FIRST boundary");
            } else {
                require(arena.try_reserve_tree(BLOCK_SIZE, offset), "Tree cannot cross 8 TiB");
                require(arena.tree_end == (8ULL << 40) + BLOCK_SIZE, "Tree wrapped at 8 TiB");
                // A 2 MiB FIRST range remains an exact 64-bit boundary.
                arena.first_begin = arena.capacity - FIRST_SEGMENT_SIZE;
                arena.claim_cursor = arena.first_begin;
                require(arena.first_begin == (tib << 40) - HUGE_PAGE_SIZE, "FIRST offset truncated");
                require(!arena.try_reserve_tree(arena.capacity - arena.tree_end, offset), "Tree crossed FIRST boundary");
            }
            Arena::Waiter waiter{arena.first_begin - arena.tree_end};
            arena.enqueue(waiter);
            require(!arena.try_reserve_tree(BLOCK_SIZE, offset), "FIFO protection truncated");
            if (waiter.bytes) require(arena.try_reserve_tree(waiter.bytes, offset, &waiter), "FIFO head cannot use its gap");
            arena.remove(waiter);
        }
        const FirstSegmentRef ref{1, (1ULL << 32) + 7, UINT64_MAX - 1};
        require(ref.segment_index == 4294967303ULL, "segment index truncated");
        require(proportional_size(UINT64_MAX, 2, 3) == 12297829382473034410ULL,
            "NUMA proportional multiplication overflowed");
        require(proportional_size(UINT64_MAX, 3, 3) == UINT64_MAX, "NUMA whole share overflowed");
        for (uint64_t gib : {8192ULL, 16384ULL, 65536ULL})
            require(main_pool_budget(gib) == (gib << 30) - 950009856ULL,
                "large main-pool budget truncated");
        for (uint64_t gib : {100ULL, 101ULL, 350ULL, 8192ULL, 16384ULL, 65536ULL})
            require(arena_watermark(gib << 30, 5) == (5ULL << 30)
                && arena_watermark(gib << 30, 8) == (8ULL << 30), "large-Arena watermarks exceed 5/8 GiB caps");
        require(arena_watermark((100ULL << 30) - (100ULL << 20), 5) == (5ULL << 30) - (4ULL << 20)
            && arena_watermark((100ULL << 30) - (100ULL << 20), 8) == (8ULL << 30) - (8ULL << 20),
            "watermarks below crossover lost percentage or 2 MiB rounding");
        require(arena_watermark((100ULL << 30) + HUGE_PAGE_SIZE, 5) == (5ULL << 30)
            && arena_watermark((100ULL << 30) + HUGE_PAGE_SIZE, 8) == (8ULL << 30),
            "watermarks immediately above crossover exceed caps");
        require(arena_watermark(1ULL << 30, 5) == (52ULL << 20)
            && arena_watermark(1ULL << 30, 8) == (82ULL << 20), "small-Arena watermarks changed");
        require(arena_watermark((350ULL << 30) / 4, 5) == (4480ULL << 20)
            && arena_watermark((350ULL << 30) / 4, 8) == (7168ULL << 20),
            "NUMA Arena watermarks must use local capacity");
        require(arena_watermark(SIZE_MAX, 5) == (5ULL << 30)
            && arena_watermark(SIZE_MAX, 8) == (8ULL << 30), "maximum capacity watermarks exceed caps");
        bool rejected = false;
        try { (void)bloom_bytes(UINT64_MAX); } catch (const std::overflow_error&) { rejected = true; }
        require(rejected, "Bloom byte calculation wrapped");
        rejected = false;
        try { (void)align_up(SIZE_MAX, HUGE_PAGE_SIZE); } catch (const std::overflow_error&) { rejected = true; }
        require(rejected, "capacity alignment overflow accepted");
        std::cout << "Large-memory numeric tests passed (no large mapping)\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
