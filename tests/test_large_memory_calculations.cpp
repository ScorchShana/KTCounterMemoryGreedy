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
            arena.capacity = tib << 40;
            arena.first_offset = arena.capacity;
            arena.high = HUGE_PAGE_SIZE;
            require(arena.reserve_tree((8ULL << 40) - BLOCK_SIZE), "Tree truncated below 8 TiB");
            require(arena.tree_offset == (8ULL << 40) - BLOCK_SIZE, "Tree offset truncated");
            require(arena.reserve_tree(BLOCK_SIZE), "Tree cannot reach 8 TiB");
            if (tib == 8) {
                require(!arena.reserve_tree(BLOCK_SIZE), "Tree crossed FIRST boundary");
            } else {
                require(arena.reserve_tree(BLOCK_SIZE), "Tree cannot cross 8 TiB");
                require(arena.tree_offset == (8ULL << 40) + BLOCK_SIZE, "Tree wrapped at 8 TiB");
                require(arena.reserve_first(), "FIRST allocation failed above 8 TiB");
                require(arena.first_offset == (tib << 40) - HUGE_PAGE_SIZE, "FIRST offset truncated");
                arena.first_offset += HUGE_PAGE_SIZE;
                require(arena.first_offset == arena.capacity, "FIRST return truncated");
            }
            arena.pending_bytes = BLOCK_SIZE;
            require(!arena.reserve_tree(BLOCK_SIZE) && !arena.reserve_first(), "pending request lost admission");
        }
        Arena arena;
        arena.capacity = arena.first_offset = 8ULL << 40;
        arena.high = HUGE_PAGE_SIZE;
        arena.tree_offset = arena.capacity - 2 * HUGE_PAGE_SIZE;
        require(arena.reserve_first(), "FIRST rejected exact high watermark");
        require(!arena.reserve_first(), "FIRST consumed high watermark");
        arena.first_enabled = false;
        arena.tree_offset = 0;
        require(!arena.reserve_first(), "FIRST grew after freeze");
        const FirstSegmentRef ref{1, (1ULL << 32) + 7, UINT64_MAX - 1};
        require(ref.segment_index == 4294967303ULL, "segment index truncated");
        require(proportional_size(UINT64_MAX, 2, 3) == 12297829382473034410ULL,
            "NUMA proportional multiplication overflowed");
        require(proportional_size(UINT64_MAX, 3, 3) == UINT64_MAX, "NUMA whole share overflowed");
        for (uint64_t gib : {8192ULL, 16384ULL, 65536ULL})
            require(main_pool_budget(gib) == (gib << 30) - 941621248ULL,
                "large main-pool budget truncated");
        require(arena_watermark(8ULL << 40, 5) == 439806328832ULL
            && arena_watermark(8ULL << 40, 8) == 703688867840ULL, "8 TiB watermark calculation truncated");
        require(arena_watermark(16ULL << 40, 5) == 879610560512ULL
            && arena_watermark(16ULL << 40, 8) == 1407375638528ULL, "16 TiB watermark calculation truncated");
        require(arena_watermark(64ULL << 40, 5) == 3518438047744ULL
            && arena_watermark(64ULL << 40, 8) == 5629500456960ULL, "64 TiB watermark calculation truncated");
        bool rejected = false;
        try { (void)bloom_bytes(UINT64_MAX); } catch (const std::overflow_error&) { rejected = true; }
        require(rejected, "Bloom byte calculation wrapped");
        rejected = false;
        try { (void)align_up(SIZE_MAX, HUGE_PAGE_SIZE); } catch (const std::overflow_error&) { rejected = true; }
        require(rejected, "capacity alignment overflow accepted");
        std::cout << "Large-memory numeric tests passed (no large mapping)\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
