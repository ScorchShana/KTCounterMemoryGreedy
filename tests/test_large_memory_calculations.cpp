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
            require(arena.tree_offset == (8ULL << 40) - BLOCK_SIZE, "Tree offset truncated");
            require(arena.try_reserve_tree(BLOCK_SIZE, offset), "Tree cannot reach 8 TiB");
            if (tib == 8) {
                require(!arena.try_reserve_tree(BLOCK_SIZE, offset), "Tree crossed FIRST boundary");
            } else {
                require(arena.try_reserve_tree(BLOCK_SIZE, offset), "Tree cannot cross 8 TiB");
                require(arena.tree_offset == (8ULL << 40) + BLOCK_SIZE, "Tree wrapped at 8 TiB");
                require(arena.try_begin_first(offset), "FIRST allocation failed above 8 TiB");
                require(arena.first_offset == (tib << 40) - HUGE_PAGE_SIZE, "FIRST offset truncated");
                arena.leave_short();
                arena.first_offset.fetch_add(HUGE_PAGE_SIZE);
                arena.admission_word.fetch_add(Arena::SEGMENT_PAGES);
                require(arena.first_offset == arena.capacity, "FIRST return truncated");
            }
            arena.admission_word.fetch_or(Arena::PRIORITY);
            require(!arena.try_reserve_tree(BLOCK_SIZE, offset) && !arena.try_begin_first(offset), "pending request lost admission");
        }
        Arena arena;
        arena.initialize(8ULL << 40);
        uint64_t offset;
        arena.high = HUGE_PAGE_SIZE;
        require(arena.try_reserve_tree(arena.capacity - 2 * HUGE_PAGE_SIZE, offset), "numeric Tree setup failed");
        require(arena.try_begin_first(offset), "FIRST rejected exact high watermark");
        arena.leave_short();
        require(!arena.try_begin_first(offset), "FIRST consumed high watermark");
        arena.set_mode(Arena::Mode::Closing);
        require(!arena.try_begin_first(offset), "FIRST grew after freeze");
        const FirstSegmentRef ref{1, (1ULL << 32) + 7, UINT64_MAX - 1};
        require(ref.segment_index == 4294967303ULL, "segment index truncated");
        require(proportional_size(UINT64_MAX, 2, 3) == 12297829382473034410ULL,
            "NUMA proportional multiplication overflowed");
        require(proportional_size(UINT64_MAX, 3, 3) == UINT64_MAX, "NUMA whole share overflowed");
        for (uint64_t gib : {8192ULL, 16384ULL, 65536ULL})
            require(main_pool_budget(gib) == (gib << 30) - 945815552ULL,
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
