#include "../src/MemoryBudget.h"
#include "../src/MsdPrefixCodec.h"
#include "../src/LowFrequencyReader.h"
#include "../src/ConcurrentMemoryPool.h"

#include <iostream>
#include <stdexcept>

struct ConcurrentMemoryPoolTestAccess
{
    static size_t available(const ConcurrentMemoryPool& pool)
    {
        size_t bytes = 0;
        for (int i = 0; i < pool.num_arenas_; ++i)
            bytes += pool.arenas_[i].block_count() * BLOCK_SIZE;
        return bytes;
    }
    static size_t usable(const ConcurrentMemoryPool& pool) { return pool.memory_size_; }
    static size_t mapped(const ConcurrentMemoryPool& pool) { return pool.mmap_size_; }
};

static void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

int main()
{
    try {
        const auto rings = ring_pool_budget();
        constexpr size_t mib = 1ULL << 20;
        require(rings.run_data == 896 * mib && rings.replay_data == 0 && rings.replay_bytes == 0 &&
            rings.final_data == 768 * mib, "RingPool lifetime data accounting");
        require(rings.peak_bytes > rings.run_data, "RingPool object metadata missing");
        require(rings.peak_bytes == std::max({rings.run_bytes, rings.replay_bytes, rings.final_bytes}),
            "RingPool peak must use simultaneous lifetimes");
        constexpr size_t writer_buffers = msd_prefix::FINAL_LOW_BUFFER_BYTES;
        require(msd_prefix::FINAL_LOW_BUFFER_BYTES == 8 * mib, "FinalLow four-buffer budget");
        const size_t peak = std::max(rings.run_bytes, rings.final_bytes) + writer_buffers;
        require(main_pool_overhead_bytes() == peak, "exact phase overhead omits a live buffer");
        require(main_pool_budget(1) == ((1ULL << 30) - peak) / HUGE_PAGE_SIZE * HUGE_PAGE_SIZE,
            "main pool stage peak deduction/alignment");
        for (uint64_t gib : {1U, 2U, 350U}) {
            const size_t main = main_pool_budget(gib);
            require(main % HUGE_PAGE_SIZE == 0 && main + peak <= gib * (1ULL << 30)
                && gib * (1ULL << 30) - main - peak < HUGE_PAGE_SIZE,
                "writer buffers escaped original memory limit");
        }
        require(bloom_bytes(4096) == 4096 * sizeof(std::atomic<uint64_t>), "Bloom needs one word per bin");
        bool overflow = false;
        try { (void)bloom_bytes(4096, SIZE_MAX); } catch (const std::overflow_error&) { overflow = true; }
        require(overflow, "Bloom aggregate capacity overflow accepted");
        for (const uint64_t bad : {uint64_t{0}, UINT64_MAX, uint64_t{UINT64_MAX / (1ULL << 30) + 1}}) {
            bool failed = false;
            try { (void)main_pool_budget(bad); } catch (const std::exception&) { failed = true; }
            require(failed, "invalid or overflowing budget accepted");
        }
        ConcurrentMemoryPool pool(32 * mib);
        const size_t usable = ConcurrentMemoryPoolTestAccess::usable(pool);
        require(ConcurrentMemoryPoolTestAccess::mapped(pool) <= 32 * mib, "mapping enlarged budget");
        pool.allocate_before_init_arenas(4 * mib);
        pool.init_arenas();
        require(ConcurrentMemoryPoolTestAccess::available(pool) == usable - 4 * mib,
            "fixed reservation deducted more than once");
        {
            ConcurrentMemoryPool active(16 * mib);
            const size_t before = ConcurrentMemoryPoolTestAccess::usable(active);
            active.allocate_before_init_arenas(2 * FIRST_SEGMENT_SIZE);
            active.init_arenas();
            require(ConcurrentMemoryPoolTestAccess::available(active) == before - 2 * FIRST_SEGMENT_SIZE,
                "pre-Arena reservation was deducted twice");
        }
        {
            ConcurrentMemoryPool too_small(8 * mib);
            bool failed = false;
            try {
                too_small.allocate_before_init_arenas(8 * mib);
                too_small.init_arenas();
            } catch (const std::bad_alloc&) { failed = true; }
            require(failed, "active reservation silently expanded the memory budget");
        }
        std::cout << "MemoryBudget tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
