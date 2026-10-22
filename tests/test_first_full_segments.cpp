#include "../src/ConcurrentMemoryPool.h"
#include <array>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

struct ConcurrentMemoryPoolTestAccess
{
    static void single(ConcurrentMemoryPool& pool) { pool.num_arenas_ = 1; pool.cpus_per_node_[0] = pool.total_cpus_ = 1; }
    static Arena& arena(ConcurrentMemoryPool& pool) { return pool.arenas_[0]; }
};
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

static void check_reuse_with_concurrent_publishers()
{
    ConcurrentMemoryPool pool(64ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(pool);
    auto* active = static_cast<unsigned char*>(pool.allocate_before_init_arenas(4 * FIRST_SEGMENT_SIZE));
    pool.init_arenas();
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(pool);
    auto* descriptors = arena.segments.get();
    const size_t gap = pool.gap_bytes(0);
    pool.request_first_drain();
    std::array<std::future<void>, 4> publishers;
    constexpr size_t per_publisher = 64;
    for (size_t p = 0; p < publishers.size(); ++p) {
        publishers[p] = std::async(std::launch::async, [&, p] {
            auto* buffer = active + p * FIRST_SEGMENT_SIZE;
            for (size_t n = 0; n < per_publisher; ++n) {
                const unsigned char id = p * per_publisher + n;
                std::memset(buffer, id, FIRST_SEGMENT_SIZE);
                while (!pool.cache_first_segment(0, buffer)) std::this_thread::yield();
            }
        });
    }
    std::array<bool, 256> seen{};
    for (size_t n = 0; n < seen.size();) {
        FirstSegmentRef ref;
        auto* segment = pool.claim_spill(0, ref);
        if (!segment) { std::this_thread::yield(); continue; }
        const auto* data = reinterpret_cast<unsigned char*>(segment->data);
        const unsigned char id = data[0];
        require(!seen[id], "concurrent publishers duplicated a full segment");
        seen[id] = true;
        for (size_t i = 0; i < pool.first_segment_record_capacity() * kmer_bytes(); ++i)
            require(data[i] == id, "Writer observed a partial Filling copy");
        const size_t before = pool.gap_bytes(0);
        pool.empty_first(ref);
        // Other producers may immediately reuse the returned slot. Metadata must
        // stay bounded even under this sustained publication/consumption race.
        require(before <= gap, "FIRST crossed its budget");
        ++n;
    }
    for (auto& publisher : publishers) publisher.get();
    require(pool.first_empty() && pool.gap_bytes(0) == gap && arena.segments.get() == descriptors,
        "repeated publication leaked space or descriptor storage");
}

int main()
{
    try {
        for (const auto [k, records, tail] : std::array<std::array<size_t, 3>, 5>{{
            {1, 2097152, 0}, {31, 262144, 0}, {33, 233016, 8}, {65, 123361, 15}, {128, 65536, 0}}}) {
            k_length = k;
            ConcurrentMemoryPool pool(32ULL << 20);
            ConcurrentMemoryPoolTestAccess::single(pool);
            auto* active = static_cast<unsigned char*>(pool.allocate_before_init_arenas(FIRST_SEGMENT_SIZE));
            pool.init_arenas();
            require(pool.gap_bytes(0) == (30ULL << 20), "active buffer was not deducted exactly once");
            require(pool.first_segment_record_capacity() == records && FIRST_SEGMENT_SIZE - records * kmer_bytes() == tail,
                "segment record capacity or padding differs");
            for (size_t i = 0; i < records * kmer_bytes(); ++i) active[i] = i % 251;
            require(pool.cache_first_segment(0, active), "full-segment admission failed");
            auto& arena = ConcurrentMemoryPoolTestAccess::arena(pool);
            const size_t gap = pool.gap_bytes(0);
            pool.request_first_drain();
            FirstSegmentRef ref, second;
            auto* source = pool.claim_spill(0, ref);
            require(source && !pool.claim_spill(0, second), "Arena allowed two consumers");
            require(!pool.cache_first_segment(0, active), "FIRST grew behind the consuming boundary");
            require(std::memcmp(source->data, active, records * kmer_bytes()) == 0, "published segment changed records");
            char* source_address = source->data;
            pool.empty_first(ref);
            require(pool.gap_bytes(0) == gap + FIRST_SEGMENT_SIZE && pool.first_empty(), "consumption did not immediately return 2 MiB");
            require(pool.allocate_large_from_numa(pool.gap_bytes(0), 0), "Tree did not reuse returned space");
            std::memset(source_address, 0x39, FIRST_SEGMENT_SIZE);
            pool.empty_first(ref); // A stale return must not release Tree memory.
            require(pool.gap_bytes(0) == 0, "stale reference returned live Tree storage");
            require(arena.segment_frontier == 0, "descriptor frontier did not retire");
        }
        k_length = 33;
        ConcurrentMemoryPool pool(32ULL << 20);
        ConcurrentMemoryPoolTestAccess::single(pool);
        pool.init_arenas();
        std::vector<unsigned char> data(FIRST_SEGMENT_SIZE, 0x73);
        auto& arena = ConcurrentMemoryPoolTestAccess::arena(pool);
        std::unique_lock held(arena.admission);
        auto storing = std::async(std::launch::async, [&] { return pool.cache_first_segment(0, data.data()); });
        const bool waited = storing.wait_for(std::chrono::milliseconds(30)) == std::future_status::timeout;
        held.unlock();
        require(storing.get() && waited, "management contention was treated as cache exhaustion");
        // Descriptor storage is reused through many publish/retire cycles.
        auto* descriptors = arena.segments.get();
        pool.freeze_first();
        require(!pool.cache_first_segment(0, data.data()), "frozen cache grew");
        require(pool.extract_first_segment(0, data.data()) == ConcurrentMemoryPool::FirstExtractStatus::Data,
            "whole-segment extraction failed");
        require(pool.first_empty() && data.front() == 0x73 && data[2097143] == 0x73,
            "extraction returned before releasing its source or changed data");
        require(arena.segments.get() == descriptors, "descriptor allocation grew");
        require(pool.extract_first_segment(0, data.data()) == ConcurrentMemoryPool::FirstExtractStatus::Done,
            "empty cache did not finish");
        check_reuse_with_concurrent_publishers();
        std::cout << "FirstFullSegments tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
