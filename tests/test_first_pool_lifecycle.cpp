#include "../src/ConcurrentMemoryPool.h"
#include "../src/ExportWriter.h"
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>

struct ConcurrentMemoryPoolTestAccess
{
    static void single(ConcurrentMemoryPool& pool) { pool.num_arenas_ = 1; pool.total_cpus_ = pool.cpus_per_node_[0] = 1; }
    static FirstSegmentDescriptor* filling(ConcurrentMemoryPool& pool) {
        auto& arena = pool.arenas_[0];
        std::lock_guard lock(arena.admission);
        if (!arena.reserve_first()) throw std::runtime_error("test Filling reservation failed");
        auto& segment = arena.segments[arena.segment_frontier++];
        ++segment.generation;
        segment.data = static_cast<char*>(arena.start_addr) + arena.first_offset;
        segment.state = FirstSegmentState::Filling;
        return &segment;
    }
    static void publish(ConcurrentMemoryPool& pool, FirstSegmentDescriptor* segment) {
        std::lock_guard lock(pool.arenas_[0].admission);
        segment->state = FirstSegmentState::Sealed;
    }
    static size_t pending(ConcurrentMemoryPool& pool) {
        std::lock_guard lock(pool.arenas_[0].admission);
        return pool.arenas_[0].pending_bytes;
    }
    static Arena& arena(ConcurrentMemoryPool& pool) { return pool.arenas_[0]; }
    static void pressure(ConcurrentMemoryPool& pool, size_t gap, bool pending) {
        auto& arena = pool.arenas_[0];
        std::lock_guard lock(arena.admission);
        arena.tree_offset = arena.first_offset - gap;
        arena.pending_bytes = pending ? BLOCK_SIZE : 0;
        pool.update_pressure(arena);
    }
};
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

static void check_publication_without_progress_lock()
{
    ConcurrentMemoryPool pool(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(pool);
    pool.init_arenas();
    std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 0x37);
    std::unique_lock held(pool.run_state().mutex);
    std::promise<void> published;
    auto result = published.get_future();
    std::thread owner([&] { pool.cache_first_segment(0, records.data()); published.set_value(); });
    const bool immediate = result.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
    held.unlock();
    owner.join();
    require(immediate, "FIRST publication waited for the global progress lock");
    FirstSegmentRef ref;
    pool.request_first_drain();
    require(pool.claim_spill(0, ref), "publication did not seal the full segment");
    pool.empty_first(ref);
}

static void check_pressure_transitions()
{
    ConcurrentMemoryPool pool(64ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(pool);
    pool.init_arenas();
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(pool);
    const auto generation = pool.pressure_generation(0);
    struct Case { size_t gap; bool pending; MemoryPressure pressure; bool reclaim; uint64_t changes; };
    for (const auto& item : {
        Case{arena.high, false, MemoryPressure::Normal, false, 0},
        Case{arena.high - BLOCK_SIZE, false, MemoryPressure::Soft, false, 1},
        Case{arena.high - BLOCK_SIZE, false, MemoryPressure::Soft, false, 1},
        Case{arena.low, false, MemoryPressure::Hard, true, 2},
        Case{arena.low, false, MemoryPressure::Hard, true, 2},
        Case{arena.low + BLOCK_SIZE, false, MemoryPressure::Soft, true, 3},
        Case{arena.high, false, MemoryPressure::Normal, false, 4},
        Case{arena.high, true, MemoryPressure::Hard, true, 5},
        Case{arena.high, false, MemoryPressure::Normal, false, 6}}) {
        ConcurrentMemoryPoolTestAccess::pressure(pool, item.gap, item.pending);
        require(pool.pressure(0) == item.pressure && arena.hard_reclaim == item.reclaim,
            "pressure threshold or reclaim hysteresis changed");
        require(pool.pressure_generation(0) == generation + item.changes,
            "pressure generation lost a transition or counted an unchanged state");
    }
}

static void check_writer_observes_publication()
{
    RunState state;
    ConcurrentMemoryPool pool(16ULL << 20, &state);
    ConcurrentMemoryPoolTestAccess::single(pool);
    pool.init_arenas();
    auto* segment = ConcurrentMemoryPoolTestAccess::filling(pool);
    std::memset(segment->data, 0x37, FIRST_SEGMENT_SIZE);
    pool.request_first_drain();
    FirstSegmentRef ref;
    require(!pool.claim_spill(0, ref), "Writer read an unpublished Filling segment");
    // Reserving another complete segment while a prior Filling copy is paused
    // must not hold admission. Its consumption must block further growth.
    std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 0x73);
    require(pool.cache_first_segment(0, records.data()), "Filling held the management lock");
    require(pool.claim_spill(0, ref), "second sealed boundary could not be consumed");
    require(!pool.cache_first_segment(0, records.data()), "consumption permitted new FIRST");
    ConcurrentMemoryPoolTestAccess::publish(pool, segment); // Prior Filling may finish.
    const auto gap = pool.gap_bytes(0);
    pool.empty_first(ref);
    require(pool.gap_bytes(0) == gap + FIRST_SEGMENT_SIZE, "return waited for an internal segment");
    // Pause publication again so Writer must poll without a progress notification.
    { std::lock_guard lock(ConcurrentMemoryPoolTestAccess::arena(pool).admission); segment->state = FirstSegmentState::Filling; }
    // Only FIRST spill is used; no ordinary Export block is published.
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> queue(BLOCK_SIZE, 1);
    const auto path = std::filesystem::temp_directory_path() / ("first-poll-" + std::to_string(getpid()));
    ExportWriter writer(&queue, path.string(), ExportWriter::OutputKind::Pending, &pool);
    writer.start();
    const auto needed = pool.gap_bytes(0) + BLOCK_SIZE;
    auto allocation = std::async(std::launch::async, [&] { return pool.allocate_large_from_numa(needed, 0); });
    while (!ConcurrentMemoryPoolTestAccess::pending(pool)) {
        const auto progress = state.progress();
        if (!ConcurrentMemoryPoolTestAccess::pending(pool)) state.wait(progress);
    }
    ConcurrentMemoryPoolTestAccess::publish(pool, segment);
    if (allocation.wait_for(std::chrono::seconds(2)) != std::future_status::ready)
        state.fail(std::make_exception_ptr(std::runtime_error("Writer did not observe published FIRST")), "test");
    require(allocation.get() && pool.first_empty(), "polling Writer did not satisfy the RAM reservation");
    queue.producer_set_finished();
    writer.join();
    require(std::filesystem::file_size(path.string() + ".0") == ConcurrentMemoryPool::first_segment_record_capacity() * kmer_bytes(), "polling spill lost its records");
    std::filesystem::remove(path.string() + ".0");
}

int main()
{
    try {
        k_length = 33;
        check_pressure_transitions();
        check_publication_without_progress_lock();
        check_writer_observes_publication();
        {
            ConcurrentMemoryPool exhausted(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::single(exhausted);
            exhausted.init_arenas();
            auto& arena = ConcurrentMemoryPoolTestAccess::arena(exhausted);
            void* page = exhausted.allocate_large_from_numa(exhausted.gap_bytes(0), 0);
            auto* free = static_cast<FreeBlock*>(page);
            free->next = nullptr;
            arena.central_free_list = free;
            std::unique_lock held(arena.mutex);
            auto allocation = std::async(std::launch::async, [&] { return exhausted.allocate(); });
            const bool waited = allocation.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
            held.unlock();
            require(waited && allocation.get() == page, "normal allocation mistook mutex contention for exhaustion");
        }
        ConcurrentMemoryPool pool(32ULL << 20);
        ConcurrentMemoryPoolTestAccess::single(pool);
        pool.init_arenas();
        bool failed = false;
        try { pool.allocate_large(SIZE_MAX); } catch (const std::bad_alloc&) { failed = true; }
        require(failed, "overflow did not fail");
        std::vector<unsigned char> records(FIRST_SEGMENT_SIZE);
        require(pool.cache_first_segment(0, records.data()), "no full first segment");
        pool.set_spill_service(true);
        const size_t gap = pool.gap_bytes(0);
        const auto progress = pool.run_state().progress();
        auto allocation = std::async(std::launch::async, [&] { return pool.allocate_large(gap + BLOCK_SIZE); });
        while (!ConcurrentMemoryPoolTestAccess::pending(pool)) pool.run_state().wait(progress);
        require(pool.try_allocate_large(BLOCK_SIZE) == nullptr, "bump stole pending request space");
        require(!pool.cache_first_segment(0, records.data()), "FIRST grew during pending request");
        FirstSegmentRef spill;
        require(pool.claim_spill(0, spill), "writer could not claim frontier");
        pool.empty_first(spill);
        require(allocation.get() != nullptr, "reserved allocation did not progress");
        pool.set_spill_service(false);
        failed = false;
        try { pool.allocate_large_from_numa(32ULL << 20, 0); } catch (const std::bad_alloc&) { failed = true; }
        require(failed, "impossible contiguous request hung");
        // A service shutdown and a first error must both release allocation
        // waits even when a Filling copy has not published its segment yet.
        for (bool inject_error : {false, true}) {
            ConcurrentMemoryPool waiting(16ULL << 20);
            ConcurrentMemoryPoolTestAccess::single(waiting);
            waiting.init_arenas();
            ConcurrentMemoryPoolTestAccess::filling(waiting);
            waiting.set_spill_service(true);
            const size_t needed = waiting.gap_bytes(0) + BLOCK_SIZE;
            auto blocked = std::async(std::launch::async, [&] {
                try { waiting.allocate_large(needed); } catch (const std::exception&) { return true; }
                return false;
            });
            while (!ConcurrentMemoryPoolTestAccess::pending(waiting)) {
                const auto progress = waiting.run_state().progress();
                if (!ConcurrentMemoryPoolTestAccess::pending(waiting)) waiting.run_state().wait(progress);
            }
            if (inject_error) waiting.run_state().fail(std::make_exception_ptr(std::runtime_error("injected")), "test");
            else waiting.set_spill_service(false);
            require(blocked.get() == inject_error, "shutdown fallback or failure propagation changed");
            require(ConcurrentMemoryPoolTestAccess::pending(waiting) == 0, "failed request reservation stranded");
        }
        // Repeated pools on one thread must not use a prior pool's TLS pages.
        for (int i = 0; i < 3; ++i) {
            ConcurrentMemoryPool other(8ULL << 20);
            other.init_arenas();
            require(other.try_allocate(), "TLS pool switch failed");
        }
        std::cout << "FirstPoolLifecycle tests passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
