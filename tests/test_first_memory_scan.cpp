#include "first_pool_fixture.h"
#include "../src/ConcurrentMemoryPool.h"
#include <array>
#include <barrier>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

struct ConcurrentMemoryPoolTestAccess
{
    static void single(ConcurrentMemoryPool& p) { p.num_arenas_ = 1; p.cpus_per_node_[0] = p.total_cpus_ = 1; }
    static Arena& arena(ConcurrentMemoryPool& p) { return p.arenas_[0]; }
    static Arena& arena(ConcurrentMemoryPool& p, int i) { return p.arenas_[i]; }
    static void two(ConcurrentMemoryPool& p) {
        p.num_arenas_ = p.total_cpus_ = 2;
        p.numa_available_ = false;
        for (int i = 0; i < 2; ++i) {
            p.cpus_per_node_[i] = 1;
            p.numa_distance_order_[i][0] = i;
            p.numa_distance_order_[i][1] = 1 - i;
        }
    }
    static void exhaust_generation(ConcurrentMemoryPool& p) {
        auto& a = p.arenas_[0];
        a.segments[a.segment_capacity - 1].tag.store(FirstSegmentDescriptor::make_tag(
            FirstSegmentDescriptor::MAX_GENERATION, FirstSegmentState::Retired));
    }
};
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using Consumer = ConcurrentMemoryPool::FirstConsumer;
using Status = ConcurrentMemoryPool::FirstClaimStatus;
using View = ConcurrentMemoryPool::FirstSegmentView;

static void check_out_of_order_return()
{
    ConcurrentMemoryPool p(32ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    std::vector<unsigned char> bytes(FIRST_SEGMENT_SIZE, 0x6d);
    for (int i = 0; i < 3; ++i) require(cache_full_first(p, 0, bytes.data()), "fixture publication failed");
    p.freeze_first();
    p.poll_first_progress(0);
    std::array<View, 3> views;
    for (auto& v : views) require(p.try_claim_first(0, Consumer::Scan, v) == Status::Data, "same Arena did not allow independent consumers");
    View none;
    require(p.try_claim_first(0, Consumer::Scan, none) == Status::Busy, "claimed is not completed");
    const size_t gap = p.gap_bytes(0);
    require(p.complete_first(views[2]) && p.complete_first(views[1]), "out-of-order completion failed");
    require(p.gap_bytes(0) == gap, "returned a noncontinuous suffix");
    require(p.complete_first(views[0]), "boundary completion failed");
    require(p.gap_bytes(0) == gap + 3 * FIRST_SEGMENT_SIZE, "continuous Done prefix was not refunded exactly once");
    require(!p.complete_first(views[0]), "duplicate completion accepted");
    require(p.try_claim_first(0, Consumer::Scan, none) == Status::Done, "last completion lost progress");
    auto* reclaimed = static_cast<unsigned char*>(p.allocate_large_from_numa(p.gap_bytes(0), 0));
    require(reclaimed, "Tree could not use the returned contiguous range");
    std::memset(reclaimed, 0x39, 32ULL << 20);
    require(!p.complete_first(views[2]) && p.gap_bytes(0) == 0, "stale completion refunded Tree memory");
}

static void check_parallel_scan()
{
    ConcurrentMemoryPool p(64ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    constexpr size_t segments = 16;
    const size_t capacity = p.first_segment_record_capacity();
    std::vector<unsigned char> bytes(FIRST_SEGMENT_SIZE);
    for (size_t s = 0; s < segments; ++s) {
        for (size_t i = 0; i < capacity; ++i) {
            const uint64_t id = s * capacity + i;
            std::memcpy(bytes.data() + i * 9, &id, 8);
            bytes[i * 9 + 8] = 0x6d;
        }
        require(cache_full_first(p, 0, bytes.data()), "scan fixture exceeded capacity");
    }
    p.freeze_first();
    p.request_first_drain();
    std::array<std::atomic<unsigned>, segments> seen{};
    std::array<std::future<void>, 4> workers;
    std::barrier start(4);
    for (size_t index = 0; index < workers.size(); ++index) workers[index] = std::async(std::launch::async, [&, index] {
        const auto consumer = index == 0 ? Consumer::Spill : Consumer::Scan;
        start.arrive_and_wait();
        for (;;) {
            View v;
            const auto status = p.try_claim_first(0, consumer, v);
            if (status == Status::Done) return;
            if (status == Status::Busy) { p.poll_first_progress(0); std::this_thread::yield(); continue; }
            uint64_t first;
            std::memcpy(&first, v.records, 8);
            const size_t segment = first / capacity;
            require(segment < segments && seen[segment].fetch_add(1) == 0, "duplicate source owner");
            require(v.record_count == capacity, "wrong valid length");
            for (size_t i = 0; i < capacity; ++i) {
                uint64_t id;
                std::memcpy(&id, v.records + i * 9, 8);
                require(id == first + i && v.records[i * 9 + 8] == 0x6d, "source was overwritten before completion");
            }
            require(p.complete_first(v), "valid owner completion rejected");
        }
    });
    for (auto& worker : workers) worker.get();
    for (const auto& n : seen) require(n == 1, "lost source");
    require(p.first_empty(), "sources remained after scan");
}

static void check_tls_partial_and_control_independence()
{
    ConcurrentMemoryPool p(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const std::array<unsigned char, 9> record{1,2,3,4,5,6,7,8,9};
    {
        std::unique_lock held(ConcurrentMemoryPoolTestAccess::arena(p).control_mutex);
        require(p.cache_first_record(0, record.data()), "FIRST fill waited for control lock");
        require(p.try_allocate_large(BLOCK_SIZE), "Tree bump waited for control lock");
    }
    require(p.cache_first_record(0, record.data()), "partial fill failed");
    // Strict NUMA cannot satisfy this request, but must seal before failing.
    bool failed = false;
    try { p.allocate_large_from_numa(32ULL << 20, 0); }
    catch (const std::bad_alloc&) { failed = true; }
    require(failed, "impossible strict request succeeded");
    p.freeze_first();
    View v;
    require(p.try_claim_first(0, Consumer::Scan, v) == Status::Data && v.record_count == 2, "wait did not seal valid partial records");
    require(std::memcmp(v.records, record.data(), 9) == 0 && std::memcmp(v.records + 9, record.data(), 9) == 0, "partial bytes changed");
    require(p.complete_first(v), "partial source completion failed");
}

static void check_partial_ram_reservation()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{};
    require(p.cache_first_record(0, record), "RAM reserve FIRST fixture failed");
    void* tree = p.allocate_large_from_numa(p.gap_bytes(0), 0);
    p.deallocate(tree);
    std::array<void*, 2> pages{};
    {
        std::unique_lock lock(ConcurrentMemoryPoolTestAccess::arena(p).mutex);
        const auto result = p.try_reserve_ram_pages(pages);
        require(result.status == ConcurrentMemoryPool::RamReserveStatus::Busy && result.acquired == 1,
            "try reservation discarded partial ownership or waited on central lock");
    }
    auto result = p.try_reserve_ram_pages(std::span<void*>(pages).subspan(1));
    require(result.status == ConcurrentMemoryPool::RamReserveStatus::NeedReclaim && result.acquired == 0,
        "true capacity shortage was not distinguished from Busy");
    p.seal_thread_first();
    p.freeze_first();
    View v;
    require(p.try_claim_first(0, Consumer::Spill, v) == Status::Data, "RAM shortage did not request independent spill");
    require(p.complete_first(v), "RAM reserve source return failed");
    result = p.try_reserve_ram_pages(std::span<void*>(pages).subspan(1));
    require(result.status == ConcurrentMemoryPool::RamReserveStatus::Ready && result.acquired == 1 && pages[0] != pages[1],
        "RAM reserve did not resume at its remaining deficit");
    p.release_ram_pages(pages);
}

static void check_ready_scan_reservation_does_not_spill()
{
    ConcurrentMemoryPool p(32ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    std::vector<unsigned char> bytes(FIRST_SEGMENT_SIZE, 0x6d);
    for (int i = 0; i < 4; ++i) require(cache_full_first(p, 0, bytes.data()), "ready-scan FIRST fixture failed");
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p);
    p.allocate_large_from_numa(p.gap_bytes(0) - arena.high, 0);
    p.freeze_first();
    // The k=33 full-source bound is 769 target pages. Preparing all of them
    // succeeds but crosses low; no allocator or scanner is asking for spill.
    std::array<void*, (3ULL << 20) / BLOCK_SIZE + 1> pages;
    const auto result = p.try_reserve_ram_pages(pages);
    require(result.status == ConcurrentMemoryPool::RamReserveStatus::Ready && result.acquired == pages.size()
        && p.gap_bytes(0) < arena.low, "ready-scan fixture did not cross low with sufficient target pages");
    View source;
    const bool unnecessary_spill = p.try_claim_first(0, Consumer::Spill, source) == Status::Data;
    size_t seen = 0;
    if (unnecessary_spill) { seen += source.record_count; p.complete_first(source); }
    while (p.try_claim_first(0, Consumer::Scan, source) == Status::Data) {
        require(source.records[0] == 0x6d, "ready-scan source changed before completion");
        seen += source.record_count;
        p.complete_first(source);
    }
    p.release_ram_pages(pages);
    require(seen == 4 * p.first_segment_record_capacity() && p.first_empty(), "ready-scan source ownership lost");
    require(!unnecessary_spill, "ready scan target pages still triggered spill to the high watermark");
}

// Relaxed scheduling flags deliberately add no happens-before edge between
// source reads and the late allocator. That edge must come from the pool.
struct Pause
{
    std::atomic<unsigned> calls{0};
    std::atomic<bool> entered{false}, resume{false};
    void stop_first() {
        if (calls.fetch_add(1, std::memory_order_relaxed)) return;
        entered.store(true, std::memory_order_relaxed);
        while (!resume.load(std::memory_order_relaxed)) std::this_thread::yield();
    }
    void wait() { while (!entered.load(std::memory_order_relaxed)) std::this_thread::yield(); }
};
static Pause* pause_point;
static void pause_once() { pause_point->stop_first(); }
static Pause* pending_pause_point;
static void pause_pending() { pending_pause_point->stop_first(); }

static void check_late_tree_cursor()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{0x6d};
    require(p.cache_first_record(0, record), "late-cursor source fixture failed");
    p.seal_thread_first();
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p);
    uint64_t initial;
    require(arena.try_reserve_tree(2ULL << 20, initial), "late-cursor initial allocation failed");
    p.freeze_first();
    View source;
    require(p.try_claim_first(0, Consumer::Scan, source) == Status::Data, "late-cursor claim failed");
    Pause pause;
    pause_point = &pause;
    Arena::after_tree_charge = pause_once;
    uint64_t late_offset = UINT64_MAX;
    std::thread late([&] {
        if (arena.try_reserve_tree(4ULL << 20, late_offset))
            std::memset(static_cast<char*>(arena.start_addr) + late_offset, 0x39, 4ULL << 20);
    });
    pause.wait();
    const auto last_read = *static_cast<const volatile unsigned char*>(source.records);
    const bool completed = p.complete_first(source);
    uint64_t early_offset = UINT64_MAX;
    const bool allocated = arena.try_reserve_tree(2ULL << 20, early_offset);
    pause.resume.store(true, std::memory_order_relaxed);
    late.join();
    Arena::after_tree_charge = nullptr;
    require(last_read == 0x6d && completed && allocated, "late-cursor reclamation failed");
    require(early_offset == (2ULL << 20) && late_offset == (4ULL << 20) && p.gap_bytes(0) == 0,
        "charged addresses overlap or exceed the returned boundary");
}

static void check_delayed_refund_and_online_impossible()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{0x6d};
    require(p.cache_first_record(0, record), "delayed-refund fixture failed");
    p.seal_thread_first();
    p.allocate_large_from_numa(p.gap_bytes(0), 0);
    p.freeze_first();
    View source;
    require(p.try_claim_first(0, Consumer::Scan, source) == Status::Data, "delayed-refund claim failed");
    Pause pause;
    pause_point = &pause;
    Arena::after_frontier_advance = pause_once;
    auto completion = std::async(std::launch::async, [&] { return p.complete_first(source); });
    pause.wait();
    p.set_spill_service(true);
    auto allocation = std::async(std::launch::async, [&] { return p.allocate_large_from_numa(1ULL << 20, 0); });
    p.poll_first_progress(0);
    const bool waited = allocation.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    const bool empty = p.first_empty();
    const bool grew = p.cache_first_record(0, record);
    pause.resume.store(true, std::memory_order_relaxed);
    const bool completed = completion.get();
    Arena::after_frontier_advance = nullptr;
    void* result = allocation.get();
    require(waited && !empty && !grew && completed && result,
        "delayed refund reopened growth, lost credit, or prematurely rejected a live request");
    require(p.gap_bytes(0) == (1ULL << 20), "delayed refund was not published exactly once");
    bool impossible = false;
    try { p.allocate_large_from_numa(2ULL << 20, 0); }
    catch (const std::bad_alloc&) { impossible = true; }
    require(impossible, "Impossible request waited for the spill service to stop");
    p.set_spill_service(false);
}

static void check_delayed_first_address()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    Pause pause;
    pause_point = &pause;
    Arena::after_first_charge = pause_once;
    const unsigned char record[9]{0x6d};
    auto owner = std::async(std::launch::async, [&] {
        const bool accepted = p.cache_first_record(0, record);
        p.seal_thread_first();
        return accepted;
    });
    pause.wait();
    p.request_first_drain();
    View view;
    const bool claimed = p.try_claim_first(0, Consumer::Spill, view) == Status::Data;
    const bool empty = p.first_empty();
    void* tree = p.try_allocate_large(6ULL << 20);
    pause.resume.store(true, std::memory_order_relaxed);
    const bool accepted = owner.get();
    Arena::after_first_charge = nullptr;
    require(!claimed && !empty && tree && accepted,
        "mode switch lost a charged FIRST reservation or overlapped its address");
    p.freeze_first();
    require(p.try_claim_first(0, Consumer::Scan, view) == Status::Data && view.records[0] == 0x6d,
        "Closing initialized the claim cursor before the reservation published");
    require(p.complete_first(view) && p.first_empty(), "delayed FIRST address was not returned");
}

static void check_generation_limit()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    ConcurrentMemoryPoolTestAccess::exhaust_generation(p);
    const unsigned char record[9]{};
    const bool accepted = p.cache_first_record(0, record);
    bool failed = false;
    try { p.run_state().check(); }
    catch (const std::overflow_error&) { failed = true; }
    require(!accepted && failed && !(ConcurrentMemoryPoolTestAccess::arena(p).admission_word.load() & Arena::ACTIVE_MASK),
        "generation limit wrapped or leaked the short-operation gate");
}

static void check_pending_during_first_address()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    p.set_spill_service(true);
    Pause first_pause, pending_pause;
    pause_point = &first_pause;
    pending_pause_point = &pending_pause;
    Arena::after_first_charge = pause_once;
    const unsigned char record[9]{0x6d};
    auto first = std::async(std::launch::async, [&] {
        const bool ok = p.cache_first_record(0, record);
        p.seal_thread_first();
        return ok;
    });
    first_pause.wait();
    Arena::before_pending_admission = pause_pending;
    auto allocation = std::async(std::launch::async, [&]() -> void* {
        try { return p.allocate_large_from_numa(7ULL << 20, 0); }
        catch (const std::bad_alloc&) { return nullptr; }
    });
    pending_pause.wait();
    first_pause.resume.store(true, std::memory_order_relaxed);
    const bool cached = first.get();
    Arena::after_first_charge = nullptr;
    pending_pause.resume.store(true, std::memory_order_relaxed);
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p);
    SpinBackoff<> backoff;
    while (arena.request.load(std::memory_order_acquire) == Arena::Request::Idle
        && allocation.wait_for(std::chrono::seconds(0)) != std::future_status::ready) backoff.backoff();
    View source;
    if (arena.request.load(std::memory_order_acquire) == Arena::Request::Waiting) {
        while (p.try_claim_first(0, Consumer::Spill, source) != Status::Data) backoff.backoff();
        p.complete_first(source);
    }
    void* result = allocation.get();
    Arena::before_pending_admission = nullptr;
    p.set_spill_service(false);
    require(cached && result, "Grow snapshot rejected a reclaimable contiguous request");
}

static void check_registered_impossible_with_late_tree()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{};
    require(p.cache_first_record(0, record), "registered Impossible fixture failed");
    p.seal_thread_first();
    p.set_spill_service(true);
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p);
    Pause pause;
    pause_point = &pause;
    Arena::after_tree_charge = pause_once;
    auto late = std::async(std::launch::async, [&] {
        uint64_t offset;
        return arena.try_reserve_tree(4ULL << 20, offset);
    });
    pause.wait();
    auto impossible = std::async(std::launch::async, [&] {
        try { p.allocate_large_from_numa(7ULL << 20, 0); }
        catch (const std::bad_alloc&) { return true; }
        return false;
    });
    SpinBackoff<> backoff;
    while (arena.request.load(std::memory_order_acquire) != Arena::Request::Waiting) backoff.backoff();
    View source;
    while (p.try_claim_first(0, Consumer::Spill, source) != Status::Data) backoff.backoff();
    require(p.complete_first(source), "registered Impossible source completion failed");
    const bool failed = impossible.get();
    pause.resume.store(true, std::memory_order_relaxed);
    const bool reserved = late.get();
    Arena::after_tree_charge = nullptr;
    p.set_spill_service(false);
    require(failed && reserved && p.gap_bytes(0) == (4ULL << 20),
        "registered Impossible depended on a late cursor or Writer shutdown");
}

static void check_impossible_retries_other_arena()
{
    ConcurrentMemoryPool p(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::two(p);
    p.init_arenas();
    const unsigned char record[9]{};
    for (int i = 0; i < 2; ++i) {
        require(p.cache_first_record(i, record), "two-Arena source fixture failed");
        p.seal_thread_first();
    }
    p.freeze_first();
    p.set_spill_service(true);
    auto& local = ConcurrentMemoryPoolTestAccess::arena(p, 0);
    auto& remote = ConcurrentMemoryPoolTestAccess::arena(p, 1);
    Pause pause;
    pause_point = &pause;
    Arena::after_tree_charge = pause_once;
    auto late = std::async(std::launch::async, [&] {
        uint64_t offset;
        return local.try_reserve_tree(4ULL << 20, offset);
    });
    pause.wait();
    auto allocation = std::async(std::launch::async, [&]() -> void* {
        try { return p.allocate_large(7ULL << 20); }
        catch (const std::bad_alloc&) { return nullptr; }
    });
    SpinBackoff<> backoff;
    while (local.request.load(std::memory_order_acquire) != Arena::Request::Waiting) backoff.backoff();
    View source;
    while (p.try_claim_first(0, Consumer::Spill, source) != Status::Data) backoff.backoff();
    p.complete_first(source);
    while (remote.request.load(std::memory_order_acquire) != Arena::Request::Waiting
        && allocation.wait_for(std::chrono::seconds(0)) != std::future_status::ready) backoff.backoff();
    if (remote.request.load(std::memory_order_acquire) == Arena::Request::Waiting) {
        while (p.try_claim_first(1, Consumer::Spill, source) != Status::Data) backoff.backoff();
        p.complete_first(source);
    }
    void* result = allocation.get();
    pause.resume.store(true, std::memory_order_relaxed);
    const bool reserved = late.get();
    Arena::after_tree_charge = nullptr;
    p.set_spill_service(false);
    require(reserved && result && remote.contains(result), "Impossible local request bypassed reclaimable remote RAM");
}

static void check_claim_pause_and_generation_reuse()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{0x6d};
    require(p.cache_first_record(0, record), "claim-pause fixture failed");
    p.seal_thread_first();
    p.request_first_drain();
    Pause pause;
    pause_point = &pause;
    Arena::before_claim_cas = pause_once;
    View source;
    auto claim = std::async(std::launch::async, [&] { return p.try_claim_first(0, Consumer::Spill, source); });
    pause.wait();
    p.poll_first_progress(0);
    const bool grew = p.cache_first_record(0, record);
    const bool empty = p.first_empty();
    pause.resume.store(true, std::memory_order_relaxed);
    const auto status = claim.get();
    Arena::before_claim_cas = nullptr;
    require(!grew && !empty && status == Status::Data && p.complete_first(source),
        "paused claim permitted reset or lost ownership");
    p.poll_first_progress(0);
    require(p.cache_first_record(0, record), "growth did not reopen after claim completed");
    p.seal_thread_first();
    p.freeze_first();
    View next;
    require(p.try_claim_first(0, Consumer::Scan, next) == Status::Data
        && next.ref.segment_index == source.ref.segment_index && next.ref.generation == source.ref.generation + 1,
        "generation reuse did not preserve the slot identity");
    require(!p.complete_first(source) && next.records[0] == 0x6d && p.complete_first(next),
        "stale claim completed a reused generation");
}

static void check_completion_publication_pause(bool after_sequence)
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{0x6d};
    for (int i = 0; i < 2; ++i) {
        require(p.cache_first_record(0, record), "completion-pause fixture failed");
        p.seal_thread_first();
    }
    p.freeze_first();
    View a, b;
    require(p.try_claim_first(0, Consumer::Scan, a) == Status::Data
        && p.try_claim_first(0, Consumer::Scan, b) == Status::Data, "completion-pause claims failed");
    Pause pause;
    pause_point = &pause;
    if (after_sequence) Arena::after_completion_sequence = pause_once;
    else Arena::after_done = pause_once;
    auto first = std::async(std::launch::async, [&] {
        const auto last_read = *static_cast<const volatile unsigned char*>(a.records);
        return last_read == 0x6d && p.complete_first(a);
    });
    pause.wait();
    const bool completed = p.complete_first(b);
    const bool empty = p.first_empty();
    const size_t bytes = p.gap_bytes(0);
    void* reused = p.try_allocate_large(bytes);
    if (reused) std::memset(reused, 0x39, bytes);
    pause.resume.store(true, std::memory_order_relaxed);
    const bool first_completed = first.get();
    Arena::after_done = Arena::after_completion_sequence = nullptr;
    require(completed && empty && bytes == (8ULL << 20) && reused && first_completed && p.gap_bytes(0) == 0,
        "last completion missed publication or reused source twice");
}

static void check_adjacent_frontier_publishers()
{
    ConcurrentMemoryPool p(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{0x6d};
    for (int i = 0; i < 2; ++i) {
        require(p.cache_first_record(0, record), "adjacent-frontier fixture failed");
        p.seal_thread_first();
    }
    p.allocate_large_from_numa(p.gap_bytes(0), 0);
    p.freeze_first();
    View a, b;
    require(p.try_claim_first(0, Consumer::Scan, a) == Status::Data
        && p.try_claim_first(0, Consumer::Scan, b) == Status::Data, "adjacent-frontier claims failed");
    Pause pause;
    pause_point = &pause;
    Arena::after_frontier_advance = pause_once;
    auto first = std::async(std::launch::async, [&] {
        const auto last_read = *static_cast<const volatile unsigned char*>(a.records);
        return last_read == 0x6d && p.complete_first(a);
    });
    pause.wait();
    const bool completed = p.complete_first(b);
    // The second refund can be spent on the first publisher's source. Its CAS
    // must acquire that earlier publication even though its refund is paused.
    auto* earlier = p.try_allocate_large(FIRST_SEGMENT_SIZE);
    const bool same = earlier == a.records;
    if (earlier) std::memset(earlier, 0x39, FIRST_SEGMENT_SIZE);
    pause.resume.store(true, std::memory_order_relaxed);
    const bool first_completed = first.get();
    Arena::after_frontier_advance = nullptr;
    require(completed && same && first_completed && p.gap_bytes(0) == FIRST_SEGMENT_SIZE,
        "adjacent publisher lost synchronization or refunded twice");
}

int main()
{
    try {
        k_length = 33;
        check_out_of_order_return();
        check_parallel_scan();
        check_tls_partial_and_control_independence();
        check_partial_ram_reservation();
        check_ready_scan_reservation_does_not_spill();
        check_late_tree_cursor();
        check_delayed_refund_and_online_impossible();
        check_delayed_first_address();
        check_generation_limit();
        check_pending_during_first_address();
        check_registered_impossible_with_late_tree();
        check_impossible_retries_other_arena();
        check_claim_pause_and_generation_reuse();
        check_completion_publication_pause(false);
        check_completion_publication_pause(true);
        check_adjacent_frontier_publishers();
        std::cout << "FirstMemoryScan tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
