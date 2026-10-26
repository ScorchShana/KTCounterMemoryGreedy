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
    static void refresh_locked(ConcurrentMemoryPool& p) { p.update_pressure(p.arenas_[0]); }
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
    p.set_spill_service(true);
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
    p.set_spill_service(true);
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
    p.set_spill_service(true);
    const std::array<unsigned char, 9> record{1,2,3,4,5,6,7,8,9};
    require(p.cache_first_record(0, record.data()), "partial fill failed");
    {
        std::unique_lock held(ConcurrentMemoryPoolTestAccess::arena(p).mutex);
        require(p.cache_first_record(0, record.data()), "held FIRST fill acquired Arena lock");
        require(!p.try_allocate_large(BLOCK_SIZE), "try Tree bump ignored Arena lock");
        p.seal_thread_first(); // Publication does not take the Arena lock.
    }
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
    p.set_spill_service(true);
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
    p.set_spill_service(true);
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

static void check_nonblocking_and_normal_admission()
{
    ConcurrentMemoryPool p(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p);
    for (bool first : {false, true}) {
        std::promise<void> started;
        std::unique_lock held(arena.mutex);
        auto normal = std::async(std::launch::async, [&] {
            started.set_value();
            const unsigned char record[9]{};
            if (first) { const bool ok = p.cache_first_record(0, record); p.seal_thread_first(); return ok; }
            return p.allocate_large(BLOCK_SIZE) != nullptr;
        });
        started.get_future().wait();
        const bool waited = normal.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
        held.unlock();
        const bool succeeded = normal.get();
        require(waited && succeeded, "normal admission mistook a busy Arena for exhaustion");
    }
    p.freeze_first();
    View view;
    while (p.try_claim_first(0, Consumer::Scan, view) == Status::Data) p.complete_first(view);
    std::unique_lock held(arena.mutex);
    auto nonblocking = std::async(std::launch::async, [&] {
        std::array<void*, 1> pages;
        p.poll_first_progress(0);
        return p.try_allocate() == nullptr && p.try_allocate_large(BLOCK_SIZE) == nullptr
            && p.try_reserve_ram_pages(pages).status == ConcurrentMemoryPool::RamReserveStatus::Busy
            && p.try_claim_first(0, Consumer::Scan, view) == Status::Busy;
    });
    const bool immediate = nonblocking.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
    held.unlock();
    const bool result = nonblocking.get();
    require(immediate && result, "Pool try path blocked, including frozen-empty claim checking");
}

static void check_bounded_prefix_and_generation()
{
    ConcurrentMemoryPool p(384ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{0x6d};
    constexpr size_t count = 130;
    for (size_t i = 0; i < count; ++i) {
        require(p.cache_first_record(0, record), "long-prefix fixture failed");
        p.seal_thread_first();
    }
    p.freeze_first();
    std::array<View, count> views;
    for (auto& view : views) require(p.try_claim_first(0, Consumer::Scan, view) == Status::Data, "long-prefix claim failed");
    const auto gap = p.gap_bytes(0);
    for (size_t i = count - 1; i; --i) require(p.complete_first(views[i]), "suffix completion failed");
    require(p.gap_bytes(0) == gap, "noncontinuous Done suffix was returned");
    require(p.complete_first(views[0]) && p.first_empty() && p.gap_bytes(0) == gap + count * FIRST_SEGMENT_SIZE,
        "last completion stranded a Done prefix longer than 64 descriptors");
    require(!p.complete_first(views[0]), "retired generation completed twice");

    ConcurrentMemoryPool limited(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(limited);
    limited.init_arenas();
    ConcurrentMemoryPoolTestAccess::exhaust_generation(limited);
    const auto before = limited.gap_bytes(0);
    require(!limited.cache_first_record(0, record) && limited.run_state().failed()
        && limited.gap_bytes(0) == before && limited.first_empty(), "generation overflow changed a boundary");
}

static void check_fifo_extra_gap_and_free_pages()
{
    ConcurrentMemoryPool p(32ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    void* free_page = p.allocate_large_from_numa(BLOCK_SIZE, 0);
    const unsigned char record[9]{};
    require(p.cache_first_record(0, record), "FIFO fixture failed");
    p.seal_thread_first();
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p);
    Arena::Waiter head{p.gap_bytes(0) - 2 * BLOCK_SIZE}, tail{BLOCK_SIZE};
    {
        std::lock_guard lock(arena.mutex);
        arena.enqueue(head);
        arena.enqueue(tail);
        ConcurrentMemoryPoolTestAccess::refresh_locked(p);
    }
    p.poll_first_progress(0);
    require(p.pressure(0) == MemoryPressure::Hard, "FIFO pressure snapshot was not Hard");
    require(p.try_allocate_large(2 * BLOCK_SIZE), "FIFO blocked surplus gap");
    require(!p.try_allocate_large(BLOCK_SIZE), "bump stole FIFO head space");
    require(!p.cache_first_record(0, record), "FIRST grew with a FIFO waiter");
    p.release_ram_pages(std::span<void* const>(&free_page, 1));
    require(p.try_allocate() == free_page, "FIFO blocked an independent free page");
    {
        std::lock_guard lock(arena.mutex);
        uint64_t offset;
        require(arena.try_reserve_tree(head.bytes, offset, &head), "FIFO head could not consume its protected gap");
        arena.remove(head);
        require(arena.wait_head == &tail && !tail.previous, "FIFO head removal damaged links");
        arena.remove(tail);
        ConcurrentMemoryPoolTestAccess::refresh_locked(p);
        require(!arena.wait_head && !arena.wait_tail, "FIFO removal stranded a node");
    }
    p.poll_first_progress(0);
}

static size_t waiter_count(ConcurrentMemoryPool& p, int index = 0)
{
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p, index);
    std::lock_guard lock(arena.mutex);
    size_t count = 0;
    for (auto* waiter = arena.wait_head; waiter; waiter = waiter->next) ++count;
    return count;
}
static void await_waiters(ConcurrentMemoryPool& p, size_t count)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (waiter_count(p) != count && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    if (waiter_count(p) != count) {
        p.set_spill_service(false);
        throw std::runtime_error("FIFO registration did not progress");
    }
}

static void claim_after_contention(ConcurrentMemoryPool& p, int preferred, Consumer consumer, View& view)
{
    SpinBackoff<> backoff;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (std::chrono::steady_clock::now() < deadline) {
        if (p.try_claim_first(preferred, consumer, view) == Status::Data) return;
        backoff.backoff();
    }
    // Failed assertions must not leave allocation futures waiting forever.
    p.run_state().fail(std::make_exception_ptr(std::runtime_error("claim did not progress")), "test");
    throw std::runtime_error("claim did not progress");
}

static void check_fifo_cancellation_and_remote_retry()
{
    for (bool failure : {false, true}) {
        ConcurrentMemoryPool p(16ULL << 20);
        ConcurrentMemoryPoolTestAccess::single(p);
        p.init_arenas();
        const unsigned char record[9]{};
        require(p.cache_first_record(0, record), "FIFO cancellation fixture failed");
        p.seal_thread_first();
        p.allocate_large_from_numa(p.gap_bytes(0), 0);
        p.set_spill_service(true);
        std::array<std::future<bool>, 3> workers;
        for (size_t i = 0; i < workers.size(); ++i) {
            workers[i] = std::async(std::launch::async, [&] {
                try { p.allocate_large_from_numa(BLOCK_SIZE, 0); return false; }
                catch (const std::exception&) { return true; }
            });
            await_waiters(p, i + 1);
        }
        if (failure) p.run_state().fail(std::make_exception_ptr(std::runtime_error("injected")), "FIFO test");
        else p.set_spill_service(false);
        for (auto& worker : workers) require(worker.get(), "cancelled strict allocation succeeded");
        require(waiter_count(p) == 0, "cancellation or failure left a FIFO node");
    }
    ConcurrentMemoryPool p(32ULL << 20);
    ConcurrentMemoryPoolTestAccess::two(p);
    p.init_arenas();
    const unsigned char record[9]{};
    for (int index = 0; index < 2; ++index) {
        require(p.cache_first_record(index, record), "remote retry fixture failed");
        p.seal_thread_first();
        p.allocate_large_from_numa(p.gap_bytes(index), index);
    }
    p.set_spill_service(true);
    p.freeze_first();
    View local_source;
    require(p.try_claim_first(0, Consumer::Scan, local_source) == Status::Data && local_source.ref.arena_index == 0,
        "local source fixture selected the wrong Arena");
    auto waiting = std::async(std::launch::async, [&] { return p.allocate_large(BLOCK_SIZE); });
    await_waiters(p, 1);
    View remote;
    claim_after_contention(p, 1, Consumer::Scan, remote);
    require(remote.ref.arena_index == 1,
        "remote claim selected the wrong Arena");
    require(p.complete_first(remote), "remote completion failed");
    const bool ready = waiting.wait_for(std::chrono::seconds(3)) == std::future_status::ready;
    p.set_spill_service(false);
    void* result = waiting.get();
    require(ready && ConcurrentMemoryPoolTestAccess::arena(p, 1).contains(result) && !waiter_count(p),
        "registered local waiter did not retry remote Arena or left its node");
    require(p.complete_first(local_source), "local in-flight fixture cleanup failed");
}

static void check_partial_tls_pool_switch()
{
    ConcurrentMemoryPool old_pool(32ULL << 20), new_pool(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::two(old_pool);
    ConcurrentMemoryPoolTestAccess::single(new_pool);
    old_pool.init_arenas(); new_pool.init_arenas();
    std::promise<void> prepared, attempt, retry;
    std::promise<bool> partial;
    auto attempt_signal = attempt.get_future();
    auto retry_signal = retry.get_future();
    auto partial_result = partial.get_future();
    std::thread worker([&] {
        old_pool.deallocate(old_pool.allocate());
        old_pool.deallocate(old_pool.allocate_large_from_numa(BLOCK_SIZE, 1));
        prepared.set_value();
        attempt_signal.wait();
        void* page = new_pool.try_allocate();
        auto& tls = ConcurrentMemoryPool::get_thread_local_cache();
        partial.set_value(!page && tls.pool == &old_pool && !tls.local_free_count && tls.remote_lists[1].count == 1);
        retry_signal.wait();
        if (!new_pool.try_allocate()) std::terminate();
    });
    prepared.get_future().wait();
    std::unique_lock held(ConcurrentMemoryPoolTestAccess::arena(old_pool, 1).mutex);
    attempt.set_value();
    const bool immediate = partial_result.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
    held.unlock();
    const bool transferred = partial_result.get();
    retry.set_value(); worker.join();
    require(immediate && transferred, "partial TLS pool switch lost ownership or waited on a busy Arena");
}

static void check_scan_deficit_includes_fifo_and_inflight()
{
    ConcurrentMemoryPool p(32ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    p.set_spill_service(true);
    const unsigned char record[9]{};
    for (size_t i = 0; i < 4; ++i) {
        require(p.cache_first_record(0, record), "projected-gap fixture failed");
        p.seal_thread_first();
    }
    p.allocate_large_from_numa(p.gap_bytes(0), 0);
    p.freeze_first();
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p);
    Arena::Waiter waiter{FIRST_SEGMENT_SIZE};
    {
        std::lock_guard lock(arena.mutex);
        arena.enqueue(waiter);
        ConcurrentMemoryPoolTestAccess::refresh_locked(p);
    }
    std::array<void*, 1> page;
    require(p.try_reserve_ram_pages(page).status == ConcurrentMemoryPool::RamReserveStatus::NeedReclaim,
        "Scan did not publish its actual deficit");
    View first, second, none;
    require(p.try_claim_first(0, Consumer::Spill, first) == Status::Data, "FIFO deficit did not authorize Spill");
    require(p.try_claim_first(0, Consumer::Spill, second) == Status::Data,
        "Scan counted FIFO-protected space as its own available page");
    require(p.try_claim_first(0, Consumer::Spill, none) == Status::Busy,
        "Spill ignored sufficient projected in-flight reclamation");
    require(p.complete_first(first) && p.complete_first(second), "projected source completion failed");
    require(p.try_reserve_ram_pages(page).status == ConcurrentMemoryPool::RamReserveStatus::Ready,
        "Scan could not use surplus space outside the FIFO reservation");
    p.release_ram_pages(page);
    {
        std::lock_guard lock(arena.mutex);
        arena.remove(waiter);
        ConcurrentMemoryPoolTestAccess::refresh_locked(p);
    }
    p.poll_first_progress(0);
    require(p.try_claim_first(0, Consumer::Spill, none) == Status::Busy, "Ready Scan left frozen watermark Spill enabled");
}

static void check_completion_then_tree_overwrite()
{
    ConcurrentMemoryPool p(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    const unsigned char record[9]{0x6d};
    require(p.cache_first_record(0, record), "overwrite fixture failed");
    p.seal_thread_first();
    p.allocate_large_from_numa(p.gap_bytes(0), 0);
    p.freeze_first();
    View source;
    require(p.try_claim_first(0, Consumer::Scan, source) == Status::Data, "overwrite claim failed");
    std::barrier start(2);
    auto allocator = std::async(std::launch::async, [&] {
        start.arrive_and_wait();
        void* memory;
        while (!(memory = p.try_allocate_large(FIRST_SEGMENT_SIZE))) std::this_thread::yield();
        std::memset(memory, 0x39, FIRST_SEGMENT_SIZE);
        return memory;
    });
    start.arrive_and_wait();
    const auto last_read = *static_cast<const volatile unsigned char*>(source.records);
    require(p.complete_first(source), "overwrite source completion failed");
    require(allocator.get() == source.records && last_read == 0x6d && p.gap_bytes(0) == 0,
        "Tree reused source before the consumer's last read or received the wrong boundary");
    require(!p.complete_first(source), "stale completion refunded overwritten Tree memory");
}

static void check_fifo_handoff_after_middle_remote_success()
{
    ConcurrentMemoryPool p(32ULL << 20);
    ConcurrentMemoryPoolTestAccess::two(p);
    p.init_arenas();
    const unsigned char record[9]{};
    for (size_t i = 0; i < 2; ++i) {
        require(p.cache_first_record(0, record), "FIFO handoff source setup failed");
        p.seal_thread_first();
    }
    p.allocate_large_from_numa(p.gap_bytes(0), 0);
    void* remote_page = p.allocate_large_from_numa(p.gap_bytes(1), 1);
    p.set_spill_service(true);
    std::array<std::future<void*>, 3> workers;
    for (size_t i = 0; i < workers.size(); ++i) {
        workers[i] = std::async(std::launch::async, [&, i] { return p.allocate_large(i == 1 ? BLOCK_SIZE : FIRST_SEGMENT_SIZE); });
        await_waiters(p, i + 1);
    }
    p.release_ram_pages(std::span<void* const>(&remote_page, 1));
    const bool middle_ready = workers[1].wait_for(std::chrono::seconds(3)) == std::future_status::ready;
    if (!middle_ready) p.set_spill_service(false);
    void* middle = workers[1].get();
    require(middle_ready && middle == remote_page && waiter_count(p) == 2, "remote success did not unlink the middle FIFO node");
    p.freeze_first();
    for (size_t index : {0ULL, 2ULL}) {
        View source;
        claim_after_contention(p, 0, Consumer::Spill, source);
        require(p.complete_first(source), "FIFO handoff completion failed");
        const bool ready = workers[index].wait_for(std::chrono::seconds(3)) == std::future_status::ready;
        if (!ready) p.set_spill_service(false);
        void* result = workers[index].get();
        require(ready && result == source.records, "later waiter stole the FIFO head's recovered range");
    }
    require(!waiter_count(p) && p.first_empty(), "successful FIFO handoff stranded a request or source");
}

static void check_first_capacity_denial_until_pressure_changes()
{
    ConcurrentMemoryPool p(32ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    p.init_arenas();
    p.set_spill_service(true);
    std::promise<void> denied, retry, recovered;
    std::promise<bool> repeated, resumed;
    auto retry_signal = retry.get_future();
    auto recovered_signal = recovered.get_future();
    auto repeated_result = repeated.get_future();
    auto resumed_result = resumed.get_future();
    std::thread owner([&] {
        const unsigned char record[9]{};
        while (p.cache_first_record(0, record)) p.seal_thread_first();
        denied.set_value();
        retry_signal.wait();
        repeated.set_value(!p.cache_first_record(0, record));
        recovered_signal.wait();
        resumed.set_value(p.cache_first_record(0, record));
        p.seal_thread_first();
    });
    denied.get_future().wait();
    const auto generation = p.pressure_generation(0);
    auto& arena = ConcurrentMemoryPoolTestAccess::arena(p);
    std::unique_lock held(arena.mutex);
    retry.set_value();
    const bool immediate = repeated_result.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
    held.unlock();
    const bool still_denied = repeated_result.get();
    p.request_first_drain();
    View source;
    while (p.try_claim_first(0, Consumer::Spill, source) == Status::Data) p.complete_first(source);
    const bool changed = p.pressure_generation(0) != generation;
    recovered.set_value();
    const bool admitted = resumed_result.get();
    owner.join();
    require(immediate && still_denied && changed && admitted,
        "unchanged FIRST capacity denial contended again or did not retry after pressure changed");
}

static void check_first_denial_before_initialization()
{
    ConcurrentMemoryPool p(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(p);
    const unsigned char record[9]{};
    require(!p.cache_first_record(0, record), "FIRST admitted before Arena initialization");
    p.init_arenas();
    require(p.cache_first_record(0, record), "pre-initialization capacity denial survived initialized capacity");
    p.seal_thread_first();
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
        check_nonblocking_and_normal_admission();
        check_bounded_prefix_and_generation();
        check_fifo_extra_gap_and_free_pages();
        check_fifo_cancellation_and_remote_retry();
        check_partial_tls_pool_switch();
        check_scan_deficit_includes_fifo_and_inflight();
        check_completion_then_tree_overwrite();
        check_fifo_handoff_after_middle_remote_success();
        check_first_capacity_denial_until_pressure_changes();
        check_first_denial_before_initialization();
        std::cout << "FirstMemoryScan tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
