#include "../src/NewKmerTree.h"
#include "first_pool_fixture.h"
#include <barrier>
#include <future>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

struct ConcurrentMemoryPoolTestAccess
{
    static uint64_t scan_goal(ConcurrentMemoryPool& pool) { return pool.arenas_[0].scan_reclaim_goal_pages.load(); }
};

struct KmerTreeAllocationTestAccess
{
    static bool building(ConcurrentOpenAddressHashMap<uint32_t>& map) { return map.building_next.load(); }
    static void child(KmerTree<uint32_t>& tree, node<uint32_t>& parent) { tree.ensure_child_slab(&parent); }
    static void map(KmerTree<uint32_t>& tree, node<uint32_t>& parent) { (void)tree.ensure_hash_map(&parent, 1024); }
};

static void check_source_return_before_queue_wait()
{
    ConcurrentMemoryPool pool(32ULL << 20);
    pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> output(64, 1);
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &output);
    const size_t count = pool.first_segment_record_capacity();
    std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 0);
    if (!cache_full_first(pool, 0, records.data())) throw std::runtime_error("queue-wait source fixture failed");
    pool.freeze_first();
    KmerTree<uint32_t>::FirstScanBatch batch;
    if (tree.try_prepare_first_scan(batch, count) != ConcurrentMemoryPool::RamReserveStatus::Ready)
        throw std::runtime_error("queue-wait target preparation failed");
    ConcurrentMemoryPool::FirstSegmentView source;
    if (pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Scan, source)
        != ConcurrentMemoryPool::FirstClaimStatus::Data) throw std::runtime_error("queue-wait claim failed");
    auto* queue = layers->get_queue(0);
    Task<uint32_t> empty{};
    size_t filled = 0;
    while (queue->try_enqueue(empty)) { layers->increase_size(0); ++filled; }
    std::array<node<uint32_t>, 256> roots{};
    KmerTree<uint32_t>::initialize_roots(roots.data());
    tree.append_first_scan_hits(batch, source.records, source.record_count, roots.data());
    if (layers->size(0) != static_cast<int64_t>(filled) || !pool.complete_first(source))
        throw std::runtime_error("scan published Tasks or failed to return source with full queue");
    const size_t bytes = pool.gap_bytes(0);
    std::memset(pool.allocate_large_from_numa(bytes, 0), 0x39, bytes);
    auto submission = std::async(std::launch::async, [&] { tree.submit_first_scan(batch); });
    while (layers->size(0) == static_cast<int64_t>(filled)) std::this_thread::yield();
    const bool waited = submission.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
    Task<uint32_t> task;
    for (size_t i = 0; i < filled; ++i) {
        while (!queue->try_dequeue(task)) std::this_thread::yield();
        layers->decrease_size(0);
        if (task.count) throw std::runtime_error("queue-wait sentinel was reordered");
    }
    submission.get();
    size_t seen = 0;
    while (queue->try_dequeue(task)) {
        layers->decrease_size(0);
        for (size_t b = 0; b < task.count; ++b) {
            seen += task.kmer_blocks[b]->count;
            pool.deallocate(task.kmer_blocks[b]);
        }
    }
    for (auto& root : roots) for (size_t b = 0; b < root.count; ++b) {
        seen += root.kmer_blocks[b]->count;
        pool.deallocate(root.kmer_blocks[b]);
    }
    tree.release_unused_first_scan_pages(batch);
    if (!waited || seen != count || !pool.first_empty() || layers->size(0))
        throw std::runtime_error("queue-wait handoff lost source data or Task ownership");
}
static void check_single_scan_preparer()
{
    ConcurrentMemoryPool pool(16ULL << 20);
    pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> output(64, 1);
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &output);
    const unsigned char record[9]{};
    if (!pool.cache_first_record(0, record)) throw std::runtime_error("preparer FIRST setup failed");
    pool.seal_thread_first();
    pool.allocate_large_from_numa(pool.gap_bytes(0) - BLOCK_SIZE, 0);
    using Status = ConcurrentMemoryPool::RamReserveStatus;
    KmerTree<uint32_t>::FirstScanBatch follower;
    {
        KmerTree<uint32_t>::FirstScanBatch owner;
        if (tree.try_prepare_first_scan(owner, 1) != Status::NeedReclaim)
            throw std::runtime_error("preparer shortage setup failed");
        if (tree.try_prepare_first_scan(follower, 1) != Status::Busy)
            throw std::runtime_error("second incomplete Scan accumulated pages");
        if (tree.try_prepare_first_scan(owner, 1) != Status::NeedReclaim)
            throw std::runtime_error("preparer could not retry its deficit");
    }
    // Destruction returns partial pages centrally and releases preparation.
    pool.freeze_first();
    pool.set_spill_service(true);
    ConcurrentMemoryPool::FirstSegmentView view;
    if (pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Spill, view)
        == ConcurrentMemoryPool::FirstClaimStatus::Data)
        throw std::runtime_error("destroyed preparer left a stale spill goal");
    if (tree.try_prepare_first_scan(follower, 1) != Status::NeedReclaim)
        throw std::runtime_error("destroyed preparer stranded preparation ownership");
    if (pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Spill, view)
        != ConcurrentMemoryPool::FirstClaimStatus::Data || !pool.complete_first(view))
        throw std::runtime_error("Scan deficit did not authorize Spill");
    if (tree.try_prepare_first_scan(follower, 1) != Status::Ready)
        throw std::runtime_error("partial Scan did not reach Ready after reclamation");
    tree.release_unused_first_scan_pages(follower);
}
static void check_parallel_scan_preparers_and_failure()
{
    for (bool fail : {false, true}) {
        ConcurrentMemoryPool pool(16ULL << 20);
        pool.init_arenas();
        RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> output(64, 1);
        auto layers = std::make_unique<LayerQueues<uint32_t>>();
        KmerTree<uint32_t> tree(&pool, layers.get(), &output);
        const unsigned char record[9]{};
        if (!pool.cache_first_record(0, record)) throw std::runtime_error("parallel preparer setup failed");
        pool.seal_thread_first();
        pool.allocate_large_from_numa(pool.gap_bytes(0) - BLOCK_SIZE, 0);
        ConcurrentMemoryPool::get_thread_local_cache().release();
        std::barrier start(5), prepared(5), cleanup(5);
        std::array<ConcurrentMemoryPool::RamReserveStatus, 4> statuses;
        std::array<std::thread, 4> workers;
        std::atomic<size_t> failures{0};
        for (size_t i = 0; i < workers.size(); ++i) workers[i] = std::thread([&, i] {
            KmerTree<uint32_t>::FirstScanBatch batch;
            start.arrive_and_wait();
            statuses[i] = tree.try_prepare_first_scan(batch, pool.first_segment_record_capacity());
            prepared.arrive_and_wait();
            cleanup.arrive_and_wait();
            if (fail) {
                try { tree.try_prepare_first_scan(batch, pool.first_segment_record_capacity()); }
                catch (const std::runtime_error&) { ++failures; }
            }
        });
        start.arrive_and_wait();
        prepared.arrive_and_wait();
        const size_t owners = std::count(statuses.begin(), statuses.end(), ConcurrentMemoryPool::RamReserveStatus::NeedReclaim);
        const size_t busy = std::count(statuses.begin(), statuses.end(), ConcurrentMemoryPool::RamReserveStatus::Busy);
        if (fail) pool.run_state().fail(std::make_exception_ptr(std::runtime_error("injected Scan failure")), "test");
        cleanup.arrive_and_wait();
        for (auto& worker : workers) worker.join();
        if (owners != 1 || busy != 3 || ConcurrentMemoryPoolTestAccess::scan_goal(pool) != 0
            || (fail && failures != 4)) throw std::runtime_error("parallel Scan preparation or failure cleanup lost ownership");
        if (!fail) {
            KmerTree<uint32_t>::FirstScanBatch next;
            if (tree.try_prepare_first_scan(next, pool.first_segment_record_capacity())
                != ConcurrentMemoryPool::RamReserveStatus::NeedReclaim)
                throw std::runtime_error("parallel preparer destruction stranded its qualification");
        }
    }
}
static void check_reduced_scan_bound_releases_preparer()
{
    ConcurrentMemoryPool pool(16ULL << 20);
    pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> output(64, 1);
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &output);
    const unsigned char record[9]{};
    if (!pool.cache_first_record(0, record)) throw std::runtime_error("reduced-bound setup failed");
    pool.seal_thread_first();
    pool.allocate_large_from_numa(pool.gap_bytes(0) - 300 * BLOCK_SIZE, 0);
    using Status = ConcurrentMemoryPool::RamReserveStatus;
    KmerTree<uint32_t>::FirstScanBatch owner, follower;
    if (tree.try_prepare_first_scan(owner, pool.first_segment_record_capacity()) != Status::NeedReclaim
        || tree.try_prepare_first_scan(owner, 1) != Status::Ready)
        throw std::runtime_error("reduced-bound fixture was not partially prepared");
    if (ConcurrentMemoryPoolTestAccess::scan_goal(pool) != 0
        || tree.try_prepare_first_scan(follower, 1) == Status::Busy)
        throw std::runtime_error("smaller Ready bound left stale Scan demand or preparation ownership");
    tree.release_unused_first_scan_pages(owner);
    if (ConcurrentMemoryPoolTestAccess::scan_goal(pool) == 0)
        throw std::runtime_error("Ready batch cleanup cancelled another preparer's deficit");
}
int main()
{
    k_length = 33;
    check_reduced_scan_bound_releases_preparer();
    check_single_scan_preparer();
    check_parallel_scan_preparers_and_failure();
    check_source_return_before_queue_wait();
    const auto failure_path = std::filesystem::temp_directory_path() / "tree-allocation-failure-existing";
    std::ofstream(failure_path).put(0);
    mapping_filename = failure_path.string(); // Exclusive creation fails after RAM exhaustion.
    ConcurrentMemoryPool pool(16ULL << 20);
    pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> output(64, 1);
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &output);
    {
        KmerTree<uint32_t>::FirstScanBatch batch;
        const auto count = ConcurrentMemoryPool::first_segment_record_capacity();
        if (tree.try_prepare_first_scan(batch, count) != ConcurrentMemoryPool::RamReserveStatus::Ready) return 1;
        std::array<node<uint32_t>, 256> prepared_roots{};
        KmerTree<uint32_t>::initialize_roots(prepared_roots.data());
        std::vector<unsigned char> records(count * kmer_bytes());
        for (size_t i = 0; i < count; ++i) records[i * kmer_bytes()] = i % 256;
        const size_t before_scan = pool.gap_bytes(0);
        tree.append_first_scan_hits(batch, records.data(), count, prepared_roots.data());
        if (pool.gap_bytes(0) != before_scan || !layers->get_queue(0)->empty()) return 1;
        tree.submit_first_scan(batch);
        size_t seen = 0;
        for (auto& root : prepared_roots) for (size_t b = 0; b < root.count; ++b) {
            seen += root.kmer_blocks[b]->count;
            pool.deallocate(root.kmer_blocks[b]);
        }
        Task<uint32_t> task;
        while (layers->get_queue(0)->try_dequeue(task)) for (size_t b = 0; b < task.count; ++b) {
            seen += task.kmer_blocks[b]->count;
            pool.deallocate(task.kmer_blocks[b]);
        }
        tree.release_unused_first_scan_pages(batch);
        if (seen != count) return 1;
    }
    std::array<node<uint32_t>, 256> local_roots{};
    KmerTree<uint32_t>::initialize_roots(local_roots.data());
    const unsigned char key[9]{};
    const size_t before = pool.gap_bytes(0);
    if (tree.try_accept_first_batch(key, 1, 0, local_roots.data()) != 0 || pool.gap_bytes(0) != before) return 1;
    tree.main_add_kmers_to_local_root(key, 1, 0, local_roots.data());
    const size_t after = pool.gap_bytes(0);
    if (tree.try_accept_first_batch(key, 1, 0, local_roots.data()) != 1 || pool.gap_bytes(0) != after) return 1;
    ConcurrentOpenAddressHashMap<uint32_t>::set_memory_pool(&pool);
    ConcurrentOpenAddressHashMap<uint32_t> map(1024);
    while (pool.try_allocate()) {}
    node<uint32_t> child_parent, map_parent;
    try { KmerTreeAllocationTestAccess::child(tree, child_parent); } catch (const std::bad_alloc&) {}
    if (child_parent.children_ptr.load() != nullptr) { std::cerr << "child sentinel stranded\n"; return 1; }
    try { KmerTreeAllocationTestAccess::map(tree, map_parent); } catch (const std::bad_alloc&) {}
    if (map_parent.hash_map.load() != nullptr) { std::cerr << "map sentinel stranded\n"; return 1; }
    try { map.ensure_next_map(); } catch (const std::bad_alloc&) {}
    if (KmerTreeAllocationTestAccess::building(map)) { std::cerr << "building_next stranded\n"; return 1; }
    std::cout << "Tree allocation failure tests passed\n";
}
