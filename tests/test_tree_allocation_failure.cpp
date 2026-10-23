#include "../src/NewKmerTree.h"
#include "first_pool_fixture.h"
#include <future>
#include <iostream>
#include <stdexcept>

struct KmerTreeAllocationTestAccess
{
    static bool building(ConcurrentOpenAddressHashMap& map) { return map.building_next.load(); }
    static void child(KmerTree& tree, node& parent) { tree.ensure_child_slab(&parent); }
    static void map(KmerTree& tree, node& parent) { (void)tree.ensure_hash_map(&parent, 1024); }
};

static void check_source_return_before_queue_wait()
{
    ConcurrentMemoryPool pool(32ULL << 20);
    pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> output(64, 1);
    auto layers = std::make_unique<LayerQueues>();
    KmerTree tree(&pool, layers.get(), &output);
    const size_t count = pool.first_segment_record_capacity();
    std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 0);
    if (!cache_full_first(pool, 0, records.data())) throw std::runtime_error("queue-wait source fixture failed");
    pool.freeze_first();
    KmerTree::FirstScanBatch batch;
    if (tree.try_prepare_first_scan(batch, count) != ConcurrentMemoryPool::RamReserveStatus::Ready)
        throw std::runtime_error("queue-wait target preparation failed");
    ConcurrentMemoryPool::FirstSegmentView source;
    if (pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Scan, source)
        != ConcurrentMemoryPool::FirstClaimStatus::Data) throw std::runtime_error("queue-wait claim failed");
    auto* queue = layers->get_queue(0);
    Task empty{};
    size_t filled = 0;
    while (queue->try_enqueue(empty)) { layers->increase_size(0); ++filled; }
    std::array<node, 256> roots{};
    tree.append_first_scan_hits(batch, source.records, source.record_count, roots.data());
    if (layers->size(0) != static_cast<int64_t>(filled) || !pool.complete_first(source))
        throw std::runtime_error("scan published Tasks or failed to return source with full queue");
    const size_t bytes = pool.gap_bytes(0);
    std::memset(pool.allocate_large_from_numa(bytes, 0), 0x39, bytes);
    auto submission = std::async(std::launch::async, [&] { tree.submit_first_scan(batch); });
    while (layers->size(0) == static_cast<int64_t>(filled)) std::this_thread::yield();
    const bool waited = submission.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
    Task task;
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
int main()
{
    k_length = 33;
    check_source_return_before_queue_wait();
    mapping_filename = "/dev/null"; // Exclusive creation fails after RAM exhaustion.
    ConcurrentMemoryPool pool(16ULL << 20);
    pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> output(64, 1);
    auto layers = std::make_unique<LayerQueues>();
    KmerTree tree(&pool, layers.get(), &output);
    {
        KmerTree::FirstScanBatch batch;
        const auto count = ConcurrentMemoryPool::first_segment_record_capacity();
        if (tree.try_prepare_first_scan(batch, count) != ConcurrentMemoryPool::RamReserveStatus::Ready) return 1;
        std::array<node, 256> prepared_roots{};
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
        Task task;
        while (layers->get_queue(0)->try_dequeue(task)) for (size_t b = 0; b < task.count; ++b) {
            seen += task.kmer_blocks[b]->count;
            pool.deallocate(task.kmer_blocks[b]);
        }
        tree.release_unused_first_scan_pages(batch);
        if (seen != count) return 1;
    }
    std::array<node, 256> local_roots{};
    const unsigned char key[9]{};
    const size_t before = pool.gap_bytes(0);
    if (tree.try_accept_first_batch(key, 1, 0, local_roots.data()) != 0 || pool.gap_bytes(0) != before) return 1;
    tree.main_add_kmers_to_local_root(key, 1, 0, local_roots.data());
    const size_t after = pool.gap_bytes(0);
    if (tree.try_accept_first_batch(key, 1, 0, local_roots.data()) != 1 || pool.gap_bytes(0) != after) return 1;
    ConcurrentOpenAddressHashMap::set_memory_pool(&pool);
    ConcurrentOpenAddressHashMap map(1024);
    while (pool.try_allocate()) {}
    node child_parent, map_parent;
    try { KmerTreeAllocationTestAccess::child(tree, child_parent); } catch (const std::bad_alloc&) {}
    if (child_parent.children_ptr.load() != nullptr) { std::cerr << "child sentinel stranded\n"; return 1; }
    try { KmerTreeAllocationTestAccess::map(tree, map_parent); } catch (const std::bad_alloc&) {}
    if (map_parent.hash_map.load() != nullptr) { std::cerr << "map sentinel stranded\n"; return 1; }
    try { map.ensure_next_map(); } catch (const std::bad_alloc&) {}
    if (KmerTreeAllocationTestAccess::building(map)) { std::cerr << "building_next stranded\n"; return 1; }
    std::cout << "Tree allocation failure tests passed\n";
}
