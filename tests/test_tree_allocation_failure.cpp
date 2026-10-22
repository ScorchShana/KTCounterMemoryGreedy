#include "../src/NewKmerTree.h"
#include <iostream>
#include <stdexcept>

struct KmerTreeAllocationTestAccess
{
    static bool building(ConcurrentOpenAddressHashMap& map) { return map.building_next.load(); }
    static void child(KmerTree& tree, node& parent) { tree.ensure_child_slab(&parent); }
    static void map(KmerTree& tree, node& parent) { (void)tree.ensure_hash_map(&parent, 1024); }
};
int main()
{
    k_length = 33;
    mapping_filename = "/dev/null"; // Exclusive creation fails after RAM exhaustion.
    ConcurrentMemoryPool pool(16ULL << 20);
    pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> output(64, 1);
    auto layers = std::make_unique<LayerQueues>();
    KmerTree tree(&pool, layers.get(), &output);
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
