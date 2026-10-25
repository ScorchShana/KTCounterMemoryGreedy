#include "../src/NewKmerTree.h"
#include "high_prefix_test_helpers.h"
#include <cstddef>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>

struct KmerTreeAllocationTestAccess
{
    template<class T> static void direct(KmerTree<T>& tree, const Task<T>& task)
    { tree.insert_kmer_in_task_to_node_hash_map_without_local_hash_map(task); }

    template<class T> static void scatter_paths();
};

namespace {
using Key = std::vector<unsigned char>;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

template<class T> void layout()
{
    using Node = node<T>;
    using Work = Task<T>;
    static_assert(sizeof(Node) == 256 && alignof(Node) == 64);
    static_assert(sizeof(Node[16]) == 4096);
    static_assert(offsetof(Node, count) == 24);
    static_assert(offsetof(Node, kmer_blocks) == 32);
    static_assert(offsetof(Node, prefix_code) == 160);
    static_assert(offsetof(Node, children_ptr) == 192);
    static_assert(std::is_same_v<decltype(Node::count), uint64_t>);
    static_assert(sizeof(Work) == 192 && alignof(Work) == 64);
    static_assert(offsetof(Work, depth) == 8 && offsetof(Work, count) == 16);
    static_assert(offsetof(Work, kmer_blocks) == 24);
}

void representation()
{
    std::mt19937 random(20261024);
    for (uint32_t k = 1; k <= 128; ++k) {
        k_length = k;
        const kmer ops;
        const size_t full_bytes = (k + 3) / 4;
        for (unsigned sample = 0; sample < 64; ++sample) {
            Key full(full_bytes);
            for (auto& byte : full) byte = random();
            if (k % 4) full.back() &= static_cast<unsigned char>(0xff << (8 - 2 * (k % 4)));
            uint32_t prefix = full[0];
            Key stored(std::max<size_t>(1, full_bytes - 1));
            ops.write_root_batch(stored.data(), full.data(), 1);
            for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth) {
                const size_t cut = std::min(full_bytes, size_t{1} + depth / 2);
                Key expected(full.begin() + cut, full.end());
                if (expected.empty()) expected.push_back(0);
                require(stored == expected, "suffix is not the literal cropped full key");
                require(ops.bytes(depth) == expected.size() && ops.removed_bytes(depth) == cut, "wrong layer width");
                Key restored(full_bytes, 0xaa);
                ops.restore_full_key(restored.data(), stored.data(), prefix, depth);
                require(restored == full, "prefix restoration differs from original full key");
                const size_t start = 4 + 2 * depth;
                uint32_t child = start < k ? (full[start / 4] >> (4 - 2 * (start % 4))) & 15 : 0;
                require(ops.node_prefix(stored.data(), depth) == child, "suffix routing differs from full-key routing");
                if (depth + 1 < MAX_DEPTH) {
                    Key next(ops.bytes(depth + 1), 0xaa);
                    ops.write_child(next.data(), stored.data(), depth);
                    stored = std::move(next);
                    prefix = (prefix << 4) | child;
                }
            }
        }
        const size_t records = (2ULL << 20) / full_bytes;
        require(KmerTree<uint32_t>::first_scan_page_bound(records) <= 772, "FIRST page bound exceeded");
        require(records / (16 * (4092 / ops.bytes(0))) + 256 <= 305, "FIRST Task bound exceeded");
    }
}

template<class T> void tree_path(uint32_t k, uint32_t cap)
{
    k_length = k; count_max = cap;
    count_max_bytes = cap <= 255 ? 1 : cap <= 65535 ? 2 : cap <= 0xffffff ? 3 : 4;
    filter_min = 1; filter_max = UINT32_MAX;
    const kmer ops;
    ConcurrentMemoryPool pool(64ULL << 20); pool.init_arenas();
    ConcurrentOpenAddressHashMap<T>::set_memory_pool(&pool);
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(64, 1);
    auto layers = std::make_unique<LayerQueues<T>>();
    KmerTree<T> tree(&pool, layers.get(), &exports);
    Key hot(kmer_bytes(), 0), cold(kmer_bytes(), 0);
    hot[0] = 0x40; cold[0] = 0x80;
    std::map<Key, uint64_t> truth{{hot, 2 * 16 * (4092 / ops.bytes(0)) + 129}, {cold, 7}};
    if (k > 10) {
        Key other = hot;
        other.back() |= 1 << (8 - 2 * (k % 4 ? k % 4 : 4));
        truth.emplace(other, 3);
    }
    for (const auto& [key, count] : truth) {
        Key batch;
        for (size_t i = 0; i < count; ++i) batch.insert(batch.end(), key.begin(), key.end());
        tree.main_add_kmers_to_local_root(batch.data(), count, key[0], tree.root_nodes);
        auto& root = tree.root_nodes[key[0]];
        require(root.prefix_code == key[0], "root route prefix was not initialized");
        if (root.active_block && root.active_block->count) {
            const size_t cut = std::min(size_t{1}, key.size());
            Key expected(key.begin() + cut, key.end());
            if (expected.empty()) expected.push_back(0);
            require(Key(root.active_block->k_mers.data() + (root.active_block->count - 1) * expected.size(), root.active_block->k_mers.data() + root.active_block->count * expected.size()) == expected,
                "Tree root still stores complete keys");
        }
    }
    bool progress;
    do {
        progress = false;
        for (size_t d = 0; d < MAX_DEPTH; ++d) {
            Task<T> task;
            while (layers->get_queue(d)->try_dequeue(task)) {
                layers->decrease_size(d); tree.thread_add_kmer(task); progress = true;
            }
        }
    } while (progress);
    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> output(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
    {
        FinalDrainWriter writer(&output);
        for (uint32_t root : {0x40u, 0x80u}) tree.final_drain_root(tree.root_nodes + root, writer);
        writer.close();
    }
    std::map<Key, uint64_t> actual, expected;
    for (const auto& [key, count] : truth) expected[key] = std::min<uint64_t>(count, cap);
    content_type content;
    while (output.consumer_try_dequeue(content)) {
        for (const auto& record : high_prefix_test::decode(
                {reinterpret_cast<const uint8_t*>(content.data), content.length}, kmer_bytes(), count_max_bytes)) {
            require(actual.emplace(record.key, record.count).second, "duplicate reconstructed key");
        }
        output.consumer_enqueue(content.data);
    }
    require(actual == expected, "Tree scatter/hash/drain differs from full-key truth");
}

template<class T> void direct_path()
{
    k_length = 33; count_max = 100;
    ConcurrentMemoryPool pool(32ULL << 20); pool.init_arenas();
    ConcurrentOpenAddressHashMap<T>::set_memory_pool(&pool);
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(64, 1);
    auto layers = std::make_unique<LayerQueues<T>>();
    KmerTree<T> tree(&pool, layers.get(), &exports);
    node<T> owner; owner.prefix_code = 0x7cea9;
    Key full{0x7c, 0xea, 0x90, 0x58, 0x22, 0x41, 0x63, 0x77, 0x40};
    Key suffix(full.begin() + 2, full.end());
    Task<T> task{}; task.current_node = &owner; task.depth = MAX_DEPTH - 1; task.count = 1;
    auto* block = static_cast<kmer_block*>(pool.allocate()); block->count = 1;
    std::memcpy(block->k_mers.data(), suffix.data(), suffix.size()); task.kmer_blocks[0] = block;
    KmerTreeAllocationTestAccess::direct(tree, task);
    auto* map = owner.hash_map.load();
    require(map && map->key_bytes() == suffix.size(), "direct insertion used a full-key map");
    size_t visited = 0;
    map->for_each_entry([&](const unsigned char* bytes, uint32_t count) {
        require(Key(bytes, bytes + suffix.size()) == suffix && count == 1, "direct suffix insertion differs");
        ++visited;
    });
    require(visited == 1, "direct suffix entry missing");
}

template<class T> void measure(uint32_t k, bool suffix)
{
    k_length = k;
    const size_t width = suffix ? kmer{}.bytes(MAX_DEPTH - 1) : kmer_bytes();
    ConcurrentMemoryPool pool(32ULL << 20); pool.init_arenas();
    using Map = ConcurrentOpenAddressHashMap<T>;
    Map::set_memory_pool(&pool);
    size_t before = 0, after = 0;
    for (int i = 0; i < pool.arena_count(); ++i) before += pool.gap_bytes(i);
    Map shared(1024, 100, width);
    for (int i = 0; i < pool.arena_count(); ++i) after += pool.gap_bytes(i);
    CountingHashMap local(width);
    const size_t local_bytes = local.keys_.size() + local.controls_.size() + local.counts_.size() * sizeof(uint16_t);
    std::cout << k << ',' << sizeof(T) * 8 << ',' << (suffix ? "suffix" : "full") << ',' << width << ','
        << local.capacity() << ',' << local_bytes << ',' << Map::get_mem_size(1024, width) << ','
        << before - after << ',' << sizeof(node<T>) << '\n';
}

template<class T> void suffix_maps()
{
    k_length = 81; count_max = 100;
    ConcurrentMemoryPool pool(32ULL << 20); pool.init_arenas();
    using Map = ConcurrentOpenAddressHashMap<T>;
    Map::set_memory_pool(&pool);
    CountingHashMap full_local(21);
    Key full_key(21, 0x3a);
    uint16_t full_count = 0;
    for (size_t width : {size_t{1}, size_t{2}, size_t{7}, size_t{19}, size_t{21}}) {
        CountingHashMap local(width);
        Map shared(32, 100, width);
        Key key(width, 0x9c);
        local.increment(key.data()); local.increment(key.data());
        uint64_t unique = 0;
        local.for_each([&](const unsigned char* bytes, uint16_t count) { shared.increment(bytes, T(count), unique); });
        require(local.keys_.size() == local.capacity() * width, "local key array uses full width");
        require(shared.key_bytes() == width, "shared key width differs");
        shared.for_each_entry([&](const unsigned char* bytes, uint32_t count) {
            require(Key(bytes, bytes + width) == key && count == 2, "suffix map corrupted key or count");
        });
        auto* next = shared.ensure_next_map();
        require(next->key_bytes() == width, "next segment lost suffix width");
        next->increment(key.data(), T{255}, unique);
        next->for_each_entry([&](const unsigned char*, uint32_t count) { require(count == 100, "next segment lost count cap"); });
        const size_t align = alignof(std::atomic<T>);
        const size_t offset = (32 * (sizeof(std::atomic<uint8_t>) + width) + align - 1) & ~(align - 1);
        require(Map::counts_offset(32, width) == offset, "suffix count offset differs");
        require(Map::get_mem_size(32, width) == offset + 32 * sizeof(std::atomic<T>), "suffix allocation size differs");
        full_local.increment(full_key.data()); ++full_count;
        full_local.for_each([&](const unsigned char* bytes, uint16_t count) {
            require(Key(bytes, bytes + full_key.size()) == full_key && count == full_count,
                "simultaneous local key widths interfered");
        });
    }
    bool rejected = false;
    try { CountingHashMap invalid(0); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "local zero width accepted");
    rejected = false;
    try { Map invalid(32, 100, 0); } catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "shared zero width accepted");
}
}

// Expectations come from full keys, independently of kmer's suffix helpers.
// Wrong cut/shift/mask, treating every 1 B suffix as empty, leaving scratch
// unchanged, or using bucket starts instead of ends must corrupt these results.
template<class T> void KmerTreeAllocationTestAccess::scatter_paths()
{
    std::mt19937 random(20261025);
    for (uint32_t k = 1; k <= 128; ++k) {
        k_length = k;
        ConcurrentMemoryPool pool(32ULL << 20); pool.init_arenas();
        RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(64, 1);
        auto layers = std::make_unique<LayerQueues<T>>();
        KmerTree<T> tree(&pool, layers.get(), &exports);
        const size_t full_bytes = (k + 3) / 4;
        for (uint32_t depth = 0; depth < 3; ++depth) {
            const size_t parent_cut = std::min(full_bytes, size_t{1} + depth / 2);
            const size_t child_cut = std::min(full_bytes, size_t{1} + (depth + 1) / 2);
            const size_t parent_bytes = std::max(size_t{1}, full_bytes - parent_cut);
            const size_t child_bytes = std::max(size_t{1}, full_bytes - child_cut);
            const size_t parent_capacity = 4092 / parent_bytes;
            const size_t child_capacity = 4092 / child_bytes;
            for (bool drain : {false, true}) for (size_t blocks : {0, 1, 3, 4, 5, 15, 16}) {
                // Empty blocks, mixed partial blocks, then dense single-bucket
                // batches at byte boundaries and representative long keys.
                const bool dense = k <= 11 || k == 81 || k == 128;
                for (unsigned shape = 0; shape < (dense ? 3U : 2U); ++shape) {
                    auto children = std::make_unique<node<T>[]>(16);
                    node<T> parent;
                    parent.children_ptr.store(children.get());
                    Task<T> task{};
                    task.current_node = &parent; task.depth = depth; task.count = blocks;
                    std::array<std::vector<Key>, 16> expected;
                    std::array<std::vector<Key>, 16> last_batch;
                    // Begin one slot short of a full child block; dense batches
                    // also force the 16-block queue/drain-stack transition.
                    if (blocks && shape) {
                        auto& child = children[0];
                        child.count = shape == 2 ? 16 : 1;
                        Key seed(child_bytes, child_cut == full_bytes ? 0 : 0x5a);
                        for (size_t b = 0; b < child.count; ++b) {
                            auto* block = static_cast<kmer_block*>(pool.allocate());
                            block->count = child_capacity - (b + 1 == child.count);
                            for (size_t i = 0; i < block->count; ++i) {
                                std::memcpy(block->k_mers.data() + i * child_bytes, seed.data(), child_bytes);
                                expected[0].push_back(seed);
                            }
                            child.kmer_blocks[b] = block;
                        }
                        child.active_block = child.kmer_blocks[child.count - 1];
                    }
                    for (size_t b = 0; b < blocks; ++b) {
                        auto* block = static_cast<kmer_block*>(pool.allocate());
                        task.kmer_blocks[b] = block;
                        block->count = shape == 0 ? 0 : shape == 2 ? parent_capacity :
                            std::min(parent_capacity, size_t{17} + 37 * b);
                        for (size_t i = 0; i < block->count; ++i) {
                            Key full(full_bytes);
                            for (auto& byte : full) byte = random();
                            full[0] = 0x40;
                            if (depth >= 1 && full_bytes > 1) full[1] &= 0x0f;
                            if (depth >= 2 && full_bytes > 1) full[1] = 0;
                            const size_t start = 4 + 2 * depth;
                            const unsigned shift = 4 - 2 * (start % 4);
                            if (shape == 2 && start < k) full[start / 4] &= ~(15U << shift);
                            if (k % 4) full.back() &= static_cast<unsigned char>(0xff << (8 - 2 * (k % 4)));
                            const unsigned prefix = start < k ? (full[start / 4] >> shift) & 15 : 0;
                            if (parent_cut == full_bytes) block->k_mers[i] = 0;
                            else std::memcpy(block->k_mers.data() + i * parent_bytes,
                                full.data() + parent_cut, parent_bytes);
                            Key suffix(full.begin() + child_cut, full.end());
                            if (suffix.empty()) suffix.push_back(0);
                            expected[prefix].push_back(suffix);
                            if (b >= ((blocks - 1) / 4) * 4) last_batch[prefix].push_back(suffix);
                        }
                    }
                    KmerTree<T>::thread_local_block_for_copy.fill(0xa5);
                    KmerTree<T>::thread_local_block_prefix_counts.fill(0xa5);
                    KmerTree<T>::thread_local_block_prefix_sums.fill(0xa5);
                    std::vector<Task<T>> pending;
                    if (drain) tree.scatter_drain_kmer_blocks_to_children(task, children.get(), pending);
                    else {
                        tree.scatter_kmer_blocks_to_children(task);
                        for (size_t b = 0; b < blocks; ++b) pool.deallocate(task.kmer_blocks[b]);
                    }
                    size_t end = 0;
                    for (size_t prefix = 0; prefix < 16; ++prefix) {
                        const size_t count = last_batch[prefix].size();
                        require(KmerTree<T>::thread_local_block_prefix_counts[prefix] == (blocks ? count : 0xa5),
                            "scatter last-batch counts differ");
                        for (const auto& suffix : last_batch[prefix]) {
                            const auto* stored = KmerTree<T>::thread_local_block_for_copy.data() + end++ * child_bytes;
                            require(Key(stored, stored + child_bytes) == suffix, "scatter scratch suffix differs");
                        }
                        require(KmerTree<T>::thread_local_block_prefix_sums[prefix] == (blocks ? end : 0xa5),
                            "scatter prefix sum is not the bucket end");
                    }
                    if (!blocks || !shape) require(KmerTree<T>::thread_local_block_for_copy[0] == 0xa5,
                        "empty scatter changed scratch");
                    if (!drain) {
                        Task<T> queued;
                        while (layers->get_queue(depth + 1)->try_dequeue(queued)) {
                            layers->decrease_size(depth + 1); pending.push_back(queued);
                        }
                        require(layers->size(depth + 1) == 0, "scatter queue size was not conserved");
                        require(KmerTree<T>::thread_local_task_stack.empty(), "unexpected local queue fallback");
                    }
                    if (blocks && shape == 2) require(!pending.empty(), "full child did not transfer a Task");
                    for (size_t prefix = 0; prefix < 16; ++prefix) {
                        std::vector<Key> actual;
                        auto collect = [&](kmer_block* block) {
                            require(block->count <= child_capacity, "scatter overfilled child block");
                            for (size_t i = 0; i < block->count; ++i) {
                                const auto* bytes = block->k_mers.data() + i * child_bytes;
                                actual.emplace_back(bytes, bytes + child_bytes);
                            }
                            pool.deallocate(block);
                        };
                        for (const auto& queued : pending) if (queued.current_node == children.get() + prefix) {
                            require(queued.depth == depth + 1 && queued.count == 16, "wrong child Task shape");
                            for (size_t b = 0; b < queued.count; ++b) collect(queued.kmer_blocks[b]);
                        }
                        auto& child = children[prefix];
                        for (size_t b = 0; b < child.count; ++b) collect(child.kmer_blocks[b]);
                        require(actual == expected[prefix], "scatter child routing, order or suffix differs");
                        require(child.writer_count.load() == 0, "scatter left an active writer");
                    }
                }
            }
        }
        // The production worker lives as long as its pool; this fixture creates
        // many pools on one thread, so return its cached page before destruction.
        if (KmerTree<T>::thread_local_spare_block) {
            pool.deallocate(KmerTree<T>::thread_local_spare_block);
            KmerTree<T>::thread_local_spare_block = nullptr;
        }
    }
}

int main(int argc, char** argv)
{
    try {
        if (argc == 2 && std::string(argv[1]) == "--measure") {
            std::cout << "k,count_bits,representation,key_bytes,local_slots,local_arrays_bytes,shared_arrays_bytes,shared_allocated_bytes,node_bytes\n";
            for (uint32_t k : {1, 9, 31, 33, 65, 81, 128}) for (bool suffix : {false, true}) {
                measure<uint8_t>(k, suffix); measure<uint16_t>(k, suffix); measure<uint32_t>(k, suffix);
            }
            return 0;
        }
        direct_path<uint8_t>(); direct_path<uint16_t>(); direct_path<uint32_t>();
        layout<uint8_t>(); layout<uint16_t>(); layout<uint32_t>();
        representation();
        KmerTreeAllocationTestAccess::scatter_paths<uint8_t>();
        KmerTreeAllocationTestAccess::scatter_paths<uint16_t>();
        KmerTreeAllocationTestAccess::scatter_paths<uint32_t>();
        suffix_maps<uint8_t>(); suffix_maps<uint16_t>(); suffix_maps<uint32_t>();
        for (uint32_t k : {1, 5, 8, 9, 10, 11, 14, 31, 33, 65, 81, 128}) {
            std::exception_ptr failure;
            std::thread worker([&] {
                try { tree_path<uint8_t>(k, 255); } catch (...) { failure = std::current_exception(); }
            });
            worker.join(); if (failure) std::rethrow_exception(failure);
            std::thread worker16([&] {
                try { tree_path<uint16_t>(k, 1000); } catch (...) { failure = std::current_exception(); }
            });
            worker16.join(); if (failure) std::rethrow_exception(failure);
            std::thread worker32([&] {
                try { tree_path<uint32_t>(k, 65536); } catch (...) { failure = std::current_exception(); }
            });
            worker32.join(); if (failure) std::rethrow_exception(failure);
        }
        std::cout << "Tree suffix representation, layouts and maps passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
