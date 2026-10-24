#include "../src/NewKmerTree.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#include <type_traits>

struct KmerTreeAllocationTestAccess
{
    template<typename CountType>
    static size_t block_capacity() { return KmerTree<CountType>::get_block_capacity(); }

    template<typename CountType>
    static void export_map(KmerTree<CountType>& tree, FinalDrainWriter& writer,
        ConcurrentOpenAddressHashMap<CountType>* map)
    { tree.export_hash_map(writer, map); }

    template<typename CountType>
    static void export_leaf(KmerTree<CountType>& tree, node<CountType>* leaf, FinalDrainWriter& writer)
    { tree.sort_and_export_leaf(leaf, writer); }
};

namespace {
using Key = std::vector<unsigned char>;
using Counts = std::map<Key, uint64_t>;
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

template<typename CountType>
void check(uint32_t k, uint32_t cap)
{
    static_assert(std::is_same_v<decltype(CountingHashMap::counts_)::value_type, uint16_t>);
    static_assert(sizeof(node<CountType>) == 256 && alignof(node<CountType>) == 64);
    static_assert(sizeof(node<CountType>[16]) == 4096);
    static_assert(sizeof(Task<CountType>) == 192 && alignof(Task<CountType>) == 64);
    static_assert(sizeof(kmer_block) == 4096 && alignof(kmer_block) == 4);
    k_length = k;
    count_max = cap;
    count_max_bytes = cap <= 255 ? 1 : cap <= 65535 ? 2 : cap <= 0xffffff ? 3 : 4;
    filter_min = 1;
    filter_max = UINT32_MAX;
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(64, 1);
    auto layers = std::make_unique<LayerQueues<CountType>>();
    KmerTree<CountType> tree(&pool, layers.get(), &exports);
    node<CountType> parent;
    Key hot(kmer_bytes(), 0), cold(kmer_bytes(), 0);
    cold[0] = 0x40;
    const size_t capacity = KmerTreeAllocationTestAccess::block_capacity<CountType>();
    constexpr size_t workers = 4, repeats = 20000, others = 7;
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&] {
            ConcurrentOpenAddressHashMap<CountType>::set_memory_pool(&pool);
            KmerTree<CountType>::initialize_worker();
            for (size_t pos = 0; pos < repeats + others;) {
                Task<CountType> task{};
                task.current_node = &parent;
                task.depth = MAX_DEPTH - 1;
                while (task.count < MAX_KMER_BLOCK_NUM && pos < repeats + others) {
                    auto* block = static_cast<kmer_block*>(pool.allocate());
                    block->count = std::min(capacity, repeats + others - pos);
                    for (size_t i = 0; i < block->count; ++i, ++pos) {
                        const auto& key = pos < repeats ? hot : cold;
                        std::memcpy(block->k_mers.data() + i * key.size(), key.data(), key.size());
                    }
                    task.kmer_blocks[task.count++] = block;
                }
                tree.thread_add_kmer(task);
            }
        });
    }
    for (auto& thread : threads) thread.join();
    auto* map = parent.hash_map.load();
    require(map, "real Task merge did not create a shared map");
    // Ground truth is calculated in uint64_t independently of the counter width.
    const Counts truth{{hot, workers * repeats}, {cold, workers * others}};
    Counts expected, actual;
    for (const auto& [key, count] : truth) expected[key] = std::min<uint64_t>(count, cap);
    map->for_each_entry([&](const unsigned char* key, uint32_t count) {
        require(actual.emplace(Key(key, key + kmer_bytes()), count).second, "duplicate shared key");
    });
    require(actual == expected, "multi-worker Task merge lost or narrowed counts");

    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> output(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
    for (uint32_t minimum : {1u, cap, cap == UINT32_MAX ? cap : cap + 1}) {
        filter_min = minimum;
        filter_max = cap;
        Counts reference, hashed, leaf_counts;
        const Counts leaf_truth{{hot, 270}, {cold, 3}};
        for (const auto& [key, count] : leaf_truth) {
            const uint64_t saturated = std::min<uint64_t>(count, cap);
            if (minimum <= saturated && saturated <= filter_max) reference[key] = saturated;
        }
        ConcurrentOpenAddressHashMap<CountType>::set_memory_pool(&pool);
        ConcurrentOpenAddressHashMap<CountType> small_map(16);
        uint64_t unique = 0;
        for (const auto& [key, count] : leaf_truth) {
            const auto value = static_cast<CountType>(std::min<uint64_t>(count, std::numeric_limits<CountType>::max()));
            small_map.increment(key.data(), value, unique);
        }
        auto collect = [&](Counts& result) {
            content_type content;
            const size_t width = kmer_bytes() + count_max_bytes;
            while (output.consumer_try_dequeue(content)) {
                require(content.length % width == 0, "partial exported record");
                for (size_t pos = 0; pos < content.length; pos += width) {
                    const auto* key = reinterpret_cast<const unsigned char*>(content.data + pos);
                    uint32_t count = 0;
                    std::memcpy(&count, key + kmer_bytes(), count_max_bytes);
                    require(result.emplace(Key(key, key + kmer_bytes()), count).second, "duplicate export");
                }
                output.consumer_enqueue(content.data);
            }
        };
        {
            FinalDrainWriter writer(&output);
            KmerTreeAllocationTestAccess::export_map(tree, writer, &small_map);
            writer.close();
        }
        collect(hashed);
        node<CountType> leaf;
        size_t used = 0;
        for (const auto& [key, count] : leaf_truth) {
            for (size_t i = 0; i < count; ++i) {
                if (!used || used == capacity) {
                    auto* block = static_cast<kmer_block*>(pool.allocate());
                    block->count = 0;
                    leaf.kmer_blocks[leaf.count++] = leaf.active_block = block;
                    used = 0;
                }
                std::memcpy(leaf.active_block->k_mers.data() + used * kmer_bytes(), key.data(), kmer_bytes());
                ++used;
                ++leaf.active_block->count;
            }
        }
        {
            FinalDrainWriter writer(&output);
            KmerTreeAllocationTestAccess::export_leaf(tree, &leaf, writer);
            writer.close();
        }
        collect(leaf_counts);
        require(hashed == reference && leaf_counts == reference, "leaf/hash saturation or output filtering differs");
    }
}
}

int main()
{
    try {
        for (uint32_t k : {1u, 31u, 33u, 65u, 128u}) {
            for (uint32_t cap : {1u, 4u, 100u, 255u}) check<uint8_t>(k, cap);
            for (uint32_t cap : {256u, 1000u, 65535u}) check<uint16_t>(k, cap);
            for (uint32_t cap : {65536u, 0xffffffu, 0x1000000u, UINT32_MAX}) check<uint32_t>(k, cap);
        }
        std::cout << "Hash count type tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
