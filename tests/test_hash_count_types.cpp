#include "../src/NewKmerTree.h"
#include "high_prefix_test_helpers.h"

#include <iostream>
#include <map>
#include <stdexcept>
#include <thread>
#include <type_traits>

struct KmerTreeAllocationTestAccess
{
    template<typename T>
    static void export_map(KmerTree<T>& tree, FinalDrainWriter& writer,
        node<T>* owner, uint32_t depth, ConcurrentOpenAddressHashMap<T>* map)
    { tree.export_hash_map(writer, owner, depth, map); }

    template<typename T>
    static void export_leaf(KmerTree<T>& tree, node<T>* leaf, uint32_t depth, FinalDrainWriter& writer)
    { tree.sort_and_export_leaf(leaf, depth, writer); }
};

namespace {
using Key = std::vector<unsigned char>;
using Counts = std::map<Key, uint64_t>;
void require(bool ok, const char* message)
{ if (!ok) throw std::runtime_error(message); }

Key suffix(const Key& key, uint32_t depth)
{
    const size_t cut = std::min(key.size(), size_t{1} + depth / 2);
    Key result(key.begin() + cut, key.end());
    if (result.empty()) result.push_back(0);
    return result;
}
uint32_t prefix(const Key& key, uint32_t depth)
{
    uint32_t result = key[0];
    for (uint32_t d = 0; d < depth; ++d) {
        const uint32_t start = 4 + 2 * d;
        result = (result << 4) | (start < k_length ? (key[start / 4] >> (4 - 2 * (start % 4))) & 15 : 0);
    }
    return result;
}

template<typename T> void check(uint32_t k, uint32_t cap)
{
    static_assert(std::is_same_v<decltype(CountingHashMap::counts_)::value_type, uint16_t>);
    static_assert(sizeof(node<T>) == 256 && alignof(node<T>) == 64);
    static_assert(sizeof(node<T>[16]) == 4096);
    static_assert(sizeof(Task<T>) == 192 && alignof(Task<T>) == 64);
    static_assert(sizeof(kmer_block) == 4096 && alignof(kmer_block) == 4);
    k_length = k; count_max = cap;
    count_max_bytes = cap <= 255 ? 1 : cap <= 65535 ? 2 : cap <= 0xffffff ? 3 : 4;
    filter_min = 1; filter_max = UINT32_MAX;
    ConcurrentMemoryPool pool(64ULL << 20); pool.init_arenas();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports(64, 1);
    auto layers = std::make_unique<LayerQueues<T>>();
    KmerTree<T> tree(&pool, layers.get(), &exports);
    Key hot(kmer_bytes(), 0), cold(kmer_bytes(), 0); cold[0] = 0x40;
    const std::array<Key, 2> keys{hot, cold};
    std::array<node<T>, 2> parents{};
    for (size_t n = 0; n < 2; ++n) parents[n].prefix_code = prefix(keys[n], MAX_DEPTH - 1);
    const size_t width = suffix(hot, MAX_DEPTH - 1).size();
    const size_t capacity = 4092 / width;
    constexpr size_t workers = 4, repeats = 20000, others = 7;
    std::vector<std::thread> threads;
    for (size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&] {
            ConcurrentOpenAddressHashMap<T>::set_memory_pool(&pool);
            KmerTree<T>::initialize_worker();
            for (size_t n = 0; n < 2; ++n) {
                const Key bytes = suffix(keys[n], MAX_DEPTH - 1);
                const size_t total = n == 0 ? repeats : others;
                for (size_t pos = 0; pos < total;) {
                    Task<T> task{}; task.current_node = &parents[n]; task.depth = MAX_DEPTH - 1;
                    while (task.count < MAX_KMER_BLOCK_NUM && pos < total) {
                        auto* block = static_cast<kmer_block*>(pool.allocate());
                        block->count = std::min(capacity, total - pos);
                        for (size_t i = 0; i < block->count; ++i)
                            std::memcpy(block->k_mers.data() + i * width, bytes.data(), width);
                        pos += block->count; task.kmer_blocks[task.count++] = block;
                    }
                    tree.thread_add_kmer(task);
                }
            }
        });
    }
    for (auto& thread : threads) thread.join();
    const Counts truth{{hot, workers * repeats}, {cold, workers * others}};
    Counts expected, actual;
    for (const auto& [key, count] : truth) expected[key] = std::min<uint64_t>(count, cap);
    for (size_t n = 0; n < 2; ++n) {
        auto* map = parents[n].hash_map.load();
        require(map && map->key_bytes() == width, "real Task merge did not create a suffix map");
        const Key bytes = suffix(keys[n], MAX_DEPTH - 1);
        map->for_each_entry([&](const unsigned char* key, uint32_t count) {
            require(Key(key, key + width) == bytes, "shared map retained full key bytes");
            require(actual.emplace(keys[n], count).second, "duplicate shared key");
        });
    }
    require(actual == expected, "multi-worker Task merge lost or narrowed counts");

    ConcurrentOpenAddressHashMap<T>::set_memory_pool(&pool);
    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> output(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
    for (uint32_t minimum : {1u, cap, cap == UINT32_MAX ? cap : cap + 1}) {
        filter_min = minimum; filter_max = cap;
        const Counts leaf_truth{{hot, 270}, {cold, 3}};
        Counts reference, hashed;
        std::array<Counts, 2> sorted;
        for (const auto& [key, count] : leaf_truth) {
            const auto saturated = std::min<uint64_t>(count, cap);
            reference[key] = saturated;
        }
        auto collect = [&](Counts& result, uint32_t depth) {
            content_type content;
            while (output.consumer_try_dequeue(content)) {
                for (const auto& record : high_prefix_test::decode(
                        {reinterpret_cast<const uint8_t*>(content.data), content.length}, kmer_bytes(), count_max_bytes)) {
                    require(record.prefix_bytes == std::min(size_t{1} + depth / 2, kmer_bytes() - 1),
                        "export did not use the node's complete prefix bytes");
                    require(result.emplace(record.key, record.count).second, "duplicate export");
                }
                output.consumer_enqueue(content.data);
            }
        };
        for (const auto& [key, count] : leaf_truth) {
            const Key bytes = suffix(key, MAX_DEPTH - 1);
            node<T> owner; owner.prefix_code = prefix(key, MAX_DEPTH - 1);
            ConcurrentOpenAddressHashMap<T> small_map(16, cap, bytes.size());
            uint64_t unique = 0;
            const T value = static_cast<T>(std::min<uint64_t>(count, std::numeric_limits<T>::max()));
            small_map.increment(bytes.data(), value, unique);
            { FinalDrainWriter writer(&output);
              KmerTreeAllocationTestAccess::export_map(tree, writer, &owner, MAX_DEPTH - 1, &small_map); writer.close(); }
            collect(hashed, MAX_DEPTH - 1);
            size_t which = 0;
            for (uint32_t depth : {0u, MAX_DEPTH - 1}) {
                node<T> leaf; leaf.prefix_code = prefix(key, depth);
                const Key stored = suffix(key, depth);
                const size_t block_capacity = 4092 / stored.size();
                for (size_t i = 0; i < count; ++i) {
                    if (!leaf.active_block || leaf.active_block->count == block_capacity) {
                        auto* block = static_cast<kmer_block*>(pool.allocate()); block->count = 0;
                        leaf.kmer_blocks[leaf.count++] = leaf.active_block = block;
                    }
                    std::memcpy(leaf.active_block->k_mers.data() + leaf.active_block->count++ * stored.size(), stored.data(), stored.size());
                }
                { FinalDrainWriter writer(&output);
                  KmerTreeAllocationTestAccess::export_leaf(tree, &leaf, depth, writer); writer.close(); }
                collect(sorted[which++], depth);
            }
        }
        require(hashed == reference && sorted[0] == reference && sorted[1] == reference,
            "suffix leaf/hash reconstruction, saturation or filtering differs");
    }
    // Exercise cap boundaries without materializing billions of occurrences.
    for (uint64_t frequency : {uint64_t(cap) - 1, uint64_t(cap), uint64_t(cap) + 1}) {
        for (uint64_t low : {0ULL, 1ULL}) {
            if (!frequency || low > frequency) continue;
            const Key bytes = suffix(hot, MAX_DEPTH - 1);
            node<T> owner; owner.prefix_code = prefix(hot, MAX_DEPTH - 1);
            ConcurrentOpenAddressHashMap<T> map(16, cap, bytes.size());
            uint64_t unique = 0;
            for (uint64_t left = frequency - low; left;) {
                const auto take = static_cast<T>(std::min<uint64_t>(left, std::numeric_limits<T>::max()));
                map.increment(bytes.data(), take, unique);
                left -= take;
            }
            { FinalDrainWriter writer(&output);
              KmerTreeAllocationTestAccess::export_map(tree, writer, &owner, MAX_DEPTH - 1, &map); writer.close(); }
            uint64_t high = 0, records = 0;
            content_type content;
            while (output.consumer_try_dequeue(content)) {
                for (const auto& record : high_prefix_test::decode(
                        {reinterpret_cast<const uint8_t*>(content.data), content.length}, kmer_bytes(), count_max_bytes)) {
                    require(record.key == hot, "cap-boundary export changed the key");
                    high = record.count;
                    ++records;
                }
                output.consumer_enqueue(content.data);
            }
            require(records == (frequency > low ? 1u : 0u) && high == std::min<uint64_t>(cap, frequency - low),
                "cap-boundary high remainder differs");
            require(std::min<uint64_t>(cap, high + low) == std::min<uint64_t>(cap, frequency),
                "split count does not reconstruct the capped frequency");
        }
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
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
