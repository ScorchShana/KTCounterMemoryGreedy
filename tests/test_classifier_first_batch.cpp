#include "../src/FastqClassifier.h"
#include "../src/ConcurrentFirstBlockRing.h"
#include "../src/FastqParser.h"
#include "../src/LowFrequencyReader.h"
#include "../src/ParserThreadPool.h"
#include "../src/ExportWriter.h"

#include <chrono>
#include <future>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

struct ConcurrentFirstBlockRingTestAccess
{
    static size_t capacity(const ConcurrentFirstBlockRing& ring) { return ring.slot_count_; }
};

template <uint32_t N>
struct FastqClassifierTestAccess
{
    static void feed(FastqClassifier<N>& classifier, std::vector<kmer<N>>& keys, bool owned)
    {
        constexpr size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>);
        for (size_t pos = 0; pos < keys.size(); pos += capacity) {
            const auto count = std::min(capacity, keys.size() - pos);
            if (owned) classifier.process_owned_block(keys.data() + pos, count);
            else classifier.process_other_block(keys.data() + pos, count);
        }
    }
    static void flush(FastqClassifier<N>& classifier) { classifier.flush_evicted_batch(); }
    static void group_and_decode(FastqClassifier<N>& classifier) { classifier.group_and_decode_evicted(); }
    static const auto* grouped(FastqClassifier<N>& classifier) { return classifier.evicted_grouped_; }
    static void finish(FastqClassifier<N>& classifier) { classifier.drain_first_occurrences(); }
    static const auto& codec(FastqClassifier<N>& classifier) { return classifier.first_codec_; }
    static size_t block_capacity(FastqClassifier<N>& classifier) { return classifier.first_block_capacity_; }
    static size_t ring_capacity(FastqClassifier<N>& classifier)
    { return ConcurrentFirstBlockRingTestAccess::capacity(*classifier.first_occurrence_ring_) * block_capacity(classifier); }
    static void push(FastqClassifier<N>& classifier, const kmer<N>& key) { classifier.push_first_occurrence(key); }
    static size_t batch_capacity(FastqClassifier<N>& classifier) { return classifier.evicted_capacity_; }
    static size_t pending(FastqClassifier<N>& classifier) { return classifier.evicted_count_; }
    static auto& packed(FastqClassifier<N>& classifier) { return classifier.evicted_buffer_; }
    static void stage(FastqClassifier<N>& classifier, const std::vector<kmer<N>>& keys)
    {
        const auto& codec = FastqClassifierTestAccess::codec(classifier);
        for (size_t i = 0; i < keys.size(); ++i)
            codec.pack(keys[i], classifier.evicted_buffer_.data() + i * codec.packed_bytes());
        classifier.evicted_count_ = keys.size();
    }
    static auto* roots(FastqClassifier<N>& classifier) { return classifier.local_root_nodes.data(); }
    static void collect_partial_export(FastqClassifier<N>& classifier, std::map<kmer<N>, uint64_t>& counts)
    {
        for (size_t i = 0; i < classifier.export_kmer_block_count; ++i)
            ++counts[classifier.export_block_ptr->k_mers[i]];
    }
};

namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

void check_parser_returns_empty_blocks()
{
    prefix_owners.fill(0);
    RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY> reads(64, 1);
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> parsed(
        PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners;
    for (int i = 0; i < 2; ++i)
        owners.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    for (size_t bases : {size_t{0}, size_t{33}}) {
        if (bases > 0) {
            char* block;
            reads.producer_dequeue(block);
            std::memset(block, 'A', bases);
            reads.producer_enqueue({block, bases});
        }
        reads.producer_set_finished();
        FastqParser<2> parser(33, owners, &global, &reads, &parsed);
        parser.parse_and_push();
        require(parser.get_total_read_kmer() == (bases == 0 ? 0 : 1), "parser input count differs");
        content_type content;
        size_t records = 0;
        for (auto& owner : owners) {
            while (owner->try_dequeue(content)) {
                records += content.length;
                parsed.consumer_enqueue(content.data);
            }
        }
        while (global.try_dequeue(content)) {
            records += content.length;
            parsed.consumer_enqueue(content.data);
        }
        require(records == parser.get_total_read_kmer(), "parser lost output records");
        std::vector<char*> returned;
        char* block;
        while (parsed.producer_try_dequeue(block)) returned.push_back(block);
        require(returned.size() == parsed.capacity(), "parser did not return empty owner blocks");
        for (auto* data : returned) parsed.consumer_enqueue(data);
        reads.reset_producers(1);
    }
}

template <uint32_t N>
kmer<N> make_key(uint32_t k, uint64_t seed)
{
    kmer<N> key{};
    for (uint32_t i = 0; i < k; ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        key.data[i / 32] |= (seed & 3) << (62 - 2 * (i % 32));
    }
    return key;
}

struct Shared
{
    ConcurrentMemoryPool pool{256ULL << 20};
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports{EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1};
    // Reused for decoded records during the reconciliation tests.
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> inputs{PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, 0};
    Shared() { pool.init_arenas(); }
};

template <uint32_t N>
struct Fixture
{
    Shared& shared;
    ConcurrentMemoryPool pool{64ULL << 20};
    std::unique_ptr<LayerQueues<N>> layers = std::make_unique<LayerQueues<N>>();
    std::unique_ptr<KmerTree<N>> tree;
    std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>> local =
        std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>();
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    std::barrier<> start{1};
    std::unique_ptr<FastqClassifier<N>> classifier;
    Fixture(Shared& state, uint32_t k) : shared(state)
    {
        pool.init_arenas();
        prefix_owners.fill(0);
        bloom_filter_capacity.fill(4096);
        tree = std::make_unique<KmerTree<N>>(k, &pool, layers.get(), &shared.exports);
        classifier = std::make_unique<FastqClassifier<N>>(k, 0, &shared.inputs, local.get(), &global,
            tree.get(), &pool, start,
            ConcurrentFirstBlockRing::create(&pool, 0, 1,
                align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
                    + 2 * ConcurrentFirstBlockRing::BLOCK_BYTES), 0);
    }
};

template <uint32_t N>
void collect_roots(node<N>* roots, std::map<kmer<N>, uint64_t>& counts)
{
    for (uint32_t prefix = 0; prefix < 256; ++prefix) {
        for (size_t b = 0; b < roots[prefix].count; ++b) {
            auto* block = roots[prefix].kmer_blocks[b];
            for (size_t i = 0; i < block->count; ++i) {
                const auto& key = block->k_mers[i];
                require(key.data[0] >> 56 == prefix, "recovered key was submitted to the wrong root");
                ++counts[key];
            }
        }
    }
}

template <uint32_t N>
void collect_exports(Shared& shared, std::map<kmer<N>, uint64_t>& counts)
{
    content_type content;
    while (shared.exports.consumer_try_dequeue(content)) {
        const auto* block = reinterpret_cast<const ExportBlock<N>*>(content.data);
        for (size_t i = 0; i < content.length; ++i) ++counts[block->k_mers[i]];
        shared.exports.consumer_enqueue(content.data);
    }
}

template <uint32_t N>
void check_group_and_decode(Shared& shared, uint32_t k)
{
    Fixture<N> fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess<N>;
    const size_t capacity = Access::batch_capacity(classifier);
    for (size_t count : {size_t{0}, size_t{1}, capacity - 1, capacity}) {
        for (bool single_root : {false, true}) {
            std::vector<kmer<N>> keys;
            std::map<kmer<N>, uint64_t> expected;
            for (size_t i = 0; i < count; ++i) {
                auto key = make_key<N>(k, i / 3 + 1);
                const uint64_t prefix = single_root ? 57 : (i / 3 * 73) % 256;
                key.data[0] = (key.data[0] & 0x00ffffffffffffffULL) | (prefix << 56);
                keys.push_back(key);
                ++expected[key];
            }
            std::reverse(keys.begin(), keys.end());
            auto ordered = keys;
            std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
                return (a.data[0] >> 56) < (b.data[0] >> 56);
            });
            Access::packed(classifier).fill(0xa5);
            Access::stage(classifier, keys);
            const auto packed_before = Access::packed(classifier);
            Access::group_and_decode(classifier);
            require(Access::packed(classifier) == packed_before, "grouping modified packed input");
            std::map<kmer<N>, uint64_t> actual;
            for (size_t i = 0; i < count; ++i) {
                const auto& restored = Access::grouped(classifier)[i];
                require(restored == ordered[i], "decoded records are not stably grouped by root");
                ++actual[restored];
            }
            require(actual == expected, "grouped decoding lost or duplicated records");
        }
    }
}

void check_root_block_rollover(Shared& shared)
{
    Fixture<2> fixture(shared, 33);
    auto* roots = FastqClassifierTestAccess<2>::roots(*fixture.classifier);
    constexpr uint32_t prefix = 57;
    auto key = make_key<2>(33, 999);
    key.data[0] = (key.data[0] & 0x00ffffffffffffffULL) | (uint64_t{prefix} << 56);
    constexpr size_t block_capacity = (KMER_BLOCK_SIZE - sizeof(uint64_t)) / sizeof(kmer<2>);
    const size_t count = (MAX_KMER_BLOCK_NUM + 1) * block_capacity + 3;
    std::vector<kmer<2>> keys(count, key);
    fixture.tree->main_add_kmers_to_local_root(keys.data(), 0, prefix, roots);
    require(roots[prefix].count == 0, "empty root append allocated a tree block");
    fixture.tree->main_add_kmers_to_local_root(keys.data(), keys.size(), prefix, roots);
    require(fixture.layers->size(0) == 1, "root rollover did not account for its queued task");
    Task<2> task;
    require(fixture.layers->get_queue(0)->try_dequeue(task), "full local root was not queued");
    require(task.current_node == fixture.tree->root_nodes + prefix && task.depth == 0,
        "rollover task points at the wrong global root");
    require(task.count == MAX_KMER_BLOCK_NUM, "rollover task omitted full blocks");
    std::map<kmer<2>, uint64_t> actual;
    for (size_t b = 0; b < task.count; ++b)
    {
        require(task.kmer_blocks[b]->count == block_capacity, "rollover published a partial block");
        for (size_t i = 0; i < task.kmer_blocks[b]->count; ++i)
            ++actual[task.kmer_blocks[b]->k_mers[i]];
    }
    // The ordinary multi-root interface must share the same append behavior.
    auto other = key;
    other.data[0] = (other.data[0] & 0x00ffffffffffffffULL) | (173ULL << 56);
    std::array<kmer<2>, 3> more{key, key, other};
    std::array<uint32_t, 256> counts{}, order{};
    counts[prefix] = 2;
    counts[173] = 1;
    order[0] = prefix;
    order[1] = 173;
    fixture.tree->main_add_kmer_block_with_local_root_nodes(more, counts, order, 2, roots);
    collect_roots(roots, actual);
    const std::map<kmer<2>, uint64_t> expected{{key, count + 2}, {other, 1}};
    require(actual == expected, "tree append lost data across a block or root boundary");
}

template <uint32_t N>
void check_grouped_queries(Shared& shared, uint32_t k)
{
    Fixture<N> fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess<N>;
    Access::flush(classifier); // Empty flush is a no-op.
    const size_t count = Access::batch_capacity(classifier);
    require(count == 32768 / ((k + 3) / 4), "batch does not use 32 KiB packed capacity");
    std::vector<kmer<N>> keys;
    std::set<std::pair<uint32_t, uint64_t>> occupied;
    std::map<kmer<N>, uint64_t> expected_tree, expected_export;
    // Distinct filter cells make positive and negative answers deterministic.
    for (uint64_t seed = 1; keys.size() < count; ++seed) {
        const auto key = make_key<N>(k, seed);
        const auto prefix = static_cast<uint32_t>(key.data[0] >> 56);
        auto& bloom = *static_cast<ConcurrentBloomFilter<N>*>(global_bloom_filter[prefix]);
        const auto probe = bloom.prepare_insert(key);
        if (!occupied.emplace(prefix, probe.block_idx).second) continue;
        const bool hit = keys.size() % 3 != 0;
        if (hit) bloom.get_bins()[2 * probe.block_idx].store(probe.insert_num);
        ++(hit ? expected_tree : expected_export)[key];
        keys.push_back(key);
    }
    Access::stage(classifier, keys);
    Access::flush(classifier);
    // A short single-root tail also exercises the partially warmed probe window.
    std::vector<kmer<N>> tail(3, keys[1]);
    expected_tree[keys[1]] += tail.size();
    Access::stage(classifier, tail);
    Access::flush(classifier);
    require(Access::pending(classifier) == 0, "batch flush retained records");
    std::map<kmer<N>, uint64_t> actual_tree, actual_export;
    collect_roots(Access::roots(classifier), actual_tree);
    collect_exports(shared, actual_export);
    Access::collect_partial_export(classifier, actual_export);
    require(actual_tree == expected_tree, "grouped query lost or duplicated tree hits");
    require(actual_export == expected_export, "grouped query exported wrong records");
}

template <uint32_t N>
void check_full_flow(Shared& shared, uint32_t k)
{
    Fixture<N> fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess<N>;
    std::vector<kmer<N>> first;
    std::map<kmer<N>, uint64_t> expected;
    const size_t count = 3 * Access::ring_capacity(classifier) + Access::batch_capacity(classifier) + 37;
    for (size_t i = 0; i < count; ++i) {
        first.push_back(make_key<N>(k, i + 1));
        ++expected[first.back()];
    }
    const size_t cut = std::min(first.size(), Access::ring_capacity(classifier));
    std::vector<kmer<N>> beginning(first.begin(), first.begin() + cut);
    Access::feed(classifier, beginning, true);
    std::vector<kmer<N>> repeats;
    for (size_t i = 0; i < cut; i += 3) {
        repeats.push_back(first[i]);
        ++expected[first[i]];
    }
    Access::feed(classifier, repeats, false);
    std::vector<kmer<N>> remaining(first.begin() + cut, first.end());
    Access::feed(classifier, remaining, true);
    require(classifier.repeat_query_count > 0, "fixture never triggered a full runtime batch");
    Access::finish(classifier);
    require(Access::pending(classifier) == 0,
        "terminal drain left pending occurrences");
    std::map<kmer<N>, uint64_t> actual;
    collect_roots(Access::roots(classifier), actual);
    collect_exports(shared, actual);
    Access::collect_partial_export(classifier, actual);
    require(actual == expected, "FIRST/REPEAT routing did not conserve the input multiset");
    require(classifier.first_occurrence_count == classifier.repeat_query_count,
        "some FIRSTs were queried zero or multiple times");
    require(classifier.total_deal_kmer_count == classifier.total_kmers_send_to_tree + classifier.total_kmers_exported,
        "input count differs from tree plus exports");
}


template <uint32_t N>
void check_first_boundaries(Shared& shared, uint32_t k)
{
    using Access = FastqClassifierTestAccess<N>;
    for (size_t blocks : {0, 1, 10}) {
        for (size_t tail : {0, 3}) {
            Fixture<N> fixture(shared, k);
            auto& classifier = *fixture.classifier;
            const size_t count = blocks * Access::block_capacity(classifier) + tail;
            std::map<kmer<N>, uint64_t> expected, actual;
            // Exercise exact FIRST counts independently of Bloom false positives.
            for (size_t i = 0; i < count; ++i) {
                auto key = make_key<N>(k, 12345 + i);
                Access::push(classifier, key);
                ++expected[key];
            }
            Access::finish(classifier);
            collect_roots(Access::roots(classifier), actual);
            collect_exports(shared, actual);
            Access::collect_partial_export(classifier, actual);
            require(actual == expected, "full-block/tail append lost records across a 32 KiB boundary");
            require(classifier.first_occurrence_count == count && classifier.repeat_query_count == count,
                "boundary FIRST queries are not conserved");
        }
    }
}

template <uint32_t N, bool WRITE_LOW = true>
void check_late_reconcile(Shared& shared, uint32_t k)
{
    Fixture<N> fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess<N>;
    std::vector<kmer<N>> pending;
    for (uint64_t i = 1; i <= 3; ++i) {
        auto key = make_key<N>(k, 100 + i);
        key.data[0] = (key.data[0] & 0x00ffffffffffffffULL) | (i << 60);
        auto* bloom = static_cast<ConcurrentBloomFilter<N>*>(global_bloom_filter[key.data[0] >> 56]);
        require(bloom->insert(key) == Occurrence::FIRST, "fixture FIRST collided");
        pending.push_back(key);
    }
    // Query the evicted FIRSTs before the second appearance arrives.
    Access::stage(classifier, pending);
    Access::flush(classifier);
    std::map<kmer<N>, uint64_t> exported;
    Access::collect_partial_export(classifier, exported);
    require(exported.size() == 3, "FIRSTs did not reach pending output");

    std::vector<kmer<N>> late{pending[0], pending[0]};
    Access::feed(classifier, late, true);
    auto recovered_early = make_key<N>(k, 999);
    recovered_early.data[0] = (recovered_early.data[0] & 0x00ffffffffffffffULL) | (4ULL << 60);
    std::vector<kmer<N>> early{recovered_early, recovered_early};
    Access::feed(classifier, early, true);
    Access::finish(classifier);

    // A final Bloom false positive must move exactly one real occurrence.
    auto* bloom = static_cast<ConcurrentBloomFilter<N>*>(global_bloom_filter[pending[2].data[0] >> 56]);
    const auto probe = bloom->prepare_insert(pending[2]);
    bloom->get_bins()[probe.block_idx * 2].fetch_or(probe.insert_num);

    char name[] = "/tmp/tree-low-reconcile-XXXXXX";
    const std::filesystem::path directory = ::mkdtemp(name);
    const auto input_path = directory / "pending.bin";
    const auto output_path = directory / "low.bin";
    const auto& codec = Access::codec(classifier);
    const size_t width = codec.packed_bytes();
    {
        std::ofstream out(input_path, std::ios::binary);
        std::array<uint8_t, 32> record{};
        for (const auto& key : pending) {
            codec.pack(key, record.data());
            out.write(reinterpret_cast<const char*>(record.data()), width);
        }
    }
    // Finish and drain the first-pass export queue before reusing it.
    classifier.classify_first_pass();
    exported.clear();
    collect_exports(shared, exported);
    require(exported.size() == pending.size(), "first-pass pending records differ");
    shared.inputs.reset_producers(1);
    SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY> reads(LOW_FREQUENCY_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    LowFrequencyReader reader(k, reads);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners{fixture.local};
    ParserThreadPool<N> parsers(k, owners, &fixture.global, nullptr, &shared.inputs, 1);
    std::unique_ptr<ExportWriter<N>> writer;
    if constexpr (WRITE_LOW) {
        shared.exports.reset_producers(1);
        writer = std::make_unique<ExportWriter<N>>(k, &shared.exports, output_path.string());
        writer->start();
    }
    parsers.start_low_pass(reads, 1);
    reader.start(input_path.string());
    classifier.template reconcile_low<WRITE_LOW>();
    reader.join();
    parsers.join();
    if constexpr (WRITE_LOW) {
        shared.exports.producer_set_finished();
        writer->join();
    }
    std::map<kmer<N>, uint64_t> tree_counts;
    collect_roots(Access::roots(classifier), tree_counts);
    require(tree_counts[pending[0]] == 3, "late FIRST was lost or counted twice");
    require(tree_counts[pending[2]] == 1, "false positive did not retain its exact count");
    require(tree_counts[recovered_early] == 2, "already recovered FIRST was counted again");
    require(tree_counts[pending[1]] == 0, "singleton was unexpectedly moved");
    require(classifier.low_records_scanned == 3 && classifier.low_records_recovered == 2,
        "pending query conservation failed");
    if constexpr (WRITE_LOW) {
        require(std::filesystem::file_size(output_path) == width, "low output lost its singleton");
        std::ifstream out(output_path, std::ios::binary);
        std::array<uint8_t, 32> record{};
        out.read(reinterpret_cast<char*>(record.data()), width);
        kmer<N> key{};
        codec.unpack(record.data(), key);
        require(key == pending[1], "low output contains the wrong key");
    } else {
        exported.clear();
        collect_exports(shared, exported);
        require(exported.empty(), "disabled low output exported singleton records");
    }
    std::filesystem::remove_all(directory);
}

void check_final_drain_filter(Shared& shared, bool hashed, uint32_t minimum, uint32_t maximum)
{
    constexpr uint32_t k = 33;
    constexpr uint32_t prefix = 57;
    Fixture<2> fixture(shared, k);
    const uint32_t saved_min = filter_min, saved_max = filter_max, saved_cap = count_max;
    filter_min = minimum;
    filter_max = maximum;
    count_max = 4;
    const auto below_before = filtered_below_min_count.load();
    const auto above_before = filtered_above_max_count.load();
    std::array<kmer<2>, 4> keys;
    const std::array<uint32_t, 4> counts{1, 2, 4, 5};
    std::map<kmer<2>, uint32_t> expected;
    uint64_t below = 0, above = 0;
    std::vector<std::unique_ptr<node<2>[]>> slabs;
    std::unique_ptr<ConcurrentOpenAddressHashMap<2>> map;
    if (hashed) {
        auto* current = fixture.tree->root_nodes + prefix;
        for (uint32_t depth = 0; depth + 1 < MAX_DEPTH; ++depth) {
            auto* children = current->children_ptr.load();
            if (!children) {
                slabs.push_back(std::make_unique<node<2>[]>(1ULL << (2 * NODE_BASES)));
                children = slabs.back().get();
                current->children_ptr.store(children);
            }
            current = children;
        }
        ConcurrentOpenAddressHashMap<2>::set_memory_pool(&fixture.pool);
        map = std::make_unique<ConcurrentOpenAddressHashMap<2>>(64);
        current->hash_map.store(map.get());
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = make_key<2>(k, i + 1);
        keys[i].data[0] = (keys[i].data[0] & 0x00ffffffffffffffULL) | (uint64_t{prefix} << 56);
        if (hashed) {
            const auto hash = ConcurrentOpenAddressHashMap<2>::hash_key(keys[i]);
            uint64_t inserted = 0;
            require(map->try_increment(hash, keys[i], ConcurrentOpenAddressHashMap<2>::fingerprint(hash),
                counts[i], inserted) == ConcurrentOpenAddressHashMap<2>::InsertResult::INSERTED,
                "failed to seed final drain hash map");
        } else {
            std::vector<kmer<2>> copies(counts[i], keys[i]);
            fixture.tree->main_add_kmers_to_local_root(copies.data(), copies.size(), prefix, fixture.tree->root_nodes);
        }
        const auto count = std::min(counts[i], count_max);
        if (count < minimum) ++below;
        else if (count > maximum) ++above;
        else expected.emplace(keys[i], count);
    }
    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> output(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
    {
        FinalDrainWriter<2> writer(k, &output);
        fixture.tree->final_drain_root(fixture.tree->root_nodes + prefix, writer);
        writer.close();
        writer.close();
    }
    require(filtered_below_min_count.load() - below_before == below,
        "incorrect below-minimum unique count or repeated close counted twice");
    require(filtered_above_max_count.load() - above_before == above,
        "incorrect above-maximum unique count or repeated close counted twice");
    filter_min = saved_min;
    filter_max = saved_max;
    count_max = saved_cap;

    PackedKmerCodec<2> codec(k);
    const size_t record_bytes = codec.packed_bytes() + count_max_bytes;
    std::map<kmer<2>, uint32_t> actual;
    content_type content;
    while (output.consumer_try_dequeue(content)) {
        require(content.length % record_bytes == 0, "final drain wrote a partial record");
        for (size_t pos = 0; pos < content.length; pos += record_bytes) {
            kmer<2> key;
            codec.unpack(reinterpret_cast<const uint8_t*>(content.data + pos), key);
            uint32_t count = 0;
            std::memcpy(&count, content.data + pos + codec.packed_bytes(), count_max_bytes);
            require(actual.emplace(key, count).second, "final drain duplicated a key");
        }
        output.consumer_enqueue(content.data);
    }
    require(actual == expected, "final drain did not apply the closed frequency interval after saturation");
}

void check_low_owner_and_global_queues(Shared& shared)
{
    constexpr uint32_t k = 33;
    using Access = FastqClassifierTestAccess<2>;
    for (size_t i = 0; i < prefix_owners.size(); ++i) prefix_owners[i] = i % 2;
    bloom_filter_capacity.fill(4096);
    shared.inputs.reset_producers(0);
    auto layers = std::make_unique<LayerQueues<2>>();
    KmerTree<2> tree(k, &shared.pool, layers.get(), &shared.exports);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local_a, local_b;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    std::barrier<> start(2);
    auto* ring = ConcurrentFirstBlockRing::create(&shared.pool, 0, 2,
        align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
            + 3 * ConcurrentFirstBlockRing::BLOCK_BYTES);
    auto create = [&](uint32_t index, auto* local) {
        return std::make_unique<FastqClassifier<2>>(k, index, &shared.inputs, local, &global,
            &tree, &shared.pool, start, ring, index);
    };
    auto constructing_a = std::async(std::launch::async, [&] { return create(0, &local_a); });
    auto b = create(1, &local_b);
    auto a = constructing_a.get();
    auto finishing_a = std::async(std::launch::async, [&] { a->classify_first_pass(); });
    b->classify_first_pass();
    finishing_a.get();
    std::map<kmer<2>, uint64_t> exports;
    collect_exports(shared, exports);
    require(exports.empty(), "empty first pass unexpectedly exported records");

    std::array<kmer<2>, 3> keys;
    shared.inputs.reset_producers(1);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = make_key<2>(k, 700 + i);
        keys[i].data[0] = (keys[i].data[0] & 0x00ffffffffffffffULL) | ((16ULL + i) << 56);
        if (i != 1) {
            auto* bloom = static_cast<ConcurrentBloomFilter<2>*>(global_bloom_filter[16 + i]);
            bloom->insert(keys[i]);
            bloom->insert(keys[i]);
        }
        char* block;
        shared.inputs.producer_dequeue(block);
        *reinterpret_cast<kmer<2>*>(block) = keys[i];
        if (i == 0) local_a.enqueue({block, 1});
        else if (i == 1) local_b.enqueue({block, 1});
        else global.enqueue({block, 1});
    }
    shared.inputs.producer_set_finished();
    // Classifier 1 must also reconcile owner 0's global task after EOF.
    b->reconcile_low<false>();
    a->reconcile_low<false>();
    std::map<kmer<2>, uint64_t> a_counts, b_counts;
    collect_roots(Access::roots(*a), a_counts);
    collect_roots(Access::roots(*b), b_counts);
    require(a_counts == std::map<kmer<2>, uint64_t>{{keys[0], 1}}, "owner queue was not reconciled exactly once");
    require(b_counts == std::map<kmer<2>, uint64_t>{{keys[2], 1}}, "foreign global task was not reconciled exactly once");
    require(a->low_records_scanned == 1 && b->low_records_scanned == 2 &&
        a->low_records_recovered == 1 && b->low_records_recovered == 1, "queue drain changed low conservation");
    require(a->local_tasks == 0 && a->global_tasks == 0 && b->local_tasks == 0 && b->global_tasks == 0,
        "low pass changed first-pass task statistics");
    content_type content;
    require(!local_a.try_dequeue(content) && !local_b.try_dequeue(content) && !global.try_dequeue(content),
        "low pass left queued tasks after EOF");
    char* block;
    std::vector<char*> returned;
    while (shared.inputs.producer_try_dequeue(block)) returned.push_back(block);
    require(returned.size() == shared.inputs.capacity(), "low queue consumer leaked blocks");
    for (auto* data : returned) shared.inputs.consumer_enqueue(data);
}

void check_cross_classifier_finish(Shared& shared)
{
    constexpr uint32_t k = 33;
    using Access = FastqClassifierTestAccess<2>;
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    auto layers = std::make_unique<LayerQueues<2>>();
    KmerTree<2> tree(k, &shared.pool, layers.get(), &shared.exports);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local_a, local_b;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    std::barrier<> start(2);
    auto* ring = ConcurrentFirstBlockRing::create(&shared.pool, 0, 2,
        align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
            + 3 * ConcurrentFirstBlockRing::BLOCK_BYTES);
    auto create = [&](uint32_t index, auto* local) {
        return std::make_unique<FastqClassifier<2>>(k, index, &shared.inputs, local, &global,
            &tree, &shared.pool, start, ring, index);
    };
    auto constructing_a = std::async(std::launch::async, [&] { return create(0, &local_a); });
    auto b = create(1, &local_b);
    auto a = constructing_a.get();
    std::vector<kmer<2>> input;
    std::map<kmer<2>, uint64_t> expected;
    for (size_t i = 0; i < Access::block_capacity(*a) + 37; ++i) {
        input.push_back(make_key<2>(k, 77 + i));
        expected[input.back()] += 2;
    }
    Access::feed(*a, input, true);
    auto finishing_a = std::async(std::launch::async, [&] { a->classify_first_pass(); a->finish_tree_input(); });
    require(finishing_a.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
        "Classifier finished before the other input producer reached the barrier");
    Access::feed(*b, input, false);
    std::vector<kmer<2>> other;
    for (size_t i = 0; i < Access::block_capacity(*b); ++i) {
        other.push_back(make_key<2>(k, 10000 + i));
        ++expected[other.back()];
    }
    Access::feed(*b, other, false);
    Access::flush(*b);
    require(b->recovered_first_count > 0, "shared slot did not recover another Classifier's FIRSTs");
    auto finishing_b = std::async(std::launch::async, [&] { b->classify_first_pass(); b->finish_tree_input(); });
    finishing_a.get();
    finishing_b.get();
    std::map<kmer<2>, uint64_t> actual_tree, actual_export;
    collect_roots(tree.root_nodes, actual_tree);
    collect_exports(shared, actual_export);
    for (const auto& [key, count] : actual_export) actual_tree[key] += count;
    require(actual_tree == expected, "shared ring and finish barrier did not conserve the input multiset");
    require(a->first_occurrence_count + b->first_occurrence_count == a->repeat_query_count + b->repeat_query_count,
        "global FIRST queries were lost or duplicated across Classifiers");
    require(a->total_deal_kmer_count + b->total_deal_kmer_count ==
        a->total_kmers_send_to_tree + b->total_kmers_send_to_tree + a->total_kmers_exported + b->total_kmers_exported,
        "global input count differs from tree plus exports");
}
}

int main()
{
    try {
        check_parser_returns_empty_blocks();
        Shared shared;
        check_group_and_decode<1>(shared, 31);
        check_group_and_decode<2>(shared, 33);
        check_group_and_decode<4>(shared, 65);
        check_group_and_decode<4>(shared, 81);
        check_group_and_decode<4>(shared, 128);
        check_root_block_rollover(shared);
        check_grouped_queries<1>(shared, 31);
        check_grouped_queries<2>(shared, 33);
        check_grouped_queries<4>(shared, 65);
        check_grouped_queries<4>(shared, 81);
        check_full_flow<1>(shared, 31);
        check_full_flow<2>(shared, 33);
        check_full_flow<4>(shared, 65);
        check_full_flow<4>(shared, 81);
        check_full_flow<4>(shared, 128);
        check_first_boundaries<1>(shared, 31);
        check_first_boundaries<2>(shared, 33);
        check_first_boundaries<4>(shared, 65);
        check_first_boundaries<4>(shared, 81);
        check_first_boundaries<4>(shared, 128);
        check_cross_classifier_finish(shared);
        check_low_owner_and_global_queues(shared);
        for (bool hashed : {false, true}) {
            for (auto [minimum, maximum] : {std::pair{2u, 3u}, {2u, 2u}, {1u, 1u},
                    {4u, 4u}, {0u, 0u}, {5u, 10u}, {0u, UINT32_MAX}})
                // The pool TLS cache belongs to one pool; isolate each fixture's allocations.
                std::async(std::launch::async, [&] {
                    check_final_drain_filter(shared, hashed, minimum, maximum);
                }).get();
        }
        check_late_reconcile<1>(shared, 31);
        check_late_reconcile<2>(shared, 33);
        check_late_reconcile<4>(shared, 81);
        check_late_reconcile<4>(shared, 128);
        check_late_reconcile<2, false>(shared, 33);
        std::cout << "ClassifierFirstBatch tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
