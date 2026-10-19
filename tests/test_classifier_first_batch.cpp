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

using Key = std::vector<unsigned char>;

static Key read_key(const unsigned char* bytes)
{ return Key(bytes, bytes + kmer_bytes()); }

static std::vector<unsigned char> flatten(const std::vector<Key>& keys)
{
    std::vector<unsigned char> bytes(keys.size() * kmer_bytes());
    for (size_t i = 0; i < keys.size(); ++i)
        std::memcpy(bytes.data() + i * kmer_bytes(), keys[i].data(), kmer_bytes());
    return bytes;
}

struct ConcurrentFirstBlockRingTestAccess
{
    static size_t capacity(const ConcurrentFirstBlockRing& ring) { return ring.slot_count_; }
};

struct FastqClassifierTestAccess
{
    static void feed(FastqClassifier& classifier, std::vector<Key>& keys, bool owned)
    {
        const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
        const auto bytes = flatten(keys);
        for (size_t pos = 0; pos < keys.size(); pos += capacity) {
            const auto count = std::min(capacity, keys.size() - pos);
            if (owned) classifier.process_owned_block(bytes.data() + pos * kmer_bytes(), count);
            else classifier.process_other_block(bytes.data() + pos * kmer_bytes(), count);
        }
    }
    template<bool WRITE_MISSES = true>
    static uint64_t flush(FastqClassifier& classifier) { return classifier.flush_evicted_batch<WRITE_MISSES>(); }
    static void group_and_decode(FastqClassifier& classifier) { classifier.group_evicted(); }
    static const auto* grouped(FastqClassifier& classifier) { return classifier.evicted_grouped_; }
    template<bool WRITE_MISSES = true>
    static uint64_t finish(FastqClassifier& classifier) { return classifier.drain_first_occurrences<WRITE_MISSES>(); }
    static size_t block_capacity(FastqClassifier& classifier) { return classifier.first_block_capacity_; }
    static size_t ring_capacity(FastqClassifier& classifier)
    { return ConcurrentFirstBlockRingTestAccess::capacity(*classifier.first_occurrence_ring_) * block_capacity(classifier); }
    static void push(FastqClassifier& classifier, const Key& key) { classifier.push_first_occurrence(key.data()); }
    static size_t batch_capacity(FastqClassifier& classifier) { return classifier.evicted_capacity_; }
    static size_t pending(FastqClassifier& classifier) { return classifier.evicted_count_; }
    static const auto& prefix_counts(FastqClassifier& classifier, bool evicted)
    { return evicted ? classifier.evicted_prefix_counts_ : classifier.local_block_prefix_counts; }
    static auto& packed(FastqClassifier& classifier) { return classifier.evicted_buffer_; }
    static void stage(FastqClassifier& classifier, const std::vector<Key>& keys)
    {
        for (size_t i = 0; i < keys.size(); ++i)
            std::memcpy(classifier.evicted_buffer_.data() + i * kmer_bytes(), keys[i].data(), kmer_bytes());
        classifier.evicted_count_ = keys.size();
    }
    static auto* roots(FastqClassifier& classifier) { return classifier.local_root_nodes.data(); }
    static void collect_partial_export(FastqClassifier& classifier, std::map<Key, uint64_t>& counts)
    {
        for (size_t i = 0; i < classifier.export_kmer_block_count; ++i)
            ++counts[read_key(classifier.export_block_ptr->k_mers.data() + i * kmer_bytes())];
    }
};

namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

struct TemporaryDirectory
{
    std::filesystem::path path;
    TemporaryDirectory()
    {
        char name[] = "/tmp/tree-first-drain-test-XXXXXX";
        const char* created = ::mkdtemp(name);
        if (created == nullptr) throw std::runtime_error("failed to create test directory");
        path = created;
    }
    ~TemporaryDirectory()
    {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void check_parser_returns_empty_blocks()
{
    k_length = 33;
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
        FastqParser parser(owners, &global, &reads, &parsed);
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

Key make_key(uint32_t k, uint64_t seed)
{
    Key key(kmer_bytes());
    for (uint32_t i = 0; i < k; ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        key[i / 4] |= (seed & 3) << (6 - 2 * (i % 4));
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

struct Fixture
{
    Shared& shared;
    ConcurrentMemoryPool pool{64ULL << 20};
    std::unique_ptr<LayerQueues> layers = std::make_unique<LayerQueues>();
    std::unique_ptr<KmerTree> tree;
    std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>> local =
        std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>();
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    std::barrier<> start{1};
    std::unique_ptr<FastqClassifier> classifier;
    Fixture(Shared& state, uint32_t k) : shared(state)
    {
        k_length = k;
        pool.init_arenas();
        prefix_owners.fill(0);
        bloom_filter_capacity.fill(4096);
        tree = std::make_unique<KmerTree>( &pool, layers.get(), &shared.exports);
        classifier = std::make_unique<FastqClassifier>(0, &shared.inputs, local.get(), &global,
            tree.get(), &pool, start,
            ConcurrentFirstBlockRing::create(&pool, 0, 1,
                align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
                    + 2 * ConcurrentFirstBlockRing::BLOCK_BYTES), 0);
    }
};

void collect_roots(node* roots, std::map<Key, uint64_t>& counts)
{
    for (uint32_t prefix = 0; prefix < 256; ++prefix) {
        for (size_t b = 0; b < roots[prefix].count; ++b) {
            auto* block = roots[prefix].kmer_blocks[b];
            for (size_t i = 0; i < block->count; ++i) {
                const auto key = read_key(block->k_mers.data() + i * kmer_bytes());
                require(key[0] == prefix, "recovered key was submitted to the wrong root");
                ++counts[key];
            }
        }
    }
}

void collect_exports(Shared& shared, std::map<Key, uint64_t>& counts)
{
    content_type content;
    while (shared.exports.consumer_try_dequeue(content)) {
        const auto* block = reinterpret_cast<const ExportBlock*>(content.data);
        for (size_t i = 0; i < content.length; ++i) ++counts[read_key(block->k_mers.data() + i * kmer_bytes())];
        shared.exports.consumer_enqueue(content.data);
    }
}

void check_group_and_decode(Shared& shared, uint32_t k)
{
    Fixture fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess;
    const size_t capacity = Access::batch_capacity(classifier);
    for (size_t count : {size_t{0}, size_t{1}, capacity - 1, capacity}) {
        for (bool single_root : {false, true}) {
            std::vector<Key> keys;
            std::map<Key, uint64_t> expected;
            for (size_t i = 0; i < count; ++i) {
                auto key = make_key(k, i / 3 + 1);
                const uint64_t prefix = single_root ? 57 : (i / 3 * 73) % 256;
                key[0] = ((prefix << 56)) >> 56;
                keys.push_back(key);
                ++expected[key];
            }
            std::reverse(keys.begin(), keys.end());
            auto ordered = keys;
            std::stable_sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) {
                return (a[0]) < (b[0]);
            });
            Access::packed(classifier).fill(0xa5);
            Access::stage(classifier, keys);
            const auto packed_before = Access::packed(classifier);
            Access::group_and_decode(classifier);
            require(Access::packed(classifier) == packed_before, "grouping modified packed input");
            std::map<Key, uint64_t> actual;
            for (size_t i = 0; i < count; ++i) {
                const auto restored = read_key(Access::grouped(classifier) + i * kmer_bytes());
                require(restored == ordered[i], "decoded records are not stably grouped by root");
                ++actual[restored];
            }
            require(actual == expected, "grouped decoding lost or duplicated records");
        }
    }
}

void check_root_block_rollover(Shared& shared)
{
    Fixture fixture(shared, 33);
    auto* roots = FastqClassifierTestAccess::roots(*fixture.classifier);
    constexpr uint32_t prefix = 57;
    auto key = make_key(33, 999);
    key[0] = ((uint64_t{prefix} << 56)) >> 56;
    const size_t block_capacity = 4092 / kmer_bytes();
    const size_t count = (MAX_KMER_BLOCK_NUM + 1) * block_capacity + 3;
    std::vector<Key> keys(count, key);
    fixture.tree->main_add_kmers_to_local_root(flatten(keys).data(), 0, prefix, roots);
    require(roots[prefix].count == 0, "empty root append allocated a tree block");
    fixture.tree->main_add_kmers_to_local_root(flatten(keys).data(), keys.size(), prefix, roots);
    require(fixture.layers->size(0) == 1, "root rollover did not account for its queued task");
    Task task;
    require(fixture.layers->get_queue(0)->try_dequeue(task), "full local root was not queued");
    require(task.current_node == fixture.tree->root_nodes + prefix && task.depth == 0,
        "rollover task points at the wrong global root");
    require(task.count == MAX_KMER_BLOCK_NUM, "rollover task omitted full blocks");
    std::map<Key, uint64_t> actual;
    for (size_t b = 0; b < task.count; ++b)
    {
        require(task.kmer_blocks[b]->count == block_capacity, "rollover published a partial block");
        for (size_t i = 0; i < task.kmer_blocks[b]->count; ++i)
            ++actual[read_key(task.kmer_blocks[b]->k_mers.data() + i * kmer_bytes())];
    }
    // The ordinary multi-root interface must share the same append behavior.
    auto other = key;
    other[0] = ((173ULL << 56)) >> 56;
    std::array<unsigned char, 27> more;
    const auto more_bytes = flatten({key, key, other});
    std::copy(more_bytes.begin(), more_bytes.end(), more.begin());
    std::array<uint32_t, 256> counts{}, order{};
    counts[prefix] = 2;
    counts[173] = 1;
    order[0] = prefix;
    order[1] = 173;
    fixture.tree->main_add_kmer_block_with_local_root_nodes(more, counts, order, 2, roots);
    collect_roots(roots, actual);
    const std::map<Key, uint64_t> expected{{key, count + 2}, {other, 1}};
    require(actual == expected, "tree append lost data across a block or root boundary");
}

template<bool WRITE_MISSES = true>
void check_grouped_queries(Shared& shared, uint32_t k)
{
    Fixture fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess;
    require(Access::flush<WRITE_MISSES>(classifier) == 0, "empty batch returned misses");
    const size_t count = Access::batch_capacity(classifier);
    require(count == 32768 / ((k + 3) / 4), "batch does not use 32 KiB packed capacity");
    std::vector<Key> keys;
    std::set<std::pair<uint32_t, uint64_t>> occupied;
    std::map<Key, uint64_t> expected_tree, expected_export;
    // Distinct filter cells make positive and negative answers deterministic.
    for (uint64_t seed = 1; keys.size() < count; ++seed) {
        const auto key = make_key(k, seed);
        const auto prefix = static_cast<uint32_t>(key[0]);
        auto& bloom = *static_cast<ConcurrentBloomFilter*>(global_bloom_filter[prefix]);
        const auto probe = bloom.prepare_insert(key.data());
        if (!occupied.emplace(prefix, probe.block_idx).second) continue;
        const bool hit = keys.size() % 3 != 0;
        if (hit) bloom.get_bins()[2 * probe.block_idx].store(probe.insert_num);
        ++(hit ? expected_tree : expected_export)[key];
        keys.push_back(key);
    }
    Access::stage(classifier, keys);
    require(Access::flush<WRITE_MISSES>(classifier) == expected_export.size(), "batch miss count differs");
    // A short single-root tail also exercises the partially warmed probe window.
    std::vector<Key> tail(3, keys[1]);
    expected_tree[keys[1]] += tail.size();
    Access::stage(classifier, tail);
    require(Access::flush<WRITE_MISSES>(classifier) == 0, "all-hit tail returned misses");
    require(Access::pending(classifier) == 0, "batch flush retained records");
    std::map<Key, uint64_t> actual_tree, actual_export;
    collect_roots(Access::roots(classifier), actual_tree);
    collect_exports(shared, actual_export);
    Access::collect_partial_export(classifier, actual_export);
    require(actual_tree == expected_tree, "grouped query lost or duplicated tree hits");
    if constexpr (WRITE_MISSES)
        require(actual_export == expected_export, "grouped query exported wrong records");
    else
        require(actual_export.empty(), "disabled batch output exported misses");
}

template<bool WRITE_MISSES>
void check_drain_miss_totals(Shared& shared)
{
    Fixture fixture(shared, 33);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess;
    const size_t staged = Access::batch_capacity(classifier) - 2;
    const size_t total = staged + Access::ring_capacity(classifier) + 3;
    std::vector<Key> keys;
    std::set<std::pair<uint32_t, uint64_t>> occupied;
    std::map<Key, uint64_t> expected_tree, expected_export;
    for (uint64_t seed = 1; keys.size() < total; ++seed) {
        const auto key = make_key(33, seed);
        auto& bloom = *static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
        const auto probe = bloom.prepare_insert(key.data());
        if (!occupied.emplace(key[0], probe.block_idx).second) continue;
        const bool hit = keys.size() % 2 == 0;
        if (hit) bloom.get_bins()[2 * probe.block_idx].store(probe.insert_num);
        ++(hit ? expected_tree : expected_export)[key];
        keys.push_back(key);
    }
    const std::vector<Key> batch(keys.begin(), keys.begin() + staged);
    Access::stage(classifier, batch);
    for (size_t i = staged; i < keys.size(); ++i) Access::push(classifier, keys[i]);
    require(Access::finish<WRITE_MISSES>(classifier) == expected_export.size(),
        "drain omitted misses from a tail, ring block, or staged batch");
    require(Access::pending(classifier) == 0, "drain left a partial evicted batch");
    std::map<Key, uint64_t> actual_tree, actual_export;
    collect_roots(Access::roots(classifier), actual_tree);
    collect_exports(shared, actual_export);
    Access::collect_partial_export(classifier, actual_export);
    require(actual_tree == expected_tree, "drain lost or duplicated tree hits");
    if constexpr (WRITE_MISSES)
        require(actual_export == expected_export, "drain lost or duplicated misses");
    else
        require(actual_export.empty(), "disabled drain output exported misses");
}

void check_compaction_runs(Shared& shared, uint32_t k)
{
    using Access = FastqClassifierTestAccess;
    // Missing tail flushes, overlapping memcpy, or overwriting the current
    // FIRST/miss must change the exact tree/export multisets below.
    for (int mode : {0, 1, 2}) {
        Fixture fixture(shared, k);
        auto& classifier = *fixture.classifier;
        std::set<std::pair<uint32_t, uint64_t>> occupied;
        std::map<Key, uint64_t> expected_tree, expected_export;
        uint64_t seed = 10000;
        size_t total = 0, retained = 0;
        for (int pattern = 0; pattern < 7; ++pattern) {
            std::vector<Key> keys;
            std::array<uint32_t, 256> expected_prefixes{};
            for (auto [prefix, count] : {std::pair{17u, 7u}, {89u, 96u}, {211u, 5u}}) {
                for (uint32_t i = 0; i < count; ++i) {
                    Key key;
                    auto& bloom = *static_cast<ConcurrentBloomFilter*>(global_bloom_filter[prefix]);
                    ConcurrentBloomFilter::InsertProbe probe;
                    do {
                        key = make_key(k, seed++);
                        key[0] = prefix;
                        probe = bloom.prepare_insert(key.data());
                    } while (!occupied.emplace(prefix, probe.block_idx).second);
                    const bool keep = pattern == 0 ||
                        (pattern == 2 && i % 2 == 1) ||
                        (pattern == 3 && i >= 2) ||
                        (pattern == 4 && i + 2 < count) ||
                        (pattern == 5 && i >= 2 && i + 2 < count) ||
                        (pattern == 6 && (prefix != 17 || i != 0));
                    if (keep) {
                        bloom.get_bins()[2 * probe.block_idx].store(probe.insert_num);
                        ++expected_tree[key];
                        ++expected_prefixes[prefix];
                        ++retained;
                    } else {
                        ++expected_export[key];
                    }
                    keys.push_back(key);
                }
            }
            total += keys.size();
            if (mode == 2) {
                Access::stage(classifier, keys);
                Access::flush(classifier);
            } else {
                Access::feed(classifier, keys, mode == 0);
            }
            require(Access::prefix_counts(classifier, mode == 2) == expected_prefixes,
                "run compaction changed per-prefix retained counts");
        }
        if (mode != 2) Access::finish(classifier);
        std::map<Key, uint64_t> actual_tree, actual_export;
        collect_roots(Access::roots(classifier), actual_tree);
        collect_exports(shared, actual_export);
        Access::collect_partial_export(classifier, actual_export);
        require(actual_tree == expected_tree, "run compaction changed retained keys or frequencies");
        require(actual_export == expected_export, "run compaction overwrote FIRST or exported records");
        require(classifier.total_kmers_send_to_tree == retained &&
            classifier.total_kmers_send_to_tree + classifier.total_kmers_exported == total,
            "run compaction did not conserve records");
        if (mode != 2)
            require(classifier.first_occurrence_count == classifier.repeat_query_count,
                "run compaction lost pending FIRST queries");
    }
}

void check_full_flow(Shared& shared, uint32_t k)
{
    Fixture fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess;
    std::vector<Key> first;
    std::map<Key, uint64_t> expected;
    const size_t count = 3 * Access::ring_capacity(classifier) + Access::batch_capacity(classifier) + 37;
    for (size_t i = 0; i < count; ++i) {
        first.push_back(make_key(k, i + 1));
        ++expected[first.back()];
    }
    const size_t cut = std::min(first.size(), Access::ring_capacity(classifier));
    std::vector<Key> beginning(first.begin(), first.begin() + cut);
    Access::feed(classifier, beginning, true);
    std::vector<Key> repeats;
    for (size_t i = 0; i < cut; i += 3) {
        repeats.push_back(first[i]);
        ++expected[first[i]];
    }
    Access::feed(classifier, repeats, false);
    std::vector<Key> remaining(first.begin() + cut, first.end());
    Access::feed(classifier, remaining, true);
    require(classifier.repeat_query_count > 0, "fixture never triggered a full runtime batch");
    Access::finish(classifier);
    require(Access::pending(classifier) == 0,
        "terminal drain left pending occurrences");
    std::map<Key, uint64_t> actual;
    collect_roots(Access::roots(classifier), actual);
    collect_exports(shared, actual);
    Access::collect_partial_export(classifier, actual);
    require(actual == expected, "FIRST/REPEAT routing did not conserve the input multiset");
    require(classifier.first_occurrence_count == classifier.repeat_query_count,
        "some FIRSTs were queried zero or multiple times");
    require(classifier.total_deal_kmer_count == classifier.total_kmers_send_to_tree + classifier.total_kmers_exported,
        "input count differs from tree plus exports");
}


void check_first_boundaries(Shared& shared, uint32_t k)
{
    using Access = FastqClassifierTestAccess;
    for (size_t blocks : {0, 1, 10}) {
        for (size_t tail : {0, 3}) {
            Fixture fixture(shared, k);
            auto& classifier = *fixture.classifier;
            const size_t count = blocks * Access::block_capacity(classifier) + tail;
            std::map<Key, uint64_t> expected, actual;
            // Exercise exact FIRST counts independently of Bloom false positives.
            for (size_t i = 0; i < count; ++i) {
                auto key = make_key(k, 12345 + i);
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

template <bool WRITE_LOW = true>
void check_late_reconcile(Shared& shared, uint32_t k)
{
    Fixture fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess;
    std::vector<Key> pending;
    for (uint64_t i = 1; i <= 3; ++i) {
        auto key = make_key(k, 100 + i);
        key[0] = ((i << 60)) >> 56;
        auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
        require(bloom->insert(key.data()) == Occurrence::FIRST, "fixture FIRST collided");
        pending.push_back(key);
    }
    // Query the evicted FIRSTs before the second appearance arrives.
    Access::stage(classifier, pending);
    Access::flush(classifier);
    std::map<Key, uint64_t> exported;
    Access::collect_partial_export(classifier, exported);
    require(exported.size() == 3, "FIRSTs did not reach pending output");

    std::vector<Key> late{pending[0], pending[0]};
    Access::feed(classifier, late, true);
    auto recovered_early = make_key(k, 999);
    recovered_early[0] = ((4ULL << 60)) >> 56;
    auto* early_bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[recovered_early[0]]);
    require(early_bloom->insert(recovered_early.data()) == Occurrence::FIRST, "early fixture FIRST collided");
    std::vector<Key> early{recovered_early};
    Access::feed(classifier, early, true);
    Access::stage(classifier, early);
    Access::flush(classifier);

    // A final Bloom false positive must move exactly one real occurrence.
    auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[pending[2][0]]);
    const auto probe = bloom->prepare_insert(pending[2].data());
    bloom->get_bins()[probe.block_idx * 2].fetch_or(probe.insert_num);

    TemporaryDirectory directory;
    const auto input_path = directory.path / "pending.bin";
    const auto output_path = directory.path / "low.bin";
    const size_t width = kmer_bytes();
    {
        std::ofstream out(input_path, std::ios::binary);
        std::array<uint8_t, 32> record{};
        for (const auto& key : pending) {
            std::memcpy(record.data(), key.data(), width);
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
    LowFrequencyReader reader(reads);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners{fixture.local};
    ParserThreadPool parsers(owners, &fixture.global, nullptr, &shared.inputs, 1);
    std::unique_ptr<ExportWriter> writer;
    if constexpr (WRITE_LOW) {
        shared.exports.reset_producers(1);
        writer = std::make_unique<ExportWriter>( &shared.exports, output_path.string());
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
    std::map<Key, uint64_t> tree_counts;
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
        Key key(kmer_bytes());
        std::memcpy(key.data(), record.data(), width);
        require(key == pending[1], "low output contains the wrong key");
    } else {
        exported.clear();
        collect_exports(shared, exported);
        require(exported.empty(), "disabled low output exported singleton records");
    }
}

template<bool WRITE_LOW>
void check_first_drain_after_writer_switch(Shared& shared, bool has_pending)
{
    std::vector<char*> available;
    char* block;
    while (shared.exports.producer_try_dequeue(block)) available.push_back(block);
    for (auto* data : available) shared.exports.consumer_enqueue(data);
    Fixture fixture(shared, 33);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess;
    TemporaryDirectory directory;
    const auto pending_path = directory.path / "pending.bin";
    const auto final_path = directory.path / "low.bin";
    std::vector<Key> pending;
    std::set<std::pair<uint32_t, uint64_t>> occupied;
    if (has_pending) {
        pending.push_back(make_key(33, 99999));
        auto& bloom = *static_cast<ConcurrentBloomFilter*>(global_bloom_filter[pending[0][0]]);
        occupied.emplace(pending[0][0], bloom.prepare_insert(pending[0].data()).block_idx);
        Access::stage(classifier, pending);
        Access::flush(classifier);
    }
    const uint64_t runtime_queries = classifier.repeat_query_count;
    const size_t staged = Access::batch_capacity(classifier) - 2;
    const size_t total = staged + Access::ring_capacity(classifier) + 3;
    std::vector<Key> keys;
    std::map<Key, uint64_t> expected_tree, expected_low;
    for (uint64_t seed = 1; keys.size() < total; ++seed) {
        const auto key = make_key(33, seed);
        auto& bloom = *static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
        const auto probe = bloom.prepare_insert(key.data());
        if (!occupied.emplace(key[0], probe.block_idx).second) continue;
        const bool hit = keys.size() % 3 == 0;
        if (hit) bloom.get_bins()[2 * probe.block_idx].store(probe.insert_num);
        ++(hit ? expected_tree : expected_low)[key];
        keys.push_back(key);
    }
    const std::vector<Key> batch(keys.begin(), keys.begin() + staged);
    Access::stage(classifier, batch);
    for (size_t i = staged; i < keys.size(); ++i) Access::push(classifier, keys[i]);
    shared.exports.reset_producers(1);
    ExportWriter pending_writer(&shared.exports, pending_path.string());
    pending_writer.start();
    classifier.classify_first_pass();
    shared.exports.producer_set_finished();
    pending_writer.join();
    require(classifier.repeat_query_count == runtime_queries, "first pass prematurely drained FIRST");
    require(std::filesystem::file_size(pending_path) == pending.size() * kmer_bytes(),
        "terminal FIRST misses reached the pending file");

    const uint64_t direct_singletons = expected_low.size();
    shared.inputs.reset_producers(0);
    if (has_pending) {
        char* block;
        shared.inputs.producer_dequeue(block);
        std::ifstream input(pending_path, std::ios::binary);
        input.read(block, kmer_bytes());
        require(input.good(), "failed to read pending test record");
        fixture.local->enqueue({block, 1});
        ++expected_low[pending[0]];
    }
    std::vector<char*> reserved;
    std::unique_ptr<ExportWriter> final_writer;
    if constexpr (WRITE_LOW) {
        shared.exports.reset_producers(1);
        final_writer = std::make_unique<ExportWriter>(&shared.exports, final_path.string());
        final_writer->start();
    } else {
        char* block;
        while (shared.exports.producer_try_dequeue(block)) reserved.push_back(block);
    }
    classifier.template reconcile_low<WRITE_LOW>();
    if constexpr (WRITE_LOW) {
        shared.exports.producer_set_finished();
        final_writer->join();
    }
    for (auto* block : reserved) shared.exports.consumer_enqueue(block);
    std::map<Key, uint64_t> actual_tree, actual_low;
    collect_roots(Access::roots(classifier), actual_tree);
    require(actual_tree == expected_tree, "second-stage FIRST drain changed recovered keys");
    require(classifier.first_drain_singletons == direct_singletons, "direct FIRST singleton count differs");
    require(classifier.low_records_scanned == pending.size() && classifier.low_records_recovered == 0,
        "memory FIRST drain changed pending scan counters");
    require(Access::pending(classifier) == 0, "second-stage FIRST drain left staged records");
    require(classifier.repeat_query_count == runtime_queries + total, "FIRST records were lost or queried twice");
    if constexpr (WRITE_LOW) {
        std::ifstream input(final_path, std::ios::binary);
        Key key(kmer_bytes());
        while (input.read(reinterpret_cast<char*>(key.data()), key.size())) ++actual_low[key];
        require(actual_low == expected_low, "final low lost or duplicated terminal singleton records");
    } else {
        collect_exports(shared, actual_low);
        require(actual_low.empty(), "disabled final low output exported FIRST misses");
    }
    std::vector<char*> returned;
    while (shared.exports.producer_try_dequeue(block)) returned.push_back(block);
    for (auto* data : returned) shared.exports.consumer_enqueue(data);
    std::sort(available.begin(), available.end(), std::less<char*>{});
    std::sort(returned.begin(), returned.end(), std::less<char*>{});
    require(returned == available, "Writer switch lost or returned an Export block twice");
}

void check_final_drain_filter(Shared& shared, bool hashed, uint32_t minimum, uint32_t maximum)
{
    constexpr uint32_t k = 33;
    k_length = k;
    constexpr uint32_t prefix = 57;
    Fixture fixture(shared, k);
    const uint32_t saved_min = filter_min, saved_max = filter_max, saved_cap = count_max;
    filter_min = minimum;
    filter_max = maximum;
    count_max = 4;
    const auto below_before = filtered_below_min_count.load();
    const auto above_before = filtered_above_max_count.load();
    std::array<Key, 4> keys;
    const std::array<uint32_t, 4> counts{1, 2, 4, 5};
    std::map<Key, uint32_t> expected;
    uint64_t below = 0, above = 0;
    std::vector<std::unique_ptr<node[]>> slabs;
    std::unique_ptr<ConcurrentOpenAddressHashMap> map;
    if (hashed) {
        auto* current = fixture.tree->root_nodes + prefix;
        for (uint32_t depth = 0; depth + 1 < MAX_DEPTH; ++depth) {
            auto* children = current->children_ptr.load();
            if (!children) {
                slabs.push_back(std::make_unique<node[]>(1ULL << (2 * NODE_BASES)));
                children = slabs.back().get();
                current->children_ptr.store(children);
            }
            current = children;
        }
        ConcurrentOpenAddressHashMap::set_memory_pool(&fixture.pool);
        map = std::make_unique<ConcurrentOpenAddressHashMap>(64);
        current->hash_map.store(map.get());
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = make_key(k, i + 1);
        keys[i][0] = ((uint64_t{prefix} << 56)) >> 56;
        if (hashed) {
            const auto hash = ConcurrentOpenAddressHashMap::hash_key(keys[i].data());
            uint64_t inserted = 0;
            require(map->try_increment(hash, keys[i].data(), ConcurrentOpenAddressHashMap::fingerprint(hash),
                counts[i], inserted) == ConcurrentOpenAddressHashMap::InsertResult::INSERTED,
                "failed to seed final drain hash map");
        } else {
            std::vector<Key> copies(counts[i], keys[i]);
            fixture.tree->main_add_kmers_to_local_root(flatten(copies).data(), copies.size(), prefix, fixture.tree->root_nodes);
        }
        const auto count = std::min(counts[i], count_max);
        if (count < minimum) ++below;
        else if (count > maximum) ++above;
        else expected.emplace(keys[i], count);
    }
    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> output(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
    {
        FinalDrainWriter writer( &output);
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

    const size_t record_bytes = kmer_bytes() + count_max_bytes;
    std::map<Key, uint32_t> actual;
    content_type content;
    while (output.consumer_try_dequeue(content)) {
        require(content.length % record_bytes == 0, "final drain wrote a partial record");
        for (size_t pos = 0; pos < content.length; pos += record_bytes) {
            const auto key = read_key(reinterpret_cast<const unsigned char*>(content.data + pos));
            uint32_t count = 0;
            std::memcpy(&count, content.data + pos + kmer_bytes(), count_max_bytes);
            require(actual.emplace(key, count).second, "final drain duplicated a key");
        }
        output.consumer_enqueue(content.data);
    }
    require(actual == expected, "final drain did not apply the closed frequency interval after saturation");
}

void check_final_drain_runs(Shared& shared, uint32_t k)
{
    Fixture fixture(shared, k);
    const auto saved_min = filter_min, saved_max = filter_max, saved_cap = count_max;
    const auto saved_count_bytes = count_max_bytes;
    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> output(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
    auto* root = fixture.tree->root_nodes + 57;
    std::array<node*, MAX_DEPTH> path{root, root->children_ptr.load()};
    std::vector<std::unique_ptr<node[]>> slabs;
    for (uint32_t depth = 2; depth < MAX_DEPTH; ++depth) {
        slabs.push_back(std::make_unique<node[]>(16));
        path[depth] = slabs.back().get();
        path[depth - 1]->children_ptr.store(path[depth]);
    }
    const size_t capacity = 4092 / kmer_bytes();
    const uint32_t padding = (4 - k % 4) % 4 * 2;
    for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth) {
        auto* leaf = path[depth];
        auto* children = leaf->children_ptr.exchange(nullptr);
        for (uint32_t count_bytes : {1u, 2u, 4u}) {
            count_max_bytes = count_bytes;
            count_max = count_bytes == 1 ? 255 : count_bytes == 2 ? 65535 : UINT32_MAX;
            for (uint32_t blocks : {0u, 1u, 2u, 3u, 8u, 16u}) {
                for (uint32_t pattern = 0; pattern < 5; ++pattern) {
                    filter_min = pattern == 3 ? 2 : 1;
                    filter_max = pattern == 3 ? 4 : UINT32_MAX;
                    std::map<Key, uint32_t> truth, expected;
                    const auto below_before = filtered_below_min_count.load();
                    const auto above_before = filtered_above_max_count.load();
                    const auto sorted_before = sorted_kmer_count.load();
                    for (uint32_t b = 0; b < blocks; ++b) {
                        auto* block = static_cast<kmer_block*>(fixture.pool.allocate());
                        // Unequal run lengths force early cursor exhaustion; the
                        // final pattern also covers full blocks and a short tail.
                        block->count = pattern == 4 ? (b + 1 == blocks ? 7 : capacity)
                            : (b == 0 ? 1 : 37 + b % 5);
                        for (uint32_t i = 0; i < block->count; ++i) {
                            uint32_t id = pattern == 0 ? b * 64 + i
                                : pattern == 1 ? i * blocks + b
                                : pattern == 2 ? 1
                                : pattern == 3 ? i % 7 : b * capacity + i;
                            Key key(kmer_bytes(), 0);
                            key[0] = 57;
                            id <<= padding;
                            for (uint32_t byte = 0; byte < 4; ++byte)
                                key[key.size() - 1 - byte] = id >> (byte * 8);
                            std::memcpy(block->k_mers.data() + i * kmer_bytes(), key.data(), key.size());
                            ++truth[key];
                        }
                        leaf->kmer_blocks[leaf->count++] = block;
                        leaf->active_block = block;
                    }
                    // The parent still has records that must be pushed into its
                    // existing child before that child can be exported.
                    if (depth > 0 && blocks == 3 && pattern == 1) {
                        auto* block = static_cast<kmer_block*>(fixture.pool.allocate());
                        block->count = 1;
                        const auto& key = truth.begin()->first;
                        std::memcpy(block->k_mers.data(), key.data(), key.size());
                        root->kmer_blocks[0] = block;
                        root->count = 1;
                        root->active_block = block;
                        ++truth[key];
                    }
                    uint64_t below = 0, above = 0;
                    for (const auto& [key, frequency] : truth) {
                        const auto count = std::min(frequency, count_max);
                        if (count < filter_min) ++below;
                        else if (count > filter_max) ++above;
                        else expected.emplace(key, count);
                    }
                    {
                        FinalDrainWriter writer(&output);
                        fixture.tree->final_drain_root(root, writer);
                        writer.close();
                        writer.close();
                    }
                    require(leaf->count == 0 && leaf->active_block == nullptr && root->count == 0,
                        "final drain left pending leaf or parent records");
                    require(filtered_below_min_count.load() - below_before == below &&
                        filtered_above_max_count.load() - above_before == above,
                        "final drain run filtering counts differ");
                    require(sorted_kmer_count.load() - sorted_before == expected.size(),
                        "final drain emitted the wrong unique count");
                    std::map<Key, uint32_t> actual;
                    content_type content;
                    const size_t record_bytes = kmer_bytes() + count_max_bytes;
                    while (output.consumer_try_dequeue(content)) {
                        require(content.length % record_bytes == 0, "final drain emitted a partial record");
                        for (size_t pos = 0; pos < content.length; pos += record_bytes) {
                            const auto key = read_key(reinterpret_cast<const unsigned char*>(content.data + pos));
                            uint32_t count = 0;
                            std::memcpy(&count, content.data + pos + kmer_bytes(), count_max_bytes);
                            require(actual.emplace(key, count).second, "final drain emitted a duplicate key");
                        }
                        output.consumer_enqueue(content.data);
                    }
                    require(actual == expected, "final drain changed the complete key/frequency mapping");
                }
            }
        }
        leaf->children_ptr.store(children);
    }
    // Two large leaves share one writer, crossing its 128 KiB block boundary.
    filter_min = 1;
    filter_max = count_max = UINT32_MAX;
    count_max_bytes = 4;
    std::map<Key, uint32_t> expected, actual;
    {
        FinalDrainWriter writer(&output);
        for (uint32_t prefix : {57u, 58u}) {
            auto* leaf = fixture.tree->root_nodes + prefix;
            auto* children = leaf->children_ptr.exchange(nullptr);
            for (uint32_t b = 0; b < MAX_KMER_BLOCK_NUM; ++b) {
                auto* block = static_cast<kmer_block*>(fixture.pool.allocate());
                block->count = capacity;
                for (uint32_t i = 0; i < capacity; ++i) {
                    Key key(kmer_bytes(), 0);
                    key[0] = prefix;
                    const uint32_t id = (b * capacity + i) << padding;
                    for (uint32_t byte = 0; byte < 4; ++byte)
                        key[key.size() - 1 - byte] = id >> (byte * 8);
                    std::memcpy(block->k_mers.data() + i * kmer_bytes(), key.data(), key.size());
                    expected.emplace(key, 1);
                }
                leaf->kmer_blocks[leaf->count++] = block;
                leaf->active_block = block;
            }
            fixture.tree->final_drain_root(leaf, writer);
            require(leaf->count == 0 && leaf->active_block == nullptr, "large leaf was not cleared");
            leaf->children_ptr.store(children);
        }
    }
    content_type content;
    size_t output_blocks = 0;
    const size_t record_bytes = kmer_bytes() + count_max_bytes;
    while (output.consumer_try_dequeue(content)) {
        ++output_blocks;
        require(content.length % record_bytes == 0, "large drain emitted a partial record");
        for (size_t pos = 0; pos < content.length; pos += record_bytes) {
            const auto key = read_key(reinterpret_cast<const unsigned char*>(content.data + pos));
            uint32_t count = 0;
            std::memcpy(&count, content.data + pos + kmer_bytes(), count_max_bytes);
            require(actual.emplace(key, count).second, "large drain emitted a duplicate key");
        }
        output.consumer_enqueue(content.data);
    }
    require(output_blocks >= 2 && actual == expected, "large drain lost records across output blocks");
    filter_min = saved_min;
    filter_max = saved_max;
    count_max = saved_cap;
    count_max_bytes = saved_count_bytes;
}

void check_low_owner_and_global_queues(Shared& shared)
{
    constexpr uint32_t k = 33;
    k_length = k;
    using Access = FastqClassifierTestAccess;
    for (size_t i = 0; i < prefix_owners.size(); ++i) prefix_owners[i] = i % 2;
    bloom_filter_capacity.fill(4096);
    shared.inputs.reset_producers(0);
    auto layers = std::make_unique<LayerQueues>();
    KmerTree tree( &shared.pool, layers.get(), &shared.exports);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local_a, local_b;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    std::barrier<> start(2);
    auto* ring = ConcurrentFirstBlockRing::create(&shared.pool, 0, 2,
        align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
            + 3 * ConcurrentFirstBlockRing::BLOCK_BYTES);
    auto create = [&](uint32_t index, auto* local) {
        return std::make_unique<FastqClassifier>(index, &shared.inputs, local, &global,
            &tree, &shared.pool, start, ring, index);
    };
    auto constructing_a = std::async(std::launch::async, [&] { return create(0, &local_a); });
    auto b = create(1, &local_b);
    auto a = constructing_a.get();
    auto finishing_a = std::async(std::launch::async, [&] { a->classify_first_pass(); });
    b->classify_first_pass();
    finishing_a.get();
    std::map<Key, uint64_t> exports;
    collect_exports(shared, exports);
    require(exports.empty(), "empty first pass unexpectedly exported records");

    std::array<Key, 3> keys;
    shared.inputs.reset_producers(1);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = make_key(k, 700 + i);
        keys[i][0] = (((16ULL + i) << 56)) >> 56;
        if (i != 1) {
            auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[16 + i]);
            bloom->insert(keys[i].data());
            bloom->insert(keys[i].data());
        }
        char* block;
        shared.inputs.producer_dequeue(block);
        std::memcpy(block, keys[i].data(), kmer_bytes());
        if (i == 0) local_a.enqueue({block, 1});
        else if (i == 1) local_b.enqueue({block, 1});
        else global.enqueue({block, 1});
    }
    shared.inputs.producer_set_finished();
    // Classifier 1 must also reconcile owner 0's global task after EOF.
    b->reconcile_low<false>();
    a->reconcile_low<false>();
    std::map<Key, uint64_t> a_counts, b_counts;
    collect_roots(Access::roots(*a), a_counts);
    collect_roots(Access::roots(*b), b_counts);
    require(a_counts == std::map<Key, uint64_t>{{keys[0], 1}}, "owner queue was not reconciled exactly once");
    require(b_counts == std::map<Key, uint64_t>{{keys[2], 1}}, "foreign global task was not reconciled exactly once");
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
    k_length = k;
    using Access = FastqClassifierTestAccess;
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    auto layers = std::make_unique<LayerQueues>();
    KmerTree tree( &shared.pool, layers.get(), &shared.exports);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local_a, local_b;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    std::barrier<> start(2);
    auto* ring = ConcurrentFirstBlockRing::create(&shared.pool, 0, 2,
        align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
            + 3 * ConcurrentFirstBlockRing::BLOCK_BYTES);
    auto create = [&](uint32_t index, auto* local) {
        return std::make_unique<FastqClassifier>(index, &shared.inputs, local, &global,
            &tree, &shared.pool, start, ring, index);
    };
    auto constructing_a = std::async(std::launch::async, [&] { return create(0, &local_a); });
    auto b = create(1, &local_b);
    auto a = constructing_a.get();
    std::vector<Key> input;
    std::map<Key, uint64_t> expected;
    for (size_t i = 0; i < Access::block_capacity(*a) + 37; ++i) {
        input.push_back(make_key(k, 77 + i));
        expected[input.back()] += 2;
    }
    Access::feed(*a, input, true);
    auto finishing_a = std::async(std::launch::async, [&] { a->classify_first_pass(); });
    require(finishing_a.wait_for(std::chrono::milliseconds(50)) == std::future_status::timeout,
        "Classifier finished before the other input producer reached the barrier");
    Access::feed(*b, input, false);
    std::vector<Key> other;
    for (size_t i = 0; i < Access::block_capacity(*b); ++i) {
        other.push_back(make_key(k, 10000 + i));
        ++expected[other.back()];
    }
    Access::feed(*b, other, false);
    Access::flush(*b);
    require(b->recovered_first_count > 0, "shared slot did not recover another Classifier's FIRSTs");
    auto finishing_b = std::async(std::launch::async, [&] { b->classify_first_pass(); });
    finishing_a.get();
    finishing_b.get();
    a->reconcile_low<true>();
    b->reconcile_low<true>();
    a->finish_tree_input();
    b->finish_tree_input();
    std::map<Key, uint64_t> actual_tree, actual_export;
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
        check_group_and_decode(shared, 31);
        check_group_and_decode(shared, 33);
        check_group_and_decode(shared, 65);
        check_group_and_decode(shared, 81);
        check_group_and_decode(shared, 128);
        check_root_block_rollover(shared);
        check_grouped_queries(shared, 31);
        check_grouped_queries(shared, 33);
        check_grouped_queries(shared, 65);
        check_grouped_queries(shared, 81);
        check_grouped_queries<false>(shared, 33);
        check_drain_miss_totals<true>(shared);
        check_drain_miss_totals<false>(shared);
        check_first_drain_after_writer_switch<true>(shared, false);
        check_first_drain_after_writer_switch<true>(shared, true);
        check_first_drain_after_writer_switch<false>(shared, false);
        check_first_drain_after_writer_switch<false>(shared, true);
        for (uint32_t k : {33u, 65u, 81u, 128u}) check_compaction_runs(shared, k);
        check_full_flow(shared, 31);
        check_full_flow(shared, 33);
        check_full_flow(shared, 65);
        check_full_flow(shared, 81);
        check_full_flow(shared, 128);
        check_first_boundaries(shared, 31);
        check_first_boundaries(shared, 33);
        check_first_boundaries(shared, 65);
        check_first_boundaries(shared, 81);
        check_first_boundaries(shared, 128);
        check_cross_classifier_finish(shared);
        check_low_owner_and_global_queues(shared);
        for (uint32_t k : {33u, 65u, 81u, 128u})
            std::async(std::launch::async, [&] { check_final_drain_runs(shared, k); }).get();
        for (bool hashed : {false, true}) {
            for (auto [minimum, maximum] : {std::pair{2u, 3u}, {2u, 2u}, {1u, 1u},
                    {4u, 4u}, {0u, 0u}, {5u, 10u}, {0u, UINT32_MAX}})
                // The pool TLS cache belongs to one pool; isolate each fixture's allocations.
                std::async(std::launch::async, [&] {
                    check_final_drain_filter(shared, hashed, minimum, maximum);
                }).get();
        }
        check_late_reconcile(shared, 31);
        check_late_reconcile(shared, 33);
        check_late_reconcile(shared, 81);
        check_late_reconcile(shared, 128);
        check_late_reconcile<false>(shared, 33);
        std::cout << "ClassifierFirstBatch tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
