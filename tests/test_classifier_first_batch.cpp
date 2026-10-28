#include "low_msd_test_helpers.h"
#include "high_prefix_test_helpers.h"
#include "../src/FastqClassifier.h"
#include "../src/ConcurrentFirstBlockRing.h"
#include "../src/FastqParser.h"
#include "../src/LowFrequencyReader.h"
#include "../src/ParserThreadPool.h"
#include "../src/ExportWriter.h"
#include "../src/ClassifierThreadPool.h"

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

static size_t tree_record_bytes(uint32_t depth = 0)
{ return std::max(size_t{1}, kmer_bytes() - std::min(kmer_bytes(), size_t{1} + depth / 2)); }

static Key read_tree_key(const unsigned char* bytes, const node<uint32_t>* owner, uint32_t depth = 0)
{
    Key full(kmer_bytes());
    const size_t cut = std::min(full.size(), size_t{1} + depth / 2);
    for (size_t i = 0; i < cut; ++i)
        full[i] = owner->prefix_code >> (8 + 4 * depth - 8 * (i + 1));
    if (cut < full.size()) std::memcpy(full.data() + cut, bytes, full.size() - cut);
    return full;
}

static void store_tree_key(unsigned char* dest, const Key& full, uint32_t depth)
{
    const size_t cut = std::min(full.size(), size_t{1} + depth / 2);
    if (cut == full.size()) dest[0] = 0;
    else std::memcpy(dest, full.data() + cut, full.size() - cut);
}

static void initialize_children(node<uint32_t>* parent, node<uint32_t>* children)
{
    for (uint32_t i = 0; i < 16; ++i) children[i].prefix_code = (parent->prefix_code << 4) | i;
}

static std::vector<unsigned char> flatten(const std::vector<Key>& keys)
{
    std::vector<unsigned char> bytes(keys.size() * kmer_bytes());
    for (size_t i = 0; i < keys.size(); ++i)
        std::memcpy(bytes.data() + i * kmer_bytes(), keys[i].data(), kmer_bytes());
    return bytes;
}

struct FastqClassifierTestAccess
{
    static void feed(FastqClassifier<uint32_t>& classifier, std::vector<Key>& keys, bool owned)
    {
        const auto bytes = flatten(keys);
        const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
        for (size_t pos = 0; pos < keys.size(); pos += capacity) {
            const size_t count = std::min(capacity, keys.size() - pos);
            if (owned) classifier.process_owned_block(bytes.data() + pos * kmer_bytes(), count);
            else classifier.process_other_block(bytes.data() + pos * kmer_bytes(), count);
        }
    }
    static auto* roots(FastqClassifier<uint32_t>& classifier) { return classifier.local_root_nodes.data(); }
    static const auto& prefix_counts(FastqClassifier<uint32_t>& classifier) { return classifier.local_block_prefix_counts; }
    static void export_batch(FastqClassifier<uint32_t>& classifier, const unsigned char* records, size_t count)
    { classifier.export_kmer_batch(records, count); }
    static size_t export_count(const FastqClassifier<uint32_t>& classifier) { return classifier.export_kmer_block_count; }
    static const auto* export_block(const FastqClassifier<uint32_t>& classifier) { return classifier.export_block_ptr; }
    static void collect_partial_export(FastqClassifier<uint32_t>& classifier, std::map<Key, uint64_t>& counts)
    {
        for (size_t i = 0; i < classifier.export_kmer_block_count; ++i)
            ++counts[read_key(classifier.export_block_ptr->k_mers.data() + i * kmer_bytes())];
    }
};

namespace {
struct ScopedCountMax
{
    uint32_t previous = count_max;
    ~ScopedCountMax() { count_max = previous; }
};

void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

struct TemporaryDirectory
{
    std::filesystem::path path;
    TemporaryDirectory()
    {
        auto name_string = (std::filesystem::temp_directory_path() / "tree-first-drain-test-XXXXXX").string();
        std::vector<char> name(name_string.begin(), name_string.end());
        name.push_back(0);
        const char* created = ::mkdtemp(name.data());
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

struct ParserTransferCase
{
    uint32_t parsers = 1;
    uint32_t owners = 110;
    size_t free_blocks = PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY;
    bool before_eof = false;
    bool defer_consumer = false;
};

void check_parser_transfer(uint32_t k, const std::vector<std::string>& sequences,
    const ParserTransferCase& scenario = {})
{
    k_length = k;
    for (size_t prefix = 0; prefix < prefix_owners.size(); ++prefix)
        prefix_owners[prefix] = prefix % scenario.owners;

    size_t input_block_bytes = 64;
    std::map<Key, uint64_t> expected;
    constexpr char bases[] = "ACGT";
    for (const auto& sequence : sequences) {
        input_block_bytes = std::max(input_block_bytes, sequence.size() + 1);
        for (size_t pos = 0; pos + k <= sequence.size(); ++pos) {
            const auto window = sequence.substr(pos, k);
            const std::string reversed(window.rbegin(), window.rend());
            std::string complement = window;
            for (auto& base : complement) base = bases[3 - (std::strchr(bases, base) - bases)];
            const auto& canonical = std::min(reversed, complement);
            Key key(kmer_bytes(), 0);
            for (size_t i = 0; i < canonical.size(); ++i)
                key[i / 4] |= (std::strchr(bases, canonical[i]) - bases) << (6 - 2 * (i % 4));
            ++expected[key];
        }
    }

    RunState state;
    ActiveRunScope active(state);
    RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY> reads(input_block_bytes, 1);
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> parsed(
        PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, scenario.parsers);
    std::vector<char*> held;
    for (size_t i = scenario.free_blocks; i < parsed.capacity(); ++i) {
        char* block;
        parsed.producer_dequeue(block);
        held.push_back(block);
    }
    for (const auto& sequence : sequences) {
        char* block;
        reads.producer_dequeue(block);
        std::memcpy(block, sequence.data(), sequence.size());
        block[sequence.size()] = '\n';
        reads.producer_enqueue({block, sequence.size() + 1});
    }
    if (!scenario.before_eof) reads.producer_set_finished();

    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners;
    for (uint32_t i = 0; i < scenario.owners; ++i)
        owners.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    const size_t capacity = parsed.blockSize() / kmer_bytes();
    std::atomic<bool> full_seen{false};
    std::atomic<bool> consume{!scenario.defer_consumer};
    std::atomic<uint64_t> generated{0};
    std::map<Key, uint64_t> actual;
    size_t global_blocks = 0;
    std::future<void> consumer;
    std::vector<std::future<void>> parsers;
    try {
        consumer = std::async(std::launch::async, [&] {
            run_worker("parser transfer consumer", [&] {
                while (!consume.load(std::memory_order_acquire)) {
                    state.check();
                    std::this_thread::yield();
                }
                for (;;) {
                    state.check();
                    // Observe completion before the final queue drain.
                    const bool finished = parsed.producer_finished();
                    bool worked = false;
                    for (size_t owner = 0; owner <= owners.size(); ++owner) {
                        content_type content;
                        const bool found = owner == owners.size() ? global.try_dequeue(content)
                            : owners[owner]->try_dequeue(content);
                        if (!found) continue;
                        require(content.data && content.length && content.length <= capacity,
                            "parser published an empty or oversized block");
                        const auto* keys = reinterpret_cast<const unsigned char*>(content.data);
                        const auto destination = prefix_owners[keys[0]];
                        require(owner == owners.size() || owner == destination, "parser used the wrong owner queue");
                        for (size_t i = 0; i < content.length; ++i) {
                            require(prefix_owners[keys[i * kmer_bytes()]] == destination,
                                "parser mixed owners within an output block");
                            ++actual[read_key(keys + i * kmer_bytes())];
                        }
                        if (content.length == capacity) full_seen.store(true, std::memory_order_release);
                        if (owner == owners.size()) ++global_blocks;
                        parsed.consumer_enqueue(content.data);
                        worked = true;
                        std::this_thread::yield();
                    }
                    if (!worked && finished) break;
                    if (!worked) std::this_thread::yield();
                }
            });
        });
        for (uint32_t i = 0; i < scenario.parsers; ++i) {
            parsers.push_back(std::async(std::launch::async, [&] {
                run_worker("parser transfer producer", [&] {
                    FastqParser parser(owners, &global, &reads, &parsed);
                    parser.parse_and_push();
                    generated.fetch_add(parser.get_total_read_kmer(), std::memory_order_relaxed);
                    parsed.producer_set_finished();
                });
            }));
        }
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        if (scenario.before_eof) {
            while (!full_seen.load(std::memory_order_acquire) && !state.failed()
                && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            require(full_seen.load(std::memory_order_acquire), "parser retained a full block until EOF");
            reads.producer_set_finished();
        }
        for (auto& parser : parsers)
            require(parser.wait_until(deadline) == std::future_status::ready, "parser waited for output blocks");
        consume.store(true, std::memory_order_release);
        require(consumer.wait_until(deadline) == std::future_status::ready, "parser consumer did not finish");
    } catch (...) {
        state.fail(std::current_exception(), "parser transfer test");
    }
    for (auto& parser : parsers) parser.get();
    if (consumer.valid()) consumer.get();
    state.check();

    require(actual == expected, "parser handoff lost, duplicated, or corrupted records");
    uint64_t expected_records = 0;
    for (const auto& [key, count] : expected) expected_records += count;
    require(generated.load() == expected_records, "parser handoff changed the input count");
    if (scenario.defer_consumer) require(global_blocks > 0, "parser did not use the global fallback queue");
    for (char* block : held) parsed.consumer_enqueue(block);
    std::set<char*> returned;
    char* block;
    while (parsed.producer_try_dequeue(block)) {
        require(block && returned.insert(block).second, "parser returned a null or duplicate block");
    }
    require(returned.size() == parsed.capacity(), "parser handoff leaked an output block");
}

void check_parser_transfers()
{
    check_parser_transfer(33, {}, {.free_blocks = 0});
    check_parser_transfer(33, {std::string(33, 'A')}, {.free_blocks = 1});
    check_parser_transfer(33, {std::string(33, 'A'), std::string(33, 'C')}, {.free_blocks = 1});
    for (uint32_t k : {1u, 33u, 128u}) {
        const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / ((k + 3) / 4);
        for (size_t records : {capacity - 1, capacity, capacity + 1})
            check_parser_transfer(k, {std::string(records + k - 1, 'A')});
    }
    check_parser_transfer(1, {std::string(PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE + 1, 'A')},
        {.before_eof = true});
    std::vector<std::string> sequences;
    constexpr char bases[] = "ACGT";
    for (size_t repeat = 0; repeat < 19; ++repeat) {
        for (uint32_t owner = 0; owner < 110; ++owner) {
            // The chosen key starts below T and ends in A, so reversal wins.
            std::string key(33, 'A');
            for (uint32_t i = 0; i < 4; ++i) key[i] = bases[(owner >> (6 - 2 * i)) & 3];
            sequences.emplace_back(key.rbegin(), key.rend());
        }
    }
    check_parser_transfer(33, sequences, {.parsers = 19, .free_blocks = 8});
    sequences.assign(CLASSIFIER_TASK_QUEUES_CAPACITY + 1,
        std::string(PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE + 1, 'A'));
    check_parser_transfer(1, sequences, {.defer_consumer = true});
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
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY> inputs{PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, 0};
    Shared() { pool.init_arenas(); }
};

struct Fixture
{
    Shared& shared;
    ConcurrentMemoryPool pool{64ULL << 20};
    std::unique_ptr<LayerQueues<uint32_t>> layers = std::make_unique<LayerQueues<uint32_t>>();
    std::unique_ptr<KmerTree<uint32_t>> tree;
    std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>> local =
        std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>();
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    std::unique_ptr<FastqClassifier<uint32_t>> classifier;
    Fixture(Shared& state, uint32_t k) : shared(state)
    {
        k_length = k;
        pool.init_arenas();
        prefix_owners.fill(0);
        bloom_filter_capacity.fill(4096);
        tree = std::make_unique<KmerTree<uint32_t>>( &pool, layers.get(), &shared.exports);
        classifier = std::make_unique<FastqClassifier<uint32_t>>(0, &shared.inputs, local.get(), &global,
            tree.get(), &pool);
        classifier->initialize_shared_bloom();
    }
};

void collect_roots(node<uint32_t>* roots, std::map<Key, uint64_t>& counts)
{
    for (uint32_t prefix = 0; prefix < 256; ++prefix) {
        for (size_t b = 0; b < roots[prefix].count; ++b) {
            auto* block = roots[prefix].kmer_blocks[b];
            for (size_t i = 0; i < block->count; ++i) {
                const auto key = read_tree_key(block->k_mers.data() + i * tree_record_bytes(), roots + prefix);
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

void check_export_batches(Shared& shared, uint32_t k)
{
    using Access = FastqClassifierTestAccess;
    Fixture fixture(shared, k);
    auto& classifier = *fixture.classifier;
    const size_t width = kmer_bytes();
    const size_t capacity = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / width;
    std::vector<Key> keys;
    for (size_t i = 0; i < 3 * capacity + 17; ++i) keys.push_back(make_key(k, i + 1));
    const auto records = flatten(keys);
    std::vector<unsigned char> actual;
    size_t appended = 0;
    for (size_t count : {size_t{0}, capacity - 3, size_t{16}, capacity + 4, capacity - 17, size_t{0}, size_t{17}}) {
        Access::export_batch(classifier, records.data() + appended * width, count);
        appended += count;
        require(classifier.total_kmers_exported == appended, "batch export statistics differ");
        require(Access::export_count(classifier) == appended % capacity, "batch export tail count differs");
        const auto* tail = Access::export_block(classifier);
        require(tail != nullptr, "batch export did not acquire the next empty block");
        content_type content;
        while (shared.exports.consumer_try_dequeue(content)) {
            require(content.length == capacity, "batch export submitted a partial block early");
            const auto* block = reinterpret_cast<const ExportBlock*>(content.data);
            require(block != tail, "batch export retained a submitted full block");
            actual.insert(actual.end(), block->k_mers.data(), block->k_mers.data() + content.length * width);
            shared.exports.consumer_enqueue(content.data);
        }
        require(actual.size() == (appended / capacity) * capacity * width, "batch export lost a full block");
        require(std::equal(actual.begin(), actual.end(), records.begin()), "batch export changed full-block byte order");
        require(std::equal(tail->k_mers.begin(), tail->k_mers.begin() + Access::export_count(classifier) * width,
            records.begin() + actual.size()), "batch export changed tail byte order");
    }
    classifier.classify_first_pass();
    content_type content;
    require(shared.exports.consumer_try_dequeue(content) && content.length == 17,
        "batch export did not submit the final partial block");
    const auto* block = reinterpret_cast<const ExportBlock*>(content.data);
    actual.insert(actual.end(), block->k_mers.data(), block->k_mers.data() + content.length * width);
    shared.exports.consumer_enqueue(content.data);
    require(actual == records, "batch export lost, duplicated, or reordered bytes");
    require(!shared.exports.consumer_try_dequeue(content), "batch export submitted an extra block");
    require(Access::export_count(classifier) == 0 && Access::export_block(classifier) == nullptr,
        "batch export retained the submitted tail");
    require(classifier.total_kmers_exported == keys.size(), "batch export final statistics differ");
}

void check_root_block_rollover(Shared& shared)
{
    Fixture fixture(shared, 33);
    auto* roots = FastqClassifierTestAccess::roots(*fixture.classifier);
    constexpr uint32_t prefix = 57;
    auto key = make_key(33, 999);
    key[0] = ((uint64_t{prefix} << 56)) >> 56;
    const size_t block_capacity = 4092 / tree_record_bytes();
    const size_t count = (MAX_KMER_BLOCK_NUM + 1) * block_capacity + 3;
    std::vector<Key> keys(count, key);
    fixture.tree->main_add_kmers_to_local_root(flatten(keys).data(), 0, prefix, roots);
    require(roots[prefix].count == 0, "empty root append allocated a tree block");
    fixture.tree->main_add_kmers_to_local_root(flatten(keys).data(), keys.size(), prefix, roots);
    require(fixture.layers->size(0) == 1, "root rollover did not account for its queued task");
    Task<uint32_t> task;
    require(fixture.layers->get_queue(0)->try_dequeue(task), "full local root was not queued");
    require(task.current_node == fixture.tree->root_nodes + prefix && task.depth == 0,
        "rollover task points at the wrong global root");
    require(task.count == MAX_KMER_BLOCK_NUM, "rollover task omitted full blocks");
    std::map<Key, uint64_t> actual;
    for (size_t b = 0; b < task.count; ++b)
    {
        require(task.kmer_blocks[b]->count == block_capacity, "rollover published a partial block");
        for (size_t i = 0; i < task.kmer_blocks[b]->count; ++i)
            ++actual[read_tree_key(task.kmer_blocks[b]->k_mers.data() + i * tree_record_bytes(task.depth), task.current_node, task.depth)];
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

void check_compaction_runs(Shared& shared, uint32_t k)
{
    using Access = FastqClassifierTestAccess;
    // Missing tail flushes, overlapping memcpy, or overwriting the current
    // FIRST/miss must change the exact tree/export multisets below.
    for (int mode : {0, 1}) {
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
                        bloom.get_bins()[probe.block_idx].store(probe.insert_num);
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
            Access::feed(classifier, keys, mode == 0);
            require(Access::prefix_counts(classifier) == expected_prefixes,
                "run compaction changed per-prefix retained counts");
        }
        std::map<Key, uint64_t> actual_tree, actual_export;
        collect_roots(Access::roots(classifier), actual_tree);
        collect_exports(shared, actual_export);
        Access::collect_partial_export(classifier, actual_export);
        require(actual_tree == expected_tree, "run compaction changed retained keys or frequencies");
        require(actual_export == expected_export, "run compaction overwrote FIRST or exported records");
        require(classifier.total_kmers_send_to_tree == retained &&
            classifier.total_kmers_send_to_tree + classifier.total_kmers_exported == total,
            "run compaction did not conserve records");
        classifier.classify_first_pass();
        std::map<Key, uint64_t> returned;
        collect_exports(shared, returned);
    }
}

void check_final_drain_filter(Shared& shared, bool hashed, uint32_t minimum, uint32_t maximum)
{
    constexpr uint32_t k = 33;
    k_length = k;
    constexpr uint32_t prefix = 57;
    ScopedCountMax restore_cap;
    const uint32_t saved_min = filter_min, saved_max = filter_max;
    filter_min = minimum;
    filter_max = maximum;
    count_max = 4;
    Fixture fixture(shared, k);
    const auto below_before = filtered_below_min_count.load();
    const auto above_before = filtered_above_max_count.load();
    std::array<Key, 4> keys;
    const std::array<uint32_t, 4> counts{1, 2, 4, 5};
    std::map<Key, uint32_t> expected;
    uint64_t below = 0, above = 0;
    std::vector<std::unique_ptr<node<uint32_t>[]>> slabs;
    std::unique_ptr<ConcurrentOpenAddressHashMap<uint32_t>> map;
    if (hashed) {
        auto* current = fixture.tree->root_nodes + prefix;
        for (uint32_t depth = 0; depth + 1 < MAX_DEPTH; ++depth) {
            auto* children = current->children_ptr.load();
            if (!children) {
                slabs.push_back(std::make_unique<node<uint32_t>[]>(1ULL << (2 * NODE_BASES)));
                children = slabs.back().get();
                initialize_children(current, children);
                current->children_ptr.store(children);
            }
            current = children;
        }
        ConcurrentOpenAddressHashMap<uint32_t>::set_memory_pool(&fixture.pool);
        map = std::make_unique<ConcurrentOpenAddressHashMap<uint32_t>>(64, count_max, tree_record_bytes(MAX_DEPTH - 1));
        current->hash_map.store(map.get());
    }
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = make_key(k, i + 1);
        keys[i][0] = ((uint64_t{prefix} << 56)) >> 56;
        if (hashed) {
            keys[i][1] = 0; keys[i][2] &= 0x0f;
            Key suffix(tree_record_bytes(MAX_DEPTH - 1));
            store_tree_key(suffix.data(), keys[i], MAX_DEPTH - 1);
            const auto hash = ConcurrentOpenAddressHashMap<uint32_t>::hash_key(suffix.data(), suffix.size());
            uint64_t inserted = 0;
            require(map->try_increment(hash, suffix.data(), ConcurrentOpenAddressHashMap<uint32_t>::fingerprint(hash),
                counts[i], inserted) == ConcurrentOpenAddressHashMap<uint32_t>::InsertResult::INSERTED,
                "failed to seed final drain hash map");
        } else {
            std::vector<Key> copies(counts[i], keys[i]);
            fixture.tree->main_add_kmers_to_local_root(flatten(copies).data(), copies.size(), prefix, fixture.tree->root_nodes);
        }
        const auto count = std::min(counts[i], count_max);
        expected.emplace(keys[i], count);
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

    std::map<Key, uint32_t> actual;
    content_type content;
    while (output.consumer_try_dequeue(content)) {
        for (const auto& record : high_prefix_test::decode(
                {reinterpret_cast<const uint8_t*>(content.data), content.length}, kmer_bytes(), count_max_bytes)) {
            require(actual.emplace(record.key, record.count).second, "final drain duplicated a key");
        }
        output.consumer_enqueue(content.data);
    }
    require(actual == expected, "final drain discarded an unfiltered saturated record");
}

void check_final_drain_runs(Shared& shared, uint32_t k)
{
    k_length = k;
    ScopedCountMax restore_cap;
    const auto saved_min = filter_min, saved_max = filter_max;
    const auto saved_count_bytes = count_max_bytes;
    RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY> output(FINAL_DRAIN_RING_POOL_BLOCK_SIZE, 1);
    const size_t capacity = 4092 / tree_record_bytes();
    const uint32_t padding = (4 - k % 4) % 4 * 2;
    for (uint32_t count_bytes : {1u, 2u, 4u}) {
        count_max_bytes = count_bytes;
        count_max = count_bytes == 1 ? 255 : count_bytes == 2 ? 65535 : UINT32_MAX;
        Fixture fixture(shared, k);
        auto* root = fixture.tree->root_nodes + 57;
        std::array<node<uint32_t>*, MAX_DEPTH> path{root, root->children_ptr.load()};
        std::vector<std::unique_ptr<node<uint32_t>[]>> slabs;
        for (uint32_t depth = 2; depth < MAX_DEPTH; ++depth) {
            slabs.push_back(std::make_unique<node<uint32_t>[]>(16));
            path[depth] = slabs.back().get();
            initialize_children(path[depth - 1], path[depth]);
            path[depth - 1]->children_ptr.store(path[depth]);
        }
        for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth) {
            const size_t capacity = 4092 / tree_record_bytes(depth);
            auto* leaf = path[depth];
            auto* children = leaf->children_ptr.exchange(nullptr);
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
                            store_tree_key(block->k_mers.data() + i * tree_record_bytes(depth), key, depth);
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
                        store_tree_key(block->k_mers.data(), key, 0);
                        root->kmer_blocks[0] = block;
                        root->count = 1;
                        root->active_block = block;
                        ++truth[key];
                    }
                    uint64_t below = 0, above = 0;
                    for (const auto& [key, frequency] : truth) {
                        const auto count = std::min(frequency, count_max);
                        expected.emplace(key, count);
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
                    while (output.consumer_try_dequeue(content)) {
                        for (const auto& record : high_prefix_test::decode(
                                {reinterpret_cast<const uint8_t*>(content.data), content.length}, kmer_bytes(), count_max_bytes)) {
                            require(record.prefix_bytes == std::min(size_t{1} + depth / 2, kmer_bytes() - 1),
                                "drain did not select the node prefix length");
                            require(actual.emplace(record.key, record.count).second, "final drain emitted a duplicate key");
                        }
                        output.consumer_enqueue(content.data);
                    }
                    require(actual == expected, "final drain changed the complete key/frequency mapping");
                }
            }
            leaf->children_ptr.store(children);
        }
    }
    // Two large leaves share one writer, crossing its 128 KiB block boundary.
    filter_min = 1;
    filter_max = count_max = UINT32_MAX;
    count_max_bytes = 4;
    Fixture fixture(shared, k);
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
                    store_tree_key(block->k_mers.data() + i * tree_record_bytes(), key, 0);
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
    while (output.consumer_try_dequeue(content)) {
        ++output_blocks;
        for (const auto& record : high_prefix_test::decode(
                {reinterpret_cast<const uint8_t*>(content.data), content.length}, kmer_bytes(), count_max_bytes)) {
            require(actual.emplace(record.key, record.count).second, "large drain emitted a duplicate key");
        }
        output.consumer_enqueue(content.data);
    }
    require(output_blocks >= 2 && actual == expected, "large drain lost records across output blocks");
    filter_min = saved_min;
    filter_max = saved_max;
    count_max_bytes = saved_count_bytes;
}

void check_direct_first(Shared& shared, uint32_t k, bool owned, bool false_positive)
{
    Fixture fixture(shared, k);
    auto& classifier = *fixture.classifier;
    Key key = make_key(k, 99871);
    auto& bloom = *static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
    if (false_positive) {
        const auto probe = bloom.prepare_insert(key.data());
        bloom.get_bins()[probe.block_idx].store(probe.insert_num);
    }
    uint64_t total = 0;
    for (size_t count : {1u, 1u, 5u}) {
        std::vector<Key> keys(count, key);
        FastqClassifierTestAccess::feed(classifier, keys, owned);
        total += count;
        std::map<Key, uint64_t> low, tree;
        collect_exports(shared, low);
        FastqClassifierTestAccess::collect_partial_export(classifier, low);
        collect_roots(FastqClassifierTestAccess::roots(classifier), tree);
        require(low[key] == (false_positive ? 0u : 1u), "FIRST was cached, duplicated, or exported after a false positive");
        require(tree[key] == total - low[key], "FIRST/REPEAT split lost an occurrence");
    }
    classifier.classify_first_pass();
    std::map<Key, uint64_t> tail;
    collect_exports(shared, tail);
    require(tail[key] == (false_positive ? 0u : 1u), "FIRST tail was not submitted exactly once");
    require(fixture.pool.first_valid_records.load() == 0 && fixture.pool.spill_bytes.load() == 0,
        "direct output allocated FIRST data storage");
}

void check_classifier_lifetime(Shared& shared)
{
    using namespace std::chrono_literals;
    k_length = 33;
    for (uint32_t round = 0; round < 16; ++round) {
        RunState state;
        ActiveRunScope active(state);
        for (uint32_t prefix = 0; prefix < 256; ++prefix) prefix_owners[prefix] = prefix % 2;
        bloom_filter_capacity.fill(4096);
        auto layers = std::make_unique<LayerQueues<uint32_t>>();
        KmerTree<uint32_t> tree(&shared.pool, layers.get(), &shared.exports);
        SchedulerThreadPool<uint32_t> scheduler(2, 2, 0, &tree, layers.get());
        std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners;
        for (size_t i = 0; i < 2; ++i) owners.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
        ClassifierThreadPool<uint32_t> workers(&tree, &shared.inputs, owners, &global, &scheduler, &shared.pool, 2);
        shared.inputs.reset_producers(1);
        char* block;
        shared.inputs.producer_dequeue(block);
        const size_t count = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
        std::map<Key, uint64_t> expected, actual;
        for (size_t i = 0; i < count; ++i) {
            Key key = make_key(33, i % 8 + 1);
            key[0] = i % 2;
            std::memcpy(block + i * kmer_bytes(), key.data(), kmer_bytes());
            ++expected[key];
        }
        // At EOF one worker claims the only global block; the other must
        // retain its Bloom objects until that mixed-prefix block is finished.
        require(global.try_enqueue({block, count}), "could not publish lifetime input");
        shared.inputs.producer_set_finished();
        workers.start();
        auto done = std::async(std::launch::async, [&] { workers.join(); });
        if (done.wait_for(5s) != std::future_status::ready)
            state.fail(std::make_exception_ptr(std::runtime_error("Classifier EOF barrier hung")), "test");
        done.get();
        collect_exports(shared, actual);
        uint64_t low_count = 0;
        for (const auto& [key, n] : actual) {
            require(n == 1, "competing Classifiers emitted duplicate FIRSTs");
            low_count += n;
        }
        require(low_count == workers.low_output_records, "thread join did not aggregate low output");
        Task<uint32_t> task;
        while (layers->get_queue(0)->try_dequeue(task)) {
            for (size_t b = 0; b < task.count; ++b) {
                auto* records = task.kmer_blocks[b];
                for (size_t i = 0; i < records->count; ++i)
                    ++actual[read_tree_key(records->k_mers.data() + i * tree_record_bytes(), task.current_node)];
                shared.pool.deallocate(records);
            }
        }
        collect_roots(tree.root_nodes, actual);
        require(actual == expected, "cross-Classifier EOF lost or duplicated occurrences");
    }
}

void check_export_cancellation(Shared& shared)
{
    using namespace std::chrono_literals;
    RunState state;
    ActiveRunScope active(state);
    Fixture fixture(shared, 33);
    std::vector<char*> held;
    char* block;
    while (shared.exports.producer_try_dequeue(block)) held.push_back(block);
    const size_t count = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
    std::vector<Key> keys(count, make_key(33, 171));
    const auto bytes = flatten(keys);
    auto blocked = std::async(std::launch::async, [&] {
        FastqClassifierTestAccess::export_batch(*fixture.classifier, bytes.data(), count);
    });
    const bool waited = blocked.wait_for(20ms) == std::future_status::timeout;
    state.fail(std::make_exception_ptr(std::runtime_error("writer failed")), "test");
    bool cancelled = false;
    try { blocked.get(); } catch (const std::runtime_error&) { cancelled = true; }
    require(waited && cancelled, "full Export Ring ignored cancellation");
    for (char* held_block : held) shared.exports.consumer_enqueue(held_block);
    std::map<Key, uint64_t> submitted;
    collect_exports(shared, submitted);
}

void check_cancelled_classifier_lifetime(Shared& shared)
{
    using namespace std::chrono_literals;
    for (uint32_t round = 0; round < 32; ++round) {
        // Keep the same RunState for workers, barriers and the injected error.
        ConcurrentMemoryPool pool(32ULL << 20);
        auto& state = pool.run_state();
        ActiveRunScope active(state);
        pool.init_arenas();
        k_length = 1;
        for (uint32_t prefix = 0; prefix < 256; ++prefix) prefix_owners[prefix] = (prefix / 64) % 2;
        bloom_filter_capacity.fill(4096);
        auto layers = std::make_unique<LayerQueues<uint32_t>>();
        KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
        SchedulerThreadPool<uint32_t> scheduler(2, 2, 0, &tree, layers.get());
        std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> owners;
        for (size_t i = 0; i < 2; ++i) owners.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
        ClassifierThreadPool<uint32_t> workers(&tree, &shared.inputs, owners, &global, &scheduler, &pool, 2);
        shared.inputs.reset_producers(1);
        char* block;
        shared.inputs.producer_dequeue(block);
        for (size_t i = 0; i < PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE; ++i)
            block[i] = (i % 2) * 64;
        require(global.try_enqueue({block, PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE}), "could not publish cancellation input");
        shared.inputs.producer_set_finished();
        workers.start();
        // One worker owns the only global task. The other's empty output tail
        // is an observable handoff before its finish barrier, while its Bloom
        // objects are still needed by that mixed-prefix task.
        content_type output;
        const auto deadline = std::chrono::steady_clock::now() + 5s;
        while (!shared.exports.consumer_try_dequeue(output)) {
            require(std::chrono::steady_clock::now() < deadline, "empty worker did not submit its tail");
            std::this_thread::yield();
        }
        shared.exports.consumer_enqueue(output.data);
        state.fail(std::make_exception_ptr(std::runtime_error("writer failed during shared Bloom access")), "test");
        bool rejected = false;
        try { workers.join(); } catch (const std::runtime_error&) { rejected = true; }
        require(rejected, "cancelled Classifiers reported successful completion");
        std::map<Key, uint64_t> discarded;
        collect_exports(shared, discarded);
    }
}
}

int main(int argc, char** argv)
{
    try {
        const bool focused = argc == 2 && std::strcmp(argv[1], "--single-bloom-only") == 0;
        if (!focused) {
            check_parser_transfers();
            check_parser_returns_empty_blocks();
        }
        Shared shared;
        for (uint32_t k : {1u, 31u, 33u, 65u, 128u}) check_export_batches(shared, k);
        for (uint32_t k : {31u, 33u, 65u, 128u}) {
            for (bool owned : {true, false}) {
                check_direct_first(shared, k, owned, false);
                check_direct_first(shared, k, owned, true);
            }
            check_compaction_runs(shared, k);
        }
        check_classifier_lifetime(shared);
        check_export_cancellation(shared);
        check_cancelled_classifier_lifetime(shared);
        if (!focused) {
            check_root_block_rollover(shared);
            for (uint32_t k : {33u, 65u, 81u, 128u})
                std::async(std::launch::async, [&] { check_final_drain_runs(shared, k); }).get();
            for (bool hashed : {false, true})
                for (auto [minimum, maximum] : {std::pair{2u, 3u}, {2u, 2u}, {1u, 1u}, {4u, 4u}, {0u, 0u}, {5u, 10u}})
                    std::async(std::launch::async, [&] { check_final_drain_filter(shared, hashed, minimum, maximum); }).get();
        }
        std::cout << "ClassifierFirstBatch tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
