#include "low_msd_test_helpers.h"
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

struct ConcurrentFirstBlockRingTestAccess
{
    static size_t capacity(const ConcurrentFirstBlockRing& ring) { return ring.slot_count_; }
};

struct FastqClassifierTestAccess
{
    static void feed(FastqClassifier<uint32_t>& classifier, std::vector<Key>& keys, bool owned)
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
    static uint64_t flush(FastqClassifier<uint32_t>& classifier) { return classifier.flush_evicted_batch<WRITE_MISSES>(); }
    static void group_and_decode(FastqClassifier<uint32_t>& classifier) { classifier.group_evicted(); }
    static const auto* grouped(FastqClassifier<uint32_t>& classifier) { return classifier.evicted_grouped_; }
    template<bool WRITE_MISSES = true>
    static uint64_t finish(FastqClassifier<uint32_t>& classifier) { return classifier.drain_first_occurrences<WRITE_MISSES>(); }
    static size_t block_capacity(FastqClassifier<uint32_t>& classifier) { return classifier.first_block_capacity_; }
    static size_t ring_capacity(FastqClassifier<uint32_t>& classifier)
    { return ConcurrentFirstBlockRingTestAccess::capacity(*classifier.first_occurrence_ring_) * block_capacity(classifier); }
    static void push(FastqClassifier<uint32_t>& classifier, const Key& key) { classifier.push_first_occurrence(key.data()); }
    static size_t batch_capacity(FastqClassifier<uint32_t>& classifier) { return classifier.evicted_capacity_; }
    static size_t pending(FastqClassifier<uint32_t>& classifier) { return classifier.evicted_count_; }
    static const auto& prefix_counts(FastqClassifier<uint32_t>& classifier, bool evicted)
    { return evicted ? classifier.evicted_prefix_counts_ : classifier.local_block_prefix_counts; }
    static auto& packed(FastqClassifier<uint32_t>& classifier) { return classifier.evicted_buffer_; }
    static void stage(FastqClassifier<uint32_t>& classifier, const std::vector<Key>& keys)
    {
        for (size_t i = 0; i < keys.size(); ++i)
            std::memcpy(classifier.evicted_buffer_.data() + i * kmer_bytes(), keys[i].data(), kmer_bytes());
        classifier.evicted_count_ = keys.size();
    }
    static auto* roots(FastqClassifier<uint32_t>& classifier) { return classifier.local_root_nodes.data(); }
    template<bool WRITE_LOW = false>
    static void low_block(FastqClassifier<uint32_t>& classifier, const content_type& content, size_t& hit_count)
    { classifier.reconcile_low_block<WRITE_LOW>(content, hit_count); }
    static void flush_low(FastqClassifier<uint32_t>& classifier, size_t& hit_count)
    { classifier.flush_recovered_tree_batch(hit_count); }
    static void export_batch(FastqClassifier<uint32_t>& classifier, const unsigned char* records, size_t count)
    { classifier.export_kmer_batch(records, count); }
    static void acquire_export_block(FastqClassifier<uint32_t>& classifier, char*& block)
    { classifier.dequeue_data_to_export_writer(block); }
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
    // Reused for decoded records during the reconciliation tests.
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
    PhaseBarrier start{1};
    std::unique_ptr<FastqClassifier<uint32_t>> classifier;
    Fixture(Shared& state, uint32_t k) : shared(state)
    {
        k_length = k;
        pool.init_arenas();
        prefix_owners.fill(0);
        bloom_filter_capacity.fill(4096);
        tree = std::make_unique<KmerTree<uint32_t>>( &pool, layers.get(), &shared.exports);
        classifier = std::make_unique<FastqClassifier<uint32_t>>(0, &shared.inputs, local.get(), &global,
            tree.get(), &pool, start,
            ConcurrentFirstBlockRing::create(&pool, 0, 1,
                align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
                    + 2 * ConcurrentFirstBlockRing::BLOCK_BYTES), 0);
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
        std::ofstream out(input_path.string() + ".0", std::ios::binary);
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
        writer = std::make_unique<ExportWriter>(&shared.exports, output_path.string(), ExportWriter::OutputKind::FinalLow);
        writer->start();
    }
    parsers.start_low_pass(reads, 1);
    reader.start(input_path.string(), 1);
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
        const auto actual = low_msd_test::read(output_path, width);
        require(actual.size() == 1 && actual[0] == pending[1], "low output contains the wrong singleton");
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
    ExportWriter pending_writer(&shared.exports, pending_path.string(), ExportWriter::OutputKind::Pending);
    pending_writer.start();
    classifier.classify_first_pass();
    shared.exports.producer_set_finished();
    pending_writer.join();
    require(classifier.repeat_query_count == runtime_queries, "first pass prematurely drained FIRST");
    require(std::filesystem::file_size(pending_path.string() + ".0") == pending.size() * kmer_bytes(),
        "terminal FIRST misses reached the pending file");

    const uint64_t direct_singletons = expected_low.size();
    shared.inputs.reset_producers(0);
    if (has_pending) {
        char* block;
        shared.inputs.producer_dequeue(block);
        std::ifstream input(pending_path.string() + ".0", std::ios::binary);
        input.read(block, kmer_bytes());
        require(input.good(), "failed to read pending test record");
        fixture.local->enqueue({block, 1});
        ++expected_low[pending[0]];
    }
    std::vector<char*> reserved;
    std::unique_ptr<ExportWriter> final_writer;
    if constexpr (WRITE_LOW) {
        shared.exports.reset_producers(1);
        final_writer = std::make_unique<ExportWriter>(&shared.exports, final_path.string(), ExportWriter::OutputKind::FinalLow);
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
        for (const auto& key : low_msd_test::read(final_path, kmer_bytes())) ++actual_low[key];
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
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree( &shared.pool, layers.get(), &shared.exports);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local_a, local_b;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    PhaseBarrier start(2);
    auto* ring = ConcurrentFirstBlockRing::create(&shared.pool, 0, 2,
        align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
            + 3 * ConcurrentFirstBlockRing::BLOCK_BYTES);
    auto create = [&](uint32_t index, auto* local) {
        return std::make_unique<FastqClassifier<uint32_t>>(index, &shared.inputs, local, &global,
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
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree( &shared.pool, layers.get(), &shared.exports);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local_a, local_b;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    PhaseBarrier start(2);
    auto* ring = ConcurrentFirstBlockRing::create(&shared.pool, 0, 2,
        align_up(sizeof(ConcurrentFirstBlockRing) + sizeof(std::atomic<uint8_t*>), PAGE_SIZE)
            + 3 * ConcurrentFirstBlockRing::BLOCK_BYTES);
    auto create = [&](uint32_t index, auto* local) {
        return std::make_unique<FastqClassifier<uint32_t>>(index, &shared.inputs, local, &global,
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
    const bool returned_early = finishing_a.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
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
    require(returned_early, "first pass still waits at an inter-Classifier barrier");
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
void check_dynamic_safe_points(Shared& shared, bool enabled)
{
    k_length = 33;
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    PhaseBarrier barrier(1);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    FastqClassifier<uint32_t> classifier(0, &shared.inputs, &local, &global, &tree, &pool, barrier,
        nullptr, 0, enabled);
    auto first = make_key(33, 8831), next = make_key(33, 12931);
    std::vector<Key> input{first};
    FastqClassifierTestAccess::feed(classifier, input, true);
    if (enabled) {
        require(!pool.first_empty(), "FIRST record did not go directly into an owned shared segment");
    }
    input.assign(1000, first);
    FastqClassifierTestAccess::feed(classifier, input, false);
    if (enabled) {
        require(!pool.first_empty(), "normal Tree calls lost the active FIRST segment");
    }
    TemporaryDirectory directory;
    shared.exports.reset_producers(1);
    ExportWriter writer(&shared.exports, (directory.path / "pending").string(), ExportWriter::OutputKind::Pending, &pool);
    writer.start();
    pool.seal_thread_first(); // The fixture is about to wait while owning TLS activity.
    pool.request_first_drain();
    SpinBackoff<> backoff;
    while (!pool.first_empty()) {
        pool.run_state().check();
        backoff.backoff();
    }
    // Resume after Writer has invalidated the old generation and returned pages.
    input.assign(1000, first);
    FastqClassifierTestAccess::feed(classifier, input, true);
    input.assign(1, next);
    FastqClassifierTestAccess::feed(classifier, input, false);
    classifier.classify_first_pass();
    tree.mark_finish_export();
    writer.join();
    pool.freeze_first();
    classifier.process_memory_first<false>();
    require(pool.first_empty(), "memory phase retained a partial FIRST segment");
    require(std::filesystem::file_size(directory.path / "pending.0") / kmer_bytes()
            + classifier.first_drain_singletons + classifier.first_memory_to_tree == 2,
        "FIRST=0/shrink/resume changed pending responsibility");
    std::map<Key, uint64_t> roots;
    collect_roots(FastqClassifierTestAccess::roots(classifier), roots);
    // The first FIRST was spilled before Bloom froze; this fixture stops before
    // pending replay. Both batches of REPEATs must still be owned by Tree.
    require(roots == std::map<Key, uint64_t>{{first, 2000}}, "REPEAT ownership changed around reclaim");
    shared.exports.reset_producers(1);
}

void check_memory_first_batches(Shared& shared, bool keep_singletons)
{
    using Access = FastqClassifierTestAccess;
    k_length = 33;
    const auto saved_min = filter_min;
    filter_min = keep_singletons ? 1 : 2;
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    PhaseBarrier barrier(1);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    FastqClassifier<uint32_t> classifier(0, &shared.inputs, &local, &global, &tree, &pool, barrier, nullptr, 0, true);
    std::vector<Key> first, repeated;
    std::set<std::pair<uint32_t, uint64_t>> occupied;
    for (size_t seed = 0; first.size() < 5000; ++seed) {
        const auto key = make_key(33, seed);
        auto& bloom = *static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
        if (!occupied.emplace(key[0], bloom.prepare_insert(key.data()).block_idx).second) continue;
        if (first.size() % 3 == 0) repeated.push_back(key);
        first.push_back(key);
    }
    // No Writer is competing during admission; all FIRST records are resident.
    Access::feed(classifier, first, true);
    Access::feed(classifier, repeated, true);
    require(classifier.first_direct_pending == 0 && !pool.first_empty(), "memory FIRST fixture bypassed RAM");
    TemporaryDirectory directory;
    shared.exports.reset_producers(1);
    ExportWriter writer(&shared.exports, (directory.path / "pending").string(), ExportWriter::OutputKind::Pending, &pool);
    writer.start();
    classifier.classify_first_pass();
    require(!pool.first_empty() && classifier.repeat_query_count == 0,
        "first pass prematurely scanned or emptied memory FIRST");
    tree.mark_finish_export();
    writer.join();
    require(std::filesystem::file_size(directory.path / "pending.0") == 0,
        "resident FIRST leaked into pending before the Writer switch");
    pool.freeze_first();
    std::unique_ptr<ExportWriter> final_writer;
    std::vector<char*> held;
    if (keep_singletons) {
        shared.exports.reset_producers(1);
        final_writer = std::make_unique<ExportWriter>(&shared.exports, (directory.path / "low").string(), ExportWriter::OutputKind::FinalLow);
        final_writer->start();
        classifier.process_memory_first<true>();
    } else {
        char* block;
        while (shared.exports.producer_try_dequeue(block)) held.push_back(block);
        classifier.process_memory_first<false>();
        for (char* data : held) shared.exports.consumer_enqueue(data);
    }
    if (final_writer) {
        tree.mark_finish_export();
        final_writer->join();
    }
    const auto misses = first.size() - repeated.size();
    require(classifier.first_memory_to_tree == repeated.size(), "memory FIRST batches or final tail lost hits");
    require(classifier.first_memory_kept == (keep_singletons ? misses : 0), "memory FIRST kept count differs");
    require(classifier.first_memory_discarded == (keep_singletons ? 0 : misses), "memory FIRST misses took the wrong branch");
    require(classifier.first_drain_singletons == misses
        && classifier.low_records_scanned == 0, "memory FIRST changed file replay statistics");
    if (keep_singletons) {
        std::map<Key, uint64_t> actual_low, expected_low;
        for (const auto& key : first) expected_low[key] = 1;
        for (const auto& key : repeated) expected_low.erase(key);
        for (const auto& key : low_msd_test::read(directory.path / "low", kmer_bytes())) ++actual_low[key];
        require(actual_low == expected_low, "memory FIRST final low multiset changed");
    }
    std::map<Key, uint64_t> actual, expected;
    collect_roots(Access::roots(classifier), actual);
    for (const auto& key : repeated) expected[key] = 2;
    require(actual == expected && pool.first_empty(), "memory FIRST changed Tree counts or retained a segment");
    require(classifier.total_deal_kmer_count == first.size() + repeated.size(), "memory FIRST recounted input records");
    filter_min = saved_min;
    shared.exports.reset_producers(1);
}

enum class FirstScanPattern { Miss, Hit, SingleRoot, Alternating, SplitRuns, LongMissRun, AllRoots };

void check_memory_first_scan(Shared& shared, uint32_t k, size_t count,
    FirstScanPattern pattern, bool keep_singletons, bool replay_pending = false)
{
    using Access = FastqClassifierTestAccess;
    k_length = k;
    const size_t width = kmer_bytes();
    const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / width;
    const size_t segment_capacity = ConcurrentMemoryPool::first_segment_record_capacity();
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    PhaseBarrier barrier(1);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    FastqClassifier<uint32_t> classifier(0, &shared.inputs, &local, &global, &tree, &pool, barrier, nullptr, 0, true);
    std::vector<Key> keys;
    std::vector<std::vector<uint64_t>> bloom_before;
    const size_t roots = pattern == FirstScanPattern::AllRoots ? 256 : 4;
    for (size_t r = 0; r < roots; ++r) {
        auto key = make_key(k, 8831 + r);
        key[0] = roots == 256 ? r : r * 64;
        keys.push_back(key);
        auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
        auto& snapshot = bloom_before.emplace_back(2 * bloom_filter_capacity[key[0]]);
        for (size_t b = 0; b < snapshot.size(); ++b) {
            snapshot[b] = b % 2 ? (0x5a5a5a5a5a5a5a5aULL ^ b)
                : (roots == 256 || r % 2 == 0 ? UINT64_MAX : 0);
            bloom->get_bins()[b].store(snapshot[b]);
        }
    }
    std::map<Key, uint64_t> expected_tree, expected_low;
    size_t hits = 0;
    std::set<size_t> hit_segments;
    for (size_t i = 0; i < count; ++i) {
        const bool hit = pattern == FirstScanPattern::Hit || pattern == FirstScanPattern::SingleRoot
            || pattern == FirstScanPattern::AllRoots
            || (pattern == FirstScanPattern::Alternating && i % 2 == 0)
            || (pattern == FirstScanPattern::LongMissRun && i == EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / width + 3)
            || (pattern == FirstScanPattern::SplitRuns
                && (i < capacity - 1 || i >= capacity - 1 + 3 * Access::batch_capacity(classifier)));
        const auto& key = keys[pattern == FirstScanPattern::SingleRoot ? 0
            : roots == 256 ? i % roots : (i % 3 == 1 ? 2 : 0) + !hit];
        Access::push(classifier, key);
        ++(hit ? expected_tree : expected_low)[key];
        hits += hit;
        if (hit) hit_segments.insert(i / segment_capacity);
    }
    require(classifier.first_tail_records() == count % segment_capacity,
        "memory scan fixture changed active segment ownership");
    require(classifier.first_direct_pending == 0, "memory scan fixture unexpectedly used pending");
    Access::packed(classifier).fill(0xa5);
    const auto packed_before = Access::packed(classifier);
    classifier.classify_first_pass();
    std::map<Key, uint64_t> actual_low;
    collect_exports(shared, actual_low);
    require(actual_low.empty(), "memory scan fixture exported resident FIRSTs");
    pool.freeze_first();
    std::vector<char*> held;
    if (keep_singletons) {
        classifier.process_memory_first<true>();
    } else {
        char* block;
        while (shared.exports.producer_try_dequeue(block)) held.push_back(block);
        classifier.process_memory_first<false>();
        for (auto* data : held) shared.exports.consumer_enqueue(data);
    }
    require(Access::packed(classifier) == packed_before,
        "memory FIRST scan copied input through the old 32KB ring buffer");
    // Whole-segment extraction copies valid records only, including k=33 padding.
    require(pool.first_empty() && classifier.first_tail_records() == 0, "memory scan retained FIRST input");
    require(classifier.repeat_query_count == count && classifier.first_memory_to_tree == hits
        && classifier.recovered_first_count == hits && classifier.total_kmers_send_to_tree == hits,
        "memory scan lost queries or counted Tree hits more than once");
    require(classifier.first_memory_tree_batches == hit_segments.size(),
        "memory scan source-batch statistic differs");
    require(classifier.first_drain_singletons == count - hits
        && classifier.first_memory_kept == (keep_singletons ? count - hits : 0)
        && classifier.first_memory_discarded == (keep_singletons ? 0 : count - hits),
        "memory scan singleton statistics differ");
    collect_exports(shared, actual_low);
    require(actual_low == (keep_singletons ? expected_low : std::map<Key, uint64_t>{}),
        "memory scan changed the singleton multiset");
    if (replay_pending) {
        // The pending pass must reuse the buffers after an incomplete hit tail.
        char* block;
        shared.inputs.producer_dequeue(block);
        for (size_t i = 0; i < 17; ++i) {
            const auto& key = keys[i % 4];
            std::memcpy(block + i * width, key.data(), width);
            if (i % 2 == 0) ++expected_tree[key];
            else if (keep_singletons) ++expected_low[key];
        }
        require(local.try_enqueue({block, 17}), "pending fixture queue is full");
        const std::vector<char> pending_before(block, block + 17 * width);
        if (keep_singletons) classifier.reconcile_low<true>();
        else classifier.reconcile_low<false>();
        require(std::equal(pending_before.begin(), pending_before.end(), block), "pending pass changed input after memory FIRST");
        require(classifier.low_records_scanned == 17 && classifier.low_records_recovered == 9,
            "memory scan changed pending pass statistics");
        require(classifier.low_tree_batches == 1 && classifier.first_memory_to_tree == hits
            && classifier.recovered_first_count == hits && classifier.total_kmers_send_to_tree == hits + 9
            && classifier.first_memory_tree_batches == hit_segments.size(),
            "pending recovery changed memory FIRST statistics");
        collect_exports(shared, actual_low);
        require(actual_low == (keep_singletons ? expected_low : std::map<Key, uint64_t>{}),
            "pending pass mixed memory FIRSTs into singleton output");
    }
    require(classifier.total_kmers_exported == (keep_singletons ? count - hits + (replay_pending ? 8 : 0) : 0),
        "memory FIRST or following pending pass export statistics differ");
    std::map<Key, uint64_t> actual_tree;
    collect_roots(Access::roots(classifier), actual_tree);
    Task<uint32_t> task;
    while (layers->get_queue(0)->try_dequeue(task)) {
        for (size_t b = 0; b < task.count; ++b)
            for (size_t i = 0; i < task.kmer_blocks[b]->count; ++i) {
                const auto key = read_tree_key(task.kmer_blocks[b]->k_mers.data() + i * tree_record_bytes(task.depth), task.current_node, task.depth);
                require(task.current_node == tree.root_nodes + key[0], "queued memory hit has the wrong root");
                ++actual_tree[key];
            }
    }
    require(actual_tree == expected_tree, "memory scan lost or corrupted Tree records during rollover");
    for (size_t r = 0; r < roots; ++r) {
        auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[keys[r][0]]);
        for (size_t b = 0; b < bloom_before[r].size(); ++b)
            require(bloom->get_bins()[b].load() == bloom_before[r][b], "memory query modified a Bloom layer");
    }
}

void check_memory_first_scan_boundaries(Shared& shared)
{
    for (uint32_t k : {1u, 31u, 33u, 65u, 128u}) {
        const size_t width = (k + 3) / 4;
        const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / width;
        const size_t slice_capacity = 32768 / width;
        const size_t export_capacity = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / width;
        for (bool keep : {false, true}) {
            for (size_t count : {size_t{0}, size_t{1}, size_t{15}, size_t{16}, size_t{17},
                    slice_capacity - 1, slice_capacity, slice_capacity + 1, capacity - 1, capacity, capacity + 1})
                check_memory_first_scan(shared, k, count, FirstScanPattern::Hit, keep);
            for (size_t count : {size_t{15}, size_t{16}, size_t{17}, slice_capacity - 1, slice_capacity, slice_capacity + 1})
                for (auto pattern : {FirstScanPattern::Miss, FirstScanPattern::Alternating})
                    check_memory_first_scan(shared, k, count, pattern, keep);
            check_memory_first_scan(shared, k, export_capacity + 17, FirstScanPattern::Miss, keep);
            check_memory_first_scan(shared, k, export_capacity + 21, FirstScanPattern::LongMissRun, keep, true);
            check_memory_first_scan(shared, k, 2 * capacity + 17, FirstScanPattern::SingleRoot, keep);
            check_memory_first_scan(shared, k, 2 * capacity + 37, FirstScanPattern::Alternating, keep, true);
            check_memory_first_scan(shared, k, 5 * capacity + 37, FirstScanPattern::SplitRuns, keep);
        }
    }
    const size_t segment_capacity = FIRST_SEGMENT_SIZE / 9;
    for (bool keep : {false, true}) {
        for (auto pattern : {FirstScanPattern::Miss, FirstScanPattern::Hit, FirstScanPattern::Alternating})
            check_memory_first_scan(shared, 33, 2 * segment_capacity + 17, pattern, keep, true);
        check_memory_first_scan(shared, 33, 2 * 7281 + 17, FirstScanPattern::AllRoots, keep);
    }
}

void check_memory_first_inline_storage(Shared& shared)
{
    using Access = FastqClassifierTestAccess;
    k_length = 33;
    constexpr size_t bytes = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE;
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    PhaseBarrier barrier(1);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local_a, local_b;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    FastqClassifier<uint32_t> a(0, &shared.inputs, &local_a, &global, &tree, &pool, barrier,
        nullptr, 0, true);
    // This Classifier owns no roots, and must still submit every global hit.
    FastqClassifier<uint32_t> b(1, &shared.inputs, &local_b, &global, &tree, &pool, barrier,
        nullptr, 0, true);
    std::array<Key, 4> keys;
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = make_key(33, 8831 + i);
        keys[i][0] = 17 + 64 * i;
        auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[keys[i][0]]);
        const auto probe = bloom->prepare_insert(keys[i].data());
        bloom->get_bins()[2 * probe.block_idx].store(probe.insert_num);
    }
    std::map<Key, uint64_t> expected;
    const size_t count = bytes / kmer_bytes() + 17;
    for (size_t i = 0; i < count; ++i) {
        Access::push(a, keys[i % 2]);
        ++expected[keys[i % 2]];
    }
    pool.seal_thread_first();
    for (size_t i = 0; i < count; ++i) {
        Access::push(b, keys[2 + i % 2]);
        ++expected[keys[2 + i % 2]];
    }
    a.classify_first_pass();
    b.classify_first_pass();
    pool.freeze_first();
    auto scan_a = std::async(std::launch::async, [&] { a.process_memory_first<false>(); });
    b.process_memory_first<false>();
    scan_a.get();
    require(a.first_memory_tree_batches + b.first_memory_tree_batches == 2,
        "inline storage did not accommodate independent 64KB hit batches");
    // Each Classifier must accumulate its own pending batch, including non-owner roots.
    for (size_t worker = 0; worker < 2; ++worker) {
        for (size_t offset = 0; offset < count;) {
            const size_t take = std::min(count / 3, count - offset);
            char* block;
            shared.inputs.producer_dequeue(block);
            for (size_t i = 0; i < take; ++i) {
                const auto& key = keys[2 * worker + (offset + i) % 2];
                std::memcpy(block + i * kmer_bytes(), key.data(), kmer_bytes());
                ++expected[key];
            }
            if (worker == 0) local_a.enqueue({block, take});
            else local_b.enqueue({block, take});
            offset += take;
        }
    }
    auto low_a = std::async(std::launch::async, [&] { a.reconcile_low<false>(); });
    b.reconcile_low<false>();
    low_a.get();
    require(a.low_tree_batches == 2 && b.low_tree_batches == 2
        && a.low_records_scanned == count && b.low_records_scanned == count
        && a.low_records_recovered == count && b.low_records_recovered == count,
        "parallel pending batches were shared or prematurely flushed");
    require(a.first_memory_to_tree + b.first_memory_to_tree == 2 * count
        && a.first_memory_tree_batches + b.first_memory_tree_batches == 2,
        "parallel pending recovery changed FIRST statistics");
    std::map<Key, uint64_t> actual;
    collect_roots(Access::roots(a), actual);
    collect_roots(Access::roots(b), actual);
    Task<uint32_t> task;
    while (layers->get_queue(0)->try_dequeue(task)) {
        for (size_t b = 0; b < task.count; ++b)
            for (size_t i = 0; i < task.kmer_blocks[b]->count; ++i) {
                const auto key = read_tree_key(task.kmer_blocks[b]->k_mers.data() + i * tree_record_bytes(task.depth), task.current_node, task.depth);
                require(task.current_node == tree.root_nodes + key[0], "parallel recovery task has the wrong root");
                ++actual[key];
            }
    }
    require(actual == expected, "parallel inline hit buffers overlapped or lost non-owner roots");
    std::map<Key, uint64_t> exported;
    collect_exports(shared, exported);
    require(exported.empty(), "inline all-hit scan exported records");
}

void check_direct_first_segments(Shared& shared)
{
    for (uint32_t k : {1u, 31u, 33u, 65u, 128u}) {
        k_length = k;
        ConcurrentMemoryPool pool(64ULL << 20);
            pool.init_arenas();
        auto layers = std::make_unique<LayerQueues<uint32_t>>();
        KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
        prefix_owners.fill(0);
        bloom_filter_capacity.fill(4096);
        PhaseBarrier barrier(1);
        MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local;
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
        FastqClassifier<uint32_t> classifier(0, &shared.inputs, &local, &global, &tree, &pool, barrier, nullptr, 0, true);
        const auto key = make_key(k, 8831);
        const size_t capacity = FIRST_SEGMENT_SIZE / ((k + 3) / 4);
        // Thousands of small input batches must keep writing the same pool-owned
        // buffer. No partially filled shared descriptor may become visible.
        for (size_t offset = 0; offset + 1 < capacity; ++offset) {
            FastqClassifierTestAccess::push(classifier, key);
            if (offset % 37 == 0) require(!pool.first_empty(), "direct fill lost its owned segment");
        }
        require(classifier.first_tail_records() == capacity - 1, "TLS continuation lost records");
        FastqClassifierTestAccess::push(classifier, key);
        require(!pool.first_empty() && classifier.first_tail_records() == 0, "full pool-owned segment was not sealed");
        pool.freeze_first();
        for (size_t i = 0; i < capacity + 17; ++i) FastqClassifierTestAccess::push(classifier, key);
        require(classifier.first_direct_pending == capacity + 17 && classifier.first_tail_records() == 0,
            "rejected FIRST records changed pending ownership");
        classifier.classify_first_pass();
        std::map<Key, uint64_t> pending;
        collect_exports(shared, pending);
        require(pending == std::map<Key, uint64_t>{{key, capacity + 17}}, "direct pending handoff lost records");
        ConcurrentMemoryPool::FirstSegmentView view;
        require(pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Scan, view) == ConcurrentMemoryPool::FirstClaimStatus::Data
            && view.record_count == capacity, "published full segment missing");
        for (size_t i = 0; i < capacity; ++i)
            require(std::memcmp(view.records + i * kmer_bytes(), key.data(), kmer_bytes()) == 0, "full segment record changed");
        require(pool.complete_first(view), "full segment completion failed");
        require(pool.first_empty(), "full extraction retained a segment");
    }
}

void check_soft_pressure_keeps_filling(Shared& shared)
{
    k_length = 33;
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    PhaseBarrier barrier(1);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    FastqClassifier<uint32_t> classifier(0, &shared.inputs, &local, &global, &tree, &pool, barrier, nullptr, 0, true);
    const auto key = make_key(33, 8831);
    FastqClassifierTestAccess::push(classifier, key);
    const size_t high = arena_watermark(pool.segment_capacity(0) * FIRST_SEGMENT_SIZE, 8);
    pool.allocate_large_from_numa(pool.gap_bytes(0) - high + BLOCK_SIZE, 0);
    require(pool.pressure(0) == MemoryPressure::Soft, "soft-pressure fixture did not reach Soft");
    std::vector<Key> input{key};
    FastqClassifierTestAccess::feed(classifier, input, true);
    const bool kept = pool.thread_first_records() == 2 && pool.first_publications.load() == 0;
    // Always finish the fixture before checking: a failed test must not retain
    // an active TLS owner or an ordinary Export block.
    classifier.classify_first_pass();
    std::map<Key, uint64_t> exported;
    collect_exports(shared, exported);
    pool.freeze_first();
    ConcurrentMemoryPool::FirstSegmentView source;
    require(pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Scan, source)
        == ConcurrentMemoryPool::FirstClaimStatus::Data && source.record_count == 2,
        "soft-pressure fixture lost its records");
    pool.complete_first(source);
    require(kept, "Soft pressure sealed a usable Filling segment");
}

void check_sparse_input_keeps_filling(Shared& shared, bool reclaim_while_idle = false)
{
    k_length = 33;
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    PhaseBarrier barrier(1);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    shared.inputs.reset_producers(1);
    std::vector<char*> held;
    char* block = nullptr;
    while (shared.inputs.producer_try_dequeue(block)) held.push_back(block);
    block = held.back();
    held.pop_back();
    auto worker = std::async(std::launch::async, [&] {
        FastqClassifier<uint32_t> classifier(0, &shared.inputs, &local, &global, &tree, &pool, barrier, nullptr, 0, true);
        classifier.classify_first_pass();
        std::map<Key, uint64_t> roots;
        collect_roots(FastqClassifierTestAccess::roots(classifier), roots);
        return roots;
    });
    std::map<Key, uint64_t> expected;
    for (size_t pulse = 0; pulse < 32; ++pulse) {
        for (size_t i = 0; i < 32; ++i) {
            auto key = make_key(33, 100000 + pulse * 32 + i);
            key[0] = 0;
            std::memcpy(block + i * 9, key.data(), 9);
            ++expected[key];
        }
        local.enqueue({block, 32});
        SpinBackoff<> backoff;
        while (!shared.inputs.producer_try_dequeue(block)) backoff.backoff();
        // Deliberately pace real input arrivals; this is normal producer idle,
        // with ample RAM and no outstanding reclaim request.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const uint64_t early_publications = pool.first_publications.load();
    std::map<Key, uint64_t> reclaimed;
    bool published_while_idle = false;
    if (reclaim_while_idle) {
        // The owner is already waiting for input when reclamation starts.
        // It must observe the request without another input block or EOF.
        pool.request_first_drain();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        ConcurrentMemoryPool::FirstSegmentView source;
        while (std::chrono::steady_clock::now() < deadline) {
            if (pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Spill, source)
                == ConcurrentMemoryPool::FirstClaimStatus::Data) {
                for (size_t i = 0; i < source.record_count; ++i) ++reclaimed[read_key(source.records + i * 9)];
                pool.complete_first(source);
                published_while_idle = true;
                break;
            }
            std::this_thread::yield();
        }
    }
    shared.inputs.producer_set_finished();
    auto actual = worker.get();
    for (const auto& [key, count] : reclaimed) actual[key] += count;
    held.push_back(block);
    for (char* page : held) shared.inputs.consumer_enqueue(page);
    collect_exports(shared, actual);
    pool.freeze_first();
    ConcurrentMemoryPool::FirstSegmentView source;
    while (pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Scan, source)
        == ConcurrentMemoryPool::FirstClaimStatus::Data) {
        for (size_t i = 0; i < source.record_count; ++i) ++actual[read_key(source.records + i * 9)];
        pool.complete_first(source);
    }
    require(actual == expected, "sparse input did not conserve exact records");
    std::cout << "Paced FIRST publications: " << pool.first_publications.load()
        << ", valid bytes: " << pool.first_valid_records.load() * kmer_bytes()
        << ", unused bytes: " << pool.first_unused_bytes.load() << '\n';
    require(early_publications == 0 && pool.first_publications.load() == 1,
        "input queue idle repeatedly sealed partially filled segments");
    require(!reclaim_while_idle || published_while_idle,
        "an input-idle owner did not publish after reclamation started");
}

void check_export_wait_reclaim(Shared& shared)
{
    k_length = 33;
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    PhaseBarrier barrier(1);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    std::promise<void> ready, resume;
    auto resume_signal = resume.get_future();
    const auto key = make_key(33, 8831);
    auto worker = std::async(std::launch::async, [&] {
        FastqClassifier<uint32_t> classifier(0, &shared.inputs, &local, &global, &tree, &pool, barrier, nullptr, 0, true);
        FastqClassifierTestAccess::push(classifier, key);
        ready.set_value();
        resume_signal.wait();
        char* block;
        FastqClassifierTestAccess::acquire_export_block(classifier, block);
        shared.exports.consumer_enqueue(block);
        classifier.classify_first_pass();
    });
    ready.get_future().get();
    std::vector<char*> held;
    char* block;
    while (shared.exports.producer_try_dequeue(block)) held.push_back(block);
    resume.set_value();
    const bool blocked = worker.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    const bool kept = pool.first_publications.load() == 0;
    pool.request_first_drain();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    ConcurrentMemoryPool::FirstSegmentView source;
    bool reclaimed = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Spill, source)
            == ConcurrentMemoryPool::FirstClaimStatus::Data) {
            reclaimed = source.record_count == 1 && std::memcmp(source.records, key.data(), 9) == 0;
            pool.complete_first(source);
            break;
        }
        std::this_thread::yield();
    }
    const bool still_blocked = worker.wait_for(std::chrono::seconds(0)) == std::future_status::timeout;
    for (char* page : held) shared.exports.consumer_enqueue(page);
    worker.get();
    std::map<Key, uint64_t> exported;
    collect_exports(shared, exported);
    require(blocked && kept && still_blocked && reclaimed && pool.first_empty(),
        "an export waiter did not retain FIRST until reclaim and publish while still waiting");
}

void check_memory_first_wait_failure(Shared& shared)
{
    Fixture fixture(shared, 33);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> queues{fixture.local};
    ClassifierThreadPool<uint32_t> workers(fixture.tree.get(), &shared.inputs, queues, &fixture.global,
        nullptr, &fixture.pool, 1);
    auto waiting = std::async(std::launch::async, [&] {
        try { workers.wait_for_memory_first(); }
        catch (const std::runtime_error& error) { return std::strcmp(error.what(), "memory-first failure") == 0; }
        return false;
    });
    const bool blocked = waiting.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    fixture.pool.run_state().fail(std::make_exception_ptr(std::runtime_error("memory-first failure")), "test");
    require(waiting.get() && blocked, "memory completion wait ignored the shared error");
}

void check_low_scan(Shared& shared, uint32_t k, size_t count, FirstScanPattern pattern,
    bool keep_singletons, bool small_blocks = false, bool direct = false)
{
    using Access = FastqClassifierTestAccess;
    Fixture fixture(shared, k);
    auto& classifier = *fixture.classifier;
    if (!direct || !keep_singletons) classifier.classify_first_pass();
    std::map<Key, uint64_t> exported;
    collect_exports(shared, exported);
    const size_t width = kmer_bytes();
    const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / width;
    const size_t roots = pattern == FirstScanPattern::AllRoots ? 256 : 4;
    std::vector<Key> keys;
    std::vector<std::vector<uint64_t>> bloom_before;
    for (size_t r = 0; r < roots; ++r) {
        auto key = make_key(k, 8831 + r);
        key[0] = roots == 256 ? r : r * 64;
        keys.push_back(key);
        auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
        auto& snapshot = bloom_before.emplace_back(2 * bloom_filter_capacity[key[0]]);
        for (size_t b = 0; b < snapshot.size(); ++b) {
            snapshot[b] = b % 2 ? (0x5a5a5a5a5a5a5a5aULL ^ b)
                : (roots == 256 || r % 2 == 0 ? UINT64_MAX : 0);
            bloom->get_bins()[b].store(snapshot[b]);
        }
    }
    Access::packed(classifier).fill(0xa5);
    const auto packed_before = Access::packed(classifier);
    std::map<Key, uint64_t> expected_tree, expected_low;
    std::vector<std::pair<char*, std::vector<char>>> input_before;
    size_t hits = 0, hit_count = 0, blocks = 0;
    for (size_t offset = 0; offset < count;) {
        const size_t take = std::min(small_blocks ? capacity / 3 : capacity, count - offset);
        char* block;
        shared.inputs.producer_dequeue(block);
        for (size_t i = 0; i < take; ++i) {
            const size_t index = offset + i;
            const bool hit = pattern == FirstScanPattern::Hit || pattern == FirstScanPattern::SingleRoot
                || pattern == FirstScanPattern::AllRoots
                || (pattern == FirstScanPattern::Alternating && index % 2 == 0)
                || (pattern == FirstScanPattern::LongMissRun && index == EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / width + 3)
                || (pattern == FirstScanPattern::SplitRuns && (index < capacity - 1 || index >= 4 * capacity - 1));
            const auto& key = keys[pattern == FirstScanPattern::SingleRoot ? 0
                : roots == 256 ? index % roots : (index % 3 == 1 ? 2 : 0) + !hit];
            std::memcpy(block + i * width, key.data(), width);
            ++(hit ? expected_tree : expected_low)[key];
            hits += hit;
        }
        input_before.emplace_back(block, std::vector<char>(block, block + take * width));
        if (direct) {
            if (keep_singletons) Access::low_block<true>(classifier, {block, take}, hit_count);
            else Access::low_block(classifier, {block, take}, hit_count);
            require(std::equal(input_before.back().second.begin(), input_before.back().second.end(), block),
                "pending scan changed input records");
            // Returned input storage can immediately be reused by the Parser.
            std::memset(block, 0xcc, take * width);
            require(hit_count == hits % capacity && classifier.low_tree_batches == hits / capacity
                && classifier.low_records_recovered == hits - hit_count,
                "pending scan flushed or discarded a partial batch before EOF");
        } else if (blocks % 2 == 0) {
            fixture.local->enqueue({block, take});
        } else {
            fixture.global.enqueue({block, take});
        }
        ++blocks;
        offset += take;
    }
    std::vector<char*> held;
    if (!keep_singletons) {
        char* block;
        while (shared.exports.producer_try_dequeue(block)) held.push_back(block);
    }
    if (direct) {
        Access::flush_low(classifier, hit_count);
        Access::flush_low(classifier, hit_count);
        require(hit_count == 0, "pending final flush retained records");
        if (keep_singletons) classifier.classify_first_pass();
    } else if (keep_singletons) {
        classifier.reconcile_low<true>();
    } else {
        classifier.reconcile_low<false>();
    }
    if (!direct)
        for (const auto& [block, bytes] : input_before)
            require(std::equal(bytes.begin(), bytes.end(), block), "pending replay changed input records");
    for (auto* block : held) shared.exports.consumer_enqueue(block);
    std::map<Key, uint64_t> actual;
    collect_roots(Access::roots(classifier), actual);
    Task<uint32_t> task;
    while (fixture.layers->get_queue(0)->try_dequeue(task)) {
        for (size_t b = 0; b < task.count; ++b)
            for (size_t i = 0; i < task.kmer_blocks[b]->count; ++i) {
                const auto key = read_tree_key(task.kmer_blocks[b]->k_mers.data() + i * tree_record_bytes(task.depth), task.current_node, task.depth);
                require(task.current_node == fixture.tree->root_nodes + key[0], "pending task has the wrong root");
                ++actual[key];
            }
    }
    require(actual == expected_tree, "pending Tree multiset differs during rollover");
    require(classifier.low_records_scanned == count && classifier.low_records_recovered == hits
        && classifier.total_kmers_send_to_tree == hits, "pending query conservation differs");
    require(classifier.total_kmers_exported == (keep_singletons ? count - hits : 0),
        "pending export statistics differ");
    require(classifier.low_tree_batches == hits / capacity + (hits % capacity != 0),
        "pending scan did not accumulate full 64KB batches across input tasks");
    require(classifier.first_memory_to_tree == 0 && classifier.recovered_first_count == 0
        && classifier.first_memory_tree_batches == 0 && classifier.repeat_query_count == 0,
        "pending recovery changed FIRST statistics");
    collect_exports(shared, exported);
    require(exported == (keep_singletons ? expected_low : std::map<Key, uint64_t>{}),
        "pending singleton multiset differs");
    require(Access::packed(classifier) == packed_before, "pending replay changed the old 32KB ring buffer");
    for (size_t r = 0; r < roots; ++r) {
        auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[keys[r][0]]);
        for (size_t b = 0; b < bloom_before[r].size(); ++b)
            require(bloom->get_bins()[b].load() == bloom_before[r][b], "pending query modified a Bloom layer");
    }
    std::set<char*> returned;
    char* block;
    while (shared.inputs.producer_try_dequeue(block))
        require(returned.insert(block).second, "pending replay returned an input block twice");
    require(returned.size() == shared.inputs.capacity(), "pending replay leaked an input block");
    for (auto* data : returned) shared.inputs.consumer_enqueue(data);
}

void check_low_scan_boundaries(Shared& shared)
{
    for (uint32_t k : {1u, 31u, 33u, 65u, 128u}) {
        const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / ((k + 3) / 4);
        const size_t export_capacity = EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / ((k + 3) / 4);
        for (bool keep : {false, true}) {
            for (size_t count : {size_t{0}, size_t{1}, size_t{15}, size_t{16}, size_t{17},
                    capacity - 1, capacity, capacity + 1, 2 * capacity + 17})
                check_low_scan(shared, k, count, FirstScanPattern::Hit, keep);
            for (size_t count : {size_t{15}, size_t{16}, size_t{17}})
                for (auto pattern : {FirstScanPattern::Miss, FirstScanPattern::Alternating})
                    check_low_scan(shared, k, count, pattern, keep);
            check_low_scan(shared, k, capacity, FirstScanPattern::Hit, keep, true);
            check_low_scan(shared, k, export_capacity + 17, FirstScanPattern::Miss, keep);
            check_low_scan(shared, k, export_capacity + 17, FirstScanPattern::Miss, keep, false, true);
            check_low_scan(shared, k, export_capacity + 21, FirstScanPattern::LongMissRun, keep, false, true);
            check_low_scan(shared, k, 2 * capacity + 17, FirstScanPattern::SingleRoot, keep);
            check_low_scan(shared, k, 2 * capacity + 17, FirstScanPattern::Alternating, keep, true);
        }
        // C-1 hits, three zero-hit blocks, one hit: never flush the tail early.
        check_low_scan(shared, k, 4 * capacity, FirstScanPattern::SplitRuns, false, false, true);
        // A long run crosses both the buffered tail and the next full-batch boundary.
        check_low_scan(shared, k, 5 * capacity + 17, FirstScanPattern::SplitRuns, false, false, true);
    }
    for (bool keep : {false, true})
        check_low_scan(shared, 33, 2 * (65536 / 9) + 17, FirstScanPattern::AllRoots, keep, true);
}

void check_scan_failures(Shared& shared)
{
    using Access = FastqClassifierTestAccess;
    for (bool keep : {false, true}) {
        Fixture fixture(shared, 33);
        auto key = make_key(33, 8831);
        auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
        bloom->insert(key.data());
        bloom->insert(key.data());
        char* block;
        shared.inputs.producer_dequeue(block);
        std::memcpy(block, key.data(), kmer_bytes());
        fixture.pool.run_state().fail(std::make_exception_ptr(std::runtime_error("pending scan failure")), "test");
        size_t hit_count = 0;
        bool failed = false;
        try {
            if (keep) Access::low_block<true>(*fixture.classifier, {block, 1}, hit_count);
            else Access::low_block(*fixture.classifier, {block, 1}, hit_count);
        } catch (const std::runtime_error& error) {
            failed = std::strcmp(error.what(), "pending scan failure") == 0;
            shared.inputs.consumer_enqueue(block);
        }
        require(failed && hit_count == 0 && fixture.classifier->low_records_scanned == 0,
            "pending block ignored an existing failure");
    }

    k_length = 33;
    ConcurrentMemoryPool pool(64ULL << 20);
    pool.init_arenas();
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&pool, layers.get(), &shared.exports);
    prefix_owners.fill(0);
    bloom_filter_capacity.fill(4096);
    PhaseBarrier barrier(1);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    FastqClassifier<uint32_t> classifier(0, &shared.inputs, &local, &global, &tree, &pool, barrier, nullptr, 0, true);
    auto key = make_key(33, 8831);
    auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
    for (size_t i = 0; i < 2 * bloom_filter_capacity[key[0]]; ++i) bloom->get_bins()[i].store(0);
    const size_t count = 2 * (EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes()) + 17;
    for (size_t i = 0; i < count; ++i) Access::push(classifier, key);
    classifier.classify_first_pass();
    std::map<Key, uint64_t> first_exports;
    collect_exports(shared, first_exports);
    pool.freeze_first();
    std::vector<char*> held;
    char* block;
    while (shared.exports.producer_try_dequeue(block)) held.push_back(block);
    require(!held.empty(), "failure fixture has no export transport");
    shared.exports.consumer_enqueue(held.back());
    held.pop_back();
    auto worker = std::async(std::launch::async, [&] {
        try { classifier.process_memory_first<true>(); }
        catch (const std::runtime_error& error) { return std::strcmp(error.what(), "memory scan failure") == 0; }
        return false;
    });
    SpinBackoff<> backoff;
    content_type content;
    while (!shared.exports.consumer_try_dequeue(content)) backoff.backoff();
    // The scan is blocked acquiring another export block. Publish failure before
    // returning that block; no global active state supplies a backoff shortcut.
    pool.run_state().fail(std::make_exception_ptr(std::runtime_error("memory scan failure")), "test");
    shared.exports.consumer_enqueue(content.data);
    while (worker.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
        while (shared.exports.consumer_try_dequeue(content)) shared.exports.consumer_enqueue(content.data);
        backoff.backoff();
    }
    const bool failed = worker.get();
    if (auto* tail = Access::export_block(classifier))
        shared.exports.consumer_enqueue(reinterpret_cast<char*>(const_cast<ExportBlock*>(tail)));
    for (auto* data : held) shared.exports.consumer_enqueue(data);
    std::map<Key, uint64_t> ignored;
    collect_exports(shared, ignored);
    require(failed && classifier.first_memory_kept < count,
        "memory FIRST scan did not stop at a scan-slice failure check");
}

void check_low_queue_pause(Shared& shared)
{
    using Access = FastqClassifierTestAccess;
    RunState state;
    ActiveRunScope active(state);
    Fixture fixture(shared, 33);
    auto& classifier = *fixture.classifier;
    classifier.classify_first_pass();
    std::map<Key, uint64_t> exported;
    collect_exports(shared, exported);
    const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
    auto key = make_key(33, 8831);
    auto* bloom = static_cast<ConcurrentBloomFilter*>(global_bloom_filter[key[0]]);
    bloom->insert(key.data());
    bloom->insert(key.data());
    char* first;
    shared.inputs.producer_dequeue(first);
    for (size_t i = 0; i < capacity - 2; ++i)
        std::memcpy(first + i * kmer_bytes(), key.data(), kmer_bytes());
    fixture.local->enqueue({first, capacity - 2});
    std::vector<char*> held;
    char* block;
    while (shared.inputs.producer_try_dequeue(block)) held.push_back(block);
    shared.inputs.reset_producers(1);
    auto consumer = std::async(std::launch::async, [&] { classifier.reconcile_low<false>(); });
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!shared.inputs.producer_try_dequeue(block)) {
        if (std::chrono::steady_clock::now() >= deadline) {
            state.fail(std::make_exception_ptr(std::runtime_error("pending queue pause timed out")), "test");
            consumer.get();
        }
        std::this_thread::yield();
    }
    require(block == first, "normal pending task returned an unexpected block");
    // The empty-queue interval must retain the C-2 hit tail.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    char* last_owner = held.back();
    held.pop_back();
    std::memcpy(last_owner, key.data(), kmer_bytes());
    std::memcpy(block, key.data(), kmer_bytes());
    fixture.local->enqueue({last_owner, 1});
    fixture.global.enqueue({block, 1});
    shared.inputs.producer_set_finished();
    consumer.get();
    for (auto* data : held) shared.inputs.consumer_enqueue(data);
    require(classifier.low_records_scanned == capacity && classifier.low_records_recovered == capacity
        && classifier.low_tree_batches == 1, "queue pause or EOF drain prematurely flushed a hit tail");
    std::map<Key, uint64_t> actual;
    collect_roots(Access::roots(classifier), actual);
    Task<uint32_t> task;
    while (fixture.layers->get_queue(0)->try_dequeue(task))
        for (size_t b = 0; b < task.count; ++b)
            for (size_t i = 0; i < task.kmer_blocks[b]->count; ++i)
                ++actual[read_tree_key(task.kmer_blocks[b]->k_mers.data() + i * tree_record_bytes(task.depth), task.current_node, task.depth)];
    require(actual == std::map<Key, uint64_t>{{key, capacity}}, "normal and EOF queues lost pending hits");
}
}

int main(int argc, char* argv[])
{
    try {
        if (argc == 2 && (std::strcmp(argv[1], "--first-soft") == 0 || std::strcmp(argv[1], "--first-idle") == 0
            || std::strcmp(argv[1], "--first-idle-reclaim") == 0 || std::strcmp(argv[1], "--first-export-wait") == 0)) {
            Shared shared;
            if (std::strcmp(argv[1], "--first-soft") == 0) check_soft_pressure_keeps_filling(shared);
            else if (std::strcmp(argv[1], "--first-export-wait") == 0) check_export_wait_reclaim(shared);
            else check_sparse_input_keeps_filling(shared, std::strcmp(argv[1], "--first-idle-reclaim") == 0);
            return 0;
        }
        check_parser_transfers();
        if (argc == 2 && std::strcmp(argv[1], "--parser-transfer") == 0) {
            std::cout << "Parser transfer tests passed\n";
            return 0;
        }
        check_parser_returns_empty_blocks();
        Shared shared;
        for (uint32_t k : {1u, 31u, 33u, 65u, 128u}) check_export_batches(shared, k);
        check_scan_failures(shared);
        if (argc == 2 && std::strcmp(argv[1], "--low-full-block") == 0) {
            check_low_scan(shared, 33, 65536 / 9, FirstScanPattern::Hit, false);
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--low-small-blocks") == 0) {
            check_low_scan(shared, 33, 65536 / 9, FirstScanPattern::Hit, false, true);
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--memory-first") == 0) {
            check_memory_first_scan_boundaries(shared);
            check_memory_first_inline_storage(shared);
            check_memory_first_batches(shared, true);
            check_memory_first_batches(shared, false);
            return 0;
        }
        check_low_scan_boundaries(shared);
        check_low_queue_pause(shared);
        if (argc == 2 && std::strcmp(argv[1], "--low-replay") == 0) return 0;
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
        check_direct_first_segments(shared);
        check_soft_pressure_keeps_filling(shared);
        check_sparse_input_keeps_filling(shared);
        check_sparse_input_keeps_filling(shared, true);
        check_export_wait_reclaim(shared);
        check_dynamic_safe_points(shared, true);
        check_dynamic_safe_points(shared, false);
        check_memory_first_batches(shared, true);
        check_memory_first_batches(shared, false);
        check_memory_first_scan_boundaries(shared);
        check_memory_first_inline_storage(shared);
        check_memory_first_wait_failure(shared);
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
