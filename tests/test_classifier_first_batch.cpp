#include "high_prefix_test_helpers.h"
#include "../src/FastqClassifier.h"
#include "../src/FastqParser.h"
#include "../src/ParserThreadPool.h"
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
    static void feed(FastqClassifier<uint32_t>& classifier, const std::vector<Key>& keys, bool owned)
    {
        const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / kmer_bytes();
        const auto bytes = flatten(keys);
        for (size_t pos = 0; pos < keys.size(); pos += capacity) {
            const auto count = std::min(capacity, keys.size() - pos);
            if (owned) classifier.process_owned_block(bytes.data() + pos * kmer_bytes(), count);
            else classifier.process_other_block(bytes.data() + pos * kmer_bytes(), count);
        }
    }
    static auto* roots(FastqClassifier<uint32_t>& classifier) { return classifier.local_root_nodes.data(); }
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
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY> exports{1, 1};
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
        for (size_t prefix = 0; prefix < prefix_owners.size(); ++prefix)
            prefix_owners[prefix] = (prefix / 64 + prefix % 2) % 2;
        tree = std::make_unique<KmerTree<uint32_t>>( &pool, layers.get(), &shared.exports);
        classifier = std::make_unique<FastqClassifier<uint32_t>>(0, &shared.inputs, local.get(), &global,
            tree.get(), &pool, start);
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

void collect_tasks(LayerQueues<uint32_t>& layers, std::map<Key, uint64_t>& counts)
{
    Task<uint32_t> task;
    while (layers.get_queue(0)->try_dequeue(task)) {
        require(task.depth == 0, "Classifier published a non-root task");
        for (size_t b = 0; b < task.count; ++b) {
            const auto* block = task.kmer_blocks[b];
            for (size_t i = 0; i < block->count; ++i)
                ++counts[read_tree_key(block->k_mers.data() + i * tree_record_bytes(), task.current_node)];
        }
    }
}

void check_all_occurrences(Shared& shared, uint32_t k, bool owned, size_t count)
{
    Fixture fixture(shared, k);
    auto& classifier = *fixture.classifier;
    using Access = FastqClassifierTestAccess;
    std::vector<Key> keys;
    std::map<Key, uint64_t> expected;
    for (uint64_t seed = 1; keys.size() < count; ++seed) {
        auto key = make_key(k, seed);
        if (owned && prefix_owners[key[0]] != 0) continue;
        ++expected[key];
        keys.push_back(std::move(key));
    }
    Access::feed(classifier, keys, owned);
    // These keys reappear only after the full batch has been submitted.
    std::vector<Key> delayed(keys.begin(), keys.begin() + std::min(size_t{17}, keys.size()));
    for (const auto& key : delayed) ++expected[key];
    Access::feed(classifier, delayed, owned);
    const auto total = keys.size() + delayed.size();
    require(classifier.total_deal_kmer_count == total && classifier.total_kmers_send_to_tree == total
        && static_cast<uint64_t>(fixture.tree->total_kmers_added.load()) == total,
        "first pass did not submit every occurrence exactly once");
    classifier.classify_first_pass();
    classifier.finish_tree_input();
    std::map<Key, uint64_t> actual;
    collect_roots(fixture.tree->root_nodes, actual);
    collect_tasks(*fixture.layers, actual);
    require(actual == expected, "first pass or local root flush changed the complete occurrence multiset");
    content_type content;
    require(!shared.exports.consumer_try_dequeue(content), "Classifier submitted an ExportBlock");
}

void check_owner_and_global_queues(Shared& shared, bool before_eof)
{
    k_length = 33;
    using Access = FastqClassifierTestAccess;
    for (size_t i = 0; i < prefix_owners.size(); ++i) prefix_owners[i] = i % 2;
    shared.inputs.reset_producers(1);
    auto layers = std::make_unique<LayerQueues<uint32_t>>();
    KmerTree<uint32_t> tree(&shared.pool, layers.get(), &shared.exports);
    MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY> local_a, local_b;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY> global;
    PhaseBarrier start(2);
    auto constructing_a = std::async(std::launch::async, [&] {
        return std::make_unique<FastqClassifier<uint32_t>>(0, &shared.inputs, &local_a, &global,
            &tree, &shared.pool, start);
    });
    FastqClassifier<uint32_t> b(1, &shared.inputs, &local_b, &global, &tree, &shared.pool, start);
    auto a = constructing_a.get();
    auto even = make_key(k_length, 700), odd = make_key(k_length, 701);
    even[0] = 16;
    odd[0] = 17;
    char* block;
    shared.inputs.producer_dequeue(block);
    std::memcpy(block, even.data(), kmer_bytes());
    local_a.enqueue({block, 1});
    shared.inputs.producer_dequeue(block);
    std::memcpy(block, odd.data(), kmer_bytes());
    local_b.enqueue({block, 1});
    // Non-owner global work and a delayed duplicate from the owner queue.
    shared.inputs.producer_dequeue(block);
    std::memcpy(block, even.data(), kmer_bytes());
    std::memcpy(block + kmer_bytes(), even.data(), kmer_bytes());
    global.enqueue({block, 2});
    if (!before_eof) shared.inputs.producer_set_finished();
    auto finishing_b = std::async(std::launch::async, [&] { b.classify_first_pass(); });
    if (before_eof) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (tree.total_kmers_added.load() != 3 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        const bool consumed = tree.total_kmers_added.load() == 3;
        shared.inputs.producer_set_finished();
        finishing_b.get();
        require(consumed, "Classifier did not consume owner and foreign global tasks before EOF");
        require(b.local_tasks == 1 && b.global_tasks == 1 && b.owner_tasks == 1,
            "live task paths were not exercised");
    } else {
        require(finishing_b.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
            "first pass waited for another Classifier after EOF");
        finishing_b.get();
    }
    a->classify_first_pass();
    std::map<Key, uint64_t> a_counts, b_counts;
    collect_roots(Access::roots(*a), a_counts);
    collect_roots(Access::roots(b), b_counts);
    require(a_counts == std::map<Key, uint64_t>{{even, 1}}, "owner queue was not consumed exactly once");
    require(b_counts == std::map<Key, uint64_t>{{even, 2}, {odd, 1}},
        "non-owner/global queue was not consumed exactly once");
    a->finish_tree_input();
    b.finish_tree_input();
    std::map<Key, uint64_t> actual;
    collect_roots(tree.root_nodes, actual);
    collect_tasks(*layers, actual);
    require(actual == std::map<Key, uint64_t>{{even, 3}, {odd, 1}},
        "cross-Classifier local root flush lost occurrences");
    content_type content;
    require(!local_a.try_dequeue(content) && !local_b.try_dequeue(content) && !global.try_dequeue(content),
        "first pass left queued tasks after EOF");
    std::set<char*> returned;
    while (shared.inputs.producer_try_dequeue(block))
        require(returned.insert(block).second, "Classifier returned an input block twice");
    require(returned.size() == shared.inputs.capacity(), "Classifier leaked an input block");
    for (auto* data : returned) shared.inputs.consumer_enqueue(data);
}

void check_wait_failure(Shared& shared, bool returning_block, bool after_eof, bool global_task)
{
    Fixture fixture(shared, 33);
    auto& state = fixture.pool.run_state();
    ActiveRunScope active(state);
    shared.inputs.reset_producers(after_eof ? 0 : 1);
    std::vector<char> record(kmer_bytes());
    if (returning_block) {
        // The return queue is deliberately full, so returning this injected
        // task blocks even though its Tree append has completed.
        content_type content{record.data(), 1};
        if (global_task) fixture.global.enqueue(content);
        else fixture.local->enqueue(content);
    }
    std::promise<void> started;
    auto running = std::async(std::launch::async, [&] {
        started.set_value();
        try { fixture.classifier->classify_first_pass(); }
        catch (const std::runtime_error& error) { return std::strcmp(error.what(), "waiting failure") == 0; }
        return false;
    });
    started.get_future().wait();
    if (returning_block) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (!fixture.tree->total_kmers_added.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
    }
    const bool waiting = running.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    state.fail(std::make_exception_ptr(std::runtime_error("waiting failure")), "test");
    require(running.wait_for(std::chrono::seconds(2)) == std::future_status::ready,
        "Classifier did not exit a wait after RunState failure");
    require(running.get() && waiting, "Classifier did not propagate the failure from a blocked wait");
    require(!returning_block || fixture.tree->total_kmers_added.load() == 1,
        "failure fixture did not reach input block return");
    shared.inputs.reset_producers(0);
}

void check_first_pass_wait_failure(Shared& shared)
{
    Fixture fixture(shared, 33);
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> queues{fixture.local};
    ClassifierThreadPool<uint32_t> workers(fixture.tree.get(), &shared.inputs, queues, &fixture.global,
        nullptr, &fixture.pool, 1);
    auto waiting = std::async(std::launch::async, [&] {
        try { workers.wait_for_first_pass(); }
        catch (const std::runtime_error& error) { return std::strcmp(error.what(), "first-pass failure") == 0; }
        return false;
    });
    fixture.pool.run_state().fail(std::make_exception_ptr(std::runtime_error("first-pass failure")), "test");
    require(waiting.wait_for(std::chrono::seconds(2)) == std::future_status::ready && waiting.get(),
        "first-pass completion wait ignored failure");
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

    std::map<Key, uint32_t> actual;
    content_type content;
    while (output.consumer_try_dequeue(content)) {
        for (const auto& record : high_prefix_test::decode(
                {reinterpret_cast<const uint8_t*>(content.data), content.length}, kmer_bytes(), count_max_bytes)) {
            require(actual.emplace(record.key, record.count).second, "final drain duplicated a key");
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

} // namespace

int main(int argc, char* argv[])
{
    try {
        check_parser_transfers();
        if (argc == 2 && std::strcmp(argv[1], "--parser-transfer") == 0) {
            std::cout << "Parser transfer tests passed\n";
            return 0;
        }
        check_parser_returns_empty_blocks();
        Shared shared;
        for (uint32_t k : {1u, 31u, 33u, 65u, 81u, 128u}) {
            const size_t capacity = PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE / ((k + 3) / 4);
            for (bool owned : {false, true}) {
                for (size_t count : {size_t{0}, size_t{1}, capacity - 1, capacity, capacity + 1})
                    std::async(std::launch::async, [&] { check_all_occurrences(shared, k, owned, count); }).get();
            }
        }
        std::async(std::launch::async, [&] { check_root_block_rollover(shared); }).get();
        for (bool before_eof : {false, true})
            std::async(std::launch::async, [&] { check_owner_and_global_queues(shared, before_eof); }).get();
        check_wait_failure(shared, false, false, false);
        for (bool after_eof : {false, true}) {
            for (bool global_task : {false, true})
                std::async(std::launch::async, [&] { check_wait_failure(shared, true, after_eof, global_task); }).get();
        }
        std::async(std::launch::async, [&] { check_first_pass_wait_failure(shared); }).get();
        for (uint32_t k : {33u, 65u, 81u, 128u})
            std::async(std::launch::async, [&] { check_final_drain_runs(shared, k); }).get();
        for (bool hashed : {false, true}) {
            for (auto [minimum, maximum] : {std::pair{2u, 3u}, {2u, 2u}, {1u, 1u},
                    {4u, 4u}, {0u, 0u}, {5u, 10u}, {0u, UINT32_MAX}})
                std::async(std::launch::async, [&] {
                    check_final_drain_filter(shared, hashed, minimum, maximum);
                }).get();
        }
        std::cout << "ClassifierFirstBatch tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
