#ifndef CLASSIFIER_THREAD_POOL_HEADER
#define CLASSIFIER_THREAD_POOL_HEADER

#include "definition.h"
#include "MPMCRingQueue.h"
#include "NewKmerTree.h"
#include "ConcurrentMemoryPool.h"
#include "RingMemoryPool.h"
#include "FastqClassifier.h"
#include "SchedulerThreadPool.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>
#include <utility>
#include <chrono>
#include <barrier>
#include <iostream>
#include <limits>
#include <latch>
#include <semaphore>

class ClassifierThreadPool
{

    const uint32_t classifier_count;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_classifier_ring_pool;
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> classifier_task_queues;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    SchedulerThreadPool* task_thread_pool;
    ConcurrentMemoryPool* memory_pool;
    KmerTree* tree;
    std::vector<std::unique_ptr<std::thread>> threads_ptr;
    std::shared_ptr<PhaseBarrier> start_barrier;
    std::shared_ptr<PhaseBarrier> finish_barrier;
    std::array<uint64_t, 1ULL << (2 * ROOT_BASES)> deal_kmer_counts{};
    std::array<uint64_t, 1ULL << (2 * ROOT_BASES)> low_scanned_counts{};
    std::array<uint64_t, 1ULL << (2 * ROOT_BASES)> low_recovered_counts{};
    std::array<uint64_t, 1ULL << (2 * ROOT_BASES)> first_drain_singleton_counts{};
    std::atomic<uint32_t> first_pass_done{0};
    std::atomic<uint32_t> memory_first_done{0};
    std::atomic<bool> low_pass_start{false};
    const std::array<void*, 256>* bloom_storage_;
    unsigned char* first_active_storage_;

public:
    uint64_t low_records_scanned = 0;
    uint64_t low_records_recovered = 0;
    uint64_t first_drain_singletons = 0;
#ifdef TEST_MODE
    std::atomic<uint64_t> producer_enqueue_spin_time{ 0 };
    std::atomic<uint64_t> producer_dequeue_spin_time{ 0 };
    std::atomic<uint64_t> consumer_enqueue_spin_time{ 0 };
    std::atomic<uint64_t> consumer_dequeue_spin_time{ 0 };
    std::atomic<uint64_t> total_kmers_exported{ 0 };
    std::atomic<uint64_t> total_kmers_send_to_tree{ 0 };
    std::atomic<uint64_t> total_classifier_wait_cycles{ 0 };
    std::atomic<uint64_t> total_local_tasks{ 0 };
    std::atomic<uint64_t> total_global_tasks{ 0 };
    std::atomic<uint64_t> total_owner_tasks{ 0 };
    std::atomic<uint64_t> total_first_occurrences{ 0 };
    std::atomic<uint64_t> total_direct_pending{0};
    std::atomic<uint64_t> total_first_memory_to_tree{0}, total_first_memory_discarded{0}, total_first_memory_kept{0};
    std::atomic<uint64_t> total_first_memory_tree_batches{0};
    std::atomic<uint64_t> input_first_records{0}, input_private_tail_records{0}, input_pending_records{0};
    std::atomic<uint64_t> total_repeat_queries{ 0 };
    std::atomic<uint64_t> total_recovered_firsts{ 0 };
#endif

    explicit ClassifierThreadPool(KmerTree* tree_ptr,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_parser_classifier_ring_pool_ptr,
        std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>>& in_classifier_task_queues,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* in_global_classifier_task_queue,
        SchedulerThreadPool* task_thread_pool_ptr,
        ConcurrentMemoryPool* in_memory_pool_ptr,
        uint32_t in_classifier_count,
        const std::array<void*, 256>* bloom_storage = nullptr, unsigned char* first_active_storage = nullptr)
        :         tree(tree_ptr),
        parser_classifier_ring_pool(in_parser_classifier_ring_pool_ptr),
        classifier_task_queues(in_classifier_task_queues),
        global_classifier_task_queue(in_global_classifier_task_queue),
        classifier_count(in_classifier_count),
        task_thread_pool(task_thread_pool_ptr),
        memory_pool(in_memory_pool_ptr), bloom_storage_(bloom_storage), first_active_storage_(first_active_storage)
    {
        threads_ptr.reserve(classifier_count);
    }

    ~ClassifierThreadPool()
    {
        RunState::fail_on_unwind();
        for (auto& thread : threads_ptr) if (thread->joinable()) thread->join();
    }

    void start()
    {
        start_barrier = std::make_shared<PhaseBarrier>(classifier_count, &memory_pool->run_state());
        finish_barrier = std::make_shared<PhaseBarrier>(classifier_count, &memory_pool->run_state());
        for (uint32_t i = 0; i < classifier_count; ++i) {
            threads_ptr.push_back(std::make_unique<std::thread>([this, i] {
                run_worker("classifier", [&] {
                    const int arena = i % memory_pool->arena_count();
#if HAS_LIBNUMA
                    numa_run_on_node(memory_pool->arena_node(arena));
#endif
                    FastqClassifier classifier(i, parser_classifier_ring_pool, classifier_task_queues[i].get(),
                        global_classifier_task_queue, tree, memory_pool, *start_barrier, nullptr, 0,
                        first_active_storage_ ? first_active_storage_ + i * FIRST_SEGMENT_SIZE : nullptr,
                        bloom_storage_);
                    classifier.classify_first_pass();
#ifdef TEST_MODE
                    input_first_records.fetch_add(classifier.first_occurrence_count, std::memory_order_relaxed);
                    input_private_tail_records.fetch_add(classifier.first_tail_records(), std::memory_order_relaxed);
                    input_pending_records.fetch_add(classifier.first_direct_pending, std::memory_order_relaxed);
#endif
                    first_pass_done.fetch_add(1, std::memory_order_release);
                    memory_pool->run_state().notify();
                    while (!low_pass_start.load(std::memory_order_acquire)) {
                        const auto progress = memory_pool->run_state().progress();
                        if (!low_pass_start.load(std::memory_order_acquire)) memory_pool->run_state().wait(progress);
                    }
                    memory_pool->run_state().check();
                    const bool write_low = filter_min <= 1 && filter_max >= 1;
                    if (write_low) classifier.template process_memory_first<true>();
                    else classifier.template process_memory_first<false>();
                    memory_first_done.fetch_add(1, std::memory_order_release);
                    memory_pool->run_state().notify();
                    if (write_low) classifier.template reconcile_low<true>();
                    else classifier.template reconcile_low<false>();
                    classifier.finish_tree_input();
                    task_thread_pool->mark_producer_done();
                    finish_barrier->arrive_and_wait();
                    deal_kmer_counts[i] = classifier.total_deal_kmer_count;
                    // Each Classifier owns one result slot; aggregate after join.
                    low_scanned_counts[i] = classifier.low_records_scanned;
                    low_recovered_counts[i] = classifier.low_records_recovered;
                    first_drain_singleton_counts[i] = classifier.first_drain_singletons;
#ifdef TEST_MODE
                    consumer_enqueue_spin_time.fetch_add(classifier.consumer_enqueue_spin_time, std::memory_order_relaxed);
                    consumer_dequeue_spin_time.fetch_add(classifier.consumer_dequeue_spin_time, std::memory_order_relaxed);
                    producer_enqueue_spin_time.fetch_add(classifier.producer_enqueue_spin_time, std::memory_order_relaxed);
                    producer_dequeue_spin_time.fetch_add(classifier.producer_dequeue_spin_time, std::memory_order_relaxed);
                    total_kmers_exported.fetch_add(classifier.total_kmers_exported, std::memory_order_relaxed);
                    total_kmers_send_to_tree.fetch_add(classifier.total_kmers_send_to_tree, std::memory_order_relaxed);
                    total_classifier_wait_cycles.fetch_add(classifier.classifier_wait_cycles, std::memory_order_relaxed);
                    total_local_tasks.fetch_add(classifier.local_tasks, std::memory_order_relaxed);
                    total_global_tasks.fetch_add(classifier.global_tasks, std::memory_order_relaxed);
                    total_owner_tasks.fetch_add(classifier.owner_tasks, std::memory_order_relaxed);
                    total_direct_pending.fetch_add(classifier.first_direct_pending, std::memory_order_relaxed);
                    total_first_memory_to_tree.fetch_add(classifier.first_memory_to_tree, std::memory_order_relaxed);
                    total_first_memory_tree_batches.fetch_add(classifier.first_memory_tree_batches, std::memory_order_relaxed);
                    total_first_memory_discarded.fetch_add(classifier.first_memory_discarded, std::memory_order_relaxed);
                    total_first_memory_kept.fetch_add(classifier.first_memory_kept, std::memory_order_relaxed);
                    total_first_occurrences.fetch_add(classifier.first_occurrence_count, std::memory_order_relaxed);
                    total_repeat_queries.fetch_add(classifier.repeat_query_count, std::memory_order_relaxed);
                    total_recovered_firsts.fetch_add(classifier.recovered_first_count, std::memory_order_relaxed);
#endif
                });
            }));
        }
    }

    void wait_for_first_pass()
    {
        while (first_pass_done.load(std::memory_order_acquire) != classifier_count) {
            const auto progress = memory_pool->run_state().progress();
            if (first_pass_done.load(std::memory_order_acquire) != classifier_count) memory_pool->run_state().wait(progress);
        }
        memory_pool->run_state().check();
    }

    void start_low_pass()
    {
        low_pass_start.store(true, std::memory_order_release);
        memory_pool->run_state().notify();
    }

    void wait_for_memory_first()
    {
        while (memory_first_done.load(std::memory_order_acquire) != classifier_count) {
            const auto progress = memory_pool->run_state().progress();
            if (memory_first_done.load(std::memory_order_acquire) != classifier_count) memory_pool->run_state().wait(progress);
        }
        memory_pool->run_state().check();
    }

    void join()
    {
        for (auto& t : threads_ptr)
        {
            if (t->joinable())
            {
                t->join();
            }
        }
        memory_pool->run_state().check();
        for (uint32_t i = 0; i < classifier_count; ++i) {
            low_records_scanned += low_scanned_counts[i];
            low_records_recovered += low_recovered_counts[i];
            first_drain_singletons += first_drain_singleton_counts[i];
        }
#ifdef TEST_MODE
        for (uint32_t i = 0; i < classifier_count; i++)
        {
            std::cout << "Classifier " << i << " dealt k-mer count: " << deal_kmer_counts[i] << "\n";
        }
        std::cout << "Classifier FIRST occurrences: " << total_first_occurrences.load() << '\n';
        std::cout << "Classifier repeat queries: " << total_repeat_queries.load() << '\n';
        std::cout << "Classifier recovered FIRSTs: " << total_recovered_firsts.load() << '\n';
        std::cout << "Classifier FIRST memory to Tree: " << total_first_memory_to_tree.load() << '\n';
        std::cout << "Classifier FIRST memory Tree batches: " << total_first_memory_tree_batches.load() << '\n';
        std::cout << "Classifier FIRST memory discarded: " << total_first_memory_discarded.load() << '\n';
        std::cout << "Classifier FIRST memory kept: " << total_first_memory_kept.load() << '\n';
        std::cout << "Classifier FIRST direct pending: " << total_direct_pending.load() << '\n';
        std::cout << "Classifier FIRST ownership conservation: "
            << (total_first_occurrences.load() == total_direct_pending.load()
                + total_first_memory_to_tree.load() + total_first_memory_discarded.load() + total_first_memory_kept.load()
                + memory_pool->spill_bytes.load() / kmer_bytes()) << '\n';
#endif
    }
};

#endif
