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

template<typename CountType = uint32_t>
class ClassifierThreadPool
{

    const uint32_t classifier_count;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_classifier_ring_pool;
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> classifier_task_queues;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    SchedulerThreadPool<CountType>* task_thread_pool;
    ConcurrentMemoryPool* memory_pool;
    KmerTree<CountType>* tree;
    std::vector<std::unique_ptr<std::thread>> threads_ptr;
    std::shared_ptr<PhaseBarrier> start_barrier;
    std::shared_ptr<PhaseBarrier> finish_barrier;
    std::array<uint64_t, 1ULL << (2 * ROOT_BASES)> deal_kmer_counts{};
    std::atomic<uint32_t> first_pass_done{0};

public:
#ifdef TEST_MODE
    std::atomic<uint64_t> consumer_enqueue_spin_time{ 0 };
    std::atomic<uint64_t> consumer_dequeue_spin_time{ 0 };
    std::atomic<uint64_t> total_kmers_send_to_tree{ 0 };
    std::atomic<uint64_t> total_classifier_wait_cycles{ 0 };
    std::atomic<uint64_t> total_local_tasks{ 0 };
    std::atomic<uint64_t> total_global_tasks{ 0 };
    std::atomic<uint64_t> total_owner_tasks{ 0 };
#endif

    explicit ClassifierThreadPool(KmerTree<CountType>* tree_ptr,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_parser_classifier_ring_pool_ptr,
        std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>>& in_classifier_task_queues,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* in_global_classifier_task_queue,
        SchedulerThreadPool<CountType>* task_thread_pool_ptr,
        ConcurrentMemoryPool* in_memory_pool_ptr,
        uint32_t in_classifier_count)
        : classifier_count(in_classifier_count),
        parser_classifier_ring_pool(in_parser_classifier_ring_pool_ptr),
        classifier_task_queues(in_classifier_task_queues),
        global_classifier_task_queue(in_global_classifier_task_queue),
        task_thread_pool(task_thread_pool_ptr),
        memory_pool(in_memory_pool_ptr), tree(tree_ptr)
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
#if HAS_LIBNUMA
                    const int arena = i % memory_pool->arena_count();
                    numa_run_on_node(memory_pool->arena_node(arena));
#endif
                    FastqClassifier<CountType> classifier(i, parser_classifier_ring_pool, classifier_task_queues[i].get(),
                        global_classifier_task_queue, tree, memory_pool, *start_barrier);
                    classifier.classify_first_pass();
                    first_pass_done.fetch_add(1, std::memory_order_release);
                    memory_pool->run_state().notify();
                    classifier.finish_tree_input();
                    task_thread_pool->mark_producer_done();
                    finish_barrier->arrive_and_wait();
                    deal_kmer_counts[i] = classifier.total_deal_kmer_count;
#ifdef TEST_MODE
                    consumer_enqueue_spin_time.fetch_add(classifier.consumer_enqueue_spin_time, std::memory_order_relaxed);
                    consumer_dequeue_spin_time.fetch_add(classifier.consumer_dequeue_spin_time, std::memory_order_relaxed);
                    total_kmers_send_to_tree.fetch_add(classifier.total_kmers_send_to_tree, std::memory_order_relaxed);
                    total_classifier_wait_cycles.fetch_add(classifier.classifier_wait_cycles, std::memory_order_relaxed);
                    total_local_tasks.fetch_add(classifier.local_tasks, std::memory_order_relaxed);
                    total_global_tasks.fetch_add(classifier.global_tasks, std::memory_order_relaxed);
                    total_owner_tasks.fetch_add(classifier.owner_tasks, std::memory_order_relaxed);
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
#ifdef TEST_MODE
        for (uint32_t i = 0; i < classifier_count; i++)
        {
            std::cout << "Classifier " << i << " dealt k-mer count: " << deal_kmer_counts[i] << "\n";
        }
#endif
    }
};

#endif
