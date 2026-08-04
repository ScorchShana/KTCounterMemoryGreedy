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
#include <bitset>
#include <barrier>
#include <iostream>

template <uint32_t N>
class ClassifierThreadPool
{

    int k_len;
    const uint32_t classifier_count;
    RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* parser_classifier_ring_pool;
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> classifier_task_queues;
    MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* global_classifier_task_queue;
    SchedulerThreadPool<N>* task_thread_pool;
    ConcurrentMemoryPool* memory_pool;
    KmerTree<N>* tree;
    std::vector<std::unique_ptr<std::thread>> threads_ptr;
    std::shared_ptr<std::barrier<>> start_barrier;
    std::shared_ptr<std::barrier<>> finish_barrier;
    std::array<uint64_t, 1ULL << (2 * ROOT_BASES)> deal_kmer_counts{};

public:
#ifdef TEST_MODE
    std::atomic<uint64_t> producer_enqueue_spin_time{ 0 };
    std::atomic<uint64_t> producer_dequeue_spin_time{ 0 };
    std::atomic<uint64_t> consumer_enqueue_spin_time{ 0 };
    std::atomic<uint64_t> consumer_dequeue_spin_time{ 0 };
    std::atomic<uint64_t> total_kmers_exported{ 0 };
    std::atomic<uint64_t> total_kmers_send_to_tree{ 0 };
#endif

    explicit ClassifierThreadPool(const int in_k, KmerTree<N>* tree_ptr,
        RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>* in_parser_classifier_ring_pool_ptr,
        std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>>& in_classifier_task_queues,
        MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>* in_global_classifier_task_queue,
        SchedulerThreadPool<N>* task_thread_pool_ptr,
        ConcurrentMemoryPool* in_memory_pool_ptr,
        uint32_t in_classifier_count)
        : k_len(in_k),
        tree(tree_ptr),
        parser_classifier_ring_pool(in_parser_classifier_ring_pool_ptr),
        classifier_task_queues(in_classifier_task_queues),
        global_classifier_task_queue(in_global_classifier_task_queue),
        classifier_count(in_classifier_count),
        task_thread_pool(task_thread_pool_ptr),
        memory_pool(in_memory_pool_ptr)
    {
        threads_ptr.reserve(classifier_count);
    }

    void start()
    {
        start_barrier = std::make_shared<std::barrier<>>(classifier_count);
        finish_barrier = std::make_shared<std::barrier<>>(classifier_count);

        for (uint32_t i = 0; i < classifier_count; i++)
        {
            threads_ptr.push_back(std::make_unique<std::thread>([&, i]
                {
#if HAS_LIBNUMA
                    numa_run_on_node(numa_nodes[i % numa_nodes.size()]);
#endif
                    std::this_thread::sleep_for(std::chrono::nanoseconds(40));
                    FastqClassifier<N> classifier(k_len, i, parser_classifier_ring_pool, classifier_task_queues[i].get(), global_classifier_task_queue, tree, memory_pool, *start_barrier);
                    classifier.classify_and_push();
                    task_thread_pool->mark_producer_done();
                    finish_barrier->arrive_and_wait();
                    deal_kmer_counts[i] = classifier.total_deal_kmer_count;
#ifdef TEST_MODE
                    consumer_enqueue_spin_time.fetch_add(classifier.consumer_enqueue_spin_time, std::memory_order_relaxed);
                    consumer_dequeue_spin_time.fetch_add(classifier.consumer_dequeue_spin_time, std::memory_order_relaxed);
                    producer_enqueue_spin_time.fetch_add(classifier.producer_enqueue_spin_time, std::memory_order_relaxed);
                    producer_dequeue_spin_time.fetch_add(classifier.producer_dequeue_spin_time, std::memory_order_relaxed);
                    total_kmers_exported.fetch_add(classifier.total_kmers_exported, std::memory_order_relaxed);
                    total_kmers_send_to_tree.fetch_add(classifier.total_kmers_send_to_tree, std::memory_order_relaxed);
#endif
                }));
        }
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
#ifdef TEST_MODE
        for (uint32_t i = 0; i < classifier_count; i++)
        {
            std::cout << "Classifier " << i << " dealt k-mer count: " << deal_kmer_counts[i] << "\n";
        }
#endif
    }
};

#endif
