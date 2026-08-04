#ifndef LAYER_QUEUES_HEADER
#define LAYER_QUEUES_HEADER

#define LAYER_QUEUES_HEADER

#include "definition.h"
#include "MPMCRingQueue.h"

#include <array>
#include <memory>
#include <atomic>

template <uint32_t N>
class LayerQueues
{
    std::atomic<long long> size_{ 0 };
    std::array<std::shared_ptr<MPMCRingQueue<Task<N>, TASK_QUEUE_CAPACITY>>, MAX_DEPTH> queues_;
    std::shared_ptr<MPMCRingQueue<Task<N>, 1ULL << (2 * ROOT_BASES)>> final_drain_queue_;

public:
    explicit LayerQueues()
    {
        size_.store(0, std::memory_order_relaxed);
        for (uint32_t i = 0; i < MAX_DEPTH; ++i)
        {
            queues_[i] = std::make_shared<MPMCRingQueue<Task<N>, TASK_QUEUE_CAPACITY>>();
        }
        final_drain_queue_ = std::make_shared<MPMCRingQueue<Task<N>, 1ULL << (2 * ROOT_BASES)>>();
    }

    void initialize_final_drain_queue(std::vector<std::atomic<uint32_t>>& prefix_counts, node<N>* root_nodes)
    {
        struct PrefixInfo
        {
            uint64_t prefix;
            uint32_t count;
        };
        PrefixInfo prefix_info_array[1ULL << (2 * ROOT_BASES)];
        for (uint64_t i = 0; i < prefix_counts.size(); i++)
        {
            prefix_info_array[i].prefix = i;
            prefix_info_array[i].count = prefix_counts[i].load(std::memory_order_relaxed);
        }

        std::sort(prefix_info_array, prefix_info_array + (1ULL << (2 * ROOT_BASES)), [](const PrefixInfo& a, const PrefixInfo& b)
            { return a.count > b.count; });

        for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); ++i)
        {
            Task<N> task;
            task.current_node = &root_nodes[prefix_info_array[i].prefix];
            final_drain_queue_->enqueue(task);
        }
    }

    MPMCRingQueue<Task<N>, TASK_QUEUE_CAPACITY>* get_queue(uint32_t depth) const
    {
        return queues_[depth].get();
    }

    MPMCRingQueue<Task<N>, 1ULL << (2 * ROOT_BASES)>* get_final_drain_queue() const
    {
        return final_drain_queue_.get();
    }

    void increase_size()
    {
        size_.fetch_add(1, std::memory_order_release);
    }

    void decrease_size()
    {
        size_.fetch_sub(1, std::memory_order_release);
    }

    long long size() const
    {
        return size_.load(std::memory_order_acquire);
    }
};

#endif