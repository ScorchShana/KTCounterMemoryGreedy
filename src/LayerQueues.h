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
    struct alignas(CACHE_LINE_SIZE) PaddedAtomicSize
    {
        std::atomic<int64_t> size{ 0 };
    };

    std::array<PaddedAtomicSize, MAX_DEPTH> depth_sizes_{ 0 };
    std::array<std::shared_ptr<MPMCRingQueue<Task<N>, TASK_QUEUE_CAPACITY>>, MAX_DEPTH> queues_;
    std::shared_ptr<MPMCRingQueue<Task<N>, 1ULL << (2 * ROOT_BASES)>> final_drain_queue_;

public:
    explicit LayerQueues()
    {
        for (uint32_t i = 0; i < MAX_DEPTH; ++i)
        {
            depth_sizes_[i].size.store(0, std::memory_order_relaxed);
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

    void increase_size(const uint32_t depth, const uint32_t count = 1)
    {
        depth_sizes_[depth].size.fetch_add(count, std::memory_order_release);
    }

    void decrease_size(const uint32_t depth, const uint32_t count = 1)
    {
        depth_sizes_[depth].size.fetch_sub(count, std::memory_order_release);
    }

    long long size(const uint32_t depth) const
    {
        return depth_sizes_[depth].size.load(std::memory_order_acquire);
    }
};

#endif