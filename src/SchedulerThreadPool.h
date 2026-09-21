#ifndef SCHEDULER_THREAD_POOL_HEADER
#define SCHEDULER_THREAD_POOL_HEADER

#include "definition.h"
#include "LayerQueues.h"
#include "SPSCRingQueue.h"
#include "NewKmerTree.h"
#include "../src/SpinBackoff.h"
#include "ConcurrentOpenAddressHashMap.h"
#include "FinalDrainWriterThread.h"
#include "SplitMix.h"

#include <memory>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <utility>
#include <barrier>
#include <limits>
#include <cmath>


template <uint32_t N>
class SchedulerThreadPool final
{

    struct alignas(CACHE_LINE_SIZE) AlignedAtomicInt {
        std::atomic<int> value{};
    };

    struct alignas(CACHE_LINE_SIZE) AlignedAtomicUint32 {
        std::atomic<uint32_t> value{};
    };

    struct alignas(CACHE_LINE_SIZE) AlignedUint128 {
        __uint128_t value{};
    };


    // Worker thread constants
    static constexpr uint32_t INVALID_DEPTH = MAX_DEPTH;
    static constexpr uint32_t DRAIN_EMPTY_CONFIRM_ROUNDS = 3;
    static constexpr uint32_t MAX_PROCESS_TASKS = 16;
    static constexpr uint32_t MAX_PROCESS_LOCAL_STACK_TASKS = 16;
    static constexpr std::size_t LOCAL_STACK_WATERMARK = MAX_PROCESS_LOCAL_STACK_TASKS * 2;
    static constexpr std::size_t LOCAL_STACK_CRITICAL_WATERMARK = 128;
    static constexpr uint32_t LAST_DEPTH_DENOMINATOR = 2;
    // Scheduler algorithm constants
    static constexpr uint32_t SCHEDULE_INTERVAL_US = 50;
    static constexpr double WORK_EMA_ALPHA = 0.3;
    static constexpr uint32_t DRAIN_INTERVAL_US = 25;

    struct alignas(CACHE_LINE_SIZE) WorkerInfo
    {
        std::atomic<uint32_t> depth{ INVALID_DEPTH };
        std::atomic<std::size_t> local_stack_size{ 0 };
        std::array<std::atomic<uint64_t>, MAX_DEPTH> depth_task_cycles{};
        std::array<std::atomic<uint64_t>, MAX_DEPTH> depth_task_count;
    };

    struct WorkerSnapshot
    {
        uint32_t worker_id;
        uint32_t depth;
        std::size_t local_stack_size;
    };

    const uint32_t thread_count_;
    const uint32_t extra_drain_thread_count_;
    alignas(CACHE_LINE_SIZE) std::atomic<bool> running_{ false };
    alignas(CACHE_LINE_SIZE) std::atomic<bool> stop_requested_{ false };
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> active_producer{ 0 };

    KmerTree<N>* tree_ptr_ = nullptr;
    LayerQueues<N>* layer_queues_ptr_ = nullptr;
    std::thread scheduler_thread_;
    std::vector<std::unique_ptr<std::thread>> worker_threads_ptr_;
    std::vector<std::thread> extra_drain_threads_;
    std::vector<AlignedAtomicUint32> worker_commands_;

    std::array<AlignedAtomicInt, MAX_DEPTH> depth_worker_count{};
    std::vector<WorkerInfo> worker_infos;
    std::array<double, MAX_DEPTH> depth_ema_work{};
    std::vector<uint32_t> movein_workers_id;

    std::barrier<> drain_all_done_barrier;
    FinalDrainWriterThread drain_writer_thread_;

    inline static thread_local SpinBackoff<16, 128, 128 + 8, 8> backoff;
    inline static thread_local SplitMix64 rng;

    inline static thread_local std::array<uint64_t, MAX_DEPTH> local_depth_task_cycles{};
    inline static thread_local std::array<uint64_t, MAX_DEPTH> local_depth_task_count{};

#ifdef TEST_MODE
    inline static thread_local bool first_flag{ false };
    inline static thread_local uint64_t backoff_time_with_tasks{ 0 };
    inline static thread_local uint64_t steal_tasks{ 0 };
    inline static thread_local uint64_t home_tasks{ 0 };
    inline static thread_local uint64_t local_stack_tasks{ 0 };
    inline static thread_local std::array<uint64_t, MAX_DEPTH> depth_steal_tasks{};
    inline static thread_local std::array<uint64_t, MAX_DEPTH> depth_home_tasks{};
    std::atomic<uint64_t> total_request_sleep_time{ 0 };
    std::atomic<uint64_t> total_backoff_time_with_tasks{ 0 };
    std::atomic<uint64_t> total_steal_tasks{ 0 };
    std::atomic<uint64_t> total_home_tasks{ 0 };
    std::atomic<uint64_t> total_local_stack_tasks{ 0 };
    std::array<std::atomic<uint64_t>, MAX_DEPTH> total_depth_steal_tasks{};
    std::array<std::atomic<uint64_t>, MAX_DEPTH> total_depth_home_tasks{};
    std::vector<std::size_t> max_local_stack_size;
    std::mutex segment_histogram_lock;
    std::map<uint32_t, uint32_t> segment_histogram;
    std::atomic<uint64_t> total_segment_probe{ 0 };
    std::atomic<uint64_t> total_segment_probe_time{ 0 };
#endif

public:

#ifdef TEST_MODE
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> final_drain_writer_producer_enqueue_spin_time{ 0 };
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> final_drain_writer_producer_dequeue_spin_time{ 0 };
#endif

    explicit SchedulerThreadPool(uint32_t thread_count, uint32_t producer_count, uint32_t extra_drain_thread_count,
        KmerTree<N>* tree_ptr, LayerQueues<N>* layer_queues_ptr)
        : thread_count_(thread_count > 1 ? thread_count : 2), extra_drain_thread_count_(extra_drain_thread_count),
        active_producer(producer_count), tree_ptr_(tree_ptr),
        layer_queues_ptr_(layer_queues_ptr), worker_commands_(thread_count_ - 1), worker_infos(thread_count_ - 1),
        drain_all_done_barrier(thread_count_ - 1 + extra_drain_thread_count_),
        drain_writer_thread_(FINAL_DRAIN_RING_POOL_BLOCK_SIZE,
            thread_count_ - 1 + extra_drain_thread_count_)
    {
        for (auto& cmd : worker_commands_)
            cmd.value.store(INVALID_DEPTH, std::memory_order_relaxed);
        worker_threads_ptr_.reserve(thread_count_ - 1);

#ifdef TEST_MODE
        max_local_stack_size.resize(thread_count_ - 1, 0);
#endif
    }

    ~SchedulerThreadPool()
    {
#ifdef TEST_MODE
        std::cout << "SchedulerThreadPool Request Sleep : " << total_request_sleep_time.load() << std::endl;
        std::cout << "SchedulerThreadPool Backoff Time With Tasks : " << total_backoff_time_with_tasks.load() << std::endl;
        std::cout << "SchedulerThreadPool Steal Tasks : " << total_steal_tasks.load() << std::endl;
        std::cout << "SchedulerThreadPool Home Tasks : " << total_home_tasks.load() << std::endl;
        std::cout << "SchedulerThreadPool Local Stack Tasks : " << total_local_stack_tasks.load() << std::endl;
        for (uint32_t d = 0; d < MAX_DEPTH; ++d)
        {
            std::cout << "SchedulerThreadPool Depth " << d << " Home Tasks : " << total_depth_home_tasks[d].load() << std::endl;
        }
        for (uint32_t d = 0; d < MAX_DEPTH; ++d)
        {
            std::cout << "SchedulerThreadPool Depth " << d << " Steal Tasks : " << total_depth_steal_tasks[d].load() << std::endl;
        }

        for (uint32_t d = 0; d < MAX_DEPTH; ++d)
        {
            uint64_t total_cycles = 0;
            uint64_t total_tasks = 0;
            for (uint32_t w = 0;w < thread_count_ - 1; ++w)
            {
                total_cycles = worker_infos[w].depth_task_cycles[d].load();
                total_tasks = worker_infos[w].depth_task_count[d].load();
            }
            const uint64_t avg_cycles_per_task = (total_tasks > 0) ? total_cycles / total_tasks : 0;
            std::cout << "SchedulerThreadPool Depth " << d << " Task cycles : " << avg_cycles_per_task << std::endl;
        }

        std::size_t max_local_stack_size_total = 0;
        for (uint32_t w = 0; w < thread_count_ - 1; ++w)
        {
            max_local_stack_size_total = std::max(max_local_stack_size_total, max_local_stack_size[w]);
        }
        std::cout << "SchedulerThreadPool Max Local Stack Size : " << max_local_stack_size_total << std::endl;

        for (const auto& p : segment_histogram)
        {
            std::cout << "Segment " << p.first << " : " << p.second << std::endl;
        }

        std::cout << "Total Segment Probe : " << total_segment_probe.load() << std::endl;
        std::cout << "Total Segment Probe Time : " << total_segment_probe_time.load() << std::endl;
        std::cout << "Average Segment Probe Time : " << (total_segment_probe_time.load() > 0 ? static_cast<double>(total_segment_probe.load()) / total_segment_probe_time.load() : 0.0) << std::endl;
#endif
    }

    void start()
    {
        running_.store(true, std::memory_order_release);
        uint32_t cur_depth = 0;
        for (uint32_t i = 0; i + 1 < thread_count_; i++)
        {
            worker_init(i, cur_depth);
            worker_threads_ptr_.push_back(std::make_unique<std::thread>(&SchedulerThreadPool::worker_thread_loop, this, i));

            // depth_worker_count[cur_depth].value.fetch_add(1, std::memory_order_release);

            cur_depth++;
            cur_depth = cur_depth % MAX_DEPTH;
        }

        scheduler_thread_ = std::thread(&SchedulerThreadPool::scheduler_thread_loop, this);
    }

    void join()
    {
        stop_requested_.store(true, std::memory_order_release);
        if (scheduler_thread_.joinable())
        {
            scheduler_thread_.join();
        }
        for (auto& t : worker_threads_ptr_)
        {
            if (t->joinable())
            {
                t->join();
            }
        }
#ifdef TEST_MODE
        std::cout << "Final drain writer producer enqueue spin time: " << final_drain_writer_producer_enqueue_spin_time.load(std::memory_order_relaxed) << std::endl;
        std::cout << "Final drain writer producer dequeue spin time: " << final_drain_writer_producer_dequeue_spin_time.load(std::memory_order_relaxed) << std::endl;
#endif
        drain_writer_thread_.join();
    }

    void mark_producer_done()
    {
        active_producer.fetch_sub(1, std::memory_order_release);
    }

private:

    bool all_producers_done() const
    {
        return active_producer.load(std::memory_order_acquire) == 0;
    }

    bool are_all_depth_queues_empty() const
    {
        for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth)
        {
            if (layer_queues_ptr_->size(depth) != 0)
                return false;
        }
        return true;
    }

    void drain_all(const uint32_t start_depth)
    {
        Task<N> task;
        uint32_t stable_empty_rounds = 0;

        while (stable_empty_rounds < DRAIN_EMPTY_CONFIRM_ROUNDS)
        {
            uint32_t local_decrease_count = 0;
            for (uint32_t k = 0; k < MAX_DEPTH; k++)
            {
                uint32_t depth = (k + start_depth) % MAX_DEPTH;
                auto queue = layer_queues_ptr_->get_queue(depth);
                while (queue->try_dequeue(task))
                {
                    tree_ptr_->thread_add_kmer(task);
                    ++local_decrease_count;

                }

                if (local_decrease_count > 0)
                {
                    layer_queues_ptr_->decrease_size(depth, local_decrease_count);
                    local_decrease_count = 0;
                }
            }

            bool found_local_task_work = tree_ptr_->check_and_deal_with_local_stack();

            if (!found_local_task_work && are_all_depth_queues_empty())
            {
                stable_empty_rounds++;
            }
            else
            {
                stable_empty_rounds = 0;
            }
        }
    }

    uint32_t process_batch_at_depth(const uint32_t depth,
        const uint32_t max_process_tasks = MAX_PROCESS_TASKS)
    {
        Task<N> task;
        auto queue = layer_queues_ptr_->get_queue(depth);
        uint32_t processed = 0;
        while (processed < max_process_tasks && queue->try_dequeue(task))
        {
            tree_ptr_->thread_add_kmer(task);
            processed++;
        }

        if (processed > 0)
        {
            layer_queues_ptr_->decrease_size(depth, processed);
        }

#ifdef TEST_MODE
        home_tasks += processed;
        depth_home_tasks[depth] += processed;
#endif

        return processed;
    }

    bool try_steal()
    {
        for (int steal_depth = MAX_DEPTH - 1; steal_depth >= 0; --steal_depth)
        {
            auto queue = layer_queues_ptr_->get_queue(steal_depth);
            Task<N> task;
            if (queue->try_dequeue(task))
            {
                tree_ptr_->thread_add_kmer(task);
                layer_queues_ptr_->decrease_size(static_cast<uint32_t>(steal_depth));

#ifdef TEST_MODE
                ++steal_tasks;
                ++depth_steal_tasks[steal_depth];
#endif

                return true;
            }
        }
        return false;
    }

    // 所有 depth 切换全部由 try_switch_depth 完成
    uint32_t try_switch_depth(const uint32_t worker_id)
    {
        const uint32_t last_depth = worker_infos[worker_id].depth.load(std::memory_order_acquire);
        const uint32_t command_depth_snapshot = worker_commands_[worker_id].value.load(std::memory_order_acquire);
        if (command_depth_snapshot == INVALID_DEPTH || command_depth_snapshot == last_depth) [[likely]]
        {
            return last_depth;
        }

        const uint32_t new_depth = worker_commands_[worker_id].value.exchange(INVALID_DEPTH, std::memory_order_acq_rel);
        const uint32_t res_depth = (new_depth != INVALID_DEPTH) ? new_depth : last_depth;
        if (res_depth != last_depth)
        {
            worker_infos[worker_id].depth.store(new_depth, std::memory_order_release);
            backoff.reset();
        }
        return res_depth;
    }

    bool ensure_local_stack_within_watermark(const uint32_t worker_id)
    {
        size_t local_size = tree_ptr_->get_local_stack_size();

        // 1. 内存保护模式
        if (local_size >= LOCAL_STACK_CRITICAL_WATERMARK)
        {
#ifdef TEST_MODE
            const std::size_t max_size = local_size;
            const std::size_t prev_max_size = max_local_stack_size[worker_id];
            max_local_stack_size[worker_id] = std::max(max_size, prev_max_size);
#endif

            do
            {
                uint64_t local_processed = tree_ptr_->deal_with_local_stack(MAX_PROCESS_LOCAL_STACK_TASKS);

#ifdef TEST_MODE
                local_stack_tasks += local_processed;
#endif

            } while (tree_ptr_->get_local_stack_size() > LOCAL_STACK_WATERMARK);

            backoff.reset();
            return true;
        }
        else if (local_size >= LOCAL_STACK_WATERMARK)
        {
            uint64_t local_processed = tree_ptr_->deal_with_local_stack(MAX_PROCESS_LOCAL_STACK_TASKS);

#ifdef TEST_MODE
            local_stack_tasks += local_processed;
            const std::size_t max_size = local_size;
            const std::size_t prev_max_size = max_local_stack_size[worker_id];
            max_local_stack_size[worker_id] = std::max(max_size, prev_max_size);
#endif

            return true;
        }
        return false;
    }

    void try_work(const uint32_t worker_id)
    {
        if (ensure_local_stack_within_watermark(worker_id))
        {
            worker_infos[worker_id].local_stack_size.store(tree_ptr_->get_local_stack_size(), std::memory_order_relaxed);
            return;
        }

        const uint32_t depth = try_switch_depth(worker_id);

        const uint32_t max_process_tasks = (depth + 1 == INVALID_DEPTH) ? MAX_PROCESS_TASKS / LAST_DEPTH_DENOMINATOR : MAX_PROCESS_TASKS;

        const uint64_t start_cycles = __rdtsc();
        uint32_t processed = process_batch_at_depth(depth, max_process_tasks);
        const uint64_t end_cycles = __rdtsc();

        if (processed == max_process_tasks)
        {
            backoff.reset();
            if (end_cycles > start_cycles) [[likely]]
            {
                local_depth_task_cycles[depth] += end_cycles - start_cycles;
                local_depth_task_count[depth] += processed;
            }
#ifdef TEST_MODE
            first_flag = true;
#endif
        }
        else if (processed)
        {
            backoff.decay();
            if (end_cycles > start_cycles) [[likely]]
            {
                local_depth_task_cycles[depth] += end_cycles - start_cycles;
                local_depth_task_count[depth] += processed;
            }
#ifdef TEST_MODE
            first_flag = true;
#endif
        }
        else if (tree_ptr_->get_local_stack_size() > 0)
        {
            uint64_t local_processed = tree_ptr_->deal_with_local_stack(MAX_PROCESS_LOCAL_STACK_TASKS);

#ifdef TEST_MODE
            local_stack_tasks += local_processed;
#endif

            backoff.decay();
        }
        else if (!try_steal())
        {

#ifdef TEST_MODE
            if (first_flag && !are_all_depth_queues_empty()) {
                ++backoff_time_with_tasks;
            }
#endif

            backoff.backoff();
        }
        else
        {
            backoff.decay();
        }
    }

    void worker_init(const uint32_t worker_id, const uint32_t depth)
    {
        worker_infos[worker_id].depth.store(depth, std::memory_order_release);
    }

    void worker_thread_loop(const uint32_t worker_id)
    {
#if HAS_LIBNUMA
        numa_run_on_node(numa_nodes[(worker_id + 1) % numa_nodes.size()]);
#endif
        ConcurrentOpenAddressHashMap<N>::set_memory_pool(tree_ptr_->get_memory_pool());
        uint32_t loop_round = 0;

        while (!stop_requested_.load(std::memory_order_acquire) || !all_producers_done())
        {
            try_work(worker_id);
            loop_round++;
            if ((loop_round & 63) == 0)
            {
                worker_infos[worker_id].local_stack_size.store(tree_ptr_->get_local_stack_size(),
                    std::memory_order_relaxed);
                for (uint32_t d = 0; d < MAX_DEPTH; ++d)
                {
                    worker_infos[worker_id].depth_task_cycles[d].store(local_depth_task_cycles[d], std::memory_order_relaxed);
                    worker_infos[worker_id].depth_task_count[d].store(local_depth_task_count[d], std::memory_order_relaxed);
                }
            }

        }

#ifdef TEST_MODE
        total_request_sleep_time.fetch_add(backoff.get_request_sleep_time(), std::memory_order_relaxed);
        total_backoff_time_with_tasks.fetch_add(backoff_time_with_tasks, std::memory_order_relaxed);
        total_steal_tasks.fetch_add(steal_tasks, std::memory_order_relaxed);
        total_home_tasks.fetch_add(home_tasks, std::memory_order_relaxed);
        total_local_stack_tasks.fetch_add(local_stack_tasks, std::memory_order_relaxed);
        for (uint32_t d = 0; d < MAX_DEPTH; ++d)
        {
            total_depth_steal_tasks[d].fetch_add(depth_steal_tasks[d], std::memory_order_relaxed);
            total_depth_home_tasks[d].fetch_add(depth_home_tasks[d], std::memory_order_relaxed);
        }
        total_segment_probe.fetch_add(ConcurrentOpenAddressHashMap<N>::segment_probe, std::memory_order_relaxed);
        total_segment_probe_time.fetch_add(ConcurrentOpenAddressHashMap<N>::segment_probe_time, std::memory_order_relaxed);
#endif

        const uint32_t total_workers = thread_count_ - 1;

        if (worker_id == 0) [[unlikely]]
        {
            drain_writer_thread_.start();
            if (extra_drain_thread_count_ > 0)
            {
                extra_drain_threads_.reserve(extra_drain_thread_count_);
                for (uint32_t i = 0; i < extra_drain_thread_count_; ++i)
                {
                    uint32_t extra_id = total_workers + i;
                    extra_drain_threads_.emplace_back([this, extra_id]()
                        {
                            ConcurrentOpenAddressHashMap<N>::set_memory_pool(tree_ptr_->get_memory_pool());
                            drain_all(extra_id % MAX_DEPTH);
                            drain_all_done_barrier.arrive_and_wait();

                            FinalDrainWriter<N> writer(tree_ptr_->get_k_length(),
                                drain_writer_thread_.pool());
                            auto fq = layer_queues_ptr_->get_final_drain_queue();
                            Task<N> task;
                            while (fq->try_dequeue(task))
                                tree_ptr_->final_drain_root(task.current_node, writer);

                            writer.close();
                            drain_writer_thread_.pool()->producer_set_finished();
#ifdef TEST_MODE
                            final_drain_writer_producer_enqueue_spin_time.fetch_add(writer.producer_enqueue_spin_time, std::memory_order_relaxed);
                            final_drain_writer_producer_dequeue_spin_time.fetch_add(writer.producer_dequeue_spin_time, std::memory_order_relaxed);
                            {
                                std::lock_guard<std::mutex> lock(segment_histogram_lock);
                                for (auto p : ConcurrentOpenAddressHashMap<N>::get_segment_histogram()) {
                                    segment_histogram[p.first] += p.second;
                                }
                            }
#endif
                        });
                }
            }
        }

        drain_all(worker_infos[worker_id].depth.load(std::memory_order_acquire));
        drain_all_done_barrier.arrive_and_wait();

        {
            FinalDrainWriter<N> writer(tree_ptr_->get_k_length(),
                drain_writer_thread_.pool());

            auto fq = layer_queues_ptr_->get_final_drain_queue();
            Task<N> task;
            while (fq->try_dequeue(task))
                tree_ptr_->final_drain_root(task.current_node, writer);

            writer.close();
            drain_writer_thread_.pool()->producer_set_finished();
#ifdef TEST_MODE
            final_drain_writer_producer_enqueue_spin_time.fetch_add(writer.producer_enqueue_spin_time, std::memory_order_relaxed);
            final_drain_writer_producer_dequeue_spin_time.fetch_add(writer.producer_dequeue_spin_time, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lock(segment_histogram_lock);
                for (auto p : ConcurrentOpenAddressHashMap<N>::get_segment_histogram()) {
                    segment_histogram[p.first] += p.second;
                }
            }
#endif
        }

        if (worker_id == 0 && extra_drain_thread_count_ > 0) [[unlikely]]
        {
            for (auto& t : extra_drain_threads_)
                if (t.joinable()) t.join();
            extra_drain_threads_.clear();
        }
    }

    void get_snapshot(std::array<uint32_t, MAX_DEPTH>& depth_worker_count_snapshot,
        std::vector<WorkerSnapshot>& worker_snapshots,
        std::array<uint64_t, MAX_DEPTH>& depth_new_cycles_snapshot,
        std::array<uint64_t, MAX_DEPTH>& depth_new_taks_snapshot)
    {
        // for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth)
        // {
        //     depth_worker_count_snapshot[depth] = depth_worker_count[depth].load(std::memory_order_acquire);
        // }

        const uint32_t total_workers = thread_count_ - 1;

        depth_worker_count_snapshot.fill(0);
        depth_new_cycles_snapshot.fill(0);
        depth_new_taks_snapshot.fill(0);

        for (uint32_t w = 0; w < total_workers; ++w)
        {
            worker_snapshots[w].worker_id = w;
            worker_snapshots[w].depth = worker_infos[w].depth.load(std::memory_order_acquire);
            worker_snapshots[w].local_stack_size = worker_infos[w].local_stack_size.load(std::memory_order_acquire);
            for (uint32_t d = 0; d < MAX_DEPTH; ++d)
            {
                depth_new_cycles_snapshot[d] += worker_infos[w].depth_task_cycles[d].load(std::memory_order_acquire);
                depth_new_taks_snapshot[d] += worker_infos[w].depth_task_count[d].load(std::memory_order_acquire);
            }

            if (worker_snapshots[w].depth < MAX_DEPTH)
            {
                ++depth_worker_count_snapshot[worker_snapshots[w].depth];
            }
            else
            {
                ++depth_worker_count_snapshot[MAX_DEPTH - 1];
            }
        }
    }

    void get_depth_cycles_per_task(std::array<double, MAX_DEPTH>& depth_cycles_per_task,
        std::array<uint64_t, MAX_DEPTH>& depth_new_cycles_snapshot,
        std::array<uint64_t, MAX_DEPTH>& depth_new_taks_snapshot,
        std::array<uint64_t, MAX_DEPTH>& depth_cycles_snapshot,
        std::array<uint64_t, MAX_DEPTH>& depth_tasks_snapshot)
    {

        constexpr double CYCLES_PER_TASK_SMOOTHING_FACTOR = 0.6;

        for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth)
        {
            const uint64_t depth_cycles = depth_new_cycles_snapshot[depth];
            const uint64_t depth_tasks = depth_new_taks_snapshot[depth];

            const uint64_t delta_depth_cycles = depth_cycles - depth_cycles_snapshot[depth];
            const uint64_t delta_depth_tasks = depth_tasks - depth_tasks_snapshot[depth];

            depth_cycles_snapshot[depth] = depth_cycles;
            depth_tasks_snapshot[depth] = depth_tasks;

            if (delta_depth_tasks > 0 && delta_depth_cycles > 0)
            {
                const double new_cycles_per_task = static_cast<double>(delta_depth_cycles) / static_cast<double>(delta_depth_tasks);
                if (depth_cycles_per_task[depth] <= 2) [[unlikely]]
                {
                    depth_cycles_per_task[depth] = new_cycles_per_task;
                }
                else
                {
                    depth_cycles_per_task[depth] = CYCLES_PER_TASK_SMOOTHING_FACTOR * new_cycles_per_task +
                        (1.0 - CYCLES_PER_TASK_SMOOTHING_FACTOR) * depth_cycles_per_task[depth];
                }
            }
        }
    }

    void compute_ema_work(std::array<double, MAX_DEPTH>& depth_cycles_per_task)
    {
        const uint32_t total_workers = thread_count_ - 1;

        double min_depth_cycles_per_task = depth_cycles_per_task[0];

        for (uint32_t d = 1; d < MAX_DEPTH; ++d)
        {
            min_depth_cycles_per_task = std::min(min_depth_cycles_per_task, depth_cycles_per_task[d]);
        }

        const double depth_cycles_per_task_corrected_factor = (min_depth_cycles_per_task < 1e-9) ? 1 : 1 / min_depth_cycles_per_task;

        for (uint32_t d = 0; d < MAX_DEPTH; ++d)
        {
            const int64_t tracked_size = layer_queues_ptr_->size(d);
            uint64_t qsize = tracked_size > 0 ? static_cast<uint64_t>(tracked_size) : 0;

            const double corrected_depth_cycles_per_task = depth_cycles_per_task[d] * depth_cycles_per_task_corrected_factor;
            const double raw = static_cast<double>(qsize) * corrected_depth_cycles_per_task;

            depth_ema_work[d] = WORK_EMA_ALPHA * raw + (1.0 - WORK_EMA_ALPHA) * depth_ema_work[d];
        }
    }

    void compute_depth_desired_worker(std::array<uint32_t, MAX_DEPTH>& depth_desired_worker,
        const std::vector<WorkerSnapshot>& worker_snapshots)
    {
        const uint32_t total_workers = thread_count_ - 1;

        std::array<uint32_t, MAX_DEPTH> depth_index{};
        for (uint32_t d = 0; d < MAX_DEPTH; ++d)
        {
            depth_index[d] = d;
        }
        for (uint32_t i = 0; i < MAX_DEPTH; ++i)
        {
            for (uint32_t j = i + 1; j < MAX_DEPTH; ++j)
            {
                if (depth_ema_work[depth_index[j]] > depth_ema_work[depth_index[i]])
                {
                    std::swap(depth_index[i], depth_index[j]);
                }
            }
        }

        if (total_workers <= MAX_DEPTH) [[unlikely]]
        {
            depth_desired_worker.fill(0);
            for (uint32_t ranking = 0;ranking < total_workers;++ranking)
            {
                depth_desired_worker[depth_index[ranking]] = 1;
            }
        }
        else
        {
            const uint32_t extra_workers = total_workers - MAX_DEPTH;
            double total_work = 0.0;

            for (uint32_t d = 0; d < MAX_DEPTH; ++d)
            {
                total_work += depth_ema_work[d];
            }

            if (total_work < 1e-9)
            {
                return;
            }

            depth_desired_worker.fill(0);

            uint32_t assigned_extra = 0;

            for (uint32_t d = 0; d < MAX_DEPTH; ++d)
            {
                double depth_shared_part = depth_ema_work[d] / total_work;

                double depth_exact_workers = static_cast<double>(extra_workers) * depth_shared_part;

                const uint32_t depth_whole_workers = static_cast<uint32_t>(std::floor(depth_exact_workers));

                depth_desired_worker[d] = 1 + depth_whole_workers;
                assigned_extra += depth_whole_workers;
            }

            uint32_t remaining_extra = extra_workers - assigned_extra;
            uint32_t ranking = 0;
            while (remaining_extra > 0)
            {
                ++depth_desired_worker[depth_index[ranking]];
                ranking = (ranking + 1) % MAX_DEPTH;
                --remaining_extra;
            }
        }
    }

    static int64_t compute_worker_score_at_depth(const WorkerSnapshot& worker_snapshot,
        const uint32_t& depth)
    {
        int64_t score = 0;
        if (worker_snapshot.depth == depth)
        {
            score += 1LL << 50;
            score += static_cast<int64_t>(worker_snapshot.local_stack_size);
        }
        else {
            score -= static_cast<int64_t>(worker_snapshot.local_stack_size);
        }
        return score;
    }

    static bool cmp_by_depth(const WorkerSnapshot& ws1, const WorkerSnapshot& ws2)
    {
        return ws1.depth < ws2.depth;
    }

    static bool cmp_by_local_stack_size(const WorkerSnapshot& ws1, const WorkerSnapshot& ws2)
    {
        return ws1.local_stack_size < ws2.local_stack_size;
    }

    void send_command_to_workers(const std::array<uint32_t, MAX_DEPTH>& depth_worker_count_snapshot,
        const std::array<uint32_t, MAX_DEPTH>& depth_desired_worker,
        std::vector<WorkerSnapshot>& worker_snapshots)
    {
        const uint32_t total_workers = thread_count_ - 1;

        std::array<int32_t, MAX_DEPTH>movein_worker_count;
        movein_worker_count.fill(0);
        movein_workers_id.clear();

        for (uint32_t d = 0;d < MAX_DEPTH;d++) {
            movein_worker_count[d] = depth_desired_worker[d] - depth_worker_count_snapshot[d];
        }

        std::sort(worker_snapshots.begin(), worker_snapshots.begin() + total_workers, cmp_by_depth);

        uint32_t worker_sum = 0;
        for (uint32_t d = 0;d < MAX_DEPTH;d++)
        {
            const uint32_t worker_count_this_depth = depth_worker_count_snapshot[d];
            if (movein_worker_count[d] < 0)
            {
                std::sort(worker_snapshots.begin() + worker_sum,
                    worker_snapshots.begin() + worker_sum + worker_count_this_depth,
                    cmp_by_local_stack_size);

                uint32_t index = 0;
                while (movein_worker_count[d] < 0) {
                    movein_workers_id.push_back(worker_snapshots[index + worker_sum].worker_id);
                    ++movein_worker_count[d];
                    ++index;
                }
            }
            worker_sum += worker_count_this_depth;
        }

        worker_sum = 0;
        for (uint32_t d = 0;d < MAX_DEPTH;d++)
        {
            const uint32_t worker_count_this_depth = depth_worker_count_snapshot[d];
            if (movein_worker_count[d] > 0)
            {
                for (uint32_t i = 0;i < movein_worker_count[d];i++)
                {
                    uint32_t worker_id = movein_workers_id.back();
                    movein_workers_id.pop_back();
                    worker_commands_[worker_id].value.store(d, std::memory_order_release);
                }
            }
            worker_sum += worker_count_this_depth;
        }
    }

    void scheduler_thread_loop()
    {
        const uint32_t total_workers = thread_count_ - 1;

        std::array<uint32_t, MAX_DEPTH> depth_worker_count_snapshot{};
        std::array<uint64_t, MAX_DEPTH> depth_new_cycles_snapshot{};
        std::array<uint64_t, MAX_DEPTH> depth_new_taks_snapshot{};

        std::array<double, MAX_DEPTH> depth_cycles_per_task{};
        std::array<uint64_t, MAX_DEPTH> depth_cycles_snapshot{};
        std::array<uint64_t, MAX_DEPTH> depth_tasks_snapshot{};
        std::array<uint32_t, MAX_DEPTH> depth_desired_worker{};

        std::vector<WorkerSnapshot> worker_snapshots(thread_count_ - 1);

        for (uint32_t w = 0;w < total_workers;++w)
        {
            worker_snapshots[w].worker_id = w;
            worker_snapshots[w].depth = w % MAX_DEPTH;
            worker_snapshots[w].local_stack_size = 0;
            ++depth_desired_worker[w % MAX_DEPTH];
        }

        depth_cycles_per_task.fill(1.0);

        while (!stop_requested_.load(std::memory_order_acquire) || !all_producers_done())
        {
            bool is_drain = all_producers_done();

            get_snapshot(depth_worker_count_snapshot, worker_snapshots, depth_new_cycles_snapshot, depth_new_taks_snapshot);
            get_depth_cycles_per_task(depth_cycles_per_task, depth_new_cycles_snapshot, depth_new_taks_snapshot, depth_cycles_snapshot, depth_tasks_snapshot);
            compute_ema_work(depth_cycles_per_task);
            compute_depth_desired_worker(depth_desired_worker, worker_snapshots);
            send_command_to_workers(depth_worker_count_snapshot, depth_desired_worker, worker_snapshots);

            std::this_thread::sleep_for(std::chrono::microseconds(SCHEDULE_INTERVAL_US));
        }
    }
};

#endif // SCHEDULING_THREAD_POOL_HEADER