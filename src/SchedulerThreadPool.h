#ifndef SCHEDULER_THREAD_POOL_HEADER
#define SCHEDULER_THREAD_POOL_HEADER

#include "definition.h"
#include "LayerQueues.h"
#include "SPSCRingQueue.h"
#include "NewKmerTree.h"
#include "../src/SpinBackoff.h"
#include "ConcurrentMap.h"
#include "FinalDrainWriterThread.h"

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
    // Worker thread constants
    static constexpr uint32_t INVALID_DEPTH = MAX_DEPTH;
    static constexpr uint32_t DRAIN_EMPTY_CONFIRM_ROUNDS = 3;
    static constexpr uint32_t MAX_PROCESS_TASKS = 128;
    static constexpr uint32_t FORCE_DEAL_WITH_LOCAL_STACK_ROUND = 32;
    static constexpr uint32_t LAST_DEPTH_DENOMINATOR = 4;
    // Scheduler algorithm constants
    static constexpr uint32_t SCHEDULE_INTERVAL_NS = 500;
    static constexpr double PRESSURE_EMA_ALPHA = 0.6;
    static constexpr double HYSTERESIS_LOG_DELTA = std::log2(1.7);          // log2(1.5)
    static constexpr uint32_t DRAIN_INTERVAL_NS = 250;
    static constexpr double SELF_SCHEDULE_LOG_DELTA = HYSTERESIS_LOG_DELTA * 0.3;

    struct WorkerInfo
    {
        alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> depth{ INVALID_DEPTH };
        alignas(CACHE_LINE_SIZE) std::atomic<uint32_t> local_stack_size{ 0 };
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
    std::vector<alignas(CACHE_LINE_SIZE) std::atomic<uint32_t>> worker_commands_;

    std::array<std::atomic<int>, MAX_DEPTH> depth_worker_count{};
    std::vector<WorkerInfo> worker_infos;
    std::array<double, MAX_DEPTH> depth_ema_pressure_{};

    std::barrier<> drain_all_done_barrier;
    std::barrier<> drain_root_done_barrier_;
    FinalDrainWriterThread drain_writer_thread_;

    inline static thread_local SpinBackoff<> backoff;
    inline static thread_local uint32_t self_schedule_depth = INVALID_DEPTH;
    inline static thread_local uint32_t self_schedule_denominator = 1;

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
        drain_root_done_barrier_(thread_count_ - 1 + extra_drain_thread_count_),
        drain_writer_thread_(FINAL_DRAIN_RING_POOL_BLOCK_SIZE,
            thread_count_ - 1 + extra_drain_thread_count_)
    {
        for (auto& cmd : worker_commands_)
            cmd.store(INVALID_DEPTH, std::memory_order_relaxed);
        worker_threads_ptr_.reserve(thread_count_ - 1);
    }

    ~SchedulerThreadPool()
    {

    }

    void start()
    {
        running_.store(true, std::memory_order_release);
        uint32_t cur_depth = 0;
        for (uint32_t i = 0; i + 1 < thread_count_; i++)
        {
            worker_init(i, cur_depth);
            worker_threads_ptr_.push_back(std::make_unique<std::thread>(&SchedulerThreadPool::worker_thread_loop, this, i));

            depth_worker_count[cur_depth].fetch_add(1, std::memory_order_release);

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
        return layer_queues_ptr_->size() == 0;
    }

    void drain_all(const uint32_t start_depth)
    {
        Task<N> task;
        uint32_t stable_empty_rounds = 0;

        while (stable_empty_rounds < DRAIN_EMPTY_CONFIRM_ROUNDS)
        {
            for (uint32_t k = 0; k < MAX_DEPTH; k++)
            {
                uint32_t depth = (k + start_depth) % MAX_DEPTH;
                auto queue = layer_queues_ptr_->get_queue(static_cast<uint32_t>(depth));
                while (queue->try_dequeue(task))
                {
                    layer_queues_ptr_->decrease_size();
                    tree_ptr_->thread_add_kmer(task);
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

    uint32_t process_batch_at_depth(const uint32_t depth, const uint32_t max_process_tasks = MAX_PROCESS_TASKS)
    {
        Task<N> task;
        auto queue = layer_queues_ptr_->get_queue(depth);
        uint32_t processed = 0;
        while (processed < max_process_tasks && queue->try_dequeue(task))
        {
            tree_ptr_->thread_add_kmer(task);
            layer_queues_ptr_->decrease_size();
            processed++;
        }
        return processed;
    }

    // 所有 depth 切换全部由 try_switch_depth 完成
    bool try_switch_depth(const uint32_t worker_id, const uint32_t depth)
    {
        const uint32_t max_process_task = (depth + 1 == INVALID_DEPTH) ? MAX_PROCESS_TASKS / LAST_DEPTH_DENOMINATOR : MAX_PROCESS_TASKS;
        uint32_t processed = process_batch_at_depth(depth, max_process_task / self_schedule_denominator);
        const uint32_t cur_denominator = self_schedule_denominator;
        self_schedule_denominator = 1;

        uint32_t new_depth = worker_commands_[worker_id].exchange(INVALID_DEPTH, std::memory_order_acq_rel);
        if (new_depth != INVALID_DEPTH)
        {
            if (self_schedule_depth != INVALID_DEPTH)
            {
                self_schedule_denominator = (self_schedule_depth == new_depth) ? 2 : 1;
                self_schedule_depth = INVALID_DEPTH;
            }
            worker_infos[worker_id].depth.store(new_depth, std::memory_order_release);
            depth_worker_count[depth].fetch_sub(1, std::memory_order_release);
            depth_worker_count[new_depth].fetch_add(1, std::memory_order_release);
            backoff.reset();
            return true;
        }
        else
        {
            if (processed == MAX_PROCESS_TASKS / cur_denominator)
            {
                backoff.double_decay();
            }
            else if (processed)
            {
                backoff.decay();
            }
            else if (tree_ptr_->get_local_stack_size() > 0)
            {
                tree_ptr_->deal_with_local_stack();
                backoff.decay();
            }
            else if (depth_worker_count[depth].load(std::memory_order_relaxed) > 1)
            {
                for (uint32_t d = 0; d < MAX_DEPTH; ++d)
                {
                    if (d == depth) continue;
                    uint32_t qsize = layer_queues_ptr_->get_queue(d)->size();
                    const int depth_worker_num = depth_worker_count[d].load(std::memory_order_relaxed);
                    // const int hard_worker_upper_bound = std::max<int>(1, (thread_count_ - 1) * 3 / 4);
                    // if (depth_worker_num >= hard_worker_upper_bound) continue;
                    // double raw = static_cast<double>(qsize) / (depth_worker_num + 1.0);
                    // double depth_ema_pressure = std::log2(raw + 1.0);
                    // if (depth_ema_pressure > SELF_SCHEDULE_LOG_DELTA)
                    // {
                    //     worker_commands_[worker_id].store(d, std::memory_order_release);
                    //     self_schedule_depth = d;
                    //     return false;
                    // }
                    if (d + 1 < MAX_DEPTH && qsize > 0.3 * depth_worker_num * MAX_PROCESS_TASKS)
                    {
                        worker_commands_[worker_id].store(d, std::memory_order_release);
                        self_schedule_depth = d;
                        return false;
                    }
                    else if (d + 1 == MAX_DEPTH && qsize > 0.3 * depth_worker_num * MAX_PROCESS_TASKS / 2)
                    {
                        worker_commands_[worker_id].store(d, std::memory_order_release);
                        self_schedule_depth = d;
                        return false;
                    }
                }
                backoff.backoff();
            }
            else
            {
                backoff.decay();
            }

            return false;
        }
    }

    void try_work(const uint32_t worker_id)
    {
        uint32_t work_depth = worker_infos[worker_id].depth.load(std::memory_order_acquire);
        try_switch_depth(worker_id, work_depth);
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
        ConcurrentMap<N>::set_thread_id(worker_id);
        uint32_t loop_round = 0;

        while (!stop_requested_.load(std::memory_order_acquire) || !all_producers_done())
        {
            try_work(worker_id);
            loop_round++;
            if ((loop_round & 0x7) == 0)
            {
                worker_infos[worker_id].local_stack_size.store(
                    static_cast<uint32_t>(tree_ptr_->get_local_stack_size()),
                    std::memory_order_release);
            }
            if (loop_round >= FORCE_DEAL_WITH_LOCAL_STACK_ROUND)
            {
                tree_ptr_->check_and_deal_with_local_stack();
                loop_round = 0;
            }
        }
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
                            ConcurrentMap<N>::set_thread_id(extra_id);
                            drain_all(extra_id % MAX_DEPTH);
                            drain_all_done_barrier.arrive_and_wait();

                            FinalDrainWriter<N> writer(tree_ptr_->get_k_length(),
                                drain_writer_thread_.pool());
                            auto fq = layer_queues_ptr_->get_final_drain_queue();
                            Task<N> task;
                            while (fq->try_dequeue(task))
                                tree_ptr_->final_drain_root(task.current_node, writer);

                            drain_root_done_barrier_.arrive_and_wait();

                            ConcurrentMap<N>::export_thread_node_count(writer, extra_id);
                            writer.close();
                            drain_writer_thread_.pool()->producer_set_finished();
#ifdef TEST_MODE
                            final_drain_writer_producer_enqueue_spin_time.fetch_add(writer.producer_enqueue_spin_time, std::memory_order_relaxed);
                            final_drain_writer_producer_dequeue_spin_time.fetch_add(writer.producer_dequeue_spin_time, std::memory_order_relaxed);
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

            drain_root_done_barrier_.arrive_and_wait();

            ConcurrentMap<N>::export_thread_node_count(writer, worker_id);
            writer.close();
            drain_writer_thread_.pool()->producer_set_finished();
#ifdef TEST_MODE
            final_drain_writer_producer_enqueue_spin_time.fetch_add(writer.producer_enqueue_spin_time, std::memory_order_relaxed);
            final_drain_writer_producer_dequeue_spin_time.fetch_add(writer.producer_dequeue_spin_time, std::memory_order_relaxed);
#endif
        }

        if (worker_id == 0 && extra_drain_thread_count_ > 0) [[unlikely]]
        {
            for (auto& t : extra_drain_threads_)
                if (t.joinable()) t.join();
            extra_drain_threads_.clear();
        }
    }

    void get_depth_worker_count_snapshot(std::array<uint32_t, MAX_DEPTH>& depth_worker_count_snapshot)
    {
        for (uint32_t depth = 0; depth < MAX_DEPTH; ++depth)
        {
            depth_worker_count_snapshot[depth] = depth_worker_count[depth].load(std::memory_order_acquire);
        }
    }

    void compute_ema_pressure(const std::array<uint32_t, MAX_DEPTH>& worker_snapshot)
    {
        const uint32_t total_workers = thread_count_ - 1;
        std::array<uint32_t, MAX_DEPTH> hidden_size{};
        for (uint32_t w = 0; w < total_workers; ++w)
        {
            uint32_t wd = worker_infos[w].depth.load(std::memory_order_acquire);
            if (wd < MAX_DEPTH)
                hidden_size[wd] +=
                worker_infos[w].local_stack_size.load(std::memory_order_acquire);
        }

        for (uint32_t d = 0; d < MAX_DEPTH; ++d)
        {
            uint32_t qsize = layer_queues_ptr_->get_queue(d)->size();
            double raw;
            if (d + 1 < MAX_DEPTH) {
                raw = static_cast<double>(qsize + hidden_size[d]) / (worker_snapshot[d] + 1.0) / MAX_PROCESS_TASKS;
            }
            else {
                raw = static_cast<double>(qsize + hidden_size[d]) / (worker_snapshot[d] + 1.0) / MAX_PROCESS_TASKS * LAST_DEPTH_DENOMINATOR;
            }
            double log_raw = std::log2(raw + 1.0);
            depth_ema_pressure_[d] = PRESSURE_EMA_ALPHA * log_raw + (1.0 - PRESSURE_EMA_ALPHA) * depth_ema_pressure_[d];
        }
    }

    void update_dynamic_lower_bounds(
        std::array<uint32_t, MAX_DEPTH>& lower_bound,
        const uint32_t total_workers,
        const uint32_t hard_upper_bound)
    {
        double total_ema = 0.0;
        for (uint32_t d = 0; d < MAX_DEPTH; ++d)
            total_ema += depth_ema_pressure_[d];

        if (total_workers <= MAX_DEPTH)
        {
            lower_bound.fill(1);
            return;
        }

        const uint32_t distributable = total_workers - MAX_DEPTH;
        const uint32_t max_extra = std::min(
            std::max<uint32_t>(1, total_workers / 2),
            distributable);

        if (total_ema > 0.05)
        {
            for (uint32_t d = 0; d < MAX_DEPTH; ++d)
            {
                double share = depth_ema_pressure_[d] / total_ema;
                uint32_t dynamic_min = 1 + static_cast<uint32_t>(max_extra * share);
                lower_bound[d] = std::min(dynamic_min, hard_upper_bound);
            }
        }
        else
        {
            lower_bound.fill(1);
        }
    }

    void scheduler_thread_loop()
    {
        std::array<uint32_t, MAX_DEPTH> depth_worker_count_snapshot{};
        std::array<uint32_t, MAX_DEPTH> depth_dynamic_worker_lower_bound{};

        const uint32_t hard_worker_upper_bound = std::max<uint32_t>(1, (thread_count_ - 1) * 3 / 4);
        const uint32_t total_workers = thread_count_ - 1;

        depth_dynamic_worker_lower_bound.fill(1);

        while (!stop_requested_.load(std::memory_order_acquire) || !all_producers_done())
        {
            bool is_drain = all_producers_done();

            get_depth_worker_count_snapshot(depth_worker_count_snapshot);
            compute_ema_pressure(depth_worker_count_snapshot);

            update_dynamic_lower_bounds(depth_dynamic_worker_lower_bound,
                total_workers, hard_worker_upper_bound);

            // Minimum worker guarantee: ensure non-empty depths have at least 1 worker
            for (uint32_t d = 0; d < MAX_DEPTH; ++d)
            {
                uint32_t qsize = layer_queues_ptr_->get_queue(d)->size();
                uint32_t min_workers = depth_dynamic_worker_lower_bound[d];
                if (qsize > 0 && depth_worker_count_snapshot[d] < min_workers)
                {
                    uint32_t max_d = 0;
                    for (uint32_t dd = 1; dd < MAX_DEPTH; ++dd)
                    {
                        if (depth_worker_count_snapshot[dd] > depth_worker_count_snapshot[max_d])
                            max_d = dd;
                    }
                    if (depth_worker_count_snapshot[max_d] > min_workers)
                    {
                        for (uint32_t w = 0; w < total_workers; ++w)
                        {
                            uint32_t wd = worker_infos[w].depth.load(std::memory_order_acquire);
                            if (wd == max_d)
                            {
                                worker_commands_[w].store(d, std::memory_order_release); break;
                            }
                        }
                    }
                }
            }

            // Gradient-based migration with hysteresis

            uint32_t max_d = 0;
            uint32_t min_d = INVALID_DEPTH;
            double max_ema = depth_ema_pressure_[0];
            double min_ema = std::numeric_limits<double>::max();

            for (uint32_t d = 0; d < MAX_DEPTH; ++d)
            {
                if (depth_ema_pressure_[d] > max_ema)
                {
                    max_ema = depth_ema_pressure_[d];
                    max_d = d;
                }
                if (depth_worker_count_snapshot[d] > 0 && depth_ema_pressure_[d] < min_ema)
                {
                    min_ema = depth_ema_pressure_[d];
                    min_d = d;
                }
            }

            if (min_d != INVALID_DEPTH && (max_ema - min_ema) > HYSTERESIS_LOG_DELTA)
            {
                if (depth_worker_count_snapshot[max_d] < hard_worker_upper_bound)
                {
                    for (uint32_t w = 0; w < total_workers; ++w)
                    {
                        uint32_t wd = worker_infos[w].depth.load(std::memory_order_acquire);
                        if (wd == min_d)
                        {
                            worker_commands_[w].store(max_d, std::memory_order_release); break;
                        }
                    }
                }
            }

            // Drain mode: aggressively move workers from empty depths to non-empty ones
            if (is_drain) [[unlikely]]
            {

                auto move_one_empty_to_work = [&]() {
                    for (uint32_t d = 0; d < MAX_DEPTH; ++d)
                    {
                        uint32_t qsize = layer_queues_ptr_->get_queue(d)->size();
                        if (qsize == 0 && depth_worker_count_snapshot[d] > 0)
                        {
                            for (uint32_t td = 0; td < MAX_DEPTH; ++td)
                            {
                                if (td == d) continue;

                                uint32_t tqsize = layer_queues_ptr_->get_queue(td)->size();
                                if (tqsize > 0 && depth_worker_count_snapshot[td] < hard_worker_upper_bound)
                                {
                                    for (uint32_t w = 0; w < total_workers; ++w)
                                    {
                                        uint32_t wd = worker_infos[w].depth.load(std::memory_order_acquire);
                                        if (wd == d)
                                        {
                                            worker_commands_[w].store(td, std::memory_order_release); return;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    };

                move_one_empty_to_work();
            }


            uint32_t interval = is_drain ? DRAIN_INTERVAL_NS : SCHEDULE_INTERVAL_NS;
            std::this_thread::sleep_for(std::chrono::nanoseconds(interval));
        }
    }
};

#endif // SCHEDULING_THREAD_POOL_HEADER