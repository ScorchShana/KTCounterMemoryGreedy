#include "NewKmerTree.h"
#include "SchedulerThreadPool.h"
#include "ConcurrentMemoryPool.h"
#include "MPMCRingQueue.h"
#include "GetKmer.h"
#include "FastqReader.h"
#include "ParserThreadPool.h"
#include "ExportWriter.h"
#include "ClassifierThreadPool.h"
#include "FastqPrefixCounter.h"
#include "FastqPreReader.h"
#include "FastqParser.h"
#include "FastqPreParser.h"
#include "definition.h"
#include "MemoryBudget.h"

#include <charconv>
#include <cstring>
#include <stdexcept>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <memory>
#include <limits>
#include <vector>
#include <thread>
#include <algorithm>
#include <bit>
#include <queue>
#include <functional>
#include <bitset>
#include <atomic>
#include <algorithm>
#include <dirent.h>
#ifdef TEST_MODE
#include <sys/resource.h>
#endif

uint32_t n_thread;     // 允许使用的总线程数
uint64_t memory_limit; // 全局最大内存限制，单位为 GB
std::vector<std::string> filenames;
size_t pool_budget_bytes = 0;

void lpt(std::vector<std::atomic<uint32_t>>& prefix_counts, uint32_t classifier_num)
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

    struct ClassifierWorkload
    {
        uint32_t classifier_index;
        uint32_t count;

        bool operator<(const ClassifierWorkload& other) const
        {
            return count > other.count; // 负载较轻的前缀优先分配
        }
    };
    std::priority_queue<ClassifierWorkload, std::vector<ClassifierWorkload>> classifier_workloads;
    for (uint32_t i = 0; i < classifier_num; i++)
    {
        classifier_workloads.push({ i, 0 });
    }

    for (uint64_t i = 0; i < (1ULL << (2 * ROOT_BASES)); i++)
    {
        ClassifierWorkload work_load = classifier_workloads.top();
        prefix_owners[prefix_info_array[i].prefix] = work_load.classifier_index;
        classifier_workloads.pop();
        work_load.count += prefix_info_array[i].count;
        classifier_workloads.push(work_load);
    }
    while (!classifier_workloads.empty())
    {
        ClassifierWorkload work_load = classifier_workloads.top();
#ifdef TEST_MODE
        std::cout << "Classifier " << work_load.classifier_index << " total prefix count: " << work_load.count << std::endl;
#endif
        classifier_workloads.pop();
    }
}

// void calculate_concurrent_map_capacity(
//     const std::vector<std::atomic<uint32_t>>& prefix_counts)
// {
//     const uint64_t max_cap = kmer_concurrent_hash_map_capacity;
//     const uint64_t min_cap = 4096ULL;
//     const uint64_t mid_cap = std::max<uint64_t>(min_cap, max_cap / 2ULL);

//     std::array<uint64_t, 256> sorted;
//     for (size_t i = 0; i < prefix_counts.size(); ++i) {
//         sorted[i] = prefix_counts[i].load(std::memory_order_relaxed);
//     }
//     std::sort(sorted.begin(), sorted.end());

//     const uint64_t p30 = sorted[77];
//     const uint64_t p85 = sorted[218];

//     const double warm_low = static_cast<double>(p30);
//     const double warm_high = static_cast<double>(p85);
//     const double warm_range = warm_high - warm_low;

// #ifdef TEST_MODE
//     uint64_t cold_cnt = 0, warm_cnt = 0, hot_cnt = 0;
// #endif

//     for (size_t i = 0; i < concurrent_map_capacity.size(); ++i) {
//         const uint64_t count = prefix_counts[i].load(std::memory_order_relaxed);
//         uint64_t cap;

//         if (count < p30) {
//             cap = min_cap;
// #ifdef TEST_MODE
//             cold_cnt++;
// #endif
//         }
//         else if (count < p85) {
//             double t = (warm_range > 0.0)
//                 ? (static_cast<double>(count) - warm_low) / warm_range
//                 : 0.0;
//             cap = min_cap + static_cast<uint64_t>((mid_cap - min_cap) * t);
// #ifdef TEST_MODE
//             warm_cnt++;
// #endif
//         }
//         else {
//             cap = max_cap;
// #ifdef TEST_MODE
//             hot_cnt++;
// #endif
//         }

//         cap = std::max(min_cap, std::bit_ceil(cap));
//         concurrent_map_capacity[i] = cap;
//     }

// #ifdef TEST_MODE
//     std::cout << "--- Hash Map Capacity Allocation ---" << std::endl;
//     std::cout << "  P30 (cold/warm): " << p30 << std::endl;
//     std::cout << "  P85 (warm/hot):  " << p85 << std::endl;
//     std::cout << "  COLD: " << cold_cnt << " -> cap=" << min_cap << std::endl;
//     std::cout << "  WARM: " << warm_cnt << " -> linear " << min_cap << "->" << mid_cap << std::endl;
//     std::cout << "  HOT:  " << hot_cnt << " -> cap=" << max_cap << std::endl;
// #endif
// }

void get_numa_nodes()
{
#if HAS_LIBNUMA
    if (numa_available() == -1)
    {
        numa_nodes.push_back(0);
    }
    else
    {
        struct bitmask* mask = numa_all_nodes_ptr;
        // 遍历所有可能的节点编号（从0到最大可能节点）
        for (int i = 0; i <= numa_max_node(); i++) {
            if (numa_bitmask_isbitset(mask, i)) {
                numa_nodes.push_back(i);
#ifdef TEST_MODE
                std::cout << "NUMA node " << i << " is available." << std::endl;
#endif
            }
        }
    }
#else
    numa_nodes.push_back(0);
#endif

}

template<typename CountType>
int process_main()
{
    RunState runtime;
    ActiveRunScope active_run(runtime);
    uint32_t gz_count = 0;
    for (const auto& f : filenames) {
        if (f.size() >= 3 && f.compare(f.size() - 3, 3, ".gz") == 0) gz_count++;
    }
    const uint32_t reader_num = (gz_count >= 2) ? std::max(std::min(gz_count, n_thread / 20), 2U) : 1;

    const uint32_t preReadThreadsNum = std::max(1U, n_thread / 8);
    const uint32_t pre_reader_num = std::min<uint32_t>(filenames.size(), preReadThreadsNum);
    const uint32_t pre_parser_num = pre_reader_num * 4;
    const uint32_t pre_counter_num = pre_reader_num * 3;

    const uint32_t remaining = n_thread - reader_num - 1;  // -1: export writer
    const uint32_t parser_num = std::max(1U, remaining / 10);
    const uint32_t worker_budget = remaining - parser_num;

    const auto init_start = std::chrono::steady_clock::now();

    // 初始化层级队列，用于在树的不同深度间传递任务
    auto layer_queues = std::make_shared<LayerQueues<CountType>>();
    // 初始化解析器环形内存池，管理 Reader 读取后的碱基字符串数据块
    auto reader_parser_ring_pool = std::make_shared<RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>>(READER_PARSER_RING_MEMORY_POOL_BLOCK_SIZE, pre_reader_num);
    // 初始化分类器环形内存池，管理 Parser 线程处理后的 k-mer 数据块
    auto parser_classifier_ring_pool = std::make_shared<RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>>(PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, pre_parser_num);
    // 初始化导出用的环形内存池，管理低频 k-mer 的导出数据块
    auto export_ring_pool = std::make_shared<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    // 初始化全局并发内存池，用于节点分配、哈希表等
    auto pool = std::make_shared<ConcurrentMemoryPool>(pool_budget_bytes, &runtime);
    //
    auto global_classifier_task_queue = std::make_shared<MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>>();

    // PreRead阶段：多个 reader 按文件分组并行读取
    std::vector<std::vector<std::string>> pre_reader_files(pre_reader_num);
    for (size_t i = 0; i < filenames.size(); ++i)
        pre_reader_files[i % pre_reader_num].push_back(filenames[i]);

    std::vector<std::unique_ptr<FastqPreReader>> pre_readers;
    pre_readers.reserve(pre_reader_num);
    for (uint32_t i = 0; i < pre_reader_num; ++i)
        pre_readers.emplace_back(std::make_unique<FastqPreReader>(pre_reader_files[i], FASTQ_FILE_CHUNK_SIZE, reader_parser_ring_pool.get()));

    std::vector<std::thread> pre_reader_threads, pre_parser_threads, prefix_counter_threads;
    pre_reader_threads.reserve(pre_reader_num);
    pre_parser_threads.reserve(pre_parser_num);
    prefix_counter_threads.reserve(pre_counter_num);
    std::vector<std::atomic<uint32_t>> prefix_counts(1U << (2 * ROOT_BASES)); // 256 个前缀的计数器
    for (auto& v : prefix_counts)
    {
        v.store(0); // 或 v.store(init_value);
    }

    try {

    for (auto& pr : pre_readers)
        pre_reader_threads.emplace_back([&pr] { run_worker("pre-reader", [&] { pr->pre_read(); }); });


    for (uint32_t i = 0; i < pre_parser_num; ++i)
    {
        pre_parser_threads.emplace_back([&]
            {
                run_worker("pre-parser", [&] {
                FastqPreParser parser(reader_parser_ring_pool.get(), parser_classifier_ring_pool.get());
                parser.parse_and_push();
                parser_classifier_ring_pool->producer_set_finished(); }); });
    }

    for (uint32_t i = 0; i < pre_counter_num; ++i)
    {
        prefix_counter_threads.emplace_back([&]
            {
                run_worker("prefix counter", [&] {
                FastqPrefixCounter prefix_counter(parser_classifier_ring_pool.get());
                prefix_counter.count_prefixes();
                for (size_t j = 0; j < prefix_counts.size(); j++)
                {
                    prefix_counts[j].fetch_add(prefix_counter.prefix_counts[j], std::memory_order_relaxed);
                } }); });
    }

    } catch (...) {
        runtime.fail(std::current_exception(), "pre-stage startup");
        for (auto& thread : pre_reader_threads) if (thread.joinable()) thread.join();
        for (auto& thread : pre_parser_threads) if (thread.joinable()) thread.join();
        for (auto& thread : prefix_counter_threads) if (thread.joinable()) thread.join();
        throw;
    }

    for (auto& t : pre_reader_threads)
    {
        t.join();
    }

    // 聚合各 reader 的质量值，得到全局 avgQuality
    uint64_t quality_sum = 0, quality_count = 0;
    for (auto& pr : pre_readers)
    {
        quality_sum += pr->get_quality_sum();
        quality_count += pr->get_quality_count();
    }
    if (quality_count > 0)
        avgQuality = static_cast<uint8_t>(quality_sum / quality_count);

    for (auto& t : pre_parser_threads)
    {
        t.join();
    }
    for (auto& t : prefix_counter_threads)
    {
        t.join();
    }

    runtime.check();

    // 预读线程已结束，质量值和前缀统计已保存。
    pre_readers.clear();

    uint64_t average_count = 0;

    for (uint64_t i = 0; i < prefix_counts.size(); i++)
    {
        if (prefix_counts[i].load(std::memory_order_relaxed) < 1000)
        {
            prefix_counts[i].store(1000, std::memory_order_relaxed);
        }
        average_count += prefix_counts[i].load(std::memory_order_relaxed);
        // std::cout << std::format("Prefix {:0>8b}: count = {}", i, prefix_counts[i].load()) << std::endl;
    }

    average_count /= prefix_counts.size();
#ifdef TEST_MODE
    std::cout << "Average prefix count: " << average_count << std::endl;
#endif

    const uint32_t fewer_worker_num = std::max<uint32_t>(1U, worker_budget / (1.0 + TASK_CLASSIFIER_RATIO + 0.8));
    const uint32_t more_worker_num = std::max<uint32_t>(1U, worker_budget - fewer_worker_num);
    const bool high_quailty = (avgQuality >= 33 + 30);
    const uint32_t classifier_num = high_quailty ? fewer_worker_num : more_worker_num;
    if (classifier_num > 256) throw std::invalid_argument("too many Classifier owners");
    const uint32_t tasker_num = high_quailty ? more_worker_num : fewer_worker_num;
    const uint32_t extra_drain_thread_count = n_thread - (tasker_num - 1);

    std::cout << "Thread split:" << std::endl;
    std::cout << "  reader threads: " << reader_num << std::endl;
    std::cout << "  parser threads: " << parser_num << std::endl;
    std::cout << "  classifier threads: " << classifier_num << std::endl;
    std::cout << "  task threads: " << tasker_num << std::endl;

    lpt(prefix_counts, classifier_num);
    // calculate_concurrent_map_capacity(prefix_counts);

    get_numa_nodes();

    pool->init_arenas();
#ifdef TEST_MODE
    std::cout << "Bloom reserved bytes: " << 0
        << "\nDynamic Arena gap bytes: ";
    size_t dynamic_bytes = 0;
    for (int i = 0; i < pool->arena_count(); ++i) dynamic_bytes += pool->gap_bytes(i);
    std::cout << dynamic_bytes << std::endl;
    size_t metadata_bytes = 0;
    for (int i = 0; i < pool->arena_count(); ++i) {
        metadata_bytes += pool->segment_capacity(i) * sizeof(FirstSegmentDescriptor);
    }
    std::cout << "FIRST descriptor bytes: " << metadata_bytes
        << "\nFIRST active reserved bytes: " << 0 << std::endl;
#endif
    // pool->perform_first_touch(n_thread);

    // 初始化 k-mer 字典树(KmerTree)的核心结构，整合前述多个组件
    auto tree = std::make_shared<KmerTree<CountType>>(pool.get(), layer_queues.get(), export_ring_pool.get());
    // 初始化 Classifier 的 MPSC 任务队列
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> classifier_task_queues;
    for (uint32_t i = 0; i < classifier_num; i++)
    {
        classifier_task_queues.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
    }
    // 初始化并构建 Tasker 线程池，负责消费层级队列并将 k-mer 路由到深层节点 / 哈希表
    auto task_thread_pool = std::make_shared<SchedulerThreadPool<CountType>>(tasker_num, classifier_num, extra_drain_thread_count, tree.get(), layer_queues.get());
    // 初始化并且构建 Classifier 线程池，负责消费 Parser 线程产生的 k-mer 数据块，进行更精细的分类和路由
    auto classifier_thread_pool = std::make_shared<ClassifierThreadPool<CountType>>(tree.get(), parser_classifier_ring_pool.get(), classifier_task_queues, global_classifier_task_queue.get(), task_thread_pool.get(), pool.get(), classifier_num);
    // 初始化并构建 Parser 线程池，负责消费 FASTQ 读取器产生的数据，提取 k-mer 并按 owner 分组
    auto parser_thread_pool = std::make_shared<ParserThreadPool>(classifier_task_queues, global_classifier_task_queue.get(), pool.get(), parser_classifier_ring_pool.get(), parser_num);
    // 初始化导出写入器，用于将低频 k-mer 单线程安全落盘
    auto export_writer = std::make_shared<ExportWriter>(export_ring_pool.get(), temp_dir + "low.pending.bin", ExportWriter::OutputKind::Pending, pool.get(), nullptr, ExportWriter::ReclaimLifetime::UntilExplicitStop);

    // 正式计数阶段
    reader_parser_ring_pool->reset_producers(reader_num);
    parser_classifier_ring_pool->reset_producers(parser_num);
    // 预置最终导出任务，全部 Classifier 提交后才执行。
    layer_queues->initialize_final_drain_queue(prefix_counts, tree->root_nodes);

    // 初始化 FASTQ 读取器，将大文件分块读取并送入 ring_pool 用作流水线起点
    auto reader_pool = std::make_unique<ReaderThreadPool>(filenames, FASTQ_FILE_CHUNK_SIZE, reader_num, reader_parser_ring_pool.get());

#ifdef TEST_MODE
    const auto init_end = std::chrono::steady_clock::now();

    const auto mid_start = std::chrono::steady_clock::now();

    const auto read_start = std::chrono::steady_clock::now();
#endif

    export_writer->start();
    task_thread_pool->start();
    classifier_thread_pool->start();
    parser_thread_pool->start(*reader_parser_ring_pool);
    reader_pool->start();

    reader_pool->join();
    runtime.check();
#ifdef TEST_MODE
    const auto read_end = std::chrono::steady_clock::now();
#endif
    reader_pool.reset();

    // 阻塞等待 Parser 线程将所有文件块解析并生成 k-mer 完成
    parser_thread_pool->join();
    runtime.check();
#ifdef TEST_MODE
    const auto parser_end = std::chrono::steady_clock::now();
#endif
    reader_parser_ring_pool.reset();
    classifier_thread_pool->wait_for_first_pass();
#ifdef TEST_MODE
    const auto classifier_end = std::chrono::steady_clock::now();
#endif

    classifier_thread_pool->join();

    // Classifier 已提交全部本地 root；关闭空的 pending Writer 并清理本次分片。
#ifdef TEST_MODE
    const auto export_join_start = std::chrono::steady_clock::now();
#endif
    tree->mark_finish_export();
    export_writer->finish_reclaim();
    export_writer->join();
#ifdef TEST_MODE
    const auto export_join_end = std::chrono::steady_clock::now();
#endif
    for (uint64_t shard = 0; shard < export_writer->shard_count(); ++shard) {
        const std::string path = temp_dir + "low.pending.bin." + std::to_string(shard);
        if (::unlink(path.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(), "remove empty pending shard");
    }
    const int low_fd = ::open((temp_dir + "low.final.tmp").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (low_fd < 0)
        throw std::system_error(errno, std::generic_category(), "create empty final low output");
    if (::close(low_fd) != 0)
        throw std::system_error(errno, std::generic_category(), "close empty final low output");

    // 所有 Classifier 已完成本地 root flush，再进入 final drain。
    // Scheduler 排空剩余任务，直接导出 high.bin。
#ifdef TEST_MODE
    const auto final_start = std::chrono::steady_clock::now();
#endif
    task_thread_pool->join();
    runtime.check();
    if (::rename((temp_dir + "high.final.tmp").c_str(), (temp_dir + "high.bin").c_str()) != 0)
        throw std::system_error(errno, std::generic_category(), "publish high output");
#ifdef TEST_MODE
    const auto final_end = std::chrono::steady_clock::now();
#endif
    if (::rename((temp_dir + "low.final.tmp").c_str(), (temp_dir + "low.bin").c_str()) != 0)
        throw std::system_error(errno, std::generic_category(), "publish empty low output");

#ifdef TEST_MODE
    const auto task_end = std::chrono::steady_clock::now();

    const auto mid_end = std::chrono::steady_clock::now();
#endif

    std::cout << "Total read k-mer count: " << parser_thread_pool->get_total_read_kmer() << std::endl;
    std::cout << "Low frequency records scanned: " << 0 << std::endl;
    std::cout << "Low frequency records recovered: " << 0 << std::endl;
    std::cout << "FIRST drain singleton records: " << 0 << std::endl;
    std::cout << "Low frequency records kept: " << 0 << std::endl;
    std::cout << "Low frequency records discarded: " << 0 << std::endl;
    const auto total_end = std::chrono::steady_clock::now();

    const auto total_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(total_end - init_start).count();

#ifdef TEST_MODE
    const auto init_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(init_end - init_start).count();
    const auto read_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();
    const auto mid_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(mid_end - mid_start).count();
    const auto export_join_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(export_join_end - export_join_start).count();
    const auto final_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(final_end - final_start).count();
    const auto classifier_task_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(task_end - classifier_end).count();
    const auto parse_classifier_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(classifier_end - parser_end).count();
    const auto read_parse_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(parser_end - read_end).count();

    std::cout << "Init elapsed us: " << init_elapsed_us << std::endl;
    std::cout << "Read elapsed us: " << read_elapsed_us << std::endl;
    std::cout << "Mid elapsed us: " << mid_elapsed_us << std::endl;
    std::cout << "Between Read and Parse elapsed us: " << read_parse_elapsed_us << std::endl;
    std::cout << "Between Parse and Classifier elapsed us: " << parse_classifier_elapsed_us << std::endl;
    std::cout << "Between Classifier and Task elapsed us: " << classifier_task_elapsed_us << std::endl;
    std::cout << "Export join elapsed us: " << export_join_elapsed_us << std::endl;
    std::cout << "Final elapsed us: " << final_elapsed_us << std::endl;
#endif

    std::cout << "Pending encoded bytes written: " << export_writer->encoded_bytes_written() << '\n';
    std::cout << "Pending encoded bytes read: " << 0 << '\n';
    std::cout << "Total elapsed us: " << total_elapsed_us << std::endl;
#ifdef TEST_MODE
    std::cout << "RUN elapsed us: " << std::chrono::duration_cast<std::chrono::microseconds>(classifier_end - mid_start).count() << '\n';
    std::cout << "CLOSE_PENDING elapsed us: " << std::chrono::duration_cast<std::chrono::microseconds>(export_join_end - export_join_start).count() << '\n';
    std::cout << "FINAL_DRAIN elapsed us: " << std::chrono::duration_cast<std::chrono::microseconds>(final_end - final_start).count() << '\n';
    std::cout << "FIRST spill bytes: " << pool->spill_bytes.load() << '\n';
    std::cout << "FIRST returned bytes: " << pool->returned_bytes.load() << '\n';
    const uint64_t publications = pool->first_publications.load();
    std::cout << "FIRST publications: " << publications << '\n';
    std::cout << "FIRST shared valid bytes: " << pool->first_valid_records.load() * kmer_bytes() << '\n';
    std::cout << "FIRST shared padding bytes: " << pool->first_unused_bytes.load() << '\n';
    std::cout << "FIRST Spill claims: " << pool->spill_claims.load() << '\n';
    std::cout << "FIRST admission capacity/watermark: " << pool->first_rejected_capacity.load() << '\n';
    std::cout << "FIRST admission Tree reservation: " << pool->first_rejected_tree.load() << '\n';
    std::cout << "FIRST admission frozen: " << pool->first_rejected_frozen.load() << '\n';
    std::cout << "Allocation waits: " << pool->allocation_waits.load() << '\n';
    std::cout << "File Arena segments: " << pool->file_segment_count.load() << '\n';
    std::cout << "File Arena reservations: " << pool->file_reservations.load() << '\n';
    std::cout << "File Arena preallocations: " << pool->file_preallocations.load() << '\n';
    std::cout << "File Arena mappings: " << pool->file_mappings.load() << '\n';
    std::cout << "File Arena expansions: " << pool->file_expansions.load() << '\n';
    std::cout << "File Arena committed bytes: " << pool->file_committed_bytes.load() << '\n';
    const uint64_t waits = pool->allocation_waits.load();
    for (uint64_t percentile : {50, 95, 99}) {
        uint64_t cumulative = 0, bound = 0;
        for (size_t bucket = 0; bucket < pool->wait_latency_buckets.size() && waits; ++bucket) {
            cumulative += pool->wait_latency_buckets[bucket].load();
            if (cumulative >= (waits * percentile + 99) / 100) { bound = uint64_t{1} << bucket; break; }
        }
        std::cout << "Allocation wait p" << percentile << " upper ns: " << bound << '\n';
    }
    rusage usage{};
    if (::getrusage(RUSAGE_SELF, &usage) == 0) std::cout << "Process peak RSS KiB: " << usage.ru_maxrss << '\n';

    SpinLock::flush_spin_loops_for_current_thread();
    std::cout << "SpinLock spin_loops: " << SpinLock::spin_loops() << std::endl;
    std::cout << "Parser producer enqueue total spin time: " << parser_thread_pool->producer_enqueue_spin_time.load() << std::endl;
    std::cout << "Parser producer dequeue total spin time: " << parser_thread_pool->producer_dequeue_spin_time.load() << std::endl;
    std::cout << "Parser consumer dequeue total spin time: " << parser_thread_pool->consumer_dequeue_spin_time.load() << std::endl;
    std::cout << "Parser consumer enqueue total spin time: " << parser_thread_pool->consumer_enqueue_spin_time.load() << std::endl;
    std::cout << "Parser total parse cycles: " << parser_thread_pool->total_parse_cycles.load() << std::endl;
    std::cout << "Parser total flush cycles: " << parser_thread_pool->total_flush_cycles.load() << std::endl;
    std::cout << "Parser total queue wait cycles: " << parser_thread_pool->total_queue_wait_cycles.load() << std::endl;
    std::cout << "Parser total flush cycles account for " << (double)parser_thread_pool->total_flush_cycles.load() / parser_thread_pool->total_parse_cycles.load() * 100 << "% of parse cycles" << std::endl;
    std::cout << "Parser total queue wait cycles account for " << (double)parser_thread_pool->total_queue_wait_cycles.load() / parser_thread_pool->total_parse_cycles.load() * 100 << "% of parse cycles" << std::endl;
    std::cout << "Classifier consumer dequeue total spin time: " << classifier_thread_pool->consumer_dequeue_spin_time.load() << std::endl;
    std::cout << "Classifier consumer enqueue total spin time: " << classifier_thread_pool->consumer_enqueue_spin_time.load() << std::endl;
    std::cout << "Classifier enqueue to tree wait cycles: " << classifier_thread_pool->total_classifier_wait_cycles.load() << std::endl;
    std::cout << "Classifier total local tasks :" << classifier_thread_pool->total_local_tasks.load() << std::endl;
    std::cout << "Classifier total global tasks :" << classifier_thread_pool->total_global_tasks.load() << std::endl;
    std::cout << "Classifier total owner tasks :" << classifier_thread_pool->total_owner_tasks.load() << std::endl;

    std::cout << "KmerTree total kmers added: " << tree->total_kmers_added.load() << std::endl;
    std::cout << "Kmer total kmers sent to tree: " << classifier_thread_pool->total_kmers_send_to_tree.load() << std::endl;
    std::cout << "equal : " << (tree->total_kmers_added.load() == classifier_thread_pool->total_kmers_send_to_tree.load()
        && tree->total_kmers_added.load() == parser_thread_pool->get_total_read_kmer()) << std::endl;

    for (uint32_t i = 0;i < MAX_DEPTH;i++)
    {
        std::cout << "Depth " << i << " queue size: " << layer_queues->get_queue(i)->size() << std::endl;
    }

#endif

    std::cout << "High frequency unique k-mer : " << sorted_kmer_count.load() << std::endl;
    std::cout << "Unique k-mers filtered below filter_min: "
        << filtered_below_min_count.load(std::memory_order_relaxed) << std::endl;
    std::cout << "Unique k-mers filtered above filter_max: "
        << filtered_above_max_count.load(std::memory_order_relaxed) << std::endl;


    return 0;
}

int main(int argc, char* argv[])
{
    std::vector<char*> positional{argv[0]};
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--first-cache=auto" || argument == "--first-cache=off"
            || argument == "--first-reclaim=spill") {}
        else if (argument == "--first-reclaim=compact") {
            std::cerr << "--first-reclaim=compact is no longer supported; use --first-reclaim=spill\n";
            return 1;
        }
        else if (argument.starts_with("--")) { std::cerr << "Unknown option: " << argument << '\n'; return 1; }
        else positional.push_back(argv[i]);
    }
    argc = static_cast<int>(positional.size());
    argv = positional.data();

    if (argc < 6 || argc > 11)
    {
        std::cerr << "Usage: " << argv[0]
            << " <fastq_file> <k_length> <n_thread> <memory_limit_gb> <temp_dir> [map_capacity] [filter_min=2] [filter_max=4294967295] [count_max=255] [mapping_filename=mapping.tmp] [--first-cache=auto|off] [--first-reclaim=spill]" << std::endl;
        return 1;
    }


#ifdef ZLIBNG_VERSION
    std::cout << "Using zlib-ng version " << ZLIBNG_VERSION << std::endl;
#endif

    try
    {
        {
            std::string arg = argv[1];
            struct stat st;
            if (::stat(arg.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
            {
                //文件夹
                DIR* dir = ::opendir(arg.c_str());
                if (dir == nullptr)
                {
                    std::cerr << "Failed to open directory: " << arg << std::endl;
                    std::exit(-1);
                }
                struct dirent* entry;
                while ((entry = ::readdir(dir)) != nullptr)
                {
                    if (entry->d_name[0] == '.') continue;
                    filenames.push_back(arg + "/" + entry->d_name);
                }
                ::closedir(dir);
                std::sort(filenames.begin(), filenames.end());
            }
            else
            {
                // 逗号分隔的文件列表或单个文件
                size_t pos = 0;
                while ((pos = arg.find(',')) != std::string::npos)
                {
                    filenames.push_back(arg.substr(0, pos));
                    arg.erase(0, pos + 1);
                }
                if (!arg.empty())
                    filenames.push_back(arg);
            }
        }
        const auto parse_count = [](const char* text) -> uint32_t {
            uint32_t value = 0;
            const char* end = text + std::strlen(text);
            const auto result = std::from_chars(text, end, value);
            if (result.ec != std::errc{} || result.ptr != end)
                throw std::invalid_argument("invalid count parameter");
            return value;
        };
        k_length = parse_count(argv[2]);
        n_thread = std::stoul(argv[3]);
        const char* memory_end = argv[4] + std::strlen(argv[4]);
        const auto memory_result = std::from_chars(argv[4], memory_end, memory_limit);
        if (memory_result.ec != std::errc{} || memory_result.ptr != memory_end)
            throw std::invalid_argument("invalid memory limit");
        pool_budget_bytes = main_pool_budget(memory_limit);
        temp_dir = argv[5];
        if (!temp_dir.empty() && temp_dir.back() != '/')
            temp_dir += '/';

        if (argc >= 7)
        {
            concurrent_hash_map_max_capacity = std::max<uint64_t>(concurrent_hash_map_min_capacity, std::bit_ceil(std::stoul(argv[6])));
        }
        if (argc >= 8)
        {
            filter_min = parse_count(argv[7]);
        }
        if (argc >= 9)
        {
            filter_max = parse_count(argv[8]);
        }
        if (argc >= 10)
        {
            count_max = parse_count(argv[9]);
        }
        if (argc >= 11) {
            mapping_filename = argv[10];
            if (mapping_filename.empty()) throw std::invalid_argument("mapping filename must not be empty");
        }

        if (filter_min > filter_max)
        {
            std::cerr << "filter_min must be <= filter_max" << std::endl;
            return 1;
        }
        if (n_thread < 6)
        {
            std::cerr << "n_thread must be >= 6" << std::endl;
            std::exit(-1);
        }

        std::cout << "Input parameters:" << std::endl;
        std::cout << "  Fastq files (" << filenames.size() << "):" << std::endl;
        for (const auto& f : filenames)
            std::cout << "    " << f << std::endl;
        std::cout << "  k-mer length: " << k_length << std::endl;
        std::cout << "  Thread count: " << n_thread << std::endl;
        std::cout << "  Memory limit (GiB): " << memory_limit << std::endl;
        std::cout << "  Map capacity: " << concurrent_hash_map_max_capacity << std::endl;
        std::cout << "  Filter min: " << filter_min << std::endl;
        std::cout << "  Filter max: " << filter_max << std::endl;
        std::cout << "  Count max: " << count_max << std::endl;
    }
    catch (const std::exception&)
    {
        std::cerr << "Usage: " << argv[0]
            << " <fastq_file> <k_length> <n_thread> <memory_limit_gb> <temp_dir> [map_capacity] [filter_min=2] [filter_max=4294967295] [count_max=255] [--first-cache=auto|off] [--first-reclaim=spill]" << std::endl;
        return 1;
    }

    if (concurrent_hash_map_max_capacity <= 1 || concurrent_hash_map_min_capacity > concurrent_hash_map_max_capacity || count_max == 0)
    {
        std::cerr << "Usage: " << argv[0]
            << " <fastq_file> <k_length> <n_thread> <memory_limit_gb> <temp_dir> [map_capacity] [filter_min=2] [filter_max=4294967295] [count_max=255] [--first-cache=auto|off] [--first-reclaim=spill]" << std::endl;
        return 1;
    }

    if (k_length < 1 || k_length > MAX_K) {
        std::cerr << "k_length must be in 1..128" << std::endl;
        return 1;
    }

#ifdef TEST_MODE
    std::cout << "RingPool peak bytes: " << ring_pool_budget().peak_bytes << "\nMain pool budget bytes: " << pool_budget_bytes << std::endl;
#endif

    int fd = ::open((temp_dir + "infos.bin").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        std::cerr << "Open infos.bin failed : " << strerror(errno) << std::endl;
        return 1;
    }
    ::write(fd, &k_length, sizeof(k_length));
    ::write(fd, &count_max, sizeof(count_max));
    ::close(fd);

    if (count_max <= 0xFF)
    {
        count_max_bytes = 1;
    }
    else if (count_max <= 0xFFFF)
    {
        count_max_bytes = 2;

    }
    else if (count_max <= 0xFFFFFF)
    {
        count_max_bytes = 3;
    }
    else
    {
        count_max_bytes = 4;
    }

    try {
        if (count_max <= UINT8_MAX) return process_main<uint8_t>();
        if (count_max <= UINT16_MAX) return process_main<uint16_t>();
        return process_main<uint32_t>();
    }
    catch (const std::exception& error) {
        std::cerr << "Counting failed: " << error.what() << std::endl;
        return 1;
    }
}
