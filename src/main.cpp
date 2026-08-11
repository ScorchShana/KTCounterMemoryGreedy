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

#include <chrono>
#include <iostream>
#include <cstdint>
#include <cstdlib>
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
#include <cmath>

uint32_t k_len;        // k-mer 的长度 (例如: 31, 41)
uint32_t n_thread;     // 允许使用的总线程数
uint64_t memory_limit; // 全局最大内存限制，单位为 GB
uint64_t estimated_file_size;    // 输入 FASTQ 文件的估计大小，单位为字节
std::vector<std::string> filenames;

void get_MAX_BLOOM_FILTER_CAPACITY()
{
#ifdef TEST_MODE
    std::cout << "Average quality score: " << avgQuality << std::endl;
#endif
    const double error_rate = std::pow(10.0, -(avgQuality - 33) * 0.1);
    const uint64_t quarter_memory_average_capacity = memory_limit * 1024ULL * 1024ULL * 1024ULL / 4 / sizeof(uint64_t) / (1ULL << (2 * ROOT_BASES));
    const double singleton_rate_cause_by_error = 1.0 - std::pow(1.0 - error_rate, k_len);
    const double error_factor = 0.5 + singleton_rate_cause_by_error;
    const auto corrected_memory_average_capacity = std::bit_ceil(static_cast<uint64_t>(static_cast<double>(quarter_memory_average_capacity) * error_factor));
    MAX_BLOOM_FILTER_CAPACITY = std::max<uint64_t>(MIN_BLOOM_FILTER_CAPACITY, corrected_memory_average_capacity);
}

uint64_t get_estimated_total_kmer(const uint64_t estimated_file_size)
{
    return estimated_file_size * 30 / 3 / std::max(k_len, 31U);
}

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

void calculate_bloom_filter_capacity(std::vector<std::atomic<uint32_t>>& prefix_counts, uint64_t estimated_file_size)
{

    const double error_rate = std::pow(10.0, -(avgQuality - 33) * 0.1);
    const uint64_t quarter_memory_average_capacity = memory_limit * 1024ULL * 1024ULL * 1024ULL / 4 / sizeof(uint64_t) / (1ULL << (2 * ROOT_BASES));
    const double singleton_rate_cause_by_error = 1.0 - std::pow(1.0 - error_rate, k_len);
    const double capacity_error_factor = std::min(0.5 + singleton_rate_cause_by_error, 1.0);

    const uint64_t estimated_total_kmers = get_estimated_total_kmer(estimated_file_size);

#ifdef TEST_MODE
    std::cout << "Estimated total k-mers: " << estimated_total_kmers << std::endl;
#endif

    uint64_t total_prefix_count = 0;

    uint64_t max_bloom_filter_capacity = 0;

#ifdef TEST_MODE
    uint64_t total_bloom_filter_size = 0;
#endif

    for (uint64_t i = 0; i < prefix_counts.size(); i++)
    {
        total_prefix_count += prefix_counts[i].load(std::memory_order_relaxed);
    }

    for (uint64_t i = 0; i < bloom_filter_capacity.size(); i++)
    {
        double prefix_ratio = static_cast<double>(prefix_counts[i].load(std::memory_order_relaxed)) / total_prefix_count;
        const uint64_t estimated_capacity = static_cast<uint64_t>(estimated_total_kmers * prefix_ratio * 4.81 * capacity_error_factor / 64);
        bloom_filter_capacity[i] = std::max(std::bit_ceil(estimated_capacity), MIN_BLOOM_FILTER_CAPACITY);
        bloom_filter_capacity[i] = std::min(bloom_filter_capacity[i], MAX_BLOOM_FILTER_CAPACITY);
        max_bloom_filter_capacity = std::max(max_bloom_filter_capacity, bloom_filter_capacity[i]);
#ifdef TEST_MODE
        total_bloom_filter_size += bloom_filter_capacity[i];
#endif
    }
#ifdef TEST_MODE
    std::cout << "Max Bloom Filter capacity: " << max_bloom_filter_capacity << std::endl;
    std::cout << "Total Bloom Filter size: " << total_bloom_filter_size * sizeof(uint64_t) / (1024 * 1024) << " MB" << std::endl;
#endif
}

void calculate_concurrent_map_capacity(
    const std::vector<std::atomic<uint32_t>>& prefix_counts)
{
    const uint64_t max_cap = kmer_concurrent_hash_map_capacity;
    const uint64_t min_cap = 1024ULL;
    const uint64_t mid_cap = std::max<uint64_t>(min_cap, max_cap / 2ULL);

    std::array<uint64_t, 256> sorted;
    for (size_t i = 0; i < prefix_counts.size(); ++i) {
        sorted[i] = prefix_counts[i].load(std::memory_order_relaxed);
    }
    std::sort(sorted.begin(), sorted.end());

    const uint64_t p30 = sorted[77];
    const uint64_t p85 = sorted[218];

    const double warm_low = static_cast<double>(p30);
    const double warm_high = static_cast<double>(p85);
    const double warm_range = warm_high - warm_low;

#ifdef TEST_MODE
    uint64_t cold_cnt = 0, warm_cnt = 0, hot_cnt = 0;
#endif

    for (size_t i = 0; i < concurrent_map_capacity.size(); ++i) {
        const uint64_t count = prefix_counts[i].load(std::memory_order_relaxed);
        uint64_t cap;

        if (count < p30) {
            cap = min_cap;
#ifdef TEST_MODE
            cold_cnt++;
#endif
        }
        else if (count < p85) {
            double t = (warm_range > 0.0)
                ? (static_cast<double>(count) - warm_low) / warm_range
                : 0.0;
            cap = min_cap + static_cast<uint64_t>((mid_cap - min_cap) * t);
#ifdef TEST_MODE
            warm_cnt++;
#endif
        }
        else {
            cap = max_cap;
#ifdef TEST_MODE
            hot_cnt++;
#endif
        }

        cap = std::max(min_cap, std::bit_ceil(cap));
        concurrent_map_capacity[i] = cap;
    }

#ifdef TEST_MODE
    std::cout << "--- Hash Map Capacity Allocation ---" << std::endl;
    std::cout << "  P30 (cold/warm): " << p30 << std::endl;
    std::cout << "  P85 (warm/hot):  " << p85 << std::endl;
    std::cout << "  COLD: " << cold_cnt << " -> cap=" << min_cap << std::endl;
    std::cout << "  WARM: " << warm_cnt << " -> linear " << min_cap << "->" << mid_cap << std::endl;
    std::cout << "  HOT:  " << hot_cnt << " -> cap=" << max_cap << std::endl;
#endif
}

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

template<uint32_t N>
int process_main()
{
    uint32_t gz_count = 0;
    for (const auto& f : filenames) {
        if (f.size() >= 3 && f.compare(f.size() - 3, 3, ".gz") == 0) gz_count++;
    }
    const uint32_t reader_num = (gz_count >= 2) ? 2 : 1;

    const uint32_t remaining = n_thread - reader_num - 1;  // -1: export writer
    const uint32_t parser_num = std::max(1U, remaining / 8);
    const uint32_t worker_budget = remaining - parser_num;

    const auto init_start = std::chrono::steady_clock::now();

    // 初始化层级队列，用于在树的不同深度间传递任务
    auto layer_queues = std::make_shared<LayerQueues<N>>();
    // 初始化解析器环形内存池，管理 Reader 读取后的碱基字符串数据块
    auto reader_parser_ring_pool = std::make_shared<RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>>(READER_PARSER_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    // 初始化分类器环形内存池，管理 Parser 线程处理后的 k-mer 数据块
    auto parser_classifier_ring_pool = std::make_shared<RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>>(PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    // 初始化导出用的环形内存池，管理低频 k-mer 的导出数据块
    auto export_ring_pool = std::make_shared<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    // 初始化全局并发内存池，用于节点分配、哈希表等
    auto pool = std::make_shared<ConcurrentMemoryPool>(memory_limit * 1024ULL * 1024ULL * 1024ULL);
    //
    auto global_classifier_task_queue = std::make_shared<MPMCRingQueue<content_type, GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY>>();

    // PreRead阶段
    FastqPreReader<N> pre_reader(filenames, k_len, FASTQ_FILE_CHUNK_SIZE, reader_parser_ring_pool.get());
    estimated_file_size = pre_reader.get_estimated_raw_fastq_file_size();
    auto pre_parser_thread = std::thread([&]
        {
            FastqPreParser<N> parser(k_len, reader_parser_ring_pool.get(), parser_classifier_ring_pool.get());
            parser.parse_and_push();
            parser_classifier_ring_pool->producer_set_finished(); });

    std::vector<std::atomic<uint32_t>> prefix_counts(1U << (2 * ROOT_BASES)); // 256 个前缀的计数器
    for (auto& v : prefix_counts)
    {
        v.store(0); // 或 v.store(init_value);
    }
    std::vector<std::thread> prefix_counter_threads;
    uint32_t prefix_counter_thread_num = std::min(std::max(1u, n_thread - 2), 8u); // 预留至少 1 个线程给前缀计数器
    for (uint32_t i = 0; i < prefix_counter_thread_num; ++i)
    {
        prefix_counter_threads.emplace_back([&]
            {
                FastqPrefixCounter<N> prefix_counter(k_len, parser_classifier_ring_pool.get());
                prefix_counter.count_prefixes();
                for (size_t j = 0; j < prefix_counts.size(); j++)
                {
                    prefix_counts[j].fetch_add(prefix_counter.prefix_counts[j], std::memory_order_relaxed);
                } });
    }
    pre_reader.pre_read();
    pre_parser_thread.join();
    for (auto& t : prefix_counter_threads)
    {
        t.join();
    }

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

    const uint32_t fewer_worker_num = std::max<uint32_t>(1U, worker_budget / (1.0 + TASK_CLASSIFIER_RATIO + 0.1));
    const uint32_t more_worker_num = std::max<uint32_t>(1U, worker_budget - fewer_worker_num);
    const bool high_quailty = (avgQuality >= 33 + 30);
    const uint32_t classifier_num = high_quailty ? fewer_worker_num : more_worker_num;
    const uint32_t tasker_num = high_quailty ? more_worker_num : fewer_worker_num;
    const uint32_t extra_drain_thread_count = n_thread - (tasker_num - 1);

    std::cout << "Thread split:" << std::endl;
    std::cout << "  parser threads: " << parser_num << std::endl;
    std::cout << "  classifier threads: " << classifier_num << std::endl;
    std::cout << "  task threads: " << tasker_num << std::endl;

    get_MAX_BLOOM_FILTER_CAPACITY();
    lpt(prefix_counts, classifier_num);
    calculate_bloom_filter_capacity(prefix_counts, estimated_file_size);
    calculate_concurrent_map_capacity(prefix_counts);

    get_numa_nodes();

    // 确保 Arena 已初始化，才能安全分配内存
    pool->init_arenas();
    // pool->perform_first_touch(n_thread);
    ConcurrentMap<N>::set_thread_num(std::max(1U, tasker_num - 1U) + n_thread);
    ConcurrentMap<N>::set_k_length(k_len);
    // 初始化 k-mer 字典树(KmerTree)的核心结构，整合前述多个组件
    auto tree = std::make_shared<KmerTree<N>>(k_len, pool.get(), layer_queues.get(), export_ring_pool.get());
    // 初始化布隆过滤器的MPSC队列
    std::vector<std::shared_ptr<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>> classifier_task_queues;
    for (uint32_t i = 0; i < classifier_num; i++)
    {
        classifier_task_queues.push_back(std::make_shared<MPSCRingQueue<content_type, CLASSIFIER_TASK_QUEUES_CAPACITY>>());
    }
    // 初始化并构建 Tasker 线程池，负责消费层级队列并将 k-mer 路由到深层节点 / 哈希表
    auto task_thread_pool = std::make_shared<SchedulerThreadPool<N>>(tasker_num, classifier_num, extra_drain_thread_count, tree.get(), layer_queues.get());
    // 初始化并且构建 Classifier 线程池，负责消费 Parser 线程产生的 k-mer 数据块，进行更精细的分类和路由
    auto classifier_thread_pool = std::make_shared<ClassifierThreadPool<N>>(k_len, tree.get(), parser_classifier_ring_pool.get(), classifier_task_queues, global_classifier_task_queue.get(), task_thread_pool.get(), pool.get(), classifier_num);
    // 初始化并构建 Parser 线程池，负责消费 FASTQ 读取器产生的数据，提取 k-mer 进行初步布隆过滤
    auto parser_thread_pool = std::make_shared<ParserThreadPool<N>>(k_len, classifier_task_queues, global_classifier_task_queue.get(), pool.get(), reader_parser_ring_pool.get(), parser_classifier_ring_pool.get(), parser_num);
    // 初始化导出写入器，用于将低频 k-mer 单线程安全落盘
    auto export_writer = std::make_shared<ExportWriter<N>>(k_len, export_ring_pool.get());

    // 正式计数阶段
    reader_parser_ring_pool->reset_producers(reader_num);
    parser_classifier_ring_pool->reset_producers(parser_num);
    // 初始化最终排干(drain)阶段所需的初始任务
    layer_queues->initialize_final_drain_queue(prefix_counts, tree->root_nodes);

    // 初始化 FASTQ 读取器，将大文件分块读取并送入 ring_pool 用作流水线起点
    ReaderThreadPool<N> reader_pool(filenames, k_len, FASTQ_FILE_CHUNK_SIZE, reader_parser_ring_pool.get());
    // FastqClassifier<N> classifier(k_len, parser_classifier_ring_pool.get(), tree.get());

    const auto init_end = std::chrono::steady_clock::now();

    const auto mid_start = std::chrono::steady_clock::now();

    const auto read_start = std::chrono::steady_clock::now();

    reader_pool.start();
    parser_thread_pool->start();
    classifier_thread_pool->start();
    task_thread_pool->start();
    export_writer->start();

    // classifier.classify_and_push();
    // task_thread_pool->mark_producer_done(); // 显式标记 Classifier 的生产者角色完成，通知 Scheduler 不再有新数据

    reader_pool.join();
    const auto read_end = std::chrono::steady_clock::now();

    // 阻塞等待 Parser 线程将所有文件块解析并生成 k-mer 完成
    parser_thread_pool->join();
    const auto parser_end = std::chrono::steady_clock::now();
    classifier_thread_pool->join();
    const auto classifier_end = std::chrono::steady_clock::now();

    // 标记所有的缓存已经发送完毕，向导出环形队列发送 nullptr 或退出标志
    const auto export_join_start = std::chrono::steady_clock::now();
    tree->mark_finish_export();
    // 阻塞等待导出器将所有低频 k-mer 都安全写入磁盘完成
    export_writer->join();
    const auto export_join_end = std::chrono::steady_clock::now();

    // 阻塞等待 Tasker 线程消费完所有的分发任务
    task_thread_pool->join();

    const auto task_end = std::chrono::steady_clock::now();

    const auto mid_end = std::chrono::steady_clock::now();

    const auto final_start = std::chrono::steady_clock::now();

    // Final drain 阶段：多线程并行遍历整个字典树，将在节点中暂存但未下发的 k-mers 全部合并到全局哈希表中
    // tree->final_drain_parallel(n_thread, std::max(1U, tasker_num - 1U));

    std::cout << "Total read k-mer count: " << parser_thread_pool->get_total_read_kmer() << std::endl;

    const auto final_end = std::chrono::steady_clock::now();

    const auto total_elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(final_end - init_start).count();

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

    std::cout << "Total elapsed us: " << total_elapsed_us << std::endl;

#ifdef TEST_MODE
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
    std::cout << "Classifier producer enqueue total spin time: " << classifier_thread_pool->producer_enqueue_spin_time.load() << std::endl;
    std::cout << "Classifier producer dequeue total spin time: " << classifier_thread_pool->producer_dequeue_spin_time.load() << std::endl;

    std::cout << "KmerTree total kmers added: " << tree->total_kmers_added.load() << std::endl;
    std::cout << "Kmer total kmers exported: " << classifier_thread_pool->total_kmers_exported.load() << std::endl;
    std::cout << "Kmer total kmers sent to tree: " << classifier_thread_pool->total_kmers_send_to_tree.load() << std::endl;
    std::cout << "equal : " << ((tree->total_kmers_added.load() + classifier_thread_pool->total_kmers_exported.load()) == parser_thread_pool->get_total_read_kmer()) << std::endl;

    for (uint32_t i = 0;i < MAX_DEPTH;i++)
    {
        std::cout << "Depth " << i << " queue size: " << layer_queues->get_queue(i)->size() << std::endl;
    }

#endif

    std::cout << "High frequency unique k-mer : " << sorted_kmer_count.load() << std::endl;


    return 0;
}

int main(int argc, char* argv[])
{

    if (argc < 6 || argc > 10)
    {
        std::cerr << "Usage: " << argv[0]
            << " <fastq_file> <k_len> <n_thread> <memory_limit_gb> <temp_dir> [map_capacity] [filter_min=2] [filter_max=4294967295] [count_max=255]" << std::endl;
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
        k_len = std::stoul(argv[2]);
        n_thread = std::stoul(argv[3]);
        memory_limit = std::stoull(argv[4]);
        temp_dir = argv[5];
        if (!temp_dir.empty() && temp_dir.back() != '/')
            temp_dir += '/';

        if (argc >= 7)
        {
            kmer_concurrent_hash_map_capacity = std::max<uint32_t>(1024, std::bit_ceil(std::stoul(argv[6])));
        }
        if (argc >= 8)
        {
            filter_min = std::stoul(argv[7]);
        }
        if (argc >= 9)
        {
            filter_max = std::stoul(argv[8]);
        }
        if (argc >= 10)
        {
            count_max = std::stoul(argv[9]);
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
        std::cout << "  k-mer length: " << k_len << std::endl;
        std::cout << "  Thread count: " << n_thread << std::endl;
        std::cout << "  Memory limit (GB): " << memory_limit << std::endl;
        std::cout << "  Map capacity: " << kmer_concurrent_hash_map_capacity << std::endl;
        std::cout << "  Filter min: " << filter_min << std::endl;
        std::cout << "  Filter max: " << filter_max << std::endl;
        std::cout << "  Count max: " << count_max << std::endl;
    }
    catch (const std::exception&)
    {
        std::cerr << "Usage: " << argv[0]
            << " <fastq_file> <k_len> <n_thread> <memory_limit_gb> <temp_dir> [map_capacity] [filter_min] [filter_max] [count_max]" << std::endl;
        return 1;
    }

    if (kmer_concurrent_hash_map_capacity <= 1 || kmer_concurrent_hash_map_capacity >= 16ULL * 1024 * 1024 || filter_max < filter_min || count_max == 0)
    {
        std::cerr << "Usage: " << argv[0]
            << " <fastq_file> <k_len> <n_thread> <memory_limit_gb> <temp_dir> [map_capacity] [filter_min] [filter_max] [count_max]" << std::endl;
        return 1;
    }

    MAX_BLOOM_FILTER_CAPACITY = std::bit_ceil(std::max<uint64_t>(MIN_BLOOM_FILTER_CAPACITY, memory_limit * 1024ULL * 1024ULL * 1024ULL / (4 * 8) / (1ULL << (2 * ROOT_BASES))));

    int fd = ::open((temp_dir + "infos.bin").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        std::cerr << "Open infos.bin failed : " << strerror(errno) << std::endl;
        return 1;
    }
    ::write(fd, &k_len, sizeof(k_len));
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

    if (k_len <= 32)
    {
        return process_main<1>();
    }
    else if (k_len <= 64)
    {
        return process_main<2>();
    }
    else if (k_len <= 128)
    {
        return process_main<4>();
    }
    else
    {
        std::cerr << "k_len must be <= 128" << std::endl;
        return 1;
    }
}
