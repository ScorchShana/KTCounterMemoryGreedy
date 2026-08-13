#ifndef TREE_DEFINITION_HEADER
#define TREE_DEFINITION_HEADER

// #define TEST_MODE

#include "SpinLock.h"

#include <cstdint>
#include <cstddef>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <barrier>
#include <string>
#include <unordered_map>
#include <vector>

// NUMA 支持 - 自动检测 libnuma 是否可用
#if __has_include(<numa.h>)
#include <numa.h>
#include <numaif.h>
#define HAS_LIBNUMA 1
#else
#define HAS_LIBNUMA 0
#endif

constexpr uint64_t PAGE_SIZE = 4096;

constexpr uint64_t BASES_PER_U64T = 32;
constexpr uint64_t KMER_BLOCK_SIZE = 4096;         // 单个小 k-mer 块大小
inline constexpr uint32_t MAX_KMER_BLOCK_NUM = 16; // 单个节点能挂载的最大 k-mer 块数量

// 树的结构参数
constexpr uint32_t NODE_BASES = 2; // 普通节点的字典基数 (即碱基数, 2^(2*2) = 16 叉树)
constexpr uint32_t ROOT_BASES = 4; // 根节点的字典基数 (即碱基数, 2^(2*4) = 256 叉树)
constexpr uint32_t MAX_DEPTH = 4;  // 树的最大深度 (第 4 层将自动合并进挂载的哈希表)
constexpr uint32_t NODE_SIZE = 256;

constexpr size_t CACHE_LINE_SIZE = 64;

constexpr uint32_t MAX_K = 128;

// 内存池配置常量
constexpr uint64_t PAGES_PER_LARGE_BLOCK = 8; // 大块占用页数
constexpr uint64_t THREAD_CACHE_LIMIT = 64;   // 线程缓存上限
constexpr uint64_t BATCH_FETCH_SIZE = 16;     // 批量获取数量
constexpr uint64_t BATCH_RETURN_SIZE = 32;    // 批量归还数量

static_assert(KMER_BLOCK_SIZE % 4096 == 0, "KMER_BLOCK_SIZE must be multiple of 4096 bytes");

// Reader与Parser之间的RingMemoryPool配置常量
constexpr uint64_t READER_PARSER_RING_MEMORY_POOL_CAPACITY = 1ULL << 12;     // 环形内存池容量（块数），必须为2的幂
constexpr uint64_t READER_PARSER_RING_MEMORY_POOL_BLOCK_SIZE = 32ULL * 1024; // 环形内存池块大小（字节）

// Parser与Classifier之间的RingMemoryPool配置常量
constexpr uint64_t PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY = 1ULL << 11;     // 环形内存池容量（块数），必须为2的幂
constexpr uint64_t PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE = 64ULL * 1024; // 环形内存池块大小（字节）

// Classifier 线程的任务队列配置常量
constexpr uint64_t GLOBAL_CLASSIFIER_TASK_QUEUE_CAPACITY = 8ULL << 10; // 全局分类器任务队列容量
constexpr uint64_t CLASSIFIER_TASK_QUEUES_CAPACITY = 64;

// 写入文件部分的RingMemoryPool配置常量
constexpr uint64_t EXPORT_RING_MEMORY_POOL_CAPACITY = 4ULL << 10;     // 导出环形内存池容量（块数），必须为2的幂
constexpr uint64_t EXPORT_RING_MEMORY_POOL_BLOCK_SIZE = 256ULL * 1024; // 导出环形内存池块大小（字节）

// RingMemoryPool 生产者队列的内容
struct content_type
{
    char* data;
    uint64_t length;
};

// 解析线程配置常量
constexpr double TASK_CLASSIFIER_RATIO = 1; // 分类与任务线程比例

// 任务线程配置常量
constexpr uint32_t LOCAL_STACK_SIZE = 64;
constexpr uint64_t KMER_BIN_SIZE = 2048;
// constexpr uint32_t MAP_SIZE_FLUSH_INTERVAL = 1024; // 每线程累计新增 key 达到该阈值后批量 flush 到 map_size，降低原子争用

// MPMP环状队列配置常量
constexpr uint32_t TASK_QUEUE_CAPACITY = 16U * 1024;

// task线程enqueue尝试次数
constexpr uint32_t TASK_ENQUEUE_RETRY_LIMIT = 1ULL << 7;

// Final drain 导出配置
constexpr uint64_t DRAIN_EXPORT_BUFFER_SIZE = 2 * 1024 * 1024; // final drain 导出缓冲区大小（字节）

// k-mer 计数过滤区间（闭区间）
inline uint32_t filter_min = 2;
inline uint32_t filter_max = std::numeric_limits<uint32_t>::max();
inline uint32_t count_max = 255;
inline uint32_t count_max_bytes = 1;

// KmerTree的哈希表大小
inline uint32_t kmer_concurrent_hash_map_capacity = 1024;

// FastqReader配置常量
constexpr uint64_t FASTQ_FILE_CHUNK_SIZE = 2 * 1024 * 1024; // FASTQ 文件块大小（字节）
constexpr uint64_t KMER_BATCH_PREFIX_BASES = ROOT_BASES;    // 根节点使用的碱基数

// FastqParser配置常量
constexpr uint64_t KMER_BATCH_SIZE = 1024; // KmerBatch 的总大小（字节），包括计数和前缀

// ExportWriter配置常量
constexpr uint64_t EXPORT_FILES_SIZE = 1ULL << (2 * ROOT_BASES); // 最大同时打开文件数量
constexpr uint64_t EXPORT_ROOT_BUFFER_SIZE = 512 * 1024;         // 每个根节点的导出缓冲区大小（字节）

// FinalDrain 环形内存池配置
constexpr uint64_t FINAL_DRAIN_RING_POOL_CAPACITY = 1ULL << 10;     // 1024 块
constexpr uint64_t FINAL_DRAIN_RING_POOL_BLOCK_SIZE = 128ULL * 1024; // 256KB/块

static_assert(KMER_BIN_SIZE < KMER_BLOCK_SIZE, "KMER_BIN_SIZE must be less than KMER_BLOCK_SIZE");
static_assert(KMER_BIN_SIZE < KMER_BLOCK_SIZE, "KMER_BIN_SIZE must be less than KMER_BLOCK_SIZE");

// 前置声明模板

template <uint32_t N>
struct kmer;

template <uint32_t N>
struct kmer_block;

template <uint32_t N>
struct node;

template <uint32_t N>
union super_block_ptr;

// 线程内部缓存桶
template <uint32_t N>
struct ThreadLocalBin
{
    std::array<kmer<N>, KMER_BIN_SIZE / sizeof(kmer<N>)> buffer;
    uint64_t count = 0;
};

// 线程任务结构体
template <uint32_t N>
struct alignas(64) Task
{
    // constexpr static uint32_t MAX_KMER_BLOCK_NUM = (NODE_SIZE - CACHE_LINE_SIZE - sizeof(uint64_t)
    // - sizeof(lock_type) - sizeof(kmer_block<N> *)) / sizeof(kmer_block<N> *);

    node<N>* current_node = nullptr; // 需要下方的节点
    uint64_t depth;                  // 需要下方的节点的深度，与LayerQueues的层级对应
    uint64_t count;
    std::array<kmer_block<N>*, MAX_KMER_BLOCK_NUM> kmer_blocks{};

    Task() = default;
    Task& operator=(const Task<N>& a) = default;
    Task(const Task<N>& a) = default;
};

template <uint32_t N>
struct KmerBatch
{
    static constexpr uint64_t KMER_BATCH_CAPACITY = (KMER_BATCH_SIZE - sizeof(uint64_t) * 2) / sizeof(kmer<N>);
    uint64_t k_mers_count = 0;
    uint64_t prefix; // 用于路由的前缀信息
    std::array<kmer<N>, KMER_BATCH_CAPACITY> k_mers;
};

template <uint32_t N>
struct alignas(PAGE_SIZE) ExportBlock
{
    std::array<kmer<N>, EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / sizeof(kmer<N>)> k_mers;
};

template <uint32_t N>
struct concurrent_node
{
    static_assert(std::is_trivially_copyable_v<kmer<N>>, "kmer<N> must be trivially copyable");
    kmer<N> k_mer;
    concurrent_node* next{ nullptr };
    std::atomic<uint32_t> count{ 0 };
};

// FinalDrain 导出使用的结构体
template <uint32_t N>
struct ExportRecord
{
    kmer<N> key;
    uint32_t count;
};

// 写入k-mer计数
alignas(CACHE_LINE_SIZE) inline std::atomic<uint64_t> sorted_kmer_count{ 0 };

// 临时文件目录
inline std::string temp_dir = "./tmp/";

// 布隆过滤器最大最小数目
constexpr uint64_t MIN_BLOOM_FILTER_CAPACITY = 1ULL << 12; // 最低Bloom Filter容量，单位为元素数量
inline uint64_t MAX_BLOOM_FILTER_CAPACITY = 1ULL << 20;      // 最大Bloom Filter容量，单位为元素数量

// 布隆过滤器的容量
inline std::array<uint64_t, 1U << (2 * ROOT_BASES)> bloom_filter_capacity;

// prefix 归属的 classifier 线程索引
inline std::array<uint8_t, 1U << (2 * ROOT_BASES)> prefix_owners;

inline std::array<void*, 1U << (2 * ROOT_BASES)> global_bloom_filter{};

// prefix 对应的哈希表容量
inline std::array<uint64_t, 1U << (2 * ROOT_BASES)> concurrent_map_capacity;

inline uint8_t avgQuality = 0;

// numa 节点
inline std::vector<int> numa_nodes;

#endif // TREE_DEFINITION_HEADER