#ifndef CONCURRENT_MEMORY_POOL_HEADER
#define CONCURRENT_MEMORY_POOL_HEADER

// 并发内存池 - NUMA感知设计
// 固定块模型（4KB），线程本地缓存，跨线程释放支持

#include "definition.h"
#include "SpinBackoff.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>

// 系统头文件
#include <sys/mman.h>
#include <sched.h>
#include <unistd.h>
#include <errno.h>



//==============================================================================
// 可配置常量
//==============================================================================

// 最大 NUMA 节点数
constexpr int MAX_NUMA_NODES = 16;

// 固定块大小（4KB）
constexpr size_t BLOCK_SIZE = 4096;

// 从 Arena 批量获取的块数
constexpr size_t BATCH_SIZE = 64;

// 线程本地栈最大缓存块数
constexpr size_t TLS_CAPACITY = 128;

// 每个远程链表的最大长度
constexpr size_t REMOTE_LIST_CAPACITY = 32;

// 大页大小（2MB）
constexpr size_t HUGE_PAGE_SIZE = 2ULL * 1024 * 1024;

//==============================================================================
// 辅助工具
//==============================================================================

// 向上对齐到指定边界
constexpr size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

// 检查是否对齐
constexpr bool is_aligned(size_t value, size_t alignment)
{
    return (value & (alignment - 1)) == 0;
}

//==============================================================================
// FreeBlock - 侵入式链表节点
//==============================================================================

// 空闲块头部结构，用于链接空闲块
// 当块被分配时，此空间可供用户使用
struct FreeBlock
{
    FreeBlock* next;
};

//==============================================================================
// Arena - NUMA 节点内存管理
//==============================================================================

// 每个 NUMA 节点一个 Arena，管理本节点的内存块
// 使用 alignas 避免伪共享
struct alignas(CACHE_LINE_SIZE) Arena
{
    void* start_addr; // 本 Arena 的起始地址
    void* end_addr;   // 本 Arena 的结束地址（不包含）
    std::atomic<char*> bump_cursor;
    FreeBlock* central_free_list; // 中央空闲链表
    std::mutex mutex;             // 保护中央空闲链表的互斥锁

    Arena() : start_addr(nullptr), end_addr(nullptr), bump_cursor(nullptr), central_free_list(nullptr) {}

    // 检查指针是否属于此 Arena
    bool contains(void* ptr) const
    {
        return ptr >= start_addr && ptr < end_addr;
    }

    // 获取此 Arena 管理的块数
    size_t block_count() const
    {
        return static_cast<size_t>(
            static_cast<char*>(end_addr) - static_cast<char*>(start_addr)) /
            BLOCK_SIZE;
    }
};

//==============================================================================
// RemoteListHead - 远程链表头
//==============================================================================

// 用于 ThreadLocalCache 中的远程链表。
// 尾部填充至缓存行大小，避免相邻 arena 的 remote list 产生伪共享。
struct alignas(CACHE_LINE_SIZE) RemoteListHead
{
    FreeBlock* head;
    size_t count;
    char padding[CACHE_LINE_SIZE - sizeof(FreeBlock*) - sizeof(size_t)];

    RemoteListHead() : head(nullptr), count(0)
    {
        static_assert(sizeof(RemoteListHead) == CACHE_LINE_SIZE,
            "RemoteListHead must occupy exactly one cache line");
    }
};

//==============================================================================
// ThreadLocalCache - 线程本地缓存
//==============================================================================

struct ThreadLocalCache
{
    // 本地空闲栈（使用侵入式链表实现栈）
    FreeBlock* local_free_stack;
    size_t local_free_count;

    // 远程链表数组，每个 Arena 一个
    // 用于暂存属于其他 Arena 的块
    alignas(CACHE_LINE_SIZE) RemoteListHead remote_lists[MAX_NUMA_NODES];

    // 指向本地 Arena 的指针
    Arena* local_arena;
    int local_arena_index;

    // 所属的 ConcurrentMemoryPool 指针（用于归还内存）
    class ConcurrentMemoryPool* pool;

    ThreadLocalCache()
        : local_free_stack(nullptr), local_free_count(0), local_arena(nullptr), local_arena_index(-1), pool(nullptr) {
    }

    // 析构时归还所有缓存
    ~ThreadLocalCache();
};

//==============================================================================
// MemoryPool - 顶层内存池类
//==============================================================================

class ConcurrentMemoryPool
{
public:
    // 构造函数
    // total_bytes: 总字节数（内部会按 BLOCK_SIZE 对齐）
    explicit ConcurrentMemoryPool(size_t total_bytes);

    // 禁止拷贝与赋值
    ConcurrentMemoryPool(const ConcurrentMemoryPool&) = delete;
    ConcurrentMemoryPool& operator=(const ConcurrentMemoryPool&) = delete;

    // 析构函数，释放所有 mmap 内存
    ~ConcurrentMemoryPool();

    // 分配一个 4KB 块，失败时调用 std::exit(-1)
    void* allocate();

    // 分配可变大小的块，失败时调用 std::exit(-1)
    void* allocate_large(size_t bytes);

    // 释放一个之前分配的块
    void deallocate(void* ptr);

    // 在 init_arenas 之前分配大块内存（2MB 对齐），不需要回收
    void* allocate_before_init_arenas(uint64_t bytes);

    // 在所有大块分配完成后，用剩余内存初始化各个 Arena
    void init_arenas();

    // 进行first-touch初始化，确保每个NUMA节点的内存被访问
    void perform_first_touch(size_t num_threads = 0);

    // 获取当前线程所属 Arena 的索引
    int get_current_arena_index() const;

    // 获取线程本地缓存
    static ThreadLocalCache& get_thread_local_cache();

private:
    //--------------------------------------------------------------------------
    // 内部辅助函数
    //--------------------------------------------------------------------------

    // 初始化 NUMA 信息
    void init_numa_info();

    // mmap 总内存（尝试大页），不设置 Arena
    void mmap_total_memory(size_t total_bytes);

    // 从 Arena 批量获取块，返回实际获取块数（0 表示获取失败）
    size_t batch_allocate_from_arena(Arena& arena, FreeBlock** out_head, FreeBlock** out_tail, size_t count);

    char* bump_allocate_from_arena(Arena& arena, size_t bytes);

    FreeBlock* batch_allocate_from_bump(Arena& arena, size_t& out_count, FreeBlock** out_tail);

    // 批量归还块到 Arena
    void batch_deallocate_to_arena(Arena& arena, FreeBlock* head, FreeBlock* tail, size_t count);

    // 根据地址查找 Arena 索引
    int find_arena_index(void* ptr) const;

    // 获取当前线程应该使用的 Arena 索引
    int get_thread_arena_index() const;

    // 构建按 NUMA 距离排序的候选 Arena 索引列表（用于本地 Arena 耗尽时窃取）
    inline void build_numa_distance_order();

    // 应用 mbind 将每个 Arena 的内存绑定到对应 NUMA 节点
    inline void apply_arena_mbind();

    //----------------------------------------------------------------------
    // 成员变量
    //--------------------------------------------------------------------------

    // Arena 数组
    alignas(CACHE_LINE_SIZE) Arena arenas_[MAX_NUMA_NODES];

    // 实际使用的 Arena 数量
    int num_arenas_;

    // mmap 原始起始地址（用于 munmap）
    void* mmap_base_;

    // mmap 原始大小（用于 munmap）
    size_t mmap_size_;

    // 可用内存区域起始地址（2MB 对齐）
    void* memory_start_;

    // 可用内存区域总大小（2MB 对齐）
    size_t memory_size_;

    // 实际使用的页大小（4KB 或 2MB）
    size_t page_size_;

    // 总块数（Arena 初始化后计算）
    size_t total_blocks_;

    // NUMA 是否可用
    bool numa_available_;

    // 每个 NUMA 节点的 CPU 数量
    int cpus_per_node_[MAX_NUMA_NODES];

    // 总可用 CPU 数
    int total_cpus_;

    // 按 NUMA 距离排序的候选 Arena 索引，[local_index][0] 为本地，后续按距离递增
    int numa_distance_order_[MAX_NUMA_NODES][MAX_NUMA_NODES];

    // init_arenas 前已分配的大块偏移
    alignas(CACHE_LINE_SIZE) std::atomic<size_t> pre_arena_offset_{ 0 };

    // 保护 pre_arena_offset_ 的互斥锁
    std::mutex pre_arena_mutex_;

    // Arena 是否已初始化
    alignas(CACHE_LINE_SIZE) std::atomic<bool> arenas_initialized_{ false };

    // 线程本地缓存
    static inline thread_local ThreadLocalCache tls_cache_;

    // 友元类
    friend struct ThreadLocalCache;
};

//==============================================================================
// ThreadLocalCache 析构函数实现
//==============================================================================

inline ThreadLocalCache::~ThreadLocalCache()
{
    if (!pool)
        return;

    // 归还本地空闲栈中的所有块
    while (local_free_stack)
    {
        FreeBlock* block = local_free_stack;
        local_free_stack = block->next;
        local_free_count--;

        // 直接归还到对应的 Arena
        int arena_idx = pool->find_arena_index(block);
        if (arena_idx >= 0 && arena_idx < pool->num_arenas_)
        {
            Arena& arena = pool->arenas_[arena_idx];
            std::lock_guard<std::mutex> lock(arena.mutex);
            block->next = arena.central_free_list;
            arena.central_free_list = block;
        }
    }

    // 归还所有远程链表中的块
    for (int i = 0; i < MAX_NUMA_NODES; ++i)
    {
        RemoteListHead& rl = remote_lists[i];
        if (rl.head)
        {
            if (i < pool->num_arenas_)
            {
                Arena& arena = pool->arenas_[i];
                std::lock_guard<std::mutex> lock(arena.mutex);
                // 将整个链表挂到中央空闲链表
                FreeBlock* tail = rl.head;
                while (tail->next)
                {
                    tail = tail->next;
                }
                tail->next = arena.central_free_list;
                arena.central_free_list = rl.head;
            }
            rl.head = nullptr;
            rl.count = 0;
        }
    }

    pool = nullptr;
}

//==============================================================================
// MemoryPool 实现
//==============================================================================

inline ConcurrentMemoryPool::ConcurrentMemoryPool(size_t total_bytes)
    : num_arenas_(1), mmap_base_(nullptr), mmap_size_(0), memory_start_(nullptr), memory_size_(0),
    page_size_(4096), total_blocks_(0), numa_available_(false), total_cpus_(1)
{
    // 初始化数组
    std::memset(cpus_per_node_, 0, sizeof(cpus_per_node_));
    std::memset(numa_distance_order_, 0, sizeof(numa_distance_order_));

    // 初始化 NUMA 信息
    init_numa_info();

    // 初始化距离顺序：默认本地优先，其余按索引
    build_numa_distance_order();

    if (total_bytes == 0)
    {
        std::cerr << "total_bytes must be > 0" << std::endl;
        std::exit(-1);
    }

    // mmap 总内存，不做 first-touch，也不切分 Arena
    mmap_total_memory(total_bytes);
}

inline ConcurrentMemoryPool::~ConcurrentMemoryPool()
{
    // 清理当前线程的 TLS 缓存（如果有的话）
    // 注意：TLS 析构会在 ConcurrentMemoryPool 析构之后发生，所以需要先清理
    ThreadLocalCache& tls = tls_cache_;
    if (tls.pool == this)
    {
        // 清空本地空闲栈，不归还到 Arena（因为 Arena 即将销毁）
        tls.local_free_stack = nullptr;
        tls.local_free_count = 0;
        tls.local_arena = nullptr;
        tls.local_arena_index = -1;

        // 清空远程链表
        for (int i = 0; i < MAX_NUMA_NODES; ++i)
        {
            tls.remote_lists[i].head = nullptr;
            tls.remote_lists[i].count = 0;
        }

        tls.pool = nullptr;
    }

    if (mmap_base_)
    {
        munmap(mmap_base_, mmap_size_);
        mmap_base_ = nullptr;
    }
}

inline void ConcurrentMemoryPool::init_numa_info()
{
#if HAS_LIBNUMA
    if (numa_available() == -1)
    {
        numa_available_ = false;
        num_arenas_ = 1;
        cpus_per_node_[0] = 1;
        total_cpus_ = 1;
        return;
    }

    numa_available_ = true;
    num_arenas_ = numa_num_configured_nodes();
    if (num_arenas_ > MAX_NUMA_NODES)
    {
        num_arenas_ = MAX_NUMA_NODES;
    }

    // 获取当前进程的 CPU 亲和性
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == 0)
    {
        // 统计每个 NUMA 节点的可用 CPU 数
        total_cpus_ = 0;
        for (int node = 0; node < num_arenas_; ++node)
        {
            int count = 0;
            for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            {
                if (CPU_ISSET(cpu, &mask))
                {
                    if (numa_node_of_cpu(cpu) == node)
                    {
                        ++count;
                    }
                }
            }
            cpus_per_node_[node] = count;
            total_cpus_ += count;
#ifdef TEST_MODE
            std::cout << "NUMA node " << node << ": " << count << " CPUs" << std::endl;
#endif
        }

        // 如果没有找到 CPU，使用默认值
        if (total_cpus_ == 0)
        {
            total_cpus_ = 1;
            cpus_per_node_[0] = 1;
        }
    }
    else
    {
        // 无法获取亲和性，使用默认值
        total_cpus_ = 1;
        cpus_per_node_[0] = 1;
    }
#else
    numa_available_ = false;
    num_arenas_ = 1;
    cpus_per_node_[0] = 1;
    total_cpus_ = 1;
#endif
}

inline void ConcurrentMemoryPool::build_numa_distance_order()
{
    // 初始化：本地节点优先，其余按 NUMA 距离排序（若可用），否则按索引顺序。
    for (int local = 0; local < MAX_NUMA_NODES; ++local)
    {
        // 默认顺序：本地优先，然后按索引
        for (int j = 0; j < MAX_NUMA_NODES; ++j)
        {
            numa_distance_order_[local][j] = (local + j) % MAX_NUMA_NODES;
        }

        if (local >= num_arenas_)
            continue;

#if HAS_LIBNUMA
        if (numa_available_)
        {
            // 计算 local 到所有其他节点的距离
            struct DistIndex
            {
                int distance;
                int index;
            };
            DistIndex dists[MAX_NUMA_NODES];
            for (int node = 0; node < num_arenas_; ++node)
            {
                dists[node].distance = numa_distance(local, node);
                dists[node].index = node;
            }
            // 按距离升序排序（距离相同则索引小的在前）
            std::sort(dists, dists + num_arenas_,
                [](const DistIndex& a, const DistIndex& b) {
                    if (a.distance != b.distance)
                        return a.distance < b.distance;
                    return a.index < b.index;
                });
            for (int j = 0; j < num_arenas_; ++j)
            {
                numa_distance_order_[local][j] = dists[j].index;
            }
            // 剩余位置保持默认，不会被访问
        }
#endif
    }
}

inline void ConcurrentMemoryPool::mmap_total_memory(size_t total_bytes)
{
    // 总大小向上对齐到 2MB，并多申请 2MB 用于地址对齐
    size_t aligned_size = align_up(total_bytes, HUGE_PAGE_SIZE);
    size_t request_size = aligned_size + HUGE_PAGE_SIZE;

    // 尝试使用大页
    void* addr = mmap(nullptr, request_size,
        PROT_READ | PROT_WRITE,
        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
        -1, 0);

    if (addr != MAP_FAILED)
    {
        mmap_base_ = addr;
        mmap_size_ = request_size;
        page_size_ = HUGE_PAGE_SIZE;
    }
    else
    {
        // 大页分配失败，回退到普通页
        addr = mmap(nullptr, request_size,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1, 0);

        if (addr == MAP_FAILED)
        {
            std::cerr << "std::bad_alloc" << std::endl;
            std::exit(-1);
        }

        mmap_base_ = addr;
        mmap_size_ = request_size;
        page_size_ = static_cast<size_t>(sysconf(_SC_PAGESIZE));

        // 尝试建议使用透明大页
        madvise(mmap_base_, mmap_size_, MADV_HUGEPAGE);
    }

    // 确保 memory_start_ 是 2MB 对齐的
    uintptr_t raw = reinterpret_cast<uintptr_t>(mmap_base_);
    uintptr_t aligned = align_up(raw, HUGE_PAGE_SIZE);
    memory_start_ = reinterpret_cast<void*>(aligned);
    memory_size_ = aligned_size;

    // 对齐检查
    if (!is_aligned(reinterpret_cast<uintptr_t>(memory_start_), HUGE_PAGE_SIZE))
    {
        std::cerr << "mmap memory_start_ is not 2MB aligned" << std::endl;
        std::exit(-1);
    }
}

inline void ConcurrentMemoryPool::apply_arena_mbind()
{
#if HAS_LIBNUMA
    if (!numa_available_ || num_arenas_ <= 1)
        return;

    // nodemask：一个 unsigned long[2] = 128 bit，足以覆盖 MAX_NUMA_NODES=16。
    // 不依赖 numaif.h 的 nodemask_t 类型细节，避免可移植性问题。
    for (int i = 0; i < num_arenas_; ++i)
    {
        void* start = arenas_[i].start_addr;
        size_t len = static_cast<size_t>(
            static_cast<char*>(arenas_[i].end_addr) -
            static_cast<char*>(arenas_[i].start_addr));
        if (!start || len == 0)
            continue;

        unsigned long nodemask[2] = { (1UL << i), 0 };
        int maxnode_bits = static_cast<int>(sizeof(nodemask) * 8);

        // flags = 0 的含义：
        //   - 不带 MPOL_MF_MOVE : 此时该区域尚无 page,无迁移成本
        //   - 不带 MPOL_MF_STRICT: 不要求已存 page 符合 policy
        // 后续 lazy fault / khugepaged collapse 均读此 VMA policy。
        if (mbind(start, len, MPOL_BIND, nodemask, maxnode_bits, 0) != 0)
        {
            std::cerr << "mbind failed on arena " << i
                << " (len=" << len << ", node=" << i
                << "): errno=" << errno
                << ", falling back to default policy" << std::endl;
            // 不 std::exit：允许降级到内核默认策略,保留可用性
        }
    }
#else
    // 没有 libnuma:无 mbind 调用,单节点本地分配即可
#endif
}

inline void* ConcurrentMemoryPool::allocate_before_init_arenas(uint64_t bytes)
{
    if (bytes == 0)
    {
        return nullptr;
    }

    size_t alloc_size = align_up(static_cast<size_t>(bytes), HUGE_PAGE_SIZE);

    std::lock_guard<std::mutex> lock(pre_arena_mutex_);

    if (arenas_initialized_.load(std::memory_order_relaxed))
    {
        std::cerr << "allocate_before_init_arenas called after init_arenas" << std::endl;
        std::exit(-1);
    }

    size_t current_offset = pre_arena_offset_;
    if (current_offset + alloc_size > memory_size_)
    {
        std::cerr << "allocate_before_init_arenas out of memory" << std::endl;
        std::exit(-1);
    }

    void* ptr = static_cast<char*>(memory_start_) + current_offset;
    pre_arena_offset_ += alloc_size;

    // 对齐断言：memory_start_ 是 2MB 对齐，current_offset 也是 2MB 对齐增长
    if (!is_aligned(reinterpret_cast<uintptr_t>(ptr), HUGE_PAGE_SIZE))
    {
        std::cerr << "allocate_before_init_arenas alignment error" << std::endl;
        std::exit(-1);
    }

    return ptr;
}

inline void ConcurrentMemoryPool::init_arenas()
{
    std::lock_guard<std::mutex> lock(pre_arena_mutex_);

    if (arenas_initialized_.load(std::memory_order_relaxed))
    {
        std::cerr << "init_arenas already called" << std::endl;
        std::exit(-1);
    }

    size_t used = pre_arena_offset_;
    char* arena_start = static_cast<char*>(memory_start_) + used;

    // 由于 memory_start_ 是 2MB 对齐，used 也是 2MB 对齐增长，
    // arena_start 自然满足 4KB 对齐。此处保留断言以防万一。
    if (!is_aligned(reinterpret_cast<uintptr_t>(arena_start), BLOCK_SIZE))
    {
        arena_start = reinterpret_cast<char*>(align_up(reinterpret_cast<uintptr_t>(arena_start), BLOCK_SIZE));
    }

    size_t arena_area_bytes = memory_size_ - used;
    if (arena_area_bytes <= static_cast<size_t>(arena_start - static_cast<char*>(memory_start_)))
    {
        std::cerr << "No memory left for arenas" << std::endl;
        std::exit(-1);
    }
    arena_area_bytes -= static_cast<size_t>(arena_start - static_cast<char*>(memory_start_));

    total_blocks_ = arena_area_bytes / BLOCK_SIZE;
    if (total_blocks_ == 0)
    {
        std::cerr << "No memory left for arenas" << std::endl;
        std::exit(-1);
    }

    char* current_addr = arena_start;
    size_t remaining_blocks = total_blocks_;

    for (int i = 0; i < num_arenas_; ++i)
    {
        size_t arena_blocks;
        if (i == num_arenas_ - 1)
        {
            arena_blocks = remaining_blocks;
        }
        else
        {
            arena_blocks = (total_blocks_ * cpus_per_node_[i]) / total_cpus_;
            if (arena_blocks > remaining_blocks)
            {
                arena_blocks = remaining_blocks;
            }
        }

        size_t arena_bytes = arena_blocks * BLOCK_SIZE;

        arenas_[i].start_addr = current_addr;
        arenas_[i].end_addr = current_addr + arena_bytes;
        arenas_[i].bump_cursor.store(current_addr, std::memory_order_relaxed);
        arenas_[i].central_free_list = nullptr;

        current_addr += arena_bytes;
        remaining_blocks -= arena_blocks;
    }

    // apply_arena_mbind();

    // 所有 Arena 设置完成后，release 发布，确保对后续线程可见
    arenas_initialized_.store(true, std::memory_order_release);
}

// NUMA-aware first-touch：将每个 Arena 的内存页按 NUMA 节点本地线程并行写入，
// 触发页面实际分配在对应节点上。
inline void ConcurrentMemoryPool::perform_first_touch(size_t num_threads)
{
    if (!num_threads) {
        num_threads = static_cast<size_t>(total_cpus_);
    }
    num_threads = std::max<size_t>(1, std::min<size_t>(num_threads, static_cast<size_t>(total_cpus_)));

    // 计算每个 Arena 应分配的线程数，按 CPU 比例分配
    std::vector<int> threads_per_arena(num_arenas_, 0);
    int remaining = static_cast<int>(num_threads);
    for (int i = 0; i < num_arenas_; ++i)
    {
        if (cpus_per_node_[i] <= 0)
        {
            threads_per_arena[i] = 0;
            continue;
        }
        if (i == num_arenas_ - 1)
        {
            threads_per_arena[i] = remaining;
        }
        else
        {
            threads_per_arena[i] = static_cast<int>((num_threads * cpus_per_node_[i]) / total_cpus_);
            if (threads_per_arena[i] < 1)
            {
                threads_per_arena[i] = 1;
            }
            remaining -= threads_per_arena[i];
        }
    }
    // 如果还有剩余线程，分配给第一个有 CPU 的节点
    if (remaining > 0)
    {
        for (int i = 0; i < num_arenas_ && remaining > 0; ++i)
        {
            if (cpus_per_node_[i] > 0)
            {
                threads_per_arena[i] += remaining;
                remaining = 0;
            }
        }
    }

    std::vector<std::thread> threads;
    for (int arena_idx = 0; arena_idx < num_arenas_; ++arena_idx)
    {
        if (threads_per_arena[arena_idx] == 0) continue;
        Arena& arena = arenas_[arena_idx];
        size_t arena_bytes = static_cast<size_t>(
            static_cast<char*>(arena.end_addr) - static_cast<char*>(arena.start_addr));
        size_t pages = arena_bytes / page_size_;
        if (pages == 0) continue;

        size_t pages_per_thread = pages / threads_per_arena[arena_idx];
        if (pages_per_thread == 0) pages_per_thread = 1;

        for (int rank = 0; rank < threads_per_arena[arena_idx]; ++rank)
        {
            size_t start_page = rank * pages_per_thread;
            size_t end_page = (rank == threads_per_arena[arena_idx] - 1) ?
                pages : (rank + 1) * pages_per_thread;
            if (end_page > pages) end_page = pages;

            threads.emplace_back([this, arena_idx, start_page, end_page]() {
#if HAS_LIBNUMA
                if (numa_available_ && cpus_per_node_[arena_idx] > 0)
                {
                    // 绑定到目标节点，失败不影响 touch（可能分配在错误节点）
                    numa_run_on_node(arena_idx);
                }
#endif
                char* start = static_cast<char*>(arenas_[arena_idx].start_addr);
                for (size_t p = start_page; p < end_page; ++p)
                {
                    // 只触摸每页的第一个字节，使用 volatile 防止 O3 优化掉
                    volatile char* vp = start + p * page_size_;
                    *vp = 0;
                }
                });
        }
    }

    for (auto& t : threads) t.join();
}

inline char* ConcurrentMemoryPool::bump_allocate_from_arena(Arena& arena, size_t bytes)
{
    if (bytes == 0)
    {
        return nullptr;
    }

    size_t aligned_bytes = align_up(bytes, BLOCK_SIZE);
    char* arena_end = static_cast<char*>(arena.end_addr);

    SpinBackoff<> backoff;
    for (;;)
    {
        char* cursor = arena.bump_cursor.load(std::memory_order_relaxed);
        if (!cursor)
        {
            return nullptr;
        }

        size_t available = static_cast<size_t>(arena_end - cursor);
        if (aligned_bytes > available)
        {
            return nullptr;
        }

        char* next = cursor + aligned_bytes;

        if (arena.bump_cursor.compare_exchange_weak(cursor, next,
            std::memory_order_relaxed,
            std::memory_order_relaxed))
        {
            return cursor;
        }
        backoff.backoff();
    }
}

inline FreeBlock* ConcurrentMemoryPool::batch_allocate_from_bump(Arena& arena, size_t& out_count, FreeBlock** out_tail)
{
    if (out_count == 0)
    {
        return nullptr;
    }

    char* arena_end = static_cast<char*>(arena.end_addr);
    SpinBackoff<16, 128, 256> backoff;

    for (;;)
    {
        char* cursor = arena.bump_cursor.load(std::memory_order_relaxed);
        if (!cursor || cursor >= arena_end)
        {
            out_count = 0;
            return nullptr;
        }

        size_t available_bytes = static_cast<size_t>(arena_end - cursor);
        size_t available_blocks = available_bytes / BLOCK_SIZE;
        if (available_blocks == 0)
        {
            out_count = 0;
            return nullptr;
        }

        size_t allocate_blocks = std::min(out_count, available_blocks);
        size_t allocate_bytes = allocate_blocks * BLOCK_SIZE;
        char* next = cursor + allocate_bytes;

        if (arena.bump_cursor.compare_exchange_weak(cursor, next,
            std::memory_order_relaxed,
            std::memory_order_relaxed))
        {
            FreeBlock* head = reinterpret_cast<FreeBlock*>(cursor);
            FreeBlock* tail = head;

            for (size_t i = 1; i < allocate_blocks; ++i)
            {
                FreeBlock* block = reinterpret_cast<FreeBlock*>(cursor + i * BLOCK_SIZE);
                tail->next = block;
                tail = block;
            }

            tail->next = nullptr;
            *out_tail = tail;
            out_count = allocate_blocks;
            return head;
        }
        backoff.backoff();
    }
}

inline size_t ConcurrentMemoryPool::batch_allocate_from_arena(Arena& arena, FreeBlock** out_head, FreeBlock** out_tail, size_t count)
{
    FreeBlock* head = nullptr;
    FreeBlock* tail = nullptr;
    size_t fetched = 0;

    {
        std::lock_guard<std::mutex> lock(arena.mutex);

        if (arena.central_free_list)
        {
            head = arena.central_free_list;
            tail = head;
            fetched = 1;

            while (tail->next && fetched < count)
            {
                tail = tail->next;
                ++fetched;
            }

            // 更新中央链表
            arena.central_free_list = tail->next;
            tail->next = nullptr;
        }
    }

    if (fetched > 0)
    {
        *out_head = head;
        *out_tail = tail;
        return fetched;
    }

    size_t bump_count = count;
    head = batch_allocate_from_bump(arena, bump_count, &tail);
    if (!head)
    {
        return 0;
    }

    *out_head = head;
    *out_tail = tail;
    return bump_count;
}

inline void ConcurrentMemoryPool::batch_deallocate_to_arena(Arena& arena, FreeBlock* head, FreeBlock* tail, size_t /*count*/)
{
    if (!head || !tail)
        return;

    std::lock_guard<std::mutex> lock(arena.mutex);
    tail->next = arena.central_free_list;
    arena.central_free_list = head;
}

inline int ConcurrentMemoryPool::find_arena_index(void* ptr) const
{
    // 线性搜索（Arena 数量很少，最多 16 个）
    for (int i = 0; i < num_arenas_; ++i)
    {
        if (arenas_[i].contains(ptr))
        {
            return i;
        }
    }
    return -1;
}

inline int ConcurrentMemoryPool::get_thread_arena_index() const
{
#if HAS_LIBNUMA
    if (numa_available_)
    {
        int cpu = sched_getcpu();
        if (cpu >= 0)
        {
            int node = numa_node_of_cpu(cpu);
            if (node >= 0 && node < num_arenas_)
            {
                return node;
            }
        }
    }
#endif
    return 0;
}

inline ThreadLocalCache& ConcurrentMemoryPool::get_thread_local_cache()
{
    return tls_cache_;
}

inline void* ConcurrentMemoryPool::allocate_large(size_t bytes)
{

    if (!arenas_initialized_.load(std::memory_order_acquire)) [[unlikely]]
    {
        std::cerr << "allocate_large called before init_arenas" << std::endl;
        std::exit(-1);
    }

    ThreadLocalCache& tls = tls_cache_;

    // 初始化 TLS（首次访问）
    if (!tls.pool) [[unlikely]]
    {
        tls.pool = this;
        tls.local_arena_index = get_thread_arena_index();
        tls.local_arena = &arenas_[tls.local_arena_index];
    }

    char* ptr = bump_allocate_from_arena(*tls.local_arena, bytes);
    if (ptr) [[likely]]
    {
        return ptr;
    }

    // 按 NUMA 距离顺序尝试其他 Arena
    for (int i = 0; i < num_arenas_; ++i)
    {
        int candidate = numa_distance_order_[tls.local_arena_index][i];
        if (candidate < 0 || candidate >= num_arenas_ || candidate == tls.local_arena_index)
            continue;

        ptr = bump_allocate_from_arena(arenas_[candidate], bytes);
        if (ptr)
        {
            return ptr;
        }
    }

    std::cerr << "std::bad_alloc" << std::endl;
    std::exit(-1);
}

inline void* ConcurrentMemoryPool::allocate()
{
    if (!arenas_initialized_.load(std::memory_order_acquire)) [[unlikely]]
    {
        std::cerr << "allocate called before init_arenas" << std::endl;
        std::exit(-1);
    }

    ThreadLocalCache& tls = tls_cache_;

    FreeBlock* head, * tail;
    size_t fetched = 0;

    // 初始化 TLS（首次访问）
    if (!tls.pool) [[unlikely]]
    {
        tls.pool = this;
        tls.local_arena_index = get_thread_arena_index();
        tls.local_arena = &arenas_[tls.local_arena_index];
        fetched = batch_allocate_from_arena(*tls.local_arena, &head, &tail, TLS_CAPACITY);
    }
    else
    {
        // 1. 尝试从本地空闲栈分配
        if (tls.local_free_stack) [[likely]]
        {
            FreeBlock* block = tls.local_free_stack;
            tls.local_free_stack = block->next;
            tls.local_free_count--;

            return block;
        }
        fetched = batch_allocate_from_arena(*tls.local_arena, &head, &tail, BATCH_SIZE);
    }

    // 2. 本地栈为空，从本地 Arena 批量获取
    if (fetched > 0)
    {
        // 第一个块立即返回
        FreeBlock* result = head;

        // 直接挂接剩余链表，避免额外遍历与反转
        tls.local_free_stack = head->next;
        result->next = nullptr;
        tls.local_free_count = fetched - 1;

        return result;
    }

    // 3. 本地 Arena 为空，尝试按 NUMA 距离顺序跨 Arena 窃取
    for (int i = 0; i < num_arenas_; ++i)
    {
        int candidate = numa_distance_order_[tls.local_arena_index][i];
        if (candidate < 0 || candidate >= num_arenas_ || candidate == tls.local_arena_index)
            continue;

        fetched = batch_allocate_from_arena(arenas_[candidate], &head, &tail, BATCH_SIZE);
        if (fetched > 0)
        {
            FreeBlock* result = head;

            // 与本地 refill 分支保持一致：直接挂接剩余链表
            tls.local_free_stack = head->next;
            result->next = nullptr;
            tls.local_free_count = fetched - 1;

            return result;
        }
    }

    // 4. 所有 Arena 都为空
    std::cerr << "std::bad_alloc" << std::endl;
    std::exit(-1);
}

inline void ConcurrentMemoryPool::deallocate(void* ptr)
{
    if (!ptr)
        return;

    if (!arenas_initialized_.load(std::memory_order_acquire))
    {
        std::cerr << "deallocate called before init_arenas" << std::endl;
        std::exit(-1);
    }

    ThreadLocalCache& tls = tls_cache_;

    // 初始化 TLS（首次访问）
    if (!tls.pool)
    {
        tls.pool = this;
        tls.local_arena_index = get_thread_arena_index();
        tls.local_arena = &arenas_[tls.local_arena_index];
    }

    FreeBlock* block = reinterpret_cast<FreeBlock*>(ptr);

    // 查找块属于哪个 Arena
    int arena_idx = find_arena_index(ptr);
    if (arena_idx < 0) [[unlikely]]
    {
        // 不属于任何 Arena，可能是无效指针
        return;
    }

    if (arena_idx == tls.local_arena_index)
    {
        // 本地 Arena
        if (tls.local_free_count < TLS_CAPACITY)
        {
            // 压入本地栈
            block->next = tls.local_free_stack;
            tls.local_free_stack = block;
            tls.local_free_count++;
        }
        else
        {
            // 栈已满，批量归还一半
            size_t half = TLS_CAPACITY / 2;
            FreeBlock* tail = tls.local_free_stack;
            for (size_t i = 1; i < half && tail->next; ++i)
            {
                tail = tail->next;
            }

            FreeBlock* remaining = tail->next;
            tail->next = nullptr;

            batch_deallocate_to_arena(*tls.local_arena, tls.local_free_stack, tail, half);

            tls.local_free_stack = remaining;
            tls.local_free_count = TLS_CAPACITY - half;

            // 将新块压入栈
            block->next = tls.local_free_stack;
            tls.local_free_stack = block;
            tls.local_free_count++;
        }
    }
    else
    {
        // 远程 Arena
        RemoteListHead& rl = tls.remote_lists[arena_idx];
        block->next = rl.head;
        rl.head = block;
        rl.count++;

        if (rl.count >= REMOTE_LIST_CAPACITY)
        {
            // 批量归还到远程 Arena
            FreeBlock* head = rl.head;
            // 找到尾节点
            FreeBlock* tail = head;
            while (tail->next)
            {
                tail = tail->next;
            }

            batch_deallocate_to_arena(arenas_[arena_idx], head, tail, rl.count);

            rl.head = nullptr;
            rl.count = 0;
        }
    }
}

inline int ConcurrentMemoryPool::get_current_arena_index() const
{
    return get_thread_arena_index();
}

#endif // CONCURRENT_MEMORY_POOL_HEADER