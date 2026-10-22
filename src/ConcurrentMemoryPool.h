#ifndef CONCURRENT_MEMORY_POOL_HEADER
#define CONCURRENT_MEMORY_POOL_HEADER

// 并发内存池 - NUMA感知设计
// 固定块模型（4KB），线程本地缓存，跨线程释放支持

#include "definition.h"
#include "SpinBackoff.h"
#include "RunState.h"
#include <memory>
#include <bit>
#ifdef TEST_MODE
#include <chrono>
#endif
#include <new>

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
#include <limits>

// 系统头文件
#include <sys/mman.h>
#include <fcntl.h>
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
constexpr size_t FIRST_SEGMENT_SIZE = 2ULL * 1024 * 1024;

//==============================================================================
// 辅助工具
//==============================================================================

// 向上对齐到指定边界
constexpr size_t align_up(size_t value, size_t alignment)
{
    if (value > SIZE_MAX - (alignment - 1)) throw std::overflow_error("alignment overflow");
    return (value + alignment - 1) & ~(alignment - 1);
}

constexpr size_t proportional_size(size_t total, size_t share, size_t sum) noexcept
{
    return total / sum * share + total % sum * share / sum;
}

constexpr size_t arena_watermark(size_t capacity, size_t percent)
{
    return align_up(capacity / 100 * percent + (capacity % 100 * percent + 99) / 100, HUGE_PAGE_SIZE);
}

// 检查是否对齐
constexpr bool is_aligned(size_t value, size_t alignment)
{
    return (value & (alignment - 1)) == 0;
}

enum class MemoryPressure : uint8_t { Normal, Soft, Hard };
enum class FirstSegmentState : uint8_t { Retired, Filling, Sealed, Consuming };

struct FirstSegmentRef
{
    uint32_t arena_index = 0;
    uint64_t segment_index = 0;
    uint64_t generation = 0;
};

// Permanent metadata for each 2 MiB slot. State and generation are protected by
// the Arena admission lock; only the Filling/Consuming holder accesses data.
struct FirstSegmentDescriptor
{
    char* data = nullptr;
    uint64_t generation = 0;
    FirstSegmentState state = FirstSegmentState::Retired;
};

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
    uint64_t capacity = 0;
    uint64_t tree_offset = 0, first_offset = 0;
    bool first_enabled = true;
    bool consuming_first = false;
    std::mutex admission;
    size_t low = 0, high = 0;
    std::atomic<MemoryPressure> pressure_hint{MemoryPressure::Normal};
    std::atomic<uint64_t> pressure_generation{0};
    bool hard_reclaim = false;
    size_t pending_bytes = 0;
    char* reserved_result = nullptr;
    std::unique_ptr<FirstSegmentDescriptor[]> segments;
    size_t segment_capacity = 0;
    size_t segment_frontier = 0;
    FreeBlock* central_free_list; // 中央空闲链表
    std::mutex mutex;             // 保护中央空闲链表的互斥锁

    Arena() : start_addr(nullptr), end_addr(nullptr), central_free_list(nullptr) {}

    // Called only under admission. Keeping arithmetic separate from addressing
    // also permits numeric validation beyond the host's virtual address space.
    bool reserve_tree(uint64_t bytes) noexcept
    {
        if (pending_bytes || bytes > first_offset - tree_offset) return false;
        tree_offset += bytes;
        return true;
    }

    bool reserve_first() noexcept
    {
        const uint64_t gap = first_offset - tree_offset;
        if (!first_enabled || consuming_first || pending_bytes || gap < FIRST_SEGMENT_SIZE || gap - FIRST_SEGMENT_SIZE < high) return false;
        first_offset -= FIRST_SEGMENT_SIZE;
        return true;
    }

    // 检查指针是否属于此 Arena
    bool contains(void* ptr) const
    {
        return ptr >= start_addr && ptr < end_addr;
    }

    // 获取此 Arena 管理的块数
    size_t block_count() const
    {
        return capacity / BLOCK_SIZE;
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

    // File pages never enter the RAM caches or the nonblocking allocation path.
    FreeBlock* file_free_stack = nullptr;
    size_t file_free_count = 0;
    bool file_started = false;
    uintptr_t file_range_begin = 0;
    size_t file_range_bytes = 0;

    ThreadLocalCache()
        : local_free_stack(nullptr), local_free_count(0), local_arena(nullptr), local_arena_index(-1), pool(nullptr) {
    }

    // 析构时归还所有缓存
    ~ThreadLocalCache();
    bool release(bool try_only = false);
};

//==============================================================================
// MemoryPool - 顶层内存池类
//==============================================================================

class ConcurrentMemoryPool
{
public:
    // 构造函数
    // total_bytes: 总字节数（内部会按 BLOCK_SIZE 对齐）
    explicit ConcurrentMemoryPool(size_t total_bytes, RunState* state = nullptr);

    // 禁止拷贝与赋值
    ConcurrentMemoryPool(const ConcurrentMemoryPool&) = delete;
    ConcurrentMemoryPool& operator=(const ConcurrentMemoryPool&) = delete;

    // 析构函数，释放所有 mmap 内存
    ~ConcurrentMemoryPool();

    // 分配一个 4KB 块，必要时等待独立 Spill；耗尽时抛出异常
    void* allocate();

    // 分配连续大块，必要时等待独立 Spill；耗尽时抛出异常
    void* allocate_large(size_t bytes);

    // 指定 Arena 的连续页分配，地址和大小按 PAGE_SIZE 对齐。
    // 不使用 TLS、空闲链表或其他 Arena。
    void* allocate_large_from_numa(size_t bytes, int numa_node);

    void* try_allocate() noexcept;
    void* try_allocate_large(size_t bytes) noexcept;
    static size_t first_segment_record_capacity() noexcept { return FIRST_SEGMENT_SIZE / kmer_bytes(); }
    bool cache_first_segment(int arena, const unsigned char* records) noexcept;
    enum class FirstExtractStatus { Data, Busy, Done };
    // Called only after cache growth is frozen and the pending Writer has joined.
    // Data owns a complete copy; its source was returned before this call returns.
    FirstExtractStatus extract_first_segment(int preferred_arena, unsigned char* destination) noexcept;
    void empty_first(const FirstSegmentRef& ref) noexcept;
    FirstSegmentDescriptor* claim_spill(int arena_index, FirstSegmentRef& result) noexcept;
    MemoryPressure pressure(int arena_index) const noexcept { return arenas_[arena_index].pressure_hint.load(std::memory_order_acquire); }
    uint64_t pressure_generation(int arena_index) const noexcept { return arenas_[arena_index].pressure_generation.load(std::memory_order_acquire); }
    size_t segment_capacity() const noexcept { return memory_size_ / FIRST_SEGMENT_SIZE; }
    size_t segment_capacity(int arena) const noexcept { return arenas_[arena].segment_capacity; }
    int arena_count() const noexcept { return num_arenas_; }
    int arena_node(int index) const noexcept { return node_ids_[index]; }
    int arena_for_node(int node) const noexcept;
    RunState& run_state() noexcept { return *state_; }
    void set_spill_service(bool running) { spill_service_.store(running, std::memory_order_release); state_->notify(); }
    void freeze_first()
    {
        for (int i = 0; i < num_arenas_; ++i) {
            std::lock_guard lock(arenas_[i].admission);
            arenas_[i].first_enabled = false;
        }
        state_->notify();
    }
#ifdef TEST_MODE
    uint64_t first_resident_records()
    {
        uint64_t records = 0;
        for (int i = 0; i < num_arenas_; ++i) {
            std::lock_guard lock(arenas_[i].admission);
            records += arenas_[i].segment_frontier * first_segment_record_capacity();
        }
        return records;
    }
    std::atomic<int64_t> drain_started_ns{0};
    std::atomic<int64_t> input_finished_ns{0};
    std::atomic<uint64_t> spill_at_drain{0};
#endif
    void request_first_drain()
    {
#ifdef TEST_MODE
        const auto started = std::chrono::steady_clock::now();
        const uint64_t spilled = spill_bytes.load(std::memory_order_relaxed);
#endif
        if (!drain_first_.exchange(true, std::memory_order_acq_rel)) {
#ifdef TEST_MODE
            spill_at_drain.store(spilled, std::memory_order_release);
            drain_started_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                started.time_since_epoch()).count(), std::memory_order_release);
#endif
        }
        state_->notify();
    }
    bool first_empty() noexcept;
    size_t gap_bytes(int index) noexcept;
#ifdef TEST_MODE
    std::array<std::atomic<uint64_t>, 64> wait_latency_buckets{};
    std::atomic<uint64_t> spill_bytes{0}, returned_bytes{0}, allocation_waits{0};
    std::atomic<uint64_t> first_publications{0}, spill_claims{0};
    std::atomic<uint64_t> first_rejected_capacity{0}, first_rejected_tree{0}, first_rejected_consuming{0}, first_rejected_frozen{0};
    std::atomic<uint64_t> file_reservations{0}, file_preallocations{0}, file_mappings{0}, file_expansions{0}, file_committed_bytes{0}, file_segment_count{0};
#endif

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
    size_t batch_allocate_from_arena(Arena& arena, FreeBlock** out_head, FreeBlock** out_tail, size_t count, bool try_only = false);

    char* bump_allocate_from_arena(Arena& arena, size_t bytes, bool try_only = false);

    FreeBlock* batch_allocate_from_bump(Arena& arena, size_t& out_count, FreeBlock** out_tail, bool try_only);

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

    RunState local_state_;
    RunState* state_;
    std::atomic<bool> spill_service_{false}, drain_first_{false};
    int node_ids_[MAX_NUMA_NODES]{};
    void update_pressure(Arena& arena) noexcept;
    void* wait_allocate(size_t bytes, int preferred = -1);
    void* allocate_small(bool try_only) noexcept;
    bool prepare_tls(bool try_only = false);

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

    struct FileSegment
    {
        char* base;
        size_t reserved, committed, used, file_offset;
    };
    static constexpr size_t FILE_SEGMENT_BYTES = 16ULL << 30;
    static constexpr size_t FILE_MAX_GROWTH = 1ULL << 30;
    const std::string file_path_ = mapping_filename;
    std::mutex file_mutex_;
    std::vector<FileSegment> file_segments_; // Sorted by virtual address.
    size_t file_current_ = SIZE_MAX, file_end_ = 0, file_growth_ = 64ULL << 20;
    FreeBlock* file_free_list_ = nullptr;
    int file_fd_ = -1;
    bool file_expanding_ = false;
    // Cold-path syscall injection is private to the existing test friend.
    decltype(&::mmap) file_map_ = ::mmap;
    decltype(&::fallocate) file_preallocate_ = ::fallocate;
    void* allocate_file(size_t bytes, bool small);
    void deallocate_file(void* ptr);

    // 友元类
    friend struct ThreadLocalCache;
    friend struct ConcurrentMemoryPoolTestAccess;
};

//==============================================================================
// ThreadLocalCache 析构函数实现
//==============================================================================

inline ThreadLocalCache::~ThreadLocalCache()
{
    release();
}

inline bool ThreadLocalCache::release(bool try_only)
{
    if (!pool)
        return true;

    // Pool changes are cold, but try allocation must not wait while flushing
    // an old pool. Acquire everything before modifying the TLS lists.
    std::unique_lock progress(pool->state_->mutex, std::defer_lock);
    if (try_only) { if (!progress.try_lock()) return false; }
    else progress.lock();
    std::unique_lock<std::mutex> locks[MAX_NUMA_NODES];
    for (int i = 0; i < pool->num_arenas_; ++i) {
        locks[i] = std::unique_lock(pool->arenas_[i].mutex, std::defer_lock);
        if (try_only) { if (!locks[i].try_lock()) return false; }
        else locks[i].lock();
    }
    std::unique_lock file_lock(pool->file_mutex_, std::defer_lock);
    if (file_free_stack) {
        if (try_only) { if (!file_lock.try_lock()) return false; }
        else file_lock.lock();
    }

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

    if (file_free_stack) {
        FreeBlock* tail = file_free_stack;
        while (tail->next) tail = tail->next;
        tail->next = pool->file_free_list_;
        pool->file_free_list_ = file_free_stack;
        file_free_stack = nullptr;
        file_free_count = 0;
    }
    file_started = false;
    file_range_begin = 0;
    file_range_bytes = 0;
    ++pool->state_->generation;
    progress.unlock();
    pool->state_->changed.notify_all();
    pool = nullptr;
    local_arena = nullptr;
    local_arena_index = -1;
    return true;
}

//==============================================================================
// MemoryPool 实现
//==============================================================================

inline ConcurrentMemoryPool::ConcurrentMemoryPool(size_t total_bytes, RunState* state)
    : state_(state ? state : &local_state_), num_arenas_(1), mmap_base_(nullptr), mmap_size_(0), memory_start_(nullptr), memory_size_(0),
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
        throw std::bad_alloc();
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
        tls.file_free_stack = nullptr;
        tls.file_free_count = 0;
        tls.file_started = false;
        tls.file_range_begin = 0;
        tls.file_range_bytes = 0;
    }

    for (const auto& segment : file_segments_) ::munmap(segment.base, segment.reserved);
    if (file_fd_ >= 0) ::close(file_fd_);

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
    cpu_set_t mask;
    CPU_ZERO(&mask);
    const bool have_mask = sched_getaffinity(0, sizeof(mask), &mask) == 0;
    num_arenas_ = 0;
    total_cpus_ = 0;
    for (int node = 0; node <= numa_max_node() && num_arenas_ < MAX_NUMA_NODES; ++node) {
        if (!numa_bitmask_isbitset(numa_all_nodes_ptr, node)) continue;
        int count = 0;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if ((!have_mask || CPU_ISSET(cpu, &mask)) && numa_node_of_cpu(cpu) == node) ++count;
        if (count == 0) continue;
        node_ids_[num_arenas_] = node;
        cpus_per_node_[num_arenas_++] = count;
        total_cpus_ += count;
    }
    if (num_arenas_ == 0) {
        num_arenas_ = 1;
        total_cpus_ = cpus_per_node_[0] = 1;
        node_ids_[0] = 0;
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
                dists[node].distance = numa_distance(node_ids_[local], node_ids_[node]);
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
    size_t request_size = total_bytes & ~(BLOCK_SIZE - 1);
    if (request_size < HUGE_PAGE_SIZE) throw std::bad_alloc();

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
            throw std::bad_alloc();
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
    const size_t prefix = aligned - raw;
    memory_size_ = (request_size - prefix) & ~(HUGE_PAGE_SIZE - 1);

    // 对齐检查
    if (!is_aligned(reinterpret_cast<uintptr_t>(memory_start_), HUGE_PAGE_SIZE))
    {
        std::cerr << "mmap memory_start_ is not 2MB aligned" << std::endl;
        throw std::bad_alloc();
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

        std::unique_ptr<bitmask, decltype(&numa_bitmask_free)> mask(numa_allocate_nodemask(), numa_bitmask_free);
        if (!mask) throw std::bad_alloc();
        numa_bitmask_setbit(mask.get(), node_ids_[i]);

        // flags = 0 的含义：
        //   - 不带 MPOL_MF_MOVE : 此时该区域尚无 page,无迁移成本
        //   - 不带 MPOL_MF_STRICT: 不要求已存 page 符合 policy
        // 后续 lazy fault / khugepaged collapse 均读此 VMA policy。
        if (mbind(start, len, MPOL_BIND, mask->maskp, mask->size, 0) != 0)
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

    if (bytes > std::numeric_limits<size_t>::max() - (HUGE_PAGE_SIZE - 1)) throw std::bad_alloc();
    size_t alloc_size = align_up(static_cast<size_t>(bytes), HUGE_PAGE_SIZE);

    std::lock_guard<std::mutex> lock(pre_arena_mutex_);

    if (arenas_initialized_.load(std::memory_order_relaxed))
    {
        std::cerr << "allocate_before_init_arenas called after init_arenas" << std::endl;
        throw std::bad_alloc();
    }

    size_t current_offset = pre_arena_offset_;
    if (alloc_size > memory_size_ - current_offset)
    {
        std::cerr << "allocate_before_init_arenas out of memory" << std::endl;
        throw std::bad_alloc();
    }

    void* ptr = static_cast<char*>(memory_start_) + current_offset;
    pre_arena_offset_ += alloc_size;

    // 对齐断言：memory_start_ 是 2MB 对齐，current_offset 也是 2MB 对齐增长
    if (!is_aligned(reinterpret_cast<uintptr_t>(ptr), HUGE_PAGE_SIZE))
    {
        std::cerr << "allocate_before_init_arenas alignment error" << std::endl;
        throw std::bad_alloc();
    }

    return ptr;
}

inline void ConcurrentMemoryPool::init_arenas()
{
    std::lock_guard<std::mutex> lock(pre_arena_mutex_);

    if (arenas_initialized_.load(std::memory_order_relaxed))
    {
        std::cerr << "init_arenas already called" << std::endl;
        throw std::bad_alloc();
    }

    size_t used = pre_arena_offset_;
    char* arena_start = static_cast<char*>(memory_start_) + used;

    // 由于 memory_start_ 是 2MB 对齐，used 也是 2MB 对齐增长，
    // arena_start 自然满足 4KB 对齐。此处保留断言以防万一。
    if (!is_aligned(reinterpret_cast<uintptr_t>(arena_start), BLOCK_SIZE))
    {
        arena_start = reinterpret_cast<char*>(align_up(reinterpret_cast<uintptr_t>(arena_start), BLOCK_SIZE));
    }

    size_t arena_area_bytes = memory_size_;
    if (arena_area_bytes <= static_cast<size_t>(arena_start - static_cast<char*>(memory_start_)))
    {
        std::cerr << "No memory left for arenas" << std::endl;
        throw std::bad_alloc();
    }
    arena_area_bytes -= static_cast<size_t>(arena_start - static_cast<char*>(memory_start_));

    total_blocks_ = arena_area_bytes / BLOCK_SIZE;
    if (total_blocks_ == 0)
    {
        std::cerr << "No memory left for arenas" << std::endl;
        throw std::bad_alloc();
    }

    char* current_addr = arena_start;
    const size_t total_segments = arena_area_bytes / HUGE_PAGE_SIZE;
    size_t remaining_segments = total_segments;

    for (int i = 0; i < num_arenas_; ++i)
    {
        size_t arena_segments;
        if (i == num_arenas_ - 1)
        {
            arena_segments = remaining_segments;
        }
        else
        {
            arena_segments = proportional_size(total_segments, cpus_per_node_[i], total_cpus_);
            if (arena_segments > remaining_segments)
            {
                arena_segments = remaining_segments;
            }
        }

        size_t arena_bytes = arena_segments * HUGE_PAGE_SIZE;

        arenas_[i].start_addr = current_addr;
        arenas_[i].end_addr = current_addr + arena_bytes;
        arenas_[i].central_free_list = nullptr;
        Arena& arena = arenas_[i];
        arena.capacity = arena.first_offset = arena_bytes;
        arena.tree_offset = 0;
        arena.low = arena_watermark(arena_bytes, 5);
        arena.high = arena_watermark(arena_bytes, 8);
        arena.segment_capacity = arena_bytes / FIRST_SEGMENT_SIZE;
        if (arena.segment_capacity > SIZE_MAX / sizeof(FirstSegmentDescriptor)) throw std::bad_alloc();
        arena.segments = std::make_unique<FirstSegmentDescriptor[]>(arena.segment_capacity);
        update_pressure(arena);

        current_addr += arena_bytes;
        remaining_segments -= arena_segments;
    }

    apply_arena_mbind();

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
            threads_per_arena[i] = static_cast<int>(proportional_size(num_threads, cpus_per_node_[i], total_cpus_));
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
                    numa_run_on_node(node_ids_[arena_idx]);
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

inline void ConcurrentMemoryPool::update_pressure(Arena& arena) noexcept
{
    const uint64_t gap = arena.first_offset - arena.tree_offset;
    if (arena.pending_bytes || gap <= arena.low) arena.hard_reclaim = true;
    else if (gap >= arena.high) arena.hard_reclaim = false;
    const MemoryPressure next = arena.pending_bytes || gap <= arena.low ? MemoryPressure::Hard
        : gap < arena.high ? MemoryPressure::Soft : MemoryPressure::Normal;
    if (arena.pressure_hint.load(std::memory_order_relaxed) != next) {
        arena.pressure_hint.store(next, std::memory_order_release);
        arena.pressure_generation.fetch_add(1, std::memory_order_release);
    }
}

inline char* ConcurrentMemoryPool::bump_allocate_from_arena(Arena& arena, size_t bytes, bool try_only)
{
    if (!bytes || bytes > SIZE_MAX - (BLOCK_SIZE - 1)) return nullptr;
    const size_t aligned = align_up(bytes, BLOCK_SIZE);
    std::unique_lock lock(arena.admission, std::defer_lock);
    if (try_only) { if (!lock.try_lock()) return nullptr; }
    else lock.lock();
    const uint64_t offset = arena.tree_offset;
    if (!arena.reserve_tree(aligned)) return nullptr;
    update_pressure(arena);
    return static_cast<char*>(arena.start_addr) + offset;
}

inline FreeBlock* ConcurrentMemoryPool::batch_allocate_from_bump(Arena& arena, size_t& out_count, FreeBlock** out_tail, bool try_only)
{
    char* cursor;
    {
        std::unique_lock lock(arena.admission, std::defer_lock);
        if (try_only) { if (!lock.try_lock()) return nullptr; }
        else lock.lock();
        const size_t available = (arena.first_offset - arena.tree_offset) / BLOCK_SIZE;
        out_count = arena.pending_bytes ? 0 : std::min(out_count, available);
        if (!out_count) return nullptr;
        const uint64_t offset = arena.tree_offset;
        arena.reserve_tree(out_count * BLOCK_SIZE);
        cursor = static_cast<char*>(arena.start_addr) + offset;
        update_pressure(arena);
    }
    auto* head = reinterpret_cast<FreeBlock*>(cursor);
    FreeBlock* tail = head;
    for (size_t i = 1; i < out_count; ++i) {
        tail->next = reinterpret_cast<FreeBlock*>(cursor + i * BLOCK_SIZE);
        tail = tail->next;
    }
    tail->next = nullptr;
    *out_tail = tail;
    return head;
}

inline size_t ConcurrentMemoryPool::batch_allocate_from_arena(Arena& arena, FreeBlock** out_head, FreeBlock** out_tail, size_t count, bool try_only)
{
    FreeBlock* head = nullptr;
    FreeBlock* tail = nullptr;
    size_t fetched = 0;

    {
        std::unique_lock lock(arena.mutex, std::defer_lock);
        if (try_only) { if (!lock.try_lock()) return 0; }
        else lock.lock();

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
    head = batch_allocate_from_bump(arena, bump_count, &tail, try_only);
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

    {
        std::lock_guard<std::mutex> lock(arena.mutex);
        tail->next = arena.central_free_list;
        arena.central_free_list = head;
    }
    state_->notify();
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
            return arena_for_node(node);
        }
    }
#endif
    return 0;
}

inline ThreadLocalCache& ConcurrentMemoryPool::get_thread_local_cache()
{
    return tls_cache_;
}

inline void* ConcurrentMemoryPool::allocate_large_from_numa(size_t bytes, int numa_node)
{
    if (!arenas_initialized_.load(std::memory_order_acquire) || numa_node < 0 || numa_node >= num_arenas_
        || !bytes || bytes > SIZE_MAX - (BLOCK_SIZE - 1)) throw std::bad_alloc();
    if (void* result = bump_allocate_from_arena(arenas_[numa_node], bytes)) return result;
    return wait_allocate(bytes, numa_node);
}

inline void* ConcurrentMemoryPool::try_allocate_large(size_t bytes) noexcept
{

    if (!arenas_initialized_.load(std::memory_order_acquire)) [[unlikely]]
    {
        return nullptr;
    }

    if (!prepare_tls(true)) return nullptr;
    ThreadLocalCache& tls = tls_cache_;

    // 初始化 TLS（首次访问）
    if (!tls.pool) [[unlikely]]
    {
        tls.pool = this;
        tls.local_arena_index = get_thread_arena_index();
        tls.local_arena = &arenas_[tls.local_arena_index];
    }

    char* ptr = bump_allocate_from_arena(*tls.local_arena, bytes, true);
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

        ptr = bump_allocate_from_arena(arenas_[candidate], bytes, true);
        if (ptr)
        {
            return ptr;
        }
    }

    return nullptr;
}

inline void* ConcurrentMemoryPool::try_allocate() noexcept
{
    return allocate_small(true);
}

inline void* ConcurrentMemoryPool::allocate_small(bool try_only) noexcept
{
    if (!arenas_initialized_.load(std::memory_order_acquire)) [[unlikely]]
    {
        return nullptr;
    }

    if (!prepare_tls(try_only)) return nullptr;
    ThreadLocalCache& tls = tls_cache_;

    FreeBlock* head, * tail;
    size_t fetched = 0;

    // 初始化 TLS（首次访问）
    if (!tls.pool) [[unlikely]]
    {
        tls.pool = this;
        tls.local_arena_index = get_thread_arena_index();
        tls.local_arena = &arenas_[tls.local_arena_index];
        fetched = batch_allocate_from_arena(*tls.local_arena, &head, &tail, TLS_CAPACITY, try_only);
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
        fetched = batch_allocate_from_arena(*tls.local_arena, &head, &tail, BATCH_SIZE, try_only);
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

        fetched = batch_allocate_from_arena(arenas_[candidate], &head, &tail, BATCH_SIZE, try_only);
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
    return nullptr;
}

inline void ConcurrentMemoryPool::deallocate(void* ptr)
{
    if (!ptr)
        return;

    if (!arenas_initialized_.load(std::memory_order_acquire))
    {
        std::cerr << "deallocate called before init_arenas" << std::endl;
        throw std::bad_alloc();
    }

    prepare_tls();
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
        deallocate_file(ptr);
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

inline bool ConcurrentMemoryPool::prepare_tls(bool try_only)
{
    if (tls_cache_.pool && tls_cache_.pool != this) {
        return tls_cache_.release(try_only);
    }
    return true;
}

inline int ConcurrentMemoryPool::arena_for_node(int node) const noexcept
{
    for (int i = 0; i < num_arenas_; ++i) if (node_ids_[i] == node) return i;
    return 0;
}

inline size_t ConcurrentMemoryPool::gap_bytes(int index) noexcept
{
    Arena& arena = arenas_[index];
    std::lock_guard lock(arena.admission);
    return arena.first_offset - arena.tree_offset;
}

inline bool ConcurrentMemoryPool::cache_first_segment(int index, const unsigned char* records) noexcept
{
    Arena& arena = arenas_[index];
    FirstSegmentDescriptor* segment;
    {
        // Contention is not exhaustion. No data copy or external call under this lock.
        std::lock_guard lock(arena.admission);
        if (!arena.reserve_first()) {
#ifdef TEST_MODE
            if (!arena.first_enabled) first_rejected_frozen.fetch_add(1, std::memory_order_relaxed);
            else if (arena.consuming_first) first_rejected_consuming.fetch_add(1, std::memory_order_relaxed);
            else if (arena.pending_bytes) first_rejected_tree.fetch_add(1, std::memory_order_relaxed);
            else first_rejected_capacity.fetch_add(1, std::memory_order_relaxed);
#endif
            return false;
        }
        segment = &arena.segments[arena.segment_frontier++];
        ++segment->generation;
        segment->data = static_cast<char*>(arena.start_addr) + arena.first_offset;
        segment->state = FirstSegmentState::Filling;
        update_pressure(arena);
    }
    std::memcpy(segment->data, records, first_segment_record_capacity() * kmer_bytes());
    {
        std::lock_guard lock(arena.admission);
        segment->state = FirstSegmentState::Sealed;
#ifdef TEST_MODE
        first_publications.fetch_add(1, std::memory_order_relaxed);
#endif
    }
    return true;
}

inline void ConcurrentMemoryPool::empty_first(const FirstSegmentRef& ref) noexcept
{
    Arena& arena = arenas_[ref.arena_index];
    {
        std::lock_guard lock(arena.admission);
        auto& segment = arena.segments[ref.segment_index];
        if (segment.generation != ref.generation || segment.state != FirstSegmentState::Consuming) return;
        // Only the boundary can be claimed, and admission stays closed until its
        // source is no longer read. Thus every return immediately reclaims 2 MiB.
        segment.state = FirstSegmentState::Retired;
        segment.data = nullptr;
        --arena.segment_frontier;
        arena.first_offset += FIRST_SEGMENT_SIZE;
        arena.consuming_first = false;
#ifdef TEST_MODE
        returned_bytes.fetch_add(FIRST_SEGMENT_SIZE, std::memory_order_relaxed);
#endif
        if (arena.pending_bytes && !arena.reserved_result && arena.pending_bytes <= arena.first_offset - arena.tree_offset) {
            arena.reserved_result = static_cast<char*>(arena.start_addr) + arena.tree_offset;
            arena.tree_offset += arena.pending_bytes;
        }
        update_pressure(arena);
    }
    state_->notify();
}

inline FirstSegmentDescriptor* ConcurrentMemoryPool::claim_spill(int index, FirstSegmentRef& result) noexcept
{
    Arena& arena = arenas_[index];
    std::unique_lock lock(arena.admission, std::try_to_lock);
    if (!lock || arena.consuming_first || !arena.segment_frontier) return nullptr;
    if (!drain_first_.load(std::memory_order_acquire) && !arena.pending_bytes && !arena.hard_reclaim) return nullptr;
    const size_t slot = arena.segment_frontier - 1;
    auto& segment = arena.segments[slot];
    if (segment.state != FirstSegmentState::Sealed) return nullptr;
    segment.state = FirstSegmentState::Consuming;
    arena.consuming_first = true;
    result = {static_cast<uint32_t>(index), slot, segment.generation};
#ifdef TEST_MODE
    spill_claims.fetch_add(1, std::memory_order_relaxed);
#endif
    return &segment;
}

inline ConcurrentMemoryPool::FirstExtractStatus ConcurrentMemoryPool::extract_first_segment(
    int preferred, unsigned char* destination) noexcept
{
    bool busy = false;
    for (int rank = 0; rank < num_arenas_; ++rank) {
        const int index = numa_distance_order_[preferred][rank];
        Arena& arena = arenas_[index];
        FirstSegmentRef ref;
        FirstSegmentDescriptor* segment;
        {
            std::unique_lock lock(arena.admission, std::try_to_lock);
            if (!lock) { busy = true; continue; }
            if (!arena.segment_frontier) continue;
            const size_t slot = arena.segment_frontier - 1;
            segment = &arena.segments[slot];
            if (arena.consuming_first || segment->state != FirstSegmentState::Sealed) { busy = true; continue; }
            segment->state = FirstSegmentState::Consuming;
            arena.consuming_first = true;
            ref = {static_cast<uint32_t>(index), slot, segment->generation};
        }
        std::memcpy(destination, segment->data, first_segment_record_capacity() * kmer_bytes());
        empty_first(ref);
        return FirstExtractStatus::Data;
    }
    return busy ? FirstExtractStatus::Busy : FirstExtractStatus::Done;
}

inline bool ConcurrentMemoryPool::first_empty() noexcept
{
    for (int i = 0; i < num_arenas_; ++i) {
        std::lock_guard lock(arenas_[i].admission);
        if (arenas_[i].segment_frontier) return false;
    }
    return true;
}

inline void* ConcurrentMemoryPool::allocate()
{
    state_->check();
    if (void* result = allocate_small(false)) return result;
    return wait_allocate(BLOCK_SIZE);
}

inline void* ConcurrentMemoryPool::allocate_large(size_t bytes)
{
    state_->check();
    if (!bytes || bytes > SIZE_MAX - (BLOCK_SIZE - 1)) throw std::bad_alloc();
    if (void* result = try_allocate_large(bytes)) return result;
    return wait_allocate(bytes);
}

inline void* ConcurrentMemoryPool::wait_allocate(size_t bytes, int preferred)
{
    if (!arenas_initialized_.load(std::memory_order_acquire)
        || !bytes || bytes > SIZE_MAX - (BLOCK_SIZE - 1)) throw std::bad_alloc();
    bytes = align_up(bytes, BLOCK_SIZE);
    const int local = preferred >= 0 ? preferred : get_thread_arena_index();
    int admitted = -1;
#ifdef TEST_MODE
    allocation_waits.fetch_add(1, std::memory_order_relaxed);
    struct WaitTimer
    {
        ConcurrentMemoryPool& pool;
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        ~WaitTimer()
        {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
            const size_t bucket = std::min<size_t>(63, std::bit_width(static_cast<uint64_t>(ns)));
            pool.wait_latency_buckets[bucket].fetch_add(1, std::memory_order_relaxed);
        }
    } timer{*this};
#endif
    try {
        for (;;) {
            state_->check();
            const uint64_t progress = state_->progress();
            if (admitted >= 0) {
                Arena& arena = arenas_[admitted];
                char* result = nullptr;
                {
                    std::lock_guard lock(arena.admission);
                    result = arena.reserved_result;
                    if (result) {
                        arena.reserved_result = nullptr;
                        arena.pending_bytes = 0;
                        update_pressure(arena);
                    }
                }
                if (result) { state_->notify(); return result; }
                if (!spill_service_.load(std::memory_order_acquire)) {
                    // Recheck under admission: a retirement may just have
                    // fulfilled the reservation. Never strand a RAM extent.
                    {
                        std::lock_guard lock(arena.admission);
                        result = arena.reserved_result;
                        arena.reserved_result = nullptr;
                        arena.pending_bytes = 0;
                        update_pressure(arena);
                    }
                    admitted = -1;
                    state_->notify();
                    if (result) return result;
                    if (preferred >= 0) throw std::bad_alloc();
                    break;
                }
            } else {
                // Existing free-list/TLS pages remain available while bump space
                // is reserved for someone else. A one-page request can use them.
                if (preferred < 0 && bytes == BLOCK_SIZE)
                    if (void* result = allocate_small(false)) return result;
                bool possible = false;
                for (int rank = 0; rank < num_arenas_; ++rank) {
                    const int candidate = preferred >= 0 ? preferred : numa_distance_order_[local][rank];
                    Arena& arena = arenas_[candidate];
                    {
                        std::lock_guard lock(arena.admission);
                        const uint64_t maximum = arena.capacity - arena.tree_offset;
                        if (bytes <= maximum) {
                            possible = true;
                            if (!arena.pending_bytes) {
                                const uint64_t offset = arena.tree_offset;
                                if (arena.reserve_tree(bytes)) {
                                    update_pressure(arena);
                                    return static_cast<char*>(arena.start_addr) + offset;
                                }
                                if (spill_service_.load(std::memory_order_acquire)) {
                                    arena.pending_bytes = bytes;
                                    arena.reserved_result = nullptr;
                                    admitted = candidate;
                                    update_pressure(arena);
                                }
                            }
                        }
                    }
                    if (admitted >= 0 || preferred >= 0) break;
                }
                // Always consume/cancel an admitted reservation in the branch
                // above, even if the service stopped just after admission.
                if (admitted >= 0) { state_->notify(); continue; }
                if (!possible || !spill_service_.load(std::memory_order_acquire)) {
                    if (preferred >= 0) throw std::bad_alloc();
                    break;
                }
            }
            state_->wait(progress);
        }
    } catch (...) {
        if (admitted >= 0) {
            Arena& arena = arenas_[admitted];
            {
                std::lock_guard lock(arena.admission);
                // A reserved bump extent cannot be rolled back after publication;
                // on failure the entire run aborts and the mapping is destroyed.
                arena.pending_bytes = 0;
                arena.reserved_result = nullptr;
                update_pressure(arena);
            }
            state_->notify();
        }
        throw;
    }
    state_->check();
    return allocate_file(bytes, bytes == BLOCK_SIZE);
}

[[gnu::noinline]] inline void* ConcurrentMemoryPool::allocate_file(size_t bytes, bool small)
{
    ThreadLocalCache& tls = tls_cache_;
    if (small && tls.file_free_stack) {
        auto* result = tls.file_free_stack;
        tls.file_free_stack = result->next;
        --tls.file_free_count;
        return result;
    }
    for (;;) {
        state_->check();
        const auto progress = state_->progress();
        std::unique_lock lock(file_mutex_);
        FreeBlock* head = nullptr;
        size_t count = 0;
        const size_t batch = tls.file_started ? BATCH_SIZE : TLS_CAPACITY;
        if (small && file_free_list_) {
            head = file_free_list_;
            auto* tail = head;
            count = 1;
            while (count < batch && tail->next) { tail = tail->next; ++count; }
            file_free_list_ = tail->next;
            tail->next = nullptr;
        } else {
            for (auto& segment : file_segments_) {
                if (bytes > segment.committed - segment.used) continue;
                char* result = segment.base + segment.used;
                if (!small) { segment.used += bytes; return result; }
                count = std::min(batch, (segment.committed - segment.used) / BLOCK_SIZE);
                segment.used += count * BLOCK_SIZE;
                head = reinterpret_cast<FreeBlock*>(result);
                // Initialize only the pages in this refill, never the mapping.
                auto* tail = head;
                for (size_t i = 1; i < count; ++i) {
                    tail->next = reinterpret_cast<FreeBlock*>(result + i * BLOCK_SIZE);
                    tail = tail->next;
                }
                tail->next = nullptr;
                break;
            }
        }
        if (head) {
            tls.file_started = true;
            tls.file_free_stack = head->next;
            tls.file_free_count = count - 1;
            return head;
        }
        if (file_expanding_) {
            lock.unlock();
            state_->wait(progress);
            continue;
        }
        // Copy descriptors across unlocks; directory insertion can move them.
        const bool new_segment = file_current_ == SIZE_MAX
            || bytes > file_segments_[file_current_].reserved - file_segments_[file_current_].used;
        FileSegment segment = new_segment ? FileSegment{nullptr, 0, 0, 0, file_end_}
                                         : file_segments_[file_current_];
        const size_t directory_size = file_segments_.size();
        const size_t growth = file_growth_;
        int fd = file_fd_;
        file_expanding_ = true;
        lock.unlock();
        bool opened = false;
        std::vector<FileSegment> directory;
        try {
            if (fd < 0) {
                const long page = ::sysconf(_SC_PAGESIZE);
                if (page <= 0 || BLOCK_SIZE % static_cast<size_t>(page) != 0 || file_path_.empty())
                    throw std::bad_alloc();
                fd = ::open(file_path_.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
                if (fd < 0) throw std::bad_alloc();
                opened = true;
                if (::unlink(file_path_.c_str()) != 0) throw std::bad_alloc();
            }
            if (new_segment) {
                if (bytes > SIZE_MAX - (FILE_SEGMENT_BYTES - 1)) throw std::bad_alloc();
                segment.reserved = align_up(bytes, FILE_SEGMENT_BYTES);
                directory.reserve(directory_size + 1);
                void* base = file_map_(nullptr, segment.reserved, PROT_NONE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (base == MAP_FAILED) throw std::bad_alloc();
                segment.base = static_cast<char*>(base);
#ifdef TEST_MODE
                file_reservations.fetch_add(1, std::memory_order_relaxed);
#endif
            }
            const size_t missing = bytes - (segment.committed - segment.used);
            const size_t remaining = segment.reserved - segment.committed;
            if (missing > SIZE_MAX - (growth - 1)) throw std::bad_alloc();
            const size_t length = std::min(align_up(missing, growth), remaining);
            const auto offset_max = static_cast<size_t>(std::numeric_limits<off_t>::max());
            if (segment.file_offset > offset_max || segment.committed > offset_max - segment.file_offset)
                throw std::bad_alloc();
            const size_t offset = segment.file_offset + segment.committed;
            if (length > offset_max - offset) throw std::bad_alloc();
#ifdef TEST_MODE
            file_preallocations.fetch_add(1, std::memory_order_relaxed);
#endif
            if (file_preallocate_(fd, 0, static_cast<off_t>(offset), static_cast<off_t>(length)) != 0)
                throw std::bad_alloc();
#ifdef TEST_MODE
            file_mappings.fetch_add(1, std::memory_order_relaxed);
#endif
            if (file_map_(segment.base + segment.committed, length, PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_FIXED, fd, static_cast<off_t>(offset)) == MAP_FAILED)
                throw std::bad_alloc();
            segment.committed += length;
            lock.lock();
            if (new_segment) {
                // Capacity was allocated outside the lock. Copy the latest used
                // offsets, which other allocators may have advanced during I/O.
                directory.insert(directory.end(), file_segments_.begin(), file_segments_.end());
                const auto position = std::lower_bound(directory.begin(), directory.end(),
                    reinterpret_cast<uintptr_t>(segment.base), [](const FileSegment& item, uintptr_t address) {
                        return reinterpret_cast<uintptr_t>(item.base) < address;
                    });
                file_current_ = static_cast<size_t>(position - directory.begin());
                directory.insert(position, segment);
                file_segments_.swap(directory);
            } else {
                file_segments_[file_current_].committed = segment.committed;
            }
            file_end_ = offset + length;
            file_growth_ = std::min(growth * 2, FILE_MAX_GROWTH);
            file_fd_ = fd;
            file_expanding_ = false;
#ifdef TEST_MODE
            file_expansions.fetch_add(1, std::memory_order_relaxed);
            file_committed_bytes.fetch_add(length, std::memory_order_relaxed);
            if (new_segment) file_segment_count.fetch_add(1, std::memory_order_relaxed);
#endif
            lock.unlock();
            state_->notify();
        } catch (...) {
            if (new_segment && segment.base) ::munmap(segment.base, segment.reserved);
            if (opened) ::close(fd);
            if (!lock.owns_lock()) lock.lock();
            file_expanding_ = false;
            lock.unlock();
            state_->notify();
            state_->check();
            throw std::bad_alloc();
        }
    }
}

[[gnu::noinline]] inline void ConcurrentMemoryPool::deallocate_file(void* ptr)
{
    const auto address = reinterpret_cast<uintptr_t>(ptr);
    auto& tls = tls_cache_;
    if (address < tls.file_range_begin || address - tls.file_range_begin >= tls.file_range_bytes) {
        std::lock_guard lock(file_mutex_);
        auto position = std::upper_bound(file_segments_.begin(), file_segments_.end(), address,
            [](uintptr_t value, const FileSegment& segment) { return value < reinterpret_cast<uintptr_t>(segment.base); });
        if (position == file_segments_.begin()) return;
        --position;
        if (address - reinterpret_cast<uintptr_t>(position->base) >= position->committed) return;
        // Committed ranges remain mapped until pool destruction; entries can move.
        tls.file_range_begin = reinterpret_cast<uintptr_t>(position->base);
        tls.file_range_bytes = position->committed;
    }
    if (tls.file_free_count == TLS_CAPACITY) {
        auto* tail = tls.file_free_stack;
        for (size_t i = 1; i < TLS_CAPACITY / 2; ++i) tail = tail->next;
        auto* remaining = tail->next;
        {
            std::lock_guard lock(file_mutex_);
            tail->next = file_free_list_;
            file_free_list_ = tls.file_free_stack;
            tls.file_free_stack = remaining;
            tls.file_free_count -= TLS_CAPACITY / 2;
        }
        state_->notify();
    }
    auto* block = static_cast<FreeBlock*>(ptr);
    block->next = tls.file_free_stack;
    tls.file_free_stack = block;
    ++tls.file_free_count;
}

inline int ConcurrentMemoryPool::get_current_arena_index() const
{
    return get_thread_arena_index();
}

#endif // CONCURRENT_MEMORY_POOL_HEADER
