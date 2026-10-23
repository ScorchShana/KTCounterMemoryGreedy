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
#include <span>
#include <cassert>

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
enum class FirstSegmentState : uint64_t { Retired, Filling, Sealed, Spilling, Scanning, Done };

struct FirstSegmentRef
{
    uint32_t arena_index = 0;
    uint64_t segment_index = 0;
    uint64_t generation = 0;
};

struct alignas(CACHE_LINE_SIZE) FirstSegmentDescriptor
{
    static constexpr uint64_t MAX_GENERATION = UINT64_MAX >> 3;
    std::atomic<uint64_t> tag{0};
    uint64_t record_count = 0; // Published by Sealed; immutable until next Filling.
    char* data = nullptr; // Fixed slot address, initialized before workers start.
    static constexpr uint64_t make_tag(uint64_t generation, FirstSegmentState state) noexcept
    { return (generation << 3) | static_cast<uint64_t>(state); }
    static constexpr FirstSegmentState state(uint64_t tag) noexcept
    { return static_cast<FirstSegmentState>(tag & 7); }
};
static_assert(sizeof(FirstSegmentDescriptor) == CACHE_LINE_SIZE);
static_assert(std::atomic<uint64_t>::is_always_lock_free);

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
    enum class Mode : uint64_t { Grow, Closing, Reclaim, Resetting };
    enum class Request : uint8_t { Idle, Waiting, Granted, Impossible };
    static constexpr uint64_t PAGE_MASK = (1ULL << 52) - 1;
    static constexpr uint64_t ACTIVE_UNIT = 1ULL << 52;
    static constexpr uint64_t ACTIVE_MASK = 511ULL << 52;
    static constexpr uint64_t PRIORITY = 1ULL << 61;
    static constexpr uint64_t MODE_MASK = 3ULL << 62;
    static constexpr uint64_t SEGMENT_PAGES = FIRST_SEGMENT_SIZE / BLOCK_SIZE;
#ifdef TEST_MODE
    // Deterministic scheduling points; absent from production builds.
    inline static void (*after_tree_charge)() = nullptr;
    inline static void (*after_first_charge)() = nullptr;
    inline static void (*after_frontier_advance)() = nullptr;
    inline static void (*before_pending_admission)() = nullptr;
    inline static void (*before_claim_cas)() = nullptr;
    inline static void (*after_done)() = nullptr;
    inline static void (*after_completion_sequence)() = nullptr;
#endif
    static constexpr Mode mode(uint64_t word) noexcept { return static_cast<Mode>(word >> 62); }
    static constexpr uint64_t with_mode(uint64_t word, Mode next) noexcept
    { return (word & ~MODE_MASK) | (static_cast<uint64_t>(next) << 62); }

    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> admission_word{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> tree_offset{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> first_offset{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> claim_offset{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> completion_sequence{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> scan_reclaim_goal_pages{0};
    alignas(CACHE_LINE_SIZE) std::mutex control_mutex;
    std::atomic<Request> request{Request::Idle};
    size_t pending_bytes = 0;
    char* reserved_result = nullptr;
    bool permanently_frozen = false;
    bool explicit_drain = false;
    bool hard_reclaim = false;
    std::atomic<MemoryPressure> pressure_hint{MemoryPressure::Normal};
    std::atomic<uint64_t> pressure_generation{0};
    alignas(CACHE_LINE_SIZE) FreeBlock* central_free_list = nullptr;
    std::mutex mutex;
    alignas(CACHE_LINE_SIZE) void* start_addr = nullptr;
    void* end_addr = nullptr;
    uint64_t capacity = 0;
    size_t low = 0, high = 0;
    std::unique_ptr<FirstSegmentDescriptor[]> segments;
    size_t segment_capacity = 0;

    void initialize(uint64_t bytes) noexcept
    {
        capacity = bytes;
        tree_offset.store(0, std::memory_order_relaxed);
        first_offset.store(bytes, std::memory_order_relaxed);
        claim_offset.store(bytes, std::memory_order_relaxed);
        admission_word.store(bytes / BLOCK_SIZE, std::memory_order_relaxed);
    }

    // Credit is charged before address assignment. The cursor RMW carries the
    // reclaimed-prefix synchronization when earlier credit holders arrive late.
    bool try_reserve_tree(uint64_t bytes, uint64_t& offset, bool pending = false) noexcept
    {
        const uint64_t pages = bytes / BLOCK_SIZE;
        uint64_t word = admission_word.load(std::memory_order_acquire);
        for (;;) {
            if (!pages || (word & PAGE_MASK) < pages || (!pending && (word & PRIORITY))) return false;
            uint64_t next = word - pages;
            if (mode(word) == Mode::Grow && (next & PAGE_MASK) <= low / BLOCK_SIZE)
                next = with_mode(next, Mode::Closing);
            if (admission_word.compare_exchange_weak(word, next, std::memory_order_acq_rel, std::memory_order_acquire)) break;
        }
#ifdef TEST_MODE
        if (after_tree_charge) after_tree_charge();
#endif
        offset = tree_offset.fetch_add(bytes, std::memory_order_acq_rel);
        return true;
    }

    bool try_begin_first(uint64_t& offset) noexcept
    {
        uint64_t word = admission_word.load(std::memory_order_acquire);
        for (;;) {
            if (mode(word) != Mode::Grow || (word & PRIORITY)
                || (word & PAGE_MASK) < SEGMENT_PAGES + high / BLOCK_SIZE) return false;
            if ((word & ACTIVE_MASK) == ACTIVE_MASK) { word = admission_word.load(std::memory_order_acquire); continue; }
            const uint64_t next = word - SEGMENT_PAGES + ACTIVE_UNIT;
            if (admission_word.compare_exchange_weak(word, next, std::memory_order_acq_rel, std::memory_order_acquire)) break;
        }
#ifdef TEST_MODE
        if (after_first_charge) after_first_charge();
#endif
        offset = first_offset.fetch_sub(FIRST_SEGMENT_SIZE, std::memory_order_relaxed) - FIRST_SEGMENT_SIZE;
        return true;
    }

    bool enter_reclaim() noexcept
    {
        uint64_t word = admission_word.load(std::memory_order_acquire);
        for (;;) {
            if (mode(word) != Mode::Reclaim || (word & ACTIVE_MASK) == ACTIVE_MASK) return false;
            if (admission_word.compare_exchange_weak(word, word + ACTIVE_UNIT,
                std::memory_order_acq_rel, std::memory_order_acquire)) return true;
        }
    }
    void leave_short() noexcept { admission_word.fetch_sub(ACTIVE_UNIT, std::memory_order_release); }
    void set_mode(Mode next) noexcept
    {
        uint64_t word = admission_word.load(std::memory_order_acquire);
        while (!admission_word.compare_exchange_weak(word, with_mode(word, next),
            std::memory_order_acq_rel, std::memory_order_acquire)) {}
    }
    bool contains(void* ptr) const { return ptr >= start_addr && ptr < end_addr; }
    size_t block_count() const { return capacity / BLOCK_SIZE; }
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
    FirstSegmentRef first_ref{};
    unsigned char* first_data = nullptr;
    size_t first_count = 0;
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
    enum class FirstConsumer { Spill, Scan };
    enum class FirstClaimStatus { Data, Busy, Done };
    struct FirstSegmentView {
        FirstSegmentRef ref;
        const unsigned char* records = nullptr;
        size_t record_count = 0;
        FirstConsumer consumer = FirstConsumer::Scan;
    };
    FirstClaimStatus try_claim_first(int preferred, FirstConsumer consumer, FirstSegmentView& out) noexcept;
    bool complete_first(const FirstSegmentView& view) noexcept;
    void poll_first_progress(int index) noexcept;
    bool cache_first_record(int arena, const unsigned char* record) noexcept;
    void seal_thread_first() noexcept;
    void seal_thread_first_if_reclaiming() noexcept;
    size_t thread_first_records() const noexcept { return tls_cache_.pool == this ? tls_cache_.first_count : 0; }
    enum class RamReserveStatus { Ready, Busy, NeedReclaim };
    struct RamReserveResult { RamReserveStatus status; size_t acquired; };
    RamReserveResult try_reserve_ram_pages(std::span<void*> out) noexcept;
    void release_ram_pages(std::span<void* const> pages) noexcept;

    MemoryPressure pressure(int index) const noexcept
    {
        const Arena& arena = arenas_[index];
        const uint64_t word = arena.admission_word.load(std::memory_order_acquire);
        if ((word & Arena::PRIORITY) || Arena::mode(word) != Arena::Mode::Grow) return MemoryPressure::Hard;
        const uint64_t gap = (word & Arena::PAGE_MASK) * BLOCK_SIZE;
        return gap >= arena.high ? MemoryPressure::Normal : gap <= arena.low ? MemoryPressure::Hard : MemoryPressure::Soft;
    }
    uint64_t pressure_generation(int index) const noexcept { return arenas_[index].pressure_generation.load(std::memory_order_acquire); }
    size_t segment_capacity() const noexcept { return memory_size_ / FIRST_SEGMENT_SIZE; }
    size_t segment_capacity(int arena) const noexcept { return arenas_[arena].segment_capacity; }
    int arena_count() const noexcept { return num_arenas_; }
    int arena_node(int index) const noexcept { return node_ids_[index]; }
    int arena_for_node(int node) const noexcept;
    RunState& run_state() noexcept { return *state_; }
    void set_spill_service(bool running) { spill_service_.store(running, std::memory_order_release); }
    void freeze_first();
#ifdef TEST_MODE
    std::atomic<int64_t> drain_started_ns{0};
    std::atomic<int64_t> input_finished_ns{0};
    std::atomic<uint64_t> spill_at_drain{0};
#endif
    void request_first_drain();
    bool first_empty() noexcept;
    size_t gap_bytes(int index) noexcept;
#ifdef TEST_MODE
    std::array<std::atomic<uint64_t>, 64> wait_latency_buckets{};
    std::atomic<uint64_t> spill_bytes{0}, returned_bytes{0}, allocation_waits{0};
    std::atomic<uint64_t> first_publications{0}, first_valid_records{0}, first_unused_bytes{0}, spill_claims{0};
    std::atomic<uint64_t> first_rejected_capacity{0}, first_rejected_tree{0}, first_rejected_frozen{0};
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
    std::atomic<bool> spill_service_{false}, first_frozen_{false};
    int node_ids_[MAX_NUMA_NODES]{};
    void update_pressure(Arena& arena) noexcept; // control_mutex held
    bool begin_first_segment(int index, FirstSegmentRef& ref, unsigned char*& records) noexcept;
    void seal_first_segment(const FirstSegmentRef& ref, size_t count) noexcept;
    void publish_first_prefix(Arena& arena) noexcept; // Reclaim short protection or exclusive Resetting
    void service_pending(Arena& arena) noexcept; // control_mutex held
    FirstClaimStatus claim_first_in_arena(int index, FirstConsumer consumer, FirstSegmentView& out) noexcept;
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

    pool->seal_thread_first();
    // Pool changes are cold, but try allocation must not wait while flushing
    // an old pool. Acquire everything before modifying the TLS lists.
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

        tls.first_data = nullptr;
        tls.first_count = 0;
        tls.first_ref = {};
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
        arena.initialize(arena_bytes);
        arena.low = arena_watermark(arena_bytes, 5);
        arena.high = arena_watermark(arena_bytes, 8);
        arena.segment_capacity = arena_bytes / FIRST_SEGMENT_SIZE;
        if (arena.segment_capacity > SIZE_MAX / sizeof(FirstSegmentDescriptor)) throw std::bad_alloc();
        arena.segments = std::make_unique<FirstSegmentDescriptor[]>(arena.segment_capacity);
        for (size_t slot = 0; slot < arena.segment_capacity; ++slot)
            arena.segments[slot].data = current_addr + slot * FIRST_SEGMENT_SIZE;
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
    const uint64_t word = arena.admission_word.load(std::memory_order_acquire);
    const uint64_t gap = (word & Arena::PAGE_MASK) * BLOCK_SIZE;
    if (word & Arena::PRIORITY) arena.hard_reclaim = true;
    else if (gap >= arena.high) arena.hard_reclaim = false;
    else if (gap <= arena.low) arena.hard_reclaim = true;
    const MemoryPressure next = (word & Arena::PRIORITY) ? MemoryPressure::Hard
        : gap >= arena.high ? MemoryPressure::Normal : gap <= arena.low ? MemoryPressure::Hard : MemoryPressure::Soft;
    if (arena.pressure_hint.load(std::memory_order_relaxed) != next) {
        arena.pressure_hint.store(next, std::memory_order_release);
        arena.pressure_generation.fetch_add(1, std::memory_order_release);
    }
}

inline char* ConcurrentMemoryPool::bump_allocate_from_arena(Arena& arena, size_t bytes, bool /*try_only*/)
{
    if (!bytes || bytes > SIZE_MAX - (BLOCK_SIZE - 1)) return nullptr;
    uint64_t offset;
    if (!arena.try_reserve_tree(align_up(bytes, BLOCK_SIZE), offset)) return nullptr;
    return static_cast<char*>(arena.start_addr) + offset;
}

inline FreeBlock* ConcurrentMemoryPool::batch_allocate_from_bump(Arena& arena, size_t& out_count, FreeBlock** out_tail, bool try_only)
{
    const uint64_t available = arena.admission_word.load(std::memory_order_acquire) & Arena::PAGE_MASK;
    out_count = std::min<uint64_t>(out_count, available);
    if (!out_count) return nullptr;
    char* cursor = bump_allocate_from_arena(arena, out_count * BLOCK_SIZE, try_only);
    if (!cursor) return nullptr;
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
    return (arenas_[index].admission_word.load(std::memory_order_acquire) & Arena::PAGE_MASK) * BLOCK_SIZE;
}

inline bool ConcurrentMemoryPool::begin_first_segment(int index, FirstSegmentRef& ref, unsigned char*& records) noexcept
{
    Arena& arena = arenas_[index];
    uint64_t offset;
    if (!arena.try_begin_first(offset)) {
#ifdef TEST_MODE
        const auto word = arena.admission_word.load(std::memory_order_relaxed);
        if (first_frozen_.load(std::memory_order_relaxed)) first_rejected_frozen.fetch_add(1, std::memory_order_relaxed);
        else if (word & Arena::PRIORITY) first_rejected_tree.fetch_add(1, std::memory_order_relaxed);
        else first_rejected_capacity.fetch_add(1, std::memory_order_relaxed);
#endif
        return false;
    }
    const uint64_t slot = offset / FIRST_SEGMENT_SIZE;
    auto& descriptor = arena.segments[slot];
    const uint64_t generation = descriptor.tag.load(std::memory_order_acquire) >> 3;
    if (generation == FirstSegmentDescriptor::MAX_GENERATION) {
        arena.leave_short();
        try { throw std::overflow_error("FIRST generation exhausted"); }
        catch (...) { state_->fail(std::current_exception(), "FIRST generation"); }
        return false;
    }
    descriptor.tag.store(FirstSegmentDescriptor::make_tag(generation + 1, FirstSegmentState::Filling), std::memory_order_relaxed);
    ref = {static_cast<uint32_t>(index), slot, generation + 1};
    records = reinterpret_cast<unsigned char*>(descriptor.data);
    arena.leave_short();
    return true;
}

inline void ConcurrentMemoryPool::seal_first_segment(const FirstSegmentRef& ref, size_t count) noexcept
{
    auto& descriptor = arenas_[ref.arena_index].segments[ref.segment_index];
    descriptor.record_count = count;
    descriptor.tag.store(FirstSegmentDescriptor::make_tag(ref.generation, FirstSegmentState::Sealed), std::memory_order_release);
#ifdef TEST_MODE
    first_publications.fetch_add(1, std::memory_order_relaxed);
    first_valid_records.fetch_add(count, std::memory_order_relaxed);
    first_unused_bytes.fetch_add(FIRST_SEGMENT_SIZE - count * kmer_bytes(), std::memory_order_relaxed);
#endif
}

inline bool ConcurrentMemoryPool::cache_first_record(int index, const unsigned char* record) noexcept
{
    if (!prepare_tls(false)) return false;
    auto& tls = tls_cache_;
    if (!tls.pool) {
        tls.pool = this;
        tls.local_arena_index = get_thread_arena_index();
        tls.local_arena = &arenas_[tls.local_arena_index];
    }
    if (!tls.first_data && !begin_first_segment(index, tls.first_ref, tls.first_data)) return false;
    std::memcpy(tls.first_data + tls.first_count * kmer_bytes(), record, kmer_bytes());
    if (++tls.first_count == first_segment_record_capacity()) seal_thread_first();
    return true;
}

inline void ConcurrentMemoryPool::seal_thread_first() noexcept
{
    auto& tls = tls_cache_;
    if (tls.pool != this || !tls.first_data) return;
    const FirstSegmentRef ref = tls.first_ref;
    const size_t count = tls.first_count;
    tls.first_data = nullptr;
    tls.first_count = 0;
    tls.first_ref = {};
    seal_first_segment(ref, count);
}

inline void ConcurrentMemoryPool::seal_thread_first_if_reclaiming() noexcept
{
    const auto& tls = tls_cache_;
    if (tls.pool != this || !tls.first_data) return;
    // Soft pressure does not need this already charged segment. Only its
    // owner publishes it, including when reclamation starts during a wait.
    const uint64_t word = arenas_[tls.first_ref.arena_index].admission_word.load(std::memory_order_acquire);
    if ((word & Arena::PRIORITY) || Arena::mode(word) != Arena::Mode::Grow) seal_thread_first();
}

inline ConcurrentMemoryPool::RamReserveResult ConcurrentMemoryPool::try_reserve_ram_pages(std::span<void*> out) noexcept
{
    if (!prepare_tls(true)) return {RamReserveStatus::Busy, 0};
    auto& tls = tls_cache_;
    if (!tls.pool) {
        tls.pool = this;
        tls.local_arena_index = get_thread_arena_index();
        tls.local_arena = &arenas_[tls.local_arena_index];
    }
    size_t acquired = 0;
    while (acquired < out.size() && tls.local_free_stack) {
        auto* page = tls.local_free_stack;
        tls.local_free_stack = page->next;
        --tls.local_free_count;
        out[acquired++] = page;
    }
    for (int rank = 0; acquired < out.size() && rank < num_arenas_; ++rank) {
        auto& remote = tls.remote_lists[numa_distance_order_[tls.local_arena_index][rank]];
        while (acquired < out.size() && remote.head) {
            auto* page = remote.head;
            remote.head = page->next;
            --remote.count;
            out[acquired++] = page;
        }
    }
    bool busy = false;
    for (int rank = 0; acquired < out.size() && rank < num_arenas_; ++rank) {
        Arena& arena = arenas_[numa_distance_order_[tls.local_arena_index][rank]];
        {
            std::unique_lock central(arena.mutex, std::try_to_lock);
            if (!central) { busy = true; continue; }
            while (acquired < out.size() && arena.central_free_list) {
                auto* page = arena.central_free_list;
                arena.central_free_list = page->next;
                out[acquired++] = page;
            }
        }
        if (acquired == out.size()) break;
        const uint64_t word = arena.admission_word.load(std::memory_order_acquire);
        if (word & Arena::PRIORITY) { busy = true; continue; }
        const size_t count = std::min<uint64_t>(out.size() - acquired, word & Arena::PAGE_MASK);
        uint64_t offset;
        if (count && arena.try_reserve_tree(count * BLOCK_SIZE, offset)) {
            for (size_t i = 0; i < count; ++i) out[acquired++] = static_cast<char*>(arena.start_addr) + offset + i * BLOCK_SIZE;
        } else if (count) busy = true;
    }
    if (acquired == out.size()) {
        for (int i = 0; i < num_arenas_; ++i) {
            uint64_t goal = arenas_[i].scan_reclaim_goal_pages.load(std::memory_order_relaxed);
            if (goal) arenas_[i].scan_reclaim_goal_pages.compare_exchange_strong(goal, 0, std::memory_order_release, std::memory_order_relaxed);
        }
        return {RamReserveStatus::Ready, acquired};
    }
    if (busy) return {RamReserveStatus::Busy, acquired};
    // This is a progress hint only. Other scanners retry their true deficit;
    // clearing/coalescing this hint never authorizes memory reuse or completion.
    for (int rank = 0; rank < num_arenas_; ++rank) {
        auto& arena = arenas_[numa_distance_order_[tls.local_arena_index][rank]];
        if (arena.first_offset.load(std::memory_order_acquire) == arena.capacity) continue;
        uint64_t goal = arena.scan_reclaim_goal_pages.load(std::memory_order_relaxed);
        const uint64_t need = out.size() - acquired;
        while (goal < need && !arena.scan_reclaim_goal_pages.compare_exchange_weak(goal, need,
            std::memory_order_release, std::memory_order_relaxed)) {}
        break;
    }
    return {RamReserveStatus::NeedReclaim, acquired};
}

inline void ConcurrentMemoryPool::release_ram_pages(std::span<void* const> pages) noexcept
{
    for (void* page : pages) deallocate(page);
}

inline void ConcurrentMemoryPool::freeze_first()
{
    for (int i = 0; i < num_arenas_; ++i) {
        auto& arena = arenas_[i];
        {
            std::lock_guard lock(arena.control_mutex);
            arena.permanently_frozen = true;
            const auto word = arena.admission_word.load(std::memory_order_acquire);
            if (Arena::mode(word) == Arena::Mode::Grow) arena.set_mode(Arena::Mode::Closing);
        }
        poll_first_progress(i);
    }
    first_frozen_.store(true, std::memory_order_release);
}

inline void ConcurrentMemoryPool::request_first_drain()
{
#ifdef TEST_MODE
    drain_started_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count(), std::memory_order_release);
    spill_at_drain.store(spill_bytes.load(std::memory_order_relaxed), std::memory_order_release);
#endif
    for (int i = 0; i < num_arenas_; ++i) {
        auto& arena = arenas_[i];
        {
            std::lock_guard lock(arena.control_mutex);
            arena.explicit_drain = true;
            if (Arena::mode(arena.admission_word.load(std::memory_order_acquire)) == Arena::Mode::Grow)
                arena.set_mode(Arena::Mode::Closing);
        }
        poll_first_progress(i);
    }
}

inline void ConcurrentMemoryPool::publish_first_prefix(Arena& arena) noexcept
{
    uint64_t first = arena.first_offset.load(std::memory_order_acquire);
    for (;;) {
        const uint64_t limit = arena.claim_offset.load(std::memory_order_acquire);
        uint64_t end = first;
        while (end < limit && FirstSegmentDescriptor::state(arena.segments[end / FIRST_SEGMENT_SIZE].tag.load(
            std::memory_order_acquire)) == FirstSegmentState::Done) end += FIRST_SEGMENT_SIZE;
        if (end == first) return;
        // Acquire the preceding publisher as credit can be spent on any part of
        // the continuous prefix, including its earlier, not-yet-refunded part.
        if (arena.first_offset.compare_exchange_weak(first, end, std::memory_order_acq_rel, std::memory_order_acquire)) {
#ifdef TEST_MODE
            if (Arena::after_frontier_advance) Arena::after_frontier_advance();
#endif
            arena.admission_word.fetch_add((end - first) / BLOCK_SIZE, std::memory_order_release);
#ifdef TEST_MODE
            returned_bytes.fetch_add(end - first, std::memory_order_relaxed);
#endif
            return;
        }
    }
}

inline void ConcurrentMemoryPool::service_pending(Arena& arena) noexcept
{
    if (arena.request.load(std::memory_order_relaxed) != Arena::Request::Waiting) return;
    uint64_t offset;
    if (arena.try_reserve_tree(arena.pending_bytes, offset, true)) {
        arena.reserved_result = static_cast<char*>(arena.start_addr) + offset;
        arena.request.store(Arena::Request::Granted, std::memory_order_release);
        arena.admission_word.fetch_and(~Arena::PRIORITY, std::memory_order_acq_rel);
    } else {
        // Read the frontier before the word: an acquire frontier publication
        // also makes its earlier short-operation entry visible in the word.
        const uint64_t first = arena.first_offset.load(std::memory_order_acquire);
        const uint64_t word = arena.admission_word.load(std::memory_order_acquire);
        if (first == arena.capacity && !(word & Arena::ACTIVE_MASK)
            && (word & Arena::PAGE_MASK) < arena.pending_bytes / BLOCK_SIZE) {
            arena.request.store(Arena::Request::Impossible, std::memory_order_release);
            arena.admission_word.fetch_and(~Arena::PRIORITY, std::memory_order_acq_rel);
        }
    }
}

inline void ConcurrentMemoryPool::poll_first_progress(int index) noexcept
{
    Arena& arena = arenas_[index];
    std::unique_lock control(arena.control_mutex, std::try_to_lock);
    if (!control) return;
    update_pressure(arena);
    uint64_t word = arena.admission_word.load(std::memory_order_acquire);
    auto mode = Arena::mode(word);
    if (mode == Arena::Mode::Grow && (arena.permanently_frozen || arena.explicit_drain || arena.hard_reclaim
        || (word & Arena::PRIORITY) || arena.scan_reclaim_goal_pages.load(std::memory_order_acquire))) {
        arena.set_mode(Arena::Mode::Closing);
        mode = Arena::Mode::Closing;
    }
    if (mode == Arena::Mode::Closing) {
        if (arena.admission_word.load(std::memory_order_acquire) & Arena::ACTIVE_MASK) return;
        arena.claim_offset.store(arena.first_offset.load(std::memory_order_acquire), std::memory_order_relaxed);
        arena.set_mode(Arena::Mode::Reclaim);
        mode = Arena::Mode::Reclaim;
    }
    if (mode == Arena::Mode::Reclaim) {
        if (arena.enter_reclaim()) { publish_first_prefix(arena); arena.leave_short(); }
        service_pending(arena);
        word = arena.admission_word.load(std::memory_order_acquire);
        const bool empty = arena.first_offset.load(std::memory_order_acquire) == arena.capacity;
        if (empty || (!arena.permanently_frozen && !(word & Arena::PRIORITY)
            && (word & Arena::PAGE_MASK) >= arena.high / BLOCK_SIZE
            && arena.claim_offset.load(std::memory_order_acquire) == arena.first_offset.load(std::memory_order_acquire))) {
            arena.set_mode(Arena::Mode::Resetting);
            mode = Arena::Mode::Resetting;
        }
    }
    if (mode == Arena::Mode::Resetting) {
        if (arena.admission_word.load(std::memory_order_acquire) & Arena::ACTIVE_MASK) return;
        publish_first_prefix(arena);
        service_pending(arena);
        if (arena.first_offset.load(std::memory_order_acquire) == arena.capacity) {
            arena.explicit_drain = false;
            arena.scan_reclaim_goal_pages.store(0, std::memory_order_release);
        }
        word = arena.admission_word.load(std::memory_order_acquire);
        for (;;) {
            const bool grow = !arena.permanently_frozen && !arena.explicit_drain && !(word & Arena::PRIORITY)
                && (word & Arena::PAGE_MASK) >= arena.high / BLOCK_SIZE
                && arena.claim_offset.load(std::memory_order_relaxed) == arena.first_offset.load(std::memory_order_relaxed);
            const uint64_t next = Arena::with_mode(word, grow ? Arena::Mode::Grow : Arena::Mode::Reclaim);
            if (arena.admission_word.compare_exchange_weak(word, next, std::memory_order_acq_rel, std::memory_order_acquire)) break;
        }
    }
    update_pressure(arena);
}

inline ConcurrentMemoryPool::FirstClaimStatus ConcurrentMemoryPool::claim_first_in_arena(
    int index, FirstConsumer consumer, FirstSegmentView& out) noexcept
{
    Arena& arena = arenas_[index];
    poll_first_progress(index);
    std::unique_lock control(arena.control_mutex, std::defer_lock);
    if (consumer == FirstConsumer::Spill) {
        if (!control.try_lock()) return FirstClaimStatus::Busy;
        update_pressure(arena);
    }
    if (!arena.enter_reclaim()) return FirstClaimStatus::Busy;
    bool claimed = false;
    uint64_t cursor = arena.claim_offset.load(std::memory_order_acquire);
    for (;;) {
        if (cursor >= arena.capacity) break;
        if (consumer == FirstConsumer::Spill && !arena.explicit_drain) {
            const uint64_t word = arena.admission_word.load(std::memory_order_acquire);
            // Frozen FIRST cannot grow again. Refill hysteresis only serves
            // input admission; scanning needs spill only for an actual deficit.
            uint64_t goal = arena.hard_reclaim && !arena.permanently_frozen ? arena.high / BLOCK_SIZE : 0;
            if (arena.request.load(std::memory_order_relaxed) == Arena::Request::Waiting)
                goal = std::max<uint64_t>(goal, arena.pending_bytes / BLOCK_SIZE);
            goal = std::max(goal, arena.scan_reclaim_goal_pages.load(std::memory_order_acquire));
            const uint64_t first = arena.first_offset.load(std::memory_order_acquire);
            const uint64_t claimed_end = arena.claim_offset.load(std::memory_order_acquire);
            const uint64_t projected = (word & Arena::PAGE_MASK) + (claimed_end - first) / BLOCK_SIZE;
            if (!goal || projected >= goal) break;
        }
        auto& descriptor = arena.segments[cursor / FIRST_SEGMENT_SIZE];
        const uint64_t tag = descriptor.tag.load(std::memory_order_acquire);
        if (FirstSegmentDescriptor::state(tag) != FirstSegmentState::Sealed) break;
#ifdef TEST_MODE
        if (Arena::before_claim_cas) Arena::before_claim_cas();
#endif
        const uint64_t ticket = cursor;
        if (!arena.claim_offset.compare_exchange_weak(cursor, ticket + FIRST_SEGMENT_SIZE,
            std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        const auto state = consumer == FirstConsumer::Spill ? FirstSegmentState::Spilling : FirstSegmentState::Scanning;
        uint64_t expected = tag;
        const bool changed = descriptor.tag.compare_exchange_strong(expected,
            FirstSegmentDescriptor::make_tag(tag >> 3, state), std::memory_order_acq_rel, std::memory_order_acquire);
        assert(changed); (void)changed;
        out = {{static_cast<uint32_t>(index), ticket / FIRST_SEGMENT_SIZE, tag >> 3},
            reinterpret_cast<const unsigned char*>(descriptor.data), static_cast<size_t>(descriptor.record_count), consumer};
        claimed = true;
#ifdef TEST_MODE
        if (consumer == FirstConsumer::Spill) spill_claims.fetch_add(1, std::memory_order_relaxed);
#endif
        break;
    }
    arena.leave_short();
    return claimed ? FirstClaimStatus::Data : FirstClaimStatus::Busy;
}

inline ConcurrentMemoryPool::FirstClaimStatus ConcurrentMemoryPool::try_claim_first(
    int preferred, FirstConsumer consumer, FirstSegmentView& out) noexcept
{
    if (consumer == FirstConsumer::Scan && !first_frozen_.load(std::memory_order_acquire)) return FirstClaimStatus::Busy;
    for (int rank = 0; rank < num_arenas_; ++rank) {
        const int index = numa_distance_order_[preferred][rank];
        if (claim_first_in_arena(index, consumer, out) == FirstClaimStatus::Data) return FirstClaimStatus::Data;
    }
    return first_frozen_.load(std::memory_order_acquire) && first_empty() ? FirstClaimStatus::Done : FirstClaimStatus::Busy;
}

inline bool ConcurrentMemoryPool::complete_first(const FirstSegmentView& view) noexcept
{
    const auto& ref = view.ref;
    if (ref.arena_index >= static_cast<uint32_t>(num_arenas_) || ref.segment_index >= arenas_[ref.arena_index].segment_capacity
        || !ref.generation || ref.generation > FirstSegmentDescriptor::MAX_GENERATION) return false;
    Arena& arena = arenas_[ref.arena_index];
    auto& descriptor = arena.segments[ref.segment_index];
    const auto state = view.consumer == FirstConsumer::Spill ? FirstSegmentState::Spilling : FirstSegmentState::Scanning;
    uint64_t expected = FirstSegmentDescriptor::make_tag(ref.generation, state);
    if (!descriptor.tag.compare_exchange_strong(expected, FirstSegmentDescriptor::make_tag(ref.generation, FirstSegmentState::Done),
        std::memory_order_release, std::memory_order_relaxed)) return false;
#ifdef TEST_MODE
    if (Arena::after_done) Arena::after_done();
#endif
    arena.completion_sequence.fetch_add(1, std::memory_order_acq_rel);
#ifdef TEST_MODE
    if (Arena::after_completion_sequence) Arena::after_completion_sequence();
#endif
    if (arena.enter_reclaim()) { publish_first_prefix(arena); arena.leave_short(); }
    return true;
}

inline bool ConcurrentMemoryPool::first_empty() noexcept
{
    for (int i = 0; i < num_arenas_; ++i) {
        const auto& arena = arenas_[i];
        if (arena.first_offset.load(std::memory_order_acquire) != arena.capacity) return false;
        if (arena.admission_word.load(std::memory_order_acquire) & Arena::ACTIVE_MASK) return false;
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
    uint32_t impossible_arenas = 0;
    SpinBackoff<> backoff;
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
    seal_thread_first();
    try {
        for (;;) {
            state_->check();
            if (admitted >= 0) {
                poll_first_progress(admitted);
                Arena& arena = arenas_[admitted];
                const auto request = arena.request.load(std::memory_order_acquire);
                if (request == Arena::Request::Granted || request == Arena::Request::Impossible
                    || !spill_service_.load(std::memory_order_acquire)) {
                    std::lock_guard lock(arena.control_mutex);
                    const auto current = arena.request.load(std::memory_order_acquire);
                    char* result = current == Arena::Request::Granted ? arena.reserved_result : nullptr;
                    arena.reserved_result = nullptr;
                    arena.pending_bytes = 0;
                    arena.request.store(Arena::Request::Idle, std::memory_order_release);
                    arena.admission_word.fetch_and(~Arena::PRIORITY, std::memory_order_acq_rel);
                    if (current == Arena::Request::Impossible) impossible_arenas |= 1u << admitted;
                    admitted = -1;
                    if (result) return result;
                    if (preferred >= 0) throw std::bad_alloc();
                    if (current == Arena::Request::Impossible) continue;
                    break;
                }
            } else {
                if (preferred < 0 && bytes == BLOCK_SIZE)
                    if (void* result = allocate_small(false)) return result;
                bool possible = false;
                for (int rank = 0; rank < (preferred >= 0 ? 1 : num_arenas_); ++rank) {
                    const int candidate = preferred >= 0 ? preferred : numa_distance_order_[local][rank];
                    if (impossible_arenas & (1u << candidate)) continue;
                    poll_first_progress(candidate);
                    Arena& arena = arenas_[candidate];
                    std::unique_lock lock(arena.control_mutex);
                    uint64_t offset;
                    if (arena.try_reserve_tree(bytes, offset)) return static_cast<char*>(arena.start_addr) + offset;
#ifdef TEST_MODE
                    if (Arena::before_pending_admission) Arena::before_pending_admission();
#endif
                    // Grow may move FIRST between separate frontier/word reads.
                    // Only this monotone upper bound can reject before closing
                    // growth. Precise credit exhaustion is decided by the
                    // registered request in Reclaim, after short entries drain.
                    if (bytes > arena.capacity - arena.tree_offset.load(std::memory_order_acquire)) continue;
                    possible = true;
                    if (spill_service_.load(std::memory_order_acquire) && arena.request.load(std::memory_order_relaxed) == Arena::Request::Idle) {
                        arena.pending_bytes = bytes;
                        arena.reserved_result = nullptr;
                        arena.admission_word.fetch_or(Arena::PRIORITY, std::memory_order_acq_rel);
                        arena.request.store(Arena::Request::Waiting, std::memory_order_release);
                        if (Arena::mode(arena.admission_word.load(std::memory_order_acquire)) == Arena::Mode::Grow)
                            arena.set_mode(Arena::Mode::Closing);
                        admitted = candidate;
                        break;
                    }
                }
                if (admitted >= 0) continue;
                if (!possible || !spill_service_.load(std::memory_order_acquire)) {
                    if (preferred >= 0) throw std::bad_alloc();
                    break;
                }
            }
            backoff.backoff();
        }
    } catch (...) {
        if (admitted >= 0) {
            Arena& arena = arenas_[admitted];
            std::lock_guard lock(arena.control_mutex);
            arena.pending_bytes = 0;
            arena.reserved_result = nullptr;
            arena.request.store(Arena::Request::Idle, std::memory_order_release);
            arena.admission_word.fetch_and(~Arena::PRIORITY, std::memory_order_acq_rel);
        }
        throw;
    }
    // A freed page may have arrived while the pending request was being closed.
    if (preferred < 0 && bytes == BLOCK_SIZE)
        if (void* result = allocate_small(false)) return result;
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
    SpinBackoff<> backoff;
    for (;;) {
        state_->check();
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
            backoff.backoff();
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
        } catch (...) {
            if (new_segment && segment.base) ::munmap(segment.base, segment.reserved);
            if (opened) ::close(fd);
            if (!lock.owns_lock()) lock.lock();
            file_expanding_ = false;
            lock.unlock();
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
