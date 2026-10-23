#include "first_pool_fixture.h"
#include "../src/ConcurrentMemoryPool.h"
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <set>
#include <stdexcept>
#include <sys/stat.h>

static int fail_operation = 0;
static bool fail_metadata = false;
void* operator new(std::size_t bytes)
{
    if (fail_metadata) { fail_metadata = false; throw std::bad_alloc(); }
    if (void* result = std::malloc(bytes ? bytes : 1)) return result;
    throw std::bad_alloc();
}
void operator delete(void* pointer) noexcept { std::free(pointer); }
void operator delete(void* pointer, std::size_t) noexcept { std::free(pointer); }
struct Reservation { char* base; size_t reserved, committed; };
static std::vector<Reservation> reservations;
static bool guard_reservations = false;
static std::vector<unsigned char*> external_guards;
static std::vector<size_t> commits;
static std::promise<void>* growing = nullptr;
static std::shared_future<void> resume_growth;
static int sparse_preallocate(int fd, int, off_t offset, off_t length)
{
    if (growing) { growing->set_value(); resume_growth.wait(); }
    if (fail_operation == 2) { fail_operation = 0; errno = ENOSPC; return -1; }
    commits.push_back(static_cast<size_t>(length));
    return ::ftruncate(fd, offset + length);
}
static void* checked_map(void* base, size_t length, int protection, int flags, int fd, off_t offset) noexcept
{
    if ((fail_operation == 1 && protection == PROT_NONE) || (fail_operation == 3 && flags & MAP_FIXED)) {
        fail_operation = 0; errno = ENOMEM; return MAP_FAILED;
    }
    if (protection == PROT_NONE) {
        const size_t reserved = length + (guard_reservations ? BLOCK_SIZE : 0);
        void* result = ::mmap(base, reserved, protection, flags, fd, offset);
        if (result != MAP_FAILED && guard_reservations) {
            auto* guard = static_cast<unsigned char*>(result) + length;
            if (::mprotect(guard, BLOCK_SIZE, PROT_READ | PROT_WRITE) != 0) {
                ::munmap(result, reserved);
                return MAP_FAILED;
            }
            *guard = 0x6d;
            external_guards.push_back(guard);
        }
        if (result != MAP_FAILED) reservations.push_back({static_cast<char*>(result), length, 0});
        return result;
    }
    for (auto i = reservations.rbegin(); i != reservations.rend(); ++i) {
        if (base != i->base + i->committed || length > i->reserved - i->committed) continue;
        void* result = ::mmap(base, length, protection, flags, fd, offset);
        if (result != MAP_FAILED) i->committed += length;
        return result;
    }
    errno = EINVAL; // Reject replacement outside an uncommitted reservation tail.
    return MAP_FAILED;
}

struct ConcurrentMemoryPoolTestAccess
{
    static void initialize(ConcurrentMemoryPool& pool)
    {
        pool.num_arenas_ = 1;
        pool.total_cpus_ = pool.cpus_per_node_[0] = 1;
        pool.init_arenas();
    }
    static void exhaust(ConcurrentMemoryPool& pool)
    {
        initialize(pool);
        pool.allocate_large_from_numa(pool.gap_bytes(0), 0);
    }
    static void sparse(ConcurrentMemoryPool& pool) { pool.file_preallocate_ = sparse_preallocate; pool.file_map_ = checked_map; }
    static size_t available(ConcurrentMemoryPool& pool) {
        const auto& segment = pool.file_segments_[pool.file_current_];
        return segment.committed - segment.used;
    }
    static char* tail(ConcurrentMemoryPool& pool) {
        const auto& segment = pool.file_segments_[pool.file_current_];
        return segment.base + segment.used;
    }
    static void initial(const ConcurrentMemoryPool& pool) {
        if (pool.file_segments_.size() != 1 || pool.file_segments_[0].reserved != (16ULL << 30)
            || pool.file_segments_[0].committed != (64ULL << 20)) throw std::runtime_error("initial reserve/commit mismatch");
        struct stat info{};
        if (::fstat(pool.file_fd_, &info) != 0 || info.st_size != (64LL << 20)
            || (info.st_mode & 0777) != 0600) throw std::runtime_error("file size or permissions mismatch");
    }
    static bool untouched(const ConcurrentMemoryPool& pool) {
        return pool.file_segments_.empty() && pool.file_fd_ == -1 && !pool.file_expanding_
            && pool.file_growth_ == (64ULL << 20);
    }
    static void validate(const ConcurrentMemoryPool& pool) {
        for (size_t i = 0; i < pool.file_segments_.size(); ++i) {
            const auto& a = pool.file_segments_[i];
            if (a.used > a.committed || a.committed > a.reserved || a.committed % BLOCK_SIZE
                || a.file_offset % BLOCK_SIZE) throw std::runtime_error("invalid file segment extent");
            for (size_t j = 0; j < i; ++j) {
                const auto& b = pool.file_segments_[j];
                if (reinterpret_cast<uintptr_t>(b.base) + b.reserved >= reinterpret_cast<uintptr_t>(a.base))
                    throw std::runtime_error("unsorted or overlapping virtual ranges");
                if (!(a.file_offset + a.committed <= b.file_offset || b.file_offset + b.committed <= a.file_offset))
                    throw std::runtime_error("overlapping file ranges");
            }
        }
    }
    static size_t segments(const ConcurrentMemoryPool& pool) { return pool.file_segments_.size(); }
    static std::mutex& mutex(ConcurrentMemoryPool& pool) { return pool.file_mutex_; }
    static std::set<void*> returned_pages(ConcurrentMemoryPool& pool) {
        std::lock_guard lock(pool.file_mutex_);
        std::set<void*> pages;
        for (auto* head : {pool.file_free_list_, pool.tls_cache_.file_free_stack}) {
            for (auto* page = head; page; page = page->next)
                if (!pages.insert(page).second) throw std::runtime_error("duplicate returned file page or cyclic list");
        }
        return pages;
    }
    static size_t central_count(ConcurrentMemoryPool& pool) {
        std::lock_guard lock(pool.file_mutex_);
        size_t count = 0;
        for (auto* page = pool.file_free_list_; page; page = page->next) ++count;
        return count;
    }
    static void offset_limit(ConcurrentMemoryPool& pool) {
        pool.file_end_ = static_cast<size_t>(std::numeric_limits<off_t>::max()) & ~(BLOCK_SIZE - 1);
    }
    static bool reservation_empty(ConcurrentMemoryPool& pool) {
        std::lock_guard lock(pool.arenas_[0].control_mutex);
        return !pool.arenas_[0].pending_bytes && !pool.arenas_[0].reserved_result;
    }
};
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

static void check_local_return_without_locks()
{
    ConcurrentMemoryPool pool(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::exhaust(pool);
    ConcurrentMemoryPoolTestAccess::sparse(pool);
    std::promise<void> warmed, start, returned, finish;
    auto start_gate = start.get_future();
    auto finish_gate = finish.get_future();
    auto result = returned.get_future();
    std::thread worker([&] {
        auto* first = pool.allocate();
        auto* second = pool.allocate();
        pool.deallocate(first); // Warm the address range without filling TLS.
        warmed.set_value();
        start_gate.wait();
        pool.deallocate(second);
        returned.set_value();
        finish_gate.wait(); // Report completion before TLS teardown takes locks.
    });
    warmed.get_future().wait();
    std::unique_lock progress(pool.run_state().mutex);
    std::unique_lock file(ConcurrentMemoryPoolTestAccess::mutex(pool));
    start.set_value();
    const bool immediate = result.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
    file.unlock();
    progress.unlock();
    finish.set_value();
    worker.join();
    require(immediate, "warm file TLS return waited for a file or progress lock");
}

static void check_batch_return_without_progress_lock()
{
    ConcurrentMemoryPool pool(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::exhaust(pool);
    ConcurrentMemoryPoolTestAccess::sparse(pool);
    std::promise<void> warmed, start, returned;
    auto start_gate = start.get_future();
    auto result = returned.get_future();
    std::thread worker([&] {
        auto* page = pool.allocate();
        pool.deallocate(page);
        warmed.set_value();
        start_gate.wait();
        ConcurrentMemoryPool::get_thread_local_cache().release();
        returned.set_value();
    });
    warmed.get_future().wait();
    std::unique_lock progress(pool.run_state().mutex);
    start.set_value();
    const bool immediate = result.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
    progress.unlock();
    worker.join();
    require(immediate, "TLS batch return waited for the global progress lock");
    require(ConcurrentMemoryPoolTestAccess::central_count(pool) == TLS_CAPACITY,
        "TLS batch return lost file pages");
}

static void check_batch_return()
{
    ConcurrentMemoryPool pool(8ULL << 20);
    ConcurrentMemoryPoolTestAccess::exhaust(pool);
    ConcurrentMemoryPoolTestAccess::sparse(pool);
    std::vector<void*> pages(512);
    for (auto& page : pages) page = pool.allocate();
    const std::set<void*> expected(pages.begin(), pages.end());
    require(expected.size() == pages.size(), "file allocation duplicated a page");
    auto& tls = ConcurrentMemoryPool::get_thread_local_cache();
    require(tls.file_free_count == 0, "batch fixture retained unused file pages");
    const auto generation = pool.run_state().progress();
    size_t published = 0;
    for (auto* page : pages) {
        const bool full = tls.file_free_count == TLS_CAPACITY;
        pool.deallocate(page);
        published += full;
        require(tls.file_free_count <= TLS_CAPACITY, "file TLS exceeded its bound");
        require(pool.run_state().progress() == generation,
            "resource return published a progress notification");
    }
    require(published == 6 && tls.file_free_count == 128
        && ConcurrentMemoryPoolTestAccess::central_count(pool) == 384, "file pages were not published in half-cache batches");
    require(ConcurrentMemoryPoolTestAccess::returned_pages(pool) == expected, "file return lost pages");
    require(!pool.try_allocate() && !pool.try_allocate_large(BLOCK_SIZE), "file pages entered the RAM try path");
    std::vector<void*> reused(384);
    std::thread consumer([&] {
        for (auto& page : reused) page = pool.allocate();
        for (auto* page : reused) pool.deallocate(page);
    });
    consumer.join(); // TLS teardown must publish the consumer's remaining pages.
    require(std::set<void*>(reused.begin(), reused.end()).size() == reused.size(), "cross-thread reuse duplicated pages");
    for (auto* page : reused) require(expected.contains(page), "central pages were not reused by another thread");
    require(ConcurrentMemoryPoolTestAccess::returned_pages(pool) == expected, "thread exit lost file pages");
}

static void check_expansion_waiters()
{
    for (bool inject_failure : {false, true}) {
        ConcurrentMemoryPool pool(8ULL << 20);
        ConcurrentMemoryPoolTestAccess::exhaust(pool);
        ConcurrentMemoryPoolTestAccess::sparse(pool);
        std::promise<void> entered, resume, waiting;
        growing = &entered;
        resume_growth = resume.get_future().share();
        auto expansion = std::async(std::launch::async, [&] {
            try { return pool.allocate_large(64ULL << 20) != nullptr; }
            catch (const std::bad_alloc&) { return false; }
        });
        entered.get_future().wait();
        auto allocator = std::async(std::launch::async, [&] {
            waiting.set_value();
            return pool.allocate();
        });
        waiting.get_future().wait();
        const bool blocked = allocator.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
        // The first preallocation is already suspended; a retry must not enter
        // that same one-shot gate after the injected failure.
        growing = nullptr;
        if (inject_failure) fail_operation = 2;
        resume.set_value();
        const bool expanded = expansion.get();
        void* page = allocator.get();
        require(blocked && page && expanded == !inject_failure,
            "file expansion waiter ignored publication or failed to retry after failure");
        pool.deallocate(page);
        ConcurrentMemoryPoolTestAccess::validate(pool);
    }
}

int main()
{
    try {
        k_length = 33;
        char name[] = "/tmp/tree-file-arena-XXXXXX";
        const auto directory = std::filesystem::path(::mkdtemp(name));
        const auto previous = std::filesystem::current_path();
        std::filesystem::current_path(directory);
        check_local_return_without_locks();
        check_batch_return_without_progress_lock();
        check_batch_return();
        check_expansion_waiters();
        {
            ConcurrentMemoryPool pool(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::exhaust(pool);
            require(!pool.try_allocate() && !pool.try_allocate_large(BLOCK_SIZE), "try allocation used a file");
            require(ConcurrentMemoryPoolTestAccess::untouched(pool), "RAM-only path initialized file arena");
            require(!std::filesystem::exists("mapping.tmp"), "RAM-only allocation created a file");
            auto* page = static_cast<unsigned char*>(pool.allocate());
            page[BLOCK_SIZE - 1] = 0xa5;
            require(page[BLOCK_SIZE - 1] == 0xa5, "fallback page is not writable");
            ConcurrentMemoryPoolTestAccess::initial(pool);
            pool.deallocate(page);
            require(pool.allocate() == page, "file TLS page was not reused");
            require(!std::filesystem::exists("mapping.tmp"), "mapping file was not unlinked immediately");
        }
        require(ConcurrentMemoryPool::get_thread_local_cache().file_range_begin == 0
            && ConcurrentMemoryPool::get_thread_local_cache().file_range_bytes == 0, "pool destruction retained a file range");
        for (int operation : {1, 2, 3, 4}) {
            ConcurrentMemoryPool pool(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::exhaust(pool);
            ConcurrentMemoryPoolTestAccess::sparse(pool);
            fail_operation = operation;
            fail_metadata = operation == 4;
            bool failed = false;
            try { pool.allocate(); } catch (const std::bad_alloc&) { failed = true; }
            require(failed && ConcurrentMemoryPoolTestAccess::untouched(pool), "failed initialization published resources or advanced growth");
            require(pool.allocate(), "allocation did not recover from initialization failure");
            ConcurrentMemoryPoolTestAccess::initial(pool);
        }
        {
            ConcurrentMemoryPool pool(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::exhaust(pool);
            ConcurrentMemoryPoolTestAccess::sparse(pool);
            commits.clear();
            guard_reservations = true;
            auto* old = static_cast<unsigned char*>(pool.allocate());
            old[BLOCK_SIZE - 1] = 0x73;
            pool.deallocate(old);
            require(pool.allocate() == old, "range warmup lost a file page");
            auto& tls = ConcurrentMemoryPool::get_thread_local_cache();
            const uintptr_t old_end = tls.file_range_begin + tls.file_range_bytes;
            const size_t free_before_invalid = tls.file_free_count;
            pool.deallocate(reinterpret_cast<void*>(old_end));
            require(tls.file_free_count == free_before_invalid, "range cache accepted an uncommitted page");
            for (size_t expected : {128ULL << 20, 256ULL << 20, 512ULL << 20, 1ULL << 30, 1ULL << 30}) {
                pool.allocate_large(ConcurrentMemoryPoolTestAccess::available(pool) + BLOCK_SIZE);
                require(commits.back() == expected, "file growth sequence diverged");
                if (expected == (128ULL << 20)) {
                    pool.deallocate(reinterpret_cast<void*>(old_end));
                    require(tls.file_range_begin + tls.file_range_bytes > old_end
                        && pool.allocate() == reinterpret_cast<void*>(old_end), "grown committed range was not refreshed");
                }
            }
            const size_t before = commits.size();
            pool.allocate_large(BLOCK_SIZE);
            require(commits.size() == before, "committed space triggered growth");
            fail_operation = 2;
            bool failed = false;
            try { pool.allocate_large(ConcurrentMemoryPoolTestAccess::available(pool) + BLOCK_SIZE); }
            catch (const std::bad_alloc&) { failed = true; }
            require(failed && old[BLOCK_SIZE - 1] == 0x73, "growth failure damaged published data");
            auto* old_tail = ConcurrentMemoryPoolTestAccess::tail(pool);
            // Move to a large segment without allocating 16 GiB of real disk.
            auto* large = static_cast<unsigned char*>(pool.allocate_large((16ULL << 30) + BLOCK_SIZE));
            large[0] = 0x41;
            large[16ULL << 30] = 0x42;
            require(commits.back() % (1ULL << 30) == 0, "large request did not commit a growth multiple");
            require(ConcurrentMemoryPoolTestAccess::segments(pool) == 2, "large request crossed a segment");
            require(old[BLOCK_SIZE - 1] == 0x73 && large[0] == 0x41 && large[16ULL << 30] == 0x42,
                "segment growth moved or overwrote old pointers");
            // Consume the new segment's larger tail; the next small extent
            // must reuse committed space in the old segment without remapping.
            pool.allocate_large(ConcurrentMemoryPoolTestAccess::available(pool));
            const auto committed_before_reuse = commits.size();
            require(pool.allocate_large(64ULL << 10) == old_tail && commits.size() == committed_before_reuse,
                "old committed tail was not reusable");
            pool.deallocate(large + (16ULL << 30));
            require(pool.allocate() == large + (16ULL << 30), "file address lookup truncated a large segment");
            pool.deallocate(old);
            require(pool.allocate() == old, "old segment range was not reusable");
            for (int i = 0; i < 3; ++i) pool.allocate_large((16ULL << 30) + BLOCK_SIZE);
            require(ConcurrentMemoryPoolTestAccess::segments(pool) == 5 && old[BLOCK_SIZE - 1] == 0x73,
                "directory reallocation invalidated published addresses");
            pool.deallocate(old);
            require(pool.allocate() == old, "directory movement invalidated the cached range");
            ConcurrentMemoryPoolTestAccess::validate(pool);
            guard_reservations = false;
        }
        {
            ConcurrentMemoryPool pool(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::exhaust(pool);
            ConcurrentMemoryPoolTestAccess::sparse(pool);
            pool.allocate_large((16ULL << 30) - (64ULL << 20));
            pool.allocate();
            require(commits.back() == (64ULL << 20), "segment tail was not truncated to its remaining reservation");
            ConcurrentMemoryPoolTestAccess::validate(pool);
        }
        {
            ConcurrentMemoryPool pool(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::exhaust(pool);
            ConcurrentMemoryPoolTestAccess::sparse(pool);
            ConcurrentMemoryPoolTestAccess::offset_limit(pool);
            bool failed = false;
            try { pool.allocate(); } catch (const std::bad_alloc&) { failed = true; }
            require(failed && ConcurrentMemoryPoolTestAccess::untouched(pool), "file offset overflow published a mapping");
        }
        {
            ConcurrentMemoryPool pool(8ULL << 20), other(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::exhaust(pool);
            ConcurrentMemoryPoolTestAccess::sparse(pool);
            other.init_arenas();
            auto* old = pool.allocate();
            const size_t needed = ConcurrentMemoryPoolTestAccess::available(pool) + BLOCK_SIZE;
            std::promise<void> entered, resume;
            growing = &entered;
            resume_growth = resume.get_future().share();
            auto expansion = std::async(std::launch::async, [&] { return pool.allocate_large(needed); });
            entered.get_future().wait();
            // These operations must finish while fallocate remains blocked.
            auto concurrent = std::async(std::launch::async, [&] {
                auto* page = pool.allocate();
                pool.deallocate(page);
                return other.try_allocate() != nullptr;
            });
            const bool progressed = concurrent.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
            resume.set_value();
            require(progressed && concurrent.get() && expansion.get(), "expansion held an internal lock during I/O");
            growing = nullptr;
            pool.deallocate(old);
            // A try operation switching away from cached file pages cannot wait.
            std::promise<void> cached, switch_now;
            auto switch_gate = switch_now.get_future();
            std::promise<void*> switched;
            auto switched_result = switched.get_future();
            std::promise<void> retry;
            auto retry_gate = retry.get_future();
            std::promise<bool> preserved, reset;
            std::thread switching([&] {
                pool.deallocate(pool.allocate());
                auto& tls = ConcurrentMemoryPool::get_thread_local_cache();
                const auto begin = tls.file_range_begin;
                const auto bytes = tls.file_range_bytes;
                cached.set_value();
                switch_gate.wait();
                switched.set_value(other.try_allocate());
                preserved.set_value(tls.pool == &pool && tls.file_range_begin == begin && tls.file_range_bytes == bytes);
                retry_gate.wait();
                reset.set_value(other.allocate() && tls.pool == &other && !tls.file_range_begin && !tls.file_range_bytes);
            });
            cached.get_future().wait();
            std::unique_lock held(ConcurrentMemoryPoolTestAccess::mutex(pool));
            switch_now.set_value();
            const bool immediate = switched_result.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
            held.unlock();
            retry.set_value();
            switching.join();
            require(immediate && !switched_result.get(), "try pool switch waited for the file lock");
            require(preserved.get_future().get() && reset.get_future().get(), "pool switch changed or retained the wrong file range");
            ConcurrentMemoryPoolTestAccess::validate(pool);
        }
        std::filesystem::create_directory("nested");
        // Race service shutdown against installing the RAM reservation. This
        // must pass through reservation cleanup before entering file fallback.
        std::vector<unsigned char> first_records(FIRST_SEGMENT_SIZE);
        for (int iteration = 0; iteration < 1000; ++iteration) {
            ConcurrentMemoryPool pool(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::initialize(pool);
            ConcurrentMemoryPoolTestAccess::sparse(pool);
            require(cache_full_first(pool, 0, first_records.data()), "shutdown fixture could not cache FIRST");
            const auto needed = pool.gap_bytes(0) + BLOCK_SIZE;
            pool.set_spill_service(true);
            auto allocation = std::async(std::launch::async, [&] { return pool.allocate_large(needed); });
            while (pool.pressure(0) != MemoryPressure::Hard) cpu_relax();
            pool.set_spill_service(false);
            require(allocation.get(), "shutdown did not fall back to the file arena");
            require(ConcurrentMemoryPoolTestAccess::reservation_empty(pool), "shutdown stranded a pending RAM reservation");
        }
        for (const std::string& path : {std::string("nested/relative.map"), (directory / "nested/absolute.map").string()}) {
            mapping_filename = path;
            ConcurrentMemoryPool pool(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::exhaust(pool);
            mapping_filename = "mapping.tmp"; // The pool must retain its own configuration.
            std::ofstream(path) << "protected";
            bool failed = false;
            try { pool.allocate(); } catch (const std::bad_alloc&) { failed = true; }
            require(failed, "configured mapping path was ignored");
            std::filesystem::remove(path);
            require(pool.allocate() && !std::filesystem::exists(path), "configured path could not grow an unlinked file");
        }
        {
            std::ofstream("mapping.tmp") << "existing";
            ConcurrentMemoryPool pool(8ULL << 20);
            ConcurrentMemoryPoolTestAccess::exhaust(pool);
            bool failed = false;
            try { pool.allocate(); } catch (const std::bad_alloc&) { failed = true; }
            require(failed, "existing mapping file was overwritten");
            std::ifstream file("mapping.tmp");
            std::string value;
            file >> value;
            require(value == "existing", "exclusive create changed an existing file");
        }
        for (auto* guard : external_guards) {
            require(*guard == 0x6d, "file mapping or cleanup overwrote an external mapping");
            ::munmap(guard, BLOCK_SIZE);
        }
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(directory);
        std::cout << "FileArena tests passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
