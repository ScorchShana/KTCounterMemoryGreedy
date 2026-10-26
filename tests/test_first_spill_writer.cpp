#include "first_pool_fixture.h"
#include "../src/ExportWriter.h"
#include "low_msd_test_helpers.h"
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <stdexcept>

static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct SlowIo
{
    std::promise<void> entered, resume;
    std::shared_future<void> release = resume.get_future().share();
    std::atomic<bool> first{true};
};
static SlowIo* slow_io = nullptr;
static int slow_submit(aiocb* cb)
{
    if (slow_io->first.exchange(false)) {
        slow_io->entered.set_value();
        slow_io->release.wait();
    }
    return ::aio_write(cb);
}
static ssize_t short_return(aiocb* cb) { const auto n = ::aio_return(cb); return n > 0 ? n - 1 : n; }

struct DelayedIo
{
    std::array<aiocb*, 8> requests{};
    std::array<unsigned, 8> returns{};
    std::atomic<unsigned> submitted{0}, ready{0}, collected{0};
    std::array<std::vector<uint8_t>, 8> snapshots;
    bool overwritten = false;
    unsigned failure = 0;
};
static DelayedIo* delayed_io;
static int delayed_submit(aiocb* cb)
{
    const auto i = delayed_io->submitted.load(std::memory_order_relaxed);
    if (i == delayed_io->requests.size()) { errno = EIO; return -1; }
    if (delayed_io->failure == 1 && i == 1) {
        delayed_io->ready.store(255, std::memory_order_release);
        errno = EIO;
        return -1;
    }
    delayed_io->requests[i] = cb;
    const auto* data = static_cast<const uint8_t*>(const_cast<const void*>(cb->aio_buf));
    delayed_io->snapshots[i].assign(data, data + cb->aio_nbytes);
    delayed_io->submitted.store(i + 1, std::memory_order_release);
    return 0;
}
static size_t delayed_index(const aiocb* cb)
{
    for (size_t i = 0; i < delayed_io->submitted.load(std::memory_order_acquire); ++i)
        if (delayed_io->requests[i] == cb && !delayed_io->returns[i]) return i;
    std::terminate();
}
static int delayed_error(const aiocb* cb)
{
    const size_t i = delayed_index(cb);
    if (!(delayed_io->ready.load(std::memory_order_acquire) & (1u << i))) return EINPROGRESS;
    return delayed_io->failure == 3 && i == 0 ? EIO : 0;
}
static ssize_t delayed_result(aiocb* cb)
{
    const size_t i = delayed_index(cb);
    ++delayed_io->returns[i];
    const void* data = const_cast<const void*>(cb->aio_buf);
    if (cb->aio_nbytes != delayed_io->snapshots[i].size()
        || std::memcmp(data, delayed_io->snapshots[i].data(), cb->aio_nbytes)) delayed_io->overwritten = true;
    const ssize_t result = ::pwrite(cb->aio_fildes, data, cb->aio_nbytes, cb->aio_offset);
    delayed_io->collected.fetch_add(1, std::memory_order_release);
    return delayed_io->failure == 2 && i == 0 && result > 0 ? result - 1 : result;
}

struct ConcurrentMemoryPoolTestAccess
{
    static Arena& arena(ConcurrentMemoryPool& pool) { return pool.arenas_[0]; }
    static void seal_empty(ConcurrentMemoryPool& pool) {
        auto& arena = pool.arenas_[0];
        const auto slot = arena.first_begin / FIRST_SEGMENT_SIZE;
        pool.seal_first_segment({0, slot, arena.segments[slot].tag.load() >> 3}, 0);
    }
    static void single(ConcurrentMemoryPool& pool) { pool.num_arenas_ = 1; pool.total_cpus_ = pool.cpus_per_node_[0] = 1; }
    static void filling(ConcurrentMemoryPool& pool) {
        FirstSegmentRef ref; unsigned char* data;
        if (!pool.begin_first_segment(0, ref, data)) throw std::runtime_error("Filling fixture capacity");
    }
    static size_t pending(ConcurrentMemoryPool& memory) {
        std::lock_guard lock(memory.arenas_[0].mutex);
        return memory.arenas_[0].wait_head ? memory.arenas_[0].wait_head->bytes : 0;
    }
};

struct ExportWriterTestAccess
{
    static bool step(ExportWriter& writer) { return writer.spill_step(true); }
    static unsigned skip(const ExportWriter& writer) { return writer.spill_retry_skip_; }
    static unsigned window(const ExportWriter& writer) { return writer.spill_retry_window_; }
    static bool ordinary(ExportWriter& writer) { return writer.normal_step(); }
    static void reach_shard_limit(ExportWriter& writer) {
        writer.shard_limit_bytes_ = 1;
        writer.file_offset = 1;
    }
    static size_t workspace_buffers(const ExportWriter& writer) {
        size_t count = (writer.low_input_ != nullptr) + (writer.low_scratch_ != nullptr);
        for (const char* buffer : writer.buffers) count += buffer != nullptr;
        return count;
    }
};

static void check_unclaimable_demand_does_not_rotate()
{
    ConcurrentMemoryPool pool(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(pool);
    pool.init_arenas();
    pool.set_spill_service(true);
    ConcurrentMemoryPoolTestAccess::filling(pool);
    pool.request_first_drain();
    auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(65536, 1);
    const auto path = std::filesystem::temp_directory_path() / ("first-no-empty-tail-" + std::to_string(getpid()));
    {
        ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &pool);
        ExportWriterTestAccess::reach_shard_limit(writer);
        require(!ExportWriterTestAccess::step(writer), "unpublished FIRST was claimed");
        const bool empty_tail = std::filesystem::exists(path.string() + ".1");
        if (empty_tail) std::filesystem::remove(path.string() + ".1");
        require(!empty_tail && writer.shard_count() == 1,
            "unclaimable FIRST demand created an empty trailing shard");
    }
    std::filesystem::remove(path.string() + ".0");
}

static void check_independent_spill_retry()
{
    using Access = ExportWriterTestAccess;
    ConcurrentMemoryPool pool(32ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(pool);
    pool.init_arenas();
    pool.set_spill_service(true);
    ConcurrentMemoryPoolTestAccess::filling(pool);
    pool.request_first_drain();
    auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    const auto path = std::filesystem::temp_directory_path() / ("first-retry-" + std::to_string(getpid()));
    {
        ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &pool);
        for (unsigned wait : {1u, 2u, 4u, 8u, 8u}) {
            require(!Access::step(writer) && Access::skip(writer) == wait,
                "Writer did not back off a repeated Filling claim");
            char* block;
            queue->producer_dequeue(block);
            queue->producer_enqueue({block, 0});
            require(Access::ordinary(writer) && Access::skip(writer) == wait,
                "ordinary output progress reset FIRST retry backoff");
            for (unsigned i = wait; i > 0; --i)
                require(!Access::step(writer) && Access::skip(writer) == i - 1,
                    "Writer skipped the wrong number of eligible loops");
        }
        ConcurrentMemoryPoolTestAccess::seal_empty(pool);
        require(Access::step(writer) && Access::skip(writer) == 0 && Access::window(writer) == 1,
            "FIRST success/completion did not reset independent retries");
        require(pool.first_empty() && !pool.writer_work_pending(), "empty completion retained demand");
        // Quiet polling must not consult the Arena lock, including while frozen.
        pool.freeze_first();
        auto& arena = ConcurrentMemoryPoolTestAccess::arena(pool);
        std::unique_lock held(arena.mutex);
        auto quiet = std::async(std::launch::async, [&] { return Access::step(writer); });
        const bool immediate = quiet.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
        held.unlock();
        require(!quiet.get() && immediate, "quiet Writer consulted frozen-empty mutexes");
    }
    std::filesystem::remove(path.string() + ".0");
}

static void check_close_wakes_allocator()
{
    RunState state;
    ConcurrentMemoryPool memory(16ULL << 20, &state);
    ConcurrentMemoryPoolTestAccess::single(memory);
    memory.init_arenas();
    ConcurrentMemoryPoolTestAccess::filling(memory);
    auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    const auto path = std::filesystem::temp_directory_path() / ("first-wake-" + std::to_string(getpid()));
    ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &memory);
    writer.start();
    const auto needed = memory.gap_bytes(0) + BLOCK_SIZE;
    auto allocation = std::async(std::launch::async, [&] {
        try { memory.allocate_large_from_numa(needed, 0); }
        catch (const std::bad_alloc&) { return true; }
        return false;
    });
    SpinBackoff<> backoff;
    while (!ConcurrentMemoryPoolTestAccess::pending(memory)) {
        state.check();
        backoff.backoff();
    }
    queue->producer_set_finished();
    writer.join();
    require(allocation.get() && ConcurrentMemoryPoolTestAccess::pending(memory) == 0,
        "Writer shutdown stranded an allocation reservation");
    std::filesystem::remove(path.string() + ".0");
}

static void check_idle_exit()
{
    for (bool inject_error : {false, true}) {
        RunState state;
        ConcurrentMemoryPool memory(16ULL << 20, &state);
        ConcurrentMemoryPoolTestAccess::single(memory);
        memory.init_arenas();
        ConcurrentMemoryPoolTestAccess::filling(memory);
        auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
        const auto path = std::filesystem::temp_directory_path() / ("first-idle-" + std::to_string(getpid()));
        ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &memory);
        writer.start();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (inject_error) state.fail(std::make_exception_ptr(std::runtime_error("idle writer failure")), "test");
        else queue->producer_set_finished();
        bool failed = false;
        try { writer.join(); }
        catch (const std::runtime_error& error) { failed = std::strcmp(error.what(), "idle writer failure") == 0; }
        require(failed == inject_error && !memory.first_empty() && memory.spill_claims.load() == 0,
            "idle Writer ignored close/failure or claimed unpublished FIRST");
        require(std::filesystem::file_size(path.string() + ".0") == 0, "idle Writer emitted unpublished data");
        std::filesystem::remove(path.string() + ".0");
    }
}

static void check_close_preserves_first()
{
    RunState state;
    ConcurrentMemoryPool memory(32ULL << 20, &state);
    ConcurrentMemoryPoolTestAccess::single(memory);
    memory.init_arenas();
    std::vector<unsigned char> active(FIRST_SEGMENT_SIZE);
    const size_t records = ConcurrentMemoryPool::first_segment_record_capacity() * 3;
    std::array<unsigned char, 9> key{};
    for (size_t i = 0; i < records; ++i) {
        std::memcpy(key.data(), &i, sizeof(i));
        const size_t slot = i % ConcurrentMemoryPool::first_segment_record_capacity();
        std::memcpy(active.data() + slot * 9, key.data(), 9);
        if (slot + 1 == ConcurrentMemoryPool::first_segment_record_capacity())
            require(cache_full_first(memory, 0, active.data()), "shutdown fixture exceeded FIRST capacity");
    }
    auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    char* block;
    queue->producer_dequeue(block);
    std::memcpy(block, &records, sizeof(records));
    block[8] = 0;
    queue->producer_enqueue({block, 1});
    const auto path = std::filesystem::temp_directory_path() / ("first-close-" + std::to_string(getpid()));
    SlowIo slow;
    slow_io = &slow;
    ExportWriter::IoOps ops;
    ops.submit = slow_submit;
    ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &memory, &ops);
    memory.request_first_drain();
    writer.start();
    slow.entered.get_future().wait();
    // The first AIO submission pauses while the Writer owns a claimed segment.
    // Closing must complete that segment without claiming its neighbours.
    memory.freeze_first();
    queue->producer_set_finished();
    slow.resume.set_value();
    writer.join();
    require(!memory.first_empty(), "Writer closing drained unclaimed FIRST");
    require(memory.spill_claims.load() == 1, "Writer claimed another segment after production finished");
    std::vector<bool> seen(records + 1);
    size_t copied = 0;
    for (uint64_t shard = 0; shard < writer.shard_count(); ++shard) {
        const auto filename = path.string() + "." + std::to_string(shard);
        for (const auto& record : low_msd_test::read(filename, key.size())) {
            size_t id;
            std::memcpy(&id, record.data(), sizeof(id));
            require(id <= records && !seen[id], "shutdown pending multiset changed");
            seen[id] = true;
            ++copied;
        }
        std::filesystem::remove(filename);
    }
    require(copied == ConcurrentMemoryPool::first_segment_record_capacity() + 1, "Writer left a claimed segment partially copied");
    for (;;) {
        ConcurrentMemoryPool::FirstSegmentView view;
        const auto status = memory.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Scan, view);
        if (status == ConcurrentMemoryPool::FirstClaimStatus::Done) break;
        require(status == ConcurrentMemoryPool::FirstClaimStatus::Data, "closing left a borrowed segment");
        for (size_t i = 0; i < view.record_count; ++i) {
            size_t id;
            std::memcpy(&id, view.records + i * key.size(), sizeof(id));
            require(id < records && !seen[id], "pending and remaining FIRST overlap");
            seen[id] = true;
            ++copied;
        }
        require(memory.complete_first(view), "remaining source completion rejected");
    }
    require(copied == records + 1 && memory.first_empty(), "shutdown lost FIRST records");
}

static void check_normal_drain_detaches_ring()
{
    ConcurrentMemoryPool memory(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(memory);
    memory.init_arenas();
    std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 0x53);
    require(cache_full_first(memory, 0, records.data()), "normal-drain source fixture failed");
    auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    char* block;
    queue->producer_dequeue(block);
    std::memset(block, 0x11, 9);
    queue->producer_enqueue({block, 1});
    queue->producer_set_finished();
    const auto path = std::filesystem::temp_directory_path() / ("first-detach-" + std::to_string(getpid()));
    ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &memory, nullptr,
        ExportWriter::ReclaimLifetime::UntilExplicitStop);
    writer.start();
    writer.wait_for_normal_drain();
    require(ExportWriterTestAccess::workspace_buffers(writer) == 4,
        "normal drain released workspace still needed for FIRST reclamation");
    queue->reset_producers(1);
    queue->producer_dequeue(block);
    std::memset(block, 0x22, 9);
    queue->producer_enqueue({block, 1});
    memory.freeze_first();
    memory.request_first_drain();
    SpinBackoff<> backoff;
    while (!memory.first_empty()) { memory.run_state().check(); backoff.backoff(); }
    writer.finish_reclaim();
    writer.join();
    require(ExportWriterTestAccess::workspace_buffers(writer) == 0,
        "stopped pending Writer retained workspace during replay");
    require(writer.encoded_bytes_written() == std::filesystem::file_size(path.string() + ".0"),
        "workspace release lost encoded output statistics");
    content_type next;
    require(queue->consumer_try_dequeue(next) && next.length == 1 && static_cast<unsigned char>(next.data[0]) == 0x22,
        "pending Writer consumed the reset Ring");
    queue->consumer_enqueue(next.data);
    queue->producer_set_finished();
    require(low_msd_test::read(path.string() + ".0", 9).size() == 1 + memory.first_segment_record_capacity(),
        "normal-drain lost ordinary or raw FIRST output");
    std::filesystem::remove(path.string() + ".0");
}

static void check_split_source_aio(unsigned failure = 0)
{
    k_length = 28; // Width 7: full FIRST has Q+1 records.
    RunState state;
    ActiveRunScope scope(state);
    ConcurrentMemoryPool memory(16ULL << 20, &state);
    ConcurrentMemoryPoolTestAccess::single(memory);
    memory.init_arenas();
    std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 0x53);
    require(cache_full_first(memory, 0, records.data()), "split source fixture failed");
    memory.freeze_first(); memory.request_first_drain();
    const size_t gap = memory.gap_bytes(0);
    auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(65536, 1);
    queue->producer_set_finished();
    DelayedIo delayed; delayed.failure = failure; delayed_io = &delayed;
    ExportWriter::IoOps ops;
    ops.submit = delayed_submit; ops.error = delayed_error; ops.result = delayed_result;
    const auto path = std::filesystem::temp_directory_path() / ("first-split-" + std::to_string(getpid()));
    ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &memory, &ops,
        ExportWriter::ReclaimLifetime::UntilExplicitStop);
    writer.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (delayed.submitted.load(std::memory_order_acquire) < 2 && !state.failed()
        && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    const bool split = delayed.submitted.load(std::memory_order_acquire) == 2;
    bool held = false;
    if (split && !failure) {
        delayed.ready.store(2, std::memory_order_release);
        while (!delayed.collected.load(std::memory_order_acquire) && !state.failed()) std::this_thread::yield();
        held = memory.gap_bytes(0) == gap && memory.spill_bytes.load() == 0 && !memory.first_empty();
    }
    delayed.ready.store(255, std::memory_order_release);
    writer.finish_reclaim();
    bool rejected = false;
    try { writer.join(); } catch (const std::exception&) { rejected = true; }
    if (failure) {
        require(rejected && state.failed() && memory.first_empty(), "failed split source ownership retained");
        require(delayed.collected.load() == delayed.submitted.load() && !delayed.overwritten,
            "failed split source did not collect each submitted AIO");
        require(memory.spill_bytes.load() == 0 && memory.returned_bytes.load() == FIRST_SEGMENT_SIZE,
            "failed split source counted success or returned space twice");
        std::filesystem::remove(path.string() + ".0");
        k_length = 33;
        return;
    }
    require(!rejected && !state.failed(), "successful split source failed");
    require(split, "full FIRST was not split at codec Q");
    require(held && memory.first_empty(), "split source reclaimed before all outputs completed");
    require(delayed.collected.load() == 2 && !delayed.overwritten, "split output reused or lost an AIO slot");
    require(memory.spill_bytes.load() == memory.first_segment_record_capacity() * kmer_bytes(), "spill byte counter became encoded bytes");
    require(low_msd_test::read(path.string() + ".0", kmer_bytes()).size() == memory.first_segment_record_capacity(),
        "split source multiset length changed");
    std::filesystem::remove(path.string() + ".0");
    k_length = 33;
}

static void check_parallel_borrowed_aio()
{
    ConcurrentMemoryPool memory(16ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(memory);
    memory.init_arenas();
    std::vector<unsigned char> records(FIRST_SEGMENT_SIZE, 0x53);
    for (int i = 0; i < 3; ++i) require(cache_full_first(memory, 0, records.data()), "parallel AIO fixture failed");
    memory.freeze_first();
    memory.request_first_drain();
    const size_t gap = memory.gap_bytes(0);
    auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
    const auto path = std::filesystem::temp_directory_path() / ("first-parallel-" + std::to_string(getpid()));
    DelayedIo delayed;
    delayed_io = &delayed;
    ExportWriter::IoOps ops;
    ops.submit = delayed_submit;
    ops.error = delayed_error;
    ops.result = delayed_result;
    ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &memory, &ops,
        ExportWriter::ReclaimLifetime::UntilExplicitStop);
    writer.start();
    SpinBackoff<> backoff;
    while (delayed.submitted.load(std::memory_order_acquire) < 2) backoff.backoff();
    delayed.ready.store(2, std::memory_order_release);
    while (delayed.submitted.load(std::memory_order_acquire) < 3) backoff.backoff();
    // Reuse the completed second slot for the third source. Complete both higher-address sources first. Their in-flight data must
    // remain intact and no space can be returned past the lowest source.
    delayed.ready.store(6, std::memory_order_release);
    while (memory.spill_bytes.load(std::memory_order_relaxed) < 2 * memory.first_segment_record_capacity() * 9) backoff.backoff();
    require(memory.gap_bytes(0) == gap, "AIO completion returned a noncontinuous suffix");
    // Ordinary Ring output must keep moving while the first raw AIO is paused.
    char* block;
    queue->producer_dequeue(block);
    std::memset(block, 0x11, 9);
    queue->producer_enqueue({block, 1});
    queue->producer_set_finished();
    while (delayed.submitted.load(std::memory_order_acquire) < 4) backoff.backoff();
    delayed.ready.store(14, std::memory_order_release);
    writer.wait_for_normal_drain();
    require(!memory.first_empty(), "normal drain incorrectly completed pending source AIO");
    delayed.ready.store(15, std::memory_order_release);
    while (!memory.first_empty()) backoff.backoff();
    writer.finish_reclaim();
    writer.join();
    require(!delayed.overwritten, "compressed output overwritten before AIO completion");
    for (size_t i = 0; i < 4; ++i) require(delayed.returns[i] == 1, "AIO result was not collected exactly once");
    require(memory.gap_bytes(0) == gap + 3 * FIRST_SEGMENT_SIZE, "parallel AIO leaked or duplicated source credit");
    require(low_msd_test::read(path.string() + ".0", 9).size() == 3 * memory.first_segment_record_capacity() + 1,
        "parallel direct AIO lost output");
    std::filesystem::remove(path.string() + ".0");
}

int main(int argc, char** argv)
{
    try {
        k_length = 33;
        for (unsigned failure = 0; failure <= 3; ++failure) check_split_source_aio(failure);
        if (argc > 1 && std::string(argv[1]) == "--split-only") return 0;
        check_unclaimable_demand_does_not_rotate();
        check_independent_spill_retry();
        check_parallel_borrowed_aio();
        check_normal_drain_detaches_ring();
        check_idle_exit();
        check_close_preserves_first();
        check_close_wakes_allocator();
        RunState state;
        ConcurrentMemoryPool memory(32ULL << 20, &state);
        ConcurrentMemoryPoolTestAccess::single(memory);
    memory.init_arenas();
        std::vector<unsigned char> active(FIRST_SEGMENT_SIZE);
        std::array<unsigned char, 9> key{};
        const size_t records = ConcurrentMemoryPool::first_segment_record_capacity() * 3;
        for (size_t i = 0; i < records; ++i) {
            std::memcpy(key.data(), &i, sizeof(i));
            const size_t slot = i % ConcurrentMemoryPool::first_segment_record_capacity();
            std::memcpy(active.data() + slot * 9, key.data(), 9);
            if (slot + 1 == ConcurrentMemoryPool::first_segment_record_capacity())
                require(cache_full_first(memory, 0, active.data()), "test cache capacity");
        }
            auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
        // Saturate normal transport. Borrowed FIRST AIO must progress directly.
        for (size_t i = 0; i < queue->capacity(); ++i) {
            char* block; queue->producer_dequeue(block); queue->producer_enqueue({block, 0});
        }
        const auto path = std::filesystem::temp_directory_path() / ("first-spill-" + std::to_string(getpid()) + ".bin");
        SlowIo slow;
        slow_io = &slow;
        ExportWriter::IoOps slow_ops;
        slow_ops.submit = slow_submit;
        ExportWriter writer(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &memory, &slow_ops);
        writer.start();
        const size_t needed = memory.gap_bytes(0) + 3 * HUGE_PAGE_SIZE - BLOCK_SIZE;
        auto allocation = std::async(std::launch::async, [&] { return memory.allocate_large_from_numa(needed, 0); });
        slow.entered.get_future().wait();
        require(allocation.wait_for(std::chrono::seconds(0)) == std::future_status::timeout,
            "fixture did not hold a Tree request behind slow I/O");
        slow.resume.set_value();
        void* tree = allocation.get();
        require(tree, "independent spill did not unblock Tree");
        std::memset(tree, 0xd3, needed); // Reuse all returned source addresses immediately.
        memory.freeze_first();
        memory.request_first_drain();
        SpinBackoff<> backoff;
        while (!memory.first_empty()) {
            state.check();
            backoff.backoff();
        }
        queue->producer_set_finished();
        writer.join();
        std::vector<bool> seen(records);
        size_t read_records = 0;
        for (uint64_t i = 0; i < writer.shard_count(); ++i) {
            const auto shard = path.string() + "." + std::to_string(i);
            const size_t bytes = std::filesystem::file_size(shard);
            require(bytes < LOW_PENDING_SHARD_BYTES + msd_prefix::BUFFER_BYTES, "spill shard too large");
            for (const auto& record : low_msd_test::read(shard, kmer_bytes())) {
                size_t id; std::memcpy(&id, record.data(), sizeof(id));
                require(id < records && !seen[id], "spill multiset changed");
                seen[id] = true;
                ++read_records;
            }
            std::filesystem::remove(shard);
        }
        require(read_records == records, "spill lost or duplicated records");
        require(memory.returned_bytes.load() == 3 * HUGE_PAGE_SIZE, "frontier not returned");

        RunState failure;
        ConcurrentMemoryPool failing_memory(8ULL << 20, &failure);
        failing_memory.init_arenas();
        auto ops = ExportWriter::IoOps{};
        ops.result = short_return;
        ExportWriter bad(queue.get(), path.string(), ExportWriter::OutputKind::Pending, &failing_memory, &ops);
        queue->reset_producers(1);
        char* block; queue->producer_dequeue(block);
        queue->producer_enqueue({block, 1}); queue->producer_set_finished();
        bad.start();
        bool rejected = false;
        try { bad.join(); } catch (const std::exception&) { rejected = true; }
        require(rejected && failure.failed(), "short AIO write reported success");
        std::filesystem::remove(path.string() + ".0");
        std::cout << "FirstSpillWriter tests passed\n";
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
