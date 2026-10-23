#include "first_pool_fixture.h"
#include "../src/ExportWriter.h"
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
    std::atomic<unsigned> submitted{0}, ready{0};
};
static DelayedIo* delayed_io;
static int delayed_submit(aiocb* cb)
{
    const auto i = delayed_io->submitted.load(std::memory_order_relaxed);
    if (i == delayed_io->requests.size()) { errno = EIO; return -1; }
    delayed_io->requests[i] = cb;
    delayed_io->submitted.store(i + 1, std::memory_order_release);
    return 0;
}
static size_t delayed_index(const aiocb* cb)
{
    for (size_t i = 0; i < delayed_io->submitted.load(std::memory_order_acquire); ++i)
        if (delayed_io->requests[i] == cb) return i;
    std::terminate();
}
static int delayed_error(const aiocb* cb)
{ return delayed_io->ready.load(std::memory_order_acquire) & (1u << delayed_index(cb)) ? 0 : EINPROGRESS; }
static ssize_t delayed_result(aiocb* cb)
{
    ++delayed_io->returns[delayed_index(cb)];
    return ::pwrite(cb->aio_fildes, const_cast<const void*>(cb->aio_buf), cb->aio_nbytes, cb->aio_offset);
}

struct ConcurrentMemoryPoolTestAccess
{
    static void single(ConcurrentMemoryPool& pool) { pool.num_arenas_ = 1; pool.total_cpus_ = pool.cpus_per_node_[0] = 1; }
    static void filling(ConcurrentMemoryPool& pool) {
        FirstSegmentRef ref; unsigned char* data;
        if (!pool.begin_first_segment(0, ref, data)) throw std::runtime_error("Filling fixture capacity");
    }
    static size_t pending(ConcurrentMemoryPool& memory) {
        std::lock_guard lock(memory.arenas_[0].control_mutex);
        return memory.arenas_[0].pending_bytes;
    }
};

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
        require(std::filesystem::file_size(filename) % key.size() == 0, "shutdown split a record");
        std::ifstream input(filename, std::ios::binary);
        while (input.read(reinterpret_cast<char*>(key.data()), key.size())) {
            size_t id;
            std::memcpy(&id, key.data(), sizeof(id));
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
    content_type next;
    require(queue->consumer_try_dequeue(next) && next.length == 1 && static_cast<unsigned char>(next.data[0]) == 0x22,
        "pending Writer consumed the reset Ring");
    queue->consumer_enqueue(next.data);
    queue->producer_set_finished();
    require(std::filesystem::file_size(path.string() + ".0") == 9 + memory.first_segment_record_capacity() * 9,
        "normal-drain lost ordinary or raw FIRST output");
    std::filesystem::remove(path.string() + ".0");
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
    while (delayed.submitted.load(std::memory_order_acquire) < 3) backoff.backoff();
    // Complete the two higher-address sources first. Their in-flight data must
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
    for (size_t i = 0; i < 4; ++i) require(delayed.returns[i] == 1, "AIO result was not collected exactly once");
    require(memory.gap_bytes(0) == gap + 3 * FIRST_SEGMENT_SIZE, "parallel AIO leaked or duplicated source credit");
    require(std::filesystem::file_size(path.string() + ".0") == 3 * memory.first_segment_record_capacity() * 9 + 9,
        "parallel direct AIO lost output");
    std::filesystem::remove(path.string() + ".0");
}

int main()
{
    try {
        k_length = 33;
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
            require(bytes % kmer_bytes() == 0, "spill shard contains padding or a partial record");
            require(bytes < LOW_PENDING_SHARD_BYTES + FIRST_SEGMENT_SIZE, "spill shard too large");
            if (i + 1 < writer.shard_count())
                require(bytes >= LOW_PENDING_SHARD_BYTES, "spill shard rotated prematurely");
            std::ifstream input(shard, std::ios::binary);
            for (size_t j = 0; j < bytes / kmer_bytes(); ++j) {
                input.read(reinterpret_cast<char*>(key.data()), key.size());
                size_t id; std::memcpy(&id, key.data(), sizeof(id));
                require(input.good() && id < records && !seen[id], "spill multiset changed");
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
