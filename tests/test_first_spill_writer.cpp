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

struct ConcurrentMemoryPoolTestAccess
{
    static void single(ConcurrentMemoryPool& pool) { pool.num_arenas_ = 1; pool.total_cpus_ = pool.cpus_per_node_[0] = 1; }
    static void filling(ConcurrentMemoryPool& pool) {
        auto& arena = pool.arenas_[0];
        std::lock_guard lock(arena.admission);
        if (!arena.reserve_first()) throw std::runtime_error("Filling fixture capacity");
        auto& segment = arena.segments[arena.segment_frontier++];
        ++segment.generation;
        segment.data = static_cast<char*>(arena.start_addr) + arena.first_offset;
        segment.state = FirstSegmentState::Filling;
    }
    static size_t pending(ConcurrentMemoryPool& memory) {
        std::lock_guard lock(memory.arenas_[0].admission);
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
    while (!ConcurrentMemoryPoolTestAccess::pending(memory)) {
        const auto progress = state.progress();
        if (!ConcurrentMemoryPoolTestAccess::pending(memory)) state.wait(progress);
    }
    queue->producer_set_finished();
    writer.join();
    require(allocation.get() && ConcurrentMemoryPoolTestAccess::pending(memory) == 0,
        "Writer shutdown stranded an allocation reservation");
    std::filesystem::remove(path.string() + ".0");
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
            require(memory.cache_first_segment(0, active.data()), "shutdown fixture exceeded FIRST capacity");
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
    // The first AIO submission interrupts a copy from a claimed segment.
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
    std::vector<unsigned char> buffer(FIRST_SEGMENT_SIZE);
    for (;;) {
        const auto status = memory.extract_first_segment(0, buffer.data());
        if (status == ConcurrentMemoryPool::FirstExtractStatus::Done) break;
        require(status == ConcurrentMemoryPool::FirstExtractStatus::Data, "closing left a Consuming segment");
        for (size_t i = 0; i < ConcurrentMemoryPool::first_segment_record_capacity(); ++i) {
            size_t id;
            std::memcpy(&id, buffer.data() + i * key.size(), sizeof(id));
            require(id < records && !seen[id], "pending and remaining FIRST overlap");
            seen[id] = true;
            ++copied;
        }
    }
    require(copied == records + 1 && memory.first_empty(), "shutdown lost FIRST records");
}

int main()
{
    try {
        k_length = 33;
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
                require(memory.cache_first_segment(0, active.data()), "test cache capacity");
        }
            auto queue = std::make_unique<RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>>(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, 1);
        // Saturate the normal transport. Reclamation must use Writer-owned AIO
        // buffers rather than enqueueing into the queue it is consuming.
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
        while (!memory.first_empty()) {
            auto progress = state.progress();
            if (!memory.first_empty()) state.wait(progress);
        }
        queue->producer_set_finished();
        writer.join();
        std::vector<bool> seen(records);
        size_t read_records = 0;
        for (uint64_t i = 0; i < writer.shard_count(); ++i) {
            const auto shard = path.string() + "." + std::to_string(i);
            const size_t bytes = std::filesystem::file_size(shard);
            require(bytes % kmer_bytes() == 0, "spill shard contains padding or a partial record");
            require(bytes < LOW_PENDING_SHARD_BYTES + EXPORT_RING_MEMORY_POOL_BLOCK_SIZE, "spill shard too large");
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
