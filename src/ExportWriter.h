#ifndef EXPORT_WRITER_HEADER
#define EXPORT_WRITER_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "ConcurrentMemoryPool.h"
#include "SpinBackoff.h"
#include <aio.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <cstring>
#include <thread>
#include <system_error>

class ExportWriter
{
public:
    enum class OutputKind { Pending, FinalLow };

    // System calls are injectable for deterministic short-write/error tests.
    struct IoOps
    {
        int (*submit)(aiocb*) = ::aio_write;
        int (*error)(const aiocb*) = ::aio_error;
        ssize_t (*result)(aiocb*) = ::aio_return;
        int (*suspend)(const aiocb* const[], int, const timespec*) = ::aio_suspend;
        int (*close)(int) = ::close;
    };
private:
    static constexpr size_t RAW_BUFFER_SIZE = 2ULL << 20;
    static constexpr size_t NUM_AIO_BUFFERS = 2;
    const size_t record_bytes = kmer_bytes();
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* ring_memory_pool;
    ConcurrentMemoryPool* memory_pool_;
    RunState local_state_;
    RunState* state_;
    const IoOps io_;
    const std::string filename_;
    const OutputKind output_kind_;
    uint64_t shard_count_ = 1;
    int fd = -1;
    std::array<aiocb, NUM_AIO_BUFFERS> cbs{};
    std::array<bool, NUM_AIO_BUFFERS> active{};
    std::array<char*, NUM_AIO_BUFFERS> buffers{};
    std::array<size_t, NUM_AIO_BUFFERS> used{};
    size_t current = 0;
    off_t file_offset = 0;
    std::thread worker;
    std::array<FirstSegmentRef, MAX_NUMA_NODES> spills_{};
    std::array<FirstSegmentDescriptor*, MAX_NUMA_NODES> spilling_{};
    std::array<size_t, MAX_NUMA_NODES> spill_offsets_{};
    int next_arena_ = 0;

    [[noreturn]] static void io_error(const char* operation)
    { throw std::system_error(errno ? errno : EIO, std::generic_category(), operation); }

public:
    explicit ExportWriter(RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* pool,
        const std::string& filename, OutputKind kind,
        ConcurrentMemoryPool* memory = nullptr, const IoOps* ops = nullptr)
        : ring_memory_pool(pool), memory_pool_(memory),
          state_(memory ? &memory->run_state() : RunState::active ? RunState::active : &local_state_),
          io_(ops ? *ops : IoOps{}), filename_(filename), output_kind_(kind)
    {
        const std::string path = kind == OutputKind::Pending ? filename + ".0" : filename;
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) io_error("open pending output");
    }
    ExportWriter(const ExportWriter&) = delete;
    ExportWriter& operator=(const ExportWriter&) = delete;

    void start()
    {
        // A successful start guarantees that both transport buffers exist before
        // any Classifier is permitted to depend on reclamation.
        for (auto& buffer : buffers) {
            void* storage = nullptr;
            if (posix_memalign(&storage, PAGE_SIZE, RAW_BUFFER_SIZE)) throw std::bad_alloc();
            buffer = static_cast<char*>(storage);
        }
        if (memory_pool_) memory_pool_->set_spill_service(true);
        try { worker = std::thread(&ExportWriter::writer_entry, this); }
        catch (...) {
            if (memory_pool_) memory_pool_->set_spill_service(false);
            throw;
        }
    }
    void join()
    {
        if (worker.joinable()) worker.join();
        close_file();
        state_->check();
    }
    // The writer thread owns this count until join() succeeds.
    uint64_t shard_count() const noexcept { return shard_count_; }
    ~ExportWriter()
    {
        if (std::uncaught_exceptions() && !state_->failed())
            state_->fail(std::make_exception_ptr(std::runtime_error("pipeline unwound")), "main");
        if (worker.joinable()) worker.join();
        cleanup_aio();
        try { close_file(); } catch (...) { state_->fail(std::current_exception(), "pending close"); }
        for (char* buffer : buffers) ::free(buffer);
    }
private:
    void close_file()
    {
        if (fd < 0) return;
        const int closing = fd;
        fd = -1;
        if (io_.close(closing)) {
            auto error = std::make_exception_ptr(std::system_error(errno, std::generic_category(), "close pending output"));
            state_->fail(error, "pending close");
            std::rethrow_exception(error);
        }
    }
    void writer_entry() noexcept
    {
        try { writer_loop(); }
        catch (...) { state_->fail(std::current_exception(), "pending writer"); }
        cleanup_aio();
        if (memory_pool_) {
            // On failure the run aborts; no held source may remain Consuming.
            for (int i = 0; i < memory_pool_->arena_count(); ++i) {
                if (spilling_[i]) memory_pool_->empty_first(spills_[i]);
                spilling_[i] = nullptr;
            }
            memory_pool_->set_spill_service(false);
        }
    }
    void writer_loop()
    {
        SpinBackoff<> backoff;
        bool closing = false;
        for (;;) {
            state_->check();
            if (!closing && ring_memory_pool->producer_finished()) {
                closing = true;
                if (memory_pool_) memory_pool_->set_spill_service(false);
            }
            bool worked = false;
            content_type content;
            // At most one 256 KiB normal block before reconsidering reclaim.
            if (ring_memory_pool->consumer_try_dequeue(content)) {
                auto* block = reinterpret_cast<ExportBlock*>(content.data);
                append(block->k_mers.data(), content.length * record_bytes);
                ring_memory_pool->consumer_enqueue(content.data);
                worked = true;
            }
            if (memory_pool_ && spill_step(!closing)) worked = true;
            if (!worked) {
                // Production ended before this round's queue/held-spill checks.
                if (closing) break;
                backoff.backoff();
            } else backoff.reset();
        }
        if (used[current]) submit(current);
        for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) wait_write(i);
    }
    bool spill_step(bool allow_claim)
    {
        const int arenas = memory_pool_->arena_count();
        for (int attempt = 0; attempt < arenas; ++attempt) {
            const int index = next_arena_;
            next_arena_ = (next_arena_ + 1) % arenas;
            if (!spilling_[index]) {
                if (!allow_claim) continue;
                spilling_[index] = memory_pool_->claim_spill(index, spills_[index]);
                if (!spilling_[index]) continue;
                spill_offsets_[index] = 0;
            }
            auto* segment = spilling_[index];
            const size_t count = std::min(EXPORT_RING_MEMORY_POOL_BLOCK_SIZE / record_bytes,
                ConcurrentMemoryPool::first_segment_record_capacity() - spill_offsets_[index]);
            append(reinterpret_cast<const unsigned char*>(segment->data) + spill_offsets_[index] * record_bytes, count * record_bytes);
            // Commit only after the copy is in Writer-owned transport.
            spill_offsets_[index] += count;
#ifdef TEST_MODE
            memory_pool_->spill_bytes.fetch_add(count * record_bytes, std::memory_order_relaxed);
#endif
            if (spill_offsets_[index] == ConcurrentMemoryPool::first_segment_record_capacity()) {
                memory_pool_->empty_first(spills_[index]);
                spilling_[index] = nullptr;
            }
            return true;
        }
        return false;
    }
    void append(const unsigned char* input, size_t bytes)
    {
        // append() receives complete records from either transport or FIRST spill.
        if (bytes && output_kind_ == OutputKind::Pending
            && static_cast<uint64_t>(file_offset) + used[current] >= LOW_PENDING_SHARD_BYTES) {
            if (used[current]) submit(current);
            for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) wait_write(i);
            close_file();
            const std::string path = filename_ + "." + std::to_string(shard_count_);
            fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) io_error("open pending shard");
            ++shard_count_;
            file_offset = 0;
        }
        while (bytes) {
            const size_t count = std::min(bytes, RAW_BUFFER_SIZE - used[current]);
            std::memcpy(buffers[current] + used[current], input, count);
            used[current] += count;
            input += count;
            bytes -= count;
            if (used[current] == RAW_BUFFER_SIZE) {
                submit(current);
                current = (current + 1) % NUM_AIO_BUFFERS;
                wait_write(current);
            }
        }
    }
    void submit(size_t i)
    {
        if (used[i] > static_cast<uint64_t>(std::numeric_limits<off_t>::max() - file_offset))
            throw std::overflow_error("pending file offset overflow");
        cbs[i] = {};
        cbs[i].aio_fildes = fd;
        cbs[i].aio_buf = buffers[i];
        cbs[i].aio_nbytes = used[i];
        cbs[i].aio_offset = file_offset;
        if (io_.submit(&cbs[i])) io_error("aio_write pending");
        active[i] = true;
        file_offset += used[i];
        used[i] = 0;
    }
    void wait_write(size_t i)
    {
        if (!active[i]) return;
        const aiocb* list[] = {&cbs[i]};
        const timespec timeout{0, 10000000};
        while (io_.error(&cbs[i]) == EINPROGRESS) {
            state_->check();
            if (io_.suspend(list, 1, &timeout) && errno != EINTR && errno != EAGAIN)
                io_error("aio_suspend pending");
        }
        const int error = io_.error(&cbs[i]);
        const ssize_t bytes = io_.result(&cbs[i]);
        active[i] = false;
        if (error) throw std::system_error(error, std::generic_category(), "aio completion pending");
        if (bytes != static_cast<ssize_t>(cbs[i].aio_nbytes)) throw std::runtime_error("short pending write");
    }
    void cleanup_aio() noexcept
    {
        // Even after cancellation/failure, AIO must stop borrowing memory before
        // buffers/descriptors are freed. No RunState check in this cleanup path.
        for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) {
            if (!active[i]) continue;
            const aiocb* list[] = {&cbs[i]};
            while (io_.error(&cbs[i]) == EINPROGRESS) io_.suspend(list, 1, nullptr);
            io_.result(&cbs[i]);
            active[i] = false;
        }
    }
};
#endif
