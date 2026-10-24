#ifndef EXPORT_WRITER_HEADER
#define EXPORT_WRITER_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "ConcurrentMemoryPool.h"
#include "SpinBackoff.h"
#include "KtpFormat.h"

#include <memory>
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
    enum class ReclaimLifetime { InputOnly, UntilExplicitStop };
    struct IoOps {
        int (*submit)(aiocb*) = ::aio_write;
        int (*error)(const aiocb*) = ::aio_error;
        ssize_t (*result)(aiocb*) = ::aio_return;
        int (*suspend)(const aiocb* const[], int, const timespec*) = ::aio_suspend;
        int (*close)(int) = ::close;
    };
private:
    static constexpr size_t RAW_BUFFER_SIZE = EXPORT_WRITER_BUFFER_BYTES;
    static constexpr size_t NUM_AIO_BUFFERS = EXPORT_WRITER_BUFFER_COUNT;
    static constexpr size_t FIRST_AIO_SLOTS = 8;
    struct alignas(CACHE_LINE_SIZE) BorrowedWrite {
        aiocb cb{};
        ConcurrentMemoryPool::FirstSegmentView view{};
        bool owned = false;
        bool active = false;
    };
    const size_t record_bytes = kmer_bytes();
    const size_t usable_bytes = RAW_BUFFER_SIZE / record_bytes * record_bytes;
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* ring_memory_pool;
    ConcurrentMemoryPool* memory_pool_;
    RunState local_state_;
    RunState* state_;
    const IoOps io_;
    const std::string filename_;
    const OutputKind output_kind_;
    const ReclaimLifetime lifetime_;
    uint64_t shard_count_ = 1;
    int fd = -1;
    off_t file_offset = 0;
    std::array<aiocb, NUM_AIO_BUFFERS> cbs{};
    std::array<bool, NUM_AIO_BUFFERS> active{};
    std::array<char*, NUM_AIO_BUFFERS> buffers{};
    std::array<size_t, NUM_AIO_BUFFERS> used{};
    std::array<BorrowedWrite, FIRST_AIO_SLOTS> borrowed_{};
    size_t current = 0;
    int next_arena_ = 0;
    std::unique_ptr<ktp::Writer> ktp_;
    bool ktp_written_ = false;
    content_type held_{};
    size_t held_offset_ = 0;
    bool normal_empty_ = false;
    bool normal_detached_ = false;
    std::atomic<bool> normal_drained_{false};
    std::atomic<bool> finish_requested_{false};
    std::thread worker;

    [[noreturn]] static void io_error(const char* operation)
    { throw std::system_error(errno ? errno : EIO, std::generic_category(), operation); }

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
    bool any_active() const noexcept
    {
        for (bool a : active) if (a) return true;
        for (const auto& slot : borrowed_) if (slot.active) return true;
        return false;
    }
    bool any_borrowed() const noexcept
    {
        for (const auto& slot : borrowed_) if (slot.owned) return true;
        return false;
    }
    bool pending_borrowed() const noexcept
    {
        for (const auto& slot : borrowed_) if (slot.owned && !slot.active) return true;
        return false;
    }
    bool prepare_output()
    {
        if (output_kind_ != OutputKind::Pending || static_cast<uint64_t>(file_offset) < LOW_PENDING_SHARD_BYTES) return true;
        if (any_active()) return false;
        close_file();
        const std::string path = filename_ + "." + std::to_string(shard_count_);
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) io_error("open pending shard");
        ++shard_count_;
        file_offset = 0;
        return true;
    }
    void submit(aiocb& cb, const void* data, size_t bytes)
    {
        if (bytes > static_cast<uint64_t>(std::numeric_limits<off_t>::max() - file_offset))
            throw std::overflow_error("pending file offset overflow");
        cb = {};
        cb.aio_fildes = fd;
        cb.aio_buf = const_cast<void*>(data);
        cb.aio_nbytes = bytes;
        cb.aio_offset = file_offset;
        if (io_.submit(&cb)) io_error("aio_write pending");
        file_offset += bytes;
    }
    bool submit_normal(size_t i)
    {
        if (!used[i]) return true;
        if (!prepare_output()) return false;
        submit(cbs[i], buffers[i], used[i]);
        active[i] = true;
        used[i] = 0;
        return true;
    }
    void collect(aiocb& cb, bool& is_active)
    {
        const int error = io_.error(&cb);
        const ssize_t bytes = io_.result(&cb);
        is_active = false; // aio_return is called once even if checking throws.
        if (error) throw std::system_error(error, std::generic_category(), "aio completion pending");
        if (bytes != static_cast<ssize_t>(cb.aio_nbytes)) throw std::runtime_error("short pending write");
    }
    bool harvest()
    {
        bool worked = false;
        for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) {
            if (!active[i] || io_.error(&cbs[i]) == EINPROGRESS) continue;
            collect(cbs[i], active[i]);
            worked = true;
        }
        for (auto& slot : borrowed_) {
            if (!slot.active || io_.error(&slot.cb) == EINPROGRESS) continue;
            collect(slot.cb, slot.active);
#ifdef TEST_MODE
            memory_pool_->spill_bytes.fetch_add(slot.cb.aio_nbytes, std::memory_order_relaxed);
#endif
            memory_pool_->complete_first(slot.view);
            slot.owned = false;
            worked = true;
        }
        return worked;
    }
    void absorb(const void* raw, size_t begin, size_t end)
    {
        const char* data = static_cast<const char*>(raw);
        for (size_t off = begin; off < end; off += record_bytes)
            ktp_->add(data + off, 1);
    }
    void finalize_ktp()
    {
        if (!ktp_ || ktp_written_) return;
        ktp_written_ = true;
        ktp_->finish();
    }
    bool normal_step()
    {
        if (!held_.data) {
            if (!ring_memory_pool->consumer_try_dequeue(held_)) { normal_empty_ = true; return false; }
            normal_empty_ = false;
            held_offset_ = 0;
        }
        if (held_offset_ != held_.length * record_bytes) {
            if (ktp_) {
                absorb(held_.data, held_offset_, held_.length * record_bytes);
                held_offset_ = held_.length * record_bytes;
            } else {
            if (pending_borrowed()) return false;
            if (active[current]) {
                const size_t other = (current + 1) % NUM_AIO_BUFFERS;
                if (active[other]) return false;
                current = other;
            }
            if (used[current] == usable_bytes && !submit_normal(current)) return false;
            if (active[current]) return true;
            const size_t bytes = std::min(held_.length * record_bytes - held_offset_, usable_bytes - used[current]);
            auto* block = reinterpret_cast<ExportBlock*>(held_.data);
            std::memcpy(buffers[current] + used[current], block->k_mers.data() + held_offset_, bytes);
            used[current] += bytes;
            held_offset_ += bytes;
            if (used[current] == usable_bytes) submit_normal(current);
            }
        }
        if (held_offset_ == held_.length * record_bytes && ring_memory_pool->consumer_try_enqueue(held_.data)) held_ = {};
        return true;
    }
    bool spill_step(bool allow_claim)
    {
        BorrowedWrite* pending = nullptr;
        for (auto& slot : borrowed_) if (slot.owned && !slot.active) { pending = &slot; break; }
        if (!pending && allow_claim) {
            for (auto& slot : borrowed_) if (!slot.owned) { pending = &slot; break; }
            if (!pending || !prepare_output()) return false;
            const int arenas = memory_pool_->arena_count();
            const int preferred = next_arena_;
            next_arena_ = (next_arena_ + 1) % arenas;
            if (memory_pool_->try_claim_first(preferred, ConcurrentMemoryPool::FirstConsumer::Spill, pending->view)
                != ConcurrentMemoryPool::FirstClaimStatus::Data) return false;
            pending->owned = true;
        }
        if (!pending) return false;
        const size_t bytes = pending->view.record_count * record_bytes;
        if (!bytes) {
            memory_pool_->complete_first(pending->view);
            pending->owned = false;
            return true;
        }
        if (ktp_) {
            absorb(pending->view.records, 0, bytes);
            memory_pool_->complete_first(pending->view);
            pending->owned = false;
            return true;
        }
        // Partial ordinary output precedes the direct source in the byte stream.
        // Every submission and insertion boundary contains complete records.
        for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) if (!submit_normal(i)) return false;
        if (!prepare_output()) return false;
        submit(pending->cb, pending->view.records, bytes);
        pending->active = true;
        return true;
    }
    void writer_loop()
    {
        SpinBackoff<> backoff;
        bool closing = false;
        for (;;) {
            state_->check();
            if (!normal_detached_ && !closing && ring_memory_pool->producer_finished()) {
                closing = true;
                normal_empty_ = false;
                if (memory_pool_ && lifetime_ == ReclaimLifetime::InputOnly) memory_pool_->set_spill_service(false);
            }
            bool worked = harvest();
            if (memory_pool_) {
                for (int i = 0; i < memory_pool_->arena_count(); ++i) memory_pool_->poll_first_progress(i);
                if (pending_borrowed()) worked = spill_step(false) || worked;
            }
            if (!normal_detached_) worked = normal_step() || worked;
            if (memory_pool_ && !finish_requested_.load(std::memory_order_acquire)
                && (!closing || lifetime_ == ReclaimLifetime::UntilExplicitStop)) worked = spill_step(true) || worked;
            if (closing && !normal_detached_ && !held_.data && normal_empty_) {
                for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i)
                    if (used[i] && submit_normal(i)) worked = true;
                bool normal_busy = false;
                for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) normal_busy = normal_busy || active[i] || used[i];
                if (!normal_busy) {
                    normal_detached_ = true;
                    for (char*& buffer : buffers) { ::free(buffer); buffer = nullptr; }
                    normal_drained_.store(true, std::memory_order_release);
                    state_->notify();
                    worked = true;
                }
            }
            if (normal_detached_ && (lifetime_ == ReclaimLifetime::InputOnly || finish_requested_.load(std::memory_order_acquire))
                && !any_active() && !any_borrowed()) break;
            if (worked) backoff.reset();
            else backoff.backoff();
        }
    }
    void cleanup_aio() noexcept
    {
        auto finish = [this](aiocb& cb, bool& is_active) {
            if (!is_active) return;
            const aiocb* list[] = {&cb};
            while (io_.error(&cb) == EINPROGRESS) io_.suspend(list, 1, nullptr);
            io_.result(&cb);
            is_active = false;
        };
        for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) finish(cbs[i], active[i]);
        for (auto& slot : borrowed_) {
            finish(slot.cb, slot.active);
            if (slot.owned) { memory_pool_->complete_first(slot.view); slot.owned = false; }
        }
    }
    void writer_entry() noexcept
    {
        try { writer_loop(); }
        catch (...) { state_->fail(std::current_exception(), "pending writer"); }
        cleanup_aio();
        if (memory_pool_) {
            for (int i = 0; i < memory_pool_->arena_count(); ++i) memory_pool_->poll_first_progress(i);
            memory_pool_->set_spill_service(false);
        }
    }
public:
    explicit ExportWriter(RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* pool,
        const std::string& filename, OutputKind kind,
        ConcurrentMemoryPool* memory = nullptr, const IoOps* ops = nullptr,
        ReclaimLifetime lifetime = ReclaimLifetime::InputOnly)
        : ring_memory_pool(pool), memory_pool_(memory),
          state_(memory ? &memory->run_state() : RunState::active ? RunState::active : &local_state_),
          io_(ops ? *ops : IoOps{}), filename_(filename), output_kind_(kind), lifetime_(lifetime)
    {
        if (kind == OutputKind::FinalLow)
            ktp_ = std::make_unique<ktp::Writer>(filename, k_length, /*has_counts=*/false);
        if (!ktp_) {
            const std::string path = kind == OutputKind::Pending ? filename + ".0" : filename;
            fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd < 0) io_error("open pending output");
        }
    }
    ExportWriter(const ExportWriter&) = delete;
    ExportWriter& operator=(const ExportWriter&) = delete;
    void start()
    {
        for (char*& buffer : buffers) {
            void* storage = nullptr;
            if (posix_memalign(&storage, PAGE_SIZE, RAW_BUFFER_SIZE)) throw std::bad_alloc();
            buffer = static_cast<char*>(storage);
        }
        if (memory_pool_) memory_pool_->set_spill_service(true);
        try { worker = std::thread(&ExportWriter::writer_entry, this); }
        catch (...) { if (memory_pool_) memory_pool_->set_spill_service(false); throw; }
    }
    void wait_for_normal_drain()
    {
        while (!normal_drained_.load(std::memory_order_acquire)) {
            const auto progress = state_->progress();
            if (!normal_drained_.load(std::memory_order_acquire)) state_->wait(progress);
        }
        state_->check();
    }
    void finish_reclaim() noexcept { finish_requested_.store(true, std::memory_order_release); }
    void join()
    {
        if (worker.joinable()) worker.join();
        finalize_ktp();
        close_file();
        state_->check();
    }
    uint64_t shard_count() const noexcept { return shard_count_; }
    ~ExportWriter()
    {
        if (std::uncaught_exceptions() && !state_->failed())
            state_->fail(std::make_exception_ptr(std::runtime_error("pipeline unwound")), "main");
        finish_reclaim();
        if (worker.joinable()) worker.join();
        try { finalize_ktp(); } catch (...) { state_->fail(std::current_exception(), "finalize low"); }
        cleanup_aio();
        try { close_file(); } catch (...) { state_->fail(std::current_exception(), "pending close"); }
        for (char* buffer : buffers) ::free(buffer);
    }
};
#endif
