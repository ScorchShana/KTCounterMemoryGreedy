#ifndef EXPORT_WRITER_HEADER
#define EXPORT_WRITER_HEADER

#include "definition.h"
#include "RingMemoryPool.h"
#include "ConcurrentMemoryPool.h"
#include "SpinBackoff.h"
#include "MsdPrefixCodec.h"
#include <aio.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <cstring>
#include <thread>
#include <system_error>

class ExportWriter
{
    friend struct ExportWriterTestAccess;
public:
    enum class OutputKind { Pending, FinalLow };
    enum class ReclaimLifetime { InputOnly, UntilExplicitStop };
    struct IoOps {
        int (*submit)(aiocb*) = ::aio_write;
        int (*error)(const aiocb*) = ::aio_error;
        ssize_t(*result)(aiocb*) = ::aio_return;
        int (*suspend)(const aiocb* const [], int, const timespec*) = ::aio_suspend;
        int (*close)(int) = ::close;
    };
private:
    static constexpr size_t RAW_BUFFER_SIZE = EXPORT_WRITER_BUFFER_BYTES;
    static constexpr size_t NUM_AIO_BUFFERS = EXPORT_WRITER_BUFFER_COUNT;
    struct FirstSource {
        ConcurrentMemoryPool::FirstSegmentView view{};
        size_t encoded_records = 0;
        size_t outstanding_outputs = 0; // Includes encoded but not yet submitted slots.
        bool owned = false;
    };
    const size_t record_bytes = kmer_bytes();
    const size_t usable_bytes = msd_prefix::batch_records(record_bytes) * record_bytes;
    RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>* ring_memory_pool;
    ConcurrentMemoryPool* memory_pool_;
    RunState local_state_;
    RunState* state_;
    const IoOps io_;
    const std::string filename_;
    const OutputKind output_kind_;
    const ReclaimLifetime lifetime_;
    uint64_t shard_count_ = 1, encoded_bytes_ = 0;
    uint64_t shard_limit_bytes_ = LOW_PENDING_SHARD_BYTES;
    int fd = -1;
    off_t file_offset = 0;
    std::array<aiocb, NUM_AIO_BUFFERS> cbs{};
    std::array<bool, NUM_AIO_BUFFERS> active{};
    std::array<char*, NUM_AIO_BUFFERS> buffers{};
    std::array<size_t, NUM_AIO_BUFFERS> used{};
    std::array<int, NUM_AIO_BUFFERS> output_sources_{-1, -1};
    uint8_t* low_input_ = nullptr;
    uint8_t* low_scratch_ = nullptr;
    std::array<FirstSource, NUM_AIO_BUFFERS> borrowed_{};
    size_t filled_ = 0, current = 0;
    int next_arena_ = 0;
    unsigned spill_retry_skip_ = 0;
    unsigned spill_retry_window_ = 1;
    content_type held_{};
    size_t held_offset_ = 0;
    bool normal_empty_ = false;
    bool normal_detached_ = false;
    std::atomic<bool> normal_drained_{false};
    std::atomic<bool> finish_requested_{false};
    std::thread worker;

    [[noreturn]] static void io_error(const char* operation)
    {
        throw std::system_error(errno ? errno : EIO, std::generic_category(), operation);
    }
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
        return false;
    }
    bool any_borrowed() const noexcept
    {
        for (const auto& source : borrowed_) if (source.owned) return true;
        return false;
    }
    bool pending_borrowed() const noexcept
    {
        for (const auto& source : borrowed_)
            if (source.owned && source.encoded_records < source.view.record_count) return true;
        return false;
    }
    int available_output() const noexcept
    {
        for (size_t n = 0; n < NUM_AIO_BUFFERS; ++n) {
            const size_t i = (current + n) % NUM_AIO_BUFFERS;
            if (!active[i] && !used[i]) return static_cast<int>(i);
        }
        return -1;
    }
    void reset_spill_retry() noexcept
    {
        spill_retry_skip_ = 0;
        spill_retry_window_ = 1;
    }
    bool prepare_output()
    {
        if (output_kind_ != OutputKind::Pending || static_cast<uint64_t>(file_offset) < shard_limit_bytes_) return true;
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
        is_active = false;
        if (error) throw std::system_error(error, std::generic_category(), "aio completion pending");
        if (bytes != static_cast<ssize_t>(cb.aio_nbytes)) throw std::runtime_error("short pending write");
        encoded_bytes_ += cb.aio_nbytes;
    }
    void complete_source(FirstSource& source)
    {
#ifdef TEST_MODE
        memory_pool_->spill_bytes.fetch_add(source.view.record_count * record_bytes, std::memory_order_relaxed);
#endif
        memory_pool_->complete_first(source.view);
        source = {};
        reset_spill_retry();
    }
    bool harvest()
    {
        bool worked = false;
        for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) {
            if (!active[i] || io_.error(&cbs[i]) == EINPROGRESS) continue;
            collect(cbs[i], active[i]);
            if (output_sources_[i] >= 0) {
                auto& source = borrowed_[output_sources_[i]];
                --source.outstanding_outputs;
                if (source.encoded_records == source.view.record_count && !source.outstanding_outputs)
                    complete_source(source);
            }
            output_sources_[i] = -1;
            worked = true;
        }
        return worked;
    }
    bool encode_output(size_t count, int source_index)
    {
        const int slot = available_output();
        if (slot < 0) return false;
        used[slot] = msd_prefix::encode(low_input_, low_scratch_, static_cast<uint32_t>(count),
            record_bytes, reinterpret_cast<uint8_t*>(buffers[slot]));
        output_sources_[slot] = source_index;
        if (source_index >= 0) {
            auto& source = borrowed_[source_index];
            source.encoded_records += count;
            ++source.outstanding_outputs;
        } else filled_ = 0;
        submit_normal(slot);
        current = (slot + 1) % NUM_AIO_BUFFERS;
        return true;
    }
    bool normal_step()
    {
        if (!held_.data) {
            if (!ring_memory_pool->consumer_try_dequeue(held_)) { normal_empty_ = true; return false; }
            normal_empty_ = false;
            held_offset_ = 0;
        }
        if (held_offset_ != held_.length * record_bytes) {
            if (pending_borrowed()) return false;
            if (filled_ == usable_bytes && !encode_output(filled_ / record_bytes, -1)) return false;
            const size_t bytes = std::min(held_.length * record_bytes - held_offset_, usable_bytes - filled_);
            auto* block = reinterpret_cast<ExportBlock*>(held_.data);
            std::memcpy(low_input_ + filled_, block->k_mers.data() + held_offset_, bytes);
            filled_ += bytes;
            held_offset_ += bytes;
            if (filled_ == usable_bytes) encode_output(filled_ / record_bytes, -1);
        }
        if (held_offset_ == held_.length * record_bytes && ring_memory_pool->consumer_try_enqueue(held_.data)) held_ = {};
        return true;
    }
    bool spill_step(bool allow_claim)
    {
        int source_index = -1;
        for (size_t i = 0; i < borrowed_.size(); ++i)
            if (borrowed_[i].owned && borrowed_[i].encoded_records < borrowed_[i].view.record_count) {
                source_index = static_cast<int>(i);
                break;
            }
        if (source_index < 0 && allow_claim) {
            for (size_t i = 0; i < borrowed_.size(); ++i)
                if (!borrowed_[i].owned) { source_index = static_cast<int>(i); break; }
            if (source_index < 0) return false;
            if (!memory_pool_->writer_work_pending()) { reset_spill_retry(); return false; }
            if (spill_retry_skip_) { --spill_retry_skip_; return false; }
            if (available_output() < 0) return false;
            const int preferred = next_arena_;
            next_arena_ = (next_arena_ + 1) % memory_pool_->arena_count();
            auto& source = borrowed_[source_index];
            if (memory_pool_->try_claim_first(preferred, ConcurrentMemoryPool::FirstConsumer::Spill, source.view)
                != ConcurrentMemoryPool::FirstClaimStatus::Data) {
                if (memory_pool_->writer_work_pending()) {
                    spill_retry_skip_ = spill_retry_window_;
                    spill_retry_window_ = std::min(8u, spill_retry_window_ * 2);
                } else reset_spill_retry();
                return false;
            }
            source.owned = true;
            reset_spill_retry();
            if (!source.view.record_count) { complete_source(source); return true; }
        }
        if (source_index < 0) return false;
        if (filled_) return encode_output(filled_ / record_bytes, -1);
        if (available_output() < 0) return false;
        auto& source = borrowed_[source_index];
        const size_t count = std::min(msd_prefix::batch_records(record_bytes), source.view.record_count - source.encoded_records);
        std::memcpy(low_input_, source.view.records + source.encoded_records * record_bytes, count * record_bytes);
        return encode_output(count, source_index);
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
            for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i)
                if (used[i] && submit_normal(i)) worked = true;
            if (memory_pool_ && pending_borrowed()) worked = spill_step(false) || worked;
            if (!normal_detached_) worked = normal_step() || worked;
            if (memory_pool_ && !finish_requested_.load(std::memory_order_acquire)
                && (!closing || lifetime_ == ReclaimLifetime::UntilExplicitStop)) worked = spill_step(true) || worked;
            if (closing && !normal_detached_ && !held_.data && normal_empty_) {
                if (filled_ && encode_output(filled_ / record_bytes, -1)) worked = true;
                bool normal_busy = filled_ != 0;
                for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i)
                    normal_busy = normal_busy || ((active[i] || used[i]) && output_sources_[i] < 0);
                if (!normal_busy) {
                    normal_detached_ = true;
                    // FIRST may still need the codec and output slots after handoff.
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
    void final_low_loop()
    {
        const size_t capacity = msd_prefix::batch_records(record_bytes) * record_bytes;
        size_t filled = 0;
        bool closing = false;
        SpinBackoff<> backoff;
        for (;;) {
            state_->check();
            if (!closing && ring_memory_pool->producer_finished()) {
                closing = true;
                normal_empty_ = false;
            }
            bool worked = harvest();
            if (!held_.data) {
                if (ring_memory_pool->consumer_try_dequeue(held_)) {
                    held_offset_ = 0;
                    normal_empty_ = false;
                    worked = true;
                } else normal_empty_ = true;
            }
            if (held_.data) {
                const size_t bytes = std::min(held_.length * record_bytes - held_offset_, capacity - filled);
                if (bytes) {
                    auto* block = reinterpret_cast<ExportBlock*>(held_.data);
                    std::memcpy(low_input_ + filled, block->k_mers.data() + held_offset_, bytes);
                    filled += bytes;
                    held_offset_ += bytes;
                    worked = true;
                }
                if (held_offset_ == held_.length * record_bytes && ring_memory_pool->consumer_try_enqueue(held_.data)) {
                    held_ = {};
                    worked = true;
                }
            }
            if (filled == capacity || (filled && closing && normal_empty_ && !held_.data)) {
                if (active[current]) current = (current + 1) % NUM_AIO_BUFFERS;
                if (!active[current]) {
                    const size_t bytes = msd_prefix::encode(low_input_, low_scratch_,
                        static_cast<uint32_t>(filled / record_bytes), record_bytes,
                        reinterpret_cast<uint8_t*>(buffers[current]));
                    submit(cbs[current], buffers[current], bytes);
                    active[current] = true;
                    filled = 0;
                    current = (current + 1) % NUM_AIO_BUFFERS;
                    worked = true;
                }
            }
            if (closing && normal_empty_ && !held_.data && !filled && !any_active()) {
                release_buffers();
                normal_drained_.store(true, std::memory_order_release);
                state_->notify();
                return;
            }
            if (worked) backoff.reset();
            else backoff.backoff();
        }
    }
    void release_buffers() noexcept
    {
        for (char*& buffer : buffers) { ::free(buffer); buffer = nullptr; }
        ::free(low_input_); low_input_ = nullptr;
        ::free(low_scratch_); low_scratch_ = nullptr;
    }
    void cleanup_aio() noexcept
    {
        for (size_t i = 0; i < NUM_AIO_BUFFERS; ++i) {
            if (!active[i]) continue;
            const aiocb* list[] = {&cbs[i]};
            while (io_.error(&cbs[i]) == EINPROGRESS) io_.suspend(list, 1, nullptr);
            io_.result(&cbs[i]);
            active[i] = false;
        }
        for (auto& source : borrowed_) if (source.owned) {
            memory_pool_->complete_first(source.view);
            source = {};
        }
    }
    void writer_entry() noexcept
    {
        try {
            if (output_kind_ == OutputKind::FinalLow) final_low_loop();
            else writer_loop();
        } catch (...) { state_->fail(std::current_exception(), "pending writer"); }
        cleanup_aio();
        release_buffers();
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
        const std::string path = kind == OutputKind::Pending ? filename + ".0" : filename;
        fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) io_error("open pending output");
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
        {
            static_assert(RAW_BUFFER_SIZE == msd_prefix::BUFFER_BYTES && NUM_AIO_BUFFERS == 2);
            void* storage = nullptr;
            if (posix_memalign(&storage, PAGE_SIZE, msd_prefix::BUFFER_BYTES)) throw std::bad_alloc();
            low_input_ = static_cast<uint8_t*>(storage);
            if (posix_memalign(&storage, PAGE_SIZE, msd_prefix::BUFFER_BYTES)) throw std::bad_alloc();
            low_scratch_ = static_cast<uint8_t*>(storage);
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
        close_file();
        state_->check();
    }
    uint64_t shard_count() const noexcept { return shard_count_; }
    uint64_t encoded_bytes_written() const noexcept { return encoded_bytes_; }
    ~ExportWriter()
    {
        if (std::uncaught_exceptions() && !state_->failed())
            state_->fail(std::make_exception_ptr(std::runtime_error("pipeline unwound")), "main");
        finish_reclaim();
        if (worker.joinable()) worker.join();
        cleanup_aio();
        try { close_file(); }
        catch (...) { state_->fail(std::current_exception(), "pending close"); }
        release_buffers();
    }
};
#endif
