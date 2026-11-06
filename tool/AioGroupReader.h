#ifndef AIO_GROUP_READER_HEADER
#define AIO_GROUP_READER_HEADER

#include <aio.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

// Sequential AIO reader that hands whole byte windows to one parser thread.
// Reads are submitted on all free slots; completed slots are queued until the
// parser calls next() and later release(). A slot stays untouched by the
// reader while its data is being parsed.
class AioGroupReader
{
public:
    static constexpr size_t AIO_BUFFER_BYTES = 2ULL << 20;
    static constexpr size_t AIO_DEPTH = 4;

    AioGroupReader() = default;
    AioGroupReader(const AioGroupReader&) = delete;
    AioGroupReader& operator=(const AioGroupReader&) = delete;
    AioGroupReader(AioGroupReader&&) = delete;
    AioGroupReader& operator=(AioGroupReader&&) = delete;

    ~AioGroupReader()
    {
        join();
    }

    // The caller must not call open() twice, and must call join() before
    // destroying the object if it was opened.
    void open(const std::string& filename)
    {
        if (worker_.joinable()) throw std::logic_error("AioGroupReader opened twice");

        fd_ = ::open(filename.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0)
            throw std::system_error(errno, std::generic_category(), "open " + filename);

        struct stat st{};
        if (::fstat(fd_, &st) != 0)
        {
            const int error = errno;
            ::close(fd_);
            fd_ = -1;
            throw std::system_error(error, std::generic_category(), "stat " + filename);
        }
        if (!S_ISREG(st.st_mode) || st.st_size < 0)
        {
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error("not a regular file: " + filename);
        }

        file_size_ = static_cast<uint64_t>(st.st_size);
        ::posix_fadvise(fd_, 0, st.st_size, POSIX_FADV_SEQUENTIAL);

        for (auto& slot : slots_)
        {
            slot.buffer = std::make_unique<uint8_t[]>(AIO_BUFFER_BYTES);
            slot.requested = 0;
            slot.offset = 0;
            slot.state = State::FREE;
            std::memset(&slot.cb, 0, sizeof(slot.cb));
        }
        ready_head_ = 0;
        ready_size_ = 0;
        next_offset_ = 0;
        publish_offset_ = 0;
        stop_ = false;
        finished_ = false;
        error_ = nullptr;

        try
        {
            worker_ = std::thread([this] { run(); });
        }
        catch (...)
        {
            ::close(fd_);
            fd_ = -1;
            throw;
        }
    }

    // Blocks until one read completes, EOF, or an error. The returned window
    // stays valid until release(). The parser may hold at most one window.
    bool next(const uint8_t*& data, size_t& bytes)
    {
        if (current_ != npos) throw std::logic_error("AioGroupReader next() before release()");

        std::unique_lock lock(mutex_);
        ready_cv_.wait(lock, [this] {
            return error_ || stop_ || finished_ || ready_size_ != 0;
        });

        if (error_) std::rethrow_exception(error_);
        if (ready_size_ == 0) return false;

        current_ = ready_[ready_head_];
        ready_head_ = (ready_head_ + 1) % AIO_DEPTH;
        --ready_size_;

        Slot& slot = slots_[current_];
        slot.state = State::PARSING;
        data = slot.buffer.get();
        bytes = slot.requested;
        return true;
    }

    void release()
    {
        std::lock_guard lock(mutex_);
        if (current_ == npos) return;
        slots_[current_].state = State::FREE;
        current_ = npos;
        free_cv_.notify_one();
    }

    uint64_t file_size() const noexcept
    {
        return file_size_;
    }

    // Requests an early stop. Reads already submitted are drained by the
    // reader thread; next() then returns any queued windows, and false after.
    void stop()
    {
        {
            std::lock_guard lock(mutex_);
            stop_ = true;
        }
        ready_cv_.notify_all();
        free_cv_.notify_all();
    }

    void join()
    {
        if (!worker_.joinable()) return;
        stop();
        worker_.join();
    }

private:
    enum class State : uint8_t
    {
        FREE,
        READING,
        COMPLETED,
        READY,
        PARSING
    };

    struct Slot
    {
        std::unique_ptr<uint8_t[]> buffer;
        struct aiocb cb;
        size_t requested = 0;
        uint64_t offset = 0;
        State state = State::FREE;
    };

    static constexpr size_t npos = static_cast<size_t>(-1);

    int fd_ = -1;
    uint64_t file_size_ = 0;
    uint64_t next_offset_ = 0;
    uint64_t publish_offset_ = 0;
    std::array<Slot, AIO_DEPTH> slots_;
    std::array<size_t, AIO_DEPTH> ready_{};
    size_t ready_head_ = 0;
    size_t ready_size_ = 0;
    size_t current_ = npos;

    std::mutex mutex_;
    std::condition_variable ready_cv_;
    std::condition_variable free_cv_;
    std::thread worker_;
    std::exception_ptr error_;
    bool stop_ = false;
    bool finished_ = false;

    bool any_free() const
    {
        for (const auto& slot : slots_)
            if (slot.state == State::FREE) return true;
        return false;
    }

    void submit(Slot& slot)
    {
        const size_t bytes = static_cast<size_t>(
            std::min<uint64_t>(AIO_BUFFER_BYTES, file_size_ - next_offset_));
        slot.requested = bytes;
        slot.offset = next_offset_;
        next_offset_ += bytes;

        std::memset(&slot.cb, 0, sizeof(slot.cb));
        slot.cb.aio_fildes = fd_;
        slot.cb.aio_buf = slot.buffer.get();
        slot.cb.aio_nbytes = bytes;
        slot.cb.aio_offset = static_cast<off_t>(slot.offset);
        slot.cb.aio_sigevent.sigev_notify = SIGEV_NONE;
        slot.state = State::READING;

        if (::aio_read(&slot.cb) != 0)
        {
            slot.state = State::FREE;
            throw std::system_error(errno, std::generic_category(), "aio_read");
        }
    }

    // Moves completed reads to COMPLETED, then publishes the contiguous
    // offset prefix to the ready queue so the parser always sees file order.
    // Called with the mutex held.
    void harvest()
    {
        for (auto& slot : slots_)
        {
            if (slot.state != State::READING) continue;
            const int error = ::aio_error(&slot.cb);
            if (error == EINPROGRESS) continue;
            if (error == -1)
            {
                const int saved = errno;
                ::aio_return(&slot.cb);
                slot.state = State::FREE;
                throw std::system_error(saved, std::generic_category(), "aio_error");
            }

            const ssize_t n = ::aio_return(&slot.cb);
            slot.state = State::FREE;
            if (error != 0)
                throw std::system_error(error, std::generic_category(), "aio read");
            if (n != static_cast<ssize_t>(slot.requested))
                throw std::runtime_error("short aio read");
            slot.state = State::COMPLETED;
        }

        bool moved = false;
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (size_t i = 0; i < AIO_DEPTH; ++i)
            {
                Slot& slot = slots_[i];
                if (slot.state != State::COMPLETED || slot.offset != publish_offset_) continue;
                ready_[(ready_head_ + ready_size_) % AIO_DEPTH] = i;
                ++ready_size_;
                publish_offset_ += slot.requested;
                slot.state = State::READY;
                progress = moved = true;
            }
        }
        if (moved) ready_cv_.notify_all();
    }

    void read_loop()
    {
        std::unique_lock lock(mutex_);
        for (;;)
        {
            if (stop_) break;

            for (;;)
            {
                if (stop_ || next_offset_ == file_size_) break;
                Slot* free_slot = nullptr;
                for (auto& slot : slots_)
                    if (slot.state == State::FREE) { free_slot = &slot; break; }
                if (!free_slot) break;
                submit(*free_slot);
            }
            if (stop_) break;

            bool reading = false;
            for (const auto& slot : slots_)
                if (slot.state == State::READING) { reading = true; break; }

            if (!reading)
            {
                if (next_offset_ == file_size_) break;
                // Every slot is queued or held by the parser; wait for release.
                free_cv_.wait(lock, [this] { return stop_ || any_free(); });
                continue;
            }

            std::array<const struct aiocb*, AIO_DEPTH> list{};
            size_t count = 0;
            for (const auto& slot : slots_)
                if (slot.state == State::READING) list[count++] = &slot.cb;

            lock.unlock();
            const int rc = ::aio_suspend(list.data(), static_cast<int>(count), nullptr);
            const int error = errno;
            lock.lock();
            if (rc != 0 && error != EINTR)
                throw std::system_error(error, std::generic_category(), "aio_suspend");
            harvest();
        }
    }

    // No slot may be READING when this returns, so closing the fd afterwards
    // cannot leave an in-flight request pointing at a reused descriptor.
    void drain() noexcept
    {
        for (auto& slot : slots_)
        {
            if (slot.state != State::READING) continue;
            const struct aiocb* list[1] = { &slot.cb };
            while (::aio_error(&slot.cb) == EINPROGRESS)
                if (::aio_suspend(list, 1, nullptr) != 0 && errno != EINTR) break;
            ::aio_return(&slot.cb);
            slot.state = State::FREE;
        }
    }

    void run() noexcept
    {
        try
        {
            read_loop();
        }
        catch (...)
        {
            {
                std::lock_guard lock(mutex_);
                error_ = std::current_exception();
                stop_ = true;
            }
            ready_cv_.notify_all();
            free_cv_.notify_all();
        }

        drain();

        {
            std::lock_guard lock(mutex_);
            if (fd_ >= 0)
            {
                ::close(fd_);
                fd_ = -1;
            }
            finished_ = true;
        }
        ready_cv_.notify_all();
        free_cv_.notify_all();
    }
};

#endif
