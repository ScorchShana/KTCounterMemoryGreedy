#ifndef RUN_STATE_HEADER
#define RUN_STATE_HEADER

#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <utility>

// All progress predicates and notifications use this mutex. In particular a
// notification cannot slip between observing a generation and going to sleep.
class RunState
{
    std::atomic<bool> failed_{false};
    std::exception_ptr error_;
    const char* stage_ = nullptr;
public:
    std::mutex mutex;
    std::condition_variable changed;
    uint64_t generation = 0;
    static inline RunState* active = nullptr; // Set before starting any workers.

    bool failed() const noexcept { return failed_.load(std::memory_order_acquire); }
    void fail(std::exception_ptr error, const char* stage) noexcept
    {
        {
            std::lock_guard lock(mutex);
            if (!failed_.load(std::memory_order_relaxed)) {
                error_ = error;
                stage_ = stage;
                failed_.store(true, std::memory_order_release);
            }
            ++generation;
        }
        changed.notify_all();
    }
    void check() const
    {
        if (failed()) std::rethrow_exception(error_);
    }
    static void fail_on_unwind() noexcept
    {
        if (active && std::uncaught_exceptions() && !active->failed())
            active->fail(std::make_exception_ptr(std::runtime_error("main thread aborted")), "main");
    }
    static void check_active() { if (active) active->check(); }
    const char* stage() const noexcept { return stage_; } // Read after failure/join.
    void notify()
    {
        { std::lock_guard lock(mutex); ++generation; }
        changed.notify_all();
    }
    uint64_t progress()
    {
        std::lock_guard lock(mutex);
        return generation;
    }
    void wait(uint64_t previous)
    {
        std::unique_lock lock(mutex);
        changed.wait(lock, [&] { return generation != previous || failed(); });
        lock.unlock();
        check();
    }
};

template<class Work>
void run_worker(const char* stage, Work&& work) noexcept
{
    try { work(); }
    catch (...) {
        if (!RunState::active) std::terminate();
        RunState::active->fail(std::current_exception(), stage);
    }
}

class PhaseBarrier
{
    RunState local_state_;
    RunState* state_;
    const size_t participants_;
    size_t arrived_ = 0;
    uint64_t round_ = 0;
public:
    explicit PhaseBarrier(size_t participants, RunState* state = RunState::active)
        : state_(state ? state : &local_state_), participants_(participants) {}
    void arrive_and_wait()
    {
        std::unique_lock lock(state_->mutex);
        const auto round = round_;
        if (++arrived_ == participants_) {
            arrived_ = 0;
            ++round_;
            ++state_->generation;
            state_->changed.notify_all();
        } else {
            state_->changed.wait(lock, [&] { return round_ != round || state_->failed(); });
        }
        lock.unlock();
        state_->check();
    }
};

class ActiveRunScope
{
    RunState* previous_;
public:
    explicit ActiveRunScope(RunState& state) : previous_(RunState::active) { RunState::active = &state; }
    ~ActiveRunScope() { RunState::active = previous_; }
};

#endif
