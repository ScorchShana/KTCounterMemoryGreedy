#include "../src/RunState.h"
#include "../src/SpinBackoff.h"
#include <array>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class Backoff>
static bool check_backoff(const std::vector<int>& expected_checks)
{
    Backoff original, periodic;
    auto next_check = expected_checks.begin();
    for (int i = 1; i <= 192; ++i) {
        original.backoff();
        const bool expected = next_check != expected_checks.end() && i == *next_check;
        if (periodic.backoff_should_check() != expected) {
            std::cerr << "Resource check mismatch at backoff " << i
                      << ": expected " << expected << '\n';
            return false;
        }
        if (expected) ++next_check;
        if (periodic.get_request_sleep_time() != original.get_request_sleep_time()) return false;
    }
    original.reset();
    periodic.reset();
    for (int i = 1; i <= 192; ++i) {
        // Ordinary calls and successful-work adjustments must not starve checks.
        original.backoff();
        periodic.backoff();
        switch (i % 3) {
        case 0: original.reset(); periodic.reset(); break;
        case 1: original.decay(); periodic.decay(); break;
        case 2: original.double_decay(); periodic.double_decay(); break;
        }
        original.backoff();
        if (!periodic.backoff_should_check()) return false;
        if (periodic.get_request_sleep_time() != original.get_request_sleep_time()) return false;
    }
    return true;
}

static bool check_backoff_adjustments()
{
    for (int adjustment = 0; adjustment < 3; ++adjustment) {
        SpinBackoff<> original, periodic;
        for (int i = 0; i < 80; ++i) {
            original.backoff();
            periodic.backoff();
        }
        // Carry seven unchecked sleep calls through each successful-work adjustment.
        for (int i = 0; i < 7; ++i) {
            original.backoff();
            if (periodic.backoff_should_check()) return false;
        }
        switch (adjustment) {
        case 0:
            original.reset();
            periodic.reset();
            // Ordinary calls advance the phase without consuming pending checks.
            for (int i = 0; i < 32; ++i) {
                original.backoff();
                periodic.backoff();
            }
            break;
        case 1: original.decay(); periodic.decay(); break;
        case 2: original.double_decay(); periodic.double_decay(); break;
        }
        // Reset/decay reach interval 16; double_decay lowers it below the carry.
        const int next_check = adjustment == 2 ? 1 : 9;
        for (int i = 1; i <= next_check; ++i) {
            original.backoff();
            if (periodic.backoff_should_check() != (i == next_check)) return false;
        }
        for (int i = 0; i < 192; ++i) {
            original.backoff();
            periodic.backoff();
            if (periodic.get_request_sleep_time() != original.get_request_sleep_time()) return false;
        }
    }
    return true;
}

static bool check_phase_waits()
{
    RunState state;
    const auto generation = state.progress();
    auto waiter = std::async(std::launch::async, [&] { state.wait(generation); });
    state.notify();
    waiter.get();
    PhaseBarrier barrier(8, &state);
    std::array<int, 8> published{};
    std::vector<std::future<bool>> workers;
    for (size_t i = 0; i < published.size(); ++i) {
        workers.push_back(std::async(std::launch::async, [&, i] {
            bool valid = true;
            for (int round = 1; round <= 32; ++round) {
                published[i] = round;
                barrier.arrive_and_wait();
                for (auto value : published) valid &= value == round;
                barrier.arrive_and_wait();
            }
            return valid;
        }));
    }
    for (auto& worker : workers) if (!worker.get()) return false;
    return true;
}

int main()
{
    if (!check_backoff<SpinBackoff<>>({
            1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64,
            66, 68, 70, 72, 74, 76, 78, 80,
            88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192})
        || !check_backoff<SpinBackoff<32, 128, 144>>({
            1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64,
            80, 96, 112, 128, 130, 132, 134, 136, 138, 140, 142, 144,
            152, 160, 168, 176, 184, 192})
        || !check_backoff<SpinBackoff<16, 128, 136, 8>>({
            1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32, 48, 64,
            80, 96, 112, 128, 130, 132, 134, 136,
            144, 152, 160, 168, 176, 184, 192})
        || !check_backoff<SpinBackoff<64, 35, 37>>({
            1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 32,
            36, 44, 52, 60, 68, 76, 84, 92, 100, 108, 116, 124, 132, 140,
            148, 156, 164, 172, 180, 188})
        || !check_backoff_adjustments()
        || !check_phase_waits()) return 1;
    RunState state;
    PhaseBarrier barrier(2, &state);
    std::promise<void> started;
    auto waiting = std::async(std::launch::async, [&] {
        started.set_value();
        try { barrier.arrive_and_wait(); } catch (const std::exception&) { return true; }
        return false;
    });
    started.get_future().wait();
    state.fail(std::make_exception_ptr(std::runtime_error("first failure")), "injected");
    state.fail(std::make_exception_ptr(std::runtime_error("second failure")), "later");
    if (!waiting.get()) return 1;
    try { state.check(); }
    catch (const std::runtime_error& error) {
        if (std::string(error.what()) != "first failure") return 1;
        std::cout << "RunState failure/wakeup tests passed\n";
        return 0;
    }
    return 1;
}
