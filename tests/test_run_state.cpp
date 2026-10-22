#include "../src/RunState.h"
#include <future>
#include <iostream>
#include <stdexcept>

int main()
{
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
