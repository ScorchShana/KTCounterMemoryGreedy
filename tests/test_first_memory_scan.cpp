#include "../src/ConcurrentMemoryPool.h"
#include <array>
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>

struct ConcurrentMemoryPoolTestAccess
{
    static std::mutex& admission(ConcurrentMemoryPool& pool) { return pool.arenas_[0].admission; }
};
static void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }

int main()
{
    try {
        k_length = 33;
        ConcurrentMemoryPool pool(128ULL << 20);
        // The same fixed reservations serve input activity and final extraction.
        auto* active = static_cast<unsigned char*>(pool.allocate_before_init_arenas(4 * FIRST_SEGMENT_SIZE));
        pool.init_arenas();
        const size_t capacity = ConcurrentMemoryPool::first_segment_record_capacity();
        constexpr size_t segments = 16;
        for (size_t s = 0; s < segments; ++s) {
            for (size_t i = 0; i < capacity; ++i) {
                const uint64_t id = s * capacity + i;
                std::memcpy(active + i * 9, &id, 8);
                active[i * 9 + 8] = 0x6d;
            }
            require(pool.cache_first_segment(s % pool.arena_count(), active), "scan fixture exceeded capacity");
        }
        pool.freeze_first();
        // A lock miss must return Busy, never Done or a lost cursor. Other Arenas
        // are already empty in this preliminary check, so use a separate pool.
        ConcurrentMemoryPool empty(16ULL << 20);
        empty.init_arenas();
        std::unique_lock empty_held(ConcurrentMemoryPoolTestAccess::admission(empty));
        auto attempted = std::async(std::launch::async, [&] { return empty.extract_first_segment(0, active); });
        const bool immediate = attempted.wait_for(std::chrono::milliseconds(200)) == std::future_status::ready;
        empty_held.unlock();
        require(attempted.get() == ConcurrentMemoryPool::FirstExtractStatus::Busy && immediate, "busy extraction blocked or reported Done");
        std::array<std::atomic<size_t>, segments> seen{};
        std::array<std::future<void>, 4> workers;
        for (size_t worker = 0; worker < workers.size(); ++worker) {
            workers[worker] = std::async(std::launch::async, [&, worker] {
                auto* buffer = active + worker * FIRST_SEGMENT_SIZE;
                for (;;) {
                    const auto status = pool.extract_first_segment(worker % pool.arena_count(), buffer);
                    if (status == ConcurrentMemoryPool::FirstExtractStatus::Done) return;
                    if (status == ConcurrentMemoryPool::FirstExtractStatus::Busy) { std::this_thread::yield(); continue; }
                    uint64_t first;
                    std::memcpy(&first, buffer, 8);
                    const size_t segment = first / capacity;
                    require(segment < segments && first % capacity == 0, "scan returned an incomplete segment");
                    require(seen[segment].fetch_add(1) == 0, "segment extracted more than once");
                    for (size_t i = 0; i < capacity; ++i) {
                        uint64_t id;
                        std::memcpy(&id, buffer + i * 9, 8);
                        require(id == first + i && buffer[i * 9 + 8] == 0x6d, "scan changed a record");
                    }
                }
            });
        }
        for (auto& worker : workers) worker.get();
        for (const auto& count : seen) require(count == 1, "scan omitted a segment");
        require(pool.first_empty(), "whole-segment extraction retained source storage");
        std::cout << "FirstMemoryScan tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
