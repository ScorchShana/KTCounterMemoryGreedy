// Compile unchanged against before/after src headers; timing excludes startup,
// thread creation and payload access. This isolates contended continuous bump.
#include "ConcurrentMemoryPool.h"
#include <barrier>
#include <chrono>
#include <iostream>

struct ConcurrentMemoryPoolTestAccess
{
    static void single(ConcurrentMemoryPool& pool) {
        pool.num_arenas_ = 1;
        pool.total_cpus_ = pool.cpus_per_node_[0] = 1;
    }
};
struct alignas(64) Sample { double seconds = 0; uintptr_t checksum = 0; };

int main(int argc, char** argv)
{
    if (argc != 2) return 1;
    const size_t threads = std::stoul(argv[1]);
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask)) return 1;
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    if (!threads || threads > cpus.size()) return 1;
    constexpr size_t per_thread = 262144;
    ConcurrentMemoryPool pool((threads * per_thread + 1024) * BLOCK_SIZE);
    ConcurrentMemoryPoolTestAccess::single(pool);
    pool.init_arenas();
    std::barrier start(static_cast<std::ptrdiff_t>(threads));
    std::vector<Sample> samples(threads);
    std::vector<std::thread> workers;
    for (size_t i = 0; i < threads; ++i) workers.emplace_back([&, i] {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpus[i], &affinity);
        if (sched_setaffinity(0, sizeof(affinity), &affinity)) std::terminate();
        start.arrive_and_wait();
        const auto begin = std::chrono::steady_clock::now();
        uintptr_t checksum = 0;
        for (size_t n = 0; n < per_thread; ++n)
            checksum ^= reinterpret_cast<uintptr_t>(pool.allocate_large_from_numa(BLOCK_SIZE, 0));
        samples[i].seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        samples[i].checksum = checksum;
    });
    for (auto& worker : workers) worker.join();
    double seconds = 0;
    uintptr_t checksum = 0;
    for (const auto& sample : samples) {
        seconds = std::max(seconds, sample.seconds);
        checksum ^= sample.checksum;
    }
    std::cout << seconds * 1e9 / (threads * per_thread) << ' ' << checksum << '\n';
}
