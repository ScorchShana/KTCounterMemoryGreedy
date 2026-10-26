// Compile against either baseline or current headers. Production timings omit
// per-operation clocks; the separate diagnostic target samples lock contention.
#include "ConcurrentMemoryPool.h"
#include <array>
#include <barrier>
#include <chrono>
#include <iostream>
#include <string>

struct ConcurrentMemoryPoolTestAccess
{
    static void single(ConcurrentMemoryPool& pool) {
        pool.num_arenas_ = 1;
        pool.total_cpus_ = pool.cpus_per_node_[0] = 1;
    }
    static Arena& arena(ConcurrentMemoryPool& pool) { return pool.arenas_[0]; }
};
struct alignas(64) Sample {
    double seconds = 0;
    uintptr_t checksum = 0;
#ifdef ARENA_BENCH_DIAGNOSTICS
    size_t lock_probes = 0, busy_probes = 0;
    std::vector<uint64_t> latency_ns, lock_wait_ns;
#endif
};

int main(int argc, char** argv)
{
    if (argc < 2 || argc > 3) return 1;
    k_length = 33;
    const size_t threads = std::stoul(argv[1]);
    const std::string mode = argc == 3 ? argv[2] : "bump";
    if (mode != "bump" && mode != "refill" && mode != "first" && mode != "mixed") return 1;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask)) return 1;
    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    if (!threads || threads > cpus.size()) return 1;
    const size_t per_thread = mode == "first" ? 4096 : 262144;
    ConcurrentMemoryPool pool(mode == "bump" ? (threads * per_thread + 1024) * BLOCK_SIZE : 256ULL << 20);
    ConcurrentMemoryPoolTestAccess::single(pool);
    pool.init_arenas();
    pool.set_spill_service(true);
    std::barrier start(static_cast<std::ptrdiff_t>(threads));
    std::vector<Sample> samples(threads);
    std::atomic<bool> finished{false};
    std::atomic<size_t> spilled{0};
    std::thread consumer;
    if (mode == "first" || mode == "mixed") consumer = std::thread([&] {
        for (;;) {
            pool.request_first_drain();
            ConcurrentMemoryPool::FirstSegmentView view;
            if (pool.try_claim_first(0, ConcurrentMemoryPool::FirstConsumer::Spill, view)
                == ConcurrentMemoryPool::FirstClaimStatus::Data) {
                if (view.record_count != 1 || view.records[0] != 0x6d) std::terminate();
                ++spilled;
                if (!pool.complete_first(view)) std::terminate();
            } else if (finished.load(std::memory_order_acquire) && pool.first_empty()) return;
            else std::this_thread::yield();
        }
    });
    std::vector<std::thread> workers;
    for (size_t i = 0; i < threads; ++i) workers.emplace_back([&, i] {
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(cpus[i], &affinity);
        if (sched_setaffinity(0, sizeof(affinity), &affinity)) std::terminate();
        constexpr size_t burst = TLS_CAPACITY + BATCH_SIZE;
        std::array<void*, burst> pages{};
        const unsigned char record[9]{0x6d};
#ifdef ARENA_BENCH_DIAGNOSTICS
        samples[i].latency_ns.reserve(per_thread / 256 + 1);
        samples[i].lock_wait_ns.reserve(per_thread / 256 + 1);
#endif
        start.arrive_and_wait();
        const auto begin = std::chrono::steady_clock::now();
        uintptr_t checksum = 0;
        for (size_t n = 0; n < per_thread; ++n) {
#ifdef ARENA_BENCH_DIAGNOSTICS
            std::chrono::steady_clock::time_point operation_begin;
            if (n % 256 == 0) {
                auto& mutex = ConcurrentMemoryPoolTestAccess::arena(pool).mutex;
                ++samples[i].lock_probes;
                uint64_t waited = 0;
                if (mutex.try_lock()) mutex.unlock();
                else {
                    ++samples[i].busy_probes;
                    const auto wait_begin = std::chrono::steady_clock::now();
                    mutex.lock();
                    waited = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - wait_begin).count();
                    mutex.unlock();
                }
                samples[i].lock_wait_ns.push_back(waited);
                operation_begin = std::chrono::steady_clock::now();
            }
#endif
            if (mode == "first" || (mode == "mixed" && n % 256 == 0)) {
                while (!pool.cache_first_record(0, record)) std::this_thread::yield();
                pool.seal_thread_first();
            }
            if (mode == "bump") {
                checksum ^= reinterpret_cast<uintptr_t>(pool.allocate_large_from_numa(BLOCK_SIZE, 0));
            } else if (mode != "first") {
                pages[n % burst] = pool.allocate();
                checksum ^= reinterpret_cast<uintptr_t>(pages[n % burst]);
                if (n % burst == burst - 1 || n + 1 == per_thread)
                    for (size_t page = 0; page <= n % burst; ++page) pool.deallocate(pages[page]);
            }
#ifdef ARENA_BENCH_DIAGNOSTICS
            if (n % 256 == 0) samples[i].latency_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - operation_begin).count());
#endif
        }
        samples[i].seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        samples[i].checksum = checksum;
    });
    for (auto& worker : workers) worker.join();
    finished.store(true, std::memory_order_release);
    if (consumer.joinable()) consumer.join();
    const size_t expected_spills = mode == "first" ? threads * per_thread
        : mode == "mixed" ? threads * ((per_thread + 255) / 256) : 0;
    if (spilled != expected_spills) return 2;
    double seconds = 0;
    uintptr_t checksum = 0;
    for (const auto& sample : samples) {
        seconds = std::max(seconds, sample.seconds);
        checksum ^= sample.checksum;
    }
    std::cout << seconds * 1e9 / (threads * per_thread) << ' ' << checksum << '\n';
#ifdef ARENA_BENCH_DIAGNOSTICS
    size_t probes = 0, busy = 0;
    uint64_t wait_total = 0;
    std::vector<uint64_t> latency, lock_wait;
    for (const auto& sample : samples) {
        probes += sample.lock_probes;
        busy += sample.busy_probes;
        latency.insert(latency.end(), sample.latency_ns.begin(), sample.latency_ns.end());
        lock_wait.insert(lock_wait.end(), sample.lock_wait_ns.begin(), sample.lock_wait_ns.end());
        for (uint64_t waited : sample.lock_wait_ns) wait_total += waited;
    }
    std::sort(latency.begin(), latency.end());
    std::sort(lock_wait.begin(), lock_wait.end());
    std::cout << "lock_probes " << probes << " busy_probes " << busy << " operation_p99_ns "
        << latency[(latency.size() * 99 - 1) / 100]
        << " probe_wait_total_ns " << wait_total << " probe_wait_p99_ns " << lock_wait[(lock_wait.size() * 99 - 1) / 100] << '\n';
#endif
}
