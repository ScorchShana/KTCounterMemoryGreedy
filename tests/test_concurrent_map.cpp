#define CONCURRENT_MAP_TESTING

#include "../src/ConcurrentMap.h"
#include "../src/ConcurrentMemoryPool.h"
#include "../src/definition.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <thread>
#include <vector>
#include <bit>

namespace
{

    constexpr uint32_t kKmerWords = 2;
    using Kmer = kmer<kKmerWords>;

    constexpr size_t kGlobalHotKeyCount = 1;
    constexpr size_t kLocalHotKeyCount = 1;

    struct Config
    {
        uint64_t total_inserts = 0;
        uint64_t inserts_per_thread = 0;
        uint64_t num_threads = 0;
        double duplicate_prob = 0.0;
    };

    void print_usage(const char *argv0)
    {
        std::cerr << "Usage: " << argv0
                  << " <inserts_per_thread> <num_threads> [duplicate_prob]\n";
    }

    bool parse_u64(const char *text, uint64_t &value)
    {
        if (!text || *text == '\0')
        {
            return false;
        }
        char *end = nullptr;
        errno = 0;
        unsigned long long parsed = std::strtoull(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0')
        {
            return false;
        }
        value = static_cast<uint64_t>(parsed);
        return true;
    }

    bool parse_double(const char *text, double &value)
    {
        if (!text || *text == '\0')
        {
            return false;
        }
        char *end = nullptr;
        errno = 0;
        double parsed = std::strtod(text, &end);
        if (errno != 0 || end == text || *end != '\0')
        {
            return false;
        }
        value = parsed;
        return true;
    }

    Kmer make_random_kmer(std::mt19937_64 &rng, std::uniform_int_distribution<uint64_t> &dist)
    {
        Kmer key{};
        for (uint32_t i = 0; i < kKmerWords; ++i)
        {
            key.data[i] = dist(rng);
        }
        return key;
    }

} // namespace

int main(int argc, char **argv)
{
    Config cfg{};
    if ((argc != 3 && argc != 4) ||
        !parse_u64(argv[1], cfg.inserts_per_thread) ||
        !parse_u64(argv[2], cfg.num_threads) ||
        (argc == 4 && !parse_double(argv[3], cfg.duplicate_prob)))
    {
        print_usage(argv[0]);
        return 1;
    }

    if (cfg.inserts_per_thread == 0 || cfg.num_threads == 0)
    {
        std::cerr << "inserts_per_thread and num_threads must be > 0\n";
        return 1;
    }

    if (cfg.duplicate_prob < 0.0 || cfg.duplicate_prob > 1.0)
    {
        std::cerr << "duplicate_prob must be in [0.0, 1.0]\n";
        return 1;
    }

    if (cfg.inserts_per_thread > std::numeric_limits<uint64_t>::max() / cfg.num_threads)
    {
        std::cerr << "inserts_per_thread * num_threads overflow\n";
        return 1;
    }

    cfg.total_inserts = cfg.inserts_per_thread * cfg.num_threads;

    if (cfg.total_inserts > std::numeric_limits<uint32_t>::max())
    {
        std::cerr << "total_inserts must be <= 2^32-1 to avoid counter overflow\n";
        return 1;
    }

    const uint64_t num_threads = cfg.num_threads;

    const uint64_t capacity = std::bit_ceil((uint64_t)(cfg.total_inserts * (1 - cfg.duplicate_prob)));
    const uint64_t bucket_bytes = capacity * sizeof(bucket<kKmerWords>);
    std::unique_ptr<char[]> bucket_memory(new char[bucket_bytes]);

    const uint64_t max_unique = cfg.total_inserts * (1 - cfg.duplicate_prob);
    const uint64_t slots_per_block = KMER_BLOCK_SIZE / sizeof(concurrent_node<kKmerWords>);
    const uint64_t blocks_needed = (max_unique + slots_per_block - 1) / slots_per_block;
    const uint64_t extra_blocks = 64;
    const uint64_t total_blocks = blocks_needed + extra_blocks;
    const uint64_t pool_bytes = total_blocks * BLOCK_SIZE;

    ConcurrentMemoryPool pool(4ULL * 1024 * 1024 * 1024);
    pool.init_arenas();
    ConcurrentMap<kKmerWords> map(capacity, bucket_memory.get(), &pool);

    std::mt19937_64 global_rng(0xBADC0FFEEULL);
    std::uniform_int_distribution<uint64_t> global_dist(0, std::numeric_limits<uint64_t>::max());
    std::vector<Kmer> global_hot_keys;
    global_hot_keys.reserve(kGlobalHotKeyCount);
    for (size_t i = 0; i < kGlobalHotKeyCount; ++i)
    {
        global_hot_keys.push_back(make_random_kmer(global_rng, global_dist));
    }

    std::atomic<long long> max_insert_ns{0};
    std::barrier sync_point(static_cast<std::ptrdiff_t>(num_threads));
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    for (uint64_t thread_index = 0; thread_index < num_threads; ++thread_index)
    {
        threads.emplace_back([&, thread_index]()
                             {
            std::mt19937_64 rng(0x12345678ULL + thread_index);
            std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());
            std::uniform_real_distribution<double> prob_dist(0.0, 1.0);

            std::vector<Kmer> local_hot_keys;
            local_hot_keys.reserve(kLocalHotKeyCount);
            for (size_t i = 0; i < kLocalHotKeyCount; ++i) {
                local_hot_keys.push_back(make_random_kmer(rng, dist));
            }

            std::vector<Kmer> insert_keys;
            insert_keys.reserve(cfg.inserts_per_thread);

            std::uniform_int_distribution<size_t> global_index(0, global_hot_keys.size() - 1);
            std::uniform_int_distribution<size_t> local_index(0, local_hot_keys.size() - 1);

            for (uint64_t i = 0; i < cfg.inserts_per_thread; ++i) {
                double roll = prob_dist(rng);
                if (roll < cfg.duplicate_prob) {
                    bool use_global = prob_dist(rng) < 0.5;
                    if (use_global) {
                        insert_keys.push_back(global_hot_keys[global_index(rng)]);
                    } else {
                        insert_keys.push_back(local_hot_keys[local_index(rng)]);
                    }
                } else {
                    insert_keys.push_back(make_random_kmer(rng, dist));
                }
            }

            sync_point.arrive_and_wait();

            const auto start_tp = std::chrono::steady_clock::now();
            for (const auto &key : insert_keys) {
                map.increment(key);
            }
            const auto end_tp = std::chrono::steady_clock::now();
            const auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_tp - start_tp).count();

            long long observed = max_insert_ns.load(std::memory_order_relaxed);
            while (elapsed_ns > observed &&
                   !max_insert_ns.compare_exchange_weak(observed, elapsed_ns,
                                                       std::memory_order_release,
                                                       std::memory_order_relaxed)) {
            }
        });
    }

    for (auto &thread : threads)
    {
        thread.join();
    }

    const long long max_ns = max_insert_ns.load(std::memory_order_acquire);
    if (max_ns <= 0)
    {
        std::cerr << "timing was not recorded\n";
        return 1;
    }
    const double elapsed_seconds = static_cast<double>(max_ns) * 1e-9;
    const double ops_per_sec = elapsed_seconds > 0.0
                                   ? static_cast<double>(cfg.total_inserts) / elapsed_seconds
                                   : 0.0;

    uint64_t observed_total = 0;
    map.debug_visit([&](const Kmer &, uint32_t value)
                    { observed_total += value; });

    if (observed_total != cfg.total_inserts)
    {
        std::cerr << "count mismatch: expected " << cfg.total_inserts
                  << ", got " << observed_total << "\n";
        return 1;
    }

    std::cout << "insert_seconds=" << elapsed_seconds << "\n"
              << ops_per_sec / 1000000.0 << "Mopt/s\n";

    return 0;
}
