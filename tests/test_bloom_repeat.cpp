#include "../src/BloomFilter.h"

#include <array>
#include <barrier>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

template <uint32_t N>
void check_queries(ConcurrentMemoryPool& pool)
{
    ConcurrentBloomFilter<N> bloom(64, &pool);
    kmer<N> key{};
    key.data[0] = 0x123456789abcdef0ULL;
    const auto probe = bloom.prepare_insert(key);
    require(!bloom.contains_repeat(key), "empty filter contains a repeat");
    require(bloom.insert(key) == Occurrence::FIRST, "first insertion not FIRST");
    require(!bloom.contains_repeat_prepared(probe), "single occurrence promoted by query");
    require(bloom.insert(key) == Occurrence::REPEAT, "second insertion not REPEAT");
    require(bloom.insert(key) == Occurrence::REPEAT, "third insertion not REPEAT");
    std::array<uint64_t, 128> before{};
    for (size_t i = 0; i < before.size(); ++i) before[i] = bloom.get_bins()[i].load();
    for (int i = 0; i < 100; ++i) {
        bloom.prefetch_contains(probe);
        require(bloom.contains_repeat(key) && bloom.contains_repeat_prepared(probe), "repeat query lost registered bits");
    }
    for (size_t i = 0; i < before.size(); ++i)
        require(before[i] == bloom.get_bins()[i].load(), "read-only query changed filter");

    ConcurrentBloomFilter<N> false_positive(64, &pool);
    false_positive.get_bins()[2 * probe.block_idx].store(probe.insert_num);
    require(false_positive.contains_repeat_prepared(probe), "repeat query inspected the wrong Bloom layer");
    require(false_positive.insert(key) == Occurrence::REPEAT, "controlled false positive routed as FIRST");
}

void check_competing_firsts(ConcurrentMemoryPool& pool)
{
    for (int round = 0; round < 32; ++round) {
        ConcurrentBloomFilter<1> bloom(64, &pool);
        kmer<1> key{};
        key.data[0] = 0x1234567800000000ULL + uint64_t(round) * 4;
        constexpr size_t threads = 8;
        std::barrier start(threads);
        std::atomic<unsigned> firsts{0};
        std::vector<std::thread> workers;
        for (size_t i = 0; i < threads; ++i) workers.emplace_back([&] {
            start.arrive_and_wait();
            if (bloom.insert(key) == Occurrence::FIRST) firsts.fetch_add(1);
        });
        for (auto& worker : workers) worker.join();
        require(firsts == 1, "concurrent insertions registered multiple FIRSTs");
        require(bloom.contains_repeat(key), "concurrent repetition was not registered");
    }
}
}

int main()
{
    try {
        ConcurrentMemoryPool pool(8ULL << 20);
        pool.init_arenas();
        check_queries<1>(pool);
        check_queries<2>(pool);
        check_queries<4>(pool);
        check_competing_firsts(pool);
        std::cout << "BloomRepeat tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
