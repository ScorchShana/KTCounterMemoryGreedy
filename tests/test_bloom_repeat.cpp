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
    k_length = N * 32 - 1;
    ConcurrentBloomFilter bloom(64, &pool);
    std::vector<unsigned char> key(kmer_bytes(), 0x12);
    const auto probe = bloom.prepare_insert(key.data());
    for (size_t i = 0; i < 64; ++i)
        require(bloom.get_bins()[i].load() == 0, "empty filter contains seen bits");
    require(bloom.insert(key.data()) == Occurrence::FIRST, "first insertion not FIRST");
    require(bloom.get_bins()[probe.block_idx].load() == probe.insert_num,
        "first insertion did not populate the single seen word");
    require(bloom.insert(key.data()) == Occurrence::REPEAT, "second insertion not REPEAT");
    require(bloom.insert(key.data()) == Occurrence::REPEAT, "third insertion not REPEAT");
    std::array<uint64_t, 64> before{};
    for (size_t i = 0; i < before.size(); ++i) before[i] = bloom.get_bins()[i].load();
    for (int i = 0; i < 100; ++i) {
        bloom.prefetch_insert(probe);
        require(bloom.insert_prepared(probe) == Occurrence::REPEAT, "repeat insertion lost seen bits");
    }
    for (size_t i = 0; i < before.size(); ++i)
        require(before[i] == bloom.get_bins()[i].load(), "repeat insertion changed seen bits");

    ConcurrentBloomFilter false_positive(64, &pool);
    false_positive.get_bins()[probe.block_idx].store(probe.insert_num);
    require(false_positive.insert(key.data()) == Occurrence::REPEAT, "controlled false positive routed as FIRST");
}

void check_competing_firsts(ConcurrentMemoryPool& pool)
{
    for (int round = 0; round < 32; ++round) {
        ConcurrentBloomFilter bloom(64, &pool);
        k_length = 31;
        std::vector<unsigned char> key(kmer_bytes(), 0x12);
        key[0] = round;
        constexpr size_t threads = 8;
        std::barrier start(threads);
        std::atomic<unsigned> firsts{0};
        std::vector<std::thread> workers;
        for (size_t i = 0; i < threads; ++i) workers.emplace_back([&] {
            start.arrive_and_wait();
            if (bloom.insert(key.data()) == Occurrence::FIRST) firsts.fetch_add(1);
        });
        for (auto& worker : workers) worker.join();
        require(firsts == 1, "concurrent insertions registered multiple FIRSTs");
        require(bloom.insert(key.data()) == Occurrence::REPEAT, "concurrent repetition was not registered");
    }
}
}

int main()
{
    try {
        ConcurrentMemoryPool pool(8ULL << 20);
        pool.init_arenas();
        bool rejected = false;
        try { ConcurrentBloomFilter huge(1ULL << 61, &pool); }
        catch (const std::overflow_error&) { rejected = true; }
        require(rejected, "overflowing Bloom capacity was not rejected before allocation");
        rejected = false;
        try { ConcurrentBloomFilter invalid(3, &pool); }
        catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "non-power-of-two Bloom capacity accepted");
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
