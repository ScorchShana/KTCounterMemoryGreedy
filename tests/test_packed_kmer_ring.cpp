#include "../src/PackedKmerRing.h"
#include "../include/xxh3.h"

#include <array>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace {
void require(bool ok, const char* message)
{
    if (!ok) throw std::runtime_error(message);
}

template <uint32_t N>
kmer<N> key_for(uint32_t k, uint64_t seed)
{
    kmer<N> key{};
    for (uint32_t i = 0; i < k; ++i) {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        key.data[i / 32] |= (seed & 3) << (62 - 2 * (i % 32));
    }
    return key;
}

template <uint32_t N>
void check_codec(uint32_t k, size_t expected_bytes)
{
    PackedKmerCodec<N> codec(k);
    require(codec.packed_bytes() == expected_bytes, "wrong packed width");
    for (uint64_t seed : {1ULL, 17ULL, 0x123456789abcdefULL}) {
        const auto key = key_for<N>(k, seed);
        std::array<uint8_t, sizeof(kmer<N>) + 2> bytes;
        bytes.fill(0xa5);
        codec.pack(key, bytes.data() + 1);
        require(bytes.front() == 0xa5 && bytes[expected_bytes + 1] == 0xa5,
            "pack wrote outside record");
        kmer<N> restored;
        restored.data.fill(~uint64_t{0});
        codec.unpack(bytes.data() + 1, restored);
        require(restored == key, "round trip lost bits or retained unused words");
        require(XXH3_64bits(&restored, sizeof(restored)) == XXH3_64bits(&key, sizeof(key)),
            "round trip changed raw object hash");
        require(codec.root_prefix(bytes.data() + 1) == key.data[0] >> 56,
            "packed root prefix is wrong");
    }
    kmer<N> dirty;
    dirty.data.fill(~uint64_t{0});
    std::array<uint8_t, sizeof(kmer<N>)> bytes{};
    codec.pack(dirty, bytes.data());
    kmer<N> restored;
    codec.unpack(bytes.data(), restored);
    kmer<N> expected{};
    for (uint32_t i = 0; i < k; ++i)
        expected.data[i / 32] |= 3ULL << (62 - 2 * (i % 32));
    require(restored == expected, "unused tail bits were not masked");
}

void check_literal_encoding()
{
    PackedKmerCodec<2> codec(33);
    kmer<2> key{};
    key.data = {0x0123456789abcdefULL, 0xc000000000000000ULL};
    std::array<uint8_t, 9> packed{};
    codec.pack(key, packed.data());
    const std::array<uint8_t, 9> expected{0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01, 0xc0};
    require(packed == expected, "encoding differs from existing little-endian disk format");
}

void check_ring(ConcurrentMemoryPool& pool)
{
    using Ring = PackedKmerRing<2>;
    constexpr size_t budget = 12287;
    auto* ring = Ring::create(&pool, 33, budget);
    auto* next = static_cast<uint8_t*>(pool.allocate_large(BLOCK_SIZE));
    const size_t consumed = next - reinterpret_cast<uint8_t*>(ring);
    require(consumed == ring->allocated_bytes() && consumed <= budget,
        "pool-rounded ring allocation exceeded budget");
    const size_t expected_capacity = (8192 - align_up(sizeof(Ring), 64)) / 9;
    require(ring->capacity() == expected_capacity, "ring wasted budget or rounded capacity to power of two");
    require((ring->capacity() & (ring->capacity() - 1)) != 0, "fixture needs non-power-of-two capacity");
    std::array<uint8_t, 9> evicted;
    evicted.fill(0xa5);
    require(!ring->pop_oldest(evicted.data()), "empty ring returned a record");
    const size_t insert_count = 3 * ring->capacity() + 7;
    std::vector<kmer<2>> expected;
    for (size_t i = 0; i < insert_count; ++i) {
        const auto key = key_for<2>(33, i + 1);
        expected.push_back(key);
        const bool replaced = ring->push(key, evicted.data());
        require(replaced == (i >= ring->capacity()), "wrong replacement boundary");
        if (replaced) {
            kmer<2> old;
            ring->codec().unpack(evicted.data(), old);
            require(old == expected[i - ring->capacity()], "ring replacement is not FIFO");
        } else {
            for (auto byte : evicted) require(byte == 0xa5, "non-evicting push changed output");
        }
    }
    require(ring->size() == ring->capacity(), "full ring has wrong size");
    for (size_t i = insert_count - ring->capacity(); i < insert_count; ++i) {
        require(ring->pop_oldest(evicted.data()), "drain ended early");
        kmer<2> old;
        ring->codec().unpack(evicted.data(), old);
        require(old == expected[i], "drain lost FIFO order");
    }
    require(ring->size() == 0 && !ring->pop_oldest(evicted.data()), "drain duplicated records");
    const auto key = key_for<2>(33, 42);
    require(!ring->push(key, evicted.data()) && ring->pop_oldest(evicted.data()), "drained ring cannot be reused");
    kmer<2> restored;
    ring->codec().unpack(evicted.data(), restored);
    require(restored == key, "reuse returned wrong record");
}

void check_invalid_budget(ConcurrentMemoryPool& pool, size_t budget)
{
    const pid_t child = fork();
    require(child >= 0, "fork failed");
    if (child == 0) {
        PackedKmerRing<1>::create(&pool, 31, budget);
        _exit(0);
    }
    int status = 0;
    require(waitpid(child, &status, 0) == child, "waitpid failed");
    require(WIFEXITED(status) && WEXITSTATUS(status) == 255,
        "invalid budget did not fail before allocation");
}
}

int main()
{
    try {
        for (auto k : {1U, 4U, 21U, 31U, 32U}) check_codec<1>(k, (k + 3) / 4);
        for (auto k : {33U, 55U, 64U}) check_codec<2>(k, (k + 3) / 4);
        for (auto k : {65U, 81U, 96U, 97U, 127U, 128U}) check_codec<4>(k, (k + 3) / 4);
        check_literal_encoding();
        ConcurrentMemoryPool pool(32ULL << 20);
        pool.init_arenas();
        check_ring(pool);
        check_invalid_budget(pool, 0);
        check_invalid_budget(pool, 4095);
        std::cout << "PackedKmerRing tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
