#include "../src/ReplacingHashMap.h"
#include "../src/ConcurrentMemoryPool.h"

#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <sys/wait.h>
#include <unistd.h>

namespace
{
void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

template <uint32_t N>
kmer<N> make_key(uint64_t value)
{
    kmer<N> key;
    for (uint32_t i = 0; i < N; ++i)
        key.data[i] = (value + i) * 0x9e3779b97f4a7c15ULL;
    return key;
}

template <uint32_t N>
void test_budget_and_alignment(ConcurrentMemoryPool& pool,
    size_t budget, size_t expected_capacity)
{
    using Map = ReplacingHashMap<N>;
    static_assert(!std::is_copy_constructible_v<Map>);
    static_assert(!std::is_move_constructible_v<Map>);
    Map* map = Map::create(&pool, budget);
    auto* next = static_cast<char*>(pool.allocate_large(BLOCK_SIZE));
    const size_t consumed = next - reinterpret_cast<char*>(map);
    require(consumed <= budget, "actual pool allocation exceeded map budget");
    require(consumed % BLOCK_SIZE == 0, "pool allocation is not page aligned");
    require(map->capacity() == expected_capacity, "wrong budget-derived capacity");
    require(reinterpret_cast<uintptr_t>(map->controls_) % 64 == 0,
        "controls are not cache-line aligned");
    require(reinterpret_cast<uintptr_t>(map->keys_) % alignof(kmer<N>) == 0,
        "keys are not aligned");
    require(map->controls_ >= reinterpret_cast<uint8_t*>(map) + sizeof(Map),
        "controls overlap the map object");
    require(reinterpret_cast<uint8_t*>(map->keys_) == map->controls_ + map->capacity(),
        "keys do not follow controls in the single allocation");
    require(reinterpret_cast<char*>(map->keys_ + map->capacity())
            + map->capacity() / 64 <= next,
        "keys and bucket state extend beyond the allocation");
    require(map->empty() && map->size() == 0 && map->load_factor() == 0,
        "new map is not empty");
    size_t visited = 0;
    map->for_each([&](const auto&) { ++visited; });
    require(visited == 0, "new map exported uninitialized keys");
}

size_t bucket_for(const kmer<1>& key, size_t buckets)
{
    return (uint64_t(uint32_t(rapidhash(&key, sizeof(key)))) * buckets) >> 32;
}

void test_replacement_and_reuse(ConcurrentMemoryPool& pool)
{
    using Map = ReplacingHashMap<1>;
    Map* map = Map::create(&pool, 4096); // Six buckets, not a power of two.
    const size_t buckets = map->capacity() / 64;
    require(!std::has_single_bit(buckets), "fixture must use non-power-of-two buckets");
    std::vector<kmer<1>> keys;
    std::set<uint8_t> fingerprints;
    for (uint64_t value = 1; keys.size() < 67; ++value)
    {
        const auto key = make_key<1>(value);
        if (bucket_for(key, buckets) != buckets - 1) continue;
        const auto f = uint8_t(rapidhash(&key, sizeof(key)) >> 56);
        fingerprints.insert(f == 0 ? uint8_t{255} : f);
        keys.push_back(key);
    }
    require(fingerprints.size() > 16,
        "same-bucket keys should retain varied fingerprints");

    kmer<1> evicted = make_key<1>(0);
    const auto unchanged = evicted;
    for (size_t i = 0; i < 64; ++i)
    {
        require(!map->insert(keys[i], evicted), "insertion evicted before bucket was full");
        require(evicted == unchanged, "non-replacement insertion changed output");
        require(map->contains(keys[i]), "inserted key not found");
    }
    require(map->size() == 64 && !map->full(), "bucket full confused with whole map full");
    require(!map->contains(keys[66]) && !map->erase(keys[66]), "absent key was found");
    require(map->insert(keys[64], evicted) && evicted == keys[0],
        "full bucket did not replace its FIFO hand");
    require(!map->contains(keys[0]) && map->contains(keys[64]),
        "replacement did not update membership");
    require(map->size() == 64, "replacement changed map size");
    require(map->erase(keys[17]) && !map->erase(keys[17]), "erase membership is incorrect");
    require(map->size() == 63, "erase did not decrement size");
    require(!map->insert(keys[65], evicted) && map->size() == 64,
        "insertion did not reuse erased slot");

    std::set<uint64_t> expected;
    for (size_t i = 1; i <= 65; ++i)
        if (i != 17) expected.insert(keys[i].data[0]);
    std::set<uint64_t> actual;
    size_t visits = 0;
    map->for_each([&](const auto& key) { actual.insert(key.data[0]); ++visits; });
    require(actual == expected && visits == expected.size(),
        "iteration omitted or duplicated resident keys");

    map->clear();
    require(map->empty() && map->size() == 0, "clear did not empty map");
    for (const auto& key : keys) require(!map->contains(key), "clear retained a key");
    visits = 0;
    map->for_each([&](const auto&) { ++visits; });
    require(visits == 0, "cleared map exported keys");
    require(!map->insert(keys[0], evicted) && map->contains(keys[0]),
        "cleared map could not be reused");
}

void test_single_bucket(ConcurrentMemoryPool& pool)
{
    auto* map = ReplacingHashMap<4>::create(&pool, 4096);
    require(map->capacity() == 64, "single-bucket fixture has wrong capacity");
    kmer<4> evicted;
    for (uint64_t i = 1; i <= 64; ++i)
        require(!map->insert(make_key<4>(i), evicted), "single bucket filled early");
    require(map->full() && map->load_factor() == 1, "full map state is incorrect");
    std::set<uint64_t> actual;
    map->for_each([&](const auto& key) { actual.insert(key.data[0]); });
    require(actual.size() == 64, "single-bucket iteration omitted keys");
    for (uint64_t i = 1; i <= 64; ++i)
        require(map->erase(make_key<4>(i)), "single-bucket erase failed");
    require(map->empty(), "erasing all keys did not empty map");
}

void test_invalid_budget(ConcurrentMemoryPool& pool, size_t budget)
{
    int pipe_fds[2];
    require(pipe(pipe_fds) == 0, "pipe failed");
    const pid_t pid = fork();
    require(pid >= 0, "fork failed");
    if (pid == 0)
    {
        close(pipe_fds[0]);
        if (dup2(pipe_fds[1], STDERR_FILENO) == -1) _exit(2);
        close(pipe_fds[1]);
        ReplacingHashMap<1>::create(&pool, budget);
        _exit(0);
    }
    close(pipe_fds[1]);
    char buffer[1024];
    std::string error_text;
    for (;;)
    {
        const ssize_t count = read(pipe_fds[0], buffer, sizeof(buffer));
        if (count < 0 && errno == EINTR) continue;
        require(count >= 0, "reading child error failed");
        if (count == 0) break;
        error_text.append(buffer, size_t(count));
    }
    close(pipe_fds[0]);
    int status = 0;
    require(waitpid(pid, &status, 0) == pid, "waitpid failed");
    if (WIFSIGNALED(status))
        std::cerr << "invalid-budget child signal: " << WTERMSIG(status) << '\n';
    require(WIFEXITED(status) && WEXITSTATUS(status) == 255,
        "insufficient map budget did not exit with allocation failure");
    require(error_text.find("budget") != std::string::npos,
        "insufficient map budget did not explain the error");
}
}

int main()
{
    try
    {
        ConcurrentMemoryPool pool(16ULL * 1024 * 1024);
        pool.init_arenas();
        test_budget_and_alignment<1>(pool, 4096, 384);
        test_budget_and_alignment<2>(pool, 4096, 192);
        test_budget_and_alignment<3>(pool, 4096, 128);
        test_budget_and_alignment<4>(pool, 4096, 64);
        test_budget_and_alignment<1>(pool, 12287, 896);
        test_budget_and_alignment<2>(pool, 12287, 448);
        test_budget_and_alignment<3>(pool, 12287, 320);
        test_budget_and_alignment<4>(pool, 12287, 192);
        test_replacement_and_reuse(pool);
        test_single_bucket(pool);
        test_invalid_budget(pool, 0);
        test_invalid_budget(pool, 4095);
        std::cout << "ReplacingHashMap tests passed\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ReplacingHashMap test failed: " << error.what() << '\n';
        return 1;
    }
}
