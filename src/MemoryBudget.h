#ifndef MEMORY_BUDGET_HEADER
#define MEMORY_BUDGET_HEADER

#include "RingMemoryPool.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>

struct RingPoolBudget
{
    size_t run_data, replay_data, final_data;
    size_t run_bytes, replay_bytes, final_bytes, peak_bytes;
};

inline RingPoolBudget ring_pool_budget() noexcept
{
    constexpr size_t reader = READER_PARSER_RING_MEMORY_POOL_CAPACITY * READER_PARSER_RING_MEMORY_POOL_BLOCK_SIZE;
    constexpr size_t parser = PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY * PARSER_CLASSIFIER_RING_MEMORY_POOL_BLOCK_SIZE;
    constexpr size_t output = EXPORT_RING_MEMORY_POOL_CAPACITY * EXPORT_RING_MEMORY_POOL_BLOCK_SIZE;
    constexpr size_t drain = FINAL_DRAIN_RING_POOL_CAPACITY * FINAL_DRAIN_RING_POOL_BLOCK_SIZE;
    constexpr size_t low = LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY * LOW_FREQUENCY_RING_MEMORY_POOL_BLOCK_SIZE;
    constexpr size_t common_metadata = sizeof(RingMemoryPool<PARSER_CLASSIFIER_RING_MEMORY_POOL_CAPACITY>)
        + sizeof(RingMemoryPool<EXPORT_RING_MEMORY_POOL_CAPACITY>)
        + sizeof(RingMemoryPool<FINAL_DRAIN_RING_POOL_CAPACITY>);
    RingPoolBudget result{reader + parser + output + drain, low + parser + output + drain,
        parser + output + drain, 0, 0, 0, 0};
    result.run_bytes = result.run_data + common_metadata + sizeof(RingMemoryPool<READER_PARSER_RING_MEMORY_POOL_CAPACITY>);
    result.replay_bytes = result.replay_data + common_metadata + sizeof(SPMCRingMemoryPool<LOW_FREQUENCY_RING_MEMORY_POOL_CAPACITY>);
    result.final_bytes = result.final_data + common_metadata;
    result.peak_bytes = std::max({result.run_bytes, result.replay_bytes, result.final_bytes});
    return result;
}

inline size_t main_pool_budget(uint64_t gib)
{
    constexpr uint64_t gib_bytes = 1ULL << 30;
    constexpr size_t segment_bytes = 2ULL << 20;
    if (gib > std::numeric_limits<size_t>::max() / gib_bytes)
        throw std::invalid_argument("memory limit GiB conversion overflow");
    const size_t total = static_cast<size_t>(gib) * gib_bytes;
    // The pending transport is freed at normal-drain before final-low starts.
    const size_t rings = ring_pool_budget().peak_bytes + EXPORT_WRITER_BUFFER_BYTES * EXPORT_WRITER_BUFFER_COUNT;
    if (total <= rings || total - rings < segment_bytes)
        throw std::invalid_argument("memory limit cannot hold RingPools and main pool");
    return (total - rings) & ~(segment_bytes - 1);
}

inline size_t bloom_bytes(uint64_t capacity, size_t filters = 1)
{
    constexpr size_t unit = 2 * sizeof(std::atomic<uint64_t>);
    if (capacity > std::numeric_limits<size_t>::max() / unit)
        throw std::overflow_error("Bloom capacity overflow");
    const size_t bytes = static_cast<size_t>(capacity) * unit;
    if (bytes && filters > std::numeric_limits<size_t>::max() / bytes)
        throw std::overflow_error("Bloom total bytes overflow");
    return bytes * filters;
}

#endif
