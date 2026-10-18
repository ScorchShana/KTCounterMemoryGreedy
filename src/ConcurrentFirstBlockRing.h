#pragma once

#include "ConcurrentMemoryPool.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>

// One ring per NUMA node. Pool-owned storage; never delete or deallocate it.
// Producers exclusively own their current block. Slots contain full blocks only.
class ConcurrentFirstBlockRing
{
    friend struct ConcurrentFirstBlockRingTestAccess;

    static_assert(std::atomic<uint8_t*>::is_always_lock_free);
    static_assert(std::atomic<uint64_t>::is_always_lock_free);

    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> head_{0};
    alignas(CACHE_LINE_SIZE) std::atomic<uint64_t> drain_cursor_{0};
    alignas(CACHE_LINE_SIZE) const size_t slot_count_;
    const size_t allocated_bytes_;
    std::atomic<uint8_t*>* const slots_;
    uint8_t* const data_;

    ConcurrentFirstBlockRing(size_t slots, size_t bytes,
        std::atomic<uint8_t*>* slot_array, uint8_t* data) noexcept
        : slot_count_(slots), allocated_bytes_(bytes), slots_(slot_array), data_(data)
    {
        for (size_t i = 0; i < slot_count_; ++i)
            ::new (slots_ + i) std::atomic<uint8_t*>{nullptr};
    }

    bool exchange_block(uint8_t*& block, uint64_t ticket) noexcept
    {
        const size_t slot = ticket % slot_count_;
        // The exchange order defines ownership, even if ticket holders are delayed.
        uint8_t* old = slots_[slot].exchange(block, std::memory_order_acq_rel);
        block = old ? old : data_ + slot * BLOCK_BYTES;
        return old != nullptr;
    }

public:
    static constexpr size_t BLOCK_BYTES = 4096;
    static_assert(BLOCK_BYTES > 0 && BLOCK_BYTES % PAGE_SIZE == 0);

    ConcurrentFirstBlockRing(const ConcurrentFirstBlockRing&) = delete;
    ConcurrentFirstBlockRing& operator=(const ConcurrentFirstBlockRing&) = delete;

    static ConcurrentFirstBlockRing* create(ConcurrentMemoryPool* pool, int numa_node,
        uint32_t producer_count, size_t max_bytes)
    {
        constexpr size_t header_bytes = align_up(sizeof(ConcurrentFirstBlockRing), CACHE_LINE_SIZE);
        // Compare in blocks before multiplying, including for very large producer counts.
        if (!pool || producer_count == 0 || max_bytes / BLOCK_BYTES <= producer_count) {
            std::cerr << "ConcurrentFirstBlockRing invalid budget or producer count\n";
            std::exit(-1);
        }
        const size_t producer_bytes = size_t{producer_count} * BLOCK_BYTES;
        const size_t remaining = max_bytes - producer_bytes;
        // No intermediate multiplication or rounding can overflow this bound.
        auto fits = [&](size_t slots) {
            const size_t metadata = header_bytes + slots * sizeof(std::atomic<uint8_t*>);
            const size_t data_bytes = slots * BLOCK_BYTES;
            if (metadata > remaining || data_bytes > remaining - metadata) return false;
            const size_t padding = (PAGE_SIZE - metadata % PAGE_SIZE) % PAGE_SIZE;
            return padding <= remaining - metadata - data_bytes;
        };
        size_t low = 0;
        size_t high = remaining / (BLOCK_BYTES + sizeof(std::atomic<uint8_t*>));
        while (low < high) {
            const size_t middle = low + (high - low + 1) / 2;
            if (fits(middle)) low = middle;
            else high = middle - 1;
        }
        if (low == 0) {
            std::cerr << "ConcurrentFirstBlockRing budget cannot hold one slot\n";
            std::exit(-1);
        }
        const size_t data_offset = align_up(header_bytes + low * sizeof(std::atomic<uint8_t*>), PAGE_SIZE);
        const size_t bytes = data_offset + low * BLOCK_BYTES + producer_bytes;
        auto* memory = static_cast<uint8_t*>(pool->allocate_large_from_numa(bytes, numa_node));
        auto* slots = reinterpret_cast<std::atomic<uint8_t*>*>(memory + header_bytes);
        return ::new (memory) ConcurrentFirstBlockRing(low, bytes, slots, memory + data_offset);
    }

    uint8_t* initial_block(uint32_t producer_rank) const noexcept
    {
        return data_ + (slot_count_ + producer_rank) * BLOCK_BYTES;
    }

    bool publish_full_block(uint8_t*& block) noexcept
    {
        return exchange_block(block, head_.fetch_add(1, std::memory_order_relaxed));
    }

    // All producers must finish publishing and synchronize before draining.
    // Slots remain non-null; each claimed range is read exactly once.
    bool claim_drain_range(uint64_t& begin, uint64_t& end) noexcept
    {
        const uint64_t valid = std::min<uint64_t>(head_.load(std::memory_order_relaxed), slot_count_);
        begin = drain_cursor_.fetch_add(4, std::memory_order_relaxed);
        if (begin >= valid) return false;
        end = begin + std::min<uint64_t>(4, valid - begin);
        return true;
    }

    const uint8_t* drain_block(uint64_t slot) const noexcept
    {
        return slots_[slot].load(std::memory_order_relaxed);
    }
};
