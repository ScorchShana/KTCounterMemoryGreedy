#pragma once

#include "definition.h"
#include "kmer.h"
#include "../include/rapidhash.h"

#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

// Single-threaded, 64-slot bucketed hash map for waiting-for-second-occurrence.
// IMPORTANT CONTRACT: insert(key, evicted_key) requires key NOT to be resident.
// erase() removes a key when its second occurrence arrives.
// Replacement is round-robin / approximate FIFO, NOT true insertion-time FIFO.
// The input and output references of insert() must not alias each other.
//
// Preserve the sizing convention and public API of ReplacingHashMap_Bucket64.
template <uint32_t N, uint64_t MAX_CAPACITY>
class ReplacingHashMap
{
public:
    using KeyType = kmer<N>;

private:
    static_assert(N > 0);
    static constexpr size_t CAPACITY = std::bit_ceil(MAX_CAPACITY * 7 / 8);
    static_assert(CAPACITY >= 64 && (CAPACITY & (CAPACITY - 1)) == 0);
    static constexpr size_t BUCKET_SIZE = 64;
    static constexpr size_t BUCKETS = CAPACITY / BUCKET_SIZE;
    static constexpr size_t BUCKET_MASK = BUCKETS - 1;

    // Low 6 bits: [0,63] FIFO hand. Bit 6: all 64 slots occupied.
    static constexpr uint8_t FULL_BIT = 0x40;
    static constexpr uint8_t HAND_MASK = 0x3f;

    size_t size_ = 0;

public:
    // Keep separated SoA metadata. One bucket's controls fit one cache line.
    alignas(64) uint8_t controls_[CAPACITY] = {};
    KeyType keys_[CAPACITY];
#ifdef TEST_MODE
    inline static thread_local uint64_t search_empty_time = 0;
    inline static thread_local uint64_t replace_time = 0;
#endif

private:
    // Exactly one byte per bucket, same footprint as the old hand_ array.
    alignas(64) uint8_t state_[BUCKETS] = {};

    static uint8_t fingerprint(uint64_t h) noexcept
    {
        const auto f = static_cast<uint8_t>(h >> 56);
        return f != 0 ? f : uint8_t{ 255 };
    }

    // ctrl is 64-byte aligned: controls_ is alignas(64), and each bucket
    // starts at a multiple of BUCKET_SIZE (64).
    // Bit i in the result corresponds to control byte i (0 <= i < 64).
    static uint64_t scan_matches(const uint8_t* ctrl, uint8_t f) noexcept
    {
#if defined(__AVX2__)
        const auto* p = reinterpret_cast<const __m256i*>(ctrl);
        const __m256i a = _mm256_load_si256(p);
        const __m256i b = _mm256_load_si256(p + 1);
        const __m256i v = _mm256_set1_epi8(static_cast<char>(f));
        const auto low = static_cast<uint32_t>(
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, v)));
        const auto high = static_cast<uint32_t>(
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(b, v)));
        return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
#elif defined(__SSE2__)
        const auto* p = reinterpret_cast<const __m128i*>(ctrl);
        const __m128i v = _mm_set1_epi8(static_cast<char>(f));
        const auto m0 = static_cast<uint16_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_load_si128(p), v)));
        const auto m1 = static_cast<uint16_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_load_si128(p + 1), v)));
        const auto m2 = static_cast<uint16_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_load_si128(p + 2), v)));
        const auto m3 = static_cast<uint16_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_load_si128(p + 3), v)));
        return static_cast<uint64_t>(m0)
            | (static_cast<uint64_t>(m1) << 16)
            | (static_cast<uint64_t>(m2) << 32)
            | (static_cast<uint64_t>(m3) << 48);
#else
        uint64_t matches = 0;
        for (uint32_t i = 0; i < BUCKET_SIZE; ++i)
        {
            matches |= static_cast<uint64_t>(ctrl[i] == f) << i;
        }
        return matches;
#endif
    }

    static uint64_t scan_empty(const uint8_t* ctrl) noexcept
    {
#if defined(__AVX2__)
        const auto* p = reinterpret_cast<const __m256i*>(ctrl);
        const __m256i a = _mm256_load_si256(p);
        const __m256i b = _mm256_load_si256(p + 1);
        const __m256i z = _mm256_setzero_si256();
        const auto low = static_cast<uint32_t>(
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(a, z)));
        const auto high = static_cast<uint32_t>(
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(b, z)));
        return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
#elif defined(__SSE2__)
        const auto* p = reinterpret_cast<const __m128i*>(ctrl);
        const __m128i z = _mm_setzero_si128();
        const auto m0 = static_cast<uint16_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_load_si128(p), z)));
        const auto m1 = static_cast<uint16_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_load_si128(p + 1), z)));
        const auto m2 = static_cast<uint16_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_load_si128(p + 2), z)));
        const auto m3 = static_cast<uint16_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(_mm_load_si128(p + 3), z)));
        return static_cast<uint64_t>(m0)
            | (static_cast<uint64_t>(m1) << 16)
            | (static_cast<uint64_t>(m2) << 32)
            | (static_cast<uint64_t>(m3) << 48);
#else
        uint64_t empty = 0;
        for (uint32_t i = 0; i < BUCKET_SIZE; ++i)
        {
            empty |= static_cast<uint64_t>(ctrl[i] == 0) << i;
        }
        return empty;
#endif
    }

    static uint32_t first_empty_from(uint64_t empty, uint32_t head) noexcept
    {
        return (head + std::countr_zero(std::rotr(empty, static_cast<int>(head)))) & 63u;
    }

public:
    ReplacingHashMap() = default;

    static constexpr size_t capacity() noexcept { return CAPACITY; }
    size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }
    bool full() const noexcept { return size_ == CAPACITY; }
    float load_factor() const noexcept { return static_cast<float>(size_) / CAPACITY; }

    bool contains(const KeyType& key) noexcept
    {
        const uint64_t h = rapidhash(&key, sizeof(KeyType));
        const size_t bucket = static_cast<size_t>(h) & BUCKET_MASK;
        const size_t base = bucket * BUCKET_SIZE;
        uint64_t matches = scan_matches(controls_ + base, fingerprint(h));
        while (matches != 0) {
            const unsigned lane = std::countr_zero(matches);
            if (keys_[base + lane] == key) return true;
            matches &= matches - 1;
        }
        return false;
    }

    bool erase(const KeyType& key) noexcept
    {
        const uint64_t h = rapidhash(&key, sizeof(KeyType));
        const size_t bucket = static_cast<size_t>(h) & BUCKET_MASK;
        const size_t base = bucket * BUCKET_SIZE;
        uint64_t matches = scan_matches(controls_ + base, fingerprint(h));
        while (matches != 0) {
            const unsigned lane = std::countr_zero(matches);
            if (keys_[base + lane] == key) {
                controls_[base + lane] = 0;
                state_[bucket] &= static_cast<uint8_t>(~FULL_BIT);
                --size_;
                return true;
            }
            matches &= matches - 1;
        }
        return false;
    }

    // Requires key to be ABSENT. Returns true only if a resident key was
    // replaced, copying that old key to evicted_key. Otherwise leaves the
    // evicted_key argument unchanged.
    [[nodiscard]] bool insert(const KeyType& key, KeyType& evicted_key) noexcept
    {
        assert(&key != &evicted_key);
        const uint64_t h = rapidhash(&key, sizeof(KeyType));
        const size_t bucket = static_cast<size_t>(h) & BUCKET_MASK;
        const size_t base = bucket * BUCKET_SIZE;
        const uint8_t state = state_[bucket];
        const uint32_t head = state & HAND_MASK;

        if (state & FULL_BIT)
        {
            // Hot path: no 64B control scan and no complete-key lookup.
#ifdef TEST_MODE
            replace_time += 1;
#endif
            const size_t slot = base + head;
            evicted_key = keys_[slot];
            keys_[slot] = key;
            controls_[slot] = fingerprint(h);
            state_[bucket] = FULL_BIT | static_cast<uint8_t>((head + 1) & HAND_MASK);
            return true;
        }

#ifdef TEST_MODE
        search_empty_time += 1;
#endif
        // After erase / during initial filling: look for an empty lane.
        const uint64_t empty = scan_empty(controls_ + base);
        const uint32_t lane = first_empty_from(empty, head);
        const size_t slot = base + lane;
        keys_[slot] = key;
        controls_[slot] = fingerprint(h);
        ++size_;

        // const uint8_t next_head = static_cast<uint8_t>(
        //     lane == head ? ((head + 1) & HAND_MASK) : head);
        const uint8_t next_head = (lane + 1) & HAND_MASK;
        const bool now_full = (empty & (empty - 1)) == 0;
        state_[bucket] = next_head | (now_full ? FULL_BIT : 0);
        return false;
    }

    void clear() noexcept
    {
        std::memset(controls_, 0, sizeof(controls_));
        std::memset(state_, 0, sizeof(state_));
        size_ = 0;
    }

    template <class F>
    void for_each(F&& func) const
    {
        size_t remaining = size_;
        if constexpr (CAPACITY > BUCKET_SIZE)
        {
            for (size_t i = 0; i < CAPACITY && remaining; i += BUCKET_SIZE)
            {
#if defined(__AVX2__)
                constexpr size_t GROUP_SIZE = 32;
                for (size_t j = 0; j < BUCKET_SIZE; j += GROUP_SIZE)
                {
                    const auto* p = reinterpret_cast<const __m256i*>(controls_ + i + j);
                    const __m256i ctrl_bucket = _mm256_load_si256(p);
                    const __m256i zero = _mm256_setzero_si256();
                    auto mask = static_cast<uint32_t>(
                        _mm256_movemask_epi8(_mm256_cmpeq_epi8(ctrl_bucket, zero)));
                    mask = ~mask;
                    while (mask != 0)
                    {
                        const unsigned lane = std::countr_zero(mask);
                        func(keys_[i + j + lane]);
                        --remaining;
                        mask &= mask - 1;
                    }
            }
#elif defined(__SSE2__)
                constexpr size_t GROUP_SIZE = 16;
                for (size_t j = 0; j < BUCKET_SIZE; j += GROUP_SIZE)
                {
                    const auto* p = reinterpret_cast<const __m128i*>(controls_ + i + j);
                    const __m128i ctrl_bucket = _mm_load_si128(p);
                    const __m128i zero = _mm_setzero_si128();
                    auto mask = static_cast<uint16_t>(
                        _mm_movemask_epi8(_mm_cmpeq_epi8(ctrl_bucket, zero)));
                    mask = ~mask;
                    while (mask != 0)
                    {
                        const unsigned lane = std::countr_zero(mask);
                        func(keys_[i + j + lane]);
                        --remaining;
                        mask &= mask - 1;
                    }
                }
#else
                for (size_t j = 0; j < BUCKET_SIZE; ++j)
                {
                    if (controls_[i + j] != 0)
                    {
                        func(keys_[i + j]);
                        --remaining;
                    }
        }
#endif

    }
}
        else {
            for (size_t i = 0; i < CAPACITY && remaining; ++i)
            {
                if (controls_[i] != 0)
                {
                    func(keys_[i]);
                    --remaining;
                }
            }
        }
    }
};
