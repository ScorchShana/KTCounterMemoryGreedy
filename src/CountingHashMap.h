#ifndef COUNTING_HASH_MAP_HEADER
#define COUNTING_HASH_MAP_HEADER

#include "definition.h"
#include "kmer.h"
#include "../include/rapidhash.h"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <emmintrin.h>
#endif

/**
 * @brief CountingHashMap - A high-performance, cache-friendly hash table for counting
 *
 * Features:
 * - Header-only, single-threaded
 * - Open addressing with quadratic probing
 * - SIMD-accelerated control byte matching (AVX2/SSE4.2/scalar fallback)
 * - Fixed capacity, no expansion
 * - No deletion support
 *
 */
class CountingHashMap
{

public:

    CountingHashMap()
        : bytes_(kmer_bytes()), RAW_CAPACITY(MAX_KMER_BLOCK_NUM * (4092 / bytes_)),
          CAPACITY(std::max(GROUP_SIZE, std::bit_ceil((RAW_CAPACITY * 8 + 6) / 7))),
          controls_(CAPACITY + GROUP_SIZE - 1), keys_(CAPACITY * bytes_), counts_(CAPACITY) {}

    /**
     * @brief Increment the count for a key, or insert with value 1 if not exists
     * @param key The key to increment
     * @return true if successful, false if table is full
     */
    void increment(const unsigned char* key);

    /**
     * @brief Iterate over all entries
     * @pre Callback must not insert into or clear this map during traversal.
     * @tparam Func Function type: void(const KeyType&, uint16_t&)
     * @param func Callback function for each entry
     */
    template <typename Func>
    void for_each(Func&& func);

    /**
     * @brief Iterate over all entries (const version)
     * @pre Callback must not insert into or clear this map during traversal.
     * @tparam Func Function type: void(const KeyType&, uint16_t)
     * @param func Callback function for each entry
     */
    template <typename Func>
    void for_each(Func&& func) const;

    /**
     * @brief Clear all entries (fast: only clears control array)
     */
    void clear();

    // Status queries
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] size_t capacity() const { return CAPACITY; }
    [[nodiscard]] float load_factor() const
    {
        return static_cast<float>(size_) / CAPACITY;
    }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] bool full() const { return size_ >= RAW_CAPACITY; }

private:

    // SIMD group size
// #if defined(__AVX2__)
//     static constexpr size_t GROUP_SIZE = 32;
#if defined(__SSE4_2__) || defined(__AVX2__)
    static constexpr size_t GROUP_SIZE = 16;
#else
    static constexpr size_t GROUP_SIZE = 8;
#endif

    const size_t bytes_;
    const size_t RAW_CAPACITY;
    const size_t CAPACITY;

    uint64_t hash_key(const unsigned char* key) const
    { return rapidhash(key, bytes_); }

    // Extract fingerprint from hash (high 8 bits, ensure non-zero)
    static uint8_t fingerprint(const uint64_t h)
    {
        uint8_t fp = static_cast<uint8_t>(h >> 56);  // 用满 8 位
        return fp == 0 ? 0xFF : fp;
    }

    // SIMD match and empty detection
    // Safe: controls_ has extra GROUP_SIZE-1 bytes, so SIMD load never overflows
    std::pair<uint32_t, uint32_t> match_and_empty(size_t base, uint8_t fp) const noexcept
    {

        // #if defined(__AVX2__)
        //         __m256i ctrl = _mm256_loadu_si256(
        //             reinterpret_cast<const __m256i*>(&controls_[base]));
        //         __m256i fp_vec = _mm256_set1_epi8(static_cast<char>(fp));

        //         uint32_t match_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(ctrl, fp_vec));
        //         uint32_t empty_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(ctrl, _mm256_setzero_si256()));
        //         return { match_mask, empty_mask };

#if defined(__SSE4_2__) || defined(__AVX2__)
        __m128i ctrl = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&controls_[base]));
        __m128i fp_vec = _mm_set1_epi8(static_cast<char>(fp));

        uint32_t match_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(ctrl, fp_vec));
        uint32_t empty_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(ctrl, _mm_setzero_si128()));
        return { match_mask, empty_mask };

#else
        uint32_t match_mask = 0, empty_mask = 0;
        for (size_t i = 0; i < GROUP_SIZE; ++i)
        {
            uint8_t c = controls_[base + i];
            if (c == 0)
                empty_mask |= (1u << i);
            else if (c == fp)
                match_mask |= (1u << i);
        }
        return { match_mask, empty_mask };
#endif
    }

    size_t size_ = 0;

public:
    // Data members
    // Extra GROUP_SIZE-1 bytes for safe SIMD load at boundary (no branch needed)
    std::vector<uint8_t> controls_;

    std::vector<unsigned char> keys_;
    std::vector<uint16_t> counts_;


};

// Implementation
inline void CountingHashMap::increment(const unsigned char* key)
{

    uint64_t h = hash_key(key);
    uint8_t fp = fingerprint(h);
    const size_t MOD = CAPACITY - 1;
    size_t base = h & MOD;
    size_t offset = 0;

    for (size_t probe = 0; probe < CAPACITY; probe += GROUP_SIZE)
    {

        // auto [match_mask, empty_mask] = match_and_empty(base, fp);

        uint32_t match_mask = 0;
        uint32_t empty_mask = 0;

#if defined(__SSE4_2__) || defined(__AVX2__)
        __m128i ctrl = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&controls_[base]));
        __m128i fp_vec = _mm_set1_epi8(static_cast<char>(fp));

        match_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(ctrl, fp_vec));
#else
        for (size_t i = 0; i < GROUP_SIZE; ++i)
        {
            uint8_t c = controls_[base + i];
            if (c == fp)
                match_mask |= (1u << i);
        }
#endif

        // Process matching candidates
        while (match_mask)
        {
            int bit = __builtin_ctz(match_mask);
            size_t slot = (base + bit) & MOD;
            if (std::memcmp(keys_.data() + slot * bytes_, key, bytes_) == 0)
            {
                counts_[slot]++;
                return;
            }
            match_mask &= (match_mask - 1);
        }

#if defined(__SSE4_2__) || defined(__AVX2__)
        empty_mask = _mm_movemask_epi8(_mm_cmpeq_epi8(ctrl, _mm_setzero_si128()));
#else
        for (size_t i = 0; i < GROUP_SIZE; ++i)
        {
            uint8_t c = controls_[base + i];
            if (c == 0)
                empty_mask |= (1u << i);
        }
#endif

        // 遇到空槽立即插入并返回
        if (empty_mask) [[likely]]
        {
            // 正常负载下，绝大多数情况在这里返回
            int bit = __builtin_ctz(empty_mask);
            size_t slot = (base + bit) & MOD;

            controls_[slot] = fp;
            if (slot < GROUP_SIZE - 1) {
                controls_[CAPACITY + slot] = fp;
            }
            std::memcpy(keys_.data() + slot * bytes_, key, bytes_);
            counts_[slot] = 1;
            ++size_;
            return;
        }

        offset += GROUP_SIZE;
        offset &= MOD;

        base += GROUP_SIZE;
        base &= MOD;
    }
}

template <typename Func>
void CountingHashMap::for_each(Func&& func)
{
#if defined(__AVX2__)
    constexpr size_t WIDTH = 32;
    for (size_t base = 0; base < CAPACITY; base += WIDTH)
    {
        const __m256i ctrl = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(controls_.data() + base));
        uint32_t occupied = ~static_cast<uint32_t>(_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(ctrl, _mm256_setzero_si256())));

        while (occupied != 0)
        {
            const unsigned lane = std::countr_zero(occupied);
            const size_t slot = base + lane;
            func(keys_.data() + slot * bytes_, counts_[slot]);
            occupied &= occupied - 1;
        }
    }

#elif defined(__SSE4_2__)
    constexpr size_t WIDTH = 16;
    for (size_t base = 0; base < CAPACITY; base += WIDTH)
    {
        const __m128i ctrl = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(controls_.data() + base));
        uint32_t occupied = ~static_cast<uint32_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(ctrl, _mm_setzero_si128()))) & 0xffffU;
        
        while (occupied != 0)
        {
            const unsigned lane = std::countr_zero(occupied);
            const size_t slot = base + lane;
            func(keys_.data() + slot * bytes_, counts_[slot]);
            occupied &= occupied - 1;
        }
    }
#else
    for (size_t i = 0; i < CAPACITY; ++i)
        if (controls_[i] != 0) func(keys_.data() + i * bytes_, counts_[i]);
#endif
}

template <typename Func>
void CountingHashMap::for_each(Func&& func) const
{
#if defined(__AVX2__)
    constexpr size_t WIDTH = 32;
    for (size_t base = 0; base < CAPACITY; base += WIDTH)
    {
        const __m256i ctrl = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(controls_.data() + base));
        uint32_t occupied = ~static_cast<uint32_t>(_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(ctrl, _mm256_setzero_si256())));

        while (occupied != 0)
        {
            const unsigned lane = std::countr_zero(occupied);
            const size_t slot = base + lane;
            func(keys_.data() + slot * bytes_, counts_[slot]);
            occupied &= occupied - 1;
        }
    }

#elif defined(__SSE4_2__)
    constexpr size_t WIDTH = 16;
    for (size_t base = 0; base < CAPACITY; base += WIDTH)
    {
        const __m128i ctrl = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(controls_.data() + base));
        uint32_t occupied = ~static_cast<uint32_t>(_mm_movemask_epi8(
            _mm_cmpeq_epi8(ctrl, _mm_setzero_si128()))) & 0xffffU;

        while (occupied != 0)
        {
            const unsigned lane = std::countr_zero(occupied);
            const size_t slot = base + lane;
            func(keys_.data() + slot * bytes_, counts_[slot]);
            occupied &= occupied - 1;
        }
    }
#else
    for (size_t i = 0; i < CAPACITY; ++i)
        if (controls_[i] != 0) func(keys_.data() + i * bytes_, counts_[i]);
#endif
}

inline void CountingHashMap::clear()
{
    std::memset(controls_.data(), 0, CAPACITY + GROUP_SIZE - 1);
    size_ = 0;
}

#endif // COUNTING_HASH_TABLE_HEADER