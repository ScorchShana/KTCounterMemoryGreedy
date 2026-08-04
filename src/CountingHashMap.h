#ifndef COUNTING_HASH_MAP_HEADER
#define COUNTING_HASH_MAP_HEADER

#include "kmer.h"
#include "../include/rapidhash.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__SSE4_2__)
#include <immintrin.h>
#endif

/**
 * @brief CountingHashMap - A high-performance, cache-friendly hash table for counting
 *
 * Features:
 * - Header-only, single-threaded
 * - Open addressing with linear probing
 * - SIMD-accelerated control byte matching (AVX2/SSE4.2/scalar fallback)
 * - Fixed capacity, no expansion
 * - No deletion support
 *
 * @tparam KeyCount Number of uint64_t elements in the key (1-4)
 * @tparam ValueType Counter type (default uint32_t)
 * @tparam MaxBytes Maximum memory usage (default 1MB)
 */
template <
    uint32_t N,
    size_t MaxBytes = 128 * 1024,
    typename ValueType = uint32_t>
class CountingHashMap
{

public:
    using KeyType = kmer<N>;

    CountingHashMap() = default;

    /**
     * @brief Increment the count for a key, or insert with value 1 if not exists
     * @param key The key to increment
     * @return true if successful, false if table is full
     */
    bool increment(const KeyType& key);

    /**
     * @brief Iterate over all entries
     * @tparam Func Function type: void(const KeyType&, ValueType&)
     * @param func Callback function for each entry
     */
    template <typename Func>
    void for_each(Func&& func);

    /**
     * @brief Iterate over all entries (const version)
     * @tparam Func Function type: void(const KeyType&, ValueType)
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
    [[nodiscard]] constexpr size_t capacity() const { return CAPACITY; }
    [[nodiscard]] float load_factor() const
    {
        return static_cast<float>(size_) / CAPACITY;
    }
    [[nodiscard]] bool empty() const { return size_ == 0; }
    [[nodiscard]] bool full() const { return size_ >= MAX_ENTRIES; }

private:

    // SIMD group size
#if defined(__AVX2__)
    static constexpr size_t GROUP_SIZE = 32;
#elif defined(__SSE4_2__)
    static constexpr size_t GROUP_SIZE = 16;
#else
    static constexpr size_t GROUP_SIZE = 8;
#endif

    // Compile-time constants
    static constexpr size_t ENTRY_SIZE = sizeof(KeyType) + sizeof(ValueType) + sizeof(uint8_t); // Key + Value + control byte
    static constexpr size_t RAW_CAPACITY = MaxBytes / ENTRY_SIZE;
    static_assert(RAW_CAPACITY >= GROUP_SIZE, "MaxBytes too small for CountingHashMap");
    static constexpr size_t CAPACITY =
        RAW_CAPACITY > 0 ? (1ULL << (63 - __builtin_clzll(RAW_CAPACITY))) : GROUP_SIZE;
    static constexpr double MAX_LOAD_FACTOR = 0.875; // recommended load factor
    static constexpr size_t MAX_ENTRIES = static_cast<size_t>(CAPACITY * MAX_LOAD_FACTOR);



    // Hash function
    static uint64_t hash_key(const KeyType& key)
    {
        const uint64_t h = rapidhash(&key, sizeof(KeyType));
        return h;
    }

    // Extract fingerprint from hash (high 7 bits, ensure non-zero)
    static uint8_t fingerprint(uint64_t hash)
    {
        return static_cast<uint8_t>((hash >> 57) | 0x01);
    }

    // SIMD match and empty detection
    // Safe: controls_ has extra GROUP_SIZE-1 bytes, so SIMD load never overflows
    std::pair<uint32_t, uint32_t> match_and_empty(size_t base, uint8_t fp) const noexcept
    {

#if defined(__AVX2__)
        __m256i ctrl = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(&controls_[base]));
        __m256i fp_vec = _mm256_set1_epi8(static_cast<char>(fp));

        uint32_t match_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(ctrl, fp_vec));
        uint32_t empty_mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(ctrl, _mm256_setzero_si256()));
        return { match_mask, empty_mask };

#elif defined(__SSE4_2__)
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

    // Data members
    // Extra GROUP_SIZE-1 bytes for safe SIMD load at boundary (no branch needed)
    alignas(32) uint8_t controls_[CAPACITY + GROUP_SIZE - 1] = {};

    KeyType keys_[CAPACITY];
    ValueType counts_[CAPACITY] = {};

    size_t size_ = 0;
};

// Implementation
template <uint32_t N, size_t MaxBytes, typename ValueType>
bool CountingHashMap<N, MaxBytes, ValueType>::increment(const KeyType& key)
{

    if(size_ >= MAX_ENTRIES) [[unlikely]]
    {
        return false; // Table is full
    }

    constexpr uint64_t mod = CAPACITY - 1;
    uint64_t h = hash_key(key);
    uint8_t fp = fingerprint(h);
    size_t idx = h & mod;

    for (size_t probe = 0; probe < CAPACITY; probe += GROUP_SIZE)
    {
        size_t base = (idx + probe) & mod;

        auto [match_mask, empty_mask] = match_and_empty(base, fp);

        // Process matching candidates
        while (match_mask)
        {
            int bit = __builtin_ctz(match_mask);
            size_t slot = (base + bit) & mod;
            if (keys_[slot] == key)
            {
                counts_[slot]++;
                return true;
            }
            match_mask &= (match_mask - 1);
        }

        // 遇到空槽立即插入并返回
        if (empty_mask) [[likely]]
        {
            // 正常负载下，绝大多数情况在这里返回
            int bit = __builtin_ctz(empty_mask);
            size_t slot = (base + bit) & mod;

            controls_[slot] = fp;
            if (slot < GROUP_SIZE - 1) {
                controls_[CAPACITY + slot] = fp;
            }
            keys_[slot] = key;
            counts_[slot] = 1;
            ++size_;
            return true;
        }
    }

    return false;
}

template <uint32_t N, size_t MaxBytes, typename ValueType>
template <typename Func>
void CountingHashMap<N, MaxBytes, ValueType>::for_each(Func&& func)
{
    for (size_t i = 0; i < CAPACITY; ++i)
    {
        if (controls_[i] != 0) [[likely]]
        {
            func(keys_[i], counts_[i]);
        }
    }
}

template <uint32_t N, size_t MaxBytes, typename ValueType>
template <typename Func>
void CountingHashMap<N, MaxBytes, ValueType>::for_each(Func&& func) const
{
    for (size_t i = 0; i < CAPACITY; ++i)
    {
        if (controls_[i] != 0) [[likely]]
        {
            func(keys_[i], counts_[i]);
        }
    }
}

template <uint32_t N, size_t MaxBytes, typename ValueType>
void CountingHashMap<N, MaxBytes, ValueType>::clear()
{
    std::memset(controls_, 0, CAPACITY + GROUP_SIZE - 1);
    size_ = 0;
}

#endif // COUNTING_HASH_TABLE_HEADER