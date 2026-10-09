#ifndef REPLACING_HASH_MAP_HEADER
#define REPLACING_HASH_MAP_HEADER

#include "definition.h"
#include "kmer.h"
#include "../include/rapidhash.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__SSE4_2__) || defined(__AVX2__)
#include <immintrin.h>
#endif

template <uint32_t N, uint64_t MAX_CAPACITY>
class ReplacingHashMap
{
public:
    using KeyType = kmer<N>;

    ReplacingHashMap() = default;

    /**
     * @brief Query whether key exists.
     *
     * A successful lookup marks the slot as recently used for approximate
     * CLOCK-style replacement.
     */
    [[nodiscard]] bool contains(const KeyType& key) noexcept;

    /**
     * @brief Erase key if present, leaving an EMPTY slot instead of a tombstone.
     * @return true only if a resident key was removed.
     */
    [[nodiscard]] bool erase(const KeyType& key) noexcept;

    /**
     * @brief Insert key if it does not already exist.
     *
     * Scans all MAX_GROUP metadata groups before checking candidate keys.
     * After ruling out duplicates in the entire window, inserts into the first
     * EMPTY slot or replaces one slot from the same probe window if it is full.
     * No tombstone is ever created.
     *
     * @param evicted_key Receives the old key only when replacement occurs.
     *                    It is left unchanged for an existing-key hit or an
     *                    insertion into an EMPTY slot.
     * @return true only if an existing resident key was replaced.
     */
    [[nodiscard]] bool insert(const KeyType& key, KeyType& evicted_key) noexcept;

    /**
     * @brief Iterate over all resident keys in increasing physical-slot order.
     *
     * The callback must not insert, erase, or clear this map during iteration.
     * @tparam Func Function type: void(const KeyType&)
     */
    template <typename Func>
    void for_each(Func&& func) const;

    /**
     * @brief Clear all entries.
     */
    void clear() noexcept;

    [[nodiscard]] size_t size() const noexcept { return size_; }
    [[nodiscard]] static constexpr size_t capacity() noexcept { return CAPACITY; }
    [[nodiscard]] float load_factor() const noexcept
    {
        return static_cast<float>(size_) / static_cast<float>(CAPACITY);
    }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] bool full() const noexcept { return size_ == CAPACITY; }

private:
    static constexpr size_t GROUP_SIZE = 16;
    static constexpr size_t MAX_GROUP = 4;
    static_assert(MAX_GROUP == 4, "Four-group probing is explicitly unrolled");

    struct Probe4
    {
        size_t bases[MAX_GROUP];
        // Bits [16*g, 16*g+15] represent the lanes of probe group g.
        uint64_t match_mask;
        uint64_t empty_mask;
    };

    static_assert(MAX_CAPACITY >= GROUP_SIZE,
        "MaxBytes too small for ReplacingHashMap");

    // Keep power-of-two capacity so masking and Swiss-style probing are cheap.
    // The old table reserved ~12.5% headroom this way; retain that sizing here.
    static constexpr size_t CAPACITY = std::bit_ceil(MAX_CAPACITY * 7 / 8);
    static constexpr size_t MOD = CAPACITY - 1;
    static_assert((CAPACITY& MOD) == 0,
        "ReplacingHashMap capacity must be a power of two");

    static constexpr size_t RECENT_WORDS = (CAPACITY + 63) / 64;

    static uint64_t hash_key(const KeyType& key) noexcept
    {
        return rapidhash(&key, sizeof(KeyType));
    }

    // 0 is reserved for EMPTY, so remap fingerprint 0 to 0xFF.
    static uint8_t fingerprint(uint64_t h) noexcept
    {
        const uint8_t fp = static_cast<uint8_t>(h >> 56);
        return fp == 0 ? uint8_t{ 0xFF } : fp;
    }

    [[nodiscard]] Probe4 probe_four_groups(uint64_t h, uint8_t fp) const noexcept
    {
        const size_t home = static_cast<size_t>(h) & MOD;
        Probe4 probe{
            { home, (home + GROUP_SIZE) & MOD,
              (home + 3 * GROUP_SIZE) & MOD, (home + 6 * GROUP_SIZE) & MOD },
            0, 0
        };

#if defined(__SSE4_2__) || defined(__AVX2__)
        // Expose four independent loads before any candidate-key comparisons.
        const __m128i ctrl0 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&controls_[probe.bases[0]]));
        const __m128i ctrl1 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&controls_[probe.bases[1]]));
        const __m128i ctrl2 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&controls_[probe.bases[2]]));
        const __m128i ctrl3 = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(&controls_[probe.bases[3]]));
        const __m128i fp_vec = _mm_set1_epi8(static_cast<char>(fp));

        const auto pack_matches = [ctrl0, ctrl1, ctrl2, ctrl3](__m128i target) noexcept
            {
                return static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl0, target)))
                    | (static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl1, target))) << 16)
                    | (static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl2, target))) << 32)
                    | (static_cast<uint64_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(ctrl3, target))) << 48);
            };
        probe.match_mask = pack_matches(fp_vec);
        probe.empty_mask = pack_matches(_mm_setzero_si128());
#else
        for (size_t group = 0; group < MAX_GROUP; ++group)
        {
            for (size_t lane = 0; lane < GROUP_SIZE; ++lane)
            {
                const uint8_t c = controls_[probe.bases[group] + lane];
                const uint64_t bit = uint64_t{ 1 } << (group * GROUP_SIZE + lane);
                if (c == 0)
                    probe.empty_mask |= bit;
                else if (c == fp)
                    probe.match_mask |= bit;
            }
        }
#endif
        return probe;
    }

    [[nodiscard]] size_t find_slot(const KeyType& key, const Probe4& probe) const noexcept
    {
        uint64_t matches = probe.match_mask;
        while (matches != 0)
        {
            // Lowest-bit order visits G0, G1, G2, G3 and then each group's lanes.
            const unsigned bit = std::countr_zero(matches);
            const size_t slot = (probe.bases[bit / GROUP_SIZE] + bit % GROUP_SIZE) & MOD;
            if (keys_[slot] == key)
                return slot;
            matches &= matches - 1;
        }
        // EMPTY cannot terminate lookup: an earlier group may have been erased.
        return CAPACITY;
    }

    void set_control(size_t slot, uint8_t fp) noexcept
    {
        controls_[slot] = fp;

        // controls_[CAPACITY ...] mirrors the beginning so a SIMD group can
        // cross the physical end of controls_ without a branch.
        if (slot < GROUP_SIZE - 1)
            controls_[CAPACITY + slot] = fp;
    }

    void mark_recent(size_t slot) noexcept
    {
        recent_[slot >> 6] |= uint64_t{ 1 } << (slot & 63);
    }

    void clear_recent(size_t slot) noexcept
    {
        recent_[slot >> 6] &= ~(uint64_t{ 1 } << (slot & 63));
    }

    /**
     * @brief Extract the 16 recent bits starting at an arbitrary probe base.
     *
     * Bit i in the returned mask corresponds to slot (base + i) & MOD.
     * For normal tables (CAPACITY >= 64), wrapping at the table end is the
     * same operation as wrapping to the next recent_ word, so both the
     * uint64_t-boundary case and the physical table-end case share one path.
     */
    [[nodiscard]] uint16_t extract_recent16(size_t base) const noexcept
    {
        // Internal callers always pass a probe base already reduced by MOD.
        if constexpr (CAPACITY < 64)
        {
            // Small-table fallback: the circular table occupies only part of
            // recent_[0], so physical table wrap is not a uint64_t wrap.
            const unsigned bit = static_cast<unsigned>(base);
            const uint64_t word = recent_[0];

            if (bit <= CAPACITY - GROUP_SIZE)
                return static_cast<uint16_t>(word >> bit);

            const unsigned tail_bits = static_cast<unsigned>(CAPACITY) - bit;
            return static_cast<uint16_t>((word >> bit) | (word << tail_bits));
        }
        else
        {
            const size_t word = base >> 6;
            const unsigned bit = static_cast<unsigned>(base & 63);
            const uint64_t low = recent_[word] >> bit;

            if (bit <= 64 - GROUP_SIZE) [[likely]]
                return static_cast<uint16_t>(low);

            // RECENT_WORDS is a power of two when CAPACITY >= 64. The mask
            // wraps the last recent_ word directly back to recent_[0].
            const size_t next_word = (word + 1) & (RECENT_WORDS - 1);
            return static_cast<uint16_t>(
                low | (recent_[next_word] << (64 - bit)));
        }
    }

    /**
     * @brief Clear the 16 recent bits starting at an arbitrary probe base.
     *
     * The common path needs one uint64_t read-modify-write. A range crossing a
     * uint64_t boundary needs two. For CAPACITY >= 64 the same next-word mask
     * also handles wrapping from the physical table tail back to the head.
     */
    void clear_recent16(size_t base) noexcept
    {
        // Internal callers always pass a probe base already reduced by MOD.
        if constexpr (CAPACITY < 64)
        {
            const unsigned bit = static_cast<unsigned>(base);

            if (bit <= CAPACITY - GROUP_SIZE)
            {
                recent_[0] &= ~(uint64_t{ 0xFFFF } << bit);
                return;
            }

            const unsigned tail_bits = static_cast<unsigned>(CAPACITY) - bit;
            const unsigned head_bits = static_cast<unsigned>(GROUP_SIZE) - tail_bits;
            const uint64_t tail_mask = UINT64_MAX << bit;
            const uint64_t head_mask = (uint64_t{ 1 } << head_bits) - 1;
            recent_[0] &= ~(tail_mask | head_mask);
        }
        else
        {
            const size_t word = base >> 6;
            const unsigned bit = static_cast<unsigned>(base & 63);

            if (bit <= 64 - GROUP_SIZE) [[likely]]
            {
                recent_[word] &= ~(uint64_t{ 0xFFFF } << bit);
                return;
            }

            const size_t next_word = (word + 1) & (RECENT_WORDS - 1);
            const unsigned high_bits = bit - (64 - GROUP_SIZE);

            // Clear [bit, 63] in the current word and the remaining low bits
            // in the circular next word. high_bits is in [1, 15].
            recent_[word] &= ~(UINT64_MAX << bit);
            recent_[next_word] &= ~((uint64_t{ 1 } << high_bits) - 1);
        }
    }

    size_t size_ = 0;

public:
    // Extra GROUP_SIZE - 1 bytes mirror the beginning of controls_.
    alignas(32) uint8_t controls_[CAPACITY + GROUP_SIZE - 1] = {};
    KeyType keys_[CAPACITY];

private:
    // One CLOCK/reference bit per physical slot.
    alignas(64) uint64_t recent_[RECENT_WORDS] = {};
};

// -----------------------------------------------------------------------------
// Implementation
// -----------------------------------------------------------------------------

template <uint32_t N, uint64_t MAX_CAPACITY>
bool ReplacingHashMap<N, MAX_CAPACITY>::contains(const KeyType& key) noexcept
{
    const uint64_t h = hash_key(key);
    const Probe4 probe = probe_four_groups(h, fingerprint(h));
    const size_t slot = find_slot(key, probe);
    if (slot == CAPACITY)
        return false;

    mark_recent(slot);
    return true;
}

template <uint32_t N, uint64_t MAX_CAPACITY>
bool ReplacingHashMap<N, MAX_CAPACITY>::erase(const KeyType& key) noexcept
{
    const uint64_t h = hash_key(key);
    const Probe4 probe = probe_four_groups(h, fingerprint(h));
    const size_t slot = find_slot(key, probe);
    if (slot == CAPACITY)
        return false;

    set_control(slot, 0);
    clear_recent(slot);
    --size_;
    return true;
}

template <uint32_t N, uint64_t MAX_CAPACITY>
bool ReplacingHashMap<N, MAX_CAPACITY>::insert(
    const KeyType& key, KeyType& evicted_key) noexcept
{
    const uint64_t h = hash_key(key);
    const uint8_t fp = fingerprint(h);
    const Probe4 probe = probe_four_groups(h, fp);

    const size_t existing = find_slot(key, probe);
    if (existing != CAPACITY)
    {
        mark_recent(existing);
        return false;
    }

    // An earlier EMPTY cannot be used until duplicates in later groups are ruled out.
    if (probe.empty_mask != 0) [[likely]]
    {
        const unsigned bit = std::countr_zero(probe.empty_mask);
        const size_t slot = (probe.bases[bit / GROUP_SIZE] + bit % GROUP_SIZE) & MOD;
        keys_[slot] = key;
        set_control(slot, fp);
        mark_recent(slot);
        ++size_;
        return false;
    }

    // Every probed group is full. Prefer the first cold slot, using the same
    // CLOCK policy as before; recent_ is only read when replacement is needed.
    size_t candidate_base = 0;
    uint16_t candidate_mask = 0;
    for (size_t group = 0; group < MAX_GROUP; ++group)
    {
        const uint16_t cold16 = static_cast<uint16_t>(~extract_recent16(probe.bases[group]));
        if (cold16 != 0)
        {
            candidate_base = probe.bases[group];
            candidate_mask = cold16;
            break;
        }
    }

    size_t victim;
    if (candidate_mask != 0) [[likely]]
    {
        const unsigned lane = std::countr_zero(candidate_mask);
        victim = (candidate_base + lane) & MOD;
    }
    else
    {
        // All probed slots were recent. Give exactly these four probe groups a
        // second chance with four 16-bit range clears, then pick a lane in G0
        // from extra hash bits so the same slot is not selected systematically.
        for (size_t group = 0; group < MAX_GROUP; ++group)
            clear_recent16(probe.bases[group]);

        const size_t lane = static_cast<size_t>((h >> 32) & (GROUP_SIZE - 1));
        victim = (probe.bases[0] + lane) & MOD;
    }

    evicted_key = keys_[victim];
    keys_[victim] = key;
    set_control(victim, fp);
    mark_recent(victim);

    // size_ is unchanged: FULL -> FULL.
    return true;
}

template <uint32_t N, uint64_t MAX_CAPACITY>
template <typename Func>
void ReplacingHashMap<N, MAX_CAPACITY>::for_each(Func&& func) const
{
    size_t remaining = size_;
    if (remaining == 0)
        return;

#if defined(__SSE4_2__) || defined(__AVX2__)

    constexpr size_t SCAN_WIDTH = GROUP_SIZE;


    // Scan physical slots only; the trailing mirrored controls are not entries.
    for (size_t base = 0; base < CAPACITY; base += SCAN_WIDTH)
    {
        uint32_t live;

        const __m128i ctrl = _mm_load_si128(
            reinterpret_cast<const __m128i*>(&controls_[base]));
        live = static_cast<uint16_t>(~_mm_movemask_epi8(
            _mm_cmpeq_epi8(ctrl, _mm_setzero_si128())));


        while (live != 0)
        {
            const unsigned lane = std::countr_zero(live);
            func(static_cast<const KeyType&>(keys_[base + lane]));
            if (--remaining == 0)
                return;
            live &= live - 1;
        }
    }
#else
    for (size_t i = 0; i < CAPACITY; ++i)
    {
        if (controls_[i] != 0)
        {
            func(static_cast<const KeyType&>(keys_[i]));
            if (--remaining == 0)
                return;
        }
    }
#endif
}

template <uint32_t N, uint64_t MAX_CAPACITY>
void ReplacingHashMap<N, MAX_CAPACITY>::clear() noexcept
{
    std::memset(controls_, 0, sizeof(controls_));
    std::memset(recent_, 0, sizeof(recent_));
    size_ = 0;
}

#endif // REPLACING_HASH_MAP_HEADER
