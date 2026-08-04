#ifndef TREE_EXPORT_RECORD_RADIX_SORT_12_HEADER
#define TREE_EXPORT_RECORD_RADIX_SORT_12_HEADER

#include "../src/kmer.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define TREE_RADIX_RESTRICT __restrict__
#else
#define TREE_RADIX_RESTRICT
#endif

namespace tree::sort
{
namespace detail
{
inline constexpr uint32_t RADIX_BITS = 12;
inline constexpr uint32_t RADIX_SIZE = 1U << RADIX_BITS;
inline constexpr uint64_t RADIX_MASK = RADIX_SIZE - 1U;

struct PassDescriptor
{
    uint32_t word_index = 0;
    uint32_t shift = 0;
    uint32_t width = 0;
    uint32_t bucket_count = 0;
    uint64_t mask = 0;
    bool crosses_word = false;
};

template <uint32_t N>
inline bool padding_is_zero(const kmer<N>& key, uint32_t low_padding_bits) noexcept
{
    uint32_t remaining = low_padding_bits;
    for (uint32_t word = N; word-- > 0 && remaining > 0;)
    {
        const uint32_t bits = std::min<uint32_t>(remaining, 64U);
        const uint64_t mask = bits == 64U ? ~uint64_t{ 0 } : ((uint64_t{ 1 } << bits) - 1U);
        if ((key.data[word] & mask) != 0)
        {
            return false;
        }
        remaining -= bits;
    }
    return true;
}

template <uint32_t N, uint32_t CommonPrefixBases>
inline uint64_t common_prefix(const kmer<N>& key) noexcept
{
    static_assert(CommonPrefixBases <= 32, "The debug prefix check expects the common prefix in data[0]");
    if constexpr (CommonPrefixBases == 0)
    {
        return 0;
    }
    else
    {
        return key.data[0] >> (64U - 2U * CommonPrefixBases);
    }
}

inline void clear_histogram(uint32_t* histogram, uint32_t bucket_count) noexcept
{
#if defined(__AVX2__)
    const __m256i zero = _mm256_setzero_si256();
    uint32_t index = 0;
    for (; index + 8U <= bucket_count; index += 8U)
    {
        _mm256_store_si256(reinterpret_cast<__m256i*>(histogram + index), zero);
    }
    for (; index < bucket_count; ++index)
    {
        histogram[index] = 0;
    }
#else
    std::fill_n(histogram, bucket_count, uint32_t{ 0 });
#endif
}

inline void exclusive_prefix_sum(uint32_t* histogram, uint32_t bucket_count) noexcept
{
#if defined(__AVX2__)
    if (bucket_count >= 8U)
    {
        const __m256i shift_indices = _mm256_setr_epi32(0, 0, 1, 2, 3, 4, 5, 6);
        uint32_t carry = 0;
        uint32_t index = 0;

        for (; index + 8U <= bucket_count; index += 8U)
        {
            __m256i values = _mm256_load_si256(reinterpret_cast<const __m256i*>(histogram + index));
            values = _mm256_add_epi32(values, _mm256_slli_si256(values, 4));
            values = _mm256_add_epi32(values, _mm256_slli_si256(values, 8));

            const uint32_t lower_half_sum = static_cast<uint32_t>(_mm256_extract_epi32(values, 3));
            __m256i bridge = _mm256_setzero_si256();
            bridge = _mm256_inserti128_si256(
                bridge,
                _mm_set1_epi32(static_cast<int>(lower_half_sum)),
                1);
            values = _mm256_add_epi32(values, bridge);

            const __m256i carry_vector = _mm256_set1_epi32(static_cast<int>(carry));
            values = _mm256_add_epi32(values, carry_vector);

            __m256i exclusive = _mm256_permutevar8x32_epi32(values, shift_indices);
            exclusive = _mm256_blend_epi32(exclusive, carry_vector, 0x01);
            _mm256_store_si256(reinterpret_cast<__m256i*>(histogram + index), exclusive);

            carry = static_cast<uint32_t>(_mm256_extract_epi32(values, 7));
        }

        for (; index < bucket_count; ++index)
        {
            const uint32_t count = histogram[index];
            histogram[index] = carry;
            carry += count;
        }
        return;
    }
#endif

    uint32_t carry = 0;
    for (uint32_t index = 0; index < bucket_count; ++index)
    {
        const uint32_t count = histogram[index];
        histogram[index] = carry;
        carry += count;
    }
}

#if defined(__AVX2__)
template <std::size_t Offset, typename T>
inline void copy_record_avx2_chunks(T* destination, const T* source) noexcept
{
    constexpr std::size_t remaining = sizeof(T) - Offset;
    auto* destination_bytes = reinterpret_cast<std::byte*>(destination);
    const auto* source_bytes = reinterpret_cast<const std::byte*>(source);

    if constexpr (remaining >= 32)
    {
        const __m256i value = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(source_bytes + Offset));
        _mm256_storeu_si256(
            reinterpret_cast<__m256i*>(destination_bytes + Offset), value);
        copy_record_avx2_chunks<Offset + 32>(destination, source);
    }
    else if constexpr (remaining >= 16)
    {
        const __m128i value = _mm_loadu_si128(
            reinterpret_cast<const __m128i*>(source_bytes + Offset));
        _mm_storeu_si128(
            reinterpret_cast<__m128i*>(destination_bytes + Offset), value);
        copy_record_avx2_chunks<Offset + 16>(destination, source);
    }
    else if constexpr (remaining >= 8)
    {
        const __m128i value = _mm_loadl_epi64(
            reinterpret_cast<const __m128i*>(source_bytes + Offset));
        _mm_storel_epi64(
            reinterpret_cast<__m128i*>(destination_bytes + Offset), value);
        copy_record_avx2_chunks<Offset + 8>(destination, source);
    }
    else if constexpr (remaining > 0)
    {
        std::memcpy(destination_bytes + Offset, source_bytes + Offset, remaining);
    }
}
#endif

template <typename T>
inline void copy_record(T* destination, const T* source) noexcept
{
    static_assert(std::is_trivially_copyable_v<T>, "Radix-sort records must be trivially copyable");
#if defined(__AVX2__)
    copy_record_avx2_chunks<0>(destination, source);
#else
    std::memcpy(destination, source, sizeof(T));
#endif
}

template <bool CrossesWord, uint32_t N>
inline uint32_t extract_digit(const ExportRecord<N>& record, const PassDescriptor& pass) noexcept
{
    uint64_t digit = record.key.data[pass.word_index] >> pass.shift;
    if constexpr (CrossesWord)
    {
        assert(pass.word_index > 0);
        digit |= record.key.data[pass.word_index - 1U] << (64U - pass.shift);
    }
    return static_cast<uint32_t>(digit & pass.mask);
}

template <uint32_t N, uint32_t CommonPrefixBases, bool CrossesWord>
inline void count_pass(
    const ExportRecord<N>* TREE_RADIX_RESTRICT source,
    std::size_t record_count,
    const PassDescriptor& pass,
    uint32_t* histogram,
    uint32_t low_padding_bits,
    uint64_t expected_prefix,
    bool validate_contract) noexcept
{
    std::size_t index = 0;
    for (; index + 4U <= record_count; index += 4U)
    {
        const uint32_t digit0 = extract_digit<CrossesWord>(source[index], pass);
        const uint32_t digit1 = extract_digit<CrossesWord>(source[index + 1U], pass);
        const uint32_t digit2 = extract_digit<CrossesWord>(source[index + 2U], pass);
        const uint32_t digit3 = extract_digit<CrossesWord>(source[index + 3U], pass);

#ifndef NDEBUG
        if (validate_contract)
        {
            for (std::size_t lane = 0; lane < 4U; ++lane)
            {
                assert((common_prefix<N, CommonPrefixBases>(source[index + lane].key) == expected_prefix));
                assert(padding_is_zero(source[index + lane].key, low_padding_bits));
            }
        }
#else
        (void)low_padding_bits;
        (void)expected_prefix;
        (void)validate_contract;
#endif

        ++histogram[digit0];
        ++histogram[digit1];
        ++histogram[digit2];
        ++histogram[digit3];
    }

    for (; index < record_count; ++index)
    {
#ifndef NDEBUG
        if (validate_contract)
        {
            assert((common_prefix<N, CommonPrefixBases>(source[index].key) == expected_prefix));
            assert(padding_is_zero(source[index].key, low_padding_bits));
        }
#endif
        ++histogram[extract_digit<CrossesWord>(source[index], pass)];
    }
}

template <std::size_t PrefetchDistance, bool CrossesWord, uint32_t N>
inline void prefetch_destination(
    const ExportRecord<N>* source,
    ExportRecord<N>* destination,
    std::size_t record_count,
    std::size_t index,
    const PassDescriptor& pass,
    const uint32_t* positions) noexcept
{
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (PrefetchDistance > 0)
    {
        const std::size_t future = index + PrefetchDistance;
        if (future < record_count)
        {
            const uint32_t digit = extract_digit<CrossesWord>(source[future], pass);
            __builtin_prefetch(destination + positions[digit], 1, 1);
        }
    }
#else
    (void)source;
    (void)destination;
    (void)record_count;
    (void)index;
    (void)pass;
    (void)positions;
#endif
}

template <uint32_t N, bool CurrentCrosses, std::size_t PrefetchDistance>
inline void scatter_final_pass(
    const ExportRecord<N>* TREE_RADIX_RESTRICT source,
    ExportRecord<N>* TREE_RADIX_RESTRICT destination,
    std::size_t record_count,
    const PassDescriptor& current_pass,
    uint32_t* current_positions) noexcept
{
    std::size_t index = 0;
    for (; index + 4U <= record_count; index += 4U)
    {
        prefetch_destination<PrefetchDistance, CurrentCrosses>(
            source, destination, record_count, index, current_pass, current_positions);

        const uint32_t digit0 = extract_digit<CurrentCrosses>(source[index], current_pass);
        const uint32_t digit1 = extract_digit<CurrentCrosses>(source[index + 1U], current_pass);
        const uint32_t digit2 = extract_digit<CurrentCrosses>(source[index + 2U], current_pass);
        const uint32_t digit3 = extract_digit<CurrentCrosses>(source[index + 3U], current_pass);

        const uint32_t output0 = current_positions[digit0]++;
        const uint32_t output1 = current_positions[digit1]++;
        const uint32_t output2 = current_positions[digit2]++;
        const uint32_t output3 = current_positions[digit3]++;

        copy_record(destination + output0, source + index);
        copy_record(destination + output1, source + index + 1U);
        copy_record(destination + output2, source + index + 2U);
        copy_record(destination + output3, source + index + 3U);
    }

    for (; index < record_count; ++index)
    {
        prefetch_destination<PrefetchDistance, CurrentCrosses>(
            source, destination, record_count, index, current_pass, current_positions);
        const uint32_t digit = extract_digit<CurrentCrosses>(source[index], current_pass);
        const uint32_t output = current_positions[digit]++;
        copy_record(destination + output, source + index);
    }
}

template <uint32_t N, bool CurrentCrosses, bool NextCrosses, std::size_t PrefetchDistance>
inline void scatter_and_count_next_pass(
    const ExportRecord<N>* TREE_RADIX_RESTRICT source,
    ExportRecord<N>* TREE_RADIX_RESTRICT destination,
    std::size_t record_count,
    const PassDescriptor& current_pass,
    const PassDescriptor& next_pass,
    uint32_t* current_positions,
    uint32_t* next_histogram) noexcept
{
    std::size_t index = 0;
    for (; index + 4U <= record_count; index += 4U)
    {
        prefetch_destination<PrefetchDistance, CurrentCrosses>(
            source, destination, record_count, index, current_pass, current_positions);

        const uint32_t current0 = extract_digit<CurrentCrosses>(source[index], current_pass);
        const uint32_t current1 = extract_digit<CurrentCrosses>(source[index + 1U], current_pass);
        const uint32_t current2 = extract_digit<CurrentCrosses>(source[index + 2U], current_pass);
        const uint32_t current3 = extract_digit<CurrentCrosses>(source[index + 3U], current_pass);

        const uint32_t next0 = extract_digit<NextCrosses>(source[index], next_pass);
        const uint32_t next1 = extract_digit<NextCrosses>(source[index + 1U], next_pass);
        const uint32_t next2 = extract_digit<NextCrosses>(source[index + 2U], next_pass);
        const uint32_t next3 = extract_digit<NextCrosses>(source[index + 3U], next_pass);

        const uint32_t output0 = current_positions[current0]++;
        const uint32_t output1 = current_positions[current1]++;
        const uint32_t output2 = current_positions[current2]++;
        const uint32_t output3 = current_positions[current3]++;

        copy_record(destination + output0, source + index);
        copy_record(destination + output1, source + index + 1U);
        copy_record(destination + output2, source + index + 2U);
        copy_record(destination + output3, source + index + 3U);

        ++next_histogram[next0];
        ++next_histogram[next1];
        ++next_histogram[next2];
        ++next_histogram[next3];
    }

    for (; index < record_count; ++index)
    {
        prefetch_destination<PrefetchDistance, CurrentCrosses>(
            source, destination, record_count, index, current_pass, current_positions);
        const uint32_t current_digit = extract_digit<CurrentCrosses>(source[index], current_pass);
        const uint32_t next_digit = extract_digit<NextCrosses>(source[index], next_pass);
        const uint32_t output = current_positions[current_digit]++;
        copy_record(destination + output, source + index);
        ++next_histogram[next_digit];
    }
}

template <uint32_t N, std::size_t PrefetchDistance>
inline void dispatch_final_scatter(
    const ExportRecord<N>* source,
    ExportRecord<N>* destination,
    std::size_t record_count,
    const PassDescriptor& current_pass,
    uint32_t* current_positions) noexcept
{
    if (current_pass.crosses_word)
    {
        scatter_final_pass<N, true, PrefetchDistance>(
            source, destination, record_count, current_pass, current_positions);
    }
    else
    {
        scatter_final_pass<N, false, PrefetchDistance>(
            source, destination, record_count, current_pass, current_positions);
    }
}

template <uint32_t N, std::size_t PrefetchDistance>
inline void dispatch_pipeline_scatter(
    const ExportRecord<N>* source,
    ExportRecord<N>* destination,
    std::size_t record_count,
    const PassDescriptor& current_pass,
    const PassDescriptor& next_pass,
    uint32_t* current_positions,
    uint32_t* next_histogram) noexcept
{
    if (current_pass.crosses_word)
    {
        if (next_pass.crosses_word)
        {
            scatter_and_count_next_pass<N, true, true, PrefetchDistance>(
                source, destination, record_count, current_pass, next_pass,
                current_positions, next_histogram);
        }
        else
        {
            scatter_and_count_next_pass<N, true, false, PrefetchDistance>(
                source, destination, record_count, current_pass, next_pass,
                current_positions, next_histogram);
        }
    }
    else if (next_pass.crosses_word)
    {
        scatter_and_count_next_pass<N, false, true, PrefetchDistance>(
            source, destination, record_count, current_pass, next_pass,
            current_positions, next_histogram);
    }
    else
    {
        scatter_and_count_next_pass<N, false, false, PrefetchDistance>(
            source, destination, record_count, current_pass, next_pass,
            current_positions, next_histogram);
    }
}

template <uint32_t N, uint32_t CommonPrefixBases>
inline auto build_passes(uint32_t k_length)
{
    static_assert(N > 0, "kmer<N> requires N > 0");
    static_assert(CommonPrefixBases <= 32, "The common prefix must fit in data[0]");
    constexpr uint32_t max_passes = (64U * N + RADIX_BITS - 1U) / RADIX_BITS;

    struct PassList
    {
        std::array<PassDescriptor, max_passes> descriptors{};
        uint32_t count = 0;
        uint32_t low_padding_bits = 0;
    };

    PassList result;
    const uint32_t total_bits = 64U * N;
    result.low_padding_bits = total_bits - 2U * k_length;
    const uint32_t sort_end = total_bits - 2U * CommonPrefixBases;
    const uint32_t sort_bits = sort_end - result.low_padding_bits;
    result.count = (sort_bits + RADIX_BITS - 1U) / RADIX_BITS;

    for (uint32_t pass_index = 0; pass_index < result.count; ++pass_index)
    {
        const uint32_t bit_offset = result.low_padding_bits + pass_index * RADIX_BITS;
        const uint32_t width = std::min<uint32_t>(RADIX_BITS, sort_end - bit_offset);
        PassDescriptor& pass = result.descriptors[pass_index];
        pass.word_index = N - 1U - bit_offset / 64U;
        pass.shift = bit_offset % 64U;
        pass.width = width;
        pass.bucket_count = 1U << width;
        pass.mask = (uint64_t{ 1 } << width) - 1U;
        pass.crosses_word = pass.shift + width > 64U;
        assert(!pass.crosses_word || pass.word_index > 0);
    }
    return result;
}

template <uint32_t N, uint32_t CommonPrefixBases>
inline void validate_arguments(
    const std::vector<ExportRecord<N>>& records,
    const std::vector<ExportRecord<N>>& scratch,
    uint32_t k_length)
{
    static_assert(std::is_trivially_copyable_v<ExportRecord<N>>,
        "ExportRecord<N> must be trivially copyable");
    static_assert(CommonPrefixBases <= 32, "The common prefix must fit in data[0]");

    if (k_length < CommonPrefixBases || k_length > 32U * N)
    {
        throw std::invalid_argument("k_length is outside the representable/common-prefix range");
    }
    if (std::addressof(records) == std::addressof(scratch))
    {
        throw std::invalid_argument("records and scratch must be different vectors");
    }
    if (records.size() > std::numeric_limits<uint32_t>::max())
    {
        throw std::length_error("12-bit radix sort supports at most UINT32_MAX records");
    }
}

template <uint32_t N, uint32_t CommonPrefixBases, bool Pipeline, std::size_t PrefetchDistance>
inline void radix_sort_impl(
    std::vector<ExportRecord<N>>& records,
    std::vector<ExportRecord<N>>& scratch,
    uint32_t k_length)
{
    validate_arguments<N, CommonPrefixBases>(records, scratch, k_length);
    const auto passes = build_passes<N, CommonPrefixBases>(k_length);
    if (records.size() <= 1U || passes.count == 0U)
    {
#ifndef NDEBUG
        if (!records.empty())
        {
            const uint64_t expected_prefix = common_prefix<N, CommonPrefixBases>(records.front().key);
            for (const auto& record : records)
            {
                assert((common_prefix<N, CommonPrefixBases>(record.key) == expected_prefix));
                assert(padding_is_zero(record.key, passes.low_padding_bits));
            }
        }
#endif
        return;
    }

    scratch.resize(records.size());
    auto* source = records.data();
    auto* destination = scratch.data();
    const std::size_t record_count = records.size();

    alignas(64) std::array<uint32_t, RADIX_SIZE> histogram_a{};
    using SecondaryHistogram = std::conditional_t<
        Pipeline,
        std::array<uint32_t, RADIX_SIZE>,
        std::array<uint32_t, 1>>;
    alignas(64) SecondaryHistogram histogram_b{};

    uint32_t* current_histogram = histogram_a.data();
    uint32_t* next_histogram = histogram_b.data();

    const uint64_t expected_prefix = common_prefix<N, CommonPrefixBases>(records.front().key);
    const PassDescriptor& first_pass = passes.descriptors[0];
    clear_histogram(current_histogram, first_pass.bucket_count);
    if (first_pass.crosses_word)
    {
        count_pass<N, CommonPrefixBases, true>(
            source, record_count, first_pass, current_histogram,
            passes.low_padding_bits, expected_prefix, true);
    }
    else
    {
        count_pass<N, CommonPrefixBases, false>(
            source, record_count, first_pass, current_histogram,
            passes.low_padding_bits, expected_prefix, true);
    }

    for (uint32_t pass_index = 0; pass_index < passes.count; ++pass_index)
    {
        const PassDescriptor& current_pass = passes.descriptors[pass_index];

        if constexpr (!Pipeline)
        {
            if (pass_index > 0)
            {
                clear_histogram(current_histogram, current_pass.bucket_count);
                if (current_pass.crosses_word)
                {
                    count_pass<N, CommonPrefixBases, true>(
                        source, record_count, current_pass, current_histogram,
                        passes.low_padding_bits, expected_prefix, false);
                }
                else
                {
                    count_pass<N, CommonPrefixBases, false>(
                        source, record_count, current_pass, current_histogram,
                        passes.low_padding_bits, expected_prefix, false);
                }
            }
        }

        exclusive_prefix_sum(current_histogram, current_pass.bucket_count);

        if constexpr (Pipeline)
        {
            if (pass_index + 1U < passes.count)
            {
                const PassDescriptor& next_pass = passes.descriptors[pass_index + 1U];
                clear_histogram(next_histogram, next_pass.bucket_count);
                dispatch_pipeline_scatter<N, PrefetchDistance>(
                    source, destination, record_count, current_pass, next_pass,
                    current_histogram, next_histogram);
                std::swap(current_histogram, next_histogram);
            }
            else
            {
                dispatch_final_scatter<N, PrefetchDistance>(
                    source, destination, record_count, current_pass, current_histogram);
            }
        }
        else
        {
            dispatch_final_scatter<N, PrefetchDistance>(
                source, destination, record_count, current_pass, current_histogram);
        }

        std::swap(source, destination);
    }

    if (source == scratch.data())
    {
        records.swap(scratch);
    }
}

template <uint32_t N, uint32_t CommonPrefixBases = 10>
inline void radix_sort_export_records_12bit_per_pass(
    std::vector<ExportRecord<N>>& records,
    std::vector<ExportRecord<N>>& scratch,
    uint32_t k_length)
{
    radix_sort_impl<N, CommonPrefixBases, false, 0>(records, scratch, k_length);
}

template <std::size_t PrefetchDistance, uint32_t N, uint32_t CommonPrefixBases = 10>
inline void radix_sort_export_records_12bit_prefetch(
    std::vector<ExportRecord<N>>& records,
    std::vector<ExportRecord<N>>& scratch,
    uint32_t k_length)
{
    static_assert(PrefetchDistance == 8 || PrefetchDistance == 16,
        "The benchmark prefetch backend supports distance 8 or 16");
    radix_sort_impl<N, CommonPrefixBases, true, PrefetchDistance>(records, scratch, k_length);
}
} // namespace detail

template <uint32_t N, uint32_t CommonPrefixBases = 10>
inline void radix_sort_export_records_12bit(
    std::vector<ExportRecord<N>>& records,
    std::vector<ExportRecord<N>>& scratch,
    uint32_t k_length)
{
    detail::radix_sort_impl<N, CommonPrefixBases, true, 0>(records, scratch, k_length);
}
} // namespace tree::sort

#undef TREE_RADIX_RESTRICT

#endif // TREE_EXPORT_RECORD_RADIX_SORT_12_HEADER
